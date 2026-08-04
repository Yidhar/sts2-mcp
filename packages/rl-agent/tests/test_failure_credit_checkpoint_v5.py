from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from sts2_rl.checkpoints import V28_100K_FROZEN, CheckpointIntegrityError
from sts2_rl.models import RecurrentCandidateModel
from sts2_rl.training import (
    TrainingState,
    build_training_resources,
    initialize_model_from_checkpoint,
    load_training_checkpoint,
    preflight_model_initialization,
    preflight_training_checkpoint,
    save_training_checkpoint,
)
from sts2_rl.training import checkpointing as checkpointing_module
from sts2_rl.training.config import (
    model_initialization_config_from_mapping,
)
from tests.test_v2_training_pipeline import FakeCombatBackend, _config

_LIVENESS_PREFIXES = (
    "liveness_cost_value_head.",
    "candidate_liveness_cost_head.",
)
_TRANSACTION_PREFIXES = (
    "candidate_effect_head.",
    "selection_delta_head.",
    "transaction_q_head.",
)


def _learning_config():
    base = _config()
    return replace(
        base,
        failure_credit=replace(base.failure_credit, mode="learning"),
    )


def _update_manifest_entry(checkpoint: Path, relative_path: str) -> None:
    manifest_path = checkpoint / "checkpoint.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload = (checkpoint / relative_path).read_bytes()
    for entry in manifest["files"]:
        if entry["path"] == relative_path:
            entry["size_bytes"] = len(payload)
            entry["sha256"] = hashlib.sha256(payload).hexdigest()
            break
    else:  # pragma: no cover - publisher always lists metadata
        raise AssertionError(f"manifest entry missing: {relative_path}")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _rewrite_as_v4_model_initialization_source(checkpoint: Path) -> None:
    metadata_path = checkpoint / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["format"] = "sts2-recurrent-vtrace-checkpoint-v4"
    for key in (
        "decision_semantics_abi",
        "failure_credit_abi",
        "failure_credit_mode",
        "liveness_cost_heads_enabled",
        "failure_credit_replay_enabled",
        "failure_credit_replay_spec",
    ):
        metadata.pop(key, None)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _update_manifest_entry(checkpoint, "metadata.json")


def _liveness_state(
    state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {key: value for key, value in state.items() if key.startswith(_LIVENESS_PREFIXES)}


def _transaction_state(
    state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {key: value for key, value in state.items() if key.startswith(_TRANSACTION_PREFIXES)}


def test_v4_policy_is_model_initialization_only_and_freshens_liveness_group(
    tmp_path: Path,
) -> None:
    source_base = _config()
    source_config = replace(
        source_base,
        transaction_learning=replace(
            source_base.transaction_learning,
            enabled=True,
        ),
    )
    source = build_training_resources(
        source_config,
        backend=FakeCombatBackend(),
    )
    try:
        parity_snapshot = source.collector.collect_episode(record=True).unrolls[0].steps[0].snapshot
        parity_batch = source.encoder.collate_snapshots((parity_snapshot,))
        source.model.eval()
        with torch.no_grad():
            source_step_zero = source.model(parity_batch)
        source_state = {key: value.detach().clone() for key, value in source.model.state_dict().items()}
        checkpoint = save_training_checkpoint(
            tmp_path / "synthetic-v28-policy",
            config=source_config,
            resources=source,
            state=TrainingState(
                environment_steps=100_000,
                learner_updates=1_569,
                policy_version=1_569,
                actor_policy_version=1_569,
            ),
            checkpoint_load_mode="fresh",
        )
    finally:
        source.close()
    _rewrite_as_v4_model_initialization_source(checkpoint)

    with pytest.raises(
        ValueError,
        match="unsupported exact-resume checkpoint format",
    ):
        preflight_training_checkpoint(
            checkpoint,
            config=source_config,
            resolved_device="cpu",
            resolved_collector_device="cpu",
        )

    target_config = _learning_config()
    assert (
        preflight_model_initialization(
            checkpoint,
            config=target_config,
        ).root
        == checkpoint.resolve()
    )
    target = build_training_resources(
        target_config,
        backend=FakeCombatBackend(),
    )
    try:
        target_before = {key: value.detach().clone() for key, value in target.model.state_dict().items()}
        fresh_liveness = _liveness_state(target_before)
        assert fresh_liveness
        assert not _liveness_state(source_state)
        dropped_transaction = _transaction_state(source_state)
        assert dropped_transaction
        assert not _transaction_state(target_before)
        assert target.failure_credit_replay is not None
        python_rng = random.getstate()
        numpy_rng = np.random.get_state()
        torch_rng = torch.get_rng_state().clone()
        collector_state = target.collector.state_dict()
        replay_state = target.failure_credit_replay.state_dict()

        parent = initialize_model_from_checkpoint(
            checkpoint,
            config=target_config,
            resources=target,
        )

        assert parent == checkpoint.resolve()
        target_after = target.model.state_dict()
        for key, expected in source_state.items():
            if key in dropped_transaction:
                continue
            assert torch.equal(target_after[key], expected), key
            assert torch.equal(
                target.collector_model.state_dict()[key],
                expected,
            ), key
        for key, expected in fresh_liveness.items():
            assert torch.equal(target_after[key], expected), key
            assert torch.equal(
                target.collector_model.state_dict()[key],
                expected,
            ), key
        target.model.eval()
        with torch.no_grad():
            target_step_zero = target.model(target.encoder.collate_snapshots((parity_snapshot,)))
        # Fresh auxiliary heads add outputs but cannot perturb any inherited
        # policy/value/recurrent tensor at model-init step zero.
        torch.testing.assert_close(
            target_step_zero.policy_logits,
            source_step_zero.policy_logits,
            atol=0.0,
            rtol=0.0,
        )
        torch.testing.assert_close(
            target_step_zero.value,
            source_step_zero.value,
            atol=0.0,
            rtol=0.0,
        )
        torch.testing.assert_close(
            target_step_zero.recurrent_state,
            source_step_zero.recurrent_state,
            atol=0.0,
            rtol=0.0,
        )
        assert len(target.optimizer.state) == 0
        assert len(target.rollout_queue) == 0
        assert target.failure_credit_replay.metrics()["size"] == 0
        assert target.failure_credit_replay.metrics()["put_count"] == 0
        assert target.failure_credit_replay.state_dict() == replay_state
        assert target.collector.state_dict() == collector_state
        assert random.getstate() == python_rng
        restored_numpy_rng = np.random.get_state()
        assert restored_numpy_rng[0] == numpy_rng[0]
        assert np.array_equal(restored_numpy_rng[1], numpy_rng[1])
        assert restored_numpy_rng[2:] == numpy_rng[2:]
        assert torch.equal(torch.get_rng_state(), torch_rng)
    finally:
        target.close()


def test_liveness_head_model_initialization_is_all_or_none() -> None:
    config = _learning_config()
    target_model = RecurrentCandidateModel(
        config.model.to_model_config(),
        enable_liveness_head=True,
    )
    target = target_model.state_dict()
    liveness = _liveness_state(target)
    assert liveness

    complete_source = {key: value.detach().clone() for key, value in target.items()}
    for key in liveness:
        complete_source[key].fill_(0.125)
    inherited = checkpointing_module._model_parameter_initialization_state(
        complete_source,
        target_state=target,
        allow_missing_transaction_heads=False,
        allow_missing_liveness_heads=True,
    )
    for key, expected in complete_source.items():
        assert torch.equal(inherited[key], expected), key

    absent = {key: value for key, value in target.items() if key not in liveness}
    migrated = checkpointing_module._model_parameter_initialization_state(
        absent,
        target_state=target,
        allow_missing_transaction_heads=False,
        allow_missing_liveness_heads=True,
    )
    for key, expected in liveness.items():
        assert torch.equal(migrated[key], expected), key

    partial = dict(target)
    partial.pop(next(iter(liveness)))
    with pytest.raises(
        ValueError,
        match="all or none.*liveness-head",
    ):
        checkpointing_module._model_parameter_initialization_state(
            partial,
            target_state=target,
            allow_missing_transaction_heads=False,
            allow_missing_liveness_heads=True,
        )

    one_family_missing = {
        key: value for key, value in target.items() if not key.startswith("candidate_liveness_cost_head.")
    }
    with pytest.raises(
        ValueError,
        match="all or none.*liveness-head",
    ):
        checkpointing_module._model_parameter_initialization_state(
            one_family_missing,
            target_state=target,
            allow_missing_transaction_heads=False,
            allow_missing_liveness_heads=True,
        )


def test_retired_transaction_v3_heads_drop_only_as_one_complete_source_group() -> None:
    config = _learning_config()
    source = RecurrentCandidateModel(
        config.model.to_model_config(),
        enable_transaction_heads=True,
    ).state_dict()
    target = RecurrentCandidateModel(
        config.model.to_model_config(),
        enable_liveness_head=True,
    ).state_dict()
    transaction = _transaction_state(source)
    assert transaction
    assert not _transaction_state(target)

    migrated = checkpointing_module._model_parameter_initialization_state(
        source,
        target_state=target,
        allow_missing_transaction_heads=False,
        allow_source_transaction_head_drop=True,
        allow_missing_liveness_heads=True,
    )
    assert set(migrated) == set(target)
    for key, expected in source.items():
        if key not in transaction:
            assert torch.equal(migrated[key], expected), key

    partial = dict(source)
    partial.pop(next(iter(transaction)))
    with pytest.raises(
        ValueError,
        match="all or none.*source-only transaction-head",
    ):
        checkpointing_module._model_parameter_initialization_state(
            partial,
            target_state=target,
            allow_missing_transaction_heads=False,
            allow_source_transaction_head_drop=True,
            allow_missing_liveness_heads=True,
        )

    with pytest.raises(
        ValueError,
        match="all or none.*source-only transaction-head",
    ):
        checkpointing_module._model_parameter_initialization_state(
            source,
            target_state=target,
            allow_missing_transaction_heads=False,
            allow_source_transaction_head_drop=False,
            allow_missing_liveness_heads=True,
        )


def test_v5_failure_credit_checkpoint_roundtrip_restores_replay_and_heads(
    tmp_path: Path,
) -> None:
    config = _learning_config()
    source = build_training_resources(config, backend=FakeCombatBackend())
    try:
        assert source.failure_credit_replay is not None
        source.optimizer.zero_grad(set_to_none=True)
        objective = sum(parameter.square().mean() for parameter in source.model.parameters())
        objective.backward()
        source.optimizer.step()
        expected_model = {key: value.detach().clone() for key, value in source.model.state_dict().items()}
        expected_replay = source.failure_credit_replay.state_dict()
        state = TrainingState(
            environment_steps=23,
            learner_updates=2,
            episodes=1,
            policy_version=2,
            actor_policy_version=2,
        )
        checkpoint = save_training_checkpoint(
            tmp_path / "failure-credit-v5",
            config=config,
            resources=source,
            state=state,
            checkpoint_load_mode="fresh",
        )
    finally:
        source.close()

    metadata = json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["format"] == "sts2-recurrent-vtrace-checkpoint-v5"
    assert metadata["decision_semantics_abi"] == (checkpointing_module._decision_semantics_abi())
    assert metadata["failure_credit_abi"] == (checkpointing_module._failure_credit_abi())
    assert metadata["failure_credit_abi"]["collector"] == checkpointing_module.FAILURE_CREDIT_COLLECTOR_VERSION
    assert metadata["failure_credit_abi"]["detector"] == checkpointing_module.FAILURE_CREDIT_DETECTOR_VERSION
    assert metadata["failure_credit_mode"] == "learning"
    assert metadata["liveness_cost_heads_enabled"] is True
    assert metadata["failure_credit_replay_enabled"] is True
    assert isinstance(metadata["failure_credit_replay_spec"], dict)
    assert (checkpoint / "failure_credit_replay.pkl").is_file()
    assert _liveness_state(expected_model)

    restored = build_training_resources(
        config,
        backend=FakeCombatBackend(),
    )
    try:
        loaded = load_training_checkpoint(
            checkpoint,
            config=config,
            resources=restored,
        )
        assert loaded == state
        assert restored.failure_credit_replay is not None
        assert restored.failure_credit_replay.state_dict() == expected_replay
        for key, expected in expected_model.items():
            assert torch.equal(
                restored.model.state_dict()[key],
                expected,
            ), key
    finally:
        restored.close()

    # The replay contains detector-produced evidence.  A process with a
    # different detector or collector contract must not append differently
    # interpreted records to the same exact-resume lineage.
    metadata["failure_credit_abi"]["detector"] = "mutated-detector-contract"
    (checkpoint / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _update_manifest_entry(checkpoint, "metadata.json")
    with pytest.raises(ValueError, match="failure-credit ABI"):
        preflight_training_checkpoint(
            checkpoint,
            config=config,
            resolved_device="cpu",
            resolved_collector_device="cpu",
        )


def test_frozen_v28_artifact_is_rejected_for_exact_resume_but_preflights_init() -> None:
    roots: list[Path] = []
    configured = os.environ.get("STS2_ARTIFACT_ROOT")
    if configured:
        roots.append(Path(configured))
    roots.extend(
        (
            Path("/mnt/e/game/project/sts2_mcp_artifacts/runtime"),
            Path(r"E:\game\project\sts2_mcp_artifacts\runtime"),
        )
    )
    artifact_root = next(
        (root for root in roots if V28_100K_FROZEN.resolve(root).is_dir()),
        None,
    )
    if artifact_root is None:
        pytest.skip("the pinned local v28 artifact is not available")
    checkpoint = V28_100K_FROZEN.resolve(artifact_root)
    metadata = json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))
    source_config = model_initialization_config_from_mapping(metadata["training_config"])
    target_config = replace(
        source_config,
        transaction_learning=replace(
            source_config.transaction_learning,
            enabled=False,
        ),
        failure_credit=replace(
            source_config.failure_credit,
            mode="learning",
        ),
    )

    # Reward-v5 now rejects this reward-v4 ancestor at the outer exact-resume
    # identity boundary. On a runtime with the historical catalog the frozen
    # ancestor guard remains the later failure. Both are exact-resume rejection;
    # model-only preflight below is the reviewed migration path.
    with pytest.raises(
        (ValueError, CheckpointIntegrityError),
        match="reward identity mismatch|model-initialization-only; exact resume is forbidden",
    ):
        preflight_training_checkpoint(
            checkpoint,
            config=target_config,
            resolved_device=str(metadata["resolved_device"]),
            resolved_collector_device=str(metadata["resolved_collector_device"]),
        )
    assert (
        preflight_model_initialization(
            checkpoint,
            config=target_config,
        ).root
        == checkpoint.resolve()
    )
