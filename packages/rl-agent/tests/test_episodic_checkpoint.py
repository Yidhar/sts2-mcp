from __future__ import annotations

import hashlib
import json
import pickle
import random
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from sts2_rl.checkpoints import CheckpointIntegrityError, ValidatedResumeCheckpoint
from sts2_rl.encoding import grounding_encoding_identity
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
from sts2_rl.training.transaction import (
    TRANSACTION_LIFECYCLE_VERSION,
    TransactionEffect,
    TransactionStep,
    TransactionTrace,
)
from tests.test_episode_replay import _linear_episode, _snapshot
from tests.test_rollout_queue import _unroll
from tests.test_v2_training_pipeline import FakeCombatBackend, _config

_LONG_HEAD_PREFIXES = (
    "combat_task_value_head.",
    "act_task_value_head.",
    "run_task_value_head.",
    "combat_revival_cost_value_head.",
    "act_revival_cost_value_head.",
    "run_revival_cost_value_head.",
)


def _episodic_config():
    base = _config()
    return replace(
        base,
        episodic_learning=replace(
            base.episodic_learning,
            enabled=True,
            replay_capacity_episodes=8,
            replay_capacity_bytes=20_000_000,
            per_episode_capacity_bytes=10_000_000,
            max_segments_per_episode=3,
            sample_sequences=2,
            burn_in_steps=2,
            learn_steps=2,
        ),
        environment=replace(base.environment, scenario="full-run"),
        curriculum=replace(base.curriculum, reward_objective="run"),
        runtime=replace(
            base.runtime,
            device="cpu",
            collector_device="cpu",
        ),
    )


def _transaction_config():
    base = _config()
    return replace(
        base,
        transaction_learning=replace(
            base.transaction_learning,
            enabled=True,
            replay_capacity=8,
            replay_byte_capacity=20_000_000,
            sample_traces=2,
            burn_in_steps=0,
        ),
        runtime=replace(
            base.runtime,
            device="cpu",
            collector_device="cpu",
        ),
    )


def _sequence_identity(sequences: tuple[Any, ...]) -> tuple[tuple[Any, ...], ...]:
    return tuple(
        (
            item.episode_id,
            item.start_step,
            item.burn_in_steps,
            tuple(step.decision.step_index for step in item.steps),
        )
        for item in sequences
    )


def _active_snapshot(config: Any):
    return replace(
        _snapshot(),
        config=config.model.to_encoding_config(),
        encoding_fingerprint=grounding_encoding_identity()["fingerprint_sha256"],
    )


def _transaction_trace(config: Any, *, trace_id: str) -> TransactionTrace:
    return TransactionTrace(
        trace_id=trace_id,
        episode_id=f"episode-{trace_id}",
        surface_key="checkpoint-selection-surface",
        start_step=0,
        policy_version=3,
        initial_recurrent_state=np.zeros(
            config.model.recurrent_hidden_dim,
            dtype=np.float32,
        ),
        steps=(
            TransactionStep(
                snapshot=_active_snapshot(config),
                action_index=0,
                node_key="checkpoint-selection-node",
                next_node_key="checkpoint-selection-exit",
                action_fingerprint="select:first-candidate",
                effect=TransactionEffect.EXIT,
                selected_count_delta=1,
                transaction_return=1.0,
                return_steps=1,
            ),
        ),
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
    else:  # pragma: no cover - test helper guard
        raise AssertionError(f"manifest entry missing: {relative_path}")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _rewrite_as_v3_without_long_heads(checkpoint: Path) -> dict[str, torch.Tensor]:
    network_path = checkpoint / "network.pt"
    state = torch.load(network_path, map_location="cpu", weights_only=True)
    assert isinstance(state, dict)
    legacy = {key: value for key, value in state.items() if not key.startswith(_LONG_HEAD_PREFIXES)}
    torch.save(legacy, network_path)

    metadata_path = checkpoint / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["format"] = "sts2-recurrent-vtrace-checkpoint-v3"
    metadata["model_state_spec"] = checkpointing_module._tensor_spec(legacy)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _update_manifest_entry(checkpoint, "network.pt")
    _update_manifest_entry(checkpoint, "metadata.json")
    return legacy


def _assert_numpy_rng_equal(left: tuple[Any, ...], right: tuple[Any, ...]) -> None:
    assert left[0] == right[0]
    assert np.array_equal(left[1], right[1])
    assert left[2:] == right[2:]


def test_transaction_lifecycle_abi_is_persisted_and_required_for_exact_resume(
    tmp_path: Path,
) -> None:
    config = _transaction_config()
    source = build_training_resources(config, backend=FakeCombatBackend())
    try:
        checkpoint = save_training_checkpoint(
            tmp_path / "transaction-lifecycle-abi",
            config=config,
            resources=source,
            state=TrainingState(environment_steps=11),
            checkpoint_load_mode="fresh",
        )
    finally:
        source.close()

    metadata_path = checkpoint / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["transaction_lifecycle_abi"] == TRANSACTION_LIFECYCLE_VERSION
    metadata.pop("transaction_lifecycle_abi")
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _update_manifest_entry(checkpoint, "metadata.json")

    with pytest.raises(ValueError, match=r"lifecycle-evidence ABI"):
        preflight_training_checkpoint(
            checkpoint,
            config=config,
            resolved_device="cpu",
            resolved_collector_device="cpu",
        )


def _assert_models_equal(
    actual: dict[str, torch.Tensor],
    expected: dict[str, torch.Tensor],
) -> None:
    assert set(actual) == set(expected)
    for key, value in expected.items():
        assert torch.equal(actual[key], value), key


def _model_copy(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def test_episodic_replay_checkpoint_roundtrip_restores_order_bytes_and_rng(
    tmp_path: Path,
) -> None:
    config = _episodic_config()
    snapshot = _active_snapshot(config)
    episodes = (
        _linear_episode(snapshot, episode_id="checkpoint-failure", length=5, won=False),
        _linear_episode(snapshot, episode_id="checkpoint-success", length=7, won=True),
    )
    source = build_training_resources(config, backend=FakeCombatBackend())
    try:
        assert source.episodic_replay is not None
        for episode in episodes:
            assert source.episodic_replay.put(episode)
        # Advance the replay-owned generator before publication so restoration
        # must recover a non-initial RNG state.
        source.episodic_replay.sample(1, learn_steps=2, burn_in_steps=1)
        saved_metrics = dict(source.episodic_replay.metrics())
        saved_rng = source.episodic_replay.state_dict()["rng_state"]
        state = TrainingState(environment_steps=37, learner_updates=4, episodes=2)
        checkpoint = save_training_checkpoint(
            tmp_path / "episodic-roundtrip",
            config=config,
            resources=source,
            state=state,
            checkpoint_load_mode="fresh",
        )
        expected_next = _sequence_identity(source.episodic_replay.sample(5, learn_steps=2, burn_in_steps=1))
    finally:
        source.close()

    metadata = json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["format"] == "sts2-recurrent-vtrace-checkpoint-v5"
    assert metadata["episodic_target_abi"] == ("sts2-episodic-task-targets-one-terminal-unit-v2")
    assert metadata["episodic_replay_enabled"] is True
    assert metadata["episodic_replay_spec"] == saved_metrics
    manifest = json.loads((checkpoint / "checkpoint.manifest.json").read_text(encoding="utf-8"))
    sidecar_entries = [entry for entry in manifest["files"] if entry["path"] == "episodic_replay.pkl"]
    assert len(sidecar_entries) == 1
    assert len(sidecar_entries[0]["sha256"]) == 64

    restored = build_training_resources(config, backend=FakeCombatBackend())
    try:
        loaded = load_training_checkpoint(checkpoint, config=config, resources=restored)
        assert loaded == state
        assert restored.episodic_replay is not None
        assert restored.episodic_replay.metrics() == saved_metrics
        assert restored.episodic_replay.state_dict()["rng_state"] == saved_rng
        assert [item.episode_id for item in restored.episodic_replay.snapshot()] == [
            item.episode_id for item in episodes
        ]
        assert _sequence_identity(restored.episodic_replay.sample(5, learn_steps=2, burn_in_steps=1)) == expected_next
    finally:
        restored.close()


def test_checkpoint_roundtrip_restores_v33_learner_dynamics_and_health_role(
    tmp_path: Path,
) -> None:
    config = _episodic_config()
    source = build_training_resources(config, backend=FakeCombatBackend())
    expected_dynamics = {
        "version": "sts2-vtrace-learner-dynamics-v2",
        "collapse_batch_streak": 3,
        "entropy_breaker_remaining_updates": 5,
        "entropy_breaker_triggers": 2,
    }
    state = TrainingState(
        environment_steps=41,
        learner_updates=7,
        episodes=3,
        policy_version=7,
        actor_policy_version=6,
        training_deadlock_streak=2,
        training_deadlock_alerts=1,
        evaluation_guard_rollbacks=1,
    )
    try:
        source.learner.load_dynamics_state_dict(expected_dynamics)
        checkpoint = save_training_checkpoint(
            tmp_path / "v33-dynamics-health-role",
            config=config,
            resources=source,
            state=state,
            checkpoint_load_mode="fresh",
            checkpoint_role="healthy_evaluation_anchor",
        )
    finally:
        source.close()

    validated = preflight_training_checkpoint(
        checkpoint,
        config=config,
        resolved_device="cpu",
        resolved_collector_device="cpu",
    )
    assert validated.metadata["checkpoint_role"] == "healthy_evaluation_anchor"

    restored = build_training_resources(config, backend=FakeCombatBackend())
    try:
        loaded = load_training_checkpoint(
            checkpoint,
            config=config,
            resources=restored,
        )
        assert loaded == state
        assert restored.learner.dynamics_state_dict() == expected_dynamics
    finally:
        restored.close()


def test_episodic_target_abi_is_exact_resume_only(
    tmp_path: Path,
) -> None:
    config = _episodic_config()
    source = build_training_resources(config, backend=FakeCombatBackend())
    try:
        checkpoint = save_training_checkpoint(
            tmp_path / "episodic-target-abi",
            config=config,
            resources=source,
            state=TrainingState(environment_steps=13),
            checkpoint_load_mode="fresh",
        )
    finally:
        source.close()

    metadata_path = checkpoint / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.pop("episodic_target_abi")
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _update_manifest_entry(checkpoint, "metadata.json")

    with pytest.raises(
        ValueError,
        match=r"exact-resume checkpoint has no episodic target ABI marker",
    ):
        preflight_training_checkpoint(
            checkpoint,
            config=config,
            resolved_device="cpu",
            resolved_collector_device="cpu",
        )

    # Parameter initialization deliberately imports only compatible learned
    # tensors. It must not pretend to resume the old optimizer/replay objective,
    # but it may use the source network as a fresh lineage initializer.
    assert preflight_model_initialization(checkpoint, config=config).root == checkpoint.resolve()


def test_missing_episodic_sidecar_fails_before_live_resource_mutation(
    tmp_path: Path,
) -> None:
    config = _episodic_config()
    source = build_training_resources(config, backend=FakeCombatBackend())
    try:
        checkpoint = save_training_checkpoint(
            tmp_path / "missing-sidecar",
            config=config,
            resources=source,
            state=TrainingState(environment_steps=11),
            checkpoint_load_mode="fresh",
        )
    finally:
        source.close()
    (checkpoint / "episodic_replay.pkl").unlink()

    target = build_training_resources(config, backend=FakeCombatBackend())
    try:
        assert target.episodic_replay is not None
        model_before = {key: value.detach().clone() for key, value in target.model.state_dict().items()}
        replay_before = target.episodic_replay.state_dict()
        collector_before = target.collector.state_dict()
        python_before = random.getstate()
        numpy_before = np.random.get_state()
        torch_before = torch.get_rng_state().clone()

        with pytest.raises(CheckpointIntegrityError):
            load_training_checkpoint(checkpoint, config=config, resources=target)

        for key, expected in model_before.items():
            assert torch.equal(target.model.state_dict()[key], expected), key
        assert target.episodic_replay.state_dict() == replay_before
        assert target.collector.state_dict() == collector_before
        assert random.getstate() == python_before
        _assert_numpy_rng_equal(np.random.get_state(), numpy_before)
        assert torch.equal(torch.get_rng_state(), torch_before)
        assert len(target.optimizer.state) == 0
        assert len(target.rollout_queue) == 0
    finally:
        target.close()


def test_invalid_hashed_episodic_payload_is_probed_before_any_live_mutation(
    tmp_path: Path,
) -> None:
    config = _episodic_config()
    source = build_training_resources(config, backend=FakeCombatBackend())
    try:
        checkpoint = save_training_checkpoint(
            tmp_path / "invalid-sidecar",
            config=config,
            resources=source,
            state=TrainingState(environment_steps=13),
            checkpoint_load_mode="fresh",
        )
    finally:
        source.close()

    sidecar = checkpoint / "episodic_replay.pkl"
    with sidecar.open("rb") as handle:
        payload = pickle.load(handle)
    payload["version"] = "unsupported-episodic-replay"
    with sidecar.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # Keep the outer atomic checkpoint internally hash-consistent. The replay
    # probe, rather than the manifest verifier, must reject this payload.
    _update_manifest_entry(checkpoint, "episodic_replay.pkl")

    target = build_training_resources(config, backend=FakeCombatBackend())
    try:
        assert target.episodic_replay is not None
        model_before = {key: value.detach().clone() for key, value in target.model.state_dict().items()}
        replay_before = target.episodic_replay.state_dict()
        collector_before = target.collector.state_dict()
        torch_before = torch.get_rng_state().clone()

        with pytest.raises(ValueError, match=r"episodic replay"):
            load_training_checkpoint(checkpoint, config=config, resources=target)

        for key, expected in model_before.items():
            assert torch.equal(target.model.state_dict()[key], expected), key
        assert target.episodic_replay.state_dict() == replay_before
        assert target.collector.state_dict() == collector_before
        assert torch.equal(torch.get_rng_state(), torch_before)
        assert len(target.optimizer.state) == 0
        assert len(target.rollout_queue) == 0
    finally:
        target.close()


@pytest.mark.parametrize(
    ("corruption", "message"),
    (
        ("fingerprint", "fingerprint differs"),
        ("config", "config differs"),
        ("candidate_shape", "action mask length"),
    ),
)
def test_hash_consistent_snapshot_abi_corruption_fails_before_live_mutation(
    tmp_path: Path,
    corruption: str,
    message: str,
) -> None:
    config = _episodic_config()
    source = build_training_resources(config, backend=FakeCombatBackend())
    try:
        assert source.episodic_replay is not None
        episode = _linear_episode(
            _active_snapshot(config),
            episode_id=f"corrupt-{corruption}",
            length=2,
            won=True,
        )
        assert source.episodic_replay.put(episode)
        checkpoint = save_training_checkpoint(
            tmp_path / f"snapshot-{corruption}",
            config=config,
            resources=source,
            state=TrainingState(environment_steps=17),
            checkpoint_load_mode="fresh",
        )
    finally:
        source.close()

    sidecar = checkpoint / "episodic_replay.pkl"
    with sidecar.open("rb") as handle:
        payload = pickle.load(handle)
    snapshot = payload["items"][0].steps[0].snapshot
    if corruption == "fingerprint":
        object.__setattr__(snapshot, "encoding_fingerprint", "0" * 64)
    elif corruption == "config":
        object.__setattr__(
            snapshot,
            "config",
            replace(snapshot.config, max_candidates=snapshot.config.max_candidates + 1),
        )
    elif corruption == "candidate_shape":
        edited_mask = np.asarray([True], dtype=np.bool_)
        edited_mask.setflags(write=False)
        object.__setattr__(snapshot, "action_mask", edited_mask)
    else:  # pragma: no cover - parametrization guard
        raise AssertionError(corruption)
    with sidecar.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    _update_manifest_entry(checkpoint, "episodic_replay.pkl")

    target = build_training_resources(config, backend=FakeCombatBackend())
    try:
        assert target.episodic_replay is not None
        model_before = _model_copy(target.model)
        actor_before = _model_copy(target.collector_model)
        replay_before = target.episodic_replay.state_dict()
        collector_before = target.collector.state_dict()
        torch_before = torch.get_rng_state().clone()

        with pytest.raises(ValueError, match=message):
            load_training_checkpoint(checkpoint, config=config, resources=target)

        _assert_models_equal(target.model.state_dict(), model_before)
        _assert_models_equal(target.collector_model.state_dict(), actor_before)
        assert target.episodic_replay.state_dict() == replay_before
        assert target.collector.state_dict() == collector_before
        assert torch.equal(torch.get_rng_state(), torch_before)
        assert len(target.optimizer.state) == 0
        assert len(target.rollout_queue) == 0
    finally:
        target.close()


def test_transaction_replay_restore_reowns_and_freezes_legacy_snapshot_arrays(
    tmp_path: Path,
) -> None:
    config = _transaction_config()
    source = build_training_resources(config, backend=FakeCombatBackend())
    try:
        assert source.transaction_replay is not None
        assert source.transaction_replay.put(_transaction_trace(config, trace_id="legacy-arrays"))
        checkpoint = save_training_checkpoint(
            tmp_path / "transaction-legacy-arrays",
            config=config,
            resources=source,
            state=TrainingState(environment_steps=17),
            checkpoint_load_mode="fresh",
        )
    finally:
        source.close()

    sidecar = checkpoint / "transaction_replay.pkl"
    with sidecar.open("rb") as handle:
        payload = pickle.load(handle)
    # Protocol 4 reproduces legacy NumPy restoration where read-only flags are
    # lost. The semantic payload and manifest remain internally consistent.
    payload = pickle.loads(pickle.dumps(payload, protocol=4))
    raw_snapshot = payload["items"][0].steps[0].snapshot
    assert raw_snapshot.action_mask.flags.writeable
    with sidecar.open("wb") as handle:
        pickle.dump(payload, handle, protocol=4)
    _update_manifest_entry(checkpoint, "transaction_replay.pkl")

    target = build_training_resources(config, backend=FakeCombatBackend())
    try:
        load_training_checkpoint(checkpoint, config=config, resources=target)
        assert target.transaction_replay is not None
        restored_snapshot = target.transaction_replay.snapshot()[0].steps[0].snapshot
        arrays = (
            restored_snapshot.world.feature_indptr,
            restored_snapshot.world.feature_indices,
            restored_snapshot.world.feature_values,
            restored_snapshot.world.ids,
            restored_snapshot.candidates.feature_indptr,
            restored_snapshot.candidates.feature_indices,
            restored_snapshot.candidates.feature_values,
            restored_snapshot.candidates.ids,
            restored_snapshot.locals.feature_indptr,
            restored_snapshot.locals.feature_indices,
            restored_snapshot.locals.feature_values,
            restored_snapshot.locals.ids,
            restored_snapshot.local_offsets,
            restored_snapshot.action_mask,
        )
        assert all(not array.flags.writeable for array in arrays)
        assert all(array.flags.owndata for array in arrays)
    finally:
        target.close()


@pytest.mark.parametrize(
    ("corruption", "message"),
    (
        ("trace_version", "unsupported transaction trace version"),
        ("step_return", "observed transaction return requires positive return_steps"),
        ("policy_node_only", "must be both present or both absent"),
        ("policy_action_only", "must be both present or both absent"),
        ("fingerprint", "fingerprint differs"),
        ("config", "config differs"),
        ("candidate_shape", "action mask length"),
        ("hidden_shape", "recurrent state differs from model hidden size"),
        ("accounting", "accounting is inconsistent"),
    ),
)
def test_hash_consistent_transaction_replay_corruption_fails_before_live_mutation(
    tmp_path: Path,
    corruption: str,
    message: str,
) -> None:
    config = _transaction_config()
    source = build_training_resources(config, backend=FakeCombatBackend())
    try:
        assert source.transaction_replay is not None
        assert source.transaction_replay.put(_transaction_trace(config, trace_id=f"corrupt-{corruption}"))
        checkpoint = save_training_checkpoint(
            tmp_path / f"transaction-{corruption}",
            config=config,
            resources=source,
            state=TrainingState(environment_steps=17),
            checkpoint_load_mode="fresh",
        )
    finally:
        source.close()

    sidecar = checkpoint / "transaction_replay.pkl"
    with sidecar.open("rb") as handle:
        payload = pickle.load(handle)
    trace = payload["items"][0]
    step = trace.steps[0]
    snapshot = step.snapshot
    if corruption == "trace_version":
        object.__setattr__(trace, "version", "corrupt-transaction-trace")
    elif corruption == "step_return":
        object.__setattr__(step, "return_steps", None)
    elif corruption == "policy_node_only":
        object.__setattr__(step, "policy_node_key", "coarse-policy-node")
    elif corruption == "policy_action_only":
        object.__setattr__(
            step,
            "policy_action_fingerprint",
            "coarse-policy-action",
        )
    elif corruption == "fingerprint":
        object.__setattr__(snapshot, "encoding_fingerprint", "0" * 64)
    elif corruption == "config":
        object.__setattr__(
            snapshot,
            "config",
            replace(snapshot.config, max_candidates=snapshot.config.max_candidates + 1),
        )
    elif corruption == "candidate_shape":
        edited_mask = np.asarray([True], dtype=np.bool_)
        edited_mask.setflags(write=False)
        object.__setattr__(snapshot, "action_mask", edited_mask)
    elif corruption == "hidden_shape":
        edited_hidden = np.zeros(
            config.model.recurrent_hidden_dim + 1,
            dtype=np.float32,
        )
        edited_hidden.setflags(write=False)
        object.__setattr__(trace, "initial_recurrent_state", edited_hidden)
    elif corruption == "accounting":
        payload["put_count"] += 1
    else:  # pragma: no cover - parametrization guard
        raise AssertionError(corruption)

    # The nested payload validator is the restore preflight that understands
    # transaction trace v3.  A hash-consistent pickle can bypass the atomic
    # manifest check, but it must not bypass these semantic/shape contracts.
    with pytest.raises(ValueError, match=message):
        checkpointing_module._validated_transaction_replay_payload(
            payload,
            config=config,
        )

    with sidecar.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    _update_manifest_entry(checkpoint, "transaction_replay.pkl")

    target = build_training_resources(config, backend=FakeCombatBackend())
    try:
        assert target.transaction_replay is not None
        queue_before = target.rollout_queue.snapshot()
        closed_before = target.rollout_queue.closed
        model_before = _model_copy(target.model)
        actor_before = _model_copy(target.collector_model)
        replay_before = target.transaction_replay.state_dict()
        collector_before = target.collector.state_dict()
        python_before = random.getstate()
        numpy_before = np.random.get_state()
        torch_before = torch.get_rng_state().clone()

        with pytest.raises(ValueError, match=message):
            load_training_checkpoint(checkpoint, config=config, resources=target)

        _assert_models_equal(target.model.state_dict(), model_before)
        _assert_models_equal(target.collector_model.state_dict(), actor_before)
        assert target.transaction_replay.state_dict() == replay_before
        assert target.collector.state_dict() == collector_before
        assert target.rollout_queue.snapshot() == queue_before
        assert target.rollout_queue.closed is closed_before
        assert random.getstate() == python_before
        _assert_numpy_rng_equal(np.random.get_state(), numpy_before)
        assert torch.equal(torch.get_rng_state(), torch_before)
        assert len(target.optimizer.state) == 0
    finally:
        target.close()


@pytest.mark.parametrize("queue_state", ("nonempty", "closed"))
def test_queue_restore_precondition_fails_before_any_live_resource_mutation(
    tmp_path: Path,
    queue_state: str,
) -> None:
    config = _episodic_config()
    source = build_training_resources(config, backend=FakeCombatBackend())
    try:
        with torch.no_grad():
            next(source.model.parameters()).add_(1.0)
        checkpoint = save_training_checkpoint(
            tmp_path / f"queue-{queue_state}",
            config=config,
            resources=source,
            state=TrainingState(environment_steps=19),
            checkpoint_load_mode="fresh",
        )
    finally:
        source.close()

    target = build_training_resources(config, backend=FakeCombatBackend())
    try:
        assert target.episodic_replay is not None
        if queue_state == "nonempty":
            target.rollout_queue.put(_unroll(91))
        else:
            target.rollout_queue.close()
        queue_before = target.rollout_queue.snapshot()
        closed_before = target.rollout_queue.closed
        model_before = _model_copy(target.model)
        actor_before = _model_copy(target.collector_model)
        replay_before = target.episodic_replay.state_dict()
        collector_before = target.collector.state_dict()
        python_before = random.getstate()
        numpy_before = np.random.get_state()
        torch_before = torch.get_rng_state().clone()

        with pytest.raises(RuntimeError, match=r"rollout queue|closed"):
            load_training_checkpoint(checkpoint, config=config, resources=target)

        _assert_models_equal(target.model.state_dict(), model_before)
        _assert_models_equal(target.collector_model.state_dict(), actor_before)
        assert target.episodic_replay.state_dict() == replay_before
        assert target.collector.state_dict() == collector_before
        assert random.getstate() == python_before
        _assert_numpy_rng_equal(np.random.get_state(), numpy_before)
        assert torch.equal(torch.get_rng_state(), torch_before)
        assert len(target.optimizer.state) == 0
        assert target.rollout_queue.snapshot() == queue_before
        assert target.rollout_queue.closed is closed_before
    finally:
        target.close()


def test_stochastic_state_captures_collector_only_cuda_rng_without_real_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = [torch.tensor([4, 2], dtype=torch.uint8)]
    calls = 0

    def get_rng_state_all() -> list[torch.Tensor]:
        nonlocal calls
        calls += 1
        return expected

    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", get_rng_state_all)
    resources = SimpleNamespace(
        device=torch.device("cpu"),
        collector_model=SimpleNamespace(parameters=lambda: iter((SimpleNamespace(device=torch.device("cuda")),))),
        collector=SimpleNamespace(state_dict=lambda: {"sentinel": True}),
        learner=SimpleNamespace(
            dynamics_state_dict=lambda: {
                "version": "sts2-vtrace-learner-dynamics-v2",
                "collapse_batch_streak": 0,
                "entropy_breaker_remaining_updates": 0,
                "entropy_breaker_triggers": 0,
            }
        ),
    )

    payload = checkpointing_module._stochastic_state(resources)

    assert payload["torch_cuda"] == expected
    assert calls == 1


def test_stochastic_state_cpu_path_does_not_probe_cuda_driver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)

    def unexpected_probe() -> bool:
        raise AssertionError("CPU-only checkpoint must not probe CUDA availability")

    monkeypatch.setattr(torch.cuda, "is_available", unexpected_probe)
    resources = SimpleNamespace(
        device=torch.device("cpu"),
        collector_model=SimpleNamespace(parameters=lambda: iter((SimpleNamespace(device=torch.device("cpu")),))),
        collector=SimpleNamespace(state_dict=lambda: {"sentinel": True}),
        learner=SimpleNamespace(
            dynamics_state_dict=lambda: {
                "version": "sts2-vtrace-learner-dynamics-v2",
                "collapse_batch_streak": 0,
                "entropy_breaker_remaining_updates": 0,
                "entropy_breaker_triggers": 0,
            }
        ),
    )

    assert checkpointing_module._stochastic_state(resources)["torch_cuda"] == []


def test_v3_is_model_initialization_only_and_six_heads_migrate_all_or_none(
    tmp_path: Path,
) -> None:
    config = _episodic_config()
    validated = ValidatedResumeCheckpoint(
        root=tmp_path,
        manifest={},
        metadata={
            "format": "sts2-recurrent-vtrace-checkpoint-v3",
            "model_config": asdict(config.model.to_model_config()),
            "encoding_contract": grounding_encoding_identity(),
            "model_state_spec": {},
        },
    )
    checkpointing_module._validate_metadata(
        validated,
        config=config,
        resolved_device=None,
        resolved_collector_device=None,
        model_only=True,
    )
    with pytest.raises(ValueError, match=r"unsupported exact-resume checkpoint format"):
        checkpointing_module._validate_metadata(
            validated,
            config=config,
            resolved_device="cpu",
            resolved_collector_device="cpu",
            model_only=False,
        )

    target_model = RecurrentCandidateModel(config.model.to_model_config())
    target = target_model.state_dict()
    legacy = {key: value.detach().clone() for key, value in target.items() if not key.startswith(_LONG_HEAD_PREFIXES)}
    fresh_heads = {key: value.detach().clone() for key, value in target.items() if key.startswith(_LONG_HEAD_PREFIXES)}
    migrated = checkpointing_module._model_parameter_initialization_state(
        legacy,
        target_state=target,
        allow_missing_transaction_heads=False,
        allow_missing_long_horizon_heads=True,
    )
    assert all(torch.equal(migrated[key], value) for key, value in legacy.items())
    assert all(torch.equal(migrated[key], value) for key, value in fresh_heads.items())

    partial = dict(target)
    partial.pop(next(iter(fresh_heads)))
    with pytest.raises(ValueError, match=r"all or none.*six long-horizon"):
        checkpointing_module._model_parameter_initialization_state(
            partial,
            target_state=target,
            allow_missing_transaction_heads=False,
            allow_missing_long_horizon_heads=True,
        )


def test_v3_parameter_initialization_starts_fresh_optimizer_replays_rng_and_lineage(
    tmp_path: Path,
) -> None:
    source_config = _config()
    source = build_training_resources(source_config, backend=FakeCombatBackend())
    try:
        source.optimizer.zero_grad(set_to_none=True)
        sum(parameter.square().mean() for parameter in source.model.parameters()).backward()
        source.optimizer.step()
        checkpoint = save_training_checkpoint(
            tmp_path / "v3-policy-source",
            config=source_config,
            resources=source,
            state=TrainingState(
                environment_steps=40_737,
                learner_updates=638,
                policy_version=638,
            ),
            checkpoint_load_mode="fresh",
        )
        legacy_state = _rewrite_as_v3_without_long_heads(checkpoint)
    finally:
        source.close()

    target_config = _episodic_config()
    with pytest.raises(ValueError, match=r"unsupported exact-resume checkpoint format"):
        preflight_training_checkpoint(
            checkpoint,
            config=target_config,
            resolved_device="cpu",
            resolved_collector_device="cpu",
        )

    target = build_training_resources(target_config, backend=FakeCombatBackend())
    try:
        assert target.episodic_replay is not None
        heads_before = {
            key: value.detach().clone()
            for key, value in target.model.state_dict().items()
            if key.startswith(_LONG_HEAD_PREFIXES)
        }
        replay_before = target.episodic_replay.state_dict()
        collector_before = target.collector.state_dict()
        python_before = random.getstate()
        numpy_before = np.random.get_state()
        torch_before = torch.get_rng_state().clone()

        parent = initialize_model_from_checkpoint(
            checkpoint,
            config=target_config,
            resources=target,
        )
        assert parent == checkpoint.resolve()
        for key, expected in legacy_state.items():
            assert torch.equal(target.model.state_dict()[key], expected), key
            assert torch.equal(target.collector_model.state_dict()[key], expected), key
        for key, expected in heads_before.items():
            assert torch.equal(target.model.state_dict()[key], expected), key
            assert torch.equal(target.collector_model.state_dict()[key], expected), key
        assert len(target.optimizer.state) == 0
        assert len(target.rollout_queue) == 0
        assert target.episodic_replay.state_dict() == replay_before
        assert target.collector.state_dict() == collector_before
        assert random.getstate() == python_before
        _assert_numpy_rng_equal(np.random.get_state(), numpy_before)
        assert torch.equal(torch.get_rng_state(), torch_before)

        migrated = save_training_checkpoint(
            tmp_path / "v4-long-horizon-lineage",
            config=target_config,
            resources=target,
            state=TrainingState(),
            parent_checkpoint=checkpoint,
            checkpoint_load_mode="model_initialization",
            parent_relation="model_parameter_initialization",
        )
        metadata = json.loads((migrated / "metadata.json").read_text(encoding="utf-8"))
        assert metadata["format"] == "sts2-recurrent-vtrace-checkpoint-v5"
        assert metadata["training_state"] == asdict(TrainingState())
        assert metadata["provenance"]["checkpoint_load_mode"] == "model_initialization"
        assert metadata["provenance"]["parent_checkpoint"]["relation"] == ("model_parameter_initialization")
        assert (migrated / "episodic_replay.pkl").is_file()
    finally:
        target.close()
