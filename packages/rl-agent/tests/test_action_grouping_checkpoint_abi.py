from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from sts2_rl.checkpoints import ValidatedResumeCheckpoint
from sts2_rl.contracts import (
    BackendCapabilities,
    CombatResetRequest,
    EnvironmentResult,
    EnvironmentTransition,
    ResetRequest,
    StepRequest,
)
from sts2_rl.encoding import grounding_encoding_identity
from sts2_rl.training import (
    CurriculumConfig,
    EnvironmentConfig,
    ModelConfig,
    OptimizationConfig,
    RolloutConfig,
    RuntimeConfig,
    TrainingConfig,
    build_training_resources,
    initialize_model_from_checkpoint,
    preflight_model_initialization,
    preflight_training_checkpoint,
    save_training_checkpoint,
)
from sts2_rl.training import checkpointing as checkpointing_module
from sts2_rl.training.checkpointing import TrainingState

_V8_ENCODING = {
    "version": "grounded-relational-runtime-encoding-v8",
    "min_token_feature_dim": 224,
    "feature_abi_end": 214,
    "fingerprint_sha256": ("8bc0204fe3201871cf3bdb3be39deaac9cc1b02f830ac2ba72d3e29bef0ddf58"),
}
_V9_ENCODING = {
    "version": "grounded-relational-runtime-encoding-v9",
    "min_token_feature_dim": 224,
    "feature_abi_end": 215,
    "fingerprint_sha256": ("a953c6a01cd0ae85e77f0916ee6f072f7a97a59cbfdae6967003f9e8894dba1b"),
}
_V10_ENCODING = {
    "version": "grounded-relational-runtime-encoding-v10",
    "min_token_feature_dim": 224,
    "feature_abi_end": 215,
    "fingerprint_sha256": ("4caae6f3c6baafb31ce476615776e22cdea2e6073ee7b4893247a4ffef2e524f"),
}
_V11_ENCODING = {
    "version": "grounded-relational-runtime-encoding-v11",
    "min_token_feature_dim": 224,
    "feature_abi_end": 215,
    "fingerprint_sha256": ("5d150d5949c70e49203f7808e663abcfcbd897bcb9d18a55852b117292503bb7"),
}


class _CombatBackend:
    def __init__(self) -> None:
        self._capabilities = BackendCapabilities(
            backend_name="checkpoint-abi-test",
            session_id="checkpoint-abi-session",
        )
        self._state_version = 0
        self._step = 0
        self._episode = 0

    @property
    def capabilities(self) -> BackendCapabilities:
        return self._capabilities

    @property
    def session_id(self) -> str:
        return self._capabilities.session_id

    @property
    def is_connected(self) -> bool:
        return True

    def health(self) -> dict[str, Any]:
        return {"ok": True}

    def get_spec(self) -> dict[str, Any]:
        return {"ok": True}

    def get_state(self) -> dict[str, Any]:
        return {"ok": True, "state_version": self._state_version}

    @staticmethod
    def _actions() -> tuple[dict[str, Any], ...]:
        return (
            {
                "action_handle": "attack",
                "kind": "play_card",
                "model_action_kind": "play_card",
                "card": {"id": "attack", "cost": 1},
                "target": {"id": "enemy", "side": "enemy"},
            },
            {
                "action_handle": "end",
                "kind": "end_turn",
                "model_action_kind": "end_turn",
            },
        )

    @staticmethod
    def _observation(*, terminal: bool) -> dict[str, Any]:
        return {
            "phase": "combat",
            "decision_domain": "combat",
            "player": {"id": "player", "hp": 50, "max_hp": 80},
            "combat": {
                "in_progress": not terminal,
                "enemies": [
                    {
                        "id": "enemy",
                        "hp": 0 if terminal else 30,
                        "max_hp": 30,
                    }
                ],
            },
            "run": {"act": 1, "floor": 1},
        }

    def reset(self, request: ResetRequest) -> EnvironmentResult:
        raise AssertionError("combat test backend requires combat_reset")

    def combat_reset(self, request: CombatResetRequest) -> EnvironmentResult:
        assert request.expected_state_version == self._state_version
        before = self._state_version
        self._state_version += 1
        self._step = 0
        self._episode += 1
        episode_id = f"episode-{self._episode}"
        return EnvironmentResult(
            episode_id=episode_id,
            step_index=0,
            observation=self._observation(terminal=False),
            legal_actions=self._actions(),
            transition=EnvironmentTransition(
                episode_id=episode_id,
                step_index=0,
                before_state_version=before,
                after_state_version=self._state_version,
                facts={"combat_result": "none", "terminal_reason": None},
            ),
            info={"reward_authority": "external-rl"},
        )

    def step(self, request: StepRequest) -> EnvironmentResult:
        assert request.expected_step_index == self._step
        before = self._state_version
        self._state_version += 1
        self._step += 1
        episode_id = f"episode-{self._episode}"
        return EnvironmentResult(
            episode_id=episode_id,
            step_index=self._step,
            observation=self._observation(terminal=True),
            legal_actions=(),
            transition=EnvironmentTransition(
                episode_id=episode_id,
                step_index=self._step,
                before_state_version=before,
                after_state_version=self._state_version,
                facts={
                    "combat_result": "victory",
                    "terminal_reason": "combat_victory",
                },
            ),
            terminated=True,
            terminal_reason="combat_victory",
            info={"reward_authority": "external-rl"},
        )

    def close(self) -> None:
        return None


def _config() -> TrainingConfig:
    base = TrainingConfig(
        profile="action-group-checkpoint-abi-test",
        model=ModelConfig(
            token_feature_dim=224,
            d_model=32,
            n_heads=4,
            ffn_dim=64,
            world_layers=1,
            latent_slots=4,
            latent_layers=1,
            local_layers=1,
            candidate_layers=1,
            recurrent_hidden_dim=32,
            dropout=0.0,
            type_vocab_size=32,
            role_vocab_size=32,
            owner_vocab_size=16,
            entity_vocab_size=128,
            zone_vocab_size=16,
            order_vocab_size=32,
            domain_count=8,
            max_world_tokens=24,
            max_candidates=256,
            max_candidate_local_tokens=5,
        ),
        optimization=OptimizationConfig(batch_unrolls=1),
        rollout=RolloutConfig(
            unroll_length=2,
            queue_capacity=8,
            minimum_unrolls=1,
            policy_sync_interval_unrolls=1,
            max_policy_lag=32,
        ),
        environment=EnvironmentConfig(
            backend="headless",
            scenario="combat",
            max_episode_steps=4,
        ),
        curriculum=CurriculumConfig(
            reward_objective="combat",
            epsilon_start=0.2,
            epsilon_end=0.1,
            epsilon_decay_steps=10,
        ),
        runtime=RuntimeConfig(
            device="cpu",
            collector_device="cpu",
            total_environment_steps=4,
            seed=7,
            checkpoint_interval_steps=100,
            evaluation_steps=(),
            evaluation_episodes=0,
        ),
    )
    return replace(
        base,
        transaction_learning=replace(base.transaction_learning, enabled=True),
    )


def _validated_metadata(
    tmp_path: Path,
    *,
    config: TrainingConfig,
    encoding: dict[str, Any],
) -> ValidatedResumeCheckpoint:
    return ValidatedResumeCheckpoint(
        root=tmp_path,
        manifest={},
        metadata={
            "format": "sts2-recurrent-vtrace-checkpoint-v4",
            "model_config": asdict(config.model.to_model_config()),
            "encoding_contract": encoding,
            "model_state_spec": {},
        },
    )


def _rewrite_checkpoint_encoding_contract(
    checkpoint: Path,
    *,
    encoding_contract: dict[str, Any],
) -> None:
    manifest_path = checkpoint / "checkpoint.manifest.json"
    metadata_path = checkpoint / "metadata.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["encoding_contract"] = dict(encoding_contract)
    metadata_bytes = (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode()
    metadata_path.write_bytes(metadata_bytes)
    for entry in manifest["files"]:
        if entry["path"] == "metadata.json":
            entry["size_bytes"] = len(metadata_bytes)
            entry["sha256"] = hashlib.sha256(metadata_bytes).hexdigest()
            break
    else:  # pragma: no cover - publisher always lists metadata
        raise AssertionError("metadata entry missing")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    "archived_encoding",
    [_V8_ENCODING, _V9_ENCODING, _V10_ENCODING],
)
def test_only_reviewed_legacy_to_v11_model_initialization_crosses_decision_abi(
    tmp_path: Path,
    archived_encoding: dict[str, Any],
) -> None:
    config = _config()
    assert grounding_encoding_identity() == _V11_ENCODING
    archived = _validated_metadata(
        tmp_path,
        config=config,
        encoding=archived_encoding,
    )

    with pytest.raises(ValueError, match="encoding contract does not match"):
        checkpointing_module._validate_metadata(
            archived,
            config=config,
            resolved_device=None,
            resolved_collector_device=None,
            model_only=False,
        )
    checkpointing_module._validate_metadata(
        archived,
        config=config,
        resolved_device=None,
        resolved_collector_device=None,
        model_only=True,
    )

    unknown_shape_compatible = _validated_metadata(
        tmp_path,
        config=config,
        encoding={**_V9_ENCODING, "fingerprint_sha256": "f" * 64},
    )
    with pytest.raises(ValueError, match="no reviewed.*initialization migration"):
        checkpointing_module._validate_metadata(
            unknown_shape_compatible,
            config=config,
            resolved_device=None,
            resolved_collector_device=None,
            model_only=True,
        )


def test_v9_observation_v2_initialization_inherits_only_model_parameters(
    tmp_path: Path,
) -> None:
    config = _config()
    source = build_training_resources(config, backend=_CombatBackend())
    try:
        source.rollout_queue.put(source.collector.collect_episode(record=True).unrolls[0])
        source.optimizer.zero_grad(set_to_none=True)
        objective = sum(parameter.square().mean() for parameter in source.model.parameters())
        objective.backward()
        source.optimizer.step()
        source_state = {key: value.detach().clone() for key, value in source.model.state_dict().items()}
        assert source.transaction_replay is not None
        source_replay = source.transaction_replay.state_dict()
        source_replay.update(
            {
                "put_count": 19,
                "sample_count": 11,
                "eviction_count": 7,
                "duplicate_count": 3,
            }
        )
        source.transaction_replay.load_state_dict(source_replay)
        source_training_state = TrainingState(
            environment_steps=83_841,
            learner_updates=1_255,
            episodes=70,
            evaluation_episodes=4,
            policy_version=1_255,
            actor_policy_version=1_255,
            consumed_unrolls=5_020,
            maximum_observed_candidates=2_068,
        )
        checkpoint = save_training_checkpoint(
            tmp_path / "pre-grouping-policy",
            config=config,
            resources=source,
            state=source_training_state,
            run_id="pre-grouping-source",
            checkpoint_load_mode="fresh",
        )
    finally:
        source.close()

    _rewrite_checkpoint_encoding_contract(
        checkpoint,
        encoding_contract=_V9_ENCODING,
    )
    with pytest.raises(ValueError, match="encoding contract does not match"):
        preflight_training_checkpoint(
            checkpoint,
            config=config,
            resolved_device="cpu",
            resolved_collector_device="cpu",
        )
    assert preflight_model_initialization(checkpoint, config=config).root == (checkpoint.resolve())

    target = build_training_resources(config, backend=_CombatBackend())
    try:
        assert target.transaction_replay is not None
        initial_python_rng = random.getstate()
        initial_numpy_rng = np.random.get_state()
        initial_torch_rng = torch.get_rng_state().clone()
        initial_collector_state = target.collector.state_dict()
        initial_replay = target.transaction_replay.state_dict()

        parent = initialize_model_from_checkpoint(
            checkpoint,
            config=config,
            resources=target,
        )

        assert parent == checkpoint.resolve()
        assert len(target.optimizer.state) == 0
        assert len(target.rollout_queue) == 0
        assert target.collector.state_dict() == initial_collector_state
        assert random.getstate() == initial_python_rng
        restored_numpy_rng = np.random.get_state()
        assert restored_numpy_rng[0] == initial_numpy_rng[0]
        assert np.array_equal(restored_numpy_rng[1], initial_numpy_rng[1])
        assert restored_numpy_rng[2:] == initial_numpy_rng[2:]
        assert torch.equal(torch.get_rng_state(), initial_torch_rng)
        assert target.transaction_replay.metrics() == {
            "version": initial_replay["version"],
            "size": 0,
            "capacity": initial_replay["capacity"],
            "storage_nbytes": 0,
            "byte_capacity": initial_replay["byte_capacity"],
            "put_count": 0,
            "sample_count": 0,
            "eviction_count": 0,
            "duplicate_count": 0,
            "deadlock_size": 0,
        }
        assert target.transaction_replay.state_dict()["rng_state"] == (initial_replay["rng_state"])
        for key, expected in source_state.items():
            assert torch.equal(target.model.state_dict()[key], expected), key
            assert torch.equal(target.collector_model.state_dict()[key], expected), key

        migrated = save_training_checkpoint(
            tmp_path / "strict-grouping-lineage-step-zero",
            config=config,
            resources=target,
            state=TrainingState(),
            parent_checkpoint=checkpoint,
            run_id="strict-card-selection-grouping",
            checkpoint_load_mode="model_initialization",
            parent_relation="model_parameter_initialization",
        )
        metadata = json.loads((migrated / "metadata.json").read_text(encoding="utf-8"))
        assert metadata["encoding_contract"] == _V11_ENCODING
        assert metadata["training_state"] == asdict(TrainingState())
        provenance = metadata["provenance"]
        assert provenance["checkpoint_load_mode"] == "model_initialization"
        parent_metadata = provenance["parent_checkpoint"]
        assert parent_metadata["relation"] == "model_parameter_initialization"
        assert parent_metadata["training_state"] == asdict(source_training_state)
    finally:
        target.close()
