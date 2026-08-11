from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pytest

from sts2_rl.checkpoints import ValidatedResumeCheckpoint
from sts2_rl.encoding import grounding_encoding_identity
from sts2_rl.training import (
    CurriculumConfig,
    EnvironmentConfig,
    ModelConfig,
    OptimizationConfig,
    RolloutConfig,
    RuntimeConfig,
    TrainingConfig,
)
from sts2_rl.training import checkpointing as checkpointing_module

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
_V12_ENCODING = {
    "version": "grounded-relational-runtime-encoding-v12",
    "min_token_feature_dim": 224,
    "feature_abi_end": 215,
    "fingerprint_sha256": ("d1bc0220f7aa58e7afacaa83c1fa1ce339b65d58729f651012cd81d5ad4febf3"),
}
_V13_ENCODING = {
    "version": "grounded-relational-runtime-encoding-v13",
    "min_token_feature_dim": 224,
    "feature_abi_end": 215,
    "fingerprint_sha256": ("ac119f0d1fe0de5c09394e091169f3b7712084bce8a90a9d02be4732f60ce5bf"),
}
_V14_ENCODING = {
    "version": "grounded-relational-runtime-encoding-v14",
    "min_token_feature_dim": 224,
    "feature_abi_end": 215,
    "fingerprint_sha256": ("6a169803fdcd399272357dfe351a8b7375f16a1cb9e7cfccdc3b047f13f746ce"),
}
_V15_ENCODING = {
    "version": "grounded-relational-runtime-encoding-v15",
    "min_token_feature_dim": 224,
    "feature_abi_end": 215,
    "fingerprint_sha256": ("d5f84bc31014e7e043934af0fc6b0f1f40092fc14a96478845d38fa08bbc9aee"),
}
_V16_ENCODING = {
    "version": "grounded-relational-runtime-encoding-v16",
    "min_token_feature_dim": 224,
    "feature_abi_end": 215,
    "fingerprint_sha256": ("3cc73fd8910b005702ee4b408116b18b1c08a3d810f7301641c09fa3957ca70a"),
}


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
            "format": "sts2-recurrent-vtrace-checkpoint-v5",
            "model_config": asdict(config.model.to_model_config()),
            "encoding_contract": encoding,
            "model_state_spec": {},
        },
    )


@pytest.mark.parametrize(
    "archived_encoding",
    (_V8_ENCODING, _V9_ENCODING, _V10_ENCODING, _V11_ENCODING),
)
def test_pre_v12_model_initialization_cannot_jump_to_current_encoding(
    tmp_path: Path,
    archived_encoding: dict[str, Any],
) -> None:
    config = _config()
    current = grounding_encoding_identity()
    assert current["version"] == "grounded-relational-runtime-encoding-v17"
    assert current != _V16_ENCODING
    archived = _validated_metadata(
        tmp_path,
        config=config,
        encoding=archived_encoding,
    )

    for model_only in (False, True):
        expected = (
            r"no reviewed.*initialization migration"
            if model_only
            else r"encoding contract does not match"
        )
        with pytest.raises(ValueError, match=expected):
            checkpointing_module._validate_metadata(
                archived,
                config=config,
                resolved_device=None,
                resolved_collector_device=None,
                model_only=model_only,
            )


def test_archived_model_initialization_migrations_end_at_v16() -> None:
    assert checkpointing_module._has_reviewed_model_initialization_encoding_path(
        _V14_ENCODING,
        _V16_ENCODING,
    )
    assert not checkpointing_module._has_reviewed_model_initialization_encoding_path(
        _V16_ENCODING,
        grounding_encoding_identity(),
    )


@pytest.mark.parametrize(
    "source_encoding",
    (_V12_ENCODING, _V13_ENCODING, _V14_ENCODING, _V15_ENCODING, _V16_ENCODING),
    ids=("v12", "v13", "v14", "v15", "v16"),
)
def test_legacy_semantic_candidate_encodings_require_explicit_new_lineage(
    tmp_path: Path,
    source_encoding: dict[str, Any],
) -> None:
    archived = _validated_metadata(
        tmp_path,
        config=_config(),
        encoding=source_encoding,
    )
    with pytest.raises(ValueError, match=r"no reviewed.*initialization migration"):
        checkpointing_module._validate_metadata(
            archived,
            config=_config(),
            resolved_device=None,
            resolved_collector_device=None,
            model_only=True,
        )
