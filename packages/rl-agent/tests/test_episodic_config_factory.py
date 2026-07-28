from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from sts2_rl.training import factory as factory_module
from sts2_rl.training.config import (
    CurriculumConfig,
    EnvironmentConfig,
    EpisodicLearningConfig,
    ModelConfig,
    TrainingConfig,
)
from sts2_rl.training.episode_replay import BoundedEpisodicReplay


class _UnusedBackend:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _CapturingLearner:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.episodic_config = kwargs["episodic_config"]


def _small_config() -> TrainingConfig:
    base = TrainingConfig()
    return replace(
        base,
        model=ModelConfig(
            token_feature_dim=224,
            d_model=16,
            n_heads=4,
            ffn_dim=32,
            world_layers=1,
            latent_slots=2,
            latent_layers=1,
            local_layers=1,
            candidate_layers=1,
            recurrent_hidden_dim=16,
            dropout=0.0,
            type_vocab_size=16,
            role_vocab_size=16,
            owner_vocab_size=16,
            entity_vocab_size=64,
            zone_vocab_size=8,
            order_vocab_size=16,
            domain_count=8,
            max_world_tokens=16,
            max_candidates=8,
            max_candidate_local_tokens=4,
        ),
        runtime=replace(
            base.runtime,
            device="cpu",
            collector_device="cpu",
            evaluation_episodes=0,
        ),
        episodic_learning=replace(
            base.episodic_learning,
            enabled=True,
            replay_capacity_episodes=3,
            replay_capacity_bytes=4096,
            per_episode_capacity_bytes=2048,
            max_segments_per_episode=2,
            sample_sequences=2,
            burn_in_steps=4,
            learn_steps=8,
            macro_sample_fraction=0.5,
        ),
    )


def test_factory_wires_bounded_episodic_replay_and_learner_config(monkeypatch: Any) -> None:
    monkeypatch.setattr(factory_module, "VTraceLearner", _CapturingLearner)
    config = _small_config()
    backend = _UnusedBackend()
    resources = factory_module.build_training_resources(config, backend=backend)  # type: ignore[arg-type]
    try:
        assert isinstance(resources.episodic_replay, BoundedEpisodicReplay)
        assert resources.learner.episodic_config is config.episodic_learning  # type: ignore[attr-defined]
        assert resources.learner.episodic_config.fresh_policy_sequences == 0  # type: ignore[attr-defined]
        assert resources.collector.episodic_learning_enabled
        assert resources.episodic_replay.metrics() == {
            "version": "sts2-episodic-replay-v3",
            "size": 0,
            "capacity": 3,
            "storage_nbytes": 0,
            "byte_capacity": 4096,
            "episode_byte_capacity": 2048,
            "max_segments_per_episode": 2,
            "put_count": 0,
            "sample_count": 0,
            "macro_sample_count": 0,
            "eviction_count": 0,
            "duplicate_count": 0,
            "oversize_count": 0,
            "maximum_observed_episode_steps": 0,
        }
    finally:
        resources.close()
    assert backend.closed


def test_factory_omits_episodic_replay_when_disabled(monkeypatch: Any) -> None:
    monkeypatch.setattr(factory_module, "VTraceLearner", _CapturingLearner)
    config = replace(
        _small_config(),
        episodic_learning=replace(_small_config().episodic_learning, enabled=False),
    )
    backend = _UnusedBackend()
    resources = factory_module.build_training_resources(config, backend=backend)  # type: ignore[arg-type]
    try:
        assert resources.episodic_replay is None
        assert not resources.learner.episodic_config.enabled  # type: ignore[attr-defined]
        assert not resources.collector.episodic_learning_enabled
    finally:
        resources.close()


def test_episodic_learning_rejects_non_full_run_scenario() -> None:
    with pytest.raises(
        ValueError,
        match="episodic learning requires environment.scenario='full-run'",
    ):
        TrainingConfig(
            episodic_learning=EpisodicLearningConfig(enabled=True),
            environment=EnvironmentConfig(scenario="combat"),
            curriculum=CurriculumConfig(reward_objective="combat"),
        )


def test_episodic_learning_rejects_non_run_reward_objective() -> None:
    with pytest.raises(
        ValueError,
        match="episodic learning requires curriculum.reward_objective='run'",
    ):
        TrainingConfig(
            episodic_learning=EpisodicLearningConfig(enabled=True),
            environment=EnvironmentConfig(scenario="full-run"),
            curriculum=CurriculumConfig(reward_objective="act1"),
        )
