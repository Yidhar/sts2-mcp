from __future__ import annotations

from dataclasses import replace

import pytest

from sts2_rl.training import (
    CONFIG_VERSION,
    EnvironmentConfig,
    ModelConfig,
    OptimizationConfig,
    ReplayConfig,
    RuntimeConfig,
    TrainingConfig,
    exploration_epsilon,
    load_training_config,
    training_config_from_mapping,
)


def test_default_and_combat_profiles_select_matching_horizons() -> None:
    default = load_training_config(profile="default")
    combat = load_training_config(profile="combat")

    assert default.version == CONFIG_VERSION
    assert default.environment.scenario == "full-run"
    assert default.curriculum.reward_objective == "run"
    assert combat.environment.scenario == "combat"
    assert combat.curriculum.reward_objective == "combat"
    assert default.model.architecture == combat.model.architecture == "grounded_candidate_v1"
    assert default.runtime.warmup_credit_policy == "discard"
    assert combat.runtime.warmup_credit_policy == "discard"


def test_strict_dotted_overrides_accept_known_and_reject_retired_keys() -> None:
    config = load_training_config(
        overrides=("runtime.total_environment_steps=123", "model.max_candidates=17")
    )
    assert config.runtime.total_environment_steps == 123
    assert config.model.max_candidates == 17

    with pytest.raises(ValueError, match="unknown config override key"):
        load_training_config(overrides=("model.mcts_simulations=64",))


def test_old_architecture_and_mutable_reward_discount_fail_closed() -> None:
    with pytest.raises(ValueError, match="grounded_candidate_v1"):
        training_config_from_mapping(
            {
                "version": CONFIG_VERSION,
                "model": {"architecture": "token_memory_v1"},
            }
        )
    with pytest.raises(ValueError, match="immutable reward-spec discount"):
        training_config_from_mapping(
            {
                "version": CONFIG_VERSION,
                "optimization": {"discount": 0.9},
            }
        )


def test_exploration_schedule_is_bounded_and_monotonic() -> None:
    config = TrainingConfig()
    values = [exploration_epsilon(config, step) for step in (0, 10, 1000, 250_000, 999_999)]
    assert values == sorted(values, reverse=True)
    assert values[0] == config.curriculum.epsilon_start
    assert values[-1] == config.curriculum.epsilon_end


@pytest.mark.parametrize(
    "factory",
    (
        lambda: OptimizationConfig(learning_rate=float("nan")),
        lambda: OptimizationConfig(policy_weight=float("inf")),
        lambda: ReplayConfig(
            coverage_fraction=float("nan"),
            recent_fraction=0.25,
            priority_fraction=0.25,
        ),
        lambda: ReplayConfig(priority_epsilon=float("nan")),
        lambda: ModelConfig(dropout=float("nan")),
    ),
)
def test_all_numeric_config_values_must_be_finite(factory: object) -> None:
    with pytest.raises(ValueError, match="finite"):
        factory()  # type: ignore[operator]


@pytest.mark.parametrize(
    "factory",
    (
        lambda: OptimizationConfig(batch_size=True),
        lambda: ReplayConfig(capacity=True),
        lambda: RuntimeConfig(total_environment_steps=True),
        lambda: ModelConfig(d_model=True),
    ),
)
def test_booleans_cannot_masquerade_as_integer_config(factory: object) -> None:
    with pytest.raises(TypeError, match="integer"):
        factory()  # type: ignore[operator]


def test_environment_backend_and_scenario_fields_fail_when_ignored() -> None:
    with pytest.raises(ValueError, match="session_path.*live"):
        EnvironmentConfig(backend="headless", session_path="session.json")
    with pytest.raises(ValueError, match="sim_exe_path.*headless"):
        EnvironmentConfig(backend="live", sim_exe_path="sim.exe")
    with pytest.raises(ValueError, match="encounter_id.*combat"):
        EnvironmentConfig(scenario="full-run", encounter_id="boss")


def test_warmup_credit_policy_is_strict_and_legacy_mode_is_explicit() -> None:
    assert RuntimeConfig(warmup_credit_policy="accrue").warmup_credit_policy == "accrue"
    with pytest.raises(ValueError, match="warmup_credit_policy"):
        RuntimeConfig(warmup_credit_policy="catch-up")  # type: ignore[arg-type]


def test_lineage_mapping_excludes_only_mutable_execution_controls() -> None:
    config = TrainingConfig()
    mutable_runtime = replace(
        config.runtime,
        total_environment_steps=2_000_000,
        log_dir="other/logs",
        checkpoint_dir="other/checkpoints",
        checkpoint_interval_steps=10,
        evaluation_interval_steps=20,
        evaluation_episodes=3,
    )
    assert replace(config, runtime=mutable_runtime).lineage_mapping() == config.lineage_mapping()

    changed_seed = replace(config, runtime=replace(config.runtime, seed=99))
    changed_cadence = replace(
        config,
        runtime=replace(config.runtime, train_every_steps=8),
    )
    changed_warmup_policy = replace(
        config,
        runtime=replace(config.runtime, warmup_credit_policy="accrue"),
    )
    assert changed_seed.lineage_mapping() != config.lineage_mapping()
    assert changed_cadence.lineage_mapping() != config.lineage_mapping()
    assert changed_warmup_policy.lineage_mapping() != config.lineage_mapping()
