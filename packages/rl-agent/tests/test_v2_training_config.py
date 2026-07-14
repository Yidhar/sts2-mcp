from __future__ import annotations

from dataclasses import replace

import pytest

from sts2_rl.training import (
    CONFIG_VERSION,
    CurriculumConfig,
    DiagnosticsConfig,
    ModelConfig,
    OptimizationConfig,
    RolloutConfig,
    RuntimeConfig,
    TrainingConfig,
    load_training_config,
    training_config_from_mapping,
)


def test_profiles_use_recurrent_vtrace_v2_without_replay() -> None:
    default = load_training_config(profile="default")
    combat = load_training_config(profile="combat")
    preheat = load_training_config(profile="preheat")
    assert CONFIG_VERSION == "sts2-recurrent-curriculum-config-v2"
    assert default.model.architecture == "recurrent_candidate_v2"
    assert default.curriculum.reward_objective == "act1"
    assert combat.curriculum.reward_objective == "combat"
    assert preheat.curriculum.mode == "native-revival-preheat"
    assert preheat.curriculum.revival_relic_id == "RELIC.LIZARD_TAIL"
    assert preheat.curriculum.revival_budget == -1
    assert preheat.optimization.discount == 1.0
    assert preheat.environment.encounter_id == "FUZZY_WURM_CRAWLER_WEAK"
    assert preheat.environment.max_episode_steps == 512
    assert default.runtime.evaluation_steps == (0, 10_000, 25_000, 50_000)
    mapping = default.to_mapping()
    assert "rollout" in mapping
    assert "replay" not in mapping
    assert "q_weight" not in mapping["optimization"]
    assert "reward_weight" not in mapping["optimization"]
    assert "terminal_weight" not in mapping["optimization"]


def test_model_and_rollout_configs_fail_closed() -> None:
    with pytest.raises(ValueError, match="recurrent_candidate_v2"):
        ModelConfig(architecture="grounded_candidate_v1")
    with pytest.raises(ValueError, match="minimum_unrolls"):
        RolloutConfig(queue_capacity=4, minimum_unrolls=5)
    with pytest.raises(ValueError, match="one collector"):
        RolloutConfig(collector_workers=2)
    with pytest.raises(TypeError, match="integer"):
        RolloutConfig(unroll_length=True)  # type: ignore[arg-type]


def test_vtrace_and_diagnostics_bounds_are_strict() -> None:
    with pytest.raises(ValueError, match="vtrace_rho_clip"):
        OptimizationConfig(vtrace_rho_clip=0.0)
    with pytest.raises(ValueError, match="deadlock_repeat_threshold"):
        DiagnosticsConfig(deadlock_window=4, deadlock_repeat_threshold=5)
    with pytest.raises(ValueError, match="strictly increasing"):
        RuntimeConfig(evaluation_steps=(0, 10, 10))


def test_environment_and_task_horizons_must_match() -> None:
    base = TrainingConfig()
    with pytest.raises(ValueError, match="full-run"):
        replace(base, curriculum=CurriculumConfig(reward_objective="combat"))


def test_native_revival_preheat_is_a_bounded_headless_combat_curriculum() -> None:
    with pytest.raises(ValueError, match="requires revival_relic_id"):
        CurriculumConfig(mode="native-revival-preheat", reward_objective="combat")
    curriculum = CurriculumConfig(
        mode="native-revival-preheat",
        reward_objective="combat",
        revival_relic_id="RELIC.LIZARD_TAIL",
        revival_budget=-1,
    )
    with pytest.raises(ValueError, match="full-run|combat scenario"):
        replace(TrainingConfig(), curriculum=curriculum)


def test_preheat_and_standard_discount_contracts_fail_closed() -> None:
    preheat = load_training_config(profile="preheat")
    with pytest.raises(ValueError, match="reward contract discount 1.0"):
        replace(
            preheat,
            optimization=replace(preheat.optimization, discount=0.997),
        )
    standard = load_training_config(profile="default")
    with pytest.raises(ValueError, match="reward contract discount 0.997"):
        replace(
            standard,
            optimization=replace(standard.optimization, discount=1.0),
        )


def test_runtime_output_schedule_is_not_lineage_but_rollout_contract_is() -> None:
    base = TrainingConfig()
    moved_outputs = replace(
        base,
        runtime=replace(
            base.runtime,
            total_environment_steps=2_000_000,
            log_dir="another/run",
            evaluation_steps=(0, 50_000),
        ),
    )
    assert moved_outputs.lineage_mapping() == base.lineage_mapping()
    changed_unroll = replace(
        base,
        rollout=replace(base.rollout, unroll_length=32),
    )
    assert changed_unroll.lineage_mapping() != base.lineage_mapping()


def test_old_v1_config_is_rejected_instead_of_migrated() -> None:
    payload = TrainingConfig().to_mapping()
    payload["version"] = "sts2-grounded-baseline-config-v2"
    with pytest.raises(ValueError, match="unsupported"):
        training_config_from_mapping(payload)


def test_unknown_replay_section_is_rejected() -> None:
    payload = TrainingConfig().to_mapping()
    payload["replay"] = {"capacity": 100_000}
    with pytest.raises(ValueError, match="unknown training config sections"):
        training_config_from_mapping(payload)
