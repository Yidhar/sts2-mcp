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


def test_profiles_use_relational_recurrent_vtrace_v3_with_bounded_transaction_sidecar() -> None:
    default = load_training_config(profile="default")
    combat = load_training_config(profile="combat")
    preheat = load_training_config(profile="preheat")
    assert CONFIG_VERSION == "sts2-relational-curriculum-config-v4"
    assert default.model.architecture == "relational_candidate_v3"
    assert default.curriculum.reward_objective == "run"
    assert combat.curriculum.reward_objective == "combat"
    assert preheat.curriculum.mode == "native-revival-preheat"
    assert preheat.curriculum.revival_relic_id == "RELIC.LIZARD_TAIL"
    assert preheat.curriculum.revival_budget == -1
    assert preheat.optimization.discount == 1.0
    assert preheat.transaction_learning.enabled
    assert preheat.transaction_learning.completion_policy_weight == 0.25
    assert preheat.environment.scenario == "full-run"
    assert preheat.curriculum.reward_objective == "run"
    assert preheat.environment.encounter_id is None
    assert preheat.environment.max_episode_steps == 30_000
    assert preheat.rollout.unroll_length == 16
    assert preheat.optimization.batch_unrolls == 4
    assert default.model.max_world_tokens == 2048
    assert combat.model.max_world_tokens == 2048
    assert preheat.model.max_world_tokens == 2048
    assert preheat.model.max_candidate_local_tokens == 64
    assert default.model.max_candidates == 256
    assert combat.model.max_candidates == 256
    assert preheat.model.max_candidates == 256
    assert preheat.rollout.minimum_unrolls == 4
    assert preheat.runtime.evaluation_steps == (0, 30_000, 100_000, 250_000)
    assert preheat.runtime.evaluation_episodes == 12
    assert preheat.diagnostics.combat_net_progress_window == 256
    assert preheat.diagnostics.noncombat_durable_progress_window == 256
    assert preheat.diagnostics.combat_min_net_hp_fraction == 0.05
    assert preheat.runtime.log_dir.endswith("v14-transaction-liveness-terminal-fix")
    assert preheat.runtime.checkpoint_dir.endswith("v14-transaction-liveness-terminal-fix")
    assert preheat.runtime.checkpoint_interval_steps == 10_000
    assert "transaction-liveness-terminal-fix" in preheat.runtime.log_dir
    assert "transaction-liveness-terminal-fix" in preheat.runtime.checkpoint_dir
    assert default.runtime.evaluation_steps == (0, 10_000, 25_000, 50_000)
    mapping = default.to_mapping()
    assert "rollout" in mapping
    assert "replay" not in mapping
    assert "q_weight" not in mapping["optimization"]
    assert "reward_weight" not in mapping["optimization"]
    assert "terminal_weight" not in mapping["optimization"]


def test_model_and_rollout_configs_fail_closed() -> None:
    with pytest.raises(ValueError, match="relational_candidate_v3"):
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
    with pytest.raises(ValueError, match="combat_net_progress_window"):
        DiagnosticsConfig(combat_net_progress_window=0)
    with pytest.raises(ValueError, match="noncombat_durable_progress_window"):
        DiagnosticsConfig(noncombat_durable_progress_window=0)
    with pytest.raises(ValueError, match="combat_min_net_hp_fraction"):
        DiagnosticsConfig(combat_min_net_hp_fraction=0.0)
    with pytest.raises(ValueError, match="strictly increasing"):
        RuntimeConfig(evaluation_steps=(0, 10, 10))


def test_environment_and_task_horizons_must_match() -> None:
    base = TrainingConfig()
    with pytest.raises(ValueError, match="full-run"):
        replace(base, curriculum=CurriculumConfig(reward_objective="combat"))


def test_native_revival_preheat_supports_a_headless_full_run_curriculum() -> None:
    with pytest.raises(ValueError, match="requires revival_relic_id"):
        CurriculumConfig(mode="native-revival-preheat", reward_objective="combat")
    curriculum = CurriculumConfig(
        mode="native-revival-preheat",
        reward_objective="combat",
        revival_relic_id="RELIC.LIZARD_TAIL",
        revival_budget=-1,
    )
    combat = TrainingConfig(
        environment=replace(TrainingConfig().environment, scenario="combat"),
        curriculum=curriculum,
        optimization=OptimizationConfig(discount=1.0),
    )
    assert combat.environment.scenario == "combat"
    full_run = load_training_config(profile="preheat")
    assert full_run.environment.scenario == "full-run"


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
