from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from sts2_rl.training import (
    CONFIG_VERSION,
    ENGINE_REVIVAL_MECHANISM,
    CurriculumConfig,
    DiagnosticsConfig,
    ModelConfig,
    OptimizationConfig,
    RolloutConfig,
    RuntimeConfig,
    TrainingConfig,
    engine_revival_identity,
    load_training_config,
    training_config_from_mapping,
)
from sts2_rl.training.config import EpisodicLearningConfig


def test_profiles_use_relational_recurrent_vtrace_v3_with_bounded_transaction_sidecar() -> None:
    default = load_training_config(profile="default")
    combat = load_training_config(profile="combat")
    preheat = load_training_config(profile="preheat")
    assert CONFIG_VERSION == "sts2-relational-curriculum-config-v10"
    assert default.model.architecture == "relational_candidate_v3"
    assert default.curriculum.reward_objective == "run"
    assert combat.curriculum.reward_objective == "combat"
    assert preheat.curriculum.mode == "native-revival-preheat"
    assert preheat.curriculum.revival_mechanism == ENGINE_REVIVAL_MECHANISM
    assert preheat.curriculum.revival_budget == -1
    assert preheat.optimization.discount == 1.0
    assert preheat.transaction_learning.enabled
    assert preheat.transaction_learning.completion_policy_weight == 0.25
    assert not default.episodic_learning.enabled
    assert not combat.episodic_learning.enabled
    assert preheat.episodic_learning.enabled
    assert preheat.episodic_learning.sample_sequences == 2
    assert preheat.episodic_learning.burn_in_steps == 32
    assert preheat.episodic_learning.learn_steps == 32
    assert preheat.episodic_learning.macro_sample_fraction == 0.5
    assert default.episodic_learning.macro_sample_fraction == 0.0
    assert combat.episodic_learning.macro_sample_fraction == 0.0
    assert preheat.episodic_learning.primary_success_tie_tolerance == 0.05
    assert preheat.episodic_learning.policy_gradient_max_lag == 128
    assert preheat.episodic_learning.revival_value_weight == 0.02
    assert preheat.transaction_learning.pairwise_ranking_weight == 0.10
    assert preheat.optimization.entropy_weight == 0.02
    assert preheat.optimization.entropy_weight_end == 0.004
    assert preheat.optimization.entropy_decay_updates == 2_000
    assert (
        preheat.episodic_learning.sample_sequences * preheat.episodic_learning.learn_steps
        == preheat.rollout.unroll_length * preheat.optimization.batch_unrolls
        == 64
    )
    assert preheat.episodic_learning.per_episode_capacity_bytes <= (preheat.episodic_learning.replay_capacity_bytes)
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
    assert preheat.rollout.queue_capacity == 64
    assert preheat.rollout.max_policy_lag == 64
    assert preheat.rollout.deterministic_probe_interval_episodes == 16
    assert preheat.rollout.deterministic_probe_environment_steps == (
        512,
        1_024,
        2_048,
        4_096,
    )
    assert preheat.runtime.evaluation_steps == (0, 100_000)
    assert preheat.runtime.evaluation_episodes == 12
    assert preheat.runtime.early_evaluation_steps == (
        5_000,
        10_000,
        20_000,
        30_000,
    )
    assert preheat.runtime.early_evaluation_episodes == 4
    assert preheat.runtime.final_audit_steps == (250_000,)
    assert preheat.runtime.final_audit_episodes == 20
    assert preheat.runtime.evaluation_liveness_guard_enabled
    assert preheat.diagnostics.combat_net_progress_window == 256
    assert preheat.diagnostics.combat_net_progress_room_windows == {
        "TEST_SUBJECT_BOSS": 128,
    }
    assert preheat.diagnostics.combat_net_progress_encounter_windows == {}
    assert default.diagnostics.combat_net_progress_room_windows == {}
    assert combat.diagnostics.combat_net_progress_room_windows == {}
    assert preheat.diagnostics.noncombat_durable_progress_window == 256
    assert preheat.diagnostics.combat_min_net_hp_fraction == 0.05
    assert preheat.runtime.log_dir.endswith("v20-liveness-guard")
    assert preheat.runtime.checkpoint_dir.endswith("v20-liveness-guard")
    assert preheat.runtime.checkpoint_interval_steps == 10_000
    assert default.runtime.log_dir.endswith("v9-long-horizon-heads")
    assert default.runtime.checkpoint_dir.endswith("v9-long-horizon-heads")
    assert combat.runtime.log_dir.endswith("v7-long-horizon-heads")
    assert combat.runtime.checkpoint_dir.endswith("v7-long-horizon-heads")
    assert default.runtime.evaluation_steps == (0, 10_000, 25_000, 50_000)
    mapping = default.to_mapping()
    assert "rollout" in mapping
    assert "episodic_learning" in mapping
    assert "replay" not in mapping
    assert "q_weight" not in mapping["optimization"]
    assert "reward_weight" not in mapping["optimization"]
    assert "terminal_weight" not in mapping["optimization"]


def test_v22_policy3918_warmstart_overlay_is_conservative_and_fully_audited() -> None:
    overlay = Path(__file__).parents[1] / "config" / "experiments" / "full_run_revival_v22_policy3918_warmstart.toml"
    config = load_training_config(profile="preheat", config_path=overlay)

    # Model-only initialization must preserve the frozen model/data contracts.
    base = load_training_config(profile="preheat")
    assert config.model == base.model
    assert config.rollout == base.rollout
    assert config.transaction_learning == base.transaction_learning
    assert config.episodic_learning == base.episodic_learning
    assert config.environment == base.environment
    assert config.curriculum.revival_budget == -1

    # A fresh Adam state and a mature inherited policy use conservative updates
    # and exploration rather than replaying the random-policy preheat schedule.
    assert config.optimization.learning_rate == pytest.approx(1.0e-4)
    assert config.optimization.entropy_weight == pytest.approx(0.006)
    assert config.optimization.entropy_weight_end == pytest.approx(0.002)
    assert config.curriculum.epsilon_start == pytest.approx(0.15)
    assert config.curriculum.epsilon_end == pytest.approx(0.05)

    assert config.runtime.total_environment_steps == 250_000
    assert config.runtime.seed == 1_000_000
    assert config.runtime.evaluation_steps == (
        0,
        50_000,
        100_000,
        150_000,
        200_000,
        240_000,
    )
    assert config.runtime.evaluation_episodes == 8
    assert config.runtime.final_audit_steps == (250_000,)
    assert config.runtime.final_audit_episodes == 20


def test_v25_budget64_overlay_changes_only_the_revival_curriculum_lineage() -> None:
    experiment_root = Path(__file__).parents[1] / "config" / "experiments"
    unlimited = load_training_config(
        profile="preheat",
        config_path=(experiment_root / "full_run_revival_v24_engine_bailout_model_init.toml"),
    )
    finite = load_training_config(
        profile="preheat",
        config_path=(experiment_root / "full_run_revival_v25_budget64_model_init.toml"),
    )

    # The model/data/replay contracts stay frozen for a strict parameter-only
    # initialization.  Only the hidden engine-bailout budget changes training
    # semantics; runtime paths and evaluation cadence are not lineage identity.
    assert finite.model == unlimited.model
    assert finite.rollout == unlimited.rollout
    assert finite.optimization == unlimited.optimization
    assert finite.transaction_learning == unlimited.transaction_learning
    assert finite.episodic_learning == unlimited.episodic_learning
    assert finite.environment == unlimited.environment
    assert finite.diagnostics == unlimited.diagnostics
    assert finite.curriculum.mode == "native-revival-preheat"
    assert finite.curriculum.revival_mechanism == ENGINE_REVIVAL_MECHANISM
    assert unlimited.curriculum.revival_budget == -1
    assert finite.curriculum.revival_budget == 64

    expected_lineage = unlimited.lineage_mapping()
    expected_lineage["curriculum"] = {
        **expected_lineage["curriculum"],
        "revival_budget": 64,
    }
    assert finite.lineage_mapping() != unlimited.lineage_mapping()
    assert finite.lineage_mapping() == expected_lineage

    assert finite.runtime.total_environment_steps == 100_000
    assert finite.runtime.seed == unlimited.runtime.seed == 1_000_000
    assert finite.runtime.log_dir.endswith("full-run-revival-v25-budget64-model-init")
    assert finite.runtime.checkpoint_dir.endswith("full-run-revival-v25-budget64-model-init")
    assert finite.runtime.checkpoint_interval_steps == 10_000
    assert finite.runtime.evaluation_steps == (0, 25_000, 50_000, 75_000, 90_000)
    assert finite.runtime.evaluation_episodes == 8
    assert finite.runtime.early_evaluation_steps == (5_000, 10_000, 20_000)
    assert finite.runtime.early_evaluation_episodes == 4
    assert finite.runtime.final_audit_steps == (100_000,)
    assert finite.runtime.final_audit_episodes == 32
    assert finite.runtime.evaluation_liveness_guard_enabled

def test_v22b_policy75_overlay_is_an_exact_continuation_lineage() -> None:
    experiment_root = Path(__file__).parents[1] / "config" / "experiments"
    warmstart = load_training_config(
        profile="preheat",
        config_path=(experiment_root / "full_run_revival_v22_policy3918_warmstart.toml"),
    )
    continuation = load_training_config(
        profile="preheat",
        config_path=(experiment_root / "full_run_revival_v22b_policy75_exact_continuation.toml"),
    )

    # Exact resume restores optimizer, queued unrolls, replay, recurrent state,
    # and RNG.  Only runtime controls excluded from lineage identity may move.
    assert continuation.lineage_mapping() == warmstart.lineage_mapping()
    assert continuation.model == warmstart.model
    assert continuation.rollout == warmstart.rollout
    assert continuation.optimization == warmstart.optimization
    assert continuation.curriculum == warmstart.curriculum
    assert continuation.transaction_learning == warmstart.transaction_learning
    assert continuation.episodic_learning == warmstart.episodic_learning
    assert continuation.environment == warmstart.environment
    assert continuation.diagnostics == warmstart.diagnostics

    assert continuation.runtime.total_environment_steps == 250_000
    assert continuation.runtime.seed == 1_000_000
    assert continuation.runtime.log_dir.endswith("full-run-revival-v22b-policy75-exact-continuation")
    assert continuation.runtime.checkpoint_dir.endswith("full-run-revival-v22b-policy75-exact-continuation")
    assert not continuation.runtime.evaluation_liveness_guard_enabled
    assert continuation.runtime.evaluation_steps == warmstart.runtime.evaluation_steps
    assert continuation.runtime.evaluation_episodes == warmstart.runtime.evaluation_episodes
    assert continuation.runtime.early_evaluation_steps == warmstart.runtime.early_evaluation_steps
    assert continuation.runtime.early_evaluation_episodes == warmstart.runtime.early_evaluation_episodes
    assert continuation.runtime.final_audit_steps == warmstart.runtime.final_audit_steps
    assert continuation.runtime.final_audit_episodes == warmstart.runtime.final_audit_episodes


def test_model_and_rollout_configs_fail_closed() -> None:
    with pytest.raises(ValueError, match="relational_candidate_v3"):
        ModelConfig(architecture="grounded_candidate_v1")
    with pytest.raises(ValueError, match="minimum_unrolls"):
        RolloutConfig(queue_capacity=4, minimum_unrolls=5)
    with pytest.raises(ValueError, match="one collector"):
        RolloutConfig(collector_workers=2)
    with pytest.raises(TypeError, match="integer"):
        RolloutConfig(unroll_length=True)  # type: ignore[arg-type]
    normalized = RolloutConfig(
        deterministic_probe_environment_steps=[512, 1_024],  # type: ignore[arg-type]
    )
    assert normalized.deterministic_probe_environment_steps == (512, 1_024)
    with pytest.raises(ValueError, match="strictly increasing"):
        RolloutConfig(deterministic_probe_environment_steps=(512, 512))
    with pytest.raises(ValueError, match="must be >= 1"):
        RolloutConfig(deterministic_probe_environment_steps=(0,))


def test_vtrace_and_diagnostics_bounds_are_strict() -> None:
    with pytest.raises(ValueError, match="vtrace_rho_clip"):
        OptimizationConfig(vtrace_rho_clip=0.0)
    with pytest.raises(ValueError, match="deadlock_repeat_threshold"):
        DiagnosticsConfig(deadlock_window=4, deadlock_repeat_threshold=5)
    with pytest.raises(ValueError, match="combat_net_progress_window"):
        DiagnosticsConfig(combat_net_progress_window=0)
    with pytest.raises(ValueError, match="noncombat_durable_progress_window"):
        DiagnosticsConfig(noncombat_durable_progress_window=0)
    with pytest.raises(TypeError, match="room_windows must be a table"):
        DiagnosticsConfig(combat_net_progress_room_windows=())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="identifiers must be non-empty"):
        DiagnosticsConfig(combat_net_progress_room_windows={" ": 8})
    with pytest.raises(TypeError, match="must be an integer"):
        DiagnosticsConfig(combat_net_progress_encounter_windows={"BOSS": True})
    with pytest.raises(ValueError, match="must be >= 1"):
        DiagnosticsConfig(combat_net_progress_encounter_windows={"BOSS": 0})
    with pytest.raises(ValueError, match="duplicate normalized identifier"):
        DiagnosticsConfig(
            combat_net_progress_room_windows={"boss": 4, "BOSS": 8},
        )
    normalized = DiagnosticsConfig(
        combat_net_progress_room_windows={" test_subject_boss ": 128},
    )
    assert normalized.combat_net_progress_room_windows == {
        "TEST_SUBJECT_BOSS": 128,
    }
    with pytest.raises(ValueError, match="combat_min_net_hp_fraction"):
        DiagnosticsConfig(combat_min_net_hp_fraction=0.0)
    with pytest.raises(ValueError, match="strictly increasing"):
        RuntimeConfig(evaluation_steps=(0, 10, 10))
    with pytest.raises(ValueError, match="must be disjoint"):
        RuntimeConfig(
            evaluation_steps=(0, 10),
            early_evaluation_steps=(10,),
            early_evaluation_episodes=2,
        )
    with pytest.raises(ValueError, match="requires early evaluation"):
        RuntimeConfig(evaluation_liveness_guard_enabled=True)
    with pytest.raises(ValueError, match="entropy_weight_end"):
        OptimizationConfig(entropy_weight=0.01, entropy_weight_end=0.02)


def test_environment_and_task_horizons_must_match() -> None:
    base = TrainingConfig()
    with pytest.raises(ValueError, match="full-run"):
        replace(base, curriculum=CurriculumConfig(reward_objective="combat"))


def test_native_revival_preheat_supports_a_headless_full_run_curriculum() -> None:
    with pytest.raises(ValueError, match="requires revival_mechanism"):
        CurriculumConfig(mode="native-revival-preheat", reward_objective="combat")
    curriculum = CurriculumConfig(
        mode="native-revival-preheat",
        reward_objective="combat",
        revival_mechanism=ENGINE_REVIVAL_MECHANISM,
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


def test_engine_revival_identity_is_explicit_and_fail_closed() -> None:
    identity = engine_revival_identity()
    assert identity["version"] == ENGINE_REVIVAL_MECHANISM
    assert identity["model_visible_game_entity"] is None
    assert identity["forced_kill_policy"] == "not intercepted"
    assert identity["native_death_prevention_order"] == (
        "native hooks before training bailout"
    )
    assert len(identity["fingerprint_sha256"]) == 64

    with pytest.raises(ValueError, match="engine revival mechanism"):
        CurriculumConfig(revival_mechanism=ENGINE_REVIVAL_MECHANISM)
    with pytest.raises(ValueError, match="engine-bailout-v1"):
        CurriculumConfig(
            mode="native-revival-preheat",
            reward_objective="combat",
            revival_mechanism="visible-relic-v0",  # type: ignore[arg-type]
            revival_budget=-1,
        )


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
    changed_combat_terminal_semantics = replace(
        base,
        diagnostics=replace(
            base.diagnostics,
            combat_net_progress_room_windows={"TEST_SUBJECT_BOSS": 128},
        ),
    )
    # A room-scoped liveness window changes which factual prefix becomes a
    # policy-failure label, so it must reject exact resume rather than masquerade
    # as an output-only runtime change. Model-only initialization remains the
    # explicit migration path.
    assert changed_combat_terminal_semantics.lineage_mapping() != base.lineage_mapping()


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


def test_episodic_learning_config_is_byte_bounded_and_fail_closed() -> None:
    config = EpisodicLearningConfig(enabled=True)
    assert config.replay_capacity_episodes == 64
    assert config.sample_sequences * config.learn_steps == 64
    assert config.policy_gradient_max_lag == 128
    with pytest.raises(TypeError, match="enabled must be a boolean"):
        EpisodicLearningConfig(enabled=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="per_episode_capacity_bytes"):
        EpisodicLearningConfig(
            replay_capacity_bytes=1024,
            per_episode_capacity_bytes=2048,
        )
    with pytest.raises(ValueError, match="secondary_advantage_fraction"):
        EpisodicLearningConfig(secondary_advantage_fraction=1.01)
    with pytest.raises(ValueError, match="primary_success_tie_tolerance"):
        EpisodicLearningConfig(primary_success_tie_tolerance=0.51)
    with pytest.raises(ValueError, match="importance_ratio_clip"):
        EpisodicLearningConfig(importance_ratio_clip=0.0)
    with pytest.raises(ValueError, match="task_value_weight"):
        EpisodicLearningConfig(task_value_weight=float("nan"))
    with pytest.raises(TypeError, match="macro_sample_fraction"):
        EpisodicLearningConfig(macro_sample_fraction=True)
    with pytest.raises(ValueError, match="macro_sample_fraction"):
        EpisodicLearningConfig(macro_sample_fraction=-0.01)
    with pytest.raises(ValueError, match="macro_sample_fraction"):
        EpisodicLearningConfig(macro_sample_fraction=1.01)
    with pytest.raises(ValueError, match="macro_sample_fraction"):
        EpisodicLearningConfig(macro_sample_fraction=float("nan"))


def test_missing_episodic_section_uses_disabled_compatibility_defaults() -> None:
    payload = TrainingConfig().to_mapping()
    payload.pop("episodic_learning")
    parsed = training_config_from_mapping(payload)
    assert parsed.episodic_learning == EpisodicLearningConfig()
    assert not parsed.episodic_learning.enabled


def test_episodic_learning_contract_is_part_of_lineage_identity() -> None:
    base = TrainingConfig()
    enabled = replace(
        base,
        episodic_learning=replace(base.episodic_learning, enabled=True),
    )
    assert enabled.lineage_mapping() != base.lineage_mapping()
    changed_sampling = replace(
        base,
        episodic_learning=replace(
            base.episodic_learning,
            macro_sample_fraction=0.5,
        ),
    )
    assert changed_sampling.lineage_mapping() != base.lineage_mapping()
    round_tripped = training_config_from_mapping(changed_sampling.to_mapping())
    assert round_tripped.episodic_learning.macro_sample_fraction == 0.5
    assert round_tripped == changed_sampling
