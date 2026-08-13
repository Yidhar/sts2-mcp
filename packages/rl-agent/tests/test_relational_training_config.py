from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from sts2_rl.training import (
    CONFIG_VERSION,
    ENGINE_REVIVAL_MECHANISM,
    CurriculumConfig,
    DiagnosticsConfig,
    FailureCreditConfig,
    ModelConfig,
    OptimizationConfig,
    RolloutConfig,
    RuntimeConfig,
    TrainingConfig,
    TransactionLearningConfig,
    engine_revival_identity,
    load_training_config,
    training_config_from_mapping,
)
from sts2_rl.training.config import (
    EpisodicLearningConfig,
    model_initialization_config_from_mapping,
)
from tests.archived_experiment_config import load_archived_training_config


def _remove_v15_transaction_lifecycle_fields(payload: dict[str, object]) -> None:
    transaction_learning = payload["transaction_learning"]
    assert isinstance(transaction_learning, dict)
    transaction_learning.pop("lifecycle_smdp_q_weight")


def _add_retired_v20_policy_fields(payload: dict[str, object]) -> None:
    """Reproduce a real pre-v20 payload: it still spelled out the retired
    entropy breaker, completion/selection-group CE and entry-support corridor
    keys that config v20 deleted."""

    optimization = payload["optimization"]
    transaction_learning = payload["transaction_learning"]
    assert isinstance(optimization, dict)
    assert isinstance(transaction_learning, dict)
    optimization["entropy_breaker"] = "policy-collapse-v2"
    transaction_learning["completion_policy_weight"] = 0.15
    transaction_learning["macro_option_group_completion_weight"] = 0.05


def _remove_v16_guard_field(payload: dict[str, object]) -> None:
    runtime = payload["runtime"]
    assert isinstance(runtime, dict)
    runtime.pop("evaluation_guard_enforcement_start_steps")


_RETIRED_V20_EPISODIC_FIELDS: dict[str, object] = {
    "fresh_policy_sequences": 1,
    "primary_policy_weight": 0.25,
    "revival_policy_weight": 0.05,
    "secondary_advantage_fraction": 0.25,
    "primary_success_tie_tolerance": 0.05,
    "importance_ratio_clip": 1.0,
    "success_policy_trust_region_epsilon": 0.20,
    "success_imitation_exempt_surfaces": ["rest_site"],
    "act_segment_imitation_enabled": True,
    "act_segment_policy_weight": 0.30,
    "act_segment_min_exit_hp_ratio": 0.35,
    "act_segment_max_revival_fraction": 0.34,
    "policy_gradient_max_lag": 128,
}
_RETIRED_V20_FAILURE_FIELDS: dict[str, object] = {
    "policy_gradient_max_lag": 128,
    "matched_outcome_pair_quota": 1,
    "liveness_risk_actor_start_update": 512,
    "liveness_cost_actor_weight": 0.10,
    "liveness_direct_policy_weight": 0.25,
    "liveness_cycle_policy_weight": 0.10,
    "liveness_contrast_policy_weight": 0.10,
    "liveness_completion_policy_weight": 0.0,
    "liveness_risk_advantage_clip": 0.25,
    "liveness_risk_actor_min_selected_probability": 0.0,
    "liveness_contrast_margin": 0.10,
}


def _add_retired_v20_imitation_and_actor_fields(payload: dict[str, object]) -> None:
    # Reproduce a real pre-v20 payload: it still spelled out the retired
    # imitation channels and liveness policy-actor keys that config v20
    # deleted.
    failure = payload["failure_credit"]
    episodic = payload["episodic_learning"]
    assert isinstance(failure, dict)
    assert isinstance(episodic, dict)
    episodic.update(_RETIRED_V20_EPISODIC_FIELDS)
    failure.update(_RETIRED_V20_FAILURE_FIELDS)


def _assert_retired_v20_fields_are_stripped(payload: dict[str, object]) -> None:
    failure = payload["failure_credit"]
    episodic = payload["episodic_learning"]
    assert isinstance(failure, dict)
    assert isinstance(episodic, dict)
    for field in _RETIRED_V20_EPISODIC_FIELDS:
        assert field not in episodic
    for field in _RETIRED_V20_FAILURE_FIELDS:
        assert field not in failure


def _remove_v18_act_prefix_fields(payload: dict[str, object]) -> None:
    episodic = payload["episodic_learning"]
    transaction = payload["transaction_learning"]
    assert isinstance(episodic, dict)
    assert isinstance(transaction, dict)
    for field in (
        "combat_hp_loss_value_weight",
        "combat_hp_loss_reference",
    ):
        episodic.pop(field)
    transaction.pop("lifecycle_smdp_horizon")


def _remove_v19_transaction_actor_fields(payload: dict[str, object]) -> None:
    transaction = payload["transaction_learning"]
    assert isinstance(transaction, dict)
    for field in (
        "lifecycle_advantage_policy_weight",
        "lifecycle_advantage_start_update",
        "lifecycle_advantage_temperature",
        "lifecycle_advantage_clip",
        "lifecycle_advantage_q_error_gate",
        "lifecycle_advantage_max_policy_lag",
        "lifecycle_advantage_max_log_probability_shift",
    ):
        transaction.pop(field)


def test_profiles_use_relational_recurrent_vtrace_v3_with_bounded_transaction_sidecar() -> None:
    default = load_training_config(profile="default")
    combat = load_training_config(profile="combat")
    preheat = load_training_config(profile="preheat")
    assert CONFIG_VERSION == "sts2-relational-curriculum-config-v20"
    assert default.model.architecture == "relational_candidate_v3"
    assert default.curriculum.reward_objective == "run"
    assert combat.curriculum.reward_objective == "combat"
    assert preheat.curriculum.mode == "native-revival-preheat"
    assert preheat.curriculum.revival_mechanism == ENGINE_REVIVAL_MECHANISM
    assert preheat.curriculum.revival_budget == -1
    assert preheat.optimization.discount == 1.0
    assert preheat.transaction_learning.enabled
    assert not default.episodic_learning.enabled
    assert not combat.episodic_learning.enabled
    assert preheat.episodic_learning.enabled
    assert preheat.episodic_learning.sample_sequences == 2
    assert preheat.episodic_learning.burn_in_steps == 32
    assert preheat.episodic_learning.learn_steps == 32
    assert preheat.episodic_learning.macro_sample_fraction == 0.5
    assert default.episodic_learning.macro_sample_fraction == 0.0
    assert combat.episodic_learning.macro_sample_fraction == 0.0
    assert preheat.episodic_learning.revival_value_weight == 0.02
    assert preheat.transaction_learning.pairwise_ranking_weight == 0.10
    assert default.failure_credit.unresolved_stall_quota == 1
    assert combat.failure_credit.unresolved_stall_quota == 1
    assert preheat.failure_credit.unresolved_stall_quota == 1
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


def test_failure_credit_quota_budget_accounts_for_all_learnable_strata() -> None:
    valid = FailureCreditConfig(
        sample_records=5,
        direct_witness_quota=1,
        multi_edge_cycle_quota=1,
        risk_sequence_quota=1,
        unresolved_stall_quota=1,
        completion_control_quota=1,
    )
    assert valid.sample_records == 5

    with pytest.raises(
        ValueError,
        match=r"evidence quotas cannot exceed sample_records",
    ):
        FailureCreditConfig(
            sample_records=4,
            direct_witness_quota=1,
            multi_edge_cycle_quota=1,
            risk_sequence_quota=1,
            unresolved_stall_quota=1,
            completion_control_quota=1,
        )


def test_v29_failure_credit_lineage_retires_transaction_v3_explicitly() -> None:
    overlay = (
        Path(__file__).parents[1] / "config" / "experiments" / "full_run_revival_v29_failure_credit_v4_model_init.toml"
    )
    config = load_archived_training_config(
        profile="preheat",
        config_path=overlay,
    )

    assert not config.transaction_learning.enabled
    assert config.failure_credit.mode == "learning"
    assert config.failure_credit.sample_records == 8
    assert config.failure_credit.unresolved_stall_quota == 1
    assert config.rollout.deterministic_probe_interval_episodes > 0
    assert config.episodic_learning.enabled
    assert config.curriculum.revival_budget == -1


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


def test_v26_fresh_policy_overlay_preserves_model_and_reward_contracts() -> None:
    experiment_root = Path(__file__).parents[1] / "config" / "experiments"
    predecessor = load_archived_training_config(
        profile="preheat",
        config_path=(experiment_root / "full_run_revival_v25_budget64_model_init.toml"),
    )
    restored_signal = load_archived_training_config(
        profile="preheat",
        config_path=(experiment_root / "full_run_revival_v26_fresh_policy_credit_model_init.toml"),
    )

    # This lineage isolates the signal-path repair: network/encoding, reward,
    # finite-revival curriculum, environment and diagnostics remain frozen.
    assert restored_signal.model == predecessor.model
    assert restored_signal.optimization == predecessor.optimization
    assert restored_signal.curriculum == predecessor.curriculum
    assert restored_signal.transaction_learning == predecessor.transaction_learning
    assert restored_signal.environment == predecessor.environment
    assert restored_signal.diagnostics == predecessor.diagnostics

    # The retired fresh-policy reservation key is projected away, so the
    # archived recipe's surviving episodic contract equals its predecessor.
    assert restored_signal.episodic_learning == predecessor.episodic_learning
    assert predecessor.rollout.queue_capacity == 64
    assert restored_signal.rollout == replace(
        predecessor.rollout,
        queue_capacity=24,
    )

    expected_lineage = predecessor.lineage_mapping()
    expected_lineage["rollout"] = {
        **expected_lineage["rollout"],
        "queue_capacity": 24,
    }
    assert restored_signal.lineage_mapping() == expected_lineage

    assert restored_signal.runtime.total_environment_steps == 100_000
    assert restored_signal.runtime.evaluation_steps == (
        0,
        25_000,
        50_000,
        75_000,
        90_000,
    )
    assert restored_signal.runtime.early_evaluation_steps == (5_000, 10_000, 20_000)
    assert restored_signal.runtime.final_audit_steps == (100_000,)
    assert restored_signal.runtime.final_audit_episodes == 32
    assert restored_signal.runtime.log_dir.endswith("full-run-revival-v26-fresh-policy-credit-model-init")


def test_v27_infinite_random_init_restores_preheat_exploration_with_v26_signal() -> None:
    experiment_root = Path(__file__).parents[1] / "config" / "experiments"
    preheat = load_training_config(profile="preheat")
    v26 = load_archived_training_config(
        profile="preheat",
        config_path=(experiment_root / "full_run_revival_v26_fresh_policy_credit_model_init.toml"),
    )
    fresh = load_archived_training_config(
        profile="preheat",
        config_path=(experiment_root / "full_run_revival_v27_infinite_random_init.toml"),
    )

    # A fresh random network must use the broad preheat optimizer/exploration
    # schedule rather than v26's conservative mature-policy hyperparameters.
    assert fresh.model == preheat.model == v26.model
    assert fresh.optimization == preheat.optimization
    assert fresh.optimization != v26.optimization
    assert fresh.curriculum == replace(
        preheat.curriculum,
        epsilon_decay_steps=150_000,
    )
    assert fresh.curriculum.mode == "native-revival-preheat"
    assert fresh.curriculum.revival_mechanism == ENGINE_REVIVAL_MECHANISM
    assert fresh.curriculum.revival_budget == -1
    assert fresh.curriculum.epsilon_start == pytest.approx(0.70)
    assert fresh.curriculum.epsilon_end == pytest.approx(0.10)

    # Preserve only the v26 sampling/backpressure signal changes; all factual
    # environment, transaction, diagnostics, and model contracts stay frozen.
    assert fresh.transaction_learning == preheat.transaction_learning
    assert fresh.environment == preheat.environment
    assert fresh.diagnostics == preheat.diagnostics
    assert fresh.episodic_learning == preheat.episodic_learning
    assert fresh.rollout == replace(preheat.rollout, queue_capacity=24)
    assert fresh.rollout.deterministic_probe_interval_episodes == 16

    expected_lineage = preheat.lineage_mapping()
    expected_lineage["curriculum"] = {
        **expected_lineage["curriculum"],
        "epsilon_decay_steps": 150_000,
    }
    expected_lineage["rollout"] = {
        **expected_lineage["rollout"],
        "queue_capacity": 24,
    }
    expected_lineage["runtime"] = {
        **expected_lineage["runtime"],
        "seed": 1_000_000,
    }
    assert fresh.lineage_mapping() == expected_lineage

    # Gates are disjoint, begin only after learning starts, and retain a large
    # never-reused final audit at the end of this full 250k preheat lineage.
    assert fresh.runtime.total_environment_steps == 250_000
    assert fresh.runtime.seed == 1_000_000
    assert fresh.runtime.rocm_sdpa_backend == "math"
    assert fresh.runtime.evaluation_steps == (
        25_000,
        50_000,
        75_000,
        100_000,
        150_000,
        200_000,
        225_000,
    )
    assert 0 not in fresh.runtime.evaluation_steps
    assert fresh.runtime.evaluation_episodes == 8
    assert fresh.runtime.early_evaluation_steps == (5_000, 10_000, 20_000)
    assert fresh.runtime.early_evaluation_episodes == 4
    assert fresh.runtime.final_audit_steps == (250_000,)
    assert fresh.runtime.final_audit_episodes == 32
    assert fresh.runtime.evaluation_liveness_guard_enabled
    assert fresh.runtime.log_dir.endswith("full-run-revival-v27-infinite-random-init")
    assert fresh.runtime.checkpoint_dir.endswith("full-run-revival-v27-infinite-random-init")


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
    with pytest.raises(ValueError, match=r"relational_candidate_v3"):
        ModelConfig(architecture="grounded_candidate_v1")
    with pytest.raises(ValueError, match=r"minimum_unrolls"):
        RolloutConfig(queue_capacity=4, minimum_unrolls=5)
    with pytest.raises(ValueError, match=r"one collector"):
        RolloutConfig(collector_workers=2)
    with pytest.raises(TypeError, match=r"integer"):
        RolloutConfig(unroll_length=True)  # type: ignore[arg-type]
    normalized = RolloutConfig(
        deterministic_probe_environment_steps=[512, 1_024],  # type: ignore[arg-type]
    )
    assert normalized.deterministic_probe_environment_steps == (512, 1_024)
    with pytest.raises(ValueError, match=r"strictly increasing"):
        RolloutConfig(deterministic_probe_environment_steps=(512, 512))
    with pytest.raises(ValueError, match=r"must be >= 1"):
        RolloutConfig(deterministic_probe_environment_steps=(0,))


def test_vtrace_and_diagnostics_bounds_are_strict() -> None:
    with pytest.raises(ValueError, match=r"vtrace_rho_clip"):
        OptimizationConfig(vtrace_rho_clip=0.0)
    with pytest.raises(ValueError, match=r"deadlock_repeat_threshold"):
        DiagnosticsConfig(deadlock_window=4, deadlock_repeat_threshold=5)
    with pytest.raises(ValueError, match=r"combat_net_progress_window"):
        DiagnosticsConfig(combat_net_progress_window=0)
    with pytest.raises(ValueError, match=r"noncombat_durable_progress_window"):
        DiagnosticsConfig(noncombat_durable_progress_window=0)
    with pytest.raises(TypeError, match=r"room_windows must be a table"):
        DiagnosticsConfig(combat_net_progress_room_windows=())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match=r"identifiers must be non-empty"):
        DiagnosticsConfig(combat_net_progress_room_windows={" ": 8})
    with pytest.raises(TypeError, match=r"must be an integer"):
        DiagnosticsConfig(combat_net_progress_encounter_windows={"BOSS": True})
    with pytest.raises(ValueError, match=r"must be >= 1"):
        DiagnosticsConfig(combat_net_progress_encounter_windows={"BOSS": 0})
    with pytest.raises(ValueError, match=r"duplicate normalized identifier"):
        DiagnosticsConfig(
            combat_net_progress_room_windows={"boss": 4, "BOSS": 8},
        )
    normalized = DiagnosticsConfig(
        combat_net_progress_room_windows={" test_subject_boss ": 128},
    )
    assert normalized.combat_net_progress_room_windows == {
        "TEST_SUBJECT_BOSS": 128,
    }
    with pytest.raises(ValueError, match=r"combat_min_net_hp_fraction"):
        DiagnosticsConfig(combat_min_net_hp_fraction=0.0)
    with pytest.raises(ValueError, match=r"strictly increasing"):
        RuntimeConfig(evaluation_steps=(0, 10, 10))
    with pytest.raises(ValueError, match=r"must be disjoint"):
        RuntimeConfig(
            evaluation_steps=(0, 10),
            early_evaluation_steps=(10,),
            early_evaluation_episodes=2,
        )
    RuntimeConfig(evaluation_liveness_guard_enabled=True)
    with pytest.raises(ValueError, match=r"requires early evaluation"):
        RuntimeConfig(
            evaluation_steps=(),
            evaluation_liveness_guard_enabled=True,
        )
    with pytest.raises(ValueError, match=r"enforcement_start_steps requires"):
        RuntimeConfig(evaluation_guard_enforcement_start_steps=5_000)
    bounded_guard = RuntimeConfig(
        evaluation_liveness_guard_enabled=True,
        evaluation_guard_enforcement_start_steps=10_000,
        evaluation_guard_failure_action="stop",
        evaluation_guard_max_rollbacks=0,
    )
    assert bounded_guard.evaluation_guard_enforcement_start_steps == 10_000
    with pytest.raises(ValueError, match=r"entropy_weight_end"):
        OptimizationConfig(entropy_weight=0.01, entropy_weight_end=0.02)


def test_environment_and_task_horizons_must_match() -> None:
    base = TrainingConfig()
    with pytest.raises(ValueError, match=r"full-run"):
        replace(base, curriculum=CurriculumConfig(reward_objective="combat"))


def test_native_revival_preheat_supports_a_headless_full_run_curriculum() -> None:
    with pytest.raises(ValueError, match=r"requires revival_mechanism"):
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
    assert identity["native_death_prevention_order"] == ("native hooks before training bailout")
    assert "fingerprint" not in identity
    assert "fingerprint_sha256" not in identity

    with pytest.raises(ValueError, match=r"engine revival mechanism"):
        CurriculumConfig(revival_mechanism=ENGINE_REVIVAL_MECHANISM)
    with pytest.raises(ValueError, match=r"engine-bailout-v1"):
        CurriculumConfig(
            mode="native-revival-preheat",
            reward_objective="combat",
            revival_mechanism="visible-relic-v0",  # type: ignore[arg-type]
            revival_budget=-1,
        )


def test_preheat_and_standard_discount_contracts_fail_closed() -> None:
    preheat = load_training_config(profile="preheat")
    with pytest.raises(ValueError, match=r"reward contract discount 1.0"):
        replace(
            preheat,
            optimization=replace(preheat.optimization, discount=0.997),
        )
    standard = load_training_config(profile="default")
    with pytest.raises(ValueError, match=r"reward contract discount 0.997"):
        replace(
            standard,
            optimization=replace(standard.optimization, discount=1.0),
        )


def test_v32_budget64_inherits_mature_schedules_without_changing_model_shape() -> None:
    experiment_root = Path(__file__).parents[1] / "config" / "experiments"
    v31 = load_archived_training_config(
        profile="preheat",
        config_path=(experiment_root / "full_run_revival_v31_failure_credit_capacity_model_init.toml"),
    )
    v32 = load_archived_training_config(
        profile="preheat",
        config_path=(experiment_root / "full_run_revival_v32_budget64_mature_model_init.toml"),
    )

    assert v32.model == v31.model
    assert v32.curriculum.revival_budget == 64
    assert v31.curriculum.revival_budget == -1
    assert v32.runtime.model_initialization_schedule_mode == "inherit"
    assert v32.failure_credit.sample_records == 4
    assert v32.failure_credit.direct_witness_quota == 0
    assert v32.failure_credit.multi_edge_cycle_quota == 0
    assert v32.failure_credit.risk_sequence_quota == 1
    assert v32.failure_credit.unresolved_stall_quota == 1
    assert v32.failure_credit.completion_control_quota == 1
    assert v32.runtime.seed == v31.runtime.seed == 5_000_000
    assert v32.runtime.evaluation_guard_min_liveness_episodes == 16
    assert v32.runtime.evaluation_guard_liveness_baseline_failures == 2
    assert v32.runtime.evaluation_guard_liveness_baseline_episodes == 16


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
    inherited_schedule = replace(
        base,
        runtime=replace(
            base.runtime,
            model_initialization_schedule_mode="inherit",
        ),
    )
    assert inherited_schedule.lineage_mapping() == base.lineage_mapping()
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
    with pytest.raises(ValueError, match=r"unsupported"):
        training_config_from_mapping(payload)


def test_v10_config_migration_is_model_initialization_only_and_opt_in() -> None:
    source = TrainingConfig().to_mapping()
    _remove_v19_transaction_actor_fields(source)
    source["version"] = "sts2-relational-curriculum-config-v10"
    _remove_v18_act_prefix_fields(source)
    _remove_v16_guard_field(source)
    _remove_v15_transaction_lifecycle_fields(source)
    source.pop("failure_credit")
    # A real v10 payload spelled out the retired imitation weights that
    # config v20 deleted; the reviewed migration must strip them.
    episodic = source["episodic_learning"]
    assert isinstance(episodic, dict)
    episodic["primary_policy_weight"] = 0.25
    episodic["revival_policy_weight"] = 0.05
    with pytest.raises(
        ValueError,
        match=r"unknown (episodic_learning|failure_credit) config keys|unsupported training config version",
    ):
        training_config_from_mapping(source)

    migrated = model_initialization_config_from_mapping(source)
    assert migrated.version == CONFIG_VERSION
    assert not migrated.failure_credit.shadow_enabled
    migrated_episodic = migrated.to_mapping()["episodic_learning"]
    assert isinstance(migrated_episodic, dict)
    assert "primary_policy_weight" not in migrated_episodic
    assert "revival_policy_weight" not in migrated_episodic
    assert "fresh_policy_sequences" not in migrated_episodic

    unsupported = dict(source)
    unsupported["version"] = "sts2-relational-curriculum-config-v9"
    with pytest.raises(ValueError, match=r"no reviewed config migration"):
        model_initialization_config_from_mapping(unsupported)


def test_v11_exact_resume_is_rejected_but_model_initialization_is_reviewed() -> None:
    source = TrainingConfig().to_mapping()
    _remove_v19_transaction_actor_fields(source)
    source["version"] = "sts2-relational-curriculum-config-v11"
    _remove_v18_act_prefix_fields(source)
    _remove_v16_guard_field(source)
    _remove_v15_transaction_lifecycle_fields(source)
    source.pop("failure_credit")

    with pytest.raises(ValueError, match=r"unsupported training config version"):
        training_config_from_mapping(source)

    migrated = model_initialization_config_from_mapping(source)
    assert migrated.version == CONFIG_VERSION
    assert not migrated.failure_credit.shadow_enabled


def test_v13_config_migration_is_model_init_only() -> None:
    source = TrainingConfig().to_mapping()
    _remove_v19_transaction_actor_fields(source)
    source["version"] = "sts2-relational-curriculum-config-v13"
    _remove_v18_act_prefix_fields(source)
    _remove_v16_guard_field(source)
    _remove_v15_transaction_lifecycle_fields(source)

    with pytest.raises(ValueError, match=r"unsupported training config version"):
        training_config_from_mapping(source)

    migrated = model_initialization_config_from_mapping(source)
    assert migrated.version == CONFIG_VERSION

    transaction_learning = source["transaction_learning"]
    assert isinstance(transaction_learning, dict)
    transaction_learning["lifecycle_entry_support_weight"] = 0.25
    with pytest.raises(ValueError, match=r"unexpectedly contains V15"):
        model_initialization_config_from_mapping(source)


def test_v14_transaction_lifecycle_migration_is_model_init_only() -> None:
    source = TrainingConfig().to_mapping()
    _remove_v19_transaction_actor_fields(source)
    source["version"] = "sts2-relational-curriculum-config-v14"
    _remove_v18_act_prefix_fields(source)
    _remove_v16_guard_field(source)
    _remove_v15_transaction_lifecycle_fields(source)

    with pytest.raises(ValueError, match=r"unsupported training config version"):
        training_config_from_mapping(source)

    migrated = model_initialization_config_from_mapping(source)
    assert migrated.version == CONFIG_VERSION
    assert migrated.transaction_learning.lifecycle_smdp_q_weight == 0.0

    unexpected = TrainingConfig().to_mapping()
    _remove_v19_transaction_actor_fields(unexpected)
    unexpected["version"] = "sts2-relational-curriculum-config-v14"
    _remove_v18_act_prefix_fields(unexpected)
    with pytest.raises(ValueError, match=r"unexpectedly contains V15"):
        model_initialization_config_from_mapping(unexpected)


def test_v15_guard_recovery_migration_is_model_init_only() -> None:
    source = TrainingConfig().to_mapping()
    _remove_v19_transaction_actor_fields(source)
    source["version"] = "sts2-relational-curriculum-config-v15"
    _remove_v18_act_prefix_fields(source)
    _remove_v16_guard_field(source)
    # A real v15 payload legitimately carried the retired entry-support
    # corridor keys; the reviewed migration must strip them.
    transaction_learning = source["transaction_learning"]
    assert isinstance(transaction_learning, dict)
    transaction_learning["lifecycle_entry_support_weight"] = 0.25
    transaction_learning["lifecycle_entry_support_probability_floor"] = 0.05

    with pytest.raises(
        ValueError,
        match=r"unknown transaction_learning config keys|unsupported training config version",
    ):
        training_config_from_mapping(source)

    migrated = model_initialization_config_from_mapping(source)
    assert migrated.version == CONFIG_VERSION
    assert migrated.runtime.evaluation_guard_enforcement_start_steps == 0
    migrated_transaction = migrated.to_mapping()["transaction_learning"]
    assert isinstance(migrated_transaction, dict)
    assert "lifecycle_entry_support_weight" not in migrated_transaction
    assert "lifecycle_entry_support_probability_floor" not in migrated_transaction

    unexpected = TrainingConfig().to_mapping()
    _remove_v19_transaction_actor_fields(unexpected)
    unexpected["version"] = "sts2-relational-curriculum-config-v15"
    _remove_v18_act_prefix_fields(unexpected)
    with pytest.raises(ValueError, match=r"V16 evaluation guard"):
        model_initialization_config_from_mapping(unexpected)


def test_v16_stability_migration_is_model_init_only() -> None:
    source = TrainingConfig().to_mapping()
    _remove_v19_transaction_actor_fields(source)
    source["version"] = "sts2-relational-curriculum-config-v16"
    _remove_v18_act_prefix_fields(source)

    with pytest.raises(ValueError, match=r"unsupported training config version"):
        training_config_from_mapping(source)

    migrated = model_initialization_config_from_mapping(source)
    assert migrated.version == CONFIG_VERSION
    assert (
        migrated.runtime.evaluation_guard_enforcement_start_steps
        == TrainingConfig().runtime.evaluation_guard_enforcement_start_steps
    )
    _assert_retired_v20_fields_are_stripped(migrated.to_mapping())


def test_v17_act_prefix_and_hp_loss_migration_is_model_init_only() -> None:
    source = TrainingConfig().to_mapping()
    _remove_v19_transaction_actor_fields(source)
    source["version"] = "sts2-relational-curriculum-config-v17"
    _remove_v18_act_prefix_fields(source)
    # A real v17 payload spelled out the retired imitation channels and
    # liveness policy-actor keys; the reviewed migration must strip them.
    _add_retired_v20_imitation_and_actor_fields(source)

    with pytest.raises(
        ValueError,
        match=r"unknown (episodic_learning|failure_credit) config keys|unsupported training config version",
    ):
        training_config_from_mapping(source)

    migrated = model_initialization_config_from_mapping(source)
    assert migrated.version == CONFIG_VERSION
    assert migrated.episodic_learning.combat_hp_loss_value_weight == 0.0
    assert migrated.transaction_learning.lifecycle_smdp_horizon == "transaction_exit"
    _assert_retired_v20_fields_are_stripped(migrated.to_mapping())

    unexpected = TrainingConfig().to_mapping()
    _remove_v19_transaction_actor_fields(unexpected)
    unexpected["version"] = "sts2-relational-curriculum-config-v17"
    with pytest.raises(ValueError, match=r"V18 episodic fields"):
        model_initialization_config_from_mapping(unexpected)


def test_v18_option_actor_migration_is_model_init_only() -> None:
    source = TrainingConfig().to_mapping()
    _remove_v19_transaction_actor_fields(source)
    source["version"] = "sts2-relational-curriculum-config-v18"

    with pytest.raises(ValueError, match=r"unsupported training config version"):
        training_config_from_mapping(source)

    migrated = model_initialization_config_from_mapping(source)
    assert migrated.version == CONFIG_VERSION
    assert migrated.transaction_learning.lifecycle_advantage_policy_weight == 0.0
    assert migrated.transaction_learning.lifecycle_advantage_start_update == 0
    assert migrated.transaction_learning.lifecycle_advantage_temperature == pytest.approx(0.25)
    assert migrated.transaction_learning.lifecycle_advantage_clip == pytest.approx(1.0)
    assert migrated.transaction_learning.lifecycle_advantage_q_error_gate == pytest.approx(0.25)
    assert migrated.transaction_learning.lifecycle_advantage_max_policy_lag == 128
    assert (
        migrated.transaction_learning.lifecycle_advantage_max_log_probability_shift
        == pytest.approx(1.0)
    )
    assert migrated.transaction_learning.lifecycle_smdp_horizon == (
        TrainingConfig().transaction_learning.lifecycle_smdp_horizon
    )

    unexpected = TrainingConfig().to_mapping()
    unexpected["version"] = "sts2-relational-curriculum-config-v18"
    with pytest.raises(ValueError, match=r"V19 transaction actor fields"):
        model_initialization_config_from_mapping(unexpected)


def test_v19_exploration_retirement_migration_is_model_init_only() -> None:
    source = TrainingConfig().to_mapping()
    source["version"] = "sts2-relational-curriculum-config-v19"
    # A real v19 payload carried the retired training-only exploration
    # contract; the reviewed migration must strip it rather than reinterpret
    # or preserve it.
    source["transaction_exploration"] = {
        "enabled": True,
        "operations": ["remove", "upgrade"],
        "entry_epsilon_floor": 0.50,
        "completion_guidance_probability": 0.95,
    }
    curriculum = source["curriculum"]
    assert isinstance(curriculum, dict)
    curriculum["selection_surface_epsilon_floor"] = 0.25
    # A real v19 payload also carried the retired entropy breaker, the
    # completion/selection-group CE weights and the entry-support corridor.
    _add_retired_v20_policy_fields(source)
    source_transaction = source["transaction_learning"]
    assert isinstance(source_transaction, dict)
    source_transaction["lifecycle_entry_support_weight"] = 0.05
    source_transaction["lifecycle_entry_support_probability_floor"] = 0.05

    # Exact resume must reject a v19 payload outright (the retired table is
    # an unknown section before the version even gets compared).
    with pytest.raises(
        ValueError,
        match=r"unknown training config sections|unsupported training config version",
    ):
        training_config_from_mapping(source)

    migrated = model_initialization_config_from_mapping(source)
    assert migrated.version == CONFIG_VERSION
    migrated_payload = migrated.to_mapping()
    assert "transaction_exploration" not in migrated_payload
    assert "selection_surface_epsilon_floor" not in migrated_payload["curriculum"]
    migrated_optimization = migrated_payload["optimization"]
    migrated_transaction = migrated_payload["transaction_learning"]
    assert isinstance(migrated_optimization, dict)
    assert isinstance(migrated_transaction, dict)
    assert "entropy_breaker" not in migrated_optimization
    assert "completion_policy_weight" not in migrated_transaction
    assert "macro_option_group_completion_weight" not in migrated_transaction
    assert "lifecycle_entry_support_weight" not in migrated_transaction
    assert "lifecycle_entry_support_probability_floor" not in migrated_transaction

    # A v20 payload that still contains the retired table/key is corrupt and
    # must fail closed on both parsing paths instead of being migrated.
    stale_table = TrainingConfig().to_mapping()
    stale_table["transaction_exploration"] = {"enabled": False}
    with pytest.raises(ValueError, match=r"unknown training config sections"):
        training_config_from_mapping(stale_table)
    with pytest.raises(ValueError, match=r"unknown training config sections"):
        model_initialization_config_from_mapping(stale_table)

    stale_floor = TrainingConfig().to_mapping()
    stale_curriculum = stale_floor["curriculum"]
    assert isinstance(stale_curriculum, dict)
    stale_curriculum["selection_surface_epsilon_floor"] = 0.25
    with pytest.raises(ValueError, match=r"unknown curriculum config keys"):
        training_config_from_mapping(stale_floor)
    with pytest.raises(ValueError, match=r"unknown curriculum config keys"):
        model_initialization_config_from_mapping(stale_floor)

    stale_breaker = TrainingConfig().to_mapping()
    stale_optimization = stale_breaker["optimization"]
    assert isinstance(stale_optimization, dict)
    stale_optimization["entropy_breaker"] = "policy-collapse-v2"
    with pytest.raises(ValueError, match=r"unknown optimization config keys"):
        training_config_from_mapping(stale_breaker)
    with pytest.raises(ValueError, match=r"unknown optimization config keys"):
        model_initialization_config_from_mapping(stale_breaker)

    stale_policy = TrainingConfig().to_mapping()
    stale_policy_transaction = stale_policy["transaction_learning"]
    assert isinstance(stale_policy_transaction, dict)
    stale_policy_transaction["completion_policy_weight"] = 0.15
    with pytest.raises(ValueError, match=r"unknown transaction_learning config keys"):
        training_config_from_mapping(stale_policy)
    with pytest.raises(ValueError, match=r"unknown transaction_learning config keys"):
        model_initialization_config_from_mapping(stale_policy)


def test_transaction_lifecycle_loss_contract_is_bounded_and_opt_in() -> None:
    enabled = TransactionLearningConfig(
        enabled=True,
        lifecycle_smdp_q_weight=0.10,
    )
    assert enabled.lifecycle_smdp_q_weight == pytest.approx(0.10)
    with pytest.raises(ValueError, match=r"require transaction_learning.enabled"):
        TransactionLearningConfig(
            enabled=False,
            lifecycle_smdp_q_weight=0.10,
        )


def test_unknown_replay_section_is_rejected() -> None:
    payload = TrainingConfig().to_mapping()
    payload["replay"] = {"capacity": 100_000}
    with pytest.raises(ValueError, match=r"unknown training config sections"):
        training_config_from_mapping(payload)


def test_episodic_learning_config_is_byte_bounded_and_fail_closed() -> None:
    config = EpisodicLearningConfig(enabled=True)
    assert config.replay_capacity_episodes == 64
    assert config.sample_sequences * config.learn_steps == 64
    with pytest.raises(TypeError, match=r"enabled must be a boolean"):
        EpisodicLearningConfig(enabled=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=r"per_episode_capacity_bytes"):
        EpisodicLearningConfig(
            replay_capacity_bytes=1024,
            per_episode_capacity_bytes=2048,
        )
    with pytest.raises(ValueError, match=r"task_value_weight"):
        EpisodicLearningConfig(task_value_weight=float("nan"))
    with pytest.raises(ValueError, match=r"combat_hp_loss_reference"):
        EpisodicLearningConfig(combat_hp_loss_reference=0.0)
    with pytest.raises(TypeError, match=r"macro_sample_fraction"):
        EpisodicLearningConfig(macro_sample_fraction=True)
    with pytest.raises(ValueError, match=r"macro_sample_fraction"):
        EpisodicLearningConfig(macro_sample_fraction=-0.01)
    with pytest.raises(ValueError, match=r"macro_sample_fraction"):
        EpisodicLearningConfig(macro_sample_fraction=1.01)
    with pytest.raises(ValueError, match=r"macro_sample_fraction"):
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

    changed_value_weight = replace(
        base,
        episodic_learning=replace(
            base.episodic_learning,
            revival_value_weight=0.02,
        ),
    )
    assert changed_value_weight.lineage_mapping() != base.lineage_mapping()
    value_round_trip = training_config_from_mapping(changed_value_weight.to_mapping())
    assert value_round_trip.episodic_learning.revival_value_weight == 0.02
    assert value_round_trip == changed_value_weight


def test_v20_imitation_and_liveness_actor_retirement_is_stripped_and_refused() -> None:
    # A real v18 payload carried every retired imitation and liveness-actor
    # key; the reviewed model-initialization migration strips them all.
    source = TrainingConfig().to_mapping()
    _remove_v19_transaction_actor_fields(source)
    source["version"] = "sts2-relational-curriculum-config-v18"
    _add_retired_v20_imitation_and_actor_fields(source)

    with pytest.raises(
        ValueError,
        match=r"unknown (episodic_learning|failure_credit) config keys|unsupported training config version",
    ):
        training_config_from_mapping(source)

    migrated = model_initialization_config_from_mapping(source)
    assert migrated.version == CONFIG_VERSION
    _assert_retired_v20_fields_are_stripped(migrated.to_mapping())

    # A v20 payload that still contains any retired key is corrupt and must
    # fail closed on both parsing paths instead of being migrated.
    for field, value in _RETIRED_V20_EPISODIC_FIELDS.items():
        stale = TrainingConfig().to_mapping()
        stale_episodic = stale["episodic_learning"]
        assert isinstance(stale_episodic, dict)
        stale_episodic[field] = value
        with pytest.raises(ValueError, match=r"unknown episodic_learning config keys"):
            training_config_from_mapping(stale)
        with pytest.raises(ValueError, match=r"unknown episodic_learning config keys"):
            model_initialization_config_from_mapping(stale)
    for field, value in _RETIRED_V20_FAILURE_FIELDS.items():
        stale = TrainingConfig().to_mapping()
        stale_failure = stale["failure_credit"]
        assert isinstance(stale_failure, dict)
        stale_failure[field] = value
        with pytest.raises(ValueError, match=r"unknown failure_credit config keys"):
            training_config_from_mapping(stale)
        with pytest.raises(ValueError, match=r"unknown failure_credit config keys"):
            model_initialization_config_from_mapping(stale)
