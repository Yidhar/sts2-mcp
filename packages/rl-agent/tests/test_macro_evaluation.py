from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import sts2_rl.macro_evaluation as macro_evaluation_module
from sts2_rl.encoding import GroundedEncodingConfig, GroundedObservationEncoder
from sts2_rl.macro_evaluation import (
    MACRO_SENSITIVITY_SCHEMA,
    MACRO_TELEMETRY_SCHEMA,
    _diagnostic_model_initialization_config,
    evaluate_macro_sensitivity,
    fixed_macro_sensitivity_cases,
    main,
    read_macro_journal,
    summarize_macro_records,
)
from sts2_rl.models import GroundedCandidateConfig, RecurrentCandidateModel
from sts2_rl.training.config import CONFIG_VERSION, load_training_config


def _remove_v15_transaction_lifecycle_fields(payload: dict[str, object]) -> None:
    transaction_learning = payload["transaction_learning"]
    assert isinstance(transaction_learning, dict)
    for key in (
        "lifecycle_entry_support_weight",
        "lifecycle_entry_support_probability_floor",
        "lifecycle_smdp_q_weight",
    ):
        del transaction_learning[key]


def _remove_v16_guard_field(payload: dict[str, object]) -> None:
    runtime = payload["runtime"]
    assert isinstance(runtime, dict)
    del runtime["evaluation_guard_enforcement_start_steps"]


def _remove_v17_stability_fields(payload: dict[str, object]) -> None:
    failure = payload["failure_credit"]
    episodic = payload["episodic_learning"]
    assert isinstance(failure, dict)
    assert isinstance(episodic, dict)
    del failure["liveness_risk_actor_min_selected_probability"]
    del episodic["success_imitation_exempt_surfaces"]


def _decision(
    *,
    surface: str,
    episode: str,
    candidate_count: int,
    probabilities: list[float],
    selected_index: int = 0,
) -> dict[str, object]:
    kind = {
        "card_reward": "card_reward",
        "map": "map",
        "rest": "rest_site",
        "shop": "shop",
    }[surface]
    actions = [
        {
            "model_action_kind": kind,
            "model_action_variant": f"candidate-{index}",
        }
        for index in range(candidate_count)
    ]
    return {
        "event": "decision",
        "record_kind": "summary",
        "episode_id": episode,
        "reset_seed": 101,
        "semantic_candidate_count": candidate_count,
        "legal_action_count": candidate_count,
        "legal_action_kinds": {kind: candidate_count},
        "selected_action": actions[selected_index],
        "policy_topk": [
            {
                "candidate_index": index,
                "probability": probability,
                "action": actions[index],
            }
            for index, probability in enumerate(probabilities)
        ],
    }


def test_macro_surface_telemetry_counts_exposure_and_bounds_partial_entropy() -> None:
    records = [
        _decision(
            surface="card_reward",
            episode="reward-episode",
            candidate_count=2,
            probabilities=[0.75, 0.25],
        ),
        _decision(
            surface="map",
            episode="map-episode",
            candidate_count=3,
            probabilities=[0.40, 0.35],
        ),
        _decision(
            surface="rest",
            episode="rest-episode",
            candidate_count=1,
            probabilities=[1.0],
        ),
        _decision(
            surface="shop",
            episode="shop-episode",
            candidate_count=4,
            probabilities=[0.30, 0.27],
            selected_index=1,
        ),
        {
            **_decision(
                surface="card_reward",
                episode="reward-episode",
                candidate_count=2,
                probabilities=[0.75, 0.25],
            ),
            "record_kind": "rich_snapshot",
        },
        {"event": "evaluation_attempt_completed", "episode_id": "boundary"},
    ]

    summary = summarize_macro_records(records)

    assert summary["schema_version"] == MACRO_TELEMETRY_SCHEMA
    assert summary["diagnostic_only"] is True
    assert summary["training_samples_emitted"] == 0
    assert summary["total_macro_decisions"] == 4

    reward = summary["surfaces"]["card_reward"]
    assert reward["decision_count"] == 1
    assert reward["multi_candidate_decision_count"] == 1
    assert reward["exact_entropy_decision_count"] == 1
    expected_reward_entropy = -(0.75 * math.log(0.75) + 0.25 * math.log(0.25)) / math.log(2)
    assert reward["mean_exact_normalized_entropy"] == pytest.approx(expected_reward_entropy)
    assert reward["mean_top_probability"] == pytest.approx(0.75)
    assert reward["candidate_count_histogram"] == {"2": 1}

    route = summary["surfaces"]["map"]
    assert route["decision_count"] == 1
    assert route["exact_entropy_decision_count"] == 0
    assert route["mean_recorded_candidate_coverage"] == pytest.approx(2 / 3)
    assert route["mean_normalized_entropy_lower_bound"] <= route["mean_normalized_entropy_upper_bound"]

    rest = summary["surfaces"]["rest"]
    assert rest["multi_candidate_decision_count"] == 0
    assert rest["forced_singleton_decision_count"] == 1
    assert rest["policy_probability_decision_count"] == 0
    assert rest["mean_exact_normalized_entropy"] is None

    shop = summary["surfaces"]["shop"]
    assert shop["selected_action_kind_counts"] == {"shop": 1}
    assert shop["mean_recorded_probability_mass"] == pytest.approx(0.57)


def test_macro_surface_telemetry_rejects_inconsistent_topk() -> None:
    record = _decision(
        surface="map",
        episode="bad",
        candidate_count=3,
        probabilities=[0.2, 0.1],
    )

    with pytest.raises(ValueError, match="residual is inconsistent"):
        summarize_macro_records([record])


def test_macro_surface_telemetry_renormalizes_reviewed_float32_mass_drift() -> None:
    # Exact values from the v34 50k held-out journal that previously terminated
    # a healthy learner.  This is a complete four-candidate distribution whose
    # raw float32 hierarchical mass is 15.04 ppm above one.
    probabilities = [
        0.9979878664016724,
        0.001463199034333229,
        0.0005639757728204131,
        0.0,
    ]
    record = _decision(
        surface="card_reward",
        episode="v34-float32-drift",
        candidate_count=4,
        probabilities=probabilities,
    )

    summary = summarize_macro_records([record])

    contract = summary["probability_mass_contract"]
    assert summary["valid"] is True
    assert contract["checked_decision_count"] == 1
    assert contract["accepted_decision_count"] == 1
    assert contract["numeric_renormalized_decision_count"] == 1
    assert contract["invalid_decision_count"] == 0
    assert contract["maximum_numeric_mass_error"] == pytest.approx(sum(probabilities) - 1.0)
    reward = summary["surfaces"]["card_reward"]
    assert reward["policy_probability_decision_count"] == 1
    assert reward["mean_recorded_probability_mass"] == pytest.approx(sum(probabilities))
    assert reward["mean_top_probability"] == pytest.approx(probabilities[0] / sum(probabilities))
    assert math.isfinite(reward["mean_exact_normalized_entropy"])


@pytest.mark.parametrize(
    ("mutator", "reason"),
    [
        (
            lambda record: record["policy_topk"].__setitem__(
                1,
                {**record["policy_topk"][1], "candidate_index": 0},
            ),
            "duplicate_candidate_index",
        ),
        (
            lambda record: record["policy_topk"][0].__setitem__(
                "candidate_index",
                3,
            ),
            "candidate_index_out_of_range",
        ),
        (
            lambda record: record["policy_topk"][0].__setitem__(
                "probability",
                0.8,
            ),
            "probability_mass_above_one",
        ),
    ],
)
def test_macro_surface_telemetry_non_strict_mode_audits_structural_corruption(
    mutator: object,
    reason: str,
) -> None:
    record = _decision(
        surface="card_reward",
        episode="invalid-policy-contract",
        candidate_count=2,
        probabilities=[0.6, 0.4],
    )
    assert callable(mutator)
    mutator(record)

    with pytest.raises(ValueError):
        summarize_macro_records([record])

    summary = summarize_macro_records(
        [record],
        strict_probability_contract=False,
    )
    contract = summary["probability_mass_contract"]
    assert summary["valid"] is False
    assert contract["checked_decision_count"] == 1
    assert contract["accepted_decision_count"] == 0
    assert contract["invalid_decision_count"] == 1
    assert contract["invalid_reason_counts"] == {reason: 1}
    assert summary["surfaces"]["card_reward"]["decision_count"] == 1
    assert summary["surfaces"]["card_reward"]["policy_probability_decision_count"] == 0


def _model_config() -> GroundedCandidateConfig:
    return GroundedCandidateConfig(
        token_feature_dim=224,
        d_model=32,
        n_heads=4,
        ffn_dim=64,
        world_layers=1,
        latent_slots=4,
        latent_layers=1,
        local_layers=1,
        candidate_layers=1,
        recurrent_hidden_dim=64,
        dropout=0.0,
        domain_count=8,
        type_vocab_size=256,
        role_vocab_size=128,
        owner_vocab_size=256,
        entity_vocab_size=1024,
        zone_vocab_size=32,
        order_vocab_size=64,
    )


def test_fixed_macro_sensitivity_is_label_free_world_only_and_restores_mode() -> None:
    torch.manual_seed(17)
    config = _model_config()
    model = RecurrentCandidateModel(
        config,
        enable_transaction_heads=True,
    )
    model.train()
    encoder = GroundedObservationEncoder(
        GroundedEncodingConfig.from_model_config(
            config,
            max_world_tokens=512,
            max_candidates=8,
            max_candidate_local_tokens=32,
        )
    )

    result = evaluate_macro_sensitivity(model, encoder)

    assert model.training is True
    assert result["schema_version"] == MACRO_SENSITIVITY_SCHEMA
    assert result["diagnostic_only"] is True
    assert result["training_samples_emitted"] == 0
    assert result["expected_action_labels"] is False
    assert result["case_count"] == len(fixed_macro_sensitivity_cases()) == 7
    assert set(result["surfaces"]) == {"card_reward", "map", "rest", "shop"}
    assert all(result["surfaces"][surface]["case_count"] >= 1 for surface in result["surfaces"])

    for case in result["cases"]:
        assert case["encoded_world_changed"] is True
        assert case["encoded_candidate_tensors_equal"] is True
        assert case["candidate_count"] >= 2
        assert len(case["reference_probabilities"]) == case["candidate_count"]
        assert len(case["comparison_probabilities"]) == case["candidate_count"]
        assert sum(case["reference_probabilities"]) == pytest.approx(1.0)
        assert sum(case["comparison_probabilities"]) == pytest.approx(1.0)
        assert math.isfinite(case["policy_total_variation"])
        assert math.isfinite(case["policy_max_abs_probability_delta"])
        assert "expected_action" not in case
        assert "preferred_direction" not in case


def test_macro_journal_cli_writes_standalone_diagnostic_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STS2_ARTIFACT_ROOT", str(tmp_path))
    journal = tmp_path / "trajectory.jsonl"
    journal.write_text(
        json.dumps(
            _decision(
                surface="shop",
                episode="held-out-shop",
                candidate_count=2,
                probabilities=[0.6, 0.4],
            )
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "macro.json"

    assert main(["--journal", str(journal), "--output", str(output)]) == 0

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["diagnostic_only"] is True
    assert payload["training_samples_emitted"] == 0
    assert payload["held_out_journal"]["surfaces"]["shop"]["decision_count"] == 1
    assert read_macro_journal(journal)["total_macro_decisions"] == 1


def test_checkpoint_probe_has_one_reviewed_v6_config_interpretation() -> None:
    source = load_training_config(profile="preheat").to_mapping()
    source["version"] = "sts2-relational-curriculum-config-v6"
    _remove_v17_stability_fields(source)
    _remove_v15_transaction_lifecycle_fields(source)
    _remove_v16_guard_field(source)
    del source["failure_credit"]
    del source["transaction_exploration"]
    episodic = source["episodic_learning"]
    rollout = source["rollout"]
    assert isinstance(episodic, dict)
    assert isinstance(rollout, dict)
    del episodic["macro_sample_fraction"]
    del episodic["fresh_policy_sequences"]
    del rollout["deterministic_probe_environment_steps"]

    migrated = _diagnostic_model_initialization_config(source)

    assert migrated.version == CONFIG_VERSION
    assert migrated.episodic_learning.macro_sample_fraction == 0.0
    assert migrated.model == load_training_config(profile="preheat").model

    unsupported = dict(source)
    unsupported["version"] = "unsupported"
    with pytest.raises(ValueError, match="no reviewed config migration"):
        _diagnostic_model_initialization_config(unsupported)

    unexpected_field = dict(source)
    unexpected_episodic = dict(episodic)
    unexpected_episodic["macro_sample_fraction"] = 0.5
    unexpected_field["episodic_learning"] = unexpected_episodic
    with pytest.raises(ValueError, match="unexpectedly contains"):
        _diagnostic_model_initialization_config(unexpected_field)


def test_checkpoint_probe_migrates_v7_runtime_defaults_only() -> None:
    active = load_training_config(profile="preheat")
    source = active.to_mapping()
    source["version"] = "sts2-relational-curriculum-config-v7"
    _remove_v17_stability_fields(source)
    _remove_v15_transaction_lifecycle_fields(source)
    _remove_v16_guard_field(source)
    del source["failure_credit"]
    del source["transaction_exploration"]
    optimization = source["optimization"]
    rollout = source["rollout"]
    episodic = source["episodic_learning"]
    runtime = source["runtime"]
    assert isinstance(optimization, dict)
    assert isinstance(rollout, dict)
    assert isinstance(episodic, dict)
    assert isinstance(runtime, dict)
    for key in ("entropy_weight_end", "entropy_decay_updates"):
        del optimization[key]
    del rollout["deterministic_probe_interval_episodes"]
    del rollout["deterministic_probe_environment_steps"]
    del episodic["policy_gradient_max_lag"]
    del episodic["fresh_policy_sequences"]
    for key in (
        "early_evaluation_steps",
        "early_evaluation_episodes",
        "final_audit_steps",
        "final_audit_episodes",
        "evaluation_liveness_guard_enabled",
        "evaluation_guard_min_confirm_ready",
        "evaluation_guard_min_multi_action_end_turn",
        "evaluation_guard_max_confirm_failure_rate",
        "evaluation_guard_max_multi_action_end_turn_rate",
        "evaluation_guard_max_selection_cycle_episode_rate",
    ):
        del runtime[key]

    migrated = _diagnostic_model_initialization_config(source)

    assert migrated.version == CONFIG_VERSION
    assert migrated.model == active.model
    assert migrated.optimization.entropy_weight_end == 0.01
    assert migrated.optimization.entropy_decay_updates == 2_000
    assert migrated.rollout.deterministic_probe_interval_episodes == 0
    assert migrated.episodic_learning.policy_gradient_max_lag == 128
    assert migrated.runtime.early_evaluation_steps == ()
    assert migrated.runtime.final_audit_steps == ()
    assert not migrated.runtime.evaluation_liveness_guard_enabled


def test_checkpoint_probe_migrates_v8_probe_schedule_default_only() -> None:
    active = load_training_config(profile="preheat")
    source = active.to_mapping()
    source["version"] = "sts2-relational-curriculum-config-v8"
    _remove_v17_stability_fields(source)
    _remove_v15_transaction_lifecycle_fields(source)
    _remove_v16_guard_field(source)
    del source["failure_credit"]
    del source["transaction_exploration"]
    rollout = source["rollout"]
    episodic = source["episodic_learning"]
    assert isinstance(rollout, dict)
    assert isinstance(episodic, dict)
    # Reconstruct the historical v8 value rather than leaking the current
    # preheat profile's recurring-probe policy into a legacy fixture.
    rollout["deterministic_probe_interval_episodes"] = 0
    del rollout["deterministic_probe_environment_steps"]
    del episodic["fresh_policy_sequences"]

    migrated = _diagnostic_model_initialization_config(source)

    assert migrated.version == CONFIG_VERSION
    assert migrated.model == active.model
    assert migrated.rollout.deterministic_probe_environment_steps == ()
    assert migrated.rollout.deterministic_probe_interval_episodes == 0

    unexpected = active.to_mapping()
    unexpected["version"] = "sts2-relational-curriculum-config-v8"
    with pytest.raises(ValueError, match="unexpectedly contains"):
        _diagnostic_model_initialization_config(unexpected)


def test_checkpoint_probe_migrates_v10_fresh_sampling_to_disabled() -> None:
    active = load_training_config(profile="preheat")
    source = active.to_mapping()
    source["version"] = "sts2-relational-curriculum-config-v10"
    _remove_v17_stability_fields(source)
    _remove_v15_transaction_lifecycle_fields(source)
    _remove_v16_guard_field(source)
    del source["failure_credit"]
    del source["transaction_exploration"]
    episodic = source["episodic_learning"]
    assert isinstance(episodic, dict)
    del episodic["fresh_policy_sequences"]

    migrated = _diagnostic_model_initialization_config(source)

    assert migrated.version == CONFIG_VERSION
    assert migrated.model == active.model
    assert migrated.episodic_learning.fresh_policy_sequences == 0

    unexpected = dict(source)
    unexpected["episodic_learning"] = {
        **episodic,
        "fresh_policy_sequences": 1,
    }
    with pytest.raises(ValueError, match="unexpectedly contains"):
        _diagnostic_model_initialization_config(unexpected)


def test_checkpoint_probe_reconstructs_the_enabled_liveness_head(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failure-credit checkpoint must be loaded with its complete model ABI."""

    config = load_training_config(
        profile="preheat",
        config_path=(
            Path(__file__).parents[1]
            / "config"
            / "experiments"
            / "full_run_revival_v32_budget64_mature_model_init.toml"
        ),
    )
    assert config.failure_credit.learning_enabled is True
    source = RecurrentCandidateModel(
        config.model.to_model_config(),
        enable_transaction_heads=config.transaction_learning.enabled,
        enable_liveness_head=True,
    )
    torch.save(source.state_dict(), tmp_path / "network.pt")

    validated = SimpleNamespace(
        root=tmp_path,
        metadata={
            "training_config": config.to_mapping(),
            "training_state": {"environment_steps": 17},
        },
        manifest={"checkpoint_id": "test-liveness-checkpoint"},
    )
    monkeypatch.setattr(
        macro_evaluation_module,
        "validate_resume_checkpoint",
        lambda _root: validated,
    )
    monkeypatch.setattr(
        macro_evaluation_module,
        "preflight_model_initialization",
        lambda _root, *, config, prevalidated=None: validated,
    )
    monkeypatch.setattr(
        macro_evaluation_module,
        "evaluate_macro_sensitivity",
        lambda model, _encoder: {
            "liveness_head_enabled": model.liveness_head_enabled,
        },
    )

    result = macro_evaluation_module.evaluate_checkpoint_macro_sensitivity(tmp_path)

    assert result["liveness_head_enabled"] is True
    assert result["checkpoint_id"] == "test-liveness-checkpoint"
    assert result["training_state"] == {"environment_steps": 17}
