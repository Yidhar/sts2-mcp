from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

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
    episodic = source["episodic_learning"]
    assert isinstance(episodic, dict)
    del episodic["macro_sample_fraction"]

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
    del episodic["policy_gradient_max_lag"]
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
