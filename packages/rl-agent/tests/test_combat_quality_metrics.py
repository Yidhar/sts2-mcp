from __future__ import annotations

import pytest

from muzero.combat_quality.metrics import (
    COMBAT_QUALITY_CARD_BLOCK_SEARCH_SUFFIXES,
    COMBAT_QUALITY_GUARD_SEARCH_SUFFIXES,
    COMBAT_QUALITY_TACTICAL_SEARCH_SUFFIXES,
    boss_card_block_waste_metrics,
    card_block_waste_metrics,
    combat_tactical_issue_metrics,
)


def _sample_search_values() -> dict[str, list[float]]:
    return {
        "combat_quality_card_block_waste_count": [0, 2, 1],
        "combat_quality_card_pure_block_count": [1, 1, 0],
        "combat_quality_card_no_damage_pressure_count": [0, 1, 1],
        "combat_quality_card_block_waste_bias_count": [0, 2, 1],
        "combat_quality_card_block_waste_bias_min": [0.0, -0.35, -0.2],
        "combat_quality_card_block_waste_hard_bias_applied": [0, 1, 1],
        "combat_quality_card_block_waste_progress_alternative": [0, 1, 1],
        "combat_quality_card_block_waste_progress_bonus_count": [0, 3, 2],
        "combat_quality_card_block_waste_progress_bonus_max": [0.0, 0.2, 0.2],
        "combat_quality_card_block_waste_selected": [0, 1, 0],
        "combat_quality_card_block_waste_with_progress_selected": [0, 1, 0],
        "combat_quality_card_pure_block_selected": [0, 1, 1],
        "combat_quality_bad_pure_block_selected": [0, 1, 0],
        "combat_quality_pure_block_survival_justified_selected": [0, 0, 1],
        "combat_quality_pure_block_progress_alternative_selected": [0, 1, 1],
        "combat_quality_pure_block_no_alternative_selected": [0, 0, 0],
        "combat_quality_pure_block_low_value_pressure_selected": [0, 0, 1],
        "combat_quality_insufficient_block_selected": [1, 0, 0],
        "combat_quality_card_no_damage_pressure_selected": [0, 0, 1],
        "combat_quality_card_no_damage_pressure_with_progress_selected": [0, 0, 1],
    }


def _sample_tactical_search_values() -> dict[str, list[float]]:
    return {
        "combat_quality_wasteful_end_turn_available": [0, 1, 0],
        "combat_quality_wasteful_end_turn_selected": [0, 1, 0],
        "combat_quality_strategic_defer_available": [0, 0, 1],
        "combat_quality_strategic_defer_end_turn_selected": [0, 0, 1],
        "combat_quality_x_cost_available_count": [0, 2, 1],
        "combat_quality_x_cost_selected": [0, 1, 1],
        "combat_quality_x_cost_selected_energy": [0, 0, 2],
        "combat_quality_zero_energy_x_cost_count": [0, 1, 0],
        "combat_quality_zero_energy_x_cost_selected": [0, 1, 0],
        "combat_quality_x_cost_zero_bad_selected": [0, 1, 0],
        "combat_quality_x_cost_zero_guard_available": [0, 1, 0],
        "combat_quality_x_cost_zero_guard_applied": [0, 1, 0],
        "combat_quality_hp_cost_self_lethal_selected": [0, 0, 0],
        "combat_quality_hp_cost_low_margin_selected": [0, 1, 0],
        "combat_quality_hp_cost_unblockable_value": [0.0, 0.5, 1.0],
        "combat_quality_refund_no_followup_available": [0, 1, 1],
        "combat_quality_refund_no_followup_selected": [0, 1, 1],
        "combat_quality_refund_no_followup_with_progress_selected": [0, 1, 0],
        "combat_quality_refund_no_followup_progress_alternative_selected": [0, 1, 0],
        "combat_quality_refund_no_followup_no_alternative_selected": [0, 0, 1],
        "combat_quality_refund_no_followup_progress_alternative_count": [0, 2, 0],
        "combat_quality_refund_no_followup_guard_available": [0, 1, 0],
        "combat_quality_refund_no_followup_guard_applied": [0, 1, 0],
        "combat_quality_refund_no_followup_guard_override": [0, 1, 0],
        "combat_quality_refund_no_followup_guard_no_alternative": [0, 0, 0],
        "combat_quality_refund_no_followup_guard_lethal_exemption": [0, 0, 0],
        "combat_quality_refund_no_followup_guard_candidate_count": [0, 2, 0],
        "combat_quality_refund_no_followup_guard_lethal_candidate": [0, 0, 0],
        "combat_quality_strategic_skip_candidate_count": [0, 2, 1],
        "combat_quality_strategic_skip_selected": [0, 0, 1],
        "combat_quality_potion_available_count": [0, 1, 1],
        "combat_quality_potion_urgent_count": [0, 1, 0],
        "combat_quality_potion_low_urgency_count": [0, 0, 1],
        "combat_quality_potion_save_recommended_count": [0, 0, 1],
        "combat_quality_potion_selected": [0, 1, 1],
        "combat_quality_potion_high_urgency_selected": [0, 1, 0],
        "combat_quality_potion_low_urgency_selected": [0, 0, 1],
        "combat_quality_potion_save_recommended_selected": [0, 0, 1],
        "combat_quality_potion_use_quality_selected": [0.0, 0.8, 0.2],
        "combat_quality_potion_waste_risk_selected": [0.0, 0.1, 0.7],
        "combat_quality_potion_bad_guard_available": [0, 1, 0],
        "combat_quality_potion_bad_guard_applied": [0, 1, 0],
    }


def test_boss_card_block_waste_metrics_aggregate_expected_rates() -> None:
    metrics = boss_card_block_waste_metrics(_sample_search_values())

    assert metrics["boss_combat/card_block_waste_available_mean"] == 1.0
    assert metrics["boss_combat/card_pure_block_available_mean"] == 2 / 3
    assert metrics["boss_combat/card_no_damage_pressure_available_mean"] == 2 / 3
    assert metrics["boss_combat/card_block_waste_bias_count_mean"] == 1.0
    assert metrics["boss_combat/card_block_waste_bias_applied_rate"] == 2 / 3
    assert metrics["boss_combat/card_block_waste_bias_min"] == -0.35
    assert metrics["boss_combat/card_block_waste_hard_bias_applied_rate"] == 2 / 3
    assert metrics["boss_combat/card_block_waste_progress_alternative_rate"] == 2 / 3
    assert metrics["boss_combat/card_block_waste_progress_bonus_count_mean"] == 5 / 3
    assert metrics["boss_combat/card_block_waste_progress_bonus_max"] == pytest.approx(0.4 / 3)
    assert metrics["boss_combat/card_block_waste_selected_rate"] == 1 / 3
    assert metrics["boss_combat/card_block_waste_with_progress_selected_rate"] == 1 / 3
    assert metrics["boss_combat/card_pure_block_selected_rate"] == 2 / 3
    assert metrics["boss_combat/bad_pure_block_selected_rate"] == 1 / 3
    assert metrics["boss_combat/pure_block_survival_justified_selected_rate"] == 1 / 3
    assert metrics["boss_combat/pure_block_progress_alternative_selected_rate"] == 2 / 3
    assert metrics["boss_combat/pure_block_no_alternative_selected_rate"] == 0.0
    assert metrics["boss_combat/pure_block_low_value_pressure_selected_rate"] == 1 / 3
    assert metrics["boss_combat/insufficient_block_selected_rate"] == 1 / 3
    assert metrics["boss_combat/card_no_damage_pressure_selected_rate"] == 1 / 3
    assert metrics["boss_combat/card_no_damage_pressure_with_progress_selected_rate"] == 1 / 3


def test_card_block_waste_metrics_emit_global_combat_quality_prefix_by_default() -> None:
    metrics = card_block_waste_metrics(_sample_search_values())

    assert metrics["combat_quality/card_block_waste_available_mean"] == 1.0
    assert metrics["combat_quality/card_pure_block_available_mean"] == 2 / 3
    assert metrics["combat_quality/card_no_damage_pressure_available_mean"] == 2 / 3
    assert metrics["combat_quality/card_block_waste_selected_rate"] == 1 / 3
    assert metrics["combat_quality/card_block_waste_with_progress_selected_rate"] == 1 / 3
    assert metrics["combat_quality/card_pure_block_selected_rate"] == 2 / 3
    assert metrics["combat_quality/bad_pure_block_selected_rate"] == 1 / 3
    assert metrics["combat_quality/pure_block_survival_justified_selected_rate"] == 1 / 3
    assert metrics["combat_quality/pure_block_progress_alternative_selected_rate"] == 2 / 3
    assert metrics["combat_quality/pure_block_no_alternative_selected_rate"] == 0.0
    assert metrics["combat_quality/pure_block_low_value_pressure_selected_rate"] == 1 / 3
    assert metrics["combat_quality/insufficient_block_selected_rate"] == 1 / 3
    assert metrics["combat_quality/card_no_damage_pressure_selected_rate"] == 1 / 3
    assert metrics["combat_quality/card_no_damage_pressure_with_progress_selected_rate"] == 1 / 3
    assert all(key.startswith("combat_quality/") for key in metrics)


def test_card_block_waste_metrics_support_custom_tier_prefix() -> None:
    metrics = card_block_waste_metrics(_sample_search_values(), prefix="normal_combat")

    assert metrics["normal_combat/card_block_waste_available_mean"] == 1.0
    assert metrics["normal_combat/card_block_waste_bias_min"] == -0.35
    assert metrics["normal_combat/card_block_waste_selected_rate"] == 1 / 3
    assert all(key.startswith("normal_combat/") for key in metrics)


def test_combat_tactical_issue_metrics_emit_global_rates() -> None:
    metrics = combat_tactical_issue_metrics(_sample_tactical_search_values())

    assert metrics["combat_quality/wasteful_end_turn_selected_rate"] == 1 / 3
    assert metrics["combat_quality/strategic_defer_end_turn_selected_rate"] == 1 / 3
    assert metrics["combat_quality/x_cost_available_count_mean"] == 1.0
    assert metrics["combat_quality/x_cost_selected_rate"] == 2 / 3
    assert metrics["combat_quality/x_cost_selected_energy_mean"] == 1.0
    assert metrics["combat_quality/zero_energy_x_cost_selected_rate"] == 1 / 3
    assert metrics["combat_quality/x_cost_zero_bad_selected_rate_p0"] == 1 / 3
    assert metrics["combat_quality/x_cost_zero_guard_applied_rate"] == 1 / 3
    assert metrics["combat_quality/hp_cost_self_lethal_selected_rate"] == 0.0
    assert metrics["combat_quality/hp_cost_low_margin_selected_rate"] == 1 / 3
    assert metrics["combat_quality/hp_cost_unblockable_value_mean"] == 0.5
    assert metrics["combat_quality/refund_no_followup_selected_rate"] == 2 / 3
    assert metrics["combat_quality/refund_no_followup_with_progress_selected_rate"] == 1 / 3
    assert metrics["combat_quality/refund_no_followup_progress_alternative_selected_rate"] == 1 / 3
    assert metrics["combat_quality/refund_no_followup_no_alternative_selected_rate"] == 1 / 3
    assert metrics["combat_quality/refund_no_followup_progress_alternative_count_mean"] == 2 / 3
    assert metrics["combat_quality/refund_no_followup_guard_applied_rate"] == 1 / 3
    assert metrics["combat_quality/refund_no_followup_guard_candidate_count_mean"] == 2 / 3
    assert metrics["combat_quality/strategic_skip_selected_rate"] == 1 / 3
    assert metrics["combat_quality/potion_urgent_available_mean"] == 1 / 3
    assert metrics["combat_quality/potion_low_urgency_selected_rate"] == 1 / 3
    assert metrics["combat_quality/potion_save_recommended_selected_rate"] == 1 / 3
    assert metrics["combat_quality/potion_use_quality_selected_mean"] == pytest.approx(0.5)
    assert metrics["combat_quality/potion_waste_risk_selected_mean"] == pytest.approx(0.4)
    assert metrics["combat_quality/potion_bad_guard_applied_rate"] == 1 / 3
    assert all(key.startswith("combat_quality/") for key in metrics)


def test_combat_tactical_issue_metrics_support_custom_tier_prefix() -> None:
    metrics = combat_tactical_issue_metrics(_sample_tactical_search_values(), prefix="normal_combat")

    assert metrics["normal_combat/zero_energy_x_cost_selected_rate"] == 1 / 3
    assert metrics["normal_combat/hp_cost_low_margin_selected_rate"] == 1 / 3
    assert metrics["normal_combat/refund_no_followup_selected_rate"] == 2 / 3
    assert metrics["normal_combat/refund_no_followup_with_progress_selected_rate"] == 1 / 3
    assert metrics["normal_combat/refund_no_followup_no_alternative_selected_rate"] == 1 / 3
    assert metrics["normal_combat/refund_no_followup_guard_applied_rate"] == 1 / 3
    assert metrics["normal_combat/potion_save_recommended_selected_rate"] == 1 / 3
    assert all(key.startswith("normal_combat/") for key in metrics)


def test_combat_tactical_issue_metrics_default_to_zero_for_missing_values() -> None:
    metrics = combat_tactical_issue_metrics({}, prefix="weak_combat/")

    assert metrics["weak_combat/zero_energy_x_cost_selected_rate"] == 0.0
    assert metrics["weak_combat/hp_cost_self_lethal_selected_rate"] == 0.0
    assert metrics["weak_combat/hp_cost_low_margin_selected_rate"] == 0.0
    assert metrics["weak_combat/refund_no_followup_selected_rate"] == 0.0
    assert metrics["weak_combat/refund_no_followup_with_progress_selected_rate"] == 0.0
    assert metrics["weak_combat/refund_no_followup_progress_alternative_count_mean"] == 0.0
    assert metrics["weak_combat/strategic_defer_end_turn_selected_rate"] == 0.0
    assert metrics["weak_combat/potion_save_recommended_selected_rate"] == 0.0
    assert metrics["weak_combat/potion_use_quality_selected_mean"] == 0.0
    assert all(value == 0.0 for value in metrics.values())


def test_boss_card_block_waste_metrics_default_to_zero_for_missing_values() -> None:
    metrics = boss_card_block_waste_metrics({})

    assert set(metrics) == {
        "boss_combat/card_block_waste_available_mean",
        "boss_combat/card_pure_block_available_mean",
        "boss_combat/card_no_damage_pressure_available_mean",
        "boss_combat/card_block_waste_bias_count_mean",
        "boss_combat/card_block_waste_bias_applied_rate",
        "boss_combat/card_block_waste_bias_min",
        "boss_combat/card_block_waste_hard_bias_applied_rate",
        "boss_combat/card_block_waste_progress_alternative_rate",
        "boss_combat/card_block_waste_progress_bonus_count_mean",
        "boss_combat/card_block_waste_progress_bonus_max",
        "boss_combat/card_block_waste_selected_rate",
        "boss_combat/card_block_waste_with_progress_selected_rate",
        "boss_combat/card_pure_block_selected_rate",
        "boss_combat/bad_pure_block_selected_rate",
        "boss_combat/pure_block_survival_justified_selected_rate",
        "boss_combat/pure_block_progress_alternative_selected_rate",
        "boss_combat/pure_block_no_alternative_selected_rate",
        "boss_combat/pure_block_low_value_pressure_selected_rate",
        "boss_combat/insufficient_block_selected_rate",
        "boss_combat/card_no_damage_pressure_selected_rate",
        "boss_combat/card_no_damage_pressure_with_progress_selected_rate",
    }
    assert all(value == 0.0 for value in metrics.values())


def test_card_block_waste_metrics_default_to_zero_for_missing_values() -> None:
    metrics = card_block_waste_metrics({}, prefix="elite_combat/")

    assert set(metrics) == {
        "elite_combat/card_block_waste_available_mean",
        "elite_combat/card_pure_block_available_mean",
        "elite_combat/card_no_damage_pressure_available_mean",
        "elite_combat/card_block_waste_bias_count_mean",
        "elite_combat/card_block_waste_bias_applied_rate",
        "elite_combat/card_block_waste_bias_min",
        "elite_combat/card_block_waste_hard_bias_applied_rate",
        "elite_combat/card_block_waste_progress_alternative_rate",
        "elite_combat/card_block_waste_progress_bonus_count_mean",
        "elite_combat/card_block_waste_progress_bonus_max",
        "elite_combat/card_block_waste_selected_rate",
        "elite_combat/card_block_waste_with_progress_selected_rate",
        "elite_combat/card_pure_block_selected_rate",
        "elite_combat/bad_pure_block_selected_rate",
        "elite_combat/pure_block_survival_justified_selected_rate",
        "elite_combat/pure_block_progress_alternative_selected_rate",
        "elite_combat/pure_block_no_alternative_selected_rate",
        "elite_combat/pure_block_low_value_pressure_selected_rate",
        "elite_combat/insufficient_block_selected_rate",
        "elite_combat/card_no_damage_pressure_selected_rate",
        "elite_combat/card_no_damage_pressure_with_progress_selected_rate",
    }
    assert all(value == 0.0 for value in metrics.values())


def test_card_block_search_suffixes_cover_source_keys() -> None:
    assert COMBAT_QUALITY_CARD_BLOCK_SEARCH_SUFFIXES == {
        "combat_quality_card_block_waste_count": "combat_quality_card_block_waste_count",
        "combat_quality_card_pure_block_count": "combat_quality_card_pure_block_count",
        "combat_quality_card_no_damage_pressure_count": "combat_quality_card_no_damage_pressure_count",
        "combat_quality_card_block_waste_bias_count": "combat_quality_card_block_waste_bias_count",
        "combat_quality_card_block_waste_bias_min": "combat_quality_card_block_waste_bias_min",
        "combat_quality_card_block_waste_hard_bias_applied": "combat_quality_card_block_waste_hard_bias_applied",
        "combat_quality_card_block_waste_progress_alternative": "combat_quality_card_block_waste_progress_alternative",
        "combat_quality_card_block_waste_progress_bonus_count": "combat_quality_card_block_waste_progress_bonus_count",
        "combat_quality_card_block_waste_progress_bonus_max": "combat_quality_card_block_waste_progress_bonus_max",
        "combat_quality_card_block_waste_selected": "combat_quality_card_block_waste_selected",
        "combat_quality_card_block_waste_with_progress_selected": "combat_quality_card_block_waste_with_progress_selected",
        "combat_quality_card_pure_block_selected": "combat_quality_card_pure_block_selected",
        "combat_quality_bad_pure_block_selected": "combat_quality_bad_pure_block_selected",
        "combat_quality_pure_block_survival_justified_selected": "combat_quality_pure_block_survival_justified_selected",
        "combat_quality_pure_block_progress_alternative_selected": "combat_quality_pure_block_progress_alternative_selected",
        "combat_quality_pure_block_no_alternative_selected": "combat_quality_pure_block_no_alternative_selected",
        "combat_quality_pure_block_low_value_pressure_selected": "combat_quality_pure_block_low_value_pressure_selected",
        "combat_quality_insufficient_block_selected": "combat_quality_insufficient_block_selected",
        "combat_quality_card_no_damage_pressure_selected": "combat_quality_card_no_damage_pressure_selected",
        "combat_quality_card_no_damage_pressure_with_progress_selected": "combat_quality_card_no_damage_pressure_with_progress_selected",
    }


def test_guard_search_suffixes_cover_no_pressure_pressure_window_keys() -> None:
    assert (
        COMBAT_QUALITY_GUARD_SEARCH_SUFFIXES["combat_quality_no_pressure_block_guard_pressure_attack_window"]
        == "combat_quality_no_pressure_block_guard_pressure_attack_window"
    )
    assert (
        COMBAT_QUALITY_GUARD_SEARCH_SUFFIXES["combat_quality_no_pressure_block_guard_low_value_pressure"]
        == "combat_quality_no_pressure_block_guard_low_value_pressure"
    )
    assert (
        COMBAT_QUALITY_GUARD_SEARCH_SUFFIXES["combat_quality_refund_no_followup_guard_applied"]
        == "combat_quality_refund_no_followup_guard_applied"
    )


def test_tactical_search_suffixes_cover_high_value_issue_keys() -> None:
    for key in (
        "combat_quality_zero_energy_x_cost_selected",
        "combat_quality_x_cost_zero_bad_selected",
        "combat_quality_hp_cost_self_lethal_selected",
        "combat_quality_hp_cost_low_margin_selected",
        "combat_quality_refund_no_followup_selected",
        "combat_quality_refund_no_followup_with_progress_selected",
        "combat_quality_refund_no_followup_progress_alternative_selected",
        "combat_quality_refund_no_followup_no_alternative_selected",
        "combat_quality_refund_no_followup_progress_alternative_count",
        "combat_quality_strategic_defer_end_turn_selected",
        "combat_quality_potion_low_urgency_selected",
        "combat_quality_potion_save_recommended_selected",
        "combat_quality_x_cost_zero_guard_applied",
        "combat_quality_hp_cost_margin_guard_applied",
    ):
        assert COMBAT_QUALITY_TACTICAL_SEARCH_SUFFIXES[key] == key
