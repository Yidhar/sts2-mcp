from muzero.training.action_diagnostics_merge import merge_action_diagnostics_into_search_stats


def test_refund_guard_preserves_selected_reset_but_keeps_availability():
    stats = {
        "combat_quality_refund_no_followup_guard_applied": 1.0,
        "combat_quality_refund_no_followup_selected": 0.0,
        "combat_quality_refund_no_followup_with_progress_selected": 0.0,
        "combat_quality_refund_no_followup_progress_alternative_selected": 0.0,
        "combat_quality_refund_no_followup_no_alternative_selected": 0.0,
        "combat_quality_refund_no_followup_progress_alternative_count": 0.0,
        "combat_quality_strategic_skip_selected": 0.0,
    }
    merge_action_diagnostics_into_search_stats(
        stats,
        {
            "refund_no_followup_available": 2.0,
            "refund_no_followup_selected": 1.0,
            "strategic_skip_selected": 1.0,
            "typed_followup_missing_count": 1.0,
        },
    )

    assert stats["combat_quality_refund_no_followup_available"] == 2.0
    assert stats["combat_quality_typed_followup_missing_count"] == 1.0
    assert stats["combat_quality_refund_no_followup_selected"] == 0.0
    assert stats["combat_quality_refund_no_followup_with_progress_selected"] == 0.0
    assert stats["combat_quality_refund_no_followup_progress_alternative_selected"] == 0.0
    assert stats["combat_quality_refund_no_followup_no_alternative_selected"] == 0.0
    assert stats["combat_quality_refund_no_followup_progress_alternative_count"] == 0.0
    assert stats["combat_quality_strategic_skip_selected"] == 0.0


def test_x_cost_and_hp_guards_preserve_their_selected_bad_resets():
    stats = {
        "combat_quality_x_cost_zero_guard_applied": 1.0,
        "combat_quality_hp_cost_margin_guard_applied": 1.0,
        "combat_quality_zero_energy_x_cost_selected": 0.0,
        "combat_quality_x_cost_zero_bad_selected": 0.0,
        "combat_quality_x_cost_zero_selected": 0.0,
        "combat_quality_hp_cost_self_lethal_selected": 0.0,
        "combat_quality_hp_cost_low_margin_selected": 0.0,
    }
    merge_action_diagnostics_into_search_stats(
        stats,
        {
            "zero_energy_x_cost_selected": 1.0,
            "x_cost_zero_bad_selected": 1.0,
            "x_cost_zero_selected": 1.0,
            "x_cost_selected": 1.0,
            "hp_cost_self_lethal_selected": 1.0,
            "hp_cost_low_margin_selected": 1.0,
            "hp_cost_unblockable_value": 6.0,
        },
    )

    assert stats["combat_quality_zero_energy_x_cost_selected"] == 0.0
    assert stats["combat_quality_x_cost_zero_bad_selected"] == 0.0
    assert stats["combat_quality_x_cost_zero_selected"] == 0.0
    assert stats["combat_quality_x_cost_selected"] == 1.0
    assert stats["combat_quality_hp_cost_self_lethal_selected"] == 0.0
    assert stats["combat_quality_hp_cost_low_margin_selected"] == 0.0
    assert stats["combat_quality_hp_cost_unblockable_value"] == 6.0


def test_no_pressure_block_guard_preserves_all_selected_pure_block_resets():
    stats = {
        "combat_quality_no_pressure_block_guard_applied": 1.0,
        "combat_quality_card_block_waste_selected": 0.0,
        "combat_quality_card_block_waste_with_progress_selected": 0.0,
        "combat_quality_card_pure_block_selected": 0.0,
        "combat_quality_card_no_damage_pressure_selected": 0.0,
        "combat_quality_card_no_damage_pressure_with_progress_selected": 0.0,
        "combat_quality_bad_pure_block_selected": 0.0,
        "combat_quality_pure_block_progress_alternative_selected": 0.0,
        "combat_quality_pure_block_survival_justified_selected": 0.0,
        "combat_quality_pure_block_no_alternative_selected": 0.0,
        "combat_quality_pure_block_low_value_pressure_selected": 0.0,
        "combat_quality_insufficient_block_selected": 0.0,
    }
    merge_action_diagnostics_into_search_stats(
        stats,
        {
            "card_block_waste_selected": 1.0,
            "card_block_waste_with_progress_selected": 1.0,
            "card_pure_block_selected": 1.0,
            "card_no_damage_pressure_selected": 1.0,
            "card_no_damage_pressure_with_progress_selected": 1.0,
            "bad_pure_block_selected": 1.0,
            "pure_block_progress_alternative_selected": 1.0,
            "pure_block_survival_justified_selected": 1.0,
            "pure_block_no_alternative_selected": 1.0,
            "pure_block_low_value_pressure_selected": 1.0,
            "insufficient_block_selected": 1.0,
            "energy": 2.0,
        },
    )

    assert stats["combat_quality_energy"] == 2.0
    assert stats["combat_quality_card_block_waste_selected"] == 0.0
    assert stats["combat_quality_card_block_waste_with_progress_selected"] == 0.0
    assert stats["combat_quality_card_pure_block_selected"] == 0.0
    assert stats["combat_quality_card_no_damage_pressure_selected"] == 0.0
    assert stats["combat_quality_card_no_damage_pressure_with_progress_selected"] == 0.0
    assert stats["combat_quality_bad_pure_block_selected"] == 0.0
    assert stats["combat_quality_pure_block_progress_alternative_selected"] == 0.0
    assert stats["combat_quality_pure_block_survival_justified_selected"] == 0.0
    assert stats["combat_quality_pure_block_no_alternative_selected"] == 0.0
    assert stats["combat_quality_pure_block_low_value_pressure_selected"] == 0.0
    assert stats["combat_quality_insufficient_block_selected"] == 0.0


def test_normal_merge_without_guard_records_selected_bad_flags():
    stats = {}
    merge_action_diagnostics_into_search_stats(
        stats,
        {
            "wasteful_end_turn_available": 1.0,
            "wasteful_end_turn_selected": 1.0,
            "refund_no_followup_selected": 1.0,
        },
    )

    assert stats["combat_quality_refund_no_followup_selected"] == 1.0
    assert stats["combat_quality_true_wasteful_end_turn_available"] == 1.0
    assert stats["combat_quality_true_wasteful_end_turn_selected"] == 1.0
    assert stats["combat_quality_wasteful_end_turn_available"] == 1.0
    assert stats["combat_quality_wasteful_end_turn_selected"] == 1.0
