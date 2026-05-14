from __future__ import annotations

from muzero.combat_quality.guard_metrics import COMBAT_HARD_GUARD_DEFAULT_KEYS


def test_hard_guard_default_keys_are_unique_strings() -> None:
    assert COMBAT_HARD_GUARD_DEFAULT_KEYS
    assert all(isinstance(key, str) for key in COMBAT_HARD_GUARD_DEFAULT_KEYS)
    assert len(COMBAT_HARD_GUARD_DEFAULT_KEYS) == len(set(COMBAT_HARD_GUARD_DEFAULT_KEYS))


def test_hard_guard_default_keys_cover_current_guard_families() -> None:
    required_prefixes = (
        "combat_quality_kaiser_facing_guard_",
        "combat_quality_insatiable_escape_force_",
        "combat_quality_x_cost_zero_guard_",
        "combat_quality_refund_no_followup_guard_",
        "combat_quality_hp_cost_margin_guard_",
        "combat_quality_potion_bad_guard_",
        "combat_quality_potion_discard_guard_",
        "combat_quality_elite_boss_lethal_end_turn_guard_",
        "combat_quality_boss_survival_potion_guard_",
        "combat_quality_boss_race_potion_guard_",
        "combat_quality_boss_survival_block_guard_",
        "combat_quality_late_normal_lethal_end_turn_guard_",
        "combat_quality_late_normal_survival_guard_",
        "combat_quality_late_normal_race_potion_guard_",
        "combat_quality_survival_non_endturn_guard_",
        "combat_quality_selection_loop_",
        "combat_quality_meaningful_damage_endturn_guard_",
        "combat_quality_urgent_endturn_guard_",
        "combat_quality_no_pressure_block_guard_",
    )

    for prefix in required_prefixes:
        assert any(key.startswith(prefix) for key in COMBAT_HARD_GUARD_DEFAULT_KEYS), prefix


def test_hard_guard_default_keys_keep_policy_target_contract() -> None:
    assert "combat_quality_hard_guard_override_any" in COMBAT_HARD_GUARD_DEFAULT_KEYS
    assert "combat_quality_hard_guard_policy_target_rewrite" in COMBAT_HARD_GUARD_DEFAULT_KEYS


def test_no_pressure_block_guard_default_keys_cover_pressure_window_metrics() -> None:
    assert "combat_quality_no_pressure_block_guard_pressure_attack_window" in COMBAT_HARD_GUARD_DEFAULT_KEYS
    assert "combat_quality_no_pressure_block_guard_low_value_pressure" in COMBAT_HARD_GUARD_DEFAULT_KEYS
    assert "combat_quality_no_pressure_block_guard_progress_override_idx" in COMBAT_HARD_GUARD_DEFAULT_KEYS
    assert "combat_quality_no_pressure_block_guard_progress_override_lock" in COMBAT_HARD_GUARD_DEFAULT_KEYS
    assert "combat_quality_survival_non_endturn_guard_no_pressure_lock_skip" in COMBAT_HARD_GUARD_DEFAULT_KEYS
    assert (
        "combat_quality_survival_non_endturn_guard_no_pressure_lock_critical_override"
        in COMBAT_HARD_GUARD_DEFAULT_KEYS
    )
