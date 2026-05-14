"""Merge bridge action diagnostics into a trajectory step safely.

``CombatEnv.step`` emits ``info["action_diagnostics"]`` after the selected
action is executed.  Self-play has already written trainer/search-side
diagnostics for the action it *actually* sent to the env, including hard-guard
overrides that may clear "selected bad action" gauges.

This module centralizes the merge contract so post-step bridge diagnostics do
not accidentally re-mark a bad pre-override action as selected after a hard
guard corrected it.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from typing import Any


ACTION_DIAGNOSTIC_STAT_KEYS: Mapping[str, str] = {
    "energy": "combat_quality_energy",
    "positive_action_count": "combat_quality_positive_action_count",
    "mandatory_positive_action_count": "combat_quality_mandatory_positive_action_count",
    "strategic_skip_candidate_count": "combat_quality_strategic_skip_candidate_count",
    "wasteful_end_turn_available": "combat_quality_true_wasteful_end_turn_available",
    "wasteful_end_turn_selected": "combat_quality_true_wasteful_end_turn_selected",
    "zero_energy_x_cost_available": "combat_quality_zero_energy_x_cost_count",
    "zero_energy_x_cost_selected": "combat_quality_zero_energy_x_cost_selected",
    "refund_no_followup_available": "combat_quality_refund_no_followup_available",
    "refund_no_followup_selected": "combat_quality_refund_no_followup_selected",
    "strategic_skip_selected": "combat_quality_strategic_skip_selected",
    "typed_followup_missing_count": "combat_quality_typed_followup_missing_count",
    "typed_future_penalty_count": "combat_quality_typed_future_penalty_count",
    "typed_no_draw_count": "combat_quality_typed_no_draw_count",
    "typed_card_state_mutation_count": "combat_quality_typed_card_state_mutation_count",
    "setup_followup_dependent_count": "combat_quality_setup_followup_dependent_count",
    "setup_followup_available_count": "combat_quality_setup_followup_available_count",
    "enchantment_seen": "combat_quality_enchantment_seen",
    "affliction_seen": "combat_quality_affliction_seen",
    "wasteful_end_turn_penalty_applied": "combat_quality_wasteful_end_turn_penalty_applied",
    # P0 hardening hooks: HP-cost / X-cost / selection / identity /
    # transient leak — sourced from the typed helper outputs in
    # combat_env.step diagnostics.
    "hp_cost_self_lethal_selected": "combat_quality_hp_cost_self_lethal_selected",
    "hp_cost_low_margin_selected": "combat_quality_hp_cost_low_margin_selected",
    "hp_cost_unblockable_value": "combat_quality_hp_cost_unblockable_value",
    "insufficient_block_selected": "combat_quality_insufficient_block_selected",
    "x_cost_selected": "combat_quality_x_cost_selected",
    "x_cost_zero_bad_selected": "combat_quality_x_cost_zero_bad_selected",
    "x_cost_zero_selected": "combat_quality_x_cost_zero_selected",
    "x_cost_energy_value": "combat_quality_x_cost_energy_value",
    "x_cost_star_value": "combat_quality_x_cost_star_value",
    "star_x_selected": "combat_quality_star_x_selected",
    "selection_text_fallback_selected": "combat_quality_selection_text_fallback_selected",
    "selection_runtime_internal_selected": "combat_quality_selection_runtime_internal_selected",
    "card_identity_text_fallback_selected": "combat_quality_card_identity_text_fallback_selected",
    "card_identity_runtime_internal_selected": "combat_quality_card_identity_runtime_internal_selected",
    "transient_leaked": "combat_quality_transient_leaked_selected",
    "prior_transient_only_end_turn": "combat_quality_prior_transient_only_end_turn",
    "post_step_frontier_attempted": "combat_quality_post_step_frontier_attempted",
    "post_step_frontier_resolved": "combat_quality_post_step_frontier_resolved",
    "post_step_frontier_timeout": "combat_quality_post_step_frontier_timeout",
    "post_step_frontier_leaked": "combat_quality_post_step_frontier_leaked",
    "post_step_frontier_stable_no_actions": "combat_quality_post_step_frontier_stable_no_actions",
    "post_step_frontier_rebind_attempted": "combat_quality_post_step_frontier_rebind_attempted",
    "post_step_frontier_rebind_succeeded": "combat_quality_post_step_frontier_rebind_succeeded",
    "post_step_frontier_suspicious_singleton": "combat_quality_post_step_frontier_suspicious_singleton",
    "post_step_frontier_wait_ms": "combat_quality_post_step_frontier_wait_ms",
    "post_step_frontier_poll_count": "combat_quality_post_step_frontier_poll_count",
}


# If one of these guards rewrote the action, the listed selected-side gauges
# were deliberately reset on the trainer side.  Bridge diagnostics may still
# contain broad/pre-step selected values, so preserve the reset instead of
# reintroducing the bad-action label.
GUARD_PRESERVED_SELECTED_STATS: Mapping[str, tuple[str, ...]] = {
    "combat_quality_refund_no_followup_guard_applied": (
        "combat_quality_refund_no_followup_selected",
        "combat_quality_refund_no_followup_with_progress_selected",
        "combat_quality_refund_no_followup_progress_alternative_selected",
        "combat_quality_refund_no_followup_no_alternative_selected",
        "combat_quality_refund_no_followup_progress_alternative_count",
        "combat_quality_strategic_skip_selected",
    ),
    "combat_quality_x_cost_zero_guard_applied": (
        "combat_quality_zero_energy_x_cost_selected",
        "combat_quality_x_cost_zero_bad_selected",
        "combat_quality_x_cost_zero_selected",
    ),
    "combat_quality_hp_cost_margin_guard_applied": (
        "combat_quality_hp_cost_self_lethal_selected",
        "combat_quality_hp_cost_low_margin_selected",
    ),
    "combat_quality_no_pressure_block_guard_applied": (
        "combat_quality_card_block_waste_selected",
        "combat_quality_card_block_waste_with_progress_selected",
        "combat_quality_card_pure_block_selected",
        "combat_quality_card_no_damage_pressure_selected",
        "combat_quality_card_no_damage_pressure_with_progress_selected",
        # The no-pressure guard also clears the narrow selected-side pure
        # block buckets in ``NoPressureBlockGuardMixin``.  Bridge
        # ``action_diagnostics`` may still report the pre-override Defend-like
        # card; do not let the post-step merge resurrect stale gate failures
        # after the trainer already sent a progress action to the env.
        "combat_quality_bad_pure_block_selected",
        "combat_quality_insufficient_block_selected",
        "combat_quality_pure_block_progress_alternative_selected",
        "combat_quality_pure_block_survival_justified_selected",
        "combat_quality_pure_block_no_alternative_selected",
        "combat_quality_pure_block_low_value_pressure_selected",
    ),
}


def _as_float(value: Any) -> float | None:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return None


def _preserved_stats(diag_stats: Mapping[str, Any]) -> set[str]:
    preserved: set[str] = set()
    for guard_key, stat_keys in GUARD_PRESERVED_SELECTED_STATS.items():
        value = _as_float(diag_stats.get(guard_key))
        if value is not None and value > 0.5:
            preserved.update(stat_keys)
    return preserved


def merge_action_diagnostics_into_search_stats(
    diag_stats: MutableMapping[str, Any],
    action_diagnostics: Mapping[str, Any],
) -> None:
    """Merge ``CombatEnv.step`` diagnostics into an existing search_stats dict.

    The function mutates ``diag_stats`` in place.  It intentionally preserves
    selected-side bad-action resets when a hard guard already rewrote the action.
    Availability/count diagnostics are still merged normally.
    """

    preserved = _preserved_stats(diag_stats)
    for diag_key, stat_key in ACTION_DIAGNOSTIC_STAT_KEYS.items():
        if diag_key not in action_diagnostics:
            continue
        value = _as_float(action_diagnostics.get(diag_key))
        if value is None:
            continue
        if stat_key in preserved and value > 0.5:
            continue
        diag_stats[stat_key] = value

    if float(diag_stats.get("combat_quality_true_wasteful_end_turn_available", 0.0) or 0.0) > 0.5:
        diag_stats["combat_quality_wasteful_end_turn_available"] = 1.0
    if float(diag_stats.get("combat_quality_true_wasteful_end_turn_selected", 0.0) or 0.0) > 0.5:
        diag_stats["combat_quality_wasteful_end_turn_selected"] = 1.0


__all__ = [
    "ACTION_DIAGNOSTIC_STAT_KEYS",
    "GUARD_PRESERVED_SELECTED_STATS",
    "merge_action_diagnostics_into_search_stats",
]
