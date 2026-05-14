"""Metric helpers for combat-quality policy diagnostics.

This module is intentionally small and dependency-light.  The legacy trainer
still owns TensorBoard writing, but metric key contracts and aggregation logic
belong next to the combat-quality policy code instead of being duplicated in
``muzero.train``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType


NumericSequence = Sequence[float | int]
SearchValues = Mapping[str, NumericSequence]


COMBAT_QUALITY_CARD_BLOCK_SEARCH_SUFFIXES: Mapping[str, str] = MappingProxyType(
    {
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
        "combat_quality_insufficient_block_selected": "combat_quality_insufficient_block_selected",
        "combat_quality_pure_block_survival_justified_selected": "combat_quality_pure_block_survival_justified_selected",
        "combat_quality_pure_block_progress_alternative_selected": "combat_quality_pure_block_progress_alternative_selected",
        "combat_quality_pure_block_no_alternative_selected": "combat_quality_pure_block_no_alternative_selected",
        "combat_quality_pure_block_low_value_pressure_selected": "combat_quality_pure_block_low_value_pressure_selected",
        "combat_quality_card_no_damage_pressure_selected": "combat_quality_card_no_damage_pressure_selected",
        "combat_quality_card_no_damage_pressure_with_progress_selected": "combat_quality_card_no_damage_pressure_with_progress_selected",
    }
)

COMBAT_QUALITY_GUARD_SEARCH_SUFFIXES: Mapping[str, str] = MappingProxyType(
    {
        "combat_quality_meaningful_damage_endturn_guard_available": "combat_quality_meaningful_damage_endturn_guard_available",
        "combat_quality_meaningful_damage_endturn_guard_applied": "combat_quality_meaningful_damage_endturn_guard_applied",
        "combat_quality_meaningful_damage_endturn_guard_override": "combat_quality_meaningful_damage_endturn_guard_override",
        "combat_quality_meaningful_damage_endturn_guard_candidate_count": "combat_quality_meaningful_damage_endturn_guard_candidate_count",
        "combat_quality_meaningful_damage_endturn_guard_no_alternative": "combat_quality_meaningful_damage_endturn_guard_no_alternative",
        "combat_quality_meaningful_damage_endturn_guard_lethal_candidate": "combat_quality_meaningful_damage_endturn_guard_lethal_candidate",
        "combat_quality_meaningful_damage_endturn_guard_pressure_skip": "combat_quality_meaningful_damage_endturn_guard_pressure_skip",
        "combat_quality_no_pressure_block_guard_available": "combat_quality_no_pressure_block_guard_available",
        "combat_quality_no_pressure_block_guard_applied": "combat_quality_no_pressure_block_guard_applied",
        "combat_quality_no_pressure_block_guard_override": "combat_quality_no_pressure_block_guard_override",
        "combat_quality_no_pressure_block_guard_candidate_count": "combat_quality_no_pressure_block_guard_candidate_count",
        "combat_quality_no_pressure_block_guard_no_alternative": "combat_quality_no_pressure_block_guard_no_alternative",
        "combat_quality_no_pressure_block_guard_pressure_skip": "combat_quality_no_pressure_block_guard_pressure_skip",
        "combat_quality_no_pressure_block_guard_pressure_attack_window": "combat_quality_no_pressure_block_guard_pressure_attack_window",
        "combat_quality_no_pressure_block_guard_low_value_pressure": "combat_quality_no_pressure_block_guard_low_value_pressure",
        "combat_quality_no_pressure_block_guard_trivial_pressure": "combat_quality_no_pressure_block_guard_trivial_pressure",
        "combat_quality_no_pressure_block_guard_lethal_candidate": "combat_quality_no_pressure_block_guard_lethal_candidate",
        "combat_quality_refund_no_followup_guard_available": "combat_quality_refund_no_followup_guard_available",
        "combat_quality_refund_no_followup_guard_applied": "combat_quality_refund_no_followup_guard_applied",
        "combat_quality_refund_no_followup_guard_override": "combat_quality_refund_no_followup_guard_override",
        "combat_quality_refund_no_followup_guard_no_alternative": "combat_quality_refund_no_followup_guard_no_alternative",
        "combat_quality_refund_no_followup_guard_end_turn_fallback": "combat_quality_refund_no_followup_guard_end_turn_fallback",
        "combat_quality_refund_no_followup_guard_lethal_exemption": "combat_quality_refund_no_followup_guard_lethal_exemption",
        "combat_quality_refund_no_followup_guard_candidate_count": "combat_quality_refund_no_followup_guard_candidate_count",
        "combat_quality_refund_no_followup_guard_lethal_candidate": "combat_quality_refund_no_followup_guard_lethal_candidate",
    }
)

COMBAT_QUALITY_TACTICAL_SEARCH_SUFFIXES: Mapping[str, str] = MappingProxyType(
    {
        # End-turn taxonomy / strategic defer.
        "combat_quality_wasteful_end_turn_available": "combat_quality_wasteful_end_turn_available",
        "combat_quality_wasteful_end_turn_selected": "combat_quality_wasteful_end_turn_selected",
        "combat_quality_true_wasteful_end_turn_available": "combat_quality_true_wasteful_end_turn_available",
        "combat_quality_true_wasteful_end_turn_selected": "combat_quality_true_wasteful_end_turn_selected",
        "combat_quality_strategic_defer_available": "combat_quality_strategic_defer_available",
        "combat_quality_strategic_defer_end_turn_selected": "combat_quality_strategic_defer_end_turn_selected",
        "combat_quality_bad_end_turn_available": "combat_quality_bad_end_turn_available",
        "combat_quality_bad_end_turn_selected": "combat_quality_bad_end_turn_selected",
        "combat_quality_forced_end_turn_available": "combat_quality_forced_end_turn_available",
        "combat_quality_forced_end_turn_selected": "combat_quality_forced_end_turn_selected",
        "combat_quality_end_turn_unknown_selected": "combat_quality_end_turn_unknown_selected",
        "combat_quality_end_turn_selected": "combat_quality_end_turn_selected",
        # X-cost / zero-energy X-cost.
        "combat_quality_x_cost_available_count": "combat_quality_x_cost_available_count",
        "combat_quality_x_cost_selected": "combat_quality_x_cost_selected",
        "combat_quality_x_cost_selected_energy": "combat_quality_x_cost_selected_energy",
        "combat_quality_x_cost_zero_energy_selected": "combat_quality_x_cost_zero_energy_selected",
        "combat_quality_zero_energy_x_cost_count": "combat_quality_zero_energy_x_cost_count",
        "combat_quality_zero_energy_x_cost_selected": "combat_quality_zero_energy_x_cost_selected",
        "combat_quality_x_cost_bad_count": "combat_quality_x_cost_bad_count",
        "combat_quality_x_cost_effective_energy_mean": "combat_quality_x_cost_effective_energy_mean",
        "combat_quality_x_cost_selected_effective_energy": "combat_quality_x_cost_selected_effective_energy",
        "combat_quality_x_cost_has_non_energy_effect_selected": "combat_quality_x_cost_has_non_energy_effect_selected",
        "combat_quality_x_cost_bad_selected": "combat_quality_x_cost_bad_selected",
        "combat_quality_x_cost_zero_bad_selected": "combat_quality_x_cost_zero_bad_selected",
        "combat_quality_x_cost_zero_selected": "combat_quality_x_cost_zero_selected",
        "combat_quality_x_cost_energy_value": "combat_quality_x_cost_energy_value",
        "combat_quality_x_cost_star_value": "combat_quality_x_cost_star_value",
        "combat_quality_star_x_selected": "combat_quality_star_x_selected",
        "combat_quality_x_cost_zero_guard_available": "combat_quality_x_cost_zero_guard_available",
        "combat_quality_x_cost_zero_guard_applied": "combat_quality_x_cost_zero_guard_applied",
        "combat_quality_x_cost_zero_guard_override": "combat_quality_x_cost_zero_guard_override",
        "combat_quality_x_cost_zero_guard_no_alternative": "combat_quality_x_cost_zero_guard_no_alternative",
        "combat_quality_x_cost_zero_guard_end_turn_fallback": "combat_quality_x_cost_zero_guard_end_turn_fallback",
        # HP-cost / self-damage safety.
        "combat_quality_hp_cost_self_lethal_selected": "combat_quality_hp_cost_self_lethal_selected",
        "combat_quality_hp_cost_low_margin_selected": "combat_quality_hp_cost_low_margin_selected",
        "combat_quality_hp_cost_unblockable_value": "combat_quality_hp_cost_unblockable_value",
        "combat_quality_hp_cost_margin_guard_available": "combat_quality_hp_cost_margin_guard_available",
        "combat_quality_hp_cost_margin_guard_applied": "combat_quality_hp_cost_margin_guard_applied",
        "combat_quality_hp_cost_margin_guard_override": "combat_quality_hp_cost_margin_guard_override",
        "combat_quality_hp_cost_margin_guard_lethal_exemption": "combat_quality_hp_cost_margin_guard_lethal_exemption",
        "combat_quality_hp_cost_margin_guard_no_alternative": "combat_quality_hp_cost_margin_guard_no_alternative",
        # Refund/setup cards that are bad when no follow-up exists.
        "combat_quality_energy_gain_without_followup_count": "combat_quality_energy_gain_without_followup_count",
        "combat_quality_typed_followup_missing_count": "combat_quality_typed_followup_missing_count",
        "combat_quality_typed_future_penalty_count": "combat_quality_typed_future_penalty_count",
        "combat_quality_typed_no_draw_count": "combat_quality_typed_no_draw_count",
        "combat_quality_typed_card_state_mutation_count": "combat_quality_typed_card_state_mutation_count",
        "combat_quality_setup_followup_dependent_count": "combat_quality_setup_followup_dependent_count",
        "combat_quality_setup_followup_available_count": "combat_quality_setup_followup_available_count",
        "combat_quality_refund_no_followup_available": "combat_quality_refund_no_followup_available",
        "combat_quality_refund_no_followup_selected": "combat_quality_refund_no_followup_selected",
        "combat_quality_refund_no_followup_with_progress_selected": "combat_quality_refund_no_followup_with_progress_selected",
        "combat_quality_refund_no_followup_progress_alternative_selected": "combat_quality_refund_no_followup_progress_alternative_selected",
        "combat_quality_refund_no_followup_no_alternative_selected": "combat_quality_refund_no_followup_no_alternative_selected",
        "combat_quality_refund_no_followup_progress_alternative_count": "combat_quality_refund_no_followup_progress_alternative_count",
        "combat_quality_strategic_skip_candidate_count": "combat_quality_strategic_skip_candidate_count",
        "combat_quality_strategic_skip_selected": "combat_quality_strategic_skip_selected",
        # Potion timing/urgency and bad-use guards.  These are global combat
        # signals, not boss-only; Act 1 can fail before the first boss.
        "combat_quality_potion_available_count": "combat_quality_potion_available_count",
        "combat_quality_potion_urgent_count": "combat_quality_potion_urgent_count",
        "combat_quality_potion_low_urgency_count": "combat_quality_potion_low_urgency_count",
        "combat_quality_potion_save_recommended_count": "combat_quality_potion_save_recommended_count",
        "combat_quality_potion_no_followup_count": "combat_quality_potion_no_followup_count",
        "combat_quality_potion_lethal_count": "combat_quality_potion_lethal_count",
        "combat_quality_potion_prevent_lethal_count": "combat_quality_potion_prevent_lethal_count",
        "combat_quality_potion_mechanism_count": "combat_quality_potion_mechanism_count",
        "combat_quality_potion_overkill_count": "combat_quality_potion_overkill_count",
        "combat_quality_potion_block_waste_count": "combat_quality_potion_block_waste_count",
        "combat_quality_potion_use_quality_mean": "combat_quality_potion_use_quality_mean",
        "combat_quality_potion_waste_risk_mean": "combat_quality_potion_waste_risk_mean",
        "combat_quality_potion_save_value_mean": "combat_quality_potion_save_value_mean",
        "combat_quality_potion_hand_context_good_count": "combat_quality_potion_hand_context_good_count",
        "combat_quality_potion_hand_context_bad_count": "combat_quality_potion_hand_context_bad_count",
        "combat_quality_potion_long_term_count": "combat_quality_potion_long_term_count",
        "combat_quality_potion_requires_followup_count": "combat_quality_potion_requires_followup_count",
        "combat_quality_bad_potion_bias_count": "combat_quality_bad_potion_bias_count",
        "combat_quality_bad_potion_bias_min": "combat_quality_bad_potion_bias_min",
        "combat_quality_potion_selected": "combat_quality_potion_selected",
        "combat_quality_potion_selected_when_available": "combat_quality_potion_selected_when_available",
        "combat_quality_potion_high_urgency_selected": "combat_quality_potion_high_urgency_selected",
        "combat_quality_potion_low_urgency_selected": "combat_quality_potion_low_urgency_selected",
        "combat_quality_potion_save_recommended_selected": "combat_quality_potion_save_recommended_selected",
        "combat_quality_potion_no_followup_selected": "combat_quality_potion_no_followup_selected",
        "combat_quality_potion_lethal_selected": "combat_quality_potion_lethal_selected",
        "combat_quality_potion_prevent_lethal_selected": "combat_quality_potion_prevent_lethal_selected",
        "combat_quality_potion_mechanism_selected": "combat_quality_potion_mechanism_selected",
        "combat_quality_potion_overkill_selected": "combat_quality_potion_overkill_selected",
        "combat_quality_potion_block_waste_selected": "combat_quality_potion_block_waste_selected",
        "combat_quality_potion_use_quality_selected": "combat_quality_potion_use_quality_selected",
        "combat_quality_potion_waste_risk_selected": "combat_quality_potion_waste_risk_selected",
        "combat_quality_potion_bad_guard_available": "combat_quality_potion_bad_guard_available",
        "combat_quality_potion_bad_guard_applied": "combat_quality_potion_bad_guard_applied",
        "combat_quality_potion_bad_guard_override": "combat_quality_potion_bad_guard_override",
        "combat_quality_potion_bad_guard_lethal_exemption": "combat_quality_potion_bad_guard_lethal_exemption",
        "combat_quality_potion_bad_guard_no_alternative": "combat_quality_potion_bad_guard_no_alternative",
        "combat_quality_potion_bad_guard_forced_end_turn_hopeless": "combat_quality_potion_bad_guard_forced_end_turn_hopeless",
        "combat_quality_potion_discard_guard_available": "combat_quality_potion_discard_guard_available",
        "combat_quality_potion_discard_guard_applied": "combat_quality_potion_discard_guard_applied",
        "combat_quality_potion_discard_guard_override": "combat_quality_potion_discard_guard_override",
        "combat_quality_potion_discard_guard_candidate_count": "combat_quality_potion_discard_guard_candidate_count",
        "combat_quality_potion_discard_guard_saved_survival": "combat_quality_potion_discard_guard_saved_survival",
    }
)


def _as_floats(values: NumericSequence | None) -> tuple[float, ...]:
    if not values:
        return ()
    out: list[float] = []
    for value in values:
        try:
            out.append(float(value))
        except (TypeError, ValueError):
            continue
    return tuple(out)


def _mean(values: NumericSequence | None) -> float:
    nums = _as_floats(values)
    if not nums:
        return 0.0
    return float(sum(nums) / len(nums))


def _positive_rate(values: NumericSequence | None) -> float:
    nums = _as_floats(values)
    if not nums:
        return 0.0
    return float(sum(1.0 for value in nums if value > 0.0) / len(nums))


def _min(values: NumericSequence | None) -> float:
    nums = _as_floats(values)
    if not nums:
        return 0.0
    return float(min(nums))


def _normalize_prefix(prefix: str) -> str:
    normalized = str(prefix or "")
    if normalized and not normalized.endswith("/"):
        normalized = f"{normalized}/"
    return normalized


def card_block_waste_metrics(
    search_values: SearchValues,
    *,
    prefix: str = "combat_quality/",
) -> dict[str, float]:
    """Aggregate card block-waste diagnostics for TensorBoard tags.

    Source values are per combat decision.  Count fields are averaged as
    availability means, selected fields are averaged as selected rates, and the
    bias count is exposed both as an action-count mean and as a per-decision
    applied rate.

    ``prefix`` lets the same contract be emitted for global combat quality and
    for encounter-tier namespaces (``weak_combat/``, ``normal_combat/``,
    ``elite_combat/`` and the legacy ``boss_combat/`` wrapper).
    """

    tag_prefix = _normalize_prefix(prefix)
    bias_counts = search_values.get("combat_quality_card_block_waste_bias_count")
    return {
        f"{tag_prefix}card_block_waste_available_mean": _mean(
            search_values.get("combat_quality_card_block_waste_count")
        ),
        f"{tag_prefix}card_pure_block_available_mean": _mean(
            search_values.get("combat_quality_card_pure_block_count")
        ),
        f"{tag_prefix}card_no_damage_pressure_available_mean": _mean(
            search_values.get("combat_quality_card_no_damage_pressure_count")
        ),
        f"{tag_prefix}card_block_waste_bias_count_mean": _mean(bias_counts),
        f"{tag_prefix}card_block_waste_bias_applied_rate": _positive_rate(bias_counts),
        f"{tag_prefix}card_block_waste_bias_min": _min(
            search_values.get("combat_quality_card_block_waste_bias_min")
        ),
        f"{tag_prefix}card_block_waste_hard_bias_applied_rate": _positive_rate(
            search_values.get("combat_quality_card_block_waste_hard_bias_applied")
        ),
        f"{tag_prefix}card_block_waste_progress_alternative_rate": _positive_rate(
            search_values.get("combat_quality_card_block_waste_progress_alternative")
        ),
        f"{tag_prefix}card_block_waste_progress_bonus_count_mean": _mean(
            search_values.get("combat_quality_card_block_waste_progress_bonus_count")
        ),
        f"{tag_prefix}card_block_waste_progress_bonus_max": _mean(
            search_values.get("combat_quality_card_block_waste_progress_bonus_max")
        ),
        f"{tag_prefix}card_block_waste_selected_rate": _mean(
            search_values.get("combat_quality_card_block_waste_selected")
        ),
        f"{tag_prefix}card_block_waste_with_progress_selected_rate": _mean(
            search_values.get("combat_quality_card_block_waste_with_progress_selected")
        ),
        f"{tag_prefix}card_pure_block_selected_rate": _mean(
            search_values.get("combat_quality_card_pure_block_selected")
        ),
        f"{tag_prefix}bad_pure_block_selected_rate": _mean(
            search_values.get("combat_quality_bad_pure_block_selected")
        ),
        f"{tag_prefix}insufficient_block_selected_rate": _mean(
            search_values.get("combat_quality_insufficient_block_selected")
        ),
        f"{tag_prefix}pure_block_survival_justified_selected_rate": _mean(
            search_values.get("combat_quality_pure_block_survival_justified_selected")
        ),
        f"{tag_prefix}pure_block_progress_alternative_selected_rate": _mean(
            search_values.get("combat_quality_pure_block_progress_alternative_selected")
        ),
        f"{tag_prefix}pure_block_no_alternative_selected_rate": _mean(
            search_values.get("combat_quality_pure_block_no_alternative_selected")
        ),
        f"{tag_prefix}pure_block_low_value_pressure_selected_rate": _mean(
            search_values.get("combat_quality_pure_block_low_value_pressure_selected")
        ),
        f"{tag_prefix}card_no_damage_pressure_selected_rate": _mean(
            search_values.get("combat_quality_card_no_damage_pressure_selected")
        ),
        f"{tag_prefix}card_no_damage_pressure_with_progress_selected_rate": _mean(
            search_values.get("combat_quality_card_no_damage_pressure_with_progress_selected")
        ),
    }


def combat_tactical_issue_metrics(
    search_values: SearchValues,
    *,
    prefix: str = "combat_quality/",
) -> dict[str, float]:
    """Aggregate global/tier tactical issue diagnostics.

    Boss diagnostics have historically emitted rich HP-cost, X-cost, refund,
    potion and end-turn taxonomy metrics, while hallway combat only exposed a
    small block-waste subset.  Act 1 recovery needs these same signals before
    the boss, so this helper mirrors the boss contracts under any prefix used
    by the all-combat/tier emitters.
    """

    tag_prefix = _normalize_prefix(prefix)

    def mean(key: str) -> float:
        return _mean(search_values.get(key))

    def selected_mean(key: str) -> float:
        return _mean(search_values.get(key))

    potion_selected_sum = sum(_as_floats(search_values.get("combat_quality_potion_selected")))
    return {
        f"{tag_prefix}wasteful_end_turn_available_rate": mean("combat_quality_wasteful_end_turn_available"),
        f"{tag_prefix}wasteful_end_turn_selected_rate": selected_mean("combat_quality_wasteful_end_turn_selected"),
        f"{tag_prefix}true_wasteful_end_turn_available_rate": mean("combat_quality_true_wasteful_end_turn_available"),
        f"{tag_prefix}true_wasteful_end_turn_selected_rate": selected_mean("combat_quality_true_wasteful_end_turn_selected"),
        f"{tag_prefix}strategic_defer_available_rate": mean("combat_quality_strategic_defer_available"),
        f"{tag_prefix}strategic_defer_end_turn_selected_rate": selected_mean("combat_quality_strategic_defer_end_turn_selected"),
        f"{tag_prefix}bad_end_turn_available_rate": mean("combat_quality_bad_end_turn_available"),
        f"{tag_prefix}bad_end_turn_selected_rate": selected_mean("combat_quality_bad_end_turn_selected"),
        f"{tag_prefix}forced_end_turn_available_rate": mean("combat_quality_forced_end_turn_available"),
        f"{tag_prefix}forced_end_turn_selected_rate": selected_mean("combat_quality_forced_end_turn_selected"),
        f"{tag_prefix}end_turn_unknown_selected_rate": selected_mean("combat_quality_end_turn_unknown_selected"),
        f"{tag_prefix}direct_end_turn_selected_rate": selected_mean("combat_quality_end_turn_selected"),
        f"{tag_prefix}x_cost_available_count_mean": mean("combat_quality_x_cost_available_count"),
        f"{tag_prefix}x_cost_selected_rate": selected_mean("combat_quality_x_cost_selected"),
        f"{tag_prefix}x_cost_selected_energy_mean": (
            sum(_as_floats(search_values.get("combat_quality_x_cost_selected_energy")))
            / max(sum(_as_floats(search_values.get("combat_quality_x_cost_selected"))), 1.0)
        ),
        f"{tag_prefix}x_cost_zero_energy_selected_rate": selected_mean("combat_quality_x_cost_zero_energy_selected"),
        f"{tag_prefix}zero_energy_x_cost_available_mean": mean("combat_quality_zero_energy_x_cost_count"),
        f"{tag_prefix}zero_energy_x_cost_selected_rate": selected_mean("combat_quality_zero_energy_x_cost_selected"),
        f"{tag_prefix}x_cost_bad_available_count_mean": mean("combat_quality_x_cost_bad_count"),
        f"{tag_prefix}x_cost_effective_energy_mean": mean("combat_quality_x_cost_effective_energy_mean"),
        f"{tag_prefix}x_cost_selected_effective_energy_mean": mean("combat_quality_x_cost_selected_effective_energy"),
        f"{tag_prefix}x_cost_has_non_energy_effect_selected_rate": selected_mean("combat_quality_x_cost_has_non_energy_effect_selected"),
        f"{tag_prefix}x_cost_bad_selected_rate": selected_mean("combat_quality_x_cost_bad_selected"),
        f"{tag_prefix}x_cost_zero_bad_selected_rate_p0": selected_mean("combat_quality_x_cost_zero_bad_selected"),
        f"{tag_prefix}x_cost_zero_selected_rate_p0": selected_mean("combat_quality_x_cost_zero_selected"),
        f"{tag_prefix}x_cost_energy_value_mean_p0": mean("combat_quality_x_cost_energy_value"),
        f"{tag_prefix}x_cost_star_value_mean_p0": mean("combat_quality_x_cost_star_value"),
        f"{tag_prefix}star_x_selected_rate": selected_mean("combat_quality_star_x_selected"),
        f"{tag_prefix}x_cost_zero_guard_available_rate": mean("combat_quality_x_cost_zero_guard_available"),
        f"{tag_prefix}x_cost_zero_guard_applied_rate": mean("combat_quality_x_cost_zero_guard_applied"),
        f"{tag_prefix}x_cost_zero_guard_override_rate": mean("combat_quality_x_cost_zero_guard_override"),
        f"{tag_prefix}x_cost_zero_guard_no_alternative_rate": mean("combat_quality_x_cost_zero_guard_no_alternative"),
        f"{tag_prefix}x_cost_zero_guard_end_turn_fallback_rate": mean("combat_quality_x_cost_zero_guard_end_turn_fallback"),
        f"{tag_prefix}hp_cost_self_lethal_selected_rate": selected_mean("combat_quality_hp_cost_self_lethal_selected"),
        f"{tag_prefix}hp_cost_low_margin_selected_rate": selected_mean("combat_quality_hp_cost_low_margin_selected"),
        f"{tag_prefix}hp_cost_unblockable_value_mean": mean("combat_quality_hp_cost_unblockable_value"),
        f"{tag_prefix}hp_cost_margin_guard_available_rate": mean("combat_quality_hp_cost_margin_guard_available"),
        f"{tag_prefix}hp_cost_margin_guard_applied_rate": mean("combat_quality_hp_cost_margin_guard_applied"),
        f"{tag_prefix}hp_cost_margin_guard_override_rate": mean("combat_quality_hp_cost_margin_guard_override"),
        f"{tag_prefix}hp_cost_margin_guard_lethal_exemption_rate": mean("combat_quality_hp_cost_margin_guard_lethal_exemption"),
        f"{tag_prefix}hp_cost_margin_guard_no_alternative_rate": mean("combat_quality_hp_cost_margin_guard_no_alternative"),
        f"{tag_prefix}energy_gain_without_followup_count_mean": mean("combat_quality_energy_gain_without_followup_count"),
        f"{tag_prefix}typed_followup_missing_count_mean": mean("combat_quality_typed_followup_missing_count"),
        f"{tag_prefix}typed_future_penalty_count_mean": mean("combat_quality_typed_future_penalty_count"),
        f"{tag_prefix}typed_no_draw_count_mean": mean("combat_quality_typed_no_draw_count"),
        f"{tag_prefix}typed_card_state_mutation_count_mean": mean("combat_quality_typed_card_state_mutation_count"),
        f"{tag_prefix}setup_followup_dependent_count_mean": mean("combat_quality_setup_followup_dependent_count"),
        f"{tag_prefix}setup_followup_available_count_mean": mean("combat_quality_setup_followup_available_count"),
        f"{tag_prefix}refund_no_followup_available_mean": mean("combat_quality_refund_no_followup_available"),
        f"{tag_prefix}refund_no_followup_selected_rate": selected_mean("combat_quality_refund_no_followup_selected"),
        f"{tag_prefix}refund_no_followup_with_progress_selected_rate": selected_mean("combat_quality_refund_no_followup_with_progress_selected"),
        f"{tag_prefix}refund_no_followup_progress_alternative_selected_rate": selected_mean("combat_quality_refund_no_followup_progress_alternative_selected"),
        f"{tag_prefix}refund_no_followup_no_alternative_selected_rate": selected_mean("combat_quality_refund_no_followup_no_alternative_selected"),
        f"{tag_prefix}refund_no_followup_progress_alternative_count_mean": mean("combat_quality_refund_no_followup_progress_alternative_count"),
        f"{tag_prefix}refund_no_followup_guard_available_rate": mean("combat_quality_refund_no_followup_guard_available"),
        f"{tag_prefix}refund_no_followup_guard_applied_rate": mean("combat_quality_refund_no_followup_guard_applied"),
        f"{tag_prefix}refund_no_followup_guard_override_rate": mean("combat_quality_refund_no_followup_guard_override"),
        f"{tag_prefix}refund_no_followup_guard_no_alternative_rate": mean("combat_quality_refund_no_followup_guard_no_alternative"),
        f"{tag_prefix}refund_no_followup_guard_end_turn_fallback_rate": mean("combat_quality_refund_no_followup_guard_end_turn_fallback"),
        f"{tag_prefix}refund_no_followup_guard_lethal_exemption_rate": mean("combat_quality_refund_no_followup_guard_lethal_exemption"),
        f"{tag_prefix}refund_no_followup_guard_candidate_count_mean": mean("combat_quality_refund_no_followup_guard_candidate_count"),
        f"{tag_prefix}refund_no_followup_guard_lethal_candidate_rate": mean("combat_quality_refund_no_followup_guard_lethal_candidate"),
        f"{tag_prefix}strategic_skip_candidate_count_mean": mean("combat_quality_strategic_skip_candidate_count"),
        f"{tag_prefix}strategic_skip_selected_rate": selected_mean("combat_quality_strategic_skip_selected"),
        f"{tag_prefix}potion_available_count_mean": mean("combat_quality_potion_available_count"),
        f"{tag_prefix}potion_urgent_available_mean": mean("combat_quality_potion_urgent_count"),
        f"{tag_prefix}potion_low_urgency_available_mean": mean("combat_quality_potion_low_urgency_count"),
        f"{tag_prefix}potion_save_recommended_available_mean": mean("combat_quality_potion_save_recommended_count"),
        f"{tag_prefix}potion_no_followup_available_mean": mean("combat_quality_potion_no_followup_count"),
        f"{tag_prefix}potion_lethal_available_mean": mean("combat_quality_potion_lethal_count"),
        f"{tag_prefix}potion_prevent_lethal_available_mean": mean("combat_quality_potion_prevent_lethal_count"),
        f"{tag_prefix}potion_mechanism_available_mean": mean("combat_quality_potion_mechanism_count"),
        f"{tag_prefix}potion_overkill_available_mean": mean("combat_quality_potion_overkill_count"),
        f"{tag_prefix}potion_block_waste_available_mean": mean("combat_quality_potion_block_waste_count"),
        f"{tag_prefix}potion_use_quality_available_mean": mean("combat_quality_potion_use_quality_mean"),
        f"{tag_prefix}potion_waste_risk_available_mean": mean("combat_quality_potion_waste_risk_mean"),
        f"{tag_prefix}potion_save_value_available_mean": mean("combat_quality_potion_save_value_mean"),
        f"{tag_prefix}potion_hand_context_good_mean": mean("combat_quality_potion_hand_context_good_count"),
        f"{tag_prefix}potion_hand_context_bad_mean": mean("combat_quality_potion_hand_context_bad_count"),
        f"{tag_prefix}potion_long_term_mean": mean("combat_quality_potion_long_term_count"),
        f"{tag_prefix}potion_requires_followup_mean": mean("combat_quality_potion_requires_followup_count"),
        f"{tag_prefix}bad_potion_bias_count_mean": mean("combat_quality_bad_potion_bias_count"),
        f"{tag_prefix}bad_potion_bias_min": _min(search_values.get("combat_quality_bad_potion_bias_min")),
        f"{tag_prefix}potion_selected_rate": selected_mean("combat_quality_potion_selected"),
        f"{tag_prefix}potion_selected_when_available_rate": selected_mean("combat_quality_potion_selected_when_available"),
        f"{tag_prefix}potion_high_urgency_selected_rate": selected_mean("combat_quality_potion_high_urgency_selected"),
        f"{tag_prefix}potion_low_urgency_selected_rate": selected_mean("combat_quality_potion_low_urgency_selected"),
        f"{tag_prefix}potion_save_recommended_selected_rate": selected_mean("combat_quality_potion_save_recommended_selected"),
        f"{tag_prefix}potion_no_followup_selected_rate": selected_mean("combat_quality_potion_no_followup_selected"),
        f"{tag_prefix}potion_lethal_selected_rate": selected_mean("combat_quality_potion_lethal_selected"),
        f"{tag_prefix}potion_prevent_lethal_selected_rate": selected_mean("combat_quality_potion_prevent_lethal_selected"),
        f"{tag_prefix}potion_mechanism_selected_rate": selected_mean("combat_quality_potion_mechanism_selected"),
        f"{tag_prefix}potion_overkill_selected_rate": selected_mean("combat_quality_potion_overkill_selected"),
        f"{tag_prefix}potion_block_waste_selected_rate": selected_mean("combat_quality_potion_block_waste_selected"),
        f"{tag_prefix}potion_use_quality_selected_mean": (
            sum(_as_floats(search_values.get("combat_quality_potion_use_quality_selected")))
            / max(potion_selected_sum, 1.0)
        ),
        f"{tag_prefix}potion_waste_risk_selected_mean": (
            sum(_as_floats(search_values.get("combat_quality_potion_waste_risk_selected")))
            / max(potion_selected_sum, 1.0)
        ),
        f"{tag_prefix}potion_bad_guard_available_rate": mean("combat_quality_potion_bad_guard_available"),
        f"{tag_prefix}potion_bad_guard_applied_rate": mean("combat_quality_potion_bad_guard_applied"),
        f"{tag_prefix}potion_bad_guard_override_rate": mean("combat_quality_potion_bad_guard_override"),
        f"{tag_prefix}potion_bad_guard_lethal_exemption_rate": mean("combat_quality_potion_bad_guard_lethal_exemption"),
        f"{tag_prefix}potion_bad_guard_no_alternative_rate": mean("combat_quality_potion_bad_guard_no_alternative"),
        f"{tag_prefix}potion_bad_guard_forced_end_turn_hopeless_rate": mean("combat_quality_potion_bad_guard_forced_end_turn_hopeless"),
        f"{tag_prefix}potion_discard_guard_available_rate": mean("combat_quality_potion_discard_guard_available"),
        f"{tag_prefix}potion_discard_guard_applied_rate": mean("combat_quality_potion_discard_guard_applied"),
        f"{tag_prefix}potion_discard_guard_override_rate": mean("combat_quality_potion_discard_guard_override"),
        f"{tag_prefix}potion_discard_guard_candidate_count_mean": mean("combat_quality_potion_discard_guard_candidate_count"),
        f"{tag_prefix}potion_discard_guard_saved_survival_rate": mean("combat_quality_potion_discard_guard_saved_survival"),
    }


def boss_card_block_waste_metrics(search_values: SearchValues) -> dict[str, float]:
    """Aggregate card block-waste diagnostics for boss-combat TensorBoard tags."""

    return card_block_waste_metrics(search_values, prefix="boss_combat/")


__all__ = [
    "COMBAT_QUALITY_CARD_BLOCK_SEARCH_SUFFIXES",
    "COMBAT_QUALITY_GUARD_SEARCH_SUFFIXES",
    "COMBAT_QUALITY_TACTICAL_SEARCH_SUFFIXES",
    "boss_card_block_waste_metrics",
    "card_block_waste_metrics",
    "combat_tactical_issue_metrics",
]
