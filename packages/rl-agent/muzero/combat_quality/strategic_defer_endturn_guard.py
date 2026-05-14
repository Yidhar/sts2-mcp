"""Hard guard for strategic-defer End Turn under safe race pressure.

The existing ``MeaningfulDamageEndTurnGuardMixin`` only rewrites End Turn when
there is no meaningful incoming pressure.  Hard-normal sandbox diagnostics
show a second failure mode: the agent passes with energy while a safe progress
card is available, even though the uncovered hit is survivable and the fight is
turning into a race.

This guard is intentionally conservative:

* it only touches selected End Turn;
* it never fires on non-combat / invalid raw observations;
* urgent pressure is respected unless the post-hit HP window is still safe in a
  weak/normal hallway;
* the replacement action is selected by the shared
  ``collect_safe_progress_candidates`` predicate, so it will not choose pure
  block, bad zero-energy X-cost, non-lethal HP-cost, or no-followup setup.
"""

from __future__ import annotations

from typing import Any

from muzero.combat_quality.progress_candidates import collect_safe_progress_candidates


class StrategicDeferEndTurnGuardMixin:
    """Rewrite safe strategic-defer End Turn to immediate fight progress."""

    @staticmethod
    def _strategic_defer_endturn_guard_reset_endturn_stats(search_stats: dict[str, Any]) -> None:
        """Clear selected-EndTurn gauges after a successful override."""

        for key in (
            "combat_quality_wasteful_end_turn_selected",
            "combat_quality_true_wasteful_end_turn_selected",
            "combat_quality_bad_end_turn_selected",
            "combat_quality_strategic_defer_end_turn_selected",
            "combat_quality_end_turn_selected",
            "combat_quality_end_turn_unknown_selected",
        ):
            search_stats[key] = 0.0

    def _apply_strategic_defer_endturn_guard(
        self,
        *,
        action_idx: int,
        legal_count: int,
        legal_actions: list[Any],
        mask_np: Any,
        raw_obs: Any | None,
        encounter: str,
        search_stats: dict[str, Any],
    ) -> int:
        """Rewrite selected End Turn if a safe progress card should be played.

        ``meaningful_damage_endturn`` covers the trivial no-pressure case.  This
        guard runs after it and handles survivable pressure/race cases such as
        normal hallway fights where taking a small hit is preferable to passing
        with damage in hand.
        """

        original_idx = int(action_idx)
        if not (0 <= original_idx < int(legal_count)):
            return original_idx
        selected = legal_actions[original_idx]
        if not isinstance(selected, dict) or self._semantic_family(selected) != "end_turn":
            return original_idx
        if not isinstance(raw_obs, dict):
            return original_idx

        current_energy = float(self._combat_energy(None, raw_obs))
        incoming, current_block, current_hp = self._incoming_damage_pressure(raw_obs)
        threat_gap = max(0.0, float(incoming) - float(current_block))
        encounter_tier = self._combat_encounter_tier_from_raw(raw_obs)
        encounter_text = self._combat_encounter_text(raw_obs, encounter)
        hallway = self._is_normal_or_weak_hallway_encounter(encounter_tier, encounter_text)

        meaningful_urgent = self._is_meaningful_block_urgent(
            block=1.0,
            threat_gap=threat_gap,
            current_hp=current_hp,
            incoming=incoming,
            encounter_tier=encounter_tier,
        )

        no_pressure = threat_gap <= 0.05
        post_hit_hp = float(current_hp) - float(threat_gap)
        safe_pressure_window = bool(
            hallway
            and threat_gap > 0.05
            and current_hp > 0.0
            and post_hit_hp >= max(18.0, 0.25 * float(current_hp))
        )

        if meaningful_urgent and not safe_pressure_window:
            search_stats["combat_quality_strategic_defer_endturn_guard_pressure_skip"] = 1.0
            return original_idx

        candidates, _rejections = collect_safe_progress_candidates(
            self,
            selected_idx=original_idx,
            legal_count=int(legal_count),
            legal_actions=legal_actions,
            mask_np=mask_np,
            raw_obs=raw_obs,
            current_energy=current_energy,
            use_mask=True,
            include_debug=False,
        )
        search_stats["combat_quality_strategic_defer_endturn_guard_candidate_count"] = float(len(candidates))
        if not candidates:
            search_stats["combat_quality_strategic_defer_endturn_guard_no_alternative"] = 1.0
            return original_idx

        search_stats["combat_quality_strategic_defer_endturn_guard_available"] = 1.0
        if safe_pressure_window:
            search_stats["combat_quality_strategic_defer_endturn_guard_safe_pressure"] = 1.0

        best = candidates[0]
        best_idx = int(best.index)
        if best_idx == original_idx:
            return original_idx

        self._dump_combat_hard_guard_record(
            kind="strategic_defer_endturn",
            raw_obs=raw_obs,
            legal_actions=legal_actions,
            original_idx=original_idx,
            override_idx=best_idx,
            risk=float(max(best.damage, best.impact, threat_gap)),
            countdown=None,
            encounter=encounter,
            lethal_exemption=False,
        )
        search_stats["combat_quality_strategic_defer_endturn_guard_applied"] = 1.0
        search_stats["combat_quality_strategic_defer_endturn_guard_override"] = 1.0
        search_stats["combat_quality_hard_guard_override_any"] = 1.0
        self._strategic_defer_endturn_guard_reset_endturn_stats(search_stats)
        return best_idx


__all__ = ["StrategicDeferEndTurnGuardMixin"]
