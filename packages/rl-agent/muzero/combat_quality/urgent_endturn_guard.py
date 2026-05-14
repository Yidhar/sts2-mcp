"""Urgent End Turn hard guard for stable combat frontiers.

This guard is deliberately narrower than the older EndTurn quality rules.  A
leftover-energy EndTurn is common and often correct after all useful cards have
already been played; it must not be treated as a bug.  The only failure mode we
patch here is the exploration-tail case observed in sandbox diagnostics:

* the sampled action is End Turn;
* the same stable legal-action frontier already contains a non-EndTurn action;
* that alternative is classified as urgent/positive (survival block, lethal,
  mechanism answer, urgent potion, etc.);
* the alternative is affordable and not a self-lethal / low-margin HP-cost play.

If no such candidate exists the guard leaves End Turn untouched, preserving
forced passes with an empty hand, status-only hands, or bridge/frontier moments
where End Turn is genuinely the only legal action.
"""

from __future__ import annotations

from typing import Any

from sts2_env.hp_cost_safety import hp_cost_safety_view


class UrgentEndTurnGuardMixin:
    """Prevent stochastic EndTurn samples when urgent legal actions exist."""

    @staticmethod
    def _urgent_endturn_guard_reset_endturn_stats(search_stats: dict[str, Any]) -> None:
        """Clear selected-EndTurn gauges after a successful override."""

        for key in (
            "combat_quality_wasteful_end_turn_selected",
            "combat_quality_true_wasteful_end_turn_selected",
            "combat_quality_bad_end_turn_selected",
            "combat_quality_end_turn_selected",
            "combat_quality_end_turn_unknown_selected",
            "combat_quality_strategic_defer_end_turn_selected",
        ):
            search_stats[key] = 0.0

    def _apply_urgent_endturn_guard(
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
        """Rewrite selected EndTurn only when urgent alternatives are present."""

        original_idx = int(action_idx)
        if not (0 <= original_idx < int(legal_count)):
            return original_idx
        selected = legal_actions[original_idx]
        if not isinstance(selected, dict) or self._semantic_family(selected) != "end_turn":
            return original_idx
        if not isinstance(raw_obs, dict):
            search_stats["combat_quality_urgent_endturn_guard_invalid_obs"] = 1.0
            return original_idx
        if int(legal_count) <= 1:
            search_stats["combat_quality_urgent_endturn_guard_forced_skip"] = 1.0
            return original_idx

        current_energy = float(self._combat_energy(None, raw_obs))
        incoming, current_block, current_hp = self._incoming_damage_pressure(raw_obs)
        threat_gap = max(0.0, float(incoming) - float(current_block))
        encounter_tier = self._combat_encounter_tier_from_raw(raw_obs)

        candidates: list[tuple[tuple[float, float, float, float, float, float], int, str]] = []
        for idx in range(int(legal_count)):
            if idx == original_idx:
                continue
            try:
                if mask_np[idx] <= 0:
                    continue
            except Exception:
                continue

            alt = legal_actions[idx]
            if not isinstance(alt, dict):
                continue
            family = self._semantic_family(alt)
            if family not in {"play_card", "use_potion", "potion"}:
                continue

            if family == "play_card":
                cost = float(self._action_cost_value(alt))
                if cost > current_energy + 1e-6:
                    continue
                x_diag = self._x_cost_diagnostic(alt, current_energy)
                if float(x_diag.get("x_cost_bad", 0.0) or 0.0) > 0.5:
                    continue
            else:
                cost = 0.0

            profile = self._classify_positive_combat_action(
                alt,
                int(idx),
                None,
                raw_obs,
                legal_actions,
                mask_np,
                current_energy,
            )
            lethal = bool(self._is_action_confirmed_lethal(alt, raw_obs))
            if not lethal and not bool(profile.get("positive", False)):
                continue

            safety = hp_cost_safety_view(alt, raw_obs)
            hp_loss = float(safety.get("hp_loss_unblockable", 0.0) or 0.0)
            if hp_loss > 0.0 and not lethal:
                if bool(safety.get("self_lethal_now", False)):
                    continue
                if bool(safety.get("low_hp_margin_after_cost", False)):
                    continue

            damage = max(
                self._action_metric(alt, "damage"),
                self._action_metric(alt, "total_damage"),
                self._action_numeric_value(
                    alt,
                    ("damage", "total_damage", "attack_damage", "preview_damage", "expected_damage"),
                ),
            )
            block = max(self._action_metric(alt, "block"), self._action_metric(alt, "total_block"))
            impact = float(self._action_immediate_impact(alt))

            potion_prevent_lethal = bool(profile.get("potion_prevent_lethal", False))
            potion_mechanism = bool(profile.get("potion_mechanism_answer", False))
            potion_lethal = bool(profile.get("potion_lethal", False))
            potion_urgent = bool(profile.get("potion_urgent", False))
            urgent = bool(profile.get("urgent", False))
            meaningful_block = bool(
                family == "play_card"
                and self._is_meaningful_block_urgent(
                    block=block,
                    threat_gap=threat_gap,
                    current_hp=current_hp,
                    incoming=incoming,
                    encounter_tier=encounter_tier,
                )
            )

            # Narrow trigger: a no-pressure leftover-energy pass should be
            # handled by the no-pressure/progress guards, not by this urgent
            # guard.  Lethal, ethereal/mechanism, and urgent potions remain
            # valid even when immediate incoming damage is absent.
            is_urgent_candidate = bool(
                lethal
                or potion_lethal
                or potion_prevent_lethal
                or potion_mechanism
                or potion_urgent
                or meaningful_block
                or (urgent and threat_gap > 0.05)
                or bool(profile.get("ethereal_urgent", False))
            )
            if not is_urgent_candidate:
                continue
            if bool(profile.get("card_block_waste", False)) and not (lethal or potion_prevent_lethal):
                continue
            if bool(profile.get("x_cost_zero", False)) and not lethal:
                continue
            if bool(profile.get("potion_low_urgency", False)) and not (
                potion_lethal or potion_prevent_lethal or potion_mechanism
            ):
                continue

            block_fill = min(float(block), float(threat_gap)) if threat_gap > 0.0 else 0.0
            score = (
                1.0 if (lethal or potion_lethal) else 0.0,
                1.0 if potion_prevent_lethal else 0.0,
                1.0 if (potion_mechanism or meaningful_block or urgent) else 0.0,
                float(block_fill),
                float(max(damage, impact)),
                -float(cost) - 1e-3 * float(idx),
            )
            candidates.append((score, int(idx), family))

        search_stats["combat_quality_urgent_endturn_guard_candidate_count"] = float(len(candidates))
        if not candidates:
            search_stats["combat_quality_urgent_endturn_guard_no_alternative"] = 1.0
            return original_idx

        candidates.sort(key=lambda item: item[0], reverse=True)
        _score, best_idx, best_family = candidates[0]
        search_stats["combat_quality_urgent_endturn_guard_available"] = 1.0
        search_stats["combat_quality_urgent_endturn_guard_threat_gap"] = float(threat_gap)
        if best_family in {"use_potion", "potion"}:
            search_stats["combat_quality_urgent_endturn_guard_potion_candidate"] = 1.0

        if int(best_idx) != original_idx:
            self._dump_combat_hard_guard_record(
                kind="urgent_endturn",
                raw_obs=raw_obs,
                legal_actions=legal_actions,
                original_idx=original_idx,
                override_idx=int(best_idx),
                risk=float(max(threat_gap, incoming)),
                countdown=None,
                encounter=encounter,
                lethal_exemption=False,
            )
            search_stats["combat_quality_urgent_endturn_guard_applied"] = 1.0
            search_stats["combat_quality_urgent_endturn_guard_override"] = 1.0
            search_stats["combat_quality_hard_guard_override_any"] = 1.0
            self._urgent_endturn_guard_reset_endturn_stats(search_stats)
            return int(best_idx)

        return original_idx


__all__ = ["UrgentEndTurnGuardMixin"]
