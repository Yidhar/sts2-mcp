"""Global no-pressure EndTurn hard guard for combat quality.

This module intentionally owns the tactical rule instead of growing
``muzero.train`` or the hard-guard orchestrator.  The guard only rewrites an
End Turn decision when the combat state has no meaningful incoming pressure and
a safe, affordable card can make immediate damage/progress.  It deliberately
avoids pure block waste, setup cards with no follow-up, bad zero-energy X-cost
plays, and non-lethal HP-cost cards.
"""

from __future__ import annotations

from typing import Any

from sts2_env.hp_cost_safety import hp_cost_safety_view


class MeaningfulDamageEndTurnGuardMixin:
    """Prevent empty passes when a safe meaningful play-card exists."""

    @staticmethod
    def _meaningful_damage_guard_reset_endturn_stats(search_stats: dict[str, Any]) -> None:
        """Clear selected-EndTurn flags after this guard rewrites the action."""

        for key in (
            "combat_quality_wasteful_end_turn_selected",
            "combat_quality_true_wasteful_end_turn_selected",
            "combat_quality_bad_end_turn_selected",
            "combat_quality_end_turn_selected",
        ):
            search_stats[key] = 0.0

    def _apply_meaningful_damage_endturn_guard(
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
        """Rewrite no-pressure EndTurn to a safe damage/progress card.

        This is intentionally global across weak/normal/elite/boss combats but
        conservative: if there is meaningful incoming pressure, let the
        survival-specific guards reason about block/potion choices instead.
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

        if self._is_meaningful_block_urgent(
            block=1.0,
            threat_gap=threat_gap,
            current_hp=current_hp,
            incoming=incoming,
            encounter_tier=encounter_tier,
        ):
            search_stats["combat_quality_meaningful_damage_endturn_guard_pressure_skip"] = 1.0
            return original_idx

        candidates: list[tuple[tuple[float, float, float, float, float], int, float, float, bool]] = []
        for idx in range(int(legal_count)):
            if idx == original_idx:
                continue
            try:
                if mask_np[idx] <= 0:
                    continue
            except Exception:
                continue
            alt = legal_actions[idx]
            if not isinstance(alt, dict) or self._semantic_family(alt) != "play_card":
                continue

            cost = float(self._action_cost_value(alt))
            if cost > current_energy + 1e-6:
                continue

            x_diag = self._x_cost_diagnostic(alt, current_energy)
            if float(x_diag.get("x_cost_bad", 0.0) or 0.0) > 0.5:
                continue

            lethal = bool(self._is_action_confirmed_lethal(alt, raw_obs))
            safety = hp_cost_safety_view(alt, raw_obs)
            hp_loss = float(safety.get("hp_loss_unblockable", 0.0) or 0.0)
            if hp_loss > 0.0 and not lethal:
                continue

            profile = self._classify_positive_combat_action(
                alt,
                int(idx),
                None,
                raw_obs,
                legal_actions,
                mask_np,
                current_energy,
            )
            if not bool(profile.get("positive", False)):
                continue
            if bool(profile.get("card_block_waste", False)) or bool(profile.get("card_pure_block", False)):
                continue

            damage = max(
                self._action_metric(alt, "damage"),
                self._action_metric(alt, "total_damage"),
                self._action_numeric_value(
                    alt,
                    ("damage", "total_damage", "attack_damage", "preview_damage", "expected_damage"),
                ),
            )
            impact = float(self._action_immediate_impact(alt))
            roles = self._action_roles(alt)
            progress_roles = bool(
                roles.intersection(
                    {
                        "attack",
                        "damage",
                        "debuff",
                        "weak",
                        "vulnerable",
                        "poison",
                        "stun",
                        "artifact_strip",
                        "lock",
                        "mechanism",
                        "facing_change",
                        "scaling",
                        "power",
                    }
                )
            )
            high_damage = bool(damage >= 8.0 or impact >= 8.0)
            meaningful_progress = bool(lethal or damage > 0.0 or impact >= 3.0 or progress_roles)
            if not meaningful_progress:
                continue

            if bool(profile.get("deferable", False)) and not (lethal or high_damage):
                continue
            if bool(profile.get("followup_missing", False)) and not (lethal or high_damage):
                continue
            if bool(profile.get("energy_without_followup", False)) and not (lethal or high_damage):
                continue
            if (
                bool(profile.get("setup_followup_dependent", False))
                and not bool(profile.get("setup_followup_available", False))
                and not (lethal or high_damage)
            ):
                continue

            score = (
                1.0 if lethal else 0.0,
                float(damage),
                float(impact),
                -float(cost),
                -float(idx),
            )
            candidates.append((score, int(idx), float(damage), float(impact), lethal))

        search_stats["combat_quality_meaningful_damage_endturn_guard_candidate_count"] = float(len(candidates))
        if not candidates:
            search_stats["combat_quality_meaningful_damage_endturn_guard_no_alternative"] = 1.0
            return original_idx

        candidates.sort(key=lambda item: item[0], reverse=True)
        _score, best_idx, best_damage, best_impact, best_lethal = candidates[0]
        search_stats["combat_quality_meaningful_damage_endturn_guard_available"] = 1.0
        if best_lethal:
            search_stats["combat_quality_meaningful_damage_endturn_guard_lethal_candidate"] = 1.0

        if int(best_idx) != original_idx:
            self._dump_combat_hard_guard_record(
                kind="meaningful_damage_endturn",
                raw_obs=raw_obs,
                legal_actions=legal_actions,
                original_idx=original_idx,
                override_idx=int(best_idx),
                risk=float(max(best_damage, best_impact)),
                countdown=None,
                encounter=encounter,
                lethal_exemption=False,
            )
            search_stats["combat_quality_meaningful_damage_endturn_guard_applied"] = 1.0
            search_stats["combat_quality_meaningful_damage_endturn_guard_override"] = 1.0
            search_stats["combat_quality_hard_guard_override_any"] = 1.0
            self._meaningful_damage_guard_reset_endturn_stats(search_stats)
            return int(best_idx)

        return original_idx
