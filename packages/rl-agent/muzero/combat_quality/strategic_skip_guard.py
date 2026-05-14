"""Hard guard for low-value strategic-skip/setup card selections.

Combat sandbox diagnostics have repeatedly shown selections such as setup,
resource, draw/discard, or no-followup cards being played when a safe immediate
progress card exists.  The existing refund/no-followup guard covers narrow
energy/refund cases; this guard covers the broader "strategic skip" family
without falling back to End Turn.

The guard only rewrites to candidates accepted by
``collect_safe_progress_candidates``.  That shared predicate is deliberately
strict and prevents this guard from choosing pure block, unsafe HP-cost cards,
bad X-cost cards, or another no-followup setup card.
"""

from __future__ import annotations

from typing import Any

from muzero.combat_quality.progress_candidates import collect_safe_progress_candidates


class StrategicSkipGuardMixin:
    """Rewrite selected low-value setup/resource plays to safe progress."""

    _STRATEGIC_SKIP_SETUP_ROLES = frozenset({"setup", "resource", "energy", "draw", "discard"})
    _STRATEGIC_SKIP_LONGTERM_ROLES = frozenset({"power", "scaling"})

    @staticmethod
    def _strategic_skip_guard_reset_selected_stats(search_stats: dict[str, Any]) -> None:
        """Clear selected-side setup/no-followup gauges after an override."""

        for key in (
            "combat_quality_strategic_skip_selected",
            "combat_quality_refund_no_followup_selected",
            "combat_quality_refund_no_followup_with_progress_selected",
            "combat_quality_refund_no_followup_progress_alternative_selected",
            "combat_quality_refund_no_followup_no_alternative_selected",
            "combat_quality_refund_no_followup_progress_alternative_count",
        ):
            search_stats[key] = 0.0

    def _strategic_skip_guard_selected_bad(
        self,
        selected: dict[str, Any],
        selected_profile: dict[str, Any],
        raw_obs: dict[str, Any],
        encounter: str,
    ) -> bool:
        """Return true for selected setup/resource plays worth overriding."""

        profile = selected_profile if isinstance(selected_profile, dict) else {}
        if bool(profile.get("deferable", False)):
            return True
        if bool(profile.get("followup_missing", False)):
            return True
        if bool(profile.get("energy_without_followup", False)):
            return True
        if (
            bool(profile.get("setup_followup_dependent", False))
            and not bool(profile.get("setup_followup_available", False))
        ):
            return True
        if bool(profile.get("typed_future_penalty", False)):
            return True
        if bool(profile.get("typed_no_draw", False)):
            return True
        if bool(profile.get("typed_card_state_mutation", False)) and not bool(
            profile.get("setup_followup_available", False)
        ):
            return True

        try:
            roles = set(self._action_roles(selected))
        except Exception:
            roles = set()
        if not roles:
            return False

        damage = max(
            float(self._action_metric(selected, "damage") or 0.0),
            float(self._action_metric(selected, "total_damage") or 0.0),
            float(
                self._action_numeric_value(
                    selected,
                    ("damage", "total_damage", "attack_damage", "preview_damage", "expected_damage"),
                )
                or 0.0
            ),
        )
        block = max(
            float(self._action_metric(selected, "block") or 0.0),
            float(self._action_metric(selected, "total_block") or 0.0),
            float(
                self._action_numeric_value(
                    selected,
                    ("block", "total_block", "preview_block", "expected_block"),
                )
                or 0.0
            ),
        )
        heal = max(
            float(self._action_metric(selected, "heal") or 0.0),
            float(self._action_numeric_value(selected, ("heal", "healing", "hp_gain")) or 0.0),
        )
        impact = float(self._action_immediate_impact(selected) or 0.0)
        if damage > 0.05 or block > 0.05 or heal > 0.05 or impact > 3.0:
            return False

        setup_like = bool(roles.intersection(self._STRATEGIC_SKIP_SETUP_ROLES))
        longterm_like = bool(roles.intersection(self._STRATEGIC_SKIP_LONGTERM_ROLES))
        if setup_like:
            return True

        # Scaling/power cards can be correct in bosses/elites.  Treat a low
        # immediate long-term card as bad by default only in hallway combats;
        # boss/elite uses still require one of the profile no-followup flags
        # above.
        if longterm_like:
            encounter_tier = self._combat_encounter_tier_from_raw(raw_obs)
            encounter_text = self._combat_encounter_text(raw_obs, encounter)
            return bool(self._is_normal_or_weak_hallway_encounter(encounter_tier, encounter_text))
        return False

    def _apply_strategic_skip_guard(
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
        """Rewrite selected low-value setup/resource card when possible."""

        original_idx = int(action_idx)
        if not (0 <= original_idx < int(legal_count)):
            return original_idx
        selected = legal_actions[original_idx]
        if not isinstance(selected, dict) or self._semantic_family(selected) != "play_card":
            return original_idx
        if not isinstance(raw_obs, dict):
            return original_idx

        current_energy = float(self._combat_energy(None, raw_obs))
        try:
            selected_profile = self._classify_positive_combat_action(
                selected,
                original_idx,
                None,
                raw_obs,
                legal_actions,
                mask_np,
                current_energy,
            )
        except Exception:
            selected_profile = {}
        if not self._strategic_skip_guard_selected_bad(selected, selected_profile, raw_obs, encounter):
            return original_idx

        search_stats["combat_quality_strategic_skip_guard_available"] = 1.0

        if bool(self._is_action_confirmed_lethal(selected, raw_obs)):
            search_stats["combat_quality_strategic_skip_guard_lethal_exemption"] = 1.0
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
        search_stats["combat_quality_strategic_skip_guard_candidate_count"] = float(len(candidates))
        if not candidates:
            search_stats["combat_quality_strategic_skip_guard_no_alternative"] = 1.0
            return original_idx

        best = candidates[0]
        best_idx = int(best.index)
        if best_idx == original_idx:
            return original_idx

        self._dump_combat_hard_guard_record(
            kind="strategic_skip",
            raw_obs=raw_obs,
            legal_actions=legal_actions,
            original_idx=original_idx,
            override_idx=best_idx,
            risk=float(max(best.damage, best.impact)),
            countdown=None,
            encounter=encounter,
            lethal_exemption=False,
        )
        search_stats["combat_quality_strategic_skip_guard_applied"] = 1.0
        search_stats["combat_quality_strategic_skip_guard_override"] = 1.0
        search_stats["combat_quality_hard_guard_override_any"] = 1.0
        self._strategic_skip_guard_reset_selected_stats(search_stats)
        return best_idx


__all__ = ["StrategicSkipGuardMixin"]
