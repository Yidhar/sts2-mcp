"""Full-energy EndTurn hard guard for stable combat frontiers.

This guard targets the specific live failure reported in full-run play:
``EndTurn`` is dispatched while the bridge still reports full energy and the
same legal frontier already contains a safe, affordable progress card.  It is
intentionally narrower than broad "leftover energy" rules:

* it only fires for a selected ``end_turn`` action;
* it only rewrites when energy is effectively full;
* it only picks from the current legal-action mask;
* it reuses :func:`collect_safe_progress_candidates` so the definition of
  "safe progress" is shared with diagnostics and other hard guards;
* if the raw hand looks playable but legal actions do not expose a play-card
  action, it only marks a legal-generation gap and leaves the action alone.

The final bullet is important: bridge/frontier mismatches must be diagnosed,
not papered over by fabricating an illegal play-card action.
"""

from __future__ import annotations

from typing import Any

from muzero.combat_quality.progress_candidates import collect_safe_progress_candidates
from muzero.diagnostics.end_turn_pre_dispatch import (
    count_ui_affordable_hand_cards,
    raw_max_energy,
)


_SELECTION_PHASE_MARKERS = frozenset(
    {
        "select",
        "selection",
        "card_selection",
        "select_card",
        "choose_card",
        "choose_cards",
        "discard",
        "exhaust",
        "upgrade",
        "transform",
        "remove",
        "target",
        "select_target",
        "combat_select",
        "modal",
    }
)


class FullEnergyEndTurnGuardMixin:
    """Prevent full-energy EndTurn when a safe progress action is legal now."""

    @staticmethod
    def _full_energy_endturn_guard_reset_endturn_stats(search_stats: dict[str, Any]) -> None:
        """Clear selected-EndTurn gauges after a successful override."""

        for key in (
            "combat_quality_wasteful_end_turn_selected",
            "combat_quality_true_wasteful_end_turn_selected",
            "combat_quality_bad_end_turn_selected",
            "combat_quality_forced_end_turn_selected",
            "combat_quality_strategic_defer_end_turn_selected",
            "combat_quality_end_turn_selected",
            "combat_quality_end_turn_unknown_selected",
            "combat_quality_full_energy_nonurgent_end_turn_selected",
            "combat_quality_safe_progress_skip_selected",
        ):
            search_stats[key] = 0.0

    @staticmethod
    def _full_energy_endturn_guard_selection_surface(raw_obs: Any | None) -> bool:
        """Return true for modal/card-selection style combat sub-surfaces."""

        if not isinstance(raw_obs, dict):
            return False
        parts: list[str] = []
        for key in ("phase", "surface", "selection", "screen", "screen_id", "screenId"):
            value = raw_obs.get(key)
            if value is not None:
                parts.append(str(value))
        combat = raw_obs.get("combat")
        if isinstance(combat, dict):
            for key in ("phase", "surface", "selection", "screen", "screen_id", "screenId"):
                value = combat.get(key)
                if value is not None:
                    parts.append(str(value))
        text = " ".join(parts).strip().lower()
        if not text:
            return False
        return any(marker in text for marker in _SELECTION_PHASE_MARKERS)

    @staticmethod
    def _full_energy_endturn_guard_mask_allows(mask_np: Any, idx: int) -> bool:
        try:
            return bool(mask_np[int(idx)] > 0)
        except Exception:
            return False

    def _apply_full_energy_endturn_guard(
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
        """Rewrite selected EndTurn only on full-energy safe-progress skips."""

        original_idx = int(action_idx)
        if not (0 <= original_idx < int(legal_count)):
            return original_idx
        selected = legal_actions[original_idx]
        if not isinstance(selected, dict) or self._semantic_family(selected) != "end_turn":
            return original_idx
        if not isinstance(raw_obs, dict):
            search_stats["combat_quality_full_energy_endturn_guard_invalid_obs"] = 1.0
            return original_idx
        if self._full_energy_endturn_guard_selection_surface(raw_obs):
            search_stats["combat_quality_full_energy_endturn_guard_selection_screen_skip"] = 1.0
            return original_idx
        if int(legal_count) <= 1:
            search_stats["combat_quality_full_energy_endturn_guard_forced_skip"] = 1.0
            return original_idx

        try:
            energy = float(self._combat_energy(None, raw_obs))
        except Exception:
            energy = 0.0
        max_energy = raw_max_energy(raw_obs, default=max(3.0, energy))
        try:
            energy_ratio = float(energy) / max(float(max_energy), 1.0)
        except Exception:
            energy_ratio = 0.0
        full_energy_like = bool(energy_ratio >= 0.95 or float(energy) >= float(max_energy) - 1e-6)
        search_stats["combat_quality_full_energy_endturn_guard_energy"] = float(energy)
        search_stats["combat_quality_full_energy_endturn_guard_energy_ratio"] = float(energy_ratio)
        search_stats["combat_quality_full_energy_endturn_guard_full_energy_selected"] = (
            1.0 if full_energy_like else 0.0
        )

        legal_play_card_count = 0
        affordable_play_card_count = 0
        for idx in range(int(legal_count)):
            if not self._full_energy_endturn_guard_mask_allows(mask_np, idx):
                continue
            action = legal_actions[idx]
            if not isinstance(action, dict) or self._semantic_family(action) != "play_card":
                continue
            legal_play_card_count += 1
            try:
                cost = float(self._action_cost_value(action))
            except Exception:
                cost = 0.0
            if cost <= float(energy) + 1e-6:
                affordable_play_card_count += 1

        try:
            ui_affordable_count, hand_count = count_ui_affordable_hand_cards(raw_obs, float(energy))
        except Exception:
            ui_affordable_count, hand_count = 0, 0
        search_stats["combat_quality_full_energy_endturn_guard_ui_affordable_count"] = float(ui_affordable_count)
        search_stats["combat_quality_full_energy_endturn_guard_legal_play_card_count"] = float(
            legal_play_card_count
        )
        search_stats["combat_quality_full_energy_endturn_guard_affordable_play_card_count"] = float(
            affordable_play_card_count
        )

        if ui_affordable_count > 0 and legal_play_card_count <= 0:
            search_stats["combat_quality_full_energy_endturn_guard_legal_generation_gap_suspect"] = 1.0
            search_stats["combat_quality_full_energy_endturn_guard_legal_surface_mismatch_skip"] = 1.0
            return original_idx

        if not full_energy_like:
            return original_idx

        try:
            incoming, current_block, _current_hp = self._incoming_damage_pressure(raw_obs)
            if float(incoming) > float(current_block) + 0.5:
                search_stats["combat_quality_full_energy_endturn_guard_pressure_skip"] = 1.0
        except Exception:
            incoming = 0.0
            current_block = 0.0

        try:
            candidates, _debug_counts = collect_safe_progress_candidates(
                self,
                selected_idx=original_idx,
                legal_count=int(legal_count),
                legal_actions=legal_actions,
                mask_np=mask_np,
                raw_obs=raw_obs,
                current_energy=float(energy),
                use_mask=True,
                include_debug=True,
            )
        except Exception:
            candidates = []

        search_stats["combat_quality_full_energy_endturn_guard_candidate_count"] = float(len(candidates))
        if not candidates:
            search_stats["combat_quality_full_energy_endturn_guard_no_alternative"] = 1.0
            return original_idx

        best = candidates[0]
        best_idx = int(best.index)
        if not self._full_energy_endturn_guard_mask_allows(mask_np, best_idx):
            search_stats["combat_quality_full_energy_endturn_guard_illegal_candidate_reject"] = 1.0
            return original_idx
        search_stats["combat_quality_full_energy_endturn_guard_available"] = 1.0
        if bool(best.lethal):
            search_stats["combat_quality_full_energy_endturn_guard_lethal_candidate"] = 1.0

        if best_idx != original_idx:
            self._dump_combat_hard_guard_record(
                kind="full_energy_endturn",
                raw_obs=raw_obs,
                legal_actions=legal_actions,
                original_idx=original_idx,
                override_idx=best_idx,
                risk=float(max(float(incoming), float(energy))),
                countdown=None,
                encounter=encounter,
                lethal_exemption=False,
                extra={
                    "energy": float(energy),
                    "max_energy": float(max_energy),
                    "energy_ratio": float(energy_ratio),
                    "ui_affordable_hand_card_count": int(ui_affordable_count),
                    "raw_hand_card_count": int(hand_count),
                    "candidate_count": int(len(candidates)),
                    "candidate_title": str(best.title or ""),
                    "candidate_damage": float(best.damage),
                    "candidate_impact": float(best.impact),
                    "candidate_cost": float(best.cost),
                    "candidate_lethal": bool(best.lethal),
                },
            )
            search_stats["combat_quality_full_energy_endturn_guard_applied"] = 1.0
            search_stats["combat_quality_full_energy_endturn_guard_override"] = 1.0
            search_stats["combat_quality_hard_guard_override_any"] = 1.0
            self._full_energy_endturn_guard_reset_endturn_stats(search_stats)
            return best_idx

        return original_idx


__all__ = ["FullEnergyEndTurnGuardMixin"]
