"""Card-selection loop hard guard for MuZero training."""

from __future__ import annotations

from collections import deque
from typing import Any


class SelectionLoopGuardMixin:
    def _apply_selection_loop_guard(self, *, action_idx: int, legal_count: int, legal_actions: list[Any], mask_np: Any, raw_obs: Any | None, encounter: str, search_stats: dict[str, Any]) -> int:
        # P2-2 (recovery 2026-05-07): Card-selection dead-loop guard.
        # When the policy revisits the same selection screen + same picked
        # option for ``_SELECTION_LOOP_STREAK_THRESHOLD`` consecutive steps
        # without progress, override to confirm / cancel / alternative pick.
        # Reset history when the selection family is left so a later
        # selection round starts fresh.
        if 0 <= int(action_idx) < legal_count and isinstance(legal_actions[int(action_idx)], dict):
            selected = legal_actions[int(action_idx)]
            selected_family = self._semantic_family(selected)
            history = getattr(self, "_selection_loop_history", None)
            if history is None:
                history = deque(maxlen=16)
                self._selection_loop_history = history
            if selected_family != "card_selection":
                if history:
                    history.clear()
            else:
                sig = self._selection_loop_signature(legal_actions, mask_np)
                if sig is not None:
                    search_stats["combat_quality_selection_loop_screen_active"] = 1.0
                    picked_card = (
                        selected.get("card") if isinstance(selected.get("card"), dict) else {}
                    )
                    picked_id = str(picked_card.get("id") or selected.get("action_id") or "")
                    sel_act = str(
                        selected.get("selection_action") or selected.get("selection") or ""
                    ).strip().lower()
                    history.append((sig, picked_id, sel_act))
                    same_pick_streak = 0
                    for past_sig, past_pick, past_act in reversed(history):
                        if past_sig == sig and past_pick == picked_id and past_act == sel_act:
                            same_pick_streak += 1
                        else:
                            break
                    if same_pick_streak >= self._SELECTION_LOOP_STREAK_THRESHOLD:
                        search_stats["combat_quality_selection_loop_detected"] = 1.0
                        search_stats["combat_quality_selection_repeated_same_option"] = 1.0
                        confirm_idx = -1
                        cancel_idx = -1
                        alt_pick_idx = -1
                        for idx in range(legal_count):
                            if idx == int(action_idx) or mask_np[idx] <= 0:
                                continue
                            alt = legal_actions[idx]
                            if not isinstance(alt, dict):
                                continue
                            if self._semantic_family(alt) != "card_selection":
                                continue
                            alt_act = str(
                                alt.get("selection_action") or alt.get("selection") or ""
                            ).strip().lower()
                            if alt_act == "confirm" and confirm_idx < 0:
                                confirm_idx = idx
                            elif alt_act in {"cancel", "close", "skip"} and cancel_idx < 0:
                                cancel_idx = idx
                            elif alt_act in {"pick", "select", ""}:
                                alt_card = (
                                    alt.get("card") if isinstance(alt.get("card"), dict) else {}
                                )
                                alt_card_id = str(alt_card.get("id") or alt.get("action_id") or "")
                                if alt_card_id and alt_card_id != picked_id and alt_pick_idx < 0:
                                    alt_pick_idx = idx
                        override_idx = -1
                        if confirm_idx >= 0:
                            override_idx = confirm_idx
                            search_stats["combat_quality_selection_loop_auto_confirm"] = 1.0
                        elif cancel_idx >= 0:
                            override_idx = cancel_idx
                            search_stats["combat_quality_selection_loop_auto_cancel"] = 1.0
                        elif alt_pick_idx >= 0:
                            override_idx = alt_pick_idx
                            search_stats["combat_quality_selection_loop_alt_pick"] = 1.0
                        else:
                            search_stats["combat_quality_selection_loop_no_alternative"] = 1.0
                        if override_idx >= 0:
                            self._dump_combat_hard_guard_record(
                                kind="selection_loop",
                                raw_obs=raw_obs,
                                legal_actions=legal_actions,
                                original_idx=int(action_idx),
                                override_idx=int(override_idx),
                                risk=float(same_pick_streak),
                                countdown=None,
                                encounter=encounter,
                                lethal_exemption=False,
                            )
                            action_idx = int(override_idx)
                            search_stats["combat_quality_selection_loop_applied"] = 1.0
                            history.clear()
        return int(action_idx)
