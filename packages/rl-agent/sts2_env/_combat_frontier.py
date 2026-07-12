"""Combat-sandbox reset, actionability, and frontier recovery behavior."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from ._combat_env_values import BLOCKED_ACTION_KINDS, _float
from .bridge_client import BridgeError
from .observation_common import MAX_ACTIONS


class CombatFrontierMixin:
    def _sample_encounter_id(self) -> str | None:
        if self.encounter_pool:
            idx = int(self.np_random.integers(len(self.encounter_pool)))
            return self.encounter_pool[idx]
        return self.encounter_id

    def _try_salvage_card_selection_reset(self, exc: BridgeError) -> dict[str, Any] | None:
        body = exc.response_body if isinstance(exc.response_body, dict) else {}
        if exc.status_code != 409:
            return None
        if str(body.get("error") or "").strip() != "combat_sandbox_not_in_combat":
            return None
        details = body.get("details") if isinstance(body.get("details"), dict) else {}
        screen = str(details.get("screen") or "").strip().upper()
        phase = str(details.get("phase") or "").strip().lower()
        actionable = bool(details.get("actionable"))
        combat_in_progress = bool(details.get("combat_in_progress"))
        if screen not in {"COMBAT", "CARD_SELECTION"} or not combat_in_progress:
            return None
        if phase not in {"card_selection", "combat", "settling"}:
            return None
        if not actionable and phase != "settling":
            return None

        rebound = self._backend_reset(
            rebind_active_run=True,
            timeout_ms=self.reset_timeout_ms,
        )
        info = rebound.get("info")
        if not isinstance(info, dict):
            info = {}
            rebound["info"] = info
        info["combat_reset_salvaged"] = True
        info["combat_reset_salvage_phase"] = phase
        info["combat_reset_salvage_screen"] = screen
        info["combat_reset_salvage_actionable"] = actionable
        return rebound


    def _update_live_state(self, result: dict[str, Any]) -> None:
        legal_actions = result.get("legal_actions", [])
        if isinstance(legal_actions, list):
            self._legal_actions = [
                action for action in legal_actions
                if not (
                    isinstance(action, dict) and
                    str(action.get("kind") or "").strip() in BLOCKED_ACTION_KINDS
                )
            ]
        else:
            self._legal_actions = []
        obs = result.get("obs", {})
        self._last_obs_raw = obs if isinstance(obs, dict) else {}
        self._decorate_sandbox_raw_obs_context()
        self._last_action_overflow = max(len(self._legal_actions) - MAX_ACTIONS, 0)

    def _decorate_sandbox_raw_obs_context(self) -> None:
        """Expose injected combat-sandbox context without rewriting bridge state.

        Combat sandbox episodes run from injected pre-combat snapshots.  The
        live bridge observation may still report a synthetic/low ``run.floor``,
        while the snapshot carries the real full-run floor and sample identity.
        Tactical guards must be able to see that context, but other consumers
        still need the raw bridge floor.  Therefore we only add explicit
        ``snapshot_*`` fields and never overwrite ``run.floor``.
        """

        if not isinstance(self._last_obs_raw, dict):
            return

        if self._current_encounter_id:
            self._last_obs_raw.setdefault("encounter_id", self._current_encounter_id)
            self._last_obs_raw.setdefault("snapshot_encounter_id", self._current_encounter_id)

        snap = self._current_snapshot if isinstance(self._current_snapshot, dict) else {}
        floor = snap.get("floor_number")
        if floor is not None:
            self._last_obs_raw.setdefault("snapshot_floor_number", floor)
            run = self._last_obs_raw.setdefault("run", {})
            if isinstance(run, dict):
                run.setdefault("snapshot_floor_number", floor)

        sample_id = snap.get("sample_id")
        if sample_id is not None:
            self._last_obs_raw.setdefault("snapshot_sample_id", sample_id)


    def _current_encounter_tier(self) -> str:
        encounter_id = str(self._current_encounter_id or "").lower()
        if "boss" in encounter_id:
            return "boss"
        if "elite" in encounter_id:
            return "elite"
        if "weak" in encounter_id:
            return "weak"
        return "normal" if encounter_id else "unknown"

    @staticmethod
    def _action_family(action: dict[str, Any] | None) -> str:
        if not isinstance(action, dict):
            return ""
        semantic = action.get("semantic") if isinstance(action.get("semantic"), dict) else {}
        for key in ("family", "action_kind", "kind", "type"):
            value = semantic.get(key) if key in semantic else action.get(key)
            if value:
                return str(value).strip().lower()
        action_id = str(action.get("action_id") or "").lower()
        if "potion" in action_id:
            return "use_potion"
        return ""

    @staticmethod
    def _is_stable_actionability(actionability: dict[str, Any] | None) -> bool:
        """Frontier is stable when bridge says non-end-turn actions exist OR
        the only-end-turn frame is genuinely settled (not transient).

        Mirrors the C1 bridge payload contract: returning True means the
        Python side can stop short-polling and accept the current frontier.
        """
        if not isinstance(actionability, dict):
            return True  # No actionability info → assume stable to avoid hangs.
        if int(actionability.get("legal_non_end_turn_count", 0) or 0) > 0:
            return True
        if bool(actionability.get("transient_only_end_turn", False)):
            return False
        # Only end_turn AND not transient → ``stable_no_actions``.
        return bool(actionability.get("frontier_stable", True))

    @staticmethod
    def wait_for_stable_actionability(
        initial_result: dict[str, Any],
        observe_fn: Callable[[], dict[str, Any]],
        *,
        max_wait_ms: int = 100,
        poll_interval_ms: int = 10,
        sleep_fn: Callable[[float], None] | None = None,
        clock_fn: Callable[[], float] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Short-poll the bridge until the post-step frontier is stable.

        Pure helper so a fake bridge can drive it in tests.  The function
        returns ``(final_result, fast_step_metrics)`` where metrics contains
        ``wait_ms``, ``poll_count``, ``timeout``, ``transient_resolved``,
        ``transient_leaked``, ``stable_no_actions``.
        """

        sleep = sleep_fn if sleep_fn is not None else time.sleep
        clock = clock_fn if clock_fn is not None else time.perf_counter
        info0 = initial_result.get("info") if isinstance(initial_result.get("info"), dict) else {}
        actionability0 = info0.get("actionability") if isinstance(info0.get("actionability"), dict) else None
        # Stable on first frame: no waiting at all.
        metrics: dict[str, Any] = {
            "wait_ms": 0.0,
            "poll_count": 0,
            "timeout": False,
            "transient_resolved": False,
            "transient_leaked": False,
            "stable_no_actions": False,
        }
        if not bool((actionability0 or {}).get("transient_only_end_turn", False)):
            if (
                isinstance(actionability0, dict)
                and int(actionability0.get("legal_non_end_turn_count", 0) or 0) == 0
                and not bool(actionability0.get("transient_only_end_turn", False))
            ):
                metrics["stable_no_actions"] = True
            return initial_result, metrics

        deadline = clock() + (max_wait_ms / 1000.0)
        interval_s = max(poll_interval_ms / 1000.0, 0.001)
        current = initial_result
        while clock() < deadline:
            sleep(interval_s)
            metrics["poll_count"] += 1
            try:
                observed = observe_fn()
            except Exception:
                break
            if not isinstance(observed, dict):
                continue
            current = observed
            obs_info = observed.get("info") if isinstance(observed.get("info"), dict) else {}
            obs_actionability = obs_info.get("actionability") if isinstance(obs_info.get("actionability"), dict) else None
            if CombatFrontierMixin._is_stable_actionability(obs_actionability):
                metrics["wait_ms"] = max((max_wait_ms / 1000.0 - max(deadline - clock(), 0.0)) * 1000.0, 0.0)
                metrics["transient_resolved"] = (
                    int((obs_actionability or {}).get("legal_non_end_turn_count", 0) or 0) > 0
                )
                metrics["stable_no_actions"] = not metrics["transient_resolved"]
                return current, metrics
        metrics["wait_ms"] = float(max_wait_ms)
        metrics["timeout"] = True
        metrics["transient_leaked"] = True
        return current, metrics

    def _fast_step_metric_inc(self, key: str, amount: int = 1) -> None:
        metrics = getattr(self, "_fast_step_metrics_total", None)
        if not isinstance(metrics, dict):
            return
        try:
            metrics[key] = int(metrics.get(key, 0) or 0) + int(amount)
        except Exception:
            metrics[key] = int(amount)


    @staticmethod
    def _result_obs(result: dict[str, Any] | None) -> dict[str, Any]:
        obs = result.get("obs") if isinstance(result, dict) else None
        return obs if isinstance(obs, dict) else {}

    @staticmethod
    def _result_actionability(result: dict[str, Any] | None) -> dict[str, Any] | None:
        info = result.get("info") if isinstance(result, dict) and isinstance(result.get("info"), dict) else {}
        actionability = info.get("actionability") if isinstance(info.get("actionability"), dict) else None
        return actionability if isinstance(actionability, dict) else None

    @staticmethod
    def _filtered_result_legal_actions(result: dict[str, Any] | None) -> list[dict[str, Any]]:
        legal_actions = result.get("legal_actions") if isinstance(result, dict) else None
        if not isinstance(legal_actions, list):
            return []
        return [
            action
            for action in legal_actions
            if isinstance(action, dict)
            and str(action.get("kind") or "").strip() not in BLOCKED_ACTION_KINDS
        ]

    def _result_is_singleton_end_turn(self, result: dict[str, Any] | None) -> bool:
        actions = self._filtered_result_legal_actions(result)
        return len(actions) == 1 and self._is_end_turn_action(actions[0])

    @staticmethod
    def _obs_energy_hand_summary(obs: dict[str, Any] | None) -> dict[str, Any]:
        obs = obs if isinstance(obs, dict) else {}
        combat = obs.get("combat") if isinstance(obs.get("combat"), dict) else {}
        player = obs.get("player") if isinstance(obs.get("player"), dict) else {}

        combat_active = bool(combat)
        if isinstance(combat, dict) and combat.get("in_progress") is False:
            combat_active = False

        energy = _float(combat.get("energy") if isinstance(combat, dict) else None, -1.0)
        if energy < 0.0:
            energy = _float(player.get("energy") if isinstance(player, dict) else None, 0.0)
        elif energy <= 0.0 and isinstance(player, dict):
            # Some bridge payloads keep the authoritative energy on player.
            energy = max(energy, _float(player.get("energy"), 0.0))

        hand_missing = True
        hand_count = 0
        hand = combat.get("hand") if isinstance(combat, dict) else None
        if not isinstance(hand, list) and isinstance(player, dict):
            hand = player.get("hand")
        if isinstance(hand, list):
            hand_missing = False
            hand_count = len(hand)
        else:
            for container in (combat, player):
                if not isinstance(container, dict):
                    continue
                for key in ("hand_count", "num_cards_in_hand", "cards_in_hand"):
                    if key in container:
                        hand_missing = False
                        hand_count = int(max(_float(container.get(key), 0.0), 0.0))
                        break
                if not hand_missing:
                    break

        return {
            "combat_active": bool(combat_active),
            "energy": float(energy),
            "hand_count": int(hand_count),
            "hand_missing": bool(hand_missing),
            "round": combat.get("round") if isinstance(combat, dict) else None,
            "turn": combat.get("turn") if isinstance(combat, dict) else None,
            "hp": player.get("hp") if isinstance(player, dict) else None,
            "block": player.get("block") if isinstance(player, dict) else None,
        }

    def _current_frontier_needs_short_wait(self) -> bool:
        """Return True for suspicious singleton EndTurn combat frontiers.

        This is the combat-sandbox mirror of EnvV2's pre-dispatch guard.  The
        post-step recovery below can repair a transient singleton EndTurn after
        an action, but it is too late for the selected EndTurn itself: by then
        the bridge has already burned the in-game turn.  This predicate is
        deliberately narrow enough to avoid delaying normal play while catching
        the exact leak reported by the operator: RL-visible actions contain
        only EndTurn, but bridge actionability or raw combat obs says cards may
        still be settling into the playable frontier.
        """

        if len(self._legal_actions) != 1 or not self._is_end_turn_action(self._legal_actions[0]):
            return False
        if getattr(self, "_fast_step_disabled", False):
            return False

        actionability = self._last_actionability if isinstance(self._last_actionability, dict) else {}
        if bool(actionability.get("transient_only_end_turn", False)):
            return True
        try:
            legal_non_end_turn = int(actionability.get("legal_non_end_turn_count", 0) or 0)
        except (TypeError, ValueError):
            legal_non_end_turn = 0
        if legal_non_end_turn > 0:
            return True
        if bool(actionability.get("frontier_stable", True)) is False:
            return True

        return self._frontier_suspicion_has_energy_and_hand()

    def _frontier_suspicion_has_energy_and_hand(self) -> bool:
        obs = self._last_obs_raw if isinstance(self._last_obs_raw, dict) else {}
        combat = obs.get("combat") if isinstance(obs.get("combat"), dict) else {}
        if not isinstance(combat, dict) or not combat:
            return False
        if combat.get("in_progress") is False:
            return False

        player = obs.get("player") if isinstance(obs.get("player"), dict) else {}
        energy = _float(combat.get("energy"), 0.0)
        if energy <= 0.0 and isinstance(player, dict):
            energy = _float(player.get("energy"), 0.0)
        if energy <= 0.0:
            return False

        hand = combat.get("hand")
        if not isinstance(hand, list) and isinstance(player, dict):
            hand = player.get("hand")
        if isinstance(hand, list):
            return len(hand) > 0
        for container in (combat, player):
            if not isinstance(container, dict):
                continue
            for key in ("hand_count", "num_cards_in_hand", "cards_in_hand"):
                if _float(container.get(key), 0.0) > 0.0:
                    return True
        return False

    def _frontier_has_affordable_raw_combat_card(self) -> bool:
        """Return True for high-confidence stale singleton EndTurn windows.

        Broad suspicion (energy + cards in hand) is enough to short-poll.  To
        *block* dispatch after the bounded wait, require a stronger raw-observe
        proof: a non-status/non-curse/non-quest card in hand is currently
        affordable, or bridge explicitly marked it playable.  This prevents
        false positives such as "only Quest cards remain" while catching full
        energy + playable hand leaks.
        """

        obs = self._last_obs_raw if isinstance(self._last_obs_raw, dict) else {}
        combat = obs.get("combat") if isinstance(obs.get("combat"), dict) else {}
        if not isinstance(combat, dict) or not combat:
            return False
        if combat.get("in_progress") is False:
            return False

        player = obs.get("player") if isinstance(obs.get("player"), dict) else {}
        energy = _float(combat.get("energy"), 0.0)
        if energy <= 0.0 and isinstance(player, dict):
            energy = _float(player.get("energy"), 0.0)
        if energy <= 0.0:
            return False

        hand = combat.get("hand")
        if not isinstance(hand, list) and isinstance(player, dict):
            hand = player.get("hand")
        if not isinstance(hand, list):
            return False

        return any(self._raw_card_is_affordable_combat_action(card, energy) for card in hand)


    def _result_needs_post_step_frontier_wait(self, result: dict[str, Any] | None) -> tuple[bool, str, dict[str, Any]]:
        if getattr(self, "_fast_step_disabled", False):
            return False, "disabled", {}
        if not isinstance(result, dict):
            return False, "non_dict_result", {}
        if bool(result.get("done", False)) or bool(result.get("truncated", False)):
            return False, "terminal_result", {}
        if not self._result_is_singleton_end_turn(result):
            return False, "not_singleton_end_turn", {}

        obs_summary = self._obs_energy_hand_summary(self._result_obs(result))
        actionability = self._result_actionability(result)
        if isinstance(actionability, dict):
            if bool(actionability.get("transient_only_end_turn", False)):
                return True, "bridge_transient_only_end_turn", obs_summary
            try:
                non_end_turn = int(actionability.get("legal_non_end_turn_count", 0) or 0)
            except (TypeError, ValueError):
                non_end_turn = 0
            if non_end_turn > 0:
                return True, "bridge_non_end_turn_count_mismatch", obs_summary
            if bool(actionability.get("frontier_stable", True)) is False:
                return True, "bridge_frontier_unstable", obs_summary
            # Bridge explicitly says: only EndTurn and stable.  This covers
            # the user's counterexample: the player may have genuinely played
            # all playable cards while retaining unused energy.
            return False, "bridge_stable_singleton_end_turn", obs_summary

        if (
            bool(obs_summary.get("combat_active"))
            and float(obs_summary.get("energy") or 0.0) > 0.0
            and (
                bool(obs_summary.get("hand_missing"))
                or int(obs_summary.get("hand_count") or 0) == 0
            )
        ):
            return True, "energy_positive_hand_empty_or_missing", obs_summary

        return False, "ambiguous_singleton_end_turn", obs_summary

    @staticmethod
    def _compact_frontier_action(action: dict[str, Any], idx: int) -> dict[str, Any]:
        card = action.get("card") if isinstance(action.get("card"), dict) else {}
        potion = action.get("potion") if isinstance(action.get("potion"), dict) else {}
        title = (
            action.get("title")
            or action.get("name")
            or card.get("title")
            or card.get("name")
            or potion.get("title")
            or potion.get("name")
        )
        return {
            "idx": int(idx),
            "action_id": action.get("action_id"),
            "kind": action.get("kind"),
            "title": title,
        }

    def _frontier_result_summary(self, result: dict[str, Any] | None, *, source: str) -> dict[str, Any]:
        actions = self._filtered_result_legal_actions(result)
        actionability = self._result_actionability(result)
        obs_summary = self._obs_energy_hand_summary(self._result_obs(result))
        return {
            "source": source,
            "legal_action_count": int(len(actions)),
            "non_end_turn_count": int(sum(1 for action in actions if not self._is_end_turn_action(action))),
            "singleton_end_turn": bool(len(actions) == 1 and self._is_end_turn_action(actions[0])),
            "actions": [self._compact_frontier_action(action, idx) for idx, action in enumerate(actions[:6])],
            "obs": obs_summary,
            "actionability": {
                "transient_only_end_turn": bool((actionability or {}).get("transient_only_end_turn", False)),
                "frontier_stable": (actionability or {}).get("frontier_stable"),
                "legal_non_end_turn_count": (actionability or {}).get("legal_non_end_turn_count"),
            } if isinstance(actionability, dict) else None,
        }


    @staticmethod
    def _state_action_is_blocked(action: dict[str, Any]) -> bool:
        return str(action.get("kind") or "").strip() in BLOCKED_ACTION_KINDS

    def _state_non_end_turn_unblocked_action_count(self, state: dict[str, Any] | None) -> int:
        return sum(
            1
            for action in self._state_available_actions(state)
            if not self._state_action_is_blocked(action) and not self._is_end_turn_action(action)
        )

    def _frontier_state_summary(self, state: dict[str, Any] | None, *, source: str) -> dict[str, Any]:
        actions = [
            action
            for action in self._state_available_actions(state)
            if not self._state_action_is_blocked(action)
        ]
        obs_summary = self._obs_energy_hand_summary(state if isinstance(state, dict) else None)
        return {
            "source": source,
            "screen": state.get("screen") if isinstance(state, dict) else None,
            "phase": state.get("phase") if isinstance(state, dict) else None,
            "legal_action_count": int(len(actions)),
            "non_end_turn_count": int(sum(1 for action in actions if not self._is_end_turn_action(action))),
            "singleton_end_turn": bool(len(actions) == 1 and self._is_end_turn_action(actions[0])),
            "actions": [self._compact_frontier_action(action, idx) for idx, action in enumerate(actions[:6])],
            "obs": obs_summary,
        }


    @staticmethod
    def _state_allows_soft_rebind(state: dict[str, Any] | None) -> bool:
        if not isinstance(state, dict):
            return False
        phase = str(state.get("phase") or "").strip().lower()
        if phase.startswith("startup_") or phase == "terminal":
            return False
        screen = str(state.get("screen") or "").strip().upper()
        if screen in {"MAIN_MENU", "TITLE_SCREEN"}:
            return False
        run = state.get("run")
        if isinstance(run, dict):
            if run.get("game_over") is True or run.get("is_game_over") is True:
                return False
            active = run.get("active")
            if active is not None:
                return bool(active)
        return True

    def _safe_reset_into_current_run(self, timeout_ms: int) -> dict[str, Any] | None:
        try:
            result = self._backend_reset(
                rebind_active_run=True,
                timeout_ms=max(1, int(timeout_ms)),
            )
        except Exception:
            return None
        return result if isinstance(result, dict) else None

    def _merge_frontier_rebind_result(
        self,
        base_result: dict[str, Any],
        rebound_result: dict[str, Any],
    ) -> dict[str, Any]:
        """Replace only the post-action frontier, not the action outcome.

        ``bridge.step`` owns reward/done/truncated for the executed action.
        ``bridge.reset(rebind_active_run=True)`` is only used as an observe /
        soft-rebind mechanism, so copying it wholesale would silently zero or
        distort the transition reward.  Keep the step outcome and refresh only
        obs/legal_actions/episode_id/info.
        """

        merged = dict(base_result)
        for key in ("obs", "legal_actions", "episode_id"):
            if key in rebound_result:
                merged[key] = rebound_result[key]

        base_info = base_result.get("info") if isinstance(base_result.get("info"), dict) else {}
        rebound_info = rebound_result.get("info") if isinstance(rebound_result.get("info"), dict) else {}
        info = dict(base_info)
        if rebound_info:
            info["frontier_rebind_info"] = rebound_info
        rebound_actionability = rebound_info.get("actionability") if isinstance(rebound_info.get("actionability"), dict) else None
        if isinstance(rebound_actionability, dict):
            info["actionability"] = rebound_actionability
        else:
            refreshed_actions = self._filtered_result_legal_actions(rebound_result)
            info["actionability"] = {
                "transient_only_end_turn": False,
                "frontier_stable": True,
                "legal_non_end_turn_count": int(
                    sum(1 for action in refreshed_actions if not self._is_end_turn_action(action))
                ),
                "source": "combat_env_post_step_rebind",
            }
        info["post_step_frontier_rebound"] = True
        merged["info"] = info
        return merged

    def _recover_pre_dispatch_end_turn_frontier(
        self,
        *,
        attempted_action_index: Any,
        selected_action: dict[str, Any] | None,
    ):
        """Block stale singleton EndTurn before it reaches ``bridge.step``.

        ``_recover_post_step_frontier`` repairs the *next* frontier after an
        action resolves.  The user-observed failure is different: the policy is
        handed a singleton EndTurn mask while the game still has energy and
        legal cards.  If we let that EndTurn through, the game turn is already
        lost and no post-step recovery can undo the HP/card-tempo damage.

        Return a Gym step tuple when dispatch was blocked/refreshed, otherwise
        return ``None`` to allow the selected EndTurn to proceed.
        """

        if not self._is_end_turn_action(selected_action):
            return None
        if not self._current_frontier_needs_short_wait():
            return None

        self._fast_step_metric_inc("frontier_pre_dispatch_attempt_count")
        metrics: dict[str, Any] = {
            "attempted": True,
            "resolved": False,
            "timeout": False,
            "rebind_attempted": False,
            "rebind_succeeded": False,
            "blocked": False,
            "high_confidence": False,
            "wait_ms": 0.0,
            "poll_count": 0,
            "selected_action_id": selected_action.get("action_id") if isinstance(selected_action, dict) else None,
            "selected_action_kind": selected_action.get("kind") if isinstance(selected_action, dict) else None,
            "obs_summary": self._obs_energy_hand_summary(self._last_obs_raw),
        }
        trace: dict[str, Any] = {
            "event": "pre_dispatch_end_turn_frontier_trace",
            "selected_action": self._compact_frontier_action(selected_action, -1)
            if isinstance(selected_action, dict)
            else None,
            "pre_legal_action_count": len(self._legal_actions),
            "pre_obs": metrics["obs_summary"],
            "pre_actionability": dict(self._last_actionability)
            if isinstance(self._last_actionability, dict)
            else None,
            "polls": [],
        }
        metrics["trace"] = trace

        max_wait_ms = max(0, int(getattr(self, "_fast_step_max_wait_ms", 100) or 0))
        poll_interval_ms = max(1, int(getattr(self, "_fast_step_poll_interval_ms", 10) or 10))
        started = time.perf_counter()
        deadline = started + (max_wait_ms / 1000.0)

        while time.perf_counter() < deadline:
            time.sleep(poll_interval_ms / 1000.0)
            metrics["poll_count"] = int(metrics["poll_count"]) + 1
            state = self._safe_get_state()
            state_summary = self._frontier_state_summary(state, source="pre_dispatch_state_poll")
            if len(trace["polls"]) < 8:
                trace["polls"].append(state_summary)
            non_end_turn_count = int(state_summary.get("non_end_turn_count") or 0)
            if non_end_turn_count <= 0:
                continue
            if not self._state_allows_soft_rebind(state):
                continue

            metrics["rebind_attempted"] = True
            self._fast_step_metric_inc("frontier_pre_dispatch_rebind_attempt_count")
            remaining_ms = max(int((deadline - time.perf_counter()) * 1000.0), 1)
            rebind_timeout_ms = max(
                250,
                min(remaining_ms, int(getattr(self, "reset_timeout_ms", 1000) or 1000)),
            )
            rebound = self._safe_reset_into_current_run(rebind_timeout_ms)
            if not isinstance(rebound, dict):
                continue
            rebound_summary = self._frontier_result_summary(rebound, source="pre_dispatch_soft_rebind_result")
            trace["rebind"] = rebound_summary
            rebound_actions = self._filtered_result_legal_actions(rebound)
            rebound_non_end_turn = sum(1 for action in rebound_actions if not self._is_end_turn_action(action))
            if rebound_actions and rebound_non_end_turn > 0:
                self._update_live_state(rebound)
                if "episode_id" in rebound:
                    self._episode_id = rebound.get("episode_id", getattr(self, "_episode_id", None))
                rebound_info = rebound.get("info") if isinstance(rebound.get("info"), dict) else {}
                rebound_actionability = (
                    rebound_info.get("actionability")
                    if isinstance(rebound_info.get("actionability"), dict)
                    else None
                )
                if isinstance(rebound_actionability, dict):
                    self._last_actionability = rebound_actionability
                metrics["wait_ms"] = max((time.perf_counter() - started) * 1000.0, 0.0)
                metrics["resolved"] = True
                metrics["rebind_succeeded"] = True
                metrics["blocked"] = True
                trace["final_status"] = "blocked_refreshed_non_end_turn_rebound"
                self._fast_step_metric_inc("frontier_pre_dispatch_resolved_count")
                self._fast_step_metric_inc("frontier_pre_dispatch_rebind_success_count")
                self._fast_step_metric_inc("frontier_pre_dispatch_blocked_count")
                self._fast_step_metric_inc("transient_resolved_count")
                return self._make_frontier_refreshed_response(
                    attempted_action_index,
                    metrics=metrics,
                    reason="frontier_refreshed_before_end_turn",
                    refreshed=True,
                    high_confidence=False,
                )

        metrics["wait_ms"] = max((time.perf_counter() - started) * 1000.0, float(max_wait_ms))
        metrics["timeout"] = True
        self._fast_step_metric_inc("frontier_pre_dispatch_timeout_count")

        if self._frontier_has_affordable_raw_combat_card():
            # High-confidence stale frontier: raw obs has an affordable real
            # combat card but the filtered bridge frontier is still singleton
            # EndTurn.  Do not burn the turn; return the same obs/mask so the
            # collector can poll/reselect after the bridge catches up.
            metrics["blocked"] = True
            metrics["high_confidence"] = True
            trace["final_status"] = "blocked_high_confidence_raw_affordable_card"
            self._fast_step_metric_inc("frontier_pre_dispatch_blocked_count")
            self._fast_step_metric_inc("frontier_pre_dispatch_high_confidence_blocked_count")
            return self._make_frontier_refreshed_response(
                attempted_action_index,
                metrics=metrics,
                reason="frontier_stale_singleton_end_turn_blocked",
                refreshed=False,
                high_confidence=True,
            )

        trace["final_status"] = "allowed_no_rebound_no_affordable_raw_card"
        return None

    def _recover_post_step_frontier(
        self,
        result: dict[str, Any],
        *,
        selected_action: dict[str, Any] | None = None,
        legal_actions_before: list[dict[str, Any]] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Short-poll singleton EndTurn frontiers after bridge.step.

        This is the production version of the transient-only-EndTurn check.
        It directly addresses the ambiguous case raised by the user:

        * if short polling rebounds into non-EndTurn actions, the immediate
          only-EndTurn frame was transient and we soft-rebind the frontier;
        * if no rebound appears and bridge did not label the frame transient,
          we classify it as stable/no-actions rather than a bug;
        * if the policy had non-EndTurn actions before selecting End Turn,
          that remains a separate policy-quality issue and is not conflated
          with hand-draw/frontier settling.
        """

        attempted, reason, obs_summary = self._result_needs_post_step_frontier_wait(result)
        metrics: dict[str, Any] = {
            "attempted": bool(attempted),
            "reason": str(reason),
            "wait_ms": 0.0,
            "poll_count": 0,
            "resolved": False,
            "timeout": False,
            "leaked": False,
            "stable_no_actions": False,
            "rebind_attempted": False,
            "rebind_succeeded": False,
            "suspicious_singleton": bool(self._result_is_singleton_end_turn(result)),
            "selected_action_id": selected_action.get("action_id") if isinstance(selected_action, dict) else None,
            "selected_action_kind": selected_action.get("kind") if isinstance(selected_action, dict) else None,
            "pre_legal_action_count": len(legal_actions_before) if isinstance(legal_actions_before, list) else None,
            "obs_summary": obs_summary,
        }
        if not attempted:
            if reason == "bridge_stable_singleton_end_turn":
                metrics["stable_no_actions"] = True
                self._fast_step_metric_inc("post_step_frontier_stable_no_actions_count")
            return result, metrics

        self._fast_step_metric_inc("post_step_frontier_attempt_count")
        self._fast_step_metric_inc("post_step_frontier_suspicious_singleton_count")
        trace: dict[str, Any] = {
            "event": "post_step_frontier_trace",
            "reason": str(reason),
            "selected_action": self._compact_frontier_action(selected_action, -1)
            if isinstance(selected_action, dict)
            else None,
            "pre_legal_action_count": len(legal_actions_before) if isinstance(legal_actions_before, list) else None,
            "immediate": self._frontier_result_summary(result, source="immediate_step_result"),
            "polls": [],
        }
        metrics["trace"] = trace

        max_wait_ms = max(0, int(getattr(self, "_fast_step_max_wait_ms", 100) or 0))
        poll_interval_ms = max(1, int(getattr(self, "_fast_step_poll_interval_ms", 10) or 10))
        started = time.perf_counter()
        deadline = started + (max_wait_ms / 1000.0)
        hard_transient_reason = reason in {
            "bridge_transient_only_end_turn",
            "bridge_frontier_unstable",
            "bridge_non_end_turn_count_mismatch",
        }

        while time.perf_counter() < deadline:
            time.sleep(poll_interval_ms / 1000.0)
            metrics["poll_count"] = int(metrics["poll_count"]) + 1
            state = self._safe_get_state()
            state_summary = self._frontier_state_summary(state, source="state_poll")
            if len(trace["polls"]) < 8:
                trace["polls"].append(state_summary)
            non_end_turn_count = int(state_summary.get("non_end_turn_count") or 0)
            if non_end_turn_count <= 0:
                continue
            if not self._state_allows_soft_rebind(state):
                continue

            metrics["rebind_attempted"] = True
            self._fast_step_metric_inc("post_step_frontier_rebind_attempt_count")
            remaining_ms = max(int((deadline - time.perf_counter()) * 1000.0), 1)
            rebind_timeout_ms = max(
                250,
                min(remaining_ms, int(getattr(self, "reset_timeout_ms", 1000) or 1000)),
            )
            rebound = self._safe_reset_into_current_run(rebind_timeout_ms)
            if not isinstance(rebound, dict):
                continue
            rebound_summary = self._frontier_result_summary(rebound, source="soft_rebind_result")
            trace["rebind"] = rebound_summary
            rebound_actions = self._filtered_result_legal_actions(rebound)
            rebound_non_end_turn = sum(1 for action in rebound_actions if not self._is_end_turn_action(action))
            if rebound_actions and rebound_non_end_turn > 0:
                merged = self._merge_frontier_rebind_result(result, rebound)
                if "episode_id" in rebound:
                    self._episode_id = rebound.get("episode_id", getattr(self, "_episode_id", None))
                metrics["wait_ms"] = max((time.perf_counter() - started) * 1000.0, 0.0)
                metrics["resolved"] = True
                metrics["rebind_succeeded"] = True
                trace["final_status"] = "resolved_non_end_turn_rebound"
                self._fast_step_metric_inc("post_step_frontier_resolved_count")
                self._fast_step_metric_inc("post_step_frontier_rebind_success_count")
                self._fast_step_metric_inc("transient_resolved_count")
                return merged, metrics

        metrics["wait_ms"] = max((time.perf_counter() - started) * 1000.0, float(max_wait_ms))
        metrics["timeout"] = True
        self._fast_step_metric_inc("post_step_frontier_timeout_count")
        self._fast_step_metric_inc("wait_timeout_count")
        if hard_transient_reason:
            metrics["leaked"] = True
            trace["final_status"] = "timeout_after_bridge_transient"
            self._fast_step_metric_inc("post_step_frontier_leaked_count")
        else:
            # Important: leftover energy + empty/missing hand alone is not a
            # proof of bug.  It may simply mean all playable cards were used
            # and the player has unspent energy.  Without a rebound, mark it
            # stable/no-actions so downstream analysis does not overclaim.
            metrics["stable_no_actions"] = True
            trace["final_status"] = "no_rebound_observed_stable_singleton"
            self._fast_step_metric_inc("post_step_frontier_stable_no_actions_count")
        return result, metrics
