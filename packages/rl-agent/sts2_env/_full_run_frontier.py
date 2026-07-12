"""Full-run legal-action filtering, reset readiness, and frontier recovery."""

from __future__ import annotations

import time
from typing import Any

from ._full_run_values import (
    ACTIONABILITY_FAST_POLL_INTERVAL_S,
    ACTIONABILITY_FAST_WAIT_MS,
    ACTIONABILITY_REBIND_TIMEOUT_MS,
    BLOCKED_ACTION_KINDS,
    DISCARD_POTION_ACTION_KIND,
    RECOVERY_POLL_INTERVAL_S,
    RESET_READY_MAX_WAIT_MS,
    RESET_READY_POLL_INTERVAL_S,
    STARTUP_ACTION_PREFIXES,
    STEP_TRANSITION_RECOVERY_MAX_WAIT_MS,
    _float,
)
from .bridge_client import BridgeError


class FullRunFrontierMixin:
    def _filter_legal_actions(
        self,
        legal_actions: list[Any],
        *,
        phase: str,
        obs: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        dict_actions = [action for action in legal_actions if isinstance(action, dict)]
        automation_filtered = [
            action
            for action in dict_actions
            if str(action.get("kind") or "").strip() not in BLOCKED_ACTION_KINDS
        ]
        non_discard_actions = [
            action
            for action in automation_filtered
            if str(action.get("kind") or "").strip() != DISCARD_POTION_ACTION_KIND
        ]
        # Drop optional discard-potion controls when any real game action is
        # present, but keep forced/singleton discard-potion cleanup so full-run
        # episodes do not terminate just because the potion inventory overflow
        # modal is the current bridge frontier.
        filtered = non_discard_actions if non_discard_actions else automation_filtered
        obs_for_filter = obs if isinstance(obs, dict) else self._last_obs_raw
        empty_slots = self._count_empty_potion_slots(obs_for_filter)
        if len(filtered) == 1 and self._is_discard_potion_action(filtered[0]) and empty_slots > 0:
            self._episode_telemetry["frontier_discard_potion_empty_slot_seen"] += 1.0
            self._episode_telemetry["frontier_discard_potion_empty_slot_blocked"] += 1.0
            self._episode_telemetry["frontier_only_discard_potion_empty_slots"] += float(empty_slots)
            return []
        if phase != "actions":
            return filtered
        return self._split_actions_phase_actions(filtered)

    @staticmethod
    def _blocked_action_drop_count(legal_actions: list[Any]) -> int:
        dict_actions = [action for action in legal_actions if isinstance(action, dict)]
        has_real_non_discard = any(
            str(action.get("kind") or "").strip() not in BLOCKED_ACTION_KINDS
            and str(action.get("kind") or "").strip() != DISCARD_POTION_ACTION_KIND
            for action in dict_actions
        )
        dropped = 0
        for action in dict_actions:
            kind = str(action.get("kind") or "").strip()
            if kind in BLOCKED_ACTION_KINDS:
                dropped += 1
            elif kind == DISCARD_POTION_ACTION_KIND and has_real_non_discard:
                dropped += 1
        return dropped

    def _split_actions_phase_actions(self, legal_actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        groups: dict[str, list[dict[str, Any]]] = {
            "map": [],
            "event_option": [],
            "reward": [],
            "card_reward": [],
            "shop": [],
            "rest_site": [],
            "deck_upgrade": [],
            "treasure_relic": [],
            "startup": [],
            "proceed": [],
        }
        fallback: list[dict[str, Any]] = []

        for action in legal_actions:
            action_id = str(action.get("action_id") or "")
            kind = str(action.get("kind") or "").strip()
            if kind == "map":
                groups["map"].append(action)
            elif kind == "event_option":
                groups["event_option"].append(action)
            elif kind == "reward":
                groups["reward"].append(action)
            elif kind == "card_reward":
                groups["card_reward"].append(action)
            elif kind == "shop":
                groups["shop"].append(action)
            elif kind == "rest_site":
                groups["rest_site"].append(action)
            elif kind == "deck_upgrade":
                groups["deck_upgrade"].append(action)
            elif kind == "treasure_relic":
                groups["treasure_relic"].append(action)
            elif kind == "proceed":
                groups["proceed"].append(action)
            elif action_id == "embark" or action_id.startswith(STARTUP_ACTION_PREFIXES):
                groups["startup"].append(action)
            else:
                fallback.append(action)

        for key in ("map", "reward", "card_reward", "event_option", "shop", "rest_site", "deck_upgrade", "treasure_relic"):
            if groups[key]:
                return groups[key]
        if groups["startup"] and not fallback:
            return groups["startup"]
        if groups["proceed"] and not fallback:
            return groups["proceed"]
        return legal_actions

    def _is_episode_lost_error(self, exc: Exception) -> bool:
        if not isinstance(exc, BridgeError):
            return False
        body = exc.response_body
        if isinstance(body, dict) and str(body.get("error") or "").strip() == "unknown_episode_id":
            return True
        return False

    def _is_transient_reset_error(self, exc: Exception) -> bool:
        if not isinstance(exc, BridgeError):
            return False
        body = exc.response_body
        if isinstance(body, dict):
            code = str(body.get("error") or "").strip()
            if code in {"env_reset_no_reset_path", "env_reset_transition_limit", "missing_or_invalid_token"}:
                return True
        return exc.status_code in (401, 409)

    def _next_seed_from_pool(self) -> str | None:
        if not self.seed_pool:
            return None
        if self.seed_strategy == "round_robin":
            seed = self.seed_pool[self._seed_pool_cursor % len(self.seed_pool)]
            self._seed_pool_cursor += 1
            return seed
        # random_per_episode — use gym's np_random for reproducibility
        if self.np_random is None:
            import numpy as np
            return self.seed_pool[int(np.random.randint(len(self.seed_pool)))]
        return self.seed_pool[int(self.np_random.integers(len(self.seed_pool)))]

    def _reset_with_ready_gate(self, *, timeout_ms: int) -> dict[str, Any]:
        deadline = time.monotonic() + (max(timeout_ms, RESET_READY_MAX_WAIT_MS) / 1000.0)
        last_exc: Exception | None = None
        # Full-run training must start every Gym reset from a fresh episode.
        # If the live game is left mid-run (for example a RestSite upgrade
        # overlay with no bridge actions), the bridge may otherwise rebind to
        # that active run and block forever waiting for a usable action.  Start
        # with force_fresh=True and keep retrying fresh after any no-action
        # reset result.
        force_fresh_next = True
        # Pin one seed for this *entire reset call* even across retries —
        # we want the completed episode to match what we promised, not a
        # different seed because a retry happened to land here.
        pinned_seed = self._next_seed_from_pool()

        while time.monotonic() < deadline:
            try:
                result = self._backend_reset(
                    character=self.character,
                    force_fresh=force_fresh_next,
                    defensive_buffs=self.defensive_buffs,
                    seed=pinned_seed,
                    timeout_ms=timeout_ms,
                )
                force_fresh_next = False
            except Exception as exc:
                if not self._is_transient_reset_error(exc):
                    raise
                last_exc = exc
                time.sleep(RESET_READY_POLL_INTERVAL_S)
                continue

            phase = self._extract_phase(result)
            raw_actions = result.get("legal_actions", [])
            filtered_actions = self._filter_legal_actions(
                raw_actions,
                phase=phase,
                obs=result.get("obs") if isinstance(result, dict) else None,
            )
            if filtered_actions:
                return result

            raw_action_count = len(raw_actions) if isinstance(raw_actions, list) else 0
            blocked_only = raw_action_count > 0 and not filtered_actions

            episode_id = result.get("episode_id") if isinstance(result, dict) else None
            if episode_id:
                self._episode_id = str(episode_id)
                self._update_live_state(result)
                recovered = self._recover_filtered_action_window(
                    timeout_ms=self._transition_recovery_timeout_ms(),
                )
                if recovered and self._legal_actions:
                    recovered_result = dict(result)
                    recovered_result["episode_id"] = self._episode_id
                    recovered_result["legal_actions"] = list(self._legal_actions)
                    if isinstance(self._last_obs_raw, dict):
                        recovered_result["obs"] = dict(self._last_obs_raw)
                    return recovered_result

            force_fresh_next = True
            if blocked_only:
                last_exc = RuntimeError(
                    f"reset returned only blocked legal actions at phase={phase}; forcing fresh reset retry"
                )
            else:
                last_exc = RuntimeError(
                    f"reset returned no usable legal actions at phase={phase}; forcing fresh reset retry"
                )
            time.sleep(RESET_READY_POLL_INTERVAL_S)

        if last_exc is not None:
            raise last_exc
        raise RuntimeError("reset ready gate timed out without a usable episode")

    def _recover_filtered_action_window(self, *, timeout_ms: int) -> bool:
        short_wait_for_end_turn = self._current_frontier_needs_short_wait()
        short_wait_for_discard_potion = self._current_frontier_needs_short_wait_for_discard_potion()
        short_wait_singleton_transient = short_wait_for_end_turn or short_wait_for_discard_potion
        if self._legal_actions and not short_wait_singleton_transient:
            self._consecutive_end_turn_leaks = 0
            return True

        if timeout_ms <= 0:
            return bool(self._legal_actions)

        self._episode_telemetry["frontier_recovery_attempts"] += 1.0
        started_with_legal_actions = bool(self._legal_actions)
        if short_wait_singleton_transient:
            timeout_ms = min(int(timeout_ms), ACTIONABILITY_FAST_WAIT_MS)
        if short_wait_for_end_turn:
            self._episode_telemetry["frontier_only_end_turn_short_waits"] += 1.0
            if self._frontier_suspicion_has_energy_and_hand():
                self._episode_telemetry["frontier_only_end_turn_with_energy"] += 1.0
            if self._frontier_has_affordable_raw_combat_card():
                self._episode_telemetry["frontier_only_end_turn_with_affordable_card"] += 1.0
        if short_wait_for_discard_potion:
            self._episode_telemetry["frontier_only_discard_potion_short_waits"] += 1.0
            if self._frontier_suspicion_has_energy_and_hand():
                self._episode_telemetry["frontier_only_discard_potion_with_combat_active"] += 1.0

        deadline = time.monotonic() + (max(int(timeout_ms), 1) / 1000.0)
        blocked_only_seen = False
        while time.monotonic() < deadline:
            state = self._safe_get_state()
            remaining_ms = max(int((deadline - time.monotonic()) * 1000.0), 0)
            unblocked_count = self._state_unblocked_action_count(state)
            non_end_turn_count = self._state_non_end_turn_unblocked_action_count(state)
            non_discard_potion_count = self._state_non_discard_potion_unblocked_action_count(state)
            blocked_only_seen = blocked_only_seen or self._state_has_only_blocked_actions(state)

            if short_wait_for_end_turn:
                settled_replacement_count = non_end_turn_count
            elif short_wait_for_discard_potion:
                settled_replacement_count = non_discard_potion_count
            else:
                settled_replacement_count = 0
            should_rebind = (
                (started_with_legal_actions and settled_replacement_count > 0)
                or (not started_with_legal_actions and unblocked_count > 0)
            )
            if should_rebind and self._state_allows_soft_rebind(state):
                attempt_timeout_ms = max(
                    250,
                    min(remaining_ms, ACTIONABILITY_REBIND_TIMEOUT_MS, self.reset_timeout_ms),
                )
                refreshed = self._safe_reset_into_current_run(attempt_timeout_ms)
                if refreshed is not None:
                    self._episode_id = refreshed.get("episode_id", self._episode_id)
                    self._update_live_state(refreshed)
                    if (
                        self._legal_actions
                        and not self._current_frontier_needs_short_wait()
                        and not self._current_frontier_needs_short_wait_for_discard_potion()
                    ):
                        self._consecutive_end_turn_leaks = 0
                        self._episode_telemetry["frontier_recovery_successes"] += 1.0
                        if short_wait_for_end_turn:
                            self._episode_telemetry["frontier_only_end_turn_resolved"] += 1.0
                        if short_wait_for_discard_potion:
                            self._episode_telemetry["frontier_only_discard_potion_resolved"] += 1.0
                        return True

            time.sleep(ACTIONABILITY_FAST_POLL_INTERVAL_S if started_with_legal_actions else RECOVERY_POLL_INTERVAL_S)

        self._episode_telemetry["frontier_recovery_timeouts"] += 1.0
        if blocked_only_seen:
            self._episode_telemetry["frontier_blocked_only_timeouts"] += 1.0
        if started_with_legal_actions and self._legal_actions:
            if short_wait_for_end_turn:
                self._episode_telemetry["frontier_only_end_turn_leaked"] += 1.0
                self._consecutive_end_turn_leaks += 1
                self._episode_telemetry["frontier_consecutive_end_turn_leaks"] = max(
                    float(self._episode_telemetry.get("frontier_consecutive_end_turn_leaks", 0.0) or 0.0),
                    float(self._consecutive_end_turn_leaks),
                )
                high_confidence_end_turn_leak = bool(
                    short_wait_for_end_turn and self._frontier_has_affordable_raw_combat_card()
                )
                rebind_threshold = 1 if high_confidence_end_turn_leak else 8
                if self._consecutive_end_turn_leaks >= rebind_threshold:
                    self._episode_telemetry["frontier_end_turn_leak_rebinds"] += 1.0
                    refreshed = None
                    state = self._safe_get_state()
                    if self._state_allows_soft_rebind(state):
                        refreshed = self._safe_reset_into_current_run(
                            max(250, min(ACTIONABILITY_REBIND_TIMEOUT_MS, self.reset_timeout_ms))
                        )
                    if refreshed is not None:
                        self._episode_id = refreshed.get("episode_id", self._episode_id)
                        self._update_live_state(refreshed)
                        if self._legal_actions and not self._current_frontier_needs_short_wait():
                            self._episode_telemetry["frontier_recovery_successes"] += 1.0
                            self._episode_telemetry["frontier_only_end_turn_resolved"] += 1.0
                            self._consecutive_end_turn_leaks = 0
                            return True
                stall_threshold = 2 if high_confidence_end_turn_leak else 16
                if self._consecutive_end_turn_leaks >= stall_threshold:
                    self._episode_telemetry["frontier_end_turn_leak_stalls"] += 1.0
                    return False
            elif short_wait_for_discard_potion:
                self._episode_telemetry["frontier_only_discard_potion_leaked"] += 1.0
                self._consecutive_end_turn_leaks = 0
            else:
                self._consecutive_end_turn_leaks = 0
            # The policy already had a real game action (typically end_turn).
            # After the bounded short-poll budget expires, accept it rather
            # than stalling the collector.
            return True

        return False

    def _recover_reset_singleton_end_turn_window(self) -> None:
        """One extra reset-time rebind for high-confidence EndTurn leaks.

        ``reset``/ready-gate can occasionally publish a combat frontier while
        the bridge is still settling the new hand.  If the only RL-visible
        action is EndTurn despite active combat energy + hand, do not let the
        very first policy decision start from that stale singleton.  This is a
        bounded best-effort refresh; it never blocks normal reset if the bridge
        cannot be rebound immediately.
        """

        if not self._current_frontier_needs_short_wait():
            return
        if not self._frontier_suspicion_has_energy_and_hand():
            return

        self._episode_telemetry["frontier_only_end_turn_reset_post_gate_waits"] += 1.0
        refreshed = None
        state = self._safe_get_state()
        if self._state_allows_soft_rebind(state):
            refreshed = self._safe_reset_into_current_run(
                max(250, min(ACTIONABILITY_REBIND_TIMEOUT_MS, self.reset_timeout_ms))
            )
        if refreshed is None:
            return

        self._episode_id = refreshed.get("episode_id", self._episode_id)
        self._update_live_state(refreshed)
        if self._legal_actions and not self._current_frontier_needs_short_wait():
            self._episode_telemetry["frontier_recovery_successes"] += 1.0
            self._episode_telemetry["frontier_only_end_turn_reset_post_gate_resolved"] += 1.0


    @staticmethod
    def _is_discard_potion_action(action: dict[str, Any] | None) -> bool:
        if not isinstance(action, dict):
            return False
        kind = str(action.get("kind") or "").strip().lower()
        action_id = str(action.get("action_id") or "").strip().lower()
        return kind == DISCARD_POTION_ACTION_KIND or action_id.startswith(f"{DISCARD_POTION_ACTION_KIND}:")

    def _current_frontier_needs_short_wait(self) -> bool:
        """Return True for suspicious singleton end_turn combat frontiers.

        Full-run EnvV2 does not have a cheap /env/observe endpoint.  The safe
        compromise is a very small poll only when the current RL-visible
        frontier is exactly end_turn *and* bridge actionability or the raw obs
        suggests the play-phase action list may still be settling.
        """

        if len(self._legal_actions) != 1 or not self._is_end_turn_action(self._legal_actions[0]):
            return False

        actionability = self._last_actionability if isinstance(self._last_actionability, dict) else {}
        if bool(actionability.get("transient_only_end_turn", False)):
            return True
        try:
            legal_non_end_turn = int(actionability.get("legal_non_end_turn_count", 0) or 0)
        except (TypeError, ValueError):
            legal_non_end_turn = 0
        if legal_non_end_turn > 0:
            if self._last_blocked_action_drop_count > 0:
                return True
            return False
        if bool(actionability.get("frontier_stable", True)) is False:
            return True

        return self._frontier_suspicion_has_energy_and_hand()

    def _current_frontier_needs_short_wait_for_discard_potion(self) -> bool:
        """Return True for suspicious singleton discard-potion combat frontiers.

        A real potion-overflow modal may legitimately expose only
        ``discard_potion`` and must remain playable.  The Act1 recovery traces
        show a different case: at combat start / action settling the bridge can
        transiently publish only ``discard_potion`` even though inventory has
        empty slots.  Empty slots are direct proof that this is not a real
        overflow modal, so allow the bounded recovery even outside combat.
        Without observable empty slots, fall back to the combat-active + energy
        + hand suspicion so true reward overflow screens keep working.
        """

        if len(self._legal_actions) != 1 or not self._is_discard_potion_action(self._legal_actions[0]):
            return False
        empty_slots = self._count_empty_potion_slots(self._last_obs_raw)
        if empty_slots > 0:
            self._episode_telemetry["frontier_discard_potion_empty_slot_seen"] += 1.0
            self._episode_telemetry["frontier_only_discard_potion_empty_slots"] += float(empty_slots)
            return True
        return self._frontier_suspicion_has_energy_and_hand()

    def _frontier_suspicion_has_energy_and_hand(self) -> bool:
        obs = self._last_obs_raw if isinstance(self._last_obs_raw, dict) else {}
        combat = obs.get("combat") if isinstance(obs.get("combat"), dict) else {}
        if not isinstance(combat, dict) or not combat:
            return False

        in_progress = combat.get("in_progress")
        if in_progress is False:
            return False

        energy = _float(combat.get("energy"), 0.0)
        player = obs.get("player") if isinstance(obs.get("player"), dict) else {}
        if energy <= 0.0:
            energy = _float(player.get("energy"), 0.0) if isinstance(player, dict) else 0.0
        if energy <= 0.0:
            return False

        hand = combat.get("hand")
        if not isinstance(hand, list) and isinstance(player, dict):
            hand = player.get("hand")
        if isinstance(hand, list):
            return len(hand) > 0
        for key in ("hand_count", "num_cards_in_hand"):
            if _float(combat.get(key), 0.0) > 0.0:
                return True
        return False

    def _frontier_has_affordable_raw_combat_card(self) -> bool:
        """Return True for high-confidence stale singleton EndTurn windows.

        ``_frontier_suspicion_has_energy_and_hand`` intentionally stays broad
        so EnvV2 does a cheap wait whenever combat has energy plus any hand.
        Blocking dispatch needs a narrower predicate: a non-status/non-curse
        card in the raw hand must be affordable by current energy, or the
        bridge must explicitly mark it playable.  This prevents false positives
        such as Quest cards (e.g. 藏宝图) while catching real action-surface
        leaks like energy=1 with four Defends in hand but only EndTurn exposed.
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



    def _transition_recovery_timeout_ms(self) -> int:
        # This is used on the hot step/frontier path, not on the fresh-reset
        # ready gate.  The old max(15s, min(reset, 60s)) made a single transient
        # frontier cost longer than a combat reset; keep step recovery bounded.
        return max(500, min(self.step_timeout_ms, STEP_TRANSITION_RECOVERY_MAX_WAIT_MS))

    def _soft_rebind_into_current_run(self, timeout_ms: int) -> dict[str, Any] | None:
        if timeout_ms <= 0:
            return None

        deadline = time.monotonic() + (timeout_ms / 1000.0)
        rebind_timeout_ms = max(
            250,
            min(timeout_ms, self.reset_timeout_ms, ACTIONABILITY_REBIND_TIMEOUT_MS),
        )

        while time.monotonic() < deadline:
            state = self._safe_get_state()
            if not self._state_allows_soft_rebind(state):
                return None

            try:
                result = self._backend_reset(
                    rebind_active_run=True,
                    defensive_buffs=self.defensive_buffs,
                    timeout_ms=rebind_timeout_ms,
                )
            except Exception as exc:
                if not self._is_transient_reset_error(exc):
                    return None
                time.sleep(RECOVERY_POLL_INTERVAL_S)
                continue

            if not isinstance(result, dict):
                time.sleep(RECOVERY_POLL_INTERVAL_S)
                continue

            phase = self._extract_phase(result)
            filtered_actions = self._filter_legal_actions(
                result.get("legal_actions", []),
                phase=phase,
                obs=result.get("obs") if isinstance(result, dict) else None,
            )
            if filtered_actions:
                return result

            time.sleep(RECOVERY_POLL_INTERVAL_S)

        return None

    def _state_allows_soft_rebind(self, state: dict[str, Any] | None) -> bool:
        if not isinstance(state, dict):
            return False

        phase = str(state.get("phase") or "").strip()
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

    def _decorate_recovery_bridge_info(self, bridge_info: Any, *, recovery_reason: str) -> dict[str, Any]:
        info = dict(bridge_info) if isinstance(bridge_info, dict) else {}
        diagnostics = info.get("action_diagnostics")
        diagnostics = dict(diagnostics) if isinstance(diagnostics, dict) else {}
        diagnostics["soft_rebind_recovery"] = 1.0
        info["action_diagnostics"] = diagnostics
        info["step_recovery"] = recovery_reason
        return info

    def _safe_reset_into_current_run(self, timeout_ms: int) -> dict[str, Any] | None:
        try:
            result = self._soft_rebind_into_current_run(timeout_ms)
        except Exception:
            return None
        return result if isinstance(result, dict) else None

    def _state_has_unblocked_actions(self, state: dict[str, Any] | None) -> bool:
        return self._state_unblocked_action_count(state) > 0

    def _state_unblocked_action_count(self, state: dict[str, Any] | None) -> int:
        actions = self._state_available_actions(state)
        return sum(1 for action in actions if not self._state_action_is_blocked(action, actions))

    def _state_non_end_turn_unblocked_action_count(self, state: dict[str, Any] | None) -> int:
        actions = self._state_available_actions(state)
        return sum(
            1
            for action in actions
            if not self._state_action_is_blocked(action, actions) and not self._is_end_turn_action(action)
        )

    def _state_non_discard_potion_unblocked_action_count(self, state: dict[str, Any] | None) -> int:
        actions = self._state_available_actions(state)
        return sum(
            1
            for action in actions
            if not self._state_action_is_blocked(action, actions)
            and not self._is_discard_potion_action(action)
        )

    def _state_has_only_blocked_actions(self, state: dict[str, Any] | None) -> bool:
        actions = self._state_available_actions(state)
        return bool(actions) and all(self._state_action_is_blocked(action, actions) for action in actions)


    @staticmethod
    def _state_action_is_blocked(
        action: dict[str, Any],
        all_actions: list[dict[str, Any]] | None = None,
    ) -> bool:
        kind = str(action.get("kind") or "").strip()
        if kind in BLOCKED_ACTION_KINDS:
            return True
        if kind != DISCARD_POTION_ACTION_KIND:
            return False
        actions = all_actions if isinstance(all_actions, list) else [action]
        # Optional discard is a UI cleanup control and should not pull recovery
        # away from real card/map/event actions.  Forced singleton discard is a
        # real forward-progress action, so count it as unblocked.
        return any(
            str(other.get("kind") or "").strip() not in BLOCKED_ACTION_KINDS
            and str(other.get("kind") or "").strip() != DISCARD_POTION_ACTION_KIND
            for other in actions
            if isinstance(other, dict)
        )
