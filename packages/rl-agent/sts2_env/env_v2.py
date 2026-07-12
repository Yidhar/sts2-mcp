# ruff: noqa: RUF003
"""Thin Gymnasium wrapper for the STS2 bridge RL env.

All episode lifecycle, state stability, and action validation is handled
by the bridge's env/reset and env/step endpoints. This wrapper only:
  - Calls bridge reset/step
  - Encodes observations into the Dict format
  - Returns action masks for MaskablePPO
  - Provides render() for human inspection

Training defaults to compact ``info`` payloads so the hot path does not keep
re-serializing large raw observation trees. Debug/eval callers can opt back in.
"""

from __future__ import annotations

import time
from typing import Any, ClassVar

import gymnasium as gym
from gymnasium import spaces

from sts2_rl.contracts import EnvironmentBackend

from ._full_run_choice_safety import FullRunChoiceSafetyMixin
from ._full_run_frontier import FullRunFrontierMixin
from ._full_run_legacy_reward import LegacyFullRunRewardMixin
from ._full_run_values import (
    ACTIONABILITY_FAST_POLL_INTERVAL_S,
    ACTIONABILITY_FAST_WAIT_MS,
    ACTIONABILITY_REBIND_TIMEOUT_MS,
    BLOCKED_ACTION_KINDS,
    DISCARD_POTION_ACTION_KIND,
    EMPTY_POTION_NAMES,
    EVENT_COMBAT_LOW_HP_THRESHOLD,
    EVENT_HP_LOSS_LOW_HP_THRESHOLD,
    INVALID_ACTION_REASON,
    RECOVERY_MAX_WAIT_MS,
    RECOVERY_POLL_INTERVAL_S,
    RESET_READY_MAX_WAIT_MS,
    RESET_READY_POLL_INTERVAL_S,
    STARTUP_ACTION_PREFIXES,
    STEP_RECOVERY_TRUNCATION_REASON,
    STEP_TRANSITION_RECOVERY_MAX_WAIT_MS,
    TRANSITION_RECOVERY_MAX_WAIT_MS,
    _float,
    _parse_act_id,
)
from .action_compact import compact_legal_actions
from .action_history import ActionHistoryTracker
from .aux_targets import build_aux_targets
from .bridge_client import BridgeClient, BridgeError
from .combat_memory import CombatMemoryTracker
from .end_turn_quality import strict_end_turn_waste_context
from .environment_runtime import EnvironmentRuntimeMixin
from .observation_common import MAX_ACTIONS, DenseObservationEncoder
from .observation_v3 import WorldTokenObservationEncoder
from .potion_timing import compute_potion_timing
from .reward_constants import (
    BOSS_ACT_FLOORS,
    BOSS_COMBAT_LOSS_DAMAGE_UNDO_PERCENT_SCALE,
    BOSS_COMBAT_LOSS_PENALTY_BASE,
    BOSS_COMBAT_LOSS_PENALTY_MISSING_HP_SCALE,
    BOSS_ENEMY_HP_DELTA_PERCENT_SCALE,
    BOSS_FLOOR_ENTRY_BONUS,
    ENEMY_HP_DELTA_REWARD_MAX_ABS,
    ENEMY_HP_DELTA_REWARD_SCALE,
    ENEMY_HP_SENTINEL_THRESHOLD,
    FLOOR_CLEAR_BONUS_PER_FLOOR,
    FLOOR_CLEAR_MIN_FLOOR,
    FULL_RUN_DEATH_PENALTY_BASE,
    FULL_RUN_DEATH_PENALTY_LATE_ACT_SCALE,
    FULL_RUN_DEATH_PENALTY_MISSING_HP_SCALE,
    INVALID_ACTION_REWARD,
    PLAYER_HP_LOSS_REWARD_SCALE,
    POTION_HOARDING_MAX_PENALTY_ABS,
    POTION_HOARDING_PENALTY_PER_POTION,
    POTION_TIMING_QUALITY_SCALE,
    POTION_TIMING_WASTE_SCALE,
    POTION_USE_BOSS_BONUS,
    POTION_USE_ELITE_BONUS,
    POTION_USE_MONSTER_BONUS,
    POTION_USE_MONSTER_PENALTY,
    REST_SITE_SKIP_HEAL_HP_THRESHOLD,
    REST_SITE_SKIP_HEAL_PENALTY,
)
from .reward_constants import (
    FULL_RUN_WASTE_BASE as END_TURN_WASTE_BASE_PENALTY,
)
from .reward_constants import (
    FULL_RUN_WASTE_ENERGY as END_TURN_WASTE_ENERGY_PENALTY,
)
from .reward_constants import (
    FULL_RUN_WASTE_EXTRA_ACTION as END_TURN_WASTE_EXTRA_ACTION_PENALTY,
)
from .reward_constants import (
    FULL_RUN_WASTE_ZERO_COST as END_TURN_WASTE_ZERO_COST_BONUS_PENALTY,
)
from .run_memory import RunMemoryTracker


class SlayTheSpire2EnvV2(
    FullRunChoiceSafetyMixin,
    LegacyFullRunRewardMixin,
    FullRunFrontierMixin,
    EnvironmentRuntimeMixin,
    gym.Env,
):
    """Gymnasium Env backed by the STS2 bridge env/reset and env/step endpoints."""

    metadata: ClassVar[dict[str, list[str]]] = {"render_modes": ["human"]}

    def __init__(
        self,
        session_file: str | None = None,
        character: str | None = None,
        defensive_buffs: bool = False,
        reset_timeout_ms: int = 60000,
        step_timeout_ms: int = 20000,
        render_mode: str | None = None,
        obs_encoder: DenseObservationEncoder | None = None,
        include_debug_info: bool = False,
        bridge: BridgeClient | None = None,
        backend: EnvironmentBackend | None = None,
        stuck_watchdog_steps: int = 400,
        seed_pool: list[str] | None = None,
        seed_strategy: str = "round_robin",
    ) -> None:
        super().__init__()

        # All mutations flow through the typed backend. ``bridge`` remains a
        # compatibility injection seam and is immediately wrapped when used.
        if backend is None:
            raw_bridge = bridge if bridge is not None else BridgeClient(session_path=session_file)
            self._initialize_environment_runtime(bridge=raw_bridge, backend_name="legacy_bridge")
        else:
            if bridge is not None:
                raise ValueError("provide backend or bridge, not both")
            self._initialize_environment_runtime(backend=backend)
        self.obs_encoder = obs_encoder or WorldTokenObservationEncoder(use_text=False)
        self.character = character
        self.defensive_buffs = defensive_buffs
        self.reset_timeout_ms = reset_timeout_ms
        self.step_timeout_ms = step_timeout_ms
        self.render_mode = render_mode
        self.include_debug_info = bool(include_debug_info)
        # Seed pool: list of 10-char STS2 seed strings. When non-empty, each
        # env.reset() picks one per seed_strategy and pins the run's RNG via
        # bridge /env/reset seed parameter. Gives deterministic map /
        # encounters / rewards / monster AI for that run. Empty/None ==
        # game's built-in random seed each run (original behavior).
        self.seed_pool: list[str] = list(seed_pool or [])
        if seed_strategy not in ("round_robin", "random_per_episode"):
            raise ValueError(
                f"seed_strategy must be round_robin|random_per_episode, got {seed_strategy}"
            )
        self.seed_strategy = seed_strategy
        self._seed_pool_cursor: int = 0
        # Phase-stuck watchdog: truncate the episode when the fingerprint
        # (phase, floor, combat_round, enemy_hp_total, player_hp) stays the
        # same for >= stuck_watchdog_steps. Set to 0 to disable. Motivated
        # by sim long-train where 23% of episodes ran 1000–6000 steps on the
        # same floor without progressing — dominating rollouts with
        # non-combat noise and zeroing aux_enemy_state / value losses.
        self.stuck_watchdog_steps = int(stuck_watchdog_steps)

        self.observation_space = self.obs_encoder.obs_space
        self.action_space = spaces.Discrete(MAX_ACTIONS)

        self._episode_id: str | None = None
        self._legal_actions: list[dict[str, Any]] = []
        self._last_obs_raw: dict[str, Any] | None = None
        self._last_action_overflow: int = 0
        self._last_bridge_info: dict[str, Any] | None = None
        self._last_actionability: dict[str, Any] | None = None
        self._last_raw_legal_action_count: int = 0
        self._last_raw_legal_actions_compact: list[dict[str, Any]] = []
        self._last_blocked_action_drop_count: int = 0
        self._max_floor_reached: int = 0
        self._stuck_fingerprint: tuple | None = None
        self._stuck_steps: int = 0
        # Bounded watchdog for repeated suspicious singleton end_turn leaks.
        # A single short leak is acceptable when bridge actionability is merely
        # late; many consecutive leaks means the collector is burning turns
        # during an unresolved action-frontier transition.
        self._consecutive_end_turn_leaks: int = 0
        # Phase 8 Tier 1: per-step action + per-turn summary history.
        # Fed as ``raw_obs["_action_history"]`` for observation_v3 to emit
        # HISTORY tokens into the world-token budget. See
        # sts2_env/action_history.py for the data-flow design.
        self._action_history = ActionHistoryTracker()
        # Phase 8.2 telemetry: per-episode counters tracking whether the
        # reward-shape patches (6643917 floor-clear/boss-damage + 65b5196
        # rest-HP/potion) are actually shifting observed behavior. Reset
        # at episode start, surfaced in info["episode_telemetry"] at
        # terminal step so async_ready_collector flattens them into
        # reset_events.jsonl.
        self._episode_telemetry: dict[str, float] = self._blank_telemetry()
        self._run_memory = RunMemoryTracker(
            episode_mode="full_run",
            potion_mechanics_available=True,
        )
        self._combat_memory = CombatMemoryTracker()

    @staticmethod
    def _blank_telemetry() -> dict[str, float]:
        """Zeroed per-episode behavior-counter dict.

        Keys chosen so a flat JSON write into reset_events is easy
        to grep/aggregate. Floats + ints both stored as floats (json
        serialization doesn't care and downstream aggregators cast).
        """
        return {
            # Potion usage ----------------------------------------------
            "potion_use_count": 0.0,
            "potion_use_boss_count": 0.0,
            "potion_use_elite_count": 0.0,
            "potion_use_bonus_total": 0.0,
            "potion_discard_count": 0.0,
            "potion_timing_quality_events": 0.0,
            "potion_timing_waste_events": 0.0,
            "potion_timing_reward_total": 0.0,
            # Rest site -------------------------------------------------
            "rest_site_encounters": 0.0,
            "rest_heal_chosen": 0.0,
            "rest_smith_chosen": 0.0,
            "rest_skip_heal_chosen": 0.0,
            "rest_skip_heal_at_low_hp": 0.0,
            "rest_penalty_total": 0.0,
            "rest_heal_exposure_low_hp": 0.0,
            "rest_heal_exposure_heal_available": 0.0,
            "rest_heal_exposure_forced": 0.0,
            "rest_heal_exposure_miss": 0.0,
            # Event combat exposure ------------------------------------
            "event_combat_option_available": 0.0,
            "event_combat_option_low_hp_available": 0.0,
            "event_combat_option_safe_alternative": 0.0,
            "event_combat_option_masked_low_hp": 0.0,
            "event_combat_option_selected": 0.0,
            "event_combat_option_selected_low_hp": 0.0,
            # Event HP-loss exposure --------------------------------------
            "event_hp_loss_option_available": 0.0,
            "event_hp_loss_option_low_hp_available": 0.0,
            "event_hp_loss_option_safe_alternative": 0.0,
            "event_hp_loss_option_masked_low_hp": 0.0,
            "event_hp_loss_option_selected": 0.0,
            "event_hp_loss_option_selected_low_hp": 0.0,
            # Boss combat ------------------------------------------------
            "boss_damage_dealt_raw": 0.0,
            "boss_damage_bonus_total": 0.0,
            "boss_damage_death_clear_guarded": 0.0,
            "boss_damage_death_clear_guarded_remaining_hp_raw": 0.0,
            "boss_damage_death_clear_guarded_raw": 0.0,
            "boss_encounter_steps": 0.0,
            "boss_death_terminal_penalty_events": 0.0,
            "boss_death_terminal_penalty_total": 0.0,
            "boss_death_terminal_damage_ratio": 0.0,
            "boss_death_terminal_missing_hp_ratio": 0.0,
            "full_run_death_terminal_penalty_events": 0.0,
            "full_run_death_terminal_penalty_total": 0.0,
            "full_run_death_terminal_floor_norm": 0.0,
            "full_run_death_terminal_missing_hp_ratio": 0.0,
            # Floor clear ladder ----------------------------------------
            "floor_clear_reward_total": 0.0,
            "floor_clear_events": 0.0,
            "boss_floor_entry_events": 0.0,
            # Hoarding penalty (fires at episode end) --------------------
            "potion_hoarding_unused_at_end": 0.0,
            "potion_hoarding_penalty_total": 0.0,
            # Full-run action frontier recovery -------------------------
            "frontier_recovery_attempts": 0.0,
            "frontier_recovery_successes": 0.0,
            "frontier_recovery_timeouts": 0.0,
            "frontier_blocked_only_timeouts": 0.0,
            "frontier_only_end_turn_short_waits": 0.0,
            "frontier_only_end_turn_resolved": 0.0,
            "frontier_only_end_turn_leaked": 0.0,
            "frontier_only_end_turn_with_energy": 0.0,
            "frontier_only_end_turn_with_affordable_card": 0.0,
            "frontier_only_end_turn_pre_dispatch_waits": 0.0,
            "frontier_only_end_turn_pre_dispatch_blocked": 0.0,
            "frontier_only_end_turn_pre_dispatch_high_confidence_blocked": 0.0,
            "frontier_only_end_turn_pre_dispatch_stalls": 0.0,
            "frontier_only_end_turn_reset_post_gate_waits": 0.0,
            "frontier_only_end_turn_reset_post_gate_resolved": 0.0,
            "frontier_only_discard_potion_short_waits": 0.0,
            "frontier_only_discard_potion_resolved": 0.0,
            "frontier_only_discard_potion_leaked": 0.0,
            "frontier_only_discard_potion_with_combat_active": 0.0,
            "frontier_discard_potion_empty_slot_seen": 0.0,
            "frontier_discard_potion_empty_slot_blocked": 0.0,
            "frontier_only_discard_potion_empty_slots": 0.0,
            "frontier_consecutive_end_turn_leaks": 0.0,
            "frontier_end_turn_leak_rebinds": 0.0,
            "frontier_end_turn_leak_stalls": 0.0,
            "frontier_actions_dropped_blocked": 0.0,
        }

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        started = time.perf_counter()
        # Reset before _update_live_state so the first floor observed gets
        # recorded into _max_floor_reached rather than left over from the
        # previous episode.
        self._max_floor_reached = 0
        self._stuck_fingerprint = None
        self._stuck_steps = 0
        self._consecutive_end_turn_leaks = 0
        self._last_bridge_info = None
        self._last_actionability = None
        self._last_raw_legal_action_count = 0
        self._last_raw_legal_actions_compact = []
        self._last_blocked_action_drop_count = 0
        self._action_history.reset()
        self._episode_telemetry = self._blank_telemetry()

        bridge_started = time.perf_counter()
        result = self._reset_with_ready_gate(timeout_ms=self.reset_timeout_ms)
        bridge_elapsed_ms = (time.perf_counter() - bridge_started) * 1000.0

        self._episode_id = result["episode_id"]
        self._update_live_state(result)
        self._inject_action_history_into_obs()
        self._recover_filtered_action_window(timeout_ms=min(self.reset_timeout_ms, RECOVERY_MAX_WAIT_MS))
        self._inject_action_history_into_obs()
        self._recover_reset_singleton_end_turn_window()
        self._inject_action_history_into_obs()
        run_memory_started = time.perf_counter()
        self._run_memory.reset(
            self._last_obs_raw,
            self._legal_actions,
            episode_mode="full_run",
            potion_mechanics_available=True,
        )
        self._combat_memory.reset(self._last_obs_raw)
        run_memory_elapsed_ms = (time.perf_counter() - run_memory_started) * 1000.0
        self._inject_run_route_snapshot_into_obs()

        planner_context = self._planner_context()
        obs_encode_started = time.perf_counter()
        obs = self.obs_encoder.encode(self._last_obs_raw, self._legal_actions, planner_context)
        obs_encode_elapsed_ms = (time.perf_counter() - obs_encode_started) * 1000.0
        info_started = time.perf_counter()
        info = self._build_info(
            result.get("info", {}),
            extra={
                "python_timing_ms": self._python_timing(
                    bridge_roundtrip=bridge_elapsed_ms,
                    run_memory_update=run_memory_elapsed_ms,
                    obs_encode=obs_encode_elapsed_ms,
                    aux_targets=0.0,
                    info_build=0.0,
                    total=(time.perf_counter() - started) * 1000.0,
                )
            },
        )
        info["python_timing_ms"]["info_build"] = (time.perf_counter() - info_started) * 1000.0
        info["python_timing_ms"]["total"] = (time.perf_counter() - started) * 1000.0
        return obs, info

    def step(self, action: int):
        if not self._legal_actions:
            recovered = self._recover_filtered_action_window(
                timeout_ms=self._transition_recovery_timeout_ms()
            )
            if not recovered:
                return self._make_terminal()
        started = time.perf_counter()

        normalized_action = self._normalize_action(action)
        if normalized_action is None or normalized_action >= len(self._legal_actions):
            return self._make_invalid_action_response(action)

        legal_action = self._legal_actions[normalized_action]
        if self._is_end_turn_action(legal_action) and self._current_frontier_needs_short_wait():
            self._episode_telemetry["frontier_only_end_turn_pre_dispatch_waits"] += 1.0
            recovered = self._recover_filtered_action_window(
                timeout_ms=self._transition_recovery_timeout_ms()
            )
            if not recovered:
                self._episode_telemetry["frontier_only_end_turn_pre_dispatch_stalls"] += 1.0
                return self._make_terminal()
            refreshed_has_non_end_turn = any(
                not self._is_end_turn_action(action_item)
                for action_item in self._legal_actions[:MAX_ACTIONS]
            )
            if refreshed_has_non_end_turn:
                self._episode_telemetry["frontier_only_end_turn_pre_dispatch_blocked"] += 1.0
                return self._make_frontier_refreshed_response(action)
            if self._frontier_has_affordable_raw_combat_card():
                # High-confidence stale frontier: the filtered bridge mask is
                # still singleton EndTurn, but raw combat obs says there is an
                # affordable real card in hand.  Do not burn the in-game turn;
                # return a no-op response so the collector can poll/reselect.
                # If the bridge never recovers, _recover_filtered_action_window
                # will terminal-stall after a small number of repeated
                # high-confidence leaks instead of dispatching EndTurn.
                self._episode_telemetry[
                    "frontier_only_end_turn_pre_dispatch_high_confidence_blocked"
                ] += 1.0
                return self._make_frontier_refreshed_response(
                    action,
                    reason="frontier_stale_singleton_end_turn_blocked",
                    refreshed=False,
                    high_confidence=True,
                )
            # The bounded wait/rebind still exposes only end_turn.  Rebind can
            # reindex singleton actions, so refresh the dispatch target before
            # calling the bridge rather than using a stale pre-wait dict.
            normalized_action = 0 if len(self._legal_actions) == 1 else min(
                int(normalized_action),
                max(len(self._legal_actions) - 1, 0),
            )
            legal_action = self._legal_actions[normalized_action]
        legal_actions_before = list(self._legal_actions)
        prev_obs = self._last_obs_raw or {}
        prev_planner_context = self._planner_context()
        end_turn_penalty = self._end_turn_waste_penalty(prev_obs, self._legal_actions, legal_action)
        if self._is_event_combat_option(legal_action):
            self._episode_telemetry["event_combat_option_selected"] += 1.0
            hp_ratio, hp_valid = self._hp_ratio_from_obs(prev_obs)
            if hp_valid and hp_ratio < EVENT_COMBAT_LOW_HP_THRESHOLD:
                self._episode_telemetry["event_combat_option_selected_low_hp"] += 1.0
        if self._is_event_hp_loss_option(legal_action, prev_obs):
            self._episode_telemetry["event_hp_loss_option_selected"] += 1.0
            hp_ratio, hp_valid = self._hp_ratio_from_obs(prev_obs)
            if hp_valid and hp_ratio < EVENT_HP_LOSS_LOW_HP_THRESHOLD:
                self._episode_telemetry["event_hp_loss_option_selected_low_hp"] += 1.0

        try:
            bridge_started = time.perf_counter()
            result = self._backend_step(
                episode_id=self._episode_id,
                action_id=legal_action.get("action_id"),
                timeout_ms=self.step_timeout_ms,
            )
            bridge_elapsed_ms = (time.perf_counter() - bridge_started) * 1000.0
        except Exception as exc:
            if self._is_episode_lost_error(exc):
                recovered = self._soft_rebind_into_current_run(
                    timeout_ms=self._transition_recovery_timeout_ms()
                )
                if recovered is not None:
                    self._episode_id = recovered.get("episode_id", self._episode_id)
                    self._update_live_state(recovered)
                    run_memory_started = time.perf_counter()
                    self._run_memory.update_transition(prev_obs, legal_action, self._last_obs_raw, legal_actions=self._legal_actions)
                    self._combat_memory.update(prev_obs, legal_action, self._last_obs_raw)
                    run_memory_elapsed_ms = (time.perf_counter() - run_memory_started) * 1000.0
                    self._inject_run_route_snapshot_into_obs()
                    planner_context = self._planner_context()
                    self._inject_action_history_into_obs()
                    obs_encode_started = time.perf_counter()
                    obs = self.obs_encoder.encode(self._last_obs_raw, self._legal_actions, planner_context)
                    obs_encode_elapsed_ms = (time.perf_counter() - obs_encode_started) * 1000.0
                    info_started = time.perf_counter()
                    info = self._build_info(
                        {},
                        extra={
                            "bridge_episode_lost": True,
                            "bridge_episode_rebound": True,
                            "bridge_exception": str(exc),
                            "step_recovery": "soft_rebind_current_run",
                            "python_timing_ms": self._python_timing(
                                bridge_roundtrip=0.0,
                                run_memory_update=run_memory_elapsed_ms,
                                obs_encode=obs_encode_elapsed_ms,
                                aux_targets=0.0,
                                info_build=0.0,
                                total=(time.perf_counter() - started) * 1000.0,
                            ),
                        },
                    )
                    info["python_timing_ms"]["info_build"] = (time.perf_counter() - info_started) * 1000.0
                    info["python_timing_ms"]["total"] = (time.perf_counter() - started) * 1000.0
                    return obs, 0.0, False, False, info
                return self._make_step_recovery_response(exc)
            raise

        canonical_reward = self._record_backend_transition(
            result,
            action_handle=str(legal_action.get("action_id") or ""),
        )
        self._update_live_state(result)
        # Keep the bridge's immediate post-action observation for diagnostics
        # that must describe the actual effect of this action.  Later soft
        # rebind/recovery can advance to a different actionable snapshot.
        direct_after_obs = self._last_obs_raw
        reward = float(canonical_reward.total)
        terminated = bool(result.get("done", False))
        truncated = bool(result.get("truncated", False))
        if not self._uses_external_v2_reward(result):
            # Deprecated v1 compatibility only. Active live/headless paths use
            # the versioned fact calculator above as the sole reward owner.
            reward += self._enemy_hp_delta_reward(prev_obs, self._last_obs_raw)
            reward += self._player_hp_delta_reward(prev_obs, self._last_obs_raw)
            reward += end_turn_penalty
            reward += self._floor_clear_reward(prev_obs, self._last_obs_raw)
            reward += self._boss_damage_bonus_reward(prev_obs, self._last_obs_raw)
            reward += self._rest_site_skip_heal_penalty(prev_obs, legal_action)
            reward += self._potion_use_bonus(prev_obs, legal_action)
            reward += self._potion_timing_step_reward(legal_action, prev_obs, legal_actions_before)
            reward += self._potion_hoarding_penalty(
                self._last_obs_raw,
                terminated=terminated,
                truncated=truncated,
            )
            reward += self._full_run_death_terminal_penalty(
                prev_obs,
                self._last_obs_raw,
                terminated=terminated,
                truncated=truncated,
            )
            reward += self._boss_death_terminal_penalty(
                prev_obs,
                self._last_obs_raw,
                terminated=terminated,
                truncated=truncated,
            )
        bridge_info = result.get("info", {})
        potion_transition_record = self._build_potion_transition_record(
            action=legal_action,
            prev_obs=prev_obs,
            after_obs=direct_after_obs,
            legal_actions_before=legal_actions_before,
            bridge_info=bridge_info if isinstance(bridge_info, dict) else {},
            reward=reward,
            terminated=terminated,
            truncated=truncated,
        )
        # Phase 8 Tier 1: record the transition into the action-history
        # tracker. Done here (before soft-rebind / recovery) so the "next"
        # state matches what the bridge returned — recovery may advance
        # the sim further, which would dilute the "this step's direct
        # consequence" signal.
        self._action_history.record(
            action=legal_action,
            prev_obs=prev_obs,
            next_obs=self._last_obs_raw,
            reward=reward,
            rejected=bool(bridge_info.get("action_error")) if isinstance(bridge_info, dict) else False,
        )

        if truncated and not terminated:
            recovered = self._soft_rebind_into_current_run(
                timeout_ms=self._transition_recovery_timeout_ms()
            )
            if recovered is not None:
                self._episode_id = recovered.get("episode_id", self._episode_id)
                self._update_live_state(recovered)
                terminated = False
                truncated = False
                bridge_info = self._decorate_recovery_bridge_info(
                    bridge_info,
                    recovery_reason="soft_rebind_after_truncated_step",
                )

        if not terminated and not truncated:
            self._recover_filtered_action_window(timeout_ms=self._transition_recovery_timeout_ms())

        if not terminated and not truncated and self.stuck_watchdog_steps > 0:
            stuck_truncated, stuck_bridge_info = self._check_stuck_watchdog(bridge_info)
            if stuck_truncated:
                truncated = True
                bridge_info = stuck_bridge_info

        run_memory_started = time.perf_counter()
        self._run_memory.update_transition(prev_obs, legal_action, self._last_obs_raw, legal_actions=self._legal_actions)
        self._combat_memory.update(prev_obs, legal_action, self._last_obs_raw)
        run_memory_elapsed_ms = (time.perf_counter() - run_memory_started) * 1000.0
        self._inject_run_route_snapshot_into_obs()
        planner_context = self._planner_context()
        self._inject_action_history_into_obs()
        obs_encode_started = time.perf_counter()
        obs = self.obs_encoder.encode(self._last_obs_raw, self._legal_actions, planner_context)
        obs_encode_elapsed_ms = (time.perf_counter() - obs_encode_started) * 1000.0
        aux_started = time.perf_counter()
        aux_targets = build_aux_targets(
            prev_obs,
            legal_action,
            self._last_obs_raw,
            prev_planner_context=prev_planner_context,
            next_planner_context=planner_context,
            terminated=terminated,
            truncated=truncated,
            legal_actions_before=legal_actions_before,
        )
        aux_elapsed_ms = (time.perf_counter() - aux_started) * 1000.0
        info_started = time.perf_counter()
        # Phase 8.2 telemetry: surface the per-episode counters into
        # info at EVERY step (cheap, tiny dict). async_ready_collector
        # only actually reads them on terminal events, but making them
        # always-available keeps the info schema uniform across steps.
        episode_telemetry_snapshot = dict(self._episode_telemetry)
        info = self._build_info(
            bridge_info,
            extra={
                "aux_targets": aux_targets,
                "episode_telemetry": episode_telemetry_snapshot,
                **({"potion_transition": potion_transition_record} if potion_transition_record is not None else {}),
                "python_timing_ms": self._python_timing(
                    bridge_roundtrip=bridge_elapsed_ms,
                    run_memory_update=run_memory_elapsed_ms,
                    obs_encode=obs_encode_elapsed_ms,
                    aux_targets=aux_elapsed_ms,
                    info_build=0.0,
                    total=(time.perf_counter() - started) * 1000.0,
                ),
            },
        )
        info["python_timing_ms"]["info_build"] = (time.perf_counter() - info_started) * 1000.0
        info["python_timing_ms"]["total"] = (time.perf_counter() - started) * 1000.0

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info


    def recover_actionable_state(self, timeout_ms: int | None = None):
        recovered = self._recover_filtered_action_window(
            timeout_ms=min(timeout_ms or self.step_timeout_ms, RECOVERY_MAX_WAIT_MS)
        )
        self._inject_action_history_into_obs()
        obs = self.obs_encoder.encode(self._last_obs_raw or {}, self._legal_actions, self._planner_context())
        info = self._build_info({})
        return recovered, obs, info

    def _make_terminal(self):
        self._inject_action_history_into_obs()
        obs = self.obs_encoder.encode(self._last_obs_raw or {}, [], self._planner_context())
        info = self._build_info(
            {},
            extra={
                "episode_telemetry": dict(self._episode_telemetry),
                "python_timing_ms": self._python_timing(
                    bridge_roundtrip=0.0,
                    run_memory_update=0.0,
                    obs_encode=0.0,
                    aux_targets=0.0,
                    info_build=0.0,
                    total=0.0,
                )
            },
        )
        return obs, 0.0, True, False, info

    def render(self) -> None:
        if self._last_obs_raw is None:
            return
        phase = self._last_obs_raw.get("phase", "?")
        player = self._last_obs_raw.get("player", {})
        hp = player.get("hp", "?")
        max_hp = player.get("max_hp", "?")
        run = self._last_obs_raw.get("run", {})
        floor_num = run.get("floor", "?")
        print(
            f"[STS2] Phase: {phase} | HP: {hp}/{max_hp} "
            f"| Floor: {floor_num} | Actions: {len(self._legal_actions)}"
        )

    def close(self) -> None:
        self._close_environment_runtime()


    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------



    def _update_live_state(self, result: dict[str, Any]) -> None:
        phase = self._extract_phase(result)
        legal_actions = result.get("legal_actions", [])
        self._last_raw_legal_action_count = len(legal_actions) if isinstance(legal_actions, list) else 0
        self._last_raw_legal_actions_compact = (
            compact_legal_actions(legal_actions) if isinstance(legal_actions, list) else []
        )
        obs = result.get("obs", {})
        self._last_obs_raw = obs if isinstance(obs, dict) else {}
        if isinstance(legal_actions, list):
            filtered_actions = self._filter_legal_actions(
                legal_actions,
                phase=phase,
                obs=self._last_obs_raw,
            )
            filtered_actions = self._apply_low_hp_rest_heal_exposure_filter(
                filtered_actions,
                self._last_obs_raw,
            )
            filtered_actions = self._apply_low_hp_event_combat_exposure_filter(
                filtered_actions,
                self._last_obs_raw,
            )
            self._legal_actions = self._apply_low_hp_event_hp_loss_exposure_filter(
                filtered_actions,
                self._last_obs_raw,
            )
        else:
            self._legal_actions = []
        self._last_blocked_action_drop_count = (
            self._blocked_action_drop_count(legal_actions) if isinstance(legal_actions, list) else 0
        )
        if self._last_blocked_action_drop_count > 0:
            self._episode_telemetry["frontier_actions_dropped_blocked"] += float(
                self._last_blocked_action_drop_count
            )
        info = result.get("info") if isinstance(result.get("info"), dict) else {}
        self._last_bridge_info = dict(info) if isinstance(info, dict) else None
        actionability = info.get("actionability") if isinstance(info.get("actionability"), dict) else None
        self._last_actionability = dict(actionability) if isinstance(actionability, dict) else None
        self._last_action_overflow = max(len(self._legal_actions) - MAX_ACTIONS, 0)
        if not (len(self._legal_actions) == 1 and self._is_end_turn_action(self._legal_actions[0])):
            self._consecutive_end_turn_leaks = 0
        # Track deepest floor this episode has reached (for episode-terminal
        # logging + Monitor CSV aggregation). obs["run"]["floor"] is now
        # populated for sim after the April 2026 translator fix.
        run = self._last_obs_raw.get("run") if isinstance(self._last_obs_raw, dict) else None
        if isinstance(run, dict):
            floor_val = run.get("floor")
            if floor_val is None:
                floor_val = run.get("total_floor")
            if floor_val is None:
                floor_val = run.get("act_floor")
            try:
                current_floor = int(floor_val) if floor_val is not None else 0
            except (TypeError, ValueError):
                current_floor = 0
            if current_floor > self._max_floor_reached:
                self._max_floor_reached = current_floor

    def _extract_phase(self, result: dict[str, Any]) -> str:
        obs = result.get("obs")
        if isinstance(obs, dict):
            phase = str(obs.get("phase") or "").strip()
            if phase:
                return phase
        info = result.get("info")
        if isinstance(info, dict):
            phase = str(info.get("phase") or "").strip()
            if phase:
                return phase
        return "unknown"

    def _inject_action_history_into_obs(self) -> None:
        """Attach the tracker's current snapshot onto ``_last_obs_raw`` so
        observation_v3 can emit HISTORY tokens. Called on a freshly-assigned
        obs dict (after ``_update_live_state``) — we own the mutation, no
        bridge reader cares about the underscore-prefixed key.
        """
        if isinstance(self._last_obs_raw, dict):
            self._last_obs_raw["_action_history"] = self._action_history.to_obs_dict()

    def _inject_run_route_snapshot_into_obs(self) -> None:
        """Attach last known map/route runway to reward/build surfaces.

        Card reward observations usually arrive after combat and no longer
        include map legal actions.  RunMemory keeps the most recent compact map
        snapshot so reward guards and token obs can still price future combo
        option value versus orphan risk.
        """
        if isinstance(self._last_obs_raw, dict):
            self._run_memory.attach_route_snapshot_to_obs(self._last_obs_raw)

    def _decorate_bridge_info(self, bridge_info: Any) -> dict[str, Any]:
        info = dict(bridge_info) if isinstance(bridge_info, dict) else {}
        diagnostics = info.get("action_diagnostics")
        diagnostics = dict(diagnostics) if isinstance(diagnostics, dict) else {}
        diagnostics["legal_action_overflow"] = float(self._last_action_overflow)
        diagnostics["raw_legal_action_count"] = float(self._last_raw_legal_action_count)
        diagnostics["blocked_action_drop_count"] = float(self._last_blocked_action_drop_count)
        if isinstance(self._last_actionability, dict):
            diagnostics["transient_only_end_turn"] = bool(
                self._last_actionability.get("transient_only_end_turn", False)
            )
            diagnostics["frontier_stable"] = bool(
                self._last_actionability.get("frontier_stable", True)
            )
        info["action_diagnostics"] = diagnostics
        return info

    def _transition_state(self) -> dict[str, Any]:
        obs = self._last_obs_raw if isinstance(self._last_obs_raw, dict) else {}
        player = obs.get("player") if isinstance(obs.get("player"), dict) else {}
        run = obs.get("run") if isinstance(obs.get("run"), dict) else {}
        combat = obs.get("combat") if isinstance(obs.get("combat"), dict) else {}
        room_model = (
            run.get("room_model")
            or run.get("encounter_id")
            or run.get("encounter")
            or run.get("room_encounter")
            or run.get("room")
            or obs.get("encounter_id")
            or obs.get("encounter")
            or ""
        )
        encounter_id = (
            run.get("encounter_id")
            or run.get("room_model")
            or run.get("encounter")
            or run.get("room_encounter")
            or run.get("room")
            or obs.get("encounter_id")
            or obs.get("encounter")
            or ""
        )

        potions = player.get("potions") if isinstance(player.get("potions"), list) else []
        relics = player.get("relics") if isinstance(player.get("relics"), list) else []
        enemies_out: list[dict[str, Any]] = []
        for enemy in combat.get("enemies") if isinstance(combat.get("enemies"), list) else []:
            if not isinstance(enemy, dict):
                continue
            intent = enemy.get("intent") if isinstance(enemy.get("intent"), dict) else {}
            enemies_out.append(
                {
                    "hp": _float(enemy.get("hp", enemy.get("current_hp"))),
                    "block": _float(enemy.get("block")),
                    "intent": {
                        "total_damage": _float(intent.get("total_damage")),
                        "damage_per_hit": _float(intent.get("damage_per_hit")),
                        "repeats": _float(intent.get("repeats")),
                    },
                }
            )

        return {
            "phase": obs.get("phase"),
            "player": {
                "hp": _float(player.get("hp")),
                "max_hp": _float(player.get("max_hp")),
                "gold": _float(player.get("gold")),
                "potions": list(potions),
                "relics": list(relics),
            },
            "run": {
                "floor": _float(run.get("floor", run.get("total_floor"))),
                "act_id": _parse_act_id(run.get("act_id"), run),
                "act_id_raw": run.get("act_id"),
                "current_act_index": _float(run.get("current_act_index"), default=-1.0),
                "total_floor": _float(run.get("total_floor")),
                "act_floor": _float(run.get("act_floor")),
                "room_type": run.get("room_type") or run.get("state_type"),
                "state_type": run.get("state_type"),
                "room_model": room_model,
                "encounter_id": encounter_id,
                "encounter": run.get("encounter") or obs.get("encounter"),
                "room_encounter": run.get("room_encounter"),
                "room": run.get("room"),
                "active": run.get("active"),
                "game_over": run.get("game_over", run.get("is_game_over")),
            },
            "combat": {
                "block": _float(combat.get("block")),
                "energy": _float(combat.get("energy")),
                "round": _float(combat.get("round")),
                "enemies": enemies_out,
            } if combat else {},
        }

    def _build_info(self, bridge_info: Any, *, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        planner_context = self._planner_context()
        run_obs = (self._last_obs_raw or {}).get("run") or {}
        current_floor_val = run_obs.get("floor")
        if current_floor_val is None:
            current_floor_val = run_obs.get("total_floor")
        try:
            current_floor = int(current_floor_val) if current_floor_val is not None else 0
        except (TypeError, ValueError):
            current_floor = 0
        info: dict[str, Any] = {
            "episode_id": self._episode_id,
            "action_mask": self.action_masks(),
            "legal_action_count": len(self._legal_actions),
            "legal_actions_compact": self.get_compact_legal_actions(),
            "action_overflow": self._last_action_overflow,
            "phase": (self._last_obs_raw or {}).get("phase", "unknown"),
            "episode_mode": "full_run",
            "potion_mechanics_available": True,
            # Floor metrics: surface current AND max-reached so SB3's
            # Monitor (with info_keywords) and episode_terminal event log
            # can aggregate progression. Real bridge used to emit these
            # natively; sim training was flying blind on progression.
            "current_floor": current_floor,
            "max_floor_reached": int(self._max_floor_reached),
            "planner_context": planner_context,
            "transition_state": self._transition_state(),
            "bridge_info": self._decorate_bridge_info(bridge_info),
            "raw_legal_action_count": int(self._last_raw_legal_action_count),
            "raw_legal_actions_compact": list(self._last_raw_legal_actions_compact),
            "blocked_action_drop_count": int(self._last_blocked_action_drop_count),
        }
        if isinstance(self._last_actionability, dict):
            info["actionability"] = dict(self._last_actionability)
        if extra:
            info.update(extra)
        if self.include_debug_info:
            info["legal_actions"] = self._legal_actions
            info["raw_obs"] = self._last_obs_raw
        return info

    def _make_invalid_action_response(self, attempted_action: Any):
        self._inject_action_history_into_obs()
        obs = self.obs_encoder.encode(self._last_obs_raw or {}, self._legal_actions, self._planner_context())
        bridge_info = {
            "action_error": INVALID_ACTION_REASON,
            "truncation_reason": INVALID_ACTION_REASON,
            "action_diagnostics": {
                "invalid_action_selected": 1.0,
            },
        }
        info = self._build_info(
            bridge_info,
            extra={
                "invalid_action_selected": True,
                "invalid_action_index": attempted_action,
                "python_timing_ms": self._python_timing(
                    bridge_roundtrip=0.0,
                    run_memory_update=0.0,
                    obs_encode=0.0,
                    aux_targets=0.0,
                    info_build=0.0,
                    total=0.0,
                ),
            },
        )
        return obs, INVALID_ACTION_REWARD, False, True, info

    def _make_frontier_refreshed_response(
        self,
        attempted_action: Any,
        *,
        reason: str = "frontier_refreshed_before_end_turn",
        refreshed: bool = True,
        high_confidence: bool = False,
    ):
        """Return a non-terminal no-op after blocking a stale EndTurn.

        This is intentionally not an invalid-action truncation: the agent chose
        EndTurn from the mask it was shown, but EnvV2 discovered before
        dispatch that the bridge frontier was stale and now has real actions.
        Returning the refreshed observation lets self-play reselect on the new
        mask without burning an in-game turn.
        """

        self._inject_action_history_into_obs()
        obs = self.obs_encoder.encode(self._last_obs_raw or {}, self._legal_actions, self._planner_context())
        bridge_info = {
            "action_error": reason,
            "step_recovery": reason,
            "action_diagnostics": {
                "frontier_pre_dispatch_refreshed": 1.0 if refreshed else 0.0,
                "frontier_pre_dispatch_end_turn_blocked": 1.0,
                "frontier_pre_dispatch_high_confidence": 1.0 if high_confidence else 0.0,
            },
        }
        info = self._build_info(
            bridge_info,
            extra={
                "frontier_refreshed_before_end_turn": bool(refreshed),
                "frontier_stale_singleton_end_turn_blocked": bool(not refreshed),
                "frontier_refreshed_attempted_action": attempted_action,
                "episode_telemetry": dict(self._episode_telemetry),
                "python_timing_ms": self._python_timing(
                    bridge_roundtrip=0.0,
                    run_memory_update=0.0,
                    obs_encode=0.0,
                    aux_targets=0.0,
                    info_build=0.0,
                    total=0.0,
                ),
            },
        )
        return obs, 0.0, False, False, info



    def _make_step_recovery_response(self, exc: Exception):
        self._episode_id = None
        self._inject_action_history_into_obs()
        obs = self.obs_encoder.encode(self._last_obs_raw or {}, [], self._planner_context())
        bridge_info = {
            "action_error": STEP_RECOVERY_TRUNCATION_REASON,
            "truncation_reason": STEP_RECOVERY_TRUNCATION_REASON,
            "action_diagnostics": {
                "episode_lost": 1.0,
            },
        }
        info = self._build_info(
            bridge_info,
            extra={
                "bridge_episode_lost": True,
                "bridge_exception": str(exc),
                "python_timing_ms": self._python_timing(
                    bridge_roundtrip=0.0,
                    run_memory_update=0.0,
                    obs_encode=0.0,
                    aux_targets=0.0,
                    info_build=0.0,
                    total=0.0,
                ),
            },
        )
        return obs, 0.0, False, True, info

__all__ = [
    'ACTIONABILITY_FAST_POLL_INTERVAL_S',
    'ACTIONABILITY_FAST_WAIT_MS',
    'ACTIONABILITY_REBIND_TIMEOUT_MS',
    'BLOCKED_ACTION_KINDS',
    'BOSS_ACT_FLOORS',
    'BOSS_COMBAT_LOSS_DAMAGE_UNDO_PERCENT_SCALE',
    'BOSS_COMBAT_LOSS_PENALTY_BASE',
    'BOSS_COMBAT_LOSS_PENALTY_MISSING_HP_SCALE',
    'BOSS_ENEMY_HP_DELTA_PERCENT_SCALE',
    'BOSS_FLOOR_ENTRY_BONUS',
    'DISCARD_POTION_ACTION_KIND',
    'EMPTY_POTION_NAMES',
    'END_TURN_WASTE_BASE_PENALTY',
    'END_TURN_WASTE_ENERGY_PENALTY',
    'END_TURN_WASTE_EXTRA_ACTION_PENALTY',
    'END_TURN_WASTE_ZERO_COST_BONUS_PENALTY',
    'ENEMY_HP_DELTA_REWARD_MAX_ABS',
    'ENEMY_HP_DELTA_REWARD_SCALE',
    'ENEMY_HP_SENTINEL_THRESHOLD',
    'EVENT_COMBAT_LOW_HP_THRESHOLD',
    'EVENT_HP_LOSS_LOW_HP_THRESHOLD',
    'FLOOR_CLEAR_BONUS_PER_FLOOR',
    'FLOOR_CLEAR_MIN_FLOOR',
    'FULL_RUN_DEATH_PENALTY_BASE',
    'FULL_RUN_DEATH_PENALTY_LATE_ACT_SCALE',
    'FULL_RUN_DEATH_PENALTY_MISSING_HP_SCALE',
    'INVALID_ACTION_REASON',
    'INVALID_ACTION_REWARD',
    'MAX_ACTIONS',
    'PLAYER_HP_LOSS_REWARD_SCALE',
    'POTION_HOARDING_MAX_PENALTY_ABS',
    'POTION_HOARDING_PENALTY_PER_POTION',
    'POTION_TIMING_QUALITY_SCALE',
    'POTION_TIMING_WASTE_SCALE',
    'POTION_USE_BOSS_BONUS',
    'POTION_USE_ELITE_BONUS',
    'POTION_USE_MONSTER_BONUS',
    'POTION_USE_MONSTER_PENALTY',
    'RECOVERY_MAX_WAIT_MS',
    'RECOVERY_POLL_INTERVAL_S',
    'RESET_READY_MAX_WAIT_MS',
    'RESET_READY_POLL_INTERVAL_S',
    'REST_SITE_SKIP_HEAL_HP_THRESHOLD',
    'REST_SITE_SKIP_HEAL_PENALTY',
    'STARTUP_ACTION_PREFIXES',
    'STEP_RECOVERY_TRUNCATION_REASON',
    'STEP_TRANSITION_RECOVERY_MAX_WAIT_MS',
    'TRANSITION_RECOVERY_MAX_WAIT_MS',
    'ActionHistoryTracker',
    'BridgeClient',
    'BridgeError',
    'CombatMemoryTracker',
    'DenseObservationEncoder',
    'RunMemoryTracker',
    'SlayTheSpire2EnvV2',
    'WorldTokenObservationEncoder',
    'build_aux_targets',
    'compact_legal_actions',
    'compute_potion_timing',
    'strict_end_turn_waste_context',
]
