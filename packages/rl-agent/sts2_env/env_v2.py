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
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .action_compact import compact_legal_actions
from .action_history import ActionHistoryTracker
from .aux_targets import build_aux_targets
from .bridge_client import BridgeClient, BridgeError
from .observation_common import DenseObservationEncoder, MAX_ACTIONS
from .observation_v3 import WorldTokenObservationEncoder
from .combat_memory import CombatMemoryTracker
from .potion_timing import compute_potion_timing
from .end_turn_quality import strict_end_turn_waste_context
from .run_memory import RunMemoryTracker

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
    FULL_RUN_WASTE_BASE as END_TURN_WASTE_BASE_PENALTY,
    FULL_RUN_WASTE_ENERGY as END_TURN_WASTE_ENERGY_PENALTY,
    FULL_RUN_WASTE_EXTRA_ACTION as END_TURN_WASTE_EXTRA_ACTION_PENALTY,
    FULL_RUN_WASTE_ZERO_COST as END_TURN_WASTE_ZERO_COST_BONUS_PENALTY,
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

INVALID_ACTION_REASON = "invalid_action_index"
# These bridge actions are UI/automation controls, not choices the RL policy
# should learn or dispatch.  In particular, when the bridge is in a transient
# combat/window state it may expose only ``automation:start_autoslay``; treating
# that as a legal action lets training either stall on a non-game control or
# start an external autoplayer.
#
# ``discard_potion`` is intentionally *not* in this hard-block list.  The live
# game can enter a forced potion-overflow modal where the only forward action is
# discarding a potion.  If EnvV2 filters that singleton away, the collector
# fabricates a terminal episode on floor 8/9 instead of continuing the run.  We
# still drop discard_potion when real game actions are also visible; only
# singleton/forced discard is allowed through as cleanup.
BLOCKED_ACTION_KINDS = {"automation"}
DISCARD_POTION_ACTION_KIND = "discard_potion"
EMPTY_POTION_NAMES = {"", "[empty]", "empty", "none", "null"}
RECOVERY_POLL_INTERVAL_S = 0.10
RECOVERY_MAX_WAIT_MS = 15_000
TRANSITION_RECOVERY_MAX_WAIT_MS = 60_000
RESET_READY_POLL_INTERVAL_S = 0.50
RESET_READY_MAX_WAIT_MS = 90_000
STEP_TRANSITION_RECOVERY_MAX_WAIT_MS = 5_000
ACTIONABILITY_FAST_WAIT_MS = 150
ACTIONABILITY_FAST_POLL_INTERVAL_S = 0.02
ACTIONABILITY_REBIND_TIMEOUT_MS = 1_000
STEP_RECOVERY_TRUNCATION_REASON = "bridge_episode_lost"
STARTUP_ACTION_PREFIXES = ("main_menu:", "run_mode:", "character_select:")
EVENT_COMBAT_LOW_HP_THRESHOLD = 0.70
EVENT_HP_LOSS_LOW_HP_THRESHOLD = 0.70


def _float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


_ACT_NAME_FRAGMENTS: tuple[tuple[str, float], ...] = (
    ("UNDERDOCKS", 1.0),
    ("UNDERDOCK", 1.0),
    ("ACT_ONE", 1.0),
    ("ACT_1", 1.0),
    ("ACT1", 1.0),
    ("HIVE", 2.0),
    ("ACT_TWO", 2.0),
    ("ACT_2", 2.0),
    ("ACT2", 2.0),
    ("GLORY", 3.0),
    ("ACT_THREE", 3.0),
    ("ACT_3", 3.0),
    ("ACT3", 3.0),
)


def _act_id_from_value(value: Any) -> float:
    """Parse STS2 act ids from both numeric and enum/name bridge payloads.

    Live bridge payloads have used strings like ``ACT.UNDERDOCKS`` instead
    of numeric ``1``.  Plain ``float(value)`` collapses those to 0, which
    makes downstream Act1-clear telemetry permanently false.  Keep this
    helper conservative: return 0 when unknown, but recognize the known act
    enum names plus strings carrying a trailing digit such as ``ACT.1``.
    """
    if value is None or isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return 0.0
    try:
        return float(text)
    except (TypeError, ValueError):
        pass
    for char in reversed(text):
        if char.isdigit():
            return float(char)
    upper = text.upper()
    for fragment, act_id in _ACT_NAME_FRAGMENTS:
        if fragment in upper:
            return float(act_id)
    return 0.0


def _parse_act_id(value: Any, run: dict[str, Any] | None = None) -> float:
    parsed = _act_id_from_value(value)
    if parsed > 0.0:
        return parsed
    if not isinstance(run, dict):
        return 0.0
    for key in ("act_id", "act_id_raw", "act", "act_number", "current_act", "act_name"):
        parsed = _act_id_from_value(run.get(key))
        if parsed > 0.0:
            return parsed
    for key in ("current_act_index", "act_index"):
        if key not in run:
            continue
        index = _float(run.get(key), default=-1.0)
        if index >= 0.0:
            # Godot / sim internals commonly store zero-based act indexes.
            return float(index + 1.0)
    return 0.0


class SlayTheSpire2EnvV2(gym.Env):
    """Gymnasium Env backed by the STS2 bridge env/reset and env/step endpoints."""

    metadata = {"render_modes": ["human"]}

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
        bridge: "BridgeClient | None" = None,
        stuck_watchdog_steps: int = 400,
        seed_pool: list[str] | None = None,
        seed_strategy: str = "round_robin",
    ) -> None:
        super().__init__()

        # Allow external injection — e.g., HeadlessSimBridgeClient driving
        # the frankqwang/sts2-ai C# sim instead of a real Godot game process.
        self.bridge = bridge if bridge is not None else BridgeClient(session_path=session_file)
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
        self._last_blocked_action_drop_count = 0
        self._action_history.reset()
        self._episode_telemetry = self._blank_telemetry()

        bridge_started = time.perf_counter()
        result = self._reset_with_ready_gate(timeout_ms=self.reset_timeout_ms)
        bridge_elapsed_ms = (time.perf_counter() - bridge_started) * 1000.0

        self._episode_id = result["episode_id"]
        self._update_live_state(result)
        direct_after_obs = self._last_obs_raw
        self._inject_action_history_into_obs()
        self._recover_filtered_action_window(timeout_ms=min(self.reset_timeout_ms, RECOVERY_MAX_WAIT_MS))
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
            result = self.bridge.step(
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

        self._update_live_state(result)
        # Keep the bridge's immediate post-action observation for diagnostics
        # that must describe the actual effect of this action.  Later soft
        # rebind/recovery can advance to a different actionable snapshot.
        direct_after_obs = self._last_obs_raw
        reward = float(result.get("reward", 0.0))
        reward += self._enemy_hp_delta_reward(prev_obs, self._last_obs_raw)
        reward += self._player_hp_delta_reward(prev_obs, self._last_obs_raw)
        reward += end_turn_penalty
        # Phase 8.2 reward density: per-floor-clear ladder +
        # boss-damage multiplier. See reward_constants.py for why.
        # These are dense positive signals meant to pull value-function
        # attention toward "go deep / kill boss" trajectories that
        # were otherwise indistinguishable from "grind floor 3-5" in
        # the 800k baseline.
        reward += self._floor_clear_reward(prev_obs, self._last_obs_raw)
        reward += self._boss_damage_bonus_reward(prev_obs, self._last_obs_raw)
        # Phase 8.2b: non-combat decision shaping. Fires only when
        # the action that was just dispatched matches specific kinds
        # (rest / use_potion) so it can't distort combat-step rewards.
        reward += self._rest_site_skip_heal_penalty(prev_obs, legal_action)
        reward += self._potion_use_bonus(prev_obs, legal_action)
        reward += self._potion_timing_step_reward(legal_action, prev_obs, legal_actions_before)
        terminated = bool(result.get("done", False))
        truncated = bool(result.get("truncated", False))
        # Phase 8.2c: end-of-episode hoarding penalty. Unused potions
        # left in inventory at episode end are wasted resources — the
        # penalty creates a gradient against "never use potions" so
        # the ranking the policy sees becomes boss > elite > monster >
        # hoard. Capped at one floor-clear bonus so it can't dominate
        # progression incentive.
        reward += self._potion_hoarding_penalty(
            self._last_obs_raw, terminated=terminated, truncated=truncated,
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

    def action_masks(self) -> np.ndarray:
        from .hp_cost_safety import is_self_lethal_action  # noqa: WPS433

        mask = np.zeros(MAX_ACTIONS, dtype=bool)
        raw_obs = self._last_obs_raw if isinstance(self._last_obs_raw, dict) else None
        for i, action in enumerate(self._legal_actions[:MAX_ACTIONS]):
            if not isinstance(action, dict):
                continue
            # P0-1: hard-mask self-lethal HP-cost actions in the run-mode env too.
            if is_self_lethal_action(action, raw_obs):
                continue
            mask[i] = True
        return mask

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
        pass

    def get_compact_legal_actions(self) -> list[dict[str, Any]]:
        return compact_legal_actions(self._legal_actions)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _normalize_action(self, action: Any) -> int | None:
        try:
            normalized = int(action)
        except (TypeError, ValueError):
            return None
        if normalized < 0:
            return None
        return normalized

    @staticmethod
    def _action_text_blob(action: dict[str, Any]) -> str:
        """Return a conservative lowercase text view of one legal action.

        The live bridge and the headless simulator do not use exactly the
        same rest-site schema.  For tonight's Act1 recovery guard we must
        recognize the *concrete* REST/HEAL option without repeating the old
        bug where the generic surface name "Rest Site" or action id
        ``rest_site:smith`` counted as healing.
        """
        if not isinstance(action, dict):
            return ""
        parts: list[str] = []
        containers = [action]
        payload = action.get("payload")
        if isinstance(payload, dict):
            containers.append(payload)
        for container in containers:
            for key in (
                "kind",
                "action_type",
                "action_id",
                "label",
                "title",
                "name",
                "option_type",
                "canonical_text",
                "description",
            ):
                value = container.get(key)
                if value is not None:
                    parts.append(str(value))
            option = container.get("option") if isinstance(container.get("option"), dict) else {}
            for key in (
                "option_id",
                "id",
                "type",
                "option_type",
                "title",
                "label",
                "name",
                "description",
                "is_enabled",
            ):
                value = option.get(key)
                if value is not None:
                    parts.append(str(value))
        return " ".join(parts).strip().lower()

    @classmethod
    def _is_rest_site_choice_action(cls, action: Any) -> bool:
        """True only for an in-campfire option, not a map node containing rest."""
        if not isinstance(action, dict):
            return False
        containers = [action]
        payload = action.get("payload")
        if isinstance(payload, dict):
            containers.append(payload)
        for container in containers:
            action_id = str(container.get("action_id") or "").strip().lower()
            # After a campfire option is resolved, the bridge exposes a
            # terminal/proceed action on the same rest_site surface.  That is
            # not a player choice between HEAL and SMITH.  Counting it as a
            # rest-site choice creates false rest_heal_exposure_miss and
            # false low-HP "skipped heal" penalties.
            if action_id in {
                "rest_site:proceed",
                "rest_site:continue",
                "rest_site:leave",
                "rest_site:done",
                "rest_site:close",
            }:
                return False
            if action_id.startswith("rest_site:proceed"):
                return False
        for container in containers:
            kind = str(container.get("kind") or container.get("action_type") or "").strip().lower()
            if kind in {"rest", "rest_site", "choose_rest_option"}:
                return True
            action_id = str(container.get("action_id") or "").strip().lower()
            if (
                action_id.startswith("rest_site:")
                or action_id.startswith("choose_rest_option:")
                or action_id.startswith("sim:choose_rest_option")
            ):
                return True
        return False

    @classmethod
    def _is_rest_heal_choice_action(cls, action: Any) -> bool:
        """Detect the concrete HEAL/REST campfire option.

        Important negative examples:
        - ``rest_site:smith`` is NOT heal just because it contains "rest".
        - generic labels like "Rest Site" are NOT heal.
        """
        if not isinstance(action, dict):
            return False

        containers = [action]
        payload = action.get("payload")
        if isinstance(payload, dict):
            containers.append(payload)

        option_identity_tokens: list[str] = []
        option_title_tokens: list[str] = []
        option_description_tokens: list[str] = []
        action_id_tokens: list[str] = []
        label_tokens: list[str] = []
        for container in containers:
            option = container.get("option") if isinstance(container.get("option"), dict) else {}
            for key in ("option_type", "type", "id", "option_id"):
                value = option.get(key)
                if value is not None:
                    option_identity_tokens.append(str(value).strip().lower())
            value = container.get("option_type")
            if value is not None:
                option_identity_tokens.append(str(value).strip().lower())
            for key in ("title", "label", "name"):
                value = option.get(key)
                if value is not None:
                    option_title_tokens.append(str(value).strip().lower())
            for key in ("title", "label", "name"):
                value = container.get(key)
                if value is not None:
                    label_tokens.append(str(value).strip().lower())
            for key in ("description",):
                value = option.get(key)
                if value is not None:
                    option_description_tokens.append(str(value).strip().lower())
                value = container.get(key)
                if value is not None:
                    option_description_tokens.append(str(value).strip().lower())
            action_id = container.get("action_id")
            if action_id is not None:
                action_id_tokens.append(str(action_id).strip().lower())

        # Live bridge payload uses internal ids/types:
        #   option_id="HEAL", option_type="HealRestSiteOption"
        # Previous code collapsed option_type before option_id, so
        # HealRestSiteOption never matched "heal" and the numeric Chinese
        # description ("回复18点生命值。") also missed "回复生命".
        heal_exact = {
            "rest",
            "heal",
            "healing",
            "sleep",
            "campfire_rest",
            "rest_option",
            "heal_option",
            "healrestsiteoption",
            "mend",
            "mendrestsiteoption",
            "休息",
            "治疗",
            "恢復",
            "恢复",
        }
        if any(token in heal_exact for token in option_identity_tokens):
            return True
        if any(token in heal_exact for token in option_title_tokens):
            return True
        if any(
            action_id in {"rest", "heal"}
            or action_id.endswith(":rest")
            or action_id.endswith(":heal")
            or ":rest:" in action_id
            or ":heal:" in action_id
            or action_id.endswith("=rest")
            or action_id.endswith("=heal")
            or action_id.endswith("=rest_option")
            or action_id.endswith("=heal_option")
            for action_id in action_id_tokens
        ):
            return True

        # Require HP/health semantics for substring matches; the bare word
        # "rest" in "Rest Site" is intentionally ignored.
        positive_substrings = (
            "heal",
            "healing",
            "restore hp",
            "restore health",
            "recover hp",
            "recover health",
            "gain hp",
            "回复生命",
            "恢復生命",
            "恢复生命",
            "治疗",
        )
        title_blob = " ".join(option_title_tokens + label_tokens + option_description_tokens).strip()
        if any(token in title_blob for token in positive_substrings):
            return True
        if (
            ("回复" in title_blob or "恢復" in title_blob or "恢复" in title_blob or "治療" in title_blob)
            and ("生命" in title_blob or "hp" in title_blob or "health" in title_blob)
        ):
            return True
        return title_blob in {"rest", "休息"}

    @classmethod
    def _is_rest_smith_choice_action(cls, action: Any) -> bool:
        """Detect the concrete SMITH/UPGRADE campfire option for telemetry.

        This deliberately does not affect masking/guarding; it only gives
        TensorBoard a stable ``env/rest_smith_chosen`` counter so Act1
        recovery can tell "policy never sees smith" from "policy chooses heal
        because HP is low".
        """
        if not isinstance(action, dict):
            return False
        if not cls._is_rest_site_choice_action(action):
            return False
        blob = cls._action_text_blob(action)
        if not blob:
            return False
        smith_tokens = (
            "smith",
            "upgrade",
            "forge",
            "improve",
            "强化",
            "升级",
            "鍛造",
            "锻造",
        )
        return any(token in blob for token in smith_tokens)

    @staticmethod
    def _action_containers(action: Any) -> list[dict[str, Any]]:
        if not isinstance(action, dict):
            return []
        containers = [action]
        payload = action.get("payload")
        if isinstance(payload, dict):
            containers.append(payload)
        return containers

    @classmethod
    def _is_event_option_action(cls, action: Any) -> bool:
        """True for concrete event-option choices exposed by bridge/sim."""
        for container in cls._action_containers(action):
            kind = str(container.get("kind") or container.get("action_type") or "").strip().lower()
            action_id = str(container.get("action_id") or "").strip().lower()
            surface = str(container.get("surface") or "").strip().lower()
            if kind == "event_option" or surface == "event":
                return True
            if (
                action_id.startswith("event_option:")
                or action_id.startswith("choose_event_option:")
                or action_id.startswith("sim:choose_event_option")
            ):
                return True
        return False

    @classmethod
    def _event_option_effect_deltas(cls, action: Any) -> dict[str, Any]:
        """Extract structured event deltas from live or nested action schemas."""
        for container in cls._action_containers(action):
            deltas = container.get("effect_deltas")
            if isinstance(deltas, dict):
                return deltas
            option = container.get("option") if isinstance(container.get("option"), dict) else {}
            deltas = option.get("effect_deltas")
            if isinstance(deltas, dict):
                return deltas
        return {}

    @classmethod
    def _is_event_combat_option(cls, action: Any) -> bool:
        """Detect optional event choices that lead into combat.

        Live bridge effect_deltas are preferred.  The text fallback deliberately
        includes the Chinese "我能打两个" pattern observed in Act1 deaths:
        that first event option leads to a forced singleton "战斗" surface, so
        waiting until the second screen is too late to save the run.
        """
        if not cls._is_event_option_action(action):
            return False
        deltas = cls._event_option_effect_deltas(action)
        if bool(deltas.get("enter_combat")):
            return True
        blob = cls._action_text_blob(action)
        if not blob:
            return False
        combat_tokens = (
            "enter combat",
            "start combat",
            "begin battle",
            "fight",
            "battle",
            "combat",
            "战斗",
            "戰鬥",
            "进入战斗",
            "進入戰鬥",
            "开始战斗",
            "開始戰鬥",
            "遭遇敌人",
            "遭遇敵人",
            "我能打",
            "打两个",
            "打兩個",
            "打一",
            "打二",
        )
        return any(token in blob for token in combat_tokens)

    @classmethod
    def _event_hp_delta_from_action(cls, action: Any) -> float:
        """Best-effort immediate HP delta advertised by an event option."""
        deltas = cls._event_option_effect_deltas(action)
        return _float(deltas.get("hp_delta"), 0.0)

    @classmethod
    def _is_event_hp_loss_option(
        cls,
        action: Any,
        obs: dict[str, Any] | None = None,
    ) -> bool:
        """Detect optional event branches that directly drain player HP.

        Prefer structured ``effect_deltas.hp_delta`` when the bridge/simulator
        provides it.  Some live events still expose only localized labels; the
        Act1 Slippery Bridge death loop is one of those surfaces.  Its risky
        continuation is text like "再撑一会" / "继续" while a safe alternative
        is visible, so keep a narrow room/text fallback instead of treating all
        generic "continue" event buttons as harmful.
        """
        if not cls._is_event_option_action(action):
            return False
        if cls._event_hp_delta_from_action(action) < 0.0:
            return True
        blob = cls._action_text_blob(action)
        if not blob:
            return False
        explicit_risk_tokens = (
            "失去生命",
            "失去生命值",
            "损失生命",
            "损失生命值",
            "lose hp",
            "lose health",
            "lose life",
            "take damage",
            "pay hp",
            "再撑一会",
            "再撐一會",
            "撑一会",
            "撐一會",
            "硬撑",
            "硬撐",
            "强撑",
            "強撐",
            "press on",
            "hold on",
            "keep going",
            "push onward",
            "go further",
        )
        if any(token in blob for token in explicit_risk_tokens):
            return True
        run = obs.get("run") if isinstance(obs, dict) and isinstance(obs.get("run"), dict) else {}
        room_model = str((run or {}).get("room_model") or "").upper()
        if "PUNCH_OFF" in room_model:
            # Punch Off has localized reward/greed branches that may not carry
            # structured effect_deltas yet.  Keep this text fallback
            # room-specific: generic "take" is far too broad on arbitrary
            # events, but in PUNCH_OFF it denotes the risky HP-cost steal/grab
            # branch seen in late-Act1 deaths.
            punch_off_risk_tokens = (
                "顺走",
                "順走",
                "偷",
                "偷走",
                "拿走",
                "抢",
                "搶",
                "take",
                "steal",
                "grab",
            )
            if any(token in blob for token in punch_off_risk_tokens):
                return True
        if "SLIPPERY_BRIDGE" in room_model:
            slippery_continue_tokens = (
                "继续",
                "繼續",
                "continue",
                "再走",
                "再試",
                "再试",
            )
            if any(token in blob for token in slippery_continue_tokens):
                return True
        return False

    @staticmethod
    def _hp_ratio_from_obs(obs: dict[str, Any] | None) -> tuple[float, bool]:
        if not isinstance(obs, dict):
            return 0.0, False
        player = obs.get("player") if isinstance(obs.get("player"), dict) else None
        if not isinstance(player, dict):
            return 0.0, False
        hp = _float(player.get("hp"))
        max_hp = max(_float(player.get("max_hp"), 1.0), 1.0)
        if hp <= 0.0 or max_hp <= 0.0:
            return 0.0, False
        return hp / max_hp, True

    def _apply_low_hp_rest_heal_exposure_filter(
        self,
        legal_actions: list[dict[str, Any]],
        obs: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        """Mask non-heal campfire choices at low HP before the policy sees them.

        The trainer-side override is useful telemetry, but it can fail when
        compact/full action alignment differs.  Filtering at EnvV2's legal
        action exposure layer is safer: the policy's chosen action index now
        truly corresponds to the executed HEAL/REST action, so replay does not
        learn that SMITH caused a heal transition.
        """
        if not isinstance(legal_actions, list) or not legal_actions:
            return legal_actions
        hp_ratio, hp_valid = self._hp_ratio_from_obs(obs)
        if not hp_valid or hp_ratio >= REST_SITE_SKIP_HEAL_HP_THRESHOLD:
            return legal_actions
        rest_indices = [
            idx for idx, action in enumerate(legal_actions)
            if self._is_rest_site_choice_action(action)
        ]
        if not rest_indices:
            return legal_actions
        self._episode_telemetry["rest_heal_exposure_low_hp"] += 1.0
        heal_indices = [
            idx for idx in rest_indices
            if self._is_rest_heal_choice_action(legal_actions[idx])
        ]
        if not heal_indices:
            self._episode_telemetry["rest_heal_exposure_miss"] += 1.0
            try:
                import json

                dump = {
                    "hp_ratio": hp_ratio,
                    "actions": [
                        {
                            "idx": idx,
                            "action_id": str(legal_actions[idx].get("action_id") or ""),
                            "kind": str(legal_actions[idx].get("kind") or ""),
                            "label": str(legal_actions[idx].get("label") or ""),
                            "option": legal_actions[idx].get("option"),
                            "payload": legal_actions[idx].get("payload"),
                            "text": self._action_text_blob(legal_actions[idx])[:500],
                        }
                        for idx in rest_indices
                        if isinstance(legal_actions[idx], dict)
                    ],
                }
                print(
                    "[rest_heal_exposure_miss] "
                    + json.dumps(dump, ensure_ascii=False, default=str)[:2000],
                    flush=True,
                )
            except Exception:
                pass
            return legal_actions
        self._episode_telemetry["rest_heal_exposure_heal_available"] += 1.0
        non_heal_rest_indices = [idx for idx in rest_indices if idx not in set(heal_indices)]
        if not non_heal_rest_indices:
            return legal_actions
        self._episode_telemetry["rest_heal_exposure_forced"] += 1.0
        # Keep non-rest actions if a mixed surface ever slips through; in the
        # normal actions phase _split_actions_phase_actions already returns
        # the rest-site group only.
        heal_set = set(heal_indices)
        rest_set = set(rest_indices)
        return [
            action
            for idx, action in enumerate(legal_actions)
            if idx not in rest_set or idx in heal_set
        ]

    def _apply_low_hp_event_combat_exposure_filter(
        self,
        legal_actions: list[dict[str, Any]],
        obs: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        """Hide optional event-combat branches when HP is too low.

        The policy was repeatedly dying before Act1 boss by choosing an event
        chain like "我能打两个" -> forced "战斗".  Once the chain advances to the
        singleton combat option the model has no alternative, so the guard must
        operate at the first optional event surface.  Forced/singleton event
        combats are still left visible to avoid deadlocking mandatory events.
        """
        if not isinstance(legal_actions, list) or not legal_actions:
            return legal_actions
        event_indices = [
            idx for idx, action in enumerate(legal_actions)
            if self._is_event_option_action(action)
        ]
        if not event_indices:
            return legal_actions
        combat_indices = [
            idx for idx in event_indices
            if self._is_event_combat_option(legal_actions[idx])
        ]
        if not combat_indices:
            return legal_actions
        self._episode_telemetry["event_combat_option_available"] += 1.0
        hp_ratio, hp_valid = self._hp_ratio_from_obs(obs)
        if not hp_valid or hp_ratio >= EVENT_COMBAT_LOW_HP_THRESHOLD:
            return legal_actions
        self._episode_telemetry["event_combat_option_low_hp_available"] += 1.0
        combat_set = set(combat_indices)
        noncombat_event_indices = [idx for idx in event_indices if idx not in combat_set]
        if not noncombat_event_indices:
            return legal_actions
        self._episode_telemetry["event_combat_option_safe_alternative"] += 1.0
        self._episode_telemetry["event_combat_option_masked_low_hp"] += 1.0
        return [
            action
            for idx, action in enumerate(legal_actions)
            if idx not in combat_set
        ]

    def _apply_low_hp_event_hp_loss_exposure_filter(
        self,
        legal_actions: list[dict[str, Any]],
        obs: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        """Hide optional event HP-loss branches when a safe exit exists.

        The live Act1 Slippery Bridge can expose a repeated low-HP option
        ("再撑一会") plus a safe alternative.  Existing event-combat masking did
        not catch it, so the policy could spend HP down to 0 before seeing the
        boss.  This guard is exposure-level (before policy/action indexing)
        and leaves singleton/forced event choices visible to avoid deadlocks.
        """
        if not isinstance(legal_actions, list) or not legal_actions:
            return legal_actions
        event_indices = [
            idx for idx, action in enumerate(legal_actions)
            if self._is_event_option_action(action)
        ]
        if not event_indices:
            return legal_actions
        risky_indices = [
            idx for idx in event_indices
            if self._is_event_hp_loss_option(legal_actions[idx], obs)
        ]
        if not risky_indices:
            return legal_actions
        self._episode_telemetry["event_hp_loss_option_available"] += 1.0
        hp_ratio, hp_valid = self._hp_ratio_from_obs(obs)
        if not hp_valid or hp_ratio >= EVENT_HP_LOSS_LOW_HP_THRESHOLD:
            return legal_actions
        self._episode_telemetry["event_hp_loss_option_low_hp_available"] += 1.0
        risky_set = set(risky_indices)
        safe_event_indices = [idx for idx in event_indices if idx not in risky_set]
        if not safe_event_indices:
            return legal_actions
        self._episode_telemetry["event_hp_loss_option_safe_alternative"] += 1.0
        self._episode_telemetry["event_hp_loss_option_masked_low_hp"] += 1.0
        return [
            action
            for idx, action in enumerate(legal_actions)
            if idx not in risky_set
        ]

    def _update_live_state(self, result: dict[str, Any]) -> None:
        phase = self._extract_phase(result)
        legal_actions = result.get("legal_actions", [])
        self._last_raw_legal_action_count = len(legal_actions) if isinstance(legal_actions, list) else 0
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

    def _combat_enemy_total_hp(self, obs: dict[str, Any] | None) -> float:
        if not isinstance(obs, dict):
            return 0.0
        combat = obs.get("combat")
        if not isinstance(combat, dict):
            return 0.0
        enemies = combat.get("enemies")
        if not isinstance(enemies, list):
            return 0.0

        total = 0.0
        for enemy in enemies:
            if not isinstance(enemy, dict):
                continue
            hp = float(enemy.get("hp", enemy.get("current_hp")) or 0.0)
            if hp > ENEMY_HP_SENTINEL_THRESHOLD:
                continue
            total += hp
        return total

    def _enemy_hp_delta_reward(self, before_obs: dict[str, Any] | None, after_obs: dict[str, Any] | None) -> float:
        before_total = self._combat_enemy_total_hp(before_obs)
        after_total = self._combat_enemy_total_hp(after_obs)
        if before_total <= 0.0 and after_total <= 0.0:
            return 0.0

        # Bridge empties `combat.enemies` the moment a combat ends (both
        # death transitions and victory transitions). If the player died
        # with enemies still alive, crediting (before_total - 0) emits a
        # false "+5.66 kill reward" on the defeat step. Guard: skip the
        # delta when after-state has empty enemies AND player is dead.
        # Full-run episode doesn't usually terminate on player death (run
        # ends), but combat-end transitions still drop enemies to [].
        after_player_hp = 0.0
        if isinstance(after_obs, dict):
            player = after_obs.get("player") if isinstance(after_obs.get("player"), dict) else {}
            after_player_hp = _float((player or {}).get("hp"))
        # Both "enemies key missing (sim)" and "enemies=[] (live)" reduce
        # to after_total==0 through _combat_enemy_total_hp. On player-death
        # transitions we see that AND player_hp<=0; skip the delta.
        if (
            before_total > 0.0
            and after_total <= 0.0
            and after_player_hp <= 0.0
        ):
            return 0.0

        raw = (before_total - after_total) * ENEMY_HP_DELTA_REWARD_SCALE
        if raw > ENEMY_HP_DELTA_REWARD_MAX_ABS:
            return ENEMY_HP_DELTA_REWARD_MAX_ABS
        if raw < -ENEMY_HP_DELTA_REWARD_MAX_ABS:
            return -ENEMY_HP_DELTA_REWARD_MAX_ABS
        return raw

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

    def _is_boss_encounter(self, obs: dict[str, Any] | None) -> bool:
        """True when the observation represents a boss-room combat.

        Primary signal is ``run.state_type == "boss"`` (sim emits it, real
        bridge emits a compatible ``room_type``). Falls back to the
        canonical act-boss floor list (17/34/51) for environments that
        don't populate the string tag. Having two independent signals
        keeps the bonus from over-firing on mis-tagged rooms — both
        must at least not contradict the boss-ness judgment.
        """
        if not isinstance(obs, dict):
            return False
        run = obs.get("run") if isinstance(obs.get("run"), dict) else {}
        state_type = str(run.get("state_type") or run.get("room_type") or "").strip().lower()
        if state_type == "boss":
            return True
        floor_val = run.get("floor")
        try:
            floor = int(floor_val) if floor_val is not None else 0
        except (TypeError, ValueError):
            floor = 0
        if floor in BOSS_ACT_FLOORS:
            # Only treat as boss if we're actually IN combat (avoids
            # awarding bonus for walking onto the boss tile without the
            # encounter starting yet).
            combat = obs.get("combat") if isinstance(obs.get("combat"), dict) else None
            if isinstance(combat, dict) and combat.get("enemies"):
                return True
        return False

    def _encounter_tier_from_obs(self, obs: dict[str, Any] | None) -> str:
        """Best-effort encounter tier for shared potion-timing shaping.

        Full-run EnvV2 sees build / route / combat screens, while the
        shared potion evaluator only needs a coarse combat tier to decide
        whether a potion is worth saving.  Prefer the bridge/sim room tag
        when present, fall back to the stricter boss helper so act-boss
        floor snapshots without a string tag still get boss weighting.
        """
        if self._is_boss_encounter(obs):
            return "boss"
        if not isinstance(obs, dict):
            return "normal"
        run = obs.get("run") if isinstance(obs.get("run"), dict) else {}
        state_type = str(run.get("state_type") or run.get("room_type") or "").strip().lower()
        if state_type in {"elite", "miniboss"}:
            return "elite"
        if state_type == "weak":
            return "weak"
        return "normal"

    def _boss_damage_bonus_reward(
        self,
        before_obs: dict[str, Any] | None,
        after_obs: dict[str, Any] | None,
    ) -> float:
        """Additive bonus on damage dealt during boss encounters.

        Full-run previously used raw HP × multiplier here, so a high-HP boss
        could hand out far more positive shaping than the terminal loss could
        undo.  Mirror CombatSandbox's percent-based shaping: cumulative boss
        damage bonus is bounded by ``BOSS_ENEMY_HP_DELTA_PERCENT_SCALE`` for a
        full kill, independent of absolute boss HP.  Sign-asymmetric: only
        POSITIVE damage (enemy losing HP) is rewarded.
        """
        if not self._is_boss_encounter(before_obs) and not self._is_boss_encounter(after_obs):
            return 0.0
        # Count any step where one side of the transition was a boss
        # encounter, even if no damage this tick (captures block / setup
        # turns so we can see "boss engagement density" not just dmg).
        self._episode_telemetry["boss_encounter_steps"] += 1.0
        before_total = self._combat_enemy_total_hp(before_obs)
        after_total = self._combat_enemy_total_hp(after_obs)
        if before_total <= 0.0:
            return 0.0
        # Same bridge contract edge case as _enemy_hp_delta_reward():
        # combat.enemies is cleared to [] on both victory and defeat.  On a
        # defeat step with enemies still alive, interpreting [] as 0 HP would
        # award a full boss-damage bonus (often 200+ raw HP), teaching the
        # policy that dying on boss is equivalent to killing it.  If the
        # after-state has no enemy HP AND the player is dead, count the guard
        # for diagnostics but emit no reward / no boss_damage_dealt_raw.
        after_player_hp = 0.0
        if isinstance(after_obs, dict):
            player = after_obs.get("player") if isinstance(after_obs.get("player"), dict) else {}
            after_player_hp = _float((player or {}).get("hp"))
        if before_total > 0.0 and after_total <= 0.0 and after_player_hp <= 0.0:
            self._episode_telemetry["boss_damage_death_clear_guarded"] += 1.0
            # When the player dies, the bridge can clear the enemy list in the
            # same post-action snapshot.  Treating ``before_total -> 0`` as
            # damage would grant false boss-kill credit and pollute Act1
            # diagnostics.  Keep the legacy raw-damage field at zero and expose
            # the suppressed remaining HP under an explicitly named metric.
            self._episode_telemetry["boss_damage_death_clear_guarded_remaining_hp_raw"] += float(
                before_total
            )
            self._episode_telemetry["boss_damage_death_clear_guarded_raw"] += 0.0
            return 0.0
        raw_damage = max(before_total - after_total, 0.0)
        if raw_damage <= 0.0:
            return 0.0
        damage_dealt_before = max(_float(self._episode_telemetry.get("boss_damage_dealt_raw")), 0.0)
        self._episode_telemetry["boss_damage_dealt_raw"] += float(raw_damage)
        boss_hp_estimate = max(damage_dealt_before + before_total, before_total, raw_damage)
        damage_ratio = (
            float(np.clip(raw_damage / boss_hp_estimate, 0.0, 1.0))
            if boss_hp_estimate > 0.0
            else 0.0
        )
        bonus = damage_ratio * float(BOSS_ENEMY_HP_DELTA_PERCENT_SCALE)
        self._episode_telemetry["boss_damage_bonus_total"] += float(bonus)
        return bonus

    def _boss_death_terminal_penalty(
        self,
        before_obs: dict[str, Any] | None,
        after_obs: dict[str, Any] | None,
        *,
        terminated: bool,
        truncated: bool,
    ) -> float:
        """One-shot full-run penalty for dying in a boss fight.

        Dense boss damage shaping made "reach boss, deal damage, die" highly
        positive in full-run training.  CombatSandbox already has boss terminal
        loss shaping, but EnvV2 did not.  Apply a terminal-only loss penalty
        here so Act1 boss death is clearly worse than living into Act2.
        """
        if not terminated or truncated:
            return 0.0
        if not self._is_boss_encounter(before_obs) and not self._is_boss_encounter(after_obs):
            return 0.0
        player = after_obs.get("player") if isinstance(after_obs, dict) and isinstance(after_obs.get("player"), dict) else {}
        hp = _float((player or {}).get("hp"))
        max_hp = max(_float((player or {}).get("max_hp"), 1.0), 1.0)
        if hp > 0.0:
            return 0.0

        missing_hp_ratio = float(np.clip((max_hp - max(hp, 0.0)) / max_hp, 0.0, 1.0))
        damage_dealt = max(_float(self._episode_telemetry.get("boss_damage_dealt_raw")), 0.0)
        remaining_guarded = max(
            _float(self._episode_telemetry.get("boss_damage_death_clear_guarded_remaining_hp_raw")),
            0.0,
        )
        # If the final bridge snapshot did not trigger the death-clear guard,
        # fall back to the pre-step enemy HP as remaining boss HP.
        if remaining_guarded <= 0.0:
            remaining_guarded = max(self._combat_enemy_total_hp(before_obs), 0.0)
        boss_hp_estimate = max(damage_dealt + remaining_guarded, damage_dealt, remaining_guarded)
        damage_ratio = (
            float(np.clip(damage_dealt / boss_hp_estimate, 0.0, 1.0))
            if boss_hp_estimate > 0.0
            else 0.0
        )

        magnitude = (
            float(BOSS_COMBAT_LOSS_PENALTY_BASE)
            + float(BOSS_COMBAT_LOSS_PENALTY_MISSING_HP_SCALE) * missing_hp_ratio
            + float(BOSS_COMBAT_LOSS_DAMAGE_UNDO_PERCENT_SCALE) * damage_ratio
        )
        penalty = -magnitude
        self._episode_telemetry["boss_death_terminal_penalty_events"] += 1.0
        self._episode_telemetry["boss_death_terminal_penalty_total"] += float(penalty)
        self._episode_telemetry["boss_death_terminal_missing_hp_ratio"] = float(missing_hp_ratio)
        self._episode_telemetry["boss_death_terminal_damage_ratio"] = float(damage_ratio)
        return penalty

    def _rest_site_skip_heal_penalty(
        self,
        before_obs: dict[str, Any] | None,
        action: dict[str, Any] | None,
    ) -> float:
        """Penalize picking a non-HEAL rest-site option when low on HP.

        Magnitude stays strictly below FLOOR_CLEAR_BONUS_PER_FLOOR so
        the policy can never prefer "skip the campfire tile entirely
        on the map" over "take the campfire tile and pick SMITH". The
        within-campfire contrast between HEAL and non-HEAL is what we
        want biased, not the campfire-vs-monster decision at map time.

        Activates only when:
          1. action kind is a rest-site choice
          2. the chosen option isn't HEAL/REST
          3. player HP ratio (pre-step) is below the threshold
        """
        if not isinstance(action, dict):
            return 0.0
        if not self._is_rest_site_choice_action(action):
            return 0.0
        heal_picked = self._is_rest_heal_choice_action(action)
        # Telemetry: always count rest-site encounters + the HEAL/non-HEAL
        # split, regardless of HP threshold. Helps diagnose "is the
        # policy even reaching campfires" vs "is it choosing correctly".
        self._episode_telemetry["rest_site_encounters"] += 1.0
        if heal_picked:
            self._episode_telemetry["rest_heal_chosen"] += 1.0
            return 0.0
        if self._is_rest_smith_choice_action(action):
            self._episode_telemetry["rest_smith_chosen"] += 1.0
        self._episode_telemetry["rest_skip_heal_chosen"] += 1.0

        hp_ratio, hp_valid = self._hp_ratio_from_obs(before_obs)
        if not hp_valid:
            return 0.0
        if hp_ratio >= REST_SITE_SKIP_HEAL_HP_THRESHOLD:
            return 0.0
        penalty = float(REST_SITE_SKIP_HEAL_PENALTY)
        self._episode_telemetry["rest_skip_heal_at_low_hp"] += 1.0
        self._episode_telemetry["rest_penalty_total"] += penalty
        return penalty

    def _potion_use_bonus(
        self,
        before_obs: dict[str, Any] | None,
        action: dict[str, Any] | None,
    ) -> float:
        """Encourage use_potion actions, with context-dependent scaling.

        STS potions are single-use combat consumables that, by the
        800k baseline, the policy had learned to hoard almost
        indefinitely. Base hp/damage rewards didn't distinguish them
        enough from card plays to overcome the implicit "save it for
        later" bias. This flat bonus lifts the expected value of
        use_potion slightly above an equivalent-damage card play, and
        the boss/elite multipliers concentrate the bias where potions
        actually matter.

        Only fires for ``use_potion`` — ``discard_potion`` gets
        nothing (discarding is itself a waste signal).
        """
        if not isinstance(action, dict):
            return 0.0
        kind = str(action.get("kind") or "").lower()
        if kind == "discard_potion":
            self._episode_telemetry["potion_discard_count"] += 1.0
            return 0.0
        if kind != "use_potion":
            return 0.0
        self._episode_telemetry["potion_use_count"] += 1.0
        # Encounter-scoped absolute bonuses. Unlike the earlier
        # base×multiplier scheme, non-boss / non-elite use gets
        # MONSTER_BONUS (default 0) or MONSTER_PENALTY (default 0)
        # — there's no longer an unconditional positive gradient for
        # use_potion in ordinary combat. First-run telemetry showed
        # the previous base bonus caused the policy to burn all
        # potions on floor 3-7 monsters before reaching the boss.
        if self._is_boss_encounter(before_obs):
            bonus = float(POTION_USE_BOSS_BONUS)
            self._episode_telemetry["potion_use_boss_count"] += 1.0
            self._episode_telemetry["potion_use_bonus_total"] += bonus
            return bonus
        run = before_obs.get("run") if isinstance(before_obs, dict) and isinstance(before_obs.get("run"), dict) else {}
        state_type = str(run.get("state_type") or run.get("room_type") or "").strip().lower()
        if state_type in {"elite", "miniboss"}:
            bonus = float(POTION_USE_ELITE_BONUS)
            self._episode_telemetry["potion_use_elite_count"] += 1.0
            self._episode_telemetry["potion_use_bonus_total"] += bonus
            return bonus
        # Monster-fight use — net signal depends on whether the
        # optional penalty is configured. Default config gives 0.
        monster_signal = float(POTION_USE_MONSTER_BONUS) + float(POTION_USE_MONSTER_PENALTY)
        self._episode_telemetry["potion_use_bonus_total"] += monster_signal
        return monster_signal

    def _potion_timing_step_reward(
        self,
        action: dict[str, Any] | None,
        before_obs: dict[str, Any] | None,
        legal_actions: list[dict[str, Any]] | None,
    ) -> float:
        """Reward good potion timing and penalize waste in full-run EnvV2.

        `combat_env.py` already uses the shared `compute_potion_timing`
        evaluator; the full-run path was only counting potion use, so
        value targets could not distinguish "Blood Potion saves a lethal
        turn" from "energy potion with no follow-up on a safe hallway".
        This wiring gives the full-run replay the same dense timing signal
        without adding a hard block that could forbid legitimate saves.
        """
        if not isinstance(action, dict):
            return 0.0
        kind = str(action.get("kind") or "").lower()
        action_id = str(action.get("action_id") or "").lower()
        if kind not in {"use_potion", "potion"} and not action_id.startswith("use_potion:"):
            return 0.0
        if not isinstance(before_obs, dict):
            return 0.0

        player = before_obs.get("player") if isinstance(before_obs.get("player"), dict) else {}
        combat = before_obs.get("combat") if isinstance(before_obs.get("combat"), dict) else {}
        energy = _float(player.get("energy", combat.get("energy", 0.0)))

        try:
            profile = compute_potion_timing(
                action,
                before_obs,
                legal_actions or [],
                None,
                energy,
                encounter_tier=self._encounter_tier_from_obs(before_obs),
            )
        except Exception:
            # Reward shaping must never turn an otherwise valid bridge
            # transition into a training crash.  The planner still has its
            # own timing metrics; this path is best-effort value shaping.
            return 0.0
        if not profile.get("is_potion"):
            return 0.0

        use_quality = _float(profile.get("use_quality"))
        waste_risk = _float(profile.get("waste_risk"))
        urgent = bool(
            profile.get("urgent")
            or profile.get("lethal")
            or profile.get("prevent_lethal")
            or profile.get("prevent_major_loss")
            or profile.get("mechanism_answer")
        )
        bad_timing = bool(
            profile.get("low_urgency")
            or profile.get("save_recommended")
            or profile.get("no_followup")
            or profile.get("block_waste")
            or profile.get("overkill")
        )
        reward = 0.0
        # Do not let the 0.08 baseline / weakly-positive profile reinforce
        # burning potions in hallway fights when the only alternative is End
        # Turn.  Only urgent or clearly high-quality use should be positive;
        # explicitly bad timing becomes negative even if use_quality was
        # clipped just above zero.
        if urgent or (use_quality >= 0.45 and not bad_timing):
            reward += float(POTION_TIMING_QUALITY_SCALE) * use_quality
            self._episode_telemetry["potion_timing_quality_events"] += 1.0
        elif bad_timing:
            effective_waste = max(float(waste_risk), 0.35)
            reward -= float(POTION_TIMING_WASTE_SCALE) * effective_waste
            self._episode_telemetry["potion_timing_waste_events"] += 1.0
        elif waste_risk > 0.0:
            reward -= float(POTION_TIMING_WASTE_SCALE) * waste_risk
            self._episode_telemetry["potion_timing_waste_events"] += 1.0
        self._episode_telemetry["potion_timing_reward_total"] += float(reward)
        return float(reward)

    @staticmethod
    def _count_nonempty_potions(obs: dict[str, Any] | None) -> int:
        """How many real potions are currently in the inventory.

        STS2 represents empty potion slots as the string "[empty]" (or
        a dict with that title). Only count actual potions. Returns 0
        on malformed obs.
        """
        if not isinstance(obs, dict):
            return 0
        player = obs.get("player") if isinstance(obs.get("player"), dict) else None
        if not isinstance(player, dict):
            return 0
        potions = player.get("potions")
        if not isinstance(potions, list):
            return 0
        count = 0
        for potion in potions:
            if isinstance(potion, str):
                s = potion.strip()
                if s.lower() not in EMPTY_POTION_NAMES:
                    count += 1
            elif isinstance(potion, dict):
                if bool(potion.get("empty")):
                    continue
                title = str(potion.get("title") or potion.get("id") or "").strip()
                if title.lower() not in EMPTY_POTION_NAMES:
                    count += 1
        return count

    @staticmethod
    def _count_empty_potion_slots(obs: dict[str, Any] | None) -> int:
        """How many potion slots are observably empty.

        Return 0 for missing/malformed payloads so the filter fails open and
        does not hide a real overflow modal. If the bridge explicitly reports
        any empty slot, a singleton ``discard_potion`` frontier is not a true
        overflow requirement and should be treated as transient/fake.
        """
        if not isinstance(obs, dict):
            return 0
        player = obs.get("player") if isinstance(obs.get("player"), dict) else None
        if not isinstance(player, dict):
            return 0
        potions = player.get("potions")
        if not isinstance(potions, list):
            return 0

        count = 0
        for potion in potions:
            if isinstance(potion, str):
                if potion.strip().lower() in EMPTY_POTION_NAMES:
                    count += 1
            elif isinstance(potion, dict):
                if bool(potion.get("empty")):
                    count += 1
                    continue
                title = str(
                    potion.get("title")
                    or potion.get("id")
                    or potion.get("name")
                    or ""
                ).strip()
                if title.lower() in EMPTY_POTION_NAMES:
                    count += 1
        return int(count)

    @staticmethod
    def _potion_slots_dump(obs: dict[str, Any] | None) -> list[dict[str, Any]]:
        """Compact potion-slot dump for full-run diagnostics.

        Full-run EnvV2 historically only surfaced aggregate
        ``potion_discard_count`` / ``potion_use_count``.  That made it
        impossible to tell whether the 4-6 discards per Act1 attempt were
        forced overflow cleanup, optional policy waste, or stale bridge slot
        state.  Keep this payload small and JSON-friendly so MuZeroTrainer can
        append it directly to diagnostics/potion_transitions.jsonl.
        """
        if not isinstance(obs, dict):
            return []
        player = obs.get("player") if isinstance(obs.get("player"), dict) else {}
        potions = player.get("potions") if isinstance(player, dict) else None
        if not isinstance(potions, list):
            return []

        out: list[dict[str, Any]] = []
        for idx, potion in enumerate(potions):
            slot = idx
            potion_id: str | None = None
            title: str | None = None
            empty = False
            is_usable: Any = None
            is_queued: Any = None
            raw_kind = type(potion).__name__
            if isinstance(potion, dict):
                try:
                    slot = int(potion.get("slot", potion.get("slot_index", idx)))
                except (TypeError, ValueError):
                    slot = idx
                potion_id = str(
                    potion.get("id")
                    or potion.get("potion_id")
                    or potion.get("internal_id")
                    or ""
                ).strip() or None
                title = str(
                    potion.get("title")
                    or potion.get("name")
                    or potion.get("label")
                    or potion_id
                    or ""
                ).strip() or None
                empty = bool(potion.get("empty"))
                is_usable = potion.get("is_usable", potion.get("usable"))
                is_queued = potion.get("is_queued", potion.get("queued"))
            elif potion is None:
                empty = True
            else:
                title = str(potion).strip() or None
                potion_id = title

            marker = str(title or potion_id or "").strip().lower()
            if marker in EMPTY_POTION_NAMES:
                empty = True
            out.append(
                {
                    "slot": int(slot),
                    "id": potion_id,
                    "title": title,
                    "empty": bool(empty),
                    "is_usable": is_usable,
                    "is_queued": is_queued,
                    "raw_kind": raw_kind,
                }
            )
        return out

    @staticmethod
    def _state_version_from_obs(obs: dict[str, Any] | None) -> int:
        if not isinstance(obs, dict):
            return 0
        for container in (obs.get("meta"), obs):
            if not isinstance(container, dict):
                continue
            for key in ("state_version", "stateVersion"):
                if key in container:
                    try:
                        return int(container.get(key) or 0)
                    except (TypeError, ValueError):
                        return 0
        return 0

    @staticmethod
    def _potion_action_slot(action: dict[str, Any] | None) -> int:
        if not isinstance(action, dict):
            return -1
        def _bounded_int(value: Any) -> int | None:
            try:
                if value is None or str(value).strip() == "":
                    return None
                slot = int(value)
                if 0 <= slot < 10:
                    return slot
            except (TypeError, ValueError):
                return None
            return None

        sources: list[dict[str, Any]] = [action]
        for key in ("target", "potion", "payload"):
            value = action.get(key)
            if isinstance(value, dict):
                sources.append(value)
        for source in sources:
            for key in ("potion_slot", "slot", "slot_index", "potion_index", "potion_idx", "index"):
                if key not in source:
                    continue
                slot = _bounded_int(source.get(key))
                if slot is not None:
                    return slot
        action_id = str(action.get("action_id") or "").strip().lower()
        if action_id.startswith("use_potion:") or action_id.startswith("discard_potion:"):
            parts = action_id.split(":")[1:]
            leading_numbers: list[int] = []
            for part in parts:
                if not part.isdigit():
                    break
                try:
                    leading_numbers.append(int(part))
                except (TypeError, ValueError):
                    break
            # Live bridge ids are {kind}:{playerIndex}:{slotIndex}[:target].
            # Legacy tests/logs also use {kind}:{slotIndex}.  Prefer slotIndex
            # when both player and slot are present so diagnostics point to the
            # actual potion changed instead of always player 0.
            if len(leading_numbers) >= 2:
                slot = _bounded_int(leading_numbers[1])
                if slot is not None:
                    return slot
            if len(leading_numbers) == 1:
                slot = _bounded_int(leading_numbers[0])
                if slot is not None:
                    return slot
        return -1

    def _build_potion_transition_record(
        self,
        *,
        action: dict[str, Any] | None,
        prev_obs: dict[str, Any] | None,
        after_obs: dict[str, Any] | None,
        legal_actions_before: list[dict[str, Any]] | None,
        bridge_info: dict[str, Any] | None,
        reward: float,
        terminated: bool,
        truncated: bool,
    ) -> dict[str, Any] | None:
        """Emit full-run use/discard potion diagnostics for MuZeroTrainer.

        combat_env already emits use-potion transitions, but the full-run
        EnvV2 path did not.  Act1 recovery currently observes many
        ``discard_potion`` selections and no per-action explanation; this
        record makes the selected action, inventory before/after, and
        "forced singleton discard" status visible without changing policy.
        """
        if not isinstance(action, dict):
            return None
        kind = str(action.get("kind") or "").strip().lower()
        action_id = str(action.get("action_id") or "").strip()
        action_id_lower = action_id.lower()
        if kind not in {"use_potion", "potion", "discard_potion"} and not (
            action_id_lower.startswith("use_potion:") or action_id_lower.startswith("discard_potion:")
        ):
            return None

        before_dump = self._potion_slots_dump(prev_obs)
        after_dump = self._potion_slots_dump(after_obs)
        before_count = sum(1 for slot in before_dump if not bool(slot.get("empty")))
        after_count = sum(1 for slot in after_dump if not bool(slot.get("empty")))
        slot_index = self._potion_action_slot(action)
        before_slot = next((slot for slot in before_dump if int(slot.get("slot", -1)) == slot_index), None)
        after_slot = next((slot for slot in after_dump if int(slot.get("slot", -1)) == slot_index), None)

        legal_before = legal_actions_before if isinstance(legal_actions_before, list) else []
        unblocked_before = [
            candidate
            for candidate in legal_before
            if isinstance(candidate, dict) and not self._state_action_is_blocked(candidate, legal_before)
        ]
        non_discard_unblocked = [
            candidate
            for candidate in unblocked_before
            if str(candidate.get("kind") or "").strip().lower() != DISCARD_POTION_ACTION_KIND
        ]

        before_run = prev_obs.get("run") if isinstance(prev_obs, dict) and isinstance(prev_obs.get("run"), dict) else {}
        after_run = after_obs.get("run") if isinstance(after_obs, dict) and isinstance(after_obs.get("run"), dict) else {}
        run = before_run if before_run else after_run
        room_model = (
            run.get("room_model")
            or run.get("encounter_id")
            or run.get("encounter")
            or run.get("room_encounter")
            or run.get("room")
            or ""
        )
        info = bridge_info if isinstance(bridge_info, dict) else {}
        execute_ok = not bool(info.get("error") or info.get("action_error"))

        return {
            "event": "discard_potion_transition" if kind == DISCARD_POTION_ACTION_KIND or action_id_lower.startswith("discard_potion:") else "use_potion_transition",
            "episode_id": self._episode_id,
            "action_id": action_id,
            "kind": kind,
            "potion_slot": int(slot_index),
            "potion_id_before": (before_slot or {}).get("id"),
            "potion_title_before": (before_slot or {}).get("title"),
            "potion_slot_before": before_slot,
            "potion_slot_after": after_slot,
            "potion_slots_before": before_dump,
            "potion_slots_after": after_dump,
            "potion_count_before": int(before_count),
            "potion_count_after": int(after_count),
            "legal_action_count_before": int(len(legal_before)),
            "unblocked_action_count_before": int(len(unblocked_before)),
            "non_discard_unblocked_action_count_before": int(len(non_discard_unblocked)),
            "forced_singleton_discard": bool(
                (kind == DISCARD_POTION_ACTION_KIND or action_id_lower.startswith("discard_potion:"))
                and len(unblocked_before) == 1
                and not non_discard_unblocked
            ),
            "optional_discard_with_alternative": bool(
                (kind == DISCARD_POTION_ACTION_KIND or action_id_lower.startswith("discard_potion:"))
                and len(non_discard_unblocked) > 0
            ),
            "execute_ok": bool(execute_ok),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "reward_after_shaping": float(reward),
            "state_version_before": self._state_version_from_obs(prev_obs),
            "state_version_after": self._state_version_from_obs(after_obs),
            "floor": _float(run.get("floor", run.get("total_floor", run.get("act_floor")))),
            "room_type": run.get("room_type") or run.get("state_type"),
            "room_model": room_model,
        }

    def _potion_hoarding_penalty(
        self,
        final_obs: dict[str, Any] | None,
        *,
        terminated: bool,
        truncated: bool,
    ) -> float:
        """One-shot penalty at episode end per unused potion in inventory.

        Fires on both natural termination (death/victory) and watchdog
        truncation — any end-of-episode unused potion is a wasted
        resource regardless of cause. Capped at
        POTION_HOARDING_MAX_PENALTY_ABS so the total penalty can never
        exceed one floor-clear bonus, preserving the "policy should
        still prefer to have potions over not having them" invariant.
        """
        if not (terminated or truncated):
            return 0.0
        unused = self._count_nonempty_potions(final_obs)
        if unused <= 0:
            return 0.0
        raw = unused * float(POTION_HOARDING_PENALTY_PER_POTION)
        cap = float(POTION_HOARDING_MAX_PENALTY_ABS)
        # Keep the sign; clip magnitude.
        penalty = max(raw, -cap) if raw < 0 else min(raw, cap)
        self._episode_telemetry["potion_hoarding_unused_at_end"] = float(unused)
        self._episode_telemetry["potion_hoarding_penalty_total"] += penalty
        return penalty

    def _full_run_death_terminal_penalty(
        self,
        before_obs: dict[str, Any] | None,
        after_obs: dict[str, Any] | None,
        *,
        terminated: bool,
        truncated: bool,
    ) -> float:
        """One-shot penalty for non-boss full-run deaths.

        The late-Act floor-clear ladder is intentionally positive so the
        policy values reaching Act 1 boss.  Without a matching terminal death
        loss, however, floor 13/14 normal-combat deaths can still end with a
        positive episode return.  Apply a terminal-only, floor-scaled penalty
        to make "got deep and died" clearly worse than surviving to the boss.

        Boss fights have a separate damage-undo terminal penalty; do not stack
        this generic term on top of it.
        """
        if not terminated or truncated:
            return 0.0
        if self._is_boss_encounter(before_obs) or self._is_boss_encounter(after_obs):
            return 0.0
        player = (
            after_obs.get("player")
            if isinstance(after_obs, dict) and isinstance(after_obs.get("player"), dict)
            else {}
        )
        hp = _float((player or {}).get("hp"))
        max_hp = max(_float((player or {}).get("max_hp"), 1.0), 1.0)
        if hp > 0.0:
            return 0.0

        missing_hp_ratio = float(np.clip((max_hp - max(hp, 0.0)) / max_hp, 0.0, 1.0))

        def _floor_from(obs: dict[str, Any] | None) -> float:
            run = obs.get("run") if isinstance(obs, dict) and isinstance(obs.get("run"), dict) else {}
            for key in ("act_floor", "floor", "total_floor"):
                val = run.get(key)
                if val is None:
                    continue
                try:
                    return float(val)
                except (TypeError, ValueError):
                    continue
            return 0.0

        # Act-local floor is ideal.  If only total floor is present, capping at
        # 17 still gives a full late-act penalty in later acts, which is safer
        # than letting Act2/3 deaths look like floor-1 deaths.
        floor_value = max(
            _floor_from(before_obs),
            _floor_from(after_obs),
            float(getattr(self, "_max_floor_reached", 0) or 0),
        )
        floor_norm = float(np.clip(floor_value / 17.0, 0.0, 1.0))
        magnitude = (
            float(FULL_RUN_DEATH_PENALTY_BASE)
            + float(FULL_RUN_DEATH_PENALTY_MISSING_HP_SCALE) * missing_hp_ratio
            + float(FULL_RUN_DEATH_PENALTY_LATE_ACT_SCALE) * floor_norm
        )
        penalty = -magnitude
        self._episode_telemetry["full_run_death_terminal_penalty_events"] += 1.0
        self._episode_telemetry["full_run_death_terminal_penalty_total"] += float(penalty)
        self._episode_telemetry["full_run_death_terminal_floor_norm"] = float(floor_norm)
        self._episode_telemetry["full_run_death_terminal_missing_hp_ratio"] = float(missing_hp_ratio)
        return penalty

    def _floor_clear_reward(
        self,
        before_obs: dict[str, Any] | None,
        after_obs: dict[str, Any] | None,
    ) -> float:
        """One-shot bonus on every floor advancement past FLOOR_CLEAR_MIN_FLOOR.

        Two-tier:
          - Reaching a boss floor (BOSS_ACT_FLOORS) → BOSS_FLOOR_ENTRY_BONUS
          - Reaching any other qualifying floor → FLOOR_CLEAR_BONUS_PER_FLOOR

        Fires exactly once per floor transition (when floor strictly
        increases), not every step. Decays back to zero on same-floor
        combats. Low-floor advancement (floor 0-10) gives nothing —
        those are the easy half of Act 1 where the policy already
        regularly reaches, and rewarding them would be wasteful
        shaping on solved states.
        """
        if not isinstance(before_obs, dict) or not isinstance(after_obs, dict):
            return 0.0
        before_run = before_obs.get("run") if isinstance(before_obs.get("run"), dict) else {}
        after_run = after_obs.get("run") if isinstance(after_obs.get("run"), dict) else {}
        try:
            before_floor = int(before_run.get("floor") or 0)
            after_floor = int(after_run.get("floor") or 0)
        except (TypeError, ValueError):
            return 0.0
        if after_floor <= before_floor:
            return 0.0
        if after_floor < FLOOR_CLEAR_MIN_FLOOR:
            return 0.0
        if after_floor in BOSS_ACT_FLOORS:
            reward = float(BOSS_FLOOR_ENTRY_BONUS)
            self._episode_telemetry["boss_floor_entry_events"] += 1.0
            self._episode_telemetry["floor_clear_reward_total"] += reward
            return reward
        reward = float(FLOOR_CLEAR_BONUS_PER_FLOOR)
        self._episode_telemetry["floor_clear_events"] += 1.0
        self._episode_telemetry["floor_clear_reward_total"] += reward
        return reward

    def _progress_fingerprint(self) -> tuple:
        """Coarse snapshot of 'has the world meaningfully advanced?' signals.

        Two consecutive steps sharing the same fingerprint means the agent
        chose an action that left the visible game state identical — no
        floor change, no combat round tick, no damage dealt, no hp loss,
        no card selection progress. A handful of these can happen
        legitimately (0-cost card draws, null-effect selections). Hundreds
        in a row means the policy is stuck in a no-op loop and should be
        truncated.

        Phase 8.1: extended with (selected_count, card_selection_prompt,
        can_confirm) so multi-step NEOW / card_reward / campfire selection
        flows don't get falsely flagged. These screens keep the basic 5-
        tuple constant for dozens of legit choice steps (no combat, no hp
        change, no floor change) — previously caused 21/21 floor-1 stuck
        cases in the Phase 8 smoke to all land on exactly stuck_steps=400.
        """
        obs = self._last_obs_raw or {}
        run = obs.get("run") if isinstance(obs, dict) else None
        combat = obs.get("combat") if isinstance(obs, dict) else None
        player = obs.get("player") if isinstance(obs, dict) else None
        floor = 0
        if isinstance(run, dict):
            for key in ("floor", "total_floor", "act_floor"):
                val = run.get(key)
                if val is not None:
                    try:
                        floor = int(val)
                        break
                    except (TypeError, ValueError):
                        continue
        combat_round = 0
        enemy_hp_total = 0
        if isinstance(combat, dict):
            try:
                combat_round = int(combat.get("round") or 0)
            except (TypeError, ValueError):
                combat_round = 0
            enemies = combat.get("enemies") if isinstance(combat.get("enemies"), list) else []
            for enemy in enemies:
                if not isinstance(enemy, dict):
                    continue
                try:
                    enemy_hp_total += int(enemy.get("hp") or 0)
                except (TypeError, ValueError):
                    continue
        player_hp = 0
        if isinstance(player, dict):
            try:
                player_hp = int(player.get("hp") or 0)
            except (TypeError, ValueError):
                player_hp = 0
        phase = str(obs.get("phase") or "")
        # Phase 8.1 additions: selection-aware fields.
        selected_count = 0
        selection_prompt = ""
        can_confirm = False
        decision = obs.get("decision") if isinstance(obs, dict) else None
        if isinstance(decision, dict):
            try:
                selected_count = int(decision.get("selected_count") or 0)
            except (TypeError, ValueError):
                selected_count = 0
        card_selection = obs.get("card_selection") if isinstance(obs, dict) else None
        if isinstance(card_selection, dict):
            selection_prompt = str(card_selection.get("prompt") or "")
            can_confirm = bool(card_selection.get("can_confirm"))
        return (
            phase, floor, combat_round, enemy_hp_total, player_hp,
            selected_count, selection_prompt, bool(can_confirm),
        )

    def _check_stuck_watchdog(self, bridge_info: Any) -> tuple[bool, Any]:
        """Increment stuck counter; truncate if fingerprint stable too long.

        Returns ``(truncated, bridge_info)``. When truncation fires, the
        returned bridge_info has ``truncation_reason="phase_stuck_watchdog"``
        plus ``stuck_phase`` / ``stuck_floor`` / ``stuck_steps`` fields so
        the async collector's ``episode_terminal`` event captures the
        reason. The bridge_info passed in is treated as the existing dict
        we should append to (not replaced).
        """
        fingerprint = self._progress_fingerprint()
        if fingerprint == self._stuck_fingerprint:
            self._stuck_steps += 1
        else:
            self._stuck_fingerprint = fingerprint
            self._stuck_steps = 1
        if self._stuck_steps < self.stuck_watchdog_steps:
            return False, bridge_info
        # Stuck — truncate and annotate.
        info_out: dict[str, Any] = dict(bridge_info) if isinstance(bridge_info, dict) else {}
        info_out["truncation_reason"] = "phase_stuck_watchdog"
        info_out["stuck_phase"] = fingerprint[0]
        info_out["stuck_floor"] = fingerprint[1]
        info_out["stuck_combat_round"] = fingerprint[2]
        info_out["stuck_enemy_hp_total"] = fingerprint[3]
        info_out["stuck_player_hp"] = fingerprint[4]
        info_out["stuck_steps"] = int(self._stuck_steps)
        return True, info_out

    def _player_hp_delta_reward(self, before_obs: dict[str, Any] | None, after_obs: dict[str, Any] | None) -> float:
        before_player = before_obs.get("player") if isinstance(before_obs, dict) else {}
        after_player = after_obs.get("player") if isinstance(after_obs, dict) else {}
        before_hp = _float((before_player or {}).get("hp"))
        after_hp = _float((after_player or {}).get("hp"))
        if before_hp <= 0.0 and after_hp <= 0.0:
            return 0.0
        # Symmetric shaping: positive for HP gain (rest site, heal potion,
        # heal event, lifesteal cards), negative for HP loss. Previously
        # we only penalized loss, which made rest-site decisions invisible
        # to PPO (0 reward whether agent rests or skips) and left HP
        # management as a distant-future credit-assignment problem.
        return (after_hp - before_hp) * PLAYER_HP_LOSS_REWARD_SCALE

    @staticmethod
    def _source_preview_metric(source: dict[str, Any] | None, key: str) -> float:
        if not isinstance(source, dict):
            return 0.0
        effect_preview = source.get("effect_preview")
        if isinstance(effect_preview, dict) and effect_preview.get(key) is not None:
            try:
                return float(effect_preview.get(key) or 0.0)
            except (TypeError, ValueError):
                return 0.0
        try:
            return float(source.get(key) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def _is_positive_progress_action(self, action: dict[str, Any]) -> bool:
        kind = str(action.get("kind") or "").strip()
        if kind not in ("play_card", "use_potion"):
            return False

        source = action.get("card") if kind == "play_card" else action.get("potion")
        if not isinstance(source, dict):
            return False

        if kind == "play_card" and str(source.get("type") or "").strip().lower() == "power":
            return True

        for key in ("damage", "block", "draw", "weak", "vulnerable", "heal", "strength", "dexterity", "summon"):
            if self._source_preview_metric(source, key) > 0.0:
                return True
        return False

    def _end_turn_waste_penalty(
        self,
        obs: dict[str, Any] | None,
        legal_actions: list[dict[str, Any]],
        chosen_action: dict[str, Any],
    ) -> float:
        if str(chosen_action.get("action_id") or "") != "end_turn":
            return 0.0

        combat = obs.get("combat") if isinstance(obs, dict) else None
        if not isinstance(combat, dict):
            return 0.0
        energy = float(combat.get("energy") or 0.0)
        if energy <= 0.0:
            return 0.0

        strict_context = strict_end_turn_waste_context(
            obs,
            legal_actions,
            chosen_action,
        )
        positive_actions = int(strict_context.get("urgent_positive_action_count", 0.0) or 0.0)
        urgent_indices = {int(idx) for idx in strict_context.get("urgent_positive_indices", [])}
        has_zero_cost_positive = False
        for idx, action in enumerate(legal_actions):
            if idx not in urgent_indices or not isinstance(action, dict):
                continue
            card = action.get("card")
            if not isinstance(card, dict):
                continue
            try:
                cost_raw = str(card.get("cost", card.get("resolved_energy_cost", 0.0))).strip().upper()
                cost = 0.0 if cost_raw == "X" else float(cost_raw or 0.0)
                if cost <= 0.0:
                    has_zero_cost_positive = True
            except (TypeError, ValueError):
                pass

        if positive_actions <= 0:
            return 0.0

        penalty = END_TURN_WASTE_BASE_PENALTY
        penalty += END_TURN_WASTE_ENERGY_PENALTY * min(energy, 3.0)
        if has_zero_cost_positive:
            penalty += END_TURN_WASTE_ZERO_COST_BONUS_PENALTY
        penalty += END_TURN_WASTE_EXTRA_ACTION_PENALTY * min(max(positive_actions - 1, 0), 2)
        return float(penalty)

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
                result = self.bridge.reset(
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
                if self._consecutive_end_turn_leaks >= 8:
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
                if self._consecutive_end_turn_leaks >= 16:
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

    @staticmethod
    def _is_end_turn_action(action: dict[str, Any] | None) -> bool:
        if not isinstance(action, dict):
            return False
        action_id = str(action.get("action_id") or "").strip().lower()
        kind = str(action.get("kind") or "").strip().lower()
        family = ""
        semantic = action.get("semantic") if isinstance(action.get("semantic"), dict) else {}
        if isinstance(semantic, dict):
            family = str(semantic.get("family") or semantic.get("action_kind") or "").strip().lower()
        return action_id == "end_turn" or kind == "end_turn" or family == "end_turn"

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

    def _safe_get_state(self) -> dict[str, Any] | None:
        try:
            state = self.bridge.get_state()
        except Exception:
            return None
        return state if isinstance(state, dict) else None

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
                result = self.bridge.reset(
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
    def _state_available_actions(state: dict[str, Any] | None) -> list[dict[str, Any]]:
        if not isinstance(state, dict):
            return []
        actions = state.get("available_actions")
        if not isinstance(actions, list):
            actions = state.get("legal_actions")
        if not isinstance(actions, list):
            return []
        return [action for action in actions if isinstance(action, dict)]

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

    def _planner_context(self) -> dict[str, Any]:
        context = self._run_memory.build_context(self._last_obs_raw, self._legal_actions)
        context["combat_memory"] = self._combat_memory.snapshot()
        return context

    @staticmethod
    def _python_timing(
        *,
        bridge_roundtrip: float,
        run_memory_update: float,
        obs_encode: float,
        aux_targets: float,
        info_build: float,
        total: float,
    ) -> dict[str, float]:
        return {
            "bridge_roundtrip": float(bridge_roundtrip),
            "run_memory_update": float(run_memory_update),
            "obs_encode": float(obs_encode),
            "aux_targets": float(aux_targets),
            "info_build": float(info_build),
            "total": float(total),
        }

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
