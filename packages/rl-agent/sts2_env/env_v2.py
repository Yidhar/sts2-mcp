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
from .run_memory import RunMemoryTracker

from .reward_constants import (
    BOSS_ACT_FLOORS,
    BOSS_DAMAGE_MULTIPLIER,
    BOSS_FLOOR_ENTRY_BONUS,
    ENEMY_HP_DELTA_REWARD_MAX_ABS,
    ENEMY_HP_DELTA_REWARD_SCALE,
    ENEMY_HP_SENTINEL_THRESHOLD,
    FLOOR_CLEAR_BONUS_PER_FLOOR,
    FLOOR_CLEAR_MIN_FLOOR,
    FULL_RUN_WASTE_BASE as END_TURN_WASTE_BASE_PENALTY,
    FULL_RUN_WASTE_ENERGY as END_TURN_WASTE_ENERGY_PENALTY,
    FULL_RUN_WASTE_EXTRA_ACTION as END_TURN_WASTE_EXTRA_ACTION_PENALTY,
    FULL_RUN_WASTE_ZERO_COST as END_TURN_WASTE_ZERO_COST_BONUS_PENALTY,
    INVALID_ACTION_REWARD,
    PLAYER_HP_LOSS_REWARD_SCALE,
    POTION_USE_BOSS_BONUS,
    POTION_USE_ELITE_BONUS,
    POTION_USE_MONSTER_BONUS,
    POTION_USE_MONSTER_PENALTY,
    REST_SITE_SKIP_HEAL_HP_THRESHOLD,
    REST_SITE_SKIP_HEAL_PENALTY,
)

INVALID_ACTION_REASON = "invalid_action_index"
BLOCKED_ACTION_KINDS = {"discard_potion"}
RECOVERY_POLL_INTERVAL_S = 0.10
RECOVERY_MAX_WAIT_MS = 15_000
TRANSITION_RECOVERY_MAX_WAIT_MS = 60_000
RESET_READY_POLL_INTERVAL_S = 0.50
RESET_READY_MAX_WAIT_MS = 90_000
STEP_RECOVERY_TRUNCATION_REASON = "bridge_episode_lost"
STARTUP_ACTION_PREFIXES = ("main_menu:", "run_mode:", "character_select:")


def _float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


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
        self._max_floor_reached: int = 0
        self._stuck_fingerprint: tuple | None = None
        self._stuck_steps: int = 0
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
            # Rest site -------------------------------------------------
            "rest_site_encounters": 0.0,
            "rest_heal_chosen": 0.0,
            "rest_skip_heal_chosen": 0.0,
            "rest_skip_heal_at_low_hp": 0.0,
            "rest_penalty_total": 0.0,
            # Boss combat ------------------------------------------------
            "boss_damage_dealt_raw": 0.0,
            "boss_damage_bonus_total": 0.0,
            "boss_encounter_steps": 0.0,
            # Floor clear ladder ----------------------------------------
            "floor_clear_reward_total": 0.0,
            "floor_clear_events": 0.0,
            "boss_floor_entry_events": 0.0,
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
        run_memory_started = time.perf_counter()
        self._run_memory.reset(
            self._last_obs_raw,
            self._legal_actions,
            episode_mode="full_run",
            potion_mechanics_available=True,
        )
        self._combat_memory.reset(self._last_obs_raw)
        run_memory_elapsed_ms = (time.perf_counter() - run_memory_started) * 1000.0

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
        terminated = bool(result.get("done", False))
        truncated = bool(result.get("truncated", False))
        bridge_info = result.get("info", {})
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
        mask = np.zeros(MAX_ACTIONS, dtype=bool)
        for i, action in enumerate(self._legal_actions[:MAX_ACTIONS]):
            if isinstance(action, dict):
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

    def _update_live_state(self, result: dict[str, Any]) -> None:
        phase = self._extract_phase(result)
        legal_actions = result.get("legal_actions", [])
        if isinstance(legal_actions, list):
            self._legal_actions = self._filter_legal_actions(legal_actions, phase=phase)
        else:
            self._legal_actions = []
        obs = result.get("obs", {})
        self._last_obs_raw = obs if isinstance(obs, dict) else {}
        self._last_action_overflow = max(len(self._legal_actions) - MAX_ACTIONS, 0)
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

    def _boss_damage_bonus_reward(
        self,
        before_obs: dict[str, Any] | None,
        after_obs: dict[str, Any] | None,
    ) -> float:
        """Additive bonus on damage dealt during boss encounters.

        The base ``_enemy_hp_delta_reward`` already handles damage
        dealing at 0.01/hp. This method adds ``(MULTIPLIER - 1) ×`` the
        same raw delta when either side of the transition is flagged as
        boss, so the NET effective multiplier on boss damage is
        MULTIPLIER. Sign-asymmetric: only POSITIVE damage (enemy losing
        HP) gets amplified — taking damage from boss still penalizes
        at base scale, otherwise bosses would become MORE aversive in
        value than normal monsters, the opposite of what we want.
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
        raw_damage = max(before_total - after_total, 0.0)
        if raw_damage <= 0.0:
            return 0.0
        self._episode_telemetry["boss_damage_dealt_raw"] += float(raw_damage)
        extra_multiplier = max(BOSS_DAMAGE_MULTIPLIER - 1.0, 0.0)
        bonus = raw_damage * ENEMY_HP_DELTA_REWARD_SCALE * extra_multiplier
        # Cap using the same safety bound as the base path, scaled by
        # the extra multiplier so total boss damage reward can be up
        # to MULTIPLIER × ENEMY_HP_DELTA_REWARD_MAX_ABS.
        bonus = min(bonus, ENEMY_HP_DELTA_REWARD_MAX_ABS * extra_multiplier)
        self._episode_telemetry["boss_damage_bonus_total"] += float(bonus)
        return bonus

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
        kind = str(action.get("kind") or "").lower()
        if kind not in {"rest", "rest_site", "choose_rest_option"}:
            return 0.0
        option = action.get("option") if isinstance(action.get("option"), dict) else None
        option_type = ""
        option_title = ""
        if isinstance(option, dict):
            option_type = str(option.get("type") or option.get("option_type") or "").lower()
            option_title = str(option.get("title") or option.get("label") or "").lower()
        # Also check the action's own label / action_id in case the
        # option dict isn't populated (sim compat path).
        action_id = str(action.get("action_id") or "").lower()
        label = str(action.get("label") or "").lower()
        heal_markers = ("rest", "heal")
        heal_picked = (
            option_type in {"rest", "heal"}
            or any(marker in option_title for marker in heal_markers)
            or any(marker in action_id for marker in heal_markers)
            or any(marker in label for marker in heal_markers)
        )
        # Telemetry: always count rest-site encounters + the HEAL/non-HEAL
        # split, regardless of HP threshold. Helps diagnose "is the
        # policy even reaching campfires" vs "is it choosing correctly".
        self._episode_telemetry["rest_site_encounters"] += 1.0
        if heal_picked:
            self._episode_telemetry["rest_heal_chosen"] += 1.0
            return 0.0
        self._episode_telemetry["rest_skip_heal_chosen"] += 1.0

        if not isinstance(before_obs, dict):
            return 0.0
        player = before_obs.get("player") if isinstance(before_obs.get("player"), dict) else None
        if not isinstance(player, dict):
            return 0.0
        hp = _float(player.get("hp"))
        max_hp = max(_float(player.get("max_hp"), 1.0), 1.0)
        if hp <= 0.0:
            return 0.0
        hp_ratio = hp / max_hp
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

        positive_actions = 0
        has_zero_cost_positive = False
        for action in legal_actions:
            if not isinstance(action, dict):
                continue
            if str(action.get("action_id") or "") == "end_turn":
                continue
            if not self._is_positive_progress_action(action):
                continue
            positive_actions += 1
            card = action.get("card")
            if isinstance(card, dict):
                try:
                    if float(card.get("cost") or 0.0) <= 0.0:
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

    def _filter_legal_actions(self, legal_actions: list[Any], *, phase: str) -> list[dict[str, Any]]:
        filtered = [
            action
            for action in legal_actions
            if isinstance(action, dict) and str(action.get("kind") or "").strip() not in BLOCKED_ACTION_KINDS
        ]
        if phase != "actions":
            return filtered
        return self._split_actions_phase_actions(filtered)

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

    def _reset_with_ready_gate(self, *, timeout_ms: int) -> dict[str, Any]:
        deadline = time.monotonic() + (max(timeout_ms, RESET_READY_MAX_WAIT_MS) / 1000.0)
        last_exc: Exception | None = None
        force_fresh_next = False

        while time.monotonic() < deadline:
            try:
                result = self.bridge.reset(
                    character=self.character,
                    force_fresh=force_fresh_next,
                    defensive_buffs=self.defensive_buffs,
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
            filtered_actions = self._filter_legal_actions(raw_actions, phase=phase)
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

            if blocked_only:
                force_fresh_next = True
                last_exc = RuntimeError(
                    f"reset returned only blocked legal actions at phase={phase}; forcing fresh reset retry"
                )
            else:
                last_exc = RuntimeError(f"reset returned no usable legal actions at phase={phase}")
            time.sleep(RESET_READY_POLL_INTERVAL_S)

        if last_exc is not None:
            raise last_exc
        raise RuntimeError("reset ready gate timed out without a usable episode")

    def _recover_filtered_action_window(self, *, timeout_ms: int) -> bool:
        if self._legal_actions:
            return True

        if timeout_ms <= 0:
            return False

        deadline = time.monotonic() + (timeout_ms / 1000.0)
        while time.monotonic() < deadline:
            state = self._safe_get_state()
            remaining_ms = max(int((deadline - time.monotonic()) * 1000.0), 0)
            attempt_timeout_ms = max(1_000, min(remaining_ms, self._transition_recovery_timeout_ms()))

            if self._state_allows_soft_rebind(state):
                refreshed = self._safe_reset_into_current_run(attempt_timeout_ms)
                if refreshed is not None:
                    self._episode_id = refreshed.get("episode_id", self._episode_id)
                    self._update_live_state(refreshed)
                    if self._legal_actions:
                        return True

            if not self._state_has_unblocked_actions(state):
                time.sleep(RECOVERY_POLL_INTERVAL_S)
                continue

            time.sleep(RECOVERY_POLL_INTERVAL_S)

        return False

    def _safe_get_state(self) -> dict[str, Any] | None:
        try:
            state = self.bridge.get_state()
        except Exception:
            return None
        return state if isinstance(state, dict) else None

    def _transition_recovery_timeout_ms(self) -> int:
        return max(
            RECOVERY_MAX_WAIT_MS,
            min(self.reset_timeout_ms, TRANSITION_RECOVERY_MAX_WAIT_MS),
        )

    def _soft_rebind_into_current_run(self, timeout_ms: int) -> dict[str, Any] | None:
        if timeout_ms <= 0:
            return None

        deadline = time.monotonic() + (timeout_ms / 1000.0)
        rebind_timeout_ms = max(1_000, min(timeout_ms, self.reset_timeout_ms))

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
            filtered_actions = self._filter_legal_actions(result.get("legal_actions", []), phase=phase)
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
        if not isinstance(state, dict):
            return False

        available_actions = state.get("available_actions")
        if not isinstance(available_actions, list):
            return False

        for action in available_actions:
            if not isinstance(action, dict):
                continue
            kind = str(action.get("kind") or "").strip()
            if kind in BLOCKED_ACTION_KINDS:
                continue
            return True

        return False

    def _decorate_bridge_info(self, bridge_info: Any) -> dict[str, Any]:
        info = dict(bridge_info) if isinstance(bridge_info, dict) else {}
        diagnostics = info.get("action_diagnostics")
        diagnostics = dict(diagnostics) if isinstance(diagnostics, dict) else {}
        diagnostics["legal_action_overflow"] = float(self._last_action_overflow)
        info["action_diagnostics"] = diagnostics
        return info

    def _transition_state(self) -> dict[str, Any]:
        obs = self._last_obs_raw if isinstance(self._last_obs_raw, dict) else {}
        player = obs.get("player") if isinstance(obs.get("player"), dict) else {}
        run = obs.get("run") if isinstance(obs.get("run"), dict) else {}
        combat = obs.get("combat") if isinstance(obs.get("combat"), dict) else {}

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
                "floor": _float(run.get("floor")),
                "act_id": _float(run.get("act_id")),
                "room_type": run.get("room_type"),
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
        }
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
