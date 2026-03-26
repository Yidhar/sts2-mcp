"""Phase 2 Gymnasium environment wrapper for Slay the Spire 2 RL training.

Uses DictObservationEncoder (observation_v2) for Dict observation spaces
with per-card and per-enemy feature arrays, suitable for attention-based
policy networks.

All auto-resolution, action filtering, reward shaping, and recovery logic
is inherited from the Phase 1 env.  Only the observation encoding differs.
"""

import time

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .bridge_client import BridgeClient, BridgeError
from .observation import ActionEncoder
from .observation_v2 import DictObservationEncoder

_MAX_STEP_RETRIES = 3
_MAX_RESET_RETRIES = 12
_RESET_RETRY_DELAY_S = 5.0
_AUTO_RESOLVE_TIMEOUT_MS = 5000

# Phases where the agent makes real decisions
_DECISION_PHASES = frozenset({
    "combat", "map", "event", "event_crystal_sphere",
    "terminal", "unknown",
})

# Phases auto-resolved without agent input
_AUTO_RESOLVE_PHASES = frozenset({
    "reward", "card_reward", "rest_site", "deck_upgrade",
    "card_selection", "shop", "treasure", "actions",
    "settling",
    "startup_main_menu", "startup_run_mode", "startup_character_select",
})

# Max auto-resolve steps before giving up
_MAX_AUTO_RESOLVE_STEPS = 30


class SlayTheSpire2EnvV2(gym.Env):
    """Gymnasium Env with Dict observations for attention-based policies.

    Auto-resolves non-critical screens (rewards, rest sites, shops, etc.)
    so the RL agent only sees combat, map, and event decisions.

    Observation space is a Dict with keys:
        scalars, hand, hand_mask, enemies, enemy_mask, player_powers
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        session_file=None,
        character=None,
        defensive_buffs=False,
        reset_timeout_ms=45000,
        step_timeout_ms=20000,
        render_mode=None,
        auto_resolve=True,
    ):
        super().__init__()

        self.bridge = BridgeClient(session_path=session_file)
        self.obs_encoder = DictObservationEncoder()
        self.action_encoder = ActionEncoder()
        self.character = character
        self.defensive_buffs = defensive_buffs
        self.reset_timeout_ms = reset_timeout_ms
        self.step_timeout_ms = step_timeout_ms
        self.render_mode = render_mode
        self.auto_resolve = auto_resolve

        # Spaces
        self.observation_space = self.obs_encoder.obs_space
        self.action_space = spaces.Discrete(self.action_encoder.MAX_ACTIONS)

        # Episode state
        self._episode_id = None
        self._legal_actions = []
        self._last_obs_raw = None
        self._last_info = {}

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        last_err = None
        for attempt in range(_MAX_RESET_RETRIES):
            try:
                result = self.bridge.reset(
                    character=self.character,
                    defensive_buffs=self.defensive_buffs,
                    timeout_ms=self.reset_timeout_ms,
                )
                break
            except BridgeError as e:
                last_err = e
                body = e.response_body if hasattr(e, "response_body") else None
                if not isinstance(body, dict):
                    raise

                error_code = body.get("error", "")
                if error_code not in (
                    "env_reset_requires_main_menu",
                    "env_reset_no_reset_path",
                    "env_reset_transition_limit",
                ):
                    raise

                details = body.get("details", {})
                run_info = details.get("run", {})
                if run_info.get("active") and not run_info.get("game_over"):
                    self._try_abandon_run()

                time.sleep(_RESET_RETRY_DELAY_S)
                continue
        else:
            raise last_err  # type: ignore[misc]

        self._episode_id = result["episode_id"]
        self._last_obs_raw = result.get("obs", {})
        self._legal_actions = self._filter_legal_actions(
            result.get("legal_actions", []), self._last_obs_raw
        )

        # Auto-resolve any non-decision phase after reset
        if self.auto_resolve:
            self._auto_resolve_until_decision()

        obs = self.obs_encoder.encode(self._last_obs_raw)
        info = self._build_info()
        return obs, info

    def step(self, action):
        # Guard: no legal actions => terminal
        if not self._legal_actions:
            return self._terminal_step()

        # Clamp out-of-range
        if action >= len(self._legal_actions):
            action = 0

        # Snapshot pre-step state for reward shaping
        pre_obs = self._last_obs_raw or {}
        chosen_action = self._legal_actions[action]

        # Execute the agent's chosen action
        result = self._step_with_recovery(action)

        if result is None:
            return self._terminal_step(truncated=True)

        self._last_obs_raw = result.get("obs", {})
        self._legal_actions = self._filter_legal_actions(
            result.get("legal_actions", []), self._last_obs_raw
        )

        reward = float(result.get("reward", 0.0))
        terminated = bool(result.get("done", False))
        truncated = bool(result.get("truncated", False))

        # Reward shaping
        reward += self._shape_reward(chosen_action, pre_obs)

        # Auto-resolve non-decision phases
        if self.auto_resolve and not terminated and not truncated:
            extra_reward, terminated, truncated = self._auto_resolve_until_decision()
            reward += extra_reward

        obs = self.obs_encoder.encode(self._last_obs_raw)
        info = self._build_info(
            step_index=result.get("step_index", 0),
            reward_breakdown=result.get("info", {}).get("reward_breakdown", {}),
        )

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Auto-resolution of non-critical screens
    # ------------------------------------------------------------------

    def _auto_resolve_until_decision(self):
        """Keep stepping through non-decision phases.

        Returns (accumulated_reward, terminated, truncated).
        """
        total_reward = 0.0

        for _ in range(_MAX_AUTO_RESOLVE_STEPS):
            phase = (self._last_obs_raw or {}).get("phase", "unknown")
            if phase in _DECISION_PHASES or not self._legal_actions:
                break

            action_idx = self._pick_auto_resolve_action(phase)
            result = self._step_with_recovery(
                action_idx, timeout_ms=_AUTO_RESOLVE_TIMEOUT_MS
            )
            if result is None:
                return total_reward, False, True

            self._last_obs_raw = result.get("obs", {})
            self._legal_actions = self._filter_legal_actions(
                result.get("legal_actions", []), self._last_obs_raw
            )
            total_reward += float(result.get("reward", 0.0))

            if result.get("done", False):
                return total_reward, True, False
            if result.get("truncated", False):
                return total_reward, False, True

        return total_reward, False, False

    def _pick_auto_resolve_action(self, phase):
        """Choose which action to take for auto-resolved phases."""
        actions = self._legal_actions
        if not actions:
            return 0

        def find_action(predicate):
            for i, a in enumerate(actions):
                if predicate(a):
                    return i
            return None

        if phase == "reward":
            idx = find_action(lambda a: a.get("kind") == "reward")
            if idx is not None:
                return idx
            idx = find_action(lambda a: a.get("kind") == "proceed")
            if idx is not None:
                return idx
            return 0

        if phase == "card_reward":
            idx = find_action(
                lambda a: "skip" in (a.get("action_id") or "")
                or a.get("kind") == "proceed"
            )
            if idx is not None:
                return idx
            return 0

        if phase == "rest_site":
            player = (self._last_obs_raw or {}).get("player", {})
            hp = player.get("hp") or 0
            max_hp = player.get("max_hp") or 1
            if hp / max_hp < 0.7:
                idx = find_action(
                    lambda a: "rest" in (a.get("action_id") or "").lower()
                    or "rest" in (a.get("kind") or "").lower()
                )
                if idx is not None:
                    return idx
            return 0

        if phase == "deck_upgrade":
            idx = find_action(
                lambda a: "confirm" in (a.get("action_id") or "")
            )
            if idx is not None:
                return idx
            idx = find_action(
                lambda a: "select" in (a.get("action_id") or "")
            )
            if idx is not None:
                return idx
            return 0

        if phase == "card_selection":
            idx = find_action(
                lambda a: "confirm" in (a.get("action_id") or "")
                or "cancel" in (a.get("action_id") or "")
                or a.get("kind") == "proceed"
            )
            if idx is not None:
                return idx
            return 0

        if phase == "shop":
            idx = find_action(
                lambda a: "leave" in (a.get("action_id") or "")
                or "back" in (a.get("action_id") or "")
                or a.get("kind") == "proceed"
            )
            if idx is not None:
                return idx
            return 0

        if phase == "treasure":
            return 0

        return 0

    # ------------------------------------------------------------------
    # Run abandonment
    # ------------------------------------------------------------------

    def _try_abandon_run(self):
        """Navigate from an active run back to main menu."""
        deadline = time.time() + 60.0
        for _ in range(200):
            if time.time() > deadline:
                return
            try:
                state = self.bridge.get_state()
            except BridgeError:
                time.sleep(2.0)
                continue

            screen = state.get("screen", "")
            if screen in ("MAIN_MENU", "CHARACTER_SELECT", "RUN_MODE_SELECT"):
                return

            run = state.get("run", {})
            room_type = run.get("room_type")
            is_map_overlay = (
                screen == "MAP"
                and room_type
                and room_type not in ("", "None")
            )

            actions = [
                a for a in state.get("available_actions", [])
                if a.get("kind") != "automation"
            ]

            if is_map_overlay:
                time.sleep(5.0)
                continue

            if not actions:
                time.sleep(1.0)
                continue

            action_id = actions[0].get("action_id", "")
            if not action_id:
                time.sleep(1.0)
                continue

            try:
                self.bridge.perform_action(action_id, wait_after_ms=500)
            except BridgeError:
                time.sleep(1.0)

    # ------------------------------------------------------------------
    # Reward shaping
    # ------------------------------------------------------------------

    @staticmethod
    def _shape_reward(chosen_action, pre_obs):
        """Penalize ending turn with unused energy/playable cards."""
        action_id = chosen_action.get("action_id", "")
        if action_id != "end_turn":
            return 0.0

        combat = pre_obs.get("combat")
        if not combat:
            return 0.0

        energy = combat.get("energy") or 0
        max_energy = combat.get("max_energy") or 0
        hand = combat.get("hand") or []

        if energy <= 0 or max_energy <= 0 or not hand:
            return 0.0

        playable = 0
        for card in hand:
            if not isinstance(card, dict):
                continue
            cost = card.get("cost")
            if cost is not None and cost <= energy:
                playable += 1
            elif card.get("x_cost"):
                playable += 1
        if playable == 0:
            return 0.0

        waste_ratio = energy / max_energy
        return -0.15 * waste_ratio

    # ------------------------------------------------------------------
    # Step execution with error recovery
    # ------------------------------------------------------------------

    def _step_with_recovery(self, action, timeout_ms=None):
        timeout = timeout_ms or self.step_timeout_ms
        for attempt in range(_MAX_STEP_RETRIES):
            if not self._legal_actions:
                return None

            if action >= len(self._legal_actions):
                action = 0

            legal_action = self._legal_actions[action]
            action_id = legal_action.get("action_id")

            try:
                return self.bridge.step(
                    episode_id=self._episode_id,
                    action_id=action_id,
                    timeout_ms=timeout,
                )
            except BridgeError as e:
                body = e.response_body if hasattr(e, "response_body") else None
                if not isinstance(body, dict):
                    raise

                error_code = body.get("error", "")
                details = body.get("details", {})

                if error_code in (
                    "action_not_available",
                    "action_index_out_of_range",
                    "action_selector_mismatch",
                ):
                    recovered = details.get("legal_actions", [])
                    if recovered:
                        self._legal_actions = self._filter_legal_actions(
                            recovered, self._last_obs_raw
                        )
                        action = 0
                        continue
                    else:
                        return None

                if error_code in ("episode_already_done", "unknown_episode_id",
                                  "no_active_episode", "episode_id_mismatch"):
                    return {"done": True, "truncated": False, "reward": 0.0,
                            "obs": self._last_obs_raw or {}, "legal_actions": [],
                            "info": {}}

                if error_code == "internal_error":
                    time.sleep(1.0)
                    continue

                raise

        return None

    def _terminal_step(self, truncated=False):
        obs = self.obs_encoder.encode(self._last_obs_raw or {})
        info = self._build_info()
        return obs, 0.0, not truncated, truncated, info

    def _build_info(self, step_index=0, reward_breakdown=None):
        info = {
            "episode_id": self._episode_id,
            "step_index": step_index,
            "action_mask": self.action_masks(),
            "legal_actions": self._legal_actions,
            "raw_obs": self._last_obs_raw,
            "phase": (self._last_obs_raw or {}).get("phase", "unknown"),
            "reward_breakdown": reward_breakdown or {},
        }
        self._last_info = info
        return info

    # ------------------------------------------------------------------
    # Action filtering & masking
    # ------------------------------------------------------------------

    def _filter_legal_actions(self, legal_actions, obs):
        """Filter out map travel actions when inside a room."""
        phase = (obs or {}).get("phase", "unknown")
        if phase == "map":
            return legal_actions

        return [
            a for a in legal_actions
            if a.get("kind") != "map" and not (a.get("action_id") or "").startswith("map:")
        ]

    def action_masks(self):
        n_legal = len(self._legal_actions)
        mask = np.zeros(self.action_encoder.MAX_ACTIONS, dtype=bool)
        mask[:n_legal] = True
        return mask

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self):
        if self._last_obs_raw is None:
            return
        obs = self._last_obs_raw
        phase = obs.get("phase", "?")
        player = obs.get("player", {})
        hp = player.get("hp", "?")
        max_hp = player.get("max_hp", "?")
        run = obs.get("run", {})
        floor_num = run.get("floor", "?")
        print(
            f"[STS2-v2] Phase: {phase} | HP: {hp}/{max_hp} "
            f"| Floor: {floor_num} | Actions: {len(self._legal_actions)}"
        )

    def close(self):
        pass
