"""Thin Gymnasium wrapper for the STS2 bridge RL env.

All episode lifecycle, state stability, and action validation is handled
by the bridge's env/reset and env/step endpoints.  This wrapper only:
  - Calls bridge reset/step
  - Encodes observations into the Dict format
  - Returns action masks for MaskablePPO
  - Provides render() for human inspection

No auto-resolve.  No UI hacks.  No retry loops.  No heuristic fallbacks.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .bridge_client import BridgeClient, BridgeError
from .observation_v2 import DictObservationEncoder

MAX_ACTIONS = 50


class SlayTheSpire2EnvV2(gym.Env):
    """Gymnasium Env backed by the STS2 bridge env/reset and env/step endpoints.

    The bridge guarantees:
      - env/reset navigates from any game state to a fresh actionable episode.
      - env/step executes one action atomically, waits for stability, and
        returns the next actionable-or-terminal state.
      - Legal actions are consistent with the returned observation.

    This wrapper does NOT second-guess the bridge.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        session_file=None,
        character=None,
        defensive_buffs=False,
        reset_timeout_ms=60000,
        step_timeout_ms=20000,
        render_mode=None,
        obs_encoder=None,
    ):
        super().__init__()

        self.bridge = BridgeClient(session_path=session_file)
        self.obs_encoder = obs_encoder or DictObservationEncoder(use_text=False)
        self.character = character
        self.defensive_buffs = defensive_buffs
        self.reset_timeout_ms = reset_timeout_ms
        self.step_timeout_ms = step_timeout_ms
        self.render_mode = render_mode

        self.observation_space = self.obs_encoder.obs_space
        self.action_space = spaces.Discrete(MAX_ACTIONS)

        self._episode_id = None
        self._legal_actions = []
        self._last_obs_raw = None

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        result = self.bridge.reset(
            character=self.character,
            defensive_buffs=self.defensive_buffs,
            timeout_ms=self.reset_timeout_ms,
        )

        self._episode_id = result["episode_id"]
        self._legal_actions = result.get("legal_actions", [])
        self._last_obs_raw = result.get("obs", {})

        obs = self.obs_encoder.encode(self._last_obs_raw, self._legal_actions)
        info = self._build_info(result.get("info", {}))
        return obs, info

    def step(self, action):
        if not self._legal_actions:
            return self._make_terminal()

        if action >= len(self._legal_actions):
            action = 0

        legal_action = self._legal_actions[action]

        result = self.bridge.step(
            episode_id=self._episode_id,
            action_id=legal_action.get("action_id"),
            timeout_ms=self.step_timeout_ms,
        )

        self._legal_actions = result.get("legal_actions", [])
        self._last_obs_raw = result.get("obs", {})

        obs = self.obs_encoder.encode(self._last_obs_raw, self._legal_actions)
        reward = float(result.get("reward", 0.0))
        terminated = bool(result.get("done", False))
        truncated = bool(result.get("truncated", False))
        info = self._build_info(result.get("info", {}))

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def action_masks(self):
        mask = np.zeros(MAX_ACTIONS, dtype=bool)
        n = min(len(self._legal_actions), MAX_ACTIONS)
        mask[:n] = True
        return mask

    def _make_terminal(self):
        obs = self.obs_encoder.encode(self._last_obs_raw or {}, [])
        info = self._build_info({})
        return obs, 0.0, True, False, info

    def render(self):
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

    def close(self):
        pass

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_info(self, bridge_info):
        return {
            "episode_id": self._episode_id,
            "action_mask": self.action_masks(),
            "legal_actions": self._legal_actions,
            "raw_obs": self._last_obs_raw,
            "phase": (self._last_obs_raw or {}).get("phase", "unknown"),
            "bridge_info": bridge_info,
        }

