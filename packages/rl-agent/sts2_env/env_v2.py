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

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .bridge_client import BridgeClient
from .observation_v2 import DictObservationEncoder, MAX_ACTIONS

INVALID_ACTION_REWARD = -1.0
INVALID_ACTION_REASON = "invalid_action_index"


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
        obs_encoder: DictObservationEncoder | None = None,
        include_debug_info: bool = False,
    ) -> None:
        super().__init__()

        self.bridge = BridgeClient(session_path=session_file)
        self.obs_encoder = obs_encoder or DictObservationEncoder(use_text=False)
        self.character = character
        self.defensive_buffs = defensive_buffs
        self.reset_timeout_ms = reset_timeout_ms
        self.step_timeout_ms = step_timeout_ms
        self.render_mode = render_mode
        self.include_debug_info = bool(include_debug_info)

        self.observation_space = self.obs_encoder.obs_space
        self.action_space = spaces.Discrete(MAX_ACTIONS)

        self._episode_id: str | None = None
        self._legal_actions: list[dict[str, Any]] = []
        self._last_obs_raw: dict[str, Any] | None = None
        self._last_action_overflow: int = 0

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)

        result = self.bridge.reset(
            character=self.character,
            defensive_buffs=self.defensive_buffs,
            timeout_ms=self.reset_timeout_ms,
        )

        self._episode_id = result["episode_id"]
        self._update_live_state(result)

        obs = self.obs_encoder.encode(self._last_obs_raw, self._legal_actions)
        info = self._build_info(result.get("info", {}))
        return obs, info

    def step(self, action: int):
        if not self._legal_actions:
            return self._make_terminal()

        normalized_action = self._normalize_action(action)
        if normalized_action is None or normalized_action >= len(self._legal_actions):
            return self._make_invalid_action_response(action)

        legal_action = self._legal_actions[normalized_action]

        result = self.bridge.step(
            episode_id=self._episode_id,
            action_id=legal_action.get("action_id"),
            timeout_ms=self.step_timeout_ms,
        )

        self._update_live_state(result)

        obs = self.obs_encoder.encode(self._last_obs_raw, self._legal_actions)
        reward = float(result.get("reward", 0.0))
        terminated = bool(result.get("done", False))
        truncated = bool(result.get("truncated", False))
        info = self._build_info(result.get("info", {}))

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        mask = np.zeros(MAX_ACTIONS, dtype=bool)
        n = min(len(self._legal_actions), MAX_ACTIONS)
        mask[:n] = True
        return mask

    def _make_terminal(self):
        obs = self.obs_encoder.encode(self._last_obs_raw or {}, [])
        info = self._build_info({})
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
        legal_actions = result.get("legal_actions", [])
        self._legal_actions = legal_actions if isinstance(legal_actions, list) else []
        obs = result.get("obs", {})
        self._last_obs_raw = obs if isinstance(obs, dict) else {}
        self._last_action_overflow = max(len(self._legal_actions) - MAX_ACTIONS, 0)

    def _decorate_bridge_info(self, bridge_info: Any) -> dict[str, Any]:
        info = dict(bridge_info) if isinstance(bridge_info, dict) else {}
        diagnostics = info.get("action_diagnostics")
        diagnostics = dict(diagnostics) if isinstance(diagnostics, dict) else {}
        diagnostics["legal_action_overflow"] = float(self._last_action_overflow)
        info["action_diagnostics"] = diagnostics
        return info

    def _build_info(self, bridge_info: Any, *, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        info: dict[str, Any] = {
            "episode_id": self._episode_id,
            "action_mask": self.action_masks(),
            "legal_action_count": len(self._legal_actions),
            "action_overflow": self._last_action_overflow,
            "phase": (self._last_obs_raw or {}).get("phase", "unknown"),
            "episode_mode": "full_run",
            "bridge_info": self._decorate_bridge_info(bridge_info),
        }
        if extra:
            info.update(extra)
        if self.include_debug_info:
            info["legal_actions"] = self._legal_actions
            info["raw_obs"] = self._last_obs_raw
        return info

    def _make_invalid_action_response(self, attempted_action: Any):
        obs = self.obs_encoder.encode(self._last_obs_raw or {}, self._legal_actions)
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
            },
        )
        return obs, INVALID_ACTION_REWARD, False, True, info
