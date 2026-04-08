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

from .bridge_client import BridgeClient, BridgeError
from .observation_v2 import DictObservationEncoder, MAX_ACTIONS

INVALID_ACTION_REWARD = -1.0
INVALID_ACTION_REASON = "invalid_action_index"
BLOCKED_ACTION_KINDS = {"discard_potion"}
RECOVERY_POLL_INTERVAL_S = 0.10
RECOVERY_MAX_WAIT_MS = 15_000
RESET_READY_POLL_INTERVAL_S = 0.50
RESET_READY_MAX_WAIT_MS = 90_000
STEP_RECOVERY_TRUNCATION_REASON = "bridge_episode_lost"
STARTUP_ACTION_PREFIXES = ("main_menu:", "run_mode:", "character_select:")


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

        result = self._reset_with_ready_gate(timeout_ms=self.reset_timeout_ms)

        self._episode_id = result["episode_id"]
        self._update_live_state(result)
        self._recover_filtered_action_window(timeout_ms=min(self.reset_timeout_ms, RECOVERY_MAX_WAIT_MS))

        obs = self.obs_encoder.encode(self._last_obs_raw, self._legal_actions)
        info = self._build_info(result.get("info", {}))
        return obs, info

    def step(self, action: int):
        if not self._legal_actions:
            recovered = self._recover_filtered_action_window(
                timeout_ms=min(self.step_timeout_ms, RECOVERY_MAX_WAIT_MS)
            )
            if not recovered:
                return self._make_terminal()

        normalized_action = self._normalize_action(action)
        if normalized_action is None or normalized_action >= len(self._legal_actions):
            return self._make_invalid_action_response(action)

        legal_action = self._legal_actions[normalized_action]

        try:
            result = self.bridge.step(
                episode_id=self._episode_id,
                action_id=legal_action.get("action_id"),
                timeout_ms=self.step_timeout_ms,
            )
        except Exception as exc:
            if self._is_episode_lost_error(exc):
                return self._make_step_recovery_response(exc)
            raise

        self._update_live_state(result)
        terminated = bool(result.get("done", False))
        truncated = bool(result.get("truncated", False))
        if not terminated and not truncated:
            self._recover_filtered_action_window(timeout_ms=min(self.step_timeout_ms, RECOVERY_MAX_WAIT_MS))

        obs = self.obs_encoder.encode(self._last_obs_raw, self._legal_actions)
        reward = float(result.get("reward", 0.0))
        info = self._build_info(result.get("info", {}))

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        mask = np.zeros(MAX_ACTIONS, dtype=bool)
        n = min(len(self._legal_actions), MAX_ACTIONS)
        mask[:n] = True
        return mask

    def recover_actionable_state(self, timeout_ms: int | None = None):
        recovered = self._recover_filtered_action_window(
            timeout_ms=min(timeout_ms or self.step_timeout_ms, RECOVERY_MAX_WAIT_MS)
        )
        obs = self.obs_encoder.encode(self._last_obs_raw or {}, self._legal_actions)
        info = self._build_info({})
        return recovered, obs, info

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
        phase = self._extract_phase(result)
        legal_actions = result.get("legal_actions", [])
        if isinstance(legal_actions, list):
            self._legal_actions = self._filter_legal_actions(legal_actions, phase=phase)
        else:
            self._legal_actions = []
        obs = result.get("obs", {})
        self._last_obs_raw = obs if isinstance(obs, dict) else {}
        self._last_action_overflow = max(len(self._legal_actions) - MAX_ACTIONS, 0)

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

        while time.monotonic() < deadline:
            try:
                result = self.bridge.reset(
                    character=self.character,
                    defensive_buffs=self.defensive_buffs,
                    timeout_ms=timeout_ms,
                )
            except Exception as exc:
                if not self._is_transient_reset_error(exc):
                    raise
                last_exc = exc
                time.sleep(RESET_READY_POLL_INTERVAL_S)
                continue

            phase = self._extract_phase(result)
            filtered_actions = self._filter_legal_actions(result.get("legal_actions", []), phase=phase)
            if filtered_actions:
                return result

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
            if not self._state_has_unblocked_actions(state):
                time.sleep(RECOVERY_POLL_INTERVAL_S)
                continue

            refreshed = self._safe_reset_into_current_run(timeout_ms)
            if refreshed is None:
                time.sleep(RECOVERY_POLL_INTERVAL_S)
                continue

            self._episode_id = refreshed.get("episode_id", self._episode_id)
            self._update_live_state(refreshed)
            if self._legal_actions:
                return True

            time.sleep(RECOVERY_POLL_INTERVAL_S)

        return False

    def _safe_get_state(self) -> dict[str, Any] | None:
        try:
            state = self.bridge.get_state()
        except Exception:
            return None
        return state if isinstance(state, dict) else None

    def _safe_reset_into_current_run(self, timeout_ms: int) -> dict[str, Any] | None:
        try:
            result = self._reset_with_ready_gate(
                timeout_ms=max(1_000, min(timeout_ms, self.reset_timeout_ms))
            )
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

    def _make_step_recovery_response(self, exc: Exception):
        self._episode_id = None
        obs = self.obs_encoder.encode(self._last_obs_raw or {}, [])
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
            },
        )
        return obs, 0.0, False, True, info
