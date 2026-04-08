"""Combat sandbox Gymnasium wrapper for STS2.

This mirrors :mod:`env_v2` but resets through ``/env/combat_reset``. Training
defaults to compact ``info`` payloads so rollout hot paths avoid carrying large
raw observation trees unless explicitly requested by debug/eval callers.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from combat_snapshot_dataset import snapshot_row_to_reset_kwargs
from .bridge_client import BridgeClient
from .observation_v2 import DictObservationEncoder, MAX_ACTIONS

INVALID_ACTION_REWARD = -1.0
INVALID_ACTION_REASON = "invalid_action_index"
BLOCKED_ACTION_KINDS = {"discard_potion"}


class CombatSandboxEnv(gym.Env):
    """Gymnasium Env for combat-only RL training via the STS2 bridge.

    Uses POST /env/combat_reset to enter a specific encounter, then
    POST /env/step for each action.  Episode ends when combat finishes.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        session_file: str | None = None,
        character: str | None = None,
        encounter_id: str | None = None,
        encounter_pool: list[str] | None = None,
        seed: int | None = None,
        current_hp: int | None = None,
        max_hp: int | None = None,
        max_energy: int | None = None,
        deck: list[str] | None = None,
        relics: list[str] | None = None,
        potions: list[str] | None = None,
        gold: int | None = None,
        snapshot_pool = None,
        reset_timeout_ms: int = 15000,
        step_timeout_ms: int = 20000,
        render_mode: str | None = None,
        obs_encoder: DictObservationEncoder | None = None,
        include_debug_info: bool = False,
    ) -> None:
        super().__init__()

        self.bridge = BridgeClient(session_path=session_file)
        self.obs_encoder = obs_encoder or DictObservationEncoder(use_text=False)
        self.character = character
        self.encounter_id = encounter_id
        self.encounter_pool = [eid for eid in (encounter_pool or []) if eid]
        self.seed = seed
        self.current_hp = current_hp
        self.max_hp = max_hp
        self.max_energy = max_energy
        self.deck = deck
        self.relics = relics
        self.potions = potions
        self.gold = gold
        self.snapshot_pool = snapshot_pool
        self.reset_timeout_ms = reset_timeout_ms
        self.step_timeout_ms = step_timeout_ms
        self.render_mode = render_mode
        self.include_debug_info = bool(include_debug_info)

        self.observation_space = self.obs_encoder.obs_space
        self.action_space = spaces.Discrete(MAX_ACTIONS)

        self._episode_id: str | None = None
        self._legal_actions: list[dict[str, Any]] = []
        self._last_obs_raw: dict[str, Any] | None = None
        self._current_encounter_id: str | None = encounter_id
        self._last_action_overflow: int = 0
        self._current_snapshot: dict[str, Any] | None = None

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)

        # Allow per-reset overrides via options dict
        opts = options or {}
        snapshot = opts.get("snapshot")
        if snapshot is None and self.snapshot_pool is not None:
            snapshot = self.snapshot_pool.sample(self.np_random)
        snapshot_kwargs = snapshot_row_to_reset_kwargs(snapshot) if isinstance(snapshot, dict) else {}

        encounter_id = opts.get("encounter_id", snapshot_kwargs.get("encounter_id"))
        if encounter_id is None:
            encounter_id = self._sample_encounter_id()
        reset_seed = opts.get("seed", self.seed)
        self._current_encounter_id = encounter_id
        self._current_snapshot = snapshot if isinstance(snapshot, dict) else None

        character = opts.get("character", snapshot_kwargs.get("character", self.character))
        current_hp = opts.get("current_hp", snapshot_kwargs.get("current_hp", self.current_hp))
        max_hp = opts.get("max_hp", snapshot_kwargs.get("max_hp", self.max_hp))
        max_energy = opts.get("max_energy", snapshot_kwargs.get("max_energy", self.max_energy))
        deck = opts.get("deck", snapshot_kwargs.get("deck", self.deck))
        relics = opts.get("relics", snapshot_kwargs.get("relics", self.relics))
        gold = opts.get("gold", snapshot_kwargs.get("gold", self.gold))
        if "potions" in opts:
            potions = opts.get("potions")
        elif snapshot_kwargs.get("potions") is not None:
            potions = snapshot_kwargs.get("potions")
        else:
            potions = self.potions

        result = self.bridge.combat_reset(
            character=character,
            encounter_id=encounter_id,
            seed=reset_seed,
            current_hp=current_hp,
            max_hp=max_hp,
            max_energy=max_energy,
            deck=deck,
            relics=relics,
            potions=potions,
            gold=gold,
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
        combat = self._last_obs_raw.get("combat", {})
        rnd = combat.get("round", "?")
        print(
            f"[CombatSandbox] Phase: {phase} | HP: {hp}/{max_hp} "
            f"| Round: {rnd} | Actions: {len(self._legal_actions)}"
        )

    def close(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _sample_encounter_id(self) -> str | None:
        if self.encounter_pool:
            idx = int(self.np_random.integers(len(self.encounter_pool)))
            return self.encounter_pool[idx]
        return self.encounter_id

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
        self._last_action_overflow = max(len(self._legal_actions) - MAX_ACTIONS, 0)

    def _decorate_bridge_info(self, bridge_info: Any) -> dict[str, Any]:
        info = dict(bridge_info) if isinstance(bridge_info, dict) else {}
        diagnostics = info.get("action_diagnostics")
        diagnostics = dict(diagnostics) if isinstance(diagnostics, dict) else {}
        diagnostics["legal_action_overflow"] = float(self._last_action_overflow)
        info["action_diagnostics"] = diagnostics
        return info

    def _build_info(self, bridge_info: Any, *, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        current_snapshot = self._current_snapshot or {}
        info: dict[str, Any] = {
            "episode_id": self._episode_id,
            "action_mask": self.action_masks(),
            "legal_action_count": len(self._legal_actions),
            "action_overflow": self._last_action_overflow,
            "phase": (self._last_obs_raw or {}).get("phase", "unknown"),
            "episode_mode": "combat_sandbox",
            "encounter_id": self._current_encounter_id,
            "encounter_pool": self.encounter_pool,
            "snapshot_sample_id": current_snapshot.get("sample_id"),
            "snapshot_run_id": current_snapshot.get("run_id"),
            "snapshot_floor_number": current_snapshot.get("floor_number"),
            "snapshot_build_id": current_snapshot.get("build_id"),
            "bridge_info": self._decorate_bridge_info(bridge_info),
        }
        if extra:
            info.update(extra)
        if self.include_debug_info:
            info["legal_actions"] = self._legal_actions
            info["raw_obs"] = self._last_obs_raw
            if current_snapshot:
                info["combat_snapshot"] = current_snapshot
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
