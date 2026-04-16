"""Fixed-template combat sandbox wrapper for small combat expert training."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .combat_env import CombatSandboxEnv
from .combat_fixed_action import (
    NUM_FIXED_COMBAT_ACTIONS,
    FixedCombatActionBinding,
    build_fixed_action_binding,
)
from .observation_common import (
    CARD_FEAT_DIM,
    DenseObservationEncoder,
    ENEMY_FEAT_DIM,
    MAX_ENEMIES,
    MAX_HAND,
    MAX_POTIONS,
    POWER_DIM,
    SCALAR_DIM,
)


COMBAT_STATE_VECTOR_DIM = (
    SCALAR_DIM
    + (MAX_HAND * CARD_FEAT_DIM)
    + MAX_HAND
    + (MAX_ENEMIES * ENEMY_FEAT_DIM)
    + MAX_ENEMIES
    + POWER_DIM
    + MAX_POTIONS
)

INVALID_FIXED_ACTION_REWARD = -1.0


@dataclass(frozen=True)
class CombatSearchContext:
    reset_options: dict[str, Any]
    history_slots: tuple[int, ...]


def flatten_combat_state(obs_dict: dict[str, Any] | None) -> np.ndarray:
    if not isinstance(obs_dict, dict):
        return np.zeros(COMBAT_STATE_VECTOR_DIM, dtype=np.float32)

    def _take(key: str, shape: tuple[int, ...]) -> np.ndarray:
        value = obs_dict.get(key)
        if value is None:
            return np.zeros(shape, dtype=np.float32)
        arr = np.asarray(value, dtype=np.float32)
        if arr.shape != shape:
            try:
                arr = arr.reshape(shape)
            except Exception:
                return np.zeros(shape, dtype=np.float32)
        return arr

    parts = [
        _take("scalars", (SCALAR_DIM,)).reshape(-1),
        _take("hand", (MAX_HAND, CARD_FEAT_DIM)).reshape(-1),
        _take("hand_mask", (MAX_HAND,)).reshape(-1),
        _take("enemies", (MAX_ENEMIES, ENEMY_FEAT_DIM)).reshape(-1),
        _take("enemy_mask", (MAX_ENEMIES,)).reshape(-1),
        _take("player_powers", (POWER_DIM,)).reshape(-1),
        _take("potion_mask", (MAX_POTIONS,)).reshape(-1),
    ]
    return np.concatenate(parts, axis=0).astype(np.float32, copy=False)


class CombatTemplateEnv(gym.Env):
    """Combat-only environment with fixed template actions."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, **combat_env_kwargs: Any) -> None:
        super().__init__()
        combat_env_kwargs = dict(combat_env_kwargs)
        combat_env_kwargs.setdefault("obs_encoder", DenseObservationEncoder(use_text=False))
        self.base_env = CombatSandboxEnv(**combat_env_kwargs)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(COMBAT_STATE_VECTOR_DIM,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(NUM_FIXED_COMBAT_ACTIONS)
        self._last_obs_dict: dict[str, Any] | None = None
        self._last_state_vector = np.zeros(COMBAT_STATE_VECTOR_DIM, dtype=np.float32)
        self._last_binding = FixedCombatActionBinding(
            mask=np.zeros(NUM_FIXED_COMBAT_ACTIONS, dtype=bool),
            slot_to_legal={},
            slot_labels=[],
        )
        self._action_history_slots: list[int] = []
        self._last_reset_options: dict[str, Any] = {}

    @property
    def raw_obs(self) -> dict[str, Any] | None:
        return self.base_env.raw_obs

    @property
    def legal_actions(self) -> list[dict[str, Any]]:
        return self.base_env.legal_actions

    @property
    def binding(self) -> FixedCombatActionBinding:
        return self._last_binding

    @property
    def state_vector(self) -> np.ndarray:
        return self._last_state_vector

    @property
    def action_history_slots(self) -> tuple[int, ...]:
        return tuple(self._action_history_slots)

    @property
    def last_reset_options(self) -> dict[str, Any]:
        return deepcopy(self._last_reset_options)

    def current_search_context(self) -> CombatSearchContext:
        return CombatSearchContext(
            reset_options=self.last_reset_options,
            history_slots=self.action_history_slots,
        )

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        obs_dict, info = self.base_env.reset(seed=seed, options=options)
        self._last_reset_options = self.base_env.last_reset_kwargs
        self._action_history_slots = []
        self._refresh_cached_state(obs_dict)
        return self._last_state_vector.copy(), self._build_info(info)

    def step(self, action: int):
        legal_index = self._last_binding.legal_index_for_slot(int(action))
        if legal_index is None:
            info = self._build_info(
                {
                    "invalid_fixed_action": True,
                    "invalid_fixed_slot": int(action),
                }
            )
            return self._last_state_vector.copy(), INVALID_FIXED_ACTION_REWARD, False, True, info

        obs_dict, reward, terminated, truncated, info = self.base_env.step(legal_index)
        self._action_history_slots.append(int(action))
        self._refresh_cached_state(obs_dict)
        return self._last_state_vector.copy(), float(reward), bool(terminated), bool(truncated), self._build_info(info)

    def action_masks(self) -> np.ndarray:
        return self._last_binding.mask.copy()

    def render(self):
        return self.base_env.render()

    def close(self):
        return self.base_env.close()

    def _refresh_cached_state(self, obs_dict: dict[str, Any] | None) -> None:
        self._last_obs_dict = obs_dict if isinstance(obs_dict, dict) else {}
        self._last_state_vector = flatten_combat_state(self._last_obs_dict)
        self._last_binding = build_fixed_action_binding(self.base_env.raw_obs, self.base_env.legal_actions)

    def _build_info(self, info: dict[str, Any] | None) -> dict[str, Any]:
        result = dict(info or {})
        result["action_mask"] = self.action_masks()
        result["fixed_action_labels"] = self._last_binding.slot_labels
        result["fixed_action_count"] = int(self._last_binding.mask.sum())
        result["fixed_slot_to_legal"] = dict(self._last_binding.slot_to_legal)
        result["history_slots"] = list(self._action_history_slots)
        result["last_reset_options"] = self.last_reset_options
        return result
