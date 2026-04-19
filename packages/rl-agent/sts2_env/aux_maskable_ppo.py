"""Aux-aware MaskablePPO for action-conditioned omni-attention training."""

from __future__ import annotations

import time
from contextlib import nullcontext
from typing import Any, Generator, NamedTuple, Optional

import numpy as np
import torch as th
import torch.nn.functional as F
from gymnasium import spaces
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.buffers import MaskableDictRolloutBuffer
from sb3_contrib.common.maskable.utils import get_action_masks, is_masking_supported
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.type_aliases import TensorDict
from stable_baselines3.common.utils import explained_variance, obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv

from .aux_targets import (
    ENEMY_STATE_SLOT_COUNT,
    NUM_BUILD_HEADS,
    NUM_CAUSALITY_HEADS,
    NUM_ENEMY_STATE_FIELDS,
    NUM_OBJECTIVE_HEADS,
    NUM_ROUTE_HEADS,
    NUM_SELECTION_HEADS,
    NUM_TRAIT_HEADS,
    NUM_TRANSITION_HEADS,
)


def _resolve_amp_dtype(amp_dtype: str | None) -> tuple[str, th.dtype]:
    normalized = str(amp_dtype or "bf16").strip().lower()
    if normalized in {"bf16", "bfloat16"}:
        return "bf16", th.bfloat16
    raise ValueError(f"Unsupported AMP dtype '{amp_dtype}'. Only bf16/bfloat16 is supported on this training path.")


def _model_autocast_context(model):
    enabled = bool(getattr(model, "_amp_enabled", False))
    amp_dtype = getattr(model, "_amp_dtype", None)
    if not enabled or amp_dtype is None:
        return nullcontext()
    device = th.device(getattr(model, "device", "cpu"))
    return th.autocast(device_type=device.type, dtype=amp_dtype)


class AuxMaskableDictRolloutBufferSamples(NamedTuple):
    observations: TensorDict
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    action_masks: th.Tensor
    aux_objective_targets: th.Tensor
    aux_objective_mask: th.Tensor
    aux_transition_targets: th.Tensor
    aux_transition_mask: th.Tensor
    aux_trait_targets: th.Tensor
    aux_trait_mask: th.Tensor
    aux_build_targets: th.Tensor
    aux_build_mask: th.Tensor
    aux_selection_targets: th.Tensor
    aux_selection_mask: th.Tensor
    aux_route_targets: th.Tensor
    aux_route_mask: th.Tensor
    aux_enemy_state_targets: th.Tensor
    aux_enemy_state_mask: th.Tensor
    # Phase 8 Tier 2:
    aux_causality_targets: th.Tensor
    aux_causality_mask: th.Tensor


class AuxMaskableDictRolloutBuffer(MaskableDictRolloutBuffer):
    """Maskable dict rollout buffer extended with auxiliary supervision tensors."""

    aux_objective_targets: np.ndarray
    aux_objective_mask: np.ndarray
    aux_transition_targets: np.ndarray
    aux_transition_mask: np.ndarray
    aux_trait_targets: np.ndarray
    aux_trait_mask: np.ndarray
    aux_build_targets: np.ndarray
    aux_build_mask: np.ndarray
    aux_selection_targets: np.ndarray
    aux_selection_mask: np.ndarray
    aux_route_targets: np.ndarray
    aux_route_mask: np.ndarray
    aux_enemy_state_targets: np.ndarray
    aux_enemy_state_mask: np.ndarray
    # Phase 8 Tier 2
    aux_causality_targets: np.ndarray
    aux_causality_mask: np.ndarray

    def reset(self) -> None:
        super().reset()
        self.aux_objective_targets = np.zeros((self.buffer_size, self.n_envs, NUM_OBJECTIVE_HEADS), dtype=np.float32)
        self.aux_objective_mask = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.aux_transition_targets = np.zeros((self.buffer_size, self.n_envs, NUM_TRANSITION_HEADS), dtype=np.float32)
        self.aux_transition_mask = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.aux_trait_targets = np.zeros((self.buffer_size, self.n_envs, NUM_TRAIT_HEADS), dtype=np.float32)
        self.aux_trait_mask = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.aux_build_targets = np.zeros((self.buffer_size, self.n_envs, NUM_BUILD_HEADS), dtype=np.float32)
        self.aux_build_mask = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.aux_selection_targets = np.zeros((self.buffer_size, self.n_envs, NUM_SELECTION_HEADS), dtype=np.float32)
        self.aux_selection_mask = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.aux_route_targets = np.zeros((self.buffer_size, self.n_envs, NUM_ROUTE_HEADS), dtype=np.float32)
        self.aux_route_mask = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.aux_enemy_state_targets = np.zeros(
            (self.buffer_size, self.n_envs, ENEMY_STATE_SLOT_COUNT, NUM_ENEMY_STATE_FIELDS),
            dtype=np.float32,
        )
        self.aux_enemy_state_mask = np.zeros(
            (self.buffer_size, self.n_envs, ENEMY_STATE_SLOT_COUNT), dtype=np.float32
        )
        # Phase 8 Tier 2 causality (single 8-d target per env-step;
        # loss dispatches it to the chosen-candidate row at training time).
        self.aux_causality_targets = np.zeros(
            (self.buffer_size, self.n_envs, NUM_CAUSALITY_HEADS), dtype=np.float32
        )
        self.aux_causality_mask = np.zeros(
            (self.buffer_size, self.n_envs), dtype=np.float32
        )

    def add(self, *args, action_masks: Optional[np.ndarray] = None, aux_targets: Optional[dict[str, np.ndarray]] = None, **kwargs) -> None:
        pos = self.pos
        if aux_targets is not None:
            self._store_aux_payload(pos, aux_targets)
        super().add(*args, action_masks=action_masks, **kwargs)

    def _store_aux_payload(self, pos: int, aux_targets: dict[str, np.ndarray]) -> None:
        self.aux_objective_targets[pos] = self._coerce_target_matrix(aux_targets.get("objective"), self.n_envs, NUM_OBJECTIVE_HEADS)
        self.aux_objective_mask[pos] = self._coerce_mask_vector(aux_targets.get("objective_mask"), self.n_envs)
        self.aux_transition_targets[pos] = self._coerce_target_matrix(aux_targets.get("transition"), self.n_envs, NUM_TRANSITION_HEADS)
        self.aux_transition_mask[pos] = self._coerce_mask_vector(aux_targets.get("transition_mask"), self.n_envs)
        self.aux_trait_targets[pos] = self._coerce_target_matrix(aux_targets.get("traits"), self.n_envs, NUM_TRAIT_HEADS)
        self.aux_trait_mask[pos] = self._coerce_mask_vector(aux_targets.get("traits_mask"), self.n_envs)
        self.aux_build_targets[pos] = self._coerce_target_matrix(aux_targets.get("build"), self.n_envs, NUM_BUILD_HEADS)
        self.aux_build_mask[pos] = self._coerce_mask_vector(aux_targets.get("build_mask"), self.n_envs)
        self.aux_selection_targets[pos] = self._coerce_target_matrix(aux_targets.get("selection"), self.n_envs, NUM_SELECTION_HEADS)
        self.aux_selection_mask[pos] = self._coerce_mask_vector(aux_targets.get("selection_mask"), self.n_envs)
        self.aux_route_targets[pos] = self._coerce_target_matrix(aux_targets.get("route"), self.n_envs, NUM_ROUTE_HEADS)
        self.aux_route_mask[pos] = self._coerce_mask_vector(aux_targets.get("route_mask"), self.n_envs)
        self.aux_enemy_state_targets[pos] = self._coerce_enemy_state_targets(
            aux_targets.get("enemy_state"), self.n_envs
        )
        self.aux_enemy_state_mask[pos] = self._coerce_enemy_state_mask(
            aux_targets.get("enemy_state_mask"), self.n_envs
        )
        self.aux_causality_targets[pos] = self._coerce_target_matrix(
            aux_targets.get("causality"), self.n_envs, NUM_CAUSALITY_HEADS
        )
        self.aux_causality_mask[pos] = self._coerce_mask_vector(
            aux_targets.get("causality_mask"), self.n_envs
        )

    @staticmethod
    def _coerce_target_matrix(value: Any, n_envs: int, dim: int) -> np.ndarray:
        if value is None:
            return np.zeros((n_envs, dim), dtype=np.float32)
        array = np.asarray(value, dtype=np.float32)
        if array.shape == (dim,):
            array = np.repeat(array.reshape(1, dim), n_envs, axis=0)
        if array.shape != (n_envs, dim):
            return np.zeros((n_envs, dim), dtype=np.float32)
        return array

    @staticmethod
    def _coerce_enemy_state_targets(value: Any, n_envs: int) -> np.ndarray:
        target_shape = (n_envs, ENEMY_STATE_SLOT_COUNT, NUM_ENEMY_STATE_FIELDS)
        if value is None:
            return np.zeros(target_shape, dtype=np.float32)
        array = np.asarray(value, dtype=np.float32)
        if array.shape == (ENEMY_STATE_SLOT_COUNT, NUM_ENEMY_STATE_FIELDS):
            array = np.repeat(array.reshape(1, *array.shape), n_envs, axis=0)
        if array.shape != target_shape:
            return np.zeros(target_shape, dtype=np.float32)
        return array

    @staticmethod
    def _coerce_enemy_state_mask(value: Any, n_envs: int) -> np.ndarray:
        target_shape = (n_envs, ENEMY_STATE_SLOT_COUNT)
        if value is None:
            return np.zeros(target_shape, dtype=np.float32)
        array = np.asarray(value, dtype=np.float32)
        if array.shape == (ENEMY_STATE_SLOT_COUNT,):
            array = np.repeat(array.reshape(1, ENEMY_STATE_SLOT_COUNT), n_envs, axis=0)
        if array.shape != target_shape:
            return np.zeros(target_shape, dtype=np.float32)
        return array

    @staticmethod
    def _coerce_mask_vector(value: Any, n_envs: int) -> np.ndarray:
        if value is None:
            return np.zeros((n_envs,), dtype=np.float32)
        array = np.asarray(value, dtype=np.float32).reshape(-1)
        if array.size == 1:
            return np.full((n_envs,), float(array[0]), dtype=np.float32)
        if array.shape != (n_envs,):
            return np.zeros((n_envs,), dtype=np.float32)
        return array

    def get(self, batch_size: Optional[int] = None) -> Generator[AuxMaskableDictRolloutBufferSamples, None, None]:  # type: ignore[override]
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        if not self.generator_ready:
            for key, obs in self.observations.items():
                self.observations[key] = self.swap_and_flatten(obs)

            tensor_names = [
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
                "action_masks",
                "aux_objective_targets",
                "aux_objective_mask",
                "aux_transition_targets",
                "aux_transition_mask",
                "aux_trait_targets",
                "aux_trait_mask",
                "aux_build_targets",
                "aux_build_mask",
                "aux_selection_targets",
                "aux_selection_mask",
                "aux_route_targets",
                "aux_route_mask",
                "aux_enemy_state_targets",
                "aux_enemy_state_mask",
                "aux_causality_targets",
                "aux_causality_mask",
            ]
            for tensor in tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(self, batch_inds: np.ndarray, env=None) -> AuxMaskableDictRolloutBufferSamples:  # type: ignore[override]
        return AuxMaskableDictRolloutBufferSamples(
            observations={key: self.to_torch(obs[batch_inds]) for (key, obs) in self.observations.items()},
            actions=self.to_torch(self.actions[batch_inds]),
            old_values=self.to_torch(self.values[batch_inds].flatten()),
            old_log_prob=self.to_torch(self.log_probs[batch_inds].flatten()),
            advantages=self.to_torch(self.advantages[batch_inds].flatten()),
            returns=self.to_torch(self.returns[batch_inds].flatten()),
            action_masks=self.to_torch(self.action_masks[batch_inds].reshape(-1, self.mask_dims)),
            aux_objective_targets=self.to_torch(self.aux_objective_targets[batch_inds]),
            aux_objective_mask=self.to_torch(self.aux_objective_mask[batch_inds].flatten()),
            aux_transition_targets=self.to_torch(self.aux_transition_targets[batch_inds]),
            aux_transition_mask=self.to_torch(self.aux_transition_mask[batch_inds].flatten()),
            aux_trait_targets=self.to_torch(self.aux_trait_targets[batch_inds]),
            aux_trait_mask=self.to_torch(self.aux_trait_mask[batch_inds].flatten()),
            aux_build_targets=self.to_torch(self.aux_build_targets[batch_inds]),
            aux_build_mask=self.to_torch(self.aux_build_mask[batch_inds].flatten()),
            aux_selection_targets=self.to_torch(self.aux_selection_targets[batch_inds]),
            aux_selection_mask=self.to_torch(self.aux_selection_mask[batch_inds].flatten()),
            aux_route_targets=self.to_torch(self.aux_route_targets[batch_inds]),
            aux_route_mask=self.to_torch(self.aux_route_mask[batch_inds].flatten()),
            aux_enemy_state_targets=self.to_torch(self.aux_enemy_state_targets[batch_inds]),
            aux_enemy_state_mask=self.to_torch(self.aux_enemy_state_mask[batch_inds]),
            aux_causality_targets=self.to_torch(self.aux_causality_targets[batch_inds]),
            aux_causality_mask=self.to_torch(self.aux_causality_mask[batch_inds].flatten()),
        )


class AsyncAuxMaskableDictRolloutBuffer:
    """Flat async rollout buffer for ready-queue collection."""

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Dict,
        action_space: spaces.Space,
        device: str | th.device,
        gamma: float,
        gae_lambda: float,
        *,
        env_count: int,
    ) -> None:
        self.buffer_size = int(buffer_size)
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = th.device(device)
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.env_count = max(int(env_count), 1)
        self.mask_dims = int(action_space.n) if isinstance(action_space, spaces.Discrete) else int(np.prod(action_space.shape))
        self.reset()

    def reset(self) -> None:
        self.pos = 0
        self.full = False
        self.observations = {
            key: np.zeros((self.buffer_size, *space.shape), dtype=space.dtype)
            for key, space in self.observation_space.spaces.items()
        }
        self.actions = np.zeros((self.buffer_size, 1), dtype=np.int64)
        self.rewards = np.zeros((self.buffer_size,), dtype=np.float32)
        self.episode_starts = np.zeros((self.buffer_size,), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size,), dtype=np.float32)
        self.values = np.zeros((self.buffer_size,), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size,), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size,), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size,), dtype=np.float32)
        self.action_masks = np.zeros((self.buffer_size, self.mask_dims), dtype=np.float32)
        self.env_indices = np.zeros((self.buffer_size,), dtype=np.int32)
        self.aux_objective_targets = np.zeros((self.buffer_size, NUM_OBJECTIVE_HEADS), dtype=np.float32)
        self.aux_objective_mask = np.zeros((self.buffer_size,), dtype=np.float32)
        self.aux_transition_targets = np.zeros((self.buffer_size, NUM_TRANSITION_HEADS), dtype=np.float32)
        self.aux_transition_mask = np.zeros((self.buffer_size,), dtype=np.float32)
        self.aux_trait_targets = np.zeros((self.buffer_size, NUM_TRAIT_HEADS), dtype=np.float32)
        self.aux_trait_mask = np.zeros((self.buffer_size,), dtype=np.float32)
        self.aux_build_targets = np.zeros((self.buffer_size, NUM_BUILD_HEADS), dtype=np.float32)
        self.aux_build_mask = np.zeros((self.buffer_size,), dtype=np.float32)
        self.aux_selection_targets = np.zeros((self.buffer_size, NUM_SELECTION_HEADS), dtype=np.float32)
        self.aux_selection_mask = np.zeros((self.buffer_size,), dtype=np.float32)
        self.aux_route_targets = np.zeros((self.buffer_size, NUM_ROUTE_HEADS), dtype=np.float32)
        self.aux_route_mask = np.zeros((self.buffer_size,), dtype=np.float32)
        self.aux_enemy_state_targets = np.zeros(
            (self.buffer_size, ENEMY_STATE_SLOT_COUNT, NUM_ENEMY_STATE_FIELDS),
            dtype=np.float32,
        )
        self.aux_enemy_state_mask = np.zeros(
            (self.buffer_size, ENEMY_STATE_SLOT_COUNT), dtype=np.float32
        )
        # Phase 8 Tier 2 (async buffer). Single-env variant — still
        # n_envs = env_count collapsed into buffer_size via per-env add().
        self.aux_causality_targets = np.zeros(
            (self.buffer_size, NUM_CAUSALITY_HEADS), dtype=np.float32
        )
        self.aux_causality_mask = np.zeros(
            (self.buffer_size,), dtype=np.float32
        )
        self._prepared_filled = -1
        self._prepared_observations: dict[str, th.Tensor] = {}
        self._prepared_tensors: dict[str, th.Tensor] = {}

    def add(
        self,
        obs: dict[str, np.ndarray],
        action: int | np.ndarray,
        reward: float,
        episode_start: bool,
        value: float | np.ndarray | th.Tensor,
        log_prob: float | np.ndarray | th.Tensor,
        *,
        done: bool,
        env_index: int,
        action_masks: np.ndarray,
        aux_targets: Optional[dict[str, np.ndarray]] = None,
    ) -> None:
        if self.full or self.pos >= self.buffer_size:
            raise IndexError("Async rollout buffer is already full.")

        index = self.pos
        for key, value_array in self.observations.items():
            value_array[index] = np.asarray(obs[key], dtype=value_array.dtype)
        self.actions[index, 0] = int(np.asarray(action).reshape(-1)[0])
        self.rewards[index] = float(reward)
        self.episode_starts[index] = float(bool(episode_start))
        self.dones[index] = float(bool(done))
        self.values[index] = float(np.asarray(value).reshape(-1)[0])
        self.log_probs[index] = float(np.asarray(log_prob).reshape(-1)[0])
        self.action_masks[index] = np.asarray(action_masks, dtype=np.float32).reshape(self.mask_dims)
        self.env_indices[index] = int(env_index)

        if aux_targets is not None:
            self.aux_objective_targets[index] = AuxMaskableDictRolloutBuffer._coerce_target_matrix(aux_targets.get("objective"), 1, NUM_OBJECTIVE_HEADS)[0]
            self.aux_objective_mask[index] = AuxMaskableDictRolloutBuffer._coerce_mask_vector(aux_targets.get("objective_mask"), 1)[0]
            self.aux_transition_targets[index] = AuxMaskableDictRolloutBuffer._coerce_target_matrix(aux_targets.get("transition"), 1, NUM_TRANSITION_HEADS)[0]
            self.aux_transition_mask[index] = AuxMaskableDictRolloutBuffer._coerce_mask_vector(aux_targets.get("transition_mask"), 1)[0]
            self.aux_trait_targets[index] = AuxMaskableDictRolloutBuffer._coerce_target_matrix(aux_targets.get("traits"), 1, NUM_TRAIT_HEADS)[0]
            self.aux_trait_mask[index] = AuxMaskableDictRolloutBuffer._coerce_mask_vector(aux_targets.get("traits_mask"), 1)[0]
            self.aux_build_targets[index] = AuxMaskableDictRolloutBuffer._coerce_target_matrix(aux_targets.get("build"), 1, NUM_BUILD_HEADS)[0]
            self.aux_build_mask[index] = AuxMaskableDictRolloutBuffer._coerce_mask_vector(aux_targets.get("build_mask"), 1)[0]
            self.aux_selection_targets[index] = AuxMaskableDictRolloutBuffer._coerce_target_matrix(aux_targets.get("selection"), 1, NUM_SELECTION_HEADS)[0]
            self.aux_selection_mask[index] = AuxMaskableDictRolloutBuffer._coerce_mask_vector(aux_targets.get("selection_mask"), 1)[0]
            self.aux_route_targets[index] = AuxMaskableDictRolloutBuffer._coerce_target_matrix(aux_targets.get("route"), 1, NUM_ROUTE_HEADS)[0]
            self.aux_route_mask[index] = AuxMaskableDictRolloutBuffer._coerce_mask_vector(aux_targets.get("route_mask"), 1)[0]
            self.aux_enemy_state_targets[index] = AuxMaskableDictRolloutBuffer._coerce_enemy_state_targets(aux_targets.get("enemy_state"), 1)[0]
            self.aux_enemy_state_mask[index] = AuxMaskableDictRolloutBuffer._coerce_enemy_state_mask(aux_targets.get("enemy_state_mask"), 1)[0]
            # Phase 8 Tier 2 causality (per-step 8-d delta + scalar mask).
            self.aux_causality_targets[index] = AuxMaskableDictRolloutBuffer._coerce_target_matrix(aux_targets.get("causality"), 1, NUM_CAUSALITY_HEADS)[0]
            self.aux_causality_mask[index] = AuxMaskableDictRolloutBuffer._coerce_mask_vector(aux_targets.get("causality_mask"), 1)[0]

        self.pos += 1
        self.full = self.pos >= self.buffer_size
        self._prepared_filled = -1

    def compute_returns_and_advantage(self, last_values_by_env: dict[int, float]) -> None:
        filled = self.pos
        if filled <= 0:
            return
        for env_index in range(self.env_count):
            env_steps = np.flatnonzero(self.env_indices[:filled] == env_index)
            if env_steps.size <= 0:
                continue
            last_gae_lam = 0.0
            for offset in range(env_steps.size - 1, -1, -1):
                step_index = int(env_steps[offset])
                if offset == env_steps.size - 1:
                    next_non_terminal = 1.0 - self.dones[step_index]
                    next_values = float(last_values_by_env.get(env_index, 0.0))
                else:
                    next_step_index = int(env_steps[offset + 1])
                    next_non_terminal = 1.0 - self.episode_starts[next_step_index]
                    next_values = float(self.values[next_step_index])
                delta = self.rewards[step_index] + self.gamma * next_values * next_non_terminal - self.values[step_index]
                last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
                self.advantages[step_index] = last_gae_lam
        self.returns[:filled] = self.advantages[:filled] + self.values[:filled]

    def get(self, batch_size: Optional[int] = None) -> Generator[AuxMaskableDictRolloutBufferSamples, None, None]:
        assert self.full, ""
        filled = self.pos
        self.prepare_for_training()
        indices = th.randperm(filled, device=self.device)
        if batch_size is None:
            batch_size = filled
        start_idx = 0
        while start_idx < filled:
            batch_inds = indices[start_idx : start_idx + batch_size]
            yield AuxMaskableDictRolloutBufferSamples(
                observations={key: tensor.index_select(0, batch_inds) for key, tensor in self._prepared_observations.items()},
                actions=self._prepared_tensors["actions"].index_select(0, batch_inds),
                old_values=self._prepared_tensors["values"].index_select(0, batch_inds),
                old_log_prob=self._prepared_tensors["log_probs"].index_select(0, batch_inds),
                advantages=self._prepared_tensors["advantages"].index_select(0, batch_inds),
                returns=self._prepared_tensors["returns"].index_select(0, batch_inds),
                action_masks=self._prepared_tensors["action_masks"].index_select(0, batch_inds),
                aux_objective_targets=self._prepared_tensors["aux_objective_targets"].index_select(0, batch_inds),
                aux_objective_mask=self._prepared_tensors["aux_objective_mask"].index_select(0, batch_inds),
                aux_transition_targets=self._prepared_tensors["aux_transition_targets"].index_select(0, batch_inds),
                aux_transition_mask=self._prepared_tensors["aux_transition_mask"].index_select(0, batch_inds),
                aux_trait_targets=self._prepared_tensors["aux_trait_targets"].index_select(0, batch_inds),
                aux_trait_mask=self._prepared_tensors["aux_trait_mask"].index_select(0, batch_inds),
                aux_build_targets=self._prepared_tensors["aux_build_targets"].index_select(0, batch_inds),
                aux_build_mask=self._prepared_tensors["aux_build_mask"].index_select(0, batch_inds),
                aux_selection_targets=self._prepared_tensors["aux_selection_targets"].index_select(0, batch_inds),
                aux_selection_mask=self._prepared_tensors["aux_selection_mask"].index_select(0, batch_inds),
                aux_route_targets=self._prepared_tensors["aux_route_targets"].index_select(0, batch_inds),
                aux_route_mask=self._prepared_tensors["aux_route_mask"].index_select(0, batch_inds),
                aux_enemy_state_targets=self._prepared_tensors["aux_enemy_state_targets"].index_select(0, batch_inds),
                aux_enemy_state_mask=self._prepared_tensors["aux_enemy_state_mask"].index_select(0, batch_inds),
                aux_causality_targets=self._prepared_tensors["aux_causality_targets"].index_select(0, batch_inds),
                aux_causality_mask=self._prepared_tensors["aux_causality_mask"].index_select(0, batch_inds),
            )
            start_idx += batch_size

    def prepare_for_training(self) -> None:
        filled = self.pos
        if filled <= 0:
            self._prepared_filled = filled
            self._prepared_observations = {}
            self._prepared_tensors = {}
            return
        if self._prepared_filled == filled and self._prepared_observations and self._prepared_tensors:
            return

        def _slice_to_torch(array: np.ndarray, *, flatten: bool = False) -> th.Tensor:
            view = array[:filled]
            if flatten:
                view = view.reshape(filled)
            if not view.flags.c_contiguous:
                view = np.ascontiguousarray(view)
            return th.as_tensor(view, device=self.device)

        self._prepared_observations = {
            key: _slice_to_torch(obs)
            for key, obs in self.observations.items()
        }
        self._prepared_tensors = {
            "actions": _slice_to_torch(self.actions),
            "values": _slice_to_torch(self.values, flatten=True),
            "log_probs": _slice_to_torch(self.log_probs, flatten=True),
            "advantages": _slice_to_torch(self.advantages, flatten=True),
            "returns": _slice_to_torch(self.returns, flatten=True),
            "action_masks": _slice_to_torch(self.action_masks),
            "aux_objective_targets": _slice_to_torch(self.aux_objective_targets),
            "aux_objective_mask": _slice_to_torch(self.aux_objective_mask, flatten=True),
            "aux_transition_targets": _slice_to_torch(self.aux_transition_targets),
            "aux_transition_mask": _slice_to_torch(self.aux_transition_mask, flatten=True),
            "aux_trait_targets": _slice_to_torch(self.aux_trait_targets),
            "aux_trait_mask": _slice_to_torch(self.aux_trait_mask, flatten=True),
            "aux_build_targets": _slice_to_torch(self.aux_build_targets),
            "aux_build_mask": _slice_to_torch(self.aux_build_mask, flatten=True),
            "aux_selection_targets": _slice_to_torch(self.aux_selection_targets),
            "aux_selection_mask": _slice_to_torch(self.aux_selection_mask, flatten=True),
            "aux_route_targets": _slice_to_torch(self.aux_route_targets),
            "aux_route_mask": _slice_to_torch(self.aux_route_mask, flatten=True),
            "aux_enemy_state_targets": _slice_to_torch(self.aux_enemy_state_targets),
            "aux_enemy_state_mask": _slice_to_torch(self.aux_enemy_state_mask),
            "aux_causality_targets": _slice_to_torch(self.aux_causality_targets),
            "aux_causality_mask": _slice_to_torch(self.aux_causality_mask, flatten=True),
        }
        self._prepared_filled = filled

    def to_torch(self, array: np.ndarray) -> th.Tensor:
        return th.as_tensor(array, device=self.device)


class AuxMaskablePPO(MaskablePPO):
    """MaskablePPO variant that optimizes candidate-conditioned auxiliary heads."""

    rollout_buffer: AuxMaskableDictRolloutBuffer

    def __init__(
        self,
        *args,
        aux_objective_coef: float = 0.25,
        aux_transition_coef: float = 0.10,
        aux_trait_coef: float = 0.10,
        aux_build_coef: float = 0.10,
        aux_selection_coef: float = 0.10,
        aux_route_coef: float = 0.10,
        aux_enemy_state_coef: float = 0.10,
        aux_causality_coef: float = 0.10,
        amp: bool = False,
        amp_dtype: str = "bf16",
        **kwargs,
    ) -> None:
        kwargs.setdefault("rollout_buffer_class", AuxMaskableDictRolloutBuffer)
        self.aux_objective_coef = float(aux_objective_coef)
        self.aux_transition_coef = float(aux_transition_coef)
        self.aux_trait_coef = float(aux_trait_coef)
        self.aux_build_coef = float(aux_build_coef)
        self.aux_selection_coef = float(aux_selection_coef)
        self.aux_route_coef = float(aux_route_coef)
        self.aux_enemy_state_coef = float(aux_enemy_state_coef)
        self.aux_causality_coef = float(aux_causality_coef)
        self._amp_requested = bool(amp)
        self._amp_dtype_name, self._amp_dtype = _resolve_amp_dtype(amp_dtype)
        self._amp_enabled = False
        self._last_train_timing: dict[str, Any] = {}
        super().__init__(*args, **kwargs)
        self._amp_enabled = self._amp_requested and th.device(self.device).type == "cuda"

    @property
    def amp_enabled(self) -> bool:
        return bool(self._amp_enabled)

    @property
    def amp_dtype_name(self) -> str:
        return self._amp_dtype_name

    def amp_autocast_context(self):
        return _model_autocast_context(self)

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: AuxMaskableDictRolloutBuffer,
        n_rollout_steps: int,
        use_masking: bool = True,
    ) -> bool:
        assert isinstance(rollout_buffer, AuxMaskableDictRolloutBuffer), "RolloutBuffer doesn't support auxiliary targets"
        assert self._last_obs is not None, "No previous observation was provided"
        self.policy.set_training_mode(False)
        n_steps = 0
        action_masks = None
        rollout_buffer.reset()
        rollout_infos: list[dict[str, Any]] = []

        if use_masking and not is_masking_supported(env):
            raise ValueError("Environment does not support action masking. Consider using ActionMasker wrapper")

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            with th.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)  # type: ignore[arg-type]
                if use_masking:
                    action_masks = get_action_masks(env)
                with _model_autocast_context(self):
                    actions, values, log_probs = self.policy(obs_tensor, action_masks=action_masks)

            actions = actions.cpu().numpy()
            new_obs, rewards, dones, infos = env.step(actions)
            aux_targets = self._stack_aux_targets(infos)

            self.num_timesteps += env.num_envs
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            rollout_infos.extend(info for info in infos if isinstance(info, dict))
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                actions = actions.reshape(-1, 1)

            for idx, done in enumerate(dones):
                if done and infos[idx].get("terminal_observation") is not None and infos[idx].get("TimeLimit.truncated", False):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        with _model_autocast_context(self):
                            terminal_value = self.policy.predict_values(terminal_obs)[0]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs,
                actions,
                rewards,
                self._last_episode_starts,
                values,
                log_probs,
                action_masks=action_masks,
                aux_targets=aux_targets,
            )
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones

        with th.no_grad():
            with _model_autocast_context(self):
                values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
        self._record_rollout_timing_stats(rollout_infos)
        callback.on_rollout_end()
        return True

    def collect_rollouts_async(
        self,
        collector,
        rollout_buffer: AsyncAuxMaskableDictRolloutBuffer,
        n_rollout_steps: int,
    ) -> dict[str, Any]:
        assert self.policy is not None
        self.policy.set_training_mode(False)
        # Hard fail if every env has been permanently failed by the
        # collector — without any live env, the rollout loop would block
        # forever waiting for items that no worker can produce.
        if hasattr(collector, "live_env_ids") and not collector.live_env_ids:
            raise RuntimeError(
                "AsyncReadyCollector has 0 live envs (all permanently failed). "
                f"Failed envs: {getattr(collector, 'permanently_failed_env_ids', [])}. "
                "Training cannot continue without at least one live env."
            )
        rollout_buffer.reset()
        ready_pool: dict[int, Any] = {}
        pending: dict[int, dict[str, Any]] = {}
        rollout_infos: list[dict[str, Any]] = []
        collect_started = time.perf_counter()
        collector_timing_s = {
            "drain_ready": 0.0,
            "process_ready": 0.0,
            "aux_targets": 0.0,
            "buffer_add": 0.0,
            "batch_build": 0.0,
            "policy_forward": 0.0,
            "tensor_extract": 0.0,
            "pending_store": 0.0,
            "dispatch_actions": 0.0,
            "bootstrap_values": 0.0,
            "compute_returns": 0.0,
            "push_ready_back": 0.0,
        }
        collector_counts = {
            "loop_iterations": 0,
            "drain_calls": 0,
            "ready_items": 0,
            "transitions_added": 0,
            "worker_restart_events": 0,
            "dropped_pending_envs": 0,
            "orphan_transition_items": 0,
            "dispatch_batches": 0,
            "dispatched_envs": 0,
            "max_dispatch_batch": 0,
            "bootstrap_envs": 0,
        }
        drain_wait_samples_s: list[float] = []
        policy_forward_samples_s: list[float] = []
        dispatch_batch_sizes: list[float] = []

        while rollout_buffer.pos < n_rollout_steps or pending:
            collector_counts["loop_iterations"] += 1
            restart_events = collector.pop_restart_events() if hasattr(collector, "pop_restart_events") else []
            if restart_events:
                collector_counts["worker_restart_events"] += len(restart_events)
                for event in restart_events:
                    env_id = int(event.get("env_id", -1))
                    if env_id < 0:
                        continue
                    ready_pool.pop(env_id, None)
                    if pending.pop(env_id, None) is not None:
                        collector_counts["dropped_pending_envs"] += 1
                    if event.get("permanently_failed"):
                        # Counted separately so the train log can surface
                        # this distinct from ordinary worker restarts.
                        collector_counts.setdefault("permanently_failed_envs", 0)
                        collector_counts["permanently_failed_envs"] += 1
            # Mid-rollout sanity check: if every env has died and we have
            # no in-flight work left, we'd block forever waiting on items
            # nobody can produce. Raise so the operator can intervene.
            if (
                hasattr(collector, "live_env_ids")
                and not collector.live_env_ids
                and not pending
                and not ready_pool
            ):
                raise RuntimeError(
                    "AsyncReadyCollector lost all live envs during rollout "
                    f"(failed: {collector.permanently_failed_env_ids}). "
                    f"Buffer had {rollout_buffer.pos}/{n_rollout_steps} steps."
                )
            need_item = rollout_buffer.pos < n_rollout_steps and not ready_pool
            wait_for_ready = need_item or (pending and not ready_pool)
            min_items = 1 if wait_for_ready else 0
            if min_items > 0 or pending or ready_pool:
                drain_timeout_s = 180.0 if rollout_buffer.pos <= 0 and not pending and not ready_pool else 60.0
                drain_started = time.perf_counter()
                ready_items = collector.drain_ready(
                    min_items=min_items,
                    timeout_s=drain_timeout_s if min_items > 0 else 0.0,
                    return_on_restart=True,
                )
                drain_elapsed = time.perf_counter() - drain_started
                collector_timing_s["drain_ready"] += drain_elapsed
                collector_counts["drain_calls"] += 1
                collector_counts["ready_items"] += len(ready_items)
                drain_wait_samples_s.append(drain_elapsed)
                process_started = time.perf_counter()
                for item in ready_items:
                    next_ready_item = item if bool(getattr(item, "ready_for_action", True)) and item.obs is not None else None
                    if item.transition_info is not None:
                        prior = pending.pop(item.env_id, None)
                        if prior is None:
                            collector_counts["orphan_transition_items"] += 1
                        else:
                            aux_started = time.perf_counter()
                            aux_targets = self._stack_aux_targets([item.transition_info])
                            aux_single = {key: value[0] if isinstance(value, np.ndarray) and value.shape[0] == 1 else value for key, value in aux_targets.items()}
                            collector_timing_s["aux_targets"] += time.perf_counter() - aux_started
                            buffer_add_started = time.perf_counter()
                            rollout_buffer.add(
                                prior["obs"],
                                prior["action"],
                                float(item.reward or 0.0),
                                bool(prior["episode_start"]),
                                prior["value"],
                                prior["log_prob"],
                                done=bool(item.terminated or item.truncated),
                                env_index=item.env_id,
                                action_masks=prior["action_mask"],
                                aux_targets=aux_single,
                            )
                            collector_timing_s["buffer_add"] += time.perf_counter() - buffer_add_started
                            rollout_infos.append(item.transition_info)
                            collector_counts["transitions_added"] += 1
                    if next_ready_item is not None:
                        ready_pool[item.env_id] = type(item)(
                            env_id=item.env_id,
                            obs=item.obs,
                            info=item.info,
                            episode_start=item.episode_start,
                            generation=getattr(item, "generation", 0),
                            ready_for_action=True,
                        )
                collector_timing_s["process_ready"] += time.perf_counter() - process_started

            remaining_dispatch = max(n_rollout_steps - (rollout_buffer.pos + len(pending)), 0)
            if remaining_dispatch <= 0:
                continue

            ready_env_ids = sorted(ready_pool.keys())[:remaining_dispatch]
            if not ready_env_ids:
                continue

            ready_items = [ready_pool.pop(env_id) for env_id in ready_env_ids]
            batch_build_started = time.perf_counter()
            batched_obs = {key: np.stack([item.obs[key] for item in ready_items], axis=0) for key in ready_items[0].obs.keys()}
            batched_masks = np.stack([np.asarray(item.info.get("action_mask", item.obs.get("action_mask")), dtype=bool) for item in ready_items], axis=0)
            collector_timing_s["batch_build"] += time.perf_counter() - batch_build_started

            policy_started = time.perf_counter()
            with th.no_grad():
                obs_tensor = obs_as_tensor(batched_obs, self.device)
                with _model_autocast_context(self):
                    actions, values, log_probs = self.policy(obs_tensor, action_masks=batched_masks)
            policy_elapsed = time.perf_counter() - policy_started
            collector_timing_s["policy_forward"] += policy_elapsed
            policy_forward_samples_s.append(policy_elapsed)

            tensor_extract_started = time.perf_counter()
            action_array = actions.detach().cpu().numpy().reshape(-1)
            value_array = values.detach().float().cpu().numpy().reshape(-1)
            log_prob_array = log_probs.detach().float().cpu().numpy().reshape(-1)
            collector_timing_s["tensor_extract"] += time.perf_counter() - tensor_extract_started

            pending_store_started = time.perf_counter()
            for offset, item in enumerate(ready_items):
                pending[item.env_id] = {
                    "obs": item.obs,
                    "episode_start": bool(item.episode_start),
                    "action": int(action_array[offset]),
                    "value": float(value_array[offset]),
                    "log_prob": float(log_prob_array[offset]),
                    "action_mask": batched_masks[offset].astype(np.float32),
                }
            collector_timing_s["pending_store"] += time.perf_counter() - pending_store_started
            dispatch_started = time.perf_counter()
            collector.dispatch_actions(ready_env_ids, action_array)
            collector_timing_s["dispatch_actions"] += time.perf_counter() - dispatch_started
            collector_counts["dispatch_batches"] += 1
            collector_counts["dispatched_envs"] += len(ready_env_ids)
            collector_counts["max_dispatch_batch"] = max(int(collector_counts["max_dispatch_batch"]), len(ready_env_ids))
            dispatch_batch_sizes.append(float(len(ready_env_ids)))

        if ready_pool:
            bootstrap_started = time.perf_counter()
            final_obs = {key: np.stack([item.obs[key] for item in ready_pool.values()], axis=0) for key in next(iter(ready_pool.values())).obs.keys()}
            with th.no_grad():
                with _model_autocast_context(self):
                    final_values = self.policy.predict_values(obs_as_tensor(final_obs, self.device)).detach().float().cpu().numpy().reshape(-1)
            collector_timing_s["bootstrap_values"] += time.perf_counter() - bootstrap_started
            collector_counts["bootstrap_envs"] = len(ready_pool)
            last_values_by_env = {
                env_id: float(final_values[index])
                for index, env_id in enumerate(ready_pool.keys())
            }
        else:
            last_values_by_env = {}

        compute_returns_started = time.perf_counter()
        rollout_buffer.compute_returns_and_advantage(last_values_by_env)
        collector_timing_s["compute_returns"] += time.perf_counter() - compute_returns_started
        push_ready_started = time.perf_counter()
        collector.push_ready_back(list(ready_pool.values()))
        collector_timing_s["push_ready_back"] += time.perf_counter() - push_ready_started
        timing_summary = self._record_rollout_timing_stats(rollout_infos)
        collector_timing_s["total"] = time.perf_counter() - collect_started
        return {
            "timing": timing_summary,
            "transitions": rollout_buffer.pos,
            "collector_timing_s": collector_timing_s,
            "collector_counts": collector_counts,
            "collector_latency_ms": {
                "drain_ready": self._summarize_timing_samples_ms(drain_wait_samples_s),
                "policy_forward": self._summarize_timing_samples_ms(policy_forward_samples_s),
                "dispatch_batch_size": self._summarize_scalar_samples(dispatch_batch_sizes),
            },
        }

    def train(self) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]
        train_started = time.perf_counter()
        train_phase_timing_s = {
            "prepare_tensors": 0.0,
            "batch_fetch": 0.0,
            "policy_eval": 0.0,
            "loss_compute": 0.0,
            "optimizer_step": 0.0,
        }
        train_counts = {
            "epochs": 0,
            "batches": 0,
        }
        batch_fetch_samples_s: list[float] = []
        policy_eval_samples_s: list[float] = []
        optimizer_step_samples_s: list[float] = []

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        objective_losses, transition_losses, trait_losses = [], [], []
        build_losses, selection_losses, route_losses = [], [], []
        enemy_state_losses: list[float] = []
        causality_losses: list[float] = []

        if hasattr(self.rollout_buffer, "prepare_for_training"):
            prepare_started = time.perf_counter()
            self.rollout_buffer.prepare_for_training()
            train_phase_timing_s["prepare_tensors"] += time.perf_counter() - prepare_started

        continue_training = True
        for epoch in range(self.n_epochs):
            train_counts["epochs"] += 1
            approx_kl_divs = []
            rollout_iter = self.rollout_buffer.get(self.batch_size)
            while True:
                batch_fetch_started = time.perf_counter()
                try:
                    rollout_data = next(rollout_iter)
                except StopIteration:
                    break
                batch_fetch_elapsed = time.perf_counter() - batch_fetch_started
                train_phase_timing_s["batch_fetch"] += batch_fetch_elapsed
                batch_fetch_samples_s.append(batch_fetch_elapsed)
                train_counts["batches"] += 1
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                policy_eval_started = time.perf_counter()
                with _model_autocast_context(self):
                    if hasattr(self.policy, "evaluate_actions_with_aux"):
                        values, log_prob, entropy, aux_outputs = self.policy.evaluate_actions_with_aux(
                            rollout_data.observations,
                            actions,
                            action_masks=rollout_data.action_masks,
                        )
                    else:
                        values, log_prob, entropy = self.policy.evaluate_actions(
                            rollout_data.observations,
                            actions,
                            action_masks=rollout_data.action_masks,
                        )
                        aux_outputs = self.policy.forward_aux_heads(rollout_data.observations)
                values = values.float()
                log_prob = log_prob.float()
                if entropy is not None:
                    entropy = entropy.float()
                aux_outputs = {
                    key: value.float() if isinstance(value, th.Tensor) and value.is_floating_point() else value
                    for key, value in aux_outputs.items()
                }
                policy_eval_elapsed = time.perf_counter() - policy_eval_started
                train_phase_timing_s["policy_eval"] += policy_eval_elapsed
                policy_eval_samples_s.append(policy_eval_elapsed)

                loss_compute_started = time.perf_counter()
                values = values.flatten()
                advantages = rollout_data.advantages
                if self.normalize_advantage:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()
                pg_losses.append(policy_loss.item())

                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                if entropy is None:
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)
                entropy_losses.append(entropy_loss.item())

                aux_total_loss, aux_metrics = self._compute_auxiliary_losses(rollout_data, actions, aux_outputs=aux_outputs)
                objective_losses.append(aux_metrics["objective"])
                transition_losses.append(aux_metrics["transition"])
                trait_losses.append(aux_metrics["traits"])
                build_losses.append(aux_metrics["build"])
                selection_losses.append(aux_metrics["selection"])
                route_losses.append(aux_metrics["route"])
                enemy_state_losses.append(aux_metrics["enemy_state"])
                causality_losses.append(aux_metrics["causality"])

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss + aux_total_loss

                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = float(th.mean((th.exp(log_ratio) - 1) - log_ratio).float().cpu().item())
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                train_phase_timing_s["loss_compute"] += time.perf_counter() - loss_compute_started
                optimizer_step_started = time.perf_counter()
                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
                optimizer_step_elapsed = time.perf_counter() - optimizer_step_started
                train_phase_timing_s["optimizer_step"] += optimizer_step_elapsed
                optimizer_step_samples_s.append(optimizer_step_elapsed)

            if not continue_training:
                break

        self._n_updates += train_counts["epochs"]
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/aux_objective_loss", np.mean(objective_losses) if objective_losses else 0.0)
        self.logger.record("train/aux_transition_loss", np.mean(transition_losses) if transition_losses else 0.0)
        self.logger.record("train/aux_trait_loss", np.mean(trait_losses) if trait_losses else 0.0)
        self.logger.record("train/aux_build_loss", np.mean(build_losses) if build_losses else 0.0)
        self.logger.record("train/aux_selection_loss", np.mean(selection_losses) if selection_losses else 0.0)
        self.logger.record("train/aux_route_loss", np.mean(route_losses) if route_losses else 0.0)
        self.logger.record("train/aux_enemy_state_loss", np.mean(enemy_state_losses) if enemy_state_losses else 0.0)
        self.logger.record("train/aux_causality_loss", np.mean(causality_losses) if causality_losses else 0.0)
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        self.logger.record("train/aux_objective_coef", self.aux_objective_coef)
        self.logger.record("train/aux_transition_coef", self.aux_transition_coef)
        self.logger.record("train/aux_trait_coef", self.aux_trait_coef)
        self.logger.record("train/aux_build_coef", self.aux_build_coef)
        self.logger.record("train/aux_selection_coef", self.aux_selection_coef)
        self.logger.record("train/aux_route_coef", self.aux_route_coef)
        self.logger.record("train/aux_enemy_state_coef", self.aux_enemy_state_coef)
        self.logger.record("train/aux_causality_coef", self.aux_causality_coef)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
        self._last_train_timing = {
            "train_phase_timing_s": {
                **train_phase_timing_s,
                "total": time.perf_counter() - train_started,
            },
            "train_counts": train_counts,
            "train_latency_ms": {
                "batch_fetch": self._summarize_timing_samples_ms(batch_fetch_samples_s),
                "policy_eval": self._summarize_timing_samples_ms(policy_eval_samples_s),
                "optimizer_step": self._summarize_timing_samples_ms(optimizer_step_samples_s),
            },
        }

    def _record_rollout_timing_stats(self, infos: list[dict[str, Any]]) -> dict[str, float]:
        metrics = {
            "step.total": self._collect_timing_values(infos, ("bridge_info", "step_timing_ms", "total")),
            "step.after_wait": self._collect_timing_values(infos, ("bridge_info", "step_timing_ms", "after_wait")),
            "python.obs_encode": self._collect_timing_values(infos, ("python_timing_ms", "obs_encode")),
            "python.total": self._collect_timing_values(infos, ("python_timing_ms", "total")),
        }
        summary: dict[str, float] = {}
        for metric_name, values in metrics.items():
            if not values:
                continue
            p50 = float(np.percentile(values, 50))
            p95 = float(np.percentile(values, 95))
            self.logger.record(f"rollout/{metric_name.replace('.', '_')}_p50_ms", p50)
            self.logger.record(f"rollout/{metric_name.replace('.', '_')}_p95_ms", p95)
            summary[f"{metric_name}.p50"] = p50
            summary[f"{metric_name}.p95"] = p95
        return summary

    @staticmethod
    def _collect_timing_values(infos: list[dict[str, Any]], path: tuple[str, ...]) -> list[float]:
        values: list[float] = []
        for info in infos:
            current: Any = info
            for key in path:
                if not isinstance(current, dict):
                    current = None
                    break
                current = current.get(key)
            if current is None:
                continue
            try:
                values.append(float(current))
            except (TypeError, ValueError):
                continue
        return values

    def _stack_aux_targets(self, infos: list[dict[str, Any]]) -> dict[str, np.ndarray]:
        n_envs = len(infos)
        objective = np.zeros((n_envs, NUM_OBJECTIVE_HEADS), dtype=np.float32)
        objective_mask = np.zeros((n_envs,), dtype=np.float32)
        transition = np.zeros((n_envs, NUM_TRANSITION_HEADS), dtype=np.float32)
        transition_mask = np.zeros((n_envs,), dtype=np.float32)
        traits = np.zeros((n_envs, NUM_TRAIT_HEADS), dtype=np.float32)
        traits_mask = np.zeros((n_envs,), dtype=np.float32)
        build = np.zeros((n_envs, NUM_BUILD_HEADS), dtype=np.float32)
        build_mask = np.zeros((n_envs,), dtype=np.float32)
        selection = np.zeros((n_envs, NUM_SELECTION_HEADS), dtype=np.float32)
        selection_mask = np.zeros((n_envs,), dtype=np.float32)
        route = np.zeros((n_envs, NUM_ROUTE_HEADS), dtype=np.float32)
        route_mask = np.zeros((n_envs,), dtype=np.float32)
        enemy_state = np.zeros(
            (n_envs, ENEMY_STATE_SLOT_COUNT, NUM_ENEMY_STATE_FIELDS), dtype=np.float32
        )
        enemy_state_mask = np.zeros((n_envs, ENEMY_STATE_SLOT_COUNT), dtype=np.float32)
        causality = np.zeros((n_envs, NUM_CAUSALITY_HEADS), dtype=np.float32)
        causality_mask = np.zeros((n_envs,), dtype=np.float32)

        for index, info in enumerate(infos):
            aux = info.get("aux_targets") if isinstance(info, dict) else None
            if not isinstance(aux, dict):
                continue
            objective[index] = self._extract_target(aux, "objective", NUM_OBJECTIVE_HEADS)
            objective_mask[index] = self._extract_scalar(aux, "objective_mask", 1.0)
            transition[index] = self._extract_target(aux, "transition", NUM_TRANSITION_HEADS)
            transition_mask[index] = self._extract_scalar(aux, "transition_mask", 1.0)
            traits[index] = self._extract_target(aux, "traits", NUM_TRAIT_HEADS)
            traits_mask[index] = self._extract_scalar(aux, "traits_mask", 1.0)
            build[index] = self._extract_target(aux, "build", NUM_BUILD_HEADS)
            build_mask[index] = self._extract_scalar(aux, "build_mask", 1.0)
            selection[index] = self._extract_target(aux, "selection", NUM_SELECTION_HEADS)
            selection_mask[index] = self._extract_scalar(aux, "selection_mask", 1.0)
            route[index] = self._extract_target(aux, "route", NUM_ROUTE_HEADS)
            route_mask[index] = self._extract_scalar(aux, "route_mask", 1.0)
            enemy_state[index] = self._extract_enemy_state_target(aux)
            enemy_state_mask[index] = self._extract_enemy_state_mask(aux)
            causality[index] = self._extract_target(aux, "causality", NUM_CAUSALITY_HEADS)
            causality_mask[index] = self._extract_scalar(aux, "causality_mask", 1.0)

        return {
            "objective": objective,
            "objective_mask": objective_mask,
            "transition": transition,
            "transition_mask": transition_mask,
            "traits": traits,
            "traits_mask": traits_mask,
            "build": build,
            "build_mask": build_mask,
            "selection": selection,
            "selection_mask": selection_mask,
            "route": route,
            "route_mask": route_mask,
            "enemy_state": enemy_state,
            "enemy_state_mask": enemy_state_mask,
            "causality": causality,
            "causality_mask": causality_mask,
        }

    @staticmethod
    def _extract_enemy_state_target(aux: dict[str, Any]) -> np.ndarray:
        value = aux.get("enemy_state")
        target_shape = (ENEMY_STATE_SLOT_COUNT, NUM_ENEMY_STATE_FIELDS)
        if value is None:
            return np.zeros(target_shape, dtype=np.float32)
        array = np.asarray(value, dtype=np.float32)
        if array.shape == target_shape:
            return array
        if array.size == ENEMY_STATE_SLOT_COUNT * NUM_ENEMY_STATE_FIELDS:
            return array.reshape(target_shape)
        return np.zeros(target_shape, dtype=np.float32)

    @staticmethod
    def _extract_enemy_state_mask(aux: dict[str, Any]) -> np.ndarray:
        value = aux.get("enemy_state_mask")
        target_shape = (ENEMY_STATE_SLOT_COUNT,)
        if value is None:
            return np.zeros(target_shape, dtype=np.float32)
        array = np.asarray(value, dtype=np.float32).reshape(-1)
        if array.shape == target_shape:
            return array
        return np.zeros(target_shape, dtype=np.float32)

    @staticmethod
    def _extract_target(aux: dict[str, Any], key: str, dim: int) -> np.ndarray:
        value = aux.get(key)
        array = np.asarray(value, dtype=np.float32).reshape(-1) if value is not None else np.zeros((0,), dtype=np.float32)
        if array.shape != (dim,):
            return np.zeros((dim,), dtype=np.float32)
        return array

    @staticmethod
    def _extract_scalar(aux: dict[str, Any], key: str, default: float) -> float:
        value = aux.get(key, default)
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    def _compute_auxiliary_losses(
        self,
        rollout_data: AuxMaskableDictRolloutBufferSamples,
        actions: th.Tensor,
        *,
        aux_outputs: dict[str, th.Tensor] | None = None,
    ) -> tuple[th.Tensor, dict[str, float]]:
        aux_total = th.zeros((), device=self.device)
        metrics = {
            "objective": 0.0, "transition": 0.0, "traits": 0.0,
            "build": 0.0, "selection": 0.0, "route": 0.0,
            "enemy_state": 0.0, "causality": 0.0,
        }
        aux_outputs = aux_outputs or self.policy.forward_aux_heads(rollout_data.observations)

        if self.aux_objective_coef > 0.0:
            objective_pred = self._select_aux_prediction(aux_outputs, actions, candidate_key="candidate_objective", global_key="objective")
            objective_loss = self._masked_regression_loss(objective_pred, rollout_data.aux_objective_targets, rollout_data.aux_objective_mask)
            aux_total = aux_total + self.aux_objective_coef * objective_loss
            metrics["objective"] = float(objective_loss.detach().cpu().item())

        if self.aux_transition_coef > 0.0:
            transition_pred = self._select_aux_prediction(aux_outputs, actions, candidate_key="candidate_transition", global_key="transition")
            transition_loss = self._masked_regression_loss(transition_pred, rollout_data.aux_transition_targets, rollout_data.aux_transition_mask)
            aux_total = aux_total + self.aux_transition_coef * transition_loss
            metrics["transition"] = float(transition_loss.detach().cpu().item())

        if self.aux_trait_coef > 0.0:
            trait_pred = self._select_aux_prediction(aux_outputs, actions, candidate_key="candidate_traits", global_key="traits")
            trait_loss = self._masked_bce_loss(trait_pred, rollout_data.aux_trait_targets, rollout_data.aux_trait_mask)
            aux_total = aux_total + self.aux_trait_coef * trait_loss
            metrics["traits"] = float(trait_loss.detach().cpu().item())

        if self.aux_build_coef > 0.0 and "candidate_build" in aux_outputs:
            build_pred = self._select_aux_prediction(aux_outputs, actions, candidate_key="candidate_build", global_key="candidate_build")
            build_loss = self._masked_regression_loss(build_pred, rollout_data.aux_build_targets, rollout_data.aux_build_mask)
            aux_total = aux_total + self.aux_build_coef * build_loss
            metrics["build"] = float(build_loss.detach().cpu().item())

        if self.aux_selection_coef > 0.0 and "candidate_selection" in aux_outputs:
            selection_pred = self._select_aux_prediction(aux_outputs, actions, candidate_key="candidate_selection", global_key="candidate_selection")
            selection_loss = self._masked_bce_loss(selection_pred, rollout_data.aux_selection_targets, rollout_data.aux_selection_mask)
            aux_total = aux_total + self.aux_selection_coef * selection_loss
            metrics["selection"] = float(selection_loss.detach().cpu().item())

        if self.aux_route_coef > 0.0 and "candidate_route" in aux_outputs:
            route_pred = self._select_aux_prediction(aux_outputs, actions, candidate_key="candidate_route", global_key="candidate_route")
            route_loss = self._masked_regression_loss(route_pred, rollout_data.aux_route_targets, rollout_data.aux_route_mask)
            aux_total = aux_total + self.aux_route_coef * route_loss
            metrics["route"] = float(route_loss.detach().cpu().item())

        if self.aux_enemy_state_coef > 0.0 and "enemy_state" in aux_outputs:
            enemy_state_pred = aux_outputs["enemy_state"]
            enemy_state_loss = self._masked_enemy_state_loss(
                enemy_state_pred,
                rollout_data.aux_enemy_state_targets,
                rollout_data.aux_enemy_state_mask,
            )
            aux_total = aux_total + self.aux_enemy_state_coef * enemy_state_loss
            metrics["enemy_state"] = float(enemy_state_loss.detach().cpu().item())

        # Phase 8 Tier 2: action_causality. candidate_causality is
        # (B, n_actions, NUM_CAUSALITY_HEADS); gather the chosen-action
        # row per batch element via _select_aux_prediction, then compare
        # against the per-step 8-d delta. Mask is scalar per env-step
        # (skipped for e.g. terminal truncation/recovery paths).
        if self.aux_causality_coef > 0.0 and "candidate_causality" in aux_outputs:
            causality_pred = self._select_aux_prediction(
                aux_outputs, actions,
                candidate_key="candidate_causality",
                global_key="candidate_causality",
            )
            causality_loss = self._masked_regression_loss(
                causality_pred,
                rollout_data.aux_causality_targets,
                rollout_data.aux_causality_mask,
            )
            aux_total = aux_total + self.aux_causality_coef * causality_loss
            metrics["causality"] = float(causality_loss.detach().cpu().item())

        return aux_total, metrics

    @staticmethod
    def _masked_enemy_state_loss(
        pred: th.Tensor,
        targets: th.Tensor,
        mask: th.Tensor,
    ) -> th.Tensor:
        """MSE on hp/attribution slots + BCE on alive_next slot, per-enemy masked."""
        if pred.shape != targets.shape:
            return th.zeros((), device=pred.device)
        mask_f = mask.to(pred.dtype)
        total_mass = mask_f.sum().clamp_min(1.0)
        # Regression slots 0 (hp delta), 1 (attribution)
        reg_pred = pred[..., :2]
        reg_targets = targets[..., :2]
        reg_diff = (reg_pred - reg_targets).pow(2).sum(dim=-1)
        reg_loss = (reg_diff * mask_f).sum() / total_mass
        # Classification slot 2 (alive_next) via BCE-with-logits
        alive_pred = pred[..., 2]
        alive_target = targets[..., 2]
        alive_loss_elem = F.binary_cross_entropy_with_logits(
            alive_pred, alive_target, reduction="none"
        )
        alive_loss = (alive_loss_elem * mask_f).sum() / total_mass
        return reg_loss + alive_loss

    @staticmethod
    def _select_aux_prediction(aux_outputs: dict[str, th.Tensor], actions: th.Tensor, *, candidate_key: str, global_key: str) -> th.Tensor:
        if candidate_key in aux_outputs:
            pred = aux_outputs[candidate_key]
            rows = th.arange(pred.shape[0], device=pred.device)
            action_idx = actions.long().flatten().clamp(min=0, max=pred.shape[1] - 1)
            return pred[rows, action_idx]
        return aux_outputs[global_key]

    @staticmethod
    def _masked_regression_loss(pred: th.Tensor, target: th.Tensor, mask: th.Tensor) -> th.Tensor:
        mask = mask.float().reshape(-1)
        if pred.dim() == 1:
            pred = pred.unsqueeze(-1)
        if target.dim() == 1:
            target = target.unsqueeze(-1)
        per_item = F.smooth_l1_loss(pred, target, reduction="none").mean(dim=-1)
        denom = mask.sum()
        if float(denom.detach().cpu().item()) <= 0.0:
            return pred.new_zeros(())
        return (per_item * mask).sum() / denom

    @staticmethod
    def _masked_bce_loss(logits: th.Tensor, target: th.Tensor, mask: th.Tensor) -> th.Tensor:
        mask = mask.float().reshape(-1)
        per_item = F.binary_cross_entropy_with_logits(logits, target, reduction="none").mean(dim=-1)
        denom = mask.sum()
        if float(denom.detach().cpu().item()) <= 0.0:
            return logits.new_zeros(())
        return (per_item * mask).sum() / denom

    @staticmethod
    def _summarize_timing_samples_ms(values_s: list[float]) -> dict[str, float]:
        if not values_s:
            return {}
        values_ms = np.asarray(values_s, dtype=np.float64) * 1000.0
        return {
            "count": float(values_ms.size),
            "mean": float(np.mean(values_ms)),
            "p50": float(np.percentile(values_ms, 50)),
            "p95": float(np.percentile(values_ms, 95)),
            "max": float(np.max(values_ms)),
        }

    @staticmethod
    def _summarize_scalar_samples(values: list[float]) -> dict[str, float]:
        if not values:
            return {}
        array = np.asarray(values, dtype=np.float64)
        return {
            "count": float(array.size),
            "mean": float(np.mean(array)),
            "p50": float(np.percentile(array, 50)),
            "p95": float(np.percentile(array, 95)),
            "max": float(np.max(array)),
        }


__all__ = [
    "AsyncAuxMaskableDictRolloutBuffer",
    "AuxMaskableDictRolloutBuffer",
    "AuxMaskableDictRolloutBufferSamples",
    "AuxMaskablePPO",
]
