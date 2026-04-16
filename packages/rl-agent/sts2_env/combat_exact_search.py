"""Exact-state root/short-depth search for combat sandbox training."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import torch

from .combat_prior import build_combat_prior_bias
from .combat_template_env import CombatTemplateEnv


@dataclass
class ExactCombatState:
    history: tuple[int, ...]
    state_vector: np.ndarray
    action_mask: np.ndarray
    raw_obs: dict[str, Any]
    legal_actions: list[dict[str, Any]]
    binding: Any
    reward_from_parent: float
    terminated: bool
    truncated: bool


@dataclass
class SearchNode:
    history: tuple[int, ...]
    prior: np.ndarray | None = None
    value_prediction: float = 0.0
    visit_count: int = 0
    value_sum: float = 0.0
    children: dict[int, "SearchNode"] = field(default_factory=dict)
    state: ExactCombatState | None = None

    @property
    def q_value(self) -> float:
        if self.visit_count <= 0:
            return 0.0
        return self.value_sum / float(self.visit_count)


class ExactCombatSearch:
    """Exact-state PUCT search via reset + action replay.

    This intentionally avoids learned latent transitions. Every expanded node is
    recreated by resetting the combat sandbox to the original encounter opening
    state and replaying the selected fixed-slot action history.
    """

    def __init__(
        self,
        network: torch.nn.Module,
        *,
        env_factory: Callable[[], CombatTemplateEnv],
        device: str = "cpu",
        gamma: float = 0.997,
        num_simulations: int = 8,
        max_depth: int = 2,
        c_puct: float = 1.5,
        dirichlet_alpha: float = 0.3,
        exploration_fraction: float = 0.25,
        value_scale: float = 5.0,
    ) -> None:
        self.network = network
        self.env_factory = env_factory
        self.device = torch.device(device)
        self.gamma = float(gamma)
        self.num_simulations = max(0, int(num_simulations))
        self.max_depth = max(0, int(max_depth))
        self.c_puct = float(c_puct)
        self.dirichlet_alpha = float(dirichlet_alpha)
        self.exploration_fraction = float(exploration_fraction)
        self.value_scale = max(1e-6, float(value_scale))
        self._state_cache: dict[tuple[int, ...], ExactCombatState] = {}

    def run(
        self,
        root_env: CombatTemplateEnv,
        *,
        add_exploration_noise: bool = True,
    ) -> tuple[int, np.ndarray, dict[str, float]]:
        self._state_cache = {}
        reset_options = root_env.last_reset_options
        root_history = root_env.action_history_slots
        root_state = ExactCombatState(
            history=root_history,
            state_vector=root_env.state_vector.copy(),
            action_mask=root_env.action_masks(),
            raw_obs=deepcopy(root_env.raw_obs or {}),
            legal_actions=deepcopy(root_env.legal_actions),
            binding=root_env.binding,
            reward_from_parent=0.0,
            terminated=False,
            truncated=False,
        )
        self._state_cache[root_history] = root_state

        root = SearchNode(history=root_history, state=root_state)
        self._expand_node(root, root_state, add_exploration_noise=add_exploration_noise)

        exact_search_enabled = self.num_simulations > 0 and self.max_depth > 0
        fallback_to_root_only = not exact_search_enabled
        simulation_error = False
        if exact_search_enabled:
            try:
                for _ in range(self.num_simulations):
                    self._simulate(root, reset_options, depth=0)
            except Exception:
                fallback_to_root_only = True
                simulation_error = True

        legal_actions = np.where(root_state.action_mask > 0)[0].tolist()
        if fallback_to_root_only:
            selected, action_probs = self._select_from_prior(root, root_state)
        else:
            action_probs = np.zeros_like(root_state.action_mask, dtype=np.float32)
            total_visits = sum(root.children.get(action, SearchNode(())).visit_count for action in legal_actions)
            if total_visits <= 0:
                selected, action_probs = self._select_from_prior(root, root_state)
            else:
                for action in legal_actions:
                    child = root.children.get(action)
                    if child is not None and child.visit_count > 0:
                        action_probs[action] = child.visit_count / float(total_visits)
                selected = int(max(legal_actions, key=lambda action: action_probs[action])) if legal_actions else 0

        metrics = {
            "root_value": float(root.value_prediction),
            "root_legal_count": float(len(legal_actions)),
            "searched_nodes": float(len(self._state_cache)),
            "root_only": 1.0 if fallback_to_root_only else 0.0,
            "simulation_error": 1.0 if simulation_error else 0.0,
        }
        if exact_search_enabled:
            self._restore_root_env(root_env, reset_options, root_history)
        return selected, action_probs.astype(np.float32), metrics

    def _simulate(
        self,
        node: SearchNode,
        reset_options: dict[str, Any],
        *,
        depth: int,
    ) -> float:
        state = node.state or self._load_state(reset_options, node.history)
        node.state = state
        if state.terminated or state.truncated:
            return 0.0
        if node.prior is None:
            self._expand_node(node, state, add_exploration_noise=False)
        if depth >= self.max_depth:
            return float(node.value_prediction)

        legal_actions = np.where(state.action_mask > 0)[0].tolist()
        if not legal_actions:
            return float(node.value_prediction)

        action = self._select_action(node, legal_actions)
        child = node.children.get(action)
        if child is None:
            child = SearchNode(history=node.history + (int(action),))
            node.children[action] = child

        child_state = child.state or self._load_state(reset_options, child.history)
        child.state = child_state

        if child_state.terminated or child_state.truncated:
            future_value = 0.0
        else:
            if child.prior is None:
                self._expand_node(child, child_state, add_exploration_noise=False)
            if depth + 1 >= self.max_depth:
                future_value = float(child.value_prediction)
            else:
                future_value = self._simulate(child, reset_options, depth=depth + 1)

        target = float(child_state.reward_from_parent) + (self.gamma * future_value)
        child.visit_count += 1
        child.value_sum += target
        node.visit_count += 1
        return target

    def _select_action(self, node: SearchNode, legal_actions: list[int]) -> int:
        assert node.prior is not None
        total_visits = max(1, node.visit_count)
        best_action = legal_actions[0]
        best_score = -float("inf")
        for action in legal_actions:
            child = node.children.get(action)
            q_value = child.q_value if child is not None else 0.0
            u_value = self.c_puct * float(node.prior[action]) * np.sqrt(total_visits) / (1.0 + (child.visit_count if child is not None else 0))
            score = q_value + u_value
            if score > best_score:
                best_score = score
                best_action = int(action)
        return best_action

    def _expand_node(
        self,
        node: SearchNode,
        state: ExactCombatState,
        *,
        add_exploration_noise: bool,
    ) -> None:
        logits, value = self._network_forward(state.state_vector)
        legal_mask = np.asarray(state.action_mask, dtype=np.float32)
        prior_bias = build_combat_prior_bias(
            state.raw_obs,
            state.legal_actions,
            state.binding,
        )
        masked_logits = logits + prior_bias
        masked_logits[legal_mask <= 0] = -1e9
        prior = self._softmax(masked_logits)
        legal_actions = np.where(legal_mask > 0)[0].tolist()
        if add_exploration_noise and len(legal_actions) > 1:
            noise = np.random.gamma(self.dirichlet_alpha, 1.0, size=len(legal_actions))
            noise_sum = float(noise.sum())
            if noise_sum > 0.0:
                noise = noise / noise_sum
                prior = prior.copy()
                prior[legal_actions] = (
                    (1.0 - self.exploration_fraction) * prior[legal_actions]
                    + (self.exploration_fraction * noise.astype(np.float32))
                )
        node.prior = prior.astype(np.float32)
        node.value_prediction = float(value)

    def _network_forward(self, state_vector: np.ndarray) -> tuple[np.ndarray, float]:
        with torch.no_grad():
            tensor = torch.as_tensor(state_vector, dtype=torch.float32, device=self.device).unsqueeze(0)
            logits, value = self.network(tensor)
        logits_np = logits.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
        value_scalar = float(value.squeeze(0).detach().cpu().item()) * self.value_scale
        return logits_np, value_scalar

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        if logits.size == 0:
            return logits.astype(np.float32)
        max_value = np.max(logits)
        shifted = np.exp(logits - max_value)
        total = float(shifted.sum())
        if total <= 0.0:
            return np.zeros_like(logits, dtype=np.float32)
        return (shifted / total).astype(np.float32)

    def _select_from_prior(
        self,
        root: SearchNode,
        root_state: ExactCombatState,
    ) -> tuple[int, np.ndarray]:
        action_probs = np.zeros_like(root_state.action_mask, dtype=np.float32)
        legal_actions = np.where(root_state.action_mask > 0)[0].tolist()
        if legal_actions:
            prior = root.prior if root.prior is not None else np.zeros_like(action_probs)
            prior_total = float(prior[legal_actions].sum())
            if prior_total > 0.0:
                action_probs[legal_actions] = prior[legal_actions] / prior_total
            else:
                action_probs[legal_actions] = 1.0 / float(len(legal_actions))
            selected = int(max(legal_actions, key=lambda action: action_probs[action]))
        else:
            selected = 0
        return selected, action_probs.astype(np.float32)

    def _load_state(
        self,
        reset_options: dict[str, Any],
        history: tuple[int, ...],
    ) -> ExactCombatState:
        cached = self._state_cache.get(history)
        if cached is not None:
            return cached

        env = self.env_factory()
        try:
            obs, info = env.reset(options=deepcopy(reset_options))
            reward = 0.0
            terminated = False
            truncated = False
            for slot in history:
                obs, reward, terminated, truncated, info = env.step(int(slot))
                if terminated or truncated:
                    break
            state = ExactCombatState(
                history=history,
                state_vector=np.asarray(obs, dtype=np.float32).copy(),
                action_mask=np.asarray(info.get("action_mask"), dtype=np.float32).copy(),
                raw_obs=deepcopy(env.raw_obs or {}),
                legal_actions=deepcopy(env.legal_actions),
                binding=env.binding,
                reward_from_parent=float(reward),
                terminated=bool(terminated),
                truncated=bool(truncated),
            )
            self._state_cache[history] = state
            return state
        finally:
            env.close()

    def _restore_root_env(
        self,
        root_env: CombatTemplateEnv,
        reset_options: dict[str, Any],
        history: tuple[int, ...],
    ) -> None:
        root_env.reset(options=deepcopy(reset_options))
        for slot in history:
            _, _, terminated, truncated, _ = root_env.step(int(slot))
            if terminated or truncated:
                break
