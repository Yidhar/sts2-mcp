"""
Monte Carlo Tree Search (MCTS) for MuZero-style RL in Slay the Spire 2.

Implements sampled MuZero MCTS with:
- UCB-based tree traversal
- Dirichlet noise for exploration
- Value/visit count-based action selection
- Min-max normalization for value bounds
"""

import torch
import numpy as np
import math
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple, Dict, List

from sts2_env.objective_heads import (
    HEAD_BUILD_PROGRESS,
    HEAD_HP_PRESERVATION,
    HEAD_RESOURCE_EFFICIENCY,
    HEAD_SURVIVAL,
    NUM_OBJECTIVE_HEADS,
    blended_objective_scalar_np,
    objective_weight_vector_np,
    scalarize_objective_components_np,
)
from sts2_env.observation_v2 import DECISION_DOMAINS, MAX_ACTIONS
from sts2_env.semantic_action import SEMANTIC_ACTION_FAMILIES, SEMANTIC_ROLE_NAMES, SEMANTIC_TARGET_SCOPES
from muzero.sts2_env.semantic_rollout import SEMANTIC_ROLLOUT_SIZE


@dataclass
class Node:
    """Tree node for MCTS."""

    visit_count: int = 0
    value_sum: float = 0.0
    value_component_sum: np.ndarray = field(default_factory=lambda: np.zeros(NUM_OBJECTIVE_HEADS, dtype=np.float32))
    prior: float = 0.0
    reward: float = 0.0
    reward_components: np.ndarray = field(default_factory=lambda: np.zeros(NUM_OBJECTIVE_HEADS, dtype=np.float32))
    hidden_state: Optional[torch.Tensor] = None
    semantic_hidden_state: Optional[torch.Tensor] = None
    action_embeddings: Optional[torch.Tensor] = None
    rollout_mode: str = "concrete"
    children: Dict[int, "Node"] = field(default_factory=dict)

    def expanded(self) -> bool:
        """Check if node has been expanded (has children)."""
        return len(self.children) > 0

    def value(self) -> float:
        """Get average value from visit count."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def value_components(self) -> np.ndarray:
        """Get average value components from visit count."""
        if self.visit_count == 0:
            return np.zeros(NUM_OBJECTIVE_HEADS, dtype=np.float32)
        return self.value_component_sum / float(self.visit_count)


@dataclass
class MinMaxStats:
    """Track min/max values in tree for UCB normalization."""

    min_value: float = float('inf')
    max_value: float = float('-inf')

    def update(self, value: float) -> None:
        """Update min/max with new value."""
        self.min_value = min(self.min_value, value)
        self.max_value = max(self.max_value, value)

    def normalize(self, value: float) -> float:
        """Normalize value to [0, 1] range using observed min/max."""
        if self.min_value == self.max_value:
            return value
        return (value - self.min_value) / (self.max_value - self.min_value)


class MCTS:
    """
    Monte Carlo Tree Search for MuZero networks.

    Implements sampled MuZero approach where if legal action set exceeds
    max_sampled_actions, randomly sample a subset for tree expansion.
    """

    def __init__(
        self,
        num_simulations: int = 50,
        discount: float = 0.997,
        c1: float = 1.25,
        c2: float = 19652,
        dirichlet_alpha: float = 0.25,
        exploration_fraction: float = 0.25,
        max_sampled_actions: int = 32,
        min_expanded_actions: int = 4,
        root_max_sampled_actions: int | None = None,
        child_max_sampled_actions: int | None = None,
        root_min_expanded_actions: int | None = None,
        child_min_expanded_actions: int | None = None,
        prior_floor: float = 0.0,
        relative_prior_floor: float = 0.0,
        surface_mask_blend: float = 0.15,
        end_turn_prior_bias: float = -1.0,
        objective_value_blend: float = 0.70,
        objective_prior_blend: float = 0.30,
        semantic_switch_depth: int = 2,
        enable_semantic_rollout: bool = True,
        semantic_rollout_chain_steps: int = 1,
        semantic_revisit_bonus: float = 0.15,
        root_progressive_widening_init: int = 2,
        child_progressive_widening_init: int = 1,
        root_progressive_widening_growth: float = 1.0,
        child_progressive_widening_growth: float = 1.0,
        root_bias_min_scale: float = 0.33,
        root_bias_decay_steps: int = 200_000,
        combat_search_mode: str = "grounded_root",
        combat_full_root_frontier: bool = True,
    ):
        """
        Initialize MCTS.

        Args:
            num_simulations: Number of tree traversals per action selection.
            discount: Reward discount factor.
            c1: UCB constant for exploration.
            c2: UCB constant for prior scaling.
            dirichlet_alpha: Alpha for Dirichlet noise on root priors.
            exploration_fraction: Fraction of Dirichlet noise to mix with priors.
            max_sampled_actions: Max actions to sample if legal set is larger.
            min_expanded_actions: Minimum actions to keep after prior pruning.
            root_max_sampled_actions: Optional root-specific candidate cap.
            child_max_sampled_actions: Optional child-node candidate cap.
            root_min_expanded_actions: Optional root-specific minimum kept actions.
            child_min_expanded_actions: Optional child-specific minimum kept actions.
            prior_floor: Absolute prior threshold for pruning expansion actions.
            relative_prior_floor: Relative-to-best prior threshold for pruning.
            surface_mask_blend: Blend factor for predicted latent legal-mask gating during child expansion.
            end_turn_prior_bias: Root-only additive logit bias for end_turn when energy remains and playable card actions exist.
            objective_value_blend: Blend factor between legacy scalar value and planner-objective scalarization.
            objective_prior_blend: Root prior bonus strength from objective-aware action semantics.
            semantic_switch_depth: Tree depth after which rollout switches from concrete actions to semantic actions.
            enable_semantic_rollout: Whether deep search may switch from concrete to semantic rollout.
            semantic_rollout_chain_steps: Additional semantic plies to auto-unroll inside the same simulation after
                entering semantic rollout. This forces semantic search to actually drill beyond the switch boundary.
            semantic_revisit_bonus: Small UCB bonus for semantic nodes so the tree revisits abstract branches instead
                of spending nearly all simulations breadth-first on concrete siblings.
            root_progressive_widening_init: Root children initially selectable before widening expands the frontier.
            child_progressive_widening_init: Non-root children initially selectable before widening expands the frontier.
            root_progressive_widening_growth: sqrt(visit)-scaled widening coefficient for the root.
            child_progressive_widening_growth: sqrt(visit)-scaled widening coefficient for non-root nodes.
            root_bias_min_scale: Final scale for heuristic root bias after decay.
            root_bias_decay_steps: Training steps over which root bias linearly decays.
            combat_search_mode: Combat-only search mode. ``grounded_root`` keeps combat planning on
                real observation-conditioned root actions and blocks deeper latent action rollout.
            combat_full_root_frontier: When grounded-root combat is enabled, expose the full root
                frontier instead of progressive widening so the shallow planner actually compares the
                available tactical choices.
        """
        self.num_simulations = num_simulations
        self.discount = discount
        self.c1 = c1
        self.c2 = c2
        self.dirichlet_alpha = dirichlet_alpha
        self.exploration_fraction = exploration_fraction
        self.max_sampled_actions = max_sampled_actions
        self.min_expanded_actions = max(int(min_expanded_actions), 1)
        self.root_max_sampled_actions = max(
            int(root_max_sampled_actions if root_max_sampled_actions is not None else max_sampled_actions), 1
        )
        default_child_max = child_max_sampled_actions if child_max_sampled_actions is not None else min(max_sampled_actions, 8)
        self.child_max_sampled_actions = max(int(default_child_max), 1)
        self.root_min_expanded_actions = max(
            int(root_min_expanded_actions if root_min_expanded_actions is not None else self.min_expanded_actions),
            1,
        )
        default_child_min = child_min_expanded_actions if child_min_expanded_actions is not None else min(self.min_expanded_actions, self.child_max_sampled_actions)
        self.child_min_expanded_actions = max(int(default_child_min), 1)
        self.prior_floor = max(float(prior_floor), 0.0)
        self.relative_prior_floor = max(float(relative_prior_floor), 0.0)
        self.surface_mask_blend = min(max(float(surface_mask_blend), 0.0), 1.0)
        self.end_turn_prior_bias = float(end_turn_prior_bias)
        self.objective_value_blend = min(max(float(objective_value_blend), 0.0), 1.0)
        self.objective_prior_blend = min(max(float(objective_prior_blend), 0.0), 1.0)
        self.semantic_switch_depth = max(int(semantic_switch_depth), 1)
        self._semantic_rollout_enabled = bool(enable_semantic_rollout)
        self.semantic_rollout_chain_steps = max(int(semantic_rollout_chain_steps), 0)
        self.semantic_revisit_bonus = max(float(semantic_revisit_bonus), 0.0)
        self.root_progressive_widening_init = max(int(root_progressive_widening_init), 1)
        self.child_progressive_widening_init = max(int(child_progressive_widening_init), 1)
        self.root_progressive_widening_growth = max(float(root_progressive_widening_growth), 0.0)
        self.child_progressive_widening_growth = max(float(child_progressive_widening_growth), 0.0)
        self.root_bias_min_scale = min(max(float(root_bias_min_scale), 0.0), 1.0)
        self.root_bias_decay_steps = max(int(root_bias_decay_steps), 0)
        combat_search_mode = str(combat_search_mode or "grounded_root").strip().lower()
        if combat_search_mode not in {"full", "grounded_root"}:
            raise ValueError(
                f"Unsupported combat_search_mode={combat_search_mode!r}; expected 'full' or 'grounded_root'."
            )
        self.combat_search_mode = combat_search_mode
        self.combat_full_root_frontier = bool(combat_full_root_frontier)
        self._root_bias_enabled = True
        self._training_step = 0
        self.last_run_stats: Dict[str, float] = {}
        self._active_decision_domain = "build"
        self._active_combat_root_only = False
        self._active_force_full_root_frontier = False
        self._active_semantic_rollout = self._semantic_rollout_enabled

    @staticmethod
    def _network_device(network) -> torch.device:
        try:
            return next(network.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _obs_to_device_batch(self, obs: Dict[str, Any], network) -> Dict[str, torch.Tensor]:
        device = self._network_device(network)
        obs_batch: Dict[str, torch.Tensor] = {}
        for key, value in obs.items():
            if isinstance(value, torch.Tensor):
                tensor = value.to(device)
            else:
                tensor = torch.as_tensor(value, device=device)

            if tensor.dim() == 0:
                tensor = tensor.unsqueeze(0)
            elif tensor.shape[0] != 1:
                tensor = tensor.unsqueeze(0)
            obs_batch[key] = tensor
        return obs_batch

    @staticmethod
    def _infer_decision_domain(obs: Dict[str, Any], fallback: str = "build") -> str:
        raw = obs.get("decision_domain") if isinstance(obs, dict) else None
        if isinstance(raw, str):
            raw_norm = raw.strip().lower()
            if raw_norm in DECISION_DOMAINS:
                return raw_norm
        try:
            raw_np = np.asarray(raw, dtype=np.float32).reshape(-1)
        except Exception:
            raw_np = np.zeros(0, dtype=np.float32)
        if raw_np.size == len(DECISION_DOMAINS) and float(raw_np.sum()) > 0.0:
            return DECISION_DOMAINS[int(np.argmax(raw_np))]
        return fallback

    def _activate_domain_search_mode(self, decision_domain: str) -> dict[str, Any]:
        domain = str(decision_domain or "build").strip().lower()
        if domain not in DECISION_DOMAINS:
            domain = "build"
        grounded_root = domain == "combat" and self.combat_search_mode == "grounded_root"
        self._active_decision_domain = domain
        self._active_combat_root_only = grounded_root
        self._active_force_full_root_frontier = grounded_root and self.combat_full_root_frontier
        self._active_semantic_rollout = self._semantic_rollout_enabled and not grounded_root
        return {
            "decision_domain": domain,
            "grounded_root": grounded_root,
            "semantic_rollout_enabled": self._active_semantic_rollout,
        }

    def _root_end_turn_context(self, obs: Dict, action_mask: np.ndarray) -> dict[str, Any]:
        actions = obs.get("actions")
        semantic_actions = obs.get("semantic_actions")
        scalars = obs.get("scalars")
        if actions is None or scalars is None:
            return {
                "wasteful": False,
                "energy": 0.0,
                "end_turn_indices": [],
                "positive_indices": [],
                "setup_scaling_indices": [],
                "positive_progress_count": 0,
                "zero_cost_positive": False,
                "severity": 0.0,
            }

        actions_np = actions.detach().cpu().numpy() if isinstance(actions, torch.Tensor) else np.asarray(actions)
        semantic_np = (
            semantic_actions.detach().cpu().numpy()
            if isinstance(semantic_actions, torch.Tensor)
            else np.asarray(semantic_actions)
        ) if semantic_actions is not None else None
        scalars_np = scalars.detach().cpu().numpy() if isinstance(scalars, torch.Tensor) else np.asarray(scalars)
        mask_np = action_mask.astype(bool, copy=False)

        if actions_np.ndim != 2 or scalars_np.ndim != 1 or actions_np.shape[0] < 1:
            return {
                "wasteful": False,
                "energy": 0.0,
                "end_turn_indices": [],
                "positive_indices": [],
                "setup_scaling_indices": [],
                "positive_progress_count": 0,
                "zero_cost_positive": False,
                "severity": 0.0,
            }

        # observation_v2 scalar index 29 = absolute energy / 10.0
        energy = float(scalars_np[29]) * 10.0 if scalars_np.shape[0] > 29 else 0.0
        if energy <= 1e-6:
            return {
                "wasteful": False,
                "energy": 0.0,
                "end_turn_indices": [],
                "positive_indices": [],
                "setup_scaling_indices": [],
                "positive_progress_count": 0,
                "zero_cost_positive": False,
                "severity": 0.0,
            }

        legal_indices = np.flatnonzero(mask_np)
        if legal_indices.size == 0:
            return {
                "wasteful": False,
                "energy": float(energy),
                "end_turn_indices": [],
                "positive_indices": [],
                "setup_scaling_indices": [],
                "positive_progress_count": 0,
                "zero_cost_positive": False,
                "severity": 0.0,
            }

        family_offset = 0
        target_offset = family_offset + len(SEMANTIC_ACTION_FAMILIES)
        role_offset = target_offset + len(SEMANTIC_TARGET_SCOPES)
        role_attack = role_offset + SEMANTIC_ROLE_NAMES.index("attack")
        role_block = role_offset + SEMANTIC_ROLE_NAMES.index("block")
        role_draw = role_offset + SEMANTIC_ROLE_NAMES.index("draw")
        role_debuff = role_offset + SEMANTIC_ROLE_NAMES.index("debuff")
        role_buff = role_offset + SEMANTIC_ROLE_NAMES.index("buff")
        role_heal = role_offset + SEMANTIC_ROLE_NAMES.index("heal")
        role_setup = role_offset + SEMANTIC_ROLE_NAMES.index("setup")
        role_scaling = role_offset + SEMANTIC_ROLE_NAMES.index("scaling")

        end_turn_indices: list[int] = []
        positive_indices: list[int] = []
        setup_scaling_indices: list[int] = []
        zero_cost_positive = False

        for raw_index in legal_indices.tolist():
            action_row = actions_np[raw_index]
            if action_row.shape[0] > 9 and action_row[9] > 0.5:
                end_turn_indices.append(int(raw_index))
                continue

            semantic_row = None
            if semantic_np is not None and semantic_np.ndim == 2 and raw_index < semantic_np.shape[0]:
                semantic_row = semantic_np[raw_index]

            direct_positive = bool(action_row.shape[0] > 38 and action_row[38] > 0.5)
            zero_cost = bool(action_row.shape[0] > 39 and action_row[39] > 0.5)
            semantic_positive = False
            setup_scaling = False
            if semantic_row is not None and semantic_row.shape[0] > role_scaling:
                semantic_positive = bool(
                    semantic_row[role_attack] > 0.5
                    or semantic_row[role_block] > 0.5
                    or semantic_row[role_draw] > 0.5
                    or semantic_row[role_debuff] > 0.5
                    or semantic_row[role_buff] > 0.5
                    or semantic_row[role_heal] > 0.5
                )
                setup_scaling = bool(
                    semantic_row[role_setup] > 0.5
                    or semantic_row[role_scaling] > 0.5
                    or action_row.shape[0] > 6 and action_row[6] > 0.5
                )

            if direct_positive or semantic_positive or setup_scaling:
                positive_indices.append(int(raw_index))
                if zero_cost:
                    zero_cost_positive = True
            if setup_scaling:
                setup_scaling_indices.append(int(raw_index))

        positive_progress_count = len(positive_indices)
        wasteful = len(end_turn_indices) > 0 and positive_progress_count > 0
        severity = 0.0
        if wasteful:
            severity = (
                1.0
                + 0.45 * min(max(float(energy), 0.0), 3.0)
                + 0.30 * min(float(positive_progress_count), 4.0)
                + (0.75 if zero_cost_positive else 0.0)
                + (0.35 if setup_scaling_indices else 0.0)
            )

        return {
            "wasteful": wasteful,
            "energy": float(energy),
            "end_turn_indices": end_turn_indices,
            "positive_indices": positive_indices,
            "setup_scaling_indices": setup_scaling_indices,
            "positive_progress_count": positive_progress_count,
            "zero_cost_positive": zero_cost_positive,
            "severity": float(severity),
        }

    def _end_turn_bias_value(self, obs: Dict, action_mask: np.ndarray) -> float:
        context = self._root_end_turn_context(obs, action_mask)
        if not bool(context.get("wasteful")):
            return 0.0

        energy = float(context.get("energy", 0.0))
        positive_progress_count = int(context.get("positive_progress_count", 0))
        zero_cost_positive = bool(context.get("zero_cost_positive", False))
        setup_scaling_count = len(context.get("setup_scaling_indices", []))
        bias = float(self.end_turn_prior_bias)

        if positive_progress_count >= 1:
            bias -= 0.75
        if positive_progress_count >= 2:
            bias -= 0.75
        if positive_progress_count >= 4:
            bias -= 0.50
        if energy >= 2.0:
            bias -= 0.75
        if energy >= 3.0:
            bias -= 0.75
        if zero_cost_positive:
            bias -= 1.0
        if setup_scaling_count > 0:
            bias -= 0.5
        return bias

    def _zero_energy_x_cost_indices(self, obs: Dict, action_mask: np.ndarray) -> list[int]:
        actions = obs.get("actions")
        semantic_actions = obs.get("semantic_actions")
        scalars = obs.get("scalars")
        if actions is None or semantic_actions is None or scalars is None:
            return []
        try:
            actions_np = actions.detach().cpu().numpy() if isinstance(actions, torch.Tensor) else np.asarray(actions)
            semantic_np = semantic_actions.detach().cpu().numpy() if isinstance(semantic_actions, torch.Tensor) else np.asarray(semantic_actions)
            scalars_np = scalars.detach().cpu().numpy() if isinstance(scalars, torch.Tensor) else np.asarray(scalars)
            energy = float(scalars_np.reshape(-1)[29]) * 10.0 if scalars_np.size > 29 else 0.0
            if energy > 0.05 or actions_np.ndim != 2 or semantic_np.ndim != 2:
                return []
            legal_mask = np.asarray(action_mask, dtype=np.float32).reshape(-1) > 0
            family_offset = 0
            role_offset = len(SEMANTIC_ACTION_FAMILIES) + len(SEMANTIC_TARGET_SCOPES)
            family_play_card = family_offset + SEMANTIC_ACTION_FAMILIES.index("play_card")
            role_x_cost = role_offset + SEMANTIC_ROLE_NAMES.index("x_cost")
            count = min(actions_np.shape[0], semantic_np.shape[0], legal_mask.shape[0], MAX_ACTIONS)
            indices: list[int] = []
            for index in range(count):
                if not legal_mask[index]:
                    continue
                semantic_row = semantic_np[index]
                is_play = bool(semantic_row.shape[0] > family_play_card and semantic_row[family_play_card] > 0.5)
                is_x = bool(semantic_row.shape[0] > role_x_cost and semantic_row[role_x_cost] > 0.5)
                if is_play and is_x:
                    indices.append(int(index))
            return indices
        except Exception:
            return []

    def _apply_zero_energy_x_cost_selection_guard(
        self,
        obs: Dict,
        action_mask: np.ndarray,
        available_actions: list[int],
        selection_scores: np.ndarray,
    ) -> tuple[np.ndarray, bool, bool]:
        x_indices = set(self._zero_energy_x_cost_indices(obs, action_mask))
        if not x_indices:
            return selection_scores, False, False
        adjusted = selection_scores.astype(np.float32, copy=True)
        for local_index, action_index in enumerate(available_actions):
            if action_index in x_indices:
                adjusted[local_index] *= 0.02
        if float(adjusted.sum()) <= 1e-9:
            return selection_scores, True, False
        raw_choice = int(available_actions[int(np.argmax(selection_scores))]) if selection_scores.size > 0 else 0
        adjusted_choice = int(available_actions[int(np.argmax(adjusted))]) if adjusted.size > 0 else raw_choice
        forced_alternative = raw_choice in x_indices and adjusted_choice not in x_indices
        return adjusted, True, forced_alternative

    def _apply_end_turn_selection_guard(
        self,
        obs: Dict,
        action_mask: np.ndarray,
        available_actions: list[int],
        selection_scores: np.ndarray,
    ) -> tuple[np.ndarray, bool, bool]:
        context = self._root_end_turn_context(obs, action_mask)
        if not bool(context.get("wasteful")):
            return selection_scores, False, False

        end_turn_indices = set(int(index) for index in context.get("end_turn_indices", []))
        positive_indices = set(int(index) for index in context.get("positive_indices", []))
        if not end_turn_indices or not positive_indices:
            return selection_scores, False, False

        adjusted = selection_scores.astype(np.float32, copy=True)
        severity = max(float(context.get("severity", 0.0)), 1.0)
        end_turn_scale = max(0.02, 0.18 / severity)
        positive_scale = 1.0 + min(0.35 * severity, 1.25)

        for local_index, action_index in enumerate(available_actions):
            if action_index in end_turn_indices:
                adjusted[local_index] *= end_turn_scale
            elif action_index in positive_indices:
                adjusted[local_index] *= positive_scale

        if float(adjusted.sum()) <= 1e-9:
            return selection_scores, True, False

        raw_choice = int(available_actions[int(np.argmax(selection_scores))]) if selection_scores.size > 0 else 0
        adjusted_choice = int(available_actions[int(np.argmax(adjusted))]) if adjusted.size > 0 else raw_choice
        forced_alternative = raw_choice in end_turn_indices and adjusted_choice not in end_turn_indices
        return adjusted, True, forced_alternative

    def set_training_step(self, step: int) -> None:
        self._training_step = max(int(step), 0)

    def set_root_bias_enabled(self, enabled: bool) -> None:
        self._root_bias_enabled = bool(enabled)

    def set_semantic_rollout_enabled(self, enabled: bool) -> None:
        self._semantic_rollout_enabled = bool(enabled)

    def _current_root_bias_scale(self) -> float:
        if not self._root_bias_enabled:
            return 0.0
        if self.root_bias_decay_steps <= 0:
            return 1.0
        progress = min(max(float(self._training_step) / float(self.root_bias_decay_steps), 0.0), 1.0)
        return float(1.0 - progress * (1.0 - self.root_bias_min_scale))

    @staticmethod
    def _objective_context_np(obs: Dict) -> np.ndarray:
        context = obs.get("objective_context")
        if context is None:
            return np.zeros(0, dtype=np.float32)
        if isinstance(context, torch.Tensor):
            return context.detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1)
        return np.asarray(context, dtype=np.float32).reshape(-1)

    def _search_value(
        self,
        scalar_value: float,
        value_components: np.ndarray,
        objective_context: np.ndarray,
    ) -> float:
        return blended_objective_scalar_np(
            scalar_value,
            value_components,
            objective_context,
            component_blend=self.objective_value_blend,
        )

    def _objective_prior_bias(self, obs: Dict, action_mask: np.ndarray) -> np.ndarray:
        actions = obs.get("actions")
        semantic_actions = obs.get("semantic_actions")
        root_bias_scale = self._current_root_bias_scale()
        if actions is None or semantic_actions is None or self.objective_prior_blend <= 0.0 or root_bias_scale <= 0.0:
            return np.zeros(MAX_ACTIONS, dtype=np.float32)

        actions_np = (
            actions.detach().cpu().numpy()
            if isinstance(actions, torch.Tensor)
            else np.asarray(actions, dtype=np.float32)
        )
        semantic_np = (
            semantic_actions.detach().cpu().numpy()
            if isinstance(semantic_actions, torch.Tensor)
            else np.asarray(semantic_actions, dtype=np.float32)
        )
        if actions_np.ndim != 2 or semantic_np.ndim != 2:
            return np.zeros(MAX_ACTIONS, dtype=np.float32)

        objective_context = self._objective_context_np(obs)
        weights = objective_weight_vector_np(objective_context)
        preserve_hp_bias = float(objective_context[4]) if objective_context.size > 4 else weights[1]
        save_potion_mode = float(objective_context[5]) if objective_context.size > 5 else 0.0
        force_rest_mode = float(objective_context[6]) if objective_context.size > 6 else 0.0
        greed_upgrade_mode = float(objective_context[7]) if objective_context.size > 7 else 0.0
        safe_route_bias = float(objective_context[10]) if objective_context.size > 10 else preserve_hp_bias
        shop_value_bias = float(objective_context[11]) if objective_context.size > 11 else weights[3]
        rest_value_bias = float(objective_context[12]) if objective_context.size > 12 else force_rest_mode
        smith_value_bias = float(objective_context[13]) if objective_context.size > 13 else greed_upgrade_mode

        family_offset = 0
        family_count = len(SEMANTIC_ACTION_FAMILIES)
        role_offset = family_count + len(SEMANTIC_TARGET_SCOPES)
        role_attack = role_offset + SEMANTIC_ROLE_NAMES.index("attack")
        role_block = role_offset + SEMANTIC_ROLE_NAMES.index("block")
        role_draw = role_offset + SEMANTIC_ROLE_NAMES.index("draw")
        role_debuff = role_offset + SEMANTIC_ROLE_NAMES.index("debuff")
        role_heal = role_offset + SEMANTIC_ROLE_NAMES.index("heal")
        role_aoe = role_offset + SEMANTIC_ROLE_NAMES.index("aoe")
        role_x_cost = role_offset + SEMANTIC_ROLE_NAMES.index("x_cost")
        role_setup = role_offset + SEMANTIC_ROLE_NAMES.index("setup")
        role_scaling = role_offset + SEMANTIC_ROLE_NAMES.index("scaling")
        role_resource = role_offset + SEMANTIC_ROLE_NAMES.index("resource")
        family_play_card = family_offset + SEMANTIC_ACTION_FAMILIES.index("play_card")
        family_use_potion = family_offset + SEMANTIC_ACTION_FAMILIES.index("use_potion")
        family_end_turn = family_offset + SEMANTIC_ACTION_FAMILIES.index("end_turn")
        family_map = family_offset + SEMANTIC_ACTION_FAMILIES.index("map")
        family_card_reward = family_offset + SEMANTIC_ACTION_FAMILIES.index("card_reward")
        family_reward = family_offset + SEMANTIC_ACTION_FAMILIES.index("reward")
        family_shop = family_offset + SEMANTIC_ACTION_FAMILIES.index("shop")
        family_rest = family_offset + SEMANTIC_ACTION_FAMILIES.index("rest")
        family_smith = family_offset + SEMANTIC_ACTION_FAMILIES.index("smith")
        family_deck_upgrade = family_offset + SEMANTIC_ACTION_FAMILIES.index("deck_upgrade")
        family_treasure_relic = family_offset + SEMANTIC_ACTION_FAMILIES.index("treasure_relic")

        legal_mask = np.asarray(action_mask, dtype=np.float32) > 0
        bias = np.zeros(MAX_ACTIONS, dtype=np.float32)
        count = min(actions_np.shape[0], semantic_np.shape[0], MAX_ACTIONS)
        for index in range(count):
            if not legal_mask[index]:
                continue
            action_row = actions_np[index]
            semantic_row = semantic_np[index]

            progress = float(action_row[38]) if action_row.shape[0] > 38 else 0.0
            zero_cost = float(action_row[39]) if action_row.shape[0] > 39 else 0.0
            preview_damage = float(action_row[22]) if action_row.shape[0] > 22 else 0.0
            preview_block = float(action_row[23]) if action_row.shape[0] > 23 else 0.0
            draw_value = float(action_row[24]) if action_row.shape[0] > 24 else 0.0
            point_type = float(action_row[15]) if action_row.shape[0] > 15 else 0.0

            score = 0.0
            score += weights[HEAD_HP_PRESERVATION] * (
                0.85 * float(semantic_row[role_block] > 0.5)
                + 0.55 * float(semantic_row[role_heal] > 0.5)
                + 0.40 * float(semantic_row[role_debuff] > 0.5)
                + 0.35 * preview_block
                + 0.18 * progress
            )
            score += weights[HEAD_SURVIVAL] * (
                0.40 * float(semantic_row[role_attack] > 0.5)
                + 0.22 * preview_damage
                + 0.12 * float(semantic_row[role_aoe] > 0.5)
            )
            score += weights[HEAD_BUILD_PROGRESS] * (
                0.70 * float(semantic_row[role_scaling] > 0.5)
                + 0.45 * float(semantic_row[role_setup] > 0.5)
                + 0.30 * float(semantic_row[role_draw] > 0.5)
                + 0.20 * draw_value
            )
            score += weights[HEAD_RESOURCE_EFFICIENCY] * (
                0.45 * float(semantic_row[role_resource] > 0.5)
                + 0.20 * float(semantic_row[role_x_cost] > 0.5)
                + 0.15 * zero_cost
            )
            try:
                scalars = obs.get("scalars")
                scalars_np = scalars.detach().cpu().numpy() if isinstance(scalars, torch.Tensor) else np.asarray(scalars)
                energy = float(scalars_np.reshape(-1)[29]) * 10.0 if scalars_np.size > 29 else 0.0
            except Exception:
                energy = 0.0
            if energy <= 0.05 and semantic_row[family_play_card] > 0.5 and semantic_row[role_x_cost] > 0.5:
                score -= 3.0

            if semantic_row[family_use_potion] > 0.5:
                # Old semantic/MCTS path has no raw potion payload, so approximate
                # timing quality from encoded roles + previews.  Potion family
                # alone is no longer a positive signal: save-mode should dominate
                # unless the potion clearly contributes damage/block/heal/debuff or
                # can be converted as draw/resource.
                potion_useful = (
                    0.50 * float(semantic_row[role_attack] > 0.5)
                    + 0.60 * float(
                        semantic_row[role_block] > 0.5
                        or semantic_row[role_heal] > 0.5
                        or semantic_row[role_debuff] > 0.5
                    )
                    + 0.35 * float(semantic_row[role_draw] > 0.5 or semantic_row[role_resource] > 0.5)
                    + 0.15 * preview_damage
                    + 0.20 * preview_block
                )
                score += 0.25 * min(float(potion_useful), 1.0)
                score -= 0.25 + 0.85 * save_potion_mode
            if semantic_row[family_end_turn] > 0.5:
                score -= 0.35 * preserve_hp_bias
            if semantic_row[family_rest] > 0.5:
                score += 0.80 * rest_value_bias
                score -= 0.25 * smith_value_bias
            if semantic_row[family_smith] > 0.5 or semantic_row[family_deck_upgrade] > 0.5:
                score += 0.80 * smith_value_bias
                score -= 0.20 * rest_value_bias
            if semantic_row[family_shop] > 0.5:
                score += 0.65 * shop_value_bias + 0.20 * weights[HEAD_BUILD_PROGRESS]
            if semantic_row[family_card_reward] > 0.5 or semantic_row[family_reward] > 0.5:
                score += 0.40 * weights[HEAD_BUILD_PROGRESS] + 0.20 * weights[HEAD_RESOURCE_EFFICIENCY]
            if semantic_row[family_treasure_relic] > 0.5:
                score += 0.35 * weights[HEAD_RESOURCE_EFFICIENCY] + 0.15 * weights[HEAD_BUILD_PROGRESS]
            if semantic_row[family_map] > 0.5:
                # point_type ordinal roughly tracks monster/elite/rest/shop/question/boss buckets
                score += 0.55 * rest_value_bias * float(point_type >= 0.55 and point_type < 0.75)
                score += 0.40 * shop_value_bias * float(point_type >= 0.75 and point_type < 0.90)
                score -= 0.55 * safe_route_bias * float(point_type >= 0.30 and point_type < 0.45)

            bias[index] = float(score)

        scale = max(self.objective_prior_blend * root_bias_scale, 0.0)
        return np.clip(bias * scale, -2.0, 2.0)

    def _select_candidate_indices(
        self,
        priors: np.ndarray,
        *,
        allowed_actions: Optional[List[int]] = None,
        semantic_rows: Optional[np.ndarray] = None,
        preserve_all_if_under_budget: bool = False,
        max_actions: int | None = None,
        min_expanded_actions: int | None = None,
    ) -> List[int]:
        """Select a compact candidate subset for expansion."""
        max_actions = max(int(max_actions if max_actions is not None else self.max_sampled_actions), 1)
        min_expanded_actions = max(int(min_expanded_actions if min_expanded_actions is not None else self.min_expanded_actions), 1)
        if allowed_actions is not None:
            candidates = np.asarray(sorted(set(int(idx) for idx in allowed_actions)), dtype=np.int64)
        else:
            candidates = np.arange(len(priors), dtype=np.int64)

        if candidates.size == 0:
            return []

        if preserve_all_if_under_budget and candidates.size <= max_actions:
            return candidates.tolist()

        candidate_priors = np.asarray(priors[candidates], dtype=np.float32)
        order = np.argsort(-candidate_priors)
        ordered_candidates = candidates[order]
        ordered_priors = candidate_priors[order]

        top_k = min(max_actions, ordered_candidates.size)
        if top_k <= 0:
            return []

        if (not np.isfinite(ordered_priors).all()) or float(ordered_priors.sum()) <= 1e-9:
            return ordered_candidates[:top_k].tolist()

        threshold = self.prior_floor
        if ordered_priors.size > 0 and self.relative_prior_floor > 0.0:
            threshold = max(threshold, float(ordered_priors[0]) * self.relative_prior_floor)

        selected_mask = ordered_priors >= threshold if threshold > 0.0 else np.ones_like(ordered_priors, dtype=bool)
        selected_candidates = ordered_candidates[selected_mask][:top_k]

        min_keep = min(min_expanded_actions, ordered_candidates.size, top_k)
        if selected_candidates.size < min_keep:
            selected_candidates = ordered_candidates[:max(min_keep, selected_candidates.size)]
        if selected_candidates.size == 0:
            selected_candidates = ordered_candidates[:top_k]

        if semantic_rows is not None and semantic_rows.ndim == 2 and semantic_rows.shape[1] >= len(SEMANTIC_ACTION_FAMILIES):
            family_best: list[int] = []
            family_seen: set[int] = set()
            for candidate in ordered_candidates[:top_k]:
                candidate_idx = int(candidate)
                if candidate_idx >= semantic_rows.shape[0]:
                    continue
                family_vector = semantic_rows[candidate_idx, :len(SEMANTIC_ACTION_FAMILIES)]
                family_idx = int(np.argmax(family_vector)) if float(np.sum(family_vector)) > 0 else -1
                if family_idx >= 0 and family_idx not in family_seen:
                    family_seen.add(family_idx)
                    family_best.append(candidate_idx)
            merged: list[int] = []
            for candidate_idx in family_best + selected_candidates.tolist():
                if candidate_idx not in merged:
                    merged.append(candidate_idx)
                if len(merged) >= top_k:
                    break
            selected_candidates = np.asarray(merged, dtype=np.int64)

        return selected_candidates.tolist()

    def _progressive_widen_limit(self, node: Node, *, is_root: bool) -> int:
        total_children = len(node.children)
        if total_children <= 1:
            return total_children
        if is_root and self._active_force_full_root_frontier:
            return total_children
        init = self.root_progressive_widening_init if is_root else self.child_progressive_widening_init
        growth = self.root_progressive_widening_growth if is_root else self.child_progressive_widening_growth
        limit = init + int(growth * math.sqrt(max(float(node.visit_count), 0.0)))
        return max(1, min(total_children, limit))

    def _eligible_child_actions(self, node: Node, *, is_root: bool) -> List[int]:
        if not node.children:
            return []

        limit = self._progressive_widen_limit(node, is_root=is_root)
        if limit >= len(node.children):
            return list(node.children.keys())

        visited_items: list[tuple[int, Node]] = []
        unopened_items: list[tuple[int, Node]] = []
        for action_idx, child in node.children.items():
            if child.visit_count > 0 or child.hidden_state is not None:
                visited_items.append((action_idx, child))
            else:
                unopened_items.append((action_idx, child))

        visited_items.sort(key=lambda item: (item[1].visit_count, item[1].prior), reverse=True)
        unopened_items.sort(key=lambda item: item[1].prior, reverse=True)

        selected: list[int] = [action_idx for action_idx, _ in visited_items]
        remaining = max(limit - len(selected), 0)
        if remaining > 0:
            selected.extend(action_idx for action_idx, _ in unopened_items[:remaining])

        if not selected:
            selected = [action_idx for action_idx, _ in unopened_items[:1]]

        return selected

    @torch.no_grad()
    def run(
        self,
        network,
        obs: Dict,
        action_mask: np.ndarray,
        num_simulations: Optional[int] = None,
        temperature: float = 1.0,
        decision_domain: str | None = None,
    ) -> Tuple[int, np.ndarray]:
        """
        Run MCTS and return action index and action probabilities.

        Args:
            network: MuZero network with initial_inference, recurrent_inference, encode_actions.
            obs: Observation dict with all required fields.
            action_mask: [MAX_ACTIONS] binary mask for legal actions.
            num_simulations: Override default num_simulations.
            temperature: Temperature for action selection (0 = greedy).
            decision_domain: Optional explicit decision domain. When omitted, inferred from obs.

        Returns:
            action_index: Selected action (int in [0, MAX_ACTIONS)).
            action_probs: [MAX_ACTIONS] probability distribution over actions.
        """
        if num_simulations is None:
            num_simulations = self.num_simulations
        mode_info = self._activate_domain_search_mode(
            decision_domain or self._infer_decision_domain(obs, fallback="build")
        )

        legal_actions = np.where(action_mask > 0)[0].tolist()
        action_probs = np.zeros(MAX_ACTIONS, dtype=np.float32)
        if not legal_actions:
            return 0, action_probs
        if len(legal_actions) == 1:
            only_action = int(legal_actions[0])
            action_probs[only_action] = 1.0
            objective_context = self._objective_context_np(obs)
            objective_weights = objective_weight_vector_np(objective_context)
            self.last_run_stats = {
                "num_simulations": float(num_simulations),
                "root_candidates": 1.0,
                "root_selectable_children_mean": 1.0,
                "mean_expanded_children": 0.0,
                "mean_predicted_legal_count": 1.0,
                "mean_surface_keep_count": 1.0,
                "max_search_depth": 0.0,
                "mean_leaf_depth": 0.0,
                "mean_concrete_leaf_depth": 0.0,
                "depth_ge_2_rate": 0.0,
                "depth_ge_3_rate": 0.0,
                "root_top1_visit_share": 1.0,
                "root_visit_entropy": 0.0,
                "objective_weight_survival": float(objective_weights[0]),
                "objective_weight_hp": float(objective_weights[1]),
                "objective_weight_build": float(objective_weights[2]),
                "objective_weight_resource": float(objective_weights[3]),
                "root_bias_scale": float(self._current_root_bias_scale()),
                "semantic_switch_depth": float(self.semantic_switch_depth),
                "semantic_rollout_enabled": 1.0 if self._active_semantic_rollout else 0.0,
                "semantic_expansion_rate": 0.0,
                "semantic_switch_rate": 0.0,
                "semantic_chain_steps_mean": 0.0,
                "semantic_drill_rate": 0.0,
                "combat_grounded_root_enabled": 1.0 if mode_info["grounded_root"] else 0.0,
                "q_value_ucb_enabled": 1.0,
            }
            return only_action, action_probs

        # Prepare obs batch (B=1) on the same device as the network.
        obs_batch = self._obs_to_device_batch(obs, network)

        # Initial inference
        initial = network.initial_inference(obs_batch)
        hidden_state = initial.hidden_state.detach()
        policy_logits = initial.policy_logits.squeeze(0).detach()  # [80]
        objective_context = self._objective_context_np(obs)
        objective_weights = objective_weight_vector_np(objective_context)
        root_value_components = initial.value_components.squeeze(0).detach().cpu().numpy()
        value = self._search_value(
            initial.value.squeeze(0).item(),
            root_value_components,
            objective_context,
        )

        # Create root node
        root = Node(
            hidden_state=hidden_state.squeeze(0).detach(),
            semantic_hidden_state=network.project_to_semantic_latent(hidden_state).squeeze(0).detach(),
            action_embeddings=initial.action_embeddings.squeeze(0).detach(),
            rollout_mode="concrete",
        )

        # Set root priors from policy logits, masking illegal actions.
        # Sampled MuZero: if too many legal actions, sample a root subset.
        action_mask_tensor = torch.as_tensor(
            action_mask,
            dtype=torch.float32,
            device=policy_logits.device,
        )
        end_turn_bias_applied = False
        end_turn_guard_applied = False
        end_turn_guard_forced_alternative = False
        zero_energy_x_cost_guard_applied = False
        zero_energy_x_cost_guard_forced_alternative = False
        root_bias_scale = self._current_root_bias_scale()
        objective_prior_bias = self._objective_prior_bias(obs, action_mask)
        objective_prior_applied = bool(np.abs(objective_prior_bias).max() > 1e-6)
        if objective_prior_applied:
            policy_logits = policy_logits.clone()
            policy_logits += torch.as_tensor(
                objective_prior_bias,
                device=policy_logits.device,
                dtype=policy_logits.dtype,
            )
        dynamic_end_turn_bias = (
            self._end_turn_bias_value(obs, action_mask) * root_bias_scale
            if self.end_turn_prior_bias != 0.0
            else 0.0
        )
        if dynamic_end_turn_bias != 0.0:
            context = self._root_end_turn_context(obs, action_mask)
            end_turn_indices = [int(index) for index in context.get("end_turn_indices", [])]
            if end_turn_indices:
                policy_logits = policy_logits.clone()
                for end_turn_idx in end_turn_indices:
                    policy_logits[end_turn_idx] += dynamic_end_turn_bias
                end_turn_bias_applied = True
        masked_logits = policy_logits - (1.0 - action_mask_tensor) * 1e9
        priors = torch.softmax(masked_logits, dim=0).cpu().numpy()

        root_action_indices = self._select_candidate_indices(
            priors,
            allowed_actions=legal_actions,
            semantic_rows=np.asarray(obs.get("semantic_actions")) if obs.get("semantic_actions") is not None else None,
            preserve_all_if_under_budget=True,
            max_actions=self.root_max_sampled_actions,
            min_expanded_actions=self.root_min_expanded_actions,
        )

        root_prior_values = np.array([priors[action_idx] for action_idx in root_action_indices], dtype=np.float32)
        prior_sum = float(root_prior_values.sum())
        if prior_sum <= 1e-9:
            root_prior_values.fill(1.0 / max(len(root_prior_values), 1))
        else:
            root_prior_values /= prior_sum

        for action_idx, prior in zip(root_action_indices, root_prior_values):
            root.children[int(action_idx)] = Node(prior=float(prior))

        # Add Dirichlet noise to root priors (only on currently sampled legal actions).
        self._add_noise(root)

        # Run simulations
        min_max_stats = MinMaxStats()
        min_max_stats.update(value)
        expansion_child_counts: List[int] = []
        search_depths: List[int] = []
        root_selectable_counts: List[float] = []
        predicted_legal_counts: List[float] = []
        surface_keep_counts: List[float] = []
        semantic_expansion_flags: List[float] = []
        semantic_switch_flags: List[float] = []
        semantic_chain_steps_list: List[float] = []
        semantic_drill_flags: List[float] = []
        concrete_search_depths: List[float] = []

        for _ in range(num_simulations):
            root_selectable_counts.append(float(len(self._eligible_child_actions(root, is_root=True))))
            parent_node, action_idx, search_path, action_path = self._select(root, min_max_stats)
            actual_leaf_depth = max(len(search_path) - 1, 0)

            if action_idx is None:
                value = search_path[-1].value()
                semantic_chain_steps = 0
                semantic_drill = 0.0
                used_semantic_rollout = search_path[-1].rollout_mode == "semantic"
                semantic_switch = False
            else:
                if parent_node.hidden_state is None:
                    raise RuntimeError("MCTS selected an unevaluated parent node without hidden_state.")
                if parent_node.action_embeddings is None:
                    raise RuntimeError("MCTS selected a parent node without action embeddings.")
                expansion_depth = len(action_path)
                used_semantic_rollout = parent_node.rollout_mode == "semantic"
                semantic_switch = (
                    (not used_semantic_rollout)
                    and self._active_semantic_rollout
                    and expansion_depth >= self.semantic_switch_depth
                )
                action_emb = parent_node.action_embeddings[action_idx]  # [64]
                value, value_components, child_count, predicted_legal_count, surface_keep_count, extra_nodes, semantic_chain_steps = self._expand(
                    parent_node,
                    action_idx,
                    parent_node.hidden_state,
                    action_emb,
                    network,
                    objective_context=objective_context,
                    search_depth=expansion_depth,
                )
                if extra_nodes:
                    search_path.extend(extra_nodes)
                actual_leaf_depth = max(len(search_path) - 1, 0)
                semantic_drill = 1.0 if semantic_chain_steps > 0 else 0.0
                expansion_child_counts.append(child_count)
                predicted_legal_counts.append(predicted_legal_count)
                surface_keep_counts.append(surface_keep_count)
            if action_idx is None:
                value_components = search_path[-1].value_components()
            search_depths.append(float(actual_leaf_depth))
            concrete_search_depths.append(float(max(actual_leaf_depth - semantic_chain_steps, 0)))
            semantic_expansion_flags.append(1.0 if (used_semantic_rollout or semantic_chain_steps > 0) else 0.0)
            semantic_switch_flags.append(1.0 if semantic_switch else 0.0)
            semantic_chain_steps_list.append(float(semantic_chain_steps))
            semantic_drill_flags.append(float(semantic_drill))

            # Backup
            self._backup(search_path, value, value_components, min_max_stats)

        # Select action from root
        selection_scores = None
        available_actions = sorted(root.children.keys())
        if available_actions:
            selection_scores = np.array(
                [root.children[action].visit_count for action in available_actions],
                dtype=np.float32,
            )
            selection_scores, end_turn_guard_applied, end_turn_guard_forced_alternative = self._apply_end_turn_selection_guard(
                obs,
                action_mask,
                available_actions,
                selection_scores,
            )
            selection_scores, zero_energy_x_cost_guard_applied, zero_energy_x_cost_guard_forced_alternative = self._apply_zero_energy_x_cost_selection_guard(
                obs,
                action_mask,
                available_actions,
                selection_scores,
            )
        action_idx = self.select_action(root, temperature=temperature, selection_scores=selection_scores)

        # Compute action probabilities
        root_visit_total = sum(child.visit_count for child in root.children.values())
        if root_visit_total > 0:
            for idx, child in root.children.items():
                action_probs[idx] = child.visit_count / root_visit_total
        else:
            prior_total = sum(child.prior for child in root.children.values())
            if prior_total > 0:
                for idx, child in root.children.items():
                    action_probs[idx] = child.prior / prior_total

        root_visit_distribution = np.array(
            [root.children[action].visit_count for action in available_actions],
            dtype=np.float32,
        ) if available_actions else np.zeros(0, dtype=np.float32)
        if float(root_visit_distribution.sum()) > 0.0:
            root_visit_probs = root_visit_distribution / float(root_visit_distribution.sum())
            root_top1_visit_share = float(np.max(root_visit_probs))
            if root_visit_probs.size > 1:
                entropy = -np.sum(root_visit_probs * np.log(np.clip(root_visit_probs, 1e-9, 1.0)))
                root_visit_entropy = float(entropy / math.log(root_visit_probs.size))
            else:
                root_visit_entropy = 0.0
        else:
            root_top1_visit_share = 0.0
            root_visit_entropy = 0.0

        self.last_run_stats = {
            "num_simulations": float(num_simulations),
            "root_candidates": float(len(root.children)),
            "root_selectable_children_mean": float(np.mean(root_selectable_counts)) if root_selectable_counts else 0.0,
            "mean_expanded_children": float(np.mean(expansion_child_counts)) if expansion_child_counts else 0.0,
            "mean_predicted_legal_count": float(np.mean(predicted_legal_counts)) if predicted_legal_counts else 0.0,
            "mean_surface_keep_count": float(np.mean(surface_keep_counts)) if surface_keep_counts else 0.0,
            "max_search_depth": float(max(search_depths)) if search_depths else 0.0,
            "mean_leaf_depth": float(np.mean(search_depths)) if search_depths else 0.0,
            "mean_concrete_leaf_depth": float(np.mean(concrete_search_depths)) if concrete_search_depths else 0.0,
            "depth_ge_2_rate": float(np.mean([depth >= 2 for depth in search_depths])) if search_depths else 0.0,
            "depth_ge_3_rate": float(np.mean([depth >= 3 for depth in search_depths])) if search_depths else 0.0,
            "root_top1_visit_share": float(root_top1_visit_share),
            "root_visit_entropy": float(root_visit_entropy),
            "end_turn_bias_applied": 1.0 if end_turn_bias_applied else 0.0,
            "end_turn_guard_applied": 1.0 if end_turn_guard_applied else 0.0,
            "end_turn_guard_forced_alternative": 1.0 if end_turn_guard_forced_alternative else 0.0,
            "objective_prior_applied": 1.0 if objective_prior_applied else 0.0,
            "objective_weight_survival": float(objective_weights[0]),
            "objective_weight_hp": float(objective_weights[1]),
            "objective_weight_build": float(objective_weights[2]),
            "objective_weight_resource": float(objective_weights[3]),
            "root_objective_value": float(scalarize_objective_components_np(root_value_components, objective_context)),
            "root_bias_scale": float(root_bias_scale),
            "semantic_switch_depth": float(self.semantic_switch_depth),
            "semantic_rollout_enabled": 1.0 if self._active_semantic_rollout else 0.0,
            "semantic_expansion_rate": float(np.mean(semantic_expansion_flags)) if semantic_expansion_flags else 0.0,
            "semantic_switch_rate": float(np.mean(semantic_switch_flags)) if semantic_switch_flags else 0.0,
            "semantic_chain_steps_mean": float(np.mean(semantic_chain_steps_list)) if semantic_chain_steps_list else 0.0,
            "semantic_drill_rate": float(np.mean(semantic_drill_flags)) if semantic_drill_flags else 0.0,
            "combat_grounded_root_enabled": 1.0 if mode_info["grounded_root"] else 0.0,
            "q_value_ucb_enabled": 1.0,
        }

        return int(action_idx), action_probs

    def _ucb_score(
        self, parent: Node, child: Node, min_max_stats: MinMaxStats
    ) -> float:
        """
        Compute UCB score for child node.

        Args:
            parent: Parent node.
            child: Child node.
            min_max_stats: Value normalization stats.

        Returns:
            UCB score (higher is better).
        """
        pb_c = math.log((parent.visit_count + self.c2 + 1) / self.c2) + self.c1
        pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

        prior_score = pb_c * child.prior
        q_value = child.reward + self.discount * child.value()
        value_score = min_max_stats.normalize(q_value)

        semantic_bonus = 0.0
        if self.semantic_revisit_bonus > 0.0 and child.rollout_mode == "semantic":
            if child.hidden_state is not None:
                semantic_bonus = self.semantic_revisit_bonus / math.sqrt(child.visit_count + 1.0)
                if parent.rollout_mode != "semantic":
                    semantic_bonus *= 1.25
            else:
                semantic_bonus = 0.5 * self.semantic_revisit_bonus * child.prior

        return prior_score + value_score + semantic_bonus

    def _select(
        self, root: Node, min_max_stats: MinMaxStats
    ) -> Tuple[Node, Optional[int], List[Node], List[int]]:
        """
        Traverse tree from root using UCB until an unexplored edge is reached.

        Args:
            root: Root node.
            min_max_stats: Value normalization stats.

        Returns:
            parent_node: Parent node of the edge to expand, or final evaluated leaf.
            action_to_expand: Action index for the edge to expand, or None when no expansion is possible.
            search_path: Path from root to selected child placeholder / evaluated leaf (inclusive).
            action_path: Actions taken along path.
        """
        search_path = [root]
        action_path = []
        node = root

        while True:
            if not node.expanded():
                return node, None, search_path, action_path

            eligible_actions = self._eligible_child_actions(node, is_root=(node is root))
            if not eligible_actions:
                eligible_actions = list(node.children.keys())
            eligible_action_set = set(eligible_actions)

            # Pick child with highest UCB among the current widening frontier.
            ucb_scores = {
                action: self._ucb_score(node, child, min_max_stats)
                for action, child in node.children.items()
                if action in eligible_action_set
            }
            action = max(ucb_scores, key=ucb_scores.get)
            child = node.children[action]
            search_path.append(child)
            action_path.append(action)
            if child.hidden_state is None:
                return node, action, search_path, action_path
            node = child

    def _expand(
        self,
        node: Node,
        action: int,
        hidden_state: torch.Tensor,
        action_embedding: torch.Tensor,
        network,
        *,
        objective_context: np.ndarray,
        search_depth: int,
    ) -> tuple[float, np.ndarray, int, float, float, list[Node], int]:
        """
        Expand node by taking action and performing recurrent inference.

        Args:
            node: Node to expand from.
            action: Action index to take.
            hidden_state: Current hidden state [256].
            action_embedding: Embedding for action [64].
            network: MuZero network.

        Returns:
            (
                search value for backup,
                value components,
                number of child candidates expanded,
                raw predicted legal mass,
                top-k keep count,
                extra semantic nodes appended inside this same simulation,
                number of chained semantic plies drilled after entry,
            ).
        """
        # Prepare inputs for recurrent inference (batch size 1)
        hidden_batch = hidden_state.unsqueeze(0)

        # Materialize the selected child in-place so the current search path remains valid.
        child = node.children.get(action)
        if child is None:
            child = Node()
            node.children[action] = child

        if node.rollout_mode == "semantic":
            child, value, value_components, expansion_actions = self._expand_semantic_edge(
                node,
                action,
                hidden_state,
                network,
                objective_context=objective_context,
            )
            predicted_legal_count = float(SEMANTIC_ROLLOUT_SIZE)
            surface_keep_count = float(self.child_max_sampled_actions)
            value, value_components, extra_nodes, semantic_chain_steps = self._continue_semantic_rollout(
                child,
                value,
                value_components,
                network,
                objective_context=objective_context,
                max_steps=self.semantic_rollout_chain_steps,
            )
            return (
                value,
                value_components.astype(np.float32, copy=False),
                expansion_actions,
                predicted_legal_count,
                surface_keep_count,
                extra_nodes,
                semantic_chain_steps,
            )

        action_emb_batch = action_embedding.unsqueeze(0)
        recurrent = network.recurrent_inference(hidden_batch, action_emb_batch)

        next_hidden = recurrent.next_hidden_state.detach().squeeze(0)
        reward_scalar = recurrent.reward.detach().squeeze(0).item()
        policy_logits = recurrent.policy_logits.detach().squeeze(0)
        value_scalar = recurrent.value.detach().squeeze(0).item()
        reward_components = recurrent.reward_components.detach().squeeze(0).cpu().numpy()
        value_components = recurrent.value_components.detach().squeeze(0).cpu().numpy()
        surface_mask_logits = recurrent.next_action_mask_logits.detach().squeeze(0)
        reward = self._search_value(reward_scalar, reward_components, objective_context)
        value = self._search_value(value_scalar, value_components, objective_context)
        surface_probs = torch.sigmoid(surface_mask_logits).cpu().numpy()
        predicted_legal_count = float(surface_probs.sum())

        child_depth = max(int(search_depth), 1)
        switch_to_semantic = self._active_semantic_rollout and child_depth >= self.semantic_switch_depth
        child.reward = reward
        child.reward_components = reward_components.astype(np.float32, copy=False)
        child.children = {}

        if self._active_combat_root_only and child_depth >= 1:
            child.hidden_state = next_hidden
            child.semantic_hidden_state = network.project_to_semantic_latent(
                recurrent.next_hidden_state.detach()
            ).squeeze(0).detach()
            child.rollout_mode = "concrete"
            child.action_embeddings = None
            return (
                value,
                value_components.astype(np.float32, copy=False),
                0,
                predicted_legal_count,
                0.0,
                [],
                0,
            )

        if switch_to_semantic:
            semantic_hidden = network.project_to_semantic_latent(
                recurrent.next_hidden_state.detach()
            ).squeeze(0).detach()
            semantic_prediction = network.semantic_prediction(semantic_hidden.unsqueeze(0))
            semantic_priors = torch.softmax(
                semantic_prediction.semantic_policy_logits.detach().squeeze(0),
                dim=0,
            ).cpu().numpy()
            child.hidden_state = semantic_hidden
            child.semantic_hidden_state = semantic_hidden
            child.rollout_mode = "semantic"
            child.action_embeddings = network.semantic_action_embeddings().detach()
            predicted_legal_count = float(SEMANTIC_ROLLOUT_SIZE)
            surface_keep_count = float(self.child_max_sampled_actions)
            expansion_actions = self._select_candidate_indices(
                semantic_priors,
                max_actions=self.child_max_sampled_actions,
                min_expanded_actions=self.child_min_expanded_actions,
            )
            for action_idx in expansion_actions:
                child.children[action_idx] = Node(prior=float(semantic_priors[action_idx]), rollout_mode="semantic")
            value, value_components, extra_nodes, semantic_chain_steps = self._continue_semantic_rollout(
                child,
                value,
                value_components.astype(np.float32, copy=False),
                network,
                objective_context=objective_context,
                max_steps=self.semantic_rollout_chain_steps,
            )
            return (
                value,
                value_components.astype(np.float32, copy=False),
                len(expansion_actions),
                predicted_legal_count,
                surface_keep_count,
                extra_nodes,
                semantic_chain_steps,
            )

        child.hidden_state = next_hidden
        child.semantic_hidden_state = network.project_to_semantic_latent(
            recurrent.next_hidden_state.detach()
        ).squeeze(0).detach()
        child.rollout_mode = "concrete"
        child.action_embeddings = recurrent.action_embeddings.detach().squeeze(0)

        # Initialize concrete children with policy priors
        priors = torch.softmax(policy_logits, dim=0).cpu().numpy()
        surface_keep_count = int(
            np.clip(
                round(predicted_legal_count),
                self.child_min_expanded_actions,
                self.child_max_sampled_actions,
            )
        )
        top_surface_indices = np.argsort(-surface_probs)[:surface_keep_count]
        surface_gate = np.zeros_like(surface_probs, dtype=np.float32)
        surface_gate[top_surface_indices] = 1.0
        blend = self.surface_mask_blend
        combined_scores = priors * ((1.0 - blend) + blend * surface_gate)
        expansion_actions = self._select_candidate_indices(
            combined_scores,
            max_actions=self.child_max_sampled_actions,
            min_expanded_actions=self.child_min_expanded_actions,
        )
        for action_idx in expansion_actions:
            child.children[action_idx] = Node(prior=priors[action_idx], rollout_mode="concrete")

        return (
            value,
            value_components.astype(np.float32, copy=False),
            len(expansion_actions),
            predicted_legal_count,
            float(surface_keep_count),
            [],
            0,
        )

    def _expand_semantic_edge(
        self,
        node: Node,
        action: int,
        semantic_hidden_state: torch.Tensor,
        network,
        *,
        objective_context: np.ndarray,
    ) -> tuple[Node, float, np.ndarray, int]:
        hidden_batch = semantic_hidden_state.unsqueeze(0)
        child = node.children.get(action)
        if child is None:
            child = Node(rollout_mode="semantic")
            node.children[action] = child

        semantic_action_index = torch.as_tensor([int(action)], device=hidden_batch.device)
        semantic_recurrent = network.semantic_recurrent_inference(
            hidden_batch,
            semantic_action_index,
        )
        next_semantic_hidden = semantic_recurrent.next_semantic_hidden_state.detach().squeeze(0)
        reward_scalar = semantic_recurrent.reward.detach().squeeze(0).item()
        value_scalar = semantic_recurrent.value.detach().squeeze(0).item()
        reward_components = semantic_recurrent.reward_components.detach().squeeze(0).cpu().numpy()
        value_components = semantic_recurrent.value_components.detach().squeeze(0).cpu().numpy()
        reward = self._search_value(reward_scalar, reward_components, objective_context)
        value = self._search_value(value_scalar, value_components, objective_context)

        child.hidden_state = next_semantic_hidden
        child.semantic_hidden_state = next_semantic_hidden
        child.rollout_mode = "semantic"
        child.reward = reward
        child.reward_components = reward_components.astype(np.float32, copy=False)
        child.action_embeddings = network.semantic_action_embeddings().detach()
        child.children = {}

        semantic_priors = torch.softmax(
            semantic_recurrent.semantic_policy_logits.detach().squeeze(0),
            dim=0,
        ).cpu().numpy()
        expansion_actions = self._select_candidate_indices(
            semantic_priors,
            max_actions=self.child_max_sampled_actions,
            min_expanded_actions=self.child_min_expanded_actions,
        )
        for action_idx in expansion_actions:
            child.children[action_idx] = Node(prior=float(semantic_priors[action_idx]), rollout_mode="semantic")

        return (
            child,
            value,
            value_components.astype(np.float32, copy=False),
            len(expansion_actions),
        )

    def _select_semantic_chain_action(self, node: Node) -> Optional[int]:
        if not node.children:
            return None

        best_action: Optional[int] = None
        best_score = float("-inf")
        for action_idx, child in node.children.items():
            unexplored_bonus = 1.25 if child.hidden_state is None else 1.0
            visit_scale = math.sqrt(child.visit_count + 1.0)
            score = float(child.prior) * unexplored_bonus / visit_scale
            if score > best_score:
                best_score = score
                best_action = int(action_idx)
        return best_action

    def _continue_semantic_rollout(
        self,
        leaf_node: Node,
        value: float,
        value_components: np.ndarray,
        network,
        *,
        objective_context: np.ndarray,
        max_steps: int,
    ) -> tuple[float, np.ndarray, list[Node], int]:
        if max_steps <= 0:
            return value, value_components, [], 0

        current_node = leaf_node
        current_value = float(value)
        current_components = np.asarray(value_components, dtype=np.float32)
        extra_nodes: list[Node] = []
        semantic_chain_steps = 0

        for _ in range(max_steps):
            if current_node.rollout_mode != "semantic" or current_node.hidden_state is None or not current_node.children:
                break
            next_action = self._select_semantic_chain_action(current_node)
            if next_action is None:
                break
            expanded_child, current_value, current_components, _ = self._expand_semantic_edge(
                current_node,
                next_action,
                current_node.hidden_state,
                network,
                objective_context=objective_context,
            )
            extra_nodes.append(expanded_child)
            current_node = expanded_child
            semantic_chain_steps += 1

        return current_value, current_components.astype(np.float32, copy=False), extra_nodes, semantic_chain_steps

    def _backup(
        self,
        search_path: List[Node],
        value: float,
        value_components: np.ndarray,
        min_max_stats: MinMaxStats,
    ) -> None:
        """
        Backup value through search path, applying discount and updating stats.

        Args:
            search_path: Path from root to leaf.
            value: Value to backup.
            value_components: Objective value components aligned with ``value``.
            min_max_stats: Stats to update.
        """
        for node in reversed(search_path):
            node.value_sum += value
            node.value_component_sum += value_components
            node.visit_count += 1
            q_value = node.reward + self.discount * node.value()
            min_max_stats.update(q_value)
            value = node.reward + self.discount * value
            value_components = node.reward_components + self.discount * value_components

    def _add_noise(self, root: Node) -> None:
        """
        Add Dirichlet noise to root priors for exploration.

        Args:
            root: Root node.
        """
        legal_actions = list(root.children.keys())

        if not legal_actions:
            return

        noise = np.random.dirichlet(
            [self.dirichlet_alpha] * len(legal_actions)
        )

        for i, action in enumerate(legal_actions):
            prior = root.children[action].prior
            root.children[action].prior = (
                (1.0 - self.exploration_fraction) * prior
                + self.exploration_fraction * noise[i]
            )

    def select_action(
        self,
        root: Node,
        temperature: float = 1.0,
        selection_scores: np.ndarray | None = None,
    ) -> int:
        """
        Select action from root using visit counts.

        Args:
            root: Root node after MCTS.
            temperature: Softmax temperature (0 = greedy).

        Returns:
            Selected action index.
        """
        available_actions = sorted(root.children.keys())
        if not available_actions:
            return 0

        if selection_scores is None:
            visit_counts = np.array(
                [root.children[action].visit_count for action in available_actions],
                dtype=np.float32,
            )
        else:
            visit_counts = np.asarray(selection_scores, dtype=np.float32).reshape(-1)
            if visit_counts.shape[0] != len(available_actions):
                visit_counts = np.array(
                    [root.children[action].visit_count for action in available_actions],
                    dtype=np.float32,
                )

        if visit_counts.sum() <= 0:
            priors = np.array(
                [root.children[action].prior for action in available_actions],
                dtype=np.float32,
            )
            if priors.sum() <= 0:
                return int(available_actions[0])
            priors /= priors.sum()
            return int(np.random.choice(available_actions, p=priors))

        if temperature == 0:
            # Greedy: argmax visit count
            return int(available_actions[int(np.argmax(visit_counts))])
        else:
            # Sample from visit_count^(1/temperature)
            scaled_counts = visit_counts ** (1.0 / temperature)
            probs = scaled_counts / (scaled_counts.sum() + 1e-9)
            return int(np.random.choice(available_actions, p=probs))
