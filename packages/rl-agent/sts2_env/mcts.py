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
from typing import Optional, Tuple, Dict, List

from .observation_v2 import MAX_ACTIONS


@dataclass
class Node:
    """Tree node for MCTS."""

    visit_count: int = 0
    value_sum: float = 0.0
    prior: float = 0.0
    reward: float = 0.0
    hidden_state: Optional[torch.Tensor] = None
    children: Dict[int, "Node"] = field(default_factory=dict)

    def expanded(self) -> bool:
        """Check if node has been expanded (has children)."""
        return len(self.children) > 0

    def value(self) -> float:
        """Get average value from visit count."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


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
        """
        self.num_simulations = num_simulations
        self.discount = discount
        self.c1 = c1
        self.c2 = c2
        self.dirichlet_alpha = dirichlet_alpha
        self.exploration_fraction = exploration_fraction
        self.max_sampled_actions = max_sampled_actions

    @torch.no_grad()
    def run(
        self,
        network,
        obs: Dict,
        action_mask: np.ndarray,
        num_simulations: Optional[int] = None,
        temperature: float = 1.0,
    ) -> Tuple[int, np.ndarray]:
        """
        Run MCTS and return action index and action probabilities.

        Args:
            network: MuZero network with initial_inference, recurrent_inference, encode_actions.
            obs: Observation dict with all required fields.
            action_mask: [MAX_ACTIONS] binary mask for legal actions.
            num_simulations: Override default num_simulations.
            temperature: Temperature for action selection (0 = greedy).

        Returns:
            action_index: Selected action (int in [0, MAX_ACTIONS)).
            action_probs: [MAX_ACTIONS] probability distribution over actions.
        """
        if num_simulations is None:
            num_simulations = self.num_simulations

        legal_actions = np.where(action_mask > 0)[0].tolist()
        action_probs = np.zeros(MAX_ACTIONS, dtype=np.float32)
        if not legal_actions:
            return 0, action_probs
        if len(legal_actions) == 1:
            only_action = int(legal_actions[0])
            action_probs[only_action] = 1.0
            return only_action, action_probs

        # Prepare obs batch (B=1)
        obs_batch = {k: torch.as_tensor(v).unsqueeze(0) if not isinstance(v, torch.Tensor)
                     else v.unsqueeze(0) if v.dim() > 0 else v.unsqueeze(0)
                     for k, v in obs.items()}

        # Initial inference
        initial = network.initial_inference(obs_batch)
        hidden_state = initial.hidden_state.detach()
        policy_logits = initial.policy_logits.squeeze(0).detach()  # [80]
        value = initial.value.squeeze(0).item()  # scalar

        # Encode actions
        action_embeddings = network.encode_actions(obs_batch)  # [1, 80, 64]
        action_embeddings = action_embeddings.detach().squeeze(0)  # [80, 64]

        # Create root node
        root = Node(hidden_state=hidden_state.squeeze(0).detach())

        # Set root priors from policy logits, masking illegal actions.
        # Sampled MuZero: if too many legal actions, sample a root subset.
        action_mask_tensor = torch.as_tensor(action_mask, dtype=torch.float32)
        masked_logits = policy_logits - (1.0 - action_mask_tensor) * 1e9
        priors = torch.softmax(masked_logits, dim=0).cpu().numpy()

        if len(legal_actions) > self.max_sampled_actions:
            root_action_indices = np.random.choice(
                legal_actions, size=self.max_sampled_actions, replace=False
            ).tolist()
        else:
            root_action_indices = legal_actions

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

        for _ in range(num_simulations):
            parent_node, action_idx, search_path, action_path = self._select(root, min_max_stats)

            if action_idx is None:
                value = search_path[-1].value()
            else:
                if parent_node.hidden_state is None:
                    raise RuntimeError("MCTS selected an unevaluated parent node without hidden_state.")
                action_emb = action_embeddings[action_idx]  # [64]
                value = self._expand(
                    parent_node,
                    action_idx,
                    parent_node.hidden_state,
                    action_emb,
                    network,
                )

            # Backup
            self._backup(search_path, value, min_max_stats)

        # Select action from root
        action_idx = self.select_action(root, temperature=temperature)

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
        value_score = min_max_stats.normalize(child.value())

        return prior_score + value_score

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

            # Pick child with highest UCB
            ucb_scores = {
                action: self._ucb_score(node, child, min_max_stats)
                for action, child in node.children.items()
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
    ) -> float:
        """
        Expand node by taking action and performing recurrent inference.

        Args:
            node: Node to expand from.
            action: Action index to take.
            hidden_state: Current hidden state [256].
            action_embedding: Embedding for action [64].
            network: MuZero network.

        Returns:
            value: Value estimate for backup.
        """
        # Prepare inputs for recurrent inference (batch size 1)
        hidden_batch = hidden_state.unsqueeze(0)  # [1, 256]
        action_emb_batch = action_embedding.unsqueeze(0)  # [1, 64]

        # Recurrent inference
        recurrent = network.recurrent_inference(hidden_batch, action_emb_batch)

        next_hidden = recurrent.next_hidden_state.detach().squeeze(0)  # [256]
        reward = recurrent.reward.detach().squeeze(0).item()  # scalar
        policy_logits = recurrent.policy_logits.detach().squeeze(0)  # [80]
        value = recurrent.value.detach().squeeze(0).item()  # scalar

        # Materialize the selected child in-place so the current search path remains valid.
        child = node.children.get(action)
        if child is None:
            child = Node()
            node.children[action] = child
        child.hidden_state = next_hidden
        child.reward = reward
        child.children = {}

        # Initialize children with policy priors
        priors = torch.softmax(policy_logits, dim=0).cpu().numpy()
        for action_idx in range(MAX_ACTIONS):
            child.children[action_idx] = Node(prior=priors[action_idx])

        return value

    def _backup(
        self,
        search_path: List[Node],
        value: float,
        min_max_stats: MinMaxStats,
    ) -> None:
        """
        Backup value through search path, applying discount and updating stats.

        Args:
            search_path: Path from root to leaf.
            value: Value to backup.
            min_max_stats: Stats to update.
        """
        for node in reversed(search_path):
            node.value_sum += value
            node.visit_count += 1
            min_max_stats.update(node.value())
            value = node.reward + self.discount * value

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

    def select_action(self, root: Node, temperature: float = 1.0) -> int:
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
