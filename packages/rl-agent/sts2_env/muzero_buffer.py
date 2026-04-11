"""Replay buffer and trajectory storage for MuZero training.

This module implements:
- GameTrajectory: stores one episode with observation, actions, rewards, and targets
- MuZeroReplayBuffer: prioritized sampling buffer for training the MuZero network
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
import torch

from .objective_heads import (
    NUM_OBJECTIVE_HEADS,
    scalarize_objective_components_np,
)
from .semantic_rollout import SEMANTIC_ROLLOUT_SIZE, aggregate_concrete_policy_to_semantic


@dataclass
class GameTrajectory:
    """Stores a single episode trajectory for MuZero training.

    Each step records the observation, action taken, reward received, action mask,
    search policy from MCTS, and root value estimate. Targets are computed on demand.
    """

    steps: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    """List of transition dicts with keys:
    - obs: dict observation
    - action: int action index [0, MAX_ACTIONS)
    - reward: float immediate reward
    - reward_components: np.ndarray [NUM_OBJECTIVE_HEADS]
    - action_mask: np.ndarray [MAX_ACTIONS] binary mask of legal actions
    - search_policy: np.ndarray [MAX_ACTIONS] action probs from MCTS
    - root_value: float value estimate at this step
    - root_value_components: np.ndarray [NUM_OBJECTIVE_HEADS]
    - semantic_action: int semantic rollout action index
    - semantic_policy: np.ndarray [SEMANTIC_ROLLOUT_SIZE] aggregated semantic policy
    - action_info: compact logical action signature for postmortem analysis
    - search_stats: compact MCTS summary for this decision
    """

    def __len__(self) -> int:
        """Return number of steps in trajectory."""
        return len(self.steps)

    def add_step(
        self,
        obs: Dict[str, Any],
        action: int,
        reward: float,
        reward_components: np.ndarray | list[float] | None,
        action_mask: np.ndarray,
        search_policy: np.ndarray,
        root_value: float,
        root_value_components: np.ndarray | list[float] | None = None,
        objective_context: np.ndarray | list[float] | None = None,
        semantic_action: int | None = None,
        semantic_policy: np.ndarray | list[float] | None = None,
        action_info: Dict[str, Any] | None = None,
        search_stats: Dict[str, Any] | None = None,
    ) -> None:
        """Add a step to the trajectory.

        Args:
            obs: Observation dict.
            action: Action index.
            reward: Immediate reward.
            reward_components: Objective-decomposed immediate reward.
            action_mask: [MAX_ACTIONS] binary mask.
            search_policy: [MAX_ACTIONS] MCTS policy.
            root_value: Value estimate from root of MCTS tree.
            root_value_components: Objective-decomposed root value estimate.
            objective_context: Planner objective-context vector at this decision.
            semantic_action: Chosen semantic rollout action index.
            semantic_policy: Aggregated semantic policy target.
            action_info: Compact action signature for debugging / replay analysis.
            search_stats: Compact search summary for this step.
        """
        reward_components_arr = np.zeros(NUM_OBJECTIVE_HEADS, dtype=np.float32)
        if reward_components is not None:
            reward_components_arr = np.asarray(reward_components, dtype=np.float32).reshape(-1)
            if reward_components_arr.shape[0] != NUM_OBJECTIVE_HEADS:
                reward_components_arr = np.resize(reward_components_arr, NUM_OBJECTIVE_HEADS).astype(np.float32, copy=False)

        root_value_components_arr = np.zeros(NUM_OBJECTIVE_HEADS, dtype=np.float32)
        if root_value_components is not None:
            root_value_components_arr = np.asarray(root_value_components, dtype=np.float32).reshape(-1)
            if root_value_components_arr.shape[0] != NUM_OBJECTIVE_HEADS:
                root_value_components_arr = np.resize(root_value_components_arr, NUM_OBJECTIVE_HEADS).astype(np.float32, copy=False)

        objective_context_arr = np.asarray(objective_context, dtype=np.float32).reshape(-1) if objective_context is not None else np.zeros(0, dtype=np.float32)
        semantic_policy_arr = np.zeros(SEMANTIC_ROLLOUT_SIZE, dtype=np.float32)
        if semantic_policy is not None:
            semantic_policy_arr = np.asarray(semantic_policy, dtype=np.float32).reshape(-1)
            if semantic_policy_arr.shape[0] != SEMANTIC_ROLLOUT_SIZE:
                semantic_policy_arr = np.resize(semantic_policy_arr, SEMANTIC_ROLLOUT_SIZE).astype(np.float32, copy=False)
        self.steps.append({
            "obs": obs,
            "action": int(action),
            "reward": float(reward),
            "reward_components": reward_components_arr.astype(np.float32, copy=False),
            "action_mask": np.array(action_mask, dtype=np.float32),
            "search_policy": np.array(search_policy, dtype=np.float32),
            "root_value": float(root_value),
            "root_value_components": root_value_components_arr.astype(np.float32, copy=False),
            "objective_context": objective_context_arr.astype(np.float32, copy=False),
            "semantic_action": int(semantic_action or 0),
            "semantic_policy": semantic_policy_arr.astype(np.float32, copy=False),
            "action_info": dict(action_info or {}),
            "search_stats": dict(search_stats or {}),
        })

    def compute_target_values(
        self,
        discount: float = 0.997,
        n_steps: int = 10,
    ) -> List[float]:
        """Compute n-step bootstrapped returns for each step.

        For step t, the n-step return is:
            G_t = sum_{k=0}^{n-1} gamma^k * r_{t+k} + gamma^n * V_{t+n}

        Where V_t is the root_value at step t. Past episode end, value is treated as 0.

        Args:
            discount: Discount factor gamma.
            n_steps: Number of steps to bootstrap.

        Returns:
            List of n-step target values, one per step.
        """
        targets = []
        ep_len = len(self.steps)

        for step_idx in range(ep_len):
            target = 0.0
            discounted = 1.0

            # Sum discounted rewards over n steps
            for k in range(n_steps):
                if step_idx + k < ep_len:
                    target += discounted * self.steps[step_idx + k]["reward"]
                    discounted *= discount
                else:
                    break

            # Bootstrap value at step_idx + n_steps
            if step_idx + n_steps < ep_len:
                bootstrap_value = self.steps[step_idx + n_steps]["root_value"]
                target += discounted * bootstrap_value

            targets.append(target)

        return targets

    def compute_target_value_components(
        self,
        discount: float = 0.997,
        n_steps: int = 10,
    ) -> List[np.ndarray]:
        """Compute n-step bootstrapped targets for each objective head."""
        targets: List[np.ndarray] = []
        ep_len = len(self.steps)

        for step_idx in range(ep_len):
            target = np.zeros(NUM_OBJECTIVE_HEADS, dtype=np.float32)
            discounted = 1.0

            for k in range(n_steps):
                if step_idx + k < ep_len:
                    target += discounted * np.asarray(
                        self.steps[step_idx + k].get("reward_components", np.zeros(NUM_OBJECTIVE_HEADS, dtype=np.float32)),
                        dtype=np.float32,
                    )
                    discounted *= discount
                else:
                    break

            if step_idx + n_steps < ep_len:
                bootstrap_value = np.asarray(
                    self.steps[step_idx + n_steps].get("root_value_components", np.zeros(NUM_OBJECTIVE_HEADS, dtype=np.float32)),
                    dtype=np.float32,
                )
                target += discounted * bootstrap_value

            targets.append(target.astype(np.float32, copy=False))

        return targets


class MuZeroReplayBuffer:
    """Prioritized replay buffer for MuZero training.

    Stores complete game trajectories and samples mini-batches for training.
    Prioritizes sampling based on absolute difference between predicted and
    bootstrapped values.
    """

    def __init__(self, capacity: int = 100_000):
        """Initialize replay buffer.

        Args:
            capacity: Maximum total transitions to store.
        """
        self.capacity = max(int(capacity), 1)
        self.trajectories: List[GameTrajectory] = []
        self.priorities: List[float] = []
        self._total_transitions = 0

    def save_episode(
        self,
        trajectory: GameTrajectory,
        *,
        discount: float = 0.997,
        n_steps: int = 10,
    ) -> None:
        """Save a completed episode trajectory to the buffer.

        If buffer exceeds capacity (by number of transitions), remove oldest
        trajectories until within capacity.

        Args:
            trajectory: Completed GameTrajectory with all steps.
        """
        if len(trajectory) == 0:
            return

        scalar_targets = trajectory.compute_target_values(discount=discount, n_steps=n_steps)
        component_targets = trajectory.compute_target_value_components(discount=discount, n_steps=n_steps)

        # Compute priorities: use deviation between predicted root value(s) and bootstrapped target(s)
        ep_priority = 0.0
        for step_index, step in enumerate(trajectory.steps):
            scalar_target = scalar_targets[step_index] if step_index < len(scalar_targets) else 0.0
            component_target = component_targets[step_index] if step_index < len(component_targets) else np.zeros(NUM_OBJECTIVE_HEADS, dtype=np.float32)
            component_pred = np.asarray(step.get("root_value_components", np.zeros(NUM_OBJECTIVE_HEADS, dtype=np.float32)), dtype=np.float32)
            objective_context = step.get("objective_context")
            scalarized_component_target = float(scalarize_objective_components_np(component_target, objective_context))
            scalarized_component_pred = float(scalarize_objective_components_np(component_pred, objective_context))
            delta = abs(float(step["root_value"]) - float(scalar_target))
            delta += 0.5 * abs(scalarized_component_pred - scalarized_component_target)
            ep_priority += delta
        ep_priority /= max(len(trajectory), 1)
        ep_priority += 1e-6  # Small epsilon to avoid zero priority

        self.trajectories.append(trajectory)
        self.priorities.append(ep_priority)
        self._total_transitions += len(trajectory)

        # Evict oldest trajectories if over capacity
        while self._total_transitions > self.capacity and len(self.trajectories) > 0:
            removed_len = len(self.trajectories[0])
            self.trajectories.pop(0)
            self.priorities.pop(0)
            self._total_transitions -= removed_len

    @staticmethod
    def _zero_obs_like(obs: Dict[str, Any]) -> Dict[str, Any]:
        """Create a zero observation with the same schema/shapes as `obs`."""
        zero_obs: Dict[str, Any] = {}
        for key, value in obs.items():
            if isinstance(value, np.ndarray):
                zero_obs[key] = np.zeros_like(value)
            elif isinstance(value, torch.Tensor):
                zero_obs[key] = torch.zeros_like(value)
            else:
                zero_obs[key] = 0
        return zero_obs

    def sample_batch(
        self,
        batch_size: int,
        unroll_steps: int = 5,
        rng: np.random.Generator | None = None,
        discount: float = 0.997,
        n_step_return: int = 10,
    ) -> Dict[str, Any]:
        """Sample a mini-batch for training.

        Samples batch_size (trajectory, position) pairs using prioritized sampling.
        For each sample, unrolls K steps forward for computing unrolled losses.

        Returns tensors with batch dimension B = batch_size:
        - obs_batch: list of B observation dicts
        - action_batch: [B, K] int actions for unroll
        - reward_target: [B, K] float rewards
        - reward_component_target: [B, K, NUM_OBJECTIVE_HEADS] decomposed rewards
        - value_target: [B, K+1] float bootstrapped values
        - value_component_target: [B, K+1, NUM_OBJECTIVE_HEADS] bootstrapped decomposed values
        - policy_target: [B, K+1, 80] MCTS policy distributions
        - semantic_action_batch: [B, K] semantic rollout action ids
        - semantic_policy_target: [B, K+1, SEMANTIC_ROLLOUT_SIZE] aggregated semantic policy distributions
        - action_mask_batch: [B, K+1, 80] legal action masks

        Past episode end, all targets are zero-padded.

        Args:
            batch_size: Number of samples.
            unroll_steps: Number of steps to unroll.
            rng: Random generator for sampling.

        Returns:
            Dict with tensors for training.
        """
        if len(self.trajectories) == 0:
            raise RuntimeError("Cannot sample from empty replay buffer")

        if rng is None:
            rng = np.random.default_rng()

        # Prioritized sampling: sample trajectory, then random position
        priorities_array = np.array(self.priorities, dtype=np.float32)
        priorities_array /= priorities_array.sum()

        batch_obs = []
        batch_obs_sequence = []
        batch_actions = []
        batch_semantic_actions = []
        batch_rewards = []
        batch_reward_components = []
        batch_values = []
        batch_value_components = []
        batch_policies = []
        batch_semantic_policies = []
        batch_masks = []

        for _ in range(batch_size):
            # Sample trajectory by priority
            traj_idx = int(rng.choice(len(self.trajectories), p=priorities_array))
            trajectory = self.trajectories[traj_idx]
            scalar_targets = trajectory.compute_target_values(discount=discount, n_steps=n_step_return)
            component_targets = trajectory.compute_target_value_components(discount=discount, n_steps=n_step_return)

            # Sample random position in trajectory
            pos = int(rng.integers(len(trajectory)))

            # Collect observation at position
            batch_obs.append(trajectory.steps[pos]["obs"])
            zero_obs = self._zero_obs_like(trajectory.steps[-1]["obs"])
            obs_sequence = []
            for k in range(unroll_steps + 1):
                step_idx = pos + k
                if step_idx < len(trajectory):
                    obs_sequence.append(trajectory.steps[step_idx]["obs"])
                else:
                    obs_sequence.append(zero_obs)
            batch_obs_sequence.append(obs_sequence)

            # Collect actions, rewards, values, policies for unroll
            actions = []
            semantic_actions = []
            rewards = []
            reward_components = []
            values = [scalar_targets[pos] if pos < len(scalar_targets) else trajectory.steps[pos]["root_value"]]
            value_components = [
                component_targets[pos] if pos < len(component_targets)
                else np.asarray(trajectory.steps[pos].get("root_value_components", np.zeros(NUM_OBJECTIVE_HEADS, dtype=np.float32)), dtype=np.float32)
            ]
            policies = [trajectory.steps[pos]["search_policy"]]
            semantic_policies = [
                np.asarray(
                    trajectory.steps[pos].get("semantic_policy", np.zeros(SEMANTIC_ROLLOUT_SIZE, dtype=np.float32)),
                    dtype=np.float32,
                )
            ]
            masks = [trajectory.steps[pos]["action_mask"]]

            for k in range(unroll_steps):
                step_idx = pos + k
                if step_idx < len(trajectory):
                    step = trajectory.steps[step_idx]
                    actions.append(step["action"])
                    semantic_actions.append(int(step.get("semantic_action", 0)))
                    rewards.append(step["reward"])
                    reward_components.append(
                        np.asarray(
                            step.get("reward_components", np.zeros(NUM_OBJECTIVE_HEADS, dtype=np.float32)),
                            dtype=np.float32,
                        )
                    )
                    if step_idx + 1 < len(trajectory):
                        next_target_idx = step_idx + 1
                        values.append(
                            scalar_targets[next_target_idx]
                            if next_target_idx < len(scalar_targets)
                            else trajectory.steps[step_idx + 1]["root_value"]
                        )
                        value_components.append(
                            component_targets[next_target_idx]
                            if next_target_idx < len(component_targets)
                            else np.asarray(
                                trajectory.steps[step_idx + 1].get(
                                    "root_value_components",
                                    np.zeros(NUM_OBJECTIVE_HEADS, dtype=np.float32),
                                ),
                                dtype=np.float32,
                            )
                        )
                        policies.append(trajectory.steps[step_idx + 1]["search_policy"])
                        semantic_policies.append(
                            np.asarray(
                                trajectory.steps[step_idx + 1].get(
                                    "semantic_policy",
                                    np.zeros(SEMANTIC_ROLLOUT_SIZE, dtype=np.float32),
                                ),
                                dtype=np.float32,
                            )
                        )
                        masks.append(trajectory.steps[step_idx + 1]["action_mask"])
                    else:
                        # Terminal state
                        values.append(0.0)
                        value_components.append(np.zeros(NUM_OBJECTIVE_HEADS, dtype=np.float32))
                        policies.append(np.zeros(len(step["search_policy"]), dtype=np.float32))
                        semantic_policies.append(np.zeros(SEMANTIC_ROLLOUT_SIZE, dtype=np.float32))
                        masks.append(np.zeros(len(step["action_mask"]), dtype=np.float32))
                else:
                    # Past episode end
                    actions.append(0)
                    semantic_actions.append(0)
                    rewards.append(0.0)
                    reward_components.append(np.zeros(NUM_OBJECTIVE_HEADS, dtype=np.float32))
                    values.append(0.0)
                    value_components.append(np.zeros(NUM_OBJECTIVE_HEADS, dtype=np.float32))
                    policies.append(np.zeros(len(trajectory.steps[-1]["search_policy"]), dtype=np.float32))
                    semantic_policies.append(np.zeros(SEMANTIC_ROLLOUT_SIZE, dtype=np.float32))
                    masks.append(np.zeros(len(trajectory.steps[-1]["action_mask"]), dtype=np.float32))

            batch_actions.append(actions)
            batch_semantic_actions.append(semantic_actions)
            batch_rewards.append(rewards)
            batch_reward_components.append(reward_components)
            batch_values.append(values)
            batch_value_components.append(value_components)
            batch_policies.append(policies)
            batch_semantic_policies.append(semantic_policies)
            batch_masks.append(masks)

        # Convert dense numeric targets through numpy first to avoid slow tensor construction
        action_batch = torch.from_numpy(np.asarray(batch_actions, dtype=np.int64))  # [B, K]
        semantic_action_batch = torch.from_numpy(np.asarray(batch_semantic_actions, dtype=np.int64))  # [B, K]
        reward_batch = torch.from_numpy(np.asarray(batch_rewards, dtype=np.float32))  # [B, K]
        reward_component_batch = torch.from_numpy(np.asarray(batch_reward_components, dtype=np.float32))  # [B, K, H]
        value_batch = torch.from_numpy(np.asarray(batch_values, dtype=np.float32))  # [B, K+1]
        value_component_batch = torch.from_numpy(np.asarray(batch_value_components, dtype=np.float32))  # [B, K+1, H]
        policy_batch = torch.from_numpy(np.asarray(batch_policies, dtype=np.float32))  # [B, K+1, 80]
        semantic_policy_batch = torch.from_numpy(np.asarray(batch_semantic_policies, dtype=np.float32))  # [B, K+1, S]
        mask_batch = torch.from_numpy(np.asarray(batch_masks, dtype=np.float32))  # [B, K+1, 80]

        return {
            "obs_batch": batch_obs,
            "obs_sequence_batch": batch_obs_sequence,
            "action_batch": action_batch,
            "semantic_action_batch": semantic_action_batch,
            "reward_target": reward_batch,
            "reward_component_target": reward_component_batch,
            "value_target": value_batch,
            "value_component_target": value_component_batch,
            "policy_target": policy_batch,
            "semantic_policy_target": semantic_policy_batch,
            "action_mask_batch": mask_batch,
        }

    def update_priorities(self, indices: List[int], new_priorities: List[float]) -> None:
        """Update priorities for sampled trajectories.

        Args:
            indices: Trajectory indices.
            new_priorities: New priority values.
        """
        for idx, priority in zip(indices, new_priorities):
            if 0 <= idx < len(self.priorities):
                self.priorities[idx] = max(float(priority), 1e-6)

    def state_dict(self) -> Dict[str, Any]:
        """Serialize replay buffer contents for checkpointing."""
        return {
            "capacity": int(self.capacity),
            "trajectories": self.trajectories,
            "priorities": list(self.priorities),
            "total_transitions": int(self._total_transitions),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore replay buffer contents from a serialized state."""
        self.capacity = max(int(state.get("capacity", self.capacity)), 1)
        self.trajectories = list(state.get("trajectories") or [])
        self.priorities = [max(float(priority), 1e-6) for priority in (state.get("priorities") or [])]
        if self.priorities and len(self.priorities) != len(self.trajectories):
            raise ValueError("Replay buffer priority count does not match trajectory count.")
        if "total_transitions" in state:
            self._total_transitions = int(state["total_transitions"])
        else:
            self._total_transitions = sum(len(trajectory) for trajectory in self.trajectories)

        # Older checkpoints may predate episode metadata / per-step action traces.
        # Backfill them here so downstream analysis code can rely on the keys.
        for trajectory in self.trajectories:
            if not hasattr(trajectory, "metadata") or trajectory.metadata is None:
                trajectory.metadata = {}
            for step in getattr(trajectory, "steps", []):
                if "reward_components" not in step or step["reward_components"] is None:
                    step["reward_components"] = np.zeros(NUM_OBJECTIVE_HEADS, dtype=np.float32)
                if "root_value_components" not in step or step["root_value_components"] is None:
                    step["root_value_components"] = np.zeros(NUM_OBJECTIVE_HEADS, dtype=np.float32)
                if "objective_context" not in step or step["objective_context"] is None:
                    step["objective_context"] = np.zeros(0, dtype=np.float32)
                if "semantic_action" not in step or step["semantic_action"] is None:
                    step["semantic_action"] = 0
                if "semantic_policy" not in step or step["semantic_policy"] is None:
                    step["semantic_policy"] = np.zeros(SEMANTIC_ROLLOUT_SIZE, dtype=np.float32)
                if "action_info" not in step or step["action_info"] is None:
                    step["action_info"] = {}
                if "search_stats" not in step or step["search_stats"] is None:
                    step["search_stats"] = {}

    def __len__(self) -> int:
        """Return total number of transitions stored."""
        return self._total_transitions
