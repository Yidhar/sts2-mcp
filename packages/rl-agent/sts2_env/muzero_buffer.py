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


@dataclass
class GameTrajectory:
    """Stores a single episode trajectory for MuZero training.

    Each step records the observation, action taken, reward received, action mask,
    search policy from MCTS, and root value estimate. Targets are computed on demand.
    """

    steps: List[Dict[str, Any]] = field(default_factory=list)
    """List of transition dicts with keys:
    - obs: dict observation
    - action: int action index [0, MAX_ACTIONS)
    - reward: float immediate reward
    - action_mask: np.ndarray [MAX_ACTIONS] binary mask of legal actions
    - search_policy: np.ndarray [MAX_ACTIONS] action probs from MCTS
    - root_value: float value estimate at this step
    """

    def __len__(self) -> int:
        """Return number of steps in trajectory."""
        return len(self.steps)

    def add_step(
        self,
        obs: Dict[str, Any],
        action: int,
        reward: float,
        action_mask: np.ndarray,
        search_policy: np.ndarray,
        root_value: float,
    ) -> None:
        """Add a step to the trajectory.

        Args:
            obs: Observation dict.
            action: Action index.
            reward: Immediate reward.
            action_mask: [MAX_ACTIONS] binary mask.
            search_policy: [MAX_ACTIONS] MCTS policy.
            root_value: Value estimate from root of MCTS tree.
        """
        self.steps.append({
            "obs": obs,
            "action": int(action),
            "reward": float(reward),
            "action_mask": np.array(action_mask, dtype=np.float32),
            "search_policy": np.array(search_policy, dtype=np.float32),
            "root_value": float(root_value),
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

    def save_episode(self, trajectory: GameTrajectory) -> None:
        """Save a completed episode trajectory to the buffer.

        If buffer exceeds capacity (by number of transitions), remove oldest
        trajectories until within capacity.

        Args:
            trajectory: Completed GameTrajectory with all steps.
        """
        if len(trajectory) == 0:
            return

        # Compute priorities: use absolute deviation between target and root value
        ep_priority = 0.0
        for step in trajectory.steps:
            delta = abs(step["root_value"] - step["reward"])
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

    def sample_batch(
        self,
        batch_size: int,
        unroll_steps: int = 5,
        rng: np.random.Generator | None = None,
    ) -> Dict[str, torch.Tensor]:
        """Sample a mini-batch for training.

        Samples batch_size (trajectory, position) pairs using prioritized sampling.
        For each sample, unrolls K steps forward for computing unrolled losses.

        Returns tensors with batch dimension B = batch_size:
        - obs_batch: list of B observation dicts
        - action_batch: [B, K] int actions for unroll
        - reward_target: [B, K] float rewards
        - value_target: [B, K+1] float bootstrapped values
        - policy_target: [B, K+1, 80] MCTS policy distributions
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
        batch_actions = []
        batch_rewards = []
        batch_values = []
        batch_policies = []
        batch_masks = []

        for _ in range(batch_size):
            # Sample trajectory by priority
            traj_idx = int(rng.choice(len(self.trajectories), p=priorities_array))
            trajectory = self.trajectories[traj_idx]

            # Sample random position in trajectory
            pos = int(rng.integers(len(trajectory)))

            # Collect observation at position
            batch_obs.append(trajectory.steps[pos]["obs"])

            # Collect actions, rewards, values, policies for unroll
            actions = []
            rewards = []
            values = [trajectory.steps[pos]["root_value"]]
            policies = [trajectory.steps[pos]["search_policy"]]
            masks = [trajectory.steps[pos]["action_mask"]]

            for k in range(unroll_steps):
                step_idx = pos + k
                if step_idx < len(trajectory):
                    step = trajectory.steps[step_idx]
                    actions.append(step["action"])
                    rewards.append(step["reward"])
                    if step_idx + 1 < len(trajectory):
                        values.append(trajectory.steps[step_idx + 1]["root_value"])
                        policies.append(trajectory.steps[step_idx + 1]["search_policy"])
                        masks.append(trajectory.steps[step_idx + 1]["action_mask"])
                    else:
                        # Terminal state
                        values.append(0.0)
                        policies.append(np.zeros(len(step["search_policy"]), dtype=np.float32))
                        masks.append(np.zeros(len(step["action_mask"]), dtype=np.float32))
                else:
                    # Past episode end
                    actions.append(0)
                    rewards.append(0.0)
                    values.append(0.0)
                    policies.append(np.zeros(len(trajectory.steps[-1]["search_policy"]), dtype=np.float32))
                    masks.append(np.zeros(len(trajectory.steps[-1]["action_mask"]), dtype=np.float32))

            batch_actions.append(actions)
            batch_rewards.append(rewards)
            batch_values.append(values)
            batch_policies.append(policies)
            batch_masks.append(masks)

        # Convert to tensors
        action_batch = torch.tensor(batch_actions, dtype=torch.long)  # [B, K]
        reward_batch = torch.tensor(batch_rewards, dtype=torch.float32)  # [B, K]
        value_batch = torch.tensor(batch_values, dtype=torch.float32)  # [B, K+1]
        policy_batch = torch.tensor(batch_policies, dtype=torch.float32)  # [B, K+1, 80]
        mask_batch = torch.tensor(batch_masks, dtype=torch.float32)  # [B, K+1, 80]

        return {
            "obs_batch": batch_obs,
            "action_batch": action_batch,
            "reward_target": reward_batch,
            "value_target": value_batch,
            "policy_target": policy_batch,
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

    def __len__(self) -> int:
        """Return total number of transitions stored."""
        return self._total_transitions
