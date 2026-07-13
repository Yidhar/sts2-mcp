"""Minimal data and objective contracts for the recurrent V-trace baseline."""

from .objective import (
    TASK_REWARD_SPEC,
    TaskObjective,
    TaskOutcome,
    TaskReward,
    TaskRewardCalculator,
    TaskRewardSpec,
    task_reward_identity,
)
from .rollout import (
    ROLLOUT_QUEUE_VERSION,
    ROLLOUT_STEP_VERSION,
    SEQUENCE_UNROLL_VERSION,
    BoundedRolloutQueue,
    RolloutQueueClosed,
    RolloutStep,
    SequenceUnroll,
)

__all__ = [
    "ROLLOUT_QUEUE_VERSION",
    "ROLLOUT_STEP_VERSION",
    "SEQUENCE_UNROLL_VERSION",
    "TASK_REWARD_SPEC",
    "BoundedRolloutQueue",
    "RolloutQueueClosed",
    "RolloutStep",
    "SequenceUnroll",
    "TaskObjective",
    "TaskOutcome",
    "TaskReward",
    "TaskRewardCalculator",
    "TaskRewardSpec",
    "task_reward_identity",
]
