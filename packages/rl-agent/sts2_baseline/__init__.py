"""Minimal data and objective contracts for the recurrent V-trace baseline."""

from .objective import (
    REVIVAL_EFFICIENCY_REWARD_SPEC,
    TASK_REWARD_SPEC,
    RevivalEfficiencyRewardCalculator,
    RevivalEfficiencyRewardSpec,
    TaskObjective,
    TaskOutcome,
    TaskReward,
    TaskRewardCalculator,
    TaskRewardSpec,
    revival_efficiency_reward_identity,
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
    "REVIVAL_EFFICIENCY_REWARD_SPEC",
    "ROLLOUT_QUEUE_VERSION",
    "ROLLOUT_STEP_VERSION",
    "SEQUENCE_UNROLL_VERSION",
    "TASK_REWARD_SPEC",
    "BoundedRolloutQueue",
    "RevivalEfficiencyRewardCalculator",
    "RevivalEfficiencyRewardSpec",
    "RolloutQueueClosed",
    "RolloutStep",
    "SequenceUnroll",
    "TaskObjective",
    "TaskOutcome",
    "TaskReward",
    "TaskRewardCalculator",
    "TaskRewardSpec",
    "revival_efficiency_reward_identity",
    "task_reward_identity",
]
