"""Minimal, auditable foundations for the restarted STS2 learning baseline.

This package deliberately does not import the legacy MuZero trainer.  It owns a
small canonical transition/target vocabulary, one fixed normalized reward
specification, and a stratified replay buffer whose sampling bias is explicit.
"""

from .replay import (
    ReplayBatch,
    ReplayMix,
    ReplaySample,
    ReplayStratum,
    StratifiedReplayBuffer,
)
from .reward import (
    BASELINE_REWARD_SPEC,
    BASELINE_TRANSITION_PROJECTION_SPEC,
    BaselineRewardCalculator,
    BaselineRewardSpec,
    BaselineTransitionProjectionSpec,
    RewardBreakdown,
    baseline_reward_identity,
)
from .transition import (
    BASELINE_TARGET_VERSION,
    BASELINE_TRANSITION_VERSION,
    BaselineTargets,
    BaselineTransition,
    PotentialState,
)

__all__ = [
    "BASELINE_REWARD_SPEC",
    "BASELINE_TARGET_VERSION",
    "BASELINE_TRANSITION_PROJECTION_SPEC",
    "BASELINE_TRANSITION_VERSION",
    "BaselineRewardCalculator",
    "BaselineRewardSpec",
    "BaselineTargets",
    "BaselineTransition",
    "BaselineTransitionProjectionSpec",
    "PotentialState",
    "ReplayBatch",
    "ReplayMix",
    "ReplaySample",
    "ReplayStratum",
    "RewardBreakdown",
    "StratifiedReplayBuffer",
    "baseline_reward_identity",
]
