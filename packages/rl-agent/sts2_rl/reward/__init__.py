"""Canonical transition facts and the single versioned reward boundary."""

from .transition import (
    REWARD_SPEC_VERSION,
    TRANSITION_SCHEMA_VERSION,
    CanonicalTransition,
    LegacyRewardCalculator,
    RewardBreakdown,
    RewardSpec,
    TransitionFacts,
    VersionedRewardCalculator,
    canonicalize_legacy_transition,
    derive_transition_facts,
)

__all__ = [
    "REWARD_SPEC_VERSION",
    "TRANSITION_SCHEMA_VERSION",
    "CanonicalTransition",
    "LegacyRewardCalculator",
    "RewardBreakdown",
    "RewardSpec",
    "TransitionFacts",
    "VersionedRewardCalculator",
    "canonicalize_legacy_transition",
    "derive_transition_facts",
]
