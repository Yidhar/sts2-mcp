"""Stable, typed public boundary for the STS2 RL runtime.

The historical sts2_env and muzero packages remain available while callers
migrate. New code should depend on this package's contracts instead of passing
untyped bridge dictionaries through every layer.
"""

from .artifacts import artifact_root, resolve_artifact_path, resolve_external_input_path, validate_artifact_component
from .contracts import (
    BackendCapabilities,
    CombatResetRequest,
    EnvironmentBackend,
    EnvironmentResult,
    ResetRequest,
    StepRequest,
)
from .reward import (
    CanonicalTransition,
    LegacyRewardCalculator,
    RewardBreakdown,
    RewardSpec,
    TransitionFacts,
    VersionedRewardCalculator,
)

__all__ = [
    "BackendCapabilities",
    "CanonicalTransition",
    "CombatResetRequest",
    "EnvironmentBackend",
    "EnvironmentResult",
    "LegacyRewardCalculator",
    "ResetRequest",
    "RewardBreakdown",
    "RewardSpec",
    "StepRequest",
    "TransitionFacts",
    "VersionedRewardCalculator",
    "artifact_root",
    "resolve_artifact_path",
    "resolve_external_input_path",
    "validate_artifact_component",
]

__version__ = "0.2.0"
