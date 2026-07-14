"""Typed environment and recurrent V-trace v2 runtime for STS2 learning."""

from .artifacts import artifact_root, resolve_artifact_path, resolve_external_input_path, validate_artifact_component
from .contracts import (
    BackendCapabilities,
    CombatResetRequest,
    EnvironmentBackend,
    EnvironmentResult,
    ResetRequest,
    StepRequest,
)

__all__ = [
    "BackendCapabilities",
    "CombatResetRequest",
    "EnvironmentBackend",
    "EnvironmentResult",
    "ResetRequest",
    "StepRequest",
    "artifact_root",
    "resolve_artifact_path",
    "resolve_external_input_path",
    "validate_artifact_component",
]

__version__ = "0.4.0"
