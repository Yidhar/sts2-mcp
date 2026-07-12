"""Typed training configuration and composition services."""

from .config import (
    CONFIG_VERSION,
    EnvironmentConfig,
    ModelConfig,
    OptimizationConfig,
    RuntimeConfig,
    TrainingConfig,
    load_config_defaults,
    parse_args_with_config,
)
from .factory import TrainingResources, build_legacy_trainer
from .services import (
    CheckpointManager,
    Collector,
    Evaluator,
    ExperimentContext,
    Learner,
    ReplayStore,
    Telemetry,
    TrainingRuntime,
)

__all__ = [
    "CONFIG_VERSION",
    "CheckpointManager",
    "Collector",
    "EnvironmentConfig",
    "Evaluator",
    "ExperimentContext",
    "Learner",
    "ModelConfig",
    "OptimizationConfig",
    "ReplayStore",
    "RuntimeConfig",
    "Telemetry",
    "TrainingConfig",
    "TrainingResources",
    "TrainingRuntime",
    "build_legacy_trainer",
    "load_config_defaults",
    "parse_args_with_config",
]
