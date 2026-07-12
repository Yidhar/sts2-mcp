"""Typed factory that collapses the legacy 94-argument constructor call."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from .config import TrainingConfig


@dataclass(frozen=True, slots=True)
class TrainingResources:
    network: Any
    mcts: Any
    replay: Any
    environment: Any
    optimizer: Any


def _derived_option(name: str, options: Mapping[str, Any]) -> tuple[bool, Any]:
    semantic_names = {
        "semantic_policy_weight",
        "semantic_value_weight",
        "semantic_reward_weight",
        "semantic_state_consistency_weight",
        "objective_diversity_weight",
    }
    if name in semantic_names:
        return True, 0.0 if options.get("disable_semantic_training") else options.get(name)
    if name == "trivial_build_fast_path":
        return True, not bool(options.get("disable_trivial_build_fast_path", False))
    if name == "potion_reward_fast_path":
        return True, not bool(options.get("disable_potion_reward_fast_path", False))
    if name == "human_demo_alignment_shadow_only":
        return True, not bool(options.get("human_demo_alignment_enable_loss", False))
    if name == "offline_alignment_shadow_only":
        return True, not bool(options.get("offline_alignment_enable_loss", False))
    return False, None


def build_legacy_trainer(
    trainer_type: Callable[..., Any],
    *,
    resources: TrainingResources,
    config: TrainingConfig,
    overrides: Mapping[str, Any] | None = None,
) -> Any:
    """Build the temporary MuZeroTrainer facade from typed config/resources."""

    explicit = dict(overrides or {})
    options = config.options
    resource_values = {
        "network": resources.network,
        "mcts": resources.mcts,
        "buffer": resources.replay,
        "env": resources.environment,
        "optimizer": resources.optimizer,
    }
    kwargs: dict[str, Any] = {}
    signature = inspect.signature(trainer_type)
    for name, parameter in signature.parameters.items():
        if name == "self":
            continue
        if name in resource_values:
            kwargs[name] = resource_values[name]
        elif name in explicit:
            kwargs[name] = explicit[name]
        elif name in options:
            kwargs[name] = options[name]
        else:
            found, value = _derived_option(name, options)
            if found and value is not None:
                kwargs[name] = value
            elif parameter.default is inspect.Parameter.empty:
                raise ValueError(f"TrainingConfig cannot satisfy required trainer argument {name!r}")
    return trainer_type(**kwargs)
