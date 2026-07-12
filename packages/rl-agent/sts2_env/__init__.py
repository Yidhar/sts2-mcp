"""sts2_env — Gymnasium environment for Slay the Spire 2 RL training.

Hole H fix (review 2026-05-08): historically this module eagerly imported
torch-heavy submodules (network, AuxMaskablePPO, observation_v3, etc.),
which made lightweight helpers like ``deck_quality`` and
``route_heuristic`` impossible to unit-test without a torch install.

We now use PEP 562 ``__getattr__`` to lazy-load every public name. The
public API is unchanged: ``from sts2_env import BridgeClient`` still
works and triggers the heavy import on first access. But submodule
imports such as ``from sts2_env.deck_quality import ...`` no longer
drag in torch.
"""
from __future__ import annotations

from typing import Any

# Map exported names to (module, attribute) pairs. None means the
# attribute matches the module-level name.
_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "BridgeClient":                          (".bridge_client",  "BridgeClient"),
    "build_aux_targets":                     (".aux_targets",    "build_aux_targets"),
    "CombatSandboxEnv":                      (".combat_env",     "CombatSandboxEnv"),
    "SlayTheSpire2EnvV2":                    (".env_v2",         "SlayTheSpire2EnvV2"),
    "ObservationEncoderV3":                  (".observation_v3", "ObservationEncoderV3"),
    "WorldTokenObservationEncoder":          (".observation_v3", "WorldTokenObservationEncoder"),
    "bind_semantic_action":                  (".action_binder",  "bind_semantic_action"),
    "compute_transition_objective_rewards":  (".objective_heads", "compute_transition_objective_rewards"),
    "RunMemoryTracker":                      (".run_memory",     "RunMemoryTracker"),
    "semantic_action_signature":             (".semantic_action", "semantic_action_signature"),
}


__all__ = list(_LAZY_EXPORTS.keys())


def __getattr__(name: str) -> Any:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module 'sts2_env' has no attribute {name!r}")
    module_name, attr_name = target
    from importlib import import_module

    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return list(globals().keys()) + list(_LAZY_EXPORTS.keys())
