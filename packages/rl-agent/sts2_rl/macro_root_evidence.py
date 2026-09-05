"""Pure evidence comparison for macro-root reconstruction diagnostics.

This module does not restore an environment, select an action, or certify a
clone.  Matching observable facts cannot establish equality of hidden simulator
RNG.  Policy tensors and reward-calculator state must be compared separately.
Inputs must be factual DTOs, not credential-bearing session descriptors.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import Any, Final

from sts2_baseline.objective import (
    ACT_FRONTLOADED_DEATH_REWARD_SPEC,
    ActFrontloadedDeathRewardCalculator,
)

# These exact wire names are available to a DTO-aware caller for explicitly
# selected transport envelope levels.  This module never removes them: the
# same name can also occur in an entity, modifier, or _training factual field.
TRANSPORT_IDENTITY_KEYS: Final[frozenset[str]] = frozenset(
    {
        "session_id",
        "request_id",
        "episode_id",
        "state_version",
        "expected_state_version",
        "committed_state_version",
        "state_version_before",
        "state_version_after",
        "before_state_version",
        "after_state_version",
        "transition_token",
        "deadline_utc",
        "process_started_at_utc",
        "published_at_utc",
    }
)


def _child_path(path: str, key: str) -> str:
    return f"{path}[{json.dumps(key, ensure_ascii=False)}]"


def _json_value(value: Any, *, path: str, active: set[int]) -> Any:
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite evidence number at {path}")
        return value
    if isinstance(value, Mapping | list | tuple):
        identity = id(value)
        if identity in active:
            raise ValueError(f"cyclic evidence container at {path}")
        active.add(identity)
        try:
            if isinstance(value, Mapping):
                if any(not isinstance(key, str) for key in value):
                    raise TypeError(f"evidence mappings require string keys at {path}")
                return {
                    key: _json_value(
                        value[key],
                        path=_child_path(path, key),
                        active=active,
                    )
                    for key in sorted(value)
                }
            return [
                _json_value(child, path=f"{path}[{index}]", active=active)
                for index, child in enumerate(value)
            ]
        finally:
            active.remove(identity)
    raise TypeError(f"unsupported evidence type {type(value).__name__} at {path}")


def canonical_facts(value: Any) -> Any:
    """Validate and deep-copy JSON facts without deleting any field.

    A caller that understands the DTO context must remove transport identity
    fields explicitly before calling this function.  Unknown fields, array
    order, multiplicity, numeric types, entity bindings, and all _training
    facts are preserved, even when named like a transport field.  No pile
    sorting or entity-ID renaming is attempted.  Unsupported objects and
    non-finite numbers fail closed; callers must convert tensor evidence.
    """

    return _json_value(value, path="$", active=set())


def _difference(expected: Any, actual: Any, *, path: str, atol: float, rtol: float) -> dict[str, Any] | None:
    if type(expected) is not type(actual):
        return {
            "path": path,
            "kind": "type_mismatch",
            "expected_type": type(expected).__name__,
            "actual_type": type(actual).__name__,
        }
    if isinstance(expected, dict):
        for key in sorted(expected.keys() | actual.keys()):
            child_path = _child_path(path, key)
            if key not in expected or key not in actual:
                return {
                    "path": child_path,
                    "kind": "missing_key",
                    "expected_present": key in expected,
                    "actual_present": key in actual,
                }
            result = _difference(expected[key], actual[key], path=child_path, atol=atol, rtol=rtol)
            if result is not None:
                return result
        return None
    if isinstance(expected, list):
        for index, (left, right) in enumerate(zip(expected, actual, strict=False)):
            result = _difference(left, right, path=f"{path}[{index}]", atol=atol, rtol=rtol)
            if result is not None:
                return result
        if len(expected) != len(actual):
            return {
                "path": f"{path}[{min(len(expected), len(actual))}]",
                "kind": "length_mismatch",
                "expected_length": len(expected),
                "actual_length": len(actual),
            }
        return None
    if isinstance(expected, float):
        matches = math.isclose(expected, actual, abs_tol=atol, rel_tol=rtol)
    else:
        matches = expected == actual
    if matches:
        return None
    return {"path": path, "kind": "value_mismatch", "expected": expected, "actual": actual}


def first_difference(
    expected: Any,
    actual: Any,
    *,
    float_atol: float = 0.0,
    float_rtol: float = 0.0,
) -> dict[str, Any] | None:
    """Report the first deterministic JSON path mismatch, or ``None``.

    Both complete inputs are validated before comparison, including fields
    beyond an earlier mismatch.  This function never strips keys implicitly.
    Facts should use the exact defaults; explicit tolerances are intended only
    for separately supplied policy hidden/logit float arrays.  Tolerances never
    equate booleans and integers, integers and floats, or unequal integers.
    """

    for name, tolerance in (("float_atol", float_atol), ("float_rtol", float_rtol)):
        if isinstance(tolerance, bool) or not isinstance(tolerance, int | float):
            raise TypeError(f"{name} must be a finite non-negative number")
        if not math.isfinite(tolerance) or tolerance < 0:
            raise ValueError(f"{name} must be a finite non-negative number")
    left = _json_value(expected, path="$", active=set())
    right = _json_value(actual, path="$", active=set())
    return _difference(left, right, path="$", atol=float_atol, rtol=float_rtol)


def reward_state(calculator: ActFrontloadedDeathRewardCalculator) -> dict[str, Any]:
    """Read reward-v9 course and Act-local state without initializing it.

    This is diagnostic evidence, not a supported state-restoration API.  The
    environment's total HP-loss/revival facts must also be compared separately.
    """

    if not isinstance(calculator, ActFrontloadedDeathRewardCalculator):
        raise TypeError("macro-root reward evidence requires the reward-v9 calculator")
    state: dict[str, Any] = {
        "reward_version": ACT_FRONTLOADED_DEATH_REWARD_SPEC.version,
        "objective": calculator.base.objective,
        "discount": calculator.base.discount,
        "maximum_episode_steps": calculator.maximum_episode_steps,
        "revival_budget": calculator.revival_budget,
        "episode_started": calculator._episode_started,
        "current_act": calculator._current_act,
        "act_revivals": calculator._act_revivals,
        "last_total_revivals": calculator._last_total_revivals,
    }
    _json_value(state, path="$", active=set())
    return state


__all__ = ["TRANSPORT_IDENTITY_KEYS", "canonical_facts", "first_difference", "reward_state"]
