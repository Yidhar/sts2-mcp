"""Fixed semantic rollout action space for deep abstract search.

This is intentionally coarser than live legal actions. The goal is to preserve
the parts of a combat action that matter for deeper search while dropping
transient concrete details such as exact hand slot identity.
"""

from __future__ import annotations

from typing import Any

import numpy as np

SEMANTIC_ROLLOUT_TYPES = (
    "attack",
    "block",
    "draw",
    "buff",
    "debuff",
    "scaling",
    "utility",
    "exhaust_synergy",
)
SEMANTIC_ROLLOUT_SCALES = (
    "small",
    "medium",
    "large",
    "x_cost",
)
SEMANTIC_ROLLOUT_TARGETS = (
    "single",
    "all",
    "self",
)

_TYPE_TO_IDX = {name: idx for idx, name in enumerate(SEMANTIC_ROLLOUT_TYPES)}
_SCALE_TO_IDX = {name: idx for idx, name in enumerate(SEMANTIC_ROLLOUT_SCALES)}
_TARGET_TO_IDX = {name: idx for idx, name in enumerate(SEMANTIC_ROLLOUT_TARGETS)}

SEMANTIC_ROLLOUT_SIZE = (
    len(SEMANTIC_ROLLOUT_TYPES)
    * len(SEMANTIC_ROLLOUT_SCALES)
    * len(SEMANTIC_ROLLOUT_TARGETS)
)
SEMANTIC_ROLLOUT_FEAT_DIM = (
    len(SEMANTIC_ROLLOUT_TYPES)
    + len(SEMANTIC_ROLLOUT_SCALES)
    + len(SEMANTIC_ROLLOUT_TARGETS)
    + 6
)


def _float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_text(value: Any) -> str:
    return str(value or "").strip().lower()


def _extract_signature(action_or_signature: Any) -> dict[str, Any]:
    if not isinstance(action_or_signature, dict):
        return {}
    if "family" in action_or_signature or "roles" in action_or_signature:
        return dict(action_or_signature)
    semantic = action_or_signature.get("semantic")
    if isinstance(semantic, dict):
        return dict(semantic)
    return dict(action_or_signature)


def _rollout_type(signature: dict[str, Any]) -> str:
    roles = {_safe_text(role) for role in (signature.get("roles") or [])}
    family = _safe_text(signature.get("family"))
    effect_summary = _safe_text(signature.get("effect_summary"))

    if "attack" in roles:
        return "attack"
    if "block" in roles:
        return "block"
    if "draw" in roles:
        return "draw"
    if "buff" in roles:
        return "buff"
    if "debuff" in roles:
        return "debuff"
    if "scaling" in roles:
        return "scaling"
    if "exhaust" in effect_summary or "exhaust" in _safe_text(signature.get("title")):
        return "exhaust_synergy"
    if family in {"end_turn", "use_potion", "proceed"}:
        return "utility"
    return "utility"


def _rollout_scale(signature: dict[str, Any]) -> str:
    if signature.get("is_x_cost") or _float(signature.get("x_cost_value")) > 0.0:
        return "x_cost"

    magnitude = max(
        _float(signature.get("damage")),
        _float(signature.get("block")),
        _float(signature.get("damage_per_hit")) * max(_float(signature.get("hits"), 1.0), 1.0),
        _float(signature.get("heal")),
        _float(signature.get("draw")) * 4.0,
        _float(signature.get("weak")) * 5.0,
        _float(signature.get("vulnerable")) * 5.0,
        _float(signature.get("price")) * 0.2,
    )
    if magnitude <= 6.0:
        return "small"
    if magnitude <= 14.0:
        return "medium"
    return "large"


def _rollout_target(signature: dict[str, Any]) -> str:
    target_scope = _safe_text(signature.get("target_scope"))
    if target_scope == "all_enemies":
        return "all"
    if target_scope == "single_enemy":
        return "single"
    return "self"


def semantic_rollout_signature(action_or_signature: Any) -> dict[str, Any]:
    signature = _extract_signature(action_or_signature)
    if not signature:
        return {}

    rollout_type = _rollout_type(signature)
    rollout_scale = _rollout_scale(signature)
    rollout_target = _rollout_target(signature)
    index = semantic_rollout_index_from_parts(
        rollout_type=rollout_type,
        rollout_scale=rollout_scale,
        rollout_target=rollout_target,
    )
    return {
        "rollout_type": rollout_type,
        "rollout_scale": rollout_scale,
        "rollout_target": rollout_target,
        "rollout_index": index,
    }


def semantic_rollout_index_from_parts(
    *,
    rollout_type: str,
    rollout_scale: str,
    rollout_target: str,
) -> int:
    type_idx = _TYPE_TO_IDX.get(_safe_text(rollout_type), 0)
    scale_idx = _SCALE_TO_IDX.get(_safe_text(rollout_scale), 0)
    target_idx = _TARGET_TO_IDX.get(_safe_text(rollout_target), 2)
    return (
        type_idx * len(SEMANTIC_ROLLOUT_SCALES) * len(SEMANTIC_ROLLOUT_TARGETS)
        + scale_idx * len(SEMANTIC_ROLLOUT_TARGETS)
        + target_idx
    )


def semantic_rollout_index(action_or_signature: Any) -> int:
    rollout = semantic_rollout_signature(action_or_signature)
    return int(rollout.get("rollout_index", 0)) if rollout else 0


def encode_semantic_rollout_numeric(action_or_signature: Any) -> np.ndarray:
    vector = np.zeros(SEMANTIC_ROLLOUT_FEAT_DIM, dtype=np.float32)
    signature = _extract_signature(action_or_signature)
    rollout = semantic_rollout_signature(signature)
    if not rollout:
        return vector

    offset = 0
    vector[offset + _TYPE_TO_IDX[rollout["rollout_type"]]] = 1.0
    offset += len(SEMANTIC_ROLLOUT_TYPES)
    vector[offset + _SCALE_TO_IDX[rollout["rollout_scale"]]] = 1.0
    offset += len(SEMANTIC_ROLLOUT_SCALES)
    vector[offset + _TARGET_TO_IDX[rollout["rollout_target"]]] = 1.0
    offset += len(SEMANTIC_ROLLOUT_TARGETS)

    vector[offset + 0] = min(_float(signature.get("damage")) / 30.0, 1.0)
    vector[offset + 1] = min(_float(signature.get("block")) / 30.0, 1.0)
    vector[offset + 2] = min(_float(signature.get("draw")) / 5.0, 1.0)
    vector[offset + 3] = min(_float(signature.get("hits")) / 8.0, 1.0)
    vector[offset + 4] = min(_float(signature.get("damage_per_hit")) / 20.0, 1.0)
    vector[offset + 5] = min(_float(signature.get("x_cost_value")) / 10.0, 1.0)
    return vector


def aggregate_concrete_policy_to_semantic(
    search_policy: np.ndarray,
    semantic_candidate_indices: list[int] | np.ndarray | None,
) -> np.ndarray:
    out = np.zeros(SEMANTIC_ROLLOUT_SIZE, dtype=np.float32)
    if semantic_candidate_indices is None:
        return out
    indices = np.asarray(semantic_candidate_indices, dtype=np.int64).reshape(-1)
    probs = np.asarray(search_policy, dtype=np.float32).reshape(-1)
    count = min(indices.shape[0], probs.shape[0])
    for slot in range(count):
        prob = float(probs[slot])
        if prob <= 0.0:
            continue
        semantic_idx = int(indices[slot])
        if 0 <= semantic_idx < SEMANTIC_ROLLOUT_SIZE:
            out[semantic_idx] += prob
    total = float(out.sum())
    if total > 1e-9:
        out /= total
    return out

