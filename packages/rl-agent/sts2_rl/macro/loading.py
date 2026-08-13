"""Tolerant trunk loading across intentional head-group ABI changes.

The isolated semantic-reset lineages evolve head shapes deliberately (the
retired liveness heads; the candidate-Q head widened to read state
features; the v47 macro residual-actor and option-value heads retired with
config v20). Loading inherits every shape-compatible tensor and refuses any
drift outside the explicitly tolerated head groups — the macro analogue of
the legacy model-parameter-initialization contract.
"""

from __future__ import annotations

from typing import Any, Final

from torch import Tensor

MACRO_LOADING_CONTRACT_VERSION: Final = "sts2-macro-trunk-loading-v1"

TOLERATED_HEAD_GROUPS: Final[tuple[str, ...]] = (
    "candidate_liveness_cost_head",
    "liveness_cost_value_head",
    "transaction_q_head",
    # Retired v47 macro-option family: v47-era champion checkpoints and
    # existing macro publications still carry these tensors; the current
    # model never constructs them, so they load as explicit drops.
    "macro_surface_candidate_embedding",
    "macro_surface_state_embedding",
    "macro_policy_head",
    "macro_option_value_head",
)


def load_trunk_state(
    model: Any,
    state: dict[str, Tensor],
    *,
    tolerated_groups: tuple[str, ...] = TOLERATED_HEAD_GROUPS,
) -> dict[str, list[str]]:
    """Load ``state`` into ``model``, inheriting shape-compatible tensors.

    Keys that are absent from the model, or whose shapes mismatch, are
    dropped ONLY when they belong to a tolerated head group (those heads
    start fresh); anything else raises. Returns the dropped/fresh key lists
    for the caller's provenance record.
    """

    model_state = model.state_dict()
    filtered: dict[str, Tensor] = {}
    dropped: list[str] = []
    for key, value in state.items():
        target = model_state.get(key)
        if target is not None and tuple(target.shape) == tuple(value.shape):
            filtered[key] = value
        else:
            dropped.append(key)
    fresh = [key for key in model_state if key not in filtered]

    def _tolerated(key: str) -> bool:
        return any(group in key for group in tolerated_groups)

    drift = [key for key in dropped if not _tolerated(key)]
    missing = [key for key in fresh if not _tolerated(key)]
    if drift or missing:
        raise RuntimeError(
            "checkpoint drift beyond tolerated head groups: "
            f"dropped={drift} missing={missing}"
        )
    result = model.load_state_dict(filtered, strict=False)
    unexpected = [key for key in result.unexpected_keys if not _tolerated(key)]
    if unexpected:  # pragma: no cover - filtered above, defensive
        raise RuntimeError(f"unexpected checkpoint keys: {unexpected}")
    return {"dropped": dropped, "fresh": fresh}


__all__ = [
    "MACRO_LOADING_CONTRACT_VERSION",
    "TOLERATED_HEAD_GROUPS",
    "load_trunk_state",
]
