"""Typed progress receipts separating flow, resources, control and costs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Final

from .field_roles import COST_KEYS, DURABLE_KEYS, collect_role_values, exact_projection
from .identity import SemanticKey, canonical_payload_bytes

PROGRESS_RECEIPT_CONTRACT_VERSION: Final = "sts2-progress-receipt-v1"


class ProgressKind(StrEnum):
    FLOW_ADVANCE = "flow_advance"
    DURABLE_COMMIT = "durable_commit"
    CONTROL_MOVE = "control_move"
    COST_ONLY = "cost_only"
    NONE = "none"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class ProgressReceipt:
    kind: ProgressKind
    source: str
    changed_paths: tuple[str, ...] = ()
    details: tuple[tuple[str, str], ...] = ()

    @property
    def flow_advanced(self) -> bool:
        return self.kind is ProgressKind.FLOW_ADVANCE

    @property
    def durable_committed(self) -> bool:
        return self.kind is ProgressKind.DURABLE_COMMIT

    @property
    def is_censored(self) -> bool:
        return self.kind is ProgressKind.UNKNOWN


def _role_payload(value: Mapping[str, Any], keys: frozenset[str]) -> bytes:
    return canonical_payload_bytes(collect_role_values(value, keys))


def _changed_paths(
    before: Any,
    after: Any,
    *,
    path: tuple[str, ...] = (),
    maximum: int = 64,
) -> tuple[str, ...]:
    if maximum <= 0:
        return ()
    if isinstance(before, Mapping) and isinstance(after, Mapping):
        paths: list[str] = []
        for key in sorted(set(before) | set(after), key=str):
            child_path = (*path, str(key))
            if key not in before or key not in after:
                paths.append(".".join(child_path))
            else:
                paths.extend(
                    _changed_paths(
                        before[key],
                        after[key],
                        path=child_path,
                        maximum=maximum - len(paths),
                    )
                )
            if len(paths) >= maximum:
                break
        return tuple(paths)
    if isinstance(before, list | tuple) and isinstance(after, list | tuple):
        if before == after:
            return ()
        return (".".join(path) or "<root>",)
    if before != after:
        return (".".join(path) or "<root>",)
    return ()


def common_progress_receipt(
    *,
    before_observation: Mapping[str, Any],
    after_observation: Mapping[str, Any],
    before_anchor: SemanticKey,
    after_anchor: SemanticKey,
    before_loop_payload: Any,
    after_loop_payload: Any,
    source: str,
) -> ProgressReceipt:
    """Classify one transition without treating cost as durable progress."""

    if before_anchor != after_anchor:
        return ProgressReceipt(
            kind=ProgressKind.FLOW_ADVANCE,
            source=f"{source}:anchor_changed",
        )
    before_durable = _role_payload(before_observation, DURABLE_KEYS)
    after_durable = _role_payload(after_observation, DURABLE_KEYS)
    if before_durable != after_durable:
        return ProgressReceipt(
            kind=ProgressKind.DURABLE_COMMIT,
            source=f"{source}:durable_resource_changed",
        )
    if canonical_payload_bytes(before_loop_payload) != canonical_payload_bytes(after_loop_payload):
        return ProgressReceipt(
            kind=ProgressKind.CONTROL_MOVE,
            source=f"{source}:control_node_changed",
        )
    before_cost = _role_payload(before_observation, COST_KEYS)
    after_cost = _role_payload(after_observation, COST_KEYS)
    if before_cost != after_cost:
        return ProgressReceipt(
            kind=ProgressKind.COST_ONLY,
            source=f"{source}:cost_ledger_changed",
        )
    if canonical_payload_bytes(exact_projection(before_observation)) != (
        canonical_payload_bytes(exact_projection(after_observation))
    ):
        return ProgressReceipt(
            kind=ProgressKind.UNKNOWN,
            source=f"{source}:unreviewed_semantic_change",
            changed_paths=_changed_paths(
                exact_projection(before_observation),
                exact_projection(after_observation),
            ),
        )
    return ProgressReceipt(
        kind=ProgressKind.NONE,
        source=f"{source}:unchanged",
    )
