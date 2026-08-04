"""Strict, fail-closed grouping of physically duplicated legal actions."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Final

from .identity import SemanticContractError, canonical_payload_bytes

STRICT_ACTION_GROUPING_CONTRACT_VERSION: Final = (
    "sts2-strict-action-grouping-contract-v1"
)
_SELECTION_OPERATIONS: Final[frozenset[str]] = frozenset({"select", "deselect"})
_SELECTION_KIND_ALIASES: Final[dict[str, str]] = {
    "select_card": "select",
    "select_hand_card": "select",
    "combat_select_card": "select",
    "select_card_option": "select",
    "deselect_card": "deselect",
    "deselect_hand_card": "deselect",
    "combat_deselect_card": "deselect",
    "deselect_card_option": "deselect",
}
_ROOT_DISPATCH_KEYS: Final[frozenset[str]] = frozenset(
    {
        "action_handle",
        "action_id",
        "action_index",
        "card_index",
        "choice_index",
        "idx",
        "index",
        "option_index",
    }
)
_CARD_INSTANCE_KEYS: Final[frozenset[str]] = frozenset(
    {
        "card_instance_id",
        "instance_id",
        "instance_uuid",
    }
)
_CARD_CONTAINERS: Final[frozenset[str]] = frozenset(
    {
        "card",
        "upgrade_preview",
    }
)


def _normalized(value: Any) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def card_selection_operation(action: Mapping[str, Any]) -> str | None:
    """Return the single reviewed operation, declining conflicting aliases."""

    if _normalized(action.get("model_action_kind", "")) != "card_selection":
        return None
    reported: list[str] = []
    for key in ("selection_operation", "model_action_variant"):
        raw = action.get(key)
        if raw is not None and str(raw).strip():
            reported.append(_normalized(raw))
    selection = action.get("selection")
    if isinstance(selection, Mapping):
        raw = selection.get(
            "operation_type",
            selection.get("selection_operation"),
        )
        if raw is not None and str(raw).strip():
            reported.append(_normalized(raw))
    for key in ("kind", "action"):
        alias = _SELECTION_KIND_ALIASES.get(_normalized(action.get(key, "")))
        if alias is not None:
            reported.append(alias)
    if not reported or len(set(reported)) != 1:
        return None
    operation = reported[0]
    return operation if operation in _SELECTION_OPERATIONS else None


def _strict_selection_value(
    value: Any,
    *,
    path: tuple[str, ...] = (),
) -> Any:
    if isinstance(value, Mapping):
        projected: dict[str, Any] = {}
        parent = _normalized(path[-1]) if path else ""
        inside_raw = bool(path and _normalized(path[0]) == "_sim_raw")
        for raw_key, child in value.items():
            if not isinstance(raw_key, str):
                raise SemanticContractError("strict action equivalence requires string mapping keys")
            key = _normalized(raw_key)
            if not path and key in _ROOT_DISPATCH_KEYS:
                continue
            if inside_raw and len(path) == 1 and key in _ROOT_DISPATCH_KEYS:
                continue
            if parent in _CARD_CONTAINERS:
                if key in _CARD_INSTANCE_KEYS or key in _ROOT_DISPATCH_KEYS:
                    continue
            projected[raw_key] = _strict_selection_value(
                child,
                path=(*path, raw_key),
            )
        return projected
    if isinstance(value, list | tuple):
        return [_strict_selection_value(child, path=(*path, str(index))) for index, child in enumerate(value)]
    if value is None or isinstance(value, str | bool | int | float):
        return value
    location = ".".join(path) or "<root>"
    raise SemanticContractError(
        f"strict action equivalence supports only JSON-compatible values, got {type(value).__name__} at {location}"
    )


@dataclass(frozen=True, slots=True)
class StrictActionGroup:
    """One learned action and the raw positions it may dispatch through."""

    prototype: Mapping[str, Any]
    member_positions: tuple[int, ...]
    equivalence_fingerprint: str | None

    @property
    def multiplicity(self) -> int:
        return len(self.member_positions)


def strict_action_groups(
    legal_actions: Sequence[Mapping[str, Any]],
) -> tuple[StrictActionGroup, ...]:
    """Group only reviewed, strictly equal select/deselect card copies.

    Every unknown semantic field participates in equality.  Unsupported or
    malformed candidates remain singleton groups rather than being dropped.
    """

    groups: list[StrictActionGroup] = []
    group_index_by_payload: dict[bytes, int] = {}
    for position, action in enumerate(legal_actions):
        if card_selection_operation(action) is None:
            groups.append(
                StrictActionGroup(
                    prototype=dict(action),
                    member_positions=(position,),
                    equivalence_fingerprint=None,
                )
            )
            continue
        try:
            projected = _strict_selection_value(action)
            canonical = canonical_payload_bytes(projected)
        except SemanticContractError:
            groups.append(
                StrictActionGroup(
                    prototype=dict(action),
                    member_positions=(position,),
                    equivalence_fingerprint=None,
                )
            )
            continue
        existing = group_index_by_payload.get(canonical)
        if existing is not None:
            previous = groups[existing]
            groups[existing] = StrictActionGroup(
                prototype=previous.prototype,
                member_positions=(*previous.member_positions, position),
                equivalence_fingerprint=previous.equivalence_fingerprint,
            )
            continue
        fingerprint = hashlib.sha256(canonical).hexdigest()
        group_index_by_payload[canonical] = len(groups)
        if not isinstance(projected, dict):
            raise AssertionError("strict action projection root must be a mapping")
        groups.append(
            StrictActionGroup(
                prototype=projected,
                member_positions=(position,),
                equivalence_fingerprint=fingerprint,
            )
        )
    return tuple(groups)


@lru_cache(maxsize=1)
def _strict_action_grouping_contract_json() -> str:
    """Serialize the contract exactly once per process.

    Every input is a module-level ``Final`` constant fixed at import time, so
    the contract payload and its fingerprint are process constants.  The cache
    holds an immutable JSON string; callers always receive freshly decoded
    mappings and cannot mutate shared state.
    """

    payload = {
        "version": STRICT_ACTION_GROUPING_CONTRACT_VERSION,
        "selection_operations": sorted(_SELECTION_OPERATIONS),
        "selection_kind_aliases": dict(sorted(_SELECTION_KIND_ALIASES.items())),
        "root_dispatch_keys": sorted(_ROOT_DISPATCH_KEYS),
        "card_instance_keys": sorted(_CARD_INSTANCE_KEYS),
        "card_containers": sorted(_CARD_CONTAINERS),
        "canonicalization": "canonical_payload_bytes",
        "unknown_field_policy": "retain-or-singleton-fail-closed",
    }
    contract = {
        **payload,
        "fingerprint_sha256": hashlib.sha256(
            canonical_payload_bytes(payload)
        ).hexdigest(),
    }
    return json.dumps(contract)


def strict_action_grouping_contract() -> Mapping[str, Any]:
    """Return the frozen contract consumed by both semantics and encoding.

    The encoder used to maintain a second implementation of this projection.
    A shared versioned manifest makes the grouping authority singular and
    ensures its exact rules participate in the model encoding ABI.
    """

    contract: dict[str, Any] = json.loads(_strict_action_grouping_contract_json())
    return contract


__all__ = [
    "STRICT_ACTION_GROUPING_CONTRACT_VERSION",
    "StrictActionGroup",
    "card_selection_operation",
    "strict_action_grouping_contract",
    "strict_action_groups",
]
