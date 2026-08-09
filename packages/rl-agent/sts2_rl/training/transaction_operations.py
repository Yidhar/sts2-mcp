"""Reviewed transaction-operation vocabulary and capability contract.

Transaction operation names used to be normalized independently by the
configuration parser and collector.  That allowed a configuration to accept
an operation which the collector silently normalized to the empty string.  A
single immutable registry now owns aliases and capabilities for all training
components.

The registry describes protocol grammar only.  It never names a concrete
card, relic, event, encounter, or strategy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


def _normalize_operation_text(value: object) -> str:
    return "_".join(
        str(value or "").strip().lower().replace("-", " ").split()
    )


@dataclass(frozen=True, slots=True)
class TransactionOperationSpec:
    """One reviewed protocol operation family and its training capabilities."""

    name: str
    aliases: frozenset[str]
    exploration_enabled: bool
    opens_selection_lifecycle: bool
    requires_completion_guidance: bool

    def __post_init__(self) -> None:
        normalized_name = _normalize_operation_text(self.name)
        if not normalized_name or normalized_name != self.name:
            raise ValueError("transaction operation names must be canonical")
        normalized_aliases = frozenset(
            _normalize_operation_text(alias) for alias in self.aliases
        )
        if "" in normalized_aliases:
            raise ValueError("transaction operation aliases must be non-empty")
        if normalized_aliases != self.aliases or self.name not in self.aliases:
            raise ValueError(
                "transaction operation aliases must be normalized and include the name"
            )
        if self.requires_completion_guidance and not self.opens_selection_lifecycle:
            raise ValueError(
                "completion guidance is valid only for selection lifecycles"
            )


TRANSACTION_OPERATION_SPECS: Final[tuple[TransactionOperationSpec, ...]] = (
    TransactionOperationSpec(
        name="upgrade",
        aliases=frozenset(
            {"forge", "open_upgrade_selection", "smith", "upgrade"}
        ),
        exploration_enabled=True,
        opens_selection_lifecycle=True,
        requires_completion_guidance=True,
    ),
    TransactionOperationSpec(
        name="remove",
        aliases=frozenset(
            {
                "card_removal",
                "purchase_card_removal",
                "remove",
                "remove_card",
            }
        ),
        exploration_enabled=True,
        opens_selection_lifecycle=True,
        requires_completion_guidance=True,
    ),
    TransactionOperationSpec(
        name="reward_skip",
        aliases=frozenset(
            {"reward_skip", "skip_card_reward", "skip_reward"}
        ),
        exploration_enabled=True,
        opens_selection_lifecycle=False,
        requires_completion_guidance=False,
    ),
    TransactionOperationSpec(
        name="relic_purchase",
        aliases=frozenset({"purchase_relic", "relic_purchase"}),
        exploration_enabled=True,
        opens_selection_lifecycle=False,
        requires_completion_guidance=False,
    ),
    # Rest/heal is not behavior-side scaffolding.  It is registered so replay
    # can give it the same factual next-rest/Act option-Q contract as forge.
    TransactionOperationSpec(
        name="rest",
        aliases=frozenset({"heal", "rest", "sleep"}),
        exploration_enabled=False,
        opens_selection_lifecycle=False,
        requires_completion_guidance=False,
    ),
)

_SPEC_BY_NAME: Final[dict[str, TransactionOperationSpec]] = {
    spec.name: spec for spec in TRANSACTION_OPERATION_SPECS
}
_SPEC_BY_ALIAS: Final[dict[str, TransactionOperationSpec]] = {}
for _spec in TRANSACTION_OPERATION_SPECS:
    for _alias in _spec.aliases:
        _previous = _SPEC_BY_ALIAS.setdefault(_alias, _spec)
        if _previous is not _spec:  # pragma: no cover - import invariant
            raise RuntimeError(
                f"transaction operation alias {_alias!r} is ambiguous"
            )

TRANSACTION_EXPLORATION_OPERATIONS: Final[frozenset[str]] = frozenset(
    spec.name for spec in TRANSACTION_OPERATION_SPECS if spec.exploration_enabled
)
TRANSACTION_SELECTION_OPERATIONS: Final[frozenset[str]] = frozenset(
    spec.name
    for spec in TRANSACTION_OPERATION_SPECS
    if spec.opens_selection_lifecycle
)
TRANSACTION_GUIDANCE_OPERATIONS: Final[frozenset[str]] = frozenset(
    spec.name
    for spec in TRANSACTION_OPERATION_SPECS
    if spec.requires_completion_guidance
)
TRANSACTION_LIFECYCLE_OPERATIONS: Final[frozenset[str]] = frozenset(
    {*TRANSACTION_SELECTION_OPERATIONS, "rest"}
)


def canonical_transaction_operation(value: object) -> str:
    """Return a canonical reviewed operation, or ``""`` when unknown."""

    spec = _SPEC_BY_ALIAS.get(_normalize_operation_text(value))
    return spec.name if spec is not None else ""


def transaction_operation_spec(value: object) -> TransactionOperationSpec | None:
    """Resolve either a canonical name or reviewed alias to its specification."""

    canonical = canonical_transaction_operation(value)
    return _SPEC_BY_NAME.get(canonical)


__all__ = [
    "TRANSACTION_EXPLORATION_OPERATIONS",
    "TRANSACTION_GUIDANCE_OPERATIONS",
    "TRANSACTION_LIFECYCLE_OPERATIONS",
    "TRANSACTION_OPERATION_SPECS",
    "TRANSACTION_SELECTION_OPERATIONS",
    "TransactionOperationSpec",
    "canonical_transaction_operation",
    "transaction_operation_spec",
]
