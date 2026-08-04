"""Reviewed field roles shared by semantic surface adapters."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import Any, Final


class FieldRole(StrEnum):
    FLOW_ANCHOR = "flow_anchor"
    FLOW_FRONTIER = "flow_frontier"
    CONTROL_NODE = "control_node"
    CONTROL_ACTION = "control_action"
    DURABLE_RESOURCE = "durable_resource"
    PROGRESS_MEASURE = "progress_measure"
    EXACT_ONLY = "exact_only"
    COST_LEDGER = "cost_ledger"
    PRESENTATION = "presentation"
    TRANSPORT = "transport"
    UNKNOWN = "unknown"


TRANSPORT_KEYS: Final[frozenset[str]] = frozenset(
    {
        # The simulator transport keeps an unnormalised copy of the current
        # view under ``_sim_raw``.  Its shape follows the active screen (for
        # example ``card_select.player.deck`` becomes ``rest.player.deck`` on
        # Cancel), so it is neither an independent gameplay fact nor a durable
        # resource authority.  Retaining it made a pure UI teardown look like
        # a deck mutation and incorrectly awarded completion credit.
        "_sim_raw",
        "action_handle",
        "client_id",
        "created_at",
        "elapsed_ms",
        "episode_id",
        "expected_state_version",
        "expected_step_index",
        "idempotency_key",
        "request_id",
        "revision",
        "session_id",
        "state_version",
        "step_index",
        "timestamp",
        "updated_at",
    }
)
PRESENTATION_KEYS: Final[frozenset[str]] = frozenset(
    {
        "description",
        "description_key",
        "hover_tip_ids",
        "label",
        "localized_text",
        "preview",
        "preview_text",
        "title",
        "tooltip",
    }
)
COST_KEYS: Final[frozenset[str]] = frozenset(
    {
        "current_hp",
        "damage_taken",
        "dead",
        "death_count",
        "hp",
        "hp_lost",
        "is_dead",
        "max_hp",
        "player_hp_lost",
        "revival_count",
        "revivals",
        "revivals_used",
    }
)
DURABLE_KEYS: Final[frozenset[str]] = frozenset(
    {
        "deck",
        "deck_cards",
        "gold",
        "open_potion_slots",
        "potions",
        "relics",
        "shop_inventory",
        "shop_stock",
        "reward_slots",
    }
)


def _project(
    value: Any,
    *,
    excluded_keys: frozenset[str],
) -> Any:
    if isinstance(value, Mapping):
        return {
            str(raw_key): _project(child, excluded_keys=excluded_keys)
            for raw_key, child in value.items()
            if str(raw_key).lower() not in excluded_keys
        }
    if isinstance(value, list | tuple):
        return [_project(child, excluded_keys=excluded_keys) for child in value]
    return value


def exact_projection(value: Any) -> Any:
    """Drop transport only; costs and unknown public facts remain exact."""

    return _project(value, excluded_keys=TRANSPORT_KEYS)


def comparison_projection(value: Any) -> Any:
    """Drop transport/presentation while retaining policy-relevant resources."""

    return _project(
        value,
        excluded_keys=TRANSPORT_KEYS | PRESENTATION_KEYS,
    )


def control_projection(value: Any) -> Any:
    """Project a control identity without deleting unreviewed semantics.

    Control/action identity has a stricter contract than state comparison:
    two candidates that are distinct after strict grouping must not silently
    alias.  Consequently presentation-like identity fallbacks (for example
    the stable ``*.title`` keys emitted for potions) and every unknown public
    field remain authoritative here.  Only transport metadata and realized
    cost-ledger facts are removed recursively.
    """

    return _project(
        value,
        excluded_keys=TRANSPORT_KEYS | COST_KEYS,
    )


def collect_role_values(value: Any, keys: frozenset[str]) -> Any:
    """Collect matching authoritative fields for deterministic diffs.

    Transport subtrees are skipped wholesale.  Merely projecting the matched
    leaf is insufficient: a duplicate raw view can relocate an otherwise
    identical ``deck`` between screen-specific paths and thereby fabricate a
    durable change from transport shape alone.
    """

    found: list[dict[str, Any]] = []

    def visit(child: Any, path: tuple[str, ...]) -> None:
        if isinstance(child, Mapping):
            for raw_key, grandchild in child.items():
                key = str(raw_key)
                if key.lower() in TRANSPORT_KEYS:
                    continue
                next_path = (*path, key)
                if key.lower() in keys:
                    found.append(
                        {
                            "path": list(next_path),
                            "value": comparison_projection(grandchild),
                        }
                    )
                else:
                    visit(grandchild, next_path)
        elif isinstance(child, list | tuple):
            for index, grandchild in enumerate(child):
                visit(grandchild, (*path, str(index)))

    visit(value, ())
    return found
