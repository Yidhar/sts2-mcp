"""Selection overlay semantics for select/deselect/confirm/cancel flows."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any, Final

from ..contracts import SurfaceSpec
from ..field_roles import comparison_projection
from ..grouping import card_selection_operation
from ..identity import canonical_payload_bytes
from ..scopes import SurfaceRole
from .base import (
    BaseSurfaceAdapter,
    action_kind,
    first_value,
    mapping_at,
    normalize_token,
)

_CARD_INSTANCE_KEYS: Final[frozenset[str]] = frozenset(
    {
        "card_instance_id",
        "instance_id",
        "instance_uuid",
        "uuid",
        "uid",
    }
)
_CARD_PERMANENT_KEYS: Final[tuple[str, ...]] = (
    "model_id",
    "card_id",
    "id",
    "upgrade_level",
    "upgrades",
    "is_upgraded",
    "enchantment",
    "enchantments",
    "affliction",
    "afflictions",
    "source_zone",
    "zone",
)


def _card_semantics(card: Any) -> Any:
    if not isinstance(card, Mapping):
        return comparison_projection(card)
    return {
        str(key): comparison_projection(value)
        for key, value in card.items()
        if str(key).lower() in _CARD_PERMANENT_KEYS and str(key).lower() not in _CARD_INSTANCE_KEYS
    }


def _selected_multiset(selection: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = first_value(
        selection.get("selected_cards"),
        selection.get("selected"),
        selection.get("selection"),
    )
    if not isinstance(raw, list | tuple):
        return []
    # The selected-card surface is a multiset.  Python's ``str(dict)`` is
    # insertion-order-sensitive and therefore cannot be a semantic identity:
    # the bridge may emit the same card facts in a different key order.  Keep
    # the canonical JSON text in the payload so the multiset remains
    # human-auditable while equality is governed by the kernel's sole
    # canonical serializer.
    canonical_items = [
        canonical_payload_bytes(_card_semantics(item)).decode("utf-8")
        for item in raw
    ]
    return [{"item": item, "multiplicity": count} for item, count in sorted(Counter(canonical_items).items())]


def _selection_state(observation: Mapping[str, Any]) -> Mapping[str, Any]:
    direct = mapping_at(observation, "selection")
    return direct if direct else mapping_at(observation, "card_selection")


class SelectionSurfaceAdapter(BaseSurfaceAdapter):
    spec = SurfaceSpec(
        spec_id="selection",
        version="selection-surface-v1",
        role=SurfaceRole.OVERLAY,
        priority=1000,
        claimed_paths=("selection", "legal_actions"),
        aliases=("card_selection", "multi_select"),
    )

    def matches(
        self,
        observation: Mapping[str, Any],
        legal_actions: Sequence[Mapping[str, Any]],
    ) -> bool:
        selection = _selection_state(observation)
        if selection:
            return True
        phase = normalize_token(
            first_value(
                observation.get("phase"),
                observation.get("decision_domain"),
                observation.get("state_type"),
            )
        )
        return phase in {"selection", "card_selection", "multi_select"} or any(
            action_kind(action) == "card_selection" or card_selection_operation(action) is not None
            for action in legal_actions
        )

    def anchor_payload(self, observation: Mapping[str, Any]) -> Mapping[str, Any]:
        selection = _selection_state(observation)
        return {
            "overlay": self.spec.spec_id,
            "prompt_id": first_value(
                selection.get("prompt_id"),
                selection.get("selection_id"),
                observation.get("prompt_id"),
            ),
        }

    def loop_node_payload(
        self,
        observation: Mapping[str, Any],
        legal_actions: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        selection = _selection_state(observation)
        return {
            "surface": self.spec.spec_id,
            "prompt_id": comparison_projection(
                first_value(
                    selection.get("prompt_id"),
                    selection.get("selection_id"),
                    observation.get("prompt_id"),
                )
            ),
            "contract": {
                key: comparison_projection(
                    first_value(
                        selection.get(key),
                        observation.get(key),
                    )
                )
                for key in (
                    "min_select",
                    "max_select",
                    "remaining_select",
                    "remaining_picks",
                    "selected_count",
                    "confirm_ready",
                    "cancelable",
                )
            },
            "selected_multiset": _selected_multiset(selection),
            "actions": sorted(
                (self.loop_action_payload(action) for action in legal_actions),
                key=canonical_payload_bytes,
            ),
        }


__all__ = ["SelectionSurfaceAdapter"]
