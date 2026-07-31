"""Shared helpers for versioned surface adapters."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Final

from ..contracts import SurfaceSpec
from ..field_roles import comparison_projection, control_projection, exact_projection
from ..identity import SemanticKey
from ..progress import ProgressReceipt, common_progress_receipt

_ACTION_DISPATCH_KEYS: Final[frozenset[str]] = frozenset(
    {
        "action_handle",
        "action_id",
        "action_index",
        "card_index",
        "choice_index",
        "idx",
        "index",
        "option_index",
        "request_id",
    }
)


def normalize_token(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def mapping_at(
    observation: Mapping[str, Any],
    *keys: str,
) -> Mapping[str, Any]:
    current: Any = observation
    for key in keys:
        if not isinstance(current, Mapping):
            return {}
        current = current.get(key)
    return current if isinstance(current, Mapping) else {}


def first_value(
    *values: Any,
) -> Any:
    for value in values:
        if value is not None and value != "":
            return value
    return None


def common_locus(
    observation: Mapping[str, Any],
    *,
    root_kind: str,
    root_entity_id: Any = None,
) -> dict[str, Any]:
    run = mapping_at(observation, "run")
    room = mapping_at(observation, "room")
    map_state = mapping_at(observation, "map")
    coordinate = first_value(
        room.get("coordinate"),
        map_state.get("current_coordinate"),
        observation.get("map_coordinate"),
    )
    return {
        "root_kind": root_kind,
        "root_entity_id": root_entity_id,
        "act": first_value(run.get("act"), observation.get("act")),
        "floor": first_value(run.get("floor"), observation.get("floor")),
        "coordinate": comparison_projection(coordinate),
        "room_type": first_value(
            room.get("room_type"),
            room.get("type"),
            observation.get("room_type"),
        ),
        "room_model_id": first_value(
            room.get("room_model_id"),
            room.get("model_id"),
            observation.get("room_model_id"),
        ),
    }


def action_kind(action: Mapping[str, Any]) -> str:
    return normalize_token(
        first_value(
            action.get("model_action_kind"),
            action.get("kind"),
            action.get("action"),
        )
    )


def action_operation(action: Mapping[str, Any]) -> str:
    selection = action.get("selection")
    nested_operation = selection.get("operation_type") if isinstance(selection, Mapping) else None
    return normalize_token(
        first_value(
            action.get("selection_operation"),
            action.get("model_action_variant"),
            action.get("operation"),
            action.get("operation_type"),
            nested_operation,
            action.get("kind"),
            action.get("action"),
        )
    )


def _without_root_dispatch(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): exact_projection(child)
        for key, child in value.items()
        if str(key).lower() not in _ACTION_DISPATCH_KEYS
    }


def control_action_projection(value: Mapping[str, Any]) -> dict[str, Any]:
    """Canonical candidate facts with only root dispatch and cost noise gone.

    Root ``index``/``idx`` fields select a transport row and therefore cannot
    define learned identity.  Nested indices and slots, however, can bind a
    physical resource (a potion slot, shop slot, or card instance) and are
    deliberately retained.  Unknown fields are retained fail-closed.
    """

    projected = control_projection(value)
    if not isinstance(projected, Mapping):
        raise TypeError("control action projection root must remain a mapping")
    return {str(key): child for key, child in projected.items() if str(key).lower() not in _ACTION_DISPATCH_KEYS}


class BaseSurfaceAdapter:
    """Conservative default behavior inherited by concrete adapters."""

    spec: SurfaceSpec

    def matches(
        self,
        observation: Mapping[str, Any],
        legal_actions: Sequence[Mapping[str, Any]],
    ) -> bool:
        raise NotImplementedError

    def anchor_payload(self, observation: Mapping[str, Any]) -> Mapping[str, Any]:
        return common_locus(observation, root_kind=self.spec.spec_id)

    def exact_node_payload(
        self,
        observation: Mapping[str, Any],
        legal_actions: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        return {
            "surface": self.spec.spec_id,
            "observation": exact_projection(observation),
            "legal_actions": [self.exact_action_payload(action) for action in legal_actions],
        }

    def loop_node_payload(
        self,
        observation: Mapping[str, Any],
        legal_actions: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        return {
            "surface": self.spec.spec_id,
            "phase": normalize_token(observation.get("phase")),
            "decision_domain": normalize_token(observation.get("decision_domain")),
            "actions": sorted(
                (self.loop_action_payload(action) for action in legal_actions),
                key=lambda item: str(item),
            ),
        }

    def comparison_node_payload(
        self,
        observation: Mapping[str, Any],
        legal_actions: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        return {
            "surface": self.spec.spec_id,
            "observation": comparison_projection(observation),
            "legal_actions": [self.comparison_action_payload(action) for action in legal_actions],
        }

    def exact_action_payload(
        self,
        action: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        return _without_root_dispatch(action)

    def loop_action_payload(
        self,
        action: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        return {
            "kind": action_kind(action),
            "operation": action_operation(action),
            "control": control_action_projection(action),
        }

    def comparison_action_payload(
        self,
        action: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        return {
            str(key): comparison_projection(child)
            for key, child in action.items()
            if str(key).lower() not in _ACTION_DISPATCH_KEYS
        }

    def classify_progress(
        self,
        *,
        before_observation: Mapping[str, Any],
        after_observation: Mapping[str, Any],
        before_anchor: SemanticKey,
        after_anchor: SemanticKey,
    ) -> ProgressReceipt:
        return common_progress_receipt(
            before_observation=before_observation,
            after_observation=after_observation,
            before_anchor=before_anchor,
            after_anchor=after_anchor,
            before_loop_payload=self.loop_node_payload(before_observation, ()),
            after_loop_payload=self.loop_node_payload(after_observation, ()),
            source=self.spec.spec_id,
        )
