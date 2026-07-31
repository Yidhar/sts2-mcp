"""Event root-surface semantics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ..contracts import SurfaceSpec
from ..field_roles import comparison_projection
from ..scopes import SurfaceRole
from .base import (
    BaseSurfaceAdapter,
    action_kind,
    common_locus,
    first_value,
    mapping_at,
    normalize_token,
)


class EventSurfaceAdapter(BaseSurfaceAdapter):
    spec = SurfaceSpec(
        spec_id="event",
        version="event-surface-v1",
        role=SurfaceRole.ROOT,
        priority=900,
        claimed_paths=("event", "phase", "decision_domain"),
        aliases=("event_room", "dialogue"),
        authoritative_phases=("event", "event_room", "dialogue"),
        authoritative_state_types=("event", "event_room", "dialogue"),
        authoritative_action_kinds=("event_option",),
    )

    def matches(
        self,
        observation: Mapping[str, Any],
        legal_actions: Sequence[Mapping[str, Any]],
    ) -> bool:
        event = mapping_at(observation, "event")
        has_event_identity = any(
            event.get(key) not in (None, "", [], {}) for key in ("event_id", "id", "model_id", "page_id", "state_id")
        )
        return (
            has_event_identity
            or normalize_token(
                first_value(
                    observation.get("phase"),
                    observation.get("decision_domain"),
                    observation.get("state_type"),
                )
            )
            in {"event", "event_room", "dialogue"}
            or any(action_kind(action) == "event_option" for action in legal_actions)
        )

    def anchor_payload(self, observation: Mapping[str, Any]) -> Mapping[str, Any]:
        event = mapping_at(observation, "event")
        event_id = first_value(
            event.get("event_id"),
            event.get("id"),
            event.get("model_id"),
            observation.get("event_id"),
        )
        return common_locus(
            observation,
            root_kind=self.spec.spec_id,
            root_entity_id=event_id,
        )

    def loop_node_payload(
        self,
        observation: Mapping[str, Any],
        legal_actions: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        event = mapping_at(observation, "event")
        page = {
            key: comparison_projection(event.get(key))
            for key in (
                "event_id",
                "id",
                "page",
                "page_id",
                "state",
                "state_id",
                "stage",
                "stage_id",
                "prompt_id",
                "is_finished",
            )
            if event.get(key) is not None
        }
        return {
            "surface": self.spec.spec_id,
            "page": page,
            "actions": sorted(
                (self.loop_action_payload(action) for action in legal_actions),
                key=lambda item: str(item),
            ),
        }
