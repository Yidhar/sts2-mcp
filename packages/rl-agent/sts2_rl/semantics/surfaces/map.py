"""Map-route root-surface semantics."""

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


class MapSurfaceAdapter(BaseSurfaceAdapter):
    spec = SurfaceSpec(
        spec_id="map",
        version="map-surface-v1",
        role=SurfaceRole.ROOT,
        priority=500,
        claimed_paths=("map", "phase", "decision_domain"),
        aliases=("route",),
        authoritative_phases=("map", "route"),
        authoritative_state_types=("map", "route"),
        authoritative_action_kinds=("map",),
    )

    def matches(
        self,
        observation: Mapping[str, Any],
        legal_actions: Sequence[Mapping[str, Any]],
    ) -> bool:
        phase = normalize_token(
            first_value(
                observation.get("phase"),
                observation.get("decision_domain"),
            )
        )
        return (
            phase in {"map", "route"}
            or any(action_kind(action) == "map" for action in legal_actions)
            or (
                bool(mapping_at(observation, "map"))
                and observation.get("combat") is None
                and observation.get("event") is None
            )
        )

    def anchor_payload(self, observation: Mapping[str, Any]) -> Mapping[str, Any]:
        map_state = mapping_at(observation, "map")
        return common_locus(
            observation,
            root_kind=self.spec.spec_id,
            root_entity_id=first_value(
                map_state.get("current_coordinate"),
                map_state.get("current_node_id"),
            ),
        )

    def loop_node_payload(
        self,
        observation: Mapping[str, Any],
        legal_actions: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        map_state = mapping_at(observation, "map")
        return {
            "surface": self.spec.spec_id,
            "current": comparison_projection(
                first_value(
                    map_state.get("current_coordinate"),
                    map_state.get("current_node_id"),
                )
            ),
            "routes": sorted(
                (self.loop_action_payload(action) for action in legal_actions),
                key=lambda item: str(item),
            ),
        }
