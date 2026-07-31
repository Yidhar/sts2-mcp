"""Rest-site root semantics with selection handled as a nested overlay."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ..contracts import SurfaceSpec
from ..scopes import SurfaceRole
from .base import (
    BaseSurfaceAdapter,
    action_kind,
    common_locus,
    first_value,
    mapping_at,
    normalize_token,
)


class RestSurfaceAdapter(BaseSurfaceAdapter):
    spec = SurfaceSpec(
        spec_id="rest",
        version="rest-surface-v1",
        role=SurfaceRole.ROOT,
        priority=800,
        claimed_paths=("rest_site", "phase", "decision_domain"),
        aliases=("rest", "campfire"),
        authoritative_phases=("rest", "rest_site", "campfire"),
        authoritative_state_types=("rest", "rest_site", "campfire"),
        authoritative_action_kinds=("rest_site", "deck_upgrade"),
    )

    def matches(
        self,
        observation: Mapping[str, Any],
        legal_actions: Sequence[Mapping[str, Any]],
    ) -> bool:
        rest_site = mapping_at(observation, "rest_site")
        if any(
            rest_site.get(key) not in (None, "", [], {}) for key in ("id", "model_id", "state", "state_id", "options")
        ):
            return True
        phase = normalize_token(
            first_value(
                observation.get("phase"),
                observation.get("decision_domain"),
                observation.get("state_type"),
            )
        )
        return phase in {"rest", "rest_site", "campfire"} or any(
            action_kind(action) in {"rest_site", "deck_upgrade"} for action in legal_actions
        )

    def anchor_payload(self, observation: Mapping[str, Any]) -> Mapping[str, Any]:
        rest = mapping_at(observation, "rest_site")
        return common_locus(
            observation,
            root_kind=self.spec.spec_id,
            root_entity_id=first_value(
                rest.get("id"),
                rest.get("model_id"),
                observation.get("room_model_id"),
            ),
        )

    def loop_node_payload(
        self,
        observation: Mapping[str, Any],
        legal_actions: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        rest = mapping_at(observation, "rest_site")
        return {
            "surface": self.spec.spec_id,
            "state": {key: rest.get(key) for key in ("state", "state_id", "is_finished") if rest.get(key) is not None},
            "options": sorted(
                (
                    self.loop_action_payload(action)
                    for action in legal_actions
                    if action_kind(action) not in {"card_selection", "deck_upgrade"}
                ),
                key=lambda item: str(item),
            ),
        }
