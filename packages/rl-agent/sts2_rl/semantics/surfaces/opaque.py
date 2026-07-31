"""Fail-closed fallback surface retaining exact facts without coarse credit."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ..contracts import SurfaceSpec
from ..field_roles import comparison_projection, exact_projection
from ..scopes import SurfaceRole
from .base import BaseSurfaceAdapter, common_locus


class OpaqueSurfaceAdapter(BaseSurfaceAdapter):
    spec = SurfaceSpec(
        spec_id="opaque",
        version="opaque-surface-v1",
        role=SurfaceRole.ROOT,
        priority=0,
        claimed_paths=("*",),
        allows_direct_policy_credit=False,
    )

    def matches(
        self,
        observation: Mapping[str, Any],
        legal_actions: Sequence[Mapping[str, Any]],
    ) -> bool:
        return True

    def anchor_payload(self, observation: Mapping[str, Any]) -> Mapping[str, Any]:
        return common_locus(
            observation,
            root_kind=self.spec.spec_id,
            root_entity_id=comparison_projection(observation.get("room_model_id")),
        )

    def loop_node_payload(
        self,
        observation: Mapping[str, Any],
        legal_actions: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        # Exact fallback deliberately avoids unsafe field omission.
        return {
            "surface": self.spec.spec_id,
            "opaque_observation": exact_projection(observation),
            "opaque_actions": [self.exact_action_payload(action) for action in legal_actions],
        }
