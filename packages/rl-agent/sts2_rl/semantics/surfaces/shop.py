"""Shop root-surface semantics."""

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


class ShopSurfaceAdapter(BaseSurfaceAdapter):
    spec = SurfaceSpec(
        spec_id="shop",
        version="shop-surface-v1",
        role=SurfaceRole.ROOT,
        priority=700,
        claimed_paths=("shop", "phase", "decision_domain"),
        aliases=("merchant",),
        authoritative_phases=("shop", "merchant"),
        authoritative_state_types=("shop", "merchant"),
        authoritative_action_kinds=("shop",),
    )

    def matches(
        self,
        observation: Mapping[str, Any],
        legal_actions: Sequence[Mapping[str, Any]],
    ) -> bool:
        return (
            any(
                mapping_at(observation, "shop").get(key) not in (None, "", [], {})
                for key in ("shop_id", "id", "model_id", "items", "stock", "inventory", "is_open")
            )
            or normalize_token(
                first_value(
                    observation.get("phase"),
                    observation.get("decision_domain"),
                )
            )
            in {"shop", "merchant"}
            or any(action_kind(action) == "shop" for action in legal_actions)
        )

    def anchor_payload(self, observation: Mapping[str, Any]) -> Mapping[str, Any]:
        shop = mapping_at(observation, "shop")
        return common_locus(
            observation,
            root_kind=self.spec.spec_id,
            root_entity_id=first_value(
                shop.get("shop_id"),
                shop.get("id"),
                shop.get("model_id"),
                observation.get("room_model_id"),
            ),
        )

    def loop_node_payload(
        self,
        observation: Mapping[str, Any],
        legal_actions: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        shop = mapping_at(observation, "shop")
        stock = first_value(
            shop.get("stock"),
            shop.get("inventory"),
            shop.get("items"),
        )
        return {
            "surface": self.spec.spec_id,
            "stock": comparison_projection(stock),
            "remove_available": comparison_projection(
                first_value(
                    shop.get("remove_available"),
                    shop.get("card_removal_available"),
                )
            ),
            "actions": sorted(
                (
                    self.loop_action_payload(action)
                    for action in legal_actions
                    if action_kind(action) != "card_selection"
                ),
                key=lambda item: str(item),
            ),
        }
