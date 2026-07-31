"""Reward root-surface semantics."""

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


class RewardSurfaceAdapter(BaseSurfaceAdapter):
    spec = SurfaceSpec(
        spec_id="reward",
        version="reward-surface-v1",
        role=SurfaceRole.ROOT,
        priority=600,
        claimed_paths=("rewards", "reward", "phase", "decision_domain"),
        aliases=("card_reward",),
        authoritative_phases=("reward", "card_reward"),
        authoritative_state_types=("reward", "card_reward", "combat_rewards"),
        authoritative_action_kinds=("reward", "card_reward"),
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
            any(
                mapping_at(observation, "reward").get(key) not in (None, "", [], {})
                for key in ("reward_id", "id", "slots", "remaining")
            )
            or bool(mapping_at(observation, "rewards"))
            or phase in {"reward", "card_reward"}
            or any(action_kind(action) in {"reward", "card_reward"} for action in legal_actions)
        )

    def anchor_payload(self, observation: Mapping[str, Any]) -> Mapping[str, Any]:
        reward = mapping_at(observation, "reward")
        return common_locus(
            observation,
            root_kind=self.spec.spec_id,
            root_entity_id=first_value(
                reward.get("reward_id"),
                reward.get("id"),
                observation.get("reward_id"),
                observation.get("room_model_id"),
            ),
        )

    def loop_node_payload(
        self,
        observation: Mapping[str, Any],
        legal_actions: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        reward = mapping_at(observation, "reward")
        slots = first_value(
            observation.get("rewards"),
            reward.get("slots"),
            reward.get("remaining"),
        )
        return {
            "surface": self.spec.spec_id,
            "remaining_slots": comparison_projection(slots),
            "actions": sorted(
                (
                    self.loop_action_payload(action)
                    for action in legal_actions
                    if action_kind(action) != "card_selection"
                ),
                key=lambda item: str(item),
            ),
        }
