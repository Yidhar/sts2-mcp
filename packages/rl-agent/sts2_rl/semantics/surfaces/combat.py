"""Combat root semantics and conservative net-progress classification."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ..contracts import SurfaceSpec
from ..field_roles import comparison_projection
from ..identity import SemanticKey
from ..progress import ProgressKind, ProgressReceipt
from ..scopes import SurfaceRole
from .base import (
    BaseSurfaceAdapter,
    action_kind,
    common_locus,
    first_value,
    mapping_at,
    normalize_token,
)


def _enemy_records(observation: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    combat = mapping_at(observation, "combat")
    raw = first_value(combat.get("enemies"), observation.get("enemies"))
    if not isinstance(raw, list | tuple):
        return []
    return [enemy for enemy in raw if isinstance(enemy, Mapping)]


def _total_enemy_hp(observation: Mapping[str, Any]) -> float | None:
    enemies = _enemy_records(observation)
    total = 0.0
    seen = False
    for enemy in enemies:
        raw = first_value(enemy.get("hp"), enemy.get("current_hp"))
        if isinstance(raw, bool) or not isinstance(raw, int | float):
            continue
        total += max(float(raw), 0.0)
        seen = True
    return total if seen else None


def _combat_phase(observation: Mapping[str, Any]) -> tuple[Any, Any]:
    combat = mapping_at(observation, "combat")
    return (
        first_value(combat.get("wave"), combat.get("wave_index")),
        first_value(
            combat.get("phase"),
            combat.get("phase_id"),
            combat.get("stage"),
        ),
    )


class CombatSurfaceAdapter(BaseSurfaceAdapter):
    spec = SurfaceSpec(
        spec_id="combat",
        version="combat-surface-v1",
        role=SurfaceRole.ROOT,
        priority=1000,
        claimed_paths=("combat", "enemies", "phase", "decision_domain"),
        aliases=("battle",),
        authoritative_phases=("combat", "battle"),
        authoritative_state_types=("combat", "battle", "monster", "elite", "boss"),
        authoritative_action_kinds=("play_card", "end_turn", "use_potion"),
    )

    def matches(
        self,
        observation: Mapping[str, Any],
        legal_actions: Sequence[Mapping[str, Any]],
    ) -> bool:
        combat = mapping_at(observation, "combat")
        phase = normalize_token(
            first_value(
                observation.get("phase"),
                observation.get("decision_domain"),
            )
        )
        return (bool(combat) and combat.get("in_progress") is not False) or phase in {"combat", "battle"}

    def anchor_payload(self, observation: Mapping[str, Any]) -> Mapping[str, Any]:
        combat = mapping_at(observation, "combat")
        return common_locus(
            observation,
            root_kind=self.spec.spec_id,
            root_entity_id=first_value(
                combat.get("encounter_id"),
                combat.get("combat_id"),
                combat.get("room_model_id"),
                observation.get("encounter_id"),
                observation.get("room_model_id"),
            ),
        )

    def loop_node_payload(
        self,
        observation: Mapping[str, Any],
        legal_actions: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        combat = mapping_at(observation, "combat")
        player = mapping_at(observation, "player")
        enemies = [comparison_projection(enemy) for enemy in _enemy_records(observation)]
        return {
            "surface": self.spec.spec_id,
            "turn": first_value(
                combat.get("turn"),
                combat.get("round"),
                observation.get("turn"),
                observation.get("round"),
            ),
            "phase": list(_combat_phase(observation)),
            "player_control": {
                key: comparison_projection(player.get(key))
                for key in (
                    "block",
                    "energy",
                    "current_energy",
                    "hand",
                    "draw_pile",
                    "discard_pile",
                    "exhaust_pile",
                    "powers",
                )
                if player.get(key) is not None
            },
            "enemies": enemies,
            "actions": sorted(
                (
                    self.loop_action_payload(action)
                    for action in legal_actions
                    if action_kind(action) != "card_selection"
                ),
                key=lambda item: str(item),
            ),
        }

    def classify_progress(
        self,
        *,
        before_observation: Mapping[str, Any],
        after_observation: Mapping[str, Any],
        before_anchor: SemanticKey,
        after_anchor: SemanticKey,
    ) -> ProgressReceipt:
        if before_anchor != after_anchor:
            return ProgressReceipt(
                kind=ProgressKind.FLOW_ADVANCE,
                source="combat:anchor_changed",
            )
        if _combat_phase(before_observation) != _combat_phase(after_observation):
            return ProgressReceipt(
                kind=ProgressKind.FLOW_ADVANCE,
                source="combat:phase_or_wave_advanced",
            )
        before_hp = _total_enemy_hp(before_observation)
        after_hp = _total_enemy_hp(after_observation)
        if before_hp is not None and after_hp is not None and after_hp < before_hp:
            return ProgressReceipt(
                kind=ProgressKind.FLOW_ADVANCE,
                source="combat:net_enemy_hp_reduced",
            )
        return super().classify_progress(
            before_observation=before_observation,
            after_observation=after_observation,
            before_anchor=before_anchor,
            after_anchor=after_anchor,
        )
