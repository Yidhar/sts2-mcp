"""Fixed combat action template mapping for combat-sandbox training.

This module builds a stable, position-based combat action space so combat
policies do not need to learn against a transient compacted legal-action list.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .observation_common import MAX_ENEMIES, MAX_HAND, MAX_POTIONS


END_TURN_SLOT = 0

PLAY_SELF_BASE = END_TURN_SLOT + 1
PLAY_SELF_COUNT = MAX_HAND

PLAY_ENEMY_BASE = PLAY_SELF_BASE + PLAY_SELF_COUNT
PLAY_ENEMY_COUNT = MAX_HAND * MAX_ENEMIES

POTION_SELF_BASE = PLAY_ENEMY_BASE + PLAY_ENEMY_COUNT
POTION_SELF_COUNT = MAX_POTIONS

POTION_ENEMY_BASE = POTION_SELF_BASE + POTION_SELF_COUNT
POTION_ENEMY_COUNT = MAX_POTIONS * MAX_ENEMIES

SELECT_CARD_BASE = POTION_ENEMY_BASE + POTION_ENEMY_COUNT
SELECT_CARD_COUNT = MAX_HAND

CONFIRM_SELECTION_SLOT = SELECT_CARD_BASE + SELECT_CARD_COUNT
CANCEL_SELECTION_SLOT = CONFIRM_SELECTION_SLOT + 1

NUM_FIXED_COMBAT_ACTIONS = CANCEL_SELECTION_SLOT + 1


def play_self_slot(hand_index: int) -> int | None:
    if 0 <= hand_index < MAX_HAND:
        return PLAY_SELF_BASE + hand_index
    return None


def play_enemy_slot(hand_index: int, enemy_index: int) -> int | None:
    if 0 <= hand_index < MAX_HAND and 0 <= enemy_index < MAX_ENEMIES:
        return PLAY_ENEMY_BASE + (hand_index * MAX_ENEMIES) + enemy_index
    return None


def potion_self_slot(potion_index: int) -> int | None:
    if 0 <= potion_index < MAX_POTIONS:
        return POTION_SELF_BASE + potion_index
    return None


def potion_enemy_slot(potion_index: int, enemy_index: int) -> int | None:
    if 0 <= potion_index < MAX_POTIONS and 0 <= enemy_index < MAX_ENEMIES:
        return POTION_ENEMY_BASE + (potion_index * MAX_ENEMIES) + enemy_index
    return None


def select_card_slot(card_index: int) -> int | None:
    if 0 <= card_index < MAX_HAND:
        return SELECT_CARD_BASE + card_index
    return None


def slot_label(slot: int) -> str:
    if slot == END_TURN_SLOT:
        return "end_turn"
    if PLAY_SELF_BASE <= slot < PLAY_SELF_BASE + PLAY_SELF_COUNT:
        hand_index = slot - PLAY_SELF_BASE
        return f"play_self[{hand_index}]"
    if PLAY_ENEMY_BASE <= slot < PLAY_ENEMY_BASE + PLAY_ENEMY_COUNT:
        offset = slot - PLAY_ENEMY_BASE
        hand_index = offset // MAX_ENEMIES
        enemy_index = offset % MAX_ENEMIES
        return f"play_enemy[{hand_index}->{enemy_index}]"
    if POTION_SELF_BASE <= slot < POTION_SELF_BASE + POTION_SELF_COUNT:
        potion_index = slot - POTION_SELF_BASE
        return f"potion_self[{potion_index}]"
    if POTION_ENEMY_BASE <= slot < POTION_ENEMY_BASE + POTION_ENEMY_COUNT:
        offset = slot - POTION_ENEMY_BASE
        potion_index = offset // MAX_ENEMIES
        enemy_index = offset % MAX_ENEMIES
        return f"potion_enemy[{potion_index}->{enemy_index}]"
    if SELECT_CARD_BASE <= slot < SELECT_CARD_BASE + SELECT_CARD_COUNT:
        card_index = slot - SELECT_CARD_BASE
        return f"select_card[{card_index}]"
    if slot == CONFIRM_SELECTION_SLOT:
        return "confirm_selection"
    if slot == CANCEL_SELECTION_SLOT:
        return "cancel_selection"
    return f"slot[{slot}]"


@dataclass(slots=True)
class FixedCombatActionBinding:
    mask: np.ndarray
    slot_to_legal: dict[int, int]
    slot_labels: list[str]

    def legal_slots(self) -> list[int]:
        return [index for index, value in enumerate(self.mask.tolist()) if value]

    def legal_index_for_slot(self, slot: int) -> int | None:
        return self.slot_to_legal.get(int(slot))


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalized_kind(action: dict[str, Any]) -> str:
    return _safe_text(action.get("kind")).lower()


def _normalized_action_id(action: dict[str, Any]) -> str:
    return _safe_text(action.get("action_id")).lower()


def _selection_token(action: dict[str, Any]) -> str:
    return _safe_text(action.get("selection")).lower()


def _enemy_lookup(raw_obs: dict[str, Any] | None) -> dict[str, int]:
    combat = raw_obs.get("combat") if isinstance(raw_obs, dict) else {}
    enemies = combat.get("enemies") if isinstance(combat, dict) else []
    lookup: dict[str, int] = {}
    if not isinstance(enemies, list):
        return lookup
    for index, enemy in enumerate(enemies[:MAX_ENEMIES]):
        if not isinstance(enemy, dict):
            continue
        for candidate in (
            enemy.get("entity_id"),
            enemy.get("id"),
            enemy.get("name"),
            enemy.get("title"),
        ):
            text = _safe_text(candidate).lower()
            if text:
                lookup[text] = index
    return lookup


def _extract_target_index(action: dict[str, Any], enemy_lookup: dict[str, int]) -> int | None:
    for key in ("target_index", "target_slot", "enemy_index"):
        value = _safe_int(action.get(key))
        if value is not None:
            return value

    target = action.get("target")
    if isinstance(target, dict):
        for key in ("target_index", "slot_index", "index"):
            value = _safe_int(target.get(key))
            if value is not None:
                return value
        for key in ("entity_id", "id", "name", "title"):
            text = _safe_text(target.get(key)).lower()
            if text and text in enemy_lookup:
                return enemy_lookup[text]
    else:
        text = _safe_text(target).lower()
        if text and text in enemy_lookup:
            return enemy_lookup[text]
    return None


def _extract_hand_index(action: dict[str, Any]) -> int | None:
    for key in ("hand_index", "slot_index", "card_index", "index"):
        value = _safe_int(action.get(key))
        if value is not None:
            return value

    card = action.get("card")
    if isinstance(card, dict):
        for key in ("hand_index", "slot_index", "index"):
            value = _safe_int(card.get(key))
            if value is not None:
                return value
    return None


def _extract_potion_index(action: dict[str, Any]) -> int | None:
    for key in ("slot_index", "potion_index", "index"):
        value = _safe_int(action.get(key))
        if value is not None:
            return value
    potion = action.get("potion")
    if isinstance(potion, dict):
        for key in ("slot_index", "index"):
            value = _safe_int(potion.get(key))
            if value is not None:
                return value
    return None


def _bind_slot_for_action(action: dict[str, Any], enemy_lookup: dict[str, int]) -> int | None:
    action_id = _normalized_action_id(action)
    kind = _normalized_kind(action)
    selection = _selection_token(action)

    if action_id == "end_turn" or kind == "end_turn":
        return END_TURN_SLOT

    if kind == "play_card" or action_id.startswith("play_card"):
        hand_index = _extract_hand_index(action)
        if hand_index is None:
            return None
        target_index = _extract_target_index(action, enemy_lookup)
        if target_index is None or target_index < 0:
            return play_self_slot(hand_index)
        return play_enemy_slot(hand_index, target_index)

    if kind == "use_potion" or action_id.startswith("use_potion"):
        potion_index = _extract_potion_index(action)
        if potion_index is None:
            return None
        target_index = _extract_target_index(action, enemy_lookup)
        if target_index is None or target_index < 0:
            return potion_self_slot(potion_index)
        return potion_enemy_slot(potion_index, target_index)

    if (
        kind in {"card_selection", "combat_select_card", "combat_select"}
        or action_id.startswith("combat_select")
    ):
        hand_index = _extract_hand_index(action)
        if hand_index is not None:
            return select_card_slot(hand_index)

    if "confirm" in action_id or selection == "confirm":
        return CONFIRM_SELECTION_SLOT

    if any(token in action_id for token in ("cancel", "skip", "close")) or selection in {"cancel", "skip", "close"}:
        return CANCEL_SELECTION_SLOT

    return None


def _binding_priority(action: dict[str, Any]) -> float:
    kind = _normalized_kind(action)
    source = action.get("card") if kind == "play_card" else action.get("potion")
    if not isinstance(source, dict):
        return 0.0
    effect_preview = source.get("effect_preview") if isinstance(source.get("effect_preview"), dict) else {}

    def _metric(key: str) -> float:
        value = effect_preview.get(key, source.get(key))
        try:
            return float(value or 0.0)
        except (TypeError, ValueError):
            return 0.0

    return (
        (2.0 if kind == "play_card" else 1.0)
        + _metric("damage")
        + _metric("block")
        + (0.5 * _metric("draw"))
        + (0.25 * _metric("heal"))
    )


def build_fixed_action_binding(
    raw_obs: dict[str, Any] | None,
    legal_actions: list[dict[str, Any]] | None,
) -> FixedCombatActionBinding:
    mask = np.zeros(NUM_FIXED_COMBAT_ACTIONS, dtype=bool)
    slot_to_legal: dict[int, int] = {}
    slot_priority: dict[int, float] = {}
    enemy_lookup = _enemy_lookup(raw_obs)

    for legal_index, action in enumerate(legal_actions or []):
        if not isinstance(action, dict):
            continue
        slot = _bind_slot_for_action(action, enemy_lookup)
        if slot is None:
            continue
        priority = _binding_priority(action)
        previous = slot_priority.get(slot)
        if previous is None or priority >= previous:
            slot_to_legal[slot] = legal_index
            slot_priority[slot] = priority
            mask[slot] = True

    return FixedCombatActionBinding(
        mask=mask,
        slot_to_legal=slot_to_legal,
        slot_labels=[slot_label(index) for index in range(NUM_FIXED_COMBAT_ACTIONS)],
    )
