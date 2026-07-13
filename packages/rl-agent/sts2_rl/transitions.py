"""Canonical factual transition derivation for transport backends.

This module contains observations-to-facts projection only. Reward ownership is
exclusively in :mod:`sts2_baseline.reward`; there is no backend-reward or legacy
passthrough field here.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast

CombatResult = Literal["none", "victory", "defeat", "escaped"]


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _ids(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list | tuple):
        return ()
    return tuple(str(item) for item in value if str(item).strip())


@dataclass(frozen=True, slots=True)
class TransitionFacts:
    hp_delta: float = 0.0
    gold_delta: float = 0.0
    floor_delta: int = 0
    cards_added: tuple[str, ...] = ()
    cards_removed: tuple[str, ...] = ()
    potions_added: tuple[str, ...] = ()
    potions_removed: tuple[str, ...] = ()
    relics_used: tuple[str, ...] = ()
    room_entered: str | None = None
    combat_result: CombatResult = "none"
    terminal_reason: str | None = None
    enemy_hp_delta: float = 0.0

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> TransitionFacts:
        data = payload if isinstance(payload, Mapping) else {}
        raw_combat_result = str(data.get("combat_result") or "none").lower()
        combat_result = (
            cast(CombatResult, raw_combat_result)
            if raw_combat_result in {"none", "victory", "defeat", "escaped"}
            else "none"
        )
        return cls(
            hp_delta=_number(data.get("hp_delta", data.get("player_hp_delta"))),
            gold_delta=_number(data.get("gold_delta")),
            floor_delta=int(_number(data.get("floor_delta"))),
            cards_added=_ids(data.get("cards_added")),
            cards_removed=_ids(data.get("cards_removed")),
            potions_added=_ids(data.get("potions_added")),
            potions_removed=_ids(data.get("potions_removed")),
            relics_used=_ids(data.get("relics_used")),
            room_entered=(
                str(data["room_entered"]) if data.get("room_entered") is not None else None
            ),
            combat_result=combat_result,
            terminal_reason=(
                str(data["terminal_reason"])
                if data.get("terminal_reason") is not None
                else None
            ),
            enemy_hp_delta=_number(data.get("enemy_hp_delta")),
        )


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _player_from_observation(observation: Mapping[str, Any]) -> Mapping[str, Any]:
    direct = observation.get("player")
    if isinstance(direct, Mapping):
        return direct
    run = _mapping(observation.get("run"))
    player = run.get("player")
    return player if isinstance(player, Mapping) else {}


def _entity_ids(value: Any) -> tuple[str, ...]:
    if isinstance(value, Mapping):
        value = value.get("cards")
    if not isinstance(value, list | tuple):
        return ()
    result: list[str] = []
    for item in value:
        if isinstance(item, str):
            result.append(item)
        elif isinstance(item, Mapping):
            entity_id = item.get("id", item.get("card_id", item.get("name")))
            if entity_id is not None:
                result.append(str(entity_id))
    return tuple(result)


def _multiset_delta(before: tuple[str, ...], after: tuple[str, ...]) -> tuple[str, ...]:
    remaining: dict[str, int] = {}
    for value in before:
        remaining[value] = remaining.get(value, 0) + 1
    added: list[str] = []
    for value in after:
        if remaining.get(value, 0) > 0:
            remaining[value] -= 1
        else:
            added.append(value)
    return tuple(added)


def _relic_used_state(value: Any) -> dict[str, bool]:
    """Return exact native relic consumption state keyed by stable relic ID.

    The simulator exports ``is_used_up`` from the game model.  Missing values
    are deliberately ignored rather than inferred from HP changes, counters or
    descriptions: a healing action must never be mislabeled as a revival.
    """

    if not isinstance(value, list | tuple):
        return {}
    result: dict[str, bool] = {}
    for item in value:
        if not isinstance(item, Mapping):
            continue
        raw_id = item.get("id", item.get("relic_id"))
        used = item.get("is_used_up")
        if raw_id is None or not isinstance(used, bool):
            continue
        result[str(raw_id)] = used
    return result


def _newly_used_relics(before: Any, after: Any) -> tuple[str, ...]:
    before_state = _relic_used_state(before)
    after_state = _relic_used_state(after)
    return tuple(
        sorted(
            relic_id
            for relic_id, is_used in after_state.items()
            if is_used and before_state.get(relic_id) is False
        )
    )


def _enemy_hp_total(observation: Mapping[str, Any]) -> float:
    enemies = _mapping(observation.get("combat")).get("enemies")
    if not isinstance(enemies, list | tuple):
        return 0.0
    total = 0.0
    for enemy in enemies:
        if not isinstance(enemy, Mapping):
            continue
        hp = _number(enemy.get("hp", enemy.get("current_hp")))
        if hp >= 0.0:
            total += hp
    return total


def derive_transition_facts(
    before_observation: Mapping[str, Any] | None,
    after_observation: Mapping[str, Any],
    *,
    terminated: bool = False,
    terminal_reason: str | None = None,
) -> TransitionFacts:
    before = before_observation if isinstance(before_observation, Mapping) else {}
    after = after_observation if isinstance(after_observation, Mapping) else {}
    before_player = _player_from_observation(before)
    after_player = _player_from_observation(after)
    before_run = _mapping(before.get("run"))
    after_run = _mapping(after.get("run"))

    before_deck = _entity_ids(before_player.get("deck", before.get("deck")))
    after_deck = _entity_ids(after_player.get("deck", after.get("deck")))
    before_potions = _entity_ids(before_player.get("potions", before.get("potions")))
    after_potions = _entity_ids(after_player.get("potions", after.get("potions")))
    before_hp = _number(before_player.get("hp", before_player.get("current_hp")))
    after_hp = _number(after_player.get("hp", after_player.get("current_hp")))
    before_gold = _number(before_player.get("gold", before_run.get("gold")))
    after_gold = _number(after_player.get("gold", after_run.get("gold")))
    before_floor = int(_number(before_run.get("floor", before.get("floor"))))
    after_floor = int(_number(after_run.get("floor", after.get("floor"))))

    reason = str(terminal_reason or "").lower()
    combat_result: CombatResult = "none"
    if terminated:
        if after_hp <= 0.0 or any(token in reason for token in ("death", "defeat", "died")):
            combat_result = "defeat"
        elif "escape" in reason:
            combat_result = "escaped"
        else:
            combat_result = "victory"
    return TransitionFacts(
        hp_delta=after_hp - before_hp,
        gold_delta=after_gold - before_gold,
        floor_delta=after_floor - before_floor,
        cards_added=_multiset_delta(before_deck, after_deck),
        cards_removed=_multiset_delta(after_deck, before_deck),
        potions_added=_multiset_delta(before_potions, after_potions),
        potions_removed=_multiset_delta(after_potions, before_potions),
        relics_used=_newly_used_relics(
            before_player.get("relics"),
            after_player.get("relics"),
        ),
        room_entered=(
            str(after_run.get("room_type") or after.get("room_type"))
            if (after_run.get("room_type") or after.get("room_type")) is not None
            else None
        ),
        combat_result=combat_result,
        terminal_reason=terminal_reason,
        enemy_hp_delta=_enemy_hp_total(before) - _enemy_hp_total(after),
    )


__all__ = ["CombatResult", "TransitionFacts", "derive_transition_facts"]
