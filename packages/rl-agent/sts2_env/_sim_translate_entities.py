"""Structural entity translation for the headless simulator transport.

The functions in this module rename protocol fields and copy simulator DTOs.
They do not consult a card registry, infer effects, classify enemies, estimate
damage, or add action-quality features.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from ._sim_translate_route import _translate_route_point


def _qualified_id(prefix: str, value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    expected = f"{prefix}."
    return raw if raw.upper().startswith(expected) else f"{expected}{raw}"


def _translate_power(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise TypeError("simulator power must be a mapping")
    power = deepcopy(dict(raw))
    power_id = raw.get("id", raw.get("power_id"))
    if power_id is not None:
        power["id"] = _qualified_id("POWER", power_id)
    if raw.get("name") is not None and raw.get("title") is None:
        power["title"] = str(raw["name"])
    return power


def _translate_powers(raw: Any) -> list[dict[str, Any]]:
    if raw is None:
        return []
    if not isinstance(raw, list | tuple):
        raise TypeError("simulator powers must be a sequence")
    return [_translate_power(item) for item in raw]


def _translate_card(
    sim_card: Any,
    *,
    pile: str | None = None,
    selection_membership: str | None = None,
) -> dict[str, Any]:
    if not isinstance(sim_card, Mapping):
        raise TypeError("simulator card must be a mapping")
    card = deepcopy(dict(sim_card))
    raw_id = sim_card.get("id", sim_card.get("card_id"))
    if raw_id is not None:
        card["id"] = _qualified_id("CARD", raw_id)
    card.pop("card_id", None)
    if sim_card.get("name") is not None and sim_card.get("title") is None:
        card["title"] = str(sim_card["name"])
    # A selection is not a physical card pile.  Native selection DTOs retain
    # the real source pile (Hand/Discard/Draw/Deck/Exhaust) separately from
    # whether the card is currently selected.  Prefer that native source over
    # a caller-provided fallback and never overwrite it with Select/Selected.
    source_pile = sim_card.get("source_pile", sim_card.get("pile"))
    resolved_pile = source_pile if source_pile is not None else pile
    if resolved_pile is not None and str(resolved_pile).strip():
        card["pile"] = str(resolved_pile)
        card["source_pile"] = str(resolved_pile)
    if selection_membership is not None:
        membership = str(selection_membership).strip().lower()
        if membership not in {"selectable", "selected"}:
            raise ValueError(f"invalid card selection membership: {selection_membership!r}")
        card["selection_membership"] = membership
        card["is_selected"] = membership == "selected"
    return card


def _translate_cards(raw: Any, *, pile: str) -> list[dict[str, Any]]:
    if raw is None:
        return []
    if not isinstance(raw, list | tuple):
        raise TypeError(f"simulator {pile} cards must be a sequence")
    return [_translate_card(item, pile=pile) for item in raw]


def _translate_relic(sim_relic: Any) -> dict[str, Any]:
    if not isinstance(sim_relic, Mapping):
        raise TypeError("simulator relic must be a mapping")
    relic = deepcopy(dict(sim_relic))
    raw_id = sim_relic.get("id", sim_relic.get("relic_id"))
    if raw_id is not None:
        relic["id"] = _qualified_id("RELIC", raw_id)
    relic.pop("relic_id", None)
    if sim_relic.get("name") is not None and sim_relic.get("title") is None:
        relic["title"] = str(sim_relic["name"])
    return relic


def _translate_potion(sim_potion: Any) -> dict[str, Any]:
    if not isinstance(sim_potion, Mapping):
        raise TypeError("simulator potion must be a mapping")
    potion = deepcopy(dict(sim_potion))
    raw_id = sim_potion.get("id", sim_potion.get("potion_id"))
    if raw_id is not None:
        potion["id"] = _qualified_id("POTION", raw_id)
    potion.pop("potion_id", None)
    if sim_potion.get("name") is not None and sim_potion.get("title") is None:
        potion["title"] = str(sim_potion["name"])
    return potion


def _translate_player(sim_player: Mapping[str, Any]) -> dict[str, Any]:
    """Translate the player DTO while retaining only one canonical view."""

    player = deepcopy(dict(sim_player))
    hp = sim_player.get("current_hp", sim_player.get("hp"))
    if hp is not None:
        player["hp"] = int(hp)
    player.pop("current_hp", None)

    zone_fields = {
        "deck": "Deck",
        "hand": "Hand",
        "draw_pile": "Draw",
        "discard_pile": "Discard",
        "exhaust_pile": "Exhaust",
        "play_pile": "Play",
    }
    for field, pile in zone_fields.items():
        if field in sim_player:
            player[field] = _translate_cards(sim_player.get(field), pile=pile)
    if "relics" in sim_player:
        raw_relics = sim_player.get("relics") or []
        if not isinstance(raw_relics, list | tuple):
            raise TypeError("simulator relics must be a sequence")
        player["relics"] = [_translate_relic(item) for item in raw_relics]
    if "potions" in sim_player:
        raw_potions = sim_player.get("potions") or []
        if not isinstance(raw_potions, list | tuple):
            raise TypeError("simulator potions must be a sequence")
        player["potions"] = [_translate_potion(item) for item in raw_potions]
    if "status" in sim_player:
        player["powers"] = _translate_powers(sim_player.get("status"))
        player.pop("status", None)
    return player


def _translate_enemy(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise TypeError("simulator enemy must be a mapping")
    enemy = deepcopy(dict(raw))
    entity_id = raw.get("entity_id", raw.get("model_id"))
    if entity_id is not None:
        enemy["model_id"] = _qualified_id("MONSTER", entity_id)
    hp = raw.get("hp", raw.get("current_hp"))
    if hp is not None:
        enemy["hp"] = int(hp)
    enemy.pop("current_hp", None)
    if "status" in raw:
        enemy["powers"] = _translate_powers(raw.get("status"))
        enemy.pop("status", None)
    if "intents" in raw:
        intents = raw.get("intents") or []
        if not isinstance(intents, list | tuple):
            raise TypeError("simulator enemy intents must be a sequence")
        enemy["intents"] = [
            deepcopy(dict(intent))
            for intent in intents
            if isinstance(intent, Mapping)
        ]
        if len(enemy["intents"]) != len(intents):
            raise TypeError("simulator enemy intent must be a mapping")
    return enemy


def _translate_combat_block(
    battle: Mapping[str, Any],
    *,
    in_progress: bool,
) -> dict[str, Any]:
    combat = deepcopy(dict(battle))
    combat.pop("player", None)
    combat.pop("card_selection", None)
    enemies = battle.get("enemies") or []
    if not isinstance(enemies, list | tuple):
        raise TypeError("simulator enemies must be a sequence")
    combat["enemies"] = [_translate_enemy(enemy) for enemy in enemies]
    combat["in_progress"] = bool(in_progress)
    return combat


def _translate_map_block(map_state: Mapping[str, Any]) -> dict[str, Any]:
    translated = deepcopy(dict(map_state))
    translated.pop("player", None)
    for field in ("next_options", "nodes", "points"):
        if field not in map_state:
            continue
        values = map_state.get(field) or []
        if not isinstance(values, list | tuple):
            raise TypeError(f"simulator map {field} must be a sequence")
        translated[field] = [_translate_route_point(item) for item in values]
    return translated


def _translate_run_block(sim_run: Mapping[str, Any]) -> dict[str, Any]:
    run = deepcopy(dict(sim_run))
    run.pop("player", None)
    return run


__all__: list[str] = []
