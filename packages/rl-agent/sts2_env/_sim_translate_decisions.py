"""Structural decision-screen translation for the headless transport."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from ._sim_translate_entities import _translate_card, _translate_relic


def _mapping_sequence(value: Any, *, label: str) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list | tuple):
        raise TypeError(f"simulator {label} must be a sequence")
    result: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise TypeError(f"simulator {label} item must be a mapping")
        result.append(deepcopy(dict(item)))
    return result


def _translate_event_options(event: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Copy event options without parsing prose into predicted effects."""

    options = _mapping_sequence(event.get("options"), label="event options")
    for option in options:
        if option.get("text") is not None and option.get("label") is None:
            option["label"] = str(option["text"])
        if "is_locked" in option and "is_enabled" not in option:
            option["is_enabled"] = not bool(option["is_locked"])
    return options


def _translate_rewards_block(
    rewards: Mapping[str, Any],
    card_reward: Mapping[str, Any],
    treasure: Mapping[str, Any],
    relic_select: Mapping[str, Any],
) -> dict[str, Any]:
    translated = deepcopy(dict(rewards))
    translated.pop("player", None)
    if "items" in rewards:
        translated["items"] = _mapping_sequence(rewards.get("items"), label="reward items")
    if card_reward:
        translated["card_reward"] = _translate_card_reward_sel_block(card_reward)
    if treasure:
        raw_relics = treasure.get("relics") or []
        if not isinstance(raw_relics, list | tuple):
            raise TypeError("simulator treasure relics must be a sequence")
        translated["treasure"] = {
            key: deepcopy(value)
            for key, value in treasure.items()
            if key not in {"player", "relics"}
        }
        translated["treasure"]["relics"] = [_translate_relic(item) for item in raw_relics]
    if relic_select:
        translated["relic_select"] = {
            key: deepcopy(value)
            for key, value in relic_select.items()
            if key != "player"
        }
    return translated


def _translate_rest_site_block(rest: Mapping[str, Any]) -> dict[str, Any]:
    translated = deepcopy(dict(rest))
    translated.pop("player", None)
    if "options" in rest:
        translated["options"] = _mapping_sequence(rest.get("options"), label="rest options")
    return translated


def _translate_shop_block(shop: Mapping[str, Any]) -> dict[str, Any]:
    translated = deepcopy(dict(shop))
    translated.pop("player", None)
    if "items" in shop:
        translated["items"] = _mapping_sequence(shop.get("items"), label="shop items")
    return translated


def _translate_card_reward_sel_block(card_reward: Mapping[str, Any]) -> dict[str, Any]:
    translated = deepcopy(dict(card_reward))
    translated.pop("player", None)
    raw_cards = card_reward.get("cards") or []
    if not isinstance(raw_cards, list | tuple):
        raise TypeError("simulator card rewards must be a sequence")
    translated["cards"] = [_translate_card(card, pile="Reward") for card in raw_cards]
    return translated


def _translate_card_sel_block(
    card_select: Mapping[str, Any],
    hand_select: Mapping[str, Any],
    combat_card_sel: Mapping[str, Any] | None,
) -> dict[str, Any]:
    source = card_select or hand_select or combat_card_sel or {}
    translated = deepcopy(dict(source))
    translated.pop("player", None)
    for field, pile in (
        ("cards", "Select"),
        ("selectable_cards", "Select"),
        ("selected_cards", "Selected"),
    ):
        if field not in source:
            continue
        raw_cards = source.get(field) or []
        if not isinstance(raw_cards, list | tuple):
            raise TypeError(f"simulator card selection {field} must be a sequence")
        translated[field] = [_translate_card(card, pile=pile) for card in raw_cards]
    return translated


__all__: list[str] = []
