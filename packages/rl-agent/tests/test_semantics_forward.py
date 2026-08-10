from __future__ import annotations

from typing import Any

from sts2_rl.semantics.forward import forward_decision


def _card(card_id: str, *, upgradable: bool = True, removable: bool = True) -> dict[str, Any]:
    return {
        "id": card_id,
        "instance_id": f"{card_id}-1",
        "cost": 1,
        "is_upgraded": False,
        "is_upgradable": upgradable,
        "is_removable": removable,
        "type": "Attack",
    }


def _rest_observation(deck: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "phase": "rest",
        "combat": {"in_progress": False},
        "player": {"deck": deck},
        "rest_site": {"options": []},
    }


def _rest_actions() -> list[dict[str, Any]]:
    return [
        {"kind": "choose_rest_option", "idx": 0, "option": {"type": "heal", "id": "HEAL"}},
        {"kind": "choose_rest_option", "idx": 1, "option": {"type": "smith", "id": "SMITH"}},
    ]


def test_rest_surface_expands_smith_targets_from_deck_facts() -> None:
    deck = [_card("CARD.BASH"), _card("CARD.DEFEND", upgradable=False), _card("CARD.ANGER")]
    decision = forward_decision(_rest_observation(deck), _rest_actions())
    assert decision is not None and decision.surface == "rest"
    assert decision.branches == ("rest", "smith")
    smith = [c for c in decision.candidates if c.branch == "smith"]
    # Only upgradable cards become Smith targets.
    assert len(smith) == 2
    assert all(c.target is not None and c.target["is_upgradable"] for c in smith)
    # Composite executor plan: entry -> select target -> confirm.
    assert [step.kind for step in smith[0].plan] == [
        "choose_rest_option",
        "select_card",
        "confirm_selection",
    ]
    rest = [c for c in decision.candidates if c.branch == "rest"]
    assert len(rest) == 1 and len(rest[0].plan) == 1


def test_shop_surface_buy_remove_leave_branches() -> None:
    deck = [_card("CARD.STRIKE"), _card("CARD.CURSE", removable=False)]
    observation = {
        "phase": "shop",
        "combat": {"in_progress": False},
        "player": {"deck": deck},
        "shop": {"is_open": True},
    }
    actions = [
        {
            "kind": "shop_purchase",
            "idx": 0,
            "item": {"category": "relic", "cost": 150, "can_afford": True},
        },
        {
            "kind": "shop_purchase",
            "idx": 1,
            "item": {"category": "card_removal", "cost": 75, "can_afford": True},
        },
        {"kind": "proceed", "idx": 2},
    ]
    decision = forward_decision(observation, actions)
    assert decision is not None and decision.surface == "shop"
    assert set(decision.branches) == {"buy_relic", "remove", "leave"}
    removes = [c for c in decision.candidates if c.branch == "remove"]
    # is_removable=False cards are excluded from Remove targets.
    assert len(removes) == 1 and removes[0].target is not None
    assert removes[0].target["id"] == "CARD.STRIKE"
    assert [step.kind for step in removes[0].plan] == [
        "shop_purchase",
        "select_card",
        "confirm_selection",
    ]
    leaves = [c for c in decision.candidates if c.branch == "leave"]
    assert len(leaves) == 1


def test_reward_map_event_are_native_atomic() -> None:
    observation = {"phase": "reward", "combat": {"in_progress": False}}
    actions = [
        {"kind": "select_card_reward", "idx": 0, "card": _card("CARD.HAVOC")},
        {"kind": "select_card_reward", "idx": 1, "card": _card("CARD.RAGE")},
        {"kind": "skip_card_reward", "idx": 2},
    ]
    decision = forward_decision(observation, actions)
    assert decision is not None and decision.surface == "reward"
    assert decision.branches == ("take", "skip")
    takes = [c for c in decision.candidates if c.branch == "take"]
    assert len(takes) == 2 and all(c.target is not None for c in takes)


def test_combat_and_unknown_surfaces_fail_closed() -> None:
    combat_observation = {"phase": "combat", "combat": {"in_progress": True}}
    assert forward_decision(combat_observation, [{"kind": "play_card"}]) is None
    unknown = {"phase": "mystery", "combat": {"in_progress": False}}
    assert forward_decision(unknown, [{"kind": "mystery_action"}]) is None
