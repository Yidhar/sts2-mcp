from __future__ import annotations

from typing import Any

from sts2_rl.semantics.forward import forward_decision


def _card(
    card_id: str,
    *,
    copy: int = 1,
    upgradable: bool = True,
    removable: bool = True,
) -> dict[str, Any]:
    return {
        "id": card_id,
        "instance_id": f"{card_id}-{copy}",
        "index": copy,
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
        {
            "kind": "choose_rest_option",
            "model_action_kind": "rest_site",
            "model_action_variant": "rest",
            "idx": 0,
            "option": {"type": "heal", "id": "HEAL"},
        },
        {
            "kind": "choose_rest_option",
            "model_action_kind": "rest_site",
            "model_action_variant": "forge",
            "idx": 1,
            "option": {"type": "smith", "id": "SMITH"},
        },
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
    assert {c.semantic_action["card"]["id"] for c in smith} == {
        "CARD.BASH",
        "CARD.ANGER",
    }
    # Native dispatch enters the same forge option, but the model scores two
    # independent target-bound candidates.
    assert {c.native_index for c in smith} == {1}
    # Composite executor plan: entry -> select target -> confirm.
    assert [step.kind for step in smith[0].plan] == [
        "choose_rest_option",
        "select_card",
        "confirm_selection",
    ]
    rest = [c for c in decision.candidates if c.branch == "rest"]
    assert len(rest) == 1 and len(rest[0].plan) == 1


def test_rest_surface_merges_only_exact_equal_smith_targets() -> None:
    exact = _card("CARD.BASH", copy=1)
    physical_copy = _card("CARD.BASH", copy=2)
    distinct = _card("CARD.BASH", copy=3)
    distinct["floor_added_to_deck"] = 7
    # Physical instance/index fields do not split one strategic action.  A
    # permanent visible card fact still does.
    deck = [dict(exact), dict(exact), physical_copy, distinct]
    decision = forward_decision(_rest_observation(deck), _rest_actions())
    assert decision is not None
    smith = [candidate for candidate in decision.candidates if candidate.branch == "smith"]
    assert len(smith) == 2
    assert len({candidate.target_key for candidate in smith}) == 2
    assert all("instance_id" not in candidate.target for candidate in smith if candidate.target)
    assert {candidate.target.get("floor_added_to_deck") for candidate in smith if candidate.target} == {
        None,
        7,
    }


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
            "model_action_kind": "shop",
            "model_action_variant": "purchase",
            "idx": 0,
            "item": {"category": "relic", "cost": 150, "can_afford": True},
        },
        {
            "kind": "shop_purchase",
            "model_action_kind": "shop",
            "model_action_variant": "purchase",
            "idx": 1,
            "item": {"category": "card_removal", "cost": 75, "can_afford": True},
        },
        {
            "kind": "proceed",
            "model_action_kind": "shop",
            "model_action_variant": "leave",
            "idx": 2,
        },
    ]
    decision = forward_decision(observation, actions)
    assert decision is not None and decision.surface == "shop"
    assert set(decision.branches) == {"buy_relic", "remove", "leave"}
    removes = [c for c in decision.candidates if c.branch == "remove"]
    # is_removable=False cards are excluded from Remove targets.
    assert len(removes) == 1 and removes[0].target is not None
    assert removes[0].target["id"] == "CARD.STRIKE"
    assert removes[0].semantic_action["card"]["id"] == "CARD.STRIKE"
    assert [step.kind for step in removes[0].plan] == [
        "shop_purchase",
        "select_card",
        "confirm_selection",
    ]
    leaves = [c for c in decision.candidates if c.branch == "leave"]
    assert len(leaves) == 1


def test_shop_surface_merges_only_exact_equal_remove_targets() -> None:
    exact = _card("CARD.STRIKE", copy=1)
    physical_copy = _card("CARD.STRIKE", copy=2)
    distinct = _card("CARD.STRIKE", copy=3)
    distinct["floor_added_to_deck"] = 9
    observation = {
        "phase": "shop",
        "combat": {"in_progress": False},
        "player": {"deck": [dict(exact), dict(exact), physical_copy, distinct]},
        "shop": {"is_open": True},
    }
    actions = [
        {
            "kind": "shop_purchase",
            "model_action_kind": "shop",
            "model_action_variant": "purchase",
            "idx": 0,
            "item": {"category": "card_removal", "cost": 75, "can_afford": True},
        }
    ]
    decision = forward_decision(observation, actions)
    assert decision is not None
    removes = [candidate for candidate in decision.candidates if candidate.branch == "remove"]
    assert len(removes) == 2
    assert all("instance_id" not in candidate.target for candidate in removes if candidate.target)
    assert {candidate.target.get("floor_added_to_deck") for candidate in removes if candidate.target} == {
        None,
        9,
    }


def test_same_branch_items_with_different_semantics_are_not_folded() -> None:
    observation = {
        "phase": "shop",
        "combat": {"in_progress": False},
        "player": {"deck": []},
        "shop": {"is_open": True},
    }
    actions = [
        {
            "kind": "shop_purchase",
            "model_action_kind": "shop",
            "idx": 0,
            "item": {"id": "ITEM.ONE", "name": "one"},
        },
        {
            "kind": "shop_purchase",
            "model_action_kind": "shop",
            "idx": 1,
            "item": {"id": "ITEM.TWO", "name": "two"},
        },
    ]
    decision = forward_decision(observation, actions)
    assert decision is not None
    buys = [candidate for candidate in decision.candidates if candidate.branch == "buy"]
    assert len(buys) == 2


def test_reward_map_event_are_native_atomic() -> None:
    observation = {"phase": "reward", "combat": {"in_progress": False}}
    actions = [
        {
            "kind": "select_card_reward",
            "model_action_kind": "card_reward",
            "idx": 0,
            "card": _card("CARD.HAVOC"),
        },
        {
            "kind": "select_card_reward",
            "model_action_kind": "card_reward",
            "idx": 1,
            "card": _card("CARD.RAGE"),
        },
        {"kind": "skip_card_reward", "model_action_kind": "card_reward", "idx": 2},
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


def test_non_shop_proceed_is_not_misclassified_as_leave_shop() -> None:
    observation = {
        "phase": "event",
        "state_type": "event",
        "combat": {"in_progress": False},
    }
    actions = [
        {"kind": "proceed", "model_action_kind": "proceed", "model_action_variant": "continue"}
    ]
    assert forward_decision(observation, actions) is None


def test_state_type_shop_with_inventory_is_recognized() -> None:
    observation = {
        "state_type": "shop",
        "combat": {"in_progress": False},
        "player": {"deck": []},
        "shop": {"items": []},
    }
    actions = [
        {"kind": "proceed", "model_action_kind": "shop", "model_action_variant": "leave"}
    ]
    decision = forward_decision(observation, actions)
    assert decision is not None and decision.surface == "shop"
    assert decision.branches == ("leave",)


def test_combat_selection_is_monotone_add_or_commit() -> None:
    observation = {"phase": "combat", "combat": {"in_progress": True}}
    actions = [
        {
            "kind": "select_card",
            "model_action_kind": "card_selection",
            "model_action_variant": "select",
            "selection_operation": "select",
            "card": _card("CARD.A", copy=1),
        },
        {
            "kind": "deselect_card",
            "model_action_kind": "card_selection",
            "model_action_variant": "deselect",
            "selection_operation": "deselect",
            "card": _card("CARD.A", copy=1),
        },
        {
            "kind": "confirm_selection",
            "model_action_kind": "card_selection",
            "model_action_variant": "confirm",
        },
        {
            "kind": "cancel_selection",
            "model_action_kind": "card_selection",
            "model_action_variant": "cancel",
        },
    ]
    decision = forward_decision(observation, actions, control_domain="combat")
    assert decision is not None and decision.surface == "combat"
    assert [candidate.branch for candidate in decision.candidates] == [
        "selection_add",
        "selection_commit",
    ]
    assert [candidate.native_index for candidate in decision.candidates] == [0, 2]


def test_control_domains_never_share_surfaces() -> None:
    combat_observation = {"phase": "combat", "combat": {"in_progress": True}}
    combat_actions = [{"kind": "end_turn", "model_action_kind": "end_turn"}]
    assert forward_decision(combat_observation, combat_actions) is None
    assert (
        forward_decision(combat_observation, combat_actions, control_domain="combat")
        is not None
    )
    assert (
        forward_decision(_rest_observation([]), _rest_actions(), control_domain="combat")
        is None
    )
