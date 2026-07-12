from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _starter_deck(strikes: int = 4, defends: int = 4) -> list[dict]:
    cards: list[dict] = []
    for _ in range(strikes):
        cards.append({"id": "CARD.STRIKE_IRONCLAD", "title": "打击", "type": "Attack", "cost": 1})
    for _ in range(defends):
        cards.append({"id": "CARD.DEFEND_IRONCLAD", "title": "防御", "type": "Skill", "cost": 1})
    return cards


def _trainer(raw_obs: dict, full_actions: list[dict]):
    from muzero.train import MuZeroTrainer

    trainer = MuZeroTrainer.__new__(MuZeroTrainer)
    trainer.build_hard_guard_policy = "full"
    trainer.env = SimpleNamespace(
        unwrapped=SimpleNamespace(
            _last_obs_raw=raw_obs,
            _legal_actions=full_actions,
        )
    )
    return trainer


def test_shop_guard_opens_inventory_before_leaving_closed_shop_with_gold():
    raw_obs = {"player": {"hp": 60, "max_hp": 80, "gold": 120, "deck_cards": _starter_deck()}}
    actions = [
        {"kind": "shop", "action_id": "shop:leave", "shop_action": "leave"},
        {"kind": "shop", "action_id": "shop:open", "shop_action": "open"},
    ]
    trainer = _trainer(raw_obs, actions)
    stats: dict = {}

    new_idx = trainer._apply_build_action_hard_guards(
        action_idx=0,
        legal_actions=actions,
        action_mask=np.array([1, 1], dtype=np.float32),
        search_stats=stats,
    )

    assert new_idx == 1
    assert stats["shop_action_guard_shop_surface"] == 1.0
    assert stats["shop_action_guard_open_available"] == 1.0
    assert stats["shop_action_guard_open_applicable"] == 1.0
    assert stats["shop_action_guard_open_applied"] == 1.0


def test_shop_guard_opens_inventory_before_backing_out_of_closed_shop_with_gold():
    raw_obs = {"player": {"hp": 60, "max_hp": 80, "gold": 120, "deck_cards": _starter_deck()}}
    actions = [
        {"kind": "shop", "action_id": "shop:back", "shop_action": "back"},
        {"kind": "shop", "action_id": "shop:open", "shop_action": "open"},
    ]
    trainer = _trainer(raw_obs, actions)
    stats: dict = {}

    new_idx = trainer._apply_build_action_hard_guards(
        action_idx=0,
        legal_actions=actions,
        action_mask=np.array([1, 1], dtype=np.float32),
        search_stats=stats,
    )

    assert new_idx == 1
    assert stats["shop_action_guard_shop_surface"] == 1.0
    assert stats["shop_action_guard_selected_back"] == 1.0
    assert stats["shop_action_guard_open_available"] == 1.0
    assert stats["shop_action_guard_open_applicable"] == 1.0
    assert stats["shop_action_guard_open_applied"] == 1.0


def test_shop_guard_does_not_open_when_gold_too_low():
    raw_obs = {"player": {"hp": 60, "max_hp": 80, "gold": 0, "deck_cards": _starter_deck()}}
    actions = [
        {"kind": "shop", "action_id": "shop:leave", "shop_action": "leave"},
        {"kind": "shop", "action_id": "shop:open", "shop_action": "open"},
    ]
    trainer = _trainer(raw_obs, actions)
    stats: dict = {}

    new_idx = trainer._apply_build_action_hard_guards(
        action_idx=0,
        legal_actions=actions,
        action_mask=np.array([1, 1], dtype=np.float32),
        search_stats=stats,
    )

    assert new_idx == 0
    assert stats["shop_action_guard_open_available"] == 1.0
    assert stats["shop_action_guard_open_applied"] == 0.0


def test_shop_guard_removes_starter_when_leaving_with_affordable_removal():
    raw_obs = {"player": {"hp": 60, "max_hp": 80, "gold": 125, "deck_cards": _starter_deck()}}
    actions = [
        {"kind": "shop", "action_id": "shop:leave", "shop_action": "leave"},
        {
            "kind": "shop",
            "action_id": "shop:buy:2",
            "shop_action": "buy",
            "item": {"item_kind": "card_removal", "title": "Remove a card", "cost": 75, "is_affordable": True},
        },
    ]
    trainer = _trainer(raw_obs, actions)
    stats: dict = {}

    new_idx = trainer._apply_build_action_hard_guards(
        action_idx=0,
        legal_actions=actions,
        action_mask=np.array([1, 1], dtype=np.float32),
        search_stats=stats,
    )

    assert new_idx == 1
    assert stats["shop_action_guard_remove_affordable_available"] == 1.0
    assert stats["shop_action_guard_starter_count"] == 8.0
    assert stats["shop_action_guard_remove_applicable"] == 1.0
    assert stats["shop_action_guard_remove_applied"] == 1.0


def test_shop_guard_removes_starter_with_flattened_compact_removal_action():
    raw_obs = {"player": {"hp": 60, "max_hp": 80, "gold": 125, "deck_cards": _starter_deck()}}
    actions = [
        {"kind": "shop", "action_id": "shop:back", "shop_action": "back"},
        {
            "kind": "shop",
            "action_id": "shop:buy:2",
            "shop_action": "buy",
            "shop_item_kind": "card_removal",
            "shop_item_title": "Remove a card",
            "shop_item_cost": 75,
            "shop_item_affordable": True,
        },
    ]
    trainer = _trainer(raw_obs, actions)
    stats: dict = {}

    new_idx = trainer._apply_build_action_hard_guards(
        action_idx=0,
        legal_actions=actions,
        action_mask=np.array([1, 1], dtype=np.float32),
        search_stats=stats,
    )

    assert new_idx == 1
    assert stats["shop_action_guard_remove_affordable_available"] == 1.0
    assert stats["shop_action_guard_remove_applicable"] == 1.0
    assert stats["shop_action_guard_remove_applied"] == 1.0


def test_shop_guard_does_not_replace_actual_purchase():
    raw_obs = {"player": {"hp": 60, "max_hp": 80, "gold": 125, "deck_cards": _starter_deck()}}
    actions = [
        {
            "kind": "shop",
            "action_id": "shop:buy:0",
            "shop_action": "buy",
            "item": {"item_kind": "card", "title": "Pommel Strike", "cost": 50, "is_affordable": True},
        },
        {
            "kind": "shop",
            "action_id": "shop:buy:2",
            "shop_action": "buy",
            "item": {"item_kind": "card_removal", "title": "Remove a card", "cost": 75, "is_affordable": True},
        },
        {"kind": "shop", "action_id": "shop:leave", "shop_action": "leave"},
    ]
    trainer = _trainer(raw_obs, actions)
    stats: dict = {}

    new_idx = trainer._apply_build_action_hard_guards(
        action_idx=0,
        legal_actions=actions,
        action_mask=np.array([1, 1, 1], dtype=np.float32),
        search_stats=stats,
    )

    assert new_idx == 0
    assert stats["shop_action_guard_selected_buy"] == 1.0
    assert stats["shop_action_guard_remove_applied"] == 0.0


def test_shop_guard_replaces_ordinary_card_buy_that_blocks_affordable_remove():
    raw_obs = {"player": {"hp": 60, "max_hp": 80, "gold": 139, "deck_cards": _starter_deck(strikes=5, defends=4)}}
    actions = [
        {
            "kind": "shop",
            "action_id": "shop:buy:0",
            "shop_action": "buy",
            "item": {"item_kind": "card", "title": "燃烧", "cost": 75, "is_affordable": True},
        },
        {
            "kind": "shop",
            "action_id": "shop:buy:10",
            "shop_action": "buy",
            "item": {"item_kind": "card_removal", "title": "Remove a card", "cost": 75, "is_affordable": True},
        },
        {"kind": "shop", "action_id": "shop:back", "shop_action": "back"},
    ]
    trainer = _trainer(raw_obs, actions)
    stats: dict = {}

    new_idx = trainer._apply_build_action_hard_guards(
        action_idx=0,
        legal_actions=actions,
        action_mask=np.array([1, 1, 1], dtype=np.float32),
        search_stats=stats,
    )

    assert new_idx == 1
    assert stats["shop_action_guard_selected_buy"] == 1.0
    assert stats["shop_action_guard_starter_count"] == 9.0
    assert stats["shop_action_guard_selected_buy_cost"] == 75.0
    assert stats["shop_action_guard_remove_cost_min"] == 75.0
    assert stats["shop_action_guard_gold_after_selected_buy"] == 64.0
    assert stats["shop_action_guard_buy_blocks_remove_applicable"] == 1.0
    assert stats["shop_action_guard_buy_blocks_remove_applied"] == 1.0
    assert stats["shop_action_guard_remove_applied"] == 1.0


def test_shop_guard_does_not_replace_card_buy_when_remove_still_affordable_afterwards():
    raw_obs = {"player": {"hp": 60, "max_hp": 80, "gold": 165, "deck_cards": _starter_deck(strikes=5, defends=4)}}
    actions = [
        {
            "kind": "shop",
            "action_id": "shop:buy:0",
            "shop_action": "buy",
            "item": {"item_kind": "card", "title": "杂耍", "cost": 75, "is_affordable": True},
        },
        {
            "kind": "shop",
            "action_id": "shop:buy:10",
            "shop_action": "buy",
            "item": {"item_kind": "card_removal", "title": "Remove a card", "cost": 75, "is_affordable": True},
        },
    ]
    trainer = _trainer(raw_obs, actions)
    stats: dict = {}

    new_idx = trainer._apply_build_action_hard_guards(
        action_idx=0,
        legal_actions=actions,
        action_mask=np.array([1, 1], dtype=np.float32),
        search_stats=stats,
    )

    assert new_idx == 0
    assert stats["shop_action_guard_gold_after_selected_buy"] == 90.0
    assert stats["shop_action_guard_buy_blocks_remove_applicable"] == 0.0
    assert stats["shop_action_guard_buy_blocks_remove_applied"] == 0.0
    assert stats["shop_action_guard_remove_applied"] == 0.0


def test_shop_guard_does_not_replace_premium_card_buy_even_if_it_blocks_remove():
    raw_obs = {"player": {"hp": 60, "max_hp": 80, "gold": 120, "deck_cards": _starter_deck(strikes=5, defends=4)}}
    actions = [
        {
            "kind": "shop",
            "action_id": "shop:buy:0",
            "shop_action": "buy",
            "item": {"item_kind": "card", "title": "耸肩无视", "cost": 75, "is_affordable": True},
        },
        {
            "kind": "shop",
            "action_id": "shop:buy:10",
            "shop_action": "buy",
            "item": {"item_kind": "card_removal", "title": "Remove a card", "cost": 75, "is_affordable": True},
        },
    ]
    trainer = _trainer(raw_obs, actions)
    stats: dict = {}

    new_idx = trainer._apply_build_action_hard_guards(
        action_idx=0,
        legal_actions=actions,
        action_mask=np.array([1, 1], dtype=np.float32),
        search_stats=stats,
    )

    assert new_idx == 0
    assert stats["shop_action_guard_buy_blocks_remove_applicable"] == 1.0
    assert stats["shop_action_guard_buy_blocks_remove_premium_allow"] == 1.0
    assert stats["shop_action_guard_buy_blocks_remove_applied"] == 0.0
    assert stats["shop_action_guard_remove_applied"] == 0.0


def test_shop_guard_does_not_replace_relic_purchase_that_blocks_remove():
    raw_obs = {"player": {"hp": 60, "max_hp": 80, "gold": 139, "deck_cards": _starter_deck(strikes=5, defends=4)}}
    actions = [
        {
            "kind": "shop",
            "action_id": "shop:buy:0",
            "shop_action": "buy",
            "item": {"item_kind": "relic", "title": "Anchor", "cost": 75, "is_affordable": True},
        },
        {
            "kind": "shop",
            "action_id": "shop:buy:10",
            "shop_action": "buy",
            "item": {"item_kind": "card_removal", "title": "Remove a card", "cost": 75, "is_affordable": True},
        },
    ]
    trainer = _trainer(raw_obs, actions)
    stats: dict = {}

    new_idx = trainer._apply_build_action_hard_guards(
        action_idx=0,
        legal_actions=actions,
        action_mask=np.array([1, 1], dtype=np.float32),
        search_stats=stats,
    )

    assert new_idx == 0
    assert stats["shop_action_guard_selected_buy"] == 1.0
    assert stats["shop_action_guard_buy_blocks_remove_applicable"] == 0.0
    assert stats["shop_action_guard_remove_applied"] == 0.0


def test_shop_guard_does_not_fire_on_route_action_to_future_shop():
    raw_obs = {"player": {"hp": 60, "max_hp": 80, "gold": 150, "deck_cards": _starter_deck()}}
    actions = [
        {
            "kind": "map",
            "action_id": "map:7:1",
            "room_type": "Shop",
            "route_summary": {"next_room_type": "shop", "shop_count": 1},
            "semantic": {"family": "shop", "domain": "route"},
        },
        {
            "kind": "map",
            "action_id": "map:7:2",
            "room_type": "Monster",
            "semantic": {"family": "map", "domain": "route"},
        },
    ]
    trainer = _trainer(raw_obs, actions)
    stats: dict = {}

    new_idx = trainer._apply_build_action_hard_guards(
        action_idx=0,
        legal_actions=actions,
        action_mask=np.array([1, 1], dtype=np.float32),
        search_stats=stats,
    )

    assert new_idx == 0
    assert stats["shop_action_guard_shop_surface"] == 0.0
    assert stats["shop_action_guard_open_applied"] == 0.0
    assert stats["shop_action_guard_remove_applied"] == 0.0


def test_shop_guard_fails_open_when_gold_missing():
    raw_obs = {"player": {"hp": 60, "max_hp": 80, "deck_cards": _starter_deck()}}
    actions = [
        {"kind": "shop", "action_id": "shop:leave", "shop_action": "leave"},
        {"kind": "shop", "action_id": "shop:open", "shop_action": "open"},
    ]
    trainer = _trainer(raw_obs, actions)
    stats: dict = {}

    new_idx = trainer._apply_build_action_hard_guards(
        action_idx=0,
        legal_actions=actions,
        action_mask=np.array([1, 1], dtype=np.float32),
        search_stats=stats,
    )

    assert new_idx == 0
    assert stats["shop_action_guard_shop_surface"] == 1.0
    assert stats["shop_action_guard_invalid_obs"] == 1.0
    assert stats["shop_action_guard_open_applied"] == 0.0


def test_shop_guard_does_not_force_remove_after_starters_are_thinned():
    raw_obs = {"player": {"hp": 60, "max_hp": 80, "gold": 125, "deck_cards": _starter_deck(strikes=1, defends=1)}}
    actions = [
        {"kind": "shop", "action_id": "shop:back", "shop_action": "back"},
        {
            "kind": "shop",
            "action_id": "shop:buy:2",
            "shop_action": "buy",
            "item": {"item_kind": "card_removal", "title": "Remove a card", "cost": 75, "is_affordable": True},
        },
    ]
    trainer = _trainer(raw_obs, actions)
    stats: dict = {}

    new_idx = trainer._apply_build_action_hard_guards(
        action_idx=0,
        legal_actions=actions,
        action_mask=np.array([1, 1], dtype=np.float32),
        search_stats=stats,
    )

    assert new_idx == 0
    assert stats["shop_action_guard_starter_count"] == 2.0
    assert stats["shop_action_guard_remove_applicable"] == 0.0
    assert stats["shop_action_guard_remove_applied"] == 0.0


def test_shop_guard_remove_selection_targets_junk_over_cancel():
    raw_obs = {"player": {"hp": 60, "max_hp": 80, "gold": 50, "deck_cards": _starter_deck()}}
    typed = {
        "operation_type": "remove",
        "source": "shop",
        "source_zone": "deck",
        "selection_required": True,
        "confidence": "runtime_internal",
    }
    actions = [
        {"kind": "card_selection", "action_id": "card_selection:cancel", "selection": "cancel"},
        {
            "kind": "card_selection",
            "action_id": "card_selection:0",
            "typed_selection": typed,
            "card": {"id": "CARD.DEFEND_IRONCLAD", "title": "防御", "type": "Skill", "upgrade_level": 0},
        },
        {
            "kind": "card_selection",
            "action_id": "card_selection:1",
            "typed_selection": typed,
            "card": {"id": "CARD.WOUND", "title": "受伤", "type": "Status"},
        },
    ]
    trainer = _trainer(raw_obs, actions)
    stats: dict = {}

    new_idx = trainer._apply_build_action_hard_guards(
        action_idx=0,
        legal_actions=actions,
        action_mask=np.array([1, 1, 1], dtype=np.float32),
        search_stats=stats,
    )

    assert new_idx == 2
    assert stats["shop_action_guard_remove_selection_surface"] == 1.0
    assert stats["shop_action_guard_remove_selection_selected_score"] == 0.0
    assert stats["shop_action_guard_remove_selection_best_score"] == 100.0
    assert stats["shop_action_guard_remove_selection_applied"] == 1.0


def test_shop_guard_remove_selection_keeps_best_starter_target():
    raw_obs = {"player": {"hp": 60, "max_hp": 80, "gold": 50, "deck_cards": _starter_deck()}}
    typed = {
        "operation_type": "remove",
        "source": "shop",
        "source_zone": "deck",
        "selection_required": True,
        "confidence": "runtime_internal",
    }
    actions = [
        {
            "kind": "card_selection",
            "action_id": "card_selection:0",
            "typed_selection": typed,
            "card": {"id": "CARD.STRIKE_IRONCLAD", "title": "打击", "type": "Attack", "upgrade_level": 0},
        },
        {
            "kind": "card_selection",
            "action_id": "card_selection:1",
            "typed_selection": typed,
            "card": {"id": "CARD.DEFEND_IRONCLAD", "title": "防御", "type": "Skill", "upgrade_level": 0},
        },
    ]
    trainer = _trainer(raw_obs, actions)
    stats: dict = {}

    new_idx = trainer._apply_build_action_hard_guards(
        action_idx=0,
        legal_actions=actions,
        action_mask=np.array([1, 1], dtype=np.float32),
        search_stats=stats,
    )

    assert new_idx == 0
    assert stats["shop_action_guard_remove_selection_surface"] == 1.0
    assert stats["shop_action_guard_remove_selection_applied"] == 0.0


def test_shop_guard_remove_selection_does_not_delete_high_value_card():
    raw_obs = {"player": {"hp": 60, "max_hp": 80, "gold": 50, "deck_cards": _starter_deck(strikes=1, defends=1)}}
    typed = {
        "operation_type": "remove",
        "source": "shop",
        "source_zone": "deck",
        "selection_required": True,
        "confidence": "runtime_internal",
    }
    actions = [
        {"kind": "card_selection", "action_id": "card_selection:cancel", "selection": "cancel"},
        {
            "kind": "card_selection",
            "action_id": "card_selection:0",
            "typed_selection": typed,
            "card": {"id": "CARD.BATTLE_TRANCE", "title": "Battle Trance", "type": "Skill", "upgrade_level": 0},
        },
    ]
    trainer = _trainer(raw_obs, actions)
    stats: dict = {}

    new_idx = trainer._apply_build_action_hard_guards(
        action_idx=0,
        legal_actions=actions,
        action_mask=np.array([1, 1], dtype=np.float32),
        search_stats=stats,
    )

    assert new_idx == 0
    assert stats["shop_action_guard_remove_selection_surface"] == 0.0
    assert stats["shop_action_guard_remove_selection_applied"] == 0.0
