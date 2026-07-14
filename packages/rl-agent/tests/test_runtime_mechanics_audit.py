from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from sts2_rl.runtime_mechanics import (
    RuntimeMechanicsAuditError,
    audit_game_catalog,
)


def _dynamic_var(name: str = "Amount") -> dict[str, Any]:
    return {
        "name": name,
        "var_type": "IntVar",
        "family": "amount",
        "base_value": 1,
        "enchanted_value": 1,
        "preview_value": 1,
        "int_value": 1,
        "was_just_upgraded": False,
    }


def _catalog() -> dict[str, Any]:
    powers = [
        {
            "power_id": f"POWER_{index}",
            "class_name": f"Power{index}",
            "type": "Buff",
            "stack_type": "Counter",
            "is_instanced": False,
            "allow_negative": False,
            "should_scale_in_multiplayer": False,
            "owner_is_secondary_enemy": False,
            "dynamic_vars": [_dynamic_var()] if index == 0 else [],
            "base_classes": ["PowerModel"],
        }
        for index in range(200)
    ]
    relics = [
        {
            "relic_id": f"RELIC_{index}",
            "class_name": f"Relic{index}",
            "rarity": "Common",
            "tags": [],
            "is_tradable": True,
            "is_allowed_in_shops": True,
            "has_upon_pickup_effect": False,
            "spawns_pets": False,
            "is_stackable": False,
            "adds_pet": False,
            "merchant_cost": 100,
            "show_counter": False,
            "display_amount": None,
            "dynamic_vars": [],
        }
        for index in range(200)
    ]
    potions = [
        {
            "potion_id": f"POTION_{index}",
            "class_name": f"Potion{index}",
            "rarity": "Common",
            "usage": "Combat",
            "target_type": "Self",
            "can_be_generated_in_combat": True,
            "passes_custom_usability_check": True,
            "dynamic_vars": [],
        }
        for index in range(50)
    ]
    enchantments = [
        {
            "enchantment_id": f"ENCHANTMENT_{index}",
            "class_name": f"Enchantment{index}",
            "show_amount": False,
            "is_stackable": False,
            "should_start_at_bottom_of_draw_pile": False,
            "should_glow_gold": False,
            "should_glow_red": False,
            "has_extra_card_text": True,
            "dynamic_vars": [],
        }
        for index in range(10)
    ]
    afflictions = [
        {
            "affliction_id": f"AFFLICTION_{index}",
            "class_name": f"Affliction{index}",
            "is_stackable": False,
            "has_extra_card_text": True,
            "can_afflict_unplayable_cards": False,
            "has_overlay": True,
        }
        for index in range(5)
    ]
    events = [
        {
            "event_id": f"EVENT_{index}",
            "class_name": f"Event{index}",
            "layout_type": "Generic",
            "is_deterministic": False,
            "is_shared": False,
            "encounter_id": None,
            "dynamic_vars": [],
        }
        for index in range(40)
    ]
    monsters = [
        {
            "monster_id": f"MONSTER_{index}",
            "class_name": f"Monster{index}",
            "min_initial_hp": 10,
            "max_initial_hp": 12,
            "powers": [],
        }
        for index in range(80)
    ]
    encounters = [
        {
            "encounter_id": f"ENCOUNTER_{index}",
            "room_type": "boss" if index < 5 else "monster",
            "monster_ids": [f"MONSTER_{index}"],
            "act_index": index % 3,
        }
        for index in range(60)
    ]
    return {
        "powers": powers,
        "relics": relics,
        "potions": potions,
        "enchantments": enchantments,
        "afflictions": afflictions,
        "events": events,
        "monsters": monsters,
        "encounters": encounters,
    }


def test_complete_runtime_mechanics_catalog_passes() -> None:
    summary = audit_game_catalog(_catalog())

    assert summary["schema"] == "sts2-runtime-mechanics-audit-v1"
    assert summary["catalog_counts"]["powers"] == 200
    assert summary["boss_encounter_count"] == 5
    assert summary["boss_monster_count"] == 5


@pytest.mark.parametrize("collection", ["powers", "relics", "events", "afflictions"])
def test_missing_mechanics_collection_fails_closed(collection: str) -> None:
    catalog = _catalog()
    del catalog[collection]

    with pytest.raises(RuntimeMechanicsAuditError, match=collection):
        audit_game_catalog(catalog)


def test_handwritten_power_hint_is_rejected() -> None:
    catalog = _catalog()
    catalog["powers"][0]["is_debuff_hint"] = True

    with pytest.raises(RuntimeMechanicsAuditError, match="handwritten"):
        audit_game_catalog(catalog)


def test_hidden_boss_transition_target_is_rejected() -> None:
    catalog = deepcopy(_catalog())
    catalog["monsters"][0]["follow_up_state_id"] = "SECRET_NEXT_MOVE"

    with pytest.raises(RuntimeMechanicsAuditError, match="hidden"):
        audit_game_catalog(catalog)


def test_boss_encounter_must_reference_known_native_monster() -> None:
    catalog = _catalog()
    catalog["encounters"][0]["monster_ids"] = ["UNKNOWN_BOSS"]

    with pytest.raises(RuntimeMechanicsAuditError, match="unknown monster_id"):
        audit_game_catalog(catalog)
