from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest
import torch

from sts2_env._sim_translate import translate_to_bridge_shape
from sts2_rl.encoding import GroundedEncodingConfig, GroundedObservationEncoder
from sts2_rl.encoding.grounded import (
    _NUMERIC_SLOT_BY_KEY,
    _bounded_number,
    _canonical_model_observation,
    _hash_id,
)
from sts2_rl.models import GroundedCandidateConfig


def _model_config() -> GroundedCandidateConfig:
    return GroundedCandidateConfig(
        token_feature_dim=224,
        d_model=32,
        n_heads=4,
        ffn_dim=64,
        world_layers=1,
        latent_slots=4,
        latent_layers=1,
        local_layers=1,
        candidate_layers=1,
        dropout=0.0,
        domain_count=8,
        type_vocab_size=32,
        role_vocab_size=32,
        owner_vocab_size=16,
        entity_vocab_size=128,
        zone_vocab_size=32,
        order_vocab_size=64,
    )


def _encoder(
    *,
    max_world_tokens: int = 256,
    max_candidates: int = 8,
    max_candidate_local_tokens: int = 32,
) -> GroundedObservationEncoder:
    model = _model_config()
    return GroundedObservationEncoder(
        GroundedEncodingConfig.from_model_config(
            model,
            max_world_tokens=max_world_tokens,
            max_candidates=max_candidates,
            max_candidate_local_tokens=max_candidate_local_tokens,
        )
    )


def _player(*, gold: int = 150) -> dict[str, Any]:
    return {
        "character_id": "CHARACTER.TEST",
        "current_hp": 48,
        "max_hp": 80,
        "gold": gold,
        "deck": [
            {"id": "STRIKE", "type": "Attack", "cost": 1},
            {"id": "DEFEND", "type": "Skill", "cost": 1},
        ],
        "relics": [],
        "potions": [],
    }


def test_full_map_and_visible_next_boss_reach_exact_world_relations() -> None:
    translated = translate_to_bridge_shape(
        {
            "state_type": "map",
            "run": {
                "active": True,
                "act": 2,
                "floor": 22,
                "room_type": "Monster",
                "room_model_id": "ROOM.CURRENT",
                "next_boss_id": "MONSTER.VISIBLE_BOSS",
            },
            "map": {
                "player": _player(),
                "current_coord": {"col": 0, "row": 5},
                "nodes": [
                    {
                        "col": 0,
                        "row": 5,
                        "point_type": "Monster",
                        "children": [
                            {"col": 1, "row": 6},
                            {"col": 2, "row": 6},
                        ],
                    },
                    {
                        "col": 1,
                        "row": 6,
                        "point_type": "Elite",
                        "children": [{"col": 2, "row": 7}],
                    },
                    {
                        "col": 2,
                        "row": 6,
                        "point_type": "RestSite",
                        "children": [{"col": 2, "row": 7}],
                    },
                    {
                        "col": 2,
                        "row": 7,
                        "point_type": "Boss",
                        "children": [],
                    },
                ],
                "next_options": [
                    {"index": 0, "col": 1, "row": 6, "point_type": "Elite"},
                    {
                        "index": 1,
                        "col": 2,
                        "row": 6,
                        "point_type": "RestSite",
                    },
                ],
            },
            "legal_actions": [
                {"action": "choose_map_node", "index": 0},
                {"action": "choose_map_node", "index": 1},
            ],
        },
        episode_id="observation-v2-map",
    )

    assert translated["run"]["next_boss_id"] == "MONSTER.VISIBLE_BOSS"
    assert translated["map"]["current_coord"] == {"x": 0, "y": 5}
    assert len(translated["map"]["nodes"]) == 4
    assert translated["map"]["nodes"][0]["coord"] == {"x": 0, "y": 5}
    assert translated["map"]["nodes"][0]["children"] == [
        {"col": 1, "row": 6},
        {"col": 2, "row": 6},
    ]

    canonical = _canonical_model_observation(translated)
    assert canonical["run"]["next_boss"] == {"id": "MONSTER.VISIBLE_BOSS"}
    assert canonical["map"]["current_coord"] == {"x": 0, "y": 5}
    assert len(canonical["map"]["points"]) == 4
    assert canonical["map"]["edges"] == [
        {
            "id": "map_coord:0:5",
            "instance_id": "map_coord:1:6",
            "type": "map_edge",
        },
        {
            "id": "map_coord:0:5",
            "instance_id": "map_coord:2:6",
            "type": "map_edge",
        },
        {
            "id": "map_coord:1:6",
            "instance_id": "map_coord:2:7",
            "type": "map_edge",
        },
        {
            "id": "map_coord:2:6",
            "instance_id": "map_coord:2:7",
            "type": "map_edge",
        },
    ]

    encoder = _encoder()
    encoded = encoder.encode(translated, translated["available_actions"])
    world_mask = encoded.batch.world.mask[0]
    world_entities = encoded.batch.world.entity_ids[0, world_mask]
    world_relations = set(
        encoded.batch.world.entity_aux_ids[0, world_mask].tolist()
    )
    model = _model_config()
    assert _hash_id(
        "entity",
        "MONSTER.VISIBLE_BOSS",
        model.entity_vocab_size,
    ) in world_entities.tolist()
    for coord in ("0:5", "1:6", "2:6", "2:7"):
        assert _hash_id(
            "entity_aux",
            f"instance:map_coord:{coord}",
            model.entity_vocab_size,
        ) in world_relations
    assert _hash_id(
        "entity_aux",
        "map_coord:0:5",
        model.entity_vocab_size,
    ) in world_relations

    # The full graph is accepted at its actual active shape.  One token less
    # must reject the state instead of silently dropping a node or edge.
    required_world_tokens = encoded.snapshot.world.token_count
    assert required_world_tokens < encoder.config.max_world_tokens
    too_small = GroundedObservationEncoder(
        replace(
            encoder.config,
            max_world_tokens=required_world_tokens - 1,
        )
    )
    with pytest.raises(ValueError, match="refusing lossy training input"):
        too_small.encode(translated, translated["available_actions"])


def test_nullable_rest_heal_is_not_coerced_and_exact_value_reaches_candidates() -> None:
    translated = translate_to_bridge_shape(
        {
            "state_type": "rest_site",
            "run": {"active": True, "act": 1, "floor": 11},
            "rest_site": {
                "player": _player(),
                "visible": True,
                "options": [
                    {
                        "index": 0,
                        "id": "REST",
                        "type": "Heal",
                        "is_enabled": True,
                        "heal_amount": 24,
                    },
                    {
                        "index": 1,
                        "id": "SMITH",
                        "type": "Upgrade",
                        "is_enabled": True,
                        "heal_amount": None,
                    },
                ],
            },
            "legal_actions": [
                {"action": "choose_rest_option", "index": 0},
                {"action": "choose_rest_option", "index": 1},
            ],
        },
        episode_id="observation-v2-rest",
    )

    assert translated["rest_site"]["options"][0]["heal_amount"] == 24
    assert translated["rest_site"]["options"][1]["heal_amount"] is None
    canonical = _canonical_model_observation(translated)
    assert canonical["rest_site"]["options"][0]["heal_amount"] == 24
    assert "heal_amount" not in canonical["rest_site"]["options"][1]

    encoded = _encoder().encode(translated, translated["available_actions"]).batch
    heal_slot = _NUMERIC_SLOT_BY_KEY["heal_amount"]
    first_rows = encoded.candidates.local_features[
        0,
        0,
        encoded.candidates.local_mask[0, 0],
    ]
    second_rows = encoded.candidates.local_features[
        0,
        1,
        encoded.candidates.local_mask[0, 1],
    ]
    assert any(
        row[heal_slot].item() == pytest.approx(_bounded_number(24))
        for row in first_rows
    )
    assert all(row[heal_slot].item() == 0.0 for row in second_rows)


def test_canonical_shop_items_keep_sparse_slots_and_nested_runtime_facts() -> None:
    dynamic_var = {
        "name": "Block",
        "var_type": "IntVar",
        "base_value": 8,
        "enchanted_value": 11,
        "preview_value": 11,
        "int_value": 11,
    }
    shop_items = [
        {
            "index": 3,
            "type": "card",
            "price": 75,
            "is_affordable": True,
            "is_stocked": True,
            "is_on_sale": True,
            "card": {
                "id": "FORTIFY",
                "type": "Skill",
                "cost": 1,
                "dynamic_vars": [dynamic_var],
                "enchantments": [
                    {
                        "id": "SHINY",
                        "amount": 1,
                        "dynamic_vars": [dynamic_var],
                    }
                ],
            },
        },
        {
            "index": 7,
            "type": "relic",
            "price": 180,
            "is_affordable": False,
            "is_stocked": True,
            "is_on_sale": False,
            "relic": {
                "id": "COUNTING_RELIC",
                "rarity": "Rare",
                "display_amount": 4,
                "stack_count": 2,
                "dynamic_vars": [dynamic_var],
            },
        },
        {
            "index": 11,
            "type": "potion",
            "price": 55,
            "is_affordable": True,
            "is_stocked": True,
            "is_on_sale": False,
            "potion": {
                "id": "TARGET_POTION",
                "usage": "Combat",
                "target_type": "AnyEnemy",
                "can_use_in_combat": True,
                "dynamic_vars": [dynamic_var],
            },
        },
    ]
    translated = translate_to_bridge_shape(
        {
            "state_type": "shop",
            "run": {"active": True, "act": 2, "floor": 27},
            "shop": {
                "player": _player(gold=150),
                "visible": True,
                "gold": 150,
                "items": shop_items,
            },
            "legal_actions": [
                {"action": "shop_purchase", "index": 3},
                {"action": "shop_purchase", "index": 7},
                {"action": "shop_purchase", "index": 11},
                {"action": "shop_skip"},
            ],
        },
        episode_id="observation-v2-shop",
    )

    translated_items = translated["shop"]["items"]
    assert [item["slot_index"] for item in translated_items] == [3, 7, 11]
    assert translated_items[0]["card"]["id"] == "CARD.FORTIFY"
    assert translated_items[1]["relic"]["id"] == "RELIC.COUNTING_RELIC"
    assert translated_items[2]["potion"]["id"] == "POTION.TARGET_POTION"
    assert [
        action["item"]["slot_index"]
        for action in translated["available_actions"][:3]
    ] == [3, 7, 11]

    canonical = _canonical_model_observation(translated)
    canonical_items = canonical["shop"]["items"]
    assert canonical_items[0]["type"] == "card"
    assert canonical_items[0]["price"] == 75
    assert canonical_items[0]["is_affordable"] is True
    assert canonical_items[0]["is_on_sale"] is True
    assert canonical_items[0]["card"]["dynamic_vars"][0]["current_value"] == 11
    assert canonical_items[0]["card"]["enchantments"][0]["id"] == "SHINY"
    assert canonical_items[1]["relic"]["display_amount"] == 4
    assert canonical_items[2]["potion"]["can_use_in_combat"] is True

    encoder = _encoder(max_candidates=4)
    encoded = encoder.encode(translated, translated["available_actions"]).batch
    assert encoded.candidates.features.shape[1] == 4
    world_relations = set(
        encoded.world.entity_aux_ids[0, encoded.world.mask[0]].tolist()
    )
    model = _model_config()
    for slot in (3, 7, 11):
        relation = _hash_id(
            "entity_aux",
            f"instance:shop_slot:{slot}",
            model.entity_vocab_size,
        )
        assert relation in world_relations
    assert set(encoded.candidates.entity_aux_ids[0, :3].tolist()).issubset(
        world_relations
    )

    card_item_relation = _hash_id(
        "entity_aux",
        "instance:shop_slot:3",
        model.entity_vocab_size,
    )
    item_type = _hash_id("type", "mapping:item", model.type_vocab_size)
    card_item_rows = torch.nonzero(
        (encoded.world.entity_aux_ids[0] == card_item_relation)
        & (encoded.world.type_ids[0] == item_type)
        & encoded.world.mask[0],
        as_tuple=False,
    ).flatten()
    assert card_item_rows.numel() == 1
    card_item_features = encoded.world.features[0, card_item_rows[0]]
    assert card_item_features[_NUMERIC_SLOT_BY_KEY["price"]].item() == (
        pytest.approx(_bounded_number(75))
    )
    assert card_item_features[
        _NUMERIC_SLOT_BY_KEY["is_affordable"]
    ].item() == pytest.approx(1.0)
    assert card_item_features[_NUMERIC_SLOT_BY_KEY["is_on_sale"]].item() == (
        pytest.approx(1.0)
    )

    with pytest.raises(ValueError, match="legal action count exceeds"):
        GroundedObservationEncoder(
            replace(encoder.config, max_candidates=3)
        ).encode(translated, translated["available_actions"])


def test_malformed_observation_v2_coordinates_and_shop_slots_fail_closed() -> None:
    with pytest.raises(TypeError, match="current_coord"):
        translate_to_bridge_shape(
            {
                "state_type": "map",
                "map": {
                    "player": _player(),
                    "current_coord": {"col": 1, "row": "two"},
                },
                "legal_actions": [],
            },
            episode_id="bad-map-coordinate",
        )
    with pytest.raises(TypeError, match="shop item index"):
        translate_to_bridge_shape(
            {
                "state_type": "shop",
                "shop": {
                    "player": _player(),
                    "items": [{"index": "three", "type": "card"}],
                },
                "legal_actions": [],
            },
            episode_id="bad-shop-slot",
        )
