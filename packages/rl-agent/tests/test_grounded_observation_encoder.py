from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest
import torch

from sts2_rl.encoding import (
    GROUNDING_ENCODING_VERSION,
    GroundedEncodingConfig,
    GroundedObservationEncoder,
    grounding_encoding_identity,
)
from sts2_rl.models import GroundedCandidateConfig, GroundedCandidateModel


def _small_model_config() -> GroundedCandidateConfig:
    return GroundedCandidateConfig(
        token_feature_dim=128,
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
        zone_vocab_size=16,
        order_vocab_size=32,
    )


def _encoder(model_config: GroundedCandidateConfig | None = None) -> GroundedObservationEncoder:
    model = model_config or _small_model_config()
    config = GroundedEncodingConfig.from_model_config(
        model,
        max_world_tokens=24,
        max_candidates=6,
        max_candidate_local_tokens=5,
    )
    return GroundedObservationEncoder(config)


def _observation() -> dict[str, object]:
    return {
        "phase": "combat",
        "decision_domain": "combat",
        "player": {
            "id": "ironclad",
            "side": "player",
            "hp": 51,
            "max_hp": 80,
            "energy": 2,
            "hand": [
                {"id": "strike", "cost": 1},
                {"id": "defend", "cost": 1},
            ],
        },
        "combat": {
            "enemies": [
                {"id": "cultist", "side": "enemy", "hp": 31, "max_hp": 48}
            ]
        },
        "available_actions": [{"kind": "retired-leak", "quality": 999999}],
        "boss_mechanics": {"forced_line": 1.0},
        "_sim_raw": {"secret": 42},
    }


def _actions() -> list[dict[str, object]]:
    return [
        {
            "action_handle": "opaque:one",
            "idx": 77,
            "kind": "play_card",
            "model_action_kind": "play_card",
            "is_enabled": True,
            "card": {"id": "strike", "cost": 1, "side": "player"},
            "target": {"id": "cultist", "side": "enemy", "hp": 31},
        },
        {
            "action_handle": "opaque:two",
            "idx": 0,
            "kind": "end_turn",
            "model_action_kind": "end_turn",
            "is_enabled": False,
        },
    ]


def test_encoder_contract_runs_through_grounded_model() -> None:
    model_config = _small_model_config()
    encoded = _encoder(model_config).encode(_observation(), _actions())
    encoded.batch.validate(model_config)

    model = GroundedCandidateModel(model_config).eval()
    with torch.no_grad():
        output = model(encoded.batch)

    assert output.policy_logits.shape == (1, 6)
    assert output.action_mask[0].tolist() == [True, False, False, False, False, False]
    assert encoded.action(0).handle == "opaque:one"


def test_candidate_containers_and_retired_features_cannot_pollute_world() -> None:
    encoder = _encoder()
    before = _observation()
    after = _observation()
    after["available_actions"] = [{"kind": "different", "quality": -12345}]
    after["boss_mechanics"] = {"forced_line": -999.0}
    after["_sim_raw"] = {"secret": "different"}
    after["action_quality"] = 999999.0
    after["state_version"] = 888
    after["card_effect_profile"] = {"damage_score": 1234.0}

    first = encoder.encode(before, _actions()).batch.world
    second = encoder.encode(after, list(reversed(_actions()))).batch.world

    for field in (
        "features",
        "mask",
        "type_ids",
        "role_ids",
        "owner_ids",
        "entity_ids",
        "zone_ids",
        "order_ids",
    ):
        assert torch.equal(getattr(first, field), getattr(second, field))


def test_dispatch_identity_and_candidate_position_are_not_model_features() -> None:
    encoder = _encoder()
    action_a = {
        "action_handle": "session-a:42",
        "action_index": 0,
        "kind": "play_card",
        "model_action_kind": "play_card",
        "card": {"id": "strike", "cost": 1, "damage": 999, "draw": 9},
    }
    action_b = {
        "action_handle": "session-b:9000",
        "action_index": 91,
        "kind": "play_card",
        "model_action_kind": "play_card",
        "card": {"id": "strike", "cost": 1},
    }
    encoded = encoder.encode(_observation(), [action_a, action_b])
    candidates = encoded.batch.candidates

    assert torch.equal(candidates.features[:, 0], candidates.features[:, 1])
    for field in ("type_ids", "role_ids", "owner_ids", "entity_ids", "zone_ids"):
        values = getattr(candidates, field)
        assert torch.equal(values[:, 0], values[:, 1])
    assert encoded.actions[0].handle != encoded.actions[1].handle


def test_live_semantic_previews_and_transport_positions_are_not_model_inputs() -> None:
    encoder = _encoder()
    first = {
        "action_handle": "first",
        "kind": "play_card",
        "model_action_kind": "play_card",
        "hand_index": 0,
        "target_combat_id": 7,
        "card": {"id": "strike", "cost": 1},
        "target": {"model_id": "MONSTER.CULTIST", "hp": 31, "max_hp": 48},
        "semantic": {"damage": 999, "role_score": 1.0},
        "effect_preview": {"damage": 999},
    }
    second = {
        **first,
        "action_handle": "second",
        "hand_index": 91,
        "target_combat_id": 300,
        "card": {"id": "strike", "cost": 1, "damage": -999, "draw": -9},
        "semantic": {"damage": -999, "role_score": -1.0},
        "effect_preview": {"damage": -999},
    }

    encoded = encoder.encode(_observation(), [first, second]).batch.candidates
    for field in (
        "features",
        "type_ids",
        "role_ids",
        "owner_ids",
        "entity_ids",
        "entity_aux_ids",
        "zone_ids",
        "target_owner_ids",
        "target_entity_ids",
        "target_entity_aux_ids",
        "local_features",
        "local_mask",
        "local_type_ids",
        "local_role_ids",
        "local_owner_ids",
        "local_entity_ids",
        "local_entity_aux_ids",
        "local_zone_ids",
        "local_order_ids",
    ):
        values = getattr(encoded, field)
        assert torch.equal(values[:, 0], values[:, 1])


def test_stable_enemy_model_identity_is_encoded_before_runtime_combat_id() -> None:
    encoder = _encoder()
    first = _observation()
    second = _observation()
    first["combat"] = {
        "enemies": [
            {"combat_id": 7, "model_id": "MONSTER.CULTIST", "hp": 30, "max_hp": 40}
        ]
    }
    second["combat"] = {
        "enemies": [
            {"combat_id": 7, "model_id": "MONSTER.LOUSE", "hp": 30, "max_hp": 40}
        ]
    }

    first_world = encoder.encode(first, _actions()).batch.world
    second_world = encoder.encode(second, _actions()).batch.world
    assert not torch.equal(first_world.entity_ids, second_world.entity_ids)


def test_core_categorical_and_combat_count_facts_are_not_dropped() -> None:
    encoder = _encoder()
    first = {
        "phase": "combat",
        "decision_domain": "combat",
        "player": {"character_id": "IRONCLAD", "hp": 50, "max_hp": 80},
        "run": {"room_type": "Monster", "room_model_id": "ROOM.A", "floor": 3},
        "combat": {"facing": "left", "draw": 5, "discard": 1, "exhaust": 0},
    }
    second = {
        "phase": "combat",
        "decision_domain": "combat",
        "player": {"character_id": "SILENT", "hp": 50, "max_hp": 80},
        "run": {"room_type": "Elite", "room_model_id": "ROOM.B", "floor": 3},
        "combat": {"facing": "right", "draw": 1, "discard": 5, "exhaust": 2},
    }

    first_world = encoder.encode(first, _actions()).batch.world
    second_world = encoder.encode(second, _actions()).batch.world
    assert not torch.equal(first_world.features, second_world.features)
    assert not torch.equal(first_world.entity_ids, second_world.entity_ids)


def test_candidate_character_and_selection_identity_use_typed_fields_not_ui_label() -> None:
    encoder = _encoder()
    actions = [
        {
            "action_handle": "left",
            "kind": "choose_character",
            "model_action_kind": "character_select",
            "selection": "left",
            "character": {"id": "CHARACTER.IRONCLAD"},
            "label": "Select character 0 / 本地化文本",
        },
        {
            "action_handle": "right",
            "kind": "choose_character",
            "model_action_kind": "character_select",
            "selection": "right",
            "character": {"id": "CHARACTER.SILENT"},
            "label": "Select character 0 / 本地化文本",
        },
    ]
    encoded = encoder.encode(_observation(), actions).batch.candidates
    assert not torch.equal(encoded.role_ids[:, 0], encoded.role_ids[:, 1])
    assert not torch.equal(encoded.entity_ids[:, 0], encoded.entity_ids[:, 1])

    relabeled = [
        {**actions[0], "label": "Select character 999 / another locale"},
        {**actions[1], "label": "Select character 123 / another locale"},
    ]
    second = encoder.encode(_observation(), relabeled).batch.candidates
    for field in (
        "features",
        "type_ids",
        "role_ids",
        "owner_ids",
        "entity_ids",
        "entity_aux_ids",
        "zone_ids",
        "local_features",
        "local_entity_ids",
    ):
        assert torch.equal(getattr(encoded, field), getattr(second, field))


def test_raw_event_options_and_run_modes_remain_distinguishable_without_effect_rules() -> None:
    encoder = _encoder()
    event_actions = [
        {
            "action_handle": "event:0",
            "kind": "choose_event_option",
            "model_action_kind": "event_option",
            "option": {
                "title": "Accept the bargain",
                "description": "Lose HP. Gain a relic.",
                "effect_deltas": {"hp": -10, "relic_score": 999},
            },
        },
        {
            "action_handle": "event:1",
            "kind": "choose_event_option",
            "model_action_kind": "event_option",
            "option": {"title": "Leave", "description": "Walk away."},
        },
    ]
    event_batch = encoder.encode(_observation(), event_actions).batch.candidates
    assert not torch.equal(event_batch.entity_ids[:, 0], event_batch.entity_ids[:, 1])

    mode_actions = [
        {
            "action_handle": "mode:standard",
            "kind": "choose_run_mode",
            "model_action_kind": "run_mode_selection",
            "run_mode_action": "standard",
        },
        {
            "action_handle": "mode:daily",
            "kind": "choose_run_mode",
            "model_action_kind": "run_mode_selection",
            "run_mode_action": "daily",
        },
    ]
    mode_batch = encoder.encode(_observation(), mode_actions).batch.candidates
    assert not torch.equal(mode_batch.role_ids[:, 0], mode_batch.role_ids[:, 1])


def test_canonical_headless_and_live_route_coordinates_match_and_distinguish() -> None:
    encoder = _encoder()
    actions = [
        {
            "action_handle": "headless:a",
            "kind": "choose_map_node",
            "model_action_kind": "map",
            "transport_kind": "map",
            "map_node": {"coord": {"x": 1, "y": 2}, "point_type": "Monster"},
        },
        {
            "action_handle": "headless:b",
            "kind": "choose_map_node",
            "model_action_kind": "map",
            "transport_kind": "map",
            "map_node": {"coord": {"x": 4, "y": 2}, "point_type": "Monster"},
        },
        {
            "action_handle": "live:a",
            "kind": "map",
            "model_action_kind": "map",
            "map_node": {"coord": {"x": 1, "y": 2}, "point_type": "Monster"},
        },
        {
            "action_handle": "live:b",
            "kind": "map",
            "model_action_kind": "map",
            "map_node": {"coord": {"x": 4, "y": 2}, "point_type": "Monster"},
        },
    ]
    batch = encoder.encode(_observation(), actions).batch.candidates
    assert not torch.equal(batch.local_features[:, 0], batch.local_features[:, 1])
    assert not torch.equal(batch.local_features[:, 2], batch.local_features[:, 3])
    assert torch.equal(batch.local_features[:, 0], batch.local_features[:, 2])
    assert torch.equal(batch.local_features[:, 1], batch.local_features[:, 3])
    assert torch.equal(batch.role_ids[:, 0], batch.role_ids[:, 2])


def test_real_live_and_headless_candidate_dtos_are_canonical_and_non_aliasing() -> None:
    encoder = _encoder()
    live_event = {
        "action_handle": "event_option:0",
        "kind": "event_option",
        "model_action_kind": "event_option",
        "index": 0,
        "option": {
            "title": "Accept",
            "description": "Visible choice text",
            "option_type": "choice",
            "is_locked": False,
            "is_proceed": False,
        },
        # Real live DTO compatibility copies. They must not create a second
        # model representation.
        "title": "Accept",
        "description": "Visible choice text",
        "option_type": "choice",
        "proceed": False,
    }
    headless_event = {
        "action_handle": "sim:9:choose_event_option",
        "kind": "choose_event_option",
        "transport_kind": "event_option",
        "model_action_kind": "event_option",
        "index": 9,
        "option": {
            "title": "Accept",
            "description": "Visible choice text",
            "type": "choice",
            "is_locked": False,
            "is_proceed": False,
        },
    }
    different_live_event = {
        **live_event,
        "action_handle": "event_option:1",
        "option": {
            **live_event["option"],
            "title": "Leave",
            "description": "A genuinely different visible option",
        },
    }
    candidates = encoder.encode(
        _observation(),
        [live_event, headless_event, different_live_event],
    ).batch.candidates
    paired_fields = (
        "features",
        "type_ids",
        "role_ids",
        "owner_ids",
        "entity_ids",
        "zone_ids",
        "target_owner_ids",
        "target_entity_ids",
        "local_features",
        "local_mask",
        "local_type_ids",
        "local_role_ids",
        "local_owner_ids",
        "local_entity_ids",
        "local_entity_aux_ids",
        "local_zone_ids",
        "local_order_ids",
    )
    for field in paired_fields:
        values = getattr(candidates, field)
        assert torch.equal(values[:, 0], values[:, 1]), field
    assert not torch.equal(candidates.entity_ids[:, 0], candidates.entity_ids[:, 2])

    live_mode = {
        "action_handle": "run_mode:standard",
        "kind": "run_mode_selection",
        "model_action_kind": "run_mode_selection",
        "run_mode_action": "standard",
    }
    headless_mode = {
        "action_handle": "sim:0:choose_run_mode",
        "kind": "choose_run_mode",
        "model_action_kind": "run_mode_selection",
        "run_mode_action": "standard",
    }
    daily_mode = {
        **live_mode,
        "action_handle": "run_mode:daily",
        "run_mode_action": "daily",
    }
    modes = encoder.encode(
        _observation(),
        [live_mode, headless_mode, daily_mode],
    ).batch.candidates
    assert torch.equal(modes.role_ids[:, 0], modes.role_ids[:, 1])
    assert not torch.equal(modes.role_ids[:, 0], modes.role_ids[:, 2])


def test_real_live_root_map_dto_matches_headless_nested_map_node() -> None:
    encoder = _encoder()
    live = {
        "action_handle": "map:1,3",
        "kind": "map",
        "model_action_kind": "map",
        "coord": {"col": 1, "row": 3},
        "point_type": "Elite",
    }
    headless = {
        "action_handle": "sim:0:choose_map_node",
        "kind": "choose_map_node",
        "model_action_kind": "map",
        "map_node": {
            "coord": {"x": 1, "y": 3},
            "point_type": "Elite",
        },
    }
    candidates = encoder.encode(_observation(), [live, headless]).batch.candidates
    for field in (
        "features",
        "type_ids",
        "role_ids",
        "owner_ids",
        "entity_ids",
        "entity_aux_ids",
        "zone_ids",
        "local_features",
        "local_mask",
        "local_type_ids",
        "local_role_ids",
        "local_owner_ids",
        "local_entity_ids",
        "local_zone_ids",
        "local_order_ids",
    ):
        values = getattr(candidates, field)
        assert torch.equal(values[:, 0], values[:, 1]), field


def test_live_target_is_joined_to_world_facts_like_headless_target() -> None:
    encoder = _encoder()
    observation = {
        "phase": "combat",
        "decision_domain": "combat",
        "player": {"hp": 60, "max_hp": 80},
        "combat": {
            "enemies": [
                {
                    "id": 7,
                    "combat_id": 7,
                    "model_id": "MONSTER.CULTIST",
                    "side": "Enemy",
                    "hp": 19,
                    "max_hp": 48,
                    "block": 2,
                }
            ]
        },
    }
    card = {"id": "CARD.STRIKE", "type": "Attack", "cost": 1}
    live = {
        "action_handle": "play_card:0:ref@7",
        "kind": "play_card",
        "model_action_kind": "play_card",
        "card": card,
        "target": {"combat_id": 7, "name": "Cultist", "side": "Enemy"},
    }
    headless = {
        "action_handle": "sim:0:play_card",
        "kind": "play_card",
        "model_action_kind": "play_card",
        "card": card,
        "target": {
            "combat_id": 7,
            "entity_id": "MONSTER.CULTIST",
            "side": "Enemy",
            "hp": 19,
            "max_hp": 48,
            "block": 2,
        },
    }
    candidates = encoder.encode(observation, [live, headless]).batch.candidates
    for field in (
        "target_owner_ids",
        "target_entity_ids",
        "target_entity_aux_ids",
        "local_features",
        "local_mask",
        "local_type_ids",
        "local_role_ids",
        "local_owner_ids",
        "local_entity_ids",
        "local_entity_aux_ids",
        "local_zone_ids",
        "local_order_ids",
    ):
        values = getattr(candidates, field)
        assert torch.equal(values[:, 0], values[:, 1]), field


def test_live_and_headless_world_projection_use_same_observable_intersection() -> None:
    encoder = _encoder()
    live = {
        "phase": "combat",
        "decision_domain": "combat",
        "run": {
            "active": True,
            "floor": 7,
            "room_type": "Monster",
            "room_model": "ROOM.TEST",
            "coord": {"col": 2, "row": 6},
        },
        "player": {
            "character_id": "CHARACTER.IRONCLAD",
            "hp": 61,
            "max_hp": 80,
            "block": 3,
            "gold": 99,
            "deck": 2,
            "deck_cards": [
                {"id": "CARD.STRIKE", "type": "Attack", "cost": 1},
                {"id": "CARD.DEFEND", "type": "Skill", "cost": 1},
            ],
            "powers": [{"id": "POWER.STRENGTH", "amount": 2}],
            "relics": [{"id": "RELIC.BURNING_BLOOD", "rarity": "Starter"}],
            "potions": [
                {"id": "POTION.FIRE", "rarity": "Common"},
                {"id": None, "title": "[empty]"},
            ],
        },
        "combat": {
            "round": 2,
            "energy": 2,
            "max_energy": 3,
            "hand": [{"id": "CARD.STRIKE", "type": "Attack", "cost": 1}],
            "draw": 2,
            "discard": 1,
            "exhaust": 0,
            "enemies": [
                {
                    "id": 7,
                    "combat_id": 7,
                    "model_id": "MONSTER.CULTIST",
                    "side": "Enemy",
                    "hp": 31,
                    "max_hp": 48,
                    "block": 0,
                    "powers": [],
                    "intents": [{"type": "Attack", "damage": 6, "repeats": 1}],
                }
            ],
        },
    }
    headless = {
        "phase": "combat",
        "decision_domain": "combat",
        "run": {
            "run_active": True,
            "total_floor": 7,
            "room_type": "Monster",
            "room_model_id": "ROOM.TEST",
            "coord": {"x": 2, "y": 6},
        },
        "player": {
            "character": "CHARACTER.IRONCLAD",
            "current_hp": 61,
            "max_hp": 80,
            "block": 3,
            "gold": 99,
            "deck": [
                {"id": "CARD.STRIKE", "type": "Attack", "cost": 1, "pile": "Deck"},
                {"id": "CARD.DEFEND", "type": "Skill", "cost": 1, "pile": "Deck"},
            ],
            "hand": [{"id": "CARD.STRIKE", "type": "Attack", "cost": 1}],
            # Identities in these piles are simulator-only and must collapse to
            # the same counts exposed by live.
            "draw_pile": [{"id": "CARD.SECRET_A"}, {"id": "CARD.SECRET_B"}],
            "discard_pile": [{"id": "CARD.SECRET_C"}],
            "exhaust_pile": [],
            "status": [{"id": "POWER.STRENGTH", "amount": 2}],
            "relics": [{"id": "RELIC.BURNING_BLOOD", "rarity": "Starter"}],
            "potions": [{"id": "POTION.FIRE", "rarity": "Common"}],
            "energy": 2,
            "max_energy": 3,
        },
        "combat": {
            "in_progress": True,
            "round": 2,
            "enemies": [
                {
                    "combat_id": 7,
                    "entity_id": "MONSTER.CULTIST",
                    "side": "Enemy",
                    "current_hp": 31,
                    "max_hp": 48,
                    "block": 0,
                    "status": [],
                    "intents": [
                        {"intent_type": "Attack", "total_damage": 6, "hits": 1}
                    ],
                }
            ],
        },
        # Inactive simulator-only sections must not fingerprint the backend.
        "map": {"nodes": [{"x": 99, "y": 99}]},
        "event": {"options": []},
        "shop": {"items": []},
    }
    first = encoder.encode(live, _actions()).batch.world
    second = encoder.encode(headless, _actions()).batch.world
    for field in (
        "features",
        "mask",
        "type_ids",
        "role_ids",
        "owner_ids",
        "entity_ids",
        "entity_aux_ids",
        "zone_ids",
        "order_ids",
    ):
        assert torch.equal(getattr(first, field), getattr(second, field)), field


def test_encoder_requires_registered_canonical_model_action_kind() -> None:
    encoder = _encoder()
    with pytest.raises(ValueError, match="missing non-empty model_action_kind"):
        encoder.encode(
            _observation(),
            [{"action_handle": "missing", "kind": "play_card"}],
        )
    with pytest.raises(ValueError, match="unregistered model_action_kind"):
        encoder.encode(
            _observation(),
            [
                {
                    "action_handle": "unknown",
                    "kind": "new_backend_enum",
                    "model_action_kind": "new_backend_enum",
                }
            ],
        )


def test_unknown_engineered_subtrees_are_pruned_at_container_boundary() -> None:
    encoder = _encoder()
    clean = {
        "phase": "combat",
        "decision_domain": "combat",
        "player": {"hp": 50, "max_hp": 80},
    }
    polluted = {
        **clean,
        "quality": {"score": 999, "hp": 999},
        "planner": {"predictedDamage": 999},
        "estimate": {"damage": 999},
        "effectPreview": {"damage": 999},
        "mystery": {"hp": 999},
    }
    first = encoder.encode(clean, _actions()).batch.world
    second = encoder.encode(polluted, _actions()).batch.world
    for field in (
        "features",
        "mask",
        "type_ids",
        "role_ids",
        "owner_ids",
        "entity_ids",
        "zone_ids",
        "order_ids",
    ):
        assert torch.equal(getattr(first, field), getattr(second, field))

    clean_action = {
        "action_handle": "clean",
        "kind": "play_card",
        "model_action_kind": "play_card",
        "card": {"id": "CARD.STRIKE", "cost": 1},
    }
    polluted_action = {
        **clean_action,
        "action_handle": "polluted",
        "card": {
            "id": "CARD.STRIKE",
            "cost": 1,
            "quality": {"score": 999},
            "planner": {"damage": 999},
            "effectPreview": {"damage": 999},
            "mystery": {"hp": 999},
        },
    }
    candidates = encoder.encode(clean, [clean_action, polluted_action]).batch.candidates
    for field in (
        "features",
        "entity_ids",
        "entity_aux_ids",
        "local_features",
        "local_mask",
        "local_entity_ids",
        "local_entity_aux_ids",
    ):
        values = getattr(candidates, field)
        assert torch.equal(values[:, 0], values[:, 1])


def test_nonfinite_raw_facts_fail_closed_before_tensor_materialization() -> None:
    encoder = _encoder()
    observation = {
        "phase": "combat",
        "decision_domain": "combat",
        "player": {"hp": float("nan"), "max_hp": 80},
    }
    with pytest.raises(ValueError, match="must be finite"):
        encoder.encode(observation, _actions())


def test_reviewed_numeric_facts_have_collision_free_feature_slots() -> None:
    encoder = _encoder()
    first = {
        "phase": "combat",
        "decision_domain": "combat",
        "player": {"block": 3, "max_hp": 80},
    }
    swapped = {
        "phase": "combat",
        "decision_domain": "combat",
        "player": {"block": 80, "max_hp": 3},
    }

    first_world = encoder.encode(first, _actions()).batch.world
    swapped_world = encoder.encode(swapped, _actions()).batch.world

    # The previous key-hash layout mapped block and max_hp to the same slot,
    # so these two states were exactly indistinguishable to the network.
    assert not torch.equal(first_world.features, swapped_world.features)


def test_encoder_rejects_feature_dimensions_below_versioned_abi() -> None:
    with pytest.raises(ValueError, match="at least 128"):
        GroundedEncodingConfig(feature_dim=127)


def test_encoding_contract_has_stable_checkpoint_identity() -> None:
    identity = grounding_encoding_identity()

    assert identity["version"] == GROUNDING_ENCODING_VERSION
    assert identity["min_token_feature_dim"] == 128
    assert identity["feature_abi_end"] <= 128
    assert len(identity["fingerprint_sha256"]) == 64
    assert set(identity["fingerprint_sha256"]) <= set("0123456789abcdef")
    assert grounding_encoding_identity() == identity


def test_secondary_entity_hash_disambiguates_primary_bucket_collisions() -> None:
    model = _small_model_config()
    size = model.entity_vocab_size

    def bucket(namespace: str, value: str) -> int:
        digest = hashlib.sha256(f"{namespace}\0{value.lower()}".encode()).digest()
        return 2 + int.from_bytes(digest[:8], "big") % (size - 2)

    seen: dict[int, str] = {}
    pair: tuple[str, str] | None = None
    for index in range(10_000):
        candidate = f"CARD.COLLISION_{index}"
        primary = bucket("entity", candidate)
        previous = seen.get(primary)
        if previous is not None and bucket("entity_aux", previous) != bucket(
            "entity_aux",
            candidate,
        ):
            pair = (previous, candidate)
            break
        seen[primary] = candidate
    assert pair is not None

    actions = [
        {
            "action_handle": f"play:{index}",
            "kind": "play_card",
            "model_action_kind": "play_card",
            "card": {"id": card_id, "cost": 1},
        }
        for index, card_id in enumerate(pair)
    ]
    candidates = _encoder(model).encode(_observation(), actions).batch.candidates
    assert torch.equal(candidates.entity_ids[:, 0], candidates.entity_ids[:, 1])
    assert not torch.equal(
        candidates.entity_aux_ids[:, 0],
        candidates.entity_aux_ids[:, 1],
    )


def test_encoder_stack_preserves_fixed_shape_batches() -> None:
    encoder = _encoder()
    first = encoder.encode(_observation(), _actions())
    second_obs = _observation()
    second_obs["decision_domain"] = "route"
    second = encoder.encode(second_obs, _actions())

    batch = encoder.stack([first, second])

    assert batch.world.features.shape[0] == 2
    assert batch.candidates.features.shape[0] == 2
    assert batch.domain_ids.tolist() == [1, 3]

    with pytest.raises(ValueError, match="different encoding contracts"):
        encoder.stack(
            [
                first,
                replace(second, encoding_fingerprint="0" * 64),
            ]
        )


def test_encoder_never_silently_truncates_legal_candidates() -> None:
    encoder = _encoder()
    actions = [
        {"action_handle": f"candidate:{index}", "kind": "choose"}
        for index in range(encoder.config.max_candidates + 1)
    ]

    with pytest.raises(ValueError, match="silently hide dispatchable candidates"):
        encoder.encode(_observation(), actions)


def test_encoder_fails_closed_on_world_or_candidate_local_overflow() -> None:
    model = _small_model_config()
    world_limited = GroundedObservationEncoder(
        GroundedEncodingConfig.from_model_config(
            model,
            max_world_tokens=1,
            max_candidates=2,
            max_candidate_local_tokens=2,
        )
    )
    with pytest.raises(ValueError, match="world observation exceeds"):
        world_limited.encode(_observation(), _actions())

    local_limited = GroundedObservationEncoder(
        GroundedEncodingConfig.from_model_config(
            model,
            max_world_tokens=24,
            max_candidates=2,
            max_candidate_local_tokens=1,
        )
    )
    action = {
        "action_handle": "event:overflow",
        "kind": "event_option",
        "model_action_kind": "event_option",
        "option": {
            "option_id": "OPTION.X",
            "coord": {"x": 1, "y": 2},
        },
    }
    with pytest.raises(ValueError, match="candidate-local observation exceeds"):
        local_limited.encode(_observation(), [action])
