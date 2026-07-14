from __future__ import annotations

import hashlib
import pickle
from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from sts2_rl.encoding import (
    GROUNDING_ENCODING_VERSION,
    GroundedEncodingConfig,
    GroundedObservationEncoder,
    grounding_encoding_identity,
)
from sts2_rl.encoding.grounded import (
    _DYNAMIC_VALUE_SLOT_BY_KEY,
    _bounded_number,
    _canonical_card,
    _canonical_enemy,
    _canonical_event,
    _canonical_potion,
    _canonical_power,
    _canonical_relic,
    _hash_id,
)
from sts2_rl.models import GroundedCandidateConfig, RecurrentCandidateModel


def _small_model_config() -> GroundedCandidateConfig:
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

    model = RecurrentCandidateModel(model_config).eval()
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


def test_selection_mutations_have_distinct_candidate_roles() -> None:
    encoder = _encoder()
    observation = {
        "phase": "combat",
        "decision_domain": "combat",
        "card_selection": {
            "selected_count": 1,
            "min_select": 1,
            "max_select": 1,
            "remaining_picks": 0,
            "can_confirm": True,
            "selected_cards": [{"id": "CARD.STRIKE", "index": 0}],
        },
    }
    actions = [
        {
            "action_handle": "selection:deselect",
            "kind": "deselect_card",
            "model_action_kind": "card_selection",
            "model_action_variant": "deselect",
            "card": {"id": "CARD.STRIKE", "pile": "Selected", "cost": 1},
        },
        {
            "action_handle": "selection:confirm",
            "kind": "confirm_selection",
            "model_action_kind": "card_selection",
            "model_action_variant": "confirm",
        },
        {
            "action_handle": "selection:cancel",
            "kind": "cancel_selection",
            "model_action_kind": "card_selection",
            "model_action_variant": "cancel_prompt",
        },
    ]

    batch = encoder.encode(observation, actions).batch
    unselected = {
        **observation,
        "card_selection": {
            **observation["card_selection"],
            "selected_count": 0,
            "remaining_picks": 1,
            "can_confirm": False,
            "selected_cards": [],
        },
    }
    unselected_world = encoder.encode(unselected, actions).batch.world

    assert batch.world.mask.any()
    assert not torch.equal(batch.world.features, unselected_world.features)
    roles = batch.candidates.role_ids[0, :3].tolist()
    assert len(set(roles)) == 3


def test_multiselect_membership_identity_is_part_of_world_state() -> None:
    encoder = _encoder()
    base = {
        "phase": "combat",
        "decision_domain": "combat",
        "player": {"hp": 50, "max_hp": 80},
        "card_selection": {
            "mode": "SimpleGrid",
            "prompt_id": "card.DREDGE.selection",
            "operation_type": "select",
            "source_zone": "Discard",
            "selected_count": 1,
            "min_select": 1,
            "max_select": 2,
            "remaining_select": 1,
            "can_confirm": True,
            "selectable_cards": [
                {"id": "CARD.BASH", "source_pile": "Discard", "cost": 2}
            ],
            "selected_cards": [
                {"id": "CARD.STRIKE", "source_pile": "Discard", "cost": 1}
            ],
        },
    }
    changed = {
        **base,
        "card_selection": {
            **base["card_selection"],
            "selectable_cards": [
                {"id": "CARD.STRIKE", "source_pile": "Discard", "cost": 1}
            ],
            "selected_cards": [
                {"id": "CARD.BASH", "source_pile": "Discard", "cost": 2}
            ],
        },
    }
    actions = [
        {
            "action_handle": "confirm",
            "kind": "confirm_selection",
            "model_action_kind": "card_selection",
            "model_action_variant": "confirm",
        }
    ]

    first = encoder.encode(base, actions).batch.world
    second = encoder.encode(changed, actions).batch.world
    assert not torch.equal(first.entity_ids, second.entity_ids)


def test_live_checkbox_options_and_confirmation_mode_are_world_facts() -> None:
    encoder = _encoder()
    base = {
        "phase": "combat",
        "decision_domain": "combat",
        "player": {"hp": 50, "max_hp": 80},
        "card_selection": {
            "mode": "SimpleGrid",
            "prompt_id": "card.TEST.checkbox_selection",
            "operation_type": "discard",
            "source_zone": "Hand",
            "destination_zone": "Discard",
            "selected_count": 1,
            "min_select": 1,
            "max_select": 2,
            "requires_manual_confirmation": True,
            "options": [
                {
                    "option_index": 0,
                    "is_selected": True,
                    "card": {"id": "CARD.STRIKE", "pile": "Hand", "cost": 1},
                },
                {
                    "option_index": 1,
                    "is_selected": False,
                    "card": {"id": "CARD.DEFEND", "pile": "Hand", "cost": 1},
                },
            ],
        },
    }
    swapped = {
        **base,
        "card_selection": {
            **base["card_selection"],
            "options": [
                {
                    "option_index": 0,
                    "is_selected": False,
                    "card": {"id": "CARD.STRIKE", "pile": "Hand", "cost": 1},
                },
                {
                    "option_index": 1,
                    "is_selected": True,
                    "card": {"id": "CARD.DEFEND", "pile": "Hand", "cost": 1},
                },
            ],
        },
    }
    automatic = {
        **base,
        "card_selection": {
            **base["card_selection"],
            "requires_manual_confirmation": False,
        },
    }
    actions = [
        {
            "action_handle": "confirm",
            "kind": "confirm_selection",
            "model_action_kind": "card_selection",
            "model_action_variant": "confirm",
        }
    ]

    first = encoder.encode(base, actions).batch.world
    second = encoder.encode(swapped, actions).batch.world
    third = encoder.encode(automatic, actions).batch.world
    assert not torch.equal(first.entity_ids, second.entity_ids)
    assert not torch.equal(first.features, third.features)


def test_selection_candidate_keeps_physical_source_zone_separate_from_membership() -> None:
    encoder = _encoder()
    observation = {
        "phase": "combat",
        "decision_domain": "combat",
        "player": {"hp": 50, "max_hp": 80},
    }
    actions = [
        {
            "action_handle": "discard-card",
            "kind": "select_card",
            "model_action_kind": "card_selection",
            "model_action_variant": "select",
            "card": {
                "id": "CARD.STRIKE",
                "pile": "Discard",
                "selection_membership": "selectable",
            },
        },
        {
            "action_handle": "hand-card",
            "kind": "select_hand_card",
            "model_action_kind": "card_selection",
            "model_action_variant": "select",
            "card": {
                "id": "CARD.STRIKE",
                "pile": "Hand",
                "selection_membership": "selectable",
            },
        },
        {
            "action_handle": "selected-discard-card",
            "kind": "deselect_card",
            "model_action_kind": "card_selection",
            "model_action_variant": "deselect",
            "card": {
                "id": "CARD.STRIKE",
                "pile": "Discard",
                "selection_membership": "selected",
                "is_selected": True,
            },
        },
    ]

    candidates = encoder.encode(observation, actions).batch.candidates
    assert not torch.equal(
        candidates.local_features[:, 0], candidates.local_features[:, 1]
    )
    assert not torch.equal(candidates.role_ids[:, 0], candidates.role_ids[:, 2])


def test_public_discard_and_exhaust_composition_is_set_like_but_not_count_only() -> None:
    encoder = _encoder()
    base = {
        "phase": "combat",
        "decision_domain": "combat",
        "player": {"hp": 50, "max_hp": 80},
        "combat": {
            "in_progress": True,
            "discard_pile": [
                {"id": "CARD.STRIKE", "pile": "Discard"},
                {"id": "CARD.DEFEND", "pile": "Discard"},
            ],
            "exhaust_pile": [{"id": "CARD.BURN", "pile": "Exhaust"}],
            "enemies": [],
        },
    }
    permuted = {
        **base,
        "combat": {
            **base["combat"],
            "discard_pile": list(reversed(base["combat"]["discard_pile"])),
        },
    }
    transformed = {
        **base,
        "combat": {
            **base["combat"],
            "discard_pile": [
                {"id": "CARD.STRIKE", "pile": "Discard"},
                {"id": "CARD.BASH", "pile": "Discard"},
            ],
        },
    }

    first = encoder.encode(base, _actions()).batch.world
    second = encoder.encode(permuted, _actions()).batch.world
    third = encoder.encode(transformed, _actions()).batch.world
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
    assert not torch.equal(first.entity_ids, third.entity_ids)


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
            "discard_pile": {
                "count": 1,
                "cards": [{"id": "CARD.SECRET_C", "pile": "Discard"}],
            },
            "exhaust_pile": {"count": 0, "cards": []},
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
            # Draw composition is redacted by live and therefore collapses to
            # a count. Discard/exhaust are public and must retain identity.
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
    with pytest.raises(ValueError, match="at least 224"):
        GroundedEncodingConfig(feature_dim=223)


def test_encoding_contract_has_stable_checkpoint_identity() -> None:
    identity = grounding_encoding_identity()

    assert identity["version"] == GROUNDING_ENCODING_VERSION
    assert identity["min_token_feature_dim"] == 224
    assert identity["feature_abi_end"] <= 224
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


def test_encoded_snapshot_is_compact_pickleable_and_exactly_collates() -> None:
    encoder = _encoder()
    encoded = encoder.encode(_observation(), _actions())
    serialized = pickle.dumps(encoded.snapshot, protocol=pickle.HIGHEST_PROTOCOL)
    restored = pickle.loads(serialized)

    restored_batch = encoder.collate_snapshots((restored,))
    original = encoded.batch
    for left, right in (
        (original.world.features, restored_batch.world.features),
        (original.world.mask, restored_batch.world.mask),
        (original.world.type_ids, restored_batch.world.type_ids),
        (original.world.entity_ids, restored_batch.world.entity_ids),
        (original.candidates.features, restored_batch.candidates.features),
        (original.candidates.local_features, restored_batch.candidates.local_features),
        (original.candidates.local_mask, restored_batch.candidates.local_mask),
        (original.candidates.action_mask, restored_batch.candidates.action_mask),
        (original.domain_ids, restored_batch.domain_ids),
    ):
        assert torch.equal(left, right)

    dense_tensors = (
        original.world.features,
        original.world.mask,
        original.world.type_ids,
        original.world.role_ids,
        original.world.owner_ids,
        original.world.entity_ids,
        original.world.entity_aux_ids,
        original.world.zone_ids,
        original.world.order_ids,
        original.candidates.features,
        original.candidates.type_ids,
        original.candidates.role_ids,
        original.candidates.owner_ids,
        original.candidates.entity_ids,
        original.candidates.entity_aux_ids,
        original.candidates.zone_ids,
        original.candidates.target_owner_ids,
        original.candidates.target_entity_ids,
        original.candidates.target_entity_aux_ids,
        original.candidates.local_features,
        original.candidates.local_mask,
        original.candidates.local_type_ids,
        original.candidates.local_role_ids,
        original.candidates.local_owner_ids,
        original.candidates.local_entity_ids,
        original.candidates.local_entity_aux_ids,
        original.candidates.local_zone_ids,
        original.candidates.local_order_ids,
        original.candidates.action_mask,
        original.domain_ids,
    )
    dense_bytes = sum(tensor.numel() * tensor.element_size() for tensor in dense_tensors)
    assert encoded.snapshot.storage_nbytes() < dense_bytes // 10
    assert len(serialized) < dense_bytes // 5


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


@pytest.mark.parametrize(
    ("raw", "expected_type", "expected_effects", "expected_labels"),
    [
        (
            {
                "id": "CARD.STRIKE_IRONCLAD",
                "type": "Attack",
                "tags": ["Strike"],
                "dynamic_vars": [
                    {
                        "name": "Damage",
                        "var_type": "DamageVar",
                        "family": "damage",
                        "base_value": 6,
                        "preview_value": 6,
                        "int_value": 6,
                    }
                ],
            },
            "Attack",
            {("Damage", "damage", 6)},
            {("Strike", "card_tag")},
        ),
        (
            {
                "id": "CARD.DEFEND_IRONCLAD",
                "type": "Skill",
                "gains_block": True,
                "tags": ["Defend"],
                "dynamic_vars": [
                    {
                        "name": "Block",
                        "var_type": "BlockVar",
                        "family": "block",
                        "base_value": 5,
                        "preview_value": 5,
                        "int_value": 5,
                    }
                ],
            },
            "Skill",
            {("Block", "block", 5)},
            {("Defend", "card_tag"), ("gains_block", "card_trait")},
        ),
        (
            {
                "id": "CARD.BURN",
                "type": "Status",
                "keywords": ["Unplayable"],
                "has_turn_end_in_hand_effect": True,
                "dynamic_vars": [
                    {
                        "name": "Damage",
                        "var_type": "DamageVar",
                        "family": "damage",
                        "base_value": 2,
                        "preview_value": 2,
                        "int_value": 2,
                    }
                ],
            },
            "Status",
            {("Damage", "damage", 2)},
            {
                ("Unplayable", "card_keyword"),
                ("has_turn_end_in_hand_effect", "card_trait"),
            },
        ),
        (
            {
                "id": "CARD.VOID",
                "type": "Status",
                "keywords": ["Unplayable", "Ethereal"],
                "has_on_draw_effect": True,
                "dynamic_vars": [
                    {
                        "name": "Energy",
                        "var_type": "EnergyVar",
                        "family": "energy",
                        "base_value": 1,
                        "preview_value": 1,
                        "int_value": 1,
                    }
                ],
            },
            "Status",
            {("Energy", "energy", 1)},
            {
                ("Unplayable", "card_keyword"),
                ("Ethereal", "card_keyword"),
                ("has_on_draw_effect", "card_trait"),
            },
        ),
        (
            {
                "id": "CARD.SHAME",
                "type": "Curse",
                "keywords": ["Unplayable"],
                "has_turn_end_in_hand_effect": True,
                "dynamic_vars": [
                    {
                        "name": "Frail",
                        "var_type": "DynamicVar",
                        "family": "value",
                        "base_value": 1,
                        "preview_value": 1,
                        "int_value": 1,
                    }
                ],
            },
            "Curse",
            {("Frail", "value", 1)},
            {
                ("Unplayable", "card_keyword"),
                ("has_turn_end_in_hand_effect", "card_trait"),
            },
        ),
        (
            {
                "id": "CARD.DEADLY_POISON",
                "type": "Skill",
                "dynamic_vars": [
                    {
                        "name": "PoisonPower",
                        "var_type": "PowerVar`1",
                        "family": "power",
                        "power_type": "PoisonPower",
                        "base_value": 5,
                        "preview_value": 5,
                        "int_value": 5,
                    }
                ],
            },
            "Skill",
            {("PoisonPower", "power", 5)},
            set(),
        ),
        (
            {
                "id": "CARD.BURNING_PACT",
                "type": "Skill",
                "hover_tip_ids": [
                    "LocString with Title=static_hover_tips.exhaust.title"
                ],
                "dynamic_vars": [
                    {
                        "name": "Cards",
                        "var_type": "CardsVar",
                        "family": "cards",
                        "base_value": 2,
                        "preview_value": 2,
                        "int_value": 2,
                    }
                ],
            },
            "Skill",
            {("Cards", "cards", 2)},
            {
                (
                    "LocString with Title=static_hover_tips.exhaust.title",
                    "card_hover_tip",
                )
            },
        ),
    ],
)
def test_runtime_card_facts_preserve_representative_mechanics(
    raw: dict[str, object],
    expected_type: str,
    expected_effects: set[tuple[str, str, int]],
    expected_labels: set[tuple[str, str]],
) -> None:
    card = _canonical_card(raw)
    assert card["type"] == expected_type
    effects = {
        (str(effect["id"]), str(effect["type"]), int(effect["current_value"]))
        for effect in card.get("dynamic_vars", [])
    }
    labels = {
        (str(label["id"]), str(label["type"]))
        for field in ("keywords", "tags", "hover_tips", "traits")
        for label in card.get(field, [])
    }
    assert effects == expected_effects
    assert labels == expected_labels


def test_dynamic_var_values_use_named_non_colliding_feature_slots() -> None:
    encoder = _encoder()
    observation = _observation()
    observation["player"]["hand"] = [
        {
            "id": "CARD.TEST",
            "type": "Attack",
            "dynamic_vars": [
                {
                    "name": "Damage",
                    "var_type": "DamageVar",
                    "family": "damage",
                    "base_value": 7,
                    "enchanted_value": 8,
                    "preview_value": 9,
                    "int_value": 7,
                    "was_just_upgraded": True,
                }
            ],
        }
    ]
    encoded = encoder.encode(observation, _actions())
    entity_id = _hash_id("entity", "Damage", encoder.config.entity_vocab_size)
    rows = torch.nonzero(
        encoded.batch.world.entity_ids[0] == entity_id,
        as_tuple=False,
    ).flatten()
    assert rows.numel() == 1
    features = encoded.batch.world.features[0, int(rows.item())]
    expected = {
        "base_value": 7,
        "enchanted_value": 8,
        "current_value": 9,
        "int_value": 7,
        "was_just_upgraded": True,
    }
    assert len(set(_DYNAMIC_VALUE_SLOT_BY_KEY.values())) == len(expected)
    for key, raw_value in expected.items():
        assert features[_DYNAMIC_VALUE_SLOT_BY_KEY[key]].item() == pytest.approx(
            _bounded_number(raw_value)
        )


def test_card_cost_rarity_and_x_cost_transport_aliases_are_preserved() -> None:
    live = _canonical_card(
        {
            "id": "CARD.TEST_LIVE",
            "rarity": "Rare",
            "canonical_energy_cost": 3,
            "resolved_energy_cost": 1,
            "costs_x": False,
        }
    )
    assert live["cost"] == 1
    assert live["rarity"] == "Rare"

    catalog = _canonical_card(
        {
            "card_id": "TEST_X",
            "rarity": "Uncommon",
            "base_cost": -1,
            "is_x_cost": True,
        }
    )
    assert catalog["cost"] == -1
    assert catalog["rarity"] == "Uncommon"
    assert {
        (str(trait["id"]), str(trait["type"]))
        for trait in catalog["traits"]
    } == {("costs_x", "card_trait")}


def test_runtime_mechanics_v6_projects_power_relic_and_potion_state() -> None:
    power = _canonical_power(
        {
            "id": "POWER.POISON",
            "class_name": "PoisonPower",
            "amount": 7,
            "amount_on_turn_start": 5,
            "display_amount": 7,
            "type": "Debuff",
            "stack_type": "Counter",
            "type_for_current_amount": "Debuff",
            "is_instanced": False,
            "is_visible": True,
            "allow_negative": False,
            "skip_next_duration_tick": True,
            "owner_side": "Enemy",
            "owner_model_id": "MONSTER.TEST",
            "dynamic_vars": {"Decay": 1},
        }
    )
    assert power["amount_on_turn_start"] == 5
    assert power["display_amount"] == 7
    assert power["stack_type"] == "Counter"
    assert power["skip_next_duration_tick"] is True
    assert power["dynamic_vars"][0]["id"] == "Decay"

    relic = _canonical_relic(
        {
            "id": "RELIC.LIZARD_TAIL",
            "rarity": "Rare",
            "status": "Active",
            "counter": 1,
            "stack_count": 2,
            "is_used_up": False,
            "show_counter": True,
            "is_wax": True,
            "dynamic_vars": {"Heal": 50},
        }
    )
    assert relic["display_amount"] == 1
    assert relic["stack_count"] == 2
    assert relic["status"] == "Active"
    assert relic["is_used_up"] is False
    assert relic["dynamic_vars"][0]["current_value"] == 50

    potion = _canonical_potion(
        {
            "id": "POTION.FIRE",
            "rarity": "Common",
            "usage": "CombatOnly",
            "target_type": "AnyEnemy",
            "can_use_in_combat": True,
            "can_throw_at_ally": False,
            "is_queued": False,
            "passes_custom_usability_check": True,
            "dynamic_vars": {"Damage": 20},
        }
    )
    assert potion["usage"] == "CombatOnly"
    assert potion["can_use_in_combat"] is True
    assert potion["dynamic_vars"][0]["current_value"] == 20


def test_runtime_mechanics_v6_projects_singular_card_modifiers() -> None:
    card = _canonical_card(
        {
            "id": "CARD.TEST",
            "type": "Attack",
            "enchantment": {
                "id": "ENCHANTMENT.SHARP",
                "class_name": "SharpEnchantment",
                "amount": 2,
                "display_amount": 2,
                "status": "Normal",
                "show_amount": True,
                "should_glow_gold": True,
                "dynamic_vars": {"Damage": 3},
            },
            "affliction": {
                "id": "AFFLICTION.BOUND",
                "class_name": "BoundAffliction",
                "amount": 1,
                "is_stackable": False,
                "can_afflict_unplayable_cards": True,
                "has_overlay": True,
            },
        }
    )
    enchantment = card["enchantments"][0]
    affliction = card["afflictions"][0]
    assert enchantment["id"] == "ENCHANTMENT.SHARP"
    assert enchantment["type"] == "enchantment"
    assert enchantment["dynamic_vars"][0]["current_value"] == 3
    assert affliction["id"] == "AFFLICTION.BOUND"
    assert affliction["type"] == "affliction"
    assert affliction["has_overlay"] is True


def test_runtime_mechanics_v6_keeps_observable_boss_state_but_not_hidden_follow_up() -> None:
    raw = {
        "model_id": "MONSTER.BOSS",
        "hp": 300,
        "max_hp": 400,
        "is_alive": True,
        "is_hittable": True,
        "is_primary_enemy": True,
        "is_stunned": False,
        "next_move_id": "PHASE_TWO_ATTACK",
        "follow_up_state_id": "HIDDEN_RANDOM_BRANCH",
        "is_move": True,
        "must_perform_once_before_transitioning": True,
        "can_transition_away": False,
        "spawned_this_turn": False,
        "is_performing_move": False,
        "move_history": ["OPENING", "PHASE_ONE"],
        "powers": [],
        "intents": [{"type": "Attack", "damage": 18, "repeats": 2}],
    }
    first = _canonical_enemy(raw)
    raw["follow_up_state_id"] = "A_DIFFERENT_HIDDEN_BRANCH"
    second = _canonical_enemy(raw)
    assert first == second
    assert first["next_move_state_id"] == "PHASE_TWO_ATTACK"
    assert first["must_perform_once_before_transitioning"] is True
    assert [item["id"] for item in first["move_history"]] == [
        "OPENING",
        "PHASE_ONE",
    ]


def test_runtime_mechanics_v6_event_state_changes_world_encoding() -> None:
    event = _canonical_event(
        {
            "event_id": "EVENT.TEST",
            "layout_type": "Ancient",
            "description_key": "TEST.pages.SECOND.description",
            "is_deterministic": True,
            "is_shared": False,
            "is_finished": False,
            "dynamic_vars": {"Gold": 75},
            "options": [
                {
                    "text_key": "BUY",
                    "title": "Buy",
                    "is_locked": False,
                    "is_chosen": False,
                    "is_proceed": False,
                }
            ],
        }
    )
    assert event["event_id"] == "EVENT.TEST"
    assert event["options"][0]["text_key"] == "BUY"

    encoder = _encoder()
    before = _observation()
    before["event"] = event
    after = deepcopy(before)
    after["event"]["description_key"] = "TEST.pages.THIRD.description"
    first = encoder.encode(before, _actions()).batch.world
    second = encoder.encode(after, _actions()).batch.world
    assert not torch.equal(first.features, second.features)


def test_runtime_mechanics_v6_keeps_act_and_ascension_context() -> None:
    encoder = _encoder()
    first_observation = _observation()
    first_observation["run"] = {"act": 1, "ascension_level": 0}
    second_observation = deepcopy(first_observation)
    second_observation["run"].update({"act": 2, "ascension_level": 10})

    first = encoder.encode(first_observation, _actions()).batch.world
    second = encoder.encode(second_observation, _actions()).batch.world

    assert not torch.equal(first.features, second.features)


def test_runtime_mechanics_v6_set_like_status_and_inventory_are_permutation_invariant() -> None:
    encoder = _encoder()
    first_observation = _observation()
    first_observation["player"]["powers"] = [
        {"id": "POWER.STRENGTH", "amount": 2},
        {"id": "POWER.WEAK", "amount": 1},
    ]
    first_observation["player"]["relics"] = [
        {"id": "RELIC.A", "stack_count": 1},
        {"id": "RELIC.B", "stack_count": 2},
    ]
    first_observation["player"]["potions"] = [
        {"id": "POTION.A", "is_queued": False},
        {"id": "POTION.B", "is_queued": True},
    ]
    second_observation = deepcopy(first_observation)
    for field in ("powers", "relics", "potions"):
        second_observation["player"][field].reverse()

    first = encoder.encode(first_observation, _actions()).batch.world
    second = encoder.encode(second_observation, _actions()).batch.world
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
