from __future__ import annotations

from unittest import mock

import pytest

import sts2_env
from sts2_env import headless_sim_bridge_client as client_module
from sts2_env._sim_translate import translate_to_bridge_shape
from sts2_env._sim_translate_shared import sim_kind_to_model_kind
from sts2_env.headless_sim_bridge_client import HeadlessSimBridgeClient


def test_package_exports_transport_clients_only() -> None:
    assert sts2_env.__all__ == ["BridgeClient", "BridgeError", "HeadlessSimBridgeClient"]


@pytest.mark.parametrize(
    ("sim_kind", "model_kind"),
    [
        ("embark", "character_select"),
        ("start_run", "main_menu"),
        ("skip_rewards", "proceed"),
        ("claim_reward", "reward"),
        ("select_card_reward", "card_reward"),
        ("select_card_option", "card_selection"),
        ("deselect_card", "card_selection"),
        ("combat_deselect_card", "card_selection"),
        ("combat_cancel_selection", "card_selection"),
        ("select_relic", "treasure_relic"),
        ("skip_relic_selection", "treasure_relic"),
        ("claim_treasure_relic", "treasure_relic"),
        ("choose_map_node", "map"),
        ("end_turn", "end_turn"),
    ],
)
def test_simulator_actions_use_closed_live_model_vocabulary(
    sim_kind: str,
    model_kind: str,
) -> None:
    assert sim_kind_to_model_kind(sim_kind) == model_kind


def test_unregistered_simulator_action_enum_fails_closed() -> None:
    with pytest.raises(ValueError, match="no canonical model mapping"):
        sim_kind_to_model_kind("future_unregistered_action")


def _player() -> dict[str, object]:
    return {
        "current_hp": 61,
        "max_hp": 80,
        "gold": 99,
        "hand": [
            {"index": 0, "id": "OFFERING", "name": "Offering", "cost": 0},
            {"index": 1, "id": "STRIKE", "name": "Strike", "cost": 1},
        ],
        "deck": [{"id": "STRIKE"}, {"id": "DEFEND"}],
        "potions": [{"slot": 0, "id": "FIRE_POTION"}],
        "status": [{"id": "SURROUNDED_POWER", "amount": 1}],
    }


def test_selection_actions_keep_simulator_order_membership_and_enabled_state() -> None:
    sim_state = {
        "state_type": "card_select",
        "card_select": {
            "player": _player(),
            "cards": [
                {"index": 0, "id": "STRIKE"},
                {"index": 1, "id": "DEFEND"},
            ],
            "selected_cards": [{"index": 0, "id": "STRIKE"}],
            "selected_count": 1,
            "remaining_picks": 1,
            "max_select": 2,
            "can_confirm": True,
        },
        "legal_actions": [
            {
                "action": "select_card",
                "selection_operation": "select",
                "is_selected": False,
                "index": 1,
            },
            {
                "action": "deselect_card",
                "selection_operation": "deselect",
                "is_selected": True,
                "index": 0,
            },
            {"action": "confirm_selection", "selection_operation": "confirm"},
            {"action": "select_card", "index": 1, "is_enabled": False},
        ],
    }

    translated = translate_to_bridge_shape(sim_state, episode_id="ep-1")
    actions = translated["available_actions"]

    assert [action["action"] for action in actions] == [
        "select_card",
        "deselect_card",
        "confirm_selection",
        "select_card",
    ]
    assert [action["idx"] for action in actions] == [0, 1, 2, 3]
    assert [action["is_enabled"] for action in actions] == [True, True, True, False]
    assert [action["_sim_raw"] for action in actions] == sim_state["legal_actions"]
    assert actions[0]["kind"] == "select_card"
    assert actions[0]["model_action_variant"] == "select"
    assert actions[0]["card"]["pile"] == "Select"
    assert actions[1]["kind"] == "deselect_card"
    assert actions[1]["model_action_variant"] == "deselect"
    assert actions[1]["card"]["pile"] == "Selected"
    assert actions[2]["kind"] == "confirm_selection"
    assert actions[2]["model_action_variant"] == "confirm"
    assert all(action["action_handle"].startswith("sim:") for action in actions)


def test_selection_operation_metadata_mismatch_fails_closed() -> None:
    with pytest.raises(ValueError, match="inconsistent operation metadata"):
        translate_to_bridge_shape(
            {
                "state_type": "card_select",
                "card_select": {"player": _player(), "cards": []},
                "legal_actions": [
                    {
                        "action": "deselect_card",
                        "selection_operation": "select",
                        "index": 0,
                    }
                ],
            },
            episode_id="ep-invalid-selection-semantics",
        )


@pytest.mark.parametrize(
    ("action", "selection_operation", "is_selected"),
    [
        ("select_card", "select", True),
        ("deselect_card", "deselect", False),
        ("deselect_card", "deselect", "true"),
    ],
)
def test_selection_membership_metadata_mismatch_fails_closed(
    action: str,
    selection_operation: str,
    is_selected: object,
) -> None:
    with pytest.raises(ValueError, match="membership metadata"):
        translate_to_bridge_shape(
            {
                "state_type": "card_select",
                "card_select": {"player": _player(), "cards": []},
                "legal_actions": [
                    {
                        "action": action,
                        "selection_operation": selection_operation,
                        "is_selected": is_selected,
                        "index": 0,
                    }
                ],
            },
            episode_id="ep-invalid-selection-membership",
        )


def test_event_text_and_route_graph_are_copied_without_inferred_features() -> None:
    event_state = {
        "state_type": "event",
        "event": {
            "player": _player(),
            "options": [
                {
                    "index": 0,
                    "text": "Lose 12 HP. Gain a relic and enter combat.",
                    "is_locked": False,
                }
            ],
        },
        "legal_actions": [{"action": "choose_event_option", "index": 0}],
    }
    translated_event = translate_to_bridge_shape(event_state, episode_id="ep-event")
    option = translated_event["event"]["options"][0]
    candidate = translated_event["available_actions"][0]

    assert option["text"] == "Lose 12 HP. Gain a relic and enter combat."
    assert "effect_deltas" not in option
    assert "effect_deltas" not in candidate["option"]

    map_state = {
        "state_type": "map",
        "map": {
            "player": _player(),
            "next_options": [
                {"index": 0, "col": 1, "row": 3, "point_type": "elite"},
                {"index": 1, "col": 2, "row": 3, "point_type": "event"},
            ],
            "nodes": [
                {
                    "col": 1,
                    "row": 3,
                    "point_type": "elite",
                    "children": [[1, 4], [2, 4]],
                }
            ],
        },
        "legal_actions": [
            {"action": "choose_map_node", "index": 1},
            {"action": "choose_map_node", "index": 0},
        ],
    }
    translated_map = translate_to_bridge_shape(map_state, episode_id="ep-map")

    assert [action["index"] for action in translated_map["available_actions"]] == [1, 0]
    assert translated_map["map"]["nodes"][0]["children"] == [[1, 4], [2, 4]]
    for action in translated_map["available_actions"]:
        assert "route_summary" not in action
        assert "route_nodes" not in action


def test_combat_entities_have_no_mechanic_or_self_damage_derivations() -> None:
    player = _player()
    sim_state = {
        "state_type": "monster",
        "battle": {
            "player": player,
            "round": 2,
            "enemies": [
                {
                    "combat_id": 7,
                    "entity_id": "INSATIABLE",
                    "name": "Insatiable",
                    "hp": 120,
                    "max_hp": 200,
                    "next_move_id": "COUNTDOWN",
                    "status": [{"id": "BACK_ATTACK_LEFT_POWER", "amount": 1}],
                    "intents": [{"type": "Attack", "damage": 15, "repeats": 2}],
                }
            ],
        },
        "legal_actions": [
            {"action": "play_card", "card_index": 0, "target_id": 7},
            {"action": "end_turn"},
        ],
    }

    translated = translate_to_bridge_shape(sim_state, episode_id="ep-combat")
    combat = translated["combat"]
    enemy = combat["enemies"][0]
    card = translated["available_actions"][0]["card"]

    assert translated["player"]["hp"] == 61
    assert [item["id"] for item in translated["player"]["deck"]] == [
        "CARD.STRIKE",
        "CARD.DEFEND",
    ]
    assert combat["in_progress"] is True
    assert "self_inflicted_hp_loss_cumulative" not in combat
    assert "incoming_damage_multiplier" not in enemy
    assert "static_traits" not in enemy
    assert "reactive_triggers" not in enemy
    assert "phase_rules" not in enemy
    assert "effect_preview" not in card
    assert "canonical_text" not in card


def test_malformed_legal_action_fails_instead_of_being_silently_filtered() -> None:
    with pytest.raises(TypeError, match="legal action 1"):
        translate_to_bridge_shape(
            {
                "state_type": "map",
                "legal_actions": [{"action": "proceed"}, "not-an-action"],
            },
            episode_id="ep-invalid",
        )


def test_end_turn_dispatch_is_exactly_one_simulator_mutation() -> None:
    client = HeadlessSimBridgeClient.__new__(HeadlessSimBridgeClient)
    client._current_episode_id = "sim-ep-1"
    client._last_legal_actions = [
        {
            "action_id": "sim:0:end_turn",
            "_sim_raw": {"action": "end_turn"},
        }
    ]
    client._rpc = mock.Mock(
        return_value={
            "accepted": True,
            "reward": 0.25,
            "state": {"state_type": "combat", "terminal": False},
        }
    )

    with mock.patch.object(
        client_module,
        "_build_bridge_step_response",
        return_value={"episode_id": "sim-ep-1"},
    ):
        client.step("sim-ep-1", action_index=0)

    client._rpc.assert_called_once_with("step", {"action": "end_turn"}, timeout_s=20.0)
