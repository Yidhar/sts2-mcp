import numpy as np

from muzero.strategy.encounters import insatiable, kaiser


def _family(action):
    action_id = str(action.get("action_id") or "")
    if action_id.startswith("play_card"):
        return "play_card"
    if action_id.startswith("use_potion"):
        return "use_potion"
    if action_id == "end_turn":
        return "end_turn"
    return ""


def test_kaiser_target_position_uses_back_attack_power_not_enemy_side():
    raw_obs = {
        "combat": {
            "facing": "left",
            "enemies": [
                {
                    "combat_id": 11,
                    "side": "Enemy",
                    "powers": [{"id": "BACK_ATTACK_RIGHT_POWER"}],
                }
            ],
        }
    }
    action = {
        "action_id": "play_card:0:11",
        "target_combat_id": 11,
        "target": {"combat_id": 11, "side": "Enemy"},
    }

    assert kaiser.action_target_side(action, raw_obs) == "right"
    assert kaiser.action_changes_facing_toward_target(action, raw_obs, semantic_family_fn=_family)


def test_kaiser_faction_side_alone_is_not_a_facing_signal():
    raw_obs = {
        "combat": {
            "facing": "left",
            "enemies": [{"combat_id": 11, "side": "Enemy", "powers": []}],
        }
    }
    action = {
        "action_id": "play_card:0:11",
        "target_combat_id": 11,
        "target": {"combat_id": 11, "side": "Enemy"},
    }

    assert kaiser.action_target_side(action, raw_obs) == ""
    assert not kaiser.action_changes_facing_toward_target(action, raw_obs, semantic_family_fn=_family)


def test_kaiser_find_facing_candidates_respects_mask():
    raw_obs = {
        "combat": {
            "facing": "left",
            "enemies": [
                {"combat_id": 1, "powers": [{"id": "BACK_ATTACK_LEFT_POWER"}]},
                {"combat_id": 2, "powers": [{"id": "BACK_ATTACK_RIGHT_POWER"}]},
            ],
        }
    }
    legal = [
        {"action_id": "play_card:0:1", "target_combat_id": 1},
        {"action_id": "play_card:1:2", "target_combat_id": 2},
        {"action_id": "end_turn"},
    ]
    mask = np.array([1, 1, 1], dtype=np.float32)

    assert kaiser.find_facing_candidates(legal, mask, raw_obs, max_actions=10, semantic_family_fn=_family) == [1]
    mask[1] = 0
    assert kaiser.find_facing_candidates(legal, mask, raw_obs, max_actions=10, semantic_family_fn=_family) == []


def test_insatiable_frantic_escape_prefers_internal_id_with_text_fallback():
    by_id = {"action_id": "play_card:0", "card": {"id": "CARD.FRANTIC_ESCAPE", "title": "x"}}
    by_text = {"action_id": "play_card:1", "card": {"title": "狂乱逃离"}}
    strike = {"action_id": "play_card:2", "card": {"id": "CARD.STRIKE", "title": "Strike"}}

    assert insatiable.is_frantic_escape_action(by_id, semantic_family_fn=_family)
    assert insatiable.is_frantic_escape_action(by_text, semantic_family_fn=_family)
    assert not insatiable.is_frantic_escape_action(strike, semantic_family_fn=_family)


def test_insatiable_sandpit_countdown_context_and_raw_fallback():
    assert insatiable.sandpit_countdown_from_context({"sandpit_countdown_min": 2}, None) == 2.0

    raw_obs = {
        "combat": {
            "enemies": [
                {"powers": [{"id": "OTHER_POWER", "amount": 9}]},
                {"powers": [{"id": "SANDPIT_COUNTDOWN_POWER", "amount": 3}]},
            ]
        }
    }
    assert insatiable.sandpit_countdown_from_context({}, raw_obs) == 3.0
