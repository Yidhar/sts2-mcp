from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from muzero.training.self_play import SelfPlayMixin


def test_compact_raw_potion_inventory_finds_nested_transition_state_inventory():
    raw_info = {
        "transition_state": {
            "player": {
                "potions": [
                    {"id": "POTION.LUCKY_TONIC", "title": "幸运药剂", "rarity": "RARE"},
                ]
            }
        }
    }

    dump = SelfPlayMixin._compact_raw_potion_inventory(raw_info, limit=5)

    assert dump == [
        {
            "slot": 0,
            "id": "POTION.LUCKY_TONIC",
            "title": "幸运药剂",
            "rarity": "RARE",
            "empty": False,
            "source": "$.transition_state.player.potions",
        }
    ]


def test_compact_raw_potion_inventory_finds_combat_potions_path():
    raw_obs = {
        "combat": {
            "potions": [
                {"potionId": "POTION.LUCKY_TONIC", "localizedTitle": "幸运药剂"},
            ]
        }
    }

    dump = SelfPlayMixin._compact_raw_potion_inventory(raw_obs, limit=5)

    assert dump[0]["slot"] == 0
    assert dump[0]["id"] == "POTION.LUCKY_TONIC"
    assert dump[0]["title"] == "幸运药剂"
    assert dump[0]["source"] == "$.combat.potions"


def test_episode_potion_history_tracks_lucky_seen_legal_not_selected():
    steps = [
        {
            "decision_domain": "combat",
            "encounter_id": "ENCOUNTER.WATERFALL_GIANT_BOSS",
            "decision_diagnostics": {
                "raw_potions": [
                    {"slot": 0, "id": "POTION.LUCKY_TONIC", "title": "幸运药剂"},
                ],
                "legal_potion_actions": [
                    {
                        "index": 2,
                        "action_id": "use_potion:0:self",
                        "potion_id": "POTION.LUCKY_TONIC",
                        "title": "幸运药剂",
                    },
                ],
                "selected": {
                    "index": 1,
                    "family": "play_card",
                    "action_id": "play_card:0:0",
                    "title": "打击",
                },
            },
        }
    ]

    hist = SelfPlayMixin._episode_potion_history_from_steps(
        steps,
        encounter_id="ENCOUNTER.WATERFALL_GIANT_BOSS",
        tail_limit=8,
    )

    assert hist["potion_history_schema"] == "episode_potion_history_v1"
    assert hist["potion_history_steps_this_combat"] == 1
    assert hist["lucky_seen_this_combat"] is True
    assert hist["lucky_legal_this_combat"] is True
    assert hist["lucky_selected_this_combat"] is False
    assert hist["last_seen_potions_this_combat"][0]["id"] == "POTION.LUCKY_TONIC"
    assert hist["last_seen_legal_potion_actions_this_combat"][0]["potion_id"] == "POTION.LUCKY_TONIC"
    assert hist["selected_potion_actions_this_combat"] == []


def test_episode_potion_history_tracks_lucky_selected():
    steps = [
        {
            "decision_domain": "combat",
            "encounter_id": "ENCOUNTER.SOUL_FYSH_BOSS",
            "decision_diagnostics": {
                "raw_potions": [
                    {"slot": 0, "id": "POTION.LUCKY_TONIC", "title": "幸运补剂"},
                ],
                "legal_potion_actions": [
                    {
                        "index": 0,
                        "action_id": "use_potion:0:self",
                        "potion_id": "POTION.LUCKY_TONIC",
                        "title": "幸运补剂",
                    },
                ],
                "selected": {
                    "index": 0,
                    "family": "use_potion",
                    "action_id": "use_potion:0:self",
                    "potion_id": "POTION.LUCKY_TONIC",
                    "title": "幸运补剂",
                },
            },
        }
    ]

    hist = SelfPlayMixin._episode_potion_history_from_steps(
        steps,
        encounter_id="ENCOUNTER.SOUL_FYSH_BOSS",
        tail_limit=8,
    )

    assert hist["lucky_seen_this_combat"] is True
    assert hist["lucky_legal_this_combat"] is True
    assert hist["lucky_selected_this_combat"] is True
    assert hist["selected_potion_actions_this_combat"][0]["action_id"] == "use_potion:0:self"
