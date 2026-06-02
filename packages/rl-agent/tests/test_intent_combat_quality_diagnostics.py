from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from muzero.diagnostics.intent_combat_quality import (  # noqa: E402
    IntentCombatQualityEpisodeTracker,
    build_intent_combat_quality_payload,
)


def _obs(*, hp: int = 20, block: int = 0, incoming: int = 16, enemy_hp: int = 30):
    return {
        "player": {"hp": hp, "max_hp": 80, "block": block, "energy": 2},
        "combat": {
            "energy": 2,
            "incoming_damage": incoming,
            "enemies": [
                {"combat_id": 1, "name": "敌人", "hp": enemy_hp, "block": 0, "intent_damage": incoming},
            ],
        },
    }


def _attack(idx: int, damage: int, *, target_hp: int = 30):
    return {
        "action_id": f"play_card:{idx}:1",
        "kind": "play_card",
        "card": {"id": "CARD.STRIKE", "title": "打击", "type": "Attack"},
        "target": {"combat_id": 1, "name": "敌人"},
        "damage": damage,
    }


def _defend(idx: int, block: int = 5):
    return {
        "action_id": f"play_card:{idx}",
        "kind": "play_card",
        "card": {"id": "CARD.DEFEND", "title": "防御", "type": "Skill"},
        "block": block,
        "semantic": {"roles": ["block"]},
    }


def test_high_pressure_attack_with_block_candidate_is_flagged():
    actions = [_attack(0, 6), _defend(1, 5)]

    payload = build_intent_combat_quality_payload(
        raw_obs=_obs(hp=20, incoming=16),
        legal_actions=actions,
        action_mask=np.ones(len(actions), dtype=np.float32),
        selected_idx=0,
        encounter_id="encounter.gremlin_merc_normal",
    )

    assert payload is not None
    assert payload["high_pressure"] is True
    assert payload["survival_candidate_available"] is True
    assert payload["selected_nonprotective_under_pressure"] is True
    assert payload["best_survival_candidate"]["card_title"] == "防御"


def test_no_pressure_pure_block_with_damage_candidate_is_flagged():
    actions = [_defend(0, 5), _attack(1, 6)]

    payload = build_intent_combat_quality_payload(
        raw_obs=_obs(hp=40, block=4, incoming=4),
        legal_actions=actions,
        action_mask=np.ones(len(actions), dtype=np.float32),
        selected_idx=0,
        encounter_id="encounter.crawler_weak",
    )

    assert payload is not None
    assert payload["no_pressure"] is True
    assert payload["no_pressure_pure_block"] is True


def test_missed_lethal_is_flagged():
    actions = [_defend(0, 5), _attack(1, 20)]

    payload = build_intent_combat_quality_payload(
        raw_obs=_obs(hp=40, incoming=0, enemy_hp=12),
        legal_actions=actions,
        action_mask=np.ones(len(actions), dtype=np.float32),
        selected_idx=0,
        encounter_id="encounter.hallway",
    )

    assert payload is not None
    assert payload["lethal_available"] is True
    assert payload["selected_lethal"] is False
    assert payload["missed_lethal"] is True
    assert payload["best_lethal_candidate"]["index"] == 1


def test_tracker_metadata_rates_use_relevant_denominators():
    tracker = IntentCombatQualityEpisodeTracker()
    tracker.update(
        {
            "high_pressure": True,
            "survival_candidate_available": True,
            "selected_nonprotective_under_pressure": True,
            "lethal_available": True,
            "missed_lethal": True,
            "selected_action": {"is_end_turn": False},
        }
    )
    tracker.update(
        {
            "no_pressure": True,
            "no_pressure_pure_block": True,
            "survival_candidate_available": False,
            "lethal_available": False,
            "selected_action": {"is_end_turn": True},
        }
    )

    meta = tracker.as_metadata()

    assert meta["intent_combat_quality_seen_count"] == 2.0
    assert meta["intent_combat_quality_high_pressure_rate"] == 0.5
    assert meta["intent_combat_quality_missed_lethal_rate"] == 1.0
    assert meta["intent_combat_quality_selected_nonprotective_under_pressure_rate"] == 1.0
    assert meta["intent_combat_quality_no_pressure_pure_block_rate"] == 1.0
    assert meta["intent_combat_quality_selected_end_turn_rate"] == 0.5
