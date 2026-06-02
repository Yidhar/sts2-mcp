from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from muzero.diagnostics.target_priority import (  # noqa: E402
    TargetPriorityEpisodeTracker,
    build_target_priority_payload,
)


def _obs(
    *,
    summoner_hp: int = 37,
    summon_hp: int = 6,
    source_intent: int = 15,
    summon_intent: int = 0,
    player_hp: int = 30,
    player_block: int = 0,
) -> dict:
    return {
        "player": {"hp": player_hp, "max_hp": 80, "block": player_block, "energy": 2},
        "combat": {
            "energy": 2,
            "incoming_damage": source_intent + summon_intent,
            "enemies": [
                {"combat_id": 1, "name": "雾菇", "hp": summoner_hp, "block": 0, "intent_damage": source_intent},
                {"combat_id": 2, "name": "利齿之眼", "hp": summon_hp, "block": 0, "intent_damage": summon_intent},
            ],
        },
    }


def _attack(idx: int, title: str, target_id: int, target_name: str, damage: int) -> dict:
    return {
        "action_id": f"play_card:{idx}:{target_id}",
        "kind": "play_card",
        "card": {"id": f"CARD.{title}", "title": title, "type": "Attack", "cost": 1},
        "target": {"combat_id": target_id, "name": target_name},
        "damage": damage,
    }


def _payload(actions: list[dict], selected_idx: int, raw_obs: dict | None = None) -> dict:
    payload = build_target_priority_payload(
        raw_obs=raw_obs or _obs(),
        legal_actions=actions,
        action_mask=np.ones(len(actions), dtype=np.float32),
        selected_idx=selected_idx,
        search_policy=np.asarray([0.999, 0.001, 0.0][: len(actions)], dtype=np.float32),
        encounter_id="encounter.fogmog_normal",
        progress={"floor": 7, "act_id": 1, "room_type": "monster"},
    )
    assert payload is not None
    return payload


def test_flags_zero_intent_summon_over_attacking_source_same_card():
    actions = [
        _attack(0, "打击+", 2, "利齿之眼", 9),
        _attack(1, "打击+", 1, "雾菇", 9),
    ]

    payload = _payload(actions, selected_idx=0)

    assert payload["source_present"] is True
    assert payload["summon_present"] is True
    assert payload["selected_summon"] is True
    assert payload["selected_source"] is False
    assert payload["source_pressure_available"] is True
    assert payload["selected_summon_over_source_pressure"] is True
    assert payload["selected_zero_intent_summon_over_attacking_source"] is True
    assert payload.get("source_pressure_exception_reason", "") == ""
    assert payload["best_source_pressure_candidate"]["index"] == 1
    assert payload["source_damage_lost"] == 9.0
    assert payload["summon_overkill"] == 3.0


def test_selected_source_is_not_source_pressure_error():
    actions = [
        _attack(0, "打击+", 2, "利齿之眼", 9),
        _attack(1, "打击+", 1, "雾菇", 9),
    ]

    payload = _payload(actions, selected_idx=1)

    assert payload["selected_source"] is True
    assert payload["selected_summon"] is False
    assert payload["source_pressure_available"] is False
    assert payload["selected_summon_over_source_pressure"] is False
    assert payload["selected_zero_intent_summon_over_attacking_source"] is False


def test_cross_card_lethal_source_is_visible_even_without_same_card_retarget():
    actions = [
        _attack(0, "打击", 2, "利齿之眼", 6),
        _attack(1, "余烬", 1, "雾菇", 37),
    ]

    payload = _payload(actions, selected_idx=0)

    assert payload["lethal_source_available"] is True
    assert payload["cross_card_lethal_source_available"] is True
    assert payload["selected_summon_over_cross_card_lethal_source"] is True
    assert payload["best_cross_card_lethal_source_candidate"]["card_title"] == "余烬"
    assert payload["source_pressure_exception_reason"] == "no_same_card_source_candidate"


def test_summon_lethal_incoming_exception_suppresses_source_pressure_flag():
    actions = [
        _attack(0, "打击+", 2, "利齿之眼", 9),
        _attack(1, "打击+", 1, "雾菇", 9),
    ]

    payload = _payload(
        actions,
        selected_idx=0,
        raw_obs=_obs(summon_hp=6, source_intent=15, summon_intent=80, player_hp=20),
    )

    assert payload["source_pressure_exception_reason"] == "summon_lethal_incoming"
    assert payload["source_pressure_available"] is False
    assert payload["selected_summon_over_source_pressure"] is False


def test_aoe_exception_suppresses_source_pressure_flag():
    actions = [
        {
            "action_id": "play_card:0:all",
            "kind": "play_card",
            "card": {"id": "CARD.CLEAVE", "title": "顺劈斩", "type": "Attack", "cost": 1},
            "semantic": {"roles": ["aoe", "damage"]},
            "target_type": "all_enemies",
            "damage": 8,
        },
        _attack(1, "打击+", 1, "雾菇", 9),
    ]

    payload = _payload(actions, selected_idx=0)

    assert payload["source_pressure_exception_reason"] == "selected_aoe"
    assert payload["source_pressure_available"] is False
    assert payload["selected_summon_over_source_pressure"] is False


def test_comparable_summon_pressure_exception_suppresses_source_pressure_flag():
    actions = [
        _attack(0, "打击+", 2, "利齿之眼", 9),
        _attack(1, "打击+", 1, "雾菇", 9),
    ]

    payload = _payload(
        actions,
        selected_idx=0,
        raw_obs=_obs(source_intent=12, summon_intent=10),
    )

    assert payload["source_pressure_exception_reason"] == "summon_has_comparable_pressure"
    assert payload["source_pressure_available"] is False
    assert payload["selected_summon_over_source_pressure"] is False


def test_tracker_metadata_uses_source_pressure_denominator_for_bad_rates():
    tracker = TargetPriorityEpisodeTracker()
    tracker.update(
        {
            "source_present": True,
            "summon_present": True,
            "selected_summon": True,
            "source_pressure_available": True,
            "selected_summon_over_source_pressure": True,
            "selected_zero_intent_summon_over_attacking_source": True,
            "source_pressure_candidate_count": 1,
            "source_damage_lost": 9,
            "summon_overkill": 3,
        }
    )
    tracker.update(
        {
            "source_present": True,
            "summon_present": True,
            "selected_source": True,
            "source_pressure_available": False,
            "source_pressure_candidate_count": 0,
        }
    )

    meta = tracker.as_metadata()

    assert meta["target_priority_seen_count"] == 2.0
    assert meta["target_priority_selected_summon_rate"] == 0.5
    assert meta["target_priority_selected_source_rate"] == 0.5
    assert meta["target_priority_source_pressure_available_rate"] == 0.5
    assert meta["target_priority_selected_summon_over_source_pressure_rate"] == 1.0
    assert meta["target_priority_selected_zero_intent_summon_over_attacking_source_rate"] == 1.0
    assert meta["target_priority_source_pressure_candidate_count_mean"] == 0.5
    assert meta["target_priority_source_damage_lost_mean"] == 4.5
    assert meta["target_priority_summon_overkill_mean"] == 1.5


def test_tracker_cross_card_lethal_source_rates():
    tracker = TargetPriorityEpisodeTracker()
    tracker.update(
        {
            "source_present": True,
            "summon_present": True,
            "selected_summon": True,
            "cross_card_lethal_source_available": True,
            "selected_summon_over_cross_card_lethal_source": True,
        }
    )
    tracker.update(
        {
            "source_present": True,
            "summon_present": True,
            "selected_source": True,
            "cross_card_lethal_source_available": False,
        }
    )

    meta = tracker.as_metadata()

    assert meta["target_priority_cross_card_lethal_source_available_rate"] == 0.5
    assert meta["target_priority_selected_summon_over_cross_card_lethal_source_rate"] == 1.0
