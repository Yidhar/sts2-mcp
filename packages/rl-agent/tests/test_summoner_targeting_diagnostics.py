from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from muzero.diagnostics.summoner_targeting import (  # noqa: E402
    SummonerTargetingEpisodeTracker,
    build_summoner_targeting_payload,
)


def _obs(*, summoner_hp: int = 6, summon_hp: int = 10, summon_intent: int = 0, player_hp: int = 30):
    return {
        "player": {"hp": player_hp, "max_hp": 80, "block": 0, "energy": 2},
        "combat": {
            "energy": 2,
            "incoming_damage": summon_intent,
            "enemies": [
                {"combat_id": 1, "name": "雾菇", "hp": summoner_hp, "block": 0, "intent_damage": 0},
                {"combat_id": 2, "name": "利齿之眼", "hp": summon_hp, "block": 0, "intent_damage": summon_intent},
            ],
        },
    }


def _attack(idx: int, title: str, target_id: int, target_name: str, damage: int):
    return {
        "action_id": f"play_card:{idx}:{target_id}",
        "kind": "play_card",
        "card": {"id": f"CARD.{title}", "title": title, "type": "Attack"},
        "target": {"combat_id": target_id, "name": target_name},
        "damage": damage,
    }


def test_flags_selected_summon_over_available_lethal_summoner():
    actions = [
        _attack(0, "打击", 2, "利齿之眼", 5),
        _attack(1, "打击", 1, "雾菇", 6),
    ]

    payload = build_summoner_targeting_payload(
        raw_obs=_obs(),
        legal_actions=actions,
        action_mask=np.ones(len(actions), dtype=np.float32),
        selected_idx=0,
        encounter_id="encounter.living_fog_normal",
        progress={"floor": 7, "act_id": 1, "room_type": "monster"},
    )

    assert payload is not None
    assert payload["lethal_summoner_available"] is True
    assert payload["selected_summon_over_lethal_summoner"] is True
    assert payload["selected_summoner_when_lethal_available"] is False
    assert payload["selected_action"]["target_is_summon"] is True


def test_selected_lethal_summoner_is_good_when_available():
    actions = [
        _attack(0, "打击", 2, "利齿之眼", 5),
        _attack(1, "打击", 1, "雾菇", 6),
    ]

    payload = build_summoner_targeting_payload(
        raw_obs=_obs(),
        legal_actions=actions,
        action_mask=np.ones(len(actions), dtype=np.float32),
        selected_idx=1,
        encounter_id="encounter.living_fog_normal",
    )

    assert payload is not None
    assert payload["lethal_summoner_available"] is True
    assert payload["selected_summoner_when_lethal_available"] is True
    assert payload["selected_summon_over_lethal_summoner"] is False
    assert payload.get("exception_reason", "") == ""


def test_cross_card_lethal_summoner_bucket_is_explicit():
    actions = [
        _attack(0, "打击", 2, "利齿之眼", 5),
        _attack(1, "余烬", 1, "雾菇", 6),
    ]

    payload = build_summoner_targeting_payload(
        raw_obs=_obs(),
        legal_actions=actions,
        action_mask=np.ones(len(actions), dtype=np.float32),
        selected_idx=0,
        encounter_id="encounter.living_fog_normal",
    )

    assert payload is not None
    assert payload["lethal_summoner_available"] is True
    assert payload["cross_card_lethal_summoner_available"] is True
    assert payload["selected_summon_over_cross_card_lethal_summoner"] is True
    assert payload["best_cross_card_lethal_summoner_candidate"]["card_title"] == "余烬"


def test_summon_lethal_incoming_exception_suppresses_bad_flag():
    actions = [
        _attack(0, "打击", 2, "利齿之眼", 10),
        _attack(1, "打击", 1, "雾菇", 6),
    ]

    payload = build_summoner_targeting_payload(
        raw_obs=_obs(summon_hp=10, summon_intent=80, player_hp=20),
        legal_actions=actions,
        action_mask=np.ones(len(actions), dtype=np.float32),
        selected_idx=0,
        encounter_id="encounter.living_fog_normal",
    )

    assert payload is not None
    assert payload["lethal_summoner_available"] is True
    assert payload["exception_reason"] == "summon_lethal_incoming"
    assert payload["selected_summon_over_lethal_summoner"] is False


def test_aoe_exception_suppresses_bad_flag():
    actions = [
        {
            "action_id": "play_card:0:all",
            "kind": "play_card",
            "card": {"id": "CARD.CLEAVE", "title": "顺劈斩", "type": "Attack"},
            "semantic": {"roles": ["aoe", "damage"]},
            "target_type": "all_enemies",
            "damage": 8,
        },
        _attack(1, "打击", 1, "雾菇", 6),
    ]

    payload = build_summoner_targeting_payload(
        raw_obs=_obs(),
        legal_actions=actions,
        action_mask=np.ones(len(actions), dtype=np.float32),
        selected_idx=0,
        encounter_id="encounter.living_fog_normal",
    )

    assert payload is not None
    assert payload["lethal_summoner_available"] is True
    assert payload["exception_reason"] == "selected_aoe"
    assert payload["selected_summon_over_lethal_summoner"] is False


def test_ordinary_non_summon_fight_returns_none():
    raw_obs = {
        "combat": {
            "enemies": [
                {"combat_id": 1, "name": "邪教徒", "hp": 20},
                {"combat_id": 2, "name": "史莱姆", "hp": 10},
            ]
        }
    }

    payload = build_summoner_targeting_payload(
        raw_obs=raw_obs,
        legal_actions=[_attack(0, "打击", 1, "邪教徒", 6)],
        action_mask=np.ones(1, dtype=np.float32),
        selected_idx=0,
        encounter_id="encounter.cultist_normal",
    )

    assert payload is None


def test_episode_tracker_metadata_uses_lethal_denominator_for_bad_rates():
    tracker = SummonerTargetingEpisodeTracker()
    tracker.update(
        {
            "summoner_present": True,
            "summon_present": True,
            "selected_action": {"target_is_summon": True},
            "lethal_summoner_available": True,
            "selected_summon_over_lethal_summoner": True,
            "candidate_attack_count": 2,
        }
    )
    tracker.update(
        {
            "summoner_present": True,
            "summon_present": True,
            "selected_action": {"target_is_summoner": True},
            "lethal_summoner_available": True,
            "selected_summoner_when_lethal_available": True,
            "candidate_attack_count": 3,
        }
    )

    meta = tracker.as_metadata()

    assert meta["summoner_targeting_seen_count"] == 2.0
    assert meta["summoner_targeting_selected_summon_over_lethal_summoner_rate"] == 0.5
    assert meta["summoner_targeting_selected_summoner_when_lethal_available_rate"] == 0.5
    assert meta["summoner_targeting_candidate_attack_count_mean"] == 2.5


def test_episode_tracker_cross_card_rates_use_cross_card_denominator():
    tracker = SummonerTargetingEpisodeTracker()
    tracker.update(
        {
            "summoner_present": True,
            "summon_present": True,
            "selected_action": {"target_is_summon": True},
            "lethal_summoner_available": True,
            "cross_card_lethal_summoner_available": True,
            "selected_summon_over_cross_card_lethal_summoner": True,
            "candidate_attack_count": 2,
        }
    )
    tracker.update(
        {
            "summoner_present": True,
            "summon_present": True,
            "selected_action": {"target_is_summoner": True},
            "lethal_summoner_available": True,
            "cross_card_lethal_summoner_available": False,
            "candidate_attack_count": 3,
        }
    )

    meta = tracker.as_metadata()

    assert meta["summoner_targeting_cross_card_lethal_summoner_available_rate"] == 0.5
    assert meta["summoner_targeting_selected_summon_over_cross_card_lethal_summoner_rate"] == 1.0
