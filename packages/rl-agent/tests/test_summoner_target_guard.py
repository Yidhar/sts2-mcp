from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from muzero.combat_quality.summoner_target_guard import (  # noqa: E402
    SummonerTargetGuardMixin,
)


class DummySummonerTargetGuard(SummonerTargetGuardMixin):
    pass


def _obs(
    *,
    summoner_hp: int = 6,
    summon_hp: int = 10,
    summon_intent: int = 0,
    player_hp: int = 30,
    player_block: int = 0,
):
    return {
        "player": {"hp": player_hp, "max_hp": 80, "block": player_block, "energy": 2},
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
        "card": {"id": f"CARD.{title}", "title": title, "type": "Attack", "cost": 1},
        "target": {"combat_id": target_id, "name": target_name},
        "damage": damage,
    }


def _call_guard(
    action_idx: int,
    actions: list[dict],
    raw_obs: dict,
    stats: dict | None = None,
    encounter: str = "encounter.living_fog_normal",
) -> tuple[int, dict]:
    guard = DummySummonerTargetGuard()
    search_stats: dict = {} if stats is None else stats
    new_idx = guard._apply_summoner_lethal_retarget_guard(
        action_idx=action_idx,
        legal_count=len(actions),
        legal_actions=actions,
        mask_np=np.ones(len(actions), dtype=np.float32),
        raw_obs=raw_obs,
        encounter=encounter,
        search_stats=search_stats,
    )
    return new_idx, search_stats


def test_retargets_selected_summon_to_lethal_summoner():
    actions = [
        _attack(0, "打击", 2, "利齿之眼", 5),
        _attack(1, "打击", 1, "雾菇", 6),
    ]

    new_idx, stats = _call_guard(0, actions, _obs())

    assert new_idx == 1
    assert stats["combat_quality_summoner_lethal_retarget_guard_applicable"] == 1.0
    assert stats["combat_quality_summoner_lethal_retarget_guard_available"] == 1.0
    assert stats["combat_quality_summoner_lethal_retarget_guard_original_is_summon"] == 1.0
    assert stats["combat_quality_summoner_lethal_retarget_guard_applied"] == 1.0
    assert stats["combat_quality_summoner_lethal_retarget_guard_override"] == 1.0
    assert stats["combat_quality_summoner_lethal_retarget_guard_original_idx"] == 0.0
    assert stats["combat_quality_summoner_lethal_retarget_guard_final_idx"] == 1.0


def test_keeps_selected_lethal_summoner():
    actions = [
        _attack(0, "打击", 2, "利齿之眼", 5),
        _attack(1, "打击", 1, "雾菇", 6),
    ]

    new_idx, stats = _call_guard(1, actions, _obs())

    assert new_idx == 1
    assert stats["combat_quality_summoner_lethal_retarget_guard_applicable"] == 1.0
    assert stats["combat_quality_summoner_lethal_retarget_guard_available"] == 1.0
    assert stats["combat_quality_summoner_lethal_retarget_guard_applied"] == 0.0
    assert stats["combat_quality_summoner_lethal_retarget_guard_override"] == 0.0
    assert stats["combat_quality_summoner_lethal_retarget_guard_final_idx"] == 1.0


def test_keeps_selected_aoe_even_when_summoner_lethal_exists():
    actions = [
        {
            "action_id": "play_card:0:all",
            "kind": "play_card",
            "card": {"id": "CARD.CLEAVE", "title": "顺劈斩", "type": "Attack", "cost": 1},
            "semantic": {"roles": ["aoe", "damage"]},
            "target_type": "all_enemies",
            "damage": 8,
        },
        _attack(1, "打击", 1, "雾菇", 6),
    ]

    new_idx, stats = _call_guard(0, actions, _obs())

    assert new_idx == 0
    assert stats["combat_quality_summoner_lethal_retarget_guard_available"] == 1.0
    assert stats["combat_quality_summoner_lethal_retarget_guard_exception"] == 1.0
    assert stats["combat_quality_summoner_lethal_retarget_guard_applied"] == 0.0


def test_keeps_summon_kill_when_summon_has_lethal_incoming():
    actions = [
        _attack(0, "打击", 2, "利齿之眼", 10),
        _attack(1, "打击", 1, "雾菇", 6),
    ]

    new_idx, stats = _call_guard(
        0,
        actions,
        _obs(summon_hp=10, summon_intent=80, player_hp=20),
    )

    assert new_idx == 0
    assert stats["combat_quality_summoner_lethal_retarget_guard_available"] == 1.0
    assert stats["combat_quality_summoner_lethal_retarget_guard_exception"] == 1.0
    assert stats["combat_quality_summoner_lethal_retarget_guard_applied"] == 0.0


def test_no_override_when_no_lethal_summoner_candidate():
    actions = [
        _attack(0, "打击", 2, "利齿之眼", 5),
        _attack(1, "打击", 1, "雾菇", 6),
    ]

    new_idx, stats = _call_guard(0, actions, _obs(summoner_hp=20))

    assert new_idx == 0
    assert stats["combat_quality_summoner_lethal_retarget_guard_applicable"] == 1.0
    assert stats["combat_quality_summoner_lethal_retarget_guard_available"] == 0.0
    assert stats["combat_quality_summoner_lethal_retarget_guard_applied"] == 0.0


def test_ordinary_non_summon_fight_is_ignored():
    raw_obs = {
        "combat": {
            "enemies": [
                {"combat_id": 1, "name": "邪教徒", "hp": 20},
                {"combat_id": 2, "name": "史莱姆", "hp": 10},
            ]
        }
    }
    actions = [_attack(0, "打击", 1, "邪教徒", 6)]

    new_idx, stats = _call_guard(0, actions, raw_obs, encounter="encounter.cultist_normal")

    assert new_idx == 0
    assert stats["combat_quality_summoner_lethal_retarget_guard_applicable"] == 0.0
    assert stats["combat_quality_summoner_lethal_retarget_guard_available"] == 0.0
    assert stats["combat_quality_summoner_lethal_retarget_guard_applied"] == 0.0
