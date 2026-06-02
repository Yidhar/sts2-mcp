from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from muzero.combat_quality.target_priority_guard import TargetPriorityGuardMixin  # noqa: E402


class DummyTargetPriorityGuard(TargetPriorityGuardMixin):
    def __init__(self) -> None:
        self.last_dump = None

    def _dump_combat_hard_guard_record(self, **kwargs):
        self.last_dump = kwargs


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


def _call_guard(
    action_idx: int,
    actions: list[dict],
    raw_obs: dict,
    *,
    mask: np.ndarray | None = None,
    encounter: str = "encounter.fogmog_normal",
) -> tuple[int, dict, DummyTargetPriorityGuard]:
    guard = DummyTargetPriorityGuard()
    search_stats: dict = {}
    mask_np = np.ones(len(actions), dtype=np.float32) if mask is None else mask
    new_idx = guard._apply_source_pressure_target_guard(
        action_idx=action_idx,
        legal_count=len(actions),
        legal_actions=actions,
        mask_np=mask_np,
        raw_obs=raw_obs,
        encounter=encounter,
        search_stats=search_stats,
    )
    return new_idx, search_stats, guard


def test_fogmog_zero_intent_summon_over_attacking_source_retargets_same_card():
    actions = [
        _attack(0, "打击+", 2, "利齿之眼", 9),
        _attack(1, "打击+", 1, "雾菇", 9),
        {"action_id": "end_turn", "kind": "end_turn"},
    ]

    new_idx, stats, guard = _call_guard(0, actions, _obs())

    assert new_idx == 1
    assert stats["combat_quality_target_priority_source_pressure_guard_applicable"] == 1.0
    assert stats["combat_quality_target_priority_source_pressure_guard_available"] == 1.0
    assert stats["combat_quality_target_priority_source_pressure_guard_applied"] == 1.0
    assert stats["combat_quality_target_priority_source_pressure_guard_override"] == 1.0
    assert stats["combat_quality_target_priority_source_pressure_guard_original_is_summon"] == 1.0
    assert stats["combat_quality_target_priority_source_pressure_guard_zero_intent_summon"] == 1.0
    assert stats["combat_quality_target_priority_source_pressure_guard_source_intent"] == 15.0
    assert stats["combat_quality_target_priority_source_pressure_guard_summon_intent"] == 0.0
    assert stats["combat_quality_target_priority_source_pressure_guard_original_idx"] == 0.0
    assert stats["combat_quality_target_priority_source_pressure_guard_final_idx"] == 1.0
    assert stats["combat_quality_hard_guard_override_any"] == 1.0
    assert guard.last_dump is not None
    assert guard.last_dump["kind"] == "source_pressure_target"
    assert guard.last_dump["override_idx"] == 1


def test_no_override_when_summon_has_lethal_incoming():
    actions = [
        _attack(0, "打击+", 2, "利齿之眼", 9),
        _attack(1, "打击+", 1, "雾菇", 9),
    ]

    new_idx, stats, guard = _call_guard(
        0,
        actions,
        _obs(summon_hp=6, source_intent=15, summon_intent=80, player_hp=20),
    )

    assert new_idx == 0
    assert stats["combat_quality_target_priority_source_pressure_guard_applicable"] == 1.0
    assert stats["combat_quality_target_priority_source_pressure_guard_exception"] == 1.0
    assert stats["combat_quality_target_priority_source_pressure_guard_available"] == 0.0
    assert stats["combat_quality_target_priority_source_pressure_guard_applied"] == 0.0
    assert guard.last_dump is None


def test_no_override_for_aoe_selection():
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

    new_idx, stats, guard = _call_guard(0, actions, _obs())

    assert new_idx == 0
    assert stats["combat_quality_target_priority_source_pressure_guard_applicable"] == 1.0
    assert stats["combat_quality_target_priority_source_pressure_guard_exception"] == 1.0
    assert stats["combat_quality_target_priority_source_pressure_guard_applied"] == 0.0
    assert guard.last_dump is None


def test_no_override_without_same_card_source_candidate():
    actions = [
        _attack(0, "打击+", 2, "利齿之眼", 9),
        _attack(1, "痛击+", 1, "雾菇", 9),
    ]

    new_idx, stats, guard = _call_guard(0, actions, _obs())

    assert new_idx == 0
    assert stats["combat_quality_target_priority_source_pressure_guard_applicable"] == 1.0
    assert stats["combat_quality_target_priority_source_pressure_guard_exception"] == 1.0
    assert stats["combat_quality_target_priority_source_pressure_guard_available"] == 0.0
    assert stats["combat_quality_target_priority_source_pressure_guard_applied"] == 0.0
    assert guard.last_dump is None


def test_no_override_when_same_card_source_candidate_masked_illegal():
    actions = [
        _attack(0, "打击+", 2, "利齿之眼", 9),
        _attack(1, "打击+", 1, "雾菇", 9),
    ]
    mask = np.asarray([1.0, 0.0], dtype=np.float32)

    new_idx, stats, guard = _call_guard(0, actions, _obs(), mask=mask)

    assert new_idx == 0
    assert stats["combat_quality_target_priority_source_pressure_guard_applicable"] == 1.0
    assert stats["combat_quality_target_priority_source_pressure_guard_available"] == 0.0
    assert stats["combat_quality_target_priority_source_pressure_guard_applied"] == 0.0
    assert guard.last_dump is None
