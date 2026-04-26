from __future__ import annotations

import sys
import unittest
from pathlib import Path

RL_AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(RL_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(RL_AGENT_ROOT))

from sts2_env.combat_tactical_local import analyze_local_combat_turn


def _attack(title: str, *, damage: int, cost: int = 1, index: int = 0) -> dict[str, object]:
    return {
        "id": title.upper().replace(" ", "_"),
        "title": title,
        "type": "Attack",
        "cost": cost,
        "damage": damage,
        "preview_damage": damage,
        "damage_per_hit": damage,
        "hits": 1,
        "can_play": True,
        "index": index,
        "description": f"Deal {damage} damage.",
    }


def _obs(enemy: dict[str, object], hand: list[dict[str, object]]) -> dict[str, object]:
    return {
        "phase": "combat",
        "player": {"hp": 50, "max_hp": 80, "block": 0},
        "combat": {
            "play_phase": True,
            "can_act": True,
            "energy": 3,
            "max_energy": 3,
            "hand": hand,
            "enemies": [enemy],
        },
    }


class CombatTacticalBossMechanicsTest(unittest.TestCase):
    def test_boss_damage_cap_prevents_false_local_lethal(self) -> None:
        enemy = {
            "id": 11,
            "model_id": "MONSTER.VANTOM",
            "name": "Vantom",
            "hp": 5,
            "max_hp": 300,
            "block": 0,
            "powers": [{"id": "SLIPPERY_POWER", "title": "Slippery", "amount": 1}],
            "intent": {"total_damage": 0},
            "is_alive": True,
        }
        analysis = analyze_local_combat_turn(_obs(enemy, [_attack("Big Strike", damage=20)]))
        self.assertTrue(analysis.available)
        self.assertFalse(analysis.lethal_exists)

    def test_revive_marker_prevents_false_local_lethal(self) -> None:
        enemy = {
            "id": 51,
            "model_id": "MONSTER.TEST_SUBJECT",
            "name": "Test Subject Revive",
            "hp": 5,
            "max_hp": 100,
            "block": 0,
            "powers": [{"id": "SECOND_LIFE", "title": "Second Life", "amount": 1}],
            "phase_rules": [{"trait": "revive_once", "description": "Revive once on death."}],
            "intent": {"total_damage": 0},
            "is_alive": True,
        }
        analysis = analyze_local_combat_turn(_obs(enemy, [_attack("Strike", damage=10)]))
        self.assertTrue(analysis.available)
        self.assertFalse(analysis.lethal_exists)

    def test_deathburst_is_counted_as_post_kill_required_block(self) -> None:
        enemy = {
            "id": 41,
            "model_id": "MONSTER.WATERFALL_GIANT",
            "name": "Waterfall Giant",
            "hp": 5,
            "max_hp": 400,
            "block": 0,
            "powers": [],
            "intent": {"total_damage": 0},
            "is_alive": True,
        }
        obs = _obs(enemy, [_attack("Strike", damage=10)])
        obs["run"] = {"room_model": "MONSTER.WATERFALL_GIANT"}
        analysis = analyze_local_combat_turn(obs)
        self.assertTrue(analysis.available)
        self.assertTrue(analysis.lethal_exists)
        self.assertGreater(analysis.lethal_required_block, 0)


if __name__ == "__main__":
    unittest.main()
