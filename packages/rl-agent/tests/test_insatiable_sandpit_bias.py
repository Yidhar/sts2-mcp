from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

RL_AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(RL_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(RL_AGENT_ROOT))

from muzero.train import MAX_ACTIONS, MuZeroTrainer


class DummyEnv:
    def __init__(self, raw_obs):
        self._last_obs_raw = raw_obs

    @property
    def unwrapped(self):
        return self


def _trainer(raw_obs):
    trainer = MuZeroTrainer.__new__(MuZeroTrainer)
    trainer.env = DummyEnv(raw_obs)
    return trainer


def _frantic():
    return {
        "kind": "play_card",
        "semantic": {"family": "play_card"},
        "action_id": "play:CARD.FRANTIC_ESCAPE",
        "card": {
            "id": "CARD.FRANTIC_ESCAPE",
            "normalized_id": "frantic_escape",
            "title": "狂乱逃离",
            "type": "Status",
            "cost": 1,
        },
    }


def _frantic_class_only():
    return {
        "kind": "play_card",
        "semantic": {"family": "play_card"},
        "action_id": "play:generated_status",
        "card": {
            "class_name": "FranticEscape",
            "kind": "FranticEscape",
            "type": "Status",
            "cost": 1,
        },
    }


def _strike(damage=6, target_hp=100):
    return {
        "kind": "play_card",
        "semantic": {"family": "play_card", "damage": damage},
        "action_id": "play:CARD.STRIKE",
        "card": {"id": "CARD.STRIKE", "title": "Strike", "type": "Attack", "damage": damage},
        "damage": damage,
        "target": {"combat_id": 1, "hp": target_hp},
    }


def _end_turn():
    return {"kind": "end_turn", "semantic": {"family": "end_turn"}, "action_id": "end_turn"}


def _raw(sandpit=1, enemy_hp=100):
    return {
        "encounter_id": "MONSTER.THE_INSATIABLE",
        "player": {"energy": 1, "hp": 50, "max_hp": 80},
        "combat": {
            "player": {"energy": 1, "hp": 50, "max_hp": 80},
            "energy": 1,
            # Source-confirmed live shape: SandpitPower is owned by The
            # Insatiable enemy while targeting the player.  Keep player_powers
            # empty so planner bias tests exercise the real bridge path.
            "player_powers": [],
            "hand": [{"id": "CARD.FRANTIC_ESCAPE", "title": "狂乱逃离"}],
            "draw_pile": [],
            "discard_pile": [],
            "exhaust_pile": [],
            "enemies": [
                {
                    "id": "MONSTER.THE_INSATIABLE",
                    "model_id": "MONSTER.THE_INSATIABLE",
                    "combat_id": 1,
                    "name": "The Insatiable",
                    "hp": enemy_hp,
                    "max_hp": 200,
                    "powers": [
                        {
                            "id": "POWER.SANDPIT_POWER",
                            "model_id": "POWER.SANDPIT_POWER",
                            "class_name": "SandpitPower",
                            "kind": "SandpitPower",
                            "title": "Sandpit",
                            "amount": sandpit,
                            "display_amount": sandpit,
                            "stack_type": "Counter",
                        }
                    ],
                    "intent": {},
                    "is_alive": True,
                }
            ],
        },
    }


class InsatiableSandpitBiasTests(unittest.TestCase):
    def test_sandpit_one_strongly_prefers_frantic_escape(self):
        raw = _raw(sandpit=1, enemy_hp=100)
        trainer = _trainer(raw)
        actions = [_frantic(), _strike(), _end_turn()]
        mask = np.ones(MAX_ACTIONS, dtype=np.float32)
        bias, stats, _ = trainer._combat_action_quality_bias({}, mask, actions)

        self.assertGreater(bias[0], bias[1])
        self.assertGreaterEqual(bias[0], 4.0)
        self.assertEqual(stats["combat_quality_insatiable_sandpit_1"], 1.0)
        self.assertEqual(stats["combat_quality_insatiable_frantic_escape_available"], 1.0)
        self.assertEqual(stats["combat_quality_insatiable_frantic_escape_bonus_applied"], 1.0)
        self.assertGreater(stats["combat_quality_insatiable_non_escape_at1_penalty_count"], 0.0)

    def test_class_metadata_only_frantic_action_is_candidate(self):
        raw = _raw(sandpit=1, enemy_hp=100)
        trainer = _trainer(raw)
        actions = [_frantic_class_only(), _strike(), _end_turn()]
        mask = np.ones(MAX_ACTIONS, dtype=np.float32)
        bias, stats, _ = trainer._combat_action_quality_bias({}, mask, actions)

        self.assertGreaterEqual(bias[0], 4.0)
        self.assertEqual(stats["combat_quality_insatiable_frantic_escape_candidate_count"], 1.0)
        self.assertEqual(stats["combat_quality_insatiable_frantic_escape_bonus_applied"], 1.0)

    def test_sandpit_two_boosts_frantic_escape(self):
        raw = _raw(sandpit=2, enemy_hp=100)
        trainer = _trainer(raw)
        actions = [_frantic(), _strike(), _end_turn()]
        mask = np.ones(MAX_ACTIONS, dtype=np.float32)
        bias, stats, _ = trainer._combat_action_quality_bias({}, mask, actions)

        self.assertGreater(bias[0], 2.0)
        self.assertEqual(stats["combat_quality_insatiable_sandpit_lt3"], 1.0)

    def test_sandpit_four_does_not_force_early_escape(self):
        raw = _raw(sandpit=4, enemy_hp=100)
        trainer = _trainer(raw)
        actions = [_frantic(), _strike(), _end_turn()]
        mask = np.ones(MAX_ACTIONS, dtype=np.float32)
        bias, stats, _ = trainer._combat_action_quality_bias({}, mask, actions)

        self.assertLessEqual(bias[0], 1.0)
        self.assertLess(stats["combat_quality_insatiable_frantic_escape_urgency"], 0.5)

    def test_sandpit_zero_is_terminal_not_actionable_escape_bias(self):
        raw = _raw(sandpit=0, enemy_hp=100)
        trainer = _trainer(raw)
        actions = [_frantic(), _strike(), _end_turn()]
        mask = np.ones(MAX_ACTIONS, dtype=np.float32)
        bias, stats, _ = trainer._combat_action_quality_bias({}, mask, actions)

        self.assertEqual(float(bias[0]), 0.0)
        self.assertEqual(stats["combat_quality_insatiable_sandpit_active"], 0.0)
        self.assertEqual(stats["combat_quality_insatiable_sandpit_lt3"], 0.0)
        self.assertEqual(stats["combat_quality_insatiable_frantic_escape_bonus_applied"], 0.0)

    def test_sandpit_one_confirmed_lethal_does_not_count_as_missed(self):
        raw = _raw(sandpit=1, enemy_hp=6)
        trainer = _trainer(raw)
        actions = [_frantic(), _strike(damage=6, target_hp=6)]
        stats = {
            "combat_quality_insatiable_frantic_escape_available": 1.0,
            "combat_quality_insatiable_sandpit_lt3": 1.0,
            "combat_quality_insatiable_sandpit_1": 1.0,
        }
        selected_stats = trainer._selected_combat_quality_stats({}, 1, actions, stats)
        self.assertEqual(selected_stats["combat_quality_insatiable_frantic_escape_missed_at_1"], 0.0)

    def test_sandpit_one_nonlethal_non_escape_counts_as_missed(self):
        raw = _raw(sandpit=1, enemy_hp=100)
        trainer = _trainer(raw)
        actions = [_frantic(), _strike(damage=6, target_hp=100)]
        stats = {
            "combat_quality_insatiable_frantic_escape_available": 1.0,
            "combat_quality_insatiable_sandpit_lt3": 1.0,
            "combat_quality_insatiable_sandpit_1": 1.0,
        }
        selected_stats = trainer._selected_combat_quality_stats({}, 1, actions, stats)
        self.assertEqual(selected_stats["combat_quality_insatiable_frantic_escape_missed_at_1"], 1.0)
        self.assertEqual(selected_stats["combat_quality_insatiable_non_escape_at_1_selected"], 1.0)


if __name__ == "__main__":
    unittest.main()
