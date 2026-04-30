"""Integration test for the consolidated boss_mechanics block.

Verifies that :func:`build_boss_mechanics_block` and
:func:`classify_action_boss_mechanism` return the shape that the
``CombatSandboxEnv.step`` info dict surfaces, and that the three Phase 4
helpers light up correctly when their respective enemies / powers are
present in the raw observation.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any


RL_AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(RL_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(RL_AGENT_ROOT))

from sts2_env.boss_mechanics import (
    build_boss_mechanics_block,
    classify_action_boss_mechanism,
)


def _kaiser_obs() -> dict[str, Any]:
    return {
        "combat": {
            "enemies": [
                {"id": "left_part", "powers": [{"id": "BackAttackLeftPower"}], "intent": {"total_damage": 12}},
                {"id": "right_part", "powers": [{"id": "BackAttackRightPower"}], "intent": {"total_damage": 12}},
            ],
        },
        "player": {"hp": 70, "max_hp": 80, "block": 0, "powers": [{"id": "SurroundedPower", "facing": "left"}]},
    }


def _ceremonial_obs() -> dict[str, Any]:
    return {
        "combat": {
            "enemies": [
                {
                    "id": "ceremonial_beast",
                    "hp": 60,
                    "max_hp": 100,
                    "powers": [{"id": "CeremonialOneCardLockPower", "amount": 2}],
                    "intent": {"total_damage": 18},
                }
            ],
        },
        "player": {"hp": 60, "max_hp": 80, "block": 0, "powers": []},
    }


def _insatiable_obs() -> dict[str, Any]:
    return {
        "combat": {
            "enemies": [
                {
                    "id": "the_insatiable_boss",
                    "hp": 120,
                    "max_hp": 200,
                    "powers": [{"id": "InsatiableGrowthPower", "amount": 3}],
                    "intent": {"total_damage": 20},
                }
            ],
        },
        "player": {"hp": 60, "max_hp": 80, "block": 4, "powers": []},
    }


def _plain_obs() -> dict[str, Any]:
    return {
        "combat": {"enemies": [{"id": "boring", "powers": [], "intent": {"total_damage": 5}}]},
        "player": {"hp": 70, "block": 0, "powers": []},
    }


class BossMechanicsBlockShapeTests(unittest.TestCase):
    def test_block_always_has_three_subkeys(self):
        block = build_boss_mechanics_block(_plain_obs())
        self.assertIn("kaiser", block)
        self.assertIn("ceremonial", block)
        self.assertIn("insatiable", block)

    def test_inactive_state_when_no_boss_mechanics(self):
        block = build_boss_mechanics_block(_plain_obs())
        self.assertFalse(block["kaiser"]["active"])
        self.assertFalse(block["ceremonial"]["active"])
        self.assertFalse(block["insatiable"]["active"])


class BossMechanicsActivationTests(unittest.TestCase):
    def test_kaiser_block_active(self):
        block = build_boss_mechanics_block(_kaiser_obs())
        self.assertTrue(block["kaiser"]["active"])
        self.assertEqual(block["kaiser"]["player_facing"], "left")
        self.assertGreater(block["kaiser"]["back_attack_risk"], 0.0)

    def test_ceremonial_block_active(self):
        block = build_boss_mechanics_block(_ceremonial_obs())
        self.assertTrue(block["ceremonial"]["active"])
        self.assertTrue(block["ceremonial"]["one_card_lock_active"])

    def test_insatiable_block_active(self):
        block = build_boss_mechanics_block(_insatiable_obs())
        self.assertTrue(block["insatiable"]["active"])
        self.assertGreater(block["insatiable"]["pressure"], 0.0)


class ActionMechanismIntegrationTests(unittest.TestCase):
    def test_targeted_strike_flips_kaiser_facing(self):
        action = {
            "kind": "play_card", "action_id": "play:strike",
            "target": {"combat_id": "right_part", "scope": "single_enemy"},
            "card": {"id": "STRIKE", "title": "Strike", "type": "Attack", "cost": 1,
                     "card_effect_profile": {"derived_view": {"combat_effect": {"damage": 8}, "lifecycle": {}, "hand_mutation": {}, "pile_mutation": {}, "cost": {}, "mechanism_effect": {}}}},
        }
        mechanism = classify_action_boss_mechanism(_kaiser_obs(), action)
        self.assertTrue(mechanism["kaiser"]["kaiser_changes_facing"])
        # Ceremonial / Insatiable defaults stay neutral when those bosses are absent.
        self.assertEqual(mechanism["ceremonial"]["ceremonial_single_action_impact_score"], 0.0)
        self.assertFalse(mechanism["insatiable"]["insatiable_strategic_skip"])

    def test_diag_propagates_to_insatiable_offenders(self):
        action = {
            "kind": "play_card", "action_id": "play:strike",
            "card": {"id": "STRIKE", "title": "Strike", "type": "Attack", "cost": 1,
                     "card_effect_profile": {"derived_view": {"combat_effect": {"damage": 6}, "lifecycle": {}, "hand_mutation": {}, "pile_mutation": {}, "cost": {}, "mechanism_effect": {}}}},
        }
        diag = {"strategic_skip_selected": True}
        mechanism = classify_action_boss_mechanism(_insatiable_obs(), action, action_diagnostics=diag)
        self.assertTrue(mechanism["insatiable"]["insatiable_strategic_skip"])


if __name__ == "__main__":
    unittest.main()
