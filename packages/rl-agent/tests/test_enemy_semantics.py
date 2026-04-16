from __future__ import annotations

import unittest
from pathlib import Path
import sys


RL_AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(RL_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(RL_AGENT_ROOT))

from content_registry import (  # noqa: E402
    build_enemy_semantic_text,
    build_live_enemy_semantic_text,
    build_live_enemy_trait_text,
    enemy_trait_vector,
    get_enemy_metadata,
)


class EnemySemanticsContractTest(unittest.TestCase):
    def test_generated_enemy_metadata_exposes_structured_channels(self) -> None:
        metadata = get_enemy_metadata("MONSTER.SPINY_TOAD")
        self.assertIsInstance(metadata, dict)
        self.assertEqual(metadata.get("title"), "Spiny Toad")
        self.assertTrue(metadata.get("static_traits"))
        self.assertTrue(metadata.get("reactive_triggers"))
        self.assertIsInstance(metadata.get("danger_profile"), dict)

        phase_metadata = get_enemy_metadata("MONSTER.DOORMAKER")
        self.assertIsInstance(phase_metadata, dict)
        self.assertTrue(phase_metadata.get("phase_rules"))

    def test_live_enemy_semantic_text_dedupes_trait_rendering(self) -> None:
        enemy = {
            "name": "Spiny Toad",
            "model_id": "MONSTER.SPINY_TOAD",
            "current_hp": 44,
            "max_hp": 50,
            "block": 0,
            "intent": {
                "description": "Attack 9",
                "total_damage": 9,
                "damage_per_hit": 9,
                "repeats": 1,
            },
            "powers": [
                {
                    "title": "Thorns",
                    "description": "Retaliates on contact.",
                    "amount": 3,
                    "display_amount": 3,
                }
            ],
            "reactive_triggers": [
                {
                    "trait": "retaliate",
                    "description": "On hit, retaliates / punishes contact damage.",
                    "trigger_type": "on_hit",
                    "condition": "contact",
                    "effect_type": "retaliate",
                }
            ],
            "phase_rules": [
                {
                    "trait": "threshold_stun",
                    "description": "At hp <= 25 becomes stunned.",
                    "trigger_type": "on_hp_threshold",
                    "condition": "hp_le_25",
                    "effect_type": "stun_self",
                    "threshold": 25,
                }
            ],
            "danger_profile": {
                "retaliation": 5,
                "attrition": 4,
                "target_priority": 4,
            },
        }

        trait_text = build_live_enemy_trait_text(enemy)
        semantic_text = build_live_enemy_semantic_text(enemy)

        self.assertIn("Punishes contact hits", trait_text)
        self.assertIn("At hp <= 25 becomes stunned", trait_text)
        self.assertEqual(trait_text.count("Punishes contact hits"), 1)
        self.assertIn("danger", semantic_text)
        self.assertIn("retal=5", semantic_text)
        self.assertIn("prio=4", semantic_text)

    def test_enemy_trait_vector_reflects_structured_danger_and_thresholds(self) -> None:
        enemy = {
            "name": "Spiny Toad",
            "model_id": "MONSTER.SPINY_TOAD",
            "powers": [
                {
                    "title": "Thorns",
                    "description": "Retaliates on contact.",
                    "amount": 3,
                }
            ],
            "phase_rules": [
                {
                    "trait": "threshold_stun",
                    "description": "At hp <= 25 becomes stunned.",
                    "trigger_type": "on_hp_threshold",
                    "condition": "hp_le_25",
                    "effect_type": "stun_self",
                    "threshold": 25,
                }
            ],
            "danger_profile": {
                "attrition": 4,
                "retaliation": 5,
                "target_priority": 4,
            },
        }

        vector = enemy_trait_vector(enemy)
        self.assertEqual(vector["retaliation"], 1.0)
        self.assertEqual(vector["threshold"], 1.0)
        self.assertGreaterEqual(vector["attrition"], 0.8)
        self.assertGreaterEqual(vector["target_priority"], 0.8)

    def test_static_enemy_semantic_text_uses_generated_registry(self) -> None:
        text = build_enemy_semantic_text("MONSTER.DOORMAKER")
        self.assertIn("Doormaker", text)
        self.assertIn("danger", text)
        self.assertTrue("door" in text.lower() or "exposed" in text.lower())


if __name__ == "__main__":
    unittest.main()
