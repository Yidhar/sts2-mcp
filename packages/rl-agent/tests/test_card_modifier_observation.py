from __future__ import annotations

import sys
import unittest
from pathlib import Path

RL_AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(RL_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(RL_AGENT_ROOT))

from content_registry import build_live_card_semantic_text
from sts2_env.observation_v3 import TOKEN_TYPE_TO_ID, WorldTokenObservationEncoder, _CARD_KEYWORD_BUCKETS


class CardModifierObservationTest(unittest.TestCase):
    def test_runtime_affliction_emits_bound_keyword_slot(self) -> None:
        enc = WorldTokenObservationEncoder(use_text=False)
        obs = {
            "phase": "combat",
            "player": {"hp": 50, "max_hp": 80, "relics": [], "potions": []},
            "combat": {
                "hand": [
                    {
                        "id": "TEST.BOUND_CARD",
                        "title": "Bound Strike",
                        "type": "Skill",
                        "cost": 1,
                        "afflictions": [
                            {
                                "id": "BOUND",
                                "title": "Bound",
                                "description": "This card is bound by chains.",
                                "amount": 1,
                            }
                        ],
                    }
                ],
                "enemies": [],
            },
            "legal_actions": [],
        }

        encoded = enc.encode(obs)
        type_ids = encoded["world_token_type_ids"]
        entity_ids = encoded["world_token_entity_ids"]
        numerics = encoded["world_tokens"][:, :96]
        keyword_rows = [
            i for i, token_type in enumerate(type_ids)
            if int(token_type) == TOKEN_TYPE_TO_ID["CARD_KEYWORD_SLOT"]
        ]

        self.assertTrue(keyword_rows)
        self.assertIn(_CARD_KEYWORD_BUCKETS["bound"], [int(entity_ids[i]) for i in keyword_rows])
        bound_rows = [i for i in keyword_rows if int(entity_ids[i]) == _CARD_KEYWORD_BUCKETS["bound"]]
        self.assertTrue(any(float(numerics[i][8]) == 1.0 for i in bound_rows))

    def test_live_card_semantic_text_includes_runtime_modifiers(self) -> None:
        text = build_live_card_semantic_text(
            {
                "id": "TEST.CARD",
                "title": "Runtime Card",
                "type": "Skill",
                "cost": 1,
                "afflictions": [{"id": "BOUND", "title": "Bound", "amount": 2}],
                "enchantments": [{"id": "RETAINED", "title": "Retained"}],
            }
        )
        self.assertIn("mods", text)
        self.assertIn("aff=Bound:2", text)
        self.assertIn("ench=Retained", text)


if __name__ == "__main__":
    unittest.main()
