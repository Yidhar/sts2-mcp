"""Regression tests for combat card-selection compact/semantic fields.

Purity/净化 style cards open a combat card-selection surface where clicking an
already-selected hand card toggles it back off.  The replay/semantic path must
therefore preserve selection state instead of collapsing every option into an
identical "select card" action.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


RL_AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(RL_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(RL_AGENT_ROOT))

from sts2_env.action_compact import compact_action_signature
from sts2_env.semantic_action import semantic_action_signature


class CardSelectionCompactSemanticsTest(unittest.TestCase):
    def test_compact_preserves_false_and_zero_selection_state(self) -> None:
        action = {
            "kind": "card_selection",
            "action_id": "card_selection:select:0",
            "selection_action": "select",
            "selection_semantics": "exhaust",
            "selection_prompt": "Choose up to 3 cards to Exhaust.",
            "is_selected": False,
            "selected_count": 0,
            "min_select": 0,
            "max_select": 3,
            "remaining_select": 0,
            "confirm_ready": False,
            "can_skip": False,
            "requires_manual_confirmation": True,
            "cancelable": False,
            "selection_ready": False,
            "opened_age_ms": 0,
            "card": {"id": "IRONCLAD.STRIKE", "title": "Strike", "type": "Attack"},
        }

        compact = compact_action_signature(action)

        self.assertEqual(compact["selection"], "select")
        self.assertEqual(compact["selection_action"], "select")
        self.assertEqual(compact["selection_semantics"], "exhaust")
        self.assertEqual(compact["selection_prompt"], "Choose up to 3 cards to Exhaust.")
        self.assertIs(compact["is_selected"], False)
        self.assertEqual(compact["selected_count"], 0)
        self.assertEqual(compact["min_select"], 0)
        self.assertEqual(compact["max_select"], 3)
        self.assertEqual(compact["remaining_select"], 0)
        self.assertIs(compact["confirm_ready"], False)
        self.assertIs(compact["can_skip"], False)
        self.assertIs(compact["requires_manual_confirmation"], True)
        self.assertIs(compact["cancelable"], False)
        self.assertIs(compact["selection_ready"], False)
        self.assertEqual(compact["opened_age_ms"], 0)

        # The nested compact semantic view intentionally drops False values, but
        # it must still classify this action as a card-selection setup action.
        self.assertEqual(compact["semantic"]["family"], "card_selection")
        self.assertEqual(compact["semantic"]["domain"], "selection")
        self.assertIn("setup", compact["semantic"]["roles"])

    def test_semantic_family_falls_back_to_card_selection_action_id(self) -> None:
        signature = semantic_action_signature(
            {
                # Replay/diagnostic compaction can retain only the action_id.
                "action_id": "card_selection:confirm",
                "selection_semantics": "exhaust",
            }
        )

        self.assertEqual(signature["family"], "card_selection")
        self.assertEqual(signature["domain"], "selection")
        self.assertEqual(signature["target_scope"], "choice")
        self.assertIn("setup", signature["roles"])

    def test_semantic_uses_selection_action_in_key_when_selection_missing(self) -> None:
        signature = semantic_action_signature(
            {
                "kind": "card_selection",
                "action_id": "card_selection:confirm",
                "selection_action": "confirm",
                "selection_semantics": "exhaust",
            }
        )

        self.assertEqual(signature["family"], "card_selection")
        self.assertIn("confirm", signature["semantic_key"])
        self.assertIn("exhaust", signature["semantic_key"])

    def test_compact_action_signature_preserves_shop_card_removal_item(self) -> None:
        action = {
            "kind": "shop",
            "action_id": "shop:buy:3",
            "shop_action": "buy",
            "item": {
                "item_kind": "card_removal",
                "title": "Remove a card",
                "cost": 75,
                "is_affordable": True,
                "used": False,
            },
        }

        compact = compact_action_signature(action)

        self.assertEqual(compact["shop_action"], "buy")
        self.assertEqual(compact["shop_item_kind"], "card_removal")
        self.assertEqual(compact["shop_item_cost"], 75)
        self.assertIs(compact["shop_item_affordable"], True)
        self.assertEqual(compact["semantic"]["family"], "shop")
        self.assertEqual(compact["semantic"]["shop_item_kind"], "card_removal")
        self.assertIs(compact["semantic"]["shop_is_remove"], True)
        self.assertIn("card_removal", compact["semantic"]["semantic_key"])

    def test_compact_action_signature_preserves_reward_card_identity(self) -> None:
        action = {
            "surface": "card_reward",
            "action_id": "card_reward:0",
            "choice_index": 0,
            "reward": {
                "type": "card",
                "card": {
                    "id": "CARD.TRUE_GRIT",
                    "title": "坚毅",
                    "type": "Skill",
                    "cost": 1,
                },
            },
        }

        compact = compact_action_signature(action)

        self.assertEqual(compact["surface"], "card_reward")
        self.assertEqual(compact["reward_type"], "card")
        self.assertEqual(compact["action_id"], "card_reward:0")
        self.assertEqual(compact["card_id"], "CARD.TRUE_GRIT")
        self.assertEqual(compact["card_title"], "坚毅")
        self.assertEqual(compact["title"], "坚毅")
        self.assertEqual(compact["card_type"], "Skill")
        self.assertEqual(compact["card_cost"], 1)


if __name__ == "__main__":
    unittest.main()
