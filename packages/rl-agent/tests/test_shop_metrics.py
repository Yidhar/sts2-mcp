from __future__ import annotations

import sys
import unittest
from pathlib import Path


RL_AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(RL_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(RL_AGENT_ROOT))

from muzero.diagnostics.shop_metrics import (
    ShopEpisodeTracker,
    build_shop_choice_payload,
    is_shop_action,
    shop_action_kind,
)


class ShopMetricsTest(unittest.TestCase):
    def test_shop_buy_card_removal_classifies_as_remove(self) -> None:
        action = {
            "kind": "shop",
            "action_id": "shop:buy:3",
            "shop_action": "buy",
            "item": {
                "item_kind": "card_removal",
                "title": "Remove a card",
                "cost": 75,
                "is_affordable": True,
            },
        }

        self.assertEqual(shop_action_kind(action), "remove")

    def test_flattened_compact_card_removal_classifies_as_remove(self) -> None:
        action = {
            "kind": "shop",
            "action_id": "shop:buy:3",
            "shop_action": "buy",
            "shop_item_kind": "card_removal",
            "shop_item_title": "Remove a card",
            "shop_item_cost": 75,
            "shop_item_affordable": True,
        }

        self.assertTrue(is_shop_action(action))
        self.assertEqual(shop_action_kind(action), "remove")

    def test_semantic_compact_card_removal_classifies_as_remove(self) -> None:
        action = {
            "action_id": "shop:buy:remove",
            "semantic": {
                "family": "shop",
                "shop_action": "buy",
                "shop_item_kind": "card_removal",
            },
        }

        self.assertTrue(is_shop_action(action))
        self.assertEqual(shop_action_kind(action), "remove")

    def test_map_action_to_future_shop_is_not_shop_payload(self) -> None:
        action = {
            "kind": "map",
            "action_id": "map:12:2",
            "room_type": "Shop",
            "route_summary": {"next_room_type": "shop", "shop_count": 1},
            # Defensive regression: even if a future/destination semantic tag
            # says shop, this is still route selection, not merchant UI.
            "semantic": {"family": "shop", "domain": "route"},
        }

        self.assertFalse(is_shop_action(action))
        self.assertEqual(shop_action_kind(action), "")
        self.assertIsNone(
            build_shop_choice_payload(
                decision_domain="route",
                phase="map",
                legal_actions=[action],
                chosen_action=action,
                chosen_signature=None,
                selected_index=0,
                progress={"floor": 6, "room_type": "Map"},
                gold=120,
                search_policy=[1.0],
            )
        )

    def test_empty_tracker_rates_are_zero_not_nan(self) -> None:
        meta = ShopEpisodeTracker().as_metadata()

        self.assertEqual(meta["shop_seen_count"], 0.0)
        for key, value in meta.items():
            self.assertEqual(value, value, key)  # no NaN
            if key.endswith("_rate"):
                self.assertEqual(value, 0.0, key)

    def test_tracker_counts_leave_with_affordable_remove(self) -> None:
        tracker = ShopEpisodeTracker()
        tracker.update(
            {
                "selected_action_kind": "leave",
                "remove_available": True,
                "remove_affordable": True,
                "leave_with_remove_affordable": True,
                "leave_with_gold_ge_100": True,
                "gold_before": 150,
                "selected_cost": 0,
                "affordable_item_count": 2,
            }
        )

        meta = tracker.as_metadata()
        self.assertEqual(meta["shop_seen_count"], 1.0)
        self.assertEqual(meta["shop_leave_count"], 1.0)
        self.assertEqual(meta["shop_remove_affordable_rate"], 1.0)
        self.assertEqual(meta["shop_leave_with_remove_affordable_rate"], 1.0)
        self.assertEqual(meta["shop_leave_with_gold_ge_100_rate"], 1.0)
        self.assertEqual(meta["shop_gold_before_mean"], 150.0)
        self.assertEqual(meta["shop_affordable_item_count_mean"], 2.0)

    def test_build_shop_choice_payload_exposes_selected_and_available_summary(self) -> None:
        legal_actions = [
            {
                "kind": "shop",
                "action_id": "shop:buy:0",
                "shop_action": "buy",
                "item": {
                    "item_kind": "card_removal",
                    "title": "Remove a card",
                    "cost": 75,
                    "is_affordable": True,
                },
            },
            {"kind": "shop", "action_id": "shop:leave", "shop_action": "leave"},
        ]

        payload = build_shop_choice_payload(
            decision_domain="build",
            phase="shop",
            legal_actions=legal_actions,
            chosen_action=legal_actions[1],
            chosen_signature=None,
            selected_index=1,
            progress={"floor": 7, "act_id": 1, "room_type": "Shop"},
            gold=120,
            search_policy=[0.25, 0.75],
            max_topk=4,
        )

        self.assertIsNotNone(payload)
        assert payload is not None
        self.assertEqual(payload["selected_action_kind"], "leave")
        self.assertTrue(payload["remove_available"])
        self.assertTrue(payload["remove_affordable"])
        self.assertTrue(payload["leave_with_remove_affordable"])
        self.assertTrue(payload["leave_with_gold_ge_100"])
        self.assertEqual(payload["affordable_item_count"], 1.0)
        self.assertEqual(payload["floor"], 7)
        self.assertEqual(payload["top_policy_shop_actions"][0]["action_kind"], "leave")

    def test_payload_marks_buy_that_blocks_affordable_remove_in_starter_heavy_deck(self) -> None:
        legal_actions = [
            {
                "kind": "shop",
                "action_id": "shop:buy:card",
                "shop_action": "buy",
                "item": {"item_kind": "card", "title": "Headbutt", "cost": 75, "is_affordable": True},
            },
            {
                "kind": "shop",
                "action_id": "shop:buy:remove",
                "shop_action": "buy",
                "item": {
                    "item_kind": "card_removal",
                    "title": "Remove a card",
                    "cost": 75,
                    "is_affordable": True,
                },
            },
        ]
        deck = [
            *[
                {"id": "CARD.STRIKE_IRONCLAD", "title": "打击", "type": "Attack", "cost": 1}
                for _ in range(5)
            ],
            *[
                {"id": "CARD.DEFEND_IRONCLAD", "title": "防御", "type": "Skill", "cost": 1}
                for _ in range(4)
            ],
            {"id": "CARD.BASH", "title": "痛击", "type": "Attack", "cost": 2},
            {"id": "CARD.POMMEL_STRIKE", "title": "剑柄打击", "type": "Attack", "cost": 1},
        ]

        payload = build_shop_choice_payload(
            decision_domain="build",
            phase="shop",
            legal_actions=legal_actions,
            chosen_action=legal_actions[0],
            chosen_signature=None,
            selected_index=0,
            progress={"floor": 6, "act_id": 1, "room_type": "Shop"},
            gold=139,
            raw_obs={"player": {"deck_cards": deck}},
            search_policy=[0.8, 0.2],
        )

        self.assertIsNotNone(payload)
        assert payload is not None
        self.assertTrue(payload["deck_context_present"])
        self.assertEqual(payload["starter_count_before"], 9.0)
        self.assertTrue(payload["starter_heavy_before"])
        self.assertTrue(payload["selected_buy_with_affordable_remove"])
        self.assertTrue(payload["selected_buy_blocks_affordable_remove"])
        self.assertTrue(payload["selected_buy_blocks_affordable_remove_starter_heavy"])
        self.assertEqual(payload["remove_affordable_cost_min"], 75.0)

        tracker = ShopEpisodeTracker()
        tracker.update(payload)
        meta = tracker.as_metadata()
        self.assertEqual(meta["shop_deck_context_present_rate"], 1.0)
        self.assertEqual(meta["shop_starter_count_mean"], 9.0)
        self.assertEqual(meta["shop_buy_blocks_affordable_remove_starter_heavy_rate"], 1.0)

    def test_payload_finds_deck_from_raw_obs_fallback_context(self) -> None:
        legal_actions = [
            {"kind": "shop", "action_id": "shop:open", "shop_action": "open"},
            {"kind": "shop", "action_id": "shop:leave", "shop_action": "leave"},
        ]
        deck = [
            *[
                {"id": "CARD.STRIKE_IRONCLAD", "title": "打击", "type": "Attack", "cost": 1}
                for _ in range(5)
            ],
            *[
                {"id": "CARD.DEFEND_IRONCLAD", "title": "防御", "type": "Skill", "cost": 1}
                for _ in range(4)
            ],
            {"id": "CARD.BASH", "title": "痛击", "type": "Attack", "cost": 2},
        ]

        payload = build_shop_choice_payload(
            decision_domain="build",
            phase="shop",
            legal_actions=legal_actions,
            chosen_action=legal_actions[0],
            chosen_signature=None,
            selected_index=0,
            progress={"floor": 5, "act_id": 1, "room_type": "Shop"},
            gold=120,
            raw_obs={
                # Compact transition payload can be missing deck_cards.
                "transition_state": {"player": {"gold": 120}},
                # The env raw obs must still be accepted as the deck source.
                "raw_obs": {"player": {"deck_cards": deck, "gold": 120}},
            },
            search_policy=[0.8, 0.2],
        )

        self.assertIsNotNone(payload)
        assert payload is not None
        self.assertTrue(payload["deck_context_present"])
        self.assertEqual(payload["deck_size_before"], 10.0)
        self.assertEqual(payload["starter_count_before"], 9.0)
        self.assertTrue(payload["starter_heavy_before"])

    def test_self_play_persists_shop_metrics_to_tb_and_episode_metadata(self) -> None:
        src = (RL_AGENT_ROOT / "muzero" / "training" / "self_play.py").read_text(encoding="utf-8-sig")

        self.assertIn("f\"build/shop_{tag_suffix}\"", src)
        self.assertIn("**shop_meta", src)
        self.assertIn("\"shop_metrics\": shop_meta", src)
        self.assertIn("_last_obs_raw", src)
        self.assertIn("shop_raw_context", src)
        self.assertIn("raw_obs=shop_raw_context", src)


if __name__ == "__main__":
    unittest.main()
