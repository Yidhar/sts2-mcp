from __future__ import annotations

import sys
import unittest
from pathlib import Path


RL_AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(RL_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(RL_AGENT_ROOT))

from muzero.diagnostics.rest_site_metrics import (
    RestSiteEpisodeTracker,
    build_rest_site_choice_payload,
    is_rest_heal_action,
    is_rest_site_action,
    is_rest_smith_action,
    rest_action_kind,
)
from sts2_env.action_compact import compact_action_signature


class RestSiteMetricsTest(unittest.TestCase):
    def test_proceed_is_not_rest_site_choice(self) -> None:
        action = {"kind": "rest_site", "action_id": "rest_site:proceed", "title": "Continue"}

        self.assertFalse(is_rest_site_action(action))
        self.assertFalse(is_rest_heal_action(action))
        self.assertFalse(is_rest_smith_action(action))
        self.assertEqual(rest_action_kind(action), "")

    def test_smith_is_not_misclassified_as_heal(self) -> None:
        action = {
            "kind": "rest_site",
            "action_id": "rest_site:smith",
            "option": {"option_type": "SmithRestSiteOption", "title": "锻造", "description": "升级一张牌。"},
        }

        self.assertTrue(is_rest_site_action(action))
        self.assertFalse(is_rest_heal_action(action))
        self.assertTrue(is_rest_smith_action(action))
        self.assertEqual(rest_action_kind(action), "smith")

    def test_compact_smith_identity_overrides_stale_heal_flags(self) -> None:
        action = {
            "index": 1,
            "kind": "rest_site",
            "action_id": "rest_site:1",
            "action_kind": "heal",
            "title": "锻造",
            "is_rest_site": True,
            "is_heal": True,
            "is_smith": True,
        }

        self.assertTrue(is_rest_site_action(action))
        self.assertTrue(is_rest_smith_action(action))
        self.assertFalse(is_rest_heal_action(action))
        self.assertEqual(rest_action_kind(action), "smith")

    def test_live_bridge_heal_schema_is_detected(self) -> None:
        action = {
            "kind": "rest_site",
            "action_id": "rest_site:0",
            "option": {
                "option_id": "HEAL",
                "option_type": "HealRestSiteOption",
                "title": "休息",
                "description": "回复18点生命值。",
            },
        }

        self.assertTrue(is_rest_site_action(action))
        self.assertTrue(is_rest_heal_action(action))
        self.assertFalse(is_rest_smith_action(action))
        self.assertEqual(rest_action_kind(action), "heal")

    def test_nested_payload_schema_is_detected(self) -> None:
        action = {
            "payload": {
                "kind": "rest_site",
                "action_id": "rest_site:1",
                "option": {
                    "option_type": "SmithRestSiteOption",
                    "title": "升级",
                    "description": "Upgrade a card.",
                },
            }
        }

        self.assertTrue(is_rest_site_action(action))
        self.assertFalse(is_rest_heal_action(action))
        self.assertTrue(is_rest_smith_action(action))
        self.assertEqual(rest_action_kind(action), "smith")

    def test_compact_action_preserves_option_identity_for_rest_diagnostics(self) -> None:
        raw = {
            "kind": "rest_site",
            "action_id": "rest_site:0",
            "option": {
                "option_id": "HEAL",
                "option_type": "HealRestSiteOption",
                "title": "休息",
                "description": "回复18点生命值。",
            },
        }

        compact = compact_action_signature(raw)

        self.assertEqual(compact["option_id"], "HEAL")
        self.assertEqual(compact["option_type"], "HealRestSiteOption")
        self.assertTrue(is_rest_heal_action(compact))
        self.assertEqual(rest_action_kind(compact), "heal")

    def test_build_payload_marks_low_hp_filter_and_raw_minus_filtered(self) -> None:
        raw_actions = [
            {
                "kind": "rest_site",
                "action_id": "rest_site:0",
                "option": {"option_type": "HealRestSiteOption", "title": "休息"},
            },
            {
                "kind": "rest_site",
                "action_id": "rest_site:1",
                "option": {"option_type": "SmithRestSiteOption", "title": "锻造"},
            },
        ]
        legal_actions = [raw_actions[0]]

        payload = build_rest_site_choice_payload(
            decision_domain="build",
            phase="rest_site",
            legal_actions=legal_actions,
            raw_legal_actions=raw_actions,
            chosen_action=legal_actions[0],
            chosen_signature=None,
            selected_index=0,
            progress={"floor": 8, "act_id": 1, "room_type": "RestSite"},
            raw_obs={"player": {"hp": 24, "max_hp": 80}},
            search_policy=[1.0],
        )

        self.assertIsNotNone(payload)
        assert payload is not None
        self.assertEqual(payload["selected_action_kind"], "heal")
        self.assertTrue(payload["is_low_hp"])
        self.assertTrue(payload["low_hp_heal_filter_applied"])
        self.assertEqual(payload["rest_action_count"], 1.0)
        self.assertEqual(payload["raw_rest_action_count"], 2.0)
        self.assertEqual(payload["raw_minus_filtered_rest_count"], 1.0)
        self.assertTrue(payload["raw_smith_available"])
        self.assertFalse(payload["smith_available"])

    def test_payload_marks_high_hp_smith_available_but_not_selected(self) -> None:
        actions = [
            {
                "kind": "rest_site",
                "action_id": "rest_site:0",
                "option": {"option_type": "HealRestSiteOption", "title": "休息"},
            },
            {
                "kind": "rest_site",
                "action_id": "rest_site:1",
                "option": {"option_type": "SmithRestSiteOption", "title": "锻造"},
            },
        ]

        payload = build_rest_site_choice_payload(
            decision_domain="build",
            phase="rest_site",
            legal_actions=actions,
            raw_legal_actions=actions,
            chosen_action=actions[0],
            chosen_signature=None,
            selected_index=0,
            progress={"floor": 12, "act_id": 1, "room_type": "RestSite"},
            raw_obs={"player": {"hp": 70, "max_hp": 80}},
            search_policy=[0.7, 0.3],
        )

        self.assertIsNotNone(payload)
        assert payload is not None
        self.assertFalse(payload["is_low_hp"])
        self.assertTrue(payload["smith_available"])
        self.assertTrue(payload["smith_available_high_hp_not_selected"])

        tracker = RestSiteEpisodeTracker()
        tracker.update(payload)
        meta = tracker.as_metadata()
        self.assertEqual(meta["rest_site_seen_count"], 1.0)
        self.assertEqual(meta["rest_site_heal_rate"], 1.0)
        self.assertEqual(meta["rest_site_smith_available_rate"], 1.0)
        self.assertEqual(meta["rest_site_smith_available_high_hp_not_selected_rate"], 1.0)

    def test_empty_tracker_rates_are_zero_not_nan(self) -> None:
        meta = RestSiteEpisodeTracker().as_metadata()

        self.assertEqual(meta["rest_site_seen_count"], 0.0)
        for key, value in meta.items():
            self.assertEqual(value, value, key)
            if key.endswith("_rate"):
                self.assertEqual(value, 0.0, key)

    def test_self_play_persists_rest_site_metrics_to_tb_and_episode_metadata(self) -> None:
        src = (RL_AGENT_ROOT / "muzero" / "training" / "self_play.py").read_text(encoding="utf-8-sig")

        self.assertIn("f\"build/rest_site_{tag_suffix}\"", src)
        self.assertIn("**rest_site_meta", src)
        self.assertIn("\"rest_site_metrics\": rest_site_meta", src)
        self.assertIn("build_rest_site_choice_payload", src)
        self.assertIn("raw_legal_actions_compact", src)
        self.assertIn("rest_site_choices.jsonl", (RL_AGENT_ROOT / "muzero" / "diagnostics" / "rest_site_metrics.py").read_text(encoding="utf-8-sig"))


if __name__ == "__main__":
    unittest.main()
