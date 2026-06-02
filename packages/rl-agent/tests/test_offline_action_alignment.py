"""Tests for conservative offline action/index alignment audits."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pytest


RL_AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(RL_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(RL_AGENT_ROOT))

pytest.importorskip("torch")

from muzero.training.offline_action_alignment import (  # noqa: E402
    audit_offline_row_alignment,
    audit_offline_task_alignment,
    build_offline_action_specs,
)


def _state(**overrides):
    base = {
        "sample_id": "sample-1",
        "split": "train",
        "build_id": "build-1",
        "run_id": "run-1",
        "deck_ids": ["Strike", "Defend"],
        "relic_ids": ["BurningBlood"],
        "scalars": [0.0] * 21,
    }
    base.update(overrides)
    return base


class OfflineActionAlignmentTests(unittest.TestCase):
    def test_candidate_task_valid_label_with_visible_skip_is_ready(self):
        row = _state(
            task="regular_card_reward",
            candidate_ids=["CARD.POMMEL_STRIKE", "CARD.BATTLE_TRANCE", "<skip>"],
            label_index=1,
            label_id="CARD.BATTLE_TRANCE",
        )

        report = audit_offline_row_alignment(row, "regular_card_reward")

        self.assertTrue(report.ok)
        self.assertEqual(report.candidate_count, 3)
        self.assertTrue(report.label_valid)
        self.assertTrue(report.skip_visible)
        self.assertFalse(report.skip_selected)
        self.assertEqual(report.selected_label, "CARD.BATTLE_TRANCE")

    def test_candidate_task_missing_optional_skip_blocks_direct_ce(self):
        rows = [
            _state(
                sample_id="s1",
                task="regular_card_reward",
                candidate_ids=["CARD.POMMEL_STRIKE", "CARD.BATTLE_TRANCE"],
                label_index=0,
                label_id="CARD.POMMEL_STRIKE",
            )
        ]

        row_report = audit_offline_row_alignment(rows[0], "regular_card_reward")
        task_report = audit_offline_task_alignment("regular_card_reward", rows)

        self.assertIn("optional_skip_candidate_missing", row_report.issues)
        self.assertFalse(task_report.alignment_ready)
        self.assertIn("optional_skip_candidate_missing", task_report.ready_blockers)
        self.assertAlmostEqual(task_report.optional_skip_missing_rate or 0.0, 1.0)

    def test_invalid_label_index_is_blocker(self):
        row = _state(
            task="smith_target",
            candidate_ids=["CARD.STRIKE", "CARD.DEFEND"],
            label_index=4,
            label_id="CARD.STRIKE",
        )

        report = audit_offline_row_alignment(row, "smith_target")

        self.assertIn("invalid_label_index", report.issues)
        self.assertFalse(report.label_valid)

    def test_route_without_candidates_is_not_policy_alignment_ready(self):
        rows = [
            _state(
                sample_id="route-1",
                task="route_room_type",
                label="Monster",
                next_room_type="Monster",
            )
        ]

        row_report = audit_offline_row_alignment(rows[0], "route_room_type")
        task_report = audit_offline_task_alignment("route_room_type", rows)

        self.assertIn("missing_candidates", row_report.issues)
        self.assertIn("route_missing_candidate_supervision", row_report.issues)
        self.assertIn("route_label_not_candidate_aligned", row_report.issues)
        self.assertFalse(task_report.alignment_ready)
        self.assertIn("route_missing_candidate_supervision", task_report.ready_blockers)

    def test_route_with_candidates_and_valid_label_is_ready(self):
        rows = [
            _state(
                sample_id="route-2",
                task="route_point_type",
                route_candidates=[
                    {"action_id": "map:0", "point_type_norm": "Monster"},
                    {"action_id": "map:1", "point_type_norm": "RestSite"},
                ],
                label_index=1,
                label="RestSite",
            )
        ]

        row_report = audit_offline_row_alignment(rows[0], "route_point_type")
        task_report = audit_offline_task_alignment("route_point_type", rows)

        self.assertTrue(row_report.ok)
        self.assertTrue(task_report.alignment_ready)
        self.assertEqual(row_report.selected_action_id, "map:1")
        self.assertAlmostEqual(task_report.route_candidate_present_rate or 0.0, 1.0)
        self.assertAlmostEqual(task_report.route_candidate_label_valid_rate or 0.0, 1.0)

    def test_classification_task_uses_stable_candidate_set(self):
        row = _state(
            task="shop_remove_binary",
            label="skip_remove",
        )

        specs, label_index = build_offline_action_specs(row, "shop_remove_binary")
        report = audit_offline_row_alignment(row, "shop_remove_binary")

        self.assertEqual([spec.label for spec in specs], ["remove_card", "skip_remove"])
        self.assertEqual(label_index, 1)
        self.assertTrue(report.ok)
        self.assertEqual(report.selected_label, "skip_remove")
        self.assertTrue(report.skip_selected)

    def test_classification_task_missing_label_returns_specs_and_none(self):
        row = _state(
            task="shop_remove_binary",
            label="unknown",
        )

        specs, label_index = build_offline_action_specs(row, "shop_remove_binary")
        report = audit_offline_row_alignment(row, "shop_remove_binary")

        self.assertEqual([spec.label for spec in specs], ["remove_card", "skip_remove"])
        self.assertIsNone(label_index)
        self.assertIn("missing_label_index", report.issues)
        self.assertFalse(report.label_valid)

    def test_candidate_count_overflow_is_blocker(self):
        row = _state(
            task="smith_target",
            candidate_ids=["A", "B", "C"],
            label_index=0,
        )

        report = audit_offline_row_alignment(row, "smith_target", max_actions=2)

        self.assertIn("candidate_count_exceeds_max_actions", report.issues)


if __name__ == "__main__":
    unittest.main()
