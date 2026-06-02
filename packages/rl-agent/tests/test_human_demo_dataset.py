"""Tests for the human demo imitation dataset (TASK-F3)."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any


RL_AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(RL_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(RL_AGENT_ROOT))

from muzero.demo_dataset import (
    DemoValidationError,
    build_demo_training_batch,
    iter_demo_jsonl,
    load_demo_dataset,
)


def _row(*, episode_id: str = "ep1", encounter_id: str = "kaiser_crab_boss",
         legal: list[dict[str, Any]] | None = None, selected: str = "play:strike",
         reasons: list[str] | None = None, **overrides) -> dict[str, Any]:
    base = {
        "version": 1,
        "source": "human",
        "timestamp": "2026-04-29T21:00:00+08:00",
        "episode_id": episode_id,
        "encounter_id": encounter_id,
        "tier": "boss",
        "turn": 3,
        "step_in_turn": 2,
        "obs": {"player": {"hp": 70}},
        "legal_actions": legal or [
            {"action_id": "play:strike", "family": "play_card", "title": "Strike"},
            {"action_id": "play:defend", "family": "play_card", "title": "Defend"},
            {"action_id": "end_turn", "family": "end_turn", "title": "End Turn"},
        ],
        "selected_action_id": selected,
        "reason_tags": reasons or ["block_incoming"],
        "comment": "test row",
        "outcome": {"combat_win": True, "hp_loss": 6, "turns": 5},
    }
    base.update(overrides)
    return base


def _write_jsonl(rows: list[dict[str, Any]]) -> Path:
    handle = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8")
    for row in rows:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    handle.close()
    return Path(handle.name)


class DemoLoadTests(unittest.TestCase):
    def test_load_basic_dataset(self):
        path = _write_jsonl([_row(), _row(episode_id="ep2", selected="play:defend")])
        try:
            samples = load_demo_dataset(path)
        finally:
            path.unlink()
        self.assertEqual(len(samples), 2)
        self.assertEqual(samples[0].selected_action_index, 0)
        self.assertEqual(samples[1].selected_action_index, 1)
        self.assertEqual(samples[0].encounter_id, "kaiser_crab_boss")
        self.assertEqual(samples[0].reason_tags, ["block_incoming"])

    def test_blank_lines_skipped(self):
        path = Path(tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8").name)
        path.write_text(
            json.dumps(_row()) + "\n\n   \n" + json.dumps(_row(episode_id="ep2", selected="play:defend")) + "\n",
            encoding="utf-8",
        )
        try:
            samples = load_demo_dataset(path)
        finally:
            path.unlink()
        self.assertEqual(len(samples), 2)


class DemoActionMatchTests(unittest.TestCase):
    def test_selected_action_index_matches_legal(self):
        path = _write_jsonl([_row(selected="end_turn")])
        try:
            samples = load_demo_dataset(path)
        finally:
            path.unlink()
        self.assertEqual(samples[0].selected_action_index, 2)
        self.assertEqual(samples[0].selected_action_id, "end_turn")

    def test_unmatched_selected_action_skipped_in_lenient_mode(self):
        path = _write_jsonl([_row(selected="play:not_in_legal"), _row(episode_id="ep2", selected="play:defend")])
        try:
            samples = load_demo_dataset(path)
        finally:
            path.unlink()
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].episode_id, "ep2")

    def test_unmatched_selected_action_raises_in_strict_mode(self):
        path = _write_jsonl([_row(selected="play:not_in_legal")])
        try:
            with self.assertRaises(DemoValidationError):
                list(iter_demo_jsonl(path, strict=True))
        finally:
            path.unlink()


class DemoErrorMessageTests(unittest.TestCase):
    def test_missing_obs_yields_clear_error(self):
        bad = _row()
        bad.pop("obs")
        path = _write_jsonl([bad])
        try:
            with self.assertRaises(DemoValidationError) as cm:
                list(iter_demo_jsonl(path, strict=True))
        finally:
            path.unlink()
        self.assertIn("missing required fields", str(cm.exception))
        self.assertIn("obs", str(cm.exception))

    def test_invalid_json_strict_raises(self):
        path = Path(tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8").name)
        path.write_text("{not json\n", encoding="utf-8")
        try:
            with self.assertRaises(DemoValidationError):
                list(iter_demo_jsonl(path, strict=True))
        finally:
            path.unlink()

    def test_unsupported_version_skipped_lenient(self):
        bad = _row()
        bad["version"] = 99
        good = _row(episode_id="ep2", selected="play:defend")
        path = _write_jsonl([bad, good])
        try:
            samples = load_demo_dataset(path)
        finally:
            path.unlink()
        self.assertEqual(len(samples), 1)


class DemoBatchTests(unittest.TestCase):
    def test_build_batch_yields_ce_targets(self):
        path = _write_jsonl([
            _row(),
            _row(episode_id="ep2", selected="play:defend"),
            _row(episode_id="ep3", selected="end_turn", reasons=["save_exhaust_card"]),
        ])
        try:
            samples = load_demo_dataset(path)
        finally:
            path.unlink()
        batch = build_demo_training_batch(samples)
        self.assertEqual(batch["selected_action_indices"], [0, 1, 2])
        self.assertEqual(len(batch["legal_action_ids"]), 3)
        self.assertEqual(len(batch["legal_action_ids"][0]), 3)
        self.assertEqual(batch["legal_action_ids"][0][0], "play:strike")
        self.assertEqual(batch["encounter_ids"], ["kaiser_crab_boss"] * 3)
        self.assertEqual(batch["reason_tags"][2], ["save_exhaust_card"])
        self.assertEqual(batch["bc_target_policy"], [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        self.assertEqual(batch["target_hp_loss"], [6.0, 6.0, 6.0])
        self.assertEqual(batch["target_hp_loss_mask"], [1.0, 1.0, 1.0])
        self.assertEqual(batch["combat_win"], [1.0, 1.0, 1.0])
        self.assertEqual(batch["combat_win_mask"], [1.0, 1.0, 1.0])
        self.assertEqual(batch["turns"], [5.0, 5.0, 5.0])
        self.assertEqual(batch["turns_mask"], [1.0, 1.0, 1.0])

    def test_build_batch_masks_missing_outcomes(self):
        path = _write_jsonl([
            _row(outcome={}),
            _row(
                episode_id="ep2",
                selected="play:defend",
                outcome={"combat_win": False, "hp_loss": "bad", "turns": None},
            ),
        ])
        try:
            samples = load_demo_dataset(path)
        finally:
            path.unlink()
        batch = build_demo_training_batch(samples)
        self.assertEqual(batch["selected_action_indices"], [0, 1])
        self.assertEqual(batch["bc_target_policy"], [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        self.assertEqual(batch["target_hp_loss"], [0.0, 0.0])
        self.assertEqual(batch["target_hp_loss_mask"], [0.0, 0.0])
        self.assertEqual(batch["combat_win"], [0.0, 0.0])
        self.assertEqual(batch["combat_win_mask"], [0.0, 1.0])
        self.assertEqual(batch["turns"], [0.0, 0.0])
        self.assertEqual(batch["turns_mask"], [0.0, 0.0])

    def test_encounter_filter(self):
        path = _write_jsonl([
            _row(encounter_id="kaiser_crab_boss"),
            _row(encounter_id="ceremonial_beast_boss", episode_id="ep2", selected="play:defend"),
        ])
        try:
            samples = load_demo_dataset(path, encounter_filter=["ceremonial_beast_boss"])
        finally:
            path.unlink()
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].encounter_id, "ceremonial_beast_boss")


class DemoFileNotFoundTests(unittest.TestCase):
    def test_missing_file_raises_filenotfound(self):
        with self.assertRaises(FileNotFoundError):
            list(iter_demo_jsonl("/nonexistent/path/demo.jsonl"))


if __name__ == "__main__":
    unittest.main()
