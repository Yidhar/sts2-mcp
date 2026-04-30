"""Tests for the selected end_turn context dump (TASK-A1).

Covers the central classifier `_classify_end_turn_action` shared between the
bias-side metrics and the JSONL tracker, and the JSONL writer's behaviour for
forced / strategic / bad / transient cases.
"""

from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


RL_AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(RL_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(RL_AGENT_ROOT))

from muzero.train import MuZeroTrainer


def _ctx(
    *,
    end_turn_indices: list[int] | None = None,
    wasteful: bool = False,
    strategic_defer_available: bool = False,
    positive_progress_count: int = 0,
    urgent_positive_count: int = 0,
    deferable_positive_count: int = 0,
    energy: float = 0.0,
) -> dict[str, object]:
    return {
        "end_turn_indices": list(end_turn_indices or [0]),
        "wasteful": wasteful,
        "strategic_defer_available": strategic_defer_available,
        "positive_progress_count": positive_progress_count,
        "urgent_positive_count": urgent_positive_count,
        "deferable_positive_count": deferable_positive_count,
        "energy": energy,
        "positive_indices": list(range(positive_progress_count)),
        "urgent_positive_indices": list(range(urgent_positive_count)),
        "deferable_positive_indices": list(range(deferable_positive_count)),
    }


class ClassifyEndTurnTests(unittest.TestCase):
    def test_bad_overrides_other_classes(self):
        # Energy left + urgent action available + chose end_turn => bad
        ctx = _ctx(
            wasteful=True,
            strategic_defer_available=True,  # bad must dominate
            energy=2.0,
            positive_progress_count=3,
            urgent_positive_count=2,
            deferable_positive_count=1,
        )
        cls, flags = MuZeroTrainer._classify_end_turn_action(ctx)
        self.assertEqual(cls, "bad_end_turn")
        self.assertTrue(flags["has_energy_and_positive_action"])
        self.assertTrue(flags["has_urgent_or_mandatory_action"])

    def test_forced_when_no_positive_action(self):
        ctx = _ctx(
            energy=0.0,
            positive_progress_count=0,
            urgent_positive_count=0,
        )
        cls, flags = MuZeroTrainer._classify_end_turn_action(ctx)
        self.assertEqual(cls, "forced_end_turn")
        self.assertTrue(flags["no_legal_positive_action"])
        self.assertFalse(flags["has_energy_and_positive_action"])

    def test_strategic_defer_when_only_deferable(self):
        # Has positive progress but none urgent, only deferable exhaust/refund =>
        # waiting for next turn loop is legitimate.
        ctx = _ctx(
            wasteful=False,
            strategic_defer_available=True,
            energy=1.0,
            positive_progress_count=2,
            urgent_positive_count=0,
            deferable_positive_count=2,
        )
        cls, flags = MuZeroTrainer._classify_end_turn_action(ctx)
        self.assertEqual(cls, "strategic_defer_end_turn")
        self.assertTrue(flags["has_strategic_defer_reason"])
        self.assertTrue(flags["has_deferable_action"])
        self.assertFalse(flags["has_urgent_or_mandatory_action"])

    def test_transient_only_end_turn_marks_forced(self):
        # Bridge says only end_turn was actionable in this transient frame:
        # treat as forced regardless of positive availability heuristic.
        ctx = _ctx(positive_progress_count=2, urgent_positive_count=1, energy=2.0)
        cls, flags = MuZeroTrainer._classify_end_turn_action(
            ctx,
            action_diagnostics={"transient_only_end_turn": True},
        )
        self.assertEqual(cls, "forced_end_turn")
        self.assertTrue(flags["transient_only_end_turn"])

    def test_unknown_when_no_end_turn_indices(self):
        # If detector says no end_turn was even available, do not classify it.
        ctx = _ctx(end_turn_indices=[], positive_progress_count=2, urgent_positive_count=1)
        cls, _flags = MuZeroTrainer._classify_end_turn_action(ctx)
        self.assertEqual(cls, "unknown")


class DumpSelectedEndTurnContextTests(unittest.TestCase):
    def _make_trainer_stub(self, log_dir: Path) -> MuZeroTrainer:
        # Build a minimal stub object with just the attributes the dump method
        # touches; we deliberately avoid constructing the heavy MuZeroTrainer.
        stub = MuZeroTrainer.__new__(MuZeroTrainer)
        stub.log_dir = str(log_dir)
        stub.total_steps = 42
        stub.episode_count = 7
        stub._end_turn_context_dump_count = 0
        stub._end_turn_context_dump_max = 100
        stub._end_turn_context_dump_disabled = False
        return stub

    def test_writes_jsonl_with_classification_and_top_actions(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = self._make_trainer_stub(Path(tmp))
            ctx = _ctx(
                end_turn_indices=[2],
                wasteful=True,
                positive_progress_count=2,
                urgent_positive_count=1,
                energy=2.0,
            )
            mask = np.zeros(8, dtype=np.float32)
            mask[0] = mask[1] = mask[2] = 1.0
            legal = [
                {"semantic": {"family": "play_card"}, "title": "Strike", "card": {"id": "STRIKE", "title": "Strike"}},
                {"semantic": {"family": "play_card"}, "title": "Defend", "card": {"id": "DEFEND", "title": "Defend"}},
                {"semantic": {"family": "end_turn"}, "title": "End Turn"},
            ]
            policy = np.array([0.1, 0.2, 0.7, 0, 0, 0, 0, 0], dtype=np.float32)
            with mock.patch.object(MuZeroTrainer, "_incoming_damage_pressure", return_value=(0.0, 0.0, 50.0)), \
                 mock.patch.object(MuZeroTrainer, "_semantic_family", side_effect=lambda a: (a or {}).get("semantic", {}).get("family", "")), \
                 mock.patch.object(MuZeroTrainer, "_is_x_cost_action", return_value=False), \
                 mock.patch.object(MuZeroTrainer, "_is_kaiser_facing_change_action", return_value=False), \
                 mock.patch.object(MuZeroTrainer, "_boss_context_max", return_value=0.0):
                trainer._dump_selected_end_turn_context(
                    encoded_obs={},
                    raw_obs={"combat": {"round": 3, "hand": ["a", "b"]}, "player": {"hp": 50, "max_hp": 80}},
                    action_mask=mask,
                    legal_actions=legal,
                    chosen_idx=2,
                    context=ctx,
                    search_policy=policy,
                    search_stats={},
                    action_diagnostics=None,
                    encounter="kaiser_crab_boss",
                    tier="boss",
                )

            path = Path(tmp) / "diagnostics" / "end_turn_contexts.jsonl"
            self.assertTrue(path.exists(), f"expected JSONL at {path}")
            lines = path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 1)
            payload = json.loads(lines[0])
            self.assertEqual(payload["selected_family"], "end_turn")
            self.assertEqual(payload["end_turn_class"], "bad_end_turn")
            self.assertEqual(payload["selected_action_idx"], 2)
            self.assertEqual(payload["encounter_id"], "kaiser_crab_boss")
            self.assertEqual(payload["tier"], "boss")
            self.assertEqual(payload["counts"]["legal_action_count"], 3)
            self.assertEqual(payload["counts"]["positive_action_count"], 2)
            self.assertEqual(payload["counts"]["urgent_action_count"], 1)
            self.assertGreaterEqual(len(payload["top_legal_actions"]), 1)
            chosen_entries = [a for a in payload["top_legal_actions"] if a["is_chosen"]]
            self.assertEqual(len(chosen_entries), 1)
            self.assertEqual(chosen_entries[0]["family"], "end_turn")

    def test_disabled_via_env_skips_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = self._make_trainer_stub(Path(tmp))
            trainer._end_turn_context_dump_disabled = True
            with mock.patch.object(MuZeroTrainer, "_incoming_damage_pressure", return_value=(0.0, 0.0, 50.0)):
                trainer._dump_selected_end_turn_context(
                    encoded_obs={},
                    raw_obs=None,
                    action_mask=np.ones(2, dtype=np.float32),
                    legal_actions=[{}, {}],
                    chosen_idx=0,
                    context=_ctx(),
                    search_policy=None,
                    search_stats=None,
                    action_diagnostics=None,
                    encounter="",
                    tier="",
                )
            self.assertFalse((Path(tmp) / "diagnostics" / "end_turn_contexts.jsonl").exists())

    def test_cap_stops_after_max(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = self._make_trainer_stub(Path(tmp))
            trainer._end_turn_context_dump_max = 1
            mask = np.array([1.0], dtype=np.float32)
            with mock.patch.object(MuZeroTrainer, "_incoming_damage_pressure", return_value=(0.0, 0.0, 50.0)), \
                 mock.patch.object(MuZeroTrainer, "_semantic_family", return_value="end_turn"), \
                 mock.patch.object(MuZeroTrainer, "_is_x_cost_action", return_value=False), \
                 mock.patch.object(MuZeroTrainer, "_is_kaiser_facing_change_action", return_value=False), \
                 mock.patch.object(MuZeroTrainer, "_boss_context_max", return_value=0.0):
                for _ in range(3):
                    trainer._dump_selected_end_turn_context(
                        encoded_obs={},
                        raw_obs={"combat": {}, "player": {}},
                        action_mask=mask,
                        legal_actions=[{"semantic": {"family": "end_turn"}}],
                        chosen_idx=0,
                        context=_ctx(),
                        search_policy=np.array([1.0], dtype=np.float32),
                        search_stats={},
                        action_diagnostics=None,
                        encounter="",
                        tier="",
                    )
            path = Path(tmp) / "diagnostics" / "end_turn_contexts.jsonl"
            self.assertTrue(path.exists())
            lines = path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 1, "cap=1 should bound the writes after the first entry")


if __name__ == "__main__":
    unittest.main()
