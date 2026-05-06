"""Tests for the extended loss-spike dump (P0-7)."""

from __future__ import annotations

import json
import math
import sys
import tempfile
import types
import unittest
from pathlib import Path
from typing import Any


RL_AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(RL_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(RL_AGENT_ROOT))


def _read_lines(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(json.loads(line))
    return out


class _MockTrainer:
    """Tiny stand-in carrying just enough state for ``_dump_loss_spike``."""

    def __init__(self, log_dir: Path):
        self.log_dir = str(log_dir)
        self.total_steps = 12345

    # Bind the real method to our mock so we exercise the production code path
    # without spinning up the full MuZeroTrainer (which needs torch + bridge).
    @classmethod
    def with_dump(cls, log_dir: Path):
        from muzero.train import MuZeroTrainer  # noqa: WPS433
        instance = cls(log_dir)
        instance._dump_loss_spike = MuZeroTrainer._dump_loss_spike.__get__(instance, cls)  # type: ignore[attr-defined]
        return instance


class LossSpikeDumpP0Seven(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.log_dir = Path(self._tmp.name)
        self.trainer = _MockTrainer.with_dump(self.log_dir)
        self.path = self.log_dir / "diagnostics" / "loss_spikes.jsonl"

    def test_below_threshold_no_dump(self):
        self.trainer._dump_loss_spike(
            step_k=0,
            future_world_aux_value=0.5,
            future_bank_state_value=0.01,
            future_bank_delta_value=0.02,
            batch_size=16,
        )
        self.assertFalse(self.path.exists())

    def test_above_threshold_writes_v2_payload(self):
        self.trainer._dump_loss_spike(
            step_k=2,
            future_world_aux_value=153.7,
            future_bank_state_value=151.2,
            future_bank_delta_value=4.5,
            batch_size=16,
            extra_losses={"policy_loss": 0.3, "value_loss": 2.1},
            sample_tier_flags=None,
        )
        rows = _read_lines(self.path)
        self.assertEqual(len(rows), 1)
        rec = rows[0]
        self.assertEqual(rec["schema_version"], 2)
        self.assertEqual(rec["kind"], "loss_spike")
        self.assertEqual(rec["step_k"], 2)
        self.assertEqual(rec["batch_size"], 16)
        self.assertIn("losses", rec)
        self.assertAlmostEqual(rec["losses"]["future_world_aux_loss"], 153.7, places=2)
        self.assertAlmostEqual(rec["losses"]["policy_loss"], 0.3, places=4)
        self.assertEqual(rec["trigger"]["dominant_key"], "future_world_aux_loss")
        self.assertIn("future_world_aux_loss", rec["trigger"]["crossed_keys"])
        self.assertIn("future_bank_state_loss", rec["trigger"]["crossed_keys"])
        self.assertNotIn("future_bank_delta_loss", rec["trigger"]["crossed_keys"])
        self.assertTrue(rec["finite_guard"]["all_finite"])
        self.assertEqual(rec["finite_guard"]["non_finite_count"], 0)
        self.assertIn("loss_above_threshold", rec["quarantine_reason"])

    def test_non_finite_value_quarantined(self):
        self.trainer._dump_loss_spike(
            step_k=1,
            future_world_aux_value=float("nan"),
            future_bank_state_value=0.01,
            future_bank_delta_value=0.02,
            batch_size=16,
        )
        rows = _read_lines(self.path)
        self.assertEqual(len(rows), 1)
        rec = rows[0]
        self.assertFalse(rec["finite_guard"]["all_finite"])
        self.assertEqual(rec["finite_guard"]["non_finite_count"], 1)
        self.assertIn("non_finite_loss", rec["quarantine_reason"])
        # NaN must be serialized as a string (JSON spec disallows NaN literal).
        self.assertIsInstance(rec["losses"]["future_world_aux_loss"], str)
        self.assertEqual(rec["trigger"]["non_finite_keys"], ["future_world_aux_loss"])

    def test_non_finite_inf_quarantined(self):
        self.trainer._dump_loss_spike(
            step_k=0,
            future_world_aux_value=float("inf"),
            future_bank_state_value=0.0,
            future_bank_delta_value=0.0,
            batch_size=16,
        )
        rows = _read_lines(self.path)
        self.assertEqual(len(rows), 1)
        self.assertFalse(rows[0]["finite_guard"]["all_finite"])
        self.assertIn("non_finite_loss", rows[0]["quarantine_reason"])

    def test_action_index_distribution_recorded(self):
        # Plain Python list — exercises the non-tensor branch.
        self.trainer._dump_loss_spike(
            step_k=0,
            future_world_aux_value=200.0,
            future_bank_state_value=0.0,
            future_bank_delta_value=0.0,
            batch_size=8,
            action_indices=[3, 3, 3, 7, 7, 1, 9, 3],
        )
        rec = _read_lines(self.path)[0]
        self.assertIn("action_index_dist", rec)
        top = rec["action_index_dist"][0]
        self.assertEqual(top["action_index"], 3)
        self.assertEqual(top["count"], 4)

    def test_max_dumps_cap(self):
        # Force the cap to 2 so the third call must be silently dropped.
        for _ in range(5):
            self.trainer._dump_loss_spike(
                step_k=0,
                future_world_aux_value=200.0,
                future_bank_state_value=0.0,
                future_bank_delta_value=0.0,
                batch_size=16,
                max_dumps=2,
            )
        rows = _read_lines(self.path)
        self.assertEqual(len(rows), 2)


if __name__ == "__main__":
    unittest.main()
