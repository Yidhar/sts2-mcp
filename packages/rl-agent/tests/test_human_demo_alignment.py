"""Tests for train-time human demo policy alignment."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest

import pytest


RL_AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(RL_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(RL_AGENT_ROOT))

torch = pytest.importorskip("torch")

from muzero.training.human_demo_alignment import (  # noqa: E402
    HumanDemoAlignmentConfig,
    HumanDemoPolicyAligner,
    compute_human_demo_policy_alignment_loss,
)
from sts2_env.observation_common import MAX_ACTIONS  # noqa: E402


def _row(*, episode_id: str = "ep1", selected: str = "play:defend") -> dict:
    legal_actions = [
        {"action_id": "play:strike", "kind": "play_card", "title": "Strike"},
        {"action_id": "play:defend", "kind": "play_card", "title": "Defend"},
        {"action_id": "end_turn", "kind": "combat", "title": "End Turn"},
    ]
    return {
        "version": 1,
        "source": "human",
        "episode_id": episode_id,
        "encounter_id": "ENCOUNTER.TEST",
        "tier": "normal",
        "turn": 1,
        "step_in_turn": 0,
        "obs": {
            "phase": "combat",
            "decision_domain": "combat",
            "player": {"hp": 60, "max_hp": 80, "energy": 3},
            "combat": {"round": 1},
        },
        "legal_actions": legal_actions,
        "selected_action_id": selected,
        "reason_tags": ["block_incoming"],
        "outcome": {"combat_win": True, "hp_loss": 0, "turns": 3},
    }


def _write_jsonl(rows: list[dict]) -> Path:
    handle = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8")
    for row in rows:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    handle.close()
    return Path(handle.name)


class _FakeNetwork(torch.nn.Module):
    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(0.0))
        self.register_buffer("_logits", logits)

    def initial_inference(self, obs_torch: dict[str, torch.Tensor]):
        batch = int(obs_torch["action_mask"].shape[0])
        return SimpleNamespace(policy_logits=self._logits[:batch].to(obs_torch["action_mask"].device) + self.bias)


class HumanDemoPolicyAlignerTests(unittest.TestCase):
    def test_next_batch_encodes_raw_obs_and_preserves_selected_index(self):
        path = _write_jsonl([_row(), _row(episode_id="ep2", selected="end_turn")])
        try:
            aligner = HumanDemoPolicyAligner(
                HumanDemoAlignmentConfig(paths=(path,), batch_size=2, shuffle=False)
            )
            batch = aligner.next_batch()
        finally:
            path.unlink()

        self.assertIsNotNone(batch)
        assert batch is not None
        self.assertEqual(batch["selected_action_indices"].tolist(), [1, 2])
        self.assertEqual(len(batch["obs_list"]), 2)
        self.assertEqual(batch["obs_list"][0]["action_mask"].shape[0], MAX_ACTIONS)
        self.assertGreaterEqual(batch["obs_list"][0]["action_mask"][:3].sum(), 3.0)
        self.assertEqual(batch["action_counts"].tolist(), [3.0, 3.0])

    def test_shadow_loss_reports_metrics_but_returns_zero(self):
        path = _write_jsonl([_row(), _row(episode_id="ep2", selected="end_turn")])
        try:
            aligner = HumanDemoPolicyAligner(
                HumanDemoAlignmentConfig(paths=(path,), batch_size=2, shuffle=False)
            )
            batch = aligner.next_batch()
        finally:
            path.unlink()
        assert batch is not None

        logits = torch.zeros(2, MAX_ACTIONS)
        logits[0, 1] = 5.0
        logits[1, 2] = 5.0
        net = _FakeNetwork(logits)
        loss, metrics = compute_human_demo_policy_alignment_loss(
            net,
            batch,
            device="cpu",
            weight=0.0,
            shadow_only=True,
        )

        self.assertEqual(float(loss.item()), 0.0)
        self.assertFalse(loss.requires_grad)
        self.assertEqual(metrics["human_demo_alignment/loss_applied"], 0.0)
        self.assertEqual(metrics["human_demo_alignment/top1_match"], 1.0)
        self.assertLess(metrics["human_demo_alignment/ce"], 0.05)

    def test_active_loss_requires_grad_and_scales_by_weight(self):
        path = _write_jsonl([_row()])
        try:
            aligner = HumanDemoPolicyAligner(
                HumanDemoAlignmentConfig(paths=(path,), batch_size=1, shuffle=False)
            )
            batch = aligner.next_batch()
        finally:
            path.unlink()
        assert batch is not None

        logits = torch.zeros(1, MAX_ACTIONS)
        logits[0, 0] = 2.0
        net = _FakeNetwork(logits)
        loss, metrics = compute_human_demo_policy_alignment_loss(
            net,
            batch,
            device="cpu",
            weight=0.1,
            shadow_only=False,
        )

        self.assertTrue(loss.requires_grad)
        self.assertGreater(float(loss.item()), 0.0)
        self.assertEqual(metrics["human_demo_alignment/loss_applied"], 1.0)
        self.assertEqual(metrics["human_demo_alignment/label_valid_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
