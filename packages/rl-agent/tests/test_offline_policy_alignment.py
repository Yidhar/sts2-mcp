"""Tests for offline parquet/history policy alignment into live action logits."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import pytest


RL_AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(RL_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(RL_AGENT_ROOT))

torch = pytest.importorskip("torch")

from muzero.training.offline_policy_alignment import (  # noqa: E402
    DEFAULT_OFFLINE_POLICY_ALIGNMENT_TASKS,
    OfflinePolicyAligner,
    OfflinePolicyAlignmentConfig,
    build_offline_policy_sample,
    compute_offline_policy_alignment_loss,
    parse_offline_alignment_tasks,
)
from sts2_env.observation_common import MAX_ACTIONS  # noqa: E402
from sts2_env.observation_v3 import WorldTokenObservationEncoder  # noqa: E402


def _row(**overrides) -> dict:
    base = {
        "sample_id": "sample-1",
        "task": "regular_card_reward",
        "candidate_ids": ["CARD.POMMEL_STRIKE", "CARD.BATTLE_TRANCE", "<skip>"],
        "candidate_upgrade_levels": [0.0, 0.0, 0.0],
        "label_index": 1,
        "label_id": "CARD.BATTLE_TRANCE",
        "deck_ids": ["CARD.STRIKE_IRONCLAD", "CARD.DEFEND_IRONCLAD"],
        "deck_counts": [5.0, 4.0],
        "deck_upgraded_counts": [0.0, 0.0],
        "deck_max_upgrade_levels": [0.0, 0.0],
        "relic_ids": ["RELIC.BURNING_BLOOD"],
        "monster_ids": [],
        "room_type": "Monster",
        "map_point_type": "Monster",
        # floor, act_index, path, asc, hp_before, current_hp, max_hp, hp ratios, gold...
        "scalars": [5.0, 0.0, 0.0, 0.0, 70.0, 62.0, 80.0, 0.875, 0.775, 0.0, 99.0]
        + [0.0] * 10,
    }
    base.update(overrides)
    return base


class _FakeNetwork(torch.nn.Module):
    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(0.0))
        self.register_buffer("_logits", logits)

    def initial_inference(self, obs_torch: dict[str, torch.Tensor]):
        batch = int(obs_torch["action_mask"].shape[0])
        return SimpleNamespace(
            policy_logits=self._logits[:batch].to(obs_torch["action_mask"].device) + self.bias
        )


class OfflinePolicyAlignmentTests(unittest.TestCase):
    def test_parse_tasks_supports_default_commas_semicolons_and_sequences(self):
        self.assertEqual(parse_offline_alignment_tasks(None), DEFAULT_OFFLINE_POLICY_ALIGNMENT_TASKS)
        self.assertEqual(
            parse_offline_alignment_tasks("regular_card_reward,smith_target"),
            ("regular_card_reward", "smith_target"),
        )
        self.assertEqual(
            parse_offline_alignment_tasks("regular_card_reward; smith_target"),
            ("regular_card_reward", "smith_target"),
        )
        self.assertEqual(
            parse_offline_alignment_tasks(["rest_action", "shop_remove_binary"]),
            ("rest_action", "shop_remove_binary"),
        )

    def test_route_tasks_are_rejected_by_default(self):
        with self.assertRaises(ValueError):
            OfflinePolicyAlignmentConfig(
                root="unused",
                tasks=("route_room_type",),
                allow_route=False,
            )

    def test_sample_conversion_encodes_selected_action_unmasked(self):
        sample = build_offline_policy_sample(_row(), "regular_card_reward")
        self.assertEqual(sample.selected_action_index, 1)
        self.assertEqual(len(sample.legal_actions), 3)

        encoded = WorldTokenObservationEncoder(use_text=False).encode(sample.obs, sample.legal_actions)
        self.assertEqual(encoded["action_mask"].shape[0], MAX_ACTIONS)
        self.assertGreater(float(encoded["action_mask"][sample.selected_action_index]), 0.0)
        self.assertGreaterEqual(float(encoded["action_mask"][:3].sum()), 3.0)

    def test_aligner_next_batch_uses_live_encoded_indices(self):
        rows = [_row(sample_id="sample-1"), _row(sample_id="sample-2", label_index=2)]

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                "muzero.training.offline_policy_alignment.load_task_rows",
                lambda root, task, fmt="parquet": rows,
            )
            aligner = OfflinePolicyAligner(
                OfflinePolicyAlignmentConfig(
                    root="unused",
                    tasks=("regular_card_reward",),
                    batch_size=2,
                    shuffle=False,
                )
            )
            batch = aligner.next_batch()

        self.assertIsNotNone(batch)
        assert batch is not None
        self.assertEqual(batch["selected_action_indices"].tolist(), [1, 2])
        self.assertEqual(batch["tasks"], ["regular_card_reward", "regular_card_reward"])
        self.assertEqual(batch["action_counts"].tolist(), [3.0, 3.0])
        self.assertEqual(len(batch["obs_list"]), 2)

    def test_shadow_loss_reports_metrics_but_returns_zero(self):
        sample = build_offline_policy_sample(_row(), "regular_card_reward")
        encoded = WorldTokenObservationEncoder(use_text=False).encode(sample.obs, sample.legal_actions)
        batch = {
            "obs_list": [encoded],
            "selected_action_indices": torch.tensor([1]).numpy(),
            "action_counts": torch.tensor([3.0]).numpy(),
            "tasks": ["regular_card_reward"],
            "sample_ids": ["sample-1"],
        }

        logits = torch.zeros(1, MAX_ACTIONS)
        logits[0, 1] = 5.0
        net = _FakeNetwork(logits)
        loss, metrics = compute_offline_policy_alignment_loss(
            net,
            batch,
            device="cpu",
            weight=0.0,
            shadow_only=True,
        )

        self.assertEqual(float(loss.item()), 0.0)
        self.assertFalse(loss.requires_grad)
        self.assertEqual(metrics["offline_alignment/loss_applied"], 0.0)
        self.assertEqual(metrics["offline_alignment/top1_match"], 1.0)
        self.assertLess(metrics["offline_alignment/ce"], 0.05)
        self.assertEqual(metrics["offline_alignment/regular_card_reward/label_valid_rate"], 1.0)

    def test_active_loss_requires_grad_and_scales_by_weight(self):
        sample = build_offline_policy_sample(_row(), "regular_card_reward")
        encoded = WorldTokenObservationEncoder(use_text=False).encode(sample.obs, sample.legal_actions)
        batch = {
            "obs_list": [encoded],
            "selected_action_indices": torch.tensor([1]).numpy(),
            "action_counts": torch.tensor([3.0]).numpy(),
            "tasks": ["regular_card_reward"],
            "sample_ids": ["sample-1"],
        }

        logits = torch.zeros(1, MAX_ACTIONS)
        logits[0, 0] = 2.0
        net = _FakeNetwork(logits)
        loss, metrics = compute_offline_policy_alignment_loss(
            net,
            batch,
            device="cpu",
            weight=0.1,
            shadow_only=False,
        )

        self.assertTrue(loss.requires_grad)
        self.assertGreater(float(loss.item()), 0.0)
        self.assertEqual(metrics["offline_alignment/loss_applied"], 1.0)
        self.assertEqual(metrics["offline_alignment/label_valid_rate"], 1.0)

    def test_negative_weight_is_clamped_to_shadow_zero_loss(self):
        sample = build_offline_policy_sample(_row(), "regular_card_reward")
        encoded = WorldTokenObservationEncoder(use_text=False).encode(sample.obs, sample.legal_actions)
        batch = {
            "obs_list": [encoded],
            "selected_action_indices": torch.tensor([1]).numpy(),
            "action_counts": torch.tensor([3.0]).numpy(),
            "tasks": ["regular_card_reward"],
            "sample_ids": ["sample-1"],
        }

        logits = torch.zeros(1, MAX_ACTIONS)
        net = _FakeNetwork(logits)
        loss, metrics = compute_offline_policy_alignment_loss(
            net,
            batch,
            device="cpu",
            weight=-1.0,
            shadow_only=False,
        )

        self.assertEqual(float(loss.item()), 0.0)
        self.assertFalse(loss.requires_grad)
        self.assertEqual(metrics["offline_alignment/weight"], 0.0)
        self.assertEqual(metrics["offline_alignment/loss_applied"], 0.0)

    def test_label_oor_reports_head_dimension_mismatch(self):
        row = _row(
            candidate_ids=[
                "CARD.STRIKE_IRONCLAD",
                "CARD.DEFEND_IRONCLAD",
                "CARD.BASH",
                "CARD.POMMEL_STRIKE",
                "CARD.BATTLE_TRANCE",
                "<skip>",
            ],
            candidate_upgrade_levels=[0.0] * 6,
            label_index=5,
        )
        sample = build_offline_policy_sample(row, "regular_card_reward")
        encoded = WorldTokenObservationEncoder(use_text=False).encode(sample.obs, sample.legal_actions)
        self.assertGreater(float(encoded["action_mask"][sample.selected_action_index]), 0.0)
        batch = {
            "obs_list": [encoded],
            "selected_action_indices": torch.tensor([sample.selected_action_index]).numpy(),
            "action_counts": torch.tensor([6.0]).numpy(),
            "tasks": ["regular_card_reward"],
            "sample_ids": ["sample-1"],
        }

        net = _FakeNetwork(torch.zeros(1, 4))
        loss, metrics = compute_offline_policy_alignment_loss(
            net,
            batch,
            device="cpu",
            weight=0.1,
            shadow_only=False,
        )

        self.assertEqual(float(loss.item()), 0.0)
        self.assertEqual(metrics["offline_alignment/label_valid_rate"], 0.0)
        self.assertEqual(metrics["offline_alignment/label_in_range_rate"], 0.0)
        self.assertEqual(metrics["offline_alignment/label_unmasked_rate"], 0.0)
        self.assertEqual(metrics["offline_alignment/label_oor_count"], 1.0)
        self.assertEqual(metrics["offline_alignment/regular_card_reward/label_oor_count"], 1.0)


if __name__ == "__main__":
    unittest.main()
