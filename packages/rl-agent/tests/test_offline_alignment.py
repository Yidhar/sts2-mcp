"""Tests for offline alignment data plumbing."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pytest


RL_AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(RL_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(RL_AGENT_ROOT))

pytest.importorskip("torch")

from muzero.training.offline_alignment import (  # noqa: E402
    OfflineAlignmentConfig,
    OfflineAlignmentDataModule,
    summarize_task_rows,
)
from offline_training_data import _normalize_v2_candidate_row  # noqa: E402


def _state(**overrides):
    base = {
        "split": "train",
        "build_id": "build-1",
        "run_id": "run-1",
        "character": "ironclad",
        "room_type": "Monster",
        "map_point_type": "Monster",
        "room_model_id": "fungalmist",
        "deck_ids": ["Strike", "Defend"],
        "deck_counts": [4.0, 4.0],
        "deck_upgraded_counts": [0.0, 0.0],
        "deck_max_upgrade_levels": [0.0, 0.0],
        "relic_ids": ["BurningBlood"],
        "monster_ids": ["FungalMist"],
        # floor, act, path, asc, hp_before, current_hp, max_hp, hp_before_ratio,
        # current_hp_ratio, then filler fields used by offline_training_data.
        "scalars": [5.0, 0.0, 0.0, 0.0, 42.0, 36.0, 80.0, 0.525, 0.45]
        + [0.0] * 12,
    }
    base.update(overrides)
    return base


def _candidate_row(sample_id: str, *, candidates, label_index: int, label_id: str):
    return {
        **_state(),
        "sample_id": sample_id,
        "task": "regular_card_reward",
        "candidate_ids": list(candidates),
        "candidate_counts": [1.0] * len(candidates),
        "candidate_upgrade_levels": [0.0] * len(candidates),
        "label_index": label_index,
        "label_id": label_id,
        "skip_available": "<skip>" in candidates,
    }


class OfflineAlignmentSummaryTests(unittest.TestCase):
    def test_candidate_summary_counts_skip_and_label_index_validity(self):
        rows = [
            _candidate_row(
                "s1",
                candidates=["PommelStrike", "BattleTrance", "<skip>"],
                label_index=1,
                label_id="BattleTrance",
            ),
            _candidate_row(
                "s2",
                candidates=["Cleave", "<skip>"],
                label_index=1,
                label_id="<skip>",
            ),
        ]

        summary = summarize_task_rows(
            "regular_card_reward",
            rows,
            metadata={"task_family": "candidate", "scalar_dim": 21},
        )

        self.assertEqual(summary.task, "regular_card_reward")
        self.assertEqual(summary.family, "candidate")
        self.assertEqual(summary.rows, 2)
        self.assertEqual(summary.scalar_dim, 21)
        self.assertEqual(summary.label_counts, {"<skip>": 1, "BattleTrance": 1})
        self.assertEqual(summary.candidate_count_min, 2)
        self.assertEqual(summary.candidate_count_max, 3)
        self.assertAlmostEqual(summary.candidate_count_mean, 2.5)
        self.assertAlmostEqual(summary.has_label_index_rate or 0.0, 1.0)
        self.assertAlmostEqual(summary.label_index_valid_rate or 0.0, 1.0)
        self.assertAlmostEqual(summary.skip_available_rate or 0.0, 1.0)
        self.assertAlmostEqual(summary.skip_label_rate or 0.0, 0.5)
        self.assertAlmostEqual(summary.deck_size_mean, 8.0)
        self.assertAlmostEqual(summary.current_hp_ratio_mean, 0.45)


class OfflineAlignmentDataModuleTests(unittest.TestCase):
    def test_data_module_loads_and_builds_v2_candidate_loader(self):
        rows = [
            _candidate_row(
                "s1",
                candidates=["PommelStrike", "BattleTrance", "<skip>"],
                label_index=0,
                label_id="PommelStrike",
            ),
            _candidate_row(
                "s2",
                candidates=["Cleave", "<skip>"],
                label_index=1,
                label_id="<skip>",
            ),
        ]

        config = OfflineAlignmentConfig(
            root=Path("unused"),
            tasks=("regular_card_reward",),
            fmt="jsonl",
            max_rows_per_task=None,
        )
        with patch("muzero.training.offline_alignment.load_task_rows", return_value=rows):
            module = OfflineAlignmentDataModule(config).load()

        self.assertEqual(module.task_summary("regular_card_reward").rows, 2)
        loader = module.make_loader("regular_card_reward", batch_size=2, shuffle=False)
        batch = next(iter(loader))

        self.assertEqual(batch["task"], "regular_card_reward")
        self.assertEqual(tuple(batch["candidate_ids"].shape), (2, 3))
        self.assertEqual(tuple(batch["candidate_mask"].shape), (2, 3))
        self.assertEqual(batch["labels"].tolist(), [0, 1])
        self.assertEqual(batch["sample_ids"], ["s1", "s2"])

    def test_max_rows_per_task_is_deterministic(self):
        rows = [
            _candidate_row(f"s{i}", candidates=["A", "B"], label_index=0, label_id="A")
            for i in range(10)
        ]
        config = OfflineAlignmentConfig(
            root=Path("unused"),
            tasks=("regular_card_reward",),
            fmt="jsonl",
            max_rows_per_task=4,
            seed=123,
        )
        with patch("muzero.training.offline_alignment.load_task_rows", return_value=rows):
            left = OfflineAlignmentDataModule(config).load()
        with patch("muzero.training.offline_alignment.load_task_rows", return_value=rows):
            right = OfflineAlignmentDataModule(config).load()

        self.assertEqual(
            [row["sample_id"] for row in left.tasks["regular_card_reward"].rows],
            [row["sample_id"] for row in right.tasks["regular_card_reward"].rows],
        )
        self.assertEqual(len(left.tasks["regular_card_reward"].rows), 4)

    def test_v2_optional_skip_tasks_always_append_skip_candidate(self):
        raw = {
            **_state(
                deck_before={
                    "cards": [
                        {"id": "CARD.STRIKE", "count": 4, "upgraded_count": 0, "max_upgrade_level": 0},
                        {"id": "CARD.DEFEND", "count": 4, "upgraded_count": 0, "max_upgrade_level": 0},
                    ],
                    "deck_size": 8,
                    "distinct_cards": 2,
                    "upgraded_card_count": 0,
                },
                relic_ids_before=["RELIC.BURNING_BLOOD"],
            ),
            "sample_id": "skip-align-1",
            "task": "regular_card_reward",
            "option_ids": ["CARD.POMMEL_STRIKE", "CARD.BATTLE_TRANCE", "CARD.CLEAVE"],
            "option_counts": {},
            "options": [],
            "label_id": "CARD.POMMEL_STRIKE",
            "skip_available": False,
        }

        sample = _normalize_v2_candidate_row(raw, "regular_card_reward")

        self.assertIsNotNone(sample)
        assert sample is not None
        self.assertEqual(sample["candidate_ids"], ["CARD.POMMEL_STRIKE", "CARD.BATTLE_TRANCE", "CARD.CLEAVE", "<skip>"])
        self.assertEqual(sample["label_index"], 0)
        self.assertTrue(sample["skip_available"])


if __name__ == "__main__":
    unittest.main()
