"""Tests for the starter_early deck-stage classifier and boost sampling.

These tests exercise only the bits of CombatSnapshotPool that don't need
the full torch / observation stack: row classification and the boost
routing in sample(). That lets them run in the Windows venv without
pulling ROCm-specific dependencies.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np


RL_AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(RL_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(RL_AGENT_ROOT))

from combat_snapshot_dataset import (
    CombatSnapshotPool,
    STARTER_EARLY_MAX_DECK_SIZE,
    STARTER_EARLY_MAX_FLOOR,
    infer_deck_stage,
    infer_deck_stage_from_row,
)


def _row(
    *,
    encounter_id: str,
    floor_number: int,
    deck_size: int,
    tier_suffix: str = "_WEAK",
) -> dict:
    """Minimal row shape needed by CombatSnapshotPool + the classifier."""
    return {
        "encounter_id": encounter_id + tier_suffix,
        "floor_number": floor_number,
        "deck_card_ids": [f"CARD.FILLER_{i}" for i in range(deck_size)],
        "room_type": "combat",
    }


class TestInferDeckStage:
    def test_starter_early_on_floor_1_starter_deck(self) -> None:
        assert infer_deck_stage(floor_number=1, deck_size=10) == "starter_early"

    def test_starter_early_on_floor_3_with_a_few_rewards(self) -> None:
        assert (
            infer_deck_stage(
                floor_number=STARTER_EARLY_MAX_FLOOR,
                deck_size=STARTER_EARLY_MAX_DECK_SIZE,
            )
            == "starter_early"
        )

    def test_rest_when_deck_grown_even_on_floor_1(self) -> None:
        # 15-card deck on floor 1 means this isn't really a starter state.
        assert infer_deck_stage(floor_number=1, deck_size=15) == "rest"

    def test_rest_when_floor_beyond_threshold(self) -> None:
        assert (
            infer_deck_stage(
                floor_number=STARTER_EARLY_MAX_FLOOR + 1,
                deck_size=10,
            )
            == "rest"
        )

    def test_rest_on_invalid_input(self) -> None:
        assert infer_deck_stage(floor_number=None, deck_size=10) == "rest"
        assert infer_deck_stage(floor_number=1, deck_size=None) == "rest"
        assert infer_deck_stage(floor_number=0, deck_size=10) == "rest"
        assert infer_deck_stage(floor_number=1, deck_size=0) == "rest"

    def test_infer_from_row_reads_deck_card_ids(self) -> None:
        row = _row(encounter_id="ENCOUNTER.LOUSE", floor_number=2, deck_size=11)
        assert infer_deck_stage_from_row(row) == "starter_early"
        row_late = _row(encounter_id="ENCOUNTER.LOUSE", floor_number=12, deck_size=20)
        assert infer_deck_stage_from_row(row_late) == "rest"


class TestCombatSnapshotPoolStarterEarlyBoost:
    def _mixed_pool(self, starter_early_boost: float = 0.0) -> CombatSnapshotPool:
        rows = [
            # 2 starter-early rows on 1 encounter
            _row(encounter_id="ENCOUNTER.LOUSE", floor_number=1, deck_size=10),
            _row(encounter_id="ENCOUNTER.LOUSE", floor_number=2, deck_size=11),
            # 1 starter-early row on a second encounter (different weak fight)
            _row(encounter_id="ENCOUNTER.JAW_WORM", floor_number=1, deck_size=10),
            # 4 non-starter rows on a mid-run encounter
            *[
                _row(
                    encounter_id="ENCOUNTER.SENTRIES",
                    floor_number=12,
                    deck_size=22,
                    tier_suffix="_ELITE",
                )
                for _ in range(4)
            ],
        ]
        return CombatSnapshotPool(
            rows,
            sample_mode="encounter_balanced",
            starter_early_boost=starter_early_boost,
        )

    def test_boost_zero_preserves_legacy_behaviour(self) -> None:
        pool = self._mixed_pool(starter_early_boost=0.0)
        rng = np.random.default_rng(42)
        samples = [pool.sample(rng) for _ in range(400)]
        counts = Counter(row["encounter_id"] for row in samples)
        # encounter_balanced with 3 encounters: each ~1/3. Crucially, the
        # starter-early rows don't get any special weight.
        assert counts["ENCOUNTER.SENTRIES_ELITE"] > 100
        # And obviously no routing through starter-early-only path.

    def test_boost_one_only_samples_starter_early_rows(self) -> None:
        pool = self._mixed_pool(starter_early_boost=1.0)
        rng = np.random.default_rng(7)
        samples = [pool.sample(rng) for _ in range(200)]
        for row in samples:
            assert infer_deck_stage_from_row(row) == "starter_early"
        encounter_ids = {row["encounter_id"] for row in samples}
        # Encounter-balanced over the 2 starter-early encounters → both seen.
        assert "ENCOUNTER.LOUSE_WEAK" in encounter_ids
        assert "ENCOUNTER.JAW_WORM_WEAK" in encounter_ids

    def test_partial_boost_mixes_distributions(self) -> None:
        pool = self._mixed_pool(starter_early_boost=0.5)
        rng = np.random.default_rng(123)
        samples = [pool.sample(rng) for _ in range(1000)]
        starter_frac = sum(
            1 for row in samples if infer_deck_stage_from_row(row) == "starter_early"
        ) / len(samples)
        # Lower bound: 0.5 from boost alone. Upper bound: 0.5 boost + some
        # starter-early hits from the 50% fallback to regular sampling
        # (which also has starter-early rows in 2 of 3 encounter buckets).
        assert 0.5 <= starter_frac <= 0.9

    def test_boost_rejects_pool_with_no_starter_early_rows(self) -> None:
        rows_no_starter = [
            _row(
                encounter_id="ENCOUNTER.SENTRIES",
                floor_number=12,
                deck_size=22,
                tier_suffix="_ELITE",
            )
            for _ in range(4)
        ]
        try:
            CombatSnapshotPool(
                rows_no_starter,
                sample_mode="encounter_balanced",
                starter_early_boost=0.3,
            )
        except ValueError as exc:
            assert "starter_early" in str(exc)
        else:
            raise AssertionError("expected ValueError when boost>0 but no starter rows")

    def test_boost_rejects_out_of_range(self) -> None:
        for bad in (-0.1, 1.5):
            try:
                CombatSnapshotPool(
                    [_row(encounter_id="ENCOUNTER.LOUSE", floor_number=1, deck_size=10)],
                    sample_mode="encounter_balanced",
                    starter_early_boost=bad,
                )
            except ValueError:
                continue
            raise AssertionError(f"expected ValueError for boost={bad}")

    def test_summary_reports_starter_early_coverage(self) -> None:
        pool = self._mixed_pool(starter_early_boost=0.25)
        summary = pool.summary()
        assert summary["starter_early_boost"] == 0.25
        assert summary["starter_early_row_count"] == 3
        assert summary["starter_early_encounter_count"] == 2
        assert summary["deck_stage_counts"] == {"starter_early": 3, "rest": 4}
