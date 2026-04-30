"""Tests for the encounter-balanced replay scheduler (TASK-F2)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


RL_AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(RL_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(RL_AGENT_ROOT))

from muzero.replay_scheduler import (
    ReplaySample,
    ReplaySchedulerConfig,
    compute_encounter_underrepresented_weight,
    compute_freshness_weight,
    compute_offender_weight,
    compute_sample_weights,
    compute_tier_weights,
)


class TierWeightTests(unittest.TestCase):
    def test_oversampled_boss_gets_lower_weight(self):
        weights = compute_tier_weights({"boss": 800, "elite": 100, "normal": 50, "weak": 50})
        self.assertLess(weights["boss"], 1.0)
        self.assertGreater(weights["elite"], 1.0)
        self.assertGreater(weights["normal"], 1.0)

    def test_starved_tier_gets_high_weight(self):
        weights = compute_tier_weights({"boss": 0, "elite": 200, "normal": 800, "weak": 0})
        self.assertGreater(weights["boss"], 1.0)
        # Normal is heavily oversampled here (0.8 share vs 0.10–0.20 target band) →
        # weight should fall well below 1.0.
        self.assertLess(weights["normal"], 0.6)

    def test_balanced_distribution_yields_neutral_weights(self):
        weights = compute_tier_weights({"boss": 550, "elite": 250, "normal": 150, "weak": 50})
        self.assertGreater(weights["boss"], 0.8)
        self.assertLess(weights["boss"], 1.5)
        self.assertGreater(weights["elite"], 0.8)
        self.assertLess(weights["elite"], 1.5)


class EncounterUnderrepresentedTests(unittest.TestCase):
    def test_underrepresented_encounter_boosted(self):
        boost = compute_encounter_underrepresented_weight(encounter_count=2, encounter_target=20)
        self.assertGreater(boost, 1.4)

    def test_well_represented_encounter_unchanged(self):
        boost = compute_encounter_underrepresented_weight(encounter_count=25, encounter_target=20)
        self.assertEqual(boost, 1.0)

    def test_boost_capped_at_max(self):
        cfg = ReplaySchedulerConfig(encounter_underrepresented_max_boost=2.5)
        boost = compute_encounter_underrepresented_weight(encounter_count=0, encounter_target=20, config=cfg)
        self.assertLessEqual(boost, 2.5)


class OffenderAndFreshnessTests(unittest.TestCase):
    def test_offender_score_zero_no_boost(self):
        self.assertEqual(compute_offender_weight(0.0), 1.0)

    def test_offender_score_one_max_boost(self):
        cfg = ReplaySchedulerConfig(offender_max_boost=2.0)
        self.assertAlmostEqual(compute_offender_weight(1.0, config=cfg), 2.0)

    def test_offender_clamped_to_unit_range(self):
        cfg = ReplaySchedulerConfig(offender_max_boost=2.0)
        self.assertAlmostEqual(compute_offender_weight(5.0, config=cfg), 2.0)
        self.assertAlmostEqual(compute_offender_weight(-1.0, config=cfg), 1.0)

    def test_freshness_decays_with_age(self):
        fresh = compute_freshness_weight(sample_age_steps=0)
        old = compute_freshness_weight(sample_age_steps=200_000)
        self.assertGreater(fresh, old)
        self.assertGreaterEqual(fresh, 1.0)
        self.assertGreater(old, 0.5)
        self.assertLess(old, 0.7)


class SampleWeightCompositionTests(unittest.TestCase):
    def _samples(self):
        return [
            ReplaySample(tier="boss", encounter_id="kaiser_crab_boss", sample_age_steps=0),
            ReplaySample(tier="boss", encounter_id="rare_boss", sample_age_steps=0),
            ReplaySample(tier="elite", encounter_id="elite_a", sample_age_steps=0),
            ReplaySample(tier="normal", encounter_id="normal_a", sample_age_steps=0),
        ]

    def test_normal_and_elite_floor_preserved_under_boss_oversample(self):
        # Heavy boss oversampling.  Normal/elite must still receive a >= min_weight
        # weight so they are not starved.
        weights = compute_sample_weights(
            self._samples(),
            counts_per_tier={"boss": 900, "elite": 50, "normal": 50, "weak": 0},
            counts_per_encounter={"kaiser_crab_boss": 800, "rare_boss": 100, "elite_a": 50, "normal_a": 50},
            encounter_target=200,
        )
        boss_kaiser, boss_rare, elite_w, normal_w = weights
        cfg = ReplaySchedulerConfig()
        self.assertGreaterEqual(elite_w, cfg.min_weight)
        self.assertGreaterEqual(normal_w, cfg.min_weight)
        # Elite/normal are starved relative to target → weight should rise above 1.
        self.assertGreater(elite_w, 1.0)
        self.assertGreater(normal_w, 1.0)
        # The under-represented boss gets a stronger boost than the oversampled one.
        self.assertGreater(boss_rare, boss_kaiser)

    def test_underrepresented_boss_boosted_within_max(self):
        cfg = ReplaySchedulerConfig(max_weight=4.0, encounter_underrepresented_max_boost=2.5)
        weights = compute_sample_weights(
            self._samples(),
            counts_per_tier={"boss": 500, "elite": 200, "normal": 200, "weak": 0},
            counts_per_encounter={"kaiser_crab_boss": 450, "rare_boss": 5, "elite_a": 200, "normal_a": 200},
            encounter_target=50,
            config=cfg,
        )
        boss_kaiser, boss_rare, _, _ = weights
        self.assertGreater(boss_rare, boss_kaiser)
        for w in weights:
            self.assertLessEqual(w, cfg.max_weight)
            self.assertGreaterEqual(w, cfg.min_weight)

    def test_offender_bias_bounded(self):
        cfg = ReplaySchedulerConfig(offender_max_boost=2.0, max_weight=5.0)
        offending = ReplaySample(tier="normal", encounter_id="x", offender_score=1.0)
        clean = ReplaySample(tier="normal", encounter_id="x", offender_score=0.0)
        offending_w, clean_w = compute_sample_weights(
            [offending, clean],
            counts_per_tier={"boss": 100, "elite": 100, "normal": 100, "weak": 0},
            counts_per_encounter={"x": 100},
            config=cfg,
        )
        self.assertGreater(offending_w, clean_w)
        # offender_score=1 doubles base weight at most; with everything else
        # neutral the ratio should be ≈ offender_max_boost.
        self.assertLessEqual(offending_w / max(clean_w, 1e-9), cfg.offender_max_boost + 1e-6)

    def test_demo_sample_gets_demo_boost(self):
        cfg = ReplaySchedulerConfig(demo_max_boost=1.5)
        demo = ReplaySample(tier="boss", encounter_id="x", is_demo=True)
        rl = ReplaySample(tier="boss", encounter_id="x", is_demo=False)
        weights = compute_sample_weights(
            [demo, rl],
            counts_per_tier={"boss": 500, "elite": 250, "normal": 200, "weak": 50},
            counts_per_encounter={"x": 100},
            config=cfg,
        )
        self.assertGreater(weights[0], weights[1])

    def test_min_weight_floor_holds(self):
        cfg = ReplaySchedulerConfig(min_weight=0.05)
        weights = compute_sample_weights(
            [ReplaySample(tier="boss", encounter_id="x", sample_age_steps=10_000_000)],
            counts_per_tier={"boss": 999_999, "elite": 1, "normal": 1, "weak": 0},
            counts_per_encounter={"x": 999_999},
            config=cfg,
        )
        self.assertGreaterEqual(weights[0], cfg.min_weight)


if __name__ == "__main__":
    unittest.main()
