"""Tests for the drift-gated direct planner score (TASK-F1).

Verifies the spec acceptance shapes:

1. High latent_drift drops the gate below 1.
2. Low legal_f1 drops the gate below 1.
3. ``drift_gate=0`` makes rollout_q irrelevant for action ordering.
4. ``drift_gate=1`` lets rollout_q drive ordering as expected.
5. A large bad-end-turn ``immediate_tactical_quality`` penalty overrides a
   misleadingly high rollout_q.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


RL_AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(RL_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(RL_AGENT_ROOT))

from muzero.drift_gate import (
    DriftGateConfig,
    PlannerScoreWeights,
    compose_planner_score,
    compute_drift_gate,
)


class DriftGateScalarTests(unittest.TestCase):
    def test_clean_signals_yield_gate_near_one(self):
        gate = compute_drift_gate(legal_f1=1.0, latent_drift=0.05, q_mae=0.05, branch_disagreement=0.05)
        self.assertGreater(gate, 0.9)

    def test_high_latent_drift_lowers_gate(self):
        clean = compute_drift_gate(legal_f1=1.0, latent_drift=0.05, q_mae=0.0, branch_disagreement=0.0)
        drifted = compute_drift_gate(legal_f1=1.0, latent_drift=1.5, q_mae=0.0, branch_disagreement=0.0)
        self.assertLess(drifted, clean)
        self.assertLess(drifted, 0.5)

    def test_low_legal_f1_lowers_gate(self):
        clean = compute_drift_gate(legal_f1=0.95)
        bad = compute_drift_gate(legal_f1=0.4)
        self.assertLess(bad, clean)

    def test_high_q_mae_lowers_gate(self):
        clean = compute_drift_gate(q_mae=0.1)
        bad = compute_drift_gate(q_mae=2.5)
        self.assertLess(bad, clean)

    def test_high_disagreement_lowers_gate(self):
        clean = compute_drift_gate(branch_disagreement=0.1)
        bad = compute_drift_gate(branch_disagreement=1.0)
        self.assertLess(bad, clean)

    def test_min_gate_floor_respected(self):
        cfg = DriftGateConfig(min_gate=0.1)
        gate = compute_drift_gate(legal_f1=0.0, latent_drift=10.0, q_mae=10.0, branch_disagreement=10.0, config=cfg)
        self.assertGreaterEqual(gate, 0.1)

    def test_gate_clamped_to_one(self):
        gate = compute_drift_gate(legal_f1=10.0, latent_drift=-10.0, q_mae=-10.0, branch_disagreement=-10.0)
        self.assertLessEqual(gate, 1.0)


class PlannerScoreCompositionTests(unittest.TestCase):
    def test_drift_gate_zero_zeros_q_components(self):
        comps = compose_planner_score(
            policy_prior=0.3,
            immediate_tactical_quality=0.5,
            rollout_q=2.0,
            objective_q=2.0,
            risk_q=2.0,
            drift_gate=0.0,
        )
        self.assertEqual(comps["rollout_q"], 0.0)
        self.assertEqual(comps["objective_q"], 0.0)
        self.assertEqual(comps["risk_q"], 0.0)
        self.assertEqual(comps["effective_rollout_q_weight"], 0.0)

    def test_drift_gate_zero_makes_rollout_q_irrelevant_for_ordering(self):
        # Two candidates with identical priors + tactical quality; rollout_q
        # differs but with drift_gate=0 the score should be identical.
        a = compose_planner_score(policy_prior=0.5, immediate_tactical_quality=0.4, rollout_q=5.0, drift_gate=0.0)
        b = compose_planner_score(policy_prior=0.5, immediate_tactical_quality=0.4, rollout_q=-5.0, drift_gate=0.0)
        self.assertAlmostEqual(a["score"], b["score"])

    def test_drift_gate_one_lets_rollout_q_drive_ordering(self):
        weights = PlannerScoreWeights(rollout_q=1.0)
        a = compose_planner_score(policy_prior=0.5, immediate_tactical_quality=0.4, rollout_q=2.0, drift_gate=1.0, weights=weights)
        b = compose_planner_score(policy_prior=0.5, immediate_tactical_quality=0.4, rollout_q=0.0, drift_gate=1.0, weights=weights)
        self.assertGreater(a["score"], b["score"])

    def test_bad_end_turn_immediate_penalty_overrides_high_q(self):
        # bad_end_turn surfaces as a strong negative immediate_tactical_quality
        # — it MUST dominate even when rollout_q is misleadingly high and the
        # gate is fully open.
        weights = PlannerScoreWeights(immediate_tactical=2.0, rollout_q=1.0)
        bad_end_turn = compose_planner_score(
            policy_prior=0.1,
            immediate_tactical_quality=-5.0,
            rollout_q=2.0,
            drift_gate=1.0,
            weights=weights,
        )
        safe_block = compose_planner_score(
            policy_prior=0.1,
            immediate_tactical_quality=0.5,
            rollout_q=0.5,
            drift_gate=1.0,
            weights=weights,
        )
        self.assertLess(bad_end_turn["score"], safe_block["score"])

    def test_legality_guard_passes_through(self):
        comps = compose_planner_score(
            policy_prior=0.0,
            immediate_tactical_quality=0.0,
            legality_guard=-1e6,
        )
        self.assertLessEqual(comps["score"], -1e6 + 1e-3)

    def test_uncertainty_penalty_subtracts(self):
        with_pen = compose_planner_score(policy_prior=0.5, immediate_tactical_quality=0.0, uncertainty_penalty=1.0)
        without_pen = compose_planner_score(policy_prior=0.5, immediate_tactical_quality=0.0, uncertainty_penalty=0.0)
        self.assertLess(with_pen["score"], without_pen["score"])


if __name__ == "__main__":
    unittest.main()
