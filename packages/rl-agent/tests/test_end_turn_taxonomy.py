"""Tests for the end_turn taxonomy (TASK-B1).

Verifies the central ``_classify_end_turn_action`` priority chain
(``bad > forced(transient) > forced > strategic_defer > unknown``) and the
boss-window override rule that upgrades end_turn to ``bad_end_turn`` when
Kaiser back-attack risk + facing candidate or Ceremonial stun-window are open.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


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
    }


class TaxonomyPriorityTests(unittest.TestCase):
    def test_energy_and_defend_with_incoming_marks_bad(self):
        # energy=2, hand has Defend, incoming threatens player → bad.
        ctx = _ctx(
            wasteful=True,
            energy=2.0,
            positive_progress_count=2,
            urgent_positive_count=1,
        )
        cls, _ = MuZeroTrainer._classify_end_turn_action(ctx)
        self.assertEqual(cls, "bad_end_turn")

    def test_no_actions_marks_forced(self):
        ctx = _ctx(positive_progress_count=0, energy=0.0)
        cls, flags = MuZeroTrainer._classify_end_turn_action(ctx)
        self.assertEqual(cls, "forced_end_turn")
        self.assertTrue(flags["no_legal_positive_action"])

    def test_low_value_exhaust_only_marks_strategic_defer(self):
        # No urgent, only deferable exhaust resource cards → strategic defer.
        ctx = _ctx(
            wasteful=False,
            strategic_defer_available=True,
            energy=1.0,
            positive_progress_count=2,
            urgent_positive_count=0,
            deferable_positive_count=2,
        )
        cls, _ = MuZeroTrainer._classify_end_turn_action(ctx)
        self.assertEqual(cls, "strategic_defer_end_turn")


class BossWindowOverrideTests(unittest.TestCase):
    def test_kaiser_back_attack_with_facing_candidate_marks_bad(self):
        # Even when the static "wasteful" detector is False (no urgent), end_turn
        # under Kaiser back-attack risk with a facing-change candidate available
        # is a bad call — the boss window override must escalate to bad_end_turn.
        ctx = _ctx(positive_progress_count=2, urgent_positive_count=0, deferable_positive_count=2,
                   strategic_defer_available=True, energy=1.0)
        boss = {
            "kaiser_back_attack_risk": 0.6,
            "kaiser_facing_change_candidate_count": 1.0,
        }
        cls, flags = MuZeroTrainer._classify_end_turn_action(ctx, None, boss)
        self.assertEqual(cls, "bad_end_turn")
        self.assertTrue(flags["kaiser_pressure_window_open"])

    def test_ceremonial_stun_window_with_high_impact_marks_bad(self):
        ctx = _ctx(positive_progress_count=1, urgent_positive_count=0, deferable_positive_count=1,
                   strategic_defer_available=True, energy=1.0)
        boss = {
            "ceremonial_stun_window": 0.7,
            "ceremonial_high_impact_count": 1.0,
        }
        cls, _ = MuZeroTrainer._classify_end_turn_action(ctx, None, boss)
        self.assertEqual(cls, "bad_end_turn")

    def test_no_facing_candidate_does_not_override(self):
        # Risk exists but no facing change actually playable — fallback to the
        # static defer judgment, do NOT mislabel as bad.
        ctx = _ctx(positive_progress_count=2, urgent_positive_count=0, deferable_positive_count=2,
                   strategic_defer_available=True, energy=1.0)
        boss = {
            "kaiser_back_attack_risk": 0.6,
            "kaiser_facing_change_candidate_count": 0.0,
            "kaiser_defense_candidate_count": 0.0,
        }
        cls, _ = MuZeroTrainer._classify_end_turn_action(ctx, None, boss)
        self.assertEqual(cls, "strategic_defer_end_turn")


class TransientHandlingTests(unittest.TestCase):
    def test_transient_only_end_turn_returns_forced_not_bad(self):
        # Bridge handed us only end_turn this transient frame.  Even with energy
        # and positive heuristic, taxonomy must mark this as forced (transient
        # flag set) so the JSONL post-mortem can isolate it.
        ctx = _ctx(positive_progress_count=2, urgent_positive_count=1, energy=2.0, wasteful=True)
        cls, flags = MuZeroTrainer._classify_end_turn_action(
            ctx,
            action_diagnostics={"transient_only_end_turn": True},
        )
        self.assertEqual(cls, "forced_end_turn")
        self.assertTrue(flags["transient_only_end_turn"])

    def test_transient_short_circuits_boss_override(self):
        ctx = _ctx(positive_progress_count=2, urgent_positive_count=0, energy=1.0)
        cls, _ = MuZeroTrainer._classify_end_turn_action(
            ctx,
            action_diagnostics={"transient_only_end_turn": True},
            boss_signals={
                "kaiser_back_attack_risk": 0.8,
                "kaiser_facing_change_candidate_count": 2.0,
            },
        )
        self.assertEqual(cls, "forced_end_turn")


class WastefulAliasTests(unittest.TestCase):
    def test_legacy_wasteful_alias_still_flagged(self):
        # The legacy wasteful detector → new bad taxonomy must keep the same
        # frames flagged so existing dashboards don't go dark.
        ctx = _ctx(wasteful=True, positive_progress_count=2, urgent_positive_count=1, energy=2.0)
        cls, _ = MuZeroTrainer._classify_end_turn_action(ctx)
        self.assertEqual(cls, "bad_end_turn")


if __name__ == "__main__":
    unittest.main()
