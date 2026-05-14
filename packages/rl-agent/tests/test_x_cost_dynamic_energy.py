"""Tests for X-cost dynamic-energy diagnostics (TASK-A3).

Verifies that ``_x_cost_diagnostic`` reads current energy and the card's
non-energy-effect flag rather than relying on a static initial 3-energy
assumption.  Also confirms ``semantic_action.signature`` exposes ``base_cost``
and ``x_cost_has_non_energy_effect`` so the trainer-side detector can read them
directly.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


RL_AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(RL_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(RL_AGENT_ROOT))

from muzero.train import MuZeroTrainer
from sts2_env.semantic_action import semantic_action_signature


def _action(*, semantic: dict | None = None, card: dict | None = None) -> dict:
    return {
        "kind": "play_card",
        "semantic": semantic or {},
        "card": card or {},
    }


class XCostDiagnosticTests(unittest.TestCase):
    def test_zero_energy_no_non_energy_effect_is_bad(self):
        action = _action(
            semantic={"is_x_cost": True, "x_cost_value": 0.0},
            card={"x_cost": True, "id": "WHIRLWIND"},
        )
        diag = MuZeroTrainer._x_cost_diagnostic(action, current_energy=0.0)
        self.assertEqual(diag["is_x_cost"], 1.0)
        self.assertEqual(diag["x_cost_effective_energy"], 0.0)
        self.assertEqual(diag["x_cost_has_non_energy_effect"], 0.0)
        self.assertEqual(diag["x_cost_bad"], 1.0)

    def test_two_energy_x_cost_effective_energy_two(self):
        action = _action(
            semantic={"is_x_cost": True, "x_cost_value": 2.0},
            card={"x_cost": True, "id": "WHIRLWIND"},
        )
        diag = MuZeroTrainer._x_cost_diagnostic(action, current_energy=2.0)
        self.assertEqual(diag["is_x_cost"], 1.0)
        self.assertEqual(diag["x_cost_effective_energy"], 2.0)
        self.assertEqual(diag["x_cost_bad"], 0.0)

    def test_zero_energy_with_non_energy_effect_not_bad(self):
        # X-cost card that exhausts/transforms hand cards regardless of energy.
        action = _action(
            semantic={
                "is_x_cost": True,
                "x_cost_has_non_energy_effect": True,
                "typed_exhaust_cards": True,
            },
            card={"x_cost": True, "id": "PURITY"},
        )
        diag = MuZeroTrainer._x_cost_diagnostic(action, current_energy=0.0)
        self.assertEqual(diag["is_x_cost"], 1.0)
        self.assertEqual(diag["x_cost_has_non_energy_effect"], 1.0)
        self.assertEqual(diag["x_cost_bad"], 0.0)

    def test_non_x_cost_returns_zeros(self):
        action = _action(
            semantic={"is_x_cost": False},
            card={"id": "STRIKE", "cost": 1},
        )
        diag = MuZeroTrainer._x_cost_diagnostic(action, current_energy=3.0)
        self.assertEqual(diag["is_x_cost"], 0.0)
        self.assertEqual(diag["x_cost_bad"], 0.0)

    def test_string_x_cost_detected(self):
        # Some bridge payloads send cost as the literal string "X" rather than a
        # numeric -1 or x_cost flag.  The detector must still recognise it.
        action = {
            "kind": "play_card",
            "card_cost": "X",
            "semantic": {},
            "card": {"id": "MAYHEM", "cost": "X"},
        }
        diag = MuZeroTrainer._x_cost_diagnostic(action, current_energy=1.0)
        self.assertEqual(diag["is_x_cost"], 1.0)
        self.assertEqual(diag["x_cost_effective_energy"], 1.0)

    def test_semantic_role_only_x_cost_detected(self):
        # Live compact-action diagnostics may expose the X-cost fact only via
        # semantic roles.  The hard guard must use the same contract as the
        # selected-action metrics, otherwise zero-energy X plays are logged as
        # offenders while the guard never becomes available.
        action = {
            "kind": "play_card",
            "title": "Whirlwind+",
            "semantic": {"roles": ["attack", "aoe", "x_cost"]},
            "card": {"id": "WHIRLWIND", "title": "Whirlwind+", "cost": 4},
        }
        diag = MuZeroTrainer._x_cost_diagnostic(action, current_energy=0.0)
        self.assertEqual(diag["is_x_cost"], 1.0)
        self.assertEqual(diag["x_cost_effective_energy"], 0.0)
        self.assertEqual(diag["x_cost_bad"], 1.0)


class SemanticSignatureTests(unittest.TestCase):
    def test_signature_exposes_base_cost_and_non_energy_flag_for_x_cost(self):
        bridge_action = {
            "kind": "play_card",
            "title": "Whirlwind",
            "card": {
                "id": "WHIRLWIND",
                "title": "Whirlwind",
                "type": "Attack",
                "cost": "X",
                "x_cost": True,
                "damage": 5,
                "x_cost_value": 0,
            },
        }
        sig = semantic_action_signature(bridge_action)
        self.assertTrue(sig.get("is_x_cost"))
        self.assertEqual(float(sig.get("base_cost")), -1.0)
        # Whirlwind has no structural side-effect tags, so non-energy-effect is False.
        self.assertFalse(bool(sig.get("x_cost_has_non_energy_effect")))

    def test_signature_base_cost_is_static_int_for_normal_cards(self):
        bridge_action = {
            "kind": "play_card",
            "title": "Strike",
            "card": {"id": "STRIKE", "title": "Strike", "type": "Attack", "cost": 1, "damage": 6},
        }
        sig = semantic_action_signature(bridge_action)
        self.assertFalse(bool(sig.get("is_x_cost")))
        self.assertEqual(float(sig.get("base_cost")), 1.0)


if __name__ == "__main__":
    unittest.main()
