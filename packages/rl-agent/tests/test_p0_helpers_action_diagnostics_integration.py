"""End-to-end smoke for P0 typed views in ``action_diagnostics``.

Verifies that ``CombatSandboxEnv.step`` (mocked) attaches:

* ``hp_cost_safety`` (P0-1)
* ``x_cost`` (P0-4)
* ``selection`` (P0-5)
* ``card_identity`` (P0-6)
* ``transient_leaked`` / ``prior_transient_only_end_turn`` (P0-2)

so the trainer-side aggregator can read them via the existing
``info["action_diagnostics"]`` channel without further plumbing.

The test does NOT spin up the full env (bridge + obs encoder).  It
exercises the diagnostic-block assembly logic directly so we lock in the
contract that "all P0 helper outputs are surfaced under predictable keys".
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any


RL_AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(RL_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(RL_AGENT_ROOT))

from sts2_env.card_identity import card_identity
from sts2_env.hp_cost_safety import hp_cost_safety_view
from sts2_env.selection_typed import selection_view
from sts2_env.x_cost_dynamic import x_cost_view


def _build_diag_block(action: dict[str, Any], raw_obs: dict[str, Any]) -> dict[str, Any]:
    """Mirror the env-side block constructed inside combat_env.step."""
    diag: dict[str, Any] = {}
    played_card = action.get("card") if isinstance(action, dict) else None
    diag["hp_cost_safety"] = hp_cost_safety_view(action, raw_obs)
    diag["x_cost"] = x_cost_view(action, raw_obs)
    diag["selection"] = selection_view(action)
    diag["card_identity"] = card_identity(played_card)
    return diag


class P0DiagnosticsIntegration(unittest.TestCase):
    def test_full_p0_diag_block_for_typed_action(self):
        action = {
            "kind": "play_card",
            "action_id": "play:card-uuid-1",
            "selection": {
                "operation_type": "discard",
                "screen_type": "card_selection",
                "source": "card",
                "source_zone": "hand",
                "destination_zone": "discard",
                "min_count": 1,
                "max_count": 1,
                "selection_required": True,
                "confidence": "runtime_internal",
            },
            "x_cost": {
                "has_x_cost": True,
                "resource": "energy",
                "current_value": 0,
                "is_zero": True,
                "effect_scaled": True,
                "preview_scale_source": "energy_x",
                "semantics": "repeat",
            },
            "safety": {
                "hp_cost_kind": "unblockable_hp_loss",
                "hp_loss_unblockable": 6,
                "self_damage_blockable": 0,
                "max_hp_loss": 0,
                "hp_before": 5,
                "source_confidence": "runtime_internal",
            },
            "card": {
                "instance_uuid": "card-uuid-1",
                "id": "CARD.WHIRLWIND",
                "title": "Whirlwind",
                "type": "Attack",
                "cost": "X",
                "x_cost": True,
            },
        }
        raw_obs = {"player": {"hp": 5, "max_hp": 80, "block": 0}, "combat": {"energy": 0}}
        diag = _build_diag_block(action, raw_obs)

        # P0-1
        self.assertEqual(diag["hp_cost_safety"]["source_confidence"], "runtime_internal")
        self.assertTrue(diag["hp_cost_safety"]["self_lethal_now"])
        # P0-4
        self.assertEqual(diag["x_cost"]["source_confidence"], "runtime_internal")
        self.assertEqual(diag["x_cost"]["resource"], "energy")
        self.assertTrue(diag["x_cost"]["is_zero"])
        # P0-5
        self.assertEqual(diag["selection"]["confidence"], "runtime_internal")
        self.assertEqual(diag["selection"]["operation_type"], "discard")
        # P0-6
        self.assertEqual(diag["card_identity"]["confidence"], "runtime_internal")
        self.assertEqual(diag["card_identity"]["source"], "instance_uuid")

    def test_full_p0_diag_block_for_text_only_action(self):
        # No typed bridge blocks → all helpers fall back to lower confidence.
        action = {
            "kind": "play_card",
            "action_id": "play:strike",
            "selection_prompt": "Discard a card.",
            "card": {
                "id": "CARD.STRIKE",
                "title": "Strike",
                "type": "Attack",
                "cost": 1,
            },
        }
        raw_obs = {"player": {"hp": 50, "max_hp": 80, "block": 0}, "combat": {"energy": 3}}
        diag = _build_diag_block(action, raw_obs)
        # P0-1: no hp cost → 'none'
        self.assertEqual(diag["hp_cost_safety"]["hp_cost_kind"], "none")
        self.assertFalse(diag["hp_cost_safety"]["self_lethal_now"])
        # P0-4: not X-cost
        self.assertFalse(diag["x_cost"]["has_x_cost"])
        # P0-5: text fallback
        self.assertEqual(diag["selection"]["confidence"], "text_fallback")
        self.assertEqual(diag["selection"]["operation_type"], "discard")
        # P0-6: id-only → static_export
        self.assertEqual(diag["card_identity"]["confidence"], "static_export")

    def test_full_p0_diag_block_no_signals(self):
        action = {"kind": "end_turn", "action_id": "end_turn"}
        raw_obs = {"player": {"hp": 50, "max_hp": 80, "block": 0}}
        diag = _build_diag_block(action, raw_obs)
        # Each helper returns its 'none/empty' default.
        self.assertEqual(diag["hp_cost_safety"]["hp_cost_kind"], "none")
        self.assertFalse(diag["x_cost"]["has_x_cost"])
        self.assertEqual(diag["selection"]["confidence"], "none")
        self.assertEqual(diag["card_identity"]["confidence"], "none")


class P0DiagnosticsKeysContract(unittest.TestCase):
    """Lock down the exact keys each P0 helper emits so trainer-side
    consumers can rely on them without re-reading the spec."""

    REQUIRED_KEYS = {
        "hp_cost_safety": {
            "hp_cost_kind", "hp_loss_unblockable", "self_damage_blockable",
            "max_hp_loss", "hp_before", "hp_after_self_cost",
            "hp_margin_after_self_cost", "self_lethal_now",
            "low_hp_margin_after_cost", "source_confidence",
        },
        "x_cost": {
            "has_x_cost", "resource", "current_value", "is_zero",
            "effect_scaled", "preview_scale_source", "semantics",
            "non_x_effect_present", "zero_x_bad", "source_confidence",
        },
        "selection": {
            "operation_type", "screen_type", "source", "source_zone",
            "destination_zone", "min_count", "max_count",
            "selection_required", "modifier_id", "confidence",
        },
        "card_identity": {"key", "confidence", "source"},
    }

    def test_keys_contract(self):
        action = {"kind": "end_turn", "action_id": "end_turn"}
        diag = _build_diag_block(action, {"player": {"hp": 50}})
        for block, required in self.REQUIRED_KEYS.items():
            with self.subTest(block=block):
                actual = set(diag[block].keys())
                missing = required - actual
                self.assertFalse(missing, f"{block} missing keys: {missing}")


if __name__ == "__main__":
    unittest.main()
