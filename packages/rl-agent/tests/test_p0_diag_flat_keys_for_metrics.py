"""Lock down the flat diagnostic keys consumed by the trainer-side
``diag_key_map`` → ``combat_quality_*`` aggregation pipeline.

When the env-side wiring (``combat_env.step``) builds an
``action_diagnostics`` block, it MUST include scalar floats for the
P0 hardening hooks so they flow through the existing rolling-window
TB metric path without any additional trainer wiring.
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
from muzero.training.action_diagnostics_merge import ACTION_DIAGNOSTIC_STAT_KEYS


P0_FLAT_KEYS = {
    # Hooks the trainer-side diag_key_map expects to read.
    "hp_cost_self_lethal_selected",
    "hp_cost_low_margin_selected",
    "hp_cost_unblockable_value",
    "x_cost_selected",
    "x_cost_zero_bad_selected",
    "x_cost_zero_selected",
    "x_cost_energy_value",
    "x_cost_star_value",
    "star_x_selected",
    "selection_text_fallback_selected",
    "selection_runtime_internal_selected",
    "card_identity_text_fallback_selected",
    "card_identity_runtime_internal_selected",
}


def _build_flat_diag(action: dict[str, Any], raw_obs: dict[str, Any]) -> dict[str, Any]:
    """Mirror exactly what combat_env.step inserts into action_diagnostics."""
    played_card = action.get("card") if isinstance(action, dict) else None
    hp_safety = hp_cost_safety_view(action, raw_obs)
    xcost = x_cost_view(action, raw_obs)
    sel = selection_view(action)
    ident = card_identity(played_card)
    diag: dict[str, Any] = {}
    diag["hp_cost_self_lethal_selected"] = 1.0 if hp_safety.get("self_lethal_now") else 0.0
    diag["hp_cost_low_margin_selected"] = 1.0 if hp_safety.get("low_hp_margin_after_cost") else 0.0
    diag["hp_cost_unblockable_value"] = float(hp_safety.get("hp_loss_unblockable") or 0.0)
    xres = str(xcost.get("resource") or "none")
    xcv = float(xcost.get("current_value") or 0.0)
    diag["x_cost_selected"] = 1.0 if xcost.get("has_x_cost") else 0.0
    diag["x_cost_zero_bad_selected"] = 1.0 if xcost.get("zero_x_bad") else 0.0
    diag["x_cost_zero_selected"] = 1.0 if (xcost.get("is_zero") and xcost.get("has_x_cost")) else 0.0
    diag["x_cost_energy_value"] = xcv if xres == "energy" else 0.0
    diag["x_cost_star_value"] = xcv if xres == "stars" else 0.0
    diag["star_x_selected"] = 1.0 if (xcost.get("has_x_cost") and xres == "stars") else 0.0
    diag["selection_text_fallback_selected"] = 1.0 if sel.get("confidence") == "text_fallback" else 0.0
    diag["selection_runtime_internal_selected"] = 1.0 if sel.get("confidence") == "runtime_internal" else 0.0
    diag["card_identity_text_fallback_selected"] = 1.0 if ident.get("confidence") == "text_fallback" else 0.0
    diag["card_identity_runtime_internal_selected"] = 1.0 if ident.get("confidence") == "runtime_internal" else 0.0
    return diag


class FlatDiagKeysContract(unittest.TestCase):
    def test_all_p0_flat_keys_present_for_any_action(self):
        """Every action must produce ALL P0 flat keys (zero or one)."""
        action = {"kind": "end_turn", "action_id": "end_turn"}
        diag = _build_flat_diag(action, {"player": {"hp": 50, "max_hp": 80, "block": 0}})
        missing = P0_FLAT_KEYS - set(diag.keys())
        self.assertFalse(missing, f"missing flat diag keys: {missing}")
        # All values must be numeric floats.
        for key in P0_FLAT_KEYS:
            self.assertIsInstance(diag[key], float, f"{key} not float: {type(diag[key])}")

    def test_lethal_hp_cost_flips_self_lethal_metric(self):
        action = {
            "kind": "play_card",
            "action_id": "play:hpcost",
            "card": {"id": "X", "title": "X", "type": "Skill",
                     "effect_preview": {"hp_loss": 5},
                     "semantic_signals": {"self_damage": 5}},
        }
        diag = _build_flat_diag(action, {"player": {"hp": 3, "max_hp": 80, "block": 99}})
        self.assertEqual(diag["hp_cost_self_lethal_selected"], 1.0)
        # block does NOT save the unblockable cost
        self.assertEqual(diag["hp_cost_unblockable_value"], 5.0)

    def test_zero_x_bad_metric_set(self):
        action = {
            "kind": "play_card",
            "action_id": "play:xcost",
            "card": {"id": "WHIRLWIND", "title": "Whirlwind", "type": "Attack",
                     "cost": "X", "x_cost": True,
                     "card_effect_profile": {"operations": [], "semantic_tags": [], "training_tags": []}},
        }
        diag = _build_flat_diag(action, {"player": {"hp": 50}, "combat": {"energy": 0}})
        self.assertEqual(diag["x_cost_selected"], 1.0)
        self.assertEqual(diag["x_cost_zero_bad_selected"], 1.0)
        self.assertEqual(diag["x_cost_zero_selected"], 1.0)
        self.assertEqual(diag["x_cost_energy_value"], 0.0)
        self.assertEqual(diag["star_x_selected"], 0.0)

    def test_star_x_resource_routing(self):
        action = {
            "kind": "play_card",
            "action_id": "play:starx",
            "card": {"id": "STAR", "title": "Star", "type": "Skill",
                     "cost": 1, "has_star_cost_x": True,
                     "card_effect_profile": {"operations": [], "semantic_tags": [], "training_tags": []}},
        }
        diag = _build_flat_diag(action, {"player": {"hp": 50}, "combat": {"stars": 2}})
        self.assertEqual(diag["x_cost_selected"], 1.0)
        self.assertEqual(diag["star_x_selected"], 1.0)
        self.assertEqual(diag["x_cost_star_value"], 2.0)
        self.assertEqual(diag["x_cost_energy_value"], 0.0)

    def test_text_fallback_selection_visible(self):
        action = {
            "kind": "play_card",
            "action_id": "play:text",
            "selection_prompt": "Discard a card.",
            "card": {"id": "X", "title": "X"},
        }
        diag = _build_flat_diag(action, {"player": {"hp": 50}})
        self.assertEqual(diag["selection_text_fallback_selected"], 1.0)
        self.assertEqual(diag["selection_runtime_internal_selected"], 0.0)

    def test_runtime_internal_identity_visible(self):
        action = {
            "kind": "play_card",
            "action_id": "play:uuid",
            "card": {"instance_uuid": "uuid-xyz", "id": "X", "title": "X"},
        }
        diag = _build_flat_diag(action, {"player": {"hp": 50}})
        self.assertEqual(diag["card_identity_runtime_internal_selected"], 1.0)
        self.assertEqual(diag["card_identity_text_fallback_selected"], 0.0)


class TrainerDiagKeyMapAlignment(unittest.TestCase):
    """Verify the trainer-side action diagnostic merge knows every P0 flat key."""

    def test_diag_key_map_contains_p0_keys(self):
        for key in P0_FLAT_KEYS:
            with self.subTest(key=key):
                self.assertIn(
                    key,
                    ACTION_DIAGNOSTIC_STAT_KEYS,
                    f"ACTION_DIAGNOSTIC_STAT_KEYS missing entry for '{key}'",
                )
        # transient_leaked has its own flat key from P0-2.
        self.assertIn("transient_leaked", ACTION_DIAGNOSTIC_STAT_KEYS)


if __name__ == "__main__":
    unittest.main()
