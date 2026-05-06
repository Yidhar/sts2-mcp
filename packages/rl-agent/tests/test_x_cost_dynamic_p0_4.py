"""Tests for the X-cost / Star-X dynamic preview helper (P0-4)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any


RL_AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(RL_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(RL_AGENT_ROOT))

from sts2_env.x_cost_dynamic import (
    is_x_cost_action,
    is_zero_x_bad,
    x_cost_resource,
    x_cost_view,
)


def _energy_x_card(*, x_cost: bool = True, has_non_x_effect: bool = False) -> dict[str, Any]:
    operations = []
    if has_non_x_effect:
        operations.append({"op": "add_modifier", "source_zone": "hand"})
    return {
        "id": "CARD.WHIRLWIND",
        "title": "Whirlwind",
        "type": "Attack",
        "cost": "X" if x_cost else 0,
        "x_cost": x_cost,
        "card_effect_profile": {
            "operations": operations,
            "semantic_tags": ["x_cost"] if x_cost else [],
            "training_tags": [],
        },
    }


def _star_x_card(*, has_non_x_effect: bool = False) -> dict[str, Any]:
    operations = []
    if has_non_x_effect:
        operations.append({"op": "exhaust_card", "source_zone": "hand"})
    return {
        "id": "CARD.STAR_X",
        "title": "StarThing",
        "type": "Skill",
        "cost": 1,
        "has_star_cost_x": True,
        "card_effect_profile": {
            "operations": operations,
            "semantic_tags": [],
            "training_tags": [],
        },
    }


def _typed_x_card(**typed_overrides: Any) -> dict[str, Any]:
    typed = {
        "has_x_cost": True,
        "resource": "energy",
        "current_value": 0,
        "is_zero": True,
        "effect_scaled": True,
        "preview_scale_source": "energy_x",
        "semantics": "repeat",
    }
    typed.update(typed_overrides)
    return {
        "kind": "play_card",
        "action_id": "play:typed_x",
        "card": _energy_x_card(),
        "x_cost": typed,
    }


def _action(card: dict[str, Any]) -> dict[str, Any]:
    return {"kind": "play_card", "action_id": "play:test", "card": card}


def _obs(*, energy: int = 0, stars: int = 0) -> dict[str, Any]:
    return {
        "combat": {"energy": energy, "stars": stars},
        "player": {"hp": 50, "max_hp": 80, "block": 0},
    }


class EnergyXFallbackTests(unittest.TestCase):
    def test_x_cost_card_at_zero_energy_zero_x_bad(self):
        view = x_cost_view(_action(_energy_x_card()), _obs(energy=0))
        self.assertTrue(view["has_x_cost"])
        self.assertEqual(view["resource"], "energy")
        self.assertEqual(view["current_value"], 0)
        self.assertTrue(view["is_zero"])
        self.assertFalse(view["non_x_effect_present"])
        self.assertTrue(view["zero_x_bad"])

    def test_x_cost_card_at_three_energy_not_bad(self):
        view = x_cost_view(_action(_energy_x_card()), _obs(energy=3))
        self.assertEqual(view["current_value"], 3)
        self.assertFalse(view["is_zero"])
        self.assertFalse(view["zero_x_bad"])

    def test_x_cost_card_with_non_x_effect_at_zero_not_bad(self):
        # 0 energy X but card also adds a modifier — non-X effect protects.
        view = x_cost_view(
            _action(_energy_x_card(has_non_x_effect=True)),
            _obs(energy=0),
        )
        self.assertTrue(view["is_zero"])
        self.assertTrue(view["non_x_effect_present"])
        self.assertFalse(view["zero_x_bad"])

    def test_non_x_card_has_no_x_cost_classification(self):
        plain = {
            "id": "CARD.STRIKE",
            "title": "Strike",
            "type": "Attack",
            "cost": 1,
        }
        view = x_cost_view(_action(plain), _obs(energy=3))
        self.assertFalse(view["has_x_cost"])
        self.assertEqual(view["resource"], "none")
        self.assertFalse(view["zero_x_bad"])


class StarXFallbackTests(unittest.TestCase):
    def test_star_x_at_zero_stars_zero_x_bad(self):
        view = x_cost_view(_action(_star_x_card()), _obs(stars=0))
        self.assertTrue(view["has_x_cost"])
        self.assertEqual(view["resource"], "stars")
        self.assertEqual(view["preview_scale_source"], "star_x")
        self.assertTrue(view["zero_x_bad"])

    def test_star_x_at_three_stars_not_bad(self):
        view = x_cost_view(_action(_star_x_card()), _obs(stars=3))
        self.assertEqual(view["current_value"], 3)
        self.assertFalse(view["is_zero"])
        self.assertFalse(view["zero_x_bad"])

    def test_star_x_with_non_x_effect_at_zero_not_bad(self):
        view = x_cost_view(
            _action(_star_x_card(has_non_x_effect=True)),
            _obs(stars=0),
        )
        self.assertTrue(view["is_zero"])
        self.assertTrue(view["non_x_effect_present"])
        self.assertFalse(view["zero_x_bad"])

    def test_star_x_does_not_share_energy_value(self):
        view = x_cost_view(_action(_star_x_card()), _obs(energy=5, stars=2))
        # current_value tracks STARS, not energy.
        self.assertEqual(view["current_value"], 2)


class TypedBridgeBlockTests(unittest.TestCase):
    def test_typed_runtime_internal_path(self):
        view = x_cost_view(_typed_x_card(), _obs(energy=5))  # obs ignored
        self.assertEqual(view["source_confidence"], "runtime_internal")
        self.assertEqual(view["resource"], "energy")
        self.assertEqual(view["current_value"], 0)
        self.assertEqual(view["semantics"], "repeat")
        self.assertEqual(view["preview_scale_source"], "energy_x")

    def test_typed_block_overrides_fallback_current_value(self):
        # Bridge says current_value=4 even though obs says energy=0.
        view = x_cost_view(_typed_x_card(current_value=4, is_zero=False), _obs(energy=0))
        self.assertEqual(view["current_value"], 4)
        self.assertFalse(view["is_zero"])

    def test_typed_block_resource_stars(self):
        view = x_cost_view(_typed_x_card(resource="stars", preview_scale_source="star_x"), _obs())
        self.assertEqual(view["resource"], "stars")
        self.assertEqual(view["preview_scale_source"], "star_x")


class ConvenienceHelperTests(unittest.TestCase):
    def test_is_x_cost_action(self):
        self.assertTrue(is_x_cost_action(_action(_energy_x_card()), _obs()))
        self.assertTrue(is_x_cost_action(_action(_star_x_card()), _obs()))
        self.assertFalse(is_x_cost_action(_action({"id": "X", "cost": 1}), _obs()))

    def test_is_zero_x_bad(self):
        self.assertTrue(is_zero_x_bad(_action(_energy_x_card()), _obs(energy=0)))
        self.assertFalse(is_zero_x_bad(_action(_energy_x_card()), _obs(energy=3)))
        self.assertFalse(is_zero_x_bad(_action(_energy_x_card(has_non_x_effect=True)), _obs(energy=0)))

    def test_x_cost_resource(self):
        self.assertEqual(x_cost_resource(_action(_energy_x_card()), _obs()), "energy")
        self.assertEqual(x_cost_resource(_action(_star_x_card()), _obs()), "stars")
        self.assertEqual(x_cost_resource(_action({"id": "X"}), _obs()), "none")


if __name__ == "__main__":
    unittest.main()
