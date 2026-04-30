"""Tests for the structured ``derived_view`` block on card effect profiles
(TASK-D1).

Verifies that:

* Armaments exposes upgrade-state-aware ``upgrade_targets``.
* Apparition (Ethereal+Exhaust) flags both lifecycle bits.
* Brand exhausts on play and exhausts a chosen target from hand without
  reporting ``removes_card_from_combat`` (Brand's hand target is exhausted,
  not transformed).
* Whirlwind's X-cost flag is reported even when catalog ``cost`` is ``0``.
* Bloodletting reports the energy_gain amount.
* The reader helpers in :mod:`sts2_env.card_effect_profile` find the
  ``derived_view`` block whether the profile is attached as ``card_effect_profile``
  or directly on the card payload.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from typing import Any


RL_AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(RL_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(RL_AGENT_ROOT))

from sts2_env.card_effect_profile import (
    card_combat_effect_view,
    card_cost_view,
    card_derived_view,
    card_hand_mutation_view,
    card_lifecycle_view,
    card_mechanism_effect_view,
    card_pile_mutation_view,
    card_source_view,
)


PROFILES_PATH = RL_AGENT_ROOT / "content" / "card_effect_profiles.generated.json"


def _load_profile(card_id: str) -> dict[str, Any]:
    with PROFILES_PATH.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    cards = data.get("cards") or {}
    profile = cards.get(card_id)
    if not isinstance(profile, dict):
        raise AssertionError(f"profile {card_id!r} not found in registry")
    return profile


def _card_payload(card_id: str) -> dict[str, Any]:
    """Wrap a registry profile into the live-card-payload shape consumed by
    observation/action encoders."""
    profile = _load_profile(card_id)
    return {
        "id": card_id,
        "title": profile.get("title_en"),
        "card_effect_profile": profile,
    }


class DerivedViewArmamentsTests(unittest.TestCase):
    def test_armaments_upgrade_targets_state_aware(self):
        view = card_hand_mutation_view(_card_payload("CARD.ARMAMENTS"))
        self.assertTrue(view.get("upgrades_hand"))
        self.assertEqual(view.get("upgrade_targets"), "one_or_all_by_upgrade_state")
        self.assertTrue(view["select_cards"]["enabled"])
        self.assertEqual(view["select_cards"]["target_zone"], "hand")


class DerivedViewLifecycleTests(unittest.TestCase):
    def test_apparition_ethereal_and_exhaust(self):
        life = card_lifecycle_view(_card_payload("CARD.APPARITION"))
        self.assertTrue(life["ethereal"])
        self.assertTrue(life["exhausts_on_play"])
        pile = card_pile_mutation_view(_card_payload("CARD.APPARITION"))
        self.assertTrue(pile["moves_to_exhaust_self"])
        self.assertFalse(pile["moves_to_discard_self"])

    def test_brand_exhausts_self_without_remove_combat(self):
        life = card_lifecycle_view(_card_payload("CARD.BRAND"))
        pile = card_pile_mutation_view(_card_payload("CARD.BRAND"))
        self.assertTrue(life["exhausts_on_play"])
        # Played card itself goes to exhaust pile (Exhaust keyword) — does not
        # constitute "removed from combat" (transform) under the schema.
        self.assertTrue(pile["moves_to_exhaust_self"])
        self.assertFalse(pile["removes_card_from_combat"])

    def test_armaments_default_pile_is_discard_self(self):
        pile = card_pile_mutation_view(_card_payload("CARD.ARMAMENTS"))
        self.assertTrue(pile["moves_to_discard_self"])
        self.assertFalse(pile["moves_to_exhaust_self"])


class DerivedViewCostTests(unittest.TestCase):
    def test_whirlwind_x_cost_flagged(self):
        cost = card_cost_view(_card_payload("CARD.WHIRLWIND"))
        self.assertTrue(cost["is_x_cost"])

    def test_armaments_static_cost_one(self):
        cost = card_cost_view(_card_payload("CARD.ARMAMENTS"))
        self.assertFalse(cost["is_x_cost"])
        self.assertEqual(cost["base"], 1)


class DerivedViewCombatEffectTests(unittest.TestCase):
    def test_bloodletting_energy_gain(self):
        combat = card_combat_effect_view(_card_payload("CARD.BLOODLETTING"))
        self.assertEqual(combat["energy_gain"], 2)

    def test_armaments_block_amount(self):
        combat = card_combat_effect_view(_card_payload("CARD.ARMAMENTS"))
        self.assertEqual(combat["block"], 5)
        self.assertEqual(combat["damage"], 0)


class DerivedViewMechanismEffectTests(unittest.TestCase):
    def test_self_target_skill_cannot_change_facing(self):
        mech = card_mechanism_effect_view(_card_payload("CARD.ARMAMENTS"))
        self.assertFalse(mech["can_change_facing"])

    def test_enemy_target_attack_can_change_facing(self):
        mech = card_mechanism_effect_view(_card_payload("CARD.WHIRLWIND"))
        self.assertTrue(mech["can_change_facing"])


class DerivedViewSourceTests(unittest.TestCase):
    def test_armaments_curated_quality(self):
        src = card_source_view(_card_payload("CARD.ARMAMENTS"))
        self.assertEqual(src["primary"], "game_internal_id")
        self.assertFalse(src["fallback_text_regex_used"])
        self.assertEqual(src["profile_quality"], "curated_internal_id")


class DerivedViewReaderShapeTests(unittest.TestCase):
    def test_reader_finds_view_when_profile_attached(self):
        view = card_derived_view(_card_payload("CARD.ARMAMENTS"))
        self.assertIn("cost", view)
        self.assertIn("lifecycle", view)
        self.assertIn("hand_mutation", view)
        self.assertIn("pile_mutation", view)
        self.assertIn("combat_effect", view)
        self.assertIn("mechanism_effect", view)
        self.assertIn("source", view)

    def test_reader_finds_view_when_attached_directly(self):
        profile = _load_profile("CARD.ARMAMENTS")
        card = {"id": profile["id"], "derived_view": profile["derived_view"]}
        view = card_derived_view(card)
        self.assertEqual(view["card_id"], "CARD.ARMAMENTS")

    def test_reader_returns_empty_for_card_without_profile(self):
        self.assertEqual(card_derived_view({"id": "X"}), {})
        self.assertEqual(card_derived_view(None), {})


if __name__ == "__main__":
    unittest.main()
