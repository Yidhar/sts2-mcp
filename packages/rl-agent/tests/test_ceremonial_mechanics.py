"""Tests for the Ceremonial Beast mechanics helper (TASK-E2).

Verifies the spec acceptance shapes:

1. Under one-card lock, a low-damage Strike scores low impact.
2. Under one-card lock, a lethal action scores high impact.
3. Inside the stun window, an end_turn or off-action is flagged
   ``ceremonial_missed_stun_window``.
4. Under one-card lock, a draw-only action has impact capped low (since the
   drawn card cannot be played this turn).
5. State detection: one_card_lock and stun_window flags pulled from
   enemy.powers (no localized text).
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any


RL_AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(RL_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(RL_AGENT_ROOT))

from sts2_env.boss_ceremonial import (
    build_ceremonial_state,
    classify_ceremonial_action_mechanism,
)


def _ceremonial_enemy(*, hp: int = 60, intent_damage: int = 18, lock: bool = False, stun_window: bool = False, lock_counter: int = 2, stun_counter: int = 0) -> dict[str, Any]:
    powers: list[dict[str, Any]] = []
    if lock:
        powers.append({"id": "CeremonialOneCardLockPower", "amount": lock_counter})
    if stun_window:
        powers.append({"id": "CeremonialStunWindowPower", "amount": stun_counter})
    return {
        "combat_id": "ceremonial_beast",
        "id": "ceremonial_beast",
        "hp": hp,
        "max_hp": 100,
        "block": 0,
        "intent": {"intent_type": "attack", "total_damage": intent_damage, "damage_per_hit": intent_damage, "repeats": 1},
        "powers": powers,
    }


def _player(hp: int = 60, block: int = 0) -> dict[str, Any]:
    return {"hp": hp, "max_hp": 80, "block": block, "powers": []}


def _combat(*, lock: bool = False, stun_window: bool = False, target_hp: int = 60, intent: int = 18) -> dict[str, Any]:
    return {"enemies": [_ceremonial_enemy(hp=target_hp, intent_damage=intent, lock=lock, stun_window=stun_window)]}


def _play(card_id: str, *, damage: int = 0, block: int = 0, weak: int = 0, vulnerable: int = 0, draw: int = 0) -> dict[str, Any]:
    derived = {
        "combat_effect": {"damage": damage, "block": block, "weak": weak, "vulnerable": vulnerable, "target_type": "AnyEnemy" if damage > 0 else "Self"},
        "hand_mutation": {"draw": draw} if draw else {},
        "lifecycle": {}, "pile_mutation": {}, "cost": {}, "mechanism_effect": {},
    }
    return {
        "kind": "play_card",
        "action_id": f"play:{card_id}",
        "card": {"id": card_id, "title": card_id, "type": "Attack" if damage > 0 else "Skill",
                 "card_effect_profile": {"derived_view": derived}},
    }


def _end_turn() -> dict[str, Any]:
    return {"kind": "end_turn", "action_id": "end_turn"}


class CeremonialStateDetectionTests(unittest.TestCase):
    def test_one_card_lock_detected_from_power(self):
        state = build_ceremonial_state(_combat(lock=True))
        self.assertTrue(state["active"])
        self.assertTrue(state["one_card_lock_active"])
        self.assertFalse(state["stun_window_active"])
        self.assertEqual(state["actions_remaining_this_turn"], 1)
        self.assertEqual(state["lock_counter"], 2.0)
        self.assertEqual(state["incoming_after_lock"], 18.0)

    def test_stun_window_detected_from_power(self):
        state = build_ceremonial_state(_combat(stun_window=True))
        self.assertTrue(state["stun_window_active"])
        self.assertFalse(state["one_card_lock_active"])

    def test_inactive_when_no_ceremonial_enemy(self):
        plain = {"enemies": [{"id": "X", "powers": [], "intent": {}}]}
        state = build_ceremonial_state(plain)
        self.assertFalse(state["active"])


class CeremonialImpactScoreTests(unittest.TestCase):
    def test_low_strike_under_lock_low_impact(self):
        action = _play("Strike", damage=6)
        mech = classify_ceremonial_action_mechanism(_combat(lock=True), action, player_obs=_player())
        self.assertLess(mech["ceremonial_single_action_impact_score"], 0.25)
        self.assertTrue(mech["ceremonial_low_impact_under_lock"])

    def test_lethal_under_lock_high_impact(self):
        action = _play("Bludgeon", damage=80)  # damage >= boss hp 60 → lethal
        mech = classify_ceremonial_action_mechanism(_combat(lock=True, target_hp=60), action, player_obs=_player())
        self.assertEqual(mech["ceremonial_single_action_impact_score"], 1.0)
        self.assertFalse(mech["ceremonial_low_impact_under_lock"])
        self.assertFalse(mech["ceremonial_wastes_one_card_lock"])

    def test_draw_only_under_lock_capped_low(self):
        action = _play("Cycle", draw=2)
        mech = classify_ceremonial_action_mechanism(_combat(lock=True), action, player_obs=_player())
        self.assertLessEqual(mech["ceremonial_single_action_impact_score"], 0.15)


class CeremonialStunWindowTests(unittest.TestCase):
    def test_end_turn_in_stun_window_is_offender(self):
        mech = classify_ceremonial_action_mechanism(_combat(stun_window=True), _end_turn(), player_obs=_player())
        self.assertTrue(mech["ceremonial_missed_stun_window"])
        self.assertFalse(mech["ceremonial_uses_stun_window"])

    def test_high_impact_in_stun_window_uses_window(self):
        action = _play("Bludgeon", damage=80)
        mech = classify_ceremonial_action_mechanism(_combat(stun_window=True, target_hp=60), action, player_obs=_player())
        self.assertTrue(mech["ceremonial_uses_stun_window"])
        self.assertFalse(mech["ceremonial_missed_stun_window"])


class CeremonialNeutralTests(unittest.TestCase):
    def test_no_ceremonial_returns_neutral_defaults(self):
        plain = {"enemies": [{"id": "X", "powers": [], "intent": {}}]}
        mech = classify_ceremonial_action_mechanism(plain, _play("Strike", damage=6), player_obs=_player())
        self.assertEqual(mech["ceremonial_single_action_impact_score"], 0.0)
        self.assertFalse(mech["ceremonial_low_impact_under_lock"])
        self.assertFalse(mech["ceremonial_missed_stun_window"])


if __name__ == "__main__":
    unittest.main()
