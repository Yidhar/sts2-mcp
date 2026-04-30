"""Tests for The Insatiable mechanics + offender taxonomy (TASK-E3).

Verifies:

1. State detection — boss is recognized from id token or growth/pressure
   power channel.
2. ``insatiable_strategic_skip`` fires when the diagnostic context flags a
   strategic skip on the selected action.
3. ``insatiable_refund_no_followup`` fires when the refund classifier
   returns ``refund_no_followup_low_value``.
4. ``insatiable_bad_end_turn`` fires when ending the turn would let
   incoming damage chunk through standing block.
5. ``insatiable_missed_pressure_window`` fires when pressure / urgent_window
   is high and the player still chose a low-impact play or end_turn.
6. Inactive state returns neutral defaults.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any


RL_AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(RL_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(RL_AGENT_ROOT))

from sts2_env.boss_insatiable import (
    build_insatiable_state,
    classify_insatiable_action_offenders,
)


def _insatiable_enemy(*, growth: int = 0, pressure: int = 0, devour: bool = False, intent_damage: int = 18, hp: int = 80) -> dict[str, Any]:
    powers: list[dict[str, Any]] = []
    if growth > 0:
        powers.append({"id": "InsatiableGrowthPower", "amount": growth})
    if pressure > 0:
        powers.append({"id": "InsatiablePressurePower", "amount": pressure})
    if devour:
        powers.append({"id": "InsatiableDevourPower", "amount": 1})
    return {
        "combat_id": "the_insatiable_boss",
        "id": "the_insatiable_boss",
        "hp": hp,
        "max_hp": 200,
        "block": 0,
        "intent": {"intent_type": "attack", "total_damage": intent_damage, "damage_per_hit": intent_damage, "repeats": 1},
        "powers": powers,
    }


def _player(block: int = 0) -> dict[str, Any]:
    return {"hp": 60, "max_hp": 80, "block": block, "powers": []}


def _combat(*, growth: int = 0, pressure: int = 0, devour: bool = False, intent: int = 18) -> dict[str, Any]:
    return {"enemies": [_insatiable_enemy(growth=growth, pressure=pressure, devour=devour, intent_damage=intent)]}


def _play(card_id: str, *, damage: int = 0, block: int = 0, weak: int = 0, vulnerable: int = 0) -> dict[str, Any]:
    derived = {
        "combat_effect": {"damage": damage, "block": block, "weak": weak, "vulnerable": vulnerable},
        "lifecycle": {}, "hand_mutation": {}, "pile_mutation": {}, "cost": {}, "mechanism_effect": {},
    }
    return {
        "kind": "play_card",
        "action_id": f"play:{card_id}",
        "card": {"id": card_id, "title": card_id, "type": "Attack" if damage > 0 else "Skill",
                 "card_effect_profile": {"derived_view": derived}},
    }


def _end_turn() -> dict[str, Any]:
    return {"kind": "end_turn", "action_id": "end_turn"}


class InsatiableStateDetectionTests(unittest.TestCase):
    def test_detected_via_id_token(self):
        state = build_insatiable_state(_combat(growth=2))
        self.assertTrue(state["active"])
        self.assertEqual(state["growth_or_scaling"], 2.0)

    def test_pressure_power_sets_urgent_window(self):
        state = build_insatiable_state(_combat(pressure=3))
        self.assertTrue(state["active"])
        self.assertTrue(state["urgent_window"])
        self.assertEqual(state["special_counter"], 3.0)

    def test_inactive_when_no_insatiable_enemy(self):
        plain = {"enemies": [{"id": "X", "powers": [], "intent": {}}]}
        state = build_insatiable_state(plain)
        self.assertFalse(state["active"])


class InsatiableStrategicSkipTests(unittest.TestCase):
    def test_strategic_skip_diag_flags_offender(self):
        action = _play("Strike", damage=6)
        diag = {"strategic_skip_selected": True}
        flags = classify_insatiable_action_offenders(_combat(growth=2), action, player_obs=_player(), action_diagnostics=diag)
        self.assertTrue(flags["insatiable_strategic_skip"])

    def test_strategic_skip_does_not_fire_without_diag(self):
        action = _play("Strike", damage=6)
        flags = classify_insatiable_action_offenders(_combat(growth=2), action, player_obs=_player())
        self.assertFalse(flags["insatiable_strategic_skip"])


class InsatiableRefundNoFollowupTests(unittest.TestCase):
    def test_refund_no_followup_diag_flags_offender(self):
        refund = _play("Refund")
        diag = {"refund_followup_class": "refund_no_followup_low_value"}
        flags = classify_insatiable_action_offenders(_combat(growth=2), refund, player_obs=_player(), action_diagnostics=diag)
        self.assertTrue(flags["insatiable_refund_no_followup"])

    def test_good_refund_followup_does_not_flag(self):
        refund = _play("Refund")
        diag = {"refund_followup_class": "refund_good_followup"}
        flags = classify_insatiable_action_offenders(_combat(growth=2), refund, player_obs=_player(), action_diagnostics=diag)
        self.assertFalse(flags["insatiable_refund_no_followup"])


class InsatiableBadEndTurnTests(unittest.TestCase):
    def test_end_turn_under_pressure_with_low_block(self):
        flags = classify_insatiable_action_offenders(_combat(growth=2, intent=20), _end_turn(), player_obs=_player(block=2))
        self.assertTrue(flags["insatiable_bad_end_turn"])

    def test_end_turn_well_blocked_does_not_flag(self):
        flags = classify_insatiable_action_offenders(_combat(growth=2, intent=14), _end_turn(), player_obs=_player(block=14))
        self.assertFalse(flags["insatiable_bad_end_turn"])


class InsatiableMissedPressureWindowTests(unittest.TestCase):
    def test_low_impact_play_inside_urgent_window(self):
        action = _play("Defend", block=4)
        flags = classify_insatiable_action_offenders(_combat(pressure=3, intent=20), action, player_obs=_player(block=2))
        self.assertTrue(flags["insatiable_missed_pressure_window"])
        self.assertFalse(flags["insatiable_pressure_action_selected"])

    def test_high_impact_play_inside_urgent_window_is_correct_play(self):
        action = _play("Bash", damage=12, vulnerable=2)
        flags = classify_insatiable_action_offenders(_combat(pressure=3, intent=20), action, player_obs=_player())
        self.assertFalse(flags["insatiable_missed_pressure_window"])
        self.assertTrue(flags["insatiable_pressure_action_selected"])


class InsatiableNeutralTests(unittest.TestCase):
    def test_inactive_returns_neutral_defaults(self):
        plain = {"enemies": [{"id": "X", "powers": [], "intent": {}}]}
        flags = classify_insatiable_action_offenders(plain, _play("Strike", damage=6), player_obs=_player(),
                                                     action_diagnostics={"strategic_skip_selected": True,
                                                                          "refund_followup_class": "refund_no_followup_low_value"})
        self.assertFalse(flags["insatiable_strategic_skip"])
        self.assertFalse(flags["insatiable_refund_no_followup"])
        self.assertFalse(flags["insatiable_bad_end_turn"])
        self.assertFalse(flags["insatiable_missed_pressure_window"])


if __name__ == "__main__":
    unittest.main()
