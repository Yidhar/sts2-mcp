"""Tests for the Kaiser facing semantics helper (TASK-E1).

Verifies the spec acceptance shapes:

1. ``enemy.side='Enemy'`` is irrelevant — the position must be derived from the
   ``BackAttackLeftPower`` / ``BackAttackRightPower`` on the enemy parts.
2. Player facing ``left`` + target on left → no facing change.
3. Player facing ``left`` + target on right → facing change (and back-attack
   risk re-estimated).
4. A targeted potion against a right-side enemy also changes facing.
5. A self-target defense card does not change facing but still counts as a
   defense candidate.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any


RL_AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(RL_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(RL_AGENT_ROOT))

from sts2_env.boss_kaiser import (
    build_kaiser_state,
    classify_kaiser_action_mechanism,
)


def _enemy(eid: str, *, side_power: str | None, intent_damage: int = 12, hp: int = 60) -> dict[str, Any]:
    powers: list[dict[str, Any]] = []
    if side_power == "left":
        powers.append({"id": "BackAttackLeftPower"})
    elif side_power == "right":
        powers.append({"id": "BackAttackRightPower"})
    return {
        "combat_id": eid,
        "id": eid,
        "side": "Enemy",  # camp axis — must NOT be used for left/right.
        "hp": hp,
        "max_hp": 100,
        "block": 0,
        "intent": {"intent_type": "attack", "total_damage": intent_damage, "damage_per_hit": intent_damage, "repeats": 1},
        "powers": powers,
    }


def _player(facing: str | None) -> dict[str, Any]:
    powers: list[dict[str, Any]] = []
    if facing in {"left", "right"}:
        powers.append({"id": "SurroundedPower", "facing": facing})
    return {"hp": 70, "max_hp": 80, "block": 0, "powers": powers}


def _combat(left_enemy_side: str | None = "left", right_enemy_side: str | None = "right") -> dict[str, Any]:
    enemies = []
    if left_enemy_side is not None:
        enemies.append(_enemy("left_part", side_power=left_enemy_side))
    if right_enemy_side is not None:
        enemies.append(_enemy("right_part", side_power=right_enemy_side))
    return {"enemies": enemies}


def _play_card(card_id: str, *, target_combat_id: str | None = None, target_scope: str = "single_enemy",
               card_target_type: str = "AnyEnemy", damage: int = 0, block: int = 0) -> dict[str, Any]:
    derived = {
        "combat_effect": {"damage": damage, "block": block, "target_type": card_target_type},
        "lifecycle": {}, "hand_mutation": {}, "pile_mutation": {}, "cost": {},
        "mechanism_effect": {},
    }
    card = {
        "id": card_id, "title": card_id, "type": "Attack" if damage > 0 else "Skill",
        "cost": 1, "target_type": card_target_type,
        "card_effect_profile": {"derived_view": derived},
    }
    action: dict[str, Any] = {
        "kind": "play_card",
        "action_id": f"play:{card_id}",
        "card": card,
        "target_scope": target_scope,
    }
    if target_combat_id:
        action["target"] = {"combat_id": target_combat_id, "scope": target_scope}
    return action


def _use_potion(potion_id: str, *, target_combat_id: str | None = None) -> dict[str, Any]:
    return {
        "kind": "use_potion",
        "action_id": f"potion:{potion_id}",
        "potion": {"id": potion_id},
        "target": {"combat_id": target_combat_id, "scope": "single_enemy"} if target_combat_id else None,
        "target_scope": "single_enemy" if target_combat_id else "self",
    }


class KaiserStateDetectionTests(unittest.TestCase):
    def test_position_from_back_attack_power_not_enemy_side(self):
        state = build_kaiser_state(_combat(), player_obs=_player("left"))
        self.assertTrue(state["active"])
        self.assertEqual(state["left_enemy_ids"], ["left_part"])
        self.assertEqual(state["right_enemy_ids"], ["right_part"])
        self.assertTrue(state["has_left_back_attack_power"])
        self.assertTrue(state["has_right_back_attack_power"])

    def test_inactive_when_no_back_attack_powers(self):
        plain = {"enemies": [{"id": "X", "powers": [], "intent": {}}]}
        state = build_kaiser_state(plain, player_obs=_player(None))
        self.assertFalse(state["active"])
        self.assertEqual(state["back_attack_risk"], 0.0)

    def test_back_attack_risk_when_facing_into_safe_side(self):
        # Player facing left; back is right; if right_enemy has BackAttackRightPower → risk.
        state = build_kaiser_state(_combat(), player_obs=_player("left"))
        self.assertAlmostEqual(state["back_attack_risk"], 1.0)
        self.assertAlmostEqual(state["incoming_multiplier"], 1.5)

    def test_no_risk_when_facing_into_threat(self):
        # Player facing right; back is left; if left_enemy has BackAttackLeftPower → risk.
        # Conversely, player facing left into right means risk; flip player to face right
        # while only the LEFT enemy has the power → back is left → still risk.  Use a
        # combat where ONLY the right enemy has a back-attack power and player faces right
        # so back is left and there is no left-back-attack power.
        combat = _combat(left_enemy_side=None, right_enemy_side="right")
        state = build_kaiser_state(combat, player_obs=_player("right"))
        self.assertAlmostEqual(state["back_attack_risk"], 0.0)
        self.assertAlmostEqual(state["incoming_multiplier"], 1.0)


class KaiserFacingChangeTests(unittest.TestCase):
    def test_target_on_same_side_no_change(self):
        action = _play_card("StrikeLeft", target_combat_id="left_part", damage=8)
        mech = classify_kaiser_action_mechanism(_combat(), action, player_obs=_player("left"))
        self.assertTrue(mech["kaiser_can_change_facing"])
        self.assertFalse(mech["kaiser_changes_facing"])
        self.assertEqual(mech["kaiser_facing_before"], "left")
        self.assertEqual(mech["kaiser_facing_after_if_action"], "left")

    def test_target_on_opposite_side_changes_facing(self):
        action = _play_card("StrikeRight", target_combat_id="right_part", damage=8)
        mech = classify_kaiser_action_mechanism(_combat(), action, player_obs=_player("left"))
        self.assertTrue(mech["kaiser_changes_facing"])
        self.assertEqual(mech["kaiser_facing_after_if_action"], "right")
        # Player flipped to facing right; back is now left and left_enemy holds
        # BackAttackLeftPower → still risky; multiplier stays 1.5; risk_delta=0.
        self.assertAlmostEqual(mech["kaiser_incoming_multiplier_after_estimate"], 1.5)
        self.assertAlmostEqual(mech["kaiser_risk_delta"], 0.0)

    def test_facing_flip_relieves_risk_when_only_one_side_dangerous(self):
        # Only the right enemy holds BackAttackRightPower; player faces left so
        # back is right → currently risky.  Targeting the right enemy flips
        # facing to right → back is now left where there is no BackAttackLeft
        # power → multiplier drops to 1.0, risk_delta = -0.5.
        combat = _combat(left_enemy_side=None, right_enemy_side="right")
        action = _play_card("Pivot", target_combat_id="right_part", damage=8)
        mech = classify_kaiser_action_mechanism(combat, action, player_obs=_player("left"))
        self.assertTrue(mech["kaiser_changes_facing"])
        self.assertAlmostEqual(mech["kaiser_incoming_multiplier_before"], 1.5)
        self.assertAlmostEqual(mech["kaiser_incoming_multiplier_after_estimate"], 1.0)
        self.assertAlmostEqual(mech["kaiser_risk_delta"], -0.5)


class KaiserPotionAndDefenseTests(unittest.TestCase):
    def test_targeted_potion_changes_facing(self):
        action = _use_potion("FireBomb", target_combat_id="right_part")
        mech = classify_kaiser_action_mechanism(_combat(), action, player_obs=_player("left"))
        self.assertTrue(mech["kaiser_can_change_facing"])
        self.assertTrue(mech["kaiser_changes_facing"])
        self.assertEqual(mech["kaiser_facing_after_if_action"], "right")

    def test_self_target_defense_no_facing_change_but_defense_candidate(self):
        action = _play_card("Defend", target_combat_id=None, target_scope="self",
                            card_target_type="Self", block=8)
        mech = classify_kaiser_action_mechanism(_combat(), action, player_obs=_player("left"))
        self.assertFalse(mech["kaiser_changes_facing"])
        self.assertTrue(mech["kaiser_defense_candidate"])
        self.assertFalse(mech["kaiser_pressure_candidate"])


if __name__ == "__main__":
    unittest.main()
