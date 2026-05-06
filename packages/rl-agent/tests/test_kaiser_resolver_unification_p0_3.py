"""Tests for P0-3 Kaiser facing resolver unification.

Verifies that:

* No code path uses faction ``enemy.side`` / ``target.side`` (Player /
  Enemy) as a left/right axis.  The only authoritative cue is
  ``BACK_ATTACK_LEFT_POWER`` / ``BACK_ATTACK_RIGHT_POWER`` on the target
  enemy's powers list.
* :func:`combat_env.CombatSandboxEnv._classify_refund_followup` recognises
  a refund-no-followup that targets the opposite-side back-attack enemy as
  intrinsic-value via the shared ``boss_kaiser`` resolver.
* :mod:`sts2_env.potion_timing` flips ``facing_change=True`` for a potion
  aimed at the opposite-side enemy via the shared resolver, instead of
  the legacy permanent ``facing_change=False`` placeholder.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any


RL_AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(RL_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(RL_AGENT_ROOT))

from sts2_env.boss_kaiser import classify_kaiser_action_mechanism
from sts2_env.combat_env import CombatSandboxEnv


def _kaiser_combat() -> dict[str, Any]:
    """Two enemies: left part has BACK_ATTACK_LEFT_POWER, right has the
    matching right power.  Both share faction ``side='Enemy'`` (faction)
    which must NOT be interpreted as left/right by any resolver."""
    return {
        "enemies": [
            {
                "combat_id": "left_part",
                "id": "left_part",
                "side": "Enemy",
                "hp": 60,
                "max_hp": 100,
                "powers": [{"id": "BackAttackLeftPower"}],
                "intent": {"intent_type": "attack", "total_damage": 18},
            },
            {
                "combat_id": "right_part",
                "id": "right_part",
                "side": "Enemy",
                "hp": 60,
                "max_hp": 100,
                "powers": [{"id": "BackAttackRightPower"}],
                "intent": {"intent_type": "attack", "total_damage": 18},
            },
        ],
    }


def _player(facing: str = "left") -> dict[str, Any]:
    return {
        "hp": 60,
        "max_hp": 80,
        "block": 0,
        "powers": [{"id": "SurroundedPower", "facing": facing}],
    }


class KaiserResolverFactionSideRejection(unittest.TestCase):
    def test_target_with_faction_side_only_does_not_change_facing(self):
        """Target dict carrying ``side='Enemy'`` (faction) must NOT be read
        as a left/right axis.  Without BACK_ATTACK powers there is no
        position information, so facing-change must be False."""
        plain_combat = {
            "enemies": [
                {
                    "combat_id": "plain",
                    "id": "plain",
                    "side": "Enemy",  # faction string — must be ignored
                    "hp": 30,
                    "max_hp": 50,
                    "powers": [],  # no BACK_ATTACK power
                    "intent": {"total_damage": 5},
                }
            ]
        }
        action = {
            "kind": "play_card",
            "action_id": "play:strike",
            "target": {"combat_id": "plain", "side": "Enemy", "scope": "single_enemy"},
            "card": {"id": "STRIKE", "title": "Strike", "type": "Attack",
                     "card_effect_profile": {"derived_view": {"combat_effect": {"damage": 6, "target_type": "AnyEnemy"}, "lifecycle": {}, "hand_mutation": {}, "pile_mutation": {}, "cost": {}, "mechanism_effect": {}}}},
        }
        mech = classify_kaiser_action_mechanism(plain_combat, action, player_obs=_player("left"))
        self.assertFalse(mech["kaiser_can_change_facing"])
        self.assertFalse(mech["kaiser_changes_facing"])

    def test_back_attack_left_power_drives_position(self):
        action = {
            "kind": "play_card",
            "action_id": "play:strike",
            "target": {"combat_id": "left_part", "side": "Enemy"},
            "card": {"id": "STRIKE", "title": "Strike", "type": "Attack",
                     "card_effect_profile": {"derived_view": {"combat_effect": {"damage": 8, "target_type": "AnyEnemy"}, "lifecycle": {}, "hand_mutation": {}, "pile_mutation": {}, "cost": {}, "mechanism_effect": {}}}},
        }
        # Player facing right → opposite of left_part → changes_facing
        mech = classify_kaiser_action_mechanism(_kaiser_combat(), action, player_obs=_player("right"))
        self.assertTrue(mech["kaiser_changes_facing"])
        self.assertEqual(mech["kaiser_facing_after_if_action"], "left")


class CombatEnvRefundIntrinsicViaResolver(unittest.TestCase):
    def _refund_card(self, *, energy: int = 2) -> dict[str, Any]:
        return {
            "id": "REFUND",
            "title": "Refund",
            "type": "Skill",
            "cost": 1,
            "preview_damage": 0,
            "preview_block": 0,
            "card_effect_profile": {
                "operations": [
                    {"op": "gain_energy", "energy": energy, "timing": "same_turn_resource"}
                ],
                "semantic_tags": [],
                "training_tags": [],
            },
        }

    def test_refund_targeting_opposite_side_classified_as_intrinsic(self):
        env = CombatSandboxEnv.__new__(CombatSandboxEnv)
        refund = self._refund_card()
        # Player faces left → targeting right_part is opposite-side → facing change
        action = {
            "kind": "play_card",
            "action_id": "play:REFUND",
            "card": refund,
            "target": {"combat_id": "right_part", "side": "Enemy"},
        }
        legal = [action, {"kind": "end_turn", "action_id": "end_turn"}]
        raw_obs = {"combat": _kaiser_combat(), "player": _player("left")}
        cls = env._classify_refund_followup(action, energy=1.0, legal_actions=legal, raw_obs=raw_obs)
        self.assertEqual(cls, "refund_no_followup_but_intrinsic_value")

    def test_refund_targeting_same_side_is_low_value(self):
        env = CombatSandboxEnv.__new__(CombatSandboxEnv)
        refund = self._refund_card()
        # Player faces left, targeting left_part is same-side → no facing change
        action = {
            "kind": "play_card",
            "action_id": "play:REFUND",
            "card": refund,
            "target": {"combat_id": "left_part", "side": "Enemy"},
        }
        legal = [action, {"kind": "end_turn", "action_id": "end_turn"}]
        raw_obs = {"combat": _kaiser_combat(), "player": _player("left")}
        cls = env._classify_refund_followup(action, energy=1.0, legal_actions=legal, raw_obs=raw_obs)
        self.assertEqual(cls, "refund_no_followup_low_value")

    def test_refund_with_faction_side_only_no_back_attack_powers_low_value(self):
        """Even if target dict has ``side='left'`` literal, without
        BACK_ATTACK powers the resolver must not classify as facing-change."""
        env = CombatSandboxEnv.__new__(CombatSandboxEnv)
        refund = self._refund_card()
        plain_combat = {
            "enemies": [
                {
                    "combat_id": "x",
                    "id": "x",
                    # Faction-style side string — should not drive position
                    "side": "left",
                    "hp": 30,
                    "max_hp": 50,
                    "powers": [],  # no BACK_ATTACK_*_POWER
                    "intent": {"total_damage": 5},
                }
            ]
        }
        action = {
            "kind": "play_card",
            "action_id": "play:REFUND",
            "card": refund,
            "target": {"combat_id": "x", "side": "right"},
        }
        legal = [action, {"kind": "end_turn", "action_id": "end_turn"}]
        raw_obs = {"combat": plain_combat, "player": _player("left")}
        cls = env._classify_refund_followup(action, energy=1.0, legal_actions=legal, raw_obs=raw_obs)
        self.assertEqual(cls, "refund_no_followup_low_value")


class PotionTimingFacingViaResolver(unittest.TestCase):
    def test_targeted_potion_opposite_side_flips_facing_change(self):
        import numpy as np

        from sts2_env.potion_timing import compute_potion_timing  # noqa: WPS433

        action = {
            "kind": "use_potion",
            "action_id": "potion:fire",
            "potion": {
                "id": "POTION.FIRE",
                "title": "Fire Potion",
                "damage": 20,
                "preview_damage": 20,
            },
            "target": {"combat_id": "right_part", "side": "Enemy"},
        }
        legal = [action]
        mask = np.asarray([1.0], dtype=np.float32)
        raw_obs = {
            "combat": _kaiser_combat(),
            "player": {**_player("left"), "potions": [{"id": "POTION.FIRE", "title": "Fire Potion"}]},
        }
        profile = compute_potion_timing(
            action=action,
            raw_obs=raw_obs,
            legal_actions=legal,
            mask=mask,
            energy=3.0,
        )
        self.assertTrue(profile.get("facing_change"), msg=f"profile={profile}")


if __name__ == "__main__":
    unittest.main()
