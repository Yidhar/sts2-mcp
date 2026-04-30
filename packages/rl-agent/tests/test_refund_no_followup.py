"""Tests for refund-no-followup taxonomy (TASK-B3).

Verifies that the refund classifier reads after-action draw / create /
cost-reduction / replay signals before declaring a refund "low value", and
that intrinsic block/damage/mechanism value upgrades the verdict.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


RL_AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(RL_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(RL_AGENT_ROOT))

from sts2_env.combat_env import CombatSandboxEnv


def _card(
    title: str,
    *,
    card_type: str = "Skill",
    cost: int = 0,
    damage: int = 0,
    block: int = 0,
    operations: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "id": title.upper().replace(" ", "_"),
        "title": title,
        "type": card_type,
        "cost": cost,
        "damage": damage,
        "block": block,
        "preview_damage": damage,
        "preview_block": block,
        "card_effect_profile": {
            "operations": operations or [],
            "semantic_tags": [],
            "training_tags": [],
        },
    }


def _refund_card(
    *, energy_gain: int = 2, draw: int = 0, modify_cost: bool = False, damage: int = 0,
    block: int = 0, generated_card: bool = False,
) -> dict[str, object]:
    ops: list[dict[str, object]] = [
        {"op": "gain_energy", "energy": energy_gain, "timing": "same_turn_resource"}
    ]
    if draw > 0:
        ops.append({"op": "draw_card", "count": draw})
    if modify_cost:
        ops.append({"op": "modify_cost", "cost_delta": -1})
    if generated_card:
        ops.append({"op": "create_card", "amount": 1})
    return _card(
        "Refund",
        card_type="Skill",
        cost=1,
        damage=damage,
        block=block,
        operations=ops,
    )


def _play(card: dict[str, object]) -> dict[str, object]:
    return {
        "kind": "play_card",
        "action_id": f"play:{card['id']}",
        "card": card,
    }


class RefundFollowupClassificationTests(unittest.TestCase):
    def test_refund_with_draw_is_good_followup(self):
        env = CombatSandboxEnv.__new__(CombatSandboxEnv)
        refund = _play(_refund_card(energy_gain=2, draw=2))
        legal = [refund, {"kind": "end_turn", "action_id": "end_turn"}]
        cls = env._classify_refund_followup(refund, energy=1.0, legal_actions=legal, raw_obs={})
        self.assertEqual(cls, "refund_good_followup")

    def test_refund_with_cost_reduction_is_good_followup(self):
        env = CombatSandboxEnv.__new__(CombatSandboxEnv)
        refund = _play(_refund_card(energy_gain=2, modify_cost=True))
        legal = [refund, {"kind": "end_turn", "action_id": "end_turn"}]
        cls = env._classify_refund_followup(refund, energy=1.0, legal_actions=legal, raw_obs={})
        self.assertEqual(cls, "refund_good_followup")

    def test_refund_with_static_playable_followup_is_good(self):
        env = CombatSandboxEnv.__new__(CombatSandboxEnv)
        refund = _play(_refund_card(energy_gain=2))
        strike = _play(_card("Strike", card_type="Attack", cost=1, damage=6))
        legal = [refund, strike, {"kind": "end_turn", "action_id": "end_turn"}]
        cls = env._classify_refund_followup(refund, energy=1.0, legal_actions=legal, raw_obs={})
        self.assertEqual(cls, "refund_good_followup")

    def test_refund_no_followup_low_value(self):
        # Refund with no draw, no cost reduction, no static followup, no
        # intrinsic block/damage — this is the genuine low-value case.
        env = CombatSandboxEnv.__new__(CombatSandboxEnv)
        refund = _play(_refund_card(energy_gain=2))
        legal = [refund, {"kind": "end_turn", "action_id": "end_turn"}]
        cls = env._classify_refund_followup(refund, energy=1.0, legal_actions=legal, raw_obs={})
        self.assertEqual(cls, "refund_no_followup_low_value")

    def test_refund_block_lethal_is_intrinsic(self):
        # Refund that adds enough block to negate incoming lethal — even with
        # no static followup, this is intrinsic value.
        env = CombatSandboxEnv.__new__(CombatSandboxEnv)
        refund = _play(_refund_card(energy_gain=2, block=20))
        legal = [refund, {"kind": "end_turn", "action_id": "end_turn"}]
        raw_obs = {
            "player": {"hp": 30, "block": 0},
            "combat": {
                "enemies": [
                    {"intent": {"total_damage": 18}, "hp": 50},
                ],
            },
        }
        cls = env._classify_refund_followup(refund, energy=1.0, legal_actions=legal, raw_obs=raw_obs)
        self.assertEqual(cls, "refund_no_followup_but_intrinsic_value")

    def test_refund_high_damage_is_intrinsic(self):
        env = CombatSandboxEnv.__new__(CombatSandboxEnv)
        refund = _play(_refund_card(energy_gain=2, damage=14))
        legal = [refund, {"kind": "end_turn", "action_id": "end_turn"}]
        cls = env._classify_refund_followup(refund, energy=1.0, legal_actions=legal, raw_obs={})
        self.assertEqual(cls, "refund_no_followup_but_intrinsic_value")

    def test_non_refund_returns_unknown(self):
        env = CombatSandboxEnv.__new__(CombatSandboxEnv)
        strike = _play(_card("Strike", card_type="Attack", cost=1, damage=6))
        cls = env._classify_refund_followup(strike, energy=1.0, legal_actions=[strike], raw_obs={})
        self.assertEqual(cls, "refund_unknown")


if __name__ == "__main__":
    unittest.main()
