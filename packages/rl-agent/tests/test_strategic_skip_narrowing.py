"""Tests for the narrowed strategic-skip detector (TASK-B2).

Verifies that pure exhaust/retain alone (without an explicit future-reason
signal) is no longer flagged as strategic_skip, and that lethal/high-impact
exhaust attacks are never skipped.
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
    cost: int | str = 0,
    damage: int = 0,
    block: int = 0,
    exhaust: bool = False,
    retain: bool = False,
    ethereal: bool = False,
    operations: list[dict[str, object]] | None = None,
    semantic_tags: list[str] | None = None,
) -> dict[str, object]:
    return {
        "id": title.upper().replace(" ", "_"),
        "title": title,
        "name": title,
        "type": card_type,
        "cost": cost,
        "damage": damage,
        "block": block,
        "preview_damage": damage,
        "preview_block": block,
        "exhaust": exhaust,
        "retain": retain,
        "ethereal": ethereal,
        "description": "",
        "text": "",
        "can_play": True,
        "card_effect_profile": {
            "operations": operations or [],
            "semantic_tags": semantic_tags or [],
            "training_tags": [],
        },
    }


def _play(card: dict[str, object]) -> dict[str, object]:
    return {
        "kind": "play_card",
        "action_id": f"play:{card['id']}",
        "card": card,
    }


class StrategicSkipNarrowingTests(unittest.TestCase):
    def test_pure_exhaust_attack_is_not_a_skip_candidate(self):
        # An exhaust attack with no future-reason signal must not be flagged —
        # otherwise the model gets trained to fear cards like Reaper / Feed.
        env = CombatSandboxEnv.__new__(CombatSandboxEnv)
        reaper_like = _card(
            "Reaper Like",
            card_type="Attack",
            cost=2,
            damage=4,
            exhaust=True,
        )
        action = _play(reaper_like)
        legal = [action, {"kind": "end_turn", "action_id": "end_turn"}]
        self.assertFalse(env._is_strategic_skip_candidate(action, energy=2.0, legal_actions=legal))

    def test_lethal_exhaust_attack_is_not_a_skip_candidate(self):
        # Even when the immediate score is moderate, a high-damage exhaust kills
        # are clearly not "skip candidates".  Threshold-based check confirms
        # that scoring 6+ damage prevents the skip flag.
        env = CombatSandboxEnv.__new__(CombatSandboxEnv)
        finisher = _card(
            "Finisher",
            card_type="Attack",
            cost=1,
            damage=12,
            exhaust=True,
        )
        legal = [_play(finisher), {"kind": "end_turn", "action_id": "end_turn"}]
        self.assertFalse(env._is_strategic_skip_candidate(legal[0], energy=1.0, legal_actions=legal))

    def test_low_value_exhaust_with_hand_mutation_is_a_skip_candidate(self):
        # Low-immediate-value exhaust card whose typed profile flags hand
        # mutation as a future-reason signal — this is the legitimate skip case.
        env = CombatSandboxEnv.__new__(CombatSandboxEnv)
        purity_like = _card(
            "Purity Like",
            card_type="Skill",
            cost=1,
            damage=0,
            block=0,
            exhaust=True,
            operations=[
                {
                    "op": "exhaust_cards",
                    "amount": 5,
                    "timing": "same_turn",
                }
            ],
            semantic_tags=["card_state_mutation", "exhaust"],
        )
        action = _play(purity_like)
        legal = [action, {"kind": "end_turn", "action_id": "end_turn"}]
        self.assertTrue(env._is_strategic_skip_candidate(action, energy=1.0, legal_actions=legal))

    def test_pure_retain_low_value_card_is_not_a_skip_candidate(self):
        # Retain alone (without future setup/replay signal) should fall through —
        # holding the card to next turn is the player's call, not a "skip".
        env = CombatSandboxEnv.__new__(CombatSandboxEnv)
        retain_card = _card(
            "Retain Like",
            card_type="Skill",
            cost=1,
            block=4,
            retain=True,
        )
        action = _play(retain_card)
        legal = [action, {"kind": "end_turn", "action_id": "end_turn"}]
        self.assertFalse(env._is_strategic_skip_candidate(action, energy=1.0, legal_actions=legal))

    def test_refund_with_followup_is_not_a_skip_candidate(self):
        # Refund + followup playable card → not a skip candidate.  This is the
        # B3 acceptance shape but lives in the same test module to demonstrate
        # the wider exhaust-rule narrowing does not break the refund path.
        env = CombatSandboxEnv.__new__(CombatSandboxEnv)
        refund = _card(
            "Refund Like",
            card_type="Skill",
            cost=1,
            operations=[
                {"op": "gain_energy", "energy": 2, "timing": "same_turn_resource"}
            ],
            semantic_tags=["energy_gen"],
        )
        strike = _card("Strike", card_type="Attack", cost=1, damage=6)
        legal = [_play(refund), _play(strike), {"kind": "end_turn", "action_id": "end_turn"}]
        self.assertFalse(env._is_strategic_skip_candidate(legal[0], energy=1.0, legal_actions=legal))


if __name__ == "__main__":
    unittest.main()
