from __future__ import annotations

import sys
import unittest
from pathlib import Path


RL_AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(RL_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(RL_AGENT_ROOT))

from sts2_env.observation_v3 import WorldTokenObservationEncoder
from sts2_env.semantic_action import (
    compact_semantic_signature,
    semantic_action_signature,
    semantic_action_text,
)


def _card(
    title: str,
    *,
    card_id: str | None = None,
    card_type: str = "Skill",
    cost: int | str = 0,
    upgrade_level: int = 0,
    operations: list[dict[str, object]] | None = None,
    semantic_tags: list[str] | None = None,
) -> dict[str, object]:
    return {
        "id": card_id or title.upper().replace(" ", "_"),
        "title": title,
        "type": card_type,
        "cost": cost,
        "upgrade_level": upgrade_level,
        "target": "Self",
        "description": "",
        "text": "",
        "can_play": True,
        "card_effect_profile": {
            "operations": operations or [],
            "semantic_tags": semantic_tags or [],
        },
    }


class SemanticActionCardEffectProfileTest(unittest.TestCase):
    def test_energy_generator_exposes_followup_requirement_without_text(self) -> None:
        action = {
            "kind": "play_card",
            "action_id": "play_card:0",
            "card": _card(
                "Production Like",
                operations=[
                    {
                        "op": "gain_energy",
                        "energy": 2,
                        "upgraded_energy": 3,
                        "timing": "same_turn_resource",
                        "strategic_skip_if_no_followup": True,
                    }
                ],
                semantic_tags=["energy_gen", "exhaust"],
            ),
        }

        signature = semantic_action_signature(action)
        text = semantic_action_text(signature)
        compact = compact_semantic_signature(signature)

        self.assertIn("resource", signature["roles"])
        self.assertTrue(signature["typed_requires_followup"])
        self.assertTrue(signature["typed_strategic_skip_if_no_followup"])
        self.assertEqual(signature["typed_gain_energy"], 2)
        self.assertIn("ops=gain_energy", text)
        self.assertIn("requires_followup", text)
        self.assertEqual(compact["typed_gain_energy"], 2)

    def test_hp_cost_energy_generator_exposes_hp_loss_and_followup(self) -> None:
        action = {
            "kind": "play_card",
            "action_id": "play_card:0",
            "card": _card(
                "Bloodletting Like",
                operations=[
                    {
                        "op": "gain_energy",
                        "energy": 2,
                        "hp_loss": 3,
                        "timing": "same_turn_resource",
                    }
                ],
                semantic_tags=["energy_gen"],
            ),
        }

        signature = semantic_action_signature(action)
        profile = WorldTokenObservationEncoder(use_text=False)._source_profile(action["card"])

        self.assertIn("resource", signature["roles"])
        self.assertEqual(signature["typed_gain_energy"], 2)
        self.assertEqual(signature["typed_hp_loss"], 3)
        self.assertEqual(signature["hp_loss"], 3)
        self.assertTrue(signature["typed_requires_followup"])
        self.assertEqual(profile["energy"], 2)
        self.assertEqual(profile["hp_loss"], 3)
        self.assertEqual(profile["strategic_skip_value"], 1)

    def test_bullet_time_like_cost_rule_marks_not_x_and_no_draw(self) -> None:
        action = {
            "kind": "play_card",
            "action_id": "play_card:0",
            "card": _card(
                "Bullet Time Like",
                operations=[
                    {
                        "op": "modify_cost",
                        "scope": "all",
                        "set_cost": 0,
                        "source_zone": "hand",
                        "duration": "this_turn",
                        "target_filter": ["not_x_cost"],
                    },
                    {
                        "op": "apply_power",
                        "power_id": "NoDrawPower",
                        "future_rule": "no_draw_this_turn",
                    },
                ],
                semantic_tags=["hand_mutation", "card_rule_modifier"],
            ),
        }

        signature = semantic_action_signature(action)
        text = semantic_action_text(signature)

        self.assertIn("setup", signature["roles"])
        self.assertTrue(signature["typed_modify_cost"])
        self.assertTrue(signature["typed_not_x_cost_filter"])
        self.assertTrue(signature["typed_no_draw"])
        self.assertTrue(signature["typed_requires_followup"])
        self.assertIn("modify_cost", text)
        self.assertIn("not_x_cost", text)
        self.assertIn("no_draw", text)

    def test_armaments_like_upgrade_is_generic_hand_mutation(self) -> None:
        action = {
            "kind": "play_card",
            "action_id": "play_card:0",
            "card": _card(
                "Practice Smith",
                cost=1,
                operations=[
                    {
                        "op": "upgrade_card",
                        "source_zone": "hand",
                        "selection": "choice",
                        "scope": "one",
                        "target_filter": ["is_upgradable"],
                    }
                ],
                semantic_tags=["hand_mutation", "upgrade_card"],
            ),
        }

        signature = semantic_action_signature(action)

        self.assertIn("setup", signature["roles"])
        self.assertTrue(signature["typed_modifies_hand"])
        self.assertTrue(signature["typed_upgrade_hand"])
        self.assertTrue(signature["typed_hand_context_dependency"])
        self.assertTrue(signature["typed_card_state_mutation"])

    def test_observation_source_profile_uses_typed_energy_and_skip_flags(self) -> None:
        card = _card(
            "Production Like",
            operations=[
                {
                    "op": "gain_energy",
                    "energy": 2,
                    "timing": "same_turn_resource",
                    "strategic_skip_if_no_followup": True,
                }
            ],
            semantic_tags=["energy_gen", "exhaust"],
        )

        profile = WorldTokenObservationEncoder(use_text=False)._source_profile(card)

        self.assertEqual(profile["energy"], 2)
        self.assertEqual(profile["typed_gain_energy_amount"], 2)
        self.assertEqual(profile["typed_requires_followup"], 1)
        self.assertEqual(profile["strategic_skip_value"], 1)

    def test_observation_source_profile_uses_typed_cost_rule_and_no_draw(self) -> None:
        card = _card(
            "Bullet Time Like",
            operations=[
                {
                    "op": "modify_cost",
                    "scope": "all",
                    "set_cost": 0,
                    "source_zone": "hand",
                    "duration": "this_turn",
                    "target_filter": ["not_x_cost"],
                },
                {
                    "op": "apply_power",
                    "power_id": "NoDrawPower",
                    "future_rule": "no_draw_this_turn",
                },
            ],
            semantic_tags=["hand_mutation", "card_rule_modifier"],
        )

        profile = WorldTokenObservationEncoder(use_text=False)._source_profile(card)

        self.assertEqual(profile["typed_modify_cost"], 1)
        self.assertEqual(profile["typed_not_x_cost_filter"], 1)
        self.assertEqual(profile["typed_no_draw"], 1)
        self.assertEqual(profile["typed_requires_followup"], 1)
        self.assertEqual(profile["strategic_skip_value"], 1)


if __name__ == "__main__":
    unittest.main()
