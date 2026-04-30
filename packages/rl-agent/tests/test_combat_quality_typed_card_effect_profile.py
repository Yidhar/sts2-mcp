from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


RL_AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(RL_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(RL_AGENT_ROOT))

from muzero.train import MuZeroTrainer
from sts2_env.combat_env import CombatSandboxEnv


def _card(
    title: str,
    *,
    card_type: str = "Skill",
    cost: int | str = 0,
    damage: int = 0,
    block: int = 0,
    x_cost: bool = False,
    operations: list[dict[str, object]] | None = None,
    semantic_tags: list[str] | None = None,
    training_tags: list[str] | None = None,
) -> dict[str, object]:
    return {
        "id": title.upper().replace(" ", "_"),
        "title": title,
        "name": title,
        "type": card_type,
        "cost": cost,
        "x_cost": x_cost,
        "costs_x": x_cost,
        "damage": damage,
        "block": block,
        "preview_damage": damage,
        "preview_block": block,
        "description": "",
        "text": "",
        "can_play": True,
        "card_effect_profile": {
            "operations": operations or [],
            "semantic_tags": semantic_tags or [],
            "training_tags": training_tags or [],
        },
    }


def _play(card: dict[str, object], *, action_id: str | None = None) -> dict[str, object]:
    return {
        "kind": "play_card",
        "action_id": action_id or f"play:{card['id']}",
        "card": card,
    }


def _obs(*, energy: float) -> dict[str, object]:
    return {
        "phase": "combat",
        "player": {"hp": 50, "max_hp": 80, "block": 0, "relics": [], "potions": []},
        "combat": {
            "energy": energy,
            "max_energy": 3,
            "round": 1,
            "hand": [],
            "draw_pile": [],
            "discard_pile": [],
            "exhaust_pile": [],
            "enemies": [],
        },
    }


class CombatQualityTypedCardEffectProfileTest(unittest.TestCase):
    def test_combat_env_marks_typed_energy_refund_without_followup_as_strategic_skip(self) -> None:
        env = CombatSandboxEnv.__new__(CombatSandboxEnv)
        production_like = _card(
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
        action = _play(production_like)
        legal_actions = [action, {"kind": "end_turn", "action_id": "end_turn"}]

        self.assertTrue(env._is_strategic_skip_candidate(action, energy=0.0, legal_actions=legal_actions))

        diagnostics = env._action_quality_diagnostics(_obs(energy=0.0), legal_actions, action)
        self.assertEqual(diagnostics["strategic_skip_candidate_count"], 1.0)
        self.assertEqual(diagnostics["refund_no_followup_available"], 1.0)
        self.assertEqual(diagnostics["refund_no_followup_selected"], 1.0)
        self.assertEqual(diagnostics["typed_followup_missing_count"], 1.0)
        self.assertEqual(diagnostics["setup_followup_dependent_count"], 1.0)
        self.assertEqual(diagnostics["setup_followup_available_count"], 0.0)

    def test_combat_env_allows_typed_energy_refund_when_followup_can_spend_it(self) -> None:
        env = CombatSandboxEnv.__new__(CombatSandboxEnv)
        production_like = _card(
            "Production Like",
            operations=[
                {
                    "op": "gain_energy",
                    "energy": 2,
                    "timing": "same_turn_resource",
                    "strategic_skip_if_no_followup": True,
                }
            ],
            semantic_tags=["energy_gen"],
        )
        strike = _card("Strike", card_type="Attack", cost=1, damage=6)
        action = _play(production_like)
        legal_actions = [action, _play(strike), {"kind": "end_turn", "action_id": "end_turn"}]

        self.assertFalse(env._is_strategic_skip_candidate(action, energy=0.0, legal_actions=legal_actions))

        diagnostics = env._action_quality_diagnostics(_obs(energy=0.0), legal_actions, action)
        self.assertEqual(diagnostics["strategic_skip_candidate_count"], 0.0)
        self.assertEqual(diagnostics["refund_no_followup_available"], 0.0)
        self.assertEqual(diagnostics["typed_followup_missing_count"], 0.0)
        self.assertEqual(diagnostics["setup_followup_available_count"], 1.0)

    def test_muzero_metrics_read_typed_energy_draw_hp_and_keyword_flags(self) -> None:
        energy_card = _play(
            _card(
                "Bloodletting Like",
                operations=[
                    {
                        "op": "gain_energy",
                        "energy": 2,
                        "hp_loss": 3,
                        "timing": "same_turn_resource",
                    },
                    {"op": "draw_card", "count": 1},
                    {"op": "exhaust_card", "source_zone": "hand", "selection": "self"},
                    {"op": "retain_card", "source_zone": "hand", "selection": "self"},
                ],
            )
        )
        trainer = MuZeroTrainer.__new__(MuZeroTrainer)

        self.assertEqual(trainer._action_metric(energy_card, "energy"), 2.0)
        self.assertEqual(trainer._action_metric(energy_card, "draw"), 1.0)
        self.assertEqual(trainer._action_metric(energy_card, "hp_loss"), 3.0)
        self.assertTrue(trainer._is_exhausting_action(energy_card))
        self.assertTrue(trainer._is_retain_action(energy_card))

    def test_muzero_classifies_unconverted_refund_as_deferable_not_urgent(self) -> None:
        trainer = MuZeroTrainer.__new__(MuZeroTrainer)
        action = _play(
            _card(
                "Production Like",
                operations=[
                    {
                        "op": "gain_energy",
                        "energy": 2,
                        "timing": "same_turn_resource",
                        "strategic_skip_if_no_followup": True,
                    }
                ],
                semantic_tags=["energy_gen"],
            )
        )
        legal_actions = [action, {"kind": "end_turn", "action_id": "end_turn"}]
        result = trainer._classify_positive_combat_action(
            action,
            0,
            None,
            None,
            legal_actions,
            np.ones(2, dtype=np.float32),
            0.0,
        )

        self.assertTrue(result["positive"])
        self.assertFalse(result["urgent"])
        self.assertTrue(result["deferable"])
        self.assertTrue(result["energy_without_followup"])
        self.assertTrue(result["followup_missing"])

    def test_muzero_classifies_no_draw_cost_reducer_without_followup_as_deferable(self) -> None:
        trainer = MuZeroTrainer.__new__(MuZeroTrainer)
        action = _play(
            _card(
                "Bullet Time Like",
                cost=2,
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
        )
        legal_actions = [action, {"kind": "end_turn", "action_id": "end_turn"}]
        result = trainer._classify_positive_combat_action(
            action,
            0,
            None,
            None,
            legal_actions,
            np.ones(2, dtype=np.float32),
            3.0,
        )

        self.assertTrue(result["positive"])
        self.assertFalse(result["urgent"])
        self.assertTrue(result["deferable"])
        self.assertTrue(result["typed_modify_cost"])
        self.assertTrue(result["typed_no_draw"])
        self.assertTrue(result["typed_future_penalty"])
        self.assertTrue(result["followup_missing"])

    def test_muzero_x_cost_is_dynamic_on_current_energy(self) -> None:
        trainer = MuZeroTrainer.__new__(MuZeroTrainer)
        action = _play(_card("Whirlwind Like", card_type="Attack", cost="X", x_cost=True, damage=0))
        legal_actions = [action, {"kind": "end_turn", "action_id": "end_turn"}]
        mask = np.ones(2, dtype=np.float32)

        zero_energy = trainer._classify_positive_combat_action(action, 0, None, None, legal_actions, mask, 0.0)
        two_energy = trainer._classify_positive_combat_action(action, 0, None, None, legal_actions, mask, 2.0)

        self.assertTrue(zero_energy["x_cost_zero"])
        self.assertTrue(zero_energy["deferable"])
        self.assertFalse(two_energy["x_cost_zero"])


if __name__ == "__main__":
    unittest.main()
