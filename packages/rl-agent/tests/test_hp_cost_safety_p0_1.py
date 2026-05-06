"""Tests for the HP-cost / self-lethal hard-safety helper (P0-1)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any


RL_AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(RL_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(RL_AGENT_ROOT))

from sts2_env.hp_cost_safety import (
    hp_cost_safety_view,
    is_low_hp_margin_action,
    is_self_lethal_action,
)


def _action(*, hp_loss: float = 0.0, max_hp_loss: float = 0.0, safety: dict[str, Any] | None = None) -> dict[str, Any]:
    card: dict[str, Any] = {
        "id": "TEST",
        "title": "Test",
        "type": "Skill",
        "cost": 1,
    }
    if hp_loss:
        card["effect_preview"] = {"hp_loss": hp_loss}
        card["semantic_signals"] = {"self_damage": hp_loss}
    if max_hp_loss:
        card.setdefault("effect_preview", {})["max_hp_loss"] = max_hp_loss
    action: dict[str, Any] = {
        "kind": "play_card",
        "action_id": "play:test",
        "card": card,
    }
    if safety is not None:
        action["safety"] = safety
    return action


def _obs(hp: float, *, block: float = 0.0, max_hp: float = 80.0) -> dict[str, Any]:
    return {"player": {"hp": hp, "max_hp": max_hp, "block": block}}


class HpCostSafetyViewFallback(unittest.TestCase):
    def test_no_hp_cost_no_threat(self):
        view = hp_cost_safety_view(_action(), _obs(40))
        self.assertEqual(view["hp_cost_kind"], "none")
        self.assertFalse(view["self_lethal_now"])
        self.assertFalse(view["low_hp_margin_after_cost"])
        self.assertEqual(view["source_confidence"], "none")

    def test_hp1_hp_cost3_self_lethal(self):
        view = hp_cost_safety_view(_action(hp_loss=3.0), _obs(1))
        self.assertTrue(view["self_lethal_now"])
        self.assertEqual(view["hp_after_self_cost"], 0.0)

    def test_hp3_hp_cost3_self_lethal(self):
        view = hp_cost_safety_view(_action(hp_loss=3.0), _obs(3))
        self.assertTrue(view["self_lethal_now"])

    def test_hp4_hp_cost3_low_margin_not_lethal(self):
        view = hp_cost_safety_view(_action(hp_loss=3.0), _obs(4))
        self.assertFalse(view["self_lethal_now"])
        self.assertTrue(view["low_hp_margin_after_cost"])
        self.assertEqual(view["hp_after_self_cost"], 1.0)

    def test_block_does_not_save_unblockable_hp_cost(self):
        # 2 HP + 99 block + hp_cost 3 → block does NOT soak unblockable
        # cardHpLoss / nonCardHpLoss per STS2 source.  Action must be lethal.
        view = hp_cost_safety_view(_action(hp_loss=3.0), _obs(2, block=99))
        self.assertTrue(view["self_lethal_now"])

    def test_max_hp_loss_separated_from_hp_cost(self):
        view = hp_cost_safety_view(_action(max_hp_loss=2.0), _obs(40))
        self.assertEqual(view["max_hp_loss"], 2.0)
        self.assertEqual(view["hp_loss_unblockable"], 0.0)
        self.assertFalse(view["self_lethal_now"])
        self.assertEqual(view["hp_cost_kind"], "max_hp_loss")


class HpCostSafetyViewBridgePayload(unittest.TestCase):
    def test_bridge_payload_marks_lethal_directly(self):
        action = _action(safety={
            "hp_cost_kind": "unblockable_hp_loss",
            "hp_loss_unblockable": 5.0,
            "self_damage_blockable": 0.0,
            "max_hp_loss": 0.0,
            "hp_before": 4.0,
            "source_confidence": "runtime_internal",
        })
        view = hp_cost_safety_view(action, _obs(4))
        self.assertTrue(view["self_lethal_now"])
        self.assertEqual(view["source_confidence"], "runtime_internal")
        self.assertEqual(view["hp_after_self_cost"], 0.0)

    def test_bridge_blockable_self_damage_uses_block(self):
        # Bridge tags some self-damage as blockable; current block soaks it.
        action = _action(safety={
            "hp_cost_kind": "blockable_self_damage",
            "hp_loss_unblockable": 0.0,
            "self_damage_blockable": 4.0,
            "hp_before": 5.0,
        })
        view = hp_cost_safety_view(action, _obs(5, block=10))
        self.assertFalse(view["self_lethal_now"])
        self.assertEqual(view["hp_after_self_cost"], 5.0)


class IsSelfLethalActionGuard(unittest.TestCase):
    def test_guard_returns_true_for_lethal(self):
        self.assertTrue(is_self_lethal_action(_action(hp_loss=3.0), _obs(2)))

    def test_guard_returns_false_for_safe(self):
        self.assertFalse(is_self_lethal_action(_action(hp_loss=3.0), _obs(20)))

    def test_guard_returns_false_for_no_action(self):
        self.assertFalse(is_self_lethal_action(None, _obs(20)))

    def test_guard_returns_false_for_no_obs(self):
        # No HP info → can't classify → safer default is False (don't mask).
        self.assertFalse(is_self_lethal_action(_action(hp_loss=3.0), None))


class LowMarginHelper(unittest.TestCase):
    def test_low_margin_when_close_to_zero(self):
        self.assertTrue(is_low_hp_margin_action(_action(hp_loss=3.0), _obs(4)))

    def test_not_low_margin_when_lethal(self):
        # Lethal takes precedence; low_margin returns False.
        self.assertFalse(is_low_hp_margin_action(_action(hp_loss=3.0), _obs(2)))


class CombatEnvActionMaskIntegration(unittest.TestCase):
    def test_action_masks_drops_self_lethal(self):
        from sts2_env.combat_env import CombatSandboxEnv

        env = CombatSandboxEnv.__new__(CombatSandboxEnv)
        env._legal_actions = [
            _action(hp_loss=3.0),  # lethal at HP=2
            {"kind": "end_turn", "action_id": "end_turn"},
            _action(),
        ]
        env._last_obs_raw = _obs(2)
        mask = env.action_masks()
        # Index 0 must be masked, indices 1 and 2 must remain True.
        self.assertFalse(bool(mask[0]))
        self.assertTrue(bool(mask[1]))
        self.assertTrue(bool(mask[2]))

    def test_action_masks_keeps_non_lethal_hp_cost(self):
        from sts2_env.combat_env import CombatSandboxEnv

        env = CombatSandboxEnv.__new__(CombatSandboxEnv)
        env._legal_actions = [_action(hp_loss=3.0)]
        env._last_obs_raw = _obs(50)
        mask = env.action_masks()
        self.assertTrue(bool(mask[0]))


if __name__ == "__main__":
    unittest.main()
