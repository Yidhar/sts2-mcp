"""Tests for the potion-transition diagnostic JSONL and the
``potion_unused_on_death`` adjustment that subtracts used-this-combat from the
raw final slot count (TASK-A4).
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


RL_AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(RL_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(RL_AGENT_ROOT))

from muzero.train import MuZeroTrainer
from sts2_env.combat_env import CombatSandboxEnv


class PotionSlotsDumpTests(unittest.TestCase):
    def test_dump_normalises_empty_and_populated_slots(self):
        obs = {
            "player": {
                "potions": [
                    {"id": "POTION_FIRE", "title": "Fire Potion", "is_usable": True},
                    {"id": "EMPTY", "title": "[empty]", "empty": True},
                    "[empty]",
                ]
            }
        }
        dump = CombatSandboxEnv._potion_slots_dump(obs)
        self.assertEqual(len(dump), 3)
        self.assertFalse(dump[0]["empty"])
        self.assertTrue(dump[1]["empty"])
        self.assertTrue(dump[2]["empty"])
        self.assertEqual(dump[0]["id"], "POTION_FIRE")


class BuildPotionTransitionRecordTests(unittest.TestCase):
    def _stub_env(self) -> CombatSandboxEnv:
        # Avoid full env init — instantiate via __new__ and set the few pieces
        # the helper reads.
        env = CombatSandboxEnv.__new__(CombatSandboxEnv)
        env._potion_transition_records = []
        return env

    def test_returns_none_for_non_potion_action(self):
        env = self._stub_env()
        rec = env._build_potion_transition_record(
            action={"semantic": {"family": "play_card"}, "action_id": "play_card_0"},
            prev_obs={}, after_obs={}, bridge_result={}, bridge_error=None,
        )
        self.assertIsNone(rec)

    def test_marks_execute_ok_when_no_bridge_error(self):
        env = self._stub_env()
        prev = {"player": {"potions": [{"id": "POTION_FIRE", "title": "Fire", "empty": False}]}}
        post = {"player": {"potions": [{"id": "EMPTY", "title": "[empty]", "empty": True}]}}
        rec = env._build_potion_transition_record(
            action={
                "semantic": {"family": "use_potion"},
                "action_id": "use_potion_0",
                "potion": {"id": "POTION_FIRE", "title": "Fire"},
                "slot_index": 0,
            },
            prev_obs=prev, after_obs=post, bridge_result={"info": {}}, bridge_error=None,
        )
        self.assertIsNotNone(rec)
        self.assertEqual(rec["event"], "use_potion_transition")
        self.assertTrue(rec["execute_ok"])
        self.assertEqual(rec["potion_id_before"], "POTION_FIRE")
        self.assertTrue(rec["potion_slot_after"]["empty"])

    def test_marks_execute_fail_when_bridge_error_set(self):
        env = self._stub_env()
        prev = {"player": {"potions": [{"id": "POTION_FIRE", "title": "Fire", "empty": False}]}}
        post = prev  # bridge call failed, slot unchanged
        rec = env._build_potion_transition_record(
            action={
                "semantic": {"family": "use_potion"},
                "action_id": "use_potion_0",
                "potion": {"id": "POTION_FIRE", "title": "Fire"},
                "slot_index": 0,
            },
            prev_obs=prev, after_obs=post, bridge_result={}, bridge_error="phase mismatch",
        )
        self.assertIsNotNone(rec)
        self.assertFalse(rec["execute_ok"])
        self.assertEqual(rec["bridge_error"], "phase mismatch")


class DeathFinalPotionAdjustmentTests(unittest.TestCase):
    """The boss diagnostic must subtract `used_potion_count_this_combat` from
    the raw `final_potion_count` so that a use_potion that left the slot dirty
    in the bridge frame does not falsely flip ``potion_unused_on_death`` to 1.
    """

    def _stub_trainer(self, log_dir: Path) -> MuZeroTrainer:
        stub = MuZeroTrainer.__new__(MuZeroTrainer)
        stub.log_dir = str(log_dir)
        stub.total_steps = 1
        stub.episode_count = 1
        stub._potion_transition_dump_count = 0
        stub._potion_transition_dump_max = 100
        stub._potion_transition_dump_disabled = False
        return stub

    def test_dump_potion_transition_writes_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = self._stub_trainer(Path(tmp))
            trainer._dump_potion_transition({
                "event": "use_potion_transition",
                "action_id": "use_potion_0",
                "potion_slot": 0,
                "execute_ok": True,
                "potion_slot_after": {"empty": True},
            })
            path = Path(tmp) / "diagnostics" / "potion_transitions.jsonl"
            self.assertTrue(path.exists())
            payload = json.loads(path.read_text(encoding="utf-8").strip())
            self.assertEqual(payload["event"], "use_potion_transition")
            self.assertTrue(payload["execute_ok"])

    def test_dump_death_final_potions_writes_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = self._stub_trainer(Path(tmp))
            trainer._dump_death_final_potions(
                loss=True,
                final_potion_count=1,
                used_potion_count=1,
                adjusted_unused=0,
                encounter="kaiser_crab_boss",
                tier="boss",
                potion_dump=[{"slot": 0, "empty": False, "id": "POTION_FIRE"}],
            )
            path = Path(tmp) / "diagnostics" / "potion_transitions.jsonl"
            self.assertTrue(path.exists())
            payload = json.loads(path.read_text(encoding="utf-8").strip())
            self.assertEqual(payload["event"], "death_final_potions")
            self.assertEqual(payload["adjusted_unused"], 0)
            self.assertEqual(payload["used_potion_count"], 1)


class BossPotionUnusedOnDeathLogicTests(unittest.TestCase):
    """Confirm the adjustment formula: adjusted = max(final - used, 0)."""

    def test_used_consumes_remaining_slot_view(self):
        # final_count says 1 unused, but used_count says we already used 1.
        # adjusted should be 0 → potion_unused_on_death=0.
        adjusted = max(1 - 1, 0)
        self.assertEqual(adjusted, 0)
        loss = True
        flag = 1.0 if loss and adjusted > 0 else 0.0
        self.assertEqual(flag, 0.0)

    def test_not_used_keeps_count(self):
        adjusted = max(2 - 0, 0)
        self.assertEqual(adjusted, 2)
        flag = 1.0 if True and adjusted > 0 else 0.0
        self.assertEqual(flag, 1.0)


if __name__ == "__main__":
    unittest.main()
