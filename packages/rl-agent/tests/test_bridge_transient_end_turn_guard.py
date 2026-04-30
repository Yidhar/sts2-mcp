"""Tests for the C1 actionability payload contract and the C2 short-poll
helper.  Verifies the four documented scenarios:

1. queue pending + only end_turn → transient_only_end_turn=True
2. no pending + only end_turn → stable_no_actions
3. non-end_turn legal action present → frontier_stable=True
4. fake bridge first transient → second non-end_turn => helper returns 2nd
   frame; stuck-transient → timeout flag set; stable from start → no waiting.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any


RL_AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(RL_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(RL_AGENT_ROOT))

from sts2_env.combat_env import CombatSandboxEnv


def _step_result(*, transient: bool, non_end_turn_count: int = 0, frontier_stable: bool | None = None) -> dict[str, Any]:
    if frontier_stable is None:
        frontier_stable = (not transient) and non_end_turn_count >= 0
    return {
        "ok": True,
        "info": {
            "actionability": {
                "frontier_stable": frontier_stable,
                "transient_only_end_turn": transient,
                "only_end_turn_reason": "queue_pending" if transient else (
                    "has_non_end_turn_actions" if non_end_turn_count > 0 else "stable_no_actions"
                ),
                "queue_pending": transient,
                "animation_pending": transient,
                "draw_shuffle_pending": transient,
                "legal_non_end_turn_count": non_end_turn_count,
                "legal_action_count": non_end_turn_count + 1,
                "state_version": 0,
                "state_hash": "hash",
            }
        },
    }


class IsStableActionabilityTests(unittest.TestCase):
    def test_non_end_turn_actions_present_is_stable(self):
        a = _step_result(transient=False, non_end_turn_count=3)["info"]["actionability"]
        self.assertTrue(CombatSandboxEnv._is_stable_actionability(a))

    def test_transient_only_end_turn_is_not_stable(self):
        a = _step_result(transient=True)["info"]["actionability"]
        self.assertFalse(CombatSandboxEnv._is_stable_actionability(a))

    def test_stable_no_actions_is_stable(self):
        a = _step_result(transient=False, non_end_turn_count=0)["info"]["actionability"]
        self.assertTrue(CombatSandboxEnv._is_stable_actionability(a))


class WaitForStableActionabilityTests(unittest.TestCase):
    def test_first_frame_stable_no_polling(self):
        polls: list[int] = []
        def observe():
            polls.append(1)
            return _step_result(transient=False, non_end_turn_count=2)
        result, metrics = CombatSandboxEnv.wait_for_stable_actionability(
            _step_result(transient=False, non_end_turn_count=2),
            observe,
            max_wait_ms=100, poll_interval_ms=10,
            sleep_fn=lambda _: None, clock_fn=lambda: 0.0,
        )
        self.assertEqual(metrics["poll_count"], 0)
        self.assertFalse(metrics["timeout"])
        self.assertFalse(metrics["transient_leaked"])

    def test_first_frame_stable_no_actions_records_metric(self):
        result, metrics = CombatSandboxEnv.wait_for_stable_actionability(
            _step_result(transient=False, non_end_turn_count=0),
            lambda: _step_result(transient=False, non_end_turn_count=0),
            max_wait_ms=100, poll_interval_ms=10,
            sleep_fn=lambda _: None, clock_fn=lambda: 0.0,
        )
        self.assertTrue(metrics["stable_no_actions"])
        self.assertEqual(metrics["poll_count"], 0)

    def test_transient_resolves_on_second_poll(self):
        # First observe still transient, second observe shows non-end-turn.
        sequence = iter([
            _step_result(transient=True),
            _step_result(transient=False, non_end_turn_count=2),
        ])
        clock_t = [0.0]
        def clock(): return clock_t[0]
        def sleep(s):
            clock_t[0] += s
        result, metrics = CombatSandboxEnv.wait_for_stable_actionability(
            _step_result(transient=True),
            lambda: next(sequence),
            max_wait_ms=200, poll_interval_ms=10,
            sleep_fn=sleep, clock_fn=clock,
        )
        self.assertTrue(metrics["transient_resolved"])
        self.assertFalse(metrics["timeout"])
        self.assertGreaterEqual(metrics["poll_count"], 1)

    def test_stuck_transient_triggers_timeout(self):
        clock_t = [0.0]
        def clock(): return clock_t[0]
        def sleep(s):
            clock_t[0] += s
        result, metrics = CombatSandboxEnv.wait_for_stable_actionability(
            _step_result(transient=True),
            lambda: _step_result(transient=True),
            max_wait_ms=50, poll_interval_ms=10,
            sleep_fn=sleep, clock_fn=clock,
        )
        self.assertTrue(metrics["timeout"])
        self.assertTrue(metrics["transient_leaked"])
        # Bounded — must not loop forever.
        self.assertLessEqual(metrics["poll_count"], 100)

    def test_observe_exception_falls_back_without_hang(self):
        def observe(): raise RuntimeError("bridge dropped")
        clock_t = [0.0]
        def clock(): return clock_t[0]
        def sleep(s): clock_t[0] += s
        result, metrics = CombatSandboxEnv.wait_for_stable_actionability(
            _step_result(transient=True),
            observe,
            max_wait_ms=50, poll_interval_ms=10,
            sleep_fn=sleep, clock_fn=clock,
        )
        # Exception breaks the loop early; must not hang.
        self.assertGreaterEqual(metrics["poll_count"], 1)


if __name__ == "__main__":
    unittest.main()
