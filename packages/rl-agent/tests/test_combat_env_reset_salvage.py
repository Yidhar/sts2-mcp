from __future__ import annotations

from pathlib import Path
import sys
import unittest


RL_AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(RL_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(RL_AGENT_ROOT))

from sts2_env.bridge_client import BridgeError
from sts2_env.combat_env import CombatSandboxEnv


class CombatEnvResetSalvageTest(unittest.TestCase):
    def _make_env(self, rebound_payload: dict[str, object]) -> tuple[CombatSandboxEnv, list[tuple[bool, int | None]]]:
        env = CombatSandboxEnv.__new__(CombatSandboxEnv)
        calls: list[tuple[bool, int | None]] = []

        class _Bridge:
            def reset(self, *, rebind_active_run: bool = False, timeout_ms: int | None = None):
                calls.append((bool(rebind_active_run), timeout_ms))
                return rebound_payload

        env.bridge = _Bridge()
        env.reset_timeout_ms = 4321
        return env, calls

    def test_settling_non_actionable_reset_error_rebinds_active_run(self) -> None:
        payload = {"obs": {}, "legal_actions": [], "info": {}}
        env, calls = self._make_env(payload)
        exc = BridgeError(
            "combat_sandbox_not_in_combat",
            status_code=409,
            response_body={
                "error": "combat_sandbox_not_in_combat",
                "details": {
                    "screen": "COMBAT",
                    "phase": "settling",
                    "actionable": False,
                    "combat_in_progress": True,
                },
            },
        )

        salvaged = CombatSandboxEnv._try_salvage_card_selection_reset(env, exc)
        self.assertIs(salvaged, payload)
        self.assertEqual(calls, [(True, 4321)])
        info = salvaged["info"]
        self.assertTrue(info["combat_reset_salvaged"])
        self.assertEqual(info["combat_reset_salvage_phase"], "settling")
        self.assertEqual(info["combat_reset_salvage_screen"], "COMBAT")
        self.assertFalse(info["combat_reset_salvage_actionable"])

    def test_non_actionable_non_settling_reset_error_is_not_salvaged(self) -> None:
        payload = {"obs": {}, "legal_actions": [], "info": {}}
        env, calls = self._make_env(payload)
        exc = BridgeError(
            "combat_sandbox_not_in_combat",
            status_code=409,
            response_body={
                "error": "combat_sandbox_not_in_combat",
                "details": {
                    "screen": "COMBAT",
                    "phase": "combat",
                    "actionable": False,
                    "combat_in_progress": True,
                },
            },
        )

        salvaged = CombatSandboxEnv._try_salvage_card_selection_reset(env, exc)
        self.assertIsNone(salvaged)
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
