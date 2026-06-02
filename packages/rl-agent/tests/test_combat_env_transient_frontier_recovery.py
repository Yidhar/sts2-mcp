"""Production combat sandbox post-step frontier recovery tests.

These tests pin the distinction that matters operationally:

* leftover energy + empty hand + only End Turn is **not** proof of a bug;
* a short /state poll that rebounds into non-EndTurn actions **is** proof the
  immediate bridge.step frontier was transient;
* a bridge-declared transient frame that never rebounds is tracked separately
  as a leak/timeout.
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


END_TURN = {"action_id": "end_turn", "kind": "end_turn", "title": "End Turn"}
STRIKE = {"action_id": "play_card:0:0", "kind": "play_card", "title": "打击"}


class FakeBridge:
    def __init__(
        self,
        *,
        states: list[dict[str, Any]] | None = None,
        rebind_result: dict[str, Any] | None = None,
        step_result: dict[str, Any] | None = None,
        step_raises: bool = False,
    ) -> None:
        self.states = list(states or [])
        self.rebind_result = rebind_result
        self.step_result = step_result
        self.step_raises = bool(step_raises)
        self.get_state_calls = 0
        self.reset_calls: list[dict[str, Any]] = []
        self.step_calls: list[dict[str, Any]] = []

    def get_state(self) -> dict[str, Any]:
        self.get_state_calls += 1
        if self.states:
            return self.states.pop(0)
        return {}

    def reset(self, **kwargs: Any) -> dict[str, Any] | None:
        self.reset_calls.append(dict(kwargs))
        return self.rebind_result

    def step(self, **kwargs: Any) -> dict[str, Any]:
        self.step_calls.append(dict(kwargs))
        if self.step_raises:
            raise AssertionError("bridge.step must not be called")
        if self.step_result is not None:
            return self.step_result
        return _step_result(energy=0, hand=[], legal_actions=[], reward=0.0)


def _env_stub(bridge: FakeBridge) -> CombatSandboxEnv:
    class _DummyObsEncoder:
        def encode(
            self,
            obs: dict[str, Any],
            legal_actions: list[dict[str, Any]],
            planner_context: dict[str, Any],
        ) -> dict[str, Any]:
            return {
                "encoded": True,
                "legal_count": len(legal_actions),
                "planner_context": dict(planner_context),
            }

    class _DummyRunMemory:
        def build_context(
            self,
            obs: dict[str, Any] | None,
            legal_actions: list[dict[str, Any]],
        ) -> dict[str, Any]:
            return {"legal_count": len(legal_actions)}

        def update_transition(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def reset(self, *_args: Any, **_kwargs: Any) -> None:
            return None

    class _DummyCombatMemory:
        def snapshot(self) -> dict[str, Any]:
            return {}

        def update(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def reset(self, *_args: Any, **_kwargs: Any) -> None:
            return None

    env = object.__new__(CombatSandboxEnv)
    env.bridge = bridge
    env.step_timeout_ms = 20_000
    env.reset_timeout_ms = 1000
    env.include_debug_info = False
    env.sandbox_supports_potions = False
    env.obs_encoder = _DummyObsEncoder()
    env._run_memory = _DummyRunMemory()
    env._combat_memory = _DummyCombatMemory()
    env._human_demo_recorder = None
    env.render_mode = None
    env._current_snapshot = {}
    env._current_encounter_id = "ENCOUNTER.TEST_NORMAL"
    env.encounter_pool = None
    env._last_action_overflow = 0
    env._legal_actions = []
    env._last_obs_raw = {}
    env._fast_step_disabled = False
    env._fast_step_max_wait_ms = 8
    env._fast_step_poll_interval_ms = 1
    env._episode_id = "episode-before"
    env._fast_step_metrics_total = {
        "transient_only_end_turn_count": 0,
        "transient_resolved_count": 0,
        "transient_leaked_count": 0,
        "wait_timeout_count": 0,
        "frontier_pre_dispatch_attempt_count": 0,
        "frontier_pre_dispatch_resolved_count": 0,
        "frontier_pre_dispatch_timeout_count": 0,
        "frontier_pre_dispatch_rebind_attempt_count": 0,
        "frontier_pre_dispatch_rebind_success_count": 0,
        "frontier_pre_dispatch_blocked_count": 0,
        "frontier_pre_dispatch_high_confidence_blocked_count": 0,
        "stable_no_actions_count": 0,
        "post_step_frontier_attempt_count": 0,
        "post_step_frontier_resolved_count": 0,
        "post_step_frontier_timeout_count": 0,
        "post_step_frontier_leaked_count": 0,
        "post_step_frontier_rebind_attempt_count": 0,
        "post_step_frontier_rebind_success_count": 0,
        "post_step_frontier_stable_no_actions_count": 0,
        "post_step_frontier_suspicious_singleton_count": 0,
    }
    env._last_actionability = None
    return env


def _step_result(
    *,
    energy: int = 3,
    hand: list[dict[str, Any]] | None = None,
    hand_count: int | None = None,
    actionability: dict[str, Any] | None = None,
    legal_actions: list[dict[str, Any]] | None = None,
    reward: float = 1.25,
) -> dict[str, Any]:
    combat: dict[str, Any] = {
        "in_progress": True,
        "energy": energy,
        "round": 1,
    }
    if hand is not None:
        combat["hand"] = hand
    if hand_count is not None:
        combat["hand_count"] = hand_count
    info: dict[str, Any] = {}
    if actionability is not None:
        info["actionability"] = actionability
    return {
        "episode_id": "episode-before",
        "reward": reward,
        "done": False,
        "truncated": False,
        "obs": {
            "combat": combat,
            "player": {"hp": 70, "block": 0, "energy": energy},
        },
        "legal_actions": list(legal_actions or [END_TURN]),
        "info": info,
    }


def _state_with_actions(actions: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "phase": "combat",
        "screen": "COMBAT",
        "run": {"active": True},
        "combat": {"in_progress": True, "energy": 3, "hand": [{"id": "CARD.STRIKE_R"}]},
        "player": {"hp": 70, "block": 0, "energy": 3},
        "available_actions": list(actions),
    }


class CombatPreDispatchEndTurnRecoveryTests(unittest.TestCase):
    def test_step_blocks_stale_singleton_end_turn_when_recovery_finds_non_end_turn(self) -> None:
        rebound = _step_result(
            energy=3,
            hand=[{"id": "CARD.STRIKE_R", "title": "打击", "type": "Attack", "cost": 1}],
            legal_actions=[STRIKE, END_TURN],
        )
        rebound["episode_id"] = "episode-after"
        rebound["info"] = {"actionability": {"frontier_stable": True, "legal_non_end_turn_count": 1}}
        bridge = FakeBridge(
            states=[_state_with_actions([STRIKE, END_TURN])],
            rebind_result=rebound,
            step_raises=True,
        )
        env = _env_stub(bridge)
        env._legal_actions = [END_TURN]
        env._last_actionability = {"frontier_stable": False, "legal_non_end_turn_count": 0}
        env._last_obs_raw = {
            "phase": "combat",
            "player": {"hp": 70, "max_hp": 80, "block": 0, "energy": 3},
            "combat": {
                "in_progress": True,
                "energy": 3,
                "hand": [{"id": "CARD.STRIKE_R", "title": "打击", "type": "Attack", "cost": 1}],
            },
        }

        obs, reward, terminated, truncated, info = env.step(0)

        self.assertEqual(len(bridge.step_calls), 0)
        self.assertEqual(len(bridge.reset_calls), 1)
        self.assertEqual(obs["legal_count"], 2)
        self.assertEqual(reward, 0.0)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertTrue(info["frontier_refreshed_before_end_turn"])
        self.assertEqual(
            info["bridge_info"]["action_diagnostics"]["frontier_pre_dispatch_end_turn_blocked"],
            1.0,
        )
        self.assertEqual(env._fast_step_metrics_total["frontier_pre_dispatch_attempt_count"], 1)
        self.assertEqual(env._fast_step_metrics_total["frontier_pre_dispatch_blocked_count"], 1)
        self.assertEqual(env._fast_step_metrics_total["frontier_pre_dispatch_rebind_success_count"], 1)

    def test_step_blocks_high_confidence_singleton_end_turn_when_raw_hand_affordable(self) -> None:
        bridge = FakeBridge(
            states=[_state_with_actions([END_TURN]) for _ in range(16)],
            rebind_result=None,
            step_raises=True,
        )
        env = _env_stub(bridge)
        env._legal_actions = [END_TURN]
        env._last_actionability = {"frontier_stable": True, "legal_non_end_turn_count": 0}
        env._last_obs_raw = {
            "phase": "combat",
            "player": {"hp": 70, "max_hp": 80, "block": 0, "energy": 1},
            "combat": {
                "in_progress": True,
                "energy": 1,
                "hand": [
                    {"id": "CARD.DEFEND_IRONCLAD", "title": "防御", "type": "Skill", "cost": 1},
                    {"id": "CARD.DEFEND_IRONCLAD", "title": "防御", "type": "Skill", "cost": 1},
                ],
            },
        }

        obs, reward, terminated, truncated, info = env.step(0)

        self.assertEqual(len(bridge.step_calls), 0)
        self.assertEqual(obs["legal_count"], 1)
        self.assertEqual(reward, 0.0)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertFalse(info["frontier_refreshed_before_end_turn"])
        self.assertTrue(info["frontier_stale_singleton_end_turn_blocked"])
        self.assertEqual(
            info["bridge_info"]["action_diagnostics"]["frontier_pre_dispatch_high_confidence"],
            1.0,
        )
        self.assertEqual(
            env._fast_step_metrics_total["frontier_pre_dispatch_high_confidence_blocked_count"],
            1,
        )

    def test_pre_dispatch_allows_singleton_end_turn_when_raw_hand_only_quest_status(self) -> None:
        bridge = FakeBridge(states=[_state_with_actions([END_TURN]) for _ in range(16)], rebind_result=None)
        env = _env_stub(bridge)
        env._legal_actions = [END_TURN]
        env._last_actionability = {"frontier_stable": True, "legal_non_end_turn_count": 0}
        env._last_obs_raw = {
            "phase": "combat",
            "player": {"hp": 70, "max_hp": 80, "block": 0, "energy": 2},
            "combat": {
                "in_progress": True,
                "energy": 2,
                "hand": [
                    {"id": "CARD.TREASURE_MAP", "title": "藏宝图", "type": "Quest", "cost": 0},
                    {"id": "CARD.DAZED", "title": "晕眩", "type": "Status", "cost": 0},
                ],
            },
        }

        response = env._recover_pre_dispatch_end_turn_frontier(
            attempted_action_index=0,
            selected_action=END_TURN,
        )

        self.assertIsNone(response)
        self.assertGreaterEqual(bridge.get_state_calls, 1)
        self.assertEqual(env._fast_step_metrics_total["frontier_pre_dispatch_blocked_count"], 0)
        self.assertFalse(env._frontier_has_affordable_raw_combat_card())


class CombatPostStepFrontierRecoveryTests(unittest.TestCase):
    def test_bridge_transient_rebounds_into_non_end_turn_and_preserves_step_reward(self) -> None:
        rebound = _step_result(
            energy=3,
            hand=[{"id": "CARD.STRIKE_R"}],
            legal_actions=[STRIKE, END_TURN],
            reward=0.0,
        )
        rebound["episode_id"] = "episode-after"
        bridge = FakeBridge(states=[_state_with_actions([STRIKE, END_TURN])], rebind_result=rebound)
        env = _env_stub(bridge)

        initial = _step_result(
            energy=3,
            hand=[],
            actionability={
                "transient_only_end_turn": True,
                "frontier_stable": False,
                "legal_non_end_turn_count": 0,
            },
            reward=7.5,
        )
        result, metrics = env._recover_post_step_frontier(
            initial,
            selected_action=END_TURN,
            legal_actions_before=[END_TURN],
        )

        self.assertTrue(metrics["attempted"])
        self.assertTrue(metrics["resolved"])
        self.assertTrue(metrics["rebind_attempted"])
        self.assertTrue(metrics["rebind_succeeded"])
        self.assertEqual(result["legal_actions"], [STRIKE, END_TURN])
        self.assertEqual(result["reward"], 7.5)  # rebind must not zero the executed action reward
        self.assertEqual(result["episode_id"], "episode-after")
        self.assertEqual(bridge.reset_calls[0]["rebind_active_run"], True)
        self.assertEqual(env._fast_step_metrics_total["post_step_frontier_resolved_count"], 1)

    def test_stable_only_end_turn_with_leftover_energy_does_not_poll_or_claim_bug(self) -> None:
        bridge = FakeBridge(states=[_state_with_actions([STRIKE, END_TURN])])
        env = _env_stub(bridge)

        initial = _step_result(
            energy=2,
            hand=[],
            actionability={
                "transient_only_end_turn": False,
                "frontier_stable": True,
                "legal_non_end_turn_count": 0,
            },
        )
        result, metrics = env._recover_post_step_frontier(
            initial,
            selected_action=END_TURN,
            legal_actions_before=[END_TURN],
        )

        self.assertIs(result, initial)
        self.assertFalse(metrics["attempted"])
        self.assertTrue(metrics["stable_no_actions"])
        self.assertFalse(metrics["leaked"])
        self.assertEqual(bridge.get_state_calls, 0)

    def test_energy_positive_empty_hand_without_actionability_polls_and_recovers_if_rebound_seen(self) -> None:
        rebound = _step_result(
            energy=3,
            hand=[{"id": "CARD.STRIKE_R"}],
            legal_actions=[STRIKE, END_TURN],
        )
        bridge = FakeBridge(states=[_state_with_actions([STRIKE, END_TURN])], rebind_result=rebound)
        env = _env_stub(bridge)

        initial = _step_result(energy=3, hand=[], actionability=None)
        result, metrics = env._recover_post_step_frontier(
            initial,
            selected_action=END_TURN,
            legal_actions_before=[END_TURN],
        )

        self.assertEqual(metrics["reason"], "energy_positive_hand_empty_or_missing")
        self.assertTrue(metrics["resolved"])
        self.assertEqual(result["legal_actions"], [STRIKE, END_TURN])

    def test_energy_positive_empty_hand_without_rebound_times_out_as_stable_not_leak(self) -> None:
        bridge = FakeBridge(states=[_state_with_actions([END_TURN])], rebind_result=None)
        env = _env_stub(bridge)

        initial = _step_result(energy=3, hand=[], actionability=None)
        result, metrics = env._recover_post_step_frontier(
            initial,
            selected_action=END_TURN,
            legal_actions_before=[END_TURN],
        )

        self.assertIs(result, initial)
        self.assertTrue(metrics["timeout"])
        self.assertTrue(metrics["stable_no_actions"])
        self.assertFalse(metrics["leaked"])
        self.assertGreaterEqual(bridge.get_state_calls, 1)

    def test_bridge_declared_transient_timeout_is_tracked_as_leak(self) -> None:
        bridge = FakeBridge(states=[_state_with_actions([END_TURN])], rebind_result=None)
        env = _env_stub(bridge)

        initial = _step_result(
            energy=3,
            hand=[],
            actionability={
                "transient_only_end_turn": True,
                "frontier_stable": False,
                "legal_non_end_turn_count": 0,
            },
        )
        result, metrics = env._recover_post_step_frontier(
            initial,
            selected_action=END_TURN,
            legal_actions_before=[END_TURN],
        )

        self.assertIs(result, initial)
        self.assertTrue(metrics["timeout"])
        self.assertTrue(metrics["leaked"])
        self.assertFalse(metrics["stable_no_actions"])
        self.assertEqual(env._fast_step_metrics_total["post_step_frontier_leaked_count"], 1)

    def test_disabled_frontier_recovery_does_not_poll(self) -> None:
        bridge = FakeBridge(states=[_state_with_actions([STRIKE, END_TURN])])
        env = _env_stub(bridge)
        env._fast_step_disabled = True

        initial = _step_result(
            energy=3,
            hand=[],
            actionability={
                "transient_only_end_turn": True,
                "frontier_stable": False,
                "legal_non_end_turn_count": 0,
            },
        )
        result, metrics = env._recover_post_step_frontier(
            initial,
            selected_action=END_TURN,
            legal_actions_before=[END_TURN],
        )

        self.assertIs(result, initial)
        self.assertFalse(metrics["attempted"])
        self.assertEqual(metrics["reason"], "disabled")
        self.assertEqual(bridge.get_state_calls, 0)


if __name__ == "__main__":
    unittest.main()
