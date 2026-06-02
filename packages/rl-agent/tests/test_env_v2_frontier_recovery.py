"""Focused tests for EnvV2 full-run action frontier recovery.

These cover the regression where the live bridge can expose only automation or
singleton end_turn while combat/action animations are still settling.  EnvV2
must not leak automation to policy, and it must keep the recovery wait bounded
on the hot step path.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any


RL_AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(RL_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(RL_AGENT_ROOT))

from sts2_env.action_compact import compact_action_signature
from sts2_env.env_v2 import SlayTheSpire2EnvV2
from sts2_env.reward_constants import (
    BOSS_COMBAT_LOSS_DAMAGE_UNDO_PERCENT_SCALE,
    BOSS_COMBAT_LOSS_PENALTY_BASE,
    BOSS_COMBAT_LOSS_PENALTY_MISSING_HP_SCALE,
    BOSS_ENEMY_HP_DELTA_PERCENT_SCALE,
    FULL_RUN_DEATH_PENALTY_BASE,
    FULL_RUN_DEATH_PENALTY_LATE_ACT_SCALE,
    FULL_RUN_DEATH_PENALTY_MISSING_HP_SCALE,
    REST_SITE_SKIP_HEAL_PENALTY,
)


def _env_stub() -> SlayTheSpire2EnvV2:
    class _DummyActionHistory:
        def record(self, **_kwargs: Any) -> None:
            return None

        def to_obs_dict(self) -> dict[str, Any]:
            return {}

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

        def attach_route_snapshot_to_obs(self, _obs: dict[str, Any]) -> None:
            return None

    class _DummyCombatMemory:
        def snapshot(self) -> dict[str, Any]:
            return {}

        def update(self, *_args: Any, **_kwargs: Any) -> None:
            return None

    env = object.__new__(SlayTheSpire2EnvV2)
    env.step_timeout_ms = 20_000
    env.reset_timeout_ms = 60_000
    env.include_debug_info = False
    env.obs_encoder = _DummyObsEncoder()
    env._run_memory = _DummyRunMemory()
    env._combat_memory = _DummyCombatMemory()
    env._action_history = _DummyActionHistory()
    env._legal_actions = []
    env._last_obs_raw = {}
    env._last_actionability = None
    env._last_bridge_info = None
    env._last_action_overflow = 0
    env._last_raw_legal_action_count = 0
    env._last_raw_legal_actions_compact = []
    env._last_blocked_action_drop_count = 0
    env._consecutive_end_turn_leaks = 0
    env._max_floor_reached = 0
    env.stuck_watchdog_steps = 0
    env._episode_id = "test-episode"
    env._episode_telemetry = SlayTheSpire2EnvV2._blank_telemetry()
    return env


class EnvV2FrontierHelpersTests(unittest.TestCase):
    def test_potion_action_slot_uses_bridge_slot_not_player_index(self):
        self.assertEqual(
            SlayTheSpire2EnvV2._potion_action_slot({"action_id": "discard_potion:0:1", "kind": "discard_potion"}),
            1,
        )
        self.assertEqual(
            SlayTheSpire2EnvV2._potion_action_slot({"action_id": "use_potion:0:2:self", "kind": "use_potion"}),
            2,
        )
        self.assertEqual(
            SlayTheSpire2EnvV2._potion_action_slot({"action_id": "discard_potion:1", "kind": "discard_potion"}),
            1,
        )

    def test_update_live_state_filters_automation_and_counts_drop(self):
        env = _env_stub()
        env._update_live_state(
            {
                "obs": {"phase": "combat"},
                "legal_actions": [
                    {"action_id": "automation:start_autoslay", "kind": "automation"},
                    {"action_id": "discard_potion:0", "kind": "discard_potion"},
                    {"action_id": "end_turn", "kind": "end_turn"},
                ],
                "info": {"actionability": {"frontier_stable": True}},
            }
        )
        self.assertEqual(env._last_raw_legal_action_count, 3)
        self.assertEqual(len(env._last_raw_legal_actions_compact), 3)
        self.assertEqual(env._last_blocked_action_drop_count, 2)
        self.assertEqual(env._legal_actions, [{"action_id": "end_turn", "kind": "end_turn"}])
        self.assertEqual(env._episode_telemetry["frontier_actions_dropped_blocked"], 2.0)

    def test_build_info_exposes_raw_compact_actions_after_filtering(self):
        env = _env_stub()
        env._update_live_state(
            {
                "obs": {
                    "phase": "rest_site",
                    "player": {"hp": 20, "max_hp": 80},
                    "run": {"floor": 7},
                },
                "legal_actions": [
                    {
                        "kind": "rest_site",
                        "action_id": "rest_site:0",
                        "option": {"option_id": "HEAL", "option_type": "HealRestSiteOption", "title": "休息"},
                    },
                    {
                        "kind": "rest_site",
                        "action_id": "rest_site:1",
                        "option": {"option_id": "SMITH", "option_type": "SmithRestSiteOption", "title": "锻造"},
                    },
                ],
                "info": {},
            }
        )

        info = env._build_info({})

        self.assertEqual(info["raw_legal_action_count"], 2)
        self.assertEqual(len(info["raw_legal_actions_compact"]), 2)
        self.assertEqual(len(info["legal_actions_compact"]), 1)
        self.assertEqual(info["raw_legal_actions_compact"][0]["option_type"], "HealRestSiteOption")
        self.assertEqual(info["raw_legal_actions_compact"][1]["option_type"], "SmithRestSiteOption")

    def test_update_live_state_keeps_forced_singleton_discard_potion(self):
        env = _env_stub()
        env._update_live_state(
            {
                "obs": {
                    "phase": "combat",
                    "player": {
                        "potions": [
                            {"slot": 0, "id": "POTION.A", "title": "A"},
                            {"slot": 1, "id": "POTION.B", "title": "B"},
                            {"slot": 2, "id": "POTION.C", "title": "C"},
                        ]
                    },
                },
                "legal_actions": [
                    {"action_id": "discard_potion:0", "kind": "discard_potion"},
                ],
                "info": {},
            }
        )
        self.assertEqual(env._last_raw_legal_action_count, 1)
        self.assertEqual(env._last_blocked_action_drop_count, 0)
        self.assertEqual(env._legal_actions, [{"action_id": "discard_potion:0", "kind": "discard_potion"}])

    def test_update_live_state_drops_singleton_discard_potion_when_empty_slot_exists(self):
        env = _env_stub()
        env._update_live_state(
            {
                "obs": {
                    "phase": "combat",
                    "player": {
                        "potions": [
                            {"slot": 0, "id": "POTION.BLOCK_POTION", "title": "格挡药水"},
                            {"slot": 1, "empty": True, "title": "[empty]"},
                            {"slot": 2, "empty": True},
                        ]
                    },
                },
                "legal_actions": [
                    {"action_id": "discard_potion:0:0", "kind": "discard_potion"},
                ],
                "info": {},
            }
        )
        self.assertEqual(env._last_raw_legal_action_count, 1)
        self.assertEqual(env._legal_actions, [])
        self.assertEqual(env._episode_telemetry["frontier_discard_potion_empty_slot_seen"], 1.0)
        self.assertEqual(env._episode_telemetry["frontier_discard_potion_empty_slot_blocked"], 1.0)
        self.assertEqual(env._episode_telemetry["frontier_only_discard_potion_empty_slots"], 2.0)

    def test_singleton_end_turn_with_energy_and_hand_short_waits(self):
        env = _env_stub()
        env._legal_actions = [{"action_id": "end_turn", "kind": "end_turn"}]
        env._last_actionability = {"frontier_stable": True, "legal_non_end_turn_count": 0}
        env._last_obs_raw = {
            "combat": {
                "in_progress": True,
                "energy": 1,
                "hand": [{"id": "strike", "cost": 1}],
            }
        }
        self.assertTrue(env._current_frontier_needs_short_wait())
        self.assertTrue(env._frontier_suspicion_has_energy_and_hand())
        self.assertTrue(env._frontier_has_affordable_raw_combat_card())

    def test_affordable_raw_combat_card_predicate_ignores_quest_status_and_expensive_cards(self):
        env = _env_stub()
        env._last_obs_raw = {
            "combat": {
                "in_progress": True,
                "energy": 1,
                "hand": [
                    {"id": "CARD.TREASURE_MAP", "title": "藏宝图", "type": "Quest", "cost": 0},
                    {"id": "CARD.DAZED", "title": "晕眩", "type": "Status", "cost": 0},
                    {"id": "CARD.EMBER", "title": "余烬", "type": "Attack", "cost": 2},
                ],
            }
        }
        self.assertFalse(env._frontier_has_affordable_raw_combat_card())

        env._last_obs_raw["combat"]["hand"].append(
            {"id": "CARD.DEFEND_IRONCLAD", "title": "防御", "type": "Skill", "cost": 1}
        )
        self.assertTrue(env._frontier_has_affordable_raw_combat_card())

    def test_singleton_end_turn_with_filtered_drop_short_waits_even_if_actionability_has_non_end_turn(self):
        env = _env_stub()
        env._legal_actions = [{"action_id": "end_turn", "kind": "end_turn"}]
        env._last_actionability = {"frontier_stable": True, "legal_non_end_turn_count": 1}
        env._last_blocked_action_drop_count = 1
        env._last_obs_raw = {
            "combat": {
                "in_progress": True,
                "energy": 0,
                "hand": [],
            }
        }

        self.assertTrue(env._current_frontier_needs_short_wait())

    def test_step_blocks_stale_singleton_end_turn_when_recovery_finds_non_end_turn(self):
        env = _env_stub()
        env._legal_actions = [{"action_id": "end_turn", "kind": "end_turn"}]
        env._last_actionability = {"frontier_stable": False, "legal_non_end_turn_count": 0}
        env._last_obs_raw = {
            "phase": "combat",
            "run": {"floor": 17, "room_type": "Boss"},
            "player": {"hp": 42, "max_hp": 80, "energy": 3},
            "combat": {
                "in_progress": True,
                "energy": 3,
                "hand": [{"id": "CARD.STRIKE_R", "title": "打击", "cost": 1}],
            },
        }
        calls = {"recover": 0, "bridge_step": 0}

        def _recover_filtered_action_window(*, timeout_ms: int) -> bool:
            calls["recover"] += 1
            env._legal_actions = [
                {
                    "action_id": "play_card:0:0",
                    "kind": "play_card",
                    "card": {"id": "CARD.STRIKE_R", "title": "打击", "cost": 1},
                },
                {"action_id": "end_turn", "kind": "end_turn"},
            ]
            env._last_actionability = {"frontier_stable": True, "legal_non_end_turn_count": 1}
            return True

        class _BridgeShouldNotStep:
            def step(self, **kwargs: Any) -> dict[str, Any]:
                calls["bridge_step"] += 1
                raise AssertionError("stale singleton EndTurn must not reach bridge.step")

        env._recover_filtered_action_window = _recover_filtered_action_window  # type: ignore[method-assign]
        env.bridge = _BridgeShouldNotStep()

        obs, reward, terminated, truncated, info = env.step(0)

        self.assertEqual(calls["recover"], 1)
        self.assertEqual(calls["bridge_step"], 0)
        self.assertEqual(obs["legal_count"], 2)
        self.assertEqual(reward, 0.0)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertTrue(info["frontier_refreshed_before_end_turn"])
        self.assertEqual(
            info["bridge_info"]["action_diagnostics"]["frontier_pre_dispatch_end_turn_blocked"],
            1.0,
        )
        self.assertEqual(env._episode_telemetry["frontier_only_end_turn_pre_dispatch_waits"], 1.0)
        self.assertEqual(env._episode_telemetry["frontier_only_end_turn_pre_dispatch_blocked"], 1.0)

    def test_step_blocks_high_confidence_singleton_end_turn_even_when_recovery_times_out(self):
        env = _env_stub()
        env._legal_actions = [{"action_id": "end_turn", "kind": "end_turn"}]
        env._last_actionability = {"frontier_stable": True, "legal_non_end_turn_count": 0}
        env._last_obs_raw = {
            "phase": "combat",
            "run": {"floor": 7, "room_type": "Monster"},
            "player": {"hp": 38, "max_hp": 80, "energy": 1},
            "combat": {
                "in_progress": True,
                "energy": 1,
                "hand": [
                    {"id": "CARD.DEFEND_IRONCLAD", "title": "防御", "type": "Skill", "cost": 1},
                    {"id": "CARD.DEFEND_IRONCLAD", "title": "防御", "type": "Skill", "cost": 1},
                ],
            },
        }
        calls = {"recover": 0, "bridge_step": 0}

        def _recover_filtered_action_window(*, timeout_ms: int) -> bool:
            calls["recover"] += 1
            # Simulate the observed bridge failure: bounded wait expires but
            # the visible frontier is still singleton EndTurn despite an
            # affordable raw card in hand.
            env._legal_actions = [{"action_id": "end_turn", "kind": "end_turn"}]
            return True

        class _BridgeShouldNotStep:
            def step(self, **kwargs: Any) -> dict[str, Any]:
                calls["bridge_step"] += 1
                raise AssertionError("high-confidence singleton EndTurn leak must not dispatch")

        env._recover_filtered_action_window = _recover_filtered_action_window  # type: ignore[method-assign]
        env.bridge = _BridgeShouldNotStep()

        obs, reward, terminated, truncated, info = env.step(0)

        self.assertEqual(calls["recover"], 1)
        self.assertEqual(calls["bridge_step"], 0)
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
            env._episode_telemetry["frontier_only_end_turn_pre_dispatch_high_confidence_blocked"],
            1.0,
        )

    def test_step_allows_singleton_end_turn_when_raw_hand_has_only_quest_card(self):
        env = _env_stub()
        env._legal_actions = [{"action_id": "end_turn", "kind": "end_turn"}]
        env._last_actionability = {"frontier_stable": True, "legal_non_end_turn_count": 0}
        env._last_obs_raw = {
            "phase": "combat",
            "run": {"floor": 17, "room_type": "Boss"},
            "player": {"hp": 39, "max_hp": 80, "energy": 2},
            "combat": {
                "in_progress": True,
                "energy": 2,
                "hand": [{"id": "CARD.TREASURE_MAP", "title": "藏宝图", "type": "Quest", "cost": 0}],
            },
        }
        calls = {"recover": 0, "bridge_step": 0}

        def _recover_filtered_action_window(*, timeout_ms: int) -> bool:
            calls["recover"] += 1
            return True

        class _BridgeAllowsStep:
            def step(self, episode_id: str, action_id: str, timeout_ms: int) -> dict[str, Any]:
                calls["bridge_step"] += 1
                return {
                    "obs": env._last_obs_raw,
                    "legal_actions": [],
                    "reward": 0.0,
                    "done": True,
                    "terminated": True,
                    "truncated": False,
                    "info": {},
                }

        env._recover_filtered_action_window = _recover_filtered_action_window  # type: ignore[method-assign]
        env.bridge = _BridgeAllowsStep()

        _obs, _reward, terminated, _truncated, _info = env.step(0)

        self.assertEqual(calls["recover"], 1)
        self.assertEqual(calls["bridge_step"], 1)
        self.assertTrue(terminated)
        self.assertEqual(
            env._episode_telemetry["frontier_only_end_turn_pre_dispatch_high_confidence_blocked"],
            0.0,
        )

    def test_singleton_end_turn_with_zero_energy_does_not_short_wait(self):
        env = _env_stub()
        env._legal_actions = [{"action_id": "end_turn", "kind": "end_turn"}]
        env._last_actionability = {"frontier_stable": True, "legal_non_end_turn_count": 0}
        env._last_obs_raw = {
            "combat": {
                "in_progress": True,
                "energy": 0,
                "hand": [{"id": "strike", "cost": 1}],
            }
        }
        self.assertFalse(env._current_frontier_needs_short_wait())

    def test_actionability_transient_forces_short_wait_even_without_energy(self):
        env = _env_stub()
        env._legal_actions = [{"action_id": "end_turn", "kind": "end_turn"}]
        env._last_actionability = {"transient_only_end_turn": True, "frontier_stable": False}
        env._last_obs_raw = {"combat": {"in_progress": True, "energy": 0, "hand": []}}
        self.assertTrue(env._current_frontier_needs_short_wait())

    def test_singleton_discard_potion_with_energy_and_hand_short_waits(self):
        env = _env_stub()
        env._legal_actions = [{"action_id": "discard_potion:0:0", "kind": "discard_potion"}]
        env._last_obs_raw = {
            "combat": {
                "in_progress": True,
                "energy": 1,
                "hand": [{"id": "CARD.STRIKE_R", "cost": 1}],
            },
            "player": {
                "potions": [
                    {"slot": 0, "id": "POTION.BLOCK_POTION", "title": "格挡药水"},
                    {"slot": 1, "empty": True, "title": "[empty]"},
                ]
            },
        }

        self.assertTrue(env._current_frontier_needs_short_wait_for_discard_potion())

    def test_singleton_discard_potion_without_combat_does_not_short_wait(self):
        env = _env_stub()
        env._legal_actions = [{"action_id": "discard_potion:0:0", "kind": "discard_potion"}]
        env._last_obs_raw = {
            "combat": {"in_progress": False, "energy": 1, "hand": [{"id": "CARD.STRIKE_R"}]},
            "player": {"potions": [{"slot": 0, "id": "POTION.A", "title": "A"}]},
        }

        self.assertFalse(env._current_frontier_needs_short_wait_for_discard_potion())

    def test_singleton_discard_potion_with_empty_slot_short_waits_even_without_combat(self):
        env = _env_stub()
        env._legal_actions = [{"action_id": "discard_potion:0:0", "kind": "discard_potion"}]
        env._last_obs_raw = {
            "phase": "actions",
            "combat": {"in_progress": False, "energy": 0, "hand": []},
            "player": {
                "potions": [
                    {"slot": 0, "id": "POTION.A", "title": "A"},
                    {"slot": 1, "empty": True, "title": "[empty]"},
                ]
            },
        }

        self.assertTrue(env._current_frontier_needs_short_wait_for_discard_potion())
        self.assertEqual(env._episode_telemetry["frontier_discard_potion_empty_slot_seen"], 1.0)
        self.assertEqual(env._episode_telemetry["frontier_only_discard_potion_empty_slots"], 1.0)

    def test_state_helpers_detect_blocked_only_and_fallback_legal_actions(self):
        env = _env_stub()
        blocked_state: dict[str, Any] = {
            "available_actions": [
                {"action_id": "automation:start_autoslay", "kind": "automation"},
            ]
        }
        self.assertTrue(env._state_has_only_blocked_actions(blocked_state))
        self.assertEqual(env._state_unblocked_action_count(blocked_state), 0)

        forced_discard_state: dict[str, Any] = {
            "available_actions": [{"action_id": "discard_potion:0", "kind": "discard_potion"}]
        }
        self.assertFalse(env._state_has_only_blocked_actions(forced_discard_state))
        self.assertEqual(env._state_unblocked_action_count(forced_discard_state), 1)
        self.assertEqual(env._state_non_end_turn_unblocked_action_count(forced_discard_state), 1)
        self.assertEqual(env._state_non_discard_potion_unblocked_action_count(forced_discard_state), 0)

        optional_discard_state: dict[str, Any] = {
            "available_actions": [
                {"action_id": "discard_potion:0", "kind": "discard_potion"},
                {"action_id": "end_turn", "kind": "end_turn"},
            ]
        }
        self.assertFalse(env._state_has_only_blocked_actions(optional_discard_state))
        self.assertEqual(env._state_unblocked_action_count(optional_discard_state), 1)
        self.assertEqual(env._state_non_end_turn_unblocked_action_count(optional_discard_state), 0)
        self.assertEqual(env._state_non_discard_potion_unblocked_action_count(optional_discard_state), 1)

        fallback_state: dict[str, Any] = {
            "legal_actions": [{"action_id": "end_turn", "kind": "end_turn"}]
        }
        self.assertEqual(env._state_unblocked_action_count(fallback_state), 1)
        self.assertEqual(env._state_non_end_turn_unblocked_action_count(fallback_state), 0)
        self.assertEqual(env._state_non_discard_potion_unblocked_action_count(fallback_state), 1)

    def test_transition_recovery_timeout_is_step_bounded(self):
        env = _env_stub()
        env.step_timeout_ms = 20_000
        self.assertEqual(env._transition_recovery_timeout_ms(), 5_000)
        env.step_timeout_ms = 250
        self.assertEqual(env._transition_recovery_timeout_ms(), 500)

    def test_transition_state_preserves_boss_room_model(self):
        env = _env_stub()
        env._last_obs_raw = {
            "phase": "combat",
            "player": {"hp": 35, "max_hp": 91, "potions": [{"empty": True}]},
            "run": {
                "floor": 17,
                "act_floor": 17,
                "act_id": "ACT.UNDERDOCKS",
                "room_type": "Boss",
                "room_model": "ENCOUNTER.LAGAVULIN_MATRIARCH_BOSS",
                "active": True,
                "game_over": True,
            },
            "combat": {"energy": 1, "enemies": []},
        }

        state = env._transition_state()

        self.assertEqual(state["run"]["room_model"], "ENCOUNTER.LAGAVULIN_MATRIARCH_BOSS")
        self.assertEqual(state["run"]["encounter_id"], "ENCOUNTER.LAGAVULIN_MATRIARCH_BOSS")
        self.assertEqual(state["run"]["room_type"], "Boss")
        self.assertEqual(state["run"]["act_floor"], 17.0)

    def test_full_run_potion_transition_records_forced_discard(self):
        env = _env_stub()
        action = {"action_id": "discard_potion:1", "kind": "discard_potion"}
        before_obs = {
            "run": {"floor": 8, "room_type": "Reward", "room_model": "REWARD.TEST"},
            "player": {
                "potions": [
                    {"slot": 0, "id": "POTION.A", "title": "A"},
                    {"slot": 1, "id": "POTION.B", "title": "B"},
                ]
            },
        }
        after_obs = {
            "run": {"floor": 8, "room_type": "Reward", "room_model": "REWARD.TEST"},
            "player": {
                "potions": [
                    {"slot": 0, "id": "POTION.A", "title": "A"},
                    {"slot": 1, "empty": True, "title": "[empty]"},
                ]
            },
        }

        record = env._build_potion_transition_record(
            action=action,
            prev_obs=before_obs,
            after_obs=after_obs,
            legal_actions_before=[action],
            bridge_info={},
            reward=0.0,
            terminated=False,
            truncated=False,
        )

        self.assertIsNotNone(record)
        assert record is not None
        self.assertEqual(record["event"], "discard_potion_transition")
        self.assertEqual(record["potion_slot"], 1)
        self.assertEqual(record["potion_count_before"], 2)
        self.assertEqual(record["potion_count_after"], 1)
        self.assertTrue(record["forced_singleton_discard"])
        self.assertFalse(record["optional_discard_with_alternative"])

    def test_full_run_potion_transition_marks_optional_discard(self):
        env = _env_stub()
        discard = {"action_id": "discard_potion:0", "kind": "discard_potion"}
        end_turn = {"action_id": "end_turn", "kind": "end_turn"}
        obs = {
            "run": {"floor": 6, "room_type": "Normal"},
            "player": {"potions": [{"slot": 0, "id": "POTION.A", "title": "A"}]},
        }

        record = env._build_potion_transition_record(
            action=discard,
            prev_obs=obs,
            after_obs=obs,
            legal_actions_before=[discard, end_turn],
            bridge_info={},
            reward=0.0,
            terminated=False,
            truncated=False,
        )

        self.assertIsNotNone(record)
        assert record is not None
        self.assertFalse(record["forced_singleton_discard"])
        self.assertTrue(record["optional_discard_with_alternative"])

    def test_recovery_accepts_existing_end_turn_after_tiny_short_wait(self):
        env = _env_stub()
        env._legal_actions = [{"action_id": "end_turn", "kind": "end_turn"}]
        env._last_actionability = {"transient_only_end_turn": True, "frontier_stable": False}
        env._last_obs_raw = {"combat": {"in_progress": True, "energy": 1, "hand": [{"id": "a"}]}}
        env._safe_get_state = lambda: {"available_actions": [{"action_id": "end_turn", "kind": "end_turn"}]}
        env._safe_reset_into_current_run = lambda _timeout_ms: None
        self.assertTrue(env._recover_filtered_action_window(timeout_ms=1))
        self.assertEqual(env._episode_telemetry["frontier_only_end_turn_short_waits"], 1.0)
        self.assertEqual(env._episode_telemetry["frontier_only_end_turn_leaked"], 1.0)

    def test_recovery_resolves_singleton_discard_potion_after_rebind(self):
        env = _env_stub()
        discard = {"action_id": "discard_potion:0:0", "kind": "discard_potion"}
        play = {"action_id": "play_card:0:e1", "kind": "play_card"}
        end = {"action_id": "end_turn", "kind": "end_turn"}
        env._legal_actions = [discard]
        env._last_obs_raw = {"phase": "combat", "combat": {"in_progress": True, "energy": 1, "hand": [{"id": "a"}]}}
        env._safe_get_state = lambda: {"available_actions": [play, end]}
        env._safe_reset_into_current_run = lambda _timeout_ms: {
            "episode_id": "test-episode",
            "obs": {
                "phase": "combat",
                "combat": {"in_progress": True, "energy": 1, "hand": [{"id": "a"}]},
            },
            "legal_actions": [play, end],
            "info": {},
        }

        self.assertTrue(env._recover_filtered_action_window(timeout_ms=150))
        self.assertEqual(env._legal_actions, [play, end])
        self.assertEqual(env._episode_telemetry["frontier_only_discard_potion_short_waits"], 1.0)
        self.assertEqual(env._episode_telemetry["frontier_only_discard_potion_resolved"], 1.0)
        self.assertEqual(env._episode_telemetry["frontier_only_discard_potion_leaked"], 0.0)

    def test_recovery_keeps_singleton_discard_potion_when_overflow_modal(self):
        env = _env_stub()
        discard = {"action_id": "discard_potion:0:0", "kind": "discard_potion"}
        env._legal_actions = [discard]
        env._last_obs_raw = {
            "phase": "actions",
            "combat": {"in_progress": False, "energy": 0, "hand": []},
            "player": {
                "potions": [
                    {"slot": 0, "id": "POTION.A", "title": "A"},
                    {"slot": 1, "id": "POTION.B", "title": "B"},
                    {"slot": 2, "id": "POTION.C", "title": "C"},
                ]
            },
        }
        env._safe_get_state = lambda: self.fail("_safe_get_state should not run for real overflow modal")
        env._safe_reset_into_current_run = lambda _timeout_ms: self.fail("no rebind expected")

        self.assertTrue(env._recover_filtered_action_window(timeout_ms=150))
        self.assertEqual(env._legal_actions, [discard])
        self.assertEqual(env._episode_telemetry["frontier_recovery_attempts"], 0.0)
        self.assertEqual(env._episode_telemetry["frontier_only_discard_potion_short_waits"], 0.0)

    def test_recovery_stalls_after_repeated_singleton_end_turn_leaks(self):
        env = _env_stub()
        env._legal_actions = [{"action_id": "end_turn", "kind": "end_turn"}]
        env._last_actionability = {"transient_only_end_turn": True, "frontier_stable": False}
        env._last_obs_raw = {"combat": {"in_progress": True, "energy": 1, "hand": [{"id": "a"}]}}
        env._consecutive_end_turn_leaks = 15
        env._safe_get_state = lambda: {"available_actions": [{"action_id": "end_turn", "kind": "end_turn"}]}
        env._safe_reset_into_current_run = lambda _timeout_ms: None

        self.assertFalse(env._recover_filtered_action_window(timeout_ms=1))
        self.assertEqual(env._episode_telemetry["frontier_end_turn_leak_stalls"], 1.0)
        self.assertGreaterEqual(env._episode_telemetry["frontier_consecutive_end_turn_leaks"], 16.0)

    def test_recovery_returns_false_for_blocked_only_without_stall(self):
        env = _env_stub()
        env._safe_get_state = lambda: {
            "available_actions": [{"action_id": "automation:start_autoslay", "kind": "automation"}]
        }
        env._safe_reset_into_current_run = lambda _timeout_ms: None
        self.assertFalse(env._recover_filtered_action_window(timeout_ms=1))
        self.assertEqual(env._episode_telemetry["frontier_recovery_timeouts"], 1.0)
        self.assertEqual(env._episode_telemetry["frontier_blocked_only_timeouts"], 1.0)

    def test_full_run_potion_timing_penalizes_no_followup_resource_potion(self):
        env = _env_stub()
        action = {
            "action_id": "use_potion:0",
            "kind": "use_potion",
            "potion": {
                "id": "TEST_ENERGY_POTION",
                "effect_profile": {"energy_gain": 2},
            },
        }
        before_obs = {
            "run": {"state_type": "normal"},
            "player": {"hp": 70, "max_hp": 80, "block": 0, "energy": 0},
            "combat": {
                "enemies": [
                    {"combat_id": "e1", "hp": 40, "intent": {"total_damage": 0}},
                ],
            },
        }
        reward = env._potion_timing_step_reward(
            action,
            before_obs,
            [action, {"action_id": "end_turn", "kind": "end_turn"}],
        )

        self.assertLess(reward, 0.0)
        self.assertEqual(env._episode_telemetry["potion_timing_quality_events"], 0.0)
        self.assertEqual(env._episode_telemetry["potion_timing_waste_events"], 1.0)
        self.assertLess(env._episode_telemetry["potion_timing_reward_total"], 0.0)

    def test_full_run_potion_timing_penalizes_strength_requires_followup_without_followup(self):
        env = _env_stub()
        action = {
            "action_id": "use_potion:0:self",
            "kind": "use_potion",
            "potion": {
                "id": "POTION.STRENGTH_POTION",
                "title": "力量药水",
                "effect_profile": {"strength": 2, "requires_followup": True},
                "timing_tags": ["requires_followup"],
            },
        }
        before_obs = {
            "run": {"state_type": "normal"},
            "player": {"hp": 84, "max_hp": 91, "block": 5, "energy": 0},
            "combat": {
                "enemies": [
                    {"combat_id": "e1", "hp": 40, "intent": {"total_damage": 8}},
                ],
            },
        }
        reward = env._potion_timing_step_reward(
            action,
            before_obs,
            [action, {"action_id": "end_turn", "kind": "end_turn"}],
        )

        self.assertLess(reward, 0.0)
        self.assertEqual(env._episode_telemetry["potion_timing_quality_events"], 0.0)
        self.assertEqual(env._episode_telemetry["potion_timing_waste_events"], 1.0)
        self.assertLess(env._episode_telemetry["potion_timing_reward_total"], 0.0)

    def test_full_run_potion_timing_penalizes_block_potion_when_already_blocked_and_safe(self):
        env = _env_stub()
        action = {
            "action_id": "use_potion:0:self",
            "kind": "use_potion",
            "potion": {
                "id": "POTION.BLOCK_POTION",
                "title": "格挡药水",
                "effect_profile": {"block": 12},
            },
        }
        before_obs = {
            "run": {"state_type": "normal"},
            "player": {"hp": 75, "max_hp": 78, "block": 10, "energy": 0},
            "combat": {
                "enemies": [
                    {"combat_id": "e1", "hp": 35, "intent": {"total_damage": 11}},
                ],
            },
        }
        reward = env._potion_timing_step_reward(
            action,
            before_obs,
            [action, {"action_id": "end_turn", "kind": "end_turn"}],
        )

        self.assertLess(reward, 0.0)
        self.assertEqual(env._episode_telemetry["potion_timing_quality_events"], 0.0)
        self.assertEqual(env._episode_telemetry["potion_timing_waste_events"], 1.0)
        self.assertLess(env._episode_telemetry["potion_timing_reward_total"], 0.0)

    def test_full_run_potion_timing_rewards_lethal_damage_potion(self):
        env = _env_stub()
        action = {
            "action_id": "use_potion:0:e1",
            "kind": "use_potion",
            "target": {"combat_id": "e1"},
            "potion": {
                "id": "TEST_DAMAGE_POTION",
                "effect_profile": {"damage": 20},
            },
        }
        before_obs = {
            "run": {"state_type": "normal"},
            "player": {"hp": 70, "max_hp": 80, "block": 0, "energy": 0},
            "combat": {
                "enemies": [
                    {"combat_id": "e1", "hp": 10, "intent": {"total_damage": 0}},
                ],
            },
        }
        reward = env._potion_timing_step_reward(
            action,
            before_obs,
            [action, {"action_id": "end_turn", "kind": "end_turn"}],
        )

        self.assertGreater(reward, 0.0)
        self.assertEqual(env._episode_telemetry["potion_timing_quality_events"], 1.0)
        self.assertGreater(env._episode_telemetry["potion_timing_reward_total"], 0.0)

    def test_rest_site_low_hp_threshold_penalizes_smith_at_62_percent_hp(self):
        env = _env_stub()
        before_obs = {"player": {"hp": 62, "max_hp": 100}}
        action = {
            "action_id": "rest_site:smith",
            "kind": "rest",
            "option": {"type": "smith", "title": "Smith"},
        }

        reward = env._rest_site_skip_heal_penalty(before_obs, action)

        self.assertEqual(reward, float(REST_SITE_SKIP_HEAL_PENALTY))
        self.assertEqual(env._episode_telemetry["rest_skip_heal_at_low_hp"], 1.0)
        self.assertEqual(env._episode_telemetry["rest_penalty_total"], float(REST_SITE_SKIP_HEAL_PENALTY))

    def test_rest_site_smith_identity_overrides_stale_heal_flags(self):
        env = _env_stub()
        action = {
            "index": 1,
            "kind": "rest_site",
            "action_id": "rest_site:1",
            "action_kind": "heal",
            "title": "锻造",
            "is_rest_site": True,
            "is_heal": True,
            "is_smith": True,
        }

        self.assertTrue(SlayTheSpire2EnvV2._is_rest_site_choice_action(action))
        self.assertTrue(SlayTheSpire2EnvV2._is_rest_smith_choice_action(action))
        self.assertFalse(SlayTheSpire2EnvV2._is_rest_heal_choice_action(action))

        reward = env._rest_site_skip_heal_penalty({"player": {"hp": 80, "max_hp": 100}}, action)
        self.assertEqual(reward, 0.0)
        self.assertEqual(env._episode_telemetry["rest_site_encounters"], 1.0)
        self.assertEqual(env._episode_telemetry["rest_smith_chosen"], 1.0)
        self.assertEqual(env._episode_telemetry["rest_heal_chosen"], 0.0)

    def test_low_hp_rest_site_exposure_filters_to_heal_only(self):
        env = _env_stub()

        env._update_live_state(
            {
                "obs": {"phase": "actions", "player": {"hp": 50, "max_hp": 100}},
                "legal_actions": [
                    {
                        "kind": "rest_site",
                        "action_id": "sim:choose_rest_option:index=0",
                        "label": "Smith",
                        "option": {
                            "option_id": "smith",
                            "option_type": "smith",
                            "title": "Smith",
                        },
                    },
                    {
                        "kind": "rest_site",
                        "action_id": "sim:choose_rest_option:index=1",
                        "label": "Rest",
                        "option": {
                            "option_id": "rest",
                            "option_type": "rest",
                            "title": "Rest",
                            "description": "Restore HP.",
                        },
                    },
                ],
                "info": {},
            }
        )

        self.assertEqual(len(env._legal_actions), 1)
        self.assertEqual(env._legal_actions[0]["option"]["option_type"], "rest")
        self.assertEqual(env._episode_telemetry["rest_heal_exposure_low_hp"], 1.0)
        self.assertEqual(env._episode_telemetry["rest_heal_exposure_heal_available"], 1.0)
        self.assertEqual(env._episode_telemetry["rest_heal_exposure_forced"], 1.0)

    def test_low_hp_rest_site_exposure_recognizes_live_bridge_heal_schema(self):
        env = _env_stub()

        env._update_live_state(
            {
                "obs": {"phase": "actions", "player": {"hp": 50, "max_hp": 100}},
                "legal_actions": [
                    {
                        "kind": "rest_site",
                        "action_id": "rest_site:0",
                        "label": "Rest site option 0: 锻造",
                        "option": {
                            "option_id": "SMITH",
                            "option_type": "SmithRestSiteOption",
                            "title": "锻造",
                            "description": "升级你牌组中的1张牌。",
                            "is_enabled": True,
                        },
                    },
                    {
                        "kind": "rest_site",
                        "action_id": "rest_site:1",
                        "label": "Rest site option 1: 休息",
                        "option": {
                            "option_id": "HEAL",
                            "option_type": "HealRestSiteOption",
                            "title": "休息",
                            "description": "回复18点生命值。",
                            "is_enabled": True,
                        },
                    },
                ],
                "info": {},
            }
        )

        self.assertEqual(len(env._legal_actions), 1)
        self.assertEqual(env._legal_actions[0]["option"]["option_id"], "HEAL")
        self.assertEqual(env._episode_telemetry["rest_heal_exposure_heal_available"], 1.0)
        self.assertEqual(env._episode_telemetry["rest_heal_exposure_forced"], 1.0)
        self.assertEqual(env._episode_telemetry["rest_heal_exposure_miss"], 0.0)

    def test_low_hp_rest_site_exposure_recognizes_nested_payload_bridge_schema(self):
        env = _env_stub()

        filtered = env._apply_low_hp_rest_heal_exposure_filter(
            [
                {
                    "action_id": "wrapper:0",
                    "payload": {
                        "kind": "rest_site",
                        "action_id": "rest_site:0",
                        "label": "Rest site option 0: Smith",
                        "option": {
                            "option_id": "SMITH",
                            "option_type": "SmithRestSiteOption",
                            "title": "Smith",
                        },
                    },
                },
                {
                    "action_id": "wrapper:1",
                    "payload": {
                        "kind": "rest_site",
                        "action_id": "rest_site:1",
                        "label": "Rest site option 1: Rest",
                        "option": {
                            "option_id": "HEAL",
                            "option_type": "HealRestSiteOption",
                            "description": "回复18点生命值。",
                        },
                    },
                },
            ],
            {"player": {"hp": 50, "max_hp": 100}},
        )

        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["payload"]["option"]["option_type"], "HealRestSiteOption")
        self.assertEqual(env._episode_telemetry["rest_heal_exposure_miss"], 0.0)

    def test_low_hp_rest_site_filter_does_not_treat_smith_as_heal(self):
        env = _env_stub()

        actions = env._apply_low_hp_rest_heal_exposure_filter(
            [
                {
                    "kind": "rest_site",
                    "action_id": "rest_site:smith",
                    "label": "Rest Site",
                    "option": {"option_type": "smith", "title": "Smith"},
                },
            ],
            {"player": {"hp": 50, "max_hp": 100}},
        )

        self.assertEqual(len(actions), 1)
        self.assertEqual(env._episode_telemetry["rest_heal_exposure_miss"], 1.0)

    def test_low_hp_rest_site_proceed_is_not_a_heal_choice_exposure_miss(self):
        env = _env_stub()
        actions = [
            {
                "kind": "rest_site",
                "action_id": "rest_site:proceed",
                "canonical_text": "动作｜营火｜",
            }
        ]

        filtered = env._apply_low_hp_rest_heal_exposure_filter(
            actions,
            {"player": {"hp": 50, "max_hp": 100}},
        )

        self.assertEqual(filtered, actions)
        self.assertEqual(env._episode_telemetry["rest_heal_exposure_low_hp"], 0.0)
        self.assertEqual(env._episode_telemetry["rest_heal_exposure_heal_available"], 0.0)
        self.assertEqual(env._episode_telemetry["rest_heal_exposure_forced"], 0.0)
        self.assertEqual(env._episode_telemetry["rest_heal_exposure_miss"], 0.0)

    def test_rest_site_proceed_is_not_counted_as_low_hp_skip_heal(self):
        env = _env_stub()
        penalty = env._rest_site_skip_heal_penalty(
            {"player": {"hp": 50, "max_hp": 100}},
            {
                "kind": "rest_site",
                "action_id": "rest_site:proceed",
                "canonical_text": "动作｜营火｜",
            },
        )

        self.assertEqual(penalty, 0.0)
        self.assertEqual(env._episode_telemetry["rest_site_encounters"], 0.0)
        self.assertEqual(env._episode_telemetry["rest_skip_heal_chosen"], 0.0)
        self.assertEqual(env._episode_telemetry["rest_skip_heal_at_low_hp"], 0.0)
        self.assertEqual(env._episode_telemetry["rest_penalty_total"], 0.0)

    def test_high_hp_rest_site_exposure_keeps_smith_available(self):
        env = _env_stub()
        actions = [
            {"kind": "rest_site", "action_id": "rest_site:smith", "option": {"option_type": "smith"}},
            {"kind": "rest_site", "action_id": "rest_site:rest", "option": {"option_type": "rest"}},
        ]

        filtered = env._apply_low_hp_rest_heal_exposure_filter(
            actions,
            {"player": {"hp": 90, "max_hp": 100}},
        )

        self.assertEqual(filtered, actions)
        self.assertEqual(env._episode_telemetry["rest_heal_exposure_forced"], 0.0)

    def test_low_hp_rest_site_exposure_accepts_max_health_schema(self):
        env = _env_stub()
        actions = [
            {"kind": "rest_site", "action_id": "rest_site:smith", "option": {"option_type": "smith"}},
            {"kind": "rest_site", "action_id": "rest_site:rest", "option": {"option_type": "rest"}},
        ]

        filtered = env._apply_low_hp_rest_heal_exposure_filter(
            actions,
            {"player": {"hp": 40, "maxHealth": 100}, "run": {"floor": 13}},
        )

        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["option"]["option_type"], "rest")
        self.assertEqual(env._episode_telemetry["rest_heal_exposure_forced"], 1.0)

    def test_low_hp_rest_site_exposure_treats_mid_act_max_hp_one_as_critical(self):
        env = _env_stub()
        actions = [
            {"kind": "rest_site", "action_id": "rest_site:smith", "option": {"option_type": "smith"}},
            {"kind": "rest_site", "action_id": "rest_site:rest", "option": {"option_type": "rest"}},
        ]

        filtered = env._apply_low_hp_rest_heal_exposure_filter(
            actions,
            {"player": {"hp": 1, "max_hp": 1}, "run": {"floor": 13}},
        )

        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["option"]["option_type"], "rest")
        self.assertEqual(env._episode_telemetry["rest_heal_exposure_forced"], 1.0)

    def test_low_hp_rest_site_exposure_fails_open_when_max_hp_missing(self):
        env = _env_stub()
        actions = [
            {"kind": "rest_site", "action_id": "rest_site:smith", "option": {"option_type": "smith"}},
            {"kind": "rest_site", "action_id": "rest_site:rest", "option": {"option_type": "rest"}},
        ]

        filtered = env._apply_low_hp_rest_heal_exposure_filter(
            actions,
            {"player": {"hp": 50}, "run": {"floor": 13}},
        )

        self.assertEqual(filtered, actions)
        self.assertEqual(env._episode_telemetry["rest_heal_exposure_forced"], 0.0)

    def test_low_hp_optional_event_combat_is_filtered_when_safe_alternative_exists(self):
        env = _env_stub()
        leave = {
            "kind": "event_option",
            "action_id": "event_option:0",
            "label": "离开",
            "option": {
                "title": "离开",
                "description": "什么都不做。",
                "effect_deltas": {"enter_combat": False},
            },
        }
        fight = {
            "kind": "event_option",
            "action_id": "event_option:1",
            "label": "我能打两个",
            "option": {
                "title": "我能打两个",
                "description": "接受挑战。",
                "effect_deltas": {"enter_combat": False},
            },
        }

        env._update_live_state(
            {
                "obs": {"phase": "actions", "player": {"hp": 55, "max_hp": 100}},
                "legal_actions": [leave, fight],
                "info": {},
            }
        )

        self.assertEqual(env._legal_actions, [leave])
        self.assertEqual(env._episode_telemetry["event_combat_option_available"], 1.0)
        self.assertEqual(env._episode_telemetry["event_combat_option_low_hp_available"], 1.0)
        self.assertEqual(env._episode_telemetry["event_combat_option_safe_alternative"], 1.0)
        self.assertEqual(env._episode_telemetry["event_combat_option_masked_low_hp"], 1.0)

    def test_high_hp_optional_event_combat_remains_available(self):
        env = _env_stub()
        leave = {
            "kind": "event_option",
            "action_id": "event_option:0",
            "option": {"title": "Leave", "effect_deltas": {"enter_combat": False}},
        }
        fight = {
            "kind": "event_option",
            "action_id": "event_option:1",
            "option": {"title": "Fight", "effect_deltas": {"enter_combat": True}},
        }

        filtered = env._apply_low_hp_event_combat_exposure_filter(
            [leave, fight],
            {"player": {"hp": 85, "max_hp": 100}},
        )

        self.assertEqual(filtered, [leave, fight])
        self.assertEqual(env._episode_telemetry["event_combat_option_available"], 1.0)
        self.assertEqual(env._episode_telemetry["event_combat_option_masked_low_hp"], 0.0)

    def test_forced_singleton_event_combat_remains_available_at_low_hp(self):
        env = _env_stub()
        fight = {
            "kind": "event_option",
            "action_id": "event_option:0",
            "option": {"title": "战斗", "effect_deltas": {"enter_combat": True}},
        }

        filtered = env._apply_low_hp_event_combat_exposure_filter(
            [fight],
            {"player": {"hp": 30, "max_hp": 100}},
        )

        self.assertEqual(filtered, [fight])
        self.assertEqual(env._episode_telemetry["event_combat_option_low_hp_available"], 1.0)
        self.assertEqual(env._episode_telemetry["event_combat_option_masked_low_hp"], 0.0)

    def test_low_hp_slippery_bridge_hp_loss_option_is_filtered_when_safe_alternative_exists(self):
        env = _env_stub()
        leave = {
            "kind": "event_option",
            "action_id": "event_option:0",
            "label": "离开",
            "option": {"title": "离开", "description": "安全离开。"},
        }
        risk = {
            "kind": "event_option",
            "action_id": "event_option:1",
            "label": "再撑一会",
            "option": {"title": "再撑一会", "description": "尝试继续通过桥。"},
        }

        env._update_live_state(
            {
                "obs": {
                    "phase": "actions",
                    "player": {"hp": 12, "max_hp": 91},
                    "run": {"room_model": "EVENT.SLIPPERY_BRIDGE"},
                },
                "legal_actions": [leave, risk],
                "info": {},
            }
        )

        self.assertEqual(env._legal_actions, [leave])
        self.assertEqual(env._episode_telemetry["event_hp_loss_option_available"], 1.0)
        self.assertEqual(env._episode_telemetry["event_hp_loss_option_low_hp_available"], 1.0)
        self.assertEqual(env._episode_telemetry["event_hp_loss_option_safe_alternative"], 1.0)
        self.assertEqual(env._episode_telemetry["event_hp_loss_option_masked_low_hp"], 1.0)

    def test_high_hp_slippery_bridge_hp_loss_option_remains_available(self):
        env = _env_stub()
        leave = {"kind": "event_option", "action_id": "event_option:0", "option": {"title": "离开"}}
        risk = {"kind": "event_option", "action_id": "event_option:1", "option": {"title": "再撑一会"}}

        filtered = env._apply_low_hp_event_hp_loss_exposure_filter(
            [leave, risk],
            {
                "player": {"hp": 80, "max_hp": 91},
                "run": {"room_model": "EVENT.SLIPPERY_BRIDGE"},
            },
        )

        self.assertEqual(filtered, [leave, risk])
        self.assertEqual(env._episode_telemetry["event_hp_loss_option_available"], 1.0)
        self.assertEqual(env._episode_telemetry["event_hp_loss_option_masked_low_hp"], 0.0)

    def test_low_hp_punch_off_hp_loss_text_option_is_filtered_when_safe_alternative_exists(self):
        env = _env_stub()
        leave = {
            "kind": "event_option",
            "action_id": "event_option:0",
            "label": "离开",
            "option": {"title": "离开", "description": "安全离开。"},
        }
        risk = {
            "kind": "event_option",
            "action_id": "event_option:1",
            "label": "顺走",
            "option": {"title": "顺走", "description": "冒险拿走奖励。"},
        }

        filtered = env._apply_low_hp_event_hp_loss_exposure_filter(
            [leave, risk],
            {
                "player": {"hp": 23, "max_hp": 91},
                "run": {"room_model": "EVENT.PUNCH_OFF"},
            },
        )

        self.assertEqual(filtered, [leave])
        self.assertEqual(env._episode_telemetry["event_hp_loss_option_available"], 1.0)
        self.assertEqual(env._episode_telemetry["event_hp_loss_option_low_hp_available"], 1.0)
        self.assertEqual(env._episode_telemetry["event_hp_loss_option_safe_alternative"], 1.0)
        self.assertEqual(env._episode_telemetry["event_hp_loss_option_masked_low_hp"], 1.0)

    def test_forced_singleton_event_hp_loss_remains_available_at_low_hp(self):
        env = _env_stub()
        risk = {
            "kind": "event_option",
            "action_id": "event_option:0",
            "option": {"title": "再撑一会"},
        }

        filtered = env._apply_low_hp_event_hp_loss_exposure_filter(
            [risk],
            {
                "player": {"hp": 12, "max_hp": 91},
                "run": {"room_model": "EVENT.SLIPPERY_BRIDGE"},
            },
        )

        self.assertEqual(filtered, [risk])
        self.assertEqual(env._episode_telemetry["event_hp_loss_option_low_hp_available"], 1.0)
        self.assertEqual(env._episode_telemetry["event_hp_loss_option_masked_low_hp"], 0.0)

    def test_structured_event_hp_delta_loss_is_filtered_at_low_hp(self):
        env = _env_stub()
        safe = {
            "kind": "event_option",
            "action_id": "event_option:0",
            "option": {"title": "Leave", "effect_deltas": {"hp_delta": 0}},
        }
        risk = {
            "kind": "event_option",
            "action_id": "event_option:1",
            "option": {"title": "Take reward", "effect_deltas": {"hp_delta": -12, "relic_gain": True}},
        }

        filtered = env._apply_low_hp_event_hp_loss_exposure_filter(
            [safe, risk],
            {"player": {"hp": 40, "max_hp": 100}},
        )

        self.assertEqual(filtered, [safe])
        self.assertEqual(env._episode_telemetry["event_hp_loss_option_masked_low_hp"], 1.0)

    def test_compact_action_signature_includes_event_effect_deltas(self):
        action = {
            "kind": "event_option",
            "action_id": "event_option:1",
            "option": {
                "title": "战斗",
                "effect_deltas": {
                    "enter_combat": True,
                    "hp_delta": -12,
                    "gold_delta": 50,
                    "card_upgrade_count": 1,
                    "relic_gain": True,
                },
            },
        }

        compact = compact_action_signature(action)

        self.assertTrue(compact["event_enter_combat"])
        self.assertEqual(compact["event_hp_delta"], -12)
        self.assertEqual(compact["event_gold_delta"], 50)
        self.assertEqual(compact["event_card_upgrade_count"], 1)
        self.assertTrue(compact["event_relic_gain"])

    def test_boss_damage_bonus_ignores_player_death_enemy_clear(self):
        env = _env_stub()
        before_obs = {
            "run": {"room_type": "boss", "floor": 17},
            "player": {"hp": 5, "max_hp": 80},
            "combat": {"enemies": [{"hp": 222}]},
        }
        after_obs = {
            "run": {"room_type": "boss", "floor": 17},
            "player": {"hp": 0, "max_hp": 80},
            "combat": {"enemies": []},
        }

        reward = env._boss_damage_bonus_reward(before_obs, after_obs)

        self.assertEqual(reward, 0.0)
        self.assertEqual(env._episode_telemetry["boss_encounter_steps"], 1.0)
        self.assertEqual(env._episode_telemetry["boss_damage_dealt_raw"], 0.0)
        self.assertEqual(env._episode_telemetry["boss_damage_bonus_total"], 0.0)
        self.assertEqual(env._episode_telemetry["boss_damage_death_clear_guarded"], 1.0)
        self.assertEqual(env._episode_telemetry["boss_damage_death_clear_guarded_raw"], 0.0)
        self.assertEqual(
            env._episode_telemetry["boss_damage_death_clear_guarded_remaining_hp_raw"],
            222.0,
        )

    def test_boss_damage_bonus_counts_alive_enemy_clear(self):
        env = _env_stub()
        before_obs = {
            "run": {"room_type": "boss", "floor": 17},
            "player": {"hp": 20, "max_hp": 80},
            "combat": {"enemies": [{"hp": 222}]},
        }
        after_obs = {
            "run": {"room_type": "boss", "floor": 17},
            "player": {"hp": 10, "max_hp": 80},
            "combat": {"enemies": []},
        }

        reward = env._boss_damage_bonus_reward(before_obs, after_obs)

        self.assertAlmostEqual(reward, BOSS_ENEMY_HP_DELTA_PERCENT_SCALE)
        self.assertEqual(env._episode_telemetry["boss_damage_dealt_raw"], 222.0)
        self.assertAlmostEqual(
            env._episode_telemetry["boss_damage_bonus_total"],
            BOSS_ENEMY_HP_DELTA_PERCENT_SCALE,
        )
        self.assertEqual(env._episode_telemetry["boss_damage_death_clear_guarded"], 0.0)

    def test_boss_death_terminal_penalty_applies_full_run_loss_gap(self):
        env = _env_stub()
        env._episode_telemetry["boss_damage_dealt_raw"] = 165.0
        env._episode_telemetry["boss_damage_death_clear_guarded_remaining_hp_raw"] = 57.0
        before_obs = {
            "run": {"room_type": "boss", "floor": 17},
            "player": {"hp": 5, "max_hp": 91},
            "combat": {"enemies": [{"hp": 57}]},
        }
        after_obs = {
            "run": {"room_type": "Boss", "floor": 17},
            "player": {"hp": 0, "max_hp": 91},
            "combat": {"enemies": []},
        }

        penalty = env._boss_death_terminal_penalty(
            before_obs,
            after_obs,
            terminated=True,
            truncated=False,
        )

        expected_damage_ratio = 165.0 / (165.0 + 57.0)
        expected = -(
            BOSS_COMBAT_LOSS_PENALTY_BASE
            + BOSS_COMBAT_LOSS_PENALTY_MISSING_HP_SCALE
            + BOSS_COMBAT_LOSS_DAMAGE_UNDO_PERCENT_SCALE * expected_damage_ratio
        )
        self.assertAlmostEqual(penalty, expected)
        self.assertEqual(env._episode_telemetry["boss_death_terminal_penalty_events"], 1.0)
        self.assertAlmostEqual(
            env._episode_telemetry["boss_death_terminal_damage_ratio"],
            expected_damage_ratio,
        )

    def test_boss_death_terminal_penalty_does_not_apply_to_alive_or_nonterminal(self):
        env = _env_stub()
        boss_obs = {
            "run": {"room_type": "boss", "floor": 17},
            "player": {"hp": 10, "max_hp": 91},
            "combat": {"enemies": []},
        }

        self.assertEqual(
            env._boss_death_terminal_penalty(
                boss_obs,
                boss_obs,
                terminated=True,
                truncated=False,
            ),
            0.0,
        )
        dead_obs = {
            "run": {"room_type": "boss", "floor": 17},
            "player": {"hp": 0, "max_hp": 91},
            "combat": {"enemies": []},
        }
        self.assertEqual(
            env._boss_death_terminal_penalty(
                dead_obs,
                dead_obs,
                terminated=False,
                truncated=False,
            ),
            0.0,
        )

    def test_full_run_death_terminal_penalty_applies_to_late_normal_death(self):
        env = _env_stub()
        env._max_floor_reached = 14
        before_obs = {
            "run": {"room_type": "Normal", "floor": 14, "act_floor": 14},
            "player": {"hp": 7, "max_hp": 91},
            "combat": {"enemies": [{"hp": 44}]},
        }
        after_obs = {
            "run": {"room_type": "Normal", "floor": 14, "act_floor": 14},
            "player": {"hp": 0, "max_hp": 91},
            "combat": {"enemies": []},
        }

        penalty = env._full_run_death_terminal_penalty(
            before_obs,
            after_obs,
            terminated=True,
            truncated=False,
        )

        expected_floor_norm = 14.0 / 17.0
        expected = -(
            FULL_RUN_DEATH_PENALTY_BASE
            + FULL_RUN_DEATH_PENALTY_MISSING_HP_SCALE
            + FULL_RUN_DEATH_PENALTY_LATE_ACT_SCALE * expected_floor_norm
        )
        self.assertAlmostEqual(penalty, expected)
        self.assertEqual(env._episode_telemetry["full_run_death_terminal_penalty_events"], 1.0)
        self.assertAlmostEqual(
            env._episode_telemetry["full_run_death_terminal_floor_norm"],
            expected_floor_norm,
        )

    def test_full_run_death_terminal_penalty_skips_boss_alive_and_truncation(self):
        env = _env_stub()
        normal_alive = {
            "run": {"room_type": "Normal", "floor": 14, "act_floor": 14},
            "player": {"hp": 1, "max_hp": 91},
            "combat": {"enemies": []},
        }
        normal_dead = {
            "run": {"room_type": "Normal", "floor": 14, "act_floor": 14},
            "player": {"hp": 0, "max_hp": 91},
            "combat": {"enemies": []},
        }
        boss_dead = {
            "run": {"room_type": "boss", "floor": 17, "act_floor": 17},
            "player": {"hp": 0, "max_hp": 91},
            "combat": {"enemies": []},
        }

        self.assertEqual(
            env._full_run_death_terminal_penalty(
                normal_alive,
                normal_alive,
                terminated=True,
                truncated=False,
            ),
            0.0,
        )
        self.assertEqual(
            env._full_run_death_terminal_penalty(
                normal_dead,
                normal_dead,
                terminated=True,
                truncated=True,
            ),
            0.0,
        )
        self.assertEqual(
            env._full_run_death_terminal_penalty(
                boss_dead,
                boss_dead,
                terminated=True,
                truncated=False,
            ),
            0.0,
        )

    def test_transition_state_parses_named_live_act_id(self):
        env = _env_stub()
        env._last_obs_raw = {
            "phase": "actions",
            "player": {"hp": 70, "max_hp": 80, "gold": 12},
            "run": {
                "floor": 17,
                "act_id": "ACT.UNDERDOCKS",
                "state_type": "boss",
                "room_type": None,
                "current_act_index": 0,
                "total_floor": 17,
            },
        }

        state = env._transition_state()

        self.assertEqual(state["run"]["act_id"], 1.0)
        self.assertEqual(state["run"]["act_id_raw"], "ACT.UNDERDOCKS")
        self.assertEqual(state["run"]["room_type"], "boss")
        self.assertEqual(state["run"]["current_act_index"], 0.0)


if __name__ == "__main__":
    unittest.main()
