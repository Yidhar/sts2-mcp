from __future__ import annotations

import queue
import threading
from pathlib import Path
from unittest import mock

import pytest

from sts2_env import headless_sim_bridge_client as module
from sts2_env.headless_sim_bridge_client import (
    HeadlessSimBridgeClient,
    HeadlessSimError,
    HeadlessSimProtocolError,
    HeadlessSimStepRejected,
    HeadlessSimUnsettledError,
    resolve_headless_sim_exe,
)


def _bare_client() -> HeadlessSimBridgeClient:
    client = HeadlessSimBridgeClient.__new__(HeadlessSimBridgeClient)
    client._episode_counter = 0
    client._current_episode_id = ""
    client._combat_episode_active = False
    client._last_observation = None
    client._last_legal_actions = []
    client._rpc_timeout_s = 1.0
    client._transition_poll_budget_s = 1.0
    client._transition_poll_max_attempts = 8
    return client


def test_resolver_uses_explicit_environment_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    executable = tmp_path / "HeadlessSim.exe"
    executable.write_bytes(b"fake")
    monkeypatch.setenv("STS2_HEADLESS_SIM_EXE", str(executable))
    assert resolve_headless_sim_exe() == executable.resolve()


def test_resolver_rejects_missing_explicit_path(tmp_path: Path) -> None:
    with pytest.raises(HeadlessSimError, match="not found"):
        resolve_headless_sim_exe(tmp_path / "missing.exe")


def test_reset_forwards_seed_and_timeout() -> None:
    client = _bare_client()
    client._rpc = mock.Mock(return_value={"state_type": "map"})
    with mock.patch.object(module, "_build_bridge_step_response", return_value={"episode_id": "ep"}):
        result = client.reset(character="CHARACTER.IRONCLAD", seed="ABC123", timeout_ms=12_000)
    assert result["episode_id"] == "ep"
    client._rpc.assert_called_once_with(
        "reset",
        {"character_id": "IRONCLAD", "seed": "ABC123"},
        timeout_s=12.0,
    )


@pytest.mark.parametrize("revival_budget", (-1, 64))
def test_full_run_reset_forwards_native_revival_build(revival_budget: int) -> None:
    client = _bare_client()
    client._rpc = mock.Mock(return_value={"state_type": "event"})
    with mock.patch.object(module, "_build_bridge_step_response", return_value={"episode_id": "ep"}):
        client.reset(
            additional_relics=["RELIC.LIZARD_TAIL"],
            training_revival_budget=revival_budget,
        )
    client._rpc.assert_called_once_with(
        "reset",
        {
            "build": {
                "additional_relics": [{"id": "LIZARD_TAIL"}],
                "training_revival_budget": revival_budget,
            }
        },
        timeout_s=45.0,
    )


def test_rebind_observes_current_run_instead_of_resetting() -> None:
    client = _bare_client()
    client._current_episode_id = "sim-ep-1"
    client._rpc = mock.Mock(return_value={"state_type": "combat"})
    with mock.patch.object(module, "_build_bridge_step_response", return_value={"episode_id": "sim-ep-1"}):
        client.reset(rebind_active_run=True)
    client._rpc.assert_called_once_with("state", timeout_s=45.0)


def test_step_rejects_stale_episode_before_dispatch() -> None:
    client = _bare_client()
    client._current_episode_id = "sim-ep-2"
    client._rpc = mock.Mock()
    with pytest.raises(HeadlessSimError, match="episode mismatch"):
        client.step("sim-ep-old", action_index=0)
    client._rpc.assert_not_called()


def test_combat_post_end_boundary_is_a_typed_victory_not_a_deadlock() -> None:
    client = _bare_client()
    client._combat_episode_active = True
    active = module._build_bridge_step_response(
        client,
        {
            "state_type": "combat",
            "battle": {
                "player": {"hp": 17, "max_hp": 50},
                "enemies": [{"id": "MONSTER.TEST", "hp": 1, "max_hp": 10}],
            },
            "training_revival_budget": -1,
            "training_revivals_used": 3,
            "training_player_hp_lost": 120,
            "legal_actions": [{"action": "end_turn"}],
        },
        episode_started=True,
        reward=0.0,
    )
    assert not active["done"]

    victory = module._build_bridge_step_response(
        client,
        {
            "state_type": "combat_post_end_pending",
            "training_revival_budget": -1,
            "training_revivals_used": 3,
            "training_player_hp_lost": 120,
            "legal_actions": [],
        },
        episode_started=False,
        reward=0.0,
    )
    assert victory["done"]
    assert victory["terminal_reason"] == "combat_victory"
    assert victory["obs"]["state_type"] == "combat_victory"
    assert victory["obs"]["player"]["hp"] == 17
    assert victory["obs"]["_training"]["revivals_used"] == 3
    assert "run_outcome" not in victory["obs"]["_sim_raw"]
    assert "sim_run_outcome" not in victory["info"]
    assert victory["legal_actions"] == []


@pytest.mark.parametrize("outcome", ["victory", "defeat"])
def test_full_run_terminal_uses_authoritative_outcome_without_player(
    outcome: str,
) -> None:
    client = _bare_client()
    result = module._build_bridge_step_response(
        client,
        {
            "state_type": "game_over",
            "terminal": True,
            "truncated": False,
            "is_actionable": False,
            "run_outcome": outcome,
            "legal_actions": [],
        },
        episode_started=True,
        reward=0.0,
    )

    assert result["done"] is True
    assert result["terminal_reason"] == f"run_{outcome}"
    assert result["info"]["terminal_reason"] == f"run_{outcome}"
    assert result["info"]["sim_run_outcome"] == outcome
    assert result["obs"]["player"] == {}
    assert result["obs"]["_sim_raw"]["run_outcome"] == outcome
    assert result["legal_actions"] == []


def test_combat_episode_terminal_defeat_keeps_combat_scope() -> None:
    client = _bare_client()
    client._combat_episode_active = True
    result = module._build_bridge_step_response(
        client,
        {
            "state_type": "game_over",
            "terminal": True,
            "truncated": False,
            "is_actionable": False,
            "run_outcome": "defeat",
            "legal_actions": [],
        },
        episode_started=True,
        reward=0.0,
    )

    assert result["terminal_reason"] == "combat_defeat"
    assert result["info"]["sim_run_outcome"] == "defeat"


@pytest.mark.parametrize("run_outcome", [None, "ongoing", 7])
def test_full_run_terminal_rejects_missing_or_invalid_outcome_before_cache_commit(
    run_outcome: object,
) -> None:
    client = _bare_client()
    previous_observation = {"sentinel": True}
    previous_actions = [{"action_id": "previous"}]
    client._last_observation = previous_observation
    client._last_legal_actions = previous_actions
    state = {
        "state_type": "game_over",
        "terminal": True,
        "truncated": False,
        "is_actionable": False,
        "legal_actions": [],
    }
    if run_outcome is not None:
        state["run_outcome"] = run_outcome

    with pytest.raises(HeadlessSimProtocolError, match="run_outcome"):
        module._build_bridge_step_response(
            client,
            state,
            episode_started=False,
            reward=0.0,
        )

    assert client._last_observation is previous_observation
    assert client._last_legal_actions is previous_actions


def test_full_run_terminal_rejects_reason_outcome_contradiction() -> None:
    client = _bare_client()
    with pytest.raises(HeadlessSimProtocolError, match="contradicts"):
        module._build_bridge_step_response(
            client,
            {
                "state_type": "game_over",
                "terminal": True,
                "is_actionable": False,
                "run_outcome": "victory",
                "terminal_reason": "run_defeat",
                "legal_actions": [],
            },
            episode_started=False,
            reward=0.0,
        )


@pytest.mark.parametrize("terminal_reason", ["death", "action_execution_error", "", 7])
def test_full_run_terminal_rejects_noncanonical_reason(terminal_reason: object) -> None:
    client = _bare_client()
    with pytest.raises(HeadlessSimProtocolError, match="terminal_reason"):
        module._build_bridge_step_response(
            client,
            {
                "state_type": "game_over",
                "terminal": True,
                "is_actionable": False,
                "run_outcome": "victory",
                "terminal_reason": terminal_reason,
                "legal_actions": [],
            },
            episode_started=False,
            reward=0.0,
        )


def test_nonterminal_state_cannot_expose_terminal_reason() -> None:
    client = _bare_client()
    with pytest.raises(HeadlessSimProtocolError, match="non-terminal.*terminal_reason"):
        module._build_bridge_step_response(
            client,
            {
                "state_type": "map",
                "terminal": False,
                "is_actionable": True,
                "terminal_reason": "death",
                "legal_actions": [{"action": "choose_map_node", "col": 0, "row": 1}],
            },
            episode_started=False,
            reward=0.0,
        )


def test_truncated_state_requires_reason_and_accepts_explicit_reason() -> None:
    client = _bare_client()
    state = {
        "state_type": "map",
        "terminal": False,
        "truncated": True,
        "is_actionable": False,
        "legal_actions": [],
    }
    with pytest.raises(HeadlessSimProtocolError, match="requires.*terminal_reason"):
        module._build_bridge_step_response(
            client, state, episode_started=False, reward=0.0
        )

    accepted = module._build_bridge_step_response(
        client,
        {**state, "terminal_reason": "collection_budget"},
        episode_started=False,
        reward=0.0,
    )
    assert accepted["truncated"] is True
    assert accepted["terminal_reason"] == "collection_budget"


def test_combat_post_end_boundary_rejects_full_run_outcome() -> None:
    client = _bare_client()
    client._combat_episode_active = True
    with pytest.raises(HeadlessSimProtocolError, match="post-end.*run_outcome"):
        module._build_bridge_step_response(
            client,
            {
                "state_type": "combat_post_end_pending",
                "terminal": False,
                "is_actionable": False,
                "run_outcome": "defeat",
                "legal_actions": [],
            },
            episode_started=True,
            reward=0.0,
        )


def test_nonterminal_or_truncated_state_cannot_expose_final_run_outcome() -> None:
    client = _bare_client()
    with pytest.raises(HeadlessSimProtocolError, match="non-terminal"):
        module._build_bridge_step_response(
            client,
            {
                "state_type": "map",
                "terminal": False,
                "run_outcome": "victory",
                "legal_actions": [{"action": "choose_map_node", "col": 0, "row": 1}],
            },
            episode_started=False,
            reward=0.0,
        )
    with pytest.raises(HeadlessSimProtocolError, match="non-terminal"):
        module._build_bridge_step_response(
            client,
            {
                "state_type": "game_over",
                "terminal": False,
                "truncated": True,
                "run_outcome": "defeat",
                "legal_actions": [],
            },
            episode_started=False,
            reward=0.0,
        )


def test_step_rejection_is_typed_and_mutation_is_sent_once() -> None:
    client = _bare_client()
    client._current_episode_id = "sim-ep-1"
    client._last_legal_actions = [
        {"action_id": "sim:0:proceed", "_sim_raw": {"action": "proceed"}}
    ]
    client._rpc = mock.Mock(
        return_value={
            "accepted": False,
            "error": "invalid action",
            "state": {},
        }
    )

    with pytest.raises(HeadlessSimStepRejected, match="invalid action") as caught:
        client.step("sim-ep-1", action_index=0)

    assert caught.value.method == "step"
    assert caught.value.response["accepted"] is False
    client._rpc.assert_called_once_with("step", {"action": "proceed"}, timeout_s=20.0)


def test_authoritative_pending_step_uses_only_bounded_read_only_polling() -> None:
    client = _bare_client()
    client._current_episode_id = "sim-ep-1"
    client._last_legal_actions = [
        {
            "action_id": "sim:0:choose_map_node",
            "_sim_raw": {"action": "choose_map_node", "col": 5, "row": 13},
        }
    ]
    client._rpc = mock.Mock(
        side_effect=[
            {
                "accepted": True,
                "action_committed": True,
                "settlement_status": "unsettled",
                "transition_token": "transition-7",
                "state": {
                    "state_type": "combat_start_pending",
                    "is_actionable": False,
                    "terminal": False,
                    "legal_actions": [],
                },
                "reward": 0.0,
            },
            {
                "accepted": True,
                "action_committed": True,
                "settlement_status": "pending",
                "transition_token": "transition-7",
                "state": {
                    "state_type": "combat_start_pending",
                    "is_actionable": False,
                    "terminal": False,
                    "legal_actions": [],
                },
            },
            {
                "accepted": True,
                "action_committed": True,
                "settlement_status": "actionable",
                "transition_token": "transition-7",
                "state": {
                    "state_type": "combat",
                    "is_actionable": True,
                    "terminal": False,
                    "legal_actions": [{"action": "end_turn"}],
                },
            },
        ]
    )

    result = client.step("sim-ep-1", action_index=0)

    assert [call.args[0] for call in client._rpc.call_args_list] == [
        "step",
        "poll_transition",
        "poll_transition",
    ]
    assert result["done"] is False
    assert len(result["legal_actions"]) == 1
    assert result["legal_actions"][0]["kind"] == "end_turn"


def test_persistent_pending_poll_uses_real_wall_clock_budget() -> None:
    client = _bare_client()
    client._transition_poll_budget_s = 1.0
    client._transition_poll_max_attempts = 8
    client._current_episode_id = "sim-ep-1"
    client._last_legal_actions = [
        {"action_id": "sim:0:proceed", "_sim_raw": {"action": "proceed"}}
    ]
    response = {
        "accepted": True,
        "action_committed": True,
        "settlement_status": "pending",
        "transition_token": "transition-pending",
        "state": {
            "state_type": "combat_start_pending",
            "is_actionable": False,
            "terminal": False,
            "legal_actions": [],
        },
    }
    client._rpc = mock.Mock(return_value=response)

    class FakeClock:
        now = 0.0

        def monotonic(self) -> float:
            return self.now

        def sleep(self, duration: float) -> None:
            assert duration > 0.0
            self.now += duration

    clock = FakeClock()
    with (
        mock.patch.object(module.time, "monotonic", side_effect=clock.monotonic),
        mock.patch.object(module.time, "sleep", side_effect=clock.sleep),
        pytest.raises(HeadlessSimUnsettledError, match="bounded read-only poll budget"),
    ):
        client.step("sim-ep-1", action_index=0)

    assert client._rpc.call_count == 1 + client._transition_poll_max_attempts
    assert clock.now >= 0.75


def test_combat_episode_stops_at_authoritative_post_end_boundary_without_polling() -> None:
    client = _bare_client()
    client._combat_episode_active = True
    client._current_episode_id = "sim-ep-1"
    client._last_observation = {
        "state_type": "combat",
        "player": {"hp": 17},
        "combat": {"in_progress": True},
        "available_actions": [{"action_id": "sim:0:end_turn"}],
    }
    client._last_legal_actions = [
        {"action_id": "sim:0:end_turn", "_sim_raw": {"action": "end_turn"}}
    ]
    client._rpc = mock.Mock(
        return_value={
            "accepted": True,
            "action_committed": True,
            "settlement_status": "unsettled",
            "transition_token": "transition-combat-victory",
            "state": {
                "state_type": "combat_post_end_pending",
                "is_actionable": False,
                "terminal": False,
                "legal_actions": [],
            },
        }
    )

    result = client.step("sim-ep-1", action_index=0)

    assert result["done"] is True
    assert result["terminal_reason"] == "combat_victory"
    client._rpc.assert_called_once_with("step", {"action": "end_turn"}, timeout_s=20.0)


def test_legacy_actionless_step_is_not_guessed_to_be_pending() -> None:
    client = _bare_client()
    client._current_episode_id = "sim-ep-1"
    client._last_legal_actions = [
        {"action_id": "sim:0:proceed", "_sim_raw": {"action": "proceed"}}
    ]
    client._rpc = mock.Mock(
        return_value={
            "accepted": True,
            "state": {
                "state_type": "combat_start_pending",
                "terminal": False,
                "legal_actions": [],
            },
        }
    )

    with pytest.raises(HeadlessSimUnsettledError, match="zero legal actions"):
        client.step("sim-ep-1", action_index=0)

    # No state-type guess, no second mutation, and no uncorrelated state poll.
    client._rpc.assert_called_once_with("step", {"action": "proceed"}, timeout_s=20.0)


def test_authoritative_terminal_surface_rejects_legal_actions() -> None:
    client = _bare_client()
    client._current_episode_id = "sim-ep-1"
    client._last_legal_actions = [
        {"action_id": "sim:0:proceed", "_sim_raw": {"action": "proceed"}}
    ]
    client._rpc = mock.Mock(
        return_value={
            "accepted": True,
            "action_committed": True,
            "settlement_status": "terminal",
            "transition_token": "transition-8",
            "state": {
                "state_type": "game_over",
                "is_actionable": False,
                "terminal": True,
                "legal_actions": [{"action": "proceed"}],
            },
        }
    )

    with pytest.raises(HeadlessSimProtocolError, match="inconsistent"):
        client.step("sim-ep-1", action_index=0)

    client._rpc.assert_called_once()


class _Stream:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _Process:
    def __init__(self) -> None:
        self.stdin = _Stream()
        self.stdout = _Stream()
        self.stderr = _Stream()
        self.pid = 123
        self.returncode = 0
        self.killed = False
        self.waits = 0

    def wait(self, timeout: float | None = None) -> int:
        self.waits += 1
        return 0

    def poll(self) -> int:
        return 0

    def kill(self) -> None:
        self.killed = True


def test_close_is_idempotent_and_reaps_process() -> None:
    client = _bare_client()
    process = _Process()
    client._lock = threading.Lock()
    client._proc = process
    client._reader_stop = threading.Event()
    client._reader_thread = None
    client._stderr_thread = None
    client._stdout_queue = queue.Queue(maxsize=2)
    client._hang_log_handle = None
    client._hang_log_path = None

    client.close()
    client.close()

    assert client._proc is None
    assert process.waits >= 1
    assert process.stdin.closed
    assert process.stdout.closed
    assert process.stderr.closed
