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
    resolve_headless_sim_exe,
)


def _bare_client() -> HeadlessSimBridgeClient:
    client = HeadlessSimBridgeClient.__new__(HeadlessSimBridgeClient)
    client._episode_counter = 0
    client._current_episode_id = ""
    client._combat_episode_active = False
    client._last_observation = None
    client._last_legal_actions = []
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


def test_full_run_reset_forwards_native_revival_build() -> None:
    client = _bare_client()
    client._rpc = mock.Mock(return_value={"state_type": "event"})
    with mock.patch.object(module, "_build_bridge_step_response", return_value={"episode_id": "ep"}):
        client.reset(
            additional_relics=["RELIC.LIZARD_TAIL"],
            training_revival_budget=-1,
        )
    client._rpc.assert_called_once_with(
        "reset",
        {
            "build": {
                "additional_relics": [{"id": "LIZARD_TAIL"}],
                "training_revival_budget": -1,
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
            "legal_actions": [],
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
    assert victory["legal_actions"] == []


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
