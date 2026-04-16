from __future__ import annotations

import importlib.util
import json
import sys
import threading
from pathlib import Path

import numpy as np


def _load_async_ready_collector_module():
    module_path = Path(__file__).resolve().parents[1] / "sts2_env" / "async_ready_collector.py"
    spec = importlib.util.spec_from_file_location("async_ready_collector_for_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


collector_module = _load_async_ready_collector_module()
AsyncReadyCollector = collector_module.AsyncReadyCollector
AsyncReadyItem = collector_module.AsyncReadyItem


def _obs(marker: int) -> dict[str, np.ndarray]:
    return {
        "feature": np.asarray([marker], dtype=np.float32),
        "action_mask": np.asarray([True, True], dtype=bool),
    }


class _FakeDoneThenResetEnv:
    def __init__(self, *, block_on_post_done_reset: bool, release_event: threading.Event, marker: int) -> None:
        self._block_on_post_done_reset = bool(block_on_post_done_reset)
        self._release_event = release_event
        self._marker = int(marker)
        self._reset_calls = 0

    def reset(self):
        self._reset_calls += 1
        if self._reset_calls >= 2 and self._block_on_post_done_reset:
            self._release_event.wait(timeout=5.0)
        return _obs(self._marker + self._reset_calls), {"action_mask": np.asarray([True, True], dtype=bool)}

    def step(self, action: int):
        del action
        return _obs(self._marker + 100), 1.0, True, False, {
            "aux_targets": {},
            "action_mask": np.asarray([True, True], dtype=bool),
            "encounter_id": "test-encounter",
            "bridge_info": {
                "action": {"action_id": "combat:end_turn", "kind": "combat"},
                "phase_before": "combat",
                "phase_after": "reward",
                "screen_before": "COMBAT",
                "screen_after": "REWARDS",
                "room_type_before": "COMBAT",
                "room_type_after": "COMBAT",
                "combat_in_progress_before": True,
                "combat_in_progress_after": False,
                "truncation_reason": None,
                "action_error": None,
                "card_selection_before": {
                    "screen_type": "NPlayerHand",
                    "selected_count": 0,
                    "confirm_ready": False,
                    "selection_ready": False,
                    "opened_age_ms": 120,
                },
                "card_selection_after": {
                    "screen_type": "NPlayerHand",
                    "selected_count": 1,
                    "confirm_ready": True,
                    "selection_ready": True,
                    "opened_age_ms": 510,
                },
                "step_timing_ms": {"after_wait": 123.4},
                "step_timing_counts": {"stable_iterations": 7},
            },
        }

    def close(self) -> None:
        self._release_event.set()


def test_async_ready_collector_publishes_terminal_transition_before_post_done_reset() -> None:
    release_event = threading.Event()

    def env_factory():
        return _FakeDoneThenResetEnv(block_on_post_done_reset=True, release_event=release_event, marker=10)

    collector = AsyncReadyCollector(
        [env_factory],
        reset_watchdog_timeout_s=10.0,
        initial_reset_watchdog_timeout_s=10.0,
    )
    collector.start()
    try:
        initial_items = collector.drain_ready(min_items=1, timeout_s=1.0)
        assert len(initial_items) == 1
        assert initial_items[0].ready_for_action is True

        collector.dispatch_actions([0], [0])
        transition_items = collector.drain_ready(min_items=1, timeout_s=1.0, return_on_restart=True)
        assert len(transition_items) == 1
        transition = transition_items[0]
        assert isinstance(transition, AsyncReadyItem)
        assert transition.ready_for_action is False
        assert transition.obs is None
        assert transition.terminated is True
        assert transition.truncated is False
        assert transition.reward == 1.0
        assert isinstance(transition.transition_info, dict)
    finally:
        release_event.set()
        collector.close()


def test_async_ready_collector_restarts_single_worker_after_stale_post_done_reset() -> None:
    release_event = threading.Event()
    creation_counter = {"count": 0}

    def env_factory():
        creation_counter["count"] += 1
        return _FakeDoneThenResetEnv(
            block_on_post_done_reset=(creation_counter["count"] == 1),
            release_event=release_event,
            marker=100 * creation_counter["count"],
        )

    collector = AsyncReadyCollector(
        [env_factory],
        reset_watchdog_timeout_s=0.2,
        initial_reset_watchdog_timeout_s=1.0,
        restart_cooldown_s=0.0,
    )
    collector.start()
    try:
        initial_items = collector.drain_ready(min_items=1, timeout_s=1.0)
        assert len(initial_items) == 1
        collector.dispatch_actions([0], [0])

        transition_items = collector.drain_ready(min_items=1, timeout_s=1.0, return_on_restart=True)
        assert len(transition_items) == 1
        assert transition_items[0].ready_for_action is False

        restarted_items = collector.drain_ready(min_items=1, timeout_s=1.0, return_on_restart=True)
        assert restarted_items == []

        restart_events = collector.pop_restart_events()
        assert len(restart_events) == 1
        assert restart_events[0]["env_id"] == 0
        assert restart_events[0]["phase"] == "resetting_after_done"
        assert restart_events[0]["generation"] == 1

        replacement_ready = collector.drain_ready(min_items=1, timeout_s=1.0)
        assert len(replacement_ready) == 1
        assert replacement_ready[0].ready_for_action is True
        assert replacement_ready[0].generation == 1
    finally:
        release_event.set()
        collector.close()


def test_async_ready_collector_logs_terminal_and_restart_events(tmp_path: Path) -> None:
    release_event = threading.Event()
    creation_counter = {"count": 0}
    event_log_path = tmp_path / "reset_events.jsonl"

    def env_factory():
        creation_counter["count"] += 1
        return _FakeDoneThenResetEnv(
            block_on_post_done_reset=(creation_counter["count"] == 1),
            release_event=release_event,
            marker=200 * creation_counter["count"],
        )

    collector = AsyncReadyCollector(
        [env_factory],
        reset_watchdog_timeout_s=0.2,
        initial_reset_watchdog_timeout_s=1.0,
        restart_cooldown_s=0.0,
        event_log_path=str(event_log_path),
    )
    collector.start()
    try:
        initial_items = collector.drain_ready(min_items=1, timeout_s=1.0)
        assert len(initial_items) == 1
        collector.dispatch_actions([0], [0])

        transition_items = collector.drain_ready(min_items=1, timeout_s=1.0, return_on_restart=True)
        assert len(transition_items) == 1
        assert transition_items[0].ready_for_action is False

        restarted_items = collector.drain_ready(min_items=1, timeout_s=1.0, return_on_restart=True)
        assert restarted_items == []
        collector.pop_restart_events()

        replacement_ready = collector.drain_ready(min_items=1, timeout_s=1.0)
        assert len(replacement_ready) == 1
    finally:
        release_event.set()
        collector.close()

    lines = [json.loads(line) for line in event_log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    terminal_events = [entry for entry in lines if entry.get("event") == "episode_terminal"]
    restart_events = [entry for entry in lines if entry.get("event") == "worker_restart"]

    assert len(terminal_events) == 1
    terminal = terminal_events[0]
    assert terminal["env_id"] == 0
    assert terminal["action_id"] == "combat:end_turn"
    assert terminal["phase_before"] == "combat"
    assert terminal["phase_after"] == "reward"
    assert terminal["combat_in_progress_before"] is True
    assert terminal["combat_in_progress_after"] is False
    assert terminal["card_selection_screen_type_before"] == "NPlayerHand"
    assert terminal["card_selection_screen_type_after"] == "NPlayerHand"
    assert terminal["card_selection_selected_count_before"] == 0
    assert terminal["card_selection_selected_count_after"] == 1
    assert terminal["card_selection_confirm_ready_before"] is False
    assert terminal["card_selection_confirm_ready_after"] is True
    assert terminal["card_selection_selection_ready_before"] is False
    assert terminal["card_selection_selection_ready_after"] is True
    assert terminal["card_selection_opened_age_ms_before"] == 120
    assert terminal["card_selection_opened_age_ms_after"] == 510
    assert terminal["after_wait_ms"] == 123.4
    assert terminal["stable_iterations"] == 7
    assert terminal["worker_restart_happened"] is False

    assert len(restart_events) == 1
    restart = restart_events[0]
    assert restart["env_id"] == 0
    assert restart["phase"] == "resetting_after_done"
    assert restart["worker_restart_happened"] is True
