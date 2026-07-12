from __future__ import annotations

from unittest import mock

from launcher_watchdog import (
    HealthSnapshot,
    InstanceAction,
    StateSnapshot,
    Watchdog,
    WatchdogInstanceView,
    _default_health_probe,
)

BASE_URL = "http://127.0.0.1:9123"


def _view() -> WatchdogInstanceView:
    return WatchdogInstanceView(
        instance_id=0,
        process_alive=True,
        base_url=BASE_URL,
        token="token",
    )


class HealthProbe:
    def __init__(self, snapshots: list[HealthSnapshot]) -> None:
        self.snapshots = list(snapshots)

    def __call__(self, base_url: str, token: str, timeout_s: float) -> HealthSnapshot:
        return self.snapshots.pop(0)


class StateProbe:
    def __init__(self, snapshots: list[StateSnapshot | None]) -> None:
        self.snapshots = list(snapshots)

    def __call__(self, base_url: str, token: str, timeout_s: float) -> StateSnapshot | None:
        return self.snapshots.pop(0)


def _healthy_v2(*, tick: int, operation: str | None = None) -> HealthSnapshot:
    return HealthSnapshot(
        transport_alive=True,
        game_thread_alive=True,
        pump_tick=tick,
        ms_since_last_pump=5,
        active_operation=operation,
        source="v2",
    )


def test_state_probe_failure_is_main_thread_strike_and_restarts() -> None:
    watchdog = Watchdog(
        strike_threshold=2,
        health_probe=HealthProbe([_healthy_v2(tick=1), _healthy_v2(tick=2)]),
        state_version_probe=StateProbe([None, None]),
    )
    first = watchdog.evaluate([_view()], now_unix_s=0.0).per_instance[0]
    second = watchdog.evaluate([_view()], now_unix_s=1.0).per_instance[0]
    assert first.action is InstanceAction.HEALTHY
    assert "main_thread_strike_1_state_probe_failed" in first.reason
    assert second.action is InstanceAction.KILL_AND_RESTART
    assert "main_thread_failed_2_consecutive_state_probe_failed" in second.reason


def test_game_thread_and_pump_health_include_active_operation() -> None:
    snapshots = [
        HealthSnapshot(
            transport_alive=True,
            game_thread_alive=False,
            pump_tick=8,
            ms_since_last_pump=60_000,
            active_operation="env.step",
            source="v2",
        )
    ]
    watchdog = Watchdog(
        strike_threshold=1,
        health_probe=HealthProbe(snapshots),
        state_version_probe=StateProbe([StateSnapshot(4, "COMBAT")]),
    )
    decision = watchdog.evaluate([_view()], now_unix_s=0.0).per_instance[0]
    assert decision.action is InstanceAction.KILL_AND_RESTART
    assert "game_thread_not_alive" in decision.reason
    assert "operation_env.step" in decision.reason


def test_stable_decision_idle_is_never_killed_for_unchanged_version() -> None:
    watchdog = Watchdog(
        state_stall_threshold_s=10.0,
        health_probe=HealthProbe([
            _healthy_v2(tick=10),
            _healthy_v2(tick=11),
            _healthy_v2(tick=12),
        ]),
        state_version_probe=StateProbe([
            StateSnapshot(40, "COMBAT", stable_decision=True),
            StateSnapshot(40, "COMBAT", stable_decision=True),
            StateSnapshot(40, "COMBAT", stable_decision=True),
        ]),
    )
    for now in (0.0, 100.0, 10_000.0):
        decision = watchdog.evaluate([_view()], now_unix_s=now).per_instance[0]
        assert decision.action is InstanceAction.HEALTHY
        assert decision.reason == "stable_decision_idle_COMBAT"


def test_active_operation_disables_stable_decision_idle_exemption() -> None:
    watchdog = Watchdog(
        state_stall_threshold_s=10.0,
        health_probe=HealthProbe([
            _healthy_v2(tick=10, operation="env.step"),
            _healthy_v2(tick=11, operation="env.step"),
        ]),
        state_version_probe=StateProbe([
            StateSnapshot(40, "COMBAT", stable_decision=True),
            StateSnapshot(40, "COMBAT", stable_decision=True),
        ]),
    )
    watchdog.evaluate([_view()], now_unix_s=0.0)
    decision = watchdog.evaluate([_view()], now_unix_s=11.0).per_instance[0]
    assert decision.action is InstanceAction.KILL_AND_RESTART
    assert "silent_hang_state_version_frozen_at_40" in decision.reason


def test_default_probe_prefers_v2_health_without_legacy_call() -> None:
    response = mock.Mock(status_code=200)
    response.json.return_value = {
        "transport_alive": True,
        "game_thread_alive": True,
        "session_id": "session",
        "api_version": "2.0.0",
        "schema_version": "schema",
        "pump_tick": 9,
        "ms_since_last_pump": 4,
        "queue_depth": 0,
        "active_operation": None,
    }
    with mock.patch("requests.get", return_value=response) as get:
        snapshot = _default_health_probe(BASE_URL, "token", 1.0)
    assert snapshot.is_v2
    assert snapshot.game_thread_alive is True
    assert snapshot.pump_tick == 9
    assert get.call_count == 1
    assert get.call_args.args[0] == f"{BASE_URL}/v2/health"
