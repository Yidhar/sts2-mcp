from __future__ import annotations

import sys
import unittest
from pathlib import Path

RL_AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(RL_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(RL_AGENT_ROOT))

from launcher_watchdog import (  # noqa: E402
    InstanceAction,
    Watchdog,
    WatchdogInstanceView,
)


class _StaticProbe:
    """Scripted health probe: per-instance queue of return values."""

    def __init__(self, per_instance_results: dict[str, list[bool]]):
        self.per_instance_results = per_instance_results
        self.calls: list[str] = []

    def __call__(self, base_url: str, token: str, timeout_s: float) -> bool:
        self.calls.append(base_url)
        # Keyed by base_url since each instance has a unique URL.
        queue = self.per_instance_results.get(base_url, [True])
        if not queue:
            return True
        value = queue.pop(0)
        return value


class _StaticBytesFn:
    """Yields a scripted sequence of directory byte totals."""

    def __init__(self, sequence: list[int]):
        self.sequence = list(sequence)

    def __call__(self, path: Path) -> int:
        if not self.sequence:
            return 0
        return self.sequence.pop(0)


class _StaticStateVersionProbe:
    """Per-instance queue of (state_version, screen) values.

    Entries can also be:
    - a plain int → paired with a default active screen (``COMBAT``)
    - None → simulates /state unreachable
    """

    def __init__(
        self,
        per_instance_results: dict[
            str, list[tuple[int, str] | int | None]
        ],
        *,
        default_screen: str = "COMBAT",
    ):
        self.per_instance_results = per_instance_results
        self._default_screen = default_screen

    def __call__(
        self, base_url: str, token: str, timeout_s: float,
    ) -> tuple[int, str] | None:
        queue = self.per_instance_results.get(base_url, [])
        if not queue:
            return None
        value = queue.pop(0)
        if value is None:
            return None
        if isinstance(value, tuple):
            return value
        return int(value), self._default_screen


def _view(instance_id: int, alive=True, bound=True) -> WatchdogInstanceView:
    base_url = f"http://127.0.0.1:900{instance_id}"
    token = "t" if bound else None
    return WatchdogInstanceView(
        instance_id=instance_id,
        process_alive=alive,
        base_url=base_url if bound else None,
        token=token,
    )


class WatchdogTest(unittest.TestCase):
    def test_all_healthy_no_action(self) -> None:
        wd = Watchdog(health_probe=_StaticProbe({"http://127.0.0.1:9000": [True]}))
        decision = wd.evaluate([_view(0)], now_unix_s=100.0)
        self.assertEqual(len(decision.per_instance), 1)
        self.assertEqual(decision.per_instance[0].action, InstanceAction.HEALTHY)

    def test_strike_then_kill_after_threshold(self) -> None:
        probe = _StaticProbe({"http://127.0.0.1:9000": [False, False, False]})
        wd = Watchdog(strike_threshold=3, health_probe=probe)
        # Tick 1: first failure → strike 1, still healthy
        d1 = wd.evaluate([_view(0)], now_unix_s=1.0)
        self.assertEqual(d1.per_instance[0].action, InstanceAction.HEALTHY)
        # Tick 2: second failure → strike 2
        d2 = wd.evaluate([_view(0)], now_unix_s=2.0)
        self.assertEqual(d2.per_instance[0].action, InstanceAction.HEALTHY)
        # Tick 3: third failure → kill
        d3 = wd.evaluate([_view(0)], now_unix_s=3.0)
        self.assertEqual(d3.per_instance[0].action, InstanceAction.KILL_AND_RESTART)

    def test_success_resets_strikes(self) -> None:
        probe = _StaticProbe(
            {"http://127.0.0.1:9000": [False, False, True, False, False]}
        )
        wd = Watchdog(strike_threshold=3, health_probe=probe)
        wd.evaluate([_view(0)], now_unix_s=1.0)
        wd.evaluate([_view(0)], now_unix_s=2.0)
        d3 = wd.evaluate([_view(0)], now_unix_s=3.0)  # success → strike reset
        self.assertEqual(d3.per_instance[0].action, InstanceAction.HEALTHY)
        # Two more failures shouldn't reach threshold since counter reset.
        d4 = wd.evaluate([_view(0)], now_unix_s=4.0)
        d5 = wd.evaluate([_view(0)], now_unix_s=5.0)
        self.assertEqual(d4.per_instance[0].action, InstanceAction.HEALTHY)
        self.assertEqual(d5.per_instance[0].action, InstanceAction.HEALTHY)

    def test_dead_process_triggers_restart(self) -> None:
        """Crashed/exited game process must be resurrected by watchdog —
        launcher has no separate dead-process reaper.
        """
        probe = _StaticProbe({})  # shouldn't be called; process is gone
        wd = Watchdog(health_probe=probe)
        d = wd.evaluate([_view(0, alive=False)], now_unix_s=1.0)
        self.assertEqual(d.per_instance[0].action, InstanceAction.KILL_AND_RESTART)
        self.assertEqual(d.per_instance[0].reason, "process_exited")
        self.assertEqual(probe.calls, [])

    def test_not_yet_bound_is_abstain(self) -> None:
        probe = _StaticProbe({})
        wd = Watchdog(health_probe=probe)
        d = wd.evaluate([_view(0, bound=False)], now_unix_s=1.0)
        self.assertEqual(d.per_instance[0].action, InstanceAction.HEALTHY)
        self.assertEqual(d.per_instance[0].reason, "not_bound_yet")
        self.assertEqual(probe.calls, [])

    def test_strike_reset_after_kill_signal(self) -> None:
        """After kill+restart emission, the counter zeroes so the launcher
        isn't retriggered while the process is mid-restart."""
        probe = _StaticProbe({"http://127.0.0.1:9000": [False, False, False, False]})
        wd = Watchdog(strike_threshold=3, health_probe=probe)
        wd.evaluate([_view(0)], now_unix_s=1.0)
        wd.evaluate([_view(0)], now_unix_s=2.0)
        d3 = wd.evaluate([_view(0)], now_unix_s=3.0)
        self.assertEqual(d3.per_instance[0].action, InstanceAction.KILL_AND_RESTART)
        d4 = wd.evaluate([_view(0)], now_unix_s=4.0)
        self.assertEqual(d4.per_instance[0].action, InstanceAction.HEALTHY)
        # Internal counter is 1 (the 4th failure); threshold is 3, so we need two more.

    def test_per_instance_independence(self) -> None:
        probe = _StaticProbe({
            "http://127.0.0.1:9000": [False, False, False],  # instance 0 fails
            "http://127.0.0.1:9001": [True, True, True],      # instance 1 healthy
        })
        wd = Watchdog(strike_threshold=3, health_probe=probe)
        for tick in range(3):
            decision = wd.evaluate([_view(0), _view(1)], now_unix_s=tick + 1.0)
        killed = [d for d in decision.per_instance if d.action == InstanceAction.KILL_AND_RESTART]
        healthy = [d for d in decision.per_instance if d.action == InstanceAction.HEALTHY]
        self.assertEqual(len(killed), 1)
        self.assertEqual(killed[0].instance_id, 0)
        self.assertEqual(len(healthy), 1)
        self.assertEqual(healthy[0].instance_id, 1)

    def test_log_flood_detected(self) -> None:
        # Sequence: 100MB → 150MB in 1 second = 50 MB/s growth (above 10MB/s trip)
        bytes_fn = _StaticBytesFn([100_000_000, 150_000_000])
        wd = Watchdog(
            log_flood_bytes_per_sec=10 * 1024 * 1024,
            logs_dir=Path("/fake/logs"),
            logs_total_bytes_fn=bytes_fn,
            health_probe=_StaticProbe({}),
        )
        # First tick just seeds baseline
        d1 = wd.evaluate([], now_unix_s=0.0)
        self.assertFalse(d1.log_flood_detected)
        # Second tick: delta/dt exceeds threshold
        d2 = wd.evaluate([], now_unix_s=1.0)
        self.assertTrue(d2.log_flood_detected)
        self.assertGreater(d2.log_flood_bytes_per_sec, 10 * 1024 * 1024)

    def test_log_flood_not_triggered_on_normal_growth(self) -> None:
        # 100MB → 100.1MB in 1 second = 100 KB/s — well under trip
        bytes_fn = _StaticBytesFn([100_000_000, 100_100_000])
        wd = Watchdog(
            log_flood_bytes_per_sec=10 * 1024 * 1024,
            logs_dir=Path("/fake/logs"),
            logs_total_bytes_fn=bytes_fn,
            health_probe=_StaticProbe({}),
        )
        wd.evaluate([], now_unix_s=0.0)
        d = wd.evaluate([], now_unix_s=1.0)
        self.assertFalse(d.log_flood_detected)

    def test_log_flood_disabled_when_no_logs_dir(self) -> None:
        wd = Watchdog(logs_dir=None, health_probe=_StaticProbe({}))
        d = wd.evaluate([], now_unix_s=1.0)
        self.assertFalse(d.log_flood_detected)

    def test_state_stall_detects_silent_hang(self) -> None:
        """Simulates a PunchOff/CORPSE_SLUGS-style silent hang: /health stays
        OK the whole time but /state's state_version never advances beyond 28.
        After the configured stall window elapses, the instance must be
        flagged for kill+restart.
        """
        probe = _StaticProbe({"http://127.0.0.1:9000": [True, True, True, True]})
        state_probe = _StaticStateVersionProbe(
            {"http://127.0.0.1:9000": [28, 28, 28, 28]},
        )
        wd = Watchdog(
            state_stall_threshold_s=30.0,
            health_probe=probe,
            state_version_probe=state_probe,
        )
        # t=0: first observation → seeds baseline, still HEALTHY
        d1 = wd.evaluate([_view(0)], now_unix_s=0.0)
        self.assertEqual(d1.per_instance[0].action, InstanceAction.HEALTHY)
        # t=15: stall is 15s, under 30s threshold → still HEALTHY
        d2 = wd.evaluate([_view(0)], now_unix_s=15.0)
        self.assertEqual(d2.per_instance[0].action, InstanceAction.HEALTHY)
        self.assertIn("state_stall_COMBAT_15s", d2.per_instance[0].reason)
        # t=35: stall is 35s, over threshold → KILL_AND_RESTART
        d3 = wd.evaluate([_view(0)], now_unix_s=35.0)
        self.assertEqual(d3.per_instance[0].action, InstanceAction.KILL_AND_RESTART)
        self.assertIn("silent_hang_state_version_frozen_at_28", d3.per_instance[0].reason)

    def test_state_stall_reset_on_state_advance(self) -> None:
        """If state_version advances between ticks, the stall timer resets."""
        probe = _StaticProbe({"http://127.0.0.1:9000": [True, True, True, True]})
        state_probe = _StaticStateVersionProbe(
            {"http://127.0.0.1:9000": [10, 10, 11, 11]},
        )
        wd = Watchdog(
            state_stall_threshold_s=30.0,
            health_probe=probe,
            state_version_probe=state_probe,
        )
        wd.evaluate([_view(0)], now_unix_s=0.0)    # seed at 10
        wd.evaluate([_view(0)], now_unix_s=20.0)   # still 10 → 20s stall
        d3 = wd.evaluate([_view(0)], now_unix_s=25.0)  # advances to 11 → reset
        self.assertEqual(d3.per_instance[0].action, InstanceAction.HEALTHY)
        # Next tick, still 11, but stall is only 0s → healthy
        d4 = wd.evaluate([_view(0)], now_unix_s=30.0)
        self.assertEqual(d4.per_instance[0].action, InstanceAction.HEALTHY)
        self.assertIn("state_stall_COMBAT_5s", d4.per_instance[0].reason)

    def test_state_stall_disabled_when_threshold_zero(self) -> None:
        """Setting state_stall_threshold_s=0 disables silent-hang detection."""
        probe = _StaticProbe({"http://127.0.0.1:9000": [True, True]})
        state_probe = _StaticStateVersionProbe(
            {"http://127.0.0.1:9000": [5, 5]},
        )
        wd = Watchdog(
            state_stall_threshold_s=0.0,
            health_probe=probe,
            state_version_probe=state_probe,
        )
        wd.evaluate([_view(0)], now_unix_s=0.0)
        d = wd.evaluate([_view(0)], now_unix_s=9999.0)
        self.assertEqual(d.per_instance[0].action, InstanceAction.HEALTHY)
        self.assertEqual(d.per_instance[0].reason, "probe_ok")

    def test_state_stall_does_not_fire_on_idle_screens(self) -> None:
        """Fresh instance sitting on MAIN_MENU / CHARACTER_SELECT / GAME_OVER
        is legitimately not advancing state — watchdog must not kill it.
        This is the bug report: a fresh instance waiting for the trainer's
        reset() would get repeatedly killed under the version-only check.
        """
        probe = _StaticProbe({"http://127.0.0.1:9000": [True] * 6})
        state_probe = _StaticStateVersionProbe({
            "http://127.0.0.1:9000": [
                (3, "MAIN_MENU"),
                (3, "MAIN_MENU"),
                (3, "CHARACTER_SELECT"),
                (3, "CHARACTER_SELECT"),
                (3, "GAME_OVER"),
                (3, "GAME_OVER"),
            ],
        })
        wd = Watchdog(
            state_stall_threshold_s=20.0,
            health_probe=probe,
            state_version_probe=state_probe,
        )
        # Even 300s on idle screens with unchanged state_version → HEALTHY.
        for i, t in enumerate([0.0, 60.0, 120.0, 180.0, 240.0, 300.0]):
            d = wd.evaluate([_view(0)], now_unix_s=t)
            self.assertEqual(
                d.per_instance[0].action,
                InstanceAction.HEALTHY,
                f"tick {i} at t={t} should be HEALTHY on idle screen",
            )
            self.assertTrue(d.per_instance[0].reason.startswith("idle_screen_"))

    def test_state_stall_fires_when_gameplay_truly_frozen(self) -> None:
        """Once we're in an active gameplay screen (COMBAT), stall counts
        normally.
        """
        probe = _StaticProbe({"http://127.0.0.1:9000": [True] * 4})
        state_probe = _StaticStateVersionProbe({
            "http://127.0.0.1:9000": [
                (50, "COMBAT"),
                (50, "COMBAT"),
                (50, "COMBAT"),
                (50, "COMBAT"),
            ],
        })
        wd = Watchdog(
            state_stall_threshold_s=30.0,
            health_probe=probe,
            state_version_probe=state_probe,
        )
        wd.evaluate([_view(0)], now_unix_s=0.0)   # seed COMBAT, v=50
        d2 = wd.evaluate([_view(0)], now_unix_s=20.0)
        self.assertEqual(d2.per_instance[0].action, InstanceAction.HEALTHY)
        # 35s stall on COMBAT screen → KILL
        d3 = wd.evaluate([_view(0)], now_unix_s=35.0)
        self.assertEqual(d3.per_instance[0].action, InstanceAction.KILL_AND_RESTART)
        self.assertIn("screen_COMBAT", d3.per_instance[0].reason)

    def test_state_stall_resets_when_leaving_idle_screen(self) -> None:
        """Transition MAIN_MENU → COMBAT should restart the stall clock so
        the first tick on COMBAT can't trip the threshold from idle-screen
        accumulated time.
        """
        probe = _StaticProbe({"http://127.0.0.1:9000": [True] * 5})
        state_probe = _StaticStateVersionProbe({
            "http://127.0.0.1:9000": [
                (3, "MAIN_MENU"),
                (3, "MAIN_MENU"),
                (10, "COMBAT"),   # transition, fresh seed
                (10, "COMBAT"),   # unchanged but only 10s later
                (10, "COMBAT"),   # now 30s — over threshold
            ],
        })
        wd = Watchdog(
            state_stall_threshold_s=25.0,
            health_probe=probe,
            state_version_probe=state_probe,
        )
        wd.evaluate([_view(0)], now_unix_s=0.0)      # MAIN_MENU
        wd.evaluate([_view(0)], now_unix_s=500.0)    # still MAIN_MENU, huge idle — ok
        wd.evaluate([_view(0)], now_unix_s=505.0)    # enters COMBAT v=10 → seed
        d = wd.evaluate([_view(0)], now_unix_s=515.0)
        self.assertEqual(d.per_instance[0].action, InstanceAction.HEALTHY)
        d2 = wd.evaluate([_view(0)], now_unix_s=535.0)
        self.assertEqual(d2.per_instance[0].action, InstanceAction.KILL_AND_RESTART)

    def test_state_stall_screen_transition_counts_as_progress(self) -> None:
        """If state_version happens to match across a screen change (e.g. a
        snapshot transition tick), the screen delta alone should still count
        as progress.
        """
        probe = _StaticProbe({"http://127.0.0.1:9000": [True] * 3})
        state_probe = _StaticStateVersionProbe({
            "http://127.0.0.1:9000": [
                (42, "MAP"),
                (42, "COMBAT"),   # same version, different screen
                (42, "COMBAT"),
            ],
        })
        wd = Watchdog(
            state_stall_threshold_s=25.0,
            health_probe=probe,
            state_version_probe=state_probe,
        )
        wd.evaluate([_view(0)], now_unix_s=0.0)
        d2 = wd.evaluate([_view(0)], now_unix_s=20.0)  # MAP→COMBAT, reset
        self.assertEqual(d2.per_instance[0].action, InstanceAction.HEALTHY)
        # Next tick at +10s (t=30) is only 10s of COMBAT stall → HEALTHY
        d3 = wd.evaluate([_view(0)], now_unix_s=30.0)
        self.assertEqual(d3.per_instance[0].action, InstanceAction.HEALTHY)

    def test_state_stall_ignores_transient_state_probe_failure(self) -> None:
        """/state transiently unreachable (returns None) shouldn't reset or
        advance the stall timer — we should wait for the next tick with data.
        """
        probe = _StaticProbe({"http://127.0.0.1:9000": [True, True, True, True]})
        state_probe = _StaticStateVersionProbe(
            # 7 → None → 7 → 7 → 7 : None in middle should neither reset nor
            # advance the stall counter; stall should still count from t=0.
            {"http://127.0.0.1:9000": [7, None, 7, 7]},
        )
        wd = Watchdog(
            state_stall_threshold_s=25.0,
            health_probe=probe,
            state_version_probe=state_probe,
        )
        wd.evaluate([_view(0)], now_unix_s=0.0)    # seed at 7
        d2 = wd.evaluate([_view(0)], now_unix_s=10.0)  # state probe None → HEALTHY, keep seed
        self.assertEqual(d2.per_instance[0].action, InstanceAction.HEALTHY)
        self.assertEqual(d2.per_instance[0].reason, "probe_ok")
        # t=20: back to 7, stall is 20s, still under 25s → HEALTHY
        d3 = wd.evaluate([_view(0)], now_unix_s=20.0)
        self.assertEqual(d3.per_instance[0].action, InstanceAction.HEALTHY)
        # t=30: stall 30s over threshold → KILL
        d4 = wd.evaluate([_view(0)], now_unix_s=30.0)
        self.assertEqual(d4.per_instance[0].action, InstanceAction.KILL_AND_RESTART)


if __name__ == "__main__":
    unittest.main()
