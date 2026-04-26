"""Tests for per-worker exception recovery in AsyncReadyCollector.

The goal: if one env worker dies (e.g., bridge restart caught the env
mid-step), spawn a replacement worker for that env_id only, bump the
generation, and let the other workers keep rolling. Crashes should only
propagate to the main training loop after the per-env restart budget is
exhausted.
"""
from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from queue import Empty


RL_AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(RL_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(RL_AGENT_ROOT))


# Same import-sidestep trick as test_bridge_client_rebind: skip
# sts2_env/__init__.py (which drags in torch) by loading the module
# directly from its source file.
if "sts2_env" not in sys.modules:
    pkg = types.ModuleType("sts2_env")
    pkg.__path__ = [str(RL_AGENT_ROOT / "sts2_env")]
    sys.modules["sts2_env"] = pkg
_spec = importlib.util.spec_from_file_location(
    "sts2_env.async_ready_collector",
    str(RL_AGENT_ROOT / "sts2_env" / "async_ready_collector.py"),
)
arc = importlib.util.module_from_spec(_spec)
sys.modules["sts2_env.async_ready_collector"] = arc
_spec.loader.exec_module(arc)
AsyncReadyCollector = arc.AsyncReadyCollector


class _NoopEnv:
    """Placeholder env for tests that don't actually start worker threads."""

    def reset(self):
        return {}, {}

    def step(self, action):
        return {}, 0.0, False, False, {}

    def close(self):
        pass


class WorkerErrorRestartTest(unittest.TestCase):
    def _build(self, *, num_envs: int = 2, max_worker_restarts: int = 3):
        factories = [lambda: _NoopEnv() for _ in range(num_envs)]
        collector = AsyncReadyCollector(
            factories,
            step_watchdog_timeout_s=60.0,
            reset_watchdog_timeout_s=60.0,
            initial_reset_watchdog_timeout_s=60.0,
            restart_cooldown_s=0.0,
            max_worker_restarts=max_worker_restarts,
        )
        # Prevent the real thread spawner — we just want to exercise the
        # restart policy / counters / event emission.
        collector._start_worker = lambda *args, **kwargs: None  # type: ignore[assignment]
        return collector

    def test_restart_succeeds_under_budget(self) -> None:
        collector = self._build(max_worker_restarts=3)
        # Initially generation 0, restart_count 0
        self.assertEqual(collector._worker_generation[0], 0)
        self.assertEqual(collector._worker_restart_count[0], 0)

        ok = collector._attempt_worker_error_restart(0, RuntimeError("boom-1"), stale_generation=0)
        self.assertTrue(ok)
        self.assertEqual(collector._worker_generation[0], 1)
        self.assertEqual(collector._worker_restart_count[0], 1)

        # Restart event got queued with worker_exception reason
        events = collector.pop_restart_events()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["phase"], "worker_exception")
        self.assertEqual(events[0]["env_id"], 0)
        self.assertIn("boom-1", events[0]["error"])

    def test_budget_exhausted_returns_false(self) -> None:
        collector = self._build(max_worker_restarts=2)
        self.assertTrue(collector._attempt_worker_error_restart(1, RuntimeError("a"), stale_generation=0))
        self.assertTrue(collector._attempt_worker_error_restart(1, RuntimeError("b"), stale_generation=1))
        # Third attempt exceeds the cap → False
        self.assertFalse(collector._attempt_worker_error_restart(1, RuntimeError("c"), stale_generation=2))
        # Generation frozen at 2 (only two successful bumps)
        self.assertEqual(collector._worker_generation[1], 2)
        self.assertEqual(collector._worker_restart_count[1], 2)

    def test_stale_generation_is_swallowed(self) -> None:
        """If watchdog already bumped the generation (because timeout path
        restarted this worker first), a late exception from the old thread
        shouldn't double-bump. We return True but don't increment.
        """
        collector = self._build(max_worker_restarts=5)
        # Simulate watchdog having already bumped gen 0 -> 1 via timeout
        collector._worker_generation[0] = 1
        collector._worker_restart_count[0] = 1

        ok = collector._attempt_worker_error_restart(0, RuntimeError("late"), stale_generation=0)
        self.assertTrue(ok)
        # Generation still 1, restart_count unchanged (the old worker's
        # exception was in a retired generation, nothing to do).
        self.assertEqual(collector._worker_generation[0], 1)
        self.assertEqual(collector._worker_restart_count[0], 1)

    def test_per_env_isolation(self) -> None:
        """env_id 0 can burn its entire restart budget without affecting
        env_id 1's counters.
        """
        collector = self._build(num_envs=2, max_worker_restarts=3)
        for i in range(3):
            self.assertTrue(collector._attempt_worker_error_restart(0, RuntimeError(f"e{i}"), stale_generation=i))
        self.assertFalse(collector._attempt_worker_error_restart(0, RuntimeError("e3"), stale_generation=3))
        # env 1 untouched
        self.assertEqual(collector._worker_generation[1], 0)
        self.assertEqual(collector._worker_restart_count[1], 0)
        self.assertTrue(collector._attempt_worker_error_restart(1, RuntimeError("f"), stale_generation=0))
        self.assertEqual(collector._worker_generation[1], 1)


class WorkerPermanentFailureTest(unittest.TestCase):
    """Permanent-failure mode: when a worker's restart budget exhausts, the
    env should be marked dead but the collector should KEEP RUNNING with
    the remaining live envs. Critical for 8h+ training runs where one
    instance crapping out shouldn't blow away the whole job.
    """

    def _build(self, *, num_envs: int = 4, max_worker_restarts: int = 2):
        factories = [lambda: _NoopEnv() for _ in range(num_envs)]
        collector = AsyncReadyCollector(
            factories,
            step_watchdog_timeout_s=60.0,
            reset_watchdog_timeout_s=60.0,
            initial_reset_watchdog_timeout_s=60.0,
            restart_cooldown_s=0.0,
            max_worker_restarts=max_worker_restarts,
        )
        collector._start_worker = lambda *args, **kwargs: None  # type: ignore[assignment]
        return collector

    def test_initial_live_env_ids_is_full(self) -> None:
        collector = self._build(num_envs=4)
        self.assertEqual(collector.live_env_ids, [0, 1, 2, 3])
        self.assertEqual(collector.permanently_failed_env_ids, [])
        for env_id in range(4):
            self.assertFalse(collector.is_env_permanently_failed(env_id))

    def test_mark_failed_removes_from_live(self) -> None:
        collector = self._build(num_envs=4)
        collector._mark_env_permanently_failed(2, RuntimeError("bridge gone"), stale_generation=0)
        self.assertEqual(collector.live_env_ids, [0, 1, 3])
        self.assertEqual(collector.permanently_failed_env_ids, [2])
        self.assertTrue(collector.is_env_permanently_failed(2))
        self.assertFalse(collector.is_env_permanently_failed(0))

    def test_mark_failed_emits_restart_event(self) -> None:
        collector = self._build(num_envs=2)
        collector._worker_restart_count[1] = 5
        collector._mark_env_permanently_failed(1, RuntimeError("dead"), stale_generation=0)
        events = collector.pop_restart_events()
        self.assertEqual(len(events), 1)
        ev = events[0]
        self.assertEqual(ev["env_id"], 1)
        self.assertEqual(ev["phase"], "permanently_failed")
        self.assertTrue(ev["permanently_failed"])
        self.assertEqual(ev["restart_count"], 5)
        self.assertIn("dead", ev["error"])

    def test_dispatch_actions_silently_skips_dead_env(self) -> None:
        """Trainer might keep a stale env_id reference for a tick after
        dispatching the failure event. Action targeting dead env must be
        a no-op, not an exception that crashes the trainer.
        """
        import numpy as np
        collector = self._build(num_envs=3)
        collector._mark_env_permanently_failed(1, RuntimeError("rip"), stale_generation=0)
        # Pretend trainer batches actions for envs [0, 1, 2]; the action
        # for env 1 should silently disappear, the others should reach
        # their queues.
        collector.dispatch_actions([0, 1, 2], np.array([7, 8, 9]))
        self.assertEqual(collector._action_queues[0].get_nowait(), 7)
        self.assertEqual(collector._action_queues[2].get_nowait(), 9)
        # Dead env's queue stays empty
        with self.assertRaises(Empty):
            collector._action_queues[1].get_nowait()

    def test_does_not_propagate_to_error_queue_on_permanent_failure(self) -> None:
        """The whole point: permanent failure must NOT push to _error_queue
        (which would crash the trainer via _raise_worker_error_if_any).
        """
        collector = self._build(num_envs=2, max_worker_restarts=2)
        # Burn the restart budget
        collector._attempt_worker_error_restart(0, RuntimeError("a"), stale_generation=0)
        collector._attempt_worker_error_restart(0, RuntimeError("b"), stale_generation=1)
        self.assertFalse(collector._attempt_worker_error_restart(0, RuntimeError("c"), stale_generation=2))
        # Simulate the worker_loop's else branch firing
        collector._mark_env_permanently_failed(0, RuntimeError("c"), stale_generation=2)
        # error_queue must be empty so the trainer's error check is a no-op
        self.assertTrue(collector._error_queue.empty())

    def test_multiple_dead_envs_keep_correct_live_set(self) -> None:
        collector = self._build(num_envs=4)
        collector._mark_env_permanently_failed(0, RuntimeError("x"), stale_generation=0)
        collector._mark_env_permanently_failed(3, RuntimeError("y"), stale_generation=0)
        self.assertEqual(collector.live_env_ids, [1, 2])
        self.assertEqual(collector.permanently_failed_env_ids, [0, 3])


if __name__ == "__main__":
    unittest.main()
