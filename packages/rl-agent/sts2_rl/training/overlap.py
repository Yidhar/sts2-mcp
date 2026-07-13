"""Bounded one-episode collector/learner overlap.

This is deliberately not a free-running producer queue.  Exactly one collection
may be in flight, replay remains main-thread owned, and policy publication occurs
only while the collector is idle.  Those constraints make policy lag explicit
and keep checkpoint/evaluation barriers quiescent.
"""

from __future__ import annotations

import math
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

from .collector import CollectedEpisode
from .factory import TrainingResources


@dataclass(frozen=True, slots=True)
class OverlappedEpisode:
    episode: CollectedEpisode
    epsilon: float
    policy_version: int
    policy_publish_ms: float
    collector_ms: float
    wait_ms: float
    collector_pre_wait_ms: float


def _collect(
    resources: TrainingResources,
    epsilon: float,
) -> tuple[CollectedEpisode, float]:
    started_ns = time.perf_counter_ns()
    episode = resources.collector.collect_episode(
        epsilon=epsilon,
        deterministic=False,
        record=True,
    )
    elapsed_ms = (time.perf_counter_ns() - started_ns) / 1_000_000.0
    return episode, elapsed_ms


class OverlappedCollector:
    """Own a single collector worker and an explicit learner-policy snapshot."""

    def __init__(self, resources: TrainingResources) -> None:
        if resources.collector_model is resources.model:
            raise ValueError("overlap requires an independent collector model")
        self._resources = resources
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="sts2-collector",
        )
        self._future: Future[tuple[CollectedEpisode, float]] | None = None
        self._epsilon: float | None = None
        self._policy_version: int | None = None
        self._policy_publish_ms = 0.0
        self._submitted_ns = 0
        self._closed = False

    @property
    def has_pending(self) -> bool:
        return self._future is not None

    def start(self, *, epsilon: float, policy_version: int) -> None:
        if self._closed:
            raise RuntimeError("overlapped collector is closed")
        if self._future is not None:
            raise RuntimeError("only one collector episode may be in flight")
        if not math.isfinite(float(epsilon)) or not 0.0 <= float(epsilon) <= 1.0:
            raise ValueError("collector epsilon must be finite and in [0, 1]")
        if (
            isinstance(policy_version, bool)
            or not isinstance(policy_version, int)
            or policy_version < 0
        ):
            raise ValueError("collector policy_version must be a non-negative integer")

        # This copy is the publication barrier.  No learner update is active in
        # the calling thread and no previous collection remains in flight.
        self._policy_publish_ms = self._resources.publish_collector_policy()
        self._epsilon = float(epsilon)
        self._policy_version = policy_version
        self._submitted_ns = time.perf_counter_ns()
        self._future = self._executor.submit(_collect, self._resources, float(epsilon))

    def wait(self) -> OverlappedEpisode:
        future = self._future
        if future is None:
            raise RuntimeError("no collector episode is in flight")
        epsilon = self._epsilon
        policy_version = self._policy_version
        if epsilon is None or policy_version is None:  # pragma: no cover - invariant
            raise RuntimeError("collector launch metadata is missing")
        wait_started_ns = time.perf_counter_ns()
        try:
            episode, collector_ms = future.result()
        except BaseException:
            # Ctrl-C interrupts the main thread's wait without cancelling the
            # running worker.  Retain the future so the interrupt path can join
            # and ingest it before checkpointing.  Worker failures are complete
            # futures and can be cleared immediately.
            if future.done():
                self._future = None
                self._epsilon = None
                self._policy_version = None
            raise
        else:
            self._future = None
            self._epsilon = None
            self._policy_version = None
        wait_ms = (time.perf_counter_ns() - wait_started_ns) / 1_000_000.0
        launch_to_result_ms = (
            time.perf_counter_ns() - self._submitted_ns
        ) / 1_000_000.0
        return OverlappedEpisode(
            episode=episode,
            epsilon=epsilon,
            policy_version=policy_version,
            policy_publish_ms=self._policy_publish_ms,
            collector_ms=collector_ms,
            wait_ms=wait_ms,
            collector_pre_wait_ms=max(
                0.0,
                min(collector_ms, launch_to_result_ms - wait_ms),
            ),
        )

    def shutdown(self) -> None:
        """Join the worker; callers must ingest a pending episode beforehand."""

        if self._closed:
            return
        self._closed = True
        self._executor.shutdown(wait=True, cancel_futures=False)


__all__ = ["OverlappedCollector", "OverlappedEpisode"]
