"""Bounded asynchronous overlap between the actor and V-trace learner."""

from __future__ import annotations

import queue
from collections.abc import Callable, Mapping
from threading import Event, Lock, Thread
from typing import Any

import torch

from sts2_baseline import RolloutQueueClosed, SequenceUnroll

from .collector import CollectedEpisode
from .factory import TrainingResources


class ActorLearnerPipeline:
    """Run one environment actor concurrently with the main-thread learner.

    The actor streams each completed unroll into the bounded FIFO immediately;
    it does not wait for the episode to end.  Model publication is applied only
    at episode boundaries, so one unroll is always labelled with the exact
    behavior-policy version that produced it.
    """

    def __init__(
        self,
        resources: TrainingResources,
        *,
        total_environment_steps: int,
        starting_environment_steps: int,
        starting_policy_version: int,
        epsilon: Callable[[int], float],
    ) -> None:
        if total_environment_steps <= 0:
            raise ValueError("total_environment_steps must be positive")
        if not 0 <= starting_environment_steps <= total_environment_steps:
            raise ValueError("starting_environment_steps is outside the run horizon")
        if starting_policy_version < 0:
            raise ValueError("starting_policy_version must be non-negative")
        self.resources = resources
        self.total_environment_steps = total_environment_steps
        self._environment_steps = starting_environment_steps
        self._actor_policy_version = starting_policy_version
        self._epsilon = epsilon
        self._episodes: queue.Queue[CollectedEpisode | BaseException] = queue.Queue()
        self._stop = Event()
        self._pause_requested = Event()
        self._paused = Event()
        self._resume = Event()
        self._resume.set()
        self._publication_lock = Lock()
        self._pending_publication: tuple[int, Mapping[str, Any]] | None = None
        self._thread = Thread(
            target=self._run,
            name="sts2-v2-actor",
            daemon=True,
        )

    @property
    def environment_steps(self) -> int:
        return self._environment_steps

    @property
    def actor_policy_version(self) -> int:
        return self._actor_policy_version

    @property
    def alive(self) -> bool:
        return self._thread.is_alive()

    @property
    def paused(self) -> bool:
        return self._paused.is_set()

    def start(self) -> None:
        if self._thread.ident is not None:
            raise RuntimeError("actor pipeline can only be started once")
        self._thread.start()

    def request_policy_publication(self, policy_version: int) -> None:
        """Copy a learner snapshot now; the actor adopts it between episodes."""

        if policy_version < self._actor_policy_version:
            raise ValueError("cannot publish an older policy version")
        learner_device = next(self.resources.model.parameters()).device
        if learner_device.type == "cuda":
            torch.cuda.synchronize(learner_device)
        state = {
            key: value.detach().cpu().clone()
            for key, value in self.resources.model.state_dict().items()
        }
        with self._publication_lock:
            pending = self._pending_publication
            if pending is None or policy_version >= pending[0]:
                self._pending_publication = (policy_version, state)

    def _adopt_publication(self) -> None:
        with self._publication_lock:
            publication = self._pending_publication
            self._pending_publication = None
        if publication is None:
            return
        version, state = publication
        self.resources.collector_model.load_state_dict(state, strict=True)
        self._actor_policy_version = version

    def request_pause(self) -> None:
        self._pause_requested.set()

    def resume(self) -> None:
        self._pause_requested.clear()
        self._paused.clear()
        self._resume.set()

    def set_paused_policy_version(self, policy_version: int) -> None:
        """Record a direct actor-model synchronization while the actor is idle."""

        if self.alive and not self.paused:
            raise RuntimeError("actor policy can only be synchronized while paused")
        if policy_version < self._actor_policy_version:
            raise ValueError("cannot synchronize an older actor policy")
        with self._publication_lock:
            self._pending_publication = None
        self._actor_policy_version = policy_version

    def stop(self) -> None:
        self._stop.set()
        self._resume.set()
        self.resources.rollout_queue.close()

    def join(self, timeout: float | None = None) -> None:
        self._thread.join(timeout)
        if self._thread.is_alive():
            raise TimeoutError("actor pipeline did not stop before the timeout")

    def next_episode(self, *, timeout: float | None = None) -> CollectedEpisode | None:
        try:
            result = self._episodes.get(timeout=timeout)
        except queue.Empty:
            return None
        if isinstance(result, BaseException):
            raise result
        return result

    def _put_unroll(self, unroll: SequenceUnroll) -> None:
        if self._stop.is_set():
            raise RolloutQueueClosed("actor pipeline was stopped")
        self.resources.rollout_queue.put(unroll)

    def _wait_if_paused(self) -> None:
        if not self._pause_requested.is_set():
            return
        self._adopt_publication()
        self._resume.clear()
        self._paused.set()
        while self._pause_requested.is_set() and not self._stop.is_set():
            self._resume.wait(0.1)
        self._paused.clear()

    def _run(self) -> None:
        try:
            while (
                not self._stop.is_set()
                and self._environment_steps < self.total_environment_steps
            ):
                self._adopt_publication()
                remaining = self.total_environment_steps - self._environment_steps
                episode = self.resources.collector.collect_episode(
                    epsilon=self._epsilon(self._environment_steps),
                    deterministic=False,
                    record=True,
                    policy_version=self._actor_policy_version,
                    maximum_steps=remaining,
                    unroll_sink=self._put_unroll,
                )
                self._environment_steps += episode.metrics.steps
                self._episodes.put(episode)
                self._wait_if_paused()
        except RolloutQueueClosed:
            if not self._stop.is_set():
                self._episodes.put(RuntimeError("rollout queue closed during collection"))
        except BaseException as exc:  # propagate the exact actor failure to main
            self._episodes.put(exc)
        finally:
            self.resources.rollout_queue.close()
            self._paused.clear()


__all__ = ["ActorLearnerPipeline"]
