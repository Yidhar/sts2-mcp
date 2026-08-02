"""Bounded asynchronous overlap between the actor and V-trace learner."""

from __future__ import annotations

import queue
from collections import Counter, deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from threading import Event, Lock, Thread
from typing import Any
from uuid import uuid4

import torch

from sts2_baseline import RolloutQueueClosed, SequenceUnroll
from sts2_rl.contracts import EnvironmentBackend

from .checkpointing import ActorSupervisorState
from .collector import CollectedEpisode, EpisodeProgress
from .factory import TrainingResources
from .failure_credit import EvidenceRecord


@dataclass(frozen=True, slots=True)
class RecoverableActorIncident:
    """A fail-closed environment incident observed at an actor boundary."""

    incident_id: str
    category: str
    fingerprint: str
    exception_type: str
    message: str
    emitted_environment_steps: int
    validated_environment_steps: int
    lost_valid_prefix_steps: int
    actor_policy_version: int
    episode_id: str | None
    reset_seed: int | None
    maximum_observed_candidates: int
    quarantine_path: str | None
    details: Mapping[str, Any]
    fingerprint_occurrences: int
    consecutive_incidents: int
    incidents_last_100_attempts: int
    circuit_breaker_open: bool


class ActorLearnerPipeline:
    """Run one environment actor concurrently with the main-thread learner.

    The actor streams each completed unroll into the bounded FIFO immediately;
    it does not wait for the episode to end.  At each episode boundary it does
    wait for the main thread to commit metrics and request any checkpoint or
    evaluation pause before resetting the simulator.  Model publication is
    applied only after complete recurrent unrolls or at acknowledged episode
    boundaries, so every unroll is labelled with the exact behavior-policy
    version that produced it.
    """

    def __init__(
        self,
        resources: TrainingResources,
        *,
        total_environment_steps: int,
        starting_environment_steps: int,
        starting_policy_version: int,
        epsilon: Callable[[int], float],
        starting_episode_count: int = 0,
        deterministic_probe_interval_episodes: int = 0,
        deterministic_probe_environment_steps: tuple[int, ...] = (),
        supervisor_state: ActorSupervisorState | None = None,
    ) -> None:
        if total_environment_steps <= 0:
            raise ValueError("total_environment_steps must be positive")
        if not 0 <= starting_environment_steps <= total_environment_steps:
            raise ValueError("starting_environment_steps is outside the run horizon")
        if starting_policy_version < 0:
            raise ValueError("starting_policy_version must be non-negative")
        if (
            isinstance(starting_episode_count, bool)
            or not isinstance(starting_episode_count, int)
            or starting_episode_count < 0
        ):
            raise ValueError("starting_episode_count must be a non-negative integer")
        if (
            isinstance(deterministic_probe_interval_episodes, bool)
            or not isinstance(deterministic_probe_interval_episodes, int)
            or deterministic_probe_interval_episodes < 0
        ):
            raise ValueError("deterministic_probe_interval_episodes must be a non-negative integer")
        if not isinstance(deterministic_probe_environment_steps, tuple):
            deterministic_probe_environment_steps = tuple(deterministic_probe_environment_steps)
        previous_probe_step = 0
        for index, probe_step in enumerate(deterministic_probe_environment_steps):
            if isinstance(probe_step, bool) or not isinstance(probe_step, int) or probe_step <= previous_probe_step:
                raise ValueError(
                    "deterministic_probe_environment_steps must contain "
                    "strictly increasing positive integers; invalid item at "
                    f"index {index}"
                )
            previous_probe_step = probe_step
        if supervisor_state is None:
            supervisor_state = ActorSupervisorState()
        elif not isinstance(supervisor_state, ActorSupervisorState):
            raise TypeError("supervisor_state must be ActorSupervisorState")
        self.resources = resources
        self.total_environment_steps = total_environment_steps
        self._environment_steps = starting_environment_steps
        self._actor_policy_version = starting_policy_version
        self._epsilon = epsilon
        self._completed_training_episodes = starting_episode_count
        self._deterministic_probe_interval_episodes = deterministic_probe_interval_episodes
        self._deterministic_probe_environment_steps = deterministic_probe_environment_steps
        # Exact resume must not replay early probes that belong to the already
        # committed prefix. A milestone exactly at the restored step is part of
        # that prefix; later milestones become due only after collection crosses
        # them. This needs no mutable checkpoint sidecar.
        self._next_deterministic_probe_step = 0
        while (
            self._next_deterministic_probe_step < len(self._deterministic_probe_environment_steps)
            and self._deterministic_probe_environment_steps[self._next_deterministic_probe_step]
            <= starting_environment_steps
        ):
            self._next_deterministic_probe_step += 1
        self._episodes: queue.Queue[CollectedEpisode | RecoverableActorIncident | BaseException] = queue.Queue()
        # Detector-authoritative local incidents are intentionally transported
        # independently of episode results.  A full run may stay alive for
        # thousands of decisions after escaping a short semantic cycle; making
        # the actor wait for that boundary would stale or lose the policy
        # evidence that explains the loop.
        self._failure_evidence: queue.Queue[tuple[EvidenceRecord, ...]] = queue.Queue()
        self._stop = Event()
        self._pause_requested = Event()
        self._paused = Event()
        self._resume = Event()
        self._resume.set()
        self._episode_boundary_waiting = Event()
        self._episode_boundary_release = Event()
        self._incident_boundary_waiting = Event()
        self._incident_boundary_release = Event()
        self._publication_lock = Lock()
        self._progress_lock = Lock()
        self._actor_progress: EpisodeProgress | None = None
        self._validated_episode_steps = 0
        self._validated_episode_maximum_observed_candidates = 0
        self._pending_publication: tuple[int, Mapping[str, Any]] | None = None
        self._episode_attempts = supervisor_state.episode_attempts
        self._consecutive_incidents = supervisor_state.consecutive_incidents
        self._incident_fingerprints: Counter[str] = Counter(dict(supervisor_state.incident_fingerprints))
        self._recent_incident_attempts: deque[int] = deque(supervisor_state.recent_incident_attempts)
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

    @property
    def at_episode_boundary(self) -> bool:
        return self._episode_boundary_waiting.is_set()

    @property
    def at_incident_boundary(self) -> bool:
        return self._incident_boundary_waiting.is_set()

    @property
    def actor_progress(self) -> EpisodeProgress | None:
        with self._progress_lock:
            return self._actor_progress

    @property
    def supervisor_state(self) -> ActorSupervisorState:
        """Return a checkpoint-safe snapshot at an actor quiescence boundary."""

        if self.alive and not (self.paused or self.at_episode_boundary or self.at_incident_boundary):
            raise RuntimeError("actor supervisor state can only be checkpointed while the actor is quiescent")
        return ActorSupervisorState(
            episode_attempts=self._episode_attempts,
            consecutive_incidents=self._consecutive_incidents,
            incident_fingerprints=tuple(sorted(self._incident_fingerprints.items())),
            recent_incident_attempts=tuple(self._recent_incident_attempts),
        )

    def start(self) -> None:
        if self._thread.ident is not None:
            raise RuntimeError("actor pipeline can only be started once")
        self._thread.start()

    def request_policy_publication(self, policy_version: int) -> None:
        """Copy a learner snapshot now; the actor adopts it between unrolls."""

        if policy_version < self._actor_policy_version:
            raise ValueError("cannot publish an older policy version")
        learner_device = next(self.resources.model.parameters()).device
        if learner_device.type == "cuda":
            torch.cuda.synchronize(learner_device)
        state = {key: value.detach().cpu().clone() for key, value in self.resources.model.state_dict().items()}
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
        self._episode_boundary_release.set()
        self._incident_boundary_release.set()
        self.resources.rollout_queue.close()

    def join(self, timeout: float | None = None) -> None:
        self._thread.join(timeout)
        if self._thread.is_alive():
            raise TimeoutError("actor pipeline did not stop before the timeout")

    def next_episode(
        self,
        *,
        timeout: float | None = None,
    ) -> CollectedEpisode | RecoverableActorIncident | None:
        try:
            result = self._episodes.get(timeout=timeout)
        except queue.Empty:
            return None
        if isinstance(result, BaseException):
            raise result
        return result

    def next_failure_evidence(
        self,
        *,
        timeout: float | None = None,
    ) -> tuple[EvidenceRecord, ...] | None:
        """Return one atomically published detector-evidence batch."""

        try:
            return self._failure_evidence.get(timeout=timeout)
        except queue.Empty:
            return None

    @property
    def pending_failure_evidence_batches(self) -> int:
        """Best-effort telemetry; correctness never depends on ``qsize``."""

        return self._failure_evidence.qsize()

    def release_episode_boundary(self) -> None:
        """Acknowledge that the main thread committed one episode result."""

        self._episode_boundary_release.set()

    def release_incident_boundary(self) -> None:
        """Release an actor after the main thread replaced its backend."""

        self._incident_boundary_release.set()

    def replace_backend(self, backend: EnvironmentBackend) -> tuple[str | None, str | None]:
        """Replace a poisoned backend while the actor is quiescent.

        Mutating requests are never replayed.  The old session is closed and a
        brand-new backend/session becomes visible to both resources and the
        collector before the actor is released.
        """

        if not self.at_incident_boundary:
            raise RuntimeError("backend replacement requires an incident boundary")
        old_backend = self.resources.backend
        if backend is old_backend:
            raise ValueError("incident recovery requires a fresh backend instance")
        old_session = getattr(old_backend, "session_id", None)
        new_session = getattr(backend, "session_id", None)
        old_backend.close()
        self.resources.backend = backend
        self.resources.collector.replace_backend(backend)
        return (
            str(old_session) if old_session is not None else None,
            str(new_session) if new_session is not None else None,
        )

    def _put_unroll(self, unroll: SequenceUnroll) -> int:
        if self._stop.is_set():
            raise RolloutQueueClosed("actor pipeline was stopped")
        self.resources.rollout_queue.put(unroll)
        # Count accepted rollout steps immediately instead of waiting for a
        # potentially very long full-run episode to end. Runtime progress and
        # total-step accounting therefore remain truthful during Act 1--3.
        self._environment_steps += unroll.environment_steps
        # Publication is adopted by the actor thread only after an entire
        # recurrent unroll has been emitted.  This permits real actor/learner
        # overlap during a long full-run episode without mutating a model in
        # the middle of a forward pass or mislabelling behavior-policy data.
        self._adopt_publication()
        return self._actor_policy_version

    def _put_failure_evidence(self, records: tuple[EvidenceRecord, ...]) -> None:
        if not records:
            raise ValueError("failure-evidence publication must not be empty")
        self._failure_evidence.put(records)

    def _record_progress(self, progress: EpisodeProgress) -> None:
        with self._progress_lock:
            self._actor_progress = progress

    def _record_accepted_step(
        self,
        episode_steps: int,
        maximum_observed_candidates: int,
    ) -> None:
        if episode_steps <= self._validated_episode_steps:
            raise RuntimeError("collector accepted-step progress must be strictly monotonic")
        if (
            isinstance(maximum_observed_candidates, bool)
            or not isinstance(maximum_observed_candidates, int)
            or maximum_observed_candidates < self._validated_episode_maximum_observed_candidates
        ):
            raise RuntimeError("collector accepted-step candidate maximum must be a monotonic integer")
        self._validated_episode_steps = episode_steps
        self._validated_episode_maximum_observed_candidates = maximum_observed_candidates

    def _wait_if_paused(self) -> None:
        if not self._pause_requested.is_set():
            return
        self._adopt_publication()
        self._resume.clear()
        self._paused.set()
        while self._pause_requested.is_set() and not self._stop.is_set():
            self._resume.wait(0.1)
        self._paused.clear()

    def _wait_for_episode_boundary_ack(self) -> None:
        try:
            while not self._stop.is_set():
                if self._episode_boundary_release.wait(0.1):
                    return
        finally:
            self._episode_boundary_waiting.clear()

    def _wait_for_incident_boundary_ack(self) -> None:
        try:
            while not self._stop.is_set():
                if self._incident_boundary_release.wait(0.1):
                    return
        finally:
            self._incident_boundary_waiting.clear()

    @staticmethod
    def _is_recoverable_infrastructure_error(exc: BaseException) -> bool:
        # Recovery is opt-in at the typed backend boundary.  Collector/model,
        # schema, candidate-capacity and unknown exceptions remain globally
        # fatal even if their messages happen to resemble a protocol failure.
        return getattr(exc, "recoverable", False) is True

    def _incident_from_exception(
        self,
        exc: BaseException,
        *,
        emitted_environment_steps: int,
    ) -> RecoverableActorIncident:
        raw_details = getattr(exc, "incident_details", {})
        if hasattr(raw_details, "to_mapping"):
            raw_details = raw_details.to_mapping()
        details = dict(raw_details) if isinstance(raw_details, Mapping) else {}
        category = str(
            getattr(exc, "incident_category", None)
            or getattr(exc, "incident_kind", None)
            or details.get("failure_code")
            or "headless_infrastructure_protocol"
        )
        fingerprint = str(
            getattr(exc, "incident_fingerprint", None)
            or getattr(exc, "fingerprint", None)
            or details.get("fingerprint")
            or f"{type(exc).__module__}.{type(exc).__qualname__}:{category}"
        )
        for attribute in ("operation", "incident_kind", "poisoned"):
            value = getattr(exc, attribute, None)
            if value is not None and attribute not in details:
                details[attribute] = value
        quarantine = getattr(exc, "quarantine_path", None) or details.get("quarantine_path")
        progress = self.actor_progress
        validated = self._validated_episode_steps
        lost = max(0, validated - emitted_environment_steps)
        self._incident_fingerprints[fingerprint] += 1
        self._consecutive_incidents += 1
        self._recent_incident_attempts.append(self._episode_attempts)
        cutoff = self._episode_attempts - 99
        while self._recent_incident_attempts and self._recent_incident_attempts[0] < cutoff:
            self._recent_incident_attempts.popleft()
        occurrences = self._incident_fingerprints[fingerprint]
        recent = len(self._recent_incident_attempts)
        circuit_open = occurrences >= 2 or self._consecutive_incidents >= 2 or recent >= 3
        return RecoverableActorIncident(
            incident_id=str(uuid4()),
            category=category,
            fingerprint=fingerprint,
            exception_type=f"{type(exc).__module__}.{type(exc).__qualname__}",
            message=str(exc),
            emitted_environment_steps=emitted_environment_steps,
            validated_environment_steps=validated,
            lost_valid_prefix_steps=lost,
            actor_policy_version=self._actor_policy_version,
            episode_id=progress.episode_id if progress is not None else None,
            reset_seed=progress.reset_seed if progress is not None else None,
            maximum_observed_candidates=(
                max(
                    progress.maximum_observed_candidates if progress is not None else 0,
                    self._validated_episode_maximum_observed_candidates,
                )
            ),
            quarantine_path=str(quarantine) if quarantine is not None else None,
            details=details,
            fingerprint_occurrences=occurrences,
            consecutive_incidents=self._consecutive_incidents,
            incidents_last_100_attempts=recent,
            circuit_breaker_open=circuit_open,
        )

    def _run(self) -> None:
        try:
            while not self._stop.is_set() and self._environment_steps < self.total_environment_steps:
                self._episode_attempts += 1
                self._adopt_publication()
                remaining = self.total_environment_steps - self._environment_steps
                emitted_before = self._environment_steps
                self._validated_episode_steps = 0
                self._validated_episode_maximum_observed_candidates = 0
                with self._progress_lock:
                    self._actor_progress = None
                try:
                    episode_probe_due = bool(
                        self._deterministic_probe_interval_episodes > 0
                        and ((self._completed_training_episodes + 1) % self._deterministic_probe_interval_episodes == 0)
                    )
                    # Collection can cross multiple early milestones before an
                    # episode boundary. Coalesce every currently overdue
                    # milestone into this one probe rather than running a burst
                    # of duplicate greedy episodes. Commit the cursor only after
                    # successful collection so a recoverable backend incident
                    # retries the probe on the replacement session.
                    next_probe_step = self._next_deterministic_probe_step
                    while (
                        next_probe_step < len(self._deterministic_probe_environment_steps)
                        and self._deterministic_probe_environment_steps[next_probe_step] <= self._environment_steps
                    ):
                        next_probe_step += 1
                    step_probe_due = next_probe_step > self._next_deterministic_probe_step
                    liveness_probe = episode_probe_due or step_probe_due
                    episode = self.resources.collector.collect_episode(
                        epsilon=self._epsilon(self._environment_steps),
                        deterministic=False,
                        record=True,
                        policy_version=self._actor_policy_version,
                        maximum_steps=remaining,
                        unroll_sink=self._put_unroll,
                        failure_credit_sink=(
                            self._put_failure_evidence if self.resources.failure_credit_replay is not None else None
                        ),
                        progress_sink=self._record_progress,
                        accepted_step_sink=self._record_accepted_step,
                        liveness_probe=liveness_probe,
                    )
                except BaseException as exc:
                    if not self._is_recoverable_infrastructure_error(exc):
                        raise
                    incident = self._incident_from_exception(
                        exc,
                        emitted_environment_steps=(self._environment_steps - emitted_before),
                    )
                    self._adopt_publication()
                    self._incident_boundary_release.clear()
                    self._incident_boundary_waiting.set()
                    self._episodes.put(incident)
                    self._wait_for_incident_boundary_ack()
                    self._wait_if_paused()
                    continue
                self._consecutive_incidents = 0
                if step_probe_due:
                    self._next_deterministic_probe_step = next_probe_step
                self._completed_training_episodes += 1
                self._episode_boundary_release.clear()
                self._episode_boundary_waiting.set()
                self._episodes.put(episode)
                self._wait_for_episode_boundary_ack()
                self._wait_if_paused()
        except RolloutQueueClosed:
            if not self._stop.is_set():
                self._episodes.put(RuntimeError("rollout queue closed during collection"))
        except BaseException as exc:  # propagate the exact actor failure to main
            self._episodes.put(exc)
        finally:
            self.resources.rollout_queue.close()
            self._paused.clear()


__all__ = [
    "ActorLearnerPipeline",
    "ActorSupervisorState",
    "RecoverableActorIncident",
]
