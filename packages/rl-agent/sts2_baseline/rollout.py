"""Short-lived recurrent rollout contracts for the v2 V-trace baseline.

This is deliberately not replay.  Unrolls enter a bounded FIFO, are consumed
once in arrival order, and never receive priorities or get sampled again.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from threading import Condition
from typing import Final

import numpy as np
import numpy.typing as npt

from sts2_rl.encoding import EncodedDecisionSnapshot, GroundedEncodingConfig

ROLLOUT_STEP_VERSION: Final = "sts2-rollout-step-v2"
SEQUENCE_UNROLL_VERSION: Final = "sts2-sequence-unroll-v2"
ROLLOUT_QUEUE_VERSION: Final = "sts2-rollout-queue-v2"


def _finite(value: float, *, label: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{label} must be finite")
    return normalized


@dataclass(frozen=True, slots=True)
class RolloutStep:
    snapshot: EncodedDecisionSnapshot
    action_index: int
    behavior_log_probability: float
    reward: float
    discount: float
    policy_decision: bool
    version: str = ROLLOUT_STEP_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.snapshot, EncodedDecisionSnapshot):
            raise TypeError("rollout step snapshot has the wrong type")
        if isinstance(self.action_index, bool) or not isinstance(
            self.action_index, int
        ):
            raise TypeError("rollout action_index must be an integer")
        if not 0 <= self.action_index < self.snapshot.candidate_count:
            raise ValueError("rollout action_index is outside the candidate range")
        if not bool(self.snapshot.action_mask[self.action_index]):
            raise ValueError("rollout selected an encoder-disabled candidate")
        behavior_log_probability = _finite(
            self.behavior_log_probability,
            label="behavior_log_probability",
        )
        if behavior_log_probability > 1.0e-7:
            raise ValueError("behavior_log_probability cannot exceed zero")
        _finite(self.reward, label="reward")
        discount = _finite(self.discount, label="discount")
        if not 0.0 <= discount <= 1.0:
            raise ValueError("rollout discount must be in [0, 1]")
        if not isinstance(self.policy_decision, bool):
            raise TypeError("rollout policy_decision must be a boolean")
        if self.version != ROLLOUT_STEP_VERSION:
            raise ValueError(f"unsupported rollout step version: {self.version!r}")


def _owned_recurrent_state(
    value: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    state = np.asarray(value)
    if state.dtype != np.float32 or state.ndim != 1:
        raise ValueError("initial recurrent state must be rank-1 float32")
    if not np.all(np.isfinite(state)):
        raise ValueError("initial recurrent state must be finite")
    result = np.ascontiguousarray(state).copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class SequenceUnroll:
    """One contiguous recurrent sequence and its optional bootstrap state."""

    episode_id: str
    start_step: int
    policy_version: int
    initial_recurrent_state: npt.NDArray[np.float32]
    steps: tuple[RolloutStep, ...]
    bootstrap_snapshot: EncodedDecisionSnapshot | None
    version: str = SEQUENCE_UNROLL_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.episode_id, str) or not self.episode_id.strip():
            raise ValueError("unroll episode_id must be non-empty")
        for name, value in (
            ("start_step", self.start_step),
            ("policy_version", self.policy_version),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"unroll {name} must be an integer")
            if value < 0:
                raise ValueError(f"unroll {name} must be non-negative")
        object.__setattr__(
            self,
            "initial_recurrent_state",
            _owned_recurrent_state(self.initial_recurrent_state),
        )
        if not isinstance(self.steps, tuple) or not self.steps:
            raise ValueError("unroll steps must be a non-empty tuple")
        if not all(isinstance(step, RolloutStep) for step in self.steps):
            raise TypeError("unroll contains a non-RolloutStep item")
        if self.bootstrap_snapshot is not None and not isinstance(
            self.bootstrap_snapshot, EncodedDecisionSnapshot
        ):
            raise TypeError("unroll bootstrap_snapshot has the wrong type")
        requires_bootstrap = self.steps[-1].discount > 0.0
        if requires_bootstrap != (self.bootstrap_snapshot is not None):
            raise ValueError(
                "unroll bootstrap_snapshot must exist exactly when final discount is positive"
            )
        if self.version != SEQUENCE_UNROLL_VERSION:
            raise ValueError(f"unsupported sequence unroll version: {self.version!r}")

    @property
    def environment_steps(self) -> int:
        return len(self.steps)

    def validate(
        self,
        *,
        expected_config: GroundedEncodingConfig,
        expected_fingerprint: str,
        recurrent_hidden_dim: int,
        maximum_length: int,
    ) -> None:
        if self.initial_recurrent_state.shape != (recurrent_hidden_dim,):
            raise ValueError("unroll recurrent state differs from model hidden size")
        if len(self.steps) > maximum_length:
            raise ValueError("unroll exceeds configured maximum length")
        for step in self.steps:
            step.snapshot.validate(
                expected_config=expected_config,
                expected_fingerprint=expected_fingerprint,
            )
        if self.bootstrap_snapshot is not None:
            self.bootstrap_snapshot.validate(
                expected_config=expected_config,
                expected_fingerprint=expected_fingerprint,
            )


class RolloutQueueClosed(RuntimeError):
    pass


class BoundedRolloutQueue:
    """Thread-safe bounded FIFO with explicit snapshot/restore boundaries."""

    def __init__(self, capacity: int) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int):
            raise TypeError("rollout queue capacity must be an integer")
        if capacity <= 0:
            raise ValueError("rollout queue capacity must be positive")
        self.capacity = capacity
        self._items: deque[SequenceUnroll] = deque()
        self._closed = False
        self._condition = Condition()
        self.put_count = 0
        self.get_count = 0
        self.producer_wait_seconds = 0.0
        self.consumer_wait_seconds = 0.0

    def __len__(self) -> int:
        with self._condition:
            return len(self._items)

    @property
    def closed(self) -> bool:
        with self._condition:
            return self._closed

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    def put(self, unroll: SequenceUnroll, *, timeout: float | None = None) -> None:
        if not isinstance(unroll, SequenceUnroll):
            raise TypeError("rollout queue accepts only SequenceUnroll")
        started = time.monotonic()
        with self._condition:
            while len(self._items) >= self.capacity and not self._closed:
                if timeout is None:
                    self._condition.wait()
                else:
                    remaining = float(timeout) - (time.monotonic() - started)
                    if remaining <= 0.0:
                        raise TimeoutError("timed out waiting for rollout queue capacity")
                    self._condition.wait(remaining)
            self.producer_wait_seconds += time.monotonic() - started
            if self._closed:
                raise RolloutQueueClosed("cannot put into a closed rollout queue")
            self._items.append(unroll)
            self.put_count += 1
            self._condition.notify_all()

    def get_batch(
        self,
        maximum: int,
        *,
        minimum: int = 1,
        timeout: float | None = None,
    ) -> tuple[SequenceUnroll, ...]:
        for name, value in (("maximum", maximum), ("minimum", minimum)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"rollout batch {name} must be an integer")
            if value <= 0:
                raise ValueError(f"rollout batch {name} must be positive")
        if minimum > maximum:
            raise ValueError("rollout batch minimum cannot exceed maximum")
        started = time.monotonic()
        with self._condition:
            while len(self._items) < minimum and not self._closed:
                if timeout is None:
                    self._condition.wait()
                else:
                    remaining = float(timeout) - (time.monotonic() - started)
                    if remaining <= 0.0:
                        raise TimeoutError("timed out waiting for rollout data")
                    self._condition.wait(remaining)
            self.consumer_wait_seconds += time.monotonic() - started
            if not self._items and self._closed:
                return ()
            count = min(maximum, len(self._items))
            batch = tuple(self._items.popleft() for _ in range(count))
            self.get_count += count
            self._condition.notify_all()
            return batch

    def snapshot(self) -> tuple[SequenceUnroll, ...]:
        """Return an immutable in-order copy for an episode-boundary checkpoint."""

        with self._condition:
            return tuple(self._items)

    def validate_restore_ready(self) -> None:
        """Fail before a checkpoint restore can mutate any other live resource.

        Exact-resume restoration replaces queue contents; it is never a merge.
        Keep the empty/open precondition behind the queue lock so checkpointing
        code does not need to inspect private synchronization state.
        """

        with self._condition:
            if self._items:
                raise RuntimeError("rollout queue must be empty before restore")
            if self._closed:
                raise RuntimeError("cannot restore a closed rollout queue")

    def restore(self, items: tuple[SequenceUnroll, ...]) -> None:
        if not isinstance(items, tuple) or not all(
            isinstance(item, SequenceUnroll) for item in items
        ):
            raise TypeError("rollout queue restore payload must be an unroll tuple")
        if len(items) > self.capacity:
            raise ValueError("rollout queue restore payload exceeds capacity")
        with self._condition:
            if self._items:
                raise RuntimeError("rollout queue must be empty before restore")
            if self._closed:
                raise RuntimeError("cannot restore a closed rollout queue")
            self._items.extend(items)
            self._condition.notify_all()

    def metrics(self) -> dict[str, int | float | str]:
        with self._condition:
            return {
                "version": ROLLOUT_QUEUE_VERSION,
                "size": len(self._items),
                "capacity": self.capacity,
                "put_count": self.put_count,
                "get_count": self.get_count,
                "producer_wait_seconds": self.producer_wait_seconds,
                "consumer_wait_seconds": self.consumer_wait_seconds,
            }


__all__ = [
    "ROLLOUT_QUEUE_VERSION",
    "ROLLOUT_STEP_VERSION",
    "SEQUENCE_UNROLL_VERSION",
    "BoundedRolloutQueue",
    "RolloutQueueClosed",
    "RolloutStep",
    "SequenceUnroll",
]
