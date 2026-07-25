"""Outcome-grounded transaction replay for multi-step decision surfaces.

This sidecar is deliberately separate from the one-shot V-trace rollout FIFO.
It stores only *training* traces whose labels came from transitions that were
actually executed.  It neither infers rewards from action names nor fabricates
counterfactual labels for legal actions that were not taken.

The collector-facing contract is intentionally game-agnostic:

``surface_key``
    Stable semantic identity for one kind of transaction surface.

``node_key``
    Stable semantic identity for a concrete state inside that surface.

``action_fingerprint``
    Stable semantic identity for the action that was actually executed.

Callers may derive those identities from semantic projections, but transport
revision numbers, request UUIDs and held-out/evaluation data must not enter the
replay.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, replace
from enum import IntEnum
from threading import Lock
from typing import Any, Final

import numpy as np
import numpy.typing as npt

from sts2_rl.encoding import EncodedDecisionSnapshot

TRANSACTION_TRACE_VERSION: Final = "sts2-transaction-trace-v2"
TRANSACTION_REPLAY_VERSION: Final = "sts2-transaction-replay-v2"
TRANSACTION_EFFECT_COUNT: Final = 4
SELECTION_DELTA_COUNT: Final = 3


class TransactionEffect(IntEnum):
    """Observed one-step relation to the current transaction surface."""

    STAY = 0
    MOVE = 1
    EXIT = 2
    REVISIT = 3


class TransactionOutcome(IntEnum):
    """Authoritative liveness outcome for one observed transaction surface.

    This is deliberately independent of the game/run return. A selection
    transaction can complete even when the run later loses, and a run-level
    return must not proxy whether select/deselect/confirm mechanics progressed.
    """

    CENSORED = 0
    COMPLETED = 1
    DEADLOCK = 2


class TransactionPolicyTarget(IntEnum):
    """Binary factual target applied directly to the legal-candidate policy."""

    AVOID = 0
    PREFER = 1


@dataclass(frozen=True, slots=True)
class FactualTransactionPolicyTarget:
    """One observed policy preference at an exact recurrent trace step."""

    step_index: int
    target: TransactionPolicyTarget

    def __post_init__(self) -> None:
        if isinstance(self.step_index, bool) or not isinstance(self.step_index, int):
            raise TypeError("transaction policy target step_index must be an integer")
        if self.step_index < 0:
            raise ValueError("transaction policy target step_index must be non-negative")
        if not isinstance(self.target, TransactionPolicyTarget):
            raise TypeError("transaction policy target must be TransactionPolicyTarget")


def selection_delta_index(delta: int) -> int:
    """Map an observed ``-1/0/+1`` selected-count delta to a CE class."""

    if isinstance(delta, bool) or not isinstance(delta, int) or delta not in {-1, 0, 1}:
        raise ValueError("selected_count_delta must be exactly -1, 0, or 1")
    return delta + 1


def _require_key(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty semantic key")
    if len(value) > 512:
        raise ValueError(f"{label} exceeds the 512-character replay ABI limit")
    return value


def _finite(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{label} must be a finite number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{label} must be finite")
    return normalized


def _owned_state(value: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    state = np.asarray(value)
    if state.dtype != np.float32 or state.ndim != 1:
        raise ValueError("transaction initial recurrent state must be rank-1 float32")
    if not np.all(np.isfinite(state)):
        raise ValueError("transaction initial recurrent state must be finite")
    result = np.ascontiguousarray(state).copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class TransactionStep:
    """One factual action/outcome pair inside a transaction trace.

    ``transaction_return`` is optional because a bounded collector window can
    end before an authoritative task return is known.  Such censored steps may
    train the immediate effect head, but are masked from transaction-Q and
    pairwise policy ranking.
    """

    snapshot: EncodedDecisionSnapshot
    action_index: int
    node_key: str
    next_node_key: str
    action_fingerprint: str
    effect: TransactionEffect
    selected_count_delta: int
    transaction_return: float | None
    return_steps: int | None

    def __post_init__(self) -> None:
        if not isinstance(self.snapshot, EncodedDecisionSnapshot):
            raise TypeError("transaction step snapshot has the wrong type")
        if isinstance(self.action_index, bool) or not isinstance(self.action_index, int):
            raise TypeError("transaction action_index must be an integer")
        if not 0 <= self.action_index < self.snapshot.candidate_count:
            raise ValueError("transaction action_index is outside the candidate range")
        if not bool(self.snapshot.action_mask[self.action_index]):
            raise ValueError("transaction selected an encoder-disabled candidate")
        _require_key(self.node_key, label="node_key")
        _require_key(self.next_node_key, label="next_node_key")
        _require_key(self.action_fingerprint, label="action_fingerprint")
        if not isinstance(self.effect, TransactionEffect):
            raise TypeError("transaction effect must be TransactionEffect")
        selection_delta_index(self.selected_count_delta)
        if self.transaction_return is None:
            if self.return_steps is not None:
                raise ValueError("censored transaction return cannot declare return_steps")
        else:
            _finite(self.transaction_return, label="transaction_return")
            if isinstance(self.return_steps, bool) or not isinstance(self.return_steps, int) or self.return_steps <= 0:
                raise ValueError("observed transaction return requires positive return_steps")

    @property
    def q_observed(self) -> bool:
        return self.transaction_return is not None

    def storage_nbytes(self) -> int:
        return (
            self.snapshot.storage_nbytes()
            + len(self.node_key.encode("utf-8"))
            + len(self.next_node_key.encode("utf-8"))
            + len(self.action_fingerprint.encode("utf-8"))
            + 64
        )


@dataclass(frozen=True, slots=True)
class TransactionTrace:
    """Contiguous recurrent trace from the training partition only.

    The recurrent state at the beginning is deliberately required to be zero.
    The current learner replays the first ``burn_in_steps`` snapshots and then
    detaches the recomputed hidden state before applying auxiliary losses.  A
    stale behavior-policy hidden state is therefore never trusted by replay.

    ``start_step`` is the absolute episode index of the first context snapshot,
    not necessarily the first transaction-labelled action.  A collector should
    normally retain 16--32 immediately preceding factual decisions, set them as
    burn-in, and start from zero; early-episode transactions may use the shorter
    context available from episode step zero.
    """

    trace_id: str
    episode_id: str
    surface_key: str
    start_step: int
    policy_version: int
    initial_recurrent_state: npt.NDArray[np.float32]
    steps: tuple[TransactionStep, ...]
    burn_in_steps: int = 0
    outcome: TransactionOutcome = TransactionOutcome.CENSORED
    data_partition: str = "training"
    version: str = TRANSACTION_TRACE_VERSION

    def __post_init__(self) -> None:
        for label, value in (
            ("trace_id", self.trace_id),
            ("episode_id", self.episode_id),
            ("surface_key", self.surface_key),
        ):
            _require_key(value, label=label)
        for integer_label, integer_value in (
            ("start_step", self.start_step),
            ("policy_version", self.policy_version),
            ("burn_in_steps", self.burn_in_steps),
        ):
            if isinstance(integer_value, bool) or not isinstance(integer_value, int):
                raise TypeError(f"transaction {integer_label} must be an integer")
            if integer_value < 0:
                raise ValueError(f"transaction {integer_label} must be non-negative")
        if self.data_partition != "training":
            raise ValueError("transaction replay rejects held-out/evaluation data")
        if not isinstance(self.outcome, TransactionOutcome):
            raise TypeError("transaction outcome must be TransactionOutcome")
        if self.version != TRANSACTION_TRACE_VERSION:
            raise ValueError(f"unsupported transaction trace version: {self.version!r}")
        if not isinstance(self.steps, tuple) or not self.steps:
            raise ValueError("transaction trace steps must be a non-empty tuple")
        if not all(isinstance(step, TransactionStep) for step in self.steps):
            raise TypeError("transaction trace contains a non-TransactionStep item")
        if self.burn_in_steps >= len(self.steps):
            raise ValueError("transaction burn-in must leave at least one learn step")
        object.__setattr__(self, "initial_recurrent_state", _owned_state(self.initial_recurrent_state))
        if bool(np.count_nonzero(self.initial_recurrent_state)):
            raise ValueError("transaction replay requires zero initial state and current-network burn-in")

    @property
    def learn_steps(self) -> tuple[TransactionStep, ...]:
        return self.steps[self.burn_in_steps :]

    def storage_nbytes(self) -> int:
        return (
            int(self.initial_recurrent_state.nbytes)
            + sum(step.storage_nbytes() for step in self.steps)
            + len(self.trace_id.encode("utf-8"))
            + len(self.episode_id.encode("utf-8"))
            + len(self.surface_key.encode("utf-8"))
            + 128
        )


def factual_transaction_policy_targets(
    trace: TransactionTrace,
) -> tuple[FactualTransactionPolicyTarget, ...]:
    """Build policy-coupled liveness labels from factual transaction paths.

    A completed trace is traversed backwards from its factual exit. For each
    semantic node, its last factual action whose successor is already known to
    reach the exit is preferred. Other *observed* actions from that same node
    are avoided. This labels an explored wrong branch without penalizing the
    corrective deselect that factually returned to the successful route.

    A deadlocked trace has no successful suffix. Only its factual
    cycle-closing ``STAY``/``REVISIT`` transitions or repeated exact
    node/action pairs are avoided. A delayed no-progress window containing only
    unique moving transitions is authoritative value evidence but cannot name
    one causal action, so it contributes no policy target. Censored traces also
    contribute no policy target.

    The key includes the full semantic transaction node, so a deselect used
    once to correct a choice on a subsequently completed route is preferred;
    deselect is never globally penalized or masked. Forced singleton choices
    are omitted because the policy cannot change their outcome.
    """

    if not isinstance(trace, TransactionTrace):
        raise TypeError("trace must be a TransactionTrace")
    if trace.outcome is TransactionOutcome.CENSORED:
        return ()

    learn_steps = trace.learn_steps
    preferred_indices: set[int] = set()
    preferred_actions: dict[str, str] = {}
    if trace.outcome is TransactionOutcome.COMPLETED:
        exit_nodes = {step.next_node_key for step in learn_steps if step.effect is TransactionEffect.EXIT}
        reachable_nodes = set(exit_nodes)
        handled_nodes: set[str] = set()
        for local_index in range(len(learn_steps) - 1, -1, -1):
            step = learn_steps[local_index]
            if step.node_key in handled_nodes:
                continue
            handled_nodes.add(step.node_key)
            if step.next_node_key not in reachable_nodes:
                continue
            reachable_nodes.add(step.node_key)
            if step.effect is not TransactionEffect.STAY:
                preferred_indices.add(trace.burn_in_steps + local_index)
                preferred_actions[step.node_key] = step.action_fingerprint

    avoided_indices: set[int] = set()
    if trace.outcome is TransactionOutcome.COMPLETED:
        for local_index, step in enumerate(learn_steps):
            preferred = preferred_actions.get(step.node_key)
            if preferred is not None and step.action_fingerprint != preferred:
                avoided_indices.add(trace.burn_in_steps + local_index)
    elif trace.outcome is TransactionOutcome.DEADLOCK:
        pair_counts: dict[tuple[str, str], int] = {}
        for step in learn_steps:
            pair = (step.node_key, step.action_fingerprint)
            pair_counts[pair] = pair_counts.get(pair, 0) + 1
        for local_index, step in enumerate(learn_steps):
            if (
                step.effect in {TransactionEffect.STAY, TransactionEffect.REVISIT}
                or pair_counts[(step.node_key, step.action_fingerprint)] > 1
            ):
                avoided_indices.add(trace.burn_in_steps + local_index)

    labels: list[FactualTransactionPolicyTarget] = []
    for step_index, step in enumerate(
        learn_steps,
        start=trace.burn_in_steps,
    ):
        if int(np.count_nonzero(step.snapshot.action_mask)) <= 1:
            continue
        if step_index in avoided_indices:
            labels.append(
                FactualTransactionPolicyTarget(
                    step_index=step_index,
                    target=TransactionPolicyTarget.AVOID,
                )
            )
        elif step_index in preferred_indices:
            labels.append(
                FactualTransactionPolicyTarget(
                    step_index=step_index,
                    target=TransactionPolicyTarget.PREFER,
                )
            )
    return tuple(labels)


@dataclass(frozen=True, slots=True)
class ObservedTransactionPair:
    """Two factual outcomes from the same semantic node and different actions."""

    better_trace: int
    better_step: int
    worse_trace: int
    worse_step: int


def observed_outcome_pairs(
    traces: tuple[TransactionTrace, ...],
    *,
    minimum_return_gap: float = 0.0,
) -> tuple[ObservedTransactionPair, ...]:
    """Build ranking pairs without inventing unexecuted counterfactuals."""

    gap = _finite(minimum_return_gap, label="minimum_return_gap")
    if gap < 0.0:
        raise ValueError("minimum_return_gap must be non-negative")
    grouped: dict[tuple[str, str], list[tuple[int, int, TransactionStep]]] = {}
    for trace_index, trace in enumerate(traces):
        for step_index, step in enumerate(trace.steps):
            if step_index < trace.burn_in_steps or not step.q_observed:
                continue
            grouped.setdefault((trace.surface_key, step.node_key), []).append((trace_index, step_index, step))
    pairs: list[ObservedTransactionPair] = []
    for observations in grouped.values():
        for left_index, left_step_index, left in observations:
            for right_index, right_step_index, right in observations:
                if (left_index, left_step_index) >= (right_index, right_step_index):
                    continue
                if left.action_fingerprint == right.action_fingerprint:
                    continue
                if left.transaction_return is None or right.transaction_return is None:
                    raise RuntimeError("q_observed contract is inconsistent")
                left_return = left.transaction_return
                right_return = right.transaction_return
                if abs(left_return - right_return) <= gap:
                    continue
                if left_return > right_return:
                    pairs.append(ObservedTransactionPair(left_index, left_step_index, right_index, right_step_index))
                else:
                    pairs.append(ObservedTransactionPair(right_index, right_step_index, left_index, left_step_index))
    return tuple(pairs)


def backfill_factual_monte_carlo_returns(
    trace: TransactionTrace,
    *,
    episode_rewards: tuple[float, ...],
    episode_discounts: tuple[float, ...],
    authoritative_outcome: bool,
) -> TransactionTrace:
    """Backfill factual task returns after an episode outcome is authoritative.

    The caller supplies the exact rewards/discounts already used by the main
    V-trace data plane.  There is no action-name reward and no transaction-exit
    bonus.  Infrastructure aborts, unresolved transport outcomes, and other
    censored endings must pass ``authoritative_outcome=False``; every Q label is
    then cleared while immediate factual effect labels remain available.
    """

    if not isinstance(trace, TransactionTrace):
        raise TypeError("trace must be a TransactionTrace")
    if not isinstance(authoritative_outcome, bool):
        raise TypeError("authoritative_outcome must be a boolean")
    if not isinstance(episode_rewards, tuple) or not isinstance(episode_discounts, tuple):
        raise TypeError("episode rewards and discounts must be tuples")
    if not episode_rewards or len(episode_rewards) != len(episode_discounts):
        raise ValueError("episode rewards/discounts must have equal non-zero length")
    rewards = tuple(_finite(value, label="episode_reward") for value in episode_rewards)
    discounts = tuple(_finite(value, label="episode_discount") for value in episode_discounts)
    if any(not 0.0 <= value <= 1.0 for value in discounts):
        raise ValueError("episode discounts must be in [0, 1]")
    if trace.start_step + len(trace.steps) > len(rewards):
        raise ValueError("transaction trace extends beyond the factual episode")
    if authoritative_outcome and discounts[-1] != 0.0:
        raise ValueError("authoritative episode outcome must end with zero discount")

    if not authoritative_outcome:
        return replace(
            trace,
            steps=tuple(replace(step, transaction_return=None, return_steps=None) for step in trace.steps),
        )

    returns = [0.0] * len(rewards)
    horizons = [0] * len(rewards)
    next_return = 0.0
    next_horizon = 0
    for index in range(len(rewards) - 1, -1, -1):
        next_return = rewards[index] + discounts[index] * next_return
        next_horizon = 1 + (next_horizon if discounts[index] > 0.0 else 0)
        returns[index] = next_return
        horizons[index] = next_horizon
    return replace(
        trace,
        steps=tuple(
            replace(
                step,
                transaction_return=returns[trace.start_step + offset],
                return_steps=horizons[trace.start_step + offset],
            )
            for offset, step in enumerate(trace.steps)
        ),
    )


class BoundedTransactionReplay:
    """Thread-safe, byte-bounded replay sidecar with exact checkpoint state."""

    def __init__(self, *, capacity: int, byte_capacity: int, seed: int) -> None:
        for label, value in (
            ("capacity", capacity),
            ("byte_capacity", byte_capacity),
            ("seed", seed),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"transaction replay {label} must be an integer")
        if capacity <= 0 or byte_capacity <= 0 or seed < 0:
            raise ValueError("transaction replay capacities must be positive and seed non-negative")
        self.capacity = capacity
        self.byte_capacity = byte_capacity
        self._rng = np.random.default_rng(seed)
        self._items: deque[TransactionTrace] = deque()
        self._trace_ids: set[str] = set()
        self._storage_nbytes = 0
        self._put_count = 0
        self._sample_count = 0
        self._eviction_count = 0
        self._duplicate_count = 0
        self._lock = Lock()

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    def put(self, trace: TransactionTrace) -> bool:
        if not isinstance(trace, TransactionTrace):
            raise TypeError("transaction replay accepts only TransactionTrace")
        size = trace.storage_nbytes()
        if size > self.byte_capacity:
            raise ValueError("one transaction trace exceeds replay byte capacity")
        with self._lock:
            if trace.trace_id in self._trace_ids:
                self._duplicate_count += 1
                return False
            while self._items and (
                len(self._items) >= self.capacity or self._storage_nbytes + size > self.byte_capacity
            ):
                # Liveness failures are sparse and carry the only guaranteed
                # AVOID targets. Prefer evicting the oldest non-deadlock trace
                # instead of letting routine completed transactions erase
                # them from the replay. If every retained item is a deadlock,
                # normal FIFO bounding still applies.
                eviction_index = next(
                    (
                        index
                        for index, item in enumerate(self._items)
                        if item.outcome is not TransactionOutcome.DEADLOCK
                    ),
                    0,
                )
                evicted = self._items[eviction_index]
                del self._items[eviction_index]
                self._trace_ids.remove(evicted.trace_id)
                self._storage_nbytes -= evicted.storage_nbytes()
                self._eviction_count += 1
            self._items.append(trace)
            self._trace_ids.add(trace.trace_id)
            self._storage_nbytes += size
            self._put_count += 1
            return True

    def sample(self, maximum: int) -> tuple[TransactionTrace, ...]:
        if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum <= 0:
            raise ValueError("transaction replay sample maximum must be a positive integer")
        with self._lock:
            if not self._items:
                return ()
            items = tuple(self._items)
            count = min(maximum, len(items))
            deadlock_indices = np.asarray(
                [index for index, item in enumerate(items) if item.outcome is TransactionOutcome.DEADLOCK],
                dtype=np.int64,
            )
            ordinary_indices = np.asarray(
                [index for index, item in enumerate(items) if item.outcome is not TransactionOutcome.DEADLOCK],
                dtype=np.int64,
            )
            # Reserve half of a normal batch for sparse liveness failures, but
            # always admit at least one when both strata exist. Fill any unused
            # reservation from the other stratum.
            if deadlock_indices.size and ordinary_indices.size:
                deadlock_count = min(
                    int(deadlock_indices.size),
                    max(1, count // 2),
                )
            else:
                deadlock_count = min(int(deadlock_indices.size), count)
            ordinary_count = min(int(ordinary_indices.size), count - deadlock_count)
            remaining = count - deadlock_count - ordinary_count
            if remaining:
                deadlock_count += min(
                    remaining,
                    int(deadlock_indices.size) - deadlock_count,
                )
                remaining = count - deadlock_count - ordinary_count
            if remaining:
                ordinary_count += min(
                    remaining,
                    int(ordinary_indices.size) - ordinary_count,
                )
            selected: list[int] = []
            if deadlock_count:
                selected.extend(
                    int(index)
                    for index in self._rng.choice(
                        deadlock_indices,
                        size=deadlock_count,
                        replace=False,
                    )
                )
            if ordinary_count:
                selected.extend(
                    int(index)
                    for index in self._rng.choice(
                        ordinary_indices,
                        size=ordinary_count,
                        replace=False,
                    )
                )
            indices = np.asarray(selected, dtype=np.int64)
            self._rng.shuffle(indices)
            self._sample_count += count
            return tuple(items[int(index)] for index in indices)

    def snapshot(self) -> tuple[TransactionTrace, ...]:
        with self._lock:
            return tuple(self._items)

    def metrics(self) -> dict[str, int | str]:
        with self._lock:
            return {
                "version": TRANSACTION_REPLAY_VERSION,
                "size": len(self._items),
                "capacity": self.capacity,
                "storage_nbytes": self._storage_nbytes,
                "byte_capacity": self.byte_capacity,
                "put_count": self._put_count,
                "sample_count": self._sample_count,
                "eviction_count": self._eviction_count,
                "duplicate_count": self._duplicate_count,
                "deadlock_size": sum(item.outcome is TransactionOutcome.DEADLOCK for item in self._items),
            }

    def state_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "version": TRANSACTION_REPLAY_VERSION,
                "capacity": self.capacity,
                "byte_capacity": self.byte_capacity,
                "items": tuple(self._items),
                "rng_state": self._rng.bit_generator.state,
                "put_count": self._put_count,
                "sample_count": self._sample_count,
                "eviction_count": self._eviction_count,
                "duplicate_count": self._duplicate_count,
            }

    def load_state_dict(self, payload: object) -> None:
        if not isinstance(payload, dict):
            raise TypeError("transaction replay checkpoint must be an object")
        expected = {
            "version",
            "capacity",
            "byte_capacity",
            "items",
            "rng_state",
            "put_count",
            "sample_count",
            "eviction_count",
            "duplicate_count",
        }
        if set(payload) != expected or payload.get("version") != TRANSACTION_REPLAY_VERSION:
            raise ValueError("unsupported transaction replay checkpoint schema")
        if payload.get("capacity") != self.capacity or payload.get("byte_capacity") != self.byte_capacity:
            raise ValueError("transaction replay checkpoint capacity differs")
        items = payload.get("items")
        if not isinstance(items, tuple) or not all(isinstance(item, TransactionTrace) for item in items):
            raise TypeError("transaction replay checkpoint items are invalid")
        if len(items) > self.capacity:
            raise ValueError("transaction replay checkpoint exceeds item capacity")
        ids = [item.trace_id for item in items]
        if len(ids) != len(set(ids)):
            raise ValueError("transaction replay checkpoint contains duplicate trace IDs")
        storage_nbytes = sum(item.storage_nbytes() for item in items)
        if storage_nbytes > self.byte_capacity:
            raise ValueError("transaction replay checkpoint exceeds byte capacity")
        counters: dict[str, int] = {}
        for name in ("put_count", "sample_count", "eviction_count", "duplicate_count"):
            value = payload.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"transaction replay {name} must be non-negative integer")
            counters[name] = value
        probe = np.random.default_rng()
        try:
            probe.bit_generator.state = dict(payload["rng_state"])
        except (TypeError, ValueError) as exc:
            raise ValueError("transaction replay RNG checkpoint is invalid") from exc
        with self._lock:
            self._items = deque(items)
            self._trace_ids = set(ids)
            self._storage_nbytes = storage_nbytes
            self._rng.bit_generator.state = dict(payload["rng_state"])
            self._put_count = counters["put_count"]
            self._sample_count = counters["sample_count"]
            self._eviction_count = counters["eviction_count"]
            self._duplicate_count = counters["duplicate_count"]


__all__ = [
    "SELECTION_DELTA_COUNT",
    "TRANSACTION_EFFECT_COUNT",
    "TRANSACTION_REPLAY_VERSION",
    "TRANSACTION_TRACE_VERSION",
    "BoundedTransactionReplay",
    "FactualTransactionPolicyTarget",
    "ObservedTransactionPair",
    "TransactionEffect",
    "TransactionOutcome",
    "TransactionPolicyTarget",
    "TransactionStep",
    "TransactionTrace",
    "backfill_factual_monte_carlo_returns",
    "factual_transaction_policy_targets",
    "observed_outcome_pairs",
    "selection_delta_index",
]
