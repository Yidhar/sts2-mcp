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

TRANSACTION_TRACE_VERSION: Final = "sts2-transaction-trace-v3"
TRANSACTION_REPLAY_VERSION: Final = "sts2-transaction-replay-v3"
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
    # ``node_key`` remains the exact reward/Q identity.  Liveness policy
    # credit sometimes needs a deliberately coarser identity: for example an
    # event page can repeat while HP, max HP and training-revival telemetry keep
    # changing.  Keeping the two keys separate prevents policy-cycle matching
    # from accidentally merging reward-distinct Q states.  ``None`` preserves
    # the exact key for ordinary and card-selection transactions.
    policy_node_key: str | None = None
    policy_action_fingerprint: str | None = None

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
        if (self.policy_node_key is None) != (self.policy_action_fingerprint is None):
            raise ValueError("transaction liveness policy node/action identities must be both present or both absent")
        if self.policy_node_key is not None:
            _require_key(self.policy_node_key, label="policy_node_key")
        _require_key(self.action_fingerprint, label="action_fingerprint")
        if self.policy_action_fingerprint is not None:
            _require_key(
                self.policy_action_fingerprint,
                label="policy_action_fingerprint",
            )
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

    @property
    def effective_policy_node_key(self) -> str:
        """Return the factual node identity used only for policy liveness."""

        return self.policy_node_key or self.node_key

    @property
    def effective_policy_action_fingerprint(self) -> str:
        """Return the action identity used only for policy liveness."""

        return self.policy_action_fingerprint or self.action_fingerprint

    def storage_nbytes(self) -> int:
        return (
            self.snapshot.storage_nbytes()
            + len(self.node_key.encode("utf-8"))
            + len(self.next_node_key.encode("utf-8"))
            + (len(self.policy_node_key.encode("utf-8")) if self.policy_node_key is not None else 0)
            + len(self.action_fingerprint.encode("utf-8"))
            + (len(self.policy_action_fingerprint.encode("utf-8")) if self.policy_action_fingerprint is not None else 0)
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

    A deadlocked trace has no successful suffix. Only repeated factual
    node/action pairs are avoided. A one-off ``STAY``/``REVISIT`` may be a
    corrective return that merely occurs earlier in the retained failure tail;
    it is not causal evidence by itself. A delayed no-progress window
    containing only unique transitions is authoritative value evidence but
    cannot name one causal action, so it contributes no policy target. Censored
    traces also contribute no policy target.

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
            # A local transaction completion is proved against the exact
            # transaction graph.  Coarse event-liveness identities may also be
            # carried by these steps so a separate global DEADLOCK trace can
            # recognize a reopen cycle, but they must never turn a locally
            # completed prompt into fabricated coarse PREFER credit.
            policy_node_key = step.node_key
            if policy_node_key in handled_nodes:
                continue
            handled_nodes.add(policy_node_key)
            if step.next_node_key not in reachable_nodes:
                continue
            reachable_nodes.add(policy_node_key)
            if step.effect is not TransactionEffect.STAY:
                preferred_indices.add(trace.burn_in_steps + local_index)
                preferred_actions[policy_node_key] = step.action_fingerprint

    avoided_indices: set[int] = set()
    if trace.outcome is TransactionOutcome.COMPLETED:
        for local_index, step in enumerate(learn_steps):
            preferred = preferred_actions.get(step.node_key)
            if preferred is not None and step.action_fingerprint != preferred:
                avoided_indices.add(trace.burn_in_steps + local_index)
    elif trace.outcome is TransactionOutcome.DEADLOCK:
        pair_counts: dict[tuple[str, str], int] = {}
        for step in learn_steps:
            pair = (
                step.effective_policy_node_key,
                step.effective_policy_action_fingerprint,
            )
            pair_counts[pair] = pair_counts.get(pair, 0) + 1
        for local_index, step in enumerate(learn_steps):
            # A forced transition can be part of a liveness failure, but there
            # is no alternative whose probability the actor could increase.
            # Keep its factual Q/value target and never manufacture policy
            # blame that would make the learner's AVOID loss undefined.
            if np.count_nonzero(step.snapshot.action_mask) <= 1:
                continue
            if (
                pair_counts[
                    (
                        step.effective_policy_node_key,
                        step.effective_policy_action_fingerprint,
                    )
                ]
                > 1
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


def _has_factual_avoid_target(trace: TransactionTrace) -> bool:
    """Return whether a trace owns at least one grounded policy AVOID label.

    Most delayed liveness failures deliberately have no policy label because
    their bounded window contains only unique moving transitions.  They remain
    useful value/Q evidence, but they must not crowd the much rarer exact
    recurrent cycles out of replay.  This predicate depends only on the same
    factual target construction consumed by the learner; it does not inspect
    action names or invent a counterfactual action.
    """

    return any(item.target is TransactionPolicyTarget.AVOID for item in factual_transaction_policy_targets(trace))


def _selection_replay_stratum(trace: TransactionTrace) -> str | None:
    """Classify factual selection structure without card/event heuristics."""

    deltas = {step.selected_count_delta for step in trace.learn_steps}
    has_positive = 1 in deltas
    has_negative = -1 in deltas
    if trace.outcome is TransactionOutcome.DEADLOCK:
        if has_positive and has_negative and _has_factual_avoid_target(trace):
            return "selection_cycle"
        return None
    if trace.outcome is not TransactionOutcome.COMPLETED or not has_positive:
        return None
    return "corrective_completion" if has_negative else "monotonic_completion"


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
        # Derived entirely from immutable traces; checkpoint state need not
        # duplicate it and load reconstructs it fail-closed.
        self._actionable_avoid_trace_ids: set[str] = set()
        self._selection_strata: dict[str, set[str]] = {
            "selection_cycle": set(),
            "monotonic_completion": set(),
            "corrective_completion": set(),
        }
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
        has_factual_avoid = _has_factual_avoid_target(trace)
        selection_stratum = _selection_replay_stratum(trace)
        with self._lock:
            if trace.trace_id in self._trace_ids:
                self._duplicate_count += 1
                return False
            while self._items and (
                len(self._items) >= self.capacity or self._storage_nbytes + size > self.byte_capacity
            ):
                # Exact recurrent cycles and explored wrong branches are the
                # only traces that carry grounded AVOID targets.  Preserve
                # those sparse actionable traces ahead of both routine
                # completions and delayed deadlocks whose unique-moving
                # windows cannot name a causal action.  Within equal strata,
                # eviction remains deterministic FIFO.
                protected_trace_ids = {
                    *self._actionable_avoid_trace_ids,
                    *(trace_id for ids in self._selection_strata.values() for trace_id in ids),
                }
                eviction_index = next(
                    (
                        index
                        for index, item in enumerate(self._items)
                        if (
                            item.outcome is not TransactionOutcome.DEADLOCK and item.trace_id not in protected_trace_ids
                        )
                    ),
                    next(
                        (index for index, item in enumerate(self._items) if item.trace_id not in protected_trace_ids),
                        0,
                    ),
                )
                evicted = self._items[eviction_index]
                del self._items[eviction_index]
                self._trace_ids.remove(evicted.trace_id)
                self._actionable_avoid_trace_ids.discard(evicted.trace_id)
                for trace_ids in self._selection_strata.values():
                    trace_ids.discard(evicted.trace_id)
                self._storage_nbytes -= evicted.storage_nbytes()
                self._eviction_count += 1
            self._items.append(trace)
            self._trace_ids.add(trace.trace_id)
            if has_factual_avoid:
                self._actionable_avoid_trace_ids.add(trace.trace_id)
            if selection_stratum is not None:
                self._selection_strata[selection_stratum].add(trace.trace_id)
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
            selected: list[int] = []

            # Selection liveness has three complementary factual structures:
            # exact +/- cycles, monotonic automatic completion, and completed
            # corrective deselection.  Sample one of every available structure
            # before the broad deadlock/ordinary pools so hundreds of unrelated
            # event traces cannot erase the actor signal needed to escape a
            # recurrent select/deselect attractor.
            def select_stratum(stratum: str) -> None:
                if len(selected) >= count:
                    return
                candidates = np.asarray(
                    [
                        index
                        for index, item in enumerate(items)
                        if item.trace_id in self._selection_strata[stratum] and index not in selected
                    ],
                    dtype=np.int64,
                )
                if candidates.size:
                    selected.append(int(self._rng.choice(candidates, size=1, replace=False)[0]))

            # An exact selection cycle is itself an actionable AVOID trace and
            # gets first refusal.  If no such cycle is available, preserve the
            # older guarantee that a generic factual AVOID cannot be displaced
            # by a routine selection completion when ``maximum`` is tiny.
            select_stratum("selection_cycle")
            actionable_indices = np.asarray(
                [
                    index
                    for index, item in enumerate(items)
                    if (item.trace_id in self._actionable_avoid_trace_ids and index not in set(selected))
                ],
                dtype=np.int64,
            )
            # Preserve the pre-existing guarantee for non-selection AVOID
            # traces as well.  A selected cycle already fulfils it.
            selected_has_actionable = any(
                items[index].trace_id in self._actionable_avoid_trace_ids for index in selected
            )
            if len(selected) < count and not selected_has_actionable and actionable_indices.size:
                selected.append(int(self._rng.choice(actionable_indices, size=1, replace=False)[0]))
            select_stratum("monotonic_completion")
            select_stratum("corrective_completion")

            selected_set = set(selected)
            deadlock_indices = np.asarray(
                [
                    index
                    for index, item in enumerate(items)
                    if (index not in selected_set and item.outcome is TransactionOutcome.DEADLOCK)
                ],
                dtype=np.int64,
            )
            ordinary_indices = np.asarray(
                [
                    index
                    for index, item in enumerate(items)
                    if (index not in selected_set and item.outcome is not TransactionOutcome.DEADLOCK)
                ],
                dtype=np.int64,
            )
            remaining_count = count - len(selected)
            # Reserve half of a normal batch for sparse liveness failures, but
            # always admit at least one when both strata exist. Fill any unused
            # reservation from the other stratum.
            if remaining_count <= 0:
                deadlock_count = 0
                ordinary_count = 0
            elif deadlock_indices.size and ordinary_indices.size:
                deadlock_count = min(
                    int(deadlock_indices.size),
                    max(1, remaining_count // 2),
                )
            else:
                deadlock_count = min(int(deadlock_indices.size), remaining_count)
            ordinary_count = min(
                int(ordinary_indices.size),
                remaining_count - deadlock_count,
            )
            remaining = remaining_count - deadlock_count - ordinary_count
            if remaining:
                deadlock_count += min(
                    remaining,
                    int(deadlock_indices.size) - deadlock_count,
                )
                remaining = remaining_count - deadlock_count - ordinary_count
            if remaining:
                ordinary_count += min(
                    remaining,
                    int(ordinary_indices.size) - ordinary_count,
                )
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
                "actionable_avoid_size": len(self._actionable_avoid_trace_ids),
                "selection_cycle_size": len(self._selection_strata["selection_cycle"]),
                "selection_monotonic_completion_size": len(self._selection_strata["monotonic_completion"]),
                "selection_corrective_completion_size": len(self._selection_strata["corrective_completion"]),
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
        if counters["put_count"] != len(items) + counters["eviction_count"]:
            raise ValueError(
                "transaction replay accounting is inconsistent: " "put_count must equal live items plus evictions"
            )
        probe = np.random.default_rng()
        try:
            probe.bit_generator.state = dict(payload["rng_state"])
        except (TypeError, ValueError) as exc:
            raise ValueError("transaction replay RNG checkpoint is invalid") from exc
        actionable_avoid_trace_ids = {item.trace_id for item in items if _has_factual_avoid_target(item)}
        selection_strata: dict[str, set[str]] = {
            "selection_cycle": set(),
            "monotonic_completion": set(),
            "corrective_completion": set(),
        }
        for item in items:
            stratum = _selection_replay_stratum(item)
            if stratum is not None:
                selection_strata[stratum].add(item.trace_id)
        with self._lock:
            self._items = deque(items)
            self._trace_ids = set(ids)
            self._actionable_avoid_trace_ids = actionable_avoid_trace_ids
            self._selection_strata = selection_strata
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
