"""Outcome-grounded, CPU-only replay for complete training episodes.

The complete episode is used to *label* old decisions; it is never retained as
one autograd graph.  Collector-owned :class:`EncodedDecisionSnapshot` objects
are immutable NumPy payloads.  Episode completion performs one reverse pass to
attach combat-, Act-, and run-horizon targets, after which the learner may
sample short recurrent sequences.  A sampled sequence explicitly identifies
the prefix that must be replayed under ``torch.no_grad()`` before the learning
suffix is evaluated.

This module intentionally has no collector or learner dependency.  In
particular, it cannot retain a CUDA tensor, a behavior-policy hidden state, or
an object with a ``grad_fn``.  Recurrent state must be reconstructed by the
current network from the sampled burn-in prefix.
"""

from __future__ import annotations

import math
from bisect import bisect_left
from collections import Counter, deque
from copy import deepcopy
from dataclasses import dataclass
from enum import IntEnum
from itertools import pairwise
from threading import Lock
from typing import Any, Final

import numpy as np
import numpy.typing as npt

from sts2_rl.encoding import EncodedDecisionSnapshot, GroundedEncodingConfig

# v5 retires the v4 Act-boundary health receipts together with the act-segment
# imitation channel; episodic replay now serves only long-horizon value
# labels.  Exact resume rejects older sidecars: their payloads carry the
# retired ``act_segment_health`` field and fail closed here instead of being
# silently reinterpreted.
EPISODE_TRAJECTORY_VERSION: Final = "sts2-complete-episode-v5"
EPISODIC_REPLAY_VERSION: Final = "sts2-episodic-replay-v5"
COMBAT_DOMAIN_ID: Final = 1
COMBAT_DECISION_SURFACE: Final = "combat"

# Canonical logical payload widths used by ``storage_nbytes``.  As with
# EncodedDecisionSnapshot.storage_nbytes(), Python/dataclass headers are not
# included; NumPy payloads, UTF-8 key bytes, and every fixed-width scalar that
# the replay retains are included exactly once.
_DECISION_SCALAR_BYTES: Final = (
    8  # step_index: int64
    + 8  # action_index: int64
    + 8  # behavior_log_probability: float64
    + 8  # policy_version: int64
    + 8  # act: int64
    + 8  # floor: int64
    + 8  # task_reward: float64
    + 8  # discount: float64
    + 8  # revivals_before: int64
    + 8  # revivals_after: int64
    + 8  # hp_loss_before: float64
    + 8  # hp_loss_after: float64
    + 1  # combat_boundary: uint8
    + 1  # act_boundary: uint8
    + 1  # policy_decision: bool
)
_OBSERVED_HORIZON_BYTES: Final = (
    1  # observed bitmap
    + 1  # success bitmap
    + 8  # future_revivals: int64
    + 8  # future_hp_loss: float64
    + 8  # task_return: float64
    + 8  # return_steps: int64
)
_UNOBSERVED_HORIZON_BYTES: Final = 1
_COMPLETION_SCALAR_BYTES: Final = (
    1  # authoritative flag
    + 1  # won flag / sentinel
    + 8  # final_revivals
    + 8  # final_hp_loss
)


class BoundaryOutcome(IntEnum):
    """Authoritative outcome attached to the action that closes a horizon."""

    NONE = 0
    SUCCEEDED = 1
    FAILED = 2
    CENSORED = 3


def _key(value: object, *, label: str, optional: bool = False) -> str | None:
    if optional and value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    if len(value.encode("utf-8")) > 1_024:
        raise ValueError(f"{label} exceeds the 1024-byte replay ABI limit")
    return value


def _integer(value: object, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    if value < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    return value


def _finite(value: object, *, label: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{label} must be a finite number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{label} must be finite")
    if minimum is not None and normalized < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    return normalized


def _snapshot_arrays(snapshot: EncodedDecisionSnapshot) -> tuple[npt.NDArray[Any], ...]:
    if not isinstance(snapshot, EncodedDecisionSnapshot):
        raise TypeError("episode decision snapshot has the wrong type")
    return (
        snapshot.world.feature_indptr,
        snapshot.world.feature_indices,
        snapshot.world.feature_values,
        snapshot.world.ids,
        snapshot.candidates.feature_indptr,
        snapshot.candidates.feature_indices,
        snapshot.candidates.feature_values,
        snapshot.candidates.ids,
        snapshot.locals.feature_indptr,
        snapshot.locals.feature_indices,
        snapshot.locals.feature_values,
        snapshot.locals.ids,
        snapshot.local_offsets,
        snapshot.action_mask,
    )


def _validate_snapshot_is_cpu_detached(snapshot: EncodedDecisionSnapshot) -> None:
    """Reject anything other than the canonical immutable NumPy snapshot ABI."""

    for array in _snapshot_arrays(snapshot):
        if not isinstance(array, np.ndarray):
            raise TypeError("episodic replay accepts only NumPy snapshot arrays")
        if array.flags.writeable:
            raise ValueError("episodic replay snapshot arrays must be immutable")
        # Defensive duck-typing check: a future alternate payload must not
        # silently introduce an autograd or device-backed object.
        if getattr(array, "grad_fn", None) is not None or bool(getattr(array, "is_cuda", False)):
            raise ValueError("episodic replay cannot retain CUDA/autograd payloads")


def _canonicalize_legacy_snapshot_arrays(episode: CompletedEpisode) -> CompletedEpisode:
    """Copy/freeze arrays whose read-only bit was lost by pickle protocol <= 4.

    The checkpoint writer uses protocol 5, which preserves NumPy writeability,
    but accepting older detached state dictionaries is inexpensive and avoids
    making a serialization implementation detail part of the public replay
    API.  Only payloads with writable arrays are copied; canonical protocol-5
    episodes retain their existing zero-copy path.  Full structural validation
    still runs afterwards, so this does not repair malformed shapes or dtypes.
    """

    if not isinstance(episode, CompletedEpisode) or not isinstance(episode.steps, tuple):
        return episode
    snapshots: list[EncodedDecisionSnapshot] = []
    for step in episode.steps:
        if not isinstance(step, BackfilledEpisodeStep) or not isinstance(
            step.decision, EpisodeDecisionStep
        ):
            return episode
        snapshot = step.decision.snapshot
        if not isinstance(snapshot, EncodedDecisionSnapshot):
            return episode
        try:
            arrays = _snapshot_arrays(snapshot)
        except AttributeError:
            return episode
        if not all(isinstance(array, np.ndarray) for array in arrays):
            return episode
        snapshots.append(snapshot)
    if not any(array.flags.writeable for snapshot in snapshots for array in _snapshot_arrays(snapshot)):
        return episode

    canonical = deepcopy(episode)
    for step in canonical.steps:
        for array in _snapshot_arrays(step.decision.snapshot):
            array.setflags(write=False)
    return canonical


@dataclass(frozen=True, slots=True)
class EpisodeDecisionStep:
    """One factual transition retained until the full episode is labelled.

    Cumulative resource counters are recorded both before and after the
    selected action.  This is required because one environment transition can
    contain more than one revival; a boolean ``did_revive`` would undercount
    the authoritative cost.

    ``combat_boundary`` and ``act_boundary`` describe the horizon that ends
    *after this action*.  ``CENSORED`` closes a segment without inventing an
    outcome target.  ``NONE`` means the segment remains open.
    """

    snapshot: EncodedDecisionSnapshot
    step_index: int
    action_index: int
    behavior_log_probability: float
    policy_decision: bool
    policy_version: int
    act: int
    combat_id: str | None
    task_reward: float
    discount: float
    revivals_before: int
    revivals_after: int
    hp_loss_before: float
    hp_loss_after: float
    combat_boundary: BoundaryOutcome = BoundaryOutcome.NONE
    act_boundary: BoundaryOutcome = BoundaryOutcome.NONE
    decision_surface: str = "other"
    floor: int = 0

    def __post_init__(self) -> None:
        _validate_snapshot_is_cpu_detached(self.snapshot)
        _integer(self.step_index, label="episode step_index")
        action_index = _integer(self.action_index, label="episode action_index")
        if action_index >= self.snapshot.candidate_count:
            raise ValueError("episode action_index is outside the candidate range")
        if not bool(self.snapshot.action_mask[action_index]):
            raise ValueError("episode action_index selects an encoder-disabled candidate")
        _finite(self.behavior_log_probability, label="behavior_log_probability")
        if not isinstance(self.policy_decision, bool):
            raise TypeError("episode policy_decision must be a boolean")
        _integer(self.policy_version, label="episode policy_version")
        _integer(self.act, label="episode act")
        _integer(self.floor, label="episode floor")
        _key(self.combat_id, label="episode combat_id", optional=True)
        _finite(self.task_reward, label="episode task_reward")
        discount = _finite(self.discount, label="episode discount")
        if not 0.0 <= discount <= 1.0:
            raise ValueError("episode discount must be in [0, 1]")
        before_revivals = _integer(self.revivals_before, label="revivals_before")
        after_revivals = _integer(self.revivals_after, label="revivals_after")
        if after_revivals < before_revivals:
            raise ValueError("episode revival counter must be non-decreasing within a step")
        before_hp = _finite(self.hp_loss_before, label="hp_loss_before", minimum=0.0)
        after_hp = _finite(self.hp_loss_after, label="hp_loss_after", minimum=0.0)
        if after_hp < before_hp:
            raise ValueError("episode HP-loss counter must be non-decreasing within a step")
        if not isinstance(self.combat_boundary, BoundaryOutcome):
            raise TypeError("combat_boundary must be BoundaryOutcome")
        if not isinstance(self.act_boundary, BoundaryOutcome):
            raise TypeError("act_boundary must be BoundaryOutcome")
        if self.combat_boundary is not BoundaryOutcome.NONE and self.combat_id is None:
            raise ValueError("a combat boundary requires a combat_id")
        _key(self.decision_surface, label="episode decision_surface")
        if self.snapshot.domain_id == COMBAT_DOMAIN_ID:
            if self.decision_surface != COMBAT_DECISION_SURFACE:
                raise ValueError(
                    "combat episode decisions must use the canonical combat surface"
                )
        elif self.decision_surface == COMBAT_DECISION_SURFACE:
            raise ValueError(
                "non-combat episode decisions cannot claim the combat surface"
            )

    @property
    def exact_revivals_delta(self) -> int:
        return self.revivals_after - self.revivals_before

    @property
    def exact_hp_loss_delta(self) -> float:
        return self.hp_loss_after - self.hp_loss_before

    def storage_nbytes(self) -> int:
        """Exact canonical payload bytes retained by this decision."""

        combat_key_bytes = 0 if self.combat_id is None else len(self.combat_id.encode("utf-8"))
        return (
            self.snapshot.storage_nbytes()
            + _DECISION_SCALAR_BYTES
            + combat_key_bytes
            + len(self.decision_surface.encode("utf-8"))
        )


@dataclass(frozen=True, slots=True)
class EpisodeCompletion:
    """Observed or censored task outcome for one collected episode.

    The authoritative flag means that success or failure is known well enough
    to train; it does not claim that the simulator itself emitted a native
    terminal. The distinct terminal_reason retains provenance such as
    run_defeat versus combat_progress_stall.
    """

    authoritative: bool
    won: bool | None
    final_revivals: int
    final_hp_loss: float
    terminal_reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.authoritative, bool):
            raise TypeError("episode completion authoritative must be a boolean")
        if self.authoritative:
            if not isinstance(self.won, bool):
                raise TypeError("authoritative episode completion requires a boolean won flag")
        elif self.won is not None:
            raise ValueError("censored episode completion cannot claim a win/loss outcome")
        _integer(self.final_revivals, label="final_revivals")
        _finite(self.final_hp_loss, label="final_hp_loss", minimum=0.0)
        _key(self.terminal_reason, label="terminal_reason")

    def storage_nbytes(self) -> int:
        return _COMPLETION_SCALAR_BYTES + len(self.terminal_reason.encode("utf-8"))


@dataclass(frozen=True, slots=True)
class HorizonTargets:
    """Observed result/cost labels from a decision to one terminal boundary."""

    success: bool | None
    future_revivals: int | None
    future_hp_loss: float | None
    task_return: float | None
    return_steps: int | None

    def __post_init__(self) -> None:
        values = (
            self.success,
            self.future_revivals,
            self.future_hp_loss,
            self.task_return,
            self.return_steps,
        )
        if all(value is None for value in values):
            return
        if any(value is None for value in values):
            raise ValueError("an observed horizon target must populate every field")
        if not isinstance(self.success, bool):
            raise TypeError("horizon success must be a boolean")
        _integer(self.future_revivals, label="future_revivals")
        _finite(self.future_hp_loss, label="future_hp_loss", minimum=0.0)
        _finite(self.task_return, label="horizon task_return")
        _integer(self.return_steps, label="horizon return_steps", minimum=1)

    @classmethod
    def unobserved(cls) -> HorizonTargets:
        return cls(
            success=None,
            future_revivals=None,
            future_hp_loss=None,
            task_return=None,
            return_steps=None,
        )

    @property
    def observed(self) -> bool:
        return self.success is not None

    @property
    def efficiency_eligible(self) -> bool:
        """Costs may rank successful trajectories, never reward early failure."""

        return self.success is True

    def storage_nbytes(self) -> int:
        return _OBSERVED_HORIZON_BYTES if self.observed else _UNOBSERVED_HORIZON_BYTES


@dataclass(frozen=True, slots=True)
class BackfilledEpisodeStep:
    """One decision plus exact combat/Act/run targets known at completion."""

    decision: EpisodeDecisionStep
    combat: HorizonTargets
    act: HorizonTargets
    run: HorizonTargets

    def __post_init__(self) -> None:
        if not isinstance(self.decision, EpisodeDecisionStep):
            raise TypeError("backfilled episode decision has the wrong type")
        if not all(isinstance(target, HorizonTargets) for target in (self.combat, self.act, self.run)):
            raise TypeError("backfilled episode targets have the wrong type")

    @property
    def snapshot(self) -> EncodedDecisionSnapshot:
        return self.decision.snapshot

    @property
    def action_index(self) -> int:
        return self.decision.action_index

    @property
    def step_index(self) -> int:
        return self.decision.step_index

    def storage_nbytes(self) -> int:
        return (
            self.decision.storage_nbytes()
            + self.combat.storage_nbytes()
            + self.act.storage_nbytes()
            + self.run.storage_nbytes()
        )


@dataclass(frozen=True, slots=True)
class CompletedEpisode:
    """Immutable complete-episode replay item from the training partition."""

    episode_id: str
    steps: tuple[BackfilledEpisodeStep, ...]
    completion: EpisodeCompletion
    data_partition: str = "training"
    version: str = EPISODE_TRAJECTORY_VERSION

    def __post_init__(self) -> None:
        _key(self.episode_id, label="episode_id")
        if not isinstance(self.steps, tuple) or not self.steps:
            raise ValueError("completed episode steps must be a non-empty tuple")
        if not all(isinstance(step, BackfilledEpisodeStep) for step in self.steps):
            raise TypeError("completed episode contains a non-backfilled step")
        if not isinstance(self.completion, EpisodeCompletion):
            raise TypeError("completed episode completion has the wrong type")
        if not isinstance(self.data_partition, str) or not self.data_partition.strip():
            raise ValueError("completed episode data_partition must be non-empty")
        if self.version != EPISODE_TRAJECTORY_VERSION:
            raise ValueError(f"unsupported completed episode version: {self.version!r}")
        for expected, step in enumerate(self.steps):
            if step.step_index != expected:
                raise ValueError("completed episode step indexes must be contiguous from zero")

    @property
    def won(self) -> bool | None:
        return self.completion.won

    def storage_nbytes(self) -> int:
        return (
            len(self.episode_id.encode("utf-8"))
            + len(self.data_partition.encode("utf-8"))
            + len(self.version.encode("utf-8"))
            + self.completion.storage_nbytes()
            + sum(step.storage_nbytes() for step in self.steps)
        )


@dataclass(slots=True)
class _ReverseHorizon:
    key: int | str
    success: bool
    end_revivals: int
    end_hp_loss: float
    return_after: float = 0.0
    steps_after: int = 0


def _advance_target(
    state: _ReverseHorizon,
    step: EpisodeDecisionStep,
) -> HorizonTargets:
    task_return = step.task_reward + step.discount * state.return_after
    return_steps = 1 + (state.steps_after if step.discount > 0.0 else 0)
    future_revivals = state.end_revivals - step.revivals_before
    future_hp_loss = state.end_hp_loss - step.hp_loss_before
    if future_revivals < 0 or future_hp_loss < 0.0:
        raise ValueError("horizon terminal counters precede a contained decision")
    state.return_after = task_return
    state.steps_after = return_steps
    return HorizonTargets(
        success=state.success,
        future_revivals=future_revivals,
        future_hp_loss=future_hp_loss,
        task_return=task_return,
        return_steps=return_steps,
    )


def _boundary_state(
    *,
    outcome: BoundaryOutcome,
    key: int | str,
    step: EpisodeDecisionStep,
) -> _ReverseHorizon | None:
    if outcome in (BoundaryOutcome.NONE, BoundaryOutcome.CENSORED):
        return None
    return _ReverseHorizon(
        key=key,
        success=outcome is BoundaryOutcome.SUCCEEDED,
        end_revivals=step.revivals_after,
        end_hp_loss=step.hp_loss_after,
    )


def _validate_episode_steps(
    steps: tuple[EpisodeDecisionStep, ...],
    completion: EpisodeCompletion,
) -> None:
    if not isinstance(steps, tuple) or not steps:
        raise ValueError("episode steps must be a non-empty tuple")
    if not all(isinstance(step, EpisodeDecisionStep) for step in steps):
        raise TypeError("episode contains a non-EpisodeDecisionStep item")
    if not isinstance(completion, EpisodeCompletion):
        raise TypeError("episode completion has the wrong type")

    closed_combats: set[str] = set()
    previous: EpisodeDecisionStep | None = None
    for expected, step in enumerate(steps):
        if step.step_index != expected:
            raise ValueError("episode step indexes must be contiguous from zero")
        if previous is not None:
            if step.revivals_before != previous.revivals_after:
                raise ValueError("episode revival counters are discontinuous")
            if step.hp_loss_before != previous.hp_loss_after:
                raise ValueError("episode HP-loss counters are discontinuous")
            if step.act < previous.act:
                raise ValueError("episode Act indexes must be non-decreasing")
            if step.act != previous.act and previous.act_boundary is not BoundaryOutcome.SUCCEEDED:
                raise ValueError("an Act transition requires a successful prior Act boundary")
            if step.combat_id != previous.combat_id and previous.combat_id is not None:
                if previous.combat_boundary is BoundaryOutcome.NONE:
                    raise ValueError("leaving a combat requires an explicit boundary")
                closed_combats.add(previous.combat_id)
            if (
                previous.combat_boundary is not BoundaryOutcome.NONE
                and step.combat_id == previous.combat_id
            ):
                raise ValueError("a closed combat cannot contain later decisions")
            if step.combat_id is not None and step.combat_id in closed_combats:
                raise ValueError("a closed combat_id cannot reappear later in an episode")
            if previous.act_boundary is not BoundaryOutcome.NONE and step.act == previous.act:
                raise ValueError("a closed Act cannot contain later decisions")
        previous = step

    last = steps[-1]
    if completion.final_revivals != last.revivals_after:
        raise ValueError("completion final_revivals differs from the last factual transition")
    if completion.final_hp_loss != last.hp_loss_after:
        raise ValueError("completion final_hp_loss differs from the last factual transition")


def backfill_completed_episode(
    *,
    episode_id: str,
    steps: tuple[EpisodeDecisionStep, ...],
    completion: EpisodeCompletion,
    data_partition: str = "training",
) -> CompletedEpisode:
    """Backfill exact long-horizon labels in one reverse ``O(T)`` pass.

    Censored run/combat/Act endings deliberately produce no target.  A failed
    observed horizon still trains success/value prediction, but
    ``HorizonTargets.efficiency_eligible`` remains false so a future learner
    cannot prefer an early failure merely because it used fewer revivals.
    """

    _key(episode_id, label="episode_id")
    _validate_episode_steps(steps, completion)
    if not isinstance(data_partition, str) or not data_partition.strip():
        raise ValueError("data_partition must be non-empty")

    unobserved = HorizonTargets.unobserved()
    reversed_results: list[BackfilledEpisodeStep] = []
    combat_state: _ReverseHorizon | None = None
    act_state: _ReverseHorizon | None = None
    run_state = (
        _ReverseHorizon(
            key="run",
            success=bool(completion.won),
            end_revivals=completion.final_revivals,
            end_hp_loss=completion.final_hp_loss,
        )
        if completion.authoritative
        else None
    )

    for step in reversed(steps):
        if step.combat_boundary is not BoundaryOutcome.NONE:
            if step.combat_id is None:  # guarded by EpisodeDecisionStep
                raise RuntimeError("combat boundary contract is inconsistent")
            combat_state = _boundary_state(
                outcome=step.combat_boundary,
                key=step.combat_id,
                step=step,
            )
        elif combat_state is not None and step.combat_id != combat_state.key:
            combat_state = None

        if step.act_boundary is not BoundaryOutcome.NONE:
            act_state = _boundary_state(
                outcome=step.act_boundary,
                key=step.act,
                step=step,
            )
        elif act_state is not None and step.act != act_state.key:
            act_state = None

        combat_target = (
            _advance_target(combat_state, step)
            if combat_state is not None and step.combat_id == combat_state.key
            else unobserved
        )
        act_target = (
            _advance_target(act_state, step)
            if act_state is not None and step.act == act_state.key
            else unobserved
        )
        run_target = _advance_target(run_state, step) if run_state is not None else unobserved
        reversed_results.append(
            BackfilledEpisodeStep(
                decision=step,
                combat=combat_target,
                act=act_target,
                run=run_target,
            )
        )

    return CompletedEpisode(
        episode_id=episode_id,
        steps=tuple(reversed(reversed_results)),
        completion=completion,
        data_partition=data_partition,
    )


def _validate_completed_episode_payload(episode: CompletedEpisode) -> CompletedEpisode:
    """Deeply validate an unpickled replay item and its derived labels.

    Pickle restores dataclass instances without calling ``__post_init__``.
    Exact-resume validation must therefore re-run every nested CPU/snapshot
    invariant and recompute all derived horizon labels from the factual
    decisions.  A hash-consistent but internally malformed sidecar must not be
    able to inject writable arrays or edited returns into the live replay.
    """

    canonical = _canonicalize_legacy_snapshot_arrays(episode)
    canonical.__post_init__()
    canonical.completion.__post_init__()
    for step in canonical.steps:
        # Pickle also bypasses the encoding dataclass validators.  Validate the
        # complete self-described snapshot ABI here; checkpointing performs the
        # additional comparison against the active encoder identity.
        if not isinstance(step.snapshot.config, GroundedEncodingConfig):
            raise TypeError("encoded decision snapshot config has the wrong type")
        step.snapshot.config.__post_init__()
        step.snapshot.validate(
            expected_config=step.snapshot.config,
            expected_fingerprint=step.snapshot.encoding_fingerprint,
        )
        step.decision.__post_init__()
        step.combat.__post_init__()
        step.act.__post_init__()
        step.run.__post_init__()
        step.__post_init__()
    rebuilt = backfill_completed_episode(
        episode_id=canonical.episode_id,
        steps=tuple(step.decision for step in canonical.steps),
        completion=canonical.completion,
        data_partition=canonical.data_partition,
    )
    stored_targets = tuple(
        (step.combat, step.act, step.run) for step in canonical.steps
    )
    rebuilt_targets = tuple(
        (step.combat, step.act, step.run) for step in rebuilt.steps
    )
    if stored_targets != rebuilt_targets:
        raise ValueError("episodic replay checkpoint contains edited horizon labels")
    return canonical


@dataclass(frozen=True, slots=True)
class ReplaySequence:
    """Short learning suffix plus an exact, sparse, mandatory no-grad prefix.

    The active split-memory model ABI updates run memory on every non-combat
    decision and combat memory only inside the current combat.  Its combat
    domain ID is fixed at :data:`COMBAT_DOMAIN_ID` (currently ``1``).  Exact
    reconstruction therefore needs every preceding non-combat decision, every
    decision in the current combat since the last non-combat decision, and at
    least the configured recent burn-in window.  The prefix can be sparse in
    absolute episode indexes; the learning suffix is always short and
    contiguous.  Only the suffix may retain an autograd graph.
    """

    episode_id: str
    start_step: int
    learn_start_step: int
    steps: tuple[BackfilledEpisodeStep, ...]
    burn_in_steps: int
    configured_burn_in_steps: int
    source_episode_won: bool | None
    source_episode_authoritative: bool
    exact_recurrent_reconstruction: bool = True

    def __post_init__(self) -> None:
        _key(self.episode_id, label="replay sequence episode_id")
        _integer(self.start_step, label="replay sequence start_step")
        _integer(self.learn_start_step, label="replay sequence learn_start_step")
        _integer(self.burn_in_steps, label="replay sequence burn_in_steps")
        _integer(
            self.configured_burn_in_steps,
            label="replay sequence configured_burn_in_steps",
        )
        if not isinstance(self.steps, tuple) or not self.steps:
            raise ValueError("replay sequence steps must be a non-empty tuple")
        if not all(isinstance(step, BackfilledEpisodeStep) for step in self.steps):
            raise TypeError("replay sequence contains a non-backfilled step")
        if self.burn_in_steps >= len(self.steps):
            raise ValueError("replay burn-in must leave at least one learning step")
        if self.steps[0].step_index != self.start_step:
            raise ValueError("replay sequence start_step differs from its first decision")
        if self.steps[self.burn_in_steps].step_index != self.learn_start_step:
            raise ValueError("replay learn_start_step differs from its first learning decision")
        if not isinstance(self.exact_recurrent_reconstruction, bool):
            raise TypeError("exact_recurrent_reconstruction must be a boolean")
        if not self.exact_recurrent_reconstruction:
            raise ValueError("episodic replay requires exact recurrent reconstruction")

        prefix = self.steps[: self.burn_in_steps]
        learning = self.steps[self.burn_in_steps :]
        if any(
            right.step_index <= left.step_index
            for left, right in pairwise(prefix)
        ):
            raise ValueError("replay burn-in indexes must be strictly increasing")
        if prefix and prefix[-1].step_index >= self.learn_start_step:
            raise ValueError("replay burn-in must precede the learning suffix")
        if any(
            right.step_index != left.step_index + 1
            for left, right in pairwise(learning)
        ):
            raise ValueError("replay learning suffix must be contiguous")
        if self.source_episode_won is not None and not isinstance(self.source_episode_won, bool):
            raise TypeError("source_episode_won must be boolean or None")
        if not isinstance(self.source_episode_authoritative, bool):
            raise TypeError("source_episode_authoritative must be boolean")

    @property
    def burn_in_no_grad(self) -> bool:
        """The entire possibly long/sparse prefix must run without autograd."""

        return True

    @property
    def burn_in(self) -> tuple[BackfilledEpisodeStep, ...]:
        return self.steps[: self.burn_in_steps]

    @property
    def learn_steps(self) -> tuple[BackfilledEpisodeStep, ...]:
        return self.steps[self.burn_in_steps :]

    def referenced_storage_nbytes(self) -> int:
        """Bytes referenced by this ephemeral view; replay owns no duplicate."""

        return len(self.episode_id.encode("utf-8")) + sum(step.storage_nbytes() for step in self.steps)


@dataclass(frozen=True, slots=True)
class _EpisodeSamplingIndex:
    """Ephemeral O(steps)-to-build metadata used by every later sample.

    The index is derived exclusively from the immutable completed episode and
    therefore does not belong in the exact-resume sidecar ABI.  It is rebuilt
    while loading a sidecar and is evicted atomically with its source episode.
    """

    episode_id: str
    storage_nbytes: int
    macro_policy_by_surface: tuple[tuple[str, tuple[int, ...]], ...]
    noncombat_step_indexes: tuple[int, ...]


def _build_episode_sampling_index(
    episode: CompletedEpisode,
    *,
    storage_nbytes: int,
) -> _EpisodeSamplingIndex:
    """Scan one immutable episode once when it enters replay."""

    macro: dict[str, list[int]] = {}
    noncombat: list[int] = []
    for step in episode.steps:
        decision = step.decision
        if step.snapshot.domain_id != COMBAT_DOMAIN_ID:
            noncombat.append(step.step_index)
        if not decision.policy_decision:
            continue
        if decision.snapshot.domain_id != COMBAT_DOMAIN_ID:
            macro.setdefault(decision.decision_surface, []).append(step.step_index)

    return _EpisodeSamplingIndex(
        episode_id=episode.episode_id,
        storage_nbytes=storage_nbytes,
        macro_policy_by_surface=tuple((surface, tuple(indexes)) for surface, indexes in macro.items()),
        noncombat_step_indexes=tuple(noncombat),
    )


def _exact_recurrent_prefix(
    episode: CompletedEpisode,
    *,
    learn_start: int,
    configured_burn_in_steps: int,
    noncombat_step_indexes: tuple[int, ...] | None = None,
) -> tuple[BackfilledEpisodeStep, ...]:
    """Return the minimal exact prefix for the current split-GRU model ABI."""

    if not 0 <= learn_start < len(episode.steps):
        raise ValueError("learn_start is outside the episode")
    selected: set[int] = set(range(max(0, learn_start - configured_burn_in_steps), learn_start))

    # Run memory is updated only by non-combat decisions, so every historical
    # non-combat decision is required even when it is thousands of steps old.
    if noncombat_step_indexes is None:
        noncombat_indexes = tuple(
            index for index in range(learn_start) if episode.steps[index].snapshot.domain_id != COMBAT_DOMAIN_ID
        )
    else:
        noncombat_indexes = noncombat_step_indexes[: bisect_left(noncombat_step_indexes, learn_start)]
    selected.update(noncombat_indexes)

    # If learning begins inside combat, combat memory must be rebuilt from the
    # reset caused by the most recent non-combat decision.  With no preceding
    # non-combat decision, the current combat started at episode step zero.
    if episode.steps[learn_start].snapshot.domain_id == COMBAT_DOMAIN_ID:
        last_noncombat = noncombat_indexes[-1] if noncombat_indexes else -1
        selected.update(range(last_noncombat + 1, learn_start))

    return tuple(episode.steps[index] for index in sorted(selected))


def _replay_sequence(
    episode: CompletedEpisode,
    *,
    learn_start: int,
    learn_steps: int,
    burn_in_steps: int,
    sampling_index: _EpisodeSamplingIndex | None = None,
) -> ReplaySequence:
    """Build one bounded learning suffix with exact split-memory burn-in."""

    learn_end = min(len(episode.steps), learn_start + learn_steps)
    prefix = _exact_recurrent_prefix(
        episode,
        learn_start=learn_start,
        configured_burn_in_steps=burn_in_steps,
        noncombat_step_indexes=(sampling_index.noncombat_step_indexes if sampling_index is not None else None),
    )
    learning = episode.steps[learn_start:learn_end]
    sequence_steps = prefix + learning
    return ReplaySequence(
        episode_id=episode.episode_id,
        start_step=sequence_steps[0].step_index,
        learn_start_step=learn_start,
        steps=sequence_steps,
        burn_in_steps=len(prefix),
        configured_burn_in_steps=burn_in_steps,
        source_episode_won=episode.won,
        source_episode_authoritative=episode.completion.authoritative,
    )


def _macro_surface_candidates(
    sampling_indexes: tuple[_EpisodeSamplingIndex, ...],
    *,
    episode_order: tuple[int, ...],
    rng: np.random.Generator,
) -> tuple[tuple[int, int], ...]:
    """Round-robin factual non-combat policy decisions by runtime surface.

    One representative decision is first drawn for each episode/surface pair;
    the least-used available surface then contributes at most one returned
    candidate per episode.  That limit
    preserves the replay-wide invariant that no episode receives a second
    segment before other eligible episodes receive their first; the ordinary
    outcome-stratified sampler may fill later quotas after the reserved macro
    pass.  Surface order and in-surface decision indexes are shuffled, but the
    source data remain exact observed training trajectories.
    """

    per_episode_candidates: dict[int, dict[str, int]] = {}
    for episode_index in episode_order:
        per_surface = sampling_indexes[episode_index].macro_policy_by_surface
        if per_surface:
            per_episode_candidates[episode_index] = {
                surface: indexes[int(rng.integers(0, len(indexes)))] for surface, indexes in per_surface
            }

    # Preserve the already outcome-stratified episode order exactly.  Within
    # that order, choose the least-used available surface (randomizing ties)
    # so the reservation does not collapse onto whichever macro surface first
    # appeared in an episode.
    surface_counts: Counter[str] = Counter()
    result: list[tuple[int, int]] = []
    for episode_index in episode_order:
        candidates = per_episode_candidates.get(episode_index)
        if not candidates:
            continue
        minimum_count = min(surface_counts[surface] for surface in candidates)
        least_used = tuple(surface for surface in candidates if surface_counts[surface] == minimum_count)
        surface = least_used[int(rng.integers(0, len(least_used)))]
        result.append((episode_index, candidates[surface]))
        surface_counts[surface] += 1
    return tuple(result)


def _stratified_episode_order(
    episodes: tuple[CompletedEpisode, ...],
    *,
    rng: np.random.Generator,
) -> tuple[int, ...]:
    """Round-robin win/failure/censored strata, shuffled within each stratum."""

    buckets: dict[str, list[int]] = {"win": [], "failure": [], "censored": []}
    for index, episode in enumerate(episodes):
        if not episode.completion.authoritative:
            buckets["censored"].append(index)
        elif episode.won:
            buckets["win"].append(index)
        else:
            buckets["failure"].append(index)

    active = [name for name, indexes in buckets.items() if indexes]
    if not active:
        return ()
    active = [active[int(index)] for index in rng.permutation(len(active))]
    shuffled: dict[str, tuple[int, ...]] = {}
    for name in active:
        indexes = buckets[name]
        shuffled[name] = tuple(indexes[int(index)] for index in rng.permutation(len(indexes)))

    result: list[int] = []
    offset = 0
    while True:
        added = False
        for name in active:
            values = shuffled[name]
            if offset < len(values):
                result.append(values[offset])
                added = True
        if not added:
            return tuple(result)
        offset += 1


class BoundedEpisodicReplay:
    """Thread-safe byte/episode bounded, outcome-stratified replay.

    ``episode_byte_capacity`` prevents one pathological episode from evicting
    the entire replay.  ``max_segments_per_episode`` is enforced independently
    on every sample call.  Win, failure, and censored strata are round-robin
    interleaved before another episode from a populous stratum is visited, and
    every episode is visited before its second sampled segment.  Neither a very
    long revival episode nor many successful runs can monopolize a batch.
    """

    def __init__(
        self,
        *,
        capacity: int,
        byte_capacity: int,
        episode_byte_capacity: int,
        max_segments_per_episode: int,
        seed: int,
    ) -> None:
        for label, value in (
            ("capacity", capacity),
            ("byte_capacity", byte_capacity),
            ("episode_byte_capacity", episode_byte_capacity),
            ("max_segments_per_episode", max_segments_per_episode),
            ("seed", seed),
        ):
            _integer(value, label=f"episodic replay {label}")
        if capacity <= 0 or byte_capacity <= 0 or episode_byte_capacity <= 0:
            raise ValueError("episodic replay capacities must be positive")
        if episode_byte_capacity > byte_capacity:
            raise ValueError("episode byte capacity cannot exceed total replay byte capacity")
        if max_segments_per_episode <= 0:
            raise ValueError("max_segments_per_episode must be positive")
        self.capacity = capacity
        self.byte_capacity = byte_capacity
        self.episode_byte_capacity = episode_byte_capacity
        self.max_segments_per_episode = max_segments_per_episode
        self._rng = np.random.default_rng(seed)
        self._items: deque[CompletedEpisode] = deque()
        self._sampling_indexes: deque[_EpisodeSamplingIndex] = deque()
        self._episode_ids: set[str] = set()
        self._storage_nbytes = 0
        self._put_count = 0
        self._sample_count = 0
        self._macro_sample_count = 0
        self._eviction_count = 0
        self._duplicate_count = 0
        self._oversize_count = 0
        self._maximum_observed_episode_steps = 0
        # Sampling owns the RNG and exact sample counters.  Keeping its
        # serialization lock separate lets put() mutate the bounded item deque
        # while an immutable replay snapshot is scanned/materialized.
        self._sample_lock = Lock()
        self._lock = Lock()

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    def put(self, episode: CompletedEpisode) -> bool:
        if not isinstance(episode, CompletedEpisode):
            raise TypeError("episodic replay accepts only CompletedEpisode")
        if episode.data_partition != "training":
            raise ValueError("episodic replay rejects held-out/evaluation data")
        size = episode.storage_nbytes()
        with self._lock:
            self._maximum_observed_episode_steps = max(
                self._maximum_observed_episode_steps,
                len(episode.steps),
            )
            if size > self.episode_byte_capacity:
                self._oversize_count += 1
                return False
            if episode.episode_id in self._episode_ids:
                self._duplicate_count += 1
                return False
        sampling_index = _build_episode_sampling_index(
            episode,
            storage_nbytes=size,
        )
        with self._lock:
            # A concurrent put of the same immutable episode may have completed
            # while its derived index was built outside the critical section.
            if episode.episode_id in self._episode_ids:
                self._duplicate_count += 1
                return False
            while self._items and (
                len(self._items) >= self.capacity or self._storage_nbytes + size > self.byte_capacity
            ):
                evicted = self._items.popleft()
                evicted_index = self._sampling_indexes.popleft()
                if evicted.episode_id != evicted_index.episode_id:  # pragma: no cover
                    raise RuntimeError("episodic replay sampling index is misaligned")
                self._episode_ids.remove(evicted.episode_id)
                self._storage_nbytes -= evicted_index.storage_nbytes
                self._eviction_count += 1
            self._items.append(episode)
            self._sampling_indexes.append(sampling_index)
            self._episode_ids.add(episode.episode_id)
            self._storage_nbytes += size
            self._put_count += 1
            return True

    def sample(
        self,
        maximum: int,
        *,
        learn_steps: int,
        burn_in_steps: int,
        macro_sample_fraction: float = 0.0,
    ) -> tuple[ReplaySequence, ...]:
        """Sample the all-age value/outcome-stratified replay view.

        ``macro_sample_fraction`` reserves sequence slots whose differentiable
        suffix starts at a factual non-combat decision, so sparse macro
        decisions still receive long-horizon value labels.  No behavior-policy
        freshness filter exists on this plane: since v20 episodic replay
        supervises only value heads and every stored trajectory remains a
        factual target regardless of its behavior-policy age.
        """

        _integer(maximum, label="episodic replay sample maximum", minimum=1)
        _integer(learn_steps, label="episodic replay learn_steps", minimum=1)
        _integer(burn_in_steps, label="episodic replay burn_in_steps")
        macro_fraction = _finite(
            macro_sample_fraction,
            label="episodic replay macro_sample_fraction",
            minimum=0.0,
        )
        if macro_fraction > 1.0:
            raise ValueError("episodic replay macro_sample_fraction must be in [0, 1]")
        with self._sample_lock:
            with self._lock:
                if not self._items:
                    return ()
                episodes = tuple(self._items)
                sampling_indexes = tuple(self._sampling_indexes)
                if len(episodes) != len(sampling_indexes):  # pragma: no cover
                    raise RuntimeError("episodic replay sampling index is misaligned")
            order = _stratified_episode_order(episodes, rng=self._rng)
            choices: dict[int, tuple[int, ...]] = {}
            for episode_index in order:
                episode_length = len(episodes[episode_index].steps)
                window_count = (episode_length + learn_steps - 1) // learn_steps
                count = min(self.max_segments_per_episode, window_count)
                selected = self._rng.choice(window_count, size=count, replace=False)
                choices[episode_index] = tuple(int(value) for value in selected)

            sequences: list[ReplaySequence] = []
            selected_starts: set[tuple[int, int]] = set()
            per_episode_count = {index: 0 for index in order}
            macro_limit = min(
                maximum,
                int(math.ceil(maximum * macro_fraction)),
            )
            macro_added = 0

            def result() -> tuple[ReplaySequence, ...]:
                self._sample_count += len(sequences)
                self._macro_sample_count += macro_added
                return tuple(sequences)

            if macro_limit:
                for episode_index, decision_index in _macro_surface_candidates(
                    sampling_indexes,
                    episode_order=order,
                    rng=self._rng,
                ):
                    if per_episode_count[episode_index] >= self.max_segments_per_episode:
                        continue
                    key = (episode_index, decision_index)
                    if key in selected_starts:
                        continue
                    sequences.append(
                        _replay_sequence(
                            episodes[episode_index],
                            learn_start=decision_index,
                            learn_steps=learn_steps,
                            burn_in_steps=burn_in_steps,
                            sampling_index=sampling_indexes[episode_index],
                        )
                    )
                    selected_starts.add(key)
                    per_episode_count[episode_index] += 1
                    macro_added += 1
                    if macro_added >= macro_limit:
                        break
                if len(sequences) >= maximum:
                    return result()

            for quota_index in range(self.max_segments_per_episode):
                for episode_index in order:
                    # Preserve the original every-episode-before-second-visit
                    # fairness even after a macro sequence was preselected.
                    if per_episode_count[episode_index] > quota_index:
                        continue
                    episode_choices = choices[episode_index]
                    available = tuple(
                        choice
                        for choice in episode_choices
                        if (
                            episode_index,
                            choice * learn_steps,
                        )
                        not in selected_starts
                    )
                    if not available:
                        continue
                    episode = episodes[episode_index]
                    learn_start = available[0] * learn_steps
                    sequences.append(
                        _replay_sequence(
                            episode,
                            learn_start=learn_start,
                            learn_steps=learn_steps,
                            burn_in_steps=burn_in_steps,
                            sampling_index=sampling_indexes[episode_index],
                        )
                    )
                    selected_starts.add((episode_index, learn_start))
                    per_episode_count[episode_index] += 1
                    if len(sequences) >= maximum:
                        return result()

            return result()

    def snapshot(self) -> tuple[CompletedEpisode, ...]:
        with self._lock:
            return tuple(self._items)

    def metrics(self) -> dict[str, int | str]:
        with self._sample_lock:
            with self._lock:
                return {
                    "version": EPISODIC_REPLAY_VERSION,
                    "size": len(self._items),
                    "capacity": self.capacity,
                    "storage_nbytes": self._storage_nbytes,
                    "byte_capacity": self.byte_capacity,
                    "episode_byte_capacity": self.episode_byte_capacity,
                    "max_segments_per_episode": self.max_segments_per_episode,
                    "put_count": self._put_count,
                    "sample_count": self._sample_count,
                    "macro_sample_count": self._macro_sample_count,
                    "eviction_count": self._eviction_count,
                    "duplicate_count": self._duplicate_count,
                    "oversize_count": self._oversize_count,
                    "maximum_observed_episode_steps": (self._maximum_observed_episode_steps),
                }

    def state_dict(self) -> dict[str, object]:
        """Return exact replay/RNG state for an exact-resume checkpoint."""

        with self._sample_lock:
            with self._lock:
                return {
                    "version": EPISODIC_REPLAY_VERSION,
                    "capacity": self.capacity,
                    "byte_capacity": self.byte_capacity,
                    "episode_byte_capacity": self.episode_byte_capacity,
                    "max_segments_per_episode": self.max_segments_per_episode,
                    "items": tuple(self._items),
                    "rng_state": deepcopy(self._rng.bit_generator.state),
                    "put_count": self._put_count,
                    "sample_count": self._sample_count,
                    "macro_sample_count": self._macro_sample_count,
                    "eviction_count": self._eviction_count,
                    "duplicate_count": self._duplicate_count,
                    "oversize_count": self._oversize_count,
                    "maximum_observed_episode_steps": (self._maximum_observed_episode_steps),
                }

    def load_state_dict(self, payload: object) -> None:
        """Validate an exact checkpoint completely, then replace atomically."""

        if not isinstance(payload, dict):
            raise TypeError("episodic replay checkpoint must be an object")
        expected = {
            "version",
            "capacity",
            "byte_capacity",
            "episode_byte_capacity",
            "max_segments_per_episode",
            "items",
            "rng_state",
            "put_count",
            "sample_count",
            "macro_sample_count",
            "eviction_count",
            "duplicate_count",
            "oversize_count",
            "maximum_observed_episode_steps",
        }
        if set(payload) != expected or payload.get("version") != EPISODIC_REPLAY_VERSION:
            raise ValueError("unsupported episodic replay checkpoint schema")
        configured = {
            "capacity": self.capacity,
            "byte_capacity": self.byte_capacity,
            "episode_byte_capacity": self.episode_byte_capacity,
            "max_segments_per_episode": self.max_segments_per_episode,
        }
        for name, expected_value in configured.items():
            if payload.get(name) != expected_value:
                raise ValueError(f"episodic replay checkpoint {name} differs")

        items = payload.get("items")
        if not isinstance(items, tuple) or not all(isinstance(item, CompletedEpisode) for item in items):
            raise TypeError("episodic replay checkpoint items are invalid")
        if len(items) > self.capacity:
            raise ValueError("episodic replay checkpoint exceeds episode capacity")
        ids = [item.episode_id for item in items]
        if len(ids) != len(set(ids)):
            raise ValueError("episodic replay checkpoint contains duplicate episode IDs")
        if any(item.data_partition != "training" for item in items):
            raise ValueError("episodic replay checkpoint contains non-training data")
        validated_items = tuple(_validate_completed_episode_payload(item) for item in items)
        sizes = [item.storage_nbytes() for item in validated_items]
        if any(size > self.episode_byte_capacity for size in sizes):
            raise ValueError("episodic replay checkpoint exceeds per-episode byte capacity")
        storage_nbytes = sum(sizes)
        if storage_nbytes > self.byte_capacity:
            raise ValueError("episodic replay checkpoint exceeds total byte capacity")
        sampling_indexes = tuple(
            _build_episode_sampling_index(item, storage_nbytes=size)
            for item, size in zip(validated_items, sizes, strict=True)
        )

        counters: dict[str, int] = {}
        for name in (
            "put_count",
            "sample_count",
            "macro_sample_count",
            "eviction_count",
            "duplicate_count",
            "oversize_count",
            "maximum_observed_episode_steps",
        ):
            value = payload.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"episodic replay {name} must be a non-negative integer")
            counters[name] = value
        if counters["macro_sample_count"] > counters["sample_count"]:
            raise ValueError("episodic replay macro_sample_count cannot exceed sample_count")
        largest_episode = max((len(item.steps) for item in items), default=0)
        if counters["maximum_observed_episode_steps"] < largest_episode:
            raise ValueError("episodic replay maximum observed length is inconsistent")
        if counters["put_count"] != len(validated_items) + counters["eviction_count"]:
            raise ValueError("episodic replay put/eviction accounting is inconsistent")

        rng_state = payload.get("rng_state")
        if not isinstance(rng_state, dict):
            raise TypeError("episodic replay RNG checkpoint must be an object")
        validated_rng_state = deepcopy(rng_state)
        probe = np.random.default_rng()
        try:
            probe.bit_generator.state = validated_rng_state
        except (TypeError, ValueError) as exc:
            raise ValueError("episodic replay RNG checkpoint is invalid") from exc

        # All validation above is side-effect free.  Only this critical section
        # mutates live replay state, so a rejected checkpoint is atomic.
        with self._sample_lock:
            with self._lock:
                self._items = deque(validated_items)
                self._sampling_indexes = deque(sampling_indexes)
                self._episode_ids = set(ids)
                self._storage_nbytes = storage_nbytes
                self._rng.bit_generator.state = deepcopy(validated_rng_state)
                self._put_count = counters["put_count"]
                self._sample_count = counters["sample_count"]
                self._macro_sample_count = counters["macro_sample_count"]
                self._eviction_count = counters["eviction_count"]
                self._duplicate_count = counters["duplicate_count"]
                self._oversize_count = counters["oversize_count"]
                self._maximum_observed_episode_steps = counters["maximum_observed_episode_steps"]

__all__ = [
    "EPISODE_TRAJECTORY_VERSION",
    "EPISODIC_REPLAY_VERSION",
    "BackfilledEpisodeStep",
    "BoundaryOutcome",
    "BoundedEpisodicReplay",
    "CompletedEpisode",
    "EpisodeCompletion",
    "EpisodeDecisionStep",
    "HorizonTargets",
    "ReplaySequence",
    "backfill_completed_episode",
]
