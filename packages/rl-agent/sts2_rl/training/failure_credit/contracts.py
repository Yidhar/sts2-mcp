"""Versioned factual failure-credit contracts.

The v4 subsystem deliberately separates three questions:

* whether a task/local outcome is authoritative;
* what factual evidence was observed by a detector; and
* which learner targets that evidence is allowed to produce.

Detector evidence is stored explicitly.  A learner must never reconstruct a
cycle from a truncated learning tail.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from itertools import pairwise
from typing import Final

import numpy as np
import numpy.typing as npt

from sts2_rl.encoding import EncodedDecisionSnapshot
from sts2_rl.semantics import IdentityTriple, SemanticKey

FAILURE_CREDIT_SCHEMA_VERSION: Final = "sts2-failure-credit-v4"
FAILURE_CREDIT_COMPILER_VERSION: Final = "sts2-failure-credit-compiler-v1"


def _key(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    if len(value) > 512:
        raise ValueError(f"{label} exceeds the 512-character ABI limit")
    return value


def _finite(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{label} must be a finite number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{label} must be finite")
    return normalized


def _non_negative_integer(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    if value < 0:
        raise ValueError(f"{label} must be non-negative")
    return value


def _owned_recurrent_state(
    value: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    state = np.asarray(value)
    if state.dtype != np.float32 or state.ndim != 1:
        raise ValueError("initial_recurrent_state must be rank-1 float32")
    if not len(state):
        raise ValueError("initial_recurrent_state must not be empty")
    if not np.all(np.isfinite(state)):
        raise ValueError("initial_recurrent_state must be finite")
    result = np.ascontiguousarray(state).copy()
    result.setflags(write=False)
    return result


def _validate_semantic_key(value: object, *, label: str) -> SemanticKey:
    """Validate both halves of a semantic identity.

    The digest is only an index.  Retaining and validating the canonical
    payload prevents a digest-only implementation from silently becoming the
    factual authority for failure attribution.
    """

    if not isinstance(value, SemanticKey):
        raise TypeError(f"{label} must be SemanticKey")
    value.verify()
    return value


def _semantic_key_identity(key: SemanticKey) -> tuple[str, str, str, bytes]:
    """Return the full collision-auditable identity, never only the digest."""

    return (
        key.namespace,
        key.schema_version,
        key.digest,
        key.canonical_payload,
    )


def _validate_identity_triple(value: object, *, label: str) -> IdentityTriple:
    if not isinstance(value, IdentityTriple):
        raise TypeError(f"{label} must be IdentityTriple")
    _validate_semantic_key(value.exact, label=f"{label}.exact")
    _validate_semantic_key(value.loop, label=f"{label}.loop")
    _validate_semantic_key(value.comparison, label=f"{label}.comparison")
    return value


def _validate_semantic_injectivity(
    values: tuple[SemanticKey, ...],
    *,
    label: str,
) -> None:
    """Fail closed on duplicate or digest-colliding candidate identities."""

    seen_digests: dict[tuple[str, str, str], bytes] = {}
    seen_identities: set[tuple[str, str, str, bytes]] = set()
    for value in values:
        key = _validate_semantic_key(value, label=label)
        digest_identity = (key.namespace, key.schema_version, key.digest)
        prior_payload = seen_digests.get(digest_identity)
        if prior_payload is not None and prior_payload != key.canonical_payload:
            raise ValueError(f"{label} contains a semantic digest collision")
        seen_digests[digest_identity] = key.canonical_payload
        identity = _semantic_key_identity(key)
        if identity in seen_identities:
            raise ValueError(f"{label} must be injective within one decision")
        seen_identities.add(identity)


class EvidenceStratum(StrEnum):
    """Stable replay indexes; values are checkpoint ABI."""

    DIRECT_WITNESS = "DIRECT_WITNESS"
    MULTI_EDGE_CYCLE = "MULTI_EDGE_CYCLE"
    MATCHED_OUTCOME_PAIR = "MATCHED_OUTCOME_PAIR"
    RISK_SEQUENCE = "RISK_SEQUENCE"
    COMPLETION_CONTROL = "COMPLETION_CONTROL"
    UNRESOLVED_STALL = "UNRESOLVED_STALL"
    CENSORED = "CENSORED"


class FailureOutcome(StrEnum):
    COMPLETED = "completed"
    DEADLOCK_CYCLE = "deadlock_cycle"
    DEADLOCK_STALL = "deadlock_stall"
    CENSORED = "censored"

    @property
    def failed(self) -> bool:
        return self in {
            FailureOutcome.DEADLOCK_CYCLE,
            FailureOutcome.DEADLOCK_STALL,
        }


class TargetAuthority(StrEnum):
    NATIVE_TERMINAL = "native_terminal"
    OBJECTIVE_CONFIRMED = "objective_confirmed"
    VERIFIED_TRANSITION = "verified_transition"
    DETECTOR_CONFIRMED = "detector_confirmed"
    CENSORED = "censored"


class WitnessKind(StrEnum):
    DIRECT_WITNESS = EvidenceStratum.DIRECT_WITNESS
    MULTI_EDGE_CYCLE = EvidenceStratum.MULTI_EDGE_CYCLE
    MATCHED_OUTCOME_PAIR = EvidenceStratum.MATCHED_OUTCOME_PAIR
    RISK_SEQUENCE = EvidenceStratum.RISK_SEQUENCE
    COMPLETION_CONTROL = EvidenceStratum.COMPLETION_CONTROL


class DirectPolicyTarget(StrEnum):
    AVOID = "avoid"
    PREFER = "prefer"


@dataclass(frozen=True, slots=True)
class CreditProvenance:
    """Exact source/semantic ABI for one incident and its compiled credit."""

    run_id: str
    game_version: str
    environment_schema_version: str
    identity_version: str
    detector_version: str
    adapter_version: str
    collector_version: str
    policy_version: int
    data_partition: str = "training"
    schema_version: str = FAILURE_CREDIT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for label, value in (
            ("run_id", self.run_id),
            ("game_version", self.game_version),
            ("environment_schema_version", self.environment_schema_version),
            ("identity_version", self.identity_version),
            ("detector_version", self.detector_version),
            ("adapter_version", self.adapter_version),
            ("collector_version", self.collector_version),
        ):
            _key(value, label=label)
        _non_negative_integer(self.policy_version, label="policy_version")
        if self.data_partition != "training":
            raise ValueError("failure-credit replay rejects held-out/evaluation data")
        if self.schema_version != FAILURE_CREDIT_SCHEMA_VERSION:
            raise ValueError(f"unsupported failure-credit schema: {self.schema_version!r}")


@dataclass(frozen=True, slots=True)
class LearningStep:
    """One learner-ready factual decision plus its semantic identities."""

    decision_id: str
    episode_step: int
    snapshot: EncodedDecisionSnapshot
    action_index: int
    behavior_log_probability: float
    policy_version: int
    node: IdentityTriple
    anchor: SemanticKey
    candidate_actions: tuple[IdentityTriple, ...]
    forced: bool

    def __post_init__(self) -> None:
        _key(self.decision_id, label="decision_id")
        _non_negative_integer(self.episode_step, label="episode_step")
        if not isinstance(self.snapshot, EncodedDecisionSnapshot):
            raise TypeError("snapshot must be EncodedDecisionSnapshot")
        if isinstance(self.action_index, bool) or not isinstance(self.action_index, int):
            raise TypeError("action_index must be an integer")
        if not 0 <= self.action_index < self.snapshot.candidate_count:
            raise ValueError("action_index is outside the snapshot candidate range")
        if not bool(self.snapshot.action_mask[self.action_index]):
            raise ValueError("action_index selects an encoder-disabled candidate")
        behavior_log_probability = _finite(
            self.behavior_log_probability,
            label="behavior_log_probability",
        )
        if behavior_log_probability > 1e-7:
            raise ValueError("behavior_log_probability cannot exceed zero")
        _non_negative_integer(self.policy_version, label="policy_version")
        _validate_identity_triple(self.node, label="node")
        _validate_semantic_key(self.anchor, label="anchor")
        if not isinstance(self.candidate_actions, tuple) or not self.candidate_actions:
            raise ValueError("candidate_actions must be a non-empty tuple")
        for candidate_index, candidate in enumerate(self.candidate_actions):
            _validate_identity_triple(
                candidate,
                label=f"candidate_actions[{candidate_index}]",
            )
        if len(self.candidate_actions) != self.snapshot.candidate_count:
            raise ValueError("candidate_actions length differs from snapshot candidate_count")
        # The semantics kernel is responsible for strict equality grouping.
        # Failure replay refuses any remaining many-to-one identity instead of
        # guessing which concrete action should receive credit.
        _validate_semantic_injectivity(
            tuple(candidate.exact for candidate in self.candidate_actions),
            label="candidate action exact identities",
        )
        _validate_semantic_injectivity(
            tuple(candidate.loop for candidate in self.candidate_actions),
            label="candidate action loop identities",
        )
        _validate_semantic_injectivity(
            tuple(candidate.comparison for candidate in self.candidate_actions),
            label="candidate action comparison identities",
        )
        if not isinstance(self.forced, bool):
            raise TypeError("forced must be a boolean")
        enabled_count = int(np.count_nonzero(self.snapshot.action_mask))
        if self.forced and enabled_count != 1:
            raise ValueError("a forced decision must expose exactly one enabled candidate")

    @property
    def actor_eligible(self) -> bool:
        return not self.forced and int(np.count_nonzero(self.snapshot.action_mask)) > 1

    @property
    def selected_action(self) -> IdentityTriple:
        return self.candidate_actions[self.action_index]


@dataclass(frozen=True, slots=True)
class LearningContext:
    """Recurrent replay context shared by one incident's compiled targets."""

    context_id: str
    episode_id: str
    start_step: int
    initial_recurrent_state: npt.NDArray[np.float32]
    steps: tuple[LearningStep, ...]
    burn_in_steps: int = 0

    def __post_init__(self) -> None:
        _key(self.context_id, label="context_id")
        _key(self.episode_id, label="episode_id")
        _non_negative_integer(self.start_step, label="start_step")
        if not isinstance(self.steps, tuple) or not self.steps:
            raise ValueError("learning context steps must be a non-empty tuple")
        if not all(isinstance(step, LearningStep) for step in self.steps):
            raise TypeError("learning context contains a non-LearningStep item")
        if self.steps[0].episode_step != self.start_step:
            raise ValueError("start_step must equal the first factual episode_step")
        episode_steps = tuple(step.episode_step for step in self.steps)
        if any(right <= left for left, right in pairwise(episode_steps)):
            raise ValueError("learning context episode steps must be strictly increasing")
        decision_ids = tuple(step.decision_id for step in self.steps)
        if len(set(decision_ids)) != len(decision_ids):
            raise ValueError("learning context decision IDs must be unique")
        _non_negative_integer(self.burn_in_steps, label="burn_in_steps")
        if self.burn_in_steps >= len(self.steps):
            raise ValueError("burn-in must leave at least one learning step")
        object.__setattr__(
            self,
            "initial_recurrent_state",
            _owned_recurrent_state(self.initial_recurrent_state),
        )

    @property
    def learn_step_indices(self) -> tuple[int, ...]:
        return tuple(range(self.burn_in_steps, len(self.steps)))


@dataclass(frozen=True, slots=True)
class OutcomeArm:
    """One complete, recurrently replayable side of an observed outcome pair."""

    incident_id: str
    context: LearningContext
    step_index: int
    outcome: FailureOutcome

    def __post_init__(self) -> None:
        _key(self.incident_id, label="incident_id")
        if not isinstance(self.context, LearningContext):
            raise TypeError("outcome arm context must be LearningContext")
        if isinstance(self.step_index, bool) or not isinstance(self.step_index, int):
            raise TypeError("outcome arm step_index must be an integer")
        if not self.context.burn_in_steps <= self.step_index < len(self.context.steps):
            raise ValueError("outcome arm must reference a learning step")
        if not isinstance(self.outcome, FailureOutcome):
            raise TypeError("outcome arm has the wrong outcome type")
        if self.outcome is FailureOutcome.CENSORED:
            raise ValueError("a censored outcome cannot form an atomic outcome pair")
        if not self.context.steps[self.step_index].actor_eligible:
            raise ValueError("an outcome-pair arm must reference a policy decision")

    @property
    def step(self) -> LearningStep:
        return self.context.steps[self.step_index]


@dataclass(frozen=True, slots=True)
class MatchedOutcomePair:
    """Atomic factual comparison; a replay can never store only one arm."""

    pair_id: str
    better: OutcomeArm
    worse: OutcomeArm

    def __post_init__(self) -> None:
        _key(self.pair_id, label="pair_id")
        if not isinstance(self.better, OutcomeArm) or not isinstance(self.worse, OutcomeArm):
            raise TypeError("matched outcome pair arms have the wrong type")
        if self.better.outcome is not FailureOutcome.COMPLETED:
            raise ValueError("better outcome-pair arm must be a completed transaction")
        if not self.worse.outcome.failed:
            raise ValueError("worse outcome-pair arm must be an authoritative local failure")
        if self.better.incident_id == self.worse.incident_id:
            raise ValueError("matched outcome pair arms must come from distinct incidents")
        better_step = self.better.step
        worse_step = self.worse.step
        if _semantic_key_identity(better_step.node.comparison) != _semantic_key_identity(worse_step.node.comparison):
            raise ValueError("matched outcome pair comparison-node semantics differ")
        better_candidates = {
            _semantic_key_identity(candidate.comparison) for candidate in better_step.candidate_actions
        }
        worse_candidates = {_semantic_key_identity(candidate.comparison) for candidate in worse_step.candidate_actions}
        if better_candidates != worse_candidates:
            raise ValueError("matched outcome pair candidate comparison semantics differ")
        if _semantic_key_identity(better_step.selected_action.comparison) == _semantic_key_identity(
            worse_step.selected_action.comparison
        ):
            raise ValueError("matched outcome pair must compare different factual actions")


@dataclass(frozen=True, slots=True)
class LoopEdgeEvidence:
    """One detector-time semantic loop edge.

    ``node`` and ``action`` are the reviewed recurrence identities, not exact
    state/action hashes and not an anchor scope.  The supporting episode steps
    are captured by the detector before any replay tail is truncated.
    """

    node: SemanticKey
    action: SemanticKey
    supporting_episode_steps: tuple[int, ...]

    def __post_init__(self) -> None:
        _validate_semantic_key(self.node, label="loop edge node")
        _validate_semantic_key(self.action, label="loop edge action")
        if not isinstance(self.supporting_episode_steps, tuple) or not self.supporting_episode_steps:
            raise ValueError("loop edge must retain detector supporting steps")
        for value in self.supporting_episode_steps:
            _non_negative_integer(value, label="loop edge supporting_episode_step")
        if tuple(sorted(self.supporting_episode_steps)) != self.supporting_episode_steps:
            raise ValueError("loop edge supporting episode steps must be sorted")
        if len(set(self.supporting_episode_steps)) != len(self.supporting_episode_steps):
            raise ValueError("loop edge supporting episode steps must be unique")

    @property
    def identity(
        self,
    ) -> tuple[
        tuple[str, str, str, bytes],
        tuple[str, str, str, bytes],
    ]:
        return (
            _semantic_key_identity(self.node),
            _semantic_key_identity(self.action),
        )


@dataclass(frozen=True, slots=True)
class PolicyWitness:
    """Immutable detector/compiler evidence; never reconstructed from a tail."""

    witness_id: str
    kind: WitnessKind
    attributed_step_indices: tuple[int, ...]
    supporting_episode_steps: tuple[int, ...]
    occurrences: int
    cycle_span: int | None
    successor_confirmed: bool
    loop_edges: tuple[LoopEdgeEvidence, ...] = ()
    behavior_mean_log_probability: float | None = None
    outcome_pair: MatchedOutcomePair | None = None

    def __post_init__(self) -> None:
        _key(self.witness_id, label="witness_id")
        if not isinstance(self.kind, WitnessKind):
            raise TypeError("policy witness kind has the wrong type")
        if not isinstance(self.attributed_step_indices, tuple):
            raise TypeError("attributed_step_indices must be a tuple")
        for value in self.attributed_step_indices:
            _non_negative_integer(value, label="attributed_step_index")
        if len(set(self.attributed_step_indices)) != len(self.attributed_step_indices):
            raise ValueError("policy witness attributed indices must be unique")
        if not isinstance(self.supporting_episode_steps, tuple) or not self.supporting_episode_steps:
            raise ValueError("policy witness must retain detector supporting steps")
        for value in self.supporting_episode_steps:
            _non_negative_integer(value, label="supporting_episode_step")
        if tuple(sorted(self.supporting_episode_steps)) != self.supporting_episode_steps:
            raise ValueError("supporting episode steps must be sorted")
        if len(set(self.supporting_episode_steps)) != len(self.supporting_episode_steps):
            raise ValueError("supporting episode steps must be unique")
        _non_negative_integer(self.occurrences, label="occurrences")
        if not isinstance(self.successor_confirmed, bool):
            raise TypeError("successor_confirmed must be a boolean")
        if self.cycle_span is not None:
            if _non_negative_integer(self.cycle_span, label="cycle_span") == 0:
                raise ValueError("cycle_span must be positive")
        if self.behavior_mean_log_probability is not None:
            behavior_probability = _finite(
                self.behavior_mean_log_probability,
                label="behavior_mean_log_probability",
            )
            if behavior_probability > 1e-7:
                raise ValueError("behavior_mean_log_probability cannot exceed zero")
        if not isinstance(self.loop_edges, tuple) or not all(
            isinstance(edge, LoopEdgeEvidence) for edge in self.loop_edges
        ):
            raise TypeError("policy witness loop_edges have the wrong type")
        edge_identities = tuple(edge.identity for edge in self.loop_edges)
        if len(set(edge_identities)) != len(edge_identities):
            raise ValueError("policy witness loop edges must be semantically unique")
        if any(
            episode_step not in self.supporting_episode_steps
            for edge in self.loop_edges
            for episode_step in edge.supporting_episode_steps
        ):
            raise ValueError("loop-edge support must be included in witness support")

        if self.kind is WitnessKind.MATCHED_OUTCOME_PAIR:
            if self.outcome_pair is None:
                raise ValueError("matched-outcome witness requires one atomic pair")
            if self.attributed_step_indices:
                raise ValueError("matched-outcome witness uses pair arms, not local attributed indices")
        elif self.outcome_pair is not None:
            raise ValueError("only matched-outcome witness may carry an outcome pair")

        if self.kind in {
            WitnessKind.DIRECT_WITNESS,
            WitnessKind.MULTI_EDGE_CYCLE,
        }:
            if self.occurrences < 2 or self.cycle_span is None or not self.successor_confirmed:
                raise ValueError("cycle witness requires repeated, successor-confirmed evidence")
            if not self.attributed_step_indices:
                raise ValueError("cycle witness must identify its learner-visible cycle core")
            if self.kind is WitnessKind.DIRECT_WITNESS and len(self.loop_edges) != 1:
                raise ValueError("direct witness requires exactly one detector-time loop edge")
            if self.kind is WitnessKind.DIRECT_WITNESS and len(self.loop_edges[0].supporting_episode_steps) < 2:
                raise ValueError("direct witness edge requires at least two detector observations")
            if self.kind is WitnessKind.MULTI_EDGE_CYCLE and len(self.loop_edges) < 2:
                raise ValueError("multi-edge witness requires at least two detector-time edges")
            if self.cycle_span is None:  # pragma: no cover - guarded above
                raise ValueError("cycle witness lost its detector span")
            if not any(
                right - left == self.cycle_span
                for edge in self.loop_edges
                for left_index, left in enumerate(edge.supporting_episode_steps)
                for right in edge.supporting_episode_steps[left_index + 1 :]
            ):
                raise ValueError("cycle span is not demonstrated by repeated detector-time edge support")
        elif self.kind in {
            WitnessKind.RISK_SEQUENCE,
            WitnessKind.COMPLETION_CONTROL,
        }:
            if not self.attributed_step_indices:
                raise ValueError("risk/completion witness must identify factual learning steps")
            if self.loop_edges:
                raise ValueError("risk/completion witness cannot carry loop-edge blame")
        elif self.kind is WitnessKind.MATCHED_OUTCOME_PAIR and self.loop_edges:
            raise ValueError("matched-outcome witness cannot carry loop-edge blame")


@dataclass(frozen=True, slots=True)
class FailureIncident:
    """One authoritative or censored local outcome with immutable evidence."""

    incident_id: str
    scope_key: str
    failure_kind: str
    outcome: FailureOutcome
    task_authority: TargetAuthority
    local_authority: TargetAuthority
    task_return: float | None
    local_failure_cost: float | None
    context: LearningContext
    witnesses: tuple[PolicyWitness, ...]
    detector_window_steps: int
    progress_epoch: int
    provenance: CreditProvenance
    schema_version: str = FAILURE_CREDIT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _key(self.incident_id, label="incident_id")
        _key(self.scope_key, label="scope_key")
        _key(self.failure_kind, label="failure_kind")
        if not isinstance(self.outcome, FailureOutcome):
            raise TypeError("incident outcome has the wrong type")
        if not isinstance(self.task_authority, TargetAuthority):
            raise TypeError("task_authority has the wrong type")
        if not isinstance(self.local_authority, TargetAuthority):
            raise TypeError("local_authority has the wrong type")
        if (self.task_authority is TargetAuthority.CENSORED) != (self.task_return is None):
            raise ValueError("task authority/return mask is inconsistent")
        if self.task_return is not None:
            _finite(self.task_return, label="task_return")
        if (self.local_authority is TargetAuthority.CENSORED) != (self.local_failure_cost is None):
            raise ValueError("local authority/failure-cost mask is inconsistent")
        if self.local_failure_cost is not None:
            cost = _finite(self.local_failure_cost, label="local_failure_cost")
            if not 0.0 <= cost <= 1.0:
                raise ValueError("local_failure_cost must be in [0, 1]")
            if self.outcome is FailureOutcome.COMPLETED and cost != 0.0:
                raise ValueError("completed local transaction must have zero failure cost")
            if self.outcome.failed and cost <= 0.0:
                raise ValueError("failed local transaction must have positive failure cost")
        if self.outcome.failed and self.local_authority is TargetAuthority.CENSORED:
            raise ValueError("failed outcome requires authoritative local failure credit")
        if not isinstance(self.context, LearningContext):
            raise TypeError("incident context must be LearningContext")
        if not isinstance(self.witnesses, tuple) or not all(
            isinstance(witness, PolicyWitness) for witness in self.witnesses
        ):
            raise TypeError("incident witnesses must be a tuple of PolicyWitness")
        witness_ids = tuple(witness.witness_id for witness in self.witnesses)
        if len(set(witness_ids)) != len(witness_ids):
            raise ValueError("incident witness IDs must be unique")
        for witness in self.witnesses:
            if any(index >= len(self.context.steps) for index in witness.attributed_step_indices):
                raise ValueError("policy witness references a step outside the incident context")
            for index in witness.attributed_step_indices:
                if self.context.steps[index].episode_step not in witness.supporting_episode_steps:
                    raise ValueError("policy witness attribution lacks detector-time episode support")
            if witness.kind is WitnessKind.MATCHED_OUTCOME_PAIR:
                pair = witness.outcome_pair
                if pair is None:  # pragma: no cover - PolicyWitness invariant
                    raise ValueError("matched witness lost its atomic pair")
                if pair.worse.incident_id != self.incident_id:
                    raise ValueError("matched pair must be owned by its worse incident")
                if pair.worse.context is not self.context:
                    raise ValueError(
                        "matched pair worse arm must alias the incident context"
                    )
        _non_negative_integer(self.detector_window_steps, label="detector_window_steps")
        if self.detector_window_steps <= 0:
            raise ValueError("detector_window_steps must be positive")
        for witness in self.witnesses:
            observed_span = witness.supporting_episode_steps[-1] - witness.supporting_episode_steps[0]
            if observed_span > self.detector_window_steps:
                raise ValueError("witness evidence exceeds its detector window")
        _non_negative_integer(self.progress_epoch, label="progress_epoch")
        if not isinstance(self.provenance, CreditProvenance):
            raise TypeError("incident provenance has the wrong type")
        if self.schema_version != FAILURE_CREDIT_SCHEMA_VERSION:
            raise ValueError(f"unsupported failure-credit schema: {self.schema_version!r}")
        if self.outcome is FailureOutcome.CENSORED:
            if (
                self.task_authority is not TargetAuthority.CENSORED
                or self.local_authority is not TargetAuthority.CENSORED
                or self.witnesses
            ):
                raise ValueError("censored incident cannot carry authoritative credit or witnesses")

    def state_dict(self) -> dict[str, object]:
        return {
            "version": FAILURE_CREDIT_SCHEMA_VERSION,
            "incident": self,
        }

    @classmethod
    def from_state_dict(cls, payload: object) -> FailureIncident:
        if not isinstance(payload, dict) or set(payload) != {"version", "incident"}:
            raise ValueError("failure incident state has the wrong schema")
        if payload.get("version") != FAILURE_CREDIT_SCHEMA_VERSION:
            raise ValueError("unsupported failure incident state version")
        incident = payload.get("incident")
        if not isinstance(incident, cls):
            raise TypeError("failure incident state has the wrong typed payload")
        return incident


@dataclass(frozen=True, slots=True)
class ScalarCredit:
    step_index: int
    target: float
    horizon: int
    witness_id: str | None = None

    def __post_init__(self) -> None:
        _non_negative_integer(self.step_index, label="scalar credit step_index")
        _finite(self.target, label="scalar credit target")
        if _non_negative_integer(self.horizon, label="scalar credit horizon") == 0:
            raise ValueError("scalar credit horizon must be positive")
        if self.witness_id is not None:
            _key(self.witness_id, label="scalar credit witness_id")


@dataclass(frozen=True, slots=True)
class DirectPolicyCredit:
    step_index: int
    target: DirectPolicyTarget
    witness_id: str

    def __post_init__(self) -> None:
        _non_negative_integer(self.step_index, label="direct policy step_index")
        if not isinstance(self.target, DirectPolicyTarget):
            raise TypeError("direct policy target has the wrong type")
        _key(self.witness_id, label="direct policy witness_id")


@dataclass(frozen=True, slots=True)
class CyclePolicyCredit:
    step_indices: tuple[int, ...]
    behavior_mean_log_probability: float
    margin: float
    witness_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.step_indices, tuple) or not self.step_indices:
            raise ValueError("cycle policy credit requires at least one step")
        for value in self.step_indices:
            _non_negative_integer(value, label="cycle policy step_index")
        if len(set(self.step_indices)) != len(self.step_indices):
            raise ValueError("cycle policy step indices must be unique")
        behavior = _finite(
            self.behavior_mean_log_probability,
            label="cycle behavior_mean_log_probability",
        )
        if behavior > 1e-7:
            raise ValueError("cycle behavior_mean_log_probability cannot exceed zero")
        if _finite(self.margin, label="cycle margin") <= 0.0:
            raise ValueError("cycle margin must be positive")
        _key(self.witness_id, label="cycle policy witness_id")


@dataclass(frozen=True, slots=True)
class ContrastPolicyCredit:
    pair: MatchedOutcomePair
    margin: float
    witness_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.pair, MatchedOutcomePair):
            raise TypeError("contrast policy pair has the wrong type")
        if _finite(self.margin, label="contrast margin") <= 0.0:
            raise ValueError("contrast margin must be positive")
        _key(self.witness_id, label="contrast policy witness_id")


@dataclass(frozen=True, slots=True)
class RiskSequenceCredit:
    step_indices: tuple[int, ...]
    terminal_cost: float
    discount: float
    witness_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.step_indices, tuple) or not self.step_indices:
            raise ValueError("risk sequence requires at least one learning step")
        for value in self.step_indices:
            _non_negative_integer(value, label="risk sequence step_index")
        if len(set(self.step_indices)) != len(self.step_indices):
            raise ValueError("risk sequence step indices must be unique")
        cost = _finite(self.terminal_cost, label="risk terminal_cost")
        if not 0.0 <= cost <= 1.0:
            raise ValueError("risk terminal_cost must be in [0, 1]")
        discount = _finite(self.discount, label="risk discount")
        if not 0.0 < discount <= 1.0:
            raise ValueError("risk discount must be in (0, 1]")
        if self.witness_id is not None:
            _key(self.witness_id, label="risk witness_id")


@dataclass(frozen=True, slots=True)
class CreditPlan:
    """Learner-ready immutable targets compiled from one incident."""

    plan_id: str
    incident_id: str
    context: LearningContext
    task_value_targets: tuple[ScalarCredit, ...]
    task_q_targets: tuple[ScalarCredit, ...]
    liveness_value_targets: tuple[ScalarCredit, ...]
    liveness_q_targets: tuple[ScalarCredit, ...]
    direct_policy_targets: tuple[DirectPolicyCredit, ...]
    cycle_policy_targets: tuple[CyclePolicyCredit, ...]
    contrast_policy_targets: tuple[ContrastPolicyCredit, ...]
    risk_sequences: tuple[RiskSequenceCredit, ...]
    strata: tuple[EvidenceStratum, ...]
    provenance: CreditProvenance
    compiler_version: str = FAILURE_CREDIT_COMPILER_VERSION
    schema_version: str = FAILURE_CREDIT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _key(self.plan_id, label="plan_id")
        _key(self.incident_id, label="incident_id")
        if not isinstance(self.context, LearningContext):
            raise TypeError("credit plan context has the wrong type")
        for label, values, expected in (
            ("task_value_targets", self.task_value_targets, ScalarCredit),
            ("task_q_targets", self.task_q_targets, ScalarCredit),
            ("liveness_value_targets", self.liveness_value_targets, ScalarCredit),
            ("liveness_q_targets", self.liveness_q_targets, ScalarCredit),
            ("direct_policy_targets", self.direct_policy_targets, DirectPolicyCredit),
            ("cycle_policy_targets", self.cycle_policy_targets, CyclePolicyCredit),
            ("contrast_policy_targets", self.contrast_policy_targets, ContrastPolicyCredit),
            ("risk_sequences", self.risk_sequences, RiskSequenceCredit),
        ):
            if not isinstance(values, tuple) or not all(isinstance(item, expected) for item in values):
                raise TypeError(f"{label} has the wrong typed tuple")
        if not isinstance(self.strata, tuple) or not self.strata:
            raise ValueError("credit plan must declare at least one evidence stratum")
        if not all(isinstance(stratum, EvidenceStratum) for stratum in self.strata):
            raise TypeError("credit plan strata have the wrong type")
        if len(set(self.strata)) != len(self.strata):
            raise ValueError("credit plan strata must be unique")
        if not isinstance(self.provenance, CreditProvenance):
            raise TypeError("credit plan provenance has the wrong type")
        for scalar_target in (
            *self.task_value_targets,
            *self.task_q_targets,
            *self.liveness_value_targets,
            *self.liveness_q_targets,
        ):
            if scalar_target.step_index >= len(self.context.steps):
                raise ValueError("credit target references a step outside its learning context")
        for direct_target in self.direct_policy_targets:
            if direct_target.step_index >= len(self.context.steps):
                raise ValueError("direct policy target references a step outside its learning context")
        for cycle_target in self.cycle_policy_targets:
            if any(index >= len(self.context.steps) for index in cycle_target.step_indices):
                raise ValueError("cycle target references a step outside its learning context")
        for risk_target in self.risk_sequences:
            if any(index >= len(self.context.steps) for index in risk_target.step_indices):
                raise ValueError("risk sequence references a step outside its learning context")
        if self.compiler_version != FAILURE_CREDIT_COMPILER_VERSION:
            raise ValueError(f"unsupported credit compiler: {self.compiler_version!r}")
        if self.schema_version != FAILURE_CREDIT_SCHEMA_VERSION:
            raise ValueError(f"unsupported failure-credit schema: {self.schema_version!r}")

    @property
    def direct_actor_label_count(self) -> int:
        """Return factual non-Q actor labels in this plan.

        Direct, cycle, and contrast targets have an explicit policy target.
        Risk sequences are reported separately because their actor direction
        is derived from the calibrated liveness Q head.
        """

        return (
            len(self.direct_policy_targets)
            + len(self.cycle_policy_targets)
            + len(self.contrast_policy_targets)
        )

    @property
    def risk_actor_candidate_count(self) -> int:
        """Return decision rows that may become centered-risk actor labels."""

        return sum(
            int(not self.context.steps[step_index].forced)
            for sequence in self.risk_sequences
            for step_index in sequence.step_indices
        )

    @property
    def actor_label_count(self) -> int:
        """Return all policy-relevant labels before freshness suppression."""

        return self.direct_actor_label_count + self.risk_actor_candidate_count

    @property
    def target_count(self) -> int:
        """Return every typed learner target represented by this plan."""

        return sum(
            (
                len(self.task_value_targets),
                len(self.task_q_targets),
                len(self.liveness_value_targets),
                len(self.liveness_q_targets),
                len(self.direct_policy_targets),
                len(self.cycle_policy_targets),
                len(self.contrast_policy_targets),
                len(self.risk_sequences),
            )
        )


__all__ = [
    "FAILURE_CREDIT_COMPILER_VERSION",
    "FAILURE_CREDIT_SCHEMA_VERSION",
    "ContrastPolicyCredit",
    "CreditPlan",
    "CreditProvenance",
    "CyclePolicyCredit",
    "DirectPolicyCredit",
    "DirectPolicyTarget",
    "EvidenceStratum",
    "FailureIncident",
    "FailureOutcome",
    "IdentityTriple",
    "LearningContext",
    "LearningStep",
    "LoopEdgeEvidence",
    "MatchedOutcomePair",
    "OutcomeArm",
    "PolicyWitness",
    "RiskSequenceCredit",
    "ScalarCredit",
    "SemanticKey",
    "TargetAuthority",
    "WitnessKind",
]
