"""Atomic, fail-closed checkpoint ABI for recurrent V-trace v2."""

from __future__ import annotations

import json
import pickle
import random
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from sts2_baseline import SequenceUnroll
from sts2_rl.checkpoints import (
    EXACT_RESUME_REQUIRED_FILES,
    AtomicCheckpointDirectory,
    CheckpointIntegrityError,
    ValidatedResumeCheckpoint,
    build_checkpoint_provenance,
    contract_metadata,
    reject_frozen_exact_resume,
    revalidate_checkpoint_identity,
    validate_attested_model_initialization_checkpoint,
    validate_model_initialization_checkpoint,
    validate_resume_checkpoint,
)
from sts2_rl.encoding import grounding_encoding_identity
from sts2_rl.semantics import (
    DECISION_IDENTITY_CONTRACT_VERSION,
    MACRO_EDGE_CONTRACT_VERSION,
    PROGRESS_RECEIPT_CONTRACT_VERSION,
    PROGRESS_SCOPE_CONTRACT_VERSION,
    SURFACE_REGISTRY_CONTRACT_VERSION,
)
from sts2_rl.semantics.identity import SEMANTIC_KEY_CONTRACT_VERSION

from .config import TrainingConfig, training_config_from_mapping
from .episode_replay import BoundedEpisodicReplay
from .factory import TrainingResources
from .failure_credit import (
    FAILURE_CREDIT_COLLECTOR_VERSION,
    FAILURE_CREDIT_COMPILER_VERSION,
    FAILURE_CREDIT_DETECTOR_VERSION,
    FAILURE_CREDIT_MATCHER_VERSION,
    FAILURE_CREDIT_SCHEMA_VERSION,
    FAILURE_EVIDENCE_REPLAY_VERSION,
    BoundedFailureCreditReplay,
)
from .sdpa import (
    validate_sdpa_execution_mapping,
    validate_sdpa_transition_mapping,
)
from .seeding import SIGNED_INT32_MAX
from .transaction import (
    TRANSACTION_LIFECYCLE_VERSION,
    BoundedTransactionReplay,
    TransactionLifecycleEvidence,
    TransactionStep,
    TransactionTrace,
)

_CHECKPOINT_FORMAT = "sts2-recurrent-vtrace-checkpoint-v5"
_MODEL_INITIALIZATION_FORMATS = frozenset(
    {
        _CHECKPOINT_FORMAT,
        "sts2-recurrent-vtrace-checkpoint-v4",
        "sts2-recurrent-vtrace-checkpoint-v3",
    }
)
_LONG_HORIZON_MISSING_FORMATS = frozenset({"sts2-recurrent-vtrace-checkpoint-v3"})
_QUEUE_PAYLOAD_VERSION = "sts2-rollout-queue-pickle-v2"
_ACTOR_SUPERVISOR_STATE_VERSION = "sts2-actor-supervisor-state-v1"
_EVALUATION_GATE_STATE_VERSION = "sts2-evaluation-gate-state-v1"
_LONG_HORIZON_VALUE_HEAD_ABI = "sts2-long-horizon-value-heads-v1"
_EPISODIC_TARGET_ABI = "sts2-episodic-task-targets-one-terminal-unit-v2"
_LIVENESS_COST_HEAD_ABI = "sts2-liveness-cost-heads-v1"
_CHECKPOINT_ROLES = frozenset(
    {
        "ordinary",
        "healthy_evaluation_anchor",
        "guard_failure_evidence",
        "deadlock_alert_evidence",
    }
)

_LONG_HORIZON_HEAD_PREFIXES = (
    "combat_task_value_head.",
    "act_task_value_head.",
    "run_task_value_head.",
    "combat_revival_cost_value_head.",
    "act_revival_cost_value_head.",
    "run_revival_cost_value_head.",
)
_LIVENESS_HEAD_PREFIXES = (
    "liveness_cost_value_head.",
    "candidate_liveness_cost_head.",
)
_TRANSACTION_HEAD_PREFIXES = (
    "candidate_effect_head.",
    "selection_delta_head.",
    "transaction_q_head.",
)
_TRANSACTION_HEAD_SUFFIXES = frozenset(
    {
        "0.weight",
        "0.bias",
        "1.weight",
        "1.bias",
        "3.weight",
        "3.bias",
    }
)


def _decision_semantics_abi() -> dict[str, str]:
    """Return the complete factual-decision identity used by failure replay.

    A checkpoint must not resume into a process that interprets the same
    stored evidence under different surface, scope, progress, or macro-edge
    semantics.  Keeping the component versions explicit makes reviewable
    migrations possible without treating a single opaque digest as authority.
    """

    return {
        "semantic_key": SEMANTIC_KEY_CONTRACT_VERSION,
        "decision_identity": DECISION_IDENTITY_CONTRACT_VERSION,
        "surface_registry": SURFACE_REGISTRY_CONTRACT_VERSION,
        "progress_scope": PROGRESS_SCOPE_CONTRACT_VERSION,
        "progress_receipt": PROGRESS_RECEIPT_CONTRACT_VERSION,
        "macro_edge": MACRO_EDGE_CONTRACT_VERSION,
    }


def _failure_credit_abi() -> dict[str, str]:
    return {
        "schema": FAILURE_CREDIT_SCHEMA_VERSION,
        "collector": FAILURE_CREDIT_COLLECTOR_VERSION,
        "detector": FAILURE_CREDIT_DETECTOR_VERSION,
        "matcher": FAILURE_CREDIT_MATCHER_VERSION,
        "compiler": FAILURE_CREDIT_COMPILER_VERSION,
        "replay": FAILURE_EVIDENCE_REPLAY_VERSION,
        "liveness_heads": _LIVENESS_COST_HEAD_ABI,
    }


# Exact resume always requires the complete active encoding identity.  Model
# parameter initialization has deliberately narrow legacy exceptions.  V9
# added strict equivalence grouping for card-selection candidates while
# preserving every v8 input tensor dimension and learned parameter shape.  V10
# adds only newly exposed factual macro entities and uses existing feature
# slots. V11 preserves all v10 tensors but changes behavior-policy semantics to
# the count-balanced hierarchical action-branch distribution. V12 keeps every
# parameter shape while admitting exact native upgrade-card projections,
# separating unknown-zone hashes from fixed zone IDs, and decoupling stable
# embedding hashes from collision-free decision-local equality bindings.  The
# v12 encoded snapshot is therefore a new runtime/data ABI even though every
# learned parameter remains shape compatible. Queue/replay action indexes,
# encoded snapshots, behavior probabilities, and optimizer moments still
# belong to their source ABI, so these exceptions are valid only for the
# model-only path. V13 admits native upgrade previews and isolates unknown
# zones. V14 fingerprints the native shop-item alias table after the V13
# implementation had changed category-only card-removal candidates without
# changing its compact identity. V13 -> V14 is shape compatible only for
# model parameters; candidate semantics, replay and optimizer state are not.
#
# Keep both sides as complete, immutable identities rather than accepting a
# version prefix or dimensions alone.  Any later encoder edit changes the
# fingerprint and fails closed until it receives a separately reviewed entry.
_V8_ENCODING_IDENTITY = {
    "version": "grounded-relational-runtime-encoding-v8",
    "min_token_feature_dim": 224,
    "feature_abi_end": 214,
    "fingerprint_sha256": "8bc0204fe3201871cf3bdb3be39deaac9cc1b02f830ac2ba72d3e29bef0ddf58",
}
_V9_ENCODING_IDENTITY = {
    "version": "grounded-relational-runtime-encoding-v9",
    "min_token_feature_dim": 224,
    "feature_abi_end": 215,
    "fingerprint_sha256": "a953c6a01cd0ae85e77f0916ee6f072f7a97a59cbfdae6967003f9e8894dba1b",
}
_V10_ENCODING_IDENTITY = {
    "version": "grounded-relational-runtime-encoding-v10",
    "min_token_feature_dim": 224,
    "feature_abi_end": 215,
    "fingerprint_sha256": "4caae6f3c6baafb31ce476615776e22cdea2e6073ee7b4893247a4ffef2e524f",
}
_V11_ENCODING_IDENTITY = {
    "version": "grounded-relational-runtime-encoding-v11",
    "min_token_feature_dim": 224,
    "feature_abi_end": 215,
    "fingerprint_sha256": "5d150d5949c70e49203f7808e663abcfcbd897bcb9d18a55852b117292503bb7",
}
_V12_ENCODING_IDENTITY = {
    "version": "grounded-relational-runtime-encoding-v12",
    "min_token_feature_dim": 224,
    "feature_abi_end": 215,
    "fingerprint_sha256": "d1bc0220f7aa58e7afacaa83c1fa1ce339b65d58729f651012cd81d5ad4febf3",
}
_V13_ENCODING_IDENTITY = {
    "version": "grounded-relational-runtime-encoding-v13",
    "min_token_feature_dim": 224,
    "feature_abi_end": 215,
    "fingerprint_sha256": "ac119f0d1fe0de5c09394e091169f3b7712084bce8a90a9d02be4732f60ce5bf",
}
_V14_ENCODING_IDENTITY = {
    "version": "grounded-relational-runtime-encoding-v14",
    "min_token_feature_dim": 224,
    "feature_abi_end": 215,
    # A literal is required here: deriving it from the active encoder would
    # silently bless any future unreviewed semantic change.
    "fingerprint_sha256": "6a169803fdcd399272357dfe351a8b7375f16a1cb9e7cfccdc3b047f13f746ce",
}
_REVIEWED_MODEL_INITIALIZATION_ENCODING_MIGRATIONS = (
    (
        _V12_ENCODING_IDENTITY,
        _V13_ENCODING_IDENTITY,
    ),
    (
        _V13_ENCODING_IDENTITY,
        _V14_ENCODING_IDENTITY,
    ),
)


def _has_reviewed_model_initialization_encoding_path(
    source: object,
    target: object,
) -> bool:
    """Return whether exact pinned identities have a fully reviewed path.

    Migrations are graph edges rather than a version-range allowlist. This
    permits V12 to reach V14 only through the independently reviewed V12 ->
    V13 and V13 -> V14 model-only edges, while unknown or tampered
    fingerprints still fail closed.
    """

    frontier: list[object] = [source]
    visited: list[object] = []
    while frontier:
        current = frontier.pop()
        if current == target:
            return True
        if any(current == item for item in visited):
            continue
        visited.append(current)
        frontier.extend(
            destination
            for origin, destination in _REVIEWED_MODEL_INITIALIZATION_ENCODING_MIGRATIONS
            if current == origin
        )
    return False


@dataclass(frozen=True, slots=True)
class TrainingState:
    environment_steps: int = 0
    learner_updates: int = 0
    episodes: int = 0
    evaluation_episodes: int = 0
    policy_version: int = 0
    actor_policy_version: int = 0
    consumed_unrolls: int = 0
    maximum_observed_candidates: int = 0
    training_deadlock_streak: int = 0
    training_deadlock_alerts: int = 0
    evaluation_guard_rollbacks: int = 0

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.policy_version < self.actor_policy_version:
            raise ValueError("actor policy version cannot be newer than learner policy")


_TRAINING_SCHEDULE_STATE_VERSION = "sts2-training-schedule-state-v1"


@dataclass(frozen=True, slots=True)
class TrainingScheduleState:
    """Independent optimization/exploration clocks for a training lineage.

    Model-parameter initialization intentionally resets optimizer, replay,
    rollout, RNG, and lineage counters. It must not silently restart mature
    entropy, epsilon, calibration, and risk-actor schedules, however. These
    offsets preserve that maturity without pretending the new run is an exact
    continuation of the source checkpoint.
    """

    environment_steps_offset: int = 0
    learner_updates_offset: int = 0
    policy_version_offset: int = 0

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")

    def effective_environment_steps(self, state: TrainingState) -> int:
        return self.environment_steps_offset + state.environment_steps

    def effective_learner_updates(self, state: TrainingState) -> int:
        return self.learner_updates_offset + state.learner_updates

    def effective_policy_version(self, state: TrainingState) -> int:
        return self.policy_version_offset + state.policy_version

    def inherited_after(self, state: TrainingState) -> TrainingScheduleState:
        """Freeze this source lineage's effective clocks as new offsets."""

        return TrainingScheduleState(
            environment_steps_offset=self.effective_environment_steps(state),
            learner_updates_offset=self.effective_learner_updates(state),
            policy_version_offset=self.effective_policy_version(state),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "version": _TRAINING_SCHEDULE_STATE_VERSION,
            "environment_steps_offset": self.environment_steps_offset,
            "learner_updates_offset": self.learner_updates_offset,
            "policy_version_offset": self.policy_version_offset,
        }


@dataclass(frozen=True, slots=True)
class EvaluationGateState:
    """Exact-resume identities for completed evaluation gates."""

    completed_validation_steps: tuple[int, ...] = ()
    completed_early_validation_steps: tuple[int, ...] = ()
    completed_final_audit_steps: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            values = getattr(self, name)
            if not isinstance(values, tuple):
                raise TypeError(f"{name} must be an immutable tuple")
            previous = -1
            for step in values:
                if isinstance(step, bool) or not isinstance(step, int) or step < 0 or step <= previous:
                    raise ValueError(f"{name} must contain strictly increasing non-negative integers")
                previous = step

    def to_mapping(self) -> dict[str, Any]:
        return {
            "version": _EVALUATION_GATE_STATE_VERSION,
            "completed_validation_steps": list(self.completed_validation_steps),
            "completed_early_validation_steps": list(self.completed_early_validation_steps),
            "completed_final_audit_steps": list(self.completed_final_audit_steps),
        }

    @classmethod
    def from_mapping(cls, payload: object) -> EvaluationGateState:
        if not isinstance(payload, Mapping):
            raise TypeError("checkpoint evaluation_state must be an object")
        expected = {
            "version",
            "completed_validation_steps",
            "completed_early_validation_steps",
            "completed_final_audit_steps",
        }
        if set(payload) != expected:
            raise ValueError("checkpoint evaluation_state keys mismatch")
        if payload.get("version") != _EVALUATION_GATE_STATE_VERSION:
            raise ValueError("unsupported checkpoint evaluation_state version")

        values: dict[str, tuple[int, ...]] = {}
        for name in (
            "completed_validation_steps",
            "completed_early_validation_steps",
            "completed_final_audit_steps",
        ):
            raw = payload.get(name)
            if not isinstance(raw, list):
                raise TypeError(f"checkpoint evaluation_state.{name} must be an array")
            values[name] = tuple(raw)
        return cls(
            completed_validation_steps=values["completed_validation_steps"],
            completed_early_validation_steps=values["completed_early_validation_steps"],
            completed_final_audit_steps=values["completed_final_audit_steps"],
        )


@dataclass(frozen=True, slots=True)
class ActorSupervisorState:
    """Exact-resume state for actor incident accounting and circuit breaking.

    Fingerprints use a sorted tuple rather than a mutable mapping so a state
    snapshot cannot be changed after it has been validated.  JSON metadata
    still uses a normal object through :meth:`to_mapping`.
    """

    episode_attempts: int = 0
    consecutive_incidents: int = 0
    incident_fingerprints: tuple[tuple[str, int], ...] = ()
    recent_incident_attempts: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        for name in ("episode_attempts", "consecutive_incidents"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if not isinstance(self.incident_fingerprints, tuple):
            raise TypeError("incident_fingerprints must be an immutable tuple")
        fingerprints: list[str] = []
        incident_count = 0
        for item in self.incident_fingerprints:
            if not isinstance(item, tuple) or len(item) != 2:
                raise TypeError("incident_fingerprints entries must be (fingerprint, count) tuples")
            fingerprint, count = item
            if not isinstance(fingerprint, str) or not fingerprint or fingerprint.strip() != fingerprint:
                raise ValueError("actor incident fingerprints must be non-empty canonical strings")
            if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                raise ValueError("actor incident fingerprint counts must be positive integers")
            fingerprints.append(fingerprint)
            incident_count += count
        if fingerprints != sorted(fingerprints) or len(fingerprints) != len(set(fingerprints)):
            raise ValueError("actor incident fingerprints must be unique and sorted")
        if incident_count > self.episode_attempts:
            raise ValueError("actor incident count cannot exceed episode attempts")
        if self.consecutive_incidents > incident_count:
            raise ValueError("consecutive actor incidents cannot exceed all incidents")

        if not isinstance(self.recent_incident_attempts, tuple):
            raise TypeError("recent_incident_attempts must be an immutable tuple")
        previous = 0
        for attempt in self.recent_incident_attempts:
            if (
                isinstance(attempt, bool)
                or not isinstance(attempt, int)
                or attempt <= previous
                or attempt > self.episode_attempts
            ):
                raise ValueError("recent actor incident attempts must be strictly increasing valid attempts")
            previous = attempt
        if len(self.recent_incident_attempts) > 100:
            raise ValueError("actor supervisor may retain at most 100 recent incidents")
        if len(self.recent_incident_attempts) > incident_count:
            raise ValueError("recent actor incidents cannot exceed all incidents")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "version": _ACTOR_SUPERVISOR_STATE_VERSION,
            "episode_attempts": self.episode_attempts,
            "consecutive_incidents": self.consecutive_incidents,
            "incident_fingerprints": dict(self.incident_fingerprints),
            "recent_incident_attempts": list(self.recent_incident_attempts),
        }

    @classmethod
    def from_mapping(cls, payload: object) -> ActorSupervisorState:
        if not isinstance(payload, Mapping):
            raise TypeError("checkpoint actor_supervisor_state must be an object")
        expected = {
            "version",
            "episode_attempts",
            "consecutive_incidents",
            "incident_fingerprints",
            "recent_incident_attempts",
        }
        if set(payload) != expected:
            raise ValueError("checkpoint actor_supervisor_state keys mismatch")
        if payload["version"] != _ACTOR_SUPERVISOR_STATE_VERSION:
            raise ValueError("unsupported checkpoint actor supervisor state")
        raw_fingerprints = payload["incident_fingerprints"]
        if not isinstance(raw_fingerprints, Mapping):
            raise TypeError("checkpoint actor incident_fingerprints must be an object")
        raw_recent = payload["recent_incident_attempts"]
        if not isinstance(raw_recent, list | tuple):
            raise TypeError("checkpoint recent_incident_attempts must be an array")
        return cls(
            episode_attempts=payload["episode_attempts"],
            consecutive_incidents=payload["consecutive_incidents"],
            incident_fingerprints=tuple(sorted(raw_fingerprints.items())),
            recent_incident_attempts=tuple(raw_recent),
        )


def _tensor_spec(state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for key, value in state.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            raise TypeError("model state must contain only named tensors")
        result[key] = {"shape": list(value.shape), "dtype": str(value.dtype)}
    return result


def _optimizer_spec(
    optimizer: torch.optim.Optimizer,
    state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = optimizer.state_dict() if state is None else state
    groups = payload.get("param_groups")
    raw_state = payload.get("state")
    if not isinstance(groups, list) or not isinstance(raw_state, dict):
        raise TypeError("optimizer state must expose param_groups and state")
    return {
        "class": f"{type(optimizer).__module__}.{type(optimizer).__qualname__}",
        "group_sizes": [len(group["params"]) for group in groups],
        "group_keys": [sorted(str(key) for key in group if key != "params") for group in groups],
        "state_entries": len(raw_state),
    }


def _validate_unrolls(
    items: tuple[SequenceUnroll, ...],
    *,
    config: TrainingConfig,
) -> None:
    if not isinstance(items, tuple) or not all(isinstance(item, SequenceUnroll) for item in items):
        raise TypeError("checkpoint rollout queue must be an unroll tuple")
    if len(items) > config.rollout.queue_capacity:
        raise ValueError("checkpoint rollout queue exceeds configured capacity")
    fingerprint = grounding_encoding_identity()["fingerprint_sha256"]
    for item in items:
        item.validate(
            expected_config=config.model.to_encoding_config(),
            expected_fingerprint=fingerprint,
            recurrent_hidden_dim=config.model.recurrent_hidden_dim,
            maximum_length=config.rollout.unroll_length,
        )


def _queue_payload(
    resources: TrainingResources,
    *,
    config: TrainingConfig,
) -> dict[str, Any]:
    items = resources.rollout_queue.snapshot()
    _validate_unrolls(items, config=config)
    return {
        "version": _QUEUE_PAYLOAD_VERSION,
        "capacity": resources.rollout_queue.capacity,
        "items": items,
    }


def _queue_spec(payload: dict[str, Any]) -> dict[str, Any]:
    items = payload["items"]
    versions = [item.policy_version for item in items]
    return {
        "version": payload["version"],
        "capacity": payload["capacity"],
        "size": len(items),
        "environment_steps": sum(item.environment_steps for item in items),
        "minimum_policy_version": min(versions) if versions else None,
        "maximum_policy_version": max(versions) if versions else None,
    }


def _transaction_replay_spec(payload: dict[str, Any]) -> dict[str, Any]:
    items = payload["items"]
    return {
        "version": payload["version"],
        "capacity": payload["capacity"],
        "byte_capacity": payload["byte_capacity"],
        "size": len(items),
        "storage_nbytes": sum(item.storage_nbytes() for item in items),
        "put_count": payload["put_count"],
        "sample_count": payload["sample_count"],
        "eviction_count": payload["eviction_count"],
        "duplicate_count": payload["duplicate_count"],
    }


def _validated_transaction_replay_payload(
    payload: object,
    *,
    config: TrainingConfig,
) -> dict[str, Any]:
    """Validate and normalize a detached transaction replay payload.

    Pickle restores dataclass instances without invoking their ``__post_init__``
    methods.  The replay container can therefore prove its own capacity, RNG,
    and index invariants while still containing a stale or malformed nested
    trace.  Reconstruct every trace/step to re-run the v4 dataclass contracts,
    and validate every encoded snapshot against the active decision ABI before
    any live learner resource is mutated.
    """

    if not isinstance(payload, dict):
        raise TypeError("transaction replay checkpoint must be an object")
    probe = BoundedTransactionReplay(
        capacity=config.transaction_learning.replay_capacity,
        byte_capacity=config.transaction_learning.replay_byte_capacity,
        seed=config.runtime.seed,
    )
    probe.load_state_dict(payload)
    configured_burn_in = config.transaction_learning.burn_in_steps
    expected_config = config.model.to_encoding_config()
    expected_fingerprint = grounding_encoding_identity()["fingerprint_sha256"]
    expected_hidden_shape = (config.model.recurrent_hidden_dim,)
    validated_traces: list[TransactionTrace] = []
    for trace in probe.snapshot():
        validated_steps: list[TransactionStep] = []
        for step in trace.steps:
            raw_snapshot = step.snapshot
            # Protocol-5 checkpoints preserve NumPy read-only flags, but direct
            # state dictionaries and older pickle protocols do not. Rebuild all
            # three sparse tables and the snapshot so every nested array is an
            # owned, immutable replay object before it can reach live state.
            validated_snapshot = replace(
                raw_snapshot,
                world=replace(raw_snapshot.world),
                candidates=replace(raw_snapshot.candidates),
                locals=replace(raw_snapshot.locals),
            )
            # ``replace`` invokes the generated constructor and consequently
            # ``TransactionStep.__post_init__``.  This is intentionally done on
            # a detached copy: validation during checkpoint publication must
            # not mutate the immutable trace already owned by the live replay.
            validated_step = replace(step, snapshot=validated_snapshot)
            validated_step.snapshot.validate(
                expected_config=expected_config,
                expected_fingerprint=expected_fingerprint,
            )
            validated_steps.append(validated_step)
        validated_lifecycle: TransactionLifecycleEvidence | None = None
        if trace.lifecycle is not None:
            raw_lifecycle = trace.lifecycle
            validated_post_snapshot = raw_lifecycle.post_snapshot
            if validated_post_snapshot is not None:
                validated_post_snapshot = replace(
                    validated_post_snapshot,
                    world=replace(validated_post_snapshot.world),
                    candidates=replace(validated_post_snapshot.candidates),
                    locals=replace(validated_post_snapshot.locals),
                )
                validated_post_snapshot.validate(
                    expected_config=expected_config,
                    expected_fingerprint=expected_fingerprint,
                )
            validated_lifecycle = replace(
                raw_lifecycle,
                post_snapshot=validated_post_snapshot,
            )
        # Likewise, reconstructing the trace re-applies its version, training
        # partition, key, outcome, burn-in, and zero-state invariants that
        # pickle itself does not execute.
        validated_trace = replace(
            trace,
            steps=tuple(validated_steps),
            lifecycle=validated_lifecycle,
        )
        if validated_trace.initial_recurrent_state.shape != expected_hidden_shape:
            raise ValueError("checkpoint transaction recurrent state differs from model hidden size")
        if validated_trace.burn_in_steps > configured_burn_in:
            raise ValueError("checkpoint transaction trace exceeds configured burn-in")
        if validated_trace.start_step > 0 and validated_trace.burn_in_steps != configured_burn_in:
            raise ValueError("checkpoint non-initial transaction trace lacks configured burn-in")
        validated_traces.append(validated_trace)
    # Load exact continuation from the reconstructed immutable objects rather
    # than the raw pickle graph.  Ordering and replay-owned RNG/counters remain
    # byte-for-byte equivalent; only construction-time ownership/invariants are
    # normalized.
    return {**payload, "items": tuple(validated_traces)}


def _new_episodic_replay(*, config: TrainingConfig) -> BoundedEpisodicReplay:
    return BoundedEpisodicReplay(
        capacity=config.episodic_learning.replay_capacity_episodes,
        byte_capacity=config.episodic_learning.replay_capacity_bytes,
        episode_byte_capacity=config.episodic_learning.per_episode_capacity_bytes,
        max_segments_per_episode=config.episodic_learning.max_segments_per_episode,
        seed=config.runtime.seed,
    )


def _validated_episodic_replay_payload(
    payload: object,
    *,
    config: TrainingConfig,
) -> tuple[dict[str, Any], dict[str, int | str]]:
    """Validate a detached episodic replay in a fresh probe.

    Exact resume must prove the complete byte-bounded corpus, accounting
    counters, and sampler RNG are loadable before any live learner resource is
    mutated.  The probe has the exact active capacities but is otherwise
    independent of the runtime replay.  The metadata spec is read from the
    same loaded probe (``metrics`` is a pure read), so the payload is
    deserialized exactly once per validation site.
    """

    if not isinstance(payload, dict):
        raise TypeError("episodic replay checkpoint must be an object")
    probe = _new_episodic_replay(config=config)
    probe.load_state_dict(payload)
    expected_config = config.model.to_encoding_config()
    expected_fingerprint = grounding_encoding_identity()["fingerprint_sha256"]
    for episode in probe.snapshot():
        for step in episode.steps:
            step.snapshot.validate(
                expected_config=expected_config,
                expected_fingerprint=expected_fingerprint,
            )
    return payload, dict(probe.metrics())


def _new_failure_credit_replay(
    *,
    config: TrainingConfig,
) -> BoundedFailureCreditReplay:
    return BoundedFailureCreditReplay(
        capacity=config.failure_credit.replay_capacity,
        byte_capacity=config.failure_credit.replay_byte_capacity,
        seed=config.runtime.seed,
    )


def _validated_failure_credit_replay_payload(
    payload: object,
    *,
    config: TrainingConfig,
) -> tuple[dict[str, object], dict[str, int | str]]:
    """Validate an exact replay-v5 continuation in a detached owner.

    The typed corpus, atomic matched pairs, capacity/byte accounting, owned RNG,
    and all replay counters are validated before any live resource is mutated.
    A v3 transaction sidecar is never accepted under this filename or ABI.
    The metadata spec is read from the same loaded probe (``metrics`` is a
    pure read), so the payload is deserialized exactly once per validation
    site.
    """

    if not isinstance(payload, dict):
        raise TypeError("failure-credit replay checkpoint must be an object")
    probe = _new_failure_credit_replay(config=config)
    probe.load_state_dict(payload)
    return payload, dict(probe.metrics())


def _stochastic_state(resources: TrainingResources) -> dict[str, Any]:
    collector_device = next(resources.collector_model.parameters()).device
    cuda_rng_is_live = (
        resources.device.type == "cuda" or collector_device.type == "cuda" or torch.cuda.is_initialized()  # type: ignore[no-untyped-call]
    )
    cuda_states = torch.cuda.get_rng_state_all() if cuda_rng_is_live and torch.cuda.is_available() else []
    return {
        "version": "sts2-recurrent-stochastic-state-v3",
        "python_random": random.getstate(),
        "numpy_random": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": cuda_states,
        "collector": resources.collector.state_dict(),
        "learner_dynamics": resources.learner.dynamics_state_dict(),
    }


def _validate_stochastic_state(payload: object) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError("checkpoint stochastic state must be an object")
    expected = {
        "version",
        "python_random",
        "numpy_random",
        "torch_cpu",
        "torch_cuda",
        "collector",
        "learner_dynamics",
    }
    if set(payload) != expected:
        raise ValueError("checkpoint stochastic state keys mismatch")
    if payload["version"] != "sts2-recurrent-stochastic-state-v3":
        raise ValueError("unsupported checkpoint stochastic state")
    python_probe = random.Random()
    numpy_probe = np.random.RandomState()
    try:
        python_probe.setstate(payload["python_random"])
        numpy_probe.set_state(payload["numpy_random"])
    except (TypeError, ValueError) as exc:
        raise ValueError("checkpoint Python/NumPy RNG state is invalid") from exc
    torch_cpu = payload["torch_cpu"]
    if not isinstance(torch_cpu, torch.Tensor):
        raise TypeError("checkpoint Torch CPU RNG state must be a tensor")
    try:
        torch.Generator(device="cpu").set_state(torch_cpu.cpu())
    except RuntimeError as exc:
        raise ValueError("checkpoint Torch CPU RNG state is invalid") from exc
    torch_cuda = payload["torch_cuda"]
    if not isinstance(torch_cuda, list) or not all(isinstance(item, torch.Tensor) for item in torch_cuda):
        raise TypeError("checkpoint Torch CUDA RNG state must be a tensor list")
    if torch_cuda:
        if not torch.cuda.is_available():
            raise ValueError("checkpoint requires unavailable CUDA RNG state")
        if len(torch_cuda) != torch.cuda.device_count():
            raise ValueError("checkpoint CUDA device count differs from runtime")
        for index, cuda_state in enumerate(torch_cuda):
            try:
                torch.Generator(device=f"cuda:{index}").set_state(cuda_state.cpu())
            except RuntimeError as exc:
                raise ValueError(f"checkpoint Torch CUDA RNG state {index} is invalid") from exc
    collector = payload["collector"]
    if not isinstance(collector, dict):
        raise TypeError("checkpoint collector state must be an object")
    if collector.get("version") != "sts2-recurrent-collector-state-v3":
        raise ValueError("unsupported checkpoint collector state")
    if set(collector) != {"version", "episode_seed", "rng_state"}:
        raise ValueError("checkpoint collector state keys mismatch")
    episode_seed = collector.get("episode_seed")
    if (
        isinstance(episode_seed, bool)
        or not isinstance(episode_seed, int)
        or episode_seed < 0
        or episode_seed > SIGNED_INT32_MAX
        or episode_seed % 2 != 0
    ):
        raise ValueError("checkpoint collector seed must be even signed 32-bit")
    rng_state = collector.get("rng_state")
    if not isinstance(rng_state, dict):
        raise TypeError("checkpoint collector RNG state must be an object")
    generator_probe = np.random.default_rng()
    try:
        generator_probe.bit_generator.state = dict(rng_state)
    except (TypeError, ValueError) as exc:
        raise ValueError("checkpoint collector RNG state is invalid") from exc
    learner_dynamics = payload["learner_dynamics"]
    if not isinstance(learner_dynamics, dict):
        raise TypeError("checkpoint learner dynamics state must be an object")
    return payload


def _restore_stochastic_state(
    payload: dict[str, Any],
    *,
    resources: TrainingResources,
) -> None:
    resources.collector.load_state_dict(payload["collector"])
    resources.learner.load_dynamics_state_dict(payload["learner_dynamics"])
    random.setstate(payload["python_random"])
    np.random.set_state(payload["numpy_random"])
    torch.set_rng_state(payload["torch_cpu"].cpu())
    if payload["torch_cuda"]:
        if not torch.cuda.is_available():
            raise ValueError("checkpoint requires CUDA RNG state")
        torch.cuda.set_rng_state_all([item.cpu() for item in payload["torch_cuda"]])


def training_state_from_metadata(metadata: dict[str, Any]) -> TrainingState:
    raw = metadata.get("training_state")
    if not isinstance(raw, dict):
        raise ValueError("checkpoint has no training_state")
    expected = set(TrainingState.__dataclass_fields__)
    actual = set(raw)
    # ``maximum_observed_candidates`` is diagnostic-only and was added to the
    # v3 payload without changing any optimizer/RNG continuation semantics.
    # Existing exact-resume checkpoints therefore migrate this one absent
    # scalar to zero while every other missing or unknown key remains a hard
    # ABI failure.
    legacy_missing = {
        "maximum_observed_candidates",
        "training_deadlock_streak",
        "training_deadlock_alerts",
        "evaluation_guard_rollbacks",
    }
    if actual <= expected and expected - actual <= legacy_missing:
        raw = {
            **raw,
            **{name: 0 for name in expected - actual},
        }
    elif actual != expected:
        raise ValueError("checkpoint training_state keys mismatch")
    return TrainingState(**raw)


def training_schedule_state_from_metadata(metadata: Mapping[str, Any]) -> TrainingScheduleState:
    """Load schedule offsets, deriving the legacy zero-offset contract.

    A checkpoint without this field predates schedule inheritance. Its own
    counters remain the complete effective clocks, represented by zero offsets.
    Present payloads are strict and never partially repaired.
    """

    raw = metadata.get("training_schedule_state")
    if raw is None:
        return TrainingScheduleState()
    if not isinstance(raw, Mapping):
        raise TypeError("checkpoint training_schedule_state must be an object")
    expected = {
        "version",
        "environment_steps_offset",
        "learner_updates_offset",
        "policy_version_offset",
    }
    if set(raw) != expected:
        raise ValueError("checkpoint training_schedule_state keys mismatch")
    if raw.get("version") != _TRAINING_SCHEDULE_STATE_VERSION:
        raise ValueError("unsupported checkpoint training_schedule_state version")
    return TrainingScheduleState(
        environment_steps_offset=raw["environment_steps_offset"],
        learner_updates_offset=raw["learner_updates_offset"],
        policy_version_offset=raw["policy_version_offset"],
    )


def actor_supervisor_state_from_metadata(
    metadata: Mapping[str, Any],
) -> ActorSupervisorState:
    """Load the optional supervisor payload with a narrow legacy migration.

    Checkpoints written before actor incident recovery existed have no such
    field, so absence alone maps to the validated zero state.  A present field
    is never repaired or partially defaulted: malformed keys, values, or
    versions fail exact resume closed.
    """

    if "actor_supervisor_state" not in metadata:
        return ActorSupervisorState()
    return ActorSupervisorState.from_mapping(metadata["actor_supervisor_state"])


def evaluation_gate_state_from_metadata(
    metadata: Mapping[str, Any],
) -> EvaluationGateState | None:
    """Load explicit completed-gate identities when the checkpoint has them.

    Absence identifies a legacy checkpoint.  A present payload is validated
    strictly; malformed or partially specified state never falls back to a
    legacy inference.
    """

    if "evaluation_state" not in metadata:
        return None
    return EvaluationGateState.from_mapping(metadata["evaluation_state"])


def _validate_evaluation_gate_state_horizon(
    evaluation_state: EvaluationGateState | None,
    training_state: TrainingState,
) -> None:
    if evaluation_state is None:
        return
    for name in (
        "completed_validation_steps",
        "completed_early_validation_steps",
        "completed_final_audit_steps",
    ):
        future_steps = [step for step in getattr(evaluation_state, name) if step > training_state.environment_steps]
        if future_steps:
            raise ValueError(
                "checkpoint evaluation_state cannot complete gates beyond "
                f"training_state.environment_steps: {name}={future_steps}"
            )


def _validate_evaluation_gate_state_schedule(
    evaluation_state: EvaluationGateState | None,
    config: TrainingConfig,
) -> None:
    if evaluation_state is None:
        return
    schedules = {
        "completed_validation_steps": set(config.runtime.evaluation_steps),
        "completed_early_validation_steps": set(config.runtime.early_evaluation_steps),
        "completed_final_audit_steps": set(config.runtime.final_audit_steps),
    }
    for name, configured in schedules.items():
        unknown = sorted(set(getattr(evaluation_state, name)) - configured)
        if unknown:
            raise ValueError(
                "checkpoint evaluation_state contains gates absent from its "
                f"training_config schedule: {name}={unknown}"
            )


def _validate_encoding_contract(
    source: object,
    *,
    model_only: bool,
) -> None:
    """Validate the full decision ABI before any checkpoint tensor is read.

    Exact continuation may never cross a changed decision space: queued
    ``action_index`` values, behavior probabilities, transaction replay and
    recurrent state all belong to the source encoder.  Explicit model
    parameter initialization may cross only a complete path of explicitly
    reviewed, pinned encoding identities. Shape-compatible but otherwise
    unknown encoders remain rejected.
    """

    target = grounding_encoding_identity()
    if source == target:
        return
    if model_only and _has_reviewed_model_initialization_encoding_path(
        source,
        target,
    ):
        return
    if model_only:
        raise ValueError(
            "checkpoint encoding contract does not match; no reviewed " "model-parameter initialization migration"
        )
    raise ValueError("checkpoint encoding contract does not match")


def _validate_metadata(
    validated: ValidatedResumeCheckpoint,
    *,
    config: TrainingConfig,
    resolved_device: str | None,
    resolved_collector_device: str | None,
    model_only: bool,
) -> None:
    metadata = validated.metadata
    checkpoint_format = metadata.get("format")
    if model_only:
        if checkpoint_format not in _MODEL_INITIALIZATION_FORMATS:
            raise ValueError("unsupported model-initialization checkpoint format: " f"{checkpoint_format!r}")
    elif checkpoint_format != _CHECKPOINT_FORMAT:
        raise ValueError(
            "unsupported exact-resume checkpoint format: " f"{checkpoint_format!r}; expected {_CHECKPOINT_FORMAT!r}"
        )
    if metadata.get("model_config") != asdict(config.model.to_model_config()):
        raise ValueError("checkpoint recurrent model config does not match")
    _validate_encoding_contract(
        metadata.get("encoding_contract"),
        model_only=model_only,
    )
    if not isinstance(metadata.get("model_state_spec"), dict):
        raise ValueError("checkpoint has no model tensor specification")
    if model_only:
        return
    checkpoint_role = metadata.get("checkpoint_role")
    if checkpoint_role is not None and checkpoint_role not in _CHECKPOINT_ROLES:
        raise ValueError("exact-resume checkpoint has an invalid checkpoint role")
    recorded_sdpa = metadata.get("sdpa_backend")
    execution_provenance = metadata.get("execution_provenance")
    if recorded_sdpa is not None:
        normalized_sdpa = validate_sdpa_execution_mapping(recorded_sdpa)
        if not isinstance(execution_provenance, Mapping):
            raise ValueError("checkpoint with SDPA state has no execution provenance")
        validate_sdpa_transition_mapping(
            execution_provenance.get("sdpa_backend"),
            expected_current=normalized_sdpa,
        )
    elif isinstance(execution_provenance, Mapping) and ("sdpa_backend" in execution_provenance):
        raise ValueError("checkpoint SDPA transition has no recorded current execution state")
    training_state = training_state_from_metadata(metadata)
    training_schedule_state_from_metadata(metadata)
    actor_supervisor_state_from_metadata(metadata)
    evaluation_state = evaluation_gate_state_from_metadata(metadata)
    _validate_evaluation_gate_state_horizon(evaluation_state, training_state)
    if evaluation_state is not None:
        source_training_config = metadata.get("training_config")
        if not isinstance(source_training_config, Mapping):
            raise ValueError("checkpoint has no training_config object")
        _validate_evaluation_gate_state_schedule(
            evaluation_state,
            training_config_from_mapping(source_training_config),
        )
    checkpoint_lineage = metadata.get("lineage_config")
    active_lineage = config.lineage_mapping()
    if isinstance(checkpoint_lineage, dict) and "transaction_learning" not in checkpoint_lineage:
        # v10 checkpoints predate the optional sidecar.  The disabled default
        # changes neither learned parameters nor continuation semantics.
        transaction_section = active_lineage.get("transaction_learning")
        if isinstance(transaction_section, dict) and transaction_section.get("enabled") is False:
            checkpoint_lineage = {
                **checkpoint_lineage,
                "transaction_learning": transaction_section,
            }
    if checkpoint_lineage != active_lineage:
        raise ValueError("exact resume requires identical immutable training lineage")
    if metadata.get("resolved_device") != resolved_device:
        raise ValueError("exact resume requires the same learner device")
    if metadata.get("resolved_collector_device") != resolved_collector_device:
        raise ValueError("exact resume requires the same actor device")
    if not isinstance(metadata.get("queue_spec"), dict):
        raise ValueError("checkpoint has no rollout queue specification")
    if metadata.get("decision_semantics_abi") != _decision_semantics_abi():
        raise ValueError("exact-resume checkpoint decision-semantics ABI differs")
    if metadata.get("failure_credit_abi") != _failure_credit_abi():
        raise ValueError("exact-resume checkpoint failure-credit ABI differs")
    if metadata.get("failure_credit_mode") != config.failure_credit.mode:
        raise ValueError("exact-resume checkpoint failure-credit mode differs")
    liveness_enabled = config.failure_credit.learning_enabled
    if metadata.get("liveness_cost_heads_enabled") is not liveness_enabled:
        raise ValueError("exact-resume checkpoint liveness-head marker differs")
    for state_spec_name in ("model_state_spec", "actor_model_state_spec"):
        raw_state_spec = metadata.get(state_spec_name)
        if not isinstance(raw_state_spec, dict):
            raise ValueError(f"checkpoint has no {state_spec_name.replace('_', ' ')}")
        liveness_state_keys = {
            key for key in raw_state_spec if isinstance(key, str) and key.startswith(_LIVENESS_HEAD_PREFIXES)
        }
        liveness_prefixes_present = {
            prefix for prefix in _LIVENESS_HEAD_PREFIXES if any(key.startswith(prefix) for key in liveness_state_keys)
        }
        if liveness_enabled and liveness_prefixes_present != set(_LIVENESS_HEAD_PREFIXES):
            raise ValueError("exact-resume checkpoint has an incomplete liveness-head " f"family in {state_spec_name}")
        if not liveness_enabled and liveness_state_keys:
            raise ValueError(
                "checkpoint contains liveness heads while failure-credit " f"learning is disabled in {state_spec_name}"
            )
    if liveness_enabled:
        if metadata.get("failure_credit_replay_enabled") is not True:
            raise ValueError("failure-credit learning checkpoint has no replay-v5 marker")
        if not isinstance(
            metadata.get("failure_credit_replay_spec"),
            dict,
        ):
            raise ValueError("failure-credit learning checkpoint has no replay-v5 specification")
    else:
        if metadata.get("failure_credit_replay_enabled") is not False:
            raise ValueError("failure-credit replay marker differs from the non-learning mode")
        if metadata.get("failure_credit_replay_spec") is not None:
            raise ValueError("non-learning checkpoint cannot contain a failure-credit " "replay specification")
    if metadata.get("long_horizon_value_head_abi") != _LONG_HORIZON_VALUE_HEAD_ABI:
        raise ValueError("exact-resume checkpoint has no long-horizon value-head ABI marker")
    if config.transaction_learning.enabled:
        if metadata.get("transaction_heads_enabled") is not True:
            raise ValueError("transaction-enabled checkpoint has no head ABI marker")
        if metadata.get("transaction_lifecycle_abi") != TRANSACTION_LIFECYCLE_VERSION:
            raise ValueError("transaction-enabled checkpoint has no lifecycle-evidence ABI marker")
        if not isinstance(metadata.get("transaction_replay_spec"), dict):
            raise ValueError("transaction-enabled checkpoint has no replay specification")
    else:
        if metadata.get("transaction_lifecycle_abi") not in (None,):
            raise ValueError("non-transaction checkpoint cannot contain a lifecycle-evidence ABI marker")
    if config.episodic_learning.enabled:
        if metadata.get("episodic_target_abi") != _EPISODIC_TARGET_ABI:
            raise ValueError("exact-resume checkpoint has no episodic target ABI marker")
        if metadata.get("episodic_replay_enabled") is not True:
            raise ValueError("episodic-learning checkpoint has no replay ABI marker")
        if not isinstance(metadata.get("episodic_replay_spec"), dict):
            raise ValueError("episodic-learning checkpoint has no replay specification")
    elif metadata.get("episodic_replay_enabled") not in (False, None):
        raise ValueError("episodic replay marker differs from the disabled configuration")


def _exact_resume_required_files(config: TrainingConfig) -> frozenset[str]:
    required = set(EXACT_RESUME_REQUIRED_FILES)
    if config.transaction_learning.enabled:
        required.add("transaction_replay.pkl")
    if config.episodic_learning.enabled:
        required.add("episodic_replay.pkl")
    if config.failure_credit.learning_enabled:
        required.add("failure_credit_replay.pkl")
    return frozenset(required)


def _reuse_prevalidated_checkpoint(
    prevalidated: ValidatedResumeCheckpoint,
    checkpoint: str | Path,
    *,
    require_current_runtime_identity: bool,
    operation: str,
) -> ValidatedResumeCheckpoint:
    """Accept a same-process handle whose directory bytes were already hashed.

    Every cheap manifest/metadata identity check is re-executed at the strength
    the requested operation needs; only the redundant whole-directory SHA-256
    pass over the immutable atomic directory is skipped.  The handle must
    denote exactly the requested checkpoint path, and it must never cross a
    process boundary: launcher and trainer each perform their own full byte
    validation.
    """

    validated = revalidate_checkpoint_identity(
        prevalidated,
        require_current_runtime_identity=require_current_runtime_identity,
        operation=operation,
    )
    root = Path(checkpoint).expanduser().resolve(strict=False)
    if validated.root != root:
        raise ValueError("prevalidated checkpoint does not match the requested checkpoint path")
    return validated


def preflight_training_checkpoint(
    checkpoint: str | Path,
    *,
    config: TrainingConfig,
    resolved_device: str,
    resolved_collector_device: str,
    prevalidated: ValidatedResumeCheckpoint | None = None,
) -> ValidatedResumeCheckpoint:
    # Validate and hash the complete atomic directory using only the stable base
    # payload set first.  This lets the metadata format gate reject a v3 source
    # explicitly as model-initialization-only instead of misreporting the new
    # v4 episodic sidecar as missing. ``validate_resume_checkpoint`` already
    # hashes every manifest-listed file, including optional sidecars, so the
    # second phase only needs to require their manifest entries.  A caller in
    # the same process may pass its already byte-validated handle back in; the
    # semantic identity checks below still run in full against that handle.
    if prevalidated is None:
        validated = validate_resume_checkpoint(checkpoint)
    else:
        validated = _reuse_prevalidated_checkpoint(
            prevalidated,
            checkpoint,
            require_current_runtime_identity=True,
            operation="exact resume",
        )
    reject_frozen_exact_resume(validated)
    _validate_metadata(
        validated,
        config=config,
        resolved_device=resolved_device,
        resolved_collector_device=resolved_collector_device,
        model_only=False,
    )
    raw_entries = validated.manifest.get("files")
    entries = raw_entries if isinstance(raw_entries, list) else []
    listed = {
        str(entry.get("path")) for entry in entries if isinstance(entry, dict) and isinstance(entry.get("path"), str)
    }
    missing = sorted(_exact_resume_required_files(config) - listed)
    if missing:
        raise CheckpointIntegrityError("exact resume checkpoint is missing required manifest entries: " f"{missing}")
    return validated


def preflight_model_initialization(
    checkpoint: str | Path,
    *,
    config: TrainingConfig,
    prevalidated: ValidatedResumeCheckpoint | None = None,
    attestation: str | Path | None = None,
    attestation_sha256: str | None = None,
) -> ValidatedResumeCheckpoint:
    if bool(attestation) != bool(attestation_sha256):
        raise ValueError(
            "model-initialization attestation and SHA-256 must be supplied together"
        )
    if prevalidated is not None and attestation is not None:
        raise ValueError(
            "prevalidated handle and cross-process attestation are mutually exclusive"
        )
    if prevalidated is None:
        if attestation is None:
            validated = validate_model_initialization_checkpoint(checkpoint)
        else:
            assert attestation_sha256 is not None
            validated = validate_attested_model_initialization_checkpoint(
                checkpoint,
                attestation=attestation,
                expected_attestation_sha256=attestation_sha256,
            )
    else:
        validated = _reuse_prevalidated_checkpoint(
            prevalidated,
            checkpoint,
            require_current_runtime_identity=False,
            operation="model initialization",
        )
    _validate_metadata(
        validated,
        config=config,
        resolved_device=None,
        resolved_collector_device=None,
        model_only=True,
    )
    return validated


def save_training_checkpoint(
    target: str | Path,
    *,
    config: TrainingConfig,
    resources: TrainingResources,
    state: TrainingState,
    schedule_state: TrainingScheduleState | None = None,
    parent_checkpoint: str | Path | None = None,
    run_id: str | None = None,
    checkpoint_load_mode: str | None = None,
    parent_relation: str | None = None,
    actor_supervisor_state: ActorSupervisorState | None = None,
    evaluation_state: EvaluationGateState | None = None,
    execution_provenance: Mapping[str, Any] | None = None,
    checkpoint_role: str = "ordinary",
) -> Path:
    """Publish a checkpoint while the actor is quiescent between episodes."""

    if schedule_state is None:
        schedule_state = TrainingScheduleState()
    elif not isinstance(schedule_state, TrainingScheduleState):
        raise TypeError("schedule_state must be TrainingScheduleState")
    if actor_supervisor_state is None:
        actor_supervisor_state = ActorSupervisorState()
    elif not isinstance(actor_supervisor_state, ActorSupervisorState):
        raise TypeError("actor_supervisor_state must be ActorSupervisorState")
    if evaluation_state is not None and not isinstance(
        evaluation_state,
        EvaluationGateState,
    ):
        raise TypeError("evaluation_state must be EvaluationGateState or None")
    _validate_evaluation_gate_state_horizon(evaluation_state, state)
    _validate_evaluation_gate_state_schedule(evaluation_state, config)
    if execution_provenance is not None and not isinstance(
        execution_provenance,
        Mapping,
    ):
        raise TypeError("execution_provenance must be a mapping or None")
    if checkpoint_role not in _CHECKPOINT_ROLES:
        raise ValueError(
            "checkpoint_role must be one of "
            f"{sorted(_CHECKPOINT_ROLES)!r}"
        )
    provenance = build_checkpoint_provenance(
        parent_checkpoint=parent_checkpoint,
        experiment_run_id=run_id,
        config_version=config.version,
        config_profile=config.profile,
        checkpoint_load_mode=checkpoint_load_mode or "fresh",
        parent_relation=parent_relation,
    )
    publisher = AtomicCheckpointDirectory(target, provenance=provenance)
    staging = publisher.prepare()
    try:
        network_state = resources.model.state_dict()
        actor_network_state = resources.collector_model.state_dict()
        optimizer_state = resources.optimizer.state_dict()
        queue_payload = _queue_payload(resources, config=config)
        transaction_replay_payload = None
        if config.transaction_learning.enabled:
            if resources.transaction_replay is None:
                raise RuntimeError("transaction-enabled resources have no replay sidecar")
            transaction_replay_payload = resources.transaction_replay.state_dict()
            _validated_transaction_replay_payload(
                transaction_replay_payload,
                config=config,
            )
        episodic_replay_payload = None
        episodic_replay_spec = None
        if config.episodic_learning.enabled:
            if resources.episodic_replay is None:
                raise RuntimeError("episodic-learning resources have no replay sidecar")
            episodic_replay_payload = resources.episodic_replay.state_dict()
            episodic_replay_payload, episodic_replay_spec = _validated_episodic_replay_payload(
                episodic_replay_payload,
                config=config,
            )
        failure_credit_replay_payload = None
        failure_credit_replay_spec = None
        if config.failure_credit.learning_enabled:
            if resources.failure_credit_replay is None:
                raise RuntimeError("failure-credit learning resources have no replay-v5 sidecar")
            failure_credit_replay_payload = resources.failure_credit_replay.state_dict()
            failure_credit_replay_payload, failure_credit_replay_spec = _validated_failure_credit_replay_payload(
                failure_credit_replay_payload,
                config=config,
            )
        torch.save(network_state, staging / "network.pt")
        torch.save(actor_network_state, staging / "actor_network.pt")
        torch.save(optimizer_state, staging / "optimizer.pt")
        with (staging / "rollout_queue.pkl").open("wb") as handle:
            pickle.dump(queue_payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with (staging / "stochastic_state.pkl").open("wb") as handle:
            pickle.dump(
                _stochastic_state(resources),
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        if transaction_replay_payload is not None:
            with (staging / "transaction_replay.pkl").open("wb") as handle:
                pickle.dump(
                    transaction_replay_payload,
                    handle,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
        if episodic_replay_payload is not None:
            with (staging / "episodic_replay.pkl").open("wb") as handle:
                pickle.dump(
                    episodic_replay_payload,
                    handle,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
        if failure_credit_replay_payload is not None:
            with (staging / "failure_credit_replay.pkl").open("wb") as handle:
                pickle.dump(
                    failure_credit_replay_payload,
                    handle,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
        checkpoint_execution_provenance = dict(execution_provenance or {})
        # The process-owned state is authoritative.  Callers may attach other
        # execution facts, but may not replace the verified SDPA transition.
        checkpoint_execution_provenance["sdpa_backend"] = dict(resources.sdpa_backend_transition)
        recorded_sdpa = resources.sdpa_backend.to_mapping()
        validate_sdpa_transition_mapping(
            checkpoint_execution_provenance["sdpa_backend"],
            expected_current=recorded_sdpa,
        )
        metadata = {
            "format": _CHECKPOINT_FORMAT,
            "checkpoint_id": publisher.checkpoint_id,
            "checkpoint_role": checkpoint_role,
            "contract": contract_metadata(),
            "provenance": provenance,
            "training_state": asdict(state),
            "training_schedule_state": schedule_state.to_mapping(),
            "actor_supervisor_state": actor_supervisor_state.to_mapping(),
            "training_config": config.to_mapping(),
            "lineage_config": config.lineage_mapping(),
            "model_config": asdict(config.model.to_model_config()),
            "encoding_contract": grounding_encoding_identity(),
            "model_state_spec": _tensor_spec(network_state),
            "actor_model_state_spec": _tensor_spec(actor_network_state),
            "optimizer_spec": _optimizer_spec(resources.optimizer, optimizer_state),
            "queue_spec": _queue_spec(queue_payload),
            "decision_semantics_abi": _decision_semantics_abi(),
            "failure_credit_abi": _failure_credit_abi(),
            "failure_credit_mode": config.failure_credit.mode,
            "liveness_cost_heads_enabled": (config.failure_credit.learning_enabled),
            "failure_credit_replay_enabled": (config.failure_credit.learning_enabled),
            "failure_credit_replay_spec": failure_credit_replay_spec,
            "long_horizon_value_head_abi": _LONG_HORIZON_VALUE_HEAD_ABI,
            "episodic_target_abi": _EPISODIC_TARGET_ABI,
            "sdpa_backend": recorded_sdpa,
            "execution_provenance": checkpoint_execution_provenance,
            "transaction_heads_enabled": config.transaction_learning.enabled,
            "transaction_lifecycle_abi": (
                TRANSACTION_LIFECYCLE_VERSION if config.transaction_learning.enabled else None
            ),
            "transaction_replay_spec": (
                _transaction_replay_spec(transaction_replay_payload) if transaction_replay_payload is not None else None
            ),
            "episodic_replay_enabled": config.episodic_learning.enabled,
            "episodic_replay_spec": episodic_replay_spec,
            "resolved_device": str(resources.device),
            "resolved_collector_device": str(next(resources.collector_model.parameters()).device),
            "total_steps": state.environment_steps,
        }
        if evaluation_state is not None:
            metadata["evaluation_state"] = evaluation_state.to_mapping()
        (staging / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return publisher.commit()
    except BaseException:
        publisher.abort()
        raise


def load_training_checkpoint(
    checkpoint: str | Path,
    *,
    config: TrainingConfig,
    resources: TrainingResources,
    prevalidated: ValidatedResumeCheckpoint | None = None,
) -> TrainingState:
    """Restore an exact continuation only after every payload probes cleanly.

    In particular, optional replay corpora are deserialized and loaded into
    fresh bounded replay instances before model, optimizer, queue, collector,
    or process RNG state is touched.  A missing, corrupt, over-capacity, or
    otherwise incompatible sidecar therefore cannot leave a partially restored
    live learner.

    ``prevalidated`` may carry the handle returned by
    :func:`preflight_training_checkpoint` earlier in this same process so the
    immutable atomic directory is not byte-hashed a second time; every
    semantic identity check still runs in full, and an independent call
    (``prevalidated=None``) performs its own complete validation.
    """

    if config.transaction_learning.enabled and resources.transaction_replay is None:
        raise RuntimeError("transaction-enabled resources have no replay sidecar")
    if config.episodic_learning.enabled and resources.episodic_replay is None:
        raise RuntimeError("episodic-learning resources have no replay sidecar")
    if config.failure_credit.learning_enabled and resources.failure_credit_replay is None:
        raise RuntimeError("failure-credit learning resources have no replay-v5 sidecar")

    validated = preflight_training_checkpoint(
        checkpoint,
        config=config,
        resolved_device=str(resources.device),
        resolved_collector_device=str(next(resources.collector_model.parameters()).device),
        prevalidated=prevalidated,
    )
    network_state = torch.load(
        validated.root / "network.pt",
        map_location=resources.device,
        weights_only=True,
    )
    actor_network_state = torch.load(
        validated.root / "actor_network.pt",
        map_location=next(resources.collector_model.parameters()).device,
        weights_only=True,
    )
    optimizer_state = torch.load(
        validated.root / "optimizer.pt",
        map_location=resources.device,
        weights_only=True,
    )
    with (validated.root / "rollout_queue.pkl").open("rb") as handle:
        queue_payload = pickle.load(handle)
    with (validated.root / "stochastic_state.pkl").open("rb") as handle:
        stochastic = pickle.load(handle)

    transaction_replay_payload = None
    if config.transaction_learning.enabled:
        with (validated.root / "transaction_replay.pkl").open("rb") as handle:
            transaction_replay_payload = pickle.load(handle)
    episodic_replay_payload = None
    if config.episodic_learning.enabled:
        with (validated.root / "episodic_replay.pkl").open("rb") as handle:
            episodic_replay_payload = pickle.load(handle)
    failure_credit_replay_payload = None
    if config.failure_credit.learning_enabled:
        with (validated.root / "failure_credit_replay.pkl").open("rb") as handle:
            failure_credit_replay_payload = pickle.load(handle)

    base_payloads = (
        network_state,
        actor_network_state,
        optimizer_state,
        queue_payload,
    )
    if not all(isinstance(item, dict) for item in base_payloads):
        raise ValueError("checkpoint payload types are invalid")
    if queue_payload.get("version") != _QUEUE_PAYLOAD_VERSION:
        raise ValueError("unsupported rollout queue checkpoint payload")
    if queue_payload.get("capacity") != config.rollout.queue_capacity:
        raise ValueError("checkpoint rollout queue capacity differs")
    items = queue_payload.get("items")
    _validate_unrolls(items, config=config)
    if validated.metadata.get("queue_spec") != _queue_spec(queue_payload):
        raise ValueError("checkpoint rollout queue metadata differs from payload")

    stochastic = _validate_stochastic_state(stochastic)
    resources.learner.validate_dynamics_state_dict(stochastic["learner_dynamics"])
    if config.transaction_learning.enabled:
        transaction_replay_payload = _validated_transaction_replay_payload(
            transaction_replay_payload,
            config=config,
        )
        if validated.metadata.get("transaction_replay_spec") != _transaction_replay_spec(transaction_replay_payload):
            raise ValueError("checkpoint transaction replay metadata differs from payload")
    if config.episodic_learning.enabled:
        episodic_replay_payload, episodic_replay_spec = _validated_episodic_replay_payload(
            episodic_replay_payload,
            config=config,
        )
        if validated.metadata.get("episodic_replay_spec") != episodic_replay_spec:
            raise ValueError("checkpoint episodic replay metadata differs from payload")
    if config.failure_credit.learning_enabled:
        failure_credit_replay_payload, failure_credit_replay_spec = _validated_failure_credit_replay_payload(
            failure_credit_replay_payload,
            config=config,
        )
        if validated.metadata.get("failure_credit_replay_spec") != failure_credit_replay_spec:
            raise ValueError("checkpoint failure-credit replay metadata differs from payload")
    state = training_state_from_metadata(validated.metadata)

    # Probe all mutable Torch payloads against independent objects.  Nothing
    # below this block has touched the live TrainingResources instance.
    temporary_model = deepcopy(resources.model)
    temporary_model.load_state_dict(network_state, strict=True)
    temporary_actor = deepcopy(resources.collector_model)
    temporary_actor.load_state_dict(actor_network_state, strict=True)
    temporary_optimizer = torch.optim.AdamW(
        temporary_model.parameters(),
        lr=config.optimization.learning_rate,
        weight_decay=config.optimization.weight_decay,
    )
    temporary_optimizer.load_state_dict(optimizer_state)
    if validated.metadata.get("model_state_spec") != _tensor_spec(network_state):
        raise ValueError("checkpoint learner tensor specification differs")
    if validated.metadata.get("actor_model_state_spec") != _tensor_spec(actor_network_state):
        raise ValueError("checkpoint actor tensor specification differs")
    if validated.metadata.get("optimizer_spec") != _optimizer_spec(
        temporary_optimizer,
        optimizer_state,
    ):
        raise ValueError("checkpoint optimizer specification differs")

    # Queue restore has a live-target precondition independent of checkpoint
    # payload validity.  Check it before model/optimizer/replay/RNG mutation so
    # the public loader cannot fail halfway through an otherwise valid restore.
    resources.rollout_queue.validate_restore_ready()

    resources.model.load_state_dict(network_state, strict=True)
    resources.collector_model.load_state_dict(actor_network_state, strict=True)
    resources.optimizer.load_state_dict(optimizer_state)
    resources.rollout_queue.restore(items)
    if transaction_replay_payload is not None:
        assert resources.transaction_replay is not None  # precondition above
        resources.transaction_replay.load_state_dict(transaction_replay_payload)
    if episodic_replay_payload is not None:
        assert resources.episodic_replay is not None  # precondition above
        resources.episodic_replay.load_state_dict(episodic_replay_payload)
    if failure_credit_replay_payload is not None:
        assert resources.failure_credit_replay is not None  # precondition above
        resources.failure_credit_replay.load_state_dict(failure_credit_replay_payload)
    _restore_stochastic_state(stochastic, resources=resources)
    return state


def initialize_model_from_checkpoint(
    checkpoint: str | Path,
    *,
    config: TrainingConfig,
    resources: TrainingResources,
    prevalidated: ValidatedResumeCheckpoint | None = None,
) -> Path:
    """Migrate only network parameters into a fresh training lineage.

    This is deliberately not exact resume: optimizer moments, queued unrolls,
    transaction/episodic replay, collector/RNG state, environment counters,
    and policy-version counters are left at their newly constructed values.
    Compatible learned tensors are inherited into a new lineage. The six
    long-horizon heads and the two-part liveness-cost head group can remain
    freshly initialized only through their explicit, all-or-none migration
    gates. Exact resume never uses these gates.

    ``prevalidated`` may carry the handle returned by
    :func:`preflight_model_initialization` earlier in this same process so
    the immutable atomic directory is not byte-hashed a second time; every
    semantic identity check still runs in full, and an independent call
    (``prevalidated=None``) performs its own complete validation.
    """

    validated = preflight_model_initialization(
        checkpoint,
        config=config,
        prevalidated=prevalidated,
    )
    state = torch.load(
        validated.root / "network.pt",
        map_location=resources.device,
        weights_only=True,
    )
    if not isinstance(state, dict):
        raise ValueError("checkpoint network payload must be an object")
    if validated.metadata.get("model_state_spec") != _tensor_spec(state):
        raise ValueError("model initialization source tensor specification differs")
    target_state = resources.model.state_dict()
    migrated = _model_parameter_initialization_state(
        state,
        target_state=target_state,
        allow_missing_transaction_heads=config.transaction_learning.enabled,
        allow_source_transaction_head_drop=(not config.transaction_learning.enabled),
        allow_missing_long_horizon_heads=(validated.metadata.get("format") in _LONG_HORIZON_MISSING_FORMATS),
        allow_missing_liveness_heads=config.failure_credit.learning_enabled,
    )
    # Preserve the freshly constructed target-only parameters and RNG lineage;
    # the migration overlays compatible learned tensors onto that exact model
    # rather than constructing a second set of randomly initialized heads.
    # No probe dry-run is needed here: ``migrated`` starts as an exact copy of
    # the live model's own state dict, every overlaid tensor was validated
    # above by exact name, shape, and dtype against that state dict, and the
    # key sets are therefore identical, so the strict load below cannot fail
    # or partially mutate the model.
    resources.model.load_state_dict(migrated, strict=True)
    resources.publish_collector_policy()
    return validated.root


def _model_parameter_initialization_state(
    source_state: object,
    *,
    target_state: dict[str, Any],
    allow_missing_transaction_heads: bool,
    allow_source_transaction_head_drop: bool = False,
    allow_missing_long_horizon_heads: bool = False,
    allow_missing_liveness_heads: bool = False,
) -> dict[str, Any]:
    """Fail-closed overlay used only by explicit parameter initialization.

    Exact resume never calls this path. Optional transaction heads, the six
    long-horizon state heads, and the two-part liveness-cost head family are
    independent all-or-none migration groups. Only a validated older-format
    checkpoint may omit long-horizon heads. Liveness heads may be omitted only
    when the target explicitly enables the new failure-credit learner. A
    target that explicitly retires transaction-v3 may drop its complete
    source-only three-head family, but a partial or unknown source family is
    rejected. Every shared tensor must still match by exact name, shape, and
    dtype. The caller keeps the target model's fresh parameters for omitted
    groups and does not import optimizer, queue, replay, RNG, or training
    counters.
    """

    if not isinstance(source_state, dict):
        raise ValueError("checkpoint network payload must be an object")
    if not all(isinstance(key, str) for key in source_state):
        raise TypeError("model initialization source must contain named tensors")
    source_transaction_heads = {key for key in source_state if key.startswith(_TRANSACTION_HEAD_PREFIXES)}
    target_transaction_heads = {key for key in target_state if key.startswith(_TRANSACTION_HEAD_PREFIXES)}
    if target_transaction_heads:
        target_transaction_suffixes = {
            prefix: {key.removeprefix(prefix) for key in target_transaction_heads if key.startswith(prefix)}
            for prefix in _TRANSACTION_HEAD_PREFIXES
        }
        if not all(suffixes == _TRANSACTION_HEAD_SUFFIXES for suffixes in target_transaction_suffixes.values()):
            raise ValueError("model initialization target has an incomplete " "transaction-head family")
    permitted_source_only: set[str] = set()
    if source_transaction_heads and not target_transaction_heads:
        source_transaction_suffixes = {
            prefix: {key.removeprefix(prefix) for key in source_transaction_heads if key.startswith(prefix)}
            for prefix in _TRANSACTION_HEAD_PREFIXES
        }
        complete_source_only_group = all(
            suffixes == _TRANSACTION_HEAD_SUFFIXES for suffixes in source_transaction_suffixes.values()
        )
        if not allow_source_transaction_head_drop or not complete_source_only_group:
            raise ValueError(
                "model initialization source must contain either all or none "
                "of the source-only transaction-head tensors"
            )
        permitted_source_only.update(source_transaction_heads)

    unexpected = sorted(set(source_state) - set(target_state) - permitted_source_only)
    if unexpected:
        raise ValueError("model initialization source contains unsupported tensors: " + ", ".join(unexpected))
    target_long_horizon_heads = {key for key in target_state if key.startswith(_LONG_HORIZON_HEAD_PREFIXES)}
    target_liveness_heads = {key for key in target_state if key.startswith(_LIVENESS_HEAD_PREFIXES)}
    if not target_long_horizon_heads:
        raise ValueError("model initialization target has no long-horizon head tensors")
    target_liveness_prefixes = {
        prefix for prefix in _LIVENESS_HEAD_PREFIXES if any(key.startswith(prefix) for key in target_liveness_heads)
    }
    if target_liveness_heads and target_liveness_prefixes != set(_LIVENESS_HEAD_PREFIXES):
        raise ValueError("model initialization target has an incomplete liveness-head family")
    if allow_missing_liveness_heads and not target_liveness_heads:
        raise ValueError("liveness-head migration was enabled but the target has no liveness tensors")

    missing = set(target_state) - set(source_state)
    permitted_missing: set[str] = set()

    missing_transaction = missing & target_transaction_heads
    if missing_transaction:
        if not allow_missing_transaction_heads or missing_transaction != target_transaction_heads:
            raise ValueError(
                "model initialization source must contain either all or none " "of the transaction-head tensors"
            )
        permitted_missing.update(target_transaction_heads)

    missing_long_horizon = missing & target_long_horizon_heads
    if missing_long_horizon:
        if not allow_missing_long_horizon_heads or missing_long_horizon != target_long_horizon_heads:
            raise ValueError(
                "model initialization source must contain either all or none " "of the six long-horizon head prefixes"
            )
        permitted_missing.update(target_long_horizon_heads)

    missing_liveness = missing & target_liveness_heads
    if missing_liveness:
        if not allow_missing_liveness_heads or missing_liveness != target_liveness_heads:
            raise ValueError(
                "model initialization source must contain either all or none " "of the liveness-head tensors"
            )
        permitted_missing.update(target_liveness_heads)

    shared_missing = sorted(missing - permitted_missing)
    if shared_missing:
        raise ValueError("model initialization source is missing shared tensors: " + ", ".join(shared_missing))

    migrated = dict(target_state)
    for key, value in source_state.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError("model initialization source must contain named tensors")
        if key in permitted_source_only:
            continue
        target = target_state[key]
        if not isinstance(target, torch.Tensor):
            raise TypeError("model initialization target must contain named tensors")
        if value.shape != target.shape or value.dtype != target.dtype:
            raise ValueError(f"model initialization tensor ABI differs for {key}")
        migrated[key] = value
    return migrated


def checkpoint_summary(checkpoint: str | Path) -> dict[str, Any]:
    validated = validate_resume_checkpoint(checkpoint)
    evaluation_state = evaluation_gate_state_from_metadata(validated.metadata)
    return {
        "root": str(validated.root),
        "checkpoint_id": validated.manifest.get("checkpoint_id"),
        "training_state": validated.metadata.get("training_state"),
        "training_schedule_state": training_schedule_state_from_metadata(validated.metadata).to_mapping(),
        "actor_supervisor_state": actor_supervisor_state_from_metadata(validated.metadata).to_mapping(),
        "evaluation_state": (evaluation_state.to_mapping() if evaluation_state is not None else None),
        "format": validated.metadata.get("format"),
        "queue_spec": validated.metadata.get("queue_spec"),
    }


__all__ = [
    "ActorSupervisorState",
    "EvaluationGateState",
    "TrainingScheduleState",
    "TrainingState",
    "actor_supervisor_state_from_metadata",
    "checkpoint_summary",
    "evaluation_gate_state_from_metadata",
    "initialize_model_from_checkpoint",
    "load_training_checkpoint",
    "preflight_model_initialization",
    "preflight_training_checkpoint",
    "save_training_checkpoint",
    "training_schedule_state_from_metadata",
    "training_state_from_metadata",
]
