"""Atomic, fail-closed checkpoint ABI for recurrent V-trace v2."""

from __future__ import annotations

import json
import pickle
import random
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from sts2_baseline import SequenceUnroll
from sts2_rl.checkpoints import (
    AtomicCheckpointDirectory,
    ValidatedResumeCheckpoint,
    build_checkpoint_provenance,
    contract_metadata,
    validate_model_initialization_checkpoint,
    validate_resume_checkpoint,
)
from sts2_rl.encoding import grounding_encoding_identity

from .config import TrainingConfig
from .factory import TrainingResources
from .seeding import SIGNED_INT32_MAX
from .transaction import BoundedTransactionReplay

_CHECKPOINT_FORMAT = "sts2-recurrent-vtrace-checkpoint-v3"
_QUEUE_PAYLOAD_VERSION = "sts2-rollout-queue-pickle-v2"
_ACTOR_SUPERVISOR_STATE_VERSION = "sts2-actor-supervisor-state-v1"


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

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.policy_version < self.actor_policy_version:
            raise ValueError("actor policy version cannot be newer than learner policy")


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
            if (
                not isinstance(fingerprint, str)
                or not fingerprint
                or fingerprint.strip() != fingerprint
            ):
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
                raise ValueError(
                    "recent actor incident attempts must be strictly increasing valid attempts"
                )
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
    if not isinstance(payload, dict):
        raise TypeError("transaction replay checkpoint must be an object")
    probe = BoundedTransactionReplay(
        capacity=config.transaction_learning.replay_capacity,
        byte_capacity=config.transaction_learning.replay_byte_capacity,
        seed=config.runtime.seed,
    )
    probe.load_state_dict(payload)
    configured_burn_in = config.transaction_learning.burn_in_steps
    for trace in probe.snapshot():
        if trace.burn_in_steps > configured_burn_in:
            raise ValueError("checkpoint transaction trace exceeds configured burn-in")
        if trace.start_step > 0 and trace.burn_in_steps != configured_burn_in:
            raise ValueError("checkpoint non-initial transaction trace lacks configured burn-in")
    return payload


def _stochastic_state(resources: TrainingResources) -> dict[str, Any]:
    cuda_states = (
        torch.cuda.get_rng_state_all() if resources.device.type == "cuda" and torch.cuda.is_available() else []
    )
    return {
        "version": "sts2-recurrent-stochastic-state-v2",
        "python_random": random.getstate(),
        "numpy_random": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": cuda_states,
        "collector": resources.collector.state_dict(),
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
    }
    if set(payload) != expected:
        raise ValueError("checkpoint stochastic state keys mismatch")
    if payload["version"] != "sts2-recurrent-stochastic-state-v2":
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
    return payload


def _restore_stochastic_state(
    payload: dict[str, Any],
    *,
    resources: TrainingResources,
) -> None:
    resources.collector.load_state_dict(payload["collector"])
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
    legacy_missing = {"maximum_observed_candidates"}
    if actual == expected - legacy_missing:
        raw = {**raw, "maximum_observed_candidates": 0}
    elif actual != expected:
        raise ValueError("checkpoint training_state keys mismatch")
    return TrainingState(**raw)


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


def _validate_metadata(
    validated: ValidatedResumeCheckpoint,
    *,
    config: TrainingConfig,
    resolved_device: str | None,
    resolved_collector_device: str | None,
    model_only: bool,
) -> None:
    metadata = validated.metadata
    if metadata.get("format") != _CHECKPOINT_FORMAT:
        raise ValueError(f"unsupported v2 checkpoint format: {metadata.get('format')!r}")
    if metadata.get("model_config") != asdict(config.model.to_model_config()):
        raise ValueError("checkpoint recurrent model config does not match")
    if metadata.get("encoding_contract") != grounding_encoding_identity():
        raise ValueError("checkpoint encoding contract does not match")
    if not isinstance(metadata.get("model_state_spec"), dict):
        raise ValueError("checkpoint has no model tensor specification")
    if model_only:
        return
    training_state_from_metadata(metadata)
    actor_supervisor_state_from_metadata(metadata)
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
        raise ValueError("exact resume requires identical immutable v2 lineage")
    if metadata.get("resolved_device") != resolved_device:
        raise ValueError("exact resume requires the same learner device")
    if metadata.get("resolved_collector_device") != resolved_collector_device:
        raise ValueError("exact resume requires the same actor device")
    if not isinstance(metadata.get("queue_spec"), dict):
        raise ValueError("checkpoint has no rollout queue specification")
    if config.transaction_learning.enabled:
        if metadata.get("transaction_heads_enabled") is not True:
            raise ValueError("transaction-enabled checkpoint has no head ABI marker")
        if not isinstance(metadata.get("transaction_replay_spec"), dict):
            raise ValueError("transaction-enabled checkpoint has no replay specification")


def preflight_training_checkpoint(
    checkpoint: str | Path,
    *,
    config: TrainingConfig,
    resolved_device: str,
    resolved_collector_device: str,
) -> ValidatedResumeCheckpoint:
    validated = validate_resume_checkpoint(checkpoint)
    _validate_metadata(
        validated,
        config=config,
        resolved_device=resolved_device,
        resolved_collector_device=resolved_collector_device,
        model_only=False,
    )
    return validated


def preflight_model_initialization(
    checkpoint: str | Path,
    *,
    config: TrainingConfig,
) -> ValidatedResumeCheckpoint:
    validated = validate_model_initialization_checkpoint(checkpoint)
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
    parent_checkpoint: str | Path | None = None,
    run_id: str | None = None,
    checkpoint_load_mode: str | None = None,
    parent_relation: str | None = None,
    actor_supervisor_state: ActorSupervisorState | None = None,
) -> Path:
    """Publish a checkpoint while the actor is quiescent between episodes."""

    if actor_supervisor_state is None:
        actor_supervisor_state = ActorSupervisorState()
    elif not isinstance(actor_supervisor_state, ActorSupervisorState):
        raise TypeError("actor_supervisor_state must be ActorSupervisorState")
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
        metadata = {
            "format": _CHECKPOINT_FORMAT,
            "checkpoint_id": publisher.checkpoint_id,
            "contract": contract_metadata(),
            "provenance": provenance,
            "training_state": asdict(state),
            "actor_supervisor_state": actor_supervisor_state.to_mapping(),
            "training_config": config.to_mapping(),
            "lineage_config": config.lineage_mapping(),
            "model_config": asdict(config.model.to_model_config()),
            "encoding_contract": grounding_encoding_identity(),
            "model_state_spec": _tensor_spec(network_state),
            "actor_model_state_spec": _tensor_spec(actor_network_state),
            "optimizer_spec": _optimizer_spec(resources.optimizer, optimizer_state),
            "queue_spec": _queue_spec(queue_payload),
            "transaction_heads_enabled": config.transaction_learning.enabled,
            "transaction_replay_spec": (
                _transaction_replay_spec(transaction_replay_payload) if transaction_replay_payload is not None else None
            ),
            "resolved_device": str(resources.device),
            "resolved_collector_device": str(next(resources.collector_model.parameters()).device),
            "total_steps": state.environment_steps,
        }
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
) -> TrainingState:
    validated = preflight_training_checkpoint(
        checkpoint,
        config=config,
        resolved_device=str(resources.device),
        resolved_collector_device=str(next(resources.collector_model.parameters()).device),
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
    if not all(isinstance(item, dict) for item in (network_state, actor_network_state, optimizer_state, queue_payload)):
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
    if config.transaction_learning.enabled:
        transaction_replay_payload = _validated_transaction_replay_payload(
            transaction_replay_payload,
            config=config,
        )
        if validated.metadata.get("transaction_replay_spec") != _transaction_replay_spec(transaction_replay_payload):
            raise ValueError("checkpoint transaction replay metadata differs from payload")
    state = training_state_from_metadata(validated.metadata)

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
    if validated.metadata.get("optimizer_spec") != _optimizer_spec(temporary_optimizer, optimizer_state):
        raise ValueError("checkpoint optimizer specification differs")

    resources.model.load_state_dict(network_state, strict=True)
    resources.collector_model.load_state_dict(actor_network_state, strict=True)
    resources.optimizer.load_state_dict(optimizer_state)
    resources.rollout_queue.restore(items)
    if transaction_replay_payload is not None:
        if resources.transaction_replay is None:
            raise RuntimeError("transaction-enabled resources have no replay sidecar")
        resources.transaction_replay.load_state_dict(transaction_replay_payload)
    _restore_stochastic_state(stochastic, resources=resources)
    return state


def initialize_model_from_checkpoint(
    checkpoint: str | Path,
    *,
    config: TrainingConfig,
    resources: TrainingResources,
) -> Path:
    """Migrate only network parameters into a fresh training lineage.

    This is deliberately not exact resume: optimizer moments, queued unrolls,
    collector/RNG state, environment counters, and policy-version counters are
    left at their newly constructed values.  The source must still match the
    exact learned-parameter and grounded feature ABI.
    """

    validated = preflight_model_initialization(checkpoint, config=config)
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
    )
    # Preserve the freshly constructed target-only parameters and RNG lineage;
    # the migration overlays compatible learned tensors onto that exact model
    # rather than constructing a second set of randomly initialized heads.
    probe = deepcopy(resources.model)
    probe.load_state_dict(migrated, strict=True)
    resources.model.load_state_dict(migrated, strict=True)
    resources.publish_collector_policy()
    return validated.root


def _model_parameter_initialization_state(
    source_state: object,
    *,
    target_state: dict[str, Any],
    allow_missing_transaction_heads: bool,
) -> dict[str, Any]:
    """Fail-closed overlay used only by explicit parameter initialization.

    Exact resume never calls this path.  The only permitted source/target ABI
    difference is the three freshly initialized v11 transaction heads.
    """

    if not isinstance(source_state, dict):
        raise ValueError("checkpoint network payload must be an object")
    if not all(isinstance(key, str) for key in source_state):
        raise TypeError("model initialization source must contain named tensors")
    unexpected = sorted(set(source_state) - set(target_state))
    if unexpected:
        raise ValueError("model initialization source contains unsupported tensors: " + ", ".join(unexpected))
    allowed_prefixes = (
        "candidate_effect_head.",
        "selection_delta_head.",
        "transaction_q_head.",
    )
    missing = set(target_state) - set(source_state)
    target_transaction_heads = {
        key for key in target_state if key.startswith(allowed_prefixes)
    }
    if missing:
        if not allow_missing_transaction_heads or missing != target_transaction_heads:
            shared_missing = sorted(missing - target_transaction_heads)
            if shared_missing:
                raise ValueError(
                    "model initialization source is missing shared tensors: "
                    + ", ".join(shared_missing)
                )
            raise ValueError(
                "model initialization source must contain either all or none "
                "of the transaction-head tensors"
            )
    migrated = dict(target_state)
    for key, value in source_state.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError("model initialization source must contain named tensors")
        target = target_state[key]
        if not isinstance(target, torch.Tensor):
            raise TypeError("model initialization target must contain named tensors")
        if value.shape != target.shape or value.dtype != target.dtype:
            raise ValueError(f"model initialization tensor ABI differs for {key}")
        migrated[key] = value
    return migrated


def checkpoint_summary(checkpoint: str | Path) -> dict[str, Any]:
    validated = validate_resume_checkpoint(checkpoint)
    return {
        "root": str(validated.root),
        "checkpoint_id": validated.manifest.get("checkpoint_id"),
        "training_state": validated.metadata.get("training_state"),
        "actor_supervisor_state": actor_supervisor_state_from_metadata(
            validated.metadata
        ).to_mapping(),
        "format": validated.metadata.get("format"),
        "queue_spec": validated.metadata.get("queue_spec"),
    }


__all__ = [
    "ActorSupervisorState",
    "TrainingState",
    "actor_supervisor_state_from_metadata",
    "checkpoint_summary",
    "initialize_model_from_checkpoint",
    "load_training_checkpoint",
    "preflight_model_initialization",
    "preflight_training_checkpoint",
    "save_training_checkpoint",
    "training_state_from_metadata",
]
