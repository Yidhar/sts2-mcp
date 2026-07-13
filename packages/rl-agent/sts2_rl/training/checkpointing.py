"""Atomic checkpoint save/resume for the new baseline."""

from __future__ import annotations

import json
import pickle
import random
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from sts2_baseline import StratifiedReplayBuffer
from sts2_rl.checkpoints import (
    AtomicCheckpointDirectory,
    ValidatedResumeCheckpoint,
    build_checkpoint_provenance,
    contract_metadata,
    validate_resume_checkpoint,
)
from sts2_rl.encoding import grounding_encoding_identity

from .config import TrainingConfig
from .experience import DecisionExperience
from .factory import TrainingResources
from .seeding import SIGNED_INT32_MAX

_CHECKPOINT_FORMAT = "sts2-grounded-baseline-checkpoint-v2"


@dataclass(frozen=True, slots=True)
class TrainingState:
    environment_steps: int = 0
    learner_updates: int = 0
    episodes: int = 0
    evaluation_episodes: int = 0
    update_credit: int = 0

    def __post_init__(self) -> None:
        for name in (
            "environment_steps",
            "learner_updates",
            "episodes",
            "evaluation_episodes",
            "update_credit",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")


def _lineage_config(config: TrainingConfig) -> dict[str, Any]:
    return config.lineage_mapping()


def _tensor_spec(state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for key, value in state.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            raise TypeError("model state must contain only named tensors")
        output[key] = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    return output


def _optimizer_spec(
    optimizer: torch.optim.Optimizer,
    state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = state or optimizer.state_dict()
    groups = payload.get("param_groups")
    raw_state = payload.get("state")
    if not isinstance(groups, list) or not isinstance(raw_state, dict):
        raise TypeError("optimizer state must expose state and param_groups")
    group_sizes: list[int] = []
    group_keys: list[list[str]] = []
    for group in groups:
        if not isinstance(group, dict) or not isinstance(group.get("params"), list):
            raise TypeError("optimizer param group is malformed")
        group_sizes.append(len(group["params"]))
        group_keys.append(sorted(str(key) for key in group if key != "params"))
    return {
        "class": f"{type(optimizer).__module__}.{type(optimizer).__qualname__}",
        "group_sizes": group_sizes,
        "group_keys": group_keys,
        "state_entries": len(raw_state),
    }


def _replay_spec(
    replay: StratifiedReplayBuffer,
    *,
    config: TrainingConfig,
) -> dict[str, Any]:
    _validate_replay_samples(replay, config=config)
    indices = replay.indices
    return {
        "version": "sts2-grounded-replay-pickle-v2",
        "size": len(replay),
        "capacity": replay.capacity,
        "recent_window": replay.recent_window,
        "alpha": replay.alpha,
        "beta": replay.beta,
        "priority_epsilon": replay.priority_epsilon,
        "mix": replay.mix.as_dict(),
        "min_index": min(indices) if indices else None,
        "max_index": max(indices) if indices else None,
    }


def _validate_replay_samples(
    replay: StratifiedReplayBuffer,
    *,
    config: TrainingConfig,
) -> None:
    for position, sample in enumerate(replay.samples):
        if not isinstance(sample.payload, DecisionExperience):
            raise TypeError(
                f"checkpoint replay sample {position} payload is not DecisionExperience"
            )
        try:
            sample.payload.validate()
            sample.payload.encoded_snapshot.validate(
                expected_config=config.model.to_encoding_config(),
                expected_fingerprint=grounding_encoding_identity()[
                    "fingerprint_sha256"
                ],
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"checkpoint replay sample {position} encoded payload is invalid: {exc}"
            ) from exc


def _validate_metadata(
    validated: ValidatedResumeCheckpoint,
    *,
    config: TrainingConfig,
    resolved_device: str | None,
    model_only: bool,
) -> None:
    metadata = validated.metadata
    if metadata.get("format") != _CHECKPOINT_FORMAT:
        raise ValueError(f"unsupported grounded checkpoint format: {metadata.get('format')!r}")
    expected_model = asdict(config.model.to_model_config())
    if metadata.get("model_config") != expected_model:
        raise ValueError("checkpoint grounded model config does not match")
    if metadata.get("encoding_contract") != grounding_encoding_identity():
        raise ValueError("checkpoint grounded encoding contract does not match")
    if not isinstance(metadata.get("model_state_spec"), dict):
        raise ValueError("checkpoint has no model tensor specification")
    if model_only:
        return
    training_state_from_metadata(metadata)
    if not isinstance(metadata.get("optimizer_spec"), dict):
        raise ValueError("checkpoint has no optimizer specification")
    if not isinstance(metadata.get("replay_spec"), dict):
        raise ValueError("checkpoint has no replay specification")
    if metadata.get("lineage_config") != _lineage_config(config):
        raise ValueError(
            "exact resume requires an identical immutable lineage config; only "
            "execution budgets/output schedules may change"
        )
    if resolved_device is not None and metadata.get("resolved_device") != resolved_device:
        raise ValueError(
            "exact resume requires the same resolved device: "
            f"checkpoint={metadata.get('resolved_device')!r} runtime={resolved_device!r}"
        )


def preflight_training_checkpoint(
    checkpoint: str | Path,
    *,
    config: TrainingConfig,
    resolved_device: str,
) -> ValidatedResumeCheckpoint:
    """Read/hash/validate resume metadata before any backend or artifact side effect."""

    validated = validate_resume_checkpoint(checkpoint)
    _validate_metadata(
        validated,
        config=config,
        resolved_device=resolved_device,
        model_only=False,
    )
    return validated


def preflight_model_initialization(
    checkpoint: str | Path,
    *,
    config: TrainingConfig,
) -> ValidatedResumeCheckpoint:
    """Validate a same-architecture new-lineage model source without loading it."""

    validated = validate_resume_checkpoint(checkpoint)
    _validate_metadata(
        validated,
        config=config,
        resolved_device=None,
        model_only=True,
    )
    return validated


def _stochastic_state(resources: TrainingResources) -> dict[str, Any]:
    cuda_states = (
        torch.cuda.get_rng_state_all()
        if resources.device.type == "cuda" and torch.cuda.is_available()
        else []
    )
    return {
        "version": "sts2-grounded-stochastic-state-v1",
        "python_random": random.getstate(),
        "numpy_random": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": cuda_states,
        "collector": resources.collector.state_dict(),
    }


def _validate_stochastic_state(
    payload: object,
    *,
    resources: TrainingResources,
) -> dict[str, Any]:
    """Validate continuation RNG payloads without mutating live resources."""

    if not isinstance(payload, dict):
        raise TypeError("checkpoint stochastic state must be a mapping")
    expected_keys = {
        "version",
        "python_random",
        "numpy_random",
        "torch_cpu",
        "torch_cuda",
        "collector",
    }
    actual_keys = set(payload)
    if actual_keys != expected_keys:
        raise ValueError(
            "checkpoint stochastic state keys mismatch: "
            f"missing={sorted(expected_keys - actual_keys)} "
            f"unknown={sorted(actual_keys - expected_keys)}"
        )
    if payload["version"] != "sts2-grounded-stochastic-state-v1":
        raise ValueError("unsupported checkpoint stochastic state")

    python_probe = random.Random()
    # Exact continuation must validate the legacy global NumPy RNG tuple that
    # ``np.random.get_state``/``set_state`` own.
    numpy_probe = np.random.RandomState()
    try:
        python_probe.setstate(payload["python_random"])
        numpy_probe.set_state(payload["numpy_random"])
    except (TypeError, ValueError) as exc:
        raise ValueError("checkpoint Python/NumPy RNG state is invalid") from exc

    torch_cpu = payload["torch_cpu"]
    if not isinstance(torch_cpu, torch.Tensor):
        raise TypeError("checkpoint torch CPU RNG state must be a tensor")
    try:
        torch.Generator(device="cpu").set_state(torch_cpu.cpu())
    except RuntimeError as exc:
        raise ValueError("checkpoint torch CPU RNG state is invalid") from exc

    torch_cuda = payload["torch_cuda"]
    if not isinstance(torch_cuda, list) or not all(
        isinstance(item, torch.Tensor) for item in torch_cuda
    ):
        raise TypeError("checkpoint torch CUDA RNG states must be a tensor list")
    if torch_cuda:
        if not torch.cuda.is_available():
            raise ValueError("checkpoint requires CUDA RNG state but CUDA is unavailable")
        if len(torch_cuda) != torch.cuda.device_count():
            raise ValueError(
                "checkpoint CUDA device count differs from the current runtime: "
                f"checkpoint={len(torch_cuda)} runtime={torch.cuda.device_count()}"
            )
        for index, state in enumerate(torch_cuda):
            try:
                torch.Generator(device=f"cuda:{index}").set_state(state.cpu())
            except RuntimeError as exc:
                raise ValueError(
                    f"checkpoint torch CUDA RNG state {index} is invalid"
                ) from exc

    collector = payload["collector"]
    if not isinstance(collector, dict):
        raise TypeError("checkpoint collector state must be a mapping")
    expected_collector_keys = {"version", "episode_seed", "rng_state"}
    if set(collector) != expected_collector_keys:
        raise ValueError("checkpoint collector state keys mismatch")
    if collector["version"] != "sts2-grounded-collector-state-v2":
        raise ValueError("unsupported checkpoint collector state")
    episode_seed = collector["episode_seed"]
    if isinstance(episode_seed, bool) or not isinstance(episode_seed, int):
        raise TypeError("checkpoint collector episode_seed must be an integer")
    if episode_seed < 0 or episode_seed > SIGNED_INT32_MAX or episode_seed % 2 != 0:
        raise ValueError("checkpoint collector episode_seed must be even signed 32-bit")
    rng_state = collector["rng_state"]
    if not isinstance(rng_state, dict):
        raise TypeError("checkpoint collector rng_state must be a mapping")
    generator_probe = np.random.default_rng()
    try:
        generator_probe.bit_generator.state = dict(rng_state)
    except (TypeError, ValueError) as exc:
        raise ValueError("checkpoint collector RNG state is invalid") from exc
    return payload


def training_state_from_metadata(metadata: dict[str, Any]) -> TrainingState:
    raw_state = metadata.get("training_state")
    if not isinstance(raw_state, dict):
        raise ValueError("checkpoint has no training_state")
    required = {
        "environment_steps",
        "learner_updates",
        "episodes",
        "evaluation_episodes",
        "update_credit",
    }
    actual = set(raw_state)
    if actual != required:
        raise ValueError(
            "checkpoint training_state keys mismatch: "
            f"missing={sorted(required - actual)} unknown={sorted(actual - required)}"
        )
    for key, value in raw_state.items():
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"checkpoint training counter {key!r} must be an integer")
    return TrainingState(**raw_state)


def _validate_replay_payload(
    replay: object,
    *,
    config: TrainingConfig,
    expected_spec: object,
) -> StratifiedReplayBuffer:
    if not isinstance(replay, StratifiedReplayBuffer):
        raise TypeError("checkpoint replay payload has the wrong type")
    if replay.capacity != config.replay.capacity:
        raise ValueError("checkpoint replay capacity does not match config")
    if replay.recent_window != config.replay.recent_window:
        raise ValueError("checkpoint replay recent_window does not match config")
    if len(replay) > replay.capacity:
        raise ValueError("checkpoint replay exceeds its declared capacity")
    indices = replay.indices
    if len(indices) != len(set(indices)) or tuple(sorted(indices)) != indices:
        raise ValueError("checkpoint replay indices must be unique and increasing")
    actual_spec = _replay_spec(replay, config=config)
    if expected_spec != actual_spec:
        raise ValueError("checkpoint replay metadata does not match replay payload")
    if indices:
        probabilities = replay.mixture_probabilities()
        if set(probabilities) != set(indices):
            raise ValueError("checkpoint replay probability/index sets differ")
        values = np.asarray(list(probabilities.values()), dtype=np.float64)
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("checkpoint replay probabilities must be finite and positive")
        if not np.isclose(values.sum(), 1.0, rtol=1e-9, atol=1e-9):
            raise ValueError("checkpoint replay probabilities must sum to one")
    return replay


def _restore_stochastic_state(
    payload: object,
    *,
    resources: TrainingResources,
) -> None:
    validated = _validate_stochastic_state(payload, resources=resources)
    torch_cpu = validated["torch_cpu"]
    torch_cuda = validated["torch_cuda"]
    collector = validated["collector"]
    resources.collector.load_state_dict(collector)
    random.setstate(validated["python_random"])
    np.random.set_state(validated["numpy_random"])
    torch.set_rng_state(torch_cpu.cpu())
    if torch_cuda:
        torch.cuda.set_rng_state_all([item.cpu() for item in torch_cuda])


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
) -> Path:
    provenance = build_checkpoint_provenance(
        parent_checkpoint=parent_checkpoint,
        experiment_run_id=run_id,
        config_version=config.version,
        config_profile=config.profile,
        checkpoint_load_mode=(
            checkpoint_load_mode
            or "fresh"
        ),
        parent_relation=(
            parent_relation
            if parent_checkpoint is not None
            else None
        ) or ("unspecified_parent" if parent_checkpoint is not None else None),
    )
    publisher = AtomicCheckpointDirectory(target, provenance=provenance)
    staging = publisher.prepare()
    try:
        network_state = resources.model.state_dict()
        optimizer_state = resources.optimizer.state_dict()
        torch.save(network_state, staging / "network.pt")
        torch.save(optimizer_state, staging / "optimizer.pt")
        with (staging / "replay_buffer.pkl").open("wb") as handle:
            pickle.dump(resources.replay, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with (staging / "stochastic_state.pkl").open("wb") as handle:
            pickle.dump(
                _stochastic_state(resources),
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        metadata = {
            "format": _CHECKPOINT_FORMAT,
            "checkpoint_id": publisher.checkpoint_id,
            "contract": contract_metadata(),
            "provenance": provenance,
            "training_state": asdict(state),
            "training_config": config.to_mapping(),
            "lineage_config": _lineage_config(config),
            "model_config": asdict(config.model.to_model_config()),
            "encoding_contract": grounding_encoding_identity(),
            "model_state_spec": _tensor_spec(network_state),
            "optimizer_spec": _optimizer_spec(resources.optimizer, optimizer_state),
            "replay_spec": _replay_spec(resources.replay, config=config),
            "resolved_device": str(resources.device),
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
    """Transactionally restore a prevalidated exact-continuation checkpoint."""

    validated = preflight_training_checkpoint(
        checkpoint,
        config=config,
        resolved_device=str(resources.device),
    )

    network_state = torch.load(
        validated.root / "network.pt",
        map_location=resources.device,
        weights_only=True,
    )
    optimizer_state = torch.load(
        validated.root / "optimizer.pt",
        map_location=resources.device,
        weights_only=True,
    )
    if not isinstance(network_state, dict) or not isinstance(optimizer_state, dict):
        raise ValueError("checkpoint model/optimizer payload must be mappings")
    with (validated.root / "replay_buffer.pkl").open("rb") as handle:
        # The complete file set, hashes and semantic identities were validated
        # before this trusted local checkpoint payload is deserialized.
        replay = pickle.load(handle)
    with (validated.root / "stochastic_state.pkl").open("rb") as handle:
        stochastic_state = pickle.load(handle)

    state = training_state_from_metadata(validated.metadata)
    replay = _validate_replay_payload(
        replay,
        config=config,
        expected_spec=validated.metadata.get("replay_spec"),
    )
    _validate_stochastic_state(stochastic_state, resources=resources)
    if validated.metadata.get("model_state_spec") != _tensor_spec(network_state):
        raise ValueError("checkpoint model tensor spec does not match network payload")

    # Load into disposable objects first. This validates state keys/shapes/dtypes
    # and optimizer parameter-group compatibility without touching live state.
    temporary_model = deepcopy(resources.model)
    temporary_model.load_state_dict(network_state, strict=True)
    temporary_optimizer = torch.optim.AdamW(
        temporary_model.parameters(),
        lr=config.optimization.learning_rate,
        weight_decay=config.optimization.weight_decay,
    )
    temporary_optimizer.load_state_dict(optimizer_state)
    actual_optimizer_spec = _optimizer_spec(
        temporary_optimizer,
        optimizer_state,
    )
    if validated.metadata.get("optimizer_spec") != actual_optimizer_spec:
        raise ValueError("checkpoint optimizer metadata does not match payload")
    del temporary_optimizer, temporary_model

    # Commit only after every byte, schema and disposable state load succeeded.
    resources.model.load_state_dict(network_state, strict=True)
    resources.optimizer.load_state_dict(optimizer_state)
    resources.replay = replay
    _restore_stochastic_state(stochastic_state, resources=resources)
    return state


def initialize_model_from_checkpoint(
    checkpoint: str | Path,
    *,
    config: TrainingConfig,
    resources: TrainingResources,
) -> Path:
    """Start a new horizon/profile from a complete RL 0.3 model state.

    This is intentionally not a legacy or partial-key migrator. The source must
    be a fully valid grounded-baseline checkpoint and the model config must be
    identical. Optimizer, replay and counters remain freshly initialized.
    """

    validated = preflight_model_initialization(checkpoint, config=config)
    network_state = torch.load(
        validated.root / "network.pt",
        map_location=resources.device,
        weights_only=True,
    )
    if not isinstance(network_state, dict):
        raise ValueError("checkpoint network payload must be a mapping")
    if validated.metadata.get("model_state_spec") != _tensor_spec(network_state):
        raise ValueError("checkpoint model tensor spec does not match network payload")
    temporary_model = deepcopy(resources.model)
    temporary_model.load_state_dict(network_state, strict=True)
    del temporary_model
    resources.model.load_state_dict(network_state, strict=True)
    return validated.root


def checkpoint_summary(checkpoint: str | Path) -> dict[str, Any]:
    validated = validate_resume_checkpoint(checkpoint)
    return {
        "root": str(validated.root),
        "checkpoint_id": validated.manifest.get("checkpoint_id"),
        "training_state": validated.metadata.get("training_state"),
        "format": validated.metadata.get("format"),
    }


__all__ = [
    "TrainingState",
    "checkpoint_summary",
    "initialize_model_from_checkpoint",
    "load_training_checkpoint",
    "preflight_model_initialization",
    "preflight_training_checkpoint",
    "save_training_checkpoint",
    "training_state_from_metadata",
]
