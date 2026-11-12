"""Atomic, cohort-boundary recovery for the synchronous macro-credit course.

Only a fully collected cohort and its completed optimizer update may be
committed. There is no simulator snapshot or asynchronous spool in this format.
An interrupted, uncommitted cohort is collected again from the previous seed
cursor. Checkpoints are trusted local torch files, not untrusted interchange.
"""

from __future__ import annotations

import math
import os
import platform
import random
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer

from sts2_rl.artifacts import resolve_artifact_path, resolve_external_input_path
from sts2_rl.macro_root_evidence import canonical_facts, first_difference

MACRO_CHECKPOINT_FORMAT = "sts2-hybrid-macro-credit-checkpoint-v1"
MACRO_PUBLICATION_FORMAT = "sts2-hybrid-macro-credit-publication-v1"


@dataclass(frozen=True, slots=True)
class MacroCheckpointIdentity:
    course_config: Mapping[str, Any]
    model_config: Mapping[str, Any]
    reward_identity: Mapping[str, Any]
    lineage_id: str
    learner_version: str

    def as_mapping(self) -> dict[str, Any]:
        for name in ("lineage_id", "learner_version"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"macro checkpoint {name} must be non-empty")
        for name in ("course_config", "model_config", "reward_identity"):
            if not isinstance(getattr(self, name), Mapping) or not getattr(self, name):
                raise ValueError(f"macro checkpoint {name} must be a non-empty mapping")
        return {
            "course_config": canonical_facts(self.course_config),
            "model_config": canonical_facts(self.model_config),
            "reward_identity": canonical_facts(self.reward_identity),
            "lineage_id": self.lineage_id,
            "learner_version": self.learner_version,
        }


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"macro checkpoint {name} must be a non-negative integer")
    return int(value)


def _snapshot(value: Any) -> Any:
    """Copy only ordinary optimizer/RNG state; reject unsupported/non-finite data."""
    if isinstance(value, torch.Tensor):
        if value.layout != torch.strided:
            raise ValueError("macro checkpoint only supports dense tensors")
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError("macro checkpoint contains a non-finite tensor")
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in "biuf" or not bool(np.isfinite(value).all()):
            raise ValueError("macro checkpoint contains invalid numpy state")
        return value.copy()
    if isinstance(value, Mapping):
        if any(not isinstance(key, str | int) or isinstance(key, bool) for key in value):
            raise ValueError("macro checkpoint state has an unsupported key")
        return {key: _snapshot(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_snapshot(item) for item in value)
    if isinstance(value, list):
        return [_snapshot(item) for item in value]
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ValueError(f"macro checkpoint contains unsupported/non-finite {type(value).__name__}")


def _runtime_environment(model: nn.Module) -> dict[str, Any]:
    devices = {str(parameter.device) for parameter in model.parameters()}
    devices.update(str(buffer.device) for buffer in model.buffers())
    if len(devices) != 1:
        raise ValueError("macro checkpoint requires a single-device model")
    device = torch.device(next(iter(devices)))
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("macro checkpoint supports only CPU or CUDA/ROCm")
    count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    return {
        "python": platform.python_version(),
        "numpy": str(np.__version__),
        "torch": str(torch.__version__),
        "device": str(device),
        "accelerator_count": count,
        "accelerator_names": [torch.cuda.get_device_name(index) for index in range(count)],
        "cuda_runtime": torch.version.cuda,
        "hip_runtime": torch.version.hip,
    }


def _rng_state(environment: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": [state.cpu().clone() for state in torch.cuda.get_rng_state_all()]
        if int(environment["accelerator_count"]) else [],
    }


def _validate_rng(state: Any, environment: Mapping[str, Any]) -> None:
    if not isinstance(state, Mapping) or set(state) != {"python", "numpy", "torch_cpu", "torch_cuda"}:
        raise ValueError("macro checkpoint RNG fields differ")
    try:
        random.Random().setstate(state["python"])
        np.random.RandomState().set_state(state["numpy"])
        torch.Generator(device="cpu").set_state(state["torch_cpu"])
        cuda_states = state["torch_cuda"]
        if not isinstance(cuda_states, list) or len(cuda_states) != environment["accelerator_count"]:
            raise ValueError("CUDA RNG count differs")
        for tensor in cuda_states:
            if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.uint8 or tensor.ndim != 1:
                raise ValueError("CUDA RNG tensor is malformed")
    except (TypeError, RuntimeError, ValueError, KeyError) as error:
        raise ValueError(f"macro checkpoint RNG state is malformed: {error}") from error


def _restore_rng(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"]:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _optimizer_layout(model: nn.Module, optimizer: Optimizer) -> dict[str, Any]:
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    groups: list[list[str]] = []
    used: set[str] = set()
    for group in optimizer.param_groups:
        group_names: list[str] = []
        for parameter in group["params"]:
            name = names.get(id(parameter))
            if name is None or name in used:
                raise ValueError("macro optimizer parameters must be unique model parameters")
            used.add(name)
            group_names.append(name)
        groups.append(group_names)
    return {"class": f"{type(optimizer).__module__}.{type(optimizer).__qualname__}", "groups": groups}


def _check_optimizer_state(state: Any, model: nn.Module, optimizer: Optimizer) -> None:
    if not isinstance(state, Mapping) or set(state) != {"state", "param_groups"}:
        raise ValueError("macro optimizer state structure differs")
    saved_groups = state["param_groups"]
    live_groups = optimizer.state_dict()["param_groups"]
    if not isinstance(saved_groups, list) or len(saved_groups) != len(live_groups):
        raise ValueError("macro optimizer param_groups differ")
    parameters: dict[int, torch.Tensor] = {}
    for saved, live, actual in zip(saved_groups, live_groups, optimizer.param_groups, strict=True):
        if not isinstance(saved, Mapping) or not isinstance(saved.get("params"), list):
            raise ValueError("macro optimizer group is malformed")
        if saved["params"] != live["params"]:
            raise ValueError("macro optimizer group parameter identifiers/order differ")
        saved_options = {key: value for key, value in saved.items() if key != "params"}
        live_options = {key: value for key, value in live.items() if key != "params"}
        # Standard AdamW options include tuples; JSON normalization keeps their order.
        if first_difference(canonical_facts(saved_options), canonical_facts(live_options)) is not None:
            raise ValueError("macro optimizer group hyperparameters differ")
        for identifier, parameter in zip(saved["params"], actual["params"], strict=True):
            if isinstance(identifier, bool) or not isinstance(identifier, int) or identifier in parameters:
                raise ValueError("macro optimizer parameter identifiers are malformed")
            parameters[identifier] = parameter
    if not isinstance(state["state"], Mapping):
        raise ValueError("macro optimizer slots are malformed")
    for identifier, slots in state["state"].items():
        if identifier not in parameters or not isinstance(slots, Mapping):
            raise ValueError("macro optimizer slots refer to an unknown parameter")
        for name, value in slots.items():
            if isinstance(value, torch.Tensor):
                expected = parameters[identifier]
                if name == "step":
                    if value.numel() != 1 or float(value.item()) < 0:
                        raise ValueError("macro optimizer step must be a non-negative scalar")
                elif value.shape != expected.shape or value.dtype != expected.dtype:
                    raise ValueError("macro optimizer tensor shape/dtype differs")
        if isinstance(optimizer, torch.optim.AdamW) and slots:
            required = {"step", "exp_avg", "exp_avg_sq"}
            if not required.issubset(slots):
                raise ValueError("macro AdamW state is missing required slots")
            for name in ("exp_avg", "exp_avg_sq"):
                if not isinstance(slots[name], torch.Tensor):
                    raise ValueError("macro AdamW moments must be tensors")
            if not isinstance(slots["step"], torch.Tensor | int | float) or isinstance(slots["step"], bool):
                raise ValueError("macro AdamW step is malformed")
            if not isinstance(slots["step"], torch.Tensor) and slots["step"] < 0:
                raise ValueError("macro AdamW step must be non-negative")


def _initialization_sources(initialization: Mapping[str, Any]) -> tuple[Path, ...]:
    return tuple(
        Path(value)
        for key in ("source", "source_path")
        if isinstance(value := initialization.get(key), str) and value
    )


def _output_path(path: Path, root: Path | None, source_paths: Sequence[Path]) -> Path:
    resolved = resolve_artifact_path(path, root=root)
    for source in source_paths:
        other = source.resolve()
        if resolved == other or (resolved.exists() and other.exists() and os.path.samefile(resolved, other)):
            raise ValueError("macro output must not overwrite an initialization/checkpoint source")
    return resolved


def _atomic_torch(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(dict(payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        # Windows cannot open/fsync directory descriptors. On POSIX the rename
        # is durable only after the containing directory has also been synced.
        if os.name != "nt":
            directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        temporary_path.unlink(missing_ok=True)


def _read_checkpoint(path: Path, identity: MacroCheckpointIdentity, root: Path | None) -> dict[str, Any]:
    resolved = resolve_artifact_path(path, root=root)
    raw = torch.load(resolved, map_location="cpu", weights_only=False)
    if not isinstance(raw, Mapping) or raw.get("format") != MACRO_CHECKPOINT_FORMAT:
        raise ValueError("not a macro-credit cohort checkpoint")
    payload = dict(raw)
    required_keys = {
        "format", "boundary", "control_scores", "course_config", "model_config", "reward_identity",
        "lineage_id", "learner_version", "model", "optimizer", "optimizer_layout", "update",
        "next_seed_index", "runtime_environment", "rng", "initialization",
    }
    if set(payload) != required_keys:
        raise ValueError("macro checkpoint fields differ")
    for key, expected in identity.as_mapping().items():
        if first_difference(expected, payload.get(key)) is not None:
            raise ValueError(f"macro checkpoint {key} differs")
    for key in ("update", "next_seed_index"):
        _nonnegative_int(payload.get(key), key)
    if payload.get("boundary") != "committed-cohort" or payload.get("control_scores") != "policy_logits":
        raise ValueError("macro checkpoint boundary/model ABI differs")
    for key in ("model", "optimizer", "optimizer_layout", "runtime_environment", "initialization"):
        if not isinstance(payload.get(key), Mapping):
            raise ValueError(f"macro checkpoint missing {key}")
    _snapshot(payload)
    _validate_rng(payload.get("rng"), payload["runtime_environment"])
    return payload


def save_macro_checkpoint(
    path: Path,
    *,
    identity: MacroCheckpointIdentity,
    model: nn.Module,
    optimizer: Optimizer,
    update: int,
    next_seed_index: int,
    initialization: Mapping[str, Any],
    artifact_root: Path | None = None,
    source_paths: Sequence[Path] = (),
) -> dict[str, Any]:
    """Commit a completed cohort (or update-zero baseline) before publication.

    ``course_config`` must contain the *complete* effective training/seed/
    dependency identities, not merely CLI overrides. Owned numpy Generator
    objects must be regenerated from the recorded seed cursor by the course.
    """
    destination = _output_path(path, artifact_root, (*source_paths, *_initialization_sources(initialization)))
    environment = _runtime_environment(model)
    payload = {
        "format": MACRO_CHECKPOINT_FORMAT,
        "boundary": "committed-cohort",
        "control_scores": "policy_logits",
        **identity.as_mapping(),
        "model": _snapshot(model.state_dict()),
        "optimizer": _snapshot(optimizer.state_dict()),
        "optimizer_layout": _optimizer_layout(model, optimizer),
        "update": _nonnegative_int(update, "update"),
        "next_seed_index": _nonnegative_int(next_seed_index, "next_seed_index"),
        "runtime_environment": environment,
        "rng": _snapshot(_rng_state(environment)),
        "initialization": canonical_facts(initialization),
    }
    _check_optimizer_state(payload["optimizer"], model, optimizer)
    _atomic_torch(destination, payload)
    return payload


def load_macro_checkpoint(
    path: Path,
    *,
    identity: MacroCheckpointIdentity,
    model: nn.Module,
    optimizer: Optimizer,
    artifact_root: Path | None = None,
    restore_rng: bool = True,
) -> dict[str, Any]:
    """Restore only at a cohort boundary, on the identical recorded runtime.

    CPU/CUDA migration, changed visible accelerator topology, and changed
    Python/numpy/torch/backend versions are not exact continuation. Even with
    identical metadata, this is not a promise of bitwise GPU determinism.
    """
    payload = _read_checkpoint(path, identity, artifact_root)
    environment = _runtime_environment(model)
    if first_difference(environment, payload["runtime_environment"]) is not None:
        raise ValueError("macro checkpoint runtime environment differs")
    if first_difference(_optimizer_layout(model, optimizer), payload["optimizer_layout"]) is not None:
        raise ValueError("macro checkpoint optimizer layout differs")
    _check_optimizer_state(payload["optimizer"], model, optimizer)
    current_model = model.state_dict()
    saved_model = payload["model"]
    if set(saved_model) != set(current_model):
        raise ValueError("macro checkpoint model state keys differ")
    for name, current in current_model.items():
        saved = saved_model[name]
        if not isinstance(saved, torch.Tensor) or saved.shape != current.shape or saved.dtype != current.dtype:
            raise ValueError(f"macro checkpoint model tensor differs: {name}")
    # Validate CUDA RNG lengths without changing any generator.
    current_rng = _rng_state(environment)
    for saved, current in zip(payload["rng"]["torch_cuda"], current_rng["torch_cuda"], strict=True):
        if saved.shape != current.shape:
            raise ValueError("macro checkpoint CUDA RNG state shape differs")
    previous_model = _snapshot(current_model)
    previous_optimizer = _snapshot(optimizer.state_dict())
    try:
        optimizer.load_state_dict(payload["optimizer"])
        model.load_state_dict(saved_model, strict=True)
        if restore_rng:
            _restore_rng(payload["rng"])
    except Exception:
        optimizer.load_state_dict(previous_optimizer)
        model.load_state_dict(previous_model, strict=True)
        _restore_rng(current_rng)
        raise
    return payload


def publish_macro_checkpoint(
    checkpoint_path: Path,
    publication_path: Path,
    *,
    identity: MacroCheckpointIdentity,
    artifact_root: Path | None = None,
    source_paths: Sequence[Path] = (),
) -> dict[str, Any]:
    """Rebuild derived actor+V publication; the committed checkpoint wins."""
    checkpoint = resolve_artifact_path(checkpoint_path, root=artifact_root)
    payload = _read_checkpoint(checkpoint, identity, artifact_root)
    destination = _output_path(
        publication_path, artifact_root,
        (*source_paths, checkpoint, *_initialization_sources(payload["initialization"])),
    )
    publication = {
        "format": MACRO_PUBLICATION_FORMAT,
        "control_scores": "policy_logits",
        **identity.as_mapping(),
        "online": payload["model"],
        "policy_version": payload["update"],
        "next_seed_index": payload["next_seed_index"],
        "initialization": payload["initialization"],
        "checkpoint_source": str(checkpoint),
        "runtime_environment": payload["runtime_environment"],
    }
    _atomic_torch(destination, publication)
    return publication


def migrate_macro_checkpoint_runtime(
    source_path: Path,
    *,
    identity: MacroCheckpointIdentity,
    model: nn.Module,
    optimizer: Optimizer,
    artifact_root: Path | None = None,
    restore_rng: bool = True,
) -> dict[str, Any]:
    """Migrate model, optimizer, RNG, update, and seed cursor across runtime/throughput upgrades.

    Enforces identical model architecture, reward identity, learner version,
    grounded encoding ABI, compiler/clock/data versions, seed recipe, and
    optimizer layout while allowing updated course throughput settings
    (e.g. roots_per_cohort, parallel_collectors, batch sizes) and source hashes.
    """
    resolved = resolve_external_input_path(source_path, root=artifact_root)
    raw = torch.load(resolved, map_location="cpu", weights_only=False)
    if not isinstance(raw, Mapping) or raw.get("format") != MACRO_CHECKPOINT_FORMAT:
        raise ValueError("not a macro-credit cohort checkpoint")
    payload = dict(raw)
    required_keys = {
        "format", "boundary", "control_scores", "course_config", "model_config", "reward_identity",
        "lineage_id", "learner_version", "model", "optimizer", "optimizer_layout", "update",
        "next_seed_index", "runtime_environment", "rng", "initialization",
    }
    if set(payload) != required_keys:
        raise ValueError("macro checkpoint fields differ")
    expected_identity = identity.as_mapping()
    for invariant_key in ("model_config", "reward_identity", "learner_version"):
        if first_difference(expected_identity[invariant_key], payload.get(invariant_key)) is not None:
            raise ValueError(f"macro checkpoint {invariant_key} differs; runtime migration forbidden")
    src_course = payload.get("course_config")
    dst_course = expected_identity["course_config"]
    if not isinstance(src_course, Mapping):
        raise ValueError("macro checkpoint missing course_config")
    for course_invariant in ("encoding", "compiler", "decision_clock", "data", "seed_recipe"):
        if first_difference(dst_course.get(course_invariant), src_course.get(course_invariant)) is not None:
            raise ValueError(
                f"macro checkpoint course_config.{course_invariant} differs; runtime migration forbidden"
            )
    for key in ("update", "next_seed_index"):
        _nonnegative_int(payload.get(key), key)
    if payload.get("boundary") != "committed-cohort" or payload.get("control_scores") != "policy_logits":
        raise ValueError("macro checkpoint boundary/model ABI differs")
    for key in ("model", "optimizer", "optimizer_layout", "runtime_environment", "initialization"):
        if not isinstance(payload.get(key), Mapping):
            raise ValueError(f"macro checkpoint missing {key}")
    _snapshot(payload)
    _validate_rng(payload.get("rng"), payload["runtime_environment"])

    environment = _runtime_environment(model)
    if first_difference(environment, payload["runtime_environment"]) is not None:
        raise ValueError("macro checkpoint runtime environment differs")
    if first_difference(_optimizer_layout(model, optimizer), payload["optimizer_layout"]) is not None:
        raise ValueError("macro checkpoint optimizer layout differs")
    _check_optimizer_state(payload["optimizer"], model, optimizer)
    current_model = model.state_dict()
    saved_model = payload["model"]
    if set(saved_model) != set(current_model):
        raise ValueError("macro checkpoint model state keys differ")
    for name, current in current_model.items():
        saved = saved_model[name]
        if not isinstance(saved, torch.Tensor) or saved.shape != current.shape or saved.dtype != current.dtype:
            raise ValueError(f"macro checkpoint model tensor differs: {name}")
    current_rng = _rng_state(environment)
    for saved, current in zip(payload["rng"]["torch_cuda"], current_rng["torch_cuda"], strict=True):
        if saved.shape != current.shape:
            raise ValueError("macro checkpoint CUDA RNG state shape differs")
    previous_model = _snapshot(current_model)
    previous_optimizer = _snapshot(optimizer.state_dict())
    try:
        optimizer.load_state_dict(payload["optimizer"])
        model.load_state_dict(saved_model, strict=True)
        if restore_rng:
            _restore_rng(payload["rng"])
    except Exception:
        optimizer.load_state_dict(previous_optimizer)
        model.load_state_dict(previous_model, strict=True)
        _restore_rng(current_rng)
        raise
    return payload
