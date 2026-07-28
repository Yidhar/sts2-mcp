"""Frozen-policy checkpoint evaluation without restoring learner state.

This module deliberately does *not* call :func:`load_training_checkpoint`.
An evaluation loads only ``network.pt`` after validating the complete atomic
checkpoint directory.  Optimizer moments, the rollout FIFO, collector RNG and
training counters therefore cannot be consumed or mutated by an evaluation.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import torch

from sts2_rl.checkpoints import validate_resume_checkpoint
from sts2_rl.contracts import EnvironmentBackend
from sts2_rl.macro_evaluation import read_macro_journal

from .checkpointing import initialize_model_from_checkpoint, preflight_model_initialization
from .config import TrainingConfig, model_initialization_config_from_mapping
from .factory import build_backend, build_training_resources
from .runtime import evaluate_policy

_AUDIT_SCHEMA = "sts2-frozen-checkpoint-evaluation-v2"


def _device_request_uses_cuda(device: str) -> bool:
    normalized = str(device).strip().lower()
    if normalized == "auto":
        return bool(torch.cuda.is_available())
    return torch.device(device).type == "cuda"


def atomic_publish_directory(
    source: str | Path,
    destination: str | Path,
    *,
    timeout_s: float = 5.0,
) -> None:
    """Atomically publish a directory, tolerating transient sharing handles.

    Windows can reject a directory rename briefly while antivirus, indexing,
    or a read-only observer is closing a child-file handle.  Retrying the same
    atomic rename does not expose partial output.  A pre-existing destination,
    a non-permission error, or an exhausted deadline still fails closed.
    """

    source_path = Path(source)
    destination_path = Path(destination)
    if not source_path.is_dir():
        raise FileNotFoundError(f"atomic publication source is not a directory: {source_path}")
    if destination_path.exists():
        raise FileExistsError(f"atomic publication destination already exists: {destination_path}")
    deadline = time.monotonic() + max(float(timeout_s), 0.0)
    delay_s = 0.01
    while True:
        try:
            os.replace(source_path, destination_path)
            return
        except PermissionError:
            if destination_path.exists() or time.monotonic() >= deadline:
                raise
            remaining = deadline - time.monotonic()
            time.sleep(min(delay_s, max(remaining, 0.0)))
            delay_s = min(delay_s * 2.0, 0.25)


@contextmanager
def _preserve_global_rng_state() -> Iterator[None]:
    """Make the in-process evaluator observationally neutral to global RNGs."""

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    # Reading CUDA generators initializes the CUDA runtime.  Preserve them
    # only when the embedding process had already initialized CUDA; a CPU
    # evaluator must not create a new global CUDA context merely to snapshot
    # state that did not previously exist.
    cuda_states = (
        torch.cuda.get_rng_state_all()
        if torch.cuda.is_initialized()  # type: ignore[no-untyped-call]
        else None
    )
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_file(manifest: dict[str, Any], name: str) -> dict[str, Any]:
    files = manifest.get("files")
    if not isinstance(files, list):
        raise ValueError("checkpoint manifest files must be an array")
    for item in files:
        if isinstance(item, dict) and item.get("path") == name:
            return dict(item)
    raise ValueError(f"checkpoint manifest does not describe {name!r}")


def checkpoint_training_config(checkpoint: str | Path) -> TrainingConfig:
    """Load the versioned training configuration embedded in a checkpoint.

    ``validate_resume_checkpoint`` verifies every required payload hash before
    the configuration is trusted.  Missing or newly unknown configuration
    fields still fail closed in the narrow model-initialization parser.  The
    only reviewed historical interpretation is V10 -> V11 with fresh-policy
    replay disabled; exact resume does not use this function.
    """

    validated = validate_resume_checkpoint(checkpoint)
    payload = validated.metadata.get("training_config")
    if not isinstance(payload, dict):
        raise ValueError("checkpoint metadata has no training_config object")
    return model_initialization_config_from_mapping(payload)


@dataclass(frozen=True, slots=True)
class FrozenEvaluationResult:
    checkpoint: Path
    checkpoint_id: str
    source_environment_steps: int
    source_policy_version: int
    output_directory: Path
    audit_path: Path
    journal_path: Path
    summary: dict[str, float | int]


def _evaluate_checkpoint_policy_unprotected(
    checkpoint: str | Path,
    *,
    output_directory: str | Path,
    episodes: int,
    base_seed: int | None = None,
    device: str = "cpu",
    collector_device: str = "cpu",
    sim_exe_path: str | None = None,
    backend: EnvironmentBackend | None = None,
    simulator_provenance: Mapping[str, object] | None = None,
) -> FrozenEvaluationResult:
    """Evaluate exactly the learner network stored in an atomic checkpoint.

    The source rollout queue is intentionally never deserialized.  A fresh
    optimizer and empty queue are constructed only because the regular
    collector composition root owns them; neither participates in evaluation.
    The audit records these negative guarantees explicitly.
    """

    if isinstance(episodes, bool) or not isinstance(episodes, int) or episodes <= 0:
        raise ValueError("episodes must be a positive integer")
    root = Path(checkpoint).expanduser().resolve()
    validated = validate_resume_checkpoint(root)
    config_payload = validated.metadata.get("training_config")
    if not isinstance(config_payload, dict):
        raise ValueError("checkpoint metadata has no training_config object")
    source_config_version = config_payload.get("version")
    config = model_initialization_config_from_mapping(config_payload)
    runtime = replace(
        config.runtime,
        device=str(device),
        collector_device=str(collector_device),
        evaluation_steps=(),
        evaluation_episodes=0,
    )
    environment = config.environment
    if sim_exe_path is not None:
        environment = replace(environment, sim_exe_path=str(sim_exe_path))
    config = replace(config, runtime=runtime, environment=environment)

    # This second preflight validates the model/encoding ABI against the
    # reconstructed config in addition to the directory/hash validation above.
    validated = preflight_model_initialization(root, config=config)
    training_state = validated.metadata.get("training_state")
    if not isinstance(training_state, dict):
        raise ValueError("checkpoint metadata has no training_state object")
    source_steps = training_state.get("environment_steps")
    source_policy_version = training_state.get("policy_version")
    if isinstance(source_steps, bool) or not isinstance(source_steps, int) or source_steps < 0:
        raise ValueError("checkpoint environment_steps is invalid")
    if (
        isinstance(source_policy_version, bool)
        or not isinstance(source_policy_version, int)
        or source_policy_version < 0
    ):
        raise ValueError("checkpoint policy_version is invalid")
    checkpoint_id = validated.manifest.get("checkpoint_id")
    if not isinstance(checkpoint_id, str) or not checkpoint_id:
        raise ValueError("checkpoint manifest has no checkpoint_id")

    output_root = Path(output_directory).expanduser().resolve()
    output_root.parent.mkdir(parents=True, exist_ok=True)
    if output_root.exists():
        raise FileExistsError("frozen evaluation output already exists")
    staging_root = output_root.with_name(
        f".{output_root.name}.staging-{uuid4().hex}"
    )
    staging_root.mkdir(parents=False, exist_ok=False)
    staging_journal_path = staging_root / "trajectory.jsonl"
    journal_path = output_root / "trajectory.jsonl"
    audit_path = output_root / "evaluation.json"

    manifest_path = validated.root / "checkpoint.manifest.json"
    network_entry = _manifest_file(validated.manifest, "network.pt")
    queue_entry = _manifest_file(validated.manifest, "rollout_queue.pkl")
    source_manifest_sha = _sha256(manifest_path)
    started = time.time()
    try:
        resources = build_training_resources(config, backend=backend)
        try:
            initialized_from = initialize_model_from_checkpoint(
                validated.root,
                config=config,
                resources=resources,
            )
            if initialized_from != validated.root:
                raise RuntimeError("model initialization resolved a different checkpoint")
            if len(resources.rollout_queue) != 0:
                raise RuntimeError("frozen evaluator must start with an empty rollout queue")
            results, summary = evaluate_policy(
                resources,
                episodes=episodes,
                base_seed=config.runtime.seed if base_seed is None else int(base_seed),
                journal_path=staging_journal_path,
                backend_factory=(None if backend is not None else lambda: build_backend(config)),
            )
            macro_surface_telemetry = read_macro_journal(staging_journal_path)
            if len(resources.rollout_queue) != 0:
                raise RuntimeError("evaluation unexpectedly populated the rollout queue")
        finally:
            resources.close()

        audit: dict[str, Any] = {
            "schema_version": _AUDIT_SCHEMA,
            "created_unix_s": time.time(),
            "duration_s": time.time() - started,
            "evaluation_of": {
                "checkpoint": str(validated.root),
                "checkpoint_id": checkpoint_id,
                "manifest_sha256": source_manifest_sha,
                "network": network_entry,
                "rollout_queue": queue_entry,
                "training_state": training_state,
            },
            "policy_source": "network.pt",
            "source_training_config_version": source_config_version,
            "evaluation_training_config_version": config.version,
            "config_load_mode": "evaluation_model_parameter_initialization",
            "learner_updates_performed": 0,
            "optimizer_loaded": False,
            "rollout_queue_loaded": False,
            "training_rng_loaded": False,
            "training_checkpoint_published": False,
            "simulator_provenance": (
                dict(simulator_provenance)
                if simulator_provenance is not None
                else None
            ),
            "evaluation": {
                "episodes": episodes,
                "base_seed": (
                    config.runtime.seed if base_seed is None else int(base_seed)
                ),
                "seed_contract": "even-training/odd-held-out-v1",
                "device": str(device),
                "collector_device": str(collector_device),
                "journal": str(journal_path),
                "episode_metrics": [asdict(item) for item in results],
                "summary": summary,
                "macro_surface_telemetry": macro_surface_telemetry,
            },
        }
        (staging_root / "evaluation.json").write_text(
            json.dumps(audit, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        atomic_publish_directory(staging_root, output_root)
    except BaseException:
        shutil.rmtree(staging_root, ignore_errors=True)
        raise
    return FrozenEvaluationResult(
        checkpoint=validated.root,
        checkpoint_id=checkpoint_id,
        source_environment_steps=source_steps,
        source_policy_version=source_policy_version,
        output_directory=output_root,
        audit_path=audit_path,
        journal_path=journal_path,
        summary=summary,
    )


def evaluate_checkpoint_policy(
    checkpoint: str | Path,
    *,
    output_directory: str | Path,
    episodes: int,
    base_seed: int | None = None,
    device: str = "cpu",
    collector_device: str = "cpu",
    sim_exe_path: str | None = None,
    backend: EnvironmentBackend | None = None,
    simulator_provenance: Mapping[str, object] | None = None,
) -> FrozenEvaluationResult:
    """Evaluate a frozen checkpoint without perturbing process-global RNGs.

    The output is assembled in a sibling staging directory and atomically
    published only after both journal and audit are complete.  Failed
    infrastructure attempts therefore never poison the requested output path.
    """

    if (
        _device_request_uses_cuda(device)
        or _device_request_uses_cuda(collector_device)
    ) and not torch.cuda.is_initialized():  # type: ignore[no-untyped-call]
        raise RuntimeError(
            "in-process frozen CUDA evaluation requires CUDA to be initialized "
            "before RNG preservation; use the CLI subprocess or initialize CUDA first"
        )

    with _preserve_global_rng_state():
        return _evaluate_checkpoint_policy_unprotected(
            checkpoint,
            output_directory=output_directory,
            episodes=episodes,
            base_seed=base_seed,
            device=device,
            collector_device=collector_device,
            sim_exe_path=sim_exe_path,
            backend=backend,
            simulator_provenance=simulator_provenance,
        )


__all__ = [
    "FrozenEvaluationResult",
    "atomic_publish_directory",
    "checkpoint_training_config",
    "evaluate_checkpoint_policy",
]
