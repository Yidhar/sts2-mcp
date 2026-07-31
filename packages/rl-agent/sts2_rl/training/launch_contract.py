"""Immutable provenance contract for supervised training launches.

The launcher and trainer are separate processes.  A launcher-side preflight
therefore cannot be treated as authority unless the exact reviewed decision is
carried across the process boundary and the trainer binds it to the checkpoint
it actually opens.  This module provides that narrow handshake.

The contract is deliberately *not* a training checkpoint and never grants
resume authority.  It records one launch nonce, the exact model-initialization
source, the pinned shadow evidence, and the required fresh training state.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, cast
from uuid import UUID

from sts2_rl.checkpoints import CheckpointIntegrityError, ValidatedResumeCheckpoint

SUPERVISED_LAUNCH_CONTRACT_VERSION: Final = "sts2-supervised-training-launch-contract-v2"
RUNTIME_READINESS_REPORT_VERSION: Final = "sts2-liveness-head-active-shape-stress-v3"
FORMAL_REPORT_GENERATION_SOURCE_VERSION: Final = "sts2-formal-report-generation-source-v1"
RUNTIME_READINESS_SEAL_VERSION: Final = "sts2-runtime-readiness-evidence-v1"
SOURCE_AUTHORITY_VERSION: Final = "sts2-two-phase-source-authority-v1"
EXTERNAL_SEAL_VERSION: Final = "sts2-external-launch-contract-seal-v1"

_SHA256_LENGTH = 64
_INITIAL_STATE_FIELDS = (
    "environment_steps",
    "learner_updates",
    "episodes",
    "evaluation_episodes",
    "policy_version",
    "actor_policy_version",
    "consumed_unrolls",
    "maximum_observed_candidates",
)
_REVIEWED_TRAINER_ENVIRONMENT_KEYS = (
    "STS2_ARTIFACT_ROOT",
    "PYTHONPATH",
    "PYTHONNOUSERSITE",
    "PYTHONUNBUFFERED",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "PATH",
)
_REMOVED_TRAINER_ENVIRONMENT_KEYS = (
    "PYTHONHOME",
    "VENV_DIR",
    "STS2_HEADLESS_SIM_EXE",
)
_RUNTIME_READINESS_GATES = frozenset(
    {
        "bounded_tbptt_segments",
        "calibration_and_mature_phase_contract",
        "finite_gradients",
        "finite_losses",
        "formal_shape_contract",
        "matched_outcome_contrast_path",
        "no_optimizer_step",
        "nonzero_expected_gradients",
        "ordinary_rows_not_globally_padded",
        "production_manifest_work_accounting",
        "stress_forward_uses_active_timestep_batch",
    }
)
_READINESS_CONFIG_FIELDS = {
    "calibration_updates",
    "fingerprint_sha256",
    "maximum_candidates",
    "maximum_context_steps",
    "profile",
    "risk_actor_start_update",
    "source",
    "tbptt_window_steps",
    "version",
}
_READINESS_INITIALIZATION_FIELDS = {
    "authority",
    "checkpoint_id",
    "checkpoint_path",
    "contract_name",
    "kind",
    "manifest_sha256",
    "metadata_sha256",
}
_READINESS_RUNTIME_FIELDS = {
    "accelerator",
    "cuda_available",
    "cuda_device_count",
    "environment",
    "platform",
    "python",
    "resolved_device",
    "torch_cuda_version",
    "torch_git_version",
    "torch_hip_version",
    "torch_version",
}
_READINESS_ACCELERATOR_FIELDS = {
    "device_index",
    "multi_processor_count",
    "name",
    "total_memory_bytes",
}
_READINESS_VISIBILITY_FIELDS = {
    "HIP_VISIBLE_DEVICES",
    "HSA_OVERRIDE_GFX_VERSION",
    "ROCR_VISIBLE_DEVICES",
}
_READINESS_SDPA_FIELDS = {
    "applicability",
    "applied",
    "current_flags",
    "devices",
    "effective_backend",
    "hip_version",
    "previous_flags",
    "requested_policy",
    "version",
}
_READINESS_SHAPE_FIELDS = {
    "candidate_decisions_per_context",
    "contexts",
    "expected_tbptt_segments",
    "legal_candidates_per_step",
    "ordinary_probe_candidates",
    "replayed_candidate_decisions",
    "replayed_steps",
    "steps_per_context",
    "tbptt_segments_per_context",
}
_SOURCE_IDENTITY_FIELDS = {
    "checkout_root",
    "git_object_format",
    "implementation_commit",
    "implementation_tree",
    "worktree_clean",
}
_FORMAL_REPORT_GENERATION_SOURCE_FIELDS = _SOURCE_IDENTITY_FIELDS | {
    "schema_version",
    "validator_relative_path",
    "validator_sha256",
}


class SupervisedLaunchContractError(ValueError):
    """An immutable supervised-launch contract is malformed or mismatched."""


def _object(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SupervisedLaunchContractError(f"{label} must be a JSON object")
    if any(not isinstance(key, str) for key in value):
        raise SupervisedLaunchContractError(f"{label} keys must be strings")
    return value


def _exact_fields(value: Mapping[str, object], expected: set[str], *, label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise SupervisedLaunchContractError(
            f"{label} fields differ: missing={sorted(expected - actual)} extra={sorted(actual - expected)}"
        )


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SupervisedLaunchContractError(f"{label} must be non-empty text")
    return value


def _sha256(value: object, *, label: str) -> str:
    digest = _text(value, label=label)
    if len(digest) != _SHA256_LENGTH or any(character not in "0123456789abcdef" for character in digest):
        raise SupervisedLaunchContractError(f"{label} must be a lowercase SHA-256 digest")
    return digest


def _integer(value: object, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise SupervisedLaunchContractError(f"{label} must be an integer >= {minimum}")
    return value


def _uuid(value: object, *, label: str) -> str:
    raw = _text(value, label=label)
    try:
        parsed = UUID(raw)
    except ValueError as exc:
        raise SupervisedLaunchContractError(f"{label} must be a canonical UUID") from exc
    if str(parsed) != raw:
        raise SupervisedLaunchContractError(f"{label} must be a canonical UUID")
    return raw


def _absolute_path(value: object, *, label: str) -> str:
    raw = _text(value, label=label)
    path = Path(raw)
    if not path.is_absolute():
        raise SupervisedLaunchContractError(f"{label} must be absolute")
    return os.fspath(path.resolve())


def _boolean(value: object, *, label: str) -> bool:
    if not isinstance(value, bool):
        raise SupervisedLaunchContractError(f"{label} must be a boolean")
    return value


def _optional_text(value: object, *, label: str) -> str | None:
    if value is None:
        return None
    return _text(value, label=label)


def _git_oid(value: object, *, label: str, object_format: str) -> str:
    oid = _text(value, label=label)
    expected_length = 40 if object_format == "sha1" else 64
    if len(oid) != expected_length or re.fullmatch(r"[0-9a-f]+", oid) is None:
        raise SupervisedLaunchContractError(
            f"{label} must be a lowercase {object_format} Git object ID",
        )
    return oid


def _canonical_json_sha256(value: object) -> str:
    raw = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _source_identity(value: object, *, label: str) -> dict[str, Any]:
    source = _object(value, label=label)
    _exact_fields(source, _SOURCE_IDENTITY_FIELDS, label=label)
    checkout_root = _absolute_path(source["checkout_root"], label=f"{label}.checkout_root")
    object_format = _text(source["git_object_format"], label=f"{label}.git_object_format")
    if object_format not in {"sha1", "sha256"}:
        raise SupervisedLaunchContractError(f"{label}.git_object_format is unsupported")
    commit = _git_oid(
        source["implementation_commit"],
        label=f"{label}.implementation_commit",
        object_format=object_format,
    )
    tree = _git_oid(
        source["implementation_tree"],
        label=f"{label}.implementation_tree",
        object_format=object_format,
    )
    if _boolean(source["worktree_clean"], label=f"{label}.worktree_clean") is not True:
        raise SupervisedLaunchContractError(f"{label}.worktree_clean must be true")
    return {
        "checkout_root": checkout_root,
        "git_object_format": object_format,
        "implementation_commit": commit,
        "implementation_tree": tree,
        "worktree_clean": True,
    }


def _runtime_identity(value: object, *, label: str) -> dict[str, Any]:
    identity = _object(value, label=label)
    _exact_fields(identity, _READINESS_RUNTIME_FIELDS, label=label)
    accelerator = _object(identity["accelerator"], label=f"{label}.accelerator")
    _exact_fields(accelerator, _READINESS_ACCELERATOR_FIELDS, label=f"{label}.accelerator")
    visibility = _object(identity["environment"], label=f"{label}.environment")
    _exact_fields(visibility, _READINESS_VISIBILITY_FIELDS, label=f"{label}.environment")
    device_index = _integer(
        accelerator["device_index"],
        label=f"{label}.accelerator.device_index",
    )
    normalized_accelerator = {
        "device_index": device_index,
        "multi_processor_count": _integer(
            accelerator["multi_processor_count"],
            label=f"{label}.accelerator.multi_processor_count",
            minimum=1,
        ),
        "name": _text(accelerator["name"], label=f"{label}.accelerator.name"),
        "total_memory_bytes": _integer(
            accelerator["total_memory_bytes"],
            label=f"{label}.accelerator.total_memory_bytes",
            minimum=1,
        ),
    }
    normalized_visibility = {
        key: _optional_text(visibility[key], label=f"{label}.environment.{key}")
        for key in sorted(_READINESS_VISIBILITY_FIELDS)
    }
    if _boolean(identity["cuda_available"], label=f"{label}.cuda_available") is not True:
        raise SupervisedLaunchContractError(f"{label}.cuda_available must be true")
    count = _integer(identity["cuda_device_count"], label=f"{label}.cuda_device_count", minimum=1)
    if device_index >= count:
        raise SupervisedLaunchContractError(f"{label}.accelerator.device_index is out of range")
    resolved_device = _text(identity["resolved_device"], label=f"{label}.resolved_device")
    if resolved_device != "cuda":
        raise SupervisedLaunchContractError(f"{label}.resolved_device must be cuda for ROCm")
    hip_version = _text(identity["torch_hip_version"], label=f"{label}.torch_hip_version")
    if identity["torch_cuda_version"] is not None:
        raise SupervisedLaunchContractError(f"{label}.torch_cuda_version must be null for ROCm")
    return {
        "accelerator": normalized_accelerator,
        "cuda_available": True,
        "cuda_device_count": count,
        "environment": normalized_visibility,
        "platform": _text(identity["platform"], label=f"{label}.platform"),
        "python": _text(identity["python"], label=f"{label}.python"),
        "resolved_device": resolved_device,
        "torch_cuda_version": None,
        "torch_git_version": _text(
            identity["torch_git_version"],
            label=f"{label}.torch_git_version",
        ),
        "torch_hip_version": hip_version,
        "torch_version": _text(identity["torch_version"], label=f"{label}.torch_version"),
    }


def current_rocm_runtime_identity() -> dict[str, Any]:
    """Return the exact runtime identity used by the readiness stress report."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - production environment invariant
        raise SupervisedLaunchContractError("PyTorch is unavailable for runtime readiness verification") from exc
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise SupervisedLaunchContractError("ROCm GPU is unavailable")
    properties = torch.cuda.get_device_properties(0)
    raw = {
        "accelerator": {
            "device_index": 0,
            "multi_processor_count": int(properties.multi_processor_count),
            "name": str(properties.name),
            "total_memory_bytes": int(properties.total_memory),
        },
        "cuda_available": True,
        "cuda_device_count": int(torch.cuda.device_count()),
        "environment": {key: os.environ.get(key) for key in sorted(_READINESS_VISIBILITY_FIELDS)},
        "platform": platform.platform(),
        "python": sys.version,
        "resolved_device": "cuda",
        "torch_cuda_version": torch.version.cuda,
        "torch_git_version": torch.version.git_version,
        "torch_hip_version": torch.version.hip,
        "torch_version": torch.__version__,
    }
    return _runtime_identity(raw, label="live ROCm runtime identity")


def _runtime_readiness_report_binding(report: Mapping[str, Any]) -> dict[str, Any]:
    expected_top_level = {
        "active_shape_probe",
        "authority",
        "config",
        "durations_seconds",
        "gates",
        "generation_source",
        "gpu_peak",
        "initialization",
        "one_shot",
        "optimizer_steps",
        "phases",
        "runtime_identity",
        "sdpa",
        "shape",
        "status",
        "training_authority",
        "version",
    }
    _exact_fields(report, expected_top_level, label="runtime readiness report")
    if report["version"] != RUNTIME_READINESS_REPORT_VERSION:
        raise SupervisedLaunchContractError("runtime readiness report version changed")
    if report["status"] != "passed":
        raise SupervisedLaunchContractError("runtime readiness report did not pass")
    if _boolean(report["training_authority"], label="runtime readiness training_authority") is not False:
        raise SupervisedLaunchContractError("runtime readiness report must not grant training authority")
    if _boolean(report["one_shot"], label="runtime readiness one_shot") is not True:
        raise SupervisedLaunchContractError("runtime readiness report must be one-shot")
    if _integer(report["optimizer_steps"], label="runtime readiness optimizer_steps") != 0:
        raise SupervisedLaunchContractError("runtime readiness report must not perform optimizer steps")
    generation_source = _formal_report_generation_source(
        report["generation_source"],
        label="runtime readiness generation_source",
    )

    config = _object(report["config"], label="runtime readiness config")
    _exact_fields(config, _READINESS_CONFIG_FIELDS, label="runtime readiness config")
    normalized_config = {
        "calibration_updates": _integer(
            config["calibration_updates"],
            label="runtime readiness config.calibration_updates",
            minimum=1,
        ),
        "fingerprint_sha256": _sha256(
            config["fingerprint_sha256"],
            label="runtime readiness config.fingerprint_sha256",
        ),
        "maximum_candidates": _integer(
            config["maximum_candidates"],
            label="runtime readiness config.maximum_candidates",
            minimum=1,
        ),
        "maximum_context_steps": _integer(
            config["maximum_context_steps"],
            label="runtime readiness config.maximum_context_steps",
            minimum=1,
        ),
        "profile": _text(config["profile"], label="runtime readiness config.profile"),
        "risk_actor_start_update": _integer(
            config["risk_actor_start_update"],
            label="runtime readiness config.risk_actor_start_update",
            minimum=1,
        ),
        "source": _absolute_path(config["source"], label="runtime readiness config.source"),
        "tbptt_window_steps": _integer(
            config["tbptt_window_steps"],
            label="runtime readiness config.tbptt_window_steps",
            minimum=1,
        ),
        "version": _text(config["version"], label="runtime readiness config.version"),
    }
    if normalized_config["maximum_context_steps"] != 256 or normalized_config["maximum_candidates"] != 256:
        raise SupervisedLaunchContractError("runtime readiness report is not the formal 256x256 shape")

    initialization = _object(report["initialization"], label="runtime readiness initialization")
    _exact_fields(
        initialization,
        _READINESS_INITIALIZATION_FIELDS,
        label="runtime readiness initialization",
    )
    normalized_initialization = {
        "authority": _text(initialization["authority"], label="runtime readiness initialization.authority"),
        "checkpoint_id": _uuid(
            initialization["checkpoint_id"],
            label="runtime readiness initialization.checkpoint_id",
        ),
        "checkpoint_path": _absolute_path(
            initialization["checkpoint_path"],
            label="runtime readiness initialization.checkpoint_path",
        ),
        "contract_name": _text(
            initialization["contract_name"],
            label="runtime readiness initialization.contract_name",
        ),
        "kind": _text(initialization["kind"], label="runtime readiness initialization.kind"),
        "manifest_sha256": _sha256(
            initialization["manifest_sha256"],
            label="runtime readiness initialization.manifest_sha256",
        ),
        "metadata_sha256": _sha256(
            initialization["metadata_sha256"],
            label="runtime readiness initialization.metadata_sha256",
        ),
    }
    if normalized_initialization["kind"] != "frozen-v28-model-init":
        raise SupervisedLaunchContractError("runtime readiness source is not frozen v28 model-init")

    shape = _object(report["shape"], label="runtime readiness shape")
    _exact_fields(shape, _READINESS_SHAPE_FIELDS, label="runtime readiness shape")
    normalized_shape = {
        key: _integer(shape[key], label=f"runtime readiness shape.{key}", minimum=1)
        for key in sorted(_READINESS_SHAPE_FIELDS)
    }
    expected_shape = {
        "candidate_decisions_per_context": 65_536,
        "contexts": 2,
        "expected_tbptt_segments": 32,
        "legal_candidates_per_step": 256,
        "ordinary_probe_candidates": 3,
        "replayed_candidate_decisions": 131_072,
        "replayed_steps": 512,
        "steps_per_context": 256,
        "tbptt_segments_per_context": 16,
    }
    if normalized_shape != expected_shape:
        raise SupervisedLaunchContractError("runtime readiness formal shape changed")

    gates = _object(report["gates"], label="runtime readiness gates")
    _exact_fields(gates, set(_RUNTIME_READINESS_GATES), label="runtime readiness gates")
    normalized_gates = {
        key: _boolean(gates[key], label=f"runtime readiness gates.{key}") for key in sorted(_RUNTIME_READINESS_GATES)
    }
    if not all(normalized_gates.values()):
        raise SupervisedLaunchContractError("runtime readiness report has a failed gate")

    gpu_peak = _object(report["gpu_peak"], label="runtime readiness gpu_peak")
    _exact_fields(
        gpu_peak,
        {"maximum_allocated_bytes", "maximum_reserved_bytes"},
        label="runtime readiness gpu_peak",
    )
    normalized_gpu_peak = {
        key: _integer(gpu_peak[key], label=f"runtime readiness gpu_peak.{key}", minimum=1)
        for key in ("maximum_allocated_bytes", "maximum_reserved_bytes")
    }
    runtime_identity = _runtime_identity(report["runtime_identity"], label="runtime readiness runtime_identity")
    if normalized_gpu_peak["maximum_allocated_bytes"] > normalized_gpu_peak["maximum_reserved_bytes"]:
        raise SupervisedLaunchContractError("runtime readiness allocated memory exceeds reserved memory")
    if normalized_gpu_peak["maximum_reserved_bytes"] >= runtime_identity["accelerator"]["total_memory_bytes"]:
        raise SupervisedLaunchContractError("runtime readiness consumed all accelerator memory")

    sdpa = _object(report["sdpa"], label="runtime readiness sdpa")
    _exact_fields(sdpa, _READINESS_SDPA_FIELDS, label="runtime readiness sdpa")
    normalized_sdpa = json.loads(json.dumps(sdpa, allow_nan=False, sort_keys=True))
    if (
        normalized_sdpa.get("version") != "sts2-rocm-sdpa-execution-v1"
        or normalized_sdpa.get("applicability") != "rocm_cuda"
        or normalized_sdpa.get("applied") is not True
        or normalized_sdpa.get("effective_backend") != "math_only"
        or normalized_sdpa.get("requested_policy") != "math"
        or normalized_sdpa.get("hip_version") != runtime_identity["torch_hip_version"]
        or normalized_sdpa.get("devices") != ["cuda"]
        or normalized_sdpa.get("current_flags")
        != {"cudnn": False, "flash": False, "math": True, "memory_efficient": False}
    ):
        raise SupervisedLaunchContractError("runtime readiness SDPA proof changed")

    return {
        "config": normalized_config,
        "gates": normalized_gates,
        "generation_source": generation_source,
        "gpu_peak": normalized_gpu_peak,
        "initialization": normalized_initialization,
        "runtime_identity": runtime_identity,
        "sdpa": normalized_sdpa,
        "shape": normalized_shape,
        "status": "passed",
        "training_authority": False,
        "version": RUNTIME_READINESS_REPORT_VERSION,
    }


def load_runtime_readiness_evidence(
    report_path: str | Path,
    *,
    expected_sha256: str,
    artifact_root: str | Path,
) -> dict[str, Any]:
    """Authenticate and normalize one independent non-training readiness proof."""

    expected_digest = _sha256(expected_sha256, label="runtime readiness expected_sha256")
    root = Path(artifact_root).expanduser()
    if not root.is_absolute():
        raise SupervisedLaunchContractError("runtime readiness artifact_root must be absolute")
    root = root.resolve()
    path = Path(report_path).expanduser()
    if not path.is_absolute():
        raise SupervisedLaunchContractError("runtime readiness report path must be absolute")
    path = path.resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise SupervisedLaunchContractError("runtime readiness report escapes artifact_root") from exc
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise SupervisedLaunchContractError(f"runtime readiness report is unreadable: {path}") from exc
    observed_digest = hashlib.sha256(raw).hexdigest()
    if observed_digest != expected_digest:
        raise SupervisedLaunchContractError(
            f"runtime readiness report SHA-256 mismatch: expected={expected_digest} actual={observed_digest}",
        )
    try:
        report = _object(json.loads(raw), label="runtime readiness report")
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SupervisedLaunchContractError("runtime readiness report is not valid UTF-8 JSON") from exc
    return {
        "schema_version": RUNTIME_READINESS_SEAL_VERSION,
        "artifact_root": os.fspath(root),
        "report_path": os.fspath(path),
        "report_sha256": observed_digest,
        "report_binding": _runtime_readiness_report_binding(report),
    }


@dataclass(frozen=True, slots=True)
class SupervisedLaunchContract:
    """A strictly validated, immutable launcher-to-trainer handshake."""

    path: Path
    sha256: str
    payload: dict[str, Any]

    @property
    def launch_id(self) -> str:
        return str(self.payload["launch_id"])

    @property
    def source_checkpoint(self) -> Mapping[str, Any]:
        value = self.payload["source_checkpoint"]
        if not isinstance(value, Mapping):  # pragma: no cover - loader invariant
            raise RuntimeError("validated launch contract lost source checkpoint")
        return value

    @property
    def initial_training_state(self) -> Mapping[str, int]:
        value = self.payload["initial_training_state"]
        if not isinstance(value, Mapping):  # pragma: no cover - loader invariant
            raise RuntimeError("validated launch contract lost initial state")
        return cast(Mapping[str, int], value)

    @property
    def implementation_source(self) -> Mapping[str, Any]:
        value = self.payload["implementation_source"]
        if not isinstance(value, Mapping):  # pragma: no cover - loader invariant
            raise RuntimeError("validated launch contract lost implementation source")
        return value

    @property
    def runtime_readiness_evidence(self) -> Mapping[str, Any]:
        value = self.payload["runtime_readiness_evidence"]
        if not isinstance(value, Mapping):  # pragma: no cover - loader invariant
            raise RuntimeError("validated launch contract lost runtime readiness evidence")
        return value

    def provenance_mapping(self) -> dict[str, Any]:
        return {
            "contract_path": os.fspath(self.path),
            "contract_sha256": self.sha256,
            "trainer_environment_sha256": self.payload["trainer_environment_sha256"],
            "implementation_commit": self.implementation_source["implementation_commit"],
            "implementation_tree": self.implementation_source["implementation_tree"],
            "worktree_clean": self.implementation_source["worktree_clean"],
            "runtime_readiness_report_sha256": self.runtime_readiness_evidence["report_sha256"],
            "contract": self.payload,
        }


def _trainer_environment_payload(
    environment: Mapping[str, str],
) -> dict[str, Any]:
    if not all(isinstance(key, str) and isinstance(value, str) for key, value in environment.items()):
        raise SupervisedLaunchContractError(
            "trainer environment keys and values must be text",
        )
    missing = [key for key in _REVIEWED_TRAINER_ENVIRONMENT_KEYS if key not in environment]
    if missing:
        raise SupervisedLaunchContractError("trainer environment is missing reviewed keys: " + ", ".join(missing))
    retained = [key for key in _REMOVED_TRAINER_ENVIRONMENT_KEYS if key in environment]
    if retained:
        raise SupervisedLaunchContractError("trainer environment retained forbidden keys: " + ", ".join(retained))
    return {
        "set": dict(environment),
        "unset": list(_REMOVED_TRAINER_ENVIRONMENT_KEYS),
    }


def trainer_environment_sha256(
    environment: Mapping[str, str],
) -> str:
    """Return the canonical reviewed environment digest seen by the trainer."""

    serialized = json.dumps(
        _trainer_environment_payload(environment),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def validate_supervised_trainer_environment(
    contract: SupervisedLaunchContract,
    *,
    environment: Mapping[str, str] | None = None,
) -> str:
    """Bind the trainer process environment to the launcher-reviewed digest."""

    observed = trainer_environment_sha256(
        os.environ if environment is None else environment,
    )
    expected = _sha256(
        contract.payload.get("trainer_environment_sha256"),
        label="trainer_environment_sha256",
    )
    if observed != expected:
        raise SupervisedLaunchContractError(
            f"trainer process environment differs from immutable launch contract: expected={expected} actual={observed}"
        )
    return observed


def _repo_relative_path(value: object, *, label: str) -> str:
    raw = _text(value, label=label)
    path = Path(raw)
    if path.is_absolute() or ".." in path.parts or raw != path.as_posix():
        raise SupervisedLaunchContractError(f"{label} must be a normalized repository-relative POSIX path")
    return raw


def _source_authority(
    value: object,
    *,
    implementation_source: Mapping[str, Any],
) -> dict[str, Any]:
    authority = _object(value, label="source_authority")
    _exact_fields(
        authority,
        {
            "schema_version",
            "evidence_generation_source",
            "validated_code_sha256",
            "validated_code_mapping_sha256",
            "allowed_seal_paths",
            "observed_seal_paths",
        },
        label="source_authority",
    )
    if authority["schema_version"] != SOURCE_AUTHORITY_VERSION:
        raise SupervisedLaunchContractError("source_authority schema changed")
    generation = _source_identity(
        authority["evidence_generation_source"],
        label="source_authority.evidence_generation_source",
    )
    if generation["checkout_root"] != implementation_source["checkout_root"]:
        raise SupervisedLaunchContractError("evidence and implementation checkout roots differ")
    if generation["git_object_format"] != implementation_source["git_object_format"]:
        raise SupervisedLaunchContractError("evidence and implementation Git object formats differ")
    if generation["implementation_commit"] == implementation_source["implementation_commit"]:
        raise SupervisedLaunchContractError("two-phase authority requires distinct generation and seal commits")

    code_hashes = _object(authority["validated_code_sha256"], label="source_authority.validated_code_sha256")
    if not code_hashes:
        raise SupervisedLaunchContractError("source_authority validated-code mapping is empty")
    normalized_hashes: dict[str, str] = {}
    for raw_path, raw_digest in sorted(code_hashes.items()):
        relative = _repo_relative_path(raw_path, label="source_authority validated-code path")
        normalized_hashes[relative] = _sha256(
            raw_digest,
            label=f"source_authority.validated_code_sha256[{relative!r}]",
        )
    mapping_digest = _sha256(
        authority["validated_code_mapping_sha256"],
        label="source_authority.validated_code_mapping_sha256",
    )
    if _canonical_json_sha256(normalized_hashes) != mapping_digest:
        raise SupervisedLaunchContractError("source_authority validated-code mapping digest differs")

    allowed_raw = authority["allowed_seal_paths"]
    observed_raw = authority["observed_seal_paths"]
    if not isinstance(allowed_raw, list) or not isinstance(observed_raw, list):
        raise SupervisedLaunchContractError("source_authority seal paths must be arrays")
    allowed = [_repo_relative_path(item, label="source_authority allowed seal path") for item in allowed_raw]
    observed = [_repo_relative_path(item, label="source_authority observed seal path") for item in observed_raw]
    if allowed != sorted(set(allowed)) or observed != sorted(set(observed)):
        raise SupervisedLaunchContractError("source_authority seal paths must be sorted and unique")
    if not observed or not set(observed).issubset(allowed):
        raise SupervisedLaunchContractError("observed seal paths are empty or exceed the allowlist")
    return {
        "schema_version": SOURCE_AUTHORITY_VERSION,
        "evidence_generation_source": generation,
        "validated_code_sha256": normalized_hashes,
        "validated_code_mapping_sha256": mapping_digest,
        "allowed_seal_paths": allowed,
        "observed_seal_paths": observed,
    }


def _seal_provenance(value: object) -> dict[str, Any]:
    seal = _object(value, label="seal_provenance")
    _exact_fields(
        seal,
        {
            "schema_version",
            "algorithm",
            "digest_location",
            "self_digest_embedded",
        },
        label="seal_provenance",
    )
    expected = {
        "schema_version": EXTERNAL_SEAL_VERSION,
        "algorithm": "sha256",
        "digest_location": "supervisor_manifest.launch_contract.sha256",
        "self_digest_embedded": False,
    }
    if dict(seal) != expected:
        raise SupervisedLaunchContractError("seal_provenance changed")
    return expected


def _runtime_readiness_evidence(value: object) -> dict[str, Any]:
    evidence = _object(value, label="runtime_readiness_evidence")
    _exact_fields(
        evidence,
        {
            "schema_version",
            "artifact_root",
            "report_path",
            "report_sha256",
            "report_binding",
        },
        label="runtime_readiness_evidence",
    )
    if evidence["schema_version"] != RUNTIME_READINESS_SEAL_VERSION:
        raise SupervisedLaunchContractError("runtime_readiness_evidence schema changed")
    artifact_root = _absolute_path(
        evidence["artifact_root"],
        label="runtime_readiness_evidence.artifact_root",
    )
    report_path = _absolute_path(
        evidence["report_path"],
        label="runtime_readiness_evidence.report_path",
    )
    report_sha256 = _sha256(
        evidence["report_sha256"],
        label="runtime_readiness_evidence.report_sha256",
    )
    observed = load_runtime_readiness_evidence(
        report_path,
        expected_sha256=report_sha256,
        artifact_root=artifact_root,
    )
    if observed.get("report_binding") != evidence["report_binding"]:
        raise SupervisedLaunchContractError("runtime readiness report binding differs from contract")
    return observed


def load_supervised_launch_contract(
    path: str | Path,
    *,
    expected_sha256: str,
) -> SupervisedLaunchContract:
    """Read and strictly validate one immutable supervised-launch contract."""

    expected_digest = _sha256(expected_sha256, label="expected_sha256")
    contract_path = Path(path).expanduser()
    if not contract_path.is_absolute():
        raise SupervisedLaunchContractError("launch contract path must be absolute")
    contract_path = contract_path.resolve()
    try:
        raw = contract_path.read_bytes()
    except OSError as exc:
        raise SupervisedLaunchContractError(f"launch contract is unreadable: {contract_path}") from exc
    observed_digest = hashlib.sha256(raw).hexdigest()
    if observed_digest != expected_digest:
        raise SupervisedLaunchContractError(
            f"launch contract SHA-256 mismatch: expected={expected_digest} actual={observed_digest}"
        )
    try:
        payload = _object(json.loads(raw), label="launch contract")
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SupervisedLaunchContractError("launch contract is not valid UTF-8 JSON") from exc

    _exact_fields(
        payload,
        {
            "schema_version",
            "launch_id",
            "run_name",
            "created_at_utc",
            "created_unix_s",
            "config_fingerprint_sha256",
            "effective_config_fingerprint_sha256",
            "trainer_environment_sha256",
            "implementation_source",
            "source_authority",
            "runtime_readiness_evidence",
            "seal_provenance",
            "source_checkpoint",
            "shadow_validation",
            "initial_training_state",
            "migration_proof_sha256",
        },
        label="launch contract",
    )
    if payload["schema_version"] != SUPERVISED_LAUNCH_CONTRACT_VERSION:
        raise SupervisedLaunchContractError("unsupported supervised launch contract version")
    _uuid(payload["launch_id"], label="launch_id")
    _text(payload["run_name"], label="run_name")
    _text(payload["created_at_utc"], label="created_at_utc")
    created_unix_s = payload["created_unix_s"]
    if (
        isinstance(created_unix_s, bool)
        or not isinstance(created_unix_s, int | float)
        or not math.isfinite(float(created_unix_s))
        or float(created_unix_s) <= 0.0
    ):
        raise SupervisedLaunchContractError("created_unix_s must be finite and positive")
    _sha256(payload["config_fingerprint_sha256"], label="config_fingerprint_sha256")
    _sha256(
        payload["effective_config_fingerprint_sha256"],
        label="effective_config_fingerprint_sha256",
    )
    _sha256(
        payload["trainer_environment_sha256"],
        label="trainer_environment_sha256",
    )
    _sha256(payload["migration_proof_sha256"], label="migration_proof_sha256")

    implementation_source = _source_identity(
        payload["implementation_source"],
        label="implementation_source",
    )
    payload["implementation_source"] = implementation_source
    payload["source_authority"] = _source_authority(
        payload["source_authority"],
        implementation_source=implementation_source,
    )
    payload["runtime_readiness_evidence"] = _runtime_readiness_evidence(
        payload["runtime_readiness_evidence"],
    )
    payload["seal_provenance"] = _seal_provenance(payload["seal_provenance"])

    source = _object(payload["source_checkpoint"], label="source_checkpoint")
    _exact_fields(
        source,
        {
            "path",
            "checkpoint_id",
            "environment_steps",
            "policy_version",
            "manifest_sha256",
            "metadata_sha256",
            "usage",
        },
        label="source_checkpoint",
    )
    source["path"] = _absolute_path(source["path"], label="source_checkpoint.path")
    _uuid(source["checkpoint_id"], label="source_checkpoint.checkpoint_id")
    _integer(source["environment_steps"], label="source_checkpoint.environment_steps")
    _integer(source["policy_version"], label="source_checkpoint.policy_version")
    _sha256(source["manifest_sha256"], label="source_checkpoint.manifest_sha256")
    _sha256(source["metadata_sha256"], label="source_checkpoint.metadata_sha256")
    if source["usage"] != "model_parameter_initialization_only":
        raise SupervisedLaunchContractError("source_checkpoint.usage must be model_parameter_initialization_only")

    readiness = payload["runtime_readiness_evidence"]
    if not isinstance(readiness, Mapping):  # pragma: no cover - normalized above
        raise RuntimeError("normalized runtime readiness evidence stopped being a mapping")
    binding = readiness["report_binding"]
    if not isinstance(binding, Mapping):  # pragma: no cover - normalized above
        raise RuntimeError("normalized runtime readiness binding stopped being a mapping")
    readiness_config = binding["config"]
    readiness_initialization = binding["initialization"]
    if not isinstance(readiness_config, Mapping) or not isinstance(readiness_initialization, Mapping):
        raise RuntimeError("normalized runtime readiness source/config stopped being mappings")
    if readiness_config["fingerprint_sha256"] != payload["config_fingerprint_sha256"]:
        raise SupervisedLaunchContractError("runtime readiness config differs from launch config")
    if (
        readiness_initialization["checkpoint_path"] != source["path"]
        or readiness_initialization["checkpoint_id"] != source["checkpoint_id"]
        or readiness_initialization["manifest_sha256"] != source["manifest_sha256"]
        or readiness_initialization["metadata_sha256"] != source["metadata_sha256"]
    ):
        raise SupervisedLaunchContractError("runtime readiness source differs from launch source")

    shadow = _object(payload["shadow_validation"], label="shadow_validation")
    _exact_fields(
        shadow,
        {
            "contract_sha256",
            "semantics_report_sha256",
            "evidence_report_sha256",
            "actor_evidence_report_sha256",
            "validated_code_mapping_sha256",
            "evidence_generation_commit",
            "evidence_generation_tree",
        },
        label="shadow_validation",
    )
    for key in (
        "contract_sha256",
        "semantics_report_sha256",
        "evidence_report_sha256",
        "actor_evidence_report_sha256",
        "validated_code_mapping_sha256",
    ):
        _sha256(shadow[key], label=f"shadow_validation.{key}")
    object_format = implementation_source["git_object_format"]
    _git_oid(
        shadow["evidence_generation_commit"],
        label="shadow_validation.evidence_generation_commit",
        object_format=object_format,
    )
    _git_oid(
        shadow["evidence_generation_tree"],
        label="shadow_validation.evidence_generation_tree",
        object_format=object_format,
    )
    source_authority = payload["source_authority"]
    if not isinstance(source_authority, Mapping):  # pragma: no cover - set above
        raise RuntimeError("normalized source authority stopped being a mapping")
    generation = source_authority["evidence_generation_source"]
    if not isinstance(generation, Mapping):  # pragma: no cover - set above
        raise RuntimeError("normalized evidence generation source stopped being a mapping")
    if (
        shadow["validated_code_mapping_sha256"] != source_authority["validated_code_mapping_sha256"]
        or shadow["evidence_generation_commit"] != generation["implementation_commit"]
        or shadow["evidence_generation_tree"] != generation["implementation_tree"]
    ):
        raise SupervisedLaunchContractError("shadow validation and source authority differ")

    initial = _object(payload["initial_training_state"], label="initial_training_state")
    _exact_fields(initial, set(_INITIAL_STATE_FIELDS), label="initial_training_state")
    for field in _INITIAL_STATE_FIELDS:
        if _integer(initial[field], label=f"initial_training_state.{field}") != 0:
            raise SupervisedLaunchContractError(f"initial_training_state.{field} must be zero for model initialization")

    # Copy through canonical JSON so callers cannot mutate aliases retained by
    # a custom JSON decoder or an input mapping.
    normalized = json.loads(json.dumps(payload, allow_nan=False, ensure_ascii=False, sort_keys=True))
    if not isinstance(normalized, dict):  # pragma: no cover - construction invariant
        raise RuntimeError("normalized launch contract stopped being an object")
    return SupervisedLaunchContract(
        path=contract_path,
        sha256=observed_digest,
        payload=normalized,
    )


def _git_stdout(checkout_root: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ("git", "-C", os.fspath(checkout_root), *arguments),
            check=False,
            capture_output=True,
            text=True,
            timeout=30.0,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SupervisedLaunchContractError(f"Git source verification failed: {exc}") from exc
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        raise SupervisedLaunchContractError(
            f"Git source verification failed ({completed.returncode}): {stderr}",
        )
    return completed.stdout


def current_implementation_source(checkout_root: str | Path) -> dict[str, Any]:
    """Return exact clean HEAD authority for the trainer's checkout."""

    root = Path(checkout_root).expanduser()
    if not root.is_absolute():
        raise SupervisedLaunchContractError("implementation checkout_root must be absolute")
    root = root.resolve()
    top_level = Path(_git_stdout(root, "rev-parse", "--show-toplevel").strip()).resolve()
    if top_level != root:
        raise SupervisedLaunchContractError("implementation checkout_root is not the Git toplevel")
    object_format = _git_stdout(root, "rev-parse", "--show-object-format").strip()
    if object_format not in {"sha1", "sha256"}:
        raise SupervisedLaunchContractError("implementation Git object format is unsupported")
    commit = _git_stdout(root, "rev-parse", "HEAD").strip()
    tree = _git_stdout(root, "rev-parse", "HEAD^{tree}").strip()
    dirty = _git_stdout(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=normal",
    )
    if dirty.strip():
        raise SupervisedLaunchContractError("implementation checkout is not clean")
    return _source_identity(
        {
            "checkout_root": os.fspath(root),
            "git_object_format": object_format,
            "implementation_commit": commit,
            "implementation_tree": tree,
            "worktree_clean": True,
        },
        label="live implementation source",
    )


def current_formal_report_generation_source(
    checkout_root: str | Path,
    validator_path: str | Path,
) -> dict[str, Any]:
    """Bind a formal report to one clean Git source and validator file."""

    source = current_implementation_source(checkout_root)
    root = Path(str(source["checkout_root"])).resolve()
    validator = Path(validator_path).expanduser().resolve()
    try:
        relative = validator.relative_to(root).as_posix()
    except ValueError as exc:
        raise SupervisedLaunchContractError("formal report validator escapes the clean checkout") from exc
    if not validator.is_file():
        raise SupervisedLaunchContractError("formal report validator is missing")
    digest = hashlib.sha256(validator.read_bytes()).hexdigest()
    return {
        **source,
        "schema_version": FORMAL_REPORT_GENERATION_SOURCE_VERSION,
        "validator_relative_path": relative,
        "validator_sha256": digest,
    }


def _formal_report_generation_source(value: object, *, label: str) -> dict[str, Any]:
    source = _object(value, label=label)
    _exact_fields(source, _FORMAL_REPORT_GENERATION_SOURCE_FIELDS, label=label)
    if source["schema_version"] != FORMAL_REPORT_GENERATION_SOURCE_VERSION:
        raise SupervisedLaunchContractError(f"{label}.schema_version changed")
    identity = _source_identity(
        {key: source[key] for key in _SOURCE_IDENTITY_FIELDS},
        label=label,
    )
    return {
        **identity,
        "schema_version": FORMAL_REPORT_GENERATION_SOURCE_VERSION,
        "validator_relative_path": _repo_relative_path(
            source["validator_relative_path"],
            label=f"{label}.validator_relative_path",
        ),
        "validator_sha256": _sha256(
            source["validator_sha256"],
            label=f"{label}.validator_sha256",
        ),
    }


def validate_supervised_source_authority(
    contract: SupervisedLaunchContract,
    *,
    observed_implementation_source: Mapping[str, Any] | None = None,
    observed_seal_paths: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Reprove clean launch commit B and the reviewed A->B seal transition."""

    expected_implementation = dict(contract.implementation_source)
    if observed_implementation_source is None:
        observed_implementation = current_implementation_source(
            str(expected_implementation["checkout_root"]),
        )
    else:
        observed_implementation = _source_identity(
            observed_implementation_source,
            label="observed implementation source",
        )
    if observed_implementation != expected_implementation:
        raise SupervisedLaunchContractError(
            "live clean implementation differs from immutable launch contract",
        )

    authority_raw = contract.payload["source_authority"]
    if not isinstance(authority_raw, Mapping):  # pragma: no cover - loader invariant
        raise RuntimeError("validated contract lost source authority")
    authority = dict(authority_raw)
    generation_raw = authority["evidence_generation_source"]
    if not isinstance(generation_raw, Mapping):  # pragma: no cover - loader invariant
        raise RuntimeError("validated contract lost evidence generation source")
    generation = dict(generation_raw)
    root = Path(str(expected_implementation["checkout_root"]))
    if observed_seal_paths is None:
        generation_commit = str(generation["implementation_commit"])
        implementation_commit = str(expected_implementation["implementation_commit"])
        generation_tree = _git_stdout(root, "rev-parse", f"{generation_commit}^{{tree}}").strip()
        if generation_tree != generation["implementation_tree"]:
            raise SupervisedLaunchContractError("evidence generation tree differs from Git")
        ancestor = subprocess.run(
            ("git", "-C", os.fspath(root), "merge-base", "--is-ancestor", generation_commit, implementation_commit),
            check=False,
            capture_output=True,
            timeout=30.0,
        )
        if ancestor.returncode != 0:
            raise SupervisedLaunchContractError("evidence generation commit is not an ancestor of launch commit")
        raw_paths = _git_stdout(
            root,
            "diff",
            "--name-only",
            "--no-renames",
            f"{generation_commit}..{implementation_commit}",
            "--",
        )
        observed_paths = sorted(line for line in raw_paths.splitlines() if line)
    else:
        observed_paths = sorted(_repo_relative_path(item, label="observed seal path") for item in observed_seal_paths)
    if observed_paths != authority["observed_seal_paths"]:
        raise SupervisedLaunchContractError("live A-to-B seal paths differ from immutable authority")

    code_hashes_raw = authority["validated_code_sha256"]
    if not isinstance(code_hashes_raw, Mapping):  # pragma: no cover - loader invariant
        raise RuntimeError("validated contract lost code mapping")
    for relative, expected_digest in code_hashes_raw.items():
        candidate = (root / str(relative)).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:  # pragma: no cover - parser rejects traversal
            raise SupervisedLaunchContractError("validated code path escapes checkout") from exc
        try:
            actual_digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
        except OSError as exc:
            raise SupervisedLaunchContractError(f"validated code is unreadable: {relative}") from exc
        if actual_digest != expected_digest:
            raise SupervisedLaunchContractError(f"validated code differs from evidence generation: {relative}")
    return {
        "implementation_source": observed_implementation,
        "source_authority": authority,
    }


def validate_supervised_runtime_readiness(
    contract: SupervisedLaunchContract,
    *,
    observed_runtime_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Reopen the independent report and bind it to the live ROCm runtime."""

    expected_raw = contract.runtime_readiness_evidence
    expected = dict(expected_raw)
    observed = load_runtime_readiness_evidence(
        str(expected["report_path"]),
        expected_sha256=str(expected["report_sha256"]),
        artifact_root=str(expected["artifact_root"]),
    )
    if observed != expected:
        raise SupervisedLaunchContractError("runtime readiness evidence differs from immutable launch contract")
    binding = observed["report_binding"]
    if not isinstance(binding, Mapping):  # pragma: no cover - loader invariant
        raise RuntimeError("runtime readiness evidence lost report binding")
    expected_runtime = binding["runtime_identity"]
    report_source = binding["generation_source"]
    if not isinstance(report_source, Mapping):  # pragma: no cover - loader invariant
        raise RuntimeError("runtime readiness evidence lost generation source")
    authority = contract.payload["source_authority"]
    if not isinstance(authority, Mapping):  # pragma: no cover - loader invariant
        raise RuntimeError("supervised launch contract lost source authority")
    expected_source = authority["evidence_generation_source"]
    if not isinstance(expected_source, Mapping):  # pragma: no cover - loader invariant
        raise RuntimeError("supervised launch contract lost evidence generation source")
    if {key: report_source[key] for key in _SOURCE_IDENTITY_FIELDS} != dict(expected_source):
        raise SupervisedLaunchContractError("runtime readiness report was generated by another source")
    code_hashes = authority["validated_code_sha256"]
    if not isinstance(code_hashes, Mapping):  # pragma: no cover - loader invariant
        raise RuntimeError("supervised launch contract lost validated code hashes")
    validator_relative = report_source["validator_relative_path"]
    if code_hashes.get(validator_relative) != report_source["validator_sha256"]:
        raise SupervisedLaunchContractError("runtime readiness validator differs from source authority")
    if observed_runtime_identity is None:
        live_runtime = current_rocm_runtime_identity()
    else:
        live_runtime = _runtime_identity(
            observed_runtime_identity,
            label="observed ROCm runtime identity",
        )
    if live_runtime != expected_runtime:
        raise SupervisedLaunchContractError("live ROCm runtime differs from readiness evidence")
    return observed


def checkpoint_source_identity(
    checkpoint: ValidatedResumeCheckpoint,
) -> dict[str, Any]:
    """Return the exact source fields used by a supervised model-init launch."""

    root = checkpoint.root.resolve()
    manifest_path = root / "checkpoint.manifest.json"
    metadata_path = root / "metadata.json"
    try:
        manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        metadata_sha256 = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
    except OSError as exc:  # pragma: no cover - preflight already requires files
        raise CheckpointIntegrityError("validated checkpoint identity files became unreadable") from exc
    checkpoint_id = checkpoint.manifest.get("checkpoint_id")
    if checkpoint_id != checkpoint.metadata.get("checkpoint_id"):
        raise CheckpointIntegrityError("checkpoint manifest/metadata IDs differ")
    training_state = checkpoint.metadata.get("training_state")
    if not isinstance(training_state, Mapping):
        raise CheckpointIntegrityError("checkpoint metadata has no training_state")
    return {
        "path": os.fspath(root),
        "checkpoint_id": checkpoint_id,
        "environment_steps": training_state.get("environment_steps"),
        "policy_version": training_state.get("policy_version"),
        "manifest_sha256": manifest_sha256,
        "metadata_sha256": metadata_sha256,
        "usage": "model_parameter_initialization_only",
    }


def validate_supervised_model_initialization_binding(
    contract: SupervisedLaunchContract,
    *,
    config_fingerprint_sha256: str,
    checkpoint: ValidatedResumeCheckpoint,
) -> dict[str, Any]:
    """Bind a reviewed launch to the config and source opened by the trainer."""

    effective = _sha256(
        config_fingerprint_sha256,
        label="trainer config fingerprint",
    )
    expected_effective = contract.payload["effective_config_fingerprint_sha256"]
    if effective != expected_effective:
        raise SupervisedLaunchContractError("trainer effective config differs from immutable launch contract")
    actual_source = checkpoint_source_identity(checkpoint)
    expected_source = dict(contract.source_checkpoint)
    if actual_source != expected_source:
        raise SupervisedLaunchContractError(
            "trainer model-initialization source differs from immutable launch contract: "
            f"expected={expected_source!r} actual={actual_source!r}"
        )
    return actual_source


__all__ = [
    "EXTERNAL_SEAL_VERSION",
    "FORMAL_REPORT_GENERATION_SOURCE_VERSION",
    "RUNTIME_READINESS_REPORT_VERSION",
    "RUNTIME_READINESS_SEAL_VERSION",
    "SOURCE_AUTHORITY_VERSION",
    "SUPERVISED_LAUNCH_CONTRACT_VERSION",
    "SupervisedLaunchContract",
    "SupervisedLaunchContractError",
    "checkpoint_source_identity",
    "current_formal_report_generation_source",
    "current_implementation_source",
    "current_rocm_runtime_identity",
    "load_runtime_readiness_evidence",
    "load_supervised_launch_contract",
    "trainer_environment_sha256",
    "validate_supervised_model_initialization_binding",
    "validate_supervised_runtime_readiness",
    "validate_supervised_source_authority",
    "validate_supervised_trainer_environment",
]
