"""Fail-closed validation for exact checkpoint resume.

This module intentionally performs only JSON, filesystem, and SHA-256 work. It
can therefore establish checkpoint identity before a caller invokes
``torch.load`` or unpickles rollout-queue state.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID

from .atomic import (
    CheckpointIntegrityError,
    checkpoint_runtime_identity,
    verify_checkpoint_directory,
)

EXACT_RESUME_REQUIRED_FILES = frozenset(
    {
        "metadata.json",
        "network.pt",
        "actor_network.pt",
        "optimizer.pt",
        "rollout_queue.pkl",
        "stochastic_state.pkl",
    }
)


@dataclass(frozen=True, slots=True)
class ValidatedResumeCheckpoint:
    """A checkpoint whose bytes and semantic identity have been validated."""

    root: Path
    manifest: dict[str, Any]
    metadata: dict[str, Any]


def _object(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CheckpointIntegrityError(f"{label} must be a JSON object")
    return {str(key): item for key, item in value.items()}


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CheckpointIntegrityError(f"{label} is unreadable or invalid JSON: {path}") from exc
    return _object(raw, label=label)


def _require_equal(actual: Any, expected: Any, *, label: str) -> None:
    if actual != expected:
        raise CheckpointIntegrityError(
            f"checkpoint {label} mismatch: expected={expected!r} actual={actual!r}"
        )


def _require_sha256(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise CheckpointIntegrityError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _dependency_lock_identity(value: Any, *, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise CheckpointIntegrityError(f"{label} must be a non-empty list")
    result: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for index, item in enumerate(value):
        payload = _object(item, label=f"{label}[{index}]")
        if set(payload) != {"path", "size_bytes", "sha256"}:
            raise CheckpointIntegrityError(
                f"{label}[{index}] has unsupported identity fields"
            )
        path = payload.get("path")
        size_bytes = payload.get("size_bytes")
        if not isinstance(path, str) or not path.strip():
            raise CheckpointIntegrityError(f"{label}[{index}].path must be non-empty text")
        if path in seen_paths:
            raise CheckpointIntegrityError(f"{label} contains duplicate path: {path}")
        if (
            isinstance(size_bytes, bool)
            or not isinstance(size_bytes, int)
            or size_bytes < 0
        ):
            raise CheckpointIntegrityError(
                f"{label}[{index}].size_bytes must be non-negative"
            )
        sha256 = _require_sha256(
            payload.get("sha256"),
            label=f"{label}[{index}].sha256",
        )
        seen_paths.add(path)
        result.append({"path": path, "size_bytes": size_bytes, "sha256": sha256})
    return result


def _validate_semantic_identity(
    *,
    manifest: dict[str, Any],
    metadata: dict[str, Any],
    require_current_runtime_identity: bool,
) -> None:
    expected = checkpoint_runtime_identity()
    expected_contract = expected["contract"]
    expected_reward = expected["reward_spec"]
    expected_locks = expected["dependency_locks"]

    manifest_contract = _object(manifest.get("contract"), label="manifest.contract")
    if set(manifest_contract) != set(expected_contract) or not all(
        isinstance(value, str) and value.strip() for value in manifest_contract.values()
    ):
        raise CheckpointIntegrityError(
            "checkpoint manifest.contract has an unsupported identity shape"
        )
    if require_current_runtime_identity:
        _require_equal(manifest_contract, expected_contract, label="contract identity")

    manifest_provenance = _object(
        manifest.get("provenance"), label="manifest.provenance"
    )
    if manifest_provenance.get("provenance_schema_version") != "sts2-checkpoint-provenance-v1":
        raise CheckpointIntegrityError(
            "checkpoint provenance_schema_version is missing or unsupported"
        )
    manifest_reward = _object(
        manifest_provenance.get("reward_spec"),
        label="manifest.provenance.reward_spec",
    )
    if not isinstance(manifest_reward.get("fingerprint"), str) or not str(
        manifest_reward["fingerprint"]
    ).strip():
        raise CheckpointIntegrityError(
            "manifest.provenance.reward_spec has no fingerprint"
        )
    reward_fingerprint_sha256 = _require_sha256(
        manifest_reward.get("fingerprint_sha256"),
        label="manifest.provenance.reward_spec.fingerprint_sha256",
    )
    if hashlib.sha256(manifest_reward["fingerprint"].encode("utf-8")).hexdigest() != (
        reward_fingerprint_sha256
    ):
        raise CheckpointIntegrityError(
            "manifest.provenance.reward_spec fingerprint digest does not match"
        )
    if require_current_runtime_identity:
        _require_equal(manifest_reward, expected_reward, label="reward identity")
    manifest_locks = _dependency_lock_identity(
        manifest_provenance.get("dependency_locks"),
        label="manifest.provenance.dependency_locks",
    )
    if require_current_runtime_identity:
        _require_equal(manifest_locks, expected_locks, label="dependency-lock identity")

    metadata_contract = _object(metadata.get("contract"), label="metadata.contract")
    _require_equal(metadata_contract, manifest_contract, label="metadata contract identity")
    metadata_provenance = _object(
        metadata.get("provenance"), label="metadata.provenance"
    )
    _require_equal(
        metadata_provenance,
        manifest_provenance,
        label="metadata/manifest provenance",
    )

    checkpoint_id = manifest.get("checkpoint_id")
    if not isinstance(checkpoint_id, str) or not checkpoint_id.strip():
        raise CheckpointIntegrityError("checkpoint manifest has no checkpoint_id")
    try:
        UUID(checkpoint_id)
    except ValueError as exc:
        raise CheckpointIntegrityError(
            f"checkpoint manifest has invalid checkpoint_id: {checkpoint_id!r}"
        ) from exc
    _require_equal(
        metadata.get("checkpoint_id"),
        checkpoint_id,
        label="metadata checkpoint_id",
    )


def _validate_checkpoint(
    checkpoint: str | Path,
    *,
    required_files: Collection[str],
    require_current_runtime_identity: bool,
    operation: str,
) -> ValidatedResumeCheckpoint:
    root = Path(checkpoint).expanduser().resolve(strict=False)
    manifest = verify_checkpoint_directory(
        root,
        require_manifest=True,
        require_hashes=True,
        require_all_files_listed=True,
    )
    if manifest is None:  # pragma: no cover - strict verifier cannot return None
        raise CheckpointIntegrityError(f"atomic checkpoint manifest is required: {root}")

    raw_entries = manifest.get("files")
    entries = raw_entries if isinstance(raw_entries, list) else []
    listed = {
        str(entry.get("path"))
        for entry in entries
        if isinstance(entry, dict) and isinstance(entry.get("path"), str)
    }
    missing = sorted(set(required_files) - listed)
    if missing:
        raise CheckpointIntegrityError(
            f"{operation} checkpoint is missing required manifest entries: {missing}"
        )

    metadata = _read_json_object(root / "metadata.json", label="checkpoint metadata")
    _validate_semantic_identity(
        manifest=manifest,
        metadata=metadata,
        require_current_runtime_identity=require_current_runtime_identity,
    )
    return ValidatedResumeCheckpoint(root=root, manifest=manifest, metadata=metadata)


def validate_resume_checkpoint(
    checkpoint: str | Path,
    *,
    required_files: Collection[str] = EXACT_RESUME_REQUIRED_FILES,
) -> ValidatedResumeCheckpoint:
    """Validate an exact-resume checkpoint without deserializing executable data.

    Exact resume requires an atomic completion manifest, a valid SHA-256 for
    every payload file, no unlisted payload, and exact contract/reward/dependency
    identity.
    """

    return _validate_checkpoint(
        checkpoint,
        required_files=required_files,
        require_current_runtime_identity=True,
        operation="exact resume",
    )


def validate_model_initialization_checkpoint(
    checkpoint: str | Path,
    *,
    required_files: Collection[str] = EXACT_RESUME_REQUIRED_FILES,
) -> ValidatedResumeCheckpoint:
    """Validate a complete model-only source without current task semantics.

    The source remains an immutable atomic training checkpoint. Its recorded
    contract, reward, and dependency identities must be structurally valid and
    internally identical between manifest and metadata, but their values may
    predate the active runtime. Learned model and encoding ABI checks run before
    any tensor is loaded. Exact resume never calls this migration path.
    """

    return _validate_checkpoint(
        checkpoint,
        required_files=required_files,
        require_current_runtime_identity=False,
        operation="model initialization",
    )
