"""Fail-closed validation for exact checkpoint resume.

This module intentionally performs only JSON, filesystem, and SHA-256 work. It
can therefore establish checkpoint identity before a caller invokes
``torch.load`` or unpickles replay state.
"""

from __future__ import annotations

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
    {"metadata.json", "network.pt", "optimizer.pt", "replay_buffer.pkl"}
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


def _game_data_identity(value: Any, *, label: str) -> dict[str, Any]:
    payload = _object(value, label=label)
    identity_keys = (
        "sha256",
        "size_bytes",
        "schema_version",
        "upstream_sts2_ai_commit",
        "generator",
        "generator_version",
    )
    missing = [key for key in identity_keys if key not in payload]
    if missing:
        raise CheckpointIntegrityError(f"{label} is missing identity fields: {missing}")
    return {key: payload[key] for key in identity_keys}


def _validate_semantic_identity(
    *,
    manifest: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    expected = checkpoint_runtime_identity()
    expected_contract = expected["contract"]
    expected_reward = expected["reward_spec"]
    expected_game_data = expected["game_data_manifest"]

    manifest_contract = _object(manifest.get("contract"), label="manifest.contract")
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
    _require_equal(manifest_reward, expected_reward, label="reward identity")
    manifest_game_data = _game_data_identity(
        manifest_provenance.get("game_data_manifest"),
        label="manifest.provenance.game_data_manifest",
    )
    _require_equal(manifest_game_data, expected_game_data, label="game-data identity")

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


def validate_resume_checkpoint(
    checkpoint: str | Path,
    *,
    required_files: Collection[str] = EXACT_RESUME_REQUIRED_FILES,
) -> ValidatedResumeCheckpoint:
    """Validate an exact-resume checkpoint without deserializing executable data.

    Exact resume requires an atomic completion manifest, a valid SHA-256 for
    every payload file, no unlisted payload, and exact contract/reward/game-data
    identity. Any missing or incompatible identity fails closed.
    """

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
            f"exact resume checkpoint is missing required manifest entries: {missing}"
        )

    metadata = _read_json_object(root / "metadata.json", label="checkpoint metadata")
    _validate_semantic_identity(manifest=manifest, metadata=metadata)
    return ValidatedResumeCheckpoint(root=root, manifest=manifest, metadata=metadata)


def validate_hashed_warm_start_checkpoint(
    checkpoint: str | Path,
) -> tuple[Path, dict[str, Any] | None, dict[str, Any]]:
    """Validate bytes for an explicitly requested weights-only migration.

    Atomic checkpoints still require a complete all-file hash manifest. A legacy
    checkpoint is represented by ``manifest=None`` and must be explicitly allowed
    by the higher-level warm-start caller.
    """

    root = Path(checkpoint).expanduser().resolve(strict=False)
    manifest_path = root / "checkpoint.manifest.json"
    manifest: dict[str, Any] | None
    if manifest_path.is_file():
        manifest = verify_checkpoint_directory(
            root,
            require_manifest=True,
            require_hashes=True,
            require_all_files_listed=True,
        )
    else:
        if not root.is_dir():
            raise CheckpointIntegrityError(f"checkpoint directory does not exist: {root}")
        manifest = None
    metadata_path = root / "metadata.json"
    if metadata_path.is_symlink():
        raise CheckpointIntegrityError(
            f"warm-start metadata.json must not be a symlink: {metadata_path}"
        )
    metadata = (
        _read_json_object(metadata_path, label="checkpoint metadata")
        if metadata_path.is_file()
        else {}
    )
    return root, manifest, metadata
