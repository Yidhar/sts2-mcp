"""Crash-safe checkpoint directory publication.

Writers save every component to a private sibling directory. Only a complete
staging directory with a manifest is renamed to the public checkpoint name.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from sts2_baseline import (
    revival_efficiency_reward_identity,
    task_reward_identity,
)
from sts2_rl.contracts.versions import (
    ACTION_SCHEMA_VERSION,
    API_VERSION,
    LEGAL_ACTION_ORDERING_VERSION,
    OBSERVATION_SCHEMA_VERSION,
    REWARD_SCHEMA_VERSION,
    SCHEMA_VERSION,
)

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_DEPENDENCY_LOCK_NAMES = (
    "requirements.lock",
    "requirements-dev.lock",
    "requirements-wsl-rocm.txt",
)


class CheckpointIntegrityError(RuntimeError):
    """Published checkpoint does not match its completion manifest."""


def contract_metadata() -> dict[str, str]:
    return {
        "api_version": API_VERSION,
        "schema_version": SCHEMA_VERSION,
        "action_schema_version": ACTION_SCHEMA_VERSION,
        "legal_action_ordering_version": LEGAL_ACTION_ORDERING_VERSION,
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "reward_schema_version": REWARD_SCHEMA_VERSION,
    }


def reward_spec_metadata() -> dict[str, Any]:
    """Return the complete, stable identity of the active reward specification."""

    payload: dict[str, Any] = {
        "version": "sts2-reward-catalog-v4",
        "standard": task_reward_identity(),
        "native_revival_preheat": revival_efficiency_reward_identity(),
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["fingerprint"] = serialized
    payload["fingerprint_sha256"] = hashlib.sha256(
        serialized.encode("utf-8")
    ).hexdigest()
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _git_metadata(root: Path) -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", "-C", str(root), *args],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return result.stdout.strip()

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain", "--untracked-files=normal")
    return {"commit": commit, "tracked_dirty": bool(status) if status is not None else None}


def _hashed_file(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return {
        "path": path.as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _json_object(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict):
        return None
    return {str(key): item for key, item in value.items()}


def game_data_manifest_metadata() -> dict[str, Any] | None:
    """Return optional audit provenance for the static game-data manifest.

    The grounded trainer does not read this catalog, so absence (for example in
    an installed wheel) must not become a delayed checkpoint failure. When a
    valid repository manifest is available, its checkout-independent identity is
    recorded for audit only.
    """

    manifest_path = _repository_root() / "game-data" / "manifest.json"
    descriptor = _hashed_file(manifest_path)
    if descriptor is None:
        return None
    payload = _json_object(manifest_path)
    if payload is None:
        return None
    identity_fields = (
        "schema_version",
        "upstream_sts2_ai_commit",
        "generator",
        "generator_version",
    )
    missing_identity = [
        key
        for key in identity_fields
        if not isinstance(payload.get(key), str) or not str(payload[key]).strip()
    ]
    if missing_identity:
        return None
    return {
        "sha256": descriptor["sha256"],
        "size_bytes": descriptor["size_bytes"],
        "schema_version": payload.get("schema_version"),
        "upstream_sts2_ai_commit": payload.get("upstream_sts2_ai_commit"),
        "generator": payload.get("generator"),
        "generator_version": payload.get("generator_version"),
    }


def dependency_lock_metadata() -> list[dict[str, Any]]:
    """Return checkout-independent identities for every committed RL lock."""

    root = _repository_root()
    result: list[dict[str, Any]] = []
    for name in _DEPENDENCY_LOCK_NAMES:
        descriptor = _hashed_file(root / "packages" / "rl-agent" / name)
        if descriptor is None:
            raise CheckpointIntegrityError(
                f"required RL dependency lock is missing: packages/rl-agent/{name}"
            )
        result.append(
            {
                "path": f"packages/rl-agent/{name}",
                "size_bytes": descriptor["size_bytes"],
                "sha256": descriptor["sha256"],
            }
        )
    return result


def checkpoint_runtime_identity() -> dict[str, Any]:
    """Return every semantic identity that an exact resume must match."""

    return {
        "contract": contract_metadata(),
        "reward_spec": reward_spec_metadata(),
        "dependency_locks": dependency_lock_metadata(),
    }


def build_checkpoint_provenance(
    *,
    parent_checkpoint: str | Path | None = None,
    experiment_run_id: str | None = None,
    config_version: str | None = None,
    config_profile: str | None = None,
    checkpoint_load_mode: str | None = None,
    parent_relation: str | None = None,
) -> dict[str, Any]:
    """Capture reproducibility inputs shared by metadata and atomic manifest."""
    if checkpoint_load_mode not in {"fresh", "exact_resume", "model_initialization"}:
        raise ValueError("checkpoint load mode is missing or unsupported")
    root = _repository_root()
    reward_payload = reward_spec_metadata()
    game_manifest_path = root / "game-data" / "manifest.json"
    game_identity = game_data_manifest_metadata()
    game_manifest = (
        None
        if game_identity is None
        else {"path": game_manifest_path.as_posix(), **game_identity}
    )
    locks = dependency_lock_metadata()
    parent: dict[str, Any] | None = None
    if parent_checkpoint:
        parent_path = Path(parent_checkpoint).expanduser().resolve(strict=False)
        parent = {"path": str(parent_path)}
        relation = parent_relation or "unspecified_parent"
        if relation not in {
            "loaded_parent",
            "model_parameter_initialization",
            "in_process_successor",
            "unspecified_parent",
        }:
            raise ValueError("checkpoint parent relation is missing or unsupported")
        parent["relation"] = relation
        manifest = _hashed_file(parent_path / "checkpoint.manifest.json")
        metadata = _hashed_file(parent_path / "metadata.json")
        parent_manifest_payload = _json_object(parent_path / "checkpoint.manifest.json")
        parent_metadata_payload = _json_object(parent_path / "metadata.json")
        if parent_manifest_payload is not None:
            parent["checkpoint_id"] = parent_manifest_payload.get("checkpoint_id")
        if parent_metadata_payload is not None:
            parent["total_steps"] = parent_metadata_payload.get("total_steps")
            parent["training_state"] = parent_metadata_payload.get("training_state")
            parent_provenance = parent_metadata_payload.get("provenance")
            if isinstance(parent_provenance, dict):
                parent_reward = parent_provenance.get("reward_spec")
                if isinstance(parent_reward, dict):
                    parent["reward_spec_fingerprint"] = parent_reward.get("fingerprint")
        if manifest is not None:
            parent["manifest"] = manifest
        if metadata is not None:
            parent["metadata"] = metadata
    return {
        "provenance_schema_version": "sts2-checkpoint-provenance-v1",
        "runtime": {
            "python": sys.version,
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
        },
        "reward_spec": reward_payload,
        "game_data_manifest": game_manifest,
        "dependency_locks": locks,
        "git": _git_metadata(root),
        "experiment_run_id": experiment_run_id,
        "training_config_version": config_version,
        "training_profile": config_profile,
        "checkpoint_load_mode": checkpoint_load_mode,
        "parent_checkpoint": parent,
    }


def verify_checkpoint_directory(
    checkpoint: str | Path,
    *,
    require_manifest: bool = False,
    require_hashes: bool = False,
    require_all_files_listed: bool = False,
) -> dict[str, Any] | None:
    """Validate the atomic completion manifest and its payload files.

    Diagnostic callers may accept ``None`` for a missing manifest. Grounded
    training resume always sets all three strict flags through
    ``validate_resume_checkpoint``.
    """

    root = Path(checkpoint).resolve(strict=False)
    if not root.is_dir():
        raise CheckpointIntegrityError(f"checkpoint directory does not exist: {root}")
    manifest_path = root / "checkpoint.manifest.json"
    if not manifest_path.is_file():
        if require_manifest:
            raise CheckpointIntegrityError(
                f"atomic checkpoint manifest is required for exact resume: {manifest_path}"
            )
        return None
    try:
        raw_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CheckpointIntegrityError(
            f"checkpoint manifest is unreadable or invalid JSON in {root}"
        ) from exc
    if not isinstance(raw_payload, dict):
        raise CheckpointIntegrityError(f"checkpoint manifest must be an object in {root}")
    payload: dict[str, Any] = {str(key): value for key, value in raw_payload.items()}
    if payload.get("format") != "sts2-atomic-checkpoint-v1":
        raise CheckpointIntegrityError(f"unsupported checkpoint manifest in {root}")
    if require_hashes and payload.get("hash_files") is not True:
        raise CheckpointIntegrityError(
            f"exact resume requires hash_files=true in checkpoint manifest: {root}"
        )
    raw_files = payload.get("files")
    if not isinstance(raw_files, list):
        raise CheckpointIntegrityError(f"checkpoint files must be a list in {root}")
    listed_paths: set[str] = set()
    for entry in raw_files:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise CheckpointIntegrityError(f"invalid checkpoint file entry in {root}")
        relative_path = entry["path"]
        if relative_path in listed_paths:
            raise CheckpointIntegrityError(
                f"duplicate checkpoint file entry: {relative_path}"
            )
        listed_paths.add(relative_path)
        candidate_path = root / relative_path
        if candidate_path.is_symlink():
            raise CheckpointIntegrityError(
                f"checkpoint payload must not be a symlink: {relative_path}"
            )
        file_path = candidate_path.resolve(strict=False)
        try:
            file_path.relative_to(root)
        except ValueError as exc:
            raise CheckpointIntegrityError(
                f"checkpoint path escapes root: {relative_path}"
            ) from exc
        if not file_path.is_file():
            raise CheckpointIntegrityError(f"checkpoint file missing: {relative_path}")
        actual_size = file_path.stat().st_size
        expected_size = entry.get("size_bytes")
        if (
            not isinstance(expected_size, int)
            or isinstance(expected_size, bool)
            or expected_size < 0
        ):
            raise CheckpointIntegrityError(
                f"invalid checkpoint size for {relative_path}"
            )
        if actual_size != expected_size:
            raise CheckpointIntegrityError(
                f"checkpoint size mismatch for {relative_path}: "
                f"expected={entry.get('size_bytes')} actual={actual_size}"
            )
        expected_hash = entry.get("sha256")
        if require_hashes and not (
            isinstance(expected_hash, str) and _SHA256_PATTERN.fullmatch(expected_hash)
        ):
            raise CheckpointIntegrityError(
                f"valid SHA-256 is required for checkpoint file: {relative_path}"
            )
        if expected_hash is not None:
            if not (
                isinstance(expected_hash, str)
                and _SHA256_PATTERN.fullmatch(expected_hash)
            ):
                raise CheckpointIntegrityError(
                    f"invalid checkpoint SHA-256 for {relative_path}"
                )
            if _sha256(file_path) != expected_hash:
                raise CheckpointIntegrityError(
                    f"checkpoint hash mismatch for {relative_path}"
                )
    if require_all_files_listed:
        actual_paths: set[str] = set()
        for file_path in root.rglob("*"):
            if file_path.is_symlink():
                raise CheckpointIntegrityError(
                    f"checkpoint tree must not contain symlinks: {file_path.relative_to(root)}"
                )
            if file_path.is_file() and file_path != manifest_path:
                actual_paths.add(file_path.relative_to(root).as_posix())
        unlisted = sorted(actual_paths - listed_paths)
        stale = sorted(listed_paths - actual_paths)
        if unlisted or stale:
            raise CheckpointIntegrityError(
                "checkpoint manifest/file set mismatch: "
                f"unlisted={unlisted[:8]} stale={stale[:8]}"
            )
    return payload


class AtomicCheckpointDirectory:
    """Two-phase publisher for one checkpoint directory."""

    def __init__(
        self,
        target: str | Path,
        *,
        hash_files: bool | None = None,
        provenance: dict[str, Any] | None = None,
    ) -> None:
        self.target = Path(target)
        self.staging = self.target.with_name(f".{self.target.name}.incomplete-{uuid4().hex}")
        self.hash_files = True if hash_files is None else bool(hash_files)
        self.provenance = dict(provenance or {})
        self.checkpoint_id = str(uuid4())
        self._prepared = False
        self._committed = False

    def prepare(self) -> Path:
        if self._prepared:
            return self.staging
        if self.target.exists():
            raise FileExistsError(
                f"published checkpoints are immutable and already exist: {self.target}"
            )
        self.target.parent.mkdir(parents=True, exist_ok=True)
        self.staging.mkdir(parents=False, exist_ok=False)
        self._prepared = True
        return self.staging

    def _manifest(self) -> dict[str, Any]:
        files: list[dict[str, Any]] = []
        for path in sorted(self.staging.rglob("*")):
            if not path.is_file() or path.name == "checkpoint.manifest.json":
                continue
            entry: dict[str, Any] = {
                "path": path.relative_to(self.staging).as_posix(),
                "size_bytes": path.stat().st_size,
            }
            if self.hash_files:
                entry["sha256"] = _sha256(path)
            files.append(entry)
        return {
            "format": "sts2-atomic-checkpoint-v1",
            "created_unix_s": time.time(),
            "checkpoint_id": self.checkpoint_id,
            "contract": contract_metadata(),
            "provenance": self.provenance,
            "hash_files": self.hash_files,
            "files": files,
        }

    def commit(self) -> Path:
        if self._committed:
            return self.target
        if not self._prepared:
            raise RuntimeError("prepare() must be called before commit()")

        if self.target.exists():
            raise FileExistsError(
                f"published checkpoints are immutable and already exist: {self.target}"
            )

        for payload_path in sorted(self.staging.rglob("*")):
            if not payload_path.is_file():
                continue
            with payload_path.open("r+b") as handle:
                os.fsync(handle.fileno())

        manifest_path = self.staging / "checkpoint.manifest.json"
        manifest_path.write_text(
            json.dumps(self._manifest(), indent=2, sort_keys=True) + os.linesep,
            encoding="utf-8",
        )
        with manifest_path.open("r+b") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        self._fsync_directory(self.staging)
        os.rename(self.staging, self.target)
        self._fsync_directory(self.target.parent)
        self._committed = True
        return self.target

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        """Best-effort directory durability on platforms that expose dir FDs."""

        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        try:
            descriptor = os.open(path, flags)
        except OSError:
            return
        try:
            os.fsync(descriptor)
        except OSError:
            pass
        finally:
            os.close(descriptor)

    def abort(self) -> None:
        if self.staging.exists():
            shutil.rmtree(self.staging)
        self._prepared = False

    def __enter__(self) -> Path:
        return self.prepare()

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if exc_type is None:
            self.commit()
        else:
            self.abort()
