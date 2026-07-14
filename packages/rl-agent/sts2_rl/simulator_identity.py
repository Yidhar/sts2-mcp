"""Fail-closed provenance gate for the pinned HeadlessSim executable.

The simulator is built from a separately restored repository, so its path or
assembly version alone cannot prove which source produced it.  A verified
build writes a sidecar identity next to the executable.  Formal headless
training accepts the executable only when the sidecar agrees with the
repository lock, the native apphost, and the managed assembly that contains
the simulator implementation.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sts2_rl.artifacts import resolve_artifact_path

IDENTITY_SCHEMA_VERSION = "1.2.0"
IDENTITY_SUFFIX = ".identity.json"
REQUIRED_BUILD_CONFIGURATION = "Release"
REQUIRED_TARGET_FRAMEWORK = "net9.0"


class SimulatorIdentityError(RuntimeError):
    """The simulator executable has no trustworthy pinned-source identity."""


@dataclass(frozen=True, slots=True)
class VerifiedSimulatorIdentity:
    executable: Path
    managed_assembly: Path
    identity_path: Path
    binary_sha256: str
    binary_size_bytes: int
    managed_assembly_sha256: str
    managed_assembly_size_bytes: int
    source_commit: str
    source_tree: str
    source_url: str
    source_project: str
    build_configuration: str
    target_framework: str
    dotnet_sdk: str

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": IDENTITY_SCHEMA_VERSION,
            "component": "HeadlessSim",
            "verification": "lock-apphost-and-managed-assembly-sha256",
            "executable": str(self.executable),
            "managed_assembly": str(self.managed_assembly),
            "identity_path": str(self.identity_path),
            "binary": {
                "sha256": self.binary_sha256,
                "size_bytes": self.binary_size_bytes,
            },
            "managed_binary": {
                "sha256": self.managed_assembly_sha256,
                "size_bytes": self.managed_assembly_size_bytes,
            },
            "source": {
                "url": self.source_url,
                "commit": self.source_commit,
                "tree": self.source_tree,
                "project": self.source_project,
            },
            "build": {
                "configuration": self.build_configuration,
                "target_framework": self.target_framework,
                "dotnet_sdk": self.dotnet_sdk,
            },
        }


def repository_root() -> Path:
    module = Path(__file__).resolve()
    for parent in module.parents:
        if (parent / "third_party" / "sts2-ai.lock.json").is_file():
            return parent
    raise SimulatorIdentityError(
        "could not locate third_party/sts2-ai.lock.json from the installed trainer"
    )


def load_sts2_ai_lock(path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    lock_path = Path(path).resolve() if path is not None else repository_root() / "third_party" / "sts2-ai.lock.json"
    try:
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SimulatorIdentityError(f"cannot read simulator source lock {lock_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SimulatorIdentityError(f"simulator source lock must be an object: {lock_path}")
    required = ("url", "commit", "tree", "canonical_headless_project")
    missing = [key for key in required if not isinstance(payload.get(key), str) or not str(payload[key]).strip()]
    if missing:
        raise SimulatorIdentityError(f"simulator source lock is missing fields {missing}: {lock_path}")
    if len(str(payload["commit"])) != 40 or len(str(payload["tree"])) != 40:
        raise SimulatorIdentityError(f"simulator source lock commit/tree must be full 40-character Git identities: {lock_path}")
    return payload


def _patch_records(lock: dict[str, Any]) -> list[dict[str, str]]:
    raw = lock.get("patches", [])
    if not isinstance(raw, list):
        raise SimulatorIdentityError("simulator source lock patches must be an array")
    root = repository_root()
    records: list[dict[str, str]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise SimulatorIdentityError(f"simulator source patch {index} must be an object")
        relative = str(item.get("path") or "")
        expected = str(item.get("sha256") or "").lower()
        path = (root / relative).resolve(strict=False)
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise SimulatorIdentityError(
                f"simulator source patch escapes repository: {relative!r}"
            ) from exc
        if not path.is_file() or sha256_file(path) != expected:
            raise SimulatorIdentityError(
                f"simulator source patch is missing or has the wrong hash: {relative}"
            )
        records.append({"path": relative, "sha256": expected})
    return records


def simulator_identity_path(executable: str | os.PathLike[str]) -> Path:
    path = Path(executable).expanduser().resolve(strict=False)
    return path.with_name(path.name + IDENTITY_SUFFIX)


def simulator_managed_assembly_path(executable: str | os.PathLike[str]) -> Path:
    """Return the managed DLL that carries the HeadlessSim implementation."""

    path = Path(executable).expanduser().resolve(strict=False)
    return path.with_suffix(".dll")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise SimulatorIdentityError(f"cannot hash simulator executable {path}: {exc}") from exc
    return digest.hexdigest()


def _mapping(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SimulatorIdentityError(f"simulator identity {label} must be an object")
    return value


def _text(mapping: dict[str, Any], key: str, *, label: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise SimulatorIdentityError(f"simulator identity {label}.{key} must be non-empty text")
    return value


def verify_headless_simulator(
    executable: str | os.PathLike[str],
    *,
    identity_path: str | os.PathLike[str] | None = None,
    lock_path: str | os.PathLike[str] | None = None,
) -> VerifiedSimulatorIdentity:
    """Verify pinned source provenance and exact apphost/assembly bytes.

    No path-based or product-version fallback is accepted.  Tests that use a
    fake in-memory backend remain unaffected because they do not launch a real
    simulator executable.
    """

    exe = Path(executable).expanduser().resolve(strict=False)
    if not exe.is_file():
        raise SimulatorIdentityError(f"HeadlessSim executable does not exist: {exe}")
    managed_assembly = simulator_managed_assembly_path(exe)
    if not managed_assembly.is_file():
        raise SimulatorIdentityError(
            f"HeadlessSim managed assembly does not exist: {managed_assembly}"
        )
    sidecar = (
        Path(identity_path).expanduser().resolve(strict=False)
        if identity_path is not None
        else simulator_identity_path(exe)
    )
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SimulatorIdentityError(
            f"HeadlessSim identity sidecar is missing: {sidecar}. Rebuild it with "
            "scripts/build_pinned_headless_sim.py; unverified binaries are refused."
        ) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise SimulatorIdentityError(f"cannot read HeadlessSim identity {sidecar}: {exc}") from exc
    root = _mapping(payload, label="root")
    if root.get("schema_version") != IDENTITY_SCHEMA_VERSION:
        raise SimulatorIdentityError(
            f"unsupported simulator identity schema {root.get('schema_version')!r}; expected {IDENTITY_SCHEMA_VERSION!r}"
        )
    if root.get("component") != "HeadlessSim":
        raise SimulatorIdentityError("simulator identity component must be 'HeadlessSim'")

    source = _mapping(root.get("source"), label="source")
    binary = _mapping(root.get("binary"), label="binary")
    managed_binary = _mapping(root.get("managed_binary"), label="managed_binary")
    build = _mapping(root.get("build"), label="build")
    lock = load_sts2_ai_lock(lock_path)
    expected_source = {
        "url": str(lock["url"]),
        "commit": str(lock["commit"]),
        "tree": str(lock["tree"]),
        "project": str(lock["canonical_headless_project"]),
    }
    for key, expected in expected_source.items():
        actual = _text(source, key, label="source")
        if actual != expected:
            raise SimulatorIdentityError(
                f"HeadlessSim source {key} mismatch: identity={actual!r}, lock={expected!r}"
            )

    expected_patches = _patch_records(lock)
    recorded_patches = source.get("patches", [])
    if recorded_patches != expected_patches:
        raise SimulatorIdentityError(
            "HeadlessSim source patch set differs from the locked curriculum patch set"
        )

    recorded_name = _text(binary, "file_name", label="binary")
    if recorded_name != exe.name:
        raise SimulatorIdentityError(
            f"HeadlessSim file-name mismatch: identity={recorded_name!r}, executable={exe.name!r}"
        )
    recorded_hash = _text(binary, "sha256", label="binary").lower()
    if len(recorded_hash) != 64 or any(char not in "0123456789abcdef" for char in recorded_hash):
        raise SimulatorIdentityError("simulator identity binary.sha256 must be a lowercase SHA-256 digest")
    recorded_size = binary.get("size_bytes")
    if isinstance(recorded_size, bool) or not isinstance(recorded_size, int) or recorded_size < 1:
        raise SimulatorIdentityError("simulator identity binary.size_bytes must be a positive integer")
    actual_size = exe.stat().st_size
    if actual_size != recorded_size:
        raise SimulatorIdentityError(
            f"HeadlessSim size mismatch: identity={recorded_size}, executable={actual_size}"
        )
    actual_hash = sha256_file(exe)
    if actual_hash != recorded_hash:
        raise SimulatorIdentityError(
            f"HeadlessSim SHA-256 mismatch: identity={recorded_hash}, executable={actual_hash}"
        )

    recorded_managed_name = _text(managed_binary, "file_name", label="managed_binary")
    if recorded_managed_name != managed_assembly.name:
        raise SimulatorIdentityError(
            "HeadlessSim managed assembly file-name mismatch: "
            f"identity={recorded_managed_name!r}, assembly={managed_assembly.name!r}"
        )
    recorded_managed_hash = _text(
        managed_binary,
        "sha256",
        label="managed_binary",
    ).lower()
    if len(recorded_managed_hash) != 64 or any(
        char not in "0123456789abcdef" for char in recorded_managed_hash
    ):
        raise SimulatorIdentityError(
            "simulator identity managed_binary.sha256 must be a lowercase SHA-256 digest"
        )
    recorded_managed_size = managed_binary.get("size_bytes")
    if (
        isinstance(recorded_managed_size, bool)
        or not isinstance(recorded_managed_size, int)
        or recorded_managed_size < 1
    ):
        raise SimulatorIdentityError(
            "simulator identity managed_binary.size_bytes must be a positive integer"
        )
    actual_managed_size = managed_assembly.stat().st_size
    if actual_managed_size != recorded_managed_size:
        raise SimulatorIdentityError(
            "HeadlessSim managed assembly size mismatch: "
            f"identity={recorded_managed_size}, assembly={actual_managed_size}"
        )
    actual_managed_hash = sha256_file(managed_assembly)
    if actual_managed_hash != recorded_managed_hash:
        raise SimulatorIdentityError(
            "HeadlessSim managed assembly SHA-256 mismatch: "
            f"identity={recorded_managed_hash}, assembly={actual_managed_hash}"
        )

    configuration = _text(build, "configuration", label="build")
    target_framework = _text(build, "target_framework", label="build")
    dotnet_sdk = _text(build, "dotnet_sdk", label="build")
    if configuration != REQUIRED_BUILD_CONFIGURATION:
        raise SimulatorIdentityError(
            f"HeadlessSim build configuration must be {REQUIRED_BUILD_CONFIGURATION!r}, got {configuration!r}"
        )
    if target_framework != REQUIRED_TARGET_FRAMEWORK:
        raise SimulatorIdentityError(
            f"HeadlessSim target framework must be {REQUIRED_TARGET_FRAMEWORK!r}, got {target_framework!r}"
        )
    return VerifiedSimulatorIdentity(
        executable=exe,
        managed_assembly=managed_assembly,
        identity_path=sidecar,
        binary_sha256=actual_hash,
        binary_size_bytes=actual_size,
        managed_assembly_sha256=actual_managed_hash,
        managed_assembly_size_bytes=actual_managed_size,
        source_commit=expected_source["commit"],
        source_tree=expected_source["tree"],
        source_url=expected_source["url"],
        source_project=expected_source["project"],
        build_configuration=configuration,
        target_framework=target_framework,
        dotnet_sdk=dotnet_sdk,
    )


def write_preflight_audit(identity: VerifiedSimulatorIdentity) -> Path:
    """Persist an immutable audit record outside the source checkout."""

    directory = resolve_artifact_path("logs/simulator-preflight")
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    destination = directory / f"{timestamp}-{identity.binary_sha256[:16]}.json"
    payload = {
        "event": "simulator_identity_verified",
        "verified_at_utc": datetime.now(UTC).isoformat(),
        **identity.to_mapping(),
    }
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    fd, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=directory)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


__all__ = [
    "IDENTITY_SCHEMA_VERSION",
    "IDENTITY_SUFFIX",
    "REQUIRED_BUILD_CONFIGURATION",
    "REQUIRED_TARGET_FRAMEWORK",
    "SimulatorIdentityError",
    "VerifiedSimulatorIdentity",
    "load_sts2_ai_lock",
    "repository_root",
    "sha256_file",
    "simulator_identity_path",
    "simulator_managed_assembly_path",
    "verify_headless_simulator",
    "write_preflight_audit",
]
