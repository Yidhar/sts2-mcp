#!/usr/bin/env python3
"""Retired, fail-closed WSL supervisor for the reviewed v28 refinement.

Historically the public ``start`` action launched exactly one training
command: a ``model_initialization`` from the frozen v27 200,044-step
checkpoint selected by the paired held-out evaluation.  The reviewed v28
100k result is now frozen as the v29 model ancestor, so a source-tree
retirement contract makes every new ``start`` fail closed.  ``status`` and
watchdog reconciliation remain available for audit of the completed lineage.

``start`` detaches a Linux supervisor, not a bare trainer.  The supervisor
records native process identities for itself and its child, binds the newly
created v28 ``metrics.jsonl`` by its model-initialization ``run_start``, and
turns every observed child exit into a durable terminal lifecycle.  Therefore
``status`` can reconcile an interrupted supervisor instead of leaving a dead
run as ``stale_unknown``.

This file deliberately has no stop command.  It must be invoked inside WSL::

    python3 scripts/launch_v28_mature_refinement.py preflight
    python3 scripts/launch_v28_mature_refinement.py start
    python3 scripts/launch_v28_mature_refinement.py status
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
import uuid
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:  # pragma: no cover - Windows imports this module for unit tests.
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]

# ``scripts`` is intentionally not a Python package.  Keep the adjacent,
# non-launching preflight as the single authority for command/config checks.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import preflight_v28_mature_refinement as v28_preflight  # noqa: E402

MANIFEST_SCHEMA = "sts2-v28-model-init-supervised-launch-v1"
STATE_SCHEMA = "sts2-v28-model-init-supervised-state-v1"
WATCHDOG_EVENT_SCHEMA = "sts2-native-exit-watchdog-v1"
RUN_NAME = v28_preflight.RUN_NAME

FIXED_SOURCE_RUN_ID = "c0163601-9e0b-49c0-812e-fe89166eeba4"
FIXED_INITIALIZATION_STEP = 200_044
FIXED_INITIALIZATION_POLICY_VERSION = 3_142
FIXED_INITIALIZATION_CHECKPOINT_ID = "ef047c18-e854-40d8-87e3-6821ab6f6a61"
FIXED_INITIALIZATION_MANIFEST_SHA256 = "607ed1aabde4da622dc4ce407f257d48415336d87cc35c2fb6b0d895f58af21d"
FIXED_INITIALIZATION_METADATA_SHA256 = "8920f8ec8c2acd7a1a2e4b0efef1eb03f9e66c3f3548b4109e3583881dbbce9f"
FIXED_CONFIG_FINGERPRINT_SHA256 = "8bea2a47474f05ec9736ee8ec0d633b036483a220fcf77a855890ea77b2c17e2"
# ``run_start`` persists the effective config *after* CLI device/backend/sim
# overrides, so its fingerprint intentionally differs from the file-only
# preflight fingerprint above.
FIXED_RUNTIME_CONFIG_FINGERPRINT_SHA256 = "a883037d1c6e9a73e4c7d97063b491702c4ea3a55880587ca8f485f259adf7a8"
FIXED_INITIALIZATION_RELATIVE = Path(
    "checkpoints/full-run-revival-v27-infinite-random-init"
    f"/run-{FIXED_SOURCE_RUN_ID}/periodic-step-{FIXED_INITIALIZATION_STEP:09d}"
)
# The reviewed v28 policy is now a frozen model-initialization ancestor for
# v29.  Keeping this retirement marker in the source tree prevents the old
# supervised recipe from silently creating another v28 lineage after the
# operator has declared the 100k result final.
FROZEN_V28_CONTRACT_RELATIVE = Path("contracts/frozen-checkpoints/v28-mature-refinement-100k.json")
TERMINAL_STATUSES = frozenset({"completed", "interrupted", "failed"})


class LaunchError(RuntimeError):
    """The reviewed v28 launch contract could not be proven."""


@dataclass(frozen=True, slots=True)
class LaunchPaths:
    preflight: v28_preflight.PreflightPaths
    launcher_dir: Path
    manifest_dir: Path
    initialization_checkpoint: Path

    @property
    def checkout_root(self) -> Path:
        return self.preflight.checkout_root

    @property
    def package_root(self) -> Path:
        return self.preflight.package_root

    @property
    def artifact_root(self) -> Path:
        return self.preflight.artifact_root

    @property
    def venv_python(self) -> Path:
        return self.preflight.venv_python


@dataclass(frozen=True, slots=True)
class ProcessIdentity:
    pid: int
    proc_start_ticks: int
    command_line_sha256: str
    executable: str


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _absolute_without_symlink_resolution(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def default_paths() -> LaunchPaths:
    preflight = v28_preflight.default_paths()
    checkpoint = preflight.artifact_root / FIXED_INITIALIZATION_RELATIVE
    return LaunchPaths(
        preflight=preflight,
        launcher_dir=preflight.artifact_root / "launcher",
        manifest_dir=preflight.artifact_root / "launchers",
        initialization_checkpoint=checkpoint,
    )


def validate_layout(
    paths: LaunchPaths,
    *,
    enforce_active_root: bool = True,
) -> LaunchPaths:
    try:
        preflight = v28_preflight.validate_layout(
            paths.preflight,
            enforce_active_root=enforce_active_root,
        )
    except v28_preflight.PreflightError as exc:
        raise LaunchError(str(exc)) from exc
    launcher_dir = paths.launcher_dir.resolve(strict=False)
    manifest_dir = paths.manifest_dir.resolve(strict=False)
    checkpoint = paths.initialization_checkpoint.resolve(strict=False)
    expected_checkpoint = (preflight.artifact_root / FIXED_INITIALIZATION_RELATIVE).resolve(strict=False)
    if checkpoint != expected_checkpoint:
        raise LaunchError(f"v28 launcher is pinned to periodic-step-000200044; actual={checkpoint}")
    for candidate in (launcher_dir, manifest_dir, checkpoint):
        if not _is_within(candidate, preflight.artifact_root):
            raise LaunchError(f"v28 runtime path escapes the artifact root: {candidate}")
    return LaunchPaths(
        preflight=preflight,
        launcher_dir=launcher_dir,
        manifest_dir=manifest_dir,
        initialization_checkpoint=checkpoint,
    )


def require_wsl() -> None:
    if os.name != "posix" or not sys.platform.startswith("linux"):
        raise LaunchError("the v28 launcher must run inside WSL")
    version = ""
    for candidate in (Path("/proc/sys/kernel/osrelease"), Path("/proc/version")):
        try:
            version += candidate.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
    if "microsoft" not in version.casefold():
        raise LaunchError("the v28 launcher requires a WSL Linux kernel")


def require_exact_artifact_environment(paths: LaunchPaths) -> None:
    raw = os.environ.get("STS2_ARTIFACT_ROOT")
    if raw and Path(raw).expanduser().resolve(strict=False) != paths.artifact_root:
        raise LaunchError("STS2_ARTIFACT_ROOT disagrees with the reviewed v28 runtime")


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    serialized = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor: int | None = None
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = None
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
        except OSError:
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LaunchError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise LaunchError(f"{label} must be a JSON object: {path}")
    return payload


def _reject_retired_v28_start(paths: LaunchPaths) -> None:
    contract_path = (paths.checkout_root / FROZEN_V28_CONTRACT_RELATIVE).resolve(strict=False)
    if not contract_path.is_file():
        raise LaunchError(
            "the mandatory frozen v28 retirement contract is missing; "
            "fail-closed refusal prevents the retired v28 recipe from "
            "starting another lineage",
        )
    contract = _load_json_object(
        contract_path,
        label="frozen v28 contract",
    )
    required = {
        "schema_version": "sts2-frozen-checkpoint-contract-v1",
        "name": "v28-mature-refinement-100k",
        "environment_steps": 100_000,
        "policy_version": 1_569,
        "exact_resume_permitted": False,
        "usage": "model_parameter_initialization_only",
    }
    for key, expected in required.items():
        if contract.get(key) != expected:
            raise LaunchError(f"frozen v28 retirement contract {key} changed")
    raise LaunchError(
        "v28 is frozen at 100,000 environment steps; the old recipe is "
        "retired and may not start another lineage. Use the reviewed v29 "
        "failure-credit-v4 model-initialization launcher."
    )


def _run_checked(
    command: Sequence[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
) -> str:
    result = subprocess.run(
        tuple(command),
        cwd=cwd,
        env=dict(environment),
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if result.returncode != 0:
        output = (result.stderr or result.stdout).strip()
        raise LaunchError(f"preflight command failed ({result.returncode}): {' '.join(command)}\n{output[-4000:]}")
    return result.stdout


def _environment_from_contract(contract: Mapping[str, Any]) -> dict[str, str]:
    raw_set = contract.get("set")
    raw_unset = contract.get("unset")
    if not isinstance(raw_set, Mapping) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in raw_set.items()
    ):
        raise LaunchError("reviewed trainer environment has a malformed set contract")
    if not isinstance(raw_unset, list) or not all(isinstance(item, str) for item in raw_unset):
        raise LaunchError("reviewed trainer environment has a malformed unset contract")
    if set(raw_set) & set(raw_unset):
        raise LaunchError("reviewed trainer environment sets and unsets the same key")
    environment = dict(os.environ)
    for key in raw_unset:
        environment.pop(key, None)
    environment.update({str(key): str(value) for key, value in raw_set.items()})
    return environment


def _validate_preflight_payload(
    payload: Mapping[str, Any],
    *,
    paths: LaunchPaths,
) -> dict[str, Any]:
    if payload.get("preflight_status", payload.get("status")) != "preflight-passed":
        raise LaunchError("reviewed preflight did not pass")
    if payload.get("training_started") is not False:
        raise LaunchError("reviewed preflight must be non-launching")
    if payload.get("run_name") != RUN_NAME:
        raise LaunchError("reviewed preflight names another training lineage")
    if payload.get("config_fingerprint_sha256") != FIXED_CONFIG_FINGERPRINT_SHA256:
        raise LaunchError("reviewed preflight config fingerprint changed")
    initialization = payload.get("initialization")
    if not isinstance(initialization, Mapping):
        raise LaunchError("reviewed preflight has no initialization proof")
    expected_initialization = {
        "mode": "model_initialization",
        "checkpoint": str(paths.initialization_checkpoint),
        "checkpoint_id": FIXED_INITIALIZATION_CHECKPOINT_ID,
        "network_parameters_inherited": True,
        "optimizer_rollouts_rng_and_counters_reset": True,
    }
    for key, expected in expected_initialization.items():
        if initialization.get(key) != expected:
            raise LaunchError(f"reviewed preflight initialization {key} changed")
    source_state = initialization.get("source_training_state")
    if not isinstance(source_state, Mapping):
        raise LaunchError("reviewed preflight has no source training state")
    if source_state.get("environment_steps") != FIXED_INITIALIZATION_STEP:
        raise LaunchError("reviewed preflight source environment step changed")
    if source_state.get("policy_version") != FIXED_INITIALIZATION_POLICY_VERSION:
        raise LaunchError("reviewed preflight source policy version changed")

    command = payload.get("trainer_command")
    if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
        raise LaunchError("reviewed preflight has no fixed trainer command")
    try:
        v28_preflight.validate_trainer_command(
            tuple(command),
            paths=paths.preflight,
            initialize_from=paths.initialization_checkpoint,
        )
    except v28_preflight.PreflightError as exc:
        raise LaunchError(str(exc)) from exc
    if "--resume" in command or command.count("--initialize-from") != 1:
        raise LaunchError("v28 launcher permits model initialization only, never exact resume")

    environment_contract = payload.get("trainer_environment")
    if not isinstance(environment_contract, Mapping):
        raise LaunchError("reviewed preflight has no trainer environment contract")
    environment = _environment_from_contract(environment_contract)
    try:
        v28_preflight.validate_trainer_environment(
            environment,
            paths=paths.preflight,
        )
    except v28_preflight.PreflightError as exc:
        raise LaunchError(str(exc)) from exc
    runtime = payload.get("runtime")
    if not isinstance(runtime, Mapping):
        raise LaunchError("reviewed preflight did not prove the GPU/simulator runtime")
    simulator = runtime.get("simulator")
    if not isinstance(runtime.get("device_name"), str) or not isinstance(simulator, Mapping):
        raise LaunchError("reviewed preflight runtime proof is incomplete")
    return dict(payload)


def _checkpoint_proof(paths: LaunchPaths) -> dict[str, Any]:
    checkpoint = paths.initialization_checkpoint
    manifest_path = checkpoint / "checkpoint.manifest.json"
    metadata_path = checkpoint / "metadata.json"
    if not manifest_path.is_file() or not metadata_path.is_file():
        raise LaunchError("fixed initialization checkpoint is not atomic/complete")
    manifest_sha = _sha256_file(manifest_path)
    metadata_sha = _sha256_file(metadata_path)
    if manifest_sha != FIXED_INITIALIZATION_MANIFEST_SHA256:
        raise LaunchError("fixed checkpoint manifest hash changed")
    if metadata_sha != FIXED_INITIALIZATION_METADATA_SHA256:
        raise LaunchError("fixed checkpoint metadata hash changed")
    manifest = _load_json_object(manifest_path, label="checkpoint manifest")
    metadata = _load_json_object(metadata_path, label="checkpoint metadata")
    state = metadata.get("training_state")
    if manifest.get("checkpoint_id") != FIXED_INITIALIZATION_CHECKPOINT_ID:
        raise LaunchError("fixed checkpoint ID changed")
    if not isinstance(state, Mapping) or state.get("environment_steps") != (FIXED_INITIALIZATION_STEP):
        raise LaunchError("fixed checkpoint training state changed")
    return {
        "path": str(checkpoint),
        "checkpoint_id": FIXED_INITIALIZATION_CHECKPOINT_ID,
        "manifest_sha256": manifest_sha,
        "metadata_sha256": metadata_sha,
        "source_environment_steps": FIXED_INITIALIZATION_STEP,
        "source_policy_version": FIXED_INITIALIZATION_POLICY_VERSION,
    }


def _git_provenance(paths: LaunchPaths) -> dict[str, Any]:
    environment = _environment_from_contract(
        v28_preflight.trainer_environment_contract(v28_preflight.build_trainer_environment(paths.preflight))
    )
    commit = _run_checked(
        ("git", "rev-parse", "HEAD"),
        cwd=paths.checkout_root,
        environment=environment,
    ).strip()
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise LaunchError(f"git returned an invalid full commit: {commit!r}")
    dirty = _run_checked(
        ("git", "status", "--porcelain=v1", "--untracked-files=normal"),
        cwd=paths.checkout_root,
        environment=environment,
    )
    if dirty.strip():
        raise LaunchError("v28 start requires a clean committed checkout")
    return {"commit": commit, "worktree_clean": True}


def run_preflight(
    paths: LaunchPaths,
    *,
    reviewed_preflight: Path | None = None,
    enforce_active_root: bool = True,
    verify_runtime: bool = True,
) -> dict[str, Any]:
    paths = validate_layout(paths, enforce_active_root=enforce_active_root)
    require_exact_artifact_environment(paths)
    try:
        live = v28_preflight.run_preflight(
            paths.preflight,
            initialize_from=paths.initialization_checkpoint,
            enforce_active_root=enforce_active_root,
            verify_runtime=verify_runtime,
        )
    except v28_preflight.PreflightError as exc:
        raise LaunchError(str(exc)) from exc
    live = _validate_preflight_payload(live, paths=paths)

    reviewed: dict[str, Any]
    review_source: dict[str, Any]
    if reviewed_preflight is None:
        reviewed = live
        review_source = {"mode": "live-fixed-preflight", "path": None, "sha256": None}
    else:
        review_path = reviewed_preflight.expanduser().resolve(strict=True)
        reviewed = _validate_preflight_payload(
            _load_json_object(review_path, label="reviewed preflight payload"),
            paths=paths,
        )
        for key in (
            "run_name",
            "initialization",
            "config_fingerprint_sha256",
            "trainer_command",
            "runtime",
        ):
            if reviewed.get(key) != live.get(key):
                raise LaunchError(f"reviewed preflight {key} no longer matches live preflight")
        review_source = {
            "mode": "explicit-reviewed-preflight",
            "path": str(review_path),
            "sha256": _sha256_file(review_path),
        }
    return {
        **reviewed,
        "schema_version": MANIFEST_SCHEMA,
        # The supervised manifest later owns the top-level ``status`` field.
        # Keep the non-launching proof under a non-colliding immutable key.
        "preflight_status": "preflight-passed",
        "selection": _checkpoint_proof(paths),
        "review_source": review_source,
        "git": _git_provenance(paths),
        "config_file_sha256": _sha256_file(paths.preflight.config_path),
        "simulator_identity_sha256": _sha256_file(paths.preflight.simulator_identity),
        "training_started": False,
    }


def capture_process_identity(pid: int) -> ProcessIdentity:
    proc_root = Path("/proc") / str(pid)
    try:
        stat = (proc_root / "stat").read_text(encoding="utf-8")
        command_line = (proc_root / "cmdline").read_bytes()
        executable = os.readlink(proc_root / "exe")
    except OSError as exc:
        raise LaunchError(f"cannot capture Linux process identity for PID {pid}: {exc}") from exc
    close = stat.rfind(")")
    fields = stat[close + 2 :].split() if close >= 0 else []
    if len(fields) < 20 or not command_line:
        raise LaunchError(f"Linux process identity is incomplete for PID {pid}")
    try:
        start_ticks = int(fields[19])
    except ValueError as exc:
        raise LaunchError(f"Linux process start marker is invalid for PID {pid}") from exc
    return ProcessIdentity(
        pid=pid,
        proc_start_ticks=start_ticks,
        command_line_sha256=_sha256_bytes(command_line),
        executable=executable,
    )


def process_identity_matches(identity: ProcessIdentity) -> bool:
    try:
        return capture_process_identity(identity.pid) == identity
    except LaunchError:
        return False


def _identity_from_mapping(value: object, *, label: str) -> ProcessIdentity:
    if not isinstance(value, Mapping):
        raise LaunchError(f"{label} process identity is malformed")
    try:
        return ProcessIdentity(
            pid=int(value["pid"]),
            proc_start_ticks=int(value["proc_start_ticks"]),
            command_line_sha256=str(value["command_line_sha256"]),
            executable=str(value["executable"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise LaunchError(f"{label} process identity is malformed") from exc


def _optional_identity(value: object, *, label: str) -> ProcessIdentity | None:
    return None if value is None else _identity_from_mapping(value, label=label)


def _capture_child(process: subprocess.Popen[bytes]) -> ProcessIdentity:
    last_error: Exception | None = None
    for _ in range(50):
        if process.poll() is not None:
            raise LaunchError(f"child exited before identity capture: {process.returncode}")
        try:
            return capture_process_identity(process.pid)
        except LaunchError as exc:
            last_error = exc
            time.sleep(0.02)
    process.terminate()
    raise LaunchError(f"could not capture child process identity: {last_error}")


def _state_path(paths: LaunchPaths) -> Path:
    return paths.launcher_dir / f"{RUN_NAME}.state.json"


def _run_root(paths: LaunchPaths) -> Path:
    return (paths.artifact_root / "runs" / RUN_NAME).resolve(strict=False)


@contextmanager
def _terminal_lock(paths: LaunchPaths, launch_id: str):  # type: ignore[no-untyped-def]
    lock_dir = paths.launcher_dir / "watchdog-locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(lock_dir / f"{launch_id}.lock", os.O_RDWR | os.O_CREAT, 0o600)
    try:
        if fcntl is not None:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        if fcntl is not None:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _state_payload(manifest_path: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": STATE_SCHEMA,
        "run_name": RUN_NAME,
        "launch_id": manifest.get("launch_id"),
        "manifest_path": str(manifest_path),
        "status": manifest.get("status"),
        "supervisor_process_identity": manifest.get("supervisor_process_identity"),
        "trainer_process_identity": manifest.get("trainer_process_identity"),
        "metrics_path": manifest.get("metrics_path"),
        "terminal_event_id": manifest.get("terminal_event_id"),
        "written_at_utc": _utc_now(),
        "written_unix_s": time.time(),
    }


def _persist_manifest(
    paths: LaunchPaths,
    *,
    manifest_path: Path,
    manifest: Mapping[str, Any],
) -> None:
    atomic_write_json(manifest_path, manifest)
    atomic_write_json(_state_path(paths), _state_payload(manifest_path, manifest))


def _validate_selection_proof(value: object, *, paths: LaunchPaths) -> None:
    if not isinstance(value, Mapping):
        raise LaunchError("supervised manifest has no fixed checkpoint proof")
    expected = _checkpoint_proof(paths)
    if dict(value) != expected:
        raise LaunchError("supervised manifest checkpoint proof changed")


def _validate_manifest(paths: LaunchPaths, manifest_path: Path) -> dict[str, Any]:
    manifest_path = manifest_path.resolve(strict=False)
    if not _is_within(manifest_path, paths.manifest_dir):
        raise LaunchError("supervised manifest escapes the manifest directory")
    manifest = _load_json_object(manifest_path, label="supervised launch manifest")
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise LaunchError("supervised manifest schema changed")
    if manifest.get("run_name") != RUN_NAME:
        raise LaunchError("supervised manifest names another lineage")
    launch_id = manifest.get("launch_id")
    if not isinstance(launch_id, str) or not launch_id:
        raise LaunchError("supervised manifest has no launch ID")
    _validate_preflight_payload(manifest, paths=paths)
    _validate_selection_proof(manifest.get("selection"), paths=paths)
    existing = manifest.get("preexisting_run_directories")
    if not isinstance(existing, list) or not all(isinstance(item, str) for item in existing):
        raise LaunchError("supervised manifest has no run-directory snapshot")
    root = _run_root(paths)
    for item in existing:
        candidate = Path(item).resolve(strict=False)
        if not _is_within(candidate, root) or not candidate.name.startswith("run-"):
            raise LaunchError("run-directory snapshot escapes the v28 run root")
    log_path = Path(str(manifest.get("log_path") or "")).resolve(strict=False)
    if not _is_within(log_path, paths.launcher_dir / "logs"):
        raise LaunchError("supervised log path escapes the launcher log root")
    return manifest


def _snapshot_run_directories(paths: LaunchPaths) -> list[str]:
    root = _run_root(paths)
    if not root.is_dir():
        return []
    return sorted(str(path.resolve(strict=False)) for path in root.glob("run-*") if path.is_dir())


def _metrics_events(metrics_path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    try:
        with metrics_path.open("rb") as handle:
            for line_number, raw in enumerate(handle, start=1):
                if len(raw) > 8 * 1024 * 1024:
                    raise LaunchError(f"metrics line {line_number} exceeds size limit")
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError:
                    if handle.read(1):
                        raise LaunchError(f"metrics has invalid JSON before EOF at line {line_number}") from None
                    break
                if isinstance(payload, dict):
                    events.append(payload)
    except FileNotFoundError:
        return []
    return events


def _matching_model_init_run_start(metrics_path: Path, paths: LaunchPaths) -> dict[str, Any] | None:
    for event in _metrics_events(metrics_path):
        if event.get("event") != "run_start":
            continue
        load = event.get("checkpoint_load")
        state = event.get("state")
        if not isinstance(load, Mapping) or not isinstance(state, Mapping):
            continue
        parent = load.get("parent_checkpoint")
        if not isinstance(parent, str):
            continue
        if (
            load.get("mode") == "model_initialization"
            and Path(parent).resolve(strict=False) == paths.initialization_checkpoint
            and load.get("network_parameters_initialized") is True
            and load.get("optimizer_rollouts_rng_and_counters_reset") is True
            and state.get("environment_steps") == 0
            and state.get("policy_version") == 0
            and event.get("config_fingerprint_sha256") == FIXED_RUNTIME_CONFIG_FINGERPRINT_SHA256
        ):
            return event
    return None


def discover_successor_metrics(
    paths: LaunchPaths,
    *,
    preexisting_run_directories: Sequence[str],
) -> tuple[Path, dict[str, Any]] | None:
    root = _run_root(paths)
    if not root.is_dir():
        return None
    previous = {Path(item).resolve(strict=False) for item in preexisting_run_directories}
    matches: list[tuple[Path, dict[str, Any]]] = []
    for run_dir in root.glob("run-*"):
        resolved = run_dir.resolve(strict=False)
        if resolved in previous or not run_dir.is_dir():
            continue
        metrics = (run_dir / "metrics.jsonl").resolve(strict=False)
        if not _is_within(metrics, root) or not metrics.is_file():
            continue
        run_start = _matching_model_init_run_start(metrics, paths)
        if run_start is not None:
            matches.append((metrics, run_start))
    if len(matches) > 1:
        raise LaunchError("multiple new runs claim the fixed model-initialization checkpoint")
    return matches[0] if matches else None


def _record_successor_binding(
    paths: LaunchPaths,
    *,
    manifest_path: Path,
    metrics_path: Path,
    run_start: Mapping[str, Any],
) -> dict[str, Any]:
    launch_id = str(_load_json_object(manifest_path, label="manifest").get("launch_id"))
    with _terminal_lock(paths, launch_id):
        manifest = _validate_manifest(paths, manifest_path)
        prior = manifest.get("metrics_path")
        if prior is not None and Path(str(prior)).resolve(strict=False) != metrics_path:
            raise LaunchError("successor metrics binding changed")
        state = run_start.get("state")
        manifest.update(
            {
                "metrics_path": str(metrics_path),
                "successor": {
                    "run_id": run_start.get("run_id"),
                    "unix_s": run_start.get("unix_s"),
                    "initial_environment_steps": (
                        state.get("environment_steps") if isinstance(state, Mapping) else None
                    ),
                    "checkpoint_load": run_start.get("checkpoint_load"),
                },
                "successor_bound_at_utc": _utc_now(),
            }
        )
        _persist_manifest(paths, manifest_path=manifest_path, manifest=manifest)
        return manifest


def _discover_from_manifest(paths: LaunchPaths, manifest: Mapping[str, Any]) -> tuple[Path, dict[str, Any]] | None:
    previous = manifest.get("preexisting_run_directories")
    if not isinstance(previous, list) or not all(isinstance(item, str) for item in previous):
        raise LaunchError("manifest run snapshot is malformed")
    return discover_successor_metrics(paths, preexisting_run_directories=previous)


def _append_jsonl_event_once(
    metrics_path: Path,
    event: Mapping[str, Any],
    *,
    event_id: str,
) -> bool:
    if any(item.get("watchdog_event_id") == event_id for item in _metrics_events(metrics_path)):
        return False
    serialized = (json.dumps(event, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    # A native abort can leave JsonlMetrics halfway through its final object.
    # The trainer is already dead here, so remove only that unterminated EOF
    # suffix before appending.  Prefixing a newline would preserve malformed
    # JSON as an internal line and break strict reconciliation on the next
    # watchdog pass.
    flags = os.O_RDWR | os.O_APPEND
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(metrics_path, flags)
    try:
        size = os.lseek(descriptor, 0, os.SEEK_END)
        if size:
            os.lseek(descriptor, size - 1, os.SEEK_SET)
            if os.read(descriptor, 1) != b"\n":
                position = size
                last_newline = -1
                while position > 0 and last_newline < 0:
                    chunk_start = max(0, position - 64 * 1024)
                    os.lseek(descriptor, chunk_start, os.SEEK_SET)
                    chunk = os.read(descriptor, position - chunk_start)
                    relative = chunk.rfind(b"\n")
                    if relative >= 0:
                        last_newline = chunk_start + relative
                        break
                    position = chunk_start
                os.ftruncate(descriptor, last_newline + 1)
                os.fsync(descriptor)
        os.lseek(descriptor, 0, os.SEEK_END)
        if os.write(descriptor, serialized) != len(serialized):
            raise LaunchError("watchdog metrics append was incomplete")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return True


def _log_tail(path: Path, *, maximum_bytes: int = 128 * 1024) -> str:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - maximum_bytes))
            return handle.read(maximum_bytes).decode("utf-8", errors="replace")
    except OSError:
        return ""


def _exit_classification(
    *,
    returncode: int | None,
    events: Sequence[Mapping[str, Any]],
    log_tail: str,
) -> dict[str, Any]:
    lifecycle = {
        str(event.get("event")) for event in events if event.get("event") in {"run_complete", "interrupt", "run_failed"}
    }
    if (returncode in (0, None)) and "run_complete" in lifecycle:
        return {"status": "completed", "kind": "clean_completion", "native_abort": False, "append_run_failed": False}
    if "interrupt" in lifecycle:
        return {"status": "interrupted", "kind": "runtime_interrupt", "native_abort": False, "append_run_failed": False}
    if "run_failed" in lifecycle:
        return {
            "status": "failed",
            "kind": "runtime_reported_failure",
            "native_abort": False,
            "append_run_failed": False,
        }
    signal_number = -returncode if returncode is not None and returncode < 0 else None
    if signal_number is None and returncode is not None and returncode >= 128:
        candidate = returncode - 128
        if 0 < candidate < signal.NSIG:
            signal_number = candidate
    try:
        signal_name = signal.Signals(signal_number).name if signal_number is not None else None
    except ValueError:
        signal_name = f"SIGNAL_{signal_number}"
    signatures = (
        "AqlQueue::HandleInsufficientScratch",
        "process compute queue fail",
        "Assertion `",
        "Aborted (core dumped)",
    )
    matched = [item for item in signatures if item in log_tail]
    native_abort = signal_number == signal.SIGABRT or bool(matched)
    kind = (
        "native_abort"
        if native_abort
        else "exit_zero_without_run_complete"
        if returncode == 0
        else "unobserved_exit_without_terminal_event"
        if returncode is None
        else "signal_exit"
        if signal_number is not None
        else "nonzero_exit"
    )
    return {
        "status": "failed",
        "kind": kind,
        "native_abort": native_abort,
        "returncode": returncode,
        "signal_number": signal_number,
        "signal_name": signal_name,
        "matched_log_signatures": matched,
        "append_run_failed": True,
    }


def _latest_environment_steps(events: Sequence[Mapping[str, Any]]) -> int:
    latest = 0
    for event in events:
        for value in (
            event.get("environment_steps"),
            event.get("state", {}).get("environment_steps") if isinstance(event.get("state"), Mapping) else None,
        ):
            if isinstance(value, int) and not isinstance(value, bool):
                latest = max(latest, value)
    return latest


def finalize_supervised_exit(
    paths: LaunchPaths,
    *,
    manifest_path: Path,
    returncode: int | None,
    supervisor_identity: ProcessIdentity | None,
    trainer_identity: ProcessIdentity | None,
    metrics_path: Path | None,
    reconciliation_reason: str | None = None,
    enforce_active_root: bool = True,
) -> dict[str, Any]:
    paths = validate_layout(paths, enforce_active_root=enforce_active_root)
    manifest = _validate_manifest(paths, manifest_path)
    launch_id = str(manifest["launch_id"])
    with _terminal_lock(paths, launch_id):
        manifest = _validate_manifest(paths, manifest_path)
        if manifest.get("status") in TERMINAL_STATUSES:
            return manifest
        if metrics_path is not None:
            metrics_path = metrics_path.resolve(strict=False)
            if not _is_within(metrics_path, _run_root(paths)):
                raise LaunchError("successor metrics path escapes the v28 run root")
            events = _metrics_events(metrics_path)
        else:
            events = []
        log_path = Path(str(manifest.get("log_path") or ""))
        existing = next(
            (
                event
                for event in reversed(events)
                if event.get("event") == "run_failed"
                and event.get("source") == "persistent-native-exit-watchdog"
                and event.get("launch_id") == launch_id
            ),
            None,
        )
        if existing is not None and isinstance(existing.get("classification"), Mapping):
            classification = dict(existing["classification"])
            classification["status"] = "failed"
            classification["append_run_failed"] = False
        else:
            classification = _exit_classification(
                returncode=returncode,
                events=events,
                log_tail=_log_tail(log_path),
            )
        event_id = (
            str(existing.get("watchdog_event_id"))
            if existing is not None and isinstance(existing.get("watchdog_event_id"), str)
            else None
        )
        appended = False
        if classification["append_run_failed"] is True:
            event_id = _sha256_bytes(f"{launch_id}:{WATCHDOG_EVENT_SCHEMA}:run_failed".encode())
            failure = {
                "event": "run_failed",
                "unix_s": time.time(),
                "schema_version": WATCHDOG_EVENT_SCHEMA,
                "source": "persistent-native-exit-watchdog",
                "watchdog_event_id": event_id,
                "launch_id": launch_id,
                "run_name": RUN_NAME,
                "environment_steps": _latest_environment_steps(events),
                "initialization_mode": "model_initialization",
                "initialization_checkpoint": str(paths.initialization_checkpoint),
                "initialization_checkpoint_id": FIXED_INITIALIZATION_CHECKPOINT_ID,
                "source_environment_steps": FIXED_INITIALIZATION_STEP,
                "returncode": returncode,
                "classification": classification,
                "reason": classification["kind"],
                "reconciliation_reason": reconciliation_reason,
                "supervisor_process_identity": asdict(supervisor_identity) if supervisor_identity else None,
                "trainer_process_identity": asdict(trainer_identity) if trainer_identity else None,
                "log_path": str(log_path),
            }
            if metrics_path is not None:
                appended = _append_jsonl_event_once(metrics_path, failure, event_id=event_id)
            else:
                fallback = paths.launcher_dir / "terminal-events" / f"{launch_id}.run_failed.json"
                atomic_write_json(fallback, failure)
        terminal = {
            "status": classification["status"],
            "classification": classification,
            "returncode": returncode,
            "observed_at_utc": _utc_now(),
            "watchdog_event_id": event_id,
            "watchdog_event_appended": appended,
            "metrics_path": str(metrics_path) if metrics_path is not None else None,
            "reconciliation_reason": reconciliation_reason,
        }
        manifest.update(
            {
                "status": classification["status"],
                "terminal": terminal,
                "supervisor_process_identity": asdict(supervisor_identity) if supervisor_identity else None,
                "trainer_process_identity": asdict(trainer_identity) if trainer_identity else None,
                "metrics_path": terminal["metrics_path"],
                "terminal_event_id": event_id,
            }
        )
        _persist_manifest(paths, manifest_path=manifest_path, manifest=manifest)
        return manifest


def _supervised_status(
    paths: LaunchPaths,
    *,
    state: Mapping[str, Any],
    manifest_path: Path,
    allow_reconciliation: bool,
    enforce_active_root: bool,
) -> dict[str, Any]:
    manifest = _validate_manifest(paths, manifest_path)
    if manifest.get("launch_id") != state.get("launch_id"):
        raise LaunchError("launcher state and manifest launch IDs disagree")
    supervisor = _optional_identity(manifest.get("supervisor_process_identity"), label="supervisor")
    trainer = _optional_identity(manifest.get("trainer_process_identity"), label="trainer")
    supervisor_live = process_identity_matches(supervisor) if supervisor else False
    trainer_live = process_identity_matches(trainer) if trainer else False
    manifest_status = str(manifest.get("status") or "invalid")
    if manifest_status in TERMINAL_STATUSES:
        status, running = manifest_status, False
    elif supervisor_live and trainer_live:
        status, running = "running", True
    elif supervisor_live and trainer is None:
        status, running = "supervisor-running", True
    elif supervisor_live:
        status, running = "supervisor-finalizing", True
    elif trainer_live:
        status, running = "unwatched-running", True
    else:
        written = state.get("written_unix_s")
        recent = (
            isinstance(written, int | float)
            and not isinstance(written, bool)
            and time.time() - float(written) < 30.0
            and supervisor is None
            and trainer is None
        )
        if recent:
            status, running = "supervisor-launching", True
        elif allow_reconciliation:
            metrics: Path | None = None
            raw_metrics = manifest.get("metrics_path")
            if isinstance(raw_metrics, str) and raw_metrics:
                metrics = Path(raw_metrics).resolve(strict=False)
            else:
                discovered = _discover_from_manifest(paths, manifest)
                if discovered is not None:
                    metrics, run_start = discovered
                    _record_successor_binding(
                        paths,
                        manifest_path=manifest_path,
                        metrics_path=metrics,
                        run_start=run_start,
                    )
            finalize_supervised_exit(
                paths,
                manifest_path=manifest_path,
                returncode=None,
                supervisor_identity=supervisor,
                trainer_identity=trainer,
                metrics_path=metrics,
                reconciliation_reason="supervisor_and_trainer_not_live",
                enforce_active_root=enforce_active_root,
            )
            repaired = _load_json_object(_state_path(paths), label="reconciled state")
            return _supervised_status(
                paths,
                state=repaired,
                manifest_path=manifest_path,
                allow_reconciliation=False,
                enforce_active_root=enforce_active_root,
            )
        else:
            status, running = "failed-unreconciled", False
    return {
        "schema_version": STATE_SCHEMA,
        "run_name": RUN_NAME,
        "status": status,
        "running": running,
        "launch_id": manifest.get("launch_id"),
        "manifest_path": str(manifest_path),
        "manifest_status": manifest_status,
        "log_path": manifest.get("log_path"),
        "metrics_path": manifest.get("metrics_path"),
        "terminal": manifest.get("terminal"),
        "supervisor": {"identity": asdict(supervisor) if supervisor else None, "identity_matches": supervisor_live},
        "trainer": {"identity": asdict(trainer) if trainer else None, "identity_matches": trainer_live},
    }


def read_status(
    paths: LaunchPaths,
    *,
    enforce_active_root: bool = True,
) -> dict[str, Any]:
    paths = validate_layout(paths, enforce_active_root=enforce_active_root)
    state_path = _state_path(paths)
    if not state_path.is_file():
        return {
            "schema_version": STATE_SCHEMA,
            "run_name": RUN_NAME,
            "status": "not-started",
            "running": False,
            "state_path": str(state_path),
        }
    state = _load_json_object(state_path, label="v28 launcher state")
    if state.get("schema_version") != STATE_SCHEMA:
        raise LaunchError("v28 launcher state schema changed")
    raw_manifest = state.get("manifest_path")
    if not isinstance(raw_manifest, str) or not raw_manifest:
        raise LaunchError("v28 launcher state has no manifest path")
    manifest_path = Path(raw_manifest).resolve(strict=False)
    if not _is_within(manifest_path, paths.manifest_dir):
        raise LaunchError("v28 launcher state escapes the manifest directory")
    return _supervised_status(
        paths,
        state=state,
        manifest_path=manifest_path,
        allow_reconciliation=True,
        enforce_active_root=enforce_active_root,
    )


def _reprove_launch_contract(paths: LaunchPaths, manifest: Mapping[str, Any]) -> None:
    current = run_preflight(paths)
    for key in (
        "initialization",
        "config_fingerprint_sha256",
        "trainer_command",
        "runtime",
        "selection",
        "git",
        "config_file_sha256",
        "simulator_identity_sha256",
    ):
        if current.get(key) != manifest.get(key):
            raise LaunchError(f"v28 launch proof changed after detach: {key}")


def supervise(paths: LaunchPaths, *, manifest_path: Path) -> dict[str, Any]:
    require_wsl()
    paths = validate_layout(paths)
    require_exact_artifact_environment(paths)
    manifest = _validate_manifest(paths, manifest_path)
    launch_id = str(manifest["launch_id"])
    supervisor_identity: ProcessIdentity | None = None
    trainer_identity: ProcessIdentity | None = None
    trainer: subprocess.Popen[bytes] | None = None
    metrics: Path | None = None
    returncode: int | None = None
    try:
        _reprove_launch_contract(paths, manifest)
        supervisor_identity = capture_process_identity(os.getpid())
        with _terminal_lock(paths, launch_id):
            manifest = _validate_manifest(paths, manifest_path)
            if manifest.get("status") in TERMINAL_STATUSES:
                return manifest
            manifest.update(
                {
                    "status": "supervisor-running",
                    "supervisor_process_identity": asdict(supervisor_identity),
                    "supervisor_started_at_utc": _utc_now(),
                }
            )
            _persist_manifest(paths, manifest_path=manifest_path, manifest=manifest)

        command = tuple(str(item) for item in manifest["trainer_command"])
        v28_preflight.validate_trainer_command(
            command,
            paths=paths.preflight,
            initialize_from=paths.initialization_checkpoint,
        )
        environment_contract = manifest.get("trainer_environment")
        if not isinstance(environment_contract, Mapping):
            raise LaunchError("manifest trainer environment is malformed")
        environment = _environment_from_contract(environment_contract)
        v28_preflight.validate_trainer_environment(environment, paths=paths.preflight)
        log_path = Path(str(manifest["log_path"])).resolve(strict=False)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("ab", buffering=0) as log_handle:
            trainer = subprocess.Popen(
                command,
                cwd=paths.package_root,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
        trainer_identity = _capture_child(trainer)
        with _terminal_lock(paths, launch_id):
            manifest = _validate_manifest(paths, manifest_path)
            manifest.update(
                {
                    "status": "running",
                    "supervisor_process_identity": asdict(supervisor_identity),
                    "trainer_process_identity": asdict(trainer_identity),
                    "trainer_started_at_utc": _utc_now(),
                }
            )
            _persist_manifest(paths, manifest_path=manifest_path, manifest=manifest)

        def forward_signal(number: int, _frame: object) -> None:
            if trainer is not None and trainer.poll() is None:
                trainer.send_signal(number)

        signal.signal(signal.SIGINT, forward_signal)
        signal.signal(signal.SIGTERM, forward_signal)
        while True:
            returncode = trainer.poll()
            if metrics is None:
                current_manifest = _load_json_object(manifest_path, label="manifest")
                discovered = _discover_from_manifest(paths, current_manifest)
                if discovered is not None:
                    metrics, run_start = discovered
                    _record_successor_binding(
                        paths,
                        manifest_path=manifest_path,
                        metrics_path=metrics,
                        run_start=run_start,
                    )
            if returncode is not None:
                break
            time.sleep(1.0)
        deadline = time.monotonic() + 5.0
        while metrics is None and time.monotonic() < deadline:
            current_manifest = _load_json_object(manifest_path, label="manifest")
            discovered = _discover_from_manifest(paths, current_manifest)
            if discovered is not None:
                metrics, run_start = discovered
                _record_successor_binding(
                    paths,
                    manifest_path=manifest_path,
                    metrics_path=metrics,
                    run_start=run_start,
                )
                break
            time.sleep(0.2)
        return finalize_supervised_exit(
            paths,
            manifest_path=manifest_path,
            returncode=returncode,
            supervisor_identity=supervisor_identity,
            trainer_identity=trainer_identity,
            metrics_path=metrics,
        )
    except Exception as exc:
        if trainer is not None and trainer.poll() is None:
            trainer.terminate()
            try:
                returncode = trainer.wait(timeout=15.0)
            except subprocess.TimeoutExpired:
                trainer.kill()
                returncode = trainer.wait(timeout=15.0)
        elif trainer is not None:
            returncode = trainer.poll()
        try:
            current_manifest = _load_json_object(manifest_path, label="manifest")
            if metrics is None:
                discovered = _discover_from_manifest(paths, current_manifest)
                if discovered is not None:
                    metrics, run_start = discovered
                    _record_successor_binding(
                        paths,
                        manifest_path=manifest_path,
                        metrics_path=metrics,
                        run_start=run_start,
                    )
            finalize_supervised_exit(
                paths,
                manifest_path=manifest_path,
                returncode=returncode if returncode is not None else 70,
                supervisor_identity=supervisor_identity,
                trainer_identity=trainer_identity,
                metrics_path=metrics,
                reconciliation_reason=f"supervisor_exception:{type(exc).__name__}:{exc}",
            )
        except Exception:
            pass
        if isinstance(exc, LaunchError | v28_preflight.PreflightError):
            raise LaunchError(str(exc)) from exc
        raise LaunchError(f"supervisor failed: {type(exc).__name__}: {exc}") from exc


def start(
    paths: LaunchPaths,
    *,
    reviewed_preflight: Path | None = None,
) -> dict[str, Any]:
    require_wsl()
    _reject_retired_v28_start(paths)
    preflight = run_preflight(paths, reviewed_preflight=reviewed_preflight)
    paths = validate_layout(paths)
    paths.launcher_dir.mkdir(parents=True, exist_ok=True)
    paths.manifest_dir.mkdir(parents=True, exist_ok=True)
    with _terminal_lock(paths, f"{RUN_NAME}-start"):
        previous = read_status(paths)
        if previous["running"] is True:
            raise LaunchError(
                "v28 already has an active trainer/supervisor: "
                f"status={previous.get('status')} launch_id={previous.get('launch_id')}"
            )
        launch_id = str(uuid.uuid4())
        timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        log_path = paths.launcher_dir / "logs" / f"{RUN_NAME}-{timestamp}-{launch_id}.log"
        manifest_path = paths.manifest_dir / f"{RUN_NAME}-{timestamp}-{launch_id}.launch.json"
        supervisor_command = (
            str(paths.venv_python),
            str(Path(__file__).resolve()),
            "supervise",
            "--manifest",
            str(manifest_path),
        )
        manifest: dict[str, Any] = {
            **preflight,
            "launch_id": launch_id,
            "created_at_utc": _utc_now(),
            "created_unix_s": time.time(),
            "status": "supervisor-launching",
            "training_started": False,
            "supervisor_command": list(supervisor_command),
            "supervisor_process_identity": None,
            "trainer_process_identity": None,
            "preexisting_run_directories": _snapshot_run_directories(paths),
            "metrics_path": None,
            "log_path": str(log_path),
            "manifest_path": str(manifest_path),
        }
        _persist_manifest(paths, manifest_path=manifest_path, manifest=manifest)
        environment_contract = manifest.get("trainer_environment")
        assert isinstance(environment_contract, Mapping)
        environment = _environment_from_contract(environment_contract)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with log_path.open("ab", buffering=0) as log_handle:
                process = subprocess.Popen(
                    supervisor_command,
                    cwd=paths.package_root,
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            identity = _capture_child(process)
        except (OSError, LaunchError) as exc:
            finalize_supervised_exit(
                paths,
                manifest_path=manifest_path,
                returncode=70,
                supervisor_identity=None,
                trainer_identity=None,
                metrics_path=None,
                reconciliation_reason=f"supervisor_spawn_failed:{exc}",
            )
            raise LaunchError(f"could not spawn v28 supervisor: {exc}") from exc
        with _terminal_lock(paths, launch_id):
            manifest = _validate_manifest(paths, manifest_path)
            if manifest.get("supervisor_process_identity") is None:
                manifest["supervisor_process_identity"] = asdict(identity)
            _persist_manifest(paths, manifest_path=manifest_path, manifest=manifest)
    return read_status(paths)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        nargs="?",
        default="status",
        choices=("preflight", "start", "status", "supervise"),
    )
    parser.add_argument(
        "--reviewed-preflight",
        type=Path,
        help="explicit operator-reviewed JSON from the non-launching v28 preflight",
    )
    parser.add_argument("--manifest", type=Path, help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = default_paths()
    try:
        if args.action != "supervise" and args.manifest is not None:
            raise LaunchError(f"{args.action} does not accept --manifest")
        if args.action not in {"preflight", "start"} and args.reviewed_preflight is not None:
            raise LaunchError(f"{args.action} does not accept --reviewed-preflight")
        if args.action == "preflight":
            require_wsl()
            payload = run_preflight(paths, reviewed_preflight=args.reviewed_preflight)
        elif args.action == "start":
            payload = start(paths, reviewed_preflight=args.reviewed_preflight)
        elif args.action == "supervise":
            if args.manifest is None:
                raise LaunchError("internal supervise action requires --manifest")
            payload = supervise(paths, manifest_path=args.manifest)
        else:
            require_wsl()
            require_exact_artifact_environment(validate_layout(paths))
            payload = read_status(paths)
    except (LaunchError, v28_preflight.PreflightError) as exc:
        print(f"[v28-launcher] {exc}", file=sys.stderr)
        return 2
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
