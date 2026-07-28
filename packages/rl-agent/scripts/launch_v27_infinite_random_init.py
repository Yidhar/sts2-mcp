#!/usr/bin/env python3
"""Fail-closed WSL launcher for the fresh v27 infinite-revival lineage.

The command is intentionally fixed.  It cannot accept a resume checkpoint,
model-initialization checkpoint, alternate artifact root, alternate simulator,
or arbitrary trainer arguments.  Run it inside WSL with one of:

    python3 scripts/launch_v27_infinite_random_init.py preflight
    python3 scripts/launch_v27_infinite_random_init.py start
    python3 scripts/launch_v27_infinite_random_init.py status

``start`` detaches one trainer process and records an atomic manifest and a
PID-reuse-resistant process identity.  This script deliberately has no stop
command; shutdown remains an explicit operator action.
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
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "sts2-v27-random-init-launch-v1"
STATE_SCHEMA_VERSION = "sts2-v27-random-init-launch-state-v1"
RUN_NAME = "full-run-revival-v27-infinite-random-init"
ACTIVE_ARTIFACT_ROOT = Path(
    "/mnt/e/game/project/sts2_mcp_artifacts/runtime"
)
CORRUPT_RECOVERY_ROOT = Path("/mnt/f/backoff/sts2_mcp_artifacts/runtime")
FORBIDDEN_SIBLING_CHECKPOINT_ROOT = Path(
    "/mnt/e/game/project/sts2_mcp_artifacts/checkpoints"
)


class LaunchError(RuntimeError):
    """The fixed launch contract could not be proven."""


@dataclass(frozen=True, slots=True)
class LaunchPaths:
    checkout_root: Path
    package_root: Path
    artifact_root: Path
    venv_python: Path
    config_path: Path
    simulator_executable: Path
    simulator_identity: Path
    launcher_dir: Path
    manifest_dir: Path


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


def _portable_path_text(path: Path) -> str:
    return path.as_posix().rstrip("/").casefold()


def _absolute_without_symlink_resolution(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _is_forbidden_recovery_path(path: Path) -> bool:
    candidate = _portable_path_text(path)
    forbidden = _portable_path_text(CORRUPT_RECOVERY_ROOT)
    return candidate == forbidden or candidate.startswith(forbidden + "/")


def _is_forbidden_sibling_checkpoint_path(path: Path) -> bool:
    candidate = _portable_path_text(path)
    forbidden = _portable_path_text(FORBIDDEN_SIBLING_CHECKPOINT_ROOT)
    return candidate == forbidden or candidate.startswith(forbidden + "/")


def default_paths() -> LaunchPaths:
    script_dir = Path(__file__).resolve().parent
    package_root = script_dir.parent
    checkout_root = script_dir.parents[2]
    artifact_root = ACTIVE_ARTIFACT_ROOT
    simulator = (
        artifact_root
        / "dependencies/sts2-ai/STS2AI/ENV/Sim/HeadlessSim/bin/Release/net9.0/HeadlessSim.exe"
    )
    return LaunchPaths(
        checkout_root=checkout_root,
        package_root=package_root,
        artifact_root=artifact_root,
        venv_python=artifact_root / "environments/wsl-rocm/bin/python",
        config_path=(
            package_root
            / "config/experiments/full_run_revival_v27_infinite_random_init.toml"
        ),
        simulator_executable=simulator,
        simulator_identity=simulator.with_name(simulator.name + ".identity.json"),
        launcher_dir=artifact_root / "launcher",
        manifest_dir=artifact_root / "launchers",
    )


def validate_layout(
    paths: LaunchPaths,
    *,
    enforce_active_root: bool = True,
) -> LaunchPaths:
    """Validate source/artifact separation before touching the artifact root."""

    all_paths = (
        paths.checkout_root,
        paths.package_root,
        paths.artifact_root,
        paths.venv_python,
        paths.config_path,
        paths.simulator_executable,
        paths.simulator_identity,
        paths.launcher_dir,
        paths.manifest_dir,
    )
    for path in all_paths:
        if _is_forbidden_recovery_path(path):
            raise LaunchError(
                f"corrupt recovery tree is forbidden for v27 launch: {path}"
            )
        if _is_forbidden_sibling_checkpoint_path(path):
            raise LaunchError(
                f"sibling recovery checkpoint tree is forbidden for v27 launch: {path}"
            )

    # A normal ``venv/bin/python`` is a symlink to /usr/bin/python.  Keep its
    # selected path lexical so the venv location, rather than the interpreter
    # symlink target, remains the launch provenance. Other paths are resolved
    # so simulator or artifact symlink escapes still fail closed.
    resolved = LaunchPaths(
        checkout_root=paths.checkout_root.resolve(strict=False),
        package_root=paths.package_root.resolve(strict=False),
        artifact_root=paths.artifact_root.resolve(strict=False),
        venv_python=_absolute_without_symlink_resolution(paths.venv_python),
        config_path=paths.config_path.resolve(strict=False),
        simulator_executable=paths.simulator_executable.resolve(strict=False),
        simulator_identity=paths.simulator_identity.resolve(strict=False),
        launcher_dir=paths.launcher_dir.resolve(strict=False),
        manifest_dir=paths.manifest_dir.resolve(strict=False),
    )
    if enforce_active_root:
        expected = ACTIVE_ARTIFACT_ROOT.resolve(strict=False)
        if resolved.artifact_root != expected:
            raise LaunchError(
                "v27 artifact root must be the original active runtime: "
                f"expected={expected}, actual={resolved.artifact_root}"
            )

    if resolved.artifact_root == Path(resolved.artifact_root.anchor):
        raise LaunchError("artifact root cannot be a filesystem root")
    if _is_within(resolved.artifact_root, resolved.checkout_root) or _is_within(
        resolved.checkout_root,
        resolved.artifact_root,
    ):
        raise LaunchError(
            "artifact root and source checkout must be completely disjoint"
        )
    for path in (
        resolved.venv_python,
        resolved.simulator_executable,
        resolved.simulator_identity,
        resolved.launcher_dir,
        resolved.manifest_dir,
    ):
        if not _is_within(path, resolved.artifact_root):
            raise LaunchError(f"runtime path escapes the active artifact root: {path}")
    if not _is_within(resolved.config_path, resolved.checkout_root):
        raise LaunchError("v27 config must remain inside the source checkout")
    return resolved


def require_wsl() -> None:
    if os.name != "posix" or not sys.platform.startswith("linux"):
        raise LaunchError("the v27 launcher must run inside WSL")
    version = ""
    for candidate in (Path("/proc/sys/kernel/osrelease"), Path("/proc/version")):
        try:
            version += candidate.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
    if "microsoft" not in version.casefold():
        raise LaunchError("the v27 launcher requires a WSL Linux kernel")


def require_exact_artifact_environment(paths: LaunchPaths) -> None:
    raw = os.environ.get("STS2_ARTIFACT_ROOT")
    if raw:
        selected = Path(raw).expanduser().resolve(strict=False)
        if selected != paths.artifact_root:
            raise LaunchError(
                "STS2_ARTIFACT_ROOT disagrees with the fixed v27 runtime: "
                f"environment={selected}, expected={paths.artifact_root}"
            )


def artifact_canary(root: Path) -> None:
    """Prove durable read/write/delete behavior without leaving residue."""

    if not root.is_dir():
        raise LaunchError(f"active artifact root does not exist: {root}")
    payload = os.urandom(128)
    path = root / f".v27-launch-canary-{uuid.uuid4().hex}"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = None
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if path.read_bytes() != payload:
            raise LaunchError("artifact canary read-back did not match written bytes")
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            path.unlink()
        except FileNotFoundError:
            pass
    if path.exists():
        raise LaunchError(f"artifact canary could not be deleted: {path}")
    try:
        directory_fd = os.open(root, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    serialized = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
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
            # Windows and some WSL-mounted filesystems cannot open a directory
            # descriptor. The file itself was fsynced before the atomic rename.
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


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
        raise LaunchError(
            f"preflight command failed ({result.returncode}): "
            f"{' '.join(command)}\n{output[-4000:]}"
        )
    return result.stdout


def _git_provenance(paths: LaunchPaths) -> dict[str, Any]:
    environment = build_environment(paths)
    commit = _run_checked(
        ("git", "rev-parse", "HEAD"),
        cwd=paths.checkout_root,
        environment=environment,
    ).strip()
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise LaunchError(f"git did not return a full commit identity: {commit!r}")
    status = _run_checked(
        ("git", "status", "--porcelain=v1", "--untracked-files=normal"),
        cwd=paths.checkout_root,
        environment=environment,
    )
    if status.strip():
        raise LaunchError(
            "v27 start requires a clean committed checkout; git status is not empty"
        )
    return {"commit": commit, "clean": True}


def build_environment(paths: LaunchPaths) -> dict[str, str]:
    environment = dict(os.environ)
    environment.pop("PYTHONHOME", None)
    environment.pop("VENV_DIR", None)
    environment.pop("STS2_HEADLESS_SIM_EXE", None)
    environment["STS2_ARTIFACT_ROOT"] = str(paths.artifact_root)
    environment["PYTHONPATH"] = str(paths.package_root)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONUNBUFFERED"] = "1"
    environment["OMP_NUM_THREADS"] = environment.get("OMP_NUM_THREADS", "4")
    environment["MKL_NUM_THREADS"] = environment.get("MKL_NUM_THREADS", "4")
    environment["OPENBLAS_NUM_THREADS"] = environment.get(
        "OPENBLAS_NUM_THREADS",
        "4",
    )
    environment["NUMEXPR_NUM_THREADS"] = environment.get(
        "NUMEXPR_NUM_THREADS",
        "4",
    )
    environment["PATH"] = (
        str(paths.venv_python.parent) + os.pathsep + environment.get("PATH", "")
    )
    return environment


def build_trainer_command(paths: LaunchPaths) -> tuple[str, ...]:
    command = (
        str(paths.venv_python),
        "-m",
        "sts2_rl.train",
        "--profile",
        "preheat",
        "--config",
        str(paths.config_path),
        "--device",
        "cuda",
        "--backend",
        "headless",
        "--sim-exe",
        str(paths.simulator_executable),
        "--sim-identity",
        str(paths.simulator_identity),
    )
    validate_fixed_trainer_command(command)
    return command


def validate_fixed_trainer_command(command: Sequence[str]) -> None:
    forbidden = {"--resume", "--initialize-from"}
    found = forbidden.intersection(command)
    if found:
        raise LaunchError(
            "fresh v27 command contains forbidden checkpoint source flags: "
            + ", ".join(sorted(found))
        )
    if tuple(command).count("--config") != 1 or tuple(command).count("--profile") != 1:
        raise LaunchError("fresh v27 command must select exactly one profile and config")


def _verify_required_files(paths: LaunchPaths) -> None:
    required = {
        "ROCm Python": paths.venv_python,
        "v27 config": paths.config_path,
        "HeadlessSim": paths.simulator_executable,
        "HeadlessSim identity": paths.simulator_identity,
    }
    missing = [f"{label}: {path}" for label, path in required.items() if not path.is_file()]
    if missing:
        raise LaunchError("required v27 runtime files are missing:\n" + "\n".join(missing))
    if not os.access(paths.venv_python, os.X_OK):
        raise LaunchError(f"ROCm Python is not executable: {paths.venv_python}")


def _verify_gpu(paths: LaunchPaths, environment: Mapping[str, str]) -> dict[str, Any]:
    source = """
import json
import torch
payload = {
    "torch_version": torch.__version__,
    "cuda_available": bool(torch.cuda.is_available()),
    "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
}
if payload["cuda_available"]:
    payload["device_name"] = torch.cuda.get_device_name(0)
print(json.dumps(payload, sort_keys=True))
if not payload["cuda_available"] or payload["device_count"] < 1:
    raise SystemExit("ROCm GPU is unavailable")
"""
    output = _run_checked(
        (str(paths.venv_python), "-c", source),
        cwd=paths.package_root,
        environment=environment,
    )
    return json.loads(output.strip().splitlines()[-1])


def _verify_simulator(
    paths: LaunchPaths,
    environment: Mapping[str, str],
) -> dict[str, Any]:
    source = """
import json
import sys
from sts2_rl.simulator_identity import verify_headless_simulator
identity = verify_headless_simulator(sys.argv[1], identity_path=sys.argv[2])
print(json.dumps(identity.to_mapping(), sort_keys=True))
"""
    output = _run_checked(
        (
            str(paths.venv_python),
            "-c",
            source,
            str(paths.simulator_executable),
            str(paths.simulator_identity),
        ),
        cwd=paths.package_root,
        environment=environment,
    )
    return json.loads(output.strip().splitlines()[-1])


def run_preflight(paths: LaunchPaths) -> dict[str, Any]:
    require_wsl()
    paths = validate_layout(paths)
    require_exact_artifact_environment(paths)
    artifact_canary(paths.artifact_root)
    _verify_required_files(paths)
    environment = build_environment(paths)
    git = _git_provenance(paths)
    gpu = _verify_gpu(paths, environment)
    simulator = _verify_simulator(paths, environment)
    command = build_trainer_command(paths)
    _run_checked(
        (*command, "--dry-run"),
        cwd=paths.package_root,
        environment=environment,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "checked_at_utc": _utc_now(),
        "run_name": RUN_NAME,
        "artifact_root": str(paths.artifact_root),
        "git": git,
        "config": {
            "path": str(paths.config_path),
            "sha256": _sha256_file(paths.config_path),
            "profile": "preheat",
        },
        "simulator": simulator,
        "simulator_identity_sha256": _sha256_file(paths.simulator_identity),
        "gpu": gpu,
        "initialization": {
            "mode": "fresh-random-network",
            "resume_checkpoint": None,
            "model_initialization_checkpoint": None,
            "optimizer_state": "new",
            "replay_state": "empty",
            "rng_state": "new-from-config-seed",
        },
        "trainer_command": list(command),
        "trainer_command_has_resume": "--resume" in command,
        "trainer_command_has_initialize_from": "--initialize-from" in command,
        "canary": "write-read-delete-passed",
    }


def capture_process_identity(pid: int) -> ProcessIdentity:
    if isinstance(pid, bool) or pid <= 0:
        raise LaunchError(f"invalid process id: {pid!r}")
    proc = Path("/proc") / str(pid)
    try:
        stat = (proc / "stat").read_text(encoding="utf-8")
        closing = stat.rfind(")")
        if closing < 0:
            raise ValueError("missing command terminator")
        fields = stat[closing + 2 :].split()
        start_ticks = int(fields[19])
        command_line = (proc / "cmdline").read_bytes()
        executable = os.readlink(proc / "exe")
    except (IndexError, OSError, ValueError) as exc:
        raise LaunchError(f"cannot capture process identity for PID {pid}: {exc}") from exc
    if not command_line:
        raise LaunchError(f"PID {pid} has an empty command line")
    return ProcessIdentity(
        pid=pid,
        proc_start_ticks=start_ticks,
        command_line_sha256=_sha256_bytes(command_line),
        executable=executable,
    )


def process_identity_matches(identity: ProcessIdentity) -> bool:
    try:
        current = capture_process_identity(identity.pid)
    except LaunchError:
        return False
    return current == identity


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LaunchError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise LaunchError(f"{label} must contain a JSON object: {path}")
    return payload


def _state_path(paths: LaunchPaths) -> Path:
    return paths.launcher_dir / f"{RUN_NAME}.state.json"


def read_status(
    paths: LaunchPaths,
    *,
    enforce_active_root: bool = True,
) -> dict[str, Any]:
    paths = validate_layout(paths, enforce_active_root=enforce_active_root)
    state_path = _state_path(paths)
    if not state_path.is_file():
        return {
            "schema_version": STATE_SCHEMA_VERSION,
            "run_name": RUN_NAME,
            "status": "not-started",
            "running": False,
            "state_path": str(state_path),
        }
    state = _load_json_object(state_path, label="v27 launcher state")
    if state.get("schema_version") != STATE_SCHEMA_VERSION:
        raise LaunchError("v27 launcher state has an unsupported schema")
    raw_manifest = state.get("manifest_path")
    if not isinstance(raw_manifest, str) or not raw_manifest:
        raise LaunchError("v27 launcher state has no manifest path")
    manifest_path = Path(raw_manifest).resolve(strict=False)
    if not _is_within(manifest_path, paths.manifest_dir):
        raise LaunchError("v27 launcher state manifest escapes the manifest directory")
    manifest = _load_json_object(manifest_path, label="v27 launch manifest")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise LaunchError("v27 launch manifest has an unsupported schema")
    if manifest.get("launch_id") != state.get("launch_id"):
        raise LaunchError("v27 launcher state and manifest launch IDs disagree")
    raw_identity = state.get("process_identity")
    if not isinstance(raw_identity, dict):
        raise LaunchError("v27 launcher state has no process identity")
    try:
        identity = ProcessIdentity(
            pid=int(raw_identity["pid"]),
            proc_start_ticks=int(raw_identity["proc_start_ticks"]),
            command_line_sha256=str(raw_identity["command_line_sha256"]),
            executable=str(raw_identity["executable"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise LaunchError("v27 launcher process identity is malformed") from exc
    running = process_identity_matches(identity)
    return {
        "schema_version": STATE_SCHEMA_VERSION,
        "run_name": RUN_NAME,
        "status": "running" if running else "not-running",
        "running": running,
        "pid": identity.pid,
        "launch_id": state.get("launch_id"),
        "manifest_path": str(manifest_path),
        "log_path": manifest.get("log_path"),
        "manifest_status": manifest.get("status"),
        "process_identity_matches": running,
    }


def start(paths: LaunchPaths) -> dict[str, Any]:
    preflight = run_preflight(paths)
    paths = validate_layout(paths)
    previous = read_status(paths)
    if previous["running"] is True:
        raise LaunchError(
            f"v27 is already running with PID {previous.get('pid')}"
        )

    launch_id = str(uuid.uuid4())
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    paths.launcher_dir.mkdir(parents=True, exist_ok=True)
    paths.manifest_dir.mkdir(parents=True, exist_ok=True)
    log_dir = paths.launcher_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{RUN_NAME}-{timestamp}-{launch_id}.log"
    manifest_path = paths.manifest_dir / f"{RUN_NAME}-{timestamp}-{launch_id}.launch.json"
    command = tuple(str(item) for item in preflight["trainer_command"])
    validate_fixed_trainer_command(command)
    manifest: dict[str, Any] = {
        **preflight,
        "launch_id": launch_id,
        "created_at_utc": _utc_now(),
        "status": "launching",
        "pid": None,
        "process_identity": None,
        "log_path": str(log_path),
        "manifest_path": str(manifest_path),
    }
    atomic_write_json(manifest_path, manifest)

    environment = build_environment(paths)
    try:
        with log_path.open("ab", buffering=0) as log_handle:
            process = subprocess.Popen(
                command,
                cwd=paths.package_root,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
    except OSError as exc:
        manifest.update(
            {
                "status": "failed-to-spawn",
                "failure_observed_at_utc": _utc_now(),
                "failure": str(exc),
            }
        )
        atomic_write_json(manifest_path, manifest)
        raise LaunchError(f"could not spawn the v27 trainer: {exc}") from exc
    time.sleep(0.5)
    if process.poll() is not None:
        manifest.update(
            {
                "status": "failed-to-start",
                "pid": process.pid,
                "failure_observed_at_utc": _utc_now(),
                "returncode": process.returncode,
            }
        )
        atomic_write_json(manifest_path, manifest)
        raise LaunchError(
            f"v27 trainer exited during launch; inspect {log_path}"
        )
    try:
        identity = capture_process_identity(process.pid)
        manifest.update(
            {
                "status": "running",
                "pid": process.pid,
                "process_identity": asdict(identity),
                "started_at_utc": _utc_now(),
            }
        )
        atomic_write_json(manifest_path, manifest)
        state = {
            "schema_version": STATE_SCHEMA_VERSION,
            "run_name": RUN_NAME,
            "launch_id": launch_id,
            "manifest_path": str(manifest_path),
            "process_identity": asdict(identity),
            "written_at_utc": _utc_now(),
        }
        atomic_write_json(_state_path(paths), state)
    except (LaunchError, OSError) as exc:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except OSError:
                pass
        manifest.update(
            {
                "status": "failed-to-record-process-identity",
                "pid": process.pid,
                "failure_observed_at_utc": _utc_now(),
                "failure": str(exc),
            }
        )
        atomic_write_json(manifest_path, manifest)
        raise LaunchError(
            "v27 trainer was terminated because durable process provenance "
            f"could not be recorded: {exc}"
        ) from exc
    return read_status(paths)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        nargs="?",
        default="status",
        choices=("preflight", "start", "status"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = default_paths()
    try:
        if args.action == "preflight":
            payload = run_preflight(paths)
        elif args.action == "start":
            payload = start(paths)
        else:
            require_wsl()
            require_exact_artifact_environment(validate_layout(paths))
            payload = read_status(paths)
    except LaunchError as exc:
        print(f"[v27-launcher] {exc}", file=sys.stderr)
        return 2
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
