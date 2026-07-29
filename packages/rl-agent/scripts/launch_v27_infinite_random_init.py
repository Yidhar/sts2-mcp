#!/usr/bin/env python3
"""Fail-closed WSL launcher for the v27 infinite-revival lineage.

The command is intentionally fixed.  It cannot accept a resume checkpoint,
model-initialization checkpoint, alternate artifact root, alternate simulator,
or arbitrary trainer arguments.  Run it inside WSL with one of:

    python3 scripts/launch_v27_infinite_random_init.py preflight
    python3 scripts/launch_v27_infinite_random_init.py start
    python3 scripts/launch_v27_infinite_random_init.py resume-preflight
    python3 scripts/launch_v27_infinite_random_init.py resume
    python3 scripts/launch_v27_infinite_random_init.py status

``start`` preserves the original fresh-random launch contract. ``resume`` is a
second, deliberately narrower contract: it can only exactly resume the audited
20,943-step checkpoint, restores optimizer/replay/RNG/counters, and detaches a
persistent supervisor rather than the trainer itself. The supervisor owns the
trainer process, observes native exits such as SIGABRT, and atomically records a
terminal lifecycle event so the dashboard cannot silently leave a dead run in
``stale_unknown``. This script deliberately has no stop command; shutdown
remains an explicit operator action.
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

try:  # pragma: no cover - Windows imports this file only for unit tests.
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]

SCHEMA_VERSION = "sts2-v27-random-init-launch-v1"
STATE_SCHEMA_VERSION = "sts2-v27-random-init-launch-state-v1"
SUPERVISED_SCHEMA_VERSION = "sts2-v27-exact-resume-supervised-launch-v1"
SUPERVISED_STATE_SCHEMA_VERSION = "sts2-v27-exact-resume-supervised-state-v1"
WATCHDOG_EVENT_SCHEMA_VERSION = "sts2-native-exit-watchdog-v1"
RUN_NAME = "full-run-revival-v27-infinite-random-init"
ACTIVE_ARTIFACT_ROOT = Path(
    "/mnt/e/game/project/sts2_mcp_artifacts/runtime"
)
CORRUPT_RECOVERY_ROOT = Path("/mnt/f/backoff/sts2_mcp_artifacts/runtime")
FORBIDDEN_SIBLING_CHECKPOINT_ROOT = Path(
    "/mnt/e/game/project/sts2_mcp_artifacts/checkpoints"
)
RESUME_RUN_ID = "7006cfc5-247f-469c-a0fc-d97f895dfbf7"
RESUME_STEP = 20_943
RESUME_CHECKPOINT_ID = "ca550565-19e9-4262-aa2d-0ef8eb2610fc"
RESUME_MANIFEST_SHA256 = (
    "3e9546cc9a2de129bb301f28af1412b82010e897d5e0cd416c0b76c31af64477"
)
RESUME_METADATA_SHA256 = (
    "33d7ec0f215580abc606cf06744c8a2a6975560071cf374c288f9738ab93250d"
)
RESUME_CHECKPOINT_RELATIVE = Path(
    "checkpoints/full-run-revival-v27-infinite-random-init"
    f"/run-{RESUME_RUN_ID}/periodic-step-{RESUME_STEP:09d}"
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


def resume_checkpoint_path(paths: LaunchPaths) -> Path:
    return (paths.artifact_root / RESUME_CHECKPOINT_RELATIVE).resolve(strict=False)


def build_resume_trainer_command(paths: LaunchPaths) -> tuple[str, ...]:
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
        "--resume",
        str(resume_checkpoint_path(paths)),
    )
    validate_fixed_resume_command(command, paths=paths)
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


def validate_fixed_resume_command(
    command: Sequence[str],
    *,
    paths: LaunchPaths,
) -> None:
    values = tuple(command)
    if "--initialize-from" in values:
        raise LaunchError("exact v27 resume command cannot initialize model parameters")
    if values.count("--resume") != 1:
        raise LaunchError("exact v27 resume command must contain exactly one --resume")
    resume_index = values.index("--resume")
    if resume_index + 1 >= len(values):
        raise LaunchError("exact v27 resume command has no checkpoint path")
    selected = Path(values[resume_index + 1]).resolve(strict=False)
    expected = resume_checkpoint_path(paths)
    if selected != expected:
        raise LaunchError(
            "exact v27 resume command selected an unreviewed checkpoint: "
            f"expected={expected}, actual={selected}"
        )
    if values.count("--config") != 1 or values.count("--profile") != 1:
        raise LaunchError("exact v27 resume command must select one profile and config")


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


def _validate_resume_checkpoint_summary(
    summary: Mapping[str, Any],
    *,
    paths: LaunchPaths,
) -> dict[str, Any]:
    expected_root = resume_checkpoint_path(paths)
    actual_root = Path(str(summary.get("root") or "")).resolve(strict=False)
    expected = {
        "checkpoint_id": RESUME_CHECKPOINT_ID,
        "manifest_sha256": RESUME_MANIFEST_SHA256,
        "metadata_sha256": RESUME_METADATA_SHA256,
        "experiment_run_id": RESUME_RUN_ID,
        "environment_steps": RESUME_STEP,
        "total_environment_steps": 250_000,
    }
    if actual_root != expected_root:
        raise LaunchError(
            "exact-resume checkpoint preflight returned the wrong root: "
            f"expected={expected_root}, actual={actual_root}"
        )
    for key, value in expected.items():
        if summary.get(key) != value:
            raise LaunchError(
                f"exact-resume checkpoint {key} mismatch: "
                f"expected={value!r}, actual={summary.get(key)!r}"
            )
    manifest_files = summary.get("manifest_files")
    if not isinstance(manifest_files, list) or not manifest_files:
        raise LaunchError("exact-resume preflight returned no manifest file identities")
    return dict(summary)


def _verify_resume_checkpoint(
    paths: LaunchPaths,
    environment: Mapping[str, str],
) -> dict[str, Any]:
    checkpoint = resume_checkpoint_path(paths)
    if not checkpoint.is_dir():
        raise LaunchError(f"fixed exact-resume checkpoint is missing: {checkpoint}")
    source = """
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from sts2_rl.training.checkpointing import preflight_training_checkpoint
from sts2_rl.training.config import load_training_config

checkpoint = Path(sys.argv[1]).resolve()
config_path = Path(sys.argv[2]).resolve()
simulator = Path(sys.argv[3]).resolve()
config = load_training_config(profile="preheat", config_path=config_path)
config = replace(
    config,
    runtime=replace(config.runtime, device="cuda"),
    environment=replace(
        config.environment,
        backend="headless",
        sim_exe_path=str(simulator),
    ),
)
validated = preflight_training_checkpoint(
    checkpoint,
    config=config,
    resolved_device="cuda",
    resolved_collector_device="cpu",
)
metadata = validated.metadata
provenance = metadata.get("provenance") or {}
runtime = (metadata.get("training_config") or {}).get("runtime") or {}
training_state = metadata.get("training_state") or {}
payload = {
    "root": str(validated.root),
    "checkpoint_id": validated.manifest.get("checkpoint_id"),
    "manifest_sha256": hashlib.sha256(
        (checkpoint / "checkpoint.manifest.json").read_bytes()
    ).hexdigest(),
    "metadata_sha256": hashlib.sha256(
        (checkpoint / "metadata.json").read_bytes()
    ).hexdigest(),
    "manifest_files": validated.manifest.get("files"),
    "experiment_run_id": provenance.get("experiment_run_id"),
    "environment_steps": training_state.get("environment_steps"),
    "policy_version": training_state.get("policy_version"),
    "learner_updates": training_state.get("learner_updates"),
    "total_environment_steps": runtime.get("total_environment_steps"),
    "config_version": (metadata.get("training_config") or {}).get("version"),
    "checkpoint_format": metadata.get("format"),
}
print(json.dumps(payload, sort_keys=True))
"""
    output = _run_checked(
        (
            str(paths.venv_python),
            "-c",
            source,
            str(checkpoint),
            str(paths.config_path),
            str(paths.simulator_executable),
        ),
        cwd=paths.package_root,
        environment=environment,
    )
    try:
        summary = json.loads(output.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise LaunchError("exact-resume checkpoint preflight returned invalid JSON") from exc
    if not isinstance(summary, dict):
        raise LaunchError("exact-resume checkpoint preflight must return an object")
    return _validate_resume_checkpoint_summary(summary, paths=paths)


def run_resume_preflight(paths: LaunchPaths) -> dict[str, Any]:
    # Reuse all source/artifact/GPU/simulator/config checks from the fresh
    # launcher, then replace its fixed command/provenance with the only reviewed
    # exact-resume source. Production checkpoint preflight hashes every
    # manifest-listed payload and validates immutable config/device/ABI state.
    payload = run_preflight(paths)
    paths = validate_layout(paths)
    environment = build_environment(paths)
    checkpoint = _verify_resume_checkpoint(paths, environment)
    command = build_resume_trainer_command(paths)
    return {
        **payload,
        "schema_version": SUPERVISED_SCHEMA_VERSION,
        "initialization": {
            "mode": "exact-resume",
            "resume_checkpoint": str(resume_checkpoint_path(paths)),
            "model_initialization_checkpoint": None,
            "optimizer_state": "restored",
            "replay_state": "restored",
            "rng_state": "restored",
            "queue_and_counters": "restored",
            "absolute_total_environment_steps": 250_000,
        },
        "resume_checkpoint": checkpoint,
        "trainer_command": list(command),
        "trainer_command_has_resume": True,
        "trainer_command_has_initialize_from": False,
        "supervision": {
            "mode": "persistent-detached-watchdog",
            "native_exit_terminalization": True,
            "automatic_restart": False,
        },
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


def _identity_from_mapping(value: object, *, label: str) -> ProcessIdentity:
    if not isinstance(value, Mapping):
        raise LaunchError(f"{label} has no process identity")
    try:
        pid = value["pid"]
        start_ticks = value["proc_start_ticks"]
        command_hash = value["command_line_sha256"]
        executable = value["executable"]
    except KeyError as exc:
        raise LaunchError(f"{label} process identity is missing {exc.args[0]}") from exc
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        raise LaunchError(f"{label} process PID is invalid")
    if (
        isinstance(start_ticks, bool)
        or not isinstance(start_ticks, int)
        or start_ticks <= 0
    ):
        raise LaunchError(f"{label} process start ticks are invalid")
    if not isinstance(command_hash, str) or not re.fullmatch(
        r"[0-9a-f]{64}",
        command_hash,
    ):
        raise LaunchError(f"{label} command-line hash is invalid")
    if not isinstance(executable, str) or not executable:
        raise LaunchError(f"{label} executable is invalid")
    return ProcessIdentity(
        pid=pid,
        proc_start_ticks=start_ticks,
        command_line_sha256=command_hash,
        executable=executable,
    )


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


def _successor_log_root(paths: LaunchPaths) -> Path:
    return (
        paths.artifact_root / "runs" / RUN_NAME
    ).resolve(strict=False)


@contextmanager
def _terminal_lock(paths: LaunchPaths, launch_id: str):  # type: ignore[no-untyped-def]
    lock_dir = paths.launcher_dir / "watchdog-locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / f"{launch_id}.lock"
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        if fcntl is not None:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        if fcntl is not None:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _metrics_events(metrics_path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    try:
        with metrics_path.open("rb") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                if len(raw_line) > 8 * 1024 * 1024:
                    raise LaunchError(
                        f"metrics line {line_number} exceeds the audited size limit"
                    )
                try:
                    payload = json.loads(raw_line)
                except json.JSONDecodeError:
                    # Only a final torn line can be ignored. A live writer may
                    # have extended it by the time the watchdog retries.
                    if handle.read(1):
                        raise LaunchError(
                            f"metrics contains invalid JSON before EOF at line {line_number}"
                        ) from None
                    break
                if isinstance(payload, dict):
                    events.append(payload)
    except FileNotFoundError:
        return []
    return events


def _matching_resume_run_start(
    metrics_path: Path,
    *,
    checkpoint: Path,
) -> dict[str, Any] | None:
    for event in _metrics_events(metrics_path):
        if event.get("event") != "run_start":
            continue
        checkpoint_load = event.get("checkpoint_load")
        if not isinstance(checkpoint_load, Mapping):
            continue
        parent = checkpoint_load.get("parent_checkpoint")
        if not isinstance(parent, str):
            continue
        if checkpoint_load.get("mode") != "exact_resume":
            continue
        if Path(parent).resolve(strict=False) == checkpoint:
            return event
    return None


def discover_successor_metrics(
    paths: LaunchPaths,
    *,
    preexisting_run_directories: Sequence[str],
) -> tuple[Path, dict[str, Any]] | None:
    root = _successor_log_root(paths)
    if not root.is_dir():
        return None
    previous = {
        Path(item).resolve(strict=False) for item in preexisting_run_directories
    }
    matches: list[tuple[Path, dict[str, Any]]] = []
    checkpoint = resume_checkpoint_path(paths)
    for run_dir in root.glob("run-*"):
        resolved = run_dir.resolve(strict=False)
        if resolved in previous or not run_dir.is_dir():
            continue
        metrics = (run_dir / "metrics.jsonl").resolve(strict=False)
        if not _is_within(metrics, root) or not metrics.is_file():
            continue
        run_start = _matching_resume_run_start(metrics, checkpoint=checkpoint)
        if run_start is not None:
            matches.append((metrics, run_start))
    if len(matches) > 1:
        raise LaunchError(
            "multiple new runs claim the fixed exact-resume checkpoint; "
            "successor binding is ambiguous"
        )
    return matches[0] if matches else None


def _append_jsonl_event_once(
    metrics_path: Path,
    event: Mapping[str, Any],
    *,
    event_id: str,
) -> bool:
    if any(
        item.get("watchdog_event_id") == event_id
        for item in _metrics_events(metrics_path)
    ):
        return False
    serialized = (
        json.dumps(event, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    flags = os.O_WRONLY | os.O_APPEND
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(metrics_path, flags)
    try:
        written = os.write(descriptor, serialized)
        if written != len(serialized):
            raise LaunchError("watchdog terminal metrics write was incomplete")
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
    lifecycle = [
        str(event.get("event"))
        for event in events
        if event.get("event") in {"run_complete", "interrupt", "run_failed"}
    ]
    if (returncode == 0 or returncode is None) and "run_complete" in lifecycle:
        return {
            "status": "completed",
            "kind": "clean_completion",
            "native_abort": False,
            "append_run_failed": False,
        }
    if "interrupt" in lifecycle:
        return {
            "status": "interrupted",
            "kind": "runtime_interrupt",
            "native_abort": False,
            "append_run_failed": False,
        }
    if "run_failed" in lifecycle:
        return {
            "status": "failed",
            "kind": "runtime_reported_failure",
            "native_abort": False,
            "append_run_failed": False,
        }
    signal_number: int | None = (
        -returncode if returncode is not None and returncode < 0 else None
    )
    if signal_number is None and returncode is not None and returncode >= 128:
        candidate = returncode - 128
        if 0 < candidate < signal.NSIG:
            signal_number = candidate
    signal_name: str | None = None
    if signal_number is not None:
        try:
            signal_name = signal.Signals(signal_number).name
        except ValueError:
            signal_name = f"SIGNAL_{signal_number}"
    native_signatures = (
        "AqlQueue::HandleInsufficientScratch",
        "process compute queue fail",
        "Assertion `",
        "Aborted (core dumped)",
    )
    matched = [signature for signature in native_signatures if signature in log_tail]
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
    latest = RESUME_STEP
    for event in events:
        value = event.get("environment_steps")
        if isinstance(value, int) and not isinstance(value, bool):
            latest = max(latest, value)
        state = event.get("state")
        if isinstance(state, Mapping):
            value = state.get("environment_steps")
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
    manifest_path = manifest_path.resolve(strict=False)
    if not _is_within(manifest_path, paths.manifest_dir):
        raise LaunchError("supervised manifest escapes the fixed manifest directory")
    manifest = _load_json_object(manifest_path, label="supervised launch manifest")
    if manifest.get("schema_version") != SUPERVISED_SCHEMA_VERSION:
        raise LaunchError("supervised launch manifest has an unsupported schema")
    launch_id = manifest.get("launch_id")
    if not isinstance(launch_id, str) or not launch_id:
        raise LaunchError("supervised launch manifest has no launch ID")
    with _terminal_lock(paths, launch_id):
        manifest = _load_json_object(
            manifest_path,
            label="supervised launch manifest",
        )
        if manifest.get("status") in {"completed", "interrupted", "failed"}:
            return manifest
        if metrics_path is not None:
            metrics_path = metrics_path.resolve(strict=False)
            if not _is_within(metrics_path, _successor_log_root(paths)):
                raise LaunchError("successor metrics path escapes the fixed run root")
            events = _metrics_events(metrics_path)
        else:
            events = []
        log_path = Path(str(manifest.get("log_path") or ""))
        existing_watchdog = next(
            (
                event
                for event in reversed(events)
                if event.get("event") == "run_failed"
                and event.get("source") == "persistent-native-exit-watchdog"
                and event.get("launch_id") == launch_id
            ),
            None,
        )
        if existing_watchdog is not None and isinstance(
            existing_watchdog.get("classification"),
            Mapping,
        ):
            classification = dict(existing_watchdog["classification"])
            classification["status"] = "failed"
            classification["append_run_failed"] = False
        else:
            classification = _exit_classification(
                returncode=returncode,
                events=events,
                log_tail=_log_tail(log_path),
            )
        event_id: str | None = (
            str(existing_watchdog.get("watchdog_event_id"))
            if existing_watchdog is not None
            and isinstance(existing_watchdog.get("watchdog_event_id"), str)
            else None
        )
        event_appended = False
        if classification["append_run_failed"] is True:
            event_id = _sha256_bytes(
                f"{launch_id}:{WATCHDOG_EVENT_SCHEMA_VERSION}:run_failed".encode()
            )
            failure_event = {
                "event": "run_failed",
                "unix_s": time.time(),
                "schema_version": WATCHDOG_EVENT_SCHEMA_VERSION,
                "source": "persistent-native-exit-watchdog",
                "watchdog_event_id": event_id,
                "launch_id": launch_id,
                "run_name": RUN_NAME,
                "environment_steps": _latest_environment_steps(events),
                "resume_checkpoint": str(resume_checkpoint_path(paths)),
                "resume_checkpoint_id": RESUME_CHECKPOINT_ID,
                "resume_environment_steps": RESUME_STEP,
                "returncode": returncode,
                "classification": classification,
                "reconciliation_reason": reconciliation_reason,
                "supervisor_process_identity": (
                    asdict(supervisor_identity)
                    if supervisor_identity is not None
                    else None
                ),
                "trainer_process_identity": (
                    asdict(trainer_identity) if trainer_identity is not None else None
                ),
                "log_path": str(log_path),
            }
            if metrics_path is not None:
                event_appended = _append_jsonl_event_once(
                    metrics_path,
                    failure_event,
                    event_id=event_id,
                )
            else:
                fallback = (
                    paths.launcher_dir
                    / "terminal-events"
                    / f"{launch_id}.run_failed.json"
                )
                atomic_write_json(fallback, failure_event)
        terminal = {
            "status": classification["status"],
            "classification": classification,
            "returncode": returncode,
            "observed_at_utc": _utc_now(),
            "watchdog_event_id": event_id,
            "watchdog_event_appended": event_appended,
            "metrics_path": str(metrics_path) if metrics_path is not None else None,
            "reconciliation_reason": reconciliation_reason,
        }
        manifest.update(
            {
                "status": classification["status"],
                "terminal": terminal,
                "supervisor_process_identity": (
                    asdict(supervisor_identity)
                    if supervisor_identity is not None
                    else None
                ),
                "trainer_process_identity": (
                    asdict(trainer_identity) if trainer_identity is not None else None
                ),
                "metrics_path": terminal["metrics_path"],
                "terminal_event_id": event_id,
            }
        )
        atomic_write_json(manifest_path, manifest)
        state = {
            "schema_version": SUPERVISED_STATE_SCHEMA_VERSION,
            "run_name": RUN_NAME,
            "launch_id": launch_id,
            "manifest_path": str(manifest_path),
            "status": classification["status"],
            "supervisor_process_identity": (
                asdict(supervisor_identity) if supervisor_identity is not None else None
            ),
            "trainer_process_identity": (
                asdict(trainer_identity) if trainer_identity is not None else None
            ),
            "metrics_path": terminal["metrics_path"],
            "terminal_event_id": event_id,
            "written_at_utc": _utc_now(),
        }
        atomic_write_json(_state_path(paths), state)
        return manifest


_TERMINAL_SUPERVISED_STATUSES = frozenset({"completed", "interrupted", "failed"})


def _optional_process_identity(value: object, *, label: str) -> ProcessIdentity | None:
    if value is None:
        return None
    return _identity_from_mapping(value, label=label)


def _supervised_state_payload(
    *,
    manifest_path: Path,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SUPERVISED_STATE_SCHEMA_VERSION,
        "run_name": RUN_NAME,
        "launch_id": manifest.get("launch_id"),
        "manifest_path": str(manifest_path),
        "status": manifest.get("status"),
        "supervisor_process_identity": manifest.get(
            "supervisor_process_identity"
        ),
        "trainer_process_identity": manifest.get("trainer_process_identity"),
        "metrics_path": manifest.get("metrics_path"),
        "terminal_event_id": manifest.get("terminal_event_id"),
        "written_at_utc": _utc_now(),
        "written_unix_s": time.time(),
    }


def _persist_supervised_manifest(
    paths: LaunchPaths,
    *,
    manifest_path: Path,
    manifest: Mapping[str, Any],
) -> None:
    atomic_write_json(manifest_path, manifest)
    atomic_write_json(
        _state_path(paths),
        _supervised_state_payload(
            manifest_path=manifest_path,
            manifest=manifest,
        ),
    )


def _validate_supervised_manifest(
    paths: LaunchPaths,
    *,
    manifest_path: Path,
) -> dict[str, Any]:
    manifest_path = manifest_path.resolve(strict=False)
    if not _is_within(manifest_path, paths.manifest_dir):
        raise LaunchError("supervised manifest escapes the fixed manifest directory")
    manifest = _load_json_object(manifest_path, label="supervised launch manifest")
    if manifest.get("schema_version") != SUPERVISED_SCHEMA_VERSION:
        raise LaunchError("supervised launch manifest has an unsupported schema")
    if manifest.get("run_name") != RUN_NAME:
        raise LaunchError("supervised launch manifest names another lineage")
    launch_id = manifest.get("launch_id")
    if not isinstance(launch_id, str) or not launch_id:
        raise LaunchError("supervised launch manifest has no launch ID")
    command = manifest.get("trainer_command")
    if not isinstance(command, list) or not all(
        isinstance(item, str) for item in command
    ):
        raise LaunchError("supervised launch manifest has no fixed trainer command")
    expected_command = build_resume_trainer_command(paths)
    if tuple(command) != expected_command:
        raise LaunchError("supervised launch manifest trainer command was modified")
    validate_fixed_resume_command(command, paths=paths)
    initialization = manifest.get("initialization")
    if not isinstance(initialization, Mapping) or initialization.get("mode") != (
        "exact-resume"
    ):
        raise LaunchError("supervised launch manifest is not an exact resume")
    if initialization.get("resume_checkpoint") != str(
        resume_checkpoint_path(paths)
    ):
        raise LaunchError("supervised launch manifest selected another checkpoint")
    checkpoint_summary = manifest.get("resume_checkpoint")
    if not isinstance(checkpoint_summary, Mapping):
        raise LaunchError("supervised launch manifest has no checkpoint proof")
    _validate_resume_checkpoint_summary(checkpoint_summary, paths=paths)
    existing = manifest.get("preexisting_run_directories")
    if not isinstance(existing, list) or not all(
        isinstance(item, str) for item in existing
    ):
        raise LaunchError("supervised launch manifest has no run-directory snapshot")
    run_root = _successor_log_root(paths)
    for item in existing:
        candidate = Path(item).resolve(strict=False)
        if not _is_within(candidate, run_root) or not candidate.name.startswith(
            "run-"
        ):
            raise LaunchError("supervised run-directory snapshot escapes its run root")
    return manifest


def _snapshot_run_directories(paths: LaunchPaths) -> list[str]:
    root = _successor_log_root(paths)
    if not root.is_dir():
        return []
    return sorted(
        str(path.resolve(strict=False))
        for path in root.glob("run-*")
        if path.is_dir()
    )


def _discover_from_manifest(
    paths: LaunchPaths,
    manifest: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]] | None:
    previous = manifest.get("preexisting_run_directories")
    if not isinstance(previous, list) or not all(
        isinstance(item, str) for item in previous
    ):
        raise LaunchError("supervised manifest run snapshot is malformed")
    return discover_successor_metrics(
        paths,
        preexisting_run_directories=previous,
    )


def _record_successor_binding(
    paths: LaunchPaths,
    *,
    manifest_path: Path,
    metrics_path: Path,
    run_start: Mapping[str, Any],
) -> dict[str, Any]:
    manifest = _load_json_object(manifest_path, label="supervised launch manifest")
    launch_id = str(manifest.get("launch_id"))
    with _terminal_lock(paths, launch_id):
        manifest = _validate_supervised_manifest(
            paths,
            manifest_path=manifest_path,
        )
        prior = manifest.get("metrics_path")
        if prior is not None and Path(str(prior)).resolve(strict=False) != metrics_path:
            raise LaunchError("supervised launch successor binding changed")
        state = run_start.get("state")
        manifest.update(
            {
                "metrics_path": str(metrics_path),
                "successor": {
                    "run_id": run_start.get("run_id"),
                    "unix_s": run_start.get("unix_s"),
                    "initial_environment_steps": (
                        state.get("environment_steps")
                        if isinstance(state, Mapping)
                        else None
                    ),
                    "checkpoint_load": run_start.get("checkpoint_load"),
                },
                "successor_bound_at_utc": _utc_now(),
            }
        )
        _persist_supervised_manifest(
            paths,
            manifest_path=manifest_path,
            manifest=manifest,
        )
        return manifest


def _supervised_status(
    paths: LaunchPaths,
    *,
    state: Mapping[str, Any],
    manifest_path: Path,
    allow_reconciliation: bool,
    enforce_active_root: bool,
) -> dict[str, Any]:
    manifest = _validate_supervised_manifest(paths, manifest_path=manifest_path)
    if manifest.get("launch_id") != state.get("launch_id"):
        raise LaunchError("supervised launcher state and manifest launch IDs disagree")
    supervisor = _optional_process_identity(
        manifest.get("supervisor_process_identity"),
        label="supervisor",
    )
    trainer = _optional_process_identity(
        manifest.get("trainer_process_identity"),
        label="trainer",
    )
    supervisor_live = (
        process_identity_matches(supervisor) if supervisor is not None else False
    )
    trainer_live = process_identity_matches(trainer) if trainer is not None else False
    manifest_status = str(manifest.get("status") or "unknown")
    terminal = manifest_status in _TERMINAL_SUPERVISED_STATUSES
    if terminal:
        status = manifest_status
        running = False
    elif supervisor_live and trainer_live:
        status = "running"
        running = True
    elif supervisor_live and trainer is None:
        status = "supervisor-running"
        running = True
    elif supervisor_live:
        status = "supervisor-finalizing"
        running = True
    elif trainer_live:
        status = "unwatched-running"
        running = True
    else:
        written = state.get("written_unix_s")
        recent_launch = (
            isinstance(written, int | float)
            and not isinstance(written, bool)
            and time.time() - float(written) < 30.0
            and supervisor is None
            and trainer is None
        )
        if recent_launch:
            status = "supervisor-launching"
            running = True
        elif allow_reconciliation:
            bound_metrics: Path | None = None
            raw_metrics = manifest.get("metrics_path")
            if isinstance(raw_metrics, str) and raw_metrics:
                bound_metrics = Path(raw_metrics).resolve(strict=False)
            else:
                discovered = _discover_from_manifest(paths, manifest)
                if discovered is not None:
                    bound_metrics, run_start = discovered
                    manifest = _record_successor_binding(
                        paths,
                        manifest_path=manifest_path,
                        metrics_path=bound_metrics,
                        run_start=run_start,
                    )
            finalize_supervised_exit(
                paths,
                manifest_path=manifest_path,
                returncode=None,
                supervisor_identity=supervisor,
                trainer_identity=trainer,
                metrics_path=bound_metrics,
                reconciliation_reason="both_supervisor_and_trainer_are_not_live",
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
            status = "not-running-unreconciled"
            running = False
    return {
        "schema_version": SUPERVISED_STATE_SCHEMA_VERSION,
        "run_name": RUN_NAME,
        "status": status,
        "running": running,
        "launch_id": manifest.get("launch_id"),
        "manifest_path": str(manifest_path),
        "manifest_status": manifest_status,
        "log_path": manifest.get("log_path"),
        "metrics_path": manifest.get("metrics_path"),
        "terminal": manifest.get("terminal"),
        "supervisor": {
            "identity": asdict(supervisor) if supervisor is not None else None,
            "identity_matches": supervisor_live,
        },
        "trainer": {
            "identity": asdict(trainer) if trainer is not None else None,
            "identity_matches": trainer_live,
        },
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
            "schema_version": STATE_SCHEMA_VERSION,
            "run_name": RUN_NAME,
            "status": "not-started",
            "running": False,
            "state_path": str(state_path),
        }
    state = _load_json_object(state_path, label="v27 launcher state")
    state_schema = state.get("schema_version")
    if state_schema not in {STATE_SCHEMA_VERSION, SUPERVISED_STATE_SCHEMA_VERSION}:
        raise LaunchError("v27 launcher state has an unsupported schema")
    raw_manifest = state.get("manifest_path")
    if not isinstance(raw_manifest, str) or not raw_manifest:
        raise LaunchError("v27 launcher state has no manifest path")
    manifest_path = Path(raw_manifest).resolve(strict=False)
    if not _is_within(manifest_path, paths.manifest_dir):
        raise LaunchError("v27 launcher state manifest escapes the manifest directory")
    if state_schema == SUPERVISED_STATE_SCHEMA_VERSION:
        return _supervised_status(
            paths,
            state=state,
            manifest_path=manifest_path,
            allow_reconciliation=True,
            enforce_active_root=enforce_active_root,
        )
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


def supervise(paths: LaunchPaths, *, manifest_path: Path) -> dict[str, Any]:
    """Own one exact-resume trainer until its terminal status is durable."""

    require_wsl()
    paths = validate_layout(paths)
    require_exact_artifact_environment(paths)
    manifest_path = manifest_path.resolve(strict=False)
    manifest = _validate_supervised_manifest(paths, manifest_path=manifest_path)
    launch_id = str(manifest["launch_id"])
    environment = build_environment(paths)
    supervisor_identity: ProcessIdentity | None = None
    trainer_identity: ProcessIdentity | None = None
    trainer: subprocess.Popen[bytes] | None = None
    bound_metrics: Path | None = None
    returncode: int | None = None
    try:
        # Re-prove the mutable inputs from the detached process.  In
        # particular, this invokes production checkpoint preflight with the
        # CLI-effective cuda/headless/simulator config and hashes every file in
        # the checkpoint manifest immediately before exec.
        _verify_required_files(paths)
        git = _git_provenance(paths)
        if git != manifest.get("git"):
            raise LaunchError("git provenance changed after resume preflight")
        config = manifest.get("config")
        if not isinstance(config, Mapping) or config.get("sha256") != (
            _sha256_file(paths.config_path)
        ):
            raise LaunchError("v27 config changed after resume preflight")
        if manifest.get("simulator_identity_sha256") != _sha256_file(
            paths.simulator_identity
        ):
            raise LaunchError("simulator identity changed after resume preflight")
        simulator = _verify_simulator(paths, environment)
        if simulator != manifest.get("simulator"):
            raise LaunchError("simulator binary proof changed after resume preflight")
        checkpoint = _verify_resume_checkpoint(paths, environment)
        if checkpoint != manifest.get("resume_checkpoint"):
            raise LaunchError("checkpoint proof changed after resume preflight")

        supervisor_identity = capture_process_identity(os.getpid())
        with _terminal_lock(paths, launch_id):
            manifest = _validate_supervised_manifest(
                paths,
                manifest_path=manifest_path,
            )
            if manifest.get("status") in _TERMINAL_SUPERVISED_STATUSES:
                return manifest
            manifest.update(
                {
                    "status": "supervisor-running",
                    "supervisor_process_identity": asdict(supervisor_identity),
                    "supervisor_started_at_utc": _utc_now(),
                }
            )
            _persist_supervised_manifest(
                paths,
                manifest_path=manifest_path,
                manifest=manifest,
            )

        command = tuple(str(item) for item in manifest["trainer_command"])
        validate_fixed_resume_command(command, paths=paths)
        log_path = Path(str(manifest.get("log_path") or "")).resolve(strict=False)
        if not _is_within(log_path, paths.launcher_dir / "logs"):
            raise LaunchError("supervised trainer log escapes the launcher log root")
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
        trainer_identity = capture_process_identity(trainer.pid)
        with _terminal_lock(paths, launch_id):
            manifest = _validate_supervised_manifest(
                paths,
                manifest_path=manifest_path,
            )
            manifest.update(
                {
                    "status": "running",
                    "supervisor_process_identity": asdict(supervisor_identity),
                    "trainer_process_identity": asdict(trainer_identity),
                    "trainer_started_at_utc": _utc_now(),
                }
            )
            _persist_supervised_manifest(
                paths,
                manifest_path=manifest_path,
                manifest=manifest,
            )

        def forward_signal(number: int, _frame: object) -> None:
            if trainer is not None and trainer.poll() is None:
                trainer.send_signal(number)

        signal.signal(signal.SIGINT, forward_signal)
        signal.signal(signal.SIGTERM, forward_signal)
        while True:
            returncode = trainer.poll()
            if bound_metrics is None:
                manifest = _load_json_object(
                    manifest_path,
                    label="supervised launch manifest",
                )
                discovered = _discover_from_manifest(paths, manifest)
                if discovered is not None:
                    bound_metrics, run_start = discovered
                    _record_successor_binding(
                        paths,
                        manifest_path=manifest_path,
                        metrics_path=bound_metrics,
                        run_start=run_start,
                    )
            if returncode is not None:
                break
            time.sleep(1.0)

        # A native exit can race the final run_start flush. Retry briefly so a
        # real successor receives run_failed rather than only the fallback
        # launcher incident.
        deadline = time.monotonic() + 5.0
        while bound_metrics is None and time.monotonic() < deadline:
            manifest = _load_json_object(
                manifest_path,
                label="supervised launch manifest",
            )
            discovered = _discover_from_manifest(paths, manifest)
            if discovered is not None:
                bound_metrics, run_start = discovered
                _record_successor_binding(
                    paths,
                    manifest_path=manifest_path,
                    metrics_path=bound_metrics,
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
            metrics_path=bound_metrics,
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
            current = _load_json_object(
                manifest_path,
                label="supervised launch manifest",
            )
            if bound_metrics is None:
                discovered = _discover_from_manifest(paths, current)
                if discovered is not None:
                    bound_metrics, run_start = discovered
                    _record_successor_binding(
                        paths,
                        manifest_path=manifest_path,
                        metrics_path=bound_metrics,
                        run_start=run_start,
                    )
            finalize_supervised_exit(
                paths,
                manifest_path=manifest_path,
                returncode=returncode if returncode is not None else 70,
                supervisor_identity=supervisor_identity,
                trainer_identity=trainer_identity,
                metrics_path=bound_metrics,
                reconciliation_reason=(
                    f"supervisor_exception:{type(exc).__name__}:{exc}"
                ),
            )
        except Exception:
            # Preserve the original supervisor failure. A later ``status``
            # call uses the same idempotent reconciliation path.
            pass
        if isinstance(exc, LaunchError):
            raise
        raise LaunchError(f"supervisor failed: {type(exc).__name__}: {exc}") from exc


def resume(paths: LaunchPaths) -> dict[str, Any]:
    """Detach the fixed exact-resume supervisor, never a bare trainer."""

    preflight = run_resume_preflight(paths)
    paths = validate_layout(paths)
    paths.launcher_dir.mkdir(parents=True, exist_ok=True)
    paths.manifest_dir.mkdir(parents=True, exist_ok=True)
    with _terminal_lock(paths, f"{RUN_NAME}-resume-start"):
        previous = read_status(paths)
        if previous["running"] is True:
            raise LaunchError(
                "v27 already has an active trainer/supervisor: "
                f"status={previous.get('status')}, "
                f"launch_id={previous.get('launch_id')}"
            )
        launch_id = str(uuid.uuid4())
        timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        log_dir = paths.launcher_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{RUN_NAME}-resume-{timestamp}-{launch_id}.log"
        manifest_path = (
            paths.manifest_dir
            / f"{RUN_NAME}-resume-{timestamp}-{launch_id}.launch.json"
        )
        command = tuple(str(item) for item in preflight["trainer_command"])
        validate_fixed_resume_command(command, paths=paths)
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
            "supervisor_command": list(supervisor_command),
            "supervisor_process_identity": None,
            "trainer_process_identity": None,
            "preexisting_run_directories": _snapshot_run_directories(paths),
            "metrics_path": None,
            "log_path": str(log_path),
            "manifest_path": str(manifest_path),
        }
        _persist_supervised_manifest(
            paths,
            manifest_path=manifest_path,
            manifest=manifest,
        )
        environment = build_environment(paths)
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
            identity = capture_process_identity(process.pid)
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
            raise LaunchError(f"could not spawn the v27 supervisor: {exc}") from exc
        # Parent and child serialize this identical identity update. The child
        # remains the sole owner of all trainer and terminal state thereafter.
        with _terminal_lock(paths, launch_id):
            manifest = _validate_supervised_manifest(
                paths,
                manifest_path=manifest_path,
            )
            if manifest.get("supervisor_process_identity") is None:
                manifest["supervisor_process_identity"] = asdict(identity)
            _persist_supervised_manifest(
                paths,
                manifest_path=manifest_path,
                manifest=manifest,
            )
    return read_status(paths)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        nargs="?",
        default="status",
        choices=(
            "preflight",
            "start",
            "resume-preflight",
            "resume",
            "status",
            "supervise",
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help=argparse.SUPPRESS,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = default_paths()
    try:
        if args.action != "supervise" and args.manifest is not None:
            raise LaunchError(f"{args.action} does not accept --manifest")
        if args.action == "preflight":
            payload = run_preflight(paths)
        elif args.action == "start":
            payload = start(paths)
        elif args.action == "resume-preflight":
            payload = run_resume_preflight(paths)
        elif args.action == "resume":
            payload = resume(paths)
        elif args.action == "supervise":
            if args.manifest is None:
                raise LaunchError("internal supervise action requires --manifest")
            payload = supervise(paths, manifest_path=args.manifest)
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
