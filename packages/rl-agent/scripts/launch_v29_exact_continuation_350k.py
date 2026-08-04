#!/usr/bin/env python3
"""Exact-resume control plane for the audited v29 100k -> 350k run.

This launcher has deliberately narrow authority.  It can only restore the
atomic v29 100k checkpoint named below, with the reviewed continuation config,
on the hermetic WSL/ROCm cuda-learner + cpu-collector + HeadlessSim runtime.
It has no fresh-start or model-initialization action and accepts no arbitrary
trainer arguments.

The durable process-identity, detached-supervisor, metrics binding and
native-exit terminalization machinery is reused from the already tested v27
exact-resume control plane.  This adapter supplies a new immutable launch
specification and a hermetic environment instead of copying that control plane
or modifying the sealed v29 model-initialization launcher.

Run inside Ubuntu-24.04 WSL using one of::

    python scripts/launch_v29_exact_continuation_350k.py resume-preflight
    python scripts/launch_v29_exact_continuation_350k.py resume
    python scripts/launch_v29_exact_continuation_350k.py status

``resume`` means an *additional* 250,000 environment steps: the restored
counter is 100,000 and the absolute configured horizon is 350,000.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class LaunchError(RuntimeError):
    """The fixed v29 continuation contract could not be proven."""


RUN_NAME = "full-run-revival-v29-failure-credit-v4-exact-continuation-350k"
SOURCE_RUN_ID = "78554f34-a31e-48ed-8e3e-8e58effc2790"
SOURCE_ENVIRONMENT_STEPS = 100_000
TARGET_ENVIRONMENT_STEPS = 350_000
ADDITIONAL_ENVIRONMENT_STEPS = TARGET_ENVIRONMENT_STEPS - SOURCE_ENVIRONMENT_STEPS
SOURCE_CHECKPOINT_ID = "075b8174-1dab-4293-90bd-9c63f0aab277"
SOURCE_MANIFEST_SHA256 = "d7c80dcfbcd222d8ed93f6b7dd8bc4735b5637ac91516fae1b707a55b53a7565"
SOURCE_METADATA_SHA256 = "a8a8396bad5799b1cd96591751c7d8c1c7f3732ad702aeafdd25468cf934c29c"
SOURCE_POLICY_VERSION = 1572
SOURCE_LEARNER_UPDATES = 1572
SOURCE_CHECKPOINT_RELATIVE = Path(
    "checkpoints/full-run-revival-v29-failure-credit-v4-model-init"
    f"/run-{SOURCE_RUN_ID}/periodic-step-{SOURCE_ENVIRONMENT_STEPS:09d}"
)
ACTIVE_ARTIFACT_ROOT = Path("/mnt/e/game/project/sts2_mcp_artifacts/runtime")
CONFIG_FILE = "full_run_revival_v29_failure_credit_v4_exact_continuation_350k.toml"

SCHEMA_VERSION = "sts2-v29-exact-continuation-350k-preflight-v1"
STATE_SCHEMA_VERSION = "sts2-v29-exact-continuation-350k-state-v1"
SUPERVISED_SCHEMA_VERSION = "sts2-v29-exact-continuation-350k-supervised-launch-v1"
SUPERVISED_STATE_SCHEMA_VERSION = "sts2-v29-exact-continuation-350k-supervised-state-v1"
# Keep the watchdog deliberately coarse: normal learner updates complete in
# seconds, while a long evaluation/checkpoint still gets ample headroom.  A
# wedged forward must become a durable failure instead of remaining unknown.
LEARNER_STALL_TIMEOUT_SECONDS = 30.0 * 60.0


def _load_supervisor_core() -> Any:
    """Load the tested exact-resume supervisor as a private implementation."""

    path = Path(__file__).resolve().with_name("launch_v27_infinite_random_init.py")
    # Every adapter gets a private mutable supervisor module.  Recovery
    # adapters rebind the core's pins at import time; sharing one fixed module
    # name would let importing a later lineage silently rewrite this launcher's
    # RUN_NAME/checkpoint identity in the same process.
    module_name = f"_{__name__.replace('.', '_')}_supervisor_core"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load exact-resume supervisor core: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_core = _load_supervisor_core()
LaunchPaths = _core.LaunchPaths


def default_paths() -> Any:
    script_dir = Path(__file__).resolve().parent
    package_root = script_dir.parent
    checkout_root = script_dir.parents[2]
    artifact_root = ACTIVE_ARTIFACT_ROOT
    simulator = artifact_root / "dependencies/sts2-ai/STS2AI/ENV/Sim/HeadlessSim/bin/Release/net9.0/HeadlessSim.exe"
    return LaunchPaths(
        checkout_root=checkout_root,
        package_root=package_root,
        artifact_root=artifact_root,
        venv_python=artifact_root / "environments/wsl-rocm/bin/python",
        config_path=package_root / "config/experiments" / CONFIG_FILE,
        simulator_executable=simulator,
        simulator_identity=simulator.with_name(simulator.name + ".identity.json"),
        launcher_dir=artifact_root / "launcher",
        manifest_dir=artifact_root / "launchers",
    )


def build_environment(paths: Any) -> dict[str, str]:
    """Return the complete reviewed trainer environment, never an overlay."""

    return {
        "LC_CTYPE": "C.UTF-8",
        "MKL_NUM_THREADS": "4",
        "NUMEXPR_NUM_THREADS": "4",
        "OMP_NUM_THREADS": "4",
        "OPENBLAS_NUM_THREADS": "4",
        "PATH": f"{paths.venv_python.parent}:/usr/bin:/bin",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": str(paths.package_root),
        "PYTHONUNBUFFERED": "1",
        "STS2_ARTIFACT_ROOT": str(paths.artifact_root),
    }


def source_checkpoint_path(paths: Any) -> Path:
    return (paths.artifact_root / SOURCE_CHECKPOINT_RELATIVE).resolve(strict=False)


def _unchecked_trainer_command(paths: Any) -> tuple[str, ...]:
    return (
        str(paths.venv_python),
        "-m",
        "sts2_rl.train",
        "--profile",
        "preheat",
        "--config",
        str(paths.config_path),
        "--device",
        "cuda",
        "--collector-device",
        "cpu",
        "--backend",
        "headless",
        "--sim-exe",
        str(paths.simulator_executable),
        "--sim-identity",
        str(paths.simulator_identity),
        "--resume",
        str(source_checkpoint_path(paths)),
    )


def validate_fixed_resume_command(command: Sequence[str], *, paths: Any) -> None:
    values = tuple(command)
    expected = _unchecked_trainer_command(paths)
    if values != expected:
        raise LaunchError("v29 exact-continuation trainer command was modified")
    if values.count("--resume") != 1 or "--initialize-from" in values:
        raise LaunchError("v29 continuation must be exact resume only")
    if values.count("--device") != 1 or values[values.index("--device") + 1] != "cuda":
        raise LaunchError("v29 continuation learner must use the ROCm cuda device")
    if values.count("--collector-device") != 1 or values[values.index("--collector-device") + 1] != "cpu":
        raise LaunchError("v29 continuation collector must use cpu")
    if values.count("--backend") != 1 or values[values.index("--backend") + 1] != "headless":
        raise LaunchError("v29 continuation must use the headless backend")


def build_resume_trainer_command(paths: Any) -> tuple[str, ...]:
    command = _unchecked_trainer_command(paths)
    validate_fixed_resume_command(command, paths=paths)
    return command


def build_supervisor_command(paths: Any, *, manifest_path: Path) -> tuple[str, ...]:
    """Build the fixed wrapper-owned supervisor command.

    The generic v27 implementation supplies the watchdog machinery, but the
    detached process must re-enter this adapter so the v29 source and target
    pins are rebound before any manifest or checkpoint is accepted.
    """

    return (
        str(paths.venv_python),
        str(Path(__file__).resolve()),
        "supervise",
        "--manifest",
        str(manifest_path),
    )


def _validate_checkpoint_summary(summary: Mapping[str, Any], *, paths: Any) -> dict[str, Any]:
    expected_root = source_checkpoint_path(paths)
    actual_root = Path(str(summary.get("root") or "")).resolve(strict=False)
    if actual_root != expected_root:
        raise LaunchError(
            "exact-resume checkpoint preflight returned the wrong root: "
            f"expected={expected_root}, actual={actual_root}"
        )
    expected = {
        "checkpoint_id": SOURCE_CHECKPOINT_ID,
        "manifest_sha256": SOURCE_MANIFEST_SHA256,
        "metadata_sha256": SOURCE_METADATA_SHA256,
        "experiment_run_id": SOURCE_RUN_ID,
        "environment_steps": SOURCE_ENVIRONMENT_STEPS,
        "policy_version": SOURCE_POLICY_VERSION,
        "learner_updates": SOURCE_LEARNER_UPDATES,
        "source_total_environment_steps": SOURCE_ENVIRONMENT_STEPS,
        "target_total_environment_steps": TARGET_ENVIRONMENT_STEPS,
        "checkpoint_format": "sts2-recurrent-vtrace-checkpoint-v5",
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            raise LaunchError(
                f"exact-resume checkpoint {key} mismatch: " f"expected={value!r}, actual={summary.get(key)!r}"
            )
    files = summary.get("manifest_files")
    if not isinstance(files, list) or not files:
        raise LaunchError("exact-resume preflight returned no manifest file identities")
    return dict(summary)


def _verify_source_checkpoint(paths: Any, environment: Mapping[str, str]) -> dict[str, Any]:
    checkpoint = source_checkpoint_path(paths)
    if not checkpoint.is_dir():
        raise LaunchError(f"fixed v29 exact-resume checkpoint is missing: {checkpoint}")
    source = r"""
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
    runtime=replace(config.runtime, device="cuda", collector_device="cpu"),
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
source_runtime = (metadata.get("training_config") or {}).get("runtime") or {}
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
    "source_total_environment_steps": source_runtime.get("total_environment_steps"),
    "target_total_environment_steps": config.runtime.total_environment_steps,
    "checkpoint_format": metadata.get("format"),
}
print(json.dumps(payload, sort_keys=True))
"""
    output = _core._run_checked(
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
        payload = json.loads(output.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise LaunchError("exact-resume checkpoint preflight returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise LaunchError("exact-resume checkpoint preflight must return an object")
    return _validate_checkpoint_summary(payload, paths=paths)


def run_resume_preflight(paths: Any) -> dict[str, Any]:
    _core.require_wsl()
    paths = _core.validate_layout(paths)
    _core.require_exact_artifact_environment(paths)
    _core.artifact_canary(paths.artifact_root)
    _core._verify_required_files(paths)
    environment = build_environment(paths)
    git = _core._git_provenance(paths)
    gpu = _core._verify_gpu(paths, environment)
    simulator = _core._verify_simulator(paths, environment)
    checkpoint = _verify_source_checkpoint(paths, environment)
    command = build_resume_trainer_command(paths)
    # This validates the complete target model/config without starting an
    # environment.  Exact checkpoint hashing/ABI validation happened above.
    _core._run_checked(
        (*command, "--dry-run"),
        cwd=paths.package_root,
        environment=environment,
    )
    return {
        "schema_version": SUPERVISED_SCHEMA_VERSION,
        "checked_at_utc": _core._utc_now(),
        "run_name": RUN_NAME,
        "artifact_root": str(paths.artifact_root),
        "git": git,
        "config": {
            "path": str(paths.config_path),
            "sha256": _core._sha256_file(paths.config_path),
            "profile": "preheat",
            "absolute_total_environment_steps": TARGET_ENVIRONMENT_STEPS,
        },
        "simulator": simulator,
        "simulator_identity_sha256": _core._sha256_file(paths.simulator_identity),
        "gpu": gpu,
        "initialization": {
            "mode": "exact-resume",
            "resume_checkpoint": str(source_checkpoint_path(paths)),
            "model_initialization_checkpoint": None,
            "optimizer_state": "restored",
            "episodic_replay_state": "restored",
            "failure_credit_replay_state": "restored",
            "rollout_queue_and_counters": "restored",
            "rng_state": "restored",
            "source_environment_steps": SOURCE_ENVIRONMENT_STEPS,
            "additional_environment_steps": ADDITIONAL_ENVIRONMENT_STEPS,
            "absolute_total_environment_steps": TARGET_ENVIRONMENT_STEPS,
        },
        "resume_checkpoint": checkpoint,
        "trainer_command": list(command),
        "trainer_command_has_resume": True,
        "trainer_command_has_initialize_from": False,
        "trainer_environment": {
            "mode": "complete-hermetic-environment",
            "set": dict(environment),
            "unset_by_construction": [
                "PYTHONHOME",
                "VENV_DIR",
                "STS2_HEADLESS_SIM_EXE",
            ],
        },
        "supervision": {
            "mode": "persistent-detached-watchdog",
            "native_exit_terminalization": True,
            "learner_stall_watchdog": {
                "enabled": True,
                "timeout_seconds": LEARNER_STALL_TIMEOUT_SECONDS,
                "event": "learner_stall_detected",
                "automatic_restart": False,
            },
            "automatic_restart": False,
        },
        "canary": "write-read-delete-passed",
    }


def _configure_supervisor_core() -> None:
    """Bind the generic durable supervisor to this fixed launch spec."""

    _core.SCHEMA_VERSION = SCHEMA_VERSION
    _core.STATE_SCHEMA_VERSION = STATE_SCHEMA_VERSION
    _core.SUPERVISED_SCHEMA_VERSION = SUPERVISED_SCHEMA_VERSION
    _core.SUPERVISED_STATE_SCHEMA_VERSION = SUPERVISED_STATE_SCHEMA_VERSION
    _core.RUN_NAME = RUN_NAME
    _core.ACTIVE_ARTIFACT_ROOT = ACTIVE_ARTIFACT_ROOT
    _core.RESUME_RUN_ID = SOURCE_RUN_ID
    _core.RESUME_STEP = SOURCE_ENVIRONMENT_STEPS
    _core.RESUME_CHECKPOINT_ID = SOURCE_CHECKPOINT_ID
    _core.RESUME_MANIFEST_SHA256 = SOURCE_MANIFEST_SHA256
    _core.RESUME_METADATA_SHA256 = SOURCE_METADATA_SHA256
    _core.RESUME_CHECKPOINT_RELATIVE = SOURCE_CHECKPOINT_RELATIVE
    _core.build_environment = build_environment
    _core.resume_checkpoint_path = source_checkpoint_path
    _core.build_resume_trainer_command = build_resume_trainer_command
    _core.validate_fixed_resume_command = validate_fixed_resume_command
    _core._validate_resume_checkpoint_summary = _validate_checkpoint_summary
    _core._verify_resume_checkpoint = _verify_source_checkpoint
    _core.SUPERVISED_STALL_TIMEOUT_SECONDS = LEARNER_STALL_TIMEOUT_SECONDS


_configure_supervisor_core()


def resume(paths: Any) -> dict[str, Any]:
    """Detach the persistent supervisor; never detach a bare trainer."""

    preflight = run_resume_preflight(paths)
    paths = _core.validate_layout(paths)
    paths.launcher_dir.mkdir(parents=True, exist_ok=True)
    paths.manifest_dir.mkdir(parents=True, exist_ok=True)
    with _core._terminal_lock(paths, f"{RUN_NAME}-resume-start"):
        previous = _core.read_status(paths)
        if previous["running"] is True:
            raise LaunchError(
                "v29 continuation already has an active trainer/supervisor: "
                f"status={previous.get('status')}, launch_id={previous.get('launch_id')}"
            )
        launch_id = str(uuid.uuid4())
        timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        log_dir = paths.launcher_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{RUN_NAME}-resume-{timestamp}-{launch_id}.log"
        manifest_path = paths.manifest_dir / f"{RUN_NAME}-resume-{timestamp}-{launch_id}.launch.json"
        command = tuple(str(item) for item in preflight["trainer_command"])
        validate_fixed_resume_command(command, paths=paths)
        supervisor_command = build_supervisor_command(
            paths,
            manifest_path=manifest_path,
        )
        manifest: dict[str, Any] = {
            **preflight,
            "launch_id": launch_id,
            "created_at_utc": _core._utc_now(),
            "created_unix_s": time.time(),
            "status": "supervisor-launching",
            "supervisor_command": list(supervisor_command),
            "supervisor_process_identity": None,
            "trainer_process_identity": None,
            "preexisting_run_directories": _core._snapshot_run_directories(paths),
            "metrics_path": None,
            "log_path": str(log_path),
            "manifest_path": str(manifest_path),
        }
        _core._persist_supervised_manifest(
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
            identity = _core.capture_process_identity(process.pid)
        except (OSError, _core.LaunchError, LaunchError) as exc:
            _core.finalize_supervised_exit(
                paths,
                manifest_path=manifest_path,
                returncode=70,
                supervisor_identity=None,
                trainer_identity=None,
                metrics_path=None,
                reconciliation_reason=f"supervisor_spawn_failed:{exc}",
            )
            raise LaunchError(f"could not spawn the v29 continuation supervisor: {exc}") from exc
        with _core._terminal_lock(paths, launch_id):
            manifest = _core._validate_supervised_manifest(
                paths,
                manifest_path=manifest_path,
            )
            if manifest.get("supervisor_process_identity") is None:
                manifest["supervisor_process_identity"] = asdict(identity)
            _core._persist_supervised_manifest(
                paths,
                manifest_path=manifest_path,
                manifest=manifest,
            )
    return _core.read_status(paths)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        nargs="?",
        default="status",
        choices=("resume-preflight", "resume", "status", "supervise"),
    )
    parser.add_argument("--manifest", type=Path, help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = default_paths()
    try:
        if args.action != "supervise" and args.manifest is not None:
            raise LaunchError(f"{args.action} does not accept --manifest")
        if args.action == "resume-preflight":
            payload = run_resume_preflight(paths)
        elif args.action == "resume":
            payload = resume(paths)
        elif args.action == "supervise":
            if args.manifest is None:
                raise LaunchError("internal supervise action requires --manifest")
            payload = _core.supervise(paths, manifest_path=args.manifest)
        else:
            _core.require_wsl()
            _core.require_exact_artifact_environment(_core.validate_layout(paths))
            payload = _core.read_status(paths)
    except (LaunchError, _core.LaunchError) as exc:
        print(f"[v29-continuation-launcher] {exc}", file=sys.stderr)
        return 2
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
