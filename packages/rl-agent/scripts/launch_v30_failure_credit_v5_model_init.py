#!/usr/bin/env python3
"""Model-initialization control plane for the reviewed v30 replay-v5 run.

This launcher has deliberately narrow authority.  It can only initialize from
the frozen v29 step-200,225 checkpoint named below, with the reviewed v30 config,
on the hermetic WSL/ROCm cuda-learner + cpu-collector + HeadlessSim runtime.
It has no fresh-start or exact-resume action and accepts no arbitrary trainer
arguments.  Network parameters are inherited; optimizer, replay, RNG, rollout,
counters and evaluation state are deliberately new.

The durable process-identity, detached-supervisor, metrics binding and
native-exit terminalization machinery is reused from the already tested v27
supervisor control plane.  This adapter supplies a new immutable launch
specification and a hermetic environment instead of copying that control plane
or modifying the sealed v29 model-initialization launcher.

Run inside Ubuntu-24.04 WSL using one of::

    python scripts/launch_v30_failure_credit_v5_model_init.py initialize-preflight
    python scripts/launch_v30_failure_credit_v5_model_init.py initialize
    python scripts/launch_v30_failure_credit_v5_model_init.py status

``initialize`` starts a new 250,000-step lineage at environment step zero.  It
must never be represented as an exact continuation of the source counters.
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
    """The fixed v30 model-initialization contract could not be proven."""


RUN_NAME = "full-run-revival-v30-failure-credit-v5-model-init"
SOURCE_RUN_ID = "a01237ca-fa62-4722-b914-22fcebc72a3b"
SOURCE_ENVIRONMENT_STEPS = 200_225
SOURCE_TOTAL_ENVIRONMENT_STEPS = 350_000
TARGET_ENVIRONMENT_STEPS = 250_000
ADDITIONAL_ENVIRONMENT_STEPS = TARGET_ENVIRONMENT_STEPS
SOURCE_CHECKPOINT_ID = "07c67825-59cb-425e-b597-26e34ab27ea8"
SOURCE_MANIFEST_SHA256 = "3037d7762b00bf6e430fc5b48b9b721654ef838a882e72b05fb1d0807ba16921"
SOURCE_METADATA_SHA256 = "0df99333fa8d1ac892d890a230fa2132edc53cc2864699261db7a71e085cd433"
SOURCE_POLICY_VERSION = 3199
SOURCE_LEARNER_UPDATES = 3199
SOURCE_CHECKPOINT_RELATIVE = Path(
    "checkpoints/full-run-revival-v29-failure-credit-v4-exact-continuation-350k"
    f"/run-{SOURCE_RUN_ID}/periodic-step-{SOURCE_ENVIRONMENT_STEPS:09d}"
)
ACTIVE_ARTIFACT_ROOT = Path("/mnt/e/game/project/sts2_mcp_artifacts/runtime")
CONFIG_FILE = "full_run_revival_v30_failure_credit_v5_model_init.toml"

SCHEMA_VERSION = "sts2-v30-failure-credit-v5-model-init-preflight-v1"
STATE_SCHEMA_VERSION = "sts2-v30-failure-credit-v5-model-init-state-v1"
SUPERVISED_SCHEMA_VERSION = "sts2-v30-failure-credit-v5-model-init-supervised-launch-v1"
SUPERVISED_STATE_SCHEMA_VERSION = "sts2-v30-failure-credit-v5-model-init-supervised-state-v1"
# Keep the watchdog deliberately coarse: normal learner updates complete in
# seconds, while a long evaluation/checkpoint still gets ample headroom.  A
# wedged forward must become a durable failure instead of remaining unknown.
LEARNER_STALL_TIMEOUT_SECONDS = 30.0 * 60.0


def _load_supervisor_core() -> Any:
    """Load the tested durable supervisor as a private implementation."""

    path = Path(__file__).resolve().with_name("launch_v27_infinite_random_init.py")
    module_name = "_sts2_v30_model_init_supervisor_core"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load durable supervisor core: {path}")
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
        "--initialize-from",
        str(source_checkpoint_path(paths)),
    )


def validate_fixed_model_initialization_command(
    command: Sequence[str],
    *,
    paths: Any,
) -> None:
    values = tuple(command)
    expected = _unchecked_trainer_command(paths)
    if values != expected:
        raise LaunchError("v30 model-initialization trainer command was modified")
    if values.count("--initialize-from") != 1 or "--resume" in values:
        raise LaunchError("v30 must initialize model parameters, never exact-resume")
    if values.count("--device") != 1 or values[values.index("--device") + 1] != "cuda":
        raise LaunchError("v30 learner must use the ROCm cuda device")
    if values.count("--collector-device") != 1 or values[values.index("--collector-device") + 1] != "cpu":
        raise LaunchError("v30 collector must use cpu")
    if values.count("--backend") != 1 or values[values.index("--backend") + 1] != "headless":
        raise LaunchError("v30 must use the headless backend")


def build_resume_trainer_command(paths: Any) -> tuple[str, ...]:
    """Compatibility hook name required by the shared supervisor core."""

    command = _unchecked_trainer_command(paths)
    validate_fixed_model_initialization_command(command, paths=paths)
    return command


def build_supervisor_command(paths: Any, *, manifest_path: Path) -> tuple[str, ...]:
    """Build the fixed wrapper-owned supervisor command.

    The generic v27 implementation supplies the watchdog machinery, but the
    detached process must re-enter this adapter so the v30 source and target
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
            "model-initialization checkpoint preflight returned the wrong root: "
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
        "source_total_environment_steps": SOURCE_TOTAL_ENVIRONMENT_STEPS,
        "target_total_environment_steps": TARGET_ENVIRONMENT_STEPS,
        "checkpoint_format": "sts2-recurrent-vtrace-checkpoint-v5",
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            raise LaunchError(
                f"model-initialization checkpoint {key} mismatch: " f"expected={value!r}, actual={summary.get(key)!r}"
            )
    files = summary.get("manifest_files")
    if not isinstance(files, list) or not files:
        raise LaunchError("model-initialization preflight returned no manifest files")
    return dict(summary)


def _verify_source_checkpoint(paths: Any, environment: Mapping[str, str]) -> dict[str, Any]:
    checkpoint = source_checkpoint_path(paths)
    if not checkpoint.is_dir():
        raise LaunchError(f"fixed v29 model source checkpoint is missing: {checkpoint}")
    source = r"""
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from sts2_rl.training.checkpointing import preflight_model_initialization
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
validated = preflight_model_initialization(
    checkpoint,
    config=config,
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
        raise LaunchError("model-initialization preflight returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise LaunchError("model-initialization preflight must return an object")
    return _validate_checkpoint_summary(payload, paths=paths)


def run_model_initialization_preflight(paths: Any) -> dict[str, Any]:
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
    # environment.  Source hashing and model-only ABI validation happened above.
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
            "mode": "model-initialization",
            "resume_checkpoint": None,
            "model_initialization_checkpoint": str(source_checkpoint_path(paths)),
            "network_parameters": "inherited",
            "optimizer_state": "fresh",
            "episodic_replay_state": "fresh",
            "failure_credit_replay_state": "fresh-v5",
            "rollout_queue_and_counters": "fresh",
            "evaluation_journal": "fresh",
            "rng_state": "fresh",
            "source_environment_steps": SOURCE_ENVIRONMENT_STEPS,
            "new_lineage_environment_steps": ADDITIONAL_ENVIRONMENT_STEPS,
            "absolute_total_environment_steps": TARGET_ENVIRONMENT_STEPS,
        },
        # The shared durable supervisor calls this compatibility key while
        # re-proving the immutable source.  It is not an exact-resume claim.
        "resume_checkpoint": checkpoint,
        "source_checkpoint": checkpoint,
        "trainer_command": list(command),
        "trainer_command_has_resume": False,
        "trainer_command_has_initialize_from": True,
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


def _validate_supervised_manifest(
    paths: Any,
    *,
    manifest_path: Path,
) -> dict[str, Any]:
    """Validate the model-init manifest without inheriting resume semantics."""

    manifest_path = manifest_path.resolve(strict=False)
    if not _core._is_within(manifest_path, paths.manifest_dir):
        raise LaunchError("supervised manifest escapes the fixed manifest directory")
    manifest = _core._load_json_object(
        manifest_path,
        label="v30 supervised launch manifest",
    )
    if manifest.get("schema_version") != SUPERVISED_SCHEMA_VERSION:
        raise LaunchError("supervised launch manifest has an unsupported schema")
    if manifest.get("run_name") != RUN_NAME:
        raise LaunchError("supervised launch manifest names another lineage")
    launch_id = manifest.get("launch_id")
    if not isinstance(launch_id, str) or not launch_id:
        raise LaunchError("supervised launch manifest has no launch ID")
    command = manifest.get("trainer_command")
    if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
        raise LaunchError("supervised launch manifest has no fixed trainer command")
    if tuple(command) != build_resume_trainer_command(paths):
        raise LaunchError("supervised launch manifest trainer command was modified")
    validate_fixed_model_initialization_command(command, paths=paths)
    initialization = manifest.get("initialization")
    if not isinstance(initialization, Mapping) or initialization.get("mode") != ("model-initialization"):
        raise LaunchError("supervised launch manifest is not model initialization")
    if initialization.get("resume_checkpoint") is not None:
        raise LaunchError("model-initialization manifest falsely claims exact resume")
    if initialization.get("model_initialization_checkpoint") != str(source_checkpoint_path(paths)):
        raise LaunchError("supervised launch manifest selected another model source")
    checkpoint_summary = manifest.get("source_checkpoint")
    if not isinstance(checkpoint_summary, Mapping):
        raise LaunchError("supervised launch manifest has no source-checkpoint proof")
    _validate_checkpoint_summary(checkpoint_summary, paths=paths)
    # The generic supervisor re-verifies the same proof through this internal
    # compatibility key.  Require exact equality so it cannot diverge.
    if manifest.get("resume_checkpoint") != checkpoint_summary:
        raise LaunchError("supervisor checkpoint compatibility proof diverged")
    existing = manifest.get("preexisting_run_directories")
    if not isinstance(existing, list) or not all(isinstance(item, str) for item in existing):
        raise LaunchError("supervised launch manifest has no run-directory snapshot")
    run_root = _core._successor_log_root(paths)
    for item in existing:
        candidate = Path(item).resolve(strict=False)
        if not _core._is_within(candidate, run_root) or not candidate.name.startswith("run-"):
            raise LaunchError("supervised run-directory snapshot escapes its run root")
    return manifest


def _matching_model_initialization_run_start(
    metrics_path: Path,
    *,
    checkpoint: Path,
) -> dict[str, Any] | None:
    """Find the one new-lineage run_start owned by this model source.

    The shared supervisor's original matcher is intentionally exact-resume
    only.  Reusing it unchanged would leave a healthy model-initialization run
    unbound and disable the metrics/native-exit watchdog.  This replacement is
    equally strict about source identity while requiring every reset claim
    that distinguishes model initialization from continuation.
    """

    for event in _core._metrics_events(metrics_path):
        if event.get("event") != "run_start":
            continue
        checkpoint_load = event.get("checkpoint_load")
        if not isinstance(checkpoint_load, Mapping):
            continue
        parent = checkpoint_load.get("parent_checkpoint")
        if not isinstance(parent, str) or Path(parent).resolve(strict=False) != (checkpoint):
            continue
        if checkpoint_load.get("mode") != "model_initialization":
            continue
        if checkpoint_load.get("network_parameters_initialized") is not True:
            continue
        if checkpoint_load.get("optimizer_rollouts_rng_and_counters_reset") is not True:
            continue
        source = checkpoint_load.get("source_checkpoint")
        if not isinstance(source, Mapping):
            continue
        expected_source = {
            "checkpoint_id": SOURCE_CHECKPOINT_ID,
            "environment_steps": SOURCE_ENVIRONMENT_STEPS,
            "manifest_sha256": SOURCE_MANIFEST_SHA256,
            "metadata_sha256": SOURCE_METADATA_SHA256,
            "policy_version": SOURCE_POLICY_VERSION,
            "usage": "model_parameter_initialization_only",
        }
        if any(source.get(key) != value for key, value in expected_source.items()):
            continue
        source_path = source.get("path")
        if not isinstance(source_path, str) or Path(source_path).resolve(strict=False) != checkpoint:
            continue
        state = event.get("state")
        if not isinstance(state, Mapping):
            continue
        if any(
            state.get(field) != 0
            for field in (
                "environment_steps",
                "episodes",
                "learner_updates",
                "policy_version",
                "actor_policy_version",
                "consumed_unrolls",
                "evaluation_episodes",
            )
        ):
            continue
        return event
    return None


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
    _core.validate_fixed_resume_command = validate_fixed_model_initialization_command
    _core._validate_resume_checkpoint_summary = _validate_checkpoint_summary
    _core._verify_resume_checkpoint = _verify_source_checkpoint
    _core._validate_supervised_manifest = _validate_supervised_manifest
    _core._matching_resume_run_start = _matching_model_initialization_run_start
    _core.SUPERVISED_STALL_TIMEOUT_SECONDS = LEARNER_STALL_TIMEOUT_SECONDS


_configure_supervisor_core()


def initialize(paths: Any) -> dict[str, Any]:
    """Detach the persistent supervisor; never detach a bare trainer."""

    preflight = run_model_initialization_preflight(paths)
    paths = _core.validate_layout(paths)
    paths.launcher_dir.mkdir(parents=True, exist_ok=True)
    paths.manifest_dir.mkdir(parents=True, exist_ok=True)
    with _core._terminal_lock(paths, f"{RUN_NAME}-initialize-start"):
        previous = _core.read_status(paths)
        if previous["running"] is True:
            raise LaunchError(
                "v30 model-initialization lineage already has an active process: "
                f"status={previous.get('status')}, launch_id={previous.get('launch_id')}"
            )
        launch_id = str(uuid.uuid4())
        timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        log_dir = paths.launcher_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{RUN_NAME}-initialize-{timestamp}-{launch_id}.log"
        manifest_path = paths.manifest_dir / (f"{RUN_NAME}-initialize-{timestamp}-{launch_id}.launch.json")
        command = tuple(str(item) for item in preflight["trainer_command"])
        validate_fixed_model_initialization_command(command, paths=paths)
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
            raise LaunchError(f"could not spawn the v30 supervisor: {exc}") from exc
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
        choices=("initialize-preflight", "initialize", "status", "supervise"),
    )
    parser.add_argument("--manifest", type=Path, help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = default_paths()
    try:
        if args.action != "supervise" and args.manifest is not None:
            raise LaunchError(f"{args.action} does not accept --manifest")
        if args.action == "initialize-preflight":
            payload = run_model_initialization_preflight(paths)
        elif args.action == "initialize":
            payload = initialize(paths)
        elif args.action == "supervise":
            if args.manifest is None:
                raise LaunchError("internal supervise action requires --manifest")
            payload = _core.supervise(paths, manifest_path=args.manifest)
        else:
            _core.require_wsl()
            _core.require_exact_artifact_environment(_core.validate_layout(paths))
            payload = _core.read_status(paths)
    except (LaunchError, _core.LaunchError) as exc:
        print(f"[v30-model-init-launcher] {exc}", file=sys.stderr)
        return 2
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
