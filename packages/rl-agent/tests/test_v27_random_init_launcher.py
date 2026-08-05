from __future__ import annotations

import importlib.util
import json
import os
import signal
import sys
from dataclasses import replace
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "launch_v27_infinite_random_init.py"
)
SPEC = importlib.util.spec_from_file_location("v27_random_init_launcher", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
launcher = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = launcher
SPEC.loader.exec_module(launcher)


def _paths(tmp_path: Path) -> object:
    checkout = tmp_path / "source"
    package = checkout / "packages" / "rl-agent"
    artifact = tmp_path / "artifacts" / "runtime"
    simulator = artifact / "dependencies" / "HeadlessSim.exe"
    return launcher.LaunchPaths(
        checkout_root=checkout,
        package_root=package,
        artifact_root=artifact,
        venv_python=artifact / "environments" / "wsl-rocm" / "bin" / "python",
        config_path=package / "config" / "v27.toml",
        simulator_executable=simulator,
        simulator_identity=simulator.with_name(simulator.name + ".identity.json"),
        launcher_dir=artifact / "launcher",
        manifest_dir=artifact / "launchers",
    )


def test_fixed_command_is_random_init_and_exposes_pinned_inputs(tmp_path: Path) -> None:
    paths = launcher.validate_layout(_paths(tmp_path), enforce_active_root=False)
    command = launcher.build_trainer_command(paths)

    assert command[:3] == (str(paths.venv_python), "-m", "sts2_rl.train")
    assert command[command.index("--profile") + 1] == "preheat"
    assert command[command.index("--config") + 1] == str(paths.config_path)
    assert command[command.index("--sim-exe") + 1] == str(
        paths.simulator_executable
    )
    assert command[command.index("--sim-identity") + 1] == str(
        paths.simulator_identity
    )
    assert "--resume" not in command
    assert "--initialize-from" not in command
    assert "--device" in command and "cuda" in command
    assert "--backend" in command and "headless" in command

    with pytest.raises(launcher.LaunchError, match="forbidden checkpoint"):
        launcher.validate_fixed_trainer_command((*command, "--resume", "bad"))
    with pytest.raises(launcher.LaunchError, match="forbidden checkpoint"):
        launcher.validate_fixed_trainer_command(
            (*command, "--initialize-from", "bad")
        )


def test_layout_rejects_overlap_recovery_and_runtime_escape(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    launcher.validate_layout(paths, enforce_active_root=False)

    with pytest.raises(launcher.LaunchError, match="completely disjoint"):
        launcher.validate_layout(
            replace(paths, artifact_root=paths.checkout_root / "artifacts"),
            enforce_active_root=False,
        )
    with pytest.raises(launcher.LaunchError, match="corrupt recovery tree"):
        launcher.validate_layout(
            replace(paths, artifact_root=launcher.CORRUPT_RECOVERY_ROOT),
            enforce_active_root=False,
        )
    with pytest.raises(launcher.LaunchError, match="escapes"):
        launcher.validate_layout(
            replace(paths, venv_python=tmp_path / "other" / "python"),
            enforce_active_root=False,
        )


def test_canary_round_trip_leaves_no_artifact(tmp_path: Path) -> None:
    launcher.artifact_canary(tmp_path)
    assert list(tmp_path.glob(".v27-launch-canary-*")) == []


def test_atomic_json_write_replaces_payload_and_cleans_temp_files(
    tmp_path: Path,
) -> None:
    target = tmp_path / "launcher" / "state.json"
    launcher.atomic_write_json(target, {"value": 1})
    launcher.atomic_write_json(target, {"value": 2})

    assert target.read_text(encoding="utf-8") == '{\n  "value": 2\n}\n'
    assert list(target.parent.glob(".*.tmp")) == []


def test_status_binds_state_to_manifest_and_process_identity(tmp_path: Path) -> None:
    paths = launcher.validate_layout(_paths(tmp_path), enforce_active_root=False)
    manifest_path = paths.manifest_dir / "launch.json"
    identity = {
        "pid": 999_999_999,
        "proc_start_ticks": 1,
        "command_line_sha256": "0" * 64,
        "executable": "/missing/python",
    }
    launcher.atomic_write_json(
        manifest_path,
        {
            "schema_version": launcher.SCHEMA_VERSION,
            "launch_id": "launch-a",
            "status": "running",
            "log_path": "/tmp/log",
        },
    )
    launcher.atomic_write_json(
        paths.launcher_dir / f"{launcher.RUN_NAME}.state.json",
        {
            "schema_version": launcher.STATE_SCHEMA_VERSION,
            "launch_id": "launch-a",
            "manifest_path": str(manifest_path),
            "process_identity": identity,
        },
    )
    status = launcher.read_status(paths, enforce_active_root=False)
    assert status["status"] == "not-running"
    assert status["process_identity_matches"] is False

    launcher.atomic_write_json(
        manifest_path,
        {
            "schema_version": launcher.SCHEMA_VERSION,
            "launch_id": "launch-b",
            "status": "running",
        },
    )
    with pytest.raises(launcher.LaunchError, match="launch IDs disagree"):
        launcher.read_status(paths, enforce_active_root=False)


@pytest.mark.skipif(os.name != "posix", reason="/proc process identity is Linux-only")
def test_process_identity_detects_pid_reuse_or_mutation() -> None:
    identity = launcher.capture_process_identity(os.getpid())
    assert launcher.process_identity_matches(identity)
    assert not launcher.process_identity_matches(
        replace(identity, proc_start_ticks=identity.proc_start_ticks + 1)
    )


def test_default_paths_are_fixed_to_original_e_runtime() -> None:
    paths = launcher.default_paths()
    assert paths.artifact_root == Path(
        "/mnt/e/game/project/sts2_mcp_artifacts/runtime"
    )
    assert "/mnt/f/" not in paths.venv_python.as_posix().casefold()
    assert "/mnt/f/" not in paths.simulator_executable.as_posix().casefold()
    assert paths.venv_python.as_posix().endswith(
        "/runtime/environments/wsl-rocm/bin/python"
    )


def _resume_checkpoint_summary(paths: object) -> dict[str, object]:
    return {
        "root": str(launcher.resume_checkpoint_path(paths)),
        "checkpoint_id": launcher.RESUME_CHECKPOINT_ID,
        "manifest_sha256": launcher.RESUME_MANIFEST_SHA256,
        "metadata_sha256": launcher.RESUME_METADATA_SHA256,
        "manifest_files": [{"path": "model.pt", "sha256": "1" * 64}],
        "experiment_run_id": launcher.RESUME_RUN_ID,
        "environment_steps": launcher.RESUME_STEP,
        "policy_version": 325,
        "learner_updates": 325,
        "total_environment_steps": 250_000,
        "config_version": "sts2-rl-v2",
        "checkpoint_format": "sts2-rl-training-checkpoint-v1",
    }


def _append_event(path: Path, event: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")


def test_stage_aware_watchdog_only_arms_inside_an_unfinished_learner_update(
    tmp_path: Path,
) -> None:
    metrics = tmp_path / "metrics.jsonl"
    _append_event(metrics, {"event": "run_start"})
    assert not launcher._learner_update_in_flight(
        launcher._last_metrics_event(metrics)
    )

    _append_event(metrics, {"event": "learner_update_start", "update_number": 7})
    assert launcher._learner_update_in_flight(launcher._last_metrics_event(metrics))

    _append_event(
        metrics,
        {
            "event": "learner_progress",
            "update_number": 7,
            "stage": "recurrent_step_forward_start",
        },
    )
    event = launcher._last_metrics_event(metrics)
    assert event is not None
    assert event["stage"] == "recurrent_step_forward_start"
    assert launcher._learner_update_in_flight(event)

    _append_event(metrics, {"event": "learner_update", "update_number": 7})
    assert not launcher._learner_update_in_flight(
        launcher._last_metrics_event(metrics)
    )

    # Evaluation/checkpoint silence must not be classified as a learner stall.
    _append_event(metrics, {"event": "evaluation_start", "environment_steps": 25_000})
    assert not launcher._learner_update_in_flight(
        launcher._last_metrics_event(metrics)
    )


def test_last_metrics_event_ignores_a_torn_final_write(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.jsonl"
    _append_event(metrics, {"event": "learner_update", "update_number": 9})
    with metrics.open("ab") as handle:
        handle.write(b'{"event":"learner_progress"')

    assert launcher._last_metrics_event(metrics) == {
        "event": "learner_update",
        "update_number": 9,
    }


def _supervised_manifest(
    paths: object,
    *,
    manifest_path: Path,
    log_path: Path,
    status: str = "running",
) -> dict[str, object]:
    return {
        "schema_version": launcher.SUPERVISED_SCHEMA_VERSION,
        "run_name": launcher.RUN_NAME,
        "launch_id": "supervised-launch-a",
        "status": status,
        "trainer_command": list(launcher.build_resume_trainer_command(paths)),
        "initialization": {
            "mode": "exact-resume",
            "resume_checkpoint": str(launcher.resume_checkpoint_path(paths)),
        },
        "resume_checkpoint": _resume_checkpoint_summary(paths),
        "preexisting_run_directories": [],
        "supervisor_process_identity": None,
        "trainer_process_identity": None,
        "metrics_path": None,
        "log_path": str(log_path),
        "manifest_path": str(manifest_path),
    }


def test_exact_resume_command_has_one_reviewed_source_and_no_model_init(
    tmp_path: Path,
) -> None:
    paths = launcher.validate_layout(_paths(tmp_path), enforce_active_root=False)
    command = launcher.build_resume_trainer_command(paths)

    assert command.count("--resume") == 1
    assert command[command.index("--resume") + 1] == str(
        launcher.resume_checkpoint_path(paths)
    )
    assert "--initialize-from" not in command
    assert launcher.RESUME_RUN_ID in command[command.index("--resume") + 1]
    assert command[command.index("--resume") + 1].endswith(
        "periodic-step-000020943"
    )

    with pytest.raises(launcher.LaunchError, match="unreviewed checkpoint"):
        launcher.validate_fixed_resume_command(
            (*command[:-1], str(tmp_path / "other-checkpoint")),
            paths=paths,
        )
    with pytest.raises(launcher.LaunchError, match="cannot initialize"):
        launcher.validate_fixed_resume_command(
            (*command, "--initialize-from", "bad"),
            paths=paths,
        )


def test_resume_checkpoint_summary_is_pinned_by_ids_hashes_and_step(
    tmp_path: Path,
) -> None:
    paths = launcher.validate_layout(_paths(tmp_path), enforce_active_root=False)
    summary = _resume_checkpoint_summary(paths)
    assert launcher._validate_resume_checkpoint_summary(summary, paths=paths) == summary

    for key, wrong in (
        ("checkpoint_id", "other"),
        ("manifest_sha256", "0" * 64),
        ("metadata_sha256", "0" * 64),
        ("environment_steps", launcher.RESUME_STEP + 1),
    ):
        altered = {**summary, key: wrong}
        with pytest.raises(launcher.LaunchError, match=f"{key} mismatch"):
            launcher._validate_resume_checkpoint_summary(altered, paths=paths)


def test_supervised_manifest_rejects_command_drift(tmp_path: Path) -> None:
    paths = launcher.validate_layout(_paths(tmp_path), enforce_active_root=False)
    manifest_path = paths.manifest_dir / "resume.launch.json"
    manifest = _supervised_manifest(
        paths,
        manifest_path=manifest_path,
        log_path=paths.launcher_dir / "logs" / "resume.log",
    )
    manifest["trainer_command"] = [
        *manifest["trainer_command"],
        "--initialize-from",
        "bad",
    ]
    launcher.atomic_write_json(manifest_path, manifest)

    with pytest.raises(launcher.LaunchError, match="command was modified"):
        launcher._validate_supervised_manifest(paths, manifest_path=manifest_path)


def test_watchdog_native_abort_appends_one_auditable_failure_and_terminalizes(
    tmp_path: Path,
) -> None:
    paths = launcher.validate_layout(_paths(tmp_path), enforce_active_root=False)
    manifest_path = paths.manifest_dir / "resume.launch.json"
    log_path = paths.launcher_dir / "logs" / "resume.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "AqlQueue::HandleInsufficientScratch Assertion `queue' failed\n",
        encoding="utf-8",
    )
    launcher.atomic_write_json(
        manifest_path,
        _supervised_manifest(
            paths,
            manifest_path=manifest_path,
            log_path=log_path,
        ),
    )
    metrics = (
        paths.artifact_root
        / "runs"
        / launcher.RUN_NAME
        / "run-successor"
        / "metrics.jsonl"
    )
    _append_event(
        metrics,
        {
            "event": "run_start",
            "unix_s": 1.0,
            "run_id": "successor",
            "state": {"environment_steps": launcher.RESUME_STEP},
            "checkpoint_load": {
                "mode": "exact_resume",
                "parent_checkpoint": str(launcher.resume_checkpoint_path(paths)),
            },
        },
    )
    supervisor = launcher.ProcessIdentity(101, 1, "1" * 64, "/python")
    trainer = launcher.ProcessIdentity(102, 2, "2" * 64, "/python")

    terminal = launcher.finalize_supervised_exit(
        paths,
        manifest_path=manifest_path,
        returncode=-signal.SIGABRT,
        supervisor_identity=supervisor,
        trainer_identity=trainer,
        metrics_path=metrics,
        enforce_active_root=False,
    )
    assert terminal["status"] == "failed"
    assert terminal["terminal"]["classification"]["kind"] == "native_abort"
    assert terminal["terminal"]["classification"]["native_abort"] is True
    events = launcher._metrics_events(metrics)
    failures = [event for event in events if event.get("event") == "run_failed"]
    assert len(failures) == 1
    assert failures[0]["source"] == "persistent-native-exit-watchdog"
    assert failures[0]["resume_checkpoint_id"] == launcher.RESUME_CHECKPOINT_ID
    assert failures[0]["trainer_process_identity"]["pid"] == 102

    # Terminalization is retry-safe across a supervisor/status race.
    launcher.finalize_supervised_exit(
        paths,
        manifest_path=manifest_path,
        returncode=-signal.SIGABRT,
        supervisor_identity=supervisor,
        trainer_identity=trainer,
        metrics_path=metrics,
        enforce_active_root=False,
    )
    failures = [
        event
        for event in launcher._metrics_events(metrics)
        if event.get("event") == "run_failed"
    ]
    assert len(failures) == 1


def test_watchdog_clean_completion_never_appends_run_failed(tmp_path: Path) -> None:
    paths = launcher.validate_layout(_paths(tmp_path), enforce_active_root=False)
    manifest_path = paths.manifest_dir / "resume.launch.json"
    log_path = paths.launcher_dir / "logs" / "resume.log"
    launcher.atomic_write_json(
        manifest_path,
        _supervised_manifest(
            paths,
            manifest_path=manifest_path,
            log_path=log_path,
        ),
    )
    metrics = (
        paths.artifact_root
        / "runs"
        / launcher.RUN_NAME
        / "run-successor"
        / "metrics.jsonl"
    )
    _append_event(metrics, {"event": "run_start", "unix_s": 1.0})
    _append_event(
        metrics,
        {
            "event": "run_complete",
            "unix_s": 2.0,
            "environment_steps": 250_000,
        },
    )
    terminal = launcher.finalize_supervised_exit(
        paths,
        manifest_path=manifest_path,
        returncode=0,
        supervisor_identity=None,
        trainer_identity=None,
        metrics_path=metrics,
        enforce_active_root=False,
    )

    assert terminal["status"] == "completed"
    assert not any(
        event.get("event") == "run_failed"
        for event in launcher._metrics_events(metrics)
    )
