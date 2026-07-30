from __future__ import annotations

import importlib.util
import json
import signal
import sys
from pathlib import Path
from typing import Any

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PACKAGE_ROOT / "scripts" / "launch_v28_mature_refinement.py"
SPEC = importlib.util.spec_from_file_location("v28_supervised_launcher", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
launcher = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = launcher
SPEC.loader.exec_module(launcher)


def _paths(tmp_path: Path) -> Any:
    checkout = tmp_path / "source"
    package = checkout / "packages" / "rl-agent"
    artifact = tmp_path / "artifacts" / "runtime"
    simulator = artifact / "dependencies" / "HeadlessSim.exe"
    preflight = launcher.v28_preflight.PreflightPaths(
        checkout_root=checkout,
        package_root=package,
        artifact_root=artifact,
        venv_python=artifact / "environments/wsl-rocm/bin/python",
        config_path=package / "config/v28.toml",
        simulator_executable=simulator,
        simulator_identity=simulator.with_name(simulator.name + ".identity.json"),
    )
    return launcher.LaunchPaths(
        preflight=preflight,
        launcher_dir=artifact / "launcher",
        manifest_dir=artifact / "launchers",
        initialization_checkpoint=artifact / launcher.FIXED_INITIALIZATION_RELATIVE,
    )


def _payload(paths: Any) -> dict[str, Any]:
    environment = launcher.v28_preflight.build_trainer_environment(
        paths.preflight,
        base_environment={"PATH": "/usr/bin"},
    )
    return {
        "status": "preflight-passed",
        "run_name": launcher.RUN_NAME,
        "training_started": False,
        "config_fingerprint_sha256": launcher.FIXED_CONFIG_FINGERPRINT_SHA256,
        "initialization": {
            "mode": "model_initialization",
            "checkpoint": str(paths.initialization_checkpoint),
            "checkpoint_id": launcher.FIXED_INITIALIZATION_CHECKPOINT_ID,
            "source_training_state": {
                "environment_steps": launcher.FIXED_INITIALIZATION_STEP,
                "policy_version": launcher.FIXED_INITIALIZATION_POLICY_VERSION,
            },
            "network_parameters_inherited": True,
            "optimizer_rollouts_rng_and_counters_reset": True,
        },
        "trainer_command": list(
            launcher.v28_preflight.build_trainer_command(
                paths.preflight,
                initialize_from=paths.initialization_checkpoint,
            )
        ),
        "trainer_environment": launcher.v28_preflight.trainer_environment_contract(environment),
        "runtime": {
            "torch_version": "test",
            "device_name": "test-gpu",
            "simulator": {"schema_version": "test"},
        },
    }


def _proof(paths: Any) -> dict[str, Any]:
    return {
        "path": str(paths.initialization_checkpoint),
        "checkpoint_id": launcher.FIXED_INITIALIZATION_CHECKPOINT_ID,
        "manifest_sha256": launcher.FIXED_INITIALIZATION_MANIFEST_SHA256,
        "metadata_sha256": launcher.FIXED_INITIALIZATION_METADATA_SHA256,
        "source_environment_steps": launcher.FIXED_INITIALIZATION_STEP,
        "source_policy_version": launcher.FIXED_INITIALIZATION_POLICY_VERSION,
    }


def _write_event(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(event) + "\n")


def _run_start(paths: Any) -> dict[str, Any]:
    return {
        "event": "run_start",
        "unix_s": 1.0,
        "run_id": "successor",
        "state": {"environment_steps": 0, "policy_version": 0},
        "config_fingerprint_sha256": launcher.FIXED_RUNTIME_CONFIG_FINGERPRINT_SHA256,
        "checkpoint_load": {
            "mode": "model_initialization",
            "parent_checkpoint": str(paths.initialization_checkpoint),
            "network_parameters_initialized": True,
            "optimizer_rollouts_rng_and_counters_reset": True,
        },
    }


def test_reviewed_payload_is_fixed_model_initialization_only(tmp_path: Path) -> None:
    paths = launcher.validate_layout(_paths(tmp_path), enforce_active_root=False)
    payload = _payload(paths)
    assert launcher._validate_preflight_payload(payload, paths=paths) == payload
    assert "--resume" not in payload["trainer_command"]
    assert payload["trainer_command"].count("--initialize-from") == 1

    changed = {**payload, "trainer_command": [*payload["trainer_command"], "--resume", "bad"]}
    with pytest.raises(launcher.LaunchError, match=r"never masquerade|differs"):
        launcher._validate_preflight_payload(changed, paths=paths)

    changed = {**payload, "initialization": {**payload["initialization"], "checkpoint": str(tmp_path / "other")}}
    with pytest.raises(launcher.LaunchError, match="checkpoint changed"):
        launcher._validate_preflight_payload(changed, paths=paths)


def test_successor_binding_requires_zeroed_model_initialization(tmp_path: Path) -> None:
    paths = launcher.validate_layout(_paths(tmp_path), enforce_active_root=False)
    metrics = tmp_path / "metrics.jsonl"
    event = _run_start(paths)
    _write_event(metrics, event)
    assert launcher._matching_model_init_run_start(metrics, paths) == event

    event["checkpoint_load"]["mode"] = "exact_resume"
    metrics.write_text(json.dumps(event) + "\n", encoding="utf-8")
    assert launcher._matching_model_init_run_start(metrics, paths) is None

    event["checkpoint_load"]["mode"] = "model_initialization"
    event["state"]["environment_steps"] = launcher.FIXED_INITIALIZATION_STEP
    metrics.write_text(json.dumps(event) + "\n", encoding="utf-8")
    assert launcher._matching_model_init_run_start(metrics, paths) is None


def test_torn_metrics_tail_is_repaired_before_idempotent_watchdog_append(
    tmp_path: Path,
) -> None:
    metrics = tmp_path / "metrics.jsonl"
    _write_event(metrics, {"event": "run_start"})
    with metrics.open("ab") as handle:
        handle.write(b'{"event":"learner_')
    event = {
        "event": "run_failed",
        "source": "persistent-native-exit-watchdog",
        "watchdog_event_id": "event-id",
    }
    assert launcher._append_jsonl_event_once(metrics, event, event_id="event-id") is True
    assert launcher._metrics_events(metrics)[-1] == event
    assert launcher._append_jsonl_event_once(metrics, event, event_id="event-id") is False
    assert metrics.read_bytes().count(b'"watchdog_event_id":"event-id"') == 1


def test_native_abort_terminalizes_bound_successor_as_new_lineage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = launcher.validate_layout(_paths(tmp_path), enforce_active_root=False)
    monkeypatch.setattr(launcher, "_checkpoint_proof", lambda _paths: _proof(paths))
    manifest_path = paths.manifest_dir / "launch.json"
    log_path = paths.launcher_dir / "logs/launch.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("AqlQueue::HandleInsufficientScratch\n", encoding="utf-8")
    manifest = {
        **_payload(paths),
        "schema_version": launcher.MANIFEST_SCHEMA,
        "preflight_status": "preflight-passed",
        "selection": _proof(paths),
        "launch_id": "launch-id",
        "status": "running",
        "preexisting_run_directories": [],
        "metrics_path": None,
        "log_path": str(log_path),
    }
    launcher._persist_manifest(paths, manifest_path=manifest_path, manifest=manifest)
    metrics = paths.artifact_root / "runs" / launcher.RUN_NAME / "run-successor/metrics.jsonl"
    _write_event(metrics, _run_start(paths))
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
    failures = [event for event in launcher._metrics_events(metrics) if event.get("event") == "run_failed"]
    assert len(failures) == 1
    assert failures[0]["initialization_mode"] == "model_initialization"
    assert failures[0]["environment_steps"] == 0
    assert "resume_checkpoint" not in failures[0]

    launcher.finalize_supervised_exit(
        paths,
        manifest_path=manifest_path,
        returncode=-signal.SIGABRT,
        supervisor_identity=supervisor,
        trainer_identity=trainer,
        metrics_path=metrics,
        enforce_active_root=False,
    )
    failures = [event for event in launcher._metrics_events(metrics) if event.get("event") == "run_failed"]
    assert len(failures) == 1
