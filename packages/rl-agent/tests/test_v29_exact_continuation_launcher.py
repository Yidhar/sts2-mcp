from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CHECKOUT_ROOT = PACKAGE_ROOT.parents[1]
SCRIPT = PACKAGE_ROOT / "scripts" / "launch_v29_exact_continuation_350k.py"
CONTRACT = CHECKOUT_ROOT / "contracts/exact-resume-checkpoints/v29-failure-credit-v4-100k.json"
SPEC = importlib.util.spec_from_file_location(
    "v29_exact_continuation_350k_launcher",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
launcher = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = launcher
SPEC.loader.exec_module(launcher)


def _paths(tmp_path: Path) -> Any:
    checkout = tmp_path / "source"
    package = checkout / "packages" / "rl-agent"
    artifact = tmp_path / "artifacts" / "runtime"
    simulator = artifact / "dependencies" / "HeadlessSim.exe"
    return launcher.LaunchPaths(
        checkout_root=checkout,
        package_root=package,
        artifact_root=artifact,
        venv_python=artifact / "environments/wsl-rocm/bin/python",
        config_path=(package / "config/experiments" / launcher.CONFIG_FILE),
        simulator_executable=simulator,
        simulator_identity=simulator.with_name(simulator.name + ".identity.json"),
        launcher_dir=artifact / "launcher",
        manifest_dir=artifact / "launchers",
    )


def _checkpoint_summary(paths: Any) -> dict[str, Any]:
    return {
        "root": str(launcher.source_checkpoint_path(paths)),
        "checkpoint_id": launcher.SOURCE_CHECKPOINT_ID,
        "manifest_sha256": launcher.SOURCE_MANIFEST_SHA256,
        "metadata_sha256": launcher.SOURCE_METADATA_SHA256,
        "manifest_files": [{"path": "model.pt", "sha256": "1" * 64}],
        "experiment_run_id": launcher.SOURCE_RUN_ID,
        "environment_steps": launcher.SOURCE_ENVIRONMENT_STEPS,
        "policy_version": launcher.SOURCE_POLICY_VERSION,
        "learner_updates": launcher.SOURCE_LEARNER_UPDATES,
        "source_total_environment_steps": launcher.SOURCE_ENVIRONMENT_STEPS,
        "target_total_environment_steps": launcher.TARGET_ENVIRONMENT_STEPS,
        "checkpoint_format": "sts2-recurrent-vtrace-checkpoint-v5",
    }


def test_launcher_pins_the_formal_v29_100k_exact_resume_contract() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))

    assert contract["exact_resume_permitted"] is True
    assert contract["expected_load_mode"] == "exact_resume"
    assert contract["usage"] == "exact_resume_only"
    assert contract["source_run_id"] == launcher.SOURCE_RUN_ID
    assert contract["checkpoint_id"] == launcher.SOURCE_CHECKPOINT_ID
    assert contract["manifest_sha256"] == launcher.SOURCE_MANIFEST_SHA256
    assert contract["metadata_sha256"] == launcher.SOURCE_METADATA_SHA256
    assert contract["relative_path"] == (launcher.SOURCE_CHECKPOINT_RELATIVE.as_posix())
    assert contract["environment_steps"] == launcher.SOURCE_ENVIRONMENT_STEPS
    assert contract["policy_version"] == launcher.SOURCE_POLICY_VERSION
    assert contract["training_state"]["learner_updates"] == (launcher.SOURCE_LEARNER_UPDATES)
    assert contract["expected_additional_environment_steps"] == (launcher.ADDITIONAL_ENVIRONMENT_STEPS)
    assert contract["expected_total_environment_steps"] == (launcher.TARGET_ENVIRONMENT_STEPS)
    assert contract["continuation_config"].endswith(launcher.CONFIG_FILE)


def test_fixed_command_is_exact_resume_from_v29_100k_to_absolute_350k(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    command = launcher.build_resume_trainer_command(paths)

    assert command[:3] == (str(paths.venv_python), "-m", "sts2_rl.train")
    assert command.count("--resume") == 1
    assert command[command.index("--resume") + 1] == str(launcher.source_checkpoint_path(paths))
    assert launcher.SOURCE_RUN_ID in command[command.index("--resume") + 1]
    assert command[command.index("--resume") + 1].endswith("periodic-step-000100000")
    assert command[command.index("--config") + 1] == str(paths.config_path)
    assert command[command.index("--device") + 1] == "cuda"
    assert command[command.index("--collector-device") + 1] == "cpu"
    assert command[command.index("--backend") + 1] == "headless"
    assert "--initialize-from" not in command
    assert launcher.ADDITIONAL_ENVIRONMENT_STEPS == 250_000
    assert launcher.TARGET_ENVIRONMENT_STEPS == 350_000

    with pytest.raises(launcher.LaunchError, match="command was modified"):
        launcher.validate_fixed_resume_command(
            (*command[:-1], str(tmp_path / "another-checkpoint")),
            paths=paths,
        )
    with pytest.raises(launcher.LaunchError, match="command was modified"):
        launcher.validate_fixed_resume_command(
            (*command, "--initialize-from", "forbidden"),
            paths=paths,
        )


def test_trainer_environment_is_complete_and_hermetic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _paths(tmp_path)
    monkeypatch.setenv("SECRET_FROM_AMBIENT_SHELL", "must-not-leak")
    monkeypatch.setenv("PYTHONHOME", "/ambient/python")

    environment = launcher.build_environment(paths)

    assert set(environment) == {
        "LC_CTYPE",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "PATH",
        "PYTHONNOUSERSITE",
        "PYTHONPATH",
        "PYTHONUNBUFFERED",
        "STS2_ARTIFACT_ROOT",
    }
    assert environment["PATH"] == (f"{paths.venv_python.parent}:/usr/bin:/bin")
    assert environment["PYTHONPATH"] == str(paths.package_root)
    assert environment["STS2_ARTIFACT_ROOT"] == str(paths.artifact_root)
    assert "SECRET_FROM_AMBIENT_SHELL" not in environment
    assert "PYTHONHOME" not in environment


def test_checkpoint_preflight_summary_is_fully_pinned(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    summary = _checkpoint_summary(paths)

    assert launcher._validate_checkpoint_summary(summary, paths=paths) == summary

    for key, wrong in (
        ("checkpoint_id", "other"),
        ("manifest_sha256", "0" * 64),
        ("metadata_sha256", "0" * 64),
        ("experiment_run_id", "other"),
        ("environment_steps", 99_999),
        ("policy_version", 1_571),
        ("learner_updates", 1_571),
        ("source_total_environment_steps", 99_999),
        ("target_total_environment_steps", 349_999),
        ("checkpoint_format", "wrong-format"),
    ):
        mutated = {**summary, key: wrong}
        with pytest.raises(launcher.LaunchError, match=key):
            launcher._validate_checkpoint_summary(mutated, paths=paths)

    with pytest.raises(launcher.LaunchError, match="manifest file identities"):
        launcher._validate_checkpoint_summary(
            {**summary, "manifest_files": []},
            paths=paths,
        )


def test_resume_preflight_emits_absolute_horizon_and_effective_runtime_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _paths(tmp_path)
    paths.config_path.parent.mkdir(parents=True, exist_ok=True)
    paths.config_path.write_text("[runtime]\ntotal_environment_steps = 350000\n")
    paths.simulator_identity.parent.mkdir(parents=True, exist_ok=True)
    paths.simulator_identity.write_text("{}\n", encoding="utf-8")
    commands: list[tuple[str, ...]] = []

    monkeypatch.setattr(launcher._core, "require_wsl", lambda: None)
    monkeypatch.setattr(launcher._core, "validate_layout", lambda value: value)
    monkeypatch.setattr(
        launcher._core,
        "require_exact_artifact_environment",
        lambda _paths: None,
    )
    monkeypatch.setattr(launcher._core, "artifact_canary", lambda _root: None)
    monkeypatch.setattr(launcher._core, "_verify_required_files", lambda _paths: None)
    monkeypatch.setattr(
        launcher._core,
        "_git_provenance",
        lambda _paths: {"commit": "a" * 40, "worktree_clean": True},
    )
    monkeypatch.setattr(
        launcher._core,
        "_verify_gpu",
        lambda _paths, _environment: {"cuda_available": True},
    )
    monkeypatch.setattr(
        launcher._core,
        "_verify_simulator",
        lambda _paths, _environment: {"status": "verified"},
    )
    monkeypatch.setattr(
        launcher,
        "_verify_source_checkpoint",
        lambda _paths, _environment: _checkpoint_summary(_paths),
    )

    def fake_run_checked(
        command: Any,
        *,
        cwd: Path,
        environment: dict[str, str],
    ) -> str:
        del cwd, environment
        commands.append(tuple(command))
        return "dry-run-passed"

    monkeypatch.setattr(launcher._core, "_run_checked", fake_run_checked)

    payload = launcher.run_resume_preflight(paths)

    assert payload["run_name"] == launcher.RUN_NAME
    assert payload["config"]["absolute_total_environment_steps"] == 350_000
    assert payload["initialization"] == {
        "mode": "exact-resume",
        "resume_checkpoint": str(launcher.source_checkpoint_path(paths)),
        "model_initialization_checkpoint": None,
        "optimizer_state": "restored",
        "episodic_replay_state": "restored",
        "failure_credit_replay_state": "restored",
        "rollout_queue_and_counters": "restored",
        "rng_state": "restored",
        "source_environment_steps": 100_000,
        "additional_environment_steps": 250_000,
        "absolute_total_environment_steps": 350_000,
    }
    assert payload["trainer_command_has_resume"] is True
    assert payload["trainer_command_has_initialize_from"] is False
    assert payload["trainer_environment"]["mode"] == ("complete-hermetic-environment")
    assert payload["supervision"] == {
        "mode": "persistent-detached-watchdog",
        "native_exit_terminalization": True,
        "automatic_restart": False,
    }
    assert commands == [(*launcher.build_resume_trainer_command(paths), "--dry-run")]


def test_adapter_rebinds_durable_supervisor_and_exposes_dashboard_state(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)

    assert launcher._core.RUN_NAME == launcher.RUN_NAME
    assert launcher._core.RESUME_RUN_ID == launcher.SOURCE_RUN_ID
    assert launcher._core.RESUME_STEP == launcher.SOURCE_ENVIRONMENT_STEPS
    assert launcher._core.RESUME_CHECKPOINT_ID == launcher.SOURCE_CHECKPOINT_ID
    assert launcher._core.build_environment is launcher.build_environment
    assert launcher._core.resume_checkpoint_path is launcher.source_checkpoint_path
    assert launcher._core.build_resume_trainer_command is launcher.build_resume_trainer_command

    manifest_path = paths.manifest_dir / "unit.launch.json"
    supervisor_command = launcher.build_supervisor_command(
        paths,
        manifest_path=manifest_path,
    )
    assert supervisor_command == (
        str(paths.venv_python),
        str(SCRIPT.resolve()),
        "supervise",
        "--manifest",
        str(manifest_path),
    )
    assert "launch_v27_infinite_random_init.py" not in supervisor_command

    status = launcher._core.read_status(paths, enforce_active_root=False)
    assert status == {
        "schema_version": launcher.STATE_SCHEMA_VERSION,
        "run_name": launcher.RUN_NAME,
        "status": "not-started",
        "running": False,
        "state_path": str(paths.launcher_dir / f"{launcher.RUN_NAME}.state.json"),
    }

    manifest = {
        "schema_version": launcher.SUPERVISED_SCHEMA_VERSION,
        "run_name": launcher.RUN_NAME,
        "launch_id": "unit-launch",
        "status": "supervisor-launching",
        "trainer_command": list(launcher.build_resume_trainer_command(paths)),
        "initialization": {
            "mode": "exact-resume",
            "resume_checkpoint": str(launcher.source_checkpoint_path(paths)),
        },
        "resume_checkpoint": _checkpoint_summary(paths),
        "preexisting_run_directories": [],
        "supervisor_process_identity": None,
        "trainer_process_identity": None,
        "metrics_path": None,
        "log_path": str(paths.launcher_dir / "logs/unit.log"),
        "manifest_path": str(manifest_path),
    }
    paths.manifest_dir.mkdir(parents=True, exist_ok=True)
    launcher._core.atomic_write_json(manifest_path, manifest)
    assert (
        launcher._core._validate_supervised_manifest(
            paths,
            manifest_path=manifest_path,
        )["launch_id"]
        == "unit-launch"
    )

    launcher._core._persist_supervised_manifest(
        paths,
        manifest_path=manifest_path,
        manifest=manifest,
    )
    state = json.loads((paths.launcher_dir / f"{launcher.RUN_NAME}.state.json").read_text(encoding="utf-8"))
    assert state["schema_version"] == launcher.SUPERVISED_STATE_SCHEMA_VERSION
    assert state["run_name"] == launcher.RUN_NAME
    assert state["status"] == "supervisor-launching"
    assert state["manifest_path"] == str(manifest_path)


def test_cli_has_no_fresh_start_or_model_initialization_action() -> None:
    parser = launcher.build_parser()

    for forbidden in ("start", "initialize", "model-init"):
        with pytest.raises(SystemExit):
            parser.parse_args([forbidden])
