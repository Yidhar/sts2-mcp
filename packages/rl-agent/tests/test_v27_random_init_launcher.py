from __future__ import annotations

import importlib.util
import os
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
