from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PACKAGE_ROOT / "scripts" / "launch_v33_stall_recovery_exact_resume.py"
SPEC = importlib.util.spec_from_file_location(
    "v33_stall_recovery_exact_resume_launcher",
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
        config_path=package / "config/experiments" / launcher.CONFIG_FILE,
        simulator_executable=simulator,
        simulator_identity=simulator.with_name(simulator.name + ".identity.json"),
        launcher_dir=artifact / "launcher",
        manifest_dir=artifact / "launchers",
    )


def _summary(paths: Any) -> dict[str, Any]:
    return {
        "root": str(launcher.source_checkpoint_path(paths)),
        "checkpoint_id": launcher.SOURCE_CHECKPOINT_ID,
        "manifest_sha256": launcher.SOURCE_MANIFEST_SHA256,
        "metadata_sha256": launcher.SOURCE_METADATA_SHA256,
        "manifest_files": [{"path": "network.pt", "sha256": "1" * 64}],
        "experiment_run_id": launcher.SOURCE_RUN_ID,
        "environment_steps": launcher.SOURCE_ENVIRONMENT_STEPS,
        "policy_version": launcher.SOURCE_POLICY_VERSION,
        "learner_updates": launcher.SOURCE_LEARNER_UPDATES,
        "source_total_environment_steps": (launcher.SOURCE_TOTAL_ENVIRONMENT_STEPS),
        "target_total_environment_steps": launcher.TARGET_ENVIRONMENT_STEPS,
        "checkpoint_format": "sts2-recurrent-vtrace-checkpoint-v5",
    }


def test_v33_recovery_is_exact_resume_to_the_original_absolute_horizon(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    command = launcher.build_resume_trainer_command(paths)

    assert command.count("--resume") == 1
    assert "--initialize-from" not in command
    assert command[command.index("--resume") + 1] == str(launcher.source_checkpoint_path(paths))
    assert command[command.index("--config") + 1] == str(paths.config_path)
    assert launcher.SOURCE_ENVIRONMENT_STEPS == 30_089
    assert launcher.TARGET_ENVIRONMENT_STEPS == 100_000
    assert launcher.ADDITIONAL_ENVIRONMENT_STEPS == 69_911
    assert launcher.LEARNER_STALL_TIMEOUT_SECONDS == 300.0
    assert launcher._core.SUPERVISED_STALL_TIMEOUT_SECONDS == 300.0


def test_v33_recovery_checkpoint_identity_is_fully_pinned(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    summary = _summary(paths)

    assert launcher._validate_checkpoint_summary(summary, paths=paths) == summary
    for key, wrong in (
        ("checkpoint_id", "wrong"),
        ("manifest_sha256", "0" * 64),
        ("metadata_sha256", "0" * 64),
        ("environment_steps", 30_088),
        ("source_total_environment_steps", 30_089),
        ("target_total_environment_steps", 99_999),
    ):
        with pytest.raises(launcher.LaunchError, match=key):
            launcher._validate_checkpoint_summary(
                {**summary, key: wrong},
                paths=paths,
            )


def test_v33_recovery_supervisor_reenters_adapter_with_recovery_pins(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    manifest = paths.manifest_dir / "recovery.launch.json"

    assert launcher._core.RUN_NAME == launcher.RUN_NAME
    assert launcher._core.RESUME_RUN_ID == launcher.SOURCE_RUN_ID
    assert launcher._core.RESUME_STEP == launcher.SOURCE_ENVIRONMENT_STEPS
    assert launcher._core.RESUME_CHECKPOINT_ID == launcher.SOURCE_CHECKPOINT_ID
    assert launcher.build_supervisor_command(paths, manifest_path=manifest) == (
        str(paths.venv_python),
        str(SCRIPT.resolve()),
        "supervise",
        "--manifest",
        str(manifest),
    )
