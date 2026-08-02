from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PACKAGE_ROOT / "scripts/launch_v31_guard_recovery.py"
SPEC = importlib.util.spec_from_file_location(
    "v31_guard_recovery_launcher",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
launcher = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = launcher
SPEC.loader.exec_module(launcher)
launcher._configure()


def _paths(tmp_path: Path) -> Any:
    checkout = tmp_path / "source"
    package = checkout / "packages" / "rl-agent"
    artifact = tmp_path / "artifacts" / "runtime"
    simulator = artifact / "dependencies" / "HeadlessSim.exe"
    return launcher._base.LaunchPaths(
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


def _summary(paths: Any) -> dict[str, Any]:
    return {
        "root": str(launcher._base.source_checkpoint_path(paths)),
        "checkpoint_id": launcher.SOURCE_CHECKPOINT_ID,
        "manifest_sha256": launcher.SOURCE_MANIFEST_SHA256,
        "metadata_sha256": launcher.SOURCE_METADATA_SHA256,
        "manifest_files": [{"path": "network.pt", "sha256": "1" * 64}],
        "experiment_run_id": launcher.SOURCE_RUN_ID,
        "environment_steps": launcher.SOURCE_ENVIRONMENT_STEPS,
        "policy_version": launcher.SOURCE_POLICY_VERSION,
        "learner_updates": launcher.SOURCE_LEARNER_UPDATES,
        "source_total_environment_steps": launcher.TARGET_ENVIRONMENT_STEPS,
        "target_total_environment_steps": launcher.TARGET_ENVIRONMENT_STEPS,
        "checkpoint_format": "sts2-recurrent-vtrace-checkpoint-v5",
    }


def test_v31_guard_recovery_is_exact_resume_only(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    command = launcher._base.build_resume_trainer_command(paths)

    assert command.count("--resume") == 1
    assert command[command.index("--resume") + 1] == str(launcher._base.source_checkpoint_path(paths))
    assert "--initialize-from" not in command
    assert launcher.SOURCE_ENVIRONMENT_STEPS == 5_564
    assert launcher.TARGET_ENVIRONMENT_STEPS == 250_000


def test_v31_guard_recovery_checkpoint_is_fully_pinned(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    summary = _summary(paths)

    assert (
        launcher._validate_checkpoint_summary(
            summary,
            paths=paths,
        )
        == summary
    )
    with pytest.raises(launcher._base.LaunchError, match="checkpoint_id"):
        launcher._validate_checkpoint_summary(
            {**summary, "checkpoint_id": "wrong"},
            paths=paths,
        )


def test_v31_guard_recovery_supervisor_reenters_adapter(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    manifest = tmp_path / "resume.launch.json"
    command = launcher._build_supervisor_command(
        paths,
        manifest_path=manifest,
    )

    assert Path(command[1]).name == SCRIPT.name
    assert command[-2:] == ("--manifest", str(manifest))
    assert launcher._base._core.RUN_NAME == launcher.RECOVERY_RUN_NAME
    assert launcher._base._core.RESUME_STEP == 5_564
