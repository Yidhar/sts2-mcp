from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

from tests.archived_experiment_config import load_archived_training_config

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PACKAGE_ROOT / "scripts/launch_v34_evaluation_recovery_exact_resume.py"
CONFIG = (
    PACKAGE_ROOT
    / "config/experiments/full_run_revival_v34_transaction_competence_model_init.toml"
)
SPEC = importlib.util.spec_from_file_location(
    "v34_evaluation_recovery_exact_resume_launcher",
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
        "source_total_environment_steps": launcher.SOURCE_TOTAL_ENVIRONMENT_STEPS,
        "target_total_environment_steps": launcher.TARGET_ENVIRONMENT_STEPS,
        "checkpoint_format": "sts2-recurrent-vtrace-checkpoint-v5",
    }


def test_v34_recovery_restores_the_same_recipe_to_original_horizon(
    tmp_path: Path,
) -> None:
    config = load_archived_training_config(profile="preheat", config_path=CONFIG)
    paths = _paths(tmp_path)
    command = launcher.build_resume_trainer_command(paths)

    assert config.runtime.total_environment_steps == 100_000
    assert config.runtime.evaluation_steps == (0, 25_000, 50_000, 75_000)
    assert config.runtime.final_audit_steps == (100_000,)
    assert command.count("--resume") == 1
    assert "--initialize-from" not in command
    assert command[command.index("--resume") + 1] == str(
        launcher.source_checkpoint_path(paths)
    )
    assert launcher.SOURCE_ENVIRONMENT_STEPS == 40_468
    assert launcher.ADDITIONAL_ENVIRONMENT_STEPS == 59_532


def test_v34_recovery_checkpoint_identity_is_fully_pinned(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    summary = _summary(paths)

    assert launcher._validate_checkpoint_summary(summary, paths=paths) == summary
    for key, wrong in (
        ("checkpoint_id", "wrong"),
        ("manifest_sha256", "0" * 64),
        ("metadata_sha256", "0" * 64),
        ("experiment_run_id", "wrong"),
        ("environment_steps", 40_467),
        ("policy_version", 653),
        ("learner_updates", 653),
    ):
        with pytest.raises(launcher.LaunchError, match=key):
            launcher._validate_checkpoint_summary(
                {**summary, key: wrong},
                paths=paths,
            )


def test_v34_recovery_supervisor_state_is_isolated_and_reenters_adapter(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    manifest = paths.manifest_dir / "recovery.launch.json"

    assert launcher._core.RUN_NAME == launcher.RUN_NAME
    assert launcher._core.STATE_NAME == launcher.STATE_NAME
    assert launcher.STATE_NAME != launcher.RUN_NAME
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
    status = launcher._core.read_status(paths, enforce_active_root=False)
    assert status["state_path"] == str(
        paths.launcher_dir / f"{launcher.STATE_NAME}.state.json"
    )
