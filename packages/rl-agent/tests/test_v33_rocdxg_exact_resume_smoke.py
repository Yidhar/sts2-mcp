from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

from tests.archived_experiment_config import load_archived_training_config

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PACKAGE_ROOT / "scripts/launch_v33_rocdxg_exact_resume_smoke.py"
SMOKE_CONFIG = PACKAGE_ROOT / "config/experiments/full_run_revival_v33_rocdxg_exact_resume_smoke.toml"
SOURCE_CONFIG = PACKAGE_ROOT / "config/experiments/full_run_revival_v33_stability_recovery_model_init.toml"
SPEC = importlib.util.spec_from_file_location("v33_rocdxg_exact_resume_smoke", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
launcher = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = launcher
SPEC.loader.exec_module(launcher)


def _paths(tmp_path: Path) -> Any:
    checkout = tmp_path / "source"
    package = checkout / "packages/rl-agent"
    artifact = tmp_path / "artifacts/runtime"
    simulator = artifact / "dependencies/HeadlessSim.exe"
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


def test_smoke_changes_only_mutable_runtime_configuration() -> None:
    source = load_archived_training_config(profile="preheat", config_path=SOURCE_CONFIG)
    smoke = load_archived_training_config(profile="preheat", config_path=SMOKE_CONFIG)

    assert smoke.lineage_mapping() == source.lineage_mapping()
    assert smoke.runtime.total_environment_steps == launcher.TARGET_ENVIRONMENT_STEPS
    assert launcher.ADDITIONAL_ENVIRONMENT_STEPS == 512
    assert smoke.runtime.log_dir.endswith("v33-rocdxg-exact-resume-smoke")
    assert smoke.runtime.checkpoint_dir.endswith("v33-rocdxg-exact-resume-smoke")
    assert smoke.runtime.checkpoint_interval_steps == 256
    assert smoke.runtime.evaluation_steps == ()
    assert smoke.runtime.early_evaluation_steps == ()
    assert smoke.runtime.final_audit_steps == ()
    assert smoke.runtime.evaluation_liveness_guard_enabled is False


def test_smoke_command_is_exact_resume_and_environment_is_rocdxg_hermetic(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    command = launcher.build_resume_trainer_command(paths)
    environment = launcher.build_environment(paths)

    assert command.count("--resume") == 1
    assert command[command.index("--resume") + 1] == str(
        launcher.source_checkpoint_path(paths)
    )
    assert "--initialize-from" not in command
    assert environment["HSA_ENABLE_DXG_DETECTION"] == "1"
    assert len(environment) == 11
    assert "PYTHONHOME" not in environment
    assert "LD_LIBRARY_PATH" not in environment


def test_smoke_checkpoint_identity_and_counters_are_fully_pinned(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    summary = {
        "root": str(launcher.source_checkpoint_path(paths)),
        "checkpoint_id": launcher.SOURCE_CHECKPOINT_ID,
        "manifest_sha256": launcher.SOURCE_MANIFEST_SHA256,
        "metadata_sha256": launcher.SOURCE_METADATA_SHA256,
        "manifest_files": [{"path": "model.pt", "sha256": "1" * 64}],
        "experiment_run_id": launcher.SOURCE_RUN_ID,
        "environment_steps": launcher.SOURCE_ENVIRONMENT_STEPS,
        "policy_version": launcher.SOURCE_POLICY_VERSION,
        "learner_updates": launcher.SOURCE_LEARNER_UPDATES,
        "source_total_environment_steps": launcher.SOURCE_TOTAL_ENVIRONMENT_STEPS,
        "target_total_environment_steps": launcher.TARGET_ENVIRONMENT_STEPS,
        "checkpoint_format": "sts2-recurrent-vtrace-checkpoint-v5",
    }

    assert launcher._validate_checkpoint_summary(summary, paths=paths) == summary
