from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

from sts2_rl.training import load_training_config

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PACKAGE_ROOT / "scripts/launch_v31_failure_credit_capacity_model_init.py"
CONFIG = (
    PACKAGE_ROOT
    / "config/experiments/full_run_revival_v31_failure_credit_capacity_model_init.toml"
)
SPEC = importlib.util.spec_from_file_location(
    "v31_failure_credit_capacity_launcher",
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
        "source_total_environment_steps": launcher.SOURCE_TOTAL_ENVIRONMENT_STEPS,
        "target_total_environment_steps": launcher.TARGET_ENVIRONMENT_STEPS,
        "checkpoint_format": "sts2-recurrent-vtrace-checkpoint-v5",
    }


def test_v31_recipe_expands_failure_replay_without_truncating_context() -> None:
    config = load_training_config(profile="preheat", config_path=CONFIG)

    assert config.failure_credit.replay_byte_capacity == 2_147_483_648
    assert config.failure_credit.maximum_context_steps == 256
    assert config.failure_credit.direct_witness_quota == 1
    assert config.failure_credit.multi_edge_cycle_quota == 1
    assert config.episodic_learning.replay_capacity_bytes == 2_147_483_648
    assert config.runtime.total_environment_steps == 250_000
    assert config.runtime.early_evaluation_episodes == 16
    assert config.runtime.evaluation_guard_liveness_baseline_failures == 9
    assert config.runtime.evaluation_guard_liveness_baseline_episodes == 16
    assert config.runtime.evaluation_guard_min_liveness_regression_rate == 0.20


def test_v31_is_model_initialization_from_frozen_v30_not_exact_resume(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    command = launcher.build_resume_trainer_command(paths)

    assert command.count("--initialize-from") == 1
    assert command[command.index("--initialize-from") + 1] == str(
        launcher.source_checkpoint_path(paths)
    )
    assert "--resume" not in command
    assert launcher.SOURCE_ENVIRONMENT_STEPS == 30_200
    assert launcher.SOURCE_POLICY_VERSION == 491
    with pytest.raises(launcher.LaunchError, match="command was modified"):
        launcher.validate_fixed_model_initialization_command(
            (*command, "--resume", str(tmp_path / "forbidden")),
            paths=paths,
        )


def test_v31_checkpoint_source_is_fully_pinned(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    summary = _checkpoint_summary(paths)

    assert launcher._validate_checkpoint_summary(summary, paths=paths) == summary
    for key in (
        "checkpoint_id",
        "manifest_sha256",
        "metadata_sha256",
        "experiment_run_id",
        "environment_steps",
        "policy_version",
        "learner_updates",
    ):
        with pytest.raises(launcher.LaunchError, match=key):
            launcher._validate_checkpoint_summary(
                {**summary, key: "wrong"},
                paths=paths,
            )


def test_v31_supervisor_reenters_capacity_adapter_and_has_isolated_core(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    manifest = tmp_path / "v31.launch.json"
    command = launcher.build_supervisor_command(paths, manifest_path=manifest)

    assert Path(command[1]).name == SCRIPT.name
    assert command[-2:] == ("--manifest", str(manifest))
    assert launcher._core.RUN_NAME == launcher.RUN_NAME
    assert launcher._core.RESUME_RUN_ID == launcher.SOURCE_RUN_ID
