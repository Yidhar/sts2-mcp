from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

from sts2_rl.training import CONFIG_VERSION, load_training_config

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PACKAGE_ROOT / "scripts" / "launch_v33_stability_recovery_model_init.py"
CONFIG = PACKAGE_ROOT / "config" / "experiments" / "full_run_revival_v33_stability_recovery_model_init.toml"
SPEC = importlib.util.spec_from_file_location("v33_stability_recovery_launcher", SCRIPT)
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


def test_v33_recipe_matches_the_reviewed_stability_package() -> None:
    config = load_training_config(profile="preheat", config_path=CONFIG)

    assert config.version == CONFIG_VERSION == "sts2-relational-curriculum-config-v13"
    assert config.curriculum.revival_budget == 64
    assert config.curriculum.selection_surface_epsilon_floor == pytest.approx(0.25)
    assert config.optimization.entropy_weight_end == pytest.approx(0.004)
    assert config.optimization.entropy_breaker == "one-hot-v1"
    assert config.failure_credit.liveness_completion_policy_weight == pytest.approx(0.15)
    assert config.failure_credit.liveness_gradient_clip_norm == pytest.approx(0.50)
    assert config.failure_credit.sample_records == 4
    assert config.failure_credit.liveness_records_per_autograd_batch == 4
    assert config.episodic_learning.success_policy_trust_region_epsilon == pytest.approx(0.20)
    assert config.runtime.model_initialization_schedule_mode == "inherit"
    assert config.runtime.model_initialization_liveness_schedule_mode == "reset"
    assert config.runtime.total_environment_steps == 100_000
    assert config.runtime.evaluation_steps == (0, 25_000, 50_000, 75_000)
    assert config.runtime.evaluation_episodes == 16
    assert config.runtime.early_evaluation_steps == (5_000, 10_000)
    assert config.runtime.early_evaluation_episodes == 16
    assert config.runtime.final_audit_steps == (100_000,)
    assert config.runtime.final_audit_episodes == 64
    assert config.runtime.training_deadlock_streak_alert_episodes == 4
    assert config.runtime.evaluation_guard_failure_action == "rollback_continue"
    assert config.runtime.evaluation_guard_max_rollbacks == 2


def test_v33_is_a_fully_pinned_model_initialization_not_exact_resume(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    command = launcher.build_resume_trainer_command(paths)

    assert command.count("--initialize-from") == 1
    assert command[command.index("--initialize-from") + 1] == str(launcher.source_checkpoint_path(paths))
    assert "--resume" not in command
    assert launcher.SOURCE_RUN_ID == "481995a5-0221-4f59-9293-9105cd336068"
    assert launcher.SOURCE_ENVIRONMENT_STEPS == 90_152
    assert launcher.SOURCE_POLICY_VERSION == 1_467
    assert launcher.SOURCE_LEARNER_UPDATES == 1_467
    assert launcher.SOURCE_CHECKPOINT_ID == "1c0d5284-6be0-4367-a669-166f37b9b3ec"
    assert launcher._validate_checkpoint_summary(
        _checkpoint_summary(paths),
        paths=paths,
    ) == _checkpoint_summary(paths)
    with pytest.raises(launcher.LaunchError, match="command was modified"):
        launcher.validate_fixed_model_initialization_command(
            (*command, "--resume", str(tmp_path / "forbidden")),
            paths=paths,
        )


def test_v33_checkpoint_pin_rejects_every_identity_mismatch(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    summary = _checkpoint_summary(paths)
    for key in (
        "checkpoint_id",
        "manifest_sha256",
        "metadata_sha256",
        "experiment_run_id",
        "environment_steps",
        "policy_version",
        "learner_updates",
        "source_total_environment_steps",
        "target_total_environment_steps",
        "checkpoint_format",
    ):
        with pytest.raises(launcher.LaunchError, match=key):
            launcher._validate_checkpoint_summary(
                {**summary, key: "wrong"},
                paths=paths,
            )


def test_v33_supervisor_reenters_the_v33_adapter_and_keeps_its_pins(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    manifest = tmp_path / "v33.launch.json"
    command = launcher.build_supervisor_command(paths, manifest_path=manifest)

    assert Path(command[1]).name == SCRIPT.name
    assert command[-2:] == ("--manifest", str(manifest))
    assert launcher._core.RUN_NAME == launcher.RUN_NAME
    assert launcher._core.RESUME_RUN_ID == launcher.SOURCE_RUN_ID
    assert launcher._core.RESUME_STEP == 90_152
