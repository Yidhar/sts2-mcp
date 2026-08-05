from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

from sts2_rl.training import CONFIG_VERSION, load_training_config

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PACKAGE_ROOT / "scripts" / "launch_v34_transaction_competence_model_init.py"
CONFIG = PACKAGE_ROOT / "config" / "experiments" / "full_run_revival_v34_transaction_competence_model_init.toml"
SPEC = importlib.util.spec_from_file_location("v34_transaction_competence_launcher", SCRIPT)
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


def test_v34_recipe_enables_only_reviewed_transaction_exploration() -> None:
    config = load_training_config(profile="preheat", config_path=CONFIG)

    assert config.version == CONFIG_VERSION == "sts2-relational-curriculum-config-v14"
    explorer = config.transaction_exploration
    assert explorer.enabled
    assert explorer.operations == ("remove", "upgrade")
    assert explorer.entry_epsilon_floor == pytest.approx(0.50)
    assert explorer.completion_guidance_probability == pytest.approx(0.95)
    assert config.curriculum.revival_budget == 64
    assert config.curriculum.selection_surface_epsilon_floor == pytest.approx(0.25)
    assert config.runtime.model_initialization_schedule_mode == "inherit"
    assert config.runtime.model_initialization_liveness_schedule_mode == "inherit"
    assert config.runtime.total_environment_steps == 100_000
    assert config.runtime.evaluation_steps == (0, 25_000, 50_000, 75_000)
    assert config.runtime.early_evaluation_steps == (5_000, 10_000)
    assert config.runtime.final_audit_steps == (100_000,)


def test_v34_is_a_pinned_model_initialization_not_exact_resume(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    command = launcher.build_resume_trainer_command(paths)

    assert command.count("--initialize-from") == 1
    assert command[command.index("--initialize-from") + 1] == str(launcher.source_checkpoint_path(paths))
    assert "--resume" not in command
    assert launcher.SOURCE_RUN_ID == "6c7e939a-9b11-4624-8847-77a8d71631ab"
    assert launcher.SOURCE_ENVIRONMENT_STEPS == 75_714
    assert launcher.SOURCE_POLICY_VERSION == 1_227
    assert launcher.SOURCE_LEARNER_UPDATES == 1_227
    assert launcher.SOURCE_CHECKPOINT_ID == "1c412b44-c288-4278-b2e0-373bf9d20a5e"
    assert launcher._validate_checkpoint_summary(
        _checkpoint_summary(paths),
        paths=paths,
    ) == _checkpoint_summary(paths)
    with pytest.raises(launcher.LaunchError, match="command was modified"):
        launcher.validate_fixed_model_initialization_command(
            (*command, "--resume", str(tmp_path / "forbidden")),
            paths=paths,
        )


def test_v34_checkpoint_pin_rejects_every_identity_mismatch(tmp_path: Path) -> None:
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


def test_v34_supervisor_reenters_adapter_and_keeps_pins(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    manifest = tmp_path / "v34.launch.json"
    command = launcher.build_supervisor_command(paths, manifest_path=manifest)

    assert Path(command[1]).name == SCRIPT.name
    assert command[-2:] == ("--manifest", str(manifest))
    assert launcher._core.RUN_NAME == launcher.RUN_NAME
    assert launcher._core.RESUME_RUN_ID == launcher.SOURCE_RUN_ID
    assert launcher._core.RESUME_STEP == 75_714
