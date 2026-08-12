from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

from sts2_rl.training import CONFIG_VERSION
from tests.archived_experiment_config import load_archived_training_config

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PACKAGE_ROOT / "scripts" / "launch_v35_transaction_lifecycle_model_init.py"
CONFIG = (
    PACKAGE_ROOT
    / "config"
    / "experiments"
    / "full_run_revival_v35_transaction_lifecycle_model_init.toml"
)
SPEC = importlib.util.spec_from_file_location(
    "v35_transaction_lifecycle_launcher",
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


def test_v35_recipe_enables_only_bounded_lifecycle_entry_credit() -> None:
    config = load_archived_training_config(profile="preheat", config_path=CONFIG)

    assert config.version == CONFIG_VERSION == "sts2-relational-curriculum-config-v20"
    transaction = config.transaction_learning
    assert transaction.enabled
    assert transaction.effect_weight == pytest.approx(0.05)
    assert transaction.transaction_q_weight == pytest.approx(0.0)
    assert transaction.completion_policy_weight == pytest.approx(0.0)
    assert transaction.pairwise_ranking_weight == pytest.approx(0.0)
    assert transaction.lifecycle_entry_support_weight == pytest.approx(0.25)
    assert transaction.lifecycle_entry_support_probability_floor == pytest.approx(0.05)
    assert transaction.lifecycle_smdp_q_weight == pytest.approx(0.10)
    assert transaction.replay_byte_capacity == 1_073_741_824
    assert config.curriculum.revival_budget == 64
    assert config.runtime.model_initialization_schedule_mode == "inherit"
    assert config.runtime.model_initialization_liveness_schedule_mode == "inherit"
    assert config.runtime.total_environment_steps == 100_000


def test_v35_is_pinned_model_initialization_from_frozen_healthy_v34(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    command = launcher.build_resume_trainer_command(paths)

    assert command.count("--initialize-from") == 1
    assert command[command.index("--initialize-from") + 1] == str(
        launcher.source_checkpoint_path(paths)
    )
    assert "--resume" not in command
    assert launcher.SOURCE_RUN_ID == "795ad740-bcf6-4707-86a0-8a3b3301a13c"
    assert launcher.SOURCE_ENVIRONMENT_STEPS == 50_320
    assert launcher.SOURCE_POLICY_VERSION == 814
    assert launcher.SOURCE_LEARNER_UPDATES == 814
    assert launcher.SOURCE_CHECKPOINT_ID == "2a3cccf8-09f5-4808-9991-86af6e6f3cfc"
    assert launcher._validate_checkpoint_summary(
        _checkpoint_summary(paths),
        paths=paths,
    ) == _checkpoint_summary(paths)
    with pytest.raises(launcher.LaunchError, match="command was modified"):
        launcher.validate_fixed_model_initialization_command(
            (*command, "--resume", str(tmp_path / "forbidden")),
            paths=paths,
        )


def test_v35_checkpoint_pin_rejects_every_identity_mismatch(tmp_path: Path) -> None:
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


def test_v35_supervisor_reenters_adapter_and_keeps_pins(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    manifest = tmp_path / "v35.launch.json"
    command = launcher.build_supervisor_command(paths, manifest_path=manifest)

    assert Path(command[1]).name == SCRIPT.name
    assert command[-2:] == ("--manifest", str(manifest))
    assert launcher._core.RUN_NAME == launcher.RUN_NAME
    assert launcher._core.STATE_NAME == launcher.RUN_NAME
    assert launcher._core._state_path(paths) == (
        paths.launcher_dir / f"{launcher.RUN_NAME}.state.json"
    )
    assert launcher._core.RESUME_RUN_ID == launcher.SOURCE_RUN_ID
    assert launcher._core.RESUME_STEP == 50_320
