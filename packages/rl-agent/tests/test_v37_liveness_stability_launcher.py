from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

from sts2_baseline import REVIVAL_EFFICIENCY_REWARD_SPEC
from sts2_rl.training import CONFIG_VERSION, load_training_config

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PACKAGE_ROOT / "scripts" / "launch_v37_liveness_stability_model_init.py"
CONFIG = PACKAGE_ROOT / "config" / "experiments" / "full_run_revival_v37_liveness_stability_model_init.toml"
V36_CONFIG = PACKAGE_ROOT / "config" / "experiments" / "full_run_revival_v36_transaction_recovery_model_init.toml"
SPEC = importlib.util.spec_from_file_location(
    "v37_liveness_stability_launcher",
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


def test_v37_keeps_v36_transaction_corridor_and_repairs_liveness_plane() -> None:
    config = load_training_config(profile="preheat", config_path=CONFIG)
    v36 = load_training_config(profile="preheat", config_path=V36_CONFIG)

    assert config.version == CONFIG_VERSION == "sts2-relational-curriculum-config-v19"
    transaction = config.transaction_learning
    assert transaction == v36.transaction_learning
    assert transaction.enabled
    assert transaction.effect_weight == pytest.approx(0.05)
    assert transaction.transaction_q_weight == pytest.approx(0.0)
    assert transaction.completion_policy_weight == pytest.approx(0.15)
    assert transaction.pairwise_ranking_weight == pytest.approx(0.0)
    assert transaction.lifecycle_entry_support_weight == pytest.approx(0.05)
    assert transaction.lifecycle_entry_support_probability_floor == pytest.approx(0.05)
    assert transaction.lifecycle_smdp_q_weight == pytest.approx(0.10)
    assert transaction.replay_byte_capacity == 1_073_741_824
    explorer = config.transaction_exploration
    assert explorer == v36.transaction_exploration
    assert explorer.enabled
    assert explorer.operations == ("remove", "upgrade")
    assert explorer.entry_epsilon_floor == pytest.approx(0.50)
    assert explorer.completion_guidance_probability == pytest.approx(0.95)
    assert REVIVAL_EFFICIENCY_REWARD_SPEC.version == "sts2-run-survival-efficiency-v7"
    assert REVIVAL_EFFICIENCY_REWARD_SPEC.revival_reference_budget == 64
    assert REVIVAL_EFFICIENCY_REWARD_SPEC.revival_cost_cap == pytest.approx(0.40)
    assert config.optimization.entropy_breaker == "policy-collapse-v2"
    assert config.failure_credit.liveness_risk_actor_min_selected_probability == pytest.approx(0.01)
    assert config.episodic_learning.success_imitation_exempt_surfaces == (
        "rest_site",
        "shop",
    )
    assert config.curriculum.revival_budget == 16
    assert config.runtime.model_initialization_schedule_mode == "inherit"
    assert config.runtime.model_initialization_liveness_schedule_mode == "inherit"
    assert config.runtime.total_environment_steps == 100_000
    assert config.runtime.evaluation_steps == (0, 20_000, 40_000, 60_000, 80_000)
    assert config.runtime.early_evaluation_steps == (
        5_000,
        10_000,
        30_000,
        50_000,
        70_000,
        90_000,
    )
    assert config.runtime.evaluation_guard_enforcement_start_steps == 10_000
    assert config.runtime.evaluation_guard_failure_action == "stop"
    assert config.runtime.evaluation_guard_max_rollbacks == 0


def test_v37_is_pinned_model_initialization_from_v36_healthy_anchor(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    command = launcher.build_resume_trainer_command(paths)

    assert command.count("--initialize-from") == 1
    assert command[command.index("--initialize-from") + 1] == str(launcher.source_checkpoint_path(paths))
    assert "--resume" not in command
    assert launcher.SOURCE_RUN_ID == "d62931b7-8dbb-4df8-bf53-09e3c290acc0"
    assert launcher.SOURCE_ENVIRONMENT_STEPS == 25_474
    assert launcher.SOURCE_POLICY_VERSION == 416
    assert launcher.SOURCE_LEARNER_UPDATES == 416
    assert launcher.SOURCE_CHECKPOINT_ID == "0824662c-7801-43c7-8d97-6ca1c49af25d"
    assert launcher.SOURCE_MANIFEST_SHA256 == ("b2c5d3d93ea43425074c552308a0173653e89ccbe552e82bd7b140b6d4586567")
    assert launcher.SOURCE_METADATA_SHA256 == ("3c26655815a20628371c17a43c0c747dacad03a7e5344d51dcbcfc982b0f13f8")
    assert launcher.source_checkpoint_path(paths).name == ("healthy-validation-step-000025474")
    assert launcher._validate_checkpoint_summary(
        _checkpoint_summary(paths),
        paths=paths,
    ) == _checkpoint_summary(paths)
    with pytest.raises(launcher.LaunchError, match="command was modified"):
        launcher.validate_fixed_model_initialization_command(
            (*command, "--resume", str(tmp_path / "forbidden")),
            paths=paths,
        )


def test_v37_checkpoint_pin_rejects_every_identity_mismatch(tmp_path: Path) -> None:
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


def test_v37_supervisor_reenters_adapter_and_keeps_pins(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    manifest = tmp_path / "v37.launch.json"
    command = launcher.build_supervisor_command(paths, manifest_path=manifest)

    assert Path(command[1]).name == SCRIPT.name
    assert command[-2:] == ("--manifest", str(manifest))
    assert launcher._core.RUN_NAME == launcher.RUN_NAME
    assert launcher._core.STATE_NAME == launcher.RUN_NAME
    assert launcher._core._state_path(paths) == (paths.launcher_dir / f"{launcher.RUN_NAME}.state.json")
    assert launcher._core.RESUME_RUN_ID == launcher.SOURCE_RUN_ID
    assert launcher._core.RESUME_STEP == 25_474
