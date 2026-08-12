from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

from sts2_rl.training import CONFIG_VERSION
from tests.archived_experiment_config import load_archived_training_config

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PACKAGE_ROOT / "scripts" / "launch_v38_natural_strategy_model_init.py"
CONFIG = PACKAGE_ROOT / "config" / "experiments" / "full_run_revival_v38_natural_strategy_model_init.toml"
V37_CONFIG = PACKAGE_ROOT / "config" / "experiments" / "full_run_revival_v37_liveness_stability_model_init.toml"
SPEC = importlib.util.spec_from_file_location(
    "v38_natural_strategy_launcher",
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
        "manifest_files": [{"path": "network.pt", "sha256": "1" * 64}],
        "experiment_run_id": launcher.SOURCE_RUN_ID,
        "environment_steps": launcher.SOURCE_ENVIRONMENT_STEPS,
        "policy_version": launcher.SOURCE_POLICY_VERSION,
        "learner_updates": launcher.SOURCE_LEARNER_UPDATES,
        "source_total_environment_steps": launcher.SOURCE_TOTAL_ENVIRONMENT_STEPS,
        "target_total_environment_steps": launcher.TARGET_ENVIRONMENT_STEPS,
        "checkpoint_format": "sts2-recurrent-vtrace-checkpoint-v5",
    }


def test_v38_withdraws_only_behavior_scaffolding() -> None:
    config = load_archived_training_config(profile="preheat", config_path=CONFIG)
    v37 = load_archived_training_config(profile="preheat", config_path=V37_CONFIG)

    assert config.version == CONFIG_VERSION == "sts2-relational-curriculum-config-v20"

    # The factual learning corridor and the liveness stability plane survive
    # the transition unchanged.
    assert config.transaction_learning == v37.transaction_learning
    assert config.failure_credit == v37.failure_credit
    assert config.episodic_learning == v37.episodic_learning
    assert config.optimization == v37.optimization
    assert config.rollout == v37.rollout
    assert config.curriculum.revival_budget == v37.curriculum.revival_budget == 16
    assert config.curriculum.epsilon_start == v37.curriculum.epsilon_start
    assert config.curriculum.epsilon_end == v37.curriculum.epsilon_end
    assert config.curriculum.epsilon_decay_steps == v37.curriculum.epsilon_decay_steps


    # No rollback gate can replace a newly learned strategy with an old one.
    assert config.runtime.model_initialization_schedule_mode == "inherit"
    assert config.runtime.model_initialization_liveness_schedule_mode == "inherit"
    assert config.runtime.total_environment_steps == 100_000
    assert config.runtime.seed == v37.runtime.seed == 6_300_000
    assert config.runtime.evaluation_guard_failure_action == "stop"
    assert config.runtime.evaluation_guard_max_rollbacks == 0


def test_v38_is_pinned_model_initialization_from_v37_healthy_anchor(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    command = launcher.build_resume_trainer_command(paths)

    assert command.count("--initialize-from") == 1
    assert command[command.index("--initialize-from") + 1] == str(launcher.source_checkpoint_path(paths))
    assert "--resume" not in command
    assert launcher.SOURCE_RUN_ID == "b70cc736-267d-45af-8e4d-9219fed2105b"
    assert launcher.SOURCE_ENVIRONMENT_STEPS == 20_337
    assert launcher.SOURCE_POLICY_VERSION == 338
    assert launcher.SOURCE_LEARNER_UPDATES == 338
    assert launcher.SOURCE_CHECKPOINT_ID == ("d0e9481f-8bb5-4e28-859f-ecc401f28685")
    assert launcher.SOURCE_MANIFEST_SHA256 == ("efe8b3c4308993f5dd288cdaa6e51a3b52410f14fb4528857ec5496e3dd8010a")
    assert launcher.SOURCE_METADATA_SHA256 == ("8c5577b7df39e6ca9c1b2452f49986481d909e3bb0ed70e5179185df59c86f0c")
    assert launcher.source_checkpoint_path(paths).name == ("healthy-validation-step-000020337")
    assert launcher._validate_checkpoint_summary(
        _checkpoint_summary(paths),
        paths=paths,
    ) == _checkpoint_summary(paths)
    with pytest.raises(launcher.LaunchError, match="command was modified"):
        launcher.validate_fixed_model_initialization_command(
            (*command, "--resume", str(tmp_path / "forbidden")),
            paths=paths,
        )


def test_v38_checkpoint_pin_rejects_every_identity_mismatch(
    tmp_path: Path,
) -> None:
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


def test_v38_supervisor_reenters_adapter_and_keeps_pins(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    manifest = tmp_path / "v38.launch.json"
    command = launcher.build_supervisor_command(paths, manifest_path=manifest)

    assert Path(command[1]).name == SCRIPT.name
    assert command[-2:] == ("--manifest", str(manifest))
    assert launcher._core.RUN_NAME == launcher.RUN_NAME
    assert launcher._core.STATE_NAME == launcher.RUN_NAME
    assert launcher._core._state_path(paths) == (paths.launcher_dir / f"{launcher.RUN_NAME}.state.json")
    assert launcher._core.RESUME_RUN_ID == launcher.SOURCE_RUN_ID
    assert launcher._core.RESUME_STEP == 20_337
