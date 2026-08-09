from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

from sts2_rl.training import CONFIG_VERSION, load_training_config

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PACKAGE_ROOT / "scripts" / "launch_v45_deck_macro_bootstrap_model_init.py"
CONFIG = (
    PACKAGE_ROOT
    / "config"
    / "experiments"
    / "full_run_revival_v45_deck_macro_bootstrap_model_init.toml"
)
V44_CONFIG = (
    PACKAGE_ROOT
    / "config"
    / "experiments"
    / "full_run_revival_v44_budget40_identification_model_init.toml"
)
SPEC = importlib.util.spec_from_file_location(
    "v45_deck_macro_bootstrap_launcher",
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


def test_v45_changes_only_exploration_roster_from_v44() -> None:
    config = load_training_config(profile="preheat", config_path=CONFIG)
    v44 = load_training_config(profile="preheat", config_path=V44_CONFIG)

    assert config.version == CONFIG_VERSION == "sts2-relational-curriculum-config-v19"
    assert config.optimization == v44.optimization
    assert config.rollout == v44.rollout
    assert v44.transaction_learning.transaction_q_weight == 0.0
    assert config.transaction_learning.transaction_q_weight == 0.05
    from dataclasses import replace as _replace
    assert _replace(config.transaction_learning, transaction_q_weight=0.0) == (
        v44.transaction_learning
    )
    assert v44.transaction_exploration.enabled is False
    assert config.transaction_exploration.enabled is True
    assert config.transaction_exploration.operations == ("relic_purchase", "reward_skip")
    assert config.transaction_exploration.entry_epsilon_floor == 0.15
    assert config.transaction_exploration.completion_guidance_probability == 0.0
    assert config.failure_credit == v44.failure_credit
    assert config.episodic_learning == v44.episodic_learning
    assert config.model == v44.model
    assert config.environment == v44.environment

    assert v44.curriculum.revival_budget == 40
    assert config.curriculum.revival_budget == 40
    assert config.curriculum.epsilon_start == v44.curriculum.epsilon_start
    assert config.curriculum.epsilon_end == v44.curriculum.epsilon_end
    assert config.curriculum.epsilon_decay_steps == v44.curriculum.epsilon_decay_steps
    assert config.curriculum.selection_surface_epsilon_floor == pytest.approx(0.0)

    assert config.runtime.model_initialization_schedule_mode == "inherit"
    assert config.runtime.model_initialization_liveness_schedule_mode == "reset"
    assert config.runtime.total_environment_steps == 250_000
    assert config.runtime.seed == v44.runtime.seed == 6_300_000
    assert config.runtime.evaluation_guard_failure_action == "stop"
    assert config.runtime.evaluation_guard_max_rollbacks == 0
    assert config.runtime.evaluation_guard_enforcement_start_steps == 10_000


def test_v45_is_pinned_model_initialization_from_v44_healthy_gate(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    command = launcher.build_resume_trainer_command(paths)

    assert command.count("--initialize-from") == 1
    assert command[command.index("--initialize-from") + 1] == str(
        launcher.source_checkpoint_path(paths)
    )
    assert "--resume" not in command
    assert launcher.SOURCE_RUN_ID == "ade6ab3a-2eda-4b5b-8fe4-3710858f9a65"
    assert launcher.SOURCE_ENVIRONMENT_STEPS == 100_478
    assert launcher.SOURCE_TOTAL_ENVIRONMENT_STEPS == 250_000
    assert launcher.TARGET_ENVIRONMENT_STEPS == 250_000
    assert launcher.SOURCE_POLICY_VERSION == 1_646
    assert launcher.SOURCE_LEARNER_UPDATES == 1_646
    assert launcher.SOURCE_CHECKPOINT_ID == (
        "d44625a7-2e50-4449-9f5c-4f7b62f3c5f3"
    )
    assert launcher.SOURCE_MANIFEST_SHA256 == (
        "77dc76e3009c0fb084c1dafd8f23acfbca3e4473d8c5d14126de62fc35e631b8"
    )
    assert launcher.SOURCE_METADATA_SHA256 == (
        "f87f4368749a36db97733fc3f208a822375a79271aaa37510ff9d0cede8049cb"
    )
    assert launcher.source_checkpoint_path(paths).name == (
        "healthy-validation-step-000100478"
    )
    assert launcher._validate_checkpoint_summary(
        _checkpoint_summary(paths),
        paths=paths,
    ) == _checkpoint_summary(paths)
    with pytest.raises(launcher.LaunchError, match="command was modified"):
        launcher.validate_fixed_model_initialization_command(
            (*command, "--resume", str(tmp_path / "forbidden")),
            paths=paths,
        )


def test_v45_checkpoint_pin_rejects_every_identity_mismatch(
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


def test_v45_supervisor_reenters_adapter_and_keeps_pins(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    manifest = tmp_path / "v45.launch.json"
    command = launcher.build_supervisor_command(paths, manifest_path=manifest)

    assert Path(command[1]).name == SCRIPT.name
    assert command[-2:] == ("--manifest", str(manifest))
    assert launcher._core.RUN_NAME == launcher.RUN_NAME
    assert launcher._core.STATE_NAME == launcher.RUN_NAME
    assert launcher._core._state_path(paths) == (
        paths.launcher_dir / f"{launcher.RUN_NAME}.state.json"
    )
    assert launcher._core.RESUME_RUN_ID == launcher.SOURCE_RUN_ID
    assert launcher._core.RESUME_STEP == 100_478
