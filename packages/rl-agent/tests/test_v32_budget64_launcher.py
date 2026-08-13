from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

from tests.archived_experiment_config import load_archived_training_config

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PACKAGE_ROOT / "scripts/launch_v32_budget64_mature_model_init.py"
CONFIG = (
    PACKAGE_ROOT
    / "config/experiments/full_run_revival_v32_budget64_mature_model_init.toml"
)
SPEC = importlib.util.spec_from_file_location("v32_budget64_launcher", SCRIPT)
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


def test_v32_recipe_is_finite_budget_and_inherits_mature_schedule() -> None:
    config = load_archived_training_config(profile="preheat", config_path=CONFIG)

    assert config.curriculum.revival_budget == 64
    assert config.runtime.model_initialization_schedule_mode == "inherit"
    assert config.runtime.total_environment_steps == 250_000
    assert config.runtime.seed == 5_000_000
    assert config.failure_credit.replay_byte_capacity == 2_147_483_648
    assert config.failure_credit.maximum_context_steps == 256
    assert config.failure_credit.sample_records == 4
    assert config.failure_credit.direct_witness_quota == 0
    assert config.failure_credit.multi_edge_cycle_quota == 0
    assert config.failure_credit.risk_sequence_quota == 1
    assert config.failure_credit.unresolved_stall_quota == 1
    assert config.failure_credit.completion_control_quota == 1
    assert config.runtime.evaluation_guard_min_liveness_episodes == 16
    assert config.runtime.evaluation_guard_liveness_baseline_failures == 2
    assert config.runtime.evaluation_guard_liveness_baseline_episodes == 16


def test_v32_is_explicit_model_initialization_not_exact_resume(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    command = launcher.build_resume_trainer_command(paths)

    assert command.count("--initialize-from") == 1
    assert command[command.index("--initialize-from") + 1] == str(
        launcher.source_checkpoint_path(paths)
    )
    assert "--resume" not in command
    assert launcher.SOURCE_RUN_ID == "21c51131-596c-4ee4-9a4b-4b2743529baa"
    assert launcher.SOURCE_ENVIRONMENT_STEPS == 80_268
    assert launcher.SOURCE_POLICY_VERSION == 1303
    assert launcher.SOURCE_LEARNER_UPDATES == 1303
    with pytest.raises(launcher.LaunchError, match="command was modified"):
        launcher.validate_fixed_model_initialization_command(
            (*command, "--resume", str(tmp_path / "forbidden")),
            paths=paths,
        )


def test_v32_supervisor_reenters_budget64_adapter_and_has_isolated_core(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    manifest = tmp_path / "v32.launch.json"
    command = launcher.build_supervisor_command(paths, manifest_path=manifest)

    assert Path(command[1]).name == SCRIPT.name
    assert command[-2:] == ("--manifest", str(manifest))
    assert launcher._core.RUN_NAME == launcher.RUN_NAME
    assert launcher._core.RESUME_RUN_ID == launcher.SOURCE_RUN_ID
