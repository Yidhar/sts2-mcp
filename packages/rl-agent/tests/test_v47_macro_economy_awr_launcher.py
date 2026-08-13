from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

from sts2_rl.training import CONFIG_VERSION
from tests.archived_experiment_config import load_archived_training_config

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PACKAGE_ROOT / "scripts" / "launch_v47_macro_economy_awr_model_init.py"
CONFIG = PACKAGE_ROOT / "config" / "experiments" / "full_run_revival_v47_macro_economy_awr_model_init.toml"
SPEC = importlib.util.spec_from_file_location("v47_macro_economy_awr_launcher", SCRIPT)
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


def test_v47_recipe_is_factual_isolated_macro_awr() -> None:
    config = load_archived_training_config(profile="preheat", config_path=CONFIG)
    tx = config.transaction_learning
    assert config.version == CONFIG_VERSION == "sts2-relational-curriculum-config-v20"
    assert tx.enabled is True
    assert tx.lifecycle_smdp_horizon == "next_resource_opportunity"
    assert config.curriculum.revival_budget == 40
    assert config.runtime.total_environment_steps == 30_000
    assert config.runtime.evaluation_steps == (0, 5_000, 10_000, 20_000)
    assert config.runtime.final_audit_steps == (30_000,)
    assert config.runtime.evaluation_liveness_guard_enabled is False


def test_v47_is_fixed_reviewed_model_init_from_v46_10035(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    command = launcher.build_resume_trainer_command(paths)
    assert command.count("--initialize-from") == 1
    assert "--resume" not in command
    assert command.count("--model-initialization-attestation") == 1
    assert command.count("--model-initialization-attestation-sha256") == 1
    assert command[command.index("--model-initialization-attestation") + 1] == str(
        launcher.source_attestation_path(paths)
    )
    assert (
        command[
            command.index("--model-initialization-attestation-sha256") + 1
        ]
        == launcher.SOURCE_ATTESTATION_SHA256
    )
    assert launcher.SOURCE_RUN_ID == "9a23c1bc-7a7b-467e-ad97-95af63fe4bec"
    assert launcher.SOURCE_ENVIRONMENT_STEPS == 10_035
    assert launcher.SOURCE_POLICY_VERSION == 164
    assert launcher.SOURCE_LEARNER_UPDATES == 164
    assert launcher.SOURCE_CHECKPOINT_ID == "a497e8e7-00ea-4a89-af41-ce30691997cd"
    assert (
        launcher.SOURCE_ATTESTATION_SHA256
        == "a00096dbccde2355471b5cdf093031db53f887357ca4b28265631865aab3d518"
    )
    assert launcher.source_checkpoint_path(paths).name == "healthy-validation-step-000010035"
    assert launcher._validate_checkpoint_summary(_checkpoint_summary(paths), paths=paths) == _checkpoint_summary(paths)


def test_v47_checkpoint_pin_rejects_identity_mismatch(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    summary = _checkpoint_summary(paths)
    for key in (
        "checkpoint_id", "manifest_sha256", "metadata_sha256",
        "experiment_run_id", "environment_steps", "policy_version",
        "learner_updates", "source_total_environment_steps",
        "target_total_environment_steps", "checkpoint_format",
    ):
        with pytest.raises(launcher.LaunchError, match=key):
            launcher._validate_checkpoint_summary({**summary, key: "wrong"}, paths=paths)


def test_v47_supervisor_keeps_fixed_adapter_pins(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    manifest = tmp_path / "v47.launch.json"
    command = launcher.build_supervisor_command(paths, manifest_path=manifest)
    assert Path(command[1]).name == SCRIPT.name
    assert command[-2:] == ("--manifest", str(manifest))
    assert launcher._core.RUN_NAME == launcher.RUN_NAME
    assert launcher._core.RESUME_RUN_ID == launcher.SOURCE_RUN_ID
    assert launcher._core.RESUME_STEP == 10_035
