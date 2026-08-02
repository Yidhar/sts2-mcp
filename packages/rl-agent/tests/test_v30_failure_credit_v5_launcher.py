from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from sts2_rl.training import load_training_config

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PACKAGE_ROOT / "scripts" / "launch_v30_failure_credit_v5_model_init.py"
CONFIG = PACKAGE_ROOT / "config/experiments/full_run_revival_v30_failure_credit_v5_model_init.toml"
SPEC = importlib.util.spec_from_file_location("v30_failure_credit_v5_launcher", SCRIPT)
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


def test_v30_recipe_enables_replay_v5_matching_as_a_new_lineage() -> None:
    config = load_training_config(profile="preheat", config_path=CONFIG)

    assert config.runtime.total_environment_steps == 250_000
    assert config.runtime.seed == 5_000_000
    assert config.curriculum.revival_budget == -1
    assert config.transaction_learning.enabled is False
    assert config.failure_credit.mode == "learning"
    assert config.failure_credit.matched_outcome_pair_quota == 1
    assert config.failure_credit.maximum_matched_pairs_per_publication == 8


def test_v30_command_is_model_initialization_not_exact_resume(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    command = launcher.build_resume_trainer_command(paths)

    assert command.count("--initialize-from") == 1
    assert command[command.index("--initialize-from") + 1] == str(launcher.source_checkpoint_path(paths))
    assert "--resume" not in command
    assert command[command.index("--device") + 1] == "cuda"
    assert command[command.index("--collector-device") + 1] == "cpu"
    assert command[command.index("--backend") + 1] == "headless"
    with pytest.raises(launcher.LaunchError, match="command was modified"):
        launcher.validate_fixed_model_initialization_command(
            (*command, "--resume", str(tmp_path / "forbidden")),
            paths=paths,
        )


def test_v30_checkpoint_source_is_fully_pinned(tmp_path: Path) -> None:
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
        "source_total_environment_steps",
        "target_total_environment_steps",
        "checkpoint_format",
    ):
        with pytest.raises(launcher.LaunchError, match=key):
            launcher._validate_checkpoint_summary(
                {**summary, key: "wrong"},
                paths=paths,
            )


def test_v30_supervised_manifest_cannot_claim_exact_resume(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    paths.manifest_dir.mkdir(parents=True)
    manifest_path = paths.manifest_dir / "v30.launch.json"
    summary = _checkpoint_summary(paths)
    manifest = {
        "schema_version": launcher.SUPERVISED_SCHEMA_VERSION,
        "run_name": launcher.RUN_NAME,
        "launch_id": "launch-v30",
        "trainer_command": list(launcher.build_resume_trainer_command(paths)),
        "initialization": {
            "mode": "model-initialization",
            "resume_checkpoint": None,
            "model_initialization_checkpoint": str(launcher.source_checkpoint_path(paths)),
        },
        "source_checkpoint": summary,
        "resume_checkpoint": summary,
        "preexisting_run_directories": [],
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    assert (
        launcher._validate_supervised_manifest(
            paths,
            manifest_path=manifest_path,
        )
        == manifest
    )

    manifest["initialization"]["resume_checkpoint"] = str(launcher.source_checkpoint_path(paths))
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(launcher.LaunchError, match="falsely claims exact resume"):
        launcher._validate_supervised_manifest(paths, manifest_path=manifest_path)


def test_v30_supervisor_binds_only_a_reset_model_initialization_run(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    run_root = paths.artifact_root / "runs" / launcher.RUN_NAME
    run_dir = run_root / "run-new-lineage"
    run_dir.mkdir(parents=True)
    metrics = run_dir / "metrics.jsonl"
    source_path = str(launcher.source_checkpoint_path(paths))
    run_start = {
        "event": "run_start",
        "run_id": "new-lineage",
        "unix_s": 1.0,
        "checkpoint_load": {
            "mode": "model_initialization",
            "parent_checkpoint": source_path,
            "network_parameters_initialized": True,
            "optimizer_rollouts_rng_and_counters_reset": True,
            "source_checkpoint": {
                "checkpoint_id": launcher.SOURCE_CHECKPOINT_ID,
                "environment_steps": launcher.SOURCE_ENVIRONMENT_STEPS,
                "manifest_sha256": launcher.SOURCE_MANIFEST_SHA256,
                "metadata_sha256": launcher.SOURCE_METADATA_SHA256,
                "path": source_path,
                "policy_version": launcher.SOURCE_POLICY_VERSION,
                "usage": "model_parameter_initialization_only",
            },
        },
        "state": {
            "actor_policy_version": 0,
            "consumed_unrolls": 0,
            "environment_steps": 0,
            "episodes": 0,
            "evaluation_episodes": 0,
            "learner_updates": 0,
            "policy_version": 0,
        },
    }
    metrics.write_text(json.dumps(run_start) + "\n", encoding="utf-8")

    assert launcher._core.discover_successor_metrics(
        paths,
        preexisting_run_directories=(),
    ) == (metrics.resolve(strict=False), run_start)

    run_start["checkpoint_load"]["mode"] = "exact_resume"
    metrics.write_text(json.dumps(run_start) + "\n", encoding="utf-8")
    assert (
        launcher._core.discover_successor_metrics(
            paths,
            preexisting_run_directories=(),
        )
        is None
    )
