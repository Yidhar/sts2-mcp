from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tests.archived_experiment_config import load_archived_training_config

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = PACKAGE_ROOT / "config" / "experiments"
V27_CONFIG = EXPERIMENT_ROOT / "full_run_revival_v27_infinite_random_init.toml"
V28_CONFIG = EXPERIMENT_ROOT / "full_run_revival_v28_mature_refinement_model_init.toml"
SCRIPT = PACKAGE_ROOT / "scripts" / "preflight_v28_mature_refinement.py"
SPEC = importlib.util.spec_from_file_location("v28_mature_refinement_preflight", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
launcher = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = launcher
SPEC.loader.exec_module(launcher)


def _config() -> Any:
    return load_archived_training_config(profile="preheat", config_path=V28_CONFIG)


def _paths(tmp_path: Path) -> Any:
    checkout = tmp_path / "source"
    package = checkout / "packages" / "rl-agent"
    artifact = tmp_path / "artifacts" / "runtime"
    simulator = artifact / "dependencies" / "HeadlessSim.exe"
    return launcher.PreflightPaths(
        checkout_root=checkout,
        package_root=package,
        artifact_root=artifact,
        venv_python=artifact / "environments" / "wsl-rocm" / "bin" / "python",
        config_path=package / "config" / "v28.toml",
        simulator_executable=simulator,
        simulator_identity=simulator.with_name(simulator.name + ".identity.json"),
    )


def _materialize_layout(paths: Any) -> Path:
    paths.config_path.parent.mkdir(parents=True, exist_ok=True)
    paths.config_path.write_text("# test placeholder\n", encoding="utf-8")
    paths.venv_python.parent.mkdir(parents=True, exist_ok=True)
    paths.venv_python.write_text("python\n", encoding="utf-8")
    paths.simulator_executable.parent.mkdir(parents=True, exist_ok=True)
    paths.simulator_executable.write_bytes(b"sim")
    paths.simulator_identity.write_text("{}\n", encoding="utf-8")
    checkpoint = (
        paths.artifact_root
        / "checkpoints"
        / launcher.SOURCE_LINEAGE
        / "run-selected-later"
        / "periodic-step-selected-later"
    )
    checkpoint.mkdir(parents=True, exist_ok=True)
    return checkpoint


def test_v28_overlay_is_conservative_model_initialization_lineage() -> None:
    source = load_archived_training_config(profile="preheat", config_path=V27_CONFIG)
    refinement = _config()

    # Network and world semantics stay load-compatible.  Changed optimizer and
    # exploration semantics intentionally make exact resume fail closed.
    assert refinement.model == source.model
    assert refinement.environment == source.environment
    assert refinement.transaction_learning == source.transaction_learning
    assert refinement.episodic_learning == source.episodic_learning
    assert refinement.rollout == source.rollout
    assert refinement.diagnostics == source.diagnostics
    assert refinement.model.to_model_config() == source.model.to_model_config()
    assert refinement.lineage_mapping() != source.lineage_mapping()

    assert refinement.optimization.learning_rate == pytest.approx(1e-4)
    assert refinement.optimization.entropy_weight == pytest.approx(0.006)
    assert refinement.optimization.entropy_weight_end == pytest.approx(0.002)
    assert refinement.optimization.entropy_decay_updates == 3000
    assert refinement.rollout.queue_capacity == 24
    assert refinement.curriculum.revival_budget == -1
    assert refinement.curriculum.epsilon_start == pytest.approx(0.15)
    assert refinement.curriculum.epsilon_end == pytest.approx(0.05)
    assert refinement.curriculum.epsilon_decay_steps == 150_000

    assert refinement.runtime.rocm_sdpa_backend == "math"
    assert refinement.runtime.total_environment_steps == 100_000
    assert refinement.runtime.seed == 3_000_000
    assert refinement.runtime.checkpoint_interval_steps == 10_000
    assert refinement.runtime.evaluation_steps == (0, 25_000, 50_000, 75_000)
    assert refinement.runtime.evaluation_episodes == 16
    assert refinement.runtime.early_evaluation_steps == (5_000, 10_000)
    assert refinement.runtime.early_evaluation_episodes == 8
    assert refinement.runtime.final_audit_steps == (100_000,)
    assert refinement.runtime.final_audit_episodes == 64
    assert refinement.runtime.evaluation_liveness_guard_enabled


def test_preflight_command_is_model_initialization_only(tmp_path: Path) -> None:
    paths = launcher.validate_layout(_paths(tmp_path), enforce_active_root=False)
    checkpoint = _materialize_layout(paths).resolve()
    checkpoint = launcher.validate_initialization_checkpoint(
        checkpoint,
        paths=paths,
    )
    command = launcher.build_trainer_command(
        paths,
        initialize_from=checkpoint,
    )
    launcher.validate_trainer_command(
        command,
        paths=paths,
        initialize_from=checkpoint,
    )

    assert command.count("--initialize-from") == 1
    assert command[command.index("--initialize-from") + 1] == str(checkpoint)
    assert "--resume" not in command
    assert command[command.index("--config") + 1] == str(paths.config_path)
    assert command[command.index("--device") + 1] == "cuda"
    assert command[command.index("--backend") + 1] == "headless"

    with pytest.raises(launcher.PreflightError, match="never masquerade"):
        launcher.validate_trainer_command(
            (*command, "--resume", str(checkpoint)),
            paths=paths,
            initialize_from=checkpoint,
        )


def test_preflight_builds_isolated_active_runtime_environment(tmp_path: Path) -> None:
    paths = launcher.validate_layout(_paths(tmp_path), enforce_active_root=False)
    environment = launcher.build_trainer_environment(
        paths,
        base_environment={
            "PATH": "/ambient/bin",
            "PYTHONHOME": "/wrong/python",
            "VENV_DIR": "/wrong/venv",
            "STS2_HEADLESS_SIM_EXE": "/wrong/simulator",
            "STS2_ARTIFACT_ROOT": "/mnt/f/backoff/sts2_mcp_artifacts/runtime",
            "PYTHONPATH": "/wrong/source",
        },
    )
    launcher.validate_trainer_environment(environment, paths=paths)
    contract = launcher.trainer_environment_contract(environment)

    assert environment["STS2_ARTIFACT_ROOT"] == str(paths.artifact_root)
    assert environment["PYTHONPATH"] == str(paths.package_root)
    assert environment["PATH"].startswith(str(paths.venv_python.parent))
    assert "/mnt/f/" not in contract["set"]["STS2_ARTIFACT_ROOT"].casefold()
    assert set(contract["unset"]) == {
        "PYTHONHOME",
        "VENV_DIR",
        "STS2_HEADLESS_SIM_EXE",
    }
    assert not (set(contract["set"]) & set(contract["unset"]))
    for key in contract["unset"]:
        assert key not in environment


def test_checkpoint_is_chosen_at_runtime_but_must_belong_to_v27(
    tmp_path: Path,
) -> None:
    paths = launcher.validate_layout(_paths(tmp_path), enforce_active_root=False)
    selected = _materialize_layout(paths)
    assert launcher.validate_initialization_checkpoint(selected, paths=paths) == selected.resolve()

    wrong = paths.artifact_root / "checkpoints" / "some-other-lineage" / "checkpoint"
    wrong.mkdir(parents=True)
    with pytest.raises(launcher.PreflightError, match="v27-infinite-random-init"):
        launcher.validate_initialization_checkpoint(wrong, paths=paths)
    with pytest.raises(launcher.PreflightError, match="absolute path"):
        launcher.validate_initialization_checkpoint(Path("relative/checkpoint"), paths=paths)


def test_full_preflight_proves_reset_provenance_without_starting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = launcher.validate_layout(_paths(tmp_path), enforce_active_root=False)
    selected = _materialize_layout(paths).resolve()
    config = _config()
    validated = SimpleNamespace(
        metadata={
            "training_config": {
                "runtime": {"log_dir": f"runs/{launcher.SOURCE_LINEAGE}"},
                "curriculum": {"revival_budget": -1},
                "model": {"architecture": "relational_candidate_v3"},
            },
            "training_state": {
                "environment_steps": 72_075,
                "policy_version": 1_100,
            },
        },
        manifest={"checkpoint_id": "selected-after-paired-evaluation"},
    )
    seen: dict[str, Any] = {}

    monkeypatch.setattr(launcher, "_load_refinement_config", lambda path: config)

    def fake_checkpoint_preflight(checkpoint: Path, *, config: Any) -> Any:
        seen["checkpoint"] = checkpoint
        seen["config"] = config
        return validated

    monkeypatch.setattr(
        launcher,
        "_preflight_selected_checkpoint",
        fake_checkpoint_preflight,
    )
    monkeypatch.setattr(
        launcher,
        "_verify_runtime_inputs",
        lambda _paths: {"torch_version": "test", "device_name": "test-gpu"},
    )

    payload = launcher.run_preflight(
        paths,
        initialize_from=selected,
        enforce_active_root=False,
    )

    assert seen == {"checkpoint": selected, "config": config}
    assert payload["status"] == "preflight-passed"
    assert payload["training_started"] is False
    assert payload["initialization"]["mode"] == "model_initialization"
    assert payload["initialization"]["network_parameters_inherited"] is True
    assert payload["initialization"]["optimizer_rollouts_rng_and_counters_reset"] is True
    assert "--initialize-from" in payload["trainer_command"]
    assert "--resume" not in payload["trainer_command"]
    assert payload["trainer_environment"]["set"]["STS2_ARTIFACT_ROOT"] == str(paths.artifact_root)
    assert payload["trainer_environment"]["set"]["PYTHONPATH"] == str(paths.package_root)
    assert payload["trainer_environment"]["unset"] == [
        "PYTHONHOME",
        "VENV_DIR",
        "STS2_HEADLESS_SIM_EXE",
    ]


def test_default_paths_use_original_e_runtime_and_no_selected_checkpoint() -> None:
    paths = launcher.default_paths()
    assert paths.artifact_root == Path("/mnt/e/game/project/sts2_mcp_artifacts/runtime")
    assert "/mnt/f/" not in paths.venv_python.as_posix().casefold()
    assert paths.config_path.name == ("full_run_revival_v28_mature_refinement_model_init.toml")
    # Selection is intentionally an operator argument after paired evaluation;
    # PreflightPaths contains no checkpoint field or hard-coded step/UUID.
    assert not hasattr(paths, "checkpoint")
