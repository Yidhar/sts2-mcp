#!/usr/bin/env python3
"""Fail-closed, non-launching preflight for v28 mature refinement.

No checkpoint is embedded in the source tree.  After a checkpoint wins the
paired frozen-policy comparison, pass its absolute directory explicitly:

    python3 scripts/preflight_v28_mature_refinement.py \
      --initialize-from /absolute/path/to/v27/checkpoint

This program validates the complete atomic checkpoint and its model/encoding
ABI, verifies that it belongs to the v27 source lineage, verifies the fixed v28
configuration and production simulator/GPU inputs, and prints the exact
foreground trainer command.  It deliberately never starts training and has no
``--resume`` option.  A later supervised launcher may consume the printed
command only after operator review.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

RUN_NAME = "full-run-revival-v28-mature-refinement-model-init"
SOURCE_LINEAGE = "full-run-revival-v27-infinite-random-init"
ACTIVE_ARTIFACT_ROOT = Path("/mnt/e/game/project/sts2_mcp_artifacts/runtime")
CORRUPT_RECOVERY_ROOT = Path("/mnt/f/backoff/sts2_mcp_artifacts/runtime")
_REMOVED_TRAINER_ENVIRONMENT_KEYS = (
    "PYTHONHOME",
    "VENV_DIR",
    "STS2_HEADLESS_SIM_EXE",
)
_REVIEWED_TRAINER_ENVIRONMENT_KEYS = (
    "STS2_ARTIFACT_ROOT",
    "PYTHONPATH",
    "PYTHONNOUSERSITE",
    "PYTHONUNBUFFERED",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "PATH",
)


class PreflightError(RuntimeError):
    """The reviewed v28 model-initialization launch contract was not proven."""


@dataclass(frozen=True, slots=True)
class PreflightPaths:
    checkout_root: Path
    package_root: Path
    artifact_root: Path
    venv_python: Path
    config_path: Path
    simulator_executable: Path
    simulator_identity: Path


def _absolute_without_symlink_resolution(path: Path) -> Path:
    # A venv Python executable is normally a symlink to the system interpreter.
    # Its selected lexical path, not the final symlink target, is provenance.
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def default_paths() -> PreflightPaths:
    script_dir = Path(__file__).resolve().parent
    package_root = script_dir.parent
    checkout_root = script_dir.parents[2]
    artifact_root = ACTIVE_ARTIFACT_ROOT
    simulator = artifact_root / "dependencies/sts2-ai/STS2AI/ENV/Sim/HeadlessSim/bin/Release/net9.0/HeadlessSim.exe"
    return PreflightPaths(
        checkout_root=checkout_root,
        package_root=package_root,
        artifact_root=artifact_root,
        venv_python=artifact_root / "environments/wsl-rocm/bin/python",
        config_path=(package_root / "config/experiments/full_run_revival_v28_mature_refinement_model_init.toml"),
        simulator_executable=simulator,
        simulator_identity=simulator.with_name(simulator.name + ".identity.json"),
    )


def validate_layout(
    paths: PreflightPaths,
    *,
    enforce_active_root: bool = True,
) -> PreflightPaths:
    resolved = PreflightPaths(
        checkout_root=paths.checkout_root.resolve(strict=False),
        package_root=paths.package_root.resolve(strict=False),
        artifact_root=paths.artifact_root.resolve(strict=False),
        venv_python=_absolute_without_symlink_resolution(paths.venv_python),
        config_path=paths.config_path.resolve(strict=False),
        simulator_executable=paths.simulator_executable.resolve(strict=False),
        simulator_identity=paths.simulator_identity.resolve(strict=False),
    )
    forbidden = CORRUPT_RECOVERY_ROOT.resolve(strict=False)
    for candidate in (
        resolved.artifact_root,
        resolved.venv_python,
        resolved.simulator_executable,
        resolved.simulator_identity,
    ):
        if candidate == forbidden or _is_within(candidate, forbidden):
            raise PreflightError(f"corrupt recovery tree is forbidden: {candidate}")
    if enforce_active_root and resolved.artifact_root != ACTIVE_ARTIFACT_ROOT.resolve(strict=False):
        raise PreflightError(f"artifact root must be the original active E: runtime: {resolved.artifact_root}")
    if (
        resolved.artifact_root == resolved.checkout_root
        or _is_within(resolved.artifact_root, resolved.checkout_root)
        or _is_within(resolved.checkout_root, resolved.artifact_root)
    ):
        raise PreflightError("artifact root and source checkout must be disjoint")
    if not _is_within(resolved.venv_python, resolved.artifact_root):
        raise PreflightError("ROCm venv escapes the active artifact root")
    if not _is_within(resolved.simulator_executable, resolved.artifact_root):
        raise PreflightError("simulator escapes the active artifact root")
    if not _is_within(resolved.config_path, resolved.package_root):
        raise PreflightError("v28 config escapes the RL package root")
    return resolved


def validate_initialization_checkpoint(
    checkpoint: str | Path,
    *,
    paths: PreflightPaths,
) -> Path:
    raw = Path(checkpoint).expanduser()
    if not raw.is_absolute():
        raise PreflightError("--initialize-from must be an absolute path")
    selected = raw.resolve(strict=False)
    source_root = (paths.artifact_root / "checkpoints" / SOURCE_LINEAGE).resolve(strict=False)
    if not _is_within(selected, source_root):
        raise PreflightError(
            f"initialization checkpoint must be an atomic checkpoint from the {SOURCE_LINEAGE!r} lineage"
        )
    if not selected.is_dir():
        raise PreflightError(f"initialization checkpoint is not a directory: {selected}")
    return selected


def build_trainer_command(
    paths: PreflightPaths,
    *,
    initialize_from: Path,
) -> tuple[str, ...]:
    return (
        str(paths.venv_python),
        "-m",
        "sts2_rl.train",
        "--profile",
        "preheat",
        "--config",
        str(paths.config_path),
        "--device",
        "cuda",
        "--collector-device",
        "cpu",
        "--backend",
        "headless",
        "--sim-exe",
        str(paths.simulator_executable),
        "--sim-identity",
        str(paths.simulator_identity),
        "--initialize-from",
        str(initialize_from),
    )


def build_trainer_environment(
    paths: PreflightPaths,
    *,
    base_environment: dict[str, str] | None = None,
) -> dict[str, str]:
    """Build the isolated environment required by the reviewed command.

    The preflight does not spawn this command.  Returning the environment as
    part of its reviewed payload prevents a later supervised launcher from
    accidentally using a stale artifact root, Python installation, or
    simulator selected through ambient process variables.
    """

    environment = dict(os.environ if base_environment is None else base_environment)
    for key in _REMOVED_TRAINER_ENVIRONMENT_KEYS:
        environment.pop(key, None)
    environment["STS2_ARTIFACT_ROOT"] = str(paths.artifact_root)
    environment["PYTHONPATH"] = str(paths.package_root)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONUNBUFFERED"] = "1"
    environment["OMP_NUM_THREADS"] = environment.get("OMP_NUM_THREADS", "4")
    environment["MKL_NUM_THREADS"] = environment.get("MKL_NUM_THREADS", "4")
    environment["OPENBLAS_NUM_THREADS"] = environment.get("OPENBLAS_NUM_THREADS", "4")
    environment["NUMEXPR_NUM_THREADS"] = environment.get("NUMEXPR_NUM_THREADS", "4")
    environment["PATH"] = str(paths.venv_python.parent) + os.pathsep + environment.get("PATH", "")
    return environment


def validate_trainer_environment(
    environment: dict[str, str],
    *,
    paths: PreflightPaths,
) -> dict[str, str]:
    for key in _REMOVED_TRAINER_ENVIRONMENT_KEYS:
        if key in environment:
            raise PreflightError(f"trainer environment retained forbidden ambient variable: {key}")
    expected = {
        "STS2_ARTIFACT_ROOT": str(paths.artifact_root),
        "PYTHONPATH": str(paths.package_root),
        "PYTHONNOUSERSITE": "1",
        "PYTHONUNBUFFERED": "1",
    }
    for key, value in expected.items():
        if environment.get(key) != value:
            raise PreflightError(f"trainer environment changed reviewed {key}")
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        thread_value = environment.get(key)
        if thread_value is None or not thread_value.isdigit() or int(thread_value) < 1:
            raise PreflightError(f"trainer environment has invalid {key}: {thread_value!r}")
    expected_path_prefix = str(paths.venv_python.parent) + os.pathsep
    if not environment.get("PATH", "").startswith(expected_path_prefix):
        raise PreflightError("trainer PATH does not prefer the reviewed ROCm venv")
    return environment


def trainer_environment_contract(environment: dict[str, str]) -> dict[str, Any]:
    """Return only reviewed launch variables; never leak the ambient environment."""

    return {
        "set": {key: environment[key] for key in _REVIEWED_TRAINER_ENVIRONMENT_KEYS},
        "unset": list(_REMOVED_TRAINER_ENVIRONMENT_KEYS),
    }


def validate_trainer_command(
    command: tuple[str, ...],
    *,
    paths: PreflightPaths,
    initialize_from: Path,
) -> tuple[str, ...]:
    if command.count("--initialize-from") != 1:
        raise PreflightError("trainer command must contain one --initialize-from")
    if "--resume" in command:
        raise PreflightError("v28 refinement must never masquerade as exact resume")
    index = command.index("--initialize-from")
    if index + 1 >= len(command) or command[index + 1] != str(initialize_from):
        raise PreflightError("trainer command initialization source changed")
    expected = build_trainer_command(paths, initialize_from=initialize_from)
    if command != expected:
        raise PreflightError("trainer command differs from the reviewed v28 command")
    return command


def _assert_refinement_config(config: Any) -> None:
    expected = {
        "architecture": "relational_candidate_v3",
        "learning_rate": 0.0001,
        "entropy_weight": 0.006,
        "entropy_weight_end": 0.002,
        "entropy_decay_updates": 3000,
        "queue_capacity": 24,
        "revival_budget": -1,
        "epsilon_start": 0.15,
        "epsilon_end": 0.05,
        "epsilon_decay_steps": 150000,
        "rocm_sdpa_backend": "math",
        "total_environment_steps": 100000,
        "seed": 3000000,
        "checkpoint_interval_steps": 10000,
    }
    actual = {
        "architecture": config.model.architecture,
        "learning_rate": config.optimization.learning_rate,
        "entropy_weight": config.optimization.entropy_weight,
        "entropy_weight_end": config.optimization.entropy_weight_end,
        "entropy_decay_updates": config.optimization.entropy_decay_updates,
        "queue_capacity": config.rollout.queue_capacity,
        "revival_budget": config.curriculum.revival_budget,
        "epsilon_start": config.curriculum.epsilon_start,
        "epsilon_end": config.curriculum.epsilon_end,
        "epsilon_decay_steps": config.curriculum.epsilon_decay_steps,
        "rocm_sdpa_backend": config.runtime.rocm_sdpa_backend,
        "total_environment_steps": config.runtime.total_environment_steps,
        "seed": config.runtime.seed,
        "checkpoint_interval_steps": config.runtime.checkpoint_interval_steps,
    }
    if actual != expected:
        raise PreflightError(
            "effective v28 refinement config changed: "
            + json.dumps({"expected": expected, "actual": actual}, sort_keys=True)
        )
    if config.environment.backend != "headless" or config.environment.scenario != "full-run":
        raise PreflightError("v28 requires the headless full-run environment")
    if config.runtime.evaluation_steps != (0, 25000, 50000, 75000):
        raise PreflightError("v28 normal evaluation schedule changed")
    if config.runtime.evaluation_episodes != 16:
        raise PreflightError("v28 normal evaluation sample count changed")
    if config.runtime.early_evaluation_steps != (5000, 10000):
        raise PreflightError("v28 early evaluation schedule changed")
    if config.runtime.early_evaluation_episodes != 8:
        raise PreflightError("v28 early evaluation sample count changed")
    if config.runtime.final_audit_steps != (100000,):
        raise PreflightError("v28 final audit schedule changed")
    if config.runtime.final_audit_episodes != 64:
        raise PreflightError("v28 final audit sample count changed")
    if not config.runtime.evaluation_liveness_guard_enabled:
        raise PreflightError("v28 liveness guard must remain enabled")
    if not config.runtime.log_dir.endswith(RUN_NAME):
        raise PreflightError("v28 log directory changed")
    if not config.runtime.checkpoint_dir.endswith(RUN_NAME):
        raise PreflightError("v28 checkpoint directory changed")


def _assert_v27_source(validated: Any) -> None:
    metadata = validated.metadata
    source_config = metadata.get("training_config")
    if not isinstance(source_config, dict):
        raise PreflightError("initialization checkpoint has no training_config")
    runtime = source_config.get("runtime")
    curriculum = source_config.get("curriculum")
    model = source_config.get("model")
    if not isinstance(runtime, dict) or not str(runtime.get("log_dir", "")).endswith(SOURCE_LINEAGE):
        raise PreflightError("initialization checkpoint is not a v27 runtime")
    if not isinstance(curriculum, dict) or curriculum.get("revival_budget") != -1:
        raise PreflightError("initialization source did not use unlimited hidden revival")
    if not isinstance(model, dict) or model.get("architecture") != "relational_candidate_v3":
        raise PreflightError("initialization source model architecture changed")


def _load_refinement_config(path: Path) -> Any:
    from sts2_rl.training.config import load_training_config

    return load_training_config(profile="preheat", config_path=path)


def _preflight_selected_checkpoint(checkpoint: Path, *, config: Any) -> Any:
    from sts2_rl.training.checkpointing import preflight_model_initialization

    return preflight_model_initialization(checkpoint, config=config)


def _verify_runtime_inputs(paths: PreflightPaths) -> dict[str, Any]:
    import torch

    from sts2_rl.simulator_identity import verify_headless_simulator

    if not torch.cuda.is_available():
        raise PreflightError("ROCm GPU is unavailable; refusing CPU fallback")
    simulator = verify_headless_simulator(
        paths.simulator_executable,
        identity_path=paths.simulator_identity,
    )
    return {
        "torch_version": torch.__version__,
        "device_name": torch.cuda.get_device_name(0),
        "simulator": simulator.to_mapping(),
    }


def run_preflight(
    paths: PreflightPaths,
    *,
    initialize_from: str | Path,
    enforce_active_root: bool = True,
    verify_runtime: bool = True,
) -> dict[str, Any]:
    paths = validate_layout(paths, enforce_active_root=enforce_active_root)
    selected = validate_initialization_checkpoint(initialize_from, paths=paths)
    if not paths.config_path.is_file():
        raise PreflightError(f"v28 config was not found: {paths.config_path}")
    if not paths.venv_python.is_file():
        raise PreflightError(f"ROCm venv Python was not found: {paths.venv_python}")

    config = _load_refinement_config(paths.config_path)
    _assert_refinement_config(config)
    validated = _preflight_selected_checkpoint(selected, config=config)
    _assert_v27_source(validated)

    runtime: dict[str, Any] | None = None
    if verify_runtime:
        runtime = _verify_runtime_inputs(paths)

    command = validate_trainer_command(
        build_trainer_command(paths, initialize_from=selected),
        paths=paths,
        initialize_from=selected,
    )
    environment = validate_trainer_environment(
        build_trainer_environment(paths),
        paths=paths,
    )
    state = validated.metadata.get("training_state")
    manifest_id = validated.manifest.get("checkpoint_id")
    return {
        "status": "preflight-passed",
        "run_name": RUN_NAME,
        "initialization": {
            "mode": "model_initialization",
            "checkpoint": str(selected),
            "checkpoint_id": manifest_id,
            "source_training_state": state,
            "network_parameters_inherited": True,
            "optimizer_rollouts_rng_and_counters_reset": True,
        },
        "config_fingerprint_sha256": config.fingerprint_sha256(),
        "trainer_command": list(command),
        "trainer_environment": trainer_environment_contract(environment),
        "runtime": runtime,
        "training_started": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate, but never launch, a selected v27 -> v28 model initialization"
    )
    parser.add_argument(
        "--initialize-from",
        required=True,
        help="absolute atomic checkpoint selected by paired frozen evaluation",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    payload = run_preflight(
        default_paths(),
        initialize_from=args.initialize_from,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
