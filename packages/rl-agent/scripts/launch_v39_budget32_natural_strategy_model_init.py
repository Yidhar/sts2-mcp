#!/usr/bin/env python3
"""Launch the reviewed v39 budget-32 natural-strategy model-init lineage.

The immutable source is v38's final checkpoint at local environment step
100,000. Compatible network parameters and mature schedule clocks are
inherited. Optimizer, replay stores, RNG, actor queue and lineage-local
counters restart because the revival budget changes from 16 to 32; carrying
budget-16 replay into the new course would mix two different return regimes.
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast


def _load_base() -> Any:
    path = Path(__file__).resolve().with_name(
        "launch_v30_failure_credit_v5_model_init.py"
    )
    module_name = "_sts2_v39_budget32_natural_strategy_model_init_base"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load model-init control plane: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_base = _load_base()
_base.__doc__ = __doc__

RUN_NAME = "full-run-revival-v39-budget32-natural-strategy-model-init"
SOURCE_RUN_ID = "3051fdbd-7987-4319-a3ac-cd680fa3e408"
SOURCE_ENVIRONMENT_STEPS = 100_000
SOURCE_TOTAL_ENVIRONMENT_STEPS = 100_000
TARGET_ENVIRONMENT_STEPS = 250_000
ADDITIONAL_ENVIRONMENT_STEPS = TARGET_ENVIRONMENT_STEPS
SOURCE_CHECKPOINT_ID = "74797dca-7c88-4f2d-b0a2-52f028291f3e"
SOURCE_MANIFEST_SHA256 = (
    "3ddbeb700c60b6deedcc8f99280a3cbe66306ef295003d4378dd46cbf0704251"
)
SOURCE_METADATA_SHA256 = (
    "be09e1d80a4c23edd7000040b61dec33d06176410d29bfa600abffad9f79733b"
)
SOURCE_POLICY_VERSION = 1_675
SOURCE_LEARNER_UPDATES = 1_675
SOURCE_CHECKPOINT_RELATIVE = Path(
    "checkpoints/full-run-revival-v38-natural-strategy-model-init"
    f"/run-{SOURCE_RUN_ID}/final-step-{SOURCE_ENVIRONMENT_STEPS:09d}"
)
ACTIVE_ARTIFACT_ROOT = Path("/mnt/e/game/project/sts2_mcp_artifacts/runtime")
CONFIG_FILE = "full_run_revival_v39_budget32_natural_strategy_model_init.toml"
LINEAGE_LABEL = "v39"

SCHEMA_VERSION = "sts2-v39-budget32-natural-strategy-model-init-preflight-v1"
STATE_SCHEMA_VERSION = "sts2-v39-budget32-natural-strategy-model-init-state-v1"
SUPERVISED_SCHEMA_VERSION = (
    "sts2-v39-budget32-natural-strategy-model-init-supervised-launch-v1"
)
SUPERVISED_STATE_SCHEMA_VERSION = (
    "sts2-v39-budget32-natural-strategy-model-init-supervised-state-v1"
)
LEARNER_STALL_TIMEOUT_SECONDS = 30.0 * 60.0

_OVERRIDES = {
    "RUN_NAME": RUN_NAME,
    "SOURCE_RUN_ID": SOURCE_RUN_ID,
    "SOURCE_ENVIRONMENT_STEPS": SOURCE_ENVIRONMENT_STEPS,
    "SOURCE_TOTAL_ENVIRONMENT_STEPS": SOURCE_TOTAL_ENVIRONMENT_STEPS,
    "TARGET_ENVIRONMENT_STEPS": TARGET_ENVIRONMENT_STEPS,
    "ADDITIONAL_ENVIRONMENT_STEPS": ADDITIONAL_ENVIRONMENT_STEPS,
    "SOURCE_CHECKPOINT_ID": SOURCE_CHECKPOINT_ID,
    "SOURCE_MANIFEST_SHA256": SOURCE_MANIFEST_SHA256,
    "SOURCE_METADATA_SHA256": SOURCE_METADATA_SHA256,
    "SOURCE_POLICY_VERSION": SOURCE_POLICY_VERSION,
    "SOURCE_LEARNER_UPDATES": SOURCE_LEARNER_UPDATES,
    "SOURCE_CHECKPOINT_RELATIVE": SOURCE_CHECKPOINT_RELATIVE,
    "ACTIVE_ARTIFACT_ROOT": ACTIVE_ARTIFACT_ROOT,
    "CONFIG_FILE": CONFIG_FILE,
    "LINEAGE_LABEL": LINEAGE_LABEL,
    "SCHEMA_VERSION": SCHEMA_VERSION,
    "STATE_SCHEMA_VERSION": STATE_SCHEMA_VERSION,
    "SUPERVISED_SCHEMA_VERSION": SUPERVISED_SCHEMA_VERSION,
    "SUPERVISED_STATE_SCHEMA_VERSION": SUPERVISED_STATE_SCHEMA_VERSION,
    "LEARNER_STALL_TIMEOUT_SECONDS": LEARNER_STALL_TIMEOUT_SECONDS,
}
for _name, _value in _OVERRIDES.items():
    setattr(_base, _name, _value)


def build_supervisor_command(
    paths: Any,
    *,
    manifest_path: Path,
) -> tuple[str, ...]:
    """Re-enter this adapter so the immutable v38 source pins remain bound."""

    return (
        str(paths.venv_python),
        str(Path(__file__).resolve()),
        "supervise",
        "--manifest",
        str(manifest_path),
    )


_base.build_supervisor_command = build_supervisor_command
_base._configure_supervisor_core()

LaunchError = _base.LaunchError
LaunchPaths = _base.LaunchPaths
default_paths = _base.default_paths
build_environment = _base.build_environment
source_checkpoint_path = _base.source_checkpoint_path
validate_fixed_model_initialization_command = (
    _base.validate_fixed_model_initialization_command
)
build_resume_trainer_command = _base.build_resume_trainer_command
_validate_checkpoint_summary = _base._validate_checkpoint_summary
_verify_source_checkpoint = _base._verify_source_checkpoint
run_model_initialization_preflight = _base.run_model_initialization_preflight
_validate_supervised_manifest = _base._validate_supervised_manifest
_matching_model_initialization_run_start = (
    _base._matching_model_initialization_run_start
)
initialize = _base.initialize
_core = _base._core


def main(argv: Sequence[str] | None = None) -> int:
    return cast(int, _base.main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
