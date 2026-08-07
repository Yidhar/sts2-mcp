#!/usr/bin/env python3
"""Launch the reviewed v40 budget-64 completion-recovery lineage.

The immutable source is v39's healthy validation anchor at local environment
step 50,261.  Compatible network parameters and mature schedule clocks are
inherited.  Optimizer, replay stores, RNG, actor queue, evaluation journal and
lineage-local counters restart because the revival budget changes from 32 to
64; this is model initialization, never exact resume.
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
    module_name = "_sts2_v40_budget64_recovery_model_init_base"
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

RUN_NAME = "full-run-revival-v40-budget64-recovery-model-init"
SOURCE_RUN_ID = "e331ab97-94a6-4f0d-bde9-407261e509b9"
SOURCE_ENVIRONMENT_STEPS = 50_261
SOURCE_TOTAL_ENVIRONMENT_STEPS = 250_000
TARGET_ENVIRONMENT_STEPS = 250_000
ADDITIONAL_ENVIRONMENT_STEPS = TARGET_ENVIRONMENT_STEPS
SOURCE_CHECKPOINT_ID = "968d0302-b712-4b04-a2df-50554722500c"
SOURCE_MANIFEST_SHA256 = (
    "4c7c36e528f98be5d450319f151d457190c3c751e4c1f27499a8bbd01ea86442"
)
SOURCE_METADATA_SHA256 = (
    "c9460511144b05f2bc97028288fdc4ca2a0bae707b8e875687e6950de9eecbf0"
)
SOURCE_POLICY_VERSION = 824
SOURCE_LEARNER_UPDATES = 824
SOURCE_CHECKPOINT_RELATIVE = Path(
    "checkpoints/full-run-revival-v39-budget32-natural-strategy-model-init"
    f"/run-{SOURCE_RUN_ID}/healthy-validation-step-{SOURCE_ENVIRONMENT_STEPS:09d}"
)
ACTIVE_ARTIFACT_ROOT = Path("/mnt/e/game/project/sts2_mcp_artifacts/runtime")
CONFIG_FILE = "full_run_revival_v40_budget64_recovery_model_init.toml"
LINEAGE_LABEL = "v40"

SCHEMA_VERSION = "sts2-v40-budget64-recovery-model-init-preflight-v1"
STATE_SCHEMA_VERSION = "sts2-v40-budget64-recovery-model-init-state-v1"
SUPERVISED_SCHEMA_VERSION = (
    "sts2-v40-budget64-recovery-model-init-supervised-launch-v1"
)
SUPERVISED_STATE_SCHEMA_VERSION = (
    "sts2-v40-budget64-recovery-model-init-supervised-state-v1"
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
    """Re-enter this adapter so the immutable v39 source pins remain bound."""

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
