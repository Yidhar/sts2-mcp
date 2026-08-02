#!/usr/bin/env python3
"""Capacity-stable model-initialization launcher for the reviewed v31 run.

The v31 lineage inherits only network parameters from the frozen v30
step-30,200 checkpoint. Optimizer, replay corpora, stochastic state, rollout
queues, counters and evaluation journals start from zero. The failure-evidence
replay byte budget changes from 512 MiB to 2 GiB, so exact resume is forbidden.

This is intentionally a thin immutable launch specification over the tested
v30 model-initialization control plane; it does not duplicate the durable
supervisor/watchdog implementation.
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def _load_base() -> Any:
    path = Path(__file__).resolve().with_name(
        "launch_v30_failure_credit_v5_model_init.py"
    )
    module_name = "_sts2_v31_capacity_model_init_base"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load v30 model-init control plane: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_base = _load_base()
_base.__doc__ = __doc__

RUN_NAME = "full-run-revival-v31-failure-credit-capacity-model-init"
SOURCE_RUN_ID = "3fdb4434-b8be-46c7-95ac-7f7a9739c8f3"
SOURCE_ENVIRONMENT_STEPS = 30_200
SOURCE_TOTAL_ENVIRONMENT_STEPS = 250_000
TARGET_ENVIRONMENT_STEPS = 250_000
ADDITIONAL_ENVIRONMENT_STEPS = TARGET_ENVIRONMENT_STEPS
SOURCE_CHECKPOINT_ID = "0cd3d3fc-aec0-44cb-95fc-e441b7fb4f02"
SOURCE_MANIFEST_SHA256 = (
    "6e7a28ed5bcf433554a8520e67c9c70aea666bb7ea040bf7fa5f4d6bd3455386"
)
SOURCE_METADATA_SHA256 = (
    "2b81747314a83b74ef0224c33abfb593062060818474a8262ce326e01a69fb2d"
)
SOURCE_POLICY_VERSION = 491
SOURCE_LEARNER_UPDATES = 491
SOURCE_CHECKPOINT_RELATIVE = Path(
    "checkpoints/full-run-revival-v30-failure-credit-v5-model-init"
    f"/run-{SOURCE_RUN_ID}/periodic-step-{SOURCE_ENVIRONMENT_STEPS:09d}"
)
ACTIVE_ARTIFACT_ROOT = Path("/mnt/e/game/project/sts2_mcp_artifacts/runtime")
CONFIG_FILE = "full_run_revival_v31_failure_credit_capacity_model_init.toml"
LINEAGE_LABEL = "v31"

SCHEMA_VERSION = "sts2-v31-failure-credit-capacity-model-init-preflight-v1"
STATE_SCHEMA_VERSION = "sts2-v31-failure-credit-capacity-model-init-state-v1"
SUPERVISED_SCHEMA_VERSION = (
    "sts2-v31-failure-credit-capacity-model-init-supervised-launch-v1"
)
SUPERVISED_STATE_SCHEMA_VERSION = (
    "sts2-v31-failure-credit-capacity-model-init-supervised-state-v1"
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
    """Re-enter this adapter so v31 pins are rebound in the supervisor."""

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
    return _base.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
