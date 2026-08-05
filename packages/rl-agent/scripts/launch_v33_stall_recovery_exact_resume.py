#!/usr/bin/env python3
"""Exact-resume the interrupted v33 lineage from its last atomic checkpoint.

This adapter restores model, optimizer, both replay stores, rollout queue,
schedule clocks, counters and RNG from the reviewed v33 checkpoint at 30,089
environment steps.  The trainer still targets the original absolute 100,000
step horizon.  It cannot fresh-start or perform model initialization.

The native-stall watchdog is stage-aware and uses a five-minute learner-only
timeout.  Evaluation and atomic checkpoint publication are not eligible for
that short timeout.
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast


def _load_base() -> Any:
    path = Path(__file__).resolve().with_name("launch_v29_exact_continuation_350k.py")
    module_name = "_sts2_v33_stall_recovery_exact_resume_base"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load exact-resume control plane: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_base = _load_base()
_base.__doc__ = __doc__

RUN_NAME = "full-run-revival-v33-stability-recovery-model-init"
SOURCE_RUN_ID = "4f27a184-1ac1-4171-a231-eb0029b05bc9"
SOURCE_ENVIRONMENT_STEPS = 30_089
SOURCE_TOTAL_ENVIRONMENT_STEPS = 100_000
TARGET_ENVIRONMENT_STEPS = 100_000
ADDITIONAL_ENVIRONMENT_STEPS = TARGET_ENVIRONMENT_STEPS - SOURCE_ENVIRONMENT_STEPS
SOURCE_CHECKPOINT_ID = "90427f20-a335-41f9-afba-5a16381cf183"
SOURCE_MANIFEST_SHA256 = "31746a6598ff6500dbd80128e457ddab94728011ba29ccc9492ebe5a2ca6408f"
SOURCE_METADATA_SHA256 = "611fb2d530ef8b9ea8ac3c1170149cf19400a34c062359b1f0862ae81af9ab17"
SOURCE_POLICY_VERSION = 486
SOURCE_LEARNER_UPDATES = 486
SOURCE_CHECKPOINT_RELATIVE = Path(
    "checkpoints/full-run-revival-v33-stability-recovery-model-init"
    f"/run-{SOURCE_RUN_ID}/periodic-step-{SOURCE_ENVIRONMENT_STEPS:09d}"
)
ACTIVE_ARTIFACT_ROOT = Path("/mnt/e/game/project/sts2_mcp_artifacts/runtime")
CONFIG_FILE = "full_run_revival_v33_stability_recovery_model_init.toml"

SCHEMA_VERSION = "sts2-v33-stall-recovery-exact-resume-preflight-v1"
STATE_SCHEMA_VERSION = "sts2-v33-stall-recovery-exact-resume-state-v1"
SUPERVISED_SCHEMA_VERSION = "sts2-v33-stall-recovery-exact-resume-supervised-launch-v1"
SUPERVISED_STATE_SCHEMA_VERSION = "sts2-v33-stall-recovery-exact-resume-supervised-state-v1"
LEARNER_STALL_TIMEOUT_SECONDS = 5.0 * 60.0

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
    """Re-enter this adapter so recovery pins survive detachment."""

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
validate_fixed_resume_command = _base.validate_fixed_resume_command
build_resume_trainer_command = _base.build_resume_trainer_command
_validate_checkpoint_summary = _base._validate_checkpoint_summary
_verify_source_checkpoint = _base._verify_source_checkpoint
run_resume_preflight = _base.run_resume_preflight
resume = _base.resume
_core = _base._core


def main(argv: Sequence[str] | None = None) -> int:
    return cast(int, _base.main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
