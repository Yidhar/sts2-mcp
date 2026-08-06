#!/usr/bin/env python3
"""Recover v34 exactly after its optional macro diagnostic terminated at 50k.

The immutable source is the last atomic checkpoint published by the failed
v34 segment at environment step 40,468. Model, optimizer, actor network, both
replay stores, rollout queue, stochastic state, schedule clocks and every
counter are restored exactly. The trainer still targets the original absolute
100,000-step horizon and repeats the not-yet-committed 50k evaluation under the
repaired diagnostic boundary. This launcher cannot fresh-start or perform
model initialization.
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast


def _load_base() -> Any:
    path = Path(__file__).resolve().with_name("launch_v29_exact_continuation_350k.py")
    module_name = "_sts2_v34_evaluation_recovery_exact_resume_base"
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

RUN_NAME = "full-run-revival-v34-transaction-competence-model-init"
STATE_NAME = "full-run-revival-v34-evaluation-recovery-exact-resume"
SOURCE_RUN_ID = "6e189527-bc83-4d8a-bde2-c921ff21080f"
SOURCE_ENVIRONMENT_STEPS = 40_468
SOURCE_TOTAL_ENVIRONMENT_STEPS = 100_000
TARGET_ENVIRONMENT_STEPS = 100_000
ADDITIONAL_ENVIRONMENT_STEPS = TARGET_ENVIRONMENT_STEPS - SOURCE_ENVIRONMENT_STEPS
SOURCE_CHECKPOINT_ID = "f29ab823-7808-4562-898c-aed1bc3e6efb"
SOURCE_MANIFEST_SHA256 = "bf65c9a2a10e71a001942de6dfb73ef134ab56003f8a843a4b5fb94ef1920911"
SOURCE_METADATA_SHA256 = "d3328526bf318a0824cc99800b0fa4b46f1ef703abc0862d5e245e9559cdd552"
SOURCE_POLICY_VERSION = 654
SOURCE_LEARNER_UPDATES = 654
SOURCE_CHECKPOINT_RELATIVE = Path(
    "checkpoints/full-run-revival-v34-transaction-competence-model-init"
    f"/run-{SOURCE_RUN_ID}/periodic-step-{SOURCE_ENVIRONMENT_STEPS:09d}"
)
ACTIVE_ARTIFACT_ROOT = Path("/mnt/e/game/project/sts2_mcp_artifacts/runtime")
CONFIG_FILE = "full_run_revival_v34_transaction_competence_model_init.toml"

SCHEMA_VERSION = "sts2-v34-evaluation-recovery-exact-resume-preflight-v1"
STATE_SCHEMA_VERSION = "sts2-v34-evaluation-recovery-exact-resume-state-v1"
SUPERVISED_SCHEMA_VERSION = "sts2-v34-evaluation-recovery-exact-resume-supervised-launch-v1"
SUPERVISED_STATE_SCHEMA_VERSION = "sts2-v34-evaluation-recovery-exact-resume-supervised-state-v1"
LEARNER_STALL_TIMEOUT_SECONDS = 5.0 * 60.0

_OVERRIDES = {
    "RUN_NAME": RUN_NAME,
    "STATE_NAME": STATE_NAME,
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
    """Re-enter this adapter so the fixed recovery identity survives detachment."""

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
