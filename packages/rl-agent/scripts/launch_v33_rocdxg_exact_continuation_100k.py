#!/usr/bin/env python3
"""Exactly continue the ROCDXG-validated v33 lineage to 100k steps.

The immutable source is the atomically published and subsequently validated
v33 smoke successor at environment step 41,450. Model, optimizer, episodic
replay, failure-credit replay, rollout queue, RNG, schedules and counters are
restored exactly. This launcher cannot fresh-start or perform model
initialization. Its trainer process receives the complete reviewed hermetic
environment, including ``HSA_ENABLE_DXG_DETECTION=1``.
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast


def _load_base() -> Any:
    path = Path(__file__).resolve().with_name("launch_v29_exact_continuation_350k.py")
    module_name = "_sts2_v33_rocdxg_exact_continuation_100k_base"
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

RUN_NAME = "full-run-revival-v33-rocdxg-exact-continuation-100k"
STATE_NAME = RUN_NAME
SOURCE_RUN_ID = "9f08f9c4-6786-4678-a724-7a1bdcf13cba"
SOURCE_ENVIRONMENT_STEPS = 41_450
SOURCE_TOTAL_ENVIRONMENT_STEPS = 41_450
TARGET_ENVIRONMENT_STEPS = 100_000
ADDITIONAL_ENVIRONMENT_STEPS = TARGET_ENVIRONMENT_STEPS - SOURCE_ENVIRONMENT_STEPS
SOURCE_CHECKPOINT_ID = "9154f4d0-820b-4bcb-a27e-813cf83f15f5"
SOURCE_MANIFEST_SHA256 = "eb8d26327c9ece0f8fec312da2dba90fa162bdf51347a3cb51f1dec817001cd0"
SOURCE_METADATA_SHA256 = "0a70224448be51e5fb2d36c3f62d3e330d14e4588d298912b7d78bf09a61d66a"
SOURCE_POLICY_VERSION = 670
SOURCE_LEARNER_UPDATES = 670
SOURCE_CHECKPOINT_RELATIVE = Path(
    "checkpoints/full-run-revival-v33-rocdxg-exact-resume-smoke"
    f"/run-{SOURCE_RUN_ID}/final-step-{SOURCE_ENVIRONMENT_STEPS:09d}"
)
ACTIVE_ARTIFACT_ROOT = Path("/mnt/e/game/project/sts2_mcp_artifacts/runtime")
CONFIG_FILE = "full_run_revival_v33_rocdxg_exact_continuation_100k.toml"

SCHEMA_VERSION = "sts2-v33-rocdxg-exact-continuation-100k-preflight-v1"
STATE_SCHEMA_VERSION = "sts2-v33-rocdxg-exact-continuation-100k-state-v1"
SUPERVISED_SCHEMA_VERSION = "sts2-v33-rocdxg-exact-continuation-100k-supervised-launch-v1"
SUPERVISED_STATE_SCHEMA_VERSION = "sts2-v33-rocdxg-exact-continuation-100k-supervised-state-v1"
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
    """Re-enter this adapter so continuation pins survive detachment."""

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
