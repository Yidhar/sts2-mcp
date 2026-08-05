#!/usr/bin/env python3
"""Run an isolated 512-step exact-resume smoke on ROCm 7.2.1 + ROCDXG.

The source is the atomically published v33 checkpoint at environment step
40,938. Model, optimizer, episodic replay, failure-credit replay, rollout
queue, RNG, schedules and counters are restored exactly. The reviewed runtime
transition changes only execution provenance, observation schedules, output
paths and the absolute horizon. The hermetic process environment requires
``HSA_ENABLE_DXG_DETECTION=1`` and never inherits the calling shell.
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast


def _load_base() -> Any:
    path = Path(__file__).resolve().with_name("launch_v29_exact_continuation_350k.py")
    module_name = "_sts2_v33_rocdxg_exact_resume_smoke_base"
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

RUN_NAME = "full-run-revival-v33-rocdxg-exact-resume-smoke"
STATE_NAME = RUN_NAME
SOURCE_RUN_ID = "53561069-b345-421c-a145-29d6ccb93448"
SOURCE_ENVIRONMENT_STEPS = 40_938
SOURCE_TOTAL_ENVIRONMENT_STEPS = 100_000
TARGET_ENVIRONMENT_STEPS = 41_450
ADDITIONAL_ENVIRONMENT_STEPS = TARGET_ENVIRONMENT_STEPS - SOURCE_ENVIRONMENT_STEPS
SOURCE_CHECKPOINT_ID = "22e20310-9b6e-4adb-87a2-158efc38a984"
SOURCE_MANIFEST_SHA256 = "86f788e3fa78eedbaf3fc5729cf0a753ffc9d921fbb9169eb6c7803d14627487"
SOURCE_METADATA_SHA256 = "dc7dff66a26c719def7731be80a30d58f3e84b098e950299551dbcd57aed2dee"
SOURCE_POLICY_VERSION = 662
SOURCE_LEARNER_UPDATES = 662
SOURCE_CHECKPOINT_RELATIVE = Path(
    "checkpoints/full-run-revival-v33-stability-recovery-model-init"
    f"/run-{SOURCE_RUN_ID}/periodic-step-{SOURCE_ENVIRONMENT_STEPS:09d}"
)
ACTIVE_ARTIFACT_ROOT = Path("/mnt/e/game/project/sts2_mcp_artifacts/runtime")
CONFIG_FILE = "full_run_revival_v33_rocdxg_exact_resume_smoke.toml"

SCHEMA_VERSION = "sts2-v33-rocdxg-exact-resume-smoke-preflight-v1"
STATE_SCHEMA_VERSION = "sts2-v33-rocdxg-exact-resume-smoke-state-v1"
SUPERVISED_SCHEMA_VERSION = "sts2-v33-rocdxg-exact-resume-smoke-supervised-launch-v1"
SUPERVISED_STATE_SCHEMA_VERSION = "sts2-v33-rocdxg-exact-resume-smoke-supervised-state-v1"
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
    """Re-enter this adapter so smoke pins survive process detachment."""

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
