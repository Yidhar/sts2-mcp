#!/usr/bin/env python3
"""Launch the reviewed v44 budget-40 identification lineage.

The immutable model source is v43's healthy validation gate at local
environment step 100,212 (guard-blessed immediately before the drvfs
checkpoint I/O crash). V44 ratchets the revival budget 48 -> 40, the first
rung where the reward-v7 unit price (0.0100) sits strictly above the
per-floor progress value — v43 measured that 48 is the exact break-even
knife edge and HP conditioning never separated there. Exploration clocks
inherit v43's mid-decay state; liveness calibration replays. Reward-v7
semantics are unchanged but the curriculum differs, so this is explicit
model initialization, never exact resume. Full source payload validation
runs in preflight; no prior attestation is reused.
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
    module_name = "_sts2_v45_deck_macro_bootstrap_model_init_base"
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

RUN_NAME = "full-run-revival-v45-deck-macro-bootstrap-model-init"
SOURCE_RUN_ID = "ade6ab3a-2eda-4b5b-8fe4-3710858f9a65"
SOURCE_ENVIRONMENT_STEPS = 100_478
SOURCE_TOTAL_ENVIRONMENT_STEPS = 250_000
TARGET_ENVIRONMENT_STEPS = 250_000
ADDITIONAL_ENVIRONMENT_STEPS = TARGET_ENVIRONMENT_STEPS
SOURCE_CHECKPOINT_ID = "d44625a7-2e50-4449-9f5c-4f7b62f3c5f3"
SOURCE_MANIFEST_SHA256 = (
    "77dc76e3009c0fb084c1dafd8f23acfbca3e4473d8c5d14126de62fc35e631b8"
)
SOURCE_METADATA_SHA256 = (
    "f87f4368749a36db97733fc3f208a822375a79271aaa37510ff9d0cede8049cb"
)
SOURCE_POLICY_VERSION = 1_646
SOURCE_LEARNER_UPDATES = 1_646
SOURCE_CHECKPOINT_RELATIVE = Path(
    "checkpoints/full-run-revival-v44-budget40-identification-model-init"
    f"/run-{SOURCE_RUN_ID}/healthy-validation-step-{SOURCE_ENVIRONMENT_STEPS:09d}"
)
ACTIVE_ARTIFACT_ROOT = Path("/mnt/e/game/project/sts2_mcp_artifacts/runtime")
CONFIG_FILE = "full_run_revival_v45_deck_macro_bootstrap_model_init.toml"
LINEAGE_LABEL = "v45"

SCHEMA_VERSION = "sts2-v45-deck-macro-bootstrap-model-init-preflight-v1"
STATE_SCHEMA_VERSION = "sts2-v45-deck-macro-bootstrap-model-init-state-v1"
SUPERVISED_SCHEMA_VERSION = (
    "sts2-v45-deck-macro-bootstrap-model-init-supervised-launch-v1"
)
SUPERVISED_STATE_SCHEMA_VERSION = (
    "sts2-v45-deck-macro-bootstrap-model-init-supervised-state-v1"
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
    """Re-enter this adapter so the immutable v44 source pins remain bound."""

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
