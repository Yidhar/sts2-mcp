#!/usr/bin/env python3
"""Exact-resume v31 after the noisy 5k evaluation guard stop.

The source is the atomic guard-stop checkpoint at environment step 5,564.
The learning lineage, optimizer, replay, RNG, counters, unlimited engine
revival curriculum and absolute 250k horizon are restored exactly.  Only the
observation-only evaluation guard schedule changes: it is calibrated against
the frozen Gate-0 result instead of treating the initializer's existing
liveness rate as a new collapse.
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

_base = None

RECOVERY_RUN_NAME = "full-run-revival-v31-failure-credit-capacity-exact-resume-5564"
SUCCESSOR_LOG_NAME = "full-run-revival-v31-failure-credit-capacity-model-init"
SOURCE_RUN_ID = "83d6c8d1-e321-4d8d-bb03-d1ee7dbfb461"
SOURCE_ENVIRONMENT_STEPS = 5_564
SOURCE_CHECKPOINT_ID = "39bac05c-2bfd-431a-93bb-747d3ece2391"
SOURCE_MANIFEST_SHA256 = "8833daa8f004fa53f8c694ba29c2d6a037f6ad9029655cd7ced81e71c623f6c7"
SOURCE_METADATA_SHA256 = "83380e126b0064a251bd5d3a414dfc34a4e118786a0e66aa9395a38ed7d67faa"
SOURCE_POLICY_VERSION = 90
SOURCE_LEARNER_UPDATES = 90
TARGET_ENVIRONMENT_STEPS = 250_000
CONFIG_FILE = "full_run_revival_v31_failure_credit_capacity_model_init.toml"
SOURCE_CHECKPOINT_RELATIVE = Path(
    "checkpoints/full-run-revival-v31-failure-credit-capacity-model-init"
    f"/run-{SOURCE_RUN_ID}/guard-stop-step-{SOURCE_ENVIRONMENT_STEPS:09d}"
)


def _build_supervisor_command(
    paths: object,
    *,
    manifest_path: Path,
) -> tuple[str, ...]:
    """Re-enter this pinned recovery adapter in the detached supervisor."""

    return (
        str(paths.venv_python),  # type: ignore[attr-defined]
        str(Path(__file__).resolve()),
        "supervise",
        "--manifest",
        str(manifest_path),
    )


def _successor_log_root(paths: object) -> Path:
    """Bind monitoring to the trainer's unchanged v31 log namespace."""

    return (
        paths.artifact_root  # type: ignore[attr-defined]
        / "runs"
        / SUCCESSOR_LOG_NAME
    ).resolve(strict=False)


def _validate_checkpoint_summary(
    summary: Mapping[str, Any],
    *,
    paths: Any,
) -> dict[str, Any]:
    expected_root = _base.source_checkpoint_path(paths)
    actual_root = Path(str(summary.get("root") or "")).resolve(strict=False)
    if actual_root != expected_root:
        raise _base.LaunchError(
            "v31 guard recovery preflight returned the wrong root: " f"expected={expected_root}, actual={actual_root}"
        )
    expected = {
        "checkpoint_id": SOURCE_CHECKPOINT_ID,
        "manifest_sha256": SOURCE_MANIFEST_SHA256,
        "metadata_sha256": SOURCE_METADATA_SHA256,
        "experiment_run_id": SOURCE_RUN_ID,
        "environment_steps": SOURCE_ENVIRONMENT_STEPS,
        "policy_version": SOURCE_POLICY_VERSION,
        "learner_updates": SOURCE_LEARNER_UPDATES,
        "source_total_environment_steps": TARGET_ENVIRONMENT_STEPS,
        "target_total_environment_steps": TARGET_ENVIRONMENT_STEPS,
        "checkpoint_format": "sts2-recurrent-vtrace-checkpoint-v5",
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            raise _base.LaunchError(
                f"v31 guard recovery checkpoint {key} mismatch: " f"expected={value!r}, actual={summary.get(key)!r}"
            )
    files = summary.get("manifest_files")
    if not isinstance(files, list) or not files:
        raise _base.LaunchError("v31 guard recovery checkpoint has no manifest file identities")
    return dict(summary)


def _configure() -> None:
    global _base
    if _base is None:
        _base = importlib.import_module("launch_v29_exact_continuation_350k")

    _base.RUN_NAME = RECOVERY_RUN_NAME
    _base.SOURCE_RUN_ID = SOURCE_RUN_ID
    _base.SOURCE_ENVIRONMENT_STEPS = SOURCE_ENVIRONMENT_STEPS
    _base.SOURCE_TOTAL_ENVIRONMENT_STEPS = TARGET_ENVIRONMENT_STEPS
    _base.TARGET_ENVIRONMENT_STEPS = TARGET_ENVIRONMENT_STEPS
    _base.ADDITIONAL_ENVIRONMENT_STEPS = TARGET_ENVIRONMENT_STEPS - SOURCE_ENVIRONMENT_STEPS
    _base.SOURCE_CHECKPOINT_ID = SOURCE_CHECKPOINT_ID
    _base.SOURCE_MANIFEST_SHA256 = SOURCE_MANIFEST_SHA256
    _base.SOURCE_METADATA_SHA256 = SOURCE_METADATA_SHA256
    _base.SOURCE_POLICY_VERSION = SOURCE_POLICY_VERSION
    _base.SOURCE_LEARNER_UPDATES = SOURCE_LEARNER_UPDATES
    _base.SOURCE_CHECKPOINT_RELATIVE = SOURCE_CHECKPOINT_RELATIVE
    _base.CONFIG_FILE = CONFIG_FILE
    _base.SCHEMA_VERSION = "sts2-v31-guard-recovery-preflight-v1"
    _base.STATE_SCHEMA_VERSION = "sts2-v31-guard-recovery-state-v1"
    _base.SUPERVISED_SCHEMA_VERSION = "sts2-v31-guard-recovery-supervised-launch-v1"
    _base.SUPERVISED_STATE_SCHEMA_VERSION = "sts2-v31-guard-recovery-supervised-state-v1"
    _base._validate_checkpoint_summary = _validate_checkpoint_summary
    _base.build_supervisor_command = _build_supervisor_command
    _base._configure_supervisor_core()
    _base._core._successor_log_root = _successor_log_root


def main(argv: Sequence[str] | None = None) -> int:
    _configure()
    return _base.main(argv)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
