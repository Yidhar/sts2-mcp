#!/usr/bin/env python3
"""Exact-resume recovery launcher for the v29 170119 checkpoint.

This is intentionally a thin adapter over the audited v29 continuation
control plane.  It changes only the immutable resume source to the latest
atomic checkpoint after a learner stall; model, config, runtime and target
horizon remain unchanged.
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

_base = None
RECOVERY_RUN_NAME = (
    "full-run-revival-v29-failure-credit-v4-exact-continuation-350k-"
    "recovery-170119"
)
RECOVERY_SUCCESSOR_LOG_NAME = (
    "full-run-revival-v29-failure-credit-v4-exact-continuation-350k"
)
RECOVERY_SOURCE_RUN_ID = "188abedf-5ec4-4521-84f7-af01ab1fcd66"
RECOVERY_SOURCE_ENVIRONMENT_STEPS = 170_119
RECOVERY_SOURCE_CHECKPOINT_ID = "d7a8113e-087d-43c4-b1af-e9253e66f09a"
RECOVERY_SOURCE_POLICY_VERSION = 2707
RECOVERY_SOURCE_LEARNER_UPDATES = 2707
RECOVERY_LEARNER_STALL_TIMEOUT_SECONDS = 1800.0


def _build_supervisor_command(paths: object, *, manifest_path: Path) -> tuple[str, ...]:
    """Re-enter this recovery adapter, not the old 100k adapter."""

    return (
        str(paths.venv_python),  # type: ignore[attr-defined]
        str(Path(__file__).resolve()),
        "supervise",
        "--manifest",
        str(manifest_path),
    )


def _recovery_successor_log_root(paths: object) -> Path:
    """Keep durable supervisor state separate from the trainer's run root."""

    return (
        paths.artifact_root  # type: ignore[attr-defined]
        / "runs"
        / RECOVERY_SUCCESSOR_LOG_NAME
    ).resolve(strict=False)


def _validate_recovery_checkpoint_summary(
    summary: Mapping[str, Any], *, paths: Any
) -> dict[str, Any]:
    """Validate a successor checkpoint whose config horizon is already 350k."""

    expected_root = _base.source_checkpoint_path(paths)
    actual_root = Path(str(summary.get("root") or "")).resolve(strict=False)
    if actual_root != expected_root:
        raise _base.LaunchError(
            f"recovery checkpoint preflight returned the wrong root: {actual_root}"
        )
    expected = {
        "checkpoint_id": _base.SOURCE_CHECKPOINT_ID,
        "manifest_sha256": _base.SOURCE_MANIFEST_SHA256,
        "metadata_sha256": _base.SOURCE_METADATA_SHA256,
        "experiment_run_id": _base.SOURCE_RUN_ID,
        "environment_steps": _base.SOURCE_ENVIRONMENT_STEPS,
        "policy_version": _base.SOURCE_POLICY_VERSION,
        "learner_updates": _base.SOURCE_LEARNER_UPDATES,
        # Successor checkpoints retain the continuation config's absolute
        # horizon; this is distinct from the restored step counter above.
        "source_total_environment_steps": _base.TARGET_ENVIRONMENT_STEPS,
        "target_total_environment_steps": _base.TARGET_ENVIRONMENT_STEPS,
        "checkpoint_format": "sts2-recurrent-vtrace-checkpoint-v5",
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            raise _base.LaunchError(
                f"recovery checkpoint {key} mismatch: "
                f"expected={value!r}, actual={summary.get(key)!r}"
            )
    files = summary.get("manifest_files")
    if not isinstance(files, list) or not files:
        raise _base.LaunchError("recovery checkpoint has no manifest file identities")
    return dict(summary)


def _configure_recovery() -> None:
    """Bind the shared adapter only when this recovery CLI is executed."""

    global _base
    if _base is None:
        _base = importlib.import_module("launch_v29_exact_continuation_350k")

    # The run lineage and target remain the same.  Only this exact checkpoint
    # is a permitted recovery source for the stalled process.
    # Use a new supervisor identity so the durable terminal manifest of the
    # previous 120256 recovery cannot be mistaken for this launch.
    _base.RUN_NAME = RECOVERY_RUN_NAME
    _base.SOURCE_RUN_ID = RECOVERY_SOURCE_RUN_ID
    _base.SOURCE_ENVIRONMENT_STEPS = RECOVERY_SOURCE_ENVIRONMENT_STEPS
    _base.ADDITIONAL_ENVIRONMENT_STEPS = _base.TARGET_ENVIRONMENT_STEPS - _base.SOURCE_ENVIRONMENT_STEPS
    _base.SOURCE_CHECKPOINT_ID = RECOVERY_SOURCE_CHECKPOINT_ID
    _base.SOURCE_MANIFEST_SHA256 = "48d4163a1fbae3048c18a7083499213de37bcca2708984b0765ed8b5cca7e7ba"
    _base.SOURCE_METADATA_SHA256 = "254129733d16dc31211fd00cbb801fa4755ee837dfb250765f9e5d82e988e141"
    _base.SOURCE_POLICY_VERSION = RECOVERY_SOURCE_POLICY_VERSION
    _base.SOURCE_LEARNER_UPDATES = RECOVERY_SOURCE_LEARNER_UPDATES
    _base.SOURCE_CHECKPOINT_RELATIVE = Path(
        "checkpoints/full-run-revival-v29-failure-credit-v4-exact-continuation-350k"
        f"/run-{_base.SOURCE_RUN_ID}/periodic-step-"
        f"{_base.SOURCE_ENVIRONMENT_STEPS:09d}"
    )
    _base._validate_checkpoint_summary = _validate_recovery_checkpoint_summary
    _base.build_supervisor_command = _build_supervisor_command
    _base._configure_supervisor_core()
    _base._core._successor_log_root = _recovery_successor_log_root


def main(argv: Sequence[str] | None = None) -> int:
    _configure_recovery()
    return _base.main(argv)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
