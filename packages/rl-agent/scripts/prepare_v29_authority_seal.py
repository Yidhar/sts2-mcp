#!/usr/bin/env python3
"""Prepare and verify the non-launching two-phase v29 authority seal.

This helper deliberately has no command that can start a trainer.  Its three
commands are evidence-only:

``dry-run``
    Validate a clean evidence-generation commit A and print the deterministic
    seal plan without writing files.

``prepare``
    Perform the same validation and atomically write a staged shadow contract
    plus a launcher-constant seal plan.  Both outputs must be outside the Git
    checkout so commit A remains clean.

``verify``
    At a clean seal commit B, reopen every report and the staged contract,
    prove A is an ancestor, require the exact reviewed A..B path set, and prove
    that every validated production file is byte-identical to commit A.

The helper is stdlib-only.  Runtime identity can be supplied as reviewed JSON
or collected by invoking the pinned ROCm Python solely for the fixed
``current_rocm_runtime_identity`` query.  Arbitrary subprocess commands and
trainer arguments are intentionally not accepted.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NoReturn, cast

PLAN_SCHEMA = "sts2-v29-authority-seal-plan-v1"
SHADOW_CONTRACT_SCHEMA = "sts2-shadow-validation-contract-v3"
READINESS_EVIDENCE_SCHEMA = "sts2-runtime-readiness-evidence-v1"
SOURCE_AUTHORITY_SCHEMA = "sts2-two-phase-source-authority-v1"

SHADOW_CONTRACT_NAME = "v29-failure-credit-v4-bootstrap"
FROZEN_CONTRACT_NAME = "v28-mature-refinement-100k"
FIXED_CONFIG_FINGERPRINT_SHA256 = "78a258fa9a4b5b7fb418b013b51ebc603d15ab249191fa21c4986e3e4f5919e7"
FIXED_CHECKPOINT_ID = "f670deda-97d5-46f2-be07-d69a5842eeec"
FIXED_CHECKPOINT_STEP = 100_000
FIXED_POLICY_VERSION = 1_569
FIXED_MANIFEST_SHA256 = "cfd4d23bf20e064e7a1b21939e94ff01a9f5bf48ab7d70690c66d71f9f2e7af3"
FIXED_METADATA_SHA256 = "e1933292cc0fbca95cd3b80e073b0b2cda40c1ea29731b59a4bfa21f4f8b5127"

SHADOW_CONTRACT_RELATIVE = "contracts/shadow-validation/v29-failure-credit-v4-bootstrap.json"
LAUNCHER_RELATIVE = "packages/rl-agent/scripts/launch_v29_failure_credit_v4.py"
REVIEWED_SEAL_PATHS = (SHADOW_CONTRACT_RELATIVE, LAUNCHER_RELATIVE)
LAUNCHER_SEAL_CONSTANT_NAMES = frozenset(
    {
        "ACTOR_EVIDENCE_SHADOW_SHA256",
        "EVIDENCE_SHADOW_SHA256",
        "RUNTIME_READINESS_REPORT_SHA256",
        "SEMANTICS_SHADOW_SHA256",
        "SHADOW_AUTHORITY_SHA256",
        "SHADOW_CONTRACT_SHA256",
        "SHADOW_VALIDATED_CODE_COUNT",
        "SHADOW_VALIDATED_CODE_MAPPING_SHA256",
    }
)

DEFAULT_CONFIG_RELATIVE = "packages/rl-agent/config/experiments/full_run_revival_v29_failure_credit_v4_model_init.toml"
DEFAULT_CHECKPOINT_RELATIVE = (
    "checkpoints/full-run-revival-v28-mature-refinement-model-init/"
    "run-071a43f5-120a-485f-a6dc-a67a55a1efc2/periodic-step-000100000"
)
DEFAULT_REPORT_RELATIVES = {
    "semantics_historical": "reports/failure-credit-v4/semantic-shadow-historical-v3.json",
    "evidence_live": "reports/failure-credit-v4/evidence-shadow-v28-live.json",
    "evidence_actor": "reports/failure-credit-v4/evidence-shadow-actor-contract-v2.json",
    "runtime_readiness": "reports/failure-credit-v4/liveness-head-stress-v29.json",
}

# The old shadow contract supplies the already reviewed semantic/training
# closure.  These files are mandatory additions and cannot be omitted by an
# old base contract.  Seal files are intentionally excluded: B is allowed to
# change exactly those two paths, while all entries below must remain A bytes.
REQUIRED_PRODUCTION_CLOSURE = frozenset(
    {
        "packages/rl-agent/scripts/prepare_v29_authority_seal.py",
        "packages/rl-agent/scripts/supervised_trainer_bootstrap.py",
        "packages/rl-agent/scripts/validate_liveness_head_stress.py",
        "packages/rl-agent/sts2_rl/checkpoints/frozen.py",
        "packages/rl-agent/sts2_rl/train.py",
        "packages/rl-agent/sts2_rl/training/checkpointing.py",
        "packages/rl-agent/sts2_rl/training/checkpoint_evaluation.py",
        "packages/rl-agent/sts2_rl/training/factory.py",
        "packages/rl-agent/sts2_rl/training/failure_credit/actor_eligibility.py",
        "packages/rl-agent/sts2_rl/training/launch_contract.py",
        "packages/rl-agent/sts2_rl/training/runtime.py",
    }
)

SEMANTICS_VERSION = "sts2-semantics-shadow-report-v2"
EVIDENCE_VERSION = "sts2-failure-evidence-shadow-report-v2"
ACTOR_VERSION = "sts2-failure-actor-evidence-shadow-report-v3"
READINESS_VERSION = "sts2-liveness-head-active-shape-stress-v3"
REPORT_GENERATION_SOURCE_VERSION = "sts2-formal-report-generation-source-v1"
REPORT_VALIDATORS = {
    "semantics_historical": "packages/rl-agent/scripts/validate_semantics_shadow.py",
    "evidence_live": "packages/rl-agent/scripts/validate_failure_evidence_shadow.py",
    "evidence_actor": "packages/rl-agent/scripts/validate_failure_actor_evidence_shadow.py",
    "runtime_readiness": "packages/rl-agent/scripts/validate_liveness_head_stress.py",
}

SEMANTICS_GATES = frozenset(
    {
        "maximum_unknown_transitions",
        "minimum_rich_records",
        "no_contract_errors",
        "observed_unknown_transitions",
        "required_minimum_rich_records",
        "unknown_transition_budget",
    }
)
EVIDENCE_GATES = frozenset(
    {
        "completion_staging_is_bounded",
        "minimum_decisions",
        "no_replay_or_learner_updates",
        "no_unjustified_completion_prefer",
        "source_manifest_unchanged",
        "source_metadata_unchanged",
        "zero_semantic_censored_transitions",
    }
)
ACTOR_GATES = frozenset(
    {
        "abandoned_cycle_has_zero_direct_or_cycle_blame",
        "all_contexts_bounded",
        "censored_has_zero_learning_targets",
        "direct_avoid_nonzero",
        "forced_only_has_zero_actor_blame",
        "learner_mask_dry_run_passed",
        "multi_edge_cycle_nonzero",
        "no_generic_prefer_target",
        "no_last_action_fallback",
        "unresolved_stall_nonzero",
    }
)
READINESS_GATES = frozenset(
    {
        "bounded_tbptt_segments",
        "calibration_and_mature_phase_contract",
        "finite_gradients",
        "finite_losses",
        "formal_shape_contract",
        "matched_outcome_contrast_path",
        "no_optimizer_step",
        "nonzero_expected_gradients",
        "ordinary_rows_not_globally_padded",
        "production_manifest_work_accounting",
        "stress_forward_uses_active_timestep_batch",
    }
)
READINESS_TOP_LEVEL = frozenset(
    {
        "active_shape_probe",
        "authority",
        "config",
        "durations_seconds",
        "gates",
        "generation_source",
        "gpu_peak",
        "initialization",
        "one_shot",
        "optimizer_steps",
        "phases",
        "runtime_identity",
        "sdpa",
        "shape",
        "status",
        "training_authority",
        "version",
    }
)
READINESS_SHAPE = {
    "candidate_decisions_per_context": 65_536,
    "contexts": 2,
    "expected_tbptt_segments": 32,
    "legal_candidates_per_step": 256,
    "ordinary_probe_candidates": 3,
    "replayed_candidate_decisions": 131_072,
    "replayed_steps": 512,
    "steps_per_context": 256,
    "tbptt_segments_per_context": 16,
}
RUNTIME_IDENTITY_FIELDS = frozenset(
    {
        "accelerator",
        "cuda_available",
        "cuda_device_count",
        "environment",
        "platform",
        "python",
        "resolved_device",
        "torch_cuda_version",
        "torch_git_version",
        "torch_hip_version",
        "torch_version",
    }
)


class SealError(RuntimeError):
    """The evidence-only v29 seal contract could not be proven."""


def _fail(message: str) -> NoReturn:
    raise SealError(message)


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    try:
        return _sha256_bytes(path.read_bytes())
    except OSError as exc:
        raise SealError(f"required file is unreadable: {path}") from exc


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SealError("seal data is not canonical JSON") from exc


def _canonical_json_sha256(value: object) -> str:
    return _sha256_bytes(_canonical_json_bytes(value))


def _pretty_json_bytes(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SealError("seal data is not JSON serializable") from exc


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except OSError as exc:
        raise SealError(f"{label} is unreadable: {path}") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SealError(f"{label} is not valid UTF-8 JSON: {path}") from exc
    if not isinstance(value, dict):
        raise SealError(f"{label} must be a JSON object")
    return value


def _run_git(root: Path, *args: str, check: bool = True) -> str:
    try:
        completed = subprocess.run(
            ("git", *args),
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30.0,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SealError(f"git {' '.join(args)} failed") from exc
    if check and completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise SealError(f"git {' '.join(args)} failed: {detail}")
    return completed.stdout.strip() if completed.returncode == 0 else ""


def _clean_source(root: Path, *, expected_commit: str | None) -> dict[str, Any]:
    root = root.expanduser().resolve()
    top = Path(_run_git(root, "rev-parse", "--show-toplevel")).resolve()
    if top != root:
        raise SealError("checkout root is not the Git toplevel")
    object_format = _run_git(root, "rev-parse", "--show-object-format")
    if object_format not in {"sha1", "sha256"}:
        raise SealError("unsupported Git object format")
    oid_length = 40 if object_format == "sha1" else 64
    commit = _run_git(root, "rev-parse", "HEAD")
    tree = _run_git(root, "rev-parse", "HEAD^{tree}")
    if (
        re.fullmatch(rf"[0-9a-f]{{{oid_length}}}", commit) is None
        or re.fullmatch(rf"[0-9a-f]{{{oid_length}}}", tree) is None
    ):
        raise SealError("Git returned malformed full object IDs")
    if expected_commit is not None:
        if re.fullmatch(rf"[0-9a-f]{{{oid_length}}}", expected_commit) is None:
            raise SealError("--generation-commit must be a full object ID")
        if expected_commit != commit:
            raise SealError(f"generation commit must equal clean HEAD: expected={expected_commit} actual={commit}")
    dirty = _run_git(root, "status", "--porcelain=v1", "--untracked-files=normal")
    if dirty:
        raise SealError("authority sealing requires a clean checkout")
    return {
        "checkout_root": os.fspath(root),
        "git_object_format": object_format,
        "implementation_commit": commit,
        "implementation_tree": tree,
        "worktree_clean": True,
    }


def _normalized_relative(value: str, *, label: str) -> str:
    path = Path(value)
    normalized = path.as_posix()
    if not value or path.is_absolute() or ".." in path.parts or normalized != value:
        raise SealError(f"{label} must be a normalized repository-relative POSIX path")
    return value


def _within(path: Path, root: Path, *, label: str) -> Path:
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise SealError(f"{label} escapes its authority root") from exc
    return resolved


def _report_path(root: Path, relative: str, *, label: str) -> Path:
    _normalized_relative(relative, label=label)
    path = _within(root / relative, root, label=label)
    if not path.is_file():
        raise SealError(f"{label} is missing: {path}")
    return path


def _positive_int(value: object, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise SealError(f"{label} must be an integer >= {minimum}")
    return value


def _sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise SealError(f"{label} must be lowercase SHA-256")
    return value


def _require_report_header(report: Mapping[str, Any], *, version: str, label: str) -> None:
    if report.get("version") != version:
        raise SealError(f"{label} schema/version changed")
    if report.get("status") != "passed":
        raise SealError(f"{label} status is not passed")
    if report.get("read_only") is not True:
        raise SealError(f"{label} must be read-only")
    if report.get("training_authority") is not False:
        raise SealError(f"{label} must not grant training authority")


def _validate_report_generation_source(
    report: Mapping[str, Any],
    *,
    expected_source: Mapping[str, Any],
    root: Path,
    report_key: str,
) -> dict[str, Any]:
    value = report.get("generation_source")
    expected_fields = set(expected_source) | {
        "schema_version",
        "validator_relative_path",
        "validator_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise SealError(f"{report_key} report generation_source schema changed")
    if value.get("schema_version") != REPORT_GENERATION_SOURCE_VERSION:
        raise SealError(f"{report_key} report generation_source version changed")
    observed_source = {key: value.get(key) for key in expected_source}
    if observed_source != dict(expected_source):
        raise SealError(f"{report_key} report was not generated by clean commit A")
    validator_relative = REPORT_VALIDATORS[report_key]
    if value.get("validator_relative_path") != validator_relative:
        raise SealError(f"{report_key} report validator path changed")
    validator_sha256 = _sha256(
        value.get("validator_sha256"),
        label=f"{report_key} report validator_sha256",
    )
    if validator_sha256 != _sha256_file(root / validator_relative):
        raise SealError(f"{report_key} report validator differs from clean commit A")
    return dict(value)


def _all_true_gates(report: Mapping[str, Any], expected: frozenset[str], *, label: str) -> dict[str, bool]:
    gates = report.get("gates")
    if not isinstance(gates, Mapping) or set(gates) != set(expected):
        raise SealError(f"{label} gate schema changed")
    if any(value is not True for value in gates.values()):
        failed = sorted(key for key, value in gates.items() if value is not True)
        raise SealError(f"{label} has failed gates: {', '.join(failed)}")
    return {key: True for key in sorted(expected)}


def _validate_semantics(
    report: Mapping[str, Any],
    *,
    expected_source: Mapping[str, Any],
    root: Path,
) -> dict[str, Any]:
    _require_report_header(report, version=SEMANTICS_VERSION, label="semantics report")
    _validate_report_generation_source(
        report,
        expected_source=expected_source,
        root=root,
        report_key="semantics_historical",
    )
    gates = report.get("gates")
    if not isinstance(gates, Mapping) or set(gates) != set(SEMANTICS_GATES):
        raise SealError("semantics report gate schema changed")
    for key in ("minimum_rich_records", "no_contract_errors", "unknown_transition_budget"):
        if gates.get(key) is not True:
            raise SealError(f"semantics report gate failed: {key}")
    if gates.get("required_minimum_rich_records") != 1_000:
        raise SealError("semantics report rich-record threshold changed")
    if gates.get("observed_unknown_transitions") != 0 or gates.get("maximum_unknown_transitions") != 0:
        raise SealError("semantics report contains unknown transitions")
    counts = report.get("counts")
    errors = report.get("errors")
    if not isinstance(counts, Mapping) or not isinstance(errors, list) or errors:
        raise SealError("semantics report counts/errors are malformed")
    parsed = _positive_int(counts.get("parsed_records"), label="semantics parsed_records", minimum=500_000)
    rich = _positive_int(
        counts.get("rich_decision_snapshots"),
        label="semantics rich_decision_snapshots",
        minimum=3_986,
    )
    manifest = report.get("semantic_manifest")
    if not isinstance(manifest, Mapping) or (
        manifest.get("namespace") != "surface_registry_manifest"
        or manifest.get("schema_version") != "sts2-surface-registry-v1"
        or manifest.get("digest") != "ff7b2e40a2aeb6cd29fd5db5ea9f42998a285deadc175d7a2480de50a0a97bf8"
        or not isinstance(manifest.get("payload"), Mapping)
        or manifest["payload"].get("contract_version") != "sts2-surface-registry-v1"
    ):
        raise SealError("semantics report source manifest changed")
    return {
        "status": "passed",
        "parsed_records": parsed,
        "rich_decision_snapshots": rich,
        "unknown_transitions": 0,
        "errors": 0,
    }


def _validate_evidence(
    report: Mapping[str, Any],
    *,
    checkpoint: Path,
    expected_source: Mapping[str, Any],
    root: Path,
) -> dict[str, Any]:
    _require_report_header(report, version=EVIDENCE_VERSION, label="evidence report")
    _validate_report_generation_source(
        report,
        expected_source=expected_source,
        root=root,
        report_key="evidence_live",
    )
    _all_true_gates(report, EVIDENCE_GATES, label="evidence report")
    config = report.get("config")
    if not isinstance(config, Mapping):
        raise SealError("evidence report config is malformed")
    if (
        config.get("mode") != "shadow"
        or config.get("collector_device") != "cpu"
        or config.get("device") != "cuda"
        or _positive_int(config.get("episodes"), label="evidence config.episodes", minimum=2) < 2
        or _positive_int(config.get("maximum_steps"), label="evidence config.maximum_steps", minimum=1_000) < 1_000
    ):
        raise SealError("evidence report config changed")
    _sha256(config.get("fingerprint_sha256"), label="evidence config.fingerprint_sha256")
    source = report.get("source")
    expected_source = {
        "checkpoint_id": FIXED_CHECKPOINT_ID,
        "environment_steps": FIXED_CHECKPOINT_STEP,
        "manifest_sha256": FIXED_MANIFEST_SHA256,
        "metadata_sha256": FIXED_METADATA_SHA256,
        "policy_version": FIXED_POLICY_VERSION,
    }
    if not isinstance(source, Mapping) or any(source.get(key) != value for key, value in expected_source.items()):
        raise SealError("evidence report frozen source changed")
    raw_checkpoint = source.get("checkpoint")
    if not isinstance(raw_checkpoint, str) or Path(raw_checkpoint).resolve() != checkpoint.resolve():
        raise SealError("evidence report checkpoint path changed")
    counts = report.get("counts")
    direct = report.get("direct_policy_targets")
    unchanged = report.get("training_state_unchanged")
    if not isinstance(counts, Mapping) or not isinstance(direct, Mapping) or not isinstance(unchanged, Mapping):
        raise SealError("evidence report summaries are malformed")
    if any(value is not True for value in unchanged.values()):
        raise SealError("evidence report changed training state")
    decisions = _positive_int(counts.get("decisions"), label="evidence decisions", minimum=1_000)
    censored = _positive_int(
        counts.get("semantic_censored_transitions"),
        label="evidence semantic_censored_transitions",
    )
    prefer = _positive_int(direct.get("prefer", 0), label="evidence prefer targets")
    if censored != 0 or prefer != 0:
        raise SealError("evidence report contains censored/prefer targets")
    maximum_bytes = _positive_int(
        counts.get("maximum_episode_completion_storage_nbytes"),
        label="evidence maximum completion bytes",
    )
    if maximum_bytes > 134_217_728:
        raise SealError("evidence report completion staging is unbounded")
    return {
        "status": "passed",
        "decisions": decisions,
        "semantic_censored_transitions": 0,
        "prefer_targets": 0,
        "maximum_episode_completion_storage_nbytes": maximum_bytes,
        "completion_staging_is_bounded": True,
        "source_checkpoint_id": FIXED_CHECKPOINT_ID,
        "training_state_unchanged": True,
    }


def _validate_actor(
    report: Mapping[str, Any],
    *,
    expected_source: Mapping[str, Any],
    root: Path,
) -> dict[str, Any]:
    _require_report_header(report, version=ACTOR_VERSION, label="actor-evidence report")
    _validate_report_generation_source(
        report,
        expected_source=expected_source,
        root=root,
        report_key="evidence_actor",
    )
    _all_true_gates(report, ACTOR_GATES, label="actor-evidence report")
    if report.get("fixture_authority") != "reviewed-contract-cases-only":
        raise SealError("actor-evidence fixture authority changed")
    counts = report.get("counts")
    if not isinstance(counts, Mapping):
        raise SealError("actor-evidence counts are malformed")
    cases = _positive_int(counts.get("cases"), label="actor cases", minimum=6)
    actionable = _positive_int(counts.get("actor_actionable_records"), label="actor actionable records", minimum=4)
    risk = _positive_int(counts.get("risk_sequence_records"), label="actor risk records", minimum=5)
    direct = counts.get("direct_targets")
    if not isinstance(direct, Mapping):
        raise SealError("actor-evidence direct targets are malformed")
    prefer = _positive_int(direct.get("prefer", 0), label="actor prefer targets")
    if prefer != 0 or _positive_int(direct.get("avoid", 0), label="actor avoid targets", minimum=1) < 1:
        raise SealError("actor-evidence direct targets changed")
    return {
        "status": "passed",
        "cases": cases,
        "actor_actionable_records": actionable,
        "risk_sequence_records": risk,
        "prefer_targets": 0,
        "learner_mask_dry_run_passed": True,
    }


def _runtime_identity(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(RUNTIME_IDENTITY_FIELDS):
        raise SealError(f"{label} schema changed")
    accelerator = value.get("accelerator")
    environment = value.get("environment")
    if not isinstance(accelerator, Mapping) or set(accelerator) != {
        "device_index",
        "multi_processor_count",
        "name",
        "total_memory_bytes",
    }:
        raise SealError(f"{label}.accelerator is malformed")
    if not isinstance(environment, Mapping) or set(environment) != {
        "HIP_VISIBLE_DEVICES",
        "HSA_OVERRIDE_GFX_VERSION",
        "ROCR_VISIBLE_DEVICES",
    }:
        raise SealError(f"{label}.environment is malformed")
    if (
        value.get("cuda_available") is not True
        or _positive_int(value.get("cuda_device_count"), label=f"{label}.cuda_device_count", minimum=1) < 1
        or value.get("resolved_device") != "cuda"
        or value.get("torch_cuda_version") is not None
        or not isinstance(value.get("torch_hip_version"), str)
        or not value.get("torch_hip_version")
    ):
        raise SealError(f"{label} is not a live ROCm CUDA-compat runtime")
    for key in ("platform", "python", "torch_git_version", "torch_version"):
        if not isinstance(value.get(key), str) or not value[key]:
            raise SealError(f"{label}.{key} is malformed")
    for key in ("device_index", "multi_processor_count", "total_memory_bytes"):
        _positive_int(
            accelerator.get(key), label=f"{label}.accelerator.{key}", minimum=0 if key == "device_index" else 1
        )
    if not isinstance(accelerator.get("name"), str) or not accelerator["name"]:
        raise SealError(f"{label}.accelerator.name is malformed")
    if accelerator["device_index"] >= value["cuda_device_count"]:
        raise SealError(f"{label}.accelerator.device_index is out of range")
    if any(item is not None and not isinstance(item, str) for item in environment.values()):
        raise SealError(f"{label}.environment values are malformed")
    # A JSON round-trip removes non-standard Mapping subclasses and refuses NaN.
    try:
        return cast(
            dict[str, Any],
            json.loads(json.dumps(value, allow_nan=False, sort_keys=True)),
        )
    except (TypeError, ValueError) as exc:
        raise SealError(f"{label} is not canonical JSON") from exc


def _validate_readiness(
    report: Mapping[str, Any],
    *,
    config_path: Path,
    checkpoint: Path,
    live_runtime_identity: Mapping[str, Any],
    expected_source: Mapping[str, Any],
    root: Path,
) -> dict[str, Any]:
    if set(report) != set(READINESS_TOP_LEVEL):
        raise SealError("runtime-readiness top-level schema changed")
    if report.get("version") != READINESS_VERSION:
        raise SealError("runtime-readiness schema/version changed")
    if report.get("status") != "passed":
        raise SealError("runtime-readiness status is not passed")
    if report.get("training_authority") is not False or report.get("one_shot") is not True:
        raise SealError("runtime-readiness authority changed")
    if report.get("optimizer_steps") != 0:
        raise SealError("runtime-readiness performed optimizer steps")
    generation_source = _validate_report_generation_source(
        report,
        expected_source=expected_source,
        root=root,
        report_key="runtime_readiness",
    )
    _all_true_gates(report, READINESS_GATES, label="runtime-readiness report")
    if report.get("shape") != READINESS_SHAPE:
        raise SealError("runtime-readiness formal 2x256x256 shape changed")
    config = report.get("config")
    expected_config = {
        "calibration_updates": 256,
        "fingerprint_sha256": FIXED_CONFIG_FINGERPRINT_SHA256,
        "maximum_candidates": 256,
        "maximum_context_steps": 256,
        "profile": "preheat",
        "risk_actor_start_update": 512,
        "source": os.fspath(config_path.resolve()),
        "tbptt_window_steps": 16,
        "version": "sts2-relational-curriculum-config-v12",
    }
    if config != expected_config:
        raise SealError("runtime-readiness config/source changed")
    initialization = report.get("initialization")
    if not isinstance(initialization, Mapping) or (
        initialization.get("kind") != "frozen-v28-model-init"
        or initialization.get("contract_name") != FROZEN_CONTRACT_NAME
        or initialization.get("checkpoint_id") != FIXED_CHECKPOINT_ID
        or initialization.get("checkpoint_path") != os.fspath(checkpoint.resolve())
        or initialization.get("manifest_sha256") != FIXED_MANIFEST_SHA256
        or initialization.get("metadata_sha256") != FIXED_METADATA_SHA256
    ):
        raise SealError("runtime-readiness initialization source changed")
    identity = _runtime_identity(report.get("runtime_identity"), label="runtime-readiness runtime identity")
    if identity != dict(live_runtime_identity):
        raise SealError("runtime-readiness runtime identity differs from the live reviewed runtime")
    gpu_peak = report.get("gpu_peak")
    if not isinstance(gpu_peak, Mapping) or set(gpu_peak) != {
        "maximum_allocated_bytes",
        "maximum_reserved_bytes",
    }:
        raise SealError("runtime-readiness GPU peak schema changed")
    allocated = _positive_int(gpu_peak.get("maximum_allocated_bytes"), label="readiness allocated bytes", minimum=1)
    reserved = _positive_int(gpu_peak.get("maximum_reserved_bytes"), label="readiness reserved bytes", minimum=1)
    if allocated > reserved or reserved >= identity["accelerator"]["total_memory_bytes"]:
        raise SealError("runtime-readiness GPU memory bound failed")
    sdpa = report.get("sdpa")
    if not isinstance(sdpa, Mapping) or (
        sdpa.get("version") != "sts2-rocm-sdpa-execution-v1"
        or sdpa.get("applicability") != "rocm_cuda"
        or sdpa.get("applied") is not True
        or sdpa.get("effective_backend") != "math_only"
        or sdpa.get("requested_policy") != "math"
        or sdpa.get("hip_version") != identity["torch_hip_version"]
        or sdpa.get("devices") != ["cuda"]
        or sdpa.get("current_flags") != {"cudnn": False, "flash": False, "math": True, "memory_efficient": False}
    ):
        raise SealError("runtime-readiness stable SDPA proof changed")
    return {
        "config": dict(config),
        "gates": {key: True for key in sorted(READINESS_GATES)},
        "generation_source": generation_source,
        "gpu_peak": dict(gpu_peak),
        "initialization": dict(initialization),
        "runtime_identity": identity,
        "sdpa": dict(sdpa),
        "shape": dict(READINESS_SHAPE),
        "status": "passed",
        "training_authority": False,
        "version": READINESS_VERSION,
    }


def _load_runtime_identity(args: argparse.Namespace, *, checkout_root: Path, artifact_root: Path) -> dict[str, Any]:
    identity_path = getattr(args, "runtime_identity_json", None)
    runtime_python = getattr(args, "runtime_python", None)
    if identity_path and runtime_python:
        raise SealError("choose exactly one of --runtime-identity-json or --runtime-python")
    if identity_path:
        return _runtime_identity(
            _load_json(Path(identity_path).expanduser().resolve(), label="live runtime identity"),
            label="live runtime identity",
        )
    # Do not resolve the interpreter symlink: resolving
    # ``.../wsl-rocm/bin/python`` to ``/usr/bin/python`` discards the virtual
    # environment's ``pyvenv.cfg`` and therefore queries the wrong runtime.
    # ``absolute()`` normalizes the lexical path while preserving the venv
    # entry point that was explicitly reviewed.
    python_path = (
        Path(runtime_python).expanduser().absolute()
        if runtime_python
        else (artifact_root / "environments/wsl-rocm/bin/python").absolute()
    )
    if not python_path.is_file():
        raise SealError("live runtime identity is required; supply --runtime-identity-json or a valid --runtime-python")
    package_root = checkout_root / "packages/rl-agent"
    code = (
        "import json;"
        "from sts2_rl.training.launch_contract import current_rocm_runtime_identity;"
        "print(json.dumps(current_rocm_runtime_identity(),allow_nan=False,sort_keys=True))"
    )
    environment = dict(os.environ)
    previous = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = os.fspath(package_root) + (os.pathsep + previous if previous else "")
    try:
        completed = subprocess.run(
            (os.fspath(python_path), "-c", code),
            cwd=package_root,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="strict",
            timeout=60.0,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SealError("pinned ROCm runtime identity query failed") from exc
    if completed.returncode != 0:
        raise SealError("pinned ROCm runtime identity query failed: " + (completed.stderr.strip() or "unknown error"))
    try:
        raw = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise SealError("pinned ROCm runtime identity query returned malformed JSON") from exc
    return _runtime_identity(raw, label="live runtime identity")


def _base_closure_paths(root: Path, commit: str) -> list[str]:
    raw_contract = _git_blob(root, commit, SHADOW_CONTRACT_RELATIVE)
    try:
        contract = json.loads(raw_contract)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SealError("generation commit A has a malformed base shadow contract") from exc
    if not isinstance(contract, Mapping):
        raise SealError("generation commit A base shadow contract must be an object")
    if contract.get("schema_version") not in {
        "sts2-shadow-validation-contract-v2",
        SHADOW_CONTRACT_SCHEMA,
    }:
        raise SealError("base shadow contract schema is not reviewed")
    mapping = contract.get("validated_code_sha256")
    if not isinstance(mapping, Mapping) or not mapping:
        raise SealError("base shadow contract has no validated-code closure")
    paths: set[str] = set(REQUIRED_PRODUCTION_CLOSURE)
    for raw_path in mapping:
        if not isinstance(raw_path, str):
            raise SealError("base shadow contract has a non-text code path")
        paths.add(_normalized_relative(raw_path, label="base validated-code path"))
    forbidden = paths.intersection(REVIEWED_SEAL_PATHS)
    if forbidden:
        raise SealError("production closure must exclude reviewed seal paths: " + ", ".join(sorted(forbidden)))
    for relative in paths:
        candidate = _within(root / relative, root, label="production closure path")
        if not candidate.is_file():
            raise SealError(f"production closure file is missing: {relative}")
    return sorted(paths)


def _git_blob(root: Path, commit: str, relative: str) -> bytes:
    try:
        completed = subprocess.run(
            ("git", "show", f"{commit}:{relative}"),
            cwd=root,
            check=False,
            capture_output=True,
            timeout=30.0,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SealError(f"could not read {relative} from commit {commit}") from exc
    if completed.returncode != 0:
        raise SealError(f"production closure path is not tracked at commit {commit}: {relative}")
    return completed.stdout


def _closure_mapping(root: Path, commit: str, paths: Sequence[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for relative in paths:
        worktree_digest = _sha256_file(root / relative)
        # Prove that the path exists in A.  Do not compare raw worktree bytes
        # with the Git blob: core.autocrlf may legitimately make those byte
        # streams differ.  The clean-index proof plus the recorded worktree
        # digest binds the exact bytes used by the validators.  B must present
        # the same worktree digest below, independently of Git normalization.
        _git_blob(root, commit, relative)
        mapping[relative] = worktree_digest
    return mapping


def _ensure_fresh_reports(paths: Sequence[Path], *, root: Path, commit: str) -> None:
    # Formal procedure is commit A, then generate the four reports.  An older
    # preserved report is therefore rejected even if it happens to parse.
    raw_timestamp = _run_git(root, "show", "-s", "--format=%ct", commit)
    try:
        commit_timestamp = int(raw_timestamp)
    except ValueError as exc:
        raise SealError("Git returned a malformed commit timestamp") from exc
    for path in paths:
        try:
            modified = path.stat().st_mtime
        except OSError as exc:
            raise SealError(f"report is unreadable: {path}") from exc
        if modified < commit_timestamp:
            raise SealError(f"report predates generation commit A and is stale: {path}")


def _shadow_specs(report_digests: Mapping[str, str]) -> dict[str, Any]:
    return {
        "semantics_historical": {
            "maximum_errors": 0,
            "maximum_unknown_transitions": 0,
            "minimum_parsed_records": 500_000,
            "minimum_rich_decision_snapshots": 3_986,
            "relative_path": DEFAULT_REPORT_RELATIVES["semantics_historical"],
            "sha256": report_digests["semantics_historical"],
            "status": "passed",
            "version": SEMANTICS_VERSION,
        },
        "evidence_live": {
            "maximum_episode_completion_bytes": 134_217_728,
            "maximum_episode_completion_controls": 32,
            "maximum_prefer_targets": 0,
            "maximum_semantic_censored_transitions": 0,
            "minimum_decisions": 1_000,
            "relative_path": DEFAULT_REPORT_RELATIVES["evidence_live"],
            "require_all_gates": True,
            "require_no_training_authority": True,
            "require_read_only": True,
            "sha256": report_digests["evidence_live"],
            "source_checkpoint_id": FIXED_CHECKPOINT_ID,
            "status": "passed",
            "version": EVIDENCE_VERSION,
        },
        "evidence_actor": {
            "maximum_prefer_targets": 0,
            "minimum_actor_actionable_records": 4,
            "minimum_cases": 6,
            "minimum_risk_sequence_records": 5,
            "relative_path": DEFAULT_REPORT_RELATIVES["evidence_actor"],
            "require_all_gates": True,
            "require_no_training_authority": True,
            "require_read_only": True,
            "sha256": report_digests["evidence_actor"],
            "status": "passed",
            "version": ACTOR_VERSION,
        },
    }


def _shadow_authority_payload(
    *,
    contract_sha256: str,
    generation_source: Mapping[str, Any],
    code_mapping: Mapping[str, str],
    report_digests: Mapping[str, str],
    summaries: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "contract": {
            "name": SHADOW_CONTRACT_NAME,
            "schema_version": SHADOW_CONTRACT_SCHEMA,
            "sha256": contract_sha256,
        },
        "generation_source": dict(generation_source),
        "validated_code_sha256": dict(code_mapping),
        "reports": {
            key: {"sha256": report_digests[key], **dict(summaries[key])}
            for key in ("semantics_historical", "evidence_live", "evidence_actor")
        },
    }


def _expected_hash(args: argparse.Namespace, key: str) -> str | None:
    raw = getattr(args, f"expected_{key}_sha256", None)
    return _sha256(raw, label=f"expected {key} report SHA-256") if raw else None


def _prepare_payload(args: argparse.Namespace) -> tuple[dict[str, Any], bytes]:
    root = Path(args.checkout_root).expanduser().resolve()
    source = _clean_source(root, expected_commit=args.generation_commit)
    artifact_root = Path(args.artifact_root).expanduser().resolve()
    if not artifact_root.is_dir():
        raise SealError(f"artifact root is missing: {artifact_root}")
    # This is a formal, recipe-specific seal rather than a generic checkpoint
    # inspection CLI.  Keep both authorities pinned to their reviewed roots;
    # accepting caller-selected paths would allow a valid-looking seal plan to
    # describe a different config or a copied checkpoint outside the declared
    # checkout/artifact boundary.
    config_path = (root / DEFAULT_CONFIG_RELATIVE).resolve()
    checkpoint = (artifact_root / DEFAULT_CHECKPOINT_RELATIVE).resolve()
    if not config_path.is_file():
        raise SealError(f"v29 config is missing: {config_path}")
    if not checkpoint.is_dir():
        raise SealError(f"frozen checkpoint is missing: {checkpoint}")
    live_identity = _load_runtime_identity(args, checkout_root=root, artifact_root=artifact_root)
    report_paths = {
        key: _report_path(artifact_root, relative, label=f"{key} report")
        for key, relative in DEFAULT_REPORT_RELATIVES.items()
    }
    _ensure_fresh_reports(
        list(report_paths.values()),
        root=root,
        commit=source["implementation_commit"],
    )
    reports = {key: _load_json(path, label=f"{key} report") for key, path in report_paths.items()}
    summaries = {
        "semantics_historical": _validate_semantics(reports["semantics_historical"], expected_source=source, root=root),
        "evidence_live": _validate_evidence(
            reports["evidence_live"], checkpoint=checkpoint, expected_source=source, root=root
        ),
        "evidence_actor": _validate_actor(reports["evidence_actor"], expected_source=source, root=root),
    }
    readiness_binding = _validate_readiness(
        reports["runtime_readiness"],
        config_path=config_path,
        checkpoint=checkpoint,
        live_runtime_identity=live_identity,
        expected_source=source,
        root=root,
    )
    report_digests = {key: _sha256_file(path) for key, path in report_paths.items()}
    for key, observed in report_digests.items():
        expected = _expected_hash(args, key)
        if expected is not None and observed != expected:
            raise SealError(f"{key} report SHA-256 mismatch: expected={expected} actual={observed}")
    closure_paths = _base_closure_paths(root, source["implementation_commit"])
    code_mapping = _closure_mapping(root, source["implementation_commit"], closure_paths)
    mapping_sha256 = _canonical_json_sha256(code_mapping)
    shadow_contract = {
        "generation_source": source,
        "name": SHADOW_CONTRACT_NAME,
        "reports": _shadow_specs(report_digests),
        "schema_version": SHADOW_CONTRACT_SCHEMA,
        "source_checkpoint_contract": FROZEN_CONTRACT_NAME,
        "training_authority": False,
        "validated_code_sha256": code_mapping,
    }
    shadow_bytes = _pretty_json_bytes(shadow_contract)
    shadow_sha256 = _sha256_bytes(shadow_bytes)
    authority_sha256 = _canonical_json_sha256(
        _shadow_authority_payload(
            contract_sha256=shadow_sha256,
            generation_source=source,
            code_mapping=code_mapping,
            report_digests=report_digests,
            summaries=summaries,
        )
    )
    constants: dict[str, object] = {
        "ACTOR_EVIDENCE_SHADOW_SHA256": report_digests["evidence_actor"],
        "EVIDENCE_SHADOW_SHA256": report_digests["evidence_live"],
        "RUNTIME_READINESS_REPORT_SHA256": report_digests["runtime_readiness"],
        "SEMANTICS_SHADOW_SHA256": report_digests["semantics_historical"],
        "SHADOW_AUTHORITY_SHA256": authority_sha256,
        "SHADOW_CONTRACT_SHA256": shadow_sha256,
        "SHADOW_VALIDATED_CODE_COUNT": len(code_mapping),
        "SHADOW_VALIDATED_CODE_MAPPING_SHA256": mapping_sha256,
    }
    generation_launcher = _git_blob(root, source["implementation_commit"], LAUNCHER_RELATIVE)
    expected_launcher = _sealed_launcher_bytes(generation_launcher, constants)
    plan = {
        "artifact_root": os.fspath(artifact_root),
        "config_path": os.fspath(config_path),
        "generation_source": source,
        "initialization_checkpoint": os.fspath(checkpoint),
        "launch_allowed": False,
        "launcher_seal": {
            "constants": constants,
            "expected_blob_sha256": _sha256_bytes(expected_launcher),
            "expected_content_utf8": expected_launcher.decode("utf-8"),
            "generation_blob_sha256": _sha256_bytes(generation_launcher),
            "launcher_relative_path": LAUNCHER_RELATIVE,
            "shadow_contract_relative_path": SHADOW_CONTRACT_RELATIVE,
        },
        "reports": {
            key: {
                "relative_path": DEFAULT_REPORT_RELATIVES[key],
                "sha256": report_digests[key],
                "status": "passed",
                "version": reports[key]["version"],
            }
            for key in sorted(report_paths)
        },
        "reviewed_seal_paths": sorted(REVIEWED_SEAL_PATHS),
        "runtime_readiness_evidence": {
            "artifact_root": os.fspath(artifact_root),
            "report_binding": readiness_binding,
            "report_path": os.fspath(report_paths["runtime_readiness"]),
            "report_sha256": report_digests["runtime_readiness"],
            "schema_version": READINESS_EVIDENCE_SCHEMA,
        },
        "schema_version": PLAN_SCHEMA,
        "shadow_contract": {
            "content": shadow_contract,
            "relative_path": SHADOW_CONTRACT_RELATIVE,
            "sha256": shadow_sha256,
        },
        "source_authority": {
            "allowed_seal_paths": sorted(REVIEWED_SEAL_PATHS),
            "evidence_generation_source": source,
            "schema_version": SOURCE_AUTHORITY_SCHEMA,
            "validated_code_mapping_sha256": mapping_sha256,
            "validated_code_sha256": code_mapping,
        },
        "training_authority": False,
    }
    return plan, shadow_bytes


def _atomic_write(path: Path, raw: bytes) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _require_external_output(path: Path, *, root: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError:
        return resolved
    raise SealError(f"{label} must be outside the tracked checkout")


def _sealed_launcher_bytes(raw: bytes, expected: Mapping[str, object]) -> bytes:
    if set(expected) != set(LAUNCHER_SEAL_CONSTANT_NAMES):
        raise SealError("launcher seal constant set changed")
    try:
        text = raw.decode("utf-8")
        module = ast.parse(text, filename=LAUNCHER_RELATIVE)
    except (UnicodeDecodeError, SyntaxError) as exc:
        raise SealError("generation commit A launcher is not valid UTF-8 Python") from exc
    assignments: dict[str, ast.Assign] = {}
    for statement in module.body:
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            continue
        target = statement.targets[0]
        if not isinstance(target, ast.Name) or target.id not in LAUNCHER_SEAL_CONSTANT_NAMES:
            continue
        if target.id in assignments:
            raise SealError(f"generation launcher assigns seal constant more than once: {target.id}")
        if statement.col_offset != 0 or statement.end_lineno != statement.lineno:
            raise SealError(f"generation launcher seal constant must be one top-level line: {target.id}")
        assignments[target.id] = statement
    if set(assignments) != set(LAUNCHER_SEAL_CONSTANT_NAMES):
        missing = sorted(LAUNCHER_SEAL_CONSTANT_NAMES - set(assignments))
        raise SealError("generation launcher omits seal constants: " + ", ".join(missing))
    lines = text.splitlines(keepends=True)
    for name, statement in assignments.items():
        value = expected[name]
        if isinstance(value, str):
            rendered = json.dumps(value, ensure_ascii=True)
        elif isinstance(value, int) and not isinstance(value, bool):
            rendered = str(value)
        else:
            raise SealError(f"launcher seal constant {name} has an unsupported type")
        index = statement.lineno - 1
        newline = "\n" if lines[index].endswith("\n") else ""
        lines[index] = f"{name} = {rendered}{newline}"
    sealed = "".join(lines).encode("utf-8")
    # Re-parse the deterministic result and prove that no duplicate/later
    # assignment can override the reviewed top-level seal.
    try:
        sealed_module = ast.parse(sealed.decode("utf-8"), filename=LAUNCHER_RELATIVE)
    except SyntaxError as exc:  # pragma: no cover - construction invariant
        raise SealError("deterministic sealed launcher is invalid Python") from exc
    observed: dict[str, object] = {}
    for node in ast.walk(sealed_module):
        assigned_value: ast.expr | None
        if isinstance(node, ast.Assign):
            targets = node.targets
            assigned_value = node.value
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
            assigned_value = node.value
        else:
            continue
        for target in targets:
            if isinstance(target, ast.Name) and target.id in LAUNCHER_SEAL_CONSTANT_NAMES:
                if target.id in observed or node not in sealed_module.body:
                    raise SealError(f"sealed launcher has ambiguous seal assignment: {target.id}")
                if assigned_value is None:
                    raise SealError(f"sealed launcher seal value is missing: {target.id}")
                try:
                    observed[target.id] = ast.literal_eval(assigned_value)
                except (TypeError, ValueError) as exc:
                    raise SealError(f"sealed launcher seal value is not literal: {target.id}") from exc
    if observed != dict(expected):
        raise SealError("deterministic launcher seal values changed")
    return sealed


def _validate_plan_shape(plan: Mapping[str, Any]) -> None:
    expected = {
        "artifact_root",
        "config_path",
        "generation_source",
        "initialization_checkpoint",
        "launch_allowed",
        "launcher_seal",
        "reports",
        "reviewed_seal_paths",
        "runtime_readiness_evidence",
        "schema_version",
        "shadow_contract",
        "source_authority",
        "training_authority",
    }
    if set(plan) != expected or plan.get("schema_version") != PLAN_SCHEMA:
        raise SealError("seal plan schema changed")
    if plan.get("launch_allowed") is not False or plan.get("training_authority") is not False:
        raise SealError("seal plan must have no training authority")
    if plan.get("reviewed_seal_paths") != sorted(REVIEWED_SEAL_PATHS):
        raise SealError("seal plan reviewed path set changed")


def _verify(args: argparse.Namespace) -> dict[str, Any]:
    plan_path = Path(args.plan).expanduser().resolve()
    plan = _load_json(plan_path, label="authority seal plan")
    _validate_plan_shape(plan)
    source = plan.get("generation_source")
    if not isinstance(source, Mapping):
        raise SealError("seal plan generation source is malformed")
    root = (
        Path(args.checkout_root).expanduser().resolve()
        if args.checkout_root
        else Path(str(source.get("checkout_root"))).resolve()
    )
    current = _clean_source(root, expected_commit=None)
    if source.get("checkout_root") != os.fspath(root) or source.get("worktree_clean") is not True:
        raise SealError("seal plan belongs to another checkout")
    generation_commit = source.get("implementation_commit")
    generation_tree = source.get("implementation_tree")
    if not isinstance(generation_commit, str) or not isinstance(generation_tree, str):
        raise SealError("seal plan generation object IDs are malformed")
    observed_tree = _run_git(root, "rev-parse", f"{generation_commit}^{{tree}}")
    if observed_tree != generation_tree:
        raise SealError("generation commit A tree differs from the plan")
    if current["implementation_commit"] == generation_commit:
        raise SealError("verify requires a distinct seal commit B")
    parent_line = _run_git(root, "rev-list", "--parents", "-n", "1", current["implementation_commit"])
    parents = parent_line.split()
    if parents != [current["implementation_commit"], generation_commit]:
        raise SealError("seal commit B must have generation commit A as its only direct parent")
    changed = _run_git(
        root,
        "diff",
        "--name-only",
        "--no-renames",
        f"{generation_commit}..{current['implementation_commit']}",
        "--",
    )
    observed_paths = sorted(line for line in changed.splitlines() if line)
    if observed_paths != sorted(REVIEWED_SEAL_PATHS):
        raise SealError(f"A..B changed paths differ from the exact reviewed seal set: {observed_paths!r}")
    authority = plan.get("source_authority")
    if not isinstance(authority, Mapping) or authority.get("schema_version") != SOURCE_AUTHORITY_SCHEMA:
        raise SealError("seal plan source authority is malformed")
    mapping = authority.get("validated_code_sha256")
    if not isinstance(mapping, Mapping) or not mapping:
        raise SealError("seal plan has no production closure")
    expected_closure_paths = _base_closure_paths(root, generation_commit)
    if set(mapping) != set(expected_closure_paths):
        raise SealError("seal plan production closure differs from generation commit A authority")
    normalized_mapping: dict[str, str] = {}
    for raw_relative, raw_digest in sorted(mapping.items()):
        if not isinstance(raw_relative, str):
            raise SealError("seal plan production path is malformed")
        relative = _normalized_relative(raw_relative, label="seal plan production path")
        digest = _sha256(raw_digest, label=f"seal plan production digest {relative}")
        _git_blob(root, generation_commit, relative)
        current_digest = _sha256_file(root / relative)
        if digest != current_digest:
            raise SealError(f"production closure drifted after report generation: {relative}")
        normalized_mapping[relative] = digest
    if set(normalized_mapping) != set(expected_closure_paths):
        raise SealError("seal plan omits generation commit A production closure paths")
    mapping_digest = _canonical_json_sha256(normalized_mapping)
    if authority.get("validated_code_mapping_sha256") != mapping_digest:
        raise SealError("seal plan production-closure mapping digest changed")
    artifact_root = Path(str(plan.get("artifact_root"))).resolve()
    config_path = Path(str(plan.get("config_path"))).resolve()
    checkpoint = Path(str(plan.get("initialization_checkpoint"))).resolve()
    live_identity = _load_runtime_identity(args, checkout_root=root, artifact_root=artifact_root)
    report_specs = plan.get("reports")
    if not isinstance(report_specs, Mapping) or set(report_specs) != set(DEFAULT_REPORT_RELATIVES):
        raise SealError("seal plan report set changed")
    loaded: dict[str, dict[str, Any]] = {}
    digests: dict[str, str] = {}
    for key, expected_relative in DEFAULT_REPORT_RELATIVES.items():
        spec = report_specs.get(key)
        if not isinstance(spec, Mapping) or spec.get("relative_path") != expected_relative:
            raise SealError(f"seal plan {key} report path changed")
        path = _report_path(artifact_root, expected_relative, label=f"{key} report")
        digest = _sha256_file(path)
        if spec.get("sha256") != digest:
            raise SealError(f"{key} report hash changed after prepare")
        loaded[key] = _load_json(path, label=f"{key} report")
        digests[key] = digest
    summaries = {
        "semantics_historical": _validate_semantics(loaded["semantics_historical"], expected_source=source, root=root),
        "evidence_live": _validate_evidence(
            loaded["evidence_live"], checkpoint=checkpoint, expected_source=source, root=root
        ),
        "evidence_actor": _validate_actor(loaded["evidence_actor"], expected_source=source, root=root),
    }
    readiness_binding = _validate_readiness(
        loaded["runtime_readiness"],
        config_path=config_path,
        checkpoint=checkpoint,
        live_runtime_identity=live_identity,
        expected_source=source,
        root=root,
    )
    readiness_evidence = plan.get("runtime_readiness_evidence")
    if not isinstance(readiness_evidence, Mapping) or (
        readiness_evidence.get("schema_version") != READINESS_EVIDENCE_SCHEMA
        or readiness_evidence.get("report_sha256") != digests["runtime_readiness"]
        or readiness_evidence.get("report_binding") != readiness_binding
    ):
        raise SealError("runtime-readiness evidence binding changed")
    shadow = plan.get("shadow_contract")
    if not isinstance(shadow, Mapping) or set(shadow) != {"content", "relative_path", "sha256"}:
        raise SealError("seal plan shadow contract is malformed")
    content = shadow.get("content")
    if not isinstance(content, Mapping):
        raise SealError("seal plan shadow contract content is malformed")
    shadow_bytes = _pretty_json_bytes(content)
    shadow_digest = _sha256_bytes(shadow_bytes)
    if shadow.get("relative_path") != SHADOW_CONTRACT_RELATIVE or shadow.get("sha256") != shadow_digest:
        raise SealError("seal plan shadow contract hash/path changed")
    staged = Path(args.staged_shadow_contract).expanduser().resolve()
    if _sha256_file(staged) != shadow_digest or staged.read_bytes() != shadow_bytes:
        raise SealError("staged shadow contract differs from the deterministic plan")
    tracked = root / SHADOW_CONTRACT_RELATIVE
    if _sha256_file(tracked) != shadow_digest or tracked.read_bytes() != shadow_bytes:
        raise SealError("seal commit B did not install the exact staged shadow contract")
    authority_digest = _canonical_json_sha256(
        _shadow_authority_payload(
            contract_sha256=shadow_digest,
            generation_source=source,
            code_mapping=normalized_mapping,
            report_digests=digests,
            summaries=summaries,
        )
    )
    launcher_seal = plan.get("launcher_seal")
    if not isinstance(launcher_seal, Mapping) or set(launcher_seal) != {
        "constants",
        "expected_blob_sha256",
        "expected_content_utf8",
        "generation_blob_sha256",
        "launcher_relative_path",
        "shadow_contract_relative_path",
    }:
        raise SealError("seal plan launcher binding changed")
    if (
        launcher_seal.get("launcher_relative_path") != LAUNCHER_RELATIVE
        or launcher_seal.get("shadow_contract_relative_path") != SHADOW_CONTRACT_RELATIVE
    ):
        raise SealError("seal plan launcher binding changed")
    constants = launcher_seal.get("constants")
    expected_constants = {
        "ACTOR_EVIDENCE_SHADOW_SHA256": digests["evidence_actor"],
        "EVIDENCE_SHADOW_SHA256": digests["evidence_live"],
        "RUNTIME_READINESS_REPORT_SHA256": digests["runtime_readiness"],
        "SEMANTICS_SHADOW_SHA256": digests["semantics_historical"],
        "SHADOW_AUTHORITY_SHA256": authority_digest,
        "SHADOW_CONTRACT_SHA256": shadow_digest,
        "SHADOW_VALIDATED_CODE_COUNT": len(normalized_mapping),
        "SHADOW_VALIDATED_CODE_MAPPING_SHA256": mapping_digest,
    }
    if constants != expected_constants:
        raise SealError("seal plan launcher constants changed")
    generation_launcher = _git_blob(root, generation_commit, LAUNCHER_RELATIVE)
    expected_launcher = _sealed_launcher_bytes(generation_launcher, expected_constants)
    if (
        launcher_seal.get("generation_blob_sha256") != _sha256_bytes(generation_launcher)
        or launcher_seal.get("expected_blob_sha256") != _sha256_bytes(expected_launcher)
        or launcher_seal.get("expected_content_utf8") != expected_launcher.decode("utf-8")
    ):
        raise SealError("seal plan deterministic launcher bytes changed")
    observed_launcher = _git_blob(root, current["implementation_commit"], LAUNCHER_RELATIVE)
    if observed_launcher != expected_launcher:
        raise SealError("seal commit B changed launcher bytes beyond the reviewed constants")
    return {
        "generation_commit": generation_commit,
        "seal_commit": current["implementation_commit"],
        "observed_seal_paths": observed_paths,
        "production_closure_files": len(normalized_mapping),
        "runtime_readiness_report_sha256": digests["runtime_readiness"],
        "schema_version": PLAN_SCHEMA,
        "shadow_authority_sha256": authority_digest,
        "shadow_contract_sha256": shadow_digest,
        "status": "verified",
        "training_authority": False,
    }


def _add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--runtime-identity-json")
    group.add_argument("--runtime-python")


def _add_prepare_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkout-root", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--generation-commit", required=True)
    for key in DEFAULT_REPORT_RELATIVES:
        parser.add_argument(f"--expected-{key.replace('_', '-')}-sha256")
    _add_runtime_arguments(parser)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    dry_run = subcommands.add_parser("dry-run", help="validate commit A and print the plan")
    _add_prepare_arguments(dry_run)
    prepare = subcommands.add_parser("prepare", help="write an external staged contract and seal plan")
    _add_prepare_arguments(prepare)
    prepare.add_argument("--plan-output", required=True)
    prepare.add_argument("--shadow-contract-output", required=True)
    verify = subcommands.add_parser("verify", help="verify a clean two-file seal commit B")
    verify.add_argument("--checkout-root")
    verify.add_argument("--plan", required=True)
    verify.add_argument("--staged-shadow-contract", required=True)
    _add_runtime_arguments(verify)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command in {"dry-run", "prepare"}:
            plan, shadow_bytes = _prepare_payload(args)
            if args.command == "dry-run":
                sys.stdout.buffer.write(_pretty_json_bytes(plan))
                return 0
            root = Path(args.checkout_root).expanduser().resolve()
            plan_output = _require_external_output(Path(args.plan_output), root=root, label="plan output")
            shadow_output = _require_external_output(
                Path(args.shadow_contract_output), root=root, label="shadow-contract output"
            )
            if plan_output == shadow_output:
                raise SealError("plan and staged shadow contract outputs must be distinct")
            _atomic_write(shadow_output, shadow_bytes)
            _atomic_write(plan_output, _pretty_json_bytes(plan))
            result = {
                "plan_output": os.fspath(plan_output),
                "plan_sha256": _sha256_file(plan_output),
                "shadow_contract_output": os.fspath(shadow_output),
                "shadow_contract_sha256": _sha256_file(shadow_output),
                "status": "prepared",
                "training_authority": False,
            }
        else:
            result = _verify(args)
    except SealError as exc:
        print(f"v29 authority seal refused: {exc}", file=sys.stderr)
        return 2
    sys.stdout.buffer.write(_pretty_json_bytes(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
