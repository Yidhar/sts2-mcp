#!/usr/bin/env python3
"""Replay rich trajectory records through the decision-semantics kernel.

This command is intentionally read-only.  It does not load a model, alter a
checkpoint, write replay, or produce learner targets.  Its sole purpose is to
prove that the versioned semantic projection can consume real DTOs without
silently merging learned candidates or guessing at unsupported fields.

Trajectory journals contain compact records for every decision and sparse
``decision_snapshot`` records with the full observation and legal-action DTOs.
Only the latter are authoritative inputs to this shadow audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Final

from sts2_rl.artifacts import resolve_artifact_path
from sts2_rl.semantics import DecisionSemanticsKernel
from sts2_rl.training.launch_contract import current_formal_report_generation_source

SHADOW_REPORT_VERSION: Final = "sts2-semantics-shadow-report-v2"
_CHECKOUT_ROOT: Final = Path(__file__).resolve().parents[3]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _trajectory_paths(inputs: Iterable[str]) -> tuple[Path, ...]:
    discovered: set[Path] = set()
    for raw in inputs:
        path = Path(raw).expanduser().resolve()
        if path.is_file():
            discovered.add(path)
            continue
        if path.is_dir():
            discovered.update(
                candidate.resolve() for candidate in path.rglob("trajectory.jsonl") if candidate.is_file()
            )
            continue
        raise FileNotFoundError(f"shadow input does not exist: {path}")
    if not discovered:
        raise ValueError("shadow audit found no trajectory.jsonl inputs")
    return tuple(sorted(discovered, key=lambda item: os.fspath(item)))


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(serialized)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def audit_trajectory_semantics(
    paths: tuple[Path, ...],
    *,
    maximum_errors: int = 100,
    maximum_unknown_transitions: int = 0,
    minimum_rich_records: int = 1,
) -> dict[str, Any]:
    """Return one deterministic, fail-closed semantic shadow report."""

    if maximum_errors <= 0:
        raise ValueError("maximum_errors must be positive")
    if maximum_unknown_transitions < 0:
        raise ValueError("maximum_unknown_transitions must be non-negative")
    if minimum_rich_records <= 0:
        raise ValueError("minimum_rich_records must be positive")
    kernel = DecisionSemanticsKernel()
    surfaces: Counter[str] = Counter()
    progress: Counter[str] = Counter()
    detector_evidence: Counter[str] = Counter()
    errors: list[dict[str, Any]] = []
    parsed_records = 0
    rich_records = 0
    transitions = 0
    semantic_candidates = 0
    maximum_candidates = 0

    for path in paths:
        with path.open("r", encoding="utf-8", errors="strict") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                parsed_records += 1
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    errors.append(
                        {
                            "path": os.fspath(path),
                            "line": line_number,
                            "stage": "json",
                            "error_type": type(exc).__name__,
                            "message": str(exc),
                        }
                    )
                    if len(errors) >= maximum_errors:
                        break
                    continue
                if not isinstance(record, dict) or record.get("event") != "decision_snapshot":
                    continue
                rich_records += 1
                observation = record.get("observation")
                legal_actions = record.get("legal_actions")
                if not isinstance(observation, dict) or not isinstance(
                    legal_actions,
                    list,
                ):
                    errors.append(
                        {
                            "path": os.fspath(path),
                            "line": line_number,
                            "stage": "record_contract",
                            "error_type": "TypeError",
                            "message": ("decision_snapshot requires object observation and array legal_actions"),
                        }
                    )
                    if len(errors) >= maximum_errors:
                        break
                    continue
                try:
                    before = kernel.identify(
                        observation=observation,
                        legal_actions=legal_actions,
                    )
                    candidate_count = len(before.actions)
                    semantic_candidates += candidate_count
                    maximum_candidates = max(maximum_candidates, candidate_count)
                    surfaces[before.scopes.root.spec_id] += 1
                    recorded_count = record.get("semantic_candidate_count")
                    if recorded_count is not None and recorded_count != candidate_count:
                        raise ValueError(
                            "kernel candidate count differs from recorded "
                            f"semantic_candidate_count: {candidate_count} != "
                            f"{recorded_count}"
                        )

                    deadlock = record.get("deadlock")
                    if isinstance(deadlock, dict):
                        raw_kind = deadlock.get("kind")
                        kind = (
                            str(raw_kind)
                            if isinstance(raw_kind, str) and raw_kind
                            else "semantic_action_cycle"
                            if {
                                "decision_fingerprint",
                                "action_fingerprint",
                            }
                            <= set(deadlock)
                            else "unclassified"
                        )
                        detector_evidence[kind] += 1

                    after_observation = record.get("result_observation")
                    after_actions = record.get("result_legal_actions")
                    if isinstance(after_observation, dict) and isinstance(after_actions, list) and after_actions:
                        after = kernel.identify(
                            observation=after_observation,
                            legal_actions=after_actions,
                            parent_scopes=before.scopes,
                        )
                        receipt = kernel.classify_transition(
                            before=before,
                            after=after,
                            before_observation=observation,
                            after_observation=after_observation,
                        )
                        progress[receipt.kind.value] += 1
                        transitions += 1
                except Exception as exc:
                    errors.append(
                        {
                            "path": os.fspath(path),
                            "line": line_number,
                            "stage": "semantic_kernel",
                            "error_type": type(exc).__name__,
                            "message": str(exc),
                            "phase": observation.get("phase"),
                            "screen": observation.get("screen"),
                            "decision_domain": observation.get("decision_domain"),
                        }
                    )
                    if len(errors) >= maximum_errors:
                        break
            if len(errors) >= maximum_errors:
                break

    manifest = kernel.registry.manifest_key(kernel.key_index)
    unknown_transitions = progress["unknown"]
    gates = {
        "no_contract_errors": not errors,
        "minimum_rich_records": rich_records >= minimum_rich_records,
        "unknown_transition_budget": (unknown_transitions <= maximum_unknown_transitions),
    }
    inputs = [
        {
            "path": os.fspath(path),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in paths
    ]
    return {
        "version": SHADOW_REPORT_VERSION,
        "status": "passed" if all(gates.values()) else "failed",
        "read_only": True,
        "training_authority": False,
        "gates": {
            **gates,
            "required_minimum_rich_records": minimum_rich_records,
            "maximum_unknown_transitions": maximum_unknown_transitions,
            "observed_unknown_transitions": unknown_transitions,
        },
        "semantic_manifest": {
            "namespace": manifest.namespace,
            "schema_version": manifest.schema_version,
            "digest": manifest.digest,
            "payload": manifest.payload,
        },
        "inputs": inputs,
        "counts": {
            "files": len(paths),
            "parsed_records": parsed_records,
            "rich_decision_snapshots": rich_records,
            "classified_transitions": transitions,
            "semantic_candidates": semantic_candidates,
            "maximum_semantic_candidates": maximum_candidates,
            "interned_semantic_keys": len(kernel.key_index),
        },
        "surfaces": dict(sorted(surfaces.items())),
        "progress_receipts": dict(sorted(progress.items())),
        "legacy_detector_evidence": dict(sorted(detector_evidence.items())),
        "errors": errors,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="+",
        help="trajectory.jsonl files or directories recursively containing them",
    )
    parser.add_argument(
        "--report",
        type=Path,
        required=True,
        help="atomic JSON report output",
    )
    parser.add_argument("--maximum-errors", type=int, default=100)
    parser.add_argument(
        "--maximum-unknown-transitions",
        type=int,
        default=0,
        help=("explicit censor budget; zero is the formal promotion gate for reviewed surfaces"),
    )
    parser.add_argument("--minimum-rich-records", type=int, default=1)
    return parser


def main() -> int:
    args = _parser().parse_args()
    generation_source = current_formal_report_generation_source(_CHECKOUT_ROOT, __file__)
    paths = _trajectory_paths(args.inputs)
    report_path = resolve_artifact_path(args.report)
    report = audit_trajectory_semantics(
        paths,
        maximum_errors=args.maximum_errors,
        maximum_unknown_transitions=args.maximum_unknown_transitions,
        minimum_rich_records=args.minimum_rich_records,
    )
    if current_formal_report_generation_source(_CHECKOUT_ROOT, __file__) != generation_source:
        raise RuntimeError("formal semantics source changed during report generation")
    report["generation_source"] = generation_source
    _atomic_json(report_path, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "report": os.fspath(report_path),
                "counts": report["counts"],
                "errors": len(report["errors"]),
            },
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
