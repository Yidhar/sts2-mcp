"""Materialize a higher-quality combat snapshot dataset from accepted runs_summary rows.

This fixes the current mismatch where ``combat_snapshot_samples.jsonl`` contains
rows from basic-filtered runs, while ``runs_summary.jsonl`` only contains
quality-accepted runs. The resulting output can be used directly by combat
sandbox training.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from combat_snapshot_dataset import infer_encounter_tier
from sts2_rl.artifacts import resolve_artifact_path, resolve_external_input_path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_run_id_file(path: Path | None) -> set[str]:
    if path is None:
        return set()
    run_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            token = line.strip()
            if not token or token.startswith("#"):
                continue
            run_ids.add(token)
    return run_ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Build quality-only combat snapshot dataset.")
    parser.add_argument("--combat-snapshots", type=Path, required=True)
    parser.add_argument("--runs-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--character", type=str, default=None)
    parser.add_argument("--encounter-tiers", type=str, default=None,
                        help="Optional comma-separated tier filter, e.g. weak,normal")
    parser.add_argument("--min-path-points", type=int, default=None)
    parser.add_argument("--wins-only", action="store_true", default=False)
    parser.add_argument("--allow-run-ids-file", type=Path, default=None)
    parser.add_argument("--exclude-run-ids-file", type=Path, default=None)
    args = parser.parse_args()

    args.combat_snapshots = resolve_external_input_path(args.combat_snapshots)
    args.runs_summary = resolve_external_input_path(args.runs_summary)
    args.allow_run_ids_file = (
        resolve_external_input_path(args.allow_run_ids_file)
        if args.allow_run_ids_file is not None
        else None
    )
    args.exclude_run_ids_file = (
        resolve_external_input_path(args.exclude_run_ids_file)
        if args.exclude_run_ids_file is not None
        else None
    )
    args.output = resolve_artifact_path(args.output)

    combat_rows = read_jsonl(args.combat_snapshots)
    run_rows = read_jsonl(args.runs_summary)
    accepted_runs = {str(row.get("run_id")): row for row in run_rows}
    tier_filter = {
        chunk.strip().lower()
        for chunk in (args.encounter_tiers or "").split(",")
        if chunk.strip()
    }
    allowed_run_ids = read_run_id_file(args.allow_run_ids_file)
    excluded_run_ids = read_run_id_file(args.exclude_run_ids_file)

    kept: list[dict[str, Any]] = []
    reject = Counter()
    for row in combat_rows:
        run_id = str(row.get("run_id") or "")
        run_summary = accepted_runs.get(run_id)
        if run_summary is None:
            reject["run_not_quality_accepted"] += 1
            continue

        if allowed_run_ids and run_id not in allowed_run_ids:
            reject["not_in_allow_list"] += 1
            continue

        if run_id in excluded_run_ids:
            reject["excluded_run_id"] += 1
            continue

        if args.character and str(row.get("character") or "") != args.character:
            reject["character_mismatch"] += 1
            continue

        tier = infer_encounter_tier(
            str(row.get("encounter_id") or ""),
            room_type=str(row.get("room_type") or ""),
        )
        if tier_filter and tier not in tier_filter:
            reject["tier_filtered"] += 1
            continue

        if args.wins_only and not bool(run_summary.get("win")):
            reject["loss_filtered"] += 1
            continue

        path_points = int(run_summary.get("path_point_count") or 0)
        if args.min_path_points is not None and path_points < args.min_path_points:
            reject["path_point_filtered"] += 1
            continue

        enriched = dict(row)
        enriched["quality_source"] = "runs_summary_accepted"
        enriched["source_file"] = run_summary.get("source_file")
        enriched["source_run_win"] = run_summary.get("win")
        enriched["source_run_path_point_count"] = run_summary.get("path_point_count")
        enriched["source_run_time_seconds"] = run_summary.get("run_time_seconds")
        enriched["source_killed_by_encounter"] = run_summary.get("killed_by_encounter")
        enriched["source_ingest_kind"] = run_summary.get("ingest_source_kind")
        kept.append(enriched)

    write_jsonl(args.output, kept)

    encounter_counts = Counter(str(row.get("encounter_id") or "unknown") for row in kept)
    report = {
        "input_combat_rows": len(combat_rows),
        "accepted_runs": len(accepted_runs),
        "kept_rows": len(kept),
        "kept_unique_runs": len({str(row.get("run_id")) for row in kept}),
        "reject_reasons": dict(sorted(reject.items())),
        "character": args.character,
        "encounter_tiers": sorted(tier_filter),
        "min_path_points": args.min_path_points,
        "wins_only": args.wins_only,
        "allow_run_ids_file": str(args.allow_run_ids_file) if args.allow_run_ids_file else None,
        "exclude_run_ids_file": str(args.exclude_run_ids_file) if args.exclude_run_ids_file else None,
        "top_encounters": encounter_counts.most_common(20),
    }
    report_path = args.output.with_suffix(args.output.suffix + ".report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
