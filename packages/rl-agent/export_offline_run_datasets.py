"""Batch-export offline datasets from Slay the Spire 2 native .run history files.

This script is intended for large-scale collection. It scans a history directory,
parses each native .run file, applies lightweight cleaning, and writes aggregated
JSONL datasets for offline training:

    - runs_summary.jsonl
    - route_samples.jsonl
    - card_choice_samples.jsonl
    - build_samples.jsonl
    - manifest.json

Examples:
    python export_offline_run_datasets.py "C:/.../saves/history" --output-dir tmp/offline_dataset
    python export_offline_run_datasets.py "C:/.../saves/history" --output-dir tmp/offline_dataset --max-runs 100
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

from run_history_parser import (
    build_offline_build_v2_samples,
    build_offline_training_samples,
    extract_run_history_bytes,
)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def _sanitize_path_component(value: str) -> str:
    sanitized = "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in str(value))
    return sanitized or "unknown"


def _merge_numeric_audit(
    target: Counter[str],
    audit: dict[str, Any] | None,
) -> None:
    if not audit:
        return
    for key, value in audit.items():
        if isinstance(value, bool):
            target[key] += int(value)
        elif isinstance(value, int):
            target[key] += value


def _bucket_rows_by_build_id(
    rows: list[dict[str, Any]],
    *,
    run_to_build_id: dict[str, str],
) -> dict[str, list[dict[str, Any]]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        run_id = str(row.get("run_id"))
        build_id = run_to_build_id.get(run_id)
        if build_id is None:
            build_id = row.get("build_id")
        build_key = str(build_id or "unknown")
        buckets.setdefault(build_key, []).append(row)
    return buckets


def _build_manifest(
    *,
    dataset_schema: str,
    input_value: str,
    processed: int,
    basic_accepted: int,
    accepted: int,
    skip_reasons: dict[str, int],
    filters: dict[str, Any],
    runs_summary: list[dict[str, Any]],
    floor_records: list[dict[str, Any]],
    decision_records: list[dict[str, Any]],
    route_samples: list[dict[str, Any]],
    card_choice_samples: list[dict[str, Any]],
    build_samples: list[dict[str, Any]],
    build_samples_by_type: dict[str, list[dict[str, Any]]],
    build_v2_task_rows: dict[str, list[dict[str, Any]]] | None = None,
    build_v2_audit: dict[str, Any] | None = None,
    extra_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    build_ids: Counter[str] = Counter()
    characters: Counter[str] = Counter()
    wins: Counter[str] = Counter()
    splits: Counter[str] = Counter()

    for summary in runs_summary:
        if summary.get("build_id"):
            build_ids[str(summary["build_id"])] += 1
        if summary.get("split"):
            splits[str(summary["split"])] += 1
        for character in summary.get("characters") or []:
            if character:
                characters[str(character)] += 1
        wins["win" if summary.get("win") else "loss"] += 1

    manifest = {
        "dataset_schema": dataset_schema,
        "input": input_value,
        "processed_run_payloads": processed,
        "processed_run_files": processed,
        "accepted_after_basic_filters": basic_accepted,
        "accepted_runs": accepted,
        "skipped_runs": processed - accepted,
        "skip_reasons": dict(sorted(skip_reasons.items())),
        "filters": filters,
        "dataset_counts": {
            "runs_summary": len(runs_summary),
            "floor_records": len(floor_records),
            "decision_records": len(decision_records),
            "route_samples": len(route_samples),
            "card_choice_samples": len(card_choice_samples),
            "build_samples": len(build_samples),
        },
        "build_dataset_counts_by_type": {
            decision_type: len(rows)
            for decision_type, rows in sorted(build_samples_by_type.items())
        },
        "build_ids": dict(sorted(build_ids.items())),
        "characters": dict(sorted(characters.items())),
        "splits": dict(sorted(splits.items())),
        "wins": dict(sorted(wins.items())),
    }
    if build_v2_task_rows is not None:
        manifest["dataset_counts"]["build_v2_total"] = sum(len(rows) for rows in build_v2_task_rows.values())
        manifest["build_v2_dataset_counts_by_task"] = {
            task: len(rows)
            for task, rows in sorted(build_v2_task_rows.items())
        }
    if build_v2_audit:
        manifest["build_v2_audit"] = build_v2_audit
    if extra_fields:
        manifest.update(extra_fields)
    return manifest


def _write_dataset_bundle(
    *,
    output_dir: Path,
    manifest: dict[str, Any],
    runs_summary: list[dict[str, Any]],
    floor_records: list[dict[str, Any]],
    decision_records: list[dict[str, Any]],
    route_samples: list[dict[str, Any]],
    card_choice_samples: list[dict[str, Any]],
    build_samples: list[dict[str, Any]],
    build_v2_task_rows: dict[str, list[dict[str, Any]]] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "runs_summary.jsonl", runs_summary)
    _write_jsonl(output_dir / "floor_records.jsonl", floor_records)
    _write_jsonl(output_dir / "decision_records.jsonl", decision_records)
    _write_jsonl(output_dir / "route_samples.jsonl", route_samples)
    _write_jsonl(output_dir / "card_choice_samples.jsonl", card_choice_samples)
    _write_jsonl(output_dir / "build_samples.jsonl", build_samples)

    build_samples_by_type: dict[str, list[dict[str, Any]]] = {}
    for sample in build_samples:
        decision_type = str(sample.get("decision_type") or "unknown")
        build_samples_by_type.setdefault(decision_type, []).append(sample)
    for decision_type, rows in sorted(build_samples_by_type.items()):
        _write_jsonl(output_dir / f"{decision_type}_samples.jsonl", rows)

    if build_v2_task_rows:
        for task_name, rows in sorted(build_v2_task_rows.items()):
            _write_jsonl(output_dir / f"{task_name}_samples.jsonl", rows)

    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _collect_run_sources(input_path: Path) -> list[dict[str, Any]]:
    if input_path.is_file():
        return list(_expand_single_input(input_path))
    if input_path.is_dir():
        sources: list[dict[str, Any]] = []
        for run_path in sorted(input_path.rglob("*.run")):
            sources.append(
                {
                    "kind": "file",
                    "display_path": str(run_path),
                    "path": run_path,
                }
            )
        for archive_path in sorted(input_path.rglob("*.zip")):
            sources.extend(_expand_zip_input(archive_path))
        return sources
    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def _expand_single_input(input_path: Path) -> list[dict[str, Any]]:
    suffix = input_path.suffix.lower()
    if suffix == ".run":
        return [
            {
                "kind": "file",
                "display_path": str(input_path),
                "path": input_path,
            }
        ]
    if suffix == ".zip":
        return _expand_zip_input(input_path)
    raise FileNotFoundError(f"Unsupported input file type: {input_path}")


def _expand_zip_input(archive_path: Path) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    with zipfile.ZipFile(archive_path) as archive:
        run_names = sorted(
            name
            for name in archive.namelist()
            if name.lower().endswith(".run")
        )
    for member_name in run_names:
        sources.append(
            {
                "kind": "zip_member",
                "display_path": f"{archive_path}!{member_name}",
                "archive_path": archive_path,
                "member_name": member_name,
            }
        )
    return sources


def _load_source_payload(source: dict[str, Any]) -> bytes:
    kind = source["kind"]
    if kind == "file":
        return Path(source["path"]).read_bytes()
    if kind == "zip_member":
        with zipfile.ZipFile(source["archive_path"]) as archive:
            return archive.read(source["member_name"])
    raise ValueError(f"Unsupported source kind: {kind}")


def _load_run_bundle(source: dict[str, Any], payload: bytes) -> dict[str, Any]:
    return extract_run_history_bytes(payload, source["display_path"])


def _load_source_digest(payload: bytes) -> str:
    return hashlib.sha1(payload).hexdigest()


def _resolve_build_family(build_id: str | None) -> str:
    build_key = str(build_id or "unknown")
    if build_key in {"v0.98.0", "v0.98.1", "v0.98.2", "v0.98.3", "v0.99.1"}:
        return "v0.98_to_v0.99.1"
    return build_key


def _clean_run_bundle(
    bundle: dict[str, Any],
    *,
    single_player_only: bool,
    game_mode: str | None,
) -> tuple[bool, str | None]:
    summary = bundle["summary"]

    if single_player_only and summary["player_count"] != 1:
        return False, "non_single_player"

    if game_mode and summary.get("game_mode") != game_mode:
        return False, "game_mode_mismatch"

    if not bundle.get("floors"):
        return False, "empty_path"

    if not summary.get("characters"):
        return False, "missing_character"

    return True, None


def _quality_filter_bundle(
    bundle: dict[str, Any],
    samples: dict[str, list[dict[str, Any]]],
    *,
    build_v2: dict[str, Any] | None,
    quality_profile: str,
    min_loss_path_points: int,
) -> tuple[bool, str | None]:
    if quality_profile == "none":
        return True, None

    summary = bundle["summary"]

    if quality_profile == "offline_training_v1":
        build_v2_total = sum(len(rows) for rows in ((build_v2 or {}).get("task_rows") or {}).values())
        if summary.get("was_abandoned"):
            return False, "abandoned_run"
        if not summary.get("win") and (summary.get("path_point_count") or 0) < min_loss_path_points:
            return False, "short_loss"
        if not summary.get("win") and not samples["card_choice_samples"] and not samples["build_samples"] and build_v2_total <= 0:
            return False, "uninformative_loss"
        return True, None

    raise ValueError(f"Unsupported quality profile: {quality_profile}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export large-scale offline route/card/build datasets from STS2 native .run history files."
    )
    parser.add_argument("input", type=str, help="Path to a .run file or directory containing .run files.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory for aggregated JSONL outputs.")
    parser.add_argument("--max-runs", type=int, default=None, help="Optional cap on number of .run files to process.")
    parser.add_argument(
        "--single-player-only",
        action="store_true",
        default=True,
        help="Keep only single-player runs. Enabled by default.",
    )
    parser.add_argument(
        "--game-mode",
        type=str,
        default="standard",
        help="Only keep runs with this game_mode. Default: standard. Use '' to disable.",
    )
    parser.add_argument(
        "--quality-profile",
        type=str,
        default="offline_training_v1",
        choices=["none", "offline_training_v1"],
        help="Quality filtering profile. Default: offline_training_v1.",
    )
    parser.add_argument(
        "--min-loss-path-points",
        type=int,
        default=12,
        help="For quality profiles that keep some losses, require at least this many path points. Default: 12.",
    )
    parser.add_argument(
        "--materialize-run-payloads",
        action="store_true",
        help="Write deduplicated raw .run payloads under <output-dir>/raw_run_payloads and emit an index file.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_sources = _collect_run_sources(input_path)
    if args.max_runs is not None:
        run_sources = run_sources[: args.max_runs]

    runs_summary: list[dict[str, Any]] = []
    floor_records: list[dict[str, Any]] = []
    decision_records: list[dict[str, Any]] = []
    route_samples: list[dict[str, Any]] = []
    card_choice_samples: list[dict[str, Any]] = []
    build_samples: list[dict[str, Any]] = []
    build_samples_by_type: dict[str, list[dict[str, Any]]] = {}
    build_v2_task_rows: dict[str, list[dict[str, Any]]] = {}
    build_v2_audit_totals: Counter[str] = Counter()

    skip_reasons: Counter[str] = Counter()
    source_kinds: Counter[str] = Counter()
    archive_members: Counter[str] = Counter()
    seen_payload_digests: set[str] = set()
    raw_payload_index: list[dict[str, Any]] = []
    raw_payload_dir = output_dir / "raw_run_payloads"
    if args.materialize_run_payloads:
        raw_payload_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    basic_accepted = 0
    accepted = 0

    requested_game_mode = args.game_mode or None

    for run_source in run_sources:
        processed += 1
        source_kinds[str(run_source["kind"])] += 1
        if run_source["kind"] == "zip_member":
            archive_members[str(run_source["archive_path"].name)] += 1

        try:
            payload = _load_source_payload(run_source)
            payload_digest = _load_source_digest(payload)
        except Exception as exc:
            skip_reasons[f"digest_error:{exc.__class__.__name__}"] += 1
            print(f"[skip] {run_source['display_path']} digest_error={exc.__class__.__name__}: {exc}")
            continue

        if payload_digest in seen_payload_digests:
            skip_reasons["duplicate_payload"] += 1
            continue
        seen_payload_digests.add(payload_digest)

        try:
            bundle = _load_run_bundle(run_source, payload)
        except Exception as exc:
            skip_reasons[f"parse_error:{exc.__class__.__name__}"] += 1
            print(f"[skip] {run_source['display_path']} parse_error={exc.__class__.__name__}: {exc}")
            continue

        if args.materialize_run_payloads:
            raw_path = raw_payload_dir / f"{payload_digest}.run"
            if not raw_path.exists():
                raw_path.write_bytes(payload)

        keep, reason = _clean_run_bundle(
            bundle,
            single_player_only=args.single_player_only,
            game_mode=requested_game_mode,
        )
        summary = bundle["summary"]
        raw_payload_index.append(
            {
                "payload_sha1": payload_digest,
                "raw_run_path": str((raw_payload_dir / f"{payload_digest}.run")) if args.materialize_run_payloads else None,
                "source_kind": run_source["kind"],
                "source_display": run_source["display_path"],
                "run_id": summary.get("run_id"),
                "build_id": summary.get("build_id"),
                "character": (summary.get("characters") or [None])[0],
                "basic_filter_passed": keep,
                "quality_filter_passed": False,
            }
        )
        if not keep:
            skip_reasons[str(reason)] += 1
            continue

        samples = build_offline_training_samples(bundle)
        build_v2 = build_offline_build_v2_samples(bundle)
        basic_accepted += 1
        keep, reason = _quality_filter_bundle(
            bundle,
            samples,
            build_v2=build_v2,
            quality_profile=args.quality_profile,
            min_loss_path_points=args.min_loss_path_points,
        )
        if not keep:
            skip_reasons[str(reason)] += 1
            continue

        raw_payload_index[-1]["quality_filter_passed"] = True
        summary["ingest_source_kind"] = run_source["kind"]
        summary["ingest_source_display"] = run_source["display_path"]
        summary["payload_sha1"] = payload_digest

        accepted += 1
        runs_summary.append(summary)
        floor_records.extend(bundle["floors"])
        decision_records.extend(bundle["decisions"])
        route_samples.extend(samples["route_samples"])
        card_choice_samples.extend(samples["card_choice_samples"])
        build_samples.extend(samples["build_samples"])
        _merge_numeric_audit(build_v2_audit_totals, build_v2.get("audit"))
        for sample in samples["build_samples"]:
            decision_type = str(sample.get("decision_type") or "unknown")
            build_samples_by_type.setdefault(decision_type, []).append(sample)
        for task_name, rows in (build_v2.get("task_rows") or {}).items():
            build_v2_task_rows.setdefault(task_name, []).extend(rows)

        if accepted % 100 == 0:
            print(
                f"[progress] accepted={accepted} processed={processed} "
                f"route={len(route_samples)} card={len(card_choice_samples)} "
                f"build={len(build_samples)} build_v2={sum(len(rows) for rows in build_v2_task_rows.values())}"
            )

    filters = {
        "single_player_only": args.single_player_only,
        "game_mode": requested_game_mode,
        "max_runs": args.max_runs,
        "quality_profile": args.quality_profile,
        "min_loss_path_points": args.min_loss_path_points,
    }
    manifest = _build_manifest(
        dataset_schema="sts2_offline_run_dataset.v1",
        input_value=str(input_path),
        processed=processed,
        basic_accepted=basic_accepted,
        accepted=accepted,
        skip_reasons=dict(sorted(skip_reasons.items())),
        filters=filters,
        runs_summary=runs_summary,
        floor_records=floor_records,
        decision_records=decision_records,
        route_samples=route_samples,
        card_choice_samples=card_choice_samples,
        build_samples=build_samples,
        build_samples_by_type=build_samples_by_type,
        build_v2_task_rows=build_v2_task_rows,
        build_v2_audit=dict(sorted(build_v2_audit_totals.items())),
        extra_fields={
            "source_counts": dict(sorted(source_kinds.items())),
            "archive_member_counts": dict(sorted(archive_members.items())),
            "unique_payloads": len(seen_payload_digests),
        },
    )
    _write_dataset_bundle(
        output_dir=output_dir,
        manifest=manifest,
        runs_summary=runs_summary,
        floor_records=floor_records,
        decision_records=decision_records,
        route_samples=route_samples,
        card_choice_samples=card_choice_samples,
        build_samples=build_samples,
        build_v2_task_rows=build_v2_task_rows,
    )
    if raw_payload_index:
        _write_jsonl(output_dir / "raw_run_payloads_index.jsonl", raw_payload_index)

    run_to_build_id = {
        str(summary["run_id"]): str(summary.get("build_id") or "unknown")
        for summary in runs_summary
    }
    by_build_dir = output_dir / "by_build_id"
    by_build_dir.mkdir(parents=True, exist_ok=True)
    runs_by_build = _bucket_rows_by_build_id(runs_summary, run_to_build_id=run_to_build_id)
    floors_by_build = _bucket_rows_by_build_id(floor_records, run_to_build_id=run_to_build_id)
    decisions_by_build = _bucket_rows_by_build_id(decision_records, run_to_build_id=run_to_build_id)
    route_by_build = _bucket_rows_by_build_id(route_samples, run_to_build_id=run_to_build_id)
    card_by_build = _bucket_rows_by_build_id(card_choice_samples, run_to_build_id=run_to_build_id)
    build_by_build = _bucket_rows_by_build_id(build_samples, run_to_build_id=run_to_build_id)
    build_v2_by_build = {
        task_name: _bucket_rows_by_build_id(rows, run_to_build_id=run_to_build_id)
        for task_name, rows in build_v2_task_rows.items()
    }

    partition_index: dict[str, dict[str, Any]] = {}
    for build_id, partition_runs in sorted(runs_by_build.items()):
        safe_name = _sanitize_path_component(build_id)
        partition_dir = by_build_dir / safe_name
        partition_build_samples = build_by_build.get(build_id, [])
        partition_build_samples_by_type: dict[str, list[dict[str, Any]]] = {}
        for sample in partition_build_samples:
            decision_type = str(sample.get("decision_type") or "unknown")
            partition_build_samples_by_type.setdefault(decision_type, []).append(sample)

        partition_manifest = _build_manifest(
            dataset_schema="sts2_offline_run_dataset.v1.partition_by_build_id",
            input_value=str(input_path),
            processed=len(partition_runs),
            basic_accepted=len(partition_runs),
            accepted=len(partition_runs),
            skip_reasons={},
            filters=filters,
            runs_summary=partition_runs,
            floor_records=floors_by_build.get(build_id, []),
            decision_records=decisions_by_build.get(build_id, []),
            route_samples=route_by_build.get(build_id, []),
            card_choice_samples=card_by_build.get(build_id, []),
            build_samples=partition_build_samples,
            build_samples_by_type=partition_build_samples_by_type,
            build_v2_task_rows={
                task_name: buckets.get(build_id, [])
                for task_name, buckets in build_v2_by_build.items()
            },
            extra_fields={
                "partition_key": "build_id",
                "partition_value": build_id,
                "partition_dir": str(partition_dir),
            },
        )
        _write_dataset_bundle(
            output_dir=partition_dir,
            manifest=partition_manifest,
            runs_summary=partition_runs,
            floor_records=floors_by_build.get(build_id, []),
            decision_records=decisions_by_build.get(build_id, []),
            route_samples=route_by_build.get(build_id, []),
            card_choice_samples=card_by_build.get(build_id, []),
            build_samples=partition_build_samples,
            build_v2_task_rows={
                task_name: buckets.get(build_id, [])
                for task_name, buckets in build_v2_by_build.items()
            },
        )
        partition_index[build_id] = {
            "dir_name": safe_name,
            "dir_path": str(partition_dir),
            "runs": len(partition_runs),
            "route_samples": len(route_by_build.get(build_id, [])),
            "card_choice_samples": len(card_by_build.get(build_id, [])),
            "build_samples": len(partition_build_samples),
            "build_v2_total": sum(len(buckets.get(build_id, [])) for buckets in build_v2_by_build.values()),
        }

    manifest["by_build_id_root"] = str(by_build_dir)
    manifest["by_build_id_partitions"] = partition_index

    by_build_family_dir = output_dir / "by_build_family"
    by_build_family_dir.mkdir(parents=True, exist_ok=True)
    run_to_build_family = {
        str(summary["run_id"]): _resolve_build_family(summary.get("build_id"))
        for summary in runs_summary
    }
    runs_by_family = _bucket_rows_by_build_id(runs_summary, run_to_build_id=run_to_build_family)
    floors_by_family = _bucket_rows_by_build_id(floor_records, run_to_build_id=run_to_build_family)
    decisions_by_family = _bucket_rows_by_build_id(decision_records, run_to_build_id=run_to_build_family)
    route_by_family = _bucket_rows_by_build_id(route_samples, run_to_build_id=run_to_build_family)
    card_by_family = _bucket_rows_by_build_id(card_choice_samples, run_to_build_id=run_to_build_family)
    build_by_family = _bucket_rows_by_build_id(build_samples, run_to_build_id=run_to_build_family)
    build_v2_by_family = {
        task_name: _bucket_rows_by_build_id(rows, run_to_build_id=run_to_build_family)
        for task_name, rows in build_v2_task_rows.items()
    }

    family_index: dict[str, dict[str, Any]] = {}
    for family_name, partition_runs in sorted(runs_by_family.items()):
        safe_name = _sanitize_path_component(family_name)
        partition_dir = by_build_family_dir / safe_name
        partition_build_samples = build_by_family.get(family_name, [])
        partition_build_samples_by_type: dict[str, list[dict[str, Any]]] = {}
        for sample in partition_build_samples:
            decision_type = str(sample.get("decision_type") or "unknown")
            partition_build_samples_by_type.setdefault(decision_type, []).append(sample)

        partition_manifest = _build_manifest(
            dataset_schema="sts2_offline_run_dataset.v1.partition_by_build_family",
            input_value=str(input_path),
            processed=len(partition_runs),
            basic_accepted=len(partition_runs),
            accepted=len(partition_runs),
            skip_reasons={},
            filters=filters,
            runs_summary=partition_runs,
            floor_records=floors_by_family.get(family_name, []),
            decision_records=decisions_by_family.get(family_name, []),
            route_samples=route_by_family.get(family_name, []),
            card_choice_samples=card_by_family.get(family_name, []),
            build_samples=partition_build_samples,
            build_samples_by_type=partition_build_samples_by_type,
            build_v2_task_rows={
                task_name: buckets.get(family_name, [])
                for task_name, buckets in build_v2_by_family.items()
            },
            extra_fields={
                "partition_key": "build_family",
                "partition_value": family_name,
                "partition_dir": str(partition_dir),
            },
        )
        _write_dataset_bundle(
            output_dir=partition_dir,
            manifest=partition_manifest,
            runs_summary=partition_runs,
            floor_records=floors_by_family.get(family_name, []),
            decision_records=decisions_by_family.get(family_name, []),
            route_samples=route_by_family.get(family_name, []),
            card_choice_samples=card_by_family.get(family_name, []),
            build_samples=partition_build_samples,
            build_v2_task_rows={
                task_name: buckets.get(family_name, [])
                for task_name, buckets in build_v2_by_family.items()
            },
        )
        family_index[family_name] = {
            "dir_name": safe_name,
            "dir_path": str(partition_dir),
            "runs": len(partition_runs),
            "route_samples": len(route_by_family.get(family_name, [])),
            "card_choice_samples": len(card_by_family.get(family_name, [])),
            "build_samples": len(partition_build_samples),
            "build_v2_total": sum(len(buckets.get(family_name, [])) for buckets in build_v2_by_family.values()),
        }

    if args.materialize_run_payloads:
        manifest["raw_run_payload_root"] = str(raw_payload_dir)
        manifest["raw_run_payload_index"] = str(output_dir / "raw_run_payloads_index.jsonl")
        manifest["raw_run_payload_count"] = len(raw_payload_index)
    manifest["by_build_family_root"] = str(by_build_family_dir)
    manifest["by_build_family_partitions"] = family_index
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(
        f"[done] accepted={accepted}/{processed} "
        f"route={len(route_samples)} card={len(card_choice_samples)} build={len(build_samples)} "
        f"build_v2={sum(len(rows) for rows in build_v2_task_rows.values())} "
        f"out={output_dir}"
    )


if __name__ == "__main__":
    main()
