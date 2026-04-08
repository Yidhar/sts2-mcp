"""Clean exported combat snapshot datasets down to structurally playable rows.

This rewrites every ``combat_snapshot_samples.jsonl`` under a dataset root,
refreshes the mirrored parquet files, and updates manifest counts so combat
sandbox training only samples rows that the current bridge contract can
faithfully reconstruct as an opening combat state.

Example:
    python clean_combat_snapshot_dataset.py E:/game/project/sts2_mcp/datasets
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from combat_snapshot_dataset import clean_combat_snapshot_rows
from convert_offline_datasets_to_parquet import _prepare_rows_for_arrow


def _read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_jsonl_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def _write_parquet_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    parquet_rows = [dict(row) for row in rows]
    parquet_rows, _ = _prepare_rows_for_arrow(parquet_rows)
    if parquet_rows:
        table = pa.Table.from_pylist(parquet_rows)
    else:
        table = pa.table({})
    pq.write_table(table, path, compression="zstd")


def _update_manifest(
    manifest_path: Path,
    *,
    row_count: int,
    source_run_count: int,
    clean_report: dict[str, Any],
    dataset_root: Path,
) -> None:
    if not manifest_path.exists():
        return

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_counts = manifest.get("dataset_counts")
    if isinstance(dataset_counts, dict):
        dataset_counts["combat_snapshot_samples"] = row_count

    manifest["combat_snapshot_source_runs"] = source_run_count
    manifest["combat_snapshot_filter_scope"] = "basic_filters_plus_strict_playable_rows"
    manifest["combat_snapshot_cleaning"] = {
        **clean_report,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    if manifest_path.parent == dataset_root:
        for key, subdir in (
            ("by_build_id_partitions", dataset_root / "by_build_id"),
            ("by_build_family_partitions", dataset_root / "by_build_family"),
        ):
            partitions = manifest.get(key)
            if not isinstance(partitions, dict):
                continue
            for partition_name, partition_meta in partitions.items():
                if not isinstance(partition_meta, dict):
                    continue
                jsonl_path = subdir / partition_name / "combat_snapshot_samples.jsonl"
                if jsonl_path.exists():
                    count = sum(1 for _ in jsonl_path.open("r", encoding="utf-8") if _.strip())
                else:
                    count = 0
                partition_meta["combat_snapshot_samples"] = count

    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def _update_parquet_manifest(
    parquet_manifest_path: Path,
    file_row_counts: dict[str, int],
) -> None:
    if not parquet_manifest_path.exists():
        return

    manifest = json.loads(parquet_manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files")
    if not isinstance(files, list):
        return

    for entry in files:
        if not isinstance(entry, dict):
            continue
        relative_path = entry.get("relative_path")
        if not isinstance(relative_path, str):
            continue
        if relative_path in file_row_counts:
            entry["row_count"] = file_row_counts[relative_path]

    parquet_manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean combat snapshot datasets to playable rows only.")
    parser.add_argument("dataset_root", type=str, help="Root directory of the exported dataset bundle.")
    parser.add_argument(
        "--supported-encounters-from-bridge",
        action="store_true",
        help="If set, query the live STS2 bridge combat catalog and drop snapshot rows whose encounter_id is not currently supported.",
    )
    parser.add_argument(
        "--session-file",
        type=str,
        default=None,
        help="Optional bridge session file path used with --supported-encounters-from-bridge.",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    if not dataset_root.exists():
        raise SystemExit(f"Dataset root does not exist: {dataset_root}")

    jsonl_files = sorted(
        path for path in dataset_root.rglob("combat_snapshot_samples.jsonl")
        if "parquet" not in {part.lower() for part in path.parts}
    )
    if not jsonl_files:
        raise SystemExit(f"No combat_snapshot_samples.jsonl files found under: {dataset_root}")

    supported_encounter_ids: set[str] | None = None
    if args.supported_encounters_from_bridge:
        from sts2_env.bridge_client import BridgeClient

        client = BridgeClient(session_path=args.session_file)
        catalog = client.combat_catalog()
        supported_encounter_ids = {
            str(entry.get("encounter_id"))
            for entry in (catalog.get("encounters") or [])
            if entry.get("encounter_id")
        }
        print(f"[catalog] supported_encounters={len(supported_encounter_ids)}")

    aggregate_report: dict[str, Any] = {
        "dataset_root": str(dataset_root),
        "processed_files": [],
        "totals": {
            "input_rows_root_dataset": 0,
            "kept_rows_root_dataset": 0,
            "dropped_rows_root_dataset": 0,
            "source_runs": 0,
        },
        "materialized_file_totals": {
            "input_rows": 0,
            "kept_rows": 0,
            "dropped_rows": 0,
        },
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    parquet_row_counts: dict[str, int] = {}
    global_source_runs: set[str] = set()
    clean_reports_by_relative_path: dict[str, dict[str, Any]] = {}

    for jsonl_path in jsonl_files:
        rows = _read_jsonl_rows(jsonl_path)
        cleaned_rows, clean_report = clean_combat_snapshot_rows(
            rows,
            supported_encounter_ids=supported_encounter_ids,
        )
        _write_jsonl_rows(jsonl_path, cleaned_rows)

        relative_jsonl = jsonl_path.relative_to(dataset_root).as_posix()
        parquet_path = (dataset_root / "parquet" / jsonl_path.relative_to(dataset_root)).with_suffix(".parquet")
        _write_parquet_rows(parquet_path, cleaned_rows)
        parquet_row_counts[relative_jsonl] = len(cleaned_rows)
        clean_reports_by_relative_path[relative_jsonl] = clean_report

        source_runs = len({str(row.get("run_id")) for row in cleaned_rows if row.get("run_id") is not None})
        global_source_runs.update(str(row.get("run_id")) for row in cleaned_rows if row.get("run_id") is not None)
        _update_manifest(
            jsonl_path.parent / "manifest.json",
            row_count=len(cleaned_rows),
            source_run_count=source_runs,
            clean_report=clean_report,
            dataset_root=dataset_root,
        )

        aggregate_report["processed_files"].append(
            {
                "relative_jsonl": relative_jsonl,
                "relative_parquet": parquet_path.relative_to(dataset_root / "parquet").as_posix(),
                "source_runs": source_runs,
                **clean_report,
            }
        )
        aggregate_report["materialized_file_totals"]["input_rows"] += int(clean_report.get("input_rows", 0) or 0)
        aggregate_report["materialized_file_totals"]["kept_rows"] += int(clean_report.get("kept_rows", 0) or 0)
        aggregate_report["materialized_file_totals"]["dropped_rows"] += int(clean_report.get("dropped_rows", 0) or 0)

        print(
            f"[cleaned] {relative_jsonl} kept={clean_report.get('kept_rows')} "
            f"dropped={clean_report.get('dropped_rows')} source_runs={source_runs}"
        )

    aggregate_report["totals"]["source_runs"] = len(global_source_runs)
    root_jsonl_path = dataset_root / "combat_snapshot_samples.jsonl"
    if root_jsonl_path.exists():
        root_rows = _read_jsonl_rows(root_jsonl_path)
        root_clean_report = clean_reports_by_relative_path.get("combat_snapshot_samples.jsonl") or clean_combat_snapshot_rows(
            root_rows,
            supported_encounter_ids=supported_encounter_ids,
        )[1]
        aggregate_report["totals"]["input_rows_root_dataset"] = int(root_clean_report.get("input_rows", 0) or 0)
        aggregate_report["totals"]["kept_rows_root_dataset"] = int(root_clean_report.get("kept_rows", 0) or 0)
        aggregate_report["totals"]["dropped_rows_root_dataset"] = int(root_clean_report.get("dropped_rows", 0) or 0)
        _update_manifest(
            dataset_root / "manifest.json",
            row_count=len(root_rows),
            source_run_count=len({str(row.get("run_id")) for row in root_rows if row.get("run_id") is not None}),
            clean_report=root_clean_report,
            dataset_root=dataset_root,
        )
    (dataset_root / "combat_snapshot_cleaning_report.json").write_text(
        json.dumps(aggregate_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    _update_parquet_manifest(dataset_root / "parquet" / "parquet_manifest.json", parquet_row_counts)
    print(
        f"[done] files={len(jsonl_files)} kept={aggregate_report['totals']['kept_rows_root_dataset']} "
        f"dropped={aggregate_report['totals']['dropped_rows_root_dataset']}"
    )


if __name__ == "__main__":
    main()
