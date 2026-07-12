"""Convert exported STS2 offline JSONL datasets into a mirrored Parquet tree.

Examples:
    python convert_offline_datasets_to_parquet.py <ARTIFACT_ROOT>/datasets
    python convert_offline_datasets_to_parquet.py <ARTIFACT_ROOT>/datasets --output-root <ARTIFACT_ROOT>/datasets/parquet
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from sts2_rl.artifacts import resolve_artifact_path, resolve_external_input_path


def _read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _value_type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, dict):
        return "dict"
    if isinstance(value, list):
        return "list"
    return type(value).__name__


def _prepare_rows_for_arrow(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    if not rows:
        return rows, []

    columns = sorted({key for row in rows for key in row.keys()})
    json_encoded_columns: list[str] = []

    for column in columns:
        type_names = {
            _value_type_name(row.get(column))
            for row in rows
            if row.get(column) is not None
        }
        if not type_names:
            continue
        needs_json_encoding = False
        if "dict" in type_names or "list" in type_names:
            needs_json_encoding = True
        elif len(type_names) > 1 and not type_names.issubset({"int", "float"}):
            needs_json_encoding = True

        if needs_json_encoding:
            json_encoded_columns.append(column)
            for row in rows:
                value = row.get(column)
                row[column] = None if value is None else json.dumps(value, ensure_ascii=False, sort_keys=True)

    return rows, json_encoded_columns


def _collect_jsonl_files(input_root: Path, output_root: Path) -> list[Path]:
    files: list[Path] = []
    for path in sorted(input_root.rglob("*.jsonl")):
        try:
            path.relative_to(output_root)
            continue
        except ValueError:
            pass
        files.append(path)
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description="Mirror STS2 offline dataset JSONL files into Parquet format.")
    parser.add_argument("input_root", type=str, help="Root dataset directory containing JSONL exports.")
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Output root for mirrored parquet files. Defaults to <input_root>/parquet.",
    )
    parser.add_argument(
        "--compression",
        type=str,
        default="zstd",
        choices=["zstd", "snappy", "gzip", "brotli", "lz4", "none"],
        help="Parquet compression codec. Default: zstd.",
    )
    args = parser.parse_args()

    input_root = resolve_external_input_path(args.input_root)
    output_root = resolve_artifact_path(args.output_root, default="datasets/parquet")
    output_root.mkdir(parents=True, exist_ok=True)

    jsonl_files = _collect_jsonl_files(input_root, output_root)
    if not jsonl_files:
        raise SystemExit(f"No JSONL files found under: {input_root}")

    compression = None if args.compression == "none" else args.compression
    manifest_rows: list[dict[str, Any]] = []

    for jsonl_path in jsonl_files:
        rows = _read_jsonl_rows(jsonl_path)
        relative_path = jsonl_path.relative_to(input_root)
        parquet_path = (output_root / relative_path).with_suffix(".parquet")
        parquet_path.parent.mkdir(parents=True, exist_ok=True)

        json_ready_rows, json_encoded_columns = _prepare_rows_for_arrow(rows)

        if json_ready_rows:
            table = pa.Table.from_pylist(json_ready_rows)
            pq.write_table(table, parquet_path, compression=compression)
            row_count = table.num_rows
        else:
            table = pa.table({})
            pq.write_table(table, parquet_path, compression=compression)
            row_count = 0

        manifest_rows.append(
            {
                "source_jsonl": str(jsonl_path),
                "output_parquet": str(parquet_path),
                "relative_path": relative_path.as_posix(),
                "row_count": row_count,
                "compression": args.compression,
                "json_encoded_columns": json_encoded_columns,
            }
        )

        print(f"[converted] {relative_path} -> {parquet_path.relative_to(output_root)} rows={row_count}")

    manifest = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "compression": args.compression,
        "file_count": len(manifest_rows),
        "files": manifest_rows,
    }
    (output_root / "parquet_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[done] parquet_files={len(manifest_rows)} out={output_root}")


if __name__ == "__main__":
    main()
