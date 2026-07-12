"""Export compact run provenance rows for manual human/AI curation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Export run provenance audit table.")
    parser.add_argument("--runs-summary", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--character", type=str, default=None)
    args = parser.parse_args()

    args.runs_summary = resolve_external_input_path(args.runs_summary)
    args.output_csv = resolve_artifact_path(args.output_csv)

    rows = read_jsonl(args.runs_summary)
    out_rows: list[dict[str, Any]] = []
    for row in rows:
        characters = row.get("characters") or []
        primary_character = characters[0] if characters else None
        if args.character and primary_character != args.character:
            continue
        out_rows.append(
            {
                "run_id": row.get("run_id"),
                "start_time": row.get("start_time"),
                "character": primary_character,
                "build_id": row.get("build_id"),
                "win": row.get("win"),
                "path_point_count": row.get("path_point_count"),
                "run_time_seconds": row.get("run_time_seconds"),
                "killed_by_encounter": row.get("killed_by_encounter"),
                "source_file": row.get("source_file"),
                "human_label": "",
                "notes": "",
            }
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out_rows[0].keys()) if out_rows else [])
        if out_rows:
            writer.writeheader()
            writer.writerows(out_rows)
    print(f"wrote {len(out_rows)} rows -> {args.output_csv}")


if __name__ == "__main__":
    main()
