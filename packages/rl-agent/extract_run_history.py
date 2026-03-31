"""Extract normalized data from Slay the Spire 2 native .run history files.

Examples:
    python extract_run_history.py C:/path/to/1773975929.run
    python extract_run_history.py C:/path/to/history --output-dir tmp/history_extract
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_history_parser import extract_run_history, write_extracted_run_history


def _collect_run_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(input_path.rglob("*.run"))
    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract summary/floor/decision views from STS2 native .run history files."
    )
    parser.add_argument("input", type=str, help="Path to a .run file or a directory containing .run files.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to write extracted artifacts. If omitted, only a JSON summary is printed.",
    )
    parser.add_argument(
        "--print-full-summary",
        action="store_true",
        help="Print the full summary JSON instead of the compact one-line summary.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    run_files = _collect_run_files(input_path)
    if not run_files:
        raise SystemExit(f"No .run files found under: {input_path}")

    output_root = Path(args.output_dir) if args.output_dir else None

    for run_file in run_files:
        bundle = extract_run_history(run_file)
        summary = bundle["summary"]
        characters = ",".join(str(character) for character in summary["characters"])

        print(
            f"[extract] {run_file.name} | character={characters} "
            f"| win={summary['win']} | seed={summary['seed']} | "
            f"path_points={summary['path_point_count']} | "
            f"floors={len(bundle['floors'])} | decisions={len(bundle['decisions'])}"
        )

        if output_root is not None:
            run_out_dir = output_root / run_file.stem
            written = write_extracted_run_history(bundle, run_out_dir)
            print(f"  wrote summary={written['summary']}")
            print(f"  wrote final_build={written['final_build']}")
            print(f"  wrote floors={written['floors']}")
            print(f"  wrote decisions={written['decisions']}")
            print(f"  wrote full={written['full']}")
        elif args.print_full_summary:
            print(json.dumps(summary, ensure_ascii=False, indent=2))
        else:
            compact = {
                "run_id": summary["run_id"],
                "source_file": summary["source_file"],
                "characters": summary["characters"],
                "win": summary["win"],
                "killed_by_encounter": summary["killed_by_encounter"],
                "seed": summary["seed"],
                "build_id": summary["build_id"],
                "path_point_count": summary["path_point_count"],
                "player_count": summary["player_count"],
            }
            print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
