"""Build provenance/quality-labelled combat snapshot buckets from raw run sources.

Rules implemented for this project:

- Known zip sources are treated as pure human provenance.
- Local history runs are treated as mixed provenance and bucketed by progress:
  - low: did not clear Act 1 (never reached Act 2)
  - medium: cleared Act 1 but did not win
  - high: full run victory

We also retain a finer label so deep Act 3 losses remain distinguishable from
mid-run losses even though both map to the coarse ``medium`` bucket.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from combat_snapshot_dataset import (
    clean_combat_snapshot_rows,
    infer_encounter_tier_from_row,
    is_failed_combat_room_snapshot,
)
from export_offline_run_datasets import (
    _clean_run_bundle,
    _expand_single_input,
    _load_run_bundle,
    _load_source_digest,
    _load_source_payload,
)
from run_history_parser import build_offline_training_samples
from sts2_env.bridge_client import BridgeClient


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def _write_run_id_list(path: Path, run_ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(run_ids) + ("\n" if run_ids else ""), encoding="utf-8")


def _load_excluded_sample_ids(path: Path | None) -> set[str]:
    if path is None:
        return set()
    if not path.exists():
        raise FileNotFoundError(f"Exclude sample ids file does not exist: {path}")
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def _write_combined_snapshot_variants(
    output_dir: Path,
    *,
    run_rows: list[dict[str, Any]],
    combat_rows_all: list[dict[str, Any]],
    excluded_sample_ids: set[str] | None = None,
) -> dict[str, dict[str, Any]]:
    combined_dir = output_dir / "combined"
    combined_dir.mkdir(parents=True, exist_ok=True)
    excluded_sample_ids = set(excluded_sample_ids or set())

    def _select_runs(predicate) -> list[dict[str, Any]]:
        return [row for row in run_rows if predicate(row)]

    def _select_rows(predicate) -> list[dict[str, Any]]:
        return [row for row in combat_rows_all if predicate(row)]

    def _apply_variant_filters(
        rows: list[dict[str, Any]],
        *,
        roomwin_only: bool = False,
        subtract_excluded_sample_ids: bool = False,
    ) -> tuple[list[dict[str, Any]], dict[str, int]]:
        filtered: list[dict[str, Any]] = []
        room_failure_rows_removed = 0
        explicit_sample_exclusions_removed = 0

        for row in rows:
            if roomwin_only and is_failed_combat_room_snapshot(row):
                room_failure_rows_removed += 1
                continue

            sample_id = str(row.get("sample_id") or "")
            if subtract_excluded_sample_ids and sample_id and sample_id in excluded_sample_ids:
                explicit_sample_exclusions_removed += 1
                continue

            filtered.append(row)

        return filtered, {
            "room_failure_rows_removed": room_failure_rows_removed,
            "explicit_sample_exclusions_removed": explicit_sample_exclusions_removed,
        }

    def _write_variant(
        name: str,
        rows: list[dict[str, Any]],
        *,
        roomwin_only: bool = False,
        subtract_excluded_sample_ids: bool = False,
    ) -> dict[str, Any]:
        filtered_rows, filter_stats = _apply_variant_filters(
            rows,
            roomwin_only=roomwin_only,
            subtract_excluded_sample_ids=subtract_excluded_sample_ids,
        )
        rows = filtered_rows
        run_ids = sorted({str(row.get("run_id")) for row in rows if row.get("run_id")})
        _write_jsonl(combined_dir / f"{name}.jsonl", rows)
        _write_run_id_list(combined_dir / f"{name}.run_ids.txt", run_ids)
        potion_state_known_rows = sum(1 for row in rows if bool(row.get("potion_state_known")))
        nonempty_potion_rows = sum(
            1
            for row in rows
            if bool(row.get("potion_state_known")) and bool(row.get("potion_ids_before"))
        )
        report: dict[str, Any] = {
            "rows": len(rows),
            "run_ids": len(run_ids),
            "path": str(combined_dir / f"{name}.jsonl"),
            "run_ids_path": str(combined_dir / f"{name}.run_ids.txt"),
            "potion_state_known_rows": potion_state_known_rows,
            "potion_state_known_rate": (
                round(potion_state_known_rows / len(rows), 4)
                if rows else 0.0
            ),
            "nonempty_potion_rows": nonempty_potion_rows,
            "nonempty_potion_rate": (
                round(nonempty_potion_rows / len(rows), 4)
                if rows else 0.0
            ),
        }
        if roomwin_only or subtract_excluded_sample_ids:
            report["filters"] = {
                "roomwin_only": roomwin_only,
                "subtract_excluded_sample_ids": subtract_excluded_sample_ids,
                **filter_stats,
            }
        return report

    human_run_pred = lambda row: str(row.get("provenance_origin") or "") == "human_zip"
    local_act1clear_run_pred = (
        lambda row: str(row.get("provenance_origin") or "") == "local_history"
        and bool(row.get("cleared_act1"))
    )

    human_rows = _select_rows(human_run_pred)
    local_act1clear_rows = _select_rows(local_act1clear_run_pred)
    bootstrap_rows = human_rows + local_act1clear_rows

    weak_normal_pred = lambda row: infer_encounter_tier_from_row(row) in {"weak", "normal"}
    base_variants: dict[str, list[dict[str, Any]]] = {
        "human_only": human_rows,
        "human_only_weak_normal": [row for row in human_rows if weak_normal_pred(row)],
        "local_act1clear_only": local_act1clear_rows,
        "bootstrap_human_plus_local_act1clear": bootstrap_rows,
        "bootstrap_human_plus_local_act1clear_weak_normal": [row for row in bootstrap_rows if weak_normal_pred(row)],
    }

    report: dict[str, Any] = {}
    for name, rows in base_variants.items():
        report[name] = _write_variant(name, rows)
        report[f"{name}_roomwin_only"] = _write_variant(
            f"{name}_roomwin_only",
            rows,
            roomwin_only=True,
        )

        if excluded_sample_ids:
            report[f"{name}_minus_combat_reset_failures"] = _write_variant(
                f"{name}_minus_combat_reset_failures",
                rows,
                subtract_excluded_sample_ids=True,
            )
            report[f"{name}_roomwin_only_minus_combat_reset_failures"] = _write_variant(
                f"{name}_roomwin_only_minus_combat_reset_failures",
                rows,
                roomwin_only=True,
                subtract_excluded_sample_ids=True,
            )

    report["rules"] = {
        "human_only": "All combat rows from pure human archives.",
        "local_act1clear_only": "Mixed local-history combat rows from runs that cleared Act 1.",
        "bootstrap_human_plus_local_act1clear": (
            "Preferred combat bootstrap set: human_zip plus local_history where cleared_act1=true."
        ),
        "roomwin_only_suffix": (
            "Drops only explicit losing-room combat snapshots; keeps winning combat rooms even when the overall run later failed."
        ),
        "weak_normal_suffix": "Subset restricted to weak/normal encounters for early curriculum.",
        "minus_combat_reset_failures_suffix": (
            "Subtracts sample_ids listed in --exclude-sample-ids-file, e.g. known /env/combat_reset failure rows."
        ),
    }
    if excluded_sample_ids:
        report["exclude_sample_ids_file_count"] = len(excluded_sample_ids)
    return report


def _collect_sources(inputs: list[Path], provenance: str) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    for input_path in inputs:
        expanded = _expand_single_input(input_path) if input_path.is_file() else []
        if input_path.is_dir():
            # Reuse the same semantics as export_offline_run_datasets for directories,
            # but only for raw .run files under the provided directory.
            for run_path in sorted(input_path.rglob("*.run")):
                expanded.append(
                    {
                        "kind": "file",
                        "display_path": str(run_path),
                        "path": run_path,
                    }
                )
        for source in expanded:
            tagged = dict(source)
            tagged["provenance_origin"] = provenance
            sources.append(tagged)
    return sources


def _max_act_index(bundle: dict[str, Any]) -> int:
    floors = bundle.get("floors") or []
    max_act = 0
    for floor in floors:
        act_index = floor.get("act_index")
        if isinstance(act_index, int):
            max_act = max(max_act, act_index)
    return max_act


def _quality_labels(bundle: dict[str, Any]) -> tuple[str, str, bool, bool, int]:
    summary = bundle["summary"]
    win = bool(summary.get("win"))
    max_act = _max_act_index(bundle)
    cleared_act1 = win or max_act >= 2
    cleared_act2 = win or max_act >= 3

    if win:
        coarse = "high"
        fine = "high_win"
    elif not cleared_act1:
        coarse = "low"
        fine = "low_act1_loss"
    elif not cleared_act2:
        coarse = "medium"
        fine = "mid_act2_loss"
    else:
        coarse = "medium"
        fine = "deep_act3_loss"
    return coarse, fine, cleared_act1, cleared_act2, max_act


def _enrich_run_row(
    bundle: dict[str, Any],
    *,
    provenance_origin: str,
    quality_coarse: str,
    quality_fine: str,
    cleared_act1: bool,
    cleared_act2: bool,
    max_act_index: int,
) -> dict[str, Any]:
    summary = dict(bundle["summary"])
    summary["provenance_origin"] = provenance_origin
    summary["quality_coarse"] = quality_coarse
    summary["quality_fine"] = quality_fine
    summary["cleared_act1"] = cleared_act1
    summary["cleared_act2"] = cleared_act2
    summary["max_act_index"] = max_act_index
    return summary


def _get_live_supported_encounter_ids(session_file: str | Path | None) -> set[str]:
    client = BridgeClient(session_path=session_file)
    catalog = client.combat_catalog()
    return {
        str(entry.get("encounter_id"))
        for entry in (catalog.get("encounters") or [])
        if entry.get("encounter_id")
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build curated combat snapshot buckets.")
    parser.add_argument("--human-source", action="append", default=[],
                        help="Pure-human source path (.zip/.run/dir). Can be repeated.")
    parser.add_argument("--local-source", action="append", default=[],
                        help="Local mixed-provenance source path (.zip/.run/dir). Can be repeated.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--character", type=str, default=None)
    parser.add_argument("--single-player-only", action="store_true", default=True)
    parser.add_argument("--game-mode", type=str, default="standard")
    parser.add_argument(
        "--session-file",
        type=str,
        default=None,
        help="Optional live bridge session file. When provided, drop combat rows whose encounter_id is not present in /env/combat_catalog.",
    )
    parser.add_argument(
        "--exclude-sample-ids-file",
        type=Path,
        default=None,
        help=(
            "Optional newline-delimited sample_id list to subtract from combined variants. "
            "Used for known combat_reset failure rows."
        ),
    )
    args = parser.parse_args()

    human_inputs = [Path(value) for value in args.human_source]
    local_inputs = [Path(value) for value in args.local_source]
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    excluded_sample_ids = _load_excluded_sample_ids(args.exclude_sample_ids_file)
    supported_encounter_ids: set[str] | None = None
    if args.session_file:
        supported_encounter_ids = _get_live_supported_encounter_ids(args.session_file)

    sources = _collect_sources(human_inputs, "human_zip") + _collect_sources(local_inputs, "local_history")

    seen_digests: set[str] = set()
    seen_run_ids: set[str] = set()
    run_rows: list[dict[str, Any]] = []
    combat_rows_all: list[dict[str, Any]] = []
    bucketed_combat_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    bucketed_run_ids: dict[str, list[str]] = defaultdict(list)
    reject_reasons: Counter[str] = Counter()
    duplicate_reasons: Counter[str] = Counter()

    for source in sources:
        try:
            payload = _load_source_payload(source)
            payload_digest = _load_source_digest(payload)
        except Exception as exc:
            reject_reasons[f"payload_error:{exc.__class__.__name__}"] += 1
            continue

        provenance_origin = str(source.get("provenance_origin") or "unknown")
        if payload_digest in seen_digests:
            duplicate_reasons[f"digest_duplicate:{provenance_origin}"] += 1
            continue

        try:
            bundle = _load_run_bundle(source, payload)
        except Exception as exc:
            reject_reasons[f"parse_error:{exc.__class__.__name__}"] += 1
            continue

        keep, reason = _clean_run_bundle(
            bundle,
            single_player_only=bool(args.single_player_only),
            game_mode=args.game_mode or None,
        )
        if not keep:
            reject_reasons[str(reason)] += 1
            continue

        summary = bundle["summary"]
        run_id = str(summary.get("run_id") or "")
        if not run_id:
            reject_reasons["missing_run_id"] += 1
            continue
        if run_id in seen_run_ids:
            duplicate_reasons[f"run_id_duplicate:{provenance_origin}"] += 1
            continue

        primary_character = (summary.get("characters") or [None])[0]
        if args.character and primary_character != args.character:
            reject_reasons["character_mismatch"] += 1
            continue

        seen_digests.add(payload_digest)
        seen_run_ids.add(run_id)

        quality_coarse, quality_fine, cleared_act1, cleared_act2, max_act_index = _quality_labels(bundle)
        enriched_run = _enrich_run_row(
            bundle,
            provenance_origin=provenance_origin,
            quality_coarse=quality_coarse,
            quality_fine=quality_fine,
            cleared_act1=cleared_act1,
            cleared_act2=cleared_act2,
            max_act_index=max_act_index,
        )
        enriched_run["payload_sha1"] = payload_digest
        run_rows.append(enriched_run)

        samples = build_offline_training_samples(bundle)
        raw_combat_rows = samples.get("combat_snapshot_samples") or []
        cleaned_combat_rows, clean_report = clean_combat_snapshot_rows(
            raw_combat_rows,
            supported_encounter_ids=supported_encounter_ids,
        )
        for reason_name, count in (clean_report.get("reject_reasons") or {}).items():
            if isinstance(count, int):
                reject_reasons[f"combat_row:{reason_name}"] += count

        coarse_bucket = f"{provenance_origin}.{quality_coarse}"
        fine_bucket = f"{provenance_origin}.{quality_fine}"
        bucketed_run_ids[provenance_origin].append(run_id)
        bucketed_run_ids[coarse_bucket].append(run_id)
        bucketed_run_ids[fine_bucket].append(run_id)

        for row in cleaned_combat_rows:
            enriched_row = dict(row)
            enriched_row["provenance_origin"] = provenance_origin
            enriched_row["quality_coarse"] = quality_coarse
            enriched_row["quality_fine"] = quality_fine
            enriched_row["cleared_act1"] = cleared_act1
            enriched_row["cleared_act2"] = cleared_act2
            enriched_row["max_act_index"] = max_act_index
            enriched_row["source_display"] = summary.get("ingest_source_display") or summary.get("source_file")
            combat_rows_all.append(enriched_row)
            bucketed_combat_rows[provenance_origin].append(enriched_row)
            bucketed_combat_rows[coarse_bucket].append(enriched_row)
            bucketed_combat_rows[fine_bucket].append(enriched_row)

    run_rows.sort(key=lambda row: str(row.get("start_time") or row.get("run_id") or ""))
    combat_rows_all.sort(key=lambda row: str(row.get("sample_id") or ""))

    _write_jsonl(output_dir / "curated_runs_summary.jsonl", run_rows)
    _write_jsonl(output_dir / "combat_snapshot_samples_all.jsonl", combat_rows_all)

    for bucket_name, rows in sorted(bucketed_combat_rows.items()):
        _write_jsonl(output_dir / "combat_buckets" / f"{bucket_name}.jsonl", rows)
    for bucket_name, run_ids in sorted(bucketed_run_ids.items()):
        _write_run_id_list(output_dir / "run_id_buckets" / f"{bucket_name}.txt", sorted(run_ids))

    combined_report = _write_combined_snapshot_variants(
        output_dir,
        run_rows=run_rows,
        combat_rows_all=combat_rows_all,
        excluded_sample_ids=excluded_sample_ids,
    )

    report = {
        "character_filter": args.character,
        "source_counts": {
            "human_source_inputs": [str(path) for path in human_inputs],
            "local_source_inputs": [str(path) for path in local_inputs],
        },
        "live_catalog_filter": {
            "enabled": supported_encounter_ids is not None,
            "session_file": args.session_file,
            "supported_encounter_count": len(supported_encounter_ids or set()),
        },
        "sample_exclusion_filter": {
            "enabled": bool(excluded_sample_ids),
            "exclude_sample_ids_file": str(args.exclude_sample_ids_file) if args.exclude_sample_ids_file else None,
            "excluded_sample_id_count": len(excluded_sample_ids),
        },
        "run_counts": {
            "total_runs": len(run_rows),
            "by_provenance": dict(sorted(Counter(str(row.get("provenance_origin")) for row in run_rows).items())),
            "by_quality_coarse": dict(sorted(Counter(str(row.get("quality_coarse")) for row in run_rows).items())),
            "by_quality_fine": dict(sorted(Counter(str(row.get("quality_fine")) for row in run_rows).items())),
            "by_provenance_quality": dict(sorted(
                Counter(f"{row.get('provenance_origin')}.{row.get('quality_coarse')}" for row in run_rows).items()
            )),
        },
        "combat_snapshot_counts": {
            "total_rows": len(combat_rows_all),
            "potion_coverage": {
                "potion_state_known_rows": sum(1 for row in combat_rows_all if bool(row.get("potion_state_known"))),
                "nonempty_potion_rows": sum(
                    1
                    for row in combat_rows_all
                    if bool(row.get("potion_state_known")) and bool(row.get("potion_ids_before"))
                ),
            },
            "by_bucket": {
                bucket_name: len(rows)
                for bucket_name, rows in sorted(bucketed_combat_rows.items())
            },
        },
        "combined_variants": combined_report,
        "reject_reasons": dict(sorted(reject_reasons.items())),
        "duplicate_reasons": dict(sorted(duplicate_reasons.items())),
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
