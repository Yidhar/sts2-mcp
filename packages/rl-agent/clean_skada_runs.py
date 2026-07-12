"""Clean + split Skada Analytics STS2 run dataset.

Input:  runs_full_detail.zip (7.8 GB uncompressed, 50k+ runs)
Output: data/skada_clean/{sp,mp}/{victory,failure}/shard_NNNN.jsonl

Performs:
  1. Drop runs with abandoned=True, floor_reached<3, or missing map_acts/floor_timeline
  2. Strip locale blobs (display_name, *_name_en/zh) to shrink ~3x
  3. Split by player_count: sp (=1) vs mp (>=2)
  4. Shard output 500 runs/file for downstream parallel loading

Also prints stats + distribution probes:
  - player_count histogram
  - drop reason counts
  - HP delta at rest-site floors (for rest-vs-forge label inference)
  - Whether multiplayer perspectives share visited_coords

Usage:
  python clean_skada_runs.py [--zip PATH] [--out DIR] [--limit N]
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import time
import zipfile
from pathlib import Path

from sts2_rl.artifacts import artifact_root, resolve_artifact_path, resolve_external_input_path

ARTIFACT_ROOT = artifact_root()
DEFAULT_ZIP = os.environ.get("STS2_SKADA_RUNS_ZIP")
DEFAULT_OUT = resolve_artifact_path("datasets/skada_clean", root=ARTIFACT_ROOT)

# Locale / display blobs — redundant with id fields, drop to shrink payload.
STRIP_KEYS: set[str] = {
    "display_name",
    "character_name",
    "character_display_name",
    "encounter_name",
    "encounter_display_name",
    "death_cause_name",
    "character_filter_name",
    "card_name",
    "room_type_display_name",
    "_notice",
}

# Splits for output shards
SPLITS = ("sp/victory", "sp/failure", "mp/victory", "mp/failure")


def strip_locale(obj):
    if isinstance(obj, dict):
        return {k: strip_locale(v) for k, v in obj.items() if k not in STRIP_KEYS}
    if isinstance(obj, list):
        return [strip_locale(x) for x in obj]
    return obj


def classify_drop(run_obj) -> str | None:
    """Return drop reason or None if the run passes cleaning."""
    run = run_obj.get("run") or {}
    if run.get("abandoned", False):
        return "abandoned"
    fr = run.get("floor_reached")
    if not isinstance(fr, int) or fr < 3:
        return "too_short"
    if not run_obj.get("map_acts"):
        return "no_map_acts"
    if not run_obj.get("floor_timeline"):
        return "no_floor_timeline"
    # Sanity: a playable run needs at least one combat.
    if not run_obj.get("combats"):
        return "no_combats"
    return None


class ShardWriter:
    """Writes JSONL output across rotating shard files per split."""

    def __init__(self, out_root: Path, shard_size: int = 500):
        self.out_root = out_root
        self.shard_size = shard_size
        self.shard_idx: dict[str, int] = {s: 0 for s in SPLITS}
        self.shard_lines: dict[str, int] = {s: 0 for s in SPLITS}
        self.shard_fh: dict[str, object] = {s: None for s in SPLITS}

    def write(self, split: str, obj: dict) -> None:
        if self.shard_fh[split] is None or self.shard_lines[split] >= self.shard_size:
            self._rotate(split)
        line = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
        self.shard_fh[split].write(line)
        self.shard_fh[split].write("\n")
        self.shard_lines[split] += 1

    def _rotate(self, split: str) -> None:
        if self.shard_fh[split] is not None:
            self.shard_fh[split].close()
        self.shard_idx[split] += 1
        path = self.out_root / split / f"shard_{self.shard_idx[split]:04d}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        self.shard_fh[split] = open(path, "w", encoding="utf-8")
        self.shard_lines[split] = 0

    def close(self) -> None:
        for fh in self.shard_fh.values():
            if fh is not None:
                fh.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--zip",
        default=DEFAULT_ZIP,
        required=DEFAULT_ZIP is None,
        help="Input archive (or set STS2_SKADA_RUNS_ZIP)",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--shard-size", type=int, default=500,
                        help="runs per output shard file")
    parser.add_argument("--limit", type=int, default=0,
                        help="stop after N raw lines (0 = no limit, for smoke tests)")
    args = parser.parse_args()

    input_zip = resolve_external_input_path(args.zip)
    out_root = resolve_artifact_path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    stats: collections.Counter[str] = collections.Counter()
    drop_reasons: collections.Counter[str] = collections.Counter()
    pc_dist: collections.Counter[int] = collections.Counter()
    rest_hp_deltas: list[int] = []
    perspectives_per_run: list[int] = []

    writer = ShardWriter(out_root, shard_size=args.shard_size)
    t_start = time.time()

    with zipfile.ZipFile(input_zip) as z:
        for entry in z.infolist():
            if not entry.filename.endswith(".jsonl"):
                continue
            if "SUMMARY" in entry.filename:
                continue
            outcome = "victory" if "victory" in entry.filename else "failure"
            print(f"[{time.time()-t_start:6.1f}s] reading {entry.filename} "
                  f"(compressed {entry.compress_size/1e6:.0f} MB)", flush=True)

            with z.open(entry) as f:
                for raw in f:
                    if args.limit and stats["raw_lines"] >= args.limit:
                        break
                    line = raw.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue
                    stats["raw_lines"] += 1

                    try:
                        obj = json.loads(line)
                    except Exception:
                        stats["parse_error"] += 1
                        continue

                    reason = classify_drop(obj)
                    if reason:
                        drop_reasons[reason] += 1
                        continue

                    run = obj["run"]
                    pc = int(run.get("player_count") or 1)
                    pc_dist[pc] += 1
                    perspectives_per_run.append(len(obj.get("perspectives") or []))

                    # Record rest-floor HP deltas (for rest-vs-forge label probe)
                    for ft in obj.get("floor_timeline") or []:
                        if ft.get("room_type") in ("R", "REST", "CAMPFIRE"):
                            hp_b = ft.get("hp_before")
                            hp_a = ft.get("hp_after")
                            if isinstance(hp_b, int) and isinstance(hp_a, int):
                                rest_hp_deltas.append(hp_a - hp_b)

                    mode = "sp" if pc == 1 else "mp"
                    split = f"{mode}/{outcome}"
                    clean = strip_locale(obj)
                    writer.write(split, clean)
                    stats[f"kept_{split}"] += 1

                    if stats["raw_lines"] % 2000 == 0:
                        elapsed = time.time() - t_start
                        rate = stats["raw_lines"] / max(elapsed, 1e-6)
                        print(f"[{elapsed:6.1f}s] processed {stats['raw_lines']} "
                              f"lines ({rate:.0f}/s), kept_sp_v={stats['kept_sp/victory']} "
                              f"kept_mp_v={stats['kept_mp/victory']}", flush=True)
            if args.limit and stats["raw_lines"] >= args.limit:
                break

    writer.close()
    elapsed = time.time() - t_start

    # ----- Report -----
    print("\n" + "=" * 50)
    print(f"DONE in {elapsed:.1f}s  ({stats['raw_lines']/max(elapsed,1):.0f} runs/sec)")
    print("=" * 50)
    print(f"raw lines:       {stats['raw_lines']:,}")
    print(f"parse errors:    {stats['parse_error']:,}")
    print(f"dropped:")
    for reason, count in drop_reasons.most_common():
        print(f"  {reason:20s} {count:>8,}")
    print(f"kept by split:")
    total_kept = 0
    for split in SPLITS:
        c = stats[f"kept_{split}"]
        total_kept += c
        print(f"  {split:20s} {c:>8,}")
    print(f"  {'TOTAL KEPT':20s} {total_kept:>8,}")
    print()
    print(f"player_count distribution:")
    for pc, count in sorted(pc_dist.items()):
        print(f"  pc={pc}: {count:>8,}")
    if perspectives_per_run:
        pcp = collections.Counter(perspectives_per_run)
        print(f"perspectives_per_run distribution (first 8):")
        for n, c in sorted(pcp.items())[:8]:
            print(f"  n={n}: {c:>8,}")
    print()
    # Rest-floor HP delta probe
    if rest_hp_deltas:
        deltas = sorted(rest_hp_deltas)
        n = len(deltas)
        print(f"rest-floor HP delta ({n:,} samples):")
        print(f"  min  = {deltas[0]}")
        print(f"  p10  = {deltas[n//10]}")
        print(f"  p50  = {deltas[n//2]}")
        print(f"  p90  = {deltas[n*9//10]}")
        print(f"  max  = {deltas[-1]}")
        rested = sum(1 for d in deltas if d > 0)
        forged = sum(1 for d in deltas if d == 0)
        hp_loss = sum(1 for d in deltas if d < 0)
        print(f"  hp_delta>0 (rested):  {rested:>8,} ({rested/n*100:.1f}%)")
        print(f"  hp_delta=0 (forged?): {forged:>8,} ({forged/n*100:.1f}%)")
        print(f"  hp_delta<0 (lost hp): {hp_loss:>8,} ({hp_loss/n*100:.1f}%)")


if __name__ == "__main__":
    main()
