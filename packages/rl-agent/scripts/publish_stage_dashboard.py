#!/usr/bin/env python3
"""Dashboard sidecar: translate stage-2/4 pipeline metrics for the panel.

The monitoring panel discovers ``runs/<lineage>/run-<uuid>/metrics.jsonl``
whose first line is a ``run_start`` event. The isolated-stage pipeline
writes its own jsonl formats, so this sidecar follows those files and
republishes dashboard-schema events (``train_episode`` per collected
episode, ``learner_update`` per trainer ingest batch). Read-only toward
the pipeline; safe to start and stop at any time.
"""

from __future__ import annotations

import argparse
import json
import time
import uuid
from pathlib import Path
from typing import Any


def _read_new_lines(path: Path, offset: int) -> tuple[list[dict[str, Any]], int]:
    if not path.is_file():
        return [], offset
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        handle.seek(offset)
        for line in handle:
            if not line.endswith("\n"):
                break  # partial write: retry next poll
            offset += len(line.encode("utf-8"))
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows, offset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-dir", required=True)
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--lineage", default="stage2-isolated-macro-v1")
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    args = parser.parse_args()

    pipeline = Path(args.pipeline_dir)
    run_id = str(uuid.uuid4())
    run_directory = Path(args.runs_root) / args.lineage / f"run-{run_id}"
    run_directory.mkdir(parents=True, exist_ok=True)
    metrics_path = run_directory / "metrics.jsonl"

    environment_steps = 0
    episodes = 0
    offsets: dict[Path, int] = {}
    sources = [pipeline / f"collector-{index}-metrics.jsonl" for index in (1, 2, 3)]
    trainer_source = pipeline / "trainer-metrics.jsonl"

    with metrics_path.open("a", encoding="utf-8") as out:
        out.write(
            json.dumps(
                {
                    "event": "run_start",
                    "run_id": run_id,
                    "unix_s": time.time(),
                    "lineage": args.lineage,
                    "note": "stage-2/4 isolated pipeline sidecar view",
                    "state": {"environment_steps": 0, "episodes": 0},
                }
            )
            + "\n"
        )
        out.flush()
        while True:
            wrote = False
            for source in sources:
                rows, offsets[source] = _read_new_lines(
                    source, offsets.get(source, 0)
                )
                for row in rows:
                    if row.get("event") != "collector_episode":
                        continue
                    episodes += 1
                    environment_steps += int(row.get("steps") or 0)
                    out.write(
                        json.dumps(
                            {
                                "event": "train_episode",
                                "unix_s": row.get("unix_s"),
                                "environment_steps": environment_steps,
                                "episode_id": (
                                    f"c{row.get('seed')}-{row.get('episode_index')}"
                                ),
                                "reset_seed": row.get("seed"),
                                "steps": row.get("steps"),
                                "max_floor": row.get("max_floor"),
                                "run_won": row.get("run_won"),
                                "terminal_reason": (
                                    "run_victory"
                                    if row.get("run_won")
                                    else "run_defeat"
                                ),
                            }
                        )
                        + "\n"
                    )
                    wrote = True
            rows, offsets[trainer_source] = _read_new_lines(
                trainer_source, offsets.get(trainer_source, 0)
            )
            for row in rows:
                if row.get("event") != "trainer_ingest":
                    continue
                learner = row.get("learner") or {}
                out.write(
                    json.dumps(
                        {
                            "event": "learner_update",
                            "unix_s": row.get("unix_s"),
                            "environment_steps": environment_steps,
                            "ingested_episodes": row.get("ingested_total"),
                            "loss": learner.get("loss"),
                            "td_error_mean": learner.get("td_error_mean"),
                            "updates": learner.get("updates"),
                        }
                    )
                    + "\n"
                )
                wrote = True
            if wrote:
                out.flush()
            time.sleep(max(args.poll_seconds, 1.0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
