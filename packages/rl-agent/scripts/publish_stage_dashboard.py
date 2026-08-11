#!/usr/bin/env python3
"""Publish one isolated pipeline to the monitor without replay on restart."""

from __future__ import annotations

import argparse
import json
import time
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any, TextIO

DASHBOARD_STATE_FORMAT = "sts2-stage2-dashboard-state-v1"
PIPELINE_FORMAT = "sts2-stage2-pipeline-v1"


def _read_new_lines(path: Path, offset: int) -> tuple[list[dict[str, Any]], int]:
    if not path.is_file():
        return [], offset
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        handle.seek(offset)
        for line in handle:
            if not line.endswith("\n"):
                break
            offset += len(line.encode("utf-8"))
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows, offset


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(payload), sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _pipeline_manifest(pipeline: Path) -> dict[str, Any]:
    path = pipeline / "pipeline.json"
    if not path.is_file():
        raise FileNotFoundError(f"pipeline manifest does not exist: {path}")
    row = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(row, dict) or row.get("format") != PIPELINE_FORMAT:
        raise ValueError(f"invalid pipeline manifest: {path}")
    producer_ids = row.get("producer_ids")
    if not isinstance(producer_ids, list) or not producer_ids:
        raise ValueError("pipeline manifest has no producers")
    if row.get("control_domain") not in {"macro", "combat"}:
        raise ValueError("pipeline manifest has no valid control domain")
    return row


def _load_or_create_state(
    pipeline: Path,
    runs_root: Path,
    *,
    lineage: str,
    manifest: Mapping[str, Any],
) -> tuple[Path, dict[str, Any], bool]:
    state_path = pipeline / "dashboard-state.json"
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if not isinstance(state, dict) or state.get("format") != DASHBOARD_STATE_FORMAT:
            raise ValueError(f"invalid dashboard state: {state_path}")
        if state.get("pipeline_id") != manifest.get("pipeline_id"):
            raise ValueError("dashboard state belongs to another pipeline")
        return Path(state["metrics_path"]), state, False

    run_id = str(uuid.uuid4())
    run_directory = runs_root / lineage / f"run-{run_id}"
    run_directory.mkdir(parents=True, exist_ok=False)
    metrics_path = run_directory / "metrics.jsonl"
    source_names = [f"{producer_id}-metrics.jsonl" for producer_id in manifest["producer_ids"]]
    source_names.append("trainer-metrics.jsonl")
    state = {
        "format": DASHBOARD_STATE_FORMAT,
        "pipeline_id": manifest["pipeline_id"],
        "run_id": run_id,
        "metrics_path": str(metrics_path.resolve()),
        "environment_steps": 0,
        "episodes": 0,
        "trainer_base_initialized": False,
        "offsets": {name: 0 for name in source_names},
        "terminal": False,
    }
    _atomic_json(state_path, state)
    return metrics_path, state, True


def _publish_available(
    pipeline: Path,
    manifest: Mapping[str, Any],
    state: dict[str, Any],
    out: TextIO,
) -> bool:
    """Publish all currently complete rows; return True at trainer terminal."""

    offsets = state["offsets"]
    trainer_name = "trainer-metrics.jsonl"
    trainer_rows, offsets[trainer_name] = _read_new_lines(pipeline / trainer_name, int(offsets.get(trainer_name, 0)))
    for row in trainer_rows:
        if row.get("event") == "trainer_start" and not state.get("trainer_base_initialized"):
            state["environment_steps"] = int(row.get("environment_steps") or 0)
            state["trainer_base_initialized"] = True

    # Establish the cumulative base before publishing producer episodes. This
    # keeps exact-continuation plots in the same coordinate system.
    if not state.get("trainer_base_initialized"):
        return False

    collected: list[dict[str, Any]] = []
    for producer_id in manifest["producer_ids"]:
        name = f"{producer_id}-metrics.jsonl"
        rows, offsets[name] = _read_new_lines(pipeline / name, int(offsets.get(name, 0)))
        collected.extend(row for row in rows if row.get("event") == "collector_episode")

    for row in sorted(collected, key=lambda item: float(item.get("unix_s") or 0.0)):
        state["episodes"] = int(state["episodes"]) + 1
        state["environment_steps"] = int(state["environment_steps"]) + int(row.get("steps") or 0)
        out.write(
            json.dumps(
                {
                    "event": "train_episode",
                    "unix_s": row.get("unix_s"),
                    "environment_steps": state["environment_steps"],
                    "episode_id": (f"{row.get('producer_id')}:{row.get('run_nonce')}:{row.get('episode_index')}"),
                    "reset_seed": row.get("seed"),
                    "steps": row.get("steps"),
                    "max_floor": row.get("max_floor"),
                    "max_act": row.get("max_act"),
                    "act1_cleared": row.get("act1_cleared"),
                    "revivals_used": row.get("revivals_used"),
                    "reward_total": row.get("reward_total"),
                    "run_won": row.get("run_won"),
                    "terminal_reason": row.get("terminal_reason")
                    or ("run_victory" if row.get("run_won") else "run_defeat"),
                }
            )
            + "\n"
        )

    terminal = False
    for row in trainer_rows:
        event = row.get("event")
        if event == "trainer_ingest":
            learner = row.get("learner") or {}
            out.write(
                json.dumps(
                    {
                        "event": "learner_update",
                        "unix_s": row.get("unix_s"),
                        # Use trainer-owned consumed progress, not whichever
                        # collector rows the sidecar happened to read first.
                        "environment_steps": row.get("environment_steps"),
                        "ingested_episodes": row.get("ingested_total"),
                        "loss": learner.get("loss"),
                        "td_error_mean": learner.get("td_error_mean"),
                        "updates": learner.get("updates"),
                    }
                )
                + "\n"
            )
        elif event == "trainer_complete":
            out.write(
                json.dumps(
                    {
                        "event": "run_complete",
                        "unix_s": row.get("unix_s"),
                        "environment_steps": row.get("environment_steps"),
                        "episodes": row.get("ingested_total"),
                        "learner_updates": (row.get("learner") or {}).get("updates"),
                        "completion_status": "horizon_complete",
                    }
                )
                + "\n"
            )
            terminal = True
        elif event == "trainer_failed":
            out.write(
                json.dumps(
                    {
                        "event": "run_failed",
                        "unix_s": row.get("unix_s"),
                        "environment_steps": row.get("environment_steps"),
                        "error": row.get("error"),
                    }
                )
                + "\n"
            )
            terminal = True
    state["terminal"] = terminal or bool(state.get("terminal"))
    return terminal


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-dir", required=True)
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--lineage", default=None)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    args = parser.parse_args()

    pipeline = Path(args.pipeline_dir)
    manifest = _pipeline_manifest(pipeline)
    lineage = str(args.lineage or manifest["lineage_id"])
    metrics_path, state, created = _load_or_create_state(
        pipeline,
        Path(args.runs_root),
        lineage=lineage,
        manifest=manifest,
    )
    state_path = pipeline / "dashboard-state.json"
    if state.get("terminal"):
        return 0

    with metrics_path.open("a", encoding="utf-8") as out:
        if created:
            out.write(
                json.dumps(
                    {
                        "event": "run_start",
                        "run_id": state["run_id"],
                        "unix_s": time.time(),
                        "lineage": lineage,
                        "pipeline_id": manifest["pipeline_id"],
                        "control_domain": manifest["control_domain"],
                        "note": "stage-2/4 isolated pipeline sidecar view",
                        "state": {"environment_steps": 0, "episodes": 0},
                    }
                )
                + "\n"
            )
            out.flush()
        while True:
            terminal = _publish_available(pipeline, manifest, state, out)
            out.flush()
            _atomic_json(state_path, state)
            if terminal:
                return 0
            time.sleep(max(args.poll_seconds, 1.0))


if __name__ == "__main__":
    raise SystemExit(main())
