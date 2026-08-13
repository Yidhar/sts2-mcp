#!/usr/bin/env python3
"""Project live Hybrid Candidate-Q telemetry into the dashboard event schema."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--lineage", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--target-runs", type=int, required=True)
    parser.add_argument("--collector-episode-capacity", type=int, required=True)
    parser.add_argument("--collector-count", type=int, default=1)
    parser.add_argument("--collector-device", default="cpu")
    parser.add_argument("--maximum-pending-runs", type=int, required=True)
    parser.add_argument("--revival-budget", type=int, required=True)
    parser.add_argument("--curriculum-mode", default="hybrid-candidate-q")
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    result: list[dict[str, Any]] = []
    for line in lines:
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            result.append(value)
    return result


def _start_time(manifest: dict[str, Any], trainer_rows: list[dict[str, Any]]) -> float:
    created = manifest.get("created_at")
    if isinstance(created, str):
        try:
            return datetime.fromisoformat(created).timestamp()
        except ValueError:
            pass
    if trainer_rows and isinstance(trainer_rows[0].get("unix_s"), int | float):
        return float(trainer_rows[0]["unix_s"])
    return time.time()


def _ingested_episodes(
    collector_rows: list[dict[str, Any]],
    *,
    ingested_runs: int,
    resumed_from: int,
) -> list[dict[str, Any]]:
    rows = [row for row in collector_rows if row.get("event") == "hybrid_collector_episode"]
    if resumed_from <= 0:
        return rows[:ingested_runs]
    reset_indexes = [
        index for index, row in enumerate(rows) if row.get("episode_index") == 0
    ]
    if len(reset_indexes) < 2:
        return rows[:ingested_runs]
    current_segment = rows[reset_indexes[-1] :]
    return rows[:resumed_from] + current_segment[: max(0, ingested_runs - resumed_from)]


def _acknowledged_episodes(
    trainer_rows: list[dict[str, Any]],
    collector_rows: list[dict[str, Any]],
    *,
    ingested_runs: int,
    resumed_from: int,
) -> list[dict[str, Any]]:
    """Prefer learner acknowledgements; retain old single-producer history."""

    acknowledged = [
        dict(row["episode"])
        for row in trainer_rows
        if row.get("event") == "hybrid_update" and isinstance(row.get("episode"), dict)
        and row["episode"]
    ]
    if acknowledged:
        return acknowledged[: max(0, ingested_runs - resumed_from)]
    return _ingested_episodes(
        collector_rows,
        ingested_runs=ingested_runs,
        resumed_from=resumed_from,
    )


def _projection(args: argparse.Namespace) -> list[dict[str, Any]]:
    run_root = Path(args.run_root)
    manifest = _read_json(run_root / "launcher" / "launch.json")
    trainer_rows = _read_jsonl(run_root / "metrics" / "trainer.jsonl")
    collector_rows = [
        row
        for path in sorted((run_root / "metrics").glob("collector-*.jsonl"))
        for row in _read_jsonl(path)
    ]
    collector_rows.sort(key=lambda row: float(row.get("unix_s", 0.0) or 0.0))
    trainer_status = _read_json(run_root / "status" / "trainer.json")
    collector_statuses = [
        _read_json(path) for path in sorted((run_root / "status").glob("collector-*.json"))
    ]
    latest = trainer_rows[-1] if trainer_rows else {}
    learner = latest.get("learner") if isinstance(latest.get("learner"), dict) else {}
    resumed_from = int(manifest.get("resumed_from_ingested_runs", 0) or 0)
    ingested = int(latest.get("ingested_runs", resumed_from) or resumed_from)
    source_environment_steps = int(manifest.get("source_environment_steps", 0) or 0)
    environment_steps = int(
        latest.get("environment_steps", source_environment_steps)
        or source_environment_steps
    )
    source_updates = int(manifest.get("source_learner_updates", 0) or 0)
    learner_updates = int(learner.get("updates", source_updates) or source_updates)
    events: list[dict[str, Any]] = [
        {
            "event": "run_start",
            "unix_s": _start_time(manifest, trainer_rows),
            "run_id": args.run_id,
            "pipeline": "hybrid-double-q-v1",
            "checkpoint_load": {
                "mode": "exact_resume" if resumed_from else "model_init",
                "parent_checkpoint": (
                    manifest.get("source_checkpoint")
                    or manifest.get("source_hybrid_publication")
                    or manifest.get("champion")
                ),
                "source_policy_version": manifest.get("source_policy_version"),
            },
            "config": {
                "profile": "hybrid-candidate-q-v1",
                "curriculum": {
                    "mode": args.curriculum_mode,
                    "revival_budget": args.revival_budget,
                },
                "runtime": {
                    "device": "cuda",
                    "collector_device": args.collector_device,
                    "collector_workers": args.collector_count,
                    "total_environment_steps": 0,
                    "target_training_episodes": args.target_runs,
                    "collector_episode_capacity": args.collector_episode_capacity,
                    "maximum_pending_runs": args.maximum_pending_runs,
                },
                "model": {"architecture": "hybrid-domain-candidate-q-v1"},
            },
            "state": {
                "environment_steps": source_environment_steps,
                "episodes": resumed_from,
                "learner_updates": source_updates,
                "policy_version": source_updates,
                "evaluation_episodes": 0,
            },
        }
    ]
    trainer_by_run = {
        int(row.get("ingested_runs", 0) or 0): row
        for row in trainer_rows
        if row.get("event") == "hybrid_update"
    }
    cumulative_steps = source_environment_steps
    for episode_number, row in enumerate(
        _acknowledged_episodes(
            trainer_rows,
            collector_rows,
            ingested_runs=ingested,
            resumed_from=resumed_from,
        ),
        start=resumed_from + 1,
    ):
        cumulative_steps += int(row.get("steps", 0) or 0)
        matching_update = trainer_by_run.get(episode_number, {})
        events.append(
            {
                "event": "train_episode",
                "unix_s": matching_update.get("unix_s"),
                "environment_steps": int(
                    matching_update.get("environment_steps", cumulative_steps) or 0
                ),
                "episode_id": f"hybrid-run-{episode_number}",
                "reset_seed": row.get("reset_seed"),
                "producer_id": row.get("producer_id"),
                "max_floor": row.get("max_floor"),
                "revivals_used": row.get("revivals_used"),
                "steps": row.get("steps"),
                "terminal_reason": row.get("terminal_reason"),
                "run_won": row.get("run_won"),
            }
        )
    pending = len(tuple((run_root / "spool").glob("*.pkl")))
    for row in trainer_rows:
        if row.get("event") != "hybrid_update":
            continue
        item = row.get("learner") if isinstance(row.get("learner"), dict) else {}
        item_replay = row.get("replay") if isinstance(row.get("replay"), dict) else {}
        events.append(
            {
                "event": "learner_update",
                "unix_s": row.get("unix_s"),
                "environment_steps": row.get("environment_steps"),
                "policy_version": item.get("updates"),
                "learner_updates": item.get("updates"),
                "loss": item.get("loss"),
                "gradient_norm": item.get("gradient_norm"),
                "maximum_policy_lag": item.get("maximum_policy_lag"),
                "hybrid_macro_loss": item.get("macro_loss"),
                "hybrid_combat_loss": item.get("combat_loss"),
                "hybrid_macro_td_abs": item.get("macro_td_abs"),
                "hybrid_combat_td_abs": item.get("combat_td_abs"),
                "hybrid_shared_gradient_cosine": item.get("shared_gradient_cosine"),
                "rollout_queue": {
                    "size": int(row.get("spool_pending", pending) or 0),
                    "capacity": int(
                        row.get("spool_capacity", args.maximum_pending_runs)
                        or args.maximum_pending_runs
                    ),
                },
                "actor_progress": {
                    "phase": "collecting",
                    "decision_domain": "joint",
                    "policy_decisions": (
                        int(item_replay.get("macro_steps", 0) or 0)
                        + int(item_replay.get("combat_steps", 0) or 0)
                    ),
                },
            }
        )
    state = trainer_status.get("state")
    if state == "complete":
        events.append(
            {
                "event": "run_complete",
                "unix_s": trainer_status.get("unix_s", time.time()),
                "completion_status": "horizon_complete",
                "environment_steps": environment_steps,
                "episodes": ingested,
                "learner_updates": learner_updates,
                "policy_version": learner_updates,
            }
        )
    elif state == "failed":
        events.append(
            {
                "event": "run_failed",
                "unix_s": trainer_status.get("unix_s", time.time()),
                "environment_steps": environment_steps,
                "error": trainer_status.get("error"),
            }
        )
    else:
        failed_collectors = [
            item for item in collector_statuses if item.get("state") == "failed"
        ]
        if failed_collectors:
            details = "; ".join(
                f"{item.get('producer_id', 'collector')}: {item.get('error', 'failed')}"
                for item in failed_collectors
            )
            events.append(
                {
                    "event": "run_failed",
                    "unix_s": max(
                        float(item.get("unix_s", 0.0) or 0.0)
                        for item in failed_collectors
                    ),
                    "environment_steps": environment_steps,
                    "error": f"collector failure: {details}",
                }
            )
    return events


def _write_if_changed(path: Path, events: list[dict[str, Any]]) -> bool:
    payload = "".join(
        json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n"
        for event in events
    )
    try:
        if path.read_text(encoding="utf-8") == payload:
            return False
    except OSError:
        pass
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)
    return True


def main() -> int:
    args = _parse_args()
    output = Path(args.output)
    while True:
        events = _projection(args)
        _write_if_changed(output, events)
        terminal = events[-1].get("event") if events else None
        if terminal in {"run_complete", "run_failed"}:
            return 0
        time.sleep(max(0.1, args.poll_seconds))


if __name__ == "__main__":
    raise SystemExit(main())
