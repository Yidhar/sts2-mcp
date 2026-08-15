#!/usr/bin/env python3
"""Project live Hybrid control telemetry into the dashboard event schema."""

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
    parser.add_argument("--final-evaluation-episodes", type=int, default=16)
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


def _canonical_trainer_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Replace an uncheckpointed tail when a trainer resumes in-place."""

    canonical: list[dict[str, Any]] = []
    for row in rows:
        ingested = row.get("ingested_runs")
        if (
            row.get("event") == "hybrid_update"
            and isinstance(ingested, int)
            and not isinstance(ingested, bool)
        ):
            canonical = [
                previous
                for previous in canonical
                if not (
                    previous.get("event") == "hybrid_update"
                    and isinstance(previous.get("ingested_runs"), int)
                    and int(previous["ingested_runs"]) >= ingested
                )
            ]
        canonical.append(row)
    return canonical


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
    trainer_rows = _canonical_trainer_rows(
        _read_jsonl(run_root / "metrics" / "trainer.jsonl")
    )
    collector_rows = [
        row
        for path in sorted((run_root / "metrics").glob("collector-*.jsonl"))
        for row in _read_jsonl(path)
    ]
    collector_rows.sort(key=lambda row: float(row.get("unix_s", 0.0) or 0.0))
    evaluation_rows = _read_jsonl(run_root / "metrics" / "evaluation.jsonl")
    trainer_status = _read_json(run_root / "status" / "trainer.json")
    collector_statuses = [
        _read_json(path) for path in sorted((run_root / "status").glob("collector-*.json"))
    ]
    latest = trainer_rows[-1] if trainer_rows else {}
    learner_raw = latest.get("learner")
    learner: dict[str, Any] = learner_raw if isinstance(learner_raw, dict) else {}
    resumed_from = int(manifest.get("resumed_from_ingested_runs", 0) or 0)
    ingested = int(latest.get("ingested_runs", resumed_from) or resumed_from)
    source_environment_steps = int(manifest.get("source_environment_steps", 0) or 0)
    environment_steps = int(
        latest.get("environment_steps", source_environment_steps)
        or source_environment_steps
    )
    source_updates = int(manifest.get("source_learner_updates", 0) or 0)
    learner_updates = int(learner.get("updates", source_updates) or source_updates)
    environment_steps_by_policy: dict[int, int] = {}
    for row in trainer_rows:
        row_learner = row.get("learner")
        if not isinstance(row_learner, dict):
            continue
        row_policy = row_learner.get("updates")
        row_environment_steps = row.get("environment_steps")
        if (
            isinstance(row_policy, int)
            and not isinstance(row_policy, bool)
            and isinstance(row_environment_steps, int)
            and not isinstance(row_environment_steps, bool)
        ):
            environment_steps_by_policy[row_policy] = row_environment_steps
    events: list[dict[str, Any]] = [
        {
            "event": "run_start",
            "unix_s": _start_time(manifest, trainer_rows),
            "run_id": args.run_id,
            "pipeline": manifest.get("algorithm", "hybrid-control"),
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
                "profile": manifest.get("algorithm", "hybrid-control"),
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
                "model": {
                    "architecture": (
                        "hybrid-domain-iql-v1"
                        if manifest.get("algorithm") == "recurrent_iql_awac_v1"
                        else "hybrid-domain-candidate-q-v1"
                    )
                },
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
                "act1_revivals": row.get("act1_revivals"),
                "steps": row.get("steps"),
                "terminal_reason": row.get("terminal_reason"),
                "run_won": row.get("run_won"),
            }
        )
    pending = len(tuple((run_root / "spool").glob("*.pkl")))
    for row in trainer_rows:
        if row.get("event") != "hybrid_update":
            continue
        item_raw = row.get("learner")
        item: dict[str, Any] = item_raw if isinstance(item_raw, dict) else {}
        death_price_raw = item.get("death_price")
        death_price: dict[str, Any] = (
            death_price_raw if isinstance(death_price_raw, dict) else {}
        )
        item_replay_raw = row.get("replay")
        item_replay: dict[str, Any] = (
            item_replay_raw if isinstance(item_replay_raw, dict) else {}
        )
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
                "hybrid_macro_q_loss": item.get("macro_q_loss"),
                "hybrid_combat_q_loss": item.get("combat_q_loss"),
                "hybrid_macro_value_loss": item.get("macro_value_loss"),
                "hybrid_combat_value_loss": item.get("combat_value_loss"),
                "hybrid_macro_actor_loss": item.get("macro_actor_loss"),
                "hybrid_combat_actor_loss": item.get("combat_actor_loss"),
                "hybrid_policy_entropy": item.get("mean_policy_entropy"),
                "hybrid_mean_advantage_weight": item.get(
                    "mean_advantage_weight"
                ),
                "hybrid_maximum_advantage_weight": item.get(
                    "maximum_advantage_weight"
                ),
                "hybrid_critic_gradient_norm": item.get(
                    "critic_gradient_norm"
                ),
                "hybrid_actor_gradient_norm": item.get("actor_gradient_norm"),
                "hybrid_shared_gradient_cosine": item.get("shared_gradient_cosine"),
                "hybrid_forward_retries": item.get("forward_retries", 0),
                "hybrid_target_syncs": item.get("target_syncs"),
                "hybrid_last_target_sync_update": item.get(
                    "last_target_sync_update"
                ),
                "hybrid_death_price_lambda": death_price.get("lambda_value"),
                "hybrid_death_price_statistic": death_price.get("active_statistic"),
                "hybrid_death_price_act1_ema": death_price.get("act1_ema"),
                "hybrid_death_price_run_ema": death_price.get("run_ema"),
                "hybrid_death_price_target": (
                    death_price.get("course", {}).get(
                        "act1_target"
                        if death_price.get("active_statistic") == "act1"
                        else "run_target"
                    )
                    if isinstance(death_price.get("course"), dict)
                    else None
                ),
                "hybrid_death_price_freeze_reason": death_price.get(
                    "freeze_reason"
                ),
                "hybrid_death_price_support_successes": death_price.get(
                    "last_support_successes"
                ),
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
    current_policy_version = learner_updates
    current_evaluation_rows = [
        row
        for row in evaluation_rows
        if row.get("policy_version") == current_policy_version
    ]
    evaluation_summaries_by_policy: dict[int, dict[str, Any]] = {}
    for row in evaluation_rows:
        row_policy = row.get("policy_version")
        if (
            row.get("event") == "hybrid_evaluation_summary"
            and isinstance(row_policy, int)
            and not isinstance(row_policy, bool)
        ):
            evaluation_summaries_by_policy[row_policy] = row
    evaluation_summaries = [
        evaluation_summaries_by_policy[policy]
        for policy in sorted(evaluation_summaries_by_policy)
    ]
    evaluation_summary = evaluation_summaries_by_policy.get(current_policy_version)
    evaluation_started = next(
        (
            row
            for row in reversed(current_evaluation_rows)
            if row.get("event") == "hybrid_evaluation_started"
        ),
        None,
    )
    evaluation_episodes = [
        row
        for row in current_evaluation_rows
        if row.get("event") == "hybrid_evaluation_episode"
    ]
    state = trainer_status.get("state")
    expected_final_evaluation_episodes = int(
        getattr(args, "final_evaluation_episodes", 16)
    )
    evaluation_pending = (
        state == "complete"
        and expected_final_evaluation_episodes > 0
        and evaluation_summary is None
    )
    evaluation_in_progress = evaluation_summary is None and (
        evaluation_pending
        or evaluation_started is not None
        or bool(evaluation_episodes)
    )
    if evaluation_in_progress:
        policy_version = (
            evaluation_started.get("policy_version")
            if evaluation_started is not None
            else (
                evaluation_episodes[-1].get("policy_version")
                if evaluation_episodes
                else current_policy_version
            )
        )
        expected_episodes = (
            int(evaluation_started.get("episodes", 0) or 0)
            if evaluation_started is not None
            else expected_final_evaluation_episodes
        )
        events.append(
            {
                "event": "evaluation_started",
                "unix_s": (
                    evaluation_episodes[-1].get("unix_s")
                    if evaluation_episodes
                    else (
                        evaluation_started.get("unix_s")
                        if evaluation_started is not None
                        else trainer_status.get("unix_s", time.time())
                    )
                ),
                "evaluation_gate": policy_version,
                "evaluation_kind": "final_audit",
                "evaluation_attempt": 1,
                "environment_steps": environment_steps,
                "policy_version": policy_version,
                "evaluation_episodes": expected_episodes,
                "completed_episodes": len(evaluation_episodes),
                "evaluation_seeds": (
                    evaluation_started.get("seeds", [])
                    if evaluation_started is not None
                    else []
                ),
            }
        )
    for evaluation_summary in evaluation_summaries:
        evaluation_policy = int(evaluation_summary.get("policy_version", 0) or 0)
        evaluation_environment_steps = environment_steps_by_policy.get(
            evaluation_policy,
            int(evaluation_summary.get("environment_steps", environment_steps) or 0),
        )
        policy_evaluation_rows = [
            row
            for row in evaluation_rows
            if row.get("policy_version") == evaluation_policy
        ]
        evaluation_count = int(evaluation_summary.get("episodes", 0) or 0)
        evaluation_wins = int(
            evaluation_summary.get(
                "successful_run_count", evaluation_summary.get("wins", 0)
            )
            or 0
        )
        evaluation_act1 = int(
            evaluation_summary.get(
                "act1_clear_count", evaluation_summary.get("act1_clears", 0)
            )
            or 0
        )
        evaluation_act3 = int(
            evaluation_summary.get(
                "act3_reach_count", evaluation_summary.get("act3_reaches", 0)
            )
            or 0
        )
        evaluation_event = {
            key: value
            for key, value in evaluation_summary.items()
            if key not in {"event", "seeds"}
        }
        evaluation_event.update(
            {
                "event": "evaluation",
                "unix_s": evaluation_summary.get("unix_s"),
                "evaluation_gate": evaluation_summary.get("policy_version"),
                "gate_kind": "validation",
                "data_partition": "validation",
                "environment_steps": evaluation_environment_steps,
                "policy_version": evaluation_summary.get("policy_version"),
                "episodes": evaluation_count,
                "successful_run_count": evaluation_wins,
                "run_win_rate": (
                    evaluation_wins / evaluation_count if evaluation_count else 0.0
                ),
                "act1_clear_count": evaluation_act1,
                "act1_clear_rate": (
                    evaluation_act1 / evaluation_count if evaluation_count else 0.0
                ),
                "act3_reach_count": evaluation_act3,
                "act3_reach_rate": (
                    evaluation_act3 / evaluation_count if evaluation_count else 0.0
                ),
                "mean_max_floor": evaluation_summary.get(
                    "mean_max_floor", evaluation_summary.get("floor_mean")
                ),
                "maximum_floor": evaluation_summary.get(
                    "maximum_floor",
                    max(
                        (
                            int(row.get("max_floor", 0) or 0)
                            for row in policy_evaluation_rows
                            if row.get("event") == "hybrid_evaluation_episode"
                        ),
                        default=0,
                    ),
                ),
                "mean_revivals_used": evaluation_summary.get(
                    "mean_revivals_used", evaluation_summary.get("mean_revivals")
                ),
                "evaluation_context": {
                    "policy_version": evaluation_summary.get("policy_version"),
                    "data_partition": "validation",
                    "epsilon": 0.0,
                    "deterministic": True,
                },
            }
        )
        events.append(evaluation_event)
    if state == "complete" and not evaluation_in_progress:
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
