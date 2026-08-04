#!/usr/bin/env python3
"""Reconstruct failure-credit replay cost from production JSONL telemetry.

This benchmark is deliberately read-only.  It consumes the learner's staged
progress events, so it can measure a real run without loading a multi-gigabyte
replay checkpoint or changing training semantics.  In particular it separates
the failure-credit interval from the broader ``backward_ms`` timer and reports
how much GPU/CPU time was spent on updates that emitted no liveness actor
labels or no policy-head gradient.

Example::

    python scripts/benchmark_liveness_replay.py \
      /path/to/run/metrics.jsonl --recent 500 --format markdown

The learner does not currently publish replay sampling, host-to-device encode,
or peak-memory timers.  Those are reported as explicit telemetry gaps rather
than being guessed from the aggregate interval.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

from sts2_rl.artifacts import resolve_artifact_path, resolve_external_input_path


@dataclass(frozen=True, slots=True)
class Distribution:
    count: int
    mean: float
    median: float
    p90: float
    p99: float
    maximum: float


def _distribution(values: Iterable[float]) -> Distribution | None:
    ordered = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not ordered:
        return None

    def percentile(fraction: float) -> float:
        return ordered[round((len(ordered) - 1) * fraction)]

    return Distribution(
        count=len(ordered),
        mean=sum(ordered) / len(ordered),
        median=statistics.median(ordered),
        p90=percentile(0.90),
        p99=percentile(0.99),
        maximum=ordered[-1],
    )


def _distribution_mapping(values: Iterable[float]) -> dict[str, float | int] | None:
    """Return a JSON-ready distribution without assuming telemetry exists."""

    distribution = _distribution(values)
    return asdict(distribution) if distribution is not None else None


def _read_metrics(path: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    updates: list[dict[str, object]] = []
    progress: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            try:
                event = json.loads(raw_line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from error
            if not isinstance(event, dict):
                continue
            if event.get("event") == "learner_update":
                updates.append(event)
            elif event.get("event") == "learner_progress":
                progress.append(event)
    return updates, progress


def _numeric(event: dict[str, object], *path: str) -> float | None:
    value: object = event
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _learner_update_number(event: dict[str, object]) -> int | None:
    """Resolve the progress/update join key across the current telemetry ABI."""

    for key in ("update_number", "policy_version_after_update", "policy_version"):
        value = event.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    return None


def _record_intervals(
    progress: list[dict[str, object]],
) -> tuple[list[dict[str, float]], dict[int, float]]:
    by_update: dict[int, list[dict[str, object]]] = defaultdict(list)
    for event in progress:
        update = event.get("update_number")
        if isinstance(update, int) and not isinstance(update, bool):
            by_update[update].append(event)

    records: list[dict[str, float]] = []
    liveness_ms_by_update: dict[int, float] = {}
    for update, events in by_update.items():
        ordered = sorted(events, key=lambda item: _numeric(item, "elapsed_ms") or 0.0)
        starts = [item for item in ordered if item.get("stage") == "liveness_record_start"]
        complete = next(
            (item for item in ordered if item.get("stage") == "liveness_replay_complete"),
            None,
        )
        if complete is not None:
            liveness_ms = _numeric(complete, "liveness_replay_ms")
            if liveness_ms is not None:
                liveness_ms_by_update[update] = liveness_ms
        for index, start in enumerate(starts):
            begin = _numeric(start, "elapsed_ms")
            end = (
                _numeric(starts[index + 1], "elapsed_ms")
                if index + 1 < len(starts)
                else (_numeric(complete, "elapsed_ms") if complete is not None else None)
            )
            if begin is None or end is None or end < begin:
                continue
            slot = _numeric(start, "liveness_record_index")
            row: dict[str, float] = {
                "update": float(update),
                "slot": slot if slot is not None else float(index),
                "elapsed_ms": end - begin,
            }
            for source, target in (
                ("liveness_record_contexts", "contexts"),
                ("liveness_record_steps", "steps"),
                ("liveness_record_candidates", "candidates"),
                ("liveness_record_autograd_segments", "autograd_segments"),
            ):
                value = _numeric(start, source)
                if value is not None:
                    row[target] = value
            records.append(row)
    return records, liveness_ms_by_update


def _correlation(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right, strict=True))
    left_scale = sum((x - left_mean) ** 2 for x in left)
    right_scale = sum((y - right_mean) ** 2 for y in right)
    denominator = math.sqrt(left_scale * right_scale)
    return numerator / denominator if denominator else None


def analyze(
    path: Path,
    *,
    recent: int,
    minimum_environment_steps: int = 0,
    maximum_environment_steps: int | None = None,
) -> dict[str, object]:
    updates, progress = _read_metrics(path)
    if not updates:
        raise ValueError(f"no learner_update events in {path}")
    records, liveness_ms_by_update = _record_intervals(progress)
    selected_updates = [
        update
        for update in updates
        if (_numeric(update, "environment_steps") or 0.0) >= minimum_environment_steps
        and (
            maximum_environment_steps is None
            or (_numeric(update, "environment_steps") or 0.0) < maximum_environment_steps
        )
    ]
    selected_updates = selected_updates[-recent:] if recent else selected_updates
    if not selected_updates:
        raise ValueError("the requested environment-step window contains no learner updates")
    selected_numbers = {
        number
        for update in selected_updates
        if (number := _learner_update_number(update)) is not None
    }
    selected_records = [record for record in records if int(record["update"]) in selected_numbers]

    actor_label_fields = (
        "liveness_cost_actor_labels",
        "liveness_direct_policy_labels",
        "liveness_cycle_policy_labels",
        "liveness_contrast_policy_labels",
        "liveness_completion_policy_labels",
    )
    actor_zero_updates: list[dict[str, object]] = []
    policy_gradient_zero_updates: list[dict[str, object]] = []
    for update in selected_updates:
        labels = sum(int(_numeric(update, field) or 0) for field in actor_label_fields)
        if labels == 0:
            actor_zero_updates.append(update)
        gradient = _numeric(update, "liveness_policy_gradient_norm")
        if gradient is not None and gradient <= 1e-12:
            policy_gradient_zero_updates.append(update)

    def update_dist(*keys: str) -> Distribution | None:
        return _distribution(
            value
            for update in selected_updates
            if (value := _numeric(update, *keys)) is not None
        )

    liveness_values = [
        liveness_ms_by_update[number]
        for number in selected_numbers
        if number in liveness_ms_by_update
    ]
    total_values = [
        value
        for update in selected_updates
        if (value := _numeric(update, "timings", "total_ms")) is not None
    ]
    backward_values = [
        value
        for update in selected_updates
        if (value := _numeric(update, "timings", "backward_ms")) is not None
    ]
    actor_zero_ms = sum(
        liveness_ms_by_update.get(_learner_update_number(update) or -1, 0.0)
        for update in actor_zero_updates
    )
    policy_gradient_zero_ms = sum(
        liveness_ms_by_update.get(_learner_update_number(update) or -1, 0.0)
        for update in policy_gradient_zero_updates
    )
    quota_selected_fresh: dict[str, int] = defaultdict(int)
    quota_deficits: dict[str, int] = defaultdict(int)
    quota_available_totals: dict[str, int] = defaultdict(int)
    for update in selected_updates:
        quota = update.get("failure_credit_quota")
        statuses = quota.get("statuses", ()) if isinstance(quota, dict) else ()
        if not isinstance(statuses, list):
            continue
        for status in statuses:
            if not isinstance(status, dict) or not isinstance(status.get("stratum"), str):
                continue
            stratum = status["stratum"]
            quota_selected_fresh[stratum] += int(status.get("selected_actor_fresh", 0))
            quota_deficits[stratum] += int(status.get("deficit", 0))
            quota_available_totals[stratum] += int(status.get("available_total", 0))

    slot_summaries: dict[str, object] = {}
    for slot in sorted({int(record["slot"]) for record in selected_records}):
        rows = [record for record in selected_records if int(record["slot"]) == slot]
        slot_summaries[str(slot)] = {
            "elapsed_ms": _distribution_mapping(row["elapsed_ms"] for row in rows),
            "mean_steps": statistics.mean(row.get("steps", 0.0) for row in rows),
            "mean_candidates": statistics.mean(row.get("candidates", 0.0) for row in rows),
            "mean_contexts": statistics.mean(row.get("contexts", 0.0) for row in rows),
            "mean_autograd_segments": statistics.mean(row.get("autograd_segments", 0.0) for row in rows),
        }

    record_elapsed = [row["elapsed_ms"] for row in selected_records]
    latest_failure_replay = selected_updates[-1].get("failure_credit_replay")
    if not isinstance(latest_failure_replay, dict):
        latest_failure_replay = None

    result: dict[str, object] = {
        "schema_version": "sts2-liveness-replay-benchmark-v1",
        "source": str(path.resolve()),
        "learner_updates": len(selected_updates),
        "first_update": _learner_update_number(selected_updates[0]),
        "last_update": _learner_update_number(selected_updates[-1]),
        "last_environment_steps": selected_updates[-1].get("environment_steps"),
        "timings_ms": {
            "total": _distribution_mapping(total_values),
            "backward_envelope": _distribution_mapping(backward_values),
            "liveness_replay": _distribution_mapping(liveness_values),
            "episodic_replay": (
                asdict(distribution)
                if (distribution := update_dist("timings", "episodic_replay_ms")) is not None
                else None
            ),
            "online_recurrent_forward": (
                asdict(distribution)
                if (distribution := update_dist("timings", "recurrent_forward_ms")) is not None
                else None
            ),
        },
        "liveness_share": {
            "mean_of_total": (
                statistics.mean(liveness_values) / statistics.mean(total_values)
                if liveness_values and total_values
                else None
            ),
            "mean_of_backward_envelope": (
                statistics.mean(liveness_values) / statistics.mean(backward_values)
                if liveness_values and backward_values
                else None
            ),
        },
        "work": {
            "records_per_update": (
                asdict(distribution)
                if (distribution := update_dist("liveness_credit_plans")) is not None
                else None
            ),
            "steps_per_update": (
                asdict(distribution)
                if (distribution := update_dist("liveness_replayed_steps")) is not None
                else None
            ),
            "candidates_per_update": (
                asdict(distribution)
                if (distribution := update_dist("liveness_replayed_candidates")) is not None
                else None
            ),
            "autograd_segments_per_update": (
                asdict(distribution)
                if (distribution := update_dist("liveness_autograd_segments")) is not None
                else None
            ),
            "per_record_elapsed_ms": _distribution_mapping(record_elapsed),
            "elapsed_correlation_with_steps": _correlation(
                [row.get("steps", 0.0) for row in selected_records],
                record_elapsed,
            ),
            "elapsed_correlation_with_candidates": _correlation(
                [row.get("candidates", 0.0) for row in selected_records],
                record_elapsed,
            ),
            "slot_summaries": slot_summaries,
        },
        "actor_zero_work": {
            "no_actor_label_updates": len(actor_zero_updates),
            "no_actor_label_fraction": len(actor_zero_updates) / len(selected_updates),
            "no_actor_label_liveness_ms": actor_zero_ms,
            "no_policy_gradient_updates": len(policy_gradient_zero_updates),
            "no_policy_gradient_fraction": len(policy_gradient_zero_updates) / len(selected_updates),
            "no_policy_gradient_liveness_ms": policy_gradient_zero_ms,
            "note": (
                "These updates can still train the liveness critics/shared trunk. "
                "The wasted actor path is policy-logit materialization and actor bookkeeping, "
                "not the complete replay interval."
            ),
        },
        "liveness_signal": {
            "policy_gradient_norm": (
                asdict(distribution)
                if (distribution := update_dist("liveness_policy_gradient_norm")) is not None
                else None
            ),
            "critic_gradient_norm": (
                asdict(distribution)
                if (distribution := update_dist("liveness_critic_gradient_norm")) is not None
                else None
            ),
            "risk_actor_labels": (
                asdict(distribution)
                if (distribution := update_dist("liveness_cost_actor_labels")) is not None
                else None
            ),
            "direct_actor_labels": (
                asdict(distribution)
                if (distribution := update_dist("liveness_direct_policy_labels")) is not None
                else None
            ),
            "cycle_actor_labels": (
                asdict(distribution)
                if (distribution := update_dist("liveness_cycle_policy_labels")) is not None
                else None
            ),
            "contrast_actor_labels": (
                asdict(distribution)
                if (distribution := update_dist("liveness_contrast_policy_labels")) is not None
                else None
            ),
            "policy_lag_suppressed_labels": (
                asdict(distribution)
                if (
                    distribution := update_dist(
                        "liveness_policy_lag_suppressed_labels"
                    )
                )
                is not None
                else None
            ),
            "online_maximum_policy_lag": (
                asdict(distribution)
                if (distribution := update_dist("maximum_policy_lag")) is not None
                else None
            ),
            "quota_selected_actor_fresh": dict(quota_selected_fresh),
            "quota_deficits": dict(quota_deficits),
            "quota_mean_available_total": {
                stratum: total / len(selected_updates)
                for stratum, total in quota_available_totals.items()
            },
        },
        "latest_failure_replay": latest_failure_replay,
        "memory": {
            "failure_replay_storage_nbytes": (
                latest_failure_replay.get("storage_nbytes")
                if latest_failure_replay is not None
                else None
            ),
            "maximum_observed_record_nbytes": (
                latest_failure_replay.get("maximum_observed_record_nbytes")
                if latest_failure_replay is not None
                else None
            ),
            "device_peak_allocated_nbytes": None,
            "device_peak_reserved_nbytes": None,
        },
        "telemetry_gaps": [
            "failure-credit sample/quota selection duration",
            "snapshot collation and host-to-device duration",
            "model forward duration separated from backward duration",
            "per-record actor/critic label counts and evidence stratum",
            "CUDA/HIP peak allocated and reserved bytes per replay pack",
        ],
    }
    return result


def _format_markdown(result: dict[str, object]) -> str:
    timings = result["timings_ms"]
    assert isinstance(timings, dict)
    lines = [
        "# Liveness replay benchmark",
        "",
        f"- Source: `{result['source']}`",
        f"- Updates: {result['learner_updates']} ({result['first_update']}..{result['last_update']})",
        f"- Last environment step: {result['last_environment_steps']}",
        "",
        "| interval | mean ms | median ms | p90 ms | p99 ms | max ms |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, key in (
        ("total", "total"),
        ("backward envelope", "backward_envelope"),
        ("liveness replay", "liveness_replay"),
        ("episodic replay", "episodic_replay"),
        ("online recurrent forward", "online_recurrent_forward"),
    ):
        distribution = timings[key]
        assert isinstance(distribution, dict)
        lines.append(
            f"| {label} | {distribution['mean']:.1f} | {distribution['median']:.1f} | "
            f"{distribution['p90']:.1f} | {distribution['p99']:.1f} | {distribution['maximum']:.1f} |"
        )
    actor = result["actor_zero_work"]
    assert isinstance(actor, dict)
    share = result["liveness_share"]
    assert isinstance(share, dict)
    lines.extend(
        [
            "",
            f"- Liveness / total mean: {100.0 * float(share['mean_of_total']):.1f}%",
            f"- Liveness / backward-envelope mean: {100.0 * float(share['mean_of_backward_envelope']):.1f}%",
            f"- No actor labels: {actor['no_actor_label_updates']}/{result['learner_updates']} "
            f"({100.0 * float(actor['no_actor_label_fraction']):.1f}%), "
            f"consuming {float(actor['no_actor_label_liveness_ms']) / 1000.0:.1f}s of liveness replay",
            f"- Exactly zero policy-head gradient: {actor['no_policy_gradient_updates']}/{result['learner_updates']} "
            f"({100.0 * float(actor['no_policy_gradient_fraction']):.1f}%), "
            f"consuming {float(actor['no_policy_gradient_liveness_ms']) / 1000.0:.1f}s",
            "",
            "## Telemetry gaps",
        ]
    )
    telemetry_gaps = result["telemetry_gaps"]
    assert isinstance(telemetry_gaps, list)
    lines.extend(f"- {item}" for item in telemetry_gaps)
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", type=Path, help="run metrics.jsonl")
    parser.add_argument(
        "--recent",
        type=int,
        default=500,
        help="analyze the most recent N learner updates; 0 means all",
    )
    parser.add_argument("--format", choices=("json", "markdown"), default="json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--min-environment-steps", type=int, default=0)
    parser.add_argument("--max-environment-steps", type=int)
    arguments = parser.parse_args()
    if arguments.recent < 0 or arguments.min_environment_steps < 0:
        parser.error("step/update bounds must be non-negative")
    if (
        arguments.max_environment_steps is not None
        and arguments.max_environment_steps <= arguments.min_environment_steps
    ):
        parser.error("--max-environment-steps must exceed --min-environment-steps")
    metrics_path = resolve_external_input_path(arguments.metrics)
    result = analyze(
        metrics_path,
        recent=arguments.recent,
        minimum_environment_steps=arguments.min_environment_steps,
        maximum_environment_steps=arguments.max_environment_steps,
    )
    rendered = (
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
        if arguments.format == "json"
        else _format_markdown(result)
    )
    if arguments.output is None:
        print(rendered)
    else:
        output_path = resolve_artifact_path(arguments.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
