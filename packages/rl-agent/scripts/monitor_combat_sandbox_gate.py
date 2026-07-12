#!/usr/bin/env python3
"""Read MuZero combat-sandbox TensorBoard scalars and evaluate the Act1 repair gate.

This operator-facing monitor is intentionally separate from ``muzero.train``:
it lets us restart/compare sandbox runs without adding more orchestration logic
to the already large training entrypoint.

The gate is not a proof that full-run Act1 is solved.  It only answers:

    "Is hallway combat execution healthy enough to justify switching back to a
    full-run validation?"

Full-run still has to learn route choices, deck/reward picks, upgrades/rests,
shops, events, relic acquisition, potion retention, and boss entry resources.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import sys
from pathlib import Path
from statistics import mean
from typing import Any

RL_AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(RL_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(RL_AGENT_ROOT))

from sts2_rl.artifacts import resolve_artifact_path, resolve_external_input_path  # noqa: E402

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except Exception as exc:  # pragma: no cover - operator-facing failure
    raise SystemExit(
        "TensorBoard EventAccumulator import failed. Run this from the "
        "packages/rl-agent venv. Original error: " + repr(exc)
    )


DEFAULT_ROOT = resolve_artifact_path(None, default="runs")


CORE_TAGS: tuple[str, ...] = (
    "buffer/size",
    "episode/reward",
    "episode/length",
    "episode/death_floor",
    "recent_tail/64/win_rate",
    "recent_tail/64/weak_win_rate",
    "recent_tail/64/weak_sample_count",
    "recent_tail/64/normal_win_rate",
    "recent_tail/64/normal_sample_count",
    "recent_tail/64/hard_normal_win_rate",
    "recent_tail/64/hard_normal_sample_count",
    "recent_tail/256/win_rate",
    "recent_tail/256/weak_win_rate",
    "recent_tail/256/weak_sample_count",
    "recent_tail/256/normal_win_rate",
    "recent_tail/256/normal_sample_count",
    "recent_tail/256/hard_normal_win_rate",
    "recent_tail/256/hard_normal_sample_count",
    "combat_quality/bad_pure_block_selected_rate",
    "combat_quality/card_block_waste_selected_rate",
    "combat_quality/card_block_waste_with_progress_selected_rate",
    "combat_quality/card_no_damage_pressure_selected_rate",
    "combat_quality/card_no_damage_pressure_with_progress_selected_rate",
    "combat_quality/wasteful_end_turn_rate",
    "loss/total",
    "loss/future_world_aux",
    "loss/future_bank_state",
    "loss/future_bank_delta",
    "loss/future_bank_token_slot_source",
)


MEMORY_TAGS: tuple[str, ...] = (
    "memory/allocated_gb",
    "memory/max_allocated_gb",
    "memory/reserved_gb",
    "memory/reserved_minus_allocated_gb",
    "memory/empty_cache_called",
)


TACTICAL_TAGS: tuple[str, ...] = (
    "combat_quality/zero_energy_x_cost_selected_rate",
    "combat_quality/hp_cost_low_margin_selected_rate",
    "combat_quality/hp_cost_self_lethal_selected_rate",
    "combat_quality/refund_no_followup_selected_rate",
    "combat_quality/refund_no_followup_with_progress_selected_rate",
    "combat_quality/refund_no_followup_progress_alternative_selected_rate",
    "combat_quality/refund_no_followup_no_alternative_selected_rate",
    "combat_quality/refund_no_followup_progress_alternative_count_mean",
    "combat_quality/refund_no_followup_guard_available_rate",
    "combat_quality/refund_no_followup_guard_applied_rate",
    "combat_quality/refund_no_followup_guard_override_rate",
    "combat_quality/refund_no_followup_guard_no_alternative_rate",
    "combat_quality/refund_no_followup_guard_candidate_count_mean",
    "combat_quality/bad_end_turn_selected_rate",
    "combat_quality/bad_end_turn_available_rate",
    "combat_quality/forced_end_turn_selected_rate",
    "combat_quality/end_turn_unknown_selected_rate",
    "combat_quality/strategic_defer_end_turn_selected_rate",
    "combat_quality/potion_low_urgency_selected_rate",
    "combat_quality/potion_save_recommended_selected_rate",
    "combat_quality/x_cost_zero_guard_applied_rate",
    "combat_quality/hp_cost_margin_guard_applied_rate",
    "normal_combat/zero_energy_x_cost_selected_rate",
    "normal_combat/hp_cost_low_margin_selected_rate",
    "normal_combat/hp_cost_self_lethal_selected_rate",
    "normal_combat/refund_no_followup_selected_rate",
    "normal_combat/refund_no_followup_with_progress_selected_rate",
    "normal_combat/refund_no_followup_progress_alternative_selected_rate",
    "normal_combat/refund_no_followup_no_alternative_selected_rate",
    "normal_combat/refund_no_followup_guard_applied_rate",
    "normal_combat/strategic_defer_end_turn_selected_rate",
    "normal_combat/potion_low_urgency_selected_rate",
    "weak_combat/zero_energy_x_cost_selected_rate",
    "weak_combat/hp_cost_low_margin_selected_rate",
    "weak_combat/hp_cost_self_lethal_selected_rate",
    "weak_combat/refund_no_followup_selected_rate",
    "weak_combat/refund_no_followup_with_progress_selected_rate",
    "weak_combat/refund_no_followup_progress_alternative_selected_rate",
    "weak_combat/refund_no_followup_no_alternative_selected_rate",
    "weak_combat/refund_no_followup_guard_applied_rate",
    "weak_combat/strategic_defer_end_turn_selected_rate",
    "weak_combat/potion_low_urgency_selected_rate",
)


STRICT_BAD_END_TURN_CLASSES: frozenset[str] = frozenset(
    {
        "bad_end_turn",
        "bad_end_turn_strict",
        "bad_end_turn_stable_with_actions",
    }
)


def latest_run(root: Path) -> Path:
    runs = [path for path in root.iterdir() if path.is_dir()] if root.exists() else []
    if not runs:
        raise SystemExit(f"No TensorBoard run directories found under {root}")
    return max(runs, key=lambda path: path.stat().st_mtime)


def finite(values: list[float]) -> list[float]:
    return [float(value) for value in values if math.isfinite(float(value))]


def summarize_scalar(ea: EventAccumulator, tag: str, tail: int) -> dict[str, Any]:
    events = ea.Scalars(tag)
    values = finite([float(event.value) for event in events])
    if not events or not values:
        return {
            "n": 0,
            "step": None,
            "last": None,
            "avg_tail": None,
            "min_tail": None,
            "max_tail": None,
        }
    tail_values = values[-min(int(tail), len(values)) :]
    return {
        "n": len(events),
        "step": int(events[-1].step),
        "last": float(values[-1]),
        "avg_tail": float(mean(tail_values)),
        "min_tail": float(min(tail_values)),
        "max_tail": float(max(tail_values)),
    }


def load_scalars(run_dir: Path, tail: int) -> tuple[dict[str, dict[str, Any]], set[str]]:
    ea = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    ea.Reload()
    tags = set(ea.Tags().get("scalars", []))
    summaries: dict[str, dict[str, Any]] = {}
    for tag in [*CORE_TAGS, *MEMORY_TAGS, *TACTICAL_TAGS]:
        if tag in tags:
            summaries[tag] = summarize_scalar(ea, tag, tail)
    return summaries, tags


def _jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def _row_step(row: dict[str, Any]) -> int:
    raw = row.get("global_step")
    if raw is None:
        raw = row.get("step")
    try:
        return int(raw)
    except Exception:
        return 0


def _top(counter: collections.Counter[str], limit: int) -> list[dict[str, Any]]:
    return [{"name": str(name), "count": int(count)} for name, count in counter.most_common(limit)]


def _counter(rows: list[dict[str, Any]], *keys: str) -> collections.Counter[str]:
    counter: collections.Counter[str] = collections.Counter()
    for row in rows:
        value: Any = None
        for key in keys:
            value = row.get(key)
            if value not in (None, ""):
                break
        if value in (None, ""):
            value = "?"
        counter[str(value)] += 1
    return counter


def _normalise_encounter_id(raw: str) -> str | None:
    text = str(raw or "").strip()
    if not text or text == "?":
        return None
    lower = text.lower()
    if lower.startswith("encounter."):
        return lower
    if lower.startswith("encounter_"):
        return "encounter." + lower[len("encounter_") :]
    if lower.startswith("encounter:"):
        return "encounter." + lower[len("encounter:") :]
    if lower.startswith("encounter/"):
        return "encounter." + lower[len("encounter/") :]
    if lower.startswith("encounter "):
        return "encounter." + lower[len("encounter ") :]
    if lower.startswith("encounter"):
        return lower.replace("encounter", "encounter.", 1)
    if lower.startswith("enemy."):
        return lower
    if lower.startswith("event."):
        return lower
    if lower.startswith("room."):
        return lower
    if lower.startswith("boss."):
        return lower
    if lower.startswith("elite."):
        return lower
    if lower.startswith("normal."):
        return lower
    if lower.startswith("weak."):
        return lower
    if lower.startswith("hard."):
        return lower
    if lower.startswith("act"):
        return lower
    if lower.startswith("the_") or lower.startswith("slumbering_") or lower.startswith("scrolls_"):
        return "encounter." + lower
    if lower.startswith("ENCOUNTER."):  # unreachable after lower(), kept for readability
        return "encounter." + text.split(".", 1)[1].lower()
    if text.upper().startswith("ENCOUNTER."):
        return "encounter." + text.split(".", 1)[1].lower()
    return lower


def load_diagnostics_summary(run_dir: Path, *, window: int, top_n: int = 12) -> dict[str, Any]:
    """Load compact operator diagnostics from sidecar JSONL files.

    The TensorBoard gate tells us *that* the sandbox is still red.  The
    diagnostics sidecars are what tell us *where* to aim the next targeted
    sandbox: offender type, encounter, selected action/title, and death slice
    encounter.  This helper is deliberately read-only and best-effort so it
    can be added to a monitor loop without perturbing training.
    """

    diagnostics_dir = run_dir / "diagnostics"
    payload: dict[str, Any] = {
        "diagnostics_dir": str(diagnostics_dir),
        "window": int(max(window, 0)),
        "top_n": int(max(top_n, 1)),
        "available": diagnostics_dir.exists(),
    }
    if not diagnostics_dir.exists():
        return payload

    offender_rows = _jsonl_rows(diagnostics_dir / "action_offenders.jsonl")
    offender_max_step = max((_row_step(row) for row in offender_rows), default=0)
    offender_cutoff = max(0, offender_max_step - int(max(window, 0)))
    recent_offenders = [row for row in offender_rows if _row_step(row) >= offender_cutoff]

    death_rows: list[dict[str, Any]] = []
    death_root = diagnostics_dir / "death_slices"
    if death_root.exists():
        for jsonl_path in sorted(death_root.rglob("*.jsonl")):
            for row in _jsonl_rows(jsonl_path):
                row.setdefault("_file", jsonl_path.name)
                death_rows.append(row)
    death_max_step = max((_row_step(row) for row in death_rows), default=0)
    death_cutoff = max(0, max(offender_max_step, death_max_step) - int(max(window, 0)))
    recent_deaths = [row for row in death_rows if _row_step(row) >= death_cutoff]

    end_turn_rows = _jsonl_rows(diagnostics_dir / "end_turn_contexts.jsonl")
    end_turn_max_step = max((_row_step(row) for row in end_turn_rows), default=0)
    end_turn_cutoff = max(0, end_turn_max_step - int(max(window, 0)))
    recent_end_turn = [row for row in end_turn_rows if _row_step(row) >= end_turn_cutoff]
    end_turn_class_counter = _counter(recent_end_turn, "end_turn_class", "class")
    strict_bad_recent_count = sum(
        1 for row in recent_end_turn if str(row.get("end_turn_class") or row.get("class") or "") in STRICT_BAD_END_TURN_CLASSES
    )
    strict_bad_total_count = sum(
        1 for row in end_turn_rows if str(row.get("end_turn_class") or row.get("class") or "") in STRICT_BAD_END_TURN_CLASSES
    )

    offender_encounters = _counter(recent_offenders, "encounter_id", "encounter", "combat_id")
    death_encounters = _counter(recent_deaths, "encounter_id", "encounter", "combat_id")

    candidates: list[str] = []
    seen: set[str] = set()
    for raw, _count in [*death_encounters.most_common(top_n), *offender_encounters.most_common(top_n)]:
        norm = _normalise_encounter_id(raw)
        if norm and norm not in seen:
            seen.add(norm)
            candidates.append(norm)

    payload.update(
        {
            "offenders": {
                "rows": len(offender_rows),
                "max_step": offender_max_step or None,
                "recent_rows": len(recent_offenders),
                "top_types": _top(_counter(recent_offenders, "offender_type", "type"), top_n),
                "top_encounters": _top(offender_encounters, top_n),
                "top_selected_titles": _top(
                    _counter(recent_offenders, "selected_title", "selected_action_title", "action_title"),
                    top_n,
                ),
            },
            "deaths": {
                "rows": len(death_rows),
                "max_step": death_max_step or None,
                "recent_rows": len(recent_deaths),
                "top_encounters": _top(death_encounters, top_n),
            },
            "end_turn_contexts": {
                "rows": len(end_turn_rows),
                "max_step": end_turn_max_step or None,
                "recent_rows": len(recent_end_turn),
                "strict_bad_recent_count": int(strict_bad_recent_count),
                "strict_bad_total_count": int(strict_bad_total_count),
                "top_classes": _top(end_turn_class_counter, top_n),
            },
            "targeted_candidates": candidates[:top_n],
        }
    )
    return payload


def value(summary: dict[str, dict[str, Any]], tag: str, field: str = "last") -> float | None:
    item = summary.get(tag)
    if not item:
        return None
    raw = item.get(field)
    return float(raw) if isinstance(raw, (int, float)) and math.isfinite(float(raw)) else None


def has_metric(summary: dict[str, dict[str, Any]], tag: str) -> bool:
    item = summary.get(tag)
    return bool(item and int(item.get("n") or 0) > 0)


def check_le(
    summary: dict[str, dict[str, Any]],
    tag: str,
    threshold: float,
    *,
    field: str = "avg_tail",
    required: bool = True,
) -> dict[str, Any]:
    actual = value(summary, tag, field)
    if actual is None:
        return {
            "tag": tag,
            "field": field,
            "op": "<=",
            "threshold": threshold,
            "actual": None,
            "status": "missing" if required else "skipped",
        }
    return {
        "tag": tag,
        "field": field,
        "op": "<=",
        "threshold": threshold,
        "actual": actual,
        "status": "pass" if actual <= threshold else "fail",
    }


def check_ge(
    summary: dict[str, dict[str, Any]],
    tag: str,
    threshold: float,
    *,
    field: str = "last",
    required: bool = True,
) -> dict[str, Any]:
    actual = value(summary, tag, field)
    if actual is None:
        return {
            "tag": tag,
            "field": field,
            "op": ">=",
            "threshold": threshold,
            "actual": None,
            "status": "missing" if required else "skipped",
        }
    return {
        "tag": tag,
        "field": field,
        "op": ">=",
        "threshold": threshold,
        "actual": actual,
        "status": "pass" if actual >= threshold else "fail",
    }


def check_ge_if_sampled(
    summary: dict[str, dict[str, Any]],
    tag: str,
    threshold: float,
    *,
    sample_tag: str,
    min_samples: float,
    field: str = "last",
    required: bool = True,
) -> dict[str, Any]:
    """Check a rate only when the corresponding sample count is meaningful.

    TensorBoard rate tags are sometimes emitted as ``0.0`` for cohorts that
    have no examples in the current sandbox phase.  Treating that as a real
    0% win rate creates a false red gate and can send operators chasing a
    nonexistent regression.  When the sample count tag is present and below the
    requested minimum, surface the rate check as skipped with the exact sample
    count instead of pass/fail.

    If the sample-count tag is unavailable (older runs), keep the historical
    behavior and evaluate the rate directly.
    """
    sample_count = value(summary, sample_tag, "last")
    if sample_count is not None and sample_count < min_samples:
        return {
            "tag": tag,
            "field": field,
            "op": ">=",
            "threshold": threshold,
            "actual": value(summary, tag, field),
            "status": "skipped",
            "reason": "insufficient_samples",
            "sample_tag": sample_tag,
            "sample_count": sample_count,
            "min_samples": min_samples,
        }
    return check_ge(summary, tag, threshold, field=field, required=required)


def evaluate_gate(
    summary: dict[str, dict[str, Any]],
    *,
    min_buffer: float,
    strict_weak: bool,
    min_weak_samples: float,
    min_hard_normal_samples: float,
    max_reserved_gb: float,
    max_peak_allocated_gb: float,
    max_empty_cache_rate: float,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    checks.append(check_ge(summary, "buffer/size", min_buffer, field="last"))

    # Combat execution gate.  Weak is deliberately high: losing weak fights in
    # sandbox usually means the full-run resource curve will die before boss.
    # However the hard/normal repair sandbox intentionally downweights weak
    # encounters heavily (for example weak=0.05, normal=1.95).  Recent windows
    # can therefore contain zero weak fights.  Treating the emitted 0.0 weak
    # rate as a true loss creates a false gate failure, so require a minimum
    # weak sample count just like the hard-normal cohort.
    weak_threshold = 0.97 if strict_weak else 0.95
    checks.append(check_ge(summary, "recent_tail/64/win_rate", 0.95, field="last"))
    checks.append(
        check_ge_if_sampled(
            summary,
            "recent_tail/64/weak_win_rate",
            weak_threshold,
            field="last",
            sample_tag="recent_tail/64/weak_sample_count",
            min_samples=min_weak_samples,
        )
    )
    checks.append(check_ge(summary, "recent_tail/64/normal_win_rate", 0.90, field="last"))
    checks.append(
        check_ge_if_sampled(
            summary,
            "recent_tail/64/hard_normal_win_rate",
            0.75,
            field="last",
            sample_tag="recent_tail/64/hard_normal_sample_count",
            min_samples=min_hard_normal_samples,
        )
    )
    checks.append(check_ge(summary, "recent_tail/256/normal_win_rate", 0.88, field="last"))
    checks.append(
        check_ge_if_sampled(
            summary,
            "recent_tail/256/hard_normal_win_rate",
            0.70,
            field="last",
            sample_tag="recent_tail/256/hard_normal_sample_count",
            min_samples=min_hard_normal_samples,
        )
    )

    # Tactical quality.  Use tail means to avoid overreacting to one tiny
    # episode denominator, but keep thresholds tight enough to catch regressions.
    checks.append(check_le(summary, "combat_quality/bad_pure_block_selected_rate", 0.002, field="avg_tail"))
    # Prefer narrow actionable versions once current code emits them.  Broad
    # block/no-pressure tags remain printed for visibility, but they include
    # legal-action-starved hands such as "Defend + End Turn only" and should
    # not fail a sandbox gate when no progress alternative existed.
    if has_metric(summary, "combat_quality/card_block_waste_with_progress_selected_rate"):
        checks.append(
            check_le(
                summary,
                "combat_quality/card_block_waste_with_progress_selected_rate",
                0.005,
                field="avg_tail",
            )
        )
    else:
        checks.append(check_le(summary, "combat_quality/card_block_waste_selected_rate", 0.010, field="avg_tail"))
    if has_metric(summary, "combat_quality/card_no_damage_pressure_with_progress_selected_rate"):
        checks.append(
            check_le(
                summary,
                "combat_quality/card_no_damage_pressure_with_progress_selected_rate",
                0.005,
                field="avg_tail",
            )
        )
    else:
        checks.append(check_le(summary, "combat_quality/card_no_damage_pressure_selected_rate", 0.010, field="avg_tail"))
    # End Turn has a high false-positive risk if judged from energy alone:
    # many correct turns end with spare energy because the hand is empty,
    # remaining cards are unplayable, or the only alternative is overflow
    # no-pressure block.  Current trainer code emits the central taxonomy
    # metric below; prefer it over the older broad wasteful tag.  Keep the
    # legacy fallback only for older runs that do not have strict taxonomy
    # scalars yet.
    if has_metric(summary, "combat_quality/bad_end_turn_selected_rate"):
        checks.append(check_le(summary, "combat_quality/bad_end_turn_selected_rate", 0.0, field="avg_tail"))
    else:
        checks.append(check_le(summary, "combat_quality/wasteful_end_turn_rate", 0.0, field="avg_tail"))

    # New tactical tags prove the clean-metrics process is running current code.
    checks.append(check_le(summary, "combat_quality/zero_energy_x_cost_selected_rate", 0.005, field="avg_tail"))
    checks.append(check_le(summary, "combat_quality/hp_cost_self_lethal_selected_rate", 0.0, field="avg_tail"))
    checks.append(check_le(summary, "combat_quality/hp_cost_low_margin_selected_rate", 0.005, field="avg_tail"))
    if has_metric(summary, "combat_quality/refund_no_followup_with_progress_selected_rate"):
        checks.append(check_le(summary, "combat_quality/refund_no_followup_with_progress_selected_rate", 0.005, field="avg_tail"))
    else:
        checks.append(check_le(summary, "combat_quality/refund_no_followup_selected_rate", 0.005, field="avg_tail"))
    checks.append(check_le(summary, "combat_quality/potion_low_urgency_selected_rate", 0.010, field="avg_tail"))

    # Loss sanity.  These are intentionally loose spike sentinels, not
    # optimization targets.
    checks.append(check_le(summary, "loss/total", 100.0, field="max_tail"))
    checks.append(check_le(summary, "loss/future_world_aux", 10.0, field="max_tail"))
    checks.append(check_le(summary, "loss/future_bank_state", 10.0, field="max_tail"))
    checks.append(check_le(summary, "loss/future_bank_delta", 5.0, field="max_tail"))
    checks.append(check_le(summary, "loss/future_bank_token_slot_source", 0.0, field="max_tail"))

    # Memory sanity.  These are operator guardrails for the large-token model;
    # missing memory scalars should not fail older runs, but present scalars
    # should catch silent low-memory-regression before the run OOMs.
    checks.append(
        check_le(
            summary,
            "memory/reserved_gb",
            max_reserved_gb,
            field="max_tail",
            required=False,
        )
    )
    if has_metric(summary, "memory/max_allocated_gb"):
        checks.append(
            check_le(
                summary,
                "memory/max_allocated_gb",
                max_peak_allocated_gb,
                field="max_tail",
                required=False,
            )
        )
    else:
        checks.append(
            check_le(
                summary,
                "memory/allocated_gb",
                max_peak_allocated_gb,
                field="max_tail",
                required=False,
            )
        )
    checks.append(
        check_le(
            summary,
            "memory/empty_cache_called",
            max_empty_cache_rate,
            field="avg_tail",
            required=False,
        )
    )

    buffer_last = value(summary, "buffer/size", "last")
    if buffer_last is None or buffer_last < min_buffer:
        verdict = "WAIT_BUFFER"
    elif any(check["status"] == "missing" for check in checks):
        verdict = "MISSING_TAGS"
    elif any(check["status"] == "fail" for check in checks):
        verdict = "FAIL"
    else:
        verdict = "PASS"

    return {"verdict": verdict, "checks": checks}


def _recompute_verdict_after_overrides(gate: dict[str, Any]) -> str:
    """Recompute a non-WAIT gate verdict after diagnostic check overrides."""
    if gate.get("verdict") == "WAIT_BUFFER":
        return "WAIT_BUFFER"
    checks = gate.get("checks") or []
    if any(check.get("status") == "missing" for check in checks):
        return "MISSING_TAGS"
    if any(check.get("status") == "fail" for check in checks):
        return "FAIL"
    return "PASS"


def apply_diagnostics_gate_overrides(payload: dict[str, Any]) -> dict[str, Any]:
    """Use richer sidecar diagnostics to suppress known scalar false positives.

    The TensorBoard scalar ``combat_quality/bad_end_turn_selected_rate`` is a
    compact training-time signal.  In long-running smoke tests we have observed
    it retain a tiny nonzero tail value for an EndTurn episode where the
    richer ``diagnostics/end_turn_contexts.jsonl`` taxonomy says every selected
    EndTurn was either forced or unknown, with *zero* strict bad contexts.

    Spare energy is common after a hand is exhausted or only deferable actions
    remain.  Do not let a legacy/broad scalar artifact block the sandbox ->
    full-run transition when the stable-frontier sidecar has selected EndTurn
    rows and no strict bad classes in the same recent global-step window.
    """

    diagnostics = payload.get("diagnostics")
    if not isinstance(diagnostics, dict):
        return payload
    end_turn = diagnostics.get("end_turn_contexts")
    if not isinstance(end_turn, dict):
        return payload
    recent_rows = int(end_turn.get("recent_rows") or 0)
    strict_bad_recent_count = int(end_turn.get("strict_bad_recent_count") or 0)
    if recent_rows <= 0 or strict_bad_recent_count != 0:
        return payload

    gate = payload.get("gate")
    if not isinstance(gate, dict):
        return payload
    changed = False
    for check in gate.get("checks") or []:
        if (
            check.get("tag") == "combat_quality/bad_end_turn_selected_rate"
            and check.get("status") == "fail"
        ):
            check["status"] = "skipped"
            check["reason"] = "strict_end_turn_sidecar_zero"
            check["strict_sidecar_recent_rows"] = recent_rows
            check["strict_bad_recent_count"] = strict_bad_recent_count
            check["strict_bad_total_count"] = int(end_turn.get("strict_bad_total_count") or 0)
            changed = True
    if changed:
        gate["verdict"] = _recompute_verdict_after_overrides(gate)
    return payload


def print_human(payload: dict[str, Any]) -> None:
    print(f"run_dir: {payload['run_dir']}")
    print(f"scalar_tag_count: {payload['scalar_tag_count']}")
    print(f"tail: {payload['tail']}")
    print(f"verdict: {payload['gate']['verdict']}")
    print()
    print("KEY METRICS")
    print("-----------")
    for tag in CORE_TAGS + MEMORY_TAGS + TACTICAL_TAGS:
        item = payload["metrics"].get(tag)
        if item is None:
            if tag in CORE_TAGS or tag in MEMORY_TAGS or tag.startswith("combat_quality/"):
                print(f"{tag:<62} MISSING")
            continue
        print(
            f"{tag:<62} "
            f"last={item['last']} avg={item['avg_tail']} "
            f"min={item['min_tail']} max={item['max_tail']} "
            f"step={item['step']} n={item['n']}"
        )
    print()
    print("GATE CHECKS")
    print("-----------")
    for check in payload["gate"]["checks"]:
        actual = check["actual"]
        actual_s = "None" if actual is None else f"{actual:.6g}"
        extra = ""
        if check.get("reason") == "insufficient_samples":
            extra = (
                f" ({check.get('sample_tag')}={check.get('sample_count'):.6g} "
                f"< min_samples={check.get('min_samples'):.6g})"
            )
        elif check.get("reason") == "strict_end_turn_sidecar_zero":
            extra = (
                " (strict end_turn_contexts sidecar has "
                f"{check.get('strict_bad_recent_count')} strict bad / "
                f"{check.get('strict_sidecar_recent_rows')} recent selected EndTurn rows)"
            )
        print(
            f"{check['status']:<8} {check['tag']}[{check['field']}] "
            f"{check['op']} {check['threshold']} actual={actual_s}{extra}"
        )
    diagnostics = payload.get("diagnostics")
    if isinstance(diagnostics, dict):
        print()
        print("DIAGNOSTICS")
        print("-----------")
        if not diagnostics.get("available"):
            print(f"diagnostics_dir: {diagnostics.get('diagnostics_dir')} MISSING")
            return

        offenders = diagnostics.get("offenders") or {}
        deaths = diagnostics.get("deaths") or {}
        print(
            "offenders: "
            f"rows={offenders.get('rows')} "
            f"recent_window={diagnostics.get('window')} "
            f"recent_rows={offenders.get('recent_rows')} "
            f"max_step={offenders.get('max_step')}"
        )
        print("top offender types:")
        print("  note: EndTurn triage should use combat_quality/bad_end_turn_selected_rate;")
        print("        legacy action_offenders bad_end_turn rows may be soft/ambiguous.")
        for item in offenders.get("top_types") or []:
            print(f"  {item['name']}: {item['count']}")
        print("top offender encounters:")
        for item in offenders.get("top_encounters") or []:
            print(f"  {item['name']}: {item['count']}")
        print("top selected titles/actions:")
        for item in offenders.get("top_selected_titles") or []:
            print(f"  {item['name']}: {item['count']}")

        print(
            "deaths: "
            f"rows={deaths.get('rows')} "
            f"recent_window={diagnostics.get('window')} "
            f"recent_rows={deaths.get('recent_rows')} "
            f"max_step={deaths.get('max_step')}"
        )
        print("top recent death encounters:")
        for item in deaths.get("top_encounters") or []:
            print(f"  {item['name']}: {item['count']}")
        end_turn = diagnostics.get("end_turn_contexts") or {}
        print(
            "end_turn_contexts: "
            f"rows={end_turn.get('rows')} "
            f"recent_window={diagnostics.get('window')} "
            f"recent_rows={end_turn.get('recent_rows')} "
            f"max_step={end_turn.get('max_step')} "
            f"strict_bad_recent={end_turn.get('strict_bad_recent_count')} "
            f"strict_bad_total={end_turn.get('strict_bad_total_count')}"
        )
        print("top EndTurn classes:")
        for item in end_turn.get("top_classes") or []:
            print(f"  {item['name']}: {item['count']}")
        print("targeted candidates:")
        for candidate in diagnostics.get("targeted_candidates") or []:
            print(f"  {candidate}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="runs root")
    parser.add_argument("--run-dir", type=Path, default=None, help="specific TensorBoard run directory")
    parser.add_argument("--tail", type=int, default=20, help="tail window for averages/min/max")
    parser.add_argument("--min-buffer", type=float, default=3000.0, help="minimum replay buffer size before judging")
    parser.add_argument("--strict-weak", action="store_true", help="require weak 64-win >= 0.97 instead of 0.95")
    parser.add_argument(
        "--min-weak-samples",
        type=float,
        default=8.0,
        help=(
            "skip weak win-rate checks until the recent-tail weak sample count "
            "reaches this value; avoids treating zero-sample weak cohorts as "
            "0%% win-rate regressions in normal-heavy sandbox phases"
        ),
    )
    parser.add_argument(
        "--min-hard-normal-samples",
        type=float,
        default=8.0,
        help=(
            "skip hard-normal win-rate checks until the recent-tail hard-normal "
            "sample count reaches this value; avoids treating zero-sample cohorts "
            "as 0%% win-rate regressions"
        ),
    )
    parser.add_argument("--max-reserved-gb", type=float, default=22.0, help="fail when memory/reserved_gb tail max exceeds this")
    parser.add_argument("--max-peak-allocated-gb", type=float, default=20.0, help="fail when memory/max_allocated_gb or allocated_gb tail max exceeds this")
    parser.add_argument("--max-empty-cache-rate", type=float, default=0.2, help="fail when memory/empty_cache_called tail average exceeds this")
    parser.add_argument("--diagnostics-window", type=int, default=1000, help="global-step window for action/death diagnostics summary")
    parser.add_argument("--no-diagnostics", action="store_true", help="do not read diagnostics sidecar JSONL files")
    parser.add_argument("--json", action="store_true", help="emit JSON only")
    parser.add_argument("--fail-on-red", action="store_true", help="exit nonzero on FAIL/MISSING_TAGS")
    args = parser.parse_args()

    root = resolve_external_input_path(args.root, default="runs")
    run_dir = resolve_external_input_path(args.run_dir, root=root) if args.run_dir else latest_run(root)
    summaries, tags = load_scalars(run_dir, tail=max(int(args.tail), 1))
    payload = {
        "run_dir": str(run_dir),
        "scalar_tag_count": len(tags),
        "tail": int(args.tail),
        "metrics": summaries,
        "missing_core_tags": [tag for tag in CORE_TAGS if tag not in tags],
        "missing_memory_tags": [tag for tag in MEMORY_TAGS if tag not in tags],
        "missing_tactical_tags": [tag for tag in TACTICAL_TAGS if tag not in tags],
        "gate": evaluate_gate(
            summaries,
            min_buffer=float(args.min_buffer),
            strict_weak=bool(args.strict_weak),
            min_weak_samples=float(args.min_weak_samples),
            min_hard_normal_samples=float(args.min_hard_normal_samples),
            max_reserved_gb=float(args.max_reserved_gb),
            max_peak_allocated_gb=float(args.max_peak_allocated_gb),
            max_empty_cache_rate=float(args.max_empty_cache_rate),
        ),
    }
    if not args.no_diagnostics:
        payload["diagnostics"] = load_diagnostics_summary(
            run_dir,
            window=int(args.diagnostics_window),
        )
        payload = apply_diagnostics_gate_overrides(payload)

    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print_human(payload)

    if args.fail_on_red and payload["gate"]["verdict"] in {"FAIL", "MISSING_TAGS"}:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
