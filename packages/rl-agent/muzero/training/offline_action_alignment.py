"""Audit offline decision samples before using them as MuZero policy targets.

Offline human/history rows are valuable, but only if their candidate ordering
matches the action indices that the policy will be supervised on.  This module
performs the pre-BC checks that should run before enabling any offline CE loss:

* every row has a valid selected label index;
* optional skip/no-buy tasks expose the skip candidate consistently;
* route samples contain full per-action route candidates, not only a chosen
  room-type label;
* candidate lists fit inside the current action head.

The output is intentionally conservative.  A task marked ``alignment_ready`` is
ready for the next shadow-policy step; it is not by itself permission to train
with a large offline loss.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from offline_training_data import (
    ACTION_ONLY_CARD_TASKS,
    AUXILIARY_TASKS,
    BUILD_V2_CANDIDATE_TASKS,
    BUILD_V2_CLASS_TASKS,
    BUILD_V2_OPTIONAL_SKIP_TASKS,
    CANDIDATE_TASKS,
    ROUTE_TASKS,
    SHOP_BUNDLE_AUX_FIELDS,
    load_task_rows,
)
from sts2_rl.artifacts import resolve_external_input_path
from sts2_env.observation_common import MAX_ACTIONS


REST_ACTION_FALLBACK_CANDIDATES: tuple[str, ...] = ("rest", "smith")
SHOP_REMOVE_BINARY_CANDIDATES: tuple[str, ...] = ("remove_card", "skip_remove")
SHOP_BUNDLE_AUX_ACTIONS: tuple[tuple[str, str], ...] = (
    ("did_buy_any_card", "shop_bundle_aux:buy_any_card"),
    ("did_buy_any_relic", "shop_bundle_aux:buy_any_relic"),
    ("did_buy_any_potion", "shop_bundle_aux:buy_any_potion"),
    ("did_remove_card", "shop_bundle_aux:remove_card"),
    ("leave_only", "shop_bundle_aux:leave_only"),
)

# These tasks are optional decisions in the online UI: "skip reward" or "stop
# buying and leave/no-purchase" is a real competing action.  If skip is only
# present when it was selected, direct CE would train on a different action set
# than the online policy sees.
OPTIONAL_SKIP_TASKS: frozenset[str] = frozenset(BUILD_V2_OPTIONAL_SKIP_TASKS)


@dataclass(frozen=True)
class OfflineActionSpec:
    index: int
    action_id: str
    kind: str
    label: str
    is_skip: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OfflineRowAlignment:
    sample_id: str
    task: str
    candidate_count: int
    label_index: int | None
    label_valid: bool
    selected_action_id: str | None
    selected_label: str | None
    skip_visible: bool
    skip_selected: bool
    issues: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.issues


@dataclass(frozen=True)
class OfflineActionAlignmentReport:
    task: str
    rows: int
    action_family: str
    alignment_ready: bool
    ready_blockers: tuple[str, ...]
    candidate_count_mean: float
    candidate_count_p95: float
    candidate_count_max: int
    label_index_valid_rate: float | None
    route_candidate_present_rate: float | None
    route_candidate_label_valid_rate: float | None
    skip_visible_rate: float | None
    skip_selected_rate: float | None
    optional_skip_missing_rate: float | None
    max_actions_overflow_rate: float
    issue_counts: dict[str, int] = field(default_factory=dict)
    selected_label_counts: dict[str, int] = field(default_factory=dict)
    example_issues: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_offline_action_specs(
    row: Mapping[str, Any],
    task: str,
    *,
    task_context: Mapping[str, Any] | None = None,
) -> tuple[list[OfflineActionSpec], int | None]:
    """Return action specs and selected index for a normalized offline row."""

    if task in BUILD_V2_CANDIDATE_TASKS or task in CANDIDATE_TASKS:
        specs: list[OfflineActionSpec] = []
        for index, candidate_id in enumerate(row.get("candidate_ids") or []):
            cid = str(candidate_id)
            is_skip = cid == "<skip>"
            specs.append(
                OfflineActionSpec(
                    index=index,
                    action_id=f"{task}:skip" if is_skip else f"{task}:{index}",
                    kind=_candidate_kind(task, cid, row),
                    label=cid,
                    is_skip=is_skip,
                )
            )
        return specs, _safe_int_or_none(row.get("label_index"))

    if task in ROUTE_TASKS:
        specs = []
        for index, candidate in enumerate(row.get("route_candidates") or []):
            if not isinstance(candidate, Mapping):
                continue
            point_type = str(candidate.get("point_type_norm") or candidate.get("point_type") or "Unknown")
            specs.append(
                OfflineActionSpec(
                    index=index,
                    action_id=str(candidate.get("action_id") or f"map:{index}"),
                    kind="map",
                    label=point_type,
                    is_skip=False,
                )
            )
        return specs, _safe_int_or_none(row.get("label_index"))

    if task in BUILD_V2_CLASS_TASKS or task == "rest_site":
        if task in {"rest_action", "rest_site"}:
            labels = set(task_context.get("rest_action_candidates", ())) if task_context else set()
            labels.update(REST_ACTION_FALLBACK_CANDIDATES)
            label = str(row.get("label") or "")
            if label:
                labels.add(label)
            candidates = sorted(labels)
            kind = "rest_site"
        elif task == "shop_remove_binary":
            candidates = list(SHOP_REMOVE_BINARY_CANDIDATES)
            label = str(row.get("label") or "")
            kind = "shop"
        else:
            candidates = []
            label = str(row.get("label") or "")
            kind = "classification"
        specs = [
            OfflineActionSpec(
                index=index,
                action_id=f"{task}:{candidate}",
                kind=kind,
                label=str(candidate),
                is_skip=str(candidate).lower() in {"skip", "skip_remove", "leave", "leave_only"},
            )
            for index, candidate in enumerate(candidates)
        ]
        label_index = None
        for spec in specs:
            if spec.label == label:
                label_index = spec.index
                break
        return specs, label_index

    if task in ACTION_ONLY_CARD_TASKS:
        specs = [
            OfflineActionSpec(
                index=index,
                action_id=f"{task}:{index}",
                kind="deck_upgrade" if task == "upgrade" else "card_selection",
                label=str(card_id),
                is_skip=False,
            )
            for index, card_id in enumerate(row.get("deck_ids") or [])
        ]
        # Cardset tasks are multi-label; no single label index.
        return specs, None

    if task in AUXILIARY_TASKS:
        specs = [
            OfflineActionSpec(
                index=index,
                action_id=action_id,
                kind="shop" if field_name != "leave_only" else "proceed",
                label=field_name,
                is_skip=field_name == "leave_only",
            )
            for index, (field_name, action_id) in enumerate(SHOP_BUNDLE_AUX_ACTIONS)
        ]
        return specs, None

    raise ValueError(f"Unsupported offline action alignment task: {task}")


def audit_offline_row_alignment(
    row: Mapping[str, Any],
    task: str,
    *,
    task_context: Mapping[str, Any] | None = None,
    max_actions: int = MAX_ACTIONS,
) -> OfflineRowAlignment:
    specs, label_index = build_offline_action_specs(row, task, task_context=task_context)
    candidate_count = len(specs)
    issues: list[str] = []

    if candidate_count <= 0:
        issues.append("missing_candidates")
    if candidate_count > max_actions:
        issues.append("candidate_count_exceeds_max_actions")

    label_valid = False
    selected: OfflineActionSpec | None = None
    if task in ACTION_ONLY_CARD_TASKS or task in AUXILIARY_TASKS:
        # Multi-label tasks are audited for candidate shape only here.
        label_valid = candidate_count > 0
    elif label_index is None:
        issues.append("missing_label_index")
    elif not (0 <= label_index < candidate_count):
        issues.append("invalid_label_index")
    else:
        label_valid = True
        selected = specs[label_index]

    if task in ROUTE_TASKS:
        if candidate_count <= 0:
            issues.append("route_missing_candidate_supervision")
            issues.append("route_label_not_candidate_aligned")
        elif label_index is None or not (0 <= label_index < candidate_count):
            issues.append("route_label_not_candidate_aligned")

    skip_visible = any(spec.is_skip for spec in specs)
    skip_selected = bool(selected and selected.is_skip)
    if task in OPTIONAL_SKIP_TASKS and not skip_visible:
        issues.append("optional_skip_candidate_missing")

    return OfflineRowAlignment(
        sample_id=str(row.get("sample_id") or ""),
        task=task,
        candidate_count=candidate_count,
        label_index=label_index,
        label_valid=label_valid,
        selected_action_id=selected.action_id if selected else None,
        selected_label=selected.label if selected else None,
        skip_visible=skip_visible,
        skip_selected=skip_selected,
        issues=tuple(dict.fromkeys(issues)),
    )


def audit_offline_task_alignment(
    task: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    max_actions: int = MAX_ACTIONS,
    max_examples: int = 5,
) -> OfflineActionAlignmentReport:
    context = _build_task_context(task, rows)
    row_reports = [
        audit_offline_row_alignment(row, task, task_context=context, max_actions=max_actions)
        for row in rows
    ]
    issue_counts: Counter[str] = Counter(issue for report in row_reports for issue in report.issues)
    candidate_counts = [report.candidate_count for report in row_reports]
    selected_label_counts = Counter(
        report.selected_label for report in row_reports if report.selected_label not in (None, "")
    )
    skip_applicable = [report for report in row_reports if report.skip_visible or task in OPTIONAL_SKIP_TASKS]
    route_candidate_present = [report.candidate_count > 0 for report in row_reports] if task in ROUTE_TASKS else None
    route_candidate_label_valid = [
        report.candidate_count > 0 and report.label_valid
        for report in row_reports
    ] if task in ROUTE_TASKS else None

    blockers = _ready_blockers(task, rows, row_reports, issue_counts)
    return OfflineActionAlignmentReport(
        task=task,
        rows=len(rows),
        action_family=_action_family(task),
        alignment_ready=len(blockers) == 0,
        ready_blockers=tuple(blockers),
        candidate_count_mean=_mean(candidate_counts),
        candidate_count_p95=_percentile(candidate_counts, 0.95),
        candidate_count_max=max(candidate_counts, default=0),
        label_index_valid_rate=_rate([report.label_valid for report in row_reports]) if row_reports else None,
        route_candidate_present_rate=_rate(route_candidate_present) if route_candidate_present is not None else None,
        route_candidate_label_valid_rate=_rate(route_candidate_label_valid) if route_candidate_label_valid is not None else None,
        skip_visible_rate=_rate([report.skip_visible for report in skip_applicable]) if skip_applicable else None,
        skip_selected_rate=_rate([report.skip_selected for report in skip_applicable]) if skip_applicable else None,
        optional_skip_missing_rate=_rate([not report.skip_visible for report in row_reports]) if task in OPTIONAL_SKIP_TASKS and row_reports else None,
        max_actions_overflow_rate=_rate([report.candidate_count > max_actions for report in row_reports]) if row_reports else 0.0,
        issue_counts=dict(sorted(issue_counts.items())),
        selected_label_counts=dict(sorted((str(k), int(v)) for k, v in selected_label_counts.items())),
        example_issues=[
            {
                "sample_id": report.sample_id,
                "candidate_count": report.candidate_count,
                "label_index": report.label_index,
                "issues": list(report.issues),
            }
            for report in row_reports
            if report.issues
        ][:max_examples],
    )


def audit_offline_alignment_from_disk(
    root: str | Path,
    tasks: Sequence[str],
    *,
    fmt: str = "parquet",
    split: str | None = None,
    partition_kind: str | None = None,
    partition_value: str | None = None,
    max_rows_per_task: int | None = None,
    max_actions: int = MAX_ACTIONS,
) -> dict[str, Any]:
    resolved_root = resolve_external_input_path(root)
    task_reports: dict[str, Any] = {}
    for task in tasks:
        rows = load_task_rows(
            resolved_root,
            task,
            fmt=fmt,
            partition_kind=partition_kind,
            partition_value=partition_value,
            split=split,
        )
        if max_rows_per_task is not None and max_rows_per_task > 0:
            rows = rows[:max_rows_per_task]
        report = audit_offline_task_alignment(task, rows, max_actions=max_actions)
        task_reports[task] = report.to_dict()
    return {
        "root": str(resolved_root),
        "fmt": fmt,
        "split": split,
        "max_actions": int(max_actions),
        "tasks": task_reports,
        "all_ready": all(report["alignment_ready"] for report in task_reports.values()),
    }


def _candidate_kind(task: str, candidate_id: str, row: Mapping[str, Any]) -> str:
    if candidate_id == "<skip>":
        return "proceed"
    option_kind = str(row.get("option_kind") or "")
    if task == "regular_card_reward":
        return "card_reward"
    if task == "smith_target":
        return "deck_upgrade"
    if "shop" in task:
        return "shop"
    if option_kind == "relic":
        return "treasure_relic"
    if option_kind == "potion":
        return "reward"
    if option_kind == "card" or task in {"remove_card_step", "transform_card_step", "event_card_bundle"}:
        return "card_selection"
    return "event_option"


def _build_task_context(task: str, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if task in {"rest_action", "rest_site"}:
        labels = {str(row.get("label")) for row in rows if row.get("label")}
        labels.update(REST_ACTION_FALLBACK_CANDIDATES)
        return {"rest_action_candidates": sorted(labels)}
    return {}


def _ready_blockers(
    task: str,
    rows: Sequence[Mapping[str, Any]],
    reports: Sequence[OfflineRowAlignment],
    issue_counts: Counter[str],
) -> list[str]:
    blockers: list[str] = []
    if not rows:
        blockers.append("empty_task")
    hard_issue_names = {
        "missing_candidates",
        "missing_label_index",
        "invalid_label_index",
        "candidate_count_exceeds_max_actions",
    }
    for issue in hard_issue_names:
        if issue_counts.get(issue, 0) > 0:
            blockers.append(issue)
    if task in ROUTE_TASKS:
        if issue_counts.get("route_missing_candidate_supervision", 0) > 0:
            blockers.append("route_missing_candidate_supervision")
        if issue_counts.get("route_label_not_candidate_aligned", 0) > 0:
            blockers.append("route_label_not_candidate_aligned")
    if task in OPTIONAL_SKIP_TASKS and issue_counts.get("optional_skip_candidate_missing", 0) > 0:
        blockers.append("optional_skip_candidate_missing")
    return sorted(dict.fromkeys(blockers))


def _action_family(task: str) -> str:
    if task in ROUTE_TASKS:
        return "route"
    if task in BUILD_V2_CANDIDATE_TASKS or task in CANDIDATE_TASKS:
        return "candidate"
    if task in BUILD_V2_CLASS_TASKS or task == "rest_site":
        return "classification"
    if task in ACTION_ONLY_CARD_TASKS:
        return "cardset"
    if task in AUXILIARY_TASKS:
        return "auxiliary"
    return "unknown"


def _safe_int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _mean(values: Iterable[int | float | bool]) -> float:
    items = [float(value) for value in values]
    if not items:
        return 0.0
    return float(sum(items) / len(items))


def _rate(values: Iterable[bool]) -> float:
    items = [bool(value) for value in values]
    if not items:
        return 0.0
    return float(sum(1.0 for value in items if value) / len(items))


def _percentile(values: Sequence[int | float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    pos = max(0.0, min(1.0, q)) * (len(ordered) - 1)
    low = int(math.floor(pos))
    high = int(math.ceil(pos))
    if low == high:
        return ordered[low]
    frac = pos - low
    return ordered[low] * (1.0 - frac) + ordered[high] * frac


def _parse_tasks(raw: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit offline action/index alignment before policy BC.")
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Offline dataset root (default: artifact datasets/offline_build_v2_full/parquet).",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="regular_card_reward,smith_target,remove_card_step,shop_remove_binary,"
        "shop_remove_target_step,shop_relic_pick_step,shop_potion_pick_step,rest_action,"
        "route_room_type,route_point_type",
    )
    parser.add_argument("--fmt", type=str, default="parquet")
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--max-rows-per-task", type=int, default=0)
    parser.add_argument("--max-actions", type=int, default=MAX_ACTIONS)
    args = parser.parse_args(argv)

    payload = audit_offline_alignment_from_disk(
        resolve_external_input_path(args.root, default="datasets/offline_build_v2_full/parquet"),
        _parse_tasks(args.tasks),
        fmt=args.fmt,
        split=args.split,
        max_rows_per_task=args.max_rows_per_task or None,
        max_actions=args.max_actions,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
