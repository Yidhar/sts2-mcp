#!/usr/bin/env python3
"""Audit human demonstration decisions before enabling behaviour cloning.

The trainer can only imitate rows that contain the exact decision surface:

* raw/encoded ``obs``;
* ordered ``legal_actions``; and
* ``selected_action_id`` present in that ordered legal-action list.

This script is intentionally stricter and more operator-facing than
``validate_human_demos.py``.  It accepts a file, a session directory, or a root
directory containing many sessions, ignores ``episodes.jsonl`` sidecars, and
prints a compact JSON report that answers the question: "is this dataset safe
to use for train-time policy alignment?".
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


RL_ROOT = Path(__file__).resolve().parents[1]
if str(RL_ROOT) not in sys.path:
    sys.path.insert(0, str(RL_ROOT))

from sts2_env.observation_common import MAX_ACTIONS  # noqa: E402


def _split_paths(raw_paths: Iterable[str | Path]) -> list[Path]:
    out: list[Path] = []
    for raw in raw_paths:
        for part in str(raw).replace(";", ",").split(","):
            part = part.strip()
            if part:
                out.append(Path(part))
    return out


def expand_demo_paths(paths: Iterable[str | Path]) -> list[Path]:
    """Expand user paths to candidate decision JSONL files.

    Directory handling mirrors train-time human-demo alignment:
    ``decisions.jsonl`` files are preferred recursively.  If a directory has no
    canonical recorder files, we fall back to non-``episodes.jsonl`` JSONLs so
    older custom exports remain auditable.
    """

    candidates: list[Path] = []
    for path in _split_paths(paths):
        if path.is_dir():
            jsonls = sorted(path.rglob("*.jsonl"))
            decisions = [item for item in jsonls if item.name == "decisions.jsonl"]
            candidates.extend(decisions or [item for item in jsonls if item.name != "episodes.jsonl"])
        else:
            candidates.append(path)

    seen: set[str] = set()
    unique: list[Path] = []
    for path in candidates:
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def _rate(numer: int | float, denom: int | float) -> float:
    denom_f = float(denom)
    if denom_f <= 0:
        return 0.0
    return float(numer) / denom_f


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _safe_action_id(action: Any) -> str:
    if not isinstance(action, dict):
        return ""
    return str(action.get("action_id") or "")


def _row_event(row: Any) -> str | None:
    if not isinstance(row, dict):
        return None
    event = row.get("event")
    if event is None:
        return None
    return str(event)


def _selected_index_from_id(legal_actions: list[Any], selected_action_id: str) -> int:
    for idx, action in enumerate(legal_actions):
        if _safe_action_id(action) == selected_action_id:
            return idx
    return -1


def _classify_bad_row(row: Any) -> str:
    if not isinstance(row, dict):
        return "row_not_object"
    missing: list[str] = []
    if not isinstance(row.get("obs"), dict):
        missing.append("obs")
    legal = row.get("legal_actions")
    if not isinstance(legal, list) or not legal:
        missing.append("legal_actions")
    if row.get("selected_action_id") in (None, ""):
        missing.append("selected_action_id")
    if missing:
        return "missing_or_invalid_" + "_".join(missing)
    selected = str(row.get("selected_action_id") or "")
    selected_idx = _selected_index_from_id(legal, selected)
    if selected_idx < 0:
        return "selected_action_not_in_legal"
    return "unknown_invalid"


def audit_demo_file(path: str | Path, *, max_actions: int = MAX_ACTIONS) -> dict[str, Any]:
    """Audit one JSONL file and return a JSON-serializable report."""

    p = Path(path)
    report: dict[str, Any] = {
        "path": str(p),
        "exists": p.exists(),
        "raw_line_count": 0,
        "blank_line_count": 0,
        "json_error_count": 0,
        "sidecar_event_row_count": 0,
        "decision_row_count": 0,
        "usable_sample_count": 0,
        "invalid_row_count": 0,
        "invalid_reasons": {},
        "obs_present_count": 0,
        "legal_actions_present_count": 0,
        "selected_action_id_present_count": 0,
        "selected_action_id_in_legal_count": 0,
        "selected_action_index_present_count": 0,
        "selected_action_index_match_count": 0,
        "selected_action_index_lt_max_actions_count": 0,
        "transition_hp_loss_present_count": 0,
        "outcome_hp_loss_present_count": 0,
        "combat_win_present_count": 0,
        "tier_counts": {},
        "encounter_counts": {},
        "source_counts": {},
        "max_legal_actions": 0,
    }
    invalid_reasons: Counter[str] = Counter()
    tier_counts: Counter[str] = Counter()
    encounter_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()

    if not p.exists():
        report["invalid_row_count"] = 1
        report["invalid_reasons"] = {"file_missing": 1}
        return report

    with p.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            report["raw_line_count"] += 1
            line = raw_line.strip()
            if not line:
                report["blank_line_count"] += 1
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                report["json_error_count"] += 1
                invalid_reasons["invalid_json"] += 1
                continue

            event = _row_event(row)
            if event not in (None, "", "decision"):
                report["sidecar_event_row_count"] += 1
                continue

            report["decision_row_count"] += 1
            if not isinstance(row, dict):
                invalid_reasons["row_not_object"] += 1
                continue

            obs_ok = isinstance(row.get("obs"), dict)
            legal_actions = row.get("legal_actions")
            legal_ok = isinstance(legal_actions, list) and len(legal_actions) > 0
            selected_id = str(row.get("selected_action_id") or "")
            selected_present = bool(selected_id)

            if obs_ok:
                report["obs_present_count"] += 1
            if legal_ok:
                report["legal_actions_present_count"] += 1
                report["max_legal_actions"] = max(int(report["max_legal_actions"]), len(legal_actions))
            if selected_present:
                report["selected_action_id_present_count"] += 1

            selected_idx_from_id = -1
            if legal_ok and selected_present:
                selected_idx_from_id = _selected_index_from_id(legal_actions, selected_id)
                if selected_idx_from_id >= 0:
                    report["selected_action_id_in_legal_count"] += 1

            selected_index_raw = row.get("selected_action_index")
            selected_index: int | None = None
            try:
                if selected_index_raw is not None:
                    selected_index = int(selected_index_raw)
            except (TypeError, ValueError):
                selected_index = None
            if selected_index is not None:
                report["selected_action_index_present_count"] += 1
                if selected_idx_from_id >= 0 and selected_index == selected_idx_from_id:
                    report["selected_action_index_match_count"] += 1

            effective_index = selected_index if selected_index is not None else selected_idx_from_id
            index_lt_max_actions = 0 <= int(effective_index) < min(
                int(max_actions), len(legal_actions) if legal_ok else 0
            )
            if index_lt_max_actions:
                report["selected_action_index_lt_max_actions_count"] += 1

            transition = row.get("transition")
            if isinstance(transition, dict) and _safe_float(transition.get("hp_loss")) is not None:
                report["transition_hp_loss_present_count"] += 1
            outcome = row.get("outcome")
            if isinstance(outcome, dict):
                if _safe_float(outcome.get("hp_loss")) is not None:
                    report["outcome_hp_loss_present_count"] += 1
                if outcome.get("combat_win") is not None:
                    report["combat_win_present_count"] += 1

            if row.get("tier") not in (None, ""):
                tier_counts[str(row.get("tier"))] += 1
            if row.get("encounter_id") not in (None, ""):
                encounter_counts[str(row.get("encounter_id"))] += 1
            if row.get("source") not in (None, ""):
                source_counts[str(row.get("source"))] += 1

            usable = (
                obs_ok
                and legal_ok
                and selected_present
                and selected_idx_from_id >= 0
                and index_lt_max_actions
            )
            if usable:
                report["usable_sample_count"] += 1
            else:
                invalid_reasons[_classify_bad_row(row)] += 1

    report["invalid_row_count"] = int(sum(invalid_reasons.values()))
    report["invalid_reasons"] = dict(sorted(invalid_reasons.items()))
    report["tier_counts"] = dict(sorted(tier_counts.items()))
    report["encounter_counts"] = dict(sorted(encounter_counts.items()))
    report["source_counts"] = dict(sorted(source_counts.items()))
    return report


def _sum_counter_dict(files: list[dict[str, Any]], key: str) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for item in files:
        payload = item.get(key)
        if isinstance(payload, dict):
            for k, v in payload.items():
                try:
                    counter[str(k)] += int(v)
                except (TypeError, ValueError):
                    continue
    return dict(sorted(counter.items()))


def audit_demo_dataset(
    paths: Iterable[str | Path],
    *,
    max_actions: int = MAX_ACTIONS,
    min_samples: int = 500,
    min_selected_in_legal_rate: float = 0.98,
    min_index_lt_max_actions_rate: float = 0.98,
    min_transition_hp_loss_rate: float = 0.0,
    min_outcome_hp_loss_rate: float = 0.0,
) -> dict[str, Any]:
    expanded = expand_demo_paths(paths)
    files = [audit_demo_file(path, max_actions=max_actions) for path in expanded]

    decision_rows = sum(int(item.get("decision_row_count") or 0) for item in files)
    usable = sum(int(item.get("usable_sample_count") or 0) for item in files)
    selected_in_legal = sum(int(item.get("selected_action_id_in_legal_count") or 0) for item in files)
    index_lt = sum(int(item.get("selected_action_index_lt_max_actions_count") or 0) for item in files)
    transition_hp = sum(int(item.get("transition_hp_loss_present_count") or 0) for item in files)
    outcome_hp = sum(int(item.get("outcome_hp_loss_present_count") or 0) for item in files)
    combat_win = sum(int(item.get("combat_win_present_count") or 0) for item in files)

    selected_rate = _rate(selected_in_legal, decision_rows)
    index_rate = _rate(index_lt, decision_rows)
    transition_hp_rate = _rate(transition_hp, decision_rows)
    outcome_hp_rate = _rate(outcome_hp, decision_rows)

    readiness_reasons: list[str] = []
    if not expanded:
        readiness_reasons.append("no_jsonl_files_matched")
    if usable < int(min_samples):
        readiness_reasons.append(f"usable_sample_count<{int(min_samples)}")
    if selected_rate < float(min_selected_in_legal_rate):
        readiness_reasons.append(f"selected_action_id_in_legal_rate<{float(min_selected_in_legal_rate):.3f}")
    if index_rate < float(min_index_lt_max_actions_rate):
        readiness_reasons.append(
            f"selected_action_index_lt_max_actions_rate<{float(min_index_lt_max_actions_rate):.3f}"
        )
    if transition_hp_rate < float(min_transition_hp_loss_rate):
        readiness_reasons.append(f"transition_hp_loss_present_rate<{float(min_transition_hp_loss_rate):.3f}")
    if outcome_hp_rate < float(min_outcome_hp_loss_rate):
        readiness_reasons.append(f"outcome_hp_loss_present_rate<{float(min_outcome_hp_loss_rate):.3f}")

    return {
        "status": "ok",
        "input_paths": [str(path) for path in _split_paths(paths)],
        "expanded_paths": [str(path) for path in expanded],
        "file_count": len(files),
        "max_actions": int(max_actions),
        "min_samples": int(min_samples),
        "raw_line_count": sum(int(item.get("raw_line_count") or 0) for item in files),
        "blank_line_count": sum(int(item.get("blank_line_count") or 0) for item in files),
        "json_error_count": sum(int(item.get("json_error_count") or 0) for item in files),
        "sidecar_event_row_count": sum(int(item.get("sidecar_event_row_count") or 0) for item in files),
        "decision_row_count": decision_rows,
        "usable_sample_count": usable,
        "invalid_row_count": sum(int(item.get("invalid_row_count") or 0) for item in files),
        "selected_action_id_in_legal_rate": selected_rate,
        "selected_action_index_match_rate": _rate(
            sum(int(item.get("selected_action_index_match_count") or 0) for item in files),
            sum(int(item.get("selected_action_index_present_count") or 0) for item in files),
        ),
        "selected_action_index_lt_max_actions_rate": index_rate,
        "transition_hp_loss_present_rate": transition_hp_rate,
        "outcome_hp_loss_present_rate": outcome_hp_rate,
        "combat_win_present_rate": _rate(combat_win, decision_rows),
        "max_legal_actions": max([int(item.get("max_legal_actions") or 0) for item in files] or [0]),
        "tier_counts": _sum_counter_dict(files, "tier_counts"),
        "encounter_counts": _sum_counter_dict(files, "encounter_counts"),
        "source_counts": _sum_counter_dict(files, "source_counts"),
        "ready_for_shadow": usable > 0 and selected_rate >= min_selected_in_legal_rate and index_rate >= min_index_lt_max_actions_rate,
        "ready_for_training": len(readiness_reasons) == 0,
        "ready_for_hp_outcome_alignment": (
            usable >= int(min_samples)
            and selected_rate >= min_selected_in_legal_rate
            and outcome_hp_rate >= max(float(min_outcome_hp_loss_rate), 0.80)
        ),
        "readiness_reasons": readiness_reasons,
        "files": files,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="Demo JSONL file(s), session dir(s), or root demo dir(s).")
    parser.add_argument("--max-actions", type=int, default=MAX_ACTIONS)
    parser.add_argument("--min-samples", type=int, default=500)
    parser.add_argument("--min-selected-in-legal-rate", type=float, default=0.98)
    parser.add_argument("--min-index-lt-max-actions-rate", type=float, default=0.98)
    parser.add_argument("--min-transition-hp-loss-rate", type=float, default=0.0)
    parser.add_argument("--min-outcome-hp-loss-rate", type=float, default=0.0)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if the dataset is not ready for training.",
    )
    args = parser.parse_args(argv)

    report = audit_demo_dataset(
        args.paths,
        max_actions=args.max_actions,
        min_samples=args.min_samples,
        min_selected_in_legal_rate=args.min_selected_in_legal_rate,
        min_index_lt_max_actions_rate=args.min_index_lt_max_actions_rate,
        min_transition_hp_loss_rate=args.min_transition_hp_loss_rate,
        min_outcome_hp_loss_rate=args.min_outcome_hp_loss_rate,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    if args.strict and not bool(report.get("ready_for_training")):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
