#!/usr/bin/env python3
"""Validate human demonstration JSONL files.

Accepts either a ``decisions.jsonl`` file or a session directory containing one.
The base schema is validated through ``muzero.demo_dataset``; this script adds
operator-facing counts for runtime identity fields that matter for the recent
card-state fixes (instance uuid, modified cost, exhaust/ethereal/retain/replay,
selection effects, enchantments).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

RL_ROOT = Path(__file__).resolve().parents[1]
if str(RL_ROOT) not in sys.path:
    sys.path.insert(0, str(RL_ROOT))

from muzero.demo_dataset import DemoValidationError, iter_demo_jsonl  # noqa: E402


RUNTIME_KEYS = (
    "instance_uuid",
    "modified_cost",
    "cost_for_turn",
    "exhaust",
    "ethereal",
    "retain",
    "replay",
    "enchantments",
    "selection_effect",
)


def _resolve_path(path: Path) -> Path:
    if path.is_dir():
        return path / "decisions.jsonl"
    return path


def _card(action: dict[str, Any]) -> dict[str, Any]:
    return action.get("card") if isinstance(action.get("card"), dict) else {}


def _runtime_dict(action: dict[str, Any]) -> dict[str, Any]:
    card = _card(action)
    runtime = card.get("runtime")
    if isinstance(runtime, dict):
        return runtime
    runtime = action.get("runtime")
    return runtime if isinstance(runtime, dict) else {}


def _has_any_key(action: dict[str, Any], key: str) -> bool:
    card = _card(action)
    runtime = _runtime_dict(action)
    return key in card or key in runtime or key in action


def validate(path: Path, *, strict_runtime: bool = False) -> dict[str, Any]:
    p = _resolve_path(path)
    samples = list(iter_demo_jsonl(p, strict=True))
    selected_runtime_counts = {key: 0 for key in RUNTIME_KEYS}
    legal_runtime_counts = {key: 0 for key in RUNTIME_KEYS}
    selected_missing_runtime_rows: list[int] = []
    for row_idx, sample in enumerate(samples, start=1):
        selected = sample.legal_actions[sample.selected_action_index]
        if not _runtime_dict(selected):
            selected_missing_runtime_rows.append(row_idx)
        for key in RUNTIME_KEYS:
            if _has_any_key(selected, key):
                selected_runtime_counts[key] += 1
            if any(_has_any_key(action, key) for action in sample.legal_actions):
                legal_runtime_counts[key] += 1
    if strict_runtime and selected_missing_runtime_rows:
        raise DemoValidationError(
            "selected actions missing runtime dict on rows: "
            + ", ".join(map(str, selected_missing_runtime_rows[:50]))
            + (" ..." if len(selected_missing_runtime_rows) > 50 else "")
        )
    return {
        "status": "ok",
        "path": str(p),
        "sample_count": len(samples),
        "selected_runtime_counts": selected_runtime_counts,
        "legal_runtime_counts": legal_runtime_counts,
        "selected_missing_runtime_count": len(selected_missing_runtime_rows),
        "selected_missing_runtime_rows_head": selected_missing_runtime_rows[:50],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--strict-runtime", action="store_true")
    args = parser.parse_args(argv)
    reports = []
    for path in args.paths:
        reports.append(validate(path, strict_runtime=args.strict_runtime))
    print(json.dumps(reports, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
