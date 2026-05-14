#!/usr/bin/env python3
"""Create a compact markdown diagnostics report for a combat sandbox run.

The sandbox gate already prints the most important pass/fail checks.  This
script is the companion "what should we inspect if the gate fails?" report:
it aggregates recent action offenders, deaths, selected cards/actions, and a
small scalar snapshot into a single markdown file.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import sys
from pathlib import Path
from typing import Iterable, Mapping, Sequence


DEFAULT_SCALAR_TAGS = (
    "buffer/size",
    "recent_tail/64/win_rate",
    "recent_tail/64/normal_win_rate",
    "recent_tail/64/hard_normal_win_rate",
    "recent_tail/256/win_rate",
    "recent_tail/256/normal_win_rate",
    "recent_tail/256/hard_normal_win_rate",
    "combat_quality/bad_pure_block_selected_rate",
    "combat_quality/bad_end_turn_selected_rate",
    "combat_quality/wasteful_end_turn_rate",
    "combat_quality/potion_low_urgency_selected_rate",
    "combat_quality/refund_no_followup_with_progress_selected_rate",
    "combat_quality/hp_cost_self_lethal_selected_rate",
    "loss/total",
    "loss/future_world_aux",
    "loss/future_bank_state",
    "memory/max_allocated_gb",
    "memory/reserved_gb",
)


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                value = {"_decode_error": True, "_line_no": line_no, "_raw": line[:500]}
            if isinstance(value, dict):
                rows.append(value)
    return rows


def load_jsonl_glob(paths: Iterable[Path]) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(paths):
        for row in load_jsonl(path):
            row.setdefault("_file", path.name)
            rows.append(row)
    return rows


def row_step(row: Mapping) -> int | None:
    for key in ("global_step", "step", "training_step"):
        value = row.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return int(value)
    return None


def filter_recent_by_step(rows: Sequence[dict], window: int) -> list[dict]:
    if window <= 0 or not rows:
        return list(rows)
    steps = [step for row in rows if (step := row_step(row)) is not None]
    if not steps:
        return list(rows)
    max_step = max(steps)
    min_step = max_step - window
    return [row for row in rows if (row_step(row) is None or row_step(row) >= min_step)]


def nested_get(row: Mapping, path: Sequence[str]) -> object | None:
    cur: object = row
    for key in path:
        if not isinstance(cur, Mapping):
            return None
        cur = cur.get(key)
    return cur


def selected_title(row: Mapping) -> str:
    for path in (
        ("selected_title",),
        ("title",),
        ("action_title",),
        ("action_info", "title"),
        ("selected_action", "title"),
    ):
        value = nested_get(row, path)
        if isinstance(value, str) and value:
            return value
    action_id = nested_get(row, ("action_info", "action_id")) or row.get("action_id")
    return str(action_id) if action_id else "<unknown>"


def encounter_id(row: Mapping) -> str:
    for path in (("encounter_id",), ("encounter",), ("snapshot", "encounter_id")):
        value = nested_get(row, path)
        if isinstance(value, str) and value:
            return value.strip().lower()
    return "<unknown>"


def top_counts(rows: Sequence[Mapping], key: str, limit: int) -> list[tuple[str, int]]:
    counter = collections.Counter(str(row.get(key) or "<unknown>") for row in rows)
    return counter.most_common(limit)


def top_selected_titles(rows: Sequence[Mapping], limit: int) -> list[tuple[str, int]]:
    return collections.Counter(selected_title(row) for row in rows).most_common(limit)


def top_encounters(rows: Sequence[Mapping], limit: int) -> list[tuple[str, int]]:
    return collections.Counter(encounter_id(row) for row in rows).most_common(limit)


STRICT_BAD_END_TURN_CLASSES = frozenset({"bad_end_turn", "bad_end_turn_stable_with_actions"})


def numeric_nested(row: Mapping, path: Sequence[str], default: float = 0.0) -> float:
    value = nested_get(row, path)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return default


def end_turn_class(row: Mapping) -> str:
    value = row.get("end_turn_class")
    return str(value) if value else "<unknown>"


def is_forced_energy_left_no_playable(row: Mapping) -> bool:
    """True for the common benign case: cards are exhausted/unplayable but energy remains.

    This explicitly prevents reports from treating "energy > 0" as evidence of a
    bad End Turn.  The important signal is the stable legal frontier: no
    playable/positive/urgent non-EndTurn action remains.  Some bridge states may
    still expose non-progress actions such as low-urgency potions or status
    cleanup cards in the legal list, so legal_action_count is intentionally not
    required to be one here.
    """

    return (
        end_turn_class(row) == "forced_end_turn"
        and numeric_nested(row, ("player", "energy")) > 0.05
        and numeric_nested(row, ("counts", "playable_cards_left")) <= 0
        and numeric_nested(row, ("counts", "positive_action_count")) <= 0
        and numeric_nested(row, ("counts", "urgent_action_count")) <= 0
    )


def is_forced_energy_left_hand_empty(row: Mapping) -> bool:
    """Stronger benign subset: the hand is literally empty but energy remains."""

    return (
        is_forced_energy_left_no_playable(row)
        and numeric_nested(row, ("combat", "hand_count")) <= 0
    )


def is_transient_only_end_turn(row: Mapping) -> bool:
    flags = row.get("reason_flags") if isinstance(row.get("reason_flags"), Mapping) else {}
    return bool(flags.get("transient_only_end_turn"))


def is_strict_bad_end_turn(row: Mapping) -> bool:
    return end_turn_class(row) in STRICT_BAD_END_TURN_CLASSES


def is_hard_bad_incoming_end_turn(row: Mapping) -> bool:
    """Strict bad EndTurn with incoming damage and an urgent non-EndTurn option.

    These are the least ambiguous cases: the model ended the turn while damage
    was uncovered and a legal urgent action existed.  No-pressure setup/hp-cost
    skips may still be useful to inspect, but should not be confused with this
    hard tactical failure.
    """

    incoming = numeric_nested(row, ("combat", "incoming_damage"))
    block = numeric_nested(row, ("player", "block"))
    urgent = numeric_nested(row, ("counts", "urgent_action_count"))
    return is_strict_bad_end_turn(row) and incoming > block + 0.05 and urgent > 0


def summarize_end_turn_contexts(rows: Sequence[Mapping], limit: int) -> dict[str, object]:
    class_counts = collections.Counter(end_turn_class(row) for row in rows)
    strict_bad = [row for row in rows if is_strict_bad_end_turn(row)]
    hard_bad = [row for row in strict_bad if is_hard_bad_incoming_end_turn(row)]
    forced_energy_left = [row for row in rows if is_forced_energy_left_no_playable(row)]
    forced_energy_left_hand_empty = [row for row in rows if is_forced_energy_left_hand_empty(row)]
    transient_only = [row for row in rows if is_transient_only_end_turn(row)]
    soft_or_ambiguous_bad = [row for row in strict_bad if row not in hard_bad]
    return {
        "total": len(rows),
        "class_counts": class_counts.most_common(),
        "strict_bad": strict_bad,
        "hard_bad": hard_bad,
        "soft_or_ambiguous_bad": soft_or_ambiguous_bad,
        "forced_energy_left": forced_energy_left,
        "forced_energy_left_hand_empty": forced_energy_left_hand_empty,
        "transient_only": transient_only,
        "top_strict_bad_encounters": top_encounters(strict_bad, limit),
    }


def compact_end_turn_example(row: Mapping) -> tuple[object, ...]:
    player = row.get("player") if isinstance(row.get("player"), Mapping) else {}
    combat = row.get("combat") if isinstance(row.get("combat"), Mapping) else {}
    counts = row.get("counts") if isinstance(row.get("counts"), Mapping) else {}
    actions = row.get("top_legal_actions") if isinstance(row.get("top_legal_actions"), Sequence) else []
    non_end_titles = []
    for action in actions:
        if not isinstance(action, Mapping):
            continue
        if action.get("family") == "end_turn":
            continue
        title = action.get("title") or action.get("action_id") or "<action>"
        tags = action.get("tags") if isinstance(action.get("tags"), Sequence) else []
        tag_text = ",".join(str(tag) for tag in tags[:3])
        non_end_titles.append(f"{title} [{tag_text}]" if tag_text else str(title))
        if len(non_end_titles) >= 3:
            break
    return (
        row_step(row),
        encounter_id(row),
        row.get("turn", ""),
        end_turn_class(row),
        f"{player.get('hp', '')}/{player.get('max_hp', '')}",
        player.get("block", ""),
        player.get("energy", ""),
        combat.get("incoming_damage", ""),
        combat.get("hand_count", ""),
        counts.get("legal_action_count", ""),
        counts.get("playable_cards_left", ""),
        counts.get("positive_action_count", ""),
        counts.get("urgent_action_count", ""),
        "; ".join(non_end_titles) if non_end_titles else "<none>",
    )


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def read_scalars(run_dir: Path, tags: Sequence[str]) -> list[tuple[str, int, float]]:
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except Exception:
        return []

    try:
        acc = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
        acc.Reload()
    except Exception:
        return []

    available = set(acc.Tags().get("scalars", []))
    out: list[tuple[str, int, float]] = []
    for tag in tags:
        if tag not in available:
            continue
        events = acc.Scalars(tag)
        if not events:
            continue
        ev = events[-1]
        out.append((tag, int(ev.step), float(ev.value)))
    return out


def priority_rows(offenders: Sequence[Mapping], deaths: Sequence[Mapping], limit: int) -> list[tuple[str, int, int, int]]:
    offender_counts = collections.Counter(encounter_id(row) for row in offenders)
    death_counts = collections.Counter(encounter_id(row) for row in deaths)
    keys = set(offender_counts) | set(death_counts)
    scored = []
    for key in keys:
        deaths_n = death_counts[key]
        offenders_n = offender_counts[key]
        score = deaths_n * 10 + offenders_n
        scored.append((key, deaths_n, offenders_n, score))
    scored.sort(key=lambda item: (-item[3], item[0]))
    return scored[:limit]


def render_report(run_dir: Path, *, recent_window: int, top_n: int) -> str:
    diagnostics_dir = run_dir / "diagnostics"
    offenders_all = load_jsonl(diagnostics_dir / "action_offenders.jsonl")
    deaths_all = load_jsonl_glob((diagnostics_dir / "death_slices").glob("*.jsonl"))
    end_turn_all = load_jsonl(diagnostics_dir / "end_turn_contexts.jsonl")
    offenders = filter_recent_by_step(offenders_all, recent_window)
    deaths = filter_recent_by_step(deaths_all, recent_window)
    end_turn_rows = filter_recent_by_step(end_turn_all, recent_window)

    lines: list[str] = []
    lines.append(f"# Combat Sandbox Diagnostics: `{run_dir.name}`")
    lines.append("")
    lines.append("## Scalar snapshot")
    scalars = read_scalars(run_dir, DEFAULT_SCALAR_TAGS)
    if scalars:
        lines.append(markdown_table(("tag", "step", "value"), [(tag, step, f"{value:.6g}") for tag, step, value in scalars]))
    else:
        lines.append("_No TensorBoard scalars available._")
    lines.append("")

    lines.append("## Recent window")
    lines.append("")
    lines.append(f"- recent_window_steps: `{recent_window}`")
    lines.append(f"- offenders: `{len(offenders)}` / all `{len(offenders_all)}`")
    lines.append(f"- deaths: `{len(deaths)}` / all `{len(deaths_all)}`")
    lines.append(f"- end_turn_contexts: `{len(end_turn_rows)}` / all `{len(end_turn_all)}`")
    lines.append("")

    lines.append("## Strict EndTurn taxonomy")
    lines.append("")
    if end_turn_rows:
        summary = summarize_end_turn_contexts(end_turn_rows, top_n)
        lines.append(
            markdown_table(
                ("metric", "count"),
                (
                    ("total_end_turn_contexts", summary["total"]),
                    ("strict_bad_end_turn", len(summary["strict_bad"])),
                    ("hard_bad_incoming_end_turn", len(summary["hard_bad"])),
                    ("soft_or_ambiguous_bad_end_turn", len(summary["soft_or_ambiguous_bad"])),
                    ("forced_energy_left_no_playable", len(summary["forced_energy_left"])),
                    ("forced_energy_left_hand_empty", len(summary["forced_energy_left_hand_empty"])),
                    ("transient_only_end_turn", len(summary["transient_only"])),
                ),
            )
        )
        lines.append("")
        lines.append(
            "_Interpretation: never treat `energy > 0` by itself as a bug.  Leftover energy is benign when "
            "the stable legal frontier has no playable/positive/urgent non-EndTurn action; the strongest "
            "benign subset is `forced_energy_left_hand_empty` where the hand is literally empty.  Treat hard "
            "bad rows as tactical failures; treat no-pressure setup/hp-cost rows as ambiguous until manually "
            "inspected._"
        )
        lines.append("")
        class_counts = summary["class_counts"]
        if class_counts:
            lines.append("### EndTurn class counts")
            lines.append(markdown_table(("end_turn_class", "count"), class_counts))
            lines.append("")
        top_bad = summary["top_strict_bad_encounters"]
        if top_bad:
            lines.append("### Top strict bad EndTurn encounters")
            lines.append(markdown_table(("encounter", "strict_bad_count"), top_bad))
            lines.append("")
        hard_bad_examples = summary["hard_bad"][-min(5, len(summary["hard_bad"])) :]
        if hard_bad_examples:
            lines.append("### Latest hard bad EndTurn examples")
            lines.append(
                markdown_table(
                    (
                        "step",
                        "encounter",
                        "turn",
                        "class",
                        "hp",
                        "block",
                        "energy",
                        "incoming",
                        "hand",
                        "legal",
                        "playable",
                        "positive",
                        "urgent",
                        "non_end_actions",
                    ),
                    [compact_end_turn_example(row) for row in hard_bad_examples],
                )
            )
            lines.append("")
        forced_examples = summary["forced_energy_left"][-min(3, len(summary["forced_energy_left"])) :]
        if forced_examples:
            lines.append("### Benign forced EndTurn with energy left examples")
            lines.append(
                "_These rows are the explicit “cards are gone/unplayable but energy remains” control group: "
                "`playable=0`, `positive=0`, `urgent=0`; if `hand=0` then the hand was literally empty._"
            )
            lines.append(
                markdown_table(
                    (
                        "step",
                        "encounter",
                        "turn",
                        "class",
                        "hp",
                        "block",
                        "energy",
                        "incoming",
                        "hand",
                        "legal",
                        "playable",
                        "positive",
                        "urgent",
                        "non_end_actions",
                    ),
                    [compact_end_turn_example(row) for row in forced_examples],
                )
            )
            lines.append("")
    else:
        lines.append("_No strict EndTurn context rows available._")
        lines.append("")

    lines.append("## Patch priority by encounter")
    priorities = priority_rows(offenders, deaths, top_n)
    if priorities:
        lines.append(markdown_table(("encounter", "recent_deaths", "recent_offenders", "priority_score"), priorities))
    else:
        lines.append("_No recent encounter rows._")
    lines.append("")

    lines.append("## Top offender types")
    lines.append("")
    lines.append(
        "_Note: `action_offenders.jsonl` rows are broad candidate diagnostics.  "
        "For EndTurn triage, prefer the strict taxonomy section above over broad "
        "`bad_end_turn`/`wasteful_end_turn` offender counts._"
    )
    lines.append(markdown_table(("offender_type", "count"), top_counts(offenders, "offender_type", top_n)) if offenders else "_No offenders._")
    lines.append("")

    lines.append("## Top offender encounters")
    lines.append(markdown_table(("encounter", "count"), top_encounters(offenders, top_n)) if offenders else "_No offenders._")
    lines.append("")

    lines.append("## Top selected titles/actions among offenders")
    lines.append(markdown_table(("selected", "count"), top_selected_titles(offenders, top_n)) if offenders else "_No offenders._")
    lines.append("")

    lines.append("## Recent death encounters")
    lines.append(markdown_table(("encounter", "count"), top_encounters(deaths, top_n)) if deaths else "_No deaths._")
    lines.append("")

    lines.append("## Latest death tail compact")
    for row in deaths[-min(5, len(deaths)) :]:
        lines.append("")
        lines.append(f"### death step={row_step(row)} encounter={encounter_id(row)} file={row.get('_file', '')}")
        tail_steps = row.get("tail_steps") or []
        compact_rows = []
        for step in tail_steps[-5:]:
            if not isinstance(step, Mapping):
                continue
            action_info = step.get("action_info") if isinstance(step.get("action_info"), Mapping) else {}
            search_stats = step.get("search_stats") if isinstance(step.get("search_stats"), Mapping) else {}
            compact_rows.append(
                (
                    step.get("phase", ""),
                    step.get("reward", ""),
                    action_info.get("title") or action_info.get("action_id") or "",
                    search_stats.get("combat_quality_wasteful_end_turn_selected", ""),
                    search_stats.get("combat_quality_potion_low_urgency_selected", ""),
                    search_stats.get("combat_quality_refund_no_followup_with_progress_selected", ""),
                )
            )
        if compact_rows:
            lines.append(
                markdown_table(
                    ("phase", "reward", "action", "legacy_wasteful_end", "low_potion", "refund_no_followup"),
                    compact_rows,
                )
            )
        else:
            lines.append("_No compact tail_steps available._")

    lines.append("")
    lines.append("## Suggested next inspection order")
    lines.append("")
    lines.append("1. If `buffer/size < 10000`, do not patch solely from this report; keep training unless loss/memory/self-lethal redlines fire.")
    lines.append("2. If gate FAIL persists after `buffer/size >= 10000`, inspect the top priority encounters above first.")
    lines.append("3. For tactical regressions, prefer narrow encounter/card guards over global policy changes.")
    lines.append("4. Any semantic guard/reward change should restart with fresh replay and optimizer.")
    lines.append("")
    return "\n".join(lines)


def resolve_run_dir(repo_dir: Path, run_dir_arg: str | None) -> Path:
    if run_dir_arg:
        path = Path(run_dir_arg).expanduser()
        return path if path.is_absolute() else repo_dir / path
    latest = (repo_dir / "logs_muzero/latest_lowmem_run_id.txt").read_text(encoding="utf-8").strip()
    return repo_dir / "logs_muzero" / latest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--recent-window", type=int, default=2000)
    parser.add_argument("--top-n", type=int, default=12)
    parser.add_argument("--output", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    repo_dir = Path(args.repo_dir).expanduser().resolve()
    run_dir = resolve_run_dir(repo_dir, args.run_dir)
    report = render_report(run_dir, recent_window=args.recent_window, top_n=args.top_n)
    if args.output:
        output = Path(args.output).expanduser()
        if not output.is_absolute():
            output = repo_dir / output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report, encoding="utf-8")
        print(output)
    else:
        sys.stdout.write(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
