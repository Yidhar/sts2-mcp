#!/usr/bin/env python3
"""Summarise recent full-run death decks.

This is intentionally a small standalone monitor helper, not trainer logic.
It reads ``diagnostics/death_deck_summary.jsonl`` emitted by
``muzero.diagnostics.trainer_dumps`` and prints the concrete deck context
behind recent deaths: floor/encounter, card reward behaviour, deck quality
metrics, and the actual card counts.

Usage from ``packages/rl-agent``:

    python scripts/report_fullrun_death_decks.py --tail 8
    python scripts/report_fullrun_death_decks.py --tail 8 --show-cards
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, deque
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
RL_AGENT_ROOT = SCRIPT_DIR.parent
if str(RL_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(RL_AGENT_ROOT))

from sts2_rl.artifacts import resolve_artifact_path, resolve_external_input_path  # noqa: E402

DEFAULT_LOG_BASE = resolve_artifact_path(None, default="runs")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _get(root: Any, *path: str, default: Any = None) -> Any:
    node = root
    for key in path:
        if not isinstance(node, dict):
            return default
        node = node.get(key)
    return default if node is None else node


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    try:
        out = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(out):
        return "n/a"
    return f"{out:.{digits}f}"


def _read_pointer(path: Path) -> str | None:
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return text or None


def _newest_run_with_deaths(base: Path) -> Path | None:
    if not base.exists():
        return None
    dirs = [p for p in base.iterdir() if p.is_dir()]
    dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for run_dir in dirs:
        if (run_dir / "diagnostics" / "death_deck_summary.jsonl").exists():
            return run_dir
    return dirs[0] if dirs else None


def resolve_run_dir(args: argparse.Namespace) -> Path:
    base = resolve_external_input_path(args.log_base, default="runs")
    if args.run_dir:
        return resolve_external_input_path(args.run_dir, root=base)

    pointer_order: list[str] = []
    if args.latest_fullrun:
        pointer_order.append("latest_fullrun_run_id.txt")
    else:
        pointer_order.extend(
            [
                "latest_pass_large_fullrun_run_id.txt",
                "latest_fullrun_run_id.txt",
            ]
        )

    for pointer_name in pointer_order:
        run_id = _read_pointer(base / pointer_name)
        if run_id:
            return (base / run_id).resolve()

    newest = _newest_run_with_deaths(base)
    if newest is None:
        raise FileNotFoundError(f"No run directory found under {base}")
    return newest.resolve()


def read_jsonl_tail(path: Path, limit: int) -> list[dict[str, Any]]:
    rows: deque[dict[str, Any]] = deque(maxlen=max(1, int(limit)))
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                obj = json.loads(text)
            except json.JSONDecodeError as exc:
                print(
                    f"[warn] skip malformed jsonl line {line_no}: {exc}",
                    file=sys.stderr,
                )
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return list(rows)


def _title_counts(row: dict[str, Any]) -> list[dict[str, Any]]:
    counts = _get(row, "deck", "title_counts", default=[])
    return counts if isinstance(counts, list) else []


def _cards(row: dict[str, Any]) -> list[dict[str, Any]]:
    cards = _get(row, "deck", "cards", default=[])
    return [c for c in cards if isinstance(c, dict)] if isinstance(cards, list) else []


def _count_starter_cards(row: dict[str, Any]) -> int:
    stored = _get(row, "deck", "starter_count", default=None)
    if stored is not None:
        return int(round(_safe_float(stored, 0.0)))
    total = 0
    for card in _cards(row):
        title = str(card.get("title") or "").lower()
        card_id = str(card.get("id") or "").lower()
        if (
            "strike_ironclad" in card_id
            or "defend_ironclad" in card_id
            or title in {"打击", "防御", "strike", "defend"}
        ):
            total += 1
    return total


def _count_upgraded(row: dict[str, Any]) -> int:
    stored = _get(row, "deck", "upgraded_count", default=None)
    if stored is not None:
        return int(round(_safe_float(stored, 0.0)))
    total = 0
    for card in _cards(row):
        if bool(card.get("upgraded")) or _safe_int(card.get("upgrade_level"), 0) > 0:
            total += 1
    return total


def _count_nonstarter_cards(row: dict[str, Any]) -> int:
    stored = _get(row, "deck", "nonstarter_count", default=None)
    if stored is not None:
        return int(round(_safe_float(stored, 0.0)))
    return max(0, _safe_int(_get(row, "deck", "size", default=0), 0) - _count_starter_cards(row))


def _metric(row: dict[str, Any], name: str) -> float:
    return _safe_float(_get(row, "deck_quality", name, default=0.0), 0.0)


def _reward_metric(row: dict[str, Any], name: str) -> float:
    return _safe_float(_get(row, "card_reward", name, default=0.0), 0.0)


def _warnings(row: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    floor = _safe_float(_get(row, "episode", "death_floor", default=0.0))
    if _reward_metric(row, "card_reward_skip_rate") >= 0.55:
        warnings.append("high_reward_skip_rate")
    if _reward_metric(row, "card_reward_consecutive_skip_max") >= 3:
        warnings.append("consecutive_reward_skips")
    if _metric(row, "boss_readiness_score") < 0.35:
        warnings.append("low_boss_readiness")
    if _metric(row, "elite_readiness_score") < 0.35:
        warnings.append("low_elite_readiness")
    if _metric(row, "raw_avg_damage_per_energy") < 4.0:
        warnings.append("low_damage_per_energy")
    if _metric(row, "raw_avg_block_per_energy") < 2.0:
        warnings.append("low_block_per_energy")
    if _metric(row, "raw_expected_cards_seen_per_turn") <= 5.05:
        warnings.append("no_draw_engine")
    if _metric(row, "draw_engine_score") < 0.10:
        warnings.append("low_draw_engine_score")
    if _metric(row, "scaling_score") < 0.10:
        warnings.append("low_scaling_score")
    if _metric(row, "raw_expected_playable_attack_damage_per_turn") < 18.0:
        warnings.append("low_playable_damage")
    if _metric(row, "raw_expected_playable_block_per_turn") < 12.0:
        warnings.append("low_playable_block")
    deck_size = _safe_float(_get(row, "deck", "size", default=0.0), 0.0)
    starter_count = float(_count_starter_cards(row))
    starter_ratio = _safe_float(_get(row, "deck", "starter_ratio", default=0.0), 0.0)
    if starter_ratio <= 0.0 and deck_size > 0.0:
        starter_ratio = starter_count / deck_size
    if floor >= 7.0 and starter_count >= 6.0 and starter_ratio >= 0.45:
        warnings.append("starter_heavy_at_death")
    if floor >= 7.0 and _count_nonstarter_cards(row) <= 7:
        warnings.append("low_nonstarter_count")
    return warnings


def _summarise_counts(row: dict[str, Any], limit: int) -> str:
    parts: list[str] = []
    for item in _title_counts(row)[: max(0, limit)]:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "<unknown>")
        count = _safe_int(item.get("count"), 0)
        upgraded = _safe_int(item.get("upgraded_count"), 0)
        if upgraded > 0:
            parts.append(f"{title} x{count}(+{upgraded})")
        else:
            parts.append(f"{title} x{count}")
    return ", ".join(parts) if parts else "n/a"


def _card_line(card: dict[str, Any]) -> str:
    title = str(card.get("title") or card.get("id") or "<unknown>")
    card_type = str(card.get("type") or "?")
    cost = card.get("cost", "?")
    up = _safe_int(card.get("upgrade_level"), 0)
    suffix = f"+{up}" if up > 0 else ""
    return f"{title}{suffix} [{card_type}, cost={cost}]"


def _average(rows: list[dict[str, Any]], getter: Any) -> float:
    values: list[float] = []
    for row in rows:
        value = _safe_float(getter(row), float("nan"))
        if math.isfinite(value):
            values.append(value)
    return sum(values) / len(values) if values else float("nan")


def print_human(rows: list[dict[str, Any]], run_dir: Path, args: argparse.Namespace) -> None:
    print(f"RUN: {run_dir.name}")
    print(f"FILE: {run_dir / 'diagnostics' / 'death_deck_summary.jsonl'}")
    print(f"ROWS: showing last {len(rows)} death deck summaries")
    print()

    if not rows:
        print("No death deck rows found.")
        return

    encounters = Counter(str(_get(r, "final_progress", "encounter_id", default="unknown")) for r in rows)
    warning_counts = Counter(w for r in rows for w in _warnings(r))
    print("AGGREGATE")
    print(
        "  death_floor avg/max: "
        f"{_fmt(_average(rows, lambda r: _get(r, 'episode', 'death_floor')), 2)} / "
        f"{_fmt(max(_safe_float(_get(r, 'episode', 'death_floor')) for r in rows), 0)}"
    )
    print(
        "  reward avg: "
        f"{_fmt(_average(rows, lambda r: _get(r, 'episode', 'reward')), 2)}"
    )
    print(
        "  deck size / starter / nonstarter / upgraded avg: "
        f"{_fmt(_average(rows, lambda r: _get(r, 'deck', 'size')), 2)} / "
        f"{_fmt(_average(rows, _count_starter_cards), 2)} / "
        f"{_fmt(_average(rows, _count_nonstarter_cards), 2)} / "
        f"{_fmt(_average(rows, _count_upgraded), 2)}"
    )
    print(
        "  reward pick/skip rate avg: "
        f"{_fmt(_average(rows, lambda r: _reward_metric(r, 'card_reward_pick_rate')), 2)} / "
        f"{_fmt(_average(rows, lambda r: _reward_metric(r, 'card_reward_skip_rate')), 2)}"
    )
    print(
        "  dmgE/blockE/cardsSeen/playable avg: "
        f"{_fmt(_average(rows, lambda r: _metric(r, 'raw_avg_damage_per_energy')), 2)} / "
        f"{_fmt(_average(rows, lambda r: _metric(r, 'raw_avg_block_per_energy')), 2)} / "
        f"{_fmt(_average(rows, lambda r: _metric(r, 'raw_expected_cards_seen_per_turn')), 2)} / "
        f"{_fmt(_average(rows, lambda r: _metric(r, 'raw_expected_playable_cards_per_turn')), 2)}"
    )
    print(
        "  boss/elite readiness avg: "
        f"{_fmt(_average(rows, lambda r: _metric(r, 'boss_readiness_score')), 2)} / "
        f"{_fmt(_average(rows, lambda r: _metric(r, 'elite_readiness_score')), 2)}"
    )
    print(f"  top encounters: {', '.join(f'{k}:{v}' for k, v in encounters.most_common(5))}")
    if warning_counts:
        print(f"  top warnings: {', '.join(f'{k}:{v}' for k, v in warning_counts.most_common(8))}")
    print()

    print("RECENT DEATH DECKS")
    for idx, row in enumerate(rows, start=1):
        episode = _get(row, "episode", default={})
        progress = _get(row, "final_progress", default={})
        print(
            f"[{idx}] ep={_get(row, 'episode_id', default='?')} "
            f"step={_get(row, 'global_step', default='?')} "
            f"floor={_fmt(_get(episode, 'death_floor'), 0)} "
            f"encounter={_get(progress, 'encounter_id', default='unknown')} "
            f"reward={_fmt(_get(episode, 'reward'), 2)} "
            f"len={_get(episode, 'length', default='?')} "
            f"deck={_get(row, 'deck', 'size', default='?')} "
            f"hp={_fmt(_get(progress, 'hp'), 0)}/{_fmt(_get(progress, 'max_hp'), 0)}"
        )
        print(
            "    reward: "
            f"seen={_fmt(_reward_metric(row, 'card_reward_seen_count'), 0)} "
            f"pick={_fmt(_reward_metric(row, 'card_reward_pick_count'), 0)} "
            f"skip={_fmt(_reward_metric(row, 'card_reward_skip_count'), 0)} "
            f"skip_rate={_fmt(_reward_metric(row, 'card_reward_skip_rate'), 2)} "
            f"consec_max={_fmt(_reward_metric(row, 'card_reward_consecutive_skip_max'), 0)}"
        )
        print(
            "    quality: "
            f"dmg/E={_fmt(_metric(row, 'raw_avg_damage_per_energy'), 2)} "
            f"blk/E={_fmt(_metric(row, 'raw_avg_block_per_energy'), 2)} "
            f"cards_seen={_fmt(_metric(row, 'raw_expected_cards_seen_per_turn'), 2)} "
            f"playable={_fmt(_metric(row, 'raw_expected_playable_cards_per_turn'), 2)} "
            f"play_dmg={_fmt(_metric(row, 'raw_expected_playable_attack_damage_per_turn'), 1)} "
            f"play_blk={_fmt(_metric(row, 'raw_expected_playable_block_per_turn'), 1)} "
            f"boss_ready={_fmt(_metric(row, 'boss_readiness_score'), 2)} "
            f"elite_ready={_fmt(_metric(row, 'elite_readiness_score'), 2)} "
            f"draw={_fmt(_metric(row, 'draw_engine_score'), 2)} "
            f"scaling={_fmt(_metric(row, 'scaling_score'), 2)}"
        )
        print(
            "    hand mix: "
            f"atk={_fmt(_metric(row, 'expected_hand_attack_cards'), 2)} "
            f"skill={_fmt(_metric(row, 'expected_hand_skill_cards'), 2)} "
            f"power={_fmt(_metric(row, 'expected_hand_power_cards'), 2)} "
            f"draw={_fmt(_metric(row, 'expected_hand_draw_cards'), 2)} "
            f"engine={_fmt(_metric(row, 'expected_hand_engine_cards'), 2)} "
            f"scaling={_fmt(_metric(row, 'expected_hand_scaling_cards'), 2)} "
            f"junk={_fmt(_metric(row, 'expected_hand_junk_share'), 2)} "
            f"starter={_count_starter_cards(row)} "
            f"nonstarter={_count_nonstarter_cards(row)} "
            f"upgraded={_count_upgraded(row)}"
        )
        print(f"    cards: {_summarise_counts(row, args.cards_limit)}")
        warnings = _warnings(row)
        print(f"    warnings: {', '.join(warnings) if warnings else 'none'}")
        if args.show_cards:
            for card in _cards(row):
                print(f"      - {_card_line(card)}")
        print()


def build_json_payload(rows: list[dict[str, Any]], run_dir: Path) -> dict[str, Any]:
    return {
        "run": run_dir.name,
        "death_deck_summary_path": str(run_dir / "diagnostics" / "death_deck_summary.jsonl"),
        "rows": rows,
        "aggregate": {
            "count": len(rows),
            "death_floor_avg": _average(rows, lambda r: _get(r, "episode", "death_floor")),
            "reward_avg": _average(rows, lambda r: _get(r, "episode", "reward")),
            "deck_size_avg": _average(rows, lambda r: _get(r, "deck", "size")),
            "starter_count_avg": _average(rows, _count_starter_cards),
            "nonstarter_count_avg": _average(rows, _count_nonstarter_cards),
            "upgraded_count_avg": _average(rows, _count_upgraded),
            "card_reward_pick_rate_avg": _average(rows, lambda r: _reward_metric(r, "card_reward_pick_rate")),
            "card_reward_skip_rate_avg": _average(rows, lambda r: _reward_metric(r, "card_reward_skip_rate")),
            "raw_avg_damage_per_energy_avg": _average(rows, lambda r: _metric(r, "raw_avg_damage_per_energy")),
            "raw_avg_block_per_energy_avg": _average(rows, lambda r: _metric(r, "raw_avg_block_per_energy")),
            "boss_readiness_score_avg": _average(rows, lambda r: _metric(r, "boss_readiness_score")),
            "elite_readiness_score_avg": _average(rows, lambda r: _metric(r, "elite_readiness_score")),
            "warnings": Counter(w for r in rows for w in _warnings(r)),
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-base", default=str(DEFAULT_LOG_BASE), help="runs directory")
    parser.add_argument("--run-dir", help="Run id or run directory. Defaults to latest pass-large/fullrun pointer.")
    parser.add_argument("--latest-fullrun", action="store_true", help="Prefer latest_fullrun_run_id.txt over pass-large pointer.")
    parser.add_argument("--tail", type=int, default=8, help="Number of recent death rows to show.")
    parser.add_argument("--cards-limit", type=int, default=14, help="Max grouped card titles per deck.")
    parser.add_argument("--show-cards", action="store_true", help="Print individual compact cards for each death.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON instead of human text.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run_dir = resolve_run_dir(args)
        path = run_dir / "diagnostics" / "death_deck_summary.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Missing death deck summary: {path}")
        rows = read_jsonl_tail(path, args.tail)
        if args.json:
            payload = build_json_payload(rows, run_dir)
            print(json.dumps(payload, ensure_ascii=False, indent=2, default=dict))
        else:
            print_human(rows, run_dir, args)
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
