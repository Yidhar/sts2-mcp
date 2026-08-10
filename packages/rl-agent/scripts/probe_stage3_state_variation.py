#!/usr/bin/env python3
"""Stage-three paired state probes (reset doc §11 item 4).

Replays real journal decision snapshots through the trained macro candidate-Q
model, then re-encodes controlled mutations of the same observation and
reports how the Q ranking over macro branches moves:

  rest surfaces : player HP swept low/high — does Q(rest)-Q(best smith) move?
  shop surfaces : player gold swept broke/rich — does Q(buy/remove)-Q(leave)
                  move?

The probe checks that the learning path is ALIVE (state reaches the ranking);
it deliberately does not grade any direction as correct ("low HP must rest"
is not encoded, per §11).

Usage (WSL ROCm venv, from the package root):
    python scripts/probe_stage3_state_variation.py \
        --config config/experiments/stage2_isolated_macro_v1.toml \
        --champion <champion checkpoint dir> \
        --macro <stage2-macro-online.pt> \
        --journal <evaluation-step jsonl with full DTO snapshots> [--limit 40]
"""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from sts2_rl.semantics.forward import forward_decision
from sts2_rl.training import build_training_resources, load_training_config


def _q_for(
    model: Any,
    encoder: Any,
    device: torch.device,
    observation: dict[str, Any],
    actions: list[dict[str, Any]],
) -> torch.Tensor | None:
    encoded = encoder.encode(observation, actions, device=device)
    hidden = model.initial_state(1, device=device)
    with torch.no_grad():
        output = model(encoded.batch, hidden, validate=False)
    q_values = output.transaction_q_values
    return None if q_values is None else q_values[0].float().cpu()


def _branch_gap(
    q_values: torch.Tensor,
    decision: Any,
    *,
    default_branch: str,
) -> float | None:
    """Q(best non-default branch) - Q(default branch entry)."""

    default_q: float | None = None
    other_q: float | None = None
    for candidate in decision.candidates:
        if candidate.native_index is None:
            continue
        value = float(q_values[int(candidate.native_index)].item())
        if candidate.branch == default_branch:
            default_q = value if default_q is None else max(default_q, value)
        else:
            other_q = value if other_q is None else max(other_q, value)
    if default_q is None or other_q is None:
        return None
    return other_q - default_q


def _set_hp(observation: dict[str, Any], fraction: float) -> dict[str, Any]:
    mutated = copy.deepcopy(observation)
    player = mutated.get("player")
    if isinstance(player, dict):
        max_hp = player.get("max_hp") or player.get("hp_max") or 80
        player["hp"] = max(1, int(round(float(max_hp) * fraction)))
    return mutated

def _set_gold(observation: dict[str, Any], gold: int) -> dict[str, Any]:
    mutated = copy.deepcopy(observation)
    player = mutated.get("player")
    if isinstance(player, dict):
        player["gold"] = int(gold)
    return mutated


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--champion", required=True)
    parser.add_argument("--macro", required=True)
    parser.add_argument("--journal", required=True, nargs="+")
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--device", default=None)
    parser.add_argument("--report-out", default="stage3-state-variation.json")
    args = parser.parse_args()

    config = load_training_config(profile="preheat", config_path=Path(args.config))
    if args.device is not None:
        config = replace(config, runtime=replace(config.runtime, device=str(args.device)))
    resources = build_training_resources(config)
    try:
        champion_state = torch.load(
            Path(args.champion) / "network.pt",
            map_location=resources.device,
            weights_only=True,
        )
        result = resources.model.load_state_dict(champion_state, strict=False)
        unexpected = [key for key in result.unexpected_keys if "liveness" not in key]
        if result.missing_keys or unexpected:
            raise SystemExit("champion checkpoint drift")

        macro_model = copy.deepcopy(resources.model)
        macro_model.load_state_dict(
            torch.load(Path(args.macro), map_location=resources.device, weights_only=True)
        )
        macro_model.eval()
        device = resources.device
        encoder = resources.encoder

        rest_rows: list[dict[str, Any]] = []
        shop_rows: list[dict[str, Any]] = []
        for journal_path in args.journal:
            with Path(journal_path).open(encoding="utf-8") as handle:
                for line in handle:
                    if len(rest_rows) >= args.limit and len(shop_rows) >= args.limit:
                        break
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    observation = record.get("observation")
                    actions = record.get("legal_actions") or record.get("actions")
                    if not isinstance(observation, dict) or not isinstance(actions, list):
                        continue
                    if len(actions) < 2:
                        continue
                    decision = forward_decision(observation, actions)
                    if decision is None:
                        continue
                    if decision.surface == "rest" and len(rest_rows) < args.limit:
                        gaps = {}
                        for label, fraction in (("hp10", 0.10), ("hp90", 0.90)):
                            mutated = _set_hp(observation, fraction)
                            q_values = _q_for(macro_model, encoder, device, mutated, actions)
                            if q_values is None:
                                continue
                            gap = _branch_gap(q_values, decision, default_branch="rest")
                            if gap is not None:
                                gaps[label] = gap
                        if len(gaps) == 2:
                            rest_rows.append(
                                {
                                    "floor": (observation.get("run") or {}).get("floor"),
                                    "smith_minus_rest_hp10": gaps["hp10"],
                                    "smith_minus_rest_hp90": gaps["hp90"],
                                    "delta": gaps["hp90"] - gaps["hp10"],
                                }
                            )
                    elif decision.surface == "shop" and len(shop_rows) < args.limit:
                        gaps = {}
                        for label, gold in (("broke", 0), ("rich", 500)):
                            mutated = _set_gold(observation, gold)
                            q_values = _q_for(macro_model, encoder, device, mutated, actions)
                            if q_values is None:
                                continue
                            gap = _branch_gap(q_values, decision, default_branch="leave")
                            if gap is not None:
                                gaps[label] = gap
                        if len(gaps) == 2:
                            shop_rows.append(
                                {
                                    "floor": (observation.get("run") or {}).get("floor"),
                                    "buy_minus_leave_broke": gaps["broke"],
                                    "buy_minus_leave_rich": gaps["rich"],
                                    "delta": gaps["rich"] - gaps["broke"],
                                }
                            )

        def _stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
            if not rows:
                return {"count": 0}
            deltas = sorted(row["delta"] for row in rows)
            moved = sum(1 for delta in deltas if abs(delta) > 1e-6)
            return {
                "count": len(rows),
                "delta_p50": deltas[len(deltas) // 2],
                "delta_mean": sum(deltas) / len(deltas),
                "moved_fraction": moved / len(deltas),
            }

        rest_stats = _stats(rest_rows)
        shop_stats = _stats(shop_rows)
        report: dict[str, Any] = {
            "event": "stage3_state_variation_report",
            "rest": {"stats": rest_stats, "rows": rest_rows},
            "shop": {"stats": shop_stats, "rows": shop_rows},
        }
        Path(args.report_out).write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "rest": rest_stats,
                    "shop": shop_stats,
                    "report": args.report_out,
                },
                indent=2,
            )
        )
        return 0
    finally:
        resources.close()


if __name__ == "__main__":
    raise SystemExit(main())
