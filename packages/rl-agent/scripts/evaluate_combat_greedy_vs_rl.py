#!/usr/bin/env python3
"""Run the combat-only frozen-RL versus one-step-greedy paired audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sts2_rl.artifacts import resolve_artifact_path, resolve_external_input_path
from sts2_rl.combat_greedy_evaluation import run_paired_combat_evaluation


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a frozen recurrent policy against exact one-step simulator "
            "branches on identical held-out combat roots."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--comparison-checkpoint",
        type=Path,
        action="append",
        default=[],
        help=(
            "Additional frozen policy to evaluate on every exact same combat root; "
            "repeat for multiple checkpoints. The primary checkpoint still drives "
            "scenario generation."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=16)
    parser.add_argument(
        "--encounter-id",
        action="append",
        default=[],
        help="Combat catalog encounter id; repeat to define the scenario cycle.",
    )
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sim-exe", type=Path)
    parser.add_argument("--revival-budget", type=int, default=0)
    parser.add_argument("--maximum-combat-steps", type=int, default=512)
    parser.add_argument("--stall-window", type=int, default=128)
    parser.add_argument("--repeat-threshold", type=int, default=8)
    parser.add_argument("--candidate-topk", type=int, default=8)
    return parser


def main() -> int:
    args = _parser().parse_args()
    checkpoint = resolve_external_input_path(args.checkpoint)
    comparison_checkpoints = tuple(
        resolve_external_input_path(path)
        for path in args.comparison_checkpoint
    )
    output = resolve_artifact_path(args.output)
    sim_exe = (
        resolve_external_input_path(args.sim_exe)
        if args.sim_exe is not None
        else None
    )
    summary = run_paired_combat_evaluation(
        checkpoint,
        comparison_checkpoints=comparison_checkpoints,
        output_directory=output,
        pairs=args.pairs,
        encounter_ids=args.encounter_id,
        base_seed=args.base_seed,
        device=args.device,
        sim_exe_path=sim_exe,
        revival_budget=args.revival_budget,
        maximum_combat_steps=args.maximum_combat_steps,
        stall_window=args.stall_window,
        repeat_threshold=args.repeat_threshold,
        candidate_topk=args.candidate_topk,
    )
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
