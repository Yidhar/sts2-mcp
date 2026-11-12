"""16-seed paired evaluation for baseline vs final macro-credit models."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sts2_rl.artifacts import resolve_artifact_path
from sts2_rl.macro_training_campaign import paired_comparison
from sts2_rl.training.seeding import held_out_evaluation_seeds

DEFAULT_16_SEEDS = held_out_evaluation_seeds(6700500, 16)


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_single_seed(
    phase: str,
    seed: int,
    *,
    model_path: Path,
    config_path: Path,
    output_dir: Path,
    sim_exe: Path,
    device: str,
    timeout_seconds: float,
    max_native_steps: int,
    reuse_dir: Path | None,
) -> dict[str, Any]:
    target_dir = output_dir / "evaluations" / phase / f"seed-{seed}"
    if reuse_dir:
        prior_dir = reuse_dir / phase / f"seed-{seed}"
        prior_summary = prior_dir / "summary.json"
        if prior_summary.is_file():
            if target_dir.exists():
                shutil.rmtree(target_dir)
            shutil.copytree(prior_dir, target_dir)
            row = json.loads(prior_summary.read_text(encoding="utf-8"))
            print(
                f"[{phase.upper()}] Seed {seed}: reused existing result -> {row.get('run_result')} "
                f"(Act {row.get('ending_act')}/Fl {row.get('ending_floor')}, revivals: {row.get('revivals_used')})",
                flush=True,
            )
            return row

    if target_dir.exists():
        shutil.rmtree(target_dir)

    cmd = [
        sys.executable, "-m", "sts2_rl.macro_episode_evaluation",
        "--config", str(config_path),
        "--model-path", str(model_path),
        "--output-dir", str(target_dir),
        "--sim-exe", str(sim_exe),
        "--seed", str(seed),
        "--device", device,
        "--timeout-seconds", str(timeout_seconds),
        "--max-native-steps", str(max_native_steps),
    ]
    try:
        import torch
        payload = torch.load(model_path, map_location="cpu", weights_only=False)
        if isinstance(payload, dict) and payload.get("format") == "sts2-hybrid-publication-v3":
            cmd.append("--initialize-from-iql")
    except Exception:
        pass
    started = time.monotonic()
    print(f"[{phase.upper()}] Seed {seed}: launching evaluation...", flush=True)
    res = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.monotonic() - started
    summary_path = target_dir / "summary.json"
    if summary_path.is_file():
        row = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        row = {
            "seed": seed,
            "status": "failed",
            "error": res.stderr[:500] if res.stderr else "child exited without summary",
            "elapsed_seconds": elapsed,
        }
    print(
        f"[{phase.upper()}] Seed {seed}: finished in {elapsed:.1f}s -> {row.get('run_result', row.get('status'))} "
        f"(Act {row.get('ending_act')}/Fl {row.get('ending_floor')}, revivals: {row.get('revivals_used')})",
        flush=True,
    )
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-model", type=Path, required=True)
    parser.add_argument("--final-model", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sim-exe", type=Path, required=True)
    parser.add_argument("--reuse-dir", type=Path, default=None)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=float, default=1200.0)
    parser.add_argument("--max-native-steps", type=int, default=8000)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_16_SEEDS)
    args = parser.parse_args(argv)

    output = resolve_artifact_path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    reuse = resolve_artifact_path(args.reuse_dir) if args.reuse_dir else None

    print(f"Starting 16-seed paired evaluation across {len(args.seeds)} seeds with {args.workers} workers...")
    print(f"Baseline: {args.baseline_model}")
    print(f"Final:    {args.final_model}")
    print(f"Output:   {output}")
    if reuse:
        print(f"Reusing from: {reuse}")

    phases = [
        ("baseline", args.baseline_model),
        ("final", args.final_model),
    ]

    all_results: dict[str, list[dict[str, Any]]] = {"baseline": [], "final": []}

    for phase, model_path in phases:
        print(f"\n=== Running Phase: {phase.upper()} ===", flush=True)
        futures = []
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for seed in args.seeds:
                fut = executor.submit(
                    run_single_seed,
                    phase,
                    seed,
                    model_path=model_path,
                    config_path=args.config,
                    output_dir=output,
                    sim_exe=args.sim_exe,
                    device=args.device,
                    timeout_seconds=args.timeout_seconds,
                    max_native_steps=args.max_native_steps,
                    reuse_dir=reuse,
                )
                futures.append(fut)
            for fut in as_completed(futures):
                all_results[phase].append(fut.result())

        all_results[phase].sort(key=lambda r: r["seed"])

    comparison = paired_comparison(all_results["baseline"], all_results["final"])
    _json(output / "comparison.json", comparison)

    # Generate progress.md table
    md_lines = [
        "# 16-Seed 宏观策略成对评测报告",
        "",
        f"- 评测时间: {datetime.now(UTC).isoformat()}",
        f"- 基线模型: `{args.baseline_model.name}`",
        f"- 目标模型: `{args.final_model.name}`",
        f"- 基线胜率: {comparison['before']['victories']}/{comparison['before']['completed']} ({comparison['before']['victories']/max(1, comparison['before']['completed'])*100:.1f}%)",
        f"- 目标胜率: {comparison['after']['victories']}/{comparison['after']['completed']} ({comparison['after']['victories']/max(1, comparison['after']['completed'])*100:.1f}%)",
        "",
        "| 种子 | 基线结果 (幕/楼层) | 基线复活 | 基线失血 | 基线回报 | 目标结果 (幕/楼层) | 目标复活 | 目标失血 | 目标回报 | 楼层变化 | 复活变化 | 战损变化 | 回报变化 |",
        "|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for pair in comparison["pairs"]:
        s = pair["seed"]
        b = pair["before"]
        a = pair["after"]
        d = pair["after_minus_before"]
        b_res = f"{b.get('run_result', '—')} ({b.get('ending_act', '?')}/{b.get('ending_floor', '?')})"
        b_rev = b.get("revivals_used", "—")
        b_hp = f"{b.get('player_hp_lost', 0):.0f}" if b.get("player_hp_lost") is not None else "—"
        b_ret = f"{b.get('observed_discounted_return', 0):.4f}" if b.get("observed_discounted_return") is not None else "—"

        a_res = f"{a.get('run_result', '—')} ({a.get('ending_act', '?')}/{a.get('ending_floor', '?')})"
        a_rev = a.get("revivals_used", "—")
        a_hp = f"{a.get('player_hp_lost', 0):.0f}" if a.get("player_hp_lost") is not None else "—"
        a_ret = f"{a.get('observed_discounted_return', 0):.4f}" if a.get("observed_discounted_return") is not None else "—"

        d_fl = f"{d.get('ending_floor', 0):+d}" if "ending_floor" in d else "—"
        d_rev = f"{d.get('revivals_used', 0):+d}" if "revivals_used" in d else "—"
        d_hp = f"{d.get('player_hp_lost', 0):+.0f}" if "player_hp_lost" in d else "—"
        d_ret = f"{d.get('observed_discounted_return', 0):+.4f}" if "observed_discounted_return" in d else "—"

        md_lines.append(
            f"| {s} | {b_res} | {b_rev} | {b_hp} | {b_ret} | {a_res} | {a_rev} | {a_hp} | {a_ret} | {d_fl} | {d_rev} | {d_hp} | {d_ret} |"
        )

    md_text = "\n".join(md_lines) + "\n"
    (output / "progress.md").write_text(md_text, encoding="utf-8")
    print("\n" + md_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
