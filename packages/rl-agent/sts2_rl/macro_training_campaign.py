"""One bounded, supervised baseline -> train -> paired full-game evaluation.

This foreground process owns its children; it is not a recurring service.
Only complete cohort checkpoints are used. No model is promoted automatically.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from sts2_rl.artifacts import resolve_artifact_path
from sts2_rl.macro_root_diagnostic import validate_seed
from sts2_rl.train_macro_credit import _json, load_macro_training_config, source_manifest

DEFAULT_EVALUATION_SEEDS = (13_401_001, 13_401_003, 13_401_005, 13_401_007)


def _utc() -> str:
    return datetime.now(UTC).isoformat()


def _sha(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def validate_budget(hours: float, episode_seconds: float, seeds: Sequence[int], max_updates: int) -> None:
    for value in (hours, episode_seconds):
        if not math.isfinite(value) or value <= 0:
            raise ValueError("time budgets must be positive and finite")
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("evaluation seeds must be nonempty and unique")
    for seed in seeds:
        validate_seed(seed)
        if seed % 2 != 1:
            raise ValueError("evaluation requires odd seeds disjoint from even training seeds")
    if isinstance(max_updates, bool) or not isinstance(max_updates, int) or max_updates < 1:
        raise ValueError("max_updates must be a positive integer")
    if hours * 3600 < 2 * len(seeds) * (episode_seconds + 90) + 600:
        raise ValueError("overall budget must fit both evaluation sets plus at least 10 minutes of training")


def paired_comparison(before: Sequence[dict[str, Any]], after: Sequence[dict[str, Any]]) -> dict[str, Any]:
    def aggregate(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
        completed = [row for row in rows if row.get("status") == "complete"]
        return {"attempted": len(rows), "completed": len(completed),
                "victories": sum(row.get("run_result") == "victory" for row in completed),
                "defeats": sum(row.get("run_result") == "defeat" for row in completed),
                "censored": sum(row.get("status") == "censored" for row in rows),
                "failed": sum(row.get("status") == "failed" for row in rows)}

    parents = {row["seed"]: row for row in before}
    pairs = []
    for row in after:
        parent = parents.get(row["seed"])
        if parent is None:
            continue
        complete = parent.get("status") == row.get("status") == "complete"
        deltas = {}
        if complete:
            for name in ("ending_floor", "revivals_used", "player_hp_lost", "observed_discounted_return"):
                a, b = parent.get(name), row.get(name)
                if isinstance(a, int | float) and isinstance(b, int | float):
                    deltas[name] = b - a
        pairs.append({"seed": row["seed"], "complete_pair": complete, "before": parent,
                      "after": row, "after_minus_before": deltas})
    return {"before": aggregate(before), "after": aggregate(after), "pairs": pairs,
            "complete_pairs": sum(pair["complete_pair"] for pair in pairs),
            "automatic_promotion": False, "quality_certified": False,
            "note": "Fixed-seed paired descriptive evidence only, not a generalization or significance claim. "
                    "Censored returns/floors are partial and excluded from complete-pair deltas."}


def supervised_command(command: list[str], *, log_dir: Path, timeout: float,
                       on_tick: Callable[[], None] = lambda: None) -> dict[str, Any]:
    """Allow Python to close the simulator on SIGINT before forced termination."""
    log_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    interrupted = False
    with (log_dir / "stdout.log").open("x", encoding="utf-8") as out, \
            (log_dir / "stderr.log").open("x", encoding="utf-8") as err:
        child = subprocess.Popen(command, stdout=out, stderr=err, start_new_session=True)
        _json(log_dir / "process.json", {"pid": child.pid, "command": command, "started_utc": _utc(),
                                         "timeout_seconds": timeout})
        try:
            while child.poll() is None:
                on_tick()
                remaining = timeout - (time.monotonic() - started)
                if remaining <= 0:
                    interrupted = True
                    break
                try:
                    child.wait(timeout=min(10.0, remaining))
                except subprocess.TimeoutExpired:
                    pass
        finally:
            if child.poll() is None:
                child.send_signal(signal.SIGINT)
                try:
                    child.wait(timeout=35)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait(timeout=5)
        result = {"exit_code": child.returncode, "budget_interrupted": interrupted,
                  "elapsed_seconds": time.monotonic() - started, "ended_utc": _utc()}
        _json(log_dir / "exit.json", result)
        return result


def _snapshot(source: Path, destination: Path) -> dict[str, Any]:
    if destination.exists():
        raise FileExistsError(destination)
    digest = _sha(source)
    with source.open("rb") as src, destination.open("xb") as dst:
        shutil.copyfileobj(src, dst)
    if _sha(destination) != digest or _sha(source) != digest:
        raise ValueError("publication changed while freezing evidence")
    payload = torch.load(destination, map_location="cpu", weights_only=True)
    if payload.get("format") != "sts2-hybrid-macro-credit-publication-v1":
        raise ValueError("campaign requires a macro-credit publication")
    version = payload.get("policy_version")
    if isinstance(version, bool) or not isinstance(version, int) or version < 0:
        raise ValueError("invalid published policy version")
    return {"path": str(destination), "sha256": digest, "policy_version": version}


def run_campaign(args: argparse.Namespace) -> dict[str, Any]:
    validate_budget(args.hours, args.episode_seconds, args.evaluation_seeds, args.max_updates)
    if os.name != "posix":
        raise ValueError("campaign process-group supervision requires Linux/WSL")
    config = load_macro_training_config(args.config)
    output, run = resolve_artifact_path(args.output_dir), resolve_artifact_path(args.run_dir)
    parent = args.parent.resolve(strict=True)
    if output.exists() or run.exists():
        raise FileExistsError("campaign and training lineage must both be new")
    if output == run or output.is_relative_to(run) or run.is_relative_to(output):
        raise ValueError("campaign evidence and mutable training run must be disjoint")
    if parent.is_relative_to(output) or parent.is_relative_to(run):
        raise ValueError("parent publication must be outside new output directories")
    output.mkdir(parents=True, exist_ok=False)
    (output / "models").mkdir()
    started = time.monotonic()
    deadline = started + args.hours * 3600
    eval_allowance = len(args.evaluation_seeds) * (args.episode_seconds + 90)
    training_deadline = deadline - eval_allowance - 60
    frozen_config = output / "course.toml"
    shutil.copyfile(args.config, frozen_config)
    baseline = _snapshot(parent, output / "models/baseline-parent.pt")
    package = Path(__file__).resolve().parent
    evaluator_sources = {name: _sha(package / name) for name in (
        "macro_training_campaign.py", "macro_episode_evaluation.py")}
    manifest = source_manifest()
    plan = {"schema": "sts2-macro-training-campaign-v1", "started_utc": _utc(), "pid": os.getpid(),
            "overall_hours": args.hours, "overall_deadline_utc": datetime.fromtimestamp(
                time.time() + args.hours * 3600, UTC).isoformat(),
            "lineage_id": args.lineage_id, "run_dir": str(run), "parent_original": str(parent),
            "parent_sha256": baseline["sha256"], "baseline": baseline,
            "initialization": "model-only; reset optimizer/RNG/update; not exact continuation of parent checkpoint",
            "evaluation_seeds": list(args.evaluation_seeds), "training_seed_partition": "even",
            "training_seed_index_offset": config.course.seed_index_offset,
            "episode_seconds": args.episode_seconds, "max_native_steps": args.max_native_steps,
            "revival_budget": config.training.curriculum.revival_budget, "device": args.device,
            "max_updates": args.max_updates, "simulator_path": str(args.sim_exe),
            "course_sha256": _sha(frozen_config), "source_manifest": manifest,
            "evaluation_sources": evaluator_sources, "automatic_promotion": False,
            "stop_policy": "stop on first training/protocol error; budget interrupt discards partial cohort; "
                           "evaluate only last committed publication; no seed skipping"}
    _json(output / "plan.json", plan)
    status: dict[str, Any] = {"state": "baseline_evaluation", "started_utc": plan["started_utc"],
                              "committed_new_updates": 0, "baseline": [], "final": [],
                              "automatic_promotion": False, "quality_certified": False}
    active_progress: Path | None = None

    def persist() -> None:
        status.update(updated_utc=_utc(), elapsed_seconds=time.monotonic() - started,
                      remaining_seconds=max(0, deadline - time.monotonic()),
                      active_episode=_read(active_progress) if active_progress else {},
                      training_status=_read(run / "status.json"))
        _json(output / "status.json", status)
        lines = ["# 宏观策略长训与整局评估", "", f"- 当前阶段: `{status['state']}`",
                 f"- 新实验已提交更新: {status['committed_new_updates']} (父模型 update {baseline['policy_version']} 不计入)",
                 f"- 已耗时: {status['elapsed_seconds'] / 3600:.2f} 小时; 总预算: {args.hours:g} 小时",
                 f"- 更新于 (UTC): {status['updated_utc']}",
                 f"- 评估为固定奇数种子、自然贪心整局、{plan['revival_budget']} 次复活预算; 不是零复活模式。",
                 "- 超时是未完成; 截止楼层和部分回报不能当成整局成绩; 不自动替换基线。", "",
                 "| 阶段 | 种子 | 状态 | 结果 | 幕/楼层 | 复活次数 | HP | 实际累计折扣回报 |",
                 "|---|---:|---|---|---|---:|---|---:|"]
        for phase in ("baseline", "final"):
            for row in status[phase]:
                lines.append(f"| {phase} | {row['seed']} | {row['status']} | {row.get('run_result', '—')} | "
                             f"{row.get('ending_act')}/{row.get('ending_floor')} | {row.get('revivals_used')} | "
                             f"{row.get('hp')}/{row.get('max_hp')} | {row.get('observed_discounted_return')} |")
        if status["active_episode"]:
            lines.extend(["", "## 当前对局 (尚未作为整局成绩)", "```json",
                          json.dumps(status["active_episode"], ensure_ascii=False, indent=2), "```"])
        if status.get("error"):
            lines.extend(["", "## 中止原因", str(status["error"])])
        (output / "progress.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def assert_sources() -> None:
        if source_manifest()["identity"] != manifest["identity"]:
            raise ValueError("training sources changed during campaign; new lineage required")
        if _sha(frozen_config) != plan["course_sha256"]:
            raise ValueError("frozen course config changed during campaign")
        if any(_sha(package / name) != digest for name, digest in evaluator_sources.items()):
            raise ValueError("evaluation protocol source changed during campaign")

    def evaluate_set(phase: str, model: Path) -> None:
        nonlocal active_progress
        for seed in args.evaluation_seeds:
            assert_sources()
            episode = output / "evaluations" / phase / f"seed-{seed}"
            active_progress = episode / "progress.json"
            status.update(state=f"{phase}_evaluation", active_seed=seed)
            command = [sys.executable, "-m", "sts2_rl.macro_episode_evaluation", "--config", str(frozen_config),
                       "--model-path", str(model), "--output-dir", str(episode), "--sim-exe", str(args.sim_exe),
                       "--seed", str(seed), "--device", args.device, "--timeout-seconds", str(args.episode_seconds),
                       "--max-native-steps", str(args.max_native_steps)]
            child = supervised_command(command, log_dir=output / "invocations" / f"{phase}-{seed}",
                                       timeout=min(args.episode_seconds + 45, max(1, deadline - time.monotonic() - 45)),
                                       on_tick=persist)
            row = _read(episode / "summary.json")
            if not row:
                row = {"seed": seed, "status": "censored" if child["budget_interrupted"] else "failed",
                       "ending": "supervisor_timeout" if child["budget_interrupted"] else "child_exit_without_summary"}
            if child["budget_interrupted"]:
                row.update(status="censored", ending="supervisor_timeout")
            elif child["exit_code"] not in (0, 2):
                row.update(status="failed", ending="unexpected_child_exit")
            row["child"] = child
            status[phase].append(row)
            active_progress = None
            persist()
            if row["status"] == "failed":
                raise RuntimeError(f"{phase} evaluation failed on seed {seed}: {row.get('error', row['ending'])}")

    persist()
    try:
        evaluate_set("baseline", Path(baseline["path"]))
        if not any(row["status"] == "complete" for row in status["baseline"]):
            raise RuntimeError("all baseline games censored; increase/repair evaluation before spending training budget")
        failure = None
        stop_reason = "max_updates"
        latest: dict[str, Any] | None = None
        for attempt in range(args.max_updates):
            remaining_training = training_deadline - time.monotonic()
            if remaining_training < 120:
                stop_reason = "training_budget_boundary"
                break
            assert_sources()
            status.update(state="training", active_seed=None, training_attempt=attempt + 1)
            initialize = ["--initialize-from-macro-publication", str(baseline["path"])] if attempt == 0 else [
                "--resume", str(run / "checkpoint.pt")]
            command = [sys.executable, "-m", "sts2_rl.train_macro_credit", "--config", str(frozen_config),
                       "--output-dir", str(run), "--lineage-id", args.lineage_id, "--device", args.device,
                       "--sim-exe", str(args.sim_exe), "--max-updates", "1", *initialize]
            child = supervised_command(command, log_dir=output / "invocations" / f"train-{attempt + 1:04d}",
                                       timeout=remaining_training, on_tick=persist)
            if (run / "publication.pt").exists():
                latest = _snapshot(run / "publication.pt", output / "models" / f"after-attempt-{attempt + 1:04d}.pt")
                status["committed_new_updates"] = latest["policy_version"]
                status["latest_publication"] = latest
            if child["budget_interrupted"]:
                stop_reason = "training_budget_interrupt_partial_cohort_discarded"
                break
            if child["exit_code"] != 0:
                failure = _read(run / "status.json") or child
                stop_reason = "training_error"
                break
            if latest is None or latest["policy_version"] != attempt + 1:
                raise RuntimeError("training exited without exactly one newly published cohort update")
            persist()
        status["training_stop_reason"] = stop_reason
        status["training_error"] = failure
        if latest is None:
            raise RuntimeError("no valid new-lineage publication is available to evaluate")
        evaluate_set("final", Path(latest["path"]))
        comparison = paired_comparison(status["baseline"], status["final"])
        _json(output / "comparison.json", comparison)
        state = "failed_training_evaluated_last_commit" if failure else (
            "completed" if status["committed_new_updates"] > 0 else "no_new_updates_evaluated")
        status.update(state=state,
                      comparison=comparison, ended_utc=_utc())
        if failure:
            status["error"] = failure
        if _sha(parent) != plan["parent_sha256"]:
            raise RuntimeError("original parent publication changed during campaign")
    except BaseException as error:
        status.update(state="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                      error=f"{type(error).__name__}: {error}", ended_utc=_utc())
        _json(output / "comparison.json", paired_comparison(status["baseline"], status["final"]))
    finally:
        active_progress = None
        persist()
    return status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "parent", "output-dir", "run-dir", "sim-exe"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--lineage-id", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--hours", type=float, default=8.0)
    parser.add_argument("--episode-seconds", type=float, default=1200.0)
    parser.add_argument("--max-native-steps", type=int, default=8000)
    parser.add_argument("--max-updates", type=int, default=64)
    parser.add_argument("--evaluation-seeds", type=int, nargs="+", default=DEFAULT_EVALUATION_SEEDS)
    args = parser.parse_args(argv)
    result = run_campaign(args)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0 if result["state"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

