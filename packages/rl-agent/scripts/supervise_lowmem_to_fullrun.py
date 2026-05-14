#!/usr/bin/env python3
"""Supervise the low-VRAM combat sandbox run and optionally hand off to full-run.

This script intentionally does not touch training semantics.  It is an operator
utility for the long-running Pass-Large pipeline:

1. repeatedly run ``monitor_combat_sandbox_gate.py`` on the current lowmem run;
2. keep the existing sandbox training alive while the verdict is WAIT_* or FAIL;
3. when the sandbox gate returns PASS, optionally stop the sandbox process and
   launch the random-seed full-run Act1 launcher.

The script is kept separate from ``muzero/train.py`` to avoid adding more
operator plumbing to the already-large training file.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence


VERDICT_RE = re.compile(r"^verdict:\s*([A-Z_]+)\s*$", re.MULTILINE)
METRIC_LAST_RE_TEMPLATE = r"^{tag}\s+last=([-+0-9.eE]+)\b"
METRIC_STEP_RE_TEMPLATE = r"^{tag}\s+.*?\bstep=([0-9]+)\b"
CHECKPOINT_STEP_RE = re.compile(r"muzero_step_([0-9]+)$")


@dataclass(frozen=True)
class GateSummary:
    verdict: str
    buffer_size: float | None
    raw_output: str
    returncode: int


@dataclass(frozen=True)
class CheckpointFreshness:
    path: Path | None
    checkpoint_step: int | None
    current_step: int | None
    age_sec: float | None
    step_lag: int | None
    fresh: bool
    reason: str


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_verdict(text: str) -> str | None:
    match = VERDICT_RE.search(text)
    return match.group(1) if match else None


def parse_metric_last(text: str, tag: str) -> float | None:
    pattern = re.compile(METRIC_LAST_RE_TEMPLATE.format(tag=re.escape(tag)), re.MULTILINE)
    match = pattern.search(text)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def parse_metric_step(text: str, tag: str) -> int | None:
    pattern = re.compile(METRIC_STEP_RE_TEMPLATE.format(tag=re.escape(tag)), re.MULTILINE)
    match = pattern.search(text)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def parse_gate_summary(raw_output: str, returncode: int) -> GateSummary:
    verdict = parse_verdict(raw_output)
    if verdict is None:
        verdict = "MONITOR_ERROR" if returncode else "UNKNOWN"
    return GateSummary(
        verdict=verdict,
        buffer_size=parse_metric_last(raw_output, "buffer/size"),
        raw_output=raw_output,
        returncode=returncode,
    )


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def read_pid(path: Path) -> int | None:
    try:
        text = read_text(path)
    except OSError:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def process_alive(pid: int | None) -> bool:
    if pid is None or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def terminate_process(pid: int, *, grace_sec: float, log) -> bool:
    if not process_alive(pid):
        log(f"process_not_alive pid={pid}")
        return True

    log(f"terminate pid={pid} grace_sec={grace_sec}")
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return True

    deadline = time.monotonic() + grace_sec
    while time.monotonic() < deadline:
        if not process_alive(pid):
            log(f"terminated pid={pid}")
            return True
        time.sleep(1.0)

    if process_alive(pid):
        log(f"kill pid={pid}")
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            return True
    return not process_alive(pid)


def checkpoint_step(path: Path) -> int | None:
    match = CHECKPOINT_STEP_RE.search(path.name)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def latest_checkpoint(checkpoint_dir: Path) -> tuple[Path | None, int | None]:
    if not checkpoint_dir.is_dir():
        return None, None

    best_path: Path | None = None
    best_step: int | None = None
    for path in checkpoint_dir.iterdir():
        if not path.is_dir():
            continue
        step = checkpoint_step(path)
        if step is None:
            continue
        if best_step is None or step > best_step:
            best_path = path
            best_step = step
    return best_path, best_step


def checkpoint_freshness(
    *,
    checkpoint_dir: Path,
    gate_summary: GateSummary,
    max_age_sec: float,
    max_step_lag: int,
    now: float | None = None,
) -> CheckpointFreshness:
    path, ckpt_step = latest_checkpoint(checkpoint_dir)
    current_step = parse_metric_step(gate_summary.raw_output, "buffer/size")
    if path is None:
        return CheckpointFreshness(
            path=None,
            checkpoint_step=None,
            current_step=current_step,
            age_sec=None,
            step_lag=None,
            fresh=False,
            reason="no_checkpoint",
        )

    timestamp = now if now is not None else time.time()
    try:
        age_sec = max(0.0, timestamp - path.stat().st_mtime)
    except OSError:
        age_sec = None

    step_lag: int | None = None
    if current_step is not None and ckpt_step is not None:
        step_lag = max(0, current_step - ckpt_step)

    stale_reasons: list[str] = []
    if age_sec is not None and age_sec > max_age_sec:
        stale_reasons.append(f"age_sec>{max_age_sec:g}")
    if step_lag is not None and step_lag > max_step_lag:
        stale_reasons.append(f"step_lag>{max_step_lag}")

    return CheckpointFreshness(
        path=path,
        checkpoint_step=ckpt_step,
        current_step=current_step,
        age_sec=age_sec,
        step_lag=step_lag,
        fresh=not stale_reasons,
        reason="fresh" if not stale_reasons else ",".join(stale_reasons),
    )


def run_gate(
    *,
    repo_dir: Path,
    python_exe: Path,
    run_dir: Path,
    tail: int,
    min_buffer: int,
    max_reserved_gb: float,
    max_peak_allocated_gb: float,
    max_empty_cache_rate: float,
    diagnostics_window: int,
) -> GateSummary:
    cmd = [
        str(python_exe),
        "scripts/monitor_combat_sandbox_gate.py",
        "--run-dir",
        str(run_dir),
        "--tail",
        str(tail),
        "--min-buffer",
        str(min_buffer),
        "--max-reserved-gb",
        str(max_reserved_gb),
        "--max-peak-allocated-gb",
        str(max_peak_allocated_gb),
        "--max-empty-cache-rate",
        str(max_empty_cache_rate),
        "--diagnostics-window",
        str(diagnostics_window),
    ]
    proc = subprocess.run(
        cmd,
        cwd=repo_dir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return parse_gate_summary(proc.stdout, proc.returncode)


def append_line(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text)
        if not text.endswith("\n"):
            handle.write("\n")


def launch_fullrun(
    *,
    repo_dir: Path,
    launcher: Path,
    env: Mapping[str, str] | None,
    log,
) -> int:
    cmd = ["bash", str(launcher)]
    log(f"launch_fullrun cmd={' '.join(cmd)} cwd={repo_dir}")
    proc = subprocess.run(cmd, cwd=repo_dir, env=dict(env or os.environ), text=True, check=False)
    log(f"launch_fullrun_exit returncode={proc.returncode}")
    return proc.returncode


def write_diagnostics_report(
    *,
    repo_dir: Path,
    python_exe: Path,
    run_dir: Path,
    recent_window: int,
    top_n: int,
    log,
) -> int:
    output = run_dir / "combat_sandbox_diagnostics_report.md"
    cmd = [
        str(python_exe),
        "scripts/report_combat_sandbox_diagnostics.py",
        "--run-dir",
        str(run_dir),
        "--recent-window",
        str(recent_window),
        "--top-n",
        str(top_n),
        "--output",
        str(output),
    ]
    proc = subprocess.run(
        cmd,
        cwd=repo_dir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if proc.stdout.strip():
        log("diagnostics_report_output " + proc.stdout.strip().replace("\n", " | "))
    log(f"diagnostics_report_exit returncode={proc.returncode} output={output}")
    return proc.returncode


def resolve_repo_dir(arg: str | None) -> Path:
    if arg:
        return Path(arg).expanduser().resolve()
    return Path(__file__).resolve().parents[1]


def absolutize_without_following_symlink(base: Path, value: str) -> Path:
    """Return an absolute path while preserving a final symlink component.

    ``Path.resolve()`` follows symlinks.  The WSL venv used by this project has
    ``.venv-wsl-rocm/bin/python -> python3``; resolving it can accidentally turn
    the interpreter into the system Python, losing TensorBoard/PyTorch deps.
    """

    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return Path(os.path.abspath(os.fspath(path)))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", default=None, help="rl-agent repo dir; default inferred from this script")
    parser.add_argument("--latest-pointer", default="logs_muzero/latest_lowmem_run_id.txt")
    parser.add_argument("--python-exe", default="./.venv-wsl-rocm/bin/python")
    parser.add_argument(
        "--fullrun-launcher",
        default="./launch_muzero_pass_large_fullrun_random_act1_20260513.sh",
    )
    parser.add_argument("--interval-sec", type=float, default=900.0)
    parser.add_argument("--max-checks", type=int, default=0, help="0 means run forever")
    parser.add_argument("--once", action="store_true", help="run one check and exit")
    parser.add_argument("--auto-launch-fullrun", action="store_true")
    parser.add_argument("--stop-sandbox-before-fullrun", action="store_true")
    parser.add_argument("--terminate-grace-sec", type=float, default=60.0)
    parser.add_argument(
        "--fail-action",
        choices=("keep-running", "exit"),
        default="keep-running",
        help="What to do if the gate verdict is FAIL.",
    )
    parser.add_argument("--tail", type=int, default=50)
    parser.add_argument("--min-buffer", type=int, default=10000)
    parser.add_argument("--max-reserved-gb", type=float, default=22.0)
    parser.add_argument("--max-peak-allocated-gb", type=float, default=20.0)
    parser.add_argument("--max-empty-cache-rate", type=float, default=0.2)
    parser.add_argument("--diagnostics-window", type=int, default=2000)
    parser.add_argument(
        "--write-diagnostics-report",
        action="store_true",
        help="Refresh combat_sandbox_diagnostics_report.md after each gate check.",
    )
    parser.add_argument("--diagnostics-report-window", type=int, default=2000)
    parser.add_argument("--diagnostics-report-top-n", type=int, default=12)
    parser.add_argument(
        "--require-fresh-checkpoint-before-fullrun",
        action="store_true",
        help=(
            "On sandbox PASS, keep sandbox training alive until a recent checkpoint "
            "exists.  This prevents losing the newest sandbox learning when the "
            "full-run launcher resumes from checkpoint."
        ),
    )
    parser.add_argument(
        "--max-checkpoint-age-sec",
        type=float,
        default=1800.0,
        help="Fresh-checkpoint gate: latest checkpoint mtime must be at most this old.",
    )
    parser.add_argument(
        "--max-checkpoint-step-lag",
        type=int,
        default=3000,
        help="Fresh-checkpoint gate: latest checkpoint step may lag current TB step by at most this many steps.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    repo_dir = resolve_repo_dir(args.repo_dir)
    latest_pointer = (repo_dir / args.latest_pointer).resolve()
    python_exe = absolutize_without_following_symlink(repo_dir, args.python_exe)
    fullrun_launcher = absolutize_without_following_symlink(repo_dir, args.fullrun_launcher)

    check_idx = 0
    while True:
        check_idx += 1
        try:
            run_id = read_text(latest_pointer)
        except OSError as exc:
            print(f"[{utc_now_iso()}] latest_pointer_error path={latest_pointer} error={exc}", flush=True)
            return 2

        run_dir = repo_dir / "logs_muzero" / run_id
        train_pid = read_pid(run_dir / "train.pid")
        log_path = run_dir / "lowmem_to_fullrun_supervisor.log"
        jsonl_path = run_dir / "lowmem_to_fullrun_supervisor.jsonl"

        def log(msg: str) -> None:
            line = f"[{utc_now_iso()}] {msg}"
            print(line, flush=True)
            append_line(log_path, line)

        alive = process_alive(train_pid)
        log(f"check={check_idx} run_id={run_id} train_pid={train_pid} alive={alive}")
        if not alive:
            append_line(
                jsonl_path,
                json.dumps(
                    {
                        "time": utc_now_iso(),
                        "check": check_idx,
                        "run_id": run_id,
                        "event": "train_not_alive",
                        "train_pid": train_pid,
                    },
                    ensure_ascii=False,
                ),
            )
            return 3

        summary = run_gate(
            repo_dir=repo_dir,
            python_exe=python_exe,
            run_dir=run_dir,
            tail=args.tail,
            min_buffer=args.min_buffer,
            max_reserved_gb=args.max_reserved_gb,
            max_peak_allocated_gb=args.max_peak_allocated_gb,
            max_empty_cache_rate=args.max_empty_cache_rate,
            diagnostics_window=args.diagnostics_window,
        )
        append_line(
            jsonl_path,
            json.dumps(
                {
                    "time": utc_now_iso(),
                    "check": check_idx,
                    "run_id": run_id,
                    "train_pid": train_pid,
                    "verdict": summary.verdict,
                    "buffer_size": summary.buffer_size,
                    "monitor_returncode": summary.returncode,
                },
                ensure_ascii=False,
            ),
        )
        append_line(log_path, "\n--- gate output begin ---\n" + summary.raw_output + "\n--- gate output end ---")
        log(f"gate verdict={summary.verdict} buffer={summary.buffer_size} returncode={summary.returncode}")

        if args.write_diagnostics_report:
            write_diagnostics_report(
                repo_dir=repo_dir,
                python_exe=python_exe,
                run_dir=run_dir,
                recent_window=args.diagnostics_report_window,
                top_n=args.diagnostics_report_top_n,
                log=log,
            )

        if summary.verdict == "PASS":
            if not args.auto_launch_fullrun:
                log("sandbox_gate_pass auto_launch_fullrun=false")
                return 0

            if args.require_fresh_checkpoint_before_fullrun:
                ckpt_dir = repo_dir / "checkpoints_muzero" / run_id
                freshness = checkpoint_freshness(
                    checkpoint_dir=ckpt_dir,
                    gate_summary=summary,
                    max_age_sec=args.max_checkpoint_age_sec,
                    max_step_lag=args.max_checkpoint_step_lag,
                )
                append_line(
                    jsonl_path,
                    json.dumps(
                        {
                            "time": utc_now_iso(),
                            "check": check_idx,
                            "run_id": run_id,
                            "event": "checkpoint_freshness",
                            "checkpoint_path": str(freshness.path) if freshness.path else None,
                            "checkpoint_step": freshness.checkpoint_step,
                            "current_step": freshness.current_step,
                            "age_sec": freshness.age_sec,
                            "step_lag": freshness.step_lag,
                            "fresh": freshness.fresh,
                            "reason": freshness.reason,
                        },
                        ensure_ascii=False,
                    ),
                )
                log(
                    "checkpoint_freshness "
                    f"fresh={freshness.fresh} reason={freshness.reason} "
                    f"ckpt_step={freshness.checkpoint_step} current_step={freshness.current_step} "
                    f"age_sec={freshness.age_sec} step_lag={freshness.step_lag} "
                    f"path={freshness.path}"
                )
                if not freshness.fresh:
                    log("sandbox_gate_pass checkpoint_not_fresh keep_sandbox_training=true")
                    if args.once or (args.max_checks and check_idx >= args.max_checks):
                        return 6
                    time.sleep(max(args.interval_sec, 1.0))
                    continue

            if args.stop_sandbox_before_fullrun and train_pid is not None:
                if not terminate_process(train_pid, grace_sec=args.terminate_grace_sec, log=log):
                    log(f"failed_to_stop_sandbox pid={train_pid}")
                    return 4

            rc = launch_fullrun(repo_dir=repo_dir, launcher=fullrun_launcher, env=os.environ, log=log)
            return rc

        if summary.verdict == "FAIL" and args.fail_action == "exit":
            log("sandbox_gate_fail fail_action=exit")
            return 5

        if args.once or (args.max_checks and check_idx >= args.max_checks):
            return 0

        time.sleep(max(args.interval_sec, 1.0))


if __name__ == "__main__":
    raise SystemExit(main())
