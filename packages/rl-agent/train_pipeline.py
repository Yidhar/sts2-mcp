"""Legacy three-stage training pipeline for the pre-attention STS2 policy lineage.

Stages:
  1. combat sandbox PPO
  2. offline build/route pretraining
  3. full-run PPO

This pipeline is archived because stage 2 still depends on the routed v2 policy
stack. The active online mainline now uses ``train_attention_policy.py``.

Every legacy stage reads and writes the same checkpoint format:
  model.safetensors + metadata.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


STAGE_ORDER = ("sandbox", "offline", "fullrun")
DEFAULT_STAGE1_TRAIN_POOL = ",".join([
    "ENCOUNTER.SLIMES_WEAK",
    "ENCOUNTER.SHRINKER_BEETLE_WEAK",
    "ENCOUNTER.FUZZY_WURM_CRAWLER_WEAK",
    "ENCOUNTER.NIBBITS_WEAK",
])
DEFAULT_STAGE1_HOLDOUT_POOL = ",".join([
    "ENCOUNTER.CORPSE_SLUGS_WEAK",
    "ENCOUNTER.SLUDGE_SPINNER_WEAK",
    "ENCOUNTER.SEAPUNK_WEAK",
    "ENCOUNTER.TOADPOLES_WEAK",
])
DEFAULT_STAGE2_TASKS = ",".join([
    "regular_card_reward",
    "event_card_bundle",
    "ancient_choice",
    "relic_choice_step",
    "potion_choice_step",
    "rest_action",
    "smith_target",
    "remove_card_step",
    "transform_card_step",
    "shop_relic_pick_step",
    "shop_potion_pick_step",
    "shop_remove_binary",
    "shop_remove_target_step",
    "shop_bundle_aux",
])
DEFAULT_STAGE2_REPEAT_FACTORS = ",".join([
    "event_card_bundle:4",
    "ancient_choice:2",
    "smith_target:8",
    "transform_card_step:8",
    "remove_card_step:2",
    "shop_relic_pick_step:2",
    "shop_potion_pick_step:2",
    "shop_remove_binary:4",
    "shop_remove_target_step:8",
    "shop_bundle_aux:2",
])


def build_pipeline_dir(out_dir: str | Path, run_name: str | None) -> Path:
    root = Path(out_dir)
    name = run_name or f"pipeline-{time.strftime('%Y%m%d-%H%M%S')}"
    return root / name


def format_command(cmd: list[str]) -> str:
    return " ".join(f'"{part}"' if " " in part else part for part in cmd)


def run_logged(cmd: list[str], *, cwd: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[pipeline] running: {format_command(cmd)}")
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(_sanitize_console_line(line))
            handle.write(line)
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, cmd)


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def _sanitize_console_line(line: str) -> str:
    encoding = sys.stdout.encoding or "utf-8"
    return line.encode(encoding, errors="replace").decode(encoding, errors="replace")


def add_common_train_v2_args(
    cmd: list[str],
    *,
    device: str,
    session_file: str | None,
    character: str | None,
    learning_rate: float,
    checkpoint_freq: int,
) -> None:
    cmd.extend(["--device", device])
    cmd.extend(["--learning-rate", str(learning_rate)])
    cmd.extend(["--checkpoint-freq", str(checkpoint_freq)])
    if session_file:
        cmd.extend(["--session-file", session_file])
    if character:
        cmd.extend(["--character", character])


def require_sandbox_source(args) -> None:
    if args.start_checkpoint:
        return
    if args.stage1_encounter_pool or args.stage1_encounter_id:
        return
    raise SystemExit(
        "Stage 1 needs --stage1-encounter-pool or --stage1-encounter-id unless --start-checkpoint is provided."
    )


def resolve_stage_checkpoint(stage_root: Path, *, kind: str, preference: str = "final") -> Path:
    if kind == "train_v2":
        return stage_root / "checkpoints" / "final"
    if kind == "offline":
        return stage_root / "offline" / preference
    raise ValueError(f"Unsupported stage kind: {kind}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run sandbox -> offline -> full-run on one checkpoint lineage.")
    parser.add_argument("--dataset-root", required=True, type=str)
    parser.add_argument("--out-dir", default="pipeline_runs", type=str)
    parser.add_argument("--run-name", default=None, type=str)
    parser.add_argument("--start-checkpoint", default=None, type=str,
                        help="Skip stage 1 and begin from an existing online checkpoint directory.")
    parser.add_argument("--stop-after", default="fullrun", choices=STAGE_ORDER)
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument("--session-file", default=None, type=str)
    parser.add_argument("--character", default=None, type=str,
                        help="Default character for stage 1 and stage 3 unless overridden.")
    parser.add_argument("--stage1-character", default=None, type=str)
    parser.add_argument("--stage1-total-timesteps", default=50_000, type=int)
    parser.add_argument("--stage1-learning-rate", default=1e-4, type=float)
    parser.add_argument("--stage1-checkpoint-freq", default=1024, type=int)
    parser.add_argument("--stage1-encounter-id", default=None, type=str)
    parser.add_argument(
        "--stage1-encounter-pool",
        default=DEFAULT_STAGE1_TRAIN_POOL,
        type=str,
        help="Comma-separated sandbox encounter pool. Defaults to starter-deck-friendly early Act 1 weak encounters.",
    )
    parser.add_argument(
        "--stage1-eval-holdout-pool",
        default=DEFAULT_STAGE1_HOLDOUT_POOL,
        type=str,
        help="Comma-separated fixed eval holdout pool. Defaults to unseen early Act 1 weak encounters.",
    )
    parser.add_argument(
        "--stage1-eval-freq",
        default=0,
        type=int,
        help="Fixed sandbox eval frequency. Default 0 because live single-instance sandbox training cannot safely interleave eval resets.",
    )

    parser.add_argument(
        "--stage2-tasks",
        default=DEFAULT_STAGE2_TASKS,
        type=str,
        help=(
            "Comma-separated offline build_v2 tasks. Route tasks are intentionally omitted "
            "because the current dataset only exports chosen_path_only route supervision."
        ),
    )
    parser.add_argument(
        "--stage2-task-repeat-factors",
        default=DEFAULT_STAGE2_REPEAT_FACTORS,
        type=str,
        help="Comma-separated task:factor sampling multipliers forwarded to train_offline_multitask.py.",
    )
    parser.add_argument(
        "--stage2-checkpoint-kind",
        default="best",
        choices=["best", "final"],
        help="Which offline checkpoint to hand off into stage 3.",
    )
    parser.add_argument("--stage2-partition-kind", default="build_family", choices=["build_id", "build_family"])
    parser.add_argument("--stage2-partition-value", default="v0.98_to_v0.99.1", type=str)
    parser.add_argument("--stage2-epochs", default=10, type=int)
    parser.add_argument("--stage2-batch-size", default=64, type=int)
    parser.add_argument("--stage2-learning-rate", default=1e-4, type=float)
    parser.add_argument("--stage2-weight-decay", default=1e-4, type=float)
    parser.add_argument("--stage2-max-train-batches", default=None, type=int)
    parser.add_argument("--stage2-max-eval-batches", default=None, type=int)

    parser.add_argument("--stage3-character", default=None, type=str)
    parser.add_argument("--stage3-total-timesteps", default=200_000, type=int)
    parser.add_argument("--stage3-learning-rate", default=3e-4, type=float)
    parser.add_argument("--stage3-checkpoint-freq", default=2048, type=int)
    args = parser.parse_args()

    require_sandbox_source(args)

    script_dir = Path(__file__).resolve().parent
    pipeline_dir = build_pipeline_dir(args.out_dir, args.run_name)
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    summary_path = pipeline_dir / "pipeline_summary.json"

    summary: dict[str, Any] = {
        "pipeline_dir": str(pipeline_dir),
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stop_after": args.stop_after,
        "start_checkpoint": args.start_checkpoint,
        "stages": {},
    }
    write_summary(summary_path, summary)

    current_checkpoint = Path(args.start_checkpoint).resolve() if args.start_checkpoint else None

    if current_checkpoint is None:
        stage_root = pipeline_dir / "stage1_sandbox"
        checkpoint_dir = stage_root / "checkpoints"
        log_dir = stage_root / "runs"
        log_path = stage_root / "sandbox.log"
        cmd = [
            sys.executable,
            "legacy/train_v2.py",
            "--combat-sandbox",
            "--total-timesteps",
            str(args.stage1_total_timesteps),
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--log-dir",
            str(log_dir),
        ]
        add_common_train_v2_args(
            cmd,
            device=args.device,
            session_file=args.session_file,
            character=args.stage1_character or args.character,
            learning_rate=args.stage1_learning_rate,
            checkpoint_freq=args.stage1_checkpoint_freq,
        )
        if args.stage1_encounter_pool:
            cmd.extend(["--encounter-pool", args.stage1_encounter_pool])
        if args.stage1_encounter_id:
            cmd.extend(["--encounter-id", args.stage1_encounter_id])
        if args.stage1_eval_holdout_pool:
            cmd.extend(["--eval-holdout-pool", args.stage1_eval_holdout_pool])
        if args.stage1_eval_freq is not None:
            cmd.extend(["--eval-freq", str(args.stage1_eval_freq)])

        run_logged(cmd, cwd=script_dir, log_path=log_path)
        current_checkpoint = resolve_stage_checkpoint(stage_root, kind="train_v2")
        summary["stages"]["sandbox"] = {
            "checkpoint": str(current_checkpoint),
            "log": str(log_path),
            "command": cmd,
        }
        write_summary(summary_path, summary)
        if args.stop_after == "sandbox":
            summary["final_checkpoint"] = str(current_checkpoint)
            write_summary(summary_path, summary)
            print(f"[pipeline] done. final checkpoint: {current_checkpoint}")
            return
    else:
        summary["stages"]["sandbox"] = {
            "skipped": True,
            "source_checkpoint": str(current_checkpoint),
        }
        write_summary(summary_path, summary)
        if args.stop_after == "sandbox":
            summary["final_checkpoint"] = str(current_checkpoint)
            write_summary(summary_path, summary)
            print(f"[pipeline] done. final checkpoint: {current_checkpoint}")
            return

    stage_root = pipeline_dir / "stage2_offline"
    log_path = stage_root / "offline.log"
    cmd = [
        sys.executable,
        "legacy/train_offline_multitask.py",
        "--dataset-root",
        args.dataset_root,
        "--out-dir",
        str(stage_root),
        "--run-name",
        "offline",
        "--init-checkpoint",
        str(current_checkpoint),
        "--tasks",
        args.stage2_tasks,
        "--partition-kind",
        args.stage2_partition_kind,
        "--partition-value",
        args.stage2_partition_value,
        "--epochs",
        str(args.stage2_epochs),
        "--batch-size",
        str(args.stage2_batch_size),
        "--learning-rate",
        str(args.stage2_learning_rate),
        "--weight-decay",
        str(args.stage2_weight_decay),
        "--device",
        args.device,
    ]
    if args.stage2_max_train_batches is not None:
        cmd.extend(["--max-train-batches", str(args.stage2_max_train_batches)])
    if args.stage2_max_eval_batches is not None:
        cmd.extend(["--max-eval-batches", str(args.stage2_max_eval_batches)])
    if args.stage2_task_repeat_factors:
        cmd.extend(["--task-repeat-factors", args.stage2_task_repeat_factors])

    run_logged(cmd, cwd=script_dir, log_path=log_path)
    current_checkpoint = resolve_stage_checkpoint(
        stage_root,
        kind="offline",
        preference=args.stage2_checkpoint_kind,
    )
    summary["stages"]["offline"] = {
        "checkpoint": str(current_checkpoint),
        "checkpoint_kind": args.stage2_checkpoint_kind,
        "log": str(log_path),
        "command": cmd,
    }
    write_summary(summary_path, summary)
    if args.stop_after == "offline":
        summary["final_checkpoint"] = str(current_checkpoint)
        write_summary(summary_path, summary)
        print(f"[pipeline] done. final checkpoint: {current_checkpoint}")
        return

    stage_root = pipeline_dir / "stage3_fullrun"
    checkpoint_dir = stage_root / "checkpoints"
    log_dir = stage_root / "runs"
    log_path = stage_root / "fullrun.log"
    cmd = [
        sys.executable,
        "legacy/train_v2.py",
        "--total-timesteps",
        str(args.stage3_total_timesteps),
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--log-dir",
        str(log_dir),
        "--init-checkpoint",
        str(current_checkpoint),
    ]
    add_common_train_v2_args(
        cmd,
        device=args.device,
        session_file=args.session_file,
        character=args.stage3_character or args.character,
        learning_rate=args.stage3_learning_rate,
        checkpoint_freq=args.stage3_checkpoint_freq,
    )

    run_logged(cmd, cwd=script_dir, log_path=log_path)
    current_checkpoint = resolve_stage_checkpoint(stage_root, kind="train_v2")
    summary["stages"]["fullrun"] = {
        "checkpoint": str(current_checkpoint),
        "log": str(log_path),
        "command": cmd,
    }
    summary["final_checkpoint"] = str(current_checkpoint)
    write_summary(summary_path, summary)
    print(f"[pipeline] done. final checkpoint: {current_checkpoint}")


if __name__ == "__main__":
    main()
