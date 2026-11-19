"""Multi-stage gate-driven automatic curriculum ratchet for Durable Afterstate Macro Credit on Colab G4.

Advances smoothly across decreasing revival budgets:
    b24 -> b20 -> b16 -> b12 -> b8 -> b6 -> b4 -> b3 -> b2 -> b1 (+ final b0 permadeath evaluation)

Key invariants:
- Fail-closed promotion: a stage promotes ONLY when a committed milestone update u >= min_warmup_updates
  passes the 48-seed odd Held-out gate on win count, mean revivals used, card-skip discipline, and
  bounded final deck size. Hitting ``max_updates_per_stage`` pauses cleanly rather than promoting a
  degraded or bloated policy.
- Lossless value warmup at every stage boundary: each promoted publication warm-starts the next
  stage via ``--initialize-from-macro-publication`` + ``--teacher-publication`` so updates 1..5
  calibrate the Twin/Shop/Build value heads with frozen trunk and actors before policy steps resume.
- Live dashboard pointer: atomically maintains ``active_ratchet_state.json`` on local NVMe and
  Google Drive so ``sync_colab_g4_to_dashboard.py --follow-ratchet-state`` automatically closes
  each promoted stage and switches telemetry to the new stage without manual intervention.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sts2_rl.artifacts import (
    artifact_root,
    resolve_artifact_path,
    resolve_external_input_path,
    validate_artifact_component,
)

RATCHET_RUNNER_VERSION = "sts2-colab-durable-ratchet-v1"


@dataclass(frozen=True, slots=True)
class RatchetStageSpec:
    budget: int
    wins_required: int
    mean_revivals_limit: float
    seed_index_offset: int
    min_skip_rate: float = 0.08
    max_mean_deck_size: float = 35.0
    reward_objective: str = "run"
    stage_tag: str = ""

    @property
    def stage_key(self) -> str:
        return self.stage_tag if self.stage_tag else f"b{self.budget}"


DEFAULT_SMOOTH_LADDER: tuple[RatchetStageSpec, ...] = (
    RatchetStageSpec(budget=24, wins_required=20, mean_revivals_limit=20.6, seed_index_offset=600_000),
    RatchetStageSpec(budget=20, wins_required=16, mean_revivals_limit=17.2, seed_index_offset=650_000),
    RatchetStageSpec(budget=16, wins_required=15, mean_revivals_limit=13.8, seed_index_offset=700_000),
    RatchetStageSpec(budget=12, wins_required=14, mean_revivals_limit=10.4, seed_index_offset=750_000),
    RatchetStageSpec(budget=8, wins_required=13, mean_revivals_limit=6.9, seed_index_offset=800_000),
    RatchetStageSpec(budget=6, wins_required=12, mean_revivals_limit=5.2, seed_index_offset=850_000),
    RatchetStageSpec(budget=4, wins_required=11, mean_revivals_limit=3.5, seed_index_offset=900_000),
    RatchetStageSpec(budget=3, wins_required=10, mean_revivals_limit=2.6, seed_index_offset=950_000),
    RatchetStageSpec(budget=2, wins_required=10, mean_revivals_limit=1.75, seed_index_offset=1_000_000),
    RatchetStageSpec(budget=1, wins_required=9, mean_revivals_limit=0.88, seed_index_offset=1_050_000),
)

ZERO_REVIVAL_ACT_LADDER: tuple[RatchetStageSpec, ...] = (
    RatchetStageSpec(
        budget=0,
        reward_objective="act1",
        stage_tag="b0-act1",
        wins_required=28,
        mean_revivals_limit=0.0,
        seed_index_offset=700_000,
        min_skip_rate=0.08,
        max_mean_deck_size=21.0,
    ),
    RatchetStageSpec(
        budget=0,
        reward_objective="act2",
        stage_tag="b0-act2",
        wins_required=18,
        mean_revivals_limit=0.0,
        seed_index_offset=800_000,
        min_skip_rate=0.10,
        max_mean_deck_size=28.0,
    ),
    RatchetStageSpec(
        budget=0,
        reward_objective="run",
        stage_tag="b0-act3",
        wins_required=12,
        mean_revivals_limit=0.0,
        seed_index_offset=900_000,
        min_skip_rate=0.12,
        max_mean_deck_size=33.0,
    ),
)


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with tmp.open("wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _atomic_copy_verified(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    src_sha = _sha256_file(src)
    if dst.is_file() and _sha256_file(dst) == src_sha:
        return src_sha
    tmp = dst.with_name(f".{dst.name}.tmp-{os.getpid()}")
    with src.open("rb") as reader, tmp.open("wb") as writer:
        shutil.copyfileobj(reader, writer, length=4 * 1024 * 1024)
        writer.flush()
        os.fsync(writer.fileno())
    if _sha256_file(tmp) != src_sha:
        tmp.unlink(missing_ok=True)
        raise OSError(f"sha256 mismatch copying {src} -> {dst}")
    os.replace(tmp, dst)
    return src_sha


def evaluate_stage_promotion_gate(
    row: Mapping[str, Any],
    stage: RatchetStageSpec,
    *,
    expected_episodes: int,
    min_policy_version: int,
) -> tuple[bool, dict[str, Any]]:
    """Evaluate whether a held-out summary JSON satisfies ``stage``'s promotion gate."""
    pv = row.get("policy_version", row.get("evaluation_gate"))
    episodes = row.get("episodes")
    wins = row.get("successful_run_count")
    budget = row.get("revival_budget")
    mean_rev = row.get("mean_revivals_used")
    skip_rate = row.get("skip_rate")
    deck_size = row.get("mean_final_deck_size")
    deadlocks = row.get("deadlock_count", 0)

    row_objective = row.get("reward_objective")

    checks = {
        "policy_version_ready": isinstance(pv, int) and not isinstance(pv, bool) and pv >= min_policy_version,
        "episodes_match": episodes == expected_episodes,
        "budget_match": budget == stage.budget,
        "objective_match": row_objective is None or row_objective == stage.reward_objective,
        "zero_deadlocks": deadlocks == 0,
        "wins_ok": isinstance(wins, int) and not isinstance(wins, bool) and wins >= stage.wins_required,
        "revivals_ok": (
            isinstance(mean_rev, int | float)
            and not isinstance(mean_rev, bool)
            and math.isfinite(float(mean_rev))
            and 0.0 <= float(mean_rev) <= stage.mean_revivals_limit
        ),
        "skip_discipline_ok": (
            isinstance(skip_rate, int | float)
            and not isinstance(skip_rate, bool)
            and math.isfinite(float(skip_rate))
            and float(skip_rate) >= stage.min_skip_rate
        ),
        "deck_size_ok": (
            isinstance(deck_size, int | float)
            and not isinstance(deck_size, bool)
            and math.isfinite(float(deck_size))
            and float(deck_size) <= stage.max_mean_deck_size
        ),
    }
    passed = all(checks.values())
    return passed, {
        "passed": passed,
        "policy_version": pv,
        "wins": wins,
        "wins_required": stage.wins_required,
        "mean_revivals_used": mean_rev,
        "mean_revivals_limit": stage.mean_revivals_limit,
        "skip_rate": skip_rate,
        "min_skip_rate": stage.min_skip_rate,
        "mean_final_deck_size": deck_size,
        "max_mean_deck_size": stage.max_mean_deck_size,
        "checks": checks,
    }


def select_intrastage_teacher_reanchor(
    rows: Sequence[Mapping[str, Any]],
    stage: RatchetStageSpec,
    *,
    expected_episodes: int,
    min_policy_version: int,
    min_win_gain: int = 2,
) -> int | None:
    """Return the best anti-bloat-verified milestone update u >= min_policy_version that strictly beats baseline."""
    baseline_wins: int | None = None
    candidates: list[tuple[int, float, float, int]] = []
    for row in rows:
        pv = row.get("policy_version", row.get("evaluation_gate"))
        episodes = row.get("episodes")
        wins = row.get("successful_run_count")
        budget = row.get("revival_budget")
        mean_rev = row.get("mean_revivals_used")
        skip_rate = row.get("skip_rate")
        deck_size = row.get("mean_final_deck_size")
        deadlocks = row.get("deadlock_count", 0)
        if not (
            isinstance(pv, int)
            and not isinstance(pv, bool)
            and episodes == expected_episodes
            and budget == stage.budget
            and isinstance(wins, int)
            and not isinstance(wins, bool)
        ):
            continue
        if pv in {0, min_policy_version}:
            baseline_wins = wins if baseline_wins is None else max(baseline_wins, wins)
        if pv <= min_policy_version or deadlocks != 0:
            continue
        if not (
            isinstance(skip_rate, int | float)
            and not isinstance(skip_rate, bool)
            and math.isfinite(float(skip_rate))
            and float(skip_rate) >= stage.min_skip_rate
        ):
            continue
        if not (
            isinstance(deck_size, int | float)
            and not isinstance(deck_size, bool)
            and math.isfinite(float(deck_size))
            and float(deck_size) <= stage.max_mean_deck_size
        ):
            continue
        rev_val = float(mean_rev) if isinstance(mean_rev, int | float) and math.isfinite(float(mean_rev)) else 1e9
        act3_val = float(row.get("act3_reach_rate") or 0.0)
        candidates.append((wins, -rev_val, act3_val, pv))

    if not candidates:
        return None
    best_wins, _, _, best_pv = max(candidates)
    if baseline_wins is not None and best_wins < baseline_wins + max(1, min_win_gain):
        return None
    return best_pv


def _read_heldout_rows(stage_dir: Path) -> list[dict[str, Any]]:
    path = stage_dir / "heldout_evaluations.jsonl"
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if raw:
            item = json.loads(raw)
            if isinstance(item, dict):
                rows.append(item)
    return rows


def _read_committed_update(stage_dir: Path) -> int:
    status_path = stage_dir / "status.json"
    if not status_path.is_file():
        return 0
    data = json.loads(status_path.read_text(encoding="utf-8"))
    return int(data.get("committed_update", 0))


def _prepare_runtime_env(root: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["STS2_ARTIFACT_ROOT"] = str(root)
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if "tcmalloc" in env.get("LD_PRELOAD", ""):
        env.pop("LD_PRELOAD", None)
    shim = Path("/root/libfix_mountinfo.so")
    if shim.is_file() and str(shim) not in env.get("LD_PRELOAD", ""):
        prev = env.get("LD_PRELOAD", "").strip()
        env["LD_PRELOAD"] = f"{shim}:{prev}" if prev else str(shim)
    cur_dn = env.get("DOTNET_ROOT", "").strip()
    if not cur_dn or not (Path(cur_dn) / "dotnet").is_file():
        for cand_dn in (root / "dotnet", Path.home() / ".dotnet", Path("/root/.dotnet")):
            if (cand_dn / "dotnet").is_file():
                env["DOTNET_ROOT"] = str(cand_dn)
                env["PATH"] = f"{cand_dn}{os.pathsep}{env.get('PATH', '')}"
                break
    return env


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "packages/rl-agent/config/experiments/hybrid_macro_credit_v22_durable_afterstate_v66_ratchet_g4_turbo.toml"
        ),
    )
    parser.add_argument("--lineage-prefix", default="durable-afterstate-v6-6-ratchet")
    parser.add_argument("--lineage-suffix", default="g4")
    parser.add_argument("--initial-publication", type=Path, required=True)
    parser.add_argument("--drive-root-dir", type=Path, required=True)
    parser.add_argument("--active-state-file", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sim-exe", type=Path, default=None)
    parser.add_argument("--step-updates", type=int, default=5)
    parser.add_argument("--max-updates-per-stage", type=int, default=80)
    parser.add_argument("--min-warmup-updates", type=int, default=5)
    parser.add_argument("--heldout-episodes", type=int, default=48)
    parser.add_argument("--heldout-workers", type=int, default=32)
    parser.add_argument("--keep-raw-trace-cohorts", type=int, default=2)
    parser.add_argument(
        "--curriculum-mode",
        choices=("revival-ratchet", "act-b0"),
        default="revival-ratchet",
        help="'revival-ratchet' decreases revival budget on full runs; 'act-b0' trains B=0 across act1 -> act2 -> run",
    )
    parser.add_argument(
        "--budgets",
        default="",
        help="optional comma-separated subset of budgets from DEFAULT_SMOOTH_LADDER (e.g. '24,20,16')",
    )
    parser.add_argument(
        "--stages",
        default="",
        help="optional comma-separated subset of stage_key values (e.g. 'b0-act1,b0-act2,b0-act3')",
    )
    return parser


def run_ratchet(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.step_updates <= 0 or args.max_updates_per_stage < args.step_updates:
        raise ValueError("invalid --step-updates / --max-updates-per-stage")

    prefix = validate_artifact_component(args.lineage_prefix, label="lineage_prefix")
    suffix = validate_artifact_component(args.lineage_suffix, label="lineage_suffix")
    ratchet_id = f"{prefix}-{suffix}"

    root = artifact_root()
    config_path = args.config if args.config.is_absolute() else Path.cwd() / args.config
    if not config_path.is_file():
        raise FileNotFoundError(f"config file not found: {config_path}")

    init_pub = resolve_external_input_path(args.initial_publication, root=root)
    if not init_pub.is_file():
        raise FileNotFoundError(f"initial publication not found: {init_pub}")

    drive_root = resolve_external_input_path(args.drive_root_dir, root=root)
    drive_ratchet_dir = drive_root / ratchet_id
    drive_ratchet_dir.mkdir(parents=True, exist_ok=True)

    active_state_path = (
        resolve_artifact_path(args.active_state_file, root=root)
        if args.active_state_file is not None
        else (root / "active_ratchet_state.json")
    )
    drive_state_path = drive_ratchet_dir / "ratchet_state.json"
    promotions_dir = root / "ratchet_promotions" / ratchet_id
    promotions_dir.mkdir(parents=True, exist_ok=True)

    if args.curriculum_mode == "act-b0":
        base_ladder = ZERO_REVIVAL_ACT_LADDER
        by_key = {spec.stage_key: spec for spec in base_ladder}
        if args.stages.strip():
            wanted_keys = [tok.strip() for tok in args.stages.split(",") if tok.strip()]
            stages = tuple(by_key[k] for k in wanted_keys)
        else:
            stages = base_ladder
    else:
        by_budget = {spec.budget: spec for spec in DEFAULT_SMOOTH_LADDER}
        if args.budgets.strip():
            wanted = [int(tok.strip()) for tok in args.budgets.split(",") if tok.strip()]
            stages = tuple(by_budget[b] for b in wanted)
        elif args.stages.strip():
            by_key = {spec.stage_key: spec for spec in DEFAULT_SMOOTH_LADDER}
            wanted_keys = [tok.strip() for tok in args.stages.split(",") if tok.strip()]
            stages = tuple(by_key[k] for k in wanted_keys)
        else:
            stages = DEFAULT_SMOOTH_LADDER
        if any(b.budget >= a.budget for a, b in zip(stages, stages[1:])):
            raise ValueError("ratchet budgets must strictly decrease")

    # Load prior ratchet state if resuming across restarts (only if ratchet_id matches!)
    prior_state: dict[str, Any] = {}
    if active_state_path.is_file():
        cand_active = json.loads(active_state_path.read_text(encoding="utf-8"))
        if isinstance(cand_active, dict) and cand_active.get("ratchet_id") == ratchet_id:
            prior_state = cand_active
    if not prior_state and drive_state_path.is_file():
        cand_drive = json.loads(drive_state_path.read_text(encoding="utf-8"))
        if isinstance(cand_drive, dict) and cand_drive.get("ratchet_id") == ratchet_id:
            prior_state = cand_drive

    promoted_history: list[dict[str, Any]] = list(prior_state.get("promoted_history") or [])
    completed_stage_keys = {
        str(item["stage_key"]) if "stage_key" in item else f"b{int(item['budget'])}"
        for item in promoted_history
        if "stage_key" in item or "budget" in item
    }

    # Cache initial seed publication locally + on Drive
    seed_cache = promotions_dir / "stage0_seed_publication.pt"
    seed_sha = _atomic_copy_verified(init_pub, seed_cache)
    _atomic_copy_verified(seed_cache, drive_ratchet_dir / "stage0_seed_publication.pt")

    current_parent_pub = seed_cache
    current_parent_sha = seed_sha
    if promoted_history:
        last_prom = promoted_history[-1]
        cand_local = Path(str(last_prom.get("local_promoted_publication", "")))
        cand_drive = Path(str(last_prom.get("drive_promoted_publication", "")))
        if cand_local.is_file():
            current_parent_pub = cand_local
            current_parent_sha = _sha256_file(cand_local)
        elif cand_drive.is_file():
            restored_p = promotions_dir / cand_drive.name
            current_parent_sha = _atomic_copy_verified(cand_drive, restored_p)
            current_parent_pub = restored_p

    env = _prepare_runtime_env(root)
    campaign_script = Path(__file__).resolve().parent / "run_colab_g4_durable_campaign.py"
    milestone_str = ",".join(
        str(u) for u in range(0, args.max_updates_per_stage + 1, args.step_updates)
    )

    def write_state(
        *,
        stage_idx: int,
        stage_spec: RatchetStageSpec,
        stage_lineage: str,
        stage_local_dir: Path,
        stage_drive_dir: Path,
        target_upd: int,
        state_label: str,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "version": RATCHET_RUNNER_VERSION,
            "ratchet_id": ratchet_id,
            "updated_at_utc": _utc_now_iso(),
            "pid": os.getpid(),
            "stage_index": stage_idx,
            "total_stages": len(stages),
            "current_stage_key": stage_spec.stage_key,
            "current_reward_objective": stage_spec.reward_objective,
            "current_budget": stage_spec.budget,
            "current_lineage": stage_lineage,
            "current_local_dir": str(stage_local_dir),
            "current_drive_dir": str(stage_drive_dir),
            "target_updates": target_upd,
            "max_updates_per_stage": args.max_updates_per_stage,
            "stage_gate": asdict(stage_spec),
            "parent_publication_sha256": current_parent_sha,
            "state": state_label,
            "promoted_history": promoted_history,
        }
        if extra:
            payload.update(extra)
        _atomic_write_json(active_state_path, payload)
        _atomic_write_json(drive_state_path, payload)

    for stage_idx, stage in enumerate(stages):
        stage_lineage = f"{prefix}-{stage.stage_key}-{suffix}"
        stage_local_dir = root / "campaigns" / stage_lineage
        stage_drive_dir = drive_root / stage_lineage
        if stage.stage_key in completed_stage_keys:
            continue

        overrides = [
            f"learner.revival_budget={stage.budget}",
            f"training.curriculum.revival_budget={stage.budget}",
            f'training.curriculum.reward_objective="{stage.reward_objective}"',
            f"course.seed_index_offset={stage.seed_index_offset}",
        ]
        print(
            json.dumps(
                {
                    "event": "ratchet_stage_begin",
                    "stage_index": stage_idx,
                    "stage_key": stage.stage_key,
                    "reward_objective": stage.reward_objective,
                    "budget": stage.budget,
                    "lineage_id": stage_lineage,
                    "parent_sha256": current_parent_sha,
                    "gate": asdict(stage),
                },
                sort_keys=True,
            ),
            flush=True,
        )

        while True:
            committed = _read_committed_update(stage_local_dir)
            rows = _read_heldout_rows(stage_local_dir)
            # Check if any already-evaluated milestone u >= min_warmup_updates passes the gate!
            passing_row: dict[str, Any] | None = None
            passing_report: dict[str, Any] | None = None
            for row in rows:
                ok, rep = evaluate_stage_promotion_gate(
                    row,
                    stage,
                    expected_episodes=args.heldout_episodes,
                    min_policy_version=args.min_warmup_updates,
                )
                if ok:
                    passing_row = row
                    passing_report = rep
                    break

            if passing_row is not None and passing_report is not None:
                win_u = int(passing_row["policy_version"])
                snap_pub = stage_drive_dir / "snapshots" / f"update-{win_u:06d}" / "publication.pt"
                src_pub = snap_pub if snap_pub.is_file() else (stage_local_dir / "publication.pt")
                safe_key = stage.stage_key.replace("-", "_")
                local_prom = promotions_dir / f"{safe_key}_u{win_u:04d}_publication.pt"
                drive_prom = drive_ratchet_dir / f"{safe_key}_u{win_u:04d}_publication.pt"
                prom_sha = _atomic_copy_verified(src_pub, local_prom)
                _atomic_copy_verified(local_prom, drive_prom)

                prom_entry = {
                    "stage_index": stage_idx,
                    "stage_key": stage.stage_key,
                    "reward_objective": stage.reward_objective,
                    "budget": stage.budget,
                    "lineage_id": stage_lineage,
                    "promoted_at_update": win_u,
                    "promoted_at_utc": _utc_now_iso(),
                    "publication_sha256": prom_sha,
                    "local_promoted_publication": str(local_prom),
                    "drive_promoted_publication": str(drive_prom),
                    "gate_report": passing_report,
                }
                promoted_history.append(prom_entry)
                completed_stage_keys.add(stage.stage_key)
                current_parent_pub = local_prom
                current_parent_sha = prom_sha
                write_state(
                    stage_idx=stage_idx,
                    stage_spec=stage,
                    stage_lineage=stage_lineage,
                    stage_local_dir=stage_local_dir,
                    stage_drive_dir=stage_drive_dir,
                    target_upd=win_u,
                    state_label="stage_promoted",
                    extra={"latest_promotion": prom_entry},
                )
                print(
                    json.dumps({"event": "ratchet_stage_promoted", **prom_entry}, sort_keys=True),
                    flush=True,
                )
                break

            if committed >= args.max_updates_per_stage:
                write_state(
                    stage_idx=stage_idx,
                    stage_spec=stage,
                    stage_lineage=stage_lineage,
                    stage_local_dir=stage_local_dir,
                    stage_drive_dir=stage_drive_dir,
                    target_upd=committed,
                    state_label="paused_stage_cap_reached",
                )
                print(
                    json.dumps(
                        {
                            "event": "ratchet_paused_stage_cap",
                            "budget": stage.budget,
                            "lineage_id": stage_lineage,
                            "committed_update": committed,
                            "max_updates_per_stage": args.max_updates_per_stage,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                return 2

            next_target = min(
                args.max_updates_per_stage,
                ((committed // args.step_updates) + 1) * args.step_updates,
            )
            effective_teacher_pub = current_parent_pub
            reanchor_u = select_intrastage_teacher_reanchor(
                rows,
                stage,
                expected_episodes=args.heldout_episodes,
                min_policy_version=args.min_warmup_updates,
            )
            reanchored = False
            if reanchor_u is not None:
                cand_reanchor = stage_drive_dir / "snapshots" / f"update-{reanchor_u:06d}" / "publication.pt"
                if cand_reanchor.is_file():
                    safe_key = stage.stage_key.replace("-", "_")
                    local_reanchor = promotions_dir / f"reanchor_{safe_key}_u{reanchor_u:04d}_publication.pt"
                    _atomic_copy_verified(cand_reanchor, local_reanchor)
                    effective_teacher_pub = local_reanchor
                    reanchored = True

            write_state(
                stage_idx=stage_idx,
                stage_spec=stage,
                stage_lineage=stage_lineage,
                stage_local_dir=stage_local_dir,
                stage_drive_dir=stage_drive_dir,
                target_upd=next_target,
                state_label="running_stage",
                extra={"intrastage_reanchor_update": reanchor_u if reanchored else None},
            )

            cmd = [
                sys.executable,
                str(campaign_script),
                "--config",
                str(config_path),
                "--output-dir",
                str(stage_local_dir),
                "--drive-campaign-dir",
                str(stage_drive_dir),
                "--lineage-id",
                stage_lineage,
                "--initialize-from-macro-publication",
                str(current_parent_pub),
                "--teacher-publication",
                str(effective_teacher_pub),
                "--resume-from-drive",
                "--device",
                args.device,
                "--target-updates",
                str(next_target),
                "--updates-per-chunk",
                str(args.step_updates),
                "--snapshot-interval",
                str(args.step_updates),
                "--keep-raw-trace-cohorts",
                str(args.keep_raw_trace_cohorts),
                "--heldout-eval-milestones",
                milestone_str,
                "--heldout-eval-episodes",
                str(args.heldout_episodes),
                "--heldout-eval-workers",
                str(args.heldout_workers),
            ]
            if reanchored:
                cmd.append("--allow-teacher-reanchor")
            if args.sim_exe is not None:
                cmd.extend(["--sim-exe", str(args.sim_exe)])
            for ov in overrides:
                cmd.extend(["--set", ov])

            completed = subprocess.run(cmd, env=env, check=False)
            if completed.returncode != 0:
                write_state(
                    stage_idx=stage_idx,
                    stage_spec=stage,
                    stage_lineage=stage_lineage,
                    stage_local_dir=stage_local_dir,
                    stage_drive_dir=stage_drive_dir,
                    target_upd=next_target,
                    state_label="aborted",
                    extra={"failed_returncode": completed.returncode},
                )
                return completed.returncode

    # Final stage passed! If final stage was b1, run a 48-seed b0 permadeath held-out audit for provenance!
    permadeath_audit: dict[str, Any] | None = None
    if stages and stages[-1].budget == 1 and current_parent_pub.is_file():
        audit_cmd = [
            sys.executable,
            "-m",
            "sts2_rl.macro_heldout_eval",
            "--publication",
            str(current_parent_pub),
            "--config",
            str(config_path),
            "--policy-version",
            "0",
            "--episodes",
            str(args.heldout_episodes),
            "--workers",
            str(args.heldout_workers),
            "--device",
            args.device,
            "--set",
            "learner.revival_budget=1",
            "--set",
            "training.curriculum.revival_budget=1",
        ]
        if args.sim_exe is not None:
            audit_cmd.extend(["--sim-exe", str(args.sim_exe)])
        audit_res = subprocess.run(audit_cmd, env=env, capture_output=True, text=True, check=False)
        if audit_res.returncode == 0:
            lines = [ln.strip() for ln in audit_res.stdout.splitlines() if ln.strip()]
            if lines:
                b1_final = json.loads(lines[-1])
                permadeath_audit = {
                    "b1_zero_revival_victories": b1_final.get("zero_revival_victories"),
                    "b1_zero_revival_win_rate": (
                        float(b1_final.get("zero_revival_victories", 0))
                        / max(1, int(b1_final.get("episodes", 48)))
                    ),
                    "b1_run_win_rate": b1_final.get("run_win_rate"),
                }
                _atomic_write_json(drive_ratchet_dir / "final_b1_zero_revival_audit.json", b1_final)

    last_stage = stages[-1]
    last_lineage = f"{prefix}-{last_stage.stage_key}-{suffix}"
    write_state(
        stage_idx=len(stages) - 1,
        stage_spec=last_stage,
        stage_lineage=last_lineage,
        stage_local_dir=root / "campaigns" / last_lineage,
        stage_drive_dir=drive_root / last_lineage,
        target_upd=_read_committed_update(root / "campaigns" / last_lineage),
        state_label="ratchet_complete",
        extra={"permadeath_audit": permadeath_audit} if permadeath_audit is not None else None,
    )
    print(
        json.dumps(
            {
                "event": "ratchet_curriculum_complete",
                "ratchet_id": ratchet_id,
                "stages_completed": len(stages),
                "final_publication": str(current_parent_pub),
                "final_publication_sha256": current_parent_sha,
                "permadeath_audit": permadeath_audit,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return run_ratchet(argv)
    except BaseException as error:
        print(
            json.dumps(
                {
                    "event": "ratchet_aborted",
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 130 if isinstance(error, KeyboardInterrupt) else 1


if __name__ == "__main__":
    raise SystemExit(main())
