"""Evaluate a PPO+attention checkpoint on a combat snapshot pool filtered
to boss encounters. Reports per-encounter win rate + tier summary.

Purpose: answer the Phase 9 P-1 question — "what's the current arch's
actual combat win rate on Act 1/2/3 boss starting states?" — using the
existing CombatSandboxEnv + CombatSnapshotPool machinery instead of
live game bridge eval.

Usage (sim backend, no live game needed):
  python eval_combat_sandbox_boss.py \
      checkpoints_attention/sandbox_starter_early_patch_20260418_122809/step_000450560 \
      --snapshot-pool datasets/last_used_sandbox_subset.jsonl \
      --tiers boss \
      --episodes 200 \
      --device cuda \
      --use-sim \
      --output-json analysis/training_curves/<name>_boss_eval_<stamp>.json
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sb3_contrib.common.wrappers import ActionMasker

from combat_snapshot_dataset import CombatSnapshotPool
from sts2_env.checkpoint import load_online_checkpoint
from sts2_env.combat_env import CombatSandboxEnv
from sts2_env.observation_v3 import WorldTokenObservationEncoder


def mask_fn(env):
    return env.unwrapped.action_masks()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_dir")
    parser.add_argument("--snapshot-pool", required=True,
                        help="Path to snapshot pool JSONL, parquet, or a curated dataset directory")
    parser.add_argument("--curated-subset", default=None,
                        help="Name of curated subset inside a dataset directory "
                             "(e.g. bootstrap_human_plus_local_all_roomwin_only_minus_combat_reset_failures)")
    parser.add_argument("--build-id", default=None,
                        help="Restrict pool to rows whose build_id matches the current sim build (e.g. v0.99.1)")
    parser.add_argument("--tiers", default="boss",
                        help="Comma-separated tiers to include (boss / elite / normal / weak / hard_normal). "
                             "Use 'all' to include everything.")
    parser.add_argument("--encounter-ids", default="",
                        help="Optional comma-separated subset of encounter_ids to restrict eval to")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--text-device", default="auto")
    parser.add_argument("--no-text", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--use-sim", action="store_true",
                        help="Drive HeadlessSim subprocess (no live game needed)")
    parser.add_argument("--sim-exe-path", default=None)
    parser.add_argument("--session-file", default=None)
    parser.add_argument("--character", default="ironclad")
    parser.add_argument("--sample-mode", default="encounter_balanced",
                        help="encounter_balanced / uniform / tier_weighted")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--max-steps-per-ep", type=int, default=500,
                        help="Hard cap on steps per episode — truncate if exceeded")
    args = parser.parse_args()

    tiers = None
    if args.tiers and args.tiers.lower() != "all":
        tiers = [t.strip() for t in args.tiers.split(",") if t.strip()]
    encounter_ids = None
    if args.encounter_ids:
        encounter_ids = [t.strip() for t in args.encounter_ids.split(",") if t.strip()]

    print(f"[eval] loading snapshot pool: {args.snapshot_pool}")
    if args.curated_subset:
        print(f"[eval] curated subset: {args.curated_subset}")
    if args.build_id:
        print(f"[eval] build_id filter: {args.build_id}")
    pool = CombatSnapshotPool.from_path(
        args.snapshot_pool,
        curated_subset=args.curated_subset,
        build_id=args.build_id,
        encounter_tiers=tiers,
        encounter_ids=encounter_ids,
        sample_mode=args.sample_mode,
    )
    print(f"[eval] pool size: {len(pool)} rows, {pool.encounter_count} unique encounters")
    summary_rows = pool.summary()
    tier_cnt = summary_rows.get("tier_counts", {})
    print(f"[eval] tier counts: {dict(tier_cnt)}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Resolve text device
    if args.text_device == "auto":
        text_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        text_device = args.text_device

    sim_bridge = None
    if args.use_sim:
        from sts2_env.headless_sim_bridge_client import HeadlessSimBridgeClient
        sim_bridge = HeadlessSimBridgeClient(exe_path=args.sim_exe_path)

    obs_encoder = WorldTokenObservationEncoder(
        use_text=(not args.no_text), text_device=text_device,
    )
    env = CombatSandboxEnv(
        session_file=args.session_file,
        character=args.character,
        snapshot_pool=pool,
        obs_encoder=obs_encoder,
        include_debug_info=True,
        bridge=sim_bridge,
    )
    env = ActionMasker(env, mask_fn)

    print(f"[eval] loading checkpoint: {args.checkpoint_dir}")
    model, metadata = load_online_checkpoint(
        args.checkpoint_dir, env, device=str(device),
    )
    print(f"[eval] checkpoint observation_api_version: {metadata.get('observation_api_version')}")
    print(f"[eval] checkpoint timesteps: {metadata.get('timesteps')}")

    # Track per-encounter outcomes
    per_encounter: dict[str, dict[str, int]] = collections.defaultdict(
        lambda: {"total": 0, "wins": 0, "losses": 0, "truncated": 0}
    )
    per_tier: dict[str, dict[str, int]] = collections.defaultdict(
        lambda: {"total": 0, "wins": 0, "losses": 0, "truncated": 0}
    )
    overall = {"total": 0, "wins": 0, "losses": 0, "truncated": 0}
    episode_steps_list: list[int] = []
    episode_rewards_list: list[float] = []

    broken_resets = 0
    broken_samples: list[dict[str, Any]] = []
    MAX_RESET_RETRIES = 8
    t_start = time.time()
    for ep in range(args.episodes):
        obs, info = None, None
        for _retry in range(MAX_RESET_RETRIES):
            obs, info = env.reset()
            n_legal = int(info.get("legal_action_count") or 0)
            phase = info.get("phase")
            ts = info.get("transition_state") or {}
            combat = ts.get("combat") if isinstance(ts, dict) else None
            enemies = combat.get("enemies") if isinstance(combat, dict) else None
            n_enemies_alive = 0
            if isinstance(enemies, list):
                n_enemies_alive = sum(
                    1 for e in enemies if isinstance(e, dict) and float(e.get("hp") or 0) > 0
                )
            if n_legal > 0 and n_enemies_alive > 0:
                break
            broken_resets += 1
            if len(broken_samples) < 3:
                snap_sample_id = info.get("snapshot_sample_id")
                snap_build_id = info.get("snapshot_build_id")
                player_ts = ts.get("player") if isinstance(ts, dict) else None
                broken_samples.append({
                    "encounter_id": info.get("encounter_id"),
                    "snapshot_sample_id": snap_sample_id,
                    "snapshot_build_id": snap_build_id,
                    "phase": phase,
                    "n_legal": n_legal,
                    "n_enemies_alive": n_enemies_alive,
                    "player": player_ts,
                    "enemies_raw_len": len(enemies) if isinstance(enemies, list) else None,
                    "first_enemy": enemies[0] if isinstance(enemies, list) and enemies else None,
                })
        encounter_id = info.get("encounter_id") or "UNKNOWN"
        tier = info.get("encounter_tier") or "?"
        terminated = False
        truncated = False
        reward_sum = 0.0
        steps = 0
        while not (terminated or truncated) and steps < args.max_steps_per_ep:
            mask = env.action_masks()
            with torch.no_grad():
                batch = {k: torch.as_tensor(v).unsqueeze(0).to(device) for k, v in obs.items()}
                action_masks_b = mask.reshape(1, -1)
                if args.deterministic:
                    actions, _, _ = model.policy(
                        batch, deterministic=True, action_masks=action_masks_b,
                    )
                else:
                    actions, _, _ = model.policy(batch, action_masks=action_masks_b)
            act_idx = int(actions.item())
            obs, reward, terminated, truncated, info = env.step(act_idx)
            reward_sum += float(reward)
            steps += 1

        # Classify outcome
        outcome = "win" if terminated and reward_sum > 0 else "truncated" if truncated else "loss"
        # More robust: bridge usually sets info['combat_won'] or similar
        outcome_bridge = info.get("combat_outcome")
        if outcome_bridge == "win":
            outcome = "win"
        elif outcome_bridge in ("loss", "defeat"):
            outcome = "loss"

        per_encounter[encounter_id]["total"] += 1
        per_encounter[encounter_id][outcome if outcome in ("wins", "losses", "truncated") else {"win": "wins", "loss": "losses", "truncated": "truncated"}[outcome]] += 1
        per_tier[tier]["total"] += 1
        per_tier[tier][{"win": "wins", "loss": "losses", "truncated": "truncated"}[outcome]] += 1
        overall["total"] += 1
        overall[{"win": "wins", "loss": "losses", "truncated": "truncated"}[outcome]] += 1
        episode_steps_list.append(steps)
        episode_rewards_list.append(reward_sum)

        if (ep + 1) % 25 == 0:
            elapsed = time.time() - t_start
            wr = overall["wins"] / max(overall["total"], 1) * 100
            tier_str = " | ".join(
                f"{t}:{v['wins']}/{v['total']}={v['wins']/max(v['total'],1)*100:.0f}%"
                for t, v in sorted(per_tier.items()) if v["total"] > 0
            )
            print(
                f"[{elapsed:7.1f}s] ep {ep+1}/{args.episodes}  overall_wr={wr:.1f}%  {tier_str}",
                flush=True,
            )

    elapsed = time.time() - t_start
    print()
    print(f"=== final @ {elapsed:.1f}s ===")
    print(f"broken_resets (retried, not counted in totals): {broken_resets}")
    for sample in broken_samples:
        print(f"  broken sample: {sample}")
    print(f"overall: {overall}")
    print(f"  win rate: {overall['wins']/max(overall['total'],1)*100:.2f}%")
    print(f"  mean steps/ep: {np.mean(episode_steps_list):.1f}")
    print(f"  mean reward/ep: {np.mean(episode_rewards_list):.3f}")
    print()
    print("per-tier:")
    for tier, v in sorted(per_tier.items()):
        wr = v["wins"] / max(v["total"], 1) * 100
        print(f"  {tier:>10}  n={v['total']:>4}  wins={v['wins']:>4}  losses={v['losses']:>4}  trunc={v['truncated']:>3}  wr={wr:>5.1f}%")
    print()
    print("per-encounter (sorted by win rate):")
    rows = []
    for enc, v in per_encounter.items():
        if v["total"] > 0:
            wr = v["wins"] / v["total"] * 100
            rows.append((enc, v["total"], v["wins"], wr))
    for enc, total, wins, wr in sorted(rows, key=lambda x: -x[3]):
        print(f"  {enc:<45}  n={total:>3}  wins={wins:>3}  wr={wr:>5.1f}%")

    # Write summary JSON
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "checkpoint_dir": str(args.checkpoint_dir),
            "checkpoint_metadata": {
                "observation_api_version": metadata.get("observation_api_version"),
                "timesteps": metadata.get("timesteps"),
                "kind": metadata.get("kind"),
            },
            "snapshot_pool_path": str(args.snapshot_pool),
            "curated_subset": args.curated_subset,
            "build_id_filter": args.build_id,
            "tiers_filter": tiers,
            "encounter_ids_filter": encounter_ids,
            "episodes": args.episodes,
            "broken_resets": broken_resets,
            "broken_samples": broken_samples,
            "elapsed_seconds": elapsed,
            "overall": overall,
            "per_tier": {t: dict(v) for t, v in per_tier.items()},
            "per_encounter": {e: dict(v) for e, v in per_encounter.items()},
            "mean_steps": float(np.mean(episode_steps_list)) if episode_steps_list else 0.0,
            "mean_reward": float(np.mean(episode_rewards_list)) if episode_rewards_list else 0.0,
        }
        output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nwrote {output_path}")


if __name__ == "__main__":
    main()
