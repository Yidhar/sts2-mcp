"""Sim v0.103.2 vs live ranking validation.

Runs the same checkpoint on the same snapshot (same seed) on both sim
and live sides. Per-snapshot: compare outcome (win/loss/truncated),
step count, reward. Aggregates agreement rate across N snapshots from
M encounters.

If sim and live agree on outcomes >=80% of the time, sim is usable for
iterative eval (absolute WR may differ but ranking is preserved). If
<60%, sim eval can't be trusted even for ranking — must use live.
"""
from __future__ import annotations

import argparse
import collections
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sb3_contrib.common.wrappers import ActionMasker

from combat_snapshot_dataset import CombatSnapshotPool
from sts2_env.bridge_client import BridgeClient
from sts2_env.checkpoint import load_online_checkpoint
from sts2_env.combat_env import CombatSandboxEnv
from sts2_env.headless_sim_bridge_client import HeadlessSimBridgeClient
from sts2_env.observation_v3 import WorldTokenObservationEncoder


def mask_fn(env):
    return env.unwrapped.action_masks()


def classify_outcome(terminated: bool, truncated: bool, info: dict) -> str:
    if truncated:
        return "truncated"
    if terminated:
        ts = info.get("transition_state") or {}
        p = ts.get("player") if isinstance(ts, dict) else None
        hp = (p or {}).get("hp") if isinstance(p, dict) else None
        if hp is None:
            raw = info.get("raw_obs") or {}
            rp = raw.get("player") if isinstance(raw, dict) else None
            hp = (rp or {}).get("hp") if isinstance(rp, dict) else None
        return "win" if (hp is not None and float(hp) > 0) else "loss"
    return "truncated"


def run_one(bridge, snapshot: dict, model, device, max_steps: int) -> dict:
    obs_encoder = WorldTokenObservationEncoder(use_text=False)
    pool_single = CombatSnapshotPool(rows=[snapshot])
    env = CombatSandboxEnv(
        session_file=None,
        character="ironclad",
        snapshot_pool=pool_single,
        obs_encoder=obs_encoder,
        include_debug_info=True,
        bridge=bridge,
    )
    env = ActionMasker(env, mask_fn)

    t0 = time.time()
    obs, info = env.reset()
    terminated = False
    truncated = False
    reward_sum = 0.0
    steps = 0
    while not (terminated or truncated) and steps < max_steps:
        mask = env.action_masks()
        with torch.no_grad():
            batch = {k: torch.as_tensor(v).unsqueeze(0).to(device) for k, v in obs.items()}
            actions, _, _ = model.policy(batch, deterministic=True, action_masks=mask.reshape(1, -1))
        act_idx = int(actions.item())
        obs, reward, terminated, truncated, info = env.step(act_idx)
        reward_sum += float(reward)
        steps += 1
    elapsed = time.time() - t0

    outcome = classify_outcome(terminated, truncated, info)
    return {
        "outcome": outcome,
        "steps": steps,
        "reward_sum": reward_sum,
        "elapsed_s": elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_dir")
    parser.add_argument("--snapshot-pool", required=True)
    parser.add_argument("--curated-subset", default=None)
    parser.add_argument("--encounters", default="ENCOUNTER.QUEEN_BOSS,ENCOUNTER.LAGAVULIN_MATRIARCH_BOSS,ENCOUNTER.CEREMONIAL_BEAST_BOSS,ENCOUNTER.SOUL_FYSH_BOSS,ENCOUNTER.KNOWLEDGE_DEMON_BOSS",
                        help="CSV encounter_ids")
    parser.add_argument("--snapshots-per-encounter", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--session-file", default=None)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    encounters = [e.strip() for e in args.encounters.split(",") if e.strip()]

    # Gather snapshot list (deterministic — first N from each encounter after filter)
    all_snapshots: list[dict] = []
    for enc in encounters:
        pool = CombatSnapshotPool.from_path(
            args.snapshot_pool,
            curated_subset=args.curated_subset,
            encounter_ids=[enc],
            sample_mode="encounter_balanced",
        )
        rng = np.random.default_rng(0)
        seen: set[str] = set()
        picked = 0
        while picked < args.snapshots_per_encounter:
            snap = pool.sample(rng)
            sid = str(snap.get("sample_id") or id(snap))
            if sid in seen:
                continue
            seen.add(sid)
            all_snapshots.append(snap)
            picked += 1
    print(f"[rank] total snapshots: {len(all_snapshots)} across {len(encounters)} encounters")

    # Load bridges + checkpoint ONCE (reuse across snapshots)
    print("[rank] starting live bridge...")
    live_bridge = BridgeClient(session_path=args.session_file)
    print("[rank] starting sim bridge (v0.103.2)...")
    sim_bridge = HeadlessSimBridgeClient()

    # Use live env for checkpoint load (obs_space must match)
    first_pool = CombatSnapshotPool(rows=[all_snapshots[0]])
    bootstrap_env = ActionMasker(
        CombatSandboxEnv(
            character="ironclad",
            snapshot_pool=first_pool,
            obs_encoder=WorldTokenObservationEncoder(use_text=False),
            include_debug_info=True,
            bridge=live_bridge,
        ),
        mask_fn,
    )
    print(f"[rank] loading checkpoint: {args.checkpoint_dir}")
    model, metadata = load_online_checkpoint(args.checkpoint_dir, bootstrap_env, device=str(device))
    print(f"[rank] obs_api={metadata.get('observation_api_version')} steps={metadata.get('timesteps')}")

    # Run each snapshot on both sides
    results: list[dict[str, Any]] = []
    agreement = 0
    total = 0
    for i, snap in enumerate(all_snapshots):
        enc = snap.get("encounter_id")
        print(f"\n[{i+1}/{len(all_snapshots)}] {enc} sample={snap.get('sample_id')}", flush=True)
        try:
            live_r = run_one(live_bridge, snap, model, device, args.max_steps)
        except Exception as e:
            live_r = {"outcome": "error", "error": str(e)[:200]}
        try:
            sim_r = run_one(sim_bridge, snap, model, device, args.max_steps)
        except Exception as e:
            sim_r = {"outcome": "error", "error": str(e)[:200]}
        matched = live_r.get("outcome") == sim_r.get("outcome")
        if matched:
            agreement += 1
        total += 1
        row = {
            "encounter": enc,
            "sample_id": snap.get("sample_id"),
            "build_id": snap.get("build_id"),
            "live": live_r,
            "sim": sim_r,
            "outcome_match": matched,
        }
        results.append(row)
        print(f"  live: {live_r.get('outcome')} steps={live_r.get('steps')} r={live_r.get('reward_sum'):+.2f} ({live_r.get('elapsed_s', 0):.1f}s)")
        print(f"  sim : {sim_r.get('outcome')} steps={sim_r.get('steps')} r={sim_r.get('reward_sum'):+.2f} ({sim_r.get('elapsed_s', 0):.1f}s)")
        print(f"  outcome_match: {matched}")

    sim_bridge.close()

    # Aggregate
    agreement_rate = agreement / max(total, 1) * 100
    outcomes_live = collections.Counter(r["live"].get("outcome") for r in results)
    outcomes_sim = collections.Counter(r["sim"].get("outcome") for r in results)
    print("\n=== AGGREGATE ===")
    print(f"snapshots: {total}, outcome agreement: {agreement}/{total} = {agreement_rate:.1f}%")
    print(f"live outcomes: {dict(outcomes_live)}")
    print(f"sim  outcomes: {dict(outcomes_sim)}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({
        "checkpoint": args.checkpoint_dir,
        "checkpoint_metadata": {
            "observation_api_version": metadata.get("observation_api_version"),
            "timesteps": metadata.get("timesteps"),
        },
        "snapshot_pool": str(args.snapshot_pool),
        "curated_subset": args.curated_subset,
        "encounters": encounters,
        "snapshots_per_encounter": args.snapshots_per_encounter,
        "max_steps": args.max_steps,
        "total_snapshots": total,
        "outcome_agreement": agreement,
        "agreement_rate_pct": agreement_rate,
        "live_outcome_distribution": dict(outcomes_live),
        "sim_outcome_distribution": dict(outcomes_sim),
        "per_snapshot": results,
    }, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {output_path}")


if __name__ == "__main__":
    main()
