"""Dump per-step RPC traces on sim and live for the SAME snapshot, SAME checkpoint,
deterministic policy. Goal: find where the two bridges diverge.

Writes:
  <outdir>/sim_trace.jsonl   - per-step row from sim bridge
  <outdir>/live_trace.jsonl  - per-step row from live bridge
  <outdir>/summary.json      - divergence report
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sb3_contrib.common.wrappers import ActionMasker

from combat_snapshot_dataset import CombatSnapshotPool, snapshot_row_to_reset_kwargs
from sts2_env.bridge_client import BridgeClient
from sts2_env.checkpoint import load_online_checkpoint
from sts2_env.combat_env import CombatSandboxEnv
from sts2_env.headless_sim_bridge_client import HeadlessSimBridgeClient
from sts2_env.observation_v3 import WorldTokenObservationEncoder


def mask_fn(env):
    return env.unwrapped.action_masks()


def compact_step(
    t: int,
    raw_obs: dict[str, Any],
    legal: list[dict[str, Any]],
    chosen_idx: int,
    reward: float,
    terminated: bool,
    truncated: bool,
    info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    player = raw_obs.get("player") or {}
    combat = raw_obs.get("combat") or {}
    enemies = combat.get("enemies") or []
    enemies_out = []
    for e in enemies:
        if not isinstance(e, dict):
            continue
        powers = e.get("powers") or []
        enemies_out.append({
            "id": e.get("id") or e.get("name"),
            "hp": e.get("hp") or e.get("current_hp"),
            "max_hp": e.get("max_hp"),
            "block": e.get("block"),
            "alive": float(e.get("hp") or e.get("current_hp") or 0) > 0,
            "intent_type": (e.get("intent") or {}).get("type"),
            "intent_dmg": (e.get("intent") or {}).get("total_damage"),
            "power_titles": [p.get("title") for p in powers if isinstance(p, dict)],
        })
    chosen = legal[chosen_idx] if 0 <= chosen_idx < len(legal) else {}
    return {
        "t": t,
        "phase": raw_obs.get("phase"),
        "screen": raw_obs.get("screen"),
        "round": combat.get("round"),
        "player_hp": player.get("hp"),
        "player_max_hp": player.get("max_hp"),
        "player_block": combat.get("block"),
        "player_energy": combat.get("energy"),
        "player_power_titles": [p.get("title") for p in (player.get("powers") or []) if isinstance(p, dict)],
        "enemies": enemies_out,
        "n_legal": len(legal),
        "legal_kinds": [l.get("kind") for l in legal[:10]],
        "chosen_idx": chosen_idx,
        "chosen_kind": chosen.get("kind"),
        "chosen_card_id": (chosen.get("card") or {}).get("id") if isinstance(chosen.get("card"), dict) else chosen.get("card_id"),
        "chosen_label": chosen.get("label") or chosen.get("display_name"),
        "reward_step": float(reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "combat_outcome": (info or {}).get("combat_outcome"),
    }


def run_one_episode(
    env,
    model,
    device,
    deterministic: bool,
    max_steps: int,
    out_file,
    label: str,
) -> dict[str, Any]:
    obs, info = env.reset()
    raw_obs = info.get("raw_obs") or {}
    t = 0
    reward_sum = 0.0
    terminated = False
    truncated = False
    last_chosen = -1
    while not (terminated or truncated) and t < max_steps:
        mask = env.action_masks()
        legal = list(env.unwrapped._legal_actions)
        with torch.no_grad():
            batch = {k: torch.as_tensor(v).unsqueeze(0).to(device) for k, v in obs.items()}
            actions, _, _ = model.policy(batch, deterministic=deterministic, action_masks=mask.reshape(1, -1))
        chosen = int(actions.item())
        # record PRE-step snapshot of state (what policy saw when choosing)
        pre_row = compact_step(t, raw_obs, legal, chosen, 0.0, False, False, info)
        pre_row["phase_slot"] = "pre"
        obs, reward, terminated, truncated, info = env.step(chosen)
        post_row = compact_step(t, info.get("raw_obs") or {}, list(env.unwrapped._legal_actions),
                                chosen, reward, terminated, truncated, info)
        post_row["phase_slot"] = "post"
        out_file.write(json.dumps(pre_row, ensure_ascii=False) + "\n")
        out_file.write(json.dumps(post_row, ensure_ascii=False) + "\n")
        out_file.flush()
        raw_obs = info.get("raw_obs") or {}
        reward_sum += float(reward)
        last_chosen = chosen
        t += 1
    final_combat = (raw_obs.get("combat") or {})
    alive = sum(1 for e in (final_combat.get("enemies") or []) if isinstance(e, dict) and float(e.get("hp") or 0) > 0)
    return {
        "label": label,
        "steps": t,
        "reward_sum": reward_sum,
        "terminated": terminated,
        "truncated": truncated,
        "final_player_hp": (raw_obs.get("player") or {}).get("hp"),
        "final_alive_enemies": alive,
        "final_phase": raw_obs.get("phase"),
        "final_screen": raw_obs.get("screen"),
        "combat_outcome": (info or {}).get("combat_outcome"),
    }


def build_env(bridge, pool, session_file, character):
    obs_encoder = WorldTokenObservationEncoder(use_text=False)
    env = CombatSandboxEnv(
        session_file=session_file,
        character=character,
        snapshot_pool=pool,
        obs_encoder=obs_encoder,
        include_debug_info=True,
        bridge=bridge,
    )
    return ActionMasker(env, mask_fn)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_dir")
    parser.add_argument("--snapshot-pool", required=True)
    parser.add_argument("--curated-subset", default=None)
    parser.add_argument("--encounter-id", default="ENCOUNTER.QUEEN_BOSS")
    parser.add_argument("--snapshot-seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--character", default="ironclad")
    parser.add_argument("--session-file", default=None)
    parser.add_argument("--sim-exe-path", default=None)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Build identical snapshot pool on both sides using same seed
    pool = CombatSnapshotPool.from_path(
        args.snapshot_pool,
        curated_subset=args.curated_subset,
        encounter_ids=[args.encounter_id],
        sample_mode="encounter_balanced",
    )
    print(f"[probe] pool size={len(pool)} for {args.encounter_id}")
    # Deterministic snapshot selection — use same RNG state for both sides
    rng = np.random.default_rng(args.snapshot_seed)
    snapshot = pool.sample(rng)
    snapshot_id = snapshot.get("sample_id") or f"seed{args.snapshot_seed}"
    print(f"[probe] using snapshot: build_id={snapshot.get('build_id')} "
          f"sample_id={snapshot_id}")

    # === LIVE side ===
    print("\n=== LIVE run ===")
    live_bridge = BridgeClient(session_path=args.session_file)
    # Re-build pool with a single-row list so env.reset() always picks THIS snapshot
    pool_single = CombatSnapshotPool(rows=[snapshot])
    live_env = build_env(live_bridge, pool_single, args.session_file, args.character)
    print("[probe] loading checkpoint for live...")
    live_model, metadata = load_online_checkpoint(args.checkpoint_dir, live_env, device=str(device))
    print(f"[probe] obs_api={metadata.get('observation_api_version')} timesteps={metadata.get('timesteps')}")
    with (outdir / "live_trace.jsonl").open("w", encoding="utf-8") as fh:
        live_summary = run_one_episode(live_env, live_model, device, True, args.max_steps, fh, "live")
    print(f"[live] {live_summary}")

    # === SIM side ===
    print("\n=== SIM run ===")
    sim_bridge = HeadlessSimBridgeClient(exe_path=args.sim_exe_path)
    pool_single2 = CombatSnapshotPool(rows=[snapshot])
    sim_env = build_env(sim_bridge, pool_single2, args.session_file, args.character)
    print("[probe] loading checkpoint for sim...")
    sim_model, _ = load_online_checkpoint(args.checkpoint_dir, sim_env, device=str(device))
    with (outdir / "sim_trace.jsonl").open("w", encoding="utf-8") as fh:
        sim_summary = run_one_episode(sim_env, sim_model, device, True, args.max_steps, fh, "sim")
    print(f"[sim] {sim_summary}")

    sim_bridge.close()

    # Divergence summary
    report = {
        "encounter": args.encounter_id,
        "snapshot": {
            "sample_id": snapshot.get("sample_id"),
            "build_id": snapshot.get("build_id"),
            "character": snapshot.get("character"),
            "max_hp": snapshot.get("snapshot_max_hp"),
            "max_energy": snapshot.get("snapshot_max_energy"),
            "deck_len": len(snapshot.get("deck_card_ids") or []),
            "relic_count": len((snapshot.get("relic_ids_before") or [])),
        },
        "live": live_summary,
        "sim": sim_summary,
        "diff": {
            "steps_delta": sim_summary["steps"] - live_summary["steps"],
            "reward_delta": sim_summary["reward_sum"] - live_summary["reward_sum"],
            "terminated_same": sim_summary["terminated"] == live_summary["terminated"],
            "final_hp_delta": (sim_summary["final_player_hp"] or 0) - (live_summary["final_player_hp"] or 0),
            "outcome_mismatch": sim_summary["combat_outcome"] != live_summary["combat_outcome"],
        },
    }
    (outdir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print("\n=== DIVERGENCE ===")
    print(json.dumps(report["diff"], ensure_ascii=False, indent=2))
    print(f"\ntraces: {outdir}/sim_trace.jsonl, {outdir}/live_trace.jsonl")
    print(f"summary: {outdir}/summary.json")


if __name__ == "__main__":
    main()
