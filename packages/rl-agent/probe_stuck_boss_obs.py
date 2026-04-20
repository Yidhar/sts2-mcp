"""Probe live boss trajectories to diagnose why 5/7 P-0'' losers are stuck at 0% WR.

Compares stuck losers (INSATIABLE, SOUL_FYSH, WATERFALL_GIANT, DOORMAKER,
TEST_SUBJECT) against controls where this checkpoint already wins
(QUEEN_BOSS @100%, KAISER_CRAB_BOSS @87%).

Per-step dump: round, phase, player hp/block/energy, player.powers,
enemies[{id, hp, block, intent, powers}], chosen action label + kind,
terminal reason + reward.

Usage (needs a live game session, NOT --use-sim):
  python probe_stuck_boss_obs.py checkpoints_attention/sim_p0pp_losers_*/step_001320960 \\
      --snapshot-pool /mnt/e/game/project/sts2_mcp/datasets/curated_combat_ironclad_mixed_provenance \\
      --curated-subset bootstrap_human_plus_local_all_roomwin_only_minus_combat_reset_failures \\
      --episodes-per-encounter 2 \\
      --device cuda --no-text --deterministic \\
      --output-dir analysis/probes/stuck_boss_20260420
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
from sts2_env.checkpoint import load_online_checkpoint
from sts2_env.combat_env import CombatSandboxEnv
from sts2_env.observation_v3 import WorldTokenObservationEncoder


STUCK = [
    "ENCOUNTER.THE_INSATIABLE_BOSS",
    "ENCOUNTER.SOUL_FYSH_BOSS",
    "ENCOUNTER.WATERFALL_GIANT_BOSS",
    "ENCOUNTER.DOORMAKER_BOSS",
    "ENCOUNTER.TEST_SUBJECT_BOSS",
]
CONTROLS = [
    "ENCOUNTER.QUEEN_BOSS",
    "ENCOUNTER.KAISER_CRAB_BOSS",
]


def mask_fn(env):
    return env.unwrapped.action_masks()


def _extract_powers(entity: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(entity, dict):
        return []
    powers = entity.get("powers") or []
    out = []
    for p in powers:
        if not isinstance(p, dict):
            continue
        out.append({
            "title": str(p.get("title") or p.get("id") or "").strip(),
            "amount": p.get("amount"),
            "duration": p.get("duration"),
        })
    return out


def _compact_step(
    step_idx: int,
    raw_obs: dict[str, Any],
    legal_actions: list[dict[str, Any]],
    chosen_idx: int,
    reward_step: float,
) -> dict[str, Any]:
    player = raw_obs.get("player") or {}
    combat = raw_obs.get("combat") or {}
    enemies = combat.get("enemies") or []
    enemy_rows = []
    for e in enemies:
        if not isinstance(e, dict):
            continue
        intent = e.get("intent") or {}
        enemy_rows.append({
            "id": e.get("id") or e.get("name"),
            "hp": e.get("hp") or e.get("current_hp"),
            "max_hp": e.get("max_hp"),
            "block": e.get("block"),
            "alive": (float(e.get("hp") or e.get("current_hp") or 0) > 0),
            "intent": {
                "type": intent.get("type"),
                "total_damage": intent.get("total_damage"),
                "damage_per_hit": intent.get("damage_per_hit"),
                "repeats": intent.get("repeats"),
            },
            "powers": _extract_powers(e),
        })
    chosen = legal_actions[chosen_idx] if 0 <= chosen_idx < len(legal_actions) else {}
    return {
        "t": step_idx,
        "round": combat.get("round"),
        "phase": raw_obs.get("phase"),
        "screen": raw_obs.get("screen"),
        "player": {
            "hp": player.get("hp"),
            "max_hp": player.get("max_hp"),
            "block": combat.get("block"),
            "energy": combat.get("energy"),
            "powers": _extract_powers(player),
        },
        "enemies": enemy_rows,
        "chosen": {
            "idx": chosen_idx,
            "kind": chosen.get("kind"),
            "label": chosen.get("label") or chosen.get("display_name"),
            "card_id": (chosen.get("card") or {}).get("id") if isinstance(chosen.get("card"), dict) else chosen.get("card_id"),
            "target_id": chosen.get("target_id"),
            "target": chosen.get("target"),
        },
        "reward_step": float(reward_step),
        "n_legal": len(legal_actions),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_dir")
    parser.add_argument("--snapshot-pool", required=True)
    parser.add_argument("--curated-subset", default=None)
    parser.add_argument("--build-id", default=None)
    parser.add_argument("--encounter-ids", default=None,
                        help="CSV; overrides the default STUCK+CONTROLS list")
    parser.add_argument("--episodes-per-encounter", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--text-device", default="auto")
    parser.add_argument("--no-text", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--session-file", default=None)
    parser.add_argument("--character", default="ironclad")
    parser.add_argument("--max-steps-per-ep", type=int, default=500)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    encounters = [s.strip() for s in (args.encounter_ids or "").split(",") if s.strip()] or (STUCK + CONTROLS)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    text_device = args.text_device
    if text_device == "auto":
        text_device = "cuda" if torch.cuda.is_available() else "cpu"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[probe] output_dir={output_dir}")

    obs_encoder = WorldTokenObservationEncoder(
        use_text=(not args.no_text), text_device=text_device,
    )

    # Load checkpoint once — use a throwaway env whose pool matches the first
    # encounter just to satisfy the observation_space pre-check.
    first_pool = CombatSnapshotPool.from_path(
        args.snapshot_pool,
        curated_subset=args.curated_subset,
        build_id=args.build_id,
        encounter_ids=[encounters[0]],
        sample_mode="encounter_balanced",
    )
    bootstrap_env = ActionMasker(
        CombatSandboxEnv(
            session_file=args.session_file,
            character=args.character,
            snapshot_pool=first_pool,
            obs_encoder=obs_encoder,
            include_debug_info=True,
        ),
        mask_fn,
    )
    print(f"[probe] loading checkpoint: {args.checkpoint_dir}")
    model, metadata = load_online_checkpoint(
        args.checkpoint_dir, bootstrap_env, device=str(device),
    )
    print(f"[probe] obs_api={metadata.get('observation_api_version')} timesteps={metadata.get('timesteps')}")

    summary: dict[str, dict[str, Any]] = {}
    for enc_id in encounters:
        kind = "stuck" if enc_id in STUCK else ("control" if enc_id in CONTROLS else "other")
        print(f"\n[probe] === {enc_id} ({kind}) ===")
        pool = CombatSnapshotPool.from_path(
            args.snapshot_pool,
            curated_subset=args.curated_subset,
            build_id=args.build_id,
            encounter_ids=[enc_id],
            sample_mode="encounter_balanced",
        )
        print(f"[probe] pool={len(pool)} rows")
        env = ActionMasker(
            CombatSandboxEnv(
                session_file=args.session_file,
                character=args.character,
                snapshot_pool=pool,
                obs_encoder=obs_encoder,
                include_debug_info=True,
            ),
            mask_fn,
        )

        enc_summary = {
            "encounter_id": enc_id,
            "kind": kind,
            "eps": [],
        }
        out_path = output_dir / f"{enc_id.replace('ENCOUNTER.', '').lower()}.jsonl"
        with out_path.open("w", encoding="utf-8") as fh:
            for ep in range(args.episodes_per_encounter):
                obs, info = env.reset()
                raw_obs = info.get("raw_obs") or {}
                steps: list[dict[str, Any]] = []
                terminated = False
                truncated = False
                reward_sum = 0.0
                t = 0
                first_round_enemy_powers: list[dict[str, Any]] = []
                # snapshot enemy powers at combat start
                for e in (raw_obs.get("combat") or {}).get("enemies") or []:
                    if isinstance(e, dict):
                        first_round_enemy_powers.append({
                            "id": e.get("id") or e.get("name"),
                            "max_hp": e.get("max_hp"),
                            "powers": _extract_powers(e),
                        })
                while not (terminated or truncated) and t < args.max_steps_per_ep:
                    mask = env.action_masks()
                    legal_actions = list(env.unwrapped._legal_actions)
                    with torch.no_grad():
                        batch = {k: torch.as_tensor(v).unsqueeze(0).to(device) for k, v in obs.items()}
                        mb = mask.reshape(1, -1)
                        if args.deterministic:
                            actions, _, _ = model.policy(batch, deterministic=True, action_masks=mb)
                        else:
                            actions, _, _ = model.policy(batch, action_masks=mb)
                    chosen_idx = int(actions.item())
                    step_dump = _compact_step(t, raw_obs, legal_actions, chosen_idx, 0.0)
                    obs, reward, terminated, truncated, info = env.step(chosen_idx)
                    raw_obs = info.get("raw_obs") or {}
                    step_dump["reward_step"] = float(reward)
                    steps.append(step_dump)
                    reward_sum += float(reward)
                    t += 1
                # final state
                final_raw = raw_obs
                enemies = (final_raw.get("combat") or {}).get("enemies") or []
                alive_enemies = sum(1 for e in enemies if isinstance(e, dict) and float(e.get("hp") or e.get("current_hp") or 0) > 0)
                player = final_raw.get("player") or {}
                final_hp = player.get("hp")
                outcome = (
                    "win" if terminated and not truncated and (final_hp is None or float(final_hp) > 0) and alive_enemies == 0
                    else "truncated" if truncated
                    else "loss"
                )
                ep_row = {
                    "ep": ep,
                    "encounter_id": enc_id,
                    "kind": kind,
                    "outcome": outcome,
                    "steps": steps,
                    "steps_count": len(steps),
                    "reward_sum": reward_sum,
                    "final_hp": final_hp,
                    "alive_enemies_at_end": alive_enemies,
                    "first_round_enemy_snapshot": first_round_enemy_powers,
                }
                fh.write(json.dumps(ep_row, ensure_ascii=False) + "\n")
                enc_summary["eps"].append({
                    "ep": ep,
                    "outcome": outcome,
                    "steps": len(steps),
                    "reward": reward_sum,
                    "final_hp": final_hp,
                })
                print(f"  ep {ep}: {outcome:>4s}  steps={len(steps):>3d}  reward={reward_sum:+.2f}  final_hp={final_hp}")
        summary[enc_id] = enc_summary

    # Summary
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"\n[probe] wrote {output_dir}/summary.json")
    print(f"[probe] per-encounter traces in {output_dir}/*.jsonl")


if __name__ == "__main__":
    main()
