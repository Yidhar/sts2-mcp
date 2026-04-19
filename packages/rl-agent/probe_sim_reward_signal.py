"""Diagnostic: verify enemy/player HP deltas actually occur between consecutive
sim steps during combat.

Hypothesis under test: sim long-train shows Monitor cumulative r=-3.4 for EVERY
episode regardless of length 17-5177 steps. value_loss ~ 1e-8 and aux_objective
~ 1e-6. Combined, this strongly suggests per-step reward ~ 0 in env_v2.py's
_enemy_hp_delta_reward / _player_hp_delta_reward paths.

What we check:
  1. Start sim, drive it into combat, play a few cards.
  2. For each transition, print (before_enemy_total_hp, after_enemy_total_hp,
     delta). Non-zero deltas prove sim damage pipeline works.
  3. Also compute what env_v2 would produce as reward for that step.

Usage:
  python probe_sim_reward_signal.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Insert this dir so sts2_env is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sts2_env.headless_sim_bridge_client import HeadlessSimBridgeClient
from sts2_env.reward_constants import (
    ENEMY_HP_DELTA_REWARD_SCALE,
    PLAYER_HP_LOSS_REWARD_SCALE,
    ENEMY_HP_SENTINEL_THRESHOLD,
)


def enemy_total_hp(obs: dict) -> float:
    combat = obs.get("combat") or {}
    enemies = combat.get("enemies") or []
    total = 0.0
    for e in enemies:
        hp = float(e.get("hp", 0) or 0)
        if hp > ENEMY_HP_SENTINEL_THRESHOLD:
            continue
        total += hp
    return total


def player_hp(obs: dict) -> float:
    player = obs.get("player") or {}
    return float(player.get("hp") or 0)


def summarize_combat(obs: dict) -> str:
    combat = obs.get("combat") or {}
    enemies = combat.get("enemies") or []
    player = obs.get("player") or {}
    e_hps = [int(e.get("hp", 0) or 0) for e in enemies]
    return (
        f"phase={obs.get('phase','?')} "
        f"state_type={obs.get('state_type') or '?'} "
        f"p_hp={player.get('hp','?')}/{player.get('max_hp','?')} "
        f"energy={combat.get('energy','?')}/{combat.get('max_energy','?')} "
        f"block={combat.get('block','?')} "
        f"enemies={e_hps} "
        f"actions={len(obs.get('available_actions') or [])}"
    )


def main() -> None:
    print("[probe] Starting sim...")
    client = HeadlessSimBridgeClient()
    try:
        r = client.reset(character="ironclad")
    except TypeError:
        r = client.reset()
    obs = r["obs"]
    print("[probe] reset done:", summarize_combat(obs))

    step_idx = 0
    deltas_seen = []
    max_steps = 120
    while step_idx < max_steps:
        legal = r.get("legal_actions") or obs.get("available_actions") or []
        if not legal:
            print(f"[probe] step {step_idx}: no legal actions, breaking")
            break

        # Pick first legal action. Prefer play_card if available.
        chosen_idx = 0
        for i, a in enumerate(legal):
            aid = str(a.get("action_id") or "")
            kind = str(a.get("kind") or "")
            if kind == "play_card" or aid.startswith("sim:play_card"):
                chosen_idx = i
                break

        before_enemy = enemy_total_hp(obs)
        before_player = player_hp(obs)
        action_desc = f"{legal[chosen_idx].get('action_id')}/{legal[chosen_idx].get('kind')}"

        r = client.step(episode_id=r.get("episode_id", ""), action_index=chosen_idx)
        obs = r["obs"]

        after_enemy = enemy_total_hp(obs)
        after_player = player_hp(obs)

        e_delta = before_enemy - after_enemy  # positive = damage dealt
        p_delta = after_player - before_player  # negative = hp lost
        e_reward = e_delta * ENEMY_HP_DELTA_REWARD_SCALE
        p_reward = p_delta * PLAYER_HP_LOSS_REWARD_SCALE
        bridge_reward = float(r.get("reward") or 0.0)

        print(
            f"[step {step_idx:3d}] act={action_desc:40s} "
            f"bridge_r={bridge_reward:+.3f} "
            f"e_hp {before_enemy:.0f}->{after_enemy:.0f} (d={e_delta:+.0f}, r={e_reward:+.3f}) "
            f"p_hp {before_player:.0f}->{after_player:.0f} (d={p_delta:+.0f}, r={p_reward:+.3f}) "
            f"done={r.get('done')}"
        )
        deltas_seen.append((e_delta, p_delta, bridge_reward))
        step_idx += 1
        if r.get("done"):
            print(f"[probe] episode ended at step {step_idx}")
            break

    print()
    print("=" * 60)
    total_enemy_deltas = sum(d[0] for d in deltas_seen)
    total_player_deltas = sum(d[1] for d in deltas_seen)
    nonzero_enemy = sum(1 for d in deltas_seen if abs(d[0]) > 1e-6)
    nonzero_player = sum(1 for d in deltas_seen if abs(d[1]) > 1e-6)
    total_bridge = sum(d[2] for d in deltas_seen)
    print(f"[summary] steps={len(deltas_seen)}")
    print(f"[summary] nonzero enemy_hp_delta steps: {nonzero_enemy}")
    print(f"[summary] nonzero player_hp_delta steps: {nonzero_player}")
    print(f"[summary] sum(enemy_hp_delta): {total_enemy_deltas:+.1f}")
    print(f"[summary] sum(player_hp_delta): {total_player_deltas:+.1f}")
    print(f"[summary] sum(bridge_reward): {total_bridge:+.3f}")
    print()
    if nonzero_enemy == 0 and nonzero_player == 0:
        print("!!! RED FLAG: ZERO enemy and player HP deltas across ALL steps !!!")
        print("    This confirms env_v2 hp-delta reward path sees no change,")
        print("    which explains value_loss~0 and aux_objective~0 in training.")
    elif nonzero_enemy > 0 or nonzero_player > 0:
        print(">>> HP deltas DO fire. Issue must be elsewhere in reward plumbing.")


if __name__ == "__main__":
    main()
