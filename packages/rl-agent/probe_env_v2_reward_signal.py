"""Run SlayTheSpire2EnvV2 with sim backend, print the actual reward each step.

Complements probe_sim_reward_signal.py (which went direct-to-bridge and confirmed
HP deltas fire). This probe goes through the full env_v2.step() pipeline to
reveal whether env_v2's reward plumbing is computing per-step reward or zeroing
them out somehow.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sts2_env.env_v2 import SlayTheSpire2EnvV2
from sts2_env.headless_sim_bridge_client import HeadlessSimBridgeClient


def summarize_raw(obs_raw: dict) -> str:
    combat = obs_raw.get("combat") or {}
    enemies = combat.get("enemies") or []
    player = obs_raw.get("player") or {}
    e_hps = [int(e.get("hp", 0) or 0) for e in enemies]
    return f"p_hp={player.get('hp','?')} e_hps={e_hps}"


def main() -> None:
    print("[probe] booting sim + env_v2 ...")
    bridge = HeadlessSimBridgeClient()
    env = SlayTheSpire2EnvV2(bridge=bridge, character="ironclad")
    obs, info = env.reset()
    print("[probe] reset ok; initial raw:", summarize_raw(env._last_obs_raw or {}))

    total_reward = 0.0
    total_nonzero_reward_steps = 0
    for t in range(120):
        legal = env._legal_actions
        if not legal:
            print(f"[probe] step {t}: no legal actions, breaking")
            break
        # Prefer play_card
        chosen_idx = 0
        for i, a in enumerate(legal):
            if str(a.get("kind") or "") == "play_card":
                chosen_idx = i
                break

        raw_before = dict(env._last_obs_raw or {})  # shallow copy of ref
        combat_before = (env._last_obs_raw or {}).get("combat") or {}
        enemies_before = [int(e.get("hp", 0) or 0) for e in (combat_before.get("enemies") or [])]

        new_obs, reward, terminated, truncated, info = env.step(chosen_idx)

        combat_after = (env._last_obs_raw or {}).get("combat") or {}
        enemies_after = [int(e.get("hp", 0) or 0) for e in (combat_after.get("enemies") or [])]

        player_before = (raw_before or {}).get("player") or {}
        player_after = (env._last_obs_raw or {}).get("player") or {}
        hp_before = int(player_before.get("hp") or 0)
        hp_after = int(player_after.get("hp") or 0)

        total_reward += reward
        if abs(reward) > 1e-6:
            total_nonzero_reward_steps += 1
        act_id = legal[chosen_idx].get("action_id")
        print(
            f"[t={t:3d}] act={act_id[:36]:36s} "
            f"enemies {enemies_before}->{enemies_after} "
            f"hp {hp_before}->{hp_after} "
            f"reward={reward:+.3f} done={terminated} trunc={truncated}"
        )
        if terminated or truncated:
            print(f"[probe] episode done at step {t}, total reward={total_reward:+.3f}")
            break

    print("=" * 60)
    print(f"total_reward = {total_reward:+.3f}")
    print(f"nonzero_reward_steps = {total_nonzero_reward_steps}")


if __name__ == "__main__":
    main()
