"""Exercise the phase-stuck watchdog against a sim-backed SlayTheSpire2EnvV2.

Force the policy into a no-progress loop by repeatedly picking an action that
the sim accepts but which (by policy side-effect) keeps the world state
identical. We approximate this by running combat with a policy that *always
picks the first legal action* once we're past the initial reward phase — the
first index tends to be a 0-cost draw with no target, producing a fingerprint
stall on several steps per combat. If the watchdog works, an episode that
*would* run too long eventually truncates with stuck_* metadata.

This is a loose approximation of what the neural policy does during
early training when its argmax collapses onto a single action idx.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sts2_env.env_v2 import SlayTheSpire2EnvV2
from sts2_env.headless_sim_bridge_client import HeadlessSimBridgeClient


def main() -> None:
    bridge = HeadlessSimBridgeClient()
    # Lower the threshold so the probe fires quickly.
    env = SlayTheSpire2EnvV2(bridge=bridge, character="ironclad", stuck_watchdog_steps=60)
    obs, _ = env.reset()

    fp_prev = None
    fp_streak = 0
    for t in range(600):
        legal = env._legal_actions
        if not legal:
            print(f"[t={t}] no legal actions, breaking")
            break
        # Pathological policy: always pick index 0.
        idx = 0
        _, reward, terminated, truncated, info = env.step(idx)

        fp = env._progress_fingerprint()
        if fp == fp_prev:
            fp_streak += 1
        else:
            fp_streak = 1
            fp_prev = fp

        if t % 20 == 0 or truncated or terminated:
            bridge_info = info.get("bridge_info") or {}
            trunc_reason = bridge_info.get("truncation_reason")
            print(
                f"t={t:3d} r={reward:+.3f} term={terminated} trunc={truncated} "
                f"fp_streak={fp_streak:3d} fp={fp} reason={trunc_reason}"
            )
        if terminated or truncated:
            bridge_info = info.get("bridge_info") or {}
            print()
            print("=== FINAL ===")
            print(f"terminated={terminated} truncated={truncated}")
            print(f"truncation_reason={bridge_info.get('truncation_reason')}")
            if bridge_info.get("truncation_reason") == "phase_stuck_watchdog":
                print("WATCHDOG FIRED (ok)")
                print(f"  stuck_phase = {bridge_info.get('stuck_phase')}")
                print(f"  stuck_floor = {bridge_info.get('stuck_floor')}")
                print(f"  stuck_steps = {bridge_info.get('stuck_steps')}")
            break
    else:
        print(f"[probe] reached step cap without terminating at t={t}")


if __name__ == "__main__":
    main()
