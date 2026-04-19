"""P1 diagnostic: is `confirm` reachable from the card_selection phase, and
are there obs fields that tell the policy "I already selected something"?

If the answer to either is NO, adding action-history tokens won't fix the
42% card_selection loop problem — we'd need a translator/schema fix first.

Strategy:
  1. Drive a random-ish policy until we hit `phase == "card_selection"`.
  2. Dump the ENTIRE legal_actions list (action_id + kind + all keys).
  3. Dump obs["card_selection"] and obs["decision"] if present.
  4. Take a selection action, then reprint: did any obs field flip?
     Is there now a confirm action?

We run multiple attempts because not every run triggers card_selection —
this phase fires on specific cards (Scry, Discovery), specific events
(Neow-ish starter choice), or specific room types (campfire smith / card
reward selection).
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sts2_env.env_v2 import SlayTheSpire2EnvV2
from sts2_env.headless_sim_bridge_client import HeadlessSimBridgeClient


def dump_legal_actions(legal: list[dict]) -> str:
    lines = []
    for i, a in enumerate(legal):
        keys = sorted(k for k in a.keys() if k != "_sim_raw")
        summary = {k: a.get(k) for k in ("action_id", "kind") if k in a}
        extras = {k: a.get(k) for k in keys if k not in ("action_id", "kind") and a.get(k) is not None}
        lines.append(f"  [{i}] {summary}  extras={extras}")
    return "\n".join(lines)


def dump_card_selection_obs(obs: dict) -> str:
    cs = obs.get("card_selection") or {}
    decision = obs.get("decision") or {}
    run = obs.get("run") or {}
    keys_of_interest = {
        "phase": obs.get("phase"),
        "screen": obs.get("screen"),
        "run.floor": run.get("floor"),
        "card_selection": cs,
        "decision": decision,
    }
    return json.dumps(keys_of_interest, indent=2, default=str, ensure_ascii=False)


def main() -> None:
    print("[probe] P1 card_selection schema diagnostic")
    print("=" * 70)

    hits = 0
    max_attempts = 30
    rng = random.Random(42)

    for attempt in range(max_attempts):
        bridge = HeadlessSimBridgeClient()
        env = SlayTheSpire2EnvV2(
            bridge=bridge, character="ironclad", stuck_watchdog_steps=0
        )
        try:
            obs, info = env.reset()
            for t in range(400):
                legal = env._legal_actions
                if not legal:
                    break
                raw = env._last_obs_raw or {}
                phase = str(raw.get("phase") or "")
                if phase == "card_selection":
                    hits += 1
                    print(f"\n=== HIT #{hits} (attempt {attempt}, t={t}) ===")
                    print(dump_card_selection_obs(raw))
                    print(f"\nlegal_actions ({len(legal)} entries):")
                    print(dump_legal_actions(legal))

                    # Identify "confirm" and "select" candidates by kind/action_id
                    confirm_candidates = [
                        (i, a) for i, a in enumerate(legal)
                        if "confirm" in str(a.get("action_id") or "").lower()
                        or "confirm" in str(a.get("kind") or "").lower()
                    ]
                    select_candidates = [
                        (i, a) for i, a in enumerate(legal)
                        if str(a.get("kind") or "") in {"card_selection", "select_card"}
                        or "select_card" in str(a.get("action_id") or "")
                    ]
                    skip_candidates = [
                        (i, a) for i, a in enumerate(legal)
                        if "skip" in str(a.get("action_id") or "").lower()
                        or str(a.get("kind") or "") == "skip"
                    ]
                    print(f"\nAction classification:")
                    print(f"  confirm actions: {len(confirm_candidates)}  {[a.get('action_id') for _, a in confirm_candidates[:3]]}")
                    print(f"  select actions:  {len(select_candidates)}  {[a.get('action_id') for _, a in select_candidates[:3]]}")
                    print(f"  skip actions:    {len(skip_candidates)}  {[a.get('action_id') for _, a in skip_candidates[:3]]}")

                    # Take the first SELECT action and see what changes
                    if select_candidates:
                        sel_idx, sel_action = select_candidates[0]
                        print(f"\n>>> Taking SELECT action [{sel_idx}]: {sel_action.get('action_id')}")
                        _, reward, term, trunc, _ = env.step(sel_idx)
                        raw_after = env._last_obs_raw or {}
                        phase_after = str(raw_after.get("phase") or "")
                        print(f"\nAfter select: phase={phase_after}, reward={reward:+.3f}, term={term}, trunc={trunc}")
                        print(dump_card_selection_obs(raw_after))
                        print(f"\nlegal_actions after select ({len(env._legal_actions)} entries):")
                        print(dump_legal_actions(env._legal_actions))

                        confirm_after = [
                            (i, a) for i, a in enumerate(env._legal_actions)
                            if "confirm" in str(a.get("action_id") or "").lower()
                            or "confirm" in str(a.get("kind") or "").lower()
                        ]
                        print(f"\nconfirm actions AFTER select: {len(confirm_after)}  {[a.get('action_id') for _, a in confirm_after[:3]]}")
                    print("=" * 70)
                    if hits >= 3:
                        return
                    # Reset for another run
                    break

                # Random legal action
                idx = rng.choice(range(len(legal)))
                _, _, term, trunc, _ = env.step(idx)
                if term or trunc:
                    break
        finally:
            try:
                env.close()
            except Exception:
                pass

    print(f"\n[probe] total hits: {hits} across {max_attempts} attempts")


if __name__ == "__main__":
    main()
