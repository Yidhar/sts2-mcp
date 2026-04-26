"""Deeper analysis: split end_turn dumps by energy / playable-cards-in-hand
to find cases where playable cards exist but were not legal."""
import json
import os
import sys

log_dir = sys.argv[1] if len(sys.argv) > 1 else None
if log_dir is None:
    runs = sorted([d for d in os.listdir("logs_muzero")
                   if d.startswith("muzero_token_memory_combat_sandbox_2envs_")])
    log_dir = f"logs_muzero/{runs[-1]}"
path = f"{log_dir}/diagnostics/end_turn_selections.jsonl"
print(f"reading: {path}\n")

entries = []
with open(path) as f:
    for line in f:
        line = line.strip()
        if line:
            entries.append(json.loads(line))

from collections import Counter

# bucket by energy / hand size / legal_actions count
bucket_counts = Counter()
suspicious_entries = []

for e in entries:
    energy_raw = (e.get("energy") or {}).get("raw") or 0
    hand = e.get("raw_hand_cards") or []
    hand_size = len(hand)
    legal_mask_gt0_non_end = sum(
        1 for la in (e.get("legal_actions") or [])
        if la.get("mask", 0) > 0 and la.get("family") != "end_turn"
    )
    # Suspicious: hand size > 0 AND energy >= 1 AND legal_mask_gt0_non_end == 0
    tag = f"energy={int(energy_raw)}, hand={hand_size}, legal_non_end={legal_mask_gt0_non_end}"
    bucket_counts[tag] += 1
    if hand_size > 0 and energy_raw >= 1 and legal_mask_gt0_non_end == 0:
        suspicious_entries.append(e)
    elif legal_mask_gt0_non_end >= 1 and not e.get("wasteful_end_turn"):
        # there ARE playable actions and detector said not wasteful — check further
        ctx = e.get("wasteful_context") or {}
        if ctx.get("positive_progress_count", 0) == 0:
            suspicious_entries.append(e)

print(f"=== bucket distribution (top 15) ===")
for tag, count in bucket_counts.most_common(15):
    print(f"  {count:3d}x  {tag}")

print(f"\n=== suspicious entries: {len(suspicious_entries)} ===")
for e in suspicious_entries[:5]:
    energy = e.get("energy") or {}
    ctx = e.get("wasteful_context") or {}
    print(f"\n  ep={e.get('episode_count')} encounter={e.get('encounter')}")
    print(f"    energy.raw={energy.get('raw')}  combat.energy={energy.get('combat.energy')}  max_energy={energy.get('combat.max_energy')}")
    print(f"    round={e.get('combat_round')} hp={e.get('player_hp')}")
    print(f"    context: positive_count={ctx.get('positive_progress_count')} end_turn_indices={ctx.get('end_turn_indices')} zero_cost_positive={ctx.get('zero_cost_positive')}")
    print(f"    raw_hand (all):")
    for c in (e.get("raw_hand_cards") or []):
        print(f"      - {c.get('title')!r} cost={c.get('cost')} playable={c.get('is_playable')}")
    print(f"    legal_actions (all, mask>=0):")
    for la in e.get("legal_actions") or []:
        mark = "*" if la.get("is_chosen") else " "
        print(f"     {mark}idx={la.get('idx')} mask={la.get('mask')} fam={la.get('family')} title={la.get('title')!r} is_pos={la.get('is_positive')} roles={la.get('roles')} cost={la.get('cost')}")

# also find entry with max energy/hand
if entries:
    best = max(entries, key=lambda e: ((e.get("energy") or {}).get("raw") or 0) * 100 + len(e.get("raw_hand_cards") or []))
    energy = best.get("energy") or {}
    ctx = best.get("wasteful_context") or {}
    print(f"\n=== best candidate (max energy+hand) ===")
    print(f"  ep={best.get('episode_count')}  encounter={best.get('encounter')}")
    print(f"  energy.raw={energy.get('raw')}  combat.energy={energy.get('combat.energy')}")
    print(f"  hand_size={len(best.get('raw_hand_cards') or [])}")
    print(f"  positive_count={ctx.get('positive_progress_count')}")
    print(f"  legal_actions count (all)={len(best.get('legal_actions') or [])}")
