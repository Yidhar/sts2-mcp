"""Analyze end_turn_selections.jsonl — focus on wasteful=false with playable cards."""
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
print(f"total end_turn selections: {len(entries)}")

wasteful_true = [e for e in entries if e.get("wasteful_end_turn")]
wasteful_false = [e for e in entries if not e.get("wasteful_end_turn")]
print(f"  wasteful_end_turn=True : {len(wasteful_true)}")
print(f"  wasteful_end_turn=False: {len(wasteful_false)}")

# Split false cases by whether there were playable non-end-turn actions
suspicious = []  # wasteful=False but had playable non-end-turn actions
justified = []   # wasteful=False because no positive playable actions

for e in wasteful_false:
    playable_non_end = [
        la for la in e.get("legal_actions", [])
        if la.get("mask", 0) > 0 and la.get("family") != "end_turn"
    ]
    is_positive_true = sum(1 for la in playable_non_end if la.get("is_positive"))
    ctx = e.get("wasteful_context") or {}
    energy = (e.get("energy") or {}).get("raw") or 0
    if playable_non_end and is_positive_true >= 1 and energy > 0.05:
        suspicious.append((e, playable_non_end, is_positive_true))
    else:
        justified.append((e, playable_non_end, is_positive_true))

print(f"\n  * SUSPICIOUS (had playable+positive+energy but wasteful=False): {len(suspicious)}")
print(f"  * Justified (no positive playable / no energy): {len(justified)}")


def show_entry(tag, e, playable_non_end, is_pos_count):
    ctx = e.get("wasteful_context") or {}
    energy = e.get("energy") or {}
    print(f"\n  [{tag}] ep={e.get('episode_count')} step={e.get('total_steps')} encounter={e.get('encounter')}")
    print(f"    wasteful_end_turn={e.get('wasteful_end_turn')}  context.wasteful={ctx.get('wasteful')}")
    print(f"    energy: raw={energy.get('raw'):.2f} enc={energy.get('encoded'):.2f} combat.energy={energy.get('combat.energy')!r}")
    print(f"    positive_progress_count={ctx.get('positive_progress_count')}  zero_cost_positive={ctx.get('zero_cost_positive')}")
    print(f"    playable_non_end legal_actions: {len(playable_non_end)}  is_positive=True: {is_pos_count}")
    print(f"    raw_hand:")
    for c in (e.get("raw_hand_cards") or [])[:6]:
        print(f"      - title={c.get('title')!r} cost={c.get('cost')} playable={c.get('is_playable')}")
    print(f"    legal_actions with mask>0 (non-end):")
    for la in playable_non_end:
        print(f"      idx={la.get('idx')} fam={la.get('family')} title={la.get('title')!r} "
              f"is_pos={la.get('is_positive')} roles={la.get('roles')} "
              f"dmg={la.get('damage')} blk={la.get('block')} cost={la.get('cost')} chosen={la.get('is_chosen')}")


# Show 5 suspicious and 3 justified examples
for i, (e, pla, ipc) in enumerate(suspicious[:5]):
    show_entry(f"SUSP {i+1}", e, pla, ipc)
for i, (e, pla, ipc) in enumerate(justified[:3]):
    show_entry(f"JUST {i+1}", e, pla, ipc)
