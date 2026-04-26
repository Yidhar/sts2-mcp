import json, csv, statistics as s, sys
from collections import Counter
from pathlib import Path

run = Path(sys.argv[1])
rows = [json.loads(l) for l in open(run / "reset_events.jsonl")]
terms = [r for r in rows if r.get("event") == "episode_terminal"]
N = len(terms)
print(f"total terminal events: {N}")
print(f"sample keys (last): {sorted(k for k in terms[-1] if 'run_chain' in k or 'potion' in k.lower() or 'boss' in k.lower())}")
print()

# Chain length distribution
chain_lens = [r.get("run_chain_length") or r.get("run_chain", {}).get("length") if isinstance(r.get("run_chain"), dict) else None for r in terms]
chain_lens = [c for c in chain_lens if c is not None]
if chain_lens:
    print(f"run_chain_length: n={len(chain_lens)}, min={min(chain_lens)}, max={max(chain_lens)}, mean={s.mean(chain_lens):.2f}")
    dist = Counter(chain_lens)
    print(f"  distribution: {dict(sorted(dist.items()))}")
print()

# Boss fraction — need to inspect chain encounters
# Look for boss indicators in encounter_id list or run_chain field
boss_contained = 0
no_chain_info = 0
boss_in_chain_detail = []
for r in terms:
    rc = r.get("run_chain")
    if isinstance(rc, dict):
        encs = rc.get("encounters") or rc.get("encounter_ids") or []
    else:
        encs = None
    if not encs:
        # try run_chain_encounter_ids
        encs = r.get("run_chain_encounter_ids") or r.get("run_chain_encounters")
    if encs:
        has_boss = any("BOSS" in str(e).upper() for e in encs)
        if has_boss:
            boss_contained += 1
    else:
        no_chain_info += 1

total_with_info = N - no_chain_info
print(f"Boss in chain: {boss_contained}/{total_with_info} = {100*boss_contained/max(total_with_info,1):.1f}% (target ~60%)")
print(f"no chain info: {no_chain_info}/{N}")
print()

# Per-env balance
by_env = Counter(r.get("env_id") for r in terms)
print(f"per env_id: {dict(sorted(by_env.items()))}")
print()

# Rewards / outcomes
rw = [r.get("reward") or 0 for r in terms]
if rw:
    print(f"reward min={min(rw):.2f} max={max(rw):.2f} mean={s.mean(rw):.3f}")
    print(f"last 50: max={max(rw[-50:]):.2f} mean={s.mean(rw[-50:]):.3f}")
# Truncation reasons
trunc = Counter(r.get("truncation_reason") or "natural" for r in terms)
print(f"termination: {dict(trunc)}")
print()

# EV from progress
prog = list(csv.DictReader(open(run / "sb3_async" / "progress.csv")))
ev = [float(r["train/explained_variance"]) for r in prog]
vl = [float(r["train/value_loss"]) for r in prog]
print(f"updates={len(prog)}")
if ev:
    print(f"EV first5={s.mean(ev[:5]):+.3f} last5={s.mean(ev[-5:]):+.3f} range=[{min(ev):+.3f},{max(ev):+.3f}]")
    print(f"vloss first5={s.mean(vl[:5]):.3f} last5={s.mean(vl[-5:]):.3f}")
