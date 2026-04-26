import json, sys, statistics as s, csv as csv_mod
from collections import Counter, defaultdict
from pathlib import Path

run = Path(sys.argv[1])
rows = [json.loads(l) for l in open(run / "reset_events.jsonl")]
terms = [r for r in rows if r.get("event") == "episode_terminal"]
N = len(terms)
print(f"=== episodes={N} ===\n")

# Encounter distribution
enc = Counter(r.get("encounter_id") or "?" for r in terms)
print("[2] encounter distribution:")
for k, n in enc.most_common():
    print(f"  {k}: {n} ({100*n/max(N,1):.1f}%)")
print()

# Win rate proxy — combat_sandbox win criterion usually reward >= threshold or termination with max_hp preserved
# Try reading: reward>0 as "survived" proxy (boss sandbox reward ~1 on win, 0 on loss)
print("[3] per-encounter outcomes (reward>0 as win proxy):")
by_enc = defaultdict(lambda: {"n": 0, "won": 0, "rewards": []})
for r in terms:
    e = r.get("encounter_id") or "?"
    rw = r.get("reward") or 0
    by_enc[e]["n"] += 1
    if rw > 0:
        by_enc[e]["won"] += 1
    by_enc[e]["rewards"].append(rw)

for e in sorted(by_enc):
    d = by_enc[e]
    mean_r = s.mean(d["rewards"]) if d["rewards"] else 0
    wr = 100 * d["won"] / max(d["n"], 1)
    print(f"  {e}: n={d['n']}, WR_proxy={wr:.1f}% ({d['won']}/{d['n']}), reward mean={mean_r:.2f}")

# Overall reward dist
rw = [r.get("reward") or 0 for r in terms]
print(f"\noverall: reward mean={s.mean(rw):.3f} min={min(rw):.2f} max={max(rw):.2f}")
wins = sum(1 for r in rw if r > 0)
print(f"eps with reward>0: {wins}/{N} = {100*wins/max(N,1):.1f}%")

# EV
prog = list(csv_mod.DictReader(open(run / "sb3_async" / "progress.csv")))
ev = [float(r["train/explained_variance"]) for r in prog]
its = [float(r["rollout/end_to_end_steps_per_s"]) for r in prog]
print(f"\n[4] EV stability: updates={len(ev)}")
print(f"  first5: {s.mean(ev[:5]):+.3f}  last5: {s.mean(ev[-5:]):+.3f}  range=[{min(ev):+.3f},{max(ev):+.3f}]  neg={sum(1 for x in ev if x<0)}/{len(ev)}")

print(f"\n[5] throughput: avg {s.mean(its):.1f} it/s  (first5 {s.mean(its[:5]):.1f}  last5 {s.mean(its[-5:]):.1f})")
