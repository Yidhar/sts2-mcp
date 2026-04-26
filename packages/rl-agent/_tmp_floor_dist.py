import json, statistics as s, sys
from collections import Counter
from pathlib import Path

run = sys.argv[1]
path = Path(run) / "reset_events.jsonl"
rows = [json.loads(l) for l in open(path)]
terms = [r for r in rows if r.get("event") == "episode_terminal"]
floors = [r["max_floor_reached"] for r in terms if r.get("max_floor_reached") is not None]
print(f"total episodes: {len(terms)}")
print()
print(f"{'floor':>5} | {'count':>5} | {'pct':>5} | bar")
print("-" * 50)
N = len(floors)
c = Counter(floors)
for f in sorted(c):
    n = c[f]
    pct = 100 * n / N
    bar = "#" * int(pct * 0.8)
    print(f"{f:>5} | {n:>5} | {pct:>4.1f}% | {bar}")
print()
print(f"mean   = {s.mean(floors):.2f}")
print(f"median = {s.median(floors):.0f}")
print(f"max    = {max(floors)}")
print(f"floor >= 7 : {sum(1 for x in floors if x>=7)}/{N} ({100*sum(1 for x in floors if x>=7)/N:.1f}%)")
print(f"floor >= 10: {sum(1 for x in floors if x>=10)}/{N} ({100*sum(1 for x in floors if x>=10)/N:.1f}%)")
print(f"floor >= 15: {sum(1 for x in floors if x>=15)}/{N} ({100*sum(1 for x in floors if x>=15)/N:.1f}%)")
print()
recent = floors[-50:] if N >= 50 else floors
print(f"--- last {len(recent)} episodes ---")
print(f"mean={s.mean(recent):.2f}  max={max(recent)}")
rc = Counter(recent)
for f in sorted(rc):
    print(f"  floor {f}: {rc[f]}")
