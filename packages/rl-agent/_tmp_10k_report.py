import json, csv, sys, statistics as s
from collections import Counter
from pathlib import Path

run = Path(sys.argv[1])
rows = [json.loads(l) for l in open(run / "reset_events.jsonl")]
terms = [r for r in rows if r.get("event") == "episode_terminal"]
N = len(terms)
print(f"=== episodes={N} ===\n")

trunc = Counter(r.get("truncation_reason") or "natural" for r in terms)
print("truncation_reason:")
for k, v in trunc.most_common():
    print(f"  {k}: {v} ({100*v/N:.1f}%)")
print()

stuck = [r for r in terms if r.get("truncation_reason") == "phase_stuck_watchdog"]
sp = Counter(r.get("stuck_phase") for r in stuck)
print("stuck_phase among watchdog trunc:")
for k, v in sp.most_common():
    print(f"  {k}: {v}")
print()

print("=== FIX 1: NEOW floor=1 card_selection stuck ===")
neow = [r for r in stuck if r.get("stuck_phase") == "card_selection" and int(r.get("stuck_floor", 0) or 0) == 1]
print(f"floor=1 card_selection stuck cases: {len(neow)}")
steps = Counter(r.get("stuck_steps") for r in neow)
print(f"stuck_steps distribution: {dict(steps)}")
print()

print("=== FIX 1b: All card_selection stuck ===")
cs_all = [r for r in stuck if r.get("stuck_phase") == "card_selection"]
print(f"card_selection stuck (all floors): {len(cs_all)}")
print(f"  stuck_steps: {dict(Counter(r.get('stuck_steps') for r in cs_all))}")
by_floor = Counter(int(r.get("stuck_floor", 0) or 0) for r in cs_all)
print(f"  by floor: {dict(sorted(by_floor.items()))}")
print()

print("=== FIX 2: stuck_phase rate ===")
print(f"stuck_total = {len(stuck)}/{N} = {100*len(stuck)/N:.1f}%  (prev smoke: 37%)")
print()

floors = [r.get("max_floor_reached") for r in terms if r.get("max_floor_reached") is not None]
if floors:
    print(f"floor dist: {dict(sorted(Counter(floors).items()))}")
    print(f"mean_floor={s.mean(floors):.2f} max={max(floors)}")
rw = [r.get("reward") for r in terms if r.get("reward") is not None]
if rw:
    print(f"reward mean={s.mean(rw):.3f} min={min(rw):.2f} max={max(rw):.2f}")

print()
prog = list(csv.DictReader(open(run / "sb3_async" / "progress.csv")))
N2 = len(prog)
def col(k):
    return [float(r[k]) for r in prog]
ev = col("train/explained_variance")
vloss = col("train/value_loss")
aux_c = col("train/aux_causality_loss")
aux_e = col("train/aux_enemy_state_loss")

print(f"=== FIX 3: EV late inversion check (updates={N2}) ===")
print(f"  first 5 ev: {[round(x,3) for x in ev[:5]]}")
print(f"  last 5  ev: {[round(x,3) for x in ev[-5:]]}")
print(f"  first 5 avg: {s.mean(ev[:5]):+.3f}")
print(f"  last 5  avg: {s.mean(ev[-5:]):+.3f}")
print(f"  range: [{min(ev):+.3f}, {max(ev):+.3f}]")
print()
print("=== trajectory ===")
print(f"aux_causality: {s.mean(aux_c[:5]):.4f} -> {s.mean(aux_c[-5:]):.4f}")
print(f"aux_enemy:     {s.mean(aux_e[:5]):.4f} -> {s.mean(aux_e[-5:]):.4f}")
print(f"value_loss:    {s.mean(vloss[:5]):.4f} -> {s.mean(vloss[-5:]):.4f}")
print(f"aux_enemy zero rate: {sum(1 for v in aux_e if v<0.005)}/{N2}")
print(f"aux_causality zero rate: {sum(1 for v in aux_c if v<0.0005)}/{N2}")
