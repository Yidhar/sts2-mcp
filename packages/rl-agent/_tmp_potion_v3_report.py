import json, sys, statistics as s, csv as csv_mod
from collections import Counter
from pathlib import Path

run = Path(sys.argv[1])
rows = [json.loads(l) for l in open(run / "reset_events.jsonl")]
terms = [r for r in rows if r.get("event") == "episode_terminal"]
N = len(terms)
print(f"=== episodes={N} ===\n")


def agg(field):
    return sum(r.get(field, 0) or 0 for r in terms)


# 1. New fields populated?
hoard_ends = [r.get("potion_hoarding_unused_at_end") for r in terms]
hoard_pens = [r.get("potion_hoarding_penalty_total") for r in terms]
nonnull_end = sum(1 for v in hoard_ends if v is not None)
nonnull_pen = sum(1 for v in hoard_pens if v is not None)
print(f"[1] NEW FIELDS populated:")
print(f"  potion_hoarding_unused_at_end: {nonnull_end}/{N} non-null")
print(f"  potion_hoarding_penalty_total: {nonnull_pen}/{N} non-null")

# 2. Boss-use rate
pc = agg("potion_use_count")
pb = agg("potion_use_boss_count")
pe = agg("potion_use_elite_count")
boss_rate = 100 * pb / pc if pc else 0
elite_rate = 100 * pe / pc if pc else 0
print(f"\n[2] POTION TIMING:")
print(f"  total uses:           {pc:.0f}  ({pc/max(N,1):.2f}/ep)")
print(f"  boss uses:            {pb:.0f}  ({100*pb/max(pc,1):.1f}% of total) [target >30%]")
print(f"  elite uses:           {pe:.0f}  ({100*pe/max(pc,1):.1f}% of total)")
print(f"  boss+elite rate:      {100*(pb+pe)/max(pc,1):.1f}%")

# 3. Episodes with 0 leftover potions
hoard_vals = [int(r.get("potion_hoarding_unused_at_end") or 0) for r in terms]
zero_left = sum(1 for v in hoard_vals if v == 0)
print(f"\n[3] HOARDING END STATE:")
print(f"  episodes ending with 0 potions: {zero_left}/{N} = {100*zero_left/max(N,1):.1f}% [target 90%+]")
dist = Counter(hoard_vals)
print(f"  leftover distribution: {dict(sorted(dist.items()))}")

# 4. Penalty distribution
pens = [float(r.get("potion_hoarding_penalty_total") or 0) for r in terms]
print(f"\n[4] HOARDING PENALTY:")
if pens:
    print(f"  penalty sum over all eps: {sum(pens):.3f}")
    print(f"  mean per ep: {s.mean(pens):.4f}")
    print(f"  min: {min(pens):.3f}  max: {max(pens):.3f}")
    nonzero = sum(1 for p in pens if p < 0)
    print(f"  eps with negative penalty: {nonzero}/{N} = {100*nonzero/N:.1f}%")

# 5. EV stability
prog = list(csv_mod.DictReader(open(run / "sb3_async" / "progress.csv")))
ev = [float(r["train/explained_variance"]) for r in prog]
print(f"\n[5] EV:")
print(f"  updates: {len(ev)}")
print(f"  first5 avg: {s.mean(ev[:5]):+.3f}")
print(f"  last5  avg: {s.mean(ev[-5:]):+.3f}  [must stay >0]")
print(f"  range: [{min(ev):+.3f}, {max(ev):+.3f}]")
print(f"  negative count: {sum(1 for x in ev if x<0)}/{len(ev)}")

# 6. Floor 18+
floors = [r.get("max_floor_reached") or 0 for r in terms]
f17 = sum(1 for f in floors if f >= 17)
f18 = sum(1 for f in floors if f >= 18)
print(f"\n[6] FLOOR:")
print(f"  dist: {dict(sorted(Counter(floors).items()))}")
print(f"  mean: {s.mean(floors):.2f}  max: {max(floors)}")
print(f"  floor >=17 (boss reached): {f17}/{N} = {100*f17/max(N,1):.1f}%")
print(f"  floor >=18 (boss killed):  {f18}/{N} = {100*f18/max(N,1):.1f}% [target 2-5%]")

# Reward range
rw = [r.get("reward") for r in terms if r.get("reward") is not None]
if rw:
    print(f"\nepisode reward: mean={s.mean(rw):.3f} min={min(rw):.2f} max={max(rw):.2f}")
