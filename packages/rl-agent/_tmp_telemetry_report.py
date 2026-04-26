import json, sys, statistics as s
from collections import Counter
from pathlib import Path

run = Path(sys.argv[1])
rows = [json.loads(l) for l in open(run / "reset_events.jsonl")]
terms = [r for r in rows if r.get("event") == "episode_terminal"]
N = len(terms)
print(f"=== {N} episodes ===\n")


def agg(field):
    vals = [r.get(field, 0) or 0 for r in terms]
    return sum(vals), vals


# 1. Potion use
p_total, _ = agg("potion_use_count")
pb, _ = agg("potion_use_boss_count")
pe, _ = agg("potion_use_elite_count")
p_bonus, _ = agg("potion_use_bonus_total")
p_discard, _ = agg("potion_discard_count")
print("[1] POTION:")
print(f"  total uses:        {p_total}  ({p_total/max(N,1):.3f}/ep)")
print(f"    on boss:         {pb}  ({pb/max(N,1):.3f}/ep)")
print(f"    on elite:        {pe}  ({pe/max(N,1):.3f}/ep)")
print(f"  discards:          {p_discard}  ({p_discard/max(N,1):.3f}/ep)")
print(f"  bonus accumulated: {p_bonus:.3f}")

# 2. Rest HEAL rate
r_enc, _ = agg("rest_site_encounters")
r_heal, _ = agg("rest_heal_chosen")
r_skip, _ = agg("rest_skip_heal_chosen")
r_skip_lowhp, _ = agg("rest_skip_heal_at_low_hp")
r_pen, _ = agg("rest_penalty_total")
print(f"\n[2/3] REST SITE:")
print(f"  encounters:         {r_enc}")
print(f"  heal chosen:        {r_heal}")
print(f"  skip-heal chosen:   {r_skip}")
print(f"  skip-heal at <60%:  {r_skip_lowhp}  ← should be low after training")
denom = r_heal + r_skip
heal_rate = 100 * r_heal / denom if denom else 0
print(f"  HEAL rate:          {heal_rate:.1f}% (target 70%+)")
print(f"  penalty accumulated:{r_pen:.3f}")

# 4. Boss
bd_raw, _ = agg("boss_damage_dealt_raw")
bd_bonus, _ = agg("boss_damage_bonus_total")
b_steps, _ = agg("boss_encounter_steps")
b_entries, _ = agg("boss_floor_entry_events")
print(f"\n[4/5] BOSS:")
print(f"  boss floor entries:   {b_entries}")
print(f"  boss encounter steps: {b_steps}")
print(f"  damage dealt (raw):   {bd_raw:.1f}")
print(f"  damage bonus total:   {bd_bonus:.3f}")

# 6. Floor 18+
floors = [r.get("max_floor_reached") or 0 for r in terms]
f17 = sum(1 for f in floors if f >= 17)
f18 = sum(1 for f in floors if f >= 18)
fc, _ = agg("floor_clear_events")
fc_reward, _ = agg("floor_clear_reward_total")
print(f"\n[6] FLOOR:")
print(f"  floor dist: {dict(sorted(Counter(floors).items()))}")
print(f"  max: {max(floors)}  mean: {s.mean(floors):.2f}")
print(f"  reach boss (>=17): {f17}/{N} = {100*f17/max(N,1):.1f}%")
print(f"  kill boss  (>=18): {f18}/{N} = {100*f18/max(N,1):.1f}%  (target 2-5%)")
print(f"  floor_clear_events total:  {fc}")
print(f"  floor_clear_reward total:  {fc_reward:.3f}")

rw = [r.get("reward") for r in terms if r.get("reward") is not None]
if rw:
    print(f"\nreward mean={s.mean(rw):.3f} min={min(rw):.2f} max={max(rw):.2f}")

# EV sanity
import csv as csv_mod

prog = list(csv_mod.DictReader(open(run / "sb3_async" / "progress.csv")))
ev = [float(r["train/explained_variance"]) for r in prog]
print(f"\n[EV] updates={len(ev)}, first5={s.mean(ev[:5]):+.3f}, last5={s.mean(ev[-5:]):+.3f}, range=[{min(ev):+.3f},{max(ev):+.3f}]")
