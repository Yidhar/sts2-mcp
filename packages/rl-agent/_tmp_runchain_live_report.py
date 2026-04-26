import json, csv, statistics as s, sys
from collections import Counter
from pathlib import Path

run = Path(sys.argv[1])
rows = [json.loads(l) for l in open(run / "reset_events.jsonl")]
terms = [r for r in rows if r.get("event") == "episode_terminal"]
N = len(terms)
print(f"=== {N} terminal events ===")
rc_keys = sorted(k for k in terms[-1] if "run_chain" in k)
print(f"run_chain fields on terminal: {rc_keys}")

pu = [r.get("potion_use_count", 0) or 0 for r in terms]
pu_boss = [r.get("potion_use_boss_count", 0) or 0 for r in terms]
pu_elite = [r.get("potion_use_elite_count", 0) or 0 for r in terms]
pu_disc = [r.get("potion_discard_count", 0) or 0 for r in terms]
print(f"\n[POTIONS]")
print(f"  total uses: {sum(pu)}  boss: {sum(pu_boss)}  elite: {sum(pu_elite)}  discards: {sum(pu_disc)}")
print(f"  episodes with any use: {sum(1 for v in pu if v>0)}/{N}")
print(f"  uses-per-ep distribution: {dict(sorted(Counter(pu).items()))}")

bf = [r.get("boss_floor_entry_events", 0) or 0 for r in terms]
bd = [r.get("boss_damage_dealt_raw", 0) or 0 for r in terms]
bs = [r.get("boss_encounter_steps", 0) or 0 for r in terms]
print(f"\n[BOSS EXPOSURE]")
print(f"  chains reaching boss: {sum(1 for v in bf if v>0)}/{N}")
print(f"  chains dealing damage: {sum(1 for v in bd if v>0)}/{N}")
print(f"  max boss damage: {max(bd):.1f}  total boss steps: {sum(bs)}")

rw = [r.get("reward", 0) or 0 for r in terms]
print(f"\n[REWARDS]")
print(f"  min={min(rw):.2f} max={max(rw):.2f} mean={s.mean(rw):.3f}")
print(f"  last 50: max={max(rw[-50:]):.2f} mean={s.mean(rw[-50:]):.3f}")
print(f"  eps with r>0: {sum(1 for r in rw if r>0)}/{N}")
print(f"  eps with r>+3 (likely win with bonus): {sum(1 for r in rw if r>3)}/{N}")

by_env = Counter(r.get("env_id") for r in terms)
print(f"\n[ENV BALANCE] {dict(sorted(by_env.items()))}")
trunc = Counter(r.get("truncation_reason") or "natural" for r in terms)
print(f"[TERMINATION] {dict(trunc)}")

# EV from progress.csv
prog = list(csv.DictReader(open(run / "sb3_async" / "progress.csv")))
ev = [float(r["train/explained_variance"]) for r in prog]
vl = [float(r["train/value_loss"]) for r in prog]
print(f"\n[TRAINING] updates={len(prog)}")
if ev:
    print(f"  EV first5={s.mean(ev[:5]):+.3f}  last5={s.mean(ev[-5:]):+.3f}  range=[{min(ev):+.3f},{max(ev):+.3f}]")
    print(f"  vloss first5={s.mean(vl[:5]):.3f}  last5={s.mean(vl[-5:]):.3f}")

# If run_chain fields exist in per-step events check them
# Fields the commit added: run_chain_carried_hp_before_combat, run_chain_carried_potions_before_combat
# These likely only on reset events (non-terminal). Check all rows:
non_term = [r for r in rows if r.get("event") != "episode_terminal"]
rc_carry = [r for r in non_term if "run_chain_carried_potions_before_combat" in r]
print(f"\n[RUN_CHAIN TELEMETRY]")
print(f"  non-terminal events: {len(non_term)}")
print(f"  events with run_chain_carried_potions_before_combat: {len(rc_carry)}")
if rc_carry:
    sample = rc_carry[:3]
    for r in sample:
        print(f"  sample: idx={r.get('run_chain_idx')} len={r.get('run_chain_length')} carried_pots={r.get('run_chain_carried_potions_before_combat')} hp={r.get('run_chain_carried_hp_before_combat')}")
