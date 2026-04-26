import json, sys
from collections import Counter, defaultdict
from pathlib import Path

run = Path(sys.argv[1])
rows = [json.loads(l) for l in open(run / "reset_events.jsonl")]
print(f"total events: {len(rows)}")

# Check what events exist beyond episode_terminal
events = Counter(r.get("event") for r in rows)
print(f"event types: {dict(events)}")
print()

# Action kind histogram across all events
action_kinds = Counter(r.get("action_kind") for r in rows if r.get("action_kind"))
print("action_kind distribution (all events):")
for k, n in action_kinds.most_common(20):
    print(f"  {k}: {n}")
print()

# Per-episode metrics
terms = [r for r in rows if r.get("event") == "episode_terminal"]
N = len(terms)
print(f"=== {N} episodes ===")

# Metric 1: potion usage
potion_action_kinds = ["use_potion", "discard_potion", "potion_use"]
potion_step_events = [r for r in rows if r.get("action_kind") in potion_action_kinds]
print(f"\n[1] potion usage:")
print(f"  potion-related step events: {len(potion_step_events)}")
print(f"  episodes: {N}")
print(f"  potion uses per episode: {len(potion_step_events) / max(N,1):.3f}")

# Metric 2: boss kill (floor >= 18)
floors = [r.get("max_floor_reached") or 0 for r in terms]
boss_killed = sum(1 for f in floors if f >= 18)
boss_reached = sum(1 for f in floors if f >= 17)
print(f"\n[2] boss reach/kill:")
print(f"  floor>=17 (boss reached): {boss_reached}/{N} = {100*boss_reached/max(N,1):.1f}%")
print(f"  floor>=18 (boss killed):  {boss_killed}/{N} = {100*boss_killed/max(N,1):.1f}%")

# Metric 3: rest HEAL (need to find rest action_kind events)
rest_events = [r for r in rows if r.get("action_kind") and "rest" in str(r.get("action_kind","")).lower()]
heal_events = [r for r in rows if r.get("action_kind") and "heal" in str(r.get("action_kind","")).lower()]
print(f"\n[3] rest/heal:")
print(f"  rest-related action events: {len(rest_events)}")
print(f"  heal-related action events: {len(heal_events)}")

# Metric 4: EV stability (read progress.csv)
import csv as csv_mod
prog = list(csv_mod.DictReader(open(run / "sb3_async" / "progress.csv")))
ev = [float(r["train/explained_variance"]) for r in prog]
import statistics as s
print(f"\n[4] EV stability:")
print(f"  updates: {len(ev)}")
print(f"  first 5 avg: {s.mean(ev[:5]):+.3f}")
print(f"  last 5 avg:  {s.mean(ev[-5:]):+.3f}")
print(f"  range: [{min(ev):+.3f}, {max(ev):+.3f}]")
print(f"  negative count: {sum(1 for x in ev if x<0)}/{len(ev)}")

# Reward range comparison
rwd = [r["reward"] for r in terms if r.get("reward") is not None]
if rwd:
    print(f"\nepisode reward: mean={s.mean(rwd):.3f} min={min(rwd):.2f} max={max(rwd):.2f}")
print(f"floor dist: {dict(sorted(Counter(floors).items()))}")
