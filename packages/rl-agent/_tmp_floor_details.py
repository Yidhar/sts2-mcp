import json, glob, csv, sys
from collections import Counter, defaultdict
from pathlib import Path

run = Path(sys.argv[1])
rows = [json.loads(l) for l in open(run / "reset_events.jsonl")]
terms = [r for r in rows if r.get("event") == "episode_terminal"]
print(f"=== episodes={len(terms)}")
print()
trunc = Counter(r.get("truncation_reason") or "natural" for r in terms)
print("truncation_reason:")
for k, v in trunc.most_common():
    print(f"  {k}: {v} ({100*v/len(terms):.1f}%)")
stuck = Counter(r.get("stuck_phase") for r in terms if r.get("truncation_reason") == "phase_stuck_watchdog")
if stuck:
    print("stuck_phase among watchdog trunc:")
    for k, v in stuck.most_common():
        print(f"  {k}: {v}")
print()

# Per-floor episode length from monitor CSVs
lens_by_floor = defaultdict(list)
lens_by_floor_trunc = defaultdict(list)
lens_by_floor_natural = defaultdict(list)

# Map (env_id, ep_idx) terminal → truncated?
# Simpler: use reset_events (has max_floor_reached + truncation_reason but not l)
# monitor has max_floor_reached + l but not truncation_reason
# Join via sequence: both in same order per worker
# Easier: just split by truncated flag
for r in terms:
    mf = r.get("max_floor_reached")
    if mf is None:
        continue
    # Estimate ep length from collect_elapsed etc? Not available.
    # Fall back to monitor
    pass

for f in sorted(glob.glob(str(run / "monitor*.csv"))):
    with open(f) as fh:
        next(fh)  # skip comment
        rdr = csv.DictReader(fh)
        for row in rdr:
            try:
                floor = int(row["max_floor_reached"])
                L = int(row["l"])
                lens_by_floor[floor].append(L)
            except Exception:
                pass

print("steps per floor (ep length distribution, from Monitor):")
print(f"  {'floor':>5} | {'n':>4} | {'mean':>6} | {'min':>4} | {'max':>5}")
print("  " + "-" * 40)
for f in sorted(lens_by_floor):
    vs = lens_by_floor[f]
    print(f"  {f:>5} | {len(vs):>4} | {sum(vs)//len(vs):>6} | {min(vs):>4} | {max(vs):>5}")

# Cross-tabulate: truncated vs natural per floor (from reset_events)
print()
print("per-floor: natural vs watchdog trunc counts:")
by_floor_reason = defaultdict(Counter)
for r in terms:
    mf = r.get("max_floor_reached")
    if mf is None:
        continue
    reason = r.get("truncation_reason") or "natural"
    by_floor_reason[mf][reason] += 1
print(f"  {'floor':>5} | {'natural':>7} | {'watchdog':>8} | watchdog_stuck_phases")
print("  " + "-" * 60)
for f in sorted(by_floor_reason):
    c = by_floor_reason[f]
    nat = c.get("natural", 0)
    wd = c.get("phase_stuck_watchdog", 0)
    # Stuck phases
    phases = Counter(
        r.get("stuck_phase")
        for r in terms
        if r.get("max_floor_reached") == f and r.get("truncation_reason") == "phase_stuck_watchdog"
    )
    ph_str = ", ".join(f"{k}:{v}" for k, v in phases.most_common())
    print(f"  {f:>5} | {nat:>7} | {wd:>8} | {ph_str}")
