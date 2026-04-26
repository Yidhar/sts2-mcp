import json, sys
from collections import Counter
from pathlib import Path

run = sys.argv[1]
rows = [json.loads(l) for l in open(Path(run) / "reset_events.jsonl")]
terms = [r for r in rows if r.get("event") == "episode_terminal"]
f1 = [r for r in terms if r.get("max_floor_reached") == 1]
print(f"floor-1 episodes: {len(f1)} / {len(terms)} ({100*len(f1)/len(terms):.1f}%)")
print()
print("breakdown by termination cause:")
print(f"  terminated(natural death)  = {sum(1 for r in f1 if r.get('terminated') and not r.get('truncated'))}")
print(f"  truncated(watchdog)        = {sum(1 for r in f1 if r.get('truncated'))}")
print()
print("truncation_reason among floor-1:")
print(dict(Counter(r.get("truncation_reason") for r in f1)))
print()
print("stuck_phase among floor-1 watchdog trunc:")
print(dict(Counter(r.get("stuck_phase") for r in f1 if r.get("truncation_reason") == "phase_stuck_watchdog")))
print()
print("reward distribution among floor-1:")
rw = [r.get("reward") for r in f1 if r.get("reward") is not None]
if rw:
    import statistics as s
    print(f"  mean={s.mean(rw):.3f} min={min(rw):.2f} max={max(rw):.2f}")
print()
print("--- vs non-floor-1:")
nf1 = [r for r in terms if r.get("max_floor_reached") != 1]
nf1_trunc = Counter(r.get("truncation_reason") for r in nf1)
print(f"non-floor-1 terminations: {len(nf1)}")
print(f"  truncation_reason: {dict(nf1_trunc)}")
nf1_stuck = Counter(r.get("stuck_phase") for r in nf1 if r.get("truncation_reason") == "phase_stuck_watchdog")
if nf1_stuck:
    print(f"  stuck_phase: {dict(nf1_stuck)}")
print()
print("sample floor-1 events (first 3):")
for r in f1[:3]:
    keys = ["env_id", "final_floor", "max_floor_reached", "terminated", "truncated", "truncation_reason", "stuck_phase", "reward"]
    print({k: r.get(k) for k in keys if k in r or r.get(k) is not None})
