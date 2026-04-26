import json, csv, statistics as s, sys
from collections import Counter, defaultdict
from pathlib import Path

run = Path(sys.argv[1])
rows = [json.loads(l) for l in open(run / "reset_events.jsonl")]
terms = [r for r in rows if r.get("event") == "episode_terminal"]
N = len(terms)
rw = [r.get("reward") or 0 for r in terms]

print(f"=== training end: {N} episodes ===")
print(f"reward all   : min={min(rw):.2f} max={max(rw):.2f} mean={s.mean(rw):.3f}")
print(f"reward last100: min={min(rw[-100:]):.2f} max={max(rw[-100:]):.2f} mean={s.mean(rw[-100:]):.3f}")
print(f"reward last500: min={min(rw[-500:]):.2f} max={max(rw[-500:]):.2f} mean={s.mean(rw[-500:]):.3f}")
by_env = Counter(r.get("env_id") for r in terms)
print(f"per env_id: {dict(sorted(by_env.items()))}")
print()

# Available win indicator fields
last = terms[-1]
print("sample fields:", [k for k in last if "won" in k.lower() or "terminated" in k or "truncated" in k])
print()

# Training tail metrics
prog = list(csv.DictReader(open(run / "sb3_async" / "progress.csv")))
ev = [float(r["train/explained_variance"]) for r in prog]
vl = [float(r["train/value_loss"]) for r in prog]
print(f"progress.csv updates={len(prog)}")
print(f"EV last10={[round(x,3) for x in ev[-10:]]}  avg={s.mean(ev[-10:]):+.3f}")
print(f"vloss first5={s.mean(vl[:5]):.3f}  last5={s.mean(vl[-5:]):.3f}")
print()

# Per-encounter — use reward threshold as win proxy (sandbox FIX2 emits +3.5 win sentinel)
WIN_THR = 2.0
by_enc = defaultdict(lambda: {"n": 0, "w": 0, "rs": []})
for r in terms:
    e = r.get("encounter_id") or "?"
    by_enc[e]["n"] += 1
    by_enc[e]["rs"].append(r.get("reward") or 0)
    if (r.get("reward") or 0) >= WIN_THR:
        by_enc[e]["w"] += 1

print(f"per-encounter (win proxy: reward >= {WIN_THR}):")
print(f"  {'encounter':<45} {'n':>4}  {'wins':>4}  {'WR':>6}  {'rmean':>6}  {'rmax':>6}")
print("  " + "-" * 78)
encs = sorted(by_enc.items(), key=lambda kv: -kv[1]["w"] / max(kv[1]["n"], 1))
for e, d in encs:
    if d["n"] < 20:
        continue
    wr = 100 * d["w"] / d["n"]
    rmean = s.mean(d["rs"])
    rmax = max(d["rs"])
    tag = e.replace("ENCOUNTER.", "")
    print(f"  {tag:<45} {d['n']:>4}  {d['w']:>4}  {wr:5.1f}%  {rmean:+6.2f}  {rmax:+6.2f}")

# Aggregate WR proxy
total_n = sum(d["n"] for d in by_enc.values())
total_w = sum(d["w"] for d in by_enc.values())
print(f"\nOVERALL proxy WR = {total_w}/{total_n} = {100*total_w/max(total_n,1):.1f}%")

# Last 200 eps proxy
recent_terms = terms[-200:]
rn = sum(1 for r in recent_terms)
rw_ = sum(1 for r in recent_terms if (r.get("reward") or 0) >= WIN_THR)
print(f"LAST 200 eps proxy WR = {rw_}/{rn} = {100*rw_/max(rn,1):.1f}%")
