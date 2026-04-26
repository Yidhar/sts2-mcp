import json, sys
d = json.load(open(sys.argv[1]))
print("summary keys:", list(d.keys()))
rows = d.get("most_sampled", []) + d.get("hardest", []) + d.get("easiest", [])
by_tier = {"boss": [], "elite": [], "normal": [], "weak": []}
seen = set()
for r in rows:
    eid = r.get("encounter_id")
    if eid in seen:
        continue
    seen.add(eid)
    t = r.get("encounter_tier", "unknown")
    if t in by_tier:
        by_tier[t].append(r)

for tier in ("boss", "elite", "normal", "weak"):
    ls = by_tier[tier]
    if not ls:
        continue
    ls.sort(key=lambda r: -(r.get("resolved_win_rate") or 0))
    print(f"\n=== {tier.upper()} ({len(ls)} entries) ===")
    print(f"{'encounter':<40} {'n':>4} {'wins':>5} {'losses':>6} {'WR':>7}")
    print("-" * 65)
    for r in ls:
        tag = r["encounter_id"].replace("ENCOUNTER.", "")
        wr = r.get("resolved_win_rate") or 0.0
        print(f"{tag:<40} {r['total']:>4} {r['wins']:>5} {r['losses']:>6} {100*wr:6.1f}%")
