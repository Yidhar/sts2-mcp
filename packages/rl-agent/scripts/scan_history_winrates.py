import urllib.request
import json

url = "http://127.0.0.1:8765/api/v1/runs"
runs_payload = json.loads(urllib.request.urlopen(url).read())
runs = runs_payload.get("runs", [])
print(f"Scanning {len(runs)} runs...")

records = []

for idx, r in enumerate(runs):
    k = r["key"]
    try:
        snap_url = f"http://127.0.0.1:8765/api/v1/snapshot?run={k}"
        snap = json.loads(urllib.request.urlopen(snap_url).read())
    except Exception:
        continue

    lineage = snap.get("run", {}).get("lineage") or r.get("lineage") or k
    evals = snap.get("evaluations", [])
    for e in evals:
        wins = e.get("run_wins")
        total = e.get("episodes")
        act3_wins = e.get("act_3_successes")
        act1_wins = e.get("act_1_successes")
        step = e.get("environment_steps")
        gate = e.get("evaluation_gate")
        mean_rev = e.get("mean_revivals")
        mean_hp = e.get("mean_player_hp_lost")
        if wins is not None and total:
            rate = wins / total
            records.append({
                "run_key": k,
                "lineage": lineage,
                "step": step,
                "gate": gate,
                "wins": wins,
                "total": total,
                "win_rate": rate,
                "act3_wins": act3_wins,
                "act1_wins": act1_wins,
                "mean_revivals": mean_rev,
                "mean_hp_lost": mean_hp,
            })

records.sort(key=lambda x: (x["win_rate"], x["wins"]), reverse=True)

print(f"\nTotal evaluation gates found across all runs: {len(records)}")
print("\nTop 25 Historical Win Rates:")
seen = set()
for rec in records:
    combo = (rec["lineage"], rec["win_rate"])
    if combo not in seen:
        seen.add(combo)
        print(f"{rec['win_rate']*100:5.1f}% ({rec['wins']}/{rec['total']}) | Act3: {rec['act3_wins']}/{rec['total']} | Act1: {rec['act1_wins']}/{rec['total']} | Step: {rec['step']} | MeanRev: {rec['mean_revivals']} | Lineage: {rec['lineage']}")
        if len(seen) >= 25:
            break
