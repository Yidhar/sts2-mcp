import json
path = "logs_muzero/muzero_token_memory_combat_sandbox_2envs_20260424_233346/diagnostics/kaiser_facing_legal_actions.jsonl"
with open(path) as f:
    d = json.loads(f.readline())
print("enemies combat_ids:")
for e in d["enemies"]:
    power_ids = [p.get("id") for p in (e.get("powers") or [])]
    print(f"  combat_id={e.get('combat_id')} name={e.get('name')} powers={power_ids}")
print()
print("actions keys union:")
all_keys = set()
for a in d["actions"]:
    all_keys.update(a.keys())
print(" ", sorted(all_keys))
print()
print("actions with any target/back_pos info (non-empty):")
count_with_target = 0
for a in d["actions"]:
    tid = a.get("target_combat_id")
    ts = a.get("target_side_or_back_attack_position")
    tb = a.get("target_back_attack_position")
    if tid or (ts and ts != "") or (tb and tb != ""):
        count_with_target += 1
        aid = a.get("action_id")
        print(f"  aid={aid!r} tid={tid!r} side={ts!r} back_pos={tb!r}")
        if count_with_target >= 10:
            break
print(f"  (found {count_with_target} actions with target info)")

print()
print("action_id suffix distribution:")
from collections import Counter
suffixes = Counter()
for a in d["actions"]:
    aid = str(a.get("action_id") or "")
    parts = aid.split(":")
    if len(parts) >= 4:
        suffixes[parts[-1]] += 1
    else:
        suffixes["(no_suffix)"] += 1
print(f"  {suffixes.most_common(10)}")

print()
print("first 3 raw actions (full):")
for a in d["actions"][:3]:
    print(f"  {json.dumps(a, ensure_ascii=False)}")
