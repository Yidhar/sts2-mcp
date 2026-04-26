from sts2_env.bridge_client import BridgeClient
from sts2_env.action_compact import compact_action_signature
import json

bridge = BridgeClient(session_path="/mnt/c/Users/yidhar/AppData/Roaming/SlayTheSpire2/bridge/session_0.json")
result = bridge.combat_reset(character="ironclad", encounter_id="ENCOUNTER.KAISER_CRAB_BOSS", timeout_ms=20000)
actions = result.get("legal_actions") or []
print(f"Total legal_actions: {len(actions)}")

# Find a :1 suffix action (targeted)
target_action = None
for a in actions:
    aid = str(a.get("action_id") or "")
    if aid.endswith(":1"):
        target_action = a
        break

if target_action is None:
    print("No :1 suffix action found")
else:
    aid = target_action.get("action_id")
    raw_target = target_action.get("target")
    print(f"\n=== TARGETED :1 action {aid} ===")
    print(f"  RAW.target = {json.dumps(raw_target, ensure_ascii=False)}")
    print(f"  RAW.target_combat_id = {target_action.get('target_combat_id')!r}")
    print(f"  RAW.target_name      = {target_action.get('target_name')!r}")

    compact = compact_action_signature(target_action)
    print(f"\n  COMPACT keys = {sorted(compact.keys())}")
    print(f"  COMPACT.target = {json.dumps(compact.get('target'), ensure_ascii=False)}")
    print(f"  COMPACT.target type = {type(compact.get('target')).__name__}")
    if isinstance(compact.get("target"), dict):
        t = compact["target"]
        print(f"    .combat_id = {t.get('combat_id')!r}")
        print(f"    .name      = {t.get('name')!r}")
        print(f"    .side      = {t.get('side')!r}")
    print(f"\n  FULL COMPACT = {json.dumps(compact, ensure_ascii=False)}")
