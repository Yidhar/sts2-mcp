from sts2_env.bridge_client import BridgeClient
import json

bridge = BridgeClient(session_path="/mnt/c/Users/yidhar/AppData/Roaming/SlayTheSpire2/bridge/session_0.json")
result = bridge.combat_reset(character="ironclad", encounter_id="ENCOUNTER.KAISER_CRAB_BOSS", timeout_ms=20000)
actions = result.get("legal_actions") or []
for a in actions:
    aid = str(a.get("action_id") or "")
    if aid.endswith(":1"):
        card = a.get("card") or {}
        print(f"action_id = {aid}")
        print(f"action.target = {json.dumps(a.get('target'), ensure_ascii=False)}")
        print()
        print(f"card keys = {sorted(card.keys())}")
        print(f"card.target = {card.get('target')!r}")
        print(f"card.target_type = {card.get('target_type')!r}")
        # any 'target' anywhere in nested?
        def walk(obj, path=""):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if k == "target":
                        print(f"  FOUND 'target' at {path}.{k} = {v!r}")
                    walk(v, path + "." + k)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    walk(item, f"{path}[{i}]")
        print("\nWalk action for 'target' keys:")
        walk(a, "action")
        break
