"""Probe raw bridge legal_action for KAISER_CRAB_BOSS to see target field content."""
import json
import sys
sys.path.insert(0, "/mnt/e/game/project/sts2_mcp/packages/rl-agent")

from sts2_env.bridge_client import BridgeClient

SESSION = "/mnt/c/Users/yidhar/AppData/Roaming/SlayTheSpire2/bridge/session_0.json"
bridge = BridgeClient(session_path=SESSION)

print("[probe] combat_reset KAISER_CRAB_BOSS")
result = bridge.combat_reset(
    character="ironclad",
    encounter_id="ENCOUNTER.KAISER_CRAB_BOSS",
    timeout_ms=20000,
)

legal = result.get("legal_actions") or []
print(f"[probe] {len(legal)} legal actions; show all with target info:")
for a in legal:
    aid = a.get("action_id")
    target = a.get("target")
    target_combat_id = a.get("target_combat_id")
    target_name = a.get("target_name")
    target_side = a.get("target_side")
    kind = a.get("kind")
    print(f"\n  action_id = {aid}")
    print(f"    kind            = {kind}")
    print(f"    target_combat_id= {target_combat_id}")
    print(f"    target_name     = {target_name}")
    print(f"    target_side     = {target_side}")
    print(f"    target          = {json.dumps(target, ensure_ascii=False)[:200] if target else 'None'}")
    if target and isinstance(target, dict):
        print(f"    target.combat_id = {target.get('combat_id')!r}")
        print(f"    target.name     = {target.get('name')!r}")
        print(f"    target.side     = {target.get('side')!r}")
