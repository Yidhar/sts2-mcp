"""Probe: dump full structure of one play_card legal action to see if
semantic.roles / block / damage fields are populated.

Reads existing snapshot pool, picks first ironclad card, hits bridge
combat_reset to get a real legal_actions list (sandbox mode), then prints
full JSON of the first play_card entry.

Use carefully if training is using session_0 — it'll briefly contend.
"""
import json
import sys
sys.path.insert(0, "/mnt/e/game/project/sts2_mcp/packages/rl-agent")

from sts2_env.bridge_client import BridgeClient

SESSION = "/mnt/c/Users/yidhar/AppData/Roaming/SlayTheSpire2/bridge/session_0.json"
ENCOUNTER = "ENCOUNTER.KAISER_CRAB_BOSS"

bridge = BridgeClient(session_path=SESSION)

print(f"[probe] combat_reset encounter={ENCOUNTER} (no overrides, default deck)")
result = bridge.combat_reset(
    character="ironclad",
    encounter_id=ENCOUNTER,
    timeout_ms=20000,
)

legal = result.get("legal_actions") or []
print(f"[probe] got {len(legal)} legal actions")

for kind in ("play_card", "use_potion", "end_turn"):
    found = next((a for a in legal if str(a.get("kind") or "").lower() == kind), None)
    print(f"\n=== first '{kind}' action ===")
    if found is None:
        print(f"  (none in legal list)")
        continue
    print(json.dumps(found, indent=2, ensure_ascii=False))
    # Highlight fields the train.py metric checks
    semantic = found.get("semantic") or {}
    roles = semantic.get("roles")
    block = semantic.get("block")
    damage = semantic.get("damage")
    family = semantic.get("family")
    print(f"  >>> semantic.family = {family!r}")
    print(f"  >>> semantic.roles  = {roles!r}")
    print(f"  >>> semantic.block  = {block!r}")
    print(f"  >>> semantic.damage = {damage!r}")
    # Also fall-through to top-level
    print(f"  >>> action.block (top-level) = {found.get('block')!r}")
