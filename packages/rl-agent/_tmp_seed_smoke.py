"""Smoke test seed injection: 2 seeds × 2 resets each, verify
info.debug_seed_override round-trips and map-reproducible across re-reset.
"""
import sys
sys.path.insert(0, "/mnt/e/game/project/sts2_mcp/packages/rl-agent")

from sts2_env.bridge_client import BridgeClient
from sts2_env.env_v2 import SlayTheSpire2EnvV2

SESSION = "/mnt/c/Users/yidhar/AppData/Roaming/SlayTheSpire2/bridge/session_0.json"
SEEDS = ["1234567890", "ABCDEFGH12"]

print("[smoke] opening bridge ...")
bridge = BridgeClient(session_path=SESSION)

# Confirm env_spec reports new seed capability
spec = bridge.get_spec()
import json
print(f"[smoke] spec.reset sub-dict: {json.dumps(spec.get('reset'), indent=2, ensure_ascii=False)[:400]}")
print(f"[smoke] env_api_version = {spec.get('env_api_version')}")
# Direct reset call bypassing env to check info keys
reset_resp = bridge.reset(seed="1234567890", force_fresh=True)
print(f"[smoke] direct bridge.reset keys: {sorted(reset_resp.keys())[:20]}")
info = reset_resp.get("info") or {}
print(f"[smoke] info keys: {sorted(info.keys())}")
print(f"[smoke] info.debug_seed_override = {info.get('debug_seed_override')}")

env = SlayTheSpire2EnvV2(bridge=bridge, character="ironclad", seed_pool=SEEDS, seed_strategy="round_robin")

def take_until_map(env, max_steps=30):
    """Step until we reach MAP screen (past Neow), return map node signature."""
    raw = env._last_obs_raw or {}
    for _ in range(max_steps):
        phase = raw.get("phase") or ""
        screen = raw.get("screen") or ""
        if str(screen).upper() == "MAP" or str(phase).lower() == "map":
            break
        legal = env._legal_actions
        if not legal: break
        _, _, term, trunc, _ = env.step(0)
        if term or trunc: break
        raw = env._last_obs_raw or {}
    # Map fingerprint: legal map:col,row actions at map screen
    legal = env._legal_actions or []
    map_actions = tuple(sorted(a.get("action_id","") for a in legal if str(a.get("action_id","")).startswith("map:")))
    return len(map_actions), map_actions, raw.get("screen"), raw.get("phase")

seen = []
# First call env.reset and step to map, dump raw obs keys to find map representation
env.reset()
take_until_map(env, max_steps=40)
raw_sample = env._last_obs_raw or {}
print(f"\n[DEBUG] raw obs keys: {sorted(raw_sample.keys())[:30]}")
import json as _j
print(f"[DEBUG] run keys: {sorted((raw_sample.get('run') or {}).keys())}")
print(f"[DEBUG] current_floor: {raw_sample.get('current_floor')}")
# look for any legal action that is choose_map_node
legal = env._legal_actions or []
map_actions = [a for a in legal if "map" in str(a.get('kind','')).lower() or "map" in str(a.get('action_id','')).lower()]
print(f"[DEBUG] map-related legal actions: {len(map_actions)}, sample: {[a.get('action_id') for a in map_actions[:3]]}")
seen = []
for trial in range(4):
    print(f"\n[smoke] trial {trial}: env.reset() (seed pool round-robin)")
    obs, info = env.reset()
    applied = info.get("debug_seed_override")
    print(f"  debug_seed_override = {applied}")
    # Step through Neow to the MAP screen
    n_nodes, sig, screen, phase = take_until_map(env, max_steps=40)
    print(f"  post-stepping: screen={screen} phase={phase}")
    print(f"  map nodes: {n_nodes}")
    print(f"  map signature: {sig}")
    seen.append((applied, n_nodes, sig))

print("\n[smoke] compare (pool=2, round-robin): trial 0 vs 2 (same seed) should match, 0 vs 1 (diff seed) should differ:")
if len(seen) >= 4:
    print(f"  trial 0 sig: {seen[0][2]}")
    print(f"  trial 2 sig: {seen[2][2]}")
    print(f"  0==2 (same seed, expect same): {seen[0][2] == seen[2][2]}")
    print(f"  0==1 (diff seed, expect diff): {seen[0][2] == seen[1][2]}")

env.close()
print("\n[smoke] done.")
