"""Probe what enemy data looks like for CEREMONIAL_BEAST_BOSS combat
to understand why the ceremonial mechanism detector never fires."""
import json
import sys
import time

sys.path.insert(0, "/mnt/e/game/project/sts2_mcp/packages/rl-agent")

from sts2_env.bridge_client import BridgeClient
from sts2_env.boss_mechanics import (
    build_boss_mechanics_context,
    enemy_mechanics_key,
    _enemy_threshold_value,
    _enemy_search_texts,
)

SESSION = "/mnt/c/Users/yidhar/AppData/Roaming/SlayTheSpire2/bridge/session_0.json"

bridge = BridgeClient(session_path=SESSION)

# Wait for bridge
for _ in range(30):
    try:
        bridge.get_spec()
        break
    except Exception:
        time.sleep(2)

print(f"[probe] combat_reset for ENCOUNTER.CEREMONIAL_BEAST_BOSS")
result = bridge.combat_reset(
    character="ironclad",
    encounter_id="ENCOUNTER.CEREMONIAL_BEAST_BOSS",
    timeout_ms=20000,
)

obs = result.get("obs") or {}
combat = obs.get("combat") or {}
enemies = combat.get("enemies") or []
print(f"[probe] num enemies: {len(enemies)}")

for i, enemy in enumerate(enemies):
    print(f"\n=== enemy {i} ===")
    print(f"  name        = {enemy.get('name')!r}")
    print(f"  model_id    = {enemy.get('model_id')!r}")
    print(f"  id          = {enemy.get('id')!r}")
    print(f"  hp          = {enemy.get('hp')!r}")
    print(f"  max_hp      = {enemy.get('max_hp')!r}")
    print(f"  is_alive    = {enemy.get('is_alive')!r}")
    intent = enemy.get("intent") or {}
    print(f"  intent.title= {intent.get('title')!r}")
    print(f"  intent.total_damage = {intent.get('total_damage')!r}")
    print(f"  intent.intent_type = {intent.get('intent_type')!r}")
    powers = enemy.get("powers") or []
    print(f"  powers (n={len(powers)}):")
    for p in powers[:8]:
        print(f"    - {p.get('title')!r} amt={p.get('amount')!r} stacks={p.get('stacks')!r}")
    pr = enemy.get("phase_rules") or []
    print(f"  phase_rules (n={len(pr)}):")
    for r in pr[:5]:
        print(f"    - {r}")
    rt = enemy.get("reactive_triggers") or []
    print(f"  reactive_triggers (n={len(rt)}):")
    for r in rt[:5]:
        print(f"    - {r}")
    st = enemy.get("static_traits") or []
    print(f"  static_traits (n={len(st)}):")
    for r in st[:5]:
        print(f"    - {r}")

    print(f"  --- detector inputs ---")
    print(f"  enemy_mechanics_key = {enemy_mechanics_key(enemy, '')!r}")
    print(f"  threshold_value     = {_enemy_threshold_value(enemy)!r}")
    print(f"  search_texts (sample 5): {_enemy_search_texts(enemy, encounter_key='ENCOUNTER.CEREMONIAL_BEAST_BOSS')[:5]}")

print(f"\n=== full mechanics context ===")
ctx = build_boss_mechanics_context(obs)
# print only ceremonial-relevant
print(f"  encounter_key  = {ctx.get('encounter_key')!r}")
print(f"  one_card_lock  = {ctx.get('one_card_lock')!r}")
print(f"  stun_window    = {ctx.get('stun_window')!r}")
print(f"  threshold_active = {ctx.get('threshold_active')!r}")
print(f"  threshold_value_norm = {ctx.get('threshold_value_norm')!r}")
print(f"  boss_special_active = {ctx.get('boss_special_active')!r}")

# Look into enemy_states_by_index — that's where ceremonial state actually lives
enemy_states = ctx.get("enemy_states_by_index") or []
print(f"\n=== enemy_states_by_index (n={len(enemy_states)}) ===")
for i, st in enumerate(enemy_states):
    print(f"--- enemy_state[{i}] (ceremonial-relevant) ---")
    for key in ["one_card_lock", "stun_window", "threshold_active",
                "threshold_value_norm", "boss_special_active",
                "transform_pending", "incoming_damage_multiplier_norm"]:
        v = st.get(key)
        print(f"  {key} = {v!r}")
    # all non-zero
    print(f"--- enemy_state[{i}] all non-zero fields ---")
    for k, v in sorted(st.items()):
        if isinstance(v, (int, float)) and abs(float(v)) > 1e-9:
            print(f"  {k} = {v}")
    print(f"--- enemy_state[{i}] all keys (count={len(st)}) ---")
    print(f"  keys sample: {sorted(list(st.keys()))[:25]}")

print(f"\n=== player_state ceremonial-relevant ===")
ps = ctx.get("player_state") or {}
for k in ["ringing_active", "play_budget_lock", "back_attack_risk"]:
    print(f"  {k} = {ps.get(k)!r}")

# Now simulate _boss_context_max behavior
def boss_max(ctx, key):
    vals = []
    ps = ctx.get("player_state") or {}
    if key in ps:
        vals.append(float(ps.get(key) or 0.0))
    for st in (ctx.get("enemy_states_by_index") or []):
        if key in st:
            vals.append(float(st.get(key) or 0.0))
    return max(vals) if vals else 0.0

print(f"\n=== _boss_context_max simulation ===")
for k in ["one_card_lock", "stun_window", "threshold_active", "back_attack_risk", "incoming_damage_multiplier_norm"]:
    print(f"  {k} -> {boss_max(ctx, k)}")

