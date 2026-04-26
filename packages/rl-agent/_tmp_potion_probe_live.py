"""Live-game version of potion probe. Uses session_0.json BridgeClient
instead of HeadlessSimBridgeClient. Compare with sim probe result.
"""
import sys
sys.path.insert(0, "/mnt/e/game/project/sts2_mcp/packages/rl-agent")

from sts2_env.bridge_client import BridgeClient
from sts2_env.combat_env import CombatSandboxEnv
from sts2_env.run_chained_combat_env import RunChainedCombatEnv
from combat_snapshot_dataset import RunChainedSnapshotPool, snapshot_row_to_reset_kwargs

POOL_PATH = "/mnt/e/game/project/sts2_mcp/datasets/curated_combat_ironclad_mixed_provenance"
CURATED = "bootstrap_human_plus_local_all_roomwin_only_minus_combat_reset_failures"
SESSION = "/mnt/c/Users/yidhar/AppData/Roaming/SlayTheSpire2/bridge/session_0.json"

print("[probe-live] loading pool ...")
pool = RunChainedSnapshotPool.from_path(
    POOL_PATH, curated_subset=CURATED, min_chain_length=3,
    quality_weights={"high_win": 0.25, "deep_act3_loss": 0.10, "mid_act2_loss": 0.25, "low_act1_loss": 0.40},
)
runs_with_potions = [
    (rid, rows) for rid, rows in pool._runs.items()
    if (rows[0].get("potion_ids_before") or [])
]
print(f"[probe-live] runs with potion in snap[0]: {len(runs_with_potions)}/{len(pool._runs)}")

rid, rows = runs_with_potions[0]
print(f"[probe-live] selected run_id={rid[:12]}  snap[0].potions={rows[0].get('potion_ids_before')}")
print(f"[probe-live] chain length={len(rows)}")

print(f"[probe-live] opening live bridge: {SESSION}")
bridge = BridgeClient(session_path=SESSION)
inner = CombatSandboxEnv(bridge=bridge, character="ironclad")
env = RunChainedCombatEnv(inner, run_pool=pool)

original = pool.sample_run
pool.sample_run = lambda *a, **kw: list(rows)
try:
    print("[probe-live] env.reset() ...")
    obs, info = env.reset()
finally:
    pool.sample_run = original

raw = inner._last_obs_raw or {}
player = raw.get("player") or {}
pots_in_obs = player.get("potions") or []
legal = inner._legal_actions or []
use_potion_actions = [a for a in legal if "use_potion" in str(a.get("kind", "")).lower() or "use_potion" in str(a.get("action_id", "")).lower()]

print(f"[probe-live] run_chain_length={info.get('run_chain_length')}")
print(f"[probe-live] obs.player.potions = {pots_in_obs}")
print(f"[probe-live] obs.player.hp = {player.get('hp')}/{player.get('max_hp')}")
print(f"[probe-live] _last_reset_kwargs.potions = {inner._last_reset_kwargs.get('potions')}")
print(f"[probe-live] sandbox_supports_potions = {inner.sandbox_supports_potions}")
print(f"[probe-live] legal_actions total: {len(legal)}")
print(f"[probe-live] use_potion actions in legal: {len(use_potion_actions)}")
for a in use_potion_actions[:5]:
    print(f"  -> {a.get('action_id')} kind={a.get('kind')}")

# Few steps
for step in range(5):
    if not inner._legal_actions:
        break
    _, _, term, trunc, _ = env.step(0)
    raw = inner._last_obs_raw or {}
    player = raw.get("player") or {}
    pots = player.get("potions") or []
    print(f"[probe-live] t={step+1}: hp={player.get('hp')} potions={pots}")
    if term or trunc:
        break

env.close()
print("[probe-live] done.")
