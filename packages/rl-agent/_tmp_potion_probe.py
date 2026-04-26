"""Probe: verify potion injection actually lands in policy-visible obs
and legal_actions. Runs on SIM (doesn't compete with live training).

Picks a chain that has potions in snapshot[0].potion_ids_before and
observes:
  - env.reset() obs.player.potions
  - legal_actions filtered for use_potion
"""
import sys
sys.path.insert(0, "/mnt/e/game/project/sts2_mcp/packages/rl-agent")

from sts2_env.combat_env import CombatSandboxEnv
from sts2_env.run_chained_combat_env import RunChainedCombatEnv
from sts2_env.headless_sim_bridge_client import HeadlessSimBridgeClient
from combat_snapshot_dataset import RunChainedSnapshotPool

SIM_EXE = "/mnt/e/game/project/sts2_mcp/third_party/sts2-ai/STS2AI/ENV/Sim/Host/bin/Debug/net9.0/headless_sim_host_0991.exe"
POOL_PATH = "/mnt/e/game/project/sts2_mcp/datasets/curated_combat_ironclad_mixed_provenance"
CURATED = "bootstrap_human_plus_local_all_roomwin_only_minus_combat_reset_failures"

print("[probe] loading pool ...")
pool = RunChainedSnapshotPool.from_path(
    POOL_PATH, curated_subset=CURATED, min_chain_length=3,
    quality_weights={"high_win": 0.25, "deep_act3_loss": 0.10, "mid_act2_loss": 0.25, "low_act1_loss": 0.40},
)

# Find a run whose snapshot[0] has potions (use internal _runs)
runs_with_potions = []
for run_id, rows in pool._runs.items():
    first = rows[0]
    pots = first.get("potion_ids_before") or []
    if pots:
        runs_with_potions.append((run_id, rows, pots))
print(f"[probe] runs with potion in snap[0]: {len(runs_with_potions)}/{len(pool._runs)}")

if not runs_with_potions:
    print("[probe] FAIL: no snapshot has potions in [0]. Check deeper snapshots.")
    for run_id, rows in list(pool._runs.items())[:5]:
        for i, r in enumerate(rows[:3]):
            pots = r.get("potion_ids_before") or []
            print(f"  run={run_id[:8]} snap[{i}]: potions={pots}")
    sys.exit(1)

# Pick first
rid, rows, snap0_pots = runs_with_potions[0]
print(f"[probe] selected run_id={rid[:12]}  snap[0].potions={snap0_pots}")
print(f"[probe] chain length={len(rows)}  encounters: {[r.get('encounter_id','?')[:30] for r in rows[:5]]}")

print(f"[probe] launching sim bridge ...")
bridge = HeadlessSimBridgeClient(exe_path=SIM_EXE)
inner = CombatSandboxEnv(bridge=bridge, character="ironclad")

# Manually inject: bypass the pool's random sampling, set chain directly
env = RunChainedCombatEnv(inner, run_pool=pool)
# Monkey-patch: force chain to our selected run
original_sample = pool.sample_run
pool.sample_run = lambda *a, **kw: list(rows)
try:
    print("[probe] calling env.reset() on selected chain...")
    obs, info = env.reset()
finally:
    pool.sample_run = original_sample

print(f"[probe] info keys: {sorted(k for k in info if 'run_chain' in k or 'snapshot' in k)}")
print(f"[probe] run_chain_length={info.get('run_chain_length')}  idx={info.get('run_chain_idx')}")

# Inspect policy-visible obs
raw = inner._last_obs_raw or {}
player = raw.get("player") or {}
pots_in_obs = player.get("potions") or []
print(f"[probe] obs.player.potions = {pots_in_obs}")
print(f"[probe] obs.player.hp = {player.get('hp')}/{player.get('max_hp')}")
print(f"[probe] _last_reset_kwargs.potions = {inner._last_reset_kwargs.get('potions')}")
print(f"[probe] _last_reset_kwargs keys: {sorted(inner._last_reset_kwargs.keys())}")
print(f"[probe] sandbox_supports_potions = {inner.sandbox_supports_potions}")
# Snapshot_row_to_reset_kwargs
from combat_snapshot_dataset import snapshot_row_to_reset_kwargs
test_kwargs = snapshot_row_to_reset_kwargs(rows[0], include_potions=True)
print(f"[probe] snapshot_row_to_reset_kwargs produced: potions={test_kwargs.get('potions')}")
print(f"[probe] snapshot raw potion_ids_before: {rows[0].get('potion_ids_before')}")
print(f"[probe] snapshot raw potion_state_known: {rows[0].get('potion_state_known')}")

# Check legal actions
legal = inner._legal_actions or []
use_potion_actions = [a for a in legal if "use_potion" in str(a.get("kind", "")).lower() or "use_potion" in str(a.get("action_id", "")).lower()]
print(f"[probe] legal_actions total: {len(legal)}")
print(f"[probe] use_potion actions in legal: {len(use_potion_actions)}")
for a in use_potion_actions[:5]:
    print(f"  -> {a.get('action_id')} kind={a.get('kind')}")

# Take a few steps and see if potions stay visible
for step in range(5):
    if not inner._legal_actions:
        break
    obs, _, term, trunc, info = env.step(0)
    raw = inner._last_obs_raw or {}
    player = raw.get("player") or {}
    pots = player.get("potions") or []
    print(f"[probe] t={step+1}: hp={player.get('hp')} potions={pots}")
    if term or trunc:
        print(f"[probe] TERMINAL at t={step+1}")
        break

env.close()
print("[probe] done.")
