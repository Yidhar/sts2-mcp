"""Smoke test RunChainedCombatEnv on live game: run 1 full chain end-to-end.

Drives policy with idx=0 (dumb) so we only care about the env wrapper mechanics:
  - sub-combat transitions work
  - HP carry applied correctly
  - Potion carry applied correctly
  - Terminal fires on chain exhaustion or player death
  - No crashes/hangs
"""
import sys

sys.path.insert(0, "/mnt/e/game/project/sts2_mcp/packages/rl-agent")

from pathlib import Path

from sts2_env.bridge_client import BridgeClient
from sts2_env.combat_env import CombatSandboxEnv
from sts2_env.run_chained_combat_env import RunChainedCombatEnv
from combat_snapshot_dataset import RunChainedSnapshotPool

POOL_PATH = "/mnt/e/game/project/sts2_mcp/datasets/curated_combat_ironclad_mixed_provenance"
CURATED = "bootstrap_human_plus_local_all_roomwin_only_minus_combat_reset_failures"
SESSION = "/mnt/c/Users/yidhar/AppData/Roaming/SlayTheSpire2/bridge/session_3.json"

print(f"[smoke] loading snapshot pool ...")
pool = RunChainedSnapshotPool.from_path(
    POOL_PATH,
    curated_subset=CURATED,
    min_chain_length=3,
    quality_weights={
        "high_win": 0.25,
        "deep_act3_loss": 0.10,
        "mid_act2_loss": 0.25,
        "low_act1_loss": 0.40,
    },
)
print(f"[smoke] pool summary: {pool.summary()}")

print(f"[smoke] opening bridge session: {SESSION}")
bridge = BridgeClient(session_path=SESSION)
inner = CombatSandboxEnv(bridge=bridge, character="ironclad")
env = RunChainedCombatEnv(inner, run_pool=pool)

print(f"[smoke] reset()...")
obs, info = env.reset()
chain_len = info.get("run_chain_length")
print(f"[smoke] started chain, length={chain_len}, sub_combat_idx={info.get('run_chain_sub_combat_index', 0)}")
if chain_len:
    encs = [c.get("encounter_id") for c in env._chain]
    print(f"[smoke] encounters in chain: {encs}")

t = 0
MAX_STEPS = 2000
sub_combat_count = 0
while t < MAX_STEPS:
    legal = inner._legal_actions if hasattr(inner, "_legal_actions") else None
    if not legal:
        print(f"  t={t}: no legal actions, breaking")
        break
    obs, rew, term, trunc, info = env.step(0)
    t += 1
    current_idx = info.get("run_chain_sub_combat_index", 0)
    if current_idx != sub_combat_count:
        print(f"  t={t}: advanced to sub_combat {current_idx}/{chain_len}")
        sub_combat_count = current_idx
    if t % 50 == 0:
        raw = inner._last_obs_raw or {}
        player = raw.get("player") or {}
        hp = player.get("hp") or player.get("current_hp")
        print(f"  t={t}: sub_combat={current_idx} hp={hp} rew_cum={env._chain_reward_accum:.2f}")
    if term or trunc:
        print(f"[smoke] TERMINAL at t={t}: term={term} trunc={trunc}")
        print(f"  chain_reward_accum: {env._chain_reward_accum:.2f}")
        print(f"  sub_combat_outcomes: {env._chain_sub_combat_outcomes}")
        print(f"  sub_combat_rewards:  {[round(r,2) for r in env._chain_sub_combat_rewards]}")
        print(f"  info keys: {sorted(info.keys())}")
        break
else:
    print(f"[smoke] reached MAX_STEPS={MAX_STEPS} without termination")

env.close()
print("[smoke] done.")
