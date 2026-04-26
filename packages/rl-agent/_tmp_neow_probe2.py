import sys
sys.path.insert(0, '/mnt/e/game/project/sts2_mcp/packages/rl-agent')
from sts2_env.env_v2 import SlayTheSpire2EnvV2
from sts2_env.headless_sim_bridge_client import HeadlessSimBridgeClient

SIM_EXE = '/mnt/e/game/project/sts2_mcp/third_party/sts2-ai/STS2AI/ENV/Sim/Host/bin/Debug/net9.0/headless_sim_host_0991.exe'

for neow_pick in [0, 1, 2]:
    print(f'=== Neow pick idx={neow_pick} ===')
    bridge = HeadlessSimBridgeClient(exe_path=SIM_EXE)
    env = SlayTheSpire2EnvV2(bridge=bridge, character='ironclad', stuck_watchdog_steps=0)
    obs, _ = env.reset()
    legal = env._legal_actions
    if neow_pick >= len(legal):
        print(f'  only {len(legal)} options, skipping'); env.close(); continue
    neow_aid = legal[neow_pick].get('action_id', '?')
    print(f'  t=0 pick: {neow_aid}')
    _, _, term, trunc, _ = env.step(neow_pick)

    last_fp = None
    stuck_count = 0
    for t in range(1, 30):
        raw = env._last_obs_raw or {}
        phase = raw.get('phase', '?')
        cs = raw.get('card_selection') or {}
        dec = raw.get('decision') or {}
        legal = env._legal_actions
        if not legal:
            print(f'  t={t}: no legal actions, break')
            break
        fp = (phase, cs.get('prompt', ''), dec.get('selected_count', 0), cs.get('can_confirm', False))
        if fp == last_fp:
            stuck_count += 1
        else:
            stuck_count = 0
            last_fp = fp
        kinds = [str(a.get('kind', '')) for a in legal[:3]]
        first_aid = str(legal[0].get('action_id', '') or '')
        confirm_at_0 = 'confirm' in first_aid.lower()
        prompt = cs.get('prompt', '')
        print(f'  t={t:2d} phase={phase[:12]:12s} prompt={prompt[:18]:18s} sel={dec.get("selected_count",0)}/{cs.get("max_select",0)} can_confirm={cs.get("can_confirm",False)} n_legal={len(legal)} conf_at_0={confirm_at_0} first_aid={first_aid[:40]}')
        _, _, term, trunc, _ = env.step(0)
        if term or trunc:
            print(f'  TERM/TRUNC at t={t}')
            break
        if stuck_count >= 5:
            print(f'  STUCK (fp unchanged for {stuck_count} steps), bailing')
            break
    else:
        print(f'  ran 30 steps without termination')
    env.close()
