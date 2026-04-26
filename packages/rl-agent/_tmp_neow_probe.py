import sys
sys.path.insert(0, '/mnt/e/game/project/sts2_mcp/packages/rl-agent')
from sts2_env.env_v2 import SlayTheSpire2EnvV2
from sts2_env.headless_sim_bridge_client import HeadlessSimBridgeClient

SIM_EXE = '/mnt/e/game/project/sts2_mcp/third_party/sts2-ai/STS2AI/ENV/Sim/Host/bin/Debug/net9.0/headless_sim_host_0991.exe'

for trial in range(5):
    bridge = HeadlessSimBridgeClient(exe_path=SIM_EXE)
    env = SlayTheSpire2EnvV2(bridge=bridge, character='ironclad', stuck_watchdog_steps=0)
    obs, _ = env.reset()
    reached_combat = False
    for t in range(60):
        legal = env._legal_actions
        if not legal:
            print(f'  trial {trial} t={t}: no legal actions, breaking')
            break
        _, reward, term, trunc, _ = env.step(0)
        raw = env._last_obs_raw or {}
        phase = raw.get('phase', '?')
        screen = raw.get('screen', '?')
        cs = raw.get('card_selection') or {}
        dec = raw.get('decision') or {}
        run = raw.get('run') or {}
        floor = run.get('floor')
        if phase == 'combat' and floor and int(floor) > 1:
            print(f'  trial {trial}: reached combat at floor {floor}, t={t}')
            reached_combat = True
            break
        if t % 5 == 0 or t < 15:
            aid = legal[0].get('action_id', '?') if legal else '?'
            prompt = cs.get('prompt', '')
            sel_count = dec.get('selected_count', 0)
            can_confirm = cs.get('can_confirm', False)
            print(f'  trial {trial} t={t:2d}: phase={phase} screen={screen} floor={floor} prompt={prompt!r} sel={sel_count} can_confirm={can_confirm} legal_n={len(legal)} act={str(aid)[:50]}')
        if term or trunc:
            print(f'  trial {trial}: terminated at t={t}')
            break
    if not reached_combat:
        print(f'  trial {trial}: NOT reached combat in 60 steps')
    env.close()
