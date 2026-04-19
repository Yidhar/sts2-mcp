# Phase 8 Tier 1+2 Smoke Hand-off

**Branch**: `feature/rl-phase3-powers` (HEAD = `becae61`)
**Task**: run 50k-step smoke on sim to validate the Phase 8 architecture before committing to an 8h long-train.

## TL;DR command

```bash
cd /mnt/e/game/project/sts2_mcp/packages/rl-agent
nohup bash launch_sim_phase8_smoke_50k.sh \
  > logs_attention/sim_phase8_smoke_50k_$(date +%Y%m%d_%H%M%S).stdout.log 2>&1 &
```

Expected runtime: **~30-45 minutes** (50k steps / 8 envs / ~25 it/s, plus Phase 8 token-budget overhead of ~15-20% vs Phase 6).

## What just shipped

Two commits on `feature/rl-phase3-powers`:
- `5946fa5` — Phase 8 Tier 1: action-history skeleton (20 step-detail + 8 turn-summary HISTORY tokens, new HISTORY world bank, history_card_bias in RelationBias, P0 card_selection translator fix, phase-stuck watchdog)
- `becae61` — Phase 8 Tier 2: causality aux head + pre/post state-delta vectors

Both together bring:
- `OBSERVATION_API_VERSION`: v3 → v4
- `MAX_WORLD_TOKENS`: 384 → 412 (+28 history tokens)
- `WORLD_BANK_NAMES`: adds `history` as 7th bank
- 8 aux heads (added `candidate_causality`, 8 dims: damage/block/hp_loss/draw/energy/strength/dex/vuln delta)
- New flags: `--stuck-watchdog-steps` (default 400), `--aux-causality-coef` (default 0.10)

## Success gates (decide at ~25k steps)

Look at TensorBoard + `monitor*.monitor.csv` + `reset_events.jsonl`. **All four must hold to proceed to 8h long-train.**

| Metric | Baseline (pre-Phase-8, post-watchdog) | Smoke target | Failure means |
|---|---|---|---|
| `train/aux_causality_loss` | N/A (head didn't exist) | nonzero at step 1k, trends down by 25k | head is dead weight — check target computation |
| `train/aux_enemy_state_loss` | 0.000 for 30/30 updates | >0.05 for >80% of updates in combat-heavy rollouts | rollouts still dominated by non-combat — stuck watchdog may need tightening |
| `train/value_loss` | ~1e-8 (broken) | >0.01 by 10k, trends toward 0.05-0.2 | rewards still not propagating — env bug |
| Monitor `r` distribution | 108/108 at exactly -3.4 | spread across at least 20 distinct values | reward shaping still stuck |
| `stuck_phase` in reset_events | 42% card_selection, 12% combat (pre-fix) | <15% combined | card_selection fix didn't stick OR new loop surfaced |

## Failure triage

### If `aux_causality_loss` stays flat or NaN
Check `sts2_env/action_history.py::_build_causality_delta` — the target values. Compare a known action:
```bash
./venv/Scripts/python.exe -c "
from sts2_env.action_history import _build_causality_delta
pre = {'combat': {'enemies': [{'hp': 30, 'max_hp': 40}], 'energy': 3, 'hand': [{}, {}]}, 'player': {'hp': 80, 'max_hp': 80}}
post = {'combat': {'enemies': [{'hp': 24, 'max_hp': 40}], 'energy': 2, 'hand': [{}]}, 'player': {'hp': 80, 'max_hp': 80}}
print(_build_causality_delta(pre, post))
# Expected: damage=0.2, draw=-0.2, energy=-0.33, others=0
"
```
If that's right, the problem is downstream — check `_select_aux_prediction` shape dispatch in `aux_maskable_ppo.py`.

### If episodes still hit exactly `-3.4` cumulative reward
Run the existing diagnostic probe that caught this before:
```bash
./venv/Scripts/python.exe probe_env_v2_reward_signal.py
```
If it shows consistent per-step nonzero rewards (it did last time we checked), but Monitor still shows -3.4, the bug is in the wrapper chain — check Monitor vs ActionMasker in `train_attention_policy.py::build_env_factory`.

### If `stuck_phase = card_selection` still >15%
The translator P0 fix should have knocked this down. Verify the fix is active:
```bash
./venv/Scripts/python.exe probe_card_selection_schema.py 2>&1 | head -30
```
Look for `confirm actions: >0` in the output when `can_confirm=true`. If still 0, the translator change reverted somehow.

### If throughput drops >25% vs Phase 6 (target 25-30 it/s)
Phase 8 adds 28 world tokens (384→412 = +7.3%) plus 1 new bank (7th cross-attention pass) plus 1 new aux head. Expected overhead 15-20%. If >25%:
1. Check `perf_stats.jsonl` — look for `request.step.runtime_ms` p95 creep
2. Consider dropping `MAX_STEP_DETAIL_TOKENS` from 20 to 12 in `action_history.py` (only affects recent-step granularity)
3. Check that `encode-pool-workers 2` is actually parallelizing — `rollout/python_obs_encode_p95_ms` should be <50ms

### If NaN or gradient explosion
Check that `_pad_state_dict_for_schema_growth` isn't being invoked (we're fresh-init, not warmstarting). The pad path has the "pair-bias re-keying" caveat noted in checkpoint.py — if somehow a v3 checkpoint got loaded, that would cause instability.

## Monitoring commands

### Live loss trends (run every ~5 min during training)
```bash
# Adjust RUN_NAME below to match your launch
RUN_NAME="sim_phase8_smoke_50k_*"
LOG_DIR=$(ls -td logs_attention/${RUN_NAME} | head -1)
echo "watching: ${LOG_DIR}"

awk -F',' 'NR==1 {
  for (i=1; i<=NF; i++) {
    h[i]=$i
    if ($i=="train/value_loss") vcol=i
    if ($i=="train/aux_causality_loss") ccol=i
    if ($i=="train/aux_enemy_state_loss") ecol=i
    if ($i=="train/explained_variance") evcol=i
  }
  next
} {
  print NR-1, "v="$vcol, "causal="$ccol, "enemy="$ecol, "ev="$evcol
}' "${LOG_DIR}/sb3_async/progress.csv" | tail -20
```

### Episode reward distribution
```bash
./venv/Scripts/python.exe -c "
import csv, collections
from pathlib import Path
import glob
logs = sorted(glob.glob('logs_attention/sim_phase8_smoke_50k_*/monitor*.csv'))
all_r = []
for f in logs:
    with open(f) as fh:
        lines = fh.readlines()
    for line in lines[2:]:
        parts = line.strip().split(',')
        try: all_r.append(float(parts[0]))
        except: pass
c = collections.Counter(round(r, 1) for r in all_r)
print(f'total episodes: {len(all_r)}')
print(f'unique (rounded 0.1) rewards: {len(c)}')
print(f'top 10 reward buckets:')
for r, n in c.most_common(10):
    print(f'  {r}: {n}')
"
```

### stuck_phase distribution (the card_selection fix health check)
```bash
./venv/Scripts/python.exe -c "
import json
import collections
from pathlib import Path
import glob
files = sorted(glob.glob('logs_attention/sim_phase8_smoke_50k_*/reset_events.jsonl'))
by_reason = collections.Counter()
stuck_phases = collections.Counter()
for f in files:
    with open(f) as fh:
        for line in fh:
            try: d = json.loads(line)
            except: continue
            if d.get('event') != 'episode_terminal': continue
            reason = d.get('truncation_reason') or 'natural'
            by_reason[reason] += 1
            if reason == 'phase_stuck_watchdog':
                stuck_phases[d.get('stuck_phase','?')] += 1
total = sum(by_reason.values())
print(f'total episodes: {total}')
for reason, n in by_reason.most_common():
    print(f'  {reason}: {n} ({n*100/max(total,1):.1f}%)')
print(f'stuck_phase breakdown:')
for phase, n in stuck_phases.most_common():
    print(f'  {phase}: {n}')
"
```

### Floor reached (what floor does the policy die/truncate at)
```bash
./venv/Scripts/python.exe -c "
import json
import collections
from pathlib import Path
import glob
files = sorted(glob.glob('logs_attention/sim_phase8_smoke_50k_*/reset_events.jsonl'))
floors = collections.Counter()
max_floors = []
for f in files:
    with open(f) as fh:
        for line in fh:
            try: d = json.loads(line)
            except: continue
            if d.get('event') != 'episode_terminal': continue
            mf = d.get('max_floor_reached', 0) or 0
            floors[int(mf)] += 1
            max_floors.append(int(mf))
print(f'floors reached distribution:')
for f, n in sorted(floors.items()):
    print(f'  floor {f}: {n}')
if max_floors:
    print(f'max={max(max_floors)}, mean={sum(max_floors)/len(max_floors):.1f}')
"
```

## TensorBoard

```bash
cd /mnt/e/game/project/sts2_mcp/packages/rl-agent
tensorboard --logdir logs_attention --port 6006
```

Scalars worth pinning:
- `train/value_loss`, `train/explained_variance`
- `train/aux_causality_loss`, `train/aux_enemy_state_loss`, `train/aux_objective_loss`
- `train/policy_gradient_loss`, `train/entropy_loss`
- `rollout/end_to_end_steps_per_s`, `rollout/collect_steps_per_s`
- `rollout/python_obs_encode_p50_ms`, `rollout/python_obs_encode_p95_ms`

## Decision matrix after smoke

| Situation | Next action |
|---|---|
| All 4 gates pass | Write `launch_sim_phase8_longtrain_8h.sh` (scale to 820k steps), warmstart from the final smoke checkpoint |
| Only causality gate fails | Lower `--aux-causality-coef` to 0.05 or investigate the target computation; re-smoke |
| stuck gate fails with new phase | Extend `_check_stuck_watchdog` fingerprint; iterate |
| Throughput <20 it/s | Drop `MAX_STEP_DETAIL_TOKENS` to 12, re-smoke (~15 min) |
| value_loss still ~1e-8 | Open: reward pipeline may need deeper debug — see failure triage above |

## Context if you need it

- Design doc: `_design_phase8_history.md` (the architecture you just built)
- Prior hand-off for Phase 6: `_handoff_phase6_longtrain.md`
- Relevant commit log: `git log --oneline -5`
- If something looks broken that wasn't broken before: `git diff becae61^ -- <file>` to see what Tier 2 changed

The sim Host exe path in the script is hard-coded to
`third_party/sts2-ai/STS2AI/ENV/Sim/Host/bin/Debug/net9.0/headless_sim_host_0991.exe`.
If that build is stale vs the in-repo C# sources, rebuild first:
```bash
cd third_party/sts2-ai/STS2AI/ENV/Sim/Host && dotnet build
```
