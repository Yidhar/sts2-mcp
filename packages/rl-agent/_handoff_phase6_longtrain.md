# Phase 6 Long-Train Handoff

Second Claude instance: take this context, run the training, monitor metrics.

## Branch & commits

```
feature/rl-phase3-powers (git log --oneline -4)
  <6.4 commit pending if smoke green>
  da68999 policy: phase 6.2 + 6.3 — POWER bank routing + per-bucket bias
  8e47d5e obs: phase 6.0 + 6.1 — POWER_SLOT tokens (schema + emission)
  7121899 sim: fix 8 schema mismatches blocking full-run training + add encode pool
```

Working dir: `E:/game/project/sts2_mcp/packages/rl-agent`

## Launch command (8h budget, n_envs=8, ~820k steps)

```bash
cd /mnt/e/game/project/sts2_mcp/packages/rl-agent
# Defaults in the script: 8 envs, 819_200 steps, bf16 AMP, async collector.
# Fresh train — no --init-checkpoint (old 700k is schema-incompatible).
bash launch_sim_longtrain_20260419.sh
```

Override via env vars if needed: `STS2_N_ENVS=4 STS2_TOTAL_TIMESTEPS=200000 bash launch_sim_longtrain_20260419.sh`.

Output lands under `logs_attention/sim_fullrun_longtrain_<timestamp>/`.

## What changed since the 700k-steps-fail baseline

Every one of these was **silently producing zero** before — policy was training on zero-filled observations + flat -1.0 reward. Eight schema bugs, one architectural omission, all fixed:

1. `obs["player"]` flat dict now populated (HP/block/gold/deck_cards/relics/potions)
2. `in_combat` detection via `battle.player` presence (sim uses `state_type="monster"`, not "combat")
3. `run.floor` emitted (previously null; 6 readers keyed off it)
4. `combat.draw_pile/discard_pile/exhaust_pile` as card lists (previously counts only)
5. `route_summary` BFS from new sim `map.nodes` DTO
6. `card.effect_preview` from `content_registry.semantic_signals` (96% coverage on active hand cards)
7. Card compact key shape (`cost`/`upgrade_level`/`target`/`canonical_text`) matching bridge
8. `obs["decision"]` phase dict with option_count/can_skip/travelable_count/etc.
9. Action entry shape: nested `target`, flat event_option, `item` (not `shop_item`)

**Phase 6 architecture (beyond schema fixes):**
- POWER_SLOT_PLAYER / POWER_SLOT_ENEMY tokens — one per buff/debuff instance, effect-algebra vector in numeric slots
- CARD_KEYWORD_SLOT tokens — Retain/Ethereal/Exhaust/Innate/etc. per hand card
- Dedicated POWER world bank — top-k router can route candidate → buff context explicitly
- `power_bucket_bias` in RelationBias — per-power scalar bias per attention head (Vulnerable vs Strength vs Poison are distinguishable)
- Process-pool observation encoder (`--encode-pool-workers 8`, 7.4× parallel speedup)
- Floor tracking in Monitor CSV + episode_terminal JSONL

## Success criteria (vs 700k-fail baseline)

| Metric | 700k fail | Phase 6 target |
|---|---|---|
| `max_floor_reached` observed | 15 (rare) | ≥ 15 AND frequency > 5% |
| `avg_max_floor` | 7 | ≥ 10 |
| `aux_enemy_state_loss` | 0.000 forever | > 0, decreasing |
| `explained_variance` | ~0 | > 0.1 stable |
| `avg_reward` | -1.0 fixed | has variance |

## Monitor during training

Per-update metrics → `logs_attention/<run>/sb3_async/progress.csv`. Watch:
- `rollout/end_to_end_steps_per_s` — should be ~30 sps early, plateauing
- `train/value_loss` — should drop from ~0.5-1.0 to <0.1 over 100k steps
- `train/explained_variance` — key signal. Negative = value head broken; positive+growing = working
- `train/aux_enemy_state_loss` — sanity that enemy-tracking head has gradient
- `rollout/python_obs_encode_p95_ms` — watch for GIL regression (should stay <50ms at n=8 thanks to pool)

Per-episode JSONL → `logs_attention/<run>/reset_events.jsonl` (or similar). Each `episode_terminal` line has `max_floor_reached` / `current_floor` / `reward` / `truncation_reason`.

Monitor CSV → `logs_attention/<run>/monitor.monitor.csv` has r,l,t + `max_floor_reached` + `current_floor` per episode.

Quick watchdog commands:
```bash
# Live steps/sec + floor distribution (every 5 min)
watch -n 60 'tail -3 logs_attention/sim_fullrun_longtrain_*/sb3_async/progress.csv | column -s, -t'

# Floor histogram from monitor CSV
python -c "
import pandas as pd, glob
csv = glob.glob('logs_attention/sim_fullrun_longtrain_*/monitor.monitor.csv')[-1]
df = pd.read_csv(csv, skiprows=1)
print(df['max_floor_reached'].value_counts().sort_index())
print(f'mean={df.max_floor_reached.mean():.1f} max={df.max_floor_reached.max()}')
"
```

## Failure-mode triage

- **Sim crash / hang**: check `/tmp/sim_hang_debug_pid*.log` for the per-sim hang watchdog output. v11 fix eliminated the main pattern; any new hang is probably a novel sim bug.
- **Worker restarts > 5**: `logs_attention/<run>/async_restart_events.jsonl` — check if one sim is repeatedly dying vs global instability.
- **aux losses all zero**: means a schema regression slipped in. Grep `obs_encoder.encode(` in env_v2.py and check obs["player"] / obs["combat"]["enemies"] are populated by hand in a fresh Python repl.
- **explained_variance stuck at 0**: reward signal still degenerate somehow. Check per-step reward distribution (not just episode-end).
- **OOM on GPU**: drop `--n-envs` to 4 via `STS2_N_ENVS=4`. Memory scales linearly.

## After training completes

1. Final checkpoint at `checkpoints_attention/sim_fullrun_longtrain_<ts>/final/`
2. Run eval: `python evaluate_attention_policy.py <ckpt_dir> --episodes 32 --combat-sandbox --device cuda`
3. Report: avg_max_floor, max_floor_reached distribution, reward distribution, compared to 700k baseline table above.

## Rollback plan if it fails worse than 700k

- `git switch feature/rl-phase2-attention` — stops at commit 7121899 (just schema fixes, no POWER_SLOT arch)
- Re-run long-train there — isolates whether Phase 6 architecture hurt vs schema fix alone was enough
