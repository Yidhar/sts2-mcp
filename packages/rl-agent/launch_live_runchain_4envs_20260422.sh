#!/usr/bin/env bash
# Live 4-instance combat-sandbox continuation in RUN-CHAIN mode.
#
# Each training episode replays one human run's full combat sequence
# (ordered by path_index) as a single chain. HP and potion carry over
# from sub-combat i to sub-combat i+1 under these rules:
#   deck + relic + gold + max_hp + max_energy + encounter_id: from snapshot[i]
#   current_hp  = min(snapshot[i].hp, policy_end_hp_from_last_combat)
#   potions     = policy_remaining + (snapshot[i].potions - snapshot[i-1].potions)
#                 capped at 3 slots
#
# Run selection weighted toward boss-containing runs:
#   high_win=0.25, deep_act3_loss=0.10, mid_act2_loss=0.25, low_act1_loss=0.40
# -> ~60% of sampled episodes include a boss combat (vs 1.1% raw run-level).
#
# Terminal:
#   player_hp <= 0 at any sub-combat -> loss
#   all snapshots played and survived -> win
#   +5.0 bonus if last sub-combat was a boss and player survived (run_complete)
#
# Init from the current-best clean checkpoint. See decision doc for why
# we pick this one: last Phase 10-ish live-trained checkpoint that isn't
# contaminated by BUG-D starter-only distribution (was identified as
# step_001710080 or latest live-sanity if better).
set -euo pipefail

cd '/mnt/e/game/project/sts2_mcp/packages/rl-agent'
export PYTORCH_ALLOC_CONF=expandable_segments:True

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="live_runchain_4envs_${STAMP}"
LOG_DIR="/mnt/e/game/project/sts2_mcp/packages/rl-agent/logs_attention/${RUN_NAME}"
CKPT_DIR="/mnt/e/game/project/sts2_mcp/packages/rl-agent/checkpoints_attention/${RUN_NAME}"
STATUS_FILE="${LOG_DIR}/async_status.json"
ANALYSIS_DIR="/mnt/e/game/project/sts2_mcp/packages/rl-agent/analysis/training_curves"
mkdir -p "${LOG_DIR}" "${CKPT_DIR}" "${ANALYSIS_DIR}"

# Latest live-trained clean checkpoint (post BUG-D fix, 4125-ep live sandbox).
# Overall training WR 50.1%, boss WR 4.9% avg (5/11 bosses 5-9%, 3/11 still 0%).
INIT_CKPT='/mnt/e/game/project/sts2_mcp/packages/rl-agent/checkpoints_attention/live_sanity_4envs_20260421_235123/step_001812480'

SESSIONS='/mnt/c/Users/yidhar/AppData/Roaming/SlayTheSpire2/bridge/session_0.json,/mnt/c/Users/yidhar/AppData/Roaming/SlayTheSpire2/bridge/session_1.json,/mnt/c/Users/yidhar/AppData/Roaming/SlayTheSpire2/bridge/session_2.json,/mnt/c/Users/yidhar/AppData/Roaming/SlayTheSpire2/bridge/session_3.json'

# 1812480 + 204800 = 2017280. Room for ~2k run-chain episodes.
# Run-chain episodes are 3-28 sub-combats (mean ~3.85), ~100-800 env steps
# each. With 4 envs x 64 n_steps = 256 env steps per rollout = 0.3-2.5
# chain episodes per rollout. Much higher per-update signal density than
# flat sandbox.
TOTAL_STEPS=2017280

echo "[runchain] RUN_NAME=${RUN_NAME}"
echo "[runchain] INIT_CKPT=${INIT_CKPT}"
echo "[runchain] live bridges: 4 instances session_0..3"
echo "[runchain] chain source: curated_combat_ironclad_mixed_provenance / bootstrap_human_plus_local_all_roomwin_only_minus_combat_reset_failures"
echo "[runchain] expected chain length: 3-28 sub-combats, mean ~4"
echo "[runchain] expected boss-episode fraction: ~60 percent"

./.venv-wsl-rocm/bin/python -u ./train_attention_policy.py \
  --init-checkpoint "${INIT_CKPT}" \
  --total-timesteps ${TOTAL_STEPS} \
  --n-envs 4 \
  --session-files "${SESSIONS}" \
  --collector-mode async \
  --combat-sandbox \
  --character ironclad \
  --snapshot-pool '/mnt/e/game/project/sts2_mcp/datasets/curated_combat_ironclad_mixed_provenance' \
  --snapshot-curated-subset bootstrap_human_plus_local_all_roomwin_only_minus_combat_reset_failures \
  --run-chain-mode \
  --run-chain-min-length 3 \
  --run-chain-quality-weights 'high_win=0.25,deep_act3_loss=0.10,mid_act2_loss=0.25,low_act1_loss=0.40' \
  --device cuda \
  --text-device cuda \
  --no-text \
  --amp --amp-dtype bf16 \
  --n-steps 64 \
  --batch-size 16 \
  --n-epochs 1 \
  --reset-timeout-ms 120000 \
  --step-timeout-ms 60000 \
  --checkpoint-interval-timesteps 10240 \
  --checkpoint-keep-last 12 \
  --stuck-watchdog-steps 1500 \
  --aux-enemy-state-coef 0.10 \
  --aux-causality-coef 0.10 \
  --perf-stats-interval-steps 500 \
  --log-dir "${LOG_DIR}" \
  --checkpoint-dir "${CKPT_DIR}" \
  --status-file "${STATUS_FILE}"

echo "[runchain] training finished."
echo "[runchain] logs=${LOG_DIR} ckpts=${CKPT_DIR}"
