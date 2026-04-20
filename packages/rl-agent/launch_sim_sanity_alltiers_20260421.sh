#!/usr/bin/env bash
# Phase 9 sanity — clean-pipeline continuation from post-curriculum checkpoint.
#
# Resumes from sim_p0pp_losers_20260420/step_001320960 (the 14% WR
# post-curriculum asset measured on clean live eval) and trains another
# 102400 combat-sandbox steps under the bugs-fixed reward pipeline:
#   - FIX 1: _enemy_hp_delta_reward no longer emits +5.66 false kill at
#            defeat terminal (combat_env.py, env_v2.py)
#   - FIX 2: HeadlessSimBridgeClient emits live-parity sandbox terminal
#            reward breakdown instead of flat -1.0 (+0.29 → +3.5 mag on loss)
#   - FIX 3: eval_combat_sandbox_boss.py classifier uses player_hp instead
#            of reward_sum > 0 (only affects eval, not training)
#
# Purpose: answer "does clean reward continue the learning, or did the
# policy internalize reward-hack behaviors that collapse under accurate
# gradients?" Expected outcomes:
#   - WR stays 14-20% on 12-boss eval → clean reward preserved learning
#   - WR climbs to >20% → policy was under-regularized, now converging
#   - WR drops to <10% → reward-hack dependence, needs rollback
#
# Pool: full mixed-tier to prevent narrow boss-only overfit from the
# preceding P-0'' 7-losers curriculum. User-recommended config:
#   all tiers enabled, default weights (no weight override).
#
# Sizing: 102400 = 200 rollouts × (8 n_envs × 64 n_steps). ~1h on sim.
# Eval cadence: 50-ep 12-boss sim eval at steps 30720, 61440, 102400
# (all tiers=boss, deterministic) + one final live 50-ep for ground truth.
set -euo pipefail

cd '/mnt/e/game/project/sts2_mcp/packages/rl-agent'
export PYTORCH_ALLOC_CONF=expandable_segments:True

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="sim_sanity_alltiers_${STAMP}"
LOG_DIR="/mnt/e/game/project/sts2_mcp/packages/rl-agent/logs_attention/${RUN_NAME}"
CKPT_DIR="/mnt/e/game/project/sts2_mcp/packages/rl-agent/checkpoints_attention/${RUN_NAME}"
STATUS_FILE="${LOG_DIR}/async_status.json"
ANALYSIS_DIR="/mnt/e/game/project/sts2_mcp/packages/rl-agent/analysis/training_curves"
SESSION0='/mnt/c/Users/yidhar/AppData/Roaming/SlayTheSpire2/bridge/session.json'
mkdir -p "${LOG_DIR}" "${CKPT_DIR}" "${ANALYSIS_DIR}"

INIT_CKPT='/mnt/e/game/project/sts2_mcp/packages/rl-agent/checkpoints_attention/sim_p0pp_losers_20260420_182657/step_001320960'

echo "[sanity] RUN_NAME=${RUN_NAME}"
echo "[sanity] INIT_CKPT=${INIT_CKPT}"
echo "[sanity] baseline: 14.0% WR (7/50) on 12-boss live eval"
echo "[sanity] gate: stable 14%+ on 50-ep sim eval every ~30k steps"
echo "[sanity] starting 102400 combat-sandbox steps with clean pipeline..."

./.venv-wsl-rocm/bin/python -u ./train_attention_policy.py \
  --use-sim \
  --sim-exe-path '/mnt/e/game/project/sts2_mcp/third_party/sts2-ai/STS2AI/ENV/Sim/Host/bin/Debug/net9.0/headless_sim_host_0991.exe' \
  --init-checkpoint "${INIT_CKPT}" \
  --total-timesteps 102400 \
  --n-envs 8 \
  --encode-pool-workers 2 \
  --collector-mode async \
  --combat-sandbox \
  --character ironclad \
  --snapshot-pool '/mnt/e/game/project/sts2_mcp/datasets/curated_combat_ironclad_mixed_provenance' \
  --snapshot-curated-subset bootstrap_human_plus_local_all_roomwin_only_minus_combat_reset_failures \
  --snapshot-sample-mode encounter_balanced \
  --device cuda \
  --text-device cuda \
  --no-text \
  --amp --amp-dtype bf16 \
  --n-steps 64 \
  --batch-size 16 \
  --n-epochs 1 \
  --reset-timeout-ms 60000 \
  --step-timeout-ms 30000 \
  --checkpoint-interval-timesteps 10240 \
  --checkpoint-keep-last 12 \
  --stuck-watchdog-steps 400 \
  --aux-enemy-state-coef 0.10 \
  --aux-causality-coef 0.10 \
  --perf-stats-interval-steps 500 \
  --log-dir "${LOG_DIR}" \
  --checkpoint-dir "${CKPT_DIR}" \
  --status-file "${STATUS_FILE}"

# Post-training eval sweep: track WR across 3 intermediate checkpoints
# plus the final one, all under the clean pipeline. 50-ep 12-boss on sim
# for quick iteration, then one live 50-ep on the final to ground-truth.
# Checkpoint names reflect CUMULATIVE training steps — resume from 1320960
# means +30720 step checkpoint is named step_001351680, not step_000030720.
echo "[sanity] training finished, running post-training eval sweep..."

INIT_STEP=1320960
EVAL_STEPS=(30720 61440 92160 102400)
for DELTA in "${EVAL_STEPS[@]}"; do
  ABS=$((INIT_STEP + DELTA))
  CKPT="${CKPT_DIR}/step_$(printf '%09d' ${ABS})"
  if [[ ! -d "${CKPT}" ]]; then
    echo "[sanity] skipping +${DELTA} (abs ${ABS}) — checkpoint missing"
    continue
  fi
  OUT="${ANALYSIS_DIR}/${RUN_NAME}_sim_eval_delta${DELTA}.json"
  echo "[sanity] sim eval @ +${DELTA} (abs ${ABS}) → ${OUT}"
  ./.venv-wsl-rocm/bin/python -u ./eval_combat_sandbox_boss.py \
    "${CKPT}" \
    --snapshot-pool '/mnt/e/game/project/sts2_mcp/datasets/curated_combat_ironclad_mixed_provenance' \
    --curated-subset bootstrap_human_plus_local_all_roomwin_only_minus_combat_reset_failures \
    --tiers boss \
    --episodes 50 \
    --device cpu --no-text --deterministic --use-sim \
    --output-json "${OUT}"
done

# Ground-truth final checkpoint on live
FINAL_ABS=$((INIT_STEP + 102400))
FINAL_CKPT="${CKPT_DIR}/step_$(printf '%09d' ${FINAL_ABS})"
if [[ -d "${FINAL_CKPT}" ]]; then
  LIVE_OUT="${ANALYSIS_DIR}/${RUN_NAME}_live_eval_final.json"
  echo "[sanity] live eval @ final → ${LIVE_OUT}"
  ./.venv-wsl-rocm/bin/python -u ./eval_combat_sandbox_boss.py \
    "${FINAL_CKPT}" \
    --snapshot-pool '/mnt/e/game/project/sts2_mcp/datasets/curated_combat_ironclad_mixed_provenance' \
    --curated-subset bootstrap_human_plus_local_all_roomwin_only_minus_combat_reset_failures \
    --tiers boss \
    --episodes 50 \
    --device cpu --no-text --deterministic \
    --session-file "${SESSION0}" \
    --output-json "${LIVE_OUT}"
fi

echo "[sanity] done."
echo "[sanity] logs:  ${LOG_DIR}"
echo "[sanity] ckpts: ${CKPT_DIR}"
echo "[sanity] eval artifacts: ${ANALYSIS_DIR}/${RUN_NAME}_*.json"
