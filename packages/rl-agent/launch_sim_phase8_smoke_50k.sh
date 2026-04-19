#!/usr/bin/env bash
# Phase 8 Tier 1 + Tier 2 smoke test: 50k steps from fresh init on sim.
#
# Why fresh (no warmstart):
#   Tier 2 destructively changed the HISTORY_STEP_DETAIL numeric layout
#   (slots 67..83 swapped from scalar deltas to pre/post state vecs). A
#   v3 checkpoint would load via the pad-migration path but the old
#   weights at those slots would be semantically wrong, masking whether
#   the arch is actually learning. Pure-fresh takes ~30 min and gives
#   a clean signal.
#
# What we're testing:
#   1. Does the new HISTORY bank + candidate_causality head train at all
#      (no NaN / gradient explosion / collapse)
#   2. Does aux_causality_loss show a real downward trend (head is
#      actually learning card-effect predictions)
#   3. Does aux_enemy_state_loss stay nonzero (rolled out of the Phase 6
#      zero-collapse pathology by the watchdog + translator fixes)
#   4. Does the phase-stuck watchdog catch remaining loops at a lower
#      rate than before (target: <15% of episodes truncated by watchdog,
#      down from 42% pre-P0-fix)
#   5. Does value_loss stop being ~1e-8 (i.e. rewards actually vary)
#
# Runtime estimate:
#   50000 steps / 8 envs / ~25-30 it/s per env ≈ 30-45 min. Add 5-10 min
#   for sim subprocess spin-up + Phase 8 token-budget overhead (15-20%
#   slower per step than Phase 6).
set -euo pipefail

cd '/mnt/e/game/project/sts2_mcp/packages/rl-agent'
export PYTORCH_ALLOC_CONF=expandable_segments:True

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="sim_phase8_smoke_50k_${STAMP}"
LOG_DIR="/mnt/e/game/project/sts2_mcp/packages/rl-agent/logs_attention/${RUN_NAME}"
CKPT_DIR="/mnt/e/game/project/sts2_mcp/packages/rl-agent/checkpoints_attention/${RUN_NAME}"
STATUS_FILE="${LOG_DIR}/async_status.json"
mkdir -p "${LOG_DIR}" "${CKPT_DIR}"

N_ENVS="${STS2_N_ENVS:-8}"
TOTAL_TIMESTEPS="${STS2_TOTAL_TIMESTEPS:-50000}"

echo "[phase8_smoke] RUN_NAME=${RUN_NAME}"
echo "[phase8_smoke] N_ENVS=${N_ENVS}"
echo "[phase8_smoke] TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS}"
echo "[phase8_smoke] LOG_DIR=${LOG_DIR}"
echo "[phase8_smoke] CKPT_DIR=${CKPT_DIR}"
echo "[phase8_smoke] starting Phase 8 Tier 1+2 smoke (no warmstart)..."

exec ./.venv-wsl-rocm/bin/python -u ./train_attention_policy.py \
  --use-sim \
  --sim-exe-path '/mnt/e/game/project/sts2_mcp/third_party/sts2-ai/STS2AI/ENV/Sim/Host/bin/Debug/net9.0/headless_sim_host_0991.exe' \
  --total-timesteps "${TOTAL_TIMESTEPS}" \
  --n-envs "${N_ENVS}" \
  --encode-pool-workers 2 \
  --collector-mode async \
  --character ironclad \
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
  --checkpoint-keep-last 5 \
  --stuck-watchdog-steps 400 \
  --aux-enemy-state-coef 0.10 \
  --aux-causality-coef 0.10 \
  --perf-stats-interval-steps 500 \
  --log-dir "${LOG_DIR}" \
  --checkpoint-dir "${CKPT_DIR}" \
  --status-file "${STATUS_FILE}"
