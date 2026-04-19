#!/usr/bin/env bash
# Phase 8 Tier 1+2 long training: 820k steps on sim, warmstarted from
# the 10k smoke that just validated.
#
# Why warmstart not fresh:
#   The 10k smoke with commit cd791a0 got to mean_floor=7.73 / max_floor=17
#   / EV +0.4 / 0% stuck-watchdog. Throwing that away and re-learning the
#   NEOW→map→first-combat loop from scratch costs ~5k steps of noise.
#   Warmstart keeps the pre-trained tokens/policy/value heads plus the
#   already-warmed causality head and picks up from a healthy trajectory.
#
# Sizing rationale (820k = 100 × 8192):
#   - 21 it/s × 8 envs × 3600 s ≈ 600k steps per hour — unrealistic,
#     that's the aggregate, not per-env
#   - Actual per-env: ~3 it/s → 820k / (3 × 8) / 3600 ≈ 9.5h
#   - Matches the "overnight + morning" budget the prior sim run used
#   - Checkpoint every 10240 steps = ~80 checkpoints, plenty of rollback
#     surface area if something regresses later in training
#
# Watch points (see _handoff_phase8_smoke.md for detailed commands):
#   - EV should stay ≥0. Even a brief -0.2 spike around 200-300k is fine,
#     sustained negativity after 400k = investigate
#   - causality_loss should keep trending down but slowly now (long-tail
#     card-effect patterns take more samples to learn)
#   - max_floor should climb from 17 → regularly 17+ → eventually crash
#     Act 1 boss. Target: first boss kill by 300k, consistent by 500k
#   - throughput should stay 20-25 it/s; if it drops to <15 check for
#     memory pressure or stuck envs not restarting
set -euo pipefail

cd '/mnt/e/game/project/sts2_mcp/packages/rl-agent'
export PYTORCH_ALLOC_CONF=expandable_segments:True

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="sim_phase8_longtrain_820k_${STAMP}"
LOG_DIR="/mnt/e/game/project/sts2_mcp/packages/rl-agent/logs_attention/${RUN_NAME}"
CKPT_DIR="/mnt/e/game/project/sts2_mcp/packages/rl-agent/checkpoints_attention/${RUN_NAME}"
STATUS_FILE="${LOG_DIR}/async_status.json"
mkdir -p "${LOG_DIR}" "${CKPT_DIR}"

# Resolve most recent 10k smoke checkpoint as warmstart. Override with
# STS2_INIT_CKPT=/abs/path/to/step_XXXXX if you want a specific one.
if [[ -n "${STS2_INIT_CKPT:-}" ]]; then
  INIT_CKPT="${STS2_INIT_CKPT}"
else
  LATEST_SMOKE_DIR=$(ls -td /mnt/e/game/project/sts2_mcp/packages/rl-agent/checkpoints_attention/sim_phase8_smoke_*/ 2>/dev/null | head -1 || true)
  if [[ -z "${LATEST_SMOKE_DIR}" ]]; then
    echo "[longtrain] WARNING: no smoke checkpoint dir found, launching fresh-init"
    INIT_CKPT=""
  else
    LATEST_STEP=$(ls -d ${LATEST_SMOKE_DIR}step_* 2>/dev/null | sort -V | tail -1 || true)
    if [[ -z "${LATEST_STEP}" ]]; then
      echo "[longtrain] WARNING: ${LATEST_SMOKE_DIR} has no step_* dirs, launching fresh-init"
      INIT_CKPT=""
    else
      INIT_CKPT="${LATEST_STEP}"
    fi
  fi
fi

N_ENVS="${STS2_N_ENVS:-8}"
TOTAL_TIMESTEPS="${STS2_TOTAL_TIMESTEPS:-819200}"

echo "[longtrain] RUN_NAME=${RUN_NAME}"
echo "[longtrain] N_ENVS=${N_ENVS}"
echo "[longtrain] TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS}"
echo "[longtrain] INIT_CKPT=${INIT_CKPT:-<fresh>}"
echo "[longtrain] LOG_DIR=${LOG_DIR}"
echo "[longtrain] CKPT_DIR=${CKPT_DIR}"
echo "[longtrain] starting Phase 8 820k long train..."

INIT_CKPT_ARG=()
if [[ -n "${INIT_CKPT}" ]]; then
  INIT_CKPT_ARG=(--init-checkpoint "${INIT_CKPT}")
fi

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
  --checkpoint-keep-last 8 \
  --stuck-watchdog-steps 400 \
  --aux-enemy-state-coef 0.10 \
  --aux-causality-coef 0.10 \
  --perf-stats-interval-steps 500 \
  --log-dir "${LOG_DIR}" \
  --checkpoint-dir "${CKPT_DIR}" \
  --status-file "${STATUS_FILE}" \
  "${INIT_CKPT_ARG[@]}"
