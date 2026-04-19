#!/usr/bin/env bash
# Long full-run training against the frankqwang/sts2-ai HeadlessSim.
#
# Context: after v11 pipe fix, sim runs stably at ~18-30 it/s per-worker
# with 0 restarts. This script scales n_envs up (sim processes are cheap
# vs Godot game instances) and targets an 8h training budget.
#
# Sizing rationale:
#   - 4 envs gave us 18.7 it/s (smoke)
#   - 8 envs should give ~30-35 it/s (sim is CPU-bound per worker,
#     scales near-linearly up to host CPU count)
#   - 8h * 3600 * 30 it/s ≈ 864k steps; round to 819200 = 100 * 8192
#
# If your box has cores to spare, bump N_ENVS to 12 or 16 (sim is
# lightweight — each process ~50-100MB RAM, 1 CPU thread mostly idle
# between actions).
set -euo pipefail

cd '/mnt/e/game/project/sts2_mcp/packages/rl-agent'
export PYTORCH_ALLOC_CONF=expandable_segments:True

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="sim_fullrun_longtrain_${STAMP}"
LOG_DIR="/mnt/e/game/project/sts2_mcp/packages/rl-agent/logs_attention/${RUN_NAME}"
CKPT_DIR="/mnt/e/game/project/sts2_mcp/packages/rl-agent/checkpoints_attention/${RUN_NAME}"
STATUS_FILE="${LOG_DIR}/async_status.json"
mkdir -p "${LOG_DIR}" "${CKPT_DIR}"
# Trainer writes perf_stats.jsonl under LOG_DIR automatically when
# --perf-stats-interval-steps > 0. No need to specify a path.

N_ENVS="${STS2_N_ENVS:-8}"
TOTAL_TIMESTEPS="${STS2_TOTAL_TIMESTEPS:-819200}"

# Optional resume from an existing checkpoint. Sim-trained and real-game-
# trained checkpoints are structurally compatible (same policy, same obs),
# but transferring real-game → sim may show some drift for the first few
# thousand steps while the policy re-adapts to sim's slightly different
# state distribution (stubbed rendering layer, missing mod fields zero-
# filled in the translator). Leave unset for fresh init.
INIT_CKPT="${STS2_INIT_CKPT:-}"

echo "[longtrain] RUN_NAME=${RUN_NAME}"
echo "[longtrain] N_ENVS=${N_ENVS}"
echo "[longtrain] TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS}"
echo "[longtrain] INIT_CKPT=${INIT_CKPT:-<fresh>}"
echo "[longtrain] LOG_DIR=${LOG_DIR}"
echo "[longtrain] CKPT_DIR=${CKPT_DIR}"
echo "[longtrain] starting sim long training..."

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
  --checkpoint-keep-last 6 \
  --aux-enemy-state-coef 0.10 \
  --perf-stats-interval-steps 500 \
  --log-dir "${LOG_DIR}" \
  --checkpoint-dir "${CKPT_DIR}" \
  --status-file "${STATUS_FILE}" \
  "${INIT_CKPT_ARG[@]}"
