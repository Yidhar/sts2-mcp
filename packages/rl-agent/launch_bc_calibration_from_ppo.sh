#!/usr/bin/env bash
# Offline BC calibration: take the current best PPO checkpoint, fine-tune
# its non-combat decision weights against the Skada human-decision corpus,
# save a new checkpoint that the next PPO run warmstarts from.
#
# Why we need this
#   After the 800k Phase 8 long-train, the policy reached Act 1 boss (2%)
#   but learned pathological non-combat behavior: 0 potion use across live
#   eval, SMITH preferred over HEAL at low HP. Reward shaping (6643917 +
#   65b5196) biases gradient direction but can't retroactively undo
#   millions of bad-decision rollouts. BC calibration injects ~1.5M human
#   non-combat decisions as a supervised signal in ~15-30 min, directly
#   shifting the logit distribution toward sensible meta play.
#
# Safety rails
#   - Low LR (1e-5): 30× smaller than default 3e-4 prevents catastrophic
#     forgetting of learned combat policy. Gradient still adjusts but
#     slowly.
#   - Phase filter NON_COMBAT: combat samples never flow through gradient,
#     so combat decision heads aren't perturbed (aside from the shared
#     backbone's slow drift).
#   - Capped at ~500k samples / ~20 min runtime: short enough that the
#     combat policy doesn't noticeably degrade even if the phase filter
#     somehow leaked.
#
# After this script runs, resume PPO via:
#   STS2_INIT_CKPT=checkpoints_attention/bc_calibration_XXXX/final \
#   bash launch_sim_phase8_longtrain_820k.sh
set -euo pipefail

cd '/mnt/e/game/project/sts2_mcp/packages/rl-agent'

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="bc_calibration_${STAMP}"
CKPT_DIR="checkpoints_attention/${RUN_NAME}"
LOG_FILE="logs_attention/${RUN_NAME}.stdout.log"
mkdir -p "${CKPT_DIR}" logs_attention

# Resolve the most recent PPO checkpoint to calibrate from. Preference:
#   1. $STS2_INIT_CKPT env var (explicit user override)
#   2. Latest step_* from the most recent sim_phase8_longtrain_* dir
#   3. Latest step_* from the most recent sim_phase8_smoke_* dir
#   4. Fail — no calibration target found
if [[ -n "${STS2_INIT_CKPT:-}" ]]; then
  INIT_CKPT="${STS2_INIT_CKPT}"
else
  LATEST_PPO_DIR=$(ls -td checkpoints_attention/sim_phase8_longtrain_*/ 2>/dev/null | head -1 || true)
  if [[ -z "${LATEST_PPO_DIR}" ]]; then
    LATEST_PPO_DIR=$(ls -td checkpoints_attention/sim_phase8_smoke_*/ 2>/dev/null | head -1 || true)
  fi
  if [[ -z "${LATEST_PPO_DIR}" ]]; then
    echo "ERROR: no PPO checkpoint found to calibrate from."
    echo "  set STS2_INIT_CKPT=/abs/path/to/step_NNNNNN or run a PPO training first."
    exit 1
  fi
  INIT_CKPT=$(ls -d ${LATEST_PPO_DIR}step_* 2>/dev/null | sort -V | tail -1 || true)
  if [[ -z "${INIT_CKPT}" ]]; then
    echo "ERROR: ${LATEST_PPO_DIR} has no step_* subdirs."
    exit 1
  fi
fi

# Defaults: sized for ~20 min on a single GPU. Override via env for
# longer / shorter runs.
TOTAL_SAMPLES="${STS2_BC_TOTAL_SAMPLES:-500000}"
BATCH_SIZE="${STS2_BC_BATCH_SIZE:-32}"
NUM_WORKERS="${STS2_BC_NUM_WORKERS:-4}"
LR="${STS2_BC_LR:-1e-5}"

echo "[bc-calibration] RUN_NAME=${RUN_NAME}"
echo "[bc-calibration] INIT_CKPT=${INIT_CKPT}"
echo "[bc-calibration] TOTAL_SAMPLES=${TOTAL_SAMPLES}"
echo "[bc-calibration] LR=${LR}"
echo "[bc-calibration] CKPT_DIR=${CKPT_DIR}"
echo "[bc-calibration] log: ${LOG_FILE}"

exec ./.venv-wsl-rocm/bin/python -u ./skada_bc_train.py \
  --samples data/skada_bc/samples.jsonl \
  --init-checkpoint "${INIT_CKPT}" \
  --phase-filter NON_COMBAT \
  --checkpoint-dir "${CKPT_DIR}" \
  --total-samples "${TOTAL_SAMPLES}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --shuffle-buffer 4096 \
  --lr "${LR}" \
  --device cuda \
  --amp-dtype bf16 \
  --grad-clip 1.0 \
  --log-interval 50 \
  --checkpoint-interval 5000 \
  --collector-mode async
