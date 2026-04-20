#!/usr/bin/env bash
# Phase 9 P-0'' — targeted boss-sandbox curriculum on the 7 losers
# from the P-1 eval (commit c554366, artifact phase8_p1_boss_eval_20260420_150923.json):
#
#   ≥87% WR: QUEEN / KNOWLEDGE_DEMON / KAISER_CRAB   ← solved
#   0%  WR: CEREMONIAL_BEAST / WATERFALL_GIANT / DOORMAKER
#           SOUL_FYSH / THE_INSATIABLE / TEST_SUBJECT / THE_KIN
#
# P-1' macro freeze-train is gated on lifting the 7-loser WR off 0.
# Target for this run: ≥20% average across the 7 losers. Even partial
# lift (e.g. WATERFALL_GIANT sentinel-HP speedcheck going from 0/6 to
# 2/6) is a meaningful value-head correction and would validate
# continuing the curriculum rather than re-doing the architecture.
#
# Snapshot pool — 49 rows all builds, encounter_balanced so the thin
# DOORMAKER / SOUL_FYSH buckets aren't starved by the deeper
# INSATIABLE bucket (14 rows). No build_id filter (user confirmed
# version differences can be ignored — curated filter already removed
# combat_reset failures).
#
# Size: 204800 = 400 × (8 n_envs × 64 n_steps) — same cadence as
# potion_v3 long-train so optimizer dynamics stay familiar. ~1.5-2h
# on sim; cheap first probe.
set -euo pipefail

cd '/mnt/e/game/project/sts2_mcp/packages/rl-agent'
export PYTORCH_ALLOC_CONF=expandable_segments:True

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="sim_p0pp_losers_${STAMP}"
LOG_DIR="/mnt/e/game/project/sts2_mcp/packages/rl-agent/logs_attention/${RUN_NAME}"
CKPT_DIR="/mnt/e/game/project/sts2_mcp/packages/rl-agent/checkpoints_attention/${RUN_NAME}"
STATUS_FILE="${LOG_DIR}/async_status.json"
ANALYSIS_DIR="/mnt/e/game/project/sts2_mcp/packages/rl-agent/analysis/training_curves"
mkdir -p "${LOG_DIR}" "${CKPT_DIR}" "${ANALYSIS_DIR}"

INIT_CKPT='/mnt/e/game/project/sts2_mcp/packages/rl-agent/checkpoints_attention/sim_phase8_resume_potion_v3_20260420_125815/step_001116160'

LOSERS='ENCOUNTER.CEREMONIAL_BEAST_BOSS,ENCOUNTER.WATERFALL_GIANT_BOSS,ENCOUNTER.DOORMAKER_BOSS,ENCOUNTER.SOUL_FYSH_BOSS,ENCOUNTER.THE_INSATIABLE_BOSS,ENCOUNTER.TEST_SUBJECT_BOSS,ENCOUNTER.THE_KIN_BOSS'

echo "[p0pp_losers] RUN_NAME=${RUN_NAME}"
echo "[p0pp_losers] INIT_CKPT=${INIT_CKPT}"
echo "[p0pp_losers] LOSERS=${LOSERS}"
echo "[p0pp_losers] starting 200k targeted boss-sandbox curriculum..."

./.venv-wsl-rocm/bin/python -u ./train_attention_policy.py \
  --use-sim \
  --sim-exe-path '/mnt/e/game/project/sts2_mcp/third_party/sts2-ai/STS2AI/ENV/Sim/Host/bin/Debug/net9.0/headless_sim_host_0991.exe' \
  --init-checkpoint "${INIT_CKPT}" \
  --total-timesteps 204800 \
  --n-envs 8 \
  --encode-pool-workers 2 \
  --collector-mode async \
  --combat-sandbox \
  --character ironclad \
  --snapshot-pool '/mnt/e/game/project/sts2_mcp/datasets/curated_combat_ironclad_mixed_provenance' \
  --snapshot-curated-subset bootstrap_human_plus_local_all_roomwin_only_minus_combat_reset_failures \
  --snapshot-encounter-ids "${LOSERS}" \
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
  --checkpoint-keep-last 6 \
  --stuck-watchdog-steps 400 \
  --aux-enemy-state-coef 0.10 \
  --aux-causality-coef 0.10 \
  --perf-stats-interval-steps 500 \
  --log-dir "${LOG_DIR}" \
  --checkpoint-dir "${CKPT_DIR}" \
  --status-file "${STATUS_FILE}"

TARGET_CKPT="$(find "${CKPT_DIR}" -maxdepth 1 -type d -name 'step_*' | sort | tail -n 1)"
if [[ -z "${TARGET_CKPT:-}" || ! -d "${TARGET_CKPT}" ]]; then
  TARGET_CKPT="${CKPT_DIR}/final"
fi
echo "[p0pp_losers] training finished, target checkpoint=${TARGET_CKPT}"

EVAL_OUT="${ANALYSIS_DIR}/${RUN_NAME}_losers_eval.json"
echo "[p0pp_losers] running 50-ep eval over the 7 losers to measure lift..."
./.venv-wsl-rocm/bin/python -u ./eval_combat_sandbox_boss.py \
  "${TARGET_CKPT}" \
  --snapshot-pool '/mnt/e/game/project/sts2_mcp/datasets/curated_combat_ironclad_mixed_provenance' \
  --curated-subset bootstrap_human_plus_local_all_roomwin_only_minus_combat_reset_failures \
  --encounter-ids "${LOSERS}" \
  --tiers all \
  --episodes 50 \
  --device cuda \
  --no-text \
  --use-sim \
  --deterministic \
  --output-json "${EVAL_OUT}"

echo "[p0pp_losers] done."
echo "[p0pp_losers] target_ckpt=${TARGET_CKPT}"
echo "[p0pp_losers] losers_eval=${EVAL_OUT}"
