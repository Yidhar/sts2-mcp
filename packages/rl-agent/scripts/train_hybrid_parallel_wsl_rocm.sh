#!/usr/bin/env bash
set -euo pipefail

# Stage-4-style Hybrid pipeline: one ROCm learner, multiple CPU HeadlessSim
# producers and one small shared spool. A continuation restores the complete
# learner checkpoint; producer processes and their seed streams are fresh.

PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
ARTIFACT_ROOT="${STS2_ARTIFACT_ROOT:?set STS2_ARTIFACT_ROOT}"
RUN_ROOT="${STS2_HYBRID_RUN_ROOT:?set STS2_HYBRID_RUN_ROOT}"
RUN_RELATIVE="${RUN_ROOT#"$ARTIFACT_ROOT/hybrid_runs/"}"
DASHBOARD_RUN_ROOT="${STS2_HYBRID_DASHBOARD_RUN_ROOT:-$ARTIFACT_ROOT/runs/$RUN_RELATIVE}"
CONFIG="${STS2_HYBRID_CONFIG:?set STS2_HYBRID_CONFIG}"
CHAMPION="${STS2_HYBRID_CHAMPION:?set STS2_HYBRID_CHAMPION}"
RESUME="${STS2_HYBRID_RESUME:?set STS2_HYBRID_RESUME}"
LINEAGE_ID="${STS2_HYBRID_LINEAGE_ID:?set STS2_HYBRID_LINEAGE_ID}"
PIPELINE_ID="${STS2_HYBRID_PIPELINE_ID:-$LINEAGE_ID}"
COLLECTORS="${STS2_HYBRID_COLLECTORS:-2}"
TARGET_RUNS="${STS2_HYBRID_TARGET_RUNS:-128}"
COLLECTOR_EPISODES="${STS2_HYBRID_COLLECTOR_EPISODES:-100}"
COLLECTOR_START_EPISODE="${STS2_HYBRID_COLLECTOR_START_EPISODE:-0}"
MAXIMUM_PENDING="${STS2_HYBRID_MAXIMUM_PENDING_RUNS:-2}"
SEED_BASE="${STS2_HYBRID_SEED_BASE:-6600000}"
COLLECTOR_DEVICE="${STS2_HYBRID_COLLECTOR_DEVICE:-cuda}"
FINAL_EVALUATION_EPISODES="${STS2_HYBRID_FINAL_EVALUATION_EPISODES:-16}"
EVALUATION_DEVICE="${STS2_HYBRID_EVALUATION_DEVICE:-cuda}"
VENV_DIR="${VENV_DIR:-$ARTIFACT_ROOT/environments/wsl-rocm}"
SIM_EXE="${STS2_HEADLESS_SIM_EXE:-$ARTIFACT_ROOT/dependencies/sts2-ai/STS2AI/ENV/Sim/HeadlessSim/bin/Release/net9.0/HeadlessSim.exe}"

[[ "$COLLECTORS" -gt 0 ]] || { echo "[hybrid-par] collectors must be positive" >&2; exit 2; }
[[ "$FINAL_EVALUATION_EPISODES" -ge 0 ]] || {
  echo "[hybrid-par] final evaluation episodes must be non-negative" >&2
  exit 2
}
[[ -f "$RESUME" ]] || { echo "[hybrid-par] resume checkpoint missing: $RESUME" >&2; exit 2; }
[[ -f "$SIM_EXE" ]] || { echo "[hybrid-par] HeadlessSim missing: $SIM_EXE" >&2; exit 2; }

export VIRTUAL_ENV="$VENV_DIR"
export PATH="$VENV_DIR/bin:$PATH"
# WSL2 ROCm/ROCDXG needs explicit DXG discovery in non-login detached shells.
export HSA_ENABLE_DXG_DETECTION="${HSA_ENABLE_DXG_DETECTION:-1}"
export OMP_NUM_THREADS="${STS2_HYBRID_COLLECTOR_OMP:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

SPOOL="$RUN_ROOT/spool"
PUBLICATION="$RUN_ROOT/publication/network.pt"
CHECKPOINT="$RUN_ROOT/checkpoint/hybrid-training.pt"
METRICS="$RUN_ROOT/metrics"
STATUS="$RUN_ROOT/status"
LOGS="$RUN_ROOT/logs"
mkdir -p "$SPOOL" "$METRICS" "$STATUS" "$LOGS" "$DASHBOARD_RUN_ROOT" "$(dirname "$PUBLICATION")" "$(dirname "$CHECKPOINT")"
[[ -z "$(find "$SPOOL" -maxdepth 1 -name '*.pkl' -print -quit)" ]] || {
  echo "[hybrid-par] continuation spool must start empty" >&2
  exit 2
}
# A publication is a derived collector cache, not the resume authority.  Drop
# any file left by the interrupted process so the existence wait below can
# only be satisfied by weights freshly restored from this checkpoint.
rm -f "$PUBLICATION"

cd "$PACKAGE_ROOT"
PRODUCER_ARGS=()
for index in $(seq 0 $((COLLECTORS - 1))); do
  PRODUCER_ARGS+=(--producer-id "collector-$index")
done
MIGRATION_ARGS=()
if [[ "${STS2_HYBRID_MIGRATE_TARGET_SCHEDULE:-0}" == "1" ]]; then
  MIGRATION_ARGS+=(--migrate-target-schedule)
fi

PIDS=()
COLLECTOR_PIDS=()
cleanup() {
  for pid in "${PIDS[@]:-}"; do kill "$pid" 2>/dev/null || true; done
}
trap cleanup EXIT INT TERM

OMP_NUM_THREADS="${STS2_HYBRID_TRAINER_OMP:-4}" \
MKL_NUM_THREADS="${STS2_HYBRID_TRAINER_OMP:-4}" \
python scripts/run_hybrid_trainer.py \
  --config "$CONFIG" \
  --champion "$CHAMPION" \
  --spool "$SPOOL" \
  --publication "$PUBLICATION" \
  --checkpoint-out "$CHECKPOINT" \
  --metrics-out "$METRICS/trainer.jsonl" \
  --status-out "$STATUS/trainer.json" \
  --lineage-id "$LINEAGE_ID" \
  --pipeline-id "$PIPELINE_ID" \
  "${PRODUCER_ARGS[@]}" \
  --resume "$RESUME" \
  --stop-after-runs "$TARGET_RUNS" \
  --updates-per-run "${STS2_HYBRID_UPDATES_PER_RUN:-1}" \
  --ingest-batch-runs "${STS2_HYBRID_INGEST_BATCH_RUNS:-2}" \
  --run-batch-size "${STS2_HYBRID_RUN_BATCH_SIZE:-4}" \
  --replay-runs "${STS2_HYBRID_REPLAY_RUNS:-128}" \
  --macro-unroll "${STS2_HYBRID_MACRO_UNROLL:-8}" \
  --combat-unroll "${STS2_HYBRID_COMBAT_UNROLL:-32}" \
  --maximum-view-reuse "${STS2_HYBRID_MAXIMUM_VIEW_REUSE:-8}" \
  --maximum-pending-runs "$MAXIMUM_PENDING" \
  --target-update-interval "${STS2_HYBRID_TARGET_UPDATE_INTERVAL:-25}" \
  "${MIGRATION_ARGS[@]}" \
  --save-interval-runs "${STS2_HYBRID_SAVE_INTERVAL_RUNS:-5}" \
  --publication-interval-runs 1 \
  --device cuda >"$LOGS/trainer.log" 2>&1 &
TRAINER_PID=$!
PIDS+=("$TRAINER_PID")

PUBLICATION_WAIT_SECONDS="${STS2_HYBRID_PUBLICATION_WAIT_SECONDS:-600}"
for _ in $(seq 1 "$PUBLICATION_WAIT_SECONDS"); do
  [[ -f "$PUBLICATION" ]] && break
  kill -0 "$TRAINER_PID" 2>/dev/null || {
    cat "$LOGS/trainer.log" >&2
    exit 1
  }
  sleep 1
done
[[ -f "$PUBLICATION" ]] || { echo "[hybrid-par] trainer publication timed out" >&2; exit 1; }

for index in $(seq 0 $((COLLECTORS - 1))); do
  producer="collector-$index"
  python scripts/run_hybrid_collector.py \
    --config "$CONFIG" \
    --champion "$CHAMPION" \
    --model-path "$PUBLICATION" \
    --spool "$SPOOL" \
    --status-out "$STATUS/$producer.json" \
    --metrics-out "$METRICS/$producer.jsonl" \
    --pipeline-id "$PIPELINE_ID" \
    --producer-id "$producer" \
    --episodes "$COLLECTOR_EPISODES" \
    --start-episode-index "$COLLECTOR_START_EPISODE" \
    --seed $((SEED_BASE + index * 100000)) \
    --reload-episodes 1 \
    --maximum-pending-runs "$MAXIMUM_PENDING" \
    --device "$COLLECTOR_DEVICE" \
    --sim-exe "$SIM_EXE" >"$LOGS/$producer.log" 2>&1 &
  PIDS+=("$!")
  COLLECTOR_PIDS+=("$!")
done

echo "[hybrid-par] pipeline=$PIPELINE_ID learner=$TRAINER_PID collectors=$COLLECTORS"
while kill -0 "$TRAINER_PID" 2>/dev/null; do
  for collector_pid in "${COLLECTOR_PIDS[@]}"; do
    if ! kill -0 "$collector_pid" 2>/dev/null; then
      echo "[hybrid-par] collector=$collector_pid exited while learner was running" >&2
      kill -TERM "$TRAINER_PID" 2>/dev/null || true
      wait "$TRAINER_PID" 2>/dev/null || true
      exit 1
    fi
  done
  sleep 2
done
set +e
wait "$TRAINER_PID"
status=$?
set -e
echo "[hybrid-par] trainer exited with $status; stopping collectors"
for collector_pid in "${COLLECTOR_PIDS[@]}"; do
  kill -TERM "$collector_pid" 2>/dev/null || true
done
for collector_pid in "${COLLECTOR_PIDS[@]}"; do
  wait "$collector_pid" 2>/dev/null || true
done
COLLECTOR_PIDS=()
PIDS=()
[[ "$status" -eq 0 ]] || exit "$status"

if [[ "$FINAL_EVALUATION_EPISODES" -gt 0 ]]; then
  EVALUATION="$RUN_ROOT/evaluation"
  FROZEN_POLICY="$EVALUATION/frozen-final-policy.pt"
  mkdir -p "$EVALUATION"
  cp -f "$PUBLICATION" "$FROZEN_POLICY"
  echo "[hybrid-par] starting final held-out evaluation episodes=$FINAL_EVALUATION_EPISODES"
  python scripts/run_hybrid_evaluation.py \
    --config "$CONFIG" \
    --champion "$CHAMPION" \
    --model-path "$FROZEN_POLICY" \
    --episodes "$FINAL_EVALUATION_EPISODES" \
    --metrics-out "$METRICS/evaluation.jsonl" \
    --journal-directory "$DASHBOARD_RUN_ROOT" \
    --device "$EVALUATION_DEVICE" \
    --sim-exe "$SIM_EXE" >"$LOGS/evaluation.log" 2>&1
fi
