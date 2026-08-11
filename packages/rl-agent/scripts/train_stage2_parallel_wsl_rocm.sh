#!/usr/bin/env bash
# Parallel spool-based stage-2/4 training: N collectors + 1 trainer.
# Env: STS2_STAGE2_CHAMPION (checkpoint dir), optional STS2_STAGE2_COLLECTORS
# (default 3), STS2_STAGE2_RUN_DIR, plus the usual artifact-root contract.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
CHECKOUT_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
ARTIFACT_ROOT="${STS2_ARTIFACT_ROOT:-$HOME/.sts2-artifacts}"
[[ "$ARTIFACT_ROOT" == /* ]] || { echo "[stage2-par] STS2_ARTIFACT_ROOT must be absolute" >&2; exit 2; }
ARTIFACT_ROOT="$(realpath -m -- "$ARTIFACT_ROOT")"
if [[ "$ARTIFACT_ROOT" == "/" || "$ARTIFACT_ROOT" == "$CHECKOUT_ROOT" ||
      "$ARTIFACT_ROOT" == "$CHECKOUT_ROOT/"* || "$CHECKOUT_ROOT" == "$ARTIFACT_ROOT/"* ]]; then
  echo "[stage2-par] artifact root must be disjoint from the checkout" >&2; exit 2
fi
export STS2_ARTIFACT_ROOT="$ARTIFACT_ROOT"

VENV_DIR="${VENV_DIR:-$ARTIFACT_ROOT/environments/wsl-rocm}"
DEFAULT_SIM_EXE="$ARTIFACT_ROOT/dependencies/sts2-ai/STS2AI/ENV/Sim/HeadlessSim/bin/Release/net9.0/HeadlessSim.exe"
SIM_EXE="${STS2_HEADLESS_SIM_EXE:-$DEFAULT_SIM_EXE}"
CHAMPION="${STS2_STAGE2_CHAMPION:?[stage2-par] set STS2_STAGE2_CHAMPION}"
COLLECTORS="${STS2_STAGE2_COLLECTORS:-3}"
RUN_DIR="${STS2_STAGE2_RUN_DIR:-$ARTIFACT_ROOT/runs/stage2-isolated-macro-v1/parallel}"
[[ -f "$CHAMPION/network.pt" ]] || { echo "[stage2-par] champion missing network.pt" >&2; exit 1; }
[[ -f "$SIM_EXE" && -f "$SIM_EXE.identity.json" ]] || { echo "[stage2-par] pinned sim or identity missing" >&2; exit 1; }

export VIRTUAL_ENV="$VENV_DIR"
export PATH="$VENV_DIR/bin:$PATH"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("[stage2-par] ROCm GPU unavailable; refusing CPU fallback")
print("[stage2-par] gpu:", torch.cuda.get_device_name(0))
PY

mkdir -p "$RUN_DIR/spool" "$RUN_DIR/logs"
MODEL="$RUN_DIR/macro-online.pt"
cd "$REPO_ROOT"

echo "[stage2-par] run dir: $RUN_DIR  collectors: $COLLECTORS"
PIDS=()
cleanup() { for pid in "${PIDS[@]:-}"; do kill "$pid" 2>/dev/null || true; done; }
trap cleanup EXIT INT TERM

python scripts/run_stage2_trainer.py \
  --config config/experiments/stage2_isolated_macro_v1.toml \
  --champion "$CHAMPION" \
  --spool "$RUN_DIR/spool" \
  --save-macro "$MODEL" \
  --init-macro "${STS2_STAGE2_INIT_MACRO:-$MODEL}" \
  --stop-after-episodes "${STS2_STAGE2_EPISODES:-1200}" \
  --updates-per-episode "${STS2_STAGE2_UPDATES:-8}" \
  --metrics-out "$RUN_DIR/trainer-metrics.jsonl" \
  --device cuda --sim-exe "$SIM_EXE" \
  > "$RUN_DIR/logs/trainer.log" 2>&1 &
TRAINER_PID=$!
PIDS+=("$TRAINER_PID")

for index in $(seq 1 "$COLLECTORS"); do
  python scripts/run_stage2_collector.py \
    --config config/experiments/stage2_isolated_macro_v1.toml \
    --champion "$CHAMPION" \
    --spool "$RUN_DIR/spool" \
    --model-path "$MODEL" \
    --episodes "${STS2_STAGE2_COLLECTOR_EPISODES:-1000}" \
    --epsilon "${STS2_STAGE2_EPSILON:-0.15}" \
    ${STS2_STAGE2_OWN_COMBAT:+--own-combat} \
    --seed $((6500000 + index * 10000)) \
    --metrics-out "$RUN_DIR/collector-$index-metrics.jsonl" \
    --device cuda --sim-exe "$SIM_EXE" \
    > "$RUN_DIR/logs/collector-$index.log" 2>&1 &
  PIDS+=("$!")
done

echo "[stage2-par] trainer pid $TRAINER_PID; waiting for trainer to finish"
wait "$TRAINER_PID"
STATUS=$?
echo "[stage2-par] trainer exited with $STATUS; stopping collectors"
exit "$STATUS"
