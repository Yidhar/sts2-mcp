#!/usr/bin/env bash
set -euo pipefail

# Parallel Stage-2 macro pipeline. Every invocation is one new pipeline segment
# with an empty spool. A resume points at a full trainer state from an older
# segment; it never reuses that segment's spool or logs.

PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
REPO_ROOT="$(cd "$PACKAGE_ROOT/../.." && pwd -P)"
ARTIFACT_ROOT="${STS2_ARTIFACT_ROOT:?set STS2_ARTIFACT_ROOT}"
case "$ARTIFACT_ROOT" in
  "$REPO_ROOT"|"$REPO_ROOT"/*) echo "[stage2-par] artifact root must be disjoint from checkout" >&2; exit 2 ;;
esac
export STS2_ARTIFACT_ROOT="$ARTIFACT_ROOT"

VENV_DIR="${VENV_DIR:-$ARTIFACT_ROOT/environments/wsl-rocm}"
DEFAULT_SIM_EXE="$ARTIFACT_ROOT/dependencies/sts2-ai/STS2AI/ENV/Sim/HeadlessSim/bin/Release/net9.0/HeadlessSim.exe"
SIM_EXE="${STS2_HEADLESS_SIM_EXE:-$DEFAULT_SIM_EXE}"
CHAMPION="${STS2_STAGE2_CHAMPION:?[stage2-par] set STS2_STAGE2_CHAMPION}"
COLLECTORS="${STS2_STAGE2_COLLECTORS:-2}"
LINEAGE_ID="${STS2_STAGE2_LINEAGE_ID:-stage2-isolated-macro-v1}"
CONTROL_DOMAIN="${STS2_STAGE2_CONTROL_DOMAIN:-macro}"
PIPELINE_ID="${STS2_STAGE2_PIPELINE_ID:-$("$VENV_DIR/bin/python" -c 'import uuid; print(uuid.uuid4())')}"
SEED_BASE="${STS2_STAGE2_SEED_BASE:-$("$VENV_DIR/bin/python" -c 'import secrets; print(100000000 + secrets.randbelow(800000000))')}"
RUN_DIR="${STS2_STAGE2_RUN_DIR:-$ARTIFACT_ROOT/runs/$LINEAGE_ID/run-$PIPELINE_ID}"
RESUME="${STS2_STAGE2_RESUME:-}"
INIT_MACRO="${STS2_STAGE2_INIT_MACRO:-}"
BRIDGE_PARTNER="${STS2_STAGE2_BRIDGE_PARTNER:-}"

[[ -f "$CHAMPION/network.pt" ]] || { echo "[stage2-par] champion missing network.pt" >&2; exit 1; }
[[ -f "$SIM_EXE" && -f "$SIM_EXE.identity.json" ]] || { echo "[stage2-par] pinned sim or identity missing" >&2; exit 1; }
[[ -z "$RESUME" || -z "$INIT_MACRO" ]] || { echo "[stage2-par] resume and model-init are mutually exclusive" >&2; exit 2; }
[[ -z "$RESUME" || -f "$RESUME" ]] || { echo "[stage2-par] resume state missing: $RESUME" >&2; exit 1; }
[[ -z "$INIT_MACRO" || -f "$INIT_MACRO" ]] || { echo "[stage2-par] model initialization missing: $INIT_MACRO" >&2; exit 1; }
# Combat training requires the explicit cross-domain bootstrap bridge: the
# guard lifts ONLY when a pinned macro publication exists on disk.
case "$CONTROL_DOMAIN" in
  macro)
    [[ -z "$BRIDGE_PARTNER" ]] || {
      echo "[stage2-par] STS2_STAGE2_BRIDGE_PARTNER applies only to the combat control domain" >&2
      exit 2
    }
    ;;
  combat)
    [[ -n "$BRIDGE_PARTNER" && -f "$BRIDGE_PARTNER" ]] || {
      echo "[stage2-par] combat training requires an explicit cross-domain bridge: set STS2_STAGE2_BRIDGE_PARTNER to an existing frozen macro publication" >&2
      exit 2
    }
    ;;
  *)
    echo "[stage2-par] unsupported control domain: $CONTROL_DOMAIN" >&2
    exit 2
    ;;
esac
if [[ -e "$RUN_DIR" ]] && [[ -n "$(find "$RUN_DIR" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "[stage2-par] run directory must be new and empty: $RUN_DIR" >&2
  exit 2
fi

export VIRTUAL_ENV="$VENV_DIR"
export PATH="$VENV_DIR/bin:$PATH"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
# The trainer's lockstep collation is CPU-bound; give it its own thread
# budget while collectors stay at one thread beside their sims.
TRAINER_OMP="${STS2_STAGE2_TRAINER_OMP:-1}"

python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("[stage2-par] ROCm GPU unavailable; refusing CPU fallback")
print("[stage2-par] gpu:", torch.cuda.get_device_name(0))
PY

SPOOL="$RUN_DIR/spool"
STATUS_DIR="$RUN_DIR/producers"
LOG_DIR="$RUN_DIR/logs"
MODEL="$RUN_DIR/macro-online.pt"
CHECKPOINT="$RUN_DIR/training-state.pt"
TRAINER_STATUS="$RUN_DIR/trainer-status.json"
mkdir -p "$SPOOL" "$STATUS_DIR" "$LOG_DIR"
[[ -z "$(find "$SPOOL" -mindepth 1 -maxdepth 1 -print -quit)" ]] || {
  echo "[stage2-par] spool must be empty at segment start" >&2; exit 2;
}

PRODUCER_IDS=()
for index in $(seq 1 "$COLLECTORS"); do PRODUCER_IDS+=("collector-$index"); done
export PIPELINE_ID LINEAGE_ID CONTROL_DOMAIN COLLECTORS RUN_DIR SEED_BASE RESUME INIT_MACRO
python - <<'PY'
import json, os
from pathlib import Path
run = Path(os.environ["RUN_DIR"])
payload = {
    "format": "sts2-stage2-pipeline-v1",
    "pipeline_id": os.environ["PIPELINE_ID"],
    "lineage_id": os.environ["LINEAGE_ID"],
    "control_domain": os.environ["CONTROL_DOMAIN"],
    "producer_ids": [f"collector-{i}" for i in range(1, int(os.environ["COLLECTORS"]) + 1)],
    "seed_base": int(os.environ["SEED_BASE"]),
    "learner_load_mode": "resume" if os.environ["RESUME"] else ("model_init" if os.environ["INIT_MACRO"] else "fresh"),
    "producer_state": "fresh",
}
(run / "pipeline.json").write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
PY

cd "$PACKAGE_ROOT"
PIDS=()
cleanup() { for pid in "${PIDS[@]:-}"; do kill "$pid" 2>/dev/null || true; done; }
trap cleanup EXIT INT TERM

MODE_ARGS=()
if [[ -n "$RESUME" ]]; then MODE_ARGS=(--resume "$RESUME"); fi
if [[ -n "$INIT_MACRO" ]]; then MODE_ARGS=(--init-macro "$INIT_MACRO"); fi
DOMAIN_ARGS=(--control-domain "$CONTROL_DOMAIN")
BRIDGE_ARGS=()
if [[ -n "$BRIDGE_PARTNER" ]]; then BRIDGE_ARGS=(--bridge-partner "$BRIDGE_PARTNER"); fi
PRODUCER_ARGS=()
for producer_id in "${PRODUCER_IDS[@]}"; do PRODUCER_ARGS+=(--producer-id "$producer_id"); done

OMP_NUM_THREADS="$TRAINER_OMP" MKL_NUM_THREADS="$TRAINER_OMP" \
python scripts/run_stage2_trainer.py \
  --config config/experiments/stage2_isolated_macro_v1.toml \
  --champion "$CHAMPION" \
  --spool "$SPOOL" \
  --save-macro "$MODEL" \
  --checkpoint-out "$CHECKPOINT" \
  --pipeline-id "$PIPELINE_ID" \
  --lineage-id "$LINEAGE_ID" \
  --producer-status-dir "$STATUS_DIR" \
  --status-out "$TRAINER_STATUS" \
  "${PRODUCER_ARGS[@]}" \
  "${MODE_ARGS[@]}" \
  "${DOMAIN_ARGS[@]}" \
  "${BRIDGE_ARGS[@]}" \
  --stop-after-episodes "${STS2_STAGE2_EPISODES:-1200}" \
  --updates-per-episode "${STS2_STAGE2_UPDATES:-8}" \
  --sample-windows "${STS2_STAGE2_SAMPLE_WINDOWS:-16}" \
  --replay-episodes "${STS2_STAGE2_REPLAY_EPISODES:-128}" \
  --idle-timeout-seconds "${STS2_STAGE2_IDLE_TIMEOUT_SECONDS:-1200}" \
  --metrics-out "$RUN_DIR/trainer-metrics.jsonl" \
  --device cuda --sim-exe "$SIM_EXE" \
  > "$LOG_DIR/trainer.log" 2>&1 &
TRAINER_PID=$!
PIDS+=("$TRAINER_PID")

# The trainer publishes initialized/restored behavior before accepting data.
# A resume continues the learner exactly while these producer processes and
# their random seed streams are intentionally new for this segment.
for _ in $(seq 1 120); do
  [[ -f "$MODEL" ]] && break
  kill -0 "$TRAINER_PID" 2>/dev/null || {
    cat "$LOG_DIR/trainer.log" >&2; echo "[stage2-par] trainer exited before publication" >&2; exit 1;
  }
  sleep 1
done
[[ -f "$MODEL" ]] || { echo "[stage2-par] initial model publication timed out" >&2; exit 1; }

for index in $(seq 1 "$COLLECTORS"); do
  producer_id="collector-$index"
  python scripts/run_stage2_collector.py \
    --config config/experiments/stage2_isolated_macro_v1.toml \
    --champion "$CHAMPION" \
    --spool "$SPOOL" \
    --model-path "$MODEL" \
    --pipeline-id "$PIPELINE_ID" \
    --lineage-id "$LINEAGE_ID" \
    --producer-id "$producer_id" \
    --status-out "$STATUS_DIR/$producer_id.json" \
    --episodes "${STS2_STAGE2_COLLECTOR_EPISODES:-1000}" \
    --epsilon "${STS2_STAGE2_EPSILON:-0.15}" \
    "${DOMAIN_ARGS[@]}" \
    "${BRIDGE_ARGS[@]}" \
    --seed $((SEED_BASE + index * 10000)) \
    --metrics-out "$RUN_DIR/$producer_id-metrics.jsonl" \
    --device cuda --sim-exe "$SIM_EXE" \
    > "$LOG_DIR/$producer_id.log" 2>&1 &
  PIDS+=("$!")
done

echo "[stage2-par] pipeline $PIPELINE_ID; trainer pid $TRAINER_PID; waiting"
set +e
wait "$TRAINER_PID"
STATUS=$?
set -e
SYNTHESIZED="$(
  TRAINER_EXIT="$STATUS" TRAINER_STATUS="$TRAINER_STATUS" \
  TRAINER_METRICS="$RUN_DIR/trainer-metrics.jsonl" PIPELINE_ID="$PIPELINE_ID" \
  python - <<'PY'
import json
import os
import time
from pathlib import Path

metrics = Path(os.environ["TRAINER_METRICS"])
rows = []
if metrics.is_file():
    for line in metrics.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)

if any(row.get("event") in {"trainer_complete", "trainer_failed"} for row in rows):
    print("0")
else:
    latest = rows[-1] if rows else {}
    failure = {
        "event": "trainer_failed",
        "state": "failed",
        "unix_s": time.time(),
        "pipeline_id": os.environ["PIPELINE_ID"],
        "environment_steps": int(latest.get("environment_steps") or 0),
        "ingested_total": int(latest.get("ingested_total") or 0),
        "policy_version": int(latest.get("policy_version") or 0),
        "error": "trainer process exited with status "
        + os.environ["TRAINER_EXIT"]
        + " without terminal metrics",
    }
    metrics.parent.mkdir(parents=True, exist_ok=True)
    with metrics.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(failure) + "\n")
    status = Path(os.environ["TRAINER_STATUS"])
    temporary = status.with_suffix(status.suffix + ".tmp")
    temporary.write_text(json.dumps(failure, sort_keys=True), encoding="utf-8")
    temporary.replace(status)
    print("1")
PY
)"
if [[ "$SYNTHESIZED" == "1" && "$STATUS" -eq 0 ]]; then STATUS=1; fi
echo "[stage2-par] trainer exited with $STATUS; stopping producers"
exit "$STATUS"
