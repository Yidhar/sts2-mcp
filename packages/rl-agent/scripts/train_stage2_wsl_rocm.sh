#!/usr/bin/env bash
# Stage-2 isolated macro training launcher (semantic decision graph reset §10).
# Frozen champion + macro Double-Q via scripts/run_stage2_isolated_macro.py.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
CHECKOUT_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
ARTIFACT_ROOT="${STS2_ARTIFACT_ROOT:-$HOME/.sts2-artifacts}"
if [[ "$ARTIFACT_ROOT" != /* ]]; then
  echo "[stage2-rocm] STS2_ARTIFACT_ROOT must be an absolute WSL path: $ARTIFACT_ROOT" >&2
  exit 2
fi
ARTIFACT_ROOT="$(realpath -m -- "$ARTIFACT_ROOT")"
if [[ "$ARTIFACT_ROOT" == "/" ||
      "$ARTIFACT_ROOT" == "$CHECKOUT_ROOT" ||
      "$ARTIFACT_ROOT" == "$CHECKOUT_ROOT/"* ||
      "$CHECKOUT_ROOT" == "$ARTIFACT_ROOT/"* ]]; then
  echo "[stage2-rocm] artifact root must be disjoint from the source checkout: $ARTIFACT_ROOT" >&2
  exit 2
fi
export STS2_ARTIFACT_ROOT="$ARTIFACT_ROOT"

VENV_DIR="${VENV_DIR:-$ARTIFACT_ROOT/environments/wsl-rocm}"
BOOTSTRAP_SCRIPT="$SCRIPT_DIR/bootstrap_wsl_rocm.sh"
DEFAULT_SIM_EXE="$ARTIFACT_ROOT/dependencies/sts2-ai/STS2AI/ENV/Sim/HeadlessSim/bin/Release/net9.0/HeadlessSim.exe"
SIM_EXE="${STS2_HEADLESS_SIM_EXE:-$DEFAULT_SIM_EXE}"

CHAMPION="${STS2_STAGE2_CHAMPION:?[stage2-rocm] set STS2_STAGE2_CHAMPION to the frozen champion checkpoint directory}"
if [[ ! -f "$CHAMPION/network.pt" ]]; then
  echo "[stage2-rocm] champion checkpoint has no network.pt: $CHAMPION" >&2
  exit 1
fi

if [[ ! -d "$VENV_DIR" ]] || ! "$VENV_DIR/bin/python" -c "import torch" >/dev/null 2>&1; then
  echo "[stage2-rocm] ROCm venv missing or incomplete; running bootstrap..."
  "$BOOTSTRAP_SCRIPT"
fi
if [[ ! -f "$SIM_EXE" ]]; then
  echo "[stage2-rocm] pinned HeadlessSim was not found: $SIM_EXE" >&2
  exit 1
fi
if [[ ! -f "$SIM_EXE.identity.json" ]]; then
  echo "[stage2-rocm] simulator identity sidecar was not found: $SIM_EXE.identity.json" >&2
  exit 1
fi

# The venv was relocated after creation, so its activate script exports a
# stale VIRTUAL_ENV; bind the interpreter by path instead of sourcing it.
export VIRTUAL_ENV="$VENV_DIR"
export PATH="$VENV_DIR/bin:$PATH"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"

python - <<'PY'
import json
import torch

payload = {
    "torch_version": torch.__version__,
    "cuda_is_available": bool(torch.cuda.is_available()),
    "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
}
if torch.cuda.is_available():
    payload["device_name"] = torch.cuda.get_device_name(0)
print("[stage2-rocm] " + json.dumps(payload, ensure_ascii=False))
if not torch.cuda.is_available():
    raise SystemExit("[stage2-rocm] ROCm GPU is not available; refusing a silent CPU fallback")
PY

RUN_DIR="$ARTIFACT_ROOT/runs/stage2-isolated-macro-v1"
mkdir -p "$RUN_DIR"

echo "[stage2-rocm] STS2_ARTIFACT_ROOT=$STS2_ARTIFACT_ROOT"
echo "[stage2-rocm] STS2_HEADLESS_SIM_EXE=$SIM_EXE"
echo "[stage2-rocm] champion=$CHAMPION"
echo "[stage2-rocm] run dir=$RUN_DIR"

cd "$REPO_ROOT"
echo "[stage2-rocm] starting isolated macro collection + Double-Q training"
exec python scripts/run_stage2_isolated_macro.py \
  --config config/experiments/stage2_isolated_macro_v1.toml \
  --champion "$CHAMPION" \
  --device cuda \
  --sim-exe "$SIM_EXE" \
  --metrics-out "$RUN_DIR/stage2-macro-metrics.jsonl" \
  --save-macro "$RUN_DIR/stage2-macro-online.pt" \
  "$@"
