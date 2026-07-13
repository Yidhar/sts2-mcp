#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
CHECKOUT_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
ARTIFACT_ROOT="${STS2_ARTIFACT_ROOT:-$HOME/.sts2-artifacts}"
if [[ "$ARTIFACT_ROOT" != /* ]]; then
  echo "[preheat-rocm] STS2_ARTIFACT_ROOT must be an absolute WSL path: $ARTIFACT_ROOT" >&2
  exit 2
fi
ARTIFACT_ROOT="$(realpath -m -- "$ARTIFACT_ROOT")"
if [[ "$ARTIFACT_ROOT" == "/" ||
      "$ARTIFACT_ROOT" == "$CHECKOUT_ROOT" ||
      "$ARTIFACT_ROOT" == "$CHECKOUT_ROOT/"* ||
      "$CHECKOUT_ROOT" == "$ARTIFACT_ROOT/"* ]]; then
  echo "[preheat-rocm] artifact root must be disjoint from the source checkout: $ARTIFACT_ROOT" >&2
  exit 2
fi
export STS2_ARTIFACT_ROOT="$ARTIFACT_ROOT"

VENV_DIR="${VENV_DIR:-$ARTIFACT_ROOT/environments/wsl-rocm}"
BOOTSTRAP_SCRIPT="$SCRIPT_DIR/bootstrap_wsl_rocm.sh"
DEFAULT_SIM_EXE="$ARTIFACT_ROOT/dependencies/sts2-ai/STS2AI/ENV/Sim/HeadlessSim/bin/Release/net9.0/HeadlessSim.exe"
SIM_EXE="${STS2_HEADLESS_SIM_EXE:-$DEFAULT_SIM_EXE}"

if [[ ! -d "$VENV_DIR" ]] || ! "$VENV_DIR/bin/python" -c "import torch" >/dev/null 2>&1; then
  echo "[preheat-rocm] ROCm venv missing or incomplete; running bootstrap..."
  "$BOOTSTRAP_SCRIPT"
fi
if [[ ! -f "$SIM_EXE" ]]; then
  echo "[preheat-rocm] pinned HeadlessSim was not found: $SIM_EXE" >&2
  echo "[preheat-rocm] build it with scripts/build_pinned_headless_sim.py or set STS2_HEADLESS_SIM_EXE." >&2
  exit 1
fi
if [[ ! -f "$SIM_EXE.identity.json" ]]; then
  echo "[preheat-rocm] simulator identity sidecar was not found: $SIM_EXE.identity.json" >&2
  exit 1
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
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
print("[preheat-rocm] " + json.dumps(payload, ensure_ascii=False))
if not torch.cuda.is_available():
    raise SystemExit("[preheat-rocm] ROCm GPU is not available; refusing a silent CPU fallback")
PY

echo "[preheat-rocm] STS2_ARTIFACT_ROOT=$STS2_ARTIFACT_ROOT"
echo "[preheat-rocm] STS2_HEADLESS_SIM_EXE=$SIM_EXE"

cd "$REPO_ROOT"
exec python -m sts2_rl.train \
  --profile preheat \
  --device cuda \
  --backend headless \
  --sim-exe "$SIM_EXE" \
  "$@"
