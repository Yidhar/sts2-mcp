#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv-wsl-rocm}"
BOOTSTRAP_SCRIPT="$SCRIPT_DIR/bootstrap_wsl_rocm.sh"
RELAY_PS_SCRIPT="$SCRIPT_DIR/start_wsl_bridge_relay.ps1"

DEFAULT_SESSION_FILE_WSL="/mnt/c/Users/yidhar/AppData/Roaming/SlayTheSpire2/bridge/session.json"
export STS2_BRIDGE_SESSION_FILE="${STS2_BRIDGE_SESSION_FILE:-$DEFAULT_SESSION_FILE_WSL}"

if [[ ! -f "$STS2_BRIDGE_SESSION_FILE" ]]; then
  echo "[wsl-rocm] STS2 bridge session file not found: $STS2_BRIDGE_SESSION_FILE" >&2
  exit 1
fi

if [[ ! -d "$VENV_DIR" ]] || ! "$VENV_DIR/bin/python" -c "import torch" >/dev/null 2>&1; then
  echo "[wsl-rocm] ROCm venv missing or incomplete; running bootstrap..."
  "$BOOTSTRAP_SCRIPT"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

session_file_win="$(wslpath -w "$STS2_BRIDGE_SESSION_FILE")"
relay_ps_win="$(wslpath -w "$RELAY_PS_SCRIPT")"

bridge_port="$(
  python - <<'PY' "$STS2_BRIDGE_SESSION_FILE"
import json
import sys
with open(sys.argv[1], "r", encoding="utf-8") as handle:
    data = json.load(handle)
port = int(data.get("port") or 0)
if port <= 0:
    base_url = str(data.get("base_url") or "")
    if ":" in base_url:
        port = int(base_url.rstrip("/").rsplit(":", 1)[1])
if port <= 0:
    raise SystemExit("Could not resolve bridge port from session.json")
print(port)
PY
)"
bridge_token="$(
  python - <<'PY' "$STS2_BRIDGE_SESSION_FILE"
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    data = json.load(handle)
print(data["token"])
PY
)"

if BRIDGE_TOKEN="$bridge_token" BRIDGE_PORT="$bridge_port" python - <<'PY'
import os
import urllib.request

port = int(os.environ["BRIDGE_PORT"])
token = os.environ["BRIDGE_TOKEN"]
req = urllib.request.Request(
    f"http://127.0.0.1:{port}/health",
    headers={"Authorization": f"Bearer {token}"},
)
with urllib.request.urlopen(req, timeout=5) as resp:
    print(f"[wsl-rocm] direct localhost bridge health={resp.status}")
PY
then
  export STS2_BRIDGE_BASE_URL="http://127.0.0.1:${bridge_port}/"
else
  relay_port="${STS2_BRIDGE_RELAY_PORT:-$((bridge_port + 1000))}"
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$relay_ps_win" \
    -SessionFile "$session_file_win" \
    -ListenHost '0.0.0.0' \
    -ListenPort "$relay_port" \
    >/dev/null
  export STS2_BRIDGE_BASE_URL="http://host.docker.internal:${relay_port}/"
fi

echo "[wsl-rocm] STS2_BRIDGE_SESSION_FILE=$STS2_BRIDGE_SESSION_FILE"
echo "[wsl-rocm] STS2_BRIDGE_BASE_URL=$STS2_BRIDGE_BASE_URL"

BRIDGE_TOKEN="$bridge_token" python - <<'PY'
import os
import urllib.request

base_url = os.environ["STS2_BRIDGE_BASE_URL"].rstrip("/")
token = os.environ["BRIDGE_TOKEN"]
req = urllib.request.Request(f"{base_url}/health", headers={"Authorization": f"Bearer {token}"})
with urllib.request.urlopen(req, timeout=5) as resp:
    print(f"[wsl-rocm] final bridge health={resp.status}")
PY

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
print("[wsl-rocm] " + json.dumps(payload, ensure_ascii=False))
PY

exec python "$REPO_ROOT/legacy/train_muzero.py" --device cuda "$@"
