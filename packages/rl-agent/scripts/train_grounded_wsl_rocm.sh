#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
CHECKOUT_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
ARTIFACT_ROOT="${STS2_ARTIFACT_ROOT:-$HOME/.sts2-artifacts}"
if [[ "$ARTIFACT_ROOT" != /* ]]; then
  echo "[wsl-rocm] STS2_ARTIFACT_ROOT must be an absolute WSL path: $ARTIFACT_ROOT" >&2
  exit 2
fi
ARTIFACT_ROOT="$(realpath -m -- "$ARTIFACT_ROOT")"
if [[ "$ARTIFACT_ROOT" == "/" ||
      "$ARTIFACT_ROOT" == "$CHECKOUT_ROOT" ||
      "$ARTIFACT_ROOT" == "$CHECKOUT_ROOT/"* ||
      "$CHECKOUT_ROOT" == "$ARTIFACT_ROOT/"* ]]; then
  echo "[wsl-rocm] artifact root must be disjoint from the source checkout: $ARTIFACT_ROOT" >&2
  exit 2
fi
export STS2_ARTIFACT_ROOT="$ARTIFACT_ROOT"
VENV_DIR="${VENV_DIR:-$ARTIFACT_ROOT/environments/wsl-rocm}"
BOOTSTRAP_SCRIPT="$SCRIPT_DIR/bootstrap_wsl_rocm.sh"
RELAY_PS_SCRIPT="$SCRIPT_DIR/start_wsl_bridge_relay.ps1"

windows_appdata_win="$(powershell.exe -NoProfile -Command '[Environment]::GetFolderPath([Environment+SpecialFolder]::ApplicationData)' | tr -d '\r')"
if [[ -z "$windows_appdata_win" ]]; then
  echo "[wsl-rocm] Could not resolve the Windows roaming AppData directory." >&2
  exit 2
fi
DEFAULT_SESSION_DIR_WSL="${STS2_BRIDGE_SESSION_DIR:-$(wslpath -u "$windows_appdata_win")/SlayTheSpire2/bridge}"
# Prefer the indexed bridge session produced by the current multi-instance
# bridge.  The unindexed session.json can be stale and has caused 401s during
# long-running grounded baseline restarts; keep it only as a fallback for older setups.
if [[ -f "$DEFAULT_SESSION_DIR_WSL/session_0.json" ]]; then
  DEFAULT_SESSION_FILE_WSL="$DEFAULT_SESSION_DIR_WSL/session_0.json"
else
  DEFAULT_SESSION_FILE_WSL="$DEFAULT_SESSION_DIR_WSL/session.json"
fi
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
tokens = data.get("capability_tokens") or {}
token = str(tokens.get("training") or "")
if not token:
    raise SystemExit("Session descriptor has no scoped training capability token")
print(token)
PY
)"

if BRIDGE_TOKEN="$bridge_token" BRIDGE_PORT="$bridge_port" python - <<'PY'
import os
import urllib.request

port = int(os.environ["BRIDGE_PORT"])
token = os.environ["BRIDGE_TOKEN"]
req = urllib.request.Request(
    f"http://127.0.0.1:{port}/v2/health",
    headers={"Authorization": f"Bearer {token}"},
)
with urllib.request.urlopen(req, timeout=5) as resp:
    print(f"[wsl-rocm] direct localhost bridge health={resp.status}")
PY
then
  export STS2_BRIDGE_BASE_URL="http://127.0.0.1:${bridge_port}/"
  unset STS2_BRIDGE_RELAY_ALLOWLIST || true
else
  relay_port="${STS2_BRIDGE_RELAY_PORT:-$((bridge_port + 1000))}"
  relay_host="${STS2_BRIDGE_RELAY_HOST:-$(awk '/^nameserver[[:space:]]+/{print $2; exit}' /etc/resolv.conf)}"
  if [[ ! "$relay_host" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "[wsl-rocm] Could not resolve the explicit Windows WSL-interface IPv4 address; set STS2_BRIDGE_RELAY_HOST." >&2
    exit 2
  fi
  relay_args=(
    -NoProfile -ExecutionPolicy Bypass -File "$relay_ps_win"
    -SessionFile "$session_file_win"
    -ListenHost "$relay_host"
    -ListenPort "$relay_port"
  )
  if [[ -n "${STS2_BRIDGE_RELAY_CLIENT_CIDR:-}" ]]; then
    relay_args+=( -AllowClientCidr "$STS2_BRIDGE_RELAY_CLIENT_CIDR" )
  fi
  powershell.exe "${relay_args[@]}" >/dev/null
  export STS2_BRIDGE_BASE_URL="http://${relay_host}:${relay_port}/"
  # BridgeClient rejects non-loopback authority unless this exact relay URL is
  # explicitly authorized.  This does not authorize a non-loopback URL inside
  # the signed session descriptor; it only permits the environment override.
  export STS2_BRIDGE_RELAY_ALLOWLIST="$STS2_BRIDGE_BASE_URL"
fi

echo "[wsl-rocm] STS2_BRIDGE_SESSION_FILE=$STS2_BRIDGE_SESSION_FILE"
echo "[wsl-rocm] STS2_BRIDGE_BASE_URL=$STS2_BRIDGE_BASE_URL"

BRIDGE_TOKEN="$bridge_token" python - <<'PY'
import os
import urllib.request

base_url = os.environ["STS2_BRIDGE_BASE_URL"].rstrip("/")
token = os.environ["BRIDGE_TOKEN"]
req = urllib.request.Request(f"{base_url}/v2/health", headers={"Authorization": f"Bearer {token}"})
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

cd "$REPO_ROOT"
exec python -m sts2_rl.train \
  --device cuda \
  --backend live \
  "$@"
