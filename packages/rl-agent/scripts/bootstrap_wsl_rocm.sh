#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

ROCM_VERSION="${ROCM_VERSION:-7.2.1}"
AMDGPU_INSTALL_DEB_URL="${AMDGPU_INSTALL_DEB_URL:-}"
TORCH_WHL_URL="${TORCH_WHL_URL:-https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/torch-2.9.1%2Brocm7.2.1.lw.gitff65f5bc-cp312-cp312-linux_x86_64.whl}"
TORCHVISION_WHL_URL="${TORCHVISION_WHL_URL:-https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/torchvision-0.24.0%2Brocm7.2.1.gitb919bd0c-cp312-cp312-linux_x86_64.whl}"
TORCHAUDIO_WHL_URL="${TORCHAUDIO_WHL_URL:-https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/torchaudio-2.9.0%2Brocm7.2.1.gite3c6ee2b-cp312-cp312-linux_x86_64.whl}"
TRITON_WHL_URL="${TRITON_WHL_URL:-https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/triton-3.5.1%2Brocm7.2.1.gita272dfa8-cp312-cp312-linux_x86_64.whl}"
VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv-wsl-rocm}"
REQ_FILE="${REQ_FILE:-$REPO_ROOT/requirements-wsl-rocm.txt}"
TARGET_USER="${TARGET_USER:-${SUDO_USER:-${USER:-}}}"

if ! grep -qi microsoft /proc/version 2>/dev/null; then
  echo "[bootstrap] This script must be run inside WSL." >&2
  exit 1
fi

if [[ $EUID -eq 0 ]]; then
  SUDO=""
else
  if ! command -v sudo >/dev/null 2>&1; then
    echo "[bootstrap] sudo is required for ROCm system package installation." >&2
    exit 1
  fi
  SUDO="sudo"
fi

run_user_shell() {
  local script="$1"
  if [[ $EUID -eq 0 && -n "$TARGET_USER" && "$TARGET_USER" != "root" ]] && id "$TARGET_USER" >/dev/null 2>&1; then
    su - "$TARGET_USER" -c "bash -lc $(printf '%q' "$script")"
  else
    bash -lc "$script"
  fi
}

echo "[bootstrap] Repo root: $REPO_ROOT"
echo "[bootstrap] ROCm version: $ROCM_VERSION"
echo "[bootstrap] Torch wheel: $TORCH_WHL_URL"

resolve_amdgpu_install_url() {
  if [[ -n "$AMDGPU_INSTALL_DEB_URL" ]]; then
    echo "$AMDGPU_INSTALL_DEB_URL"
    return
  fi

  python3 - <<'PY' "$ROCM_VERSION"
import re
import sys

version = sys.argv[1]
parts = [int(x) for x in version.split(".")]
if len(parts) != 3:
    raise SystemExit(f"Unexpected ROCm version format: {version}")
major, minor, patch = parts
build = f"{major}.{minor}.{major:01d}{minor:01d}{patch:02d}{major:01d}"
filename = f"amdgpu-install_{build}-1_all.deb"
print(f"https://repo.radeon.com/amdgpu-install/{major}.{minor}.{patch}/ubuntu/noble/{filename}")
PY
}

install_system_rocm() {
  if command -v rocminfo >/dev/null 2>&1 && [[ "${FORCE_ROCM_SYSTEM_INSTALL:-0}" != "1" ]]; then
    echo "[bootstrap] rocminfo already present; skipping amdgpu-install. Set FORCE_ROCM_SYSTEM_INSTALL=1 to reinstall."
    return
  fi

  echo "[bootstrap] Installing ROCm WSL runtime packages via amdgpu-install..."
  $SUDO apt-get update
  $SUDO apt-get install -y wget gpg python3-venv python3-pip

  local tmp_deb=""
  local deb_url
  deb_url="$(resolve_amdgpu_install_url)"
  tmp_deb="$(mktemp /tmp/amdgpu-install.XXXXXX.deb)"
  trap 'rm -f "$tmp_deb"' EXIT
  wget -O "$tmp_deb" "$deb_url"
  $SUDO apt-get install -y --allow-downgrades "$tmp_deb"
  $SUDO amdgpu-install -y --usecase=wsl,rocm --no-dkms

  if [[ -n "$TARGET_USER" ]] && id "$TARGET_USER" >/dev/null 2>&1; then
    $SUDO usermod -a -G render,video "$TARGET_USER" || true
  fi
}

install_python_env() {
  echo "[bootstrap] Creating/updating venv: $VENV_DIR"
  run_user_shell "
set -euo pipefail
python3 -m venv '$VENV_DIR'
source '$VENV_DIR/bin/activate'
python -m pip install --upgrade pip setuptools wheel
python -m pip install --timeout 120 --retries 10 \
  '$TORCH_WHL_URL' \
  '$TORCHVISION_WHL_URL' \
  '$TORCHAUDIO_WHL_URL' \
  '$TRITON_WHL_URL'
python -m pip install -r '$REQ_FILE'
"
}

verify_stack() {
  run_user_shell "
set -euo pipefail
source '$VENV_DIR/bin/activate'
python - <<'PY'
import json
import torch

payload = {
    'torch_version': torch.__version__,
    'cuda_is_available': bool(torch.cuda.is_available()),
    'device_count': int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
}
if torch.cuda.is_available():
    payload['device_name'] = torch.cuda.get_device_name(0)
print(json.dumps(payload, ensure_ascii=False))
if not payload['cuda_is_available']:
    raise SystemExit(2)
PY
"

  if command -v rocminfo >/dev/null 2>&1; then
    echo "[bootstrap] rocminfo detected:"
    rocminfo | sed -n '1,24p' || true
  else
    echo "[bootstrap] Warning: rocminfo still missing after install." >&2
  fi
}

install_system_rocm
install_python_env
verify_stack

echo
echo "[bootstrap] WSL ROCm environment is ready."
echo "[bootstrap] If GPU is still unavailable, update the Windows AMD WSL-capable driver and restart WSL:"
echo "             wsl.exe --shutdown"
