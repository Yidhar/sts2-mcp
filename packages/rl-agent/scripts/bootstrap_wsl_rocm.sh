#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
CHECKOUT_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"

ARTIFACT_ROOT="${STS2_ARTIFACT_ROOT:-$HOME/.sts2-artifacts}"
if [[ "$ARTIFACT_ROOT" != /* ]]; then
  echo "[bootstrap] STS2_ARTIFACT_ROOT must be an absolute WSL path: $ARTIFACT_ROOT" >&2
  exit 2
fi
ARTIFACT_ROOT="$(realpath -m -- "$ARTIFACT_ROOT")"
if [[ "$ARTIFACT_ROOT" == "/" ||
      "$ARTIFACT_ROOT" == "$CHECKOUT_ROOT" ||
      "$ARTIFACT_ROOT" == "$CHECKOUT_ROOT/"* ||
      "$CHECKOUT_ROOT" == "$ARTIFACT_ROOT/"* ]]; then
  echo "[bootstrap] artifact root must be disjoint from the source checkout: $ARTIFACT_ROOT" >&2
  exit 2
fi
export STS2_ARTIFACT_ROOT="$ARTIFACT_ROOT"
export PIP_DEFAULT_TIMEOUT="${PIP_DEFAULT_TIMEOUT:-120}"
export PIP_RETRIES="${PIP_RETRIES:-10}"

ROCM_VERSION="${ROCM_VERSION:-7.2.1}"
if [[ "$ROCM_VERSION" != "7.2.1" ]]; then
  echo "[bootstrap] Unsupported ROCM_VERSION=$ROCM_VERSION; update the reviewed URL/hash lock first." >&2
  exit 2
fi
AMDGPU_INSTALL_DEB_URL="${AMDGPU_INSTALL_DEB_URL:-https://repo.radeon.com/amdgpu-install/7.2.1/ubuntu/noble/amdgpu-install_7.2.1.70201-1_all.deb}"
AMDGPU_INSTALL_DEB_SHA256="${AMDGPU_INSTALL_DEB_SHA256:-4c0338a241c15b12c14eb3aeb4012ea0d55dba681737ea8482248041a16c2afa}"
TORCH_WHL_URL="${TORCH_WHL_URL:-https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/torch-2.9.1%2Brocm7.2.1.lw.gitff65f5bc-cp312-cp312-linux_x86_64.whl}"
TORCH_WHL_SHA256="${TORCH_WHL_SHA256:-fb45ace0a27e9f0d0e3c4c6efd8932162743f8376f2aa4752a4d31ef5a1bd3d7}"
TORCHVISION_WHL_URL="${TORCHVISION_WHL_URL:-https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/torchvision-0.24.0%2Brocm7.2.1.gitb919bd0c-cp312-cp312-linux_x86_64.whl}"
TORCHVISION_WHL_SHA256="${TORCHVISION_WHL_SHA256:-d5fca8cda173235a3b7434baeebe04c3ebffec3c6fc191e79aa8aa300633f2c9}"
TORCHAUDIO_WHL_URL="${TORCHAUDIO_WHL_URL:-https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/torchaudio-2.9.0%2Brocm7.2.1.gite3c6ee2b-cp312-cp312-linux_x86_64.whl}"
TORCHAUDIO_WHL_SHA256="${TORCHAUDIO_WHL_SHA256:-023d1ce5d847b2a0fbebacf52d35b4c7a233ca07b3dbd0f1cbde84362cbcf33d}"
TRITON_WHL_URL="${TRITON_WHL_URL:-https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/triton-3.5.1%2Brocm7.2.1.gita272dfa8-cp312-cp312-linux_x86_64.whl}"
TRITON_WHL_SHA256="${TRITON_WHL_SHA256:-07787af1d28c273852f897bfeaa7bca29f2fa4a13ca0f28f535832b240ce7016}"
VENV_DIR="${VENV_DIR:-$ARTIFACT_ROOT/environments/wsl-rocm}"
WHEEL_CACHE_DIR="${ROCM_WHEEL_CACHE_DIR:-$ARTIFACT_ROOT/downloads/rocm-$ROCM_VERSION}"
REQ_FILE="$REPO_ROOT/requirements-wsl-rocm.txt"
EXPECTED_TORCH_DISTRIBUTION_VERSION="${EXPECTED_TORCH_DISTRIBUTION_VERSION:-2.9.1+rocm7.2.1.lw.gitff65f5bc}"
EXPECTED_TORCH_RUNTIME_VERSION="${EXPECTED_TORCH_RUNTIME_VERSION:-2.9.1+rocm7.2.1.gitff65f5bc}"
TARGET_USER="${TARGET_USER:-${SUDO_USER:-${USER:-}}}"

WHEEL_CACHE_DIR="$(realpath -m -- "$WHEEL_CACHE_DIR")"
if [[ "$WHEEL_CACHE_DIR" != "$ARTIFACT_ROOT/"* ]]; then
  echo "[bootstrap] ROCm wheel cache must remain under STS2_ARTIFACT_ROOT: $WHEEL_CACHE_DIR" >&2
  exit 2
fi

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
  local variable value
  local exported_environment=""
  for variable in \
    PIP_DEFAULT_TIMEOUT PIP_RETRIES \
    http_proxy https_proxy no_proxy HTTP_PROXY HTTPS_PROXY NO_PROXY; do
    value="${!variable:-}"
    if [[ -n "$value" ]]; then
      exported_environment+="export $variable=$(printf '%q' "$value")"$'\n'
    fi
  done
  script="$exported_environment$script"
  if [[ $EUID -eq 0 && -n "$TARGET_USER" && "$TARGET_USER" != "root" ]] && id "$TARGET_USER" >/dev/null 2>&1; then
    su - "$TARGET_USER" -c "bash -lc $(printf '%q' "$script")"
  else
    bash -lc "$script"
  fi
}

echo "[bootstrap] Repo root: $REPO_ROOT"
echo "[bootstrap] ROCm version: $ROCM_VERSION"
echo "[bootstrap] Torch wheel: $TORCH_WHL_URL"

verify_sha256() {
  local path="$1"
  local expected="$2"
  local label="$3"
  if [[ ! "$expected" =~ ^[0-9a-f]{64}$ ]]; then
    echo "[bootstrap] $label expected SHA-256 is malformed." >&2
    return 2
  fi
  local actual
  actual="$(sha256sum -- "$path" | awk '{print $1}')"
  if [[ "$actual" != "$expected" ]]; then
    echo "[bootstrap] $label SHA-256 mismatch: expected=$expected actual=$actual" >&2
    return 2
  fi
}

download_verified() {
  local url="$1"
  local expected="$2"
  local destination="$3"
  local label="$4"
  local partial="$destination.partial"
  if [[ -f "$destination" ]]; then
    verify_sha256 "$destination" "$expected" "$label"
    echo "[bootstrap] Reusing verified $label: $destination"
    return
  fi
  if [[ -e "$destination" || -L "$destination" ]]; then
    echo "[bootstrap] Refusing non-file wheel-cache entry: $destination" >&2
    return 2
  fi
  rm -f -- "$partial"
  wget --no-verbose --tries=10 --timeout=120 -O "$partial" "$url"
  verify_sha256 "$partial" "$expected" "$label"
  mv -- "$partial" "$destination"
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
  tmp_deb="$(mktemp /tmp/amdgpu-install.XXXXXX.deb)"
  trap 'rm -f "$tmp_deb"' EXIT
  wget --no-verbose --tries=10 --timeout=120 -O "$tmp_deb" "$AMDGPU_INSTALL_DEB_URL"
  verify_sha256 "$tmp_deb" "$AMDGPU_INSTALL_DEB_SHA256" "amdgpu-install package"
  $SUDO apt-get install -y --allow-downgrades "$tmp_deb"
  $SUDO amdgpu-install -y --usecase=wsl,rocm --no-dkms

  if [[ -n "$TARGET_USER" ]] && id "$TARGET_USER" >/dev/null 2>&1; then
    $SUDO usermod -a -G render,video "$TARGET_USER" || true
  fi
}

install_python_env() {
  echo "[bootstrap] Creating/updating venv: $VENV_DIR"
  local wheel_dir="$WHEEL_CACHE_DIR"
  mkdir -p -- "$wheel_dir"
  download_verified "$TORCH_WHL_URL" "$TORCH_WHL_SHA256" \
    "$wheel_dir/torch-2.9.1+rocm7.2.1.lw.gitff65f5bc-cp312-cp312-linux_x86_64.whl" "torch wheel"
  download_verified "$TORCHVISION_WHL_URL" "$TORCHVISION_WHL_SHA256" \
    "$wheel_dir/torchvision-0.24.0+rocm7.2.1.gitb919bd0c-cp312-cp312-linux_x86_64.whl" "torchvision wheel"
  download_verified "$TORCHAUDIO_WHL_URL" "$TORCHAUDIO_WHL_SHA256" \
    "$wheel_dir/torchaudio-2.9.0+rocm7.2.1.gite3c6ee2b-cp312-cp312-linux_x86_64.whl" "torchaudio wheel"
  download_verified "$TRITON_WHL_URL" "$TRITON_WHL_SHA256" \
    "$wheel_dir/triton-3.5.1+rocm7.2.1.gita272dfa8-cp312-cp312-linux_x86_64.whl" "triton wheel"
  run_user_shell "
set -euo pipefail
python3 -m venv '$VENV_DIR'
source '$VENV_DIR/bin/activate'
python -m pip install --no-deps \
  '$wheel_dir/torch-2.9.1+rocm7.2.1.lw.gitff65f5bc-cp312-cp312-linux_x86_64.whl' \
  '$wheel_dir/torchvision-0.24.0+rocm7.2.1.gitb919bd0c-cp312-cp312-linux_x86_64.whl' \
  '$wheel_dir/torchaudio-2.9.0+rocm7.2.1.gite3c6ee2b-cp312-cp312-linux_x86_64.whl' \
  '$wheel_dir/triton-3.5.1+rocm7.2.1.gita272dfa8-cp312-cp312-linux_x86_64.whl'
python -m pip install -r '$REQ_FILE'
python -m pip install -e '$REPO_ROOT' --no-deps --no-build-isolation
python -m pip check
"
}

verify_stack() {
  run_user_shell "
set -euo pipefail
source '$VENV_DIR/bin/activate'
python '$SCRIPT_DIR/verify_rocm_stack.py' \
  --expected-distribution-version '$EXPECTED_TORCH_DISTRIBUTION_VERSION' \
  --expected-runtime-version '$EXPECTED_TORCH_RUNTIME_VERSION'
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
