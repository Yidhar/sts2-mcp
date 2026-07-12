# WSL ROCm training for `rl-agent`

This repo can now run MuZero training/evaluation from WSL while the STS2 game +
bridge stay on Windows.

## System networking prerequisite

WSL is now configured through:

`C:\Users\<WINDOWS_USER>\.wslconfig`

```ini
[wsl2]
networkingMode=mirrored
dnsTunneling=true
autoProxy=true
```

After editing `.wslconfig`, restart WSL once:

```powershell
wsl.exe --shutdown
```

With mirrored networking enabled, WSL can now reach the Windows bridge
directly at the loopback URL already written into `session.json`
(`http://127.0.0.1:27100/` in the current setup).

## What changed

- `sts2_env/bridge_client.py`
  - supports `STS2_BRIDGE_BASE_URL` override
  - resolves Windows session paths correctly from WSL
- `python -m muzero.train` / `python -m muzero.evaluate`
  - normalize Windows paths when launched from WSL
  - accept `--device auto` / `--device rocm` aliases (`rocm -> cuda`)
  - print WSL bridge/device setup at startup
- New bridge relay:
  - `scripts/bridge_wsl_relay.py`
  - `scripts/start_wsl_bridge_relay.ps1`
  - `scripts/stop_wsl_bridge_relay.ps1`
- New WSL ROCm bootstrap + launcher:
  - `scripts/bootstrap_wsl_rocm.sh`
  - `scripts/train_muzero_wsl_rocm.sh`

## Why the relay still exists

Your session file still points at a Windows loopback URL like:

```json
"base_url": "http://127.0.0.1:27100/"
```

With mirrored networking enabled, WSL can reach this directly.

The relay remains as a fallback for machines where mirrored networking is not
available or gets disabled later.

## One-time bootstrap inside WSL

```bash
export REPO_ROOT='/mnt/<drive>/path/to/sts2_mcp'
export STS2_ARTIFACT_ROOT='/mnt/<drive>/sts2-artifacts'
cd "$REPO_ROOT/packages/rl-agent"
bash scripts/bootstrap_wsl_rocm.sh
```

This installs:

- AMD WSL ROCm runtime (`amdgpu-install --usecase=wsl,rocm --no-dkms`)
- a rebuildable WSL venv at
  `$STS2_ARTIFACT_ROOT/environments/wsl-rocm` (never inside the checkout)
- the four exact ROCm wheel artifacts in the persistent verified cache at
  `$STS2_ARTIFACT_ROOT/downloads/rocm-7.2.1`; an existing cache entry is reused
  only after its locked SHA-256 passes
- PyTorch from the reviewed ROCm wheel index, with both its distribution metadata
  version and its runtime `torch.__version__` checked (the upstream strings differ)
- Python deps from the hashed `requirements-wsl-rocm.txt`
- this project in editable mode without build isolation, using the already locked
  `setuptools`/`wheel`, followed by `pip check` and a real GPU visibility check

## Launch MuZero training from WSL

```bash
export REPO_ROOT='/mnt/<drive>/path/to/sts2_mcp'
export STS2_ARTIFACT_ROOT='/mnt/<drive>/sts2-artifacts'
cd "$REPO_ROOT/packages/rl-agent"
bash scripts/train_muzero_wsl_rocm.sh \
  --resume-from "$STS2_ARTIFACT_ROOT/checkpoints/your_checkpoint" \
  --log-dir "$STS2_ARTIFACT_ROOT/runs/example" \
  --checkpoint-dir "$STS2_ARTIFACT_ROOT/checkpoints/example" \
  --total-timesteps 200000
```

The launcher will:

1. ensure the WSL ROCm venv exists
2. try direct `127.0.0.1` bridge access from WSL
3. only start the Windows relay if direct localhost access fails
4. launch `python -m muzero.train --device cuda ...`

The deleted top-level compatibility wrappers are not supported. Maintained WSL
automation invokes the `muzero` module entry points directly.

## Custom bridge session path

```bash
export STS2_BRIDGE_SESSION_FILE='/mnt/c/Users/<WINDOWS_USER>/AppData/Roaming/SlayTheSpire2/bridge/session.json'
```

## Stop the Windows relay

From Windows PowerShell:

```powershell
Set-Location <PATH_TO_REPOSITORY>
.\packages\rl-agent\scripts\stop_wsl_bridge_relay.ps1
```

## If GPU is still not visible

Check:

```bash
source "$STS2_ARTIFACT_ROOT/environments/wsl-rocm/bin/activate"
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
rocminfo | head -40
```

If that still fails, update the Windows AMD WSL-capable driver and restart WSL:

```powershell
wsl.exe --shutdown
```
