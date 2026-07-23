# WSL ROCm training for `rl-agent`

This repo can run recurrent V-trace baseline training from WSL while the STS2 game +
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
- `python -m sts2_rl.train`
  - uses the typed live backend and scoped training capability
  - accepts `--device auto` or the ROCm-compatible PyTorch device name `cuda`
- New bridge relay:
  - `scripts/bridge_wsl_relay.py`
  - `scripts/start_wsl_bridge_relay.ps1`
  - `scripts/stop_wsl_bridge_relay.ps1`
- New WSL ROCm bootstrap + launcher:
  - `scripts/bootstrap_wsl_rocm.sh`
  - `scripts/train_grounded_wsl_rocm.sh`

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

## Launch recurrent V-trace v4 training from WSL

```bash
export REPO_ROOT='/mnt/<drive>/path/to/sts2_mcp'
export STS2_ARTIFACT_ROOT='/mnt/<drive>/sts2-artifacts'
cd "$REPO_ROOT/packages/rl-agent"
bash scripts/train_grounded_wsl_rocm.sh \
  --profile default \
  --steps 200000
```

The launcher will:

1. ensure the WSL ROCm venv exists
2. try direct `127.0.0.1` bridge access from WSL
3. only start the Windows relay if direct localhost access fails
4. launch `python -m sts2_rl.train --device cuda --backend live ...`

The deleted MuZero/token-memory compatibility wrappers are not supported.
Checkpoints and logs resolve below `STS2_ARTIFACT_ROOT` through the typed config.

## Launch the observation-v2/macro-credit preheat lineage

The `preheat` profile uses the patched, identity-pinned headless simulator rather
than the Windows live bridge. Before launch, build the locked dependency from
the repository root so the simulator exports the complete visible map graph,
visible next boss, nullable native rest-heal preview and canonical shop item
facts:

```powershell
python .\packages\rl-agent\scripts\build_pinned_headless_sim.py
```

The current v19 campaign is a fresh lineage initialized from the validated v18
checkpoint whose metadata records policy version 3,936:

```bash
export REPO_ROOT='/mnt/<drive>/path/to/sts2_mcp'
export STS2_ARTIFACT_ROOT='/mnt/<drive>/sts2-artifacts'
cd "$REPO_ROOT/packages/rl-agent"
bash scripts/train_preheat_wsl_rocm.sh \
  --initialize-from "<V18_POLICY_3936_CHECKPOINT>"
```

The launcher refuses CPU fallback and runs the complete native-revival game
flow. `--initialize-from` imports only compatible network parameters. The v19
optimizer, FIFO queue, transaction/complete-episode replay, RNGs, counters and
policy versions start fresh at step zero. Do not replace it with `--resume`:
observation v2, config v7 and replay v3 deliberately form a new lineage.

The producer exposes factual visible state only. It does not calculate deck
strength, route desirability, future rewards or future card/shop predictions.
The preheat replay merely reserves part of complete-episode sampling for exact
observed non-combat policy decisions; it never fabricates a target action.
Odd-seed held-out journals and macro sensitivity reports are diagnostic-only
and never train the model.

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
