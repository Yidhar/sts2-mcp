# Synchronous macro-credit dashboard

The new synchronous course does not emit the legacy
`runs/<lineage>/run-<uuid>/metrics.jsonl` layout. The old `monitor_dashboard`
therefore cannot correctly discover this course. Do not point the old UI at it
and interpret a previous lineage's metrics as the current training run.

The separate `sts2_rl.macro_dashboard` reads one explicitly selected external
run's `course.json`, `status.json`, `cohorts/*/update.json`, and a bounded tail of
its newest active trace. An optional, explicitly selected invocation directory
adds the recorded launch target/PIDs and exit-file/log-size observations. Its
lineage must match before those observations are used.

## Safety and measurement scope

- IPv4 loopback only; default URL is `http://127.0.0.1:8766/`.
- Read-only HTTP. No start/stop/restart, seed skipping, parameter change, arbitrary
  file selection, simulator startup, or checkpoint deserialization endpoints.
- Uses Python standard library only; no torch/model/learner imports or GPU allocation.
- Windows artifact readers explicitly share delete access while requesting read
  access only. The actual supported writer is the **WSL/POSIX trainer**: an
  integration test holds a Windows dashboard read handle while a WSL Python
  process successfully performs atomic `os.replace`, and the old read handle
  still sees its complete old record. Ordinary Windows `open()` fails this
  same WSL-writer test. Native Windows `MoveFileEx`/`os.replace` semantics still
  differ; Windows-reader/native-Windows-trainer live use is **not supported**.
  Run both reader and trainer in WSL for that alternative rather than assuming
  a Windows-native writer has been certified.
- Does not create translated legacy metrics or alter the training source manifest.
- A writer-lock file, recorded PID, absent exit file or recently modified trace
  is not a verified live-process claim. The UI labels these as recorded/observed
  activity, not OS process proof.
- Displayed commits come from course status and immutable update metrics.
  The checkpoint remains recovery authority; a temporarily incomplete metrics
  publication is called out instead of guessed away.
- GPU metrics are **learner-window CUDA peaks**, not full-course/system peaks.
  RSS is the **process historical high-water mark**.
- Root candidate `λ-return Y` is an estimated return, not a win rate or a newly
  introduced action-Q model head. Projected `q || teacher` KL is
  distinct from measured post-update `policy || teacher` KL. No value calibration,
  quality promotion or unattended-long-run certification is implied.
- JSON reads are capped at 1 MiB, trace-tail reads at 1 MiB, history at 200 recent
  attempts, and directory discovery at 10,000 attempts. Incomplete data is not
  replaced with an unrelated run.

## Windows background start

Use an available standard-library Python, preserving the pinned ROCm runtime
for training. The dashboard needs no environment installation. Example from
PowerShell; the executable below was checked to import the dashboard package:

```powershell
$package = 'E:\game\project\sts2_mcp\packages\rl-agent'
$root = 'E:\game\project\sts2_mcp_artifacts\runtime'
$run = Join-Path $root 'hybrid_runs\macro-credit-v1-20260905-smoke01'
$invocation = Join-Path $root 'analysis\macro-credit-pilot-20260905-20updates'
$monitor = Join-Path $root 'monitor\macro-credit-8766'
New-Item -ItemType Directory -Force -Path $monitor | Out-Null
$env:PYTHONDONTWRITEBYTECODE = '1'
$arguments = @('-u', '-m', 'sts2_rl.macro_dashboard',
  '--artifact-root', $root, '--run-dir', $run, '--invocation-dir', $invocation,
  '--host', '127.0.0.1', '--port', '8766')
# First inspect http://127.0.0.1:8766/health and the port owner. Reuse only when
# schema, run_dir and lineage match; do not start a duplicate or kill another service.
$process = Start-Process -FilePath 'C:\Python313\python.exe' -ArgumentList $arguments `
  -WorkingDirectory $package -WindowStyle Hidden -PassThru `
  -RedirectStandardOutput (Join-Path $monitor 'stdout.log') `
  -RedirectStandardError (Join-Path $monitor 'stderr.log')
Invoke-RestMethod 'http://127.0.0.1:8766/api/v1/snapshot'
```

The dashboard can be stopped by its independently recorded dashboard PID after
verifying that PID still belongs to this service. This does not stop training.
Never use the WSL training or wrapper PID as the dashboard PID.

## HTTP surface

| Route | Purpose |
| --- | --- |
| `/` | New-course UI, 2-second refresh |
| `/health` or `/api/v1/health` | Schema, selected run/lineage, read-only flag, server PID |
| `/api/v1/snapshot` | Current selected-run observations and bounded update history |

No route accepts a run/file query parameter. Remote Host headers are refused;
mutation methods return 405. The API must name
`sts2-macro-credit-dashboard-v1` and the selected new lineage, not the old
`sts2-training-dashboard-v1` schema.

## Current verification and startup limitation — 2026-09-05

**The dashboard service has not been started.** Ordinary Windows
`Start-Process -WindowStyle Hidden` requests were rejected by the execution
policy before process creation, both from the dashboard task and from the root
task. No escalation, alternative launcher or startup retry is being used to
work around that restriction. There is no confirmed live dashboard PID or
live service URL to report. `http://127.0.0.1:8766/` is the prepared address,
not a claim that a service is listening.

The implementation is verified independently of that launch limitation:

- **13 focused tests** exercise a short-lived local HTTP test server and verify the
  new schema, exact selected-run identity, refresh, partial JSONL handling,
  metadata limits, invocation binding, loopback Host checks, mutation rejection,
  source-boundary protection, single-flight UI polling, and nonblocking atomic
  metadata replacement by the actual WSL/POSIX writer. This test server is shut
  down at test completion. Tests do not establish native Windows writer support.
- Ruff and strict mypy pass.
- A read-only `MacroDashboardStore` call against the real selected run observed
  `sts2-macro-credit-dashboard-v1`, lineage `macro-credit-v1-20260905-smoke01`,
  committed update 6, phase `updating`, next training seed 13400012, invocation
  target update 22, recorded WSL PID 271 / Windows wrapper PID 74140, empty
  stderr, and no dashboard warnings. This is a point-in-time file read, not a
  running HTTP service or current-process-liveness claim.

The Windows-reader / WSL-writer integration is explicitly opt-in on Windows
(it requires the existing `Ubuntu-24.04` distribution, not just `wsl.exe`).
Without that flag the Windows focused suite skips this integration; POSIX
executes its native replacement test directly. Reproduce the actual supported
writer contract from the package directory without starting a dashboard service:

```powershell
Set-Location 'E:\game\project\sts2_mcp\packages\rl-agent'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:STS2_RUN_WSL_INTEGRATION = '1'
C:\Python313\python.exe -m pytest tests/test_macro_dashboard.py -q -p no:cacheprovider `
  --basetemp=E:\game\project\sts2_mcp_artifacts\runtime\diagnostics\macro-dashboard-unit-tests
```

The integration holds the read handle open throughout WSL `os.replace` and
checks both the complete old record through that handle and the new record
through the destination path. This does **not** assert that native Windows
`os.replace` can replace an open destination. The final default/opt-in test and
static-check transcript is stored outside source at
`E:/game/project/sts2_mcp_artifacts/runtime/diagnostics/macro-dashboard-verification.txt`.

The Windows command above is supplied for the user to run manually in their
own terminal if desired. First recheck port ownership, then verify `/health`
and `/api/v1/snapshot` name the expected schema and selected lineage. Do not
interpret a different service's page as this dashboard.

Training itself was started separately and is documented in
[macro-credit-training.md](E:/game/project/sts2_mcp/docs/runbooks/macro-credit-training.md).
The dashboard launch restriction does not stop or restart that training.
