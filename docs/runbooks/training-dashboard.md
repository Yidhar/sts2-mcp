# Local training dashboard

The RL package includes a loopback-only, read-only dashboard for persistent
training telemetry. It exists so routine progress checks do not require an
interactive repository inspection.

## Start it on Windows

From any PowerShell working directory:

```powershell
& <REPOSITORY_ROOT>\packages\rl-agent\scripts\start_training_dashboard.ps1
```

The launcher uses `STS2_ARTIFACT_ROOT` when it is set. Otherwise it looks for a
sibling `<repository-name>_artifacts\runtime` directory. An explicit root is
also supported:

```powershell
& .\packages\rl-agent\scripts\start_training_dashboard.ps1 `
  -ArtifactRoot <ARTIFACT_ROOT> `
  -Port 8765
```

The dashboard opens at <http://127.0.0.1:8765/>. It runs in a hidden background
process by default. Launcher logs and the informational PID file are written
under `<artifact-root>/monitor/`, never in a run or checkpoint directory.
The launcher reuses an existing service only when its health response identifies
the same normalized artifact root; a dashboard on the same port for another
root is rejected instead of silently showing the wrong experiment.

For foreground diagnostics:

```powershell
& .\packages\rl-agent\scripts\start_training_dashboard.ps1 -Foreground -NoBrowser
```

The package entry point is equivalent:

```powershell
$env:STS2_ARTIFACT_ROOT = "<ARTIFACT_ROOT>"
python -m sts2_rl.monitor_dashboard --open-browser
```

## What the dashboard reports

- authoritative lifecycle state (`run_complete`, interrupt, failure, or
  conservatively labelled stale/unknown);
- cumulative environment steps, episodes, policy version, learner updates,
  throughput and ETA;
- current actor and learner telemetry;
- recent online-training episodes;
- fixed-seed held-out gates, including Act 1, Act 3, full-run, revival, HP-loss,
  deadlock and infrastructure-retry statistics;
- safely published checkpoints and separately labelled atomic staging
  directories;
- candidate-action peaks, incidents and JSONL data-quality warnings.

Exact-resume run segments are joined only through checkpoint provenance.
`model_initialization` creates a new training lineage boundary rather than
pretending to be an exact resume.

Without a `?run=` query the page automatically follows the newest discovered
run, including runs created after the page was opened. Selecting a historical
run pins it; selecting the option marked **latest** restores auto-follow.

## Metric interpretation

Online training episodes and held-out evaluation are deliberately separated.
A training `run_won` is a sampled experience, not a generalization result.
Held-out rates always show their numerator and denominator.

When `curriculum.mode=native-revival-preheat` and `revival_budget=-1`, the page
shows an **unlimited hidden-engine-bailout preheat** badge. Its Act/run coverage is not
a standard zero-revival win rate. The panel therefore reports all-episode
revival/HP cost separately from the conditional Act 1 boundary cost and the
zero/at-most-one-revival metrics.

The configured `device=cuda` value is configuration provenance, not measured
GPU utilization. This first version intentionally avoids shelling out to GPU
drivers from the web service.

## Read-only and failure boundaries

The service:

- binds only to a loopback address;
- accepts GET/HEAD for a fixed UI and fixed JSON endpoints;
- has no stop, resume, delete, checkpoint-load or file-download action;
- never deserializes `.pkl` or `.pt` files;
- incrementally tails only direct `metrics.jsonl` files;
- uses completed `evaluation` summaries instead of repeatedly scanning large
  held-out decision journals;
- ignores `.incomplete-*` checkpoints as recovery points;
- never trusts a bare PID file as proof of liveness.

A checkpoint labelled `valid · size_verified` has a supported atomic manifest,
matching metadata, safe regular payload files, and matching declared sizes. It
has **not** been deserialized, restore-smoke-tested, or hash-verified by this
dashboard.

A stale run without a terminal event is labelled `stale_unknown`, not
"crashed". The default stale threshold is 15 minutes because evaluation and
atomic checkpoint publication can legitimately leave the main metrics file
quiet for several minutes.

## HTTP endpoints

All endpoints are same-origin and return `Cache-Control: no-store`.

| Endpoint | Purpose |
| --- | --- |
| `/` | Dashboard UI |
| `/api/v1/health` | Read-only service health |
| `/api/v1/runs` | Discovered server-issued run keys |
| `/api/v1/snapshot?run=<key>` | Bounded aggregate for one run/continuation chain |

The `run` query accepts only a key returned by `/api/v1/runs`; it is not an
arbitrary filesystem path.
