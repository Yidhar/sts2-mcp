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

The launcher prefers the package virtual environment only when it can import
the complete dashboard entry point. An incomplete recovered environment falls
back to the system `python` rather than starting a monitor that cannot serve
the map-replay endpoint.

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
- an on-demand held-out run replay with an Act-scoped graphical route map,
  room timeline, final deck/relic/potion chips, card-reward candidates and
  selected/skip action, macro choices, authoritative per-floor HP-loss/revival
  deltas, combat outcomes, deadlock cycles, and
  checkpoint/policy/seed/simulator provenance;
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
- lists held-out journals from filename, stat, and the bounded first
  `evaluation_started` row only; it scans compact `decision` rows only after an
  operator explicitly selects that journal;
- consumes `decision_snapshot` rows only during explicit drill-down, projects
  only bounded final/latest player-loadout facts, discards the full rich
  observation, keeps at most four projections in a
  `path + size + mtime_ns` LRU cache, and never writes a derived index into the
  artifact tree;
- reconstructs maps only after an operator opens one episode, by resetting the
  exact pinned HeadlessSim with the recorded seed and replaying the journal's
  recorded action indexes; identity or state/action divergence fails closed,
  and successful maps live only in an eight-entry process-memory cache;
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
| `/api/v1/heldout-journals?run=<key>` | Cheap journal directory/provenance list; does not parse decision rows |
| `/api/v1/heldout-episodes?run=<key>&journal=<key>` | Lazily parse one selected journal and return a bounded episode index |
| `/api/v1/heldout-episode?run=<key>&journal=<key>&episode=<id>` | One bounded run projection with route/floor/loadout/reward/anomaly facts |
| `/api/v1/heldout-replay-map?run=<key>&journal=<key>&episode=<id>` | Deterministically reconstruct all visited Act maps and exact macro outcomes from seed plus recorded actions; memory-cached |

The `run`, `journal`, and `episode` queries accept only server-issued keys from
the preceding endpoint. They are not arbitrary filesystem paths; extra or
repeated query parameters fail closed.

The 15-second aggregate refresh calls only `runs` and `snapshot`. It never
triggers a full held-out parse. On a representative 59.05 MiB / 12,629-row
journal, a local cold projection including bounded rich-snapshot facts took
about 0.72 seconds, the stat-keyed cache
hit took under 0.1 milliseconds, the 16-episode index encoded to about 6.5 KiB,
and the largest single compact episode detail was about 280 KiB. These figures
are a local implementation check, not a throughput guarantee.

Card-reward candidates are explicitly labelled as `policy_topk` coverage: the
view does not claim full legal-candidate coverage when the compact journal did
not persist it. Per-floor HP loss and revivals use deltas of the episode's
authoritative monotonic `player_hp_lost` and `revivals_used` counters; raw HP
decrease is shown only as a separate net resource change. Missing fields such
as a legacy journal's game version or event option text are displayed as
`未记录` rather than inferred.

The graphical map uses one durable contract for every supported journal:

1. the compact route and `policy_topk` branches render immediately;
2. the monitor resets the matching simulator from the recorded seed, replays
   the exact recorded action indexes, captures each visited Act's `map.nodes`,
   and projects macro transitions such as healing, card upgrades/removals,
   purchases, and acquired entities from their exact before/after states;
3. a repeated view uses the stat-keyed in-memory result rather than replaying.

The trajectory journal deliberately does **not** persist one full map topology
per Act. Existing `sts2-trajectory-journal-v4` logs already contain the seed,
pinned apphost/managed-simulator hashes, and every dispatched action index, so
they are fully compatible with reconstruction and require neither a downgrade
path nor a new logging ABI. In a real three-Act old-journal validation, the
parser found and replayed all 958 decisions without state/action divergence,
captured 66/51/51 nodes for Acts 1/2/3, and completed in about 14.5 seconds on
the test machine. A cached HTTP view returned in about 71 milliseconds. Timings
are hardware-, process-, and episode-dependent and are not a service-level
guarantee.

New `sts2-trajectory-journal-v5` summaries additionally preserve a bounded,
reviewed action-identity projection for card, shop-item, rest-option, and
selection fields. This lets the monitor name upgraded, removed, and purchased
entities without duplicating complete card descriptions on every decision.
Version-4 logs remain fully supported: the same deterministic seed-and-action
replay that reconstructs the map also recovers exact macro action outcomes.

The legacy response field `forced=true` means only that
`legal_action_count == 1`. The dashboard renders this as automatic advancement
with an `only_legal_action` explanation; it must not be interpreted as a model
preference among multiple strategies.

The final loadout follows the same rule. It is rendered only from the latest
available bounded player snapshot, with snapshot step/reason and omission
coverage retained in the API. It is never reconstructed from card-reward
choices because shops, transforms, removes, upgrades, events, and other deck
mutations would make that reconstruction false.
