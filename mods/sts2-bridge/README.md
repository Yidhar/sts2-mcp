# STS2 Bridge

`sts2-bridge` is the game-process adapter for STS2. It owns factual snapshots,
legal action handles, serialized mutation execution, session discovery, and the
loopback HTTP/SSE transport. Policy, reward shaping, model observations, and
route strategy belong outside this mod.

The Bridge exposes contract API `2.0.0`. The original endpoints exist only as an
explicitly enabled, independently authenticated `legacy-v1` migration surface.
Canonical schemas and generated
version constants live under `../../contracts`; embedded static game data lives
under `../../game-data`.

## Source boundaries

`BridgeGameApi` is a partial facade rather than a single implementation file.
The boundaries are intentional:

- `BridgeGameApi.cs` owns request/frontier orchestration and fingerprints only;
  it is held below 1,600 lines.
- `BridgeGameApi.Actions.*.cs` owns legal-action construction and execution,
  separated into registry, combat, shop, menu, navigation, and selection flows.
- `BridgeGameApi.Payloads.*.cs` owns factual combat, enemy, selection, and
  navigation payload projection.
- `BridgeGameApi.GameAdapter.*.cs` isolates game/UI/reflection compatibility,
  context capture, card semantics, glossary extraction, and localization text.
- `BridgeGameCompatibilityGate.cs` owns the exact retail-assembly profile
  registry, assembly identity capture, and required startup capability probes.
- `BridgeGameApi.Snapshots.*.cs` owns frontier/state payload assembly and SSE
  publication; `BridgeGameApi.Models.cs` owns the internal facade models.
- `BridgeGameApi.GameCompatibility.cs` is the narrow version-difference shim.
- `BridgeGameApi.Env*.cs` is the remaining privileged legacy/training adapter;
  it must not leak reward or policy ownership back into the control surface.

Dependency-free source-boundary tests require every newly extracted semantic
partial to stay below 2,000 lines, verify definition-level ownership, and reject
context-free empty catches. New behavior must extend the owning partial instead
of rebuilding the former monolith.

## Safety model

- HTTP binds to `127.0.0.1` only.
- Session discovery publishes capability-scoped bearer tokens atomically.
- `player-control` is enabled by default and emits an explicitly player-visible
  projection; the complete mod is correctly declared gameplay-affecting and not
  globally visible-only.
- `training` v2 is disabled by default and uses a different token.
- `legacy-v1` is disabled by default, has an independent `legacy-privileged`
  token, and never accepts the player or training token.
- Public `/health` contains readiness/version only; detailed `/v2/health`
  requires any currently published scoped token.
- Every v2 mutation requires a UUID `request_id`.
- Results are retained in a bounded 10-minute idempotency cache.
- Reusing a request ID with a different payload fails closed.
- A single mutation gate serializes legacy actions, environment operations, and
  v2 commands.
- v2 action revision validation, current-handle re-resolution, and action start
  happen in one main-thread work item.
- `/v2/state` uses an explicit root allowlist and recursively removes hidden
  draw-pile order, legacy hashes, and policy/training annotations.
- `/v2/events` carries only bounded revision notifications; clients fetch the
  latest `/v2/state` snapshot after observing a newer revision.
- v2 training results contain canonical `transition_facts`; scalar `reward` is
  `null`/not computed and reward authority is the external RL service.
- Request bodies, concurrent requests, event streams, main-thread queue depth,
  and per-frame pump work are bounded.
- Before any Harmony patch or HTTP listener starts, the Bridge must match an
  exact audited retail profile (assembly name/version, informational version,
  and module version ID) and pass every required type/member probe. Unknown
  builds, missing members, probe exceptions, and patch activation failures set
  compatibility health to `degraded` and return from initialization without a
  listener, session descriptor, or mutation surface. There is intentionally no
  environment-variable bypass.

AutoSlay is not part of the Bridge action registry or any player payload.
Automation requires a separate external process and cannot be invoked through
player-control, training, or legacy-v1.

## Permission matrix

| Endpoint family | Public | `player-control` | `training` | `legacy-privileged` |
|---|---:|---:|---:|---:|
| `/`, `/health` | readiness/version | readiness/version | readiness/version | readiness/version |
| `/v2/health` | denied | detailed health | detailed health, when enabled | detailed health, only when legacy is enabled |
| `/v2/state`, `/v2/events`, `/v2/commands` | denied | allowed | denied | denied |
| `/v2/commands/{request_id}` | denied | player entries only | training entries only | legacy entries only; none are currently created |
| `/v2/env/spec`, `/v2/env/state`, `/v2/env/combat_catalog` | denied | denied | allowed | denied |
| `/v2/env/reset`, `/v2/env/step` | denied | denied | allowed | denied |
| `/state`, `/action`, `/events`, `/env/*` | denied | denied | denied | allowed only when legacy-v1 is explicitly enabled |
| `/static/export` | denied | denied | denied | `410 Gone` only when legacy-v1 is explicitly enabled |

Request IDs are global across capabilities. Reusing an ID with a different
capability or payload fails closed. Status results are readable only with the
same capability that created them. Unexpired completed IDs are never evicted to
make room; a full store rejects new IDs until a completed entry expires.

## Endpoints

### Contract v2

| Method | Path | Capability | Purpose |
|---|---|---|---|
| `GET` | `/v2/health` | any enabled scope | Authenticated transport, pump, queue, snapshot diagnostics |
| `GET` | `/v2/state` | `player-control` | Canonical player-visible state and stable legal handles |
| `POST` | `/v2/commands` | `player-control` | Strict idempotent action command |
| `GET` | `/v2/commands/{request_id}` | either v2 scope | Retained/in-flight command status |
| `GET` | `/v2/events` | `player-control` | Bounded drop-oldest SSE frontier stream |
| `GET` | `/v2/env/spec` | `training` | Reward-free environment capabilities and schema metadata |
| `GET` | `/v2/env/state` | `training` | Privileged observation, legal actions, episode and frontier revision |
| `GET` | `/v2/env/combat_catalog` | `training` | Available combat-sandbox encounters |
| `POST` | `/v2/env/reset` | `training` | Idempotent revision-checked full-run/combat reset |
| `POST` | `/v2/env/step` | `training` | Idempotent episode/index-checked step |

A v2 reset must include `expected_state_version` from the latest
`/v2/env/state`. The Bridge validates that revision on the game thread inside
the global mutation gate before reset work begins. Reset and step results expose
real `state_version_before`/`state_version_after`; `committed_state_version`
uses the actual final frontier revision and never the episode `step_index`.

Player commands require `deadline_utc` no more than 120 seconds in the future,
a non-empty `action_handle`, and `wait_after_ms` from 0 through 5000. Unknown or
incorrectly cased fields in typed v2 request envelopes are rejected rather than
ignored. Legacy-v1 request parsing remains case-insensitive for compatibility.

Example player command:

```json
{
  "request_id": "9cbaf17d-1baa-4d7a-b790-b686ad72db2c",
  "session_id": "<active-session-id>",
  "capability": "player-control",
  "expected_state_version": 42,
  "deadline_utc": "2026-07-11T12:00:00Z",
  "command": {
    "kind": "perform_action",
    "action_handle": "end_turn",
    "wait_after_ms": 0
  }
}
```

A command result has one of:

- `rejected_before_execution`
- `accepted`
- `executing`
- `committed`
- `outcome_unknown`

A duplicate identical request returns the retained result with
`replayed_result=true` and never executes again. If an action may have started
but its response/outcome cannot be confirmed, the result is
`outcome_unknown`; clients must query the request ID rather than invent a new
retry policy.

### Legacy migration surface

Public `/health` is no longer a legacy diagnostic endpoint; it always returns a
minimal readiness/version payload. The original `/state`, `/action`, `/events`,
and `/env/*` routes return `404 legacy_v1_disabled` unless
`STS2_BRIDGE_ENABLE_LEGACY_V1=true`. When enabled they require only the separate
`legacy-privileged` token; player-control and training tokens receive `403`.
Legacy mutations remain non-idempotent and should be migrated to v2. Legacy
environment responses still expose a scalar reward marked
`legacy-bridge-v1-deprecated` for compatibility only.

`POST /static/export` is permanently retired and returns `410 Gone`. Use the
dependency-free offline publisher instead:

```powershell
$env:STS2_ARTIFACT_ROOT = '<ABSOLUTE_ARTIFACT_ROOT_OUTSIDE_CHECKOUT>'
python .\tools\catalog-export\export_catalog.py --output 'catalogs\verified-game-data'
```

The offline command validates every source and copied file against the canonical
`game-data/manifest.json`; it never performs game-main-thread reflection or
accepts a write path through HTTP.

## Session discovery

The descriptor is written beneath the current user's application-data STS2
`bridge` directory as `session.json` or `session_<instance>.json`. Publication
uses a same-directory temporary file, flush-to-disk, and atomic replacement.
On owned shutdown it is removed only if its `session_id` still matches.

New clients should read:

- `api_versions`
- `schema_version`
- `capabilities`
- `capability_tokens`
- `capability_details`
- `process_started_at_utc`
- `game_compatibility` (matched profile, exact assembly identity, and every
  startup probe result)

The top-level `token` member is absent by default and appears only when legacy-v1
is explicitly enabled; in that case it duplicates only the independent
`legacy-privileged` credential. Contract-v2 clients must never fall back from a
missing scoped token to the top-level token.

Do not print the descriptor or tokens in tools, logs, errors, or test snapshots.

## Build

The project requires the installed game's managed assemblies. Set one of:

```powershell
$env:STS2_DIR = '<STS2_INSTALL_DIR>'
# or
$env:SLAY_THE_SPIRE_2_DIR = '<STS2_INSTALL_DIR>'
```

A normal build **does not deploy into the game**:

```powershell
dotnet build .\mods\sts2-bridge\sts2-bridge.csproj
```

Deployment is explicit:

```powershell
dotnet build .\mods\sts2-bridge\sts2-bridge.csproj -p:Sts2Deploy=true
```

`Sts2Dir`, `Sts2DataDir`, and `Sts2ModDir` can be supplied as MSBuild
properties. `Sts2SkipDeploy=true` remains an emergency compatibility override,
but deployment is already off unless `Sts2Deploy=true` is set.

## Dependency-free core tests

The idempotency store, result replay semantics, command/environment service,
HTTP host compilation, and mutation gate can be checked without commercial
game assemblies:

```powershell
dotnet run --project .\mods\sts2-bridge\tests\BridgeCore.Tests\BridgeCore.Tests.csproj --configuration Release
```

A full Bridge build and live-game smoke still require the matching STS2,
Harmony, and GodotSharp assemblies.

## Retail game compatibility gate

The currently audited profile is `retail-2026-06-23-5926027`:

| Identity field | Required value |
|---|---|
| Assembly | `sts2` |
| Assembly version | `0.1.0.0` |
| Informational version | `0.1.0+59260271157f76a2896f0eab5bc6ea1245d8b314` |
| Module version ID | `97f10687-c306-4798-ab75-8b9f23f34dfb` |

The profile probes the lifecycle patch targets plus critical combat, run setup,
and reward-skip members before activation. This is an activation gate, not a
claim that all private reflection has disappeared: the localized
`BridgeGameApi.GameAdapter.*` and training sandbox code still adapt runtime UI
and private members. Exact identity matching prevents those paths from running
against an unaudited retail binary; live scene/action fixtures remain a release
requirement after the gate passes.

When a retail update changes any identity field, startup logs
`unsupported_game_assembly`, `health=degraded`, `http_started=false`, and
`mutation_enabled=false`. If identity matches but a required member has moved,
the log reports `required_game_capability_missing` and the failed probe IDs.
The operator must not edit the fingerprint or weaken a probe merely to make an
update start. Add a new profile only after auditing the new binary, updating
adapter shims, passing Bridge core/full builds, and recording a live-game smoke.

## Runtime configuration

| Variable | Default | Meaning |
|---|---:|---|
| `STS2_BRIDGE_INSTANCE_ID` | empty | Numeric parallel instance ID, `0..3000` |
| `STS2_BRIDGE_ENABLE_TRAINING_V2` | `false` | Publish and accept the separate training capability/token |
| `STS2_BRIDGE_ENABLE_LEGACY_V1` | `false` | Publish the independent legacy-privileged token and enable legacy routes |
| `STS2_BRIDGE_HEARTBEAT_TIMEOUT_MS` | `5000` | Pump-staleness threshold, clamped to `1000..60000` ms |

Each instance receives a ten-port range beginning at 27100. Invalid instance
IDs fail startup rather than being interpolated into paths or port numbers.

## Main-thread lifecycle

The coordinator completes queued Tasks with errors when detached, bounds both
queue capacity and pump work, and reports its heartbeat through `/v2/health`.
The HTTP host limits request/event concurrency and body size, cancels in-flight
operations on shutdown, closes the listener, removes its owned descriptor, and
unpatches Harmony during process/assembly unload.
