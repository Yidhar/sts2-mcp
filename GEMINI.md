# GEMINI.md — project context

## Project state

`sts2-mcp` is a local Slay the Spire 2 control and RL monorepo. The repository
is in the architecture-v2 cutover. New work targets contract API `2.0.0`;
`legacy-v1` exists only for an explicit compatibility window.

Do not describe or extend the deleted PPO or MuZero/token-memory/MCTS flows.
The maintained training entry point is:

```text
python -m sts2_rl.train
```

## Canonical components

1. `contracts/`: the wire-format source of truth. JSON Schema 2020-12, OpenAPI,
   fixtures, and generated C#/TypeScript/Python version constants.
2. `game-data/`: policy-free static card/relic/potion facts for offline catalog
   tooling; it is not a trainer input.
3. `mods/sts2-bridge/`: C#/.NET 9 game adapter. It owns factual snapshots, legal
   action handles, state revision, session lifecycle, bounded events, and the
   single-writer mutation gate.
4. `packages/mcp-server/`: Node 22 + TypeScript MCP server using the official
   `@modelcontextprotocol/sdk`. Its default tool profile is `minimal`.
5. `packages/rl-agent/`: Python 3.11+ grounded legal-candidate actor-critic,
   typed v2 backends, immutable normalized reward, replay and checkpoints.
6. `tools/`: contract, game-data, artifact, release, third-party, and repository
   validation utilities.

Component versions come from `release-manifest.json`. Architecture decisions
come from `docs/adr/`; migration status and removal gates come from
`docs/migration/v2-cutover.md`.

## Required dependency direction

```text
contracts -> bridge / mcp-server / rl-agent
game-data -> offline catalog tools
```

- Bridge and MCP must never import through an RL package path.
- Godot, Harmony, and private game reflection belong behind the Bridge game
  adapter boundary.
- Bridge emits facts; it does not own policy, observation encoding, route
  strategy, or training reward.
- RL owns episode, observation, action encoding, reward, curriculum, replay,
  model, checkpoint, and experiment semantics.
- MCP owns protocol, validation, presentation, and small control workflows. It
  does not own RL state, journal persistence, or arbitrary knowledge files.

## Security and mutation invariants

- `player-control` and `training` use separate capability tokens.
- The normal MCP profile is `minimal`; `debug` is privileged and explicit.
- Player-control state must not expose hidden draw-pile order.
- Every v2 mutation requires `request_id`, `session_id`, capability, expected
  revision, and deadline.
- One mutation gate serializes game mutations. Validation, current-handle
  resolution, execution start, and result recording occur as one game-thread
  transaction.
- Replaying an identical request ID returns its retained result and never
  executes twice.
- A legacy mutation is sent once. A timeout is an unknown outcome, not a retry
  invitation.
- Never print session descriptors, bearer tokens, Authorization headers, or
  capability-token maps.
- Session URLs must remain loopback-only.

## Compatibility policy

- Unknown major contract versions and unsupported checkpoint/replay identities
  fail closed.
- Replay and checkpoints must record contract, action ordering, observation,
  reward, dependency-lock and grounded-encoding identities plus stochastic
  continuation state. Valid static game-data may be recorded as optional provenance
  but is not a runtime dependency or model compatibility identity.
- Architecture-v2 code must use maintained module entry points directly and must
  not depend on deleted wrappers or `legacy/`.
- Historical PPO, attention, AutoSlay, Draft Tracker, journal/knowledge, checked-
  in logs, and release binaries are not architecture-v2 control-plane features.

## Artifact policy

Checkpoint, optimizer state, replay, datasets, logs, virtual environments,
release binaries, PIDs, and decompilations live outside the checkout under an
operator-selected `STS2_ARTIFACT_ROOT`. Never delete or move existing training
assets without first stopping writers, producing an inventory, verifying the
destination, and keeping a backup. `tools/artifacts/move_to_artifact_root.ps1`
is dry-run unless `-Execute` is supplied.

## Build and verification

Repository root:

```powershell
python .\tools\contracts\check_contracts.py
python .\tools\game_data\verify_manifest.py
python .\tools\release\check_versions.py
python .\tools\ci\check_dependency_locks.py
python .\tools\ci\check_markdown_links.py
python .\tools\ci\check_generated.py
python .\tools\ci\check_repository.py
```

MCP:

```powershell
Set-Location .\packages\mcp-server
npm ci
npm run typecheck
npm test
```

Bridge core, without retail game assemblies:

```powershell
dotnet run --project .\mods\sts2-bridge\tests\BridgeCore.Tests\BridgeCore.Tests.csproj --configuration Release
```

Full Bridge build, with an authorized local game installation:

```powershell
$env:STS2_DIR = '<PATH_TO_STS2>'
dotnet build .\mods\sts2-bridge\sts2-bridge.csproj
```

Deployment is opt-in with `-p:Sts2Deploy=true`.

RL:

```powershell
Set-Location .\packages\rl-agent
python -m pip install -r requirements-bootstrap.lock
python -m pip install -r requirements-dev.lock
python -m pip install -e . --no-deps
python -m pytest tests -q -p no:cacheprovider
python -m sts2_rl.train --dry-run
```

## Documentation rules

- `README.md`, `docs/architecture.md`, `contracts/README.md`, component READMEs,
  accepted ADRs, and migration pages are normative.
- Dated experiment documents are historical unless a canonical page explicitly
  marks them active.
- `docs/generated/` is generated; do not hand-edit it.
- Portable examples use placeholders or environment variables, never a
  developer-specific absolute path.
