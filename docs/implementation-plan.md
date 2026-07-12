# Architecture-v2 implementation plan

Status: active migration plan. Completion claims must be backed by the named tests;
the checkbox authority is [`migration/v2-cutover.md`](./migration/v2-cutover.md).

## Operating rules

1. Preserve user work and training assets before changing layout.
2. Do not perform a destructive cleanup on a dirty training worktree.
3. Make the contract executable before deleting compatibility code.
4. Keep v1 and v2 dual-stack only for a measured migration window.
5. Fail closed on unknown outcome, incompatible version, or unsupported checkpoint.
6. Do not start expensive long training while environment/reward/parity gates are red.

## Phase 0 — baseline and asset protection

- Preserve the pre-refactor dirty patch, Git identity, and untracked-file hashes.
- Inventory checkpoint, replay, optimizer, dataset, log, virtual-environment, and
  release directories.
- Record third-party commit and license status.
- Capture golden state/command/environment fixtures.
- Stop writers before any artifact move.

Exit: every irreplaceable asset is backed up or represented by a verified manifest.

## Phase 1 — repository and test baseline

- Track canonical docs, schemas, scripts, and component READMEs.
- Stop tracking logs, PIDs, binaries, decompilations, and temporary slices.
- Add explicit ignore, line-ending, editor, runtime-version, and dependency locks.
- Make the existing Python suite green or quarantine an obsolete test with a written
  replacement gate.
- Add dependency-free Node and Bridge-core tests and required CI.
- Remove credential disclosure from status, errors, smoke output, and logs.

Exit: a clean checkout can run every dependency-free CI job reproducibly.

## Phase 2 — contract-first v2

- Define session, health, state, command, command-result, environment, and
  transition schemas.
- Publish OpenAPI, fixtures, and generated cross-language version constants.
- Advertise API/schema/action/reward/observation identities and capabilities.
- Validate generated files and release versions in CI.
- Keep v1 read compatibility without extending its schema.

Exit: C#, TypeScript, and Python consume the same versions and fixtures.

## Phase 3 — Bridge command and lifecycle safety

- Serialize all game mutations through one bounded gate.
- Add UUID request identity, dedupe TTL/capacity, payload mismatch rejection, and
  command-status lookup.
- Validate revision and re-resolve legal handles atomically on the game thread.
- Serialize reset/step by episode ID and expected step index.
- Split player-control and training tokens; disable training v2 by default.
- Bound HTTP bodies/concurrency, SSE streams, main-thread queue, and pump budget.
- Publish atomic session descriptors and complete all waiters on shutdown.
- Report game-thread heartbeat rather than only transport health.

Exit: duplicate, lost-response, concurrent-revision, reset/step race, startup failure,
and shutdown tests pass.

## Phase 4 — MCP TypeScript cutover

- Replace the hand-written JSON-RPC monolith with the official MCP SDK.
- Validate every input and redact credentials recursively.
- Restrict session forwarding to loopback.
- Default to `minimal`; make training explicit `debug` only.
- Require strict revision; never auto-retry a legacy mutation.
- Use v2 request IDs and status reconciliation when advertised.
- Remove the in-process RL state machine, journal/knowledge persistence, AutoSlay,
  and obsolete live-only runners from the control-plane package.

Exit: SDK integration, profiles, redaction, stale session, loopback, v2 command,
legacy one-shot, and timeout tests pass.

## Phase 5 — Bridge decomposition

Move behavior from the legacy static facade into contracts, host/transport, session,
commanding, snapshots/events, player control, training control, and a versioned game
adapter. Remove reward and catalog export from gameplay HTTP. Keep a facade only while
callers are being migrated; the legacy files are in a no-growth state.

Exit: Godot/Harmony/private reflection is isolated, Bridge has no RL dependency, and
full build plus live scene fixtures pass.

## Phase 6 — RL backend, environment, and reward

- Use one typed `EnvironmentBackend` for live and headless execution.
- Use one episode controller for full-run and combat scenarios.
- Replace duplicated Env reset/step/recovery state machines with composable services.
- Make a pure versioned reward calculator the sole reward authority.
- Make HeadlessSim path/config portable, lifecycle bounded, and contract-compatible.
- Enforce live/headless transition and reward parity.
- Reduce hard guards to legality, safety, and explicit deadlock recovery; record any
  executed-action difference.

Exit: both backends pass the same contract suite and parity gate.

## Phase 7 — trainer and artifact migration

- Replace trainer mixins and large constructor/CLI surfaces with typed configuration
  and Collector, Learner, Evaluator, ReplayStore, CheckpointManager, Curriculum, and
  Telemetry services.
- Write checkpoints atomically with checksums and completion markers.
- Record Git state plus contract, data, reward, observation, action, dependency, and
  parent-checkpoint identities.
- Provide explicit migrators for supported old replay/checkpoint formats.
- Move runtime outputs to `STS2_ARTIFACT_ROOT` and enforce retention.
- Archive/delete PPO, duplicate wrappers, hard-coded launchers, and obsolete probes
  only after their replacement tests and migration gates pass.

Exit: deterministic resume restores optimizer, replay, scheduler, step, and experiment
identity; unsupported formats fail closed.

## Phase 8 — cutover and removal

- Run at least two compatibility releases.
- Observe v1 client usage reaching zero.
- Pass live player-control E2E and live/headless RL promotion gates.
- Remove v1 endpoints, DTOs, wrappers, duplicate simulator sources, and historical
  launch surfaces.
- Publish checksummed release artifacts, SBOM/provenance, and migration notes.

Exit: only architecture-v2 paths are supported and a fresh checkout contains no
runtime artifacts.

## Definition of done

- Duplicate request IDs execute exactly once; ambiguous results are queryable.
- Concurrent commands and reset/step are deterministic.
- Player tokens cannot perform training/catalog actions or see hidden draw order.
- Credentials never leave the process boundary through output or logs.
- Cross-language contract fixtures and generated versions are green.
- All required CI tests pass; live-game compatibility has a recorded result.
- Live/headless facts and reward agree for shared fixtures.
- Resume reproduces the recorded experiment lineage.
- Repository tests/builds leave `git status` clean.
- No new source depends on `legacy/`, `train_pipeline.py`, RL-owned content paths, or
  developer-specific absolute paths.
