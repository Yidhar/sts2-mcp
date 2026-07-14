# Documentation index

This page defines which documents are authoritative during the architecture-v2
cutover. A document being present in the repository does not by itself make it a
current runtime contract.

## Normative sources

Use these sources in this order:

1. [`../contracts/manifest.json`](../contracts/manifest.json), schemas, OpenAPI,
   and fixtures for wire behavior.
2. [`../release-manifest.json`](../release-manifest.json) for component versions.
3. [`architecture.md`](./architecture.md) and
   [`architecture/target-architecture.md`](./architecture/target-architecture.md),
   plus [`rl-grounded-baseline.md`](./rl-grounded-baseline.md),
   [`card-facts-abi.md`](./card-facts-abi.md), and
   [`runtime-mechanics-abi.md`](./runtime-mechanics-abi.md)
   for boundaries and ownership.
4. Accepted decisions under [`adr/`](./adr).
5. [`migration/README.md`](./migration/README.md) and
   [`migration/v2-cutover.md`](./migration/v2-cutover.md) for compatibility and
   removal gates.
6. Component READMEs:
   - [`../mods/sts2-bridge/README.md`](../mods/sts2-bridge/README.md)
   - [`../packages/mcp-server/README.md`](../packages/mcp-server/README.md)
   - [`rl-grounded-baseline.md`](./rl-grounded-baseline.md)
   - [`../game-data/README.md`](../game-data/README.md)
7. Operational runbooks under [`runbooks/`](./runbooks).

If prose conflicts with a validated schema or generated version constant, the
contract asset wins and the prose must be corrected.

## Architecture decisions

- [ADR 0001: modular monorepo](./adr/0001-modular-monorepo.md)
- [ADR 0002: idempotent command actor](./adr/0002-command-actor.md)
- [ADR 0003: capability split](./adr/0003-capability-split.md)
- [ADR 0004: RL owns reward](./adr/0004-reward-ownership.md)
- [ADR 0005: external artifact boundary](./adr/0005-artifact-boundary.md)

New cross-component exceptions require a new ADR. Do not silently bypass these
decisions from a launcher, compatibility wrapper, or build script.

## Runbooks

- [Development and verification](./runbooks/development.md)
- [Artifact inventory and externalization](./runbooks/artifacts.md)
- [WSL ROCm training](./runbooks/wsl-rocm-training.md)
- [HeadlessSim build identity](./headless-simulator-identity.md)
- [Runtime card-facts ABI](./card-facts-abi.md)
- [Runtime mechanics ABI](./runtime-mechanics-abi.md)
- [Release validation and evidence](./runbooks/release.md)
- [V2 migration guide](./migration/README.md)

## Historical and generated material

- [`archive/`](./archive) contains preserved v1 architecture and implementation
  history. It is non-normative.
- Files whose names include experiment dates, recovery plans, mechanism audits,
  win-rate goals, or one-off training campaigns are research records. They do not
  override current entry points, contracts, reward ownership, or safety rules.
- [`generated/`](./generated) is produced by tooling. Do not hand-edit generated
  tables or JSON.
- Historical RL worktree patches, artifact inventories, checkpoint paths, and replay
  hashes are intentionally not retained in the rebooted repository. Git history is
  the only archive; none of it is a supported training input.

## Documentation maintenance rules

- Use repository-relative links and portable placeholders such as
  `<REPOSITORY_ROOT>`, `<PATH_TO_STS2>`, and `<ARTIFACT_ROOT>`.
- Never put a bearer token, session descriptor, developer home path, PID, or real
  checkpoint path in documentation.
- Update English and Chinese root READMEs together when user-facing behavior changes.
- Update `CHANGELOG.md` and `release-manifest.json` for component releases.
- State whether a command is read-only, writes generated files, deploys to the game,
  or moves artifacts.
- Do not claim live-game compatibility from dependency-free tests alone.
