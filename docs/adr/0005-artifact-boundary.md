# ADR 0005: Runtime artifacts live outside the source checkout

- Status: accepted
- Date: 2026-07-11

Checkpoints, optimizer state, replay, datasets, virtual environments, logs,
release binaries, process ids, and temporary decompilations are not source.
They live under `STS2_ARTIFACT_ROOT` or CI release storage. Git tracks only
manifests, checksums, schemas, migrations, and intentionally small fixtures.
