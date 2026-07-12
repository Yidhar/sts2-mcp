# Checkpoint resume and weights-only migration

MuZero has two deliberately separate checkpoint operations. Do not use the terms
"resume" and "warm-start" interchangeably.

## Exact resume

`--resume-from` means continuation of the same training lineage. Before any
`torch.load` or replay unpickle, the loader requires and verifies:

- an `sts2-atomic-checkpoint-v1` completion manifest;
- `hash_files=true`, a valid SHA-256 for every payload file, and no unlisted files;
- current API, contract schema, action schema, legal-action ordering, observation
  schema, and reward schema versions;
- the complete canonical reward-spec fingerprint and weights;
- the canonical `game-data/manifest.json` SHA-256 and source identity;
- matching MuZero model constructor/state, optimizer class/layout, replay, and
  observation-shape schemas;
- complete network, optimizer, replay, token-target (when enabled), and AMP scaler
  (when enabled) state.

Missing manifests, missing hashes, changed bytes, missing payloads, or any identity
mismatch stop the process. Exact resume never falls back to partial model keys, a
fresh optimizer, or an empty replay buffer.

```powershell
python -m muzero.train `
  --resume-from "$env:STS2_ARTIFACT_ROOT/checkpoints/<run>/<checkpoint>" `
  --total-timesteps 500000
```

The deprecated `--no-resume-load-buffer`, `--resume-without-buffer`,
`--no-resume-load-optimizer`, and `--resume-without-optimizer` switches remain
parseable only so old automation fails with an actionable error. They cannot turn an
exact resume into a silent cold continuation.

## Explicit weights-only warm-start

A partial or legacy checkpoint starts a **new** training lineage. The only supported
migration is `sts2-weights-only-v1`: it copies only tensors whose key, shape, and
dtype match. It never restores optimizer moments, replay, GradScaler, global step, or
episode count.

```powershell
python -m muzero.train `
  --resume-from "$env:STS2_ARTIFACT_ROOT/checkpoints/<old-run>/<checkpoint>" `
  --warm-start `
  --checkpoint-migration-id sts2-weights-only-v1 `
  --total-timesteps 500000
```

Atomic warm-start sources still require every payload hash to verify. A checkpoint
created before atomic manifests existed additionally requires the auditable risk flag:

```powershell
  --allow-legacy-checkpoint
```

That flag applies only to the weights-only migration. It can never bypass exact-resume
identity checks.

## Operational checks

1. Keep the source checkpoint immutable during verification and loading.
2. Record its checkpoint ID (or the legacy path), migration ID, and destination run ID.
3. Run a bounded smoke after an exact resume and confirm restored step, replay size,
   optimizer state, and output directory.
4. Run a warm-start as a new experiment; never copy the old step or episode counters.
5. Do not edit `checkpoint.manifest.json` to "fix" a mismatch. Regenerate an atomic
   checkpoint through a supported migrator instead.
