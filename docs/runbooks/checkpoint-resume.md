# Grounded-baseline checkpoint resume

The restarted learner supports exactly two explicit operations:

1. **Exact resume** of the same grounded-baseline lineage.
2. **Model initialization** of a new combat/full-run lineage from a complete,
   valid RL 0.3 grounded-baseline checkpoint with current contract/reward/dependency
   identities and an identical model/encoding contract.

Old MuZero, token-memory, PPO, planner, offline-supervised, partial state dict,
or manifest-less weights/replay are not inputs to either operation.

## Exact resume invariants

Before `torch.load` or replay deserialization, the loader verifies:

- an `sts2-atomic-checkpoint-v1` completion manifest;
- `hash_files=true`, SHA-256 for every payload, and no unlisted payload;
- current API/schema/action-ordering/observation/reward identities;
- the immutable grounded reward-spec fingerprint;
- dependency-lock identities and the resolved learner/collector devices;
- the grounded encoding ABI version and fingerprint;
- the immutable lineage portion of the typed training configuration;
- strict model state keys/shapes;
- optimizer state/specification and replay type/specification;
- checkpoint-v2 sparse encoded replay payloads, including their canonical CSR
  arrays, capacities, vocabulary bounds and source/encoding fingerprints;
- environment steps, learner updates, episode/evaluation counters and pending
  update credit; and
- Python, NumPy, Torch CPU/CUDA and collector RNG/seed state.

When a valid static catalog manifest is available it is recorded as optional audit
provenance, but it is deliberately not an exact-resume gate or runtime dependency: the
grounded model does not read `game-data`. A changed encoder allowlist/slot/vocabulary is
a gate through its own fingerprint.

Every exact checkpoint contains:

```text
checkpoint.manifest.json
metadata.json
network.pt
optimizer.pt
replay_buffer.pkl
stochastic_state.pkl
```

Missing files, changed bytes or any identity/config mismatch stop loading. There
is no partial key load, empty replay fallback, fresh optimizer fallback, legacy
risk flag or automatic tensor remapping.

The current internal trainer format is
`sts2-grounded-baseline-checkpoint-v2`, with
`sts2-grounded-replay-pickle-v2` and
`grounded-structural-encoding-v2`. Pre-snapshot grounded checkpoints are
rejected before replay deserialization. Do not use a profiling checkpoint from
the earlier raw-observation replay path as an exact-resume or model-
initialization parent.

Each invocation writes to a fresh unique run directory; every atomically published
checkpoint child is immutable:

```text
$STS2_ARTIFACT_ROOT/checkpoints/grounded-baseline/
  run-<uuid>/
    step-000050000/
    final-step-001000000/
```

```powershell
python -m sts2_rl.train `
  --profile default `
  --resume "$env:STS2_ARTIFACT_ROOT/checkpoints/grounded-baseline/run-<uuid>/step-<steps>" `
  --steps 2000000
```

Exact resume may change only the total execution budget and output/evaluation
schedule: total environment steps, log/checkpoint roots, checkpoint/evaluation
intervals and evaluation episode count. Model, reward, environment scenario,
optimization, replay, seed, device, collection cadence and update cadence remain
immutable. Execution mode and overlap collector device are also lineage values:
an overlap run cannot be resumed as synchronous (or vice versa). The new total
step target must exceed the restored step count.

Overlap checkpoints are emitted only at quiescent episode boundaries. The actor
replica is not a second checkpoint authority: only the learner network is saved,
and a validated load republishes that network to the collector replica. An
interrupt first joins and ingests any in-flight episode before reading collector
RNG/seed state and publishing the checkpoint. It settles only the remaining
single-update cycles from the preceding episode, leaving the drained episode's
credit deferred so resume preserves the one-episode publication phase. For this
reason overlap mode rejects `updates_per_cycle` values other than one.

The referenced checkpoint must remain immutable during verification. Never edit
its manifest or metadata to bypass an incompatibility. Preflight hashes and
validates metadata before creating run directories or launching a backend;
model, optimizer and stochastic payloads are validated in disposable objects
before live training state is committed.

## Combat-to-full-run initialization

The optional curriculum boundary loads **only** a strictly matching grounded
model from a complete atomic checkpoint. It starts a fresh optimizer, replay,
counter set, collector RNG and output lineage:

```powershell
python -m sts2_rl.train `
  --profile default `
  --initialize-from "$env:STS2_ARTIFACT_ROOT/checkpoints/grounded-combat-bootstrap/run-<uuid>/final-step-<steps>"
```

This mode still requires the current atomic manifest, all hashes, the RL 0.3
checkpoint format, current contract/reward/dependency-lock identities, identical model
configuration, strict state keys/shapes and the same encoding fingerprint. A deliberate architecture or encoder change
starts from random initialization; it is never partially remapped.

Every checkpoint records a stable run origin (`fresh`, `exact_resume`, or
`model_initialization`). Its parent relation separately records whether the parent was
loaded at process start or is an in-process immutable predecessor.

## Operational verification

1. Validate the checkpoint directory before starting a long process.
2. Resume with the same immutable lineage values and an increased execution
   budget.
3. Confirm restored environment steps, update credit, update count, replay size
   and stochastic-state version.
4. Run a bounded backend smoke and publish a new atomic checkpoint.
5. Record live-game and ROCm validation separately from unit-test evidence.
