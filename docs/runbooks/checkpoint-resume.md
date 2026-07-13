# Recurrent V-trace v2 checkpoint resume

The v2 learner supports two explicit operations:

1. **Exact resume** of the same recurrent V-trace lineage.
2. **Model initialization** from a complete, same-ABI v2 checkpoint.

The failed v1 replay baseline, MuZero, PPO, token-memory, planner, partial state
dict, manifest-less weights and edited checkpoints are not inputs. In particular,
a checkpoint containing `replay_buffer.pkl` is rejected before deserialization.

## Exact resume invariants

Before `torch.load` or queue deserialization, preflight verifies:

- the atomic completion manifest and SHA-256 for every listed payload;
- no unlisted payload;
- current API/schema/action-order/observation/reward identities;
- dependency-lock identities and resolved learner/actor devices;
- recurrent model and grounded encoding configuration/fingerprint;
- immutable task, rollout, optimizer, seed and environment lineage;
- learner and actor state tensor specifications;
- optimizer state/group specification;
- pending `SequenceUnroll` types, versions, lengths, sparse snapshots, recurrent
  state width, behavior versions and queue capacity;
- training/evaluation/policy/unroll counters; and
- Python, NumPy, Torch CPU/CUDA and collector RNG/seed state.

Every v2 checkpoint contains:

```text
checkpoint.manifest.json
metadata.json
network.pt
actor_network.pt
optimizer.pt
rollout_queue.pkl
stochastic_state.pkl
```

The internal format is `sts2-recurrent-vtrace-checkpoint-v3`; the queue payload
is `sts2-rollout-queue-pickle-v2`. Missing files, changed bytes, wrong versions
or any semantic mismatch stop loading. There is no empty-queue, fresh-optimizer,
partial-key or automatic tensor-remap fallback.

Learner and actor networks are both stored because the bounded asynchronous
pipeline permits a versioned actor snapshot to lag the learner. Checkpoints are
published only while the actor is quiescent between episodes. Pending unrolls
retain their original behavior policy versions, so V-trace correction resumes
with the same data-plane state.

## Paths and invocation

Each invocation owns a unique immutable run directory below the artifact root:

```text
$STS2_ARTIFACT_ROOT/checkpoints/recurrent-vtrace/
  run-<uuid>/
    periodic-step-000025000/
    final-step-001000000/
```

```powershell
python -m sts2_rl.train `
  --profile default `
  --resume "$env:STS2_ARTIFACT_ROOT/checkpoints/recurrent-vtrace/run-<uuid>/periodic-step-<steps>" `
  --steps 2000000
```

Exact resume may change the execution horizon and observation-only output
schedules: total environment steps, log/checkpoint roots, checkpoint interval,
evaluation gates and evaluation episode count. Model, reward objective,
environment scenario, optimization, rollout queue/unroll semantics, seed and
devices remain immutable. The new total-step target may equal or exceed the
restored count; an equal target performs validation and final publication only.

The referenced parent must remain immutable during validation. Never edit a
manifest or metadata file to bypass incompatibility. Preflight happens before
backend launch; model/actor/optimizer states are loaded into disposable objects
before any live resource is changed.

## Model initialization

`--initialize-from` loads only a complete same-model v2 learner network and
starts a fresh optimizer, queue, counters, collector RNG and lineage:

```powershell
python -m sts2_rl.train `
  --profile default `
  --initialize-from "$env:STS2_ARTIFACT_ROOT/checkpoints/recurrent-vtrace-combat/run-<uuid>/final-step-<steps>"
```

This remains strict: atomic manifest, hashes, v3 format, current identities,
identical model/encoding configuration and strict state keys/shapes are required.
A deliberate architecture or encoder change starts from random initialization.
The initial v2 mainline should therefore start fresh rather than importing v1.

## Operational verification

1. Validate the directory and metadata before a long process.
2. Resume with identical immutable lineage and the intended larger horizon.
3. Confirm environment/learner/policy counters and queue specification.
4. Confirm actor and learner versions and restored stochastic-state version.
5. Run a bounded typed-backend smoke and publish a new atomic checkpoint.
6. Record real headless/ROCm evidence separately from unit tests.
