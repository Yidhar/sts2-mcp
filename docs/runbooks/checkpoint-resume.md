# Recurrent V-trace v4 checkpoint resume

The v4 learner supports two explicit operations:

1. **Exact resume** of the same recurrent V-trace lineage.
2. **Model initialization** from a complete, explicitly compatible checkpoint.

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
- every enabled transaction/complete-episode replay payload, byte/capacity
  contract, accounting counters, factual decision-surface labels and sampler
  RNG state;
- training/evaluation/policy/unroll counters; and
- Python, NumPy, Torch CPU/CUDA and collector RNG/seed state.

Every v4 checkpoint contains the base payloads below, plus
`transaction_replay.pkl` and/or `episodic_replay.pkl` when those learners are
enabled:

```text
checkpoint.manifest.json
metadata.json
network.pt
actor_network.pt
optimizer.pt
rollout_queue.pkl
stochastic_state.pkl
```

The internal format is `sts2-recurrent-vtrace-checkpoint-v4`; the queue payload
is `sts2-rollout-queue-pickle-v2`. Missing files, changed bytes, wrong versions
or any semantic mismatch stop loading. There is no empty-queue, fresh-optimizer,
partial-key or automatic tensor-remap fallback. Payloads are loaded into
independent probe model/optimizer/replay objects before any live runtime object
is mutated.

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

`--initialize-from` loads only compatible tensors from a complete learner network and
starts a fresh optimizer, queue, counters, collector RNG and lineage:

```powershell
python -m sts2_rl.train `
  --profile default `
  --initialize-from "$env:STS2_ARTIFACT_ROOT/checkpoints/recurrent-vtrace-combat/run-<uuid>/final-step-<steps>"
```

This remains strict: the source must be a complete immutable atomic checkpoint;
every listed payload hash and the closed directory set are verified before any
tensor load. The recorded source contract, reward, and dependency identities must
be structurally valid and identical between its manifest and metadata. Their
values may predate the active runtime because this operation starts a new task
lineage; exact resume still requires them to equal the current runtime exactly.

The learned-parameter configuration and strict state keys/shapes must match.
Tensor-independent encoding capacities may change because they do not alter
network parameter shapes. A changed encoding identity is rejected unless its
complete source and target identities match one reviewed model-only migration.
The current reviewed list is v8-to-v10 and v9-to-v10; version prefixes,
dimensions, or shape compatibility alone are not sufficient. These migrations
reuse only network tensors because encoded observations, candidate indexes,
behavior probabilities and replay payloads still belong to the source ABI.
Exact resume never uses this exception.

Initialization imports only `network.pt`, republishes it to the actor, and
leaves optimizer, rollout queue, transaction replay, complete-episode replay,
counters, collector/RNG state, and policy-version counters fresh. The first
child checkpoint records a `model_parameter_initialization` parent relation
plus the source contract, reward-fingerprint SHA-256, dependency locks,
manifest hash, and metadata hash. Never edit an archived manifest to make it
look current. An unreviewed architecture or feature-encoding change starts
randomly; v1 remains unsupported. The one explicit architecture migration is a
recognized v3 source that predates the long-horizon heads: all shared tensors
must still match exactly, and the complete six-prefix combat/Act/run task and
revival-cost head group remains freshly initialized. A partial group is
rejected. This operation always creates a fresh lineage and can never
masquerade as exact resume.

## Observation-v2/macro-credit v19 initialization

The v19 preheat lineage uses config v7, grounded encoding v10 and
complete-episode replay v3. Its reviewed source is the immutable v18 checkpoint
whose metadata records policy version 3,936:

```bash
bash scripts/train_preheat_wsl_rocm.sh \
  --initialize-from "<V18_POLICY_3936_CHECKPOINT>"
```

This is a model-only initialization. The v18 network parameters are checked
before import, while v19 starts from environment step zero with a fresh
optimizer, FIFO queue, transaction replay, macro-stratified complete-episode
replay, RNG stream, actor/learner counters and policy-version counter. Do not
substitute `--resume`; config, encoding and replay semantics changed, so an
exact continuation would be false provenance.

## Operational verification

1. Validate the directory and metadata before a long process.
2. Resume with identical immutable lineage and the intended larger horizon.
3. Confirm environment/learner/policy counters and queue specification.
4. Confirm actor and learner versions and restored stochastic-state version.
5. Run a bounded typed-backend smoke and publish a new atomic checkpoint.
6. Record real headless/ROCm evidence separately from unit tests.
