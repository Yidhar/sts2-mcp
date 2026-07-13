# Grounded Candidate RL Baseline

Status: **implemented, unit/integration tested, not yet long-run trained**.

This document defines the only maintained RL architecture after the July 2026
reboot. The previous MuZero/token-memory/MCTS line and its boss, card, potion,
route and action-guard heuristics were deleted because its historical runs did
not clear Act 1 reliably. Those checkpoints are not valid parents for this
baseline.

## Non-goals

The baseline does not contain:

- learned or deterministic latent dynamics;
- MCTS, direct planners, root priors or search-policy distillation;
- card-, boss-, route-, shop-, potion- or end-turn-specific rules;
- action filtering, action retargeting or policy-logit rewrites beyond the
  authoritative environment legality mask;
- objective vectors, settlement rewards or backend-provided reward fallbacks;
- ordinal candidate embeddings; or
- compatibility loading for old model/checkpoint tensors.

## Architecture

```mermaid
flowchart LR
    Backend["Typed live/headless backend"] --> Result["EnvironmentResult"]
    Result --> World["Structural world encoder"]
    Result --> Candidates["Grounded legal-candidate encoder"]
    World --> WorldModel["Candidate-independent world transformer"]
    Candidates --> CandidateModel["Candidate/local encoder"]
    WorldModel --> CandidateModel
    WorldModel --> Values["Combat value + run value"]
    CandidateModel --> Policy["Masked policy"]
    CandidateModel --> Q["Candidate Q"]
    CandidateModel --> Outcome["Reward + terminal heads"]
    Result --> Reward["Fixed normalized reward"]
    Reward --> Replay["Coverage/recent/PER replay"]
    Policy --> Replay
    Replay --> Learner["Off-policy actor-critic learner"]
    Learner --> WorldModel
    Learner --> CandidateModel
```

### Structural encoder

`packages/rl-agent/sts2_rl/encoding/grounded.py` first projects live and
headless observations into one versioned model DTO. The projection deliberately
uses the observable intersection: for example, headless draw/discard/exhaust
card identities become the same counts exposed by live, inactive simulator UI
sections are omitted, `deck`/`deck_cards` aliases are normalized, and episode-
local target IDs are joined to stable enemy facts before being removed from
model input.

The remaining object/list tree is encoded without a game-mechanics table.
Reviewed numeric keys have unique, fixed feature slots; categorical facts and
arbitrary game-provided dynamic variables use separate SHA-256 regions. Entity
identity uses two independently namespaced hashes and two embedding tables, so
a collision in one bucket is not an identity alias. Feature capacity is a
versioned 128-dimensional ABI (the default tensor is 160-dimensional).

The encoder has two separate paths:

1. World encoding excludes candidate containers, dispatch handles, simulator
   internals and retired engineered fields.
2. Candidate encoding reads only the current legal action objects and their
   local source/target objects. Candidate transport position and opaque action
   handle are not model features.

Every adapter must emit a `model_action_kind` from a closed vocabulary. Unknown
simulator enums, missing kinds, excess legal candidates, non-finite facts, and
world/local token overflow fail closed instead of creating a new category or
silently truncating training data. Event option text is an opaque visible
identity only; it is never parsed into predicted effects.

The dispatch table stays outside the tensors. Therefore changing action handles
or permuting equivalent candidates cannot change the candidate-independent
state representation. The complete allowlist, exclusion firewall, slot layout,
domain vocabulary and action vocabulary are SHA-256 fingerprinted. Decisions,
replay payloads and checkpoints reject a different encoding fingerprint.

### Model

`packages/rl-agent/sts2_rl/models/grounded_candidate.py` is a small transformer
with **3,642,824 parameters** at the default configuration.

- `encode_world(world, domain_ids)` cannot access candidates by type/API.
- Candidates cross-attend the already-computed world latents.
- The candidate-set transformer has no position embedding and is permutation
  equivariant.
- Illegal candidates have zero probability and zero Q/reward/outcome outputs.
- State values are split into combat-horizon and run-horizon heads.
- Candidate Q, immediate reward and terminal-class heads make outcome errors
  visible without inventing a latent world model.
- Masked policy probabilities are calculated and returned in float32 even when
  model logits use fp16/bfloat16, avoiding low-probability underflow.

### Reward

`packages/rl-agent/sts2_baseline/reward.py` owns one immutable reward version.

- combat win/loss and run win/loss are separate task terminals;
- HP and enemy progress are ratios rather than absolute values;
- run progress is normalized to `[0, 1]`;
- dense potential shaping is capped at `0.25` magnitude;
- task-terminal potential is forced to zero; and
- backend reward, objective vectors and settlement bonuses have no API entry.

The reward discount and return discount must match. Configuration rejects a
different value rather than silently changing reward semantics. The fingerprint
includes both calculator coefficients and the fact-to-potential/terminal projection
(floor cap, result map and truncation rules); replay decisions and checkpoints reject
a different fingerprint.

### Replay and learner

Replay samples are stratified by structural domain/encounter metadata before
sampling inside each stratum. The default mixture is:

- 50% coverage-uniform;
- 25% recent-uniform; and
- 25% proportional priority replay.

Sampling returns stable replay IDs, exact mixture probabilities and importance
weights. The learner refreshes priorities from current TD error after every
update.

The collector reuses the structural encoding it already computed for action
selection. Replay stores that decision as a versioned sparse snapshot: exact
non-zero float features, categorical IDs, candidate-local offsets, action mask,
domain ID, the complete encoding configuration, and the encoding fingerprint.
Dispatch handles and raw policy output never enter the snapshot. A SHA-256 of
the policy-free compact source facts is retained for audit, while the raw
observation/action JSON is not duplicated in replay.

Learner batches collate these sparse CPU snapshots directly into the fixed
model tensor contract. They do not walk or re-encode raw observations when a
sample is drawn. Checkpoint replay format v2 validates every snapshot after
deserialization, including sparse-array canonical form, vocabulary bounds,
capacities, source fingerprint, reward fingerprint and encoding identity.

Online trajectories receive auditable discounted Monte Carlo targets. Forced
single-action protocol decisions are stored but make no policy-gradient
contribution. The learner jointly trains masked policy, selected-candidate Q,
the active horizon value, immediate reward and terminal classification.

### Execution pipeline and overlap experiment

The maintained profiles use the `synchronous` execution mode. An opt-in
`overlap` mode exists for bounded systems work, but it is not a free-running
actor queue:

- the learner owns the authoritative model, optimizer, replay and counters;
- the collector owns a separate read-only actor model, optionally on a separate
  device;
- exactly one episode may be in flight;
- replay is extended, sampled and reprioritized only by the main thread;
- actor parameters are published only between episodes;
- evaluation, periodic checkpoint and final-checkpoint boundaries do not launch
  another episode; and
- Ctrl-C joins and ingests an in-flight episode before publishing an interrupt
  checkpoint.

Overlap currently requires `updates_per_cycle=1`. On Ctrl-C, the runtime also
finishes only the update cycles that belonged to the preceding episode before
saving; update credit from the drained episode remains deferred. This preserves
the same one-episode policy-publication phase on exact resume. Synchronous
collection instead restores its pre-episode collector RNG/seed snapshot if it is
interrupted before producing a complete episode.

This gives a race-free one-episode pipeline: learner work for episode N can run
while the collector produces episode N+1. The tradeoff is explicit off-policy
lag. Metrics record `collector_policy_version`,
`collector_policy_lag_updates`, parameter-publication time, collector wait time,
and `collector_pre_wait` (collector work completed before the main thread began
or completed its wait). Behavior probabilities are still recorded per decision,
but the current experiment does not refresh the actor within a long episode.

Opt in only for controlled profiling:

```powershell
python -m sts2_rl.train `
  --profile default `
  --execution-mode overlap `
  --collector-device cpu `
  --sim-exe <PINNED_HEADLESS_SIM_RELEASE_EXE>
```

## Training curriculum

There are two configurations, not an A/B comparison or an attempt to preserve
the failed route:

1. `combat` is an optional short-horizon bootstrap using the same model and
   encoder with the combat value/reward horizon.
2. `default` is the required full-run mainline and optimizes the run horizon.

The cleanest route is to start `default` from random initialization. If early
full-run collection is too sparse, `combat` may train the same tensor/model ABI
first, followed by a **new lineage** that imports only the complete RL 0.3 model
state with `--initialize-from`. The source must also pass the current atomic hashes,
contract, reward, dependency-lock and encoding gates. Optimizer, replay, counters,
reward horizon and RNG state are never carried across that curriculum boundary. This is a
same-baseline curriculum transition, not loading a MuZero/legacy checkpoint.

No metric gate injects rules into the policy. Curriculum changes only the task/reward
horizon, environment data distribution, exploration schedule and execution/evaluation
budget. The evaluation surface reports:

- Act 1 clear rate (Act 2 reached);
- run win rate;
- combat win rate for combat evaluation;
- maximum act and floor; and
- mean undiscounted reward total.

Training resets use an even signed-32-bit seed namespace. Evaluation uses a fixed odd
prefix derived from the configured base seed, so held-out seeds never enter training.
Every evaluation log records the seed and complete per-episode metrics as well as the
aggregate.

The first acceptance target is a statistically meaningful non-zero Act 1 clear
rate on held-out seeds. **No such result is claimed yet.** Dry-run, unit and fake
backend integration tests validate software behavior only.

## Commands

From `packages/rl-agent`:

```powershell
python -m sts2_rl.train --dry-run
python -m sts2_rl.train --profile combat
python -m sts2_rl.train --profile default
```

Optional combat-to-run model initialization:

```powershell
python -m sts2_rl.train `
  --profile default `
  --initialize-from "$env:STS2_ARTIFACT_ROOT/checkpoints/grounded-combat-bootstrap/run-<uuid>/final-step-<steps>"
```

Use strict dotted overrides for experiments:

```powershell
python -m sts2_rl.train --set runtime.total_environment_steps=2000000
```

ROCm/WSL live-bridge launch:

```bash
bash scripts/train_grounded_wsl_rocm.sh --profile default
```

All mutable logs, replay and checkpoints live below `STS2_ARTIFACT_ROOT` (or
`~/.sts2-artifacts`). Every invocation creates a fresh unique `run-<uuid>` child,
so a resume never overwrites its parent. Each published checkpoint child is immutable.
Checkpoints atomically publish model,
optimizer, replay, training counters, update credit, Python/NumPy/Torch RNG and
collector RNG/seed state. Exact resume rejects contract, reward, dependency
lock, resolved learner/collector device, model tensor, encoding fingerprint,
optimizer, replay or
immutable-lineage drift. Static `game-data` is optional audit provenance when a valid
catalog manifest is present; it is neither a runtime dependency nor an exact-resume
identity because this baseline does not read it.

The current execution/config cutover uses
`sts2-grounded-baseline-config-v2`; the sparse replay cutover uses
`grounded-structural-encoding-v2` and
`sts2-grounded-baseline-checkpoint-v2`. Checkpoints created before either cutover
are intentionally rejected rather than lazily converting raw replay during
training. No pre-cutover checkpoint is an official baseline parent.

### Encoding-path performance evidence

A controlled 536-step ROCm/Release-HeadlessSim A/B run used identical seeds,
episodes, learner-update schedule and learner metrics. Reusing sparse snapshots
reduced learner encoding from 1,574.8 ms to 23.1 ms per update, total learner
time from 1,724.4 ms to 151.5 ms, and end-to-end runtime from 249.4 s to 40.5 s.
Throughput increased from 2.15 to 13.25 environment steps/s. The 536-sample
replay pickle also fell from 3.85 MiB to 2.20 MiB. These are engineering
throughput measurements, not Act 1 or policy-quality claims.

### Collector/learner overlap evidence

A second controlled ROCm/Release-HeadlessSim investigation used the same 512-step
target, seed, replay warm-up, update ratio and disabled evaluation. Episode-level
policy lag changes trajectories, so these are throughput comparisons rather than
identical-policy or learning-quality comparisons:

| pipeline | actor device | realized steps | elapsed | throughput |
|---|---:|---:|---:|---:|
| synchronous | ROCm | 536 | 38.44 s | 13.94 steps/s |
| one-episode overlap | ROCm | 556 | 38.56 s | 14.42 steps/s |
| one-episode overlap | CPU | 556 | 36.12 s | 15.39 steps/s |

Same-device overlap caused severe contention: aggregate collector time rose from
18.1 s to 30.6 s and mean learner time rose from 146.6 ms to 208.8 ms/update.
Moving the actor to CPU reduced wall time, but the resulting 10.4% throughput
gain over the new synchronous control is far below the theoretical two-stage
pipeline ceiling. Observed episode-level policy lag ranged from 3 to 28 learner
updates after warm-up; full-run episodes can be longer. Therefore overlap remains
opt-in and the formal profiles remain synchronous. A future attempt should test
process/core isolation and decision-boundary actor refresh before considering
multiple collectors. These measurements are not Act 1 performance claims.
The ROCm CPU-actor processes also emitted a two-signal `SharedSignalPool`
shutdown warning after otherwise successful checkpoint publication; this is an
additional reason not to treat the thread experiment as production-ready.

## Validation

```powershell
python -m ruff check sts2_rl sts2_baseline
python -m mypy sts2_rl sts2_baseline
python -m pytest tests -q
python -m sts2_rl.train --dry-run
```

Long-run training and held-out Act 1 evaluation remain operational work, not a
completed result.
