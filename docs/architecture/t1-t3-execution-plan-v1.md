# T1–T3 execution plan v1

Status: **design and isolated T1 benchmark only**.  This document does not
change the v32 recipe, checkpoint ABI, learner, collector, reward contract, or
active lineage.  The 100,105-step v32 run stopped at its evaluation guard and
therefore provides a safe window for these isolated tests; its 100k checkpoint
is failure evidence rather than an automatic parent for a successor.

## 1. Findings that constrain every experiment

1. v32 is a single-actor system, not a dormant multi-actor configuration.
   `RolloutConfig` rejects `collector_workers != 1`, and
   `ActorLearnerPipeline` owns one typed backend, one collector model, and one
   actor thread.  Multi-actor support therefore requires an architecture/ABI
   change rather than a TOML edit.
2. The collector already stores the exact recurrent state at every unroll
   boundary in `SequenceUnroll.initial_recurrent_state`.  The learner restores
   it before replaying the sequence.  "Add stored state" is already complete;
   replacing it with an approximate implementation would be a regression.
3. The learner executes the time dimension in Python and only batches active
   rows at each time index.  Changing `unroll_length = 16` to `64` does not make
   the GPU batch four times wider.  With `batch_unrolls = 4`, it changes an
   update from 64 to 256 environment steps and quarters optimizer, sidecar
   replay, and model-publication frequency per environment step.  It is a
   learning-recipe change, not a free throughput switch.
4. The measured production bottleneck is learner backward, while the actor
   queue has little consumer wait.  Extra actors would initially increase
   queueing and policy lag rather than GPU utilization.
5. The model has data-dependent Python control flow in `_safe_valid_mask` and
   active world/candidate/local shapes vary by state.  This is exactly the
   compiler workload in which graph breaks and recompilation must be measured,
   not assumed away.
6. v32 intentionally forces the stable math SDPA backend.  Compiler/precision
   tests must retain math SDPA; an SDPA backend change is a separate variable.

## 2. T1 — execution-engine experiment

### 2.1 Isolated benchmark added

`packages/rl-agent/scripts/benchmark_t1_model_execution.py` loads the real
v32 model configuration and exercises production model code on synthetic
active-shape batches.  It creates no environment, run directory, replay,
checkpoint, or learner.  Every cell must run in a fresh process.  It reports:

- eager/compiled FP32 or BF16 forward/training latency;
- compile plus warm-up cost and steady-state latency separately;
- peak device memory;
- finite-output checks and valid-candidate greedy-action agreement against an
  eager FP32 oracle for forward-only cells;
- explicit FP32 boundaries for policy logits, value, recurrent state, and
  liveness returns;
- stable math SDPA by default.

The first RX 7900 XTX / PyTorch 2.9.1+ROCm 7.2.1 cells (batch 4, world 96,
candidate 48, local 12) are:

| Cell | Median | Mean | Peak allocation | Result |
|---|---:|---:|---:|---|
| eager FP32 forward | 10.65 ms | 10.63 ms | 231.1 MB | reference |
| eager BF16 forward | 15.71 ms | 16.15 ms | 205.1 MB | slower; greedy 4/4 agrees |
| compiled FP32 forward, graph breaks allowed | 9.95 ms | 9.68 ms | 222.2 MB | only ~9% mean gain after 131 s warm-up |
| compiled BF16 forward, graph breaks allowed | 19.91 ms | 19.37 ms | 195.2 MB | materially slower; greedy 4/4 agrees |
| compiled FP32 `fullgraph=True` | — | — | — | fails at `_safe_valid_mask: if missing.any()` |
| eager FP32 synthetic train step | 24.55 ms | 27.49 ms | 262.2 MB | reference |
| eager BF16 synthetic train step | 28.92 ms | 32.36 ms | 249.2 MB | slower |

These are microbenchmarks, not a learner-speed claim.  They are already enough
to reject **"enable BF16 by default"** and **"fullgraph compile works now"**.
ROCm reports BF16 support, but the measured shape is slower and AMD's current
Radeon compatibility table does not list BF16 as a validated training datatype
for this card.  Graph-break-tolerant compile remains a candidate only after a
real learner-step benchmark and a multi-shape recompilation audit.

### 2.2 Required benchmark matrix

Keep checkpoint weights, sampled batches, SDPA backend, and RNG fixed.  Run:

1. **Execution:** eager FP32, compile FP32, eager BF16, compile BF16.
2. **Shapes:** p50, p90, and maximum observed active world/candidate/local
   shapes, plus the known 111-candidate regression state and 256-candidate
   capacity boundary.
3. **Path:** forward-only, base V-trace backward, liveness replay backward,
   episodic replay backward, and the complete learner update.
4. **Unroll semantics:** `(T=16,B=4)`, `(T=32,B=2)`, `(T=64,B=1)` first, so all
   cells retain 64 online environment steps per update.  Only after equivalence
   should `(T=64,B=4)` be tested as a new recipe.
5. **Duration:** at least 1,000 complete learner updates after compilation and
   allocator warm-up, with compiler cache size/recompile count and every timing
   component persisted.

Example cell:

```bash
python scripts/benchmark_t1_model_execution.py \
  --profile preheat \
  --config config/experiments/full_run_revival_v32_budget64_mature_model_init.toml \
  --device cuda --sdpa math --batch-size 4 \
  --world-tokens 96 --candidates 48 --local-tokens 12 \
  --precision fp32 --compile --forward-only
```

### 2.3 Promotion gates

No production switch unless all gates pass:

- zero non-finite output, loss, or gradient;
- valid-action greedy agreement >= 99.9% on recorded states and no statistically
  significant held-out policy/value regression;
- compile cache reaches a bounded steady state on real active shapes; no
  per-shape minutes-long recompilation;
- end-to-end learner updates/second improve >= 20% at equal 64 steps/update;
- peak device memory leaves the existing evaluation/checkpoint margin;
- exact resume and eager rollback reproduce the same checkpoint ABI;
- failure-credit and episodic replay update counts per environment step remain
  unchanged in the first execution-only lineage.

Rollback is a runtime execution-backend flag to eager FP32, never a checkpoint
conversion.  Optimizer/model tensors remain FP32 in the first compiler trial.

### 2.4 Multi-actor design (only after learner acceleration)

The formal design is one backend session and one actor-local recurrent model per
actor, feeding a single typed many-producer FIFO.  It needs:

- disjoint training seed namespaces and globally unique episode IDs;
- actor ID, behavior policy version, actor-local unroll sequence, and exact
  initial recurrent state in every publication;
- adoption of a new policy only between complete recurrent unrolls;
- a coordinator barrier at episode/evaluation/checkpoint quiescence;
- fair admission (per-actor queue quotas or round-robin drain) so one fast actor
  cannot monopolize training;
- persisted per-actor supervisor and RNG state for exact resume;
- policy-lag histograms and rejection by both update lag and wall-clock age.

Start with two actors only after the learner is no longer continuously busy.
Promote 2→4 actors only if steps/second rises, p95 lag stays inside the current
contract, and paired held-out results do not regress.  Sample Factory's own
architecture documentation warns that recurrent policies are especially
sensitive to policy lag because hidden states also change with policy weights.

## 3. T2 — versioned learning mechanisms

T2 must not be merged as one bundle.  First define and shadow-validate the
semantic ABI; then introduce one head or behavior change per explicit
model-initialization lineage.

### 3.1 Semantic ABI first

Define immutable, versioned facts:

```text
AchievementFactV1
  fact_type, act, floor, encounter/event definition, occurrence_index

MacroBoundaryV1
  decision_surface, act, floor, room/event identity, entry_step, exit_step

MacroTransitionV1
  start_boundary, end_boundary, duration, extrinsic_return, terminal,
  start/end observation refs, behavior policy version
```

Shadow extraction must be deterministic under journal replay, collision-audited,
version isolated, and must never read hidden future RNG.  Promotion requires
coverage tables by surface and tests for terminal, revival, repeated event,
select/deselect, and act-transition boundaries.

### 3.2 Next-achievement contrastive auxiliary head

This is the lowest-risk T2 mechanism.  Following Achievement Distillation,
label each state with the **next first future engine fact**, censor states with
no label, and contrast a state projection against learned achievement
embeddings with in-batch negatives.  In v1 it is auxiliary only:

```text
L = L_current + lambda_achievement * InfoNCE(state, next_fact)
```

Do not expose the future label to online action selection.  Do not turn it into
reward in the first lineage.  This adds model/replay ABI fields and therefore
requires explicit model-init, not exact resume.

### 3.3 Multi-gamma auxiliary critics

Retain the existing undiscounted primary objective for the actor.  Add critic-
only heads with half-life-derived discounts, for example
`gamma(H) = 2**(-1/H)` for H in `{32, 256, 2048}`, plus the current run head.
Normalize each target and average, rather than sum, auxiliary losses.  Log
calibration and gradient cosine similarity by horizon.  Do not mix advantages
from these heads into the actor until they predict held-out returns better than
the primary critic on their intended horizons.

### 3.4 SMDP macro bootstrapping

For consecutive macro boundaries `i -> j` separated by `tau` primitive steps:

```text
target_i = sum_{t=i}^{j-1} gamma^(t-i) r_t + gamma^tau V_macro(s_j)
```

With the current primary `gamma = 1`, duration has no discount; choosing any
other macro gamma is an objective-version change.  Round one is a macro critic
only.  It must not duplicate the same terminal reward in both the primitive
and macro primary actor return.  Only after calibration may a later lineage use
the macro advantage on macro actions.

### 3.5 Episodic first-visit facts

A naive positive reward for every new fact changes the optimal policy and can
teach event/relic/card novelty instead of winning.  V1 therefore uses first-
visit facts only as an auxiliary target, replay-priority signal, and diagnostic.
If later injected into behavior reward, it needs a new reward ABI, a small cap,
decay to zero, immutable fact keys, and promotion solely on the original
extrinsic held-out metrics.  It is not automatically potential-based shaping.

### 3.6 First-k card-pick exploration

Treat this as an unverified project hypothesis, not an established "ByteRL"
result.  Count actual card-reward commit transactions per run.  Apply an epsilon
floor only to non-forced initial card choices among eligible legal candidates,
not select/deselect/confirm steps.  Record the exact mixed behavior probability
for V-trace.  The counter and schedule become behavior/checkpoint ABI.  Test it
alone with paired seeds; remove it if it raises deck entropy without improving
act/run outcomes.

### 3.7 T2 dependency order

1. semantic ABI + offline shadow extractor + boundary tests;
2. next-achievement and multi-gamma auxiliary heads, model-init lineage;
3. SMDP macro critic and target/calibration audit;
4. first-visit fact as auxiliary/replay priority only;
5. optional macro actor after critic validation;
6. isolated first-k exploration A/B.

## 4. T3 — experiments isolated from the baseline

### 4.1 Motif-style LLM shaping

Produce concise captions only for non-combat macro outcomes.  Ask an LLM for
offline pairwise preferences, train a frozen local reward model, and version
the caption schema, preference dataset, reward model, and reward ABI.  There is
no online LLM call and no combat shaping.  Promotion is based only on original
held-out success/revival/HP/deadlock metrics.  Audit adversarial captions and
compositional reward hacking; Motif itself calls out "misalignment by
composition" as a limitation.

### 4.2 Frozen-combat deck evaluator

Freeze one complete combat executor snapshot and route every combat action to
it.  Train/evaluate only macro choices under that fixed executor.  Targets are
run/act success, revival count, and HP loss under paired seeds.  This isolates
whether deck construction and pathing improve without moving combat skill.
If the environment can clone a decision state, evaluate each legal reward card
with common-random-number rollouts.  Without state cloning, an observational
deck evaluator is confounded and must not claim causal card values.  The use of
fixed battle agents to evaluate drafting policies follows Vieira et al.'s CCG
drafting setup.

### 4.3 Kickstarting

Use a frozen teacher on the same legal candidate set:

```text
L_student = L_RL + lambda(t) * KL(pi_teacher || pi_student)
```

Apply distillation only on teacher-supported macro surfaces, exclude forced
steps and known deadlock behavior, anneal `lambda` to zero, and optionally gate
it by a calibrated teacher advantage.  The student must remain free to surpass
the teacher.  Track teacher action agreement, held-out extrinsic return, and
forgetting separately.

## 5. Experiment governance

- Every T2/T3 item is a named isolated lineage with one changed mechanism.
- Use a fixed cross-lineage paired seed namespace and at least 16 seeds for an
  intermediate check, 64 for a promotion decision.
- The default parent is selected by paired evaluation among the last sound
  checkpoints (for example 75k/80,086/90,152), not automatically the failed
  100k guard checkpoint.
- Exact resume is allowed only for execution-backend changes that do not alter
  tensor/optimizer/replay/behavior ABI.  New heads, targets, exploration state,
  or rewards require explicit model-init and a new lineage.
- One-lineage-at-a-time rollout order: shadow/offline -> smoke -> fixed-budget
  paired A/B -> held-out promotion -> only then baseline integration.

## 6. Primary references

- PyTorch compiler programming model, graph breaks, and dynamic shapes:
  <https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/compile/programming_model.html>,
  <https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_dynamic_shapes.html>
- PyTorch automatic mixed precision: <https://docs.pytorch.org/docs/stable/amp.html>
- AMD PyTorch/ROCm compatibility and Radeon limitations:
  <https://rocm.docs.amd.com/en/docs-7.2.2/compatibility/ml-compatibility/pytorch-compatibility.html>,
  <https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2/docs/limitations/limitationsrad.html>
- LeanRL compiler-oriented examples: <https://github.com/meta-pytorch/LeanRL>
- Sample Factory asynchronous architecture and policy lag:
  <https://www.samplefactory.dev/06-architecture/overview/>,
  <https://www.samplefactory.dev/07-advanced-topics/policy-lag/>
- Achievement Distillation: <https://arxiv.org/abs/2307.03486>
- Options/SMDP framework: <https://doi.org/10.1016/S0004-3702(99)00052-1>
- Motif: <https://arxiv.org/abs/2310.00166>
- Kickstarting: <https://arxiv.org/abs/1803.03835>
- Vieira et al., drafting with fixed battle agents:
  <https://homepages.dcc.ufmg.br/~ronaldo.vieira/assets/pdf/sbgames-2020.pdf>
