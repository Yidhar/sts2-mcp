# Synchronous macro credit v1

Status: **engineering-ready for the next explicitly bounded reward-root pilot**,
2026-09-05. This is a separate experimental course, not a quality-certified
replacement for the baseline. B0 reconstruction, GPU stochastic collection,
GPU update/resume, final-publication held-out execution and checkpoint/publication
forward equivalence have passed their measured gates. Related CPU regression
passed 455 tests. The earlier greedy B1 artifact is not retroactively certified.
**Quality improvement, all-surface/late-Act coverage, value calibration and
large-scale long-training certification have not been established.**

## Context and decision

The current baseline is [hybrid recurrent V-trace recovery](hybrid-recurrent-vtrace-recovery-v1.md):
a stochastic candidate policy and scalar state value over macro/combat streams,
shared fusion, and split recurrent memory. Its public candidate scores are
policy logits. Archived IQL Q heads, expectile critics, replay and target-network
assumptions do not describe this controller. Importing an old healthy publication
is model initialization, not an exact IQL or V-trace training resume.

This course retains the model and semantic executor. It adds simulator-derived,
same-root conditional-action evidence, a long macro value target, and a
prior-preserving root policy projection. It does **not** add a candidate-Q head,
card-specific preference rules, inference-time search, quantile critic, or a new
replay store. Earlier documents' tentative candidate-Q proposal is not the
decision implemented here.

The statistical object is a noisy estimate of the frozen continuation policy's
conditional action value, with an explicit finite tail and bootstrap. It is not
optimal Q, exact expected regret, or an oracle that knows the best card.

## One frozen, synchronous cohort

1. Freeze the learner's actor-critic at version `k` before collection. Disable
   dropout for behavior and target evaluation; disabling dropout does not disable
   categorical policy sampling or learner autograd.
2. Use distinct training seeds and collect a natural reference from a full fresh
   reset. Every semantic action follows the exact flat legal distribution
   `pi_teacher`; no branch-balanced epsilon mixture is silently substituted.
3. Find the declared occurrence of a generic macro surface: reward, rest, shop,
   map, or event. Root selection is structural and fixed before outcomes, not
   chosen retrospectively for a favorable advantage.
4. Re-run that root's natural reference in an independent simulator process.
   Require its own same-action replay gate, not merely a previous B0 pass at
   another root. Fixed RNG replay is a reproducibility control, not a new sample.
5. For each remaining legal root candidate, fresh-reset and replay the full
   prefix. Verify state, ordered candidates, reward history and frozen recurrent
   evidence before dispatch. Force exactly that root candidate, then follow the
   same frozen stochastic teacher with the same fixed stopping rule.
6. Accept a root only when the complete executable candidate set and every
   required control pass complete. Failure, unknown truncation, byte/time/native
   budgets or a missing candidate abort the group; they are not zero returns.
7. Reconstruct all target-policy memory from semantic snapshots under version
   `k`, build detached targets, and perform one optimizer update for the cohort.
   Only then advance the teacher version and collect the next cohort.

The first implementation is deliberately `teacher_version == learner_version`
at target construction: policy lag is zero, not merely small on average. Reject
an aged or already-consumed cohort. Multiple epochs, asynchronous teacher
refreshes, or stale queue replay would be another course requiring new contracts.

The sampler consumes one RNG draw even at a forced root so matched continuations
can use common random numbers. Different candidates and deterministic controls
are correlated observations of one root. They must not be counted as independent
roots, rollout replicas, confidence samples or separate train/test examples.

## The three clocks and exact reward composition

There are three different counts: native dispatches, semantic policy decisions,
and macro semantic nodes. Mechanical select/confirm suffixes are native steps,
but do not advance recurrent policy memory or create extra learning decisions.
Combat decisions are semantic decisions, but do not consume the macro horizon.

Keep the established physical discount, with base `g = 0.997`:

```text
d_native(i) = 0                         for factual task terminal
            = g ** (floor_after - floor_before) otherwise

r_semantic(t) = sum_i [product_{j<i} d_native(j)] * r_native(i)
d_semantic(t) = product_i d_native(i)

R_macro(m) = sum_t [product_{u<t} d_semantic(u)] * r_semantic(t)
Gamma_macro(m) = product_t d_semantic(t)
```

Indices in each product start at that edge's beginning. A transition's own
discount affects its successors, not its own reward. Apply the formula to each
reward component as well as total reward. Keep raw component sums separately
for factual accounting. Do not sum all suffix rewards first and discount only
at the edge endpoint: a suffix can cross a floor before its final reward.

For example, native transitions `1->2: r=2`, `2->3: r=3`, `3->3: r=5` in one
semantic edge produce `2 + g*3 + g^2*5`, not `10`. The edge's future discount is
`g^2`. A same-floor suffix has discount one and preserves the raw sum. Durable
floor regression or a discontinuous native chain fails closed; a clamp must not
hide malformed training evidence. Do not apply another native-duration gamma
when folding an already discounted semantic edge.

Lambda is a trace parameter, not the physical clock. For mixed semantic row `t`,
use `ell_t = lambda_macro` only when the **next state** is macro; otherwise use
one. Thus a reward choice followed by many combat actions does not incur lambda
decay on each card play. The real bootstrap state's domain participates in the
last boundary decision. A factual terminal has zero future discount regardless.

## Long value targets and forced-root action targets

V-trace uses recorded behavior probabilities to correct a mismatch between
actor and evaluation policies. Its clipped ratios are not a proof of unbiased
evaluation of an arbitrary target policy. The source algorithm is
[IMPALA / V-trace, Espeholt et al. (2018)](https://proceedings.mlr.press/v80/espeholt18a.html).
Our macro-successor lambda placement is the course's additional clock contract.

With detached current/frozen values, let `rho_t` and `c_t` be the clipped
importance ratios from each mixed-domain row, including intervening combat:

```text
v_t = V_t + rho_t * (r_t + d_t * V_next - V_t)
          + d_t * ell_t * c_t * (v_next - V_next)

Y_t = r_t + d_t * [V_next + ell_t * (v_next - V_next)]
```

`Y_t` is the conditional-action target: its own root action is fixed. Its own
importance factors must not attenuate a rare forced candidate; downstream
sampled actions remain corrected through `v_next`. The actual intervention
behavior log probability is zero, not a fabricated teacher probability. It
cannot enter an ordinary sampled actor loss. In the lag-zero natural reference,
ratios are one up to numerical tolerance; the mixed formulation still makes the
required probabilities and future correction explicit.

Known macro-horizon censoring requires the actual next macro snapshot, memory
reconstructed under the same evaluation weights, and that state's scalar V.
Do not substitute zero, a different candidate's tail, the pre-action value,
the next observation with stale hidden, or an old reward-v8 source critic.
Unknown environment truncation or an administrative budget is rejected rather
than manufactured into terminal failure. A factual run terminal has no
bootstrap. An Act boundary alone is not the run terminal for this course.

The B1 greedy diagnostic deliberately has no calibrated tail V and therefore
exports `training_label_eligible=false`. Its partial observed returns cannot be
renamed stochastic teacher labels. A correctly implemented current-V bootstrap
still has estimation error and requires calibration evidence; it is not an
automatic long-horizon accuracy certificate.

## Root actor: teacher-prior projection, not branch balancing

For one complete root, average genuinely independent rollout replicas within
each candidate first, if replicas are later introduced. The current diagnostic
replays are not such replicas. Let `Y(a)` denote the resulting detached action
targets and `p(a) = pi_teacher(a | root)` the full legal teacher distribution.
The course solves a finite-root relative-entropy projection:

```text
maximize_q sum_a q(a) * Y(a)
subject to KL(q || p) <= epsilon_root, sum_a q(a) = 1

q_eta(a) proportional to p(a) * exp(Y(a) / eta)
L_macro_actor(root) = -sum_a q_eta(a) * log pi_learner(a | root)
V_root_target = sum_a p(a) * Y(a)
```

This prior-weighted exponential form and subsequent policy fitting are inspired
by [Maximum a Posteriori Policy Optimisation, Abdolmaleki et al. (2018), Eq. 8](https://arxiv.org/html/1806.06920#S3.SS2).
Here simulator return targets replace a learned Q estimator, and the constraint
is per collected root. This is MPO-style, not an assertion that the full MPO
algorithm or its theoretical improvement assumptions have been reproduced.

The prior is **not** uniform over candidates, uniform over branches, multiplied
by branch population, or divided by `n_branch`. Such factors alter the existing
flat policy law even when all outcomes tie. Equal targets retain the prior and
disable the root actor term rather than adding a cloning objective. The value
target evaluates the teacher (`p * Y`), not its improved posterior (`q * Y`).

Average the root actor loss over all complete accepted roots, not simulator
work, raw candidate count, branch count or repeated controls. An equal-target
root contributes zero actor loss but remains in that denominator. A missing candidate never
becomes a renormalized smaller root. Adaptive allocation/progressive pruning
and winner-only targets are out of scope: they add selection and censoring
bias. The largest noisy observed return is subject to selection optimism, not
an unbiased estimate of expected regret.

The root KL bound constrains the detached target distribution. One shared
neural-network optimizer step, including combat and value gradients, does not
guarantee a global policy KL bound or actual policy improvement. Report realized
post-update KL on collected roots separately; do not present it as a bound at
unvisited states. Tiny teacher prior can limit practical recovery even with
legal support; report it rather than silently flooring probabilities.

## Loss routing and occupancy

Each root contributes exactly one ordinary trajectory: its natural reference
from reset. The same-action control is discarded from learning. Forced-root
branches supply conditional-action targets at that root only; their suffixes
do not become ordinary macro/combat actor or value trajectories.

This restriction matters even though suffix actions follow the teacher: forcing
the root changes which suffix states are visited. Per-action likelihood ratios
there do not by themselves repair the altered state occupancy. Appending all
candidate suffixes to ordinary losses also overweights roots with many options.

| Objective | Source and treatment |
| --- | --- |
| Macro actor | Complete-root projection only; **replaces**, not adds to, old sampled macro V-trace policy loss. |
| Macro V | Long macro-successor-lambda targets on natural reference macro states. At a grouped root, use the teacher-weighted candidate target once rather than duplicate sampled and grouped supervision. |
| Combat actor / V | Existing short mixed-sequence V-trace targets, loss rows restricted to combat in the natural reference. |
| Intervention suffix | Detached return construction for its own root action; no ordinary occupancy loss. |
| Controls / diagnostics | No optimizer rows. |

Do not accidentally retain the old macro actor or macro entropy objective merely
because a shared baseline helper returns them. Any additional macro regularizer
requires an explicit course decision. The shared representation may still
receive both the new macro objectives and natural-reference combat gradients;
report their scales and interaction rather than asserting independence.

## Data, recurrent replay and module boundaries

- `joint/macro_credit_collection.py`: stochastic authority, fresh process reset /
  prefix protocol, exact behavior log probabilities and CPU semantic snapshots.
- `joint/macro_credit_data.py`: complete-root/cohort data contracts, training seed
  partition, immutable sparse snapshots, exact prefix/root matching and explicit
  ending/bootstrap facts. Natural candidate reuse must match the reference
  suffix, not just its first action. The selected teacher prior must match the
  natural root's recorded behavior probability.
- `joint/macro_returns.py`: detached mixed long-return calculation plus independent
  folded macro-edge oracle. Every macro and combat likelihood is available.
- `joint/macro_targets.py`: full-root prior-preserving projection and teacher V
  expectation. No model, optimizer or candidate-count weighting is owned here.
- `joint/macro_credit_learner.py`: one cohort update, reconstructed target memory,
  detached long targets, bounded gradient chunks and explicit loss routing.
  Macro and combat objectives have fixed one-half weights; macro value rows,
  combat value rows and choice-bearing combat actor rows have their own counts.
- `joint/macro_checkpoint.py`: atomic cohort-boundary state and publication.
- `macro_branch_diagnostic.py`: B1 greedy evidence runner and shared reset/dispatch
  mechanics; its CLI remains diagnostically ineligible for training.

Store semantic observation/candidate snapshots and selected actions, not saved
behavior hidden as learner inputs. Reconstruct the full semantic prefix under
one evaluation version. Do not let truncated backpropagation boundaries silently
cut target recursion: carry suffix targets across reverse chunks if chunked.
Mechanical steps change environment/reward facts without extra recurrent
forwards. Diagnostic hidden snapshots are audit evidence only.

Snapshot budgets count unique CPU array payload, including real bootstrap
snapshots and already retained reference/candidate trajectories. A new candidate
capture receives only the remaining root budget; its independently encoded
prefix is released after exact verification and reference-prefix sharing. This
is not a process-RSS cap: Python object headers, model/optimizer state, native
simulator processes and temporary encoder work are separate resources. The
current encoded snapshot is checked after allocation, bounded by the encoder's
own capacities. Report process/GPU peaks separately. Cooperative guards cannot
preempt an in-flight model kernel.

Future outcomes are legitimate training targets, not automatically information
leakage. Leakage occurs if the deployed actor sees future-only features, an
outcome-selected hidden state, or an oracle-selected continuation at inference.
The actor must consume only its causal observation prefix at each decision.

## Seeds, consumption and boundary recovery

Training uses non-negative signed-int32 **even** seeds. Diagnostics and held-out
work use the disjoint odd partition; a consumed diagnostic seed is not a fresh
final-audit seed. Partition by run seed, not by root, candidate or replay pass.
Outcome-driven seed replacement and reporting only successful roots would bias
the experiment; record every attempted root and abort reason.

There is no mid-simulator exact resume. Commit model, optimizer, relevant RNG
state, teacher/update version, next seed cursor, consumed-cohort cursor and
course/model/reward identity only after a complete update. Recovery loads that
boundary and recollects an uncommitted cohort using a new evidence directory;
it does not consume partially collected data or silently skip its seeds.
An interrupted update returns to the previous committed boundary, not to a
partially mutated model. Publication is inference output, not a replacement
for the training checkpoint. Model-only initialization must be labelled as such.

## Evidence and minimal acceptance gates

### Completed diagnostic and collection evidence

The [B0 runbook](../runbooks/macro-root-b0.md) records the completed bounded
observable reconstruction smoke at the first reward root of diagnostic seed
13400001. It is not a proof of hidden RNG cloning, revival/Act-boundary coverage,
or other roots' reconstructibility.

The first GPU B1 artifact, `analysis/macro-branch-b1-20260905-smoke01` under the
external runtime artifact root, completed a same-action gate and all four root
candidates with differing observed returns. It used six subsequent macro nodes.
Its original summary predates the intra-semantic native-discount repair and
the separate observed-combat-exit counter. Recompute from retained transition
facts or rerun bounded evidence before citing its numbers under the new contract.
An observed combat exit with typed result `none` is not a declared victory.

The new stochastic collector's independent artifact is
`analysis/macro-credit-collector-20260905-smoke01/summary.json` under the external
runtime artifact root. Read-back on 2026-09-05 confirmed `status=complete`,
`learner_updated=false`, `same_action_gate=matched`, and:

| Item | Observed fact |
| --- | --- |
| Source and policy law | Healthy IQL parent used only for actor-critic initialization; teacher version 0, frozen stochastic continuation on GPU. |
| Training root | Even seed 13400000, first reward root, zero-based semantic root index 53. |
| Horizon | 12 post-root macro decisions, durable floor 3 to 7, all passes ended at `macro_horizon`. |
| Natural reference | 141 semantic steps: 20 macro and 121 combat. |
| Candidate set | Four complete legal candidates; reference action reused, three other actions intervened. |
| Native dispatches | **722 total**: reference 145 + control 145 + alternatives 151, 147 and 134. |
| Combat coverage | Every pass observed three starts and three exits; authoritative completed count remained zero because typed results were `none`. |
| Retained sparse payload | 5,953,008 bytes, including candidate/bootstrap storage; not RSS. |
| Evidence storage | 185,051,779 JSONL bytes; 185,392,233 bytes across all 13 artifact files at read-back. |

This evidence uses the corrected native-discount composition. It establishes
stochastic collection and per-root replay at this one bounded reward root, not
an optimizer update, a calibrated tail value, or policy improvement. It neither
rewrites nor retrospectively validates the old greedy B1 artifact. The summary's
five pass-level returns include the repeated control; they are not five
independent candidates or independent training samples.

### Regression and implementation follow-up

The related CPU regression log,
`analysis/macro-credit-readiness-20260905/regression-tests.txt`, ends with
**455 passed in 47.56 seconds**. This is the selected related regression set,
not a claim that every repository/environment test passed. After the active
snapshot-budget repair, the implementation owner additionally reported 17
collector and 18 policy focused tests passing, plus mypy checks on three
production modules. These follow-up tests overlap the broader regression and
must not be added to 455 as new independent coverage.

### Bounded first-reward-course readiness checklist

At this revision, two real GPU updates across separate invocations have been
committed, and the final held-out execution and checkpoint/publication audit
artifacts have been read back as passed. The following checks are measured
engineering gates, not formal policy-quality evaluation. Successful static
preflight or the presence of an output directory would not have been enough.

- [x] Related arithmetic, data, collector, learner, checkpoint and runner CPU
  regression completed, with its bounded test log retained.
- [x] A real GPU stochastic full-root collection completed with its own matched
  same-action control and natural-reference macro/combat coverage.
- [x] Bootstrap storage and active-capture remaining-budget accounting have been
  repaired and covered by focused tests; the residual RSS/encoder-allocation
  limitations above remain explicit.
- [x] First complete CLI GPU update commits version 0 to 1 from an even-seed
  cohort, with finite separate macro/combat losses, policy/value/shared gradients,
  parameter change, detached root targets, measured post-fit root KL and no old
  sampled macro actor contribution.
- [x] A separate boundary-resume invocation commits version 1 to 2, advances the
  next seed cursor and does not reuse the preceding cohort or a stale teacher.
  Version-1 missing-publication repair was independently verified tensor-exact.
- [x] The final version-2 checkpoint and publication pass the tensor, strict
  restore and reconstructed-forward equivalence audit. The recorded partial
  source manifest matches; this is not a complete transitive dependency lock.
- [x] A bounded held-out odd-seed execution check loads the resulting publication,
  records its actual termination/censoring and coverage, and performs no optimizer
  update. Report its outcome even if poor; one held-out smoke is not a quality
  improvement estimate or tail-V calibration.
- [x] Learner wall time, learner-phase GPU allocation/reservation peaks, process
  RSS peaks, final output bytes and frozen runtime/partial-source identity are
  recorded. The bounded run is stopped, writer lock is absent, and checkpoint /
  publication checks passed. **Whole-course GPU peak was not measured.**

Closing these gates means engineering readiness for the explicitly bounded
**first-reward-root** course only. It does not establish all rest/shop/event/route
roots, late-Act or Act-3 correctness, or a safe unconstrained long training run.
The CLI records `quality_certified=false` and `long_run_certified=false` for
bounded completion. A scoped human-readable pilot-readiness decision must name
its evidence and remaining limitations, not reinterpret these flags as quality
certification. An uncalibrated value bootstrap prevents quality promotion and
large-scale long-run certification, but does not by itself prohibit a subsequent
explicitly budgeted, evaluated pilot after these engineering gates close.

### Real GPU update, resume and publication-repair evidence

The experiment is `hybrid_runs/macro-credit-v1-20260905-smoke01` under the external
runtime artifact root. Its two `cohorts/*/update.json` files and the readiness
directory's `initial-update.txt` / `resume-update.txt` were read back directly.

| Item | Initial invocation | Independent resume invocation |
| --- | --- | --- |
| Teacher to committed update | 0 to 1 | 1 to 2 |
| Training seed | 13400000 | 13400002 |
| Next seed cursor | 1 | 2, corresponding to seed 13400004 |
| Natural macro / combat rows | 20 / 121 | 17 / 67 |
| Policy lag | 0 | 0 |
| Total reported loss | 0.5980033565 | 0.5242816410 |
| Parameter delta norm | 0.1084993430 | 0.08279072995 |
| Maximum autograd span | 16 semantic steps | 16 semantic steps |
| Learner update wall time | 19.748962 seconds | 14.667777 seconds |
| Peak CUDA allocation / reservation | 317,266,432 / 354,418,688 bytes | 357,869,056 / 379,584,512 bytes |
| Process peak RSS | 1,940,942,848 bytes | 1,993,113,600 bytes |

These wall times are learner-update durations, not complete simulator collection
times. The CUDA peaks were reset for each learner update, so they do not measure
the GPU peak of all collection/initialization/checkpoint activity. Process RSS is
a process peak, not a simultaneous process-tree/system peak including HeadlessSim.
Both updates reported finite nonzero policy-head, value-head, domain-stream
and shared gradient norms, separate macro/combat objectives and actual post-fit
root KL. `status.json` records committed update 2, next seed index 2 and the
checkpoint/publication paths. The version-2 artifact audit described below
verified these facts separately rather than inferring them solely from status.

Before the resume invocation, the version-1 derived publication was deliberately
removed. `analysis/macro-credit-readiness-20260905/publication-repair-proof.json`
records successful reconstruction from the committed checkpoint, with all **238
online tensors exactly equal** to the saved version-1 publication. The observed
training state at that proof was collecting at committed update 1 / seed cursor
1; only subsequently did the invocation commit update 2. This is real recovery
evidence rather than merely a unit-test claim. The later final-version audit is
a separate check and is recorded below.

For the first update, the action with the highest computed lambda target was
candidate 0. Its teacher probability was 32.393953%, its projected target was
40.407670%, and its measured post-update probability was 36.687748%.
`KL(q || teacher)` was 0.02; measured `KL(updated || teacher)` was approximately
0.006284634. This confirms an outcome-target-directed parameter update at the
measured root, not an environment performance improvement or a global KL bound.

In particular, descending observed-prefix return order was `[0, 3, 2, 1]`, while
the constructed lambda-action-target order was `[0, 2, 3, 1]`. Intervening macro
V predictions and the horizon bootstrap therefore affect the ordering; the
targets are not simply relabelled factual prefix returns. The source scalar V
was inherited from a reward-v8 parent into a reward-v9 course and has not been
calibrated for this use. The numeric `expected_improvement` metric is improvement
against those estimated targets only, not measured policy-quality gain or
optimal-Q evidence. One or two working updates do not resolve that limitation.

### Final-publication execution and artifact audit

`analysis/macro-credit-heldout-20260905-smoke01/summary.json` records a successful
bounded greedy B1 execution check of the **version-2 macro-credit publication**
on odd seed **13400003**. The complete four-candidate group and its same-action
control passed: 12 subsequent macro decisions, durable floor 2 to 5, three
observed combat exits per pass, all ending at `macro_horizon`. Native dispatches
totalled **430** (`84 + 84 + 86 + 86 + 90`). The reference contained 80 actual
semantic decisions. `training_label_eligible=false` and `tail_value_used=false`
remain explicit: this check performed no learning and supplies no calibrated
full-run quality estimate. Its consumed seed is not a fresh future final audit.

`analysis/macro-credit-readiness-20260905/checkpoint-publication-audit.json`
records `status=passed` and separately confirms:

- committed update 2, next seed cursor 2 and next training seed 13400004;
- all **238 model tensors** exactly equal between checkpoint and publication;
- fresh strict model/optimizer restoration, with **238 optimizer parameter
  states** present;
- the recorded **18-file partial source manifest** matches the frozen runtime
  sources; its declared scope is selected direct dependencies, not a complete
  transitive dependency/environment lock;
- re-encoding and reset-to-prefix reconstruction of **80 actual held-out
  semantic decisions**, with maximum logit error **0.0** and maximum recurrent
  hidden error **0.0** between restored checkpoint and publication inference.

The audit explicitly sets `next_bounded_reward_root_training_ready=true`, while
`quality_certified=false` and `long_training_certified=false`. At closeout the
training experiment contained **31 files / 440,655,630 bytes**, including recovery
backups; the writer lock was absent. The operator also confirmed no active
HeadlessSim process. These counts are for the training experiment directory,
not the separate collector, held-out or audit evidence directories.

**Decision:** permit the next bounded reward-root pilot with explicit collection,
update and evaluation budgets, preserving the frozen identities and checkpoint
boundary rules. Do not promote this result to a claim about all rest/shop/event/
route roots, Act-3 reconstruction, calibrated long-horizon values, policy quality,
or unrestricted training scale.

### Mathematical and longer-course acceptance requirements

The following requirements remain the basis for review. The engineering gates
above have now passed their bounded checks; this does not establish the final
item's broader quality/coverage conditions:

1. **Arithmetic:** native-to-semantic-to-macro discount oracle, same-floor suffix,
   macro-successor lambda, terminal/censored bootstrap, forced-root likelihood
   exclusion, detach and reverse-chunk equivalence tests.
2. **Root law:** unequal candidate/branch cardinalities, equal-return prior
   preservation, normalized teacher probabilities, root-weight normalization,
   missing-candidate rejection and duplicate-consumption rejection.
3. **Collection:** stochastic behavior probability reconstruction, per-root replay
   gate, natural reference reuse, causal prefix replay, all budget/error cleanup,
   no intervention-suffix ordinary rows, even/odd seed rejection.
4. **Bounded training smoke:** a complete even-seed cohort, one finite
   optimizer update with separate macro/combat metrics, no legacy macro actor
   contribution, output entirely outside source, and successful boundary
   checkpoint load/identity verification. Initial update and independent resume
   have now passed, as have final-artifact forward equivalence and bounded
   held-out execution. This is scoped pilot readiness, not quality certification.
5. **Before a longer course:** corrected B1 coverage, tail-V calibration on held-out
   completed tails, several independent roots/seeds, revival/Act-boundary replay,
   realized root KL and prior-support telemetry. Report negative results and
   uncertainty; do not promise improvement from one successful optimizer step.

No inference policy improvement, risk-sensitive improvement, global KL control,
unbiased off-policy estimate or statistically significant advantage is implied
by unit tests, deterministic replay, root projection feasibility, or a smoke.
