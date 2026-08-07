# V39 budget-32 natural-strategy lineage

## Decision

V38 completed 100,000 local environment steps but produced no successful run
at any held-out gate. Its final 64-seed audit exhausted all 16 revivals in
every episode. V39 raises the **episode-level revival budget from 16 to 32** so
the collector can again generate full-run successes while investigation of the
missing contextual macro credit continues.

This is a reviewed **model initialization**, not an exact resume. The changed
revival budget changes transition and return semantics, so optimizer state and
budget-16 replay must not be mixed into the new course.

Immutable source:

- run: `full-run-revival-v38-natural-strategy-model-init`
- run ID: `3051fdbd-7987-4319-a3ac-cd680fa3e408`
- checkpoint: `final-step-000100000`
- checkpoint ID: `74797dca-7c88-4f2d-b0a2-52f028291f3e`
- policy version / learner updates: `1675 / 1675`
- checkpoint manifest SHA256:
  `3ddbeb700c60b6deedcc8f99280a3cbe66306ef295003d4378dd46cbf0704251`
- metadata SHA256:
  `be09e1d80a4c23edd7000040b61dec33d06176410d29bfa600abffad9f79733b`

## What changes

```toml
[curriculum]
revival_budget = 32

[runtime]
total_environment_steps = 250000
```

All v38 natural-strategy behavior contracts remain unchanged: no transaction
entry forcing, no completion guidance, no selection-surface epsilon floor, no
HP threshold, and no hand-authored rest/forge/removal preference.

## Reset and inheritance contract

Inherited:

- all compatible network parameters;
- mature environment-step, policy-version, learner-update, entropy, epsilon,
  and liveness schedule offsets.

Fresh:

- optimizer;
- actor synchronization and rollout queue;
- episodic, transaction and failure-credit replay;
- RNG and lineage-local counters;
- evaluation journal.

## Evaluation

The held-out seed namespace stays fixed at `6,300,000`. Gates are scheduled at
0/10k/25k/50k/75k/100k/125k/150k/175k/200k/225k, with a 64-seed final audit at
250k. The guard remains stop-only and cannot roll back learned policy.

The budget change is a course recovery action, not a claim that the strategy
credit defect is solved. Root-cause inspection continues in parallel through
macro sensitivity, HP-conditioned rest/forge behavior, combat micro metrics,
entropy/advantage balance, and shared-trunk parameter drift.
