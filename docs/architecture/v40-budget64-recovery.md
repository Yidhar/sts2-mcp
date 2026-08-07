# V40 budget-64 completion-recovery lineage

## Decision

V39 proved that a 32-revival course can still produce successful runs, but it
did not keep a sufficiently dense success distribution.  At its reviewed
50,261-step gate it cleared Act 1 on all 16 held-out seeds, reached Act 3 on
7/16, completed one run, and had no liveness failure.  By the 75k gate the
success count had returned to zero, and the last ten inspected training
episodes all failed while almost every one exhausted the 32-revival budget.

V40 therefore restores the episode-level revival budget to **64**.  The fixed
source is V39's healthy 50,261-step evaluation anchor, not the later online
state.  This is a reviewed model initialization because the changed revival
budget changes transition and replay semantics; it is not an exact resume.

Immutable source:

- run: `full-run-revival-v39-budget32-natural-strategy-model-init`
- run ID: `e331ab97-94a6-4f0d-bde9-407261e509b9`
- checkpoint: `healthy-validation-step-000050261`
- checkpoint ID: `968d0302-b712-4b04-a2df-50554722500c`
- policy version / learner updates: `824 / 824`
- checkpoint manifest SHA256:
  `4c7c36e528f98be5d450319f151d457190c3c751e4c1f27499a8bbd01ea86442`
- metadata SHA256:
  `c9460511144b05f2bc97028288fdc4ca2a0bae707b8e875687e6950de9eecbf0`

## Course contract

```toml
[curriculum]
revival_budget = 64

[runtime]
total_environment_steps = 250000
```

The transaction behavior scaffolding remains disabled.  V40 does not force
forge, heal, card removal, selection completion, or an HP threshold.  It
changes only the completion budget and runtime lineage paths relative to V39.

The current v6 scalar revival cost reaches its bounded cap before revival 18.
Consequently V40 is explicitly a **completion/success-data recovery course**,
not the final low-revival efficiency course.  Exact future-revival labels and
the episodic revival-cost heads remain active, but a later reviewed reward or
finite-budget curriculum will still be required to drive successful policies
towards 0--1 revival.

## Reset and inheritance

Inherited:

- compatible network parameters from the fixed healthy V39 anchor;
- mature environment-step, learner-update, policy-version, entropy, epsilon,
  and liveness schedule clocks.

Fresh:

- optimizer;
- rollout queue and actor synchronization state;
- episodic, transaction, and failure-credit replay;
- RNG and lineage-local counters;
- evaluation journal.

Evaluation uses the same held-out seed namespace and gates at
0/10k/25k/50k/75k/100k/125k/150k/175k/200k/225k, followed by a 64-seed final
audit at 250k.  The evaluation guard is stop-only and cannot roll back or
discard a learned policy segment.
