# V41 budget-64 / reward-v7 recovery lineage

## Root cause

Reward v6 used the cumulative revival cost

```text
C(k) = min(0.40, 0.010 k + 0.0009 k²)
```

and reached its cap at revival 17.  In a 32- or 64-revival course, revival 18
through the terminal budget therefore had exactly zero direct scalar marginal
cost.  V39 telemetry matched this defect: late training episodes consumed the
entire 32-revival budget, and held-out runs could spend 28--30 revivals in one
Act-1 boss fight without any additional revival reward cost.

This is a reward-definition error, not an optimization plateau.  Continuing
v6 would generate rational corpse-grinding behavior.

## Reward v7

V7 replaces the early-saturating quadratic schedule with the finite-course
schedule

```text
C(k; B) = 0.40 * min(k / B, 1)
```

where `B` is the positive `observation._training.revival_budget`.  Historical
or explicitly unlimited observations use the reviewed reference budget 64.

Consequences:

- budget 64: every revival 1--64 costs `0.00625`;
- budget 32: every revival 1--32 costs `0.0125`;
- budget 16: every revival 1--16 costs `0.025`;
- the complete revival preference still telescopes to at most `0.40`;
- together with HP-loss (`0.55`) and pace (`0.04`), survival efficiency stays
  bounded by `0.99`, preserving terminal-outcome priority.

The budget must remain constant inside an episode.  A positive non-integral or
mid-episode-changing budget fails closed.

## Lineage and source

Changing the reward fingerprint forbids exact resume.  V41 is a model-init
lineage from the same mature, healthy V39 anchor used to test V40:

- run ID: `e331ab97-94a6-4f0d-bde9-407261e509b9`;
- checkpoint: `healthy-validation-step-000050261`;
- checkpoint ID: `968d0302-b712-4b04-a2df-50554722500c`;
- policy / learner updates: `824 / 824`;
- manifest SHA-256:
  `4c7c36e528f98be5d450319f151d457190c3c751e4c1f27499a8bbd01ea86442`;
- metadata SHA-256:
  `c9460511144b05f2bc97028288fdc4ca2a0bae707b8e875687e6950de9eecbf0`.

Network parameters and mature schedule clocks are inherited.  Optimizer,
replay stores, RNG, rollout queue and lineage-local counters are fresh.  The
revival budget is 64 and behavior-side transaction forcing remains disabled.

## Fast source validation

The fixed source contains roughly 2.8 GiB of replay sidecars.  V40 already ran
a complete per-file SHA-256 preflight over this exact atomic directory.  V41
pins a dedicated immutable attestation extracted from that successful proof:

```text
runtime/validation/checkpoint-attestations/
  v39-healthy-050261-byte-verified-v1.json
```

Attestation SHA-256:

```text
d357e2e3a176ffd9fdd30350be8fb5a4af030f94fe771ac4f039a3e6918ffc58
```

Launcher and trainer re-hash the small attestation, checkpoint manifest and
metadata, then validate checkpoint ID, source run/step, complete file set,
path containment, sizes, hash descriptors, semantic identity and model ABI.
They do **not** read and hash every large payload again.  This authority is
accepted only for supervised model initialization; exact resume continues to
require a fresh full payload hash pass.
