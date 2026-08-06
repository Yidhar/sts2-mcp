# V38 natural-strategy lineage

## Decision

V38 withdraws the temporary behavior-side transaction scaffolding after v37
demonstrated reliable factual completion experience for forge and card removal.
It is a reviewed **model initialization**, not an exact resume, because the
behavior policy that generated trajectories changes even though the network
shape does not.

Immutable source:

- run: `full-run-revival-v37-liveness-stability-model-init`
- run ID: `b70cc736-267d-45af-8e4d-9219fed2105b`
- checkpoint: `healthy-validation-step-000020337`
- checkpoint ID: `d0e9481f-8bb5-4e28-859f-ecc401f28685`
- environment steps: `20,337`
- policy version / learner updates: `338 / 338`
- checkpoint manifest SHA256:
  `efe8b3c4308993f5dd288cdaa6e51a3b52410f14fb4528857ec5496e3dd8010a`
- metadata SHA256:
  `8c5577b7df39e6ca9c1b2452f49986481d909e3bb0ed70e5179185df59c86f0c`

## Withdrawn behavior scaffolding

V38 explicitly sets:

```toml
[curriculum]
selection_surface_epsilon_floor = 0.0

[transaction_exploration]
enabled = false
operations = []
entry_epsilon_floor = 0.0
completion_guidance_probability = 0.0
```

Consequently neither rest-site entry, shop removal entry, nor transaction
completion is selected by a hand-authored behavior mixture. Greedy evaluation
and training behavior now expose the learned policy preference directly.

## Retained learning and safety mechanisms

The following are deliberately retained because they are learning contracts or
bug fixes rather than action preferences:

- the factual transaction completion loss and lifecycle/SMDP-Q corridor;
- the five-percent numerical support floor used only by the loss, not action
  selection;
- success-imitation exemptions on `rest_site` and `shop` to prevent a one-way
  macro-policy ratchet;
- saturation-aware liveness risk-actor eligibility;
- hierarchical entropy normalization and the soft-collapse breaker;
- reward v6, revival budget 16, and the existing HP-loss/revival economics.

No HP threshold, forge reward, rest reward, removal reward, action mask, or
forced completion rule is introduced.

## Reset and inheritance contract

Inherited:

- compatible network parameters;
- the mature environment-step, policy-version, learner-update, entropy, epsilon,
  and liveness schedule offsets recorded by the source checkpoint.

Fresh:

- optimizer;
- actor network synchronization state;
- rollout queue;
- episodic, failure-credit, and transaction replay;
- RNG and lineage-local counters;
- evaluation journal.

## Evaluation contract

The seed namespace remains `6,300,000`, allowing paired v37/v38 held-out
comparisons. Early gates are `5k/10k`, standard gates are `0/20k/40k/60k/80k`,
and the final audit is at `100k`.

The liveness guard is stop-only from 10k onward. `max_rollbacks = 0` guarantees
that a newly learned strategy is not silently discarded and replaced with an
older gate checkpoint.

Primary acceptance measurements:

1. rest versus forge probability stratified by pre-rest HP fraction;
2. shop removal entry rate conditional on the action being legal and affordable;
3. forge/removal transaction completion and cancel-cycle rates without guidance;
4. Act 1 / Act 3 / run success, HP loss, revivals, and deadlocks on paired seeds;
5. entropy, liveness actor loss, and shared-trunk gradient norms for recurrence
   of the v36 collapse.
