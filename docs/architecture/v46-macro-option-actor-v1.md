# v46 factual macro-option actor package (design v1)

Status: IMPLEMENTED, pending clean preflight and launch. Target lineage: v46,
explicit model-parameter initialization from the fixed v44 healthy checkpoint
at local step 100,478. Reward stays `sts2-run-survival-efficiency-v7` and the
revival budget stays 40.

## 1. Root cause

The v44 plateau is not evidence that the shared model is too small. The
measured failure is a missing actor-credit path:

1. Forge/remove completion experience reached the replay, but entry actions at
   literal policy probability zero received no effective V-trace, entropy or
   importance-weighted imitation gradient.
2. A completed lifecycle previously gave every selected card the entry-origin
   return, so the model could not distinguish which upgrade/removal target
   caused the later outcome.
3. The simulator knew the upgraded card state, but it was not transported from
   the nested selectable-card DTO into the grounded tensor. The model therefore
   had no exact factual representation of "this card after one upgrade".
4. Forge owned a short option target while rest did not. Rest-versus-forge was
   forced back through a noisy full-run return even though both actions occur
   on the same legal surface.

Increasing hidden size would leave all four defects intact. It could fit the
same incomplete objective faster, not learn the missing relation.

## 2. Implemented semantic path

### 2.1 Native upgrade preview

The pinned HeadlessSim patch clones the exact selectable `CardModel`, applies
one canonical upgrade to the detached clone, and emits it as
`upgrade_preview`. The Python translator:

- recursively translates the preview with the same card contract;
- rejects preview-of-preview recursion;
- rebinds the detached clone to the source pile and source instance relation;
- lifts the preview onto the candidate root while retaining it under the card
  for the world-selection projection.

Grounded encoding v15 incorporates this transport contract in its fingerprint.
Shapes are unchanged, but exact resume from v14 fails closed because replay
tokens have different meanings. The reviewed v14 -> v15 edge permits network
parameter initialization only.

### 2.2 Selection-origin factual option target

For every positive card selection inside a committed upgrade/removal
lifecycle, the target starts at that selection step and ends at the same
reviewed next-rest-site, Act-boundary or terminal endpoint as the entry option.
Entry-prefix reward and select/deselect churn are not copied onto the card.
Each selected card therefore trains `Q(state, card candidate)` against its own
factual suffix.

### 2.3 Symmetric rest/forge horizon

A rest-site action produces a lifecycle only when the authoritative next state
shows a strict HP increase. A label or action name is not sufficient. The rest
entry then uses the same next-rest/Act factual horizon as forge. This lets the
candidate-conditioned Q head compare rest and forge in the same state and on
the same return scale without a hand-authored HP threshold.

### 2.4 Bounded actor bridge

The new bridge converts a calibrated factual option advantage into a direct
legal-candidate cross-entropy term. It exists because
`rho * A * grad(log pi)` and entropy both become ineffective after a branch is
numerically absorbed at zero.

For selected action `a`, the detached counterfactual baseline is the current
policy-weighted Q over **other legal actions only**:

```text
B(s,a) = sum[b != a] softmax(log pi(b) | b != a) * stopgrad(Q(s,b))
A_opt  = stopgrad(G_factual(a) - B(s,a))
```

Positive advantage minimizes `-log pi(a)`. Negative advantage minimizes the
negative log probability mass of the legal alternatives. The baseline must not
include the selected action: doing so lets a probability-one incumbent cancel
its own advantage and recreates the absorbing state.

The path is fail-closed. It emits no actor label unless all conditions hold:

- learner update is at or past the configured calibration phase;
- collection policy version is not from the future and lag is bounded;
- exact collection/current **model log-probability** drift is bounded (the log
  value is recorded directly so float32 probability underflow to zero loses no
  provenance);
- selected transaction-Q residual against the factual target is below the
  configured gate;
- the state has at least one other legal action.

Target, counterfactual Q baseline, strength and every gate are detached. Only
the actor distribution receives gradient. The coefficient is 0.05, strength is
clipped at 1.0, and activation begins at learner update 256.

## 3. Explicit non-features

- No fixed preference for rest, forge, upgrade, removal, card name or HP band.
- No reward bonus for entering or completing a transaction.
- No action rewrite, forced confirm, legality override or evaluator guidance.
- No exact continuation across config-v18/v19, encoding-v14/v15 or transaction
  replay-v5/v6.
- Existing `rest_site`/`shop` success-imitation exemptions remain in place;
  the new factual option path, not an incumbent imitation ratchet, owns the
  contextual preference.

The only behavior-side bootstrap retained from the unrun v45 recipe is a 0.15
entry exploration floor for the single-decision `reward_skip` and
`relic_purchase` branches. Upgrade/remove/rest scaffolding is zero, and
completion guidance is structurally zero.

## 4. Compatibility and lineage

V46 initializes network tensors from exactly:

```text
run_id: ade6ab3a-2eda-4b5b-8fe4-3710858f9a65
checkpoint: healthy-validation-step-000100478
checkpoint_id: d44625a7-2e50-4449-9f5c-4f7b62f3c5f3
policy_version / learner_updates: 1646 / 1646
```

The launcher pins the fixed source path within the operator artifact-root contract plus
checkpoint ID, manifest SHA-256, metadata SHA-256, run ID, step, policy version
and learner update. It uses `--initialize-from`; `--resume` is forbidden.
Training horizon is 250,000 new environment steps with paired 16-seed gates and
a 64-episode final audit.

## 5. Acceptance and diagnostics

At 25k/50k/75k/100k gates, evaluate all of the following together:

1. `transaction_advantage_policy_labels` becomes non-zero after update 256;
   suppression counters explain every rejected label.
2. Q residual and policy-drift suppressions fall rather than remaining 100%.
3. Rest probability conditions on HP/Act and no longer flips as an entire gate
   between all-rest/all-forge modes.
4. Upgrade targets separate from removal targets; the same basic card must not
   dominate both surfaces.
5. Completed selection lifecycles do not regress into select/cancel cycles.
6. Held-out Act-1/Act-3 clear rates, revival counts and HP loss do not regress
   while macro competence improves.
7. Entropy, main gradient norm, liveness gradient norm and cost-actor loss show
   no v32/v36-style shared-trunk collapse.

Loss alone is not an acceptance metric. Paired held-out outcomes and the
per-decision run detail remain authoritative.

## 6. Required verification

- focused transaction DTO/replay/learner tests;
- config/checkpoint/encoding compatibility tests;
- full RL test suite, Ruff and strict mypy;
- repository contract/dependency checks;
- pinned native build with exact patch-tree verification;
- deterministic held-out forge E2E proving that removing only
  `upgrade_preview` changes the grounded tensor;
- launcher dry-run/preflight before any detached process starts.
