# Semantic reset: stage-boundary entry conditions (v1)

Status: REVIEW CHECKLIST for docs/architecture/semantic-decision-graph-reset-v1.md.
These five items came out of the adversarial review of the reset decision.
Each later stage may begin only when its listed conditions are written down,
numbered, and (where applicable) measured — not agreed to verbally.

## EC-1. Bridge value calibration (blocks Stage 3)

The frozen combat champion's value output was trained under the per-step
gamma=1 run-scoped return contract. The macro candidate-Q trains under the
durable-floor clock (sts2_rl/semantics/clock.py, 0.997 per floor). These are
different units: bootstrapping macro targets directly from the champion's
value head injects a systematic scale error at every combat boundary.

Required before Stage 3:
1. A written bridge definition choosing ONE of:
   a. an affine recalibration V_bridge = a * V_champion + b fitted by
      regression on Stage-1/2 shadow data against realized new-clock combat
      returns, refreshed with each champion snapshot; or
   b. no value bridge at all: the combat segment contributes its REALIZED
      new-clock return (within-combat rewards + terminal floor tick) as an
      n-step Monte-Carlo segment between macro decisions.
2. A measured residual: on held-out shadow episodes the chosen bridge must
   explain combat-boundary returns with residual sigma below the smallest
   macro advantage the design intends to resolve (reference: the rest-vs-
   forge differential band measured across v42-v47, ~0.01-0.03 in old units;
   convert under EC-3's scale before comparing).
3. The bridge owns no learner head, no replay, no preference objective
   (reset doc §5) — the calibration constants live with the champion
   snapshot identity and are pinned like checkpoint hashes.

Option (b) is the default recommendation: it removes the calibration
surface entirely at the cost of higher-variance combat-boundary targets,
and Stage-3 canary data decides whether the variance is acceptable.

## EC-2. Compiler correctness regime (blocks Stage 2 collection)

1. Legality-set equivalence, per surface: for every decision snapshot, the
   compiled semantic candidate set and the engine legal-action set must be
   bijective up to declared mechanical suffixes. The shadow audit
   (scripts/validate_semantics_shadow.py) reports the equivalence counters;
   the Stage-2 entry bar is 100% on rest/shop/reward/map/event surfaces
   across every lineage journal audited in Stage 1, with unknown-transition
   rate quarantined to the opaque adapter only.
2. Fail-closed unknown surfaces: any observation no specific adapter claims
   routes to the opaque adapter and is EXECUTED as a raw decision (native
   candidates exposed unmodified), never guessed. Opaque-claim rate is a
   standing telemetry line; a rise is an incident, not a patch prompt.
3. Reveal boundary (§4.2): an action whose outcome introduces new candidates
   (random reveal) must terminate the current semantic action and open a new
   decision. Validation: the Stage-1 boundary review must include at least
   one recorded multi-select and one random-reveal flow replayed through the
   kernel with the boundary asserted.

## EC-3. Objective arithmetic under the floor clock (blocks Stage 2 training)

Reward version bump (v8) with a one-page derivation, computed before any
training run:
1. terminal_success_gap * (DECISION_CLOCK_BASE ** max_supported_floors)
   must exceed the maximum cumulative bounded secondary cost reachable in
   one run (HP-loss cost cap + revival cost cap + pace cost cap under v8).
   With base 0.997 and 60 floors the retained terminal fraction is >= 0.83;
   the v7 bundle (bounded by 0.99 < 1 terminal unit) does NOT automatically
   satisfy this inequality under re-scaling — it must be re-derived, not
   assumed.
2. The revival-budget curriculum stays OUT of the objective (engine
   curriculum only); the objective must not regain per-channel actor terms
   (reset doc §6).
3. Acceptance: the derivation lands in sts2_baseline as the versioned v8
   identity with tests asserting the inequality across supported budgets.

## EC-4. Double-Q coverage fallback trigger (blocks Stage 2 sign-off)

"Persistent unsupported value peaks" becomes a number before Stage 2 starts:
1. Definition: a candidate whose lifetime executed-sample count is below
   K_min = 8 appearing as the greedy argmax of its decision in more than
   F_max = 5% of held-out macro decisions, sustained across two consecutive
   evaluation gates.
2. If triggered: switch the macro learner to the predefined recurrent
   IQL/AWAC form (reset doc §8.2) in the NEXT lineage; do not add a
   correction loss to the Double-Q head. The switch is a config identity
   change, not a concurrent second objective.
3. Telemetry required from day one of Stage 2: per-decision greedy-argmax
   sample-count histogram, so the trigger is measurable without archaeology.

## EC-5. Invariant rewrite scope (blocks Stage 5 deletion; drafted at Stage 1)

CLAUDE.md changes staged with the migration, drafted now so deletion is a
paste, not a negotiation:
1. REWRITE at semantic-authority handoff: the action-rewrite ban is recast
   one level up — "the executor executes compiled semantic decisions;
   strategic choice is never rewritten below the semantic layer; the
   legality mask remains the only suppressor of semantic candidates."
2. DELETE with Stage 5: the two-sided corridor invariant paragraph
   (superseded by Q-learning's rank-independent gradients) and the
   transaction-liveness policy-logit obligations.
3. RE-HOME: forced-singleton "no policy target" survives naturally (a
   singleton is a mechanical suffix, never a semantic decision); revival
   cost invariants survive in the v8 objective wording.
4. The deletion commit for every retired mechanism cites this file plus the
   consolidation audit (wf_20d47fca) — no mechanism leaves silently.

## Standing rule

Any new rule, gate, exemption, or loss proposed during the migration must be
answered first with: "which existing stage artifact should have prevented
this need?" If none, the artifact list above is incomplete and gets amended
BEFORE the mechanism is considered. This checklist exists so that the reset
does not reproduce the accretion it replaces.
