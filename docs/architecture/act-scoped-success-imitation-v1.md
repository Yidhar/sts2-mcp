# Act-scoped success-conditional imitation (design v1)

Status: PROPOSED (design review pending). Target lineage: v42+ (after v41
budget-64 / reward-v7 measurement completes). Config identity bump required
(sts2-relational-curriculum-config-v18). No checkpoint tensor ABI change.

## 1. Problem statement (measured)

Complete-episode replay applies selected-action imitation only when a
decision's *primary* long horizon ended in success (`learner.py:3237-3258`:
the primary is the longest observed horizon, in practice the run horizon;
`primary_target.success is not True` suppresses the label). Failed runs are
value-only by design — anti-imitation of a whole failed trajectory is worse
than nothing and the suppression rationale documented in that block remains
correct.

The measured consequence across v38/v39 is a boom-bust supply problem:

- v38 (budget 16): 0 run victories in 218 training episodes over 100k steps.
  `episodic_primary_policy_loss` was zero for the entire run. The imitation
  plane was fully starved and the policy drifted into an HP-blind indifferent
  equilibrium (P(non-rest) ≈ 0.5 flat, mode-flipping between gates).
- v39 (budget 32): 6 run victories out of 137 episodes re-energized the
  channel (728/1439 updates with nonzero `episodic_primary_policy_loss`),
  but the final third of the run had no victories and the channel went silent
  again.
- Meanwhile `act1_cleared` was 105/137 (76.6%) in v39 and 76.6-100% at v38/v39
  evaluation gates. A run that dies on floor 40 typically contains a factually
  complete, authoritative Act-1 success whose decisions are currently
  discarded by the policy plane.

Success supply at run granularity is ~4%; at act granularity it is ~77%. The
proposal raises imitation data supply by an order of magnitude using only
facts already recorded, without imitating anything that failed.

## 2. Existing facts this design builds on (all already implemented)

1. Per-step act identity and authoritative act boundary outcomes live in the
   replay ABI: `episode_replay.py:223` (`act_boundary: BoundaryOutcome`),
   with construction invariants at `episode_replay.py:520-533` (an act index
   may only advance past a SUCCEEDED boundary) and horizon extraction at
   `episode_replay.py:591-593`.
2. Per-decision horizon targets already include the act horizon with its own
   `success` flag (combat/act/run task-value head family). The imitation gate
   currently reads only the longest observed horizon (`learner.py:3237-3243`).
3. Per-act revival usage at each boundary is computed during collection:
   `collector.py:3713` (`act_boundary_efficiency`), `collector.py:4326-4343`,
   `collector.py:5031-5056` (`revival_free_act1_clear`). These are exact
   monotonic simulator counters from `observation._training`, which are
   legitimate learning-plane facts and are never model inputs.
4. The trust-region, freshness-lag, importance-clip and surface-exemption
   machinery is shared and unchanged: `success_policy_trust_region_epsilon`
   (config.py:697), `policy_gradient_max_lag` (config.py:708),
   `success_imitation_exempt_surfaces` (config.py:703), detached
   `min(rho, clip)` ratios (`learner.py:3281-3289`).

## 3. Design

### 3.1 Eligibility ladder (replaces the binary primary-success gate)

For each policy decision in a replayed episode, evaluate in order:

1. **Run-success imitation (unchanged).** Primary horizon observed with
   `success is True` → imitation label with weight 1.0. Identical to today.
2. **Act-segment imitation (new).** Otherwise, if ALL of:
   - the decision's own act `k` closed with `act_boundary is SUCCEEDED`
     (authoritative, from the replay ABI — never inferred from floor numbers);
   - the act-`k` horizon target for this decision is `observed` with
     `success is True` (uses the existing act task-value target, so the
     decision's horizon bookkeeping and the boundary agree);
   - the **boundary health gate** for act `k` passes (§3.2);
   then emit an imitation label with weight
   `act_segment_policy_weight` (default **0.3**) multiplied into the existing
   per-label loss contribution. All downstream guards still apply in existing
   order: surface exemption → policy lag → trust region → importance clip.
3. **Suppressed (unchanged).** Otherwise value-only, counted in
   `failure_policy_suppressed_labels` exactly as today.

The final act of a failed run never qualifies: the act that contains the
death has no SUCCEEDED boundary, so its decisions remain value-only. Only
factually completed prefixes are ever imitated.

### 3.2 Boundary health gate

Purpose: refuse to imitate an act that "succeeded" by mortgaging the next act.
An act-1 clear that exits with 5% HP and 60% of the revival budget spent is a
liability, not a lesson. Gate inputs are exact facts at the SUCCEEDED
boundary step, computed once at collection time and stored per act in the
episode record (new frozen field, e.g.
`act_segment_health: tuple[ActSegmentHealth, ...]`):

- `hp_ratio_at_boundary >= act_segment_min_exit_hp_ratio` (default **0.35**);
- `revivals_used_through_boundary <= ceil(act_segment_max_revival_fraction *
  revival_budget)` with `act_segment_max_revival_fraction` default **0.34**
  (at B=64: ≤ 22 through act-1 exit; at B=16: ≤ 6). Unlimited-budget
  observations (budget -1/absent) use the reward-v7 reference budget 64
  (`sts2_baseline/objective.py:_training_revival_budget`) for the same
  fraction, keeping the gate meaningful on historical fixtures.

Both facts come from the boundary decision's stored observation summary and
the `_training` counters; no new observation surface is read. The gate is
deliberately coarse — its job is to exclude obviously mortgaged prefixes, not
to rank them. Contextual fine preference remains owned by return/Q learning.

### 3.3 Weighting and interaction rules

- `act_segment_policy_weight = 0.3` multiplies ONLY the selected-action
  likelihood term of qualifying labels. Value supervision is untouched (all
  horizons already train value heads on every episode).
- Run-success labels always dominate: if a run succeeded, rung 1 applies and
  the act-segment path is never consulted (no double counting by
  construction — the ladder is exclusive).
- `success_imitation_exempt_surfaces` (currently `["rest_site", "shop"]`)
  applies to act-segment labels identically — the corridor/exemption
  reasoning is orthogonal to which horizon granted success, and this closes
  the re-lock hazard for the surfaces that historically saturated.
- The revival-efficiency tie-break machinery (`primary_success_tie_tolerance`,
  revival policy caps) applies only on rung 1, unchanged. Act-segment labels
  carry no revival-cost policy term: per the invariant, revival cost trains
  policy only on factually successful *run* horizons. Act success is not run
  success and must not unlock the cost tie-break.
- Telemetry for suppression reasons stays disjoint (§6) so existing counters
  keep their exact meaning.

### 3.4 What this is NOT

- NOT a behavior gate: collection, legality, curricula and evaluation are
  untouched. Act boundaries remain episode metrics; this is a learning-side
  label-eligibility widening only, so the "act boundaries are never behavior
  gates" invariant is preserved.
- NOT counterfactual: every imitated decision is a factual action from a
  factually completed act with an authoritative boundary record.
- NOT a reward change: reward identity stays sts2-task-reward-v4 +
  sts2-run-survival-efficiency-v7.
- NOT retroactive: existing replay records lacking the new health field are
  treated as gate-failed (fail-closed), never re-derived heuristically.

## 4. Invariant compliance review

| Invariant (CLAUDE.md) | Compliance |
|---|---|
| Failure horizons receive no cost label; early failure never becomes cheap | Unchanged — act labels carry no revival-cost policy term (§3.3) |
| Act boundaries are episode metrics, never behavior gates | Learning-side only; collection untouched (§3.4) |
| No fabricated counterfactual actions | Only factual completed-act decisions are imitated |
| `_training` facts are not model inputs | Gate consumes them at collection/learning time only; encoder untouched |
| Forced singleton actions generate no policy target | Upstream `policy_decision` check unchanged (`learner.py:3235`) |
| Versioned immutable rewards | No reward change |
| Checkpoint ABI | No new tensors; new frozen replay field ⇒ replay sidecar schema version bump, handled like prior replay schema revisions (older checkpoints resume via model-init lineage as usual) |

## 5. Configuration (config-v18 additions, `episodic_learning` block)

```toml
act_segment_imitation_enabled = true      # default false in code; explicit in v42 config
act_segment_policy_weight = 0.3           # (0, 1); validated strictly below 1.0
act_segment_min_exit_hp_ratio = 0.35      # [0, 1)
act_segment_max_revival_fraction = 0.34   # (0, 1]
```

Validation mirrors the existing config style: strict types, finite ranges,
`act_segment_policy_weight < 1.0` enforced so run success always strictly
dominates. Flag default-off preserves byte-identical behavior for every
existing lineage config.

## 6. Telemetry (new learner_update fields)

- `episodic_act_segment_policy_labels` — labels emitted via rung 2
- `episodic_act_segment_health_gate_suppressed_labels` — act succeeded,
  health gate refused
- `episodic_act_segment_policy_loss` — the weighted loss component
- train_episode: `act_segment_healthy_acts` (count per episode)

Existing counters (`failure_policy_suppressed_labels`,
`success_policy_candidate_labels`, exemption/lag/trust-region counters) keep
their current semantics; rung-2 labels are counted separately, never merged.

## 7. Acceptance criteria (v42, gates at +25k/+50k vs v41 baseline)

1. **Supply stability (primary):** fraction of learner updates with a nonzero
   episodic policy loss ≥ 0.8 sustained (v39: 0.51 with boom-bust to 0; v38:
   0.0), with `episodic_act_segment_policy_labels` > 0 in every telemetry
   bucket.
2. **No act-1 mortgage regression (safety):** mean HP ratio and mean revival
   usage at act-1 SUCCEEDED boundaries must not degrade vs v41 baseline
   (one-sided tolerance 10%). This is the direct measurement of the
   act-1-greedy hazard.
3. **Health-gate calibration:** `health_gate_suppressed / (rung-2 candidates)`
   between 0.05 and 0.6. Near-0 means the gate is vacuous; near-1 means it is
   strangling supply — either triggers a threshold review, not a silent tune.
4. Win rate and floor p50 non-inferior to v41 at matched gates (this change
   targets signal supply, not an immediate win-rate jump; regression is the
   stop signal).

## 8. Hazards and mitigations

1. **Act-1-greedy lock-in** (imitating prefixes that win act 1 by spending
   act-2 resources): health gate (§3.2) + weight 0.3 + run-success dominance
   + acceptance criterion 2 as the measured tripwire.
2. **Easy-seed overrepresentation:** eligible segments skew toward easy maps.
   Mitigated by existing per-episode replay sampling quotas (unchanged) and
   the low weight; monitored via per-episode label concentration telemetry.
3. **Interaction with entropy/saturation:** more imitation pressure can
   sharpen distributions. The surface exemptions keep the historically
   saturating surfaces out; the policy-collapse-v2 breaker and corridor
   remain armed. No new mechanism needed.
4. **Stale-behavior imitation:** unchanged `policy_gradient_max_lag` and
   trust region already bound this; act-segment labels pass the same guards.

## 9. Test plan

- Unit: eligibility ladder truth table (run success / act success ×
  boundary outcome × health gate × exemption × lag) — every rung and every
  suppression counter asserted, including the fail-closed path for records
  without the health field.
- Unit: health-gate arithmetic at budget -1/absent (reference-64 fallback),
  budget 16/32/64 boundaries, and non-integer/NaN rejection.
- Determinism: identical episode replayed twice yields identical labels and
  identical counters (stateless gate, no RNG).
- Integration: synthetic 3-act episode fixtures — (a) died in act 2 with
  healthy act-1 boundary → act-1 decisions imitated at weight 0.3, act-2
  value-only; (b) same but exit HP below threshold → all value-only;
  (c) full victory → rung 1 only, rung-2 counters zero.
- Full suite + mypy strict + ruff + CLI dry-run per change discipline.

## 10. Non-goals

- No change to MATCHED_OUTCOME_PAIR (stays dormant liveness insurance; its
  strict same-node/same-candidate-multiset pairing principle is correct and
  is not widened toward run-outcome credit).
- No combat-segment imitation (combat horizons are too dense and the
  hazard analysis has not been done; act granularity only).
- No curriculum or reward change in this proposal.
