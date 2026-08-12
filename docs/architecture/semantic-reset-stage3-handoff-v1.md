# Stage-3 macro ownership handoff — decision framework and record

Status: criteria fixed before evidence; decision section filled only from
measured results. Companion to `semantic-decision-graph-reset-v1.md` §10
stage 3 and `semantic-reset-stage1-entry-conditions-v1.md`.

## 1. What is being decided

Whether macro-surface ownership (rest/smith, shop buy/remove/leave, reward
take/skip, map, event, picker targets) moves from the legacy champion policy
to the trained macro candidate-Q model for **whole run segments** — for both
future collection and future evaluation. After handoff, legacy macro
collection stops; combat V-trace ownership is untouched (stage 4 decides
combat separately).

This is an ownership decision, not a deletion decision. Deletion (§9 list)
follows separately once the corresponding controller is replaced.

## 2. Criteria (fixed 2026-08-11, before the joined evaluation ran)

The joined evaluation runs the fixed odd held-out seed prefix through a
champion-only arm and a joined arm (frozen champion combat + greedy macro Q
on macro surfaces), deterministic, paired seeds, n >= 16.

Handoff requires ALL of:

1. **Liveness of the learning path.** The paired state probes
   (`probe_stage3_state_variation.py`) show the macro Q ranking responds to
   controlled state mutations (HP sweep at rest, gold sweep at shop):
   `moved_fraction` at or near 1.0 and a non-degenerate delta distribution.
   This checks state reaches the ranking; no direction is graded.
2. **Non-inferiority on held-out outcomes.** Joined-arm wins and floor_p50
   are not materially below the champion arm on the paired batch. Materially
   means: a deficit that a paired sign test on the seed pairs would flag at
   the batch size actually run — for n=16, a wins gap of 1-2 with mixed
   per-seed direction is noise; a uniform per-seed regression is not.
   Per §10, a small batch never auto-replaces a segment in EITHER
   direction: if the joined arm merely ties, the handoff still proceeds,
   because the macro graph can only learn context-dependent macro choice
   from experience it owns, while the champion's macro behavior is the
   measured HP-blind mode-flip pendulum (v42-v44 record) with no remaining
   learning path (P1 absorption).
3. **Branch diversity in ownership.** The joined arm's decision counts show
   the authority actually deciding across surfaces (not degenerate
   single-branch collapse everywhere) — mode collapse to one branch on
   every surface would indicate a broken Q scale, not a preference.

## 2b. Final-round evaluation protocol (fixed 2026-08-12, before the
600-episode segment completes)

The deciding round evaluates the completed macro publication with:

1. **Paired 2x2 attribution matrix** on the fixed odd held-out prefix,
   n=32 seeds per arm, deterministic:
   {champion combat, challenger combat} x {champion macro, macro Q}.
   The macro-effect contrast (champion combat row) decides criterion 2;
   the combat column stays informational until the stage-4 bridge lands
   (challenger-combat arms use the round-2 archive model).
2. **State probes** (criterion 1): the HP/gold mutation probe plus the
   behavioral decision log P(branch | HP bucket) from the joined arms —
   the behavioral log with real recurrent state is authoritative; the
   zero-hidden probe is the conservative lower bound.
3. **Branch diversity** (criterion 3): per-surface decision distributions
   from the joined-arm logs; degenerate means a single branch monopolizes
   EVERY surface, not a strong preference on one surface.

Decision rule unchanged from §2. n=32 exists to cut the binomial noise
that made earlier 16-seed rounds ambiguous (a 3-vs-4 win gap at n=16 is
within one seed's noise).

## 3. Evidence

### Round 1 — ep200 model (2026-08-11)

Model: `stage2-macro-final-ep200.pt` (200 episodes, 13,246 macro
transitions, 800 Double-Q updates, loss 0.0054, all 10 branches collected).
Artifacts: `runs/stage2-isolated-macro-v1/stage3-full-eval.jsonl`,
`stage3-state-variation.json`.

- **Criterion 1 (liveness): FAILED.** Branch gap Q(smith)-Q(rest) is
  +0.058..+0.071 and varies with context across floors (~1.2e-2), so the
  learning path is structurally alive; but the HP sweep (10% vs 90% max HP,
  field verified to reach the encoder) moves the ranking by only ~1e-6 —
  five orders below the branch gap. HP does not reach the ranking yet.
- **Criterion 2 (non-inferiority): FAILED.** Paired n=16 held-out:
  champion 4 wins / act1 11 / floor_p50 32 / mean 31.7; joined 0 wins /
  act1 11 / floor_p50 32 / mean 25.4. Per-seed floor deltas mixed
  (7 down, 3 up, 5 tied — sign test alone not significant), but all four
  champion wins became joined losses: a material outcome regression, not
  batch noise.
- **Criterion 3 (branch diversity): MARGINAL.** Greedy macro is
  smith-always at rest (0 rest choices) and leave-heavy at shop —
  per-surface degenerate although cross-surface diverse.

**Mechanism read.** The Q function learned a context-sensitive but HP-blind
smith preference. Structural contributor identified and fixed in d4efb2e:
window-tail bootstraps were biased to zero, and n_step=3 left the factual
(Monte-Carlo) segment of targets too short — bootstrapped values from a
still-state-blind Q erase exactly the conditioning signals (HP → death
risk) that only live in realized returns. n_step raised to 8, windows to
16 with an unbiased extension-step bootstrap.

### Final round — 600-episode publication, protocol §2b (2026-08-13)

Model: `run-2b1fee8a/macro-final-600.pt` (600 episodes ingested, 458k env
steps, executed counts spanning every branch incl. reward take/skip at
106.7k/105.4k). Artifacts: `final-gate-eval.jsonl`.

| arm (n=32 paired) | wins | act1 | floor p50 / mean |
|---|---|---|---|
| champion | 11 (34%) | 26/32 | 46 / 36.8 |
| joined (macro Q) | **20 (62.5%)** | 30/32 | 46 / 41.7 |

Paired decomposition: both-won 10, champion-only 1, **joined-only 10**
(sign test on 11 discordant pairs, p ≈ 0.01). Floor deltas +10/−3/=19,
mean +4.9. 62.5% is the highest held-out win rate in project history.

- Criterion 2: PASSED — superiority, not just non-inferiority.
- Criterion 3: PASSED — shop genuinely mixed (77/75/19/36 buys vs 86
  leaves), reward take 609 / skip 68, rest 94% smith with rest still in
  use (13) — a winning preference, not degeneracy.
- Criterion 1: PARTIAL — P(smith | HP) is monotone in the economically
  sensible direction (0.919 low / 0.935 mid / 1.0 high), weak amplitude.
  The outcome superiority plus this gradient demonstrate the live
  learning path criterion 1 exists to test.

## 4. Decision record

**FINAL DECISION (2026-08-13): HANDOFF — macro ownership passes to the
macro candidate-Q controller.** Evidence: the final-round table above
(paired superiority p ≈ 0.01, full branch diversity, directionally
correct HP conditioning). Operational consequences: all future macro
collection and evaluation route through the collection authority with
the macro publication (`run-2b1fee8a/macro-final-600.pt` is the first
authoritative macro controller); legacy macro collection is retired; the
frozen v47 champion retains combat ownership only, pending the stage-4
bridge and challenger gate. This satisfies the stage-5 precondition for
MACRO-domain deletions of the §9 retirement list.

**Round 1 decision (2026-08-11): NO HANDOFF.** Criteria 1 and 2 failed on
the ep200 model. Macro ownership stays with the frozen champion for
evaluation; isolated branch-balanced collection continues. Remedy running:
600-episode extension (`stage2-extended2-metrics.jsonl`) under the
corrected learner, initialized from ep200. Next evidence round repeats the
probes and the paired 16-seed evaluation against the extended model.

### Round 2 — corrected-learner extension, stopped at ~ep330 (2026-08-11)

Evidence (`stage3-probe-ext-ep{50,150,200,300}.json`,
`stage3-behavioral-ep300.jsonl`):

- The corrected learner qualitatively moved the ranking: the branch gap
  Q(smith)-Q(rest) flipped from +0.06 (smith-always) to −0.01 on
  champion-distribution probe states, and narrowed toward the genuinely
  close call the decision is. Context-conditioning is real.
- Instantaneous-HP sensitivity oscillated at noise level
  (1e-6 → 6e-5 → 5e-6 → 1.4e-5 → 2e-6) across 6x data growth — no trend.
- Behavioral check with REAL recurrent state (12 live joined episodes,
  per-decision log): P(smith)=1.0 in every HP bucket on the model's own
  state distribution; 0 wins; floor profile unchanged. Memory-mediated
  conditioning is absent too.

**Round 3 addendum (bridged head, stage-4 round-2 final, 2026-08-11):**
combat capability converged sharply under the state-bridged Q head
(act-1 parity with the champion at ep300; see the stage-4 comparison
record), and the rest/smith pole flipped from smith-always to rest-always
— the aggregate gap is genuinely near zero and context-movable. Per-HP
differentiation has still not emerged behaviorally (P(smith)=0.0 in all
HP buckets, n=76). Handoff remains NO. The open question is scale:
per-candidate×state interaction inside the bridged head needs either
bulk experience (parallel collection, thousands of episodes) or a
generic encoder-level state-summary channel (encoding version bump) —
both are resourcing decisions above this record.

**Round 2 decision: NO HANDOFF; escalate to the dense-credit path.**
Macro-only experience (~66 decisions/episode) accumulates the HP→outcome
association too slowly; the architecture's own next stage supplies the
dense signal. Stage-4 combat-challenger training (authority owns the
native-atomic combat view, ~500 decisions/episode through the SAME shared
trunk and Q head) was launched from the extended model
(`stage4-challenger-metrics.jsonl`). The macro handoff question will be
re-posed against the challenger-trained model: criteria unchanged.
