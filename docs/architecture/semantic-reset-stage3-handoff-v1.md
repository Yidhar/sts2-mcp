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

## 4. Decision record

**Round 1 decision (2026-08-11): NO HANDOFF.** Criteria 1 and 2 failed on
the ep200 model. Macro ownership stays with the frozen champion for
evaluation; isolated branch-balanced collection continues. Remedy running:
600-episode extension (`stage2-extended2-metrics.jsonl`) under the
corrected learner, initialized from ep200. Next evidence round repeats the
probes and the paired 16-seed evaluation against the extended model.
