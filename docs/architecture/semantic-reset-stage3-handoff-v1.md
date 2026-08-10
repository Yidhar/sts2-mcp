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

(To be filled from `stage3-joined-eval.jsonl` and
`stage3-state-variation.json` after the stage-2 training run completes.)

## 4. Decision record

(To be filled: handoff yes/no, date, evidence hashes, and the operational
consequence — which collection entry points switch to authority ownership.)
