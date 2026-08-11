# CLAUDE.md rewrite draft (EC-5) — staged for stage-5 deletion

Status: draft prepared 2026-08-11 per EC-5 ("drafted at Stage 1 so deletion
is a paste, not a negotiation"). Applied to
`packages/rl-agent/CLAUDE.md` ONLY in the same commit series that executes
the corresponding §9 deletion; nothing here is active before that.

## 1. REWRITE at semantic-authority handoff

Replace the invariant bullet
"No MCTS, planner, root bias, action rewrite or policy distillation is
allowed in the baseline." with:

> - No MCTS, planner, root bias or policy distillation is allowed. The
>   semantic executor executes compiled semantic decisions: UI mechanics
>   (selection traffic, confirmation, pickers) are mechanical suffixes of a
>   chosen semantic candidate, never independently optimized actions.
>   Strategic choice is never rewritten below the semantic layer; the
>   authoritative legality mask remains the only suppressor of semantic
>   candidates.

## 2. DELETE with the corresponding §9 deletion commit

- The two-sided legal-policy corridor paragraph ("A verified
  upgrade/removal lifecycle may restore numerical support ...") —
  superseded by candidate-Q learning, whose executed-action regression
  gradient is rank-independent (no absorption state to rescue).
- The transaction-liveness policy-logit obligations ("Transaction liveness
  must also optimize normalized policy logits directly ...") — the
  liveness/completion policy channels retire with §9 item 6; progress,
  stall and cycle detection remain as episode termination and diagnostic
  facts only.
- The bounded-sidecar sentence fragments naming failure-credit POLICY
  replay obligations; the factual-evidence storage wording stays.

## 3. RE-HOME (wording survives, owner changes)

- Forced singletons: "Forced singleton actions generate no policy target or
  policy-gradient term" survives verbatim — under the semantic controller a
  singleton is a mechanical suffix, never a semantic decision; in Q
  learning it may still receive a value label.
- Revival-cost invariants: success-conditional cost labeling wording
  migrates into the objective (v8) description; the engine-bailout facility
  invariants are untouched.
- The macro decision clock: add one bullet under runtime ownership:

> - Macro credit uses the durable floor clock (Gamma = 0.997^max(dfloor,0),
>   terminal cuts to zero); combat decisions within a floor are
>   undiscounted nodes of the same decision graph.

## 4. Standing anti-accretion rule (added at stage-5)

> - One domain, one primary preference-learning objective. A proposed
>   second policy-shaping channel for a domain (imitation, corridor, CE,
>   actor bridge, exploration floor) is an architecture change and requires
>   a written case for why the primary objective cannot carry the signal —
>   not a patch lineage.

## 5. Citation discipline

Every deletion commit cites: this file, the stage-5 inventory
(`semantic-reset-stage5-deletion-inventory-v1.md`), and the consolidation
audit (workflow wf_20d47fca). No mechanism leaves silently.
