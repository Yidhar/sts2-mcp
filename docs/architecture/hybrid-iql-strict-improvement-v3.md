# Hybrid IQL strict-improvement actor v3

Status: implementation and paired short-run validation candidate.

This document supersedes the actor-weight and collection-role portions of
`hybrid-iql-factual-support-v2.md`.  Reward v8, the factual mixed run, balanced
macro/combat replay, expectile value critic and smooth target network remain
unchanged.  No second policy objective is added.

## 1. Evidence from the v2 isolation

The policy-100 parent won 6/8 paired held-out seeds.  Under v2 factual support,
the same set moved through 6/8, 4/8, 4/8 and 2/8; the final 16-seed audit won
2/16.  Mean floor and Act-3 reach declined while deadlock rose.

The intended support path itself worked: 44% of eligible macro decisions were
exploratory and replay sampled Rest, Smith, reward Skip and shop Leave anchors.
The contradiction was in the actor equation.  When Q and V were unresolved,
plain AWAC assigned `exp(0) = 1`, so a support-executed action became a full CE
label.  Mean actor weight rose toward 0.92 while the policy head continued to
move after Q and V losses had largely settled.  Factual coverage was therefore
not preference-neutral in the implemented algorithm.

The model-init also rewound domain decision clocks.  Combat epsilon returned
from its mature value to the schedule maximum, while both collectors enabled
the macro coverage floor.  Only 4/50 training runs won, making the implicit
behavior-cloning corpus much worse than the parent policy.

## 2. One actor objective with a zero neutral baseline

The critic equations stay unchanged.  Let

```text
A = (Q_target(s, a_executed) - V_target(s)) / temperature
w = min(w_max, max(0, exp(A) - 1))
actor = -w * log pi(a_executed | s, legal_candidates)
```

The implementation uses `expm1` and clamps before exponentiation.  The
resulting semantics are:

- negative or unresolved advantage: critic/value training only;
- strictly positive advantage: direct bounded policy CE;
- larger positive advantage: smoothly larger policy-improvement weight.

This replaces the old AWAC weight; it does not run beside it.  It has no
surface, action, card, event or domain exception.  The existing detached policy
features remain: actor CE changes only the policy head and cannot train the
critic representation.

## 3. Two collection roles, one factual replay

Parallel collectors alternate roles:

- `exploitation`: ordinary branch-balanced domain epsilon only;
- `support`: the same domain epsilon plus the generic multi-branch macro
  factual-coverage schedule.

Both produce immutable factual `HybridRun` objects for the same replay and the
same Q/V/actor algorithm.  There is no role-specific reward or loss.  The split
keeps complete successful trajectories available while still measuring rare
macro outcomes.

## 4. Model-init schedule continuity

The new lineage explicitly model-initializes network weights from the healthy
policy-100 IQL publication and starts fresh optimizer, replay, learner updates,
policy version and RNG.  It carries only the parent's macro/combat decision
counts because those counts locate the behavior exploration curriculum.  They
do not claim optimizer or replay continuation.

The inherited reward-v8 death price is retained because reward identity and
budget are unchanged.

## 5. Evidence to record

The short run records, without automatic rollback:

1. paired held-out success at policy 0 and later policies;
2. exploitation/support run outcomes separately;
3. macro/combat positive-advantage fractions and effective actor weights;
4. policy, Q and value losses plus actor/critic gradient norms;
5. Rest, Skip and shop Leave probabilities in readable journals.

More runs are not evidence of recovery if the paired held-out policy continues
to decline.  No MCTS, planner, operation-specific reward or rescue CE is part
of this design.
