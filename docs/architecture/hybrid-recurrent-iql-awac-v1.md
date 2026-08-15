# Hybrid Recurrent IQL/AWAC v1

Status: implementation and short-isolation validation.

This document supersedes the **control objective** in
`hybrid-domain-fusion-candidate-q-v1.md`. The factual graph, semantic actions,
domain adapters, split memory and balanced replay remain unchanged. The old
Double-Q learner is retained only as the frozen comparison that produced the
failure; it is not trained beside this learner.

## 1. Production evidence that triggered the predefined fallback

The fallback condition in `semantic-decision-graph-reset-v1.md` §8.2 was met.
In the reward-v8 budget-64 lineage, the healthy policy at update 198 reached
Act 1 on 15/16 fixed seeds and won 9/16. By update 250, an unsupported ranking
flip made `end_turn` greedy in full-energy combat states:

- at policy 198, only 8 of 1,873 `end_turn` decisions still had a playable-card
  candidate;
- at policy 250, 3,591 of 3,711 `end_turn` decisions still had one;
- the mean top-one/top-two Q margin fell from about 0.0084 to 0.0022;
- the fixed-seed gate fell to Act 1 1/16 and wins 0/16.

The learner regressed only the executed candidate and bootstrapped through an
online argmax. Correct zero-energy `end_turn` samples could therefore move the
same scorer in full-energy states while the unexecuted card candidates had no
within-state ranking derivative. This is not repaired by a larger model or by
another operation-specific CE rule.

The same failed run later produced non-finite losses at update 270 and
published non-finite weights at update 271. The two collectors then failed to
map the selected score to a native legal action. Finite loss/gradient/model
boundaries are now explicit so a numerical failure stops before publication.

The first isolation run later observed one non-finite forward at update 16.
The saved model, optimizer, replay facts and pending runs were all finite, and
the same checkpoint/RNG/batch completed successfully on three exact replays.
This identifies a transient ROCm forward fault rather than a deterministic
learning target. The learner therefore recomputes the **same immutable batch
once**, before backward or optimizer mutation. A second failure remains fatal
and reports the exact domain, surface, branch and first non-finite quantity.
The retry count is telemetry, not another policy or reward mechanism.

## 2. Permanent control equations

For one factual transition and its actual semantic successor:

```text
Q target: y = r - lambda * delta_revivals + gamma * V_target(s')
Q loss:   Huber(Q(s, a_executed), y)

V error:  e = Q_target(s, a_executed) - V(s)
V loss:   |expectile - 1[e < 0]| * e^2

weight:   w = min(w_max, exp((Q_target(s,a) - V_target(s)) / temperature))
actor:    -w * log pi(a_executed | s, legal_candidates)
```

There is no candidate max in the target. An unexecuted optimistic candidate
cannot raise the predecessor target. The actor receives a direct CE derivative
for every executed action, including one whose current probability is tiny;
there is no importance-ratio multiplier that makes this derivative disappear.

This is one standard offline-to-online control algorithm, not three policy
sidecars. Q, V and the AWAC actor are the coupled components of recurrent IQL.
The old Double-Q target is removed rather than combined with it.

## 3. Representation and optimization ownership

The topology remains:

```text
grounded factual tokens
        -> macro or combat local adapter
        -> slow run strategy token + shallow shared core
        -> candidate Q, state V, candidate policy
```

Q and V train the factual/fusion representation. The actor consumes detached
critic features, so a large cross-entropy scale cannot drag the shared trunk.
The actor head still tracks the changing critic representation through its own
updates. Critic and actor gradients are clipped separately.

Macro and combat losses are reduced independently and then averaged 1:1. This
preserves equal optimizer opportunity without changing physical reward units.
Replay still samples one macro and one combat view from a uniformly sampled
run; run/encounter recurrent state is reconstructed from factual prefixes under
current weights.

## 4. Behavior and exploration

Collection and greedy held-out evaluation use `policy_logits`, never Q argmax.
Branch-balanced macro/combat epsilon remains collection-only. It provides
support; it does not define learned preference. Legal masks remain the only
action suppressor.

There are no special rules for `end_turn`, Rest, Smith, Skip, shop Leave or card
removal. Their ranking must come from factual executed trajectories and the
shared reward.

## 5. Model-init boundary

The first lineage starts from the last healthy finite Candidate-Q publication,
policy 198. It is an explicit model initialization:

- factual encoder, split memory, local adapters, joint core and Q head inherit;
- the new actor head is initialized to the source Q head, so gate-zero greedy
  behavior exactly matches the healthy parent;
- the new value head starts on the same scalar bias and then learns its
  expectile;
- optimizer, replay and learner state start fresh;
- the inherited absolute death price is retained because the reward identity
  is unchanged.

The collapsed policy 250 and the non-finite live publication are diagnostic
comparisons only.

## 6. Short-isolation acceptance

Do not start another long lineage until one fixed-seed short run establishes:

1. gate-zero matches the healthy parent within ordinary evaluation variance;
2. no non-finite loss, gradient, parameter or publication;
3. full-energy `end_turn` does not become the dominant combat action;
4. actor entropy and advantage weights remain finite and non-degenerate;
5. Act-1 and run success do not show the policy-198 to policy-250 cliff;
6. Rest/Smith, Take/Skip and Buy/Leave probabilities remain visible in the
   existing readable journal rather than being replaced by hand rules.

No MCTS or look-ahead search is introduced. The purpose of this change is to
repair the supported-action learning target while keeping the factual online
RL system model-free.
