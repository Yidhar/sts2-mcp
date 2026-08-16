# Hybrid IQL factual support v2

Status: implementation and short-isolation candidate.

This document amends `hybrid-recurrent-iql-awac-v1.md`.  It does not add a
second policy objective.  Q, expectile V and advantage-weighted actor CE remain
the one recurrent IQL/AWAC algorithm.  Reward v8 and the absolute death-price
controller are unchanged.

## 1. Evidence

The first 100-run IQL isolation proved that the optimizer and recurrent model
were active but exposed a sparse-action support failure inside the macro
domain:

- held-out `P(rest)` fell from about 0.484 at policy 0 to 0.038 at policy 100;
- all 38 policy-100 rest-site decisions selected Smith, including states below
  25% HP;
- held-out `P(skip_card_reward)` fell from about 0.244 to 0.022, with no skip;
- shop Leave while purchases remained legal fell toward the same corner;
- successful-run mean revivals rose from 36 at policy 25 to 56 at policy 100;
- critic losses and TD error nevertheless declined, and deadlock fell to zero.

The training corpus contained only 14 factual Rest choices against 115 Smith
choices.  IQL regresses executed actions and its actor fits executed actions.
It can recover a low-probability action once that action is executed, but it
cannot manufacture the missing alternative outcome.  Equal macro/combat loss
means do not balance Rest against the much denser map, reward and event corpus.

## 2. One correction at each factual boundary

### 2.1 Collection: multi-branch factual support

During training collection only, a macro semantic surface with more than one
distinct branch receives a branch-balanced epsilon floor:

```text
epsilon_coverage: 0.50 -> 0.20 over 50,000 macro decisions
```

The existing sampler first chooses a branch uniformly and then a candidate in
that branch.  Consequently Smith's many card targets do not dilute Rest, and a
shop inventory's many item rows do not erase Leave.  The rule is generic: it
does not name Rest, Smith, Skip, Leave, a card, or an item, and supplies no
positive label.  Single-branch map/event decisions keep the ordinary domain
epsilon.  Mechanical selection surfaces are excluded because their parent
atomic action already owns the preference.

Held-out evaluation always uses epsilon zero and bypasses this schedule.

### 2.2 Replay: surface/branch anchor balance and freshness

Replay continues to choose complete runs uniformly and returns one macro plus
one combat view.  Inside each chosen run, the macro anchor now rotates through
the least-sampled `surface:branch` bucket before reusing a dense bucket.  View
reuse remains bounded.  This changes sampling opportunity, not reward units or
per-node loss coefficients.

An anchor more than 32 learner versions behind the current online policy is
not eligible.  Old immutable runs may remain stored for diagnostics, but they
cannot occupy the active learner batch indefinitely.  The dashboard reports
stale macro/combat anchors and the sampled macro-anchor distribution.

### 2.3 Target: smooth update

The target model is updated after every optimizer step:

```text
theta_target <- 0.98 * theta_target + 0.02 * theta_online
```

This replaces the 25-update hard copy that produced periodic actor/critic
shocks.  It is optimization hygiene, not a substitute for factual branch
support.

## 3. What is deliberately not added

- no Rest/Smith, Take/Skip, Buy/Leave reward rule;
- no completion corridor, policy floor, AWR side bridge, or rescue CE;
- no MCTS or inference-time search;
- no replay reinterpretation of one physical run under several rewards;
- no immediate death-price acceleration in the same isolation.

Paired simulator rollouts from the same semantic snapshot remain a possible
second experiment if ordinary branch-balanced collection still cannot resolve
the alternative values.  They are not part of v2, so this run can identify
whether factual coverage alone closes the gap.

## 4. Lineage boundary and evidence

Replay v5, learner v2 and the smooth-target state are a new training contract.
The first run therefore model-initializes network weights from the frozen
policy-100 IQL publication, inherits its reward-v8 death price, and starts a
fresh optimizer/replay lineage.  It must not claim exact continuation of the
old replay.

The short isolation should answer four direct questions rather than rely on a
single aggregate loss:

1. do training journals contain factual Rest, Skip and shop Leave choices;
2. does replay telemetry show those `surface:branch` anchors being sampled;
3. does low-HP held-out Rest probability stop its monotonic collapse and begin
   to separate from healthy-HP states;
4. do success revivals stop rising without reviving deadlock or the combat
   `end_turn` regression.

Only after this evidence is visible should the death-price controller be made
faster or the run extended into a long course.
