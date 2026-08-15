# Absolute Death Price, Dual Ratchet, and Revival-Budget Annealing v1

Status: accepted written case for the first reward successor to hybrid
Candidate-Q v1.

This document changes the scalar task target, not the optimizer family.  The
controller remains one legal-candidate Double-Q model trained by one Huber
loss.  It adds no Rest/Smith rule, operation-specific CE label, actor loss,
search policy, or preference head.

## 1. Written case

### 1.1 Observed failure

At the 440-run held-out gate, every rest-site decision greedily selected
Smith.  The choice was not confident: after converting the two legal
candidate Q values to a two-way preference, the median was approximately
`Smith 0.505 / Rest 0.495`, corresponding to a Q difference of only `0.02`.
The sign was nevertheless nearly constant, so greedy evaluation turned a
near tie into 100% Smith.

The reward-v7 revival term is:

```text
C_revive(k; B) = 0.40 * min(k / B, 1)
```

At the completion-preheat budget `B=64`, one additional revival therefore
costs `0.00625`.  Rest does not subtract from the monotonic cumulative
`player_hp_lost` fact, and the bounded HP-loss term is already close to
saturation late in a long run.  A permanent card upgrade can consequently be
worth more than restoring enough HP to avoid one or more cheaply priced
revivals.  The observed Q ordering is compatible with the current target; it
is not evidence of a severed derivative.

The same economics explains both symptoms:

- high successful-run revival counts do not approach zero;
- HP has too little continuation value, so Rest cannot reliably outrank
  Smith at low HP.

### 1.2 Why the existing primary target cannot carry the requested signal

Reward v7 couples two different quantities:

- `revival_budget` is an environment continuation/exploration limit;
- `0.40 / revival_budget` is the learned price of one death.

Increasing the budget to expose full-run completion therefore makes every
death cheaper.  More data, a larger network, a fixed-seed teacher, or another
policy loss cannot make Candidate-Q prefer a behavior that its factual scalar
target values less.  Adding such a channel would recreate the conflicting
preference stack removed by the hybrid reset.

The repair belongs inside the single Bellman target:

```text
r_effective = r_task_without_revival - lambda * revivals_delta
```

`lambda` is an absolute price and never divides by the current revival
budget.  Replay stores the factual `revivals_delta`; the learner applies the
current price when constructing a target.

### 1.3 Why this is not a second loss

The only optimization objective remains:

```text
Huber(Q_online(s, executed_legal_action), DoubleQ_target)
```

The price changes what return the model is asked to predict.  It does not add
a preference regularizer, action label, imitation path, or extra model head.

## 2. Three-part control design

### 2.1 Absolute death price

Within course `s`:

```text
r_t^s = r_task_v5,t + r_hp,t + r_pace,t - lambda_s * delta_revivals_t
```

Properties:

- `lambda_s >= 0` has the same units at budgets 64, 32, 16, and below;
- every revival has the same immediate scalar price inside an update;
- there is no cumulative revival cap in the reward function;
- `revival_budget` remains an environment-dynamics fact, not a reward
  denominator;
- the HP-loss and pace facts remain small secondary tie-breakers; neither is
  treated as a substitute for death price.

### 2.2 Monotone dual ratchet

The price is updated from successful prefixes only.  Failed runs that end
early with few revivals must not be interpreted as efficient play.

The controller first uses the dense prefix statistic:

```text
K_act1 | act1_cleared
```

and, once complete-run successes have adequate support, switches to:

```text
K_run | run_success
```

For the active statistic:

```text
lambda_next = min(
    lambda_max_s,
    lambda + eta_lambda_s * max(0, EMA(K | success) - T_s),
)
```

`lambda` never decreases inside a course.  The target `T_s` is the only
release/ratchet schedule.  This avoids two simultaneous control semantics
where a falling target and a falling price can cancel one another.

### 2.3 Revival-budget annealing

The final zero-revival constraint is reached by environment courses:

```text
64 -> 32 -> 16 -> 8 -> 4 -> 2 -> 0
```

Changing `B` changes transition dynamics and starts a new course with fresh
replay.  Network parameters may initialize the next course, but the handoff is
not an exact continuation.  `lambda` and its target evolve continuously
inside a fixed-budget course without clearing factual replay.

The wide budget is therefore used to expose successful full trajectories;
the absolute price gives HP value while that budget is wide; budget annealing
enforces the final hard constraint after competence is retained.

## 3. Success-support freeze gate

Success conditioning can otherwise create a selection ratchet:

```text
lambda rises
  -> risky wins disappear
  -> remaining wins have lower K
  -> conditional EMA falls while total win rate collapses
```

Price increases are allowed only when both support conditions hold:

1. the most recent `N=50` ingested training runs contain at least `m=8`
   successes for the active statistic (`act1_cleared` during the prefix
   phase, `run_success` during the run phase);
2. the latest completed held-out gate win rate for the active outcome is at
   least `0.5 * course_baseline_win_rate`.

If either condition fails, `lambda` is frozen.  It is not decreased.  A new
held-out result may reopen the gate; absence of a held-out result never counts
as positive evidence.  One completed held-out policy version can authorize at
most one supported 50-run control block.  Reusing the same gate for a later
block freezes the price with `stale_heldout_gate`.  The launcher therefore
freezes and evaluates the current publication every 50 ingested runs, with at
most one evaluator in flight.  A slow or failed evaluator does not stop task
learning; it only prevents another price increase until fresh evidence is
complete.  The baseline is the fixed gate-zero measurement for that course
and uses the same seed set as subsequent gates.

The controller updates only when a complete new block of 50 training runs is
available.  Overlapping per-run updates would overweight almost identical
windows and make the nominal step size depend on ingestion frequency.

## 4. Target ladder

Each fixed-budget course owns a source-versioned schedule rather than CLI
coefficients.  The first implementation uses this conservative ladder:

| Course budget B | Prefix target `T_act1` | Run target `T_run` | Initial lambda | Lambda max | Eta per 50-run block |
|---:|---:|---:|---:|---:|---:|
| 64 | 6 | 32 | 0.00625 | 0.030 | 0.00025 |
| 32 | 4 | 16 | inherited, capped | 0.040 | 0.00025 |
| 16 | 2 | 8 | inherited, capped | 0.055 | 0.00025 |
| 8 | 1 | 4 | inherited, capped | 0.075 | 0.00020 |
| 4 | 1 | 2 | inherited, capped | 0.100 | 0.00015 |
| 2 | 0 | 1 | inherited, capped | 0.140 | 0.00010 |
| 0 | disabled | 0 | no revival transition exists | disabled | disabled |

The numbers are an initial control law, not a claim that they are globally
optimal.  They are deliberately slow: for budget 64 an excess of eight
revivals raises the unit price by `0.002` per supported 50-run block.  The
price cannot jump from completion preheat to a survival-dominant target in a
single evaluation interval.

The prefix-to-run switch occurs after both are true:

- two consecutive supported 50-run blocks have `EMA(K_act1 | act1_cleared)
  <= T_act1`;
- the last 50 training runs contain at least eight full-run successes.

After the switch, the controller does not switch back.  A loss of success
support freezes price movement instead.

Moving to the next target or budget is an operator-visible curriculum choice
based on held-out completion retention and revival reduction; it is not an
automatic rollback or hidden behavior rule.

## 5. Discount-space outcome ordering

The semantic clock discounts only durable floor progress with base `0.997`.
From the first floor of a roughly 60-floor run, terminal credit is multiplied
by approximately:

```text
d_min = 0.997 ** 59 ~= 0.84
```

Revival cost is charged when the death occurs and can therefore be much less
discounted than the final outcome.  Reward bounds must consequently be
checked in Q space, not by comparing undiscounted scalar constants.

Reward v8 raises terminal success/failure to `+2 / -2`.  Its terminal margin
is `4`, whose conservative earliest-decision value is about `3.36`.  For the
budget-64 course:

```text
maximum revival span = 64 * 0.030 = 1.92
HP-loss span          = 0.55
pace span             = 0.04
secondary span        = 2.51
```

The required invariant is:

```text
discounted terminal margin
    >= 1.25 * maximum same-course secondary span
```

For budget 64, `3.36 >= 1.25 * 2.51 ~= 3.14`.  Every later course has a
smaller `B * lambda_max` product and satisfies the same bound.  Tests evaluate
this formula from the versioned course table rather than assuming an
undiscounted `0.99` cap is sufficient.

The progress term is common to both outcome arms near the same terminal
prefix and is not used to hide a failed inequality.  A course whose exact
reward bounds violate the inequality is a design error, not a reason to clip
individual revival events.

## 6. One-shot Q-scale calibration

Reward v8 changes the target distribution twice: terminal outcomes move from
`+/-1` to `+/-2`, and the absolute death price begins to move.  A newly
initialized Q head can therefore have a common offset far from empirical
returns.  Executed actions then regress toward truth while unexecuted actions
retain the old offset, causing greedy action selection to chase unsupported
optimistic candidates.

Before the first gradient update of a new reward-v8 lineage, after factual
replay contains at least one learner batch:

1. compute empirical discounted returns from stored factual edges using the
   current `lambda`;
2. compute the mean legal Q over a bounded sample of replay states;
3. add `mean_return - mean_legal_q` to the final Candidate-Q bias of both
   online and target networks;
4. record the sample counts, both means, and applied shift;
5. checkpoint/publicize the calibrated state before ordinary learning.

The calibration runs exactly once for a model-init reward-v8 lineage.  Exact
resume restores the calibrated flag and never repeats it.  A budget-course
model-init runs it again because both dynamics and replay are new.  Continuous
lambda changes inside one course do not rerun calibration.

Runs ingested while replay is still too small for calibration accumulate an
explicit learner-update debt.  After the calibrated state is durably
published, those updates are executed before the debt is cleared; the first
few factual runs are not silently converted into replay-only data.  Calibration
temporarily evaluates both networks with stochastic layers disabled and then
restores their prior train/eval modes.

This is a common scalar offset calibration, not an action preference and not
a learned loss.

## 7. Factual replay and version boundaries

`HybridStep` transition v2 stores:

```text
task_reward_without_revival
revivals_delta
discount
terminal
... existing semantic decision facts
```

The collector derives `revivals_delta` from the same monotonic simulator
counter already audited by reward v7.  A transition with a decreasing or
non-integral counter is invalid.  The learner constructs every sampled edge
with one lambda snapshot per optimizer update, so all items in that update use
the same price.

Consequences:

- continuous lambda or target movement within a course does not invalidate or
  clear replay;
- exact resume restores lambda, EMAs, the active statistic, window contents,
  gate state, target ladder position, and calibration state;
- changing reward identity from v7 to v8 requires a new model-init lineage and
  fresh replay because v1 transitions contain only an already-composed scalar
  reward;
- changing revival budget starts a new course and fresh replay because it
  changes environment dynamics;
- old v1 transitions are not silently interpreted as v2.

## 8. Ownership and implementation surface

| Owner | Contract change |
|---|---|
| `packages/rl-agent/sts2_baseline/objective.py` | task reward v5 with terminal `+2/-2`; survival reward v8 removes `0.40/B` revival pricing and exports the exact revival delta fact |
| `packages/rl-agent/sts2_rl/joint/transitions.py` | `HybridStep` v2 adds factual `revivals_delta` and a reward explicitly excluding learner-priced revival cost |
| `packages/rl-agent/sts2_rl/joint/learner.py` | learner v3 owns the monotone lambda controller, successful-prefix/run EMAs, support freeze, target composition, checkpoint state, and metrics |
| `packages/rl-agent/scripts/run_hybrid_trainer.py` | ingests episode outcomes into the controller, receives held-out gate results, performs one-shot Q calibration, and emits controller telemetry |
| source-versioned course table | owns `B`, targets, eta, lambda maxima, support window, and held-out floor; these are not free CLI tuning flags |

The simulator/bridge continues to emit facts and enforce the selected revival
budget.  It does not own lambda or reward policy.

## 9. Acceptance probes

The reward-v8 smoke and first held-out gates must report these three symptom
families without adding a rollback rubric:

1. **survival conditioning:** `P(Rest | HP <= 50%)`, Rest/Smith Q margin by HP
   band and Act, and successful-run/Act revival counts;
2. **macro competence preservation:** card-removal base-card hit rate and the
   existing Take/Skip, Buy/Leave and upgrade-target probes;
3. **run economics:** revivals per reached floor, `K_act1 | act1_cleared`,
   `K_run | run_success`, unconditional completion rate, lambda, target,
   support counts, and freeze reason.

The expected first change is not "always Rest".  Low-HP and near-boss states
should cross the Rest/Smith Q boundary before high-HP states, while full-run
completion remains supported.  A lower unconditional revival average caused
only by earlier failure is explicitly not success.

## 10. Not included

- no fixed-seed behavior cloning or hand-authored Rest threshold;
- no MCTS or look-ahead controller;
- no new value, actor, cost, imitation, CE, AWR, or corridor head;
- no per-operation exploration floor;
- no replay clear for an ordinary lambda update;
- no automatic budget rollback mechanism.

Fixed seeds remain useful for paired counterfactual calibration: replay the
same rest-site prefix through Rest and each legal Smith target, estimate the
change in subsequent revival count, and verify that the selected absolute
price flips low-HP states without turning healthy states into unconditional
Rest.  They are evidence for pricing, not a second policy.
