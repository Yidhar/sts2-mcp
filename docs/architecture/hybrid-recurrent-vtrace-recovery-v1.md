# Hybrid recurrent V-trace recovery v1

Status: implemented; intended successor to the hybrid-IQL v3 experiment.

## Why the IQL branch is not the main learner

The hybrid representation reset was useful: factual semantic candidates,
macro/combat front streams, one shared fusion stream, slow run memory and
encounter memory are all retained.  The strict-improvement IQL controller was
not useful as the main online learner.  At the observed operating point only
about four percent of executed rows had positive critic advantage, the actor
gradient was roughly two orders of magnitude below the critic, entropy was
telemetry rather than an optimized term, and complete-run replay made a long
run stale before its first update.

That is the wrong failure mode for a simulator that can supply current online
experience continuously.  The repair changes the learning algorithm, not the
semantic model and not the action-specific preference stack.

## Controller

```text
authoritative semantic decision
        -> macro or combat front stream
        -> shared fusion stream
        -> split run/combat recurrent memory
        -> stochastic legal-candidate policy + scalar state value
```

Training uses recurrent V-trace on short FIFO unrolls.  An unroll is emitted
during a run as soon as its next semantic bootstrap state exists; it is
consumed once and is never put in replay.  The behavior probability is the
exact mixture

```text
mu(a|s) = (1 - epsilon) * pi(a|s)
        + epsilon * uniform_branch_then_target(a|s).
```

Held-out evaluation remains greedy and uses no exploration.  The ordinary
and sparse-support collectors may use different epsilon schedules, but both
record the same exact mixture probability and feed the same V-trace learner.

V-trace recursion follows the full mixed-domain sequence.  After targets are
formed, macro and combat policy/value/entropy rows are each averaged inside
their domain and the two domain objectives are averaged 1:1.  This preserves
one physical reward scale while preventing dense combat traffic from
numerically erasing sparse macro decisions.

The actor is no longer detached from the representation.  Policy, value and
annealed entropy gradients all reach the shared stream.  There is no replay
critic, target network, AWR/CE corridor, liveness head, imitation plane, MCTS,
dynamic death-price controller or automatic performance rollback.

## Reward v9

For revival counts `k_a` in each Act and course budget `B`, the cumulative
death cost is

```text
C = sum_a 0.20 * k_a / (k_a + 1)
  + (0.80 / B) * sum_a k_a.
```

At `B=64`, the first three marginal costs in every Act are approximately
`0.1125`, `0.04583`, and `0.02917`; every later revival still costs at least
`0.0125`.  Entering a new Act resets only the front-loaded ordinal, never the
run-wide tail or costs already paid.  Terminal outcome remains `+2/-2`, with
the existing bounded HP-loss, progress and pace facts.  The price is fixed
inside a course.  Budget changes are explicit course boundaries.

This restores the useful early-death pressure seen in reward-v4 without
restoring its effectively free long tail.  In particular, low HP regains
value at rest sites because an avoided first death in the next Act is no
longer priced like the 64th death of the run.

## Initialization and operation

The first lineage is a model initialization from the healthy hybrid-IQL v1
policy-100 publication:

`E:\game\project\sts2_mcp_artifacts\runtime\hybrid_runs\hybrid-iql-v1-reward-v8-budget64-isolation\run-5f98f3b4-b3c2-48de-84ed-90288022cdb1\publication\network.pt`

Compatible encoder, dual streams, fusion, split memories, policy and value
parameters are inherited exactly.  Domain exploration clocks are inherited.
Optimizer, replay, target-network, IQL and adaptive death-price state are not.

Operational files:

- `packages/rl-agent/config/experiments/hybrid_vtrace_v1_reward_v9_budget64_on_policy.toml`
- `packages/rl-agent/scripts/run_hybrid_vtrace_collector.py`
- `packages/rl-agent/scripts/run_hybrid_vtrace_trainer.py`
- `packages/rl-agent/scripts/run_hybrid_vtrace_evaluation.py`
- `packages/rl-agent/scripts/train_hybrid_vtrace_parallel_wsl_rocm.sh`
- `packages/rl-agent/scripts/run_hybrid_vtrace_parallel_windows.ps1`

The existing hybrid dashboard bridge accepts the new episode/update events and
shows domain policy/value losses, entropy weight, importance ratio, lag and
shared-gradient interaction alongside the held-out journals.

## Scope boundary and the next macro-credit experiment

This lineage restores the recent, entropy-bearing recurrent online learning
path and repairs the death-price geometry.  It does **not** claim to solve
same-state counterfactual credit for long-horizon macro choices.

In particular:

- the V-trace recursion reaches only across one short mixed-domain unroll; it
  is not an episode-wide macro SMDP lambda-return;
- branch-balanced epsilon collection guarantees that a legal branch can be
  observed, but an epsilon-selected action with learned probability near zero
  still has a small target-policy importance ratio; support collection alone
  is therefore not a non-vanishing actor-recovery mechanism;
- one observed run still cannot identify what would have happened if a
  different rest, reward, shop, upgrade or route action had been taken at the
  same root state.

The first run of this lineage is consequently a clean baseline, not the final
macro-credit system.  The next experiment should remain separate: first build
a read-only same-root macro rollout diagnostic, then—only if the measured
ranking/regret gap warrants it—train a compact macro candidate-Q path from
macro SMDP lambda-returns and grouped same-root simulator returns.  Results are
normalized per semantic root and branch, and are distilled into the existing
macro policy; no operation-specific rule, CE corridor or inference-time MCTS
is added to this baseline.
