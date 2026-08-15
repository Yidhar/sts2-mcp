# Hybrid Domain Fusion Candidate-Q v1

Status: factual/model topology retained; the Double-Q control objective was
retired after the production fallback condition fired. Its active successor is
`hybrid-recurrent-iql-awac-v1.md`.

This document supersedes the independent-control-trunk ownership described in
`semantic-decision-graph-reset-v1.md` §7 for the new hybrid lineage.  The old
Stage-2 implementation remains a comparison/migration path; it is not the
target training topology.

## 1. Why this reset exists

The legacy actor learned combat and rare run-level choices through eleven
partly overlapping policy channels.  A legal action whose softmax probability
reached zero also lost V-trace/importance/entropy gradient.  Each newly found
absorbing action then acquired its own CE corridor, AWR bridge, exception or
guard.  More capacity could not repair a severed derivative.

The reset changes the primitive rather than adding a twelfth rescue loss:

```text
Q(state, executed legal semantic candidate) -> factual Double-Q target
```

An executed non-greedy action receives a direct Huber derivative regardless
of its previous rank or probability. Exploration remains necessary to execute
an unsupported action, but it no longer needs an operation-specific learning
channel afterward.

## 2. Model

```text
authoritative observation
        |
shared factual entity / relation / candidate embeddings
        |
        +----------------------------+
        |                            |
slow run strategy token         domain-local fast adapter
(persistent run memory)         macro adapter | combat adapter
        |                            |
        +--------- shared shallow joint core --------+
                                                     |
                                        one Candidate-Q readout
```

This is not literal flow matching. Only one domain adapter is active for a
state. The useful analogy is two specialized front ends feeding a limited
shared relational core, while a persistent strategy token carries the slow
run context into combat. Run memory writes only on macro facts; combat memory
writes only in combat and is cleared on macro facts.

There is no policy/value/liveness/completion/AWR head in this controller. The
readout sees joint candidate, joint state and split memory directly.

## 3. One factual graph, two views

A physical run is stored once as ordered meaningful semantic decisions. UI
picker suffixes are mechanical executor actions and do not become Q nodes.

Macro view:

- prefix: all earlier macro nodes, rebuilt under current parameters;
- suffix: K macro nodes, counted in macro nodes rather than raw actions;
- edge: exact accumulated reward and discount product through intervening
  combat until the next macro node.

Combat view:

- prefix: all earlier macro nodes plus earlier combat nodes in the current
  encounter;
- suffix: K combat nodes in that encounter;
- boundary: bootstrap to the real next macro node, never a recorded scalar
  bridge.

Actor hidden state is not replay truth. It changes when network parameters
change. Replay therefore stores factual encoded decisions and reconstructs
slow/encounter memory under current online and target networks.

## 4. One objective and balanced optimization

Each update samples a run uniformly and returns one macro and one combat view.
The loss is:

```text
L = 0.5 * mean(Huber TD over macro view)
  + 0.5 * mean(Huber TD over combat view)
```

This is domain-normalized optimization, not macro reward multiplication.
Reward units and clocks remain physical and shared, so cross-domain bootstrap
is meaningful. The shared factual/joint trunk uses a lower learning rate than
domain-local adapters and the fresh Q readout. Huber bounds individual target
influence. Sample reuse and replay age are diagnostics because a fixed 1:1
ratio must not become unlimited over-training of a small macro corpus.

## 5. Exploration

Collection uses branch-balanced epsilon-greedy: choose a semantic branch,
then a target within it. Flat candidate-uniform exploration would reintroduce
candidate-cardinality bias. Macro and combat have independent decision clocks;
mechanical actions advance neither. Held-out evaluation sets both to zero.

No action-specific preference is injected for Rest, Smith, Remove, Skip or
LeaveShop. Their only learned preference comes from factual Q targets.

## 6. ByteRL: selected lessons, not a template

ByteDance's Hearthstone agent is a useful comparison because it also combines
stage-specialized construction/battle paths, shared card semantics, complete
cross-stage trajectories, entity action scoring and legal masks. It did not
use look-ahead search or MCTS. Its most relevant systems result is that
reducing producer/consumer lag and keeping sample reuse low improved results;
blindly adding actors or stale replay did not.

Primary references:

- [Mastering Strategy Card Game (ByteRL, arXiv:2303.05197)](https://arxiv.org/abs/2303.05197)
- [ByteRL's official 2022 competition inference submission](https://github.com/acatai/Strategy-Card-Game-AI-Competition/tree/master/contest-2022-08-COG/ByteRL)
- [Learning to Play the LCM Card Game (the earlier LoCM environment work)](https://arxiv.org/abs/2303.04096)

We deliberately do not copy its PPO/UPGO/V-trace stack, OSFP/self-play,
Hearthstone type-target taxonomy, hero isolation, or compute recipe. STS2 is
single-player, has a much longer semantic horizon, and already exposes atomic
legal candidates. Candidate-Q plus factual run views is our design.

## 7. Original mechanism set and successor boundary

- factual grounded embeddings and relation bindings;
- atomic semantic legal candidates;
- strict-equivalent candidate merging;
- branch-balanced domain exploration;
- slow run memory and encounter-local combat memory;
- shallow shared joint core;
- online/target Double-Q (retired from active control after unsupported
  candidate peaks appeared; see the IQL/AWAC successor);
- Huber loss and target network (retained for Q regression in the successor);
- run-balanced macro/combat replay views;
- one scalar reward contract.

Not retained: success imitation actor, act-prefix actor, completion CE, AWR
bridge, support corridors, risk/liveness/cost actor, scalar domain bridge,
operation-specific exploration floors, or MCTS.

## 8. Acceptance evidence

Before a long lineage, require only evidence tied to this architecture:

1. a low-ranked executed action receives non-zero Q gradient;
2. candidate permutation equivariance and candidate-independent world facts;
3. run/combat memory lifecycle and current-network prefix reconstruction;
4. macro event edge includes intervening factual combat reward;
5. domain losses are averaged separately before the 1:1 combine;
6. invalid candidates cannot win masked Double-Q argmax;
7. replay reports unique runs and view reuse;
8. paired held-out seeds report run/Act performance plus Rest/Smith,
   Take/Skip, Buy/Leave and Remove/Keep conditional Q behavior.

Only after this loop learns should width/depth scaling be tested. Scaling
width or depth before validating the factual graph and optimization share
would merely fit the wrong target faster.

## 9. Operational entry points

- model/learner config:
  `packages/rl-agent/config/experiments/hybrid_candidate_q_v1.toml`;
- GPU learner and publication loop:
  `packages/rl-agent/scripts/run_hybrid_trainer.py`;
- complete-run collector:
  `packages/rl-agent/scripts/run_hybrid_collector.py`.

The trainer starts a **new model-init lineage** from a grounded champion. It
inherits only shape-compatible factual encoder and split-memory parameters;
the two local adapters, shared joint block and Q readout start fresh. Its own
checkpoint carries online/target networks, optimizer, replay, random states,
domain counters and learner update number for exact continuation. Collector
publications carry model weights, policy version and the two domain clocks.

The initial operating point is deliberately modest: 128 complete runs in
replay, at most eight samples per domain anchor, two updates per new run,
8-node macro suffixes and 32-node encounter-local combat suffixes. Add actors
only when replay age and policy lag show that the learner can consume fresher
data; actor count is not a substitute for freshness.

## 10. Implementation validation (2026-08-13)

The landed v1 model has 4,807,873 trainable parameters at the default 128-wide
configuration. A real headless-simulator smoke run exercised reward claims,
standalone macro selection, map/event and combat surfaces without invalidating
the semantic graph: 214 environment steps became 207 learned semantic nodes
(18 macro, 189 combat), with two composite picker suffixes kept mechanical.

The resulting run was consumed on the ROCm GPU by one complete learner update:
8 macro and 32 combat nodes contributed separately averaged Huber losses,
then one optimizer step. The checkpoint retained the factual 214-step count,
both domain clocks, online/target models, optimizer and replay. Model-init
publication and exact continuation were also exercised with the same model
configuration. These are transport/learning-path validations, not a gameplay
performance claim; held-out seed learning remains the next experiment.
