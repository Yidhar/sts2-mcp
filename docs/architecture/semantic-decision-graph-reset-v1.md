# Semantic Decision Graph control reset

Status: architecture decision. This document defines the next control framework;
it is not another experiment recipe layered onto v47.

## 1. Decision

Stop extending the shared softmax actor with operation-specific rescue losses.
The long-term control architecture is:

```text
authoritative raw game events
            |
            v
semantic action compiler / executor
            |
            v
one meaningful-decision graph
   |                         |
   | combat decision        | macro decision
   v                         v
combat recurrent head    macro recurrent candidate-Q
   |                         |
   +------ cross-domain Bellman target ------+
            |
            v
one factual trajectory store
   |                         |
   v                         v
fresh combat view         macro sequence view
```

The first production migration keeps the proven combat V-trace controller as
the champion and replaces the macro control plane with recurrent semantic
candidate Double-Q. A combat candidate-Q controller is developed as a challenger
and replaces V-trace only after it wins direct combat comparisons. This is a
deliberate migration strategy, not a commitment to keep two algorithms forever.

The permanent architectural commitments are independent of that later choice:

1. only meaningful decisions are learned;
2. mechanical UI transactions are executed, not optimized;
3. combat and macro parameters have explicit ownership;
4. both domains are nodes in one Bellman decision graph;
5. factual trajectories are stored once and sampled through domain views;
6. one domain has one primary preference-learning objective;
7. adding a game action adds semantics and an embedding, not a new loss.

## 2. Why the current framework keeps growing

The current model combines a shared entity/candidate encoder and run/combat
memory with a base policy, macro residual, multiple task and cost values,
transaction heads, and liveness heads. The learner then updates overlapping
parameter sets from online V-trace, transaction completion, macro AWR,
episodic imitation, revival credit, and several liveness actor objectives.

The key defect is not merely the number of scalar losses. It is the absence of
a stable ownership rule:

- the shared trunk is moved by objectives with different horizons;
- the macro residual is part of the acting policy and is still affected by
  online and transaction objectives outside macro AWR;
- liveness has a separate backward and clipping path;
- three replay systems reinterpret overlapping factual episodes into different
  policy targets.

The v47 data confirms that model size is not the immediate limit. Macro AWR was
active overall, yet rest, card-reward and upgrade labels were absent while shop
and removal labels were present. The transaction sampler fills a small batch in
fixed operation order, so early operation families can occupy it before later
families are considered. Shop exit and stale card-reward surface detection also
have semantic coverage defects. A wider model cannot learn labels that were
never produced or sampled.

## 3. Why a fixed legal-action probability mixture is not the framework

Let the learned policy be `q = softmax(z)` and mix it with a fixed legal uniform
distribution:

```text
p = (1 - lambda) q + lambda U
```

This makes every legal action executable, but the fixed component does not
depend on `z`. When `q(a)` is already near zero, the derivative of `log p(a)`
with respect to the learned logits still approaches zero. The mixture is useful
as behavior exploration, but it does not restore a trainable preference by
itself. It also does not repair missing semantic transitions, replay starvation,
UI transaction credit or the run-length problem.

A flat candidate-uniform mixture also violates the hierarchical policy. If
branch `b` contains `n_b` of `N` legal candidates, its mixed marginal becomes:

```text
P_mixed(b) = (1 - lambda) * P(b) + lambda * n_b / N
```

The second term favors branches merely because they expose more targets and can
reverse a close branch-level ordering. Exploration therefore remains
branch-balanced: choose a legal semantic branch uniformly, then choose a legal
candidate uniformly inside it. The learned candidate values are not mixed with
a fixed probability component.

The replacement therefore separates exploration from preference learning:

- behavior: one branch-balanced epsilon-greedy rule;
- learning: factual TD error on the executed candidate;
- evaluation: greedy legal candidate value.

An executed action receives a Q gradient regardless of its current greedy rank.

## 4. Meaningful action boundary

The agent decides only when at least one of the following is true:

1. new information has appeared;
2. the choice changes the game state or future legal choices;
3. the choice is an irreversible strategic commitment.

Mechanical entry, selection confirmation and page exit do not meet this test
when all targets were already known at the entry state.

### 4.1 Atomic semantic candidates

The macro action surface should directly expose candidates such as:

```text
Rest
Smith(card_instance)
RemoveCard(card_instance)
Buy(item_instance)
LeaveShop
TakeReward(card_instance)
SkipReward
ChooseMap(node)
ChooseEventOption(option)
```

Policy branches are defined by strategic alternatives, not by raw UI roles or
candidate count. `Rest` and `Smith`, `TakeReward` and `SkipReward`, and `Buy`
and `LeaveShop` are distinct branches. Multiple cards or items populate only
the corresponding target distribution inside a branch. A singleton refusal
therefore keeps an independent exploration share without an operation-specific
entry remap.

The executor translates the selected semantic action into native UI steps. It
does not choose the strategic target. Candidate legality and upgrade/removal
targets come from the engine or simulator, not from a Python preference rule.

This makes `Smith(card)` one action for control and replay even if native
execution needs `choose -> select -> confirm -> proceed`.

### 4.2 Multi-select and changing information

Not every selection screen should be flattened into a giant combination list.

- A single known target is folded into the parent semantic action.
- A set chosen from a known collection uses a monotone set controller:
  `Add(item)` and `CommitSet(current)`. In the compiled control graph,
  `Add(item)` is an irreversible semantic commitment; native deselection and
  cancellation are not exposed when they only return to the same state.
- If an action reveals a random result or a new candidate set, that result starts
  a new meaningful decision. The compiler must not predict it in advance or
  reintroduce raw toggle actions to represent the new choice.

This one rule covers discard, transform and multi-pick flows without creating a
special completion loss for each screen.

## 5. One joint decision graph, not two hidden environments

The macro SMDP abstraction folds only deterministic mechanical execution. It
must not hide an entire combat controlled by a concurrently learning combat
policy.

For example, the decision chain remains visible as:

```text
ChooseMap(node)
    -> combat entry
    -> PlayCard / UsePotion / EndTurn ...
    -> TakeReward(card) or SkipReward
    -> ChooseMap(next_node)
```

Domain heads may be separate, but the target uses the head belonging to the
next meaningful state:

```text
y = r + Gamma * Q_target[next_domain](
        s_next,
        argmax_a Q_online[next_domain](s_next, a)
    )
```

Thus a macro action can bootstrap into a combat entry value, and the final
combat action can bootstrap into the following macro value. Neither controller
treats the other controller's changing policy as an opaque environment option.

The combat V-trace champion uses the same graph through a state-value bridge
during migration. The bridge reads the frozen champion's existing value output
or its corresponding target snapshot. It owns no learner head, supervision,
replay or preference objective; it only adapts the boundary bootstrap to the
next meaningful state.

## 6. Decision clock and primary objective

Native API calls are not a valid time unit. A longer confirmation sequence must
not make forging look worse than resting, and thousands of card plays must not
erase the value of an Act-1 deck decision.

The new trajectory schema records a durable floor clock separately from raw
execution events. For the first implementation:

```text
Gamma_t = 0                              on terminal transitions
Gamma_t = 0.997 ^ max(floor_next - floor_now, 0)  otherwise
```

Consequently:

- mechanical executor steps do not create control transitions at all;
- decisions within one combat or room have `Gamma = 1`;
- advancing one durable floor applies the discount once, regardless of how many
  card plays or UI calls were required;
- terminal states have no bootstrap.

This retains substantial terminal credit across a normal run while preventing
interface length from becoming a strategic cost. Ordinary pace reward and the
existing stall termination handle wasted same-floor actions; discounting is not
used as a loop detector.

The first implementation keeps the existing bounded, outcome-first scalar
objective: completing the run remains more valuable than any possible saving in
HP or revivals, and efficiency differentiates runs only inside that priority.
Under the floor clock, the terminal success/failure gap must remain larger than
the complete bounded secondary-cost range across the supported run length. This
is one ordinary objective calculation before training, not a runtime gate. The
objective must not be supplemented with separate HP, revival, success and
liveness actor losses.

If exact outcome priority later proves impossible to express with the scalar
contract, the clean extension is a two-component value ordered by success first
and efficiency second. It is not a return to multiple competing actors.

## 7. Model and parameter ownership

The first version reuses the factual schemas and initializes from existing
entity/candidate representations, but it does not retain a shared trainable
control trunk.

```text
common factual schema and normalization
       |
       +-- combat encoder adapter -> combat memory -> combat control head
       |
       +-- macro encoder adapter  -> run memory    -> macro candidate-Q
```

The two domain encoders may start from the same learned weights, then train
independently. Static semantic field definitions remain shared. This avoids
recreating gradient arbitration through a shared transformer or embedding table.

Each domain has one primary control owner:

- macro: recurrent candidate Double-Q TD;
- combat during migration: fresh-unroll V-trace only;
- combat challenger: recurrent candidate Double-Q TD.

Outcome prediction may remain for diagnosis or representation experiments, but
it cannot independently modify action preferences in the first version.

## 8. Replay and learning

### 8.1 One factual store, two layers of facts

The store retains:

1. authoritative raw events, so executor behavior can be understood;
2. derived semantic transitions, which are the only transitions used for
   control learning.

Combat and macro samplers are indices over this store, not separate competing
evidence systems.

### 8.2 Macro learner

The initial macro learner uses:

- recurrent candidate scoring over the complete legal candidate set;
- n-step Double-Q targets;
- a slowly updated target network;
- sequence replay with burn-in;
- sequence priority based on TD error;
- branch-balanced epsilon-greedy collection. Combat and macro may have one
  domain-level epsilon schedule each because a random macro commitment has a
  much longer consequence than a random card play; there are no per-operation
  floors.

Macro trajectories are compact enough to sample long contiguous sequences. A
sample contains all encountered macro decision types in that trajectory; combat
decisions remain in the combat view and connect through the cross-domain target.
There is no fixed per-operation slot order.

The strongest alternative is recurrent IQL/AWAC. It avoids a max over weakly
covered candidates and gives sampled actions a non-vanishing weighted
cross-entropy gradient. It is the predefined fallback if Double-Q develops
persistent unsupported value peaks. It is not trained concurrently and does not
become an additional loss on the same head.

### 8.3 Combat learner

The existing combat V-trace policy remains the champion because paired combat
evaluation has not shown it to be worse than the one-step greedy baseline. A
combat R2D2 challenger is trained from combat sequence view with the same legal
candidate encoding.

V-trace is retired only if the challenger preserves tactical win rate and HP
efficiency while improving stability or sample efficiency. Architectural
neatness alone is not evidence that the challenger is better.

## 9. Mechanisms retired by the reset

Semantic-shadow agreement authorizes construction of the new path; it does not
authorize removal from a still-authoritative legacy controller. The mechanisms
below are not inherited by a semantic optimizer. They remain sealed inside a
frozen legacy domain owner until that domain's semantic controller becomes
behavior-authoritative. No run segment mixes legacy and semantic preference
gradients. After that ownership handoff, the new control path does not carry
forward:

- operation-specific entry or selection exploration floors;
- completion guidance;
- entry support corridors;
- transaction completion and selection-group policy CE;
- macro residual AWR and the old lifecycle actor bridge;
- direct AVOID, cycle, contrast, completion and risk/cost actor objectives;
- run-success, Act-prefix and revival selected-action imitation actors;
- policy exemptions and ratchets created only to arbitrate those objectives;
- the transaction and failure-credit policy sidecars;
- the macro entropy collapse breaker.

Progress, stall and cycle detection remain as ordinary episode termination and
diagnostic facts. HP loss, revivals, Act completion and run completion remain in
the objective and reports. They no longer own independent policy gradients.

## 10. Migration without another patch lineage

### Stage 1: semantic shadow

Implement the semantic action compiler and derive semantic transitions from
existing episodes while the current controller continues unchanged. Check the
common real flows: rest/forge, reward take/skip, buy/remove/leave, map/event and
multi-select. The purpose is to establish correct decision boundaries, not to
add training labels to v47.

### Stage 2: isolated macro control

Freeze the combat champion. Train a new macro candidate-Q model from the macro
view and collect new branch-balanced experience. The old policy heads and old
transaction/failure policy sidecars are not part of the new optimizer.

The legacy macro controller is not surgically simplified during isolated
training. It remains a frozen champion outside the isolated semantic collector;
completion CE and macro AWR belong only to that sealed legacy controller and
never supervise candidate-Q.

### Stage 3: joined live graph

Run macro Q with the frozen combat champion through the explicit domain bridge.
Inspect whether rest/forge, take/skip, buy/leave and target selection vary with
state and whether held-out run outcomes improve. Do not automatically replace a
new segment with an old one because of a small evaluation batch.

Joined-live execution is a canary, not deletion or per-state controller mixing.
Macro ownership is assigned for a whole run segment. After an explicit macro
ownership handoff, legacy macro collection stops while combat V-trace remains
untouched.

### Stage 4: combat challenger

Train candidate-Q combat control from the combat view. Keep the V-trace champion
available for direct comparisons. Retire it only after the challenger wins on
actual combat outcomes.

### Stage 5: deletion

Once the new path is authoritative, delete the retired control paths instead of
leaving them disabled behind configuration switches. This is necessary to stop
future experiments from accidentally reactivating the old interaction graph.
Deletion is domain-scoped and occurs only after the corresponding controller has
been replaced.

## 11. Focused validation

Validation stays small and tied to likely failures:

1. replay real rest, shop, reward and multi-select episodes through the semantic
   compiler and compare the final game state;
2. show each encountered macro surface in producer, store and sampler counts;
3. backpropagate combat and macro batches separately and confirm domain ownership;
4. use paired state probes to see whether changing HP, upgrade preview, deck or
   shop inventory changes the corresponding value ranking;
5. compare held-out complete runs and direct combat outcomes against the frozen
   champion.

These checks do not encode a hand-written answer such as "low HP must rest".
They check that the model can use the relevant state and that the learning path
is alive.

## 12. What is explicitly out of scope

The first reset does not add:

- a larger model merely to fit the current labels faster;
- Agent57 curiosity or a meta-controller;
- RUDDER-style learned reward redistribution;
- LLM reward shaping;
- a per-operation rescue objective;
- learned option discovery;
- automatic policy rollback from small evaluation batches.

Model scaling is reconsidered only after the semantic control path learns
context-dependent macro choices. At that point width/depth comparisons answer a
capacity question instead of hiding a missing supervision path.
