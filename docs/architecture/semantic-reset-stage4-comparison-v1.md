# Stage-4 combat challenger — comparison record

Companion to `semantic-decision-graph-reset-v1.md` §10 stage 4 and the
stage-3 handoff record. All numbers from paired deterministic held-out
seeds (fixed odd prefix, n=16, seed namespace 6400000).

## Round 1 (2026-08-11)

**Setup.** Challenger = the extended stage-2/3 macro model continued with
`--own-combat`: the authority owns the native-atomic combat view AND all
macro surfaces; 300 episodes, epsilon 0.15 branch-balanced, ~140k combat
decisions trained through the shared candidate-Q head
(`stage4-challenger-metrics.jsonl`). Champion = frozen v47@30k.

**Result.**

| arm | wins | act1 clears | floor p50 | floor mean |
|---|---|---|---|---|
| champion (V-trace, combat + legacy macro) | 4/16 | 11/16 | 32 | 31.7 |
| challenger (candidate-Q, combat + macro) | 0/16 | 4/16 | 14 | 13.1 |

**Decision: champion RETAINED** per the stage-4 gate ("retire it only
after the challenger wins on actual combat outcomes"). No configuration
switch, no mixing; the champion remains the combat authority for every
non-isolated run.

**Mechanism.** Training-side floors plateaued at ~15-17 from ep30 onward
(train-time greedy+epsilon), end_turn was 33% of trained combat decisions
— a cold Q head against a 150k-step V-trace incumbent. Loss converged to
~4e-4: the learner fits the value of the policy it has, and policy
improvement through the argmax operator alone is slow at this data scale.

**Contributing structural defect found and fixed after this round
(723c0dd).** The frozen-trunk probe showed the Q head's input features
carry ~4e-5 relative HP signal in EVERY model (champion included): state
context reached per-candidate features only through the additive
memory_to_candidate bottleneck. Combat Q-learning was therefore trying to
rank actions while nearly blind to HP/state — consistent with both the
challenger's plateau and fifteen lineages of missing HP-conditioned
macro behavior. The candidate-Q head now reads
[policy_features || next_recurrent_state] directly. Round 2 trains the
challenger on the bridged head (trunk inherited, Q head fresh) — the
retirement gate is unchanged.
