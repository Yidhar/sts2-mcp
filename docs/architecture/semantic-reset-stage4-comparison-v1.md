# Stage-4 combat challenger — comparison record

Status: historical experiment record. The current production Stage-2 runners
do not expose combat candidate-Q training. The experiment may be repeated only
after an explicit encounter-terminal or cross-domain bootstrap boundary is
implemented; realized-return folding is valid only while the combat champion
is frozen.

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

## Round 2 — bridged head (2026-08-11, eval at ~ep191/300)

| arm | wins | act1 clears | floor p50 | floor mean |
|---|---|---|---|---|
| champion (baseline, same seeds) | 4/16 | 11/16 | 32 | 31.7 |
| challenger round 1 | 0/16 | 4/16 | 14 | 13.1 |
| challenger round 2 (bridged) | 0/16 | 9/16 | 20 | 21.7 |

The bridge moved combat capability materially (act-1 clears 4 -> 9,
floor mean +8.6) — the state-starvation diagnosis was load-bearing.
The champion still leads decisively: **retained**, gate unchanged.

**Round-2 final (ep300 completed model):** wins 0/16, act1 clears
**11/16 — act-1 parity with the champion**, floor_p50 30, floor_mean
26.9. Challenger trajectory across rounds: 4/14 -> 9/20 -> 11/30
(act1/floor_p50). The remaining gap is deep-run (act-2+ bosses, run
wins 0 vs 4). Champion retained; the slope suggests the gate is a
compute question, not a design question.

Macro behavior note (decision log, 59 rest decisions with both branches
offered): the rest/smith pole flipped — P(smith) 1.0 -> 0.0 in every HP
bucket. The aggregate gap crossed zero; per-HP differentiation has still
not emerged behaviorally. Probe-methodology caveat recorded: zero-hidden
probes under-measure the bridged pathway because single-observation HP is
pooling-diluted in the state embedding (generic log1p numeric transform is
not the cause); HP accumulates through run-scale memory, so behavioral
evaluation with real recurrent state is the authoritative instrument.
