# v47 factual macro-economy AWR (design and launch contract v1)

Status: IMPLEMENTED.  Target lineage: `full-run-revival-v47-macro-economy-awr-model-init`.
The reviewed parameter source is the fixed v46 healthy-validation checkpoint
at local environment step 10,035.  V47 is a model-parameter initialization,
not an exact continuation: compatible network tensors are inherited, while the
optimizer, replay stores, RNG state, rollout queue, and recurrent actor state
start fresh.  The fixed v46 source received one complete 1,820,210,752-byte
SHA-256 validation during v47 landing.  A pinned cross-process attestation
reuses that proof; launcher and trainer startup still revalidate the
attestation digest, checkpoint manifest/metadata digests, complete file set,
sizes, semantic identity, and migration ABI, but do not repeatedly stream and
hash the replay sidecars.

## 1. Measured defect

V44-v46 could complete upgrade/removal transactions, but the broader deck
economy remained trapped in saturated policy branches:

- card rewards were almost always taken and skip probability approached zero;
- an affordable shop was almost never left voluntarily;
- rest versus upgrade, upgrade target, and removal target did not track the
  observed downstream resource outcome reliably;
- ordinary V-trace, entropy, and importance-weighted imitation all lost useful
  gradient once the selected policy probability was numerically near zero.

This is a missing supervision path, not evidence that the shared model is too
small.  Increasing hidden size would fit the same incomplete objective faster.
V47 therefore changes the factual credit contract while keeping model scale,
reward-v7, and the revival budget fixed.

## 2. Non-negotiable invariants

V47 does **not** contain:

- an HP threshold that chooses rest or forge;
- a card, relic, potion, event, or item preference table;
- a reward for taking, skipping, buying, leaving, upgrading, or removing;
- a guessed return for an action that was not executed;
- deterministic-evaluation action rewriting;
- a rollback gate that silently discards a newly learned segment.

Every macro target comes from an executed action and an authoritative observed
post-state.  Missing receipts are censored, never inferred.

## 3. Unified macro-economy contract

The reviewed operation families are:

| Surface | Executed alternatives | Required factual receipt |
|---|---|---|
| Card reward | take / skip | deck count +1 and exit / unchanged deck and exit |
| Shop | buy card, relic, potion, generic item / leave / open removal | gold/resource mutation / unchanged resources and exit / verified removal lifecycle |
| Rest site | rest / open upgrade | strict HP increase / verified upgrade lifecycle |
| Upgrade selection | select target / cancel / confirm | verified upgraded deck mutation |
| Removal selection | select target / cancel / confirm | verified one-card deck reduction |

`proceed` on the shop screen is explicitly encoded as `leave_shop`, not as a
zero-item purchase.  This semantic rule is part of grounded encoding v16's
fingerprint.

For a committed option, the factual return ends immediately before the next
opportunity of the **same resource family**:

- rest/upgrade -> next rest site;
- purchase/leave/removal -> next shop;
- card take/skip -> next card reward;
- any family -> earlier Act boundary or authoritative run terminal.

No learned post-state bootstrap is appended at that boundary.

## 4. Encoding and model isolation

Each decision snapshot carries one reviewed routing ID:

```text
0 none
1 rest site
2 shop
3 card reward
4 upgrade selection
5 removal selection
```

The ID is a surface fact, not a preference.  Snapshot ABI is v4 and grounded
encoding is v16.  V15 replay cannot exact-resume under v16; the reviewed
v15->v16 edge permits parameter initialization only.

When transaction heads are enabled, the model adds:

1. `macro_option_value_head`: candidate-independent `V_macro(s)`;
2. `macro_policy_head`: a residual over inherited policy logits;
3. surface embeddings for state and candidate routing.

The residual and macro-value final linear layers are zero-initialized.  Thus
gate-zero behavior equals the inherited v46 policy exactly and the first AWR
advantages are measured against a known neutral baseline rather than a random
new-head offset.  The inherited base logits, policy features, and recurrent
state are detached at the new heads.  A macro loss can update only the macro
sidecar; it cannot move the combat/deck encoder, recurrent trunk, or base
policy head.

## 5. Factual AWR objective

For an executed option `a` with observed option return `G`:

```text
L_value = Huber(V_macro(s), G)
A       = stopgrad(G - V_macro(s))
w       = exp(clip(A / temperature, -c, c))
L_surface = mean_rows(-stopgrad(w) * log pi_macro(a | s))
L_actor   = mean_reviewed_surfaces(L_surface)
```

The baseline is candidate independent.  It is not `Q` for an unexecuted
alternative.  Rows are averaged inside rest, shop, card reward, upgrade
selection, and removal selection, then the represented surfaces are averaged
with equal weight.  Absolute AWR weights are deliberately **not** divided by
their observed batch mean: doing so would turn a singleton surface into weight
one and erase its factual advantage.  This surface-level reduction prevents
frequent card rewards from dominating scarce shop/rest opportunities by count
alone.  Policy-version lag,
collection-to-current log-probability drift, forced singleton decisions, and
unknown surfaces suppress the actor label fail-closed.  Value labels remain
usable when actor admission is suppressed.

This cross-entropy-form actor gradient stays non-zero at a saturated selected
probability, which is the specific gap ordinary importance-weighted paths could
not repair.

## 6. Selection-group completion

A committed upgrade/removal lifecycle supplies a second, deliberately coarse
label: aggregate probability of the semantic **Select** branch should increase
relative to cancel.  All exact card candidates in that branch are summed.
The label never says which card is best.  Exact target choice remains governed
by the selected card's factual option return and the existing transaction-Q
representation loss.

## 7. Exploration and imitation

Winner/Act-prefix imitation is exempt on rest, shop, card-reward,
upgrade-selection, and removal-selection surfaces. Otherwise an incumbent
winner that always takes/buys/forges or chooses one target would continually
re-imprint the absorbing branch.

Training uses a temporary, logged branch-balanced epsilon floor of 0.10 only
for `reward_skip` and `shop_leave`.  Behavior probabilities are stored for
off-policy accounting.  Completion guidance is zero and evaluation bypasses
the explorer.  Upgrade/removal entry and selection scaffolding remain removed.

## 8. Checkpoint and migration contract

Fixed source:

- run ID: `9a23c1bc-7a7b-467e-ad97-95af63fe4bec`
- checkpoint: `healthy-validation-step-000010035`
- checkpoint ID: `a497e8e7-00ea-4a89-af41-ce30691997cd`
- environment steps: `10035`
- policy version / learner updates: `164 / 164`
- manifest SHA256: `775bf964fb54ffa86166495d80fba39dd6a49de13522032b5c3654e6860cba35`
- metadata SHA256: `0a2de33f13ce49807065f264ee3d739833c4044602907075ef5284cfc0f144f9`
- byte-attestation SHA256: `a00096dbccde2355471b5cdf093031db53f887357ca4b28265631865aab3d518`

The launcher must use `--initialize-from`, never `--resume`.  Model-init accepts
either the complete macro-head family or none of it; a partial family fails.
Every v47 exact checkpoint records
`sts2-macro-option-value-residual-policy-heads-v1`; exact resume requires both
learner and actor state specs to contain the complete family.

## 9. Diagnostic run and acceptance

The first run is limited to 30,000 environment steps with paired held-out
evaluations at 0, 5k, 10k, 20k and a 30k final audit.  Evaluation rollback and
liveness policy gates are disabled; infrastructure watchdogs still fail closed.

Before considering a longer continuation, verify:

1. every encountered reviewed surface produces non-zero macro-value labels;
2. non-forced, recent rows produce macro-actor labels rather than only
   lag/drift/singleton suppressions;
3. AWR weights and all losses remain finite and bounded;
4. committed upgrade/removal paths produce selection-group labels;
5. deterministic reward-skip and affordable-shop-leave probabilities are no
   longer literal zero and vary by state rather than becoming unconditional;
6. rest/upgrade, target choice, deck size, gold retention, HP loss, revival,
   Act reach, deadlock, entropy, and tactical evaluation are reviewed together;
7. no shared-trunk gradient appears under an isolated macro-loss test.

Passing these checks establishes that the learning path is alive.  It does not
by itself prove the resulting macro strategy is optimal; a longer exact
continuation is a separate, explicit decision.
