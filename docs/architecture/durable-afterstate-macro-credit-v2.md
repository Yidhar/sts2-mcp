# Durable Afterstate Macro Credit v2

Status: **active implementation specification**, 2026-11-12. This course evolves
[synchronous macro credit v1](macro-credit-v1.md) to eliminate single-root credit
starvation, candidate-count cardinality bias, upgrade-preview representation
aliasing, and noisy single-trajectory counterfactual splicing while preserving
inherited tactical competence from [hybrid IQL strict improvement v3](hybrid-iql-strict-improvement-v3.md).

## Problem diagnosis

Four interacting structural bottlenecks prevented deckbuilding and long-horizon
resource management from emerging in earlier courses:

1. **Single-root credit starvation**: `root_occurrence = 1` updated only one macro
   decision node out of roughly 138 macro decisions per full episode, while
   downstream macro choices had zero credit attribution.
2. **Flat categorical cardinality bias**: flat softmax over heterogeneous action
   branches (`15 Smith` candidates vs `1 Rest`, `8 Buy/Remove` candidates vs
   `1 Leave`, `3 Take` candidates vs `1 Skip`) gave singleton strategic actions a
   structural prior disadvantage of $1 / (N + 1)$, causing systematic over-drafting,
   compulsive gold spending, and low-HP campfire suicide.
3. **Unchosen counterfactual valuation gap**: without full-episode counterfactual
   rollouts at every floor (computationally intractable across ~138 macro nodes),
   unchosen economic actions (`Skip`, `Rest`, `Remove`, `BuyCard`) lacked a
   homogeneous post-decision value comparator, while candidate-conditioned value
   heads leaked candidate availability into state value baselines.
4. **Upgrade & build representation dilution**: `upgrade_preview` tokens lacked a
   dedicated fixed zone ID and explicit generic difference features against their
   source card, and persistent build tokens (`deck`, `relic`, `potion`, `player`,
   `run`) competed with large map-graph token sets inside global mean pooling.

## Architectural decisions

### 1. Multi-episode parallel full-run spine (`target_mode = "durable_afterstate_v2"`)

- Retain synchronous zero-lag teacher cohorts (`teacher_version == learner_version`
  at target construction) from [macro-credit-v1.md](macro-credit-v1.md), initialized
  via explicit in-memory model parameter transfer from clean `hybrid-iql-v3` P100
  (`sts2-hybrid-publication-v3`), never from polluted shadow runs.
- Collect `roots_per_cohort` (default `8`) natural full-run episodes in parallel
  across independent `HeadlessSim` worker processes with thread-safe GPU teacher
  inference and exact recorded `behavior_log_probability` at every semantic step.
- Every non-singleton macro decision node across every episode in the cohort
  participates in credit assignment (zero `root_occurrence = 1` bottleneck).

### 2. Candidate-independent durable value head $V_{\text{durable}}(W, h^{\text{run}})$

- `HybridDomainActorCriticModel` exposes an opt-in candidate-independent value head
  `durable_value_head` consuming only:
  - world state & persistent-build pooled embedding from `encode_world(W, domain_id)`, and
  - slow strategic recurrent memory $h^{\text{run}}$ updated via `run_recurrent_cell`
    on non-combat domains (and held read-only on combat domains).
- `forward_durable_value` never invokes `encode_candidates`, `combat_stream`,
  `macro_stream`, or `joint_stream`, guaranteeing by construction that legal
  candidate set composition cannot leak into $V_{\text{durable}}$.
- Supervision uses detached `mixed_vtrace_lambda_targets` (`trace_lambda` decaying
  strictly at macro-successor boundaries, never per combat card play) across all
  macro states, combat-entry states, and realized deterministic post-states
  $W_{\text{post}}(s_t, a_t)$.

### 3. Fail-closed deterministic afterstate synthesis and homogeneous MPO

- `sts2_rl/joint/afterstate.py` synthesizes exact post-decision world snapshots
  $W_{\text{post}}(s_t, a)$ **only** on reviewed deterministic economic surfaces:
  - **Card reward (`card_reward`)**: `take` (adds canonical card to `player.deck_cards`,
    increments `player.deck`, clears card-reward picker) and `skip` (clears card-reward
    picker without deck mutation).
  - **Rest site (`rest_site` / `deck_upgrade`)**: `rest` (applies factual
    `heal_amount` bounded by `max_hp`) and `smith` / `upgrade` (replaces one matching
    unupgraded deck copy with its exact runtime `upgrade_preview`).
  - **Shop (`shop` / `card_selection` removal)**: `buy_card` (deducts `price`, adds
    card to `player.deck_cards`, removes shop slot), `buy_potion` (deducts `price`,
    adds potion when `open_potion_slots > 0`, removes shop slot), `buy_relic`
    (**strictly when `has_upon_pickup_effect` is falsy**; deducts `price`, adds relic,
    removes shop slot), `remove_card` (deducts removal price if unpaid, removes one
    matching deck copy), and `leave` / `proceed` (closes shop view).
- **Fail-closed fallback**: any stochastic or unverified surface (`map`, `event`,
  `treasure`, combat hand/pile selections, random card packs, or any shop relic with
  `has_upon_pickup_effect == True`) returns `None`. If *any* legal candidate on a
  macro decision node fails deterministic synthesis, the entire node falls back
  cleanly to clipped importance-weighted mixed V-trace policy gradient:
  $$\mathcal{L}_{\text{macro-pg}}(t) = -\rho_t \cdot \text{stopgrad}(y_{\text{action}, t} - V_{\text{durable}}(s_t)) \log \pi_\theta(a_t \mid s_t)$$
- **Homogeneous counterfactual MPO**: on complete deterministic nodes, *every* legal
  candidate $a \in \mathcal{A}(s_t)$ (chosen and unchosen alike) is scored
  homogeneously via
  $$\hat{Q}_{\text{after}}(s_t, a) = r_{\text{imm}}(s_t, a) + d_{\text{imm}}(s_t, a) \, V_{\text{durable}}\!\left(W_{\text{post}}(s_t, a), h_{\text{post}}^{\text{run}}(s_t, a)\right)$$
  and projected through `project_macro_policy` under KL budget `effective_macro_kl`.
  Single-trajectory realized $y_{\text{action}, t}$ is **never** spliced into the
  chosen slot against expected $V_{\text{durable}}$ in unchosen slots. Realized
  targets supervise $V_{\text{durable}}$ exclusively, while MPO `effective_macro_kl`
  ramps from `0.0` to `macro_kl` across `durable_warmup_updates`.

### 4. Domain-safe two-stage hierarchical policy parameterization

Using the exact semantic `policy_branch_ids` already carried on every
`CandidateTokenBatch` (`candidates.ids[:, 13]`):

$$p_\theta(a \mid s) = p_\theta(b(a) \mid s) \cdot p_\theta(a \mid b(a), s)$$

- **Within-branch conditional law** $p_\theta(a \mid b, s)$: softmax of inherited
  `candidate_policy_head` scores $u_a$ restricted to legal candidates in branch $b$.
- **Combat domain (`domain_id == COMBAT_DOMAIN_ID`)**:
  $$\ell_{\text{branch}}^{\text{combat}}(b) = \log \sum_{a \in b} \exp(u_a) + g_\theta^{\text{branch}}(z_b)$$
  with zero-initialized final linear projection in $g_\theta^{\text{branch}}$. At
  Step 0 ($g_\theta^{\text{branch}} \equiv 0$), the induced joint distribution
  $\log p_\theta(a \mid s)$ is **mathematically identical** to the inherited
  `hybrid-iql-v3` flat softmax, preventing Turn-1 `EndTurn` collapse.
- **Macro domains (`domain_id != COMBAT_DOMAIN_ID`)**:
  $$\ell_{\text{branch}}^{\text{macro}}(b) = g_\theta^{\text{branch}}(z_b)$$
  where $z_b$ pools (`max` + `mean`) over candidate embeddings in branch $b$
  concatenated with `joint_state_embedding` and `recurrent_state`. At Step 0, all
  active macro branches (`Smith` vs `Rest`, `Buy` vs `Leave`, `Take` vs `Skip`)
  receive uniform $1 / B$ branch mass regardless of candidate multiplicity, while
  within-branch card/item rankings inherit `candidate_policy_head` preferences.
- `HybridDomainActorCriticModel.forward` emits exact normalized joint log-probabilities
  on legal slots (`0.0` on masked slots to satisfy finite-tensor validation), so
  downstream `torch.log_softmax(logits.masked_fill(~mask, -inf))` reproduces
  $\log p_\theta(a \mid s)$ identically. Macro greedy evaluation selects
  $b^* = \arg\max_b \log p_\theta(b \mid s)$ first, then $\arg\max_{a \in b^*} \ell_a$.

### 5. Encoding v19 & zero-init persistent-build residual grounding

- Preserve all learned tensor dimensions (`token_feature_dim = 224`, `d_model = 128`,
  `zone_vocab_size = 32`) for exact trunk weight compatibility with `hybrid-iql-v3`.
- Bump `GROUNDING_ENCODING_VERSION` to `grounded-relational-runtime-encoding-v19`:
  - Assign dedicated fixed `_ZONE_IDS["upgrade_preview"] = 26` (`_UNKNOWN_ZONE_HASH_START = 27`),
    preventing `upgrade_preview` tokens from colliding with base `card` zones.
  - Keep feature slots `0..214` untouched on all non-upgrade tokens; use trailing
    previously unused slots `215..218` (`_FEATURE_ABI_END = 219 <= 224`) exclusively
    on `upgrade_preview` tokens (`zone == 26`) for generic structural
    `upgrade_preview - source_card` deltas (`cost` delta, `dynamic_vars` sum delta,
    `dynamic_vars` max delta, and `keywords + tags` count delta) with zero card-ID
    rules and zero LayerNorm perturbation on ordinary combat/world tokens.
- Add zero-initialized `persistent_build` pooling & cross-attention over fixed
  persistent build zones (`deck`, `relic`, `potion`, `player`, `run`), feeding
  `durable_value_head` directly and injecting a zero-init residual into candidate
  representations without map-node dilution.

### 6. Two-phase co-training schedule & competence probes

- **Phase 1 (`update < combat_actor_warmup_updates`)**: `combat_policy_weight` is
  held at `0.0` (anchoring tactical combat behavior to the clean `hybrid-iql-v3`
  policy) while `durable_value_head` calibrates on realized macro/combat returns and
  the macro actor warms up.
- **Phase 2 (`update >= combat_actor_warmup_updates`)**: unlock joint Combat + Macro
  actor updates so tactical card play co-adapts to newly drafted archetypes while
  `V_durable` reflects evolving combat competence.
- Track three paired competence probes in `sts2_rl/macro_competence_probes.py`
  without adding any reward shaping:
  1. Campfire low-HP (20% HP) vs high-HP (85% HP) $\Delta V_{\text{durable}}(\text{Rest} - \text{Smith})$ and $p(\text{Rest})$.
  2. Smith vs Remove card target separation across starter strikes/defends vs non-starter cards.
  3. Rolling card-reward `Skip` rate and terminal deck size distribution at `revival_budget = 64` prior to `ratchet_curriculum` promotion.

### 7. Remote Colab G4 compute & atomic Google Drive persistence

- Training runs execute remotely on NVIDIA G4 GPU sessions managed through Colab CLI,
  using native Linux `net9.0` `HeadlessSim` builds verified against
  `third_party/sts2-ai.lock.json`.
- Active I/O writes to fast local NVMe (`<LOCAL_RUNTIME_ROOT>`) and atomically syncs
  committed `checkpoint/`, `publication.pt`, `course.json`, `episodes.jsonl`,
  `metrics.jsonl`, and `campaign_status.json` after every cohort update to
  `<GDRIVE_ARTIFACT_ROOT>/campaigns/<lineage_id>/`, enabling exact crash-safe
  `--resume` across ephemeral VM lifetimes.
