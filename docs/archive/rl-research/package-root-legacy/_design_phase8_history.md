# Phase 8 — Action-History Architecture

**Status**: design (not yet implemented). Target start: after the current sim long-train finishes.

## Motivation

Phase 6 (POWER_SLOT + power bank) gave the policy static knowledge of buff stacks, but it's still **stateless between steps**. This creates three classes of failure we've observed in training:

1. **Card-selection double-click trap** — picked `select_card(0)` once, now sees identical obs next step (confirm is masked open but policy keeps re-selecting). **Partially mitigated in P0 by translator filtering + confirm-hoist, but the underlying "can't see what I just did" problem remains.**
2. **Combat play-order reasoning** — Ironclad: play Inflame → next Strike hits for +3. Defect: DoubleTap → next Attack triggers twice. Silent: Concentrate → next Shiv deals +4. Without history, policy can't reason "play Inflame BEFORE Strike, not after".
3. **Startup / windup sequences** — Power cards (Demon Form, Echo Form) want to go turn 1-2. Scaling attacks (Bludgeon, Reaper) want to go late. Potion use wants to be reactive to enemy intent. All of these require knowing "where am I in the combat timeline" which current architecture can't express.

Quantitatively: 12% of episodes still hit `combat stuck` after P0 (card_selection trap fix). Those 12% are exactly this class — policy picks same card every step without advancing combat state.

## Scope decision — Tier 1 first

Build in two tiers; evaluate Tier 1 before committing to Tier 2:

| Tier | Scope | Cost | Fixes |
|---|---|---|---|
| **Tier 1** | 16 history tokens carrying (action_family, semantic_role, card_id_bucket, target_type, step_offset, same_turn_flag, result_flags, canonical_text) | ~1.5 days | card_selection residual + "don't play same card twice" patterns |
| **Tier 2** | Per-token `pre_state` / `post_state` state-delta vectors + new `aux_causality` head predicting card-effect deltas | ~3 more days | True causal reasoning: "Inflame buffs every future Strike", "Vulnerable stack survives 2 turns" |

Tier 1 alone should address the 12% combat-stuck residual. If stuck rate falls <5% and Act 1 boss victory rate goes from 0 to nonzero, Tier 1 is sufficient and we skip Tier 2 for now.

## Architecture — Tier 1

### Storage (env_v2)

```python
from collections import deque

class HistoryEntry:
    family: str                # from semantic_action.semantic_action_signature
    semantic_role_flags: list[str]  # attack/skill/power/draw/block/scaling/...
    card_id_bucket: int        # hash(card_id) % 8192, 0 if not a card action
    target_type: str           # single_enemy / all_enemies / self / hand / none
    same_turn: bool            # True if combat.round unchanged since append
    same_encounter: bool       # True if still in same combat
    same_floor: bool           # True if same run.floor
    step_offset: int           # 0 = most recent, grows as new entries shift this back
    result_flags: dict         # {reward_nonzero, rejected, phase_changed, combat_ended}
    canonical_text: str        # for text encoder

self._action_history: deque[HistoryEntry] = deque(maxlen=16)
```

**Update in `step()`**:
- Before `bridge.step`, capture `pre_combat_round` / `pre_encounter_id` / `pre_floor`
- After `_update_live_state`, diff to set `same_turn` / `same_encounter` / `same_floor` on the *new* entry being appended. Existing entries get their flags updated in-place if relevant (e.g., all entries on a cleared combat flip `same_encounter=False`).
- Compute `result_flags` from the transition
- Append. Maxlen=16 auto-drops oldest.
- Also increment `step_offset` on each surviving older entry (or compute on the fly by `len(history) - 1 - index`; the latter is cleaner).

**Reset in `reset()`**: `self._action_history.clear()`.

### Observation changes

New fields added to raw obs dict (emitted by env_v2):

```python
raw_obs["action_history"] = [
    entry.to_dict() for entry in self._action_history
]
```

No wire/bridge changes — this is synthesized in env_v2, not in the bridge layer.

### Encoder changes (`observation_v3.py`)

- Bump `OBSERVATION_API_VERSION`: `"attention_obs_v3"` → `"attention_obs_v4"`
- Bump `MAX_WORLD_TOKENS`: `384` → `400` (+16 history slots)
- Bump `NUM_TOKEN_TYPES`: `66` → `67` (add `HISTORY_ACTION`; don't subtype combat/route since `same_encounter` flag captures the distinction)
- New constants:
  - `MAX_HISTORY_TOKENS = 16`
  - `OWNER_HISTORY = 53`
  - `ROLE_HISTORY_ACTION = <next free id>`
  - `ZONE_HISTORY = <next free id>`
- New method `_append_history_tokens(obs, tokens, start_offset)`:
  - Emits 16 tokens (pad with "empty history" token if deque shorter than 16; pad token has all zeros + an `is_empty=1.0` bit)
  - Each token's 96 numeric features packed per schema below
  - 64 text features: `TextEncoder.encode(entry.canonical_text)` — reuse existing text_encoder caching

**Numeric feature packing (96-d per token):**

```
[0:49]   action_family one-hot (49 families from semantic_action)
[49:65]  semantic_role flags  (16 binary: attack/skill/power/curse/status/draw/block/damage/scaling/...)
[65:73]  target_type one-hot  (8: single_enemy/all_enemy/self/random_enemy/all/hand/none/other)
[73:89]  step_offset one-hot  (16: positions 0..15 + "empty" beyond history len)
[89:90]  same_turn             (float 0/1)
[90:91]  same_encounter        (float 0/1)
[91:92]  same_floor            (float 0/1)
[92:96]  result_flags         (4: reward_nonzero, rejected, phase_changed, combat_ended)
```

Card_id_bucket is handled via the existing `entity_ids` path (like POWER_SLOT tokens' `power_id_bucket`) — adds a `history_card_bias` embedding table in RelationBias (see below).

### Policy changes (`omni_attention_policy.py`)

Add a **7th world bank `history`**:

```python
WORLD_BANK_NAMES = ("runtime", "support", "enemy", "build", "route", "powers", "history")

_WORLD_BANK_ROLE_NAMES["history"] = ("HISTORY_ACTION",)
_WORLD_BANK_ZONE_NAMES["history"] = ()  # route by role only
```

History bank cross-attends with every other bank via the existing `CrossAttentionBlock` infrastructure. The critical cross-attentions for this phase to work:

- **runtime ← history**: "this card in my hand was played 2 steps ago" (enables Double-Tap / Echo Form reasoning)
- **powers ← history**: "Demon Form was played 4 steps ago, I've accumulated +8 strength since"
- **enemy ← history**: "Vulnerable was applied 2 steps ago, still live"
- **candidate ← history**: each candidate action attends to history to learn "if Inflame was played recently, attack cards become higher-priority candidates"

### Relation bias (`attention_blocks.py`)

Add `history_card_bias = nn.Embedding(ENTITY_HASH_BUCKETS, n_heads)`, mirroring the Phase 6 `power_bucket_bias` pattern. In `forward()`, when key token role is HISTORY_ACTION, add bias indexed by the history token's `card_id_bucket`. Zero-init so Phase 6 checkpoints load cleanly.

This lets the model learn "when Strike just hit, the next Strike-adjacent card is hot" without baking explicit per-card rules.

### Aux supervision

**Tier 1 keeps existing 7 aux heads unchanged.** No new head.

### Observation API version + checkpoint compat

- `OBSERVATION_API_VERSION = "attention_obs_v4"`
- Checkpoint loader (`checkpoint.py::load_online_policy_state_dict`) already uses `strict=False`. New parameters to zero-init:
  - History bank `TransformerEncoderBlock` weights
  - Cross-attention blocks targeting/from history
  - `history_card_bias` embedding
  - Type/role/owner/zone embedding rows for new IDs
- Phase 6 checkpoint → Phase 8 warmstart should work out of the box with `strict=False`.

## Architecture — Tier 2 (deferred, conditional)

Add per-token state-delta vectors and a new aux head. Only build if Tier 1 eval shows residual combat-stuck >5%.

### Token schema extension (Tier 2)

Replace the 4 result_flag bits with a richer 16-d pre/post state pair:

```
[92:108]  pre_state   (16-d: energy, hand_size, player_hp_ratio, enemy_total_hp_ratio,
                              strength, dex, focus, block_on_player, vuln_on_any_enemy,
                              weak_on_any_enemy, draw_size, discard_size, exhaust_size,
                              stars, gold_norm, turn_num_norm)
[108:124] post_state  (16-d: same fields, post-action values)
```

This brings the numeric block to 124-d; either grow TOKEN_FEAT_DIM or shrink one_hot blocks.

### New aux head: `action_causality`

Input: per-candidate token (from `CandidateDecoderBlock`), conditioned on current history bank state.
Output: 8 heads predicting `post_state - pre_state` delta **if this candidate were played**:
- `predicted_damage` (normalized to /50)
- `predicted_block` (normalized to /30)
- `predicted_self_hp_loss` (normalized to /20)
- `predicted_draw` (normalized to /5)
- `predicted_energy_delta` (normalized to /3)
- `predicted_strength_delta` (normalized to /5)
- `predicted_dex_delta` (normalized to /5)
- `predicted_vuln_delta` (normalized to /5)

Target is self-supervised from the ACTUAL step transition whenever this candidate was the chosen action. Mask=1 for chosen-candidate row, 0 elsewhere — same pattern as existing `candidate_objective` head.

This head gives **every combat step a dense supervision signal** about "what does each card do, given current buffs". This is the bit that teaches real causal reasoning.

## PR decomposition

| PR | Contents | LoC est. | Eval gate |
|---|---|---|---|
| **P8.1** | `HistoryEntry`, `self._action_history` deque in env_v2, raw_obs["action_history"] emission, reset handling, unit tests | ~120 | Probe: deque contents correct, reset works |
| **P8.2** | observation_v3 constants (MAX_WORLD_TOKENS 400, NUM_TOKEN_TYPES 67, HISTORY_ROLE/OWNER/ZONE), `_append_history_tokens()`, obs_api_version bump to v4 | ~180 | Encoder roundtrip: encode a fake 5-entry history and decode features back |
| **P8.3** | omni_attention_policy `history` bank + routing, cross-attention wiring, zero-init for Phase 6 checkpoint load | ~80 | Smoke 5k steps: value_loss > 0, policy doesn't NaN, grad norm sane |
| **P8.4** | attention_blocks.RelationBias `history_card_bias` embedding + forward integration | ~40 | Smoke: bias values are nonzero after 1k steps |
| **P8.5** | Phase 6 → Phase 8 checkpoint warmstart script (drops incompatible keys, zero-inits history-specific params, re-saves) | ~60 | Load and resume training for 1k steps without loss spike |
| **P8.6** *(conditional)* | Tier 2: state-delta vectors + `action_causality` aux head + target computation | ~250 | Eval 50k: action_causality/damage_explained_variance > 0.3 |

P8.1–P8.5 is Tier 1 (~480 LoC). P8.6 is Tier 2.

## Evaluation plan

Warmstart from current Phase 6 + P0 checkpoint (the one finishing now). Run 200k step smoke and compare:

| Metric | Pre-P8 baseline | P8 Tier 1 target | P8 Tier 2 target |
|---|---|---|---|
| combat stuck rate | 12% | <5% | <2% |
| card_selection stuck rate | ~5% (post P0 residual) | <2% | <1% |
| average max_floor | 7-8 | 10+ | 13+ |
| Act 1 boss victory rate | 0% | >0% | >5% |
| value_loss gradient norm | healthy | healthy | healthier |
| aux_causality explained_var (damage head) | N/A | N/A | >0.3 |

Kill criteria for Tier 1: if stuck rates don't meaningfully drop by 50k steps, Tier 1 alone is insufficient — escalate to Tier 2 immediately.

## Risks & mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| History tokens add compute → training throughput drops | Med | Profile P8.3 smoke; if >15% slowdown, reduce N_HISTORY_TOKENS to 8 |
| 16 new world tokens break attention_mask shape assumptions | Low | Covered by `MAX_WORLD_TOKENS` bump; test_observation_v3 should catch regressions |
| Phase 6 checkpoint won't load cleanly | Low | P8.5 explicitly handles it; worst case is fresh init |
| Policy attends to history but can't reason about it (needs Tier 2 state-delta) | Med | Eval gate at 50k triggers Tier 2 |
| `same_turn` flag requires tracking combat.round which sim emits inconsistently | Low | Probe combat.round values before P8.1; already exposed per sim translator |
| Text encoder cost for 16 extra tokens | Low | Already deduplicated by `TextEncoder`'s cache — `canonical_text` for a given action is stable |

## What stays stable

- Phase 6 POWER_SLOT / power bank / power_bucket_bias — untouched
- Aux heads objective/transition/traits/build/selection/route/enemy_state — untouched
- All Skada BC pipeline (skada_bc_*.py) — untouched; BC doesn't need history (human data is single-decision)
- Bridge / sim translator — untouched (history is synthesized Python-side)
- PPO algorithm / AuxMaskablePPO buffer — untouched

## Open questions

1. **Pad strategy** — when history has <16 entries (early in episode), should the padding tokens attend to other banks or be mask-ignored? Currently the existing token mask pattern in observation_v3 would mark them present-but-zero. Recommend an `is_history_empty` feature bit + leave them unmasked so attention can learn to ignore empty slots organically.

2. **Cross-encounter retention** — when a combat ends, do we clear the history (so new combat starts fresh) or keep the last few entries for cross-combat scaling tracking? Recommend **keep** but flip `same_encounter=False` — Act-1 strategy often compounds across combats (relic pickups, deck composition choices).

3. **BC warmstart and history** — Skada BC trains on map/rest/relic/card_reward decisions which are typically single-shot (no action history needed). Does BC-warmstart-then-PPO still work with the new history bank? Answer: yes, history tokens will be empty at BC eval (BC samples are single decisions), so history bank contributes zero — identical to no-history behavior. BC warmstart remains valid.

4. **Do we need a new run_memory field?** run_memory is per-episode summary (deck, relics, floor). History is per-step action sequence. They're orthogonal; no run_memory change.

## Next step

When the current long-run finishes and the P0 translator fix is committed, start P8.1. Suggest doing P8.1–P8.3 as a single "Tier 1 skeleton" PR, then P8.4–P8.5 as the "wire it up" PR. Eval at 50k steps, decide Tier 2 or ship.
