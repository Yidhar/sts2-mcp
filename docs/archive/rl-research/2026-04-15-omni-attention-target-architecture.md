# Omni-Attention Target Architecture — 2026-04-15

## Goal

Make the online STS2 policy fully **search-free** at runtime by pushing combat,
route, and build reasoning into a unified attention model rather than relying on
MCTS / exact search / MuZero-style rollout logic.

## Mainline stack

- Observation encoder: `sts2_env/observation_v3.py`
- Shared dense primitives: `sts2_env/observation_common.py`
- Policy: `sts2_env/omni_attention_policy.py`
- Attention blocks: `sts2_env/attention_blocks.py`

## Core principle

Every legal action is represented as:

1. a **candidate query token**
2. a **candidate-local context token set**
3. cross-attention into a **shared world memory token set**

This lets the model learn:

- hand -> pile loop control
- potion / relic timing
- energy and X-cost spending
- enemy trait / intent conditioning
- route choice and build shaping

inside one model.

## World token domains

### Global

- `CLS_WORLD`
- `CLS_COMBAT`
- `CLS_BUILD`
- `CLS_ROUTE`
- `PLAYER_SURVIVAL`
- `RESOURCE_BUDGET`
- `THREAT_SUMMARY`
- `OBJECTIVE_CONTEXT`
- `RUN_CONTEXT`

### Runtime combat / build entities

- `HAND_CARD`
- `DRAW_PREVIEW_CARD`
- `DISCARD_CARD`
- `EXHAUST_CARD`
- `PLAY_PILE_CARD`
- `DECK_CARD`
- `RELIC`
- `POTION`
- `ENEMY_CORE`
- `ENEMY_POWER`
- `ENEMY_INTENT`
- `ENEMY_REACTIVE_TRAIT`
- `ENEMY_PHASE_RULE`

### Route

- `ROUTE_NODE`
- `ROUTE_SUMMARY_TOKEN`

## Candidate-local token domains

### Action source / explicit object

- `SOURCE_CARD_LOCAL`
- `SOURCE_POTION_LOCAL`
- `CARD_REWARD_LOCAL`
- `SHOP_ITEM_LOCAL`
- `UPGRADE_PREVIEW_LOCAL`
- `REWARD_LOCAL`
- `TARGET_LOCAL`

### Newly strengthened combat-local context

- `PLAYER_STATE_LOCAL`
- `ENERGY_CONTEXT_LOCAL`
- `DRAW_CONTEXT_LOCAL`
- `DISCARD_CONTEXT_LOCAL`
- `EXHAUST_CONTEXT_LOCAL`
- `PLAY_PILE_CONTEXT_LOCAL`
- `RELIC_TRIGGER_LOCAL`
- `POTION_OPTION_LOCAL`

These local tokens are action-conditional anchors. They intentionally duplicate
part of the world state so the decoder does not need to rediscover everything
from scratch for each candidate.

## Relation bias design

The attention bias layer now uses:

1. **token-type pair bias**
2. **owner-pair bias**
3. **same-owner bonus**
4. **same-entity bonus**

This is stronger than a plain typed-attention prior and gives the model an
explicit path to align:

- source card <-> matching hand/world token
- candidate <-> same entity across local/world memory
- owner-scoped interactions such as enemy-targeted or pile-specific attention

## Action owner semantics

Candidate owner assignment should favor the **source side** of the action:

- `play_card` -> `OWNER_HAND`
- `use_potion` / `discard_potion` -> `OWNER_POTION`
- `card_reward` -> `OWNER_REWARD`
- `deck_upgrade` -> `OWNER_UPGRADE`
- `map` -> `OWNER_ROUTE`

Only otherwise fall back to target enemy ownership. This keeps query identity
anchored on “what is being played / selected” instead of over-biasing toward the
target.

## Current implementation status

Implemented in mainline:

- `observation_v3` no longer depends on `observation_v2` directly
- candidate-local combat context expanded with pile / relic / potion / energy /
  player-state tokens
- relic and potion world tokens now carry numeric support features instead of
  being text-only placeholders
- relation bias upgraded with owner-pair and same-entity structure
- route candidate local token capacity expanded to avoid truncation

## Remaining high-priority follow-ups

1. Add richer **route/build local summaries**:
   - deck engine summary
   - path risk frontier
   - shop/removal opportunity summary

2. Add **entity-role channels** beyond owner ids:
   - source-role
   - target-role
   - trigger-role

3. Add auxiliary training heads for:
   - survival margin
   - lethal window
   - route risk
   - engine readiness
   - trait prediction

4. Demote remaining search-centric combat expert tooling to explicit legacy /
   offline analysis status once parity is acceptable.
