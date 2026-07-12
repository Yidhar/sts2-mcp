# CardEffectProfile derived_view — schema

The `derived_view` block on every card in
`game-data/generated/card_effect_profiles.generated.json` is a stable structured summary of
the card's mechanical profile.  It is derived from internal card ids, C# source
facts, curated overrides and (only as last resort) localized text regex.

## Field source priority

Per TASK-D1 the derivation order is:

1. **Game-internal card data / effect components / power ids / card modifiers.**
2. **Bridge C# reflection or explicit profile.**
3. **Hand-maintained override table** (`CURATED` in
   `tools/generate_card_effect_profiles.py`).
4. **Localized text regex** — only as fallback, and the result must be flagged
   with `source.fallback_text_regex_used = true`.

## Top-level fields

| Field | Description |
|---|---|
| `card_id` | Canonical `CARD.<NORMALIZED>` id. |
| `title` | English title (display only — not used for parsing). |
| `color` | Card class (`ironclad`, `colorless`, ...). |
| `type` | `attack`, `skill`, `power`, `status`, `curse`, `quest`, `token`. |
| `rarity` | `basic`, `common`, `uncommon`, `rare`, `event`, `ancient`, ... |

## Field groups

### `cost`
| Field | Description |
|---|---|
| `base` | Base energy cost (`-1` for X-cost cards). |
| `upgraded` | Upgraded energy cost (`-1` for X-cost cards). |
| `is_x_cost` | True if the card resolves cost as X (e.g. Whirlwind). |
| `can_change_cost` | True if any operation modifies the card's cost. |
| `cost_reduction_tags` | Tags describing cost-modifier ops (`set_cost_0`, `delta_-1`, `duration_this_turn`, ...). |

### `lifecycle`
| Field | Description |
|---|---|
| `exhausts_on_play` | True if the played card itself exhausts on play (Exhaust keyword). |
| `ethereal` | True if Ethereal keyword is present. |
| `retain` | True if Retain keyword is present. |
| `self_purge` | True if the card explicitly removes itself from combat. |
| `returns_to_hand` | True if the card returns to hand after play. |
| `replay_or_duplicate` | True if the card replays itself or duplicates other cards. |

### `hand_mutation`
| Field | Description |
|---|---|
| `upgrades_hand` | True if any op upgrades a card in hand. |
| `upgrade_targets` | One of `one`, `choice_one`, `random_one`, `one_or_all_by_upgrade_state`. |
| `transforms_cards` | True if any op transforms a hand card. |
| `copies_cards` | True if any op copies a hand card. |
| `creates_cards` | True if any op generates new cards into combat. |
| `discard_hand` | True if any op discards the entire hand. |
| `draw` | Total cards drawn (sum of `count` across `draw_card` ops). |
| `select_cards` | `{enabled, min, max, target_zone}` — selection constraints if the card requires choosing target cards. |

### `pile_mutation`
| Field | Description |
|---|---|
| `moves_to_exhaust_self` | Played card itself moves to exhaust pile. |
| `moves_to_discard_self` | Played card itself moves to discard pile (default for non-exhaust, non-power cards). |
| `moves_to_exhaust` | Any op moves a card to exhaust (self or other). |
| `moves_to_discard` | Any op moves a card to discard (other than the played card). |
| `shuffles_into_draw` | Any op moves a card to the draw pile. |
| `puts_card_on_top` | Any op puts a card on top of the draw pile. |
| `removes_card_from_combat` | Any op removes a card from combat permanently (transform without `result_card`). |

### `combat_effect`
| Field | Description |
|---|---|
| `damage` | Base damage roll (parsed from `DamageVar`). |
| `block` | Base block roll (parsed from `BlockVar`). |
| `hit_count` | Multi-hit count parsed from `WithHitCount`. |
| `weak`, `vulnerable`, `frail`, `poison` | Counts of debuff power applications. |
| `strength`, `dexterity`, `artifact`, `thorns` | Counts of buff power applications. |
| `energy_gain` | Energy gained by `gain_energy` ops. |
| `target_type` | `Self`, `AnyEnemy`, `AllEnemies`, `AnyOpponent`, ... |

### `mechanism_effect`
| Field | Description |
|---|---|
| `can_change_facing` | True if the card targets an enemy (Kaiser facing relevance). |
| `can_strip_artifact` | True if the source mentions Artifact-strip primitives. |
| `can_trigger_stun` | True if the card applies StunPower. |
| `one_card_lock_impact` | `unknown` for now — Phase 4 (TASK-E2 Ceremonial) will populate. |

### `source`
| Field | Description |
|---|---|
| `primary` | Always `game_internal_id` (we never derive primarily from text). |
| `fallback_text_regex_used` | True only when no internal/curated path could derive the profile. |
| `profile_quality` | `curated_internal_id` (CURATED override), `generated_source_facts` (C# parse), or `unknown`. |
| `source_available` | True if the C# source file was reachable. |
| `source_sha1` | SHA1 of the C# source file at generation time. |

## Reading the derived_view

`sts2_env.card_effect_profile` exposes:

```python
from sts2_env.card_effect_profile import (
    card_derived_view,
    card_cost_view,
    card_lifecycle_view,
    card_hand_mutation_view,
    card_pile_mutation_view,
    card_combat_effect_view,
    card_mechanism_effect_view,
    card_source_view,
)
```

Each helper accepts a card-payload dict and returns the named subsection (or
`{}` if the profile is missing).  Observation/action encoders consume these
helpers — they MUST NOT walk localized card text.
