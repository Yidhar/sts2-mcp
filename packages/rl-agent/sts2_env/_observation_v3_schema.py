"""Stable token schema and pure helpers for observation v3."""

from __future__ import annotations

from typing import Any

import numpy as np

from . import observation_common as obs_common
from .text_encoder import TEXT_DIM

MAX_ACTIONS = obs_common.MAX_ACTIONS
# Phase 6 (attention_obs_v3): +64 tokens for dedicated POWER_SLOT /
# CARD_KEYWORD tokens, split out of the entity numeric inlining. At typical
# STS2 scale we see <= 35 active powers (player + 4 enemies 脳 5 powers) and
# a handful of card-keyword tokens; 64 is comfortable headroom.
# Phase 8 Tier 1 (attention_obs_v4): +28 HISTORY tokens (20 step-detail +
# 8 turn-summary) so the policy can reason about "what did I just do"
# without needing a recurrent architecture. See action_history.py.
# STS2-Pass-Large-v1: expand token and candidate-local caps to leave route /
# build / history headroom for full-run training and avoid silent truncation.
MAX_WORLD_TOKENS = 640
MAX_CANDIDATE_LOCAL_TOKENS = 40
TOKEN_NUMERIC_DIM = 96
TOKEN_TEXT_DIM = 64
TOKEN_FEAT_DIM = TOKEN_NUMERIC_DIM + TOKEN_TEXT_DIM
ENTITY_HASH_BUCKETS = 8192
MAX_OWNER_ID = 127
MAX_ORDER_ID = 63
# v4: HISTORY_STEP_DETAIL / HISTORY_TURN_SUMMARY token types exist.
# v3 checkpoints load with strict=False; new history-specific embeddings
# zero-init. See _design_phase8_history.md for the migration plan.
OBSERVATION_API_VERSION = "attention_obs_v5_pass_large"

# Maximum number of POWER_SLOT tokens emitted per step. Covers the typical
# worst case (3-5 player buffs + 4 enemies 脳 5 powers each 鈮?25-30) with
# headroom. Anything beyond is dropped by emission order.
MAX_POWER_SLOT_TOKENS = 40
# Bucket space for power_id categorical embedding. STS2 has ~50 canonical
# powers + mod headroom. Use a prime-ish power of 2 and hash unknowns into
# it deterministically.
POWER_ID_BUCKETS = 128

# Phase 8 Tier 1: action-history tokens. Imported at module load so we
# can also expose MAX_HISTORY_TOKENS / MAX_STEP_DETAIL_TOKENS /
# MAX_TURN_SUMMARY_TOKENS without cross-module duplication.

# Phase 6.4: per-card keyword tokens (Retain/Ethereal/Exhaust/Innate/...).
# Previously these were bitflags buried inside HAND_CARD numerics; breaking
# them out lets attention learn per-keyword patterns like "Ethereal +
# unplayed + end-of-turn approaching 鈫?penalty" directly.
MAX_CARD_KEYWORD_SLOTS = 16
# Stable bucket ids for card-keyword categorical embedding. Source: the
# content_registry semantic_tags vocabulary; only keywords that affect
# play-phase decisions are surfaced. 0 reserved for PAD/unknown.
_CARD_KEYWORD_BUCKETS: dict[str, int] = {
    "retain": 1,
    "ethereal": 2,
    "exhaust_self": 3,
    "innate": 4,
    "unplayable": 5,
    "x_cost": 6,
    "purge": 7,
    "scry": 8,
    "return_to_hand": 9,
    "add_to_hand": 10,
    "upgrade_self": 11,
    "add_to_draw": 12,
    "bound": 13,
    "card_lock": 14,
    "cost_lock": 15,
    "forced_play": 16,
    "temporary": 17,
}

OWNER_NONE = 0
OWNER_ENEMY_BASE = 1
OWNER_PLAYER = 32
OWNER_HAND = 40
OWNER_DRAW = 41
OWNER_DISCARD = 42
OWNER_EXHAUST = 43
OWNER_PLAY = 44
OWNER_DECK = 45
OWNER_RELIC = 50
OWNER_POTION = 51
# v3: distinguish power-owned slots from their host entity so attention
# can route power鈫抍ard edges without colliding with entity鈫抍ard routing.
OWNER_POWER = 52
# Phase 8 Tier 1: dedicated owner id for HISTORY tokens so the
# owner_pair_bias can learn "history-to-candidate" edges distinct from
# any entity-hosted token's owner. All history tokens (step-detail and
# turn-summary alike) share this one owner.
OWNER_HISTORY = 53
OWNER_ROUTE = 60
OWNER_SHOP = 61
OWNER_REWARD = 62
OWNER_UPGRADE = 63

TOKEN_TYPES = [
    "PAD",
    "CLS_WORLD",
    "CLS_COMBAT",
    "CLS_BUILD",
    "CLS_ROUTE",
    "PLAYER_SURVIVAL",
    "RESOURCE_BUDGET",
    "THREAT_SUMMARY",
    "OBJECTIVE_CONTEXT",
    "RUN_CONTEXT",
    "HAND_CARD",
    "DRAW_PREVIEW_CARD",
    "DISCARD_CARD",
    "EXHAUST_CARD",
    "PLAY_PILE_CARD",
    "DECK_CARD",
    "RELIC",
    "POTION",
    "ENEMY_CORE",
    "ENEMY_POWER",
    "ENEMY_INTENT",
    "ENEMY_REACTIVE_TRAIT",
    "ENEMY_PHASE_RULE",
    "ROUTE_NODE",
    "ROUTE_SUMMARY_TOKEN",
    "SHOP_ITEM_LOCAL",
    "UPGRADE_PREVIEW_LOCAL",
    "CARD_REWARD_LOCAL",
    "COMBAT_CANDIDATE",
    "BUILD_CANDIDATE",
    "SELECTION_CANDIDATE",
    "ROUTE_CANDIDATE",
    "TARGET_LOCAL",
    "SOURCE_CARD_LOCAL",
    "SOURCE_POTION_LOCAL",
    "SELECTION_POOL_CARD_LOCAL",
    "REWARD_LOCAL",
    "SELECTION_OPERATOR_LOCAL",
    "SELECTION_SEMANTICS_LOCAL",
    "PREVIEW_RESULT_LOCAL",
    "PLAYER_STATE_LOCAL",
    "ENERGY_CONTEXT_LOCAL",
    "DRAW_CONTEXT_LOCAL",
    "DISCARD_CONTEXT_LOCAL",
    "EXHAUST_CONTEXT_LOCAL",
    "PLAY_PILE_CONTEXT_LOCAL",
    "RELIC_TRIGGER_LOCAL",
    "POTION_OPTION_LOCAL",
    "DRAW_BINDING_LOCAL",
    "DISCARD_BINDING_LOCAL",
    "EXHAUST_BINDING_LOCAL",
    "PLAY_BINDING_LOCAL",
    "CYCLE_PLAN_LOCAL",
    "RELIC_POTION_GRAPH_LOCAL",
    "ENERGY_BUDGET_LOCAL",
    "TARGET_REACTION_LOCAL",
    "BUILD_STATE_LOCAL",
    "DECK_SYNERGY_LOCAL",
    "SHOP_ECON_LOCAL",
    "ROUTE_RISK_LOCAL",
    "ROUTE_VALUE_LOCAL",
    # v3 additions:
    # - POWER_SLOT_PLAYER / POWER_SLOT_ENEMY are split from the inline
    #   power numerics previously stuffed into PLAYER_SURVIVAL / ENEMY_POWER.
    #   One token per power instance. owner_id encodes which creature it
    #   belongs to (player or enemy_base+combat_idx).
    # - CARD_KEYWORD_SLOT surfaces retain/ethereal/exhaust/innate/etc. as
    #   first-class tokens instead of bitflags buried inside card numerics.
    "POWER_SLOT_PLAYER",
    "POWER_SLOT_ENEMY",
    "CARD_KEYWORD_SLOT",
    # v4 additions (Phase 8 Tier 1):
    # - HISTORY_STEP_DETAIL is one-per-recent-env.step record. Numeric
    #   block packs family one-hot, semantic_role bitmap, target scope,
    #   step_offset, same_turn/encounter/floor flags, and scalar deltas
    #   (enemy_hp_delta, player_hp_delta, block_delta, energy_delta).
    # - HISTORY_TURN_SUMMARY is one-per-completed-combat-turn aggregate.
    #   Numeric block packs attack/skill/power counts, total damage /
    #   block / hp-lost, end-of-turn strength/dex/focus, key-power-card
    #   bitmap, and outcome flags.
    # Both share the HISTORY role + HISTORY zone so the new 7th world
    # bank ("history" in omni_attention_policy.WORLD_BANK_NAMES) routes
    # them cleanly without also sweeping in entity-hosted tokens.
    "HISTORY_STEP_DETAIL",
    "HISTORY_TURN_SUMMARY",
    # v5: append-only so old token_type ids stay stable for warm-starts.
    "HAND_MUTATION_LOCAL",
    "HAND_MUTATION_TARGET_LOCAL",
    "POST_HAND_PREVIEW_LOCAL",
    # v6: action-local counterfactuals for loop-aware combat policy.
    "CARD_FLOW_COUNTERFACTUAL_LOCAL",
    "END_TURN_HAND_FLOW_LOCAL",
    "ENERGY_CHAIN_LOCAL",
]
TOKEN_TYPE_TO_ID = {name: idx for idx, name in enumerate(TOKEN_TYPES)}
NUM_TOKEN_TYPES = len(TOKEN_TYPES)

TOKEN_ROLES = [
    "NONE",
    "WORLD",
    "PLAYER_STATE",
    "RESOURCE",
    "THREAT",
    "OBJECTIVE",
    "RUN_MEMORY",
    "HAND_CARD",
    "DRAW_PILE",
    "DISCARD_PILE",
    "EXHAUST_PILE",
    "PLAY_PILE",
    "DECK_CARD",
    "RELIC_SUPPORT",
    "POTION_SUPPORT",
    "PILE_LINK",
    "CYCLE_PLAN",
    "ENERGY_BUDGET",
    "SUPPORT_GRAPH",
    "ENEMY_CORE",
    "ENEMY_INTENT",
    "ENEMY_POWER",
    "ENEMY_TRAIT",
    "ENEMY_REACTION",
    "ROUTE_SUMMARY",
    "ROUTE_NODE",
    "QUERY_COMBAT",
    "QUERY_BUILD",
    "QUERY_SELECTION",
    "QUERY_ROUTE",
    "SOURCE_CARD",
    "SOURCE_POTION",
    "SELECTION_POOL",
    "SELECTION_OPERATOR",
    "SELECTION_SEMANTICS",
    "PREVIEW_RESULT",
    "TARGET",
    "REWARD_OPTION",
    "SHOP_OPTION",
    "UPGRADE_OPTION",
    "BUILD_STATE",
    "DECK_SYNERGY",
    "SHOP_ECON",
    "ROUTE_RISK",
    "ROUTE_VALUE",
    # v3: dedicated role for POWER_SLOT / CARD_KEYWORD tokens so the
    # eventual POWER bank can top-k-route exclusively to them.
    "POWER_SLOT",
    "CARD_KEYWORD",
    # v4: single HISTORY role shared by step-detail + turn-summary
    # tokens. The token_type bit separates the two fine-grained views;
    # routing by role is what lets the dedicated "history" bank collect
    # them exclusively.
    "HISTORY",
]
TOKEN_ROLE_TO_ID = {name: idx for idx, name in enumerate(TOKEN_ROLES)}
MAX_ROLE_ID = len(TOKEN_ROLES) - 1

TOKEN_ZONES = [
    "NONE",
    "WORLD",
    "PLAYER",
    "HAND",
    "DRAW",
    "DISCARD",
    "EXHAUST",
    "PLAY",
    "DECK",
    "RELIC",
    "POTION",
    "ENEMY",
    "ROUTE",
    "SHOP",
    "REWARD",
    "UPGRADE",
    "SELECTION",
    # v4: dedicated zone for HISTORY tokens. Lets the world_bank_router
    # pick up history tokens via zone==HISTORY AND/OR role==HISTORY 鈥?
    # the bank definition in omni_attention_policy uses role-only so
    # adding this zone entry doesn't sweep any existing tokens into the
    # history bank.
    "HISTORY",
]
TOKEN_ZONE_TO_ID = {name: idx for idx, name in enumerate(TOKEN_ZONES)}
MAX_ZONE_ID = len(TOKEN_ZONES) - 1


# ---------------------------------------------------------------------------
# Power effect-algebra table (Phase 6.0)
# ---------------------------------------------------------------------------
#
# Canonical STS2 powers have known multiplicative/additive effects on
# damage dealt / damage taken / block / draw / energy / stacks. Encoding
# those coefficients directly into each POWER_SLOT token's numeric
# vector lets attention LEARN interactions (e.g. "this card's damage
# input 脳 target's incoming_dmg_mult") without having to memorize
# the combinatorial lookup table.
#
# All coefficients are *additive deltas from neutral 1.0/0.0*. Neutral
# tokens (PAD, unknown) read zeros.
#
# Fields per power (indexed slot in the numeric vector):
#   0:  damage_mult_given     鈥?multiplier on damage THIS creature deals
#                                (e.g. Weak: -0.25; no effect: 0.0)
#   1:  damage_flat_given     鈥?additive damage per hit (Strength: +1/stack)
#   2:  damage_mult_received  鈥?multiplier on damage THIS creature receives
#                                (Vulnerable: +0.5; Intangible: -0.75)
#   3:  damage_flat_received  鈥?additive damage reduction (Buffer: absorb)
#   4:  block_mult_given      鈥?multiplier on block THIS creature grants
#                                (Frail: -0.25)
#   5:  block_flat_given      鈥?additive block (Dexterity: +1/stack)
#   6:  block_persistent      鈥?1.0 if block persists across turns
#                                (Barricade)
#   7:  end_of_turn_dmg_self  鈥?self-damage at turn end (Poison on owner
#                                鈫?stacks dealt to self each enemy turn)
#   8:  end_of_turn_dmg_given 鈥?damage dealt at turn end to attackers
#                                (Thorns, Plated Armor)
#   9:  stacks_on_applied     鈥?counter (1=stacks accumulate, 0=duration)
#   10: decays_each_turn      鈥?1 if Amount decreases each owner turn (most
#                                debuffs: Vulnerable, Weak, Frail, Poison)
#   11: is_buff               鈥?classification (1 buff, 0 debuff/neutral)
#   12: is_debuff             鈥?(1 debuff, 0 buff/neutral)
#
# All unmentioned powers default to zeros (no algebraic effect
# surfaced). Attention can still learn from their power_id bucket + text.

_POWER_ALGEBRA_DIM = 13
_POWER_ALGEBRA: dict[str, tuple[float, ...]] = {
    # Core debuffs
    "VULNERABLE_POWER": (0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0),
    "WEAK_POWER":       (-0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0),
    "FRAIL_POWER":      (0.0, 0.0, 0.0, 0.0, -0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0),
    "POISON_POWER":     (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0),
    # Core buffs
    "STRENGTH_POWER":   (0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0),
    "DEXTERITY_POWER":  (0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0),
    "INTANGIBLE_POWER": (0.0, 0.0, -0.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0),
    "METALLICIZE_POWER":(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0),
    "THORNS_POWER":     (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0),
    "BARRICADE_POWER":  (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
    "BUFFER_POWER":     (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0),
    "ARTIFACT_POWER":   (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0),
    # Enemy buffs
    "RITUAL_POWER":     (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0),
    "CURL_UP_POWER":    (0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0),
    "WEBBED_POWER":     (-0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0),
    # Rocket / map-specific (downstream fork)
    "BACK_ATTACK_LEFT_POWER":  (0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
    "BACK_ATTACK_RIGHT_POWER": (0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
    "SURROUNDED_POWER":        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
}


def _power_algebra(power_id: str) -> tuple[float, ...]:
    """Return the 13-dim effect-algebra vector for a canonical power id.

    Unknown/mod powers return all-zeros so attention falls back to the
    power_id bucket + text features alone 鈥?safe degradation, no crash.
    """
    if not power_id:
        return (0.0,) * _POWER_ALGEBRA_DIM
    return _POWER_ALGEBRA.get(
        power_id.upper(),
        (0.0,) * _POWER_ALGEBRA_DIM,
    )


def _power_id_bucket(power_id: str) -> int:
    """Stable categorical bucket for a power_id.

    Canonical powers get fixed low buckets (0..N-1, sorted by id).
    Unknown/mod powers hash into the tail region so attention still
    distinguishes them without colliding with canonicals.
    """
    if not power_id:
        return 0
    pid = power_id.upper()
    # Canonical powers occupy buckets 1..len(_POWER_ALGEBRA) 鈥?reserve 0
    # for PAD / unknown-collision.
    canonical_order = sorted(_POWER_ALGEBRA.keys())
    if pid in _POWER_ALGEBRA:
        return 1 + canonical_order.index(pid)
    # Non-canonical: hash into tail of bucket space.
    tail_start = 1 + len(_POWER_ALGEBRA)
    tail_size = max(POWER_ID_BUCKETS - tail_start, 1)
    return tail_start + (_stable_bucket(pid) % tail_size)


def _compress_numeric(values: np.ndarray | list[float] | tuple[float, ...], out_dim: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    out = np.zeros(out_dim, dtype=np.float32)
    if arr.size == 0:
        return out
    if arr.size <= out_dim:
        out[: arr.size] = arr
        return out
    chunks = np.array_split(arr, out_dim)
    out[:] = np.asarray([float(chunk.mean()) if chunk.size else 0.0 for chunk in chunks], dtype=np.float32)
    return out


def _compress_text_embedding(embedding: np.ndarray | list[float] | None) -> np.ndarray:
    out = np.zeros(TOKEN_TEXT_DIM, dtype=np.float32)
    if embedding is None:
        return out
    arr = np.asarray(embedding, dtype=np.float32).reshape(-1)
    if arr.size == TOKEN_TEXT_DIM:
        out[:] = arr
        return out
    if arr.size != TEXT_DIM:
        return out
    out[:] = arr.reshape(TOKEN_TEXT_DIM, -1).mean(axis=1)
    return out


def _stable_bucket(value: Any, buckets: int = ENTITY_HASH_BUCKETS) -> int:
    text = str(value or "").strip().lower()
    if not text:
        return 0
    h = 2166136261
    for byte in text.encode("utf-8"):
        h ^= int(byte)
        h = (h * 16777619) & 0xFFFFFFFF
    return 1 + (h % max(buckets - 1, 1))


def _owner_for_pile(token_type: str) -> int:
    return {
        "HAND_CARD": OWNER_HAND,
        "DRAW_PREVIEW_CARD": OWNER_DRAW,
        "DISCARD_CARD": OWNER_DISCARD,
        "EXHAUST_CARD": OWNER_EXHAUST,
        "PLAY_PILE_CARD": OWNER_PLAY,
        "DECK_CARD": OWNER_DECK,
    }.get(token_type, OWNER_NONE)


def _role_for_token(token_type: str) -> int:
    role_name = {
        "CLS_WORLD": "WORLD",
        "CLS_COMBAT": "WORLD",
        "CLS_BUILD": "WORLD",
        "CLS_ROUTE": "WORLD",
        "PLAYER_SURVIVAL": "PLAYER_STATE",
        "PLAYER_STATE_LOCAL": "PLAYER_STATE",
        "RESOURCE_BUDGET": "RESOURCE",
        "ENERGY_CONTEXT_LOCAL": "RESOURCE",
        "THREAT_SUMMARY": "THREAT",
        "OBJECTIVE_CONTEXT": "OBJECTIVE",
        "RUN_CONTEXT": "RUN_MEMORY",
        "HAND_CARD": "HAND_CARD",
        "SOURCE_CARD_LOCAL": "SOURCE_CARD",
        "SELECTION_POOL_CARD_LOCAL": "SELECTION_POOL",
        "CARD_REWARD_LOCAL": "REWARD_OPTION",
        "DRAW_PREVIEW_CARD": "DRAW_PILE",
        "DRAW_CONTEXT_LOCAL": "DRAW_PILE",
        "DISCARD_CARD": "DISCARD_PILE",
        "DISCARD_CONTEXT_LOCAL": "DISCARD_PILE",
        "EXHAUST_CARD": "EXHAUST_PILE",
        "EXHAUST_CONTEXT_LOCAL": "EXHAUST_PILE",
        "PLAY_PILE_CARD": "PLAY_PILE",
        "PLAY_PILE_CONTEXT_LOCAL": "PLAY_PILE",
        "PLAY_BINDING_LOCAL": "PILE_LINK",
        "DECK_CARD": "DECK_CARD",
        "BUILD_STATE_LOCAL": "BUILD_STATE",
        "DECK_SYNERGY_LOCAL": "DECK_SYNERGY",
        "RELIC": "RELIC_SUPPORT",
        "RELIC_TRIGGER_LOCAL": "RELIC_SUPPORT",
        "POTION": "POTION_SUPPORT",
        "POTION_OPTION_LOCAL": "POTION_SUPPORT",
        "DRAW_BINDING_LOCAL": "PILE_LINK",
        "DISCARD_BINDING_LOCAL": "PILE_LINK",
        "EXHAUST_BINDING_LOCAL": "PILE_LINK",
        "CYCLE_PLAN_LOCAL": "CYCLE_PLAN",
        "RELIC_POTION_GRAPH_LOCAL": "SUPPORT_GRAPH",
        "ENERGY_BUDGET_LOCAL": "ENERGY_BUDGET",
        "SOURCE_POTION_LOCAL": "SOURCE_POTION",
        "ENEMY_CORE": "ENEMY_CORE",
        "ENEMY_INTENT": "ENEMY_INTENT",
        "ENEMY_POWER": "ENEMY_POWER",
        "ENEMY_REACTIVE_TRAIT": "ENEMY_TRAIT",
        "ENEMY_PHASE_RULE": "ENEMY_TRAIT",
        "TARGET_REACTION_LOCAL": "ENEMY_REACTION",
        "HAND_MUTATION_LOCAL": "SOURCE_CARD",
        "HAND_MUTATION_TARGET_LOCAL": "HAND_CARD",
        "POST_HAND_PREVIEW_LOCAL": "PREVIEW_RESULT",
        "CARD_FLOW_COUNTERFACTUAL_LOCAL": "CYCLE_PLAN",
        "END_TURN_HAND_FLOW_LOCAL": "CYCLE_PLAN",
        "ENERGY_CHAIN_LOCAL": "ENERGY_BUDGET",
        "COMBAT_CANDIDATE": "QUERY_COMBAT",
        "BUILD_CANDIDATE": "QUERY_BUILD",
        "SELECTION_CANDIDATE": "QUERY_SELECTION",
        "ROUTE_CANDIDATE": "QUERY_ROUTE",
        "SELECTION_OPERATOR_LOCAL": "SELECTION_OPERATOR",
        "SELECTION_SEMANTICS_LOCAL": "SELECTION_SEMANTICS",
        "PREVIEW_RESULT_LOCAL": "PREVIEW_RESULT",
        "TARGET_LOCAL": "TARGET",
        "REWARD_LOCAL": "REWARD_OPTION",
        "SHOP_ITEM_LOCAL": "SHOP_OPTION",
        "SHOP_ECON_LOCAL": "SHOP_ECON",
        "UPGRADE_PREVIEW_LOCAL": "UPGRADE_OPTION",
        "ROUTE_SUMMARY_TOKEN": "ROUTE_SUMMARY",
        "ROUTE_NODE": "ROUTE_NODE",
        "ROUTE_RISK_LOCAL": "ROUTE_RISK",
        "ROUTE_VALUE_LOCAL": "ROUTE_VALUE",
        # v3:
        "POWER_SLOT_PLAYER": "POWER_SLOT",
        "POWER_SLOT_ENEMY": "POWER_SLOT",
        "CARD_KEYWORD_SLOT": "CARD_KEYWORD",
        # v4 (Phase 8 Tier 1): HISTORY role is shared between step-detail
        # and turn-summary tokens so the single "history" bank picks them
        # both up via role-filter; the token_type one-hot separates them
        # in the EntityTokenEmbedder's type embedding.
        "HISTORY_STEP_DETAIL": "HISTORY",
        "HISTORY_TURN_SUMMARY": "HISTORY",
    }.get(token_type, "NONE")
    return TOKEN_ROLE_TO_ID[role_name]


def _zone_for_token(token_type: str) -> int:
    zone_name = {
        "CLS_WORLD": "WORLD",
        "CLS_COMBAT": "WORLD",
        "CLS_BUILD": "WORLD",
        "CLS_ROUTE": "WORLD",
        "PLAYER_SURVIVAL": "PLAYER",
        "PLAYER_STATE_LOCAL": "PLAYER",
        "RESOURCE_BUDGET": "PLAYER",
        "ENERGY_CONTEXT_LOCAL": "PLAYER",
        "THREAT_SUMMARY": "WORLD",
        "OBJECTIVE_CONTEXT": "WORLD",
        "RUN_CONTEXT": "WORLD",
        "HAND_CARD": "HAND",
        "SOURCE_CARD_LOCAL": "HAND",
        "SELECTION_POOL_CARD_LOCAL": "SELECTION",
        "DRAW_PREVIEW_CARD": "DRAW",
        "DRAW_CONTEXT_LOCAL": "DRAW",
        "DRAW_BINDING_LOCAL": "DRAW",
        "DISCARD_CARD": "DISCARD",
        "DISCARD_CONTEXT_LOCAL": "DISCARD",
        "DISCARD_BINDING_LOCAL": "DISCARD",
        "EXHAUST_CARD": "EXHAUST",
        "EXHAUST_CONTEXT_LOCAL": "EXHAUST",
        "EXHAUST_BINDING_LOCAL": "EXHAUST",
        "PLAY_PILE_CARD": "PLAY",
        "PLAY_PILE_CONTEXT_LOCAL": "PLAY",
        "PLAY_BINDING_LOCAL": "PLAY",
        "DECK_CARD": "DECK",
        "RELIC": "RELIC",
        "RELIC_TRIGGER_LOCAL": "RELIC",
        "POTION": "POTION",
        "POTION_OPTION_LOCAL": "POTION",
        "SOURCE_POTION_LOCAL": "POTION",
        "ENERGY_BUDGET_LOCAL": "PLAYER",
        "CYCLE_PLAN_LOCAL": "PLAYER",
        "RELIC_POTION_GRAPH_LOCAL": "PLAYER",
        "ENEMY_CORE": "ENEMY",
        "ENEMY_INTENT": "ENEMY",
        "ENEMY_POWER": "ENEMY",
        "ENEMY_REACTIVE_TRAIT": "ENEMY",
        "ENEMY_PHASE_RULE": "ENEMY",
        "TARGET_LOCAL": "ENEMY",
        "TARGET_REACTION_LOCAL": "ENEMY",
        "HAND_MUTATION_LOCAL": "HAND",
        "HAND_MUTATION_TARGET_LOCAL": "HAND",
        "POST_HAND_PREVIEW_LOCAL": "HAND",
        "CARD_FLOW_COUNTERFACTUAL_LOCAL": "HAND",
        "END_TURN_HAND_FLOW_LOCAL": "HAND",
        "ENERGY_CHAIN_LOCAL": "PLAYER",
        "COMBAT_CANDIDATE": "HAND",
        "BUILD_CANDIDATE": "REWARD",
        "SELECTION_CANDIDATE": "SELECTION",
        "ROUTE_CANDIDATE": "ROUTE",
        "SELECTION_OPERATOR_LOCAL": "SELECTION",
        "SELECTION_SEMANTICS_LOCAL": "SELECTION",
        "PREVIEW_RESULT_LOCAL": "SELECTION",
        "REWARD_LOCAL": "REWARD",
        "CARD_REWARD_LOCAL": "REWARD",
        "SHOP_ITEM_LOCAL": "SHOP",
        "SHOP_ECON_LOCAL": "SHOP",
        "UPGRADE_PREVIEW_LOCAL": "UPGRADE",
        "BUILD_STATE_LOCAL": "PLAYER",
        "DECK_SYNERGY_LOCAL": "DECK",
        "ROUTE_SUMMARY_TOKEN": "ROUTE",
        "ROUTE_NODE": "ROUTE",
        "ROUTE_RISK_LOCAL": "ROUTE",
        "ROUTE_VALUE_LOCAL": "ROUTE",
        # v3:
        "POWER_SLOT_PLAYER": "PLAYER",
        "POWER_SLOT_ENEMY": "ENEMY",
        "CARD_KEYWORD_SLOT": "HAND",
        # v4 (Phase 8 Tier 1): dedicated HISTORY zone. Role-only bank
        # filter in omni_attention_policy means this zone tag is more
        # documentation than routing, but it's cheap insurance against
        # any downstream code that probes by zone_id.
        "HISTORY_STEP_DETAIL": "HISTORY",
        "HISTORY_TURN_SUMMARY": "HISTORY",
    }.get(token_type, "NONE")
    return TOKEN_ZONE_TO_ID[zone_name]
