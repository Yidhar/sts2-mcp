"""Token-world observation encoder for the omni-attention online policy."""

from __future__ import annotations

from typing import Any

import numpy as np

from content_registry import (
    build_live_enemy_semantic_text,
    build_live_potion_semantic_text,
    build_live_relic_semantic_text,
    get_enemy_metadata,
)

from . import observation_common as obs_common
from .boss_mechanics import build_boss_mechanics_context, enemy_mechanics_key
from .hand_mutation import (
    infer_hand_mutation,
    mutation_summary_numeric,
    mutation_target_numeric,
    post_hand_preview_numeric,
)
from .potion_profiles import (
    DEFAULT_EFFECT_PROFILE,
    get_potion_profile as _get_potion_profile,
    potion_is_enabled_for_training as _potion_enabled,
)
from .text_encoder import TEXT_DIM


def _resolve_potion_effect(potion: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """Merge bridge live effect_profile with the Python registry (bridge wins).

    Returns (profile_entry, effect_profile_dict) — profile_entry includes
    effect_family/timing_tags/etc; effect_profile_dict is the numeric slots.
    """
    if not isinstance(potion, dict):
        return ({}, dict(DEFAULT_EFFECT_PROFILE))
    pid = str(potion.get("id") or "").strip()
    registry_entry = _get_potion_profile(pid) if pid else {}
    base_effect = dict(DEFAULT_EFFECT_PROFILE)
    base_effect.update(registry_entry.get("effect_profile") or {})
    bridge_effect = potion.get("effect_profile")
    if isinstance(bridge_effect, dict):
        base_effect.update({k: v for k, v in bridge_effect.items() if v is not None})
    merged_entry = dict(registry_entry)
    for key in ("effect_family", "semantic_tags", "timing_tags", "training_tags", "target_scope"):
        live_val = potion.get(key)
        if live_val:
            merged_entry[key] = live_val
    if "enabled_for_training" in potion:
        merged_entry["enabled_for_training"] = bool(potion["enabled_for_training"])
    return (merged_entry, base_effect)

MAX_ACTIONS = obs_common.MAX_ACTIONS
# Phase 6 (attention_obs_v3): +64 tokens for dedicated POWER_SLOT /
# CARD_KEYWORD tokens, split out of the entity numeric inlining. At typical
# STS2 scale we see <= 35 active powers (player + 4 enemies 脳 5 powers) and
# a handful of card-keyword tokens; 64 is comfortable headroom.
# Phase 8 Tier 1 (attention_obs_v4): +28 HISTORY tokens (20 step-detail +
# 8 turn-summary) so the policy can reason about "what did I just do"
# without needing a recurrent architecture. See action_history.py.
MAX_WORLD_TOKENS = 412
MAX_CANDIDATE_LOCAL_TOKENS = 32
TOKEN_NUMERIC_DIM = 96
TOKEN_TEXT_DIM = 64
TOKEN_FEAT_DIM = TOKEN_NUMERIC_DIM + TOKEN_TEXT_DIM
ENTITY_HASH_BUCKETS = 8192
MAX_OWNER_ID = 127
MAX_ORDER_ID = 63
# v4: HISTORY_STEP_DETAIL / HISTORY_TURN_SUMMARY token types exist.
# v3 checkpoints load with strict=False; new history-specific embeddings
# zero-init. See _design_phase8_history.md for the migration plan.
OBSERVATION_API_VERSION = "attention_obs_v4"

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
from .action_history import (
    KEY_POWER_CARD_BUCKETS,
    MAX_HISTORY_TOKENS,
    MAX_STEP_DETAIL_TOKENS,
    MAX_TURN_SUMMARY_TOKENS,
    NUM_KEY_POWER_FLAGS,
    NUM_SEMANTIC_ROLES,
)

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


class WorldTokenObservationEncoder(obs_common.DenseObservationEncoder):
    """Observation V3 built as a tokenized world memory plus candidate tokens."""

    def __init__(self, use_text: bool = True, text_device: str = "cpu"):
        super().__init__(use_text=use_text, text_device=text_device)
        # Pre-allocate output buffers for _materialize_entries / _materialize_nested_entries
        # to avoid ~3 MB of np.zeros allocation every encode() call.
        self._buf_world = self._alloc_flat_bufs(MAX_WORLD_TOKENS)
        self._buf_candidate = self._alloc_flat_bufs(MAX_ACTIONS)
        self._buf_candidate_local = self._alloc_nested_bufs(MAX_ACTIONS, MAX_CANDIDATE_LOCAL_TOKENS)
        self._current_planner_context: dict[str, Any] | None = None
        self._current_boss_context: dict[str, Any] | None = None

    @staticmethod
    def _alloc_flat_bufs(n):
        return {
            "tokens": np.zeros((n, TOKEN_FEAT_DIM), dtype=np.float32),
            "mask": np.zeros(n, dtype=np.float32),
            "type_ids": np.zeros(n, dtype=np.int32),
            "role_ids": np.zeros(n, dtype=np.int32),
            "owner_ids": np.zeros(n, dtype=np.int32),
            "entity_ids": np.zeros(n, dtype=np.int32),
            "zone_ids": np.zeros(n, dtype=np.int32),
            "order_ids": np.zeros(n, dtype=np.int32),
            "target_owner_ids": np.zeros(n, dtype=np.int32),
            "target_entity_ids": np.zeros(n, dtype=np.int32),
        }

    @staticmethod
    def _alloc_nested_bufs(n_outer, n_inner):
        return {
            "tokens": np.zeros((n_outer, n_inner, TOKEN_FEAT_DIM), dtype=np.float32),
            "mask": np.zeros((n_outer, n_inner), dtype=np.float32),
            "type_ids": np.zeros((n_outer, n_inner), dtype=np.int32),
            "role_ids": np.zeros((n_outer, n_inner), dtype=np.int32),
            "owner_ids": np.zeros((n_outer, n_inner), dtype=np.int32),
            "entity_ids": np.zeros((n_outer, n_inner), dtype=np.int32),
            "zone_ids": np.zeros((n_outer, n_inner), dtype=np.int32),
            "order_ids": np.zeros((n_outer, n_inner), dtype=np.int32),
        }

    @property
    def obs_space(self):
        from gymnasium import spaces

        inf = np.inf
        return spaces.Dict(
            {
                "world_tokens": spaces.Box(-inf, inf, (MAX_WORLD_TOKENS, TOKEN_FEAT_DIM), dtype=np.float32),
                "world_token_mask": spaces.Box(0, 1, (MAX_WORLD_TOKENS,), dtype=np.float32),
                "world_token_type_ids": spaces.Box(0, NUM_TOKEN_TYPES, (MAX_WORLD_TOKENS,), dtype=np.int32),
                "world_token_role_ids": spaces.Box(0, MAX_ROLE_ID, (MAX_WORLD_TOKENS,), dtype=np.int32),
                "world_entity_owner_ids": spaces.Box(0, MAX_OWNER_ID, (MAX_WORLD_TOKENS,), dtype=np.int32),
                "world_token_entity_ids": spaces.Box(0, ENTITY_HASH_BUCKETS, (MAX_WORLD_TOKENS,), dtype=np.int32),
                "world_token_zone_ids": spaces.Box(0, MAX_ZONE_ID, (MAX_WORLD_TOKENS,), dtype=np.int32),
                "world_token_order_ids": spaces.Box(0, MAX_ORDER_ID, (MAX_WORLD_TOKENS,), dtype=np.int32),
                "candidate_query_tokens": spaces.Box(-inf, inf, (MAX_ACTIONS, TOKEN_FEAT_DIM), dtype=np.float32),
                "candidate_query_type_ids": spaces.Box(0, NUM_TOKEN_TYPES, (MAX_ACTIONS,), dtype=np.int32),
                "candidate_query_role_ids": spaces.Box(0, MAX_ROLE_ID, (MAX_ACTIONS,), dtype=np.int32),
                "candidate_query_owner_ids": spaces.Box(0, MAX_OWNER_ID, (MAX_ACTIONS,), dtype=np.int32),
                "candidate_query_entity_ids": spaces.Box(0, ENTITY_HASH_BUCKETS, (MAX_ACTIONS,), dtype=np.int32),
                "candidate_query_zone_ids": spaces.Box(0, MAX_ZONE_ID, (MAX_ACTIONS,), dtype=np.int32),
                "candidate_query_order_ids": spaces.Box(0, MAX_ORDER_ID, (MAX_ACTIONS,), dtype=np.int32),
                "candidate_query_target_owner_ids": spaces.Box(0, MAX_OWNER_ID, (MAX_ACTIONS,), dtype=np.int32),
                "candidate_query_target_entity_ids": spaces.Box(0, ENTITY_HASH_BUCKETS, (MAX_ACTIONS,), dtype=np.int32),
                "candidate_local_tokens": spaces.Box(-inf, inf, (MAX_ACTIONS, MAX_CANDIDATE_LOCAL_TOKENS, TOKEN_FEAT_DIM), dtype=np.float32),
                "candidate_local_masks": spaces.Box(0, 1, (MAX_ACTIONS, MAX_CANDIDATE_LOCAL_TOKENS), dtype=np.float32),
                "candidate_local_type_ids": spaces.Box(0, NUM_TOKEN_TYPES, (MAX_ACTIONS, MAX_CANDIDATE_LOCAL_TOKENS), dtype=np.int32),
                "candidate_local_role_ids": spaces.Box(0, MAX_ROLE_ID, (MAX_ACTIONS, MAX_CANDIDATE_LOCAL_TOKENS), dtype=np.int32),
                "candidate_local_owner_ids": spaces.Box(0, MAX_OWNER_ID, (MAX_ACTIONS, MAX_CANDIDATE_LOCAL_TOKENS), dtype=np.int32),
                "candidate_local_entity_ids": spaces.Box(0, ENTITY_HASH_BUCKETS, (MAX_ACTIONS, MAX_CANDIDATE_LOCAL_TOKENS), dtype=np.int32),
                "candidate_local_zone_ids": spaces.Box(0, MAX_ZONE_ID, (MAX_ACTIONS, MAX_CANDIDATE_LOCAL_TOKENS), dtype=np.int32),
                "candidate_local_order_ids": spaces.Box(0, MAX_ORDER_ID, (MAX_ACTIONS, MAX_CANDIDATE_LOCAL_TOKENS), dtype=np.int32),
                "action_mask": spaces.Box(0, 1, (MAX_ACTIONS,), dtype=np.float32),
            }
        )

    def encode(
        self,
        obs: dict | None,
        legal_actions: list | None = None,
        planner_context: dict | None = None,
    ) -> dict[str, np.ndarray]:
        obs_dict = obs or {}
        action_list = legal_actions or []
        planner_context = self._normalize_planner_context(obs_dict, action_list, planner_context)
        world_entries: list[dict[str, Any]] = []
        candidate_entries: list[dict[str, Any]] = []
        candidate_local_entries: list[list[dict[str, Any]]] = [[] for _ in range(MAX_ACTIONS)]
        action_mask = np.zeros(MAX_ACTIONS, dtype=np.float32)

        try:
            self._begin_text_registry()
            features = self._build_feature_view(obs_dict, action_list, planner_context)
            self._current_planner_context = planner_context
            self._current_boss_context = build_boss_mechanics_context(obs_dict)
            self._append_global_tokens(world_entries, obs_dict, features, planner_context)
            self._append_entity_tokens(world_entries, obs_dict, features, planner_context)
            # v3: POWER_SLOT tokens split out of entity-inlined numerics.
            # One token per power instance on player + each enemy. Emits
            # (power_id bucket as entity_id) + (effect-algebra vector in
            # numeric slots 0..12) + (amount scalars in slots 13..15).
            self._append_power_slot_tokens(world_entries, obs_dict)
            # v3: CARD_KEYWORD_SLOT tokens for Retain/Ethereal/Exhaust/etc.
            # One token per keyword per hand card, surfaces keyword binding
            # to the POWER bank where attention can learn the interaction
            # patterns ("Ethereal + turn ending + not playable = waste").
            self._append_card_keyword_slot_tokens(world_entries, obs_dict)
            # v4 (Phase 8 Tier 1): HISTORY tokens. 20 step-detail slots +
            # 8 turn-summary slots, populated from env_v2's
            # ActionHistoryTracker snapshot attached under
            # obs["_action_history"]. Padded tokens carry is_empty=1.
            self._append_history_tokens(world_entries, obs_dict)
            self._append_candidate_tokens(candidate_entries, candidate_local_entries, action_mask, obs_dict, features, action_list)
            self._resolve_entry_text_embeddings(world_entries, candidate_entries, candidate_local_entries)
            self._resolve_text_registry()

            (
                world_tokens,
                world_token_mask,
                world_token_type_ids,
                world_token_role_ids,
                world_owner_ids,
                world_entity_ids,
                world_zone_ids,
                world_order_ids,
                _world_target_owner_ids,
                _world_target_entity_ids,
            ) = self._materialize_entries(world_entries, MAX_WORLD_TOKENS, bufs=self._buf_world)
            (
                candidate_query_tokens,
                _unused_mask,
                candidate_query_type_ids,
                candidate_query_role_ids,
                candidate_query_owner_ids,
                candidate_query_entity_ids,
                candidate_query_zone_ids,
                candidate_query_order_ids,
                candidate_query_target_owner_ids,
                candidate_query_target_entity_ids,
            ) = self._materialize_entries(candidate_entries, MAX_ACTIONS, bufs=self._buf_candidate)
            (
                candidate_local_tokens,
                candidate_local_masks,
                candidate_local_type_ids,
                candidate_local_role_ids,
                candidate_local_owner_ids,
                candidate_local_entity_ids,
                candidate_local_zone_ids,
                candidate_local_order_ids,
            ) = self._materialize_nested_entries(candidate_local_entries, MAX_ACTIONS, MAX_CANDIDATE_LOCAL_TOKENS, bufs=self._buf_candidate_local)

            return {
                "world_tokens": world_tokens,
                "world_token_mask": world_token_mask,
                "world_token_type_ids": world_token_type_ids,
                "world_token_role_ids": world_token_role_ids,
                "world_entity_owner_ids": world_owner_ids,
                "world_token_entity_ids": world_entity_ids,
                "world_token_zone_ids": world_zone_ids,
                "world_token_order_ids": world_order_ids,
                "candidate_query_tokens": candidate_query_tokens,
                "candidate_query_type_ids": candidate_query_type_ids,
                "candidate_query_role_ids": candidate_query_role_ids,
                "candidate_query_owner_ids": candidate_query_owner_ids,
                "candidate_query_entity_ids": candidate_query_entity_ids,
                "candidate_query_zone_ids": candidate_query_zone_ids,
                "candidate_query_order_ids": candidate_query_order_ids,
                "candidate_query_target_owner_ids": candidate_query_target_owner_ids,
                "candidate_query_target_entity_ids": candidate_query_target_entity_ids,
                "candidate_local_tokens": candidate_local_tokens,
                "candidate_local_masks": candidate_local_masks,
                "candidate_local_type_ids": candidate_local_type_ids,
                "candidate_local_role_ids": candidate_local_role_ids,
                "candidate_local_owner_ids": candidate_local_owner_ids,
                "candidate_local_entity_ids": candidate_local_entity_ids,
                "candidate_local_zone_ids": candidate_local_zone_ids,
                "candidate_local_order_ids": candidate_local_order_ids,
                "action_mask": action_mask,
                "decision_domain": features["decision_domain"],
            }
        except Exception:
            self._clear_text_registry()
            raise
        finally:
            self._current_planner_context = None
            self._current_boss_context = None

    def _build_feature_view(
        self,
        obs: dict[str, Any],
        legal_actions: list[dict[str, Any]],
        planner_context: dict[str, Any],
    ) -> dict[str, np.ndarray]:
        scalars = np.zeros(obs_common.SCALAR_DIM, dtype=np.float32)
        decision_domain = np.zeros(obs_common.NUM_DOMAINS, dtype=np.float32)
        player_powers = np.zeros(obs_common.POWER_DIM, dtype=np.float32)
        relic_signals = np.zeros(obs_common.RELIC_SIGNAL_DIM, dtype=np.float32)
        run_memory = np.asarray(planner_context["run_memory_vector"], dtype=np.float32).copy()
        objective_context = np.asarray(planner_context["objective_context_vector"], dtype=np.float32).copy()

        if obs:
            self._enc_scalars(scalars, obs, legal_actions, planner_context)
            self._enc_decision_domain(decision_domain, obs)
            self._enc_powers(player_powers, obs)
            self._enc_relic_signals(relic_signals, obs)

        hand_cards = self._runtime_cards(obs, "hand", "hand")
        deck_cards = (obs.get("player") or {}).get("deck_cards") or []
        enemies = (obs.get("combat") or {}).get("enemies") or []
        relics = (obs.get("player") or {}).get("relics") or []
        potions = (obs.get("player") or {}).get("potions") or []

        hand, hand_text, hand_mask = self._encode_cards_with_limit(hand_cards, obs_common.MAX_HAND)
        deck, deck_text, deck_mask = self._encode_cards_with_limit(deck_cards, obs_common.MAX_DECK)
        enemies_numeric, enemy_text, enemy_mask = self._encode_enemy_view(enemies)
        relic_text, relic_mask = self._encode_support_text_view(relics, obs_common.MAX_RELICS, "relics")
        potion_text, potion_mask = self._encode_support_text_view(potions, obs_common.MAX_POTIONS, "potions")
        (
            actions,
            action_text,
            semantic_actions,
            semantic_action_text,
            route_summary,
            route_nodes,
            route_node_mask,
            action_mask,
        ) = self._encode_action_view(legal_actions, planner_context)

        return {
            "scalars": scalars,
            "decision_domain": decision_domain,
            "hand": hand,
            "hand_text": hand_text,
            "hand_mask": hand_mask,
            "deck": deck,
            "deck_text": deck_text,
            "deck_mask": deck_mask,
            "enemies": enemies_numeric,
            "enemy_text": enemy_text,
            "enemy_mask": enemy_mask,
            "player_powers": player_powers,
            "relic_signals": relic_signals,
            "run_memory": run_memory,
            "objective_context": objective_context,
            "relics": relic_text,
            "relic_mask": relic_mask,
            "potions": potion_text,
            "potion_mask": potion_mask,
            "actions": actions,
            "action_text": action_text,
            "semantic_actions": semantic_actions,
            "semantic_action_text": semantic_action_text,
            "route_summary": route_summary,
            "route_nodes": route_nodes,
            "route_node_mask": route_node_mask,
            "action_mask": action_mask,
        }

    def _encode_cards_with_limit(self, cards: list[Any], limit: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        count = min(len(cards) if isinstance(cards, list) else 0, limit)
        numeric = np.zeros((count, obs_common.CARD_FEAT_DIM), dtype=np.float32)
        text = np.zeros((count, TEXT_DIM), dtype=np.float32)
        mask = np.zeros(count, dtype=np.float32)
        if count > 0:
            self._enc_card_collection(cards[:count], numeric, text, mask)
        return numeric, text, mask

    def _encode_enemy_view(self, enemies: list[Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        count = min(len(enemies) if isinstance(enemies, list) else 0, obs_common.MAX_ENEMIES)
        numeric = np.zeros((count, obs_common.ENEMY_FEAT_DIM), dtype=np.float32)
        text = np.zeros((count, TEXT_DIM), dtype=np.float32)
        mask = np.zeros(count, dtype=np.float32)
        if count > 0:
            self._enc_enemies(numeric, text, mask, {"combat": {"enemies": enemies[:count]}})
        return numeric, text, mask

    def _encode_support_text_view(
        self,
        entries: list[Any],
        limit: int,
        kind: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        count = min(len(entries) if isinstance(entries, list) else 0, limit)
        text = np.zeros((count, TEXT_DIM), dtype=np.float32)
        mask = np.zeros(count, dtype=np.float32)
        if count > 0:
            payload = {"player": {kind: entries[:count]}}
            if kind == "relics":
                self._enc_relics(text, mask, payload)
            else:
                self._enc_potions(text, mask, payload)
        return text, mask

    def _encode_action_view(
        self,
        legal_actions: list[dict[str, Any]],
        planner_context: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        count = min(len(legal_actions), MAX_ACTIONS)
        actions = np.zeros((count, obs_common.ACTION_FEAT_DIM), dtype=np.float32)
        action_text = np.zeros((count, TEXT_DIM), dtype=np.float32)
        semantic_actions = np.zeros((count, obs_common.SEM_ACTION_FEAT_DIM), dtype=np.float32)
        semantic_action_text = np.zeros((count, TEXT_DIM), dtype=np.float32)
        route_summary = np.zeros((count, obs_common.ROUTE_SUMMARY_DIM), dtype=np.float32)
        route_nodes = np.zeros((count, obs_common.MAX_ROUTE_NODES, obs_common.ROUTE_NODE_FEAT_DIM), dtype=np.float32)
        route_node_mask = np.zeros((count, obs_common.MAX_ROUTE_NODES), dtype=np.float32)
        action_mask = np.zeros(count, dtype=np.float32)
        if count > 0:
            self._enc_actions(
                actions,
                action_text,
                semantic_actions,
                semantic_action_text,
                route_summary,
                route_nodes,
                route_node_mask,
                action_mask,
                legal_actions[:count],
                planner_context,
            )
        return actions, action_text, semantic_actions, semantic_action_text, route_summary, route_nodes, route_node_mask, action_mask

    def _resolve_entry_text_embeddings(self, *entry_groups: list[Any]) -> None:
        def _iter_entries(group: list[Any]) -> Any:
            for item in group:
                if isinstance(item, list):
                    for nested in item:
                        if isinstance(nested, dict):
                            yield nested
                elif isinstance(item, dict):
                    yield item

        for group in entry_groups:
            for entry in _iter_entries(group):
                text_embedding = entry.get("text_embedding")
                if text_embedding is not None:
                    entry["text_embedding"] = _compress_text_embedding(text_embedding)
                    entry["text"] = ""
                    continue

                raw_text = str(entry.get("text") or "").strip()
                if not self.use_text or not raw_text:
                    entry["text_embedding"] = np.zeros(TOKEN_TEXT_DIM, dtype=np.float32)
                    entry["text"] = ""
                    continue

                self._register_text_assignment(
                    raw_text,
                    lambda embedding, entry=entry: entry.update(text_embedding=np.asarray(embedding, dtype=np.float32), text=""),
                    postprocess=_compress_text_embedding,
                )

    def _append_global_tokens(
        self,
        world_entries: list[dict[str, Any]],
        obs: dict[str, Any],
        features: dict[str, np.ndarray],
        planner_context: dict[str, Any] | None = None,
    ) -> None:
        scalars = features["scalars"]
        decision_domain = features["decision_domain"]
        world_numeric = np.zeros(TOKEN_NUMERIC_DIM, dtype=np.float32)
        world_numeric[: obs_common.SCALAR_DIM] = scalars
        world_numeric[obs_common.SCALAR_DIM : obs_common.SCALAR_DIM + obs_common.NUM_DOMAINS] = decision_domain
        world_entries.append(self._entry("CLS_WORLD", world_numeric, owner_id=OWNER_NONE, entity_id=0))

        for idx, token_type in enumerate(("CLS_COMBAT", "CLS_BUILD", "CLS_ROUTE")):
            numeric = np.zeros(TOKEN_NUMERIC_DIM, dtype=np.float32)
            numeric[0] = float(decision_domain[idx] > 0.5)
            numeric[1: 1 + obs_common.NUM_DOMAINS] = decision_domain
            world_entries.append(self._entry(token_type, numeric, owner_id=OWNER_NONE, entity_id=0))

        player = obs.get("player") or {}
        combat = obs.get("combat") or {}
        enemies = combat.get("enemies") or []
        hp = obs_common._float(player.get("hp", player.get("current_hp")))
        max_hp = obs_common._float(player.get("max_hp"))
        block = obs_common._float(player.get("block"))
        incoming = sum(obs_common._float(((enemy or {}).get("intent") or {}).get("total_damage")) for enemy in enemies if isinstance(enemy, dict))

        survival = np.zeros(TOKEN_NUMERIC_DIM, dtype=np.float32)
        survival[0] = min(hp / max(max_hp, 1.0), 1.0) if max_hp > 0 else 0.0
        survival[1] = obs_common._log_norm(hp, obs_common._LOG1P_200)
        survival[2] = obs_common._log_norm(block, obs_common._LOG1P_200)
        survival[3] = obs_common._log_norm(incoming, obs_common._LOG1P_200)
        survival[4] = obs_common._signed_log_norm(hp + block - incoming, obs_common._LOG1P_200)
        survival[5 : 5 + min(obs_common.POWER_DIM, TOKEN_NUMERIC_DIM - 5)] = features["player_powers"][: TOKEN_NUMERIC_DIM - 5]
        world_entries.append(self._entry("PLAYER_SURVIVAL", survival, owner_id=OWNER_PLAYER, entity_id=0))

        budget = np.zeros(TOKEN_NUMERIC_DIM, dtype=np.float32)
        max_energy = obs_common._float(combat.get("max_energy"))
        budget[0] = min(obs_common._float(combat.get("energy")) / max(max_energy, 1.0), 1.0) if max_energy > 0 else 0.0
        budget[1] = min(obs_common._float(combat.get("energy")) / 10.0, 1.0)
        budget[2] = min(max_energy / 10.0, 1.0)
        budget[3] = min(obs_common._float(combat.get("stars")) / 10.0, 1.0)
        budget[4] = obs_common._log_norm(obs_common._float(player.get("gold")), obs_common._LOG1P_500)
        budget[5 : 5 + min(obs_common.RELIC_SIGNAL_DIM, TOKEN_NUMERIC_DIM - 5)] = features["relic_signals"][: TOKEN_NUMERIC_DIM - 5]
        world_entries.append(self._entry("RESOURCE_BUDGET", budget, owner_id=OWNER_PLAYER, entity_id=0))

        threat = np.zeros(TOKEN_NUMERIC_DIM, dtype=np.float32)
        threat[0] = min(len([enemy for enemy in enemies if isinstance(enemy, dict) and enemy.get("is_alive", True)]) / max(obs_common.MAX_ENEMIES, 1), 1.0)
        threat[1] = obs_common._log_norm(incoming, obs_common._LOG1P_200)
        threat[2] = obs_common._log_norm(sum(obs_common._float(enemy.get("hp", enemy.get("current_hp"))) for enemy in enemies if isinstance(enemy, dict)), obs_common._LOG1P_1200)
        threat[3] = obs_common._log_norm(sum(obs_common._float(enemy.get("block")) for enemy in enemies if isinstance(enemy, dict)), obs_common._LOG1P_200)
        threat[4] = float(any(self._infer_enemy_traits(enemy)[0] for enemy in enemies if isinstance(enemy, dict)))
        combat_memory = (planner_context or {}).get("combat_memory") or {}
        if combat_memory:
            player_max_hp = max(obs_common._float(combat_memory.get("player_max_hp"), max_hp or 1.0), 1.0)
            initial_total_hp = max(obs_common._float(combat_memory.get("initial_enemy_total_hp"), 1.0), 1.0)
            threat[5] = min(obs_common._float(combat_memory.get("turns_in_combat")) / 20.0, 1.0)
            threat[6] = obs_common._signed_log_norm(obs_common._float(combat_memory.get("player_hp_delta_last_turn")), obs_common._LOG1P_200)
            threat[7] = obs_common._signed_log_norm(obs_common._float(combat_memory.get("player_block_delta_last_turn")), obs_common._LOG1P_200)
            threat[8] = max(-1.0, min(obs_common._float(combat_memory.get("player_energy_delta_last_turn")) / 5.0, 1.0))
            enemy_hp_delta = obs_common._float(combat_memory.get("enemy_total_hp_delta_last_turn"))
            threat[9] = max(-1.0, min(enemy_hp_delta / initial_total_hp, 1.0))
            threat[10] = min(obs_common._float(combat_memory.get("surprise_damage_last_turn")) / player_max_hp, 1.0)
            threat[11] = max(-1.0, min(obs_common._float(combat_memory.get("enemy_count_delta_last_turn")) / 3.0, 1.0))
            threat[12] = min(obs_common._float(combat_memory.get("cum_surprise_damage")) / player_max_hp, 1.0)
        world_entries.append(self._entry("THREAT_SUMMARY", threat, owner_id=OWNER_NONE, entity_id=0))

        objective_numeric = np.zeros(TOKEN_NUMERIC_DIM, dtype=np.float32)
        objective_numeric[: min(obs_common.OBJECTIVE_DIM, TOKEN_NUMERIC_DIM)] = features["objective_context"][:TOKEN_NUMERIC_DIM]
        world_entries.append(self._entry("OBJECTIVE_CONTEXT", objective_numeric, owner_id=OWNER_NONE, entity_id=0))
        run_numeric = np.zeros(TOKEN_NUMERIC_DIM, dtype=np.float32)
        run_numeric[: min(obs_common.RUN_MEMORY_DIM, TOKEN_NUMERIC_DIM)] = features["run_memory"][:TOKEN_NUMERIC_DIM]
        world_entries.append(self._entry("RUN_CONTEXT", run_numeric, owner_id=OWNER_NONE, entity_id=0))

    def _append_entity_tokens(
        self,
        world_entries: list[dict[str, Any]],
        obs: dict[str, Any],
        features: dict[str, np.ndarray],
        planner_context: dict[str, Any] | None = None,
    ) -> None:
        player = obs.get("player") or {}
        hand_cards = self._runtime_cards(obs, "hand", "hand")
        deck_cards = player.get("deck_cards") or []
        draw_cards = self._runtime_cards(obs, "draw_pile", "draw_preview_cards")
        discard_cards = self._runtime_cards(obs, "discard_pile", "discard_cards")
        exhaust_cards = self._runtime_cards(obs, "exhaust_pile", "exhaust_cards")
        play_cards = self._runtime_cards(obs, "play_pile", "play_pile_cards")

        self._append_card_collection(world_entries, features["hand"], features["hand_text"], features["hand_mask"], "HAND_CARD", entity_keys=hand_cards)
        self._append_card_collection(world_entries, features["deck"], features["deck_text"], features["deck_mask"], "DECK_CARD", entity_keys=deck_cards)
        self._append_card_collection(world_entries, *self._encode_runtime_pile(obs, "draw_pile", "draw_preview_cards", limit=12), "DRAW_PREVIEW_CARD", entity_keys=draw_cards)
        self._append_card_collection(world_entries, *self._encode_runtime_pile(obs, "discard_pile", "discard_cards", limit=24), "DISCARD_CARD", entity_keys=discard_cards)
        self._append_card_collection(world_entries, *self._encode_runtime_pile(obs, "exhaust_pile", "exhaust_cards", limit=24), "EXHAUST_CARD", entity_keys=exhaust_cards)
        self._append_card_collection(world_entries, *self._encode_runtime_pile(obs, "play_pile", "play_pile_cards", limit=12), "PLAY_PILE_CARD", entity_keys=play_cards)

        self._append_relic_collection(world_entries, player.get("relics") or [], features["relics"], features["relic_mask"])
        self._append_potion_collection(world_entries, player.get("potions") or [], features["potions"], features["potion_mask"])

        enemies = obs.get("combat", {}).get("enemies") or []
        combat_memory = (planner_context or {}).get("combat_memory") or {}
        memory_enemies = combat_memory.get("enemies") if isinstance(combat_memory, dict) else None
        for enemy_index in range(features["enemy_mask"].shape[0]):
            if features["enemy_mask"][enemy_index] <= 0:
                continue
            owner_id = self._enemy_owner_id(enemy_index)
            enemy = enemies[enemy_index] if enemy_index < len(enemies) and isinstance(enemies[enemy_index], dict) else {}
            enemy_entity_id = _stable_bucket(self._enemy_entity_key(enemy, enemy_index))
            enemy_memory = None
            if isinstance(memory_enemies, dict):
                enemy_memory = memory_enemies.get(self._enemy_entity_key(enemy, enemy_index))
            core = np.zeros(TOKEN_NUMERIC_DIM, dtype=np.float32)
            core[: obs_common.ENEMY_FEAT_DIM] = features["enemies"][enemy_index]
            if isinstance(enemy_memory, dict):
                base = obs_common.ENEMY_FEAT_DIM
                core[base + 0] = obs_common._float(enemy_memory.get("hp_delta_last_turn_ratio"))
                core[base + 1] = obs_common._float(enemy_memory.get("hp_delta_last_3_turns_ratio"))
                core[base + 2] = obs_common._float(enemy_memory.get("block_delta_last_turn_ratio"))
                core[base + 3] = obs_common._float(enemy_memory.get("turns_alive_norm"))
                core[base + 4] = obs_common._float(enemy_memory.get("is_new_this_turn"))
                core[base + 5] = obs_common._float(enemy_memory.get("cum_damage_dealt_to_player_ratio"))
                core[base + 6] = obs_common._float(enemy_memory.get("died_last_turn"))
                core[base + 7] = obs_common._float(enemy_memory.get("attributable_damage_last_turn_ratio"))
                core[base + 8] = obs_common._float(enemy_memory.get("alive_rank_by_hp"))
                core[base + 9] = obs_common._float(enemy_memory.get("threat_rank_by_intent"))
            world_entries.append(
                self._entry(
                    "ENEMY_CORE",
                    core,
                    owner_id=owner_id,
                    entity_id=enemy_entity_id,
                    order_id=1,
                    text_embedding=features["enemy_text"][enemy_index],
                )
            )

            intent = enemy.get("intent") if isinstance(enemy, dict) and isinstance(enemy.get("intent"), dict) else None
            if isinstance(intent, dict):
                numeric = np.zeros(TOKEN_NUMERIC_DIM, dtype=np.float32)
                numeric[0] = obs_common._log_norm(obs_common._float(intent.get("total_damage")), obs_common._LOG1P_200)
                numeric[1] = obs_common._log_norm(obs_common._float(intent.get("damage_per_hit")), obs_common._LOG1P_100)
                numeric[2] = min(obs_common._float(intent.get("repeats")) / 5.0, 1.0)
                numeric[3:7] = np.asarray(obs_common._infer_enemy_intent_flags(intent, obs_common._float(intent.get("total_damage"))), dtype=np.float32)
                if isinstance(enemy_memory, dict):
                    numeric[7] = obs_common._float(enemy_memory.get("intent_changed_this_turn"))
                    numeric[8] = obs_common._float(enemy_memory.get("turns_since_intent_change_norm"))
                    numeric[9] = obs_common._float(enemy_memory.get("intent_total_damage_delta"))
                    numeric[10] = obs_common._float(enemy_memory.get("intent_damage_per_hit_delta"))
                    numeric[11] = obs_common._float(enemy_memory.get("intent_damage_trend_3_turns"))
                    numeric[12] = obs_common._float(enemy_memory.get("intent_predicted_vs_actual"))
                world_entries.append(
                    self._entry(
                        "ENEMY_INTENT",
                        numeric,
                        owner_id=owner_id,
                        entity_id=enemy_entity_id,
                        order_id=2,
                        text=str(intent.get("description") or intent.get("label") or ""),
                    )
                )

            memory_powers = enemy_memory.get("powers") if isinstance(enemy_memory, dict) else None
            for power in (enemy.get("powers") or [])[:5]:
                if not isinstance(power, dict):
                    continue
                numeric = np.zeros(TOKEN_NUMERIC_DIM, dtype=np.float32)
                amount = obs_common._float(power.get("amount") or power.get("display_amount"))
                title = str(power.get("title") or "").lower()
                numeric[0] = obs_common._normalize_power_amount(amount)
                numeric[1] = float("thorn" in title or "spike" in title)
                numeric[2] = float("artifact" in title)
                numeric[3] = float("buffer" in title)
                numeric[4] = float("intang" in title)
                power_key = None
                for candidate in ("id", "power_id", "type", "key"):
                    value = power.get(candidate)
                    if value not in (None, ""):
                        power_key = str(value)
                        break
                if power_key is None and power.get("title") not in (None, ""):
                    power_key = f"title::{power.get('title')}"
                power_memory = memory_powers.get(power_key) if isinstance(memory_powers, dict) and power_key else None
                if isinstance(power_memory, dict):
                    numeric[5] = obs_common._normalize_power_amount(
                        obs_common._float(power_memory.get("amount_delta_last_turn"))
                    ) * (1.0 if obs_common._float(power_memory.get("amount_delta_last_turn")) >= 0 else -1.0)
                    numeric[6] = obs_common._normalize_power_amount(
                        obs_common._float(power_memory.get("amount_delta_since_first_seen"))
                    ) * (1.0 if obs_common._float(power_memory.get("amount_delta_since_first_seen")) >= 0 else -1.0)
                    numeric[7] = min(obs_common._float(power_memory.get("turns_since_first_seen")) / 10.0, 1.0)
                    numeric[8] = obs_common._float(power_memory.get("stack_trend_3_turns"))
                    numeric[9] = obs_common._float(power_memory.get("is_new_this_turn"))
                    numeric[10] = obs_common._float(power_memory.get("is_growing_without_player_action"))
                power_index = sum(
                    1
                    for entry in world_entries
                    if entry["type_id"] == TOKEN_TYPE_TO_ID["ENEMY_POWER"] and entry["owner_id"] == owner_id
                )
                world_entries.append(
                    self._entry(
                        "ENEMY_POWER",
                        numeric,
                        owner_id=owner_id,
                        entity_id=enemy_entity_id,
                        order_id=3 + power_index,
                        text=f"{power.get('title') or ''} | {power.get('description') or ''}",
                    )
                )

            reactive_traits, phase_rules = self._infer_enemy_traits(enemy)
            for trait_index, trait in enumerate(reactive_traits[:4]):
                world_entries.append(
                    self._entry(
                        "ENEMY_REACTIVE_TRAIT",
                        self._trait_numeric(trait),
                        owner_id=owner_id,
                        entity_id=enemy_entity_id,
                        order_id=12 + trait_index,
                        text=str(trait.get("description") or trait.get("trait") or trait.get("effect_type") or ""),
                    )
                )
            for rule_index, rule in enumerate(phase_rules[:3]):
                world_entries.append(
                    self._entry(
                        "ENEMY_PHASE_RULE",
                        self._trait_numeric(rule),
                        owner_id=owner_id,
                        entity_id=enemy_entity_id,
                        order_id=20 + rule_index,
                        text=str(rule.get("description") or rule.get("trait") or ""),
                    )
                )

    def _append_candidate_tokens(
        self,
        candidate_entries: list[dict[str, Any]],
        candidate_local_entries: list[list[dict[str, Any]]],
        action_mask: np.ndarray,
        obs: dict[str, Any],
        features: dict[str, np.ndarray],
        legal_actions: list[dict[str, Any]],
    ) -> None:
        count = min(len(legal_actions), MAX_ACTIONS)
        combat = obs.get("combat") or {}
        for action_index in range(count):
            action = legal_actions[action_index]
            if not isinstance(action, dict):
                continue
            action_mask[action_index] = 1.0
            numeric = np.zeros(TOKEN_NUMERIC_DIM, dtype=np.float32)
            numeric[: obs_common.ACTION_FEAT_DIM] = features["actions"][action_index]
            numeric[obs_common.ACTION_FEAT_DIM : obs_common.ACTION_FEAT_DIM + 16] = _compress_numeric(features["semantic_actions"][action_index], 16)
            query_type = {
                "combat": "COMBAT_CANDIDATE",
                "build": "BUILD_CANDIDATE",
                "selection": "SELECTION_CANDIDATE",
                "route": "ROUTE_CANDIDATE",
            }.get(self._infer_action_domain(action, obs), "BUILD_CANDIDATE")
            target_enemy_index, target_enemy = self._match_target_enemy(action, combat)
            target_owner_id = self._enemy_owner_id(target_enemy_index)
            target_entity_id = _stable_bucket(self._enemy_entity_key(target_enemy, target_enemy_index)) if target_enemy_index is not None else 0
            action_owner_id, action_zone_id, action_order_id = self._action_source_binding(action, obs, action_index)
            text_embedding = _compress_numeric(
                np.concatenate(
                    [
                        _compress_text_embedding(features["action_text"][action_index]),
                        _compress_text_embedding(features["semantic_action_text"][action_index]),
                    ]
                ),
                TOKEN_TEXT_DIM,
            )
            candidate_entries.append(
                self._entry(
                    query_type,
                    numeric,
                    owner_id=action_owner_id,
                    entity_id=_stable_bucket(self._action_entity_key(action)),
                    zone_id=action_zone_id or self._query_zone_id(action, obs, query_type, target_enemy_index),
                    order_id=action_order_id,
                    target_owner_id=target_owner_id,
                    target_entity_id=target_entity_id,
                    text_embedding=text_embedding,
                )
            )
            candidate_local_entries[action_index] = self._build_candidate_local(action, obs, features, action_index, target_enemy_index, target_enemy)

    def _build_candidate_local(
        self,
        action: dict[str, Any],
        obs: dict[str, Any],
        features: dict[str, np.ndarray],
        action_index: int,
        target_enemy_index: int | None,
        target_enemy: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        base_numeric = np.zeros(TOKEN_NUMERIC_DIM, dtype=np.float32)
        base_numeric[: obs_common.ACTION_FEAT_DIM] = features["actions"][action_index]

        kind = str(action.get("kind") or "").strip()
        action_domain = self._infer_action_domain(action, obs)
        action_card = action.get("card") if isinstance(action.get("card"), dict) else None
        action_potion = action.get("potion") if isinstance(action.get("potion"), dict) else None
        action_owner_id, action_zone_id, action_order_id = self._action_source_binding(action, obs, action_index)

        if action_card is not None:
            if action_domain == "selection":
                source_token_type = "SELECTION_POOL_CARD_LOCAL"
            else:
                source_token_type = "CARD_REWARD_LOCAL" if kind == "card_reward" else "SOURCE_CARD_LOCAL"
            entries.append(
                self._entry(
                    source_token_type,
                    base_numeric,
                    owner_id=action_owner_id,
                    zone_id=action_zone_id,
                    entity_id=_stable_bucket(action_card.get("id") or action_card.get("title")),
                    order_id=action_order_id,
                    text=self._build_live_card_text(action_card),
                )
            )
        if action_potion is not None:
            entries.append(
                self._entry(
                    "SOURCE_POTION_LOCAL",
                    base_numeric,
                    owner_id=OWNER_POTION,
                    entity_id=_stable_bucket(action_potion.get("id") or action_potion.get("title")),
                    order_id=self._source_order_id(action, obs, action_index),
                    text=build_live_potion_semantic_text(action_potion),
                )
            )
        item = action.get("item") if isinstance(action.get("item"), dict) else None
        if item is not None:
            entries.append(
                self._entry(
                    "SHOP_ITEM_LOCAL",
                    base_numeric,
                    owner_id=OWNER_SHOP,
                    entity_id=_stable_bucket(self._shop_item_entity_key(item)),
                    order_id=action_index + 1,
                    text=self._shop_item_text(item),
                )
            )
        if isinstance(action.get("upgrade_preview"), dict):
            preview = action.get("upgrade_preview") or {}
            entries.append(
                self._entry(
                    "PREVIEW_RESULT_LOCAL" if action_domain == "selection" else "UPGRADE_PREVIEW_LOCAL",
                    base_numeric,
                    owner_id=OWNER_UPGRADE if action_domain != "selection" else action_owner_id,
                    zone_id=TOKEN_ZONE_TO_ID["SELECTION"] if action_domain == "selection" else TOKEN_ZONE_TO_ID["UPGRADE"],
                    entity_id=_stable_bucket(preview.get("id") or preview.get("title")),
                    order_id=action_index + 1,
                    text=self._build_live_card_text(preview),
                )
            )
        if isinstance(action.get("reward"), dict):
            reward = action.get("reward") or {}
            entries.append(
                self._entry(
                    "REWARD_LOCAL",
                    base_numeric,
                    owner_id=OWNER_REWARD,
                    entity_id=_stable_bucket(self._reward_entity_key(reward)),
                    order_id=action_index + 1,
                    text=self._reward_text(reward),
                )
            )

        if target_enemy_index is not None and isinstance(target_enemy, dict):
            numeric = np.zeros(TOKEN_NUMERIC_DIM, dtype=np.float32)
            numeric[0] = obs_common._log_norm(obs_common._float(target_enemy.get("hp", target_enemy.get("current_hp"))), obs_common._LOG1P_1200)
            numeric[1] = obs_common._log_norm(obs_common._float(target_enemy.get("block")), obs_common._LOG1P_200)
            numeric[2] = obs_common._log_norm(obs_common._float(((target_enemy.get("intent") or {}).get("total_damage"))), obs_common._LOG1P_200)
            entries.append(
                self._entry(
                    "TARGET_LOCAL",
                    numeric,
                    owner_id=self._enemy_owner_id(target_enemy_index),
                    entity_id=_stable_bucket(self._enemy_entity_key(target_enemy, target_enemy_index)),
                    order_id=1,
                    text=build_live_enemy_semantic_text(target_enemy),
                )
            )
            target_context_entries: list[dict[str, Any]] = []
            self._append_target_enemy_local_context(target_context_entries, target_enemy_index, target_enemy)
            self._extend_with_budget(entries, target_context_entries, limit=3)
            source_for_target = action_card if action_card is not None else action_potion
            target_reaction_entries: list[dict[str, Any]] = []
            self._append_target_reaction_local(target_reaction_entries, target_enemy_index, target_enemy, source_for_target)
            self._extend_with_budget(entries, target_reaction_entries, limit=1)

        if action_domain == "combat":
            self._append_combat_candidate_context(entries, obs, features, action, action_index, target_enemy_index, target_enemy)
        elif action_domain == "build":
            self._append_build_candidate_context(entries, obs, features, action, action_index)
        elif action_domain == "selection":
            self._append_selection_candidate_context(entries, obs, features, action, action_index)
        elif action_domain == "route":
            self._append_route_candidate_context(entries, obs, features, action, action_index)

        route_entries: list[dict[str, Any]] = []
        if features["route_summary"][action_index].any():
            numeric = np.zeros(TOKEN_NUMERIC_DIM, dtype=np.float32)
            numeric[: obs_common.ROUTE_SUMMARY_DIM] = features["route_summary"][action_index]
            route_entries.append(self._entry("ROUTE_SUMMARY_TOKEN", numeric, owner_id=OWNER_ROUTE, entity_id=0))
        for node_index in range(obs_common.MAX_ROUTE_NODES):
            if features["route_node_mask"][action_index, node_index] <= 0:
                continue
            numeric = np.zeros(TOKEN_NUMERIC_DIM, dtype=np.float32)
            numeric[: obs_common.ROUTE_NODE_FEAT_DIM] = features["route_nodes"][action_index, node_index]
            route_entries.append(
                self._entry(
                    "ROUTE_NODE",
                    numeric,
                    owner_id=OWNER_ROUTE,
                    entity_id=_stable_bucket(f"route:{action_index}:{node_index}"),
                    order_id=node_index + 1,
                )
            )
        self._extend_with_budget(entries, route_entries)
        return entries

    def _extend_with_budget(
        self,
        target: list[dict[str, Any]],
        additions: list[dict[str, Any]],
        *,
        limit: int | None = None,
        scorer=None,
    ) -> None:
        remaining = max(MAX_CANDIDATE_LOCAL_TOKENS - len(target), 0)
        if remaining <= 0 or not additions:
            return
        ordered = list(additions)
        if scorer is not None:
            ordered.sort(key=scorer, reverse=True)
        take = remaining if limit is None else min(remaining, max(limit, 0))
        if take <= 0:
            return
        target.extend(ordered[:take])

    def _append_combat_candidate_context(
        self,
        entries: list[dict[str, Any]],
        obs: dict[str, Any],
        features: dict[str, np.ndarray],
        action: dict[str, Any],
        action_index: int,
        target_enemy_index: int | None,
        target_enemy: dict[str, Any] | None,
    ) -> None:
        combat = obs.get("combat") or {}
        player = obs.get("player") or {}
        if not combat:
            return

        hp = obs_common._float(player.get("hp", player.get("current_hp")))
        max_hp = obs_common._float(player.get("max_hp"))
        block = obs_common._float(player.get("block"))
        enemies = combat.get("enemies") or []
        incoming = sum(
            obs_common._float(((enemy or {}).get("intent") or {}).get("total_damage"))
            for enemy in enemies
            if isinstance(enemy, dict)
        )
        boss_player_state = self._boss_player_state()

        player_numeric = np.zeros(TOKEN_NUMERIC_DIM, dtype=np.float32)
        player_numeric[0] = min(hp / max(max_hp, 1.0), 1.0) if max_hp > 0 else 0.0
        player_numeric[1] = obs_common._log_norm(hp, obs_common._LOG1P_200)
        player_numeric[2] = obs_common._log_norm(block, obs_common._LOG1P_200)
        player_numeric[3] = obs_common._log_norm(incoming, obs_common._LOG1P_200)
        player_numeric[4] = obs_common._signed_log_norm(hp + block - incoming, obs_common._LOG1P_200)
        player_numeric[5 : 5 + min(obs_common.POWER_DIM, TOKEN_NUMERIC_DIM - 5)] = features["player_powers"][: TOKEN_NUMERIC_DIM - 5]
        player_numeric[29] = float(boss_player_state.get("facing_left", 0.0))
        player_numeric[30] = float(boss_player_state.get("facing_right", 0.0))
        player_numeric[31] = float(boss_player_state.get("sandpit_active", 0.0))
        player_numeric[32] = float(boss_player_state.get("sandpit_turns_norm", 0.0))
        player_numeric[33] = float(boss_player_state.get("ringing_active", 0.0))
        player_numeric[34] = float(boss_player_state.get("ringing_amount_norm", 0.0))
        player_numeric[35] = float(boss_player_state.get("chains_active", 0.0))
        player_numeric[36] = float(boss_player_state.get("bound_active", 0.0))
        player_numeric[37] = float(boss_player_state.get("hunger_active", 0.0))
        player_numeric[38] = float(boss_player_state.get("scrutiny_active", 0.0))
        player_numeric[39] = float(boss_player_state.get("grasp_active", 0.0))
        player_numeric[40] = float(boss_player_state.get("frantic_escape_hand_norm", 0.0))
        player_numeric[41] = float(boss_player_state.get("frantic_escape_draw_norm", 0.0))
        player_numeric[42] = float(boss_player_state.get("frantic_escape_discard_norm", 0.0))
        player_numeric[43] = float(boss_player_state.get("frantic_escape_total_norm", 0.0))
        player_numeric[44] = float(boss_player_state.get("escape_card_available", 0.0))
        player_numeric[45] = float(boss_player_state.get("play_budget_lock", 0.0))
        player_numeric[46] = float(boss_player_state.get("doormaker_lock_pressure", 0.0))
        player_numeric[47] = float(boss_player_state.get("back_attack_risk", 0.0))
        player_numeric[48] = float(boss_player_state.get("linked_support_alive", 0.0))
        player_numeric[49] = float(boss_player_state.get("countdown_active", 0.0))
        player_numeric[50] = float(boss_player_state.get("escape_card_tax", 0.0))
        entries.append(self._entry("PLAYER_STATE_LOCAL", player_numeric, owner_id=OWNER_PLAYER, entity_id=0))

        energy_numeric = np.zeros(TOKEN_NUMERIC_DIM, dtype=np.float32)
        source_profile = self._source_profile(action.get("card") if isinstance(action.get("card"), dict) else action.get("potion") if isinstance(action.get("potion"), dict) else None)
        current_energy = obs_common._float(combat.get("energy"))
        max_energy = obs_common._float(combat.get("max_energy"))
        spend_cost = source_profile["cost"]
        energy_numeric[0] = min(current_energy / max(max_energy, 1.0), 1.0) if max_energy > 0 else 0.0
        energy_numeric[1] = min(current_energy / 10.0, 1.0)
        energy_numeric[2] = min(max_energy / 10.0, 1.0)
        energy_numeric[3] = min(spend_cost / 5.0, 1.0)
        energy_numeric[4] = float(source_profile["zero_cost"] > 0.5)
        energy_numeric[5] = float(source_profile["x_cost"] > 0.5)
        energy_numeric[6] = min(source_profile["energy"] / 5.0, 1.0)
        energy_numeric[7] = min(source_profile["draw"] / 5.0, 1.0)
        energy_numeric[8] = min(source_profile["hits"] / 10.0, 1.0)
        energy_numeric[9] = float(current_energy + source_profile["energy"] >= spend_cost)
        energy_numeric[10] = float(current_energy >= spend_cost)
        energy_numeric[11] = min(obs_common._float(combat.get("stars")) / 10.0, 1.0)
        energy_numeric[12] = features["actions"][action_index, 39] if features["actions"].shape[1] > 39 else 0.0
        energy_numeric[13] = features["actions"][action_index, 48] if features["actions"].shape[1] > 48 else 0.0
        energy_numeric[14] = float(boss_player_state.get("play_budget_lock", 0.0))
        energy_numeric[15] = float(boss_player_state.get("doormaker_lock_pressure", 0.0))
        energy_numeric[16] = float(boss_player_state.get("sandpit_active", 0.0))
        energy_numeric[17] = float(boss_player_state.get("sandpit_turns_norm", 0.0))
        energy_numeric[18] = float(boss_player_state.get("frantic_escape_hand_norm", 0.0))
        energy_numeric[19] = float(boss_player_state.get("frantic_escape_total_norm", 0.0))
        energy_numeric[20] = float(boss_player_state.get("escape_card_available", 0.0))
        energy_numeric[21] = float(boss_player_state.get("back_attack_risk", 0.0))
        energy_numeric[22] = float(boss_player_state.get("countdown_active", 0.0))
        energy_numeric[23] = float(boss_player_state.get("escape_card_tax", 0.0))
        energy_numeric[24] = float((boss_player_state.get("play_budget_lock", 0.0) > 0.0) and current_energy > 0.0)
        energy_numeric[25] = float((boss_player_state.get("sandpit_active", 0.0) > 0.0) and (source_profile["damage"] > 0.0))
        energy_numeric[26] = float((boss_player_state.get("escape_card_available", 0.0) > 0.0) and (source_profile["draw"] > 0.0 or source_profile["energy"] > 0.0))
        entries.append(self._entry("ENERGY_CONTEXT_LOCAL", energy_numeric, owner_id=OWNER_PLAYER, entity_id=0))
        energy_plan_entries: list[dict[str, Any]] = []
        self._append_energy_budget_local(energy_plan_entries, obs, features, action, source_profile)

        source_card = action.get("card") if isinstance(action.get("card"), dict) else None
        pile_summary_entries: list[dict[str, Any]] = []
        self._append_pile_context(
            pile_summary_entries,
            "DRAW_CONTEXT_LOCAL",
            OWNER_DRAW,
            self._runtime_cards(obs, "draw_pile", "draw_preview_cards"),
            source_card,
            "draw",
        )
        self._append_pile_context(
            pile_summary_entries,
            "DISCARD_CONTEXT_LOCAL",
            OWNER_DISCARD,
            self._runtime_cards(obs, "discard_pile", "discard_cards"),
            source_card,
            "discard",
        )
        self._append_pile_context(
            pile_summary_entries,
            "EXHAUST_CONTEXT_LOCAL",
            OWNER_EXHAUST,
            self._runtime_cards(obs, "exhaust_pile", "exhaust_cards"),
            source_card,
            "exhaust",
        )
        self._append_pile_context(
            pile_summary_entries,
            "PLAY_PILE_CONTEXT_LOCAL",
            OWNER_PLAY,
            self._runtime_cards(obs, "play_pile", "play_pile_cards"),
            source_card,
            "play",
        )


        hand_mutation_entries: list[dict[str, Any]] = []
        self._append_hand_mutation_locals(
            hand_mutation_entries,
            obs,
            source_card,
            current_energy=current_energy,
        )

        binding_entries: list[dict[str, Any]] = []
        self._append_source_pile_binding_locals(binding_entries, obs, source_card)

        cycle_entries: list[dict[str, Any]] = []
        self._append_cycle_plan_local(cycle_entries, obs, action, source_card)
        self._append_card_flow_counterfactual_local(cycle_entries, obs, action, source_card, source_profile, current_energy)
        self._append_energy_chain_local(cycle_entries, obs, action, source_profile, current_energy)

        peek_entries: list[dict[str, Any]] = []
        self._append_pile_peek_locals(
            peek_entries,
            "DRAW_PREVIEW_CARD",
            OWNER_DRAW,
            self._runtime_cards(obs, "draw_pile", "draw_preview_cards"),
            limit=1,
        )
        self._append_pile_peek_locals(
            peek_entries,
            "DISCARD_CARD",
            OWNER_DISCARD,
            self._runtime_cards(obs, "discard_pile", "discard_cards"),
            limit=1,
        )
        self._append_pile_peek_locals(
            peek_entries,
            "EXHAUST_CARD",
            OWNER_EXHAUST,
            self._runtime_cards(obs, "exhaust_pile", "exhaust_cards"),
            limit=1,
        )
        self._append_pile_peek_locals(
            peek_entries,
            "PLAY_PILE_CARD",
            OWNER_PLAY,
            self._runtime_cards(obs, "play_pile", "play_pile_cards"),
            limit=1,
        )

        support_entries: list[dict[str, Any]] = []
        self._append_relic_trigger_locals(support_entries, obs, action)
        self._append_potion_option_locals(support_entries, obs, action, features)

        support_graph_entries: list[dict[str, Any]] = []
        self._append_relic_potion_graph_local(
            support_graph_entries,
            obs,
            action,
            features,
            target_enemy_index,
            target_enemy,
        )

        self._extend_with_budget(entries, energy_plan_entries, limit=1)
        self._extend_with_budget(entries, hand_mutation_entries, limit=5)
        self._extend_with_budget(entries, cycle_entries, limit=1)
        self._extend_with_budget(entries, pile_summary_entries, limit=4)
        self._extend_with_budget(
            entries,
            binding_entries,
            limit=2,
            scorer=lambda entry: float(entry["numeric"][19]) * 10.0 - float(entry["numeric"][3]),
        )
        self._extend_with_budget(
            entries,
            peek_entries,
            limit=2,
            scorer=lambda entry: 2.0 - float(entry["order_id"]),
        )
        self._extend_with_budget(
            entries,
            support_entries,
            limit=2,
            scorer=lambda entry: float(entry["numeric"][0]),
        )
        self._extend_with_budget(entries, support_graph_entries, limit=1)


    def _append_card_flow_counterfactual_local(
        self,
        entries: list[dict[str, Any]],
        obs: dict[str, Any],
        action: dict[str, Any],
        source_card: dict[str, Any] | None,
        source_profile: dict[str, float],
        current_energy: float,
    ) -> None:
        combat = obs.get("combat") or {}
        hand_cards = self._runtime_cards(obs, "hand", "hand_cards")
        draw_cards = self._runtime_cards(obs, "draw_pile", "draw_preview_cards")
        discard_cards = self._runtime_cards(obs, "discard_pile", "discard_cards")
        exhaust_cards = self._runtime_cards(obs, "exhaust_pile", "exhaust_cards")
        if str(action.get("action_id") or "").lower() == "end_turn" or str((action.get("semantic") or {}).get("family") or "").lower() == "end_turn":
            numeric = np.zeros(TOKEN_NUMERIC_DIM, dtype=np.float32)
            ethereal_count = 0.0
            retain_count = 0.0
            exhaust_count = 0.0
            total_cost = 0.0
            playable_cost = 0.0
            for card in hand_cards:
                profile = self._source_profile(card)
                text = self._source_text(card)
                ethereal_count += profile["ethereal"]
                retain_count += profile["retain"]
                exhaust_count += profile["exhaust"]
                total_cost += profile["cost"]
                playable_cost += float(profile["cost"] <= current_energy or profile["zero_cost"] > 0.5)
            numeric[0] = min(len(hand_cards) / 10.0, 1.0)
            numeric[1] = min(draw_cards.__len__() / 30.0, 1.0)
            numeric[2] = min(discard_cards.__len__() / 30.0, 1.0)
            numeric[3] = min(exhaust_cards.__len__() / 20.0, 1.0)
            numeric[4] = min(ethereal_count / 5.0, 1.0)
            numeric[5] = min(retain_count / 5.0, 1.0)
            numeric[6] = min(exhaust_count / 5.0, 1.0)
            numeric[7] = min(playable_cost / 10.0, 1.0)
            numeric[8] = obs_common._log_norm(total_cost, obs_common._LOG1P_100)
            numeric[9] = float(len(draw_cards) <= 3 and len(discard_cards) > 0)
            numeric[10] = float(len(hand_cards) > 0 and current_energy > 0.0)
            entries.append(self._entry("END_TURN_HAND_FLOW_LOCAL", numeric, owner_id=OWNER_PLAYER, entity_id=0, text="end_turn hand flow: unplayed non-ethereal cards move toward discard; retain stays; ethereal/exhaust leaves loop"))
            return
        if not isinstance(source_card, dict):
            return
        numeric = np.zeros(TOKEN_NUMERIC_DIM, dtype=np.float32)
        text = self._source_text(source_card)
        exhausts = source_profile["exhaust"] > 0.5 or "exhaust" in text
        ethereal = source_profile["ethereal"] > 0.5 or "ethereal" in text
        retain = source_profile["retain"] > 0.5 or "retain" in text
        power = source_profile["power"] > 0.5
        cost = source_profile["cost"]
        energy_after = current_energy - cost + source_profile["energy"] - source_profile.get("energy_loss", 0.0)
        numeric[0] = float(exhausts)
        numeric[1] = float(power)
        numeric[2] = float(not exhausts and not power)
        numeric[3] = float(retain)
        numeric[4] = float(ethereal)
        numeric[5] = min(len(draw_cards) / 30.0, 1.0)
        numeric[6] = min(len(discard_cards) / 30.0, 1.0)
        numeric[7] = min(len(exhaust_cards) / 20.0, 1.0)
        numeric[8] = float(len(draw_cards) <= 3 and len(discard_cards) > 0)
        numeric[9] = float(exhausts and not ethereal)
        numeric[10] = float(exhausts and source_profile["draw"] <= 0.0 and source_profile["energy"] <= 0.0)
        numeric[11] = obs_common._signed_log_norm(energy_after, obs_common._LOG1P_100)
        numeric[12] = min(source_profile["damage"] / 80.0, 1.0)
        numeric[13] = min(source_profile["block"] / 80.0, 1.0)
        numeric[14] = min(source_profile["draw"] / 5.0, 1.0)
        numeric[15] = min(source_profile["energy"] / 5.0, 1.0)
        numeric[16] = min(source_profile.get("energy_loss", 0.0) / 5.0, 1.0)
        numeric[17] = float(source_profile.get("strategic_skip_value", 0.0) > 0.5)
        entries.append(self._entry("CARD_FLOW_COUNTERFACTUAL_LOCAL", numeric, owner_id=OWNER_HAND, entity_id=_stable_bucket((source_card.get("id") if isinstance(source_card, dict) else None) or (source_card.get("title") if isinstance(source_card, dict) else None) or "flow"), text="play_now destination vs skip/end_turn loop counterfactual"))

    def _append_energy_chain_local(
        self,
        entries: list[dict[str, Any]],
        obs: dict[str, Any],
        action: dict[str, Any],
        source_profile: dict[str, float],
        current_energy: float,
    ) -> None:
        if str(action.get("action_id") or "").lower() == "end_turn":
            return
        combat = obs.get("combat") or {}
        hand_cards = self._runtime_cards(obs, "hand", "hand_cards")
        cost = source_profile["cost"]
        energy_gain = source_profile["energy"]
        energy_loss = source_profile.get("energy_loss", 0.0)
        energy_after = max(0.0, current_energy - cost + energy_gain - energy_loss)
        followup_count = 0.0
        followup_damage = 0.0
        followup_block = 0.0
        for card in hand_cards:
            profile = self._source_profile(card)
            if profile["cost"] <= energy_after + 1e-6 and card is not action.get("card"):
                followup_count += 1.0
                followup_damage = max(followup_damage, profile["damage"])
                followup_block = max(followup_block, profile["block"])
        numeric = np.zeros(TOKEN_NUMERIC_DIM, dtype=np.float32)
        numeric[0] = min(current_energy / 10.0, 1.0)
        numeric[1] = min(cost / 5.0, 1.0)
        numeric[2] = min(energy_gain / 5.0, 1.0)
        numeric[10] = min(energy_loss / 5.0, 1.0)
        numeric[3] = obs_common._signed_log_norm(energy_after, obs_common._LOG1P_100)
        numeric[4] = min(followup_count / 10.0, 1.0)
        numeric[5] = min(followup_damage / 80.0, 1.0)
        numeric[6] = min(followup_block / 80.0, 1.0)
        numeric[7] = float(source_profile["x_cost"] > 0.5 and current_energy <= 0.0)
        numeric[8] = float(energy_gain > 0.0 and followup_count <= 0.0)
        numeric[9] = min(source_profile["hp_loss"] / 20.0, 1.0)
        numeric[11] = float(source_profile.get("strategic_skip_value", 0.0) > 0.5)
        entries.append(self._entry("ENERGY_CHAIN_LOCAL", numeric, owner_id=OWNER_PLAYER, entity_id=0, text="energy spend/gain follow-up chain and zero-energy x-cost affordance"))

    def _append_hand_mutation_locals(
        self,
        entries: list[dict[str, Any]],
        obs: dict[str, Any],
        source_card: dict[str, Any] | None,
        *,
        current_energy: float = 0.0,
    ) -> None:
        """Expose generic action->hand mutation affordances as local tokens.

        This is intentionally not card-name-specific: Armaments, cost reducers,
        hand exhaust/discard/transform/copy/retain effects, draw/add/return-to-
        hand effects all flow through the same summary/target/post-preview
        structure.  The policy can cross-attend source action <-> current hand
        cards <-> post-hand preview without requiring MCTS to discover the
        intermediate hand state.
        """
        if not isinstance(source_card, dict):
            return
        hand_cards = self._runtime_cards(obs, "hand", "hand_cards")
        combat = obs.get("combat") or {}
        plan = infer_hand_mutation(
            source_card,
            hand_cards,
            self._runtime_cards(obs, "draw_pile", "draw_preview_cards"),
            self._runtime_cards(obs, "discard_pile", "discard_cards"),
            self._runtime_cards(obs, "exhaust_pile", "exhaust_cards"),
            current_energy=current_energy,
        )
        if not plan.will_mutate_hand:
            return

        source_key = source_card.get("id") or source_card.get("title") or "hand-mutation"
        summary = np.asarray(mutation_summary_numeric(plan), dtype=np.float32)
        entries.append(
            self._entry(
                "HAND_MUTATION_LOCAL",
                summary,
                owner_id=OWNER_HAND,
                zone_id=TOKEN_ZONE_TO_ID["HAND"],
                entity_id=_stable_bucket(f"hand-mut:{source_key}"),
                order_id=1,
                text=f"hand mutation | {self._build_live_card_text(source_card)}",
            )
        )

        post = np.asarray(post_hand_preview_numeric(plan), dtype=np.float32)
        entries.append(
            self._entry(
                "POST_HAND_PREVIEW_LOCAL",
                post,
                owner_id=OWNER_HAND,
                zone_id=TOKEN_ZONE_TO_ID["HAND"],
                entity_id=_stable_bucket(f"post-hand:{source_key}"),
                order_id=2,
                text=f"post hand preview | {self._build_live_card_text(source_card)}",
            )
        )

        for rank, target in enumerate(plan.targets[:6], start=1):
            if not target.affected and rank > 2:
                continue
            row = np.asarray(mutation_target_numeric(target), dtype=np.float32)
            card = target.card
            entries.append(
                self._entry(
                    "HAND_MUTATION_TARGET_LOCAL",
                    row,
                    owner_id=OWNER_HAND,
                    zone_id=TOKEN_ZONE_TO_ID["HAND"],
                    entity_id=_stable_bucket(card.get("id") or card.get("title") or f"hand-target:{target.index}"),
                    order_id=min(target.index + 1, MAX_ORDER_ID),
                    text=f"hand mutation target | {self._build_live_card_text(card)}",
                )
            )

    def _append_selection_candidate_context(
        self,
        entries: list[dict[str, Any]],
        obs: dict[str, Any],
        features: dict[str, np.ndarray],
        action: dict[str, Any],
        action_index: int,
    ) -> None:
        action_owner_id, _action_zone_id, action_order_id = self._action_source_binding(action, obs, action_index)
        semantics_text = self._selection_semantics_text(action)
        in_combat = bool(obs.get("combat"))

        operator_numeric = np.zeros(TOKEN_NUMERIC_DIM, dtype=np.float32)
        selection = str(action.get("selection") or "").strip().lower()
        semantics_lower = semantics_text.lower()
        operator_numeric[0] = float(selection in {"confirm", "confirm_selection"})
        operator_numeric[1] = float(selection in {"cancel", "close"})
        operator_numeric[2] = float(selection == "skip")
        operator_numeric[3] = float(any(token in semantics_lower for token in ("upgrade", "smith")))
        operator_numeric[4] = float(any(token in semantics_lower for token in ("transform", "mutate", "change")))
        operator_numeric[5] = float(any(token in semantics_lower for token in ("remove", "purge")))
        operator_numeric[6] = float(any(token in semantics_lower for token in ("exhaust", "consume")))
        operator_numeric[7] = float("discard" in semantics_lower)
        operator_numeric[8] = float(any(token in semantics_lower for token in ("discover", "draft", "reward", "choose", "pick")))
        operator_numeric[9] = float(in_combat)
        operator_numeric[10] = float(not in_combat)
        operator_numeric[11] = min(max(float(action_order_id), 0.0) / max(MAX_ORDER_ID, 1), 1.0)
        operator_numeric[12] = features["actions"][action_index, 12] if features["actions"].shape[1] > 12 else 0.0
        operator_numeric[13] = features["actions"][action_index, 13] if features["actions"].shape[1] > 13 else 0.0
        # §4C: surface "this card is already selected in the current pick set".
        # Bridge emits is_selected on each card_selection:select action so the
        # model can directly see pick→deselect oscillations on multi-pick burn
        # cards (POTION.GLOWWATER, 净化, etc.) instead of inferring from history.
        operator_numeric[14] = float(bool(action.get("is_selected"))) if isinstance(action.get("is_selected"), (bool, int)) else 0.0
        entries.append(
            self._entry(
                "SELECTION_OPERATOR_LOCAL",
                operator_numeric,
                owner_id=action_owner_id,
                zone_id=TOKEN_ZONE_TO_ID["SELECTION"],
                entity_id=_stable_bucket(f"selection-op:{self._action_entity_key(action)}"),
                order_id=1,
                text=selection or semantics_text or "selection operator",
            )
        )

        semantics_numeric = np.zeros(TOKEN_NUMERIC_DIM, dtype=np.float32)
        semantics_numeric[0] = float("hand" in semantics_lower)
        semantics_numeric[1] = float("draw" in semantics_lower or "draw pile" in semantics_lower)
        semantics_numeric[2] = float("discard" in semantics_lower)
        semantics_numeric[3] = float("exhaust" in semantics_lower)
        semantics_numeric[4] = float("deck" in semantics_lower)
        semantics_numeric[5] = float("play pile" in semantics_lower or "played" in semantics_lower)
        semantics_numeric[6] = float(any(token in semantics_lower for token in ("upgrade", "smith")))
        semantics_numeric[7] = float(any(token in semantics_lower for token in ("transform", "mutate")))
        semantics_numeric[8] = float(any(token in semantics_lower for token in ("remove", "purge")))
        semantics_numeric[9] = float(any(token in semantics_lower for token in ("retain", "keep")))
        semantics_numeric[10] = float(any(token in semantics_lower for token in ("bottle", "duplicate", "copy")))
        semantics_numeric[11] = float("reward" in semantics_lower or "discover" in semantics_lower)
        entries.append(
            self._entry(
                "SELECTION_SEMANTICS_LOCAL",
                semantics_numeric,
                owner_id=OWNER_NONE,
                zone_id=TOKEN_ZONE_TO_ID["SELECTION"],
                entity_id=_stable_bucket(f"selection-semantics:{semantics_text or action.get('action_id') or action_index}"),
                order_id=2,
                text=semantics_text,
            )
        )

        selected_card = self._resolve_build_candidate_card(action)
        if selected_card is not None:
            synergy_numeric = self._deck_synergy_numeric((obs.get("player") or {}).get("deck_cards") or [], selected_card)
            entries.append(
                self._entry(
                    "DECK_SYNERGY_LOCAL",
                    synergy_numeric,
                    owner_id=OWNER_DECK,
                    entity_id=_stable_bucket(selected_card.get("id") or selected_card.get("title") or "selection-deck-synergy"),
                    order_id=3,
                    text=self._build_live_card_text(selected_card),
                )
            )

        if in_combat:
            source_card = action.get("card") if isinstance(action.get("card"), dict) else None
            pile_summary_entries: list[dict[str, Any]] = []
            self._append_pile_context(
                pile_summary_entries,
                "DRAW_CONTEXT_LOCAL",
                OWNER_DRAW,
                self._runtime_cards(obs, "draw_pile", "draw_preview_cards"),
                source_card,
                "draw",
            )
            self._append_pile_context(
                pile_summary_entries,
                "DISCARD_CONTEXT_LOCAL",
                OWNER_DISCARD,
                self._runtime_cards(obs, "discard_pile", "discard_cards"),
                source_card,
                "discard",
            )
            self._append_pile_context(
                pile_summary_entries,
                "EXHAUST_CONTEXT_LOCAL",
                OWNER_EXHAUST,
                self._runtime_cards(obs, "exhaust_pile", "exhaust_cards"),
                source_card,
                "exhaust",
            )
            self._append_pile_context(
                pile_summary_entries,
                "PLAY_PILE_CONTEXT_LOCAL",
                OWNER_PLAY,
                self._runtime_cards(obs, "play_pile", "play_pile_cards"),
                source_card,
                "play",
            )
            self._extend_with_budget(entries, pile_summary_entries, limit=4)

            support_entries: list[dict[str, Any]] = []
            self._append_relic_trigger_locals(support_entries, obs, action)
            self._append_potion_option_locals(support_entries, obs, action, features)
            self._extend_with_budget(entries, support_entries, limit=2, scorer=lambda entry: float(entry["numeric"][0]))
        else:
            self._append_build_candidate_context(entries, obs, features, action, action_index)

    def _append_build_candidate_context(
        self,
        entries: list[dict[str, Any]],
        obs: dict[str, Any],
        features: dict[str, np.ndarray],
        action: dict[str, Any],
        action_index: int,
    ) -> None:
        player = obs.get("player") or {}
        deck_cards = player.get("deck_cards") or []
        gold = obs_common._float(player.get("gold"))

        build_numeric = np.zeros(TOKEN_NUMERIC_DIM, dtype=np.float32)
        run_memory = features["run_memory"]
        objective = features["objective_context"]
        build_numeric[: min(24, run_memory.shape[0])] = run_memory[:24]
        build_numeric[24 : 24 + min(24, objective.shape[0])] = objective[:24]
        build_numeric[48] = min(len(deck_cards) / 50.0, 1.0) if isinstance(deck_cards, list) else 0.0
        build_numeric[49] = obs_common._log_norm(gold, obs_common._LOG1P_500)
        build_numeric[50] = min(len(player.get("relics") or []) / max(obs_common.MAX_RELICS, 1), 1.0)
        build_numeric[51] = min(len(player.get("potions") or []) / max(obs_common.MAX_POTIONS, 1), 1.0)
        build_numeric[52] = features["actions"][action_index, 12] if features["actions"].shape[1] > 12 else 0.0
        build_numeric[53] = features["actions"][action_index, 13] if features["actions"].shape[1] > 13 else 0.0
        build_numeric[54] = features["actions"][action_index, 14] if features["actions"].shape[1] > 14 else 0.0
        build_numeric[55] = features["actions"][action_index, 18] if features["actions"].shape[1] > 18 else 0.0

        # Event-option structured effect deltas (from bridge regex). These slots
        # replace the text encoder as the primary signal for event choice
        # outcomes; slots stay zero for non-event-option actions.
        if str(action.get("kind") or "") == "event_option":
            option = action.get("option") if isinstance(action.get("option"), dict) else None
            deltas = option.get("effect_deltas") if isinstance(option, dict) else None
            if isinstance(deltas, dict):
                player_max_hp = max(obs_common._float((obs.get("player") or {}).get("max_hp"), 1.0), 1.0)
                hp_delta = obs_common._float(deltas.get("hp_delta"))
                max_hp_delta = obs_common._float(deltas.get("max_hp_delta"))
                gold_delta = obs_common._float(deltas.get("gold_delta"))
                # Signed normalized 鈥?positive = gain, negative = loss.
                build_numeric[56] = max(-1.0, min(hp_delta / player_max_hp, 1.0))
                build_numeric[57] = obs_common._signed_log_norm(max_hp_delta, obs_common._LOG1P_100)
                build_numeric[58] = obs_common._signed_log_norm(gold_delta, obs_common._LOG1P_500)
                build_numeric[59] = 1.0 if deltas.get("heal_full") else 0.0
                build_numeric[60] = min(obs_common._float(deltas.get("card_add_count")) / 3.0, 1.0)
                build_numeric[61] = 1.0 if deltas.get("card_add_attack") else 0.0
                build_numeric[62] = 1.0 if deltas.get("card_add_skill") else 0.0
                build_numeric[63] = 1.0 if deltas.get("card_add_power") else 0.0
                build_numeric[64] = 1.0 if deltas.get("card_add_curse") else 0.0
                build_numeric[65] = 1.0 if deltas.get("card_add_status") else 0.0
                build_numeric[66] = min(obs_common._float(deltas.get("card_remove_count")) / 3.0, 1.0)
                build_numeric[67] = min(obs_common._float(deltas.get("card_transform_count")) / 3.0, 1.0)
                build_numeric[68] = min(obs_common._float(deltas.get("card_upgrade_count")) / 3.0, 1.0)
                build_numeric[69] = min(obs_common._float(deltas.get("card_duplicate_count")) / 3.0, 1.0)
                build_numeric[70] = 1.0 if deltas.get("relic_gain") else 0.0
                build_numeric[71] = 1.0 if deltas.get("potion_gain") else 0.0
                build_numeric[72] = 1.0 if deltas.get("enter_combat") else 0.0
                # Aggregate cost / benefit magnitudes as quick-lookup summaries.
                total_cost_magnitude = max(0.0, -hp_delta) / player_max_hp + max(0.0, -gold_delta) / 500.0
                total_benefit_magnitude = (
                    max(0.0, hp_delta) / player_max_hp
                    + max(0.0, gold_delta) / 500.0
                    + (1.0 if deltas.get("relic_gain") else 0.0)
                    + (1.0 if deltas.get("potion_gain") else 0.0)
                )
                build_numeric[73] = min(total_cost_magnitude, 1.0)
                build_numeric[74] = min(total_benefit_magnitude, 1.0)

        entries.append(
            self._entry(
                "BUILD_STATE_LOCAL",
                build_numeric,
                owner_id=OWNER_PLAYER,
                entity_id=0,
                text=f"build state | deck {len(deck_cards) if isinstance(deck_cards, list) else 0} | gold {int(gold)}",
            )
        )

        candidate_card = self._resolve_build_candidate_card(action)
        if candidate_card is not None:
            synergy_numeric = self._deck_synergy_numeric(deck_cards if isinstance(deck_cards, list) else [], candidate_card)
            entries.append(
                self._entry(
                    "DECK_SYNERGY_LOCAL",
                    synergy_numeric,
                    owner_id=OWNER_DECK,
                    entity_id=_stable_bucket(candidate_card.get("id") or candidate_card.get("title") or "deck-synergy"),
                    text=self._build_live_card_text(candidate_card),
                )
            )

        if str(action.get("kind") or "") == "shop":
            shop_numeric = np.zeros(TOKEN_NUMERIC_DIM, dtype=np.float32)
            item = action.get("item") if isinstance(action.get("item"), dict) else {}
            cost = obs_common._float(item.get("cost"))
            shop_numeric[0] = obs_common._log_norm(gold, obs_common._LOG1P_500)
            shop_numeric[1] = obs_common._log_norm(cost, obs_common._LOG1P_500)
            shop_numeric[2] = float(gold >= cost and cost > 0)
            shop_numeric[3] = min(cost / max(gold, 1.0), 1.0) if gold > 0 else float(cost > 0)
            shop_numeric[4] = min(len(deck_cards) / 50.0, 1.0) if isinstance(deck_cards, list) else 0.0
            shop_numeric[5] = float(isinstance(item.get("card"), dict))
            shop_numeric[6] = float(isinstance(item.get("relic"), dict))
            shop_numeric[7] = float(isinstance(item.get("potion"), dict))
            shop_numeric[8] = float(str(action.get("shop_action") or "").strip().lower().find("remove") >= 0)
            entries.append(
                self._entry(
                    "SHOP_ECON_LOCAL",
                    shop_numeric,
                    owner_id=OWNER_SHOP,
                    entity_id=_stable_bucket(self._shop_item_entity_key(item)),
                    text=self._shop_item_text(item),
                )
            )

    def _append_route_candidate_context(
        self,
        entries: list[dict[str, Any]],
        obs: dict[str, Any],
        features: dict[str, np.ndarray],
        action: dict[str, Any],
        action_index: int,
    ) -> None:
        player = obs.get("player") or {}
        route_summary = features["route_summary"][action_index]
        if route_summary.any():
            risk_numeric = np.zeros(TOKEN_NUMERIC_DIM, dtype=np.float32)
            value_numeric = np.zeros(TOKEN_NUMERIC_DIM, dtype=np.float32)

            risk_numeric[: min(obs_common.ROUTE_SUMMARY_DIM, TOKEN_NUMERIC_DIM)] = route_summary[:TOKEN_NUMERIC_DIM]
            risk_numeric[20] = route_summary[5] if route_summary.shape[0] > 5 else 0.0
            risk_numeric[21] = route_summary[6] if route_summary.shape[0] > 6 else 0.0
            risk_numeric[22] = route_summary[12] if route_summary.shape[0] > 12 else 0.0
            risk_numeric[23] = route_summary[18] if route_summary.shape[0] > 18 else 0.0
            risk_numeric[24] = min(obs_common._float((player or {}).get("hp")) / max(obs_common._float((player or {}).get("max_hp")), 1.0), 1.0) if isinstance(player, dict) else 0.0
            risk_numeric[25] = obs_common._log_norm(obs_common._float((player or {}).get("gold")), obs_common._LOG1P_500)

            value_numeric[: min(obs_common.ROUTE_SUMMARY_DIM, TOKEN_NUMERIC_DIM)] = route_summary[:TOKEN_NUMERIC_DIM]
            value_numeric[20] = route_summary[9] if route_summary.shape[0] > 9 else 0.0
            value_numeric[21] = route_summary[10] if route_summary.shape[0] > 10 else 0.0
            value_numeric[22] = route_summary[11] if route_summary.shape[0] > 11 else 0.0
            value_numeric[23] = route_summary[7] if route_summary.shape[0] > 7 else 0.0
            value_numeric[24] = route_summary[14] if route_summary.shape[0] > 14 else 0.0
            value_numeric[25] = route_summary[15] if route_summary.shape[0] > 15 else 0.0

            entries.append(
                self._entry(
                    "ROUTE_RISK_LOCAL",
                    risk_numeric,
                    owner_id=OWNER_ROUTE,
                    entity_id=_stable_bucket(f"route-risk:{action_index}"),
                    text=f"route risk | action {action_index}",
                )
            )
            entries.append(
                self._entry(
                    "ROUTE_VALUE_LOCAL",
                    value_numeric,
                    owner_id=OWNER_ROUTE,
                    entity_id=_stable_bucket(f"route-value:{action_index}"),
                    text=f"route value | action {action_index}",
                )
            )

    @staticmethod
    def _resolve_build_candidate_card(action: dict[str, Any]) -> dict[str, Any] | None:
        for candidate in (
            action.get("card"),
            action.get("upgrade_preview"),
            (action.get("item") or {}).get("card") if isinstance(action.get("item"), dict) else None,
            (action.get("reward") or {}).get("card") if isinstance(action.get("reward"), dict) else None,
        ):
            if isinstance(candidate, dict):
                return candidate
        return None

    def _deck_synergy_numeric(self, deck_cards: list[Any], candidate_card: dict[str, Any]) -> np.ndarray:
        numeric = np.zeros(TOKEN_NUMERIC_DIM, dtype=np.float32)
        deck_count = len(deck_cards) if isinstance(deck_cards, list) else 0
        if deck_count <= 0:
            source_profile = self._source_profile(candidate_card)
            numeric[0] = min(source_profile["damage"] / 40.0, 1.0)
            numeric[1] = min(source_profile["block"] / 40.0, 1.0)
            numeric[2] = min(source_profile["draw"] / 5.0, 1.0)
            numeric[3] = min(source_profile["energy"] / 5.0, 1.0)
            return numeric

        candidate_profile = self._source_profile(candidate_card)
        candidate_id = str(candidate_card.get("id") or "").strip()
        candidate_title = str(candidate_card.get("title") or "").strip().lower()
        same_count = 0.0
        attack = skill = power = 0.0
        total_cost = total_damage = total_block = total_draw = total_energy = total_exhaust = 0.0

        for card in deck_cards:
            if not isinstance(card, dict):
                continue
            profile = self._source_profile(card)
            attack += profile["attack"]
            skill += profile["skill"]
            power += profile["power"]
            total_cost += profile["cost"]
            total_damage += profile["damage"]
            total_block += profile["block"]
            total_draw += profile["draw"]
            total_energy += profile["energy"]
            total_exhaust += profile["exhaust"]
            deck_id = str(card.get("id") or "").strip()
            deck_title = str(card.get("title") or "").strip().lower()
            if (candidate_id and deck_id and candidate_id == deck_id) or (candidate_title and deck_title and candidate_title == deck_title):
                same_count += 1.0

        denom = max(float(deck_count), 1.0)
        avg_damage = total_damage / denom
        avg_block = total_block / denom
        avg_draw = total_draw / denom
        avg_energy = total_energy / denom
        avg_cost = total_cost / denom
        avg_exhaust = total_exhaust / denom

        numeric[0] = min(deck_count / 50.0, 1.0)
        numeric[1] = attack / denom
        numeric[2] = skill / denom
        numeric[3] = power / denom
        numeric[4] = min(avg_cost / 5.0, 1.0)
        numeric[5] = obs_common._log_norm(avg_damage, obs_common._LOG1P_100)
        numeric[6] = obs_common._log_norm(avg_block, obs_common._LOG1P_100)
        numeric[7] = min(avg_draw / 5.0, 1.0)
        numeric[8] = min(avg_energy / 5.0, 1.0)
        numeric[9] = min(avg_exhaust, 1.0)
        numeric[10] = min(same_count / 4.0, 1.0)
        numeric[11] = float(candidate_profile["attack"] > 0.0 and avg_damage < 10.0)
        numeric[12] = float(candidate_profile["block"] > 0.0 and avg_block < 8.0)
        numeric[13] = float(candidate_profile["draw"] > 0.0 and avg_draw < 1.0)
        numeric[14] = float(candidate_profile["energy"] > 0.0 and avg_energy < 0.5)
        numeric[15] = float(candidate_profile["power"] > 0.0 and power / denom < 0.15)
        numeric[16] = float(candidate_profile["exhaust"] > 0.0 and avg_exhaust < 0.2)
        numeric[17] = candidate_profile["zero_cost"]
        numeric[18] = candidate_profile["x_cost"]
        numeric[19] = obs_common._log_norm(candidate_profile["damage"], obs_common._LOG1P_100)
        numeric[20] = obs_common._log_norm(candidate_profile["block"], obs_common._LOG1P_100)
        numeric[21] = min(candidate_profile["draw"] / 5.0, 1.0)
        numeric[22] = min(candidate_profile["energy"] / 5.0, 1.0)
        numeric[23] = float(candidate_profile["retain"] > 0.0)
        return numeric

    def _append_pile_context(
        self,
        entries: list[dict[str, Any]],
        token_type: str,
        owner_id: int,
        cards: list[Any],
        source_card: dict[str, Any] | None,
        pile_label: str,
    ) -> None:
        numeric = self._pile_context_numeric(cards, source_card)
        top_titles = ", ".join(
            str(card.get("title") or card.get("id") or "").strip()
            for card in cards[:2]
            if isinstance(card, dict) and str(card.get("title") or card.get("id") or "").strip()
        )
        text = f"{pile_label} pile | {len(cards)} cards"
        if top_titles:
            text = f"{text} | top {top_titles}"
        entries.append(self._entry(token_type, numeric, owner_id=owner_id, entity_id=_stable_bucket(f"{pile_label}:context"), text=text))

    def _append_source_pile_binding_locals(
        self,
        entries: list[dict[str, Any]],
        obs: dict[str, Any],
        source_card: dict[str, Any] | None,
    ) -> None:
        if not isinstance(source_card, dict):
            return
        source_entity_id = _stable_bucket(source_card.get("id") or source_card.get("title") or "source-card")
        for token_type, owner_id, pile_key, fallback_key, pile_label in (
            ("DRAW_BINDING_LOCAL", OWNER_DRAW, "draw_pile", "draw_preview_cards", "draw"),
            ("DISCARD_BINDING_LOCAL", OWNER_DISCARD, "discard_pile", "discard_cards", "discard"),
            ("EXHAUST_BINDING_LOCAL", OWNER_EXHAUST, "exhaust_pile", "exhaust_cards", "exhaust"),
            ("PLAY_BINDING_LOCAL", OWNER_PLAY, "play_pile", "play_pile_cards", "play"),
        ):
            cards = self._runtime_cards(obs, pile_key, fallback_key)
            numeric, closest_position, same_count = self._pile_binding_numeric(cards, source_card)
            entries.append(
                self._entry(
                    token_type,
                    numeric,
                    owner_id=owner_id,
                    entity_id=source_entity_id,
                    order_id=closest_position,
                    text=f"{pile_label} bind | same {same_count:.0f} | closest {closest_position}",
                )
            )

    def _pile_binding_numeric(
        self,
        cards: list[Any],
        source_card: dict[str, Any],
    ) -> tuple[np.ndarray, int, float]:
        numeric = np.zeros(TOKEN_NUMERIC_DIM, dtype=np.float32)
        if not isinstance(source_card, dict):
            return numeric, 0, 0.0

        source_profile = self._source_profile(source_card)
        source_text = self._source_text(source_card)
        source_id = str(source_card.get("id") or "").strip()
        source_title = str(source_card.get("title") or "").strip().lower()
        count = len(cards) if isinstance(cards, list) else 0
        if count <= 0:
            numeric[6] = source_profile["zero_cost"]
            numeric[7] = min(source_profile["draw"] / 5.0, 1.0)
            numeric[8] = min(source_profile["energy"] / 5.0, 1.0)
            numeric[9] = min(source_profile["hits"] / 10.0, 1.0)
            numeric[10] = source_profile["exhaust"]
            numeric[11] = source_profile["retain"]
            numeric[17] = float(any(keyword in source_text for keyword in ("discard", "draw pile", "shuffle", "return to your hand")))
            numeric[18] = float("exhaust" in source_text or "ethereal" in source_text)
            return numeric, 0, 0.0

        same_id_count = 0.0
        same_title_count = 0.0
        closest_position = 0
        top_match = 0.0
        near_match = 0.0
        zero_cost = 0.0
        draw_density = 0.0
        exhaust_density = 0.0
        retain_density = 0.0
        energy_density = 0.0
        total_cost = 0.0

        for index, card in enumerate(cards):
            if not isinstance(card, dict):
                continue
            profile = self._source_profile(card)
            entry_id = str(card.get("id") or "").strip()
            entry_title = str(card.get("title") or "").strip().lower()
            matched = False
            if source_id and entry_id and entry_id == source_id:
                same_id_count += 1.0
                matched = True
            if source_title and entry_title and entry_title == source_title:
                same_title_count += 1.0
                matched = True
            if matched and closest_position <= 0:
                closest_position = index + 1
                near_match = float(index < 3)
            if matched and index == 0:
                top_match = 1.0
            zero_cost += profile["zero_cost"]
            draw_density += float(profile["draw"] > 0.0)
            exhaust_density += float(profile["exhaust"] > 0.0 or profile["ethereal"] > 0.0)
            retain_density += profile["retain"]
            energy_density += float(profile["energy"] > 0.0)
            total_cost += max(profile["cost"], 0.0)

        denom = max(float(count), 1.0)
        numeric[0] = min(count / 30.0, 1.0)
        numeric[1] = min(same_id_count / 4.0, 1.0)
        numeric[2] = min(same_title_count / 4.0, 1.0)
        numeric[3] = min(closest_position / 10.0, 1.0) if closest_position > 0 else 0.0
        numeric[4] = top_match
        numeric[5] = near_match
        numeric[6] = source_profile["zero_cost"]
        numeric[7] = min(source_profile["draw"] / 5.0, 1.0)
        numeric[8] = min(max(source_profile["energy"], float(source_profile["x_cost"] > 0.5)) / 5.0, 1.0)
        numeric[9] = min(source_profile["hits"] / 10.0, 1.0)
        numeric[10] = source_profile["exhaust"]
        numeric[11] = source_profile["retain"]
        numeric[12] = zero_cost / denom
        numeric[13] = draw_density / denom
        numeric[14] = exhaust_density / denom
        numeric[15] = retain_density / denom
        numeric[16] = min(total_cost / denom / 5.0, 1.0)
        numeric[17] = float(any(keyword in source_text for keyword in ("discard", "draw pile", "shuffle", "return to your hand")))
        numeric[18] = float("exhaust" in source_text or "ethereal" in source_text)
        numeric[19] = float((same_id_count + same_title_count) > 0.0)
        return numeric, closest_position, max(same_id_count, same_title_count)

    def _append_cycle_plan_local(
        self,
        entries: list[dict[str, Any]],
        obs: dict[str, Any],
        action: dict[str, Any],
        source_card: dict[str, Any] | None,
    ) -> None:
        source = source_card if isinstance(source_card, dict) else action.get("potion") if isinstance(action.get("potion"), dict) else None
        source_profile = self._source_profile(source)
        source_text = self._source_text(source)
        hand_cards = self._runtime_cards(obs, "hand", "hand")
        draw_cards = self._runtime_cards(obs, "draw_pile", "draw_preview_cards")
        discard_cards = self._runtime_cards(obs, "discard_pile", "discard_cards")
        exhaust_cards = self._runtime_cards(obs, "exhaust_pile", "exhaust_cards")
        play_cards = self._runtime_cards(obs, "play_pile", "play_pile_cards")

        draw_match = self._find_source_position(draw_cards, source)
        discard_match = self._find_source_position(discard_cards, source)
        exhaust_match = self._find_source_position(exhaust_cards, source)
        play_match = self._find_source_position(play_cards, source)
        draw_count = len(draw_cards)
        discard_count = len(discard_cards)
        exhaust_count = len(exhaust_cards)
        play_count = len(play_cards)
        hand_count = len(hand_cards)

        reshuffle_pressure = float(draw_count <= 2 and discard_count >= 4)
        cycle_keyword = float(
            any(
                keyword in source_text
                for keyword in (
                    "draw",
                    "discard",
                    "shuffle",
                    "return to your hand",
                    "draw pile",
                    "discard pile",
                )
            )
        )
        exhaust_keyword = float("exhaust" in source_text or "ethereal" in source_text)
        numeric = np.zeros(TOKEN_NUMERIC_DIM, dtype=np.float32)
        numeric[0] = min(hand_count / 10.0, 1.0)
        numeric[1] = min(draw_count / 30.0, 1.0)
        numeric[2] = min(discard_count / 30.0, 1.0)
        numeric[3] = min(exhaust_count / 20.0, 1.0)
        numeric[4] = min(play_count / 12.0, 1.0)
        numeric[5] = reshuffle_pressure
        numeric[6] = float(draw_count <= 3)
        numeric[7] = float(discard_count >= 4)
        numeric[8] = min(draw_match / 10.0, 1.0) if draw_match > 0 else 0.0
        numeric[9] = min(discard_match / 10.0, 1.0) if discard_match > 0 else 0.0
        numeric[10] = min(exhaust_match / 10.0, 1.0) if exhaust_match > 0 else 0.0
        numeric[11] = min(play_match / 10.0, 1.0) if play_match > 0 else 0.0
        numeric[12] = min(source_profile["draw"] / 5.0, 1.0)
        numeric[13] = source_profile["zero_cost"]
        numeric[14] = source_profile["exhaust"]
        numeric[15] = source_profile["retain"]
        numeric[16] = cycle_keyword
        numeric[17] = float(any(keyword in source_text for keyword in ("return", "shuffle", "draw pile", "discard pile")))
        numeric[18] = exhaust_keyword
        numeric[19] = float((cycle_keyword > 0.0 or source_profile["zero_cost"] > 0.0) and (discard_count > 0 or draw_count <= 3))
        numeric[20] = float((discard_match > 0 or draw_match > 0) and cycle_keyword > 0.0)
        numeric[21] = float(exhaust_match > 0 and exhaust_keyword > 0.0)
        numeric[22] = float(source_profile["energy"] > 0.0 and draw_count <= 3)
        numeric[23] = float(source_profile["hits"] > 1.0 and play_count > 0)
        source_entity_id = _stable_bucket(source.get("id") or source.get("title") or self._action_entity_key(action)) if isinstance(source, dict) else _stable_bucket(self._action_entity_key(action))
        entries.append(
            self._entry(
                "CYCLE_PLAN_LOCAL",
                numeric,
                owner_id=OWNER_PLAYER,
                entity_id=source_entity_id,
                order_id=6,
                text=f"cycle plan | draw {draw_count} discard {discard_count} exhaust {exhaust_count} play {play_count}",
            )
        )

    def _append_energy_budget_local(
        self,
        entries: list[dict[str, Any]],
        obs: dict[str, Any],
        features: dict[str, np.ndarray],
        action: dict[str, Any],
        source_profile: dict[str, float],
    ) -> None:
        combat = obs.get("combat") or {}
        player = obs.get("player") or {}
        current_energy = obs_common._float(combat.get("energy"))
        max_energy = obs_common._float(combat.get("max_energy"))
        source_cost = max(source_profile["cost"], 0.0)
        source_is_x = float(source_profile["x_cost"] > 0.5)

        best_energy_potion = 0.0
        energy_potion_count = 0.0
        for potion in (player.get("potions") or [])[: obs_common.MAX_POTIONS]:
            if isinstance(potion, str) and potion.strip() == "[empty]":
                continue
            potion_profile = self._source_profile(potion if isinstance(potion, dict) else None)
            potion_energy = max(potion_profile["energy"], obs_common._float(potion.get("energy")) if isinstance(potion, dict) else 0.0)
            if potion_energy > 0.0:
                energy_potion_count += 1.0
                best_energy_potion = max(best_energy_potion, potion_energy)

        relic_signals = features["relic_signals"] if "relic_signals" in features else np.zeros(obs_common.RELIC_SIGNAL_DIM, dtype=np.float32)
        relic_energy_signal = float(relic_signals[0]) if relic_signals.shape[0] > 0 else 0.0
        relic_draw_signal = float(relic_signals[1]) if relic_signals.shape[0] > 1 else 0.0
        extra_energy = source_profile["energy"] + best_energy_potion + 2.0 * relic_energy_signal

        can_play_now = float((source_is_x > 0.0 and current_energy > 0.0) or current_energy >= source_cost)
        can_expand_with_support = float((source_is_x > 0.0 and current_energy + extra_energy > 0.0) or current_energy + extra_energy >= source_cost)
        max_spend = current_energy + best_energy_potion + max(source_profile["energy"], 0.0)

        numeric = np.zeros(TOKEN_NUMERIC_DIM, dtype=np.float32)
        numeric[0] = min(current_energy / max(max_energy, 1.0), 1.0) if max_energy > 0 else 0.0
        numeric[1] = min(current_energy / 10.0, 1.0)
        numeric[2] = min(max_energy / 10.0, 1.0)
        numeric[3] = min(source_cost / 5.0, 1.0)
        numeric[4] = source_is_x
        numeric[5] = min(source_profile["energy"] / 5.0, 1.0)
        numeric[6] = min(best_energy_potion / 5.0, 1.0)
        numeric[7] = min(energy_potion_count / max(obs_common.MAX_POTIONS, 1), 1.0)
        numeric[8] = relic_energy_signal
        numeric[9] = relic_draw_signal
        numeric[10] = can_play_now
        numeric[11] = can_expand_with_support
        numeric[12] = min(max_spend / 10.0, 1.0)
        numeric[13] = source_profile["zero_cost"]
        numeric[14] = min(source_profile["draw"] / 5.0, 1.0)
        numeric[15] = min(source_profile["hits"] / 10.0, 1.0)
        numeric[16] = source_profile["attack"]
        numeric[17] = max(source_profile["skill"], float(source_profile["block"] > 0.0))
        numeric[18] = float(source_is_x > 0.0 and max_spend >= 3.0)
        numeric[19] = float(source_profile["energy"] > 0.0 and current_energy < source_cost)
        entries.append(
            self._entry(
                "ENERGY_BUDGET_LOCAL",
                numeric,
                owner_id=OWNER_PLAYER,
                entity_id=_stable_bucket(self._action_entity_key(action)),
                order_id=3,
                text=f"energy budget | now {int(current_energy)} | support {best_energy_potion:.0f} | x {int(source_is_x)}",
            )
        )

    def _append_relic_potion_graph_local(
        self,
        entries: list[dict[str, Any]],
        obs: dict[str, Any],
        action: dict[str, Any],
        features: dict[str, np.ndarray],
        target_enemy_index: int | None,
        target_enemy: dict[str, Any] | None,
    ) -> None:
        player = obs.get("player") or {}
        source = action.get("card") if isinstance(action.get("card"), dict) else action.get("potion") if isinstance(action.get("potion"), dict) else None
        source_profile = self._source_profile(source)

        relic_scores: list[float] = []
        for index, relic in enumerate((player.get("relics") or [])[: obs_common.MAX_RELICS]):
            if not isinstance(relic, (dict, str)):
                continue
            relic_scores.append(float(self._relic_numeric(relic, action=action, index=index, total=max(len(player.get("relics") or []), 1))[0]))

        potion_scores: list[float] = []
        best_energy_potion = 0.0
        best_damage_potion = 0.0
        best_block_potion = 0.0
        best_draw_potion = 0.0
        for index, potion in enumerate((player.get("potions") or [])[: obs_common.MAX_POTIONS]):
            title = potion if isinstance(potion, str) else (potion.get("title") if isinstance(potion, dict) else "")
            if str(title or "").strip() == "[empty]":
                continue
            potion_numeric = self._potion_numeric(potion, source_profile=source_profile, slot_index=index, total_slots=max(len(player.get("potions") or []), 1))
            potion_scores.append(float(potion_numeric[0]))
            best_damage_potion = max(best_damage_potion, float(potion_numeric[1]))
            best_block_potion = max(best_block_potion, float(potion_numeric[2]))
            best_draw_potion = max(best_draw_potion, float(potion_numeric[3]))
            best_energy_potion = max(best_energy_potion, float(potion_numeric[4]))

        relic_signals = features["relic_signals"] if "relic_signals" in features else np.zeros(obs_common.RELIC_SIGNAL_DIM, dtype=np.float32)
        enemy_reactions = self._enemy_reaction_flags(target_enemy)

        top_relic = max(relic_scores, default=0.0)
        top_potion = max(potion_scores, default=0.0)
        avg_relic = float(np.mean(sorted(relic_scores, reverse=True)[:3])) if relic_scores else 0.0
        avg_potion = float(np.mean(sorted(potion_scores, reverse=True)[:2])) if potion_scores else 0.0

        numeric = np.zeros(TOKEN_NUMERIC_DIM, dtype=np.float32)
        numeric[0] = top_relic
        numeric[1] = avg_relic
        numeric[2] = top_potion
        numeric[3] = avg_potion
        numeric[4] = best_energy_potion
        numeric[5] = best_damage_potion
        numeric[6] = best_block_potion
        numeric[7] = best_draw_potion
        numeric[8] = float(relic_signals[0]) if relic_signals.shape[0] > 0 else 0.0
        numeric[9] = float(relic_signals[1]) if relic_signals.shape[0] > 1 else 0.0
        numeric[10] = max(float(relic_signals[2]) if relic_signals.shape[0] > 2 else 0.0, float(relic_signals[4]) if relic_signals.shape[0] > 4 else 0.0)
        numeric[11] = max(float(relic_signals[3]) if relic_signals.shape[0] > 3 else 0.0, float(relic_signals[10]) if relic_signals.shape[0] > 10 else 0.0)
        numeric[12] = enemy_reactions["contact_punish"] * max(best_damage_potion, best_block_potion, best_energy_potion) * max(source_profile["attack"], float(source_profile["damage"] > 0.0))
        numeric[13] = float(np.clip(top_relic + top_potion, 0.0, 1.0))
        numeric[14] = source_profile["attack"]
        numeric[15] = max(source_profile["skill"], float(source_profile["block"] > 0.0))
        numeric[16] = float(source_profile["x_cost"] > 0.5) * max(best_energy_potion, numeric[8])
        numeric[17] = max(source_profile["exhaust"], float("exhaust" in self._source_text(source))) * max(top_relic, top_potion)
        numeric[18] = float(target_enemy_index is not None and enemy_reactions["contact_punish"] > 0.0)
        numeric[19] = float(source_profile["hits"] > 1.0) * enemy_reactions["thorns"]
        entries.append(
            self._entry(
                "RELIC_POTION_GRAPH_LOCAL",
                numeric,
                owner_id=OWNER_PLAYER,
                entity_id=_stable_bucket(self._action_entity_key(action)),
                order_id=7,
                text=f"support graph | relic {top_relic:.2f} potion {top_potion:.2f} contact {enemy_reactions['contact_punish']:.0f}",
            )
        )

    def _pile_context_numeric(self, cards: list[Any], source_card: dict[str, Any] | None) -> np.ndarray:
        numeric = np.zeros(TOKEN_NUMERIC_DIM, dtype=np.float32)
        count = len(cards) if isinstance(cards, list) else 0
        if count <= 0:
            return numeric

        source_id = str((source_card or {}).get("id") or "").strip()
        source_title = str((source_card or {}).get("title") or "").strip().lower()
        attack = skill = power = status_or_curse = 0.0
        zero_cost = exhaust_kw = ethereal_kw = retain_kw = innate_kw = 0.0
        same_source = 0.0
        total_cost = total_damage = total_block = total_draw = total_energy = total_hits = 0.0

        for card in cards:
            if not isinstance(card, dict):
                continue
            card_type = str(card.get("type") or "").capitalize()
            attack += 1.0 if card_type == "Attack" else 0.0
            skill += 1.0 if card_type == "Skill" else 0.0
            power += 1.0 if card_type == "Power" else 0.0
            status_or_curse += 1.0 if card_type in {"Status", "Curse"} else 0.0
            cost = obs_common._float(card.get("cost"))
            total_cost += max(cost, 0.0)
            zero_cost += 1.0 if cost == 0 else 0.0
            preview = obs_common._build_card_preview_bundle(card)
            total_damage += preview["preview_damage"]
            total_block += preview["preview_block"]
            total_draw += obs_common._preview_metric(card, "draw")
            total_energy += obs_common._preview_metric(card, "energy")
            total_hits += obs_common._get_card_extra_metrics(card)[3]
            kw_flags, _ = obs_common._get_card_keywords(card)
            exhaust_kw += 1.0 if kw_flags[0] else 0.0
            ethereal_kw += 1.0 if kw_flags[1] else 0.0
            retain_kw += 1.0 if kw_flags[2] else 0.0
            innate_kw += 1.0 if kw_flags[3] else 0.0
            card_id = str(card.get("id") or "").strip()
            card_title = str(card.get("title") or "").strip().lower()
            if (source_id and card_id and card_id == source_id) or (source_title and card_title and card_title == source_title):
                same_source += 1.0

        denom = max(float(count), 1.0)
        numeric[0] = min(count / 30.0, 1.0)
        numeric[1] = attack / denom
        numeric[2] = skill / denom
        numeric[3] = power / denom
        numeric[4] = status_or_curse / denom
        numeric[5] = min(total_cost / denom / 5.0, 1.0)
        numeric[6] = zero_cost / denom
        numeric[7] = exhaust_kw / denom
        numeric[8] = ethereal_kw / denom
        numeric[9] = retain_kw / denom
        numeric[10] = innate_kw / denom
        numeric[11] = same_source / denom
        numeric[12] = obs_common._log_norm(total_damage, obs_common._LOG1P_200)
        numeric[13] = obs_common._log_norm(total_block, obs_common._LOG1P_200)
        numeric[14] = min(total_draw / 10.0, 1.0)
        numeric[15] = min(total_energy / 10.0, 1.0)
        numeric[16] = min(total_hits / 20.0, 1.0)
        return numeric

    def _append_pile_peek_locals(
        self,
        entries: list[dict[str, Any]],
        token_type: str,
        owner_id: int,
        cards: list[Any],
        *,
        limit: int,
    ) -> None:
        if not isinstance(cards, list) or not cards:
            return
        numeric = np.zeros((limit, obs_common.CARD_FEAT_DIM), dtype=np.float32)
        text = np.zeros((limit, TEXT_DIM), dtype=np.float32)
        mask = np.zeros(limit, dtype=np.float32)
        self._enc_card_collection(cards[:limit], numeric, text, mask)
        for index in range(limit):
            if mask[index] <= 0:
                continue
            row = np.zeros(TOKEN_NUMERIC_DIM, dtype=np.float32)
            row[: min(TOKEN_NUMERIC_DIM, numeric.shape[1])] = numeric[index][:TOKEN_NUMERIC_DIM]
            row[min(TOKEN_NUMERIC_DIM - 1, numeric.shape[1])] = min((index + 1) / max(limit, 1), 1.0)
            card = cards[index] if index < len(cards) and isinstance(cards[index], dict) else None
            entity_key = card.get("id") if isinstance(card, dict) and card.get("id") else card.get("title") if isinstance(card, dict) else f"{token_type}:{index}"
            entries.append(
                self._entry(
                    token_type,
                    row,
                    owner_id=owner_id,
                    entity_id=_stable_bucket(entity_key),
                    order_id=index + 1,
                    text_embedding=text[index],
                )
            )

    def _append_target_reaction_local(
        self,
        entries: list[dict[str, Any]],
        target_enemy_index: int,
        target_enemy: dict[str, Any],
        source: dict[str, Any] | None,
    ) -> None:
        source_profile = self._source_profile(source)
        reactions = self._enemy_reaction_flags(target_enemy)
        hp = obs_common._float(target_enemy.get("hp", target_enemy.get("current_hp")))
        block = obs_common._float(target_enemy.get("block"))
        intent = target_enemy.get("intent") if isinstance(target_enemy.get("intent"), dict) else {}
        expected_damage = source_profile["damage"] * max(source_profile["hits"], 1.0 if source_profile["damage"] > 0.0 else 0.0)
        attack_like = max(source_profile["attack"], float(source_profile["damage"] > 0.0))

        numeric = np.zeros(TOKEN_NUMERIC_DIM, dtype=np.float32)
        numeric[0] = reactions["thorns"]
        numeric[1] = reactions["contact_punish"]
        numeric[2] = reactions["split"]
        numeric[3] = reactions["threshold"]
        numeric[4] = reactions["threshold_value"]
        numeric[5] = attack_like
        numeric[6] = float(source_profile["hits"] > 1.0)
        numeric[7] = reactions["contact_punish"] * attack_like
        numeric[8] = reactions["contact_punish"] * float(source_profile["hits"] > 1.0)
        numeric[9] = float(expected_damage >= max(hp + block, 1.0) and expected_damage > 0.0)
        numeric[10] = float(obs_common._float(intent.get("total_damage")) > 0.0)
        numeric[11] = obs_common._log_norm(obs_common._float(intent.get("total_damage")), obs_common._LOG1P_200)
        numeric[12] = obs_common._log_norm(block, obs_common._LOG1P_200)
        numeric[13] = obs_common._log_norm(hp, obs_common._LOG1P_1200)
        numeric[14] = float(source_profile["x_cost"] > 0.5)
        numeric[15] = max(reactions["artifact"], reactions["buffer"], reactions["intangible"])
        numeric[16] = float(source_profile["weak"] > 0.0 or source_profile["vulnerable"] > 0.0)
        numeric[17] = float(source_profile["aoe_target"] > 0.0)
        numeric[18] = reactions["incoming_damage_multiplier"]
        numeric[19] = reactions["back_attack"]
        numeric[20] = reactions["damage_cap"]
        numeric[21] = reactions["damage_cap_value"]
        numeric[22] = reactions["deathburst"]
        numeric[23] = reactions["deathburst_damage"]
        numeric[24] = reactions["revive"]
        numeric[25] = reactions["transform"]
        numeric[26] = reactions["linked_support_alive"]
        numeric[27] = reactions["special_phase"]
        numeric[28] = reactions["one_card_lock"]
        numeric[29] = reactions["skill_punish"]
        numeric[30] = reactions["choice_debuffs"]
        numeric[31] = reactions["escape_card_tax"]
        numeric[32] = reactions["countdown"]
        numeric[33] = reactions["stun_window"]
        numeric[34] = reactions["binding_control"]
        numeric[35] = reactions["wound_phase"]
        entries.append(
            self._entry(
                "TARGET_REACTION_LOCAL",
                numeric,
                owner_id=self._enemy_owner_id(target_enemy_index),
                entity_id=_stable_bucket(self._enemy_entity_key(target_enemy, target_enemy_index)),
                order_id=24,
                text=(
                    "target reaction"
                    f" | thorns {reactions['thorns']:.0f}"
                    f" | threshold {reactions['threshold']:.0f}"
                    f" | back {reactions['back_attack']:.0f}"
                    f" | cap {reactions['damage_cap']:.0f}"
                    f" | burst {reactions['deathburst']:.0f}"
                ),
            )
        )

    def _append_target_enemy_local_context(
        self,
        entries: list[dict[str, Any]],
        target_enemy_index: int,
        target_enemy: dict[str, Any],
    ) -> None:
        owner_id = self._enemy_owner_id(target_enemy_index)
        entity_id = _stable_bucket(self._enemy_entity_key(target_enemy, target_enemy_index))
        combat_memory = (self._current_planner_context or {}).get("combat_memory") or {}
        memory_enemies = combat_memory.get("enemies") if isinstance(combat_memory, dict) else None
        enemy_memory = None
        if isinstance(memory_enemies, dict):
            enemy_memory = memory_enemies.get(self._enemy_entity_key(target_enemy, target_enemy_index))
        intent = target_enemy.get("intent") if isinstance(target_enemy.get("intent"), dict) else None
        if isinstance(intent, dict):
            numeric = np.zeros(TOKEN_NUMERIC_DIM, dtype=np.float32)
            numeric[0] = obs_common._log_norm(obs_common._float(intent.get("total_damage")), obs_common._LOG1P_200)
            numeric[1] = obs_common._log_norm(obs_common._float(intent.get("damage_per_hit")), obs_common._LOG1P_100)
            numeric[2] = min(obs_common._float(intent.get("repeats")) / 5.0, 1.0)
            numeric[3:7] = np.asarray(obs_common._infer_enemy_intent_flags(intent, obs_common._float(intent.get("total_damage"))), dtype=np.float32)
            if isinstance(enemy_memory, dict):
                numeric[7] = obs_common._float(enemy_memory.get("intent_changed_this_turn"))
                numeric[8] = obs_common._float(enemy_memory.get("turns_since_intent_change_norm"))
                numeric[9] = obs_common._float(enemy_memory.get("intent_total_damage_delta"))
                numeric[10] = obs_common._float(enemy_memory.get("intent_damage_per_hit_delta"))
                numeric[11] = obs_common._float(enemy_memory.get("intent_damage_trend_3_turns"))
                numeric[12] = obs_common._float(enemy_memory.get("intent_predicted_vs_actual"))
            entries.append(
                self._entry(
                    "ENEMY_INTENT",
                    numeric,
                    owner_id=owner_id,
                    entity_id=entity_id,
                    order_id=2,
                    text=str(intent.get("description") or intent.get("label") or ""),
                )
            )
        memory_powers = enemy_memory.get("powers") if isinstance(enemy_memory, dict) else None
        for power_index, power in enumerate((target_enemy.get("powers") or [])[:2]):
            if not isinstance(power, dict):
                continue
            numeric = np.zeros(TOKEN_NUMERIC_DIM, dtype=np.float32)
            amount = obs_common._float(power.get("amount") or power.get("display_amount"))
            title = str(power.get("title") or "").lower()
            numeric[0] = obs_common._normalize_power_amount(amount)
            numeric[1] = float("thorn" in title or "spike" in title)
            numeric[2] = float("artifact" in title)
            numeric[3] = float("buffer" in title)
            numeric[4] = float("intang" in title)
            power_key = None
            for candidate in ("id", "power_id", "type", "key"):
                value = power.get(candidate)
                if value not in (None, ""):
                    power_key = str(value)
                    break
            if power_key is None and power.get("title") not in (None, ""):
                power_key = f"title::{power.get('title')}"
            power_memory = memory_powers.get(power_key) if isinstance(memory_powers, dict) and power_key else None
            if isinstance(power_memory, dict):
                delta_last = obs_common._float(power_memory.get("amount_delta_last_turn"))
                delta_since = obs_common._float(power_memory.get("amount_delta_since_first_seen"))
                numeric[5] = obs_common._normalize_power_amount(delta_last) * (1.0 if delta_last >= 0 else -1.0)
                numeric[6] = obs_common._normalize_power_amount(delta_since) * (1.0 if delta_since >= 0 else -1.0)
                numeric[7] = min(obs_common._float(power_memory.get("turns_since_first_seen")) / 10.0, 1.0)
                numeric[8] = obs_common._float(power_memory.get("stack_trend_3_turns"))
                numeric[9] = obs_common._float(power_memory.get("is_new_this_turn"))
                numeric[10] = obs_common._float(power_memory.get("is_growing_without_player_action"))
            entries.append(
                self._entry(
                    "ENEMY_POWER",
                    numeric,
                    owner_id=owner_id,
                    entity_id=entity_id,
                    order_id=3 + power_index,
                    text=f"{power.get('title') or ''} | {power.get('description') or ''}",
                )
            )
        reactive_traits, phase_rules = self._infer_enemy_traits(target_enemy)
        for trait_index, trait in enumerate(reactive_traits[:2]):
            entries.append(
                self._entry(
                    "ENEMY_REACTIVE_TRAIT",
                    self._trait_numeric(trait),
                    owner_id=owner_id,
                    entity_id=entity_id,
                    order_id=12 + trait_index,
                    text=str(trait.get("description") or trait.get("trait") or trait.get("effect_type") or ""),
                )
            )
        for rule_index, rule in enumerate(phase_rules[:1]):
            entries.append(
                self._entry(
                    "ENEMY_PHASE_RULE",
                    self._trait_numeric(rule),
                    owner_id=owner_id,
                    entity_id=entity_id,
                    order_id=20 + rule_index,
                    text=str(rule.get("description") or rule.get("trait") or ""),
                )
            )

    def _append_power_slot_tokens(
        self,
        world_entries: list[dict[str, Any]],
        obs: dict[str, Any],
    ) -> None:
        """Emit one POWER_SLOT token per active buff/debuff instance.

        Phase 6.1: splits the previously-inlined per-entity power numerics
        into first-class tokens. Each token exposes the power's categorical
        id (via entity_id bucket) and its effect-algebra coefficients (via
        numeric vector) so that attention can learn interactions directly
        (e.g. "this card's damage 脳 target's damage_mult_received").

        The original inline power features in PLAYER_SURVIVAL / ENEMY tokens
        are preserved for back-compat; POWER_SLOT tokens add a richer
        per-power view on top. Phase 6.2/6.3 will wire a dedicated POWER
        bank + effect_class 脳 target_power_bucket relational bias that
        consumes these tokens exclusively.
        """
        emitted = 0
        budget = MAX_POWER_SLOT_TOKENS

        # ---- Player-owned powers ----
        player = obs.get("player") or {}
        player_powers = player.get("status") or player.get("powers") or []
        if isinstance(player_powers, list):
            for slot_index, power in enumerate(player_powers):
                if emitted >= budget:
                    break
                if not isinstance(power, dict):
                    continue
                token = self._build_power_slot_numeric(power)
                if token is None:
                    continue
                world_entries.append(
                    self._entry(
                        "POWER_SLOT_PLAYER",
                        token["numeric"],
                        owner_id=OWNER_PLAYER,
                        entity_id=token["bucket"],
                        order_id=min(slot_index, MAX_ORDER_ID),
                        text=token["label"],
                    )
                )
                emitted += 1

        # ---- Enemy-owned powers ----
        combat = obs.get("combat") or {}
        enemies = combat.get("enemies") or []
        if isinstance(enemies, list):
            for enemy_index, enemy in enumerate(enemies):
                if emitted >= budget:
                    break
                if not isinstance(enemy, dict):
                    continue
                owner_id = self._enemy_owner_id(enemy_index)
                enemy_powers = enemy.get("powers") or enemy.get("status") or []
                if not isinstance(enemy_powers, list):
                    continue
                for slot_index, power in enumerate(enemy_powers):
                    if emitted >= budget:
                        break
                    if not isinstance(power, dict):
                        continue
                    token = self._build_power_slot_numeric(power)
                    if token is None:
                        continue
                    world_entries.append(
                        self._entry(
                            "POWER_SLOT_ENEMY",
                            token["numeric"],
                            owner_id=owner_id,
                            entity_id=token["bucket"],
                            order_id=min(slot_index, MAX_ORDER_ID),
                            text=token["label"],
                        )
                    )
                    emitted += 1

    def _runtime_card_modifier_tags(self, card: dict[str, Any]) -> list[str]:
        tags: list[str] = []
        for field_name in ("afflictions", "enchantments", "modifiers", "card_modifiers"):
            modifiers = card.get(field_name)
            if not isinstance(modifiers, list):
                continue
            for modifier in modifiers[:8]:
                if isinstance(modifier, dict):
                    semantic_tags = modifier.get("semantic_tags")
                    if isinstance(semantic_tags, list):
                        for tag in semantic_tags:
                            tag_text = str(tag or "").strip().lower()
                            if tag_text in {"adds_retain", "retain"}:
                                tags.append("retain")
                            elif tag_text in {"adds_ethereal", "ethereal"}:
                                tags.append("ethereal")
                            elif tag_text in {"adds_exhaust", "exhaust", "exhaust_self"}:
                                tags.append("exhaust_self")
                            elif tag_text in {"removes_exhaust"}:
                                tags.append("purge")
                            elif tag_text in {"cost_randomizes_on_draw", "cost_reduction_until_played", "sets_cost_zero", "energy_loss_on_play"}:
                                tags.append("cost_lock")
                            elif tag_text in {"autoplay_round_1"}:
                                tags.append("forced_play")
                    semantic_values = modifier.get("semantic_values")
                    if isinstance(semantic_values, dict):
                        if semantic_values.get("adds_retain"):
                            tags.append("retain")
                        if semantic_values.get("adds_ethereal"):
                            tags.append("ethereal")
                        if semantic_values.get("adds_exhaust"):
                            tags.append("exhaust_self")
                        if semantic_values.get("removes_exhaust"):
                            tags.append("purge")
                        if semantic_values.get("autoplay_round_1"):
                            tags.append("forced_play")
                        if any(semantic_values.get(key) for key in ("energy_loss_on_play", "cost_randomizes_on_draw", "cost_reduction_until_played", "sets_cost_zero")):
                            tags.append("cost_lock")
                    joined = " ".join(
                        str(modifier.get(key) or "")
                        for key in ("id", "title", "type", "description", "kind")
                    ).lower()
                else:
                    joined = str(modifier or "").lower()
                if not joined.strip():
                    continue
                if any(token in joined for token in ("bind", "bound", "chain", "shackle")):
                    tags.append("bound")
                if any(token in joined for token in ("lock", "forbid", "disabled", "unplayable", "can't play", "cannot play")):
                    tags.append("card_lock")
                if any(token in joined for token in ("cost", "energy")) and any(token in joined for token in ("lock", "increase", "reduce", "set")):
                    tags.append("cost_lock")
                if any(token in joined for token in ("forced", "must play", "required")):
                    tags.append("forced_play")
                if any(token in joined for token in ("temporary", "this turn", "until", "expire")):
                    tags.append("temporary")
                if "retain" in joined:
                    tags.append("retain")
                if "ethereal" in joined:
                    tags.append("ethereal")
                if "exhaust" in joined:
                    tags.append("exhaust_self")
                if "purge" in joined or "remove" in joined:
                    tags.append("purge")
        deduped: list[str] = []
        for tag in tags:
            if tag in _CARD_KEYWORD_BUCKETS and tag not in deduped:
                deduped.append(tag)
        return deduped

    def _append_card_keyword_slot_tokens(
        self,
        world_entries: list[dict[str, Any]],
        obs: dict[str, Any],
    ) -> None:
        """Emit one CARD_KEYWORD_SLOT token per (hand card 脳 recognized
        keyword) pair.

        Looks up each hand card's semantic_tags in content_registry and
        emits a token per matched keyword. owner_id = OWNER_HAND + card
        position so attention can route keyword 鈫?source card binding.
        entity_id = keyword bucket id (shared categorical space with the
        POWER_ID vocabulary via distinct low-end numbering).
        """
        combat = obs.get("combat") or {}
        hand = combat.get("hand") or []
        if not isinstance(hand, list) or not hand:
            return

        try:
            from content_registry import get_card_metadata  # noqa: PLC0415
        except Exception:
            return

        emitted = 0
        budget = MAX_CARD_KEYWORD_SLOTS

        for hand_index, card in enumerate(hand):
            if emitted >= budget:
                break
            if not isinstance(card, dict):
                continue
            card_id = str(card.get("id") or "")
            md = get_card_metadata(card_id) if card_id else None
            tags = md.get("semantic_tags") if isinstance(md, dict) else []
            if not isinstance(tags, list):
                tags = []

            # Also fold explicit card.keywords (from sim translator/live bridge)
            # and runtime per-card modifier tags from boss/event effects.
            card_keywords = card.get("keywords") or []
            modifier_tags = self._runtime_card_modifier_tags(card)
            all_tags: list[str] = []
            for t in list(tags) + list(card_keywords if isinstance(card_keywords, list) else []) + modifier_tags:
                t_norm = str(t).strip().lower()
                if t_norm in _CARD_KEYWORD_BUCKETS and t_norm not in all_tags:
                    all_tags.append(t_norm)
            if not all_tags:
                continue

            # Bind this keyword slot to the hand card via owner_id so the
            # attention owner_pair_bias can learn "keyword-for-this-card"
            # as a same-owner relation.
            owner_id = min(OWNER_HAND + hand_index, MAX_OWNER_ID)
            for slot_index, kw in enumerate(all_tags):
                if emitted >= budget:
                    break
                bucket = _CARD_KEYWORD_BUCKETS[kw]
                numeric = np.zeros(TOKEN_NUMERIC_DIM, dtype=np.float32)
                # Minimal feature set 鈥?keywords are mostly categorical.
                numeric[0] = 1.0                               # active flag
                numeric[1] = hand_index / 10.0                 # card position hint
                numeric[2] = float(card.get("is_playable", True))
                numeric[3] = obs_common._float(card.get("cost")) / 3.0
                # Flag which keyword family (for fast linear probing by
                # other downstream heads without embedding lookup).
                if kw in ("ethereal", "exhaust_self", "purge"):
                    numeric[4] = 1.0  # auto-removal-on-use/eot
                if kw in ("retain",):
                    numeric[5] = 1.0  # persists across turns
                if kw in ("innate",):
                    numeric[6] = 1.0  # opening-hand guarantee
                if kw in ("unplayable", "bound", "card_lock", "forced_play"):
                    numeric[7] = 1.0  # curse/blank/restricted
                if kw in ("bound", "card_lock"):
                    numeric[8] = 1.0  # boss/event per-card lock
                if kw in ("cost_lock", "x_cost"):
                    numeric[9] = 1.0  # energy/cost interaction
                if kw in ("temporary", "forced_play"):
                    numeric[10] = 1.0  # timing/obligation
                world_entries.append(
                    self._entry(
                        "CARD_KEYWORD_SLOT",
                        numeric,
                        owner_id=owner_id,
                        entity_id=bucket,
                        order_id=min(slot_index, MAX_ORDER_ID),
                        text=kw,
                    )
                )
                emitted += 1

    def _build_power_slot_numeric(self, power: dict[str, Any]) -> dict[str, Any] | None:
        """Pack a single power dict into the numeric + categorical fields
        a POWER_SLOT token needs. Returns None on malformed input.

        Numeric layout (TOKEN_NUMERIC_DIM=96 slots available; we use 16):
          [0..12]  effect algebra (damage_mult_given, damage_flat_given,
                                   damage_mult_received, damage_flat_received,
                                   block_mult_given, block_flat_given,
                                   block_persistent, end_of_turn_dmg_self,
                                   end_of_turn_dmg_given, stacks_on_applied,
                                   decays_each_turn, is_buff, is_debuff)
          [13]     amount clipped + log-normalized
          [14]     amount sign (positive/negative for reversible powers)
          [15]     amount ratio vs typical cap (amount/10, clipped to 1)
          [16..95] unused 鈥?reserved for Phase 6.x extensions (duration,
                            applier/target hints, etc.)
        """
        pid = str(power.get("id") or "").strip()
        if not pid:
            return None
        amount = power.get("amount")
        try:
            amount_f = float(amount) if amount is not None else 0.0
        except (TypeError, ValueError):
            amount_f = 0.0

        numeric = np.zeros(TOKEN_NUMERIC_DIM, dtype=np.float32)
        algebra = _power_algebra(pid)
        numeric[: _POWER_ALGEBRA_DIM] = algebra
        # Amount features: log-scaled magnitude, sign, clipped ratio.
        numeric[13] = obs_common._log_norm(abs(amount_f), obs_common._LOG1P_200)
        numeric[14] = 1.0 if amount_f > 0 else (-1.0 if amount_f < 0 else 0.0)
        numeric[15] = max(-1.0, min(amount_f / 10.0, 1.0))
        return {
            "numeric": numeric,
            "bucket": _power_id_bucket(pid),
            "label": pid,
        }

    def _append_history_tokens(
        self,
        world_entries: list[dict[str, Any]],
        obs: dict[str, Any],
    ) -> None:
        """Emit MAX_STEP_DETAIL_TOKENS step-detail + MAX_TURN_SUMMARY_TOKENS
        turn-summary tokens from ``obs["_action_history"]``.

        Always emits a fixed number of tokens (padded with is_empty=1 when
        the tracker is shorter) so MAX_WORLD_TOKENS stays invariant across
        calls 鈥?rollout buffers and type_id arrays are fixed-shape.

        Numeric layout is split between step-detail and turn-summary so a
        single shared HISTORY role / zone / owner still produces distinct
        feature distributions the model can separate via the token_type
        one-hot (type id carried on world_token_type_ids).
        """
        history_dict = obs.get("_action_history") if isinstance(obs, dict) else None
        if not isinstance(history_dict, dict):
            history_dict = {"step_detail": [], "turn_summary": []}
        step_detail_entries = history_dict.get("step_detail") if isinstance(history_dict.get("step_detail"), list) else []
        turn_summary_entries = history_dict.get("turn_summary") if isinstance(history_dict.get("turn_summary"), list) else []

        # --- Step-detail tokens ---
        for slot in range(MAX_STEP_DETAIL_TOKENS):
            entry = step_detail_entries[slot] if slot < len(step_detail_entries) else None
            numeric, card_bucket, text = self._build_history_step_numeric(entry, slot)
            world_entries.append(
                self._entry(
                    "HISTORY_STEP_DETAIL",
                    numeric,
                    owner_id=OWNER_HISTORY,
                    entity_id=int(card_bucket),
                    order_id=min(slot, MAX_ORDER_ID),
                    zone_id=TOKEN_ZONE_TO_ID.get("HISTORY", 0),
                    text=text,
                )
            )

        # --- Turn-summary tokens ---
        for slot in range(MAX_TURN_SUMMARY_TOKENS):
            entry = turn_summary_entries[slot] if slot < len(turn_summary_entries) else None
            numeric, text = self._build_history_turn_summary_numeric(entry, slot)
            world_entries.append(
                self._entry(
                    "HISTORY_TURN_SUMMARY",
                    numeric,
                    owner_id=OWNER_HISTORY,
                    entity_id=0,
                    order_id=min(slot, MAX_ORDER_ID),
                    zone_id=TOKEN_ZONE_TO_ID.get("HISTORY", 0),
                    text=text,
                )
            )

    def _build_history_step_numeric(
        self, entry: dict[str, Any] | None, slot: int
    ) -> tuple[np.ndarray, int, str]:
        """Pack a single step-detail numeric block.

        Layout (TOKEN_NUMERIC_DIM = 96):
          [0]       is_empty             (1 = padding, tracker had no entry)
          [1]       is_step_detail       (always 1 here; turn-summary sets [1]=0)
          [2..]     family one-hot       (len = NUM_FAMILIES)
          [next]    semantic_role flags  (NUM_SEMANTIC_ROLES bits)
          [next]    target_scope one-hot (len = NUM_TARGET_SCOPES)
          [next]    step_offset one-hot  (MAX_STEP_DETAIL_TOKENS slots)
          [next]    flags (7):            same_turn / same_encounter /
                    same_floor / phase_changed / combat_ended / rejected /
                    reward_nonzero
          [next]    reward scalars (2):   reward clipped [-1,1], abs(reward)
          [next]    pre_state_vec  (STATE_SNAPSHOT_DIM = 8) 鈥?Tier 2
          [next]    post_state_vec (STATE_SNAPSHOT_DIM = 8) 鈥?Tier 2
          [next]    step_offset_norm (1)
          remainder zero (~11 spare slots 鈥?room for future additions)

        The pre/post snapshot pair lets any downstream linear probe
        compute "delta = post - pre" on every dimension. The separate
        causality_delta tensor in the tracker is NOT packed here: it's
        delivered to the aux head's loss target path directly (via
        env_v2 info 鈫?aux_maskable_ppo buffer), NOT through the obs
        tensor. Keeping targets off the observation tensor avoids
        teaching the policy to trivially shortcut 鈥?the policy sees
        pre/post context but has to actively predict the delta.
        """
        from .semantic_action import SEMANTIC_ACTION_FAMILIES, SEMANTIC_TARGET_SCOPES
        from .action_history import STATE_SNAPSHOT_DIM

        numeric = np.zeros(TOKEN_NUMERIC_DIM, dtype=np.float32)
        numeric[1] = 1.0

        if not isinstance(entry, dict):
            numeric[0] = 1.0
            return numeric, 0, ""

        cursor = 2
        family_idx = int(entry.get("family_idx") or 0)
        if 0 <= family_idx < len(SEMANTIC_ACTION_FAMILIES):
            numeric[cursor + family_idx] = 1.0
        cursor += len(SEMANTIC_ACTION_FAMILIES)
        role_flags = int(entry.get("semantic_role_flags") or 0)
        for bit in range(NUM_SEMANTIC_ROLES):
            if role_flags & (1 << bit):
                numeric[cursor + bit] = 1.0
        cursor += NUM_SEMANTIC_ROLES
        scope_idx = int(entry.get("target_scope_idx") or 0)
        if 0 <= scope_idx < len(SEMANTIC_TARGET_SCOPES):
            numeric[cursor + scope_idx] = 1.0
        cursor += len(SEMANTIC_TARGET_SCOPES)
        step_offset = int(entry.get("step_offset") or 0)
        step_offset = max(0, min(step_offset, MAX_STEP_DETAIL_TOKENS - 1))
        numeric[cursor + step_offset] = 1.0
        cursor += MAX_STEP_DETAIL_TOKENS

        numeric[cursor + 0] = 1.0 if entry.get("same_turn") else 0.0
        numeric[cursor + 1] = 1.0 if entry.get("same_encounter") else 0.0
        numeric[cursor + 2] = 1.0 if entry.get("same_floor") else 0.0
        numeric[cursor + 3] = 1.0 if entry.get("phase_changed") else 0.0
        numeric[cursor + 4] = 1.0 if entry.get("combat_ended") else 0.0
        numeric[cursor + 5] = 1.0 if entry.get("rejected") else 0.0
        numeric[cursor + 6] = 1.0 if entry.get("reward_nonzero") else 0.0
        cursor += 7

        reward = float(entry.get("reward") or 0.0)
        numeric[cursor + 0] = max(-1.0, min(reward, 1.0))
        numeric[cursor + 1] = min(abs(reward), 1.0)
        cursor += 2

        # Tier 2: pre_state_vec
        pre_vec = entry.get("pre_state_vec") or []
        for i in range(STATE_SNAPSHOT_DIM):
            numeric[cursor + i] = float(pre_vec[i]) if i < len(pre_vec) else 0.0
        cursor += STATE_SNAPSHOT_DIM
        # Tier 2: post_state_vec
        post_vec = entry.get("post_state_vec") or []
        for i in range(STATE_SNAPSHOT_DIM):
            numeric[cursor + i] = float(post_vec[i]) if i < len(post_vec) else 0.0
        cursor += STATE_SNAPSHOT_DIM

        numeric[cursor] = step_offset / max(MAX_STEP_DETAIL_TOKENS - 1, 1)
        cursor += 1

        card_bucket = int(entry.get("card_id_bucket") or 0)
        text = str(entry.get("canonical_text") or "")
        return numeric, card_bucket, text

    def _build_history_turn_summary_numeric(
        self, entry: dict[str, Any] | None, slot: int
    ) -> tuple[np.ndarray, str]:
        """Pack a single turn-summary numeric block.

        Layout (TOKEN_NUMERIC_DIM = 96):
          [0]       is_empty                (1 = padding)
          [1]       is_step_detail          (0 鈥?this is a turn summary)
          [2..10]   turn_offset one-hot     (MAX_TURN_SUMMARY_TOKENS=8 slots, offset 1..8)
          [10..14]  action counts           (n_attacks/n_skills/n_powers/n_potions normalized)
          [14..17]  totals                  (damage/block/hp_lost normalized)
          [17..22]  end-of-turn stats       (strength/dex/focus/block/enemy_hp_ratio)
          [22]      end_enemy_vuln_total normalized
          [23]      turn_num normalized
          [24..40]  key_power_card_flags    (16 bits, one per slot)
          [40..44]  outcome flags           (enemy_killed / player_took_dmg /
                                             player_scaled / low_energy_waste)
          remainder zero 鈥?Tier 2 can add cross-turn comparison scalars here.
        """
        numeric = np.zeros(TOKEN_NUMERIC_DIM, dtype=np.float32)
        # [1] = 0: this is NOT step_detail. Stays zero for turn summaries.

        if not isinstance(entry, dict):
            numeric[0] = 1.0
            return numeric, ""

        cursor = 2
        # turn_offset one-hot
        turn_offset = int(entry.get("turn_offset") or 1)
        turn_offset_slot = max(0, min(turn_offset - 1, MAX_TURN_SUMMARY_TOKENS - 1))
        numeric[cursor + turn_offset_slot] = 1.0
        cursor += MAX_TURN_SUMMARY_TOKENS

        # Action counts (normalized by a generous ceiling 鈥?8 cards/turn is a lot)
        numeric[cursor + 0] = min(float(entry.get("n_attacks") or 0) / 8.0, 1.0)
        numeric[cursor + 1] = min(float(entry.get("n_skills") or 0) / 8.0, 1.0)
        numeric[cursor + 2] = min(float(entry.get("n_powers") or 0) / 4.0, 1.0)
        numeric[cursor + 3] = min(float(entry.get("n_potions_used") or 0) / 3.0, 1.0)
        cursor += 4

        # Totals
        numeric[cursor + 0] = min(float(entry.get("total_damage_dealt") or 0) / 50.0, 1.0)
        numeric[cursor + 1] = min(float(entry.get("total_block_gained") or 0) / 30.0, 1.0)
        numeric[cursor + 2] = min(float(entry.get("total_hp_lost") or 0) / 30.0, 1.0)
        cursor += 3

        # End-of-turn stats
        numeric[cursor + 0] = min(float(entry.get("end_strength") or 0) / 10.0, 1.0)
        numeric[cursor + 1] = min(float(entry.get("end_dex") or 0) / 10.0, 1.0)
        numeric[cursor + 2] = min(float(entry.get("end_focus") or 0) / 10.0, 1.0)
        numeric[cursor + 3] = min(float(entry.get("end_player_block") or 0) / 40.0, 1.0)
        numeric[cursor + 4] = max(0.0, min(float(entry.get("end_enemy_total_hp_ratio") or 0), 1.0))
        cursor += 5

        numeric[cursor + 0] = min(float(entry.get("end_enemy_vuln_total") or 0) / 10.0, 1.0)
        cursor += 1
        numeric[cursor + 0] = min(float(entry.get("turn_num") or 0) / 30.0, 1.0)
        cursor += 1

        # Key-power-card flags (16 bits, one per slot so attention doesn't
        # have to learn a bitmap decoder)
        key_flags = int(entry.get("key_power_card_flags") or 0)
        for bit in range(NUM_KEY_POWER_FLAGS):
            if key_flags & (1 << bit):
                numeric[cursor + bit] = 1.0
        cursor += NUM_KEY_POWER_FLAGS

        # Outcome flags
        numeric[cursor + 0] = 1.0 if entry.get("enemy_killed") else 0.0
        numeric[cursor + 1] = 1.0 if entry.get("player_took_dmg") else 0.0
        numeric[cursor + 2] = 1.0 if entry.get("player_scaled") else 0.0
        numeric[cursor + 3] = 1.0 if entry.get("low_energy_waste") else 0.0
        cursor += 4

        text = str(entry.get("canonical_text") or "")
        return numeric, text

    def _append_relic_collection(
        self,
        world_entries: list[dict[str, Any]],
        relic_entries: list[Any],
        relic_text: np.ndarray,
        relic_mask: np.ndarray,
    ) -> None:
        total = len(relic_entries) if isinstance(relic_entries, list) else 0
        for index in range(min(relic_text.shape[0], total)):
            if relic_mask[index] <= 0:
                continue
            relic = relic_entries[index]
            numeric = self._relic_numeric(relic, index=index, total=max(total, 1))
            entity_key = self._support_entity_key(relic, fallback=f"relic:{index}")
            world_entries.append(
                self._entry(
                    "RELIC",
                    numeric,
                    owner_id=OWNER_RELIC,
                    entity_id=_stable_bucket(entity_key),
                    order_id=index + 1,
                    text_embedding=relic_text[index],
                )
            )

    def _append_potion_collection(
        self,
        world_entries: list[dict[str, Any]],
        potion_entries: list[Any],
        potion_text: np.ndarray,
        potion_mask: np.ndarray,
    ) -> None:
        total = len(potion_entries) if isinstance(potion_entries, list) else 0
        for index in range(min(potion_text.shape[0], total)):
            if potion_mask[index] <= 0:
                continue
            potion = potion_entries[index]
            title = potion if isinstance(potion, str) else (potion.get("title") if isinstance(potion, dict) else "")
            if str(title or "").strip() == "[empty]":
                continue
            numeric = self._potion_numeric(potion, source_profile=None, slot_index=index, total_slots=max(total, 1))
            entity_key = self._support_entity_key(potion, fallback=f"potion:{index}")
            world_entries.append(
                self._entry(
                    "POTION",
                    numeric,
                    owner_id=OWNER_POTION,
                    entity_id=_stable_bucket(entity_key),
                    order_id=index + 1,
                    text_embedding=potion_text[index],
                )
            )

    def _append_relic_trigger_locals(
        self,
        entries: list[dict[str, Any]],
        obs: dict[str, Any],
        action: dict[str, Any],
    ) -> None:
        player = obs.get("player") or {}
        relics = player.get("relics") or []
        if not isinstance(relics, list) or not relics:
            return

        scored_entries: list[tuple[float, dict[str, Any]]] = []
        for index, relic in enumerate(relics[: obs_common.MAX_RELICS]):
            if not isinstance(relic, (dict, str)):
                continue
            numeric = self._relic_numeric(relic, action=action, index=index, total=max(len(relics), 1))
            entry = self._entry(
                "RELIC_TRIGGER_LOCAL",
                numeric,
                owner_id=OWNER_RELIC,
                entity_id=_stable_bucket(self._support_entity_key(relic, fallback=f"relic:{index}")),
                order_id=index + 1,
                text=self._relic_text(relic),
            )
            scored_entries.append((float(numeric[0]), entry))

        scored_entries.sort(key=lambda item: item[0], reverse=True)
        for _, entry in scored_entries[:3]:
            entries.append(entry)

    def _append_potion_option_locals(
        self,
        entries: list[dict[str, Any]],
        obs: dict[str, Any],
        action: dict[str, Any],
        features: dict[str, np.ndarray],
    ) -> None:
        player = obs.get("player") or {}
        potions = player.get("potions") or []
        if not isinstance(potions, list) or not potions:
            return

        source_profile = self._source_profile(action.get("card") if isinstance(action.get("card"), dict) else None)
        scored_entries: list[tuple[float, dict[str, Any]]] = []
        for index, potion in enumerate(potions[: obs_common.MAX_POTIONS]):
            title = potion if isinstance(potion, str) else (potion.get("title") if isinstance(potion, dict) else "")
            if str(title or "").strip() == "[empty]":
                continue
            numeric = self._potion_numeric(potion, source_profile=source_profile, slot_index=index, total_slots=max(len(potions), 1))
            text_embedding = features["potions"][index] if index < features["potions"].shape[0] and features["potion_mask"][index] > 0 else None
            entry = self._entry(
                "POTION_OPTION_LOCAL",
                numeric,
                owner_id=OWNER_POTION,
                entity_id=_stable_bucket(self._support_entity_key(potion, fallback=f"potion:{index}")),
                order_id=index + 1,
                text=self._potion_text(potion),
                text_embedding=text_embedding,
            )
            scored_entries.append((float(numeric[0]), entry))

        scored_entries.sort(key=lambda item: item[0], reverse=True)
        for _, entry in scored_entries[:2]:
            entries.append(entry)

    def _relic_numeric(
        self,
        relic: dict[str, Any] | str,
        *,
        action: dict[str, Any] | None = None,
        index: int = 0,
        total: int = 1,
    ) -> np.ndarray:
        numeric = np.zeros(TOKEN_NUMERIC_DIM, dtype=np.float32)
        text = self._relic_text(relic).lower()
        signal_vector = np.zeros(obs_common.RELIC_SIGNAL_DIM, dtype=np.float32)
        obs_common._encode_relic_signals(signal_vector, [relic])
        profile = self._source_profile(action.get("card") if isinstance(action, dict) and isinstance(action.get("card"), dict) else action.get("potion") if isinstance(action, dict) and isinstance(action.get("potion"), dict) else None)

        relevance = float(np.clip(signal_vector.sum() / 4.0, 0.0, 1.0))
        relevance += 0.20 * signal_vector[0] * float(profile["x_cost"] > 0.5 or profile["cost"] >= 2.0)
        relevance += 0.15 * signal_vector[1] * float(profile["draw"] > 0.0 or profile["zero_cost"] > 0.5)
        relevance += 0.20 * max(signal_vector[2], signal_vector[4]) * profile["attack"]
        relevance += 0.20 * max(signal_vector[3], signal_vector[10]) * max(profile["skill"], float(profile["block"] > 0.0))
        relevance += 0.10 * float("attack" in text or "hit" in text) * profile["attack"]
        relevance += 0.10 * float("skill" in text or "block" in text) * max(profile["skill"], float(profile["block"] > 0.0))
        relevance += 0.10 * float("power" in text) * profile["power"]
        relevance += 0.10 * float("exhaust" in text or "ethereal" in text or "burn" in text) * max(profile["exhaust"], profile["ethereal"])
        relevance += 0.10 * float("potion" in text) * float(isinstance(action, dict) and str(action.get("kind") or "") in {"use_potion", "discard_potion"})
        relevance = float(np.clip(relevance, 0.0, 1.0))

        numeric[0] = relevance
        numeric[1 : 1 + obs_common.RELIC_SIGNAL_DIM] = signal_vector
        base = 1 + obs_common.RELIC_SIGNAL_DIM
        numeric[base] = min((index + 1) / max(total, 1), 1.0)
        numeric[base + 1] = float("attack" in text or "hit" in text)
        numeric[base + 2] = float("skill" in text or "block" in text)
        numeric[base + 3] = float("power" in text)
        numeric[base + 4] = float("exhaust" in text or "ethereal" in text or "burn" in text)
        numeric[base + 5] = float("potion" in text)
        return numeric

    def _potion_numeric(
        self,
        potion: dict[str, Any] | str,
        *,
        source_profile: dict[str, float] | None,
        slot_index: int,
        total_slots: int,
    ) -> np.ndarray:
        numeric = np.zeros(TOKEN_NUMERIC_DIM, dtype=np.float32)
        potion_profile = self._source_profile(potion if isinstance(potion, dict) else None)
        text = self._potion_text(potion).lower()
        merged_entry, effect = _resolve_potion_effect(potion)
        effect_family = list(merged_entry.get("effect_family") or [])
        timing_tags = list(merged_entry.get("timing_tags") or [])
        training_tags = list(merged_entry.get("training_tags") or [])

        damage_v = float(effect.get("damage") or potion_profile["damage"] or 0.0)
        block_v = float(effect.get("block") or potion_profile["block"] or 0.0)
        draw_v = float(effect.get("draw") or potion_profile["draw"] or 0.0)
        energy_v = float(effect.get("energy_gain") or potion_profile["energy"] or 0.0)
        heal_v = float(effect.get("heal") or potion_profile["heal"] or 0.0)
        weak_v = float(effect.get("weak") or potion_profile["weak"] or 0.0)
        vuln_v = float(effect.get("vulnerable") or potion_profile["vulnerable"] or 0.0)
        poison_v = float(effect.get("poison") or 0.0)
        str_v = float(effect.get("strength") or 0.0)
        dex_v = float(effect.get("dexterity") or 0.0)
        intang_v = float(effect.get("intangible") or 0.0)
        prevent_v = float(effect.get("prevent_damage") or 0.0)
        gen_card_v = float(effect.get("generate_card_count") or 0.0)
        discover_v = float(effect.get("discover_count") or 0.0)
        upgrade_v = float(effect.get("upgrade_hand") or 0.0)
        dup_next_v = float(effect.get("duplicate_next") or 0.0)
        retrieve_v = float(effect.get("retrieve_from_discard") or 0.0)
        replace_v = float(effect.get("replace_or_transform_hand") or 0.0)
        is_aoe = bool(effect.get("aoe"))
        is_single = bool(effect.get("single_target")) or potion_profile["single_target"] > 0.5
        is_random = bool(effect.get("random_target"))
        target_required = bool(effect.get("target_required"))
        can_change_facing = bool(effect.get("can_change_facing_if_targeted_enemy"))
        requires_followup = bool(effect.get("requires_followup"))
        long_term_value = bool(effect.get("long_term_value"))
        passive_or_triggered = bool(effect.get("passive_or_triggered"))
        enabled_training = bool(merged_entry.get("enabled_for_training", True))

        relevance = 0.0
        if source_profile is not None:
            relevance += 0.20 * float(damage_v > 0.0) * max(source_profile["attack"], float(source_profile["damage"] > 0.0))
            relevance += 0.20 * float(block_v > 0.0) * max(source_profile["skill"], float(source_profile["block"] > 0.0))
            relevance += 0.20 * float(energy_v > 0.0) * float(source_profile["x_cost"] > 0.5 or source_profile["cost"] >= 2.0)
            relevance += 0.15 * float(draw_v > 0.0) * max(source_profile["zero_cost"], float(source_profile["draw"] > 0.0))
            relevance += 0.10 * float(heal_v > 0.0 or potion_profile["hp_loss"] < 0.0) * float(source_profile["hp_loss"] > 0.0)
        relevance += 0.10 * float("attack" in text) * float(damage_v > 0.0)
        relevance = float(np.clip(relevance, 0.0, 1.0))

        numeric[0] = relevance
        numeric[1] = obs_common._log_norm(damage_v, obs_common._LOG1P_200)
        numeric[2] = obs_common._log_norm(block_v, obs_common._LOG1P_200)
        numeric[3] = min(draw_v / 5.0, 1.0)
        numeric[4] = min(energy_v / 5.0, 1.0)
        numeric[5] = obs_common._log_norm(heal_v, obs_common._LOG1P_200)
        numeric[6] = min(max(weak_v, vuln_v) / 5.0, 1.0)
        numeric[7] = float(is_single)
        numeric[8] = float(is_aoe)
        numeric[9] = min((slot_index + 1) / max(total_slots, 1), 1.0)
        numeric[10] = float("discard" not in text)

        # Phase 3 (potion timing v1): structured slot expansion. Slots 11-38
        # consume the bridge effect_profile / Python registry merge so the
        # potion world token and use_potion action token observe identical
        # capability signals. See docs/potion-timing-modeling-plan.md §7.3.
        numeric[11] = min(poison_v / 10.0, 1.0)
        numeric[12] = min(weak_v / 5.0, 1.0)
        numeric[13] = min(vuln_v / 5.0, 1.0)
        numeric[14] = min(str_v / 5.0, 1.0)
        numeric[15] = min(dex_v / 5.0, 1.0)
        numeric[16] = min(max(intang_v, prevent_v / 10.0), 1.0)
        numeric[17] = min(gen_card_v / 5.0, 1.0)
        numeric[18] = min(discover_v / 5.0, 1.0)
        numeric[19] = min(upgrade_v, 1.0)
        numeric[20] = min(dup_next_v, 1.0)
        numeric[21] = min(retrieve_v, 1.0)
        numeric[22] = min(replace_v, 1.0)
        numeric[23] = float(is_random)
        numeric[24] = float(requires_followup)
        numeric[25] = float("save_if_low_threat" in training_tags)
        numeric[26] = float(long_term_value or "long_term_value" in timing_tags)
        numeric[27] = float(passive_or_triggered)
        numeric[28] = float(can_change_facing)
        numeric[29] = float(target_required)
        numeric[30] = min((slot_index + 1) / max(total_slots, 1), 1.0)
        numeric[31] = float(not enabled_training)
        numeric[32] = float(any(t in timing_tags for t in ("setup_tool", "scaling_setup")) or "setup" in effect_family)
        numeric[33] = float(any(t in timing_tags for t in ("scaling_setup",)) or any(f in effect_family for f in ("strength", "dexterity", "focus", "scaling")))
        numeric[34] = float("mechanism_answer_candidate" in timing_tags or can_change_facing)
        numeric[35] = float("hand_context_dependency" in timing_tags or upgrade_v > 0.0 or dup_next_v > 0.0 or replace_v > 0.0)
        numeric[36] = float(retrieve_v > 0.0 or "discard_context_dependency" in timing_tags)
        numeric[37] = float(any(t in timing_tags for t in ("dig_for_answer", "draw_pile_context_dependency")) or draw_v > 0.0)
        numeric[38] = float("exhaust_pile_context_dependency" in timing_tags)
        return numeric

    def _source_profile(self, source: dict[str, Any] | None) -> dict[str, float]:
        profile = {
            "cost": 0.0,
            "x_cost": 0.0,
            "attack": 0.0,
            "skill": 0.0,
            "power": 0.0,
            "strength": 0.0,
            "dexterity": 0.0,
            "zero_cost": 0.0,
            "damage": 0.0,
            "block": 0.0,
            "draw": 0.0,
            "energy": 0.0,
            "heal": 0.0,
            "hp_loss": 0.0,
            "weak": 0.0,
            "vulnerable": 0.0,
            "hits": 0.0,
            "single_target": 0.0,
            "aoe_target": 0.0,
            "exhaust": 0.0,
            "ethereal": 0.0,
            "retain": 0.0,
            "energy_loss": 0.0,
            "self_damage": 0.0,
            "play_count_bonus": 0.0,
            "damage_add": 0.0,
            "damage_mult": 1.0,
            "block_add": 0.0,
            "adds_exhaust": 0.0,
            "removes_exhaust": 0.0,
            "adds_retain": 0.0,
            "adds_ethereal": 0.0,
            "once_per_combat": 0.0,
            "disabled_after_play": 0.0,
            "cost_randomizes_on_draw": 0.0,
            "sets_cost_zero": 0.0,
            "shuffle_top": 0.0,
            "typed_ops_count": 0.0,
            "typed_modifies_hand": 0.0,
            "typed_upgrade_hand": 0.0,
            "typed_exhaust_cards": 0.0,
            "typed_discard_cards": 0.0,
            "typed_transform_cards": 0.0,
            "typed_copy_cards": 0.0,
            "typed_modify_cost": 0.0,
            "typed_set_replay": 0.0,
            "typed_retain_cards": 0.0,
            "typed_add_modifier": 0.0,
            "typed_add_keyword": 0.0,
            "typed_add_generated_card": 0.0,
            "typed_draw_cards": 0.0,
            "typed_draw_amount": 0.0,
            "typed_gain_energy": 0.0,
            "typed_gain_energy_amount": 0.0,
            "typed_hp_loss": 0.0,
            "typed_apply_power": 0.0,
            "typed_no_draw": 0.0,
            "typed_future_penalty": 0.0,
            "typed_requires_followup": 0.0,
            "typed_strategic_skip_if_no_followup": 0.0,
            "typed_not_x_cost_filter": 0.0,
            "typed_x_cost_filter": 0.0,
            "typed_hand_context_dependency": 0.0,
            "typed_discard_context_dependency": 0.0,
            "typed_exhaust_context_dependency": 0.0,
            "typed_draw_context_dependency": 0.0,
            "typed_deck_context_dependency": 0.0,
            "typed_consumes_future_resource": 0.0,
            "typed_once_or_exhaust_self": 0.0,
            "typed_selection_required": 0.0,
            "typed_all_scope": 0.0,
            "typed_card_rule_modifier": 0.0,
            "typed_card_state_mutation": 0.0,
            "typed_hand_context_needed": 0.0,
            "strategic_skip_value": 0.0,
        }
        if not isinstance(source, dict):
            return profile

        preview = obs_common._build_card_preview_bundle(source)
        strength, dexterity, energy, hits = obs_common._get_card_extra_metrics(source)
        kw_flags, _ = obs_common._get_card_keywords(source)
        modifier_sem = obs_common._aggregate_card_modifier_semantics(source)
        effect_sem = obs_common._aggregate_card_effect_profile_semantics(source)
        typed_energy = effect_sem.get("typed_gain_energy_amount", 0.0)
        typed_draw = effect_sem.get("typed_draw_amount", 0.0)
        typed_hp_loss = effect_sem.get("typed_hp_loss", 0.0)
        card_type = str(source.get("type") or "").capitalize()
        target = str(source.get("target_type") or source.get("target") or "").lower()
        cost = obs_common._float(source.get("cost"))

        profile.update(
            {
                "cost": max(cost, 0.0),
                "x_cost": float(
                    bool(source.get("x_cost") or source.get("costs_x"))
                    or str(source.get("cost") or source.get("canonical_energy_cost") or "").strip().upper() == "X"
                ),
                "attack": 1.0 if card_type == "Attack" else 0.0,
                "skill": 1.0 if card_type == "Skill" else 0.0,
                "power": 1.0 if card_type == "Power" else 0.0,
                "zero_cost": 1.0 if cost == 0 else 0.0,
                "damage": preview["preview_damage"],
                "block": preview["preview_block"],
                "draw": max(obs_common._preview_metric(source, "draw") + modifier_sem.get("draw", 0.0), typed_draw),
                "energy": max(energy, typed_energy),
                "heal": obs_common._preview_metric(source, "heal"),
                "hp_loss": max(obs_common._preview_metric(source, "hp_loss") + modifier_sem.get("self_damage", 0.0), typed_hp_loss),
                "energy_loss": modifier_sem.get("energy_loss_on_play", 0.0),
                "self_damage": modifier_sem.get("self_damage", 0.0),
                "weak": obs_common._preview_metric(source, "weak") + modifier_sem.get("weak", 0.0),
                "vulnerable": obs_common._preview_metric(source, "vulnerable"),
                "hits": hits,
                "single_target": 1.0 if "single" in target or "anyenemy" in target else 0.0,
                "aoe_target": 1.0 if "all" in target else 0.0,
                "exhaust": 1.0 if kw_flags[0] or effect_sem.get("typed_once_or_exhaust_self", 0.0) > 0.0 else 0.0,
                "ethereal": 1.0 if kw_flags[1] else 0.0,
                "retain": 1.0 if kw_flags[2] or effect_sem.get("typed_retain_cards", 0.0) > 0.0 else 0.0,
                "play_count_bonus": modifier_sem.get("play_count_bonus", 0.0),
                "damage_add": modifier_sem.get("damage_add", 0.0),
                "damage_mult": modifier_sem.get("damage_mult", 1.0),
                "block_add": modifier_sem.get("block_add", 0.0) + modifier_sem.get("block_on_play", 0.0),
                "adds_exhaust": modifier_sem.get("adds_exhaust", 0.0),
                "removes_exhaust": modifier_sem.get("removes_exhaust", 0.0),
                "adds_retain": modifier_sem.get("adds_retain", 0.0),
                "adds_ethereal": modifier_sem.get("adds_ethereal", 0.0),
                "once_per_combat": modifier_sem.get("once_per_combat", 0.0),
                "disabled_after_play": modifier_sem.get("disabled_after_play", 0.0),
                "cost_randomizes_on_draw": modifier_sem.get("cost_randomizes_on_draw", 0.0),
                "sets_cost_zero": modifier_sem.get("sets_cost_zero", 0.0),
                "shuffle_top": modifier_sem.get("shuffle_top", 0.0),
            }
        )
        profile.update(effect_sem)
        profile["strength"] = strength
        profile["dexterity"] = dexterity
        profile["draw"] = max(profile["draw"], typed_draw)
        profile["energy"] = max(profile["energy"], typed_energy)
        profile["hp_loss"] = max(profile["hp_loss"], typed_hp_loss)
        profile["exhaust"] = max(profile["exhaust"], effect_sem.get("typed_once_or_exhaust_self", 0.0))
        profile["retain"] = max(profile["retain"], effect_sem.get("typed_retain_cards", 0.0))
        profile["strategic_skip_value"] = float(
            profile["exhaust"] > 0.5
            or profile["retain"] > 0.5
            or profile["energy_loss"] > 0.0
            or profile["self_damage"] > 0.0
            or profile.get("typed_strategic_skip_if_no_followup", 0.0) > 0.5
            or profile.get("typed_requires_followup", 0.0) > 0.5
            or profile.get("typed_no_draw", 0.0) > 0.5
            or profile.get("typed_future_penalty", 0.0) > 0.5
            or profile.get("typed_consumes_future_resource", 0.0) > 0.5
            or profile.get("typed_exhaust_cards", 0.0) > 0.5
            or profile.get("typed_transform_cards", 0.0) > 0.5
            or profile.get("typed_modify_cost", 0.0) > 0.5
            or profile.get("typed_set_replay", 0.0) > 0.5
        )
        return profile

    def _source_text(self, source: dict[str, Any] | None) -> str:
        if not isinstance(source, dict):
            return ""
        parts: list[str] = []
        for key in ("title", "name", "description", "text", "canonical_text"):
            value = str(source.get(key) or "").strip()
            if value:
                parts.append(value)
        keywords = source.get("keywords")
        if isinstance(keywords, list):
            for keyword in keywords:
                value = str(keyword or "").strip()
                if value:
                    parts.append(value)
        return " | ".join(parts).lower()

    @staticmethod
    def _support_entity_key(entry: Any, *, fallback: str) -> str:
        if isinstance(entry, dict):
            for key in ("id", "model_id", "title", "name", "canonical_text"):
                value = str(entry.get(key) or "").strip()
                if value:
                    return value
        elif isinstance(entry, str):
            value = entry.strip()
            if value:
                return value
        return fallback

    def _relic_text(self, relic: Any) -> str:
        if isinstance(relic, dict):
            semantic = build_live_relic_semantic_text(relic)
            if semantic:
                return semantic
        return " | ".join(obs_common._get_relic_text_candidates(relic)) or str(relic or "")

    def _potion_text(self, potion: Any) -> str:
        if isinstance(potion, dict):
            semantic = build_live_potion_semantic_text(potion)
            if semantic:
                return semantic
            return str(potion.get("canonical_text") or potion.get("title") or "")
        return str(potion or "")

    def _shop_item_text(self, item: dict[str, Any]) -> str:
        if isinstance(item.get("card"), dict):
            return self._build_live_card_text(item.get("card"))
        if isinstance(item.get("relic"), dict):
            return self._relic_text(item.get("relic"))
        if isinstance(item.get("potion"), dict):
            return self._potion_text(item.get("potion"))
        return str(item.get("title") or item.get("canonical_text") or "")

    def _shop_item_entity_key(self, item: dict[str, Any]) -> str:
        for key in ("card", "relic", "potion"):
            payload = item.get(key)
            if isinstance(payload, dict):
                return self._support_entity_key(payload, fallback=f"shop:{key}")
        return self._support_entity_key(item, fallback="shop:item")

    def _reward_text(self, reward: dict[str, Any]) -> str:
        if isinstance(reward.get("card"), dict):
            return self._build_live_card_text(reward.get("card"))
        if isinstance(reward.get("relic"), dict):
            return self._relic_text(reward.get("relic"))
        if isinstance(reward.get("potion"), dict):
            return self._potion_text(reward.get("potion"))
        return str(reward.get("canonical_text") or reward.get("type") or reward.get("reward_type") or "")

    def _reward_entity_key(self, reward: dict[str, Any]) -> str:
        for key in ("card", "relic", "potion"):
            payload = reward.get(key)
            if isinstance(payload, dict):
                return self._support_entity_key(payload, fallback=f"reward:{key}")
        return self._support_entity_key(reward, fallback="reward")

    def _append_card_collection(
        self,
        world_entries: list[dict[str, Any]],
        numeric: np.ndarray,
        text: np.ndarray,
        mask: np.ndarray,
        token_type: str,
        *,
        entity_keys: list[Any] | None = None,
    ) -> None:
        owner_id = _owner_for_pile(token_type)
        for index in range(numeric.shape[0]):
            if mask[index] <= 0:
                continue
            row = np.zeros(TOKEN_NUMERIC_DIM, dtype=np.float32)
            row[: min(TOKEN_NUMERIC_DIM, numeric.shape[1])] = numeric[index][:TOKEN_NUMERIC_DIM]
            row[min(TOKEN_NUMERIC_DIM - 1, numeric.shape[1])] = min((index + 1) / max(numeric.shape[0], 1), 1.0)
            entity_key = f"{token_type}:{index}"
            if entity_keys is not None and index < len(entity_keys):
                card = entity_keys[index]
                if isinstance(card, dict):
                    entity_key = card.get("id") or card.get("title") or entity_key
                elif card:
                    entity_key = card
            world_entries.append(
                self._entry(
                    token_type,
                    row,
                    owner_id=owner_id,
                    entity_id=_stable_bucket(entity_key),
                    order_id=index + 1,
                    text_embedding=text[index],
                )
            )

    def _append_text_only_collection(
        self,
        world_entries: list[dict[str, Any]],
        text: np.ndarray,
        mask: np.ndarray,
        token_type: str,
        owner_id: int,
    ) -> None:
        for index in range(text.shape[0]):
            if mask[index] <= 0:
                continue
            row = np.zeros(TOKEN_NUMERIC_DIM, dtype=np.float32)
            row[0] = 1.0
            row[1] = min((index + 1) / max(text.shape[0], 1), 1.0)
            world_entries.append(
                self._entry(
                    token_type,
                    row,
                    owner_id=owner_id,
                    entity_id=_stable_bucket(f"{token_type}:{index}"),
                    order_id=index + 1,
                    text_embedding=text[index],
                )
            )

    def _encode_runtime_pile(self, obs: dict[str, Any], pile_key: str, fallback_key: str, *, limit: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cards = self._runtime_cards(obs, pile_key, fallback_key)
        count = min(len(cards) if isinstance(cards, list) else 0, limit)
        numeric = np.zeros((count, obs_common.CARD_FEAT_DIM), dtype=np.float32)
        text = np.zeros((count, TEXT_DIM), dtype=np.float32)
        mask = np.zeros(count, dtype=np.float32)
        if count > 0:
            self._enc_card_collection(cards[:count], numeric, text, mask)
        return numeric, text, mask

    def _runtime_cards(self, obs: dict[str, Any], pile_key: str, fallback_key: str) -> list[Any]:
        combat = obs.get("combat") or {}
        cards: list[Any] = []
        pile = combat.get(pile_key)
        if pile_key == "hand":
            if isinstance(pile, dict):
                cards = pile.get("cards") or []
            elif isinstance(combat.get("hand"), list):
                cards = combat.get("hand") or []
        elif isinstance(pile, dict):
            cards = pile.get("cards") or []
        elif isinstance(pile, list):
            cards = pile
        if not cards and isinstance(combat.get(fallback_key), list):
            cards = combat.get(fallback_key) or []
        return cards if isinstance(cards, list) else []

    def _materialize_entries(
        self,
        entries: list[dict[str, Any]],
        max_count: int,
        bufs: dict[str, np.ndarray] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if bufs is not None:
            token_array, token_mask = bufs["tokens"], bufs["mask"]
            type_ids, role_ids = bufs["type_ids"], bufs["role_ids"]
            owner_ids, entity_ids = bufs["owner_ids"], bufs["entity_ids"]
            zone_ids, order_ids = bufs["zone_ids"], bufs["order_ids"]
            target_owner_ids, target_entity_ids = bufs["target_owner_ids"], bufs["target_entity_ids"]
            for arr in (token_array, token_mask, type_ids, role_ids, owner_ids, entity_ids, zone_ids, order_ids, target_owner_ids, target_entity_ids):
                arr[:] = 0
        else:
            token_array = np.zeros((max_count, TOKEN_FEAT_DIM), dtype=np.float32)
            token_mask = np.zeros(max_count, dtype=np.float32)
            type_ids = np.zeros(max_count, dtype=np.int32)
            role_ids = np.zeros(max_count, dtype=np.int32)
            owner_ids = np.zeros(max_count, dtype=np.int32)
            entity_ids = np.zeros(max_count, dtype=np.int32)
            zone_ids = np.zeros(max_count, dtype=np.int32)
            order_ids = np.zeros(max_count, dtype=np.int32)
            target_owner_ids = np.zeros(max_count, dtype=np.int32)
            target_entity_ids = np.zeros(max_count, dtype=np.int32)

        for index, entry in enumerate(entries[:max_count]):
            token_array[index, :TOKEN_NUMERIC_DIM] = entry["numeric"][:TOKEN_NUMERIC_DIM]
            token_array[index, TOKEN_NUMERIC_DIM:] = entry["text_embedding"]
            token_mask[index] = 1.0
            type_ids[index] = int(entry["type_id"])
            role_ids[index] = int(entry["role_id"])
            owner_ids[index] = int(entry["owner_id"])
            entity_ids[index] = int(entry["entity_id"])
            zone_ids[index] = int(entry["zone_id"])
            order_ids[index] = int(entry["order_id"])
            target_owner_ids[index] = int(entry["target_owner_id"])
            target_entity_ids[index] = int(entry["target_entity_id"])
        return token_array, token_mask, type_ids, role_ids, owner_ids, entity_ids, zone_ids, order_ids, target_owner_ids, target_entity_ids

    def _materialize_nested_entries(
        self,
        entries: list[list[dict[str, Any]]],
        max_outer: int,
        max_inner: int,
        bufs: dict[str, np.ndarray] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if bufs is not None:
            token_array, token_mask = bufs["tokens"], bufs["mask"]
            type_ids, role_ids = bufs["type_ids"], bufs["role_ids"]
            owner_ids, entity_ids = bufs["owner_ids"], bufs["entity_ids"]
            zone_ids, order_ids = bufs["zone_ids"], bufs["order_ids"]
            for arr in (token_array, token_mask, type_ids, role_ids, owner_ids, entity_ids, zone_ids, order_ids):
                arr[:] = 0
        else:
            token_array = np.zeros((max_outer, max_inner, TOKEN_FEAT_DIM), dtype=np.float32)
            token_mask = np.zeros((max_outer, max_inner), dtype=np.float32)
            type_ids = np.zeros((max_outer, max_inner), dtype=np.int32)
            role_ids = np.zeros((max_outer, max_inner), dtype=np.int32)
            owner_ids = np.zeros((max_outer, max_inner), dtype=np.int32)
            entity_ids = np.zeros((max_outer, max_inner), dtype=np.int32)
            zone_ids = np.zeros((max_outer, max_inner), dtype=np.int32)
            order_ids = np.zeros((max_outer, max_inner), dtype=np.int32)
        for outer_index in range(min(len(entries), max_outer)):
            for inner_index, entry in enumerate(entries[outer_index][:max_inner]):
                token_array[outer_index, inner_index, :TOKEN_NUMERIC_DIM] = entry["numeric"][:TOKEN_NUMERIC_DIM]
                token_array[outer_index, inner_index, TOKEN_NUMERIC_DIM:] = entry["text_embedding"]
                token_mask[outer_index, inner_index] = 1.0
                type_ids[outer_index, inner_index] = int(entry["type_id"])
                role_ids[outer_index, inner_index] = int(entry["role_id"])
                owner_ids[outer_index, inner_index] = int(entry["owner_id"])
                entity_ids[outer_index, inner_index] = int(entry["entity_id"])
                zone_ids[outer_index, inner_index] = int(entry["zone_id"])
                order_ids[outer_index, inner_index] = int(entry["order_id"])
        return token_array, token_mask, type_ids, role_ids, owner_ids, entity_ids, zone_ids, order_ids

    def _entry(
        self,
        token_type: str,
        numeric: np.ndarray,
        *,
        role_id: int | None = None,
        zone_id: int | None = None,
        order_id: int = 0,
        owner_id: int,
        entity_id: int,
        target_owner_id: int = 0,
        target_entity_id: int = 0,
        text: str | None = None,
        text_embedding: np.ndarray | None = None,
    ) -> dict[str, Any]:
        if text_embedding is not None:
            text_embedding = _compress_text_embedding(text_embedding)
        return {
            "type_id": TOKEN_TYPE_TO_ID[token_type],
            "role_id": int(_role_for_token(token_type) if role_id is None else max(0, min(role_id, MAX_ROLE_ID))),
            "zone_id": int(_zone_for_token(token_type) if zone_id is None else max(0, min(zone_id, MAX_ZONE_ID))),
            "order_id": int(max(0, min(order_id, MAX_ORDER_ID))),
            "numeric": np.asarray(numeric, dtype=np.float32).reshape(-1),
            "owner_id": int(max(0, min(owner_id, MAX_OWNER_ID))),
            "entity_id": int(max(0, min(entity_id, ENTITY_HASH_BUCKETS - 1))),
            "target_owner_id": int(max(0, min(target_owner_id, MAX_OWNER_ID))),
            "target_entity_id": int(max(0, min(target_entity_id, ENTITY_HASH_BUCKETS - 1))),
            "text_embedding": text_embedding,
            "text": str(text or "").strip(),
        }

    def _boss_player_state(self) -> dict[str, float]:
        context = self._current_boss_context
        if isinstance(context, dict):
            player_state = context.get("player_state")
            if isinstance(player_state, dict):
                return player_state
        return {}

    def _boss_enemy_state(self, enemy: dict[str, Any] | None) -> dict[str, float]:
        if not isinstance(enemy, dict):
            return {}
        context = self._current_boss_context
        if not isinstance(context, dict):
            return {}
        enemy_states_by_key = context.get("enemy_states_by_key")
        if not isinstance(enemy_states_by_key, dict):
            return {}
        state = enemy_states_by_key.get(enemy_mechanics_key(enemy, ""))
        return state if isinstance(state, dict) else {}

    def _boss_enemy_traits(self, enemy: dict[str, Any] | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if not isinstance(enemy, dict):
            return [], []
        context = self._current_boss_context
        if not isinstance(context, dict):
            return [], []
        enemy_traits_by_key = context.get("enemy_traits_by_key")
        if not isinstance(enemy_traits_by_key, dict):
            return [], []
        payload = enemy_traits_by_key.get(enemy_mechanics_key(enemy, ""))
        if not isinstance(payload, dict):
            return [], []
        reactive_traits = payload.get("reactive_traits")
        phase_rules = payload.get("phase_rules")
        return (
            reactive_traits if isinstance(reactive_traits, list) else [],
            phase_rules if isinstance(phase_rules, list) else [],
        )

    def _infer_enemy_traits(self, enemy: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        reactive_traits: list[dict[str, Any]] = []
        phase_rules: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str, str]] = set()

        def _classify(entry: dict[str, Any]) -> list[dict[str, Any]]:
            category = str(entry.get("category") or "").lower()
            text = " ".join(
                str(entry.get(field) or "")
                for field in ("trait", "description", "effect_type", "condition", "state")
            ).lower()
            if category == "phase" or any(keyword in text for keyword in ("phase", "split", "threshold", "stun", "intangible")):
                return phase_rules
            return reactive_traits

        def _add(entry: Any) -> None:
            if not isinstance(entry, dict):
                return
            key = (
                str(entry.get("category") or ""),
                str(entry.get("trait") or entry.get("effect_type") or ""),
                str(entry.get("trigger_type") or ""),
                str(entry.get("condition") or entry.get("state") or entry.get("description") or ""),
            )
            if key in seen:
                return
            seen.add(key)
            _classify(entry).append(entry)

        metadata = get_enemy_metadata(str(enemy.get("model_id") or "").strip())
        if isinstance(metadata, dict):
            for collection_name in ("static_traits", "reactive_triggers", "phase_rules", "trait_tokens"):
                for item in metadata.get(collection_name) or []:
                    _add(item)

        for collection_name in ("static_traits", "reactive_triggers", "phase_rules"):
            for item in enemy.get(collection_name) or []:
                _add(item)

        name_text = " ".join(
            part for part in (
                str(enemy.get("name") or "").strip(),
                str(enemy.get("model_id") or "").strip(),
                str(((enemy.get("intent") or {}).get("description") or "")).strip(),
            ) if part
        ).lower()
        if "split" in name_text:
            _add({"category": "phase", "trait": "split_on_threshold", "description": "split_on_threshold"})
        if "phase" in name_text or "threshold" in name_text:
            _add({"category": "phase", "trait": "hp_threshold_phase_shift", "description": "hp_threshold_phase_shift"})
        for power in enemy.get("powers") or []:
            if not isinstance(power, dict):
                continue
            title = str(power.get("title") or "").lower()
            if "thorn" in title or "spike" in title:
                _add({"category": "reactive", "trait": "thorns", "effect_amount": power.get("amount"), "description": power.get("title")})
            if "retali" in title or "contact" in title:
                _add({"category": "reactive", "trait": "contact_retaliate", "effect_amount": power.get("amount"), "description": power.get("title")})
            if "intang" in title:
                _add({"category": "phase", "trait": "gain_intangible", "description": power.get("title")})
        boss_reactive_traits, boss_phase_rules = self._boss_enemy_traits(enemy)
        for item in boss_reactive_traits:
            _add(item)
        for item in boss_phase_rules:
            _add(item)
        return reactive_traits, phase_rules

    def _enemy_reaction_flags(self, enemy: dict[str, Any] | None) -> dict[str, float]:
        if not isinstance(enemy, dict):
            return {
                "thorns": 0.0,
                "contact_punish": 0.0,
                "split": 0.0,
                "threshold": 0.0,
                "threshold_value": 0.0,
                "artifact": 0.0,
                "buffer": 0.0,
                "intangible": 0.0,
                "incoming_damage_multiplier": 0.0,
                "back_attack": 0.0,
                "damage_cap": 0.0,
                "damage_cap_value": 0.0,
                "deathburst": 0.0,
                "deathburst_damage": 0.0,
                "revive": 0.0,
                "transform": 0.0,
                "linked_support_alive": 0.0,
                "special_phase": 0.0,
                "one_card_lock": 0.0,
                "skill_punish": 0.0,
                "choice_debuffs": 0.0,
                "escape_card_tax": 0.0,
                "countdown": 0.0,
                "stun_window": 0.0,
                "binding_control": 0.0,
                "wound_phase": 0.0,
            }

        reactive_traits, phase_rules = self._infer_enemy_traits(enemy)
        boss_state = self._boss_enemy_state(enemy)
        texts: list[str] = [build_live_enemy_semantic_text(enemy).lower()]
        threshold_value = 0.0
        for collection_name in ("powers", "static_traits", "reactive_triggers", "phase_rules"):
            collection = enemy.get(collection_name)
            if not isinstance(collection, list):
                continue
            for item in collection:
                if not isinstance(item, dict):
                    continue
                texts.extend(
                    str(item.get(key) or "").strip().lower()
                    for key in ("title", "description", "trait", "effect_type", "condition", "state")
                    if str(item.get(key) or "").strip()
                )
                threshold_value = max(
                    threshold_value,
                    abs(obs_common._float(item.get("threshold"))),
                    abs(obs_common._float(item.get("amount"))),
                    abs(obs_common._float(item.get("effect_amount"))),
                )
        for item in reactive_traits + phase_rules:
            threshold_value = max(
                threshold_value,
                abs(obs_common._float(item.get("threshold"))),
                abs(obs_common._float(item.get("amount"))),
                abs(obs_common._float(item.get("effect_amount"))),
            )

        joined = " | ".join(texts)
        return {
            "thorns": float(any(keyword in joined for keyword in ("thorn", "spike"))),
            "contact_punish": float(any(keyword in joined for keyword in ("thorn", "spike", "retaliat", "contact", "punish"))),
            "split": float(any(keyword in joined for keyword in ("split",))),
            "threshold": float(any(keyword in joined for keyword in ("threshold", "phase", "stun", "hp_le"))),
            "threshold_value": min(threshold_value / 100.0, 1.0),
            "artifact": float("artifact" in joined),
            "buffer": float("buffer" in joined),
            "intangible": float("intang" in joined),
            "incoming_damage_multiplier": float(boss_state.get("incoming_damage_multiplier_norm", 0.0)),
            "back_attack": float(boss_state.get("back_attack_active", 0.0)),
            "damage_cap": float(boss_state.get("damage_cap_active", 0.0)),
            "damage_cap_value": float(boss_state.get("damage_cap_value_norm", 0.0)),
            "deathburst": float(boss_state.get("deathburst", 0.0)),
            "deathburst_damage": float(boss_state.get("deathburst_damage_norm", 0.0)),
            "revive": float(boss_state.get("revive_once", 0.0)),
            "transform": float(boss_state.get("transform_pending", 0.0)),
            "linked_support_alive": float(boss_state.get("linked_support_alive", 0.0)),
            "special_phase": float(boss_state.get("special_phase_active", 0.0)),
            "one_card_lock": float(boss_state.get("one_card_lock", 0.0)),
            "skill_punish": float(boss_state.get("skill_punish", 0.0)),
            "choice_debuffs": float(boss_state.get("choice_debuffs", 0.0)),
            "escape_card_tax": float(boss_state.get("escape_card_tax", 0.0)),
            "countdown": float(boss_state.get("countdown_active", 0.0)),
            "stun_window": float(boss_state.get("stun_window", 0.0)),
            "binding_control": float(boss_state.get("binding_control", 0.0)),
            "wound_phase": float(boss_state.get("wound_phase", 0.0)),
        }

    def _trait_numeric(self, trait: dict[str, Any]) -> np.ndarray:
        numeric = np.zeros(TOKEN_NUMERIC_DIM, dtype=np.float32)
        text = " | ".join(
            str(trait.get(key) or "").lower()
            for key in ("trait", "effect_type", "description", "condition", "state")
            if str(trait.get(key) or "").strip()
        )
        numeric[0] = float("thorn" in text)
        numeric[1] = float("contact" in text or "retali" in text)
        numeric[2] = float("split" in text)
        numeric[3] = float("phase" in text)
        numeric[4] = float("threshold" in text or "stun" in text)
        numeric[5] = min(abs(obs_common._float(trait.get("effect_amount") or trait.get("amount") or trait.get("threshold"))) / 10.0, 1.0)
        numeric[6] = float("back_attack" in text or "matched_facing" in text or "damage_multiplier" in text)
        numeric[7] = float("damage_cap" in text or "slippery" in text or "opening_cycle" in text)
        numeric[8] = float("death_explosion" in text or "deathburst" in text or "steam eruption" in text)
        numeric[9] = float("revive" in text or "reborn" in text or "resurrect" in text)
        numeric[10] = float("binding" in text or "choice_debuffs" in text)
        numeric[11] = float("linked_support" in text or "support_body" in text or "door_destroyed" in text)
        numeric[12] = float("countdown" in text or "doom_clock" in text or "escape_card_tax" in text)
        numeric[13] = float("skill_punish" in text or "on_play_skill" in text)
        numeric[14] = float("intang" in text)
        numeric[15] = float("one_card_lock" in text or "play_budget_locked" in text)
        numeric[16] = float("wound_phase" in text or "chip_damage" in text)
        numeric[17] = float("phase_three" in text or "opening" in text or "phase_two" in text)
        return numeric

    @staticmethod
    def _is_selection_action(action: dict[str, Any]) -> bool:
        kind = str(action.get("kind") or "").strip()
        action_id = str(action.get("action_id") or "").strip()
        return kind in {"card_selection", "combat_select_card", "combat_select"} or action_id.startswith("combat_select")

    @staticmethod
    def _selection_semantics_text(action: dict[str, Any]) -> str:
        return " | ".join(
            part
            for part in (
                str(action.get("selection_semantics") or "").strip(),
                str(action.get("selection_prompt") or "").strip(),
                str(action.get("surface") or "").strip(),
            )
            if part
        )

    def _infer_action_domain(self, action: dict[str, Any], obs: dict[str, Any]) -> str:
        kind = str(action.get("kind") or "").strip()
        if self._is_selection_action(action):
            return "selection"
        if kind in {"play_card", "use_potion", "discard_potion", "combat", "proceed"} and obs.get("combat"):
            return "combat"
        if kind == "map" or isinstance(action.get("route_summary"), dict):
            return "route"
        if (obs.get("phase") or "") == "map":
            return "route"
        return "build"

    def _infer_card_selection_source_pile(self, action: dict[str, Any], obs: dict[str, Any]) -> tuple[str, int]:
        card = action.get("card") if isinstance(action.get("card"), dict) else None
        if card is None:
            return "none", 0

        pile_candidates = [
            ("hand", self._runtime_cards(obs, "hand", "hand")),
            ("draw", self._runtime_cards(obs, "draw_pile", "draw_preview_cards")),
            ("discard", self._runtime_cards(obs, "discard_pile", "discard_cards")),
            ("exhaust", self._runtime_cards(obs, "exhaust_pile", "exhaust_cards")),
            ("play", self._runtime_cards(obs, "play_pile", "play_pile_cards")),
            ("deck", (obs.get("player") or {}).get("deck_cards") if isinstance(obs.get("player"), dict) else []),
        ]

        matches: list[tuple[str, int]] = []
        for pile_name, cards in pile_candidates:
            position = self._find_source_position(cards if isinstance(cards, list) else [], card)
            if position > 0:
                matches.append((pile_name, position))
        if len(matches) == 1:
            return matches[0]

        prompt = " ".join(
            str(value or "")
            for value in (
                action.get("selection_semantics"),
                action.get("selection_prompt"),
                action.get("label"),
                action.get("screen_type"),
            )
        ).lower()
        keyword_map = (
            ("discard", ("discard",)),
            ("draw", ("draw pile", "draw")),
            ("exhaust", ("exhaust",)),
            ("hand", ("hand",)),
            ("play", ("play pile", "played")),
            ("deck", ("deck",)),
        )
        for pile_name, keywords in keyword_map:
            if any(keyword in prompt for keyword in keywords):
                for matched_name, matched_pos in matches:
                    if matched_name == pile_name:
                        return matched_name, matched_pos
                return pile_name, 0

        if matches:
            return matches[0]
        return "unknown", 0

    @staticmethod
    def _owner_zone_from_source_pile(source_pile: str) -> tuple[int, int]:
        return {
            "hand": (OWNER_HAND, TOKEN_ZONE_TO_ID["HAND"]),
            "draw": (OWNER_DRAW, TOKEN_ZONE_TO_ID["DRAW"]),
            "discard": (OWNER_DISCARD, TOKEN_ZONE_TO_ID["DISCARD"]),
            "exhaust": (OWNER_EXHAUST, TOKEN_ZONE_TO_ID["EXHAUST"]),
            "play": (OWNER_PLAY, TOKEN_ZONE_TO_ID["PLAY"]),
            "deck": (OWNER_DECK, TOKEN_ZONE_TO_ID["DECK"]),
            "reward": (OWNER_REWARD, TOKEN_ZONE_TO_ID["REWARD"]),
            "upgrade": (OWNER_UPGRADE, TOKEN_ZONE_TO_ID["UPGRADE"]),
            "potion": (OWNER_POTION, TOKEN_ZONE_TO_ID["POTION"]),
            "route": (OWNER_ROUTE, TOKEN_ZONE_TO_ID["ROUTE"]),
            "selection": (OWNER_NONE, TOKEN_ZONE_TO_ID["SELECTION"]),
        }.get(source_pile, (OWNER_NONE, TOKEN_ZONE_TO_ID["NONE"]))

    def _action_source_binding(self, action: dict[str, Any], obs: dict[str, Any], action_index: int) -> tuple[int, int, int]:
        kind = str(action.get("kind") or "").strip()
        if kind == "play_card" and isinstance(action.get("card"), dict):
            return (
                OWNER_HAND,
                TOKEN_ZONE_TO_ID["HAND"],
                self._find_source_position(self._runtime_cards(obs, "hand", "hand"), action.get("card")),
            )
        if kind in {"use_potion", "discard_potion"} and isinstance(action.get("potion"), dict):
            player = obs.get("player") or {}
            return (
                OWNER_POTION,
                TOKEN_ZONE_TO_ID["POTION"],
                self._find_source_position(player.get("potions") if isinstance(player.get("potions"), list) else [], action.get("potion")),
            )
        if kind == "card_reward":
            return OWNER_REWARD, TOKEN_ZONE_TO_ID["REWARD"], min(action_index + 1, MAX_ORDER_ID)
        if kind == "deck_upgrade":
            return OWNER_UPGRADE, TOKEN_ZONE_TO_ID["UPGRADE"], min(action_index + 1, MAX_ORDER_ID)
        if kind == "map":
            return OWNER_ROUTE, TOKEN_ZONE_TO_ID["ROUTE"], min(action_index + 1, MAX_ORDER_ID)
        if self._is_selection_action(action) and isinstance(action.get("card"), dict):
            source_pile, position = self._infer_card_selection_source_pile(action, obs)
            owner_id, zone_id = self._owner_zone_from_source_pile(source_pile)
            if zone_id == TOKEN_ZONE_TO_ID["NONE"]:
                zone_id = TOKEN_ZONE_TO_ID["SELECTION"]
            return owner_id, zone_id, position or min(action_index + 1, MAX_ORDER_ID)
        if self._is_selection_action(action):
            return OWNER_NONE, TOKEN_ZONE_TO_ID["SELECTION"], min(action_index + 1, MAX_ORDER_ID)
        return OWNER_NONE, TOKEN_ZONE_TO_ID["NONE"], min(action_index + 1, MAX_ORDER_ID)

    def _action_entity_key(self, action: dict[str, Any]) -> str:
        for source in (
            action.get("card"),
            action.get("potion"),
            action.get("upgrade_preview"),
            (action.get("item") or {}).get("card") if isinstance(action.get("item"), dict) else None,
            (action.get("item") or {}).get("relic") if isinstance(action.get("item"), dict) else None,
            (action.get("item") or {}).get("potion") if isinstance(action.get("item"), dict) else None,
            (action.get("reward") or {}).get("card") if isinstance(action.get("reward"), dict) else None,
            (action.get("reward") or {}).get("relic") if isinstance(action.get("reward"), dict) else None,
            (action.get("reward") or {}).get("potion") if isinstance(action.get("reward"), dict) else None,
        ):
            if isinstance(source, dict):
                return str(source.get("id") or source.get("model_id") or source.get("title") or source.get("name") or action.get("action_id") or "")
        return str(action.get("action_id") or action.get("kind") or "")

    @staticmethod
    def _enemy_entity_key(enemy: dict[str, Any] | None, fallback: Any) -> str:
        if isinstance(enemy, dict):
            for key in ("combat_id", "id", "model_id", "name"):
                value = enemy.get(key)
                if value not in (None, ""):
                    return str(value)
        return str(fallback)

    def _query_zone_id(self, action: dict[str, Any], obs: dict[str, Any], query_type: str, target_enemy_index: int | None) -> int:
        kind = str(action.get("kind") or "").strip()
        if kind == "play_card":
            return TOKEN_ZONE_TO_ID["HAND"]
        if kind in {"use_potion", "discard_potion"}:
            return TOKEN_ZONE_TO_ID["POTION"]
        if self._is_selection_action(action):
            source_pile, _position = self._infer_card_selection_source_pile(action, obs)
            _owner_id, zone_id = self._owner_zone_from_source_pile(source_pile)
            if zone_id != TOKEN_ZONE_TO_ID["NONE"]:
                return zone_id
            return TOKEN_ZONE_TO_ID["SELECTION"]
        if kind == "map" or query_type == "ROUTE_CANDIDATE":
            return TOKEN_ZONE_TO_ID["ROUTE"]
        if kind == "shop":
            return TOKEN_ZONE_TO_ID["SHOP"]
        if kind == "deck_upgrade":
            return TOKEN_ZONE_TO_ID["UPGRADE"]
        if kind in {"reward", "card_reward", "event_option", "treasure_relic"}:
            return TOKEN_ZONE_TO_ID["REWARD"]
        if target_enemy_index is not None:
            return TOKEN_ZONE_TO_ID["ENEMY"]
        return _zone_for_token(query_type)

    def _source_order_id(self, action: dict[str, Any], obs: dict[str, Any], action_index: int) -> int:
        _owner_id, _zone_id, order_id = self._action_source_binding(action, obs, action_index)
        return order_id

    @staticmethod
    def _find_source_position(entries: list[Any], source: dict[str, Any] | None) -> int:
        if not isinstance(entries, list) or not isinstance(source, dict):
            return 0
        source_id = str(source.get("id") or "").strip()
        source_title = str(source.get("title") or "").strip().lower()
        for index, entry in enumerate(entries[:MAX_ORDER_ID]):
            if isinstance(entry, dict):
                entry_id = str(entry.get("id") or "").strip()
                entry_title = str(entry.get("title") or "").strip().lower()
                if source_id and entry_id and source_id == entry_id:
                    return index + 1
                if source_title and entry_title and source_title == entry_title:
                    return index + 1
            elif isinstance(entry, str):
                entry_text = entry.strip().lower()
                if source_id and entry_text == source_id.lower():
                    return index + 1
                if source_title and entry_text == source_title:
                    return index + 1
        return 0

    def _match_target_enemy(self, action: dict[str, Any], combat: dict[str, Any]) -> tuple[int | None, dict[str, Any] | None]:
        target = action.get("target") if isinstance(action.get("target"), dict) else None
        enemies = combat.get("enemies") or []
        if not isinstance(target, dict):
            return None, None
        target_combat_id = target.get("combat_id")
        target_name = str(target.get("name") or "").strip().lower()
        for index, enemy in enumerate(enemies[: obs_common.MAX_ENEMIES]):
            if not isinstance(enemy, dict):
                continue
            if target_combat_id is not None and enemy.get("combat_id") == target_combat_id:
                return index, enemy
            if target_name and str(enemy.get("name") or "").strip().lower() == target_name:
                return index, enemy
        return None, None

    def _infer_action_owner(self, action: dict[str, Any], target_enemy_index: int | None) -> int:
        if self._is_selection_action(action) and isinstance(action.get("card"), dict):
            source_pile, _position = self._infer_card_selection_source_pile(action, {})
            owner_id, _zone_id = self._owner_zone_from_source_pile(source_pile)
            if owner_id != OWNER_NONE:
                return owner_id
        owner_id, _zone_id, _order_id = self._action_source_binding(action, {}, 0)
        if owner_id != OWNER_NONE:
            return owner_id
        if target_enemy_index is not None:
            return self._enemy_owner_id(target_enemy_index)
        return OWNER_NONE

    def _enemy_owner_id(self, enemy_index: int | None) -> int:
        if enemy_index is None:
            return OWNER_NONE
        return int(max(OWNER_ENEMY_BASE, min(OWNER_ENEMY_BASE + enemy_index, MAX_OWNER_ID)))


ObservationEncoderV3 = WorldTokenObservationEncoder

