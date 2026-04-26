"""Token-world memory encoder and latent transition blocks for MuZero.

This module ports the mainline token-world attention ideas into the archived
MuZero stack without pulling in SB3 policy dependencies. The goal is to let
MuZero consume the richer world-token observation stream and keep a latent
memory-slot state that can model:

- hand / draw / discard / exhaust loop structure
- relic / potion support context
- energy / X-cost and resource budget context
- enemy core state, buffs, intent, and reactions
- route / build / history context when those banks are present

The external MuZero interface still sees a flat hidden state tensor so MCTS
does not need structural changes; internally we unflatten to
``[batch, num_memory_slots, d_model]`` and update it with attention.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from muzero.sts2_env.semantic_rollout import SEMANTIC_ROLLOUT_SIZE
from sts2_env.attention_blocks import CrossAttentionBlock, EntityPooling, RelationBias, TransformerEncoderBlock
from sts2_env.observation_v2 import MAX_ACTIONS, NUM_DOMAINS, NUM_PHASES
from sts2_env.observation_v3 import (
    ENTITY_HASH_BUCKETS,
    MAX_OWNER_ID,
    MAX_ORDER_ID,
    MAX_ROLE_ID,
    MAX_WORLD_TOKENS,
    MAX_ZONE_ID,
    NUM_TOKEN_TYPES,
    POWER_ID_BUCKETS,
    TOKEN_FEAT_DIM,
    TOKEN_NUMERIC_DIM,
    TOKEN_ROLE_TO_ID,
    TOKEN_TEXT_DIM,
    TOKEN_TYPE_TO_ID,
    TOKEN_ZONE_TO_ID,
)
from sts2_env.objective_heads import (
    HEAD_HP_PRESERVATION,
    HEAD_SURVIVAL,
    NUM_OBJECTIVE_HEADS,
    scalarize_objective_components_torch,
)


_POWER_SLOT_ROLE_ID = TOKEN_ROLE_TO_ID.get("POWER_SLOT", 0)
_HISTORY_ROLE_ID = TOKEN_ROLE_TO_ID.get("HISTORY", 0)
_RELATION_BIAS_KW = {
    "num_token_types": NUM_TOKEN_TYPES,
    "max_owner_id": MAX_OWNER_ID,
    "max_role_id": MAX_ROLE_ID,
    "max_zone_id": MAX_ZONE_ID,
    "max_order_id": MAX_ORDER_ID,
    "power_bucket_count": POWER_ID_BUCKETS,
    "power_slot_role_id": _POWER_SLOT_ROLE_ID,
    "history_card_bucket_count": ENTITY_HASH_BUCKETS,
    "history_role_id": _HISTORY_ROLE_ID,
}

WORLD_BANK_NAMES = ("runtime", "support", "enemy", "build", "route", "powers", "history")
_WORLD_BANK_ROLE_NAMES = {
    "runtime": (
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
        "PILE_LINK",
        "CYCLE_PLAN",
        "ENERGY_BUDGET",
    ),
    "support": ("RELIC_SUPPORT", "POTION_SUPPORT", "SUPPORT_GRAPH"),
    "enemy": ("ENEMY_CORE", "ENEMY_INTENT", "ENEMY_POWER", "ENEMY_TRAIT", "ENEMY_REACTION"),
    "build": ("DECK_CARD", "BUILD_STATE", "DECK_SYNERGY", "REWARD_OPTION", "SHOP_OPTION", "UPGRADE_OPTION"),
    "route": ("ROUTE_SUMMARY", "ROUTE_NODE", "ROUTE_RISK", "ROUTE_VALUE"),
    "powers": ("POWER_SLOT", "CARD_KEYWORD"),
    "history": ("HISTORY",),
}
_WORLD_BANK_ZONE_NAMES = {
    "runtime": ("WORLD", "PLAYER", "HAND", "DRAW", "DISCARD", "EXHAUST", "PLAY"),
    "support": ("RELIC", "POTION"),
    "enemy": ("ENEMY",),
    "build": ("DECK", "REWARD", "SHOP", "UPGRADE"),
    "route": ("ROUTE",),
    "powers": (),
    "history": (),
}
WORLD_BANK_ROLE_IDS = {
    bank_name: tuple(
        TOKEN_ROLE_TO_ID[role_name]
        for role_name in role_names
        if role_name in TOKEN_ROLE_TO_ID
    )
    for bank_name, role_names in _WORLD_BANK_ROLE_NAMES.items()
}
WORLD_BANK_ZONE_IDS = {
    bank_name: tuple(
        TOKEN_ZONE_TO_ID[zone_name]
        for zone_name in zone_names
        if zone_name in TOKEN_ZONE_TO_ID
    )
    for bank_name, zone_names in _WORLD_BANK_ZONE_NAMES.items()
}
GLOBAL_MEMORY_BANK_NAME = "global"
MEMORY_BANK_NAMES = WORLD_BANK_NAMES + (GLOBAL_MEMORY_BANK_NAME,)
GLOBAL_MEMORY_BANK_INDEX = len(WORLD_BANK_NAMES)
RISK_OBJECTIVE_HEAD_INDICES = (HEAD_SURVIVAL, HEAD_HP_PRESERVATION)


def _should_activation_checkpoint(enabled: bool, *args: object) -> bool:
    return bool(enabled) and torch.is_grad_enabled() and any(
        isinstance(arg, torch.Tensor) and bool(arg.requires_grad)
        for arg in args
    )


def _run_encoder_block(
    enabled: bool,
    block: TransformerEncoderBlock,
    x: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    attn_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    if _should_activation_checkpoint(enabled, x):
        def _forward(x_: torch.Tensor) -> torch.Tensor:
            return block(x_, mask=mask, attn_bias=attn_bias)

        return activation_checkpoint(_forward, x, use_reentrant=False)
    return block(x, mask=mask, attn_bias=attn_bias)


def _run_cross_block(
    enabled: bool,
    block: CrossAttentionBlock,
    query: torch.Tensor,
    memory: torch.Tensor,
    *,
    query_mask: torch.Tensor | None = None,
    memory_mask: torch.Tensor | None = None,
    attn_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    if _should_activation_checkpoint(enabled, query, memory):
        def _forward(query_: torch.Tensor, memory_: torch.Tensor) -> torch.Tensor:
            return block(
                query_,
                memory_,
                query_mask=query_mask,
                memory_mask=memory_mask,
                attn_bias=attn_bias,
            )

        return activation_checkpoint(_forward, query, memory, use_reentrant=False)
    return block(query, memory, query_mask=query_mask, memory_mask=memory_mask, attn_bias=attn_bias)


def _run_function_with_checkpoint(enabled: bool, fn, *args):
    if _should_activation_checkpoint(enabled, *args):
        return activation_checkpoint(fn, *args, use_reentrant=False)
    return fn(*args)


def build_memory_slot_bank_ids(num_memory_slots: int) -> list[int]:
    """Assign each latent slot to a persistent bank identity.

    Preferred layout:
      - one reserved slot per explicit world bank
      - any extra slots become global/planning slots

    When the caller requests fewer slots than explicit banks, we compress the
    coverage by evenly subsampling bank identities. The default configuration
    should keep at least one slot per bank.
    """
    slot_count = max(int(num_memory_slots), 1)
    bank_count = len(WORLD_BANK_NAMES)
    if slot_count >= bank_count:
        return list(range(bank_count)) + [GLOBAL_MEMORY_BANK_INDEX] * (slot_count - bank_count)
    if slot_count == 1:
        return [GLOBAL_MEMORY_BANK_INDEX]
    positions = torch.linspace(0, bank_count - 1, steps=slot_count).round().long().tolist()
    return [int(position) for position in positions]


def build_zone_transport_prior() -> torch.Tensor:
    """Construct a zone-to-zone transport prior for copy / migration logits."""

    num_zones = MAX_ZONE_ID + 1
    prior = torch.zeros((num_zones, num_zones), dtype=torch.float32)
    prior.fill_diagonal_(0.2)

    def add(target_zone: str, source_zone: str, value: float) -> None:
        target_id = TOKEN_ZONE_TO_ID.get(target_zone)
        source_id = TOKEN_ZONE_TO_ID.get(source_zone)
        if target_id is None or source_id is None:
            return
        prior[target_id, source_id] += float(value)

    for stable_zone in (
        "WORLD",
        "PLAYER",
        "ENEMY",
        "RELIC",
        "POTION",
        "DECK",
        "ROUTE",
        "SHOP",
        "REWARD",
        "UPGRADE",
        "HISTORY",
    ):
        add(stable_zone, stable_zone, 0.55)

    add("HAND", "DRAW", 1.25)
    add("HAND", "HAND", 0.35)
    add("DISCARD", "HAND", 1.10)
    add("EXHAUST", "HAND", 1.00)
    add("PLAY", "HAND", 0.85)
    add("DISCARD", "PLAY", 0.75)
    add("EXHAUST", "PLAY", 0.65)
    add("PLAY", "PLAY", 0.35)
    add("DRAW", "DISCARD", 0.60)
    add("DISCARD", "DISCARD", 0.40)
    add("DRAW", "DRAW", 0.35)
    add("EXHAUST", "EXHAUST", 0.65)
    add("HAND", "DISCARD", 0.20)
    add("DRAW", "HAND", 0.15)
    return prior


class TokenMemoryEncoderOutput(NamedTuple):
    hidden_state: torch.Tensor
    action_embeddings: torch.Tensor
    world_pool: torch.Tensor
    world_bank_summaries: torch.Tensor | None
    world_bank_occupancy: torch.Tensor | None
    world_bank_token_presence: torch.Tensor | None
    world_bank_token_distribution: torch.Tensor | None
    world_bank_token_slot_states: torch.Tensor | None
    world_bank_token_slot_mask: torch.Tensor | None
    world_bank_token_slot_type_ids: torch.Tensor | None
    world_bank_token_slot_zone_ids: torch.Tensor | None
    world_bank_token_slot_entity_ids: torch.Tensor | None
    world_bank_token_slot_order_ids: torch.Tensor | None
    decision_domain: torch.Tensor | None


def infer_token_decision_domain(
    obs: dict[str, torch.Tensor],
    *,
    device: torch.device | None = None,
) -> torch.Tensor | None:
    """Infer dense-v2 style decision_domain one-hot from token query roles."""
    query_roles = obs.get("candidate_query_role_ids")
    if query_roles is None:
        return None
    role_ids = torch.as_tensor(query_roles, dtype=torch.long, device=device)
    if role_ids.ndim == 1:
        role_ids = role_ids.unsqueeze(0)

    action_mask = obs.get("action_mask")
    if action_mask is not None:
        action_mask = torch.as_tensor(action_mask, dtype=torch.bool, device=role_ids.device)
        if action_mask.ndim == 1:
            action_mask = action_mask.unsqueeze(0)
    else:
        action_mask = torch.ones_like(role_ids, dtype=torch.bool)

    inferred = torch.zeros((role_ids.shape[0], NUM_DOMAINS), dtype=torch.float32, device=role_ids.device)
    role_to_domain = {
        TOKEN_ROLE_TO_ID.get("QUERY_COMBAT", -1): 0,
        TOKEN_ROLE_TO_ID.get("QUERY_BUILD", -1): 1,
        TOKEN_ROLE_TO_ID.get("QUERY_SELECTION", -1): 1,
        TOKEN_ROLE_TO_ID.get("QUERY_ROUTE", -1): 2,
    }
    for role_id, domain_idx in role_to_domain.items():
        if role_id < 0:
            continue
        inferred[:, domain_idx] = ((role_ids == role_id) & action_mask).any(dim=1).float()

    unresolved = inferred.sum(dim=1, keepdim=True) <= 0.0
    if unresolved.any():
        world_type_ids = obs.get("world_token_type_ids")
        world_token_mask = obs.get("world_token_mask")
        if world_type_ids is not None:
            world_type_ids = torch.as_tensor(world_type_ids, dtype=torch.long, device=role_ids.device)
            if world_type_ids.ndim == 1:
                world_type_ids = world_type_ids.unsqueeze(0)
            if world_token_mask is not None:
                world_token_mask = torch.as_tensor(world_token_mask, dtype=torch.bool, device=role_ids.device)
                if world_token_mask.ndim == 1:
                    world_token_mask = world_token_mask.unsqueeze(0)
            else:
                world_token_mask = torch.ones_like(world_type_ids, dtype=torch.bool)
            cls_type_to_domain = {
                TOKEN_TYPE_TO_ID.get("CLS_COMBAT", -1): 0,
                TOKEN_TYPE_TO_ID.get("CLS_BUILD", -1): 1,
                TOKEN_TYPE_TO_ID.get("CLS_ROUTE", -1): 2,
            }
            fallback = torch.zeros_like(inferred)
            for type_id, domain_idx in cls_type_to_domain.items():
                if type_id < 0:
                    continue
                fallback[:, domain_idx] = ((world_type_ids == type_id) & world_token_mask).any(dim=1).float()
            inferred = torch.where(unresolved, fallback, inferred)
            unresolved = inferred.sum(dim=1, keepdim=True) <= 0.0
    if unresolved.any():
        inferred = inferred.clone()
        inferred[unresolved.squeeze(-1), 1] = 1.0
    return inferred


class EntityTokenEmbedder(nn.Module):
    """Standalone token embedder copied out of the mainline omni policy."""

    def __init__(
        self,
        *,
        d_model: int,
        num_token_types: int,
        max_owner_id: int,
        max_role_id: int,
        max_zone_id: int,
        max_order_id: int,
        entity_hash_buckets: int,
        use_internal_numeric_proj: bool = True,
        use_internal_text_proj: bool = True,
    ) -> None:
        super().__init__()
        self.num_token_types = int(num_token_types)
        self.max_owner_id = int(max_owner_id)
        self.max_role_id = int(max_role_id)
        self.max_zone_id = int(max_zone_id)
        self.max_order_id = int(max_order_id)
        self.entity_hash_buckets = int(entity_hash_buckets)
        half = d_model // 2
        self.numeric_width = half
        self.text_width = d_model - half
        self.numeric_proj = (
            nn.Sequential(
                nn.Linear(TOKEN_NUMERIC_DIM, self.numeric_width),
                nn.GELU(),
                nn.Linear(self.numeric_width, self.numeric_width),
            )
            if use_internal_numeric_proj
            else None
        )
        self.text_proj = (
            nn.Sequential(
                nn.Linear(TOKEN_TEXT_DIM, self.text_width),
                nn.GELU(),
                nn.Linear(self.text_width, self.text_width),
            )
            if use_internal_text_proj
            else None
        )
        self.type_embedding = nn.Embedding(self.num_token_types, d_model)
        self.role_embedding = nn.Embedding(self.max_role_id + 1, d_model)
        self.owner_embedding = nn.Embedding(self.max_owner_id + 1, d_model)
        self.zone_embedding = nn.Embedding(self.max_zone_id + 1, d_model)
        self.order_embedding = nn.Embedding(self.max_order_id + 1, d_model)
        self.entity_embedding = nn.Embedding(self.entity_hash_buckets, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        token_features: torch.Tensor,
        token_type_ids: torch.Tensor,
        role_ids: torch.Tensor,
        owner_ids: torch.Tensor,
        entity_ids: torch.Tensor,
        zone_ids: torch.Tensor,
        order_ids: torch.Tensor,
        *,
        target_owner_ids: torch.Tensor | None = None,
        target_entity_ids: torch.Tensor | None = None,
        projected_numeric: torch.Tensor | None = None,
        projected_text: torch.Tensor | None = None,
    ) -> torch.Tensor:
        numeric = token_features[..., :TOKEN_NUMERIC_DIM]
        text = token_features[..., TOKEN_NUMERIC_DIM : TOKEN_NUMERIC_DIM + TOKEN_TEXT_DIM]
        token_type_ids = token_type_ids.clamp(min=0, max=self.num_token_types - 1)
        role_ids = role_ids.clamp(min=0, max=self.max_role_id)
        owner_ids = owner_ids.clamp(min=0, max=self.max_owner_id)
        zone_ids = zone_ids.clamp(min=0, max=self.max_zone_id)
        order_ids = order_ids.clamp(min=0, max=self.max_order_id)
        entity_ids = entity_ids.clamp(min=0, max=self.entity_hash_buckets - 1)
        if projected_numeric is None:
            if self.numeric_proj is None:
                raise RuntimeError("projected_numeric must be provided when use_internal_numeric_proj=False")
            projected_numeric = self.numeric_proj(numeric)
        if projected_text is None:
            if self.text_proj is None:
                raise RuntimeError("projected_text must be provided when use_internal_text_proj=False")
            projected_text = self.text_proj(text)
        x = torch.cat([projected_numeric, projected_text], dim=-1)
        x = (
            x
            + self.type_embedding(token_type_ids)
            + self.role_embedding(role_ids)
            + self.owner_embedding(owner_ids)
            + self.zone_embedding(zone_ids)
            + self.order_embedding(order_ids)
            + self.entity_embedding(entity_ids)
        )
        if target_owner_ids is not None:
            target_owner_ids = target_owner_ids.clamp(min=0, max=self.max_owner_id)
            x = x + 0.5 * self.owner_embedding(target_owner_ids)
        if target_entity_ids is not None:
            target_entity_ids = target_entity_ids.clamp(min=0, max=self.entity_hash_buckets - 1)
            x = x + 0.5 * self.entity_embedding(target_entity_ids)
        return self.norm(x)


class TokenMemoryEncoder(nn.Module):
    """Main token-world observation encoder that emits MuZero latent memory slots."""

    def __init__(
        self,
        *,
        d_model: int = 128,
        n_heads: int = 4,
        ffn_dim: int = 512,
        world_layers: int = 4,
        local_layers: int = 1,
        decoder_layers: int = 2,
        candidate_set_layers: int = 1,
        world_bank_top_k: int = 3,
        bank_token_slots: int = 4,
        num_memory_slots: int = 8,
        action_embed_dim: int = 64,
        dropout: float = 0.0,
        activation_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.ffn_dim = int(ffn_dim)
        self.world_layers = int(world_layers)
        self.local_layers = int(local_layers)
        self.decoder_layers = int(decoder_layers)
        self.candidate_set_layers = int(candidate_set_layers)
        self.world_bank_top_k = max(1, min(int(world_bank_top_k), len(WORLD_BANK_NAMES)))
        self.bank_token_slots = max(1, min(int(bank_token_slots), MAX_WORLD_TOKENS))
        self.num_memory_slots = max(int(num_memory_slots), 1)
        self.action_embed_dim = int(action_embed_dim)
        self.hidden_dim = self.num_memory_slots * self.d_model
        self.activation_checkpointing = bool(activation_checkpointing)
        self.projection_reuse_max_rows = 16_384
        self.num_slot_banks = len(MEMORY_BANK_NAMES)
        self.global_memory_bank_index = GLOBAL_MEMORY_BANK_INDEX

        self.shared_numeric_width = self.d_model // 2
        self.shared_text_width = self.d_model - self.shared_numeric_width
        self.shared_numeric_trunk = nn.Sequential(
            nn.Linear(TOKEN_NUMERIC_DIM, self.shared_numeric_width),
            nn.GELU(),
            nn.Linear(self.shared_numeric_width, self.shared_numeric_width),
        )
        self.shared_text_trunk = nn.Sequential(
            nn.Linear(TOKEN_TEXT_DIM, self.shared_text_width),
            nn.GELU(),
            nn.Linear(self.shared_text_width, self.shared_text_width),
        )
        self.world_embedder = EntityTokenEmbedder(
            d_model=self.d_model,
            num_token_types=NUM_TOKEN_TYPES,
            max_owner_id=MAX_OWNER_ID,
            max_role_id=MAX_ROLE_ID,
            max_zone_id=MAX_ZONE_ID,
            max_order_id=MAX_ORDER_ID,
            entity_hash_buckets=ENTITY_HASH_BUCKETS,
            use_internal_numeric_proj=False,
            use_internal_text_proj=False,
        )
        self.query_embedder = EntityTokenEmbedder(
            d_model=self.d_model,
            num_token_types=NUM_TOKEN_TYPES,
            max_owner_id=MAX_OWNER_ID,
            max_role_id=MAX_ROLE_ID,
            max_zone_id=MAX_ZONE_ID,
            max_order_id=MAX_ORDER_ID,
            entity_hash_buckets=ENTITY_HASH_BUCKETS,
            use_internal_numeric_proj=False,
            use_internal_text_proj=False,
        )
        self.local_embedder = EntityTokenEmbedder(
            d_model=self.d_model,
            num_token_types=NUM_TOKEN_TYPES,
            max_owner_id=MAX_OWNER_ID,
            max_role_id=MAX_ROLE_ID,
            max_zone_id=MAX_ZONE_ID,
            max_order_id=MAX_ORDER_ID,
            entity_hash_buckets=ENTITY_HASH_BUCKETS,
            use_internal_numeric_proj=False,
            use_internal_text_proj=False,
        )

        self.world_relation_bias = RelationBias(n_heads=self.n_heads, **_RELATION_BIAS_KW)
        self.local_relation_bias = RelationBias(n_heads=self.n_heads, **_RELATION_BIAS_KW)
        self.query_local_relation_bias = RelationBias(n_heads=self.n_heads, **_RELATION_BIAS_KW)
        self.candidate_set_relation_bias = RelationBias(n_heads=self.n_heads, **_RELATION_BIAS_KW)
        self.world_bank_relation_bias = nn.ModuleList(
            [RelationBias(n_heads=self.n_heads, **_RELATION_BIAS_KW) for _ in WORLD_BANK_NAMES]
        )

        self.world_blocks = nn.ModuleList(
            [TransformerEncoderBlock(self.d_model, self.n_heads, self.ffn_dim, dropout=dropout) for _ in range(self.world_layers)]
        )
        self.local_blocks = nn.ModuleList(
            [TransformerEncoderBlock(self.d_model, self.n_heads, self.ffn_dim, dropout=dropout) for _ in range(self.local_layers)]
        )
        self.query_local_bridge = CrossAttentionBlock(self.d_model, self.n_heads, self.ffn_dim, dropout=dropout)
        self.decoder_self_blocks = nn.ModuleList(
            [TransformerEncoderBlock(self.d_model, self.n_heads, self.ffn_dim, dropout=dropout) for _ in range(self.decoder_layers)]
        )
        self.world_bank_cross_blocks = nn.ModuleList(
            [
                nn.ModuleList(
                    [CrossAttentionBlock(self.d_model, self.n_heads, self.ffn_dim, dropout=dropout) for _ in WORLD_BANK_NAMES]
                )
                for _ in range(self.decoder_layers)
            ]
        )
        self.candidate_set_blocks = nn.ModuleList(
            [TransformerEncoderBlock(self.d_model, self.n_heads, self.ffn_dim, dropout=dropout) for _ in range(self.candidate_set_layers)]
        )
        self.world_pool = EntityPooling(self.d_model, self.n_heads, dropout=dropout)
        self.local_pool = EntityPooling(self.d_model, self.n_heads, dropout=dropout)
        self.world_bank_poolers = nn.ModuleList(
            [EntityPooling(self.d_model, self.n_heads, dropout=dropout) for _ in WORLD_BANK_NAMES]
        )
        self.world_bank_router_q = nn.Sequential(nn.LayerNorm(self.d_model), nn.Linear(self.d_model, self.d_model))
        self.world_bank_router_k = nn.Sequential(nn.LayerNorm(self.d_model), nn.Linear(self.d_model, self.d_model))
        self.world_bank_router_bias = nn.Parameter(torch.zeros(len(WORLD_BANK_NAMES)))

        self.register_buffer(
            "_slot_bank_ids",
            torch.as_tensor(build_memory_slot_bank_ids(self.num_memory_slots), dtype=torch.long),
            persistent=False,
        )
        self.memory_seed = nn.Parameter(torch.randn(1, self.num_memory_slots, self.d_model) * 0.02)
        self.memory_slot_bank_embedding = nn.Embedding(self.num_slot_banks, self.d_model)
        self.memory_bank_summary_proj = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
        )
        self.memory_from_world_banks = nn.ModuleList(
            [CrossAttentionBlock(self.d_model, self.n_heads, self.ffn_dim, dropout=dropout) for _ in WORLD_BANK_NAMES]
        )
        self.memory_from_world = CrossAttentionBlock(self.d_model, self.n_heads, self.ffn_dim, dropout=dropout)
        self.memory_from_candidates = CrossAttentionBlock(self.d_model, self.n_heads, self.ffn_dim, dropout=dropout)
        self.memory_blocks = nn.ModuleList(
            [TransformerEncoderBlock(self.d_model, self.n_heads, self.ffn_dim, dropout=dropout) for _ in range(2)]
        )
        self.memory_norm = nn.LayerNorm(self.d_model)
        self.world_to_memory_gate = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.num_memory_slots),
        )
        self.domain_head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, max(self.d_model // 2, NUM_DOMAINS * 2)),
            nn.GELU(),
            nn.Linear(max(self.d_model // 2, NUM_DOMAINS * 2), NUM_DOMAINS),
        )
        self.action_out_proj = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.action_embed_dim),
        )

        for bank_name in WORLD_BANK_NAMES:
            self.register_buffer(
                f"_bank_role_ids_{bank_name}",
                torch.as_tensor(WORLD_BANK_ROLE_IDS.get(bank_name, ()), dtype=torch.long),
                persistent=False,
            )
            self.register_buffer(
                f"_bank_zone_ids_{bank_name}",
                torch.as_tensor(WORLD_BANK_ZONE_IDS.get(bank_name, ()), dtype=torch.long),
                persistent=False,
            )

    def _project_shared_modal_trunk_with_reuse(self, trunk, feature_batches, *, output_dim: int):
        flattened_batches = [batch.reshape(-1, batch.shape[-1]) for batch in feature_batches]
        total_rows = sum(batch.shape[0] for batch in flattened_batches)
        if total_rows == 0:
            return [batch.new_zeros(*batch.shape[:-1], output_dim) for batch in feature_batches]
        if self.projection_reuse_max_rows > 0 and total_rows > self.projection_reuse_max_rows:
            return [trunk(batch).reshape(*original.shape[:-1], output_dim) for batch, original in zip(flattened_batches, feature_batches)]
        merged_features = torch.cat(flattened_batches, dim=0)
        unique_features, inverse_indices = torch.unique(merged_features, dim=0, return_inverse=True)
        projected_unique = trunk(unique_features)
        projected_merged = projected_unique.index_select(0, inverse_indices)
        projected_batches = []
        row_offset = 0
        for batch, flattened in zip(feature_batches, flattened_batches):
            row_count = flattened.shape[0]
            projected_batches.append(
                projected_merged[row_offset : row_offset + row_count].reshape(*batch.shape[:-1], output_dim)
            )
            row_offset += row_count
        return projected_batches

    @staticmethod
    def _float_tensor(value: torch.Tensor | object, *, device: torch.device) -> torch.Tensor:
        tensor = torch.as_tensor(value, dtype=torch.float32, device=device)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        return tensor

    @staticmethod
    def _long_tensor(value: torch.Tensor | object, *, device: torch.device) -> torch.Tensor:
        tensor = torch.as_tensor(value, dtype=torch.long, device=device)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        return tensor

    @staticmethod
    def _bool_tensor(value: torch.Tensor | object, *, device: torch.device) -> torch.Tensor:
        tensor = torch.as_tensor(value, dtype=torch.bool, device=device)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        return tensor

    def _build_world_bank_masks(self, world_role_ids, world_zone_ids, world_mask):
        world_mask = torch.as_tensor(world_mask, dtype=torch.bool, device=world_role_ids.device)
        if world_mask.ndim == 1:
            world_mask = world_mask.unsqueeze(0)
        bank_masks = []
        for bank_name in WORLD_BANK_NAMES:
            bank_role_ids = getattr(self, f"_bank_role_ids_{bank_name}")
            bank_zone_ids = getattr(self, f"_bank_zone_ids_{bank_name}")
            role_mask = torch.zeros_like(world_mask)
            zone_mask = torch.zeros_like(world_mask)
            if bank_role_ids.numel():
                role_mask = torch.isin(world_role_ids, bank_role_ids)
            if bank_zone_ids.numel():
                zone_mask = torch.isin(world_zone_ids, bank_zone_ids)
            bank_masks.append((role_mask | zone_mask) & world_mask)
        stacked = torch.stack(bank_masks, dim=1)
        unmatched = world_mask & ~stacked.any(dim=1)
        if unmatched.any():
            stacked = stacked.clone()
            stacked[:, 0, :] = stacked[:, 0, :] | unmatched
        return stacked

    def _compute_world_bank_summaries(self, world_x, world_bank_masks):
        summaries = []
        for bank_index, pooler in enumerate(self.world_bank_poolers):
            summaries.append(pooler(world_x, mask=world_bank_masks[:, bank_index, :]))
        return torch.stack(summaries, dim=1)

    @staticmethod
    def _compute_world_bank_occupancy(world_bank_masks: torch.Tensor, world_mask: torch.Tensor) -> torch.Tensor:
        bank_counts = world_bank_masks.sum(dim=-1).float()
        total_tokens = world_mask.sum(dim=-1, keepdim=True).float().clamp(min=1.0)
        return torch.clamp(bank_counts / total_tokens, min=0.0, max=1.0)

    @staticmethod
    def _compute_world_bank_token_signatures(
        world_bank_masks: torch.Tensor,
        world_type_ids: torch.Tensor,
        world_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Summarise which token families each world bank contains.

        Returns:
            token_presence: [batch, num_banks, num_token_types] binary indicator
            token_distribution: [batch, num_banks, num_token_types] normalized counts
        """

        masked_world_types = world_type_ids.clamp(min=0, max=NUM_TOKEN_TYPES - 1)
        type_one_hot = torch.nn.functional.one_hot(masked_world_types, num_classes=NUM_TOKEN_TYPES).float()
        type_one_hot = type_one_hot * world_mask.unsqueeze(-1).float()
        bank_counts = torch.einsum("bkt,btc->bkc", world_bank_masks.float(), type_one_hot)
        token_presence = (bank_counts > 0.0).float()
        bank_totals = bank_counts.sum(dim=-1, keepdim=True)
        safe_totals = bank_totals.clamp(min=1.0)
        token_distribution = bank_counts / safe_totals
        token_distribution = torch.where(bank_totals > 0.0, token_distribution, torch.zeros_like(token_distribution))
        return token_presence, token_distribution

    def _compute_world_bank_token_slots(
        self,
        world_x: torch.Tensor,
        world_bank_masks: torch.Tensor,
        world_type_ids: torch.Tensor,
        world_zone_ids: torch.Tensor,
        world_entity_ids: torch.Tensor,
        world_order_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract a stable top-k token view per bank for finer future-world reconstruction."""

        batch_size, world_token_count, _ = world_x.shape
        device = world_x.device
        select_k = min(self.bank_token_slots, world_token_count)
        token_positions = torch.arange(world_token_count, device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
        normalized_order = torch.where(
            world_order_ids > 0,
            world_order_ids,
            (MAX_ORDER_ID + 1) + token_positions,
        )
        rank = normalized_order * (world_token_count + 1) + token_positions
        invalid_rank = torch.full_like(rank, (MAX_ORDER_ID + world_token_count + 2) * (world_token_count + 1))

        selected_states: list[torch.Tensor] = []
        selected_masks: list[torch.Tensor] = []
        selected_type_ids: list[torch.Tensor] = []
        selected_zone_ids: list[torch.Tensor] = []
        selected_entity_ids: list[torch.Tensor] = []
        selected_order_ids: list[torch.Tensor] = []

        for bank_index in range(len(WORLD_BANK_NAMES)):
            bank_mask = world_bank_masks[:, bank_index, :]
            bank_rank = torch.where(bank_mask, rank, invalid_rank)
            topk_idx = bank_rank.topk(select_k, dim=-1, largest=False).indices
            gathered_mask = bank_mask.gather(1, topk_idx)
            gathered_states = world_x.gather(1, topk_idx.unsqueeze(-1).expand(-1, -1, self.d_model))
            gathered_states = gathered_states * gathered_mask.unsqueeze(-1).float()
            gathered_type_ids = world_type_ids.gather(1, topk_idx)
            gathered_type_ids = torch.where(gathered_mask, gathered_type_ids, torch.zeros_like(gathered_type_ids))
            gathered_zone_ids = world_zone_ids.gather(1, topk_idx)
            gathered_zone_ids = torch.where(gathered_mask, gathered_zone_ids, torch.zeros_like(gathered_zone_ids))
            gathered_entity_ids = world_entity_ids.gather(1, topk_idx)
            gathered_entity_ids = torch.where(gathered_mask, gathered_entity_ids, torch.zeros_like(gathered_entity_ids))
            gathered_order_ids = world_order_ids.gather(1, topk_idx)
            gathered_order_ids = torch.where(gathered_mask, gathered_order_ids, torch.zeros_like(gathered_order_ids))
            if select_k < self.bank_token_slots:
                pad_slots = self.bank_token_slots - select_k
                gathered_states = torch.cat(
                    [gathered_states, gathered_states.new_zeros((batch_size, pad_slots, self.d_model))],
                    dim=1,
                )
                gathered_mask = torch.cat(
                    [gathered_mask, torch.zeros((batch_size, pad_slots), dtype=torch.bool, device=device)],
                    dim=1,
                )
                gathered_type_ids = torch.cat(
                    [gathered_type_ids, torch.zeros((batch_size, pad_slots), dtype=torch.long, device=device)],
                    dim=1,
                )
                gathered_zone_ids = torch.cat(
                    [gathered_zone_ids, torch.zeros((batch_size, pad_slots), dtype=torch.long, device=device)],
                    dim=1,
                )
                gathered_entity_ids = torch.cat(
                    [gathered_entity_ids, torch.zeros((batch_size, pad_slots), dtype=torch.long, device=device)],
                    dim=1,
                )
                gathered_order_ids = torch.cat(
                    [gathered_order_ids, torch.zeros((batch_size, pad_slots), dtype=torch.long, device=device)],
                    dim=1,
                )
            selected_states.append(gathered_states)
            selected_masks.append(gathered_mask)
            selected_type_ids.append(gathered_type_ids)
            selected_zone_ids.append(gathered_zone_ids)
            selected_entity_ids.append(gathered_entity_ids)
            selected_order_ids.append(gathered_order_ids)

        return (
            torch.stack(selected_states, dim=1),
            torch.stack(selected_masks, dim=1),
            torch.stack(selected_type_ids, dim=1),
            torch.stack(selected_zone_ids, dim=1),
            torch.stack(selected_entity_ids, dim=1),
            torch.stack(selected_order_ids, dim=1),
        )

    def _slot_bank_id_tensor(self, device: torch.device) -> torch.Tensor:
        return self._slot_bank_ids.to(device=device)

    def _slot_bank_mask(self, slot_bank_ids: torch.Tensor, bank_index: int, batch_size: int) -> torch.Tensor:
        return slot_bank_ids.eq(int(bank_index)).unsqueeze(0).expand(batch_size, -1)

    def _slot_bank_summary_context(
        self,
        slot_bank_ids: torch.Tensor,
        *,
        world_bank_summaries: torch.Tensor,
        world_pool: torch.Tensor,
    ) -> torch.Tensor:
        summary_table = torch.cat([world_bank_summaries, world_pool.unsqueeze(1)], dim=1)
        return summary_table[:, slot_bank_ids, :]

    def _build_world_bank_biases(
        self,
        *,
        candidate_query_type_ids,
        candidate_query_role_ids,
        candidate_query_owner_ids,
        candidate_query_entity_ids,
        candidate_query_zone_ids,
        candidate_query_order_ids,
        candidate_query_target_owner_ids,
        candidate_query_target_entity_ids,
        world_type_ids,
        world_role_ids,
        world_owner_ids,
        world_entity_ids,
        world_zone_ids,
        world_order_ids,
    ):
        return [
            relation_bias(
                candidate_query_type_ids,
                world_type_ids,
                candidate_query_owner_ids,
                world_owner_ids,
                candidate_query_entity_ids,
                world_entity_ids,
                candidate_query_role_ids,
                world_role_ids,
                candidate_query_zone_ids,
                world_zone_ids,
                candidate_query_order_ids,
                world_order_ids,
                candidate_query_target_owner_ids,
                candidate_query_target_entity_ids,
            )
            for relation_bias in self.world_bank_relation_bias
        ]

    def _compute_world_bank_routing(self, candidate_x, candidate_mask, world_bank_summaries, world_bank_masks):
        bank_available = world_bank_masks.any(dim=-1)
        router_q = self.world_bank_router_q(candidate_x)
        router_k = self.world_bank_router_k(world_bank_summaries)
        router_logits = torch.einsum("bad,bkd->bak", router_q, router_k) / math.sqrt(float(self.d_model))
        router_logits = router_logits + self.world_bank_router_bias.view(1, 1, -1)
        router_logits = router_logits.masked_fill(~bank_available.unsqueeze(1), -1e4)

        top_k = min(self.world_bank_top_k, len(WORLD_BANK_NAMES))
        if top_k < len(WORLD_BANK_NAMES):
            top_indices = router_logits.topk(top_k, dim=-1).indices
            selected = torch.zeros_like(router_logits, dtype=torch.bool)
            selected.scatter_(-1, top_indices, True)
            selected = selected & bank_available.unsqueeze(1)
        else:
            selected = bank_available.unsqueeze(1).expand_as(router_logits)

        routed_logits = router_logits.masked_fill(~selected, -1e4)
        bank_weights = torch.softmax(routed_logits, dim=-1)
        bank_weights = torch.nan_to_num(bank_weights, nan=0.0, posinf=0.0, neginf=0.0)
        has_selected_bank = selected.any(dim=-1, keepdim=True)
        bank_weights = torch.where(has_selected_bank, bank_weights, torch.zeros_like(bank_weights))
        bank_weights = bank_weights * candidate_mask.unsqueeze(-1).float()
        selected = selected & candidate_mask.unsqueeze(-1)
        return bank_weights, selected

    def _apply_banked_world_cross_attention(
        self,
        *,
        layer_index: int,
        candidate_x: torch.Tensor,
        candidate_mask: torch.Tensor,
        world_x: torch.Tensor,
        world_bank_masks: torch.Tensor,
        world_bank_summaries: torch.Tensor,
        world_bank_biases: list[torch.Tensor],
    ) -> torch.Tensor:
        bank_weights, bank_selected = self._compute_world_bank_routing(
            candidate_x,
            candidate_mask,
            world_bank_summaries,
            world_bank_masks,
        )
        fused_delta = torch.zeros_like(candidate_x)
        for bank_index, _bank_name in enumerate(WORLD_BANK_NAMES):
            bank_query_mask = bank_selected[..., bank_index]
            if not bool(bank_query_mask.any().item()):
                continue
            bank_output = _run_cross_block(
                self.activation_checkpointing,
                self.world_bank_cross_blocks[layer_index][bank_index],
                candidate_x,
                world_x,
                query_mask=bank_query_mask,
                memory_mask=world_bank_masks[:, bank_index, :],
                attn_bias=world_bank_biases[bank_index],
            )
            bank_output = torch.where(bank_query_mask.unsqueeze(-1), bank_output, candidate_x)
            fused_delta = fused_delta + bank_weights[..., bank_index].unsqueeze(-1) * (bank_output - candidate_x)
        return candidate_x + fused_delta

    def _memory_slots(
        self,
        *,
        world_x: torch.Tensor,
        world_mask: torch.Tensor,
        world_bank_masks: torch.Tensor,
        world_bank_summaries: torch.Tensor,
        candidate_x: torch.Tensor,
        candidate_mask: torch.Tensor,
        world_pool: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = world_x.shape[0]
        slot_bank_ids = self._slot_bank_id_tensor(world_x.device)
        memory = self.memory_seed.expand(batch_size, -1, -1)
        memory = memory + self.memory_slot_bank_embedding(slot_bank_ids).unsqueeze(0)
        slot_mask = torch.ones((batch_size, self.num_memory_slots), dtype=torch.bool, device=world_x.device)

        gate_logits = self.world_to_memory_gate(world_pool)
        gate = torch.sigmoid(gate_logits).unsqueeze(-1)
        slot_bank_context = self.memory_bank_summary_proj(
            self._slot_bank_summary_context(
                slot_bank_ids,
                world_bank_summaries=world_bank_summaries,
                world_pool=world_pool,
            )
        )
        memory = memory + gate * slot_bank_context
        for bank_index, bank_cross in enumerate(self.memory_from_world_banks):
            bank_slot_mask = self._slot_bank_mask(slot_bank_ids, bank_index, batch_size)
            if not bool(bank_slot_mask.any().item()):
                continue
            bank_output = _run_cross_block(
                self.activation_checkpointing,
                bank_cross,
                memory,
                world_x,
                query_mask=bank_slot_mask,
                memory_mask=world_bank_masks[:, bank_index, :],
            )
            memory = torch.where(bank_slot_mask.unsqueeze(-1), bank_output, memory)
        global_slot_mask = self._slot_bank_mask(slot_bank_ids, self.global_memory_bank_index, batch_size)
        if bool(global_slot_mask.any().item()):
            global_output = _run_cross_block(
                self.activation_checkpointing,
                self.memory_from_world,
                memory,
                world_x,
                query_mask=global_slot_mask,
                memory_mask=world_mask,
            )
            memory = torch.where(global_slot_mask.unsqueeze(-1), global_output, memory)
        memory = _run_cross_block(
            self.activation_checkpointing,
            self.memory_from_candidates,
            memory,
            candidate_x,
            query_mask=slot_mask,
            memory_mask=candidate_mask,
        )
        for block in self.memory_blocks:
            memory = _run_encoder_block(self.activation_checkpointing, block, memory, mask=slot_mask)
        return self.memory_norm(memory)

    def _flatten_hidden(self, memory_slots: torch.Tensor) -> torch.Tensor:
        return memory_slots.reshape(memory_slots.shape[0], -1)

    def forward(self, obs: dict[str, torch.Tensor]) -> TokenMemoryEncoderOutput:
        device = None
        for value in obs.values():
            if isinstance(value, torch.Tensor):
                device = value.device
                break
        if device is None:
            device = torch.device("cpu")

        world_tokens = self._float_tensor(obs["world_tokens"], device=device)
        world_mask = self._bool_tensor(obs["world_token_mask"], device=device)
        world_type_ids = self._long_tensor(obs["world_token_type_ids"], device=device)
        world_role_ids = self._long_tensor(obs["world_token_role_ids"], device=device)
        world_owner_ids = self._long_tensor(obs["world_entity_owner_ids"], device=device)
        world_entity_ids = self._long_tensor(obs["world_token_entity_ids"], device=device)
        world_zone_ids = self._long_tensor(obs["world_token_zone_ids"], device=device)
        world_order_ids = self._long_tensor(obs["world_token_order_ids"], device=device)
        world_type_ids = world_type_ids.clamp(min=0, max=NUM_TOKEN_TYPES - 1)
        world_role_ids = world_role_ids.clamp(min=0, max=MAX_ROLE_ID)
        world_owner_ids = world_owner_ids.clamp(min=0, max=MAX_OWNER_ID)
        world_entity_ids = world_entity_ids.clamp(min=0, max=ENTITY_HASH_BUCKETS - 1)
        world_zone_ids = world_zone_ids.clamp(min=0, max=MAX_ZONE_ID)
        world_order_ids = world_order_ids.clamp(min=0, max=MAX_ORDER_ID)

        candidate_query_tokens = self._float_tensor(obs["candidate_query_tokens"], device=device)
        candidate_query_type_ids = self._long_tensor(obs["candidate_query_type_ids"], device=device)
        candidate_query_role_ids = self._long_tensor(obs["candidate_query_role_ids"], device=device)
        candidate_query_owner_ids = self._long_tensor(obs["candidate_query_owner_ids"], device=device)
        candidate_query_entity_ids = self._long_tensor(obs["candidate_query_entity_ids"], device=device)
        candidate_query_zone_ids = self._long_tensor(obs["candidate_query_zone_ids"], device=device)
        candidate_query_order_ids = self._long_tensor(obs["candidate_query_order_ids"], device=device)
        candidate_query_target_owner_ids = self._long_tensor(obs["candidate_query_target_owner_ids"], device=device)
        candidate_query_target_entity_ids = self._long_tensor(obs["candidate_query_target_entity_ids"], device=device)
        candidate_query_type_ids = candidate_query_type_ids.clamp(min=0, max=NUM_TOKEN_TYPES - 1)
        candidate_query_role_ids = candidate_query_role_ids.clamp(min=0, max=MAX_ROLE_ID)
        candidate_query_owner_ids = candidate_query_owner_ids.clamp(min=0, max=MAX_OWNER_ID)
        candidate_query_entity_ids = candidate_query_entity_ids.clamp(min=0, max=ENTITY_HASH_BUCKETS - 1)
        candidate_query_zone_ids = candidate_query_zone_ids.clamp(min=0, max=MAX_ZONE_ID)
        candidate_query_order_ids = candidate_query_order_ids.clamp(min=0, max=MAX_ORDER_ID)
        candidate_query_target_owner_ids = candidate_query_target_owner_ids.clamp(min=0, max=MAX_OWNER_ID)
        candidate_query_target_entity_ids = candidate_query_target_entity_ids.clamp(min=0, max=ENTITY_HASH_BUCKETS - 1)
        candidate_mask = self._bool_tensor(obs["action_mask"], device=device)

        candidate_local_tokens = self._float_tensor(obs["candidate_local_tokens"], device=device)
        candidate_local_masks = self._bool_tensor(obs["candidate_local_masks"], device=device)
        candidate_local_type_ids = self._long_tensor(obs["candidate_local_type_ids"], device=device)
        candidate_local_role_ids = self._long_tensor(obs["candidate_local_role_ids"], device=device)
        candidate_local_owner_ids = self._long_tensor(obs["candidate_local_owner_ids"], device=device)
        candidate_local_entity_ids = self._long_tensor(obs["candidate_local_entity_ids"], device=device)
        candidate_local_zone_ids = self._long_tensor(obs["candidate_local_zone_ids"], device=device)
        candidate_local_order_ids = self._long_tensor(obs["candidate_local_order_ids"], device=device)
        candidate_local_type_ids = candidate_local_type_ids.clamp(min=0, max=NUM_TOKEN_TYPES - 1)
        candidate_local_role_ids = candidate_local_role_ids.clamp(min=0, max=MAX_ROLE_ID)
        candidate_local_owner_ids = candidate_local_owner_ids.clamp(min=0, max=MAX_OWNER_ID)
        candidate_local_entity_ids = candidate_local_entity_ids.clamp(min=0, max=ENTITY_HASH_BUCKETS - 1)
        candidate_local_zone_ids = candidate_local_zone_ids.clamp(min=0, max=MAX_ZONE_ID)
        candidate_local_order_ids = candidate_local_order_ids.clamp(min=0, max=MAX_ORDER_ID)

        batch_size, action_count, local_count, _ = candidate_local_tokens.shape
        flat_local_tokens = candidate_local_tokens.reshape(batch_size * action_count, local_count, TOKEN_FEAT_DIM)
        flat_local_masks = candidate_local_masks.reshape(batch_size * action_count, local_count)
        flat_local_type_ids = candidate_local_type_ids.reshape(batch_size * action_count, local_count)
        flat_local_role_ids = candidate_local_role_ids.reshape(batch_size * action_count, local_count)
        flat_local_owner_ids = candidate_local_owner_ids.reshape(batch_size * action_count, local_count)
        flat_local_entity_ids = candidate_local_entity_ids.reshape(batch_size * action_count, local_count)
        flat_local_zone_ids = candidate_local_zone_ids.reshape(batch_size * action_count, local_count)
        flat_local_order_ids = candidate_local_order_ids.reshape(batch_size * action_count, local_count)

        numeric_slice = slice(0, TOKEN_NUMERIC_DIM)
        text_slice = slice(TOKEN_NUMERIC_DIM, TOKEN_NUMERIC_DIM + TOKEN_TEXT_DIM)
        world_numeric, query_numeric, flat_local_numeric = self._project_shared_modal_trunk_with_reuse(
            self.shared_numeric_trunk,
            [
                world_tokens[..., numeric_slice],
                candidate_query_tokens[..., numeric_slice],
                flat_local_tokens[..., numeric_slice],
            ],
            output_dim=self.shared_numeric_width,
        )
        world_text, query_text, flat_local_text = self._project_shared_modal_trunk_with_reuse(
            self.shared_text_trunk,
            [
                world_tokens[..., text_slice],
                candidate_query_tokens[..., text_slice],
                flat_local_tokens[..., text_slice],
            ],
            output_dim=self.shared_text_width,
        )

        world_x = self.world_embedder(
            world_tokens,
            world_type_ids,
            world_role_ids,
            world_owner_ids,
            world_entity_ids,
            world_zone_ids,
            world_order_ids,
            projected_numeric=world_numeric,
            projected_text=world_text,
        )
        world_bias = self.world_relation_bias(
            world_type_ids,
            world_type_ids,
            world_owner_ids,
            world_owner_ids,
            world_entity_ids,
            world_entity_ids,
            world_role_ids,
            world_role_ids,
            world_zone_ids,
            world_zone_ids,
            world_order_ids,
            world_order_ids,
        )
        for block in self.world_blocks:
            world_x = _run_encoder_block(
                self.activation_checkpointing,
                block,
                world_x,
                mask=world_mask,
                attn_bias=world_bias,
            )
        world_pool = self.world_pool(world_x, mask=world_mask)

        query_x = self.query_embedder(
            candidate_query_tokens,
            candidate_query_type_ids,
            candidate_query_role_ids,
            candidate_query_owner_ids,
            candidate_query_entity_ids,
            candidate_query_zone_ids,
            candidate_query_order_ids,
            target_owner_ids=candidate_query_target_owner_ids,
            target_entity_ids=candidate_query_target_entity_ids,
            projected_numeric=query_numeric,
            projected_text=query_text,
        )
        local_x = self.local_embedder(
            flat_local_tokens,
            flat_local_type_ids,
            flat_local_role_ids,
            flat_local_owner_ids,
            flat_local_entity_ids,
            flat_local_zone_ids,
            flat_local_order_ids,
            projected_numeric=flat_local_numeric,
            projected_text=flat_local_text,
        )
        local_bias = self.local_relation_bias(
            flat_local_type_ids,
            flat_local_type_ids,
            flat_local_owner_ids,
            flat_local_owner_ids,
            flat_local_entity_ids,
            flat_local_entity_ids,
            flat_local_role_ids,
            flat_local_role_ids,
            flat_local_zone_ids,
            flat_local_zone_ids,
            flat_local_order_ids,
            flat_local_order_ids,
        )
        for block in self.local_blocks:
            local_x = _run_encoder_block(
                self.activation_checkpointing,
                block,
                local_x,
                mask=flat_local_masks,
                attn_bias=local_bias,
            )
        local_pool = self.local_pool(local_x, mask=flat_local_masks).reshape(batch_size, action_count, self.d_model)

        flat_query_x = query_x.reshape(batch_size * action_count, 1, self.d_model)
        flat_query_masks = candidate_mask.reshape(batch_size * action_count, 1)
        flat_query_type_ids = candidate_query_type_ids.reshape(batch_size * action_count, 1)
        flat_query_role_ids = candidate_query_role_ids.reshape(batch_size * action_count, 1)
        flat_query_owner_ids = candidate_query_owner_ids.reshape(batch_size * action_count, 1)
        flat_query_entity_ids = candidate_query_entity_ids.reshape(batch_size * action_count, 1)
        flat_query_zone_ids = candidate_query_zone_ids.reshape(batch_size * action_count, 1)
        flat_query_order_ids = candidate_query_order_ids.reshape(batch_size * action_count, 1)
        flat_query_target_owner_ids = candidate_query_target_owner_ids.reshape(batch_size * action_count, 1)
        flat_query_target_entity_ids = candidate_query_target_entity_ids.reshape(batch_size * action_count, 1)
        query_local_bias = self.query_local_relation_bias(
            flat_query_type_ids,
            flat_local_type_ids,
            flat_query_owner_ids,
            flat_local_owner_ids,
            flat_query_entity_ids,
            flat_local_entity_ids,
            flat_query_role_ids,
            flat_local_role_ids,
            flat_query_zone_ids,
            flat_local_zone_ids,
            flat_query_order_ids,
            flat_local_order_ids,
            flat_query_target_owner_ids,
            flat_query_target_entity_ids,
        )
        bridged_query_x = _run_cross_block(
            self.activation_checkpointing,
            self.query_local_bridge,
            flat_query_x,
            local_x,
            query_mask=flat_query_masks,
            memory_mask=flat_local_masks,
            attn_bias=query_local_bias,
        )
        has_local_context = flat_local_masks.any(dim=1, keepdim=True).unsqueeze(-1)
        bridged_query_x = torch.where(has_local_context, bridged_query_x, flat_query_x)
        candidate_x = bridged_query_x.reshape(batch_size, action_count, self.d_model) + local_pool

        set_bias = self.candidate_set_relation_bias(
            candidate_query_type_ids,
            candidate_query_type_ids,
            candidate_query_owner_ids,
            candidate_query_owner_ids,
            candidate_query_entity_ids,
            candidate_query_entity_ids,
            candidate_query_role_ids,
            candidate_query_role_ids,
            candidate_query_zone_ids,
            candidate_query_zone_ids,
            candidate_query_order_ids,
            candidate_query_order_ids,
        )
        world_bank_masks = self._build_world_bank_masks(world_role_ids, world_zone_ids, world_mask)
        world_bank_summaries = self._compute_world_bank_summaries(world_x, world_bank_masks)
        world_bank_biases = self._build_world_bank_biases(
            candidate_query_type_ids=candidate_query_type_ids,
            candidate_query_role_ids=candidate_query_role_ids,
            candidate_query_owner_ids=candidate_query_owner_ids,
            candidate_query_entity_ids=candidate_query_entity_ids,
            candidate_query_zone_ids=candidate_query_zone_ids,
            candidate_query_order_ids=candidate_query_order_ids,
            candidate_query_target_owner_ids=candidate_query_target_owner_ids,
            candidate_query_target_entity_ids=candidate_query_target_entity_ids,
            world_type_ids=world_type_ids,
            world_role_ids=world_role_ids,
            world_owner_ids=world_owner_ids,
            world_entity_ids=world_entity_ids,
            world_zone_ids=world_zone_ids,
            world_order_ids=world_order_ids,
        )
        for layer_index in range(self.decoder_layers):
            candidate_x = _run_encoder_block(
                self.activation_checkpointing,
                self.decoder_self_blocks[layer_index],
                candidate_x,
                mask=candidate_mask,
                attn_bias=set_bias,
            )
            def _apply_bank_cross(
                candidate_x_: torch.Tensor,
                world_x_: torch.Tensor,
                layer_index_: int = layer_index,
            ) -> torch.Tensor:
                return self._apply_banked_world_cross_attention(
                    layer_index=layer_index_,
                    candidate_x=candidate_x_,
                    candidate_mask=candidate_mask,
                    world_x=world_x_,
                    world_bank_masks=world_bank_masks,
                    world_bank_summaries=world_bank_summaries,
                    world_bank_biases=world_bank_biases,
                )

            candidate_x = _run_function_with_checkpoint(
                self.activation_checkpointing,
                _apply_bank_cross,
                candidate_x,
                world_x,
            )
        for block in self.candidate_set_blocks:
            candidate_x = _run_encoder_block(
                self.activation_checkpointing,
                block,
                candidate_x,
                mask=candidate_mask,
                attn_bias=set_bias,
            )

        world_bank_token_presence, world_bank_token_distribution = self._compute_world_bank_token_signatures(
            world_bank_masks=world_bank_masks,
            world_type_ids=world_type_ids,
            world_mask=world_mask,
        )
        (
            world_bank_token_slot_states,
            world_bank_token_slot_mask,
            world_bank_token_slot_type_ids,
            world_bank_token_slot_zone_ids,
            world_bank_token_slot_entity_ids,
            world_bank_token_slot_order_ids,
        ) = self._compute_world_bank_token_slots(
            world_x=world_x,
            world_bank_masks=world_bank_masks,
            world_type_ids=world_type_ids,
            world_zone_ids=world_zone_ids,
            world_entity_ids=world_entity_ids,
            world_order_ids=world_order_ids,
        )

        memory_slots = self._memory_slots(
            world_x=world_x,
            world_mask=world_mask,
            world_bank_masks=world_bank_masks,
            world_bank_summaries=world_bank_summaries,
            candidate_x=candidate_x,
            candidate_mask=candidate_mask,
            world_pool=world_pool,
        )
        hidden_state = self._flatten_hidden(memory_slots)
        action_embeddings = self.action_out_proj(candidate_x) * candidate_mask.unsqueeze(-1).float()
        decision_domain = infer_token_decision_domain(obs, device=device)
        if decision_domain is None:
            domain_logits = self.domain_head(world_pool)
            decision_domain = torch.softmax(domain_logits, dim=-1)
        return TokenMemoryEncoderOutput(
            hidden_state=hidden_state,
            action_embeddings=action_embeddings,
            world_pool=world_pool,
            world_bank_summaries=world_bank_summaries,
            world_bank_occupancy=self._compute_world_bank_occupancy(world_bank_masks, world_mask),
            world_bank_token_presence=world_bank_token_presence,
            world_bank_token_distribution=world_bank_token_distribution,
            world_bank_token_slot_states=world_bank_token_slot_states,
            world_bank_token_slot_mask=world_bank_token_slot_mask,
            world_bank_token_slot_type_ids=world_bank_token_slot_type_ids,
            world_bank_token_slot_zone_ids=world_bank_token_slot_zone_ids,
            world_bank_token_slot_entity_ids=world_bank_token_slot_entity_ids,
            world_bank_token_slot_order_ids=world_bank_token_slot_order_ids,
            decision_domain=decision_domain,
        )


class TokenDynamicsNetwork(nn.Module):
    """Action-conditioned latent slot transition for token-memory MuZero."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        action_embed_dim: int,
        d_model: int,
        num_memory_slots: int,
        support_size: int = 25,
        num_transition_layers: int = 2,
        dropout: float = 0.0,
        activation_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.action_embed_dim = int(action_embed_dim)
        self.d_model = int(d_model)
        self.num_memory_slots = int(num_memory_slots)
        self.support_size = int(support_size)
        self.num_bins = 2 * self.support_size + 1
        self.activation_checkpointing = bool(activation_checkpointing)

        if self.hidden_dim != self.d_model * self.num_memory_slots:
            raise ValueError(
                f"TokenDynamicsNetwork expects hidden_dim == d_model * num_memory_slots, "
                f"got {self.hidden_dim} vs {self.d_model} * {self.num_memory_slots}."
            )

        self.num_slot_banks = len(MEMORY_BANK_NAMES)
        self.register_buffer(
            "_slot_bank_ids",
            torch.as_tensor(build_memory_slot_bank_ids(self.num_memory_slots), dtype=torch.long),
            persistent=False,
        )
        self.slot_index_embedding = nn.Embedding(self.num_memory_slots, self.d_model)
        self.slot_bank_embedding = nn.Embedding(self.num_slot_banks, self.d_model)
        self.transition_type_embedding = nn.Parameter(torch.randn(1, 1, self.d_model) * 0.02)
        self.action_token_proj = nn.Sequential(
            nn.Linear(self.action_embed_dim, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
        )
        self.action_gate = nn.Sequential(
            nn.Linear(self.action_embed_dim, self.d_model),
            nn.Sigmoid(),
        )
        self.slot_action_gate = nn.Sequential(
            nn.LayerNorm(self.d_model + self.action_embed_dim),
            nn.Linear(self.d_model + self.action_embed_dim, self.d_model),
            nn.Sigmoid(),
        )
        self.slot_action_write = nn.Sequential(
            nn.LayerNorm(self.d_model + self.action_embed_dim),
            nn.Linear(self.d_model + self.action_embed_dim, 1),
            nn.Sigmoid(),
        )
        self.slot_action_cross = CrossAttentionBlock(self.d_model, max(1, min(4, self.d_model // 32)), self.d_model * 4, dropout=dropout)
        self.transition_blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    self.d_model,
                    max(1, min(4, self.d_model // 32)),
                    self.d_model * 4,
                    dropout=dropout,
                )
                for _ in range(max(int(num_transition_layers), 1))
            ]
        )
        self.slot_pool = EntityPooling(self.d_model, max(1, min(4, self.d_model // 32)), dropout=dropout)
        reward_hidden_dim = max(self.d_model * 2, self.action_embed_dim * 2)
        self.reward_net = nn.Sequential(
            nn.LayerNorm(self.d_model + self.action_embed_dim),
            nn.Linear(self.d_model + self.action_embed_dim, reward_hidden_dim),
            nn.GELU(),
            nn.Linear(reward_hidden_dim, self.num_bins),
        )
        self.reward_component_net = nn.Sequential(
            nn.LayerNorm(self.d_model + self.action_embed_dim),
            nn.Linear(self.d_model + self.action_embed_dim, reward_hidden_dim),
            nn.GELU(),
            nn.Linear(reward_hidden_dim, self.num_bins * NUM_OBJECTIVE_HEADS),
        )
        self.surprise_net = nn.Sequential(
            nn.LayerNorm(self.d_model + self.action_embed_dim),
            nn.Linear(self.d_model + self.action_embed_dim, reward_hidden_dim),
            nn.GELU(),
            nn.Linear(reward_hidden_dim, 1),
        )
        self.output_norm = nn.LayerNorm(self.d_model)

    def _unflatten(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return hidden_state.reshape(hidden_state.shape[0], self.num_memory_slots, self.d_model)

    def _flatten(self, slots: torch.Tensor) -> torch.Tensor:
        return slots.reshape(slots.shape[0], -1)

    def _slot_bank_id_tensor(self, device: torch.device) -> torch.Tensor:
        return self._slot_bank_ids.to(device=device)

    def forward(self, hidden_state: torch.Tensor, action_embedding: torch.Tensor):
        from muzero.sts2_env.muzero_model import support_to_scalar, support_tensor_to_scalar

        slots = self._unflatten(hidden_state)
        batch_size = slots.shape[0]
        slot_positions = torch.arange(self.num_memory_slots, device=slots.device, dtype=torch.long)
        slot_bank_ids = self._slot_bank_id_tensor(slots.device)
        slots = (
            slots
            + self.slot_index_embedding(slot_positions).unsqueeze(0)
            + self.slot_bank_embedding(slot_bank_ids).unsqueeze(0)
            + self.transition_type_embedding
        )
        slot_mask = torch.ones((batch_size, self.num_memory_slots), dtype=torch.bool, device=slots.device)
        action_token = self.action_token_proj(action_embedding).unsqueeze(1)
        action_mask = torch.ones((batch_size, 1), dtype=torch.bool, device=slots.device)
        gate = self.action_gate(action_embedding).unsqueeze(1)
        slot_gate_input = torch.cat(
            [
                slots,
                action_embedding.unsqueeze(1).expand(-1, self.num_memory_slots, -1),
            ],
            dim=-1,
        )
        slot_gate = self.slot_action_gate(slot_gate_input)
        slot_write = self.slot_action_write(slot_gate_input)

        slots = slots + slot_write * slot_gate * (gate * action_token)
        slots = _run_cross_block(
            self.activation_checkpointing,
            self.slot_action_cross,
            slots,
            action_token,
            query_mask=slot_mask,
            memory_mask=action_mask,
        )
        for block in self.transition_blocks:
            slots = _run_encoder_block(self.activation_checkpointing, block, slots, mask=slot_mask)
        slots = self.output_norm(slots)

        pooled = self.slot_pool(slots, mask=slot_mask)
        reward_input = torch.cat([pooled, action_embedding], dim=-1)
        reward_logits = self.reward_net(reward_input)
        reward = support_to_scalar(reward_logits, self.support_size)
        reward_component_logits = self.reward_component_net(reward_input).view(
            reward_input.shape[0],
            NUM_OBJECTIVE_HEADS,
            self.num_bins,
        )
        reward_components = support_tensor_to_scalar(reward_component_logits, self.support_size)
        surprise_logits = self.surprise_net(reward_input).squeeze(-1)
        surprise = torch.nn.functional.softplus(surprise_logits)
        return (
            self._flatten(slots),
            reward_logits,
            reward,
            reward_component_logits,
            reward_components,
            surprise_logits,
            surprise,
        )


class TokenLatentProjector(nn.Module):
    """Token-aware latent projector that preserves the slot structure."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        d_model: int,
        num_memory_slots: int,
        n_heads: int = 4,
        ffn_dim: int = 512,
        num_layers: int = 2,
        latent_type_vocab: int = 4,
        dropout: float = 0.0,
        activation_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.d_model = int(d_model)
        self.num_memory_slots = int(num_memory_slots)
        self.n_heads = max(1, int(n_heads))
        self.ffn_dim = int(ffn_dim)
        self.activation_checkpointing = bool(activation_checkpointing)
        if self.hidden_dim != self.d_model * self.num_memory_slots:
            raise ValueError(
                f"TokenLatentProjector expects hidden_dim == d_model * num_memory_slots, "
                f"got {self.hidden_dim} vs {self.d_model} * {self.num_memory_slots}."
            )

        self.num_slot_banks = len(MEMORY_BANK_NAMES)
        self.register_buffer(
            "_slot_bank_ids",
            torch.as_tensor(build_memory_slot_bank_ids(self.num_memory_slots), dtype=torch.long),
            persistent=False,
        )
        self.slot_index_embedding = nn.Embedding(self.num_memory_slots, self.d_model)
        self.slot_bank_embedding = nn.Embedding(self.num_slot_banks, self.d_model)
        self.latent_type_embeddings = nn.Embedding(max(int(latent_type_vocab), 1), self.d_model)
        self.input_proj = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
        )
        self.slot_blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(self.d_model, self.n_heads, self.ffn_dim, dropout=dropout)
                for _ in range(max(int(num_layers), 1))
            ]
        )
        self.output_norm = nn.LayerNorm(self.d_model)

    def _unflatten(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return hidden_state.reshape(hidden_state.shape[0], self.num_memory_slots, self.d_model)

    def _flatten(self, slots: torch.Tensor) -> torch.Tensor:
        return slots.reshape(slots.shape[0], -1)

    def _slot_mask(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.ones((batch_size, self.num_memory_slots), dtype=torch.bool, device=device)

    def _slot_bank_id_tensor(self, device: torch.device) -> torch.Tensor:
        return self._slot_bank_ids.to(device=device)

    def forward(self, hidden_state: torch.Tensor, *, latent_type_id: int = 0) -> torch.Tensor:
        slots = self._unflatten(hidden_state)
        batch_size = slots.shape[0]
        slot_positions = torch.arange(self.num_memory_slots, device=slots.device, dtype=torch.long)
        slots = slots + self.input_proj(slots)
        slots = (
            slots
            + self.slot_index_embedding(slot_positions).unsqueeze(0)
            + self.slot_bank_embedding(self._slot_bank_id_tensor(slots.device)).unsqueeze(0)
        )
        latent_type = self.latent_type_embeddings(
            torch.full((batch_size,), int(latent_type_id), dtype=torch.long, device=slots.device)
        ).unsqueeze(1)
        slots = slots + latent_type
        slot_mask = self._slot_mask(batch_size, slots.device)
        for block in self.slot_blocks:
            slots = _run_encoder_block(self.activation_checkpointing, block, slots, mask=slot_mask)
        return self._flatten(self.output_norm(slots))


class TokenTransitionSurfaceHead(nn.Module):
    """Slot-aware auxiliary head for next legal surface prediction."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        d_model: int,
        num_memory_slots: int,
        n_heads: int = 4,
        ffn_dim: int = 512,
        dropout: float = 0.0,
        activation_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.d_model = int(d_model)
        self.num_memory_slots = int(num_memory_slots)
        self.n_heads = max(1, int(n_heads))
        self.ffn_dim = int(ffn_dim)
        self.activation_checkpointing = bool(activation_checkpointing)
        if self.hidden_dim != self.d_model * self.num_memory_slots:
            raise ValueError(
                f"TokenTransitionSurfaceHead expects hidden_dim == d_model * num_memory_slots, "
                f"got {self.hidden_dim} vs {self.d_model} * {self.num_memory_slots}."
            )

        self.num_slot_banks = len(MEMORY_BANK_NAMES)
        self.register_buffer(
            "_slot_bank_ids",
            torch.as_tensor(build_memory_slot_bank_ids(self.num_memory_slots), dtype=torch.long),
            persistent=False,
        )
        self.slot_index_embedding = nn.Embedding(self.num_memory_slots, self.d_model)
        self.slot_bank_embedding = nn.Embedding(self.num_slot_banks, self.d_model)
        self.surface_type_embedding = nn.Parameter(torch.randn(1, 1, self.d_model) * 0.02)
        self.slot_blocks = nn.ModuleList(
            [TransformerEncoderBlock(self.d_model, self.n_heads, self.ffn_dim, dropout=dropout) for _ in range(2)]
        )
        self.slot_pool = EntityPooling(self.d_model, self.n_heads, dropout=dropout)

        self.action_queries = nn.Parameter(torch.randn(1, MAX_ACTIONS, self.d_model) * 0.02)
        self.action_cross = CrossAttentionBlock(self.d_model, self.n_heads, self.ffn_dim, dropout=dropout)
        self.action_refine = TransformerEncoderBlock(self.d_model, self.n_heads, self.ffn_dim, dropout=dropout)
        self.action_mask_head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, 1),
        )
        self.decision_domain_head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, NUM_DOMAINS),
        )
        self.phase_head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, NUM_PHASES),
        )

        nn.init.constant_(self.action_mask_head[-1].bias, -2.0)

    def _unflatten(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return hidden_state.reshape(hidden_state.shape[0], self.num_memory_slots, self.d_model)

    def _slot_mask(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.ones((batch_size, self.num_memory_slots), dtype=torch.bool, device=device)

    def _slot_bank_id_tensor(self, device: torch.device) -> torch.Tensor:
        return self._slot_bank_ids.to(device=device)

    def _slot_states(self, hidden_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        slots = self._unflatten(hidden_state)
        batch_size = slots.shape[0]
        slot_positions = torch.arange(self.num_memory_slots, device=slots.device, dtype=torch.long)
        slots = (
            slots
            + self.slot_index_embedding(slot_positions).unsqueeze(0)
            + self.slot_bank_embedding(self._slot_bank_id_tensor(slots.device)).unsqueeze(0)
            + self.surface_type_embedding
        )
        slot_mask = self._slot_mask(batch_size, slots.device)
        for block in self.slot_blocks:
            slots = _run_encoder_block(self.activation_checkpointing, block, slots, mask=slot_mask)
        return slots, slot_mask

    def forward(self, hidden_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        slots, slot_mask = self._slot_states(hidden_state)
        batch_size = slots.shape[0]
        pooled = self.slot_pool(slots, mask=slot_mask)
        action_tokens = self.action_queries.expand(batch_size, -1, -1)
        action_mask = torch.ones((batch_size, MAX_ACTIONS), dtype=torch.bool, device=slots.device)
        action_tokens = _run_cross_block(
            self.activation_checkpointing,
            self.action_cross,
            action_tokens,
            slots,
            query_mask=action_mask,
            memory_mask=slot_mask,
        )
        action_tokens = _run_encoder_block(
            self.activation_checkpointing,
            self.action_refine,
            action_tokens,
            mask=action_mask,
        )
        return (
            self.action_mask_head(action_tokens).squeeze(-1),
            self.decision_domain_head(pooled),
            self.phase_head(pooled),
        )


class TokenFutureWorldBankHead(nn.Module):
    """Predict future bank-level world states from bank-aware latent slots."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        d_model: int,
        num_memory_slots: int,
        bank_token_slots: int = 4,
        slot_source_same_bank_bias: float = 0.35,
        slot_source_same_slot_bias: float = 0.2,
        slot_source_type_match_scale: float = 0.5,
        slot_source_zone_transport_scale: float = 0.35,
        n_heads: int = 4,
        ffn_dim: int = 512,
        dropout: float = 0.0,
        activation_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.d_model = int(d_model)
        self.num_memory_slots = int(num_memory_slots)
        self.bank_token_slots = max(1, int(bank_token_slots))
        self.slot_source_same_bank_bias = float(slot_source_same_bank_bias)
        self.slot_source_same_slot_bias = float(slot_source_same_slot_bias)
        self.slot_source_type_match_scale = float(slot_source_type_match_scale)
        self.slot_source_zone_transport_scale = float(slot_source_zone_transport_scale)
        self.n_heads = max(1, int(n_heads))
        self.ffn_dim = int(ffn_dim)
        self.num_slot_banks = len(MEMORY_BANK_NAMES)
        self.activation_checkpointing = bool(activation_checkpointing)
        if self.hidden_dim != self.d_model * self.num_memory_slots:
            raise ValueError(
                f"TokenFutureWorldBankHead expects hidden_dim == d_model * num_memory_slots, "
                f"got {self.hidden_dim} vs {self.d_model} * {self.num_memory_slots}."
            )

        self.register_buffer(
            "_slot_bank_ids",
            torch.as_tensor(build_memory_slot_bank_ids(self.num_memory_slots), dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "_zone_transport_prior",
            build_zone_transport_prior(),
            persistent=False,
        )
        self.slot_index_embedding = nn.Embedding(self.num_memory_slots, self.d_model)
        self.slot_bank_embedding = nn.Embedding(self.num_slot_banks, self.d_model)
        self.current_type_embedding = nn.Parameter(torch.randn(1, 1, self.d_model) * 0.02)
        self.future_type_embedding = nn.Parameter(torch.randn(1, 1, self.d_model) * 0.02)
        self.slot_blocks = nn.ModuleList(
            [TransformerEncoderBlock(self.d_model, self.n_heads, self.ffn_dim, dropout=dropout) for _ in range(2)]
        )
        self.bank_pool = EntityPooling(self.d_model, self.n_heads, dropout=dropout)
        self.bank_refine = TransformerEncoderBlock(self.d_model, self.n_heads, self.ffn_dim, dropout=dropout)
        self.bank_out_proj = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
        )
        self.bank_occupancy_head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, 1),
        )
        self.bank_token_presence_head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, NUM_TOKEN_TYPES),
        )
        self.bank_token_distribution_head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, NUM_TOKEN_TYPES),
        )
        self.bank_token_slot_seed = nn.Parameter(
            torch.randn(1, len(WORLD_BANK_NAMES), self.bank_token_slots, self.d_model) * 0.02
        )
        self.bank_token_slot_embedding = nn.Embedding(self.bank_token_slots, self.d_model)
        self.bank_token_slot_cross = CrossAttentionBlock(self.d_model, self.n_heads, self.ffn_dim, dropout=dropout)
        self.bank_token_slot_copy_cross = CrossAttentionBlock(self.d_model, self.n_heads, self.ffn_dim, dropout=dropout)
        self.bank_token_slot_refine = TransformerEncoderBlock(self.d_model, self.n_heads, self.ffn_dim, dropout=dropout)
        self.bank_token_slot_copy_refine = TransformerEncoderBlock(self.d_model, self.n_heads, self.ffn_dim, dropout=dropout)
        self.bank_token_slot_out_proj = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
        )
        self.bank_token_slot_mask_head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, 1),
        )
        self.bank_token_slot_type_head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, NUM_TOKEN_TYPES),
        )
        self.bank_token_slot_zone_head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, MAX_ZONE_ID + 1),
        )
        self.bank_token_slot_source_q = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model),
        )
        self.bank_token_slot_source_k = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model),
        )
        self.bank_token_slot_new_head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, 1),
        )

    def _unflatten(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return hidden_state.reshape(hidden_state.shape[0], self.num_memory_slots, self.d_model)

    def _slot_mask(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.ones((batch_size, self.num_memory_slots), dtype=torch.bool, device=device)

    def _slot_bank_id_tensor(self, device: torch.device) -> torch.Tensor:
        return self._slot_bank_ids.to(device=device)

    def _slot_states(self, hidden_state: torch.Tensor, *, future_mode: bool) -> torch.Tensor:
        slots = self._unflatten(hidden_state)
        batch_size = slots.shape[0]
        slot_positions = torch.arange(self.num_memory_slots, device=slots.device, dtype=torch.long)
        type_embedding = self.future_type_embedding if future_mode else self.current_type_embedding
        slots = (
            slots
            + self.slot_index_embedding(slot_positions).unsqueeze(0)
            + self.slot_bank_embedding(self._slot_bank_id_tensor(slots.device)).unsqueeze(0)
            + type_embedding
        )
        slot_mask = self._slot_mask(batch_size, slots.device)
        for block in self.slot_blocks:
            slots = _run_encoder_block(self.activation_checkpointing, block, slots, mask=slot_mask)
        return slots

    def _pool_bank_states(self, slots: torch.Tensor) -> torch.Tensor:
        batch_size = slots.shape[0]
        slot_bank_ids = self._slot_bank_id_tensor(slots.device)
        bank_states = []
        bank_mask_rows = []
        for bank_index in range(len(WORLD_BANK_NAMES)):
            bank_slot_mask = slot_bank_ids.eq(bank_index).unsqueeze(0).expand(batch_size, -1)
            bank_states.append(self.bank_pool(slots, mask=bank_slot_mask))
            bank_mask_rows.append(bool(bank_slot_mask[0].any().item()))
        bank_states_tensor = torch.stack(bank_states, dim=1)
        bank_mask = torch.as_tensor(bank_mask_rows, dtype=torch.bool, device=slots.device).unsqueeze(0).expand(batch_size, -1)
        bank_states_tensor = _run_encoder_block(
            self.activation_checkpointing,
            self.bank_refine,
            bank_states_tensor,
            mask=bank_mask,
        )
        return bank_states_tensor

    def read_current_bank_states(self, hidden_state: torch.Tensor) -> torch.Tensor:
        slots = self._slot_states(hidden_state, future_mode=False)
        return self.bank_out_proj(self._pool_bank_states(slots))

    def _predict_bank_token_slots(
        self,
        bank_states_tensor: torch.Tensor,
        slots: torch.Tensor,
        *,
        future_mode: bool,
        current_bank_token_slot_states: torch.Tensor | None = None,
        current_bank_token_slot_mask: torch.Tensor | None = None,
        current_bank_token_slot_type_ids: torch.Tensor | None = None,
        current_bank_token_slot_zone_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        batch_size = slots.shape[0]
        bank_count = len(WORLD_BANK_NAMES)
        slot_positions = torch.arange(self.bank_token_slots, device=slots.device, dtype=torch.long)
        token_queries = self.bank_token_slot_seed.expand(batch_size, -1, -1, -1)
        token_queries = token_queries + bank_states_tensor.unsqueeze(2)
        token_queries = token_queries + self.bank_token_slot_embedding(slot_positions).view(
            1,
            1,
            self.bank_token_slots,
            self.d_model,
        )
        token_query_type = self.future_type_embedding if future_mode else self.current_type_embedding
        token_queries = token_queries + token_query_type.unsqueeze(2)
        flat_queries = token_queries.reshape(batch_size, bank_count * self.bank_token_slots, self.d_model)
        flat_query_mask = torch.ones(
            (batch_size, bank_count * self.bank_token_slots),
            dtype=torch.bool,
            device=slots.device,
        )
        flat_queries = _run_cross_block(
            self.activation_checkpointing,
            self.bank_token_slot_cross,
            flat_queries,
            slots,
            query_mask=flat_query_mask,
            memory_mask=self._slot_mask(batch_size, slots.device),
        )
        flat_queries = _run_encoder_block(
            self.activation_checkpointing,
            self.bank_token_slot_refine,
            flat_queries,
            mask=flat_query_mask,
        )
        source_logits: torch.Tensor | None = None
        if current_bank_token_slot_states is not None:
            flat_current_slot_states = current_bank_token_slot_states.reshape(batch_size, -1, self.d_model)
            if current_bank_token_slot_mask is None:
                flat_current_slot_mask = torch.ones(
                    (batch_size, flat_current_slot_states.shape[1]),
                    dtype=torch.bool,
                    device=slots.device,
                )
            else:
                flat_current_slot_mask = current_bank_token_slot_mask.reshape(batch_size, -1).bool()
            copy_queries = _run_cross_block(
                self.activation_checkpointing,
                self.bank_token_slot_copy_cross,
                flat_queries,
                flat_current_slot_states,
                query_mask=flat_query_mask,
                memory_mask=flat_current_slot_mask,
            )
            copy_queries = _run_encoder_block(
                self.activation_checkpointing,
                self.bank_token_slot_copy_refine,
                copy_queries,
                mask=flat_query_mask,
            )
            transport_type_logits = self.bank_token_slot_type_head(copy_queries).reshape(
                batch_size,
                bank_count * self.bank_token_slots,
                NUM_TOKEN_TYPES,
            )
            transport_zone_logits = self.bank_token_slot_zone_head(copy_queries).reshape(
                batch_size,
                bank_count * self.bank_token_slots,
                MAX_ZONE_ID + 1,
            )
            source_q = self.bank_token_slot_source_q(copy_queries)
            source_k = self.bank_token_slot_source_k(flat_current_slot_states)
            source_memory_logits = torch.einsum("bqd,bkd->bqk", source_q, source_k) / math.sqrt(float(self.d_model))
            source_memory_logits = source_memory_logits.masked_fill(~flat_current_slot_mask.unsqueeze(1), -1e4)
            query_bank_ids = (
                torch.arange(bank_count, device=slots.device, dtype=torch.long)
                .view(1, bank_count, 1)
                .expand(batch_size, bank_count, self.bank_token_slots)
                .reshape(batch_size, -1)
            )
            query_slot_ids = (
                torch.arange(self.bank_token_slots, device=slots.device, dtype=torch.long)
                .view(1, 1, self.bank_token_slots)
                .expand(batch_size, bank_count, self.bank_token_slots)
                .reshape(batch_size, -1)
            )
            current_bank_ids = (
                torch.arange(bank_count, device=slots.device, dtype=torch.long)
                .view(1, bank_count, 1)
                .expand(batch_size, bank_count, self.bank_token_slots)
                .reshape(batch_size, -1)
            )
            current_slot_ids = (
                torch.arange(self.bank_token_slots, device=slots.device, dtype=torch.long)
                .view(1, 1, self.bank_token_slots)
                .expand(batch_size, bank_count, self.bank_token_slots)
                .reshape(batch_size, -1)
            )
            if self.slot_source_same_bank_bias != 0.0:
                same_bank = query_bank_ids.unsqueeze(-1).eq(current_bank_ids.unsqueeze(1))
                source_memory_logits = source_memory_logits + self.slot_source_same_bank_bias * same_bank.float()
            if self.slot_source_same_slot_bias != 0.0:
                same_slot = query_bank_ids.unsqueeze(-1).eq(current_bank_ids.unsqueeze(1)) & query_slot_ids.unsqueeze(-1).eq(
                    current_slot_ids.unsqueeze(1)
                )
                source_memory_logits = source_memory_logits + self.slot_source_same_slot_bias * same_slot.float()
            if current_bank_token_slot_type_ids is not None and self.slot_source_type_match_scale != 0.0:
                flat_transport_type_probs = torch.softmax(transport_type_logits, dim=-1)
                flat_current_type_ids = current_bank_token_slot_type_ids.reshape(batch_size, -1).clamp(
                    min=0,
                    max=NUM_TOKEN_TYPES - 1,
                )
                gathered_type_match = flat_transport_type_probs.gather(
                    -1,
                    flat_current_type_ids.unsqueeze(1).expand(-1, bank_count * self.bank_token_slots, -1),
                )
                source_memory_logits = source_memory_logits + self.slot_source_type_match_scale * gathered_type_match
            if current_bank_token_slot_zone_ids is not None and self.slot_source_zone_transport_scale != 0.0:
                flat_current_zone_ids = current_bank_token_slot_zone_ids.reshape(batch_size, -1).clamp(
                    min=0,
                    max=MAX_ZONE_ID,
                )
                current_zone_one_hot = torch.nn.functional.one_hot(
                    flat_current_zone_ids,
                    num_classes=MAX_ZONE_ID + 1,
                ).float()
                transport_zone_probs = torch.softmax(transport_zone_logits, dim=-1)
                zone_bias = torch.einsum(
                    "bqz,zc,bkc->bqk",
                    transport_zone_probs,
                    self._zone_transport_prior.to(device=slots.device),
                    current_zone_one_hot,
                )
                source_memory_logits = source_memory_logits + self.slot_source_zone_transport_scale * zone_bias
            new_token_logits = self.bank_token_slot_new_head(copy_queries)
            source_logits = torch.cat([source_memory_logits, new_token_logits], dim=-1)
            source_probs = torch.softmax(source_logits, dim=-1)
            source_probs = torch.nan_to_num(source_probs, nan=0.0, posinf=0.0, neginf=0.0)
            copy_probs = source_probs[..., :-1]
            new_prob = source_probs[..., -1:].clamp(min=0.0, max=1.0)
            copied_state = torch.einsum("bqk,bkd->bqd", copy_probs, flat_current_slot_states)
            flat_queries = new_prob * flat_queries + (1.0 - new_prob) * copied_state
        token_slot_states = self.bank_token_slot_out_proj(flat_queries).reshape(
            batch_size,
            bank_count,
            self.bank_token_slots,
            self.d_model,
        )
        token_slot_mask_logits = self.bank_token_slot_mask_head(flat_queries).reshape(
            batch_size,
            bank_count,
            self.bank_token_slots,
        )
        token_slot_type_logits = self.bank_token_slot_type_head(flat_queries).reshape(
            batch_size,
            bank_count,
            self.bank_token_slots,
            NUM_TOKEN_TYPES,
        )
        token_slot_zone_logits = self.bank_token_slot_zone_head(flat_queries).reshape(
            batch_size,
            bank_count,
            self.bank_token_slots,
            MAX_ZONE_ID + 1,
        )
        if source_logits is not None:
            source_logits = source_logits.reshape(
                batch_size,
                bank_count,
                self.bank_token_slots,
                bank_count * self.bank_token_slots + 1,
            )
        return token_slot_states, token_slot_mask_logits, token_slot_type_logits, token_slot_zone_logits, source_logits

    def read_current_bank_token_slots(
        self,
        hidden_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        slots = self._slot_states(hidden_state, future_mode=False)
        bank_states_tensor = self._pool_bank_states(slots)
        (
            token_slot_states,
            token_slot_mask_logits,
            token_slot_type_logits,
            token_slot_zone_logits,
            _source_logits,
        ) = self._predict_bank_token_slots(
            bank_states_tensor,
            slots,
            future_mode=False,
        )
        return token_slot_states, token_slot_mask_logits, token_slot_type_logits, token_slot_zone_logits

    def forward(
        self,
        hidden_state: torch.Tensor,
        *,
        current_bank_token_slot_states: torch.Tensor | None = None,
        current_bank_token_slot_mask: torch.Tensor | None = None,
        current_bank_token_slot_type_ids: torch.Tensor | None = None,
        current_bank_token_slot_zone_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        slots = self._slot_states(hidden_state, future_mode=True)
        bank_states_tensor = self._pool_bank_states(slots)
        bank_state_pred = self.bank_out_proj(bank_states_tensor)
        bank_occupancy_logits = self.bank_occupancy_head(bank_states_tensor).squeeze(-1)
        bank_token_presence_logits = self.bank_token_presence_head(bank_states_tensor)
        bank_token_distribution_logits = self.bank_token_distribution_head(bank_states_tensor)
        (
            bank_token_slot_states,
            bank_token_slot_mask_logits,
            bank_token_slot_type_logits,
            bank_token_slot_zone_logits,
            bank_token_slot_source_logits,
        ) = self._predict_bank_token_slots(
            bank_states_tensor,
            slots,
            future_mode=True,
            current_bank_token_slot_states=current_bank_token_slot_states,
            current_bank_token_slot_mask=current_bank_token_slot_mask,
            current_bank_token_slot_type_ids=current_bank_token_slot_type_ids,
            current_bank_token_slot_zone_ids=current_bank_token_slot_zone_ids,
        )
        return (
            bank_state_pred,
            bank_occupancy_logits,
            bank_token_presence_logits,
            bank_token_distribution_logits,
            bank_token_slot_states,
            bank_token_slot_mask_logits,
            bank_token_slot_type_logits,
            bank_token_slot_zone_logits,
            bank_token_slot_source_logits,
        )


class TokenPredictionNetwork(nn.Module):
    """Slot-aware prediction head for token-memory MuZero."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        action_embed_dim: int,
        d_model: int,
        num_memory_slots: int,
        support_size: int = 25,
        internal_planner_blend: float = 0.7,
        internal_planner_q_blend: float = 0.5,
        internal_planner_objective_q_blend: float = 0.35,
        internal_planner_risk_blend: float = 0.25,
        n_heads: int = 4,
        ffn_dim: int = 512,
        dropout: float = 0.0,
        activation_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.action_embed_dim = int(action_embed_dim)
        self.d_model = int(d_model)
        self.num_memory_slots = int(num_memory_slots)
        self.support_size = int(support_size)
        self.num_bins = 2 * self.support_size + 1
        self.internal_planner_blend = float(internal_planner_blend)
        self.internal_planner_q_blend = float(internal_planner_q_blend)
        self.internal_planner_objective_q_blend = float(internal_planner_objective_q_blend)
        self.internal_planner_risk_blend = float(internal_planner_risk_blend)
        self.n_heads = max(1, int(n_heads))
        self.ffn_dim = int(ffn_dim)
        self.activation_checkpointing = bool(activation_checkpointing)
        if self.hidden_dim != self.d_model * self.num_memory_slots:
            raise ValueError(
                f"TokenPredictionNetwork expects hidden_dim == d_model * num_memory_slots, "
                f"got {self.hidden_dim} vs {self.d_model} * {self.num_memory_slots}."
            )

        self.num_slot_banks = len(MEMORY_BANK_NAMES)
        self.register_buffer(
            "_slot_bank_ids",
            torch.as_tensor(build_memory_slot_bank_ids(self.num_memory_slots), dtype=torch.long),
            persistent=False,
        )
        self.slot_index_embedding = nn.Embedding(self.num_memory_slots, self.d_model)
        self.slot_bank_embedding = nn.Embedding(self.num_slot_banks, self.d_model)
        self.latent_type_embeddings = nn.Embedding(2, self.d_model)
        self.slot_blocks = nn.ModuleList(
            [TransformerEncoderBlock(self.d_model, self.n_heads, self.ffn_dim, dropout=dropout) for _ in range(2)]
        )
        self.slot_pool = EntityPooling(self.d_model, self.n_heads, dropout=dropout)
        self.bank_pool = EntityPooling(self.d_model, self.n_heads, dropout=dropout)
        self.bank_state_refine = TransformerEncoderBlock(self.d_model, self.n_heads, self.ffn_dim, dropout=dropout)

        self.domain_seed = nn.Parameter(torch.randn(1, NUM_DOMAINS, self.d_model) * 0.02)
        self.domain_cross = CrossAttentionBlock(self.d_model, self.n_heads, self.ffn_dim, dropout=dropout)
        self.domain_refine = TransformerEncoderBlock(self.d_model, self.n_heads, self.ffn_dim, dropout=dropout)
        self.domain_gate = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, NUM_DOMAINS),
        )

        self.candidate_in_proj = nn.Sequential(
            nn.LayerNorm(self.action_embed_dim),
            nn.Linear(self.action_embed_dim, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
        )
        self.candidate_cross = CrossAttentionBlock(self.d_model, self.n_heads, self.ffn_dim, dropout=dropout)
        self.candidate_bank_cross = CrossAttentionBlock(self.d_model, self.n_heads, self.ffn_dim, dropout=dropout)
        self.candidate_refine = TransformerEncoderBlock(self.d_model, self.n_heads, self.ffn_dim, dropout=dropout)
        self.candidate_score_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(self.d_model),
                    nn.Linear(self.d_model, self.d_model),
                    nn.GELU(),
                    nn.Linear(self.d_model, 1),
                )
                for _ in range(NUM_DOMAINS)
            ]
        )
        self.planner_bank_seed = nn.Parameter(torch.randn(1, self.num_slot_banks, self.d_model) * 0.02)
        self.planner_slot_cross = CrossAttentionBlock(self.d_model, self.n_heads, self.ffn_dim, dropout=dropout)
        self.planner_bank_cross = CrossAttentionBlock(self.d_model, self.n_heads, self.ffn_dim, dropout=dropout)
        self.planner_refine = TransformerEncoderBlock(self.d_model, self.n_heads, self.ffn_dim, dropout=dropout)
        self.planner_pool = EntityPooling(self.d_model, self.n_heads, dropout=dropout)
        self.planner_score_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(self.d_model),
                    nn.Linear(self.d_model, self.d_model),
                    nn.GELU(),
                    nn.Linear(self.d_model, 1),
                )
                for _ in range(NUM_DOMAINS)
            ]
        )
        self.planner_q_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(self.d_model),
                    nn.Linear(self.d_model, self.d_model),
                    nn.GELU(),
                    nn.Linear(self.d_model, self.num_bins),
                )
                for _ in range(NUM_DOMAINS)
            ]
        )
        self.planner_q_component_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(self.d_model),
                    nn.Linear(self.d_model, self.d_model),
                    nn.GELU(),
                    nn.Linear(self.d_model, self.num_bins * NUM_OBJECTIVE_HEADS),
                )
                for _ in range(NUM_DOMAINS)
            ]
        )

        self.latent_bank_queries = nn.Parameter(torch.randn(NUM_DOMAINS, MAX_ACTIONS, self.d_model) * 0.02)
        self.latent_bank_cross = nn.ModuleList(
            [CrossAttentionBlock(self.d_model, self.n_heads, self.ffn_dim, dropout=dropout) for _ in range(NUM_DOMAINS)]
        )
        self.latent_bank_bank_cross = nn.ModuleList(
            [CrossAttentionBlock(self.d_model, self.n_heads, self.ffn_dim, dropout=dropout) for _ in range(NUM_DOMAINS)]
        )
        self.latent_bank_refine = nn.ModuleList(
            [TransformerEncoderBlock(self.d_model, self.n_heads, self.ffn_dim, dropout=dropout) for _ in range(NUM_DOMAINS)]
        )
        self.latent_policy_score_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(self.d_model),
                    nn.Linear(self.d_model, self.d_model),
                    nn.GELU(),
                    nn.Linear(self.d_model, 1),
                )
                for _ in range(NUM_DOMAINS)
            ]
        )
        self.latent_action_proj = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.action_embed_dim),
        )

        self.value_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(self.d_model),
                    nn.Linear(self.d_model, self.d_model),
                    nn.GELU(),
                    nn.Linear(self.d_model, self.num_bins),
                )
                for _ in range(NUM_DOMAINS)
            ]
        )
        self.value_component_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(self.d_model),
                    nn.Linear(self.d_model, self.d_model),
                    nn.GELU(),
                    nn.Linear(self.d_model, self.num_bins * NUM_OBJECTIVE_HEADS),
                )
                for _ in range(NUM_DOMAINS)
            ]
        )
        self.semantic_policy_head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, SEMANTIC_ROLLOUT_SIZE),
        )

    def _unflatten(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return hidden_state.reshape(hidden_state.shape[0], self.num_memory_slots, self.d_model)

    def _slot_mask(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.ones((batch_size, self.num_memory_slots), dtype=torch.bool, device=device)

    def _slot_bank_id_tensor(self, device: torch.device) -> torch.Tensor:
        return self._slot_bank_ids.to(device=device)

    def _slot_states(self, hidden_state: torch.Tensor, *, latent_type_id: int) -> torch.Tensor:
        slots = self._unflatten(hidden_state)
        batch_size = slots.shape[0]
        slot_positions = torch.arange(self.num_memory_slots, device=slots.device, dtype=torch.long)
        slots = (
            slots
            + self.slot_index_embedding(slot_positions).unsqueeze(0)
            + self.slot_bank_embedding(self._slot_bank_id_tensor(slots.device)).unsqueeze(0)
        )
        latent_type = self.latent_type_embeddings(
            torch.full((batch_size,), int(latent_type_id), dtype=torch.long, device=slots.device)
        ).unsqueeze(1)
        slots = slots + latent_type
        slot_mask = self._slot_mask(batch_size, slots.device)
        for block in self.slot_blocks:
            slots = _run_encoder_block(self.activation_checkpointing, block, slots, mask=slot_mask)
        return slots

    def _bank_states(self, slots: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = slots.shape[0]
        slot_bank_ids = self._slot_bank_id_tensor(slots.device)
        bank_states = []
        bank_mask_rows = []
        for bank_index in range(self.num_slot_banks):
            bank_slot_mask = slot_bank_ids.eq(bank_index).unsqueeze(0).expand(batch_size, -1)
            bank_states.append(self.bank_pool(slots, mask=bank_slot_mask))
            bank_mask_rows.append(bool(bank_slot_mask[0].any().item()))
        bank_states_tensor = torch.stack(bank_states, dim=1)
        bank_mask = torch.as_tensor(bank_mask_rows, dtype=torch.bool, device=slots.device).unsqueeze(0).expand(batch_size, -1)
        bank_states_tensor = _run_encoder_block(
            self.activation_checkpointing,
            self.bank_state_refine,
            bank_states_tensor,
            mask=bank_mask,
        )
        return bank_states_tensor, bank_mask

    @staticmethod
    def _routing_weights(
        *,
        domain_weights: torch.Tensor,
        decision_domain: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if decision_domain is None or decision_domain.dim() != 2 or decision_domain.shape[1] < NUM_DOMAINS:
            return domain_weights
        routing = decision_domain[:, :NUM_DOMAINS].float()
        routing_sum = routing.sum(dim=-1, keepdim=True)
        normalized = routing / routing_sum.clamp(min=1.0)
        valid = routing_sum > 0
        return torch.where(valid, normalized, domain_weights)

    def _domain_states(
        self,
        slots: torch.Tensor,
        *,
        bank_states: torch.Tensor | None = None,
        bank_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = slots.shape[0]
        slot_mask = self._slot_mask(batch_size, slots.device)
        if bank_states is None or bank_mask is None:
            bank_states, bank_mask = self._bank_states(slots)
        domain_queries = self.domain_seed.expand(batch_size, -1, -1)
        domain_mask = torch.ones((batch_size, NUM_DOMAINS), dtype=torch.bool, device=slots.device)
        domain_states = _run_cross_block(
            self.activation_checkpointing,
            self.domain_cross,
            domain_queries,
            bank_states,
            query_mask=domain_mask,
            memory_mask=bank_mask,
        )
        domain_states = _run_encoder_block(
            self.activation_checkpointing,
            self.domain_refine,
            domain_states,
            mask=domain_mask,
        )
        pooled = self.slot_pool(slots, mask=slot_mask)
        pooled = pooled + self.bank_pool(bank_states, mask=bank_mask)
        domain_logits = self.domain_gate(pooled)
        return pooled, domain_states, domain_logits

    def _candidate_tokens(
        self,
        action_embeddings: torch.Tensor,
        slots: torch.Tensor,
        bank_states: torch.Tensor,
        bank_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = slots.shape[0]
        slot_mask = self._slot_mask(batch_size, slots.device)
        action_mask = action_embeddings.abs().sum(dim=-1) > 0
        candidate_x = self.candidate_in_proj(action_embeddings)
        candidate_x = _run_cross_block(
            self.activation_checkpointing,
            self.candidate_cross,
            candidate_x,
            slots,
            query_mask=action_mask,
            memory_mask=slot_mask,
        )
        candidate_x = _run_cross_block(
            self.activation_checkpointing,
            self.candidate_bank_cross,
            candidate_x,
            bank_states,
            query_mask=action_mask,
            memory_mask=bank_mask,
        )
        candidate_x = _run_encoder_block(
            self.activation_checkpointing,
            self.candidate_refine,
            candidate_x,
            mask=action_mask,
        )
        return candidate_x * action_mask.unsqueeze(-1).float()

    def _score_candidates(
        self,
        candidate_tokens: torch.Tensor,
        routing_weights: torch.Tensor,
        domain_states: torch.Tensor,
    ) -> torch.Tensor:
        domain_logits = []
        for domain_idx, head in enumerate(self.candidate_score_heads):
            conditioned = candidate_tokens + domain_states[:, domain_idx : domain_idx + 1, :]
            domain_logits.append(head(conditioned).squeeze(-1))
        stacked = torch.stack(domain_logits, dim=1)
        return (stacked * routing_weights.unsqueeze(-1)).sum(dim=1)

    def _planner_features(
        self,
        candidate_tokens: torch.Tensor,
        slots: torch.Tensor,
        bank_states: torch.Tensor,
        bank_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, action_count, _ = candidate_tokens.shape
        action_mask = candidate_tokens.abs().sum(dim=-1) > 0
        if not action_mask.any():
            return candidate_tokens.new_zeros((batch_size, action_count, self.d_model)), action_mask

        slot_mask = self._slot_mask(batch_size, slots.device)
        planner_queries = (
            self.planner_bank_seed.view(1, 1, self.num_slot_banks, self.d_model)
            + candidate_tokens.unsqueeze(2)
            + bank_states.unsqueeze(1)
        )
        flat_queries = planner_queries.reshape(batch_size * action_count, self.num_slot_banks, self.d_model)
        flat_query_mask = action_mask.reshape(batch_size * action_count, 1).expand(-1, self.num_slot_banks)
        flat_slots = slots.unsqueeze(1).expand(batch_size, action_count, self.num_memory_slots, self.d_model).reshape(
            batch_size * action_count,
            self.num_memory_slots,
            self.d_model,
        )
        flat_slot_mask = slot_mask.unsqueeze(1).expand(batch_size, action_count, self.num_memory_slots).reshape(
            batch_size * action_count,
            self.num_memory_slots,
        )
        flat_bank_states = bank_states.unsqueeze(1).expand(batch_size, action_count, self.num_slot_banks, self.d_model).reshape(
            batch_size * action_count,
            self.num_slot_banks,
            self.d_model,
        )
        flat_bank_mask = bank_mask.unsqueeze(1).expand(batch_size, action_count, self.num_slot_banks).reshape(
            batch_size * action_count,
            self.num_slot_banks,
        )
        planner_states = _run_cross_block(
            self.activation_checkpointing,
            self.planner_slot_cross,
            flat_queries,
            flat_slots,
            query_mask=flat_query_mask,
            memory_mask=flat_slot_mask,
        )
        planner_states = _run_cross_block(
            self.activation_checkpointing,
            self.planner_bank_cross,
            planner_states,
            flat_bank_states,
            query_mask=flat_query_mask,
            memory_mask=flat_bank_mask,
        )
        planner_states = _run_encoder_block(
            self.activation_checkpointing,
            self.planner_refine,
            planner_states,
            mask=flat_query_mask,
        )
        planner_tokens = self.planner_pool(planner_states, mask=flat_query_mask).reshape(batch_size, action_count, self.d_model)
        return planner_tokens * action_mask.unsqueeze(-1).float(), action_mask

    def _planner_policy_logits(
        self,
        planner_tokens: torch.Tensor,
        candidate_tokens: torch.Tensor,
        routing_weights: torch.Tensor,
        domain_states: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, action_count, _ = candidate_tokens.shape
        action_mask = candidate_tokens.abs().sum(dim=-1) > 0

        domain_logits = []
        for domain_idx, head in enumerate(self.planner_score_heads):
            conditioned = planner_tokens + candidate_tokens + domain_states[:, domain_idx : domain_idx + 1, :]
            domain_logits.append(head(conditioned).squeeze(-1))
        stacked = torch.stack(domain_logits, dim=1)
        planner_logits = (stacked * routing_weights.unsqueeze(-1)).sum(dim=1)
        return planner_logits * action_mask.float()

    def _planner_q_outputs(
        self,
        planner_tokens: torch.Tensor,
        candidate_tokens: torch.Tensor,
        routing_weights: torch.Tensor,
        domain_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from muzero.sts2_env.muzero_model import support_to_scalar

        batch_size, action_count, _ = candidate_tokens.shape
        action_mask = candidate_tokens.abs().sum(dim=-1) > 0
        domain_q_logits = []
        for domain_idx, head in enumerate(self.planner_q_heads):
            conditioned = planner_tokens + candidate_tokens + domain_states[:, domain_idx : domain_idx + 1, :]
            domain_q_logits.append(head(conditioned))
        stacked = torch.stack(domain_q_logits, dim=1)
        mixed_q_logits = (
            stacked * routing_weights.unsqueeze(-1).unsqueeze(-1)
        ).sum(dim=1)
        flat_q = support_to_scalar(mixed_q_logits.reshape(batch_size * action_count, self.num_bins), self.support_size)
        planner_q = flat_q.reshape(batch_size, action_count)
        planner_q = planner_q * action_mask.float()
        return mixed_q_logits, planner_q

    def _planner_objective_q_outputs(
        self,
        planner_tokens: torch.Tensor,
        candidate_tokens: torch.Tensor,
        routing_weights: torch.Tensor,
        domain_states: torch.Tensor,
        *,
        objective_context: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from muzero.sts2_env.muzero_model import support_tensor_to_scalar

        batch_size, action_count, _ = candidate_tokens.shape
        action_mask = candidate_tokens.abs().sum(dim=-1) > 0
        domain_component_logits = []
        for domain_idx, head in enumerate(self.planner_q_component_heads):
            conditioned = planner_tokens + candidate_tokens + domain_states[:, domain_idx : domain_idx + 1, :]
            domain_component_logits.append(
                head(conditioned).view(batch_size, action_count, NUM_OBJECTIVE_HEADS, self.num_bins)
            )
        stacked = torch.stack(domain_component_logits, dim=1)
        mixed_component_logits = (
            stacked * routing_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        ).sum(dim=1)
        planner_q_components = support_tensor_to_scalar(
            mixed_component_logits.reshape(batch_size * action_count, NUM_OBJECTIVE_HEADS, self.num_bins),
            self.support_size,
        ).reshape(batch_size, action_count, NUM_OBJECTIVE_HEADS)
        planner_objective_q = scalarize_objective_components_torch(
            planner_q_components,
            objective_context,
        )
        planner_q_components = planner_q_components * action_mask.unsqueeze(-1).float()
        planner_objective_q = planner_objective_q * action_mask.float()
        return mixed_component_logits, planner_q_components, planner_objective_q

    @staticmethod
    def _normalize_action_values(
        values: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        masked_values = torch.where(action_mask, values, torch.zeros_like(values))
        valid_count = action_mask.float().sum(dim=-1, keepdim=True).clamp(min=1.0)
        mean = masked_values.sum(dim=-1, keepdim=True) / valid_count
        centered = torch.where(action_mask, values - mean, torch.zeros_like(values))
        variance = centered.pow(2).sum(dim=-1, keepdim=True) / valid_count
        std = variance.clamp(min=1e-6).sqrt()
        normalized = centered / std
        return torch.where(action_mask, normalized, torch.zeros_like(normalized))

    def _latent_policy_logits(
        self,
        slots: torch.Tensor,
        routing_weights: torch.Tensor,
        domain_states: torch.Tensor,
        bank_states: torch.Tensor,
        bank_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = slots.shape[0]
        slot_mask = self._slot_mask(batch_size, slots.device)
        action_mask = torch.ones((batch_size, MAX_ACTIONS), dtype=torch.bool, device=slots.device)
        domain_logits = []
        domain_embeddings = []
        for domain_idx in range(NUM_DOMAINS):
            latent_tokens = self.latent_bank_queries[domain_idx].unsqueeze(0).expand(batch_size, -1, -1)
            latent_tokens = latent_tokens + domain_states[:, domain_idx : domain_idx + 1, :]
            latent_tokens = _run_cross_block(
                self.activation_checkpointing,
                self.latent_bank_cross[domain_idx],
                latent_tokens,
                slots,
                query_mask=action_mask,
                memory_mask=slot_mask,
            )
            latent_tokens = _run_cross_block(
                self.activation_checkpointing,
                self.latent_bank_bank_cross[domain_idx],
                latent_tokens,
                bank_states,
                query_mask=action_mask,
                memory_mask=bank_mask,
            )
            latent_tokens = _run_encoder_block(
                self.activation_checkpointing,
                self.latent_bank_refine[domain_idx],
                latent_tokens,
                mask=action_mask,
            )
            domain_logits.append(self.latent_policy_score_heads[domain_idx](latent_tokens).squeeze(-1))
            domain_embeddings.append(self.latent_action_proj(latent_tokens))
        logits = torch.stack(domain_logits, dim=1)
        embeddings = torch.stack(domain_embeddings, dim=1)
        mixed_logits = (logits * routing_weights.unsqueeze(-1)).sum(dim=1)
        mixed_embeddings = (embeddings * routing_weights.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)
        return mixed_logits, mixed_embeddings

    def _value_outputs(
        self,
        slots: torch.Tensor,
        *,
        routing_weights: torch.Tensor,
        objective_context: torch.Tensor | None = None,
        domain_states: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        from muzero.sts2_env.muzero_model import support_to_scalar, support_tensor_to_scalar

        if domain_states is None:
            _pooled, domain_states, _domain_logits = self._domain_states(slots)
        value_stack = torch.stack([head(domain_states[:, idx, :]) for idx, head in enumerate(self.value_heads)], dim=1)
        component_stack = torch.stack(
            [
                head(domain_states[:, idx, :]).view(slots.shape[0], NUM_OBJECTIVE_HEADS, self.num_bins)
                for idx, head in enumerate(self.value_component_heads)
            ],
            dim=1,
        )
        value_logits = (value_stack * routing_weights.unsqueeze(-1)).sum(dim=1)
        value_component_logits = (
            component_stack * routing_weights.unsqueeze(-1).unsqueeze(-1)
        ).sum(dim=1)
        value = support_to_scalar(value_logits, self.support_size)
        value_components = support_tensor_to_scalar(value_component_logits, self.support_size)
        objective_value = scalarize_objective_components_torch(value_components, objective_context)
        return value_logits, value, value_component_logits, value_components, objective_value

    def value_only(
        self,
        hidden_state: torch.Tensor,
        *,
        decision_domain: torch.Tensor | None = None,
        objective_context: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Cheap value-only path for explicit action-conditioned rollout planning.

        This skips candidate scoring / planner-policy heads and only evaluates the
        latent state through the shared domain/value heads.
        """
        slots = self._slot_states(hidden_state, latent_type_id=0)
        bank_states, bank_mask = self._bank_states(slots)
        def _domain_features(slots_: torch.Tensor, bank_states_: torch.Tensor, bank_mask_: torch.Tensor):
            return self._domain_states(
                slots_,
                bank_states=bank_states_,
                bank_mask=bank_mask_,
            )

        _pooled, domain_states, domain_logits = _run_function_with_checkpoint(
            self.activation_checkpointing,
            _domain_features,
            slots,
            bank_states,
            bank_mask,
        )
        domain_weights = torch.softmax(domain_logits, dim=-1)
        routing_weights = self._routing_weights(
            domain_weights=domain_weights,
            decision_domain=decision_domain,
        )
        def _value_head(slots_: torch.Tensor, routing_weights_: torch.Tensor, domain_states_: torch.Tensor):
            return self._value_outputs(
                slots_,
                routing_weights=routing_weights_,
                objective_context=objective_context,
                domain_states=domain_states_,
            )

        return _run_function_with_checkpoint(
            self.activation_checkpointing,
            _value_head,
            slots,
            routing_weights,
            domain_states,
        )

    def latent_policy_only(
        self,
        hidden_state: torch.Tensor,
        *,
        decision_domain: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Cheap latent-action prior path for multi-step search-free rollout."""
        slots = self._slot_states(hidden_state, latent_type_id=0)
        bank_states, bank_mask = self._bank_states(slots)
        def _domain_features(slots_: torch.Tensor, bank_states_: torch.Tensor, bank_mask_: torch.Tensor):
            return self._domain_states(
                slots_,
                bank_states=bank_states_,
                bank_mask=bank_mask_,
            )

        _pooled, domain_states, domain_logits = _run_function_with_checkpoint(
            self.activation_checkpointing,
            _domain_features,
            slots,
            bank_states,
            bank_mask,
        )
        domain_weights = torch.softmax(domain_logits, dim=-1)
        routing_weights = self._routing_weights(
            domain_weights=domain_weights,
            decision_domain=decision_domain,
        )
        def _latent_policy(
            slots_: torch.Tensor,
            routing_weights_: torch.Tensor,
            domain_states_: torch.Tensor,
            bank_states_: torch.Tensor,
            bank_mask_: torch.Tensor,
        ):
            return self._latent_policy_logits(
                slots_,
                routing_weights_,
                domain_states_,
                bank_states_,
                bank_mask_,
            )

        return _run_function_with_checkpoint(
            self.activation_checkpointing,
            _latent_policy,
            slots,
            routing_weights,
            domain_states,
            bank_states,
            bank_mask,
        )

    def forward(
        self,
        hidden_state: torch.Tensor,
        *,
        action_embeddings: torch.Tensor | None = None,
        decision_domain: torch.Tensor | None = None,
        objective_context: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        slots = self._slot_states(hidden_state, latent_type_id=0)
        bank_states, bank_mask = self._bank_states(slots)
        def _domain_features(slots_: torch.Tensor, bank_states_: torch.Tensor, bank_mask_: torch.Tensor):
            return self._domain_states(
                slots_,
                bank_states=bank_states_,
                bank_mask=bank_mask_,
            )

        pooled, domain_states, domain_logits = _run_function_with_checkpoint(
            self.activation_checkpointing,
            _domain_features,
            slots,
            bank_states,
            bank_mask,
        )
        domain_weights = torch.softmax(domain_logits, dim=-1)
        routing_weights = self._routing_weights(domain_weights=domain_weights, decision_domain=decision_domain)

        def _latent_policy(
            slots_: torch.Tensor,
            routing_weights_: torch.Tensor,
            domain_states_: torch.Tensor,
            bank_states_: torch.Tensor,
            bank_mask_: torch.Tensor,
        ):
            return self._latent_policy_logits(
                slots_,
                routing_weights_,
                domain_states_,
                bank_states_,
                bank_mask_,
            )

        latent_policy_logits, latent_action_embeddings = _run_function_with_checkpoint(
            self.activation_checkpointing,
            _latent_policy,
            slots,
            routing_weights,
            domain_states,
            bank_states,
            bank_mask,
        )
        policy_action_embeddings = action_embeddings if action_embeddings is not None else latent_action_embeddings
        candidate_tokens = _run_function_with_checkpoint(
            self.activation_checkpointing,
            lambda action_embeddings_, slots_, bank_states_, bank_mask_: self._candidate_tokens(
                action_embeddings_,
                slots_,
                bank_states_,
                bank_mask_,
            ),
            policy_action_embeddings,
            slots,
            bank_states,
            bank_mask,
        )
        base_policy_logits = _run_function_with_checkpoint(
            self.activation_checkpointing,
            lambda candidate_tokens_, routing_weights_, domain_states_: self._score_candidates(
                candidate_tokens_,
                routing_weights_,
                domain_states_,
            ),
            candidate_tokens,
            routing_weights,
            domain_states,
        )
        planner_tokens, planner_action_mask = _run_function_with_checkpoint(
            self.activation_checkpointing,
            lambda candidate_tokens_, slots_, bank_states_, bank_mask_: self._planner_features(
                candidate_tokens_,
                slots_,
                bank_states_,
                bank_mask_,
            ),
            candidate_tokens,
            slots,
            bank_states,
            bank_mask,
        )
        planner_policy_logits = _run_function_with_checkpoint(
            self.activation_checkpointing,
            lambda planner_tokens_, candidate_tokens_, routing_weights_, domain_states_: self._planner_policy_logits(
                planner_tokens_,
                candidate_tokens_,
                routing_weights_,
                domain_states_,
            ),
            planner_tokens,
            candidate_tokens,
            routing_weights,
            domain_states,
        )
        planner_q_logits, planner_q = _run_function_with_checkpoint(
            self.activation_checkpointing,
            lambda planner_tokens_, candidate_tokens_, routing_weights_, domain_states_: self._planner_q_outputs(
                planner_tokens_,
                candidate_tokens_,
                routing_weights_,
                domain_states_,
            ),
            planner_tokens,
            candidate_tokens,
            routing_weights,
            domain_states,
        )
        (
            planner_q_component_logits,
            planner_q_components,
            planner_objective_q,
        ) = _run_function_with_checkpoint(
            self.activation_checkpointing,
            lambda planner_tokens_, candidate_tokens_, routing_weights_, domain_states_: self._planner_objective_q_outputs(
                planner_tokens_,
                candidate_tokens_,
                routing_weights_,
                domain_states_,
                objective_context=objective_context,
            ),
            planner_tokens,
            candidate_tokens,
            routing_weights,
            domain_states,
        )
        planner_q_bias = self._normalize_action_values(planner_q, planner_action_mask)
        planner_objective_q_bias = self._normalize_action_values(planner_objective_q, planner_action_mask)
        planner_risk_q = planner_q_components[..., list(RISK_OBJECTIVE_HEAD_INDICES)].mean(dim=-1)
        planner_risk_bias = self._normalize_action_values(planner_risk_q, planner_action_mask)
        policy_logits = (
            base_policy_logits
            + self.internal_planner_blend * planner_policy_logits
            + self.internal_planner_q_blend * planner_q_bias
            + self.internal_planner_objective_q_blend * planner_objective_q_bias
            + self.internal_planner_risk_blend * planner_risk_bias
        )

        del pooled
        value_logits, value, value_component_logits, value_components, objective_value = _run_function_with_checkpoint(
            self.activation_checkpointing,
            lambda slots_, routing_weights_, domain_states_: self._value_outputs(
                slots_,
                routing_weights=routing_weights_,
                objective_context=objective_context,
                domain_states=domain_states_,
            ),
            slots,
            routing_weights,
            domain_states,
        )
        return (
            policy_logits,
            value_logits,
            value,
            latent_policy_logits,
            latent_action_embeddings,
            value_component_logits,
            value_components,
            objective_value,
            planner_q_logits,
            planner_q,
            planner_objective_q,
            planner_q_component_logits,
            planner_q_components,
        )

    def semantic_forward(
        self,
        hidden_state: torch.Tensor,
        *,
        decision_domain: torch.Tensor | None = None,
        objective_context: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        slots = self._slot_states(hidden_state, latent_type_id=1)
        bank_states, bank_mask = self._bank_states(slots)
        def _domain_features(slots_: torch.Tensor, bank_states_: torch.Tensor, bank_mask_: torch.Tensor):
            return self._domain_states(
                slots_,
                bank_states=bank_states_,
                bank_mask=bank_mask_,
            )

        pooled, _domain_states, domain_logits = _run_function_with_checkpoint(
            self.activation_checkpointing,
            _domain_features,
            slots,
            bank_states,
            bank_mask,
        )
        domain_weights = torch.softmax(domain_logits, dim=-1)
        routing_weights = self._routing_weights(domain_weights=domain_weights, decision_domain=decision_domain)
        semantic_policy_logits = self.semantic_policy_head(pooled)
        value_logits, value, value_component_logits, value_components, objective_value = _run_function_with_checkpoint(
            self.activation_checkpointing,
            lambda slots_, routing_weights_, domain_states_: self._value_outputs(
                slots_,
                routing_weights=routing_weights_,
                objective_context=objective_context,
                domain_states=domain_states_,
            ),
            slots,
            routing_weights,
            _domain_states,
        )
        return (
            semantic_policy_logits,
            value_logits,
            value,
            value_component_logits,
            value_components,
            objective_value,
        )
