"""Unified search-free omni-attention policy for STS2 online inference."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from gymnasium import spaces

from sb3_contrib.common.maskable.distributions import MaskableCategoricalDistribution
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.type_aliases import Schedule

from sts2_env.attention_blocks import CandidateDecoderBlock, CrossAttentionBlock, EntityPooling, RelationBias, TransformerEncoderBlock
from sts2_env.aux_targets import (
    ENEMY_STATE_SLOT_COUNT,
    NUM_BUILD_HEADS,
    NUM_ENEMY_STATE_FIELDS,
    NUM_OBJECTIVE_HEADS,
    NUM_ROUTE_HEADS,
    NUM_SELECTION_HEADS,
    NUM_TRAIT_HEADS,
    NUM_TRANSITION_HEADS,
)
from sts2_env.observation_v3 import (
    ENTITY_HASH_BUCKETS,
    MAX_ACTIONS,
    MAX_OWNER_ID,
    MAX_ORDER_ID,
    MAX_ROLE_ID,
    MAX_ZONE_ID,
    NUM_TOKEN_TYPES,
    POWER_ID_BUCKETS,
    TOKEN_FEAT_DIM,
    TOKEN_NUMERIC_DIM,
    TOKEN_TEXT_DIM,
    TOKEN_ROLE_TO_ID,
    TOKEN_ZONE_TO_ID,
)

# Phase 6.3: shared constants for power-bucket bias plumbing.
_POWER_SLOT_ROLE_ID = TOKEN_ROLE_TO_ID.get("POWER_SLOT", 0)
# Phase 8 Tier 1: shared constants for history-card-bucket bias plumbing.
# ENTITY_HASH_BUCKETS matches the hash space used by _stable_card_bucket
# inside action_history.py, so HISTORY tokens' entity_ids land in the
# same bucket namespace as hand/deck/discard card entity_ids.
from sts2_env.observation_v3 import ENTITY_HASH_BUCKETS

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

DEFAULT_POLICY_CLASS_PATH = "sts2_env.omni_attention_policy.STS2OmniAttentionPolicy"
ATTENTION_ARCHITECTURE_VERSION = "omni_attention_v1_frozen"
GLOBAL_AUX_HEAD_NAMES = ("objective", "transition", "traits")
CANDIDATE_AUX_HEAD_NAMES = (
    "candidate_objective",
    "candidate_transition",
    "candidate_traits",
    "candidate_build",
    "candidate_selection",
    "candidate_route",
    # Phase 8 Tier 2:
    "candidate_causality",
)

# Phase 6.2: dedicated "powers" bank. Before this, POWER_SLOT tokens were
# routed via role_id "POWER_SLOT" which fell into no existing bank and
# thus got dropped from candidate cross-attention. Giving them their own
# bank lets the top-k router explicitly opt-in to buff/debuff context per
# candidate (attack candidates → enemy+powers; defense candidates →
# enemy_intent+powers+support; map candidates → route, skip powers).
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
    # v3: dedicated bank for POWER_SLOT_* and CARD_KEYWORD_SLOT tokens.
    # Keeps enemy bank from having to carry both core/intent + every buff
    # stacked on every enemy at tight token budget.
    "powers": ("POWER_SLOT", "CARD_KEYWORD"),
    # v4 (Phase 8 Tier 1): dedicated bank for HISTORY_STEP_DETAIL and
    # HISTORY_TURN_SUMMARY tokens. Lets the top-k router opt in to
    # "what did I just do" context per candidate — attack candidates
    # often need runtime+enemy+powers+history (last 2 turns' scaling);
    # map candidates often need route+history (did I just buy a
    # strategy-defining card at the last shop).
    "history": ("HISTORY",),
}
_WORLD_BANK_ZONE_NAMES = {
    "runtime": ("WORLD", "PLAYER", "HAND", "DRAW", "DISCARD", "EXHAUST", "PLAY"),
    "support": ("RELIC", "POTION"),
    "enemy": ("ENEMY",),
    "build": ("DECK", "REWARD", "SHOP", "UPGRADE"),
    "route": ("ROUTE",),
    # No zone entry for powers — role-only filter. Bank-mask uses role_id
    # OR zone_id; adding PLAYER/ENEMY/HAND here would also sweep
    # PLAYER_SURVIVAL / ENEMY_CORE / HAND_CARD into the powers bank,
    # which defeats the purpose of a dedicated bank. Role POWER_SLOT /
    # CARD_KEYWORD are exclusive to the new token types, so a role-only
    # filter catches them precisely.
    "powers": (),
    # Same role-only filter for history. HISTORY zone is set on
    # history tokens for documentation/probe ease but we don't want
    # any other token sweeping in via zone.
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


class EntityTokenEmbedder(nn.Module):
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
    ):
        super().__init__()
        self.num_token_types = num_token_types
        self.max_owner_id = max_owner_id
        self.max_role_id = max_role_id
        self.max_zone_id = max_zone_id
        self.max_order_id = max_order_id
        self.entity_hash_buckets = entity_hash_buckets
        half = d_model // 2
        self.numeric_width = half
        self.text_width = d_model - half
        self.numeric_proj = (
            nn.Sequential(nn.Linear(TOKEN_NUMERIC_DIM, self.numeric_width), nn.GELU(), nn.Linear(self.numeric_width, self.numeric_width))
            if use_internal_numeric_proj
            else None
        )
        self.text_proj = (
            nn.Sequential(nn.Linear(TOKEN_TEXT_DIM, self.text_width), nn.GELU(), nn.Linear(self.text_width, self.text_width))
            if use_internal_text_proj
            else None
        )
        self.type_embedding = nn.Embedding(num_token_types, d_model)
        self.role_embedding = nn.Embedding(max_role_id + 1, d_model)
        self.owner_embedding = nn.Embedding(max_owner_id + 1, d_model)
        self.zone_embedding = nn.Embedding(max_zone_id + 1, d_model)
        self.order_embedding = nn.Embedding(max_order_id + 1, d_model)
        self.entity_embedding = nn.Embedding(entity_hash_buckets, d_model)
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
                raise RuntimeError("EntityTokenEmbedder requires projected_numeric when use_internal_numeric_proj=False.")
            projected_numeric = self.numeric_proj(numeric)
        if projected_text is None:
            if self.text_proj is None:
                raise RuntimeError("EntityTokenEmbedder requires projected_text when use_internal_text_proj=False.")
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


class STS2OmniAttentionPolicy(MaskableActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Discrete,
        lr_schedule: Schedule,
        *,
        d_model: int = 256,
        n_heads: int = 8,
        ffn_dim: int = 1024,
        world_layers: int = 6,
        local_layers: int = 1,
        decoder_layers: int = 2,
        candidate_set_layers: int = 1,
        world_bank_top_k: int = 3,
        dropout: float = 0.0,
        **kwargs,
    ):
        kwargs.pop("features_extractor_class", None)
        kwargs.pop("features_extractor_kwargs", None)
        kwargs.pop("net_arch", None)

        self._d_model = d_model
        self._n_heads = n_heads
        self._ffn_dim = ffn_dim
        self._world_layers = world_layers
        self._local_layers = local_layers
        self._decoder_layers = decoder_layers
        self._candidate_set_layers = candidate_set_layers
        self._world_bank_top_k = max(1, min(int(world_bank_top_k), len(WORLD_BANK_NAMES)))
        self._dropout = dropout

        super().__init__(observation_space, action_space, lr_schedule, net_arch=[], **kwargs)

    def _build_mlp_extractor(self) -> None:
        self.shared_numeric_width = self._d_model // 2
        self.shared_text_width = self._d_model - self.shared_numeric_width
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
            d_model=self._d_model,
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
            d_model=self._d_model,
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
            d_model=self._d_model,
            num_token_types=NUM_TOKEN_TYPES,
            max_owner_id=MAX_OWNER_ID,
            max_role_id=MAX_ROLE_ID,
            max_zone_id=MAX_ZONE_ID,
            max_order_id=MAX_ORDER_ID,
            entity_hash_buckets=ENTITY_HASH_BUCKETS,
            use_internal_numeric_proj=False,
            use_internal_text_proj=False,
        )

        self.world_relation_bias = RelationBias(n_heads=self._n_heads, **_RELATION_BIAS_KW)
        self.local_relation_bias = RelationBias(n_heads=self._n_heads, **_RELATION_BIAS_KW)
        self.query_local_relation_bias = RelationBias(n_heads=self._n_heads, **_RELATION_BIAS_KW)
        self.query_world_relation_bias = RelationBias(n_heads=self._n_heads, **_RELATION_BIAS_KW)
        self.candidate_set_relation_bias = RelationBias(n_heads=self._n_heads, **_RELATION_BIAS_KW)
        self.world_bank_relation_bias = nn.ModuleList(
            [
                RelationBias(n_heads=self._n_heads, **_RELATION_BIAS_KW)
                for _ in WORLD_BANK_NAMES
            ]
        )

        self.world_blocks = nn.ModuleList(
            [TransformerEncoderBlock(self._d_model, self._n_heads, self._ffn_dim, dropout=self._dropout) for _ in range(self._world_layers)]
        )
        self.local_blocks = nn.ModuleList(
            [TransformerEncoderBlock(self._d_model, self._n_heads, self._ffn_dim, dropout=self._dropout) for _ in range(self._local_layers)]
        )
        self.query_local_bridge = CrossAttentionBlock(self._d_model, self._n_heads, self._ffn_dim, dropout=self._dropout)
        self.decoder_blocks = nn.ModuleList(
            [CandidateDecoderBlock(self._d_model, self._n_heads, self._ffn_dim, dropout=self._dropout) for _ in range(self._decoder_layers)]
        )
        self.world_bank_cross_blocks = nn.ModuleList(
            [
                nn.ModuleList(
                    [CrossAttentionBlock(self._d_model, self._n_heads, self._ffn_dim, dropout=self._dropout) for _ in WORLD_BANK_NAMES]
                )
                for _ in range(self._decoder_layers)
            ]
        )
        self.candidate_set_blocks = nn.ModuleList(
            [TransformerEncoderBlock(self._d_model, self._n_heads, self._ffn_dim, dropout=self._dropout) for _ in range(self._candidate_set_layers)]
        )

        self.world_pool = EntityPooling(self._d_model, self._n_heads, dropout=self._dropout)
        self.local_pool = EntityPooling(self._d_model, self._n_heads, dropout=self._dropout)
        self.world_bank_poolers = nn.ModuleList([EntityPooling(self._d_model, self._n_heads, dropout=self._dropout) for _ in WORLD_BANK_NAMES])
        self.world_bank_router_q = nn.Sequential(nn.LayerNorm(self._d_model), nn.Linear(self._d_model, self._d_model))
        self.world_bank_router_k = nn.Sequential(nn.LayerNorm(self._d_model), nn.Linear(self._d_model, self._d_model))
        self.world_bank_router_bias = nn.Parameter(torch.zeros(len(WORLD_BANK_NAMES)))

        self.policy_head = nn.Sequential(nn.LayerNorm(self._d_model), nn.Linear(self._d_model, self._d_model), nn.GELU(), nn.Linear(self._d_model, 1))
        self.value_head = nn.Sequential(nn.LayerNorm(self._d_model), nn.Linear(self._d_model, self._d_model), nn.GELU(), nn.Linear(self._d_model, 1))
        self.objective_head = nn.Sequential(nn.LayerNorm(self._d_model), nn.Linear(self._d_model, NUM_OBJECTIVE_HEADS))
        self.transition_head = nn.Sequential(nn.LayerNorm(self._d_model), nn.Linear(self._d_model, NUM_TRANSITION_HEADS))
        self.trait_head = nn.Sequential(nn.LayerNorm(self._d_model), nn.Linear(self._d_model, NUM_TRAIT_HEADS))
        self.candidate_objective_head = nn.Sequential(nn.LayerNorm(self._d_model), nn.Linear(self._d_model, NUM_OBJECTIVE_HEADS))
        self.candidate_transition_head = nn.Sequential(nn.LayerNorm(self._d_model), nn.Linear(self._d_model, NUM_TRANSITION_HEADS))
        self.candidate_trait_head = nn.Sequential(nn.LayerNorm(self._d_model), nn.Linear(self._d_model, NUM_TRAIT_HEADS))
        self.candidate_build_head = nn.Sequential(nn.LayerNorm(self._d_model), nn.Linear(self._d_model, NUM_BUILD_HEADS))
        self.candidate_selection_head = nn.Sequential(nn.LayerNorm(self._d_model), nn.Linear(self._d_model, NUM_SELECTION_HEADS))
        self.candidate_route_head = nn.Sequential(nn.LayerNorm(self._d_model), nn.Linear(self._d_model, NUM_ROUTE_HEADS))
        self.enemy_state_head = nn.Sequential(
            nn.LayerNorm(self._d_model),
            nn.Linear(self._d_model, ENEMY_STATE_SLOT_COUNT * NUM_ENEMY_STATE_FIELDS),
        )
        # Phase 8 Tier 2: action_causality head.
        # Per-candidate prediction of the 8-d state delta that candidate
        # would produce if played (damage_dealt / block_gained /
        # self_hp_loss / draw_delta / energy_delta / strength_delta /
        # dex_delta / vuln_applied). Target is self-supervised from the
        # actual post-pre delta at each step, masked to the chosen
        # candidate's row (same pattern as candidate_objective).
        # Distinguishes "this card deals 6 base damage" from "this card
        # deals 6 damage in current buff context" — pure prediction,
        # no rules.
        from sts2_env.action_history import NUM_CAUSALITY_HEADS
        self.candidate_causality_head = nn.Sequential(
            nn.LayerNorm(self._d_model),
            nn.Linear(self._d_model, NUM_CAUSALITY_HEADS),
        )

        # Pre-register constant bank role/zone ID tensors as buffers to avoid
        # re-creating them via torch.as_tensor on every forward pass.
        for bank_name in WORLD_BANK_NAMES:
            role_ids = WORLD_BANK_ROLE_IDS.get(bank_name, ())
            zone_ids = WORLD_BANK_ZONE_IDS.get(bank_name, ())
            self.register_buffer(f"_bank_role_ids_{bank_name}", torch.as_tensor(role_ids, dtype=torch.long), persistent=False)
            self.register_buffer(f"_bank_zone_ids_{bank_name}", torch.as_tensor(zone_ids, dtype=torch.long), persistent=False)

        self.mlp_extractor = _DummyExtractor()

    def _build(self, lr_schedule) -> None:
        self._build_mlp_extractor()
        self.action_dist = MaskableCategoricalDistribution(self.action_space.n)
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def forward(self, obs, deterministic=False, action_masks=None):
        logits, world_pool, _ = self._forward_policy(obs)
        masked_logits, masks = self._mask_logits(logits, action_masks=action_masks if action_masks is not None else obs.get("action_mask"))
        distribution = self.action_dist.proba_distribution(action_logits=masked_logits)
        values = self._sanitize_tensor(self.value_head(world_pool))
        if deterministic:
            actions = masked_logits.argmax(dim=1)
        else:
            actions = distribution.get_actions(deterministic=False)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob

    def evaluate_actions(self, obs, actions, action_masks=None):
        logits, world_pool, _ = self._forward_policy(obs)
        distribution = self._build_distribution(logits, action_masks=action_masks if action_masks is not None else obs.get("action_mask"), actions=actions)
        values = self._sanitize_tensor(self.value_head(world_pool))
        return values, distribution.log_prob(actions), distribution.entropy()

    def evaluate_actions_with_aux(self, obs, actions, action_masks=None):
        logits, world_pool, candidate_x = self._forward_policy(obs)
        distribution = self._build_distribution(logits, action_masks=action_masks if action_masks is not None else obs.get("action_mask"), actions=actions)
        values = self._sanitize_tensor(self.value_head(world_pool))
        aux_outputs = self._aux_from_embeddings(world_pool, candidate_x)
        return values, distribution.log_prob(actions), distribution.entropy(), aux_outputs

    def get_distribution(self, obs, action_masks=None):
        logits, _, _ = self._forward_policy(obs)
        masked_logits, _ = self._mask_logits(logits, action_masks=action_masks if action_masks is not None else obs.get("action_mask"))
        return self.action_dist.proba_distribution(action_logits=masked_logits)

    def predict_values(self, obs):
        _, world_pool, _ = self._forward_policy(obs)
        return self._sanitize_tensor(self.value_head(world_pool))

    def score_action_logits(self, obs, action_masks=None):
        logits, _, _ = self._forward_policy(obs)
        masked_logits, _ = self._mask_logits(logits, action_masks=action_masks if action_masks is not None else obs.get("action_mask"))
        return masked_logits

    def forward_aux_heads(self, obs):
        _, world_pool, candidate_x = self._forward_policy(obs)
        return self._aux_from_embeddings(world_pool, candidate_x)

    def _predict(self, observation, deterministic=False, action_masks=None):
        logits, _, _ = self._forward_policy(observation)
        masked_logits, _ = self._mask_logits(logits, action_masks=action_masks if action_masks is not None else observation.get("action_mask"))
        if deterministic:
            return masked_logits.argmax(dim=1)
        return self.action_dist.proba_distribution(action_logits=masked_logits).get_actions(deterministic=False)

    def _forward_policy(self, obs, *, return_debug: bool = False):
        world_tokens = self._float_tensor(obs["world_tokens"])
        world_mask = self._bool_tensor(obs["world_token_mask"])
        world_type_ids = self._long_tensor(obs["world_token_type_ids"])
        world_role_ids = self._long_tensor(obs["world_token_role_ids"])
        world_owner_ids = self._long_tensor(obs["world_entity_owner_ids"])
        world_entity_ids = self._long_tensor(obs["world_token_entity_ids"])
        world_zone_ids = self._long_tensor(obs["world_token_zone_ids"])
        world_order_ids = self._long_tensor(obs["world_token_order_ids"])

        candidate_query_tokens = self._float_tensor(obs["candidate_query_tokens"])
        candidate_query_type_ids = self._long_tensor(obs["candidate_query_type_ids"])
        candidate_query_role_ids = self._long_tensor(obs["candidate_query_role_ids"])
        candidate_query_owner_ids = self._long_tensor(obs["candidate_query_owner_ids"])
        candidate_query_entity_ids = self._long_tensor(obs["candidate_query_entity_ids"])
        candidate_query_zone_ids = self._long_tensor(obs["candidate_query_zone_ids"])
        candidate_query_order_ids = self._long_tensor(obs["candidate_query_order_ids"])
        candidate_query_target_owner_ids = self._long_tensor(obs["candidate_query_target_owner_ids"])
        candidate_query_target_entity_ids = self._long_tensor(obs["candidate_query_target_entity_ids"])
        candidate_mask = self._bool_tensor(obs["action_mask"])

        candidate_local_tokens = self._float_tensor(obs["candidate_local_tokens"])
        candidate_local_masks = self._bool_tensor(obs["candidate_local_masks"])
        candidate_local_type_ids = self._long_tensor(obs["candidate_local_type_ids"])
        candidate_local_role_ids = self._long_tensor(obs["candidate_local_role_ids"])
        candidate_local_owner_ids = self._long_tensor(obs["candidate_local_owner_ids"])
        candidate_local_entity_ids = self._long_tensor(obs["candidate_local_entity_ids"])
        candidate_local_zone_ids = self._long_tensor(obs["candidate_local_zone_ids"])
        candidate_local_order_ids = self._long_tensor(obs["candidate_local_order_ids"])

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
            world_x = block(world_x, mask=world_mask, attn_bias=world_bias)
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
            local_x = block(local_x, mask=flat_local_masks, attn_bias=local_bias)
        local_pool = self.local_pool(local_x, mask=flat_local_masks).reshape(batch_size, action_count, self._d_model)

        flat_query_x = query_x.reshape(batch_size * action_count, 1, self._d_model)
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
        bridged_query_x = self.query_local_bridge(
            flat_query_x,
            local_x,
            query_mask=flat_query_masks,
            memory_mask=flat_local_masks,
            attn_bias=query_local_bias,
        )
        has_local_context = flat_local_masks.any(dim=1, keepdim=True).unsqueeze(-1)
        bridged_query_x = torch.where(has_local_context, bridged_query_x, flat_query_x)
        candidate_x = bridged_query_x.reshape(batch_size, action_count, self._d_model) + local_pool
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
        routing_debug = None
        for layer_index, block in enumerate(self.decoder_blocks):
            candidate_x = block.self_block(candidate_x, mask=candidate_mask, attn_bias=set_bias)
            candidate_x, layer_debug = self._apply_banked_world_cross_attention(
                layer_index=layer_index,
                candidate_x=candidate_x,
                candidate_mask=candidate_mask,
                world_x=world_x,
                world_bank_masks=world_bank_masks,
                world_bank_summaries=world_bank_summaries,
                world_bank_biases=world_bank_biases,
            )
            if return_debug:
                routing_debug = layer_debug
        for block in self.candidate_set_blocks:
            candidate_x = block(candidate_x, mask=candidate_mask, attn_bias=set_bias)

        logits = self._sanitize_logits(self.policy_head(candidate_x).squeeze(-1))
        if return_debug:
            if routing_debug is None:
                routing_debug = {
                    "bank_names": list(WORLD_BANK_NAMES),
                    "bank_available": world_bank_masks.any(dim=-1),
                    "bank_selected": torch.zeros((batch_size, action_count, len(WORLD_BANK_NAMES)), dtype=torch.bool, device=candidate_x.device),
                    "bank_weights": torch.zeros((batch_size, action_count, len(WORLD_BANK_NAMES)), dtype=candidate_x.dtype, device=candidate_x.device),
                }
            return logits, world_pool, candidate_x, routing_debug
        return logits, world_pool, candidate_x

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
        router_logits = torch.einsum("bad,bkd->bak", router_q, router_k) / math.sqrt(float(self._d_model))
        router_logits = router_logits + self.world_bank_router_bias.view(1, 1, -1)
        router_logits = router_logits.masked_fill(~bank_available.unsqueeze(1), -1e4)

        top_k = min(self._world_bank_top_k, len(WORLD_BANK_NAMES))
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
        return bank_weights, selected, bank_available

    def _apply_banked_world_cross_attention(
        self,
        *,
        layer_index: int,
        candidate_x,
        candidate_mask,
        world_x,
        world_bank_masks,
        world_bank_summaries,
        world_bank_biases,
    ):
        bank_weights, bank_selected, bank_available = self._compute_world_bank_routing(candidate_x, candidate_mask, world_bank_summaries, world_bank_masks)
        fused_delta = torch.zeros_like(candidate_x)
        for bank_index, bank_name in enumerate(WORLD_BANK_NAMES):
            del bank_name
            bank_query_mask = bank_selected[..., bank_index]
            if not bool(bank_query_mask.any().item()):
                continue
            bank_output = self.world_bank_cross_blocks[layer_index][bank_index](
                candidate_x,
                world_x,
                query_mask=bank_query_mask,
                memory_mask=world_bank_masks[:, bank_index, :],
                attn_bias=world_bank_biases[bank_index],
            )
            bank_output = torch.where(bank_query_mask.unsqueeze(-1), bank_output, candidate_x)
            fused_delta = fused_delta + bank_weights[..., bank_index].unsqueeze(-1) * (bank_output - candidate_x)
        return candidate_x + fused_delta, {
            "bank_names": list(WORLD_BANK_NAMES),
            "bank_available": bank_available,
            "bank_selected": bank_selected,
            "bank_weights": bank_weights,
        }

    @staticmethod
    def _project_shared_modal_trunk_with_reuse(trunk, feature_batches, *, output_dim):
        flattened_batches = [batch.reshape(-1, batch.shape[-1]) for batch in feature_batches]
        total_rows = sum(batch.shape[0] for batch in flattened_batches)
        if total_rows == 0:
            return [batch.new_zeros(*batch.shape[:-1], output_dim) for batch in feature_batches]

        merged_features = torch.cat(flattened_batches, dim=0)
        unique_features, inverse_indices = torch.unique(merged_features, dim=0, return_inverse=True)
        projected_unique = trunk(unique_features)
        projected_merged = projected_unique.index_select(0, inverse_indices)

        projected_batches = []
        row_offset = 0
        for batch, flattened in zip(feature_batches, flattened_batches):
            row_count = flattened.shape[0]
            projected_batches.append(projected_merged[row_offset : row_offset + row_count].reshape(*batch.shape[:-1], output_dim))
            row_offset += row_count
        return projected_batches

    def _aux_from_embeddings(self, world_pool, candidate_x):
        enemy_state_flat = self.enemy_state_head(world_pool)
        enemy_state = enemy_state_flat.view(
            enemy_state_flat.shape[0], ENEMY_STATE_SLOT_COUNT, NUM_ENEMY_STATE_FIELDS
        )
        return {
            "objective": self.objective_head(world_pool),
            "transition": self.transition_head(world_pool),
            "traits": self.trait_head(world_pool),
            "candidate_objective": self.candidate_objective_head(candidate_x),
            "candidate_transition": self.candidate_transition_head(candidate_x),
            "candidate_traits": self.candidate_trait_head(candidate_x),
            "candidate_build": self.candidate_build_head(candidate_x),
            "candidate_selection": self.candidate_selection_head(candidate_x),
            "candidate_route": self.candidate_route_head(candidate_x),
            "enemy_state": enemy_state,
            # Phase 8 Tier 2: per-candidate causality prediction
            # (batch, num_candidates, NUM_CAUSALITY_HEADS). Loss
            # computed at the chosen-candidate row only, via the same
            # masked-regression pattern as candidate_objective.
            "candidate_causality": self.candidate_causality_head(candidate_x),
        }

    def forward_world_bank_routing(self, obs):
        _, _, _, debug = self._forward_policy(obs, return_debug=True)
        return {
            "architecture_version": ATTENTION_ARCHITECTURE_VERSION,
            "bank_names": list(debug["bank_names"]),
            "bank_available": debug["bank_available"],
            "bank_selected": debug["bank_selected"],
            "bank_weights": debug["bank_weights"],
            "world_bank_top_k": int(self._world_bank_top_k),
        }

    def architecture_spec(self) -> dict[str, object]:
        return {
            "architecture_version": ATTENTION_ARCHITECTURE_VERSION,
            "world_banks": list(WORLD_BANK_NAMES),
            "world_bank_top_k": int(self._world_bank_top_k),
            "shared_trunks": {"numeric": True, "text": True},
            "query_local_bridge": True,
            "banked_world_cross_attention": True,
            "global_aux_heads": list(GLOBAL_AUX_HEAD_NAMES),
            "candidate_aux_heads": list(CANDIDATE_AUX_HEAD_NAMES),
        }

    def _build_distribution(self, logits, action_masks=None, actions=None):
        masked_logits = self._sanitize_logits(logits)
        if action_masks is not None:
            masks = self._normalize_action_masks(action_masks, masked_logits, actions=actions)
            masked_logits = self._apply_action_mask(masked_logits, masks)
        return self.action_dist.proba_distribution(action_logits=masked_logits)

    def _mask_logits(self, logits, action_masks=None, actions=None):
        masked_logits = self._sanitize_logits(logits)
        masks = None
        if action_masks is not None:
            masks = self._normalize_action_masks(action_masks, masked_logits, actions=actions)
            masked_logits = self._apply_action_mask(masked_logits, masks)
        return masked_logits, masks

    @staticmethod
    def _sanitize_tensor(tensor):
        return torch.nan_to_num(tensor, nan=0.0, posinf=1e4, neginf=-1e4)

    def _sanitize_logits(self, logits):
        logits = self._sanitize_tensor(logits)
        row_max = logits.max(dim=-1, keepdim=True).values
        logits = logits - torch.nan_to_num(row_max, nan=0.0, posinf=0.0, neginf=0.0)
        return logits.clamp(min=-50.0)

    @staticmethod
    def _normalize_action_masks(action_masks, logits, actions=None):
        masks = torch.as_tensor(action_masks, dtype=torch.bool, device=logits.device).reshape(logits.shape)
        no_valid = ~masks.any(dim=1)
        if no_valid.any():
            masks = masks.clone()
            if actions is not None:
                chosen = torch.as_tensor(actions, dtype=torch.long, device=logits.device).reshape(-1)
                chosen = chosen.clamp(min=0, max=logits.shape[1] - 1)
                rows = no_valid.nonzero(as_tuple=False).reshape(-1)
                masks[rows, chosen[rows]] = True
            else:
                masks[no_valid, 0] = True
        return masks

    def _apply_action_mask(self, logits, masks):
        return logits.masked_fill(~masks, -50.0)

    @staticmethod
    def _float_tensor(value):
        tensor = torch.as_tensor(value, dtype=torch.float32)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        return tensor

    @staticmethod
    def _long_tensor(value):
        tensor = torch.as_tensor(value, dtype=torch.long)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        return tensor

    @staticmethod
    def _bool_tensor(value):
        tensor = torch.as_tensor(value, dtype=torch.bool)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        return tensor

    def _get_constructor_parameters(self):
        data = super()._get_constructor_parameters()
        data.update(
            d_model=self._d_model,
            n_heads=self._n_heads,
            ffn_dim=self._ffn_dim,
            world_layers=self._world_layers,
            local_layers=self._local_layers,
            decoder_layers=self._decoder_layers,
            candidate_set_layers=self._candidate_set_layers,
            world_bank_top_k=self._world_bank_top_k,
            dropout=self._dropout,
        )
        data.pop("features_extractor_class", None)
        data.pop("features_extractor_kwargs", None)
        return data

    def extract_features(self, obs, features_extractor=None):
        return obs


class _DummyExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.latent_dim_pi = 1
        self.latent_dim_vf = 1

    def forward(self, x):
        return x, x

    def forward_actor(self, x):
        return x

    def forward_critic(self, x):
        return x
