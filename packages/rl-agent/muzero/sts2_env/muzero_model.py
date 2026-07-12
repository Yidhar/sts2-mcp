"""MuZero-style model-based RL networks for Slay the Spire 2.

This module implements the core networks for a MuZero agent:
  - RepresentationNetwork: encodes full observations to hidden state
  - DynamicsNetwork: predicts next hidden state and immediate reward
  - PredictionNetwork: predicts policy and value from hidden state
  - ActionEncoder: encodes action features + text to action embedding
  - MuZeroNetwork: main wrapper combining all networks

Design follows AlphaZero/MuZero architectures with unified hidden state
representation and support-based value estimation.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn

from muzero.sts2_env._muzero_action_rollout import (
    DEFAULT_ACTION_ROLLOUT_BUCKETS,
    RISK_OBJECTIVE_HEAD_INDICES,
    ActionRolloutPlannerMixin,
    ActionRolloutPlannerOutput,
)
from muzero.sts2_env._muzero_dense_ops import (
    _domain_masks_from_obs,
    _masked_mean,
    _safe_cross_attn,
    _safe_pool,
    _safe_self_attn,
)
from muzero.sts2_env.semantic_rollout import (
    SEMANTIC_ROLLOUT_FEAT_DIM,
    SEMANTIC_ROLLOUT_SIZE,
)
from muzero.sts2_env.token_memory import (
    TokenDynamicsNetwork,
    TokenFutureWorldBankHead,
    TokenLatentProjector,
    TokenMemoryEncoder,
    TokenMemoryEncoderOutput,
    TokenPredictionNetwork,
    TokenTransitionSurfaceHead,
    infer_token_decision_domain,
)
from muzero.sts2_env.value_support import (
    scalar_to_support,
    support_tensor_to_scalar,
    support_to_scalar,
)
from sts2_env.objective_heads import (
    HEAD_HP_PRESERVATION,
    HEAD_SURVIVAL,
    NUM_OBJECTIVE_HEADS,
    scalarize_objective_components_torch,
)
from sts2_env.observation_v2 import (
    ACTION_FEAT_DIM,
    CARD_FEAT_DIM,
    DECK_FEAT_DIM,
    ENEMY_FEAT_DIM,
    MAX_ACTIONS,
    MAX_DECK,
    MAX_ENEMIES,
    MAX_HAND,
    MAX_POTIONS,
    MAX_RELICS,
    NUM_DOMAINS,
    NUM_PHASES,
    OBJECTIVE_DIM,
    POWER_DIM,
    RELIC_SIGNAL_DIM,
    ROUTE_NODE_FEAT_DIM,
    ROUTE_SUMMARY_DIM,
    RUN_MEMORY_DIM,
    SCALAR_DIM,
    SEM_ACTION_FEAT_DIM,
)
from sts2_env.text_encoder import TEXT_DIM

# ============================================================================
# Value Support Functions (Categorical value representation)
# ============================================================================



# ============================================================================
# Helper Functions (reimplement from model.py patterns)
# ============================================================================



# ============================================================================
# ActionEncoder: encode action features + text to embeddings
# ============================================================================

class DomainActionEncoder(nn.Module):
    """Generic action encoder for combat/build domains."""

    def __init__(self, *, action_dim: int, text_proj_dim: int = 32):
        super().__init__()
        self.output_dim = action_dim
        self.action_text_proj = nn.Linear(TEXT_DIM, text_proj_dim)
        self.semantic_action_text_proj = nn.Linear(TEXT_DIM, text_proj_dim)
        self.net = nn.Sequential(
            nn.Linear(ACTION_FEAT_DIM + SEM_ACTION_FEAT_DIM + (2 * text_proj_dim), action_dim),
            nn.ReLU(),
            nn.Linear(action_dim, action_dim),
        )

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        action_text = self.action_text_proj(obs["action_text"])
        semantic_text = self.semantic_action_text_proj(obs["semantic_action_text"])
        combined = torch.cat([obs["actions"], obs["semantic_actions"], action_text, semantic_text], dim=-1)
        return self.net(combined)


class RouteActionEncoder(nn.Module):
    """Route encoder that consumes future-subtree summaries/tokens."""

    def __init__(self, *, action_dim: int, text_proj_dim: int = 32, node_dim: int = 32, n_heads: int = 2):
        super().__init__()
        self.output_dim = action_dim

        self.action_text_proj = nn.Linear(TEXT_DIM, text_proj_dim)
        self.semantic_action_text_proj = nn.Linear(TEXT_DIM, text_proj_dim)
        self.generic_net = nn.Sequential(
            nn.Linear(ACTION_FEAT_DIM + SEM_ACTION_FEAT_DIM + (2 * text_proj_dim), action_dim),
            nn.ReLU(),
        )
        self.summary_net = nn.Sequential(
            nn.Linear(ROUTE_SUMMARY_DIM, node_dim),
            nn.ReLU(),
        )
        self.node_net = nn.Sequential(
            nn.Linear(ROUTE_NODE_FEAT_DIM, node_dim),
            nn.ReLU(),
        )
        self.node_attn = nn.MultiheadAttention(node_dim, n_heads, batch_first=True)
        self.node_norm = nn.LayerNorm(node_dim)
        self.node_seed = nn.Parameter(torch.randn(1, 1, node_dim) * 0.02)
        self.node_pool = nn.MultiheadAttention(node_dim, n_heads, batch_first=True)
        self.final_net = nn.Sequential(
            nn.Linear(action_dim + node_dim + node_dim, action_dim),
            nn.ReLU(),
            nn.Linear(action_dim, action_dim),
        )

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        batch_size, action_count, node_count, _ = obs["route_nodes"].shape

        action_text = self.action_text_proj(obs["action_text"])
        semantic_text = self.semantic_action_text_proj(obs["semantic_action_text"])
        generic = self.generic_net(torch.cat([obs["actions"], obs["semantic_actions"], action_text, semantic_text], dim=-1))
        summary = self.summary_net(obs["route_summary"])

        flat_nodes = obs["route_nodes"].reshape(batch_size * action_count, node_count, -1)
        flat_masks = obs["route_node_mask"].reshape(batch_size * action_count, node_count).bool()
        node_emb = self.node_net(flat_nodes)
        node_emb = self.node_norm(node_emb + _safe_self_attn(self.node_attn, node_emb, flat_masks))
        pooled_nodes = _safe_pool(self.node_pool, self.node_seed, node_emb, flat_masks)
        pooled_nodes = pooled_nodes.reshape(batch_size, action_count, -1)

        combined = torch.cat([generic, summary, pooled_nodes], dim=-1)
        return self.final_net(combined)


class ActionEncoder(nn.Module):
    """Hard-routed action encoder for combat/build/route domains."""

    def __init__(
        self,
        action_embed_dim: int = 64,
        text_proj_dim: int = 32,
        route_node_dim: int = 32,
        n_heads: int = 2,
    ):
        super().__init__()
        self.action_embed_dim = action_embed_dim
        self.combat_action_encoder = DomainActionEncoder(
            action_dim=action_embed_dim,
            text_proj_dim=text_proj_dim,
        )
        self.build_action_encoder = DomainActionEncoder(
            action_dim=action_embed_dim,
            text_proj_dim=text_proj_dim,
        )
        self.route_action_encoder = RouteActionEncoder(
            action_dim=action_embed_dim,
            text_proj_dim=text_proj_dim,
            node_dim=route_node_dim,
            n_heads=n_heads,
        )

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        device = obs["actions"].device
        batch_size, action_count, _ = obs["actions"].shape
        output = torch.zeros(batch_size, action_count, self.action_embed_dim, device=device)

        combat_mask, build_mask, route_mask = _domain_masks_from_obs(obs, device)

        if combat_mask.any():
            output[combat_mask] = self.combat_action_encoder({key: value[combat_mask] for key, value in obs.items()})
        if build_mask.any():
            output[build_mask] = self.build_action_encoder({key: value[build_mask] for key, value in obs.items()})
        if route_mask.any():
            output[route_mask] = self.route_action_encoder({key: value[route_mask] for key, value in obs.items()})

        return output


# ============================================================================
# RepresentationNetwork: encode observations to hidden state
# ============================================================================

class RepresentationNetwork(nn.Module):
    """Encode full observation to hidden state via hard-routed domain experts."""

    def __init__(
        self,
        hidden_dim: int = 512,
        embed_dim: int = 64,
        n_heads: int = 4,
        text_proj_dim: int = 32,
    ):
        """Initialize representation network.

        Args:
            hidden_dim: output hidden state dimension (default 512)
            embed_dim: attention embedding dimension (default 64)
            n_heads: number of attention heads (default 4)
            text_proj_dim: text projection dimension (default 32)
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.text_proj_dim = text_proj_dim
        self.shared_dim = 128
        self.route_dim = 48

        # ---- Shared Context ----
        self.context_text_proj = nn.Linear(TEXT_DIM, 48)
        self.relic_text_proj = nn.Linear(TEXT_DIM, text_proj_dim)
        self.potion_text_proj = nn.Linear(TEXT_DIM, text_proj_dim)
        self.relic_enc = nn.Sequential(nn.Linear(text_proj_dim, 16), nn.ReLU())
        self.potion_enc = nn.Sequential(nn.Linear(text_proj_dim, 16), nn.ReLU())
        self.shared_net = nn.Sequential(
            nn.Linear(
                SCALAR_DIM + POWER_DIM + RELIC_SIGNAL_DIM + RUN_MEMORY_DIM + OBJECTIVE_DIM + 48 + 16 + 16 + NUM_DOMAINS,
                96,
            ),
            nn.ReLU(),
            nn.Linear(96, self.shared_dim),
            nn.ReLU(),
        )

        # ---- Combat expert ----
        self.card_text_proj = nn.Linear(TEXT_DIM, text_proj_dim)
        self.enemy_text_proj = nn.Linear(TEXT_DIM, text_proj_dim)
        self.card_enc = nn.Sequential(nn.Linear(CARD_FEAT_DIM + text_proj_dim, embed_dim), nn.ReLU())
        self.enemy_enc = nn.Sequential(nn.Linear(ENEMY_FEAT_DIM + text_proj_dim, embed_dim), nn.ReLU())
        self.card_attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.card_norm = nn.LayerNorm(embed_dim)
        self.enemy_attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.enemy_norm = nn.LayerNorm(embed_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim, n_heads, kdim=embed_dim, vdim=embed_dim, batch_first=True)
        self.cross_norm = nn.LayerNorm(embed_dim)
        self.card_seed = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.enemy_seed = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.card_pool = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.enemy_pool = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.combat_net = nn.Sequential(
            nn.Linear(self.shared_dim + (2 * embed_dim), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # ---- Build expert ----
        self.deck_text_proj = nn.Linear(TEXT_DIM, text_proj_dim)
        self.deck_enc = nn.Sequential(nn.Linear(DECK_FEAT_DIM + text_proj_dim, embed_dim), nn.ReLU())
        self.deck_attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.deck_norm = nn.LayerNorm(embed_dim)
        self.deck_seed = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.deck_pool = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.build_net = nn.Sequential(
            nn.Linear(self.shared_dim + embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # ---- Route expert ----
        self.route_encoder = nn.Sequential(
            nn.Linear(self.shared_dim, self.route_dim),
            nn.ReLU(),
        )
        self.route_net = nn.Sequential(
            nn.Linear(self.shared_dim + self.route_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode observation to hidden state.

        Args:
            obs: observation dict with all standard STS2 observation keys

        Returns:
            hidden state tensor of shape [batch, hidden_dim]
        """
        device = obs["scalars"].device

        # ---- Encode shared context ----
        context = self.context_text_proj(obs["context_text"])
        relics = self.relic_text_proj(obs["relics"])
        potions = self.potion_text_proj(obs["potions"])

        relic_out = _masked_mean(self.relic_enc(relics), obs["relic_mask"].bool())
        potion_out = _masked_mean(self.potion_enc(potions), obs["potion_mask"].bool())

        shared = self.shared_net(torch.cat([
            obs["scalars"],
            obs["player_powers"],
            obs["relic_signals"],
            obs["run_memory"],
            obs["objective_context"],
            context,
            relic_out,
            potion_out,
            obs["decision_domain"],
        ], dim=-1))

        # ---- Combat expert state ----
        card_text = self.card_text_proj(obs["hand_text"])
        enemy_text = self.enemy_text_proj(obs["enemy_text"])

        card_emb = self.card_enc(torch.cat([obs["hand"], card_text], dim=-1))
        enemy_emb = self.enemy_enc(torch.cat([obs["enemies"], enemy_text], dim=-1))

        hand_mask = obs["hand_mask"].bool()
        enemy_mask = obs["enemy_mask"].bool()

        # Self-attention on cards and enemies
        card_emb = self.card_norm(card_emb + _safe_self_attn(self.card_attn, card_emb, hand_mask))
        enemy_emb = self.enemy_norm(enemy_emb + _safe_self_attn(self.enemy_attn, enemy_emb, enemy_mask))

        # Cross-attention: cards interact with enemies
        card_emb = self.cross_norm(
            card_emb + _safe_cross_attn(self.cross_attn, card_emb, enemy_emb, hand_mask, enemy_mask)
        )

        # Pool both
        hand_out = _safe_pool(self.card_pool, self.card_seed, card_emb, hand_mask)
        enemy_out = _safe_pool(self.enemy_pool, self.enemy_seed, enemy_emb, enemy_mask)
        combat_hidden = self.combat_net(torch.cat([shared, hand_out, enemy_out], dim=-1))

        # ---- Build expert state ----
        deck_text = self.deck_text_proj(obs["deck_text"])
        deck_emb = self.deck_enc(torch.cat([obs["deck"], deck_text], dim=-1))
        deck_mask = obs["deck_mask"].bool()

        deck_emb = self.deck_norm(deck_emb + _safe_self_attn(self.deck_attn, deck_emb, deck_mask))
        deck_out = _safe_pool(self.deck_pool, self.deck_seed, deck_emb, deck_mask)
        build_hidden = self.build_net(torch.cat([shared, deck_out], dim=-1))

        # ---- Route expert state ----
        route_state = self.route_encoder(shared)
        route_hidden = self.route_net(torch.cat([shared, route_state], dim=-1))

        combat_mask, build_mask, route_mask = _domain_masks_from_obs(obs, device)
        hidden_state = torch.zeros(shared.shape[0], self.hidden_dim, device=device)

        if combat_mask.any():
            hidden_state[combat_mask] = combat_hidden[combat_mask]
        if build_mask.any():
            hidden_state[build_mask] = build_hidden[build_mask]
        if route_mask.any():
            hidden_state[route_mask] = route_hidden[route_mask]

        return self.output_norm(hidden_state)


# ============================================================================
# DynamicsNetwork: predict next hidden state and reward
# ============================================================================

class DynamicsNetwork(nn.Module):
    """Predict next hidden state and immediate reward from current state and action.

    Architecture:
    - Concat hidden_state + action_embedding
    - MLP with residual connection back to hidden_state
    - LayerNorm on residual output
    - Separate reward head on pre-residual features
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        action_embed_dim: int = 64,
        support_size: int = 25,
        num_res_blocks: int = 4,
    ):
        """Initialize dynamics network.

        Args:
            hidden_dim: hidden state dimension (default 512)
            action_embed_dim: action embedding dimension (default 64)
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.action_embed_dim = action_embed_dim
        self.support_size = support_size
        self.num_bins = 2 * support_size + 1
        self.num_res_blocks = max(int(num_res_blocks), 1)

        # Input projector for the first transition update
        self.input_proj = nn.Sequential(
            nn.Linear(hidden_dim + action_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Deeper residual stack for latent transition modeling
        self.res_blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(self.num_res_blocks)
        ])
        self.dynamics_norm = nn.LayerNorm(hidden_dim)

        # Separate small MLP for reward prediction
        self.reward_net = nn.Sequential(
            nn.Linear(hidden_dim + action_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.num_bins),
        )
        self.reward_component_net = nn.Sequential(
            nn.Linear(hidden_dim + action_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.num_bins * NUM_OBJECTIVE_HEADS),
        )
        self.surprise_net = nn.Sequential(
            nn.Linear(hidden_dim + action_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        hidden_state: torch.Tensor,
        action_embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict next hidden state and reward.

        Args:
            hidden_state: [batch, hidden_dim]
            action_embedding: [batch, action_embed_dim]

        Returns:
            (
                next_hidden_state [batch, hidden_dim],
                reward_logits [batch, 2*support_size+1],
                reward [batch, 1],
                reward_component_logits [batch, NUM_OBJECTIVE_HEADS, 2*support_size+1],
                reward_components [batch, NUM_OBJECTIVE_HEADS],
                surprise_logits [batch],
                surprise [batch]
            )
        """
        combined = torch.cat([hidden_state, action_embedding], dim=-1)

        # Transition with deeper residual refinement
        next_hidden = hidden_state + self.input_proj(combined)
        for block in self.res_blocks:
            next_hidden = next_hidden + block(next_hidden)
        next_hidden = self.dynamics_norm(next_hidden)

        # Reward prediction
        reward_logits = self.reward_net(combined)
        reward = support_to_scalar(reward_logits, self.support_size)
        reward_component_logits = self.reward_component_net(combined).view(
            combined.shape[0],
            NUM_OBJECTIVE_HEADS,
            self.num_bins,
        )
        reward_components = support_tensor_to_scalar(
            reward_component_logits,
            self.support_size,
        )
        surprise_logits = self.surprise_net(combined).squeeze(-1)
        surprise = torch.nn.functional.softplus(surprise_logits)

        return (
            next_hidden,
            reward_logits,
            reward,
            reward_component_logits,
            reward_components,
            surprise_logits,
            surprise,
        )


# ============================================================================
# Candidate-conditioned Policy Scorer
# ============================================================================

class CandidatePolicyScorer(nn.Module):
    """Score candidate action embeddings conditioned on latent hidden state."""

    def __init__(self, *, state_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.state_to_action = nn.Linear(state_dim, action_dim)
        self.scorer = nn.Sequential(
            nn.Linear(state_dim + action_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, hidden_state: torch.Tensor, action_embeddings: torch.Tensor) -> torch.Tensor:
        batch_size, action_count, _ = action_embeddings.shape
        expanded_state = hidden_state.unsqueeze(1).expand(-1, action_count, -1)
        projected_state = self.state_to_action(expanded_state)
        scorer_input = torch.cat([expanded_state, action_embeddings, projected_state * action_embeddings], dim=-1)
        return self.scorer(scorer_input).squeeze(-1)


class LatentActionBank(nn.Module):
    """Generate latent per-slot action embeddings from hidden state."""

    def __init__(self, *, hidden_dim: int, action_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim

        self.slot_embeddings = nn.Parameter(
            torch.randn(NUM_DOMAINS, MAX_ACTIONS, action_dim) * 0.02
        )
        self.domain_adapters = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim * 2),
            )
            for _ in range(NUM_DOMAINS)
        ])
        self.domain_norms = nn.ModuleList([
            nn.LayerNorm(action_dim)
            for _ in range(NUM_DOMAINS)
        ])

    def domain_banks(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Return domain-specific latent action banks [B, D, A, E]."""
        banks = []
        batch_size = hidden_state.shape[0]
        for domain_idx in range(NUM_DOMAINS):
            bias, gate = self.domain_adapters[domain_idx](hidden_state).chunk(2, dim=-1)
            slot_bank = self.slot_embeddings[domain_idx].unsqueeze(0).expand(batch_size, -1, -1)
            bank = slot_bank * torch.sigmoid(gate).unsqueeze(1) + bias.unsqueeze(1)
            bank = self.domain_norms[domain_idx](bank)
            banks.append(bank)
        return torch.stack(banks, dim=1)


class TransitionSurfaceHead(nn.Module):
    """Predict next-step legal surface from latent hidden state."""

    def __init__(self, *, hidden_dim: int):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.action_mask_head = nn.Linear(hidden_dim, MAX_ACTIONS)
        self.decision_domain_head = nn.Linear(hidden_dim, NUM_DOMAINS)
        self.phase_head = nn.Linear(hidden_dim, NUM_PHASES)

        # Start by predicting all actions as illegal: sigmoid(-2.0) ≈ 0.12
        # This fixes cold-start over-prediction in class-imbalanced setting
        # (typically ~8 legal out of 80 slots)
        nn.init.constant_(self.action_mask_head.bias, -2.0)

    def forward(self, hidden_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.trunk(hidden_state)
        return (
            self.action_mask_head(features),
            self.decision_domain_head(features),
            self.phase_head(features),
        )


# ============================================================================
# PredictionNetwork: predict policy and value from hidden state
# ============================================================================

class PredictionNetwork(nn.Module):
    """Predict policy/value from hidden state with latent domain MoE heads."""

    def __init__(
        self,
        hidden_dim: int = 512,
        action_embed_dim: int = 64,
        support_size: int = 25,
    ):
        """Initialize prediction network.

        Args:
            hidden_dim: hidden state dimension (default 512)
            support_size: number of support bins on each side (default 25, -> 51 total)
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.support_size = support_size
        self.num_bins = 2 * support_size + 1
        self.latent_type_embeddings = nn.Embedding(2, hidden_dim)

        self.domain_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, NUM_DOMAINS),
        )
        self.latent_action_bank = LatentActionBank(
            hidden_dim=hidden_dim,
            action_dim=action_embed_dim,
        )

        self.candidate_policy_heads = nn.ModuleList([
            CandidatePolicyScorer(
                state_dim=hidden_dim,
                action_dim=action_embed_dim,
                hidden_dim=hidden_dim,
            )
            for _ in range(NUM_DOMAINS)
        ])

        self.value_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.num_bins),
            )
            for _ in range(NUM_DOMAINS)
        ])
        self.value_component_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.num_bins * NUM_OBJECTIVE_HEADS),
            )
            for _ in range(NUM_DOMAINS)
        ])
        self.semantic_policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, SEMANTIC_ROLLOUT_SIZE),
        )

    def _head_state(self, hidden_state: torch.Tensor, latent_type_id: int) -> torch.Tensor:
        type_index = torch.full(
            (hidden_state.shape[0],),
            int(latent_type_id),
            device=hidden_state.device,
            dtype=torch.long,
        )
        return hidden_state + self.latent_type_embeddings(type_index)

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

    def _score_candidates(
        self,
        hidden_state: torch.Tensor,
        action_embeddings: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> torch.Tensor:
        domain_logits = torch.stack([
            scorer(hidden_state, action_embeddings)
            for scorer in self.candidate_policy_heads
        ], dim=1)
        return (domain_logits * routing_weights.unsqueeze(-1)).sum(dim=1)

    def _latent_policy_logits(
        self,
        hidden_state: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        domain_banks = self.latent_action_bank.domain_banks(hidden_state)
        mixed_embeddings = (domain_banks * routing_weights.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)
        domain_logits = torch.stack([
            scorer(hidden_state, domain_banks[:, domain_idx])
            for domain_idx, scorer in enumerate(self.candidate_policy_heads)
        ], dim=1)
        latent_logits = (domain_logits * routing_weights.unsqueeze(-1)).sum(dim=1)
        return latent_logits, mixed_embeddings

    def _conditioned_policy_logits(
        self,
        hidden_state: torch.Tensor,
        action_embeddings: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> torch.Tensor:
        return self._score_candidates(hidden_state, action_embeddings, routing_weights)

    def _value_outputs(
        self,
        head_state: torch.Tensor,
        *,
        decision_domain: torch.Tensor | None = None,
        objective_context: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        domain_logits = self.domain_gate(head_state)
        domain_weights = torch.softmax(domain_logits, dim=-1)
        routing_weights = self._routing_weights(
            domain_weights=domain_weights,
            decision_domain=decision_domain,
        )
        value_stack = torch.stack([head(head_state) for head in self.value_heads], dim=1)
        component_stack = torch.stack([
            head(head_state).view(head_state.shape[0], NUM_OBJECTIVE_HEADS, self.num_bins)
            for head in self.value_component_heads
        ], dim=1)
        value_logits = (value_stack * routing_weights.unsqueeze(-1)).sum(dim=1)
        value_component_logits = (
            component_stack * routing_weights.unsqueeze(-1).unsqueeze(-1)
        ).sum(dim=1)
        value = support_to_scalar(value_logits, self.support_size)
        value_components = support_tensor_to_scalar(
            value_component_logits,
            self.support_size,
        )
        objective_value = scalarize_objective_components_torch(
            value_components,
            objective_context,
        )
        return value_logits, value, value_component_logits, value_components, objective_value

    def value_only(
        self,
        hidden_state: torch.Tensor,
        *,
        decision_domain: torch.Tensor | None = None,
        objective_context: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Cheap value-only path for explicit action-conditioned rollout planning."""
        head_state = self._head_state(hidden_state, latent_type_id=0)
        return self._value_outputs(
            head_state,
            decision_domain=decision_domain,
            objective_context=objective_context,
        )

    def latent_policy_only(
        self,
        hidden_state: torch.Tensor,
        *,
        decision_domain: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Cheap latent-action prior path for multi-step search-free rollout."""
        head_state = self._head_state(hidden_state, latent_type_id=0)
        domain_logits = self.domain_gate(head_state)
        domain_weights = torch.softmax(domain_logits, dim=-1)
        routing_weights = self._routing_weights(
            domain_weights=domain_weights,
            decision_domain=decision_domain,
        )
        return self._latent_policy_logits(
            head_state,
            routing_weights,
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
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        """Predict policy and value.

        Args:
            hidden_state: [batch, hidden_dim]

        Returns:
            (
                policy_logits [batch, MAX_ACTIONS],
                value_logits [batch, 2*support_size+1],
                value [batch, 1],
                latent_policy_logits [batch, MAX_ACTIONS],
                latent_action_embeddings [batch, MAX_ACTIONS, action_embed_dim],
                value_component_logits [batch, NUM_OBJECTIVE_HEADS, 2*support_size+1],
                value_components [batch, NUM_OBJECTIVE_HEADS],
                objective_value [batch]
            )
        """
        head_state = self._head_state(hidden_state, latent_type_id=0)
        domain_logits = self.domain_gate(head_state)
        domain_weights = torch.softmax(domain_logits, dim=-1)
        routing_weights = self._routing_weights(
            domain_weights=domain_weights,
            decision_domain=decision_domain,
        )
        component_stack = torch.stack([
            head(head_state).view(head_state.shape[0], NUM_OBJECTIVE_HEADS, self.num_bins)
            for head in self.value_component_heads
        ], dim=1)
        value_stack = torch.stack([head(head_state) for head in self.value_heads], dim=1)

        latent_policy_logits, latent_action_embeddings = self._latent_policy_logits(
            head_state,
            routing_weights,
        )
        policy_logits = latent_policy_logits
        if action_embeddings is not None and decision_domain is not None:
            policy_logits = self._conditioned_policy_logits(
                head_state,
                action_embeddings,
                routing_weights,
            )
        value_logits = (value_stack * routing_weights.unsqueeze(-1)).sum(dim=1)
        value_component_logits = (
            component_stack * routing_weights.unsqueeze(-1).unsqueeze(-1)
        ).sum(dim=1)

        # Convert categorical value to scalar
        value = support_to_scalar(value_logits, self.support_size)
        value_components = support_tensor_to_scalar(
            value_component_logits,
            self.support_size,
        )
        objective_value = scalarize_objective_components_torch(
            value_components,
            objective_context,
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
            None,
            None,
            None,
            None,
            None,
        )

    def semantic_forward(
        self,
        hidden_state: torch.Tensor,
        *,
        decision_domain: torch.Tensor | None = None,
        objective_context: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        head_state = self._head_state(hidden_state, latent_type_id=1)
        semantic_policy_logits = self.semantic_policy_head(head_state)
        value_logits, value, value_component_logits, value_components, objective_value = self._value_outputs(
            head_state,
            decision_domain=decision_domain,
            objective_context=objective_context,
        )
        return (
            semantic_policy_logits,
            value_logits,
            value,
            value_component_logits,
            value_components,
            objective_value,
        )


# ============================================================================
# MuZeroNetwork: main wrapper
# ============================================================================

class MuZeroNetworkOutput(NamedTuple):
    """Output from initial inference."""
    hidden_state: torch.Tensor
    policy_logits: torch.Tensor
    value_logits: torch.Tensor
    value: torch.Tensor
    objective_value: torch.Tensor
    value_component_logits: torch.Tensor
    value_components: torch.Tensor
    action_embeddings: torch.Tensor
    latent_policy_logits: torch.Tensor
    latent_action_embeddings: torch.Tensor
    planner_q_logits: torch.Tensor | None
    planner_q: torch.Tensor | None
    planner_objective_q: torch.Tensor | None
    planner_q_component_logits: torch.Tensor | None
    planner_q_components: torch.Tensor | None


class SemanticMuZeroOutput(NamedTuple):
    """Output from semantic latent prediction."""
    semantic_hidden_state: torch.Tensor
    semantic_policy_logits: torch.Tensor
    value_logits: torch.Tensor
    value: torch.Tensor
    objective_value: torch.Tensor
    value_component_logits: torch.Tensor
    value_components: torch.Tensor


class RecurrentMuZeroOutput(NamedTuple):
    """Output from recurrent inference."""
    next_hidden_state: torch.Tensor
    reward_logits: torch.Tensor
    reward: torch.Tensor
    reward_component_logits: torch.Tensor
    reward_components: torch.Tensor
    surprise_logits: torch.Tensor
    surprise: torch.Tensor
    policy_logits: torch.Tensor
    value_logits: torch.Tensor
    value: torch.Tensor
    objective_value: torch.Tensor
    value_component_logits: torch.Tensor
    value_components: torch.Tensor
    action_embeddings: torch.Tensor
    latent_policy_logits: torch.Tensor
    latent_action_embeddings: torch.Tensor
    planner_q_logits: torch.Tensor | None
    planner_q: torch.Tensor | None
    planner_objective_q: torch.Tensor | None
    planner_q_component_logits: torch.Tensor | None
    planner_q_components: torch.Tensor | None
    next_action_mask_logits: torch.Tensor
    next_decision_domain_logits: torch.Tensor
    next_phase_logits: torch.Tensor
    teacher_hidden_state: torch.Tensor | None
    current_world_bank_state_ref: torch.Tensor | None
    next_world_bank_state_pred: torch.Tensor | None
    next_world_bank_state_target: torch.Tensor | None
    next_world_bank_delta_pred: torch.Tensor | None
    next_world_bank_delta_target: torch.Tensor | None
    next_world_bank_occupancy_logits: torch.Tensor | None
    next_world_bank_occupancy_target: torch.Tensor | None
    next_world_bank_token_presence_logits: torch.Tensor | None
    next_world_bank_token_presence_target: torch.Tensor | None
    next_world_bank_token_distribution_logits: torch.Tensor | None
    next_world_bank_token_distribution_target: torch.Tensor | None
    next_world_bank_token_slot_state_pred: torch.Tensor | None
    next_world_bank_token_slot_state_target: torch.Tensor | None
    next_world_bank_token_slot_mask_logits: torch.Tensor | None
    next_world_bank_token_slot_mask_target: torch.Tensor | None
    next_world_bank_token_slot_type_logits: torch.Tensor | None
    next_world_bank_token_slot_type_target: torch.Tensor | None
    next_world_bank_token_slot_zone_logits: torch.Tensor | None
    next_world_bank_token_slot_zone_target: torch.Tensor | None
    next_world_bank_token_slot_source_logits: torch.Tensor | None
    next_world_bank_token_slot_source_target: torch.Tensor | None




class SemanticRecurrentMuZeroOutput(NamedTuple):
    """Output from semantic recurrent inference."""
    next_semantic_hidden_state: torch.Tensor
    reward_logits: torch.Tensor
    reward: torch.Tensor
    reward_component_logits: torch.Tensor
    reward_components: torch.Tensor
    surprise_logits: torch.Tensor
    surprise: torch.Tensor
    semantic_policy_logits: torch.Tensor
    value_logits: torch.Tensor
    value: torch.Tensor
    objective_value: torch.Tensor
    value_component_logits: torch.Tensor
    value_components: torch.Tensor


# The output type remains publicly owned by this compatibility module even
# though its implementation lives beside the planner mixin.
ActionRolloutPlannerOutput.__module__ = __name__
for _support_function in (
    scalar_to_support,
    support_tensor_to_scalar,
    support_to_scalar,
):
    _support_function.__module__ = __name__
del _support_function


class MuZeroNetwork(ActionRolloutPlannerMixin, nn.Module):
    """MuZero-style model combining representation, dynamics, and prediction networks.

    Provides both initial_inference (initial state from observation) and
    recurrent_inference (future prediction from hidden state + action).
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        action_embed_dim: int = 64,
        embed_dim: int = 64,
        n_heads: int = 4,
        text_proj_dim: int = 32,
        support_size: int = 25,
        dynamics_res_blocks: int = 4,
        obs_mode: str = "dense_v2",
        model_arch: str = "dense_v1",
        token_d_model: int = 128,
        token_n_heads: int = 4,
        token_ffn_dim: int = 512,
        token_world_layers: int = 4,
        token_local_layers: int = 1,
        token_decoder_layers: int = 2,
        token_candidate_set_layers: int = 1,
        token_memory_slots: int = 8,
        token_memory_slot_layout: str = "legacy",
        token_world_bank_top_k: int = 3,
        token_bank_token_slots: int = 4,
        token_slot_source_same_bank_bias: float = 0.35,
        token_slot_source_same_slot_bias: float = 0.2,
        token_slot_source_type_match_scale: float = 0.5,
        token_slot_source_zone_transport_scale: float = 0.35,
        token_internal_planner_blend: float = 0.7,
        token_internal_planner_q_blend: float = 0.5,
        token_internal_planner_objective_q_blend: float = 0.35,
        token_internal_planner_risk_blend: float = 0.25,
        token_dropout: float = 0.0,
        action_rollout_buckets: str | tuple[int, ...] | list[int] | None = None,
        action_rollout_chunk_size: int = 0,
        activation_checkpointing: bool = False,
    ):
        """Initialize MuZero network.

        Args:
            hidden_dim: hidden state dimension (default 512)
            action_embed_dim: action embedding dimension (default 64)
            embed_dim: attention embedding dimension (default 64)
            n_heads: number of attention heads (default 4)
            text_proj_dim: text projection dimension (default 32)
            support_size: value support size (default 25 -> 51 bins)
            dynamics_res_blocks: number of residual transition refinement blocks
        """
        super().__init__()

        self.obs_mode = str(obs_mode or "dense_v2").strip().lower()
        self.model_arch = str(model_arch or "dense_v1").strip().lower()
        self.is_token_mode = self.obs_mode == "token_v3" or self.model_arch == "token_memory_v1"
        if self.is_token_mode:
            hidden_dim = int(token_d_model) * int(token_memory_slots)

        self.hidden_dim = hidden_dim
        self.action_embed_dim = action_embed_dim
        self.support_size = support_size
        self.action_rollout_buckets = self._normalize_action_rollout_buckets(action_rollout_buckets)
        self.action_rollout_chunk_size = max(int(action_rollout_chunk_size or 0), 0)
        self.constructor_kwargs = {
            "hidden_dim": int(hidden_dim),
            "action_embed_dim": int(action_embed_dim),
            "embed_dim": int(embed_dim),
            "n_heads": int(n_heads),
            "text_proj_dim": int(text_proj_dim),
            "support_size": int(support_size),
            "dynamics_res_blocks": int(dynamics_res_blocks),
            "obs_mode": self.obs_mode,
            "model_arch": self.model_arch,
            "token_d_model": int(token_d_model),
            "token_n_heads": int(token_n_heads),
            "token_ffn_dim": int(token_ffn_dim),
            "token_world_layers": int(token_world_layers),
            "token_local_layers": int(token_local_layers),
            "token_decoder_layers": int(token_decoder_layers),
            "token_candidate_set_layers": int(token_candidate_set_layers),
            "token_memory_slots": int(token_memory_slots),
            "token_memory_slot_layout": str(token_memory_slot_layout or "legacy").strip().lower(),
            "token_world_bank_top_k": int(token_world_bank_top_k),
            "token_bank_token_slots": int(token_bank_token_slots),
            "token_slot_source_same_bank_bias": float(token_slot_source_same_bank_bias),
            "token_slot_source_same_slot_bias": float(token_slot_source_same_slot_bias),
            "token_slot_source_type_match_scale": float(token_slot_source_type_match_scale),
            "token_slot_source_zone_transport_scale": float(token_slot_source_zone_transport_scale),
            "token_internal_planner_blend": float(token_internal_planner_blend),
            "token_internal_planner_q_blend": float(token_internal_planner_q_blend),
            "token_internal_planner_objective_q_blend": float(token_internal_planner_objective_q_blend),
            "token_internal_planner_risk_blend": float(token_internal_planner_risk_blend),
            "token_dropout": float(token_dropout),
            "action_rollout_buckets": tuple(int(bucket) for bucket in self.action_rollout_buckets),
            "action_rollout_chunk_size": int(self.action_rollout_chunk_size),
            "activation_checkpointing": bool(activation_checkpointing),
        }

        # Core networks
        if self.is_token_mode:
            self.token_encoder = TokenMemoryEncoder(
                d_model=token_d_model,
                n_heads=token_n_heads,
                ffn_dim=token_ffn_dim,
                world_layers=token_world_layers,
                local_layers=token_local_layers,
                decoder_layers=token_decoder_layers,
                candidate_set_layers=token_candidate_set_layers,
                world_bank_top_k=token_world_bank_top_k,
                bank_token_slots=token_bank_token_slots,
                num_memory_slots=token_memory_slots,
                memory_slot_layout=token_memory_slot_layout,
                action_embed_dim=action_embed_dim,
                dropout=token_dropout,
                activation_checkpointing=activation_checkpointing,
            )
            self.representation = None
            self.action_encoder = None
            self.dynamics = TokenDynamicsNetwork(
                hidden_dim=hidden_dim,
                action_embed_dim=action_embed_dim,
                d_model=token_d_model,
                num_memory_slots=token_memory_slots,
                memory_slot_layout=token_memory_slot_layout,
                support_size=support_size,
                num_transition_layers=max(2, dynamics_res_blocks // 2),
                dropout=token_dropout,
                activation_checkpointing=activation_checkpointing,
            )
            self.prediction = TokenPredictionNetwork(
                hidden_dim=hidden_dim,
                action_embed_dim=action_embed_dim,
                d_model=token_d_model,
                num_memory_slots=token_memory_slots,
                memory_slot_layout=token_memory_slot_layout,
                support_size=support_size,
                internal_planner_blend=token_internal_planner_blend,
                internal_planner_q_blend=token_internal_planner_q_blend,
                internal_planner_objective_q_blend=token_internal_planner_objective_q_blend,
                internal_planner_risk_blend=token_internal_planner_risk_blend,
                n_heads=token_n_heads,
                ffn_dim=token_ffn_dim,
                dropout=token_dropout,
                activation_checkpointing=activation_checkpointing,
                planner_action_chunk_size=self.action_rollout_chunk_size,
            )
            self.transition_surface = TokenTransitionSurfaceHead(
                hidden_dim=hidden_dim,
                d_model=token_d_model,
                num_memory_slots=token_memory_slots,
                memory_slot_layout=token_memory_slot_layout,
                n_heads=token_n_heads,
                ffn_dim=token_ffn_dim,
                dropout=token_dropout,
                activation_checkpointing=activation_checkpointing,
                planner_action_chunk_size=self.action_rollout_chunk_size,
            )
            self.future_world_bank_head = TokenFutureWorldBankHead(
                hidden_dim=hidden_dim,
                d_model=token_d_model,
                num_memory_slots=token_memory_slots,
                memory_slot_layout=token_memory_slot_layout,
                bank_token_slots=token_bank_token_slots,
                slot_source_same_bank_bias=token_slot_source_same_bank_bias,
                slot_source_same_slot_bias=token_slot_source_same_slot_bias,
                slot_source_type_match_scale=token_slot_source_type_match_scale,
                slot_source_zone_transport_scale=token_slot_source_zone_transport_scale,
                n_heads=token_n_heads,
                ffn_dim=token_ffn_dim,
                dropout=token_dropout,
                activation_checkpointing=activation_checkpointing,
            )
            self.state_projector = TokenLatentProjector(
                hidden_dim=hidden_dim,
                d_model=token_d_model,
                num_memory_slots=token_memory_slots,
                memory_slot_layout=token_memory_slot_layout,
                n_heads=token_n_heads,
                ffn_dim=token_ffn_dim,
                num_layers=2,
                dropout=token_dropout,
                activation_checkpointing=activation_checkpointing,
            )
            self.semantic_state_projector = TokenLatentProjector(
                hidden_dim=hidden_dim,
                d_model=token_d_model,
                num_memory_slots=token_memory_slots,
                memory_slot_layout=token_memory_slot_layout,
                n_heads=token_n_heads,
                ffn_dim=token_ffn_dim,
                num_layers=2,
                dropout=token_dropout,
                activation_checkpointing=activation_checkpointing,
            )
        else:
            self.token_encoder = None
            self.representation = RepresentationNetwork(
                hidden_dim=hidden_dim,
                embed_dim=embed_dim,
                n_heads=n_heads,
                text_proj_dim=text_proj_dim,
            )
            self.dynamics = DynamicsNetwork(
                hidden_dim=hidden_dim,
                action_embed_dim=action_embed_dim,
                support_size=support_size,
                num_res_blocks=dynamics_res_blocks,
            )
            self.prediction = PredictionNetwork(
                hidden_dim=hidden_dim,
                action_embed_dim=action_embed_dim,
                support_size=support_size,
            )
            self.transition_surface = TransitionSurfaceHead(hidden_dim=hidden_dim)
            self.future_world_bank_head = None
            self.state_projector = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

        if not self.is_token_mode:
            self.action_encoder = ActionEncoder(
                action_embed_dim=action_embed_dim,
                text_proj_dim=text_proj_dim,
                route_node_dim=max(32, action_embed_dim // 2),
                n_heads=max(2, n_heads // 2),
            )
            self.semantic_state_projector = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )
        self.semantic_action_embedding = nn.Embedding(SEMANTIC_ROLLOUT_SIZE, action_embed_dim)
        if self.is_token_mode:
            self.semantic_dynamics = TokenDynamicsNetwork(
                hidden_dim=hidden_dim,
                action_embed_dim=action_embed_dim,
                d_model=token_d_model,
                num_memory_slots=token_memory_slots,
                memory_slot_layout=token_memory_slot_layout,
                support_size=support_size,
                num_transition_layers=max(2, dynamics_res_blocks // 2),
                dropout=token_dropout,
                activation_checkpointing=activation_checkpointing,
            )
        else:
            self.semantic_dynamics = DynamicsNetwork(
                hidden_dim=hidden_dim,
                action_embed_dim=action_embed_dim,
                support_size=support_size,
                num_res_blocks=max(2, dynamics_res_blocks - 1),
            )

    def constructor_spec(self) -> dict[str, int | float | str]:
        return dict(self.constructor_kwargs)

    def _decision_domain_tensor(self, obs: dict[str, torch.Tensor] | None) -> torch.Tensor | None:
        if obs is None:
            return None
        decision_domain = obs.get("decision_domain")
        if decision_domain is not None:
            return decision_domain
        if self.is_token_mode:
            return infer_token_decision_domain(obs)
        return None

    def _encode_token_obs(
        self,
        obs: dict[str, torch.Tensor] | None,
    ) -> TokenMemoryEncoderOutput | None:
        if not self.is_token_mode or obs is None:
            return None
        return self.token_encoder(obs)

    @staticmethod
    def _derive_bank_token_slot_source_targets(
        *,
        current_slot_mask: torch.Tensor,
        current_slot_type_ids: torch.Tensor,
        current_slot_entity_ids: torch.Tensor,
        current_slot_order_ids: torch.Tensor,
        next_slot_mask: torch.Tensor,
        next_slot_type_ids: torch.Tensor,
        next_slot_entity_ids: torch.Tensor,
        next_slot_order_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Build copy/migration labels from current teacher slots to next teacher slots.

        Output indices address flattened current slots across all world banks, with the
        final index reserved for the "new/generated token" sentinel.
        """

        batch_size, bank_count, slot_count = current_slot_mask.shape
        total_current_slots = bank_count * slot_count
        sentinel_index = total_current_slots
        device = current_slot_mask.device
        targets = torch.full(
            (batch_size, bank_count, slot_count),
            sentinel_index,
            dtype=torch.long,
            device=device,
        )

        current_mask_flat = current_slot_mask.reshape(batch_size, total_current_slots).bool()
        current_type_flat = current_slot_type_ids.reshape(batch_size, total_current_slots).long()
        current_entity_flat = current_slot_entity_ids.reshape(batch_size, total_current_slots).long()
        current_order_flat = current_slot_order_ids.reshape(batch_size, total_current_slots).long()
        current_bank_flat = (
            torch.arange(bank_count, device=device, dtype=torch.long)
            .view(1, bank_count, 1)
            .expand(batch_size, bank_count, slot_count)
            .reshape(batch_size, total_current_slots)
        )

        next_mask = next_slot_mask.bool()
        next_type = next_slot_type_ids.long()
        next_entity = next_slot_entity_ids.long()
        next_order = next_slot_order_ids.long()

        for batch_index in range(batch_size):
            valid_current_idx = torch.nonzero(current_mask_flat[batch_index], as_tuple=False).flatten()
            if valid_current_idx.numel() == 0:
                continue
            current_entities = current_entity_flat[batch_index, valid_current_idx]
            current_types = current_type_flat[batch_index, valid_current_idx]
            current_orders = current_order_flat[batch_index, valid_current_idx]
            current_banks = current_bank_flat[batch_index, valid_current_idx]

            for bank_index in range(bank_count):
                for slot_index in range(slot_count):
                    if not bool(next_mask[batch_index, bank_index, slot_index].item()):
                        continue
                    next_entity_id = int(next_entity[batch_index, bank_index, slot_index].item())
                    next_type_id = int(next_type[batch_index, bank_index, slot_index].item())
                    next_order_id = int(next_order[batch_index, bank_index, slot_index].item())

                    chosen_index: int | None = None
                    if next_entity_id > 0:
                        entity_match_mask = current_entities.eq(next_entity_id)
                        if entity_match_mask.any():
                            match_indices = valid_current_idx[entity_match_mask]
                            if match_indices.numel() > 1:
                                candidate_types = current_types[entity_match_mask]
                                candidate_orders = current_orders[entity_match_mask]
                                penalties = (
                                    candidate_types.ne(next_type_id).long() * 1024
                                    + (candidate_orders - next_order_id).abs()
                                )
                                chosen_index = int(match_indices[penalties.argmin()].item())
                            else:
                                chosen_index = int(match_indices[0].item())

                    if chosen_index is None and next_type_id >= 0:
                        fallback_mask = current_types.eq(next_type_id) & current_banks.eq(bank_index)
                        if fallback_mask.any():
                            fallback_indices = valid_current_idx[fallback_mask]
                            fallback_orders = current_orders[fallback_mask]
                            chosen_index = int(fallback_indices[(fallback_orders - next_order_id).abs().argmin()].item())

                    if chosen_index is not None:
                        targets[batch_index, bank_index, slot_index] = chosen_index

        return targets

    def initial_inference(
        self,
        obs: dict[str, torch.Tensor],
    ) -> MuZeroNetworkOutput:
        """Compute initial hidden state and policy/value from observation.

        Args:
            obs: observation dict with all standard STS2 observation keys

        Returns:
            MuZeroNetworkOutput with policy_logits, value, hidden_state
        """
        if self.is_token_mode:
            encoded = self.token_encoder(obs)
            hidden_state = encoded.hidden_state
            action_embeddings = encoded.action_embeddings
            decision_domain = encoded.decision_domain
        else:
            hidden_state = self.representation(obs)
            action_embeddings = self.encode_actions(obs)
            decision_domain = self._decision_domain_tensor(obs)
        (
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
        ) = self.prediction(
            hidden_state,
            action_embeddings=action_embeddings,
            decision_domain=decision_domain,
            objective_context=obs.get("objective_context"),
        )

        return MuZeroNetworkOutput(
            hidden_state=hidden_state,
            policy_logits=policy_logits,
            value_logits=value_logits,
            value=value,
            objective_value=objective_value,
            value_component_logits=value_component_logits,
            value_components=value_components,
            action_embeddings=action_embeddings,
            latent_policy_logits=latent_policy_logits,
            latent_action_embeddings=latent_action_embeddings,
            planner_q_logits=planner_q_logits,
            planner_q=planner_q,
            planner_objective_q=planner_objective_q,
            planner_q_component_logits=planner_q_component_logits,
            planner_q_components=planner_q_components,
        )

    def recurrent_inference(
        self,
        hidden_state: torch.Tensor,
        action_embedding: torch.Tensor,
        current_obs: dict[str, torch.Tensor] | None = None,
        teacher_current_encoded: TokenMemoryEncoderOutput | None = None,
        next_obs: dict[str, torch.Tensor] | None = None,
        teacher_next_encoded: TokenMemoryEncoderOutput | None = None,
    ) -> RecurrentMuZeroOutput:
        """Predict next state, reward, policy, and value from hidden state and action.

        Args:
            hidden_state: [batch, hidden_dim]
            action_embedding: [batch, action_embed_dim]

        Returns:
            RecurrentMuZeroOutput with next_hidden_state, reward, policy_logits, value
        """
        (
            next_hidden_state,
            reward_logits,
            reward,
            reward_component_logits,
            reward_components,
            surprise_logits,
            surprise,
        ) = self.dynamics(hidden_state, action_embedding)
        next_action_mask_logits, next_decision_domain_logits, next_phase_logits = self.transition_surface(next_hidden_state)
        current_world_bank_state_ref: torch.Tensor | None = None
        next_world_bank_state_pred: torch.Tensor | None = None
        next_world_bank_delta_pred: torch.Tensor | None = None
        next_world_bank_occupancy_logits: torch.Tensor | None = None
        next_world_bank_token_presence_logits: torch.Tensor | None = None
        next_world_bank_token_distribution_logits: torch.Tensor | None = None
        next_world_bank_token_slot_state_pred: torch.Tensor | None = None
        next_world_bank_token_slot_mask_logits: torch.Tensor | None = None
        next_world_bank_token_slot_type_logits: torch.Tensor | None = None
        next_world_bank_token_slot_zone_logits: torch.Tensor | None = None
        next_world_bank_token_slot_source_logits: torch.Tensor | None = None
        if self.is_token_mode and self.future_world_bank_head is not None:
            current_world_bank_state_ref = self.future_world_bank_head.read_current_bank_states(hidden_state)
            (
                current_bank_token_slot_ref,
                current_bank_token_slot_mask_ref_logits,
                current_bank_token_slot_type_ref_logits,
                current_bank_token_slot_zone_ref_logits,
            ) = (
                self.future_world_bank_head.read_current_bank_token_slots(hidden_state)
            )
            source_bank_token_slot_states = current_bank_token_slot_ref
            source_bank_token_slot_mask = current_bank_token_slot_mask_ref_logits > 0.0
            source_bank_token_slot_type_ids = current_bank_token_slot_type_ref_logits.argmax(dim=-1)
            source_bank_token_slot_zone_ids = current_bank_token_slot_zone_ref_logits.argmax(dim=-1)
            if (
                teacher_current_encoded is not None
                and teacher_current_encoded.world_bank_token_slot_states is not None
                and teacher_current_encoded.world_bank_token_slot_mask is not None
                and teacher_current_encoded.world_bank_token_slot_type_ids is not None
                and teacher_current_encoded.world_bank_token_slot_zone_ids is not None
            ):
                source_bank_token_slot_states = teacher_current_encoded.world_bank_token_slot_states.detach()
                source_bank_token_slot_mask = teacher_current_encoded.world_bank_token_slot_mask.detach().bool()
                source_bank_token_slot_type_ids = teacher_current_encoded.world_bank_token_slot_type_ids.detach().long()
                source_bank_token_slot_zone_ids = teacher_current_encoded.world_bank_token_slot_zone_ids.detach().long()
            (
                next_world_bank_state_pred,
                next_world_bank_occupancy_logits,
                next_world_bank_token_presence_logits,
                next_world_bank_token_distribution_logits,
                next_world_bank_token_slot_state_pred,
                next_world_bank_token_slot_mask_logits,
                next_world_bank_token_slot_type_logits,
                next_world_bank_token_slot_zone_logits,
                next_world_bank_token_slot_source_logits,
            ) = self.future_world_bank_head(
                next_hidden_state,
                current_bank_token_slot_states=source_bank_token_slot_states,
                current_bank_token_slot_mask=source_bank_token_slot_mask,
                current_bank_token_slot_type_ids=source_bank_token_slot_type_ids,
                current_bank_token_slot_zone_ids=source_bank_token_slot_zone_ids,
            )
            next_world_bank_delta_pred = next_world_bank_state_pred - current_world_bank_state_ref
        action_embeddings: torch.Tensor | None = None
        planner_q_logits: torch.Tensor | None = None
        planner_q: torch.Tensor | None = None
        planner_objective_q: torch.Tensor | None = None
        planner_q_component_logits: torch.Tensor | None = None
        planner_q_components: torch.Tensor | None = None
        teacher_hidden_state: torch.Tensor | None = None
        next_world_bank_state_target: torch.Tensor | None = None
        next_world_bank_delta_target: torch.Tensor | None = None
        next_world_bank_occupancy_target: torch.Tensor | None = None
        next_world_bank_token_presence_target: torch.Tensor | None = None
        next_world_bank_token_distribution_target: torch.Tensor | None = None
        next_world_bank_token_slot_state_target: torch.Tensor | None = None
        next_world_bank_token_slot_mask_target: torch.Tensor | None = None
        next_world_bank_token_slot_type_target: torch.Tensor | None = None
        next_world_bank_token_slot_zone_target: torch.Tensor | None = None
        next_world_bank_token_slot_source_target: torch.Tensor | None = None
        current_token_encoded = teacher_current_encoded if teacher_current_encoded is not None else self._encode_token_obs(current_obs)
        if next_obs is not None:
            decision_domain = self._decision_domain_tensor(next_obs)
            if self.is_token_mode:
                student_encoded = self._encode_token_obs(next_obs)
                assert student_encoded is not None
                encoded = teacher_next_encoded if teacher_next_encoded is not None else student_encoded
                action_embeddings = student_encoded.action_embeddings
                decision_domain = student_encoded.decision_domain
                teacher_hidden_state = encoded.hidden_state.detach()
                next_world_bank_state_target = (
                    encoded.world_bank_summaries.detach() if encoded.world_bank_summaries is not None else None
                )
                next_world_bank_occupancy_target = (
                    encoded.world_bank_occupancy.detach() if encoded.world_bank_occupancy is not None else None
                )
                next_world_bank_token_presence_target = (
                    encoded.world_bank_token_presence.detach() if encoded.world_bank_token_presence is not None else None
                )
                next_world_bank_token_distribution_target = (
                    encoded.world_bank_token_distribution.detach()
                    if encoded.world_bank_token_distribution is not None
                    else None
                )
                next_world_bank_token_slot_state_target = (
                    encoded.world_bank_token_slot_states.detach()
                    if encoded.world_bank_token_slot_states is not None
                    else None
                )
                next_world_bank_token_slot_mask_target = (
                    encoded.world_bank_token_slot_mask.detach()
                    if encoded.world_bank_token_slot_mask is not None
                    else None
                )
                next_world_bank_token_slot_type_target = (
                    encoded.world_bank_token_slot_type_ids.detach()
                    if encoded.world_bank_token_slot_type_ids is not None
                    else None
                )
                next_world_bank_token_slot_zone_target = (
                    encoded.world_bank_token_slot_zone_ids.detach()
                    if encoded.world_bank_token_slot_zone_ids is not None
                    else None
                )
                if (
                    current_token_encoded is not None
                    and current_token_encoded.world_bank_token_slot_mask is not None
                    and current_token_encoded.world_bank_token_slot_type_ids is not None
                    and current_token_encoded.world_bank_token_slot_entity_ids is not None
                    and current_token_encoded.world_bank_token_slot_order_ids is not None
                    and encoded.world_bank_token_slot_mask is not None
                    and encoded.world_bank_token_slot_type_ids is not None
                    and encoded.world_bank_token_slot_entity_ids is not None
                    and encoded.world_bank_token_slot_order_ids is not None
                ):
                    next_world_bank_token_slot_source_target = self._derive_bank_token_slot_source_targets(
                        current_slot_mask=current_token_encoded.world_bank_token_slot_mask.detach(),
                        current_slot_type_ids=current_token_encoded.world_bank_token_slot_type_ids.detach(),
                        current_slot_entity_ids=current_token_encoded.world_bank_token_slot_entity_ids.detach(),
                        current_slot_order_ids=current_token_encoded.world_bank_token_slot_order_ids.detach(),
                        next_slot_mask=encoded.world_bank_token_slot_mask.detach(),
                        next_slot_type_ids=encoded.world_bank_token_slot_type_ids.detach(),
                        next_slot_entity_ids=encoded.world_bank_token_slot_entity_ids.detach(),
                        next_slot_order_ids=encoded.world_bank_token_slot_order_ids.detach(),
                    )
                if current_world_bank_state_ref is not None and next_world_bank_state_target is not None:
                    next_world_bank_delta_target = next_world_bank_state_target - current_world_bank_state_ref.detach()
            else:
                teacher_hidden_state = self.representation(next_obs).detach()
                action_embeddings = self.action_encoder(next_obs)
            (
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
            ) = self.prediction(
                next_hidden_state,
                action_embeddings=action_embeddings,
                decision_domain=decision_domain,
                objective_context=next_obs.get("objective_context"),
            )
        else:
            (
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
            ) = self.prediction(
                next_hidden_state,
            )
            action_embeddings = latent_action_embeddings

        return RecurrentMuZeroOutput(
            next_hidden_state=next_hidden_state,
            reward_logits=reward_logits,
            reward=reward,
            reward_component_logits=reward_component_logits,
            reward_components=reward_components,
            surprise_logits=surprise_logits,
            surprise=surprise,
            policy_logits=policy_logits,
            value_logits=value_logits,
            value=value,
            objective_value=objective_value,
            value_component_logits=value_component_logits,
            value_components=value_components,
            action_embeddings=action_embeddings,
            latent_policy_logits=latent_policy_logits,
            latent_action_embeddings=latent_action_embeddings,
            planner_q_logits=planner_q_logits,
            planner_q=planner_q,
            planner_objective_q=planner_objective_q,
            planner_q_component_logits=planner_q_component_logits,
            planner_q_components=planner_q_components,
            next_action_mask_logits=next_action_mask_logits,
            next_decision_domain_logits=next_decision_domain_logits,
            next_phase_logits=next_phase_logits,
            teacher_hidden_state=teacher_hidden_state,
            current_world_bank_state_ref=current_world_bank_state_ref,
            next_world_bank_state_pred=next_world_bank_state_pred,
            next_world_bank_state_target=next_world_bank_state_target,
            next_world_bank_delta_pred=next_world_bank_delta_pred,
            next_world_bank_delta_target=next_world_bank_delta_target,
            next_world_bank_occupancy_logits=next_world_bank_occupancy_logits,
            next_world_bank_occupancy_target=next_world_bank_occupancy_target,
            next_world_bank_token_presence_logits=next_world_bank_token_presence_logits,
            next_world_bank_token_presence_target=next_world_bank_token_presence_target,
            next_world_bank_token_distribution_logits=next_world_bank_token_distribution_logits,
            next_world_bank_token_distribution_target=next_world_bank_token_distribution_target,
            next_world_bank_token_slot_state_pred=next_world_bank_token_slot_state_pred,
            next_world_bank_token_slot_state_target=next_world_bank_token_slot_state_target,
            next_world_bank_token_slot_mask_logits=next_world_bank_token_slot_mask_logits,
            next_world_bank_token_slot_mask_target=next_world_bank_token_slot_mask_target,
            next_world_bank_token_slot_type_logits=next_world_bank_token_slot_type_logits,
            next_world_bank_token_slot_type_target=next_world_bank_token_slot_type_target,
            next_world_bank_token_slot_zone_logits=next_world_bank_token_slot_zone_logits,
            next_world_bank_token_slot_zone_target=next_world_bank_token_slot_zone_target,
            next_world_bank_token_slot_source_logits=next_world_bank_token_slot_source_logits,
            next_world_bank_token_slot_source_target=next_world_bank_token_slot_source_target,
        )

    def project_to_semantic_latent(self, hidden_state: torch.Tensor) -> torch.Tensor:
        if self.is_token_mode:
            return self.semantic_state_projector(hidden_state, latent_type_id=1)
        return self.semantic_state_projector(hidden_state)

    def semantic_action_embeddings(self) -> torch.Tensor:
        return self.semantic_action_embedding.weight

    def encode_semantic_action_indices(self, action_indices: torch.Tensor) -> torch.Tensor:
        return self.semantic_action_embedding(action_indices.long())

    def semantic_prediction(
        self,
        semantic_hidden_state: torch.Tensor,
        *,
        obs: dict[str, torch.Tensor] | None = None,
    ) -> SemanticMuZeroOutput:
        objective_context = obs.get("objective_context") if obs is not None else None
        decision_domain = self._decision_domain_tensor(obs)
        (
            semantic_policy_logits,
            value_logits,
            value,
            value_component_logits,
            value_components,
            objective_value,
        ) = self.prediction.semantic_forward(
            semantic_hidden_state,
            decision_domain=decision_domain,
            objective_context=objective_context,
        )
        return SemanticMuZeroOutput(
            semantic_hidden_state=semantic_hidden_state,
            semantic_policy_logits=semantic_policy_logits,
            value_logits=value_logits,
            value=value,
            objective_value=objective_value,
            value_component_logits=value_component_logits,
            value_components=value_components,
        )

    def semantic_recurrent_inference(
        self,
        semantic_hidden_state: torch.Tensor,
        semantic_action_index: torch.Tensor,
        *,
        next_obs: dict[str, torch.Tensor] | None = None,
    ) -> SemanticRecurrentMuZeroOutput:
        semantic_action_embedding = self.encode_semantic_action_indices(semantic_action_index)
        (
            next_semantic_hidden_state,
            reward_logits,
            reward,
            reward_component_logits,
            reward_components,
            surprise_logits,
            surprise,
        ) = self.semantic_dynamics(semantic_hidden_state, semantic_action_embedding)
        objective_context = next_obs.get("objective_context") if next_obs is not None else None
        decision_domain = self._decision_domain_tensor(next_obs)
        (
            semantic_policy_logits,
            value_logits,
            value,
            value_component_logits,
            value_components,
            objective_value,
        ) = self.prediction.semantic_forward(
            next_semantic_hidden_state,
            decision_domain=decision_domain,
            objective_context=objective_context,
        )
        return SemanticRecurrentMuZeroOutput(
            next_semantic_hidden_state=next_semantic_hidden_state,
            reward_logits=reward_logits,
            reward=reward,
            reward_component_logits=reward_component_logits,
            reward_components=reward_components,
            surprise_logits=surprise_logits,
            surprise=surprise,
            semantic_policy_logits=semantic_policy_logits,
            value_logits=value_logits,
            value=value,
            objective_value=objective_value,
            value_component_logits=value_component_logits,
            value_components=value_components,
        )

    def encode_actions(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode all action candidates to embeddings.

        Args:
            obs: observation dict with 'actions' and 'action_text' keys

        Returns:
            action embeddings of shape [batch, MAX_ACTIONS, action_embed_dim]
        """
        if self.is_token_mode:
            return self.token_encoder(obs).action_embeddings
        return self.action_encoder(obs)

    def get_action_embedding(self, action_idx: int, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract embedding for a specific action index.

        Args:
            action_idx: index in [0, MAX_ACTIONS)
            obs: observation dict

        Returns:
            action embedding of shape [batch, action_embed_dim]
        """
        action_embeddings = self.encode_actions(obs)
        return action_embeddings[:, action_idx, :]

    def project_hidden_state(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Project hidden states into a consistency space for dynamics alignment."""
        if self.is_token_mode:
            return self.state_projector(hidden_state, latent_type_id=0)
        return self.state_projector(hidden_state)


__all__ = [
    'ACTION_FEAT_DIM',
    'CARD_FEAT_DIM',
    'DECK_FEAT_DIM',
    'DEFAULT_ACTION_ROLLOUT_BUCKETS',
    'ENEMY_FEAT_DIM',
    'HEAD_HP_PRESERVATION',
    'HEAD_SURVIVAL',
    'MAX_ACTIONS',
    'MAX_DECK',
    'MAX_ENEMIES',
    'MAX_HAND',
    'MAX_POTIONS',
    'MAX_RELICS',
    'NUM_DOMAINS',
    'NUM_OBJECTIVE_HEADS',
    'NUM_PHASES',
    'OBJECTIVE_DIM',
    'POWER_DIM',
    'RELIC_SIGNAL_DIM',
    'RISK_OBJECTIVE_HEAD_INDICES',
    'ROUTE_NODE_FEAT_DIM',
    'ROUTE_SUMMARY_DIM',
    'RUN_MEMORY_DIM',
    'SCALAR_DIM',
    'SEMANTIC_ROLLOUT_FEAT_DIM',
    'SEMANTIC_ROLLOUT_SIZE',
    'SEM_ACTION_FEAT_DIM',
    'TEXT_DIM',
    'ActionEncoder',
    'ActionRolloutPlannerOutput',
    'CandidatePolicyScorer',
    'DomainActionEncoder',
    'DynamicsNetwork',
    'LatentActionBank',
    'MuZeroNetwork',
    'MuZeroNetworkOutput',
    'PredictionNetwork',
    'RecurrentMuZeroOutput',
    'RepresentationNetwork',
    'RouteActionEncoder',
    'SemanticMuZeroOutput',
    'SemanticRecurrentMuZeroOutput',
    'TokenDynamicsNetwork',
    'TokenFutureWorldBankHead',
    'TokenLatentProjector',
    'TokenMemoryEncoder',
    'TokenMemoryEncoderOutput',
    'TokenPredictionNetwork',
    'TokenTransitionSurfaceHead',
    'TransitionSurfaceHead',
    'infer_token_decision_domain',
    'scalar_to_support',
    'scalarize_objective_components_torch',
    'support_tensor_to_scalar',
    'support_to_scalar',
]
