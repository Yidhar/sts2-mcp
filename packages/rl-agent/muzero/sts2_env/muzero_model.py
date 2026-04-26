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

import math

import torch
import torch.nn as nn
from typing import NamedTuple

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
    POWER_DIM,
    RELIC_SIGNAL_DIM,
    RUN_MEMORY_DIM,
    OBJECTIVE_DIM,
    SEM_ACTION_FEAT_DIM,
    ROUTE_NODE_FEAT_DIM,
    ROUTE_SUMMARY_DIM,
    SCALAR_DIM,
)
from muzero.sts2_env.semantic_rollout import SEMANTIC_ROLLOUT_FEAT_DIM, SEMANTIC_ROLLOUT_SIZE
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
from sts2_env.text_encoder import TEXT_DIM


RISK_OBJECTIVE_HEAD_INDICES = (HEAD_SURVIVAL, HEAD_HP_PRESERVATION)
DEFAULT_ACTION_ROLLOUT_BUCKETS = (8, 16, 32, 64, MAX_ACTIONS)


# ============================================================================
# Value Support Functions (Categorical value representation)
# ============================================================================

def scalar_to_support(value: torch.Tensor, support_size: int = 25) -> torch.Tensor:
    """Convert scalar value to categorical support representation.

    Args:
        value: scalar tensor of shape [batch]
        support_size: number of support points on each side (2*support_size+1 total)

    Returns:
        soft categorical target of shape [batch, 2*support_size+1]
    """
    batch_size = value.shape[0]
    device = value.device

    # Clamp value to support range [-support_size, support_size]
    value = torch.clamp(value, -support_size, support_size)

    # Compute lower and upper indices
    lower = torch.floor(value).long()
    upper = lower + 1

    # Compute interpolation weights
    weights_upper = (value - lower.float()).clamp(0, 1)
    weights_lower = 1.0 - weights_upper

    # Create target distribution
    num_bins = 2 * support_size + 1
    target = torch.zeros(batch_size, num_bins, device=device, dtype=torch.float32)

    # Offset indices to [0, num_bins)
    lower_offset = lower + support_size
    upper_offset = upper + support_size

    # Clip to valid range
    lower_offset = torch.clamp(lower_offset, 0, num_bins - 1)
    upper_offset = torch.clamp(upper_offset, 0, num_bins - 1)

    # Scatter weights
    target.scatter_(1, lower_offset.unsqueeze(1), weights_lower.unsqueeze(1))
    target.scatter_add_(1, upper_offset.unsqueeze(1), weights_upper.unsqueeze(1))

    return target


def support_to_scalar(logits: torch.Tensor, support_size: int = 25) -> torch.Tensor:
    """Convert categorical support logits to scalar value.

    Args:
        logits: shape [batch, 2*support_size+1]
        support_size: number of support points on each side

    Returns:
        scalar tensor of shape [batch, 1]
    """
    num_bins = 2 * support_size + 1
    device = logits.device

    # Compute probabilities
    probs = torch.softmax(logits, dim=-1)

    # Create support values [-support_size, ..., support_size]
    support_values = torch.arange(
        -support_size, support_size + 1, dtype=torch.float32, device=device
    )

    # Compute expected value
    value = (probs * support_values.unsqueeze(0)).sum(dim=1, keepdim=True)
    return value


def support_tensor_to_scalar(logits: torch.Tensor, support_size: int = 25) -> torch.Tensor:
    """Vectorized support-to-scalar for tensors with an arbitrary leading shape."""
    if logits.dim() == 2:
        return support_to_scalar(logits, support_size)
    flat = logits.reshape(-1, logits.shape[-1])
    scalars = support_to_scalar(flat, support_size)
    return scalars.reshape(*logits.shape[:-1])


# ============================================================================
# Helper Functions (reimplement from model.py patterns)
# ============================================================================

def _safe_self_attn(
    attn: nn.MultiheadAttention,
    x: torch.Tensor,
    mask: torch.Tensor
) -> torch.Tensor:
    """Apply self-attention with safe handling of empty sequences.

    Args:
        attn: MultiheadAttention module
        x: [batch, seq_len, dim]
        mask: [batch, seq_len] bool tensor (True = valid)

    Returns:
        attention output of same shape as x
    """
    if not mask.any():
        return torch.zeros_like(x)

    empty = ~mask.any(dim=1)
    safe_mask = mask.clone()
    safe_mask[empty, 0] = True

    out, _ = attn(x, x, x, key_padding_mask=~safe_mask)
    return out * mask.unsqueeze(-1).float()


def _safe_cross_attn(
    attn: nn.MultiheadAttention,
    query: torch.Tensor,
    key_value: torch.Tensor,
    query_mask: torch.Tensor,
    key_value_mask: torch.Tensor,
) -> torch.Tensor:
    """Apply cross-attention with safe handling of empty sequences.

    Args:
        attn: MultiheadAttention module
        query: [batch, query_len, dim]
        key_value: [batch, kv_len, dim]
        query_mask: [batch, query_len] bool tensor
        key_value_mask: [batch, kv_len] bool tensor

    Returns:
        attention output matching query shape
    """
    if not query_mask.any() or not key_value_mask.any():
        return torch.zeros_like(query)

    empty = ~key_value_mask.any(dim=1)
    safe_mask = key_value_mask.clone()
    safe_mask[empty, 0] = True

    out, _ = attn(query, key_value, key_value, key_padding_mask=~safe_mask)
    out = out * query_mask.unsqueeze(-1).float()
    out[empty] = 0.0
    return out


def _safe_pool(
    attn: nn.MultiheadAttention,
    seed: torch.Tensor,
    x: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Pool sequence using attention with learnable seed.

    Args:
        attn: MultiheadAttention module
        seed: [1, 1, dim]
        x: [batch, seq_len, dim]
        mask: [batch, seq_len] bool tensor

    Returns:
        pooled output [batch, dim]
    """
    batch_size, _, dim = x.shape
    if not mask.any():
        return torch.zeros(batch_size, dim, device=x.device)

    empty = ~mask.any(dim=1)
    safe_mask = mask.clone()
    safe_mask[empty, 0] = True

    out, _ = attn(seed.expand(batch_size, -1, -1), x, x, key_padding_mask=~safe_mask)
    pooled = out.squeeze(1)
    pooled[empty] = 0.0
    return pooled


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Compute masked mean across sequence dimension.

    Args:
        x: [batch, seq_len, dim]
        mask: [batch, seq_len] bool tensor

    Returns:
        mean of shape [batch, dim]
    """
    batch_size, _, dim = x.shape
    if not mask.any():
        return torch.zeros(batch_size, dim, device=x.device)

    weights = mask.unsqueeze(-1).float()
    return (x * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)


def _domain_masks_from_obs(obs: dict[str, torch.Tensor], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build hard routing masks from decision_domain one-hot vectors.

    Expected layout:
      0 = combat
      1 = build
      2 = route
    """
    domain = obs["decision_domain"]
    if domain.dim() != 2 or domain.shape[1] < NUM_DOMAINS:
        batch = domain.shape[0] if domain.dim() > 0 else 1
        default = torch.zeros(batch, dtype=torch.bool, device=device)
        return default, ~default, default

    domain_idx = torch.argmax(domain, dim=1)
    combat_mask = domain_idx == 0
    build_mask = domain_idx == 1
    route_mask = domain_idx == 2

    # Safety fallback: rows with invalid/empty domain route to build.
    unresolved = ~(combat_mask | build_mask | route_mask)
    if unresolved.any():
        build_mask = build_mask | unresolved

    return combat_mask, build_mask, route_mask


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


class ActionRolloutPlannerOutput(NamedTuple):
    """Explicit action-conditioned one-step latent rollout for search-free planning."""

    next_hidden_states: torch.Tensor
    reward_logits: torch.Tensor
    reward: torch.Tensor
    reward_component_logits: torch.Tensor
    reward_components: torch.Tensor
    next_value_logits: torch.Tensor
    next_value: torch.Tensor
    next_value_component_logits: torch.Tensor
    next_value_components: torch.Tensor
    next_objective_value: torch.Tensor
    next_action_mask_logits: torch.Tensor
    next_decision_domain_logits: torch.Tensor
    next_phase_logits: torch.Tensor
    planner_q: torch.Tensor
    planner_q_components: torch.Tensor
    planner_objective_q: torch.Tensor
    planner_risk_q: torch.Tensor
    planner_uncertainty: torch.Tensor
    planner_surprise: torch.Tensor
    planner_surface_entropy: torch.Tensor
    planner_latent_drift: torch.Tensor
    planner_branch_disagreement: torch.Tensor
    action_mask: torch.Tensor
    rollout_steps_used: int
    rollout_branch_count_mean: float
    rollout_root_valid_count: int
    rollout_root_bucket_size: int
    rollout_bucket_padding_ratio: float
    rollout_max_branch_bucket_size: int
    rollout_branch_padding_ratio: float


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


class MuZeroNetwork(nn.Module):
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
                support_size=support_size,
                internal_planner_blend=token_internal_planner_blend,
                internal_planner_q_blend=token_internal_planner_q_blend,
                internal_planner_objective_q_blend=token_internal_planner_objective_q_blend,
                internal_planner_risk_blend=token_internal_planner_risk_blend,
                n_heads=token_n_heads,
                ffn_dim=token_ffn_dim,
                dropout=token_dropout,
                activation_checkpointing=activation_checkpointing,
            )
            self.transition_surface = TokenTransitionSurfaceHead(
                hidden_dim=hidden_dim,
                d_model=token_d_model,
                num_memory_slots=token_memory_slots,
                n_heads=token_n_heads,
                ffn_dim=token_ffn_dim,
                dropout=token_dropout,
                activation_checkpointing=activation_checkpointing,
            )
            self.future_world_bank_head = TokenFutureWorldBankHead(
                hidden_dim=hidden_dim,
                d_model=token_d_model,
                num_memory_slots=token_memory_slots,
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
                support_size=support_size,
                num_transition_layers=max(2, dynamics_res_blocks // 2),
                dropout=token_dropout,
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

    @staticmethod
    def _normalize_action_rollout_buckets(
        raw_buckets: str | tuple[int, ...] | list[int] | None,
    ) -> tuple[int, ...]:
        if raw_buckets is None:
            values = list(DEFAULT_ACTION_ROLLOUT_BUCKETS)
        elif isinstance(raw_buckets, str):
            parts = [
                part.strip()
                for part in raw_buckets.replace(";", ",").split(",")
                if part.strip()
            ]
            values = [int(part) for part in parts]
        else:
            values = [int(value) for value in raw_buckets]

        normalized = sorted({int(value) for value in values if int(value) > 0})
        if not normalized:
            normalized = list(DEFAULT_ACTION_ROLLOUT_BUCKETS)
        if normalized[-1] < MAX_ACTIONS:
            normalized.append(MAX_ACTIONS)
        return tuple(normalized)

    def _action_rollout_bucket_size(self, active_count: int) -> int:
        active_count = max(int(active_count), 0)
        if active_count <= 0:
            return 0
        for bucket in self.action_rollout_buckets:
            if active_count <= int(bucket):
                return int(bucket)
        largest = max(int(self.action_rollout_buckets[-1]), 1)
        return int(math.ceil(active_count / float(largest)) * largest)

    @staticmethod
    def _pad_first_dim_to_bucket(
        tensor: torch.Tensor,
        bucket_size: int,
        *,
        fill_value: float | int | bool = 0,
    ) -> torch.Tensor:
        active_count = int(tensor.shape[0])
        bucket_size = int(bucket_size)
        if active_count == bucket_size:
            return tensor
        if active_count > bucket_size:
            return tensor[:bucket_size]
        pad_shape = (bucket_size - active_count, *tensor.shape[1:])
        padding = tensor.new_full(pad_shape, fill_value)
        return torch.cat([tensor, padding], dim=0)

    def action_rollout_planner(
        self,
        hidden_state: torch.Tensor,
        action_embeddings: torch.Tensor,
        *,
        action_mask: torch.Tensor | None = None,
        objective_context: torch.Tensor | None = None,
        decision_domain: torch.Tensor | None = None,
        discount: float = 0.997,
        rollout_steps: int = 1,
        continuation_beam_width: int = 2,
        continuation_legal_logit_scale: float = 0.75,
        uncertainty_surprise_weight: float = 1.0,
        uncertainty_surface_entropy_weight: float = 0.10,
        uncertainty_latent_drift_weight: float = 0.05,
        uncertainty_branch_disagreement_weight: float = 0.25,
        continuation_uncertainty_penalty: float = 0.25,
    ) -> ActionRolloutPlannerOutput:
        """Roll shared dynamics for every candidate action, optionally with deeper latent continuation.

        This is the explicit search-free planner path:
        - generate next latent state for each candidate action
        - predict next-step value/objective value from the shared value heads
        - optionally continue with latent-action beam rollout using predicted legality/surface
        - combine discounted rewards + terminal bootstrap into Q-like action scores
        """

        batch_size, action_count, _ = action_embeddings.shape
        device = hidden_state.device
        rollout_steps = max(int(rollout_steps), 1)
        continuation_beam_width = max(int(continuation_beam_width), 1)
        if action_mask is None:
            action_mask = action_embeddings.abs().sum(dim=-1) > 0
        else:
            action_mask = action_mask.to(device=device, dtype=torch.bool)
            if action_mask.dim() == 1:
                action_mask = action_mask.unsqueeze(0)
        action_mask = action_mask.reshape(batch_size, action_count)
        flat_valid = action_mask.reshape(-1)
        valid_indices = torch.nonzero(flat_valid, as_tuple=False).flatten()
        num_valid = int(valid_indices.numel())
        root_bucket_size = self._action_rollout_bucket_size(num_valid)
        root_bucket_padding_ratio = (
            float(root_bucket_size - num_valid) / float(root_bucket_size)
            if root_bucket_size > 0
            else 0.0
        )
        num_bins = 2 * self.support_size + 1

        def _zeros(*shape: int, dtype: torch.dtype | None = None) -> torch.Tensor:
            return torch.zeros(*shape, device=device, dtype=dtype or hidden_state.dtype)

        if num_valid <= 0:
            return ActionRolloutPlannerOutput(
                next_hidden_states=_zeros(batch_size, action_count, self.hidden_dim),
                reward_logits=_zeros(batch_size, action_count, num_bins),
                reward=_zeros(batch_size, action_count),
                reward_component_logits=_zeros(batch_size, action_count, NUM_OBJECTIVE_HEADS, num_bins),
                reward_components=_zeros(batch_size, action_count, NUM_OBJECTIVE_HEADS),
                next_value_logits=_zeros(batch_size, action_count, num_bins),
                next_value=_zeros(batch_size, action_count),
                next_value_component_logits=_zeros(batch_size, action_count, NUM_OBJECTIVE_HEADS, num_bins),
                next_value_components=_zeros(batch_size, action_count, NUM_OBJECTIVE_HEADS),
                next_objective_value=_zeros(batch_size, action_count),
                next_action_mask_logits=_zeros(batch_size, action_count, MAX_ACTIONS),
                next_decision_domain_logits=_zeros(batch_size, action_count, NUM_DOMAINS),
                next_phase_logits=_zeros(batch_size, action_count, NUM_PHASES),
                planner_q=_zeros(batch_size, action_count),
                planner_q_components=_zeros(batch_size, action_count, NUM_OBJECTIVE_HEADS),
                planner_objective_q=_zeros(batch_size, action_count),
                planner_risk_q=_zeros(batch_size, action_count),
                planner_uncertainty=_zeros(batch_size, action_count),
                planner_surprise=_zeros(batch_size, action_count),
                planner_surface_entropy=_zeros(batch_size, action_count),
                planner_latent_drift=_zeros(batch_size, action_count),
                planner_branch_disagreement=_zeros(batch_size, action_count),
                action_mask=action_mask,
                rollout_steps_used=0,
                rollout_branch_count_mean=0.0,
                rollout_root_valid_count=0,
                rollout_root_bucket_size=0,
                rollout_bucket_padding_ratio=0.0,
                rollout_max_branch_bucket_size=0,
                rollout_branch_padding_ratio=0.0,
            )

        hidden_expanded = hidden_state.unsqueeze(1).expand(-1, action_count, -1)
        root_active_mask = torch.arange(root_bucket_size, device=device) < num_valid
        root_active_float = root_active_mask.to(dtype=hidden_state.dtype)
        safe_valid_indices = self._pad_first_dim_to_bucket(valid_indices, root_bucket_size, fill_value=0)
        flat_hidden = hidden_expanded.reshape(batch_size * action_count, self.hidden_dim).index_select(
            0,
            safe_valid_indices,
        )
        flat_actions = action_embeddings.reshape(batch_size * action_count, -1).index_select(0, safe_valid_indices)
        flat_hidden = flat_hidden * root_active_float.unsqueeze(-1)
        flat_actions = flat_actions * root_active_float.unsqueeze(-1)

        repeated_objective_context: torch.Tensor | None = None
        bucket_objective_context: torch.Tensor | None = None
        if objective_context is not None:
            if objective_context.dim() == 1:
                objective_context = objective_context.unsqueeze(0)
            bucket_objective_context = (
                objective_context
                .unsqueeze(1)
                .expand(-1, action_count, -1)
                .reshape(batch_size * action_count, -1)
                .index_select(0, safe_valid_indices)
            )
            bucket_objective_context = bucket_objective_context * root_active_float.unsqueeze(-1)
            repeated_objective_context = bucket_objective_context[:num_valid]

        (
            next_hidden_valid,
            reward_logits_valid,
            reward_valid,
            reward_component_logits_valid,
            reward_components_valid,
            surprise_logits_valid,
            surprise_valid,
        ) = self.dynamics(flat_hidden, flat_actions)
        (
            next_action_mask_logits_valid,
            next_decision_domain_logits_valid,
            next_phase_logits_valid,
        ) = self.transition_surface(next_hidden_valid)
        next_decision_domain_valid = torch.softmax(next_decision_domain_logits_valid, dim=-1)
        (
            next_value_logits_valid,
            next_value_valid,
            next_value_component_logits_valid,
            next_value_components_valid,
            next_objective_value_valid,
        ) = self.prediction.value_only(
            next_hidden_valid,
            decision_domain=next_decision_domain_valid,
            objective_context=bucket_objective_context,
        )
        if reward_valid.dim() > 1 and reward_valid.shape[-1] == 1:
            reward_valid = reward_valid.squeeze(-1)
        if next_value_valid.dim() > 1 and next_value_valid.shape[-1] == 1:
            next_value_valid = next_value_valid.squeeze(-1)
        if next_objective_value_valid.dim() > 1 and next_objective_value_valid.shape[-1] == 1:
            next_objective_value_valid = next_objective_value_valid.squeeze(-1)
        if surprise_valid.dim() > 1 and surprise_valid.shape[-1] == 1:
            surprise_valid = surprise_valid.squeeze(-1)

        next_hidden_valid = next_hidden_valid[:num_valid]
        reward_logits_valid = reward_logits_valid[:num_valid]
        reward_valid = reward_valid[:num_valid]
        reward_component_logits_valid = reward_component_logits_valid[:num_valid]
        reward_components_valid = reward_components_valid[:num_valid]
        surprise_valid = surprise_valid[:num_valid]
        next_action_mask_logits_valid = next_action_mask_logits_valid[:num_valid]
        next_decision_domain_logits_valid = next_decision_domain_logits_valid[:num_valid]
        next_phase_logits_valid = next_phase_logits_valid[:num_valid]
        next_value_logits_valid = next_value_logits_valid[:num_valid]
        next_value_valid = next_value_valid[:num_valid]
        next_value_component_logits_valid = next_value_component_logits_valid[:num_valid]
        next_value_components_valid = next_value_components_valid[:num_valid]
        next_objective_value_valid = next_objective_value_valid[:num_valid]

        planner_q_components_valid = reward_components_valid + float(discount) * next_value_components_valid
        planner_q_valid = reward_valid + float(discount) * next_value_valid
        planner_objective_q_valid = scalarize_objective_components_torch(
            planner_q_components_valid,
            repeated_objective_context,
        )
        planner_risk_q_valid = planner_q_components_valid[..., list(RISK_OBJECTIVE_HEAD_INDICES)].mean(dim=-1)
        action_surface_prob = torch.sigmoid(next_action_mask_logits_valid)
        surface_entropy_valid = -(
            action_surface_prob * action_surface_prob.clamp(min=1e-6).log()
            + (1.0 - action_surface_prob) * (1.0 - action_surface_prob).clamp(min=1e-6).log()
        ).mean(dim=-1)
        expected_norm = math.sqrt(float(max(self.hidden_dim, 1)))
        latent_drift_valid = (next_hidden_valid.norm(dim=-1) / expected_norm - 1.0).abs()
        planner_surprise_valid = surprise_valid
        planner_surface_entropy_valid = surface_entropy_valid
        planner_latent_drift_valid = latent_drift_valid
        planner_branch_disagreement_valid = planner_q_valid.new_zeros(planner_q_valid.shape)
        planner_uncertainty_valid = (
            float(uncertainty_surprise_weight) * planner_surprise_valid
            + float(uncertainty_surface_entropy_weight) * planner_surface_entropy_valid
            + float(uncertainty_latent_drift_weight) * planner_latent_drift_valid
            + float(uncertainty_branch_disagreement_weight) * planner_branch_disagreement_valid
        )

        rollout_steps_used = 1
        rollout_branch_count_mean = 1.0
        rollout_max_branch_bucket_size = 0
        rollout_branch_bucket_total = 0.0
        rollout_branch_padding_total = 0.0

        if rollout_steps > 1:
            branch_root_index = torch.arange(num_valid, device=device, dtype=torch.long)
            branch_hidden = next_hidden_valid
            branch_action_mask_logits = next_action_mask_logits_valid
            branch_decision_domain_logits = next_decision_domain_logits_valid
            branch_return = reward_valid
            branch_components = reward_components_valid
            branch_surprise = planner_surprise_valid
            branch_surface_entropy = planner_surface_entropy_valid
            branch_latent_drift = planner_latent_drift_valid
            branch_weight = torch.ones_like(branch_return)
            branch_discount_power = torch.full_like(branch_return, float(discount))
            accumulated_branch_states = float(branch_root_index.numel())

            for depth in range(1, rollout_steps):
                if branch_hidden.numel() == 0:
                    break
                branch_active_count = int(branch_hidden.shape[0])
                branch_bucket_size = self._action_rollout_bucket_size(branch_active_count)
                rollout_max_branch_bucket_size = max(rollout_max_branch_bucket_size, int(branch_bucket_size))
                rollout_branch_bucket_total += float(branch_bucket_size)
                rollout_branch_padding_total += float(max(branch_bucket_size - branch_active_count, 0))
                branch_active_mask = torch.arange(branch_bucket_size, device=device) < branch_active_count
                branch_hidden_bucket = self._pad_first_dim_to_bucket(branch_hidden, branch_bucket_size)
                branch_decision_domain_logits_bucket = self._pad_first_dim_to_bucket(
                    branch_decision_domain_logits,
                    branch_bucket_size,
                )
                current_decision_domain = torch.softmax(branch_decision_domain_logits_bucket, dim=-1)
                latent_policy_logits, latent_action_embeddings = self.prediction.latent_policy_only(
                    branch_hidden_bucket,
                    decision_domain=current_decision_domain,
                )
                latent_policy_logits = latent_policy_logits[:branch_active_count]
                latent_action_embeddings = latent_action_embeddings[:branch_active_count]
                legal_mask = branch_action_mask_logits > 0.0
                any_legal = legal_mask.any(dim=-1, keepdim=True)
                fallback_mask = torch.ones_like(legal_mask, dtype=torch.bool)
                effective_mask = torch.where(any_legal, legal_mask, fallback_mask)
                effective_scores = latent_policy_logits + float(continuation_legal_logit_scale) * branch_action_mask_logits
                effective_scores = effective_scores.masked_fill(~effective_mask, -1e9)
                branch_k = min(continuation_beam_width, effective_scores.shape[1])
                top_scores, top_indices = torch.topk(effective_scores, k=branch_k, dim=-1)
                top_probs = torch.softmax(top_scores, dim=-1)
                top_probs = torch.where(
                    torch.isfinite(top_probs),
                    top_probs,
                    torch.full_like(top_probs, 1.0 / max(branch_k, 1)),
                )
                gathered_indices = top_indices.unsqueeze(-1).expand(-1, -1, latent_action_embeddings.shape[-1])
                top_action_embeddings = latent_action_embeddings.gather(1, gathered_indices)

                branch_count = branch_hidden.shape[0]
                expanded_hidden = (
                    branch_hidden.unsqueeze(1)
                    .expand(-1, branch_k, -1)
                    .reshape(branch_count * branch_k, self.hidden_dim)
                )
                expanded_actions = top_action_embeddings.reshape(branch_count * branch_k, self.action_embed_dim)
                expanded_root_index = (
                    branch_root_index.unsqueeze(1)
                    .expand(-1, branch_k)
                    .reshape(branch_count * branch_k)
                )
                expanded_weight = (
                    branch_weight.unsqueeze(1)
                    .expand(-1, branch_k)
                    .reshape(branch_count * branch_k)
                    * top_probs.reshape(branch_count * branch_k)
                )
                expanded_discount_power = (
                    branch_discount_power.unsqueeze(1)
                    .expand(-1, branch_k)
                    .reshape(branch_count * branch_k)
                )
                expanded_return = (
                    branch_return.unsqueeze(1)
                    .expand(-1, branch_k)
                    .reshape(branch_count * branch_k)
                )
                expanded_components = (
                    branch_components.unsqueeze(1)
                    .expand(-1, branch_k, -1)
                    .reshape(branch_count * branch_k, NUM_OBJECTIVE_HEADS)
                )
                expanded_surprise = (
                    branch_surprise.unsqueeze(1)
                    .expand(-1, branch_k)
                    .reshape(branch_count * branch_k)
                )
                expanded_surface_entropy = (
                    branch_surface_entropy.unsqueeze(1)
                    .expand(-1, branch_k)
                    .reshape(branch_count * branch_k)
                )
                expanded_latent_drift = (
                    branch_latent_drift.unsqueeze(1)
                    .expand(-1, branch_k)
                    .reshape(branch_count * branch_k)
                )
                expanded_objective_context: torch.Tensor | None = None
                if repeated_objective_context is not None:
                    expanded_objective_context = repeated_objective_context[expanded_root_index]
                expanded_active_count = int(expanded_hidden.shape[0])
                expanded_bucket_size = self._action_rollout_bucket_size(expanded_active_count)
                rollout_max_branch_bucket_size = max(rollout_max_branch_bucket_size, int(expanded_bucket_size))
                rollout_branch_bucket_total += float(expanded_bucket_size)
                rollout_branch_padding_total += float(max(expanded_bucket_size - expanded_active_count, 0))
                expanded_hidden_bucket = self._pad_first_dim_to_bucket(expanded_hidden, expanded_bucket_size)
                expanded_actions_bucket = self._pad_first_dim_to_bucket(expanded_actions, expanded_bucket_size)
                expanded_objective_context_bucket: torch.Tensor | None = None
                if expanded_objective_context is not None:
                    expanded_objective_context_bucket = self._pad_first_dim_to_bucket(
                        expanded_objective_context,
                        expanded_bucket_size,
                    )

                (
                    continued_hidden,
                    _continued_reward_logits,
                    continued_reward,
                    _continued_reward_component_logits,
                    continued_reward_components,
                    _continued_surprise_logits,
                    continued_surprise,
                ) = self.dynamics(expanded_hidden_bucket, expanded_actions_bucket)
                (
                    continued_action_mask_logits,
                    continued_decision_domain_logits,
                    _continued_phase_logits,
                ) = self.transition_surface(continued_hidden)
                continued_decision_domain = torch.softmax(continued_decision_domain_logits, dim=-1)
                (
                    _continued_value_logits,
                    continued_value,
                    _continued_value_component_logits,
                    continued_value_components,
                    _continued_objective_value,
                ) = self.prediction.value_only(
                    continued_hidden,
                    decision_domain=continued_decision_domain,
                    objective_context=expanded_objective_context_bucket,
                )
                if continued_reward.dim() > 1 and continued_reward.shape[-1] == 1:
                    continued_reward = continued_reward.squeeze(-1)
                if continued_value.dim() > 1 and continued_value.shape[-1] == 1:
                    continued_value = continued_value.squeeze(-1)
                if continued_surprise.dim() > 1 and continued_surprise.shape[-1] == 1:
                    continued_surprise = continued_surprise.squeeze(-1)
                continued_hidden = continued_hidden[:expanded_active_count]
                continued_reward = continued_reward[:expanded_active_count]
                continued_reward_components = continued_reward_components[:expanded_active_count]
                continued_surprise = continued_surprise[:expanded_active_count]
                continued_action_mask_logits = continued_action_mask_logits[:expanded_active_count]
                continued_decision_domain_logits = continued_decision_domain_logits[:expanded_active_count]
                continued_value = continued_value[:expanded_active_count]
                continued_value_components = continued_value_components[:expanded_active_count]

                continued_return = expanded_return + expanded_discount_power * continued_reward
                continued_components = (
                    expanded_components
                    + expanded_discount_power.unsqueeze(-1) * continued_reward_components
                )
                continued_surface_prob = torch.sigmoid(continued_action_mask_logits)
                continued_surface_entropy = -(
                    continued_surface_prob * continued_surface_prob.clamp(min=1e-6).log()
                    + (1.0 - continued_surface_prob)
                    * (1.0 - continued_surface_prob).clamp(min=1e-6).log()
                ).mean(dim=-1)
                continued_latent_drift = (continued_hidden.norm(dim=-1) / expected_norm - 1.0).abs()
                continued_surprise_total = expanded_surprise + expanded_discount_power * continued_surprise
                continued_surface_entropy_total = (
                    expanded_surface_entropy + expanded_discount_power * continued_surface_entropy
                )
                continued_latent_drift_total = expanded_latent_drift + expanded_discount_power * continued_latent_drift
                continued_uncertainty = (
                    float(uncertainty_surprise_weight) * continued_surprise_total
                    + float(uncertainty_surface_entropy_weight) * continued_surface_entropy_total
                    + float(uncertainty_latent_drift_weight) * continued_latent_drift_total
                )
                next_discount_power = expanded_discount_power * float(discount)
                prune_q_components = (
                    continued_components
                    + next_discount_power.unsqueeze(-1) * continued_value_components
                )
                prune_scores = scalarize_objective_components_torch(
                    prune_q_components,
                    expanded_objective_context,
                )
                prune_scores = prune_scores - float(continuation_uncertainty_penalty) * continued_uncertainty

                # Scheme-B: keep continuation pruning in the same allocator-stable,
                # root-major layout as the heavy dynamics/value calls above.
                #
                # Invariant maintained by this planner:
                #   - branch tensors are root-major: all paths for root 0, then root 1, ...
                #   - every active branch expands the same branch_k latent actions.  When the
                #     predicted legal surface is empty we deliberately use a full fallback mask,
                #     so a root never disappears mid-rollout.
                #
                # Therefore candidates per root are regular and can be reshaped to
                # [num_roots, candidates_per_root].  The previous implementation used
                # per-root torch.nonzero + Python list + torch.cat, producing many small
                # dynamic tensors on ROCm/HIP.  Here topk runs on a root-bucket-padded
                # score matrix and only the active rows are flattened back to indices.
                if expanded_active_count <= 0:
                    break
                candidates_per_root = expanded_active_count // max(num_valid, 1)
                if candidates_per_root <= 0 or candidates_per_root * num_valid != expanded_active_count:
                    raise RuntimeError(
                        "action_rollout_planner expected root-major regular continuation "
                        f"layout, got expanded_active_count={expanded_active_count}, "
                        f"num_valid={num_valid}, branch_count={branch_count}, branch_k={branch_k}"
                    )
                root_keep_k = min(continuation_beam_width, candidates_per_root)
                grouped_prune_scores = prune_scores.reshape(num_valid, candidates_per_root)
                grouped_prune_scores_bucket = prune_scores.new_full(
                    (root_bucket_size, candidates_per_root),
                    -1e9,
                )
                grouped_prune_scores_bucket[:num_valid] = grouped_prune_scores
                _, local_keep_bucket = torch.topk(
                    grouped_prune_scores_bucket,
                    k=root_keep_k,
                    dim=-1,
                )
                local_keep = local_keep_bucket[:num_valid]
                root_offsets = (
                    torch.arange(num_valid, device=device, dtype=torch.long).unsqueeze(1)
                    * candidates_per_root
                )
                keep_indices = (root_offsets + local_keep).reshape(num_valid * root_keep_k)

                branch_root_index = expanded_root_index[keep_indices]
                branch_hidden = continued_hidden[keep_indices]
                branch_action_mask_logits = continued_action_mask_logits[keep_indices]
                branch_decision_domain_logits = continued_decision_domain_logits[keep_indices]
                branch_return = continued_return[keep_indices]
                branch_components = continued_components[keep_indices]
                branch_surprise = continued_surprise_total[keep_indices]
                branch_surface_entropy = continued_surface_entropy_total[keep_indices]
                branch_latent_drift = continued_latent_drift_total[keep_indices]
                branch_weight = expanded_weight[keep_indices]
                branch_discount_power = next_discount_power[keep_indices]
                rollout_steps_used = depth + 1
                accumulated_branch_states += float(branch_root_index.numel())

            if branch_hidden.numel() > 0:
                final_branch_count = int(branch_hidden.shape[0])
                final_bucket_size = self._action_rollout_bucket_size(final_branch_count)
                rollout_max_branch_bucket_size = max(rollout_max_branch_bucket_size, int(final_bucket_size))
                rollout_branch_bucket_total += float(final_bucket_size)
                rollout_branch_padding_total += float(max(final_bucket_size - final_branch_count, 0))
                branch_hidden_bucket = self._pad_first_dim_to_bucket(branch_hidden, final_bucket_size)
                branch_decision_domain_logits_bucket = self._pad_first_dim_to_bucket(
                    branch_decision_domain_logits,
                    final_bucket_size,
                )
                branch_decision_domain = torch.softmax(branch_decision_domain_logits_bucket, dim=-1)
                branch_objective_context: torch.Tensor | None = None
                if repeated_objective_context is not None:
                    branch_objective_context = repeated_objective_context[branch_root_index]
                    branch_objective_context = self._pad_first_dim_to_bucket(
                        branch_objective_context,
                        final_bucket_size,
                    )
                (
                    _final_value_logits,
                    final_branch_value,
                    _final_value_component_logits,
                    final_branch_value_components,
                    _final_objective_value,
                ) = self.prediction.value_only(
                    branch_hidden_bucket,
                    decision_domain=branch_decision_domain,
                    objective_context=branch_objective_context,
                )
                if final_branch_value.dim() > 1 and final_branch_value.shape[-1] == 1:
                    final_branch_value = final_branch_value.squeeze(-1)
                final_branch_value = final_branch_value[:final_branch_count]
                final_branch_value_components = final_branch_value_components[:final_branch_count]
                final_branch_q = branch_return + branch_discount_power * final_branch_value
                final_branch_q_components = (
                    branch_components
                    + branch_discount_power.unsqueeze(-1) * final_branch_value_components
                )
                aggregated_q = final_branch_q.new_zeros((num_valid,))
                aggregated_q_components = final_branch_q_components.new_zeros((num_valid, NUM_OBJECTIVE_HEADS))
                aggregated_q_second_moment = final_branch_q.new_zeros((num_valid,))
                aggregated_surprise = final_branch_q.new_zeros((num_valid,))
                aggregated_surface_entropy = final_branch_q.new_zeros((num_valid,))
                aggregated_latent_drift = final_branch_q.new_zeros((num_valid,))
                aggregated_weights = final_branch_q.new_zeros((num_valid,))
                aggregated_q.index_add_(0, branch_root_index, branch_weight * final_branch_q)
                aggregated_q_second_moment.index_add_(
                    0,
                    branch_root_index,
                    branch_weight * final_branch_q.square(),
                )
                aggregated_q_components.index_add_(
                    0,
                    branch_root_index,
                    branch_weight.unsqueeze(-1) * final_branch_q_components,
                )
                aggregated_surprise.index_add_(0, branch_root_index, branch_weight * branch_surprise)
                aggregated_surface_entropy.index_add_(
                    0,
                    branch_root_index,
                    branch_weight * branch_surface_entropy,
                )
                aggregated_latent_drift.index_add_(0, branch_root_index, branch_weight * branch_latent_drift)
                aggregated_weights.index_add_(0, branch_root_index, branch_weight)
                normalized_weights = aggregated_weights.clamp(min=1e-6)
                planner_q_valid = aggregated_q / normalized_weights
                planner_q_components_valid = aggregated_q_components / normalized_weights.unsqueeze(-1)
                branch_q_second_moment_valid = aggregated_q_second_moment / normalized_weights
                planner_branch_disagreement_valid = (
                    branch_q_second_moment_valid - planner_q_valid.square()
                ).clamp(min=1e-8).sqrt()
                planner_surprise_valid = aggregated_surprise / normalized_weights
                planner_surface_entropy_valid = aggregated_surface_entropy / normalized_weights
                planner_latent_drift_valid = aggregated_latent_drift / normalized_weights
                planner_uncertainty_valid = (
                    float(uncertainty_surprise_weight) * planner_surprise_valid
                    + float(uncertainty_surface_entropy_weight) * planner_surface_entropy_valid
                    + float(uncertainty_latent_drift_weight) * planner_latent_drift_valid
                    + float(uncertainty_branch_disagreement_weight) * planner_branch_disagreement_valid
                )
                planner_objective_q_valid = scalarize_objective_components_torch(
                    planner_q_components_valid,
                    repeated_objective_context,
                )
                planner_risk_q_valid = planner_q_components_valid[..., list(RISK_OBJECTIVE_HEAD_INDICES)].mean(dim=-1)
                rollout_branch_count_mean = accumulated_branch_states / max(float(num_valid * rollout_steps_used), 1.0)

        rollout_branch_padding_ratio = (
            rollout_branch_padding_total / rollout_branch_bucket_total
            if rollout_branch_bucket_total > 0.0
            else 0.0
        )

        def _scatter(values: torch.Tensor) -> torch.Tensor:
            out = values.new_zeros((batch_size * action_count, *values.shape[1:]))
            out[flat_valid] = values
            return out.reshape(batch_size, action_count, *values.shape[1:])

        return ActionRolloutPlannerOutput(
            next_hidden_states=_scatter(next_hidden_valid),
            reward_logits=_scatter(reward_logits_valid),
            reward=_scatter(reward_valid),
            reward_component_logits=_scatter(reward_component_logits_valid),
            reward_components=_scatter(reward_components_valid),
            next_value_logits=_scatter(next_value_logits_valid),
            next_value=_scatter(next_value_valid),
            next_value_component_logits=_scatter(next_value_component_logits_valid),
            next_value_components=_scatter(next_value_components_valid),
            next_objective_value=_scatter(next_objective_value_valid),
            next_action_mask_logits=_scatter(next_action_mask_logits_valid),
            next_decision_domain_logits=_scatter(next_decision_domain_logits_valid),
            next_phase_logits=_scatter(next_phase_logits_valid),
            planner_q=_scatter(planner_q_valid),
            planner_q_components=_scatter(planner_q_components_valid),
            planner_objective_q=_scatter(planner_objective_q_valid),
            planner_risk_q=_scatter(planner_risk_q_valid),
            planner_uncertainty=_scatter(planner_uncertainty_valid),
            planner_surprise=_scatter(planner_surprise_valid),
            planner_surface_entropy=_scatter(planner_surface_entropy_valid),
            planner_latent_drift=_scatter(planner_latent_drift_valid),
            planner_branch_disagreement=_scatter(planner_branch_disagreement_valid),
            action_mask=action_mask,
            rollout_steps_used=rollout_steps_used,
            rollout_branch_count_mean=float(rollout_branch_count_mean),
            rollout_root_valid_count=int(num_valid),
            rollout_root_bucket_size=int(root_bucket_size),
            rollout_bucket_padding_ratio=float(root_bucket_padding_ratio),
            rollout_max_branch_bucket_size=int(rollout_max_branch_bucket_size),
            rollout_branch_padding_ratio=float(rollout_branch_padding_ratio),
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
