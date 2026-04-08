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

import torch
import torch.nn as nn
from typing import NamedTuple

from .observation_v2 import (
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
    POWER_DIM,
    SCALAR_DIM,
)
from .text_encoder import TEXT_DIM


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


# ============================================================================
# ActionEncoder: encode action features + text to embeddings
# ============================================================================

class ActionEncoder(nn.Module):
    """Encode action features and text to action embeddings.

    Combines ACTION_FEAT_DIM features with TEXT_DIM text embeddings
    and projects to a unified action_embed_dim representation.
    """

    def __init__(self, action_embed_dim: int = 64, text_proj_dim: int = 32):
        """Initialize action encoder.

        Args:
            action_embed_dim: output embedding dimension (default 64)
            text_proj_dim: intermediate text projection dimension
        """
        super().__init__()
        self.action_embed_dim = action_embed_dim
        self.text_proj_dim = text_proj_dim

        # Project text embeddings
        self.text_proj = nn.Linear(TEXT_DIM, text_proj_dim)

        # Main encoding MLP: concat features + projected text -> embedding
        self.net = nn.Sequential(
            nn.Linear(ACTION_FEAT_DIM + text_proj_dim, action_embed_dim),
            nn.ReLU(),
            nn.Linear(action_embed_dim, action_embed_dim),
        )

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode action candidates.

        Args:
            obs: observation dict with keys:
                - actions: [batch, max_actions, ACTION_FEAT_DIM]
                - action_text: [batch, max_actions, TEXT_DIM]

        Returns:
            action embeddings of shape [batch, max_actions, action_embed_dim]
        """
        action_text = self.text_proj(obs["action_text"])
        combined = torch.cat([obs["actions"], action_text], dim=-1)
        return self.net(combined)


# ============================================================================
# RepresentationNetwork: encode observations to hidden state
# ============================================================================

class RepresentationNetwork(nn.Module):
    """Encode full observation to unified hidden state representation.

    Processes all observation types:
    - Shared context (scalars, powers, relics, potions, context_text)
    - Combat state (hand cards + enemies with cross-attention)
    - Build state (deck with self-attention)

    Outputs a single hidden_dim=256 vector.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        embed_dim: int = 64,
        n_heads: int = 4,
        text_proj_dim: int = 32,
    ):
        """Initialize representation network.

        Args:
            hidden_dim: output hidden state dimension (default 256)
            embed_dim: attention embedding dimension (default 64)
            n_heads: number of attention heads (default 4)
            text_proj_dim: text projection dimension (default 32)
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.text_proj_dim = text_proj_dim

        # ---- Shared Context Encoding ----
        self.context_text_proj = nn.Linear(TEXT_DIM, text_proj_dim)
        self.relic_text_proj = nn.Linear(TEXT_DIM, text_proj_dim)
        self.potion_text_proj = nn.Linear(TEXT_DIM, text_proj_dim)

        self.relic_enc = nn.Sequential(
            nn.Linear(text_proj_dim, 16),
            nn.ReLU(),
        )
        self.potion_enc = nn.Sequential(
            nn.Linear(text_proj_dim, 16),
            nn.ReLU(),
        )

        # ---- Combat State Encoding ----
        self.card_text_proj = nn.Linear(TEXT_DIM, text_proj_dim)
        self.enemy_text_proj = nn.Linear(TEXT_DIM, text_proj_dim)

        self.card_enc = nn.Sequential(
            nn.Linear(CARD_FEAT_DIM + text_proj_dim, embed_dim),
            nn.ReLU(),
        )
        self.enemy_enc = nn.Sequential(
            nn.Linear(ENEMY_FEAT_DIM + text_proj_dim, embed_dim),
            nn.ReLU(),
        )

        # Self-attention on cards and enemies
        self.card_attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.card_norm = nn.LayerNorm(embed_dim)
        self.enemy_attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.enemy_norm = nn.LayerNorm(embed_dim)

        # Cross-attention: cards attend to enemies
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, n_heads, kdim=embed_dim, vdim=embed_dim, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(embed_dim)

        # Pooling seeds and attention
        self.card_seed = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.enemy_seed = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.card_pool = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.enemy_pool = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)

        # ---- Build State Encoding (Deck) ----
        self.deck_text_proj = nn.Linear(TEXT_DIM, text_proj_dim)
        self.deck_enc = nn.Sequential(
            nn.Linear(DECK_FEAT_DIM + text_proj_dim, embed_dim),
            nn.ReLU(),
        )
        self.deck_attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.deck_norm = nn.LayerNorm(embed_dim)
        self.deck_seed = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.deck_pool = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)

        # ---- Final projection to hidden_dim ----
        # Inputs: scalars, powers, context_text, relics, potions, decision_domain,
        #         hand output, enemy output, deck output
        shared_size = SCALAR_DIM + POWER_DIM + text_proj_dim + 16 + 16 + NUM_DOMAINS
        combat_size = 2 * embed_dim  # hand + enemies pooled
        build_size = embed_dim  # deck pooled
        total_input = shared_size + combat_size + build_size

        self.final_net = nn.Sequential(
            nn.Linear(total_input, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode observation to hidden state.

        Args:
            obs: observation dict with all standard STS2 observation keys

        Returns:
            hidden state tensor of shape [batch, hidden_dim]
        """
        # ---- Encode Shared Context ----
        context = self.context_text_proj(obs["context_text"])
        relics = self.relic_text_proj(obs["relics"])
        potions = self.potion_text_proj(obs["potions"])

        relic_out = _masked_mean(self.relic_enc(relics), obs["relic_mask"].bool())
        potion_out = _masked_mean(self.potion_enc(potions), obs["potion_mask"].bool())

        shared = torch.cat([
            obs["scalars"],
            obs["player_powers"],
            context,
            relic_out,
            potion_out,
            obs["decision_domain"],
        ], dim=-1)

        # ---- Encode Combat State ----
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

        combat = torch.cat([hand_out, enemy_out], dim=-1)

        # ---- Encode Build State (Deck) ----
        deck_text = self.deck_text_proj(obs["deck_text"])
        deck_emb = self.deck_enc(torch.cat([obs["deck"], deck_text], dim=-1))
        deck_mask = obs["deck_mask"].bool()

        deck_emb = self.deck_norm(deck_emb + _safe_self_attn(self.deck_attn, deck_emb, deck_mask))
        deck_out = _safe_pool(self.deck_pool, self.deck_seed, deck_emb, deck_mask)

        # ---- Final projection ----
        combined = torch.cat([shared, combat, deck_out], dim=-1)
        hidden_state = self.final_net(combined)

        return hidden_state


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
        hidden_dim: int = 256,
        action_embed_dim: int = 64,
        support_size: int = 25,
    ):
        """Initialize dynamics network.

        Args:
            hidden_dim: hidden state dimension (default 256)
            action_embed_dim: action embedding dimension (default 64)
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.action_embed_dim = action_embed_dim
        self.support_size = support_size
        self.num_bins = 2 * support_size + 1

        # MLP for next hidden state prediction with residual
        self.dynamics_net = nn.Sequential(
            nn.Linear(hidden_dim + action_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.dynamics_norm = nn.LayerNorm(hidden_dim)

        # Separate small MLP for reward prediction
        self.reward_net = nn.Sequential(
            nn.Linear(hidden_dim + action_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.num_bins),
        )

    def forward(
        self,
        hidden_state: torch.Tensor,
        action_embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict next hidden state and reward.

        Args:
            hidden_state: [batch, hidden_dim]
            action_embedding: [batch, action_embed_dim]

        Returns:
            (
                next_hidden_state [batch, hidden_dim],
                reward_logits [batch, 2*support_size+1],
                reward [batch, 1]
            )
        """
        combined = torch.cat([hidden_state, action_embedding], dim=-1)

        # Dynamics with residual: next_state = norm(current + MLP(concat))
        dynamics_out = self.dynamics_net(combined)
        next_hidden = self.dynamics_norm(hidden_state + dynamics_out)

        # Reward prediction
        reward_logits = self.reward_net(combined)
        reward = support_to_scalar(reward_logits, self.support_size)

        return next_hidden, reward_logits, reward


# ============================================================================
# PredictionNetwork: predict policy and value from hidden state
# ============================================================================

class PredictionNetwork(nn.Module):
    """Predict policy (action logits) and value from hidden state.

    - Policy head: MLP to MAX_ACTIONS logits
    - Value head: categorical support-based value (51 bins for range [-5, 5])
    """

    def __init__(self, hidden_dim: int = 256, support_size: int = 25):
        """Initialize prediction network.

        Args:
            hidden_dim: hidden state dimension (default 256)
            support_size: number of support bins on each side (default 25, -> 51 total)
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.support_size = support_size
        self.num_bins = 2 * support_size + 1

        # Policy head: predict logits over MAX_ACTIONS
        self.policy_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, MAX_ACTIONS),
        )

        # Value head: categorical with support bins
        self.value_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.num_bins),
        )

    def forward(self, hidden_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict policy and value.

        Args:
            hidden_state: [batch, hidden_dim]

        Returns:
            (
                policy_logits [batch, MAX_ACTIONS],
                value_logits [batch, 2*support_size+1],
                value [batch, 1]
            )
        """
        policy_logits = self.policy_net(hidden_state)
        value_logits = self.value_net(hidden_state)

        # Convert categorical value to scalar
        value = support_to_scalar(value_logits, self.support_size)

        return policy_logits, value_logits, value


# ============================================================================
# MuZeroNetwork: main wrapper
# ============================================================================

class MuZeroNetworkOutput(NamedTuple):
    """Output from initial inference."""
    hidden_state: torch.Tensor
    policy_logits: torch.Tensor
    value_logits: torch.Tensor
    value: torch.Tensor


class RecurrentMuZeroOutput(NamedTuple):
    """Output from recurrent inference."""
    next_hidden_state: torch.Tensor
    reward_logits: torch.Tensor
    reward: torch.Tensor
    policy_logits: torch.Tensor
    value_logits: torch.Tensor
    value: torch.Tensor


class MuZeroNetwork(nn.Module):
    """MuZero-style model combining representation, dynamics, and prediction networks.

    Provides both initial_inference (initial state from observation) and
    recurrent_inference (future prediction from hidden state + action).
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        action_embed_dim: int = 64,
        embed_dim: int = 64,
        n_heads: int = 4,
        text_proj_dim: int = 32,
        support_size: int = 25,
    ):
        """Initialize MuZero network.

        Args:
            hidden_dim: hidden state dimension (default 256)
            action_embed_dim: action embedding dimension (default 64)
            embed_dim: attention embedding dimension (default 64)
            n_heads: number of attention heads (default 4)
            text_proj_dim: text projection dimension (default 32)
            support_size: value support size (default 25 -> 51 bins)
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.action_embed_dim = action_embed_dim
        self.support_size = support_size

        # Core networks
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
        )

        self.prediction = PredictionNetwork(
            hidden_dim=hidden_dim,
            support_size=support_size,
        )

        self.action_encoder = ActionEncoder(
            action_embed_dim=action_embed_dim,
            text_proj_dim=text_proj_dim,
        )

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
        hidden_state = self.representation(obs)
        policy_logits, value_logits, value = self.prediction(hidden_state)

        return MuZeroNetworkOutput(
            hidden_state=hidden_state,
            policy_logits=policy_logits,
            value_logits=value_logits,
            value=value,
        )

    def recurrent_inference(
        self,
        hidden_state: torch.Tensor,
        action_embedding: torch.Tensor,
    ) -> RecurrentMuZeroOutput:
        """Predict next state, reward, policy, and value from hidden state and action.

        Args:
            hidden_state: [batch, hidden_dim]
            action_embedding: [batch, action_embed_dim]

        Returns:
            RecurrentMuZeroOutput with next_hidden_state, reward, policy_logits, value
        """
        next_hidden_state, reward_logits, reward = self.dynamics(hidden_state, action_embedding)
        policy_logits, value_logits, value = self.prediction(next_hidden_state)

        return RecurrentMuZeroOutput(
            next_hidden_state=next_hidden_state,
            reward_logits=reward_logits,
            reward=reward,
            policy_logits=policy_logits,
            value_logits=value_logits,
            value=value,
        )

    def encode_actions(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode all action candidates to embeddings.

        Args:
            obs: observation dict with 'actions' and 'action_text' keys

        Returns:
            action embeddings of shape [batch, MAX_ACTIONS, action_embed_dim]
        """
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
