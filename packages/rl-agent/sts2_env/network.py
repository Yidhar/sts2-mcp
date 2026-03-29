"""Dual-tower feature encoders for STS2 RL.

StateEncoder: scalars + hand/enemy attention + cross-attention + relics/potions + context
ActionEncoder: per-action numeric + text projection

Both output fixed-dim embeddings for the candidate-scoring policy.
"""

import torch
import torch.nn as nn

from .text_encoder import TEXT_DIM
from .observation_v2 import (
    SCALAR_DIM, CARD_FEAT_DIM, ENEMY_FEAT_DIM, POWER_DIM,
    ACTION_FEAT_DIM, MAX_ACTIONS,
)


class StateEncoder(nn.Module):
    """Encode game state into a fixed-size vector."""

    def __init__(self, embed_dim=64, n_heads=2, text_proj_dim=32):
        super().__init__()
        self.embed_dim = embed_dim

        # Text projection
        self.card_text_proj = nn.Linear(TEXT_DIM, text_proj_dim)
        self.enemy_text_proj = nn.Linear(TEXT_DIM, text_proj_dim)
        self.relic_text_proj = nn.Linear(TEXT_DIM, text_proj_dim)
        self.potion_text_proj = nn.Linear(TEXT_DIM, text_proj_dim)
        self.context_text_proj = nn.Linear(TEXT_DIM, text_proj_dim)

        # Entity encoders
        self.card_enc = nn.Sequential(nn.Linear(CARD_FEAT_DIM + text_proj_dim, embed_dim), nn.ReLU())
        self.enemy_enc = nn.Sequential(nn.Linear(ENEMY_FEAT_DIM + text_proj_dim, embed_dim), nn.ReLU())

        # Self-attention
        self.card_attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.card_norm = nn.LayerNorm(embed_dim)
        self.enemy_attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.enemy_norm = nn.LayerNorm(embed_dim)

        # Cross-attention (cards → enemies), gated by combat phase
        self.cross_attn = nn.MultiheadAttention(embed_dim, n_heads, kdim=embed_dim, vdim=embed_dim, batch_first=True)
        self.cross_norm = nn.LayerNorm(embed_dim)

        # Pooling
        self.card_seed = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.enemy_seed = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.card_pool = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.enemy_pool = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)

        # Set encoders
        self.relic_enc = nn.Sequential(nn.Linear(text_proj_dim, 16), nn.ReLU())
        self.potion_enc = nn.Sequential(nn.Linear(text_proj_dim, 16), nn.ReLU())

        # Scalar
        self.scalar_net = nn.Sequential(
            nn.Linear(SCALAR_DIM + POWER_DIM + text_proj_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
        )

        # Output dim: card(E) + enemy(E) + scalar(64) + relic(16) + potion(16)
        self.output_dim = embed_dim + embed_dim + 64 + 16 + 16

    def forward(self, obs):
        # Text projections
        ct = self.card_text_proj(obs["hand_text"])
        et = self.enemy_text_proj(obs["enemy_text"])
        rt = self.relic_text_proj(obs["relics"])
        pt = self.potion_text_proj(obs["potions"])
        ctx = self.context_text_proj(obs["context_text"])

        # Cards
        card_emb = self.card_enc(torch.cat([obs["hand"], ct], dim=-1))
        hm = obs["hand_mask"].bool()
        card_emb = self.card_norm(card_emb + _safe_self_attn(self.card_attn, card_emb, hm))

        # Enemies
        enemy_emb = self.enemy_enc(torch.cat([obs["enemies"], et], dim=-1))
        em = obs["enemy_mask"].bool()
        enemy_emb = self.enemy_norm(enemy_emb + _safe_self_attn(self.enemy_attn, enemy_emb, em))

        # Cross-attention: only meaningful in combat (hand_mask has entries)
        cross = _safe_cross_attn(self.cross_attn, card_emb, enemy_emb, hm, em)
        card_emb = self.cross_norm(card_emb + cross)

        # Pool
        card_out = _safe_pool(self.card_pool, self.card_seed, card_emb, hm)
        enemy_out = _safe_pool(self.enemy_pool, self.enemy_seed, enemy_emb, em)

        # Relics / Potions
        relic_out = _masked_mean(self.relic_enc(rt), obs["relic_mask"].bool())
        potion_out = _masked_mean(self.potion_enc(pt), obs["potion_mask"].bool())

        # Scalars + context
        scalar_in = torch.cat([obs["scalars"], obs["player_powers"], ctx], dim=-1)
        scalar_out = self.scalar_net(scalar_in)

        return torch.cat([card_out, enemy_out, scalar_out, relic_out, potion_out], dim=-1)


class ActionEncoder(nn.Module):
    """Encode each candidate action into an embedding."""

    def __init__(self, embed_dim=64, text_proj_dim=32):
        super().__init__()
        self.action_text_proj = nn.Linear(TEXT_DIM, text_proj_dim)
        self.action_net = nn.Sequential(
            nn.Linear(ACTION_FEAT_DIM + text_proj_dim, embed_dim), nn.ReLU(),
        )
        self.output_dim = embed_dim

    def forward(self, obs):
        at = self.action_text_proj(obs["action_text"])
        combined = torch.cat([obs["actions"], at], dim=-1)
        return self.action_net(combined)  # (B, MAX_ACTIONS, embed_dim)


# ---------------------------------------------------------------------------
# Safe attention helpers
# ---------------------------------------------------------------------------

def _safe_self_attn(attn, x, mask):
    if not mask.any():
        return torch.zeros_like(x)
    empty = ~mask.any(dim=1)
    safe = mask.clone()
    safe[empty, 0] = True
    out, _ = attn(x, x, x, key_padding_mask=~safe)
    return out * mask.unsqueeze(-1).float()

def _safe_cross_attn(attn, q, kv, q_mask, kv_mask):
    if not q_mask.any() or not kv_mask.any():
        return torch.zeros_like(q)
    empty = ~kv_mask.any(dim=1)
    safe = kv_mask.clone()
    safe[empty, 0] = True
    out, _ = attn(q, kv, kv, key_padding_mask=~safe)
    out = out * q_mask.unsqueeze(-1).float()
    out[empty] = 0.0
    return out

def _safe_pool(attn, seed, x, mask):
    b, _, d = x.shape
    if not mask.any():
        return torch.zeros(b, d, device=x.device)
    empty = ~mask.any(dim=1)
    safe = mask.clone()
    safe[empty, 0] = True
    out, _ = attn(seed.expand(b, -1, -1), x, x, key_padding_mask=~safe)
    result = out.squeeze(1)
    result[empty] = 0.0
    return result

def _masked_mean(x, mask):
    b, _, d = x.shape
    if not mask.any():
        return torch.zeros(b, d, device=x.device)
    m = mask.unsqueeze(-1).float()
    return (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
