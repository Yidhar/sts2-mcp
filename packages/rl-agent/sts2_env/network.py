"""Network building blocks for the routed STS2 RL policy."""

from __future__ import annotations

import torch
import torch.nn as nn

from .observation_v2 import (
    ACTION_FEAT_DIM,
    CARD_FEAT_DIM,
    DECK_FEAT_DIM,
    ENEMY_FEAT_DIM,
    NUM_DOMAINS,
    POWER_DIM,
    ROUTE_NODE_FEAT_DIM,
    ROUTE_SUMMARY_DIM,
    SCALAR_DIM,
)
from .text_encoder import TEXT_DIM


class SharedContextEncoder(nn.Module):
    """Encode global run context shared across all decision domains."""

    def __init__(
        self,
        *,
        context_text_dim: int = 48,
        set_text_dim: int = 32,
        hidden_dim: int = 96,
        output_dim: int = 128,
    ):
        super().__init__()
        self.context_text_dim = context_text_dim
        self.set_text_dim = set_text_dim
        self.output_dim = output_dim

        self.context_text_proj = nn.Linear(TEXT_DIM, context_text_dim)
        self.relic_text_proj = nn.Linear(TEXT_DIM, set_text_dim)
        self.potion_text_proj = nn.Linear(TEXT_DIM, set_text_dim)

        self.relic_enc = nn.Sequential(nn.Linear(set_text_dim, 16), nn.ReLU())
        self.potion_enc = nn.Sequential(nn.Linear(set_text_dim, 16), nn.ReLU())

        self.scalar_net = nn.Sequential(
            nn.Linear(SCALAR_DIM + POWER_DIM + context_text_dim + 16 + 16 + NUM_DOMAINS, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        context = self.context_text_proj(obs["context_text"])
        relics = self.relic_text_proj(obs["relics"])
        potions = self.potion_text_proj(obs["potions"])
        relic_out = _masked_mean(self.relic_enc(relics), obs["relic_mask"].bool())
        potion_out = _masked_mean(self.potion_enc(potions), obs["potion_mask"].bool())
        scalar_input = torch.cat(
            [obs["scalars"], obs["player_powers"], context, relic_out, potion_out, obs["decision_domain"]],
            dim=-1,
        )
        return self.scalar_net(scalar_input)


class CombatStateEncoder(nn.Module):
    """Encode hand + enemies for combat decisions."""

    def __init__(self, *, embed_dim: int = 64, n_heads: int = 2, text_proj_dim: int = 32):
        super().__init__()
        self.output_dim = embed_dim * 2

        self.card_text_proj = nn.Linear(TEXT_DIM, text_proj_dim)
        self.enemy_text_proj = nn.Linear(TEXT_DIM, text_proj_dim)

        self.card_enc = nn.Sequential(nn.Linear(CARD_FEAT_DIM + text_proj_dim, embed_dim), nn.ReLU())
        self.enemy_enc = nn.Sequential(nn.Linear(ENEMY_FEAT_DIM + text_proj_dim, embed_dim), nn.ReLU())

        self.card_attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.card_norm = nn.LayerNorm(embed_dim)
        self.enemy_attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.enemy_norm = nn.LayerNorm(embed_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim,
            n_heads,
            kdim=embed_dim,
            vdim=embed_dim,
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(embed_dim)

        self.card_seed = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.enemy_seed = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.card_pool = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.enemy_pool = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        card_text = self.card_text_proj(obs["hand_text"])
        enemy_text = self.enemy_text_proj(obs["enemy_text"])

        card_emb = self.card_enc(torch.cat([obs["hand"], card_text], dim=-1))
        enemy_emb = self.enemy_enc(torch.cat([obs["enemies"], enemy_text], dim=-1))

        hand_mask = obs["hand_mask"].bool()
        enemy_mask = obs["enemy_mask"].bool()

        card_emb = self.card_norm(card_emb + _safe_self_attn(self.card_attn, card_emb, hand_mask))
        enemy_emb = self.enemy_norm(enemy_emb + _safe_self_attn(self.enemy_attn, enemy_emb, enemy_mask))
        card_emb = self.cross_norm(card_emb + _safe_cross_attn(self.cross_attn, card_emb, enemy_emb, hand_mask, enemy_mask))

        hand_out = _safe_pool(self.card_pool, self.card_seed, card_emb, hand_mask)
        enemy_out = _safe_pool(self.enemy_pool, self.enemy_seed, enemy_emb, enemy_mask)
        return torch.cat([hand_out, enemy_out], dim=-1)


class BuildStateEncoder(nn.Module):
    """Encode the current deck for card reward / shop / upgrade decisions."""

    def __init__(self, *, embed_dim: int = 64, n_heads: int = 2, text_proj_dim: int = 32):
        super().__init__()
        self.output_dim = embed_dim

        self.deck_text_proj = nn.Linear(TEXT_DIM, text_proj_dim)
        self.deck_enc = nn.Sequential(nn.Linear(DECK_FEAT_DIM + text_proj_dim, embed_dim), nn.ReLU())
        self.deck_attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.deck_norm = nn.LayerNorm(embed_dim)
        self.deck_seed = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.deck_pool = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        deck_text = self.deck_text_proj(obs["deck_text"])
        deck_emb = self.deck_enc(torch.cat([obs["deck"], deck_text], dim=-1))
        deck_mask = obs["deck_mask"].bool()
        deck_emb = self.deck_norm(deck_emb + _safe_self_attn(self.deck_attn, deck_emb, deck_mask))
        return _safe_pool(self.deck_pool, self.deck_seed, deck_emb, deck_mask)


class RouteStateEncoder(nn.Module):
    """Compact route-domain state summary.

    First version intentionally stays small; route quality depends more on
    structured future summaries than on raw network capacity.
    """

    def __init__(self, *, input_dim: int = 128, output_dim: int = 48):
        super().__init__()
        self.output_dim = output_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, shared_ctx: torch.Tensor) -> torch.Tensor:
        return self.net(shared_ctx)


class DomainActionEncoder(nn.Module):
    """Encode candidate actions for one decision domain."""

    def __init__(self, *, action_dim: int, text_proj_dim: int = 32):
        super().__init__()
        self.output_dim = action_dim
        self.action_text_proj = nn.Linear(TEXT_DIM, text_proj_dim)
        self.net = nn.Sequential(
            nn.Linear(ACTION_FEAT_DIM + text_proj_dim, action_dim),
            nn.ReLU(),
        )

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        action_text = self.action_text_proj(obs["action_text"])
        combined = torch.cat([obs["actions"], action_text], dim=-1)
        return self.net(combined)


class RouteActionEncoder(nn.Module):
    """Encode route candidates from generic action features + future subtree tokens."""

    def __init__(self, *, action_dim: int, text_proj_dim: int = 32, node_dim: int = 32, n_heads: int = 2):
        super().__init__()
        self.output_dim = action_dim

        self.action_text_proj = nn.Linear(TEXT_DIM, text_proj_dim)
        self.generic_net = nn.Sequential(
            nn.Linear(ACTION_FEAT_DIM + text_proj_dim, action_dim),
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
        )

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        batch_size, action_count, node_count, _ = obs["route_nodes"].shape

        action_text = self.action_text_proj(obs["action_text"])
        generic = self.generic_net(torch.cat([obs["actions"], action_text], dim=-1))
        summary = self.summary_net(obs["route_summary"])

        flat_nodes = obs["route_nodes"].reshape(batch_size * action_count, node_count, -1)
        flat_masks = obs["route_node_mask"].reshape(batch_size * action_count, node_count).bool()
        node_emb = self.node_net(flat_nodes)
        node_emb = self.node_norm(node_emb + _safe_self_attn(self.node_attn, node_emb, flat_masks))
        pooled_nodes = _safe_pool(self.node_pool, self.node_seed, node_emb, flat_masks)
        pooled_nodes = pooled_nodes.reshape(batch_size, action_count, -1)

        combined = torch.cat([generic, summary, pooled_nodes], dim=-1)
        return self.final_net(combined)


class CandidateScorer(nn.Module):
    """Score [state, action, interaction] for all candidate actions."""

    def __init__(self, *, state_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.state_to_action = nn.Linear(state_dim, action_dim)
        self.scorer = nn.Sequential(
            nn.Linear(state_dim + action_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        batch_size, action_count, _ = actions.shape
        expanded_state = state.unsqueeze(1).expand(-1, action_count, -1)
        projected_state = self.state_to_action(expanded_state)
        scorer_input = torch.cat([expanded_state, actions, projected_state * actions], dim=-1)
        return self.scorer(scorer_input).squeeze(-1)


def _safe_self_attn(attn: nn.MultiheadAttention, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
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
    batch_size, _, dim = x.shape
    if not mask.any():
        return torch.zeros(batch_size, dim, device=x.device)
    weights = mask.unsqueeze(-1).float()
    return (x * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)
