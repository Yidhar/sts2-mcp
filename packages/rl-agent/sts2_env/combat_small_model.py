"""Small policy-value network for combat-only exact-search training."""

from __future__ import annotations

import torch
import torch.nn as nn


class SmallCombatPolicyValueNet(nn.Module):
    """Compact policy/value network for fixed-template combat actions.

    This model is intentionally much smaller than the unified full-run MuZero
    network. It consumes a flattened combat state vector and predicts:

    - policy logits over the fixed combat action template
    - scalar value in [-1, 1]
    """

    def __init__(
        self,
        *,
        state_dim: int,
        num_actions: int,
        hidden_dim: int = 256,
        action_embed_dim: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.num_actions = int(num_actions)
        self.hidden_dim = int(hidden_dim)
        self.action_embed_dim = int(action_embed_dim)

        self.input_norm = nn.LayerNorm(self.state_dim)
        self.trunk = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.policy_query = nn.Linear(self.hidden_dim, self.action_embed_dim)
        self.action_embeddings = nn.Parameter(
            torch.randn(self.num_actions, self.action_embed_dim) * 0.02
        )
        self.policy_bias = nn.Parameter(torch.zeros(self.num_actions))
        self.value_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 1),
            nn.Tanh(),
        )

    def forward(self, state_vector: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if state_vector.dim() == 1:
            state_vector = state_vector.unsqueeze(0)
        x = self.input_norm(state_vector)
        hidden = self.trunk(x)
        query = self.policy_query(hidden)
        logits = torch.matmul(query, self.action_embeddings.transpose(0, 1)) + self.policy_bias
        value = self.value_head(hidden).squeeze(-1)
        return logits, value
