"""Phase 2 attention-based feature extractor for STS2 RL training.

Custom SB3 feature extractor that uses:
- MLP for scalar features (phase, player stats, combat meta)
- Self-attention + learned pooling for variable-length card sets (hand)
- Self-attention + learned pooling for variable-length enemy sets
- Fusion MLP that combines all three streams

Handles edge cases where hand_mask or enemy_mask is all-False
(no cards / no enemies) by producing zero vectors for those streams.
"""

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class STS2AttentionExtractor(BaseFeaturesExtractor):
    """Attention-based feature extractor for Dict observation spaces.

    Expected observation keys:
        scalars       (batch, scalar_dim)
        player_powers (batch, power_dim)
        hand          (batch, max_hand, card_feat_dim)
        hand_mask     (batch, max_hand)           -- 1 where card present
        enemies       (batch, max_enemies, enemy_feat_dim)
        enemy_mask    (batch, max_enemies)         -- 1 where enemy present
    """

    def __init__(
        self,
        observation_space,
        features_dim: int = 128,
        card_embed_dim: int = 32,
        enemy_embed_dim: int = 32,
        n_heads: int = 2,
        scalar_hidden: int = 64,
    ):
        super().__init__(observation_space, features_dim=features_dim)

        # ---- dimensions from observation space ----------------------------
        scalar_dim = observation_space["scalars"].shape[0]
        power_dim = observation_space["player_powers"].shape[0]
        card_feat_dim = observation_space["hand"].shape[1]
        enemy_feat_dim = observation_space["enemies"].shape[1]

        # ---- scalar encoder -----------------------------------------------
        self.scalar_net = nn.Sequential(
            nn.Linear(scalar_dim + power_dim, scalar_hidden),
            nn.ReLU(),
            nn.Linear(scalar_hidden, scalar_hidden),
            nn.ReLU(),
        )

        # ---- card encoder with self-attention + learned pooling -----------
        self.card_proj = nn.Sequential(
            nn.Linear(card_feat_dim, card_embed_dim),
            nn.ReLU(),
        )
        self.card_attn = nn.MultiheadAttention(
            card_embed_dim, n_heads, batch_first=True,
        )
        self.card_pool = nn.Parameter(
            torch.randn(1, 1, card_embed_dim) * 0.02,
        )
        self.card_pool_attn = nn.MultiheadAttention(
            card_embed_dim, n_heads, batch_first=True,
        )
        self.card_norm = nn.LayerNorm(card_embed_dim)

        # ---- enemy encoder with self-attention + learned pooling ----------
        self.enemy_proj = nn.Sequential(
            nn.Linear(enemy_feat_dim, enemy_embed_dim),
            nn.ReLU(),
        )
        self.enemy_attn = nn.MultiheadAttention(
            enemy_embed_dim, n_heads, batch_first=True,
        )
        self.enemy_pool = nn.Parameter(
            torch.randn(1, 1, enemy_embed_dim) * 0.02,
        )
        self.enemy_pool_attn = nn.MultiheadAttention(
            enemy_embed_dim, n_heads, batch_first=True,
        )
        self.enemy_norm = nn.LayerNorm(enemy_embed_dim)

        # ---- fusion -------------------------------------------------------
        combined_dim = scalar_hidden + card_embed_dim + enemy_embed_dim
        self.fusion = nn.Sequential(
            nn.Linear(combined_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        scalars = observations["scalars"]
        powers = observations["player_powers"]
        hand = observations["hand"]
        hand_mask = observations["hand_mask"]
        enemies = observations["enemies"]
        enemy_mask = observations["enemy_mask"]

        batch_size = scalars.shape[0]

        # ---- scalar stream ------------------------------------------------
        s = self.scalar_net(torch.cat([scalars, powers], dim=-1))

        # ---- card stream --------------------------------------------------
        c_out = self._encode_set(
            hand, hand_mask,
            self.card_proj, self.card_attn, self.card_norm,
            self.card_pool, self.card_pool_attn,
            batch_size,
        )

        # ---- enemy stream -------------------------------------------------
        e_out = self._encode_set(
            enemies, enemy_mask,
            self.enemy_proj, self.enemy_attn, self.enemy_norm,
            self.enemy_pool, self.enemy_pool_attn,
            batch_size,
        )

        # ---- fuse ---------------------------------------------------------
        return self.fusion(torch.cat([s, c_out, e_out], dim=-1))

    @staticmethod
    def _encode_set(
        items: torch.Tensor,           # (B, N, feat_dim)
        mask: torch.Tensor,            # (B, N)  -- 1.0 where present
        proj: nn.Module,
        self_attn: nn.MultiheadAttention,
        norm: nn.LayerNorm,
        pool_token: nn.Parameter,      # (1, 1, embed_dim)
        pool_attn: nn.MultiheadAttention,
        batch_size: int,
    ) -> torch.Tensor:
        """Project, self-attend, then pool a variable-length set.

        Returns a (batch, embed_dim) tensor.  If the mask is all-False
        for a sample, the output for that sample is zeros.
        """
        embed_dim = pool_token.shape[-1]

        # Check for fully-empty masks (no items at all)
        # key_padding_mask uses True to MASK (ignore) positions
        key_pad = ~(mask.bool())  # (B, N) -- True where absent

        # If every sample has at least one item we can run attention normally.
        # If some samples are fully empty we need to handle them separately
        # to avoid NaN from softmax over all-masked inputs.
        any_present = mask.sum(dim=-1) > 0  # (B,) bool

        if not any_present.any():
            # All samples empty -- return zeros
            return torch.zeros(batch_size, embed_dim, device=items.device)

        # Project all items
        x = proj(items)  # (B, N, embed_dim)

        # Self-attention with padding mask
        # For fully-empty rows, temporarily set one position as unmasked
        # to prevent NaN, then zero out the result afterward.
        safe_pad = key_pad.clone()
        fully_empty = ~any_present  # (B,)
        if fully_empty.any():
            # Unmask position 0 for empty rows to prevent all-masked softmax
            safe_pad[fully_empty, 0] = False

        x_attn, _ = self_attn(x, x, x, key_padding_mask=safe_pad)
        x = norm(x + x_attn)

        # Learned pooling: cross-attend from a single query token
        seed = pool_token.expand(batch_size, -1, -1)  # (B, 1, embed_dim)
        pooled, _ = pool_attn(seed, x, x, key_padding_mask=safe_pad)
        out = pooled.squeeze(1)  # (B, embed_dim)

        # Zero out results for fully-empty rows
        if fully_empty.any():
            out = out * any_present.unsqueeze(-1).float()

        return out
