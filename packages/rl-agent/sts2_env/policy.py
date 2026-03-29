"""Candidate-scoring policy for STS2 RL.

Actor: scores each legal action via MLP([state, action, state*action]) -> scalar
Critic: scores state only via MLP(state) -> scalar

Replaces the default MultiInputPolicy approach where the policy head
outputs logits over blind slot indices.
"""

import torch
import torch.nn as nn
import numpy as np
from gymnasium import spaces

from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.maskable.distributions import MaskableCategoricalDistribution
from stable_baselines3.common.type_aliases import Schedule

from .network import StateEncoder, ActionEncoder
from .observation_v2 import MAX_ACTIONS


class STS2CandidateScoringPolicy(MaskableActorCriticPolicy):
    """Policy that scores each candidate action against the state.

    Actor logits[i] = scorer(state, action_i) for each legal action.
    Value = value_head(state).
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Discrete,
        lr_schedule: Schedule,
        # Network params
        embed_dim: int = 64,
        n_heads: int = 2,
        text_proj_dim: int = 32,
        scorer_hidden: int = 64,
        **kwargs,
    ):
        # Don't let parent create default feature extractor / mlp
        kwargs.pop("features_extractor_class", None)
        kwargs.pop("features_extractor_kwargs", None)
        kwargs.pop("net_arch", None)

        self._embed_dim = embed_dim
        self._n_heads = n_heads
        self._text_proj_dim = text_proj_dim
        self._scorer_hidden = scorer_hidden

        # Pre-compute features_dim before parent __init__ tries to set it
        self._features_dim_value = embed_dim + embed_dim + 64 + 16 + 16

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=[],  # we handle actor/critic ourselves
            **kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        """Override: build dual-tower encoders + scorer instead of default MLP."""
        self.state_encoder = StateEncoder(
            embed_dim=self._embed_dim,
            n_heads=self._n_heads,
            text_proj_dim=self._text_proj_dim,
        )
        self.action_encoder = ActionEncoder(
            embed_dim=self._embed_dim,
            text_proj_dim=self._text_proj_dim,
        )

        state_dim = self.state_encoder.output_dim
        action_dim = self.action_encoder.output_dim

        # Learned projection for state→action interaction
        self.state_to_action_proj = nn.Linear(state_dim, action_dim)

        # Actor: score(state, action, state_proj*action) -> scalar
        scorer_input = state_dim + action_dim + action_dim  # [state, action, projected_state*action]
        self.action_scorer = nn.Sequential(
            nn.Linear(scorer_input, self._scorer_hidden),
            nn.ReLU(),
            nn.Linear(self._scorer_hidden, 1),
        )

        # Critic: value(state) -> scalar
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, self._scorer_hidden),
            nn.ReLU(),
            nn.Linear(self._scorer_hidden, 1),
        )

        # Dummy attributes expected by parent class
        self.mlp_extractor = _DummyExtractor()

    def _build(self, lr_schedule) -> None:
        """Override: skip default action/value net creation."""
        self._build_mlp_extractor()

        # Action distribution
        self.action_dist = MaskableCategoricalDistribution(self.action_space.n)

        # Optimizer
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )

    def forward(self, obs, deterministic=False, action_masks=None):
        """Forward pass: compute action, value, log_prob."""
        state_emb = self._sanitize_tensor(self.state_encoder(obs))
        action_embs = self._sanitize_tensor(self.action_encoder(obs))
        logits = self._score_actions(state_emb, action_embs)
        distribution = self._build_distribution(logits, action_masks=action_masks)
        values = self._sanitize_tensor(self.value_net(state_emb))  # (B, 1) — consistent with parent convention

        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob

    def evaluate_actions(self, obs, actions, action_masks=None):
        """Evaluate actions for PPO loss computation."""
        state_emb = self._sanitize_tensor(self.state_encoder(obs))
        action_embs = self._sanitize_tensor(self.action_encoder(obs))
        logits = self._score_actions(state_emb, action_embs)
        distribution = self._build_distribution(logits, action_masks=action_masks, actions=actions)
        values = self._sanitize_tensor(self.value_net(state_emb))  # (B, 1)

        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        return values, log_prob, entropy

    def get_distribution(self, obs, action_masks=None):
        """Get action distribution for rollout collection."""
        state_emb = self._sanitize_tensor(self.state_encoder(obs))
        action_embs = self._sanitize_tensor(self.action_encoder(obs))
        logits = self._score_actions(state_emb, action_embs)
        return self._build_distribution(logits, action_masks=action_masks)

    def predict_values(self, obs):
        """Predict state values (critic only)."""
        state_emb = self._sanitize_tensor(self.state_encoder(obs))
        return self._sanitize_tensor(self.value_net(state_emb))

    def _score_actions(self, state_emb, action_embs):
        """Score each action against the state.

        Args:
            state_emb: (B, state_dim)
            action_embs: (B, MAX_ACTIONS, action_dim)

        Returns:
            logits: (B, MAX_ACTIONS)
        """
        B, N, D = action_embs.shape
        state_expanded = state_emb.unsqueeze(1).expand(-1, N, -1)   # (B, N, state_dim)

        # Learned projection for element-wise interaction
        state_proj = self.state_to_action_proj(state_expanded)       # (B, N, action_dim)

        scorer_input = torch.cat([
            state_expanded,
            action_embs,
            state_proj * action_embs,  # interaction term (full state projected)
        ], dim=-1)  # (B, N, state_dim + action_dim + action_dim)

        logits = self.action_scorer(scorer_input).squeeze(-1)  # (B, N)
        return self._sanitize_logits(logits)

    def _build_distribution(self, logits, action_masks=None, actions=None):
        """Create a numerically stable masked categorical distribution."""
        masked_logits = self._sanitize_logits(logits)
        if action_masks is not None:
            masks = self._normalize_action_masks(action_masks, masked_logits, actions=actions)
            masked_logits = self._apply_action_mask(masked_logits, masks)
        return self.action_dist.proba_distribution(action_logits=masked_logits)

    @staticmethod
    def _sanitize_tensor(tensor):
        return torch.nan_to_num(tensor, nan=0.0, posinf=1e4, neginf=-1e4)

    def _sanitize_logits(self, logits):
        logits = self._sanitize_tensor(logits)
        row_max = logits.max(dim=-1, keepdim=True).values
        row_max = torch.nan_to_num(row_max, nan=0.0, posinf=0.0, neginf=0.0)
        logits = logits - row_max
        return logits.clamp(min=-50.0, max=0.0)

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
        masked_logits = logits.masked_fill(~masks, -50.0)
        masked_logits = self._sanitize_logits(masked_logits)
        return masked_logits

    def _get_constructor_parameters(self):
        data = super()._get_constructor_parameters()
        data.update(
            embed_dim=self._embed_dim,
            n_heads=self._n_heads,
            text_proj_dim=self._text_proj_dim,
            scorer_hidden=self._scorer_hidden,
        )
        # Remove parent keys that we pop in __init__
        data.pop("features_extractor_class", None)
        data.pop("features_extractor_kwargs", None)
        return data

    def extract_features(self, obs, features_extractor=None):
        """Not used — dual tower replaces feature extraction."""
        return obs


class _DummyExtractor(nn.Module):
    """Placeholder to satisfy parent class expectations."""
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
