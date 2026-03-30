"""Hard-routed multi-head candidate-scoring policy for STS2 RL."""

from __future__ import annotations

import torch
import torch.nn as nn
from gymnasium import spaces

from sb3_contrib.common.maskable.distributions import MaskableCategoricalDistribution
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.type_aliases import Schedule

from .network import (
    BuildStateEncoder,
    CandidateScorer,
    CombatStateEncoder,
    DomainActionEncoder,
    RouteActionEncoder,
    RouteStateEncoder,
    SharedContextEncoder,
)


class STS2CandidateScoringPolicy(MaskableActorCriticPolicy):
    """Policy with hard-routed combat/build/route experts.

    Routing is not learned. The bridge provides decision_domain and the policy
    activates exactly one expert per sample.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Discrete,
        lr_schedule: Schedule,
        *,
        combat_embed_dim: int = 64,
        build_embed_dim: int = 64,
        route_embed_dim: int = 48,
        n_heads: int = 2,
        text_proj_dim: int = 32,
        context_text_dim: int = 48,
        shared_hidden_dim: int = 96,
        shared_output_dim: int = 128,
        combat_scorer_hidden: int = 128,
        build_scorer_hidden: int = 96,
        route_scorer_hidden: int = 64,
        critic_domain_dim: int = 64,
        **kwargs,
    ):
        kwargs.pop("features_extractor_class", None)
        kwargs.pop("features_extractor_kwargs", None)
        kwargs.pop("net_arch", None)

        self._combat_embed_dim = combat_embed_dim
        self._build_embed_dim = build_embed_dim
        self._route_embed_dim = route_embed_dim
        self._n_heads = n_heads
        self._text_proj_dim = text_proj_dim
        self._context_text_dim = context_text_dim
        self._shared_hidden_dim = shared_hidden_dim
        self._shared_output_dim = shared_output_dim
        self._combat_scorer_hidden = combat_scorer_hidden
        self._build_scorer_hidden = build_scorer_hidden
        self._route_scorer_hidden = route_scorer_hidden
        self._critic_domain_dim = critic_domain_dim
        self._features_dim_value = shared_output_dim + critic_domain_dim

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=[],
            **kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        self.shared_encoder = SharedContextEncoder(
            context_text_dim=self._context_text_dim,
            set_text_dim=self._text_proj_dim,
            hidden_dim=self._shared_hidden_dim,
            output_dim=self._shared_output_dim,
        )

        self.combat_state_encoder = CombatStateEncoder(
            embed_dim=self._combat_embed_dim,
            n_heads=self._n_heads,
            text_proj_dim=self._text_proj_dim,
        )
        self.build_state_encoder = BuildStateEncoder(
            embed_dim=self._build_embed_dim,
            n_heads=self._n_heads,
            text_proj_dim=self._text_proj_dim,
        )
        self.route_state_encoder = RouteStateEncoder(
            input_dim=self._shared_output_dim,
            output_dim=self._route_embed_dim,
        )

        self.combat_action_encoder = DomainActionEncoder(
            action_dim=self._combat_embed_dim,
            text_proj_dim=self._text_proj_dim,
        )
        self.build_action_encoder = DomainActionEncoder(
            action_dim=self._build_embed_dim,
            text_proj_dim=self._text_proj_dim,
        )
        self.route_action_encoder = RouteActionEncoder(
            action_dim=self._route_embed_dim,
            text_proj_dim=self._text_proj_dim,
            n_heads=self._n_heads,
        )

        self.combat_scorer = CandidateScorer(
            state_dim=self._shared_output_dim + self.combat_state_encoder.output_dim,
            action_dim=self.combat_action_encoder.output_dim,
            hidden_dim=self._combat_scorer_hidden,
        )
        self.build_scorer = CandidateScorer(
            state_dim=self._shared_output_dim + self.build_state_encoder.output_dim,
            action_dim=self.build_action_encoder.output_dim,
            hidden_dim=self._build_scorer_hidden,
        )
        self.route_scorer = CandidateScorer(
            state_dim=self._shared_output_dim + self.route_state_encoder.output_dim,
            action_dim=self.route_action_encoder.output_dim,
            hidden_dim=self._route_scorer_hidden,
        )

        self.combat_critic_proj = nn.Sequential(
            nn.Linear(self.combat_state_encoder.output_dim, self._critic_domain_dim),
            nn.ReLU(),
        )
        self.build_critic_proj = nn.Sequential(
            nn.Linear(self.build_state_encoder.output_dim, self._critic_domain_dim),
            nn.ReLU(),
        )
        self.route_critic_proj = nn.Sequential(
            nn.Linear(self.route_state_encoder.output_dim, self._critic_domain_dim),
            nn.ReLU(),
        )
        self.value_net = nn.Sequential(
            nn.Linear(self._shared_output_dim + self._critic_domain_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        self.mlp_extractor = _DummyExtractor()

    def _build(self, lr_schedule) -> None:
        self._build_mlp_extractor()
        self.action_dist = MaskableCategoricalDistribution(self.action_space.n)
        self.optimizer = self.optimizer_class(
            self.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )

    def forward(self, obs, deterministic=False, action_masks=None):
        shared_ctx = self._sanitize_tensor(self.shared_encoder(obs))
        logits, critic_state = self._route_domains(obs, shared_ctx)
        distribution = self._build_distribution(logits, action_masks=action_masks)
        values = self._sanitize_tensor(self.value_net(torch.cat([shared_ctx, critic_state], dim=-1)))
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob

    def evaluate_actions(self, obs, actions, action_masks=None):
        shared_ctx = self._sanitize_tensor(self.shared_encoder(obs))
        logits, critic_state = self._route_domains(obs, shared_ctx)
        distribution = self._build_distribution(logits, action_masks=action_masks, actions=actions)
        values = self._sanitize_tensor(self.value_net(torch.cat([shared_ctx, critic_state], dim=-1)))
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        return values, log_prob, entropy

    def get_distribution(self, obs, action_masks=None):
        shared_ctx = self._sanitize_tensor(self.shared_encoder(obs))
        logits, _ = self._route_domains(obs, shared_ctx)
        return self._build_distribution(logits, action_masks=action_masks)

    def predict_values(self, obs):
        shared_ctx = self._sanitize_tensor(self.shared_encoder(obs))
        _, critic_state = self._route_domains(obs, shared_ctx)
        return self._sanitize_tensor(self.value_net(torch.cat([shared_ctx, critic_state], dim=-1)))

    def _route_domains(self, obs, shared_ctx):
        batch_size = shared_ctx.shape[0]
        action_count = obs["actions"].shape[1]
        logits = torch.full((batch_size, action_count), -50.0, device=shared_ctx.device)
        critic_state = torch.zeros((batch_size, self._critic_domain_dim), device=shared_ctx.device)

        combat_mask, build_mask, route_mask = self._domain_masks(obs, shared_ctx.device)

        if combat_mask.any():
            sub_obs = self._slice_obs(obs, combat_mask)
            shared_sub = shared_ctx[combat_mask]
            combat_state = self._sanitize_tensor(self.combat_state_encoder(sub_obs))
            combat_actions = self._sanitize_tensor(self.combat_action_encoder(sub_obs))
            combat_input = torch.cat([shared_sub, combat_state], dim=-1)
            logits[combat_mask] = self._sanitize_logits(self.combat_scorer(combat_input, combat_actions))
            critic_state[combat_mask] = self._sanitize_tensor(self.combat_critic_proj(combat_state))

        if build_mask.any():
            sub_obs = self._slice_obs(obs, build_mask)
            shared_sub = shared_ctx[build_mask]
            build_state = self._sanitize_tensor(self.build_state_encoder(sub_obs))
            build_actions = self._sanitize_tensor(self.build_action_encoder(sub_obs))
            build_input = torch.cat([shared_sub, build_state], dim=-1)
            logits[build_mask] = self._sanitize_logits(self.build_scorer(build_input, build_actions))
            critic_state[build_mask] = self._sanitize_tensor(self.build_critic_proj(build_state))

        if route_mask.any():
            sub_obs = self._slice_obs(obs, route_mask)
            shared_sub = shared_ctx[route_mask]
            route_state = self._sanitize_tensor(self.route_state_encoder(shared_sub))
            route_actions = self._sanitize_tensor(self.route_action_encoder(sub_obs))
            route_input = torch.cat([shared_sub, route_state], dim=-1)
            logits[route_mask] = self._sanitize_logits(self.route_scorer(route_input, route_actions))
            critic_state[route_mask] = self._sanitize_tensor(self.route_critic_proj(route_state))

        return self._sanitize_logits(logits), critic_state

    def _domain_masks(self, obs, device):
        domain = obs.get("decision_domain")
        if domain is None:
            batch_size = obs["scalars"].shape[0]
            build_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
            return (
                torch.zeros(batch_size, dtype=torch.bool, device=device),
                build_mask,
                torch.zeros(batch_size, dtype=torch.bool, device=device),
            )

        domain = torch.as_tensor(domain, dtype=torch.float32, device=device)
        if domain.ndim == 1:
            domain = domain.unsqueeze(0)
        combat_mask = domain[:, 0] > 0.5
        build_mask = domain[:, 1] > 0.5
        route_mask = domain[:, 2] > 0.5
        unresolved = ~(combat_mask | build_mask | route_mask)
        if unresolved.any():
            build_mask = build_mask.clone()
            build_mask[unresolved] = True
        return combat_mask, build_mask, route_mask

    @staticmethod
    def _slice_obs(obs, mask):
        return {key: value[mask] for key, value in obs.items()}

    def _build_distribution(self, logits, action_masks=None, actions=None):
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
        return self._sanitize_logits(masked_logits)

    def _get_constructor_parameters(self):
        data = super()._get_constructor_parameters()
        data.update(
            combat_embed_dim=self._combat_embed_dim,
            build_embed_dim=self._build_embed_dim,
            route_embed_dim=self._route_embed_dim,
            n_heads=self._n_heads,
            text_proj_dim=self._text_proj_dim,
            context_text_dim=self._context_text_dim,
            shared_hidden_dim=self._shared_hidden_dim,
            shared_output_dim=self._shared_output_dim,
            combat_scorer_hidden=self._combat_scorer_hidden,
            build_scorer_hidden=self._build_scorer_hidden,
            route_scorer_hidden=self._route_scorer_hidden,
            critic_domain_dim=self._critic_domain_dim,
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
