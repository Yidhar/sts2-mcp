"""Single-file STS2 online RL model definition.

This file is the source of truth for the live acting model:
  - shared/domain encoders
  - candidate scorers
  - routed maskable PPO policy

Offline trainers may keep their own task-specific code for now, but the
online game-facing learner should converge here instead of spreading across
multiple files.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from gymnasium import spaces

from sb3_contrib.common.maskable.distributions import MaskableCategoricalDistribution
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.type_aliases import Schedule

from .observation_v2 import (
    ACTION_FEAT_DIM,
    ACTION_KIND_TO_ORD,
    CARD_FEAT_DIM,
    DECK_FEAT_DIM,
    ENEMY_FEAT_DIM,
    NUM_ACTION_KINDS,
    NUM_DOMAINS,
    POWER_DIM,
    ROUTE_NODE_FEAT_DIM,
    ROUTE_SUMMARY_DIM,
    SCALAR_DIM,
)
from .text_encoder import TEXT_DIM

_PLAY_CARD_KIND_ORD = ACTION_KIND_TO_ORD["play_card"]
_USE_POTION_KIND_ORD = ACTION_KIND_TO_ORD["use_potion"]
_ACTION_INDEX_SCALE = 20.0


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
    """Compact route-domain state summary."""

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


class CombatTurnPolicyHead(nn.Module):
    """Hierarchical combat gate: continue-vs-end_turn, then conditional action choice."""

    def __init__(self, *, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(state_dim + action_dim + 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(
        self,
        state: torch.Tensor,
        actions: torch.Tensor,
        base_logits: torch.Tensor,
        valid_mask: torch.Tensor,
        end_turn_mask: torch.Tensor,
    ) -> torch.Tensor:
        continue_mask = valid_mask & ~end_turn_mask
        continue_summary = _masked_mean(actions, continue_mask)
        flags = torch.stack(
            [
                continue_mask.any(dim=1).float(),
                end_turn_mask.any(dim=1).float(),
            ],
            dim=-1,
        )
        gate_logits = self.gate(torch.cat([state, continue_summary, flags], dim=-1))
        continue_count = continue_mask.sum(dim=1, keepdim=True).clamp(min=1).float()
        end_count = end_turn_mask.sum(dim=1, keepdim=True).clamp(min=1).float()
        gate_prior = torch.cat([continue_count.log(), end_count.log()], dim=-1)
        gate_logits = gate_logits + gate_prior
        continue_gate = gate_logits[:, 0:1]
        end_gate = gate_logits[:, 1:2]

        final_logits = torch.full_like(base_logits, -50.0)

        if continue_mask.any():
            normalized_continue = _masked_log_softmax(base_logits, continue_mask)
            final_logits = torch.where(continue_mask, continue_gate + normalized_continue, final_logits)

        if end_turn_mask.any():
            end_count = end_turn_mask.sum(dim=1, keepdim=True).clamp(min=1).float()
            end_logits = end_gate - end_count.log()
            final_logits = torch.where(end_turn_mask, end_logits, final_logits)

        rows_without_end = ~end_turn_mask.any(dim=1)
        if rows_without_end.any():
            final_logits[rows_without_end] = torch.where(
                continue_mask[rows_without_end],
                continue_gate[rows_without_end] + _masked_log_softmax(base_logits[rows_without_end], continue_mask[rows_without_end]),
                final_logits[rows_without_end],
            )

        rows_without_continue = ~continue_mask.any(dim=1)
        if rows_without_continue.any():
            final_logits[rows_without_continue] = torch.where(
                end_turn_mask[rows_without_continue],
                end_gate[rows_without_continue],
                final_logits[rows_without_continue],
            )

        return final_logits


class STS2CandidateScoringPolicy(MaskableActorCriticPolicy):
    """Policy with hard-routed combat/build/route experts."""

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Discrete,
        lr_schedule: Schedule,
        *,
        embed_dim: int | None = None,
        scorer_hidden: int | None = None,
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
        combat_continue_hidden: int = 64,
        critic_domain_dim: int = 64,
        **kwargs,
    ):
        kwargs.pop("features_extractor_class", None)
        kwargs.pop("features_extractor_kwargs", None)
        kwargs.pop("net_arch", None)

        if embed_dim is not None:
            combat_embed_dim = embed_dim
            build_embed_dim = embed_dim
            route_embed_dim = embed_dim
        if scorer_hidden is not None:
            combat_scorer_hidden = scorer_hidden
            build_scorer_hidden = scorer_hidden
            route_scorer_hidden = scorer_hidden

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
        self._combat_continue_hidden = combat_continue_hidden
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
        self.combat_turn_head = CombatTurnPolicyHead(
            state_dim=self._shared_output_dim + self.combat_state_encoder.output_dim,
            action_dim=self.combat_action_encoder.output_dim,
            hidden_dim=self._combat_continue_hidden,
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
        masked_logits, masks = self._mask_logits(logits, action_masks=action_masks)
        distribution = self.action_dist.proba_distribution(action_logits=masked_logits)
        values = self._sanitize_tensor(self.value_net(torch.cat([shared_ctx, critic_state], dim=-1)))
        if deterministic:
            actions = self._select_deterministic_actions(masked_logits, obs, masks)
        else:
            actions = distribution.get_actions(deterministic=False)
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
        masked_logits, _ = self._mask_logits(logits, action_masks=action_masks)
        return self.action_dist.proba_distribution(action_logits=masked_logits)

    def score_action_logits(self, obs, action_masks=None):
        shared_ctx = self._sanitize_tensor(self.shared_encoder(obs))
        logits, _ = self._route_domains(obs, shared_ctx)
        masked_logits, _ = self._mask_logits(logits, action_masks=action_masks)
        return masked_logits

    def predict_values(self, obs):
        shared_ctx = self._sanitize_tensor(self.shared_encoder(obs))
        _, critic_state = self._route_domains(obs, shared_ctx)
        return self._sanitize_tensor(self.value_net(torch.cat([shared_ctx, critic_state], dim=-1)))

    def _predict(self, observation, deterministic=False, action_masks=None):
        shared_ctx = self._sanitize_tensor(self.shared_encoder(observation))
        logits, _ = self._route_domains(observation, shared_ctx)
        masked_logits, masks = self._mask_logits(logits, action_masks=action_masks)
        if deterministic:
            return self._select_deterministic_actions(masked_logits, observation, masks)
        return self.action_dist.proba_distribution(action_logits=masked_logits).get_actions(deterministic=False)

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
            combat_base_logits = self._sanitize_logits(self.combat_scorer(combat_input, combat_actions))
            combat_valid_mask = torch.as_tensor(
                sub_obs.get("action_mask", torch.ones_like(combat_base_logits)),
                dtype=torch.bool,
                device=shared_ctx.device,
            ).reshape(combat_base_logits.shape)
            combat_end_turn_mask = combat_valid_mask & (sub_obs["actions"][..., 9] > 0.5)
            combat_logits = self.combat_turn_head(
                combat_input,
                combat_actions,
                combat_base_logits,
                combat_valid_mask,
                combat_end_turn_mask,
            )
            logits[combat_mask] = self._sanitize_logits(combat_logits)
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

    def _mask_logits(self, logits, action_masks=None, actions=None):
        masked_logits = self._sanitize_logits(logits)
        masks = None
        if action_masks is not None:
            masks = self._normalize_action_masks(action_masks, masked_logits, actions=actions)
            masked_logits = self._apply_action_mask(masked_logits, masks)
        return masked_logits, masks

    def _select_deterministic_actions(self, masked_logits, obs, masks):
        if masks is None or "actions" not in obs:
            return masked_logits.argmax(dim=1)

        action_rows = torch.as_tensor(obs["actions"], dtype=torch.float32, device=masked_logits.device)
        if action_rows.ndim == 2:
            action_rows = action_rows.unsqueeze(0)

        batch_actions = []
        kind_scale = float(NUM_ACTION_KINDS)
        for batch_index in range(masked_logits.shape[0]):
            row_logits = masked_logits[batch_index]
            row_mask = masks[batch_index]
            valid_indices = row_mask.nonzero(as_tuple=False).reshape(-1)
            if valid_indices.numel() == 0:
                batch_actions.append(torch.zeros((), dtype=torch.long, device=masked_logits.device))
                continue

            row_actions = action_rows[batch_index]
            group_members: dict[tuple[int, int], list[int]] = {}
            for idx_tensor in valid_indices:
                candidate_index = int(idx_tensor.item())
                row = row_actions[candidate_index]
                kind_ord = int(torch.round(row[0] * kind_scale).item())
                if kind_ord in (_PLAY_CARD_KIND_ORD, _USE_POTION_KIND_ORD):
                    logical_index = int(torch.round(row[19] * _ACTION_INDEX_SCALE).item())
                    group_key = (kind_ord, logical_index)
                else:
                    group_key = (-1, candidate_index)
                group_members.setdefault(group_key, []).append(candidate_index)

            best_action = int(valid_indices[0].item())
            best_group_score = None
            for members in group_members.values():
                member_tensor = torch.as_tensor(members, dtype=torch.long, device=masked_logits.device)
                group_score = torch.logsumexp(row_logits[member_tensor], dim=0)
                if best_group_score is None or group_score > best_group_score:
                    best_group_score = group_score
                    best_action = int(member_tensor[row_logits[member_tensor].argmax()].item())

            batch_actions.append(torch.tensor(best_action, dtype=torch.long, device=masked_logits.device))

        return torch.stack(batch_actions, dim=0)

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
            combat_continue_hidden=self._combat_continue_hidden,
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


def _masked_log_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if not mask.any():
        return torch.full_like(logits, -50.0)

    masked_logits = logits.masked_fill(~mask, -1e9)
    log_norm = torch.logsumexp(masked_logits, dim=1, keepdim=True)
    normalized = masked_logits - log_norm
    normalized = torch.where(mask, normalized, torch.full_like(normalized, -50.0))
    no_valid = ~mask.any(dim=1, keepdim=True)
    normalized = torch.where(no_valid, torch.full_like(normalized, -50.0), normalized)
    return torch.nan_to_num(normalized, nan=-50.0, posinf=0.0, neginf=-50.0)
