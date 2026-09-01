"""Recurrent hybrid actor-critic for the fresh FIFO V-trace lineage.

The factual encoder, dual macro/combat stems, shared fusion stream and split
run/combat memory are retained from the hybrid controller.  Control is once
again a dense stochastic policy with a candidate-independent scalar value.
Unlike the retired IQL actor, policy features are *not* detached: current
on-policy gradients and entropy train the shared representation together with
the value regression.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Final

import torch
from torch import Tensor, nn

from .grounded_candidate import GroundedCandidateBatch, GroundedCandidateConfig
from .hybrid_candidate_q import HybridCandidateQConfig, HybridDomainCandidateQModel

HYBRID_ACTOR_CRITIC_MODEL_VERSION: Final = "sts2-hybrid-domain-actor-critic-v1"


@dataclass(frozen=True, slots=True)
class HybridActorCriticConfig:
    grounded: GroundedCandidateConfig = field(default_factory=GroundedCandidateConfig)
    domain_layers: int = 1
    joint_layers: int = 2

    def __post_init__(self) -> None:
        HybridCandidateQConfig(
            grounded=self.grounded,
            domain_layers=self.domain_layers,
            joint_layers=self.joint_layers,
        )

    def as_mapping(self) -> dict[str, Any]:
        return {
            "version": HYBRID_ACTOR_CRITIC_MODEL_VERSION,
            "grounded": asdict(self.grounded),
            "domain_layers": self.domain_layers,
            "joint_layers": self.joint_layers,
        }

    def candidate_q_config(self) -> HybridCandidateQConfig:
        return HybridCandidateQConfig(
            grounded=self.grounded,
            domain_layers=self.domain_layers,
            joint_layers=self.joint_layers,
        )


@dataclass(frozen=True, slots=True)
class HybridActorCriticOutput:
    policy_logits: Tensor  # [B, A]
    value: Tensor  # [B]
    recurrent_state: Tensor  # [B, H]
    joint_state_embedding: Tensor  # [B, D]
    world_state_embedding: Tensor  # [B, D]
    candidate_embeddings: Tensor  # [B, A, D]
    action_mask: Tensor  # [B, A]

    def validate(self, config: HybridActorCriticConfig) -> tuple[int, int]:
        if self.policy_logits.ndim != 2:
            raise ValueError("hybrid actor logits must have rank 2")
        batch_size, action_count = self.policy_logits.shape
        cfg = config.grounded
        expected = {
            "value": (batch_size,),
            "recurrent_state": (batch_size, cfg.recurrent_hidden_dim),
            "joint_state_embedding": (batch_size, cfg.d_model),
            "world_state_embedding": (batch_size, cfg.d_model),
            "candidate_embeddings": (batch_size, action_count, cfg.d_model),
            "action_mask": (batch_size, action_count),
        }
        for name, shape in expected.items():
            if tuple(getattr(self, name).shape) != shape:
                raise ValueError(f"hybrid actor {name} has the wrong shape")
        if self.action_mask.dtype != torch.bool:
            raise TypeError("hybrid actor action mask must use torch.bool")
        for name in (
            "policy_logits",
            "value",
            "recurrent_state",
            "joint_state_embedding",
            "world_state_embedding",
            "candidate_embeddings",
        ):
            value = getattr(self, name)
            if not torch.is_floating_point(value) or not bool(torch.isfinite(value).all().item()):
                raise ValueError(f"hybrid actor {name} must be finite floating point")
        return batch_size, action_count


class HybridDomainActorCriticModel(HybridDomainCandidateQModel):
    """Hybrid dual-stream representation with one policy and one value head."""

    def __init__(self, config: HybridActorCriticConfig | None = None) -> None:
        actor_critic = config or HybridActorCriticConfig()
        super().__init__(actor_critic.candidate_q_config())
        self.actor_critic_config = actor_critic
        del self.candidate_q_head
        cfg = actor_critic.grounded
        control_input = cfg.d_model * 2 + cfg.recurrent_hidden_dim
        self.candidate_policy_head = nn.Sequential(
            nn.Linear(control_input, cfg.ffn_dim),
            nn.SiLU(),
            nn.Linear(cfg.ffn_dim, 1),
        )
        self.state_value_head = nn.Sequential(
            nn.Linear(cfg.d_model + cfg.recurrent_hidden_dim, cfg.ffn_dim),
            nn.SiLU(),
            nn.Linear(cfg.ffn_dim, 1),
        )

    def forward(  # type: ignore[override]
        self,
        batch: GroundedCandidateBatch,
        recurrent_state: Tensor | None = None,
        *,
        validate: bool = True,
    ) -> HybridActorCriticOutput:
        base = self.forward_representation(
            batch,
            recurrent_state,
            validate=validate,
        )
        action_count = int(base.action_mask.shape[1])
        state = base.joint_state_embedding.unsqueeze(1).expand(-1, action_count, -1)
        memory = base.recurrent_state.unsqueeze(1).expand(-1, action_count, -1)
        policy_logits = self.candidate_policy_head(
            torch.cat([base.candidate_embeddings, state, memory], dim=-1)
        ).squeeze(-1)
        policy_logits = policy_logits.masked_fill(~base.action_mask, 0.0)
        value = self.state_value_head(
            torch.cat([base.joint_state_embedding, base.recurrent_state], dim=-1)
        ).squeeze(-1)
        output = HybridActorCriticOutput(
            policy_logits=policy_logits,
            value=value,
            recurrent_state=base.recurrent_state,
            joint_state_embedding=base.joint_state_embedding,
            world_state_embedding=base.world_state_embedding,
            candidate_embeddings=base.candidate_embeddings,
            action_mask=base.action_mask,
        )
        if validate:
            output.validate(self.actor_critic_config)
        return output
def initialize_actor_critic_from_iql(
    model: HybridDomainActorCriticModel,
    source_state: dict[str, Tensor],
) -> dict[str, tuple[str, ...]]:
    """Model-init from the healthy hybrid-IQL policy/value/trunk only."""

    destination = model.state_dict()
    loaded: dict[str, Tensor] = {}
    for key, value in source_state.items():
        target = destination.get(key)
        if target is None or tuple(target.shape) != tuple(value.shape):
            continue
        if value.is_floating_point() and not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"hybrid actor source tensor is non-finite: {key}")
        loaded[key] = value
    result = model.load_state_dict(loaded, strict=False)
    required_prefixes = (
        "candidate_policy_head.",
        "state_value_head.",
        "macro_stream.",
        "combat_stream.",
        "joint_stream.",
    )
    absent = [
        prefix for prefix in required_prefixes if not any(key.startswith(prefix) for key in loaded)
    ]
    if absent:
        raise ValueError("hybrid actor source misses: " + ", ".join(absent))
    return {
        "inherited": tuple(sorted(loaded)),
        "fresh": tuple(sorted(result.missing_keys)),
    }


__all__ = [
    "HYBRID_ACTOR_CRITIC_MODEL_VERSION",
    "HybridActorCriticConfig",
    "HybridActorCriticOutput",
    "HybridDomainActorCriticModel",
    "initialize_actor_critic_from_iql",
]
