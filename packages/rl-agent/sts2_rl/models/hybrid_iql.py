"""Recurrent hybrid IQL/AWAC controller.

The factual encoder, slow/fast memories and domain-fusion trunk are inherited
from :mod:`hybrid_candidate_q`.  Control no longer takes a max over weakly
supported candidate-Q values.  A candidate-independent value head supplies
the critic bootstrap, while a separate candidate policy head is trained by
advantage-weighted regression and owns collection/evaluation behavior.

Policy features are detached from the critic trunk.  The actor therefore
cannot move the shared factual representation merely because cross-entropy is
larger in scale than a TD error; Q and V remain the representation learners.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Final

import torch
from torch import Tensor, nn

from .grounded_candidate import GroundedCandidateBatch, GroundedCandidateConfig
from .hybrid_candidate_q import (
    HybridCandidateQConfig,
    HybridDomainCandidateQModel,
)

HYBRID_IQL_MODEL_VERSION: Final = "sts2-hybrid-domain-iql-v1"


@dataclass(frozen=True, slots=True)
class HybridIQLConfig:
    """Topology identity for the IQL/AWAC successor."""

    grounded: GroundedCandidateConfig = field(default_factory=GroundedCandidateConfig)
    domain_layers: int = 1
    joint_layers: int = 2

    def __post_init__(self) -> None:
        # Reuse the reviewed topology validation rather than duplicate it.
        HybridCandidateQConfig(
            grounded=self.grounded,
            domain_layers=self.domain_layers,
            joint_layers=self.joint_layers,
        )

    def as_mapping(self) -> dict[str, Any]:
        return {
            "version": HYBRID_IQL_MODEL_VERSION,
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
class HybridIQLOutput:
    q_values: Tensor  # [B, A]
    state_value: Tensor  # [B]
    policy_logits: Tensor  # [B, A]
    recurrent_state: Tensor  # [B, H]
    joint_state_embedding: Tensor  # [B, D]
    world_state_embedding: Tensor  # [B, D]
    candidate_embeddings: Tensor  # [B, A, D]
    action_mask: Tensor  # [B, A]

    def validate(self, config: HybridIQLConfig) -> tuple[int, int]:
        if self.q_values.ndim != 2:
            raise ValueError("hybrid IQL q_values must have rank 2")
        batch_size, action_count = self.q_values.shape
        grounded = config.grounded
        expected = {
            "state_value": (batch_size,),
            "policy_logits": (batch_size, action_count),
            "recurrent_state": (batch_size, grounded.recurrent_hidden_dim),
            "joint_state_embedding": (batch_size, grounded.d_model),
            "world_state_embedding": (batch_size, grounded.d_model),
            "candidate_embeddings": (batch_size, action_count, grounded.d_model),
            "action_mask": (batch_size, action_count),
        }
        for name, shape in expected.items():
            value = getattr(self, name)
            if tuple(value.shape) != shape:
                raise ValueError(
                    f"hybrid IQL {name} must have shape {shape}, got {tuple(value.shape)}"
                )
        if self.action_mask.dtype != torch.bool:
            raise TypeError("hybrid IQL action_mask must use torch.bool")
        for name in (
            "q_values",
            "state_value",
            "policy_logits",
            "recurrent_state",
            "joint_state_embedding",
            "world_state_embedding",
            "candidate_embeddings",
        ):
            value = getattr(self, name)
            if not torch.is_floating_point(value):
                raise TypeError(f"hybrid IQL {name} must be floating point")
            if not bool(torch.isfinite(value).all().item()):
                raise ValueError(f"hybrid IQL {name} contains NaN or infinity")
        return batch_size, action_count


class HybridDomainIQLModel(HybridDomainCandidateQModel):
    """Hybrid fusion critic plus a detached-feature AWAC actor."""

    def __init__(self, config: HybridIQLConfig | None = None) -> None:
        iql = config or HybridIQLConfig()
        super().__init__(iql.candidate_q_config())
        self.iql_config = iql
        cfg = iql.grounded
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
    ) -> HybridIQLOutput:
        base = super().forward(batch, recurrent_state, validate=validate)
        action_count = int(base.q_values.shape[1])
        state = base.joint_state_embedding.unsqueeze(1).expand(-1, action_count, -1)
        memory = base.recurrent_state.unsqueeze(1).expand(-1, action_count, -1)
        policy_features = torch.cat(
            [base.candidate_embeddings, state, memory],
            dim=-1,
        ).detach()
        policy_logits = self.candidate_policy_head(policy_features).squeeze(-1)
        policy_logits = policy_logits.masked_fill(~base.action_mask, 0.0)
        state_value = self.state_value_head(
            torch.cat(
                [base.joint_state_embedding, base.recurrent_state],
                dim=-1,
            )
        ).squeeze(-1)
        output = HybridIQLOutput(
            q_values=base.q_values,
            state_value=state_value,
            policy_logits=policy_logits,
            recurrent_state=base.recurrent_state,
            joint_state_embedding=base.joint_state_embedding,
            world_state_embedding=base.world_state_embedding,
            candidate_embeddings=base.candidate_embeddings,
            action_mask=base.action_mask,
        )
        if validate:
            output.validate(self.iql_config)
        return output


def initialize_iql_from_candidate_q(
    model: HybridDomainIQLModel,
    source_state: dict[str, Tensor],
) -> dict[str, tuple[str, ...]]:
    """Model-init migration from a finite hybrid Candidate-Q publication.

    The factual trunk, memories, fusion blocks and Q head are inherited.  The
    actor starts with the exact source Q ranking, which preserves the healthy
    parent's behavior at gate zero without pretending this is an exact
    continuation.  The value head starts on the same scalar offset as Q and is
    then fitted by expectile regression.
    """

    destination = model.state_dict()
    loaded: dict[str, Tensor] = {}
    inherited: list[str] = []
    for key, value in source_state.items():
        target = destination.get(key)
        if target is None or tuple(target.shape) != tuple(value.shape):
            continue
        if value.is_floating_point() and not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"Candidate-Q source tensor is non-finite: {key}")
        loaded[key] = value
        inherited.append(key)
    result = model.load_state_dict(loaded, strict=False)
    q_keys = tuple(key for key in inherited if key.startswith("candidate_q_head."))
    if not q_keys:
        raise ValueError("IQL model-init source has no compatible Candidate-Q head")

    # Actor gate-zero ordering exactly matches the healthy Q controller.
    model.candidate_policy_head.load_state_dict(
        model.candidate_q_head.state_dict(),
        strict=True,
    )
    with torch.no_grad():
        q_final = model.candidate_q_head[-1]
        value_final = model.state_value_head[-1]
        if not isinstance(q_final, nn.Linear) or not isinstance(value_final, nn.Linear):
            raise TypeError("hybrid control heads do not end in Linear layers")
        value_final.bias.copy_(q_final.bias)

    fresh = tuple(
        sorted(
            key
            for key in result.missing_keys
            if key.startswith(("candidate_policy_head.", "state_value_head."))
        )
    )
    return {
        "inherited": tuple(sorted(inherited)),
        "fresh": fresh,
    }


__all__ = [
    "HYBRID_IQL_MODEL_VERSION",
    "HybridDomainIQLModel",
    "HybridIQLConfig",
    "HybridIQLOutput",
    "initialize_iql_from_candidate_q",
]
