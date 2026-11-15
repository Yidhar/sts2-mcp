"""Recurrent hybrid actor-critic for the fresh FIFO V-trace lineage.

The factual encoder, dual macro/combat stems, shared fusion stream and split
run/combat memory are retained from the hybrid controller.  Control is once
again a dense stochastic policy with a candidate-independent scalar value.
Unlike the retired IQL actor, policy features are *not* detached: current
on-policy gradients and entropy train the shared representation together with
the value regression.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Final

import torch
from torch import Tensor, nn

from .grounded_candidate import (
    COMBAT_DOMAIN_ID,
    GroundedCandidateBatch,
    GroundedCandidateConfig,
    WorldEncoding,
    WorldTokenBatch,
    _safe_valid_mask,
)
from .hybrid_candidate_q import (
    HybridCandidateQConfig,
    HybridDomainCandidateQModel,
    HybridRepresentationOutput,
)

HYBRID_ACTOR_CRITIC_MODEL_VERSION: Final = "sts2-hybrid-domain-actor-critic-v1"
HYBRID_ACTOR_CRITIC_V2_MODEL_VERSION: Final = "sts2-hybrid-domain-actor-critic-v2"
# Fixed zone IDs from encoding.grounded._ZONE_IDS representing persistent build
# facts (deck=2, player=13, relic=15, potion=16, modifier=20, run=22), isolated
# from variable-size map graphs and transient picker containers.
DECK_ZONE_ID: Final = 2
PLAYER_ZONE_ID: Final = 13
MODIFIER_ZONE_ID: Final = 20
RUN_ZONE_ID: Final = 22
NUMERIC_FACT_SLOT_START: Final = 6
NUMERIC_FACT_SLOT_END: Final = 166
DECK_FLOOR_ADDED_SLOT: Final = 52
DECK_QUANTITY_SLOT: Final = 131
DECK_REFERENCE_SIZE: Final = 12.0
DURABLE_DECK_SIZE_DIM: Final = 4
PLAYER_CURRENT_HP_OFFSET: Final = 34 - NUMERIC_FACT_SLOT_START
PLAYER_GOLD_OFFSET: Final = 54 - NUMERIC_FACT_SLOT_START
PLAYER_HP_OFFSET: Final = 62 - NUMERIC_FACT_SLOT_START
PLAYER_MAX_HP_OFFSET: Final = 111 - NUMERIC_FACT_SLOT_START
PLAYER_OPEN_POTION_SLOTS_OFFSET: Final = 119 - NUMERIC_FACT_SLOT_START
RUN_ACT_OFFSET: Final = 6 - NUMERIC_FACT_SLOT_START
RUN_ACT_SLOT: Final = 6
RUN_FLOOR_OFFSET: Final = 51 - NUMERIC_FACT_SLOT_START
SALIENT_VITALS_DIM: Final = 4
DURABLE_PROGRESS_VITALS_DIM: Final = 10
DURABLE_STAGE_DIM: Final = 5
DURABLE_HP_BASIS_DIM: Final = 5
DURABLE_LIQUID_BASIS_DIM: Final = 4
DURABLE_CYCLE_BASIS_DIM: Final = 3
DURABLE_TILT_BETAS: Final[tuple[float, ...]] = (1.5, 3.0, 6.0)
CRITIC_REVIVAL_CONTEXT_DIM: Final = 6
PERSISTENT_BUILD_ZONE_IDS: Final[tuple[int, ...]] = (2, 13, 15, 16, 20, 22)
IDENTITY_BLIND_CARD_COUNT_ZONE_IDS: Final[frozenset[int]] = frozenset({2, 20})
V2_FRESH_TENSOR_PREFIXES: Final[tuple[str, ...]] = (
    "branch_policy_head.",
    "branch_vitals_",
    "candidate_role_",
    "durable_cycle_",
    "durable_deck_",
    "durable_hp_",
    "durable_liquid_",
    "durable_progress_vitals_",
    "durable_revival_",
    "durable_stage_",
    "durable_value_head.",
    "durable_vitals_",
    "persistent_build_",
    "persistent_vitals_",
    "state_revival_",
)
RETIRED_DECONFOUNDED_SOURCE_PREFIXES: Final[tuple[str, ...]] = (
    "durable_deck_size_head.",
    "durable_deck_size_linear.",
    "durable_vitals_head.",
    "durable_vitals_linear.",
)
RETIRED_MONOTONE_DECONFOUNDED_SOURCE_PREFIXES: Final[tuple[str, ...]] = (
    *RETIRED_DECONFOUNDED_SOURCE_PREFIXES,
    "durable_progress_vitals_head.",
)


@dataclass(frozen=True, slots=True)
class HybridActorCriticConfig:
    grounded: GroundedCandidateConfig = field(default_factory=GroundedCandidateConfig)
    domain_layers: int = 1
    joint_layers: int = 2
    enable_hierarchical_branch_policy: bool = False
    enable_durable_value_head: bool = False
    enable_persistent_build_context: bool = False
    enable_direct_vitals_context: bool = False
    enable_macro_branch_prior: bool = False
    enable_persistent_only_durable_value: bool = False
    enable_asymmetric_revival_critic: bool = False
    enable_draw_weighted_durable_value: bool = False
    enable_role_conditioned_candidate_policy: bool = False
    enable_deconfounded_durable_value: bool = False
    enable_monotone_deconfounded_durable_value: bool = False
    enable_balanced_deconfounded_durable_value: bool = False

    def __post_init__(self) -> None:
        HybridCandidateQConfig(
            grounded=self.grounded,
            domain_layers=self.domain_layers,
            joint_layers=self.joint_layers,
        )
        for name in (
            "enable_hierarchical_branch_policy",
            "enable_durable_value_head",
            "enable_persistent_build_context",
            "enable_direct_vitals_context",
            "enable_macro_branch_prior",
            "enable_persistent_only_durable_value",
            "enable_asymmetric_revival_critic",
            "enable_draw_weighted_durable_value",
            "enable_role_conditioned_candidate_policy",
            "enable_deconfounded_durable_value",
            "enable_monotone_deconfounded_durable_value",
            "enable_balanced_deconfounded_durable_value",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"hybrid actor-critic {name} must be boolean")
        if self.enable_draw_weighted_durable_value and not self.enable_persistent_only_durable_value:
            raise ValueError(
                "enable_draw_weighted_durable_value requires enable_persistent_only_durable_value=True"
            )
        if self.enable_deconfounded_durable_value and not self.enable_draw_weighted_durable_value:
            raise ValueError(
                "enable_deconfounded_durable_value requires enable_draw_weighted_durable_value=True"
            )
        if (
            self.enable_monotone_deconfounded_durable_value
            and not self.enable_deconfounded_durable_value
        ):
            raise ValueError(
                "enable_monotone_deconfounded_durable_value requires enable_deconfounded_durable_value=True"
            )
        if (
            self.enable_balanced_deconfounded_durable_value
            and not self.enable_monotone_deconfounded_durable_value
        ):
            raise ValueError(
                "enable_balanced_deconfounded_durable_value requires enable_monotone_deconfounded_durable_value=True"
            )

    @property
    def has_v2_extensions(self) -> bool:
        return (
            self.enable_hierarchical_branch_policy
            or self.enable_durable_value_head
            or self.enable_persistent_build_context
            or self.enable_direct_vitals_context
            or self.enable_macro_branch_prior
            or self.enable_persistent_only_durable_value
            or self.enable_asymmetric_revival_critic
            or self.enable_draw_weighted_durable_value
            or self.enable_role_conditioned_candidate_policy
            or self.enable_deconfounded_durable_value
            or self.enable_monotone_deconfounded_durable_value
            or self.enable_balanced_deconfounded_durable_value
        )

    @classmethod
    def durable_v2(
        cls,
        grounded: GroundedCandidateConfig | None = None,
        *,
        domain_layers: int = 1,
        joint_layers: int = 2,
    ) -> HybridActorCriticConfig:
        return cls(
            grounded=grounded or GroundedCandidateConfig(),
            domain_layers=domain_layers,
            joint_layers=joint_layers,
            enable_hierarchical_branch_policy=True,
            enable_durable_value_head=True,
            enable_persistent_build_context=True,
        )

    @classmethod
    def durable_v3(
        cls,
        grounded: GroundedCandidateConfig | None = None,
        *,
        domain_layers: int = 1,
        joint_layers: int = 2,
    ) -> HybridActorCriticConfig:
        return cls(
            grounded=grounded or GroundedCandidateConfig(),
            domain_layers=domain_layers,
            joint_layers=joint_layers,
            enable_hierarchical_branch_policy=True,
            enable_durable_value_head=True,
            enable_persistent_build_context=True,
            enable_direct_vitals_context=True,
            enable_macro_branch_prior=True,
        )

    @classmethod
    def durable_v4(
        cls,
        grounded: GroundedCandidateConfig | None = None,
        *,
        domain_layers: int = 1,
        joint_layers: int = 2,
    ) -> HybridActorCriticConfig:
        return cls(
            grounded=grounded or GroundedCandidateConfig(),
            domain_layers=domain_layers,
            joint_layers=joint_layers,
            enable_hierarchical_branch_policy=True,
            enable_durable_value_head=True,
            enable_persistent_build_context=True,
            enable_direct_vitals_context=True,
            enable_macro_branch_prior=True,
            enable_persistent_only_durable_value=True,
            enable_asymmetric_revival_critic=True,
        )

    @classmethod
    def durable_v5(
        cls,
        grounded: GroundedCandidateConfig | None = None,
        *,
        domain_layers: int = 1,
        joint_layers: int = 2,
    ) -> HybridActorCriticConfig:
        return cls(
            grounded=grounded or GroundedCandidateConfig(),
            domain_layers=domain_layers,
            joint_layers=joint_layers,
            enable_hierarchical_branch_policy=True,
            enable_durable_value_head=True,
            enable_persistent_build_context=True,
            enable_direct_vitals_context=True,
            enable_macro_branch_prior=True,
            enable_persistent_only_durable_value=True,
            enable_asymmetric_revival_critic=True,
            enable_draw_weighted_durable_value=True,
            enable_role_conditioned_candidate_policy=True,
        )

    @classmethod
    def durable_v6(
        cls,
        grounded: GroundedCandidateConfig | None = None,
        *,
        domain_layers: int = 1,
        joint_layers: int = 2,
    ) -> HybridActorCriticConfig:
        return cls(
            grounded=grounded or GroundedCandidateConfig(),
            domain_layers=domain_layers,
            joint_layers=joint_layers,
            enable_hierarchical_branch_policy=True,
            enable_durable_value_head=True,
            enable_persistent_build_context=True,
            enable_direct_vitals_context=True,
            enable_macro_branch_prior=True,
            enable_persistent_only_durable_value=True,
            enable_asymmetric_revival_critic=True,
            enable_draw_weighted_durable_value=True,
            enable_role_conditioned_candidate_policy=True,
            enable_deconfounded_durable_value=True,
        )

    @classmethod
    def durable_v7(
        cls,
        grounded: GroundedCandidateConfig | None = None,
        *,
        domain_layers: int = 1,
        joint_layers: int = 2,
    ) -> HybridActorCriticConfig:
        return cls(
            grounded=grounded or GroundedCandidateConfig(),
            domain_layers=domain_layers,
            joint_layers=joint_layers,
            enable_hierarchical_branch_policy=True,
            enable_durable_value_head=True,
            enable_persistent_build_context=True,
            enable_direct_vitals_context=True,
            enable_macro_branch_prior=True,
            enable_persistent_only_durable_value=True,
            enable_asymmetric_revival_critic=True,
            enable_draw_weighted_durable_value=True,
            enable_role_conditioned_candidate_policy=True,
            enable_deconfounded_durable_value=True,
            enable_monotone_deconfounded_durable_value=True,
        )

    @classmethod
    def durable_v8(
        cls,
        grounded: GroundedCandidateConfig | None = None,
        *,
        domain_layers: int = 1,
        joint_layers: int = 2,
    ) -> HybridActorCriticConfig:
        return cls(
            grounded=grounded or GroundedCandidateConfig(),
            domain_layers=domain_layers,
            joint_layers=joint_layers,
            enable_hierarchical_branch_policy=True,
            enable_durable_value_head=True,
            enable_persistent_build_context=True,
            enable_direct_vitals_context=True,
            enable_macro_branch_prior=True,
            enable_persistent_only_durable_value=True,
            enable_asymmetric_revival_critic=True,
            enable_draw_weighted_durable_value=True,
            enable_role_conditioned_candidate_policy=True,
            enable_deconfounded_durable_value=True,
            enable_monotone_deconfounded_durable_value=True,
            enable_balanced_deconfounded_durable_value=True,
        )

    @classmethod
    def from_mapping(
        cls,
        mapping: dict[str, Any],
        *,
        fallback_grounded: GroundedCandidateConfig,
    ) -> HybridActorCriticConfig:
        version = mapping.get("version")
        if version == HYBRID_ACTOR_CRITIC_MODEL_VERSION:
            return cls(
                grounded=fallback_grounded,
                domain_layers=int(mapping.get("domain_layers", 1)),
                joint_layers=int(mapping.get("joint_layers", 2)),
            )
        if version == HYBRID_ACTOR_CRITIC_V2_MODEL_VERSION:
            return cls(
                grounded=fallback_grounded,
                domain_layers=int(mapping.get("domain_layers", 1)),
                joint_layers=int(mapping.get("joint_layers", 2)),
                enable_hierarchical_branch_policy=bool(
                    mapping.get("enable_hierarchical_branch_policy", False)
                ),
                enable_durable_value_head=bool(
                    mapping.get("enable_durable_value_head", False)
                ),
                enable_persistent_build_context=bool(
                    mapping.get("enable_persistent_build_context", False)
                ),
                enable_direct_vitals_context=bool(
                    mapping.get("enable_direct_vitals_context", False)
                ),
                enable_macro_branch_prior=bool(
                    mapping.get("enable_macro_branch_prior", False)
                ),
                enable_persistent_only_durable_value=bool(
                    mapping.get("enable_persistent_only_durable_value", False)
                ),
                enable_asymmetric_revival_critic=bool(
                    mapping.get("enable_asymmetric_revival_critic", False)
                ),
                enable_draw_weighted_durable_value=bool(
                    mapping.get("enable_draw_weighted_durable_value", False)
                ),
                enable_role_conditioned_candidate_policy=bool(
                    mapping.get("enable_role_conditioned_candidate_policy", False)
                ),
                enable_deconfounded_durable_value=bool(
                    mapping.get("enable_deconfounded_durable_value", False)
                ),
                enable_monotone_deconfounded_durable_value=bool(
                    mapping.get("enable_monotone_deconfounded_durable_value", False)
                ),
                enable_balanced_deconfounded_durable_value=bool(
                    mapping.get("enable_balanced_deconfounded_durable_value", False)
                ),
            )
        raise ValueError(f"unsupported hybrid actor-critic model_config version: {version!r}")

    def as_mapping(self) -> dict[str, Any]:
        base: dict[str, Any] = {
            "version": (
                HYBRID_ACTOR_CRITIC_V2_MODEL_VERSION
                if self.has_v2_extensions
                else HYBRID_ACTOR_CRITIC_MODEL_VERSION
            ),
            "grounded": asdict(self.grounded),
            "domain_layers": self.domain_layers,
            "joint_layers": self.joint_layers,
        }
        if self.has_v2_extensions:
            base.update(
                enable_hierarchical_branch_policy=self.enable_hierarchical_branch_policy,
                enable_durable_value_head=self.enable_durable_value_head,
                enable_persistent_build_context=self.enable_persistent_build_context,
            )
            if self.enable_direct_vitals_context or self.enable_macro_branch_prior:
                base.update(
                    enable_direct_vitals_context=self.enable_direct_vitals_context,
                    enable_macro_branch_prior=self.enable_macro_branch_prior,
                )
            if (
                self.enable_persistent_only_durable_value
                or self.enable_asymmetric_revival_critic
            ):
                base.update(
                    enable_persistent_only_durable_value=(
                        self.enable_persistent_only_durable_value
                    ),
                    enable_asymmetric_revival_critic=(
                        self.enable_asymmetric_revival_critic
                    ),
                )
            if (
                self.enable_draw_weighted_durable_value
                or self.enable_role_conditioned_candidate_policy
            ):
                base.update(
                    enable_draw_weighted_durable_value=(
                        self.enable_draw_weighted_durable_value
                    ),
                    enable_role_conditioned_candidate_policy=(
                        self.enable_role_conditioned_candidate_policy
                    ),
                )
            if self.enable_deconfounded_durable_value:
                base.update(
                    enable_deconfounded_durable_value=self.enable_deconfounded_durable_value,
                )
            if self.enable_monotone_deconfounded_durable_value:
                base.update(
                    enable_monotone_deconfounded_durable_value=(
                        self.enable_monotone_deconfounded_durable_value
                    ),
                )
            if self.enable_balanced_deconfounded_durable_value:
                base.update(
                    enable_balanced_deconfounded_durable_value=(
                        self.enable_balanced_deconfounded_durable_value
                    ),
                )
        return base

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
    durable_value: Tensor | None = None  # [B], candidate-independent when enabled

    def validate(
        self,
        config: HybridActorCriticConfig,
        *,
        require_durable_value: bool = True,
    ) -> tuple[int, int]:
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
        if require_durable_value and config.enable_durable_value_head:
            if self.durable_value is None:
                raise ValueError("hybrid actor durable_value is required when enabled")
            if tuple(self.durable_value.shape) != (batch_size,):
                raise ValueError("hybrid actor durable_value has the wrong shape")
            if not torch.is_floating_point(self.durable_value) or not bool(
                torch.isfinite(self.durable_value).all().item()
            ):
                raise ValueError("hybrid actor durable_value must be finite floating point")
        return batch_size, action_count


def hierarchical_greedy_action_index(
    logits: Tensor,
    snapshot_policy_branch_ids: Tensor,
    action_mask: Tensor,
    *,
    domain_id: int,
) -> int:
    """Select greedy candidate honoring macro branch marginals when hierarchical."""

    mask = action_mask.bool()
    if not bool(mask.any().item()):
        raise ValueError("cannot select greedy action on an empty mask")
    safe_logs = torch.log_softmax(logits[: mask.numel()].float().masked_fill(~mask, -torch.inf), dim=0)
    if domain_id == COMBAT_DOMAIN_ID:
        return int(safe_logs.argmax(dim=0).item())
    branch_ids = snapshot_policy_branch_ids[: mask.numel()].long()
    action_count = int(mask.numel())
    in_branch = (
        branch_ids.unsqueeze(0)
        == torch.arange(action_count, device=logits.device).unsqueeze(1)
    ) & mask.unsqueeze(0)
    branch_active = in_branch.any(dim=-1)
    safe_in_branch = _safe_valid_mask(in_branch)
    branch_marginals = torch.logsumexp(
        safe_logs.unsqueeze(0).expand(action_count, -1).masked_fill(~safe_in_branch, -torch.inf),
        dim=-1,
    )
    branch_marginals = branch_marginals.masked_fill(~branch_active, -torch.inf)
    best_branch = int(branch_marginals.argmax(dim=0).item())
    in_selected = (branch_ids == best_branch) & mask
    within_scores = safe_logs.masked_fill(~in_selected, -torch.inf)
    return int(within_scores.argmax(dim=0).item())


@dataclass(frozen=True, slots=True)
class _PreparedPersistentWorld:
    world: WorldTokenBatch
    token_weights: Tensor  # [B, W]
    deck_card_mask: Tensor  # [B, W]
    draw_prob: Tensor  # [B, W]
    deck_size: Tensor  # [B]
    has_deck: Tensor  # [B]
    child_uv: Tensor  # [B, W, W], (child u, parent card v)


class HybridDomainActorCriticModel(HybridDomainCandidateQModel):
    """Hybrid dual-stream representation with optional v2 durable/branch extensions."""

    def __init__(self, config: HybridActorCriticConfig | None = None) -> None:
        actor_critic = config or HybridActorCriticConfig()
        super().__init__(actor_critic.candidate_q_config())
        self.actor_critic_config = actor_critic
        del self.candidate_q_head
        cfg = actor_critic.grounded
        player_owner_digest = hashlib.sha256(b"owner\0player").digest()
        self._player_owner_id = 2 + int.from_bytes(player_owner_digest[:8], "big") % (
            cfg.owner_vocab_size - 2
        )
        mapping_item_digest = hashlib.sha256(b"type\0mapping:item").digest()
        self._mapping_item_type_id = 2 + int.from_bytes(mapping_item_digest[:8], "big") % (
            cfg.type_vocab_size - 2
        )
        control_input = cfg.d_model * 2 + cfg.recurrent_hidden_dim
        self.candidate_policy_head = nn.Sequential(
            nn.Linear(control_input, cfg.ffn_dim),
            nn.SiLU(),
            nn.Linear(cfg.ffn_dim, 1),
        )
        if actor_critic.enable_role_conditioned_candidate_policy:
            self.candidate_role_gate = nn.Linear(cfg.d_model, cfg.d_model)
            nn.init.normal_(self.candidate_role_gate.weight, mean=0.0, std=0.15)
            nn.init.zeros_(self.candidate_role_gate.bias)
            role_fc1 = nn.Linear(cfg.d_model * 4, cfg.ffn_dim)
            nn.init.normal_(role_fc1.weight, mean=0.0, std=0.05)
            nn.init.zeros_(role_fc1.bias)
            role_out = nn.Linear(cfg.ffn_dim, 1)
            nn.init.zeros_(role_out.weight)
            nn.init.zeros_(role_out.bias)
            self.candidate_role_policy_head = nn.Sequential(
                nn.LayerNorm(cfg.d_model * 4),
                role_fc1,
                nn.SiLU(),
                role_out,
            )
        self.state_value_head = nn.Sequential(
            nn.Linear(cfg.d_model + cfg.recurrent_hidden_dim, cfg.ffn_dim),
            nn.SiLU(),
            nn.Linear(cfg.ffn_dim, 1),
        )
        if actor_critic.enable_persistent_build_context:
            self.persistent_build_queries = nn.Parameter(
                torch.empty(1, 4, cfg.d_model)
            )
            nn.init.normal_(self.persistent_build_queries, mean=0.0, std=0.02)
            self.persistent_build_attn = nn.MultiheadAttention(
                cfg.d_model,
                cfg.n_heads,
                dropout=cfg.dropout,
                batch_first=True,
            )
            self.persistent_build_norm = nn.LayerNorm(cfg.d_model)
            self.persistent_build_to_state = nn.Linear(cfg.d_model, cfg.d_model)
            nn.init.zeros_(self.persistent_build_to_state.weight)
            nn.init.zeros_(self.persistent_build_to_state.bias)
            cand_build_out = nn.Linear(cfg.ffn_dim, cfg.d_model)
            nn.init.zeros_(cand_build_out.weight)
            nn.init.zeros_(cand_build_out.bias)
            self.persistent_build_to_candidate = nn.Sequential(
                nn.LayerNorm(cfg.d_model * 2),
                nn.Linear(cfg.d_model * 2, cfg.ffn_dim),
                nn.SiLU(),
                cand_build_out,
            )
            if actor_critic.enable_draw_weighted_durable_value:
                deck_proj_out = nn.Linear(cfg.d_model, cfg.d_model)
                nn.init.zeros_(deck_proj_out.weight)
                nn.init.zeros_(deck_proj_out.bias)
                self.persistent_build_deck_proj = nn.Sequential(
                    nn.LayerNorm(cfg.d_model * 3),
                    nn.Linear(cfg.d_model * 3, cfg.d_model),
                    nn.SiLU(),
                    deck_proj_out,
                )
        rev_dim = (
            CRITIC_REVIVAL_CONTEXT_DIM
            if actor_critic.enable_asymmetric_revival_critic
            else 0
        )
        if actor_critic.enable_asymmetric_revival_critic:
            self.state_revival_linear = nn.Linear(rev_dim, 1, bias=False)
            nn.init.zeros_(self.state_revival_linear.weight)
            state_rev_fc1 = nn.Linear(rev_dim, cfg.ffn_dim, bias=False)
            nn.init.normal_(state_rev_fc1.weight, mean=0.0, std=0.35)
            state_rev_out = nn.Linear(cfg.ffn_dim, 1, bias=False)
            nn.init.zeros_(state_rev_out.weight)
            self.state_revival_head = nn.Sequential(
                state_rev_fc1,
                nn.SiLU(),
                state_rev_out,
            )
            if actor_critic.enable_durable_value_head:
                self.durable_revival_linear = nn.Linear(rev_dim, 1, bias=False)
                nn.init.zeros_(self.durable_revival_linear.weight)
                dur_rev_fc1 = nn.Linear(rev_dim, cfg.ffn_dim, bias=False)
                nn.init.normal_(dur_rev_fc1.weight, mean=0.0, std=0.35)
                dur_rev_out = nn.Linear(cfg.ffn_dim, 1, bias=False)
                nn.init.zeros_(dur_rev_out.weight)
                self.durable_revival_head = nn.Sequential(
                    dur_rev_fc1,
                    nn.SiLU(),
                    dur_rev_out,
                )
        if actor_critic.enable_direct_vitals_context:
            numeric_width = min(cfg.token_feature_dim, NUMERIC_FACT_SLOT_END) - NUMERIC_FACT_SLOT_START
            salient_dim = (
                SALIENT_VITALS_DIM
                if actor_critic.enable_persistent_only_durable_value
                else 0
            )
            vitals_in_dim = numeric_width * 2 + len(PERSISTENT_BUILD_ZONE_IDS) + salient_dim
            vitals_fc1 = nn.Linear(vitals_in_dim, cfg.ffn_dim)
            nn.init.normal_(vitals_fc1.weight, mean=0.0, std=0.35)
            nn.init.zeros_(vitals_fc1.bias)
            self.persistent_vitals_encoder = nn.Sequential(
                vitals_fc1,
                nn.SiLU(),
                nn.Linear(cfg.ffn_dim, cfg.d_model),
            )
            self.persistent_vitals_to_state = nn.Linear(cfg.d_model, cfg.d_model)
            nn.init.zeros_(self.persistent_vitals_to_state.weight)
            nn.init.zeros_(self.persistent_vitals_to_state.bias)
            if actor_critic.enable_durable_value_head:
                if actor_critic.enable_monotone_deconfounded_durable_value:
                    stage_in_dim = DURABLE_STAGE_DIM + rev_dim
                    stage_mid_dim = max(16, cfg.ffn_dim // 2)
                    stage_fc1 = nn.Linear(stage_in_dim, stage_mid_dim)
                    nn.init.normal_(stage_fc1.weight, mean=0.0, std=0.15)
                    nn.init.zeros_(stage_fc1.bias)
                    stage_out = nn.Linear(stage_mid_dim, 1)
                    nn.init.zeros_(stage_out.weight)
                    nn.init.zeros_(stage_out.bias)
                    self.durable_stage_baseline_head = nn.Sequential(
                        nn.LayerNorm(stage_in_dim),
                        stage_fc1,
                        nn.SiLU(),
                        stage_out,
                    )

                    balanced_init = actor_critic.enable_balanced_deconfounded_durable_value
                    hp_fc1 = nn.Linear(stage_in_dim, stage_mid_dim)
                    nn.init.normal_(hp_fc1.weight, mean=0.0, std=0.10)
                    nn.init.zeros_(hp_fc1.bias)
                    hp_out = nn.Linear(stage_mid_dim, DURABLE_HP_BASIS_DIM)
                    nn.init.zeros_(hp_out.weight)
                    nn.init.constant_(hp_out.bias, -5.2 if balanced_init else -2.8)
                    self.durable_hp_gate = nn.Sequential(
                        nn.LayerNorm(stage_in_dim),
                        hp_fc1,
                        nn.SiLU(),
                        hp_out,
                    )

                    liq_fc1 = nn.Linear(stage_in_dim, stage_mid_dim)
                    nn.init.normal_(liq_fc1.weight, mean=0.0, std=0.10)
                    nn.init.zeros_(liq_fc1.bias)
                    liq_out = nn.Linear(stage_mid_dim, DURABLE_LIQUID_BASIS_DIM)
                    nn.init.zeros_(liq_out.weight)
                    nn.init.constant_(liq_out.bias, -5.2 if balanced_init else -3.2)
                    self.durable_liquid_gate = nn.Sequential(
                        nn.LayerNorm(stage_in_dim),
                        liq_fc1,
                        nn.SiLU(),
                        liq_out,
                    )

                    cyc_fc1 = nn.Linear(stage_in_dim, stage_mid_dim)
                    nn.init.normal_(cyc_fc1.weight, mean=0.0, std=0.10)
                    nn.init.zeros_(cyc_fc1.bias)
                    cyc_out = nn.Linear(stage_mid_dim, DURABLE_CYCLE_BASIS_DIM)
                    nn.init.zeros_(cyc_out.weight)
                    nn.init.constant_(cyc_out.bias, -5.5 if balanced_init else -3.8)
                    self.durable_cycle_gate = nn.Sequential(
                        nn.LayerNorm(stage_in_dim),
                        cyc_fc1,
                        nn.SiLU(),
                        cyc_out,
                    )
                elif actor_critic.enable_deconfounded_durable_value:
                    prog_in_dim = DURABLE_PROGRESS_VITALS_DIM + rev_dim
                    prog_mid_dim = max(16, cfg.ffn_dim // 2)
                    prog_fc1 = nn.Linear(prog_in_dim, cfg.ffn_dim)
                    nn.init.normal_(prog_fc1.weight, mean=0.0, std=0.15)
                    nn.init.zeros_(prog_fc1.bias)
                    prog_fc2 = nn.Linear(cfg.ffn_dim, prog_mid_dim)
                    nn.init.normal_(prog_fc2.weight, mean=0.0, std=0.10)
                    nn.init.zeros_(prog_fc2.bias)
                    prog_out = nn.Linear(prog_mid_dim, 1)
                    nn.init.zeros_(prog_out.weight)
                    nn.init.zeros_(prog_out.bias)
                    self.durable_progress_vitals_head = nn.Sequential(
                        nn.LayerNorm(prog_in_dim),
                        prog_fc1,
                        nn.SiLU(),
                        prog_fc2,
                        nn.SiLU(),
                        prog_out,
                    )
                else:
                    if actor_critic.enable_persistent_only_durable_value:
                        self.durable_vitals_linear = nn.Linear(SALIENT_VITALS_DIM, 1, bias=False)
                        nn.init.zeros_(self.durable_vitals_linear.weight)
                    dur_vitals_fc1 = nn.Linear(vitals_in_dim + rev_dim, cfg.ffn_dim)
                    nn.init.normal_(dur_vitals_fc1.weight, mean=0.0, std=0.35)
                    nn.init.zeros_(dur_vitals_fc1.bias)
                    dur_vitals_out = nn.Linear(cfg.ffn_dim, 1)
                    nn.init.zeros_(dur_vitals_out.weight)
                    nn.init.zeros_(dur_vitals_out.bias)
                    self.durable_vitals_head = nn.Sequential(
                        dur_vitals_fc1,
                        nn.SiLU(),
                        dur_vitals_out,
                    )
            if actor_critic.enable_hierarchical_branch_policy:
                branch_vitals_out = nn.Linear(cfg.ffn_dim, 1)
                nn.init.zeros_(branch_vitals_out.weight)
                nn.init.zeros_(branch_vitals_out.bias)
                self.branch_vitals_head = nn.Sequential(
                    nn.Linear(cfg.d_model * 3 + salient_dim, cfg.ffn_dim),
                    nn.SiLU(),
                    branch_vitals_out,
                )
                if salient_dim > 0:
                    self.branch_vitals_norm = nn.LayerNorm(cfg.d_model * 2)
                    self.branch_vitals_bilinear = nn.Linear(
                        cfg.d_model * 2,
                        salient_dim + 1,
                        bias=False,
                    )
                    nn.init.zeros_(self.branch_vitals_bilinear.weight)
        if actor_critic.enable_durable_value_head:
            memory_half = cfg.recurrent_hidden_dim // 2
            durable_context_blocks = (
                1
                + int(actor_critic.enable_persistent_build_context)
                + int(actor_critic.enable_direct_vitals_context)
            )
            durable_dim = cfg.d_model * durable_context_blocks + memory_half + rev_dim
            durable_out = nn.Linear(cfg.ffn_dim, 1)
            nn.init.zeros_(durable_out.weight)
            nn.init.zeros_(durable_out.bias)
            self.durable_value_head = nn.Sequential(
                nn.LayerNorm(durable_dim),
                nn.Linear(durable_dim, cfg.ffn_dim),
                nn.SiLU(),
                durable_out,
            )
            if actor_critic.enable_draw_weighted_durable_value:
                deck_ctx_dim = (
                    cfg.d_model * (1 + int(actor_critic.enable_direct_vitals_context))
                    + memory_half
                    + rev_dim
                )
                self.durable_deck_norm = nn.LayerNorm(cfg.d_model * 3)
                self.durable_deck_card_fc = nn.Linear(cfg.d_model * 3, cfg.ffn_dim)
                self.durable_deck_raw_fc = nn.Linear(cfg.token_feature_dim * 2, cfg.ffn_dim)
                self.durable_deck_ctx_norm = nn.LayerNorm(deck_ctx_dim)
                self.durable_deck_ctx_fc = nn.Linear(deck_ctx_dim, cfg.ffn_dim, bias=False)
                self.durable_deck_out = nn.Linear(cfg.ffn_dim * 3, 1)
                nn.init.zeros_(self.durable_deck_out.weight)
                nn.init.zeros_(self.durable_deck_out.bias)
                if actor_critic.enable_monotone_deconfounded_durable_value:
                    self.durable_deck_tilted_out = nn.Linear(
                        cfg.ffn_dim * 3,
                        len(DURABLE_TILT_BETAS),
                    )
                    nn.init.zeros_(self.durable_deck_tilted_out.weight)
                    nn.init.zeros_(self.durable_deck_tilted_out.bias)
                    if actor_critic.enable_balanced_deconfounded_durable_value:
                        self.durable_deck_linear_logit = nn.Parameter(
                            torch.tensor(math.log(math.e - 1.0))
                        )
                        self.durable_deck_tilt_logits = nn.Parameter(
                            torch.tensor([-4.2, -4.0, -4.2])
                        )
                    else:
                        self.durable_deck_linear_logit = nn.Parameter(torch.zeros(()))
                        self.durable_deck_tilt_logits = nn.Parameter(
                            torch.tensor([-1.8, -1.5, -1.8])
                        )
                if not actor_critic.enable_deconfounded_durable_value:
                    deck_size_fc1 = nn.Linear(DURABLE_DECK_SIZE_DIM + rev_dim, cfg.ffn_dim)
                    nn.init.normal_(deck_size_fc1.weight, mean=0.0, std=0.25)
                    nn.init.zeros_(deck_size_fc1.bias)
                    deck_size_out = nn.Linear(cfg.ffn_dim, 1)
                    nn.init.zeros_(deck_size_out.weight)
                    nn.init.zeros_(deck_size_out.bias)
                    self.durable_deck_size_head = nn.Sequential(
                        deck_size_fc1,
                        nn.SiLU(),
                        deck_size_out,
                    )
                    self.durable_deck_size_linear = nn.Linear(DURABLE_DECK_SIZE_DIM, 1, bias=False)
                    nn.init.zeros_(self.durable_deck_size_linear.weight)
        if actor_critic.enable_hierarchical_branch_policy:
            branch_context_blocks = 3 + int(actor_critic.enable_direct_vitals_context)
            branch_dim = cfg.d_model * branch_context_blocks + cfg.recurrent_hidden_dim
            branch_out = nn.Linear(cfg.ffn_dim, 1)
            nn.init.zeros_(branch_out.weight)
            nn.init.zeros_(branch_out.bias)
            self.branch_policy_head = nn.Sequential(
                nn.LayerNorm(branch_dim),
                nn.Linear(branch_dim, cfg.ffn_dim),
                nn.SiLU(),
                branch_out,
            )

    def _persistent_token_mask(
        self,
        zone_ids: Tensor,
        owner_ids: Tensor,
        valid_mask: Tensor,
    ) -> Tensor:
        player_owned = owner_ids.eq(self._player_owner_id)
        build_mask = zone_ids.eq(RUN_ZONE_ID)
        for zone_id in PERSISTENT_BUILD_ZONE_IDS:
            if zone_id == RUN_ZONE_ID:
                continue
            build_mask = build_mask | (zone_ids.eq(zone_id) & player_owned)
        return build_mask & valid_mask.bool()

    def _pool_persistent_build(
        self,
        world: WorldEncoding,
        zone_ids: Tensor,
        owner_ids: Tensor,
        *,
        token_weights: Tensor | None = None,
    ) -> Tensor:
        batch_size = int(world.tokens.shape[0])
        build_mask = self._persistent_token_mask(zone_ids, owner_ids, world.world_mask)
        null_token = self.world_null_token.expand(batch_size, -1, -1)
        keys = torch.cat([null_token, world.tokens], dim=1)
        queries = self.persistent_build_queries.expand(batch_size, -1, -1)
        if token_weights is None:
            null_mask = torch.ones((batch_size, 1), dtype=torch.bool, device=world.tokens.device)
            key_mask = torch.cat([null_mask, build_mask], dim=1)
            delta, _ = self.persistent_build_attn(
                queries,
                keys,
                keys,
                key_padding_mask=~key_mask,
                need_weights=False,
            )
        else:
            eff_w = torch.where(
                build_mask,
                token_weights.to(dtype=world.tokens.dtype),
                torch.zeros_like(token_weights, dtype=world.tokens.dtype),
            )
            null_w = torch.ones((batch_size, 1), dtype=world.tokens.dtype, device=world.tokens.device)
            full_w = torch.cat([null_w, eff_w], dim=1)
            key_log_bias = torch.where(
                full_w > 0.0,
                torch.log(full_w.clamp_min(1e-12)),
                torch.full_like(full_w, -1e9),
            )
            delta, _ = self.persistent_build_attn(
                queries,
                keys,
                keys,
                key_padding_mask=key_log_bias,
                need_weights=False,
            )
        pooled: Tensor = self.persistent_build_norm(queries + delta).mean(dim=1)
        return pooled

    @staticmethod
    def _compact_world_by_mask(
        world: WorldTokenBatch,
        keep_mask: Tensor,
    ) -> tuple[WorldTokenBatch, Tensor]:
        batch_size, world_cap, feat_dim = world.features.shape
        if world_cap <= 1:
            idx0 = torch.zeros(
                (batch_size, world_cap),
                dtype=torch.long,
                device=world.features.device,
            )
            return world, idx0
        max_keep = max(1, int(keep_mask.long().sum(dim=1).max().item()))
        if max_keep >= world_cap:
            idx_id = (
                torch.arange(world_cap, device=world.features.device)
                .unsqueeze(0)
                .expand(batch_size, -1)
            )
            return (
                WorldTokenBatch(
                    features=world.features,
                    type_ids=world.type_ids,
                    role_ids=world.role_ids,
                    owner_ids=world.owner_ids,
                    entity_ids=world.entity_ids,
                    entity_aux_ids=world.entity_aux_ids,
                    definition_binding_ids=world.definition_binding_ids,
                    relation_binding_ids=world.relation_binding_ids,
                    zone_ids=world.zone_ids,
                    order_ids=world.order_ids,
                    mask=keep_mask,
                ),
                idx_id,
            )
        gather_idx = torch.argsort(
            (~keep_mask).to(dtype=torch.int8),
            dim=1,
            stable=True,
        )[:, :max_keep]
        feat_idx = gather_idx.unsqueeze(-1).expand(-1, -1, feat_dim)
        return (
            WorldTokenBatch(
                features=world.features.gather(1, feat_idx),
                type_ids=world.type_ids.gather(1, gather_idx),
                role_ids=world.role_ids.gather(1, gather_idx),
                owner_ids=world.owner_ids.gather(1, gather_idx),
                entity_ids=world.entity_ids.gather(1, gather_idx),
                entity_aux_ids=world.entity_aux_ids.gather(1, gather_idx),
                definition_binding_ids=world.definition_binding_ids.gather(1, gather_idx),
                relation_binding_ids=world.relation_binding_ids.gather(1, gather_idx),
                zone_ids=world.zone_ids.gather(1, gather_idx),
                order_ids=world.order_ids.gather(1, gather_idx),
                mask=keep_mask.gather(1, gather_idx),
            ),
            gather_idx,
        )

    @torch.no_grad()
    def _prepare_draw_weighted_persistent_world(
        self,
        world: WorldTokenBatch,
    ) -> _PreparedPersistentWorld:
        pers_mask_full = self._persistent_token_mask(world.zone_ids, world.owner_ids, world.mask)
        player_owned_full = world.owner_ids.eq(self._player_owner_id) & world.mask.bool()
        is_map_full = world.type_ids.eq(self._mapping_item_type_id)
        cand_label_seq_full = (
            player_owned_full
            & (~is_map_full)
            & (~world.zone_ids.eq(MODIFIER_ZONE_ID))
            & ((world.features[:, :, 2] - 0.25).abs() < 1e-4)
        )
        world, _ = self._compact_world_by_mask(world, pers_mask_full | cand_label_seq_full)

        dtype = world.features.dtype
        device = world.features.device
        batch_size, world_cap, _ = world.features.shape
        pers_mask = self._persistent_token_mask(world.zone_ids, world.owner_ids, world.mask)
        is_mapping_item = world.type_ids.eq(self._mapping_item_type_id)
        deck_card_mask = (
            pers_mask
            & world.zone_ids.eq(DECK_ZONE_ID)
            & is_mapping_item
            & ((world.features[:, :, 1] - 0.1875).abs() < 1e-4)
        )
        has_deck = deck_card_mask.any(dim=1)
        raw_qty_slot = world.features[:, :, DECK_QUANTITY_SLOT].clamp(0.0, 0.8)
        card_qty = torch.where(
            deck_card_mask,
            torch.expm1(10.0 * raw_qty_slot).round().clamp_min(1.0),
            torch.zeros((batch_size, world_cap), dtype=dtype, device=device),
        )
        deck_size = card_qty.sum(dim=1)
        draw_prob = card_qty / deck_size.unsqueeze(1).clamp_min(1.0)
        card_weight = DECK_REFERENCE_SIZE * draw_prob

        deck_seq_mask = (
            pers_mask
            & world.zone_ids.eq(DECK_ZONE_ID)
            & (~is_mapping_item)
            & ((world.features[:, :, 2] - 0.125).abs() < 1e-4)
        )
        deck_label_mask = (
            pers_mask
            & world.zone_ids.eq(DECK_ZONE_ID)
            & is_mapping_item
            & ((world.features[:, :, 1] - 0.3125).abs() < 1e-4)
        )

        rel_ids = world.relation_binding_ids
        valid_rel = rel_ids > 1
        mod_mask = pers_mask & world.zone_ids.eq(MODIFIER_ZONE_ID)
        match_mod = (
            (mod_mask & valid_rel).unsqueeze(2)
            & (deck_card_mask & valid_rel).unsqueeze(1)
            & rel_ids.unsqueeze(2).eq(rel_ids.unsqueeze(1))
        )
        mod_parent_count = match_mod.to(dtype=dtype).sum(dim=2)
        has_mod_parent = mod_parent_count > 0.0
        mod_weight = (
            torch.einsum("buv,bv->bu", match_mod.to(dtype=dtype), card_weight)
            / mod_parent_count.clamp_min(1.0)
        )
        mod_parent_entity_id = (
            torch.where(match_mod, world.entity_ids.unsqueeze(1).expand(-1, world_cap, -1), 0)
            .max(dim=2)
            .values
        )

        player_owned = world.owner_ids.eq(self._player_owner_id) & world.mask.bool()
        card_rel_match = (
            (player_owned & valid_rel).unsqueeze(2)
            & (deck_card_mask & valid_rel).unsqueeze(1)
            & rel_ids.unsqueeze(2).eq(rel_ids.unsqueeze(1))
        )
        has_card_rel = card_rel_match.any(dim=2)
        label_seq_mask = (
            player_owned
            & (~is_mapping_item)
            & (~world.zone_ids.eq(MODIFIER_ZONE_ID))
            & ((world.features[:, :, 2] - 0.25).abs() < 1e-4)
            & has_card_rel
        )
        seq_len = torch.where(
            label_seq_mask,
            (world.features[:, :, 1] * 64.0).round().long().clamp_min(0),
            torch.zeros((batch_size, world_cap), dtype=torch.long, device=device),
        )
        seq_end = seq_len.cumsum(dim=1)
        seq_start = seq_end - seq_len
        label_ord = deck_label_mask.long().cumsum(dim=1) - 1
        counts_exact = (seq_len.sum(dim=1) == deck_label_mask.long().sum(dim=1)) & has_deck
        in_seq_span = (
            counts_exact.view(batch_size, 1, 1)
            & deck_label_mask.unsqueeze(2)
            & label_seq_mask.unsqueeze(1)
            & (label_ord.unsqueeze(2) >= seq_start.unsqueeze(1))
            & (label_ord.unsqueeze(2) < seq_end.unsqueeze(1))
        )
        match_label = (
            torch.einsum(
                "bus,bsv->buv",
                in_seq_span.to(dtype=dtype),
                card_rel_match.to(dtype=dtype),
            )
            > 0.5
        )
        label_parent_count = match_label.to(dtype=dtype).sum(dim=2)
        has_label_parent = label_parent_count > 0.0
        label_weight_matched = (
            torch.einsum("buv,bv->bu", match_label.to(dtype=dtype), card_weight)
            / label_parent_count.clamp_min(1.0)
        )
        def_ids = world.definition_binding_ids
        same_def = (
            deck_label_mask.unsqueeze(2)
            & deck_label_mask.unsqueeze(1)
            & def_ids.unsqueeze(2).eq(def_ids.unsqueeze(1))
        )
        same_def_count = same_def.to(dtype=dtype).sum(dim=2).clamp_min(1.0)
        label_weight = torch.where(has_label_parent, label_weight_matched, 1.0 / same_def_count)

        kept_deck_zone = deck_card_mask | deck_seq_mask | deck_label_mask
        eff_mask = torch.where(
            has_deck.unsqueeze(1) & world.zone_ids.eq(DECK_ZONE_ID),
            kept_deck_zone,
            pers_mask,
        )
        token_weights = torch.where(
            eff_mask,
            torch.ones((batch_size, world_cap), dtype=dtype, device=device),
            torch.zeros((batch_size, world_cap), dtype=dtype, device=device),
        )
        active_deck_row = has_deck.unsqueeze(1)
        token_weights = torch.where(active_deck_row & deck_card_mask, card_weight, token_weights)
        token_weights = torch.where(active_deck_row & has_mod_parent, mod_weight, token_weights)
        token_weights = torch.where(active_deck_row & deck_label_mask, label_weight, token_weights)

        features = world.features.clone()
        seq_rows = active_deck_row & deck_seq_mask
        if self.actor_critic_config.enable_deconfounded_durable_value:
            seq_len_val: Tensor | float = DECK_REFERENCE_SIZE / 64.0
        else:
            seq_len_val = (deck_size.unsqueeze(1) / 64.0).clamp(0.0, 1.0)
        features[:, :, 1] = torch.where(
            seq_rows,
            seq_len_val,
            features[:, :, 1],
        )
        card_rows = active_deck_row & deck_card_mask
        label_rows = active_deck_row & deck_label_mask
        mod_item_mask = mod_mask & is_mapping_item
        mod_item_rows = active_deck_row & has_mod_parent & mod_item_mask
        for slot in (2, 3, 5, DECK_FLOOR_ADDED_SLOT, DECK_QUANTITY_SLOT):
            if slot < features.shape[-1]:
                features[:, :, slot] = torch.where(card_rows, 0.0, features[:, :, slot])
        features[:, :, 2] = torch.where(label_rows | mod_item_rows, 0.0, features[:, :, 2])

        canon_order_ids = torch.where(card_rows | label_rows | mod_item_rows, 0, world.order_ids)
        canon_role_ids = world.role_ids
        canon_entity_ids = world.entity_ids
        canon_aux_ids = torch.where(card_rows, world.entity_ids, world.entity_aux_ids)
        canon_aux_ids = torch.where(
            active_deck_row & has_mod_parent,
            mod_parent_entity_id,
            canon_aux_ids,
        )
        if self.actor_critic_config.enable_monotone_deconfounded_durable_value:
            player_pers_rows = eff_mask & world.zone_ids.eq(PLAYER_ZONE_ID)
            p_slot_end = min(int(features.shape[-1]), NUMERIC_FACT_SLOT_END)
            if p_slot_end > 3:
                features[:, :, 3:p_slot_end] = torch.where(
                    player_pers_rows.unsqueeze(-1),
                    0.0,
                    features[:, :, 3:p_slot_end],
                )
            run_pers_rows = eff_mask & world.zone_ids.eq(RUN_ZONE_ID)
            r_slot_end = min(int(features.shape[-1]), 198)
            saved_act = features[:, :, RUN_ACT_SLOT].clone() if RUN_ACT_SLOT < r_slot_end else None
            if r_slot_end > 3:
                features[:, :, 3:r_slot_end] = torch.where(
                    run_pers_rows.unsqueeze(-1),
                    0.0,
                    features[:, :, 3:r_slot_end],
                )
            if saved_act is not None:
                features[:, :, RUN_ACT_SLOT] = torch.where(
                    run_pers_rows,
                    saved_act,
                    features[:, :, RUN_ACT_SLOT],
                )
            canon_role_ids = torch.where(run_pers_rows, 1, canon_role_ids)
            canon_entity_ids = torch.where(run_pers_rows, 1, canon_entity_ids)
            canon_aux_ids = torch.where(run_pers_rows, 1, canon_aux_ids)
            canon_order_ids = torch.where(run_pers_rows, 0, canon_order_ids)

        child_uv = (match_mod & mod_item_mask.unsqueeze(2)) | match_label
        prepared_world = WorldTokenBatch(
            features=features,
            type_ids=world.type_ids,
            role_ids=canon_role_ids,
            owner_ids=world.owner_ids,
            entity_ids=canon_entity_ids,
            entity_aux_ids=canon_aux_ids,
            definition_binding_ids=world.definition_binding_ids,
            relation_binding_ids=world.relation_binding_ids,
            zone_ids=world.zone_ids,
            order_ids=canon_order_ids,
            mask=eff_mask,
        )
        comp_world, eff_idx = self._compact_world_by_mask(prepared_world, eff_mask)
        w_eff = int(eff_idx.shape[1])
        if w_eff < world_cap:
            token_weights = token_weights.gather(1, eff_idx)
            deck_card_mask = deck_card_mask.gather(1, eff_idx)
            draw_prob = draw_prob.gather(1, eff_idx)
            idx_u = eff_idx.unsqueeze(2).expand(-1, -1, world_cap)
            idx_v = eff_idx.unsqueeze(1).expand(-1, w_eff, -1)
            child_uv = child_uv.gather(1, idx_u).gather(2, idx_v)
        return _PreparedPersistentWorld(
            world=comp_world,
            token_weights=token_weights,
            deck_card_mask=deck_card_mask,
            draw_prob=draw_prob,
            deck_size=deck_size,
            has_deck=has_deck,
            child_uv=child_uv,
        )

    @staticmethod
    def _weighted_multihead_attention(
        mha: nn.MultiheadAttention,
        query: Tensor,
        key_value: Tensor,
        key_log_bias: Tensor,
    ) -> Tensor:
        batch_size, q_len, d_model = query.shape
        _, k_len, _ = key_value.shape
        n_heads = mha.num_heads
        head_dim = d_model // n_heads
        w_q, w_k, w_v = mha.in_proj_weight.chunk(3, dim=0)
        if mha.in_proj_bias is not None:
            b_q, b_k, b_v = mha.in_proj_bias.chunk(3, dim=0)
        else:
            b_q = b_k = b_v = None
        q = nn.functional.linear(query, w_q, b_q).view(batch_size, q_len, n_heads, head_dim).transpose(1, 2)
        k = nn.functional.linear(key_value, w_k, b_k).view(batch_size, k_len, n_heads, head_dim).transpose(1, 2)
        v = nn.functional.linear(key_value, w_v, b_v).view(batch_size, k_len, n_heads, head_dim).transpose(1, 2)
        attn_mask = key_log_bias.to(dtype=q.dtype)[:, None, None, :]
        dropout_p = float(mha.dropout) if mha.training else 0.0
        out = nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
        )
        merged = out.transpose(1, 2).contiguous().view(batch_size, q_len, d_model)
        return nn.functional.linear(merged, mha.out_proj.weight, mha.out_proj.bias)

    @classmethod
    def _weighted_world_encoder_forward(
        cls,
        encoder: nn.TransformerEncoder,
        tokens: Tensor,
        key_log_bias: Tensor,
    ) -> Tensor:
        h = tokens
        for mod in encoder.layers:
            if not isinstance(mod, nn.TransformerEncoderLayer):
                raise TypeError(f"Unsupported world_encoder layer type: {type(mod)!r}")
            if mod.norm_first:
                normed = mod.norm1(h)
                sa = cls._weighted_multihead_attention(mod.self_attn, normed, normed, key_log_bias)
                h = h + mod.dropout1(sa)
                ff = mod.linear2(mod.dropout(mod.activation(mod.linear1(mod.norm2(h)))))
                h = h + mod.dropout2(ff)
            else:
                sa = cls._weighted_multihead_attention(mod.self_attn, h, h, key_log_bias)
                h = mod.norm1(h + mod.dropout1(sa))
                ff = mod.linear2(mod.dropout(mod.activation(mod.linear1(h))))
                h = mod.norm2(h + mod.dropout2(ff))
        if encoder.norm is not None:
            h = encoder.norm(h)
        return h

    def _encode_world_with_weights(
        self,
        world: WorldTokenBatch,
        domain_ids: Tensor,
        token_weights: Tensor,
        *,
        _validated: bool = False,
    ) -> tuple[WorldEncoding, Tensor]:
        if _validated:
            batch_size = int(world.features.shape[0])
        else:
            batch_size, _ = world.validate(self.config)
        cfg = self.config
        domain_ids = domain_ids.long().clamp(min=0, max=cfg.domain_count - 1)
        world_mask = world.mask.bool() & (token_weights > 0.0)
        raw_world_tokens = self.token_embedder(
            world.features,
            type_ids=world.type_ids,
            role_ids=world.role_ids,
            owner_ids=world.owner_ids,
            entity_ids=world.entity_ids,
            entity_aux_ids=world.entity_aux_ids,
            zone_ids=world.zone_ids,
            order_ids=world.order_ids,
            segment_id=0,
        )
        world_tokens = raw_world_tokens * world_mask.unsqueeze(-1).to(dtype=raw_world_tokens.dtype)

        null_token = self.world_null_token.expand(batch_size, -1, -1)
        encoded_tokens = torch.cat([null_token, world_tokens], dim=1)
        null_weight = torch.ones(
            (batch_size, 1),
            dtype=raw_world_tokens.dtype,
            device=raw_world_tokens.device,
        )
        full_weights = torch.cat(
            [null_weight, token_weights.to(dtype=raw_world_tokens.dtype)],
            dim=1,
        )
        neg_mask_val = -1e4 if raw_world_tokens.dtype == torch.float16 else -1e9
        key_log_bias = torch.where(
            full_weights > 0.0,
            torch.log(full_weights.clamp_min(1e-12)),
            torch.full_like(full_weights, neg_mask_val),
        )
        encoded_tokens = self._weighted_world_encoder_forward(
            self.world_encoder,
            encoded_tokens,
            key_log_bias,
        )
        relation_tokens = encoded_tokens[:, 1:] * world_mask.unsqueeze(-1).to(
            dtype=encoded_tokens.dtype
        )

        domain_context = self.domain_embedding(domain_ids).unsqueeze(1)
        latent_queries = self.latent_queries.expand(batch_size, -1, -1) + domain_context
        latent_delta = self._weighted_multihead_attention(
            self.world_to_latent,
            latent_queries,
            encoded_tokens,
            key_log_bias,
        )
        latents = self.world_to_latent_norm(latent_queries + latent_delta)
        latents = self.latent_encoder(latents)
        state_embedding = self.state_norm(latents.mean(dim=1))
        return (
            WorldEncoding(
                latents=latents,
                state_embedding=state_embedding,
                tokens=relation_tokens,
                world_mask=world_mask,
                entity_ids=world.entity_ids,
                entity_aux_ids=world.entity_aux_ids,
                definition_binding_ids=world.definition_binding_ids,
                relation_binding_ids=world.relation_binding_ids,
            ),
            world_tokens,
        )

    def _extract_scaled_vitals(
        self,
        world: WorldTokenBatch,
    ) -> Tensor:
        pers_valid = self._persistent_token_mask(world.zone_ids, world.owner_ids, world.mask)
        slot_end = min(int(world.features.shape[-1]), NUMERIC_FACT_SLOT_END)
        feats = world.features[:, :, NUMERIC_FACT_SLOT_START:slot_end]
        dtype = feats.dtype
        player_mask = (world.zone_ids.eq(PLAYER_ZONE_ID) & pers_valid).to(dtype=dtype)
        run_mask = (world.zone_ids.eq(RUN_ZONE_ID) & pers_valid).to(dtype=dtype)
        player_feats = torch.einsum("bw,bwf->bf", player_mask, feats) / player_mask.sum(
            dim=1, keepdim=True
        ).clamp_min(1.0)
        run_feats = torch.einsum("bw,bwf->bf", run_mask, feats) / run_mask.sum(
            dim=1, keepdim=True
        ).clamp_min(1.0)
        # Scale log-compressed numeric slots (0.1 * log1p(x)) by 5.0 into O(1) dynamic range
        # while softly bounding unit boolean indicators via tanh.
        scaled_player = torch.where(player_feats.abs() >= 0.99, player_feats * 0.5, player_feats * 8.0)
        scaled_run = torch.where(run_feats.abs() >= 0.99, run_feats * 0.5, run_feats * 8.0)
        pers_only = self.actor_critic_config.enable_persistent_only_durable_value
        batch_size = int(world.features.shape[0])
        zero_col = torch.zeros((batch_size,), dtype=dtype, device=feats.device)
        zone_counts = torch.stack(
            [
                zero_col
                if (pers_only and zid in IDENTITY_BLIND_CARD_COUNT_ZONE_IDS)
                else 0.8
                * torch.log1p(
                    (world.zone_ids.eq(zid) & pers_valid).to(dtype=dtype).sum(dim=1)
                )
                for zid in PERSISTENT_BUILD_ZONE_IDS
            ],
            dim=-1,
        )
        if not pers_only:
            return torch.cat([scaled_player, scaled_run, zone_counts], dim=-1)
        hp_log = torch.maximum(
            player_feats[:, PLAYER_HP_OFFSET],
            player_feats[:, PLAYER_CURRENT_HP_OFFSET],
        )
        hp_val = torch.expm1((10.0 * hp_log).clamp(0.0, 8.0))
        max_hp_val = torch.expm1(
            (10.0 * player_feats[:, PLAYER_MAX_HP_OFFSET]).clamp(0.0, 8.0)
        ).clamp_min(1.0)
        hp_ratio = (hp_val / max_hp_val).clamp(0.0, 2.0)
        missing_hp_ratio = (1.0 - hp_ratio).clamp(0.0, 1.0)
        low_hp_crisis = (0.5 - hp_ratio).clamp_min(0.0) * 2.0
        gold_per_100 = (
            torch.expm1((10.0 * player_feats[:, PLAYER_GOLD_OFFSET]).clamp(0.0, 9.0)) / 100.0
        ).clamp(0.0, 10.0)
        salient = torch.stack(
            [
                4.0 * hp_ratio,
                4.0 * missing_hp_ratio,
                4.0 * low_hp_crisis,
                2.0 * gold_per_100,
            ],
            dim=-1,
        )
        return torch.cat([scaled_player, scaled_run, zone_counts, salient], dim=-1)

    def _mask_world_to_persistent(self, world: WorldTokenBatch) -> WorldTokenBatch:
        persistent_mask = self._persistent_token_mask(world.zone_ids, world.owner_ids, world.mask)
        return WorldTokenBatch(
            features=world.features,
            type_ids=world.type_ids,
            role_ids=world.role_ids,
            owner_ids=world.owner_ids,
            entity_ids=world.entity_ids,
            entity_aux_ids=world.entity_aux_ids,
            definition_binding_ids=world.definition_binding_ids,
            relation_binding_ids=world.relation_binding_ids,
            zone_ids=world.zone_ids,
            order_ids=world.order_ids,
            mask=persistent_mask,
        )

    def _normalize_critic_revival_context(
        self,
        critic_revival_context: Tensor | None,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor | None:
        if not self.actor_critic_config.enable_asymmetric_revival_critic:
            return None
        if critic_revival_context is None:
            return torch.zeros(
                (batch_size, CRITIC_REVIVAL_CONTEXT_DIM),
                device=device,
                dtype=dtype,
            )
        if tuple(critic_revival_context.shape) != (batch_size, CRITIC_REVIVAL_CONTEXT_DIM):
            raise ValueError(
                "critic_revival_context must have shape "
                f"{(batch_size, CRITIC_REVIVAL_CONTEXT_DIM)}, got {tuple(critic_revival_context.shape)}"
            )
        if not torch.is_floating_point(critic_revival_context) or not bool(
            torch.isfinite(critic_revival_context).all().item()
        ):
            raise ValueError("critic_revival_context must be finite floating point")
        return critic_revival_context.to(device=device, dtype=dtype)

    def _encode_direct_vitals(
        self,
        world: WorldTokenBatch,
    ) -> tuple[Tensor, Tensor]:
        raw_vitals = self._extract_scaled_vitals(world)
        vitals_emb: Tensor = self.persistent_vitals_encoder(raw_vitals)
        return vitals_emb, raw_vitals

    def _encode_canonical_neutral_vitals(
        self,
        world: WorldTokenBatch,
    ) -> tuple[Tensor, Tensor]:
        pers_valid = self._persistent_token_mask(world.zone_ids, world.owner_ids, world.mask)
        slot_end = min(int(world.features.shape[-1]), NUMERIC_FACT_SLOT_END)
        feats = world.features[:, :, NUMERIC_FACT_SLOT_START:slot_end]
        dtype = feats.dtype
        device = feats.device
        batch_size, _, num_width = feats.shape

        scaled_player = torch.zeros((batch_size, num_width), dtype=dtype, device=device)
        for offset, canon_val in (
            (PLAYER_CURRENT_HP_OFFSET, 60.0),
            (PLAYER_HP_OFFSET, 60.0),
            (PLAYER_MAX_HP_OFFSET, 80.0),
            (PLAYER_GOLD_OFFSET, 100.0),
            (PLAYER_OPEN_POTION_SLOTS_OFFSET, 2.0),
        ):
            if 0 <= offset < num_width:
                scaled_player[:, offset] = 0.8 * math.log1p(canon_val)

        run_mask = (world.zone_ids.eq(RUN_ZONE_ID) & pers_valid).to(dtype=dtype)
        run_feats = torch.einsum("bw,bwf->bf", run_mask, feats) / run_mask.sum(
            dim=1, keepdim=True
        ).clamp_min(1.0)
        scaled_run = torch.zeros((batch_size, num_width), dtype=dtype, device=device)
        if 0 <= RUN_ACT_OFFSET < num_width:
            act_feat = run_feats[:, RUN_ACT_OFFSET]
            scaled_run[:, RUN_ACT_OFFSET] = torch.where(
                act_feat.abs() >= 0.99, act_feat * 0.5, act_feat * 8.0
            )

        pers_only = self.actor_critic_config.enable_persistent_only_durable_value
        zero_col = torch.zeros((batch_size,), dtype=dtype, device=device)
        zone_counts = torch.stack(
            [
                zero_col
                if (pers_only and zid in IDENTITY_BLIND_CARD_COUNT_ZONE_IDS)
                else 0.8
                * torch.log1p(
                    (world.zone_ids.eq(zid) & pers_valid).to(dtype=dtype).sum(dim=1)
                )
                for zid in PERSISTENT_BUILD_ZONE_IDS
            ],
            dim=-1,
        )
        if not pers_only:
            raw_vitals = torch.cat([scaled_player, scaled_run, zone_counts], dim=-1)
        else:
            salient = torch.tensor(
                [3.0, 1.0, 0.0, 2.0],
                dtype=dtype,
                device=device,
            ).unsqueeze(0).expand(batch_size, -1)
            raw_vitals = torch.cat([scaled_player, scaled_run, zone_counts, salient], dim=-1)
        vitals_emb: Tensor = self.persistent_vitals_encoder(raw_vitals)
        return vitals_emb, raw_vitals

    def _extract_progress_vitals(
        self,
        world: WorldTokenBatch,
    ) -> Tensor:
        pers_valid = self._persistent_token_mask(world.zone_ids, world.owner_ids, world.mask)
        slot_end = min(int(world.features.shape[-1]), NUMERIC_FACT_SLOT_END)
        feats = world.features[:, :, NUMERIC_FACT_SLOT_START:slot_end]
        dtype = feats.dtype
        player_valid = (world.zone_ids.eq(PLAYER_ZONE_ID) & pers_valid).unsqueeze(-1)
        run_valid = (world.zone_ids.eq(RUN_ZONE_ID) & pers_valid).unsqueeze(-1)
        player_max = torch.where(player_valid, feats, 0.0).amax(dim=1)
        run_max = torch.where(run_valid, feats, 0.0).amax(dim=1)

        hp_log = torch.maximum(
            player_max[:, PLAYER_HP_OFFSET],
            player_max[:, PLAYER_CURRENT_HP_OFFSET],
        )
        hp_val = torch.expm1((10.0 * hp_log).clamp(0.0, 8.0))
        max_hp_val = torch.expm1((10.0 * player_max[:, PLAYER_MAX_HP_OFFSET]).clamp(0.0, 8.0)).clamp_min(1.0)
        hp_ratio = (hp_val / max_hp_val).clamp(0.0, 2.0)
        missing_hp_ratio = (1.0 - hp_ratio).clamp(0.0, 1.0)
        low_hp_crisis = (0.5 - hp_ratio).clamp_min(0.0) * 2.0
        gold_per_100 = (
            torch.expm1((10.0 * player_max[:, PLAYER_GOLD_OFFSET]).clamp(0.0, 9.0)) / 100.0
        ).clamp(0.0, 10.0)
        open_potions = (
            torch.expm1((10.0 * player_max[:, PLAYER_OPEN_POTION_SLOTS_OFFSET]).clamp(0.0, 4.0)) / 3.0
        ).clamp(0.0, 2.0)

        floor_val = torch.expm1((10.0 * run_max[:, RUN_FLOOR_OFFSET]).clamp(0.0, 5.0))
        floor_norm = (floor_val / 50.0).clamp(0.0, 1.5)
        act_val = torch.expm1((10.0 * run_max[:, RUN_ACT_OFFSET]).clamp(0.0, 3.0))
        act_norm = (act_val / 3.0).clamp(0.0, 1.5)
        is_act1 = (1.0 - (act_val - 1.0).abs()).clamp(0.0, 1.0)
        is_act2 = (1.0 - (act_val - 2.0).abs()).clamp(0.0, 1.0)
        is_act3 = (1.0 - (act_val - 3.0).abs()).clamp(0.0, 1.0)

        return torch.stack(
            [
                hp_ratio,
                missing_hp_ratio,
                low_hp_crisis,
                gold_per_100,
                open_potions,
                floor_norm,
                act_norm,
                is_act1,
                is_act2,
                is_act3,
            ],
            dim=-1,
        ).to(dtype=dtype)

    def _extract_monotone_stage_and_vitals(
        self,
        world: WorldTokenBatch,
    ) -> tuple[Tensor, Tensor, Tensor]:
        pers_valid = self._persistent_token_mask(world.zone_ids, world.owner_ids, world.mask)
        slot_end = min(int(world.features.shape[-1]), NUMERIC_FACT_SLOT_END)
        feats = world.features[:, :, NUMERIC_FACT_SLOT_START:slot_end]
        dtype = feats.dtype
        player_valid = (world.zone_ids.eq(PLAYER_ZONE_ID) & pers_valid).unsqueeze(-1)
        run_valid = (world.zone_ids.eq(RUN_ZONE_ID) & pers_valid).unsqueeze(-1)
        player_max = torch.where(player_valid, feats, 0.0).amax(dim=1)
        run_max = torch.where(run_valid, feats, 0.0).amax(dim=1)

        hp_log = torch.maximum(
            player_max[:, PLAYER_HP_OFFSET],
            player_max[:, PLAYER_CURRENT_HP_OFFSET],
        )
        hp_val = torch.expm1((10.0 * hp_log).clamp(0.0, 8.0)).clamp_min(0.0)
        max_hp_val = torch.expm1(
            (10.0 * player_max[:, PLAYER_MAX_HP_OFFSET]).clamp(0.0, 8.0)
        ).clamp_min(1.0)
        hp_ratio = (hp_val / max_hp_val).clamp(0.0, 2.0)
        r_c = hp_ratio.clamp(0.0, 1.0)
        hp_basis = torch.stack(
            [
                1.0 - (1.0 - r_c).pow(3),
                1.0 - (1.0 - r_c).pow(2),
                torch.tanh(hp_val / 35.0),
                torch.tanh(hp_val / 75.0),
                torch.tanh(max_hp_val / 100.0),
            ],
            dim=-1,
        ).to(dtype=dtype)

        gold_per_100 = (
            torch.expm1((10.0 * player_max[:, PLAYER_GOLD_OFFSET]).clamp(0.0, 9.0)) / 100.0
        ).clamp(0.0, 10.0)
        liquid_basis = torch.stack(
            [
                torch.tanh(gold_per_100 / 0.75),
                torch.tanh(gold_per_100 / 1.50),
                torch.tanh(gold_per_100 / 3.00),
                0.5 * torch.log1p(gold_per_100),
            ],
            dim=-1,
        ).to(dtype=dtype)

        floor_val = torch.expm1((10.0 * run_max[:, RUN_FLOOR_OFFSET]).clamp(0.0, 5.0))
        floor_norm = (floor_val / 50.0).clamp(0.0, 1.5)
        act_val = torch.expm1((10.0 * run_max[:, RUN_ACT_OFFSET]).clamp(0.0, 3.0))
        act_norm = (act_val / 3.0).clamp(0.0, 1.5)
        is_act1 = (1.0 - (act_val - 1.0).abs()).clamp(0.0, 1.0)
        is_act2 = (1.0 - (act_val - 2.0).abs()).clamp(0.0, 1.0)
        is_act3 = (1.0 - (act_val - 3.0).abs()).clamp(0.0, 1.0)
        stage_feats = torch.stack(
            [
                floor_norm,
                act_norm,
                is_act1,
                is_act2,
                is_act3,
            ],
            dim=-1,
        ).to(dtype=dtype)
        return stage_feats, hp_basis, liquid_basis

    def forward_durable_stage_baseline(
        self,
        world: WorldTokenBatch,
        *,
        critic_revival_context: Tensor | None = None,
        validate: bool = False,
    ) -> Tensor:
        if not self.actor_critic_config.enable_monotone_deconfounded_durable_value:
            raise RuntimeError(
                "forward_durable_stage_baseline requires enable_monotone_deconfounded_durable_value=True"
            )
        if validate:
            batch_size, _ = world.validate(self.config)
        else:
            batch_size = int(world.features.shape[0])
        rev_ctx = self._normalize_critic_revival_context(
            critic_revival_context,
            batch_size=batch_size,
            device=world.features.device,
            dtype=world.features.dtype,
        )
        stage_feats, _, _ = self._extract_monotone_stage_and_vitals(world)
        stage_in = (
            torch.cat([stage_feats, rev_ctx * 4.0], dim=-1)
            if rev_ctx is not None
            else stage_feats
        )
        stage_val: Tensor = self.durable_stage_baseline_head(stage_in).squeeze(-1)
        return stage_val

    @staticmethod
    def _scale_card_and_modifier_features(features: Tensor) -> Tensor:
        scaled = features.clone()
        slot_end = min(int(scaled.shape[-1]), NUMERIC_FACT_SLOT_END)
        if slot_end > NUMERIC_FACT_SLOT_START:
            num_block = scaled[:, :, NUMERIC_FACT_SLOT_START:slot_end]
            scaled[:, :, NUMERIC_FACT_SLOT_START:slot_end] = torch.where(
                num_block.abs() >= 0.99,
                num_block,
                num_block * 8.0,
            )
        if scaled.shape[-1] > 198:
            mod_block = scaled[:, :, 198:]
            scaled[:, :, 198:] = torch.where(
                mod_block.abs() >= 0.99,
                mod_block,
                mod_block * 8.0,
            )
        return scaled

    def _durable_readout(
        self,
        *,
        world_state_embedding: Tensor,
        build_embedding: Tensor | None,
        vitals_embedding: Tensor | None = None,
        raw_vitals: Tensor | None = None,
        progress_vitals: Tensor | None = None,
        monotone_parts: tuple[Tensor, Tensor, Tensor] | None = None,
        run_memory: Tensor,
        revival_context: Tensor | None = None,
        card_repr: Tensor | None = None,
        card_raw_feats: Tensor | None = None,
        deck_card_mask: Tensor | None = None,
        draw_prob: Tensor | None = None,
        deck_size: Tensor | None = None,
        has_deck: Tensor | None = None,
    ) -> Tensor:
        parts = [world_state_embedding]
        if build_embedding is not None:
            parts.append(build_embedding)
        if vitals_embedding is not None:
            parts.append(vitals_embedding)
        parts.append(run_memory)
        rev_scaled = revival_context * 4.0 if revival_context is not None else None
        if rev_scaled is not None:
            parts.append(rev_scaled)
        out: Tensor = self.durable_value_head(torch.cat(parts, dim=-1)).squeeze(-1)
        stage_in: Tensor | None = None
        if self.actor_critic_config.enable_monotone_deconfounded_durable_value:
            if monotone_parts is not None:
                stage_feats, hp_basis, liquid_basis = monotone_parts
                stage_in = (
                    torch.cat([stage_feats, rev_scaled], dim=-1)
                    if rev_scaled is not None
                    else stage_feats
                )
                v_stage = self.durable_stage_baseline_head(stage_in).squeeze(-1)
                w_hp = nn.functional.softplus(self.durable_hp_gate(stage_in))
                w_liq = nn.functional.softplus(self.durable_liquid_gate(stage_in))
                stage_term = (
                    v_stage
                    if self.actor_critic_config.enable_balanced_deconfounded_durable_value
                    else v_stage.detach()
                )
                out = (
                    out
                    + stage_term
                    + (w_hp * hp_basis).sum(dim=-1)
                    + (w_liq * liquid_basis).sum(dim=-1)
                )
        elif self.actor_critic_config.enable_deconfounded_durable_value:
            if progress_vitals is not None:
                prog_in = (
                    torch.cat([progress_vitals, rev_scaled], dim=-1)
                    if rev_scaled is not None
                    else progress_vitals
                )
                out = out + self.durable_progress_vitals_head(prog_in).squeeze(-1)
        elif raw_vitals is not None:
            vitals_in = (
                torch.cat([raw_vitals, rev_scaled], dim=-1)
                if rev_scaled is not None
                else raw_vitals
            )
            out = out + self.durable_vitals_head(vitals_in).squeeze(-1)
            if self.actor_critic_config.enable_persistent_only_durable_value:
                out = out + self.durable_vitals_linear(
                    raw_vitals[:, -SALIENT_VITALS_DIM:]
                ).squeeze(-1)
        if (
            self.actor_critic_config.enable_draw_weighted_durable_value
            and card_repr is not None
            and card_raw_feats is not None
            and draw_prob is not None
            and deck_size is not None
            and has_deck is not None
        ):
            deck_ctx_parts = [world_state_embedding]
            if vitals_embedding is not None:
                deck_ctx_parts.append(vitals_embedding)
            deck_ctx_parts.append(run_memory)
            if rev_scaled is not None:
                deck_ctx_parts.append(rev_scaled)
            deck_ctx = (
                0.5
                * self.durable_deck_ctx_fc(
                    self.durable_deck_ctx_norm(torch.cat(deck_ctx_parts, dim=-1))
                ).unsqueeze(1)
            )
            card_hidden = nn.functional.silu(
                self.durable_deck_card_fc(self.durable_deck_norm(card_repr)) + deck_ctx
            )
            raw_hidden = nn.functional.silu(self.durable_deck_raw_fc(card_raw_feats))
            bilinear_card = torch.cat(
                [card_hidden, raw_hidden, card_hidden * raw_hidden],
                dim=-1,
            )
            card_utilities = self.durable_deck_out(bilinear_card).squeeze(-1)
            expected_card_utility = torch.einsum("bw,bw->b", draw_prob, card_utilities)
            if self.actor_critic_config.enable_monotone_deconfounded_durable_value:
                has_deck_f = has_deck.to(dtype=out.dtype)
                eff_card_mask = (
                    deck_card_mask
                    if deck_card_mask is not None
                    else (draw_prob > 0.0)
                )
                u_tilt_all = 2.5 * torch.tanh(
                    (card_utilities.unsqueeze(-1) + self.durable_deck_tilted_out(bilinear_card))
                    / 2.5
                )
                u_beta_cols: list[Tensor] = []
                for m_idx, beta_m in enumerate(DURABLE_TILT_BETAS):
                    u_m = u_tilt_all[:, :, m_idx]
                    u_masked = u_m.masked_fill(~eff_card_mask, -1e9)
                    u_max = torch.where(
                        has_deck,
                        u_masked.amax(dim=1),
                        torch.zeros_like(expected_card_utility),
                    )
                    shifted_exp = torch.where(
                        eff_card_mask,
                        torch.exp((beta_m * (u_m - u_max.unsqueeze(1))).clamp(-60.0, 0.0)),
                        torch.zeros_like(u_m),
                    )
                    log_mean_exp = u_max + (
                        torch.log(
                            torch.einsum("bw,bw->b", draw_prob, shifted_exp).clamp_min(1e-12)
                        )
                        / beta_m
                    )
                    u_beta_cols.append(log_mean_exp * has_deck_f)
                u_betas = torch.stack(u_beta_cols, dim=-1)
                alpha_lin = (
                    nn.functional.softplus(self.durable_deck_linear_logit)
                    if self.actor_critic_config.enable_balanced_deconfounded_durable_value
                    else torch.sigmoid(self.durable_deck_linear_logit)
                )
                w_tilt = nn.functional.softplus(self.durable_deck_tilt_logits)
                deck_utility = alpha_lin * expected_card_utility + (w_tilt * u_betas).sum(dim=-1)
                if stage_in is not None:
                    safe_n = deck_size.clamp_min(1.0)
                    bloat_basis = (
                        torch.stack(
                            [
                                nn.functional.softplus((safe_n - 16.0) / 6.0),
                                nn.functional.softplus((safe_n - 24.0) / 6.0),
                                nn.functional.softplus((safe_n - 32.0) / 6.0),
                            ],
                            dim=-1,
                        )
                        * has_deck_f.unsqueeze(-1)
                    )
                    w_cycle = nn.functional.softplus(self.durable_cycle_gate(stage_in))
                    v_cycle = -(w_cycle * bloat_basis).sum(dim=-1)
                    deck_utility = deck_utility + v_cycle
                out = out + deck_utility
            elif self.actor_critic_config.enable_deconfounded_durable_value:
                out = out + expected_card_utility
            else:
                has_deck_f = has_deck.to(dtype=out.dtype)
                safe_n = deck_size.clamp_min(1.0)
                deck_size_vec = (
                    torch.stack(
                        [
                            ((safe_n - DECK_REFERENCE_SIZE) / 10.0).clamp(-1.2, 5.0),
                            (safe_n - 20.0).clamp_min(0.0) / 10.0,
                            (DECK_REFERENCE_SIZE / safe_n).clamp(0.0, 4.0) - 1.0,
                            torch.log1p(safe_n) - math.log1p(DECK_REFERENCE_SIZE),
                        ],
                        dim=-1,
                    )
                    * has_deck_f.unsqueeze(-1)
                )
                deck_size_in = (
                    torch.cat([deck_size_vec, rev_scaled], dim=-1)
                    if rev_scaled is not None
                    else deck_size_vec
                )
                deck_size_delta = (
                    self.durable_deck_size_head(deck_size_in).squeeze(-1)
                    + self.durable_deck_size_linear(deck_size_vec).squeeze(-1)
                ) * has_deck_f
                out = out + expected_card_utility + deck_size_delta
        if rev_scaled is not None:
            out = (
                out
                + self.durable_revival_linear(rev_scaled).squeeze(-1)
                + self.durable_revival_head(rev_scaled).squeeze(-1)
            )
        return out

    def _extract_previous_run_memory(
        self,
        recurrent_state: Tensor | None,
        *,
        batch_size: int,
        reference: Tensor,
        validate: bool,
    ) -> Tensor:
        memory_half = self.config.recurrent_hidden_dim // 2
        if recurrent_state is None:
            return self.initial_state(
                batch_size,
                device=reference.device,
            )[:, :memory_half]
        if recurrent_state.shape[-1] == memory_half:
            return recurrent_state
        if validate:
            self._validate_recurrent_state(
                recurrent_state,
                batch_size=batch_size,
                reference=reference,
            )
        return recurrent_state[:, :memory_half]

    def _compute_persistent_durable_value(
        self,
        world: WorldTokenBatch,
        recurrent_state: Tensor | None,
        *,
        critic_revival_context: Tensor | None,
        validate: bool,
    ) -> tuple[Tensor, Tensor]:
        batch_size = int(world.features.shape[0])
        previous_run = self._extract_previous_run_memory(
            recurrent_state,
            batch_size=batch_size,
            reference=world.features,
            validate=validate,
        )
        rev_ctx = self._normalize_critic_revival_context(
            critic_revival_context,
            batch_size=batch_size,
            device=world.features.device,
            dtype=world.features.dtype,
        )
        zero_domains = torch.zeros(
            (batch_size,),
            dtype=torch.long,
            device=world.features.device,
        )
        monotone_parts: tuple[Tensor, Tensor, Tensor] | None = None
        if self.actor_critic_config.enable_monotone_deconfounded_durable_value:
            monotone_parts = self._extract_monotone_stage_and_vitals(world)
        card_repr: Tensor | None = None
        card_raw_feats: Tensor | None = None
        deck_card_mask: Tensor | None = None
        draw_prob: Tensor | None = None
        deck_size: Tensor | None = None
        has_deck: Tensor | None = None
        token_weights: Tensor | None = None
        if self.actor_critic_config.enable_draw_weighted_durable_value:
            prep = self._prepare_draw_weighted_persistent_world(world)
            pers_world = prep.world
            token_weights = prep.token_weights
            deck_card_mask = prep.deck_card_mask
            draw_prob = prep.draw_prob
            deck_size = prep.deck_size
            has_deck = prep.has_deck
            pers_enc, raw_world_tokens = self._encode_world_with_weights(
                pers_world,
                zero_domains,
                token_weights,
                _validated=not validate,
            )
            child_uv_f = prep.child_uv.to(dtype=pers_enc.tokens.dtype)
            per_card_child_w = child_uv_f / child_uv_f.sum(dim=2, keepdim=True).clamp_min(1.0)
            child_tok_sum = torch.einsum("buv,bud->bvd", per_card_child_w, pers_enc.tokens)
            scaled_feats = self._scale_card_and_modifier_features(
                pers_world.features.to(dtype=pers_enc.tokens.dtype)
            )
            child_feat_sum = torch.einsum("buv,buf->bvf", per_card_child_w, scaled_feats)
            card_mask_f = prep.deck_card_mask.unsqueeze(-1).to(dtype=pers_enc.tokens.dtype)
            card_repr = (
                torch.cat([pers_enc.tokens, raw_world_tokens, child_tok_sum], dim=-1)
                * card_mask_f
            )
            card_raw_feats = (
                torch.cat([scaled_feats, child_feat_sum], dim=-1)
                * card_mask_f
            )
        else:
            pers_world = self._mask_world_to_persistent(world)
            pers_enc = self.encode_world(pers_world, zero_domains, _validated=not validate)
        run_prev_for_durable = (
            torch.zeros_like(previous_run)
            if self.actor_critic_config.enable_monotone_deconfounded_durable_value
            else previous_run
        )
        pers_run = self.run_recurrent_norm(
            self.run_recurrent_cell(pers_enc.state_embedding, run_prev_for_durable)
        )
        build_emb = (
            self._pool_persistent_build(
                pers_enc,
                pers_world.zone_ids,
                pers_world.owner_ids,
                token_weights=token_weights,
            )
            if self.actor_critic_config.enable_persistent_build_context
            else None
        )
        if (
            build_emb is not None
            and card_repr is not None
            and draw_prob is not None
            and self.actor_critic_config.enable_draw_weighted_durable_value
        ):
            build_emb = build_emb + torch.einsum(
                "bw,bwd->bd",
                draw_prob.to(dtype=build_emb.dtype),
                self.persistent_build_deck_proj(card_repr),
            )
        vitals_emb: Tensor | None = None
        raw_vitals: Tensor | None = None
        progress_vitals: Tensor | None = None
        if self.actor_critic_config.enable_direct_vitals_context:
            if self.actor_critic_config.enable_balanced_deconfounded_durable_value:
                vitals_emb, raw_vitals = self._encode_canonical_neutral_vitals(pers_world)
            else:
                vitals_emb, raw_vitals = self._encode_direct_vitals(pers_world)
            if (
                self.actor_critic_config.enable_deconfounded_durable_value
                and not self.actor_critic_config.enable_monotone_deconfounded_durable_value
            ):
                progress_vitals = self._extract_progress_vitals(pers_world)
        durable_val = self._durable_readout(
            world_state_embedding=pers_enc.state_embedding,
            build_embedding=build_emb,
            vitals_embedding=vitals_emb,
            raw_vitals=raw_vitals,
            progress_vitals=progress_vitals,
            monotone_parts=monotone_parts,
            run_memory=pers_run,
            revival_context=rev_ctx,
            card_repr=card_repr,
            card_raw_feats=card_raw_feats,
            deck_card_mask=deck_card_mask,
            draw_prob=draw_prob,
            deck_size=deck_size,
            has_deck=has_deck,
        )
        return durable_val, pers_run

    def forward_durable_value(
        self,
        world: WorldTokenBatch,
        domain_ids: Tensor,
        recurrent_state: Tensor | None = None,
        *,
        critic_revival_context: Tensor | None = None,
        validate: bool = True,
    ) -> tuple[Tensor, Tensor]:
        """Evaluate candidate-independent durable value V_durable(W, h_run).

        Consumes world tokens and slow run memory exclusively; never touches
        candidate tokens, combat_stream, macro_stream, or joint_stream.
        Returns ``(durable_value [B], next_run_memory [B, H/2])``.
        """

        if not self.actor_critic_config.enable_durable_value_head:
            raise RuntimeError("forward_durable_value requires enable_durable_value_head=True")
        if validate:
            batch_size, _ = world.validate(self.config)
        else:
            batch_size = int(world.features.shape[0])
        if self.actor_critic_config.enable_persistent_only_durable_value:
            return self._compute_persistent_durable_value(
                world,
                recurrent_state,
                critic_revival_context=critic_revival_context,
                validate=validate,
            )

        previous_run = self._extract_previous_run_memory(
            recurrent_state,
            batch_size=batch_size,
            reference=world.features,
            validate=validate,
        )
        rev_ctx = self._normalize_critic_revival_context(
            critic_revival_context,
            batch_size=batch_size,
            device=world.features.device,
            dtype=world.features.dtype,
        )
        world_enc = self.encode_world(world, domain_ids, _validated=not validate)
        combat_domain = domain_ids.eq(COMBAT_DOMAIN_ID).unsqueeze(-1)
        run_update = self.run_recurrent_norm(
            self.run_recurrent_cell(world_enc.state_embedding, previous_run)
        )
        next_run = torch.where(combat_domain, previous_run, run_update)
        build_emb = (
            self._pool_persistent_build(world_enc, world.zone_ids, world.owner_ids)
            if self.actor_critic_config.enable_persistent_build_context
            else None
        )
        vitals_emb: Tensor | None = None
        raw_vitals: Tensor | None = None
        if self.actor_critic_config.enable_direct_vitals_context:
            vitals_emb, raw_vitals = self._encode_direct_vitals(world)
        durable_val = self._durable_readout(
            world_state_embedding=world_enc.state_embedding,
            build_embedding=build_emb,
            vitals_embedding=vitals_emb,
            raw_vitals=raw_vitals,
            run_memory=next_run,
            revival_context=rev_ctx,
        )
        return durable_val, next_run

    def _hierarchical_policy_logits(
        self,
        *,
        raw_candidate_logits: Tensor,
        candidate_embeddings: Tensor,
        joint_state_embedding: Tensor,
        vitals_embedding: Tensor | None = None,
        salient_vitals: Tensor | None = None,
        recurrent_state: Tensor,
        policy_branch_ids: Tensor,
        action_mask: Tensor,
        domain_ids: Tensor,
    ) -> Tensor:
        _batch_size, action_count = raw_candidate_logits.shape
        branch_ids = policy_branch_ids.long().clamp(min=0, max=max(0, action_count - 1))
        branch_axis = torch.arange(action_count, device=raw_candidate_logits.device).view(
            1, action_count, 1
        )
        in_branch = (branch_ids.unsqueeze(1) == branch_axis) & action_mask.unsqueeze(1)
        branch_active = in_branch.any(dim=-1)
        safe_in_branch = _safe_valid_mask(in_branch)

        scoped_logits = (
            raw_candidate_logits.unsqueeze(1)
            .expand(-1, action_count, -1)
            .masked_fill(~safe_in_branch, -torch.inf)
        )
        within_all = torch.log_softmax(scoped_logits, dim=-1)
        within_log_probs = within_all.gather(1, branch_ids.unsqueeze(1)).squeeze(1)

        weights = in_branch.to(dtype=candidate_embeddings.dtype)
        branch_counts = weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
        branch_mean = torch.einsum("bca,bad->bcd", weights, candidate_embeddings) / branch_counts
        expanded_cand = candidate_embeddings.unsqueeze(1).expand(
            -1, action_count, -1, -1
        )
        min_fill = torch.finfo(candidate_embeddings.dtype).min * 0.5
        branch_max = (
            expanded_cand.masked_fill(~safe_in_branch.unsqueeze(-1), min_fill)
            .max(dim=2)
            .values
            * branch_active.unsqueeze(-1).to(dtype=candidate_embeddings.dtype)
        )
        state_exp = joint_state_embedding.unsqueeze(1).expand(-1, action_count, -1)
        mem_exp = recurrent_state.unsqueeze(1).expand(-1, action_count, -1)
        branch_parts = [branch_mean, branch_max, state_exp]
        vitals_exp: Tensor | None = None
        if vitals_embedding is not None:
            vitals_exp = vitals_embedding.unsqueeze(1).expand(-1, action_count, -1)
            branch_parts.append(vitals_exp)
        branch_parts.append(mem_exp)
        branch_residual = self.branch_policy_head(torch.cat(branch_parts, dim=-1)).squeeze(-1)
        if vitals_exp is not None:
            if salient_vitals is not None:
                salient_exp = salient_vitals.unsqueeze(1).expand(-1, action_count, -1)
                branch_repr = self.branch_vitals_norm(
                    torch.cat([branch_mean, branch_max], dim=-1)
                )
                salient_aug = torch.cat(
                    [torch.ones_like(salient_vitals[:, :1]), salient_vitals],
                    dim=-1,
                )
                bilinear_delta = (
                    self.branch_vitals_bilinear(branch_repr) * salient_aug.unsqueeze(1)
                ).sum(dim=-1)
                branch_residual = (
                    branch_residual
                    + self.branch_vitals_head(
                        torch.cat([vitals_exp, salient_exp, branch_mean, branch_max], dim=-1)
                    ).squeeze(-1)
                    + bilinear_delta
                )
            else:
                branch_residual = branch_residual + self.branch_vitals_head(
                    torch.cat([vitals_exp, branch_mean, branch_max], dim=-1)
                ).squeeze(-1)

        combat_lse = torch.logsumexp(scoped_logits, dim=-1)
        combat_rows = domain_ids.eq(COMBAT_DOMAIN_ID).unsqueeze(-1)
        if self.actor_critic_config.enable_macro_branch_prior:
            macro_prior = combat_lse - branch_counts.squeeze(-1).log()
            raw_branch_logits = torch.where(
                combat_rows,
                combat_lse + branch_residual,
                macro_prior + branch_residual,
            )
        else:
            raw_branch_logits = torch.where(
                combat_rows,
                combat_lse + branch_residual,
                branch_residual,
            )
        safe_branch_active = _safe_valid_mask(branch_active)
        branch_log_probs_all = torch.log_softmax(
            raw_branch_logits.masked_fill(~safe_branch_active, -torch.inf),
            dim=-1,
        )
        candidate_branch_log_probs = branch_log_probs_all.gather(1, branch_ids)
        joint_log_probs = candidate_branch_log_probs + within_log_probs
        return joint_log_probs.masked_fill(~action_mask, 0.0)

    def _forward_representation_with_build(
        self,
        batch: GroundedCandidateBatch,
        recurrent_state: Tensor | None = None,
        *,
        validate: bool = True,
    ) -> tuple[HybridRepresentationOutput, Tensor | None, Tensor | None, Tensor | None]:
        has_build = self.actor_critic_config.enable_persistent_build_context
        has_vitals = self.actor_critic_config.enable_direct_vitals_context
        if not has_build and not has_vitals:
            return (
                self.forward_representation(batch, recurrent_state, validate=validate),
                None,
                None,
                None,
            )
        if validate:
            batch.validate(self.config)
        batch_size = int(batch.domain_ids.shape[0])
        if recurrent_state is None:
            recurrent_state = self.initial_state(
                batch_size,
                device=batch.world.features.device,
            )
        elif validate:
            self._validate_recurrent_state(
                recurrent_state,
                batch_size=batch_size,
                reference=batch.world.features,
            )
        world = self.encode_world(batch.world, batch.domain_ids, _validated=True)
        candidates = self.encode_candidates(
            batch.candidates,
            world,
            _validated=True,
        )
        build_emb = (
            self._pool_persistent_build(world, batch.world.zone_ids, batch.world.owner_ids)
            if has_build
            else None
        )
        vitals_emb: Tensor | None = None
        raw_vitals: Tensor | None = None
        if has_vitals:
            vitals_emb, raw_vitals = self._encode_direct_vitals(batch.world)
        mask = candidates.action_mask.bool()
        cfg = self.config
        memory_half = cfg.recurrent_hidden_dim // 2
        previous_run = recurrent_state[:, :memory_half]
        previous_combat = recurrent_state[:, memory_half:]
        combat_domain = batch.domain_ids.eq(COMBAT_DOMAIN_ID).unsqueeze(-1)

        run_update = self.run_recurrent_norm(
            self.run_recurrent_cell(world.state_embedding, previous_run)
        )
        next_run = torch.where(combat_domain, previous_run, run_update)
        combat_update = self.combat_recurrent_norm(
            self.combat_recurrent_cell(world.state_embedding, previous_combat)
        )
        next_combat = torch.where(
            combat_domain,
            combat_update,
            torch.zeros_like(combat_update),
        )
        next_recurrent_state = torch.cat([next_run, next_combat], dim=-1)

        memory_context = self.memory_to_candidate(next_recurrent_state)
        state_token = world.state_embedding + memory_context
        if build_emb is not None:
            state_token = state_token + self.persistent_build_to_state(build_emb)
        if vitals_emb is not None:
            state_token = state_token + self.persistent_vitals_to_state(vitals_emb)
        slow_strategy_token = self.strategy_token(next_run)
        action_count = int(mask.shape[1])
        candidate_tokens = candidates.embeddings + memory_context.unsqueeze(1)
        if build_emb is not None:
            build_for_cand = build_emb.unsqueeze(1).expand(-1, action_count, -1)
            cand_build_delta = self.persistent_build_to_candidate(
                torch.cat([candidates.embeddings, build_for_cand], dim=-1)
            )
            candidate_tokens = candidate_tokens + cand_build_delta
        stream_tokens = torch.cat(
            [
                state_token.unsqueeze(1),
                slow_strategy_token.unsqueeze(1),
                candidate_tokens,
            ],
            dim=1,
        )
        stream_mask = torch.cat(
            [
                torch.ones((batch_size, 1), dtype=torch.bool, device=mask.device),
                torch.ones((batch_size, 1), dtype=torch.bool, device=mask.device),
                mask,
            ],
            dim=1,
        )
        padding_mask = ~_safe_valid_mask(stream_mask)
        combat_rows = batch.domain_ids.eq(COMBAT_DOMAIN_ID)
        macro_rows = ~combat_rows
        selected = torch.empty_like(stream_tokens)
        if bool(macro_rows.any().item()):
            selected[macro_rows] = self.macro_stream(
                self.macro_stream_norm(stream_tokens[macro_rows]),
                src_key_padding_mask=padding_mask[macro_rows],
            )
        if bool(combat_rows.any().item()):
            selected[combat_rows] = self.combat_stream(
                self.combat_stream_norm(stream_tokens[combat_rows]),
                src_key_padding_mask=padding_mask[combat_rows],
            )
        joint = self.joint_stream(
            selected,
            src_key_padding_mask=padding_mask,
        )
        joint_state = self.joint_state_norm(joint[:, 0])
        joint_candidates = self.joint_candidate_norm(joint[:, 2:])
        joint_candidates = joint_candidates * mask.unsqueeze(-1).to(joint_candidates.dtype)
        return (
            HybridRepresentationOutput(
                recurrent_state=next_recurrent_state,
                joint_state_embedding=joint_state,
                world_state_embedding=world.state_embedding,
                candidate_embeddings=joint_candidates,
                action_mask=mask,
            ),
            build_emb,
            vitals_emb,
            raw_vitals,
        )

    def forward(  # type: ignore[override]
        self,
        batch: GroundedCandidateBatch,
        recurrent_state: Tensor | None = None,
        *,
        critic_revival_context: Tensor | None = None,
        compute_durable_value: bool = True,
        validate: bool = True,
    ) -> HybridActorCriticOutput:
        base, build_emb, vitals_emb, raw_vitals = self._forward_representation_with_build(
            batch,
            recurrent_state,
            validate=validate,
        )
        batch_size, action_count = base.action_mask.shape
        rev_ctx = self._normalize_critic_revival_context(
            critic_revival_context,
            batch_size=int(batch_size),
            device=batch.world.features.device,
            dtype=batch.world.features.dtype,
        )
        state = base.joint_state_embedding.unsqueeze(1).expand(-1, int(action_count), -1)
        memory = base.recurrent_state.unsqueeze(1).expand(-1, int(action_count), -1)
        raw_logits = self.candidate_policy_head(
            torch.cat([base.candidate_embeddings, state, memory], dim=-1)
        ).squeeze(-1)
        if self.actor_critic_config.enable_role_conditioned_candidate_policy:
            bounded_roles = batch.candidates.role_ids.long().clamp(
                min=0,
                max=self.config.role_vocab_size - 1,
            )
            role_emb = self.token_embedder.role_embedding(bounded_roles)
            cand_modulated = base.candidate_embeddings * torch.tanh(
                self.candidate_role_gate(role_emb)
            )
            role_delta = self.candidate_role_policy_head(
                torch.cat(
                    [cand_modulated, base.candidate_embeddings, role_emb, state],
                    dim=-1,
                )
            ).squeeze(-1)
            raw_logits = raw_logits + role_delta
        if self.actor_critic_config.enable_hierarchical_branch_policy:
            salient_for_actor = (
                raw_vitals[:, -SALIENT_VITALS_DIM:]
                if (
                    raw_vitals is not None
                    and self.actor_critic_config.enable_persistent_only_durable_value
                )
                else None
            )
            policy_logits = self._hierarchical_policy_logits(
                raw_candidate_logits=raw_logits,
                candidate_embeddings=base.candidate_embeddings,
                joint_state_embedding=base.joint_state_embedding,
                vitals_embedding=vitals_emb,
                salient_vitals=salient_for_actor,
                recurrent_state=base.recurrent_state,
                policy_branch_ids=batch.candidates.policy_branch_ids,
                action_mask=base.action_mask,
                domain_ids=batch.domain_ids,
            )
        else:
            policy_logits = raw_logits.masked_fill(~base.action_mask, 0.0)
        value = self.state_value_head(
            torch.cat([base.joint_state_embedding, base.recurrent_state], dim=-1)
        ).squeeze(-1)
        if rev_ctx is not None:
            rev_scaled = rev_ctx * 4.0
            value = (
                value
                + self.state_revival_linear(rev_scaled).squeeze(-1)
                + self.state_revival_head(rev_scaled).squeeze(-1)
            )
        durable_value: Tensor | None = None
        if compute_durable_value and self.actor_critic_config.enable_durable_value_head:
            if self.actor_critic_config.enable_persistent_only_durable_value:
                durable_value, _ = self._compute_persistent_durable_value(
                    batch.world,
                    recurrent_state,
                    critic_revival_context=critic_revival_context,
                    validate=False,
                )
            else:
                memory_half = self.config.recurrent_hidden_dim // 2
                durable_value = self._durable_readout(
                    world_state_embedding=base.world_state_embedding,
                    build_embedding=build_emb,
                    vitals_embedding=vitals_emb,
                    raw_vitals=raw_vitals,
                    run_memory=base.recurrent_state[:, :memory_half],
                    revival_context=rev_ctx,
                )
        output = HybridActorCriticOutput(
            policy_logits=policy_logits,
            value=value,
            recurrent_state=base.recurrent_state,
            joint_state_embedding=base.joint_state_embedding,
            world_state_embedding=base.world_state_embedding,
            candidate_embeddings=base.candidate_embeddings,
            action_mask=base.action_mask,
            durable_value=durable_value,
        )
        if validate:
            output.validate(
                self.actor_critic_config,
                require_durable_value=compute_durable_value,
            )
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
    if model.actor_critic_config.has_v2_extensions:
        # Preserve feature_projection exactly so LayerNorm(224) + Linear(224, D)
        # stays bitwise identical to v18 on all non-upgrade tokens, and copy
        # zone 24 (candidate) into newly introduced zone 26 (upgrade_preview).
        with torch.no_grad():
            zone_emb = model.token_embedder.zone_embedding
            if zone_emb.weight.shape[0] > 26:
                zone_emb.weight[26].copy_(zone_emb.weight[24])
    return {
        "inherited": tuple(sorted(loaded)),
        "fresh": tuple(sorted(result.missing_keys)),
    }


def initialize_actor_critic_from_macro_publication(
    model: HybridDomainActorCriticModel,
    source_state: dict[str, Tensor],
) -> dict[str, tuple[str, ...]]:
    """Hot-start an actor-critic from a parent macro-credit publication.

    Every tensor in ``source_state`` must match an existing parameter in
    ``model`` with identical shape (zero dropped parent weights), and any newly
    added destination tensors must belong to ``V2_FRESH_TENSOR_PREFIXES`` under
    ``has_v2_extensions``.
    """

    destination = model.state_dict()
    loaded: dict[str, Tensor] = {}
    retired_source: list[str] = []
    unmatched_source: list[str] = []
    allow_retired = bool(model.actor_critic_config.enable_deconfounded_durable_value)
    retired_prefixes = (
        RETIRED_MONOTONE_DECONFOUNDED_SOURCE_PREFIXES
        if model.actor_critic_config.enable_monotone_deconfounded_durable_value
        else RETIRED_DECONFOUNDED_SOURCE_PREFIXES
    )
    for key, value in source_state.items():
        if value.is_floating_point() and not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"hybrid macro publication tensor is non-finite: {key}")
        target = destination.get(key)
        if target is None or tuple(target.shape) != tuple(value.shape):
            if allow_retired and target is None and key.startswith(retired_prefixes):
                retired_source.append(key)
                continue
            unmatched_source.append(key)
            continue
        loaded[key] = value
    if unmatched_source:
        raise ValueError(
            "macro publication source tensors incompatible with destination model: "
            + ", ".join(sorted(unmatched_source))
        )
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
        raise ValueError("macro publication source misses: " + ", ".join(absent))
    unexpected_fresh = [
        name
        for name in result.missing_keys
        if not (
            model.actor_critic_config.has_v2_extensions
            and name.startswith(V2_FRESH_TENSOR_PREFIXES)
        )
    ]
    if unexpected_fresh:
        raise ValueError(
            "macro publication leaves non-extension tensors uninitialized: "
            + ", ".join(sorted(unexpected_fresh))
        )
    summary: dict[str, tuple[str, ...]] = {
        "inherited": tuple(sorted(loaded)),
        "fresh": tuple(sorted(result.missing_keys)),
    }
    if retired_source:
        summary["retired"] = tuple(sorted(retired_source))
    return summary


__all__ = [
    "CRITIC_REVIVAL_CONTEXT_DIM",
    "HYBRID_ACTOR_CRITIC_MODEL_VERSION",
    "HYBRID_ACTOR_CRITIC_V2_MODEL_VERSION",
    "PERSISTENT_BUILD_ZONE_IDS",
    "V2_FRESH_TENSOR_PREFIXES",
    "HybridActorCriticConfig",
    "HybridActorCriticOutput",
    "HybridDomainActorCriticModel",
    "hierarchical_greedy_action_index",
    "initialize_actor_critic_from_iql",
    "initialize_actor_critic_from_macro_publication",
]
