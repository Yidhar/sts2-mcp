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
PLAYER_ZONE_ID: Final = 13
RUN_ZONE_ID: Final = 22
NUMERIC_FACT_SLOT_START: Final = 6
NUMERIC_FACT_SLOT_END: Final = 166
PLAYER_CURRENT_HP_OFFSET: Final = 34 - NUMERIC_FACT_SLOT_START
PLAYER_GOLD_OFFSET: Final = 54 - NUMERIC_FACT_SLOT_START
PLAYER_HP_OFFSET: Final = 62 - NUMERIC_FACT_SLOT_START
PLAYER_MAX_HP_OFFSET: Final = 111 - NUMERIC_FACT_SLOT_START
SALIENT_VITALS_DIM: Final = 4
CRITIC_REVIVAL_CONTEXT_DIM: Final = 6
PERSISTENT_BUILD_ZONE_IDS: Final[tuple[int, ...]] = (2, 13, 15, 16, 20, 22)
IDENTITY_BLIND_CARD_COUNT_ZONE_IDS: Final[frozenset[int]] = frozenset({2, 20})
V2_FRESH_TENSOR_PREFIXES: Final[tuple[str, ...]] = (
    "branch_policy_head.",
    "branch_vitals_",
    "durable_revival_",
    "durable_value_head.",
    "durable_vitals_",
    "persistent_build_",
    "persistent_vitals_",
    "state_revival_",
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
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"hybrid actor-critic {name} must be boolean")

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
        if config.enable_durable_value_head:
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
    ) -> Tensor:
        batch_size = int(world.tokens.shape[0])
        build_mask = self._persistent_token_mask(zone_ids, owner_ids, world.world_mask)
        null_token = self.world_null_token.expand(batch_size, -1, -1)
        keys = torch.cat([null_token, world.tokens], dim=1)
        null_mask = torch.ones((batch_size, 1), dtype=torch.bool, device=world.tokens.device)
        key_mask = torch.cat([null_mask, build_mask], dim=1)
        queries = self.persistent_build_queries.expand(batch_size, -1, -1)
        delta, _ = self.persistent_build_attn(
            queries,
            keys,
            keys,
            key_padding_mask=~key_mask,
            need_weights=False,
        )
        pooled: Tensor = self.persistent_build_norm(queries + delta).mean(dim=1)
        return pooled

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

    def _durable_readout(
        self,
        *,
        world_state_embedding: Tensor,
        build_embedding: Tensor | None,
        vitals_embedding: Tensor | None = None,
        raw_vitals: Tensor | None = None,
        run_memory: Tensor,
        revival_context: Tensor | None = None,
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
        if raw_vitals is not None:
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
        pers_world = self._mask_world_to_persistent(world)
        zero_domains = torch.zeros(
            (batch_size,),
            dtype=torch.long,
            device=world.features.device,
        )
        pers_enc = self.encode_world(pers_world, zero_domains, _validated=not validate)
        pers_run = self.run_recurrent_norm(
            self.run_recurrent_cell(pers_enc.state_embedding, previous_run)
        )
        build_emb = (
            self._pool_persistent_build(pers_enc, pers_world.zone_ids, pers_world.owner_ids)
            if self.actor_critic_config.enable_persistent_build_context
            else None
        )
        vitals_emb: Tensor | None = None
        raw_vitals: Tensor | None = None
        if self.actor_critic_config.enable_direct_vitals_context:
            vitals_emb, raw_vitals = self._encode_direct_vitals(pers_world)
        durable_val = self._durable_readout(
            world_state_embedding=pers_enc.state_embedding,
            build_embedding=build_emb,
            vitals_embedding=vitals_emb,
            raw_vitals=raw_vitals,
            run_memory=pers_run,
            revival_context=rev_ctx,
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
        if self.actor_critic_config.enable_durable_value_head:
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
]
