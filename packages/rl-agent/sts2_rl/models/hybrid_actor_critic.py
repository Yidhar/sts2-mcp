"""Recurrent hybrid actor-critic for the fresh FIFO V-trace lineage.

The factual encoder, dual macro/combat stems, shared fusion stream and split
run/combat memory are retained from the hybrid controller.  Control is once
again a dense stochastic policy with a candidate-independent scalar value.
Unlike the retired IQL actor, policy features are *not* detached: current
on-policy gradients and entropy train the shared representation together with
the value regression.
"""

from __future__ import annotations

import contextlib
import hashlib
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Final

import torch
from torch import Tensor, nn

from .grounded_candidate import (
    COMBAT_DOMAIN_ID,
    MACRO_ECONOMIC_SURFACE_REST,
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
HAND_ZONE_ID: Final = 3
SHOP_ZONE_ID: Final = 9
REST_SITE_ZONE_ID: Final = 11
PLAYER_ZONE_ID: Final = 13
ENEMY_ZONE_ID: Final = 14
INTENT_ZONE_ID: Final = 19
MODIFIER_ZONE_ID: Final = 20
RUN_ZONE_ID: Final = 22
COMBAT_ZONE_ID: Final = 23
SHOP_SUB_ZONE_ID: Final = 28
NUMERIC_FACT_SLOT_START: Final = 6
NUMERIC_FACT_SLOT_END: Final = 166
CARD_CURRENT_UPGRADE_LEVEL_SLOT: Final = 37
DECK_FLOOR_ADDED_SLOT: Final = 52
CANDIDATE_HEAL_AMOUNT_SLOT: Final = 59
CARD_IS_UPGRADED_SLOT: Final = 102
DECK_QUANTITY_SLOT: Final = 131
CARD_UPGRADE_LEVEL_SLOT: Final = 158
CARD_UPGRADED_SLOT: Final = 159
CARD_UPGRADES_SLOT: Final = 160
DECK_REFERENCE_SIZE: Final = 12.0
DURABLE_DECK_SIZE_DIM: Final = 4
DURABLE_DECK_SUMMARY_DIM: Final = 4
DURABLE_CARD_ID_EMB_DIM: Final = 64
PLAYER_BLOCK_OFFSET: Final = 15 - NUMERIC_FACT_SLOT_START
PLAYER_CURRENT_HP_OFFSET: Final = 34 - NUMERIC_FACT_SLOT_START
INTENT_DAMAGE_OFFSET: Final = 38 - NUMERIC_FACT_SLOT_START
COMBAT_ENERGY_OFFSET: Final = 44 - NUMERIC_FACT_SLOT_START
PLAYER_GOLD_OFFSET: Final = 54 - NUMERIC_FACT_SLOT_START
PLAYER_HP_OFFSET: Final = 62 - NUMERIC_FACT_SLOT_START
ENEMY_IS_ALIVE_OFFSET: Final = 67 - NUMERIC_FACT_SLOT_START
COMBAT_MAX_ENERGY_OFFSET: Final = 110 - NUMERIC_FACT_SLOT_START
PLAYER_MAX_HP_OFFSET: Final = 111 - NUMERIC_FACT_SLOT_START
PLAYER_OPEN_POTION_SLOTS_OFFSET: Final = 119 - NUMERIC_FACT_SLOT_START
INTENT_REPEATS_OFFSET: Final = 133 - NUMERIC_FACT_SLOT_START
COMBAT_ROUND_OFFSET: Final = 136 - NUMERIC_FACT_SLOT_START
INTENT_TOTAL_DAMAGE_OFFSET: Final = 156 - NUMERIC_FACT_SLOT_START
RUN_ACT_OFFSET: Final = 6 - NUMERIC_FACT_SLOT_START
RUN_ACT_SLOT: Final = 6
RUN_FLOOR_OFFSET: Final = 51 - NUMERIC_FACT_SLOT_START
SALIENT_VITALS_DIM: Final = 4
COMBAT_VITALS_DIM: Final = 16
DURABLE_PROGRESS_VITALS_DIM: Final = 10
DURABLE_STAGE_DIM: Final = 5
DURABLE_HP_BASIS_DIM: Final = 5
DURABLE_LIQUID_BASIS_DIM: Final = 4
DURABLE_CYCLE_BASIS_DIM: Final = 3
DURABLE_TILT_BETAS: Final[tuple[float, ...]] = (1.5, 3.0, 6.0)
DURABLE_BUILD_QUERIES_V9: Final = 8
CRITIC_REVIVAL_CONTEXT_DIM: Final = 6
PERSISTENT_BUILD_ZONE_IDS: Final[tuple[int, ...]] = (2, 13, 15, 16, 20, 22)
IDENTITY_BLIND_CARD_COUNT_ZONE_IDS: Final[frozenset[int]] = frozenset({2, 20})
V2_FRESH_TENSOR_PREFIXES: Final[tuple[str, ...]] = (
    "branch_policy_head.",
    "branch_vitals_",
    "candidate_role_",
    "combat_vitals_",
    "durable_active_",
    "durable_cycle_",
    "durable_deck_",
    "durable_hp_",
    "durable_liquid_",
    "durable_progress_vitals_",
    "durable_revival_",
    "durable_shop_",
    "durable_stage_",
    "durable_twin_",
    "durable_value_head.",
    "durable_vitals_",
    "latent_slot_readout.",
    "local_combat_value_head.",
    "local_combat_value_vitals_",
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
    enable_expressive_synergy_backbone: bool = False
    enable_debloated_synergy_durable_value: bool = False
    enable_build_conditioned_durable_value: bool = False
    enable_twin_durable_value_heads: bool = False
    enable_local_combat_value_head: bool = False
    enable_direct_combat_vitals_context: bool = False
    enable_active_durable_vitals_gates: bool = False
    enable_marginal_card_id_durable_value: bool = False
    suppress_full_hp_campfire_heal: bool = False

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
            "enable_expressive_synergy_backbone",
            "enable_debloated_synergy_durable_value",
            "enable_build_conditioned_durable_value",
            "enable_twin_durable_value_heads",
            "enable_local_combat_value_head",
            "enable_direct_combat_vitals_context",
            "enable_active_durable_vitals_gates",
            "enable_marginal_card_id_durable_value",
            "suppress_full_hp_campfire_heal",
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
        if (
            self.enable_expressive_synergy_backbone
            and not self.enable_balanced_deconfounded_durable_value
        ):
            raise ValueError(
                "enable_expressive_synergy_backbone requires enable_balanced_deconfounded_durable_value=True"
            )
        if (
            self.enable_debloated_synergy_durable_value
            and not self.enable_expressive_synergy_backbone
        ):
            raise ValueError(
                "enable_debloated_synergy_durable_value requires enable_expressive_synergy_backbone=True"
            )
        if (
            self.enable_build_conditioned_durable_value
            and not self.enable_debloated_synergy_durable_value
        ):
            raise ValueError(
                "enable_build_conditioned_durable_value requires enable_debloated_synergy_durable_value=True"
            )
        if self.enable_twin_durable_value_heads and not (
            self.enable_durable_value_head and self.enable_persistent_only_durable_value
        ):
            raise ValueError(
                "enable_twin_durable_value_heads requires enable_durable_value_head=True and enable_persistent_only_durable_value=True"
            )
        if (
            self.enable_active_durable_vitals_gates
            and not self.enable_monotone_deconfounded_durable_value
        ):
            raise ValueError(
                "enable_active_durable_vitals_gates requires enable_monotone_deconfounded_durable_value=True"
            )
        if (
            self.enable_marginal_card_id_durable_value
            and not self.enable_draw_weighted_durable_value
        ):
            raise ValueError(
                "enable_marginal_card_id_durable_value requires enable_draw_weighted_durable_value=True"
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
            or self.enable_expressive_synergy_backbone
            or self.enable_debloated_synergy_durable_value
            or self.enable_build_conditioned_durable_value
            or self.enable_twin_durable_value_heads
            or self.enable_local_combat_value_head
            or self.enable_direct_combat_vitals_context
            or self.enable_active_durable_vitals_gates
            or self.enable_marginal_card_id_durable_value
            or self.suppress_full_hp_campfire_heal
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
    def durable_v9(
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
            enable_expressive_synergy_backbone=True,
        )

    @classmethod
    def durable_v10(
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
            enable_expressive_synergy_backbone=True,
            enable_debloated_synergy_durable_value=True,
        )

    @classmethod
    def durable_v11(
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
            enable_expressive_synergy_backbone=True,
            enable_debloated_synergy_durable_value=True,
            enable_build_conditioned_durable_value=True,
            enable_twin_durable_value_heads=True,
        )

    @classmethod
    def from_mapping(
        cls,
        mapping: dict[str, Any],
        *,
        fallback_grounded: GroundedCandidateConfig,
    ) -> HybridActorCriticConfig:
        version = mapping.get("version")
        raw_grounded = mapping.get("grounded")
        resolved_grounded = (
            GroundedCandidateConfig(**raw_grounded)
            if isinstance(raw_grounded, dict)
            else fallback_grounded
        )
        if version == HYBRID_ACTOR_CRITIC_MODEL_VERSION:
            return cls(
                grounded=resolved_grounded,
                domain_layers=int(mapping.get("domain_layers", 1)),
                joint_layers=int(mapping.get("joint_layers", 2)),
            )
        if version == HYBRID_ACTOR_CRITIC_V2_MODEL_VERSION:
            return cls(
                grounded=resolved_grounded,
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
                enable_expressive_synergy_backbone=bool(
                    mapping.get("enable_expressive_synergy_backbone", False)
                ),
                enable_debloated_synergy_durable_value=bool(
                    mapping.get("enable_debloated_synergy_durable_value", False)
                ),
                enable_build_conditioned_durable_value=bool(
                    mapping.get("enable_build_conditioned_durable_value", False)
                ),
                enable_twin_durable_value_heads=bool(
                    mapping.get("enable_twin_durable_value_heads", False)
                ),
                enable_local_combat_value_head=bool(
                    mapping.get("enable_local_combat_value_head", False)
                ),
                enable_direct_combat_vitals_context=bool(
                    mapping.get("enable_direct_combat_vitals_context", False)
                ),
                enable_active_durable_vitals_gates=bool(
                    mapping.get("enable_active_durable_vitals_gates", False)
                ),
                enable_marginal_card_id_durable_value=bool(
                    mapping.get("enable_marginal_card_id_durable_value", False)
                ),
                suppress_full_hp_campfire_heal=bool(
                    mapping.get("suppress_full_hp_campfire_heal", False)
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
            if self.enable_expressive_synergy_backbone:
                base.update(
                    enable_expressive_synergy_backbone=(
                        self.enable_expressive_synergy_backbone
                    ),
                )
            if self.enable_debloated_synergy_durable_value:
                base.update(
                    enable_debloated_synergy_durable_value=(
                        self.enable_debloated_synergy_durable_value
                    ),
                )
            if self.enable_build_conditioned_durable_value:
                base.update(
                    enable_build_conditioned_durable_value=(
                        self.enable_build_conditioned_durable_value
                    ),
                )
            if self.enable_twin_durable_value_heads:
                base.update(
                    enable_twin_durable_value_heads=(
                        self.enable_twin_durable_value_heads
                    ),
                )
            if self.enable_local_combat_value_head:
                base.update(
                    enable_local_combat_value_head=(
                        self.enable_local_combat_value_head
                    ),
                )
            if self.enable_direct_combat_vitals_context:
                base.update(
                    enable_direct_combat_vitals_context=(
                        self.enable_direct_combat_vitals_context
                    ),
                )
            if self.enable_active_durable_vitals_gates:
                base.update(
                    enable_active_durable_vitals_gates=(
                        self.enable_active_durable_vitals_gates
                    ),
                )
            if self.enable_marginal_card_id_durable_value:
                base.update(
                    enable_marginal_card_id_durable_value=(
                        self.enable_marginal_card_id_durable_value
                    ),
                )
            if self.suppress_full_hp_campfire_heal:
                base.update(
                    suppress_full_hp_campfire_heal=(
                        self.suppress_full_hp_campfire_heal
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
    combat_value: Tensor | None = None  # [B], encounter-local critic when enabled

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
        if config.enable_local_combat_value_head:
            if self.combat_value is None:
                raise ValueError("hybrid actor combat_value is required when enabled")
            if tuple(self.combat_value.shape) != (batch_size,):
                raise ValueError("hybrid actor combat_value has the wrong shape")
            if not torch.is_floating_point(self.combat_value) or not bool(
                torch.isfinite(self.combat_value).all().item()
            ):
                raise ValueError("hybrid actor combat_value must be finite floating point")
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
        self._detach_value_heads_from_trunk: bool = False
        self._detach_macro_branch_prior_lse: bool = False
        self._suppress_full_hp_campfire_heal: bool = bool(
            actor_critic.suppress_full_hp_campfire_heal
        )
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
        rest_heal_role_digest = hashlib.sha256(b"role\0rest_site:rest").digest()
        self._rest_heal_role_id = 2 + int.from_bytes(rest_heal_role_digest[:8], "big") % (
            cfg.role_vocab_size - 2
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
        if actor_critic.enable_local_combat_value_head:
            lc_in_dim = cfg.d_model + cfg.recurrent_hidden_dim
            lc_fc1 = nn.Linear(lc_in_dim, cfg.ffn_dim)
            nn.init.normal_(lc_fc1.weight, mean=0.0, std=0.05)
            nn.init.zeros_(lc_fc1.bias)
            lc_out = nn.Linear(cfg.ffn_dim, 1)
            nn.init.zeros_(lc_out.weight)
            nn.init.zeros_(lc_out.bias)
            self.local_combat_value_head = nn.Sequential(
                nn.LayerNorm(lc_in_dim),
                lc_fc1,
                nn.SiLU(),
                lc_out,
            )
            if actor_critic.enable_direct_combat_vitals_context:
                self.local_combat_value_vitals_linear = nn.Linear(
                    COMBAT_VITALS_DIM, 1, bias=False
                )
                nn.init.zeros_(self.local_combat_value_vitals_linear.weight)
                lcv_mid = max(32, cfg.ffn_dim // 2)
                lcv_fc1 = nn.Linear(COMBAT_VITALS_DIM, lcv_mid)
                nn.init.normal_(lcv_fc1.weight, mean=0.0, std=0.20)
                nn.init.zeros_(lcv_fc1.bias)
                lcv_out = nn.Linear(lcv_mid, 1)
                nn.init.zeros_(lcv_out.weight)
                nn.init.zeros_(lcv_out.bias)
                self.local_combat_value_vitals_head = nn.Sequential(
                    lcv_fc1,
                    nn.SiLU(),
                    lcv_out,
                )
        if actor_critic.enable_direct_combat_vitals_context:
            self.combat_vitals_to_state = nn.Linear(COMBAT_VITALS_DIM, cfg.d_model)
            nn.init.zeros_(self.combat_vitals_to_state.weight)
            nn.init.zeros_(self.combat_vitals_to_state.bias)
            self.combat_vitals_policy_norm = nn.LayerNorm(cfg.d_model)
            self.combat_vitals_policy_bilinear = nn.Linear(
                cfg.d_model,
                COMBAT_VITALS_DIM + 1,
                bias=False,
            )
            nn.init.zeros_(self.combat_vitals_policy_bilinear.weight)
            cv_pol_mid = max(32, cfg.ffn_dim // 2)
            cv_pol_fc1 = nn.Linear(cfg.d_model + COMBAT_VITALS_DIM, cv_pol_mid)
            nn.init.normal_(cv_pol_fc1.weight, mean=0.0, std=0.10)
            nn.init.zeros_(cv_pol_fc1.bias)
            cv_pol_out = nn.Linear(cv_pol_mid, 1)
            nn.init.zeros_(cv_pol_out.weight)
            nn.init.zeros_(cv_pol_out.bias)
            self.combat_vitals_policy_head = nn.Sequential(
                cv_pol_fc1,
                nn.SiLU(),
                cv_pol_out,
            )
        if actor_critic.enable_expressive_synergy_backbone:
            slot_in_dim = cfg.latent_slots * cfg.d_model
            slot_out = nn.Linear(cfg.d_model, cfg.d_model)
            nn.init.zeros_(slot_out.weight)
            nn.init.zeros_(slot_out.bias)
            self.latent_slot_readout = nn.Sequential(
                nn.LayerNorm(slot_in_dim),
                nn.Linear(slot_in_dim, cfg.d_model),
                nn.GELU(),
                slot_out,
            )
        if actor_critic.enable_persistent_build_context:
            n_build_queries = (
                DURABLE_BUILD_QUERIES_V9
                if actor_critic.enable_expressive_synergy_backbone
                else 4
            )
            self.persistent_build_queries = nn.Parameter(
                torch.empty(1, n_build_queries, cfg.d_model)
            )
            nn.init.normal_(self.persistent_build_queries, mean=0.0, std=0.02)
            self.persistent_build_attn = nn.MultiheadAttention(
                cfg.d_model,
                cfg.n_heads,
                dropout=cfg.dropout,
                batch_first=True,
            )
            self.persistent_build_norm = nn.LayerNorm(cfg.d_model)
            if actor_critic.enable_expressive_synergy_backbone:
                bslot_in_dim = n_build_queries * cfg.d_model
                bslot_out = nn.Linear(cfg.d_model, cfg.d_model)
                nn.init.zeros_(bslot_out.weight)
                nn.init.zeros_(bslot_out.bias)
                self.persistent_build_slot_readout = nn.Sequential(
                    nn.LayerNorm(bslot_in_dim),
                    nn.Linear(bslot_in_dim, cfg.d_model),
                    nn.SiLU(),
                    bslot_out,
                )
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
                    if actor_critic.enable_active_durable_vitals_gates:
                        def _make_active_gate(out_d: int, init_b: float) -> nn.Sequential:
                            a_fc1 = nn.Linear(stage_in_dim, stage_mid_dim)
                            nn.init.normal_(a_fc1.weight, mean=0.0, std=0.10)
                            nn.init.zeros_(a_fc1.bias)
                            a_out = nn.Linear(stage_mid_dim, out_d)
                            nn.init.zeros_(a_out.weight)
                            nn.init.constant_(a_out.bias, init_b)
                            return nn.Sequential(
                                nn.LayerNorm(stage_in_dim), a_fc1, nn.SiLU(), a_out
                            )

                        self.durable_active_hp_gate = _make_active_gate(
                            DURABLE_HP_BASIS_DIM, -1.2
                        )
                        self.durable_active_liquid_gate = _make_active_gate(
                            DURABLE_LIQUID_BASIS_DIM, -2.0
                        )
                        self.durable_active_cycle_gate = _make_active_gate(
                            DURABLE_CYCLE_BASIS_DIM, -2.0
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
                if actor_critic.enable_marginal_card_id_durable_value:
                    marg_mid = max(32, cfg.ffn_dim // 2)
                    marg_in_dim = DURABLE_CARD_ID_EMB_DIM + cfg.ffn_dim * 2
                    self.durable_deck_card_id_emb = nn.Embedding(
                        cfg.entity_vocab_size * 2,
                        DURABLE_CARD_ID_EMB_DIM,
                        padding_idx=0,
                    )
                    gen_h1 = torch.Generator(device="cpu")
                    gen_h1.manual_seed(20261121)
                    with torch.no_grad():
                        self.durable_deck_card_id_emb.weight.copy_(
                            torch.randn(
                                self.durable_deck_card_id_emb.weight.shape,
                                generator=gen_h1,
                                dtype=torch.float32,
                            )
                            * 0.10
                        )
                        self.durable_deck_card_id_emb.weight[0].zero_()
                    self.durable_deck_marginal_id_linear = nn.Linear(
                        DURABLE_CARD_ID_EMB_DIM, 1, bias=False
                    )
                    nn.init.zeros_(self.durable_deck_marginal_id_linear.weight)
                    marg_fc1 = nn.Linear(marg_in_dim, marg_mid)
                    nn.init.normal_(marg_fc1.weight, mean=0.0, std=0.08)
                    nn.init.zeros_(marg_fc1.bias)
                    marg_fc_out = nn.Linear(marg_mid, 1)
                    nn.init.zeros_(marg_fc_out.weight)
                    nn.init.zeros_(marg_fc_out.bias)
                    self.durable_deck_marginal_mlp = nn.Sequential(
                        nn.LayerNorm(marg_in_dim),
                        marg_fc1,
                        nn.SiLU(),
                        marg_fc_out,
                    )
                if actor_critic.enable_expressive_synergy_backbone:
                    synergy_heads = (
                        cfg.n_heads if cfg.ffn_dim % cfg.n_heads == 0 else 1
                    )
                    self.durable_deck_synergy_null = nn.Parameter(
                        torch.zeros(1, 1, cfg.ffn_dim)
                    )
                    self.durable_deck_synergy_norm1 = nn.LayerNorm(cfg.ffn_dim)
                    self.durable_deck_synergy_attn = nn.MultiheadAttention(
                        cfg.ffn_dim,
                        synergy_heads,
                        dropout=cfg.dropout,
                        batch_first=True,
                    )
                    nn.init.zeros_(self.durable_deck_synergy_attn.out_proj.weight)
                    if self.durable_deck_synergy_attn.out_proj.bias is not None:
                        nn.init.zeros_(self.durable_deck_synergy_attn.out_proj.bias)
                    syn_ffn_out = nn.Linear(cfg.ffn_dim, cfg.ffn_dim)
                    nn.init.zeros_(syn_ffn_out.weight)
                    nn.init.zeros_(syn_ffn_out.bias)
                    self.durable_deck_synergy_ffn = nn.Sequential(
                        nn.LayerNorm(cfg.ffn_dim),
                        nn.Linear(cfg.ffn_dim, cfg.ffn_dim),
                        nn.SiLU(),
                        syn_ffn_out,
                    )
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
            if actor_critic.enable_build_conditioned_durable_value:
                stage_in_dim_bc = DURABLE_STAGE_DIM + rev_dim
                stage_mid_bc = max(16, cfg.ffn_dim // 2)
                z_build_dim = cfg.d_model * 2 + cfg.ffn_dim + DURABLE_DECK_SUMMARY_DIM
                build_stage_dim = z_build_dim + stage_in_dim_bc
                shop_heads = cfg.n_heads if cfg.d_model % cfg.n_heads == 0 else 1
                shop_readout_dim = z_build_dim + cfg.d_model + stage_in_dim_bc

                def _make_zero_mlp(in_d: int, mid_d: int, out_d: int, bias_val: float = 0.0) -> nn.Sequential:
                    fc1 = nn.Linear(in_d, mid_d)
                    nn.init.normal_(fc1.weight, mean=0.0, std=0.08)
                    nn.init.zeros_(fc1.bias)
                    fc_out = nn.Linear(mid_d, out_d)
                    nn.init.zeros_(fc_out.weight)
                    nn.init.constant_(fc_out.bias, bias_val)
                    return nn.Sequential(nn.LayerNorm(in_d), fc1, nn.SiLU(), fc_out)

                self.durable_hp_build_gate = _make_zero_mlp(
                    build_stage_dim, stage_mid_bc, DURABLE_HP_BASIS_DIM, 0.0
                )
                self.durable_liquid_build_gate = _make_zero_mlp(
                    build_stage_dim, stage_mid_bc, DURABLE_LIQUID_BASIS_DIM, 0.0
                )
                self.durable_shop_query_proj = nn.Sequential(
                    nn.LayerNorm(build_stage_dim),
                    nn.Linear(build_stage_dim, cfg.d_model),
                )
                self.durable_shop_null = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
                self.durable_shop_attn = nn.MultiheadAttention(
                    cfg.d_model,
                    shop_heads,
                    dropout=cfg.dropout,
                    batch_first=True,
                )
                self.durable_shop_base_head = _make_zero_mlp(shop_readout_dim, stage_mid_bc, 1, 0.0)
                self.durable_shop_liquid_gate = _make_zero_mlp(
                    shop_readout_dim, stage_mid_bc, DURABLE_LIQUID_BASIS_DIM, -6.5
                )
            self._twin_dropout_active = False
            if actor_critic.enable_twin_durable_value_heads:
                self.durable_twin_head1_dropout = nn.Dropout(p=0.05)
                self.durable_twin_dropout = nn.Dropout(p=0.05)
                self.durable_twin_head1_dropout.eval()
                self.durable_twin_dropout.eval()
                twin_val_out = nn.Linear(cfg.ffn_dim, 1)
                nn.init.zeros_(twin_val_out.weight)
                nn.init.zeros_(twin_val_out.bias)
                self.durable_twin_value_head = nn.Sequential(
                    nn.LayerNorm(durable_dim),
                    nn.Linear(durable_dim, cfg.ffn_dim),
                    nn.SiLU(),
                    twin_val_out,
                )
                if actor_critic.enable_monotone_deconfounded_durable_value:
                    t_stage_in_dim = DURABLE_STAGE_DIM + rev_dim
                    t_stage_mid = max(16, cfg.ffn_dim // 2)
                    t_bal = actor_critic.enable_balanced_deconfounded_durable_value

                    def _make_twin_gate(out_d: int, init_b: float) -> nn.Sequential:
                        g_fc1 = nn.Linear(t_stage_in_dim, t_stage_mid)
                        nn.init.normal_(g_fc1.weight, mean=0.0, std=0.10)
                        nn.init.zeros_(g_fc1.bias)
                        g_out = nn.Linear(t_stage_mid, out_d)
                        nn.init.zeros_(g_out.weight)
                        nn.init.constant_(g_out.bias, init_b)
                        return nn.Sequential(nn.LayerNorm(t_stage_in_dim), g_fc1, nn.SiLU(), g_out)

                    self.durable_twin_hp_gate = _make_twin_gate(
                        DURABLE_HP_BASIS_DIM, -5.2 if t_bal else -2.8
                    )
                    self.durable_twin_liquid_gate = _make_twin_gate(
                        DURABLE_LIQUID_BASIS_DIM, -5.2 if t_bal else -3.2
                    )
                    self.durable_twin_cycle_gate = _make_twin_gate(
                        DURABLE_CYCLE_BASIS_DIM, -5.5 if t_bal else -3.8
                    )
                    if actor_critic.enable_active_durable_vitals_gates:
                        self.durable_active_twin_hp_gate = _make_twin_gate(
                            DURABLE_HP_BASIS_DIM, -1.2
                        )
                        self.durable_active_twin_liquid_gate = _make_twin_gate(
                            DURABLE_LIQUID_BASIS_DIM, -2.0
                        )
                        self.durable_active_twin_cycle_gate = _make_twin_gate(
                            DURABLE_CYCLE_BASIS_DIM, -2.0
                        )
                if actor_critic.enable_draw_weighted_durable_value:
                    t_deck_ctx_dim = (
                        cfg.d_model * (1 + int(actor_critic.enable_direct_vitals_context))
                        + memory_half
                        + rev_dim
                    )
                    self.durable_twin_deck_norm = nn.LayerNorm(cfg.d_model * 3)
                    self.durable_twin_deck_card_fc = nn.Linear(cfg.d_model * 3, cfg.ffn_dim)
                    self.durable_twin_deck_raw_fc = nn.Linear(cfg.token_feature_dim * 2, cfg.ffn_dim)
                    self.durable_twin_deck_ctx_norm = nn.LayerNorm(t_deck_ctx_dim)
                    self.durable_twin_deck_ctx_fc = nn.Linear(t_deck_ctx_dim, cfg.ffn_dim, bias=False)
                    self.durable_twin_deck_out = nn.Linear(cfg.ffn_dim * 3, 1)
                    nn.init.zeros_(self.durable_twin_deck_out.weight)
                    nn.init.zeros_(self.durable_twin_deck_out.bias)
                    if actor_critic.enable_marginal_card_id_durable_value:
                        t_marg_mid = max(32, cfg.ffn_dim // 2)
                        t_marg_in_dim = DURABLE_CARD_ID_EMB_DIM + cfg.ffn_dim * 2
                        self.durable_twin_deck_card_id_emb = nn.Embedding(
                            cfg.entity_vocab_size * 2,
                            DURABLE_CARD_ID_EMB_DIM,
                            padding_idx=0,
                        )
                        gen_h2 = torch.Generator(device="cpu")
                        gen_h2.manual_seed(20261122)
                        with torch.no_grad():
                            self.durable_twin_deck_card_id_emb.weight.copy_(
                                torch.randn(
                                    self.durable_twin_deck_card_id_emb.weight.shape,
                                    generator=gen_h2,
                                    dtype=torch.float32,
                                )
                                * 0.10
                            )
                            self.durable_twin_deck_card_id_emb.weight[0].zero_()
                        self.durable_twin_deck_marginal_id_linear = nn.Linear(
                            DURABLE_CARD_ID_EMB_DIM, 1, bias=False
                        )
                        nn.init.zeros_(self.durable_twin_deck_marginal_id_linear.weight)
                        t_marg_fc1 = nn.Linear(t_marg_in_dim, t_marg_mid)
                        nn.init.normal_(t_marg_fc1.weight, mean=0.0, std=0.08)
                        nn.init.zeros_(t_marg_fc1.bias)
                        t_marg_fc_out = nn.Linear(t_marg_mid, 1)
                        nn.init.zeros_(t_marg_fc_out.weight)
                        nn.init.zeros_(t_marg_fc_out.bias)
                        self.durable_twin_deck_marginal_mlp = nn.Sequential(
                            nn.LayerNorm(t_marg_in_dim),
                            t_marg_fc1,
                            nn.SiLU(),
                            t_marg_fc_out,
                        )
                    if actor_critic.enable_monotone_deconfounded_durable_value:
                        self.durable_twin_deck_tilted_out = nn.Linear(
                            cfg.ffn_dim * 3,
                            len(DURABLE_TILT_BETAS),
                        )
                        nn.init.zeros_(self.durable_twin_deck_tilted_out.weight)
                        nn.init.zeros_(self.durable_twin_deck_tilted_out.bias)
                        if actor_critic.enable_balanced_deconfounded_durable_value:
                            self.durable_twin_deck_linear_logit = nn.Parameter(
                                torch.tensor(math.log(math.e - 1.0))
                            )
                            self.durable_twin_deck_tilt_logits = nn.Parameter(
                                torch.tensor([-4.2, -4.0, -4.2])
                            )
                        else:
                            self.durable_twin_deck_linear_logit = nn.Parameter(torch.zeros(()))
                            self.durable_twin_deck_tilt_logits = nn.Parameter(
                                torch.tensor([-1.8, -1.5, -1.8])
                            )
                if actor_critic.enable_build_conditioned_durable_value:
                    t_stage_in = DURABLE_STAGE_DIM + rev_dim
                    t_mid = max(16, cfg.ffn_dim // 2)
                    t_zb = cfg.d_model * 2 + cfg.ffn_dim + DURABLE_DECK_SUMMARY_DIM
                    t_bs = t_zb + t_stage_in
                    t_sheads = cfg.n_heads if cfg.d_model % cfg.n_heads == 0 else 1
                    t_sread = t_zb + cfg.d_model + t_stage_in

                    def _make_twin_zero_mlp(
                        in_d: int, mid_d: int, out_d: int, bias_val: float = 0.0
                    ) -> nn.Sequential:
                        fc1 = nn.Linear(in_d, mid_d)
                        nn.init.normal_(fc1.weight, mean=0.0, std=0.08)
                        nn.init.zeros_(fc1.bias)
                        fc_out = nn.Linear(mid_d, out_d)
                        nn.init.zeros_(fc_out.weight)
                        nn.init.constant_(fc_out.bias, bias_val)
                        return nn.Sequential(nn.LayerNorm(in_d), fc1, nn.SiLU(), fc_out)

                    self.durable_twin_hp_build_gate = _make_twin_zero_mlp(
                        t_bs, t_mid, DURABLE_HP_BASIS_DIM, 0.0
                    )
                    self.durable_twin_liquid_build_gate = _make_twin_zero_mlp(
                        t_bs, t_mid, DURABLE_LIQUID_BASIS_DIM, 0.0
                    )
                    self.durable_twin_shop_query_proj = nn.Sequential(
                        nn.LayerNorm(t_bs),
                        nn.Linear(t_bs, cfg.d_model),
                    )
                    self.durable_twin_shop_null = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
                    self.durable_twin_shop_attn = nn.MultiheadAttention(
                        cfg.d_model,
                        t_sheads,
                        dropout=cfg.dropout,
                        batch_first=True,
                    )
                    self.durable_twin_shop_base_head = _make_twin_zero_mlp(t_sread, t_mid, 1, 0.0)
                    self.durable_twin_shop_liquid_gate = _make_twin_zero_mlp(
                        t_sread, t_mid, DURABLE_LIQUID_BASIS_DIM, -6.5
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
        *,
        token_weights: Tensor | None = None,
    ) -> Tensor:
        batch_size = int(world.tokens.shape[0])
        build_mask = self._persistent_token_mask(zone_ids, owner_ids, world.world_mask)
        null_token = self.world_null_token.expand(batch_size, -1, -1).to(dtype=world.tokens.dtype)
        keys = torch.cat([null_token, world.tokens], dim=1)
        queries = self.persistent_build_queries.expand(batch_size, -1, -1).to(
            dtype=world.tokens.dtype
        )
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
        normed_slots: Tensor = self.persistent_build_norm(queries + delta)
        pooled: Tensor = normed_slots.mean(dim=1)
        if self.actor_critic_config.enable_expressive_synergy_backbone:
            slot_delta: Tensor = self.persistent_build_slot_readout(
                normed_slots.reshape(batch_size, -1)
            )
            pooled = pooled + slot_delta
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
        if self.actor_critic_config.enable_debloated_synergy_durable_value:
            canon_def_ids = world.definition_binding_ids.clamp(0, 1)
            canon_rel_ids = world.relation_binding_ids.clamp(0, 1)
        else:
            canon_def_ids = world.definition_binding_ids
            canon_rel_ids = world.relation_binding_ids
        prepared_world = WorldTokenBatch(
            features=features,
            type_ids=world.type_ids,
            role_ids=canon_role_ids,
            owner_ids=world.owner_ids,
            entity_ids=canon_entity_ids,
            entity_aux_ids=canon_aux_ids,
            definition_binding_ids=canon_def_ids,
            relation_binding_ids=canon_rel_ids,
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

        null_token = self.world_null_token.expand(batch_size, -1, -1).to(
            dtype=raw_world_tokens.dtype
        )
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
        latent_queries = (self.latent_queries.expand(batch_size, -1, -1) + domain_context).to(
            dtype=encoded_tokens.dtype
        )
        latent_delta = self._weighted_multihead_attention(
            self.world_to_latent,
            latent_queries,
            encoded_tokens,
            key_log_bias,
        )
        latents = self.world_to_latent_norm(latent_queries + latent_delta)
        latents = self.latent_encoder(latents)
        state_embedding = self._readout_world_state(latents)
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

    def _readout_world_state(self, latents: Tensor) -> Tensor:
        base_state: Tensor = self.state_norm(latents.mean(dim=1))
        if not self.actor_critic_config.enable_expressive_synergy_backbone:
            return base_state
        batch_size = int(latents.shape[0])
        slot_delta: Tensor = self.latent_slot_readout(latents.reshape(batch_size, -1))
        return base_state + slot_delta

    def encode_world(
        self,
        world: WorldTokenBatch,
        domain_ids: Tensor,
        *,
        _validated: bool = False,
    ) -> WorldEncoding:
        encoded = super().encode_world(world, domain_ids, _validated=_validated)
        if not self.actor_critic_config.enable_expressive_synergy_backbone:
            return encoded
        batch_size = int(encoded.latents.shape[0])
        slot_delta: Tensor = self.latent_slot_readout(
            encoded.latents.reshape(batch_size, -1)
        )
        return WorldEncoding(
            latents=encoded.latents,
            state_embedding=encoded.state_embedding + slot_delta,
            tokens=encoded.tokens,
            world_mask=encoded.world_mask,
            entity_ids=encoded.entity_ids,
            entity_aux_ids=encoded.entity_aux_ids,
            definition_binding_ids=encoded.definition_binding_ids,
            relation_binding_ids=encoded.relation_binding_ids,
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

    def _extract_combat_vitals(
        self,
        world: WorldTokenBatch,
        domain_ids: Tensor,
    ) -> Tensor:
        """Extract pure factual combat tactical vitals directly from world tokens.

        Returns a ``[B, COMBAT_VITALS_DIM]`` float tensor that is strictly zero
        on non-combat (macro) rows so macro policy/value behavior is untouched.
        """

        slot_end = min(int(world.features.shape[-1]), NUMERIC_FACT_SLOT_END)
        feats = world.features[:, :, NUMERIC_FACT_SLOT_START:slot_end]
        batch_size, world_cap, num_width = feats.shape
        device = feats.device
        dtype = feats.dtype
        valid = world.mask.bool()
        zeros_bw = torch.zeros((batch_size, world_cap), dtype=dtype, device=device)

        def _col(offset: int) -> Tensor:
            if 0 <= offset < num_width:
                return feats[:, :, offset]
            return zeros_bw

        def _masked_max(mask_bw: Tensor, values_bw: Tensor) -> Tensor:
            return torch.where(mask_bw, values_bw, zeros_bw).amax(dim=1)

        player_mask = world.zone_ids.eq(PLAYER_ZONE_ID) & valid
        combat_mask = world.zone_ids.eq(COMBAT_ZONE_ID) & valid
        combat_or_player = player_mask | combat_mask
        enemy_mask = world.zone_ids.eq(ENEMY_ZONE_ID) & valid
        intent_mask = world.zone_ids.eq(INTENT_ZONE_ID) & valid

        hp_col = _col(PLAYER_HP_OFFSET)
        cur_hp_col = _col(PLAYER_CURRENT_HP_OFFSET)
        max_hp_col = _col(PLAYER_MAX_HP_OFFSET)
        block_col = _col(PLAYER_BLOCK_OFFSET)
        energy_col = _col(COMBAT_ENERGY_OFFSET)
        max_energy_col = _col(COMBAT_MAX_ENERGY_OFFSET)
        round_col = _col(COMBAT_ROUND_OFFSET)
        is_alive_col = _col(ENEMY_IS_ALIVE_OFFSET)
        dmg_col = _col(INTENT_DAMAGE_OFFSET)
        rep_col = _col(INTENT_REPEATS_OFFSET)
        tot_dmg_col = _col(INTENT_TOTAL_DAMAGE_OFFSET)

        hp_log = torch.maximum(
            _masked_max(player_mask, hp_col),
            _masked_max(player_mask, cur_hp_col),
        )
        hp_val = torch.expm1((10.0 * hp_log).clamp(0.0, 8.0)).clamp_min(0.0)
        max_hp_val = torch.expm1(
            (10.0 * _masked_max(player_mask, max_hp_col)).clamp(0.0, 8.0)
        ).clamp_min(1.0)
        hp_ratio = (hp_val / max_hp_val).clamp(0.0, 2.0)
        low_hp_urgency = (0.5 - hp_ratio).clamp_min(0.0) * 2.0
        block_val = torch.expm1(
            (10.0 * _masked_max(player_mask, block_col)).clamp(0.0, 8.0)
        ).clamp_min(0.0)

        energy_val = torch.expm1(
            (10.0 * _masked_max(combat_or_player, energy_col)).clamp(0.0, 5.0)
        ).clamp_min(0.0)
        max_energy_val = torch.expm1(
            (10.0 * _masked_max(combat_or_player, max_energy_col)).clamp(0.0, 5.0)
        ).clamp_min(1.0)
        round_val = torch.expm1(
            (10.0 * _masked_max(combat_or_player, round_col)).clamp(0.0, 6.0)
        ).clamp_min(0.0)

        enemy_hp_bw = torch.expm1((10.0 * hp_col).clamp(0.0, 9.0)).clamp_min(0.0)
        enemy_blk_bw = torch.expm1((10.0 * block_col).clamp(0.0, 8.0)).clamp_min(0.0)
        enemy_alive = enemy_mask & ((is_alive_col > 0.5) | (enemy_hp_bw > 0.5))
        alive_enemy_count = enemy_alive.to(dtype=dtype).sum(dim=1)
        total_enemy_ehp = torch.where(
            enemy_alive, enemy_hp_bw + enemy_blk_bw, zeros_bw
        ).sum(dim=1)
        big_hp = torch.full_like(enemy_hp_bw, 1e6)
        min_enemy_hp_raw = torch.where(enemy_alive, enemy_hp_bw, big_hp).amin(dim=1)
        has_alive_enemy = alive_enemy_count > 0.5
        min_enemy_hp = torch.where(
            has_alive_enemy,
            min_enemy_hp_raw,
            torch.zeros_like(min_enemy_hp_raw),
        )

        # Match intent tokens against alive enemies via decision-local relation IDs
        # in O(B * W * 32) without host sync or O(W^2) memory.
        rel = world.relation_binding_ids
        k_enemy = min(world_cap, 32)
        top_alive_scores, top_enemy_idx = enemy_alive.to(torch.int8).topk(k_enemy, dim=1)
        top_alive_mask = top_alive_scores > 0
        top_enemy_rels = rel.gather(1, top_enemy_idx)
        has_alive_rel = (top_alive_mask & (top_enemy_rels > 1)).any(dim=1, keepdim=True)
        matches_alive_enemy = (
            intent_mask.unsqueeze(2)
            & top_alive_mask.unsqueeze(1)
            & (rel.unsqueeze(2) == top_enemy_rels.unsqueeze(1))
            & (rel.unsqueeze(2) > 1)
        ).any(dim=2)
        active_intent = intent_mask & torch.where(
            has_alive_rel & (rel > 1),
            matches_alive_enemy,
            torch.ones_like(intent_mask),
        )
        per_hit = torch.expm1((10.0 * dmg_col).clamp(0.0, 8.0)).clamp_min(0.0)
        rep_raw = torch.expm1((10.0 * rep_col).clamp(0.0, 5.0)).clamp_min(0.0)
        tot_dmg_slot = torch.expm1((10.0 * tot_dmg_col).clamp(0.0, 8.0)).clamp_min(0.0)
        eff_repeats = torch.where(rep_raw > 0.5, rep_raw, torch.ones_like(rep_raw))
        intent_dmg_bw = torch.where(per_hit > 0.05, per_hit * eff_repeats, tot_dmg_slot)
        total_incoming = torch.where(active_intent, intent_dmg_bw, zeros_bw).sum(dim=1)

        is_attacked = (total_incoming > 0.5).to(dtype=dtype)
        incoming_norm = (total_incoming / 30.0).clamp(0.0, 4.0)
        block_norm = (block_val / 30.0).clamp(0.0, 4.0)
        unblocked_incoming = (total_incoming - block_val).clamp_min(0.0)
        unblocked_norm = (unblocked_incoming / 30.0).clamp(0.0, 4.0)
        needs_block = (unblocked_incoming > 0.5).to(dtype=dtype)
        block_surplus_norm = ((block_val - total_incoming).clamp_min(0.0) / 30.0).clamp(0.0, 4.0)
        unblocked_hp_frac = (unblocked_incoming / hp_val.clamp_min(1.0)).clamp(0.0, 2.0)
        lethal_risk = (unblocked_incoming >= (hp_val.clamp_min(1.0) - 0.5)).to(dtype=dtype) * is_attacked
        energy_norm = (energy_val / 3.0).clamp(0.0, 3.0)
        energy_frac = (energy_val / max_energy_val).clamp(0.0, 2.0)
        round_norm = (round_val / 10.0).clamp(0.0, 3.0)
        alive_enemies_norm = (alive_enemy_count / 3.0).clamp(0.0, 3.0)
        total_enemy_ehp_norm = (total_enemy_ehp / 100.0).clamp(0.0, 4.0)
        min_enemy_hp_scale = torch.tanh(min_enemy_hp / 15.0) * has_alive_enemy.to(dtype=dtype)

        stacked = torch.stack(
            [
                is_attacked,
                incoming_norm,
                block_norm,
                unblocked_norm,
                needs_block,
                block_surplus_norm,
                unblocked_hp_frac,
                lethal_risk,
                hp_ratio,
                low_hp_urgency,
                energy_norm,
                energy_frac,
                round_norm,
                alive_enemies_norm,
                total_enemy_ehp_norm,
                min_enemy_hp_scale,
            ],
            dim=-1,
        )
        combat_row_mask = domain_ids.eq(COMBAT_DOMAIN_ID).to(dtype=dtype).unsqueeze(-1)
        return stacked * combat_row_mask

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

    def _encode_shop_tokens(
        self,
        world: WorldTokenBatch,
    ) -> tuple[Tensor | None, Tensor | None, Tensor]:
        shop_mask_full = (
            world.zone_ids.eq(SHOP_ZONE_ID) | world.zone_ids.eq(SHOP_SUB_ZONE_ID)
        ) & world.mask.bool()
        in_shop = shop_mask_full.any(dim=1)
        if not bool(in_shop.any().item()):
            return None, None, in_shop
        shop_world, _ = self._compact_world_by_mask(world, shop_mask_full)
        shop_valid = shop_world.mask.bool()
        raw_shop = self.token_embedder(
            shop_world.features,
            type_ids=shop_world.type_ids,
            role_ids=shop_world.role_ids,
            owner_ids=shop_world.owner_ids,
            entity_ids=shop_world.entity_ids,
            entity_aux_ids=shop_world.entity_aux_ids,
            zone_ids=shop_world.zone_ids,
            order_ids=shop_world.order_ids,
            segment_id=0,
        )
        shop_tokens = raw_shop * shop_valid.unsqueeze(-1).to(dtype=raw_shop.dtype)
        return shop_tokens, shop_valid, in_shop

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
        card_upg_ids: Tensor | None = None,
        draw_prob: Tensor | None = None,
        deck_size: Tensor | None = None,
        has_deck: Tensor | None = None,
        shop_context: tuple[Tensor | None, Tensor | None, Tensor] | None = None,
        use_twin: bool = False,
    ) -> Tensor:
        with torch.autocast(device_type=world_state_embedding.device.type, enabled=False):
            mp_f = (
                (
                    monotone_parts[0].float(),
                    monotone_parts[1].float(),
                    monotone_parts[2].float(),
                )
                if monotone_parts is not None
                else None
            )
            shop_f = (
                (
                    None if shop_context[0] is None else shop_context[0].float(),
                    shop_context[1],
                    shop_context[2],
                )
                if shop_context is not None
                else None
            )
            return self._durable_readout_fp32(
                world_state_embedding=world_state_embedding.float(),
                build_embedding=None if build_embedding is None else build_embedding.float(),
                vitals_embedding=None if vitals_embedding is None else vitals_embedding.float(),
                raw_vitals=None if raw_vitals is None else raw_vitals.float(),
                progress_vitals=None if progress_vitals is None else progress_vitals.float(),
                monotone_parts=mp_f,
                run_memory=run_memory.float(),
                revival_context=None if revival_context is None else revival_context.float(),
                card_repr=None if card_repr is None else card_repr.float(),
                card_raw_feats=None if card_raw_feats is None else card_raw_feats.float(),
                deck_card_mask=deck_card_mask,
                card_upg_ids=card_upg_ids,
                draw_prob=None if draw_prob is None else draw_prob.float(),
                deck_size=None if deck_size is None else deck_size.float(),
                has_deck=has_deck,
                shop_context=shop_f,
                use_twin=use_twin,
            )

    def _durable_readout_fp32(
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
        card_upg_ids: Tensor | None = None,
        draw_prob: Tensor | None = None,
        deck_size: Tensor | None = None,
        has_deck: Tensor | None = None,
        shop_context: tuple[Tensor | None, Tensor | None, Tensor] | None = None,
        use_twin: bool = False,
    ) -> Tensor:
        ac = self.actor_critic_config
        has_twin = ac.enable_twin_durable_value_heads
        if use_twin and not has_twin:
            raise RuntimeError("use_twin=True requires enable_twin_durable_value_heads=True")
        head_drop = (
            (self.durable_twin_dropout if use_twin else self.durable_twin_head1_dropout)
            if (has_twin and getattr(self, "_twin_dropout_active", False))
            else None
        )
        ws_emb = head_drop(world_state_embedding) if head_drop is not None else world_state_embedding
        b_emb = (
            head_drop(build_embedding)
            if (head_drop is not None and build_embedding is not None)
            else build_embedding
        )
        parts = [ws_emb]
        if b_emb is not None:
            parts.append(b_emb)
        if vitals_embedding is not None:
            parts.append(vitals_embedding)
        parts.append(run_memory)
        rev_scaled = revival_context * 4.0 if revival_context is not None else None
        if rev_scaled is not None:
            parts.append(rev_scaled)
        val_head = self.durable_twin_value_head if use_twin else self.durable_value_head
        out: Tensor = val_head(torch.cat(parts, dim=-1)).squeeze(-1)
        stage_in: Tensor | None = None
        if ac.enable_monotone_deconfounded_durable_value and monotone_parts is not None:
            stage_feats, _, _ = monotone_parts
            stage_in = (
                torch.cat([stage_feats, rev_scaled], dim=-1)
                if rev_scaled is not None
                else stage_feats
            )

        deck_summary = torch.zeros(
            (int(world_state_embedding.shape[0]), DURABLE_DECK_SUMMARY_DIM),
            dtype=out.dtype,
            device=out.device,
        )
        deck_archetype_emb = torch.zeros(
            (int(world_state_embedding.shape[0]), self.config.ffn_dim),
            dtype=out.dtype,
            device=out.device,
        )
        if (
            ac.enable_draw_weighted_durable_value
            and card_repr is not None
            and card_raw_feats is not None
            and draw_prob is not None
            and deck_size is not None
            and has_deck is not None
        ):
            d_norm = self.durable_twin_deck_norm if use_twin else self.durable_deck_norm
            d_card_fc = self.durable_twin_deck_card_fc if use_twin else self.durable_deck_card_fc
            d_raw_fc = self.durable_twin_deck_raw_fc if use_twin else self.durable_deck_raw_fc
            d_ctx_norm = self.durable_twin_deck_ctx_norm if use_twin else self.durable_deck_ctx_norm
            d_ctx_fc = self.durable_twin_deck_ctx_fc if use_twin else self.durable_deck_ctx_fc
            d_out = self.durable_twin_deck_out if use_twin else self.durable_deck_out

            deck_ctx_parts = [ws_emb]
            if vitals_embedding is not None:
                deck_ctx_parts.append(vitals_embedding)
            deck_ctx_parts.append(run_memory)
            if rev_scaled is not None:
                deck_ctx_parts.append(rev_scaled)
            deck_ctx = (
                0.5
                * d_ctx_fc(
                    d_ctx_norm(torch.cat(deck_ctx_parts, dim=-1))
                ).unsqueeze(1)
            )
            card_repr_in = head_drop(card_repr) if head_drop is not None else card_repr
            card_hidden = nn.functional.silu(
                d_card_fc(d_norm(card_repr_in)) + deck_ctx
            )
            if ac.enable_expressive_synergy_backbone:
                syn_batch = int(card_hidden.shape[0])
                syn_card_mask = (
                    deck_card_mask
                    if deck_card_mask is not None
                    else (draw_prob > 0.0)
                )
                syn_mask_f = syn_card_mask.unsqueeze(-1).to(dtype=card_hidden.dtype)
                normed_cards = self.durable_deck_synergy_norm1(card_hidden)
                null_seq = self.durable_deck_synergy_null.expand(syn_batch, -1, -1).to(
                    dtype=card_hidden.dtype
                )
                normed_seq = torch.cat([null_seq, normed_cards], dim=1)
                card_syn_w = torch.where(
                    syn_card_mask,
                    (DECK_REFERENCE_SIZE * draw_prob.clamp_min(0.0)).to(dtype=card_hidden.dtype),
                    torch.zeros_like(draw_prob, dtype=card_hidden.dtype),
                )
                null_syn_w = torch.ones(
                    (syn_batch, 1),
                    dtype=card_hidden.dtype,
                    device=card_hidden.device,
                )
                full_syn_w = torch.cat([null_syn_w, card_syn_w], dim=1)
                syn_neg_val = -1e4 if card_hidden.dtype == torch.float16 else -1e9
                syn_key_log_bias = torch.where(
                    full_syn_w > 0.0,
                    torch.log(full_syn_w.clamp_min(1e-12)),
                    torch.full_like(full_syn_w, syn_neg_val),
                )
                syn_delta = self._weighted_multihead_attention(
                    self.durable_deck_synergy_attn,
                    normed_cards,
                    normed_seq,
                    syn_key_log_bias,
                )
                if ac.enable_debloated_synergy_durable_value:
                    syn_delta = 0.5 * torch.tanh(2.0 * syn_delta)
                card_hidden = card_hidden + syn_delta * syn_mask_f
                syn_ffn_delta: Tensor = self.durable_deck_synergy_ffn(card_hidden)
                if ac.enable_debloated_synergy_durable_value:
                    syn_ffn_delta = 0.5 * torch.tanh(2.0 * syn_ffn_delta)
                card_hidden = card_hidden + syn_ffn_delta * syn_mask_f
            raw_hidden = nn.functional.silu(d_raw_fc(card_raw_feats))
            bilinear_card = torch.cat(
                [card_hidden, raw_hidden, card_hidden * raw_hidden],
                dim=-1,
            )
            deck_archetype_emb = (
                torch.einsum(
                    "bw,bwd->bd",
                    draw_prob.to(dtype=card_hidden.dtype),
                    card_hidden,
                )
                * has_deck.to(dtype=card_hidden.dtype).unsqueeze(-1)
            ).to(dtype=out.dtype)
            card_utilities = d_out(bilinear_card).squeeze(-1)
            expected_card_utility = torch.einsum("bw,bw->b", draw_prob, card_utilities)
            marginal_deck_sum: Tensor | None = None
            if ac.enable_marginal_card_id_durable_value and card_upg_ids is not None:
                d_card_id_emb = (
                    self.durable_twin_deck_card_id_emb
                    if use_twin
                    else self.durable_deck_card_id_emb
                )
                d_marg_id_lin = (
                    self.durable_twin_deck_marginal_id_linear
                    if use_twin
                    else self.durable_deck_marginal_id_linear
                )
                d_marg_mlp = (
                    self.durable_twin_deck_marginal_mlp
                    if use_twin
                    else self.durable_deck_marginal_mlp
                )
                marg_card_mask = (
                    deck_card_mask
                    if deck_card_mask is not None
                    else (draw_prob > 0.0)
                )
                safe_n_marg = deck_size.clamp_min(1.0)
                card_count_w = torch.where(
                    marg_card_mask,
                    (draw_prob * safe_n_marg.unsqueeze(1)).to(dtype=out.dtype),
                    torch.zeros_like(draw_prob, dtype=out.dtype),
                )
                bounded_upg_ids = card_upg_ids.long().clamp(
                    0, d_card_id_emb.num_embeddings - 1
                )
                id_emb = d_card_id_emb(bounded_upg_ids).to(dtype=card_hidden.dtype)
                if head_drop is not None:
                    id_emb = head_drop(id_emb)
                marg_in = torch.cat([id_emb, card_hidden, raw_hidden], dim=-1)
                marg_raw = (d_marg_id_lin(id_emb) + d_marg_mlp(marg_in)).squeeze(-1)
                marginal_card_delta = 0.30 * torch.tanh(marg_raw / 0.30)
                marginal_deck_sum = (
                    torch.einsum(
                        "bw,bw->b",
                        card_count_w,
                        marginal_card_delta.to(dtype=out.dtype),
                    )
                    * has_deck.to(dtype=out.dtype)
                )
            if ac.enable_monotone_deconfounded_durable_value:
                d_tilted = (
                    self.durable_twin_deck_tilted_out if use_twin else self.durable_deck_tilted_out
                )
                d_lin_logit = (
                    self.durable_twin_deck_linear_logit
                    if use_twin
                    else self.durable_deck_linear_logit
                )
                d_tilt_logits = (
                    self.durable_twin_deck_tilt_logits
                    if use_twin
                    else self.durable_deck_tilt_logits
                )
                cyc_gate = self.durable_twin_cycle_gate if use_twin else self.durable_cycle_gate

                has_deck_f = has_deck.to(dtype=out.dtype)
                eff_card_mask = (
                    deck_card_mask
                    if deck_card_mask is not None
                    else (draw_prob > 0.0)
                )
                u_tilt_all = 2.5 * torch.tanh(
                    (card_utilities.unsqueeze(-1) + d_tilted(bilinear_card))
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
                    nn.functional.softplus(d_lin_logit)
                    if ac.enable_balanced_deconfounded_durable_value
                    else torch.sigmoid(d_lin_logit)
                )
                w_tilt = nn.functional.softplus(d_tilt_logits)
                deck_utility = alpha_lin * expected_card_utility + (w_tilt * u_betas).sum(dim=-1)
                safe_n = deck_size.clamp_min(1.0)
                deck_summary = torch.stack(
                    [
                        expected_card_utility * has_deck_f,
                        u_betas[:, 0],
                        u_betas[:, 1],
                        ((safe_n - DECK_REFERENCE_SIZE) / 16.0).clamp(-1.0, 4.0) * has_deck_f,
                    ],
                    dim=-1,
                ).to(dtype=out.dtype)
                if stage_in is not None:
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
                    cyc_logits = cyc_gate(stage_in)
                    if ac.enable_debloated_synergy_durable_value:
                        cyc_logits = cyc_logits + 1.5
                    w_cycle = nn.functional.softplus(cyc_logits)
                    if ac.enable_active_durable_vitals_gates:
                        act_cyc_gate = (
                            self.durable_active_twin_cycle_gate
                            if use_twin
                            else self.durable_active_cycle_gate
                        )
                        raw_act_cyc = act_cyc_gate(stage_in)
                        eff_act_cyc = (
                            nn.functional.leaky_relu(raw_act_cyc + 4.0, negative_slope=0.1) - 4.0
                        ).clamp(max=3.5)
                        w_cycle = w_cycle + nn.functional.softplus(eff_act_cyc)
                    v_cycle = -(w_cycle * bloat_basis).sum(dim=-1)
                    deck_utility = deck_utility + v_cycle
                if marginal_deck_sum is not None:
                    deck_utility = deck_utility + marginal_deck_sum
                out = out + deck_utility
            elif ac.enable_deconfounded_durable_value:
                out = out + expected_card_utility + (
                    marginal_deck_sum if marginal_deck_sum is not None else 0.0
                )
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
                out = (
                    out
                    + expected_card_utility
                    + deck_size_delta
                    + (marginal_deck_sum if marginal_deck_sum is not None else 0.0)
                )

        if ac.enable_monotone_deconfounded_durable_value:
            if monotone_parts is not None and stage_in is not None:
                _, hp_basis, liquid_basis = monotone_parts
                v_stage = self.durable_stage_baseline_head(stage_in).squeeze(-1)
                hp_gate = self.durable_twin_hp_gate if use_twin else self.durable_hp_gate
                liq_gate = self.durable_twin_liquid_gate if use_twin else self.durable_liquid_gate
                hp_logits = hp_gate(stage_in)
                liq_logits = liq_gate(stage_in)
                if ac.enable_build_conditioned_durable_value:
                    b_vec = (
                        b_emb
                        if b_emb is not None
                        else torch.zeros_like(ws_emb)
                    )
                    z_build = torch.cat(
                        [ws_emb, b_vec, deck_archetype_emb, deck_summary], dim=-1
                    )
                    build_stage_in = torch.cat([z_build, stage_in], dim=-1)
                    hp_b_gate = (
                        self.durable_twin_hp_build_gate
                        if use_twin
                        else self.durable_hp_build_gate
                    )
                    liq_b_gate = (
                        self.durable_twin_liquid_build_gate
                        if use_twin
                        else self.durable_liquid_build_gate
                    )
                    hp_logits = hp_logits + hp_b_gate(build_stage_in)
                    liq_logits = liq_logits + liq_b_gate(build_stage_in)
                    if shop_context is not None:
                        shop_tokens, shop_valid, in_shop = shop_context
                        if shop_tokens is not None and shop_valid is not None:
                            s_q_proj = (
                                self.durable_twin_shop_query_proj
                                if use_twin
                                else self.durable_shop_query_proj
                            )
                            s_null = (
                                self.durable_twin_shop_null
                                if use_twin
                                else self.durable_shop_null
                            )
                            s_attn = (
                                self.durable_twin_shop_attn
                                if use_twin
                                else self.durable_shop_attn
                            )
                            s_base_h = (
                                self.durable_twin_shop_base_head
                                if use_twin
                                else self.durable_shop_base_head
                            )
                            s_liq_g = (
                                self.durable_twin_shop_liquid_gate
                                if use_twin
                                else self.durable_shop_liquid_gate
                            )
                            s_bsz = int(z_build.shape[0])
                            q_shop = s_q_proj(build_stage_in).unsqueeze(1)
                            kv_shop = torch.cat(
                                [
                                    s_null.expand(s_bsz, -1, -1).to(dtype=shop_tokens.dtype),
                                    shop_tokens,
                                ],
                                dim=1,
                            )
                            s_neg = -1e4 if shop_tokens.dtype == torch.float16 else -1e9
                            null_v = torch.ones(
                                (s_bsz, 1), dtype=torch.bool, device=shop_tokens.device
                            )
                            kv_valid = torch.cat([null_v, shop_valid], dim=1)
                            s_log_bias = torch.where(
                                kv_valid,
                                torch.zeros((s_bsz, int(kv_valid.shape[1])), dtype=shop_tokens.dtype, device=shop_tokens.device),
                                torch.full((s_bsz, int(kv_valid.shape[1])), s_neg, dtype=shop_tokens.dtype, device=shop_tokens.device),
                            )
                            shop_pooled = self._weighted_multihead_attention(
                                s_attn, q_shop, kv_shop, s_log_bias
                            ).squeeze(1)
                            shop_in = torch.cat([z_build, shop_pooled, stage_in], dim=-1)
                            v_shop_raw = s_base_h(shop_in).squeeze(-1) + (
                                nn.functional.softplus(s_liq_g(shop_in)) * liquid_basis
                            ).sum(dim=-1)
                            out = out + v_shop_raw * in_shop.to(dtype=out.dtype)
                w_hp = nn.functional.softplus(hp_logits)
                w_liq = nn.functional.softplus(liq_logits)
                if ac.enable_active_durable_vitals_gates:
                    act_hp_gate = (
                        self.durable_active_twin_hp_gate
                        if use_twin
                        else self.durable_active_hp_gate
                    )
                    act_liq_gate = (
                        self.durable_active_twin_liquid_gate
                        if use_twin
                        else self.durable_active_liquid_gate
                    )
                    raw_act_hp = act_hp_gate(stage_in)
                    raw_act_liq = act_liq_gate(stage_in)
                    eff_act_hp = (
                        nn.functional.leaky_relu(raw_act_hp + 4.0, negative_slope=0.1) - 4.0
                    ).clamp(max=3.5)
                    eff_act_liq = (
                        nn.functional.leaky_relu(raw_act_liq + 4.0, negative_slope=0.1) - 4.0
                    ).clamp(max=3.5)
                    w_hp = w_hp + nn.functional.softplus(eff_act_hp)
                    w_liq = w_liq + nn.functional.softplus(eff_act_liq)
                stage_term = (
                    v_stage
                    if ac.enable_balanced_deconfounded_durable_value
                    else v_stage.detach()
                )
                out = (
                    out
                    + stage_term
                    + (w_hp * hp_basis).sum(dim=-1)
                    + (w_liq * liquid_basis).sum(dim=-1)
                )
        elif ac.enable_deconfounded_durable_value:
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
            if ac.enable_persistent_only_durable_value:
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

    def _compute_persistent_durable_value_pair(
        self,
        world: WorldTokenBatch,
        recurrent_state: Tensor | None,
        *,
        critic_revival_context: Tensor | None,
        validate: bool,
    ) -> tuple[Tensor, Tensor, Tensor]:
        trunk_ctx = (
            torch.no_grad()
            if getattr(self, "_detach_value_heads_from_trunk", False)
            else contextlib.nullcontext()
        )
        with trunk_ctx:
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
            card_upg_ids: Tensor | None = None
            draw_prob: Tensor | None = None
            deck_size: Tensor | None = None
            has_deck: Tensor | None = None
            token_weights: Tensor | None = None
            prep: _PreparedPersistentWorld | None = None
            if self.actor_critic_config.enable_draw_weighted_durable_value:
                prep = self._prepare_draw_weighted_persistent_world(world)
                pers_world = prep.world
                token_weights = prep.token_weights
                deck_card_mask = prep.deck_card_mask
                draw_prob = prep.draw_prob
                deck_size = prep.deck_size
                has_deck = prep.has_deck
                if self.actor_critic_config.enable_marginal_card_id_durable_value:
                    f_w = pers_world.features
                    f_dim = int(f_w.shape[-1])
                    is_upg = torch.zeros_like(deck_card_mask, dtype=torch.bool)
                    if f_dim > CARD_IS_UPGRADED_SLOT:
                        is_upg = is_upg | (f_w[:, :, CARD_IS_UPGRADED_SLOT] > 0.5)
                    if f_dim > CARD_UPGRADED_SLOT:
                        is_upg = is_upg | (f_w[:, :, CARD_UPGRADED_SLOT] > 0.5)
                    if f_dim > CARD_UPGRADE_LEVEL_SLOT:
                        is_upg = is_upg | (f_w[:, :, CARD_UPGRADE_LEVEL_SLOT] > 1e-4)
                    if f_dim > CARD_CURRENT_UPGRADE_LEVEL_SLOT:
                        is_upg = is_upg | (f_w[:, :, CARD_CURRENT_UPGRADE_LEVEL_SLOT] > 1e-4)
                    if f_dim > CARD_UPGRADES_SLOT:
                        is_upg = is_upg | (f_w[:, :, CARD_UPGRADES_SLOT] > 1e-4)
                    bounded_eids = pers_world.entity_ids.long().clamp(
                        0, self.config.entity_vocab_size - 1
                    )
                    card_upg_ids = torch.where(
                        deck_card_mask,
                        bounded_eids * 2 + is_upg.long(),
                        torch.zeros_like(bounded_eids),
                    )
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
                    torch.cat(
                        [
                            pers_enc.tokens,
                            raw_world_tokens.to(dtype=pers_enc.tokens.dtype),
                            child_tok_sum,
                        ],
                        dim=-1,
                    )
                    * card_mask_f
                )
                card_raw_feats = (
                    torch.cat([scaled_feats, child_feat_sum], dim=-1)
                    * card_mask_f
                )
                max_cards = max(1, int(deck_card_mask.long().sum(dim=1).max().item()))
                if max_cards < int(card_repr.shape[1]):
                    card_idx = torch.argsort(
                        (~deck_card_mask).to(dtype=torch.int8),
                        dim=1,
                        stable=True,
                    )[:, :max_cards]
                    card_repr = card_repr.gather(
                        1,
                        card_idx.unsqueeze(-1).expand(-1, -1, int(card_repr.shape[-1])),
                    )
                    card_raw_feats = card_raw_feats.gather(
                        1,
                        card_idx.unsqueeze(-1).expand(-1, -1, int(card_raw_feats.shape[-1])),
                    )
                    deck_card_mask = deck_card_mask.gather(1, card_idx)
                    if card_upg_ids is not None:
                        card_upg_ids = card_upg_ids.gather(1, card_idx)
                    draw_prob = draw_prob.gather(1, card_idx)
            else:
                pers_world = self._mask_world_to_persistent(world)
                pers_enc = self.encode_world(pers_world, zero_domains, _validated=not validate)
            run_prev_for_durable = (
                torch.zeros_like(previous_run)
                if self.actor_critic_config.enable_monotone_deconfounded_durable_value
                else previous_run
            )
            if (
                self.actor_critic_config.enable_debloated_synergy_durable_value
                and prep is not None
            ):
                non_deck_mask = (
                    pers_world.mask.bool()
                    & (~pers_world.zone_ids.eq(DECK_ZONE_ID))
                    & (~pers_world.zone_ids.eq(MODIFIER_ZONE_ID))
                    & (~prep.child_uv.any(dim=2))
                )
                non_deck_world, _ = self._compact_world_by_mask(pers_world, non_deck_mask)
                base_enc = self.encode_world(
                    non_deck_world,
                    zero_domains,
                    _validated=not validate,
                )
                base_state_emb = base_enc.state_embedding
                with torch.autocast(device_type=base_state_emb.device.type, enabled=False):
                    pers_run = self.run_recurrent_norm(
                        self.run_recurrent_cell(base_state_emb.float(), run_prev_for_durable.float())
                    )
                build_emb = (
                    self._pool_persistent_build(
                        base_enc,
                        non_deck_world.zone_ids,
                        non_deck_world.owner_ids,
                    )
                    if self.actor_critic_config.enable_persistent_build_context
                    else None
                )
            else:
                base_state_emb = pers_enc.state_embedding
                with torch.autocast(device_type=base_state_emb.device.type, enabled=False):
                    pers_run = self.run_recurrent_norm(
                        self.run_recurrent_cell(base_state_emb.float(), run_prev_for_durable.float())
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
                    deck_proj = self.persistent_build_deck_proj(card_repr)
                    build_emb = build_emb + torch.einsum(
                        "bw,bwd->bd",
                        draw_prob.to(dtype=deck_proj.dtype),
                        deck_proj,
                    ).to(dtype=build_emb.dtype)
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
            shop_ctx = (
                self._encode_shop_tokens(world)
                if self.actor_critic_config.enable_build_conditioned_durable_value
                else None
            )
        v1 = self._durable_readout(
            world_state_embedding=base_state_emb,
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
            card_upg_ids=card_upg_ids,
            draw_prob=draw_prob,
            deck_size=deck_size,
            has_deck=has_deck,
            shop_context=shop_ctx,
            use_twin=False,
        )
        if not self.actor_critic_config.enable_twin_durable_value_heads:
            return v1, v1, pers_run
        v2 = self._durable_readout(
            world_state_embedding=base_state_emb,
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
            card_upg_ids=card_upg_ids,
            draw_prob=draw_prob,
            deck_size=deck_size,
            has_deck=has_deck,
            shop_context=shop_ctx,
            use_twin=True,
        )
        return v1, v2, pers_run

    def _compute_persistent_durable_value(
        self,
        world: WorldTokenBatch,
        recurrent_state: Tensor | None,
        *,
        critic_revival_context: Tensor | None,
        validate: bool,
    ) -> tuple[Tensor, Tensor]:
        v1, v2, pers_run = self._compute_persistent_durable_value_pair(
            world,
            recurrent_state,
            critic_revival_context=critic_revival_context,
            validate=validate,
        )
        if self.actor_critic_config.enable_twin_durable_value_heads:
            return 0.5 * (v1 + v2), pers_run
        return v1, pers_run

    def forward_durable_value_pair(
        self,
        world: WorldTokenBatch,
        domain_ids: Tensor,
        recurrent_state: Tensor | None = None,
        *,
        critic_revival_context: Tensor | None = None,
        validate: bool = True,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return ``(v1 [B], v2 [B], next_run_memory [B, H/2])`` for Twin Durable Heads."""
        if not self.actor_critic_config.enable_durable_value_head:
            raise RuntimeError("forward_durable_value_pair requires enable_durable_value_head=True")
        if validate:
            world.validate(self.config)
        if self.actor_critic_config.enable_persistent_only_durable_value:
            return self._compute_persistent_durable_value_pair(
                world,
                recurrent_state,
                critic_revival_context=critic_revival_context,
                validate=validate,
            )
        val, next_run = self.forward_durable_value(
            world,
            domain_ids,
            recurrent_state,
            critic_revival_context=critic_revival_context,
            validate=False,
        )
        return val, val, next_run

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
        with torch.autocast(device_type=world.features.device.type, enabled=False):
            prev_run_f = previous_run.float()
            run_update = self.run_recurrent_norm(
                self.run_recurrent_cell(world_enc.state_embedding.float(), prev_run_f)
            )
            next_run = torch.where(combat_domain, prev_run_f, run_update)
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
            macro_lse = (
                torch.logsumexp(scoped_logits.detach(), dim=-1)
                if getattr(self, "_detach_macro_branch_prior_lse", False)
                else combat_lse
            )
            macro_prior = macro_lse - branch_counts.squeeze(-1).log()
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
    ) -> tuple[
        HybridRepresentationOutput,
        Tensor | None,
        Tensor | None,
        Tensor | None,
        Tensor | None,
    ]:
        has_build = self.actor_critic_config.enable_persistent_build_context
        has_vitals = self.actor_critic_config.enable_direct_vitals_context
        has_combat_vitals = self.actor_critic_config.enable_direct_combat_vitals_context
        if not has_build and not has_vitals and not has_combat_vitals:
            return (
                self.forward_representation(batch, recurrent_state, validate=validate),
                None,
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
        combat_vitals: Tensor | None = (
            self._extract_combat_vitals(batch.world, batch.domain_ids)
            if has_combat_vitals
            else None
        )
        mask = candidates.action_mask.bool()
        cfg = self.config
        memory_half = cfg.recurrent_hidden_dim // 2
        previous_run = recurrent_state[:, :memory_half]
        previous_combat = recurrent_state[:, memory_half:]
        combat_domain = batch.domain_ids.eq(COMBAT_DOMAIN_ID).unsqueeze(-1)

        with torch.autocast(device_type=world.state_embedding.device.type, enabled=False):
            ws_f = world.state_embedding.float()
            prev_run_f = previous_run.float()
            prev_com_f = previous_combat.float()
            run_update = self.run_recurrent_norm(
                self.run_recurrent_cell(ws_f, prev_run_f)
            )
            next_run = torch.where(combat_domain, prev_run_f, run_update)
            combat_update = self.combat_recurrent_norm(
                self.combat_recurrent_cell(ws_f, prev_com_f)
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
        if combat_vitals is not None:
            state_token = state_token + (
                self.combat_vitals_to_state(combat_vitals.to(dtype=state_token.dtype))
                * combat_domain.to(dtype=state_token.dtype)
            )
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
            macro_out = self.macro_stream(
                self.macro_stream_norm(stream_tokens[macro_rows]),
                src_key_padding_mask=padding_mask[macro_rows],
            )
            selected[macro_rows] = macro_out.to(dtype=selected.dtype)
        if bool(combat_rows.any().item()):
            combat_out = self.combat_stream(
                self.combat_stream_norm(stream_tokens[combat_rows]),
                src_key_padding_mask=padding_mask[combat_rows],
            )
            selected[combat_rows] = combat_out.to(dtype=selected.dtype)
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
            combat_vitals,
        )

    def _apply_full_hp_campfire_heal_suppression(
        self,
        *,
        policy_logits: Tensor,
        batch: GroundedCandidateBatch,
        action_mask: Tensor,
    ) -> Tensor:
        """Suppress zero-gain Campfire HEAL (`rest`) when `hp >= max_hp - 1` and another legal action exists."""
        world_valid = batch.world.mask.bool()
        is_rest_surface = (~batch.domain_ids.eq(COMBAT_DOMAIN_ID)) & (
            batch.macro_economic_surface_ids.eq(MACRO_ECONOMIC_SURFACE_REST)
            | (batch.world.zone_ids.eq(REST_SITE_ZONE_ID) & world_valid).any(dim=1)
        )
        if not bool(is_rest_surface.any().item()):
            return policy_logits

        w_feats = batch.world.features
        f_dim = int(w_feats.shape[-1])
        hp_slot = NUMERIC_FACT_SLOT_START + PLAYER_HP_OFFSET
        cur_hp_slot = NUMERIC_FACT_SLOT_START + PLAYER_CURRENT_HP_OFFSET
        max_hp_slot = NUMERIC_FACT_SLOT_START + PLAYER_MAX_HP_OFFSET
        if f_dim <= max(hp_slot, cur_hp_slot, max_hp_slot):
            return policy_logits

        player_mask = batch.world.zone_ids.eq(PLAYER_ZONE_ID) & world_valid
        zeros_bw = torch.zeros_like(w_feats[:, :, 0])
        hp_log = torch.where(
            player_mask,
            torch.maximum(w_feats[:, :, hp_slot], w_feats[:, :, cur_hp_slot]),
            zeros_bw,
        ).amax(dim=1)
        max_hp_log = torch.where(
            player_mask,
            w_feats[:, :, max_hp_slot],
            zeros_bw,
        ).amax(dim=1)
        hp_val = torch.expm1((10.0 * hp_log.float()).clamp(0.0, 8.0))
        max_hp_val = torch.expm1((10.0 * max_hp_log.float()).clamp(0.0, 8.0))
        is_full_hp = (max_hp_val >= 2.0) & (hp_val >= (max_hp_val - 1.25))
        if not bool((is_rest_surface & is_full_hp).any().item()):
            return policy_logits

        is_heal_cand = batch.candidates.role_ids.eq(self._rest_heal_role_id)
        loc_feats = batch.candidates.local_features
        if int(loc_feats.shape[-1]) > CANDIDATE_HEAL_AMOUNT_SLOT:
            loc_heal = torch.where(
                batch.candidates.local_mask.bool(),
                loc_feats[:, :, :, CANDIDATE_HEAL_AMOUNT_SLOT],
                torch.zeros_like(loc_feats[:, :, :, 0]),
            ).amax(dim=-1)
            is_heal_cand = is_heal_cand | (loc_heal > 1e-4)
        is_heal_cand = action_mask & is_heal_cand
        has_other_legal = (action_mask & (~is_heal_cand)).any(dim=1)
        suppress_mask = (
            (is_rest_surface & is_full_hp & has_other_legal).unsqueeze(1) & is_heal_cand
        )
        if not bool(suppress_mask.any().item()):
            return policy_logits
        return torch.where(suppress_mask, policy_logits - 25.0, policy_logits)

    def forward(  # type: ignore[override]
        self,
        batch: GroundedCandidateBatch,
        recurrent_state: Tensor | None = None,
        *,
        critic_revival_context: Tensor | None = None,
        compute_durable_value: bool = True,
        policy_detach_trunk_mask: Tensor | None = None,
        validate: bool = True,
    ) -> HybridActorCriticOutput:
        (
            base,
            build_emb,
            vitals_emb,
            raw_vitals,
            combat_vitals,
        ) = self._forward_representation_with_build(
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
        with torch.autocast(device_type=batch.world.features.device.type, enabled=False):
            joint_state_f = base.joint_state_embedding.float()
            cand_emb_f = base.candidate_embeddings.float()
            rec_state_f = base.recurrent_state.float()
            vitals_emb_f = None if vitals_emb is None else vitals_emb.float()
            raw_vitals_f = None if raw_vitals is None else raw_vitals.float()
            combat_vitals_f = None if combat_vitals is None else combat_vitals.float()
            rev_ctx_f = None if rev_ctx is None else rev_ctx.float()

            has_pol_detach = policy_detach_trunk_mask is not None and bool(
                policy_detach_trunk_mask.any().item()
            )
            if has_pol_detach:
                assert policy_detach_trunk_mask is not None
                dm_2d = policy_detach_trunk_mask.to(
                    device=joint_state_f.device, dtype=torch.bool
                ).view(-1, 1)
                dm_3d = dm_2d.unsqueeze(-1)
                pol_joint_state_f = torch.where(dm_2d, joint_state_f.detach(), joint_state_f)
                pol_cand_emb_f = torch.where(dm_3d, cand_emb_f.detach(), cand_emb_f)
                pol_rec_state_f = torch.where(dm_2d, rec_state_f.detach(), rec_state_f)
                pol_vitals_emb_f = (
                    None
                    if vitals_emb_f is None
                    else torch.where(dm_2d, vitals_emb_f.detach(), vitals_emb_f)
                )
            else:
                dm_3d = None
                pol_joint_state_f = joint_state_f
                pol_cand_emb_f = cand_emb_f
                pol_rec_state_f = rec_state_f
                pol_vitals_emb_f = vitals_emb_f

            state = pol_joint_state_f.unsqueeze(1).expand(-1, int(action_count), -1)
            memory = pol_rec_state_f.unsqueeze(1).expand(-1, int(action_count), -1)
            raw_logits = self.candidate_policy_head(
                torch.cat([pol_cand_emb_f, state, memory], dim=-1)
            ).squeeze(-1)
            if self.actor_critic_config.enable_role_conditioned_candidate_policy:
                bounded_roles = batch.candidates.role_ids.long().clamp(
                    min=0,
                    max=self.config.role_vocab_size - 1,
                )
                role_emb = self.token_embedder.role_embedding(bounded_roles).float()
                if dm_3d is not None:
                    role_emb = torch.where(dm_3d, role_emb.detach(), role_emb)
                cand_modulated = pol_cand_emb_f * torch.tanh(
                    self.candidate_role_gate(role_emb)
                )
                role_delta = self.candidate_role_policy_head(
                    torch.cat(
                        [cand_modulated, pol_cand_emb_f, role_emb, state],
                        dim=-1,
                    )
                ).squeeze(-1)
                raw_logits = raw_logits + role_delta
            if combat_vitals_f is not None:
                combat_domain_f = batch.domain_ids.eq(COMBAT_DOMAIN_ID).to(
                    dtype=raw_logits.dtype
                ).unsqueeze(-1)
                cand_cv_norm = self.combat_vitals_policy_norm(pol_cand_emb_f)
                cv_aug = torch.cat(
                    [torch.ones_like(combat_vitals_f[:, :1]), combat_vitals_f],
                    dim=-1,
                )
                cv_bilinear = (
                    self.combat_vitals_policy_bilinear(cand_cv_norm)
                    * cv_aug.unsqueeze(1)
                ).sum(dim=-1)
                cv_exp = combat_vitals_f.unsqueeze(1).expand(-1, int(action_count), -1)
                cv_mlp = self.combat_vitals_policy_head(
                    torch.cat([cand_cv_norm, cv_exp], dim=-1)
                ).squeeze(-1)
                raw_logits = raw_logits + (cv_bilinear + cv_mlp) * combat_domain_f
            if self.actor_critic_config.enable_hierarchical_branch_policy:
                salient_for_actor = (
                    raw_vitals_f[:, -SALIENT_VITALS_DIM:]
                    if (
                        raw_vitals_f is not None
                        and self.actor_critic_config.enable_persistent_only_durable_value
                    )
                    else None
                )
                policy_logits = self._hierarchical_policy_logits(
                    raw_candidate_logits=raw_logits,
                    candidate_embeddings=pol_cand_emb_f,
                    joint_state_embedding=pol_joint_state_f,
                    vitals_embedding=pol_vitals_emb_f,
                    salient_vitals=salient_for_actor,
                    recurrent_state=pol_rec_state_f,
                    policy_branch_ids=batch.candidates.policy_branch_ids,
                    action_mask=base.action_mask,
                    domain_ids=batch.domain_ids,
                )
            else:
                policy_logits = raw_logits.masked_fill(~base.action_mask, 0.0)
            if (
                self.actor_critic_config.suppress_full_hp_campfire_heal
                or bool(getattr(self, "_suppress_full_hp_campfire_heal", False))
            ):
                policy_logits = self._apply_full_hp_campfire_heal_suppression(
                    policy_logits=policy_logits,
                    batch=batch,
                    action_mask=base.action_mask,
                )
            detach_val = bool(getattr(self, "_detach_value_heads_from_trunk", False))
            val_joint = joint_state_f.detach() if detach_val else joint_state_f
            val_rec = rec_state_f.detach() if detach_val else rec_state_f
            value = self.state_value_head(
                torch.cat([val_joint, val_rec], dim=-1)
            ).squeeze(-1)
            combat_value: Tensor | None = None
            if self.actor_critic_config.enable_local_combat_value_head:
                cv_pred: Tensor = self.local_combat_value_head(
                    torch.cat([val_joint, val_rec], dim=-1)
                ).squeeze(-1)
                if combat_vitals_f is not None:
                    combat_row_f = batch.domain_ids.eq(COMBAT_DOMAIN_ID).to(
                        dtype=cv_pred.dtype
                    )
                    cv_pred = cv_pred + (
                        self.local_combat_value_vitals_linear(combat_vitals_f).squeeze(-1)
                        + self.local_combat_value_vitals_head(combat_vitals_f).squeeze(-1)
                    ) * combat_row_f
                combat_value = cv_pred
            if rev_ctx_f is not None:
                rev_scaled = rev_ctx_f * 4.0
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
                    world_state_embedding=(
                        base.world_state_embedding.detach() if detach_val else base.world_state_embedding
                    ),
                    build_embedding=(
                        None if build_emb is None else (build_emb.detach() if detach_val else build_emb)
                    ),
                    vitals_embedding=(
                        None if vitals_emb is None else (vitals_emb.detach() if detach_val else vitals_emb)
                    ),
                    raw_vitals=None if raw_vitals is None else (raw_vitals.detach() if detach_val else raw_vitals),
                    run_memory=val_rec[:, :memory_half],
                    revival_context=rev_ctx_f,
                )
        output = HybridActorCriticOutput(
            policy_logits=policy_logits,
            value=value,
            recurrent_state=rec_state_f,
            joint_state_embedding=joint_state_f,
            world_state_embedding=base.world_state_embedding.float(),
            candidate_embeddings=cand_emb_f,
            action_mask=base.action_mask,
            durable_value=durable_value,
            combat_value=combat_value,
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
    if (
        model.actor_critic_config.enable_twin_durable_value_heads
        and "durable_twin_value_head.1.weight" in result.missing_keys
    ):
        # Warm-start Twin Head 2 from Head 1's inherited readout + small deterministic symmetry-breaking noise
        twin_map = (
            ("durable_value_head.", "durable_twin_value_head."),
            ("durable_hp_gate.", "durable_twin_hp_gate."),
            ("durable_liquid_gate.", "durable_twin_liquid_gate."),
            ("durable_cycle_gate.", "durable_twin_cycle_gate."),
            ("durable_deck_norm.", "durable_twin_deck_norm."),
            ("durable_deck_card_fc.", "durable_twin_deck_card_fc."),
            ("durable_deck_raw_fc.", "durable_twin_deck_raw_fc."),
            ("durable_deck_ctx_norm.", "durable_twin_deck_ctx_norm."),
            ("durable_deck_ctx_fc.", "durable_twin_deck_ctx_fc."),
            ("durable_deck_out.", "durable_twin_deck_out."),
            ("durable_deck_tilted_out.", "durable_twin_deck_tilted_out."),
            ("durable_deck_linear_logit", "durable_twin_deck_linear_logit"),
            ("durable_deck_tilt_logits", "durable_twin_deck_tilt_logits"),
        )
        param_dict = dict(model.named_parameters())
        gen = torch.Generator(device="cpu")
        gen.manual_seed(20261117)
        with torch.no_grad():
            for dst_name, dst_param in sorted(param_dict.items()):
                if not dst_name.startswith("durable_twin_"):
                    continue
                for src_pref, dst_pref in twin_map:
                    if dst_name.startswith(dst_pref):
                        src_name = src_pref + dst_name[len(dst_pref) :]
                        src_param = param_dict.get(src_name)
                        if src_param is not None and src_param.shape == dst_param.shape:
                            dst_param.copy_(src_param)
                            if dst_param.ndim == 2:
                                noise_mult = (
                                    0.05
                                    if dst_name.startswith(
                                        (
                                            "durable_twin_value_head.",
                                            "durable_twin_deck_out.",
                                            "durable_twin_deck_tilted_out.",
                                        )
                                    )
                                    else 0.02
                                )
                                scale = (
                                    float(src_param.detach().std().clamp_min(1e-3).item())
                                    * noise_mult
                                )
                                noise = torch.randn(
                                    dst_param.shape, generator=gen, dtype=torch.float32
                                ).to(device=dst_param.device, dtype=dst_param.dtype)
                                dst_param.add_(noise * scale)
                        break
    summary: dict[str, tuple[str, ...]] = {
        "inherited": tuple(sorted(loaded)),
        "fresh": tuple(sorted(result.missing_keys)),
    }
    if retired_source:
        summary["retired"] = tuple(sorted(retired_source))
    return summary


__all__ = [
    "COMBAT_VITALS_DIM",
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
