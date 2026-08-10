"""Grounded recurrent legal-candidate policy/value model.

The model deliberately separates a candidate-independent world representation
from candidate-conditioned action scoring:

``encode_world(world, domain_ids)``
    Encodes only generic world token features and categorical identities.  It
    cannot inspect legal candidates by construction.

``encode_candidates(candidates, world_encoding)``
    Encodes the currently grounded legal candidates and lets them attend to the
    already-computed world latents.  No action-position embedding is used, so
    permuting the candidate axis permutes every candidate output in the same
    way without changing state values.

Concrete runtime identities occupy a separate relation channel from definition
IDs.  Candidate sources and targets are matched back to the exact encoded world
tokens before scoring, while definition matches remain available for unseen
reward/shop entities.  The recurrent state is split into a run-scale memory and
a combat-scale memory: combat decisions cannot overwrite long-horizon build
context, and leaving combat clears only the tactical half.  There are no
hand-written card/boss scores or predicted effect deltas in this module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, cast

import torch
from torch import Tensor, nn

# The structural encoder owns a versioned, collision-free region for all
# reviewed numeric facts, followed by disjoint categorical and dynamic-value
# hash regions.  Keep the minimum here so model and encoder cannot silently
# disagree about that tensor ABI.
MIN_TOKEN_FEATURE_DIM: Final = 224
COMBAT_DOMAIN_ID: Final = 1
MACRO_ECONOMIC_SURFACE_NONE: Final = 0
MACRO_ECONOMIC_SURFACE_REST: Final = 1
MACRO_ECONOMIC_SURFACE_SHOP: Final = 2
MACRO_ECONOMIC_SURFACE_CARD_REWARD: Final = 3
MACRO_ECONOMIC_SURFACE_UPGRADE_SELECTION: Final = 4
MACRO_ECONOMIC_SURFACE_REMOVAL_SELECTION: Final = 5
MACRO_ECONOMIC_SURFACE_COUNT: Final = 6
TRANSACTION_EFFECT_COUNT: Final = 4
SELECTION_DELTA_COUNT: Final = 3
_INTEGER_DTYPES = frozenset({torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8})
_MAX_BINDING_ID: Final = 2**31 - 1


def _require_rank(name: str, value: Tensor, rank: int) -> None:
    if value.ndim != rank:
        raise ValueError(f"{name} must have rank {rank}, got shape {tuple(value.shape)}")


def _require_shape(name: str, value: Tensor, shape: tuple[int, ...]) -> None:
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}")


def _require_bool(name: str, value: Tensor) -> None:
    if value.dtype != torch.bool:
        raise TypeError(f"{name} must use torch.bool, got {value.dtype}")


def _require_finite_floating(name: str, value: Tensor) -> None:
    if not torch.is_floating_point(value):
        raise TypeError(f"{name} must be floating point, got {value.dtype}")
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{name} contains NaN or infinity")


def _require_ids(name: str, value: Tensor, size: int) -> None:
    if value.dtype not in _INTEGER_DTYPES:
        raise TypeError(f"{name} must use an integer dtype, got {value.dtype}")
    if value.numel() and (int(value.min().item()) < 0 or int(value.max().item()) >= size):
        raise ValueError(f"{name} contains an ID outside [0, {size})")


def _require_binding_ids(name: str, value: Tensor) -> None:
    """Validate collision-free decision-local semantic binding IDs.

    Binding IDs are not embedding-table indexes.  They only participate in
    exact equality pooling and therefore use the full non-negative int32
    snapshot range rather than ``entity_vocab_size``.
    """

    if value.dtype not in _INTEGER_DTYPES:
        raise TypeError(f"{name} must use an integer dtype, got {value.dtype}")
    if value.numel() and (int(value.min().item()) < 0 or int(value.max().item()) > _MAX_BINDING_ID):
        raise ValueError(f"{name} contains a binding ID outside [0, {_MAX_BINDING_ID}]")


def _require_same_device(name: str, reference: Tensor, value: Tensor) -> None:
    if value.device != reference.device:
        raise ValueError(f"{name} must be on {reference.device}, got {value.device}")


def _candidate_axis(value: Tensor, permutation: Tensor) -> Tensor:
    return value.index_select(1, permutation.to(device=value.device, dtype=torch.long))


@dataclass(frozen=True, slots=True)
class GroundedCandidateConfig:
    """Shape and capacity contract for :class:`RecurrentCandidateModel`.

    Vocabulary sizes intentionally live here instead of importing a legacy
    observation schema.  Encoders may use a smaller or larger generic contract
    as long as their emitted IDs remain within these declared bounds.
    """

    token_feature_dim: int = 224
    d_model: int = 128
    n_heads: int = 4
    ffn_dim: int = 384
    world_layers: int = 3
    latent_slots: int = 12
    latent_layers: int = 2
    local_layers: int = 1
    candidate_layers: int = 1
    recurrent_hidden_dim: int = 256
    dropout: float = 0.05
    # unknown/combat/build/route/terminal/chance plus two reserved domains.
    # Keeping this generic vocabulary here avoids importing any environment
    # heuristic schema into the model.
    domain_count: int = 8
    type_vocab_size: int = 128
    role_vocab_size: int = 64
    owner_vocab_size: int = 128
    entity_vocab_size: int = 8192
    zone_vocab_size: int = 32
    order_vocab_size: int = 128

    def __post_init__(self) -> None:
        positive = {
            "token_feature_dim": self.token_feature_dim,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "ffn_dim": self.ffn_dim,
            "world_layers": self.world_layers,
            "latent_slots": self.latent_slots,
            "latent_layers": self.latent_layers,
            "local_layers": self.local_layers,
            "candidate_layers": self.candidate_layers,
            "recurrent_hidden_dim": self.recurrent_hidden_dim,
            "domain_count": self.domain_count,
            "type_vocab_size": self.type_vocab_size,
            "role_vocab_size": self.role_vocab_size,
            "owner_vocab_size": self.owner_vocab_size,
            "entity_vocab_size": self.entity_vocab_size,
            "zone_vocab_size": self.zone_vocab_size,
            "order_vocab_size": self.order_vocab_size,
        }
        wrong_types = [
            name for name, value in positive.items() if isinstance(value, bool) or not isinstance(value, int)
        ]
        if wrong_types:
            raise TypeError("grounded-candidate dimensions must be exact integers: " + ", ".join(wrong_types))
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"grounded-candidate dimensions must be positive: {', '.join(invalid)}")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.recurrent_hidden_dim % 2 != 0:
            raise ValueError("recurrent_hidden_dim must be even for run/combat memory partitioning")
        if (
            isinstance(self.dropout, bool)
            or not isinstance(self.dropout, int | float)
            or not math.isfinite(float(self.dropout))
            or not 0.0 <= self.dropout < 1.0
        ):
            raise ValueError("dropout must be in [0, 1)")
        if self.token_feature_dim < MIN_TOKEN_FEATURE_DIM:
            raise ValueError(
                "token_feature_dim must be at least " f"{MIN_TOKEN_FEATURE_DIM} for the grounded feature ABI"
            )
        for name in (
            "type_vocab_size",
            "role_vocab_size",
            "owner_vocab_size",
            "entity_vocab_size",
            "zone_vocab_size",
        ):
            if int(getattr(self, name)) < 4:
                raise ValueError(f"{name} must be at least 4")
        if self.order_vocab_size < 2:
            raise ValueError("order_vocab_size must be at least 2")
        if self.domain_count <= 5:
            raise ValueError("domain_count must cover the six fixed baseline domains")


@dataclass(frozen=True, slots=True)
class WorldTokenBatch:
    """Generic world-token tensors.

    Shapes use ``B`` for batch, ``W`` for active/padded world-token capacity,
    and ``F`` for ``GroundedCandidateConfig.token_feature_dim``.
    """

    features: Tensor  # [B, W, F]
    mask: Tensor  # [B, W]
    type_ids: Tensor  # [B, W]
    role_ids: Tensor  # [B, W]
    owner_ids: Tensor  # [B, W]
    entity_ids: Tensor  # [B, W]
    entity_aux_ids: Tensor  # [B, W]
    definition_binding_ids: Tensor  # [B, W], exact decision-local identity
    relation_binding_ids: Tensor  # [B, W], exact decision-local relation
    zone_ids: Tensor  # [B, W]
    order_ids: Tensor  # [B, W]

    def validate(self, config: GroundedCandidateConfig) -> tuple[int, int]:
        _require_rank("world.features", self.features, 3)
        _require_finite_floating("world.features", self.features)
        batch_size, world_count, feature_dim = self.features.shape
        if batch_size <= 0:
            raise ValueError("world batch size must be positive")
        if feature_dim != config.token_feature_dim:
            raise ValueError(
                "world.features last dimension must equal "
                f"token_feature_dim={config.token_feature_dim}, got {feature_dim}"
            )
        expected = (batch_size, world_count)
        for name, value in (
            ("mask", self.mask),
            ("type_ids", self.type_ids),
            ("role_ids", self.role_ids),
            ("owner_ids", self.owner_ids),
            ("entity_ids", self.entity_ids),
            ("entity_aux_ids", self.entity_aux_ids),
            ("definition_binding_ids", self.definition_binding_ids),
            ("relation_binding_ids", self.relation_binding_ids),
            ("zone_ids", self.zone_ids),
            ("order_ids", self.order_ids),
        ):
            _require_shape(f"world.{name}", value, expected)
            _require_same_device(f"world.{name}", self.features, value)
        _require_bool("world.mask", self.mask)
        for name, value, size in (
            ("type_ids", self.type_ids, config.type_vocab_size),
            ("role_ids", self.role_ids, config.role_vocab_size),
            ("owner_ids", self.owner_ids, config.owner_vocab_size),
            ("entity_ids", self.entity_ids, config.entity_vocab_size),
            ("entity_aux_ids", self.entity_aux_ids, config.entity_vocab_size),
            ("zone_ids", self.zone_ids, config.zone_vocab_size),
            ("order_ids", self.order_ids, config.order_vocab_size),
        ):
            _require_ids(f"world.{name}", value, size)
        for name, value in (
            ("definition_binding_ids", self.definition_binding_ids),
            ("relation_binding_ids", self.relation_binding_ids),
        ):
            _require_binding_ids(f"world.{name}", value)
        if world_count <= 0:
            raise ValueError("world token capacity must be positive")
        return batch_size, world_count


@dataclass(frozen=True, slots=True)
class CandidateTokenBatch:
    """Generic grounded-candidate tensors.

    ``A`` is the current candidate capacity and ``L`` is per-candidate local
    token capacity.  There is intentionally no candidate-position ID: action
    order is transport metadata, not action semantics.
    """

    features: Tensor  # [B, A, F]
    type_ids: Tensor  # [B, A]
    role_ids: Tensor  # [B, A]
    owner_ids: Tensor  # [B, A]
    entity_ids: Tensor  # [B, A]
    entity_aux_ids: Tensor  # [B, A]
    definition_binding_ids: Tensor  # [B, A]
    relation_binding_ids: Tensor  # [B, A]
    zone_ids: Tensor  # [B, A]
    target_owner_ids: Tensor  # [B, A]
    target_entity_ids: Tensor  # [B, A]
    target_entity_aux_ids: Tensor  # [B, A]
    target_definition_binding_ids: Tensor  # [B, A]
    target_relation_binding_ids: Tensor  # [B, A]
    local_features: Tensor  # [B, A, L, F]
    local_mask: Tensor  # [B, A, L]
    local_type_ids: Tensor  # [B, A, L]
    local_role_ids: Tensor  # [B, A, L]
    local_owner_ids: Tensor  # [B, A, L]
    local_entity_ids: Tensor  # [B, A, L]
    local_entity_aux_ids: Tensor  # [B, A, L]
    local_definition_binding_ids: Tensor  # [B, A, L]
    local_relation_binding_ids: Tensor  # [B, A, L]
    local_zone_ids: Tensor  # [B, A, L]
    local_order_ids: Tensor  # [B, A, L]
    action_mask: Tensor  # [B, A]

    def validate(self, config: GroundedCandidateConfig) -> tuple[int, int, int]:
        _require_rank("candidates.features", self.features, 3)
        _require_finite_floating("candidates.features", self.features)
        batch_size, action_count, feature_dim = self.features.shape
        if batch_size <= 0:
            raise ValueError("candidate batch size must be positive")
        if feature_dim != config.token_feature_dim:
            raise ValueError(
                "candidates.features last dimension must equal "
                f"token_feature_dim={config.token_feature_dim}, got {feature_dim}"
            )
        candidate_shape = (batch_size, action_count)
        for name, value in (
            ("type_ids", self.type_ids),
            ("role_ids", self.role_ids),
            ("owner_ids", self.owner_ids),
            ("entity_ids", self.entity_ids),
            ("entity_aux_ids", self.entity_aux_ids),
            ("definition_binding_ids", self.definition_binding_ids),
            ("relation_binding_ids", self.relation_binding_ids),
            ("zone_ids", self.zone_ids),
            ("target_owner_ids", self.target_owner_ids),
            ("target_entity_ids", self.target_entity_ids),
            ("target_entity_aux_ids", self.target_entity_aux_ids),
            (
                "target_definition_binding_ids",
                self.target_definition_binding_ids,
            ),
            ("target_relation_binding_ids", self.target_relation_binding_ids),
            ("action_mask", self.action_mask),
        ):
            _require_shape(f"candidates.{name}", value, candidate_shape)
            _require_same_device(f"candidates.{name}", self.features, value)

        _require_bool("candidates.action_mask", self.action_mask)
        for name, value, size in (
            ("type_ids", self.type_ids, config.type_vocab_size),
            ("role_ids", self.role_ids, config.role_vocab_size),
            ("owner_ids", self.owner_ids, config.owner_vocab_size),
            ("entity_ids", self.entity_ids, config.entity_vocab_size),
            ("entity_aux_ids", self.entity_aux_ids, config.entity_vocab_size),
            ("zone_ids", self.zone_ids, config.zone_vocab_size),
            ("target_owner_ids", self.target_owner_ids, config.owner_vocab_size),
            ("target_entity_ids", self.target_entity_ids, config.entity_vocab_size),
            (
                "target_entity_aux_ids",
                self.target_entity_aux_ids,
                config.entity_vocab_size,
            ),
        ):
            _require_ids(f"candidates.{name}", value, size)
        for name, value in (
            ("definition_binding_ids", self.definition_binding_ids),
            ("relation_binding_ids", self.relation_binding_ids),
            (
                "target_definition_binding_ids",
                self.target_definition_binding_ids,
            ),
            ("target_relation_binding_ids", self.target_relation_binding_ids),
        ):
            _require_binding_ids(f"candidates.{name}", value)

        _require_rank("candidates.local_features", self.local_features, 4)
        _require_finite_floating("candidates.local_features", self.local_features)
        _require_same_device("candidates.local_features", self.features, self.local_features)
        local_batch, local_actions, local_count, local_feature_dim = self.local_features.shape
        if (local_batch, local_actions) != candidate_shape:
            raise ValueError(
                "candidates.local_features leading dimensions must equal "
                f"{candidate_shape}, got {(local_batch, local_actions)}"
            )
        if local_feature_dim != config.token_feature_dim:
            raise ValueError(
                "candidates.local_features last dimension must equal "
                f"token_feature_dim={config.token_feature_dim}, got {local_feature_dim}"
            )
        local_shape = (batch_size, action_count, local_count)
        for name, value in (
            ("local_mask", self.local_mask),
            ("local_type_ids", self.local_type_ids),
            ("local_role_ids", self.local_role_ids),
            ("local_owner_ids", self.local_owner_ids),
            ("local_entity_ids", self.local_entity_ids),
            ("local_entity_aux_ids", self.local_entity_aux_ids),
            (
                "local_definition_binding_ids",
                self.local_definition_binding_ids,
            ),
            ("local_relation_binding_ids", self.local_relation_binding_ids),
            ("local_zone_ids", self.local_zone_ids),
            ("local_order_ids", self.local_order_ids),
        ):
            _require_shape(f"candidates.{name}", value, local_shape)
            _require_same_device(f"candidates.{name}", self.features, value)
        _require_bool("candidates.local_mask", self.local_mask)
        for name, value, size in (
            ("local_type_ids", self.local_type_ids, config.type_vocab_size),
            ("local_role_ids", self.local_role_ids, config.role_vocab_size),
            ("local_owner_ids", self.local_owner_ids, config.owner_vocab_size),
            ("local_entity_ids", self.local_entity_ids, config.entity_vocab_size),
            (
                "local_entity_aux_ids",
                self.local_entity_aux_ids,
                config.entity_vocab_size,
            ),
            ("local_zone_ids", self.local_zone_ids, config.zone_vocab_size),
            ("local_order_ids", self.local_order_ids, config.order_vocab_size),
        ):
            _require_ids(f"candidates.{name}", value, size)
        for name, value in (
            (
                "local_definition_binding_ids",
                self.local_definition_binding_ids,
            ),
            ("local_relation_binding_ids", self.local_relation_binding_ids),
        ):
            _require_binding_ids(f"candidates.{name}", value)
        if action_count <= 0 or local_count <= 0:
            raise ValueError("candidate and candidate-local capacities must be positive")
        return batch_size, action_count, local_count

    def permute_candidates(self, permutation: Tensor) -> CandidateTokenBatch:
        """Return a batch with every candidate-aligned tensor permuted equally."""

        _require_rank("permutation", permutation, 1)
        action_count = int(self.features.shape[1])
        if permutation.numel() != action_count:
            raise ValueError(f"permutation must contain {action_count} entries")
        normalized = permutation.detach().to(device="cpu", dtype=torch.long)
        if sorted(normalized.tolist()) != list(range(action_count)):
            raise ValueError("permutation must contain every candidate index exactly once")
        return CandidateTokenBatch(
            features=_candidate_axis(self.features, permutation),
            type_ids=_candidate_axis(self.type_ids, permutation),
            role_ids=_candidate_axis(self.role_ids, permutation),
            owner_ids=_candidate_axis(self.owner_ids, permutation),
            entity_ids=_candidate_axis(self.entity_ids, permutation),
            entity_aux_ids=_candidate_axis(self.entity_aux_ids, permutation),
            definition_binding_ids=_candidate_axis(
                self.definition_binding_ids,
                permutation,
            ),
            relation_binding_ids=_candidate_axis(
                self.relation_binding_ids,
                permutation,
            ),
            zone_ids=_candidate_axis(self.zone_ids, permutation),
            target_owner_ids=_candidate_axis(self.target_owner_ids, permutation),
            target_entity_ids=_candidate_axis(self.target_entity_ids, permutation),
            target_entity_aux_ids=_candidate_axis(
                self.target_entity_aux_ids,
                permutation,
            ),
            target_definition_binding_ids=_candidate_axis(
                self.target_definition_binding_ids,
                permutation,
            ),
            target_relation_binding_ids=_candidate_axis(
                self.target_relation_binding_ids,
                permutation,
            ),
            local_features=_candidate_axis(self.local_features, permutation),
            local_mask=_candidate_axis(self.local_mask, permutation),
            local_type_ids=_candidate_axis(self.local_type_ids, permutation),
            local_role_ids=_candidate_axis(self.local_role_ids, permutation),
            local_owner_ids=_candidate_axis(self.local_owner_ids, permutation),
            local_entity_ids=_candidate_axis(self.local_entity_ids, permutation),
            local_entity_aux_ids=_candidate_axis(
                self.local_entity_aux_ids,
                permutation,
            ),
            local_definition_binding_ids=_candidate_axis(
                self.local_definition_binding_ids,
                permutation,
            ),
            local_relation_binding_ids=_candidate_axis(
                self.local_relation_binding_ids,
                permutation,
            ),
            local_zone_ids=_candidate_axis(self.local_zone_ids, permutation),
            local_order_ids=_candidate_axis(self.local_order_ids, permutation),
            action_mask=_candidate_axis(self.action_mask, permutation),
        )


@dataclass(frozen=True, slots=True)
class GroundedCandidateBatch:
    """Complete input contract for the grounded-candidate baseline."""

    world: WorldTokenBatch
    candidates: CandidateTokenBatch
    domain_ids: Tensor  # [B]
    macro_economic_surface_ids: Tensor  # [B]

    def validate(self, config: GroundedCandidateConfig) -> tuple[int, int, int, int]:
        world_batch, world_count = self.world.validate(config)
        candidate_batch, action_count, local_count = self.candidates.validate(config)
        if world_batch != candidate_batch:
            raise ValueError(f"world/candidate batch sizes differ: {world_batch} != {candidate_batch}")
        _require_same_device("candidates.features", self.world.features, self.candidates.features)
        _require_shape("domain_ids", self.domain_ids, (world_batch,))
        _require_same_device("domain_ids", self.world.features, self.domain_ids)
        _require_ids("domain_ids", self.domain_ids, config.domain_count)
        _require_shape(
            "macro_economic_surface_ids",
            self.macro_economic_surface_ids,
            (world_batch,),
        )
        _require_same_device(
            "macro_economic_surface_ids",
            self.world.features,
            self.macro_economic_surface_ids,
        )
        _require_ids(
            "macro_economic_surface_ids",
            self.macro_economic_surface_ids,
            MACRO_ECONOMIC_SURFACE_COUNT,
        )
        return world_batch, world_count, action_count, local_count

    def permute_candidates(self, permutation: Tensor) -> GroundedCandidateBatch:
        return GroundedCandidateBatch(
            world=self.world,
            candidates=self.candidates.permute_candidates(permutation),
            domain_ids=self.domain_ids,
            macro_economic_surface_ids=self.macro_economic_surface_ids,
        )


@dataclass(frozen=True, slots=True)
class WorldEncoding:
    """Candidate-independent representation produced by ``encode_world``."""

    latents: Tensor  # [B, S, D]
    state_embedding: Tensor  # [B, D]
    tokens: Tensor  # [B, W, D]
    world_mask: Tensor  # [B, W]
    entity_ids: Tensor  # [B, W]
    entity_aux_ids: Tensor  # [B, W], concrete/group relation identity
    definition_binding_ids: Tensor  # [B, W], collision-free per decision
    relation_binding_ids: Tensor  # [B, W], collision-free per decision

    def validate(
        self,
        config: GroundedCandidateConfig,
        *,
        batch_size: int,
    ) -> None:
        _require_shape(
            "world_encoding.latents",
            self.latents,
            (batch_size, config.latent_slots, config.d_model),
        )
        _require_shape(
            "world_encoding.state_embedding",
            self.state_embedding,
            (batch_size, config.d_model),
        )
        _require_rank("world_encoding.tokens", self.tokens, 3)
        if self.tokens.shape[0] != batch_size or self.tokens.shape[2] != config.d_model:
            raise ValueError("world_encoding.tokens has an invalid batch/model shape")
        _require_rank("world_encoding.world_mask", self.world_mask, 2)
        token_shape = (batch_size, int(self.tokens.shape[1]))
        if tuple(self.world_mask.shape) != token_shape:
            raise ValueError("world_encoding.world_mask shape differs from tokens")
        for name, value in (
            ("entity_ids", self.entity_ids),
            ("entity_aux_ids", self.entity_aux_ids),
        ):
            _require_shape(f"world_encoding.{name}", value, token_shape)
            _require_ids(f"world_encoding.{name}", value, config.entity_vocab_size)
        for name, value in (
            ("definition_binding_ids", self.definition_binding_ids),
            ("relation_binding_ids", self.relation_binding_ids),
        ):
            _require_shape(f"world_encoding.{name}", value, token_shape)
            _require_binding_ids(f"world_encoding.{name}", value)
        _require_finite_floating("world_encoding.latents", self.latents)
        _require_finite_floating("world_encoding.state_embedding", self.state_embedding)
        _require_finite_floating("world_encoding.tokens", self.tokens)
        _require_bool("world_encoding.world_mask", self.world_mask)
        _require_same_device("world_encoding.state_embedding", self.latents, self.state_embedding)
        _require_same_device("world_encoding.world_mask", self.latents, self.world_mask)
        for name, value in (
            ("tokens", self.tokens),
            ("entity_ids", self.entity_ids),
            ("entity_aux_ids", self.entity_aux_ids),
            ("definition_binding_ids", self.definition_binding_ids),
            ("relation_binding_ids", self.relation_binding_ids),
        ):
            _require_same_device(f"world_encoding.{name}", self.latents, value)


@dataclass(frozen=True, slots=True)
class CandidateEncoding:
    """Grounded candidate representations produced after world encoding."""

    embeddings: Tensor  # [B, A, D], zero on invalid candidates
    action_mask: Tensor  # [B, A] bool


@dataclass(frozen=True, slots=True)
class RecurrentCandidateOutput:
    """One recurrent policy/value decision.

    ``recurrent_state`` is the state *after* consuming the current observation.
    It must be passed to the next decision in the same episode and reset at an
    environment/task terminal.
    """

    world_latents: Tensor  # [B, S, D]
    state_embedding: Tensor  # [B, D]
    recurrent_state: Tensor  # [B, H]
    candidate_embeddings: Tensor  # [B, A, D]
    policy_logits: Tensor  # [B, A], invalid candidates use dtype minimum
    # Same macro residual as acting, but with the base policy and shared
    # representation detached.  Factual AWR therefore cannot displace the
    # combat/run trunk or the general policy head.
    macro_policy_logits: Tensor | None  # [B, A]
    # Stable semantic action-branch identity for the two-stage policy.  The
    # grounded encoder's candidate role is ``model_action_kind`` optionally
    # refined by ``model_action_variant``.  Examples are ``play_card``,
    # ``end_turn``, ``reward``, ``proceed``, and the distinct
    # ``card_selection:{select,deselect,confirm,cancel_prompt}`` branches.
    #
    # This tensor is an output/runtime ABI only; it adds no learned parameter
    # and therefore preserves the complete state_dict tensor ABI.
    policy_branch_ids: Tensor  # [B, A] integer
    policy_branch_count: int
    value: Tensor  # [B]
    # Candidate-independent state values at the three learning boundaries.
    # ``value`` above remains the legacy online-V-trace ABI; these heads are
    # deliberately separate so an episodic learner can supervise combat, Act,
    # and full-run outcomes without silently changing the meaning of the
    # existing target.
    combat_task_value: Tensor  # [B]
    act_task_value: Tensor  # [B]
    run_task_value: Tensor  # [B]
    # Expected non-negative future revival costs to the corresponding
    # boundary.  These are state values, not policy penalties by themselves;
    # the learner owns conditioning, advantages, and primary-task protection.
    combat_revival_cost_value: Tensor  # [B]
    act_revival_cost_value: Tensor  # [B]
    run_revival_cost_value: Tensor  # [B]
    # Bounded factual HP-loss estimate to the current combat boundary.  This
    # is candidate-independent and never acts as a hand-authored policy bonus.
    combat_hp_loss_value: Tensor  # [B], [0, 1]
    action_mask: Tensor  # [B, A] bool
    candidate_effect_logits: Tensor | None = None  # [B, A, 4]
    selection_delta_logits: Tensor | None = None  # [B, A, 3]
    transaction_q_values: Tensor | None = None  # [B, A]
    # Candidate-independent factual baseline for macro-option AWR.  Its shared
    # recurrent input is structurally detached in ``forward``.
    macro_option_value: Tensor | None = None  # [B]
    # Bounded factual liveness-risk estimate for every currently legal
    # candidate.  This is deliberately separate from the task-return Q/value
    # heads: once an ordinary failure baseline has converged to ``-1``, a
    # centered candidate risk can still distinguish the action that re-enters
    # a witnessed loop from an available exit.  Invalid candidates are zero.
    candidate_liveness_cost_values: Tensor | None = None  # [B, A], [0, 1]
    # Candidate-independent bounded risk, trained from the formal
    # liveness-value targets rather than aliasing them onto the candidate Q.
    liveness_cost_value: Tensor | None = None  # [B], [0, 1]

    def validate(self, config: GroundedCandidateConfig) -> tuple[int, int]:
        """Validate the complete model-output ABI and return ``(B, A)``.

        The value heads are intentionally validated independently instead of
        being stacked into a positional tensor.  This keeps their horizon
        semantics explicit at every learner call site and prevents an index
        mix-up from silently training the wrong target.
        """

        _require_rank("output.state_embedding", self.state_embedding, 2)
        batch_size, model_dim = self.state_embedding.shape
        if batch_size <= 0:
            raise ValueError("output batch size must be positive")
        if model_dim != config.d_model:
            raise ValueError(
                "output.state_embedding last dimension must equal " f"d_model={config.d_model}, got {model_dim}"
            )
        _require_shape(
            "output.world_latents",
            self.world_latents,
            (batch_size, config.latent_slots, config.d_model),
        )
        _require_shape(
            "output.recurrent_state",
            self.recurrent_state,
            (batch_size, config.recurrent_hidden_dim),
        )
        _require_rank("output.candidate_embeddings", self.candidate_embeddings, 3)
        candidate_batch, action_count, candidate_dim = self.candidate_embeddings.shape
        if candidate_batch != batch_size or candidate_dim != config.d_model:
            raise ValueError(
                "output.candidate_embeddings must have shape "
                f"[B, A, {config.d_model}], got {tuple(self.candidate_embeddings.shape)}"
            )
        candidate_shape = (batch_size, action_count)
        _require_shape("output.policy_logits", self.policy_logits, candidate_shape)
        if self.macro_policy_logits is not None:
            _require_shape(
                "output.macro_policy_logits",
                self.macro_policy_logits,
                candidate_shape,
            )
        _require_shape(
            "output.policy_branch_ids",
            self.policy_branch_ids,
            candidate_shape,
        )
        _require_shape("output.action_mask", self.action_mask, candidate_shape)
        _require_bool("output.action_mask", self.action_mask)
        if (
            isinstance(self.policy_branch_count, bool)
            or not isinstance(self.policy_branch_count, int)
            or self.policy_branch_count <= 0
        ):
            raise ValueError("output.policy_branch_count must be a positive integer")
        _require_ids(
            "output.policy_branch_ids",
            self.policy_branch_ids,
            self.policy_branch_count,
        )

        floating_outputs = (
            ("world_latents", self.world_latents),
            ("state_embedding", self.state_embedding),
            ("recurrent_state", self.recurrent_state),
            ("candidate_embeddings", self.candidate_embeddings),
            ("policy_logits", self.policy_logits),
        )
        for name, value in floating_outputs:
            _require_finite_floating(f"output.{name}", value)
            _require_same_device(f"output.{name}", self.state_embedding, value)
        _require_same_device("output.action_mask", self.state_embedding, self.action_mask)
        _require_same_device(
            "output.policy_branch_ids",
            self.state_embedding,
            self.policy_branch_ids,
        )

        for name, value in (
            ("value", self.value),
            ("combat_task_value", self.combat_task_value),
            ("act_task_value", self.act_task_value),
            ("run_task_value", self.run_task_value),
            ("combat_revival_cost_value", self.combat_revival_cost_value),
            ("act_revival_cost_value", self.act_revival_cost_value),
            ("run_revival_cost_value", self.run_revival_cost_value),
            ("combat_hp_loss_value", self.combat_hp_loss_value),
        ):
            _require_shape(f"output.{name}", value, (batch_size,))
            _require_finite_floating(f"output.{name}", value)
            _require_same_device(f"output.{name}", self.state_embedding, value)

        for name, value in (
            ("combat_revival_cost_value", self.combat_revival_cost_value),
            ("act_revival_cost_value", self.act_revival_cost_value),
            ("run_revival_cost_value", self.run_revival_cost_value),
        ):
            if bool((value < 0.0).any().item()):
                raise ValueError(f"output.{name} must be non-negative")
        if bool(
            (
                (self.combat_hp_loss_value < 0.0)
                | (self.combat_hp_loss_value > 1.0)
            ).any().item()
        ):
            raise ValueError("output.combat_hp_loss_value must be in [0, 1]")

        for name, optional_value, shape in (
            (
                "candidate_effect_logits",
                self.candidate_effect_logits,
                (batch_size, action_count, TRANSACTION_EFFECT_COUNT),
            ),
            (
                "selection_delta_logits",
                self.selection_delta_logits,
                (batch_size, action_count, SELECTION_DELTA_COUNT),
            ),
            (
                "transaction_q_values",
                self.transaction_q_values,
                candidate_shape,
            ),
            (
                "macro_policy_logits",
                self.macro_policy_logits,
                candidate_shape,
            ),
            (
                "macro_option_value",
                self.macro_option_value,
                (batch_size,),
            ),
            (
                "candidate_liveness_cost_values",
                self.candidate_liveness_cost_values,
                candidate_shape,
            ),
            (
                "liveness_cost_value",
                self.liveness_cost_value,
                (batch_size,),
            ),
        ):
            if optional_value is None:
                continue
            _require_shape(f"output.{name}", optional_value, shape)
            _require_finite_floating(f"output.{name}", optional_value)
            _require_same_device(
                f"output.{name}",
                self.state_embedding,
                optional_value,
            )
        if self.candidate_liveness_cost_values is not None:
            liveness_cost = self.candidate_liveness_cost_values
            if bool((self.action_mask & ((liveness_cost < 0.0) | (liveness_cost > 1.0))).any().item()):
                raise ValueError("output.candidate_liveness_cost_values must be in [0, 1]")
            if bool((liveness_cost.masked_select(~self.action_mask) != 0.0).any().item()):
                raise ValueError("output.candidate_liveness_cost_values must be zero on " "invalid candidates")
        if self.liveness_cost_value is not None and bool(
            ((self.liveness_cost_value < 0.0) | (self.liveness_cost_value > 1.0)).any().item()
        ):
            raise ValueError("output.liveness_cost_value must be in [0, 1]")

        return batch_size, action_count

    def _policy_components(
        self,
        logits_override: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return hierarchical policy terms without candidate-count bias.

        A flat candidate softmax makes one singleton branch compete with every
        concrete member of a many-candidate branch.  In deterministic mode this
        can select ``end_turn`` over a collectively preferred set of playable
        cards simply because the play probability is divided among cards.  The
        same pathology exists for ``proceed`` versus multiple reward claims.

        The policy is instead factorized as::

            p(action) = p(semantic_branch) * p(action | semantic_branch)

        A branch logit is the log-mean-exp of its legal candidate logits.  The
        mean, rather than sum, prevents the number of equivalent/near-equivalent
        candidates from automatically increasing branch probability.  Candidate
        choice inside a branch remains a normal softmax over learned logits.

        Returns ``(joint_log_p, branch_log_p, within_log_p, branch_counts)``.
        Invalid candidate entries use ``-inf`` and all-invalid rows remain safe.
        Everything is float32 so low-precision inference cannot underflow a
        small but valid branch before the collector mixes epsilon exploration.
        """

        mask = self.action_mask.bool()
        logits = (
            self.policy_logits
            if logits_override is None
            else logits_override
        ).float()
        branch_ids = self.policy_branch_ids.long()
        batch_size = int(logits.shape[0])
        branch_shape = (batch_size, self.policy_branch_count)

        counts = torch.zeros(
            branch_shape,
            device=logits.device,
            dtype=logits.dtype,
        )
        counts.scatter_add_(1, branch_ids, mask.to(dtype=logits.dtype))
        branch_mask = counts > 0.0

        masked_logits = torch.where(
            mask,
            logits,
            torch.full_like(logits, -torch.inf),
        )
        branch_max = torch.full(
            branch_shape,
            -torch.inf,
            device=logits.device,
            dtype=logits.dtype,
        )
        branch_max.scatter_reduce_(
            1,
            branch_ids,
            masked_logits,
            reduce="amax",
            include_self=True,
        )
        safe_branch_max = torch.where(
            branch_mask,
            branch_max,
            torch.zeros_like(branch_max),
        )
        candidate_branch_max = safe_branch_max.gather(1, branch_ids)
        centered_exp = torch.where(
            mask,
            torch.exp(logits - candidate_branch_max),
            torch.zeros_like(logits),
        )
        branch_exp_sum = torch.zeros_like(counts)
        branch_exp_sum.scatter_add_(1, branch_ids, centered_exp)
        branch_logsumexp = safe_branch_max + torch.log(branch_exp_sum.clamp_min(torch.finfo(logits.dtype).tiny))
        branch_logits = branch_logsumexp - torch.log(counts.clamp_min(1.0))
        branch_logits = torch.where(
            branch_mask,
            branch_logits,
            torch.full_like(branch_logits, -torch.inf),
        )

        branch_normalizer = torch.logsumexp(branch_logits, dim=-1, keepdim=True)
        safe_branch_normalizer = torch.where(
            torch.isfinite(branch_normalizer),
            branch_normalizer,
            torch.zeros_like(branch_normalizer),
        )
        branch_log_probabilities = torch.where(
            branch_mask,
            branch_logits - safe_branch_normalizer,
            torch.full_like(branch_logits, -torch.inf),
        )

        candidate_branch_logsumexp = branch_logsumexp.gather(1, branch_ids)
        within_log_probabilities = torch.where(
            mask,
            logits - candidate_branch_logsumexp,
            torch.full_like(logits, -torch.inf),
        )
        joint_log_probabilities = torch.where(
            mask,
            branch_log_probabilities.gather(1, branch_ids) + within_log_probabilities,
            torch.full_like(logits, -torch.inf),
        )
        return (
            joint_log_probabilities,
            branch_log_probabilities,
            within_log_probabilities,
            counts,
        )

    def policy_log_probabilities(self) -> Tensor:
        """Return the masked two-stage candidate log distribution."""

        return self._policy_components()[0]

    def macro_policy_log_probabilities(self) -> Tensor:
        """Return the isolated macro-residual policy distribution."""

        if self.macro_policy_logits is None:
            raise RuntimeError("macro policy logits are disabled")
        return self._policy_components(self.macro_policy_logits)[0]

    def policy_probabilities(self) -> Tensor:
        """Return float32 hierarchical probabilities.

        Valid rows sum to one. All-invalid rows contain only zeros.
        """

        log_probabilities = self.policy_log_probabilities()
        probabilities = torch.where(
            self.action_mask.bool(),
            torch.exp(log_probabilities),
            torch.zeros_like(log_probabilities),
        )
        # The hierarchical factorization contains multiple float32 scatter
        # reductions.  On GPU their accumulated roundoff can leave a legal row
        # a few parts per million away from one even though the log policy is
        # mathematically normalized.  Canonicalize the inference distribution
        # before it reaches collectors and diagnostic journals.  All-masked
        # rows deliberately remain zero.
        mass = probabilities.sum(dim=-1, keepdim=True)
        return torch.where(
            mass > 0.0,
            probabilities / mass.clamp_min(torch.finfo(probabilities.dtype).tiny),
            probabilities,
        )

    def policy_branch_probabilities(self) -> Tensor:
        """Return branch marginals, independent of within-branch cardinality."""

        branch_log_probabilities = self._policy_components()[1]
        return torch.where(
            torch.isfinite(branch_log_probabilities),
            torch.exp(branch_log_probabilities),
            torch.zeros_like(branch_log_probabilities),
        )

    def greedy_action_indices(self) -> Tensor:
        """Choose branch first, then the best candidate inside that branch.

        Returning ``-1`` for an all-invalid row keeps the primitive total; the
        collector already rejects such an environment state before dispatch.
        This differs intentionally from ``argmax(policy_probabilities())``:
        joint probability is divided among candidates inside a branch, whereas
        deterministic hierarchical choice must honor the branch marginal.
        """

        _, branch_log_probabilities, _, _ = self._policy_components()
        branch_valid = torch.isfinite(branch_log_probabilities)
        selected_branch = branch_log_probabilities.argmax(dim=-1)
        in_selected_branch = (self.policy_branch_ids.long() == selected_branch.unsqueeze(-1)) & self.action_mask.bool()
        candidate_logits = torch.where(
            in_selected_branch,
            self.policy_logits.float(),
            torch.full_like(self.policy_logits.float(), -torch.inf),
        )
        selected = candidate_logits.argmax(dim=-1)
        return torch.where(
            branch_valid.any(dim=-1),
            selected,
            torch.full_like(selected, -1),
        )

    def policy_entropy_and_normalized(self) -> tuple[Tensor, Tensor]:
        """Return count-balanced entropy and its exact normalized value.

        The branch entropy is conventional.  Conditional entropy is normalized
        by ``log(candidate_count)`` before being averaged under branch
        probability, so merely exposing more candidates in one semantic branch
        cannot make that branch more attractive to the entropy objective.

        This objective is not bounded by ``log(total_candidate_count)``.  If
        ``q_b`` is one for a branch with multiple candidates and zero for a
        singleton branch, its exact maximum is ``log(sum_b(exp(q_b)))``.  The
        second return value uses that capacity, keeping policy-health telemetry
        in ``[0, 1]`` without mixing the hierarchical objective with ordinary
        flat-categorical entropy.
        """

        (
            _,
            branch_log_probabilities,
            within_log_probabilities,
            counts,
        ) = self._policy_components()
        branch_probabilities = torch.where(
            torch.isfinite(branch_log_probabilities),
            torch.exp(branch_log_probabilities),
            torch.zeros_like(branch_log_probabilities),
        )
        safe_branch_logs = torch.where(
            torch.isfinite(branch_log_probabilities),
            branch_log_probabilities,
            torch.zeros_like(branch_log_probabilities),
        )
        branch_entropy = -(branch_probabilities * safe_branch_logs).sum(dim=-1)

        within_probabilities = torch.where(
            self.action_mask.bool(),
            torch.exp(within_log_probabilities),
            torch.zeros_like(within_log_probabilities),
        )
        safe_within_logs = torch.where(
            self.action_mask.bool(),
            within_log_probabilities,
            torch.zeros_like(within_log_probabilities),
        )
        candidate_entropy_terms = -within_probabilities * safe_within_logs
        conditional_entropy = torch.zeros_like(counts)
        conditional_entropy.scatter_add_(
            1,
            self.policy_branch_ids.long(),
            candidate_entropy_terms,
        )
        conditional_entropy = torch.where(
            counts > 1.0,
            conditional_entropy / torch.log(counts.clamp_min(2.0)),
            torch.zeros_like(conditional_entropy),
        )
        entropy = branch_entropy + (branch_probabilities * conditional_entropy).sum(dim=-1)

        branch_capacity_weights = torch.where(
            counts > 1.0,
            torch.full_like(counts, math.e),
            torch.where(counts > 0.0, torch.ones_like(counts), torch.zeros_like(counts)),
        )
        capacity = torch.log(
            branch_capacity_weights.sum(dim=-1).clamp_min(1.0)
        )
        normalized = torch.where(
            capacity > 0.0,
            entropy / capacity.clamp_min(torch.finfo(entropy.dtype).tiny),
            torch.zeros_like(entropy),
        )
        # Both values are assembled from float32 scatter reductions.  Clamp
        # only the dimensionless diagnostic after applying the mathematically
        # matching capacity; this canonicalizes roundoff without changing the
        # entropy objective or hiding non-finite model outputs.
        normalized = normalized.clamp(min=0.0, max=1.0)
        return entropy, normalized

    def policy_entropy(self) -> Tensor:
        """Return count-balanced hierarchical entropy for each batch row."""

        return self.policy_entropy_and_normalized()[0]


def _make_transformer_stack(
    *,
    d_model: int,
    n_heads: int,
    ffn_dim: int,
    dropout: float,
    layers: int,
) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=n_heads,
        dim_feedforward=ffn_dim,
        dropout=dropout,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )
    return nn.TransformerEncoder(
        layer,
        num_layers=layers,
        norm=nn.LayerNorm(d_model),
        enable_nested_tensor=False,
    )


def _safe_valid_mask(mask: Tensor) -> Tensor:
    """Ensure attention has at least one valid key without changing outputs' mask."""

    valid = mask.bool()
    # Do not branch on ``missing.any()`` here. Turning a device tensor into a
    # Python boolean synchronizes the host on every world/local/candidate
    # attention call; recurrent replay invokes this path hundreds of times per
    # learner update. The broadcast fallback is identical: only column zero is
    # enabled, and only for rows that were entirely invalid.
    missing = ~valid.any(dim=-1, keepdim=True)
    first_column = torch.arange(valid.shape[-1], device=valid.device).eq(0)
    return valid | (missing & first_column)


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    weights = mask.to(dtype=values.dtype).unsqueeze(-1)
    numerator = (values * weights).sum(dim=-2)
    denominator = weights.sum(dim=-2).clamp_min(1.0)
    return numerator / denominator


class _StructuredTokenEmbedder(nn.Module):
    """One shared embedder for world, candidate-query, and local tokens."""

    def __init__(self, config: GroundedCandidateConfig) -> None:
        super().__init__()
        self.config = config
        self.feature_projection = nn.Sequential(
            nn.LayerNorm(config.token_feature_dim),
            nn.Linear(config.token_feature_dim, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )
        self.type_embedding = nn.Embedding(config.type_vocab_size, config.d_model)
        self.role_embedding = nn.Embedding(config.role_vocab_size, config.d_model)
        self.owner_embedding = nn.Embedding(config.owner_vocab_size, config.d_model)
        self.entity_embedding = nn.Embedding(config.entity_vocab_size, config.d_model)
        self.entity_aux_embedding = nn.Embedding(
            config.entity_vocab_size,
            config.d_model,
        )
        self.target_owner_projection = nn.Linear(
            config.d_model,
            config.d_model,
            bias=False,
        )
        self.target_entity_projection = nn.Linear(
            config.d_model,
            config.d_model,
            bias=False,
        )
        self.target_relation_projection = nn.Linear(
            config.d_model,
            config.d_model,
            bias=False,
        )
        self.zone_embedding = nn.Embedding(config.zone_vocab_size, config.d_model)
        self.order_embedding = nn.Embedding(config.order_vocab_size, config.d_model)
        self.segment_embedding = nn.Embedding(3, config.d_model)
        self.output_norm = nn.LayerNorm(config.d_model)

    @staticmethod
    def _bounded(ids: Tensor, size: int) -> Tensor:
        return ids.long().clamp(min=0, max=size - 1)

    def forward(
        self,
        features: Tensor,
        *,
        type_ids: Tensor,
        role_ids: Tensor,
        owner_ids: Tensor,
        entity_ids: Tensor,
        entity_aux_ids: Tensor,
        zone_ids: Tensor,
        order_ids: Tensor | None,
        segment_id: int,
        target_owner_ids: Tensor | None = None,
        target_entity_ids: Tensor | None = None,
        target_entity_aux_ids: Tensor | None = None,
    ) -> Tensor:
        x = self.feature_projection(features.to(dtype=self.type_embedding.weight.dtype))
        x = x + self.type_embedding(self._bounded(type_ids, self.config.type_vocab_size))
        x = x + self.role_embedding(self._bounded(role_ids, self.config.role_vocab_size))
        x = x + self.owner_embedding(self._bounded(owner_ids, self.config.owner_vocab_size))
        x = x + self.entity_embedding(self._bounded(entity_ids, self.config.entity_vocab_size))
        x = x + self.entity_aux_embedding(self._bounded(entity_aux_ids, self.config.entity_vocab_size))
        x = x + self.zone_embedding(self._bounded(zone_ids, self.config.zone_vocab_size))
        if order_ids is not None:
            x = x + self.order_embedding(self._bounded(order_ids, self.config.order_vocab_size))
        if target_owner_ids is not None:
            target_owner = self.owner_embedding(self._bounded(target_owner_ids, self.config.owner_vocab_size))
            x = x + self.target_owner_projection(target_owner)
        if target_entity_ids is not None:
            target_entity = self.entity_embedding(self._bounded(target_entity_ids, self.config.entity_vocab_size))
            x = x + self.target_entity_projection(target_entity)
        if target_entity_aux_ids is not None:
            target_relation = self.entity_aux_embedding(
                self._bounded(
                    target_entity_aux_ids,
                    self.config.entity_vocab_size,
                )
            )
            x = x + self.target_relation_projection(target_relation)
        segment = torch.full(type_ids.shape, segment_id, dtype=torch.long, device=features.device)
        x = x + self.segment_embedding(segment)
        return cast(Tensor, self.output_norm(x))


class RecurrentCandidateModel(nn.Module):
    """Candidate-order-equivariant recurrent actor/value baseline."""

    def __init__(
        self,
        config: GroundedCandidateConfig | None = None,
        *,
        enable_transaction_heads: bool = False,
        enable_liveness_head: bool = False,
    ) -> None:
        super().__init__()
        if not isinstance(enable_transaction_heads, bool):
            raise TypeError("enable_transaction_heads must be a boolean")
        if not isinstance(enable_liveness_head, bool):
            raise TypeError("enable_liveness_head must be a boolean")
        self.config = config or GroundedCandidateConfig()
        self.transaction_heads_enabled = enable_transaction_heads
        self.liveness_head_enabled = enable_liveness_head
        cfg = self.config

        self.token_embedder = _StructuredTokenEmbedder(cfg)
        self.domain_embedding = nn.Embedding(cfg.domain_count, cfg.d_model)

        self.world_null_token = nn.Parameter(torch.empty(1, 1, cfg.d_model))
        self.world_encoder = _make_transformer_stack(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            ffn_dim=cfg.ffn_dim,
            dropout=cfg.dropout,
            layers=cfg.world_layers,
        )
        self.latent_queries = nn.Parameter(torch.empty(1, cfg.latent_slots, cfg.d_model))
        self.world_to_latent = nn.MultiheadAttention(
            cfg.d_model,
            cfg.n_heads,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.world_to_latent_norm = nn.LayerNorm(cfg.d_model)
        self.latent_encoder = _make_transformer_stack(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            ffn_dim=cfg.ffn_dim,
            dropout=cfg.dropout,
            layers=cfg.latent_layers,
        )
        self.state_norm = nn.LayerNorm(cfg.d_model)

        self.local_encoder = _make_transformer_stack(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            ffn_dim=cfg.ffn_dim,
            dropout=cfg.dropout,
            layers=cfg.local_layers,
        )
        self.local_projection = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
        )
        self.candidate_to_world = nn.MultiheadAttention(
            cfg.d_model,
            cfg.n_heads,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.candidate_to_world_norm = nn.LayerNorm(cfg.d_model)
        # Keep exact-runtime and same-definition evidence in separate channels.
        # Their relative usefulness is learned; there is deliberately no fixed
        # "instance match is worth N times a definition match" policy rule.
        self.relation_projection = nn.Sequential(
            nn.LayerNorm(4 * cfg.d_model),
            nn.Linear(4 * cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )
        self.candidate_encoder = _make_transformer_stack(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            ffn_dim=cfg.ffn_dim,
            dropout=cfg.dropout,
            layers=cfg.candidate_layers,
        )
        self.state_to_candidate = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )
        self.candidate_norm = nn.LayerNorm(cfg.d_model)

        if cfg.recurrent_hidden_dim % 2:
            raise ValueError("recurrent_hidden_dim must be even for run/combat memory partitioning")
        memory_half = cfg.recurrent_hidden_dim // 2
        self.run_recurrent_cell = nn.GRUCell(cfg.d_model, memory_half)
        self.combat_recurrent_cell = nn.GRUCell(cfg.d_model, memory_half)
        self.run_recurrent_norm = nn.LayerNorm(memory_half)
        self.combat_recurrent_norm = nn.LayerNorm(memory_half)
        self.memory_to_candidate = nn.Sequential(
            nn.LayerNorm(cfg.recurrent_hidden_dim),
            nn.Linear(cfg.recurrent_hidden_dim, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )
        self.policy_feature_norm = nn.LayerNorm(cfg.d_model)
        self.policy_head = self._scalar_head(cfg.d_model)
        self.value_head = self._scalar_head(cfg.recurrent_hidden_dim)
        # Keep the legacy value head intact and add independent state heads for
        # completed-combat, completed-Act, and completed-run supervision.  The
        # separation is structural only: no game identity or hand-written rule
        # enters these generic boundary-value estimators.
        self.combat_task_value_head = self._scalar_head(cfg.recurrent_hidden_dim)
        self.act_task_value_head = self._scalar_head(cfg.recurrent_hidden_dim)
        self.run_task_value_head = self._scalar_head(cfg.recurrent_hidden_dim)
        self.combat_revival_cost_value_head = self._nonnegative_scalar_head(cfg.recurrent_hidden_dim)
        self.act_revival_cost_value_head = self._nonnegative_scalar_head(cfg.recurrent_hidden_dim)
        self.run_revival_cost_value_head = self._nonnegative_scalar_head(cfg.recurrent_hidden_dim)
        self.combat_hp_loss_value_head = self._bounded_scalar_head(
            cfg.recurrent_hidden_dim
        )
        if enable_transaction_heads:
            self.candidate_effect_head: nn.Module | None = self._categorical_head(
                cfg.d_model,
                TRANSACTION_EFFECT_COUNT,
            )
            self.selection_delta_head: nn.Module | None = self._categorical_head(
                cfg.d_model,
                SELECTION_DELTA_COUNT,
            )
            self.transaction_q_head: nn.Module | None = self._scalar_head(cfg.d_model)
            self.macro_surface_candidate_embedding: nn.Embedding | None = (
                nn.Embedding(MACRO_ECONOMIC_SURFACE_COUNT, cfg.d_model)
            )
            self.macro_surface_state_embedding: nn.Embedding | None = (
                nn.Embedding(
                    MACRO_ECONOMIC_SURFACE_COUNT,
                    cfg.recurrent_hidden_dim,
                )
            )
            self.macro_policy_head: nn.Module | None = self._scalar_head(
                cfg.d_model
            )
            self.macro_option_value_head: nn.Module | None = self._scalar_head(
                cfg.recurrent_hidden_dim
            )
            # A v46 -> v47 reviewed model-init must begin with exactly the
            # inherited policy.  Only factual v47 labels may open the residual.
            macro_final = self.macro_policy_head[-1]
            if not isinstance(macro_final, nn.Linear):  # pragma: no cover
                raise RuntimeError("macro policy head lost its final linear")
            nn.init.zeros_(macro_final.weight)
            nn.init.zeros_(macro_final.bias)
            # The factual baseline also starts from a known neutral estimate.
            # A random new value head would turn the first AWR labels into
            # arbitrary positive/negative advantages before seeing one fact.
            macro_value_final = self.macro_option_value_head[-1]
            if not isinstance(macro_value_final, nn.Linear):  # pragma: no cover
                raise RuntimeError("macro value head lost its final linear")
            nn.init.zeros_(macro_value_final.weight)
            nn.init.zeros_(macro_value_final.bias)
        else:
            # Keeping disabled heads as ``None`` preserves the exact v10 state
            # dict ABI.  Enabling them is therefore an explicit model-parameter
            # initialization migration rather than a disguised exact resume.
            self.candidate_effect_head = None
            self.selection_delta_head = None
            self.transaction_q_head = None
            self.macro_surface_candidate_embedding = None
            self.macro_surface_state_embedding = None
            self.macro_policy_head = None
            self.macro_option_value_head = None
        self.candidate_liveness_cost_head: nn.Module | None = (
            self._bounded_scalar_head(cfg.d_model) if enable_liveness_head else None
        )
        self.liveness_cost_value_head: nn.Module | None = (
            self._bounded_scalar_head(cfg.recurrent_hidden_dim) if enable_liveness_head else None
        )

        nn.init.normal_(self.world_null_token, mean=0.0, std=0.02)
        nn.init.normal_(self.latent_queries, mean=0.0, std=0.02)

    @staticmethod
    def _scalar_head(input_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, 1),
        )

    @staticmethod
    def _nonnegative_scalar_head(input_dim: int) -> nn.Sequential:
        """Smooth non-negative state-cost estimator.

        Revival-count targets cannot be negative.  Encoding that support in
        the final activation avoids impossible negative predictions while
        preserving gradients near zero; it does not impose any hand-authored
        trade-off against the primary task.
        """

        return nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, 1),
            nn.Softplus(),
        )

    @staticmethod
    def _bounded_scalar_head(input_dim: int) -> nn.Sequential:
        """Small candidate head for a calibrated probability-like cost.

        A sigmoid gives the learner an explicit, finite ``[0, 1]`` support for
        factual liveness targets.  The narrower hidden layer keeps this
        auxiliary head cheap relative to candidate/world attention and avoids
        turning a credit-repair objective into a model-capacity expansion.
        """

        hidden_dim = max(16, input_dim // 2)
        return nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    @staticmethod
    def _categorical_head(input_dim: int, classes: int) -> nn.Sequential:
        if classes <= 1:
            raise ValueError("categorical head requires at least two classes")
        return nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, classes),
        )

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
    ) -> Tensor:
        """Return an all-zero recurrent state using model dtype/device."""

        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise TypeError("recurrent batch_size must be an integer")
        if batch_size <= 0:
            raise ValueError("recurrent batch_size must be positive")
        reference = next(self.parameters())
        return torch.zeros(
            (batch_size, self.config.recurrent_hidden_dim),
            device=reference.device if device is None else device,
            dtype=reference.dtype,
        )

    def _validate_recurrent_state(
        self,
        recurrent_state: Tensor,
        *,
        batch_size: int,
        reference: Tensor,
    ) -> None:
        _require_shape(
            "recurrent_state",
            recurrent_state,
            (batch_size, self.config.recurrent_hidden_dim),
        )
        _require_finite_floating("recurrent_state", recurrent_state)
        _require_same_device("recurrent_state", reference, recurrent_state)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def encode_world(
        self,
        world: WorldTokenBatch,
        domain_ids: Tensor,
        *,
        _validated: bool = False,
    ) -> WorldEncoding:
        """Encode world/domain tensors without any candidate input."""

        if _validated:
            batch_size = int(world.features.shape[0])
        else:
            batch_size, _ = world.validate(self.config)
            _require_shape("domain_ids", domain_ids, (batch_size,))
            _require_same_device("domain_ids", world.features, domain_ids)
            _require_ids("domain_ids", domain_ids, self.config.domain_count)
        cfg = self.config
        domain_ids = domain_ids.long().clamp(min=0, max=cfg.domain_count - 1)
        world_mask = world.mask.bool()
        world_tokens = self.token_embedder(
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
        world_tokens = world_tokens * world_mask.unsqueeze(-1).to(dtype=world_tokens.dtype)

        null_token = self.world_null_token.expand(batch_size, -1, -1)
        encoded_tokens = torch.cat([null_token, world_tokens], dim=1)
        null_mask = torch.ones((batch_size, 1), dtype=torch.bool, device=world_tokens.device)
        encoded_mask = torch.cat([null_mask, world_mask], dim=1)
        encoded_tokens = self.world_encoder(
            encoded_tokens,
            src_key_padding_mask=~encoded_mask,
        )
        relation_tokens = encoded_tokens[:, 1:] * world_mask.unsqueeze(-1).to(dtype=encoded_tokens.dtype)

        domain_context = self.domain_embedding(domain_ids).unsqueeze(1)
        latent_queries = self.latent_queries.expand(batch_size, -1, -1) + domain_context
        latent_delta, _ = self.world_to_latent(
            latent_queries,
            encoded_tokens,
            encoded_tokens,
            key_padding_mask=~encoded_mask,
            need_weights=False,
        )
        latents = self.world_to_latent_norm(latent_queries + latent_delta)
        latents = self.latent_encoder(latents)
        state_embedding = self.state_norm(latents.mean(dim=1))
        return WorldEncoding(
            latents=latents,
            state_embedding=state_embedding,
            tokens=relation_tokens,
            world_mask=world_mask,
            entity_ids=world.entity_ids,
            entity_aux_ids=world.entity_aux_ids,
            definition_binding_ids=world.definition_binding_ids,
            relation_binding_ids=world.relation_binding_ids,
        )

    def encode_candidates(
        self,
        candidates: CandidateTokenBatch,
        world_encoding: WorldEncoding,
        *,
        _validated: bool = False,
    ) -> CandidateEncoding:
        """Ground the current candidates against a precomputed world encoding."""

        if _validated:
            batch_size, action_count, local_count = (
                int(candidates.features.shape[0]),
                int(candidates.features.shape[1]),
                int(candidates.local_features.shape[2]),
            )
        else:
            batch_size, action_count, local_count = candidates.validate(self.config)
            world_encoding.validate(self.config, batch_size=batch_size)
            _require_same_device(
                "world_encoding.latents",
                candidates.features,
                world_encoding.latents,
            )
        cfg = self.config
        action_mask = candidates.action_mask.bool()

        query_tokens = self.token_embedder(
            candidates.features,
            type_ids=candidates.type_ids,
            role_ids=candidates.role_ids,
            owner_ids=candidates.owner_ids,
            entity_ids=candidates.entity_ids,
            entity_aux_ids=candidates.entity_aux_ids,
            zone_ids=candidates.zone_ids,
            order_ids=None,
            segment_id=1,
            target_owner_ids=candidates.target_owner_ids,
            target_entity_ids=candidates.target_entity_ids,
            target_entity_aux_ids=candidates.target_entity_aux_ids,
        )

        local_features = candidates.local_features.reshape(
            batch_size * action_count,
            local_count,
            cfg.token_feature_dim,
        )
        local_shape = (batch_size * action_count, local_count)
        local_tokens = self.token_embedder(
            local_features,
            type_ids=candidates.local_type_ids.reshape(local_shape),
            role_ids=candidates.local_role_ids.reshape(local_shape),
            owner_ids=candidates.local_owner_ids.reshape(local_shape),
            entity_ids=candidates.local_entity_ids.reshape(local_shape),
            entity_aux_ids=candidates.local_entity_aux_ids.reshape(local_shape),
            zone_ids=candidates.local_zone_ids.reshape(local_shape),
            order_ids=candidates.local_order_ids.reshape(local_shape),
            segment_id=2,
        )
        local_mask = candidates.local_mask.reshape(local_shape).bool()
        local_tokens = self.local_encoder(
            local_tokens,
            src_key_padding_mask=~_safe_valid_mask(local_mask),
        )
        local_pool = _masked_mean(local_tokens, local_mask).reshape(
            batch_size,
            action_count,
            cfg.d_model,
        )
        candidate_tokens = query_tokens + self.local_projection(local_pool)

        world_delta, _ = self.candidate_to_world(
            candidate_tokens,
            world_encoding.tokens,
            world_encoding.tokens,
            key_padding_mask=~_safe_valid_mask(world_encoding.world_mask),
            need_weights=False,
        )
        source_exact, source_definition = self._matched_world_contexts(
            definition_binding_ids=candidates.definition_binding_ids,
            relation_binding_ids=candidates.relation_binding_ids,
            world_encoding=world_encoding,
        )
        target_exact, target_definition = self._matched_world_contexts(
            definition_binding_ids=(candidates.target_definition_binding_ids),
            relation_binding_ids=candidates.target_relation_binding_ids,
            world_encoding=world_encoding,
        )
        relation_context = self.relation_projection(
            torch.cat(
                [
                    source_exact,
                    source_definition,
                    target_exact,
                    target_definition,
                ],
                dim=-1,
            )
        )
        candidate_tokens = self.candidate_to_world_norm(candidate_tokens + world_delta + relation_context)
        candidate_tokens = self.candidate_encoder(
            candidate_tokens,
            src_key_padding_mask=~_safe_valid_mask(action_mask),
        )
        state_context = self.state_to_candidate(world_encoding.state_embedding).unsqueeze(1)
        candidate_tokens = self.candidate_norm(candidate_tokens + state_context)
        candidate_tokens = candidate_tokens * action_mask.unsqueeze(-1).to(candidate_tokens.dtype)
        return CandidateEncoding(embeddings=candidate_tokens, action_mask=action_mask)

    @staticmethod
    def _matched_world_contexts(
        *,
        definition_binding_ids: Tensor,
        relation_binding_ids: Tensor,
        world_encoding: WorldEncoding,
    ) -> tuple[Tensor, Tensor]:
        """Pool exact-instance and same-definition evidence independently.

        Stable entity hash IDs remain inputs to the learned embedding tables,
        but are deliberately *not* used here: two different semantic keys can
        share a finite hash bucket.  Collision-free decision-local binding IDs
        drive equality pooling instead.  A missing/unknown binding (0/1) never
        matches, so padded candidates and unseen entities receive zero context
        rather than an accidental global pool.  The projection consuming these
        two channels learns how much to use each kind of relation instead of a
        hand-written ratio.
        """

        world_mask = world_encoding.world_mask.unsqueeze(1)
        relation_valid = relation_binding_ids.unsqueeze(-1) > 1
        definition_valid = definition_binding_ids.unsqueeze(-1) > 1
        relation_match = relation_valid & (
            relation_binding_ids.unsqueeze(-1) == world_encoding.relation_binding_ids.unsqueeze(1)
        )
        definition_match = definition_valid & (
            definition_binding_ids.unsqueeze(-1) == world_encoding.definition_binding_ids.unsqueeze(1)
        )

        def _pool(matches: Tensor) -> Tensor:
            weights = matches.to(dtype=world_encoding.tokens.dtype) * world_mask.to(dtype=world_encoding.tokens.dtype)
            numerator = torch.einsum("baw,bwd->bad", weights, world_encoding.tokens)
            denominator = weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
            return numerator / denominator

        return _pool(relation_match), _pool(definition_match)

    def forward(
        self,
        batch: GroundedCandidateBatch,
        recurrent_state: Tensor | None = None,
        *,
        validate: bool = True,
        detach_liveness_shared_features: bool = False,
    ) -> RecurrentCandidateOutput:
        """Run the model, validating untrusted tensor contracts by default.

        The production encoder creates typed tensors at the data boundary and
        the collector/learner pass ``validate=False`` to avoid dozens of
        device-to-host synchronization points on ROCm.  External callers keep
        the fail-closed default.
        """

        if not isinstance(detach_liveness_shared_features, bool):
            raise TypeError("detach_liveness_shared_features must be a boolean")
        if detach_liveness_shared_features and not self.liveness_head_enabled:
            raise ValueError("detach_liveness_shared_features requires enabled liveness heads")
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
        mask = candidates.action_mask
        memory_half = self.config.recurrent_hidden_dim // 2
        previous_run = recurrent_state[:, :memory_half]
        previous_combat = recurrent_state[:, memory_half:]
        combat_domain = batch.domain_ids.eq(COMBAT_DOMAIN_ID).unsqueeze(-1)

        # Long-horizon run memory updates only on macro/post-combat states;
        # hundreds of individual card plays therefore cannot overwrite deck,
        # route, shop and resource context.  Tactical memory updates inside a
        # combat and is cleared as soon as the environment leaves that domain.
        run_update = self.run_recurrent_norm(self.run_recurrent_cell(world.state_embedding, previous_run))
        next_run = torch.where(combat_domain, previous_run, run_update)
        combat_update = self.combat_recurrent_norm(self.combat_recurrent_cell(world.state_embedding, previous_combat))
        next_combat = torch.where(
            combat_domain,
            combat_update,
            torch.zeros_like(combat_update),
        )
        next_recurrent_state = torch.cat([next_run, next_combat], dim=-1)
        memory_context = self.memory_to_candidate(next_recurrent_state).unsqueeze(1)
        policy_features = self.policy_feature_norm(candidates.embeddings + memory_context)
        policy_features = policy_features * mask.unsqueeze(-1).to(dtype=policy_features.dtype)
        raw_policy_logits = self.policy_head(policy_features).squeeze(-1)
        invalid_logit = torch.finfo(raw_policy_logits.dtype).min
        policy_logits = raw_policy_logits.masked_fill(~mask, invalid_logit)
        macro_policy_logits = None
        macro_option_value = None

        candidate_effect_logits = None
        selection_delta_logits = None
        transaction_q_values = None
        if self.transaction_heads_enabled:
            if (
                self.candidate_effect_head is None
                or self.selection_delta_head is None
                or self.transaction_q_head is None
            ):  # pragma: no cover - constructor invariant
                raise RuntimeError("transaction head configuration is inconsistent")
            candidate_effect_logits = self.candidate_effect_head(policy_features)
            selection_delta_logits = self.selection_delta_head(policy_features)
            transaction_q_values = self.transaction_q_head(policy_features).squeeze(-1)
            candidate_effect_logits = candidate_effect_logits.masked_fill(
                ~mask.unsqueeze(-1),
                0.0,
            )
            selection_delta_logits = selection_delta_logits.masked_fill(
                ~mask.unsqueeze(-1),
                0.0,
            )
            transaction_q_values = transaction_q_values.masked_fill(~mask, 0.0)
            if (
                self.macro_surface_candidate_embedding is None
                or self.macro_surface_state_embedding is None
                or self.macro_policy_head is None
                or self.macro_option_value_head is None
            ):  # pragma: no cover - constructor invariant
                raise RuntimeError("macro option head configuration is inconsistent")
            macro_surface_ids = batch.macro_economic_surface_ids
            macro_state_mask = macro_surface_ids.ne(
                MACRO_ECONOMIC_SURFACE_NONE
            )
            macro_candidate_context = self.macro_surface_candidate_embedding(
                macro_surface_ids
            ).unsqueeze(1)
            macro_delta = self.macro_policy_head(
                policy_features.detach() + macro_candidate_context
            ).squeeze(-1)
            macro_delta = macro_delta * macro_state_mask.unsqueeze(-1).to(
                dtype=macro_delta.dtype
            )
            policy_logits = (raw_policy_logits + macro_delta).masked_fill(
                ~mask,
                invalid_logit,
            )
            # The macro learner sees the live residual over a detached base;
            # its CE/AWR derivative reaches only this sidecar.
            macro_policy_logits = (
                raw_policy_logits.detach() + macro_delta
            ).masked_fill(~mask, invalid_logit)
            macro_state_context = self.macro_surface_state_embedding(
                macro_surface_ids
            )
            macro_option_value = self.macro_option_value_head(
                next_recurrent_state.detach() + macro_state_context
            ).squeeze(-1)
            macro_option_value = macro_option_value * macro_state_mask.to(
                dtype=macro_option_value.dtype
            )
        candidate_liveness_cost_values = None
        liveness_cost_value = None
        if self.liveness_head_enabled:
            if self.candidate_liveness_cost_head is None or self.liveness_cost_value_head is None:  # pragma: no cover
                raise RuntimeError("liveness head configuration is inconsistent")
            liveness_policy_features = policy_features.detach() if detach_liveness_shared_features else policy_features
            liveness_recurrent_state = (
                next_recurrent_state.detach() if detach_liveness_shared_features else next_recurrent_state
            )
            candidate_liveness_cost_values = (
                self.candidate_liveness_cost_head(liveness_policy_features).squeeze(-1).masked_fill(~mask, 0.0)
            )
            liveness_cost_value = self.liveness_cost_value_head(liveness_recurrent_state).squeeze(-1)

        output = RecurrentCandidateOutput(
            world_latents=world.latents,
            state_embedding=world.state_embedding,
            recurrent_state=next_recurrent_state,
            candidate_embeddings=policy_features,
            policy_logits=policy_logits,
            macro_policy_logits=macro_policy_logits,
            policy_branch_ids=batch.candidates.role_ids,
            policy_branch_count=self.config.role_vocab_size,
            value=self.value_head(next_recurrent_state).squeeze(-1),
            combat_task_value=self.combat_task_value_head(next_recurrent_state).squeeze(-1),
            act_task_value=self.act_task_value_head(next_recurrent_state).squeeze(-1),
            run_task_value=self.run_task_value_head(next_recurrent_state).squeeze(-1),
            combat_revival_cost_value=self.combat_revival_cost_value_head(next_recurrent_state).squeeze(-1),
            act_revival_cost_value=self.act_revival_cost_value_head(next_recurrent_state).squeeze(-1),
            run_revival_cost_value=self.run_revival_cost_value_head(next_recurrent_state).squeeze(-1),
            combat_hp_loss_value=self.combat_hp_loss_value_head(
                next_recurrent_state
            ).squeeze(-1),
            action_mask=mask,
            candidate_effect_logits=candidate_effect_logits,
            selection_delta_logits=selection_delta_logits,
            transaction_q_values=transaction_q_values,
            macro_option_value=macro_option_value,
            candidate_liveness_cost_values=candidate_liveness_cost_values,
            liveness_cost_value=liveness_cost_value,
        )
        if validate:
            output.validate(self.config)
        return output


__all__ = [
    "MACRO_ECONOMIC_SURFACE_CARD_REWARD",
    "MACRO_ECONOMIC_SURFACE_COUNT",
    "MACRO_ECONOMIC_SURFACE_NONE",
    "MACRO_ECONOMIC_SURFACE_REMOVAL_SELECTION",
    "MACRO_ECONOMIC_SURFACE_REST",
    "MACRO_ECONOMIC_SURFACE_SHOP",
    "MACRO_ECONOMIC_SURFACE_UPGRADE_SELECTION",
    "SELECTION_DELTA_COUNT",
    "TRANSACTION_EFFECT_COUNT",
    "CandidateEncoding",
    "CandidateTokenBatch",
    "GroundedCandidateBatch",
    "GroundedCandidateConfig",
    "RecurrentCandidateModel",
    "RecurrentCandidateOutput",
    "WorldEncoding",
    "WorldTokenBatch",
]
