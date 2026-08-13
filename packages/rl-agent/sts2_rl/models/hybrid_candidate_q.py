"""Hybrid slow/fast candidate-Q controller.

The model keeps the grounded factual encoder and the split run/combat memory,
but replaces the legacy shared actor/value head family with one control
objective.  Domain-local transformer streams first absorb the very different
combat and macro candidate distributions.  Their canonical token sequence is
then processed by one shared joint stream before a single candidate-Q readout.

This is deliberately a *control* model, not a compatibility wrapper around the
legacy actor.  It contains no policy logits, imitation heads, liveness actors,
transaction completion heads, or operation-specific preference channels.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Final

import torch
from torch import Tensor, nn

from .grounded_candidate import (
    COMBAT_DOMAIN_ID,
    GroundedCandidateBatch,
    GroundedCandidateConfig,
    RecurrentCandidateModel,
    _make_transformer_stack,
    _safe_valid_mask,
)

HYBRID_CANDIDATE_Q_VERSION: Final = "sts2-hybrid-domain-candidate-q-v1"


@dataclass(frozen=True, slots=True)
class HybridCandidateQConfig:
    """Topology layered on top of the grounded tensor contract."""

    grounded: GroundedCandidateConfig = field(default_factory=GroundedCandidateConfig)
    domain_layers: int = 1
    joint_layers: int = 2

    def __post_init__(self) -> None:
        for name, value in (
            ("domain_layers", self.domain_layers),
            ("joint_layers", self.joint_layers),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"hybrid candidate-Q {name} must be an integer")
            if value <= 0:
                raise ValueError(f"hybrid candidate-Q {name} must be positive")

    def as_mapping(self) -> dict[str, Any]:
        return {
            "version": HYBRID_CANDIDATE_Q_VERSION,
            "grounded": asdict(self.grounded),
            "domain_layers": self.domain_layers,
            "joint_layers": self.joint_layers,
        }


@dataclass(frozen=True, slots=True)
class HybridCandidateQOutput:
    q_values: Tensor  # [B, A]
    recurrent_state: Tensor  # [B, H]
    joint_state_embedding: Tensor  # [B, D], candidate-conditioned
    world_state_embedding: Tensor  # [B, D], candidate-independent
    candidate_embeddings: Tensor  # [B, A, D]
    action_mask: Tensor  # [B, A]

    def validate(self, config: HybridCandidateQConfig) -> tuple[int, int]:
        if self.q_values.ndim != 2:
            raise ValueError("hybrid q_values must have rank 2")
        batch_size, action_count = self.q_values.shape
        grounded = config.grounded
        expected = {
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
                    f"hybrid {name} must have shape {shape}, got {tuple(value.shape)}"
                )
        if self.action_mask.dtype != torch.bool:
            raise TypeError("hybrid action_mask must use torch.bool")
        for name in (
            "q_values",
            "recurrent_state",
            "joint_state_embedding",
            "world_state_embedding",
            "candidate_embeddings",
        ):
            value = getattr(self, name)
            if not torch.is_floating_point(value):
                raise TypeError(f"hybrid {name} must be floating point")
            if not bool(torch.isfinite(value).all().item()):
                raise ValueError(f"hybrid {name} contains NaN or infinity")
        return batch_size, action_count


class HybridDomainCandidateQModel(RecurrentCandidateModel):
    """Domain-local dual streams followed by one shared candidate-Q stream.

    The inherited encoder methods preserve the reviewed world/candidate
    separation and exact relation grounding.  The inherited constructor is
    used only to build those modules and the split recurrent cells; all legacy
    actor/value/auxiliary heads are deleted from this module before training.
    """

    _RETIRED_HEADS: Final[tuple[str, ...]] = (
        "policy_head",
        "value_head",
        "combat_task_value_head",
        "act_task_value_head",
        "run_task_value_head",
        "combat_revival_cost_value_head",
        "act_revival_cost_value_head",
        "run_revival_cost_value_head",
        "combat_hp_loss_value_head",
        "candidate_effect_head",
        "selection_delta_head",
        "candidate_liveness_cost_head",
        "liveness_cost_value_head",
        "policy_feature_norm",
    )

    def __init__(self, config: HybridCandidateQConfig | None = None) -> None:
        hybrid = config or HybridCandidateQConfig()
        super().__init__(hybrid.grounded, enable_transaction_heads=True)
        self.hybrid_config = hybrid
        self.transaction_heads_enabled = False

        if self.transaction_q_head is None:  # pragma: no cover - constructor invariant
            raise RuntimeError("grounded constructor did not create candidate Q head")
        del self.transaction_q_head
        for name in self._RETIRED_HEADS:
            if hasattr(self, name):
                delattr(self, name)

        cfg = hybrid.grounded
        self.macro_stream_norm = nn.LayerNorm(cfg.d_model)
        self.combat_stream_norm = nn.LayerNorm(cfg.d_model)
        self.macro_stream = _make_transformer_stack(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            ffn_dim=cfg.ffn_dim,
            dropout=cfg.dropout,
            layers=hybrid.domain_layers,
        )
        self.combat_stream = _make_transformer_stack(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            ffn_dim=cfg.ffn_dim,
            dropout=cfg.dropout,
            layers=hybrid.domain_layers,
        )
        self.joint_stream = _make_transformer_stack(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            ffn_dim=cfg.ffn_dim,
            dropout=cfg.dropout,
            layers=hybrid.joint_layers,
        )
        self.joint_state_norm = nn.LayerNorm(cfg.d_model)
        self.joint_candidate_norm = nn.LayerNorm(cfg.d_model)
        self.strategy_token = nn.Sequential(
            nn.Linear(cfg.recurrent_hidden_dim // 2, cfg.d_model),
            nn.SiLU(),
            nn.LayerNorm(cfg.d_model),
        )
        # The input semantics differ from the legacy transaction head: the
        # shared joint state is explicit and the complete split memory is
        # retained.  This readout therefore always starts fresh.
        self.candidate_q_head = nn.Sequential(
            nn.Linear(cfg.d_model * 2 + cfg.recurrent_hidden_dim, cfg.ffn_dim),
            nn.SiLU(),
            nn.Linear(cfg.ffn_dim, 1),
        )

    def forward(  # type: ignore[override]
        self,
        batch: GroundedCandidateBatch,
        recurrent_state: Tensor | None = None,
        *,
        validate: bool = True,
    ) -> HybridCandidateQOutput:
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
        mask = candidates.action_mask.bool()
        cfg = self.config
        memory_half = cfg.recurrent_hidden_dim // 2
        previous_run = recurrent_state[:, :memory_half]
        previous_combat = recurrent_state[:, memory_half:]
        combat_domain = batch.domain_ids.eq(COMBAT_DOMAIN_ID).unsqueeze(-1)

        # Slow strategic state is read-only during combat.  The first
        # post-combat macro state contains the factual durable outcome and
        # writes it once.  Tactical state updates only inside combat and is
        # cleared on every macro state, so it starts clean next encounter.
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
        slow_strategy_token = self.strategy_token(next_run)
        candidate_tokens = candidates.embeddings + memory_context.unsqueeze(1)
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

        # Pure domain microbatches execute only their local adapter. Mixed
        # collector batches dispatch subsets and restore their original rows,
        # avoiding the previous 2x local-transformer compute and preventing
        # inactive stems from accumulating zero-valued gradients.
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

        q_state = next_recurrent_state.unsqueeze(1).expand(
            -1,
            joint_candidates.shape[1],
            -1,
        )
        q_values = self.candidate_q_head(
            torch.cat(
                [
                    joint_candidates,
                    joint_state.unsqueeze(1).expand_as(joint_candidates),
                    q_state,
                ],
                dim=-1,
            )
        ).squeeze(-1)
        q_values = q_values.masked_fill(~mask, 0.0)
        output = HybridCandidateQOutput(
            q_values=q_values,
            recurrent_state=next_recurrent_state,
            joint_state_embedding=joint_state,
            world_state_embedding=world.state_embedding,
            candidate_embeddings=joint_candidates,
            action_mask=mask,
        )
        if validate:
            output.validate(self.hybrid_config)
        return output

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def initialize_hybrid_from_grounded(
    model: HybridDomainCandidateQModel,
    grounded_state: dict[str, Tensor],
) -> dict[str, tuple[str, ...]]:
    """Inherit the reviewed grounded factual encoder and split memory.

    New domain and joint stream parameters intentionally start fresh.  Legacy
    policy/value/auxiliary/Q heads are intentionally ignored.  The function is
    a model-initialization migration only; it is not an exact resume contract.
    """

    target = model.state_dict()
    loaded: dict[str, Tensor] = {}
    inherited: list[str] = []
    for key, value in grounded_state.items():
        if key.startswith("transaction_q_head."):
            continue
        mapped = key
        target_value = target.get(mapped)
        if target_value is None or tuple(target_value.shape) != tuple(value.shape):
            continue
        loaded[mapped] = value
        inherited.append(mapped)

    result = model.load_state_dict(loaded, strict=False)
    if result.unexpected_keys:  # pragma: no cover - filtered above
        raise RuntimeError(
            "unexpected hybrid initialization keys: "
            + ", ".join(result.unexpected_keys)
        )
    fresh = tuple(sorted(result.missing_keys))
    for key in fresh:
        if not key.startswith(
            (
                "macro_stream",
                "combat_stream",
                "joint_stream",
                "joint_state_norm",
                "joint_candidate_norm",
                "strategy_token",
            )
        ):
            # The candidate-Q readout starts fresh together with the fusion
            # blocks. Any other missing trunk tensor is a real drift.
            if not key.startswith("candidate_q_head"):
                raise RuntimeError(
                    "grounded initialization did not cover hybrid trunk tensor "
                    f"{key}"
                )
    return {
        "inherited": tuple(sorted(inherited)),
        "fresh": fresh,
    }


__all__ = [
    "HYBRID_CANDIDATE_Q_VERSION",
    "HybridCandidateQConfig",
    "HybridCandidateQOutput",
    "HybridDomainCandidateQModel",
    "initialize_hybrid_from_grounded",
]
