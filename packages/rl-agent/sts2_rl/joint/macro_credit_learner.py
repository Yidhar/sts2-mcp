"""One fresh-cohort update with long macro credit and bounded recurrent graphs.

Only natural references train ordinary state occupancy. Counterfactual suffixes
produce conditional-action labels, never extra ordinary policy/value examples.
All parameters stay fixed through no-grad target reconstruction and chunk-wise
gradient accumulation; one AdamW step closes the cohort. No replay, target Q,
saved behavior hidden state, scheduler or EMA state is owned by this learner.
"""

from __future__ import annotations

import contextlib
import math
import random
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, TypeAlias, cast

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from sts2_rl.encoding import EncodedDecisionSnapshot, EncodedWorldSnapshot
from sts2_rl.encoding.snapshot import (
    PackedSnapshotBatch,
    collate_encoded_snapshots,
    collate_encoded_worlds,
    pack_encoded_snapshots_to_device,
    scatter_packed_snapshots,
)
from sts2_rl.models.grounded_candidate import WorldTokenBatch, _safe_valid_mask
from sts2_rl.models.hybrid_actor_critic import HybridActorCriticOutput, HybridDomainActorCriticModel

from .macro_credit_data import (
    MacroAfterstateNode,
    MacroCounterfactualRoot,
    MacroCreditCohort,
    MacroCreditTrajectory,
    macro_mixture_log_probs,
)
from .macro_returns import mixed_vtrace_lambda_targets
from .macro_targets import (
    MacroPolicyProjection,
    project_hierarchical_macro_policy,
    project_macro_policy,
    project_subset_macro_policy,
)
from .shadow_price import evaluate_combat_value_from_snapshot
from .transitions import HybridVTraceStep

MACRO_CREDIT_LEARNER_VERSION: Final = "sts2-fresh-macro-credit-learner-v1"
MacroCreditForward: TypeAlias = Callable[[EncodedDecisionSnapshot, Tensor | None], tuple[Tensor, Tensor, Tensor]]
_PURE_VALUE_PARAM_PREFIXES: Final[tuple[str, ...]] = (
    "state_value_head.",
    "state_revival_",
    "durable_",
)


@dataclass(frozen=True, slots=True)
class MacroCreditLearnerConfig:
    learning_rate: float = 1e-4
    shared_learning_rate_scale: float = 0.5
    weight_decay: float = 0.01
    macro_trace_lambda: float = 0.95
    macro_kl: float = 0.02
    chunk_steps: int = 16
    combat_unroll_steps: int = 16
    macro_policy_weight: float = 1.0
    macro_value_weight: float = 0.5
    combat_policy_weight: float = 1.0
    combat_value_weight: float = 0.5
    combat_entropy_weight: float = 0.02
    rho_clip: float = 1.0
    c_clip: float = 1.0
    policy_rho_clip: float = 1.0
    gradient_clip_norm: float = 1.0
    likelihood_tolerance: float = 1e-5
    target_mode: str = "counterfactual_rollout"
    trainable_scope: str = "all"
    durable_warmup_updates: int = 6
    combat_actor_warmup_updates: int = 8
    macro_entropy_weight: float = 0.005
    durable_post_value_weight: float = 0.5
    eval_batch_size: int = 32
    chunk_batch_size: int = 64
    allow_tf32: bool = False
    separate_value_grad_clip: bool = False
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
    domain_layers: int = 1
    joint_layers: int = 2
    enable_partial_afterstates: bool = False
    enable_partial_afterstate_conditional_mpo: bool = False
    enable_manual_confirm_selection_afterstates: bool = False
    durable_stage_baseline_weight: float = 0.5
    enable_hierarchical_afterstate_mpo: bool = False
    macro_mpo_min_temperature: float = 0.0
    macro_within_branch_kl: float = 0.04
    macro_within_branch_mpo_weight: float = 0.5
    revival_budget: int = 0
    zero_macro_pace_penalty: bool = False
    clip_macro_pg_fallback: bool = False
    macro_pg_clip_eps: float = 0.20
    macro_pg_adv_clip: float = 0.25
    macro_pg_kl_weight: float = 0.05
    macro_pg_within_branch_entropy_only: bool = False
    clip_combat_pg: bool = False
    combat_pg_clip_eps: float = 0.15
    combat_pg_adv_clip: float = 0.25
    combat_pg_kl_weight: float = 0.05
    distill_teacher_rollout_updates: int = 0
    distill_total_updates: int = 0
    distill_epochs_per_cohort: int = 3
    distill_policy_kl_weight: float = 1.0
    distill_value_weight: float = 0.5
    distill_afterstate_margin_weight: float = 1.0
    kickstart_kl_weight: float = 0.25
    kickstart_kl_floor: float = 0.0
    enable_build_conditioned_durable_value: bool = False
    enable_twin_durable_value_heads: bool = False
    double_mpo_pessimism_coef: float = 0.0
    ppo_epochs_per_cohort: int = 1
    ppo_minibatches_per_cohort: int = 1
    target_macro_step_kl: float = 0.0
    macro_mpo_trust_kl_weight: float = 0.0
    enable_bf16_amp: bool = False
    freeze_trunk_during_durable_warmup: bool = False
    enable_post_step_kl_line_search: bool = False
    line_search_max_backtracks: int = 4
    line_search_min_scale: float = 0.10

    def __post_init__(self) -> None:
        for bool_name in (
            "allow_tf32",
            "enable_bf16_amp",
            "freeze_trunk_during_durable_warmup",
            "enable_post_step_kl_line_search",
            "separate_value_grad_clip",
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
            "enable_partial_afterstates",
            "enable_partial_afterstate_conditional_mpo",
            "enable_manual_confirm_selection_afterstates",
            "enable_hierarchical_afterstate_mpo",
            "zero_macro_pace_penalty",
            "clip_macro_pg_fallback",
            "macro_pg_within_branch_entropy_only",
            "clip_combat_pg",
        ):
            if not isinstance(getattr(self, bool_name), bool):
                raise ValueError(f"macro credit {bool_name} must be a boolean")
        if self.enable_draw_weighted_durable_value and not self.enable_persistent_only_durable_value:
            raise ValueError(
                "macro credit enable_draw_weighted_durable_value requires enable_persistent_only_durable_value=True"
            )
        if self.enable_deconfounded_durable_value and not self.enable_draw_weighted_durable_value:
            raise ValueError(
                "macro credit enable_deconfounded_durable_value requires enable_draw_weighted_durable_value=True"
            )
        if (
            self.enable_monotone_deconfounded_durable_value
            and not self.enable_deconfounded_durable_value
        ):
            raise ValueError(
                "macro credit enable_monotone_deconfounded_durable_value requires enable_deconfounded_durable_value=True"
            )
        if (
            self.enable_balanced_deconfounded_durable_value
            and not self.enable_monotone_deconfounded_durable_value
        ):
            raise ValueError(
                "macro credit enable_balanced_deconfounded_durable_value requires enable_monotone_deconfounded_durable_value=True"
            )
        if (
            self.enable_expressive_synergy_backbone
            and not self.enable_balanced_deconfounded_durable_value
        ):
            raise ValueError(
                "macro credit enable_expressive_synergy_backbone requires enable_balanced_deconfounded_durable_value=True"
            )
        if (
            self.enable_debloated_synergy_durable_value
            and not self.enable_expressive_synergy_backbone
        ):
            raise ValueError(
                "macro credit enable_debloated_synergy_durable_value requires enable_expressive_synergy_backbone=True"
            )
        if (
            self.enable_build_conditioned_durable_value
            and not self.enable_debloated_synergy_durable_value
        ):
            raise ValueError(
                "macro credit enable_build_conditioned_durable_value requires enable_debloated_synergy_durable_value=True"
            )
        for layer_name in ("domain_layers", "joint_layers"):
            layer_val = getattr(self, layer_name)
            if isinstance(layer_val, bool) or not isinstance(layer_val, int) or not 1 <= layer_val <= 12:
                raise ValueError(f"macro credit {layer_name} must be an integer in [1, 12]")
        if (
            self.enable_partial_afterstate_conditional_mpo
            and not self.enable_partial_afterstates
        ):
            raise ValueError(
                "macro credit enable_partial_afterstate_conditional_mpo requires enable_partial_afterstates=True"
            )
        if self.target_mode not in {"counterfactual_rollout", "combat_shadow_price", "durable_afterstate_v2"}:
            raise ValueError(
                "macro credit target_mode must be 'counterfactual_rollout', "
                f"'combat_shadow_price', or 'durable_afterstate_v2', got {self.target_mode}"
            )
        if self.enable_twin_durable_value_heads and self.target_mode != "durable_afterstate_v2":
            raise ValueError(
                "macro credit enable_twin_durable_value_heads requires target_mode='durable_afterstate_v2'"
            )
        if self.trainable_scope not in {"all", "macro_only"}:
            raise ValueError(f"macro credit trainable_scope must be 'all' or 'macro_only', got {self.trainable_scope}")
        for name in (
            "chunk_steps",
            "combat_unroll_steps",
            "distill_epochs_per_cohort",
            "ppo_epochs_per_cohort",
            "ppo_minibatches_per_cohort",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 16:
                raise ValueError(f"macro credit {name} must be an integer in [1, 16]")
        for name in ("eval_batch_size", "chunk_batch_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"macro credit {name} must be a positive integer")
        for name in (
            "durable_warmup_updates",
            "combat_actor_warmup_updates",
            "revival_budget",
            "distill_teacher_rollout_updates",
            "distill_total_updates",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"macro credit {name} must be a non-negative integer")
        if self.distill_teacher_rollout_updates > self.distill_total_updates:
            raise ValueError(
                "macro credit distill_teacher_rollout_updates must be <= distill_total_updates"
            )
        if self.distill_total_updates > 0 and self.target_mode != "durable_afterstate_v2":
            raise ValueError(
                "macro credit teacher distillation requires target_mode='durable_afterstate_v2'"
            )
        positive = (
            "learning_rate",
            "shared_learning_rate_scale",
            "rho_clip",
            "c_clip",
            "policy_rho_clip",
            "gradient_clip_norm",
            "likelihood_tolerance",
            "macro_pg_clip_eps",
            "macro_pg_adv_clip",
            "combat_pg_clip_eps",
            "combat_pg_adv_clip",
        )
        non_negative = (
            "weight_decay",
            "macro_kl",
            "macro_mpo_min_temperature",
            "macro_within_branch_kl",
            "macro_within_branch_mpo_weight",
            "macro_policy_weight",
            "macro_value_weight",
            "combat_policy_weight",
            "combat_value_weight",
            "combat_entropy_weight",
            "macro_entropy_weight",
            "durable_post_value_weight",
            "durable_stage_baseline_weight",
            "macro_pg_kl_weight",
            "combat_pg_kl_weight",
            "distill_policy_kl_weight",
            "distill_value_weight",
            "distill_afterstate_margin_weight",
            "kickstart_kl_weight",
            "kickstart_kl_floor",
            "double_mpo_pessimism_coef",
            "target_macro_step_kl",
            "macro_mpo_trust_kl_weight",
        )
        for name in (*positive, *non_negative, "macro_trace_lambda"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0 or (name in positive and value == 0):
                raise ValueError(f"macro credit {name} has an invalid finite range")
        if self.kickstart_kl_floor > self.kickstart_kl_weight:
            raise ValueError("macro credit kickstart_kl_floor must be <= kickstart_kl_weight")
        if (
            isinstance(self.line_search_max_backtracks, bool)
            or not isinstance(self.line_search_max_backtracks, int)
            or not 0 <= self.line_search_max_backtracks <= 16
        ):
            raise ValueError("macro credit line_search_max_backtracks must be an integer in [0, 16]")
        if (
            isinstance(self.line_search_min_scale, bool)
            or not math.isfinite(self.line_search_min_scale)
            or not 0.0 < self.line_search_min_scale <= 1.0
        ):
            raise ValueError("macro credit line_search_min_scale must be in (0, 1]")
        if self.macro_trace_lambda > 1 or self.c_clip > self.rho_clip:
            raise ValueError("macro lambda must be <= 1 and c_clip <= rho_clip")


@dataclass(slots=True)
class _AfterstateStepTarget:
    node: MacroAfterstateNode
    action_values: Tensor
    projection: MacroPolicyProjection
    post_target: Tensor
    has_chosen_post: bool = True
    action_disagreement: Tensor | None = None


@dataclass(slots=True)
class _CachedEvalGroup:
    step_actives: list[list[int]]
    step_packed: list[PackedSnapshotBatch]
    boot_indices: list[int]
    boot_packed: PackedSnapshotBatch | None


@dataclass(slots=True)
class _Evaluation:
    values: Tensor
    selected_logs: Tensor
    bootstrap_value: Tensor | None
    # Ephemeral CPU checkpoints rebuilt under THIS unchanged teacher, never
    # collector/behavior hidden states and never serialized as training data.
    boundary_hidden: dict[int, Tensor | None]
    root_logs: Tensor | None
    forward_calls: int
    maximum_likelihood_error: float
    durable_values: Tensor | None = None
    durable_bootstrap_value: Tensor | None = None
    step_log_probs: dict[int, Tensor] | None = None
    post_step_hidden: dict[int, Tensor] | None = None


@dataclass(slots=True)
class _PreparedReference:
    root: MacroCounterfactualRoot | None
    reference: MacroCreditTrajectory
    evaluation: _Evaluation
    macro_values: Tensor
    combat_values: Tensor
    combat_advantages: Tensor
    projection: MacroPolicyProjection | None
    action_values: Tensor
    durable_state_targets: Tensor | None = None
    durable_action_targets: Tensor | None = None
    durable_macro_advantages: Tensor | None = None
    afterstate_targets: dict[int, _AfterstateStepTarget] | None = None
    step_revival_ctx: Tensor | None = None
    post_revival_ctx: Tensor | None = None


def _finite_tree(value: Any) -> bool:
    if isinstance(value, Tensor):
        return not (value.is_floating_point() or value.is_complex()) or bool(torch.isfinite(value).all().item())
    if isinstance(value, Mapping):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, tuple | list):
        return all(_finite_tree(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def _configure_cuda_tf32(device: torch.device, allow_tf32: bool) -> None:
    if device.type == "cuda":
        flag = bool(allow_tf32)
        torch.backends.cuda.matmul.allow_tf32 = flag
        torch.backends.cudnn.allow_tf32 = flag
        torch.set_float32_matmul_precision("high" if flag else "highest")


def hierarchical_macro_entropy(
    safe_logs: Tensor,
    masks: Tensor,
    policy_branch_ids: Tensor,
    *,
    include_branch_entropy: bool = True,
) -> Tensor:
    """Cardinality-neutral macro entropy: H(pi_branch) + mean_{multi-item b} H(pi_{a|b}).

    Flat Shannon entropy H(pi) = H(pi_branch) + sum_b pi_b H(pi_{a|b}) gives a
    multi-candidate branch of size K a +ln(K) gradient wrt its branch probability
    relative to a singleton branch (such as Rest, Skip, or Leave). Averaging
    within-branch conditional entropy uniformly over active multi-candidate
    branches removes branch-cardinality bias while preserving exploration both
    across branches and among targets within each branch. When
    ``include_branch_entropy=False``, returns only the mean within-branch term
    so cross-branch refusal probabilities (e.g. ``leave``) receive zero
    unconditional entropy uplift on non-afterstate surfaces.
    """
    batch_size, action_count = safe_logs.shape
    if action_count == 0:
        return torch.zeros((batch_size,), device=safe_logs.device, dtype=safe_logs.dtype)
    branch_ids = policy_branch_ids.long().clamp(min=0, max=max(0, action_count - 1))
    branch_axis = torch.arange(action_count, device=safe_logs.device).view(1, action_count, 1)
    in_branch = (branch_ids.unsqueeze(1) == branch_axis) & masks.unsqueeze(1)
    branch_active = in_branch.any(dim=-1)
    safe_in_branch = _safe_valid_mask(in_branch)

    scoped_logs = (
        safe_logs.unsqueeze(1)
        .expand(-1, action_count, -1)
        .masked_fill(~safe_in_branch, -torch.inf)
    )
    branch_log_p = torch.logsumexp(scoped_logs, dim=-1)
    safe_branch_log_p = branch_log_p.masked_fill(~branch_active, 0.0)
    branch_p = safe_branch_log_p.exp().masked_fill(~branch_active, 0.0)
    branch_entropy = -(branch_p * safe_branch_log_p).sum(dim=-1)

    within_log_p = (scoped_logs - safe_branch_log_p.unsqueeze(-1)).masked_fill(~in_branch, 0.0)
    within_p = within_log_p.exp().masked_fill(~in_branch, 0.0)
    within_ent = -(within_p * within_log_p).sum(dim=-1)
    multi_branch = in_branch.sum(dim=-1) > 1
    multi_count = multi_branch.sum(dim=-1).clamp_min(1).to(dtype=safe_logs.dtype)
    mean_within_entropy = torch.where(multi_branch, within_ent, 0.0).sum(dim=-1) / multi_count
    if include_branch_entropy:
        return branch_entropy + mean_within_entropy
    return mean_within_entropy


def _rss_peak() -> int | None:
    if sys.platform == "win32":
        return None
    import resource

    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


class MacroCreditLearner:
    """Frozen-target accumulation, exactly one optimizer step per cohort.

    L = .5 [w_mp sum_root CE/R_all + w_mv sum_macro Huber/N_macro]
      + .5 [w_cp sum_choice_combat PG/N_choice_combat
            + w_cv sum_combat Huber/N_combat - w_e sum_choice_combat H/N_choice_combat].

    Disabled/equal-outcome roots contribute zero CE but remain in R_all.
    Each reference root's prior-weighted value REPLACES its ordinary long value
    label. No macro sampled actor or macro entropy is added. Missing combat
    contributes zero, retaining the explicit .5 domain weights rather than
    silently changing the scale. Short combat targets and BPTT chunk boundaries
    are independent: changing BPTT length never changes the loss denominators.

    Successful boundary recovery needs only model, optimizer, config and update.
    A failed optimizer/post-step validation poisons this instance; reload the
    last committed checkpoint instead of attempting to continue mutated state.
    """

    def __init__(
        self, *, model: nn.Module, config: MacroCreditLearnerConfig | None = None,
        update: int = 0, forward: MacroCreditForward | None = None,
        teacher_model: HybridDomainActorCriticModel | None = None,
    ) -> None:
        self.model = model
        self.config = config or MacroCreditLearnerConfig()
        self.update = update
        self._validate_update()
        self._forward_override = forward
        self.teacher_model: HybridDomainActorCriticModel | None = None
        if self.config.target_mode == "durable_afterstate_v2":
            if self._forward_override is not None:
                raise ValueError("durable_afterstate_v2 requires a HybridDomainActorCriticModel without forward override")
            if (
                not isinstance(model, HybridDomainActorCriticModel)
                or not model.actor_critic_config.enable_durable_value_head
            ):
                raise ValueError(
                    "durable_afterstate_v2 requires HybridDomainActorCriticModel with enable_durable_value_head=True"
                )
        if self.config.trainable_scope == "macro_only":
            macro_tokens = (
                "macro_stream",
                "run_recurrent",
                "branch_policy_head",
                "branch_vitals_head",
                "candidate_role_",
                "durable_cycle_",
                "durable_deck_",
                "durable_hp_",
                "durable_liquid_",
                "durable_progress_vitals_",
                "durable_revival_",
                "durable_shop_",
                "durable_stage_",
                "durable_twin_",
                "durable_value_head",
                "durable_vitals_",
                "persistent_build_",
                "persistent_vitals_",
                "state_revival_",
            )
            for name, item in model.named_parameters():
                item.requires_grad_(any(token in name for token in macro_tokens))
        elif self.config.trainable_scope == "all":
            if not all(parameter.requires_grad for parameter in model.parameters()):
                raise ValueError("macro learner requires a non-empty, trainable actor-critic model")
        else:
            raise ValueError(f"unknown trainable_scope: {self.config.trainable_scope}")
        self._parameters = [p for p in model.parameters() if p.requires_grad]
        if not self._parameters:
            raise ValueError("macro learner requires a non-empty, trainable actor-critic model")
        self._is_pure_value_param = [
            name.startswith(_PURE_VALUE_PARAM_PREFIXES)
            for name, p in model.named_parameters()
            if p.requires_grad
        ]
        isolate_stage_clip = bool(
            (
                self.config.enable_balanced_deconfounded_durable_value
                or (
                    isinstance(model, HybridDomainActorCriticModel)
                    and model.actor_critic_config.enable_balanced_deconfounded_durable_value
                )
            )
            and self.config.durable_stage_baseline_weight > 0.0
        )
        base_value_prefixes = (
            "state_value_head",
            "state_revival_",
            "durable_cycle_",
            "durable_deck_",
            "durable_hp_",
            "durable_liquid_",
            "durable_progress_vitals_",
            "durable_shop_",
            "durable_twin_",
            "durable_value_head",
            "durable_vitals_",
            "durable_revival_",
        )
        value_prefixes = (
            base_value_prefixes
            if isolate_stage_clip
            else (*base_value_prefixes, "durable_stage_")
        )
        all_value_or_stage_prefixes = (*base_value_prefixes, "durable_stage_")
        self._stage_baseline_parameters = (
            [
                p
                for name, p in model.named_parameters()
                if p.requires_grad and name.startswith("durable_stage_")
            ]
            if isolate_stage_clip
            else []
        )
        self._value_head_parameters = [
            p for name, p in model.named_parameters() if p.requires_grad and name.startswith(value_prefixes)
        ]
        self._non_value_parameters = [
            p
            for name, p in model.named_parameters()
            if p.requires_grad and not name.startswith(all_value_or_stage_prefixes)
        ]
        if any("candidate_q_head" in name or "target_q" in name for name, _ in model.named_parameters()):
            raise ValueError("macro learner must not own a learned Q head")
        all_params = list(model.parameters())
        parameter = all_params[0]
        self.device = parameter.device
        self.dtype = parameter.dtype
        _configure_cuda_tf32(self.device, self.config.allow_tf32)
        local_tokens = (
            "macro_stream",
            "combat_stream",
            "macro_stream_norm",
            "combat_stream_norm",
            "candidate_policy_head",
            "candidate_role_",
            "state_value_head",
            "state_revival_",
            "branch_policy_head",
            "branch_vitals_",
            "durable_cycle_",
            "durable_deck_",
            "durable_hp_",
            "durable_liquid_",
            "durable_progress_vitals_",
            "durable_revival_",
            "durable_shop_",
            "durable_stage_",
            "durable_twin_",
            "durable_value_head",
            "durable_vitals_",
            "persistent_build_",
            "persistent_vitals_",
        )
        local: list[nn.Parameter] = []
        shared: list[nn.Parameter] = []
        for name, item in model.named_parameters():
            if not item.requires_grad:
                continue
            is_local = name.startswith(local_tokens) and not name.startswith("durable_deck_synergy_")
            (local if is_local else shared).append(item)
        groups: list[dict[str, Any]] = []
        if local:
            groups.append({"params": local, "lr": self.config.learning_rate})
        if shared:
            groups.append({"params": shared, "lr": self.config.learning_rate * self.config.shared_learning_rate_scale})
        self.optimizer = torch.optim.AdamW(groups, weight_decay=self.config.weight_decay)
        self.model.eval()
        self.set_teacher_model(teacher_model)
        self.metrics: dict[str, Any] = {"version": MACRO_CREDIT_LEARNER_VERSION, "update": self.update}
        self._failed = False
        self._resource_guard: Callable[[], None] | None = None
        self._resource_checks = 0

    def set_teacher_model(self, teacher_model: HybridDomainActorCriticModel | None) -> None:
        if teacher_model is None:
            self.teacher_model = None
            return
        if not isinstance(teacher_model, HybridDomainActorCriticModel):
            raise TypeError("teacher_model must be a HybridDomainActorCriticModel")
        if teacher_model is self.model:
            raise ValueError("teacher_model must be a distinct frozen model instance")
        if (
            self.config.target_mode == "durable_afterstate_v2"
            and not teacher_model.actor_critic_config.enable_durable_value_head
        ):
            raise ValueError("teacher_model requires enable_durable_value_head=True")
        teacher_model.to(self.device)
        teacher_model.eval()
        teacher_model.requires_grad_(False)
        self.teacher_model = teacher_model

    def release_teacher(self) -> None:
        self.teacher_model = None

    @property
    def active_rollout_model(self) -> HybridDomainActorCriticModel:
        if self.update < self.config.distill_teacher_rollout_updates:
            if self.teacher_model is None:
                raise RuntimeError(
                    "distill_teacher_rollout_updates requires an attached frozen teacher_model"
                )
            return self.teacher_model
        return cast(HybridDomainActorCriticModel, self.model)

    @property
    def policy_version(self) -> int:
        return self.update

    def _validate_update(self) -> None:
        if isinstance(self.update, bool) or not isinstance(self.update, int) or self.update < 0:
            raise ValueError("macro learner update must be a non-negative integer")

    def _use_revival_ctx(self, target_model: nn.Module | None = None) -> bool:
        mod = self.model if target_model is None else target_model
        return bool(
            self.config.enable_asymmetric_revival_critic
            and isinstance(mod, HybridDomainActorCriticModel)
            and mod.actor_critic_config.enable_asymmetric_revival_critic
        )

    def _uses_stateless_durable_value(self, target_model: nn.Module | None = None) -> bool:
        mod = self.model if target_model is None else target_model
        return bool(
            isinstance(mod, HybridDomainActorCriticModel)
            and mod.actor_critic_config.enable_durable_value_head
            and mod.actor_critic_config.enable_persistent_only_durable_value
            and mod.actor_critic_config.enable_monotone_deconfounded_durable_value
        )

    def _autocast_ctx(self) -> Any:
        if self.config.enable_bf16_amp and self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def _has_high_vram(self) -> bool:
        if self.device.type != "cuda":
            return False
        try:
            return int(torch.cuda.get_device_properties(self.device).total_memory) >= 40 * (1024**3)
        except Exception:
            return False

    def _critic_revival_vector(self, revivals_used: int, cum_revival_penalty: float) -> list[float]:
        budget = self.config.revival_budget
        u = max(0, int(revivals_used))
        pen = max(0.0, float(cum_revival_penalty))
        ratio = min(u / max(budget, 1), 2.0) if budget > 0 else 0.0
        at_permadeath_cliff = 1.0 if (budget > 0 and u >= budget) else 0.0
        near_exhaust = 1.0 if (budget > 0 and u >= max(1, budget - 2)) else 0.0
        return [
            ratio,
            min(ratio * ratio, 4.0),
            0.25 * math.log1p(u),
            min(pen, 2.0),
            at_permadeath_cliff,
            near_exhaust,
        ]

    def _trajectory_critic_revival_contexts(
        self, trajectory: MacroCreditTrajectory
    ) -> tuple[Tensor, Tensor, Tensor]:
        u = 0
        cum_pen = 0.0
        step_rows: list[list[float]] = []
        post_rows: list[list[float]] = []
        for facts in trajectory.reward_facts:
            step_rows.append(self._critic_revival_vector(u, cum_pen))
            u += int(facts.revivals_used_delta)
            cum_pen += max(0.0, -float(facts.revival_penalty))
            post_rows.append(self._critic_revival_vector(u, cum_pen))
        boot_row = self._critic_revival_vector(u, cum_pen)
        return (
            torch.tensor(step_rows, dtype=torch.float32),
            torch.tensor(post_rows, dtype=torch.float32),
            torch.tensor([boot_row], dtype=torch.float32),
        )

    def _effective_step_reward(self, trajectory: MacroCreditTrajectory, index: int) -> float:
        step = trajectory.steps[index]
        if self.config.zero_macro_pace_penalty and step.control_domain == "macro":
            return float(step.reward - trajectory.reward_facts[index].pace_penalty)
        return float(step.reward)

    def _forward_full(
        self,
        snapshot: EncodedDecisionSnapshot,
        hidden: Tensor | None,
        *,
        critic_revival_context: Tensor | None = None,
        model_override: HybridDomainActorCriticModel | None = None,
        compute_durable_value: bool = True,
    ) -> tuple[Tensor, Tensor, Tensor | None, Tensor]:
        self._check_resource()
        durable_value: Tensor | None = None
        active_mod = self.model if model_override is None else model_override
        if self._forward_override is not None and model_override is None:
            logits, value, next_hidden = self._forward_override(snapshot, hidden)
        else:
            batch = collate_encoded_snapshots(
                (snapshot,), expected_config=snapshot.config,
                expected_fingerprint=snapshot.encoding_fingerprint, device=self.device,
            )
            with self._autocast_ctx():
                if isinstance(active_mod, HybridDomainActorCriticModel):
                    output = cast(
                        HybridActorCriticOutput,
                        active_mod(
                            batch,
                            hidden,
                            critic_revival_context=(
                                critic_revival_context if self._use_revival_ctx(active_mod) else None
                            ),
                            compute_durable_value=compute_durable_value,
                            validate=False,
                        ),
                    )
                else:
                    output = cast(HybridActorCriticOutput, active_mod(batch, hidden, validate=False))
            logits, value, next_hidden = output.policy_logits[0], output.value.reshape(()), output.recurrent_state
            if output.durable_value is not None:
                durable_value = output.durable_value.reshape(())
        if logits.ndim != 1 or logits.numel() < snapshot.candidate_count or value.numel() != 1:
            raise ValueError("macro actor-critic forward has invalid logits/value shapes")
        if next_hidden.ndim != 2 or next_hidden.shape[0] != 1:
            raise ValueError("macro actor-critic forward must return one recurrent state")
        checked: tuple[Tensor, ...] = (
            (logits, value, next_hidden, durable_value)
            if durable_value is not None
            else (logits, value, next_hidden)
        )
        if not all(bool(torch.isfinite(item).all().item()) for item in checked):
            raise FloatingPointError("macro actor-critic forward is non-finite")
        return logits, value.reshape(()), durable_value, next_hidden

    def _forward(self, snapshot: EncodedDecisionSnapshot, hidden: Tensor | None) -> tuple[Tensor, Tensor, Tensor]:
        logits, value, _, next_hidden = self._forward_full(snapshot, hidden)
        return logits, value, next_hidden

    def _forward_durable_worlds(
        self,
        worlds: tuple[EncodedWorldSnapshot, ...],
        hidden: Tensor | None,
        *,
        critic_revival_context: Tensor | None = None,
        model_override: HybridDomainActorCriticModel | None = None,
    ) -> Tensor:
        self._check_resource()
        active_mod = cast(
            HybridDomainActorCriticModel,
            self.model if model_override is None else model_override,
        )
        world_batch, domain_ids = collate_encoded_worlds(
            worlds,
            expected_config=worlds[0].config,
            expected_fingerprint=worlds[0].encoding_fingerprint,
            device=self.device,
        )
        expanded_hidden = (
            None
            if hidden is None
            else hidden.expand(len(worlds), -1).contiguous()
        )
        expanded_rev = (
            None
            if critic_revival_context is None
            else critic_revival_context.expand(len(worlds), -1).contiguous()
        )
        with self._autocast_ctx():
            durable_vals, _ = active_mod.forward_durable_value(
                world_batch,
                domain_ids,
                expanded_hidden,
                critic_revival_context=expanded_rev,
                validate=False,
            )
        if durable_vals.ndim != 1 or durable_vals.numel() != len(worlds):
            raise ValueError("forward_durable_value returned invalid batch shape")
        if not bool(torch.isfinite(durable_vals).all().item()):
            raise FloatingPointError("forward_durable_value output is non-finite")
        return durable_vals

    def _forward_durable_worlds_pair(
        self,
        worlds: tuple[EncodedWorldSnapshot, ...],
        hidden: Tensor | None,
        *,
        critic_revival_context: Tensor | None = None,
        model_override: HybridDomainActorCriticModel | None = None,
    ) -> tuple[Tensor, Tensor]:
        self._check_resource()
        active_mod = cast(
            HybridDomainActorCriticModel,
            self.model if model_override is None else model_override,
        )
        world_batch, domain_ids = collate_encoded_worlds(
            worlds,
            expected_config=worlds[0].config,
            expected_fingerprint=worlds[0].encoding_fingerprint,
            device=self.device,
        )
        expanded_hidden = (
            None
            if hidden is None
            else hidden.expand(len(worlds), -1).contiguous()
        )
        expanded_rev = (
            None
            if critic_revival_context is None
            else critic_revival_context.expand(len(worlds), -1).contiguous()
        )
        with self._autocast_ctx():
            v1, v2, _ = active_mod.forward_durable_value_pair(
                world_batch,
                domain_ids,
                expanded_hidden,
                critic_revival_context=expanded_rev,
                validate=False,
            )
        if v1.ndim != 1 or v1.numel() != len(worlds) or tuple(v2.shape) != tuple(v1.shape):
            raise ValueError("forward_durable_value_pair returned invalid batch shape")
        if not bool((torch.isfinite(v1).all() & torch.isfinite(v2).all()).item()):
            raise FloatingPointError("forward_durable_value_pair output is non-finite")
        return v1, v2

    def _check_resource(self) -> None:
        if self._resource_guard is not None:
            self._resource_guard()
            self._resource_checks += 1

    @staticmethod
    def _policy(step: HybridVTraceStep, logits: Tensor) -> tuple[Tensor, Tensor, bool]:
        mask = torch.tensor(step.snapshot.action_mask, device=logits.device, dtype=torch.bool)
        logs = torch.log_softmax(logits[:mask.numel()].masked_fill(~mask, -torch.inf), dim=0)
        legal = logs[mask]
        if not bool(torch.isfinite(legal).all().item()):
            raise FloatingPointError("macro legal log probabilities are non-finite")
        entropy = -(legal.exp() * legal).sum()
        return logs, entropy, bool(int(mask.sum().item()) > 1)

    @torch.no_grad()
    def _evaluate(
        self,
        trajectory: MacroCreditTrajectory,
        *,
        root_index: int | None = None,
        model_override: HybridDomainActorCriticModel | None = None,
        check_likelihood: bool = True,
        record_all_choice_logs: bool = False,
        compute_durable_value: bool = True,
    ) -> _Evaluation:
        _configure_cuda_tf32(self.device, self.config.allow_tf32)
        active_mod = self.model if model_override is None else model_override
        hidden: Tensor | None = None
        calls = 0
        is_durable_v2 = self.config.target_mode == "durable_afterstate_v2"
        need_durable = is_durable_v2 and compute_durable_value
        include_unvisited_partial = bool(self.config.enable_partial_afterstate_conditional_mpo)
        afterstate_indices = (
            {
                node.step_index
                for node in trajectory.afterstate_nodes
                if include_unvisited_partial
                or node.post_world_for_action(trajectory.steps[node.step_index].action_index) is not None
            }
            if is_durable_v2
            else set()
        )
        use_rev = self._use_revival_ctx(active_mod)
        step_rev_cpu: Tensor | None = None
        boot_rev_cpu: Tensor | None = None
        if use_rev:
            step_rev_cpu, _, boot_rev_cpu = self._trajectory_critic_revival_contexts(trajectory)
        for snapshot in trajectory.prefix:
            _, _, _, hidden = self._forward_full(
                snapshot,
                hidden,
                model_override=model_override,
                compute_durable_value=False,
            )
            hidden = hidden.detach()
            calls += 1
        boundaries: dict[int, Tensor | None] = {}
        values: list[Tensor] = []
        durable_values: list[Tensor] = []
        logs: list[Tensor] = []
        step_log_probs: dict[int, Tensor] = {}
        post_step_hidden: dict[int, Tensor] = {}
        root_logs: Tensor | None = None
        maximum_error = 0.0
        for index, step in enumerate(trajectory.steps):
            if root_index is not None and index % self.config.chunk_steps == 0:
                boundaries[index] = None if hidden is None else hidden.detach().cpu().clone()
            rev_step_gpu = (
                None
                if step_rev_cpu is None
                else step_rev_cpu[index : index + 1].to(device=self.device, dtype=self.dtype)
            )
            logits, value, durable_val, hidden = self._forward_full(
                step.snapshot,
                hidden,
                critic_revival_context=rev_step_gpu,
                model_override=model_override,
                compute_durable_value=need_durable,
            )
            hidden = hidden.detach()
            policy, _, _ = self._policy(step, logits)
            selected = policy[step.action_index]
            if check_likelihood and not (index == 0 and trajectory.root_intervened):
                reconstructed_behavior = macro_mixture_log_probs(
                    policy,
                    step.snapshot,
                    trajectory.macro_branch_epsilon,
                )
                error = abs(float(reconstructed_behavior[step.action_index].item()) - step.behavior_log_probability)
                maximum_error = max(maximum_error, error)
                if error > self.config.likelihood_tolerance:
                    raise ValueError(
                        "fresh teacher likelihood differs from recorded stochastic behavior "
                        f"(err={error:.6g}, tol={self.config.likelihood_tolerance:.6g}, "
                        f"step={index}/{len(trajectory.steps)}, domain={step.control_domain})"
                    )
            if index == root_index:
                root_logs = policy.detach().cpu().clone()
            if is_durable_v2:
                if need_durable:
                    if durable_val is None:
                        raise ValueError("durable_afterstate_v2 requires durable_value on every step")
                    durable_values.append(durable_val.detach().cpu().clone())
                if index in afterstate_indices:
                    step_log_probs[index] = policy.detach().cpu().clone()
                    post_step_hidden[index] = hidden.detach().cpu().clone()
                elif (
                    record_all_choice_logs or step.control_domain == "macro"
                ) and int(step.snapshot.action_mask.sum()) > 1:
                    step_log_probs[index] = policy.detach().cpu().clone()
            elif record_all_choice_logs and int(step.snapshot.action_mask.sum()) > 1:
                step_log_probs[index] = policy.detach().cpu().clone()
            values.append(value.detach().cpu().clone())
            logs.append(selected.detach().cpu().clone())
            calls += 1
        bootstrap = None
        durable_bootstrap = None
        if trajectory.bootstrap_snapshot is not None:
            boot_rev_gpu = (
                None
                if boot_rev_cpu is None
                else boot_rev_cpu.to(device=self.device, dtype=self.dtype)
            )
            _, bootstrap_val, durable_boot, _ = self._forward_full(
                trajectory.bootstrap_snapshot,
                hidden,
                critic_revival_context=boot_rev_gpu,
                model_override=model_override,
                compute_durable_value=need_durable,
            )
            bootstrap = bootstrap_val.detach().cpu().clone()
            if need_durable:
                if durable_boot is None:
                    raise ValueError("durable_afterstate_v2 requires durable_value on bootstrap snapshot")
                durable_bootstrap = durable_boot.detach().cpu().clone()
            calls += 1
        return _Evaluation(
            torch.stack(values),
            torch.stack(logs),
            bootstrap,
            boundaries,
            root_logs,
            calls,
            maximum_error,
            durable_values=torch.stack(durable_values) if need_durable else None,
            durable_bootstrap_value=durable_bootstrap,
            step_log_probs=step_log_probs if (is_durable_v2 or record_all_choice_logs) else None,
            post_step_hidden=post_step_hidden if is_durable_v2 else None,
        )

    def _targets(
        self, trajectory: MacroCreditTrajectory, evaluation: _Evaluation, *, start: int = 0,
        stop: int | None = None, trace_lambda: float,
    ) -> tuple[Tensor, Tensor]:
        stop = len(trajectory.steps) if stop is None else stop
        steps = trajectory.steps[start:stop]
        values = evaluation.values[start:stop]
        kwargs: dict[str, Any] = {}
        if stop < len(trajectory.steps):
            kwargs = {"bootstrap_value": evaluation.values[stop], "bootstrap_domain": trajectory.steps[stop].control_domain}
        elif evaluation.bootstrap_value is not None:
            kwargs = {"bootstrap_value": evaluation.bootstrap_value, "bootstrap_domain": trajectory.bootstrap_domain}
        result = mixed_vtrace_lambda_targets(
            rewards=values.new_tensor([self._effective_step_reward(trajectory, idx) for idx in range(start, stop)]),
            discounts=values.new_tensor([step.discount for step in steps]), values=values,
            target_log_probs=evaluation.selected_logs[start:stop],
            behavior_log_probs=values.new_tensor([step.behavior_log_probability for step in steps]),
            control_domains=tuple(step.control_domain for step in steps), trace_lambda=trace_lambda,
            rho_clip=self.config.rho_clip, c_clip=self.config.c_clip, **kwargs,
        )
        return result.value_targets, result.action_value_targets

    def _durable_targets(
        self, trajectory: MacroCreditTrajectory, evaluation: _Evaluation, *, trace_lambda: float
    ) -> tuple[Tensor, Tensor]:
        assert evaluation.durable_values is not None
        values = evaluation.durable_values
        kwargs: dict[str, Any] = {}
        if evaluation.durable_bootstrap_value is not None:
            kwargs = {
                "bootstrap_value": evaluation.durable_bootstrap_value,
                "bootstrap_domain": trajectory.bootstrap_domain,
            }
        result = mixed_vtrace_lambda_targets(
            rewards=values.new_tensor(
                [self._effective_step_reward(trajectory, idx) for idx in range(len(trajectory.steps))]
            ),
            discounts=values.new_tensor([step.discount for step in trajectory.steps]),
            values=values,
            target_log_probs=evaluation.selected_logs,
            behavior_log_probs=values.new_tensor([step.behavior_log_probability for step in trajectory.steps]),
            control_domains=tuple(step.control_domain for step in trajectory.steps),
            trace_lambda=trace_lambda,
            rho_clip=self.config.rho_clip,
            c_clip=self.config.c_clip,
            **kwargs,
        )
        return result.value_targets, result.action_value_targets

    @torch.no_grad()
    def _evaluate_durable_batch(
        self,
        references: Sequence[MacroCreditTrajectory],
        *,
        model_override: HybridDomainActorCriticModel | None = None,
        check_likelihood: bool = True,
        record_all_choice_logs: bool = False,
        compute_durable_value: bool = True,
        packed_cache: dict[tuple[int, int], _CachedEvalGroup] | None = None,
    ) -> list[_Evaluation]:
        _configure_cuda_tf32(self.device, self.config.allow_tf32)
        ac_model = cast(
            HybridDomainActorCriticModel,
            self.model if model_override is None else model_override,
        )
        use_rev = self._use_revival_ctx(ac_model)
        results: list[_Evaluation] = []
        high_vram = self._has_high_vram()
        batch_cap = (
            self.config.eval_batch_size
            if high_vram or int(ac_model.config.d_model) < 256
            else min(self.config.eval_batch_size, 16)
        )
        for group_start in range(0, len(references), batch_cap):
            group = references[group_start : group_start + batch_cap]
            m = len(group)
            cache_key = (group_start, m)
            cached_group = packed_cache.get(cache_key) if packed_cache is not None else None
            building_cache = packed_cache is not None and cached_group is None
            new_step_packed: list[PackedSnapshotBatch] = []
            rev_contexts = (
                [self._trajectory_critic_revival_contexts(traj) for traj in group]
                if use_rev
                else None
            )
            hidden = ac_model.initial_state(m, device=self.device)
            has_hidden = [False] * m
            calls = [0] * m
            for i, traj in enumerate(group):
                for snap in traj.prefix:
                    _, _, _, next_h = self._forward_full(
                        snap,
                        hidden[i : i + 1] if has_hidden[i] else None,
                        model_override=ac_model,
                        compute_durable_value=False,
                    )
                    hidden[i] = next_h[0].detach()
                    has_hidden[i] = True
                    calls[i] += 1
            include_unvisited_partial = bool(self.config.enable_partial_afterstate_conditional_mpo)
            afterstate_sets = [
                {
                    node.step_index
                    for node in traj.afterstate_nodes
                    if include_unvisited_partial
                    or node.post_world_for_action(traj.steps[node.step_index].action_index) is not None
                }
                for traj in group
            ]
            boundaries_list: list[dict[int, Tensor | None]] = [{} for _ in range(m)]
            values_list: list[list[Tensor]] = [[] for _ in range(m)]
            durable_values_list: list[list[Tensor]] = [[] for _ in range(m)]
            logs_list: list[list[Tensor]] = [[] for _ in range(m)]
            step_log_probs_list: list[dict[int, Tensor]] = [{} for _ in range(m)]
            post_step_hidden_list: list[dict[int, Tensor]] = [{} for _ in range(m)]
            max_errors = [0.0] * m
            t_max = max((len(traj.steps) for traj in group), default=0)

            step_actives: list[list[int]] = []
            step_had_hidden: list[list[bool]] = []
            step_logs_gpu: list[Tensor] = []
            step_vals_gpu: list[Tensor] = []
            step_durs_gpu: list[Tensor] = []
            boundary_hids_gpu: dict[int, Tensor | None] = {}
            post_hids_gpu: dict[int, Tensor] = {}
            finite_ok = torch.ones((), device=self.device, dtype=torch.bool)

            for t in range(t_max):
                self._check_resource()
                if cached_group is not None:
                    active = cached_group.step_actives[t]
                else:
                    active = [i for i in range(m) if t < len(group[i].steps)]
                step_actives.append(active)
                if t % self.config.chunk_steps == 0:
                    step_had_hidden.append([has_hidden[i] for i in active])
                    boundary_hids_gpu[t] = (
                        hidden[active].detach().clone() if any(has_hidden[i] for i in active) else None
                    )
                else:
                    step_had_hidden.append([])
                if cached_group is not None:
                    batch = scatter_packed_snapshots(cached_group.step_packed[t])
                elif building_cache:
                    snaps = tuple(group[i].steps[t].snapshot for i in active)
                    packed_t = pack_encoded_snapshots_to_device(
                        snaps,
                        cfg=snaps[0].config,
                        dev=self.device,
                    )
                    new_step_packed.append(packed_t)
                    batch = scatter_packed_snapshots(packed_t)
                else:
                    snaps = tuple(group[i].steps[t].snapshot for i in active)
                    batch = collate_encoded_snapshots(
                        snaps,
                        expected_config=snaps[0].config,
                        expected_fingerprint=snaps[0].encoding_fingerprint,
                        device=self.device,
                        validate=False,
                    )
                rev_t_gpu = (
                    None
                    if rev_contexts is None
                    else torch.stack([rev_contexts[i][0][t] for i in active], dim=0).to(
                        device=self.device, dtype=self.dtype
                    )
                )
                with self._autocast_ctx():
                    output = cast(
                        HybridActorCriticOutput,
                        ac_model(
                            batch,
                            hidden[active],
                            critic_revival_context=rev_t_gpu,
                            compute_durable_value=compute_durable_value,
                            validate=False,
                        ),
                    )
                if compute_durable_value and output.durable_value is None:
                    raise ValueError("durable_afterstate_v2 requires durable_value on every step")
                masks = batch.candidates.action_mask
                batch_logs = torch.log_softmax(output.policy_logits.masked_fill(~masks, -torch.inf), dim=-1)
                safe_batch_logs = batch_logs.masked_fill(~masks, 0.0)
                finite_ok = (
                    finite_ok
                    & torch.isfinite(output.policy_logits).all()
                    & torch.isfinite(output.value).all()
                    & torch.isfinite(output.recurrent_state).all()
                    & torch.isfinite(safe_batch_logs).all()
                )
                if compute_durable_value:
                    assert output.durable_value is not None
                    finite_ok = finite_ok & torch.isfinite(output.durable_value).all()
                hidden[active] = output.recurrent_state.detach()
                for i in active:
                    has_hidden[i] = True
                step_logs_gpu.append(batch_logs.detach())
                step_vals_gpu.append(output.value.detach())
                if compute_durable_value:
                    assert output.durable_value is not None
                    step_durs_gpu.append(output.durable_value.detach())
                if any(t in afterstate_sets[i] for i in active):
                    post_hids_gpu[t] = output.recurrent_state.detach().clone()

            if not bool(finite_ok.item()):
                raise FloatingPointError("macro actor-critic forward or legal log probabilities are non-finite")

            if t_max > 0:
                flat_vals_cpu = torch.cat(step_vals_gpu, dim=0).cpu()
                flat_durs_cpu = torch.cat(step_durs_gpu, dim=0).cpu() if compute_durable_value else None
                flat_logs_cpu = torch.cat([lg.reshape(-1) for lg in step_logs_gpu], dim=0).cpu()
                boundary_hids_cpu = {
                    t: (None if h is None else h.cpu()) for t, h in boundary_hids_gpu.items()
                }
                post_hids_cpu = {t: h.cpu() for t, h in post_hids_gpu.items()}

                v_offset = 0
                l_offset = 0
                for t in range(t_max):
                    active = step_actives[t]
                    k_len = len(active)
                    c_width = int(step_logs_gpu[t].shape[1])
                    logs_t_cpu = flat_logs_cpu[l_offset : l_offset + k_len * c_width].reshape(k_len, c_width)
                    vals_t_cpu = flat_vals_cpu[v_offset : v_offset + k_len]
                    durs_t_cpu = (
                        flat_durs_cpu[v_offset : v_offset + k_len]
                        if flat_durs_cpu is not None
                        else None
                    )
                    v_offset += k_len
                    l_offset += k_len * c_width
                    if t % self.config.chunk_steps == 0:
                        hid_snap = boundary_hids_cpu[t]
                        had_h = step_had_hidden[t]
                        for k, i in enumerate(active):
                            boundaries_list[i][t] = (
                                None if not had_h[k] or hid_snap is None else hid_snap[k : k + 1].clone()
                            )
                    post_hid_cpu = post_hids_cpu.get(t)
                    for k, i in enumerate(active):
                        traj = group[i]
                        step = traj.steps[t]
                        cand_n = step.snapshot.candidate_count
                        policy = logs_t_cpu[k, :cand_n].clone()
                        selected = policy[step.action_index]
                        if check_likelihood and not (t == 0 and traj.root_intervened):
                            reconstructed = macro_mixture_log_probs(policy, step.snapshot, traj.macro_branch_epsilon)
                            err = abs(float(reconstructed[step.action_index].item()) - step.behavior_log_probability)
                            max_errors[i] = max(max_errors[i], err)
                            if err > self.config.likelihood_tolerance:
                                raise ValueError(
                                    "fresh teacher likelihood differs from recorded stochastic behavior "
                                    f"(err={err:.6g}, tol={self.config.likelihood_tolerance:.6g}, "
                                    f"step={t}/{len(traj.steps)}, domain={step.control_domain})"
                                )
                        if durs_t_cpu is not None:
                            durable_values_list[i].append(durs_t_cpu[k].clone())
                        values_list[i].append(vals_t_cpu[k].clone())
                        logs_list[i].append(selected.clone())
                        if t in afterstate_sets[i]:
                            assert post_hid_cpu is not None
                            step_log_probs_list[i][t] = policy
                            post_step_hidden_list[i][t] = post_hid_cpu[k : k + 1].clone()
                        elif (
                            record_all_choice_logs or step.control_domain == "macro"
                        ) and int(step.snapshot.action_mask.sum()) > 1:
                            step_log_probs_list[i][t] = policy
                        calls[i] += 1

            boot_indices = (
                cached_group.boot_indices
                if cached_group is not None
                else [i for i, traj in enumerate(group) if traj.bootstrap_snapshot is not None]
            )
            boot_vals: dict[int, Tensor] = {}
            dur_boot_vals: dict[int, Tensor] = {}
            new_boot_packed: PackedSnapshotBatch | None = None
            if boot_indices:
                self._check_resource()
                if cached_group is not None and cached_group.boot_packed is not None:
                    boot_batch = scatter_packed_snapshots(cached_group.boot_packed)
                elif building_cache:
                    boot_snaps = tuple(
                        cast(EncodedDecisionSnapshot, group[i].bootstrap_snapshot) for i in boot_indices
                    )
                    new_boot_packed = pack_encoded_snapshots_to_device(
                        boot_snaps,
                        cfg=boot_snaps[0].config,
                        dev=self.device,
                    )
                    boot_batch = scatter_packed_snapshots(new_boot_packed)
                else:
                    boot_snaps = tuple(
                        cast(EncodedDecisionSnapshot, group[i].bootstrap_snapshot) for i in boot_indices
                    )
                    boot_batch = collate_encoded_snapshots(
                        boot_snaps,
                        expected_config=boot_snaps[0].config,
                        expected_fingerprint=boot_snaps[0].encoding_fingerprint,
                        device=self.device,
                        validate=False,
                    )
                boot_rev_gpu = (
                    None
                    if rev_contexts is None
                    else torch.cat([rev_contexts[i][2] for i in boot_indices], dim=0).to(
                        device=self.device, dtype=self.dtype
                    )
                )
                with self._autocast_ctx():
                    boot_out = cast(
                        HybridActorCriticOutput,
                        ac_model(
                            boot_batch,
                            hidden[boot_indices],
                            critic_revival_context=boot_rev_gpu,
                            compute_durable_value=compute_durable_value,
                            validate=False,
                        ),
                    )
                if compute_durable_value and boot_out.durable_value is None:
                    raise ValueError("durable_afterstate_v2 requires durable_value on bootstrap snapshot")
                boot_ok = (
                    torch.isfinite(boot_out.policy_logits).all()
                    & torch.isfinite(boot_out.value).all()
                    & torch.isfinite(boot_out.recurrent_state).all()
                )
                if compute_durable_value:
                    assert boot_out.durable_value is not None
                    boot_ok = boot_ok & torch.isfinite(boot_out.durable_value).all()
                if not bool(boot_ok.item()):
                    raise FloatingPointError("macro actor-critic bootstrap forward is non-finite")
                b_val_cpu = boot_out.value.detach().cpu()
                b_dur_cpu = (
                    boot_out.durable_value.detach().cpu()
                    if compute_durable_value and boot_out.durable_value is not None
                    else None
                )
                for k, i in enumerate(boot_indices):
                    boot_vals[i] = b_val_cpu[k].clone()
                    if b_dur_cpu is not None:
                        dur_boot_vals[i] = b_dur_cpu[k].clone()
                    calls[i] += 1
            if building_cache and packed_cache is not None:
                packed_cache[cache_key] = _CachedEvalGroup(
                    step_actives=step_actives,
                    step_packed=new_step_packed,
                    boot_indices=boot_indices,
                    boot_packed=new_boot_packed,
                )
            for i in range(m):
                results.append(
                    _Evaluation(
                        torch.stack(values_list[i]),
                        torch.stack(logs_list[i]),
                        boot_vals.get(i),
                        boundaries_list[i],
                        None,
                        calls[i],
                        max_errors[i],
                        durable_values=torch.stack(durable_values_list[i]) if compute_durable_value else None,
                        durable_bootstrap_value=dur_boot_vals.get(i) if compute_durable_value else None,
                        step_log_probs=step_log_probs_list[i],
                        post_step_hidden=post_step_hidden_list[i],
                    )
                )
        return results

    def _project_afterstate_node(
        self,
        *,
        node: MacroAfterstateNode,
        step: HybridVTraceStep,
        q_after: Tensor,
        teacher_logs: Tensor,
        effective_macro_kl: float,
        q_after_twin: Tensor | None = None,
    ) -> MacroPolicyProjection:
        if self.config.enable_hierarchical_afterstate_mpo:
            effective_within_kl = (
                self.config.macro_within_branch_kl
                if self.update >= self.config.durable_warmup_updates
                else 0.0
            )
            branch_ids = torch.tensor(
                step.snapshot.candidates.ids[: teacher_logs.numel(), 13],
                device=teacher_logs.device,
                dtype=torch.long,
            )
            return project_hierarchical_macro_policy(
                det_action_values=q_after.to(dtype=teacher_logs.dtype),
                teacher_log_probs=teacher_logs,
                policy_branch_ids=branch_ids,
                deterministic_indices=node.resolved_candidate_indices if node.is_partial else None,
                branch_max_kl=effective_macro_kl,
                within_max_kl=effective_within_kl,
                min_temperature=self.config.macro_mpo_min_temperature,
                det_action_values_twin=(
                    None if q_after_twin is None else q_after_twin.to(dtype=teacher_logs.dtype)
                ),
                pessimism_coef=self.config.double_mpo_pessimism_coef if q_after_twin is not None else 0.0,
            )
        if q_after_twin is not None:
            q_eff = 0.5 * (q_after + q_after_twin) - self.config.double_mpo_pessimism_coef * (
                q_after - q_after_twin
            ).abs()
        else:
            q_eff = q_after
        if node.is_partial:
            return project_subset_macro_policy(
                det_action_values=q_eff.to(dtype=teacher_logs.dtype),
                teacher_log_probs=teacher_logs,
                deterministic_indices=node.resolved_candidate_indices,
                max_kl=effective_macro_kl,
                min_temperature=self.config.macro_mpo_min_temperature,
            )
        return project_macro_policy(
            action_value_targets=q_eff.to(dtype=teacher_logs.dtype),
            teacher_log_probs=teacher_logs,
            max_kl=effective_macro_kl,
            min_temperature=self.config.macro_mpo_min_temperature,
        )

    @torch.no_grad()
    def _prepare_durable_natural_batch(
        self,
        references: Sequence[MacroCreditTrajectory],
        *,
        model_override: HybridDomainActorCriticModel | None = None,
        check_likelihood: bool = True,
        record_all_choice_logs: bool = False,
        packed_cache: dict[tuple[int, int], _CachedEvalGroup] | None = None,
    ) -> tuple[list[_PreparedReference], int, float]:
        if not references:
            return [], 0, 0.0
        ac_model = cast(
            HybridDomainActorCriticModel,
            self.model if model_override is None else model_override,
        )
        use_rev = self._use_revival_ctx(ac_model)
        rev_all = (
            [self._trajectory_critic_revival_contexts(ref) for ref in references]
            if use_rev
            else None
        )
        evaluations = self._evaluate_durable_batch(
            references,
            model_override=model_override,
            check_likelihood=check_likelihood,
            record_all_choice_logs=record_all_choice_logs,
            packed_cache=packed_cache,
        )
        flat_worlds: list[EncodedWorldSnapshot] = []
        flat_hiddens: list[Tensor] = []
        flat_revs: list[Tensor] = []
        node_slices: list[tuple[int, MacroAfterstateNode, int, int]] = []
        total_calls = sum(ev.forward_calls for ev in evaluations)
        max_error = max((ev.maximum_likelihood_error for ev in evaluations), default=0.0)
        include_unvisited_partial = bool(self.config.enable_partial_afterstate_conditional_mpo)
        for ref_idx, (reference, evaluation) in enumerate(zip(references, evaluations, strict=True)):
            assert evaluation.post_step_hidden is not None
            for node in reference.afterstate_nodes:
                has_chosen_post = (
                    node.post_world_for_action(reference.steps[node.step_index].action_index)
                    is not None
                )
                if not has_chosen_post and not include_unvisited_partial:
                    continue
                start_w = len(flat_worlds)
                w_count = len(node.post_worlds)
                flat_worlds.extend(node.post_worlds)
                saved_h = evaluation.post_step_hidden[node.step_index]
                flat_hiddens.append(saved_h.expand(w_count, -1))
                if rev_all is not None:
                    post_r = rev_all[ref_idx][1][node.step_index : node.step_index + 1]
                    flat_revs.append(post_r.expand(w_count, -1))
                node_slices.append((ref_idx, node, start_w, start_w + w_count))
                total_calls += 1
        use_twin_after = bool(ac_model.actor_critic_config.enable_twin_durable_value_heads)
        if flat_worlds:
            n_worlds = len(flat_worlds)
            perm = sorted(range(n_worlds), key=lambda idx: flat_worlds[idx].world.token_count)
            sorted_worlds = [flat_worlds[idx] for idx in perm]
            stacked_hiddens = torch.cat(flat_hiddens, dim=0)[perm]
            stacked_revs = torch.cat(flat_revs, dim=0)[perm] if rev_all is not None else None
            world_cap = (
                192
                if self._has_high_vram()
                else (
                    64
                    if int(ac_model.config.d_model) >= 256
                    else max(128, min(256, self.config.eval_batch_size * 8))
                )
            )
            gpu_q1_chunks: list[Tensor] = []
            gpu_q2_chunks: list[Tensor] = []
            for w_start in range(0, n_worlds, world_cap):
                self._check_resource()
                w_end = min(w_start + world_cap, n_worlds)
                w_tuple = tuple(sorted_worlds[w_start:w_end])
                world_batch, domain_ids = collate_encoded_worlds(
                    w_tuple,
                    expected_config=w_tuple[0].config,
                    expected_fingerprint=w_tuple[0].encoding_fingerprint,
                    device=self.device,
                    validate=False,
                )
                h_gpu = stacked_hiddens[w_start:w_end].to(device=self.device, dtype=self.dtype)
                r_gpu = (
                    None
                    if stacked_revs is None
                    else stacked_revs[w_start:w_end].to(device=self.device, dtype=self.dtype)
                )
                with self._autocast_ctx():
                    if use_twin_after:
                        dur_v1, dur_v2, _ = ac_model.forward_durable_value_pair(
                            world_batch,
                            domain_ids,
                            h_gpu,
                            critic_revival_context=r_gpu,
                            validate=False,
                        )
                        gpu_q1_chunks.append(dur_v1.detach())
                        gpu_q2_chunks.append(dur_v2.detach())
                    else:
                        dur_vals, _ = ac_model.forward_durable_value(
                            world_batch,
                            domain_ids,
                            h_gpu,
                            critic_revival_context=r_gpu,
                            validate=False,
                        )
                        gpu_q1_chunks.append(dur_vals.detach())
            sorted_q1_gpu = torch.cat(gpu_q1_chunks, dim=0)
            if not bool(torch.isfinite(sorted_q1_gpu).all().item()):
                raise FloatingPointError("forward_durable_value output is non-finite")
            sorted_q1_cpu = sorted_q1_gpu.cpu()
            all_q1_after = torch.empty_like(sorted_q1_cpu)
            all_q1_after[perm] = sorted_q1_cpu
            if use_twin_after:
                sorted_q2_gpu = torch.cat(gpu_q2_chunks, dim=0)
                if not bool(torch.isfinite(sorted_q2_gpu).all().item()):
                    raise FloatingPointError("forward_durable_value_pair output is non-finite")
                sorted_q2_cpu = sorted_q2_gpu.cpu()
                all_q2_after: Tensor | None = torch.empty_like(sorted_q2_cpu)
                assert all_q2_after is not None
                all_q2_after[perm] = sorted_q2_cpu
            else:
                all_q2_after = None
        else:
            all_q1_after = torch.empty(0)
            all_q2_after = torch.empty(0) if use_twin_after else None
        per_ref_nodes: list[list[tuple[MacroAfterstateNode, Tensor, Tensor | None]]] = [
            [] for _ in references
        ]
        for ref_idx, node, s_w, e_w in node_slices:
            per_ref_nodes[ref_idx].append(
                (
                    node,
                    all_q1_after[s_w:e_w],
                    None if all_q2_after is None else all_q2_after[s_w:e_w],
                )
            )

        effective_macro_kl = (
            self.config.macro_kl if self.update >= self.config.durable_warmup_updates else 0.0
        )
        prepared_out: list[_PreparedReference] = []
        for ref_idx, (reference, evaluation) in enumerate(zip(references, evaluations, strict=True)):
            assert evaluation.durable_values is not None and evaluation.step_log_probs is not None
            macro_values, macro_action_targets = self._targets(
                reference, evaluation, trace_lambda=self.config.macro_trace_lambda
            )
            combat_values, combat_advantages = self._combat_targets(reference, evaluation)
            durable_state_targets, durable_action_targets = self._durable_targets(
                reference, evaluation, trace_lambda=self.config.macro_trace_lambda
            )
            behavior = evaluation.durable_values.new_tensor(
                [step.behavior_log_probability for step in reference.steps]
            )
            macro_rho = torch.exp(
                (evaluation.selected_logs - behavior).clamp(max=math.log(self.config.policy_rho_clip))
            )
            if self.config.enable_persistent_only_durable_value:
                raw_macro_adv = macro_rho * (macro_action_targets - evaluation.values)
            else:
                raw_macro_adv = macro_rho * (durable_action_targets - evaluation.durable_values)
            if self.config.clip_macro_pg_fallback:
                adv_cap = self.config.macro_pg_adv_clip
                raw_macro_adv = adv_cap * torch.tanh(raw_macro_adv / adv_cap)
            durable_macro_advantages = raw_macro_adv.detach()
            afterstate_targets: dict[int, _AfterstateStepTarget] = {}
            for node, q1_after, q2_after in per_ref_nodes[ref_idx]:
                teacher_logs = evaluation.step_log_probs[node.step_index]
                step = reference.steps[node.step_index]
                projection = self._project_afterstate_node(
                    node=node,
                    step=step,
                    q_after=q1_after,
                    teacher_logs=teacher_logs,
                    effective_macro_kl=effective_macro_kl,
                    q_after_twin=q2_after,
                )
                q_mean_after = q1_after if q2_after is None else 0.5 * (q1_after + q2_after)
                q_disagree = None if q2_after is None else (q1_after - q2_after).abs().detach()
                has_chosen_post = node.post_world_for_action(step.action_index) is not None
                if not has_chosen_post or step.terminal:
                    post_target = durable_action_targets.new_zeros(())
                else:
                    eff_r = self._effective_step_reward(reference, node.step_index)
                    post_target = (
                        (durable_action_targets[node.step_index] - eff_r) / step.discount
                    ).detach()
                afterstate_targets[node.step_index] = _AfterstateStepTarget(
                    node=node,
                    action_values=q_mean_after.detach(),
                    projection=projection,
                    post_target=post_target,
                    has_chosen_post=has_chosen_post,
                    action_disagreement=q_disagree,
                )
            prepared_out.append(
                _PreparedReference(
                    None,
                    reference,
                    evaluation,
                    macro_values,
                    combat_values,
                    combat_advantages,
                    None,
                    torch.empty(0),
                    durable_state_targets=durable_state_targets,
                    durable_action_targets=durable_action_targets,
                    durable_macro_advantages=durable_macro_advantages,
                    afterstate_targets=afterstate_targets,
                    step_revival_ctx=None if rev_all is None else rev_all[ref_idx][0],
                    post_revival_ctx=None if rev_all is None else rev_all[ref_idx][1],
                )
            )
        return prepared_out, total_calls, max_error

    @torch.no_grad()
    def _prepare_durable_natural(
        self,
        reference: MacroCreditTrajectory,
        *,
        model_override: HybridDomainActorCriticModel | None = None,
        check_likelihood: bool = True,
        record_all_choice_logs: bool = False,
    ) -> tuple[_PreparedReference, int, float]:
        active_mod = self.model if model_override is None else model_override
        use_rev = self._use_revival_ctx(active_mod)
        step_rev_cpu: Tensor | None = None
        post_rev_cpu: Tensor | None = None
        if use_rev:
            step_rev_cpu, post_rev_cpu, _ = self._trajectory_critic_revival_contexts(reference)
        evaluation = self._evaluate(
            reference,
            root_index=-1,
            model_override=model_override,
            check_likelihood=check_likelihood,
            record_all_choice_logs=record_all_choice_logs,
        )
        assert (
            evaluation.durable_values is not None
            and evaluation.step_log_probs is not None
            and evaluation.post_step_hidden is not None
        )
        macro_values, macro_action_targets = self._targets(
            reference, evaluation, trace_lambda=self.config.macro_trace_lambda
        )
        combat_values, combat_advantages = self._combat_targets(reference, evaluation)
        durable_state_targets, durable_action_targets = self._durable_targets(
            reference, evaluation, trace_lambda=self.config.macro_trace_lambda
        )
        behavior = evaluation.durable_values.new_tensor([step.behavior_log_probability for step in reference.steps])
        macro_rho = torch.exp(
            (evaluation.selected_logs - behavior).clamp(max=math.log(self.config.policy_rho_clip))
        )
        if self.config.enable_persistent_only_durable_value:
            raw_macro_adv = macro_rho * (macro_action_targets - evaluation.values)
        else:
            raw_macro_adv = macro_rho * (durable_action_targets - evaluation.durable_values)
        if self.config.clip_macro_pg_fallback:
            adv_cap = self.config.macro_pg_adv_clip
            raw_macro_adv = adv_cap * torch.tanh(raw_macro_adv / adv_cap)
        durable_macro_advantages = raw_macro_adv.detach()
        effective_macro_kl = (
            self.config.macro_kl if self.update >= self.config.durable_warmup_updates else 0.0
        )
        afterstate_targets: dict[int, _AfterstateStepTarget] = {}
        calls = evaluation.forward_calls
        include_unvisited_partial = bool(self.config.enable_partial_afterstate_conditional_mpo)
        for node in reference.afterstate_nodes:
            step = reference.steps[node.step_index]
            has_chosen_post = node.post_world_for_action(step.action_index) is not None
            if not has_chosen_post and not include_unvisited_partial:
                continue
            saved_hidden = evaluation.post_step_hidden[node.step_index].to(device=self.device, dtype=self.dtype)
            post_rev_gpu = (
                None
                if post_rev_cpu is None
                else post_rev_cpu[node.step_index : node.step_index + 1].to(
                    device=self.device, dtype=self.dtype
                )
            )
            use_twin_single = (
                isinstance(active_mod, HybridDomainActorCriticModel)
                and active_mod.actor_critic_config.enable_twin_durable_value_heads
            )
            if use_twin_single:
                q1_gpu, q2_gpu = self._forward_durable_worlds_pair(
                    node.post_worlds,
                    saved_hidden,
                    critic_revival_context=post_rev_gpu,
                    model_override=model_override,
                )
                q1_after = q1_gpu.detach().cpu()
                q2_after_t = q2_gpu.detach().cpu()
                q2_after_opt: Tensor | None = q2_after_t
                q_mean_after = 0.5 * (q1_after + q2_after_t)
                q_disagree = (q1_after - q2_after_t).abs().detach()
            else:
                q1_after = (
                    self._forward_durable_worlds(
                        node.post_worlds,
                        saved_hidden,
                        critic_revival_context=post_rev_gpu,
                        model_override=model_override,
                    )
                    .detach()
                    .cpu()
                )
                q2_after_opt = None
                q_mean_after = q1_after
                q_disagree = None
            calls += 1
            teacher_logs = evaluation.step_log_probs[node.step_index]
            projection = self._project_afterstate_node(
                node=node,
                step=step,
                q_after=q1_after,
                teacher_logs=teacher_logs,
                effective_macro_kl=effective_macro_kl,
                q_after_twin=q2_after_opt,
            )
            if not has_chosen_post or step.terminal:
                post_target = durable_action_targets.new_zeros(())
            else:
                eff_r = self._effective_step_reward(reference, node.step_index)
                post_target = ((durable_action_targets[node.step_index] - eff_r) / step.discount).detach()
            afterstate_targets[node.step_index] = _AfterstateStepTarget(
                node=node,
                action_values=q_mean_after.detach(),
                projection=projection,
                post_target=post_target,
                has_chosen_post=has_chosen_post,
                action_disagreement=q_disagree,
            )
        return (
            _PreparedReference(
                None,
                reference,
                evaluation,
                macro_values,
                combat_values,
                combat_advantages,
                None,
                torch.empty(0),
                durable_state_targets=durable_state_targets,
                durable_action_targets=durable_action_targets,
                durable_macro_advantages=durable_macro_advantages,
                afterstate_targets=afterstate_targets,
                step_revival_ctx=step_rev_cpu,
                post_revival_ctx=post_rev_cpu,
            ),
            calls,
            evaluation.maximum_likelihood_error,
        )

    def _need_all_choice_logs(self) -> bool:
        return bool(
            self.config.clip_combat_pg
            or self.config.ppo_epochs_per_cohort > 1
            or self.config.ppo_minibatches_per_cohort > 1
            or self.config.target_macro_step_kl > 0.0
        )

    def _prepare_natural(self, reference: MacroCreditTrajectory) -> tuple[_PreparedReference, int, float]:
        if self.config.target_mode == "durable_afterstate_v2":
            return self._prepare_durable_natural(
                reference,
                record_all_choice_logs=self._need_all_choice_logs(),
            )
        # -1 requests recurrent chunk boundaries, but no artificial macro root.
        step_rev_cpu: Tensor | None = None
        post_rev_cpu: Tensor | None = None
        if self._use_revival_ctx():
            step_rev_cpu, post_rev_cpu, _ = self._trajectory_critic_revival_contexts(reference)
        evaluation = self._evaluate(
            reference,
            root_index=-1,
            record_all_choice_logs=self.config.clip_combat_pg,
        )
        macro_values, _ = self._targets(reference, evaluation, trace_lambda=self.config.macro_trace_lambda)
        combat_values, combat_advantages = self._combat_targets(reference, evaluation)
        return (
            _PreparedReference(
                None,
                reference,
                evaluation,
                macro_values,
                combat_values,
                combat_advantages,
                None,
                torch.empty(0),
                step_revival_ctx=step_rev_cpu,
                post_revival_ctx=post_rev_cpu,
            ),
            evaluation.forward_calls,
            evaluation.maximum_likelihood_error,
        )

    def _combat_targets(self, reference: MacroCreditTrajectory, evaluation: _Evaluation) -> tuple[Tensor, Tensor]:
        combat_values = torch.empty_like(evaluation.values)
        combat_advantages = torch.empty_like(evaluation.values)
        for start in range(0, len(reference.steps), self.config.combat_unroll_steps):
            stop = min(start + self.config.combat_unroll_steps, len(reference.steps))
            targets, action_targets = self._targets(reference, evaluation, start=start, stop=stop, trace_lambda=1.0)
            behavior = evaluation.values.new_tensor([step.behavior_log_probability for step in reference.steps[start:stop]])
            policy_rho = torch.exp((evaluation.selected_logs[start:stop] - behavior).clamp(max=math.log(self.config.policy_rho_clip)))
            raw_adv = policy_rho * (action_targets - evaluation.values[start:stop])
            if self.config.clip_combat_pg:
                c_adv_cap = self.config.combat_pg_adv_clip
                raw_adv = c_adv_cap * torch.tanh(raw_adv / c_adv_cap)
            combat_values[start:stop] = targets
            combat_advantages[start:stop] = raw_adv
        return combat_values, combat_advantages

    def _prepare(self, root: MacroCounterfactualRoot) -> tuple[_PreparedReference, int, float]:
        step_rev_cpu: Tensor | None = None
        post_rev_cpu: Tensor | None = None
        if self._use_revival_ctx():
            step_rev_cpu, post_rev_cpu, _ = self._trajectory_critic_revival_contexts(root.reference)
        evaluation = self._evaluate(
            root.reference,
            root_index=root.root_step_index,
            record_all_choice_logs=self.config.clip_combat_pg,
        )
        assert evaluation.root_logs is not None
        recorded = evaluation.root_logs.new_tensor(root.teacher_log_probabilities)
        if not torch.allclose(evaluation.root_logs, recorded, rtol=0, atol=self.config.likelihood_tolerance):
            raise ValueError("full root prior differs from fresh reconstructed teacher")
        calls = evaluation.forward_calls
        maximum_error = max(evaluation.maximum_likelihood_error, float((evaluation.root_logs - recorded).abs().max().item()))
        action_values: list[Tensor] = []
        if self.config.target_mode == "combat_shadow_price":
            for trajectory in root.candidates:
                if len(trajectory.steps) > 1:
                    candidate = self._evaluate(trajectory)
                    _, conditional = self._targets(trajectory, candidate, trace_lambda=self.config.macro_trace_lambda)
                    action_values.append(conditional[0])
                    calls += candidate.forward_calls
                    maximum_error = max(maximum_error, candidate.maximum_likelihood_error)
                else:
                    eff_r0 = self._effective_step_reward(trajectory, 0)
                    if trajectory.steps[0].terminal:
                        action_val = evaluation.values.new_tensor(eff_r0)
                    else:
                        target_snap = trajectory.bootstrap_snapshot or trajectory.steps[0].snapshot
                        combat_val = evaluate_combat_value_from_snapshot(
                            cast(HybridDomainActorCriticModel, self.model), target_snap, device=self.device
                        )
                        combat_tensor = evaluation.values.new_tensor(combat_val)
                        reward_tensor = combat_tensor.new_tensor(eff_r0)
                        action_val = reward_tensor + self.config.macro_trace_lambda * trajectory.steps[0].discount * combat_tensor
                    action_values.append(action_val.squeeze())
                    calls += 1
        else:
            for trajectory in root.candidates:
                candidate = self._evaluate(trajectory)
                _, conditional = self._targets(trajectory, candidate, trace_lambda=self.config.macro_trace_lambda)
                action_values.append(conditional[0])
                calls += candidate.forward_calls
                maximum_error = max(maximum_error, candidate.maximum_likelihood_error)
        candidate_values = torch.stack(action_values)
        projection = project_macro_policy(
            action_value_targets=candidate_values, teacher_log_probs=evaluation.root_logs, max_kl=self.config.macro_kl,
        )
        macro_values, _ = self._targets(root.reference, evaluation, trace_lambda=self.config.macro_trace_lambda)
        macro_values[root.root_step_index] = projection.value_target
        combat_values, combat_advantages = self._combat_targets(root.reference, evaluation)
        return (
            _PreparedReference(
                root,
                root.reference,
                evaluation,
                macro_values,
                combat_values,
                combat_advantages,
                projection,
                candidate_values,
                step_revival_ctx=step_rev_cpu,
                post_revival_ctx=post_rev_cpu,
            ),
            calls,
            maximum_error,
        )

    def _backward_chunk(
        self,
        prepared: _PreparedReference,
        start: int,
        stop: int,
        denominators: Mapping[str, int],
        *,
        teacher_step_logs: Mapping[int, Tensor] | None = None,
        kickstart_kl_weight: float = 0.0,
    ) -> dict[str, float]:
        self._check_resource()
        saved = prepared.evaluation.boundary_hidden[start]
        hidden = None if saved is None else saved.to(device=self.device, dtype=self.dtype)
        totals = {
            key: 0.0
            for key in (
                "macro_policy_loss",
                "macro_mpo_policy_loss",
                "macro_mpo_within_policy_loss",
                "macro_pg_policy_loss",
                "macro_entropy",
                "macro_value_loss",
                "durable_value_loss",
                "durable_post_value_loss",
                "durable_stage_baseline_loss",
                "combat_policy_loss",
                "combat_value_loss",
                "combat_entropy",
                "distill_macro_kl",
                "distill_combat_kl",
                "distill_afterstate_margin_loss",
                "distill_afterstate_value_loss",
                "observed_macro_step_kl",
                "observed_combat_step_kl",
                "backward_chunks",
            )
        }
        terms: list[Tensor] = []
        cfg = self.config
        is_durable_v2 = cfg.target_mode == "durable_afterstate_v2"
        has_stage_base = (
            is_durable_v2
            and isinstance(self.model, HybridDomainActorCriticModel)
            and self.model.actor_critic_config.enable_monotone_deconfounded_durable_value
        )
        has_twin_dur = (
            is_durable_v2
            and isinstance(self.model, HybridDomainActorCriticModel)
            and self.model.actor_critic_config.enable_twin_durable_value_heads
        )
        durable_actor_active = (not is_durable_v2) or (self.update >= cfg.durable_warmup_updates)
        combat_actor_active = (not is_durable_v2) or (self.update >= cfg.combat_actor_warmup_updates)
        for index in range(start, stop):
            step = prepared.reference.steps[index]
            prev_hidden = hidden
            step_rev_gpu = (
                None
                if prepared.step_revival_ctx is None
                else prepared.step_revival_ctx[index : index + 1].to(device=self.device, dtype=self.dtype)
            )
            logits, value, durable_value, hidden = self._forward_full(
                step.snapshot,
                hidden,
                critic_revival_context=step_rev_gpu,
            )
            logs, entropy, choice = self._policy(step, logits)
            if is_durable_v2:
                is_combat_entry = step.control_domain == "combat" and (
                    index == 0 or prepared.reference.steps[index - 1].control_domain == "macro"
                )
                if step.control_domain == "macro" or is_combat_entry:
                    assert durable_value is not None and prepared.durable_state_targets is not None
                    durable_target = prepared.durable_state_targets[index].to(
                        device=self.device, dtype=durable_value.dtype
                    )
                    if has_twin_dur:
                        dv1_chunk, dv2_chunk = self._forward_durable_worlds_pair(
                            (cast(EncodedWorldSnapshot, step.snapshot),),
                            prev_hidden,
                            critic_revival_context=step_rev_gpu,
                        )
                        durable_loss = (
                            0.5
                            * (
                                F.smooth_l1_loss(dv1_chunk[0], durable_target)
                                + F.smooth_l1_loss(dv2_chunk[0], durable_target)
                            )
                            / max(1, denominators["durable_value_steps"])
                        )
                    else:
                        durable_loss = F.smooth_l1_loss(durable_value, durable_target) / max(
                            1, denominators["durable_value_steps"]
                        )
                    totals["durable_value_loss"] += float(durable_loss.detach().item())
                    if cfg.macro_value_weight:
                        terms.append(0.5 * cfg.macro_value_weight * durable_loss)
                    if has_stage_base:
                        step_w_batch, _ = collate_encoded_worlds(
                            (cast(EncodedWorldSnapshot, step.snapshot),),
                            expected_config=step.snapshot.config,
                            expected_fingerprint=step.snapshot.encoding_fingerprint,
                            device=self.device,
                            validate=False,
                        )
                        if cfg.durable_stage_baseline_weight:
                            stage_val = cast(
                                HybridDomainActorCriticModel, self.model
                            ).forward_durable_stage_baseline(
                                step_w_batch,
                                critic_revival_context=step_rev_gpu,
                            )[0]
                        else:
                            with torch.no_grad():
                                stage_val = cast(
                                    HybridDomainActorCriticModel, self.model
                                ).forward_durable_stage_baseline(
                                    step_w_batch,
                                    critic_revival_context=step_rev_gpu,
                                )[0]
                        stage_loss = F.smooth_l1_loss(stage_val, durable_target) / max(
                            1, denominators["durable_value_steps"]
                        )
                        totals["durable_stage_baseline_loss"] += float(stage_loss.detach().item())
                        if cfg.durable_stage_baseline_weight:
                            terms.append(0.5 * cfg.durable_stage_baseline_weight * stage_loss)
            if step.control_domain == "macro":
                target = prepared.macro_values[index].to(device=self.device, dtype=value.dtype)
                value_loss = F.smooth_l1_loss(value, target) / denominators["macro_steps"]
                totals["macro_value_loss"] += float(value_loss.detach().item())
                if cfg.macro_value_weight:
                    terms.append(0.5 * cfg.macro_value_weight * value_loss)
                if is_durable_v2:
                    assert prepared.afterstate_targets is not None
                    as_target = prepared.afterstate_targets.get(index)
                    if as_target is not None and as_target.has_chosen_post:
                        chosen_world = as_target.node.post_world_for_action(step.action_index)
                        assert chosen_world is not None
                        post_rev_gpu = (
                            None
                            if prepared.post_revival_ctx is None
                            else prepared.post_revival_ctx[index : index + 1].to(
                                device=self.device, dtype=self.dtype
                            )
                        )
                        if has_twin_dur:
                            cp1_val, cp2_val = self._forward_durable_worlds_pair(
                                (chosen_world,),
                                hidden,
                                critic_revival_context=post_rev_gpu,
                            )
                            post_target = as_target.post_target.to(
                                device=self.device, dtype=cp1_val.dtype
                            )
                            post_loss = (
                                0.5
                                * (
                                    F.smooth_l1_loss(cp1_val[0], post_target)
                                    + F.smooth_l1_loss(cp2_val[0], post_target)
                                )
                                / max(1, denominators["afterstate_steps"])
                            )
                        else:
                            chosen_post_val = self._forward_durable_worlds(
                                (chosen_world,),
                                hidden,
                                critic_revival_context=post_rev_gpu,
                            )[0]
                            post_target = as_target.post_target.to(
                                device=self.device, dtype=chosen_post_val.dtype
                            )
                            post_loss = F.smooth_l1_loss(chosen_post_val, post_target) / max(
                                1, denominators["afterstate_steps"]
                            )
                        totals["durable_post_value_loss"] += float(post_loss.detach().item())
                        if cfg.durable_post_value_weight:
                            terms.append(0.5 * cfg.durable_post_value_weight * post_loss)
                        if has_stage_base:
                            post_w_batch, _ = collate_encoded_worlds(
                                (chosen_world,),
                                expected_config=chosen_world.config,
                                expected_fingerprint=chosen_world.encoding_fingerprint,
                                device=self.device,
                                validate=False,
                            )
                            if cfg.durable_stage_baseline_weight:
                                post_stage_val = cast(
                                    HybridDomainActorCriticModel, self.model
                                ).forward_durable_stage_baseline(
                                    post_w_batch,
                                    critic_revival_context=post_rev_gpu,
                                )[0]
                            else:
                                with torch.no_grad():
                                    post_stage_val = cast(
                                        HybridDomainActorCriticModel, self.model
                                    ).forward_durable_stage_baseline(
                                        post_w_batch,
                                        critic_revival_context=post_rev_gpu,
                                    )[0]
                            post_stage_loss = F.smooth_l1_loss(post_stage_val, post_target) / max(
                                1, denominators["afterstate_steps"]
                            )
                            totals["durable_stage_baseline_loss"] += float(
                                post_stage_loss.detach().item()
                            )
                            if cfg.durable_stage_baseline_weight:
                                terms.append(
                                    0.5 * cfg.durable_stage_baseline_weight * post_stage_loss
                                )
                        if durable_actor_active and as_target.projection.actor_enabled:
                            denom_m_pol = max(1, denominators["macro_policy_steps"])
                            mpo_step_terms: list[Tensor] = []
                            if not torch.equal(
                                as_target.projection.target_probabilities,
                                as_target.projection.teacher_probabilities,
                            ):
                                target_policy = as_target.projection.target_probabilities.to(
                                    device=self.device, dtype=logs.dtype
                                )
                                safe_step_logs = logs.masked_fill(~torch.isfinite(logs), 0.0)
                                joint_ce = -(target_policy * safe_step_logs[: target_policy.numel()]).sum() / denom_m_pol
                                mpo_step_terms.append(joint_ce)
                            if (
                                cfg.macro_within_branch_mpo_weight > 0.0
                                and as_target.projection.within_branch_targets
                            ):
                                within_terms: list[Tensor] = []
                                for cand_idx_tuple, w_star_cpu in as_target.projection.within_branch_targets:
                                    idx_gpu = torch.tensor(cand_idx_tuple, device=self.device, dtype=torch.long)
                                    cond_logs = torch.log_softmax(logs[idx_gpu], dim=0)
                                    safe_cond_logs = cond_logs.masked_fill(~torch.isfinite(cond_logs), 0.0)
                                    w_star_gpu = w_star_cpu.to(device=self.device, dtype=logs.dtype)
                                    within_terms.append(-(w_star_gpu * safe_cond_logs).sum())
                                within_ce = (
                                    cfg.macro_within_branch_mpo_weight
                                    * torch.stack(within_terms).mean()
                                    / denom_m_pol
                                )
                                totals["macro_mpo_within_policy_loss"] += float(within_ce.detach().item())
                                mpo_step_terms.append(within_ce)
                            if mpo_step_terms:
                                actor = torch.stack(mpo_step_terms).sum()
                                totals["macro_policy_loss"] += float(actor.detach().item())
                                totals["macro_mpo_policy_loss"] += float(actor.detach().item())
                                if cfg.macro_policy_weight:
                                    terms.append(0.5 * cfg.macro_policy_weight * actor)
                    else:
                        if as_target is not None and durable_actor_active and as_target.projection.actor_enabled:
                            denom_m_pol = max(1, denominators["macro_policy_steps"])
                            cond_mpo_terms: list[Tensor] = []
                            if not torch.equal(
                                as_target.projection.target_probabilities,
                                as_target.projection.teacher_probabilities,
                            ):
                                det_idx_gpu = torch.tensor(
                                    as_target.node.resolved_candidate_indices,
                                    device=self.device,
                                    dtype=torch.long,
                                )
                                target_det = as_target.projection.target_probabilities.to(
                                    device=self.device, dtype=logs.dtype
                                )[det_idx_gpu]
                                cond_det_logs = torch.log_softmax(logs[det_idx_gpu], dim=0)
                                safe_cond_det_logs = cond_det_logs.masked_fill(
                                    ~torch.isfinite(cond_det_logs), 0.0
                                )
                                cond_joint_ce = -(target_det * safe_cond_det_logs).sum() / denom_m_pol
                                cond_mpo_terms.append(cond_joint_ce)
                            if (
                                cfg.macro_within_branch_mpo_weight > 0.0
                                and as_target.projection.within_branch_targets
                            ):
                                within_terms = []
                                for cand_idx_tuple, w_star_cpu in as_target.projection.within_branch_targets:
                                    idx_gpu = torch.tensor(cand_idx_tuple, device=self.device, dtype=torch.long)
                                    cond_logs = torch.log_softmax(logs[idx_gpu], dim=0)
                                    safe_cond_logs = cond_logs.masked_fill(~torch.isfinite(cond_logs), 0.0)
                                    w_star_gpu = w_star_cpu.to(device=self.device, dtype=logs.dtype)
                                    within_terms.append(-(w_star_gpu * safe_cond_logs).sum())
                                within_ce = (
                                    cfg.macro_within_branch_mpo_weight
                                    * torch.stack(within_terms).mean()
                                    / denom_m_pol
                                )
                                totals["macro_mpo_within_policy_loss"] += float(within_ce.detach().item())
                                cond_mpo_terms.append(within_ce)
                            if cond_mpo_terms:
                                cond_actor = torch.stack(cond_mpo_terms).sum()
                                totals["macro_policy_loss"] += float(cond_actor.detach().item())
                                totals["macro_mpo_policy_loss"] += float(cond_actor.detach().item())
                                if cfg.macro_policy_weight:
                                    terms.append(0.5 * cfg.macro_policy_weight * cond_actor)
                        if choice and durable_actor_active:
                            assert prepared.durable_macro_advantages is not None
                            macro_adv = prepared.durable_macro_advantages[index].to(
                                device=self.device, dtype=logs.dtype
                            )
                            if cfg.clip_macro_pg_fallback:
                                assert prepared.evaluation.step_log_probs is not None
                                t_logs = prepared.evaluation.step_log_probs[index].to(
                                    device=self.device, dtype=logs.dtype
                                )
                                valid_t = torch.isfinite(t_logs)
                                safe_t_logs = t_logs.masked_fill(~valid_t, 0.0)
                                safe_cur_logs = logs[: t_logs.numel()].masked_fill(~valid_t, 0.0)
                                t_probs = t_logs.exp().masked_fill(~valid_t, 0.0)
                                ratio = torch.exp(
                                    (logs[step.action_index] - t_logs[step.action_index]).clamp(-10.0, 10.0)
                                )
                                clipped_ratio = ratio.clamp(
                                    1.0 - cfg.macro_pg_clip_eps, 1.0 + cfg.macro_pg_clip_eps
                                )
                                surr = torch.minimum(ratio * macro_adv, clipped_ratio * macro_adv)
                                kl_pen = (t_probs * (safe_t_logs - safe_cur_logs)).sum()
                                actor = (-surr + cfg.macro_pg_kl_weight * kl_pen) / max(
                                    1, denominators["macro_policy_steps"]
                                )
                            else:
                                actor = -logs[step.action_index] * macro_adv / max(
                                    1, denominators["macro_policy_steps"]
                                )
                            if cfg.enable_macro_branch_prior:
                                step_mask = torch.tensor(
                                    step.snapshot.action_mask,
                                    device=logs.device,
                                    dtype=torch.bool,
                                )
                                step_branches = torch.tensor(
                                    step.snapshot.candidates.ids[: step_mask.numel(), 13],
                                    device=logs.device,
                                    dtype=torch.long,
                                )
                                macro_step_entropy = hierarchical_macro_entropy(
                                    logs[: step_mask.numel()].masked_fill(~step_mask, 0.0).unsqueeze(0),
                                    step_mask.unsqueeze(0),
                                    step_branches.unsqueeze(0),
                                    include_branch_entropy=not cfg.macro_pg_within_branch_entropy_only,
                                )[0]
                            else:
                                macro_step_entropy = entropy
                            norm_macro_entropy = macro_step_entropy / max(1, denominators["macro_policy_steps"])
                            totals["macro_policy_loss"] += float(actor.detach().item())
                            totals["macro_pg_policy_loss"] += float(actor.detach().item())
                            totals["macro_entropy"] += float(norm_macro_entropy.detach().item())
                            if cfg.macro_policy_weight:
                                terms.append(0.5 * cfg.macro_policy_weight * actor)
                            if cfg.macro_entropy_weight:
                                terms.append(-0.5 * cfg.macro_entropy_weight * norm_macro_entropy)
                elif (prepared.root is not None and prepared.projection is not None
                        and index == prepared.root.root_step_index and prepared.projection.actor_enabled):
                    target_policy = prepared.projection.target_probabilities.to(device=self.device, dtype=logs.dtype)
                    actor = -(target_policy * logs).sum() / denominators["root_count"]
                    totals["macro_policy_loss"] += float(actor.detach().item())
                    if cfg.macro_policy_weight:
                        terms.append(0.5 * cfg.macro_policy_weight * actor)
            else:
                target = prepared.combat_values[index].to(device=self.device, dtype=value.dtype)
                value_loss = F.smooth_l1_loss(value, target) / denominators["combat_steps"]
                totals["combat_value_loss"] += float(value_loss.detach().item())
                if cfg.combat_value_weight:
                    terms.append(0.5 * cfg.combat_value_weight * value_loss)
                if choice:
                    advantage = prepared.combat_advantages[index].to(device=self.device, dtype=logs.dtype)
                    denom_c_pol = max(1, denominators["combat_policy_steps"])
                    if cfg.clip_combat_pg:
                        assert prepared.evaluation.step_log_probs is not None
                        c_t_logs = prepared.evaluation.step_log_probs[index].to(
                            device=self.device, dtype=logs.dtype
                        )
                        c_valid_t = torch.isfinite(c_t_logs)
                        c_safe_t_logs = c_t_logs.masked_fill(~c_valid_t, 0.0)
                        c_safe_cur_logs = logs[: c_t_logs.numel()].masked_fill(~c_valid_t, 0.0)
                        c_t_probs = c_t_logs.exp().masked_fill(~c_valid_t, 0.0)
                        c_ratio = torch.exp(
                            (logs[step.action_index] - c_t_logs[step.action_index]).clamp(-10.0, 10.0)
                        )
                        c_clipped = c_ratio.clamp(
                            1.0 - cfg.combat_pg_clip_eps, 1.0 + cfg.combat_pg_clip_eps
                        )
                        c_surr = torch.minimum(c_ratio * advantage, c_clipped * advantage)
                        c_kl_pen = (c_t_probs * (c_safe_t_logs - c_safe_cur_logs)).sum()
                        actor = (-c_surr + cfg.combat_pg_kl_weight * c_kl_pen) / denom_c_pol
                    else:
                        actor = -logs[step.action_index] * advantage / denom_c_pol
                    normalized_entropy = entropy / denom_c_pol
                    totals["combat_policy_loss"] += float(actor.detach().item())
                    totals["combat_entropy"] += float(normalized_entropy.detach().item())
                    if combat_actor_active:
                        if cfg.combat_policy_weight:
                            terms.append(0.5 * cfg.combat_policy_weight * actor)
                        if cfg.combat_entropy_weight:
                            terms.append(-0.5 * cfg.combat_entropy_weight * normalized_entropy)
            if choice and kickstart_kl_weight > 0.0 and teacher_step_logs is not None:
                ks_t_cpu = teacher_step_logs.get(index)
                if ks_t_cpu is not None:
                    ks_t_gpu = ks_t_cpu.to(device=self.device, dtype=logs.dtype)
                    ks_valid = torch.isfinite(ks_t_gpu)
                    ks_safe_t = ks_t_gpu.masked_fill(~ks_valid, 0.0)
                    ks_safe_s = logs[: ks_t_gpu.numel()].masked_fill(~ks_valid, 0.0)
                    ks_probs_t = ks_t_gpu.exp().masked_fill(~ks_valid, 0.0)
                    ks_kl = (ks_probs_t * (ks_safe_t - ks_safe_s)).sum()
                    if step.control_domain == "macro":
                        norm_ks = ks_kl / max(1, denominators["macro_policy_steps"])
                        totals["distill_macro_kl"] += float(norm_ks.detach().clamp_min(0.0).item())
                    else:
                        norm_ks = ks_kl / max(1, denominators["combat_policy_steps"])
                        totals["distill_combat_kl"] += float(norm_ks.detach().clamp_min(0.0).item())
                    terms.append(0.5 * kickstart_kl_weight * norm_ks)
            if choice and prepared.evaluation.step_log_probs is not None:
                old_t_cpu = prepared.evaluation.step_log_probs.get(index)
                if old_t_cpu is not None:
                    old_t_gpu = old_t_cpu.to(device=self.device, dtype=logs.dtype)
                    old_valid = torch.isfinite(old_t_gpu)
                    old_safe_t = old_t_gpu.masked_fill(~old_valid, 0.0)
                    old_safe_s = logs[: old_t_gpu.numel()].masked_fill(~old_valid, 0.0)
                    old_probs_t = old_t_gpu.exp().masked_fill(~old_valid, 0.0)
                    step_kl = (old_probs_t * (old_safe_t - old_safe_s)).sum()
                    if step.control_domain == "macro":
                        denom_m_kl = max(1, denominators["macro_policy_steps"])
                        norm_m_kl = step_kl / denom_m_kl
                        totals["observed_macro_step_kl"] += float(norm_m_kl.detach().clamp_min(0.0).item())
                        has_as_here = (
                            is_durable_v2
                            and prepared.afterstate_targets is not None
                            and prepared.afterstate_targets.get(index) is not None
                            and prepared.afterstate_targets[index].has_chosen_post
                        )
                        if (
                            has_as_here
                            and durable_actor_active
                            and cfg.macro_mpo_trust_kl_weight > 0.0
                            and cfg.macro_policy_weight > 0.0
                        ):
                            trust_term = cfg.macro_mpo_trust_kl_weight * norm_m_kl
                            totals["macro_policy_loss"] += float(trust_term.detach().item())
                            totals["macro_mpo_policy_loss"] += float(trust_term.detach().item())
                            terms.append(0.5 * cfg.macro_policy_weight * trust_term)
                    else:
                        norm_c_kl = step_kl / max(1, denominators["combat_policy_steps"])
                        totals["observed_combat_step_kl"] += float(norm_c_kl.detach().clamp_min(0.0).item())
        if terms:
            loss = torch.stack(terms).sum()
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError("macro cohort chunk loss is non-finite")
            loss.backward()  # type: ignore[no-untyped-call]
            totals["backward_chunks"] = 1.0
        self._check_resource()
        return totals

    def _select_recurrent_chunk_slice_end(
        self,
        ac_model: HybridDomainActorCriticModel,
        keyed_chunks: Sequence[tuple[int, int, int, _PreparedReference, int, int]],
        slice_start: int,
    ) -> int:
        total_chunks = len(keyed_chunks)
        cfg = self.config
        d_model = max(128, int(ac_model.config.d_model))
        width_scale = min(1.0, (128.0 / float(d_model)) ** 1.25)
        high_vram = self._has_high_vram()
        world_step_budget = (
            49_152
            if high_vram
            else (12_288 if d_model >= 256 else max(16_384, int(49_152 * width_scale)))
        )
        local_step_budget = (
            98_304
            if high_vram
            else (24_576 if d_model >= 256 else max(24_576, int(98_304 * width_scale)))
        )
        max_attn_floats = 2_400_000_000 if high_vram else 1_200_000_000
        n_heads = max(1, int(ac_model.config.n_heads))
        world_layers = max(1, int(ac_model.config.world_layers))
        limit = min(total_chunks, slice_start + cfg.chunk_batch_size)

        cur_cl = 1
        cur_w = 1
        cur_len = 1
        slice_end = slice_start + 1
        for cand_idx in range(slice_start, limit):
            row_cl, row_w, _, _, r_start, r_stop = keyed_chunks[cand_idx]
            next_cl = max(cur_cl, max(1, int(row_cl)))
            next_w = max(cur_w, max(1, int(row_w)))
            next_len = max(cur_len, max(1, int(r_stop - r_start)))
            attn_floats_per_chunk = next_len * world_layers * n_heads * ((next_w + 1) ** 2)
            allowed_b = max(
                1,
                min(
                    cfg.chunk_batch_size,
                    local_step_budget // next_cl,
                    world_step_budget // next_w,
                    max_attn_floats // max(1, attn_floats_per_chunk),
                ),
            )
            if (cand_idx - slice_start + 1) > allowed_b:
                break
            cur_cl = next_cl
            cur_w = next_w
            cur_len = next_len
            slice_end = cand_idx + 1
        return slice_end

    def _backward_stateless_pre_durable(
        self,
        *,
        dur_entries: Sequence[tuple[EncodedWorldSnapshot, int]],
        flat_dur_target: Tensor,
        flat_stage_target: Tensor,
        flat_step_rev_gpu: Tensor | None,
        denom_dur_val: int,
        durable_value_weight: float,
        stage_baseline_weight: float,
        dur_ref_ids: Sequence[int] | None = None,
        twin_ref_weights: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[float, float]:
        if not dur_entries:
            return 0.0, 0.0
        ac_model = cast(HybridDomainActorCriticModel, self.model)
        has_stage_base = bool(ac_model.actor_critic_config.enable_monotone_deconfounded_durable_value)
        has_twin = bool(ac_model.actor_critic_config.enable_twin_durable_value_heads)
        indexed_entries = [
            (entry[0], entry[1], (0 if dur_ref_ids is None else int(dur_ref_ids[pos])))
            for pos, entry in enumerate(dur_entries)
        ]
        sorted_entries = sorted(indexed_entries, key=lambda row: row[0].world.token_count)
        high_vram = self._has_high_vram()
        max_sub = 96 if high_vram else (32 if int(ac_model.config.d_model) >= 256 else 64)
        tok_cap = 65_536 if high_vram else 24_576
        dur_total = torch.zeros((), device=self.device, dtype=torch.float64)
        stage_total = torch.zeros((), device=self.device, dtype=torch.float64)

        if has_twin and durable_value_weight:
            ac_model._twin_dropout_active = True
            ac_model.durable_twin_head1_dropout.train(True)
            ac_model.durable_twin_dropout.train(True)
        try:
            idx = 0
            n_total = len(sorted_entries)
            while idx < n_total:
                self._check_resource()
                tentative_end = min(n_total, idx + max_sub)
                max_w = max(1, sorted_entries[tentative_end - 1][0].world.token_count)
                step_cap = max(1, min(max_sub, tok_cap // max_w))
                end = min(n_total, idx + step_cap)
                sub_entries = sorted_entries[idx:end]
                idx = end

                sub_worlds = tuple(row[0] for row in sub_entries)
                wb, dom = collate_encoded_worlds(
                    sub_worlds,
                    expected_config=sub_worlds[0].config,
                    expected_fingerprint=sub_worlds[0].encoding_fingerprint,
                    device=self.device,
                    validate=False,
                )
                sub_gids = torch.tensor([row[1] for row in sub_entries], device=self.device, dtype=torch.long)
                sub_rev = None if flat_step_rev_gpu is None else flat_step_rev_gpu[sub_gids]
                sub_terms: list[Tensor] = []

                if has_twin:
                    with self._autocast_ctx():
                        if durable_value_weight:
                            v1, v2, _ = ac_model.forward_durable_value_pair(
                                wb,
                                dom,
                                None,
                                critic_revival_context=sub_rev,
                                validate=False,
                            )
                        else:
                            with torch.no_grad():
                                v1, v2, _ = ac_model.forward_durable_value_pair(
                                    wb,
                                    dom,
                                    None,
                                    critic_revival_context=sub_rev,
                                    validate=False,
                                )
                    if not bool((torch.isfinite(v1).all() & torch.isfinite(v2).all()).item()):
                        raise FloatingPointError("stateless durable_value_pair forward is non-finite")
                    t_sub = flat_dur_target[sub_gids]
                    l1 = F.smooth_l1_loss(v1, t_sub, reduction="none")
                    l2 = F.smooth_l1_loss(v2, t_sub, reduction="none")
                    if twin_ref_weights is not None and dur_ref_ids is not None:
                        sub_rids = torch.tensor(
                            [row[2] for row in sub_entries], device=self.device, dtype=torch.long
                        )
                        w1 = twin_ref_weights[0][sub_rids]
                        w2 = twin_ref_weights[1][sub_rids]
                        dur_loss = 0.5 * ((w1 * l1).sum() + (w2 * l2).sum()) / denom_dur_val
                    else:
                        dur_loss = 0.5 * (l1.sum() + l2.sum()) / denom_dur_val
                else:
                    with self._autocast_ctx():
                        if durable_value_weight:
                            dur_vals, _ = ac_model.forward_durable_value(
                                wb,
                                dom,
                                None,
                                critic_revival_context=sub_rev,
                                validate=False,
                            )
                        else:
                            with torch.no_grad():
                                dur_vals, _ = ac_model.forward_durable_value(
                                    wb,
                                    dom,
                                    None,
                                    critic_revival_context=sub_rev,
                                    validate=False,
                                )
                    if not bool(torch.isfinite(dur_vals).all().item()):
                        raise FloatingPointError("stateless durable_value forward is non-finite")
                    dur_loss = (
                        F.smooth_l1_loss(dur_vals, flat_dur_target[sub_gids], reduction="sum")
                        / denom_dur_val
                    )
                dur_total = dur_total + dur_loss.detach().double()
                if durable_value_weight:
                    sub_terms.append(0.5 * durable_value_weight * dur_loss)

                if has_stage_base:
                    if stage_baseline_weight:
                        stage_vals = ac_model.forward_durable_stage_baseline(
                            wb,
                            critic_revival_context=sub_rev,
                        )
                    else:
                        with torch.no_grad():
                            stage_vals = ac_model.forward_durable_stage_baseline(
                                wb,
                                critic_revival_context=sub_rev,
                            )
                    if not bool(torch.isfinite(stage_vals).all().item()):
                        raise FloatingPointError("stateless durable_stage_baseline forward is non-finite")
                    stage_loss = (
                        F.smooth_l1_loss(stage_vals, flat_stage_target[sub_gids], reduction="sum") / denom_dur_val
                    )
                    stage_total = stage_total + stage_loss.detach().double()
                    if stage_baseline_weight:
                        sub_terms.append(0.5 * stage_baseline_weight * stage_loss)

                if sub_terms:
                    sub_mega = torch.stack(sub_terms).sum()
                    sub_terms.clear()
                    if not bool(torch.isfinite(sub_mega).item()):
                        raise FloatingPointError("stateless durable loss is non-finite")
                    sub_mega.backward()  # type: ignore[no-untyped-call]
                    del sub_mega
        finally:
            if has_twin:
                ac_model._twin_dropout_active = False
                ac_model.durable_twin_head1_dropout.eval()
                ac_model.durable_twin_dropout.eval()
        return float(dur_total.item()), float(stage_total.item())

    def _backward_stateless_chosen_post_durable(
        self,
        *,
        post_entries: Sequence[tuple[EncodedWorldSnapshot, Tensor, Tensor | None]],
        denom_afterstate: int,
        post_ref_ids: Sequence[int] | None = None,
        twin_ref_weights: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[float, float]:
        if not post_entries:
            return 0.0, 0.0
        ac_model = cast(HybridDomainActorCriticModel, self.model)
        cfg = self.config
        has_stage_base = bool(ac_model.actor_critic_config.enable_monotone_deconfounded_durable_value)
        has_twin = bool(ac_model.actor_critic_config.enable_twin_durable_value_heads)
        indexed_entries = [
            (row[0], row[1], row[2], (0 if post_ref_ids is None else int(post_ref_ids[pos])))
            for pos, row in enumerate(post_entries)
        ]
        sorted_entries = sorted(indexed_entries, key=lambda row: row[0].world.token_count)
        high_vram = self._has_high_vram()
        max_sub = 96 if high_vram else (32 if int(ac_model.config.d_model) >= 256 else 64)
        tok_cap = 65_536 if high_vram else 24_576
        post_total = torch.zeros((), device=self.device, dtype=torch.float64)
        stage_total = torch.zeros((), device=self.device, dtype=torch.float64)

        if has_twin and cfg.durable_post_value_weight:
            ac_model._twin_dropout_active = True
            ac_model.durable_twin_head1_dropout.train(True)
            ac_model.durable_twin_dropout.train(True)
        try:
            idx = 0
            n_total = len(sorted_entries)
            while idx < n_total:
                self._check_resource()
                tentative_end = min(n_total, idx + max_sub)
                max_w = max(1, sorted_entries[tentative_end - 1][0].world.token_count)
                step_cap = max(1, min(max_sub, tok_cap // max_w))
                end = min(n_total, idx + step_cap)
                sub_entries = sorted_entries[idx:end]
                idx = end

                sub_worlds = tuple(row[0] for row in sub_entries)
                wb, dom = collate_encoded_worlds(
                    sub_worlds,
                    expected_config=sub_worlds[0].config,
                    expected_fingerprint=sub_worlds[0].encoding_fingerprint,
                    device=self.device,
                    validate=False,
                )
                stacked_targets = torch.stack([row[1] for row in sub_entries], dim=0)
                stacked_revs = (
                    torch.stack([cast(Tensor, row[2]) for row in sub_entries], dim=0)
                    if sub_entries[0][2] is not None
                    else None
                )
                sub_terms: list[Tensor] = []

                if has_twin:
                    with self._autocast_ctx():
                        if cfg.durable_post_value_weight:
                            pv1, pv2, _ = ac_model.forward_durable_value_pair(
                                wb,
                                dom,
                                None,
                                critic_revival_context=stacked_revs,
                                validate=False,
                            )
                        else:
                            with torch.no_grad():
                                pv1, pv2, _ = ac_model.forward_durable_value_pair(
                                    wb,
                                    dom,
                                    None,
                                    critic_revival_context=stacked_revs,
                                    validate=False,
                                )
                    if not bool((torch.isfinite(pv1).all() & torch.isfinite(pv2).all()).item()):
                        raise FloatingPointError("stateless post durable_value_pair forward is non-finite")
                    pl1 = F.smooth_l1_loss(pv1, stacked_targets, reduction="none")
                    pl2 = F.smooth_l1_loss(pv2, stacked_targets, reduction="none")
                    if twin_ref_weights is not None and post_ref_ids is not None:
                        sub_rids = torch.tensor(
                            [row[3] for row in sub_entries], device=self.device, dtype=torch.long
                        )
                        w1 = twin_ref_weights[0][sub_rids]
                        w2 = twin_ref_weights[1][sub_rids]
                        post_loss = 0.5 * ((w1 * pl1).sum() + (w2 * pl2).sum()) / denom_afterstate
                    else:
                        post_loss = 0.5 * (pl1.sum() + pl2.sum()) / denom_afterstate
                else:
                    with self._autocast_ctx():
                        if cfg.durable_post_value_weight:
                            post_vals, _ = ac_model.forward_durable_value(
                                wb,
                                dom,
                                None,
                                critic_revival_context=stacked_revs,
                                validate=False,
                            )
                        else:
                            with torch.no_grad():
                                post_vals, _ = ac_model.forward_durable_value(
                                    wb,
                                    dom,
                                    None,
                                    critic_revival_context=stacked_revs,
                                    validate=False,
                                )
                    if not bool(torch.isfinite(post_vals).all().item()):
                        raise FloatingPointError("stateless post durable_value forward is non-finite")
                    post_loss = F.smooth_l1_loss(post_vals, stacked_targets, reduction="sum") / denom_afterstate
                post_total = post_total + post_loss.detach().double()
                if cfg.durable_post_value_weight:
                    sub_terms.append(0.5 * cfg.durable_post_value_weight * post_loss)

                if has_stage_base:
                    if cfg.durable_stage_baseline_weight:
                        post_stage_vals = ac_model.forward_durable_stage_baseline(
                            wb,
                            critic_revival_context=stacked_revs,
                        )
                    else:
                        with torch.no_grad():
                            post_stage_vals = ac_model.forward_durable_stage_baseline(
                                wb,
                                critic_revival_context=stacked_revs,
                            )
                    if not bool(torch.isfinite(post_stage_vals).all().item()):
                        raise FloatingPointError("stateless post durable_stage_baseline forward is non-finite")
                    post_stage_loss = (
                        F.smooth_l1_loss(post_stage_vals, stacked_targets, reduction="sum") / denom_afterstate
                    )
                    stage_total = stage_total + post_stage_loss.detach().double()
                    if cfg.durable_stage_baseline_weight:
                        sub_terms.append(0.5 * cfg.durable_stage_baseline_weight * post_stage_loss)

                if sub_terms:
                    sub_mega = torch.stack(sub_terms).sum()
                    sub_terms.clear()
                    if not bool(torch.isfinite(sub_mega).item()):
                        raise FloatingPointError("stateless post durable loss is non-finite")
                    sub_mega.backward()  # type: ignore[no-untyped-call]
                    del sub_mega
        finally:
            if has_twin:
                ac_model._twin_dropout_active = False
                ac_model.durable_twin_head1_dropout.eval()
                ac_model.durable_twin_dropout.eval()
        return float(post_total.item()), float(stage_total.item())

    def _backward_stateless_distill_afterstates(
        self,
        *,
        node_entries: Sequence[
            tuple[
                int,
                _AfterstateStepTarget,
                int | None,
                EncodedWorldSnapshot | None,
                Tensor | None,
            ]
        ],
        denom_afterstate: int,
    ) -> tuple[float, float, float, float]:
        if not node_entries:
            return 0.0, 0.0, 0.0, 0.0
        ac_model = cast(HybridDomainActorCriticModel, self.model)
        cfg = self.config
        has_stage_base = bool(ac_model.actor_critic_config.enable_monotone_deconfounded_durable_value)
        sorted_nodes = sorted(node_entries, key=lambda row: row[0])
        max_worlds_per_group = 32 if int(ac_model.config.d_model) >= 256 else 64

        val_total = torch.zeros((), device=self.device, dtype=torch.float64)
        margin_total = torch.zeros((), device=self.device, dtype=torch.float64)
        post_total = torch.zeros((), device=self.device, dtype=torch.float64)
        stage_total = torch.zeros((), device=self.device, dtype=torch.float64)

        n_nodes = len(sorted_nodes)
        n_idx = 0
        while n_idx < n_nodes:
            self._check_resource()
            group_end = n_idx
            group_worlds = 0
            group_max_w = 1
            while group_end < n_nodes:
                cand_max_w, cand_as_t, _, _, _ = sorted_nodes[group_end]
                w_cnt = len(cand_as_t.node.post_worlds)
                next_worlds = group_worlds + w_cnt
                next_max_w = max(group_max_w, max(1, int(cand_max_w)))
                if group_end > n_idx and (
                    next_worlds > max_worlds_per_group or (next_worlds * next_max_w) > 24_576
                ):
                    break
                group_worlds = next_worlds
                group_max_w = next_max_w
                group_end += 1

            sub_nodes = sorted_nodes[n_idx:group_end]
            n_idx = group_end

            all_cand_worlds: list[EncodedWorldSnapshot] = []
            all_cand_revs: list[Tensor] = []
            node_specs: list[
                tuple[int, int, Tensor, int | None, Tensor, EncodedWorldSnapshot | None, Tensor | None]
            ] = []
            for _, as_t, chosen_pos, chosen_w_obj, p_rev_item in sub_nodes:
                s_w = len(all_cand_worlds)
                w_cnt = len(as_t.node.post_worlds)
                all_cand_worlds.extend(as_t.node.post_worlds)
                if p_rev_item is not None:
                    all_cand_revs.extend([p_rev_item] * w_cnt)
                node_specs.append(
                    (
                        s_w,
                        s_w + w_cnt,
                        as_t.action_values.to(device=self.device, dtype=self.dtype),
                        chosen_pos,
                        as_t.post_target.to(device=self.device, dtype=self.dtype),
                        chosen_w_obj,
                        p_rev_item,
                    )
                )

            q_s_parts: list[Tensor] = []
            n_all_w = len(all_cand_worlds)
            for ws in range(0, n_all_w, max_worlds_per_group):
                self._check_resource()
                we = min(ws + max_worlds_per_group, n_all_w)
                sub_w = tuple(all_cand_worlds[ws:we])
                wb_sub, dom_sub = collate_encoded_worlds(
                    sub_w,
                    expected_config=sub_w[0].config,
                    expected_fingerprint=sub_w[0].encoding_fingerprint,
                    device=self.device,
                    validate=False,
                )
                sub_rev = torch.stack(all_cand_revs[ws:we], dim=0) if all_cand_revs else None
                q_sub, _ = ac_model.forward_durable_value(
                    wb_sub,
                    dom_sub,
                    None,
                    critic_revival_context=sub_rev,
                    validate=False,
                )
                q_s_parts.append(q_sub)
            q_s_all = torch.cat(q_s_parts, dim=0)
            if not bool(torch.isfinite(q_s_all).all().item()):
                raise FloatingPointError("stateless distill afterstate forward is non-finite")

            node_val_losses: list[Tensor] = []
            node_margin_losses: list[Tensor] = []
            chosen_post_losses: list[Tensor] = []
            chosen_stage_worlds: list[EncodedWorldSnapshot] = []
            chosen_stage_revs: list[Tensor] = []
            chosen_stage_targets: list[Tensor] = []
            for s_w, e_w, q_t_gpu, chosen_pos, post_t_gpu, chosen_w_obj, p_rev_item in node_specs:
                q_s_node = q_s_all[s_w:e_w]
                node_val_losses.append(F.smooth_l1_loss(q_s_node, q_t_gpu))
                if q_s_node.numel() > 1:
                    node_margin_losses.append(
                        F.smooth_l1_loss(q_s_node - q_s_node.mean(), q_t_gpu - q_t_gpu.mean())
                    )
                if chosen_pos is not None and chosen_w_obj is not None:
                    post_blend = 0.5 * q_t_gpu[chosen_pos] + 0.5 * post_t_gpu
                    chosen_post_losses.append(F.smooth_l1_loss(q_s_node[chosen_pos], post_blend))
                    if has_stage_base:
                        chosen_stage_worlds.append(chosen_w_obj)
                        if p_rev_item is not None:
                            chosen_stage_revs.append(p_rev_item)
                        chosen_stage_targets.append(post_t_gpu)

            sub_terms: list[Tensor] = []
            if node_val_losses:
                as_val_loss = torch.stack(node_val_losses).sum() / denom_afterstate
                val_total = val_total + as_val_loss.detach().double()
                if cfg.distill_value_weight:
                    sub_terms.append(0.5 * cfg.distill_value_weight * as_val_loss)
            if node_margin_losses:
                as_margin_loss = torch.stack(node_margin_losses).sum() / denom_afterstate
                margin_total = margin_total + as_margin_loss.detach().double()
                if cfg.distill_afterstate_margin_weight:
                    sub_terms.append(0.5 * cfg.distill_afterstate_margin_weight * as_margin_loss)
            if chosen_post_losses:
                post_loss = torch.stack(chosen_post_losses).sum() / denom_afterstate
                post_total = post_total + post_loss.detach().double()
                if cfg.durable_post_value_weight:
                    sub_terms.append(0.5 * cfg.durable_post_value_weight * post_loss)
            if has_stage_base and chosen_stage_worlds:
                cw_tuple = tuple(chosen_stage_worlds)
                cwb, _ = collate_encoded_worlds(
                    cw_tuple,
                    expected_config=cw_tuple[0].config,
                    expected_fingerprint=cw_tuple[0].encoding_fingerprint,
                    device=self.device,
                    validate=False,
                )
                crev = torch.stack(chosen_stage_revs, dim=0) if chosen_stage_revs else None
                ctarg = torch.stack(chosen_stage_targets, dim=0)
                if cfg.durable_stage_baseline_weight:
                    p_stage_vals = ac_model.forward_durable_stage_baseline(
                        cwb,
                        critic_revival_context=crev,
                    )
                else:
                    with torch.no_grad():
                        p_stage_vals = ac_model.forward_durable_stage_baseline(
                            cwb,
                            critic_revival_context=crev,
                        )
                if not bool(torch.isfinite(p_stage_vals).all().item()):
                    raise FloatingPointError("stateless distill afterstate stage_baseline is non-finite")
                p_stage_loss = F.smooth_l1_loss(p_stage_vals, ctarg, reduction="sum") / denom_afterstate
                stage_total = stage_total + p_stage_loss.detach().double()
                if cfg.durable_stage_baseline_weight:
                    sub_terms.append(0.5 * cfg.durable_stage_baseline_weight * p_stage_loss)

            if sub_terms:
                sub_mega = torch.stack(sub_terms).sum()
                sub_terms.clear()
                if not bool(torch.isfinite(sub_mega).item()):
                    raise FloatingPointError("stateless distill afterstate loss is non-finite")
                sub_mega.backward()  # type: ignore[no-untyped-call]
                del sub_mega
            del q_s_all, q_s_parts

        return (
            float(val_total.item()),
            float(margin_total.item()),
            float(post_total.item()),
            float(stage_total.item()),
        )

    def _backward_chunks_durable_batch(
        self,
        prepared_items: Sequence[_PreparedReference],
        denominators: Mapping[str, int],
        *,
        kickstart_teacher_logs: Sequence[Mapping[int, Tensor]] | None = None,
        kickstart_kl_weight: float = 0.0,
        twin_ref_weights: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[dict[str, float], int]:
        ac_model = cast(HybridDomainActorCriticModel, self.model)
        cfg = self.config
        has_stage_base = bool(ac_model.actor_critic_config.enable_monotone_deconfounded_durable_value)
        has_twin = bool(ac_model.actor_critic_config.enable_twin_durable_value_heads)
        warmup_trunk_frozen = (
            cfg.freeze_trunk_during_durable_warmup and self.update < cfg.durable_warmup_updates
        )
        durable_actor_active = (not warmup_trunk_frozen) and (self.update >= cfg.durable_warmup_updates)
        combat_actor_active = (not warmup_trunk_frozen) and (self.update >= cfg.combat_actor_warmup_updates)
        totals = {
            key: 0.0
            for key in (
                "macro_policy_loss",
                "macro_mpo_policy_loss",
                "macro_mpo_within_policy_loss",
                "macro_pg_policy_loss",
                "macro_entropy",
                "macro_value_loss",
                "durable_value_loss",
                "durable_post_value_loss",
                "durable_stage_baseline_loss",
                "combat_policy_loss",
                "combat_value_loss",
                "combat_entropy",
                "distill_macro_kl",
                "distill_combat_kl",
                "distill_afterstate_margin_loss",
                "distill_afterstate_value_loss",
                "observed_macro_step_kl",
                "observed_combat_step_kl",
                "backward_chunks",
            )
        }
        if not prepared_items:
            return totals, 0

        use_rev = self._use_revival_ctx()
        ref_step_offsets: list[int] = []
        flat_is_macro_cpu: list[bool] = []
        flat_has_dur_cpu: list[bool] = []
        flat_is_choice_cpu: list[bool] = []
        flat_has_as_cpu: list[bool] = []
        flat_contributes_cpu: list[bool] = []
        flat_action_idx_cpu: list[int] = []
        staged_afterstates: list[
            dict[
                int,
                tuple[
                    _AfterstateStepTarget,
                    Tensor,
                    Tensor | None,
                    tuple[Tensor, Tensor] | None,
                    tuple[tuple[Tensor, Tensor], ...],
                    Tensor | None,
                ],
            ]
        ] = []

        cursor_steps = 0
        for item in prepared_items:
            ref_step_offsets.append(cursor_steps)
            as_map: dict[
                int,
                tuple[
                    _AfterstateStepTarget,
                    Tensor,
                    Tensor | None,
                    tuple[Tensor, Tensor] | None,
                    tuple[tuple[Tensor, Tensor], ...],
                    Tensor | None,
                ],
            ] = {}
            assert item.afterstate_targets is not None
            for step_idx, as_t in item.afterstate_targets.items():
                post_rev_item = (
                    None
                    if item.post_revival_ctx is None
                    else item.post_revival_ctx[step_idx].to(device=self.device, dtype=self.dtype)
                )
                probs_differ = not torch.equal(
                    as_t.projection.target_probabilities,
                    as_t.projection.teacher_probabilities,
                )
                joint_prob_item = (
                    as_t.projection.target_probabilities.to(device=self.device, dtype=self.dtype)
                    if (as_t.has_chosen_post and probs_differ)
                    else None
                )
                if (not as_t.has_chosen_post) and probs_differ:
                    det_idx_item = torch.tensor(
                        as_t.node.resolved_candidate_indices,
                        device=self.device,
                        dtype=torch.long,
                    )
                    cond_det_item: tuple[Tensor, Tensor] | None = (
                        det_idx_item,
                        as_t.projection.target_probabilities.to(
                            device=self.device, dtype=self.dtype
                        )[det_idx_item],
                    )
                else:
                    cond_det_item = None
                within_items = (
                    tuple(
                        (
                            torch.tensor(c_idx, device=self.device, dtype=torch.long),
                            w_s.to(device=self.device, dtype=self.dtype),
                        )
                        for c_idx, w_s in as_t.projection.within_branch_targets
                    )
                    if cfg.macro_within_branch_mpo_weight > 0.0 and as_t.projection.within_branch_targets
                    else ()
                )
                as_map[step_idx] = (
                    as_t,
                    as_t.post_target.to(device=self.device, dtype=self.dtype),
                    joint_prob_item,
                    cond_det_item,
                    within_items,
                    post_rev_item,
                )
            staged_afterstates.append(as_map)
            steps = item.reference.steps
            has_stage_w = has_stage_base and bool(cfg.durable_stage_baseline_weight)
            for idx, s in enumerate(steps):
                is_mac = s.control_domain == "macro"
                is_c_entry = (not is_mac) and (idx == 0 or steps[idx - 1].control_domain == "macro")
                has_dur = is_mac or is_c_entry
                is_ch = bool(int(s.snapshot.action_mask.sum()) > 1)
                as_entry = as_map.get(idx)
                has_as = as_entry is not None and as_entry[0].has_chosen_post
                if is_mac:
                    contrib = bool(cfg.macro_value_weight) or has_stage_w
                    if has_as:
                        assert as_entry is not None
                        has_active_mpo = (as_entry[2] is not None) or bool(as_entry[4])
                        contrib = (
                            contrib
                            or bool(cfg.durable_post_value_weight)
                            or has_stage_w
                            or (
                                durable_actor_active
                                and as_entry[0].projection.actor_enabled
                                and has_active_mpo
                                and bool(cfg.macro_policy_weight)
                            )
                        )
                    else:
                        if as_entry is not None:
                            has_active_cond_mpo = (as_entry[3] is not None) or bool(as_entry[4])
                            contrib = contrib or (
                                durable_actor_active
                                and as_entry[0].projection.actor_enabled
                                and has_active_cond_mpo
                                and bool(cfg.macro_policy_weight)
                            )
                        if is_ch and durable_actor_active:
                            contrib = contrib or bool(cfg.macro_policy_weight) or bool(cfg.macro_entropy_weight)
                    if is_ch and kickstart_kl_weight > 0.0 and kickstart_teacher_logs is not None:
                        contrib = True
                    if (
                        is_ch
                        and has_as
                        and durable_actor_active
                        and cfg.macro_mpo_trust_kl_weight > 0.0
                        and bool(cfg.macro_policy_weight)
                    ):
                        contrib = True
                else:
                    contrib = (
                        (is_c_entry and (bool(cfg.macro_value_weight) or has_stage_w))
                        or bool(cfg.combat_value_weight)
                        or (
                            is_ch
                            and combat_actor_active
                            and (bool(cfg.combat_policy_weight) or bool(cfg.combat_entropy_weight))
                        )
                        or (
                            is_ch
                            and kickstart_kl_weight > 0.0
                            and kickstart_teacher_logs is not None
                        )
                    )
                flat_is_macro_cpu.append(is_mac)
                flat_has_dur_cpu.append(has_dur)
                flat_is_choice_cpu.append(is_ch)
                flat_has_as_cpu.append(has_as)
                flat_contributes_cpu.append(contrib)
                flat_action_idx_cpu.append(int(s.action_index))
            cursor_steps += len(steps)

        flat_macro_val = torch.cat([item.macro_values for item in prepared_items], dim=0).to(
            device=self.device, dtype=self.dtype
        )
        flat_combat_val = torch.cat([item.combat_values for item in prepared_items], dim=0).to(
            device=self.device, dtype=self.dtype
        )
        flat_combat_adv = torch.cat([item.combat_advantages for item in prepared_items], dim=0).to(
            device=self.device, dtype=self.dtype
        )
        flat_dur_target = torch.cat(
            [cast(Tensor, item.durable_state_targets) for item in prepared_items], dim=0
        ).to(device=self.device, dtype=self.dtype)
        flat_dur_macro_adv = torch.cat(
            [cast(Tensor, item.durable_macro_advantages) for item in prepared_items], dim=0
        ).to(device=self.device, dtype=self.dtype)
        flat_step_rev_gpu = (
            torch.cat([cast(Tensor, item.step_revival_ctx) for item in prepared_items], dim=0).to(
                device=self.device, dtype=self.dtype
            )
            if use_rev
            else None
        )

        flat_is_macro_gpu = torch.tensor(flat_is_macro_cpu, device=self.device, dtype=torch.bool)
        flat_is_choice_gpu = torch.tensor(flat_is_choice_cpu, device=self.device, dtype=torch.bool)
        flat_has_as_gpu = torch.tensor(flat_has_as_cpu, device=self.device, dtype=torch.bool)
        flat_action_idx_gpu = torch.tensor(flat_action_idx_cpu, device=self.device, dtype=torch.long)

        # Sort chunks by active candidate*local footprint so combat chunks pack tightly without padding bloat.
        keyed_chunks: list[tuple[int, int, int, _PreparedReference, int, int]] = []
        for ref_idx, item in enumerate(prepared_items):
            steps = item.reference.steps
            for start in range(0, len(steps), cfg.chunk_steps):
                stop = min(start + cfg.chunk_steps, len(steps))
                max_cl = 1
                max_w = 1
                for s in steps[start:stop]:
                    snap = s.snapshot
                    loc_d = np.diff(snap.local_offsets)
                    l_max = max(1, int(loc_d.max())) if loc_d.size else 1
                    cl = snap.candidate_count * l_max
                    if cl > max_cl:
                        max_cl = cl
                    if snap.world.token_count > max_w:
                        max_w = snap.world.token_count
                keyed_chunks.append((max_cl, max_w, ref_idx, item, start, stop))
        keyed_chunks.sort(key=lambda row: (row[1] // 64, row[0], row[1]))

        maximum_chunk = max((stop - start for _, _, _, _, start, stop in keyed_chunks), default=0)
        zero_row = ac_model.initial_state(1, device=self.device)[0].detach()
        stateless_durable = self._uses_stateless_durable_value(ac_model)

        denom_dur_val = max(1, denominators["durable_value_steps"])
        denom_macro_steps = max(1, denominators["macro_steps"])
        denom_afterstate = max(1, denominators["afterstate_steps"])
        denom_macro_pol = max(1, denominators["macro_policy_steps"])
        denom_combat_steps = max(1, denominators["combat_steps"])
        denom_combat_pol = max(1, denominators["combat_policy_steps"])

        metric_keys = (
            "macro_policy_loss",
            "macro_mpo_policy_loss",
            "macro_mpo_within_policy_loss",
            "macro_pg_policy_loss",
            "macro_entropy",
            "macro_value_loss",
            "durable_value_loss",
            "durable_post_value_loss",
            "durable_stage_baseline_loss",
            "combat_policy_loss",
            "combat_value_loss",
            "combat_entropy",
            "distill_macro_kl",
            "distill_combat_kl",
            "distill_afterstate_margin_loss",
            "distill_afterstate_value_loss",
            "observed_macro_step_kl",
            "observed_combat_step_kl",
        )

        slice_start = 0
        total_chunks = len(keyed_chunks)
        while slice_start < total_chunks:
            self._check_resource()
            slice_end = self._select_recurrent_chunk_slice_end(ac_model, keyed_chunks, slice_start)
            chunk_slice = [
                (ref_idx, item, start, stop)
                for _, _, ref_idx, item, start, stop in keyed_chunks[slice_start:slice_end]
            ]
            slice_start = slice_end

            hidden_list: list[Tensor] = []
            chunk_has_terms_count = 0
            for ref_idx, item, start, stop in chunk_slice:
                saved = item.evaluation.boundary_hidden[start]
                hidden_list.append(
                    zero_row if saved is None else saved[0].to(device=self.device, dtype=self.dtype)
                )
                base_g = ref_step_offsets[ref_idx]
                if any(flat_contributes_cpu[base_g + s_i] for s_i in range(start, stop)):
                    chunk_has_terms_count += 1

            max_len = max(stop - start for _, _, start, stop in chunk_slice)
            batch_terms: list[Tensor] = []
            slice_totals_gpu: dict[str, Tensor] = {
                key: torch.zeros((), device=self.device, dtype=torch.float64) for key in metric_keys
            }
            finite_ok = torch.ones((), device=self.device, dtype=torch.bool)

            for offset in range(max_len):
                self._check_resource()
                active_c = [c for c, (_, _, start, stop) in enumerate(chunk_slice) if start + offset < stop]
                snaps = tuple(
                    chunk_slice[c][1].reference.steps[chunk_slice[c][2] + offset].snapshot for c in active_c
                )
                gids_cpu = [
                    ref_step_offsets[chunk_slice[c][0]] + chunk_slice[c][2] + offset for c in active_c
                ]
                gids_gpu = torch.tensor(gids_cpu, device=self.device, dtype=torch.long)

                batch = collate_encoded_snapshots(
                    snaps,
                    expected_config=snaps[0].config,
                    expected_fingerprint=snaps[0].encoding_fingerprint,
                    device=self.device,
                    validate=False,
                )
                active_hidden = torch.stack([hidden_list[c] for c in active_c], dim=0)
                active_rev = None if flat_step_rev_gpu is None else flat_step_rev_gpu[gids_gpu]
                with self._autocast_ctx():
                    output = cast(
                        HybridActorCriticOutput,
                        ac_model(
                            batch,
                            active_hidden,
                            critic_revival_context=active_rev,
                            compute_durable_value=False,
                            validate=False,
                        ),
                    )
                for k, c in enumerate(active_c):
                    hidden_list[c] = output.recurrent_state[k]

                masks = batch.candidates.action_mask
                all_logs = torch.log_softmax(output.policy_logits.masked_fill(~masks, -torch.inf), dim=-1)
                safe_logs = all_logs.masked_fill(~masks, 0.0)
                finite_ok = (
                    finite_ok
                    & torch.isfinite(output.policy_logits).all()
                    & torch.isfinite(output.value).all()
                    & torch.isfinite(output.recurrent_state).all()
                    & torch.isfinite(safe_logs).all()
                )
                probs = all_logs.exp().masked_fill(~masks, 0.0)
                all_entropies = -(probs * safe_logs).sum(dim=-1)
                act_idx_k = flat_action_idx_gpu[gids_gpu]
                chosen_logs = safe_logs.gather(1, act_idx_k.unsqueeze(1)).squeeze(1)

                is_macro_k = flat_is_macro_gpu[gids_gpu]
                is_choice_k = flat_is_choice_gpu[gids_gpu]
                has_as_k = flat_has_as_gpu[gids_gpu]

                any_dur = any(flat_has_dur_cpu[g] for g in gids_cpu)
                any_macro = any(flat_is_macro_cpu[g] for g in gids_cpu)
                any_combat = any(not flat_is_macro_cpu[g] for g in gids_cpu)
                any_macro_pg = any(
                    flat_is_macro_cpu[g] and (not flat_has_as_cpu[g]) and flat_is_choice_cpu[g]
                    for g in gids_cpu
                )
                any_combat_choice = any(
                    (not flat_is_macro_cpu[g]) and flat_is_choice_cpu[g] for g in gids_cpu
                )

                if (not stateless_durable) and any_dur:
                    dur_k = [k for k, g in enumerate(gids_cpu) if flat_has_dur_cpu[g]]
                    dur_k_gpu = torch.tensor(dur_k, device=self.device, dtype=torch.long)
                    dur_gids_gpu = gids_gpu[dur_k_gpu]
                    dur_world_batch = WorldTokenBatch(
                        features=batch.world.features[dur_k_gpu],
                        type_ids=batch.world.type_ids[dur_k_gpu],
                        role_ids=batch.world.role_ids[dur_k_gpu],
                        owner_ids=batch.world.owner_ids[dur_k_gpu],
                        entity_ids=batch.world.entity_ids[dur_k_gpu],
                        entity_aux_ids=batch.world.entity_aux_ids[dur_k_gpu],
                        definition_binding_ids=batch.world.definition_binding_ids[dur_k_gpu],
                        relation_binding_ids=batch.world.relation_binding_ids[dur_k_gpu],
                        zone_ids=batch.world.zone_ids[dur_k_gpu],
                        order_ids=batch.world.order_ids[dur_k_gpu],
                        mask=batch.world.mask[dur_k_gpu],
                    )
                    dur_rev = None if active_rev is None else active_rev[dur_k_gpu]
                    if has_twin:
                        with self._autocast_ctx():
                            dv1_s, dv2_s, _ = ac_model.forward_durable_value_pair(
                                dur_world_batch,
                                batch.domain_ids[dur_k_gpu],
                                active_hidden[dur_k_gpu],
                                critic_revival_context=dur_rev,
                                validate=False,
                            )
                        finite_ok = finite_ok & torch.isfinite(dv1_s).all() & torch.isfinite(dv2_s).all()
                        dl1_s = F.smooth_l1_loss(dv1_s, flat_dur_target[dur_gids_gpu], reduction="none")
                        dl2_s = F.smooth_l1_loss(dv2_s, flat_dur_target[dur_gids_gpu], reduction="none")
                        if twin_ref_weights is not None:
                            dur_rids_gpu = torch.tensor(
                                [chunk_slice[active_c[k]][0] for k in dur_k],
                                device=self.device,
                                dtype=torch.long,
                            )
                            durable_loss = (
                                0.5
                                * (
                                    (twin_ref_weights[0][dur_rids_gpu] * dl1_s).sum()
                                    + (twin_ref_weights[1][dur_rids_gpu] * dl2_s).sum()
                                )
                                / denom_dur_val
                            )
                        else:
                            durable_loss = 0.5 * (dl1_s.sum() + dl2_s.sum()) / denom_dur_val
                    else:
                        with self._autocast_ctx():
                            dur_vals, _ = ac_model.forward_durable_value(
                                dur_world_batch,
                                batch.domain_ids[dur_k_gpu],
                                active_hidden[dur_k_gpu],
                                critic_revival_context=dur_rev,
                                validate=False,
                            )
                        finite_ok = finite_ok & torch.isfinite(dur_vals).all()
                        durable_loss = (
                            F.smooth_l1_loss(
                                dur_vals, flat_dur_target[dur_gids_gpu], reduction="sum"
                            )
                            / denom_dur_val
                        )
                    slice_totals_gpu["durable_value_loss"] = (
                        slice_totals_gpu["durable_value_loss"] + durable_loss.detach().double()
                    )
                    if cfg.macro_value_weight:
                        batch_terms.append(0.5 * cfg.macro_value_weight * durable_loss)
                    if has_stage_base:
                        if cfg.durable_stage_baseline_weight:
                            stage_vals = ac_model.forward_durable_stage_baseline(
                                dur_world_batch,
                                critic_revival_context=dur_rev,
                            )
                        else:
                            with torch.no_grad():
                                stage_vals = ac_model.forward_durable_stage_baseline(
                                    dur_world_batch,
                                    critic_revival_context=dur_rev,
                                )
                        finite_ok = finite_ok & torch.isfinite(stage_vals).all()
                        stage_loss = (
                            F.smooth_l1_loss(
                                stage_vals, flat_dur_target[dur_gids_gpu], reduction="sum"
                            )
                            / denom_dur_val
                        )
                        slice_totals_gpu["durable_stage_baseline_loss"] = (
                            slice_totals_gpu["durable_stage_baseline_loss"]
                            + stage_loss.detach().double()
                        )
                        if cfg.durable_stage_baseline_weight:
                            batch_terms.append(0.5 * cfg.durable_stage_baseline_weight * stage_loss)

                if any_macro:
                    macro_huber = F.smooth_l1_loss(
                        output.value, flat_macro_val[gids_gpu], reduction="none"
                    )
                    m_val_loss = torch.where(is_macro_k, macro_huber, 0.0).sum() / denom_macro_steps
                    slice_totals_gpu["macro_value_loss"] = (
                        slice_totals_gpu["macro_value_loss"] + m_val_loss.detach().double()
                    )
                    if cfg.macro_value_weight:
                        batch_terms.append(0.5 * cfg.macro_value_weight * m_val_loss)

                    as_worlds: list[EncodedWorldSnapshot] = []
                    as_hiddens: list[Tensor] = []
                    as_revs: list[Tensor] = []
                    as_post_targets: list[Tensor] = []
                    as_ref_ids: list[int] = []
                    as_act_k: list[int] = []
                    as_act_probs: list[Tensor] = []
                    as_cond_items: list[tuple[int, Tensor, Tensor]] = []
                    as_within_items: list[tuple[int, tuple[tuple[Tensor, Tensor], ...]]] = []
                    for k, c in enumerate(active_c):
                        ref_idx, item, start, _ = chunk_slice[c]
                        index = start + offset
                        staged = staged_afterstates[ref_idx].get(index)
                        if staged is not None:
                            (
                                as_t,
                                post_target_gpu,
                                target_prob_gpu,
                                cond_det_item,
                                staged_within,
                                post_rev_gpu,
                            ) = staged
                            if (not stateless_durable) and as_t.has_chosen_post:
                                step = item.reference.steps[index]
                                chosen_w = as_t.node.post_world_for_action(step.action_index)
                                assert chosen_w is not None
                                as_worlds.append(chosen_w)
                                as_hiddens.append(output.recurrent_state[k])
                                if post_rev_gpu is not None:
                                    as_revs.append(post_rev_gpu)
                                as_post_targets.append(post_target_gpu)
                                as_ref_ids.append(ref_idx)
                            if durable_actor_active and as_t.projection.actor_enabled:
                                if target_prob_gpu is not None:
                                    as_act_k.append(k)
                                    as_act_probs.append(target_prob_gpu)
                                if cond_det_item is not None:
                                    as_cond_items.append((k, cond_det_item[0], cond_det_item[1]))
                                if staged_within:
                                    as_within_items.append((k, staged_within))

                    if as_worlds:
                        self._check_resource()
                        w_tuple = tuple(as_worlds)
                        world_batch, domain_ids = collate_encoded_worlds(
                            w_tuple,
                            expected_config=w_tuple[0].config,
                            expected_fingerprint=w_tuple[0].encoding_fingerprint,
                            device=self.device,
                            validate=False,
                        )
                        stacked_as_revs = torch.stack(as_revs, dim=0) if as_revs else None
                        stacked_post_targets = torch.stack(as_post_targets, dim=0)
                        if has_twin:
                            with self._autocast_ctx():
                                pv1_s, pv2_s, _ = ac_model.forward_durable_value_pair(
                                    world_batch,
                                    domain_ids,
                                    torch.stack(as_hiddens, dim=0),
                                    critic_revival_context=stacked_as_revs,
                                    validate=False,
                                )
                            finite_ok = finite_ok & torch.isfinite(pv1_s).all() & torch.isfinite(pv2_s).all()
                            pl1_s = F.smooth_l1_loss(pv1_s, stacked_post_targets, reduction="none")
                            pl2_s = F.smooth_l1_loss(pv2_s, stacked_post_targets, reduction="none")
                            if twin_ref_weights is not None:
                                as_rids_gpu = torch.tensor(as_ref_ids, device=self.device, dtype=torch.long)
                                post_loss = (
                                    0.5
                                    * (
                                        (twin_ref_weights[0][as_rids_gpu] * pl1_s).sum()
                                        + (twin_ref_weights[1][as_rids_gpu] * pl2_s).sum()
                                    )
                                    / denom_afterstate
                                )
                            else:
                                post_loss = 0.5 * (pl1_s.sum() + pl2_s.sum()) / denom_afterstate
                        else:
                            with self._autocast_ctx():
                                post_vals, _ = ac_model.forward_durable_value(
                                    world_batch,
                                    domain_ids,
                                    torch.stack(as_hiddens, dim=0),
                                    critic_revival_context=stacked_as_revs,
                                    validate=False,
                                )
                            finite_ok = finite_ok & torch.isfinite(post_vals).all()
                            post_loss = (
                                F.smooth_l1_loss(
                                    post_vals, stacked_post_targets, reduction="sum"
                                )
                                / denom_afterstate
                            )
                        slice_totals_gpu["durable_post_value_loss"] = (
                            slice_totals_gpu["durable_post_value_loss"] + post_loss.detach().double()
                        )
                        if cfg.durable_post_value_weight:
                            batch_terms.append(0.5 * cfg.durable_post_value_weight * post_loss)
                        if has_stage_base:
                            if cfg.durable_stage_baseline_weight:
                                post_stage_vals = ac_model.forward_durable_stage_baseline(
                                    world_batch,
                                    critic_revival_context=stacked_as_revs,
                                )
                            else:
                                with torch.no_grad():
                                    post_stage_vals = ac_model.forward_durable_stage_baseline(
                                        world_batch,
                                        critic_revival_context=stacked_as_revs,
                                    )
                            finite_ok = finite_ok & torch.isfinite(post_stage_vals).all()
                            post_stage_loss = (
                                F.smooth_l1_loss(
                                    post_stage_vals, stacked_post_targets, reduction="sum"
                                )
                                / denom_afterstate
                            )
                            slice_totals_gpu["durable_stage_baseline_loss"] = (
                                slice_totals_gpu["durable_stage_baseline_loss"]
                                + post_stage_loss.detach().double()
                            )
                            if cfg.durable_stage_baseline_weight:
                                batch_terms.append(
                                    0.5 * cfg.durable_stage_baseline_weight * post_stage_loss
                                )

                    mpo_actor_terms: list[Tensor] = []
                    if as_act_k:
                        c_cap_step = int(safe_logs.shape[1])
                        padded_probs = torch.zeros(
                            (len(as_act_k), c_cap_step), device=self.device, dtype=safe_logs.dtype
                        )
                        for m_i, tp in enumerate(as_act_probs):
                            padded_probs[m_i, : tp.numel()] = tp.to(dtype=safe_logs.dtype)
                        k_idx_gpu = torch.tensor(as_act_k, device=self.device, dtype=torch.long)
                        joint_mpo_actor = -(padded_probs * safe_logs[k_idx_gpu]).sum() / denom_macro_pol
                        mpo_actor_terms.append(joint_mpo_actor)
                    if as_cond_items:
                        cond_step_losses: list[Tensor] = []
                        for k_c, det_idx_gpu, target_det_gpu in as_cond_items:
                            cond_det_logs = torch.log_softmax(all_logs[k_c][det_idx_gpu], dim=0)
                            safe_cond_det_logs = cond_det_logs.masked_fill(
                                ~torch.isfinite(cond_det_logs), 0.0
                            )
                            cond_step_losses.append(
                                -(target_det_gpu.to(dtype=safe_logs.dtype) * safe_cond_det_logs).sum()
                            )
                        cond_joint_mpo_actor = torch.stack(cond_step_losses).sum() / denom_macro_pol
                        mpo_actor_terms.append(cond_joint_mpo_actor)
                    if as_within_items:
                        within_step_losses: list[Tensor] = []
                        for k_w, staged_within in as_within_items:
                            row_logs = all_logs[k_w]
                            b_losses: list[Tensor] = []
                            for idx_gpu, w_star_gpu in staged_within:
                                cond_logs = torch.log_softmax(row_logs[idx_gpu], dim=0)
                                safe_cond_logs = cond_logs.masked_fill(~torch.isfinite(cond_logs), 0.0)
                                b_losses.append(-(w_star_gpu.to(dtype=safe_logs.dtype) * safe_cond_logs).sum())
                            within_step_losses.append(torch.stack(b_losses).mean())
                        within_mpo_actor = (
                            cfg.macro_within_branch_mpo_weight
                            * torch.stack(within_step_losses).sum()
                            / denom_macro_pol
                        )
                        slice_totals_gpu["macro_mpo_within_policy_loss"] = (
                            slice_totals_gpu["macro_mpo_within_policy_loss"]
                            + within_mpo_actor.detach().double()
                        )
                        mpo_actor_terms.append(within_mpo_actor)
                    if mpo_actor_terms:
                        mpo_actor = torch.stack(mpo_actor_terms).sum()
                        mpo_det = mpo_actor.detach().double()
                        slice_totals_gpu["macro_policy_loss"] = (
                            slice_totals_gpu["macro_policy_loss"] + mpo_det
                        )
                        slice_totals_gpu["macro_mpo_policy_loss"] = (
                            slice_totals_gpu["macro_mpo_policy_loss"] + mpo_det
                        )
                        if cfg.macro_policy_weight:
                            batch_terms.append(0.5 * cfg.macro_policy_weight * mpo_actor)

                    if any_macro_pg and durable_actor_active:
                        pg_mask = is_macro_k & (~has_as_k) & is_choice_k
                        if cfg.clip_macro_pg_fallback:
                            c_cap_pg = int(safe_logs.shape[1])
                            pg_rows_k: list[int] = []
                            pg_adv_list: list[Tensor] = []
                            pg_t_chosen_list: list[float] = []
                            pg_t_logs_rows: list[Tensor] = []
                            for k, c in enumerate(active_c):
                                g_id = gids_cpu[k]
                                if (
                                    flat_is_macro_cpu[g_id]
                                    and (not flat_has_as_cpu[g_id])
                                    and flat_is_choice_cpu[g_id]
                                ):
                                    _, item_c, start_c, _ = chunk_slice[c]
                                    step_idx_c = start_c + offset
                                    assert item_c.evaluation.step_log_probs is not None
                                    t_logs_cpu = item_c.evaluation.step_log_probs[step_idx_c]
                                    pg_rows_k.append(k)
                                    pg_adv_list.append(flat_dur_macro_adv[g_id])
                                    pg_t_chosen_list.append(
                                        float(t_logs_cpu[flat_action_idx_cpu[g_id]].item())
                                    )
                                    padded_tl = torch.full((c_cap_pg,), -torch.inf, dtype=torch.float32)
                                    padded_tl[: t_logs_cpu.numel()] = t_logs_cpu.float()
                                    pg_t_logs_rows.append(padded_tl)
                            k_pg_gpu = torch.tensor(pg_rows_k, device=self.device, dtype=torch.long)
                            adv_pg_gpu = torch.stack(pg_adv_list, dim=0)
                            t_chosen_gpu = torch.tensor(
                                pg_t_chosen_list, device=self.device, dtype=safe_logs.dtype
                            )
                            t_logs_gpu = torch.stack(pg_t_logs_rows, dim=0).to(
                                device=self.device, dtype=safe_logs.dtype
                            )
                            valid_t = torch.isfinite(t_logs_gpu)
                            safe_t_logs = t_logs_gpu.masked_fill(~valid_t, 0.0)
                            t_probs_gpu = t_logs_gpu.exp().masked_fill(~valid_t, 0.0)
                            cur_chosen = chosen_logs[k_pg_gpu]
                            ratio = torch.exp((cur_chosen - t_chosen_gpu).clamp(-10.0, 10.0))
                            clipped_ratio = ratio.clamp(
                                1.0 - cfg.macro_pg_clip_eps, 1.0 + cfg.macro_pg_clip_eps
                            )
                            surr = torch.minimum(ratio * adv_pg_gpu, clipped_ratio * adv_pg_gpu)
                            kl_pen = (t_probs_gpu * (safe_t_logs - safe_logs[k_pg_gpu])).sum(dim=-1)
                            pg_actor = (-surr + cfg.macro_pg_kl_weight * kl_pen).sum() / denom_macro_pol
                        else:
                            pg_actor = (
                                torch.where(pg_mask, -chosen_logs * flat_dur_macro_adv[gids_gpu], 0.0).sum()
                                / denom_macro_pol
                            )
                        macro_ent_source = (
                            hierarchical_macro_entropy(
                                safe_logs,
                                masks,
                                batch.candidates.policy_branch_ids,
                                include_branch_entropy=not cfg.macro_pg_within_branch_entropy_only,
                            )
                            if cfg.enable_macro_branch_prior
                            else all_entropies
                        )
                        pg_ent = torch.where(pg_mask, macro_ent_source, 0.0).sum() / denom_macro_pol
                        pg_det = pg_actor.detach().double()
                        slice_totals_gpu["macro_policy_loss"] = (
                            slice_totals_gpu["macro_policy_loss"] + pg_det
                        )
                        slice_totals_gpu["macro_pg_policy_loss"] = (
                            slice_totals_gpu["macro_pg_policy_loss"] + pg_det
                        )
                        slice_totals_gpu["macro_entropy"] = (
                            slice_totals_gpu["macro_entropy"] + pg_ent.detach().double()
                        )
                        if cfg.macro_policy_weight:
                            batch_terms.append(0.5 * cfg.macro_policy_weight * pg_actor)
                        if cfg.macro_entropy_weight:
                            batch_terms.append(-0.5 * cfg.macro_entropy_weight * pg_ent)

                if any_combat:
                    is_combat_k = ~is_macro_k
                    combat_huber = F.smooth_l1_loss(
                        output.value, flat_combat_val[gids_gpu], reduction="none"
                    )
                    c_val_loss = torch.where(is_combat_k, combat_huber, 0.0).sum() / denom_combat_steps
                    slice_totals_gpu["combat_value_loss"] = (
                        slice_totals_gpu["combat_value_loss"] + c_val_loss.detach().double()
                    )
                    if cfg.combat_value_weight:
                        batch_terms.append(0.5 * cfg.combat_value_weight * c_val_loss)
                    if any_combat_choice:
                        cpol_mask = is_combat_k & is_choice_k
                        if cfg.clip_combat_pg:
                            c_cap_cp = int(safe_logs.shape[1])
                            cp_rows_k: list[int] = []
                            cp_adv_list: list[Tensor] = []
                            cp_t_chosen_list: list[float] = []
                            cp_t_logs_rows: list[Tensor] = []
                            for k, c in enumerate(active_c):
                                g_id = gids_cpu[k]
                                if (not flat_is_macro_cpu[g_id]) and flat_is_choice_cpu[g_id]:
                                    _, item_c, start_c, _ = chunk_slice[c]
                                    step_idx_c = start_c + offset
                                    assert item_c.evaluation.step_log_probs is not None
                                    ct_logs_cpu = item_c.evaluation.step_log_probs[step_idx_c]
                                    cp_rows_k.append(k)
                                    cp_adv_list.append(flat_combat_adv[g_id])
                                    cp_t_chosen_list.append(
                                        float(ct_logs_cpu[flat_action_idx_cpu[g_id]].item())
                                    )
                                    padded_ctl = torch.full((c_cap_cp,), -torch.inf, dtype=torch.float32)
                                    padded_ctl[: ct_logs_cpu.numel()] = ct_logs_cpu.float()
                                    cp_t_logs_rows.append(padded_ctl)
                            k_cp_gpu = torch.tensor(cp_rows_k, device=self.device, dtype=torch.long)
                            adv_cp_gpu = torch.stack(cp_adv_list, dim=0)
                            t_cp_chosen_gpu = torch.tensor(
                                cp_t_chosen_list, device=self.device, dtype=safe_logs.dtype
                            )
                            t_cp_logs_gpu = torch.stack(cp_t_logs_rows, dim=0).to(
                                device=self.device, dtype=safe_logs.dtype
                            )
                            valid_cp_t = torch.isfinite(t_cp_logs_gpu) & masks[k_cp_gpu]
                            safe_cp_t_logs = t_cp_logs_gpu.masked_fill(~valid_cp_t, 0.0)
                            t_cp_probs_gpu = t_cp_logs_gpu.exp().masked_fill(~valid_cp_t, 0.0)
                            cur_cp_chosen = chosen_logs[k_cp_gpu]
                            cp_ratio = torch.exp((cur_cp_chosen - t_cp_chosen_gpu).clamp(-10.0, 10.0))
                            cp_clipped = cp_ratio.clamp(
                                1.0 - cfg.combat_pg_clip_eps, 1.0 + cfg.combat_pg_clip_eps
                            )
                            cp_surr = torch.minimum(cp_ratio * adv_cp_gpu, cp_clipped * adv_cp_gpu)
                            cp_kl_pen = (
                                t_cp_probs_gpu * (safe_cp_t_logs - safe_logs[k_cp_gpu].masked_fill(~valid_cp_t, 0.0))
                            ).sum(dim=-1)
                            c_actor = (-cp_surr + cfg.combat_pg_kl_weight * cp_kl_pen).sum() / denom_combat_pol
                        else:
                            c_actor = (
                                torch.where(cpol_mask, -chosen_logs * flat_combat_adv[gids_gpu], 0.0).sum()
                                / denom_combat_pol
                            )
                        c_ent = torch.where(cpol_mask, all_entropies, 0.0).sum() / denom_combat_pol
                        slice_totals_gpu["combat_policy_loss"] = (
                            slice_totals_gpu["combat_policy_loss"] + c_actor.detach().double()
                        )
                        slice_totals_gpu["combat_entropy"] = (
                            slice_totals_gpu["combat_entropy"] + c_ent.detach().double()
                        )
                        if combat_actor_active:
                            if cfg.combat_policy_weight:
                                batch_terms.append(0.5 * cfg.combat_policy_weight * c_actor)
                            if cfg.combat_entropy_weight:
                                batch_terms.append(-0.5 * cfg.combat_entropy_weight * c_ent)

                if (
                    kickstart_kl_weight > 0.0
                    and kickstart_teacher_logs is not None
                    and any(flat_is_choice_cpu[g] for g in gids_cpu)
                ):
                    c_cap_ks = int(safe_logs.shape[1])
                    ks_rows_k: list[int] = []
                    ks_is_mac: list[bool] = []
                    ks_t_logs_rows: list[Tensor] = []
                    for k, c in enumerate(active_c):
                        g_id = gids_cpu[k]
                        if flat_is_choice_cpu[g_id]:
                            r_idx_c, _, start_c, _ = chunk_slice[c]
                            step_idx_c = start_c + offset
                            ks_t_logs_cpu = kickstart_teacher_logs[r_idx_c].get(step_idx_c)
                            if ks_t_logs_cpu is not None:
                                ks_rows_k.append(k)
                                ks_is_mac.append(flat_is_macro_cpu[g_id])
                                padded_ks = torch.full((c_cap_ks,), -torch.inf, dtype=torch.float32)
                                padded_ks[: ks_t_logs_cpu.numel()] = ks_t_logs_cpu.float()
                                ks_t_logs_rows.append(padded_ks)
                    if ks_rows_k:
                        k_ks_gpu = torch.tensor(ks_rows_k, device=self.device, dtype=torch.long)
                        is_mac_ks_gpu = torch.tensor(ks_is_mac, device=self.device, dtype=torch.bool)
                        t_ks_gpu = torch.stack(ks_t_logs_rows, dim=0).to(
                            device=self.device, dtype=safe_logs.dtype
                        )
                        valid_ks = torch.isfinite(t_ks_gpu) & masks[k_ks_gpu]
                        safe_t_ks = t_ks_gpu.masked_fill(~valid_ks, 0.0)
                        probs_t_ks = t_ks_gpu.exp().masked_fill(~valid_ks, 0.0)
                        safe_s_ks = safe_logs[k_ks_gpu].masked_fill(~valid_ks, 0.0)
                        ks_kl_vec = (probs_t_ks * (safe_t_ks - safe_s_ks)).sum(dim=-1)
                        if any(ks_is_mac):
                            ks_mac_kl = torch.where(is_mac_ks_gpu, ks_kl_vec, 0.0).sum() / denom_macro_pol
                            slice_totals_gpu["distill_macro_kl"] = (
                                slice_totals_gpu["distill_macro_kl"]
                                + ks_mac_kl.detach().clamp_min(0.0).double()
                            )
                            batch_terms.append(0.5 * kickstart_kl_weight * ks_mac_kl)
                        if any(not flag for flag in ks_is_mac):
                            ks_com_kl = torch.where(~is_mac_ks_gpu, ks_kl_vec, 0.0).sum() / denom_combat_pol
                            slice_totals_gpu["distill_combat_kl"] = (
                                slice_totals_gpu["distill_combat_kl"]
                                + ks_com_kl.detach().clamp_min(0.0).double()
                            )
                            batch_terms.append(0.5 * kickstart_kl_weight * ks_com_kl)

                if any(flat_is_choice_cpu[g] for g in gids_cpu):
                    c_cap_obs = int(safe_logs.shape[1])
                    obs_rows_k: list[int] = []
                    obs_is_mac: list[bool] = []
                    obs_has_as: list[bool] = []
                    obs_old_rows: list[Tensor] = []
                    for k, c in enumerate(active_c):
                        g_id = gids_cpu[k]
                        if flat_is_choice_cpu[g_id]:
                            _, item_c, start_c, _ = chunk_slice[c]
                            step_idx_c = start_c + offset
                            if item_c.evaluation.step_log_probs is not None:
                                old_logs_cpu = item_c.evaluation.step_log_probs.get(step_idx_c)
                                if old_logs_cpu is not None:
                                    obs_rows_k.append(k)
                                    obs_is_mac.append(flat_is_macro_cpu[g_id])
                                    obs_has_as.append(flat_has_as_cpu[g_id])
                                    padded_old = torch.full((c_cap_obs,), -torch.inf, dtype=torch.float32)
                                    padded_old[: old_logs_cpu.numel()] = old_logs_cpu.float()
                                    obs_old_rows.append(padded_old)
                    if obs_rows_k:
                        k_obs_gpu = torch.tensor(obs_rows_k, device=self.device, dtype=torch.long)
                        is_mac_obs_gpu = torch.tensor(obs_is_mac, device=self.device, dtype=torch.bool)
                        has_as_obs_gpu = torch.tensor(obs_has_as, device=self.device, dtype=torch.bool)
                        old_obs_gpu = torch.stack(obs_old_rows, dim=0).to(
                            device=self.device, dtype=safe_logs.dtype
                        )
                        valid_obs = torch.isfinite(old_obs_gpu) & masks[k_obs_gpu]
                        safe_old_obs = old_obs_gpu.masked_fill(~valid_obs, 0.0)
                        probs_old_obs = old_obs_gpu.exp().masked_fill(~valid_obs, 0.0)
                        safe_cur_obs = safe_logs[k_obs_gpu].masked_fill(~valid_obs, 0.0)
                        obs_kl_vec = (probs_old_obs * (safe_old_obs - safe_cur_obs)).sum(dim=-1)
                        if any(obs_is_mac):
                            obs_mac_kl = torch.where(is_mac_obs_gpu, obs_kl_vec, 0.0).sum() / denom_macro_pol
                            slice_totals_gpu["observed_macro_step_kl"] = (
                                slice_totals_gpu["observed_macro_step_kl"]
                                + obs_mac_kl.detach().clamp_min(0.0).double()
                            )
                            if (
                                durable_actor_active
                                and cfg.macro_mpo_trust_kl_weight > 0.0
                                and cfg.macro_policy_weight > 0.0
                                and any(m and a for m, a in zip(obs_is_mac, obs_has_as, strict=True))
                            ):
                                as_mac_mask = is_mac_obs_gpu & has_as_obs_gpu
                                mpo_trust_kl = (
                                    cfg.macro_mpo_trust_kl_weight
                                    * torch.where(as_mac_mask, obs_kl_vec, 0.0).sum()
                                    / denom_macro_pol
                                )
                                mpo_trust_det = mpo_trust_kl.detach().clamp_min(0.0).double()
                                slice_totals_gpu["macro_policy_loss"] = (
                                    slice_totals_gpu["macro_policy_loss"] + mpo_trust_det
                                )
                                slice_totals_gpu["macro_mpo_policy_loss"] = (
                                    slice_totals_gpu["macro_mpo_policy_loss"] + mpo_trust_det
                                )
                                batch_terms.append(0.5 * cfg.macro_policy_weight * mpo_trust_kl)
                        if any(not flag for flag in obs_is_mac):
                            obs_com_kl = torch.where(~is_mac_obs_gpu, obs_kl_vec, 0.0).sum() / denom_combat_pol
                            slice_totals_gpu["observed_combat_step_kl"] = (
                                slice_totals_gpu["observed_combat_step_kl"]
                                + obs_com_kl.detach().clamp_min(0.0).double()
                            )

            if batch_terms:
                mega_loss = torch.stack(batch_terms).sum()
                batch_terms.clear()
                if not bool((finite_ok & torch.isfinite(mega_loss)).item()):
                    raise FloatingPointError("macro cohort chunk forward or loss is non-finite")
                mega_loss.backward()  # type: ignore[no-untyped-call]
                del mega_loss
            elif not bool(finite_ok.item()):
                raise FloatingPointError("macro cohort chunk forward is non-finite")
            if chunk_has_terms_count:
                totals["backward_chunks"] += float(chunk_has_terms_count)
            hidden_list.clear()
            stacked_metrics = torch.stack([slice_totals_gpu[key] for key in metric_keys]).cpu().tolist()
            for key, val in zip(metric_keys, stacked_metrics, strict=True):
                totals[key] += float(val)
            self._check_resource()

        if stateless_durable:
            dur_entries: list[tuple[EncodedWorldSnapshot, int]] = []
            dur_ref_ids: list[int] = []
            post_entries: list[tuple[EncodedWorldSnapshot, Tensor, Tensor | None]] = []
            post_ref_ids: list[int] = []
            for ref_idx, item in enumerate(prepared_items):
                base_g = ref_step_offsets[ref_idx]
                steps = item.reference.steps
                for idx, s in enumerate(steps):
                    if flat_has_dur_cpu[base_g + idx]:
                        dur_entries.append((cast(EncodedWorldSnapshot, s.snapshot), base_g + idx))
                        dur_ref_ids.append(ref_idx)
                for step_idx, staged in staged_afterstates[ref_idx].items():
                    as_t, post_target_gpu, _, _, _, post_rev_gpu = staged
                    if as_t.has_chosen_post:
                        chosen_w = as_t.node.post_world_for_action(steps[step_idx].action_index)
                        assert chosen_w is not None
                        post_entries.append((chosen_w, post_target_gpu, post_rev_gpu))
                        post_ref_ids.append(ref_idx)
            dur_v_loss, dur_s_loss = self._backward_stateless_pre_durable(
                dur_entries=dur_entries,
                flat_dur_target=flat_dur_target,
                flat_stage_target=flat_dur_target,
                flat_step_rev_gpu=flat_step_rev_gpu,
                denom_dur_val=denom_dur_val,
                durable_value_weight=cfg.macro_value_weight,
                stage_baseline_weight=cfg.durable_stage_baseline_weight,
                dur_ref_ids=dur_ref_ids,
                twin_ref_weights=twin_ref_weights,
            )
            post_v_loss, post_s_loss = self._backward_stateless_chosen_post_durable(
                post_entries=post_entries,
                denom_afterstate=denom_afterstate,
                post_ref_ids=post_ref_ids,
                twin_ref_weights=twin_ref_weights,
            )
            totals["durable_value_loss"] += dur_v_loss
            totals["durable_post_value_loss"] += post_v_loss
            totals["durable_stage_baseline_loss"] += dur_s_loss + post_s_loss
            self._check_resource()
        return totals, maximum_chunk

    def _distill_epoch_durable(
        self,
        prepared_items: Sequence[_PreparedReference],
        student_boundaries: Sequence[Mapping[int, Tensor | None]],
        denominators: Mapping[str, int],
    ) -> tuple[dict[str, float], int]:
        ac_model = cast(HybridDomainActorCriticModel, self.model)
        cfg = self.config
        has_stage_base = bool(ac_model.actor_critic_config.enable_monotone_deconfounded_durable_value)
        metric_keys = (
            "macro_policy_loss",
            "macro_mpo_policy_loss",
            "macro_mpo_within_policy_loss",
            "macro_pg_policy_loss",
            "macro_entropy",
            "macro_value_loss",
            "durable_value_loss",
            "durable_post_value_loss",
            "durable_stage_baseline_loss",
            "combat_policy_loss",
            "combat_value_loss",
            "combat_entropy",
            "distill_macro_kl",
            "distill_combat_kl",
            "distill_afterstate_margin_loss",
            "distill_afterstate_value_loss",
            "observed_macro_step_kl",
            "observed_combat_step_kl",
        )
        totals = {key: 0.0 for key in (*metric_keys, "backward_chunks")}
        if not prepared_items:
            return totals, 0

        use_rev = self._use_revival_ctx(ac_model)
        step_rev_list: list[Tensor] = []
        post_rev_per_ref: list[Tensor | None] = []
        for item in prepared_items:
            if use_rev:
                if item.step_revival_ctx is not None and item.post_revival_ctx is not None:
                    step_rev_list.append(item.step_revival_ctx)
                    post_rev_per_ref.append(item.post_revival_ctx)
                else:
                    s_r, p_r, _ = self._trajectory_critic_revival_contexts(item.reference)
                    step_rev_list.append(s_r)
                    post_rev_per_ref.append(p_r)
            else:
                post_rev_per_ref.append(None)

        ref_step_offsets: list[int] = []
        flat_is_macro_cpu: list[bool] = []
        flat_has_dur_cpu: list[bool] = []
        flat_is_choice_cpu: list[bool] = []
        cursor_steps = 0
        for item in prepared_items:
            ref_step_offsets.append(cursor_steps)
            steps = item.reference.steps
            for idx, s in enumerate(steps):
                is_mac = s.control_domain == "macro"
                is_c_entry = (not is_mac) and (idx == 0 or steps[idx - 1].control_domain == "macro")
                flat_is_macro_cpu.append(is_mac)
                flat_has_dur_cpu.append(is_mac or is_c_entry)
                flat_is_choice_cpu.append(bool(int(s.snapshot.action_mask.sum()) > 1))
            cursor_steps += len(steps)

        flat_macro_val = torch.cat(
            [0.5 * item.evaluation.values + 0.5 * item.macro_values for item in prepared_items],
            dim=0,
        ).to(device=self.device, dtype=self.dtype)
        flat_combat_val = torch.cat(
            [0.5 * item.evaluation.values + 0.5 * item.combat_values for item in prepared_items],
            dim=0,
        ).to(device=self.device, dtype=self.dtype)
        flat_dur_target = torch.cat(
            [
                0.5 * cast(Tensor, item.evaluation.durable_values)
                + 0.5 * cast(Tensor, item.durable_state_targets)
                for item in prepared_items
            ],
            dim=0,
        ).to(device=self.device, dtype=self.dtype)
        flat_stage_target = torch.cat(
            [cast(Tensor, item.durable_state_targets) for item in prepared_items],
            dim=0,
        ).to(device=self.device, dtype=self.dtype)
        flat_step_rev_gpu = (
            torch.cat(step_rev_list, dim=0).to(device=self.device, dtype=self.dtype)
            if use_rev
            else None
        )
        flat_is_macro_gpu = torch.tensor(flat_is_macro_cpu, device=self.device, dtype=torch.bool)
        flat_is_choice_gpu = torch.tensor(flat_is_choice_cpu, device=self.device, dtype=torch.bool)

        keyed_chunks: list[tuple[int, int, int, _PreparedReference, int, int]] = []
        for ref_idx, item in enumerate(prepared_items):
            steps = item.reference.steps
            for start in range(0, len(steps), cfg.chunk_steps):
                stop = min(start + cfg.chunk_steps, len(steps))
                max_cl = 1
                max_w = 1
                for s in steps[start:stop]:
                    snap = s.snapshot
                    loc_d = np.diff(snap.local_offsets)
                    l_max = max(1, int(loc_d.max())) if loc_d.size else 1
                    cl = snap.candidate_count * l_max
                    if cl > max_cl:
                        max_cl = cl
                    if snap.world.token_count > max_w:
                        max_w = snap.world.token_count
                keyed_chunks.append((max_cl, max_w, ref_idx, item, start, stop))
        keyed_chunks.sort(key=lambda row: (row[1] // 64, row[0], row[1]))

        maximum_chunk = max((stop - start for _, _, _, _, start, stop in keyed_chunks), default=0)
        zero_row = ac_model.initial_state(1, device=self.device)[0].detach()
        stateless_durable = self._uses_stateless_durable_value(ac_model)

        denom_dur_val = max(1, denominators["durable_value_steps"])
        denom_macro_steps = max(1, denominators["macro_steps"])
        denom_afterstate = max(1, denominators["afterstate_steps"])
        denom_macro_pol = max(1, denominators["macro_policy_steps"])
        denom_combat_steps = max(1, denominators["combat_steps"])
        denom_combat_pol = max(1, denominators["combat_policy_steps"])

        slice_start = 0
        total_chunks = len(keyed_chunks)
        afterstate_grad_cap = 32 if int(ac_model.config.d_model) >= 256 else 128

        while slice_start < total_chunks:
            self._check_resource()
            slice_end = self._select_recurrent_chunk_slice_end(ac_model, keyed_chunks, slice_start)
            chunk_slice = [
                (ref_idx, item, start, stop)
                for _, _, ref_idx, item, start, stop in keyed_chunks[slice_start:slice_end]
            ]
            slice_start = slice_end

            hidden_list: list[Tensor] = []
            for ref_idx, _, start, _ in chunk_slice:
                saved = student_boundaries[ref_idx].get(start)
                hidden_list.append(
                    zero_row if saved is None else saved[0].to(device=self.device, dtype=self.dtype)
                )

            max_len = max(stop - start for _, _, start, stop in chunk_slice)
            batch_terms: list[Tensor] = []
            slice_totals_gpu: dict[str, Tensor] = {
                key: torch.zeros((), device=self.device, dtype=torch.float64) for key in metric_keys
            }
            finite_ok = torch.ones((), device=self.device, dtype=torch.bool)

            for offset in range(max_len):
                self._check_resource()
                active_c = [c for c, (_, _, start, stop) in enumerate(chunk_slice) if start + offset < stop]
                snaps = tuple(
                    chunk_slice[c][1].reference.steps[chunk_slice[c][2] + offset].snapshot for c in active_c
                )
                gids_cpu = [
                    ref_step_offsets[chunk_slice[c][0]] + chunk_slice[c][2] + offset for c in active_c
                ]
                gids_gpu = torch.tensor(gids_cpu, device=self.device, dtype=torch.long)

                batch = collate_encoded_snapshots(
                    snaps,
                    expected_config=snaps[0].config,
                    expected_fingerprint=snaps[0].encoding_fingerprint,
                    device=self.device,
                    validate=False,
                )
                active_hidden = torch.stack([hidden_list[c] for c in active_c], dim=0)
                active_rev = None if flat_step_rev_gpu is None else flat_step_rev_gpu[gids_gpu]
                with self._autocast_ctx():
                    output = cast(
                        HybridActorCriticOutput,
                        ac_model(
                            batch,
                            active_hidden,
                            critic_revival_context=active_rev,
                            compute_durable_value=False,
                            validate=False,
                        ),
                    )
                for k, c in enumerate(active_c):
                    hidden_list[c] = output.recurrent_state[k]

                masks = batch.candidates.action_mask
                all_logs = torch.log_softmax(output.policy_logits.masked_fill(~masks, -torch.inf), dim=-1)
                safe_logs = all_logs.masked_fill(~masks, 0.0)
                finite_ok = (
                    finite_ok
                    & torch.isfinite(output.policy_logits).all()
                    & torch.isfinite(output.value).all()
                    & torch.isfinite(output.recurrent_state).all()
                    & torch.isfinite(safe_logs).all()
                )
                probs = all_logs.exp().masked_fill(~masks, 0.0)
                all_entropies = -(probs * safe_logs).sum(dim=-1)

                is_macro_k = flat_is_macro_gpu[gids_gpu]
                is_choice_k = flat_is_choice_gpu[gids_gpu]
                any_dur = any(flat_has_dur_cpu[g] for g in gids_cpu)
                any_macro = any(flat_is_macro_cpu[g] for g in gids_cpu)
                any_combat = any(not flat_is_macro_cpu[g] for g in gids_cpu)
                any_choice = any(flat_is_choice_cpu[g] for g in gids_cpu)

                if (not stateless_durable) and any_dur:
                    dur_k = [k for k, g in enumerate(gids_cpu) if flat_has_dur_cpu[g]]
                    dur_k_gpu = torch.tensor(dur_k, device=self.device, dtype=torch.long)
                    dur_gids_gpu = gids_gpu[dur_k_gpu]
                    dur_world_batch = WorldTokenBatch(
                        features=batch.world.features[dur_k_gpu],
                        type_ids=batch.world.type_ids[dur_k_gpu],
                        role_ids=batch.world.role_ids[dur_k_gpu],
                        owner_ids=batch.world.owner_ids[dur_k_gpu],
                        entity_ids=batch.world.entity_ids[dur_k_gpu],
                        entity_aux_ids=batch.world.entity_aux_ids[dur_k_gpu],
                        definition_binding_ids=batch.world.definition_binding_ids[dur_k_gpu],
                        relation_binding_ids=batch.world.relation_binding_ids[dur_k_gpu],
                        zone_ids=batch.world.zone_ids[dur_k_gpu],
                        order_ids=batch.world.order_ids[dur_k_gpu],
                        mask=batch.world.mask[dur_k_gpu],
                    )
                    dur_rev = None if active_rev is None else active_rev[dur_k_gpu]
                    dur_vals, _ = ac_model.forward_durable_value(
                        dur_world_batch,
                        batch.domain_ids[dur_k_gpu],
                        active_hidden[dur_k_gpu],
                        critic_revival_context=dur_rev,
                        validate=False,
                    )
                    finite_ok = finite_ok & torch.isfinite(dur_vals).all()
                    durable_loss = (
                        F.smooth_l1_loss(dur_vals, flat_dur_target[dur_gids_gpu], reduction="sum")
                        / denom_dur_val
                    )
                    slice_totals_gpu["durable_value_loss"] = (
                        slice_totals_gpu["durable_value_loss"] + durable_loss.detach().double()
                    )
                    if cfg.distill_value_weight:
                        batch_terms.append(0.5 * cfg.distill_value_weight * durable_loss)
                    if has_stage_base:
                        if cfg.durable_stage_baseline_weight:
                            stage_vals = ac_model.forward_durable_stage_baseline(
                                dur_world_batch,
                                critic_revival_context=dur_rev,
                            )
                        else:
                            with torch.no_grad():
                                stage_vals = ac_model.forward_durable_stage_baseline(
                                    dur_world_batch,
                                    critic_revival_context=dur_rev,
                                )
                        finite_ok = finite_ok & torch.isfinite(stage_vals).all()
                        stage_loss = (
                            F.smooth_l1_loss(
                                stage_vals, flat_stage_target[dur_gids_gpu], reduction="sum"
                            )
                            / denom_dur_val
                        )
                        slice_totals_gpu["durable_stage_baseline_loss"] = (
                            slice_totals_gpu["durable_stage_baseline_loss"]
                            + stage_loss.detach().double()
                        )
                        if cfg.durable_stage_baseline_weight:
                            batch_terms.append(0.5 * cfg.durable_stage_baseline_weight * stage_loss)

                if any_macro:
                    macro_huber = F.smooth_l1_loss(
                        output.value, flat_macro_val[gids_gpu], reduction="none"
                    )
                    m_val_loss = torch.where(is_macro_k, macro_huber, 0.0).sum() / denom_macro_steps
                    slice_totals_gpu["macro_value_loss"] = (
                        slice_totals_gpu["macro_value_loss"] + m_val_loss.detach().double()
                    )
                    if cfg.distill_value_weight:
                        batch_terms.append(0.5 * cfg.distill_value_weight * m_val_loss)

                    all_cand_worlds: list[EncodedWorldSnapshot] = []
                    all_cand_hiddens: list[Tensor] = []
                    all_cand_revs: list[Tensor] = []
                    node_specs: list[
                        tuple[int, int, Tensor, int | None, Tensor, EncodedWorldSnapshot | None, Tensor | None]
                    ] = []
                    for k, c in enumerate(active_c):
                        ref_idx, item, start, _ = chunk_slice[c]
                        index = start + offset
                        assert item.afterstate_targets is not None
                        as_t = item.afterstate_targets.get(index)
                        if (not stateless_durable) and as_t is not None and as_t.node.post_worlds:
                            s_w = len(all_cand_worlds)
                            w_cnt = len(as_t.node.post_worlds)
                            all_cand_worlds.extend(as_t.node.post_worlds)
                            all_cand_hiddens.extend([output.recurrent_state[k]] * w_cnt)
                            p_rev_ref = post_rev_per_ref[ref_idx]
                            p_rev_item = (
                                None
                                if p_rev_ref is None
                                else p_rev_ref[index].to(device=self.device, dtype=self.dtype)
                            )
                            if p_rev_item is not None:
                                all_cand_revs.extend([p_rev_item] * w_cnt)
                            step_obj = item.reference.steps[index]
                            chosen_pos: int | None = None
                            chosen_w_obj: EncodedWorldSnapshot | None = None
                            if as_t.has_chosen_post:
                                chosen_pos = as_t.node.resolved_candidate_indices.index(
                                    step_obj.action_index
                                )
                                chosen_w_obj = as_t.node.post_worlds[chosen_pos]
                            node_specs.append(
                                (
                                    s_w,
                                    s_w + w_cnt,
                                    as_t.action_values.to(device=self.device, dtype=self.dtype),
                                    chosen_pos,
                                    as_t.post_target.to(device=self.device, dtype=self.dtype),
                                    chosen_w_obj,
                                    p_rev_item,
                                )
                            )

                    if all_cand_worlds:
                        q_s_parts: list[Tensor] = []
                        n_all_w = len(all_cand_worlds)
                        for ws in range(0, n_all_w, afterstate_grad_cap):
                            self._check_resource()
                            we = min(ws + afterstate_grad_cap, n_all_w)
                            sub_w = tuple(all_cand_worlds[ws:we])
                            wb_sub, dom_sub = collate_encoded_worlds(
                                sub_w,
                                expected_config=sub_w[0].config,
                                expected_fingerprint=sub_w[0].encoding_fingerprint,
                                device=self.device,
                                validate=False,
                            )
                            sub_rev = (
                                torch.stack(all_cand_revs[ws:we], dim=0)
                                if all_cand_revs
                                else None
                            )
                            q_sub, _ = ac_model.forward_durable_value(
                                wb_sub,
                                dom_sub,
                                torch.stack(all_cand_hiddens[ws:we], dim=0),
                                critic_revival_context=sub_rev,
                                validate=False,
                            )
                            q_s_parts.append(q_sub)
                        q_s_all = torch.cat(q_s_parts, dim=0)
                        finite_ok = finite_ok & torch.isfinite(q_s_all).all()

                        node_val_losses: list[Tensor] = []
                        node_margin_losses: list[Tensor] = []
                        chosen_post_losses: list[Tensor] = []
                        chosen_stage_worlds: list[EncodedWorldSnapshot] = []
                        chosen_stage_revs: list[Tensor] = []
                        chosen_stage_targets: list[Tensor] = []
                        for (
                            s_w,
                            e_w,
                            q_t_gpu,
                            chosen_pos,
                            post_t_gpu,
                            chosen_w_obj,
                            p_rev_item,
                        ) in node_specs:
                            q_s_node = q_s_all[s_w:e_w]
                            node_val_losses.append(F.smooth_l1_loss(q_s_node, q_t_gpu))
                            if q_s_node.numel() > 1:
                                node_margin_losses.append(
                                    F.smooth_l1_loss(
                                        q_s_node - q_s_node.mean(),
                                        q_t_gpu - q_t_gpu.mean(),
                                    )
                                )
                            if chosen_pos is not None and chosen_w_obj is not None:
                                post_blend = 0.5 * q_t_gpu[chosen_pos] + 0.5 * post_t_gpu
                                chosen_post_losses.append(
                                    F.smooth_l1_loss(q_s_node[chosen_pos], post_blend)
                                )
                                if has_stage_base:
                                    chosen_stage_worlds.append(chosen_w_obj)
                                    if p_rev_item is not None:
                                        chosen_stage_revs.append(p_rev_item)
                                    chosen_stage_targets.append(post_t_gpu)

                        if node_val_losses:
                            as_val_loss = torch.stack(node_val_losses).sum() / denom_afterstate
                            slice_totals_gpu["distill_afterstate_value_loss"] = (
                                slice_totals_gpu["distill_afterstate_value_loss"]
                                + as_val_loss.detach().double()
                            )
                            if cfg.distill_value_weight:
                                batch_terms.append(0.5 * cfg.distill_value_weight * as_val_loss)
                        if node_margin_losses:
                            as_margin_loss = (
                                torch.stack(node_margin_losses).sum() / denom_afterstate
                            )
                            slice_totals_gpu["distill_afterstate_margin_loss"] = (
                                slice_totals_gpu["distill_afterstate_margin_loss"]
                                + as_margin_loss.detach().double()
                            )
                            if cfg.distill_afterstate_margin_weight:
                                batch_terms.append(
                                    0.5 * cfg.distill_afterstate_margin_weight * as_margin_loss
                                )
                        if chosen_post_losses:
                            post_loss = torch.stack(chosen_post_losses).sum() / denom_afterstate
                            slice_totals_gpu["durable_post_value_loss"] = (
                                slice_totals_gpu["durable_post_value_loss"]
                                + post_loss.detach().double()
                            )
                            if cfg.durable_post_value_weight:
                                batch_terms.append(0.5 * cfg.durable_post_value_weight * post_loss)
                        if has_stage_base and chosen_stage_worlds:
                            cw_tuple = tuple(chosen_stage_worlds)
                            cwb, _ = collate_encoded_worlds(
                                cw_tuple,
                                expected_config=cw_tuple[0].config,
                                expected_fingerprint=cw_tuple[0].encoding_fingerprint,
                                device=self.device,
                                validate=False,
                            )
                            crev = (
                                torch.stack(chosen_stage_revs, dim=0)
                                if chosen_stage_revs
                                else None
                            )
                            ctarg = torch.stack(chosen_stage_targets, dim=0)
                            if cfg.durable_stage_baseline_weight:
                                p_stage_vals = ac_model.forward_durable_stage_baseline(
                                    cwb,
                                    critic_revival_context=crev,
                                )
                            else:
                                with torch.no_grad():
                                    p_stage_vals = ac_model.forward_durable_stage_baseline(
                                        cwb,
                                        critic_revival_context=crev,
                                    )
                            finite_ok = finite_ok & torch.isfinite(p_stage_vals).all()
                            p_stage_loss = (
                                F.smooth_l1_loss(p_stage_vals, ctarg, reduction="sum")
                                / denom_afterstate
                            )
                            slice_totals_gpu["durable_stage_baseline_loss"] = (
                                slice_totals_gpu["durable_stage_baseline_loss"]
                                + p_stage_loss.detach().double()
                            )
                            if cfg.durable_stage_baseline_weight:
                                batch_terms.append(
                                    0.5 * cfg.durable_stage_baseline_weight * p_stage_loss
                                )

                if any_combat:
                    is_combat_k = ~is_macro_k
                    combat_huber = F.smooth_l1_loss(
                        output.value, flat_combat_val[gids_gpu], reduction="none"
                    )
                    c_val_loss = torch.where(is_combat_k, combat_huber, 0.0).sum() / denom_combat_steps
                    slice_totals_gpu["combat_value_loss"] = (
                        slice_totals_gpu["combat_value_loss"] + c_val_loss.detach().double()
                    )
                    if cfg.distill_value_weight:
                        batch_terms.append(0.5 * cfg.distill_value_weight * c_val_loss)

                if any_choice:
                    c_cap_ch = int(safe_logs.shape[1])
                    ch_rows_k: list[int] = []
                    ch_is_mac: list[bool] = []
                    ch_t_logs_rows: list[Tensor] = []
                    for k, c in enumerate(active_c):
                        g_id = gids_cpu[k]
                        if flat_is_choice_cpu[g_id]:
                            _, item_c, start_c, _ = chunk_slice[c]
                            step_idx_c = start_c + offset
                            assert item_c.evaluation.step_log_probs is not None
                            t_logs_cpu = item_c.evaluation.step_log_probs[step_idx_c]
                            ch_rows_k.append(k)
                            ch_is_mac.append(flat_is_macro_cpu[g_id])
                            padded_ch = torch.full((c_cap_ch,), -torch.inf, dtype=torch.float32)
                            padded_ch[: t_logs_cpu.numel()] = t_logs_cpu.float()
                            ch_t_logs_rows.append(padded_ch)
                    k_ch_gpu = torch.tensor(ch_rows_k, device=self.device, dtype=torch.long)
                    is_mac_ch_gpu = torch.tensor(ch_is_mac, device=self.device, dtype=torch.bool)
                    t_ch_gpu = torch.stack(ch_t_logs_rows, dim=0).to(
                        device=self.device, dtype=safe_logs.dtype
                    )
                    valid_ch = torch.isfinite(t_ch_gpu) & masks[k_ch_gpu]
                    safe_t_ch = t_ch_gpu.masked_fill(~valid_ch, 0.0)
                    probs_t_ch = t_ch_gpu.exp().masked_fill(~valid_ch, 0.0)
                    safe_s_ch = safe_logs[k_ch_gpu].masked_fill(~valid_ch, 0.0)
                    kl_ch_vec = (probs_t_ch * (safe_t_ch - safe_s_ch)).sum(dim=-1)

                    if any(ch_is_mac):
                        m_kl = torch.where(is_mac_ch_gpu, kl_ch_vec, 0.0).sum() / denom_macro_pol
                        m_kl_det = m_kl.detach().clamp_min(0.0).double()
                        slice_totals_gpu["distill_macro_kl"] = (
                            slice_totals_gpu["distill_macro_kl"] + m_kl_det
                        )
                        slice_totals_gpu["macro_policy_loss"] = (
                            slice_totals_gpu["macro_policy_loss"] + m_kl_det
                        )
                        macro_ent_src = (
                            hierarchical_macro_entropy(
                                safe_logs,
                                masks,
                                batch.candidates.policy_branch_ids,
                                include_branch_entropy=not cfg.macro_pg_within_branch_entropy_only,
                            )
                            if cfg.enable_macro_branch_prior
                            else all_entropies
                        )
                        m_ent = (
                            torch.where(is_macro_k & is_choice_k, macro_ent_src, 0.0)
                            .detach()
                            .sum()
                            / denom_macro_pol
                        )
                        slice_totals_gpu["macro_entropy"] = (
                            slice_totals_gpu["macro_entropy"] + m_ent.double()
                        )
                        if cfg.distill_policy_kl_weight and cfg.macro_policy_weight:
                            batch_terms.append(
                                0.5 * cfg.distill_policy_kl_weight * cfg.macro_policy_weight * m_kl
                            )

                    if any(not flag for flag in ch_is_mac):
                        c_kl = torch.where(~is_mac_ch_gpu, kl_ch_vec, 0.0).sum() / denom_combat_pol
                        c_kl_det = c_kl.detach().clamp_min(0.0).double()
                        slice_totals_gpu["distill_combat_kl"] = (
                            slice_totals_gpu["distill_combat_kl"] + c_kl_det
                        )
                        slice_totals_gpu["combat_policy_loss"] = (
                            slice_totals_gpu["combat_policy_loss"] + c_kl_det
                        )
                        c_ent = (
                            torch.where((~is_macro_k) & is_choice_k, all_entropies, 0.0)
                            .detach()
                            .sum()
                            / denom_combat_pol
                        )
                        slice_totals_gpu["combat_entropy"] = (
                            slice_totals_gpu["combat_entropy"] + c_ent.double()
                        )
                        if cfg.distill_policy_kl_weight and cfg.combat_policy_weight:
                            batch_terms.append(
                                0.5 * cfg.distill_policy_kl_weight * cfg.combat_policy_weight * c_kl
                            )

            if batch_terms:
                mega_loss = torch.stack(batch_terms).sum()
                batch_terms.clear()
                if not bool((finite_ok & torch.isfinite(mega_loss)).item()):
                    raise FloatingPointError("macro distillation chunk forward or loss is non-finite")
                mega_loss.backward()  # type: ignore[no-untyped-call]
                del mega_loss
                totals["backward_chunks"] += float(len(chunk_slice))
            elif not bool(finite_ok.item()):
                raise FloatingPointError("macro distillation chunk forward is non-finite")
            elif stateless_durable and (
                cfg.distill_value_weight
                or cfg.distill_afterstate_margin_weight
                or cfg.durable_post_value_weight
                or (has_stage_base and cfg.durable_stage_baseline_weight)
            ):
                totals["backward_chunks"] += float(len(chunk_slice))
            hidden_list.clear()
            stacked_metrics = torch.stack([slice_totals_gpu[key] for key in metric_keys]).cpu().tolist()
            for key, val in zip(metric_keys, stacked_metrics, strict=True):
                totals[key] += float(val)
            self._check_resource()

        if stateless_durable:
            dur_entries: list[tuple[EncodedWorldSnapshot, int]] = []
            node_entries: list[
                tuple[
                    int,
                    _AfterstateStepTarget,
                    int | None,
                    EncodedWorldSnapshot | None,
                    Tensor | None,
                ]
            ] = []
            for ref_idx, item in enumerate(prepared_items):
                base_g = ref_step_offsets[ref_idx]
                steps = item.reference.steps
                for idx, s in enumerate(steps):
                    if flat_has_dur_cpu[base_g + idx]:
                        dur_entries.append((cast(EncodedWorldSnapshot, s.snapshot), base_g + idx))
                assert item.afterstate_targets is not None
                p_rev_ref = post_rev_per_ref[ref_idx]
                for step_idx, as_t in item.afterstate_targets.items():
                    if not as_t.node.post_worlds:
                        continue
                    node_max_w = max(w.world.token_count for w in as_t.node.post_worlds)
                    p_rev_item = (
                        None
                        if p_rev_ref is None
                        else p_rev_ref[step_idx].to(device=self.device, dtype=self.dtype)
                    )
                    s_chosen_pos: int | None = None
                    s_chosen_w_obj: EncodedWorldSnapshot | None = None
                    if as_t.has_chosen_post:
                        s_chosen_pos = as_t.node.resolved_candidate_indices.index(
                            steps[step_idx].action_index
                        )
                        s_chosen_w_obj = as_t.node.post_worlds[s_chosen_pos]
                    node_entries.append((node_max_w, as_t, s_chosen_pos, s_chosen_w_obj, p_rev_item))

            dur_v_loss, dur_s_loss = self._backward_stateless_pre_durable(
                dur_entries=dur_entries,
                flat_dur_target=flat_dur_target,
                flat_stage_target=flat_stage_target,
                flat_step_rev_gpu=flat_step_rev_gpu,
                denom_dur_val=denom_dur_val,
                durable_value_weight=cfg.distill_value_weight,
                stage_baseline_weight=cfg.durable_stage_baseline_weight,
            )
            as_v_loss, as_m_loss, post_v_loss, post_s_loss = (
                self._backward_stateless_distill_afterstates(
                    node_entries=node_entries,
                    denom_afterstate=denom_afterstate,
                )
            )
            totals["durable_value_loss"] += dur_v_loss
            totals["distill_afterstate_value_loss"] += as_v_loss
            totals["distill_afterstate_margin_loss"] += as_m_loss
            totals["durable_post_value_loss"] += post_v_loss
            totals["durable_stage_baseline_loss"] += dur_s_loss + post_s_loss
            self._check_resource()
        return totals, maximum_chunk

    @torch.no_grad()
    def _updated_root_policy(self, prepared: _PreparedReference) -> dict[str, Any]:
        assert prepared.root is not None and prepared.projection is not None
        hidden: Tensor | None = None
        updated_logs: Tensor | None = None
        for step in prepared.root.reference.steps[:prepared.root.root_step_index + 1]:
            logits, _, hidden = self._forward(step.snapshot, hidden)
            updated_logs, _, _ = self._policy(step, logits)
        assert updated_logs is not None and prepared.evaluation.root_logs is not None
        new_logs = updated_logs.detach().cpu().double()
        old_logs = prepared.evaluation.root_logs.double()
        new_logs = new_logs - torch.logsumexp(new_logs, dim=0)
        old_logs = old_logs - torch.logsumexp(old_logs, dim=0)
        target = prepared.projection.target_probabilities.double()
        target = target / target.sum()
        positive = target > 0
        actual_kl = (new_logs.exp() * (new_logs - old_logs)).sum()
        fitting_kl = (target[positive] * (target[positive].log() - new_logs[positive])).sum()
        return {
            "updated_root_probabilities": new_logs.exp().tolist(),
            "updated_policy_kl_to_teacher": max(0.0, float(actual_kl.item())),
            "projection_kl_to_updated_policy": max(0.0, float(fitting_kl.item())),
        }

    def _clip_and_step_optimizer(self) -> tuple[float, dict[str, float]]:
        groups = {"policy_head": 0.0, "value_head": 0.0, "domain_stream": 0.0, "shared": 0.0}
        for name, parameter in self.model.named_parameters():
            if parameter.grad is not None:
                if not bool(torch.isfinite(parameter.grad).all().item()):
                    raise FloatingPointError("macro gradient is non-finite")
                if name.startswith(
                    (
                        "candidate_policy_head",
                        "candidate_role_",
                        "branch_policy_head",
                        "branch_vitals_",
                    )
                ):
                    group = "policy_head"
                elif name.startswith(
                    (
                        "state_value_head",
                        "state_revival_",
                        "durable_cycle_",
                        "durable_deck_",
                        "durable_hp_",
                        "durable_liquid_",
                        "durable_progress_vitals_",
                        "durable_shop_",
                        "durable_stage_",
                        "durable_twin_",
                        "durable_value_head",
                        "durable_vitals_",
                        "durable_revival_",
                    )
                ):
                    group = "value_head"
                elif name.startswith(("macro_stream", "combat_stream", "persistent_build_", "persistent_vitals_")):
                    group = "domain_stream"
                else:
                    group = "shared"
                groups[group] += float(parameter.grad.detach().double().square().sum().item())
        if self.config.separate_value_grad_clip:
            val_gn = (
                float(
                    torch.nn.utils.clip_grad_norm_(
                        self._value_head_parameters,
                        self.config.gradient_clip_norm,
                        error_if_nonfinite=True,
                    ).item()
                )
                if self._value_head_parameters
                else 0.0
            )
            stage_gn = (
                float(
                    torch.nn.utils.clip_grad_norm_(
                        self._stage_baseline_parameters,
                        self.config.gradient_clip_norm,
                        error_if_nonfinite=True,
                    ).item()
                )
                if self._stage_baseline_parameters
                else 0.0
            )
            non_val_gn = (
                float(
                    torch.nn.utils.clip_grad_norm_(
                        self._non_value_parameters,
                        self.config.gradient_clip_norm,
                        error_if_nonfinite=True,
                    ).item()
                )
                if self._non_value_parameters
                else 0.0
            )
            gradient_norm_val = math.hypot(val_gn, stage_gn, non_val_gn)
        else:
            gradient_norm_val = float(
                torch.nn.utils.clip_grad_norm_(
                    self._parameters,
                    self.config.gradient_clip_norm,
                    error_if_nonfinite=True,
                ).item()
            )
        self._check_resource()
        self.optimizer.step()
        self._check_resource()
        if not _finite_tree(self.model.state_dict()) or not _finite_tree(self.optimizer.state_dict()):
            raise FloatingPointError("macro optimizer step produced non-finite state")
        return gradient_norm_val, groups

    def _compute_denominators(
        self,
        references: Sequence[MacroCreditTrajectory],
        *,
        root_count: int = 0,
    ) -> dict[str, int]:
        denominators = {
            "root_count": int(root_count),
            "macro_steps": 0,
            "macro_policy_steps": 0,
            "durable_value_steps": 0,
            "afterstate_steps": 0,
            "partial_afterstate_steps": 0,
            "partial_afterstate_pg_fallback_steps": 0,
            "partial_afterstate_cond_mpo_steps": 0,
            "macro_pg_fallback_steps": 0,
            "combat_steps": 0,
            "combat_policy_steps": 0,
        }
        for reference in references:
            raw_nodes = getattr(reference, "afterstate_nodes", ())
            afterstate_indices = {
                node.step_index
                for node in raw_nodes
                if node.post_world_for_action(reference.steps[node.step_index].action_index) is not None
            }
            denominators["afterstate_steps"] += len(afterstate_indices)
            denominators["partial_afterstate_steps"] += sum(
                1
                for node in raw_nodes
                if node.is_partial
                and node.post_world_for_action(reference.steps[node.step_index].action_index) is not None
            )
            unvisited_partial = sum(
                1
                for node in raw_nodes
                if node.post_world_for_action(reference.steps[node.step_index].action_index) is None
            )
            denominators["partial_afterstate_pg_fallback_steps"] += unvisited_partial
            if self.config.enable_partial_afterstate_conditional_mpo:
                denominators["partial_afterstate_cond_mpo_steps"] += unvisited_partial
            for idx, step in enumerate(reference.steps):
                denominators[f"{step.control_domain}_steps"] += 1
                is_choice = int(step.snapshot.action_mask.sum()) > 1
                if step.control_domain == "macro":
                    denominators["durable_value_steps"] += 1
                    if is_choice:
                        denominators["macro_policy_steps"] += 1
                        if idx not in afterstate_indices:
                            denominators["macro_pg_fallback_steps"] += 1
                else:
                    if idx == 0 or reference.steps[idx - 1].control_domain == "macro":
                        denominators["durable_value_steps"] += 1
                    if is_choice:
                        denominators["combat_policy_steps"] += 1
        return denominators

    @torch.no_grad()
    def _measure_updated_durable_kl(
        self,
        prepared_items: Sequence[_PreparedReference],
        denominators: Mapping[str, int],
        *,
        packed_cache: dict[tuple[int, int], _CachedEvalGroup] | None = None,
    ) -> tuple[float, float, int]:
        nat_items = [it for it in prepared_items if it.root is None]
        if not nat_items:
            return 0.0, 0.0, 0
        refs = [it.reference for it in nat_items]
        want_combat = denominators["combat_policy_steps"] > 0 and any(
            it.evaluation.step_log_probs is not None
            and any(
                it.reference.steps[s_i].control_domain != "macro"
                for s_i in it.evaluation.step_log_probs
            )
            for it in nat_items
        )
        if self.config.eval_batch_size > 1 and len(refs) > 1:
            post_evals = self._evaluate_durable_batch(
                refs,
                check_likelihood=False,
                record_all_choice_logs=want_combat,
                compute_durable_value=False,
                packed_cache=packed_cache,
            )
        else:
            post_evals = [
                self._evaluate(
                    ref,
                    root_index=-1,
                    check_likelihood=False,
                    record_all_choice_logs=want_combat,
                    compute_durable_value=False,
                )
                for ref in refs
            ]
        calls = sum(ev.forward_calls for ev in post_evals)
        macro_kl_sum = 0.0
        macro_kl_count = 0
        combat_kl_sum = 0.0
        combat_kl_count = 0
        for it, post_ev in zip(nat_items, post_evals, strict=True):
            old_map = it.evaluation.step_log_probs or {}
            new_map = post_ev.step_log_probs or {}
            for s_idx, old_logs in old_map.items():
                new_logs = new_map.get(s_idx)
                if new_logs is None:
                    continue
                n_c = min(int(old_logs.numel()), int(new_logs.numel()))
                o_l = old_logs[:n_c].double()
                n_l = new_logs[:n_c].double()
                valid = torch.isfinite(o_l) & torch.isfinite(n_l)
                if not bool(valid.any().item()):
                    continue
                p_o = o_l.exp().masked_fill(~valid, 0.0)
                kl_val = max(
                    0.0,
                    float(
                        (
                            p_o
                            * (o_l.masked_fill(~valid, 0.0) - n_l.masked_fill(~valid, 0.0))
                        )
                        .sum()
                        .item()
                    ),
                )
                if it.reference.steps[s_idx].control_domain == "macro":
                    macro_kl_sum += kl_val
                    macro_kl_count += 1
                else:
                    combat_kl_sum += kl_val
                    combat_kl_count += 1
        return (
            macro_kl_sum / max(1, macro_kl_count),
            combat_kl_sum / max(1, combat_kl_count),
            calls,
        )

    def _sample_twin_ref_weights(
        self,
        count: int,
        *,
        epoch_idx: int,
        mb_idx: int,
    ) -> tuple[Tensor, Tensor] | None:
        if (
            count <= 0
            or not isinstance(self.model, HybridDomainActorCriticModel)
            or not self.model.actor_critic_config.enable_twin_durable_value_heads
        ):
            return None
        gen = torch.Generator(device="cpu")
        gen.manual_seed(20261117 + self.update * 997 + epoch_idx * 37 + mb_idx)
        bits = torch.randint(0, 2, (2, count), generator=gen, dtype=torch.long)
        weights = (0.35 + 1.30 * bits.to(dtype=self.dtype)).to(device=self.device)
        return weights[0], weights[1]

    def learn(self, cohort: MacroCreditCohort, *, resource_guard: Callable[[], None] | None = None) -> dict[str, Any]:
        if self._failed:
            raise RuntimeError("failed macro learner must reload its last committed checkpoint")
        self._validate_update()
        if not isinstance(cohort, MacroCreditCohort) or cohort.teacher_version != self.update:
            raise ValueError("macro cohort must match the learner's exact fresh teacher version")
        in_stage_a = self.update < self.config.distill_teacher_rollout_updates
        in_stage_b = (
            self.config.distill_teacher_rollout_updates
            <= self.update
            < self.config.distill_total_updates
        )
        if (in_stage_a or in_stage_b) and self.teacher_model is None:
            raise RuntimeError(
                f"update {self.update} < distill_total_updates={self.config.distill_total_updates} "
                "requires an attached frozen teacher_model"
            )
        if (
            not in_stage_a
            and not in_stage_b
            and self.config.kickstart_kl_floor > 0.0
            and self.config.distill_total_updates > 0
            and self.teacher_model is None
        ):
            raise RuntimeError(
                f"kickstart_kl_floor={self.config.kickstart_kl_floor} requires an attached frozen teacher_model"
            )
        distillation_stage = (
            "teacher_rollout_distill"
            if in_stage_a
            else ("on_policy_kickstart" if in_stage_b else "pure_rl")
        )
        warmup_trunk_frozen = (
            (not in_stage_a)
            and self.config.target_mode == "durable_afterstate_v2"
            and self.config.freeze_trunk_during_durable_warmup
            and (self.update < self.config.durable_warmup_updates)
        )
        if warmup_trunk_frozen:
            effective_kickstart_kl_weight = 0.0
        elif in_stage_b:
            ks_span = max(
                1,
                self.config.distill_total_updates - self.config.distill_teacher_rollout_updates,
            )
            ks_prog = (self.update - self.config.distill_teacher_rollout_updates) / ks_span
            effective_kickstart_kl_weight = self.config.kickstart_kl_floor + (
                (self.config.kickstart_kl_weight - self.config.kickstart_kl_floor)
                * 0.5
                * (1.0 + math.cos(math.pi * ks_prog))
            )
        elif (
            not in_stage_a
            and self.config.kickstart_kl_floor > 0.0
            and self.teacher_model is not None
        ):
            effective_kickstart_kl_weight = self.config.kickstart_kl_floor
        else:
            effective_kickstart_kl_weight = 0.0

        self._resource_guard = resource_guard
        self._resource_checks = 0
        self._check_resource()
        cohort.__post_init__()
        for root in cohort.roots:
            root.__post_init__()
            for trajectory in (root.reference, *root.candidates):
                self._check_resource()
                trajectory.__post_init__()
        for reference in cohort.natural_only:
            reference.__post_init__()
        if self.config.target_mode == "durable_afterstate_v2" and cohort.roots:
            raise ValueError(
                "durable_afterstate_v2 learns across full multi-node episodes in natural_only, without single-root counterfactuals"
            )
        if not _finite_tree(self.model.state_dict()) or not _finite_tree(self.optimizer.state_dict()):
            raise FloatingPointError("macro model/optimizer state is non-finite before update")
        self.model.eval()
        self.optimizer.zero_grad(set_to_none=True)
        started = time.monotonic()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        denominators = self._compute_denominators(cohort.references, root_count=len(cohort.roots))
        prepared: list[_PreparedReference] = []
        target_calls = 0
        maximum_error = 0.0
        kickstart_teacher_logs: list[dict[int, Tensor]] | None = None
        eval_packed_cache: dict[tuple[int, int], _CachedEvalGroup] = {}
        try:
            if in_stage_a:
                assert self.teacher_model is not None
                if self.config.eval_batch_size > 1 and len(cohort.natural_only) > 1:
                    batch_items, batch_calls, batch_error = self._prepare_durable_natural_batch(
                        cohort.natural_only,
                        model_override=self.teacher_model,
                        check_likelihood=True,
                        record_all_choice_logs=True,
                        packed_cache=eval_packed_cache,
                    )
                    prepared.extend(batch_items)
                    target_calls += batch_calls
                    maximum_error = max(maximum_error, batch_error)
                else:
                    for reference in cohort.natural_only:
                        item, calls, error = self._prepare_durable_natural(
                            reference,
                            model_override=self.teacher_model,
                            check_likelihood=True,
                            record_all_choice_logs=True,
                        )
                        prepared.append(item)
                        target_calls += calls
                        maximum_error = max(maximum_error, error)
            else:
                for root in cohort.roots:
                    item, calls, error = self._prepare(root)
                    prepared.append(item)
                    target_calls += calls
                    maximum_error = max(maximum_error, error)
                if (
                    self.config.target_mode == "durable_afterstate_v2"
                    and self.config.eval_batch_size > 1
                    and len(cohort.natural_only) > 1
                ):
                    batch_items, batch_calls, batch_error = self._prepare_durable_natural_batch(
                        cohort.natural_only,
                        record_all_choice_logs=self._need_all_choice_logs(),
                        packed_cache=eval_packed_cache,
                    )
                    prepared.extend(batch_items)
                    target_calls += batch_calls
                    maximum_error = max(maximum_error, batch_error)
                else:
                    for reference in cohort.natural_only:
                        item, calls, error = self._prepare_natural(reference)
                        prepared.append(item)
                        target_calls += calls
                        maximum_error = max(maximum_error, error)
                if (
                    effective_kickstart_kl_weight > 0.0
                    and self.teacher_model is not None
                    and cohort.natural_only
                ):
                    assert self.teacher_model is not None
                    if self.config.eval_batch_size > 1 and len(cohort.natural_only) > 1:
                        t_evals = self._evaluate_durable_batch(
                            cohort.natural_only,
                            model_override=self.teacher_model,
                            check_likelihood=False,
                            record_all_choice_logs=True,
                            compute_durable_value=False,
                            packed_cache=eval_packed_cache,
                        )
                    else:
                        t_evals = [
                            self._evaluate(
                                ref,
                                root_index=-1,
                                model_override=self.teacher_model,
                                check_likelihood=False,
                                record_all_choice_logs=True,
                                compute_durable_value=False,
                            )
                            for ref in cohort.natural_only
                        ]
                    target_calls += sum(ev.forward_calls for ev in t_evals)
                    kickstart_teacher_logs = [
                        cast(dict[int, Tensor], ev.step_log_probs) for ev in t_evals
                    ]
        except BaseException:
            eval_packed_cache.clear()
            raise
        before = [parameter.detach().clone() for parameter in self._parameters]
        totals = {
            key: 0.0
            for key in (
                "macro_policy_loss",
                "macro_mpo_policy_loss",
                "macro_mpo_within_policy_loss",
                "macro_pg_policy_loss",
                "macro_entropy",
                "macro_value_loss",
                "durable_value_loss",
                "durable_post_value_loss",
                "durable_stage_baseline_loss",
                "combat_policy_loss",
                "combat_value_loss",
                "combat_entropy",
                "distill_macro_kl",
                "distill_combat_kl",
                "distill_afterstate_margin_loss",
                "distill_afterstate_value_loss",
                "observed_macro_step_kl",
                "observed_combat_step_kl",
                "backward_chunks",
            )
        }
        maximum_chunk = 0
        distill_epochs_executed = (
            self.config.distill_epochs_per_cohort if in_stage_a else 1
        )
        ppo_optimizer_steps_executed = 0
        ppo_kl_early_stopped = False
        executed_forward_steps = 0
        post_macro_kl = 0.0
        post_combat_kl = 0.0
        raw_post_macro_kl = 0.0
        line_search_scale = 1.0
        line_search_backtracks = 0
        saved_req_grads: list[bool] | None = None
        # This remains set through ALL post-step validation/CPU copies/metrics,
        # including BaseException interruptions. Only complete success clears it.
        self._failed = True
        try:
            if warmup_trunk_frozen:
                saved_req_grads = [p.requires_grad for p in self._parameters]
                for p, is_pure_val in zip(self._parameters, self._is_pure_value_param, strict=True):
                    p.requires_grad_(is_pure_val)
            if in_stage_a:
                total_bw_chunks = 0.0
                gradient_norm_val = 0.0
                groups: dict[str, float] = {
                    "policy_head": 0.0,
                    "value_head": 0.0,
                    "domain_stream": 0.0,
                    "shared": 0.0,
                }
                for ep_idx in range(distill_epochs_executed):
                    if self.config.eval_batch_size > 1 and len(cohort.natural_only) > 1:
                        s_evals = self._evaluate_durable_batch(
                            cohort.natural_only,
                            check_likelihood=False,
                            record_all_choice_logs=False,
                            compute_durable_value=False,
                            packed_cache=eval_packed_cache,
                        )
                    else:
                        s_evals = [
                            self._evaluate(
                                ref,
                                root_index=-1,
                                check_likelihood=False,
                                record_all_choice_logs=False,
                                compute_durable_value=False,
                            )
                            for ref in cohort.natural_only
                        ]
                    target_calls += sum(ev.forward_calls for ev in s_evals)
                    student_boundaries = [ev.boundary_hidden for ev in s_evals]
                    self.optimizer.zero_grad(set_to_none=True)
                    ep_totals, ep_max_chunk = self._distill_epoch_durable(
                        prepared,
                        student_boundaries,
                        denominators,
                    )
                    maximum_chunk = max(maximum_chunk, ep_max_chunk)
                    total_bw_chunks += ep_totals["backward_chunks"]
                    gradient_norm_val, groups = self._clip_and_step_optimizer()
                    ppo_optimizer_steps_executed += 1
                    executed_forward_steps += (
                        denominators["macro_steps"]
                        + denominators["combat_steps"]
                        + denominators["afterstate_steps"]
                    )
                    if ep_idx == distill_epochs_executed - 1:
                        for key, value in ep_totals.items():
                            totals[key] = value
                totals["backward_chunks"] = total_bw_chunks
            else:
                nat_offset = len(cohort.roots)
                multi_step_rl = (
                    self.config.target_mode == "durable_afterstate_v2"
                    and len(prepared) > 0
                    and (
                        self.config.ppo_epochs_per_cohort > 1
                        or self.config.ppo_minibatches_per_cohort > 1
                    )
                )
                if multi_step_rl:
                    n_ep = self.config.ppo_epochs_per_cohort
                    n_mb = min(len(prepared), self.config.ppo_minibatches_per_cohort)
                    mb_rng = random.Random(20261117 + self.update * 1009)
                    accum_totals = {key: 0.0 for key in totals}
                    gradient_norm_val = 0.0
                    groups = {
                        "policy_head": 0.0,
                        "value_head": 0.0,
                        "domain_stream": 0.0,
                        "shared": 0.0,
                    }
                    for ep_idx in range(n_ep):
                        order = list(range(len(prepared)))
                        if n_mb > 1 or ep_idx > 0:
                            mb_rng.shuffle(order)
                        mb_slices = [order[m::n_mb] for m in range(n_mb) if order[m::n_mb]]
                        for mb_idx, idx_list in enumerate(mb_slices):
                            mb_items = [prepared[i] for i in idx_list]
                            mb_denoms = self._compute_denominators(
                                [it.reference for it in mb_items],
                                root_count=sum(it.root is not None for it in mb_items),
                            )
                            mb_ks_logs = (
                                [
                                    kickstart_teacher_logs[i - nat_offset]
                                    if (it.root is None and i >= nat_offset)
                                    else {}
                                    for i, it in zip(idx_list, mb_items, strict=True)
                                ]
                                if kickstart_teacher_logs is not None
                                else None
                            )
                            twin_w = self._sample_twin_ref_weights(
                                len(mb_items), epoch_idx=ep_idx, mb_idx=mb_idx
                            )
                            self.optimizer.zero_grad(set_to_none=True)
                            if self.config.chunk_batch_size > 1:
                                mb_totals, mb_max_chunk = self._backward_chunks_durable_batch(
                                    mb_items,
                                    mb_denoms,
                                    kickstart_teacher_logs=mb_ks_logs,
                                    kickstart_kl_weight=effective_kickstart_kl_weight,
                                    twin_ref_weights=twin_w,
                                )
                            else:
                                mb_totals = {key: 0.0 for key in totals}
                                mb_max_chunk = 0
                                for loc_i, it in enumerate(mb_items):
                                    it_ks = mb_ks_logs[loc_i] if mb_ks_logs is not None else None
                                    for start in range(0, len(it.reference.steps), self.config.chunk_steps):
                                        stop = min(start + self.config.chunk_steps, len(it.reference.steps))
                                        out_c = self._backward_chunk(
                                            it,
                                            start,
                                            stop,
                                            mb_denoms,
                                            teacher_step_logs=it_ks,
                                            kickstart_kl_weight=effective_kickstart_kl_weight,
                                        )
                                        for k_c, v_c in out_c.items():
                                            mb_totals[k_c] += v_c
                                        mb_max_chunk = max(mb_max_chunk, stop - start)
                            maximum_chunk = max(maximum_chunk, mb_max_chunk)
                            executed_forward_steps += (
                                mb_denoms["macro_steps"]
                                + mb_denoms["combat_steps"]
                                + mb_denoms["afterstate_steps"]
                            )
                            if (
                                ppo_optimizer_steps_executed > 0
                                and self.config.target_macro_step_kl > 0.0
                                and mb_totals["observed_macro_step_kl"] > self.config.target_macro_step_kl
                            ):
                                self.optimizer.zero_grad(set_to_none=True)
                                ppo_kl_early_stopped = True
                                break
                            gradient_norm_val, groups = self._clip_and_step_optimizer()
                            ppo_optimizer_steps_executed += 1
                            for k_t, v_t in mb_totals.items():
                                accum_totals[k_t] += v_t
                        if ppo_kl_early_stopped:
                            break
                    steps_div = max(1, ppo_optimizer_steps_executed)
                    for k_t, v_t in accum_totals.items():
                        totals[k_t] = v_t if k_t == "backward_chunks" else (v_t / steps_div)
                else:
                    twin_w_single = self._sample_twin_ref_weights(
                        len(prepared), epoch_idx=0, mb_idx=0
                    )
                    if (
                        self.config.target_mode == "durable_afterstate_v2"
                        and self.config.chunk_batch_size > 1
                        and prepared
                    ):
                        batch_totals, maximum_chunk = self._backward_chunks_durable_batch(
                            prepared,
                            denominators,
                            kickstart_teacher_logs=kickstart_teacher_logs,
                            kickstart_kl_weight=effective_kickstart_kl_weight,
                            twin_ref_weights=twin_w_single,
                        )
                        for key, value in batch_totals.items():
                            totals[key] += value
                    else:
                        for item_idx, item in enumerate(prepared):
                            item_ks_logs = (
                                kickstart_teacher_logs[item_idx - nat_offset]
                                if (
                                    kickstart_teacher_logs is not None
                                    and item.root is None
                                    and item_idx >= nat_offset
                                )
                                else None
                            )
                            for start in range(0, len(item.reference.steps), self.config.chunk_steps):
                                stop = min(start + self.config.chunk_steps, len(item.reference.steps))
                                output = self._backward_chunk(
                                    item,
                                    start,
                                    stop,
                                    denominators,
                                    teacher_step_logs=item_ks_logs,
                                    kickstart_kl_weight=effective_kickstart_kl_weight,
                                )
                                for key, value in output.items():
                                    totals[key] += value
                                maximum_chunk = max(maximum_chunk, stop - start)
                    gradient_norm_val, groups = self._clip_and_step_optimizer()
                    ppo_optimizer_steps_executed = 1
                    executed_forward_steps = (
                        denominators["macro_steps"]
                        + denominators["combat_steps"]
                        + (denominators["afterstate_steps"] if self.config.target_mode == "durable_afterstate_v2" else 0)
                    )
                if (
                    self.config.target_mode == "durable_afterstate_v2"
                    and prepared
                    and not warmup_trunk_frozen
                ):
                    post_macro_kl, post_combat_kl, post_kl_calls = self._measure_updated_durable_kl(
                        prepared,
                        denominators,
                        packed_cache=eval_packed_cache,
                    )
                    target_calls += post_kl_calls
                    raw_post_macro_kl = post_macro_kl
                    if (
                        self.config.enable_post_step_kl_line_search
                        and self.update >= self.config.durable_warmup_updates
                        and self.config.target_macro_step_kl > 0.0
                        and post_macro_kl > self.config.target_macro_step_kl
                        and self.config.line_search_max_backtracks > 0
                    ):
                        policy_trunk_pairs = [
                            (p, b, (p.detach() - b).clone())
                            for p, b, is_pure_val in zip(
                                self._parameters,
                                before,
                                self._is_pure_value_param,
                                strict=True,
                            )
                            if not is_pure_val
                        ]
                        for _ in range(self.config.line_search_max_backtracks):
                            ratio = min(
                                0.75,
                                0.90
                                * math.sqrt(
                                    self.config.target_macro_step_kl / max(post_macro_kl, 1e-12)
                                ),
                            )
                            next_scale = max(
                                self.config.line_search_min_scale,
                                line_search_scale * ratio,
                            )
                            if next_scale >= line_search_scale - 1e-6:
                                break
                            line_search_scale = next_scale
                            line_search_backtracks += 1
                            with torch.no_grad():
                                for p, b, d in policy_trunk_pairs:
                                    p.copy_(b + d * line_search_scale)
                            post_macro_kl, post_combat_kl, ls_calls = self._measure_updated_durable_kl(
                                prepared,
                                denominators,
                                packed_cache=eval_packed_cache,
                            )
                            target_calls += ls_calls
                            if (
                                post_macro_kl <= self.config.target_macro_step_kl
                                or line_search_scale <= self.config.line_search_min_scale + 1e-9
                            ):
                                break
        except BaseException:
            self.optimizer.zero_grad(set_to_none=True)
            self._failed = True
            raise
        finally:
            if saved_req_grads is not None:
                for p, req in zip(self._parameters, saved_req_grads, strict=True):
                    p.requires_grad_(req)
            eval_packed_cache.clear()
        delta_squared = sum(
            float((parameter.detach().double() - previous.double()).square().sum().item())
            for parameter, previous in zip(self._parameters, before, strict=True)
        )
        prior_update = self.update
        self.update += 1
        cfg = self.config
        is_durable_v2 = cfg.target_mode == "durable_afterstate_v2"
        durable_actor_active = (
            in_stage_a
            or (not is_durable_v2)
            or ((not warmup_trunk_frozen) and prior_update >= cfg.durable_warmup_updates)
        )
        combat_actor_active = (
            in_stage_a
            or (not is_durable_v2)
            or ((not warmup_trunk_frozen) and prior_update >= cfg.combat_actor_warmup_updates)
        )
        if in_stage_a:
            macro_loss = (
                cfg.distill_policy_kl_weight * cfg.macro_policy_weight * totals["macro_policy_loss"]
                + cfg.distill_value_weight
                * (
                    totals["macro_value_loss"]
                    + totals["durable_value_loss"]
                    + totals["distill_afterstate_value_loss"]
                )
                + cfg.distill_afterstate_margin_weight * totals["distill_afterstate_margin_loss"]
                + cfg.durable_post_value_weight * totals["durable_post_value_loss"]
                + cfg.durable_stage_baseline_weight * totals["durable_stage_baseline_loss"]
            )
            combat_loss = (
                cfg.distill_policy_kl_weight * cfg.combat_policy_weight * totals["combat_policy_loss"]
                + cfg.distill_value_weight * totals["combat_value_loss"]
            )
        else:
            macro_loss = (
                cfg.macro_policy_weight * totals["macro_policy_loss"]
                + cfg.macro_value_weight * (totals["macro_value_loss"] + totals["durable_value_loss"])
                + cfg.durable_post_value_weight * totals["durable_post_value_loss"]
                + cfg.durable_stage_baseline_weight * totals["durable_stage_baseline_loss"]
                + effective_kickstart_kl_weight * totals["distill_macro_kl"]
                - (cfg.macro_entropy_weight * totals["macro_entropy"] if is_durable_v2 else 0.0)
            )
            combat_loss = (
                (cfg.combat_policy_weight * totals["combat_policy_loss"] if combat_actor_active else 0.0)
                + cfg.combat_value_weight * totals["combat_value_loss"]
                + effective_kickstart_kl_weight * totals["distill_combat_kl"]
                - (cfg.combat_entropy_weight * totals["combat_entropy"] if combat_actor_active else 0.0)
            )
        root_targets = [{
            "root_id": item.root.root_id, "action_values": item.action_values.tolist(),
            "teacher_probabilities": item.projection.teacher_probabilities.tolist(),
            "target_probabilities": item.projection.target_probabilities.tolist(),
            "value_target": float(item.projection.value_target.item()), "temperature": item.projection.temperature,
            "kl": item.projection.kl, "expected_improvement": item.projection.expected_improvement,
            "actor_enabled": item.projection.actor_enabled,
            **self._updated_root_policy(item),
        } for item in prepared if item.root is not None and item.projection is not None]
        afterstate_projections = [
            as_item.projection
            for item in prepared
            if item.afterstate_targets is not None
            for as_item in item.afterstate_targets.values()
        ]
        twin_disagreements = [
            float(as_item.action_disagreement.mean().item())
            for item in prepared
            if item.afterstate_targets is not None
            for as_item in item.afterstate_targets.values()
            if as_item.action_disagreement is not None and as_item.action_disagreement.numel() > 0
        ]
        updated_policy_kl_val = (
            sum(row["updated_policy_kl_to_teacher"] for row in root_targets) / max(1, len(root_targets))
            if root_targets
            else post_macro_kl
        )
        self.metrics = {
            "version": MACRO_CREDIT_LEARNER_VERSION, "update": self.update, "teacher_version": cohort.teacher_version,
            "policy_lag": 0, "cohort_id": cohort.cohort_id, "target_mode": cfg.target_mode,
            "distillation_stage": distillation_stage,
            "distill_epochs_executed": distill_epochs_executed,
            "ppo_optimizer_steps_executed": ppo_optimizer_steps_executed,
            "ppo_kl_early_stopped": ppo_kl_early_stopped,
            "warmup_trunk_frozen": warmup_trunk_frozen,
            "raw_updated_macro_policy_kl": raw_post_macro_kl,
            "line_search_scale": line_search_scale,
            "line_search_backtracks": line_search_backtracks,
            "effective_kickstart_kl_weight": effective_kickstart_kl_weight,
            "durable_actor_active": durable_actor_active,
            "combat_actor_active": combat_actor_active,
            "effective_macro_kl": (0.0 if in_stage_a else (cfg.macro_kl if durable_actor_active else 0.0)),
            "effective_combat_policy_weight": cfg.combat_policy_weight if combat_actor_active else 0.0,
            **denominators, **totals,
            "macro_domain_weight": 0.5, "combat_domain_weight": 0.5,
            "macro_loss": macro_loss, "combat_loss": combat_loss, "loss": 0.5 * (macro_loss + combat_loss),
            "macro_policy_roots": sum(item.projection.actor_enabled for item in prepared if item.projection is not None),
            "enabled_afterstate_projections": sum(proj.actor_enabled for proj in afterstate_projections),
            "mean_afterstate_projection_kl": (
                sum(proj.kl for proj in afterstate_projections) / max(1, len(afterstate_projections))
            ),
            "mean_afterstate_expected_improvement": (
                sum(proj.expected_improvement for proj in afterstate_projections) / max(1, len(afterstate_projections))
            ),
            "mean_afterstate_twin_disagreement": (
                sum(twin_disagreements) / max(1, len(twin_disagreements))
            ),
            "mean_updated_macro_policy_kl": post_macro_kl,
            "mean_updated_combat_policy_kl": post_combat_kl,
            "natural_only_games": len(cohort.natural_only),
            "root_targets": root_targets, "mean_root_kl": sum(item.projection.kl for item in prepared if item.projection is not None) / max(1, len(root_targets)),
            "kl_semantics": "mean_root_kl is projection q||teacher; updated_policy_kl_to_teacher is measured after fitting",
            "mean_updated_policy_kl_to_teacher": updated_policy_kl_val,
            "gradient_norm": gradient_norm_val,
            **{f"{name}_gradient_norm": math.sqrt(value) for name, value in groups.items()},
            "parameter_delta_norm": math.sqrt(delta_squared),
            "target_forward_calls": target_calls,
            "training_forward_calls": executed_forward_steps,
            "max_autograd_steps": maximum_chunk, "maximum_likelihood_error": maximum_error,
            "cohort_storage_bytes": cohort.storage_nbytes(), "elapsed_seconds": time.monotonic() - started,
            "resource_checks": self._resource_checks,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(self.device) if self.device.type == "cuda" else 0,
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(self.device) if self.device.type == "cuda" else 0,
            "process_peak_rss_bytes": _rss_peak(),
        }
        if not _finite_tree(self.metrics):
            raise FloatingPointError("macro update metrics are non-finite")
        self._check_resource()
        self.metrics["resource_checks"] = self._resource_checks
        self._failed = False
        self._resource_guard = None
        return self.metrics.copy()

    def as_mapping(self) -> dict[str, Any]:
        return self.metrics.copy()


__all__ = ["MACRO_CREDIT_LEARNER_VERSION", "MacroCreditForward", "MacroCreditLearner", "MacroCreditLearnerConfig"]
