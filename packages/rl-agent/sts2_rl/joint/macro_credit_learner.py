"""One fresh-cohort update with long macro credit and bounded recurrent graphs.

Only natural references train ordinary state occupancy. Counterfactual suffixes
produce conditional-action labels, never extra ordinary policy/value examples.
All parameters stay fixed through no-grad target reconstruction and chunk-wise
gradient accumulation; one AdamW step closes the cohort. No replay, target Q,
saved behavior hidden state, scheduler or EMA state is owned by this learner.
"""

from __future__ import annotations

import math
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
from sts2_rl.encoding.snapshot import collate_encoded_snapshots, collate_encoded_worlds
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
from .macro_targets import MacroPolicyProjection, project_macro_policy
from .shadow_price import evaluate_combat_value_from_snapshot
from .transitions import HybridVTraceStep

MACRO_CREDIT_LEARNER_VERSION: Final = "sts2-fresh-macro-credit-learner-v1"
MacroCreditForward: TypeAlias = Callable[[EncodedDecisionSnapshot, Tensor | None], tuple[Tensor, Tensor, Tensor]]


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
    revival_budget: int = 0
    zero_macro_pace_penalty: bool = False
    clip_macro_pg_fallback: bool = False
    macro_pg_clip_eps: float = 0.20
    macro_pg_adv_clip: float = 0.25
    macro_pg_kl_weight: float = 0.05
    macro_pg_within_branch_entropy_only: bool = False

    def __post_init__(self) -> None:
        for bool_name in (
            "allow_tf32",
            "separate_value_grad_clip",
            "enable_direct_vitals_context",
            "enable_macro_branch_prior",
            "enable_persistent_only_durable_value",
            "enable_asymmetric_revival_critic",
            "enable_draw_weighted_durable_value",
            "enable_role_conditioned_candidate_policy",
            "zero_macro_pace_penalty",
            "clip_macro_pg_fallback",
            "macro_pg_within_branch_entropy_only",
        ):
            if not isinstance(getattr(self, bool_name), bool):
                raise ValueError(f"macro credit {bool_name} must be a boolean")
        if self.enable_draw_weighted_durable_value and not self.enable_persistent_only_durable_value:
            raise ValueError(
                "macro credit enable_draw_weighted_durable_value requires enable_persistent_only_durable_value=True"
            )
        if self.target_mode not in {"counterfactual_rollout", "combat_shadow_price", "durable_afterstate_v2"}:
            raise ValueError(
                "macro credit target_mode must be 'counterfactual_rollout', "
                f"'combat_shadow_price', or 'durable_afterstate_v2', got {self.target_mode}"
            )
        if self.trainable_scope not in {"all", "macro_only"}:
            raise ValueError(f"macro credit trainable_scope must be 'all' or 'macro_only', got {self.trainable_scope}")
        for name in ("chunk_steps", "combat_unroll_steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 16:
                raise ValueError(f"macro credit {name} must be an integer in [1, 16]")
        for name in ("eval_batch_size", "chunk_batch_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"macro credit {name} must be a positive integer")
        for name in ("durable_warmup_updates", "combat_actor_warmup_updates", "revival_budget"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"macro credit {name} must be a non-negative integer")
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
        )
        non_negative = (
            "weight_decay",
            "macro_kl",
            "macro_policy_weight",
            "macro_value_weight",
            "combat_policy_weight",
            "combat_value_weight",
            "combat_entropy_weight",
            "macro_entropy_weight",
            "durable_post_value_weight",
            "macro_pg_kl_weight",
        )
        for name in (*positive, *non_negative, "macro_trace_lambda"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0 or (name in positive and value == 0):
                raise ValueError(f"macro credit {name} has an invalid finite range")
        if self.macro_trace_lambda > 1 or self.c_clip > self.rho_clip:
            raise ValueError("macro lambda must be <= 1 and c_clip <= rho_clip")


@dataclass(slots=True)
class _AfterstateStepTarget:
    node: MacroAfterstateNode
    action_values: Tensor
    projection: MacroPolicyProjection
    post_target: Tensor


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
    ) -> None:
        self.model = model
        self.config = config or MacroCreditLearnerConfig()
        self.update = update
        self._validate_update()
        self._forward_override = forward
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
                "durable_deck_",
                "durable_revival_",
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
        value_prefixes = (
            "state_value_head",
            "state_revival_",
            "durable_deck_",
            "durable_value_head",
            "durable_vitals_",
            "durable_revival_",
        )
        self._value_head_parameters = [
            p for name, p in model.named_parameters() if p.requires_grad and name.startswith(value_prefixes)
        ]
        self._non_value_parameters = [
            p for name, p in model.named_parameters() if p.requires_grad and not name.startswith(value_prefixes)
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
            "durable_deck_",
            "durable_revival_",
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
            (local if name.startswith(local_tokens) else shared).append(item)
        groups: list[dict[str, Any]] = []
        if local:
            groups.append({"params": local, "lr": self.config.learning_rate})
        if shared:
            groups.append({"params": shared, "lr": self.config.learning_rate * self.config.shared_learning_rate_scale})
        self.optimizer = torch.optim.AdamW(groups, weight_decay=self.config.weight_decay)
        self.model.eval()
        self.metrics: dict[str, Any] = {"version": MACRO_CREDIT_LEARNER_VERSION, "update": self.update}
        self._failed = False
        self._resource_guard: Callable[[], None] | None = None
        self._resource_checks = 0

    @property
    def policy_version(self) -> int:
        return self.update

    def _validate_update(self) -> None:
        if isinstance(self.update, bool) or not isinstance(self.update, int) or self.update < 0:
            raise ValueError("macro learner update must be a non-negative integer")

    def _use_revival_ctx(self) -> bool:
        return bool(
            self.config.enable_asymmetric_revival_critic
            and isinstance(self.model, HybridDomainActorCriticModel)
            and self.model.actor_critic_config.enable_asymmetric_revival_critic
        )

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
    ) -> tuple[Tensor, Tensor, Tensor | None, Tensor]:
        self._check_resource()
        durable_value: Tensor | None = None
        if self._forward_override is not None:
            logits, value, next_hidden = self._forward_override(snapshot, hidden)
        else:
            batch = collate_encoded_snapshots(
                (snapshot,), expected_config=snapshot.config,
                expected_fingerprint=snapshot.encoding_fingerprint, device=self.device,
            )
            if self._use_revival_ctx():
                output = cast(
                    HybridActorCriticOutput,
                    cast(HybridDomainActorCriticModel, self.model)(
                        batch,
                        hidden,
                        critic_revival_context=critic_revival_context,
                        validate=False,
                    ),
                )
            else:
                output = cast(HybridActorCriticOutput, self.model(batch, hidden, validate=False))
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
    ) -> Tensor:
        self._check_resource()
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
        durable_vals, _ = cast(HybridDomainActorCriticModel, self.model).forward_durable_value(
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
    def _evaluate(self, trajectory: MacroCreditTrajectory, *, root_index: int | None = None) -> _Evaluation:
        _configure_cuda_tf32(self.device, self.config.allow_tf32)
        hidden: Tensor | None = None
        calls = 0
        is_durable_v2 = self.config.target_mode == "durable_afterstate_v2"
        afterstate_indices = {node.step_index for node in trajectory.afterstate_nodes} if is_durable_v2 else set()
        use_rev = self._use_revival_ctx()
        step_rev_cpu: Tensor | None = None
        boot_rev_cpu: Tensor | None = None
        if use_rev:
            step_rev_cpu, _, boot_rev_cpu = self._trajectory_critic_revival_contexts(trajectory)
        for snapshot in trajectory.prefix:
            _, _, _, hidden = self._forward_full(snapshot, hidden)
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
            )
            hidden = hidden.detach()
            policy, _, _ = self._policy(step, logits)
            selected = policy[step.action_index]
            if not (index == 0 and trajectory.root_intervened):
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
                if durable_val is None:
                    raise ValueError("durable_afterstate_v2 requires durable_value on every step")
                durable_values.append(durable_val.detach().cpu().clone())
                if index in afterstate_indices:
                    step_log_probs[index] = policy.detach().cpu().clone()
                    post_step_hidden[index] = hidden.detach().cpu().clone()
                elif step.control_domain == "macro" and int(step.snapshot.action_mask.sum()) > 1:
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
            )
            bootstrap = bootstrap_val.detach().cpu().clone()
            if is_durable_v2:
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
            durable_values=torch.stack(durable_values) if is_durable_v2 else None,
            durable_bootstrap_value=durable_bootstrap,
            step_log_probs=step_log_probs if is_durable_v2 else None,
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
    def _evaluate_durable_batch(self, references: Sequence[MacroCreditTrajectory]) -> list[_Evaluation]:
        _configure_cuda_tf32(self.device, self.config.allow_tf32)
        ac_model = cast(HybridDomainActorCriticModel, self.model)
        use_rev = self._use_revival_ctx()
        results: list[_Evaluation] = []
        batch_cap = self.config.eval_batch_size
        for group_start in range(0, len(references), batch_cap):
            group = references[group_start : group_start + batch_cap]
            m = len(group)
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
                    _, _, _, next_h = self._forward_full(snap, hidden[i : i + 1] if has_hidden[i] else None)
                    hidden[i] = next_h[0].detach()
                    has_hidden[i] = True
                    calls[i] += 1
            afterstate_sets = [{node.step_index for node in traj.afterstate_nodes} for traj in group]
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
                active = [i for i in range(m) if t < len(group[i].steps)]
                step_actives.append(active)
                if t % self.config.chunk_steps == 0:
                    step_had_hidden.append([has_hidden[i] for i in active])
                    boundary_hids_gpu[t] = (
                        hidden[active].detach().clone() if any(has_hidden[i] for i in active) else None
                    )
                else:
                    step_had_hidden.append([])
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
                output = cast(
                    HybridActorCriticOutput,
                    ac_model(
                        batch,
                        hidden[active],
                        critic_revival_context=rev_t_gpu,
                        validate=False,
                    ),
                )
                if output.durable_value is None:
                    raise ValueError("durable_afterstate_v2 requires durable_value on every step")
                masks = batch.candidates.action_mask
                batch_logs = torch.log_softmax(output.policy_logits.masked_fill(~masks, -torch.inf), dim=-1)
                safe_batch_logs = batch_logs.masked_fill(~masks, 0.0)
                finite_ok = (
                    finite_ok
                    & torch.isfinite(output.policy_logits).all()
                    & torch.isfinite(output.value).all()
                    & torch.isfinite(output.recurrent_state).all()
                    & torch.isfinite(output.durable_value).all()
                    & torch.isfinite(safe_batch_logs).all()
                )
                hidden[active] = output.recurrent_state.detach()
                for i in active:
                    has_hidden[i] = True
                step_logs_gpu.append(batch_logs.detach())
                step_vals_gpu.append(output.value.detach())
                step_durs_gpu.append(output.durable_value.detach())
                if any(t in afterstate_sets[i] for i in active):
                    post_hids_gpu[t] = output.recurrent_state.detach().clone()

            if not bool(finite_ok.item()):
                raise FloatingPointError("macro actor-critic forward or legal log probabilities are non-finite")

            if t_max > 0:
                flat_vals_cpu = torch.cat(step_vals_gpu, dim=0).cpu()
                flat_durs_cpu = torch.cat(step_durs_gpu, dim=0).cpu()
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
                    durs_t_cpu = flat_durs_cpu[v_offset : v_offset + k_len]
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
                        if not (t == 0 and traj.root_intervened):
                            reconstructed = macro_mixture_log_probs(policy, step.snapshot, traj.macro_branch_epsilon)
                            err = abs(float(reconstructed[step.action_index].item()) - step.behavior_log_probability)
                            max_errors[i] = max(max_errors[i], err)
                            if err > self.config.likelihood_tolerance:
                                raise ValueError(
                                    "fresh teacher likelihood differs from recorded stochastic behavior "
                                    f"(err={err:.6g}, tol={self.config.likelihood_tolerance:.6g}, "
                                    f"step={t}/{len(traj.steps)}, domain={step.control_domain})"
                                )
                        durable_values_list[i].append(durs_t_cpu[k].clone())
                        values_list[i].append(vals_t_cpu[k].clone())
                        logs_list[i].append(selected.clone())
                        if t in afterstate_sets[i]:
                            assert post_hid_cpu is not None
                            step_log_probs_list[i][t] = policy
                            post_step_hidden_list[i][t] = post_hid_cpu[k : k + 1].clone()
                        elif step.control_domain == "macro" and int(step.snapshot.action_mask.sum()) > 1:
                            step_log_probs_list[i][t] = policy
                        calls[i] += 1

            boot_indices = [i for i, traj in enumerate(group) if traj.bootstrap_snapshot is not None]
            boot_vals: dict[int, Tensor] = {}
            dur_boot_vals: dict[int, Tensor] = {}
            if boot_indices:
                self._check_resource()
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
                boot_out = cast(
                    HybridActorCriticOutput,
                    ac_model(
                        boot_batch,
                        hidden[boot_indices],
                        critic_revival_context=boot_rev_gpu,
                        validate=False,
                    ),
                )
                if boot_out.durable_value is None:
                    raise ValueError("durable_afterstate_v2 requires durable_value on bootstrap snapshot")
                boot_ok = (
                    torch.isfinite(boot_out.policy_logits).all()
                    & torch.isfinite(boot_out.value).all()
                    & torch.isfinite(boot_out.recurrent_state).all()
                    & torch.isfinite(boot_out.durable_value).all()
                )
                if not bool(boot_ok.item()):
                    raise FloatingPointError("macro actor-critic bootstrap forward is non-finite")
                b_val_cpu = boot_out.value.detach().cpu()
                b_dur_cpu = boot_out.durable_value.detach().cpu()
                for k, i in enumerate(boot_indices):
                    boot_vals[i] = b_val_cpu[k].clone()
                    dur_boot_vals[i] = b_dur_cpu[k].clone()
                    calls[i] += 1
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
                        durable_values=torch.stack(durable_values_list[i]),
                        durable_bootstrap_value=dur_boot_vals.get(i),
                        step_log_probs=step_log_probs_list[i],
                        post_step_hidden=post_step_hidden_list[i],
                    )
                )
        return results

    @torch.no_grad()
    def _prepare_durable_natural_batch(
        self, references: Sequence[MacroCreditTrajectory]
    ) -> tuple[list[_PreparedReference], int, float]:
        if not references:
            return [], 0, 0.0
        ac_model = cast(HybridDomainActorCriticModel, self.model)
        use_rev = self._use_revival_ctx()
        rev_all = (
            [self._trajectory_critic_revival_contexts(ref) for ref in references]
            if use_rev
            else None
        )
        evaluations = self._evaluate_durable_batch(references)
        flat_worlds: list[EncodedWorldSnapshot] = []
        flat_hiddens: list[Tensor] = []
        flat_revs: list[Tensor] = []
        node_slices: list[tuple[int, MacroAfterstateNode, int, int]] = []
        total_calls = sum(ev.forward_calls for ev in evaluations)
        max_error = max((ev.maximum_likelihood_error for ev in evaluations), default=0.0)
        for ref_idx, (reference, evaluation) in enumerate(zip(references, evaluations, strict=True)):
            assert evaluation.post_step_hidden is not None
            for node in reference.afterstate_nodes:
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
        if flat_worlds:
            n_worlds = len(flat_worlds)
            perm = sorted(range(n_worlds), key=lambda idx: flat_worlds[idx].world.token_count)
            sorted_worlds = [flat_worlds[idx] for idx in perm]
            stacked_hiddens = torch.cat(flat_hiddens, dim=0)[perm]
            stacked_revs = torch.cat(flat_revs, dim=0)[perm] if rev_all is not None else None
            world_cap = max(128, min(256, self.config.eval_batch_size * 8))
            gpu_q_chunks: list[Tensor] = []
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
                dur_vals, _ = ac_model.forward_durable_value(
                    world_batch,
                    domain_ids,
                    h_gpu,
                    critic_revival_context=r_gpu,
                    validate=False,
                )
                gpu_q_chunks.append(dur_vals.detach())
            sorted_q_gpu = torch.cat(gpu_q_chunks, dim=0)
            if not bool(torch.isfinite(sorted_q_gpu).all().item()):
                raise FloatingPointError("forward_durable_value output is non-finite")
            sorted_q_cpu = sorted_q_gpu.cpu()
            all_q_after = torch.empty_like(sorted_q_cpu)
            all_q_after[perm] = sorted_q_cpu
        else:
            all_q_after = torch.empty(0)
        per_ref_nodes: list[list[tuple[MacroAfterstateNode, Tensor]]] = [[] for _ in references]
        for ref_idx, node, s_w, e_w in node_slices:
            per_ref_nodes[ref_idx].append((node, all_q_after[s_w:e_w]))

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
            for node, q_after in per_ref_nodes[ref_idx]:
                teacher_logs = evaluation.step_log_probs[node.step_index]
                projection = project_macro_policy(
                    action_value_targets=q_after.to(dtype=teacher_logs.dtype),
                    teacher_log_probs=teacher_logs,
                    max_kl=effective_macro_kl,
                )
                step = reference.steps[node.step_index]
                if step.terminal:
                    post_target = durable_action_targets.new_zeros(())
                else:
                    eff_r = self._effective_step_reward(reference, node.step_index)
                    post_target = (
                        (durable_action_targets[node.step_index] - eff_r) / step.discount
                    ).detach()
                afterstate_targets[node.step_index] = _AfterstateStepTarget(
                    node=node,
                    action_values=q_after.detach(),
                    projection=projection,
                    post_target=post_target,
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
    def _prepare_durable_natural(self, reference: MacroCreditTrajectory) -> tuple[_PreparedReference, int, float]:
        use_rev = self._use_revival_ctx()
        step_rev_cpu: Tensor | None = None
        post_rev_cpu: Tensor | None = None
        if use_rev:
            step_rev_cpu, post_rev_cpu, _ = self._trajectory_critic_revival_contexts(reference)
        evaluation = self._evaluate(reference, root_index=-1)
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
        for node in reference.afterstate_nodes:
            saved_hidden = evaluation.post_step_hidden[node.step_index].to(device=self.device, dtype=self.dtype)
            post_rev_gpu = (
                None
                if post_rev_cpu is None
                else post_rev_cpu[node.step_index : node.step_index + 1].to(
                    device=self.device, dtype=self.dtype
                )
            )
            q_after = (
                self._forward_durable_worlds(
                    node.post_worlds,
                    saved_hidden,
                    critic_revival_context=post_rev_gpu,
                )
                .detach()
                .cpu()
            )
            calls += 1
            teacher_logs = evaluation.step_log_probs[node.step_index]
            projection = project_macro_policy(
                action_value_targets=q_after.to(dtype=teacher_logs.dtype),
                teacher_log_probs=teacher_logs,
                max_kl=effective_macro_kl,
            )
            step = reference.steps[node.step_index]
            if step.terminal:
                post_target = durable_action_targets.new_zeros(())
            else:
                eff_r = self._effective_step_reward(reference, node.step_index)
                post_target = ((durable_action_targets[node.step_index] - eff_r) / step.discount).detach()
            afterstate_targets[node.step_index] = _AfterstateStepTarget(
                node=node,
                action_values=q_after.detach(),
                projection=projection,
                post_target=post_target,
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

    def _prepare_natural(self, reference: MacroCreditTrajectory) -> tuple[_PreparedReference, int, float]:
        if self.config.target_mode == "durable_afterstate_v2":
            return self._prepare_durable_natural(reference)
        # -1 requests recurrent chunk boundaries, but no artificial macro root.
        step_rev_cpu: Tensor | None = None
        post_rev_cpu: Tensor | None = None
        if self._use_revival_ctx():
            step_rev_cpu, post_rev_cpu, _ = self._trajectory_critic_revival_contexts(reference)
        evaluation = self._evaluate(reference, root_index=-1)
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
            combat_values[start:stop] = targets
            combat_advantages[start:stop] = policy_rho * (action_targets - evaluation.values[start:stop])
        return combat_values, combat_advantages

    def _prepare(self, root: MacroCounterfactualRoot) -> tuple[_PreparedReference, int, float]:
        step_rev_cpu: Tensor | None = None
        post_rev_cpu: Tensor | None = None
        if self._use_revival_ctx():
            step_rev_cpu, post_rev_cpu, _ = self._trajectory_critic_revival_contexts(root.reference)
        evaluation = self._evaluate(root.reference, root_index=root.root_step_index)
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
        self, prepared: _PreparedReference, start: int, stop: int, denominators: Mapping[str, int],
    ) -> dict[str, float]:
        self._check_resource()
        saved = prepared.evaluation.boundary_hidden[start]
        hidden = None if saved is None else saved.to(device=self.device, dtype=self.dtype)
        totals = {
            key: 0.0
            for key in (
                "macro_policy_loss",
                "macro_mpo_policy_loss",
                "macro_pg_policy_loss",
                "macro_entropy",
                "macro_value_loss",
                "durable_value_loss",
                "durable_post_value_loss",
                "combat_policy_loss",
                "combat_value_loss",
                "combat_entropy",
                "backward_chunks",
            )
        }
        terms: list[Tensor] = []
        cfg = self.config
        is_durable_v2 = cfg.target_mode == "durable_afterstate_v2"
        durable_actor_active = (not is_durable_v2) or (self.update >= cfg.durable_warmup_updates)
        combat_actor_active = (not is_durable_v2) or (self.update >= cfg.combat_actor_warmup_updates)
        for index in range(start, stop):
            step = prepared.reference.steps[index]
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
                    durable_loss = F.smooth_l1_loss(durable_value, durable_target) / max(
                        1, denominators["durable_value_steps"]
                    )
                    totals["durable_value_loss"] += float(durable_loss.detach().item())
                    if cfg.macro_value_weight:
                        terms.append(0.5 * cfg.macro_value_weight * durable_loss)
            if step.control_domain == "macro":
                target = prepared.macro_values[index].to(device=self.device, dtype=value.dtype)
                value_loss = F.smooth_l1_loss(value, target) / denominators["macro_steps"]
                totals["macro_value_loss"] += float(value_loss.detach().item())
                if cfg.macro_value_weight:
                    terms.append(0.5 * cfg.macro_value_weight * value_loss)
                if is_durable_v2:
                    assert prepared.afterstate_targets is not None
                    as_target = prepared.afterstate_targets.get(index)
                    if as_target is not None:
                        chosen_world = as_target.node.post_worlds[step.action_index]
                        post_rev_gpu = (
                            None
                            if prepared.post_revival_ctx is None
                            else prepared.post_revival_ctx[index : index + 1].to(
                                device=self.device, dtype=self.dtype
                            )
                        )
                        chosen_post_val = self._forward_durable_worlds(
                            (chosen_world,),
                            hidden,
                            critic_revival_context=post_rev_gpu,
                        )[0]
                        post_target = as_target.post_target.to(device=self.device, dtype=chosen_post_val.dtype)
                        post_loss = F.smooth_l1_loss(chosen_post_val, post_target) / max(
                            1, denominators["afterstate_steps"]
                        )
                        totals["durable_post_value_loss"] += float(post_loss.detach().item())
                        if cfg.durable_post_value_weight:
                            terms.append(0.5 * cfg.durable_post_value_weight * post_loss)
                        if durable_actor_active and as_target.projection.actor_enabled:
                            target_policy = as_target.projection.target_probabilities.to(
                                device=self.device, dtype=logs.dtype
                            )
                            actor = -(target_policy * logs).sum() / max(1, denominators["macro_policy_steps"])
                            totals["macro_policy_loss"] += float(actor.detach().item())
                            totals["macro_mpo_policy_loss"] += float(actor.detach().item())
                            if cfg.macro_policy_weight:
                                terms.append(0.5 * cfg.macro_policy_weight * actor)
                    elif choice and durable_actor_active:
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
                    actor = -logs[step.action_index] * advantage / denominators["combat_policy_steps"]
                    normalized_entropy = entropy / denominators["combat_policy_steps"]
                    totals["combat_policy_loss"] += float(actor.detach().item())
                    totals["combat_entropy"] += float(normalized_entropy.detach().item())
                    if combat_actor_active:
                        if cfg.combat_policy_weight:
                            terms.append(0.5 * cfg.combat_policy_weight * actor)
                        if cfg.combat_entropy_weight:
                            terms.append(-0.5 * cfg.combat_entropy_weight * normalized_entropy)
        if terms:
            loss = torch.stack(terms).sum()
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError("macro cohort chunk loss is non-finite")
            loss.backward()  # type: ignore[no-untyped-call]
            totals["backward_chunks"] = 1.0
        self._check_resource()
        return totals

    def _backward_chunks_durable_batch(
        self,
        prepared_items: Sequence[_PreparedReference],
        denominators: Mapping[str, int],
    ) -> tuple[dict[str, float], int]:
        ac_model = cast(HybridDomainActorCriticModel, self.model)
        cfg = self.config
        durable_actor_active = self.update >= cfg.durable_warmup_updates
        combat_actor_active = self.update >= cfg.combat_actor_warmup_updates
        totals = {
            key: 0.0
            for key in (
                "macro_policy_loss",
                "macro_mpo_policy_loss",
                "macro_pg_policy_loss",
                "macro_entropy",
                "macro_value_loss",
                "durable_value_loss",
                "durable_post_value_loss",
                "combat_policy_loss",
                "combat_value_loss",
                "combat_entropy",
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
        staged_afterstates: list[dict[int, tuple[_AfterstateStepTarget, Tensor, Tensor, Tensor | None]]] = []

        cursor_steps = 0
        for item in prepared_items:
            ref_step_offsets.append(cursor_steps)
            as_map: dict[int, tuple[_AfterstateStepTarget, Tensor, Tensor, Tensor | None]] = {}
            assert item.afterstate_targets is not None
            for step_idx, as_t in item.afterstate_targets.items():
                post_rev_item = (
                    None
                    if item.post_revival_ctx is None
                    else item.post_revival_ctx[step_idx].to(device=self.device, dtype=self.dtype)
                )
                as_map[step_idx] = (
                    as_t,
                    as_t.post_target.to(device=self.device, dtype=self.dtype),
                    as_t.projection.target_probabilities.to(device=self.device, dtype=self.dtype),
                    post_rev_item,
                )
            staged_afterstates.append(as_map)
            steps = item.reference.steps
            for idx, s in enumerate(steps):
                is_mac = s.control_domain == "macro"
                is_c_entry = (not is_mac) and (idx == 0 or steps[idx - 1].control_domain == "macro")
                has_dur = is_mac or is_c_entry
                is_ch = bool(int(s.snapshot.action_mask.sum()) > 1)
                as_entry = as_map.get(idx)
                has_as = as_entry is not None
                if is_mac:
                    contrib = bool(cfg.macro_value_weight)
                    if as_entry is not None:
                        contrib = contrib or bool(cfg.durable_post_value_weight) or (
                            durable_actor_active
                            and as_entry[0].projection.actor_enabled
                            and bool(cfg.macro_policy_weight)
                        )
                    elif is_ch and durable_actor_active:
                        contrib = contrib or bool(cfg.macro_policy_weight) or bool(cfg.macro_entropy_weight)
                else:
                    contrib = (
                        (is_c_entry and bool(cfg.macro_value_weight))
                        or bool(cfg.combat_value_weight)
                        or (
                            is_ch
                            and combat_actor_active
                            and (bool(cfg.combat_policy_weight) or bool(cfg.combat_entropy_weight))
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
        keyed_chunks.sort(key=lambda row: (row[0], row[1]))

        maximum_chunk = max((stop - start for _, _, _, _, start, stop in keyed_chunks), default=0)
        zero_row = ac_model.initial_state(1, device=self.device)[0].detach()

        denom_dur_val = max(1, denominators["durable_value_steps"])
        denom_macro_steps = max(1, denominators["macro_steps"])
        denom_afterstate = max(1, denominators["afterstate_steps"])
        denom_macro_pol = max(1, denominators["macro_policy_steps"])
        denom_combat_steps = max(1, denominators["combat_steps"])
        denom_combat_pol = max(1, denominators["combat_policy_steps"])

        metric_keys = (
            "macro_policy_loss",
            "macro_mpo_policy_loss",
            "macro_pg_policy_loss",
            "macro_entropy",
            "macro_value_loss",
            "durable_value_loss",
            "durable_post_value_loss",
            "combat_policy_loss",
            "combat_value_loss",
            "combat_entropy",
        )

        slice_start = 0
        total_chunks = len(keyed_chunks)
        max_local_token_budget = 98_304
        max_world_token_budget = 65_536
        while slice_start < total_chunks:
            self._check_resource()
            tentative_end = min(total_chunks, slice_start + cfg.chunk_batch_size)
            worst_cl = max(1, keyed_chunks[tentative_end - 1][0])
            worst_w = max(1, max(row[1] for row in keyed_chunks[slice_start:tentative_end]))
            dynamic_cap = max(
                8,
                min(
                    cfg.chunk_batch_size,
                    max_local_token_budget // worst_cl,
                    max_world_token_budget // worst_w,
                ),
            )
            slice_end = min(total_chunks, slice_start + dynamic_cap)
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

                if any_dur:
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
                    as_act_k: list[int] = []
                    as_act_probs: list[Tensor] = []
                    for k, c in enumerate(active_c):
                        ref_idx, item, start, _ = chunk_slice[c]
                        index = start + offset
                        staged = staged_afterstates[ref_idx].get(index)
                        if staged is not None:
                            as_t, post_target_gpu, target_prob_gpu, post_rev_gpu = staged
                            step = item.reference.steps[index]
                            as_worlds.append(as_t.node.post_worlds[step.action_index])
                            as_hiddens.append(output.recurrent_state[k])
                            if post_rev_gpu is not None:
                                as_revs.append(post_rev_gpu)
                            as_post_targets.append(post_target_gpu)
                            if durable_actor_active and as_t.projection.actor_enabled:
                                as_act_k.append(k)
                                as_act_probs.append(target_prob_gpu)

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
                        post_vals, _ = ac_model.forward_durable_value(
                            world_batch,
                            domain_ids,
                            torch.stack(as_hiddens, dim=0),
                            critic_revival_context=(torch.stack(as_revs, dim=0) if as_revs else None),
                            validate=False,
                        )
                        finite_ok = finite_ok & torch.isfinite(post_vals).all()
                        post_loss = (
                            F.smooth_l1_loss(
                                post_vals, torch.stack(as_post_targets, dim=0), reduction="sum"
                            )
                            / denom_afterstate
                        )
                        slice_totals_gpu["durable_post_value_loss"] = (
                            slice_totals_gpu["durable_post_value_loss"] + post_loss.detach().double()
                        )
                        if cfg.durable_post_value_weight:
                            batch_terms.append(0.5 * cfg.durable_post_value_weight * post_loss)

                        if as_act_k:
                            c_cap_step = int(safe_logs.shape[1])
                            padded_probs = torch.zeros(
                                (len(as_act_k), c_cap_step), device=self.device, dtype=safe_logs.dtype
                            )
                            for m_i, tp in enumerate(as_act_probs):
                                padded_probs[m_i, : tp.numel()] = tp.to(dtype=safe_logs.dtype)
                            k_idx_gpu = torch.tensor(as_act_k, device=self.device, dtype=torch.long)
                            mpo_actor = -(padded_probs * safe_logs[k_idx_gpu]).sum() / denom_macro_pol
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

            if batch_terms:
                mega_loss = torch.stack(batch_terms).sum()
                batch_terms.clear()
                if not bool((finite_ok & torch.isfinite(mega_loss)).item()):
                    raise FloatingPointError("macro cohort chunk forward or loss is non-finite")
                mega_loss.backward()  # type: ignore[no-untyped-call]
                del mega_loss
                totals["backward_chunks"] += float(chunk_has_terms_count)
            elif not bool(finite_ok.item()):
                raise FloatingPointError("macro cohort chunk forward is non-finite")
            hidden_list.clear()
            stacked_metrics = torch.stack([slice_totals_gpu[key] for key in metric_keys]).cpu().tolist()
            for key, val in zip(metric_keys, stacked_metrics, strict=True):
                totals[key] += float(val)
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

    def learn(self, cohort: MacroCreditCohort, *, resource_guard: Callable[[], None] | None = None) -> dict[str, Any]:
        if self._failed:
            raise RuntimeError("failed macro learner must reload its last committed checkpoint")
        self._validate_update()
        if not isinstance(cohort, MacroCreditCohort) or cohort.teacher_version != self.update:
            raise ValueError("macro cohort must match the learner's exact fresh teacher version")
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
        denominators = {
            "root_count": len(cohort.roots),
            "macro_steps": 0,
            "macro_policy_steps": 0,
            "durable_value_steps": 0,
            "afterstate_steps": 0,
            "macro_pg_fallback_steps": 0,
            "combat_steps": 0,
            "combat_policy_steps": 0,
        }
        for reference in cohort.references:
            afterstate_indices = {node.step_index for node in getattr(reference, "afterstate_nodes", ())}
            denominators["afterstate_steps"] += len(afterstate_indices)
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
        prepared: list[_PreparedReference] = []
        target_calls = 0
        maximum_error = 0.0
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
            batch_items, batch_calls, batch_error = self._prepare_durable_natural_batch(cohort.natural_only)
            prepared.extend(batch_items)
            target_calls += batch_calls
            maximum_error = max(maximum_error, batch_error)
        else:
            for reference in cohort.natural_only:
                item, calls, error = self._prepare_natural(reference)
                prepared.append(item)
                target_calls += calls
                maximum_error = max(maximum_error, error)
        before = [parameter.detach().cpu().clone() for parameter in self._parameters]
        totals = {
            key: 0.0
            for key in (
                "macro_policy_loss",
                "macro_mpo_policy_loss",
                "macro_pg_policy_loss",
                "macro_entropy",
                "macro_value_loss",
                "durable_value_loss",
                "durable_post_value_loss",
                "combat_policy_loss",
                "combat_value_loss",
                "combat_entropy",
                "backward_chunks",
            )
        }
        maximum_chunk = 0
        # This remains set through ALL post-step validation/CPU copies/metrics,
        # including BaseException interruptions. Only complete success clears it.
        self._failed = True
        try:
            if (
                self.config.target_mode == "durable_afterstate_v2"
                and self.config.chunk_batch_size > 1
                and prepared
            ):
                batch_totals, maximum_chunk = self._backward_chunks_durable_batch(prepared, denominators)
                for key, value in batch_totals.items():
                    totals[key] += value
            else:
                for item in prepared:
                    for start in range(0, len(item.reference.steps), self.config.chunk_steps):
                        stop = min(start + self.config.chunk_steps, len(item.reference.steps))
                        output = self._backward_chunk(item, start, stop, denominators)
                        for key, value in output.items():
                            totals[key] += value
                        maximum_chunk = max(maximum_chunk, stop - start)
            groups = {"policy_head": 0.0, "value_head": 0.0, "domain_stream": 0.0, "shared": 0.0}
            for name, parameter in self.model.named_parameters():
                if parameter.grad is not None:
                    if not bool(torch.isfinite(parameter.grad).all().item()):
                        raise FloatingPointError("macro gradient is non-finite")
                    if name.startswith(("candidate_policy_head", "branch_policy_head", "branch_vitals_")):
                        group = "policy_head"
                    elif name.startswith(
                        (
                            "state_value_head",
                            "state_revival_",
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
                gradient_norm_val = math.hypot(val_gn, non_val_gn)
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
        except BaseException:
            self.optimizer.zero_grad(set_to_none=True)
            self._failed = True
            raise
        delta_squared = sum(float((parameter.detach().cpu().double() - previous.double()).square().sum().item())
                            for parameter, previous in zip(self._parameters, before, strict=True))
        prior_update = self.update
        self.update += 1
        cfg = self.config
        is_durable_v2 = cfg.target_mode == "durable_afterstate_v2"
        durable_actor_active = (not is_durable_v2) or (prior_update >= cfg.durable_warmup_updates)
        combat_actor_active = (not is_durable_v2) or (prior_update >= cfg.combat_actor_warmup_updates)
        macro_loss = (
            cfg.macro_policy_weight * totals["macro_policy_loss"]
            + cfg.macro_value_weight * (totals["macro_value_loss"] + totals["durable_value_loss"])
            + cfg.durable_post_value_weight * totals["durable_post_value_loss"]
            - (cfg.macro_entropy_weight * totals["macro_entropy"] if is_durable_v2 else 0.0)
        )
        combat_loss = (
            (cfg.combat_policy_weight * totals["combat_policy_loss"] if combat_actor_active else 0.0)
            + cfg.combat_value_weight * totals["combat_value_loss"]
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
        self.metrics = {
            "version": MACRO_CREDIT_LEARNER_VERSION, "update": self.update, "teacher_version": cohort.teacher_version,
            "policy_lag": 0, "cohort_id": cohort.cohort_id, "target_mode": cfg.target_mode,
            "durable_actor_active": durable_actor_active,
            "combat_actor_active": combat_actor_active,
            "effective_macro_kl": cfg.macro_kl if durable_actor_active else 0.0,
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
            "natural_only_games": len(cohort.natural_only),
            "root_targets": root_targets, "mean_root_kl": sum(item.projection.kl for item in prepared if item.projection is not None) / max(1, len(root_targets)),
            "kl_semantics": "mean_root_kl is projection q||teacher; updated_policy_kl_to_teacher is measured after fitting",
            "mean_updated_policy_kl_to_teacher": sum(row["updated_policy_kl_to_teacher"] for row in root_targets) / max(1, len(root_targets)),
            "gradient_norm": gradient_norm_val,
            **{f"{name}_gradient_norm": math.sqrt(value) for name, value in groups.items()},
            "parameter_delta_norm": math.sqrt(delta_squared),
            "target_forward_calls": target_calls,
            "training_forward_calls": denominators["macro_steps"] + denominators["combat_steps"] + (denominators["afterstate_steps"] if is_durable_v2 else 0),
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
