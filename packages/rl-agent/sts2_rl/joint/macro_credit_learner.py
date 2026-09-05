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
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Final, TypeAlias, cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from sts2_rl.encoding import EncodedDecisionSnapshot
from sts2_rl.encoding.snapshot import collate_encoded_snapshots
from sts2_rl.models.hybrid_actor_critic import HybridActorCriticOutput

from .macro_credit_data import MacroCounterfactualRoot, MacroCreditCohort, MacroCreditTrajectory
from .macro_returns import mixed_vtrace_lambda_targets
from .macro_targets import MacroPolicyProjection, project_macro_policy
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

    def __post_init__(self) -> None:
        for name in ("chunk_steps", "combat_unroll_steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 16:
                raise ValueError(f"macro credit {name} must be an integer in [1, 16]")
        positive = ("learning_rate", "shared_learning_rate_scale", "rho_clip", "c_clip", "policy_rho_clip",
                    "gradient_clip_norm", "likelihood_tolerance")
        non_negative = ("weight_decay", "macro_kl", "macro_policy_weight", "macro_value_weight",
                        "combat_policy_weight", "combat_value_weight", "combat_entropy_weight")
        for name in (*positive, *non_negative, "macro_trace_lambda"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0 or (name in positive and value == 0):
                raise ValueError(f"macro credit {name} has an invalid finite range")
        if self.macro_trace_lambda > 1 or self.c_clip > self.rho_clip:
            raise ValueError("macro lambda must be <= 1 and c_clip <= rho_clip")


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


@dataclass(slots=True)
class _PreparedReference:
    root: MacroCounterfactualRoot
    evaluation: _Evaluation
    macro_values: Tensor
    combat_values: Tensor
    combat_advantages: Tensor
    projection: MacroPolicyProjection
    action_values: Tensor


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
        self._parameters = list(model.parameters())
        if not self._parameters or not all(parameter.requires_grad for parameter in self._parameters):
            raise ValueError("macro learner requires a non-empty, trainable actor-critic model")
        if any("candidate_q_head" in name or "target_q" in name for name, _ in model.named_parameters()):
            raise ValueError("macro learner must not own a learned Q head")
        parameter = self._parameters[0]
        self.device = parameter.device
        self.dtype = parameter.dtype
        local_tokens = ("macro_stream", "combat_stream", "macro_stream_norm", "combat_stream_norm",
                        "candidate_policy_head", "state_value_head")
        local: list[nn.Parameter] = []
        shared: list[nn.Parameter] = []
        for name, item in model.named_parameters():
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

    def _forward(self, snapshot: EncodedDecisionSnapshot, hidden: Tensor | None) -> tuple[Tensor, Tensor, Tensor]:
        self._check_resource()
        if self._forward_override is not None:
            logits, value, next_hidden = self._forward_override(snapshot, hidden)
        else:
            batch = collate_encoded_snapshots(
                (snapshot,), expected_config=snapshot.config,
                expected_fingerprint=snapshot.encoding_fingerprint, device=self.device,
            )
            output = cast(HybridActorCriticOutput, self.model(batch, hidden, validate=False))
            logits, value, next_hidden = output.policy_logits[0], output.value.reshape(()), output.recurrent_state
        if logits.ndim != 1 or logits.numel() < snapshot.candidate_count or value.numel() != 1:
            raise ValueError("macro actor-critic forward has invalid logits/value shapes")
        if next_hidden.ndim != 2 or next_hidden.shape[0] != 1:
            raise ValueError("macro actor-critic forward must return one recurrent state")
        if not all(bool(torch.isfinite(item).all().item()) for item in (logits, value, next_hidden)):
            raise FloatingPointError("macro actor-critic forward is non-finite")
        return logits, value.reshape(()), next_hidden

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
        hidden: Tensor | None = None
        calls = 0
        for snapshot in trajectory.prefix:
            _, _, hidden = self._forward(snapshot, hidden)
            hidden = hidden.detach()
            calls += 1
        boundaries: dict[int, Tensor | None] = {}
        values: list[Tensor] = []
        logs: list[Tensor] = []
        root_logs: Tensor | None = None
        maximum_error = 0.0
        for index, step in enumerate(trajectory.steps):
            if root_index is not None and index % self.config.chunk_steps == 0:
                boundaries[index] = None if hidden is None else hidden.detach().cpu().clone()
            logits, value, hidden = self._forward(step.snapshot, hidden)
            hidden = hidden.detach()
            policy, _, _ = self._policy(step, logits)
            selected = policy[step.action_index]
            if not (index == 0 and trajectory.root_intervened):
                error = abs(float(selected.item()) - step.behavior_log_probability)
                maximum_error = max(maximum_error, error)
                if error > self.config.likelihood_tolerance:
                    raise ValueError("fresh teacher likelihood differs from recorded stochastic behavior")
            if index == root_index:
                root_logs = policy.detach().cpu().clone()
            values.append(value.detach().cpu().clone())
            logs.append(selected.detach().cpu().clone())
            calls += 1
        bootstrap = None
        if trajectory.bootstrap_snapshot is not None:
            _, bootstrap, _ = self._forward(trajectory.bootstrap_snapshot, hidden)
            bootstrap = bootstrap.detach().cpu().clone()
            calls += 1
        return _Evaluation(torch.stack(values), torch.stack(logs), bootstrap, boundaries, root_logs, calls, maximum_error)

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
            rewards=values.new_tensor([step.reward for step in steps]),
            discounts=values.new_tensor([step.discount for step in steps]), values=values,
            target_log_probs=evaluation.selected_logs[start:stop],
            behavior_log_probs=values.new_tensor([step.behavior_log_probability for step in steps]),
            control_domains=tuple(step.control_domain for step in steps), trace_lambda=trace_lambda,
            rho_clip=self.config.rho_clip, c_clip=self.config.c_clip, **kwargs,
        )
        return result.value_targets, result.action_value_targets

    def _prepare(self, root: MacroCounterfactualRoot) -> tuple[_PreparedReference, int, float]:
        evaluation = self._evaluate(root.reference, root_index=root.root_step_index)
        assert evaluation.root_logs is not None
        recorded = evaluation.root_logs.new_tensor(root.teacher_log_probabilities)
        if not torch.allclose(evaluation.root_logs, recorded, rtol=0, atol=self.config.likelihood_tolerance):
            raise ValueError("full root prior differs from fresh reconstructed teacher")
        calls = evaluation.forward_calls
        maximum_error = max(evaluation.maximum_likelihood_error, float((evaluation.root_logs - recorded).abs().max().item()))
        action_values: list[Tensor] = []
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
        combat_values = torch.empty_like(evaluation.values)
        combat_advantages = torch.empty_like(evaluation.values)
        for start in range(0, len(root.reference.steps), self.config.combat_unroll_steps):
            stop = min(start + self.config.combat_unroll_steps, len(root.reference.steps))
            targets, action_targets = self._targets(root.reference, evaluation, start=start, stop=stop, trace_lambda=1.0)
            behavior = evaluation.values.new_tensor([step.behavior_log_probability for step in root.reference.steps[start:stop]])
            policy_rho = torch.exp((evaluation.selected_logs[start:stop] - behavior).clamp(max=math.log(self.config.policy_rho_clip)))
            combat_values[start:stop] = targets
            combat_advantages[start:stop] = policy_rho * (action_targets - evaluation.values[start:stop])
        return _PreparedReference(root, evaluation, macro_values, combat_values, combat_advantages, projection, candidate_values), calls, maximum_error

    def _backward_chunk(
        self, prepared: _PreparedReference, start: int, stop: int, denominators: Mapping[str, int],
    ) -> dict[str, float]:
        self._check_resource()
        saved = prepared.evaluation.boundary_hidden[start]
        hidden = None if saved is None else saved.to(device=self.device, dtype=self.dtype)
        totals = {key: 0.0 for key in ("macro_policy_loss", "macro_value_loss", "combat_policy_loss",
                                      "combat_value_loss", "combat_entropy", "backward_chunks")}
        terms: list[Tensor] = []
        cfg = self.config
        for index in range(start, stop):
            step = prepared.root.reference.steps[index]
            logits, value, hidden = self._forward(step.snapshot, hidden)
            logs, entropy, choice = self._policy(step, logits)
            if step.control_domain == "macro":
                target = prepared.macro_values[index].to(device=self.device, dtype=value.dtype)
                value_loss = F.smooth_l1_loss(value, target) / denominators["macro_steps"]
                totals["macro_value_loss"] += float(value_loss.detach().item())
                if cfg.macro_value_weight:
                    terms.append(0.5 * cfg.macro_value_weight * value_loss)
                if index == prepared.root.root_step_index and prepared.projection.actor_enabled:
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

    @torch.no_grad()
    def _updated_root_policy(self, prepared: _PreparedReference) -> dict[str, Any]:
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
        if not _finite_tree(self.model.state_dict()) or not _finite_tree(self.optimizer.state_dict()):
            raise FloatingPointError("macro model/optimizer state is non-finite before update")
        self.model.eval()
        self.optimizer.zero_grad(set_to_none=True)
        started = time.monotonic()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        denominators = {"root_count": len(cohort.roots), "macro_steps": 0, "combat_steps": 0, "combat_policy_steps": 0}
        for root in cohort.roots:
            for step in root.reference.steps:
                denominators[f"{step.control_domain}_steps"] += 1
                if step.control_domain == "combat" and int(step.snapshot.action_mask.sum()) > 1:
                    denominators["combat_policy_steps"] += 1
        prepared: list[_PreparedReference] = []
        target_calls = 0
        maximum_error = 0.0
        for root in cohort.roots:
            item, calls, error = self._prepare(root)
            prepared.append(item)
            target_calls += calls
            maximum_error = max(maximum_error, error)
        before = [parameter.detach().cpu().clone() for parameter in self._parameters]
        totals = {key: 0.0 for key in ("macro_policy_loss", "macro_value_loss", "combat_policy_loss",
                                      "combat_value_loss", "combat_entropy", "backward_chunks")}
        maximum_chunk = 0
        # This remains set through ALL post-step validation/CPU copies/metrics,
        # including BaseException interruptions. Only complete success clears it.
        self._failed = True
        try:
            for item in prepared:
                for start in range(0, len(item.root.reference.steps), self.config.chunk_steps):
                    stop = min(start + self.config.chunk_steps, len(item.root.reference.steps))
                    output = self._backward_chunk(item, start, stop, denominators)
                    for key, value in output.items():
                        totals[key] += value
                    maximum_chunk = max(maximum_chunk, stop - start)
            groups = {"policy_head": 0.0, "value_head": 0.0, "domain_stream": 0.0, "shared": 0.0}
            for name, parameter in self.model.named_parameters():
                if parameter.grad is not None:
                    if not bool(torch.isfinite(parameter.grad).all().item()):
                        raise FloatingPointError("macro gradient is non-finite")
                    if name.startswith("candidate_policy_head"):
                        group = "policy_head"
                    elif name.startswith("state_value_head"):
                        group = "value_head"
                    elif name.startswith(("macro_stream", "combat_stream")):
                        group = "domain_stream"
                    else:
                        group = "shared"
                    groups[group] += float(parameter.grad.detach().double().square().sum().item())
            gradient_norm = torch.nn.utils.clip_grad_norm_(self._parameters, self.config.gradient_clip_norm, error_if_nonfinite=True)
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
        self.update += 1
        cfg = self.config
        macro_loss = cfg.macro_policy_weight * totals["macro_policy_loss"] + cfg.macro_value_weight * totals["macro_value_loss"]
        combat_loss = (cfg.combat_policy_weight * totals["combat_policy_loss"] + cfg.combat_value_weight * totals["combat_value_loss"]
                       - cfg.combat_entropy_weight * totals["combat_entropy"])
        root_targets = [{
            "root_id": item.root.root_id, "action_values": item.action_values.tolist(),
            "teacher_probabilities": item.projection.teacher_probabilities.tolist(),
            "target_probabilities": item.projection.target_probabilities.tolist(),
            "value_target": float(item.projection.value_target.item()), "temperature": item.projection.temperature,
            "kl": item.projection.kl, "expected_improvement": item.projection.expected_improvement,
            "actor_enabled": item.projection.actor_enabled,
            **self._updated_root_policy(item),
        } for item in prepared]
        self.metrics = {
            "version": MACRO_CREDIT_LEARNER_VERSION, "update": self.update, "teacher_version": cohort.teacher_version,
            "policy_lag": 0, "cohort_id": cohort.cohort_id, **denominators, **totals,
            "macro_domain_weight": 0.5, "combat_domain_weight": 0.5,
            "macro_loss": macro_loss, "combat_loss": combat_loss, "loss": 0.5 * (macro_loss + combat_loss),
            "macro_policy_roots": sum(item.projection.actor_enabled for item in prepared),
            "root_targets": root_targets, "mean_root_kl": sum(item.projection.kl for item in prepared) / len(prepared),
            "kl_semantics": "mean_root_kl is projection q||teacher; updated_policy_kl_to_teacher is measured after fitting",
            "mean_updated_policy_kl_to_teacher": sum(row["updated_policy_kl_to_teacher"] for row in root_targets) / len(root_targets),
            "gradient_norm": float(gradient_norm.item()),
            **{f"{name}_gradient_norm": math.sqrt(value) for name, value in groups.items()},
            "parameter_delta_norm": math.sqrt(delta_squared),
            "target_forward_calls": target_calls, "training_forward_calls": denominators["macro_steps"] + denominators["combat_steps"],
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
