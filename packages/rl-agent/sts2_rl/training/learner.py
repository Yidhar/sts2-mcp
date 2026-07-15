"""Recurrent sequence actor-critic learner using IMPALA V-trace targets."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from sts2_baseline import SequenceUnroll
from sts2_rl.encoding import GroundedObservationEncoder
from sts2_rl.encoding.grounded import grounding_encoding_identity
from sts2_rl.encoding.snapshot import collate_encoded_snapshots
from sts2_rl.models import RecurrentCandidateModel

from .config import OptimizationConfig


@dataclass(frozen=True, slots=True)
class LearnerTimings:
    validation_ms: float
    recurrent_forward_ms: float
    target_and_loss_ms: float
    backward_ms: float
    optimizer_step_ms: float
    total_ms: float

    def to_mapping(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class LearnerMetrics:
    loss: float
    policy_loss: float
    value_loss: float
    entropy: float
    advantage_mean: float
    value_target_mean: float
    importance_ratio_mean: float
    importance_ratio_max: float
    importance_clip_fraction: float
    gradient_norm: float
    unrolls: int
    environment_steps: int
    policy_decisions: int
    maximum_policy_lag: int
    timings: LearnerTimings

    def to_mapping(self) -> dict[str, float | int | dict[str, float]]:
        payload: dict[str, float | int | dict[str, float]] = asdict(self)
        payload["batch_environment_steps"] = payload.pop("environment_steps")
        payload["timings"] = self.timings.to_mapping()
        return payload


def _elapsed_ms(started_ns: int) -> float:
    return (time.perf_counter_ns() - started_ns) / 1_000_000.0


def _require_finite(stage: str, values: tuple[tuple[str, Tensor], ...]) -> None:
    invalid = [
        name
        for name, value in values
        if not bool(torch.isfinite(value).all().item())
    ]
    if invalid:
        raise FloatingPointError(f"non-finite learner {stage}: {', '.join(invalid)}")


class VTraceLearner:
    """Consume FIFO unrolls once; no replay sampling or priority updates."""

    def __init__(
        self,
        *,
        model: RecurrentCandidateModel,
        encoder: GroundedObservationEncoder,
        optimizer: torch.optim.Optimizer,
        config: OptimizationConfig,
        maximum_unroll_length: int,
        maximum_policy_lag: int,
    ) -> None:
        if maximum_unroll_length <= 0 or maximum_policy_lag <= 0:
            raise ValueError("unroll length and policy lag limits must be positive")
        self.model = model
        self.encoder = encoder
        self.optimizer = optimizer
        self.config = config
        self.maximum_unroll_length = maximum_unroll_length
        self.maximum_policy_lag = maximum_policy_lag

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def update(
        self,
        unrolls: tuple[SequenceUnroll, ...],
        *,
        current_policy_version: int,
        progress: Callable[[str, dict[str, int | float]], None] | None = None,
    ) -> LearnerMetrics:
        total_started_ns = time.perf_counter_ns()

        def report(stage: str, **fields: int | float) -> None:
            if progress is not None:
                progress(
                    stage,
                    {"elapsed_ms": _elapsed_ms(total_started_ns), **fields},
                )

        if not unrolls:
            raise ValueError("learner unroll batch cannot be empty")
        if isinstance(current_policy_version, bool) or not isinstance(
            current_policy_version, int
        ):
            raise TypeError("current_policy_version must be an integer")
        if current_policy_version < 0:
            raise ValueError("current_policy_version must be non-negative")

        validation_started_ns = time.perf_counter_ns()
        encoding_fingerprint = grounding_encoding_identity()["fingerprint_sha256"]
        lags: list[int] = []
        for unroll in unrolls:
            unroll.validate(
                expected_config=self.encoder.config,
                expected_fingerprint=encoding_fingerprint,
                recurrent_hidden_dim=self.model.config.recurrent_hidden_dim,
                maximum_length=self.maximum_unroll_length,
            )
            lag = current_policy_version - unroll.policy_version
            if lag < 0:
                raise ValueError("rollout policy version is newer than the learner")
            if lag > self.maximum_policy_lag:
                raise ValueError(
                    f"rollout policy lag {lag} exceeds limit {self.maximum_policy_lag}"
                )
            lags.append(lag)
        validation_ms = _elapsed_ms(validation_started_ns)
        report("validation_complete", validation_ms=validation_ms)

        recurrent_started_ns = time.perf_counter_ns()
        batch_size = len(unrolls)
        maximum_time = max(len(unroll.steps) for unroll in unrolls)
        hidden = torch.stack(
            [
                torch.from_numpy(unroll.initial_recurrent_state.copy())
                for unroll in unrolls
            ],
            dim=0,
        ).to(device=self.device, dtype=next(self.model.parameters()).dtype)

        log_prob_rows: list[Tensor] = []
        value_rows: list[Tensor] = []
        entropy_rows: list[Tensor] = []
        valid_rows: list[Tensor] = []
        policy_rows: list[Tensor] = []
        behavior_rows: list[Tensor] = []
        reward_rows: list[Tensor] = []
        discount_rows: list[Tensor] = []

        self.model.train()
        for time_index in range(maximum_time):
            active = [
                index
                for index, unroll in enumerate(unrolls)
                if time_index < len(unroll.steps)
            ]
            active_tensor = torch.tensor(active, device=self.device, dtype=torch.long)
            snapshots = tuple(unrolls[index].steps[time_index].snapshot for index in active)
            encoded = collate_encoded_snapshots(
                snapshots,
                expected_config=self.encoder.config,
                expected_fingerprint=encoding_fingerprint,
                device=self.device,
            )
            output = self.model(
                encoded,
                hidden.index_select(0, active_tensor),
                validate=False,
            )
            hidden = hidden.index_copy(0, active_tensor, output.recurrent_state)
            log_policy = F.log_softmax(output.policy_logits.float(), dim=-1)
            policy = output.policy_probabilities()
            selected = torch.tensor(
                [unrolls[index].steps[time_index].action_index for index in active],
                device=self.device,
                dtype=torch.long,
            )
            selected_log_prob = log_policy.gather(1, selected[:, None]).squeeze(1)
            entropy = -(policy * log_policy).sum(dim=-1)

            floating_zero = output.value.new_zeros(batch_size)
            bool_zero = torch.zeros(batch_size, device=self.device, dtype=torch.bool)
            log_prob_rows.append(
                floating_zero.index_copy(0, active_tensor, selected_log_prob)
            )
            value_rows.append(floating_zero.index_copy(0, active_tensor, output.value))
            entropy_rows.append(floating_zero.index_copy(0, active_tensor, entropy))
            valid_rows.append(
                bool_zero.index_fill(0, active_tensor, True)
            )
            policy_rows.append(
                bool_zero.index_copy(
                    0,
                    active_tensor,
                    torch.tensor(
                        [
                            unrolls[index].steps[time_index].policy_decision
                            for index in active
                        ],
                        device=self.device,
                        dtype=torch.bool,
                    ),
                )
            )
            behavior_rows.append(
                floating_zero.index_copy(
                    0,
                    active_tensor,
                    torch.tensor(
                        [
                            unrolls[index]
                            .steps[time_index]
                            .behavior_log_probability
                            for index in active
                        ],
                        device=self.device,
                        dtype=torch.float32,
                    ),
                )
            )
            reward_rows.append(
                floating_zero.index_copy(
                    0,
                    active_tensor,
                    torch.tensor(
                        [unrolls[index].steps[time_index].reward for index in active],
                        device=self.device,
                        dtype=torch.float32,
                    ),
                )
            )
            discount_rows.append(
                floating_zero.index_copy(
                    0,
                    active_tensor,
                    torch.tensor(
                        [
                            unrolls[index].steps[time_index].discount
                            for index in active
                        ],
                        device=self.device,
                        dtype=torch.float32,
                    ),
                )
            )
            completed_steps = time_index + 1
            if completed_steps == 1 or completed_steps == maximum_time or completed_steps % 4 == 0:
                report(
                    "recurrent_forward_progress",
                    completed_time_steps=completed_steps,
                    maximum_time_steps=maximum_time,
                )

        bootstrap_values = hidden.new_zeros(batch_size)
        bootstrapped = [
            index
            for index, unroll in enumerate(unrolls)
            if unroll.bootstrap_snapshot is not None
        ]
        if bootstrapped:
            bootstrap_indices = torch.tensor(
                bootstrapped, device=self.device, dtype=torch.long
            )
            bootstrap_snapshots = []
            for index in bootstrapped:
                snapshot = unrolls[index].bootstrap_snapshot
                if snapshot is None:  # pragma: no cover - filtered above
                    raise RuntimeError("bootstrap index has no snapshot")
                bootstrap_snapshots.append(snapshot)
            bootstrap_batch = collate_encoded_snapshots(
                tuple(bootstrap_snapshots),
                expected_config=self.encoder.config,
                expected_fingerprint=encoding_fingerprint,
                device=self.device,
            )
            bootstrap_output = self.model(
                bootstrap_batch,
                hidden.index_select(0, bootstrap_indices),
                validate=False,
            )
            bootstrap_values = bootstrap_values.index_copy(
                0,
                bootstrap_indices,
                bootstrap_output.value,
            )
        recurrent_forward_ms = _elapsed_ms(recurrent_started_ns)
        report("recurrent_forward_complete", recurrent_forward_ms=recurrent_forward_ms)

        target_started_ns = time.perf_counter_ns()
        log_probs = torch.stack(log_prob_rows)
        values = torch.stack(value_rows)
        entropies = torch.stack(entropy_rows)
        valid = torch.stack(valid_rows)
        policy_decisions = torch.stack(policy_rows) & valid
        behavior_log_probs = torch.stack(behavior_rows)
        rewards = torch.stack(reward_rows)
        discounts = torch.stack(discount_rows)

        ratios = torch.exp((log_probs - behavior_log_probs).clamp(-20.0, 20.0))
        rho = ratios.clamp(max=self.config.vtrace_rho_clip)
        trace_c = ratios.clamp(max=self.config.vtrace_c_clip)
        policy_rho = ratios.clamp(max=self.config.policy_rho_clip)

        value_targets = torch.zeros_like(values)
        next_targets = torch.zeros_like(values)
        next_baseline = bootstrap_values
        next_vtrace = bootstrap_values
        for time_index in range(maximum_time - 1, -1, -1):
            active_mask = valid[time_index]
            next_targets[time_index] = torch.where(
                active_mask, next_vtrace, torch.zeros_like(next_vtrace)
            )
            delta = rho[time_index] * (
                rewards[time_index]
                + discounts[time_index] * next_baseline
                - values[time_index]
            )
            candidate_target = (
                values[time_index]
                + delta
                + discounts[time_index]
                * trace_c[time_index]
                * (next_vtrace - next_baseline)
            )
            value_targets[time_index] = torch.where(
                active_mask, candidate_target, torch.zeros_like(candidate_target)
            )
            next_baseline = torch.where(
                active_mask, values[time_index], next_baseline
            )
            next_vtrace = torch.where(active_mask, candidate_target, next_vtrace)

        advantages = policy_rho * (
            rewards + discounts * next_targets - values
        )
        valid_float = valid.to(dtype=values.dtype)
        policy_float = policy_decisions.to(dtype=values.dtype)
        policy_denominator = policy_float.sum().clamp_min(1.0)
        value_denominator = valid_float.sum().clamp_min(1.0)
        policy_loss = -(
            log_probs * advantages.detach() * policy_float
        ).sum() / policy_denominator
        value_loss = (
            F.smooth_l1_loss(
                values,
                value_targets.detach(),
                reduction="none",
            )
            * valid_float
        ).sum() / value_denominator
        entropy = (entropies * policy_float).sum() / policy_denominator
        total_loss = (
            self.config.policy_weight * policy_loss
            + self.config.value_weight * value_loss
            - self.config.entropy_weight * entropy
        )
        _require_finite(
            "targets/loss",
            (
                ("value_targets", value_targets),
                ("advantages", advantages),
                ("loss", total_loss),
            ),
        )
        target_and_loss_ms = _elapsed_ms(target_started_ns)
        report("targets_complete", target_and_loss_ms=target_and_loss_ms)

        backward_started_ns = time.perf_counter_ns()
        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        gradients = tuple(
            (name, parameter.grad)
            for name, parameter in self.model.named_parameters()
            if parameter.grad is not None
        )
        _require_finite("gradients", gradients)
        gradient_norm_tensor = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.gradient_clip_norm,
            error_if_nonfinite=True,
        )
        gradient_norm = float(gradient_norm_tensor.item())
        backward_ms = _elapsed_ms(backward_started_ns)
        report("backward_complete", backward_ms=backward_ms)

        optimizer_started_ns = time.perf_counter_ns()
        self.optimizer.step()
        _require_finite(
            "parameters",
            tuple((name, parameter) for name, parameter in self.model.named_parameters()),
        )
        optimizer_step_ms = _elapsed_ms(optimizer_started_ns)
        total_ms = _elapsed_ms(total_started_ns)
        report(
            "optimizer_complete",
            optimizer_step_ms=optimizer_step_ms,
            total_ms=total_ms,
        )

        active_ratios = ratios[valid]
        active_advantages = advantages[policy_decisions]
        active_targets = value_targets[valid]
        clipped = (ratios[valid] > self.config.vtrace_rho_clip).float()
        timings = LearnerTimings(
            validation_ms=validation_ms,
            recurrent_forward_ms=recurrent_forward_ms,
            target_and_loss_ms=target_and_loss_ms,
            backward_ms=backward_ms,
            optimizer_step_ms=optimizer_step_ms,
            total_ms=total_ms,
        )
        return LearnerMetrics(
            loss=float(total_loss.detach().item()),
            policy_loss=float(policy_loss.detach().item()),
            value_loss=float(value_loss.detach().item()),
            entropy=float(entropy.detach().item()),
            advantage_mean=(
                float(active_advantages.detach().mean().item())
                if active_advantages.numel()
                else 0.0
            ),
            value_target_mean=float(active_targets.detach().mean().item()),
            importance_ratio_mean=float(active_ratios.detach().mean().item()),
            importance_ratio_max=float(active_ratios.detach().max().item()),
            importance_clip_fraction=float(clipped.detach().mean().item()),
            gradient_norm=gradient_norm,
            unrolls=len(unrolls),
            environment_steps=int(valid.sum().item()),
            policy_decisions=int(policy_decisions.sum().item()),
            maximum_policy_lag=max(lags),
            timings=timings,
        )


__all__ = ["LearnerMetrics", "LearnerTimings", "VTraceLearner"]
