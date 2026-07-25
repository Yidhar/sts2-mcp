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
from sts2_rl.models import RecurrentCandidateModel, RecurrentCandidateOutput

from .config import (
    EpisodicLearningConfig,
    OptimizationConfig,
    TransactionLearningConfig,
)
from .episode_replay import HorizonTargets, ReplaySequence
from .transaction import (
    TransactionPolicyTarget,
    TransactionTrace,
    factual_transaction_policy_targets,
    observed_outcome_pairs,
    selection_delta_index,
)


@dataclass(frozen=True, slots=True)
class LearnerTimings:
    validation_ms: float
    recurrent_forward_ms: float
    target_and_loss_ms: float
    backward_ms: float
    episodic_replay_ms: float
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
    entropy_weight: float
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
    transaction_effect_loss: float
    transaction_delta_loss: float
    transaction_q_loss: float
    transaction_pairwise_ranking_loss: float
    transaction_completion_policy_loss: float
    transaction_traces: int
    transaction_effect_labels: int
    transaction_q_labels: int
    transaction_pairs: int
    transaction_policy_labels: int
    transaction_policy_preferred_labels: int
    transaction_policy_avoided_labels: int
    episodic_loss: float
    episodic_primary_policy_loss: float
    episodic_task_value_loss: float
    episodic_revival_value_loss: float
    episodic_revival_policy_loss: float
    episodic_sequences: int
    episodic_burn_in_steps: int
    episodic_learn_steps: int
    episodic_policy_labels: int
    episodic_failure_policy_suppressed_labels: int
    episodic_policy_lag_suppressed_labels: int
    episodic_task_value_labels: int
    episodic_revival_value_labels: int
    episodic_efficiency_policy_labels: int
    episodic_importance_ratio_mean: float
    episodic_importance_ratio_max: float
    episodic_importance_clip_fraction: float
    episodic_maximum_policy_lag: int
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


def _annealed_entropy_weight(
    config: OptimizationConfig,
    *,
    policy_version: int,
) -> float:
    """Return the explicit non-zero entropy schedule for this update."""

    start = float(config.entropy_weight)
    end = float(config.entropy_weight_end)
    decay_updates = int(config.entropy_decay_updates)
    progress = min(1.0, max(0.0, float(policy_version) / float(decay_updates)))
    return start + progress * (end - start)


@dataclass(frozen=True, slots=True)
class _EpisodicLossBatch:
    total_loss: Tensor
    primary_policy_loss: Tensor
    task_value_loss: Tensor
    revival_value_loss: Tensor
    revival_policy_loss: Tensor
    burn_in_steps: int
    learn_steps: int
    policy_labels: int
    failure_policy_suppressed_labels: int
    policy_lag_suppressed_labels: int
    task_value_labels: int
    revival_value_labels: int
    efficiency_policy_labels: int
    importance_ratios: Tensor
    maximum_policy_lag: int


class VTraceLearner:
    """Consume main-policy unrolls FIFO with an optional factual transaction sidecar."""

    def __init__(
        self,
        *,
        model: RecurrentCandidateModel,
        encoder: GroundedObservationEncoder,
        optimizer: torch.optim.Optimizer,
        config: OptimizationConfig,
        maximum_unroll_length: int,
        maximum_policy_lag: int,
        transaction_config: TransactionLearningConfig | None = None,
        episodic_config: EpisodicLearningConfig | None = None,
    ) -> None:
        if maximum_unroll_length <= 0 or maximum_policy_lag <= 0:
            raise ValueError("unroll length and policy lag limits must be positive")
        self.model = model
        self.encoder = encoder
        self.optimizer = optimizer
        self.config = config
        self.maximum_unroll_length = maximum_unroll_length
        self.maximum_policy_lag = maximum_policy_lag
        self.transaction_config = transaction_config or TransactionLearningConfig()
        self.episodic_config = episodic_config or EpisodicLearningConfig()
        if self.transaction_config.enabled != self.model.transaction_heads_enabled:
            raise ValueError(
                "transaction learner config and model-head configuration differ"
            )

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def update(
        self,
        unrolls: tuple[SequenceUnroll, ...],
        *,
        current_policy_version: int,
        transaction_traces: tuple[TransactionTrace, ...] = (),
        episodic_sequences: tuple[ReplaySequence, ...] = (),
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
        if not isinstance(transaction_traces, tuple) or not all(
            isinstance(trace, TransactionTrace) for trace in transaction_traces
        ):
            raise TypeError("transaction_traces must be a TransactionTrace tuple")
        if transaction_traces and not self.transaction_config.enabled:
            raise ValueError("transaction traces require transaction learning to be enabled")
        if not isinstance(episodic_sequences, tuple) or not all(
            isinstance(sequence, ReplaySequence) for sequence in episodic_sequences
        ):
            raise TypeError("episodic_sequences must be a ReplaySequence tuple")
        if episodic_sequences and not self.episodic_config.enabled:
            raise ValueError("episodic sequences require episodic learning to be enabled")

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
            log_policy = output.policy_log_probabilities()
            selected = torch.tensor(
                [unrolls[index].steps[time_index].action_index for index in active],
                device=self.device,
                dtype=torch.long,
            )
            selected_log_prob = log_policy.gather(1, selected[:, None]).squeeze(1)
            entropy = output.policy_entropy()

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
        entropy_weight = _annealed_entropy_weight(
            self.config,
            policy_version=current_policy_version,
        )
        total_loss = (
            self.config.policy_weight * policy_loss
            + self.config.value_weight * value_loss
            - entropy_weight * entropy
        )
        (
            transaction_effect_loss,
            transaction_delta_loss,
            transaction_q_loss,
            transaction_pairwise_loss,
            transaction_completion_policy_loss,
            transaction_effect_labels,
            transaction_q_labels,
            transaction_pair_count,
            transaction_policy_labels,
            transaction_policy_preferred_labels,
            transaction_policy_avoided_labels,
        ) = self._transaction_losses(transaction_traces)
        total_loss = (
            total_loss
            + self.transaction_config.effect_weight
            * (transaction_effect_loss + transaction_delta_loss)
            + self.transaction_config.transaction_q_weight * transaction_q_loss
            + self.transaction_config.pairwise_ranking_weight
            * transaction_pairwise_loss
            + self.transaction_config.completion_policy_weight
            * transaction_completion_policy_loss
        )
        _require_finite(
            "targets/loss",
            (
                ("value_targets", value_targets),
                ("advantages", advantages),
                ("loss", total_loss),
                ("transaction_effect_loss", transaction_effect_loss),
                ("transaction_delta_loss", transaction_delta_loss),
                ("transaction_q_loss", transaction_q_loss),
                ("transaction_pairwise_loss", transaction_pairwise_loss),
                (
                    "transaction_completion_policy_loss",
                    transaction_completion_policy_loss,
                ),
            ),
        )
        target_and_loss_ms = _elapsed_ms(target_started_ns)
        report("targets_complete", target_and_loss_ms=target_and_loss_ms)

        backward_started_ns = time.perf_counter_ns()
        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        episodic_started_ns = time.perf_counter_ns()
        episodic_losses = self._episodic_losses(
            episodic_sequences,
            current_policy_version=current_policy_version,
        )
        if episodic_sequences:
            _require_finite(
                "episodic targets/loss",
                (
                    ("episodic_loss", episodic_losses.total_loss),
                    (
                        "episodic_primary_policy_loss",
                        episodic_losses.primary_policy_loss,
                    ),
                    ("episodic_task_value_loss", episodic_losses.task_value_loss),
                    (
                        "episodic_revival_value_loss",
                        episodic_losses.revival_value_loss,
                    ),
                    (
                        "episodic_revival_policy_loss",
                        episodic_losses.revival_policy_loss,
                    ),
                ),
            )
            episodic_losses.total_loss.backward()  # type: ignore[no-untyped-call]
        episodic_replay_ms = _elapsed_ms(episodic_started_ns)
        report(
            "episodic_replay_complete",
            episodic_replay_ms=episodic_replay_ms,
            episodic_sequences=len(episodic_sequences),
            episodic_burn_in_steps=episodic_losses.burn_in_steps,
            episodic_learn_steps=episodic_losses.learn_steps,
        )
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
            episodic_replay_ms=episodic_replay_ms,
            optimizer_step_ms=optimizer_step_ms,
            total_ms=total_ms,
        )
        return LearnerMetrics(
            loss=float(
                total_loss.detach().item()
                + episodic_losses.total_loss.detach().item()
            ),
            policy_loss=float(policy_loss.detach().item()),
            value_loss=float(value_loss.detach().item()),
            entropy=float(entropy.detach().item()),
            entropy_weight=entropy_weight,
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
            transaction_effect_loss=float(transaction_effect_loss.detach().item()),
            transaction_delta_loss=float(transaction_delta_loss.detach().item()),
            transaction_q_loss=float(transaction_q_loss.detach().item()),
            transaction_pairwise_ranking_loss=float(
                transaction_pairwise_loss.detach().item()
            ),
            transaction_completion_policy_loss=float(
                transaction_completion_policy_loss.detach().item()
            ),
            transaction_traces=len(transaction_traces),
            transaction_effect_labels=transaction_effect_labels,
            transaction_q_labels=transaction_q_labels,
            transaction_pairs=transaction_pair_count,
            transaction_policy_labels=transaction_policy_labels,
            transaction_policy_preferred_labels=transaction_policy_preferred_labels,
            transaction_policy_avoided_labels=transaction_policy_avoided_labels,
            episodic_loss=float(episodic_losses.total_loss.detach().item()),
            episodic_primary_policy_loss=float(
                episodic_losses.primary_policy_loss.detach().item()
            ),
            episodic_task_value_loss=float(
                episodic_losses.task_value_loss.detach().item()
            ),
            episodic_revival_value_loss=float(
                episodic_losses.revival_value_loss.detach().item()
            ),
            episodic_revival_policy_loss=float(
                episodic_losses.revival_policy_loss.detach().item()
            ),
            episodic_sequences=len(episodic_sequences),
            episodic_burn_in_steps=episodic_losses.burn_in_steps,
            episodic_learn_steps=episodic_losses.learn_steps,
            episodic_policy_labels=episodic_losses.policy_labels,
            episodic_failure_policy_suppressed_labels=(
                episodic_losses.failure_policy_suppressed_labels
            ),
            episodic_policy_lag_suppressed_labels=(
                episodic_losses.policy_lag_suppressed_labels
            ),
            episodic_task_value_labels=episodic_losses.task_value_labels,
            episodic_revival_value_labels=(
                episodic_losses.revival_value_labels
            ),
            episodic_efficiency_policy_labels=(
                episodic_losses.efficiency_policy_labels
            ),
            episodic_importance_ratio_mean=(
                float(episodic_losses.importance_ratios.detach().mean().item())
                if episodic_losses.importance_ratios.numel()
                else 0.0
            ),
            episodic_importance_ratio_max=(
                float(episodic_losses.importance_ratios.detach().max().item())
                if episodic_losses.importance_ratios.numel()
                else 0.0
            ),
            episodic_importance_clip_fraction=(
                float(
                    (
                        episodic_losses.importance_ratios
                        > self.episodic_config.importance_ratio_clip
                    )
                    .float()
                    .mean()
                    .item()
                )
                if episodic_losses.importance_ratios.numel()
                else 0.0
            ),
            episodic_maximum_policy_lag=episodic_losses.maximum_policy_lag,
            timings=timings,
        )

    @staticmethod
    def _episodic_task_target(target: HorizonTargets) -> float:
        """Return a completion-dominant, cost-free primary target.

        ``task_return`` contains only the base terminal/progress reward captured
        by the collector; revival, HP-loss, and pace shaping are deliberately
        absent.  The explicit boundary outcome is added once so a successful
        combat/Act remains supervised even when no run-progress reward happened
        inside that short horizon.  This outcome term is primary-task evidence,
        not a hand-authored action preference.
        """

        if (
            not target.observed
            or target.success is None
            or target.task_return is None
        ):
            raise ValueError("episodic task target is not observed")
        return float(target.task_return + (1.0 if target.success else -1.0))

    @staticmethod
    def _horizon_outputs(
        output: RecurrentCandidateOutput,
        row: int,
        *,
        combat: HorizonTargets,
        act: HorizonTargets,
        run: HorizonTargets,
    ) -> tuple[tuple[str, HorizonTargets, Tensor, Tensor], ...]:
        return (
            (
                "combat",
                combat,
                output.combat_task_value[row],
                output.combat_revival_cost_value[row],
            ),
            (
                "act",
                act,
                output.act_task_value[row],
                output.act_revival_cost_value[row],
            ),
            (
                "run",
                run,
                output.run_task_value[row],
                output.run_revival_cost_value[row],
            ),
        )

    def _episodic_losses(
        self,
        sequences: tuple[ReplaySequence, ...],
        *,
        current_policy_version: int,
    ) -> _EpisodicLossBatch:
        """Compute episodic losses with deterministic recurrent reconstruction.

        Episodic sequences deliberately omit recurrently irrelevant combat
        prefixes.  Dropout would nevertheless consume a different RNG stream
        for the sparse and full histories and would make the supposedly exact
        reconstruction depend on which irrelevant steps were skipped.  Run
        both the no-grad prefix and differentiable suffix in evaluation mode so
        the sparse replay has deterministic semantics, while preserving the
        caller's model mode even when validation or collation raises.

        Evaluation mode does not disable autograd.  The learning suffix still
        owns the complete bounded graph; only dropout is disabled (the active
        model contains no batch-normalization state).
        """

        was_training = self.model.training
        self.model.eval()
        try:
            return self._episodic_losses_deterministic(
                sequences,
                current_policy_version=current_policy_version,
            )
        finally:
            self.model.train(was_training)

    def _episodic_losses_deterministic(
        self,
        sequences: tuple[ReplaySequence, ...],
        *,
        current_policy_version: int,
    ) -> _EpisodicLossBatch:
        """Replay complete-episode labels through one bounded GPU graph.

        The caller holds the model in evaluation mode.  Every sequence first
        reconstructs the current model's split recurrent
        state from its exact sparse prefix under ``torch.no_grad()``.  Only the
        configured contiguous suffix is batched with autograd.  Thus a 30,000
        step source episode can increase CPU storage and no-grad compute, but it
        cannot increase activation memory beyond
        ``len(sequences) * episodic_config.learn_steps``.

        Successful-horizon selected-action replay is clipped by the factual
        behavior probability. Failed horizons remain authoritative value
        targets but never become blanket anti-imitation policy labels. Revival
        cost can affect policy only for a successfully completed horizon.
        Outside a configured, success-classified primary
        tie band, the already-weighted cost signal is capped below the absolute
        primary residual.  Inside that narrow band, a small nominal-primary
        floor keeps the cost tie-break alive even when the task advantage is
        calibrated to zero.  This implements the staged ordering ``complete
        first, then reduce revivals`` instead of silently deleting the
        secondary objective at primary convergence.
        """

        zero = next(self.model.parameters()).sum() * 0.0
        empty_ratios = torch.empty(0, device=self.device, dtype=torch.float32)
        if not sequences:
            return _EpisodicLossBatch(
                total_loss=zero,
                primary_policy_loss=zero,
                task_value_loss=zero,
                revival_value_loss=zero,
                revival_policy_loss=zero,
                burn_in_steps=0,
                learn_steps=0,
                policy_labels=0,
                failure_policy_suppressed_labels=0,
                policy_lag_suppressed_labels=0,
                task_value_labels=0,
                revival_value_labels=0,
                efficiency_policy_labels=0,
                importance_ratios=empty_ratios,
                maximum_policy_lag=0,
            )

        fingerprint = grounding_encoding_identity()["fingerprint_sha256"]
        hidden_rows: list[Tensor] = []
        total_burn_in_steps = 0
        total_learn_steps = 0
        maximum_policy_lag = 0
        for sequence in sequences:
            if not sequence.exact_recurrent_reconstruction:
                raise ValueError("episodic replay sequence is not an exact reconstruction")
            if sequence.configured_burn_in_steps != self.episodic_config.burn_in_steps:
                raise ValueError("episodic replay burn-in configuration differs from learner")
            if not 0 < len(sequence.learn_steps) <= self.episodic_config.learn_steps:
                raise ValueError("episodic replay learning suffix exceeds configured length")
            hidden = self.model.initial_state(1, device=self.device)
            with torch.no_grad():
                for step in sequence.burn_in:
                    step.snapshot.validate(
                        expected_config=self.encoder.config,
                        expected_fingerprint=fingerprint,
                    )
                    encoded = collate_encoded_snapshots(
                        (step.snapshot,),
                        expected_config=self.encoder.config,
                        expected_fingerprint=fingerprint,
                        device=self.device,
                    )
                    hidden = self.model(encoded, hidden, validate=False).recurrent_state
            hidden_rows.append(hidden[0].detach())
            total_burn_in_steps += len(sequence.burn_in)
            total_learn_steps += len(sequence.learn_steps)
            for step in sequence.learn_steps:
                lag = current_policy_version - step.decision.policy_version
                if lag < 0:
                    raise ValueError(
                        "episodic behavior policy version is newer than the learner"
                    )
                maximum_policy_lag = max(maximum_policy_lag, lag)

        hidden = torch.stack(hidden_rows, dim=0)
        maximum_time = max(len(sequence.learn_steps) for sequence in sequences)
        task_predictions: list[Tensor] = []
        task_targets: list[float] = []
        revival_predictions: list[Tensor] = []
        revival_targets: list[float] = []
        primary_policy_terms: list[Tensor] = []
        revival_policy_terms: list[Tensor] = []
        combined_policy_terms: list[Tensor] = []
        importance_ratios: list[Tensor] = []
        efficiency_policy_labels = 0
        failure_policy_suppressed_labels = 0
        policy_lag_suppressed_labels = 0

        for time_index in range(maximum_time):
            active = [
                index
                for index, sequence in enumerate(sequences)
                if time_index < len(sequence.learn_steps)
            ]
            active_tensor = torch.tensor(active, device=self.device, dtype=torch.long)
            active_steps = tuple(
                sequences[index].learn_steps[time_index] for index in active
            )
            snapshots = tuple(step.snapshot for step in active_steps)
            encoded = collate_encoded_snapshots(
                snapshots,
                expected_config=self.encoder.config,
                expected_fingerprint=fingerprint,
                device=self.device,
            )
            output = self.model(
                encoded,
                hidden.index_select(0, active_tensor),
                validate=False,
            )
            hidden = hidden.index_copy(0, active_tensor, output.recurrent_state)
            log_policy = output.policy_log_probabilities()

            for row, step in enumerate(active_steps):
                decision = step.decision
                action_index = decision.action_index
                selected_log_probability = log_policy[row, action_index]
                horizons = self._horizon_outputs(
                    output,
                    row,
                    combat=step.combat,
                    act=step.act,
                    run=step.run,
                )

                for _, target, task_prediction, revival_prediction in horizons:
                    if not target.observed:
                        continue
                    task_predictions.append(task_prediction)
                    task_targets.append(self._episodic_task_target(target))
                    if target.efficiency_eligible:
                        if target.future_revivals is None:  # pragma: no cover - invariant
                            raise RuntimeError("successful horizon has no revival target")
                        revival_predictions.append(revival_prediction)
                        revival_targets.append(float(target.future_revivals))

                if not decision.policy_decision:
                    continue
                policy_lag = current_policy_version - decision.policy_version
                if policy_lag > self.episodic_config.policy_gradient_max_lag:
                    # Complete episodes remain authoritative long-horizon value
                    # supervision after their behavior policy becomes stale.
                    # They must not, however, keep applying selected-action
                    # likelihood gradients to a policy hundreds of updates
                    # newer than the one that generated those decisions.
                    policy_lag_suppressed_labels += 1
                    continue
                primary = next(
                    (
                        item
                        for item in reversed(horizons)
                        if item[1].observed
                    ),
                    None,
                )
                if primary is None:
                    continue
                _, primary_target, primary_prediction, _ = primary
                if primary_target.success is not True:
                    # A failed long horizon is authoritative evidence for the
                    # task-value heads, but it is not a counterfactual action
                    # label. Applying a negative selected-action likelihood
                    # update to every decision in a long failed run performs
                    # anti-imitation of the entire factual trajectory. With a
                    # hierarchical policy this systematically suppresses the
                    # frequently sampled, many-candidate branches and leaks
                    # probability into unsampled singleton branches (for
                    # example END_TURN). FIFO V-trace and bounded factual
                    # transaction failures still provide local policy
                    # gradients; complete-episode replay reinforces policy
                    # only after an observed successful horizon.
                    failure_policy_suppressed_labels += 1
                    continue
                raw_ratio = torch.exp(
                    (
                        selected_log_probability
                        - float(decision.behavior_log_probability)
                    ).clamp(-20.0, 20.0)
                )
                # Importance sampling corrects the factual behavior/current
                # distribution mismatch; it is not itself a differentiable
                # policy objective.  Letting gradients flow through rho would
                # add a ``rho * log(pi)`` product-rule term and can reverse a
                # positive-advantage update whenever ``log(pi) < -1``.
                detached_ratio = raw_ratio.detach()
                importance_ratios.append(detached_ratio)
                clipped_ratio = detached_ratio.clamp(
                    max=self.episodic_config.importance_ratio_clip
                )
                primary_advantage = (
                    primary_prediction.new_tensor(
                        self._episodic_task_target(primary_target)
                    )
                    - primary_prediction
                )
                primary_signal = (
                    self.episodic_config.primary_policy_weight
                    * primary_advantage.detach()
                )
                primary_policy_terms.append(
                    -clipped_ratio
                    * selected_log_probability
                    * primary_advantage.detach()
                )

                secondary_signal = primary_signal.new_zeros(())
                efficiency = next(
                    (
                        item
                        for item in reversed(horizons)
                        if item[1].efficiency_eligible
                    ),
                    None,
                )
                if efficiency is not None:
                    _, efficiency_target, _, cost_prediction = efficiency
                    if efficiency_target.future_revivals is None:  # pragma: no cover
                        raise RuntimeError("successful horizon has no revival target")
                    cost_advantage = (
                        cost_prediction.new_tensor(
                            float(efficiency_target.future_revivals)
                        )
                        - cost_prediction
                    )
                    raw_secondary_signal = (
                        self.episodic_config.revival_policy_weight
                        * cost_advantage.detach()
                    )
                    # A pure ``fraction * abs(primary_advantage)`` cap makes
                    # the revival objective exactly zero once the task value
                    # is calibrated.  That prevents successful 0-revival and
                    # 100-revival paths from ever becoming distinguishable at
                    # the point where primary outcomes tie.
                    #
                    # Before the state value is on the successful side of the
                    # signed task target, or while its residual lies outside
                    # the explicit tie tolerance, retain the strict
                    # residual-relative cap.  Only inside that narrow success
                    # tie stratum admit a fraction of one nominal primary
                    # policy unit. Failure/censored horizons never enter this
                    # branch at all.
                    primary_tie = (
                        (primary_prediction.detach() > 0.0)
                        & (
                            primary_advantage.detach().abs()
                            <= self.episodic_config.primary_success_tie_tolerance
                        )
                    ).to(dtype=primary_signal.dtype)
                    success_tie_floor = primary_signal.new_tensor(
                        self.episodic_config.primary_policy_weight
                    ) * primary_tie
                    protected_primary_scale = torch.maximum(
                        primary_signal.abs(),
                        success_tie_floor,
                    )
                    secondary_limit = (
                        self.episodic_config.secondary_advantage_fraction
                        * protected_primary_scale
                    )
                    secondary_signal = torch.maximum(
                        torch.minimum(raw_secondary_signal, secondary_limit),
                        -secondary_limit,
                    )
                    revival_policy_terms.append(
                        clipped_ratio
                        * selected_log_probability
                        * secondary_signal
                    )
                    efficiency_policy_labels += 1

                combined_policy_terms.append(
                    -clipped_ratio
                    * selected_log_probability
                    * (primary_signal - secondary_signal)
                )

        task_value_loss = (
            F.smooth_l1_loss(
                torch.stack(task_predictions).float(),
                torch.tensor(task_targets, device=self.device, dtype=torch.float32),
            )
            if task_predictions
            else zero
        )
        # Revival counts have an extremely long tail under unlimited native
        # revival.  Regressing raw counts makes this auxiliary value head
        # numerically dominate the complete-episode objective even though its
        # policy advantage is already secondary.  A log1p observation model
        # keeps zero exact, remains monotone over factual counts, and prevents
        # a 100-revival path from contributing roughly 100x the representation
        # gradient of a one-revival path.
        revival_value_loss = (
            F.smooth_l1_loss(
                torch.log1p(torch.stack(revival_predictions).float()),
                torch.log1p(
                    torch.tensor(
                        revival_targets,
                        device=self.device,
                        dtype=torch.float32,
                    )
                ),
            )
            if revival_predictions
            else zero
        )
        primary_policy_loss = (
            torch.stack(primary_policy_terms).mean()
            if primary_policy_terms
            else zero
        )
        revival_policy_loss = (
            torch.stack(revival_policy_terms).mean()
            if revival_policy_terms
            else zero
        )
        combined_policy_loss = (
            torch.stack(combined_policy_terms).mean()
            if combined_policy_terms
            else zero
        )
        total_loss = (
            combined_policy_loss
            + self.episodic_config.task_value_weight * task_value_loss
            + self.episodic_config.revival_value_weight * revival_value_loss
        )
        ratio_tensor = (
            torch.stack(importance_ratios).float()
            if importance_ratios
            else empty_ratios
        )
        return _EpisodicLossBatch(
            total_loss=total_loss,
            primary_policy_loss=primary_policy_loss,
            task_value_loss=task_value_loss,
            revival_value_loss=revival_value_loss,
            revival_policy_loss=revival_policy_loss,
            burn_in_steps=total_burn_in_steps,
            learn_steps=total_learn_steps,
            policy_labels=len(primary_policy_terms),
            failure_policy_suppressed_labels=failure_policy_suppressed_labels,
            policy_lag_suppressed_labels=policy_lag_suppressed_labels,
            task_value_labels=len(task_predictions),
            revival_value_labels=len(revival_predictions),
            efficiency_policy_labels=efficiency_policy_labels,
            importance_ratios=ratio_tensor,
            maximum_policy_lag=maximum_policy_lag,
        )

    def _transaction_losses(
        self,
        traces: tuple[TransactionTrace, ...],
    ) -> tuple[
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        int,
        int,
        int,
        int,
        int,
        int,
    ]:
        """Compute factual transaction heads and direct policy liveness loss."""

        zero = next(self.model.parameters()).sum() * 0.0
        if not traces:
            return zero, zero, zero, zero, zero, 0, 0, 0, 0, 0, 0
        fingerprint = grounding_encoding_identity()["fingerprint_sha256"]
        effect_logits: list[Tensor] = []
        effect_targets: list[int] = []
        delta_logits: list[Tensor] = []
        delta_targets: list[int] = []
        q_values: list[Tensor] = []
        q_targets: list[float] = []
        selected_policy_log_probabilities: dict[tuple[int, int], Tensor] = {}
        completion_policy_trace_losses: list[Tensor] = []
        completion_policy_preferred_labels = 0
        completion_policy_avoided_labels = 0

        for trace_index, trace in enumerate(traces):
            configured_burn_in = self.transaction_config.burn_in_steps
            if trace.burn_in_steps > configured_burn_in:
                raise ValueError("transaction trace exceeds configured burn-in length")
            if trace.start_step > 0 and trace.burn_in_steps != configured_burn_in:
                raise ValueError(
                    "non-initial transaction context must provide the configured burn-in"
                )
            if trace.initial_recurrent_state.shape != (
                self.model.config.recurrent_hidden_dim,
            ):
                raise ValueError("transaction recurrent state differs from model hidden size")
            hidden = torch.from_numpy(trace.initial_recurrent_state.copy()).to(
                device=self.device,
                dtype=next(self.model.parameters()).dtype,
            )[None, :]
            policy_targets = {
                item.step_index: item.target
                for item in factual_transaction_policy_targets(trace)
            }
            trace_policy_losses: list[Tensor] = []
            for step_index, step in enumerate(trace.steps):
                step.snapshot.validate(
                    expected_config=self.encoder.config,
                    expected_fingerprint=fingerprint,
                )
                encoded = collate_encoded_snapshots(
                    (step.snapshot,),
                    expected_config=self.encoder.config,
                    expected_fingerprint=fingerprint,
                    device=self.device,
                )
                output = self.model(encoded, hidden, validate=False)
                hidden = output.recurrent_state
                if step_index + 1 == trace.burn_in_steps:
                    hidden = hidden.detach()
                if step_index < trace.burn_in_steps:
                    continue
                if (
                    output.candidate_effect_logits is None
                    or output.selection_delta_logits is None
                    or output.transaction_q_values is None
                ):
                    raise RuntimeError("transaction-enabled learner received a headless model")
                action_index = step.action_index
                effect_logits.append(output.candidate_effect_logits[0, action_index])
                effect_targets.append(int(step.effect))
                delta_logits.append(output.selection_delta_logits[0, action_index])
                delta_targets.append(selection_delta_index(step.selected_count_delta))
                # Both pairwise outcome ranking and factual liveness supervise
                # normalized legal-candidate preference, never a free logit
                # offset or a fabricated unexecuted action target.
                policy_log_probabilities = (
                    output.policy_log_probabilities()
                )
                selected_policy_log_probability = policy_log_probabilities[
                    0, action_index
                ]
                selected_policy_log_probabilities[(trace_index, step_index)] = (
                    selected_policy_log_probability
                )
                policy_target = policy_targets.get(step_index)
                if policy_target is TransactionPolicyTarget.PREFER:
                    trace_policy_losses.append(-selected_policy_log_probability)
                    completion_policy_preferred_labels += 1
                elif policy_target is TransactionPolicyTarget.AVOID:
                    other_action_indices = [
                        index
                        for index, enabled in enumerate(step.snapshot.action_mask)
                        if bool(enabled) and index != action_index
                    ]
                    if not other_action_indices:  # pragma: no cover - target invariant
                        raise RuntimeError(
                            "transaction avoid target has no legal alternative"
                        )
                    other_indices = torch.tensor(
                        other_action_indices,
                        device=self.device,
                        dtype=torch.long,
                    )
                    trace_policy_losses.append(
                        -torch.logsumexp(
                            policy_log_probabilities[0, other_indices],
                            dim=0,
                        )
                    )
                    completion_policy_avoided_labels += 1
                if step.q_observed:
                    if step.transaction_return is None:  # pragma: no cover - property invariant
                        raise RuntimeError("q_observed transaction has no return")
                    q_values.append(output.transaction_q_values[0, action_index])
                    q_targets.append(step.transaction_return)
            if trace_policy_losses:
                # Equal trace weight prevents a long repeated cycle from
                # overwhelming many short, factual completion paths.
                completion_policy_trace_losses.append(
                    torch.stack(trace_policy_losses).mean()
                )

        effect_loss = F.cross_entropy(
            torch.stack(effect_logits),
            torch.tensor(effect_targets, device=self.device, dtype=torch.long),
        )
        delta_loss = F.cross_entropy(
            torch.stack(delta_logits),
            torch.tensor(delta_targets, device=self.device, dtype=torch.long),
        )
        if q_values:
            q_loss = F.smooth_l1_loss(
                torch.stack(q_values).float(),
                torch.tensor(q_targets, device=self.device, dtype=torch.float32),
            )
        else:
            q_loss = zero

        pairs = (
            observed_outcome_pairs(
                traces,
                minimum_return_gap=self.transaction_config.minimum_return_gap,
            )[: self.transaction_config.maximum_pairs]
            if self.transaction_config.pairwise_ranking_weight > 0.0
            else ()
        )
        if pairs:
            better = torch.stack(
                [
                    selected_policy_log_probabilities[
                        (pair.better_trace, pair.better_step)
                    ]
                    for pair in pairs
                ]
            ).float()
            worse = torch.stack(
                [
                    selected_policy_log_probabilities[
                        (pair.worse_trace, pair.worse_step)
                    ]
                    for pair in pairs
                ]
            ).float()
            pairwise_loss = F.softplus(
                self.transaction_config.pairwise_margin - (better - worse)
            ).mean()
        else:
            pairwise_loss = zero
        completion_policy_loss = (
            torch.stack(completion_policy_trace_losses).mean()
            if completion_policy_trace_losses
            else zero
        )
        completion_policy_labels = (
            completion_policy_preferred_labels + completion_policy_avoided_labels
        )
        return (
            effect_loss,
            delta_loss,
            q_loss,
            pairwise_loss,
            completion_policy_loss,
            len(effect_targets),
            len(q_targets),
            len(pairs),
            completion_policy_labels,
            completion_policy_preferred_labels,
            completion_policy_avoided_labels,
        )


__all__ = ["LearnerMetrics", "LearnerTimings", "VTraceLearner"]
