"""Off-policy actor-critic learner for grounded legal candidates."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from sts2_baseline import ReplayBatch, StratifiedReplayBuffer, baseline_reward_identity
from sts2_rl.encoding import GroundedObservationEncoder
from sts2_rl.encoding.grounded import grounding_encoding_identity
from sts2_rl.models import GroundedCandidateModel

from .config import OptimizationConfig
from .experience import DecisionExperience


@dataclass(frozen=True, slots=True)
class LearnerMetrics:
    loss: float
    policy_loss: float
    value_loss: float
    q_loss: float
    reward_loss: float
    terminal_loss: float
    entropy: float
    advantage_mean: float
    td_error_mean: float
    gradient_norm: float

    def to_mapping(self) -> dict[str, float]:
        return {
            "loss": self.loss,
            "policy_loss": self.policy_loss,
            "value_loss": self.value_loss,
            "q_loss": self.q_loss,
            "reward_loss": self.reward_loss,
            "terminal_loss": self.terminal_loss,
            "entropy": self.entropy,
            "advantage_mean": self.advantage_mean,
            "td_error_mean": self.td_error_mean,
            "gradient_norm": self.gradient_norm,
        }


def _weighted_mean(values: Tensor, weights: Tensor) -> Tensor:
    if values.shape != weights.shape:
        raise ValueError(f"weighted mean shape mismatch: {values.shape} != {weights.shape}")
    return (values * weights).sum() / weights.sum().clamp_min(1e-8)


def _require_finite_tensors(
    stage: str,
    tensors: tuple[tuple[str, Tensor], ...],
) -> None:
    """Fail before mutation when any tensor in a learner stage is non-finite."""

    if not tensors:
        return
    checks_by_device: dict[torch.device, list[Tensor]] = {}
    for _, value in tensors:
        checks_by_device.setdefault(value.device, []).append(torch.isfinite(value).all())
    if all(
        bool(torch.stack(checks).all().item())
        for checks in checks_by_device.values()
    ):
        return
    invalid = [
        name
        for name, value in tensors
        if not bool(torch.isfinite(value).all().item())
    ]
    raise FloatingPointError(f"non-finite learner {stage}: {', '.join(invalid)}")


def _optimizer_tensors(optimizer: torch.optim.Optimizer) -> tuple[tuple[str, Tensor], ...]:
    tensors: list[tuple[str, Tensor]] = []
    for state_index, state in enumerate(optimizer.state.values()):
        for key, value in state.items():
            if isinstance(value, Tensor) and (value.is_floating_point() or value.is_complex()):
                tensors.append((f"optimizer.state[{state_index}].{key}", value))
    return tuple(tensors)


class GroundedLearner:
    """Learn policy/Q/value/outcome heads from refreshable replay samples."""

    def __init__(
        self,
        *,
        model: GroundedCandidateModel,
        encoder: GroundedObservationEncoder,
        optimizer: torch.optim.Optimizer,
        config: OptimizationConfig,
    ) -> None:
        self.model = model
        self.encoder = encoder
        self.optimizer = optimizer
        self.config = config

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def update(
        self,
        batch: ReplayBatch,
        *,
        replay: StratifiedReplayBuffer | None = None,
    ) -> LearnerMetrics:
        if not batch.samples:
            raise ValueError("learner batch cannot be empty")
        experiences: list[DecisionExperience] = []
        targets: list[float] = []
        rewards: list[float] = []
        active_encoding_fingerprint = grounding_encoding_identity()[
            "fingerprint_sha256"
        ]
        active_reward_fingerprint = baseline_reward_identity()["fingerprint_sha256"]
        for sample in batch.samples:
            if not isinstance(sample.payload, DecisionExperience):
                raise TypeError("replay sample payload is not DecisionExperience")
            if sample.targets.value is None:
                raise ValueError("online baseline replay requires a scalar return target")
            if sample.payload.encoding_fingerprint != active_encoding_fingerprint:
                raise ValueError("replay decision uses a different encoding contract")
            if sample.payload.reward_fingerprint != active_reward_fingerprint:
                raise ValueError("replay decision uses a different reward contract")
            experiences.append(sample.payload)
            targets.append(float(sample.targets.value))
            rewards.append(float(sample.targets.reward))

        decisions = [
            self.encoder.encode(
                experience.observation,
                experience.legal_actions,
                device=self.device,
            )
            for experience in experiences
        ]
        model_batch = self.encoder.stack(decisions)
        action_indices = torch.tensor(
            [experience.action_index for experience in experiences],
            dtype=torch.long,
            device=self.device,
        )
        row_indices = torch.arange(len(experiences), device=self.device)
        return_targets = torch.tensor(targets, dtype=torch.float32, device=self.device)
        reward_targets = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        behavior_log_probabilities = torch.tensor(
            [experience.behavior_log_probability for experience in experiences],
            dtype=torch.float32,
            device=self.device,
        )
        terminal_targets = torch.tensor(
            [experience.terminal_class for experience in experiences],
            dtype=torch.long,
            device=self.device,
        )
        weights = torch.as_tensor(
            np.asarray(batch.importance_weights, dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        )
        probabilities = torch.as_tensor(
            np.asarray(batch.probabilities, dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        )
        _require_finite_tensors(
            "inputs",
            (
                ("world.features", model_batch.world.features),
                ("candidates.features", model_batch.candidates.features),
                ("candidates.local_features", model_batch.candidates.local_features),
                ("return_targets", return_targets),
                ("reward_targets", reward_targets),
                ("behavior_log_probabilities", behavior_log_probabilities),
                ("replay_probabilities", probabilities),
                ("importance_weights", weights),
            ),
        )
        if bool((probabilities <= 0.0).any().item()):
            raise ValueError("replay probabilities must be positive")
        if bool((weights <= 0.0).any().item()):
            raise ValueError("replay importance weights must be positive")
        _require_finite_tensors(
            "model parameters",
            tuple(
                (name, parameter)
                for name, parameter in self.model.named_parameters()
                if parameter.is_floating_point() or parameter.is_complex()
            ),
        )
        _require_finite_tensors("optimizer state", _optimizer_tensors(self.optimizer))

        self.model.train()
        output = self.model(model_batch, validate=False)
        _require_finite_tensors(
            "outputs",
            (
                ("world_latents", output.world_latents),
                ("state_embedding", output.state_embedding),
                ("candidate_embeddings", output.candidate_embeddings),
                ("policy_logits", output.policy_logits),
                ("candidate_q", output.candidate_q),
                ("combat_value", output.combat_value),
                ("run_value", output.run_value),
                ("candidate_reward", output.candidate_reward),
                ("candidate_terminal_logits", output.candidate_terminal_logits),
            ),
        )
        selected_legal = output.action_mask[row_indices, action_indices]
        if not bool(selected_legal.all().item()):
            raise ValueError("replay selected an action that the grounded encoder marks illegal")

        combat_objective = torch.tensor(
            [experience.objective == "combat" for experience in experiences],
            dtype=torch.bool,
            device=self.device,
        )
        predicted_values = torch.where(
            combat_objective,
            output.combat_value,
            output.run_value,
        ).float()
        predicted_q = output.candidate_q[row_indices, action_indices].float()
        predicted_reward = output.candidate_reward[row_indices, action_indices].float()
        predicted_terminal = output.candidate_terminal_logits[
            row_indices, action_indices
        ].float()

        value_loss = _weighted_mean(
            F.smooth_l1_loss(predicted_values, return_targets, reduction="none"),
            weights,
        )
        q_loss = _weighted_mean(
            F.smooth_l1_loss(predicted_q, return_targets, reduction="none"),
            weights,
        )
        reward_loss = _weighted_mean(
            F.smooth_l1_loss(predicted_reward, reward_targets, reduction="none"),
            weights,
        )
        terminal_loss = _weighted_mean(
            F.cross_entropy(predicted_terminal, terminal_targets, reduction="none"),
            weights,
        )

        log_probabilities = torch.log_softmax(output.policy_logits.float(), dim=-1)
        selected_log_probabilities = log_probabilities[row_indices, action_indices]
        probabilities = output.policy_probabilities().float()
        masked_log_probabilities = torch.where(
            output.action_mask,
            log_probabilities,
            torch.zeros_like(log_probabilities),
        )
        entropy_per_row = -(probabilities * masked_log_probabilities).sum(dim=-1)
        policy_rows = output.action_mask.sum(dim=-1) > 1
        policy_weights = weights * policy_rows.float()
        advantage = (return_targets - predicted_values).detach()
        log_importance_ratio = (
            selected_log_probabilities - behavior_log_probabilities
        ).detach()
        max_log_importance_ratio = math.log(self.config.importance_ratio_clip)
        importance_ratio = torch.exp(
            log_importance_ratio.clamp(max=max_log_importance_ratio)
        )
        if bool(policy_rows.any().item()):
            policy_loss = _weighted_mean(
                -importance_ratio * advantage * selected_log_probabilities,
                policy_weights,
            )
            entropy = _weighted_mean(entropy_per_row, policy_weights)
        else:
            policy_loss = selected_log_probabilities.sum() * 0.0
            entropy = entropy_per_row.sum() * 0.0

        loss = (
            self.config.policy_weight * policy_loss
            + self.config.value_weight * value_loss
            + self.config.q_weight * q_loss
            + self.config.reward_weight * reward_loss
            + self.config.terminal_weight * terminal_loss
            - self.config.entropy_weight * entropy
        )
        _require_finite_tensors(
            "losses",
            (
                ("value_loss", value_loss),
                ("q_loss", q_loss),
                ("reward_loss", reward_loss),
                ("terminal_loss", terminal_loss),
                ("policy_loss", policy_loss),
                ("entropy", entropy),
                ("advantage", advantage),
                ("importance_ratio", importance_ratio),
                ("total_loss", loss),
            ),
        )
        self.optimizer.zero_grad(set_to_none=True)
        try:
            loss.backward()  # type: ignore[no-untyped-call]
            gradients = tuple(
                (f"{name}.grad", parameter.grad)
                for name, parameter in self.model.named_parameters()
                if parameter.grad is not None
            )
            if not gradients:
                raise RuntimeError("learner backward produced no gradients")
            _require_finite_tensors("gradients", gradients)
            gradient_norm_tensor = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.gradient_clip_norm,
            )
            _require_finite_tensors(
                "gradient norm",
                (("gradient_norm", gradient_norm_tensor),),
            )
            self.optimizer.step()
            _require_finite_tensors(
                "updated model parameters",
                tuple(
                    (name, parameter)
                    for name, parameter in self.model.named_parameters()
                    if parameter.is_floating_point() or parameter.is_complex()
                ),
            )
            _require_finite_tensors(
                "updated optimizer state",
                _optimizer_tensors(self.optimizer),
            )
        except BaseException:
            self.optimizer.zero_grad(set_to_none=True)
            raise

        td_errors = (predicted_q.detach() - return_targets).abs()
        _require_finite_tensors("TD errors", (("td_errors", td_errors),))
        if replay is not None:
            replay.update_priorities(
                batch.indices.tolist(),
                (td_errors.cpu().numpy() + replay.priority_epsilon).tolist(),
            )
        return LearnerMetrics(
            loss=float(loss.detach().cpu()),
            policy_loss=float(policy_loss.detach().cpu()),
            value_loss=float(value_loss.detach().cpu()),
            q_loss=float(q_loss.detach().cpu()),
            reward_loss=float(reward_loss.detach().cpu()),
            terminal_loss=float(terminal_loss.detach().cpu()),
            entropy=float(entropy.detach().cpu()),
            advantage_mean=float(advantage.mean().cpu()),
            td_error_mean=float(td_errors.mean().cpu()),
            gradient_norm=float(gradient_norm_tensor.detach().cpu()),
        )


__all__ = ["GroundedLearner", "LearnerMetrics"]
