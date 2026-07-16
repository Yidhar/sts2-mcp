"""Composition root for the recurrent actor/V-trace learner pipeline."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass

import numpy as np
import torch

from sts2_baseline import BoundedRolloutQueue, RevivalEfficiencyRewardCalculator
from sts2_rl.backends import HeadlessBackend, LiveBackend
from sts2_rl.contracts import EnvironmentBackend
from sts2_rl.encoding import GroundedObservationEncoder
from sts2_rl.models import RecurrentCandidateModel

from .collector import GroundedCollector
from .config import TrainingConfig
from .learner import VTraceLearner


@dataclass(slots=True)
class TrainingResources:
    model: RecurrentCandidateModel
    collector_model: RecurrentCandidateModel
    encoder: GroundedObservationEncoder
    rollout_queue: BoundedRolloutQueue
    backend: EnvironmentBackend
    optimizer: torch.optim.Optimizer
    collector: GroundedCollector
    learner: VTraceLearner
    device: torch.device

    def publish_collector_policy(self) -> float:
        """Publish one consistent learner snapshot to the idle collector model.

        Actor and learner never mutate the same module.  Callers publish only
        between actor episodes, giving every unroll one exact policy version.
        """

        if self.collector_model is self.model:
            return 0.0
        started_ns = time.perf_counter_ns()
        learner_device = next(self.model.parameters()).device
        collector_device = next(self.collector_model.parameters()).device
        if learner_device.type == "cuda":
            torch.cuda.synchronize(learner_device)
        self.collector_model.load_state_dict(self.model.state_dict(), strict=True)
        if collector_device.type == "cuda":
            torch.cuda.synchronize(collector_device)
        elif learner_device.type == "cuda":
            # Host copies from a CUDA/ROCm source must be complete before the
            # worker can read the CPU replica.
            torch.cuda.synchronize(learner_device)
        return (time.perf_counter_ns() - started_ns) / 1_000_000.0

    def close(self) -> None:
        self.rollout_queue.close()
        self.backend.close()


def resolve_device(requested: str) -> torch.device:
    normalized = str(requested).strip().lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA/ROCm device requested but torch.cuda.is_available() is false")
    return device


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_backend(config: TrainingConfig) -> EnvironmentBackend:
    environment = config.environment
    if environment.backend == "live":
        return LiveBackend(
            session_path=environment.session_path,
            allow_legacy_fallback=False,
        )
    return HeadlessBackend(exe_path=environment.sim_exe_path)


def _build_collector_model(
    config: TrainingConfig,
    *,
    learner_model: RecurrentCandidateModel,
) -> RecurrentCandidateModel:
    """Build a replica without advancing the checkpointed Torch RNG streams."""

    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    try:
        collector_model = RecurrentCandidateModel(config.model.to_model_config()).to(
            resolve_device(config.runtime.collector_device)
        )
        collector_model.load_state_dict(learner_model.state_dict(), strict=True)
    finally:
        torch.set_rng_state(cpu_rng)
        if cuda_rng:
            torch.cuda.set_rng_state_all(cuda_rng)
    return collector_model


def build_training_resources(
    config: TrainingConfig,
    *,
    backend: EnvironmentBackend | None = None,
) -> TrainingResources:
    seed_everything(config.runtime.seed)
    device = resolve_device(config.runtime.device)
    model = RecurrentCandidateModel(config.model.to_model_config()).to(device)
    collector_model = _build_collector_model(config, learner_model=model)
    encoder = GroundedObservationEncoder(config.model.to_encoding_config())
    rollout_queue = BoundedRolloutQueue(
        config.rollout.queue_capacity,
    )
    environment_backend = backend or build_backend(config)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.optimization.learning_rate,
        weight_decay=config.optimization.weight_decay,
    )
    revival_relic_id = config.curriculum.revival_relic_id
    reward_calculator = (
        RevivalEfficiencyRewardCalculator(
            revival_relic_id=revival_relic_id,
            objective=config.curriculum.reward_objective,
            discount=config.optimization.discount,
            maximum_episode_steps=config.environment.max_episode_steps,
        )
        if config.curriculum.mode == "native-revival-preheat"
        and revival_relic_id is not None
        else None
    )
    collector = GroundedCollector(
        model=collector_model,
        encoder=encoder,
        backend=environment_backend,
        scenario=config.environment.scenario,
        objective=config.curriculum.reward_objective,
        discount=config.optimization.discount,
        max_episode_steps=config.environment.max_episode_steps,
        character=config.environment.character,
        encounter_id=config.environment.encounter_id,
        seed=config.runtime.seed,
        unroll_length=config.rollout.unroll_length,
        deadlock_window=config.diagnostics.deadlock_window,
        deadlock_repeat_threshold=config.diagnostics.deadlock_repeat_threshold,
        combat_net_progress_window=(
            config.diagnostics.combat_net_progress_window
        ),
        combat_min_net_hp_fraction=(
            config.diagnostics.combat_min_net_hp_fraction
        ),
        journal_policy_topk=config.diagnostics.journal_policy_topk,
        reward_calculator=reward_calculator,
        additional_relics=(
            (revival_relic_id,)
            if config.curriculum.mode == "native-revival-preheat"
            and revival_relic_id is not None
            else ()
        ),
        revival_relic_id=revival_relic_id,
        training_revival_budget=config.curriculum.revival_budget,
        horizon_as_failure=config.curriculum.mode == "native-revival-preheat",
    )
    learner = VTraceLearner(
        model=model,
        encoder=encoder,
        optimizer=optimizer,
        config=config.optimization,
        maximum_unroll_length=config.rollout.unroll_length,
        maximum_policy_lag=config.rollout.max_policy_lag,
    )
    return TrainingResources(
        model=model,
        collector_model=collector_model,
        encoder=encoder,
        rollout_queue=rollout_queue,
        backend=environment_backend,
        optimizer=optimizer,
        collector=collector,
        learner=learner,
        device=device,
    )


__all__ = [
    "TrainingResources",
    "build_backend",
    "build_training_resources",
    "resolve_device",
    "seed_everything",
]
