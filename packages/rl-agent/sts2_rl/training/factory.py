"""Composition root for the grounded baseline; no legacy trainer facade."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch

from sts2_baseline import ReplayMix, StratifiedReplayBuffer
from sts2_rl.backends import HeadlessBackend, LiveBackend
from sts2_rl.contracts import EnvironmentBackend
from sts2_rl.encoding import GroundedObservationEncoder
from sts2_rl.models import GroundedCandidateModel

from .collector import GroundedCollector
from .config import TrainingConfig
from .learner import GroundedLearner


@dataclass(slots=True)
class TrainingResources:
    model: GroundedCandidateModel
    encoder: GroundedObservationEncoder
    replay: StratifiedReplayBuffer
    backend: EnvironmentBackend
    optimizer: torch.optim.Optimizer
    collector: GroundedCollector
    learner: GroundedLearner
    device: torch.device

    def close(self) -> None:
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


def build_training_resources(
    config: TrainingConfig,
    *,
    backend: EnvironmentBackend | None = None,
) -> TrainingResources:
    seed_everything(config.runtime.seed)
    device = resolve_device(config.runtime.device)
    model = GroundedCandidateModel(config.model.to_model_config()).to(device)
    encoder = GroundedObservationEncoder(config.model.to_encoding_config())
    replay = StratifiedReplayBuffer(
        config.replay.capacity,
        recent_window=config.replay.recent_window,
        mix=ReplayMix(
            coverage=config.replay.coverage_fraction,
            recent=config.replay.recent_fraction,
            per=config.replay.priority_fraction,
        ),
        alpha=config.replay.alpha,
        beta=config.replay.beta,
        priority_epsilon=config.replay.priority_epsilon,
        seed=config.runtime.seed,
    )
    environment_backend = backend or build_backend(config)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.optimization.learning_rate,
        weight_decay=config.optimization.weight_decay,
    )
    collector = GroundedCollector(
        model=model,
        encoder=encoder,
        backend=environment_backend,
        scenario=config.environment.scenario,
        objective=config.curriculum.reward_objective,
        discount=config.optimization.discount,
        max_episode_steps=config.environment.max_episode_steps,
        character=config.environment.character,
        encounter_id=config.environment.encounter_id,
        seed=config.runtime.seed,
    )
    learner = GroundedLearner(
        model=model,
        encoder=encoder,
        optimizer=optimizer,
        config=config.optimization,
    )
    return TrainingResources(
        model=model,
        encoder=encoder,
        replay=replay,
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
