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
from .episode_replay import BoundedEpisodicReplay
from .learner import VTraceLearner
from .transaction import BoundedTransactionReplay


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
    transaction_replay: BoundedTransactionReplay | None
    episodic_replay: BoundedEpisodicReplay | None
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


def seed_everything(seed: int, *, seed_accelerators: bool = True) -> None:
    """Seed the requested training RNG domains without waking unused devices.

    ``torch.manual_seed`` also queues CUDA/MPS/XPU seed callbacks.  A frozen
    CPU evaluator must not leave such a callback behind when CUDA was
    previously uninitialized, because that would alter the first later GPU
    initialization even though the evaluator restores the CPU RNG state.
    """

    random.seed(seed)
    np.random.seed(seed)
    if seed_accelerators:
        torch.manual_seed(seed)
    else:
        torch.default_generator.manual_seed(seed)


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
    cuda_rng = (
        torch.cuda.get_rng_state_all()
        if torch.cuda.is_initialized()  # type: ignore[no-untyped-call]
        else []
    )
    try:
        collector_model = RecurrentCandidateModel(
            config.model.to_model_config(),
            enable_transaction_heads=config.transaction_learning.enabled,
        ).to(
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
    device = resolve_device(config.runtime.device)
    collector_device = resolve_device(config.runtime.collector_device)
    seed_everything(
        config.runtime.seed,
        seed_accelerators=(
            device.type == "cuda"
            or collector_device.type == "cuda"
            or torch.cuda.is_initialized()  # type: ignore[no-untyped-call]
        ),
    )
    model = RecurrentCandidateModel(
        config.model.to_model_config(),
        enable_transaction_heads=config.transaction_learning.enabled,
    ).to(device)
    collector_model = _build_collector_model(config, learner_model=model)
    encoder = GroundedObservationEncoder(config.model.to_encoding_config())
    rollout_queue = BoundedRolloutQueue(
        config.rollout.queue_capacity,
    )
    transaction_replay = (
        BoundedTransactionReplay(
            capacity=config.transaction_learning.replay_capacity,
            byte_capacity=config.transaction_learning.replay_byte_capacity,
            seed=config.runtime.seed,
        )
        if config.transaction_learning.enabled
        else None
    )
    episodic_replay = (
        BoundedEpisodicReplay(
            capacity=config.episodic_learning.replay_capacity_episodes,
            byte_capacity=config.episodic_learning.replay_capacity_bytes,
            episode_byte_capacity=config.episodic_learning.per_episode_capacity_bytes,
            max_segments_per_episode=config.episodic_learning.max_segments_per_episode,
            seed=config.runtime.seed,
        )
        if config.episodic_learning.enabled
        else None
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.optimization.learning_rate,
        weight_decay=config.optimization.weight_decay,
    )
    reward_calculator = (
        RevivalEfficiencyRewardCalculator(
            objective=config.curriculum.reward_objective,
            discount=config.optimization.discount,
            maximum_episode_steps=config.environment.max_episode_steps,
        )
        if config.curriculum.mode == "native-revival-preheat"
        else None
    )
    owns_backend = backend is None
    environment_backend = backend if backend is not None else build_backend(config)
    try:
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
            combat_net_progress_room_windows=(
                config.diagnostics.combat_net_progress_room_windows
            ),
            combat_net_progress_encounter_windows=(
                config.diagnostics.combat_net_progress_encounter_windows
            ),
            noncombat_durable_progress_window=(
                config.diagnostics.noncombat_durable_progress_window
            ),
            combat_min_net_hp_fraction=(
                config.diagnostics.combat_min_net_hp_fraction
            ),
            journal_policy_topk=config.diagnostics.journal_policy_topk,
            reward_calculator=reward_calculator,
            # The maintained preheat uses private engine state and injects no
            # model-visible relic. GroundedCollector retains generic
            # ``additional_relics`` support for explicit scenario fixtures.
            additional_relics=(),
            training_revival_budget=config.curriculum.revival_budget,
            horizon_as_failure=config.curriculum.mode == "native-revival-preheat",
            transaction_burn_in_steps=(
                config.transaction_learning.burn_in_steps
                if config.transaction_learning.enabled
                else None
            ),
            episodic_learning_enabled=config.episodic_learning.enabled,
        )
        learner = VTraceLearner(
            model=model,
            encoder=encoder,
            optimizer=optimizer,
            config=config.optimization,
            maximum_unroll_length=config.rollout.unroll_length,
            maximum_policy_lag=config.rollout.max_policy_lag,
            transaction_config=config.transaction_learning,
            episodic_config=config.episodic_learning,
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
            transaction_replay=transaction_replay,
            episodic_replay=episodic_replay,
            device=device,
        )
    except BaseException:
        # Once returned, TrainingResources owns and closes any backend.  If
        # composition fails before return, close only a backend created here;
        # an injected backend remains the caller's responsibility.
        rollout_queue.close()
        if owns_backend:
            environment_backend.close()
        raise


__all__ = [
    "TrainingResources",
    "build_backend",
    "build_training_resources",
    "resolve_device",
    "seed_everything",
]
