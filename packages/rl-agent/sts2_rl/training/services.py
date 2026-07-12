"""Composition services around the temporary MuZeroTrainer compatibility facade."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, cast
from uuid import uuid4

from sts2_rl.contracts.versions import API_VERSION, SCHEMA_VERSION

from .config import TrainingConfig


@dataclass(frozen=True, slots=True)
class ExperimentContext:
    config: TrainingConfig
    run_id: str
    created_unix_s: float
    api_version: str = API_VERSION
    schema_version: str = SCHEMA_VERSION

    @classmethod
    def create(cls, config: TrainingConfig) -> ExperimentContext:
        return cls(config=config, run_id=str(uuid4()), created_unix_s=time.time())


@dataclass(slots=True)
class ReplayStore:
    replay: Any

    def __len__(self) -> int:
        return len(self.replay)

    def save_episode(self, trajectory: Any, *, discount: float, n_steps: int) -> None:
        self.replay.save_episode(trajectory, discount=discount, n_steps=n_steps)


@dataclass(slots=True)
class Collector:
    facade: Any

    def temperature(self, step: int, total_steps: int) -> float:
        return float(self.facade.compute_temperature(step, total_steps))

    def collect_episode(self, *, temperature: float) -> tuple[float, int]:
        return cast(tuple[float, int], self.facade.self_play_episode(temperature=temperature))

    def collect_on(self, environment: Any, *, temperature: float) -> tuple[float, int]:
        return cast(tuple[float, int], self.facade.self_play_episode_on_env(environment, temperature=temperature))


@dataclass(slots=True)
class Learner:
    facade: Any

    def update(self, *, batch_size: int, unroll_steps: int) -> dict[str, float]:
        return cast(dict[str, float], self.facade.train_step(batch_size=batch_size, unroll_steps=unroll_steps))


@dataclass(slots=True)
class Evaluator:
    facade: Any

    def evaluate_episode(self, *, temperature: float = 0.0) -> tuple[float, int]:
        return cast(tuple[float, int], self.facade.self_play_episode(temperature=temperature))


@dataclass(slots=True)
class CheckpointManager:
    facade: Any

    def save(self, *, tag: str = "") -> None:
        self.facade.save_checkpoint(tag=tag)


@dataclass(slots=True)
class Telemetry:
    writer: Any

    def scalar(self, name: str, value: float, step: int) -> None:
        self.writer.add_scalar(name, value, step)

    def flush(self) -> None:
        self.writer.flush()

    def close(self) -> None:
        self.writer.close()


@dataclass(slots=True)
class TrainingRuntime:
    context: ExperimentContext
    collector: Collector
    learner: Learner
    evaluator: Evaluator
    replay_store: ReplayStore
    checkpoints: CheckpointManager
    telemetry: Telemetry
    compatibility_facade: Any

    @classmethod
    def from_legacy(cls, facade: Any, config: TrainingConfig) -> TrainingRuntime:
        context = ExperimentContext.create(config)
        facade.experiment_context = context
        facade.parent_checkpoint = config.option("resume_from")
        return cls(
            context=context,
            collector=Collector(facade),
            learner=Learner(facade),
            evaluator=Evaluator(facade),
            replay_store=ReplayStore(facade.buffer),
            checkpoints=CheckpointManager(facade),
            telemetry=Telemetry(facade.writer),
            compatibility_facade=facade,
        )
