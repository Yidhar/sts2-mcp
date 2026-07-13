from __future__ import annotations

import json
import random
import threading
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

import sts2_rl.training.checkpointing as checkpointing_module
from sts2_rl.contracts import (
    BackendCapabilities,
    CombatResetRequest,
    EnvironmentResult,
    EnvironmentTransition,
    ResetRequest,
    StepRequest,
)
from sts2_rl.encoding import GroundedObservationEncoder
from sts2_rl.models import GroundedCandidateModel
from sts2_rl.training import (
    CurriculumConfig,
    DecisionExperience,
    EnvironmentConfig,
    GroundedCollector,
    ModelConfig,
    OptimizationConfig,
    ReplayConfig,
    RuntimeConfig,
    TrainingConfig,
    build_training_resources,
    evaluate_policy,
    initialize_model_from_checkpoint,
    load_training_checkpoint,
    run_training,
    save_training_checkpoint,
)
from sts2_rl.training.checkpointing import TrainingState
from sts2_rl.training.experience import (
    baseline_transition,
    compact_decision,
    potential_state,
)
from sts2_rl.training.learner import GroundedLearner
from sts2_rl.training.seeding import training_seed_start


class FakeCombatBackend:
    def __init__(self) -> None:
        self._capabilities = BackendCapabilities(
            backend_name="fake",
            session_id="fake-session",
        )
        self._state_version = 0
        self._step = 0
        self._episode = 0
        self.closed = False
        self.reset_seeds: list[int | str | None] = []

    @property
    def capabilities(self) -> BackendCapabilities:
        return self._capabilities

    @property
    def session_id(self) -> str:
        return self._capabilities.session_id

    @property
    def is_connected(self) -> bool:
        return not self.closed

    def health(self) -> dict[str, Any]:
        return {"ok": True}

    def get_spec(self) -> dict[str, Any]:
        return {"ok": True}

    def get_state(self) -> dict[str, Any]:
        return {"ok": True, "state_version": self._state_version}

    @staticmethod
    def _observation(step: int, terminal: bool = False) -> dict[str, Any]:
        enemy_hp = 0 if terminal else 30 - 5 * step
        return {
            "phase": "combat",
            "decision_domain": "combat",
            "player": {"id": "player", "side": "player", "hp": 50, "max_hp": 80},
            "combat": {
                "in_progress": not terminal,
                "encounter_id": "fake-encounter",
                "tier": "normal",
                "enemies": [
                    {
                        "id": "enemy",
                        "side": "enemy",
                        "hp": enemy_hp,
                        "max_hp": 30,
                    }
                ],
            },
            "run": {"act": 1, "floor": 1},
        }

    @staticmethod
    def _actions() -> tuple[dict[str, Any], ...]:
        return (
            {
                "action_handle": "play:attack",
                "kind": "play_card",
                "model_action_kind": "play_card",
                "card": {"id": "attack", "cost": 1},
                "target": {"id": "enemy", "side": "enemy"},
            },
            {
                "action_handle": "end",
                "kind": "end_turn",
                "model_action_kind": "end_turn",
            },
        )

    def reset(self, request: ResetRequest) -> EnvironmentResult:
        raise AssertionError("combat test must use combat_reset")

    def combat_reset(self, request: CombatResetRequest) -> EnvironmentResult:
        assert request.expected_state_version == self._state_version
        self.reset_seeds.append(request.seed)
        before_version = self._state_version
        self._episode += 1
        self._step = 0
        self._state_version += 1
        episode_id = f"episode-{self._episode}"
        return EnvironmentResult(
            episode_id=episode_id,
            step_index=0,
            observation=self._observation(0),
            legal_actions=self._actions(),
            transition=EnvironmentTransition(
                episode_id=episode_id,
                step_index=0,
                before_state_version=before_version,
                after_state_version=self._state_version,
                facts={"combat_result": "none", "terminal_reason": None},
            ),
            info={"reward_authority": "external-rl"},
        )

    def step(self, request: StepRequest) -> EnvironmentResult:
        assert request.expected_step_index == self._step
        before_version = self._state_version
        self._step += 1
        self._state_version += 1
        terminal = self._step >= 2
        return EnvironmentResult(
            episode_id=f"episode-{self._episode}",
            step_index=self._step,
            observation=self._observation(self._step, terminal=terminal),
            legal_actions=() if terminal else self._actions(),
            transition=EnvironmentTransition(
                episode_id=f"episode-{self._episode}",
                step_index=self._step,
                before_state_version=before_version,
                after_state_version=self._state_version,
                facts={
                    "combat_result": "victory" if terminal else "none",
                    "terminal_reason": "combat_victory" if terminal else None,
                },
            ),
            terminated=terminal,
            terminal_reason="combat_victory" if terminal else None,
            info={"reward_authority": "external-rl"},
        )

    def close(self) -> None:
        self.closed = True


class InvalidStateVersionBackend(FakeCombatBackend):
    def get_state(self) -> dict[str, Any]:
        return {"ok": True, "state_version": True}


class StaleTransitionBackend(FakeCombatBackend):
    def step(self, request: StepRequest) -> EnvironmentResult:
        result = super().step(request)
        assert result.transition is not None
        return replace(
            result,
            transition=replace(
                result.transition,
                before_state_version=result.transition.before_state_version - 1,
            ),
        )


class TransportTruncationBackend(FakeCombatBackend):
    def step(self, request: StepRequest) -> EnvironmentResult:
        result = super().step(request)
        assert result.transition is not None
        reason = "transport_timeout"
        return replace(
            result,
            legal_actions=(),
            terminated=False,
            truncated=True,
            terminal_reason=reason,
            transition=replace(
                result.transition,
                facts={"combat_result": "none", "terminal_reason": reason},
            ),
        )


class OverlapProbeBackend(FakeCombatBackend):
    """Require learner work to begin while the second episode is in flight."""

    def __init__(self) -> None:
        super().__init__()
        self.learner_started = threading.Event()
        self.observed_overlap = False

    def step(self, request: StepRequest) -> EnvironmentResult:
        if self._episode >= 2 and self._step == 0:
            self.observed_overlap = self.learner_started.wait(timeout=5.0)
            if not self.observed_overlap:
                raise AssertionError(
                    "second collection did not overlap the first episode's learner work"
                )
        return super().step(request)


def _small_config() -> TrainingConfig:
    model = ModelConfig(
        token_feature_dim=128,
        d_model=32,
        n_heads=4,
        ffn_dim=64,
        world_layers=1,
        latent_slots=4,
        latent_layers=1,
        local_layers=1,
        candidate_layers=1,
        dropout=0.0,
        type_vocab_size=32,
        role_vocab_size=32,
        owner_vocab_size=16,
        entity_vocab_size=128,
        zone_vocab_size=16,
        order_vocab_size=32,
        domain_count=8,
        max_world_tokens=24,
        max_candidates=6,
        max_candidate_local_tokens=5,
    )
    return TrainingConfig(
        profile="test-combat",
        model=model,
        optimization=OptimizationConfig(batch_size=2),
        replay=ReplayConfig(capacity=20, recent_window=10, minimum_size=2),
        environment=EnvironmentConfig(
            backend="headless",
            scenario="combat",
            max_episode_steps=4,
        ),
        curriculum=CurriculumConfig(
            reward_objective="combat",
            epsilon_start=0.2,
            epsilon_end=0.1,
            epsilon_decay_steps=10,
        ),
        runtime=RuntimeConfig(
            device="cpu",
            total_environment_steps=2,
            train_every_steps=1,
            updates_per_cycle=1,
            seed=7,
            log_dir="tests/run",
            checkpoint_dir="tests/checkpoints",
            checkpoint_interval_steps=2,
            evaluation_interval_steps=2,
            evaluation_episodes=0,
        ),
    )


def test_collector_returns_auditable_monte_carlo_targets() -> None:
    config = _small_config()
    backend = FakeCombatBackend()
    model = GroundedCandidateModel(config.model.to_model_config())
    encoder = GroundedObservationEncoder(config.model.to_encoding_config())
    collector = GroundedCollector(
        model=model,
        encoder=encoder,
        backend=backend,
        scenario="combat",
        objective="combat",
        discount=config.optimization.discount,
        max_episode_steps=4,
        seed=3,
    )

    episode = collector.collect_episode(epsilon=1.0)

    assert episode.metrics.steps == 2
    assert episode.metrics.combat_won
    assert len(episode.samples) == 2
    assert episode.timings is not None
    collector_timings = episode.timings.to_mapping()
    assert collector_timings["reset"]["count"] == 1
    assert collector_timings["target_finalize"]["count"] == 1
    for stage_name in (
        "observation_encoding",
        "policy_forward",
        "sim_step",
        "transition_reward_compaction",
    ):
        stage = collector_timings[stage_name]
        assert stage["count"] == episode.metrics.steps
        assert 0.0 <= stage["min_ms"] <= stage["mean_ms"] <= stage["max_ms"]
        assert stage["total_ms"] >= stage["max_ms"]
    assert episode.samples[-1].transition.combat_result == "win"
    assert episode.samples[-1].targets.value == episode.samples[-1].targets.reward
    assert episode.samples[0].targets.value is not None


def test_collector_fails_closed_on_state_version_and_transition_identity() -> None:
    config = _small_config()
    invalid_state = build_training_resources(
        config,
        backend=InvalidStateVersionBackend(),
    )
    try:
        with pytest.raises(RuntimeError, match="state_version.*exact"):
            invalid_state.collector.collect_episode(epsilon=0.0)
    finally:
        invalid_state.close()

    stale_transition = build_training_resources(
        config,
        backend=StaleTransitionBackend(),
    )
    try:
        with pytest.raises(RuntimeError, match="revision chain"):
            stale_transition.collector.collect_episode(epsilon=0.0)
    finally:
        stale_transition.close()


def test_transport_truncation_is_discarded_before_replay() -> None:
    resources = build_training_resources(
        _small_config(),
        backend=TransportTruncationBackend(),
    )
    try:
        with pytest.raises(RuntimeError, match="discarded before replay"):
            resources.collector.collect_episode(epsilon=0.0)
        assert len(resources.replay) == 0
    finally:
        resources.close()


def test_forced_collector_horizon_has_no_loss_or_terminal_label() -> None:
    config = _small_config()
    model = GroundedCandidateModel(config.model.to_model_config())
    collector = GroundedCollector(
        model=model,
        encoder=GroundedObservationEncoder(config.model.to_encoding_config()),
        backend=FakeCombatBackend(),
        scenario="combat",
        objective="combat",
        discount=config.optimization.discount,
        max_episode_steps=1,
        seed=3,
    )

    episode = collector.collect_episode(epsilon=0.0)

    assert len(episode.samples) == 1
    sample = episode.samples[0]
    assert sample.transition.truncated
    assert sample.transition.metadata["truncation_kind"] == "collector_horizon"
    assert sample.targets.reward > -1.0
    assert sample.targets.value == sample.targets.reward
    assert isinstance(sample.payload, DecisionExperience)
    assert sample.payload.terminal_class == 0


def test_terminal_outcome_and_reward_facts_are_exact_and_fail_closed() -> None:
    before = EnvironmentResult(
        episode_id="run-1",
        step_index=0,
        observation=FakeCombatBackend._observation(0),
        legal_actions=FakeCombatBackend._actions(),
    )
    malformed_terminal = EnvironmentResult(
        episode_id="run-1",
        step_index=1,
        observation=FakeCombatBackend._observation(1, terminal=True),
        transition=EnvironmentTransition(
            episode_id="run-1",
            step_index=1,
            before_state_version=1,
            after_state_version=2,
            facts={"combat_result": "none", "terminal_reason": "run_incomplete"},
        ),
        terminated=True,
        terminal_reason="run_incomplete",
        info={"reward_authority": "external-rl"},
    )
    with pytest.raises(ValueError, match="no exact typed terminal outcome"):
        baseline_transition(
            before=before,
            after=malformed_terminal,
            action_handle="action:end",
            objective="run",
        )

    with pytest.raises(ValueError, match="no typed player"):
        potential_state({}, objective="run")
    terminal_without_player = {
        "terminated": True,
        "run": {"floor": 12},
        "combat": {"in_progress": False},
    }
    projected_terminal = potential_state(terminal_without_player, objective="run")
    assert projected_terminal.player_hp == 0.0
    assert projected_terminal.player_max_hp == 0.0
    assert projected_terminal.player_hp_ratio == 0.0
    terminal_partial_player = {
        **terminal_without_player,
        "player": {"max_hp": 80},
    }
    with pytest.raises(ValueError, match="player.hp is required"):
        potential_state(terminal_partial_player, objective="run")
    invalid_hp = FakeCombatBackend._observation(0)
    invalid_hp["player"] = {"hp": float("nan"), "max_hp": 80}
    with pytest.raises(ValueError, match="player.hp must be finite"):
        potential_state(invalid_hp, objective="combat")


def test_decision_experience_owns_encoded_snapshot_and_source_identity() -> None:
    observation = {
        "player": {"hp": 50, "max_hp": 80},
        "nested": {"values": [1]},
    }
    actions = (
        {
            "action_handle": "play:attack",
            "kind": "play_card",
            "model_action_kind": "play_card",
            "target": {"id": "enemy", "tags": ["original"]},
        },
    )
    config = _small_config()
    encoder = GroundedObservationEncoder(config.model.to_encoding_config())
    encoded = encoder.encode(observation, actions)
    experience = compact_decision(
        observation,
        actions,
        encoded_snapshot=encoded.snapshot,
        action_index=0,
        behavior_log_probability=0.0,
        terminal_class=0,
        objective="combat",
    )
    source_fingerprint = experience.source_fingerprint
    world_ids = experience.encoded_snapshot.world.ids.copy()

    observation["nested"]["values"].append(2)
    actions[0]["target"]["tags"].append("caller-mutation")
    assert experience.source_fingerprint == source_fingerprint
    assert np.array_equal(experience.encoded_snapshot.world.ids, world_ids)
    assert experience.encoded_snapshot.world.ids.flags.writeable is False


class StatefulEvaluationCollector:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.state = {
            "rng": {"draw": 11},
            "episode_seed": 7,
            "steps": 13,
            "episode_count": 3,
        }

    def state_dict(self) -> dict[str, Any]:
        # Deliberately return the owned object: evaluate_policy must snapshot it
        # rather than relying on every collector implementation to deep-copy.
        return self.state

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        self.state = deepcopy(payload)

    def collect_episode(self, **_: Any) -> SimpleNamespace:
        self.state["rng"]["draw"] += 1
        self.state["episode_seed"] += 1
        self.state["steps"] += 2
        self.state["episode_count"] += 1
        if self.fail:
            raise RuntimeError("synthetic evaluation failure")
        return SimpleNamespace(
            metrics=SimpleNamespace(
                act1_cleared=False,
                run_won=False,
                combat_won=True,
                max_floor=1,
                max_act=1,
                reward_total=0.5,
            )
        )


@pytest.mark.parametrize("fail", (False, True))
def test_evaluate_policy_restores_all_collector_state_on_success_or_failure(
    fail: bool,
) -> None:
    collector = StatefulEvaluationCollector(fail=fail)
    resources: Any = SimpleNamespace(
        collector=collector,
        publish_collector_policy=lambda: 0.0,
    )
    before = deepcopy(collector.state)

    if fail:
        with pytest.raises(RuntimeError, match="synthetic evaluation failure"):
            evaluate_policy(resources, episodes=2)
    else:
        evaluated, summary = evaluate_policy(resources, episodes=2)
        assert len(evaluated) == 2
        assert summary["episodes"] == 2

    assert collector.state == before


def test_evaluate_policy_preserves_grounded_training_collector_rng_and_seed() -> None:
    backend = FakeCombatBackend()
    resources = build_training_resources(_small_config(), backend=backend)
    before = resources.collector.state_dict()
    try:
        evaluated, _ = evaluate_policy(resources, episodes=2)
        assert len(evaluated) == 2
        assert backend.reset_seeds == [1, 3]
        assert all(item.reset_seed % 2 == 1 for item in evaluated)
        assert resources.collector.state_dict() == before
        resources.collector.collect_episode(epsilon=0.0, deterministic=True)
        assert backend.reset_seeds[-1] == training_seed_start(_small_config().runtime.seed)
        assert backend.reset_seeds[-1] not in {1, 3}
    finally:
        resources.close()


def test_learner_updates_model_without_reencoding_raw_observation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _small_config()
    backend = FakeCombatBackend()
    resources = build_training_resources(config, backend=backend)
    episode = resources.collector.collect_episode(epsilon=1.0)
    resources.replay.extend(episode.samples)
    batch = resources.replay.sample(2)
    before = next(resources.model.parameters()).detach().clone()
    monkeypatch.setattr(
        resources.learner.encoder,
        "encode",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("learner must collate replay snapshots, not re-encode raw JSON")
        ),
    )

    metrics = resources.learner.update(batch, replay=resources.replay)

    after = next(resources.model.parameters()).detach()
    assert not torch.equal(before, after)
    assert metrics.loss == metrics.loss
    assert metrics.timings is not None
    assert set(metrics.timings.to_mapping()) == {
        "encoding",
        "forward",
        "loss_compute",
        "backward",
        "finite_check",
        "optimizer_step",
        "replay_priority",
        "total",
    }
    assert all(value >= 0.0 for value in metrics.timings.to_mapping().values())
    assert "timings" not in metrics.to_mapping()
    assert all(resources.replay.priority(int(index)) > 0.0 for index in batch.indices)


def test_overlap_uses_an_independent_published_collector_model() -> None:
    base = _small_config()
    config = replace(
        base,
        runtime=replace(base.runtime, execution_mode="overlap"),
    )
    resources = build_training_resources(config, backend=FakeCombatBackend())
    try:
        assert resources.collector_model is not resources.model
        learner_parameter = next(resources.model.parameters())
        collector_parameter = next(resources.collector_model.parameters())
        assert torch.equal(learner_parameter, collector_parameter)
        with torch.no_grad():
            learner_parameter.add_(1.0)
        assert not torch.equal(learner_parameter, collector_parameter)

        publish_ms = resources.publish_collector_policy()

        assert publish_ms >= 0.0
        assert torch.equal(learner_parameter, collector_parameter)
    finally:
        resources.close()


def test_overlap_replica_construction_does_not_advance_torch_rng() -> None:
    base = _small_config()
    synchronous = build_training_resources(base, backend=FakeCombatBackend())
    try:
        synchronous_next = torch.rand(8)
    finally:
        synchronous.close()

    overlap_config = replace(
        base,
        runtime=replace(base.runtime, execution_mode="overlap"),
    )
    overlap = build_training_resources(overlap_config, backend=FakeCombatBackend())
    try:
        overlap_next = torch.rand(8)
    finally:
        overlap.close()

    assert torch.equal(synchronous_next, overlap_next)


def test_runtime_overlaps_one_bounded_episode_with_learner_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STS2_ARTIFACT_ROOT", str(tmp_path))
    backend = OverlapProbeBackend()
    original_update = GroundedLearner.update

    def observed_update(self: GroundedLearner, *args: Any, **kwargs: Any) -> Any:
        backend.learner_started.set()
        return original_update(self, *args, **kwargs)

    monkeypatch.setattr(GroundedLearner, "update", observed_update)
    base = _small_config()
    config = replace(
        base,
        runtime=replace(
            base.runtime,
            execution_mode="overlap",
            total_environment_steps=4,
            warmup_credit_policy="accrue",
            checkpoint_interval_steps=100,
            evaluation_interval_steps=100,
        ),
    )

    state = run_training(config, backend=backend)

    assert backend.observed_overlap
    assert backend.closed
    assert state.environment_steps == 4
    assert state.episodes == 2
    assert state.learner_updates == 4
    metric_directory = next((tmp_path / "tests" / "run").glob("run-*"))
    records = [
        json.loads(line)
        for line in (metric_directory / "metrics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    start = next(record for record in records if record["event"] == "run_start")
    episodes = [record for record in records if record["event"] == "train_episode"]
    assert start["execution_mode"] == "overlap"
    assert start["collector_model_is_independent"] is True
    assert [record["collector_policy_version"] for record in episodes] == [0, 0]
    assert [record["collector_policy_lag_updates"] for record in episodes] == [0, 2]
    assert [record["prefetched_next_episode"] for record in episodes] == [True, False]
    for record in episodes:
        stages = record["timings"]["stages"]
        assert stages["collector_policy_publish"]["count"] == 1
        assert stages["collector_wait"]["count"] == 1
        assert stages["collector_pre_wait"]["count"] == 1


def test_overlap_interrupt_drains_inflight_episode_before_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STS2_ARTIFACT_ROOT", str(tmp_path))

    original_update = GroundedLearner.update
    interrupted_once = False

    def interrupt_update(self: GroundedLearner, *args: Any, **kwargs: Any) -> Any:
        nonlocal interrupted_once
        if not interrupted_once:
            interrupted_once = True
            raise KeyboardInterrupt
        return original_update(self, *args, **kwargs)

    monkeypatch.setattr(GroundedLearner, "update", interrupt_update)
    base = _small_config()
    config = replace(
        base,
        runtime=replace(
            base.runtime,
            execution_mode="overlap",
            total_environment_steps=4,
            warmup_credit_policy="accrue",
            checkpoint_interval_steps=100,
            evaluation_interval_steps=100,
        ),
    )
    backend = FakeCombatBackend()

    with pytest.raises(KeyboardInterrupt):
        run_training(config, backend=backend)

    assert backend.closed
    metric_directory = next((tmp_path / "tests" / "run").glob("run-*"))
    records = [
        json.loads(line)
        for line in (metric_directory / "metrics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    drained = next(record for record in records if record["event"] == "interrupt_drain")
    interrupted = next(record for record in records if record["event"] == "interrupted")
    assert drained["state"]["environment_steps"] == 4
    assert drained["state"]["episodes"] == 2
    assert drained["state"]["learner_updates"] == 2
    assert drained["state"]["update_credit"] == 2
    assert drained["completed_prior_update_cycles"] == 2
    assert interrupted["state"]["environment_steps"] == 4
    assert Path(interrupted["path"]).name == "interrupt-step-000000004"


def test_synchronous_interrupt_rolls_back_partial_collector_rng_and_seed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STS2_ARTIFACT_ROOT", str(tmp_path))

    def interrupt_collection(
        self: GroundedCollector,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        self._episode_seed += 2
        self._rng.random()
        raise KeyboardInterrupt

    monkeypatch.setattr(GroundedCollector, "collect_episode", interrupt_collection)
    config = _small_config()
    backend = FakeCombatBackend()

    with pytest.raises(KeyboardInterrupt):
        run_training(config, backend=backend)

    assert backend.closed
    metric_directory = next((tmp_path / "tests" / "run").glob("run-*"))
    records = [
        json.loads(line)
        for line in (metric_directory / "metrics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert any(
        record["event"] == "interrupt_collector_rollback" for record in records
    )
    interrupted = next(record for record in records if record["event"] == "interrupted")
    checkpoint = Path(interrupted["path"])
    monkeypatch.undo()
    restored = build_training_resources(config, backend=FakeCombatBackend())
    try:
        state = load_training_checkpoint(
            checkpoint,
            config=config,
            resources=restored,
        )
        assert state == TrainingState()
        assert restored.collector.state_dict()["episode_seed"] == training_seed_start(
            config.runtime.seed
        )
    finally:
        restored.close()


def test_learner_rejects_replay_reward_contract_drift() -> None:
    resources = build_training_resources(_small_config(), backend=FakeCombatBackend())
    try:
        episode = resources.collector.collect_episode(epsilon=1.0)
        resources.replay.extend(episode.samples)
        batch = resources.replay.sample(2)
        payload = batch.samples[0].payload
        assert isinstance(payload, DecisionExperience)
        object.__setattr__(payload, "reward_fingerprint", "0" * 64)

        with pytest.raises(ValueError, match="different reward contract"):
            resources.learner.update(batch, replay=resources.replay)
    finally:
        resources.close()


def test_learner_rejects_non_finite_state_before_optimizer_step() -> None:
    resources = build_training_resources(_small_config(), backend=FakeCombatBackend())
    try:
        episode = resources.collector.collect_episode(epsilon=1.0)
        resources.replay.extend(episode.samples)
        batch = resources.replay.sample(2)
        with torch.no_grad():
            next(resources.model.parameters()).fill_(float("nan"))

        with pytest.raises(FloatingPointError, match="model parameters"):
            resources.learner.update(batch, replay=resources.replay)
    finally:
        resources.close()


def test_atomic_checkpoint_roundtrip_and_runtime(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setenv("STS2_ARTIFACT_ROOT", str(tmp_path))
    # This checkpoint/timing test intentionally exercises the learner in its
    # only two collected steps.  Opt into the legacy warm-up credit behavior;
    # the default discard policy is covered independently below.
    base_config = _small_config()
    config = replace(
        base_config,
        runtime=replace(base_config.runtime, warmup_credit_policy="accrue"),
    )
    first_backend = FakeCombatBackend()

    state = run_training(config, backend=first_backend)

    assert state.environment_steps == 2
    assert state.learner_updates > 0
    metric_directories = list((tmp_path / "tests" / "run").glob("run-*"))
    assert len(metric_directories) == 1
    metric_records = [
        json.loads(line)
        for line in (metric_directories[0] / "metrics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    train_record = next(
        record for record in metric_records if record["event"] == "train_episode"
    )
    timing_payload = train_record["timings"]
    assert timing_payload["schema_version"] == 1
    assert timing_payload["scope"] == "train_episode"
    assert timing_payload["unit"] == "milliseconds"
    stages = timing_payload["stages"]
    assert stages["collect_episode"]["count"] == 1
    assert stages["replay_extend"]["count"] == 1
    assert stages["collector.reset"]["count"] == 1
    assert stages["collector.target_finalize"]["count"] == 1
    for name in (
        "collector.observation_encoding",
        "collector.policy_forward",
        "collector.sim_step",
        "collector.transition_reward_compaction",
    ):
        assert stages[name]["count"] == train_record["episode"]["steps"]
    assert stages["replay_sample"]["count"] == state.learner_updates
    assert stages["learner_update"]["count"] == state.learner_updates
    for name in (
        "learner.encoding",
        "learner.forward",
        "learner.loss_compute",
        "learner.backward",
        "learner.finite_check",
        "learner.optimizer_step",
        "learner.replay_priority",
        "learner.total",
    ):
        stage = stages[name]
        assert stage["count"] == state.learner_updates
        assert 0.0 <= stage["min_ms"] <= stage["mean_ms"] <= stage["max_ms"]
        assert stage["total_ms"] >= stage["max_ms"]
    run_directories = list((tmp_path / "tests" / "checkpoints").glob("run-*"))
    assert len(run_directories) == 1
    final = run_directories[0] / "final-step-000000002"
    assert (final / "checkpoint.manifest.json").is_file()
    periodic = run_directories[0] / "step-000000002"
    periodic_metadata = json.loads((periodic / "metadata.json").read_text(encoding="utf-8"))
    final_metadata = json.loads((final / "metadata.json").read_text(encoding="utf-8"))
    assert periodic_metadata["provenance"]["checkpoint_load_mode"] == "fresh"
    assert final_metadata["provenance"]["checkpoint_load_mode"] == "fresh"
    assert final_metadata["provenance"]["parent_checkpoint"]["relation"] == (
        "in_process_successor"
    )

    second_backend = FakeCombatBackend()
    resources = build_training_resources(config, backend=second_backend)
    loaded = load_training_checkpoint(final, config=config, resources=resources)
    assert loaded == state
    assert len(resources.replay) == 2
    resources.close()


def test_training_discards_unusable_warmup_credit(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("STS2_ARTIFACT_ROOT", str(tmp_path))

    state = run_training(_small_config(), backend=FakeCombatBackend())

    assert state.environment_steps == 2
    assert state.learner_updates == 0
    assert state.update_credit == 0


def test_checkpoint_rejects_config_drift(tmp_path: Path) -> None:
    config = _small_config()
    resources = build_training_resources(config, backend=FakeCombatBackend())
    episode = resources.collector.collect_episode(epsilon=1.0)
    resources.replay.extend(episode.samples)
    checkpoint = save_training_checkpoint(
        tmp_path / "checkpoint",
        config=config,
        resources=resources,
        state=TrainingState(environment_steps=2, episodes=1),
    )
    drifted = replace(
        config,
        optimization=replace(config.optimization, learning_rate=1.0e-4),
    )
    other = build_training_resources(drifted, backend=FakeCombatBackend())
    try:
        try:
            load_training_checkpoint(checkpoint, config=drifted, resources=other)
        except ValueError as exc:
            assert "immutable lineage config" in str(exc)
        else:  # pragma: no cover
            raise AssertionError("config drift should fail closed")
    finally:
        resources.close()
        other.close()


def test_overlap_checkpoint_restores_and_publishes_authoritative_model(
    tmp_path: Path,
) -> None:
    base = _small_config()
    config = replace(
        base,
        runtime=replace(base.runtime, execution_mode="overlap"),
    )
    source = build_training_resources(config, backend=FakeCombatBackend())
    target = build_training_resources(config, backend=FakeCombatBackend())
    try:
        with torch.no_grad():
            next(source.model.parameters()).fill_(0.25)
            next(source.collector_model.parameters()).fill_(0.75)
        checkpoint = save_training_checkpoint(
            tmp_path / "overlap-authority",
            config=config,
            resources=source,
            state=TrainingState(),
        )
        metadata = json.loads(
            (checkpoint / "metadata.json").read_text(encoding="utf-8")
        )
        assert metadata["resolved_collector_device"] == "cpu"

        load_training_checkpoint(checkpoint, config=config, resources=target)

        for learner_value, collector_value in zip(
            target.model.state_dict().values(),
            target.collector_model.state_dict().values(),
            strict=True,
        ):
            assert torch.equal(learner_value, collector_value)
        target_parameter = next(target.model.parameters())
        assert torch.equal(
            target_parameter,
            torch.full_like(target_parameter, 0.25),
        )
    finally:
        source.close()
        target.close()


def test_checkpoint_refuses_invalid_encoded_replay_snapshot(tmp_path: Path) -> None:
    config = _small_config()
    resources = build_training_resources(config, backend=FakeCombatBackend())
    try:
        episode = resources.collector.collect_episode(epsilon=1.0)
        resources.replay.extend(episode.samples)
        payload = resources.replay.samples[0].payload
        assert isinstance(payload, DecisionExperience)
        object.__setattr__(payload, "source_fingerprint", "not-a-sha256")

        with pytest.raises(ValueError, match="source fingerprint"):
            save_training_checkpoint(
                tmp_path / "invalid-snapshot",
                config=config,
                resources=resources,
                state=TrainingState(environment_steps=2, episodes=1),
            )
        assert not (tmp_path / "invalid-snapshot").exists()
    finally:
        resources.close()


def test_checkpoint_rejects_grounding_encoder_drift(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    config = _small_config()
    source = build_training_resources(config, backend=FakeCombatBackend())
    checkpoint = save_training_checkpoint(
        tmp_path / "encoder-contract",
        config=config,
        resources=source,
        state=TrainingState(),
    )
    target = build_training_resources(config, backend=FakeCombatBackend())
    monkeypatch.setattr(
        checkpointing_module,
        "grounding_encoding_identity",
        lambda: {
            "version": "grounded-structural-encoding-v2",
            "min_token_feature_dim": 128,
            "feature_abi_end": 119,
            "fingerprint_sha256": "0" * 64,
        },
    )
    try:
        with pytest.raises(ValueError, match="encoding contract"):
            load_training_checkpoint(checkpoint, config=config, resources=target)
    finally:
        source.close()
        target.close()


def test_exact_resume_allows_only_execution_budget_extension(tmp_path: Path) -> None:
    config = _small_config()
    resources = build_training_resources(config, backend=FakeCombatBackend())
    checkpoint = save_training_checkpoint(
        tmp_path / "extendable",
        config=config,
        resources=resources,
        state=TrainingState(environment_steps=2, episodes=1),
    )
    extended = replace(
        config,
        runtime=replace(
            config.runtime,
            total_environment_steps=10,
            checkpoint_interval_steps=5,
            evaluation_interval_steps=5,
        ),
    )
    target = build_training_resources(extended, backend=FakeCombatBackend())
    try:
        loaded = load_training_checkpoint(
            checkpoint,
            config=extended,
            resources=target,
        )
        assert loaded.environment_steps == 2
    finally:
        resources.close()
        target.close()


def test_exact_resume_restores_all_training_rng_and_update_credit(
    tmp_path: Path,
) -> None:
    config = _small_config()
    resources = build_training_resources(config, backend=FakeCombatBackend())
    episode = resources.collector.collect_episode(epsilon=1.0)
    resources.replay.extend(episode.samples)
    collector_state = resources.collector.state_dict()
    checkpoint = save_training_checkpoint(
        tmp_path / "exact",
        config=config,
        resources=resources,
        state=TrainingState(
            environment_steps=2,
            learner_updates=1,
            episodes=1,
            update_credit=2,
        ),
    )
    expected_python = random.random()
    expected_numpy = float(np.random.random())
    expected_torch = torch.rand(4)

    restored = build_training_resources(config, backend=FakeCombatBackend())
    try:
        state = load_training_checkpoint(
            checkpoint,
            config=config,
            resources=restored,
        )
        assert state.update_credit == 2
        assert restored.collector.state_dict() == collector_state
        assert random.random() == expected_python
        assert float(np.random.random()) == expected_numpy
        assert torch.equal(torch.rand(4), expected_torch)
    finally:
        resources.close()
        restored.close()


def test_model_initialization_starts_fresh_profile_state(tmp_path: Path) -> None:
    combat_config = _small_config()
    source = build_training_resources(combat_config, backend=FakeCombatBackend())
    checkpoint = save_training_checkpoint(
        tmp_path / "combat-source",
        config=combat_config,
        resources=source,
        state=TrainingState(environment_steps=123, learner_updates=17, episodes=9),
    )
    source_parameters = {
        name: value.detach().clone()
        for name, value in source.model.state_dict().items()
    }
    run_config = replace(
        combat_config,
        profile="test-run",
        environment=replace(combat_config.environment, scenario="full-run"),
        curriculum=replace(combat_config.curriculum, reward_objective="run"),
    )
    target = build_training_resources(run_config, backend=FakeCombatBackend())
    try:
        initialize_model_from_checkpoint(
            checkpoint,
            config=run_config,
            resources=target,
        )
        assert len(target.replay) == 0
        for name, value in target.model.state_dict().items():
            assert torch.equal(value, source_parameters[name])
        assert target.collector.state_dict()["episode_seed"] == training_seed_start(
            run_config.runtime.seed
        )
    finally:
        source.close()
        target.close()


def test_resume_target_is_rejected_before_artifacts_or_backend_side_effects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _small_config()
    source = build_training_resources(config, backend=FakeCombatBackend())
    checkpoint = save_training_checkpoint(
        tmp_path / "completed-source",
        config=config,
        resources=source,
        state=TrainingState(environment_steps=config.runtime.total_environment_steps),
    )
    source.close()

    artifact_root = tmp_path / "must-not-exist"
    monkeypatch.setenv("STS2_ARTIFACT_ROOT", str(artifact_root))
    untouched_backend = FakeCombatBackend()
    with pytest.raises(ValueError, match="already reached"):
        run_training(config, backend=untouched_backend, resume_from=checkpoint)

    assert not artifact_root.exists()
    assert untouched_backend.closed is False
