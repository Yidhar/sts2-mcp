from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch

from sts2_rl.contracts import (
    BackendCapabilities,
    CombatResetRequest,
    EnvironmentResult,
    EnvironmentTransition,
    ResetRequest,
    StepRequest,
)
from sts2_rl.training import (
    CurriculumConfig,
    EnvironmentConfig,
    ModelConfig,
    OptimizationConfig,
    RolloutConfig,
    RuntimeConfig,
    TrainingConfig,
    build_training_resources,
    evaluate_policy,
    load_training_checkpoint,
    save_training_checkpoint,
)
from sts2_rl.training.checkpointing import TrainingState
from sts2_rl.training.collector import CollectionProtocolError
from sts2_rl.training.pipeline import ActorLearnerPipeline


class FakeCombatBackend:
    def __init__(self, *, terminal_step: int = 2) -> None:
        self._capabilities = BackendCapabilities(
            backend_name="fake",
            session_id="fake-session",
        )
        self._state_version = 0
        self._step = 0
        self._episode = 0
        self.terminal_step = terminal_step
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
    def _actions() -> tuple[dict[str, Any], ...]:
        return (
            {
                "action_handle": "attack",
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

    def _observation(self, *, terminal: bool) -> dict[str, Any]:
        return {
            "phase": "combat",
            "decision_domain": "combat",
            "player": {"id": "player", "hp": 50, "max_hp": 80},
            "combat": {
                "in_progress": not terminal,
                "enemies": [
                    {
                        "id": "enemy",
                        "hp": 0 if terminal else max(1, 30 - 5 * self._step),
                        "max_hp": 30,
                    }
                ],
            },
            "run": {"act": 1, "floor": 1},
        }

    def reset(self, request: ResetRequest) -> EnvironmentResult:
        raise AssertionError("combat fake must use combat_reset")

    def combat_reset(self, request: CombatResetRequest) -> EnvironmentResult:
        assert request.expected_state_version == self._state_version
        self.reset_seeds.append(request.seed)
        before = self._state_version
        self._state_version += 1
        self._step = 0
        self._episode += 1
        episode_id = f"episode-{self._episode}"
        return EnvironmentResult(
            episode_id=episode_id,
            step_index=0,
            observation=self._observation(terminal=False),
            legal_actions=self._actions(),
            transition=EnvironmentTransition(
                episode_id=episode_id,
                step_index=0,
                before_state_version=before,
                after_state_version=self._state_version,
                facts={"combat_result": "none", "terminal_reason": None},
            ),
            info={"reward_authority": "external-rl"},
        )

    def step(self, request: StepRequest) -> EnvironmentResult:
        assert request.expected_step_index == self._step
        before = self._state_version
        self._state_version += 1
        self._step += 1
        terminal = self._step >= self.terminal_step
        reason = "combat_victory" if terminal else None
        return EnvironmentResult(
            episode_id=f"episode-{self._episode}",
            step_index=self._step,
            observation=self._observation(terminal=terminal),
            legal_actions=() if terminal else self._actions(),
            transition=EnvironmentTransition(
                episode_id=f"episode-{self._episode}",
                step_index=self._step,
                before_state_version=before,
                after_state_version=self._state_version,
                facts={
                    "combat_result": "victory" if terminal else "none",
                    "terminal_reason": reason,
                },
            ),
            terminated=terminal,
            terminal_reason=reason,
            info={"reward_authority": "external-rl"},
        )

    def close(self) -> None:
        self.closed = True


class StaleRevisionBackend(FakeCombatBackend):
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


def _config(*, total_steps: int = 4) -> TrainingConfig:
    return TrainingConfig(
        profile="v2-test",
        model=ModelConfig(
            token_feature_dim=128,
            d_model=32,
            n_heads=4,
            ffn_dim=64,
            world_layers=1,
            latent_slots=4,
            latent_layers=1,
            local_layers=1,
            candidate_layers=1,
            recurrent_hidden_dim=32,
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
        ),
        optimization=OptimizationConfig(batch_unrolls=1),
        rollout=RolloutConfig(
            unroll_length=2,
            queue_capacity=8,
            minimum_unrolls=1,
            policy_sync_interval_unrolls=1,
            max_policy_lag=32,
        ),
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
            collector_device="cpu",
            total_environment_steps=total_steps,
            seed=7,
            log_dir="tests/v2-run",
            checkpoint_dir="tests/v2-checkpoints",
            checkpoint_interval_steps=100,
            evaluation_steps=(),
            evaluation_episodes=0,
        ),
    )


def test_collector_emits_contiguous_recurrent_unroll() -> None:
    resources = build_training_resources(_config(), backend=FakeCombatBackend())
    try:
        episode = resources.collector.collect_episode(
            epsilon=0.2,
            record=True,
            policy_version=3,
        )
        assert episode.metrics.steps == 2
        assert episode.metrics.combat_won
        assert not episode.metrics.deadlocked
        assert len(episode.unrolls) == 1
        unroll = episode.unrolls[0]
        assert unroll.policy_version == 3
        assert len(unroll.steps) == 2
        assert unroll.bootstrap_snapshot is None
        assert unroll.steps[-1].discount == 0.0
        assert unroll.initial_recurrent_state.shape == (32,)
    finally:
        resources.close()


def test_runtime_budget_cut_bootstraps_instead_of_fabricating_preheat_loss() -> None:
    base = _config(total_steps=1)
    config = replace(
        base,
        optimization=replace(base.optimization, discount=1.0),
        curriculum=CurriculumConfig(
            mode="native-revival-preheat",
            reward_objective="combat",
            revival_relic_id="RELIC.LIZARD_TAIL",
            revival_budget=-1,
            epsilon_start=0.2,
            epsilon_end=0.1,
            epsilon_decay_steps=10,
        ),
    )
    resources = build_training_resources(
        config,
        backend=FakeCombatBackend(terminal_step=10),
    )
    try:
        episode = resources.collector.collect_episode(
            record=True,
            maximum_steps=1,
        )
        assert episode.metrics.terminal_reason == "collection_budget"
        assert not episode.metrics.combat_won
        assert episode.metrics.reward_total > -0.1
        assert len(episode.unrolls) == 1
        assert episode.unrolls[0].steps[-1].discount == 1.0
        assert episode.unrolls[0].bootstrap_snapshot is not None
    finally:
        resources.close()


def test_collector_fails_closed_on_stale_transition_revision() -> None:
    resources = build_training_resources(_config(), backend=StaleRevisionBackend())
    try:
        with pytest.raises(CollectionProtocolError, match="stale"):
            resources.collector.collect_episode(record=True)
    finally:
        resources.close()


def test_vtrace_learner_updates_policy_value_and_recurrent_parameters() -> None:
    resources = build_training_resources(_config(), backend=FakeCombatBackend())
    try:
        unroll = resources.collector.collect_episode(record=True).unrolls[0]
        before = {
            name: value.detach().clone()
            for name, value in resources.model.state_dict().items()
        }
        metrics = resources.learner.update((unroll,), current_policy_version=0)
        assert metrics.environment_steps == 2
        assert metrics.unrolls == 1
        assert torch.isfinite(torch.tensor(metrics.loss))
        assert metrics.importance_ratio_mean > 0.0
        assert any(
            not torch.equal(before[name], value)
            for name, value in resources.model.state_dict().items()
        )
    finally:
        resources.close()


def test_async_pipeline_streams_fifo_data_and_finishes_exact_horizon() -> None:
    config = _config(total_steps=4)
    resources = build_training_resources(config, backend=FakeCombatBackend())
    pipeline = ActorLearnerPipeline(
        resources,
        total_environment_steps=4,
        starting_environment_steps=0,
        starting_policy_version=0,
        epsilon=lambda _: 0.1,
    )
    try:
        pipeline.start()
        first = resources.rollout_queue.get_batch(1, minimum=1, timeout=5.0)
        metrics = resources.learner.update(first, current_policy_version=0)
        assert metrics.environment_steps == 2
        pipeline.request_policy_publication(1)
        pipeline.join(timeout=10.0)
        assert pipeline.environment_steps == 4
        episodes = []
        while True:
            episode = pipeline.next_episode(timeout=0.0)
            if episode is None:
                break
            episodes.append(episode)
        assert sum(item.metrics.steps for item in episodes) == 4
    finally:
        resources.close()


def test_evaluation_uses_odd_heldout_seeds_and_records_no_unrolls(
    tmp_path: Path,
) -> None:
    backend = FakeCombatBackend()
    resources = build_training_resources(_config(), backend=backend)
    try:
        episodes, summary = evaluate_policy(
            resources,
            episodes=2,
            base_seed=6,
            journal_path=tmp_path / "trajectory.jsonl",
        )
        assert len(episodes) == 2
        assert summary["combat_win_rate"] == 1.0
        assert all(int(seed) % 2 == 1 for seed in backend.reset_seeds)
        assert (tmp_path / "trajectory.jsonl").read_text(encoding="utf-8")
    finally:
        resources.close()


def test_v2_checkpoint_roundtrip_restores_models_optimizer_queue_and_rng(
    tmp_path: Path,
) -> None:
    config = _config()
    source = build_training_resources(config, backend=FakeCombatBackend())
    try:
        unroll = source.collector.collect_episode(record=True).unrolls[0]
        source.rollout_queue.put(unroll)
        state = TrainingState(
            environment_steps=2,
            learner_updates=0,
            episodes=1,
            policy_version=0,
            actor_policy_version=0,
        )
        checkpoint = save_training_checkpoint(
            tmp_path / "checkpoint",
            config=config,
            resources=source,
            state=state,
            run_id="test-run",
            checkpoint_load_mode="fresh",
        )
    finally:
        source.close()

    restored = build_training_resources(config, backend=FakeCombatBackend())
    try:
        loaded = load_training_checkpoint(
            checkpoint,
            config=config,
            resources=restored,
        )
        assert loaded == state
        assert len(restored.rollout_queue) == 1
        assert restored.rollout_queue.snapshot()[0].episode_id == unroll.episode_id
        assert (checkpoint / "actor_network.pt").is_file()
        assert (checkpoint / "rollout_queue.pkl").is_file()
        assert not (checkpoint / "replay_buffer.pkl").exists()
    finally:
        restored.close()
