from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import pytest

from muzero.training.cli_args import build_arg_parser
from sts2_rl.training import (
    CONFIG_VERSION,
    TrainingConfig,
    TrainingResources,
    TrainingRuntime,
    build_legacy_trainer,
    parse_args_with_config,
)


def test_default_profile_selects_active_token_architecture() -> None:
    args = parse_args_with_config(build_arg_parser(), [])
    assert args.obs_mode == "token_v3"
    assert args.model_arch == "token_memory_v1"
    assert args.profile == "default"
    config = TrainingConfig.from_namespace(args)
    assert config.version == CONFIG_VERSION
    assert config.model.obs_mode == "token_v3"
    assert args.combat_hard_guard_policy == "off"
    assert args.build_hard_guard_policy == "off"
    assert args.hard_guard_target_rewrite == "off"
    assert args.route_safety_guard is False


def test_safety_deadlock_profile_is_explicit_guard_opt_in() -> None:
    args = parse_args_with_config(
        build_arg_parser(),
        ["--profile", "safety-deadlock"],
    )
    assert args.profile == "safety-deadlock"
    assert args.combat_hard_guard_policy == "emergency"
    assert args.build_hard_guard_policy == "emergency"
    assert args.hard_guard_target_rewrite == "off"
    assert args.route_safety_guard is True


def test_combat_profile_does_not_rewrite_actions_by_default() -> None:
    args = parse_args_with_config(build_arg_parser(), ["--profile", "combat"])
    assert args.combat_hard_guard_policy == "off"
    assert args.build_hard_guard_policy == "off"
    assert args.hard_guard_target_rewrite == "off"


def test_cli_and_set_override_profile_defaults() -> None:
    args = parse_args_with_config(
        build_arg_parser(),
        ["--batch-size", "12", "--set", "optimization.learning_rate=0.0002"],
    )
    assert args.batch_size == 12
    assert args.learning_rate == pytest.approx(0.0002)


def test_custom_toml_merges_over_profile(tmp_path: Path) -> None:
    config_path = tmp_path / "training.toml"
    config_path.write_text(
        f'''version = "{CONFIG_VERSION}"
profile = "custom"

[optimization]
batch_size = 7
''',
        encoding="utf-8",
    )
    args = parse_args_with_config(
        build_arg_parser(),
        ["--config", str(config_path)],
    )
    assert args.profile == "custom"
    assert args.batch_size == 7


def test_dense_legacy_pair_is_rejected_by_active_training_config() -> None:
    args = parse_args_with_config(
        build_arg_parser(),
        ["--obs-mode", "dense_v2", "--model-arch", "dense_v1"],
    )
    with pytest.raises(ValueError, match="active training requires"):
        TrainingConfig.from_namespace(args)


class FakeTrainer:
    def __init__(
        self,
        network,
        mcts,
        buffer,
        env,
        optimizer,
        *,
        device="cpu",
        semantic_policy_weight=1.0,
        trivial_build_fast_path=True,
    ):
        self.network = network
        self.mcts = mcts
        self.buffer = buffer
        self.env = env
        self.optimizer = optimizer
        self.device = device
        self.semantic_policy_weight = semantic_policy_weight
        self.trivial_build_fast_path = trivial_build_fast_path
        self.writer = FakeWriter()
        self.saved: list[str] = []

    def compute_temperature(self, step, total):
        return 0.5

    def self_play_episode(self, temperature):
        return (1.0, 2)

    def self_play_episode_on_env(self, env, temperature):
        return (2.0, 3)

    def train_step(self, batch_size, unroll_steps):
        return {"loss/total": 1.0}

    def save_checkpoint(self, tag=""):
        self.saved.append(tag)


class FakeWriter:
    def __init__(self):
        self.closed = False

    def add_scalar(self, name, value, step):
        pass

    def flush(self):
        pass

    def close(self):
        self.closed = True


class FakeReplay:
    def __init__(self):
        self.items = []

    def __len__(self):
        return len(self.items)

    def save_episode(self, trajectory, *, discount, n_steps):
        self.items.append((trajectory, discount, n_steps))


def _minimal_config() -> TrainingConfig:
    args = parse_args_with_config(build_arg_parser(), [])
    return TrainingConfig.from_namespace(args)


def test_factory_collapses_constructor_and_applies_derived_options() -> None:
    config = _minimal_config()
    resources = TrainingResources("net", "mcts", FakeReplay(), "env", "optim")
    trainer = build_legacy_trainer(FakeTrainer, resources=resources, config=config)
    assert trainer.network == "net"
    assert trainer.semantic_policy_weight == config.option("semantic_policy_weight")
    assert trainer.trivial_build_fast_path is (
        not bool(config.option("disable_trivial_build_fast_path"))
    )


def test_training_runtime_routes_core_operations_through_services() -> None:
    replay = FakeReplay()
    trainer = FakeTrainer("net", "mcts", replay, "env", "optim")
    runtime = TrainingRuntime.from_legacy(trainer, _minimal_config())

    assert runtime.collector.temperature(0, 100) == 0.5
    assert runtime.collector.collect_episode(temperature=1.0) == (1.0, 2)
    assert runtime.learner.update(batch_size=4, unroll_steps=2)["loss/total"] == 1.0
    runtime.replay_store.save_episode("trajectory", discount=0.9, n_steps=3)
    runtime.checkpoints.save(tag="final")
    runtime.telemetry.close()

    assert replay.items == [("trajectory", 0.9, 3)]
    assert trainer.saved == ["final"]
    assert trainer.writer.closed
