from __future__ import annotations

import importlib.util
import io
import json
import random
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from sts2_rl.macro import MacroQConfig, MacroQLearner, MacroSequenceReplay

SCRIPT_ROOT = Path(__file__).resolve().parents[1] / "scripts"


def _script(name: str) -> ModuleType:
    path = SCRIPT_ROOT / name
    spec = importlib.util.spec_from_file_location(f"test_{path.stem}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _append(path: Path, *rows: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_parallel_training_state_restores_models_learner_replay_and_rng(
    tmp_path: Path,
) -> None:
    trainer = _script("run_stage2_trainer.py")
    online = torch.nn.Linear(3, 2)
    target = torch.nn.Linear(3, 2)
    replay = MacroSequenceReplay(capacity_episodes=7, window_length=4, seed=19)
    learner = MacroQLearner(
        online_parameters=list(online.parameters()),
        forward_online=lambda _step, hidden: (torch.zeros(1), hidden),
        forward_target=lambda _step, hidden: (torch.zeros(1), hidden),
        sync_target=lambda: None,
        initial_state=lambda: None,
        replay=replay,
        config=MacroQConfig(sample_windows=1),
    )
    # Materialize Adam state and non-zero schedule counters.
    learner.optimizer.zero_grad(set_to_none=True)
    online(torch.ones(1, 3)).sum().backward()
    learner.optimizer.step()
    learner.metrics.updates = 17
    learner.metrics.executed_counts = {"rest:smith": 3}
    expected_online = {key: value.detach().clone() for key, value in online.state_dict().items()}
    expected_target = {key: value.detach().clone() for key, value in target.state_dict().items()}

    random.seed(11)
    np.random.seed(12)
    torch.manual_seed(13)
    state_path = tmp_path / "training-state.pt"
    trainer._save_training_state(
        state_path,
        online=online,
        target=target,
        learner=learner,
        replay=replay,
        lineage_id="lineage-a",
        control_domain="macro",
        ingested=41,
        environment_steps=12345,
    )
    persisted = torch.load(state_path, map_location="cpu", weights_only=False)
    assert persisted["policy_version"] == 17
    expected_rng = (random.random(), float(np.random.random()), float(torch.rand(())))

    with torch.no_grad():
        for parameter in online.parameters():
            parameter.zero_()
        for parameter in target.parameters():
            parameter.zero_()
    learner.metrics.updates = 0
    random.seed(99)
    np.random.seed(99)
    torch.manual_seed(99)

    ingested, environment_steps = trainer._load_training_state(
        state_path,
        online=online,
        target=target,
        learner=learner,
        replay=replay,
        lineage_id="lineage-a",
        control_domain="macro",
        device=torch.device("cpu"),
    )
    assert (ingested, environment_steps) == (41, 12345)
    assert learner.metrics.updates == 17
    assert learner.metrics.executed_counts == {"rest:smith": 3}
    for key, expected in expected_online.items():
        assert torch.equal(online.state_dict()[key], expected)
    for key, expected in expected_target.items():
        assert torch.equal(target.state_dict()[key], expected)
    actual_rng = (random.random(), float(np.random.random()), float(torch.rand(())))
    assert actual_rng == expected_rng


def test_dashboard_restart_uses_persisted_offsets_and_trainer_progress(
    tmp_path: Path,
) -> None:
    dashboard = _script("publish_stage_dashboard.py")
    pipeline = tmp_path / "pipeline"
    pipeline.mkdir()
    manifest = {
        "format": dashboard.PIPELINE_FORMAT,
        "pipeline_id": "pipeline-a",
        "lineage_id": "lineage-a",
        "control_domain": "macro",
        "producer_ids": ["collector-1", "collector-2"],
    }
    (pipeline / "pipeline.json").write_text(json.dumps(manifest), encoding="utf-8")
    # Collector rows are not durable acknowledgements and must not be shown.
    _append(
        pipeline / "collector-2-metrics.jsonl",
        {
            "event": "collector_episode",
            "unix_s": 10.0,
            "producer_id": "collector-2",
            "run_nonce": "restart-a",
            "episode_index": 0,
            "reset_seed": 999,
            "steps": 10,
            "run_won": True,
        },
    )
    _append(
        pipeline / "trainer-metrics.jsonl",
        {
            "event": "trainer_start",
            "unix_s": 5.0,
            "environment_steps": 700,
            "ingested_total": 40,
        },
        {
            "event": "trainer_ingest",
            "unix_s": 30.0,
            "environment_steps": 777,
            "ingested_total": 42,
            "policy_version": 9,
            "acknowledged_episodes": [
                {
                    "episode_id": "ack-1",
                    "unix_s": 20.0,
                    "reset_seed": 2,
                    "steps": 10,
                    "run_won": True,
                    "ingested_total": 41,
                    "environment_steps": 710,
                },
                {
                    "episode_id": "ack-2",
                    "unix_s": 25.0,
                    "reset_seed": 1,
                    "steps": 20,
                    "run_won": False,
                    "ingested_total": 42,
                    "environment_steps": 730,
                },
            ],
            "learner": {"loss": 0.25, "td_error_mean": 0.5, "updates": 9},
        },
    )

    metrics_path, state, created = dashboard._load_or_create_state(
        pipeline,
        tmp_path / "runs",
        lineage="lineage-a",
        manifest=manifest,
    )
    assert created is True
    output = io.StringIO()
    assert dashboard._publish_available(pipeline, manifest, state, output) is False
    rows = [json.loads(line) for line in output.getvalue().splitlines()]
    episodes = [row for row in rows if row["event"] == "train_episode"]
    assert [row["reset_seed"] for row in episodes] == [2, 1]
    assert [row["environment_steps"] for row in episodes] == [710, 730]
    assert state["episodes"] == 42
    assert episodes[0]["episode_id"] == "ack-1"
    update = next(row for row in rows if row["event"] == "learner_update")
    assert update["environment_steps"] == 777
    assert update["policy_version"] == 9

    dashboard._atomic_json(pipeline / "dashboard-state.json", state)
    same_metrics, restored, created = dashboard._load_or_create_state(
        pipeline,
        tmp_path / "runs",
        lineage="lineage-a",
        manifest=manifest,
    )
    assert created is False
    assert same_metrics == metrics_path
    _append(
        pipeline / "trainer-metrics.jsonl",
        {
            "event": "trainer_complete",
            "unix_s": 40.0,
            "environment_steps": 888,
            "ingested_total": 43,
            "policy_version": 12,
            "learner": {"updates": 12},
        },
    )
    second = io.StringIO()
    assert dashboard._publish_available(pipeline, manifest, restored, second) is True
    second_rows = [json.loads(line) for line in second.getvalue().splitlines()]
    assert [row["event"] for row in second_rows] == ["run_complete"]
    assert second_rows[0]["environment_steps"] == 888
    assert second_rows[0]["policy_version"] == 12


def test_producer_status_summary_distinguishes_pending_complete_and_failed(
    tmp_path: Path,
) -> None:
    trainer = _script("run_stage2_trainer.py")
    (tmp_path / "collector-1.json").write_text(json.dumps({"state": "complete"}), encoding="utf-8")
    (tmp_path / "collector-2.json").write_text(json.dumps({"state": "failed"}), encoding="utf-8")
    assert trainer._producer_states(tmp_path, ("collector-1", "collector-2", "collector-3")) == {
        "collector-1": "complete",
        "collector-2": "failed",
        "collector-3": "pending",
    }
    with pytest.raises(RuntimeError, match="collector-2"):
        trainer._require_producers_not_failed(
            tmp_path, ("collector-1", "collector-2", "collector-3")
        )


def test_spool_item_is_deleted_only_after_checkpoint_succeeds(tmp_path: Path) -> None:
    trainer = _script("run_stage2_trainer.py")
    spool_item = tmp_path / "episode.pkl"
    spool_item.write_bytes(b"episode")

    def fail_save() -> None:
        raise OSError("checkpoint unavailable")

    published: list[str] = []
    with pytest.raises(OSError, match="checkpoint unavailable"):
        trainer._save_then_acknowledge(
            fail_save, lambda: published.append("ack"), [spool_item]
        )
    assert spool_item.is_file()
    assert published == []

    def publish() -> None:
        assert spool_item.is_file()
        published.append("ack")

    trainer._save_then_acknowledge(lambda: None, publish, [spool_item])
    assert not spool_item.exists()
    assert published == ["ack"]


def test_durable_acknowledgement_is_buffered_until_interval_or_final_target() -> None:
    trainer = _script("run_stage2_trainer.py")

    assert not trainer._checkpoint_due(
        ingested=41,
        last_saved_at=40,
        interval=20,
        target=100,
    )
    assert trainer._checkpoint_due(
        ingested=60,
        last_saved_at=40,
        interval=20,
        target=100,
    )
    assert trainer._checkpoint_due(
        ingested=100,
        last_saved_at=95,
        interval=20,
        target=100,
    )


def test_collector_publication_load_is_strict(tmp_path: Path) -> None:
    collector = _script("run_stage2_collector.py")
    model = torch.nn.Linear(3, 2)
    incomplete = {"weight": model.weight.detach().clone()}
    path = tmp_path / "publication.pt"
    torch.save(incomplete, path)

    with pytest.raises(RuntimeError, match="Missing key"):
        collector._load_published_model_state(model, path, torch.device("cpu"))


def test_collector_summary_uses_episode_reset_seed() -> None:
    collector = _script("run_stage2_collector.py")
    episode = SimpleNamespace(
        metrics=SimpleNamespace(
            episode_id="episode-1",
            reset_seed=123456,
            steps=17,
            run_won=False,
            max_floor=8,
            max_act=1,
            act1_cleared=False,
            revivals_used=2,
            player_hp_lost=31.0,
            reward_total=-0.5,
            terminal_reason="run_defeat",
        )
    )
    summary = collector._episode_summary(
        episode,
        producer_id="collector-1",
        run_nonce="nonce",
        episode_index=3,
        collected_at=42.0,
    )
    assert summary["reset_seed"] == 123456
    assert summary["episode_id"] == "collector-1:nonce:3"
    assert summary["source_episode_id"] == "episode-1"


def test_dashboard_publishes_trainer_failure_before_startup_completed(
    tmp_path: Path,
) -> None:
    dashboard = _script("publish_stage_dashboard.py")
    pipeline = tmp_path / "pipeline"
    pipeline.mkdir()
    manifest = {
        "format": dashboard.PIPELINE_FORMAT,
        "pipeline_id": "pipeline-startup-failure",
        "lineage_id": "lineage-a",
        "control_domain": "macro",
        "producer_ids": ["collector-1"],
    }
    _append(
        pipeline / "trainer-metrics.jsonl",
        {
            "event": "trainer_failed",
            "unix_s": 5.0,
            "error": "FileNotFoundError: missing model-init",
        },
    )
    _, state, _ = dashboard._load_or_create_state(
        pipeline,
        tmp_path / "runs",
        lineage="lineage-a",
        manifest=manifest,
    )
    output = io.StringIO()
    assert dashboard._publish_available(pipeline, manifest, state, output) is True
    assert json.loads(output.getvalue()) == {
        "event": "run_failed",
        "unix_s": 5.0,
        "environment_steps": 0,
        "error": "FileNotFoundError: missing model-init",
    }
    assert state["terminal"] is True
