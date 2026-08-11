from __future__ import annotations

import importlib.util
import io
import json
import random
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
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
    _append(
        pipeline / "collector-1-metrics.jsonl",
        {
            "event": "collector_episode",
            "unix_s": 20.0,
            "producer_id": "collector-1",
            "run_nonce": "restart-b",
            "episode_index": 0,
            "seed": 1,
            "steps": 20,
            "run_won": False,
        },
    )
    _append(
        pipeline / "collector-2-metrics.jsonl",
        {
            "event": "collector_episode",
            "unix_s": 10.0,
            "producer_id": "collector-2",
            "run_nonce": "restart-a",
            "episode_index": 0,
            "seed": 2,
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
            "ingested_total": 0,
        },
        {
            "event": "trainer_ingest",
            "unix_s": 30.0,
            "environment_steps": 777,
            "ingested_total": 2,
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
    assert episodes[0]["episode_id"] == "collector-2:restart-a:0"
    update = next(row for row in rows if row["event"] == "learner_update")
    assert update["environment_steps"] == 777

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
            "ingested_total": 3,
            "learner": {"updates": 12},
        },
    )
    second = io.StringIO()
    assert dashboard._publish_available(pipeline, manifest, restored, second) is True
    second_rows = [json.loads(line) for line in second.getvalue().splitlines()]
    assert [row["event"] for row in second_rows] == ["run_complete"]
    assert second_rows[0]["environment_steps"] == 888


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
