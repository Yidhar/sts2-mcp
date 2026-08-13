from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace


SCRIPT_ROOT = Path(__file__).resolve().parents[1] / "scripts"


def _script(name: str) -> ModuleType:
    path = SCRIPT_ROOT / name
    spec = importlib.util.spec_from_file_location(f"test_{path.stem}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _append(path: Path, *rows: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_parallel_dashboard_uses_learner_acknowledgement_order(tmp_path: Path) -> None:
    bridge = _script("run_hybrid_dashboard_bridge.py")
    run_root = tmp_path / "run"
    (run_root / "launcher").mkdir(parents=True)
    (run_root / "status").mkdir()
    (run_root / "launcher" / "launch.json").write_text(
        json.dumps({"created_at": "2026-08-13T12:00:00+08:00"}),
        encoding="utf-8",
    )
    (run_root / "status" / "trainer.json").write_text(
        json.dumps({"state": "running"}), encoding="utf-8"
    )
    # Diagnostic completion order differs from durable trainer ingest order.
    _append(
        run_root / "metrics" / "collector-0.jsonl",
        {"event": "hybrid_collector_episode", "unix_s": 10.0, "reset_seed": 10},
    )
    _append(
        run_root / "metrics" / "collector-1.jsonl",
        {"event": "hybrid_collector_episode", "unix_s": 9.0, "reset_seed": 20},
    )
    for ingested, producer, seed, pending in (
        (1, "collector-0", 10, 1),
        (2, "collector-1", 20, 0),
    ):
        _append(
            run_root / "metrics" / "trainer.jsonl",
            {
                "event": "hybrid_update",
                "unix_s": 20.0 + ingested,
                "ingested_runs": ingested,
                "environment_steps": ingested * 100,
                "producer_id": producer,
                "episode": {
                    "producer_id": producer,
                    "reset_seed": seed,
                    "steps": 100,
                    "max_floor": 5,
                    "revivals_used": 1,
                    "terminal_reason": "run_defeat",
                    "run_won": False,
                },
                "spool_pending": pending,
                "spool_capacity": 2,
                "learner": {"updates": ingested},
                "replay": {},
            },
        )

    args = SimpleNamespace(
        run_root=str(run_root),
        lineage="hybrid-parallel",
        run_id="run-a",
        target_runs=100,
        collector_episode_capacity=120,
        collector_count=2,
        maximum_pending_runs=2,
        revival_budget=64,
        curriculum_mode="hybrid-completion-preheat",
    )
    events = bridge._projection(args)
    episodes = [row for row in events if row["event"] == "train_episode"]
    assert [row["producer_id"] for row in episodes] == ["collector-0", "collector-1"]
    assert [row["reset_seed"] for row in episodes] == [10, 20]
    assert events[0]["config"]["runtime"]["collector_workers"] == 2
    updates = [row for row in events if row["event"] == "learner_update"]
    assert [row["rollout_queue"]["size"] for row in updates] == [1, 0]


def test_parallel_dashboard_resume_starts_at_checkpoint_counters(tmp_path: Path) -> None:
    bridge = _script("run_hybrid_dashboard_bridge.py")
    run_root = tmp_path / "resumed"
    (run_root / "launcher").mkdir(parents=True)
    (run_root / "status").mkdir()
    (run_root / "launcher" / "launch.json").write_text(
        json.dumps(
            {
                "created_at": "2026-08-13T12:00:00+08:00",
                "resumed_from_ingested_runs": 30,
                "source_environment_steps": 23080,
                "source_learner_updates": 60,
            }
        ),
        encoding="utf-8",
    )
    (run_root / "status" / "trainer.json").write_text(
        json.dumps({"state": "running"}), encoding="utf-8"
    )
    _append(
        run_root / "metrics" / "trainer.jsonl",
        {
            "event": "hybrid_update",
            "unix_s": 31.0,
            "ingested_runs": 31,
            "environment_steps": 23180,
            "episode": {
                "producer_id": "collector-1",
                "reset_seed": 20,
                "steps": 100,
            },
            "learner": {"updates": 61},
            "replay": {},
        },
    )
    args = SimpleNamespace(
        run_root=str(run_root),
        lineage="hybrid-parallel",
        run_id="run-b",
        target_runs=128,
        collector_episode_capacity=200,
        collector_count=2,
        maximum_pending_runs=2,
        revival_budget=64,
        curriculum_mode="hybrid-completion-preheat",
    )
    events = bridge._projection(args)
    assert events[0]["checkpoint_load"]["mode"] == "exact_resume"
    assert events[0]["state"] == {
        "environment_steps": 23080,
        "episodes": 30,
        "learner_updates": 60,
        "policy_version": 60,
        "evaluation_episodes": 0,
    }
    episode = next(row for row in events if row["event"] == "train_episode")
    assert episode["episode_id"] == "hybrid-run-31"
    assert episode["environment_steps"] == 23180


def test_parallel_launcher_defaults_to_two_collectors_and_one_update() -> None:
    script = (SCRIPT_ROOT / "train_hybrid_parallel_wsl_rocm.sh").read_text(
        encoding="utf-8"
    )
    assert 'COLLECTORS="${STS2_HYBRID_COLLECTORS:-2}"' in script
    assert '--updates-per-run "${STS2_HYBRID_UPDATES_PER_RUN:-1}"' in script
    assert '--ingest-batch-runs "${STS2_HYBRID_INGEST_BATCH_RUNS:-2}"' in script
    assert '--target-sync-every-runs "${STS2_HYBRID_TARGET_SYNC_RUNS:-100}"' in script
    assert 'COLLECTOR_DEVICE="${STS2_HYBRID_COLLECTOR_DEVICE:-cuda}"' in script
    assert "--producer-id \"collector-$index\"" in script
