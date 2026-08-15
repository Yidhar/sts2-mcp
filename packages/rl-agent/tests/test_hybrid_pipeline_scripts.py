from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

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
    (run_root / "status" / "trainer.json").write_text(json.dumps({"state": "running"}), encoding="utf-8")
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
                    "act1_revivals": 1,
                    "terminal_reason": "run_defeat",
                    "run_won": False,
                },
                "spool_pending": pending,
                "spool_capacity": 2,
                "learner": {
                    "updates": ingested,
                    "death_price": {
                        "lambda_value": 0.01 + ingested * 0.001,
                        "active_statistic": "act1",
                        "act1_ema": 5.0,
                        "run_ema": None,
                        "freeze_reason": "none",
                        "last_support_successes": 12,
                        "course": {"act1_target": 4.0, "run_target": 16.0},
                    },
                },
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
        collector_device="cuda",
        maximum_pending_runs=2,
        revival_budget=64,
        curriculum_mode="hybrid-completion-preheat",
    )
    events = bridge._projection(args)
    episodes = [row for row in events if row["event"] == "train_episode"]
    assert [row["producer_id"] for row in episodes] == ["collector-0", "collector-1"]
    assert [row["reset_seed"] for row in episodes] == [10, 20]
    assert [row["act1_revivals"] for row in episodes] == [1, 1]
    assert events[0]["config"]["runtime"]["collector_workers"] == 2
    assert events[0]["config"]["runtime"]["collector_device"] == "cuda"
    updates = [row for row in events if row["event"] == "learner_update"]
    assert [row["rollout_queue"]["size"] for row in updates] == [1, 0]
    assert updates[-1]["hybrid_death_price_lambda"] == pytest.approx(0.012)
    assert updates[-1]["hybrid_death_price_target"] == 4.0
    assert updates[-1]["hybrid_death_price_support_successes"] == 12


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
    (run_root / "status" / "trainer.json").write_text(json.dumps({"state": "running"}), encoding="utf-8")
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
        collector_device="cuda",
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
    script = (SCRIPT_ROOT / "train_hybrid_parallel_wsl_rocm.sh").read_text(encoding="utf-8")
    assert 'COLLECTORS="${STS2_HYBRID_COLLECTORS:-2}"' in script
    assert '--updates-per-run "${STS2_HYBRID_UPDATES_PER_RUN:-1}"' in script
    assert '--ingest-batch-runs "${STS2_HYBRID_INGEST_BATCH_RUNS:-2}"' in script
    assert '--target-update-interval "${STS2_HYBRID_TARGET_UPDATE_INTERVAL:-25}"' in script
    assert "target-sync-every-runs" not in script
    assert "--migrate-target-schedule" not in script
    assert 'INITIALIZE_FROM="${STS2_HYBRID_INITIALIZE_FROM:-}"' in script
    assert 'MODE_ARGS+=(--initialize-from "$INITIALIZE_FROM")' in script
    assert 'MODE_ARGS+=(--resume "$RESUME")' in script
    assert 'COLLECTOR_DEVICE="${STS2_HYBRID_COLLECTOR_DEVICE:-cuda}"' in script
    assert 'COLLECTOR_START_EPISODE="${STS2_HYBRID_COLLECTOR_START_EPISODE:-0}"' in script
    assert '--producer-id "collector-$index"' in script
    assert '--start-episode-index "$COLLECTOR_START_EPISODE"' in script
    assert 'COLLECTOR_PIDS+=("$!")' in script
    assert "exited while learner was running" in script
    assert 'rm -f "$PUBLICATION"' in script
    assert 'PUBLICATION_WAIT_SECONDS="${STS2_HYBRID_PUBLICATION_WAIT_SECONDS:-600}"' in script


def test_parallel_launcher_refreshes_heldout_gate_every_control_block() -> None:
    script = (SCRIPT_ROOT / "train_hybrid_parallel_wsl_rocm.sh").read_text(encoding="utf-8")
    assert "STS2_HYBRID_PERIODIC_EVALUATION_RUN_INTERVAL:-50" in script
    assert "STS2_HYBRID_PERIODIC_EVALUATION_EPISODES:-16" in script
    assert 'PERIODIC_EVALUATION="$RUN_ROOT/evaluation/periodic-run-$current_ingested"' in script
    assert 'cp -f "$PUBLICATION" "$PERIODIC_FROZEN_POLICY"' in script
    assert "PERIODIC_EVALUATION_PID=$!" in script
    assert '[[ -z "$PERIODIC_EVALUATION_PID" ]]' in script
    assert "lambda remains frozen until fresh evidence arrives" in script


def test_v8_q_calibration_preserves_early_update_debt_and_disables_dropout() -> None:
    source = (SCRIPT_ROOT / "run_hybrid_trainer.py").read_text(encoding="utf-8")
    assert "runs = replay.runs" in source
    assert "online.eval()" in source
    assert "target.eval()" in source
    assert "online.train(online_was_training)" in source
    assert "target.train(target_was_training)" in source
    assert "pending_learner_updates += args.updates_per_run * len(accepted)" in source
    assert "while pending_learner_updates > 0:" in source
    assert "pending_learner_updates -= 1" in source
    assert '"pending_learner_updates": pending_learner_updates' in source


def test_parallel_dashboard_surfaces_collector_failure(tmp_path: Path) -> None:
    bridge = _script("run_hybrid_dashboard_bridge.py")
    run_root = tmp_path / "failed-collector"
    (run_root / "launcher").mkdir(parents=True)
    (run_root / "status").mkdir()
    (run_root / "launcher" / "launch.json").write_text("{}", encoding="utf-8")
    (run_root / "status" / "trainer.json").write_text(json.dumps({"state": "running"}), encoding="utf-8")
    (run_root / "status" / "collector-0.json").write_text(
        json.dumps(
            {
                "state": "failed",
                "producer_id": "collector-0",
                "error": "publication temporarily unavailable",
                "unix_s": 30.0,
            }
        ),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        run_root=str(run_root),
        lineage="hybrid-parallel",
        run_id="run-c",
        target_runs=128,
        collector_episode_capacity=200,
        collector_count=2,
        collector_device="cuda",
        maximum_pending_runs=2,
        revival_budget=64,
        curriculum_mode="hybrid-completion-preheat",
    )
    events = bridge._projection(args)
    assert events[-1]["event"] == "run_failed"
    assert "collector-0" in events[-1]["error"]


def test_collector_reload_keeps_loaded_policy_during_publication_swap() -> None:
    source = (SCRIPT_ROOT / "run_hybrid_collector.py").read_text(encoding="utf-8")
    assert "except FileNotFoundError:" in source
    assert "if initial or model_mtime == 0:" in source
    assert "return False" in source


def test_hybrid_collector_restarts_only_the_poisoned_backend_session() -> None:
    source = (SCRIPT_ROOT / "run_hybrid_collector.py").read_text(encoding="utf-8")
    assert "except _RECOVERABLE_BACKEND_ERRORS as error:" in source
    assert "authority.discard_run()" in source
    assert '"event": "hybrid_collector_backend_restart"' in source
    assert "replacement = build_backend(config)" in source
    assert "resources.collector.replace_backend(replacement)" in source
    assert "while True:" in source


def test_hybrid_collector_continuation_advances_its_seed_stream() -> None:
    source = (SCRIPT_ROOT / "run_hybrid_collector.py").read_text(encoding="utf-8")
    assert "collector_seed = args.seed + args.start_episode_index" in source
    assert "replace(config.runtime, seed=collector_seed)" in source
    assert "seed=collector_seed" in source


def test_trainer_publication_uses_unique_atomic_replacement() -> None:
    source = (SCRIPT_ROOT / "run_hybrid_trainer.py").read_text(encoding="utf-8")
    assert "uuid.uuid4().hex" in source
    assert "os.replace(publication_temporary, publication)" in source


def test_exact_resume_republishes_without_rewriting_the_replay_checkpoint() -> None:
    source = (SCRIPT_ROOT / "run_hybrid_trainer.py").read_text(encoding="utf-8")
    assert 'torch.load(args.resume, map_location="cpu", weights_only=False)' in source
    assert "del payload\n            gc.collect()" in source
    startup = source.index("        if args.resume:\n            # The checkpoint")
    loop = source.index("        spool = Path(args.spool)", startup)
    block = source[startup:loop]
    assert "publish()" in block
    assert "else:\n            save()" in block
    assert "schedule_migration" not in source


def test_trainer_discards_uncheckpointed_metric_tail_before_resume() -> None:
    source = (SCRIPT_ROOT / "run_hybrid_trainer.py").read_text(encoding="utf-8")
    assert "def _discard_metrics_beyond_checkpoint" in source
    assert "row_ingested > ingested" in source
    assert "_discard_metrics_beyond_checkpoint(metrics_path, ingested=ingested)" in source


def test_hybrid_evaluation_is_greedy_and_does_not_write_replay() -> None:
    source = (SCRIPT_ROOT / "run_hybrid_evaluation.py").read_text(encoding="utf-8")
    assert "evaluation_ownership=True" in source
    assert "deterministic=True" in source
    assert "held_out_evaluation_seeds" in source
    assert "TrajectoryJournal" in source
    assert "trajectory_journal=journal" in source
    assert "journal_episode_id_prefix=episode_namespace" in source
    assert '"act3_reached": bool(episode.metrics.max_act >= 3)' in source
    assert "summarize_evaluation" in source
    assert '"simulator": simulator_provenance' in source
    assert "--spool" not in source
    assert '"event": "hybrid_evaluation_started"' in source


def test_hybrid_evaluation_retains_other_policies_and_replaces_same_policy() -> None:
    source = (SCRIPT_ROOT / "run_hybrid_evaluation.py").read_text(encoding="utf-8")
    assert "def _retained_evaluation_rows(" in source
    assert 'row.get("policy_version") == replacing_policy_version' in source
    assert "for retained in retained_evaluations:" in source
    assert 'metrics.write(json.dumps(retained) + "\\n")' in source


def test_parallel_launcher_runs_final_heldout_after_clean_training() -> None:
    source = (SCRIPT_ROOT / "train_hybrid_parallel_wsl_rocm.sh").read_text(encoding="utf-8")
    trainer_exit = source.index('[[ "$status" -eq 0 ]] || exit "$status"')
    evaluation = source.rindex("python scripts/run_hybrid_evaluation.py")
    assert trainer_exit < evaluation
    assert "STS2_HYBRID_FINAL_EVALUATION_EPISODES:-16" in source
    assert 'cp -f "$PUBLICATION" "$FROZEN_POLICY"' in source
    assert '--metrics-out "$METRICS/evaluation.jsonl"' in source
    assert '--journal-directory "$DASHBOARD_RUN_ROOT"' in source


def test_parallel_launcher_establishes_heldout_floor_before_collection() -> None:
    source = (SCRIPT_ROOT / "train_hybrid_parallel_wsl_rocm.sh").read_text(encoding="utf-8")
    startup_evaluation = source.index("python scripts/run_hybrid_evaluation.py")
    first_collector = source.index("python scripts/run_hybrid_collector.py")
    assert startup_evaluation < first_collector
    assert "STS2_HYBRID_STARTUP_EVALUATION_EPISODES:-16" in source
    assert '--metrics-out "$METRICS/evaluation.jsonl"' in source


def test_parallel_dashboard_preserves_standard_hybrid_evaluation_fields(
    tmp_path: Path,
) -> None:
    bridge = _script("run_hybrid_dashboard_bridge.py")
    run_root = tmp_path / "evaluated"
    (run_root / "launcher").mkdir(parents=True)
    (run_root / "status").mkdir()
    (run_root / "launcher" / "launch.json").write_text("{}", encoding="utf-8")
    (run_root / "status" / "trainer.json").write_text(json.dumps({"state": "complete"}), encoding="utf-8")
    _append(
        run_root / "metrics" / "trainer.jsonl",
        {
            "event": "hybrid_update",
            "ingested_runs": 200,
            "environment_steps": 139_035,
            "learner": {"updates": 240},
            "replay": {},
        },
    )
    _append(
        run_root / "metrics" / "evaluation.jsonl",
        {
            "event": "hybrid_evaluation_summary",
            "unix_s": 20.0,
            "policy_version": 240,
            "episodes": 16,
            "successful_run_count": 5,
            "run_win_rate": 5 / 16,
            "act1_clear_count": 14,
            "act1_clear_rate": 14 / 16,
            "act3_reach_count": 11,
            "act3_reach_rate": 11 / 16,
            "mean_max_floor": 39.4375,
            "maximum_floor": 46,
            "mean_revivals_used": 51.3125,
            "mean_act1_revivals_on_clear": 7.25,
            "mean_revivals_on_success": 38.5,
            "mean_player_hp_lost": 900.0,
            "deadlock_rate": 0.0,
        },
    )
    args = SimpleNamespace(
        run_root=str(run_root),
        lineage="hybrid-parallel",
        run_id="run-eval",
        target_runs=200,
        collector_episode_capacity=200,
        collector_count=2,
        collector_device="cuda",
        maximum_pending_runs=2,
        revival_budget=64,
        curriculum_mode="hybrid-completion-preheat",
    )
    evaluation = next(row for row in bridge._projection(args) if row["event"] == "evaluation")
    assert evaluation["act1_clear_count"] == 14
    assert evaluation["act3_reach_count"] == 11
    assert evaluation["act3_reach_rate"] == 11 / 16
    assert evaluation["successful_run_count"] == 5
    assert evaluation["mean_player_hp_lost"] == 900.0
    assert evaluation["mean_act1_revivals_on_clear"] == 7.25
    assert evaluation["mean_revivals_on_success"] == 38.5


def test_dashboard_waits_for_current_policy_evaluation_not_old_summary(
    tmp_path: Path,
) -> None:
    bridge = _script("run_hybrid_dashboard_bridge.py")
    run_root = tmp_path / "evaluation-race"
    (run_root / "launcher").mkdir(parents=True)
    (run_root / "status").mkdir()
    (run_root / "launcher" / "launch.json").write_text("{}", encoding="utf-8")
    (run_root / "status" / "trainer.json").write_text(json.dumps({"state": "complete"}), encoding="utf-8")
    _append(
        run_root / "metrics" / "trainer.jsonl",
        {
            "event": "hybrid_update",
            "ingested_runs": 400,
            "environment_steps": 277_507,
            "learner": {"updates": 440},
            "replay": {},
        },
    )
    _append(
        run_root / "metrics" / "evaluation.jsonl",
        {
            "event": "hybrid_evaluation_summary",
            "policy_version": 240,
            "episodes": 16,
            "run_win_rate": 5 / 16,
        },
    )
    args = SimpleNamespace(
        run_root=str(run_root),
        lineage="hybrid-parallel",
        run_id="run-evaluation-race",
        target_runs=400,
        collector_episode_capacity=220,
        collector_count=2,
        collector_device="cuda",
        maximum_pending_runs=2,
        revival_budget=64,
        final_evaluation_episodes=16,
        curriculum_mode="hybrid-completion-preheat",
    )

    events = bridge._projection(args)

    assert not any(row["event"] == "run_complete" for row in events)
    progress = next(row for row in events if row["event"] == "evaluation_started")
    assert progress["policy_version"] == 440
    assert progress["completed_episodes"] == 0
    assert progress["evaluation_episodes"] == 16
    historical = [row for row in events if row["event"] == "evaluation"]
    assert [row["policy_version"] for row in historical] == [240]


def test_dashboard_projects_evaluation_history_at_each_policy_step(
    tmp_path: Path,
) -> None:
    bridge = _script("run_hybrid_dashboard_bridge.py")
    run_root = tmp_path / "evaluation-history"
    (run_root / "launcher").mkdir(parents=True)
    (run_root / "status").mkdir()
    (run_root / "launcher" / "launch.json").write_text("{}", encoding="utf-8")
    (run_root / "status" / "trainer.json").write_text(json.dumps({"state": "complete"}), encoding="utf-8")
    for policy, environment_steps in ((240, 139_035), (440, 277_507)):
        _append(
            run_root / "metrics" / "trainer.jsonl",
            {
                "event": "hybrid_update",
                "ingested_runs": policy - 40,
                "environment_steps": environment_steps,
                "learner": {"updates": policy},
                "replay": {},
            },
        )
        _append(
            run_root / "metrics" / "evaluation.jsonl",
            {
                "event": "hybrid_evaluation_summary",
                "policy_version": policy,
                "episodes": 16,
                "wins": 5 if policy == 240 else 9,
                "act1_clears": 14 if policy == 240 else 15,
                "act3_reaches": 11,
            },
        )
    args = SimpleNamespace(
        run_root=str(run_root),
        lineage="hybrid-parallel",
        run_id="run-evaluation-history",
        target_runs=400,
        collector_episode_capacity=220,
        collector_count=2,
        collector_device="cuda",
        maximum_pending_runs=2,
        revival_budget=64,
        final_evaluation_episodes=16,
        curriculum_mode="hybrid-completion-preheat",
    )

    events = bridge._projection(args)

    evaluations = [row for row in events if row["event"] == "evaluation"]
    assert [row["policy_version"] for row in evaluations] == [240, 440]
    assert [row["environment_steps"] for row in evaluations] == [139_035, 277_507]


def test_hybrid_checkpoint_uses_streaming_serialization() -> None:
    source = (SCRIPT_ROOT / "run_hybrid_trainer.py").read_text(encoding="utf-8")
    assert "_use_new_zipfile_serialization=False" in source


def test_windows_supervisor_marks_native_wsl_exit() -> None:
    source = (SCRIPT_ROOT / "run_hybrid_parallel_windows.ps1").read_text(encoding="utf-8")
    assert 'state = "interrupted"' in source
    assert 'error = "native_wsl_exit:$exitCode"' in source


def test_dashboard_replaces_abandoned_tail_after_in_place_resume() -> None:
    bridge = _script("run_hybrid_dashboard_bridge.py")
    rows = [{"event": "hybrid_update", "ingested_runs": value} for value in (44, 45, 46, 47, 46, 47, 48)]
    canonical = bridge._canonical_trainer_rows(rows)
    assert [row["ingested_runs"] for row in canonical] == [44, 45, 46, 47, 48]


def test_dashboard_reads_hybrid_evaluation_summary() -> None:
    source = (SCRIPT_ROOT / "run_hybrid_dashboard_bridge.py").read_text(encoding="utf-8")
    assert 'run_root / "metrics" / "evaluation.jsonl"' in source
    assert '"event": "evaluation"' in source
    assert '"hybrid_target_syncs": item.get("target_syncs")' in source
    assert '"hybrid_last_target_sync_update": item.get(' in source
    assert 'row.get("event") == "hybrid_evaluation_started"' in source
    assert 'state == "complete" and not evaluation_in_progress' in source


def test_monitor_projects_hybrid_target_sync_clock() -> None:
    from sts2_rl.monitoring import _project_learner

    point, latest = _project_learner(
        {
            "event": "learner_update",
            "policy_version": 164,
            "hybrid_target_syncs": 2,
            "hybrid_last_target_sync_update": 164,
        }
    )
    for projected in (point, latest):
        assert projected["hybrid_target_syncs"] == 2
        assert projected["hybrid_last_target_sync_update"] == 164


def test_dashboard_tail_replacement_preserves_non_update_rows() -> None:
    bridge = _script("run_hybrid_dashboard_bridge.py")
    rows = [
        {"event": "run_start"},
        {"event": "hybrid_update", "ingested_runs": 45},
        {"event": "hybrid_update", "ingested_runs": 46},
        {"event": "hybrid_update", "ingested_runs": 46},
    ]
    canonical = bridge._canonical_trainer_rows(rows)
    assert canonical == [
        {"event": "run_start"},
        {"event": "hybrid_update", "ingested_runs": 45},
        {"event": "hybrid_update", "ingested_runs": 46},
    ]
