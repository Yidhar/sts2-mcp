#!/usr/bin/env python3
"""Stage-2 macro spool trainer with explicit fresh/model-init/resume semantics.

The online weight file is a lightweight publication for collectors.  The
training-state file is the resumable artifact: it contains both networks,
optimizer, replay, learner counters, and random state.  Exact continuation
restores that learner state into a new segment whose producer processes and
spool are intentionally fresh.  A missing requested initialization or resume
path is an error; it never becomes a fresh run.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pickle
import random
import time
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from sts2_rl.encoding.snapshot import collate_encoded_snapshots
from sts2_rl.macro import (
    JoinedCollectionAuthority,
    MacroCollectionAuthority,
    MacroEpisode,
    MacroQConfig,
    MacroQLearner,
    MacroSequenceReplay,
    MacroStep,
    load_trunk_state,
)
from sts2_rl.training import build_training_resources, load_training_config
from sts2_rl.training.seeding import held_out_evaluation_seeds

TRAINING_STATE_FORMAT = "sts2-stage2-pipeline-state-v1"
SPOOL_ENVELOPE_FORMAT = "sts2-stage2-spool-envelope-v1"


@contextmanager
def _evaluation_mode(model: Any) -> Any:
    was_training = bool(model.training)
    model.eval()
    try:
        yield
    finally:
        model.train(was_training)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(payload), sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _forward_factory(
    model: Any,
    encoder: Any,
    device: torch.device,
    *,
    detach_hidden: bool,
) -> Any:
    def forward(step: Any, hidden: Any) -> tuple[torch.Tensor, Any]:
        snapshot = step.snapshot if isinstance(step, MacroStep) else step
        encoded = collate_encoded_snapshots(
            (snapshot,),
            expected_config=encoder.config,
            expected_fingerprint=snapshot.encoding_fingerprint,
            device=device,
        )
        if hidden is None:
            hidden = model.initial_state(1, device=device)
        output = model(encoded, hidden, validate=False)
        q_values = output.transaction_q_values
        if q_values is None:
            raise RuntimeError("macro Q model produced no candidate Q values")
        recurrent = output.recurrent_state
        return q_values[0], recurrent.detach() if detach_hidden else recurrent

    return forward


def _batched_forward_factory(model: Any, encoder: Any, device: torch.device, *, detach_hidden: bool) -> Any:
    def forward(snapshots: Any, hidden: Any) -> tuple[torch.Tensor, Any]:
        encoded = collate_encoded_snapshots(
            tuple(snapshots),
            expected_config=encoder.config,
            expected_fingerprint=snapshots[0].encoding_fingerprint,
            device=device,
        )
        if hidden is None:
            hidden = model.initial_state(len(snapshots), device=device)
        output = model(encoded, hidden, validate=False)
        q_values = output.transaction_q_values
        if q_values is None:
            raise RuntimeError("macro Q model produced no candidate Q values")
        recurrent = output.recurrent_state
        return q_values, recurrent.detach() if detach_hidden else recurrent

    return forward


def _replay_state(replay: MacroSequenceReplay) -> dict[str, Any]:
    return replay.state_dict()


def _restore_replay(replay: MacroSequenceReplay, state: Mapping[str, Any]) -> None:
    replay.load_state_dict(dict(state))


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "accelerator": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    accelerator = state.get("accelerator")
    if accelerator is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in accelerator])


def _save_training_state(
    path: Path,
    *,
    online: Any,
    target: Any,
    learner: MacroQLearner,
    replay: MacroSequenceReplay,
    lineage_id: str,
    control_domain: str,
    ingested: int,
    environment_steps: int,
) -> None:
    payload = {
        "format": TRAINING_STATE_FORMAT,
        "lineage_id": lineage_id,
        "control_domain": control_domain,
        "online": online.state_dict(),
        "target": target.state_dict(),
        "learner": learner.state_dict(),
        "replay": _replay_state(replay),
        "ingested": ingested,
        "environment_steps": environment_steps,
        "policy_version": learner.metrics.updates,
        "rng": _rng_state(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _load_training_state(
    path: Path,
    *,
    online: Any,
    target: Any,
    learner: MacroQLearner,
    replay: MacroSequenceReplay,
    lineage_id: str,
    control_domain: str,
    device: torch.device,
) -> tuple[int, int]:
    if not path.is_file():
        raise FileNotFoundError(f"resume training state does not exist: {path}")
    payload = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(payload, Mapping) or payload.get("format") != TRAINING_STATE_FORMAT:
        raise ValueError(f"not a {TRAINING_STATE_FORMAT} training state: {path}")
    if payload.get("lineage_id") != lineage_id:
        raise ValueError("resume lineage differs from requested lineage")
    if payload.get("control_domain") != control_domain:
        raise ValueError("resume control domain differs from requested control domain")
    online.load_state_dict(payload["online"], strict=True)
    target.load_state_dict(payload["target"], strict=True)
    learner.load_state_dict(payload["learner"])
    _restore_replay(replay, payload["replay"])
    _restore_rng(payload["rng"])
    return int(payload["ingested"]), int(payload["environment_steps"])


def _producer_states(status_dir: Path, producer_ids: tuple[str, ...]) -> dict[str, str]:
    states: dict[str, str] = {}
    for producer_id in producer_ids:
        path = status_dir / f"{producer_id}.json"
        if not path.is_file():
            states[producer_id] = "pending"
            continue
        row = json.loads(path.read_text(encoding="utf-8"))
        states[producer_id] = str(row.get("state") or "unknown")
    return states


def _require_producers_not_failed(
    status_dir: Path,
    producer_ids: tuple[str, ...],
) -> dict[str, str]:
    states = _producer_states(status_dir, producer_ids)
    failed = [producer_id for producer_id, state in states.items() if state == "failed"]
    if failed:
        raise RuntimeError(f"producer failed: {', '.join(failed)}")
    return states


def _save_then_acknowledge(
    save_all: Any,
    publish_acknowledgement: Any,
    spool_paths: list[Path],
) -> None:
    """Make the accepted learner state durable before deleting source items."""

    save_all()
    publish_acknowledgement()
    for path in spool_paths:
        path.unlink()


def _checkpoint_due(
    *,
    ingested: int,
    last_saved_at: int,
    interval: int,
    target: int,
) -> bool:
    return ingested >= target or ingested - last_saved_at >= interval


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--champion", required=True)
    parser.add_argument("--spool", required=True)
    parser.add_argument("--save-macro", required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--init-macro", default=None, help="model-only initialization")
    mode.add_argument("--resume", default=None, help="full training-state continuation")
    parser.add_argument("--checkpoint-out", required=True)
    parser.add_argument("--pipeline-id", required=True)
    parser.add_argument("--lineage-id", default="stage2-isolated-macro-v1")
    parser.add_argument(
        "--control-domain",
        choices=("macro", "combat"),
        default="macro",
        help="control domain this trainer optimizes; combat requires --bridge-partner",
    )
    parser.add_argument(
        "--bridge-partner",
        default=None,
        help="frozen macro publication (.pt) pinned as the cross-domain bootstrap "
        "partner; required for --control-domain combat, forbidden otherwise",
    )
    parser.add_argument("--producer-status-dir", required=True)
    parser.add_argument("--producer-id", action="append", dest="producer_ids", required=True)
    parser.add_argument("--idle-timeout-seconds", type=float, default=1200.0)
    parser.add_argument("--status-out", required=True)
    parser.add_argument("--stop-after-episodes", type=int, default=1200)
    parser.add_argument("--updates-per-episode", type=int, default=8)
    parser.add_argument(
        "--ingest-batch-episodes",
        type=int,
        default=4,
        help="episodes accepted per ingest cycle; keeps update bursts, "
        "publications, and metrics latency bounded",
    )
    parser.add_argument(
        "--max-updates-per-cycle",
        type=int,
        default=32,
        help="hard cap on learner updates per ingest cycle",
    )
    parser.add_argument("--sample-windows", type=int, default=16)
    parser.add_argument("--replay-episodes", type=int, default=128)
    parser.add_argument("--save-interval-episodes", type=int, default=20)
    parser.add_argument(
        "--eval-interval-episodes",
        type=int,
        default=100,
        help="held-out evaluation gate cadence in ingested episodes (0 disables)",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=8,
        help="held-out seeds per evaluation gate (fixed odd prefix)",
    )
    parser.add_argument("--metrics-out", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--sim-exe", default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    init_path = Path(args.init_macro) if args.init_macro is not None else None
    resume_path = Path(args.resume) if args.resume is not None else None
    metrics_path = Path(args.metrics_out)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    status_path = Path(args.status_out)
    _atomic_json(
        status_path,
        {"pipeline_id": args.pipeline_id, "state": "running", "unix_s": time.time()},
    )
    resources = None
    ingested = 0
    environment_steps = 0
    durable_ingested = 0
    durable_environment_steps = 0
    learner: MacroQLearner | None = None
    bridge_partner_path = Path(args.bridge_partner) if args.bridge_partner is not None else None
    try:
        if init_path is not None and not init_path.is_file():
            raise FileNotFoundError(f"model initialization does not exist: {init_path}")
        if resume_path is not None and not resume_path.is_file():
            raise FileNotFoundError(f"resume training state does not exist: {resume_path}")
        if args.control_domain == "combat" and bridge_partner_path is None:
            raise ValueError(
                "--control-domain combat requires --bridge-partner: the combat "
                "learner bootstraps encounter boundaries from a pinned macro publication"
            )
        if args.control_domain != "combat" and bridge_partner_path is not None:
            raise ValueError("--bridge-partner applies only to --control-domain combat")
        if bridge_partner_path is not None and not bridge_partner_path.is_file():
            raise FileNotFoundError(f"bridge partner publication does not exist: {bridge_partner_path}")
        config = load_training_config(profile="preheat", config_path=Path(args.config))
        if args.device is not None:
            config = replace(config, runtime=replace(config.runtime, device=str(args.device)))
        if args.sim_exe is not None:
            config = replace(
                config,
                environment=replace(
                    config.environment,
                    backend="headless",
                    sim_exe_path=str(args.sim_exe),
                ),
            )
        if not config.transaction_learning.enabled:
            raise RuntimeError("trainer requires transaction_learning.enabled=true")
        if args.save_interval_episodes <= 0:
            raise ValueError("save interval must be a positive episode count")

        spool = Path(args.spool)
        spool.mkdir(parents=True, exist_ok=True)
        if any(spool.iterdir()):
            raise RuntimeError(
                "a trainer segment must start with a new empty spool; "
                "resume restores learner state, not old producer files"
            )
        resources = build_training_resources(config)
        champion_state = torch.load(
            Path(args.champion) / "network.pt",
            map_location=resources.device,
            weights_only=True,
        )
        load_trunk_state(resources.model, dict(champion_state))
        for parameter in resources.model.parameters():
            parameter.requires_grad_(False)

        macro_online = copy.deepcopy(resources.model)
        if init_path is not None:
            load_trunk_state(
                macro_online,
                dict(torch.load(init_path, map_location=resources.device, weights_only=True)),
            )
        for parameter in macro_online.parameters():
            parameter.requires_grad_(True)
        # Q is the baseline. Keep stochastic regularizers off in both streams.
        macro_online.eval()
        macro_target = copy.deepcopy(macro_online)
        macro_target.eval()
        for parameter in macro_target.parameters():
            parameter.requires_grad_(False)

        device = resources.device

        # Cross-domain bootstrap bridge: bridge values are captured at
        # COLLECTION time by the pinned macro partner's live recurrent state
        # (they arrive on the spooled transitions as MacroStep.bridge_value),
        # so the trainer's learn loop never evaluates the partner model.  The
        # trainer still pins the partner's identity for provenance and only
        # loads its weights — lazily, exactly once per segment — when a
        # held-out evaluation gate composes the macro arm (bridge doc §2.4).
        bridge_partner_record: dict[str, str] | None = None
        partner_model: Any = None
        if bridge_partner_path is not None:
            bridge_partner_record = {
                "path": str(bridge_partner_path),
                "sha256": hashlib.sha256(bridge_partner_path.read_bytes()).hexdigest(),
            }

        def _bridge_partner_model() -> Any:
            nonlocal partner_model
            if partner_model is None:
                assert bridge_partner_path is not None
                assert resources is not None
                loaded = copy.deepcopy(resources.model)
                load_trunk_state(
                    loaded,
                    dict(
                        torch.load(
                            bridge_partner_path, map_location=device, weights_only=True
                        )
                    ),
                )
                loaded.eval()
                for parameter in loaded.parameters():
                    parameter.requires_grad_(False)
                partner_model = loaded
            return partner_model

        replay = MacroSequenceReplay(
            capacity_episodes=args.replay_episodes,
            window_length=16,
            control_domain=args.control_domain,
        )
        learner = MacroQLearner(
            online_parameters=list(macro_online.parameters()),
            forward_online=_forward_factory(macro_online, resources.encoder, device, detach_hidden=False),
            forward_target=_forward_factory(macro_target, resources.encoder, device, detach_hidden=True),
            sync_target=lambda: macro_target.load_state_dict(macro_online.state_dict()),
            initial_state=lambda: None,
            replay=replay,
            config=MacroQConfig(sample_windows=args.sample_windows),
            forward_online_batch=_batched_forward_factory(macro_online, resources.encoder, device, detach_hidden=False),
            forward_target_batch=_batched_forward_factory(macro_target, resources.encoder, device, detach_hidden=True),
            target_evaluation_context=lambda: _evaluation_mode(macro_target),
        )

        if resume_path is not None:
            ingested, environment_steps = _load_training_state(
                resume_path,
                online=macro_online,
                target=macro_target,
                learner=learner,
                replay=replay,
                lineage_id=args.lineage_id,
                control_domain=args.control_domain,
                device=device,
            )
            if args.stop_after_episodes <= ingested:
                raise ValueError("resume target must exceed the restored ingested episode count")

        def save_publication() -> None:
            path = Path(args.save_macro)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".tmp")
            torch.save(macro_online.state_dict(), temporary)
            temporary.replace(path)

        def save_all() -> None:
            _save_training_state(
                Path(args.checkpoint_out),
                online=macro_online,
                target=macro_target,
                learner=learner,
                replay=replay,
                lineage_id=args.lineage_id,
                control_domain=args.control_domain,
                ingested=ingested,
                environment_steps=environment_steps,
            )
            # Collectors only see a policy version after the resumable learner
            # state containing that version is durable.
            save_publication()

        # Publish the restored/initialized behavior before producers start.
        save_all()
        durable_ingested = ingested
        durable_environment_steps = environment_steps
        last_saved_at = ingested
        last_eval_at = ingested
        last_progress = time.monotonic()
        producer_ids = tuple(args.producer_ids)
        pending_paths: list[Path] = []
        pending_path_set: set[Path] = set()
        pending_summaries: list[dict[str, Any]] = []
        last_update_metrics: dict[str, Any] = learner.metrics.as_mapping()
        # Native monitoring-panel stream, exactly like the legacy trainer
        # lineages: runs/<lineage>/run-<uuid>/metrics.jsonl starting with a
        # run_start whose run_id matches the run directory.
        dashboard_path = status_path.parent / "metrics.jsonl"
        dashboard_file = dashboard_path.open("a", encoding="utf-8")

        def dashboard_write(payload: Mapping[str, Any]) -> None:
            dashboard_file.write(json.dumps(dict(payload)) + "\n")
            dashboard_file.flush()

        # The panel derives its progress bar from the config's step target;
        # the pipeline budgets in episodes, so the step target is the episode
        # budget times the observed (or a nominal prior) steps-per-episode.
        steps_per_episode = (
            environment_steps / ingested if ingested > 0 else 520.0
        )
        environment_step_target = int(args.stop_after_episodes * steps_per_episode)
        dashboard_config = config.to_mapping()
        dashboard_runtime = dict(dashboard_config.get("runtime") or {})
        dashboard_runtime["total_environment_steps"] = environment_step_target
        dashboard_config["runtime"] = dashboard_runtime
        dashboard_write(
            {
                "event": "run_start",
                "run_id": args.pipeline_id,
                "unix_s": time.time(),
                "lineage": args.lineage_id,
                "control_domain": args.control_domain,
                "state": {
                    "environment_steps": environment_steps,
                    "episodes": ingested,
                },
                "episodes_target": args.stop_after_episodes,
                "environment_step_target_estimate": environment_step_target,
                "config": dashboard_config,
            }
        )

        def run_evaluation_gate(gate_environment_steps: int) -> None:
            """Held-out fixed-seed gate on the trainer's own idle backend.

            Reuses the collection authority stack: the online publication
            drives this trainer's domain greedily; combat gates compose the
            pinned bridge partner over macro surfaces through the router.
            Emits the panel's native evaluation record pair — the summary
            event in metrics.jsonl and the per-episode journal file.
            """

            assert resources is not None
            assert learner is not None
            seeds = held_out_evaluation_seeds(config.runtime.seed, args.eval_episodes)
            online_forward = _forward_factory(
                macro_online, resources.encoder, device, detach_hidden=True
            )
            if args.control_domain == "combat":
                partner_forward = _forward_factory(
                    _bridge_partner_model(), resources.encoder, device, detach_hidden=True
                )
                macro_arm = MacroCollectionAuthority(
                    forward_q=partner_forward,
                    initial_state=lambda: None,
                    epsilon=0.0,
                    control_domain="macro",
                    evaluation_ownership=True,
                )
                combat_arm = MacroCollectionAuthority(
                    forward_q=online_forward,
                    initial_state=lambda: None,
                    epsilon=0.0,
                    control_domain="combat",
                    evaluation_ownership=True,
                )
                authority: Any = JoinedCollectionAuthority(
                    macro=macro_arm, combat=combat_arm
                )
            else:
                authority = MacroCollectionAuthority(
                    forward_q=online_forward,
                    initial_state=lambda: None,
                    epsilon=0.0,
                    control_domain="macro",
                    evaluation_ownership=True,
                )
            journal_path = (
                status_path.parent
                / f"evaluation-step-{gate_environment_steps:09d}.jsonl"
            )
            rows: list[dict[str, Any]] = []
            resources.collector.macro_authority = authority
            try:
                with _evaluation_mode(macro_online), journal_path.open(
                    "a", encoding="utf-8"
                ) as journal:
                    for seed in seeds:
                        episode = resources.collector.collect_episode(
                            epsilon=0.0,
                            deterministic=True,
                            record=False,
                            evaluation_seed=seed,
                        )
                        row = {
                            "event": "evaluation_episode",
                            "evaluation_seed": seed,
                            "unix_s": time.time(),
                            "steps": episode.metrics.steps,
                            "run_won": episode.metrics.run_won,
                            "act1_cleared": episode.metrics.act1_cleared,
                            "max_act": episode.metrics.max_act,
                            "max_floor": episode.metrics.max_floor,
                            "revivals_used": episode.metrics.revivals_used,
                            "reward_total": episode.metrics.reward_total,
                            "terminal_reason": episode.metrics.terminal_reason,
                        }
                        rows.append(row)
                        journal.write(json.dumps(row) + "\n")
                        journal.flush()
            finally:
                resources.collector.macro_authority = None
            floors = sorted(row["max_floor"] for row in rows)
            summary = {
                "episodes": len(rows),
                "wins": sum(1 for row in rows if row["run_won"]),
                "act1_clears": sum(1 for row in rows if row["act1_cleared"]),
                "floor_p50": floors[len(floors) // 2] if floors else None,
                "floor_mean": (sum(floors) / len(floors)) if floors else None,
            }
            dashboard_write(
                {
                    "event": "evaluation",
                    "unix_s": time.time(),
                    "environment_steps": gate_environment_steps,
                    "evaluation_gate": gate_environment_steps,
                    "evaluation_attempt": 1,
                    "gate_kind": "validation",
                    "data_partition": "held_out",
                    "policy_version": learner.metrics.updates,
                    "seeds": list(seeds),
                    **summary,
                }
            )
            metrics_file.write(
                json.dumps(
                    {
                        "event": "trainer_evaluation",
                        "unix_s": time.time(),
                        "pipeline_id": args.pipeline_id,
                        "ingested_total": ingested,
                        "environment_steps": gate_environment_steps,
                        **summary,
                    }
                )
                + "\n"
            )
            metrics_file.flush()
        with metrics_path.open("a", encoding="utf-8") as metrics_file:
            metrics_file.write(
                json.dumps(
                    {
                        "event": "trainer_start",
                        "unix_s": time.time(),
                        "pipeline_id": args.pipeline_id,
                        "lineage_id": args.lineage_id,
                        "load_mode": "resume"
                        if resume_path is not None
                        else ("model_init" if init_path is not None else "fresh"),
                        "control_domain": args.control_domain,
                        "bridge_partner": bridge_partner_record,
                        "ingested_total": ingested,
                        "environment_steps": environment_steps,
                        "policy_version": learner.metrics.updates,
                        "learner": learner.metrics.as_mapping(),
                    }
                )
                + "\n"
            )
            metrics_file.flush()

            def flush_pending() -> None:
                """Checkpoint one buffered ingest group, then acknowledge it."""

                nonlocal last_saved_at, durable_ingested, durable_environment_steps
                if not pending_paths:
                    return
                ingest_row = {
                    "event": "trainer_ingest",
                    "unix_s": time.time(),
                    "pipeline_id": args.pipeline_id,
                    "bridge_partner": bridge_partner_record,
                    "ingested_total": ingested,
                    "fresh": len(pending_summaries),
                    "environment_steps": environment_steps,
                    "policy_version": learner.metrics.updates,
                    "acknowledged_episodes": list(pending_summaries),
                    "replay": replay.metrics(),
                    "learner": last_update_metrics,
                }

                def publish_acknowledgement() -> None:
                    metrics_file.write(json.dumps(ingest_row) + "\n")
                    metrics_file.flush()
                    for summary in pending_summaries:
                        dashboard_write(
                            {
                                "event": "train_episode",
                                "unix_s": summary.get("unix_s"),
                                "environment_steps": summary.get("environment_steps"),
                                "episode_id": summary.get("episode_id"),
                                "reset_seed": summary.get("reset_seed"),
                                "steps": summary.get("steps"),
                                "run_won": summary.get("run_won"),
                                "max_floor": summary.get("max_floor"),
                                "max_act": summary.get("max_act"),
                                "act1_cleared": summary.get("act1_cleared"),
                                "revivals_used": summary.get("revivals_used"),
                                "player_hp_lost": summary.get("player_hp_lost"),
                                "reward_total": summary.get("reward_total"),
                                "terminal_reason": summary.get("terminal_reason"),
                            }
                        )
                    dashboard_write(
                        {
                            "event": "learner_update",
                            "unix_s": time.time(),
                            "environment_steps": environment_steps,
                            "policy_version": learner.metrics.updates,
                            "loss": last_update_metrics.get("loss"),
                            "td_error_mean": last_update_metrics.get("td_error_mean"),
                            "windows_trained": last_update_metrics.get("windows_trained"),
                        }
                    )

                _save_then_acknowledge(
                    save_all,
                    publish_acknowledgement,
                    pending_paths,
                )
                pending_paths.clear()
                pending_path_set.clear()
                pending_summaries.clear()
                last_saved_at = ingested
                durable_ingested = ingested
                durable_environment_steps = environment_steps

            while ingested < args.stop_after_episodes:
                states = _require_producers_not_failed(
                    Path(args.producer_status_dir), producer_ids
                )
                remaining = args.stop_after_episodes - ingested
                batch = [
                    path
                    for path in sorted(spool.glob("*.pkl"))
                    if path not in pending_path_set
                ][: min(args.ingest_batch_episodes, remaining)]
                if not batch:
                    if states and all(state == "complete" for state in states.values()):
                        raise RuntimeError("all producers completed before the trainer reached its episode target")
                    if time.monotonic() - last_progress > args.idle_timeout_seconds:
                        raise TimeoutError("no producer or spool progress before idle timeout")
                    time.sleep(2.0)
                    continue

                fresh = 0
                fresh_environment_steps = 0
                for path in batch:
                    try:
                        with path.open("rb") as handle:
                            envelope = pickle.load(handle)
                    except (EOFError, pickle.UnpicklingError, OSError):
                        continue
                    if not isinstance(envelope, Mapping) or envelope.get("format") != SPOOL_ENVELOPE_FORMAT:
                        raise RuntimeError(f"unrecognized spool envelope: {path}")
                    if envelope.get("pipeline_id") != args.pipeline_id:
                        raise RuntimeError(f"spool item belongs to another pipeline: {path}")
                    if envelope.get("lineage_id") != args.lineage_id:
                        raise RuntimeError(f"spool item belongs to another lineage: {path}")
                    if envelope.get("control_domain") != args.control_domain:
                        raise RuntimeError(f"spool item belongs to another control domain: {path}")
                    episode = envelope.get("episode")
                    if not isinstance(episode, MacroEpisode):
                        raise RuntimeError(f"spool item contains no MacroEpisode: {path}")
                    summary = envelope.get("episode_summary")
                    if not isinstance(summary, Mapping):
                        raise RuntimeError(f"spool item contains no episode summary: {path}")
                    replay.put(episode)
                    episode_environment_steps = int(envelope.get("environment_steps") or 0)
                    fresh += 1
                    fresh_environment_steps += episode_environment_steps
                    pending_paths.append(path)
                    pending_path_set.add(path)
                    pending_summaries.append(
                        {
                            **dict(summary),
                            "ingested_total": ingested + fresh,
                            "environment_steps": environment_steps
                            + fresh_environment_steps,
                        }
                    )
                if not fresh:
                    continue
                last_progress = time.monotonic()
                ingested += fresh
                environment_steps += fresh_environment_steps
                update_budget = min(
                    args.updates_per_episode * fresh, args.max_updates_per_cycle
                )
                for _ in range(update_budget):
                    last_update_metrics = learner.update()
                if _checkpoint_due(
                    ingested=ingested,
                    last_saved_at=last_saved_at,
                    interval=args.save_interval_episodes,
                    target=args.stop_after_episodes,
                ):
                    flush_pending()
                if (
                    args.eval_interval_episodes > 0
                    and ingested - last_eval_at >= args.eval_interval_episodes
                ):
                    flush_pending()
                    run_evaluation_gate(environment_steps)
                    last_eval_at = ingested

            flush_pending()
            _require_producers_not_failed(Path(args.producer_status_dir), producer_ids)
            complete = {
                "event": "trainer_complete",
                "unix_s": time.time(),
                "pipeline_id": args.pipeline_id,
                "ingested_total": ingested,
                "environment_steps": environment_steps,
                "policy_version": learner.metrics.updates,
                "learner": learner.metrics.as_mapping(),
            }
            metrics_file.write(json.dumps(complete) + "\n")
            metrics_file.flush()
            dashboard_write(
                {
                    "event": "run_complete",
                    "unix_s": time.time(),
                    "environment_steps": environment_steps,
                    "episodes": ingested,
                }
            )
            dashboard_file.close()
        _atomic_json(status_path, {**complete, "state": "complete"})
        return 0
    except BaseException as exc:
        failure_status = {
            "event": "trainer_failed",
            "unix_s": time.time(),
            "pipeline_id": args.pipeline_id,
            "state": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "environment_steps": durable_environment_steps,
            "ingested_total": durable_ingested,
            "pending_environment_steps": environment_steps
            - durable_environment_steps,
            "pending_episodes": ingested - durable_ingested,
            "policy_version": learner.metrics.updates if learner is not None else 0,
        }
        _atomic_json(status_path, failure_status)
        with metrics_path.open("a", encoding="utf-8") as metrics_file:
            metrics_file.write(json.dumps(failure_status) + "\n")
        raise
    finally:
        if resources is not None:
            resources.close()


if __name__ == "__main__":
    raise SystemExit(main())
