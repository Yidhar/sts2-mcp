#!/usr/bin/env python3
"""Unified recurrent hybrid IQL trainer over complete factual runs."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import pickle
import random
import time
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from sts2_baseline import absolute_death_price_reward_identity
from sts2_rl.encoding.snapshot import collate_encoded_snapshots
from sts2_rl.joint import (
    HybridIQLLearner,
    HybridIQLLearnerConfig,
    HybridRun,
    HybridRunReplay,
)
from sts2_rl.models import (
    HybridDomainIQLModel,
    HybridIQLConfig,
    initialize_iql_from_candidate_q,
)
from sts2_rl.training import load_training_config
from sts2_rl.training.sdpa import configure_rocm_sdpa_backend

HYBRID_TRAINING_STATE = "sts2-hybrid-training-state-v4"
_CANDIDATE_Q_TRAINING_STATE = "sts2-hybrid-training-state-v2"
_PREVIOUS_HYBRID_TRAINING_STATE = "sts2-hybrid-training-state-v1"
HYBRID_SPOOL_ENVELOPE = "sts2-hybrid-spool-envelope-v2"
HYBRID_PUBLICATION = "sts2-hybrid-publication-v3"
_CANDIDATE_Q_PUBLICATION = "sts2-hybrid-publication-v2"
_PREVIOUS_HYBRID_PUBLICATION = "sts2-hybrid-publication-v1"
_Q_SCALE_CALIBRATION_THRESHOLD = 0.05


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(payload), sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _discard_metrics_beyond_checkpoint(path: Path, *, ingested: int) -> None:
    """Remove an uncheckpointed telemetry tail before exact continuation."""

    if not path.is_file():
        return
    retained: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        row_ingested = row.get("ingested_runs") if isinstance(row, Mapping) else None
        if isinstance(row_ingested, int) and not isinstance(row_ingested, bool) and row_ingested > ingested:
            continue
        retained.append(line)
    payload = "".join(f"{line}\n" for line in retained)
    temporary = path.with_suffix(path.suffix + ".resume.tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


def _heldout_summaries(path: Path) -> tuple[dict[str, Any], ...]:
    """Read completed held-out summaries in file order.

    The evaluation process owns this small JSONL feed.  Episode rows are not
    control evidence; only complete fixed-seed summaries update the lambda
    support gate.
    """

    if not path.is_file():
        return ()
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and row.get("event") == "hybrid_evaluation_summary":
            rows.append(row)
    return tuple(rows)


def _episode_outcome(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Project the collector's factual outcome without truthy coercion."""

    act1_cleared = summary.get("act1_cleared")
    run_success = summary.get("run_won")
    run_revivals = summary.get("revivals_used")
    act1_revivals = summary.get("act1_revivals")
    if not isinstance(act1_cleared, bool) or not isinstance(run_success, bool):
        raise ValueError("hybrid episode outcome flags are not booleans")
    if isinstance(run_revivals, bool) or not isinstance(run_revivals, int) or run_revivals < 0:
        raise ValueError("hybrid episode run revival count is invalid")
    if act1_revivals is not None and (
        isinstance(act1_revivals, bool) or not isinstance(act1_revivals, int) or act1_revivals < 0
    ):
        raise ValueError("hybrid episode Act-1 revival count is invalid")
    return {
        "act1_cleared": act1_cleared,
        "run_success": run_success,
        "act1_revivals": act1_revivals,
        "run_revivals": run_revivals,
    }


def _source_death_price(source: Mapping[str, Any]) -> float | None:
    """Read the scalar lambda from a v8 publication or training checkpoint."""

    direct = source.get("death_price")
    if isinstance(direct, Mapping):
        value = direct.get("lambda_value")
        if isinstance(value, int | float) and not isinstance(value, bool):
            return float(value)
    learner = source.get("learner")
    if isinstance(learner, Mapping):
        controller = learner.get("death_price")
        if isinstance(controller, Mapping):
            metrics = controller.get("metrics")
            if isinstance(metrics, Mapping):
                value = metrics.get("lambda_value")
                if isinstance(value, int | float) and not isinstance(value, bool):
                    return float(value)
    return None


def _source_domain_decisions(source: Mapping[str, Any]) -> dict[str, int] | None:
    """Carry collection-curriculum clocks across a model-init boundary.

    Learner updates, policy versions, optimizer state and replay still start a
    new lineage.  Domain decision counts are behavior-schedule progress: if
    they rewind, a mature parent abruptly returns to its maximum epsilon and
    the first corpus no longer represents the policy being initialized.
    """

    raw = source.get("domain_decisions")
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError("hybrid model-init domain decision clocks must be a mapping")
    decisions: dict[str, int] = {}
    for domain in ("macro", "combat"):
        value = raw.get(domain)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(
                f"hybrid model-init {domain} decision clock must be non-negative"
            )
        decisions[domain] = value
    return decisions


def _forward(model: HybridDomainIQLModel, encoder: Any, device: torch.device) -> Any:
    def call(
        step: Any,
        hidden: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded = collate_encoded_snapshots(
            (step.snapshot,),
            expected_config=encoder.config,
            expected_fingerprint=step.snapshot.encoding_fingerprint,
            device=device,
        )
        if hidden is None:
            hidden = model.initial_state(1, device=device)
        output = model(encoded, hidden, validate=False)
        return (
            output.q_values[0],
            output.state_value[0],
            output.policy_logits[0],
            output.recurrent_state,
        )

    return call


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
    if state.get("accelerator") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([item.cpu() for item in state["accelerator"]])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--champion", required=True)
    parser.add_argument("--spool", required=True)
    parser.add_argument("--publication", required=True)
    parser.add_argument("--checkpoint-out", required=True)
    parser.add_argument("--metrics-out", required=True)
    parser.add_argument("--status-out", required=True)
    parser.add_argument("--lineage-id", required=True)
    parser.add_argument("--pipeline-id", required=True)
    parser.add_argument("--producer-id", action="append", dest="producer_ids", required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--initialize-from", default=None)
    mode.add_argument("--resume", default=None)
    parser.add_argument("--stop-after-runs", type=int, default=400)
    parser.add_argument("--updates-per-run", type=int, default=2)
    parser.add_argument("--ingest-batch-runs", type=int, default=1)
    parser.add_argument("--run-batch-size", type=int, default=4)
    parser.add_argument("--replay-runs", type=int, default=128)
    parser.add_argument("--macro-unroll", type=int, default=8)
    parser.add_argument("--combat-unroll", type=int, default=32)
    parser.add_argument("--maximum-view-reuse", type=int, default=8)
    parser.add_argument("--maximum-policy-lag", type=int, default=32)
    parser.add_argument("--save-interval-runs", type=int, default=10)
    parser.add_argument("--publication-interval-runs", type=int, default=1)
    parser.add_argument("--maximum-pending-runs", type=int, default=2)
    parser.add_argument("--target-update-tau", type=float, default=0.02)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    status = Path(args.status_out)
    status_base = {
        "pipeline_id": args.pipeline_id,
        "producer_ids": list(args.producer_ids),
    }
    _atomic_json(status, {**status_base, "state": "running", "unix_s": time.time()})
    try:
        config = load_training_config(profile="preheat", config_path=Path(args.config))
        revival_budget = config.curriculum.revival_budget
        if isinstance(revival_budget, bool) or not isinstance(revival_budget, int):
            raise ValueError("reward-v8 hybrid training requires a finite revival budget")
        device_name = args.device
        if device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device_name)
        sdpa_backend = configure_rocm_sdpa_backend(
            config.runtime.rocm_sdpa_backend,
            devices=(device,),
        )
        print(
            json.dumps({"event": "sdpa_backend", **sdpa_backend.to_mapping()}),
            flush=True,
        )
        encoder_config = config.model.to_encoding_config()
        from sts2_rl.encoding import GroundedObservationEncoder

        encoder = GroundedObservationEncoder(encoder_config)
        grounded_config = config.model.to_model_config()
        hybrid_config = HybridIQLConfig(grounded_config)
        online = HybridDomainIQLModel(hybrid_config).to(device)
        target = copy.deepcopy(online).to(device)
        replay = HybridRunReplay(
            capacity_runs=args.replay_runs,
            macro_unroll=args.macro_unroll,
            combat_unroll=args.combat_unroll,
            maximum_view_reuse=args.maximum_view_reuse,
            maximum_policy_lag=args.maximum_policy_lag,
            seed=config.runtime.seed,
        )
        learner = HybridIQLLearner(
            online_model=online,
            target_model=target,
            replay=replay,
            forward_online=_forward(online, encoder, device),
            forward_target=_forward(target, encoder, device),
            initial_state=lambda: online.initial_state(1, device=device),
            config=HybridIQLLearnerConfig(
                learning_rate=config.optimization.learning_rate,
                run_batch_size=args.run_batch_size,
                target_update_tau=args.target_update_tau,
                revival_budget=revival_budget,
            ),
        )
        ingested = 0
        environment_steps = 0
        domain_decisions = {"macro": 0, "combat": 0}
        reward_identity = absolute_death_price_reward_identity()
        q_scale_calibrated = False
        pending_learner_updates = 0
        processed_heldout_policy_versions: set[int] = set()
        source = Path(args.initialize_from) if args.initialize_from else Path(args.champion) / "network.pt"
        if args.resume:
            # Replay is authoritative CPU-side factual state. Loading the
            # multi-gigabyte continuation directly onto the accelerator keeps
            # duplicate model/optimizer tensors (and any tensor-backed replay
            # fields) resident beside the live learner until the payload dies.
            # Restore from CPU; module/optimizer loaders move only their owned
            # state to the parameter device.
            payload = torch.load(args.resume, map_location="cpu", weights_only=False)
            if payload.get("format") != HYBRID_TRAINING_STATE:
                raise ValueError("resume is not a hybrid training state")
            if payload.get("lineage_id") != args.lineage_id:
                raise ValueError("resume lineage differs")
            if payload.get("model_config") != hybrid_config.as_mapping():
                raise ValueError("resume hybrid model config differs")
            if payload.get("reward_identity") != reward_identity:
                raise ValueError("resume reward identity differs")
            online.load_state_dict(payload["online"], strict=True)
            target.load_state_dict(payload["target"], strict=True)
            replay.load_state_dict(payload["replay"])
            learner.load_state_dict(payload["learner"])
            ingested = int(payload["ingested"])
            environment_steps = int(payload["environment_steps"])
            domain_decisions = {
                domain: int(payload.get("domain_decisions", {}).get(domain, 0)) for domain in ("macro", "combat")
            }
            q_scale_calibrated = bool(payload.get("q_scale_calibrated", False))
            pending_learner_updates = int(payload["pending_learner_updates"])
            if pending_learner_updates < 0:
                raise ValueError("resume pending learner-update count is negative")
            processed_heldout_policy_versions = {
                int(value) for value in payload.get("processed_heldout_policy_versions", ())
            }
            _restore_rng(payload["rng"])
            del payload
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        else:
            source_file = source / "network.pt" if source.is_dir() else source
            state = torch.load(source_file, map_location=device, weights_only=True)
            source_state = (
                state.get("online")
                if isinstance(state, Mapping)
                and state.get("format")
                in {
                    HYBRID_PUBLICATION,
                    _CANDIDATE_Q_PUBLICATION,
                    _PREVIOUS_HYBRID_PUBLICATION,
                    HYBRID_TRAINING_STATE,
                    _CANDIDATE_Q_TRAINING_STATE,
                    _PREVIOUS_HYBRID_TRAINING_STATE,
                }
                else state
            )
            if not isinstance(source_state, Mapping):
                raise ValueError("hybrid initialization has no model state")
            source_format = state.get("format") if isinstance(state, Mapping) else None
            if source_format == HYBRID_PUBLICATION:
                if state.get("model_config") != hybrid_config.as_mapping():
                    raise ValueError("IQL model-init source config differs")
                online.load_state_dict(dict(source_state), strict=True)
            elif source_format == _CANDIDATE_Q_PUBLICATION:
                initialize_iql_from_candidate_q(online, dict(source_state))
            else:
                raise ValueError(
                    "hybrid IQL model-init requires a finite Candidate-Q or IQL publication"
                )
            target.load_state_dict(online.state_dict(), strict=True)
            inherited_domain_decisions = _source_domain_decisions(state)
            if inherited_domain_decisions is not None:
                domain_decisions = inherited_domain_decisions
            # Candidate-Q v8 parents already completed their one-time scalar
            # calibration.  IQL inherits that Q scale; its V head starts on
            # the same offset and is fitted by the expectile objective.
            q_scale_calibrated = True
            inherited_price = _source_death_price(state)
            if inherited_price is not None:
                learner.initialize_death_price(inherited_price)
            elif revival_budget not in {0, 64}:
                raise ValueError("a tighter death-price course requires a v8 source lambda")

        def require_finite_model(model: torch.nn.Module, *, label: str) -> None:
            bad = [
                name
                for name, value in model.state_dict().items()
                if value.is_floating_point() and not bool(torch.isfinite(value).all().item())
            ]
            if bad:
                raise FloatingPointError(
                    f"{label} contains non-finite model tensors: {', '.join(bad[:8])}"
                )

        def publish() -> None:
            # The preceding production failure published NaN weights and then
            # killed both collectors.  A publication is now a finite model
            # boundary, not a best-effort serialization point.
            require_finite_model(online, label="hybrid IQL publication")
            publication = Path(args.publication)
            publication.parent.mkdir(parents=True, exist_ok=True)
            # A fixed ``network.pt.tmp`` is not safe once collectors reload in
            # parallel.  More importantly, Path.replace() can expose a brief
            # missing-path window on the Windows-mounted WSL filesystem.  A
            # unique completed file plus os.replace() keeps the old policy
            # readable until the new one is ready and atomically swaps the
            # directory entry.
            publication_temporary = publication.with_name(f".{publication.name}.{uuid.uuid4().hex}.tmp")
            try:
                torch.save(
                    {
                        "format": HYBRID_PUBLICATION,
                        "model_config": hybrid_config.as_mapping(),
                        "reward_identity": reward_identity,
                        "online": online.state_dict(),
                        "policy_version": int(learner.metrics.updates),
                        "control_scores": "policy_logits",
                        "domain_decisions": dict(domain_decisions),
                        "death_price": learner.death_price.as_mapping(),
                    },
                    publication_temporary,
                )
                os.replace(publication_temporary, publication)
            finally:
                publication_temporary.unlink(missing_ok=True)

        def save() -> None:
            require_finite_model(online, label="hybrid IQL checkpoint online model")
            require_finite_model(target, label="hybrid IQL checkpoint target model")
            checkpoint = Path(args.checkpoint_out)
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            temporary = checkpoint.with_suffix(checkpoint.suffix + ".tmp")
            torch.save(
                {
                    "format": HYBRID_TRAINING_STATE,
                    "lineage_id": args.lineage_id,
                    "model_config": hybrid_config.as_mapping(),
                    "reward_identity": reward_identity,
                    "online": online.state_dict(),
                    "target": target.state_dict(),
                    "learner": learner.state_dict(),
                    "replay": replay.state_dict(),
                    "ingested": ingested,
                    "environment_steps": environment_steps,
                    "domain_decisions": dict(domain_decisions),
                    "q_scale_calibrated": q_scale_calibrated,
                    "pending_learner_updates": pending_learner_updates,
                    "processed_heldout_policy_versions": tuple(sorted(processed_heldout_policy_versions)),
                    "rng": _rng_state(),
                },
                temporary,
                # The replay is mostly immutable Python facts rather than
                # tensor storage.  Zip serialization first builds a very
                # large in-memory pickle; the legacy writer streams that
                # pickle and cuts checkpoint-save RSS substantially.
                _use_new_zipfile_serialization=False,
            )
            temporary.replace(checkpoint)
            publish()

        def ingest_heldout_summaries() -> list[dict[str, Any]]:
            records: list[dict[str, Any]] = []
            feed = Path(args.metrics_out).parent / "evaluation.jsonl"
            for row in _heldout_summaries(feed):
                policy_version = row.get("policy_version")
                episodes = row.get("episodes")
                wins = row.get("wins")
                act1_clears = row.get("act1_clears")
                if not all(
                    isinstance(value, int) and not isinstance(value, bool)
                    for value in (policy_version, episodes, wins, act1_clears)
                ):
                    continue
                assert isinstance(policy_version, int)
                assert isinstance(episodes, int)
                assert isinstance(wins, int)
                assert isinstance(act1_clears, int)
                if (
                    policy_version in processed_heldout_policy_versions
                    or policy_version > learner.metrics.updates
                    or episodes <= 0
                ):
                    continue
                learner.observe_heldout(
                    act1_rate=act1_clears / episodes,
                    run_rate=wins / episodes,
                    policy_version=policy_version,
                )
                processed_heldout_policy_versions.add(policy_version)
                records.append(
                    {
                        "event": "death_price_heldout_observed",
                        "unix_s": time.time(),
                        "ingested_runs": ingested,
                        "policy_version": policy_version,
                        "episodes": episodes,
                        "act1_rate": act1_clears / episodes,
                        "run_rate": wins / episodes,
                        "death_price": learner.death_price.as_mapping(),
                    }
                )
            return records

        def maybe_calibrate_q_scale() -> dict[str, Any] | None:
            """Shift the common Candidate-Q bias before the first v8 gradient."""

            nonlocal q_scale_calibrated
            if q_scale_calibrated:
                return None
            runs = replay.runs
            eligible_runs = tuple(run for run in runs if run.macro_steps and run.combat_steps)
            if len(eligible_runs) < learner.config.run_batch_size:
                return None
            price = learner.death_price.lambda_value
            returns: list[float] = []
            for run in eligible_runs:
                running = 0.0
                for step in reversed(run.steps):
                    running = step.reward - price * step.revivals_delta + step.discount * running
                    returns.append(float(running))
            sampled_steps = [
                (run, index)
                for run in eligible_runs[:8]
                for index in range(0, len(run.steps), max(1, len(run.steps) // 16))
            ][:96]
            q_samples: list[float] = []
            online_was_training = online.training
            target_was_training = target.training
            online.eval()
            target.eval()
            try:
                with torch.no_grad():
                    for run, index in sampled_steps:
                        online_state, _ = learner._state_before(run, index)
                        values, _, _, _ = learner.forward_online(run.steps[index], online_state)
                        count = len(run.steps[index].snapshot.action_mask)
                        legal = values[:count]
                        mask = torch.tensor(
                            run.steps[index].snapshot.action_mask.tolist(),
                            dtype=torch.bool,
                            device=legal.device,
                        )
                        q_samples.extend(legal[mask].detach().cpu().tolist())
            finally:
                online.train(online_was_training)
                target.train(target_was_training)
            if not returns or not q_samples:
                return None
            mean_return = float(sum(returns) / len(returns))
            mean_q = float(sum(q_samples) / len(q_samples))
            delta = mean_return - mean_q
            q_scale_calibrated = True
            if abs(delta) >= _Q_SCALE_CALIBRATION_THRESHOLD:
                with torch.no_grad():
                    for model in (online, target):
                        final_layer = model.candidate_q_head[-1]
                        if not isinstance(final_layer, torch.nn.Linear):
                            raise TypeError("hybrid Candidate-Q final layer is not linear")
                        final_layer.bias.add_(delta)
                        value_final = model.state_value_head[-1]
                        if not isinstance(value_final, torch.nn.Linear):
                            raise TypeError("hybrid IQL value head final layer is not linear")
                        value_final.bias.add_(delta)
            return {
                "event": "hybrid_q_scale_calibration",
                "unix_s": time.time(),
                "ingested_runs": ingested,
                "mean_factual_return": mean_return,
                "mean_legal_q": mean_q,
                "bias_shift": (delta if abs(delta) >= _Q_SCALE_CALIBRATION_THRESHOLD else 0.0),
                "return_samples": len(returns),
                "q_samples": len(q_samples),
                "lambda_value": price,
            }

        if args.resume:
            # The checkpoint is already the durable authority.  Re-publish its
            # restored online weights so collectors cannot start from a newer,
            # uncheckpointed publication left by the interrupted process.
            # Rewriting the multi-gigabyte replay checkpoint here adds no state.
            publish()
        else:
            save()
        _atomic_json(
            status,
            {
                **status_base,
                "state": "running",
                "ingested_runs": ingested,
                "environment_steps": environment_steps,
                "learner_updates": learner.metrics.updates,
                "pending_learner_updates": pending_learner_updates,
                "death_price": learner.death_price.as_mapping(),
                "unix_s": time.time(),
            },
        )
        spool = Path(args.spool)
        spool.mkdir(parents=True, exist_ok=True)
        metrics_path = Path(args.metrics_out)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        if args.resume:
            _discard_metrics_beyond_checkpoint(metrics_path, ingested=ingested)
        last_save = ingested
        last_publication = ingested
        with metrics_path.open("a", encoding="utf-8") as metrics:
            for record in ingest_heldout_summaries():
                metrics.write(json.dumps(record) + "\n")
            metrics.flush()
            while ingested < args.stop_after_runs:
                for record in ingest_heldout_summaries():
                    metrics.write(json.dumps(record) + "\n")
                metrics.flush()
                files = sorted(
                    spool.glob("*.pkl"),
                    key=lambda item: item.stat().st_mtime_ns,
                )
                if not files:
                    time.sleep(0.25)
                    continue
                accepted: list[tuple[Path, Mapping[str, Any], HybridRun]] = []
                death_price_block_records: list[dict[str, Any]] = []
                remaining = args.stop_after_runs - ingested
                for path in files[: min(args.ingest_batch_runs, remaining)]:
                    with path.open("rb") as handle:
                        envelope = pickle.load(handle)
                    if envelope.get("format") != HYBRID_SPOOL_ENVELOPE:
                        raise ValueError(f"invalid hybrid spool item: {path}")
                    if envelope.get("pipeline_id") != args.pipeline_id:
                        raise ValueError(f"hybrid spool item belongs to another pipeline: {path}")
                    if envelope.get("producer_id") not in args.producer_ids:
                        raise ValueError(f"hybrid spool item has an unknown producer: {path}")
                    run = envelope.get("run")
                    if not isinstance(run, HybridRun):
                        raise ValueError(f"hybrid spool item has no run: {path}")
                    episode_summary = envelope.get("episode")
                    if not isinstance(episode_summary, Mapping):
                        raise ValueError(f"hybrid spool item has no episode summary: {path}")
                    replay.put(run)
                    ingested += 1
                    environment_steps += int(run.environment_steps)
                    for domain in ("macro", "combat"):
                        domain_decisions[domain] += sum(step.control_domain == domain for step in run.steps)
                    block_closed = learner.observe_episode_outcome(**_episode_outcome(episode_summary))
                    if block_closed:
                        death_price_block_records.append(
                            {
                                "event": "death_price_control_block",
                                "unix_s": time.time(),
                                "ingested_runs": ingested,
                                "death_price": learner.death_price.as_mapping(),
                            }
                        )
                    accepted.append((path, envelope, run))
                pending_learner_updates += args.updates_per_run * len(accepted)
                calibration = maybe_calibrate_q_scale()
                if calibration is not None:
                    # The calibrated common offset is durable and visible to
                    # collectors before the first ordinary gradient.
                    save()
                    last_save = ingested
                    last_publication = ingested
                    metrics.write(json.dumps(calibration) + "\n")
                for record in death_price_block_records:
                    metrics.write(json.dumps(record) + "\n")
                metrics.flush()
                update_metrics: Mapping[str, Any] = learner.as_mapping()
                update_started = time.monotonic()
                if q_scale_calibrated:
                    while pending_learner_updates > 0:
                        updates_before = learner.metrics.updates
                        update_metrics = learner.update()
                        if learner.metrics.updates == updates_before:
                            break
                        pending_learner_updates -= 1
                update_seconds = time.monotonic() - update_started
                rows: list[dict[str, Any]] = []
                first_ingested = ingested - len(accepted) + 1
                first_environment_steps = environment_steps - sum(int(run.environment_steps) for _, _, run in accepted)
                cumulative_environment_steps = first_environment_steps
                for offset, (path, envelope, run) in enumerate(accepted):
                    path.unlink()
                    cumulative_environment_steps += int(run.environment_steps)
                    episode_summary = envelope.get("episode")
                    if not isinstance(episode_summary, Mapping):
                        episode_summary = {}
                    rows.append(
                        {
                            "event": "hybrid_update",
                            "unix_s": time.time(),
                            "ingested_runs": first_ingested + offset,
                            "environment_steps": cumulative_environment_steps,
                            "producer_id": envelope.get("producer_id"),
                            "episode": dict(episode_summary),
                            "spool_pending": len(tuple(spool.glob("*.pkl"))),
                            "spool_capacity": args.maximum_pending_runs,
                            "learner_update_seconds": update_seconds,
                            "pending_learner_updates": pending_learner_updates,
                            "learner": dict(update_metrics),
                            "replay": replay.metrics(),
                        }
                    )
                for row in rows:
                    metrics.write(json.dumps(row) + "\n")
                metrics.flush()
                _atomic_json(
                    status,
                    {
                        **status_base,
                        "state": "running",
                        "ingested_runs": ingested,
                        "environment_steps": environment_steps,
                        "learner_updates": learner.metrics.updates,
                        "pending_learner_updates": pending_learner_updates,
                        "death_price": learner.death_price.as_mapping(),
                        "unix_s": time.time(),
                    },
                )
                checkpoint_due = ingested - last_save >= args.save_interval_runs
                if checkpoint_due:
                    save()
                    last_save = ingested
                    last_publication = ingested
                elif ingested - last_publication >= args.publication_interval_runs:
                    publish()
                    last_publication = ingested
        save()
        _atomic_json(
            status,
            {
                **status_base,
                "state": "complete",
                "ingested_runs": ingested,
                "environment_steps": environment_steps,
                "learner_updates": learner.metrics.updates,
                "pending_learner_updates": pending_learner_updates,
                "death_price": learner.death_price.as_mapping(),
                "unix_s": time.time(),
            },
        )
        return 0
    except BaseException as error:
        _atomic_json(
            status,
            {
                **status_base,
                "state": "failed",
                "error": f"{type(error).__name__}: {error}",
                "unix_s": time.time(),
            },
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
