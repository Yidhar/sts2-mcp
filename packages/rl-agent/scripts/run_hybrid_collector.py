#!/usr/bin/env python3
"""Collect complete mixed-domain runs for the hybrid Candidate-Q learner."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time
import uuid
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from sts2_rl.encoding.snapshot import collate_encoded_snapshots
from sts2_rl.joint import DomainExplorationSchedule, HybridCollectionAuthority
from sts2_rl.macro import load_trunk_state
from sts2_rl.models import HybridCandidateQConfig, HybridDomainCandidateQModel
from sts2_rl.training import build_training_resources, load_training_config

HYBRID_SPOOL_ENVELOPE = "sts2-hybrid-spool-envelope-v1"
HYBRID_PUBLICATION = "sts2-hybrid-publication-v1"


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(payload), sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--champion", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--spool", required=True)
    parser.add_argument("--status-out", required=True)
    parser.add_argument("--metrics-out", required=True)
    parser.add_argument("--pipeline-id", required=True)
    parser.add_argument("--producer-id", required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--reload-episodes", type=int, default=2)
    parser.add_argument("--maximum-pending-runs", type=int, default=2)
    parser.add_argument("--device", default=None)
    parser.add_argument("--sim-exe", default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    status = Path(args.status_out)
    base = {"pipeline_id": args.pipeline_id, "producer_id": args.producer_id}
    _atomic_json(status, {**base, "state": "running", "unix_s": time.time()})
    resources = None
    try:
        config = load_training_config(profile="preheat", config_path=Path(args.config))
        config = replace(config, runtime=replace(config.runtime, seed=args.seed))
        if args.device:
            config = replace(config, runtime=replace(config.runtime, device=args.device))
        if args.sim_exe:
            config = replace(
                config,
                environment=replace(
                    config.environment, backend="headless", sim_exe_path=args.sim_exe
                ),
            )
        resources = build_training_resources(config)
        if config.transaction_learning.enabled:
            raise RuntimeError(
                "hybrid collection requires transaction_learning.enabled=false; "
                "semantic Candidate-Q owns those choices directly"
            )
        champion_state = torch.load(
            Path(args.champion) / "network.pt",
            map_location=resources.device,
            weights_only=True,
        )
        legacy_head_groups = (
            "candidate_liveness_cost_head",
            "liveness_cost_value_head",
            "transaction_q_head",
            "macro_surface_candidate_embedding",
            "macro_surface_state_embedding",
            "macro_policy_head",
            "macro_option_value_head",
            # The hybrid profile disables legacy auxiliary heads.  They are
            # needed only by old fallback collection, not by Candidate-Q.
            "candidate_effect_head",
            "selection_delta_head",
        )
        for legacy_model in (resources.model, resources.collector_model):
            load_trunk_state(
                legacy_model,
                dict(champion_state),
                tolerated_groups=legacy_head_groups,
            )
        model = HybridDomainCandidateQModel(
            HybridCandidateQConfig(config.model.to_model_config())
        ).to(resources.device)
        model_path = Path(args.model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"hybrid publication does not exist: {model_path}")
        model_mtime = 0
        publication_state: dict[str, Any] = {}

        def reload() -> None:
            nonlocal model_mtime, publication_state
            mtime = model_path.stat().st_mtime_ns
            if mtime <= model_mtime:
                return
            payload = torch.load(
                model_path,
                map_location=resources.device,
                weights_only=True,
            )
            if not isinstance(payload, Mapping) or payload.get("format") != HYBRID_PUBLICATION:
                raise ValueError("collector publication is not a hybrid publication")
            if payload.get("model_config") != model.hybrid_config.as_mapping():
                raise ValueError("collector publication model config differs")
            online = payload.get("online")
            if not isinstance(online, Mapping):
                raise ValueError("hybrid publication has no online model")
            model.load_state_dict(dict(online), strict=True)
            model.eval()
            model_mtime = mtime
            publication_state = dict(payload)

        reload()

        def forward(snapshot: Any, hidden: Any) -> tuple[torch.Tensor, torch.Tensor]:
            encoded = collate_encoded_snapshots(
                (snapshot,),
                expected_config=resources.encoder.config,
                expected_fingerprint=snapshot.encoding_fingerprint,
                device=resources.device,
            )
            if hidden is None:
                hidden = model.initial_state(1, device=resources.device)
            with torch.no_grad():
                output = model(encoded, hidden, validate=False)
            return output.q_values[0], output.recurrent_state.detach()

        macro_schedule = DomainExplorationSchedule(
            start=config.curriculum.epsilon_start,
            end=config.curriculum.epsilon_end,
            decay_decisions=config.curriculum.epsilon_decay_steps,
        )
        # Combat receives its own clock even when both start from the same
        # config. Their counters advance only on decisions in that domain.
        combat_schedule = DomainExplorationSchedule(
            start=config.curriculum.epsilon_start,
            end=config.curriculum.epsilon_end,
            decay_decisions=config.curriculum.epsilon_decay_steps,
        )
        authority = HybridCollectionAuthority(
            forward_q=forward,
            initial_state=lambda: model.initial_state(1, device=resources.device),
            epsilon=0.0,
            seed=args.seed,
            macro_exploration=macro_schedule,
            combat_exploration=combat_schedule,
        )
        resources.collector.macro_authority = authority
        spool = Path(args.spool)
        spool.mkdir(parents=True, exist_ok=True)
        metrics_path = Path(args.metrics_out)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        nonce = uuid.uuid4().hex[:12]
        produced = 0
        with metrics_path.open("a", encoding="utf-8") as metrics:
            for episode_index in range(args.episodes):
                while len(tuple(spool.glob("*.pkl"))) >= args.maximum_pending_runs:
                    time.sleep(0.25)
                if episode_index % max(args.reload_episodes, 1) == 0:
                    reload()
                authority.load_exploration_state_dict(publication_state)
                authority.begin_episode(
                    f"{args.producer_id}:{nonce}:{episode_index}"
                )
                episode = resources.collector.collect_episode(
                    epsilon=0.0, deterministic=False, record=False
                )
                if not authority.terminal_observed:
                    terminal_reason = str(episode.metrics.terminal_reason or "")
                    factual_terminal = bool(
                        episode.metrics.run_won
                        or terminal_reason
                        in {
                        "deadlock",
                        "semantic_deadlock",
                        "combat_progress_stall",
                        "noncombat_event_action_cycle",
                        "selection_action_cycle",
                        "noncombat_progress_stall",
                        "revival_budget_exhausted",
                        }
                    )
                    if factual_terminal:
                        run_observation = getattr(resources.backend, "last_observation", None)
                        floor = None
                        if isinstance(run_observation, Mapping):
                            run = run_observation.get("run")
                            if isinstance(run, Mapping):
                                floor = run.get("floor")
                        authority.observe_step(
                            # The reward calculator already emitted this
                            # terminal cost on the last actual transition.
                            reward=0.0,
                            floor=floor if isinstance(floor, int) else None,
                            terminal=True,
                        )
                run = authority.finish_run(
                    f"{args.producer_id}:{nonce}:{episode_index}",
                    environment_steps=episode.metrics.steps,
                )
                if run is not None:
                    envelope = {
                        "format": HYBRID_SPOOL_ENVELOPE,
                        "pipeline_id": args.pipeline_id,
                        "producer_id": args.producer_id,
                        "run": run,
                    }
                    temporary = spool / f".tmp-{uuid.uuid4().hex}"
                    with temporary.open("wb") as handle:
                        pickle.dump(envelope, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    os.replace(temporary, spool / f"{uuid.uuid4().hex}.pkl")
                    produced += 1
                metrics.write(
                    json.dumps(
                        {
                            "event": "hybrid_collector_episode",
                            "episode_index": episode_index,
                            "steps": episode.metrics.steps,
                            "run_won": episode.metrics.run_won,
                            "max_floor": episode.metrics.max_floor,
                            "revivals_used": episode.metrics.revivals_used,
                            "terminal_reason": episode.metrics.terminal_reason,
                            "hybrid_steps": len(run.steps) if run else 0,
                            "authority": authority.metrics(),
                        }
                    )
                    + "\n"
                )
                metrics.flush()
        _atomic_json(
            status,
            {**base, "state": "complete", "produced": produced, "unix_s": time.time()},
        )
        return 0
    except BaseException as error:
        _atomic_json(
            status,
            {**base, "state": "failed", "error": f"{type(error).__name__}: {error}", "unix_s": time.time()},
        )
        raise
    finally:
        if resources is not None:
            resources.close()


if __name__ == "__main__":
    raise SystemExit(main())
