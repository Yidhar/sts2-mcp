from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from gymnasium import spaces
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv


RL_AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(RL_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(RL_AGENT_ROOT))

from train_attention_policy import (  # noqa: E402
    build_env_factory,
    build_policy_kwargs,
    parse_csv,
    resolve_snapshot_pool,
    resolve_text_device,
    resolve_training_session_files,
)
from sts2_env.async_ready_collector import AsyncReadyCollector, AsyncReadyItem, resolve_action_mask, stack_observations  # noqa: E402
from sts2_env.observation_v3 import MAX_ACTIONS, WorldTokenObservationEncoder  # noqa: E402
from sts2_env.omni_attention_policy import STS2OmniAttentionPolicy  # noqa: E402


DEFAULT_METRICS = (
    ("step.total", ("bridge_info", "step_timing_ms", "total")),
    ("step.after_wait", ("bridge_info", "step_timing_ms", "after_wait")),
    ("python.obs_encode", ("python_timing_ms", "obs_encode")),
    ("python.total", ("python_timing_ms", "total")),
)


@dataclass(slots=True)
class RolloutResult:
    rollout_index: int
    warmup: bool
    transitions: int
    elapsed_s: float
    steps_per_s: float
    timing: dict[str, dict[str, float]]
    env_step_counts: dict[int, int]


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _resolve_policy_device(device: str | None) -> str:
    candidate = str(device or "").strip().lower()
    if not candidate or candidate == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return str(device).strip()


def _metric_values(infos: list[dict[str, Any]], path: tuple[str, ...]) -> list[float]:
    values: list[float] = []
    for info in infos:
        current: Any = info
        for key in path:
            if not isinstance(current, dict):
                current = None
                break
            current = current.get(key)
        if current is None:
            continue
        try:
            values.append(float(current))
        except (TypeError, ValueError):
            continue
    return values


def _summarize_infos(infos: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for metric_name, path in DEFAULT_METRICS:
        values = _metric_values(infos, path)
        if not values:
            continue
        summary[metric_name] = {
            "count": float(len(values)),
            "mean": float(np.mean(values)),
            "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
        }
    return summary


def _aggregate_rollouts(rollouts: list[RolloutResult]) -> dict[str, Any]:
    measured = [rollout for rollout in rollouts if not rollout.warmup]
    total_elapsed = float(sum(rollout.elapsed_s for rollout in measured))
    total_transitions = int(sum(rollout.transitions for rollout in measured))
    env_step_counts: Counter[int] = Counter()
    infos_by_metric: dict[str, list[float]] = {metric_name: [] for metric_name, _ in DEFAULT_METRICS}

    for rollout in measured:
        for env_id, count in rollout.env_step_counts.items():
            env_step_counts[int(env_id)] += int(count)
        for metric_name, _path in DEFAULT_METRICS:
            metric_summary = rollout.timing.get(metric_name)
            if metric_summary and metric_summary.get("count", 0.0) > 0.0:
                if "samples" in metric_summary:
                    infos_by_metric[metric_name].extend(float(value) for value in metric_summary["samples"])  # type: ignore[index]

    timing_summary: dict[str, dict[str, float]] = {}
    for metric_name, values in infos_by_metric.items():
        if not values:
            continue
        timing_summary[metric_name] = {
            "count": float(len(values)),
            "mean": float(np.mean(values)),
            "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
        }

    per_rollout = [
        {
            "rollout_index": rollout.rollout_index,
            "warmup": rollout.warmup,
            "transitions": rollout.transitions,
            "elapsed_s": rollout.elapsed_s,
            "steps_per_s": rollout.steps_per_s,
            "timing": {
                metric_name: {
                    key: value
                    for key, value in metric.items()
                    if key != "samples"
                }
                for metric_name, metric in rollout.timing.items()
            },
            "env_step_counts": {str(env_id): int(count) for env_id, count in sorted(rollout.env_step_counts.items())},
        }
        for rollout in rollouts
    ]

    return {
        "measured_rollouts": len(measured),
        "measured_transitions": total_transitions,
        "elapsed_s": total_elapsed,
        "steps_per_s": (total_transitions / total_elapsed) if total_elapsed > 0 else 0.0,
        "timing": timing_summary,
        "env_step_counts": {str(env_id): int(count) for env_id, count in sorted(env_step_counts.items())},
        "per_rollout": per_rollout,
    }


def _attach_metric_samples(infos: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    summary = _summarize_infos(infos)
    for metric_name, path in DEFAULT_METRICS:
        values = _metric_values(infos, path)
        if metric_name not in summary:
            continue
        summary[metric_name]["samples"] = values  # type: ignore[index]
    return summary


def _make_policy(
    *,
    observation_space,
    device: str,
    policy_kwargs: dict[str, object],
) -> STS2OmniAttentionPolicy:
    policy = STS2OmniAttentionPolicy(
        observation_space,
        action_space=spaces.Discrete(MAX_ACTIONS),
        lr_schedule=lambda _progress: 0.0,
        **policy_kwargs,
    )
    policy = policy.to(torch.device(device))
    policy.set_training_mode(False)
    return policy


def _policy_actions(
    policy: STS2OmniAttentionPolicy,
    batched_obs: dict[str, np.ndarray],
    batched_masks: np.ndarray,
    device: str,
) -> np.ndarray:
    with torch.no_grad():
        obs_tensor = obs_as_tensor(batched_obs, device)
        actions, _values, _log_probs = policy(obs_tensor, action_masks=batched_masks)
    return actions.detach().cpu().numpy().reshape(-1)


def _build_env_fns(
    *,
    args: argparse.Namespace,
    session_files: list[str | None],
    snapshot_pool,
    log_dir: Path,
) -> list:
    log_dir.mkdir(parents=True, exist_ok=True)
    use_text = True
    text_device = resolve_text_device(args.text_device, args.device)
    encounter_pool = parse_csv(args.encounter_pool)
    return [
        build_env_factory(
            env_index=index,
            session_file=session_files[index],
            use_text=use_text,
            text_device=text_device,
            log_dir=str(log_dir),
            combat_sandbox=True,
            character=args.character,
            encounter_id=args.encounter_id,
            encounter_pool=encounter_pool,
            snapshot_pool=snapshot_pool,
            reset_timeout_ms=args.reset_timeout_ms,
            step_timeout_ms=args.step_timeout_ms,
        )
        for index in range(args.n_envs)
    ]


def _run_smoke(env_fns: list, *, n_envs: int) -> list[dict[str, Any]]:
    smoke_rows: list[dict[str, Any]] = []
    for env_index in range(n_envs):
        env = env_fns[env_index]()
        try:
            obs, info = env.reset()
            action_mask = np.asarray(info.get("action_mask"), dtype=np.float32)
            smoke_rows.append(
                {
                    "env_id": env_index,
                    "legal_action_count": int(info.get("legal_action_count", 0)),
                    "action_mask_nonzero": int(np.count_nonzero(action_mask > 0.5)),
                    "phase": str(info.get("phase") or "unknown"),
                    "episode_id": str(info.get("episode_id") or ""),
                    "world_tokens_shape": list(np.asarray(obs["world_tokens"]).shape),
                    "candidate_local_shape": list(np.asarray(obs["candidate_local_tokens"]).shape),
                }
            )
        finally:
            env.close()
    return smoke_rows


def _run_sync_mode(
    *,
    args: argparse.Namespace,
    env_fns: list,
    policy: STS2OmniAttentionPolicy,
    device: str,
) -> dict[str, Any]:
    vec_env = DummyVecEnv(env_fns) if args.n_envs == 1 else SubprocVecEnv(env_fns)
    try:
        obs = vec_env.reset()
        rollout_results: list[RolloutResult] = []
        total_rollouts = int(args.warmup_rollouts) + int(args.measured_rollouts)
        for rollout_index in range(total_rollouts):
            warmup = rollout_index < int(args.warmup_rollouts)
            infos: list[dict[str, Any]] = []
            start = time.perf_counter()
            env_counts = {env_id: 0 for env_id in range(args.n_envs)}
            for _ in range(int(args.n_steps)):
                action_masks = np.asarray(get_action_masks(vec_env), dtype=bool)
                actions = _policy_actions(policy, obs, action_masks, device)
                obs, _rewards, _dones, step_infos = vec_env.step(actions)
                for env_id, info in enumerate(step_infos):
                    if isinstance(info, dict):
                        infos.append(info)
                    env_counts[env_id] += 1
            elapsed_s = time.perf_counter() - start
            transitions = int(args.n_steps) * int(args.n_envs)
            rollout_results.append(
                RolloutResult(
                    rollout_index=rollout_index,
                    warmup=warmup,
                    transitions=transitions,
                    elapsed_s=elapsed_s,
                    steps_per_s=(transitions / elapsed_s) if elapsed_s > 0 else 0.0,
                    timing=_attach_metric_samples(infos),
                    env_step_counts=env_counts,
                )
            )
        return _aggregate_rollouts(rollout_results)
    finally:
        vec_env.close()


def _clean_ready_item(item: AsyncReadyItem) -> AsyncReadyItem:
    return AsyncReadyItem(
        env_id=item.env_id,
        obs=item.obs,
        info=item.info,
        episode_start=item.episode_start,
    )


def _run_async_mode(
    *,
    args: argparse.Namespace,
    env_fns: list,
    policy: STS2OmniAttentionPolicy,
    device: str,
) -> dict[str, Any]:
    collector = AsyncReadyCollector(env_fns)
    collector.start()
    try:
        rollout_results: list[RolloutResult] = []
        total_rollouts = int(args.warmup_rollouts) + int(args.measured_rollouts)
        rollout_target = int(args.n_steps) * int(args.n_envs)
        for rollout_index in range(total_rollouts):
            warmup = rollout_index < int(args.warmup_rollouts)
            ready_pool: dict[int, AsyncReadyItem] = {}
            pending: set[int] = set()
            infos: list[dict[str, Any]] = []
            env_counts: Counter[int] = Counter()
            transitions = 0
            start = time.perf_counter()

            while transitions < rollout_target or pending:
                need_item = transitions < rollout_target and not ready_pool
                min_items = 1 if (need_item or pending) else 0
                ready_items = collector.drain_ready(min_items=min_items, timeout_s=float(args.queue_timeout_s))
                for item in ready_items:
                    next_ready = item
                    if item.transition_info is not None and item.env_id in pending:
                        pending.remove(item.env_id)
                        transitions += 1
                        if isinstance(item.transition_info, dict):
                            infos.append(item.transition_info)
                        env_counts[item.env_id] += 1
                        next_ready = _clean_ready_item(item)
                    elif item.transition_info is not None:
                        next_ready = _clean_ready_item(item)
                    ready_pool[item.env_id] = next_ready

                remaining_dispatch = max(rollout_target - (transitions + len(pending)), 0)
                if remaining_dispatch <= 0:
                    continue

                ready_env_ids = sorted(ready_pool.keys())[:remaining_dispatch]
                if not ready_env_ids:
                    continue

                dispatch_items = [ready_pool.pop(env_id) for env_id in ready_env_ids]
                batched_obs = stack_observations([item.obs for item in dispatch_items])
                batched_masks = np.stack(
                    [resolve_action_mask(item.obs, item.info) for item in dispatch_items],
                    axis=0,
                ).astype(bool)
                actions = _policy_actions(policy, batched_obs, batched_masks, device)
                collector.dispatch_actions(ready_env_ids, actions)
                pending.update(ready_env_ids)

            elapsed_s = time.perf_counter() - start
            collector.push_ready_back(list(ready_pool.values()))
            rollout_results.append(
                RolloutResult(
                    rollout_index=rollout_index,
                    warmup=warmup,
                    transitions=transitions,
                    elapsed_s=elapsed_s,
                    steps_per_s=(transitions / elapsed_s) if elapsed_s > 0 else 0.0,
                    timing=_attach_metric_samples(infos),
                    env_step_counts=dict(env_counts),
                )
            )

        return _aggregate_rollouts(rollout_results)
    finally:
        collector.close()


def _comparison_summary(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    sync = results.get("sync")
    async_result = results.get("async")
    comparison: dict[str, Any] = {}
    if sync and async_result:
        sync_steps = float(sync.get("steps_per_s", 0.0))
        async_steps = float(async_result.get("steps_per_s", 0.0))
        comparison["steps_per_s_ratio_async_vs_sync"] = (async_steps / sync_steps) if sync_steps > 0 else 0.0

        for metric_name in ("step.total", "step.after_wait", "python.obs_encode", "python.total"):
            sync_metric = ((sync.get("timing") or {}).get(metric_name) or {})
            async_metric = ((async_result.get("timing") or {}).get(metric_name) or {})
            if "p50" in sync_metric and "p50" in async_metric:
                sync_p50 = float(sync_metric["p50"])
                async_p50 = float(async_metric["p50"])
                comparison[f"{metric_name}.p50_ratio_async_vs_sync"] = (async_p50 / sync_p50) if sync_p50 > 0 else 0.0
                comparison[f"{metric_name}.p50_delta_ms_async_minus_sync"] = async_p50 - sync_p50
    return comparison


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark sync vs async attention collectors on real 4-env combat sandbox.")
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--session-file", type=str, default=None)
    parser.add_argument("--session-files", type=str, default=None)
    parser.add_argument("--collector-modes", type=str, default="sync,async")
    parser.add_argument("--warmup-rollouts", type=int, default=1)
    parser.add_argument("--measured-rollouts", type=int, default=2)
    parser.add_argument("--n-steps", type=int, default=16)
    parser.add_argument("--queue-timeout-s", type=float, default=120.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--character", type=str, default=None)
    parser.add_argument("--encounter-id", type=str, default=None)
    parser.add_argument("--encounter-pool", type=str, default=None)
    parser.add_argument("--snapshot-pool", type=str, default="E:/game/project/sts2_mcp/datasets/curated_combat_allchars_mixed_provenance")
    parser.add_argument(
        "--snapshot-curated-subset",
        type=str,
        default="bootstrap_human_plus_local_act1clear_weak_normal_roomwin_only_minus_combat_reset_failures",
    )
    parser.add_argument(
        "--snapshot-sample-mode",
        type=str,
        default="encounter_balanced",
        choices=("row_uniform", "encounter_balanced", "tier_weighted_encounter_balanced"),
    )
    parser.add_argument("--snapshot-encounter-tiers", type=str, default="weak,normal")
    parser.add_argument("--snapshot-max-rows", type=int, default=None)
    parser.add_argument("--log-dir", type=str, default="logs_attention/benchmarks")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--text-device", type=str, default="auto")
    parser.add_argument("--reset-timeout-ms", type=int, default=60000)
    parser.add_argument("--step-timeout-ms", type=int, default=20000)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--ffn-dim", type=int, default=1024)
    parser.add_argument("--world-layers", type=int, default=6)
    parser.add_argument("--local-layers", type=int, default=1)
    parser.add_argument("--decoder-layers", type=int, default=2)
    parser.add_argument("--candidate-set-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = _resolve_policy_device(args.device)
    text_device = resolve_text_device(args.text_device, device)
    session_files = resolve_training_session_files(
        n_envs=int(args.n_envs),
        session_file=args.session_file,
        session_files=parse_csv(args.session_files),
    )
    snapshot_pool = resolve_snapshot_pool(args)

    timestamp = _timestamp()
    output_root = (RL_AGENT_ROOT / args.log_dir / f"attention_benchmark_{timestamp}").resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "started_at_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_envs": int(args.n_envs),
        "session_files": [str(path) if path is not None else None for path in session_files],
        "collector_modes": [mode.strip().lower() for mode in str(args.collector_modes).split(",") if mode.strip()],
        "warmup_rollouts": int(args.warmup_rollouts),
        "measured_rollouts": int(args.measured_rollouts),
        "n_steps": int(args.n_steps),
        "device": device,
        "text_device": text_device,
        "snapshot_pool": args.snapshot_pool,
        "snapshot_curated_subset": args.snapshot_curated_subset,
        "snapshot_sample_mode": args.snapshot_sample_mode,
        "snapshot_encounter_tiers": args.snapshot_encounter_tiers,
        "snapshot_max_rows": args.snapshot_max_rows,
        "character": args.character,
        "encounter_id": args.encounter_id,
        "encounter_pool": args.encounter_pool,
    }
    _write_json(output_root / "manifest.json", manifest)

    smoke_env_fns = _build_env_fns(
        args=args,
        session_files=session_files,
        snapshot_pool=snapshot_pool,
        log_dir=output_root / "smoke_logs",
    )
    smoke = _run_smoke(smoke_env_fns, n_envs=int(args.n_envs))

    obs_encoder = WorldTokenObservationEncoder(use_text=True, text_device=text_device)
    policy_kwargs = build_policy_kwargs(args)
    requested_modes = [mode.strip().lower() for mode in str(args.collector_modes).split(",") if mode.strip()]
    results: dict[str, dict[str, Any]] = {}

    for mode in requested_modes:
        mode_output_dir = output_root / f"{mode}_logs"
        env_fns = _build_env_fns(
            args=args,
            session_files=session_files,
            snapshot_pool=snapshot_pool,
            log_dir=mode_output_dir,
        )
        policy = _make_policy(
            observation_space=obs_encoder.obs_space,
            device=device,
            policy_kwargs=policy_kwargs,
        )

        print(
            json.dumps(
                {
                    "event": "benchmark_mode_start",
                    "mode": mode,
                    "started_at_local": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "n_envs": int(args.n_envs),
                    "n_steps": int(args.n_steps),
                    "warmup_rollouts": int(args.warmup_rollouts),
                    "measured_rollouts": int(args.measured_rollouts),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

        if mode == "sync":
            mode_result = _run_sync_mode(
                args=args,
                env_fns=env_fns,
                policy=policy,
                device=device,
            )
        elif mode == "async":
            mode_result = _run_async_mode(
                args=args,
                env_fns=env_fns,
                policy=policy,
                device=device,
            )
        else:
            raise ValueError(f"Unsupported collector mode: {mode}")

        results[mode] = mode_result
        _write_json(output_root / f"{mode}_result.json", mode_result)
        print(
            json.dumps(
                {
                    "event": "benchmark_mode_done",
                    "mode": mode,
                    "steps_per_s": mode_result.get("steps_per_s"),
                    "elapsed_s": mode_result.get("elapsed_s"),
                    "measured_transitions": mode_result.get("measured_transitions"),
                    "timing": mode_result.get("timing"),
                    "env_step_counts": mode_result.get("env_step_counts"),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    final_payload = {
        "manifest": manifest,
        "smoke": smoke,
        "results": results,
        "comparison": _comparison_summary(results),
    }
    _write_json(output_root / "summary.json", final_payload)
    print(json.dumps(final_payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
