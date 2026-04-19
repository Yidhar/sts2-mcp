"""Train the unified omni-attention policy without any runtime search."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import time
from pathlib import Path
from typing import Callable

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from combat_snapshot_dataset import CombatSnapshotPool
from launcher import get_session_files as get_default_multi_session_files
from sts2_env.async_ready_collector import (
    AsyncReadyCollector,
    PrewarmedEnvSlot,
    resolve_action_mask,
    stack_observations,
)
from sts2_env.aux_maskable_ppo import AsyncAuxMaskableDictRolloutBuffer, AuxMaskablePPO
from sts2_env.checkpoint import load_online_policy_state_dict, save_online_checkpoint, save_rotating_online_checkpoint
from sts2_env.combat_env import CombatSandboxEnv
from sts2_env.env_v2 import SlayTheSpire2EnvV2
from sts2_env.omni_attention_policy import (
    ATTENTION_ARCHITECTURE_VERSION,
    CANDIDATE_AUX_HEAD_NAMES,
    DEFAULT_POLICY_CLASS_PATH,
    GLOBAL_AUX_HEAD_NAMES,
    STS2OmniAttentionPolicy,
    WORLD_BANK_NAMES,
)
from sts2_env.observation_v3 import MAX_ACTIONS, MAX_CANDIDATE_LOCAL_TOKENS, OBSERVATION_API_VERSION, WorldTokenObservationEncoder


def mask_fn(env):
    return env.unwrapped.action_masks()


class PerfStatsPeriodicLogger(gym.Wrapper):
    """Pulls bridge.perf_stats() every N successful steps and appends to a
    JSONL log. Used to confirm whether the C# hang-relief drain patches are
    firing — non-zero deadline counters in the dumps point at paths still
    hitting their wall-clock budgets."""

    def __init__(self, env, *, log_path, interval_steps: int, env_index: int):
        super().__init__(env)
        self._log_path = Path(log_path)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._interval = max(int(interval_steps), 1)
        self._env_index = int(env_index)
        self._step_count = 0

    def step(self, action):
        out = self.env.step(action)
        self._step_count += 1
        if self._step_count % self._interval == 0:
            try:
                stats = self.env.unwrapped.bridge.perf_stats()
            except Exception as exc:  # noqa: BLE001
                stats = {"_error": repr(exc)}
            line = {
                "t": time.time(),
                "env_index": self._env_index,
                "step_count": self._step_count,
                "stats": stats,
            }
            try:
                with self._log_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(line, ensure_ascii=False, sort_keys=True))
                    fh.write("\n")
            except OSError:
                pass
        return out


def parse_csv(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [entry.strip() for entry in raw.split(",") if entry.strip()]


def parse_weight_map(raw: str | None) -> dict[str, float]:
    if not raw:
        return {}
    parsed: dict[str, float] = {}
    for token in [entry.strip() for entry in str(raw).split(",") if entry.strip()]:
        if ":" not in token:
            continue
        key, value = token.split(":", 1)
        normalized_key = key.strip().lower()
        if not normalized_key:
            continue
        try:
            weight = float(value)
        except ValueError:
            continue
        if weight > 0.0:
            parsed[normalized_key] = weight
    return parsed


def resolve_snapshot_pool(args) -> CombatSnapshotPool | None:
    if not args.snapshot_pool:
        return None

    encounter_tiers = parse_csv(getattr(args, "snapshot_encounter_tiers", None))
    tier_weights = parse_weight_map(getattr(args, "snapshot_tier_weights", None))
    return CombatSnapshotPool.from_path(
        args.snapshot_pool,
        curated_subset=getattr(args, "snapshot_curated_subset", None),
        encounter_tiers=encounter_tiers or None,
        max_rows=getattr(args, "snapshot_max_rows", None),
        sample_mode=getattr(args, "snapshot_sample_mode", "encounter_balanced"),
        tier_weights=tier_weights or None,
        starter_early_boost=float(getattr(args, "snapshot_starter_early_boost", 0.0) or 0.0),
    )


def resolve_training_session_files(*, n_envs: int, session_file: str | None, session_files: list[str]) -> list[str | None]:
    if n_envs < 1:
        raise ValueError("--n-envs must be >= 1")
    if session_file and session_files:
        raise ValueError("Use either --session-file or --session-files, not both.")
    if session_files:
        if len(session_files) != n_envs:
            raise ValueError(f"--session-files count ({len(session_files)}) must match --n-envs ({n_envs}).")
        return session_files
    if n_envs == 1:
        return [session_file]
    return get_default_multi_session_files(n_envs)


def resolve_text_device(text_device: str | None, fallback_device: str | None = None) -> str:
    candidate = str(text_device or "").strip().lower()
    if candidate and candidate != "auto":
        return str(text_device).strip()

    fallback = str(fallback_device or "").strip()
    if fallback and fallback.lower() != "auto":
        return fallback

    return "cuda" if torch.cuda.is_available() else "cpu"


def build_env_factory(
    *,
    env_index: int,
    session_file: str | None,
    use_text: bool,
    text_device: str,
    log_dir: str,
    combat_sandbox: bool,
    character: str | None,
    encounter_id: str | None,
    encounter_pool: list[str],
    snapshot_pool: CombatSnapshotPool | None,
    reset_timeout_ms: int,
    step_timeout_ms: int,
    use_sim: bool = False,
    sim_exe_path: str | None = None,
    perf_stats_log_path: str | None = None,
    perf_stats_interval_steps: int = 0,
    encode_pool: "Any | None" = None,
    stuck_watchdog_steps: int = 400,
) -> Callable[[], object]:
    def _factory():
        local_encoder = WorldTokenObservationEncoder(use_text=use_text, text_device=text_device)
        if encode_pool is not None:
            from sts2_env.encode_pool import PooledObsEncoder
            obs_encoder = PooledObsEncoder(pool=encode_pool, local_encoder=local_encoder)
        else:
            obs_encoder = local_encoder
        # When --use-sim, drive the frankqwang/sts2-ai C# headless sim
        # instead of a live Godot game instance. Sim bridge is injected per
        # env (one subprocess each). This bypasses all session_file / HTTP
        # bridge machinery — sim speaks line-delimited JSON over stdio.
        sim_bridge = None
        if use_sim:
            from sts2_env.headless_sim_bridge_client import HeadlessSimBridgeClient
            sim_bridge = HeadlessSimBridgeClient(exe_path=sim_exe_path)
        if combat_sandbox:
            env = CombatSandboxEnv(
                session_file=session_file,
                character=character,
                encounter_id=encounter_id,
                encounter_pool=encounter_pool,
                snapshot_pool=snapshot_pool,
                reset_timeout_ms=reset_timeout_ms,
                step_timeout_ms=step_timeout_ms,
                obs_encoder=obs_encoder,
                bridge=sim_bridge,
            )
        else:
            env = SlayTheSpire2EnvV2(
                session_file=session_file,
                character=character,
                reset_timeout_ms=reset_timeout_ms,
                step_timeout_ms=step_timeout_ms,
                obs_encoder=obs_encoder,
                bridge=sim_bridge,
                stuck_watchdog_steps=stuck_watchdog_steps,
            )
        if use_sim and perf_stats_log_path and perf_stats_interval_steps > 0:
            env = PerfStatsPeriodicLogger(
                env,
                log_path=perf_stats_log_path,
                interval_steps=perf_stats_interval_steps,
                env_index=env_index,
            )
        monitor_name = "monitor" if env_index == 0 else f"monitor_{env_index}"
        # info_keywords: SB3 Monitor persists these info fields into
        # monitor.monitor.csv alongside r,l,t when an episode ends. Gives
        # us per-episode floor progression at the training-log level.
        env = Monitor(
            env,
            filename=str(Path(log_dir) / monitor_name),
            info_keywords=("max_floor_reached", "current_floor"),
        )
        env = ActionMasker(env, mask_fn)
        return env

    return _factory


def build_policy_kwargs(args) -> dict[str, object]:
    return {
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "ffn_dim": args.ffn_dim,
        "world_layers": args.world_layers,
        "local_layers": args.local_layers,
        "decoder_layers": args.decoder_layers,
        "candidate_set_layers": args.candidate_set_layers,
        "world_bank_top_k": args.world_bank_top_k,
        "dropout": args.dropout,
    }


def build_checkpoint_metadata(
    *,
    model: AuxMaskablePPO,
    args,
    policy_kwargs: dict[str, object],
    collector_mode: str,
    use_text: bool,
    text_device: str,
    started_at_unix_s: float,
) -> dict[str, object]:
    elapsed_s = time.time() - float(started_at_unix_s)
    return {
        "format": "sts2-online-policy-attention-v3-frozen",
        "attention_architecture_version": ATTENTION_ARCHITECTURE_VERSION,
        "policy_class": DEFAULT_POLICY_CLASS_PATH,
        "policy_kwargs": policy_kwargs,
        "observation_class": "sts2_env.observation_v3.WorldTokenObservationEncoder",
        "observation_api_version": OBSERVATION_API_VERSION,
        "collector_mode": collector_mode,
        "candidate_local_tokens": int(MAX_CANDIDATE_LOCAL_TOKENS),
        "world_banks": list(WORLD_BANK_NAMES),
        "world_bank_top_k": int(args.world_bank_top_k),
        "global_aux_heads": list(GLOBAL_AUX_HEAD_NAMES),
        "candidate_aux_heads": list(CANDIDATE_AUX_HEAD_NAMES),
        "use_text": use_text,
        "text_device": text_device,
        "amp_enabled": bool(getattr(model, "amp_enabled", False)),
        "amp_requested": bool(args.amp),
        "amp_dtype": str(getattr(model, "amp_dtype_name", args.amp_dtype)),
        "algorithm_class": "sts2_env.aux_maskable_ppo.AuxMaskablePPO",
        "aux_objective_coef": float(args.aux_objective_coef),
        "aux_transition_coef": float(args.aux_transition_coef),
        "aux_trait_coef": float(args.aux_trait_coef),
        "aux_build_coef": float(args.aux_build_coef),
        "aux_selection_coef": float(args.aux_selection_coef),
        "aux_route_coef": float(args.aux_route_coef),
        "aux_enemy_state_coef": float(args.aux_enemy_state_coef),
        "snapshot_pool_root": args.snapshot_pool,
        "snapshot_curated_subset": getattr(args, "snapshot_curated_subset", None),
        "snapshot_sample_mode": getattr(args, "snapshot_sample_mode", None),
        "snapshot_encounter_tiers": parse_csv(getattr(args, "snapshot_encounter_tiers", None)),
        "snapshot_tier_weights": parse_weight_map(getattr(args, "snapshot_tier_weights", None)),
        "snapshot_starter_early_boost": float(getattr(args, "snapshot_starter_early_boost", 0.0) or 0.0),
        "timesteps": int(model.num_timesteps),
        "mode": "combat_sandbox" if args.combat_sandbox else "full_run",
        "trained_at_unix_s": time.time(),
        "elapsed_train_s": elapsed_s,
    }


def _zeros_from_space(space):
    if isinstance(space, spaces.Dict):
        return {key: _zeros_from_space(subspace) for key, subspace in space.spaces.items()}
    return np.zeros(space.shape, dtype=space.dtype)


class ObservationSpecEnv(gym.Env):
    def __init__(self, observation_space: spaces.Dict):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = spaces.Discrete(MAX_ACTIONS)
        self._obs = _zeros_from_space(observation_space)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        return self._obs, {}

    def step(self, action):
        return self._obs, 0.0, False, False, {}


def _format_rollout_stats(rollout_index: int, total_timesteps: int, payload: dict[str, object]) -> dict[str, object]:
    summary = {
        "rollout": rollout_index,
        "timesteps": int(total_timesteps),
        "transitions": int(payload.get("transitions", 0)),
    }
    timing = payload.get("timing") if isinstance(payload, dict) else None
    if isinstance(timing, dict):
        summary["timing"] = timing
    phase_timing = payload.get("phase_timing_s") if isinstance(payload, dict) else None
    if isinstance(phase_timing, dict):
        summary["phase_timing_s"] = phase_timing
    for key in (
        "collector_timing_s",
        "collector_counts",
        "collector_latency_ms",
        "train_phase_timing_s",
        "train_counts",
        "train_latency_ms",
    ):
        value = payload.get(key) if isinstance(payload, dict) else None
        if isinstance(value, dict):
            summary[key] = value
    return summary


def _emit_json_event(payload: dict[str, object], *, enabled: bool) -> None:
    if not enabled:
        return
    print(json.dumps(payload, ensure_ascii=False))


def _write_status_payload(status_path: Path | None, payload: dict[str, object]) -> None:
    if status_path is None:
        return
    status_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = status_path.with_suffix(status_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(status_path)


def serial_prewarm_async_envs(env_fns: list[Callable[[], object]]) -> tuple[dict[int, PrewarmedEnvSlot], dict[str, object]]:
    prewarmed: dict[int, PrewarmedEnvSlot] = {}
    rows: list[dict[str, object]] = []
    total_started = time.perf_counter()
    try:
        for env_id, env_fn in enumerate(env_fns):
            env = env_fn()
            reset_started = time.perf_counter()
            obs, info = env.reset()
            reset_elapsed_ms = (time.perf_counter() - reset_started) * 1000.0
            info_dict = dict(info) if isinstance(info, dict) else {}
            info_dict["_serial_prewarm_reset_ms"] = float(reset_elapsed_ms)
            python_timing = info_dict.get("python_timing_ms") if isinstance(info_dict, dict) else None
            rows.append(
                {
                    "env_id": env_id,
                    "reset_ms": float(reset_elapsed_ms),
                    "phase": str(info_dict.get("phase") or "unknown"),
                    "legal_action_count": int(info_dict.get("legal_action_count", 0) or 0),
                    "python_obs_encode_ms": float((python_timing or {}).get("obs_encode", 0.0)) if isinstance(python_timing, dict) else 0.0,
                    "python_total_ms": float((python_timing or {}).get("total", 0.0)) if isinstance(python_timing, dict) else 0.0,
                    "bridge_roundtrip_ms": float((python_timing or {}).get("bridge_roundtrip", 0.0)) if isinstance(python_timing, dict) else 0.0,
                }
            )
            prewarmed[env_id] = PrewarmedEnvSlot(env=env, obs=obs, info=info_dict)
    except Exception:
        for slot in prewarmed.values():
            try:
                slot.env.close()
            except Exception:
                pass
        raise

    reset_values = [float(row["reset_ms"]) for row in rows]
    summary = {
        "phase": "serial_prewarm",
        "env_count": len(rows),
        "total_s": time.perf_counter() - total_started,
        "reset_ms": {
            "mean": float(np.mean(reset_values)) if reset_values else 0.0,
            "p50": float(np.percentile(reset_values, 50)) if reset_values else 0.0,
            "p95": float(np.percentile(reset_values, 95)) if reset_values else 0.0,
            "max": float(np.max(reset_values)) if reset_values else 0.0,
        },
        "envs": rows,
    }
    return prewarmed, summary


def warm_policy_on_initial_ready(model: AuxMaskablePPO, collector: AsyncReadyCollector) -> dict[str, object]:
    ready_items = collector.drain_ready(min_items=collector.num_envs, timeout_s=180.0)
    try:
        batched_obs = stack_observations([item.obs for item in ready_items])
        batched_masks = np.stack([resolve_action_mask(item.obs, item.info) for item in ready_items], axis=0).astype(bool)
        warm_started = time.perf_counter()
        autocast_context = model.amp_autocast_context() if hasattr(model, "amp_autocast_context") else nullcontext()
        with torch.no_grad():
            obs_tensor = obs_as_tensor(batched_obs, model.device)
            with autocast_context:
                model.policy(obs_tensor, action_masks=batched_masks)
        warm_elapsed_s = time.perf_counter() - warm_started
        return {
            "phase": "policy_warmup",
            "env_count": len(ready_items),
            "forward_s": warm_elapsed_s,
            "batch_size": int(len(ready_items)),
        }
    finally:
        collector.push_ready_back(ready_items)


def run_async_training(
    *,
    model: AuxMaskablePPO,
    collector: AsyncReadyCollector,
    total_timesteps: int,
    rollout_transitions: int,
    startup_events: list[dict[str, object]] | None = None,
    print_startup_events: bool = False,
    rollout_log_interval: int = 0,
    status_path: Path | None = None,
    checkpoint_dir: Path | None = None,
    checkpoint_interval_timesteps: int = 0,
    checkpoint_keep_last: int = 3,
    checkpoint_metadata_factory: Callable[[], dict[str, object]] | None = None,
) -> None:
    collector.start()
    warmup_event = warm_policy_on_initial_ready(model, collector)
    if startup_events is None:
        startup_events = []
    startup_events = [*startup_events, warmup_event]
    for event in startup_events or []:
        _emit_json_event(event, enabled=print_startup_events)
    _write_status_payload(
        status_path,
        {
            "phase": "collecting",
            "rollout": 0,
            "timesteps": int(model.num_timesteps),
            "timestamp_unix_s": time.time(),
            "message": "startup complete; waiting for first async rollout collection",
        },
    )
    rollout_index = 0
    checkpoint_interval_timesteps = max(int(checkpoint_interval_timesteps), 0)
    checkpoint_keep_last = max(int(checkpoint_keep_last), 1)
    next_checkpoint_timesteps = 0
    if checkpoint_dir is not None and checkpoint_interval_timesteps > 0:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        next_checkpoint_timesteps = (
            ((int(model.num_timesteps) // checkpoint_interval_timesteps) + 1) * checkpoint_interval_timesteps
        )
    while model.num_timesteps < total_timesteps:
        remaining = max(int(total_timesteps) - int(model.num_timesteps), 0)
        if remaining <= 0:
            break
        current_rollout = min(int(rollout_transitions), remaining)
        if current_rollout <= 0:
            break
        collect_started = time.perf_counter()
        payload = model.collect_rollouts_async(collector, model.rollout_buffer, current_rollout)
        collect_elapsed = time.perf_counter() - collect_started
        transitions = int(payload.get("transitions", current_rollout))
        model.num_timesteps += int(payload.get("transitions", current_rollout))
        model._update_current_progress_remaining(model.num_timesteps, total_timesteps)
        rollout_index += 1
        _write_status_payload(
            status_path,
            {
                "phase": "updating",
                "rollout": rollout_index,
                "timesteps": int(model.num_timesteps),
                "transitions": transitions,
                "collect_elapsed_s": float(collect_elapsed),
                "timestamp_unix_s": time.time(),
            },
        )
        update_started = time.perf_counter()
        model.train()
        update_elapsed = time.perf_counter() - update_started
        total_elapsed = collect_elapsed + update_elapsed
        last_train_timing = getattr(model, "_last_train_timing", None)
        if isinstance(last_train_timing, dict):
            payload.update(last_train_timing)
        payload["phase_timing_s"] = {
            "collect": collect_elapsed,
            "update": update_elapsed,
            "total": total_elapsed,
            "collect_steps_per_s": (float(transitions) / collect_elapsed) if collect_elapsed > 0.0 else 0.0,
            "end_to_end_steps_per_s": (float(transitions) / total_elapsed) if total_elapsed > 0.0 else 0.0,
        }
        model.logger.record("rollout/phase_collect_s", float(collect_elapsed))
        model.logger.record("rollout/phase_update_s", float(update_elapsed))
        model.logger.record("rollout/phase_total_s", float(total_elapsed))
        model.logger.record(
            "rollout/collect_steps_per_s",
            (float(transitions) / collect_elapsed) if collect_elapsed > 0.0 else 0.0,
        )
        model.logger.record(
            "rollout/end_to_end_steps_per_s",
            (float(transitions) / total_elapsed) if total_elapsed > 0.0 else 0.0,
        )
        model.logger.dump(step=int(model.num_timesteps))
        periodic_checkpoint_dir: Path | None = None
        if (
            checkpoint_dir is not None and
            checkpoint_interval_timesteps > 0 and
            checkpoint_metadata_factory is not None and
            int(model.num_timesteps) >= int(next_checkpoint_timesteps)
        ):
            periodic_checkpoint_dir = save_rotating_online_checkpoint(
                model,
                checkpoint_dir,
                timesteps=int(model.num_timesteps),
                metadata=checkpoint_metadata_factory(),
                keep_last=checkpoint_keep_last,
            )
            next_checkpoint_timesteps = (
                ((int(model.num_timesteps) // checkpoint_interval_timesteps) + 1) * checkpoint_interval_timesteps
            )
        _write_status_payload(
            status_path,
            {
                "phase": "collecting",
                "rollout": rollout_index,
                "timesteps": int(model.num_timesteps),
                "transitions": transitions,
                "collect_elapsed_s": float(collect_elapsed),
                "update_elapsed_s": float(update_elapsed),
                "total_elapsed_s": float(total_elapsed),
                "last_periodic_checkpoint_dir": str(periodic_checkpoint_dir) if periodic_checkpoint_dir is not None else None,
                "timestamp_unix_s": time.time(),
            },
        )
        if rollout_log_interval > 0 and rollout_index % rollout_log_interval == 0:
            _emit_json_event(
                _format_rollout_stats(rollout_index, model.num_timesteps, payload),
                enabled=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the search-free omni-attention policy.")
    parser.add_argument("--total-timesteps", type=int, default=20000)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--session-file", type=str, default=None)
    parser.add_argument("--session-files", type=str, default=None)
    parser.add_argument("--collector-mode", type=str, default="async", choices=("async", "sync"))
    parser.add_argument("--no-serial-prewarm", action="store_true")
    parser.add_argument("--combat-sandbox", action="store_true")
    parser.add_argument("--character", type=str, default=None)
    parser.add_argument("--encounter-id", type=str, default=None)
    parser.add_argument("--encounter-pool", type=str, default=None)
    parser.add_argument("--snapshot-pool", type=str, default=None)
    parser.add_argument("--snapshot-curated-subset", type=str, default=None)
    parser.add_argument(
        "--snapshot-sample-mode",
        type=str,
        default="encounter_balanced",
        choices=("row_uniform", "encounter_balanced", "tier_weighted_encounter_balanced"),
    )
    parser.add_argument("--snapshot-encounter-tiers", type=str, default=None)
    parser.add_argument("--snapshot-tier-weights", type=str, default=None)
    parser.add_argument(
        "--snapshot-starter-early-boost",
        type=float,
        default=0.0,
        help=(
            "Probability of drawing a starter-early deck snapshot "
            "(floor<=%d AND deck_size<=%d) for each reset, bypassing the "
            "normal sample mode. 0 disables (default); 0.2-0.3 is a sane "
            "starting point when bridging sandbox->full_run."
        ) % (3, 13),
    )
    parser.add_argument("--snapshot-max-rows", type=int, default=None)
    parser.add_argument(
        "--use-sim",
        action="store_true",
        default=False,
        help=(
            "Drive training against the frankqwang/sts2-ai HeadlessSim "
            "(decompiled-game headless C# simulator) instead of live Godot "
            "instances. Bypasses session-file plumbing. 100-1000x faster; "
            "no game crashes. See sts2_env/headless_sim_bridge_client.py."
        ),
    )
    parser.add_argument(
        "--sim-exe-path",
        type=str,
        default=None,
        help="Override HeadlessSim exe path (defaults to third_party/sts2-ai build).",
    )
    parser.add_argument(
        "--perf-stats-interval-steps",
        type=int,
        default=0,
        help="When >0 and --use-sim, every N successful env steps pull bridge.perf_stats() and append to {log_dir}/perf_stats.jsonl for hang diagnosis.",
    )
    parser.add_argument(
        "--encode-pool-workers",
        type=int,
        default=0,
        help=(
            "Offload observation encoding to a multiprocessing pool of N "
            "workers. 0 disables (in-thread encoding, GIL-serialized). A "
            "positive N sidesteps GIL contention when n_envs>1 — typical "
            "win is 3-7x on the encode step. Recommended: set equal to "
            "--n-envs. Only effective for --collector-mode async."
        ),
    )
    parser.add_argument("--print-startup-events", action="store_true")
    parser.add_argument("--stdout-rollout-log-interval", type=int, default=0)
    parser.add_argument("--logger-stdout", action="store_true")
    parser.add_argument("--status-file", type=str, default=None)
    parser.add_argument("--log-dir", type=str, default="logs_attention")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints_attention")
    parser.add_argument(
        "--checkpoint-interval-timesteps",
        type=int,
        default=10000,
        help="Save a rotating step_* checkpoint at or after every N timesteps; set 0 to disable.",
    )
    parser.add_argument(
        "--checkpoint-keep-last",
        type=int,
        default=3,
        help="Keep only the most recent N periodic step_* checkpoints under --checkpoint-dir.",
    )
    parser.add_argument("--init-checkpoint", type=str, default=None)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--aux-objective-coef", type=float, default=0.25)
    parser.add_argument("--aux-transition-coef", type=float, default=0.10)
    parser.add_argument("--aux-trait-coef", type=float, default=0.10)
    parser.add_argument("--aux-build-coef", type=float, default=0.10)
    parser.add_argument("--aux-selection-coef", type=float, default=0.10)
    parser.add_argument("--aux-route-coef", type=float, default=0.10)
    parser.add_argument("--aux-enemy-state-coef", type=float, default=0.10)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--text-device", type=str, default="auto")
    parser.add_argument(
        "--no-text",
        action="store_true",
        help="Disable text features in observations. Attention runtime stays unchanged; text slices are zeroed.",
    )
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--amp-dtype", type=str, default="bf16", choices=("bf16", "bfloat16"))
    parser.add_argument("--reset-timeout-ms", type=int, default=60000)
    parser.add_argument("--step-timeout-ms", type=int, default=20000)
    parser.add_argument(
        "--stuck-watchdog-steps",
        type=int,
        default=400,
        help=(
            "Truncate full-run episode when the (phase, floor, combat_round, "
            "enemy_hp_total, player_hp) fingerprint stays constant for this "
            "many consecutive steps. 0 disables. Default 400 catches the "
            "sim-training pathology where 23%% of episodes ran 1000-6000 "
            "steps on the same floor without progressing."
        ),
    )
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--ffn-dim", type=int, default=1024)
    parser.add_argument("--world-layers", type=int, default=6)
    parser.add_argument("--local-layers", type=int, default=1)
    parser.add_argument("--decoder-layers", type=int, default=2)
    parser.add_argument("--candidate-set-layers", type=int, default=1)
    parser.add_argument("--world-bank-top-k", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.0)
    args = parser.parse_args()

    use_text = not bool(args.no_text)
    text_device = resolve_text_device(args.text_device, args.device)
    session_files = resolve_training_session_files(
        n_envs=args.n_envs,
        session_file=args.session_file,
        session_files=parse_csv(args.session_files),
    )
    encounter_pool = parse_csv(args.encounter_pool)
    snapshot_pool = resolve_snapshot_pool(args)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    if snapshot_pool is not None:
        _emit_json_event(
            {
                "phase": "snapshot_pool_summary",
                "snapshot_pool_root": args.snapshot_pool,
                "snapshot_curated_subset": getattr(args, "snapshot_curated_subset", None),
                "snapshot_sample_mode": getattr(args, "snapshot_sample_mode", None),
                "snapshot_encounter_tiers": parse_csv(getattr(args, "snapshot_encounter_tiers", None)),
                "snapshot_tier_weights": parse_weight_map(getattr(args, "snapshot_tier_weights", None)),
                "snapshot_starter_early_boost": float(getattr(args, "snapshot_starter_early_boost", 0.0) or 0.0),
                "summary": snapshot_pool.summary(),
            },
            enabled=bool(args.print_startup_events),
        )

    encode_pool = None
    encode_pool_workers = int(getattr(args, "encode_pool_workers", 0) or 0)
    if encode_pool_workers > 0:
        from sts2_env.encode_pool import build_encode_pool
        encode_pool = build_encode_pool(
            max_workers=encode_pool_workers,
            encoder_kwargs={"use_text": use_text, "text_device": text_device},
        )

    env_fns = [
        build_env_factory(
            env_index=index,
            session_file=session_files[index],
            use_text=use_text,
            text_device=text_device,
            log_dir=args.log_dir,
            combat_sandbox=args.combat_sandbox,
            character=args.character,
            encounter_id=args.encounter_id,
            encounter_pool=encounter_pool,
            snapshot_pool=snapshot_pool,
            reset_timeout_ms=args.reset_timeout_ms,
            step_timeout_ms=args.step_timeout_ms,
            use_sim=bool(getattr(args, "use_sim", False)),
            sim_exe_path=getattr(args, "sim_exe_path", None),
            perf_stats_log_path=str(Path(args.log_dir) / "perf_stats.jsonl"),
            perf_stats_interval_steps=int(getattr(args, "perf_stats_interval_steps", 0) or 0),
            encode_pool=encode_pool,
            stuck_watchdog_steps=int(getattr(args, "stuck_watchdog_steps", 400) or 0),
        )
        for index in range(args.n_envs)
    ]
    collector_mode = str(args.collector_mode or "async").strip().lower()
    obs_encoder = WorldTokenObservationEncoder(use_text=use_text, text_device=text_device)
    if collector_mode == "async":
        spec_env_fns = [lambda obs_space=obs_encoder.obs_space: ObservationSpecEnv(obs_space) for _ in range(args.n_envs)]
        env = DummyVecEnv(spec_env_fns)
    else:
        env = DummyVecEnv(env_fns) if len(env_fns) == 1 else SubprocVecEnv(env_fns)

    policy_kwargs = build_policy_kwargs(args)
    model = AuxMaskablePPO(
        STS2OmniAttentionPolicy,
        env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        aux_objective_coef=args.aux_objective_coef,
        aux_transition_coef=args.aux_transition_coef,
        aux_trait_coef=args.aux_trait_coef,
        aux_build_coef=args.aux_build_coef,
        aux_selection_coef=args.aux_selection_coef,
        aux_route_coef=args.aux_route_coef,
        aux_enemy_state_coef=args.aux_enemy_state_coef,
        amp=args.amp,
        amp_dtype=args.amp_dtype,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=args.log_dir,
        device=args.device,
    )
    logger_outputs = ["csv"]
    if args.logger_stdout:
        logger_outputs.insert(0, "stdout")
    if args.log_dir:
        logger_outputs.append("tensorboard")
    model.set_logger(configure(str(Path(args.log_dir) / "sb3_async"), logger_outputs))

    if args.init_checkpoint:
        load_online_policy_state_dict(model, args.init_checkpoint, device=args.device, strict=False)

    started = time.time()
    checkpoint_metadata_factory = lambda: build_checkpoint_metadata(
        model=model,
        args=args,
        policy_kwargs=policy_kwargs,
        collector_mode=collector_mode,
        use_text=use_text,
        text_device=text_device,
        started_at_unix_s=started,
    )
    if collector_mode == "async":
        model.rollout_buffer = AsyncAuxMaskableDictRolloutBuffer(
            buffer_size=int(args.n_steps) * int(args.n_envs),
            observation_space=model.observation_space,
            action_space=model.action_space,
            device=model.device,
            gamma=model.gamma,
            gae_lambda=model.gae_lambda,
            env_count=int(args.n_envs),
        )
        startup_events: list[dict[str, object]] = []
        prewarmed_slots: dict[int, PrewarmedEnvSlot] = {}
        if not args.no_serial_prewarm:
            prewarmed_slots, prewarm_summary = serial_prewarm_async_envs(env_fns)
            startup_events.append(prewarm_summary)
        collector = AsyncReadyCollector(
            env_fns,
            prewarmed_slots=prewarmed_slots,
            event_log_path=str(Path(args.log_dir) / "reset_events.jsonl") if args.log_dir else None,
        )
        try:
            status_path = Path(args.status_file) if args.status_file else (Path(args.log_dir) / "async_status.json")
            run_async_training(
                model=model,
                collector=collector,
                total_timesteps=int(args.total_timesteps),
                rollout_transitions=int(args.n_steps) * int(args.n_envs),
                startup_events=startup_events,
                print_startup_events=bool(args.print_startup_events),
                rollout_log_interval=max(int(args.stdout_rollout_log_interval), 0),
                status_path=status_path,
                checkpoint_dir=Path(args.checkpoint_dir),
                checkpoint_interval_timesteps=max(int(args.checkpoint_interval_timesteps), 0),
                checkpoint_keep_last=max(int(args.checkpoint_keep_last), 1),
                checkpoint_metadata_factory=checkpoint_metadata_factory,
            )
        finally:
            collector.close()
    else:
        model.learn(total_timesteps=int(args.total_timesteps), progress_bar=False)
    elapsed_s = time.time() - started
    env.close()
    if encode_pool is not None:
        encode_pool.shutdown(wait=True, cancel_futures=True)

    metadata = checkpoint_metadata_factory()
    output_dir = Path(args.checkpoint_dir) / "final"
    save_online_checkpoint(model, output_dir, metadata=metadata)
    print(json.dumps({"checkpoint_dir": str(output_dir), "elapsed_s": elapsed_s, "timesteps": int(model.num_timesteps)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
