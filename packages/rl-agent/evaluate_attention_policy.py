"""Evaluate a saved omni-attention policy checkpoint."""

from __future__ import annotations

import argparse
import json
import statistics
import time

import numpy as np
import torch
from sb3_contrib.common.wrappers import ActionMasker

from sts2_env.checkpoint import load_online_checkpoint, load_online_checkpoint_metadata
from sts2_env.combat_env import CombatSandboxEnv
from sts2_env.env_v2 import SlayTheSpire2EnvV2
from sts2_env.observation_v3 import WorldTokenObservationEncoder


def mask_fn(env):
    return env.unwrapped.action_masks()


def resolve_text_device(text_device: str | None, fallback_device: str | None = None) -> str:
    candidate = str(text_device or "").strip().lower()
    if candidate and candidate != "auto":
        return str(text_device).strip()

    fallback = str(fallback_device or "").strip()
    if fallback and fallback.lower() != "auto":
        return fallback

    return "cuda" if torch.cuda.is_available() else "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the omni-attention policy.")
    parser.add_argument("checkpoint_dir", type=str)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--session-file", type=str, default=None)
    parser.add_argument("--combat-sandbox", action="store_true")
    parser.add_argument("--character", type=str, default=None)
    parser.add_argument("--encounter-id", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--text-device", type=str, default=None)
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()

    metadata = load_online_checkpoint_metadata(args.checkpoint_dir)
    use_text = bool(metadata.get("use_text", True))
    text_device = resolve_text_device(args.text_device or metadata.get("text_device"), args.device)
    obs_encoder = WorldTokenObservationEncoder(use_text=use_text, text_device=text_device)
    if args.combat_sandbox:
        env = CombatSandboxEnv(
            session_file=args.session_file,
            character=args.character,
            encounter_id=args.encounter_id,
            obs_encoder=obs_encoder,
            include_debug_info=True,
        )
    else:
        env = SlayTheSpire2EnvV2(
            session_file=args.session_file,
            character=args.character,
            obs_encoder=obs_encoder,
            include_debug_info=True,
        )
    env = ActionMasker(env, mask_fn)

    model, _loaded_metadata = load_online_checkpoint(args.checkpoint_dir, env, device=args.device)

    episode_rewards: list[float] = []
    episode_lengths: list[int] = []
    decision_latencies_ms: list[float] = []
    step_total_ms: list[float] = []
    step_after_wait_ms: list[float] = []
    python_obs_encode_ms: list[float] = []
    python_total_ms: list[float] = []

    for episode in range(args.episodes):
        obs, info = env.reset()
        terminated = False
        truncated = False
        reward_sum = 0.0
        steps = 0
        while not terminated and not truncated:
            start = time.time()
            action, _state = model.predict(obs, action_masks=env.unwrapped.action_masks(), deterministic=args.deterministic)
            decision_latencies_ms.append((time.time() - start) * 1000.0)
            obs, reward, terminated, truncated, info = env.step(int(action))
            bridge_info = info.get("bridge_info") if isinstance(info, dict) else None
            step_timing = bridge_info.get("step_timing_ms") if isinstance(bridge_info, dict) else None
            python_timing = info.get("python_timing_ms") if isinstance(info, dict) else None
            if isinstance(step_timing, dict):
                if step_timing.get("total") is not None:
                    step_total_ms.append(float(step_timing["total"]))
                if step_timing.get("after_wait") is not None:
                    step_after_wait_ms.append(float(step_timing["after_wait"]))
            if isinstance(python_timing, dict):
                if python_timing.get("obs_encode") is not None:
                    python_obs_encode_ms.append(float(python_timing["obs_encode"]))
                if python_timing.get("total") is not None:
                    python_total_ms.append(float(python_timing["total"]))
            reward_sum += float(reward)
            steps += 1
        episode_rewards.append(reward_sum)
        episode_lengths.append(steps)
        print(f"[eval] episode={episode + 1} reward={reward_sum:.3f} steps={steps} terminated={terminated} truncated={truncated}")

    summary = {
        "episodes": args.episodes,
        "avg_reward": statistics.mean(episode_rewards) if episode_rewards else 0.0,
        "avg_length": statistics.mean(episode_lengths) if episode_lengths else 0.0,
        "median_latency_ms": statistics.median(decision_latencies_ms) if decision_latencies_ms else 0.0,
        "rollout_timing_ms": {
            "step.total.p50": statistics.median(step_total_ms) if step_total_ms else 0.0,
            "step.total.p95": float(np.percentile(step_total_ms, 95)) if step_total_ms else 0.0,
            "step.after_wait.p50": statistics.median(step_after_wait_ms) if step_after_wait_ms else 0.0,
            "step.after_wait.p95": float(np.percentile(step_after_wait_ms, 95)) if step_after_wait_ms else 0.0,
            "python.obs_encode.p50": statistics.median(python_obs_encode_ms) if python_obs_encode_ms else 0.0,
            "python.obs_encode.p95": float(np.percentile(python_obs_encode_ms, 95)) if python_obs_encode_ms else 0.0,
            "python.total.p50": statistics.median(python_total_ms) if python_total_ms else 0.0,
            "python.total.p95": float(np.percentile(python_total_ms, 95)) if python_total_ms else 0.0,
        },
        "metadata": metadata,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
