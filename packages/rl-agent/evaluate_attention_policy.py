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
    parser.add_argument(
        "--no-text",
        action="store_true",
        help="Force text features off during evaluation, regardless of checkpoint metadata.",
    )
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--use-sim", action="store_true", help="Drive the frankqwang/sts2-ai HeadlessSim subprocess instead of a live game bridge.")
    parser.add_argument("--sim-exe-path", type=str, default=None, help="Override HeadlessSim exe path when --use-sim.")
    args = parser.parse_args()

    metadata = load_online_checkpoint_metadata(args.checkpoint_dir)
    use_text = bool(metadata.get("use_text", True)) and not bool(args.no_text)
    text_device = resolve_text_device(args.text_device or metadata.get("text_device"), args.device)
    obs_encoder = WorldTokenObservationEncoder(use_text=use_text, text_device=text_device)
    sim_bridge = None
    if args.use_sim:
        from sts2_env.headless_sim_bridge_client import HeadlessSimBridgeClient
        sim_bridge = HeadlessSimBridgeClient(exe_path=args.sim_exe_path)
    if args.combat_sandbox:
        env = CombatSandboxEnv(
            session_file=args.session_file,
            character=args.character,
            encounter_id=args.encounter_id,
            obs_encoder=obs_encoder,
            include_debug_info=True,
            bridge=sim_bridge,
        )
    else:
        env = SlayTheSpire2EnvV2(
            session_file=args.session_file,
            character=args.character,
            obs_encoder=obs_encoder,
            include_debug_info=True,
            bridge=sim_bridge,
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
    episode_final_hp: list[float] = []
    episode_max_hp: list[float] = []
    episode_max_floor: list[float] = []
    episode_wins: list[bool] = []

    for episode in range(args.episodes):
        obs, info = env.reset()
        terminated = False
        truncated = False
        reward_sum = 0.0
        steps = 0
        max_floor_seen = 0.0
        last_transition_state: dict | None = None
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
            raw_obs = info.get("raw_obs") if isinstance(info, dict) else None
            if isinstance(raw_obs, dict):
                # sim/bridge shape: players[0].creature.current_hp + run.act_floor
                players_list = raw_obs.get("players") if isinstance(raw_obs.get("players"), list) else []
                player0 = players_list[0] if players_list and isinstance(players_list[0], dict) else {}
                creature = player0.get("creature", {}) if isinstance(player0.get("creature"), dict) else {}
                hp = creature.get("current_hp")
                if hp is None:
                    hp = creature.get("hp")
                if hp is None:
                    hp = player0.get("hp")
                max_hp = creature.get("max_hp")
                if max_hp is None:
                    max_hp = player0.get("max_hp")
                run = raw_obs.get("run", {}) if isinstance(raw_obs.get("run"), dict) else {}
                floor = run.get("floor")
                if floor is None:
                    floor = run.get("act_floor")
                if floor is None:
                    floor = run.get("total_floor")
                if hp is not None and max_hp:
                    last_transition_state = {"player": {"hp": float(hp), "max_hp": float(max_hp)}}
                if isinstance(floor, (int, float)):
                    max_floor_seen = max(max_floor_seen, float(floor))
            reward_sum += float(reward)
            steps += 1
        episode_rewards.append(reward_sum)
        episode_lengths.append(steps)
        final_hp = 0.0
        max_hp = 0.0
        if last_transition_state:
            player = last_transition_state.get("player", {}) if isinstance(last_transition_state.get("player"), dict) else {}
            final_hp = float(player.get("hp") or 0.0)
            max_hp = float(player.get("max_hp") or 0.0)
        won = bool(terminated and final_hp > 0.0)
        episode_final_hp.append(final_hp)
        episode_max_hp.append(max_hp)
        episode_max_floor.append(max_floor_seen)
        episode_wins.append(won)
        print(
            f"[eval] episode={episode + 1} reward={reward_sum:.3f} steps={steps} "
            f"hp={final_hp:.0f}/{max_hp:.0f} floor={max_floor_seen:.0f} "
            f"terminated={terminated} truncated={truncated} won={won}"
        )

    from collections import Counter
    floor_counter = Counter(int(f) for f in episode_max_floor)
    floor_dist = {str(k): int(v) for k, v in sorted(floor_counter.items())}
    hp_losses = [
        (mx - hp) for mx, hp in zip(episode_max_hp, episode_final_hp) if mx > 0
    ]
    summary = {
        "episodes": args.episodes,
        "win_rate": (sum(1 for w in episode_wins if w) / len(episode_wins)) if episode_wins else 0.0,
        "avg_hp_loss": (statistics.mean(hp_losses) if hp_losses else 0.0),
        "avg_final_hp": (statistics.mean(episode_final_hp) if episode_final_hp else 0.0),
        "avg_max_floor": (statistics.mean(episode_max_floor) if episode_max_floor else 0.0),
        "max_floor_reached": (max(episode_max_floor) if episode_max_floor else 0.0),
        "floor_distribution": floor_dist,
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
