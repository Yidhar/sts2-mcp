"""Evaluate a trained MuZero / token-memory checkpoint in live STS2 environments.

Preferred entrypoint:
    python -m muzero.evaluate CHECKPOINT_DIR [--flags...]

Invoke with ``python -m muzero.evaluate``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from sts2_env.env_v2 import SlayTheSpire2EnvV2
from muzero.sts2_env.mcts import MCTS
from muzero.training.env_factory import create_environment_backend
from muzero.sts2_env.muzero_model import MuZeroNetwork
from muzero.sts2_env.planner_memory_profile import (
    VALID_PLANNER_MEMORY_PROFILES,
    apply_planner_memory_profile_to_network,
)
from sts2_env.observation_v2 import DictObservationEncoder
from sts2_env.observation_v3 import WorldTokenObservationEncoder
from sts2_env.path_utils import normalize_path_str, resolve_torch_device, running_in_wsl
from sts2_rl.artifacts import resolve_external_input_path


def load_muzero_network(
    checkpoint_dir: str | Path,
    device: str,
    *,
    planner_memory_profile: str = "eval",
) -> tuple[MuZeroNetwork, dict[str, object]]:
    checkpoint_path = resolve_external_input_path(checkpoint_dir)
    network_path = checkpoint_path / "network.pt"
    metadata_path = checkpoint_path / "metadata.json"
    if not network_path.exists():
        raise FileNotFoundError(f"Missing network.pt in checkpoint: {checkpoint_path}")

    metadata: dict[str, object] = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    network_kwargs = metadata.get("network") if isinstance(metadata.get("network"), dict) else {}
    network = MuZeroNetwork(**network_kwargs)
    network.load_state_dict(torch.load(network_path, map_location=device))
    network.to(device)
    network.eval()
    profile_settings = apply_planner_memory_profile_to_network(
        network,
        planner_memory_profile,
    )
    profile_report: dict[str, object] = {
        "planner_memory_profile": profile_settings.profile,
        "action_rollout_buckets": tuple(int(bucket) for bucket in getattr(network, "action_rollout_buckets", ())),
        "action_rollout_chunk_size": int(getattr(network, "action_rollout_chunk_size", 0) or 0),
    }
    return network, profile_report


def evaluate_full_run(
    checkpoint_dir: str | Path,
    *,
    session_file: str | None,
    character: str | None,
    defensive_buffs: bool,
    max_steps: int,
    num_simulations: int,
    temperature: float,
    device: str,
    disable_root_bias: bool,
    disable_semantic_rollout: bool,
    obs_mode: str | None,
    planner_memory_profile: str,
    environment_backend: str = "live",
    sim_exe_path: str | None = None,
) -> None:
    network, profile_report = load_muzero_network(
        checkpoint_dir,
        device=device,
        planner_memory_profile=planner_memory_profile,
    )
    resolved_obs_mode = str(obs_mode or getattr(network, "obs_mode", "dense_v2")).strip().lower()
    if resolved_obs_mode == "token_v3":
        obs_encoder = WorldTokenObservationEncoder(use_text=False)
    else:
        obs_encoder = DictObservationEncoder(use_text=False)
    backend = create_environment_backend(
        kind=environment_backend,
        session_file=session_file,
        sim_exe_path=sim_exe_path,
    )
    env = SlayTheSpire2EnvV2(
        session_file=session_file,
        character=character,
        defensive_buffs=defensive_buffs,
        obs_encoder=obs_encoder,
        include_debug_info=True,
        render_mode="human",
        backend=backend,
    )
    mcts = MCTS(num_simulations=num_simulations)
    mcts.set_training_step(mcts.root_bias_decay_steps)
    mcts.set_root_bias_enabled(not disable_root_bias)
    mcts.set_semantic_rollout_enabled(not disable_semantic_rollout)

    obs, info = env.reset()
    total_reward = 0.0

    print(
        json.dumps(
            {
                "checkpoint": str(checkpoint_dir),
                "character": character,
                "defensive_buffs": defensive_buffs,
                "num_simulations": num_simulations,
                "temperature": temperature,
                "max_steps": max_steps,
                "device": device,
                "disable_root_bias": disable_root_bias,
                "disable_semantic_rollout": disable_semantic_rollout,
                **profile_report,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    for step_idx in range(max_steps):
        if int(info.get("legal_action_count", 0)) <= 0:
            recovered, obs, info = env.recover_actionable_state()
            print(
                json.dumps(
                    {
                        "step": step_idx,
                        "recovered_empty_action_window": bool(recovered),
                        "phase": info.get("phase"),
                        "legal_action_count": int(info.get("legal_action_count", 0)),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if not recovered or int(info.get("legal_action_count", 0)) <= 0:
                break

        action_mask = info.get("action_mask")
        if action_mask is None:
            raise RuntimeError("Env info is missing action_mask.")

        action_idx, action_probs = mcts.run(
            network,
            obs,
            action_mask,
            temperature=temperature,
        )

        legal_actions = info.get("legal_actions") or []
        chosen_action = legal_actions[action_idx] if action_idx < len(legal_actions) else None
        print(
            json.dumps(
                {
                    "step": step_idx,
                    "phase": info.get("phase"),
                    "action_index": int(action_idx),
                    "action_id": chosen_action.get("action_id") if isinstance(chosen_action, dict) else None,
                    "kind": chosen_action.get("kind") if isinstance(chosen_action, dict) else None,
                    "top_prob": float(action_probs[action_idx]) if action_idx < len(action_probs) else None,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

        obs, reward, terminated, truncated, info = env.step(int(action_idx))
        total_reward += float(reward)
        print(
            json.dumps(
                {
                    "step_result": step_idx,
                    "reward": float(reward),
                    "total_reward": float(total_reward),
                    "phase_after": info.get("phase"),
                    "legal_action_count": int(info.get("legal_action_count", 0)),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

        if terminated or truncated:
            break

    print(
        json.dumps(
            {
                "done": True,
                "steps": step_idx + 1,
                "total_reward": float(total_reward),
                "final_phase": info.get("phase"),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MuZero checkpoints on live STS2.")
    parser.add_argument("checkpoint", type=str, help="Checkpoint directory containing network.pt")
    parser.add_argument("--session-file", type=str, default=None)
    parser.add_argument("--environment-backend", choices=("live", "headless"), default="live")
    parser.add_argument("--sim-exe-path", type=str, default=None)
    parser.add_argument("--character", type=str, default="ironclad")
    parser.add_argument("--defensive-buffs", action="store_true")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--num-simulations", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--disable-root-bias", action="store_true")
    parser.add_argument("--disable-semantic-rollout", action="store_true")
    parser.add_argument("--obs-mode", type=str, default=None, choices=["dense_v2", "token_v3"])
    parser.add_argument(
        "--planner-memory-profile",
        type=str,
        default="eval",
        choices=VALID_PLANNER_MEMORY_PROFILES,
        help=(
            "Override checkpoint planner memory knobs after loading. "
            "'eval' is the safe stronger default; use 'max' for full eval-time planning."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.checkpoint = normalize_path_str(args.checkpoint) or args.checkpoint
    args.session_file = normalize_path_str(args.session_file)
    args.sim_exe_path = normalize_path_str(args.sim_exe_path)
    args.device = resolve_torch_device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested GPU evaluation but torch.cuda.is_available() is False.")
    if running_in_wsl():
        print(
            json.dumps(
                {
                    "wsl": True,
                    "bridge_session_file": args.session_file,
                    "bridge_base_url_override": os.environ.get("STS2_BRIDGE_BASE_URL"),
                    "device": args.device,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    evaluate_full_run(
        args.checkpoint,
        session_file=args.session_file,
        character=args.character,
        defensive_buffs=args.defensive_buffs,
        max_steps=args.max_steps,
        num_simulations=args.num_simulations,
        temperature=args.temperature,
        device=args.device,
        disable_root_bias=args.disable_root_bias,
        disable_semantic_rollout=args.disable_semantic_rollout,
        obs_mode=args.obs_mode,
        planner_memory_profile=args.planner_memory_profile,
        environment_backend=args.environment_backend,
        sim_exe_path=args.sim_exe_path,
    )


if __name__ == "__main__":
    main()
