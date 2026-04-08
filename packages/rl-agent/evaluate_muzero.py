"""Evaluate a trained MuZero checkpoint in live STS2 environments.

Supports full-run evaluation through the existing bridge-backed env wrapper.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sts2_env.env_v2 import SlayTheSpire2EnvV2
from sts2_env.mcts import MCTS
from sts2_env.muzero_model import MuZeroNetwork
from sts2_env.observation_v2 import DictObservationEncoder


def load_muzero_network(checkpoint_dir: str | Path, device: str) -> MuZeroNetwork:
    checkpoint_path = Path(checkpoint_dir)
    network_path = checkpoint_path / "network.pt"
    if not network_path.exists():
        raise FileNotFoundError(f"Missing network.pt in checkpoint: {checkpoint_path}")

    network = MuZeroNetwork()
    network.load_state_dict(torch.load(network_path, map_location=device))
    network.to(device)
    network.eval()
    return network


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
) -> None:
    obs_encoder = DictObservationEncoder(use_text=False)
    env = SlayTheSpire2EnvV2(
        session_file=session_file,
        character=character,
        defensive_buffs=defensive_buffs,
        obs_encoder=obs_encoder,
        include_debug_info=True,
        render_mode="human",
    )

    network = load_muzero_network(checkpoint_dir, device=device)
    mcts = MCTS(num_simulations=num_simulations)

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
    parser.add_argument("--character", type=str, default="ironclad")
    parser.add_argument("--defensive-buffs", action="store_true")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--num-simulations", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate_full_run(
        args.checkpoint,
        session_file=args.session_file,
        character=args.character,
        defensive_buffs=args.defensive_buffs,
        max_steps=args.max_steps,
        num_simulations=args.num_simulations,
        temperature=args.temperature,
        device=args.device,
    )


if __name__ == "__main__":
    main()
