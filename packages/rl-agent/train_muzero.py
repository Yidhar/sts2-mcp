"""MuZero training for Slay the Spire 2 combat scenarios.

Usage:
    python train_muzero.py --total-timesteps 100000 --combat-sandbox \\
        --encounter-pool "ENCOUNTER.SLIMES_WEAK,ENCOUNTER.SHRINKER_BEETLE_WEAK"

This implements self-play MCTS + replay buffer training for the MuZero network.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import signal
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter

from combat_snapshot_dataset import CombatSnapshotPool, load_combat_snapshot_rows
from launcher import get_session_files as get_default_multi_session_files
from sts2_env.combat_env import CombatSandboxEnv
from sts2_env.env_v2 import SlayTheSpire2EnvV2
from sts2_env.mcts import MCTS
from sts2_env.muzero_buffer import GameTrajectory, MuZeroReplayBuffer
from sts2_env.muzero_model import MuZeroNetwork, scalar_to_support
from sts2_env.observation_v2 import DictObservationEncoder, MAX_ACTIONS


DEFAULT_COMBAT_SANDBOX_TRAIN_POOL = ",".join([
    "ENCOUNTER.SLIMES_WEAK",
    "ENCOUNTER.SHRINKER_BEETLE_WEAK",
    "ENCOUNTER.FUZZY_WURM_CRAWLER_WEAK",
    "ENCOUNTER.NIBBITS_WEAK",
])


def mask_fn(env):
    """Extract action mask from environment."""
    return env.unwrapped.action_masks()


def parse_encounter_pool(raw: str | None) -> list[str]:
    """Parse comma-separated encounter pool."""
    if not raw:
        return []
    return [entry.strip() for entry in raw.split(",") if entry.strip()]


def parse_session_files(raw: str | None) -> list[str]:
    """Parse comma-separated session files."""
    if not raw:
        return []
    return [entry.strip() for entry in raw.split(",") if entry.strip()]


def resolve_training_session_files(
    *,
    n_envs: int,
    session_file: str | None,
    session_files: list[str],
) -> list[str | None]:
    """Resolve session files for parallel environments."""
    if n_envs < 1:
        raise ValueError("--n-envs must be >= 1")

    if session_file and session_files:
        raise ValueError("Use either --session-file or --session-files, not both.")

    if session_files:
        if len(session_files) != n_envs:
            raise ValueError(
                f"--session-files count ({len(session_files)}) must match --n-envs ({n_envs})."
            )
        return session_files

    if n_envs == 1:
        return [session_file]

    return get_default_multi_session_files(n_envs)


def build_train_env(
    *,
    env_index: int,
    session_file: str | None,
    combat_sandbox: bool,
    character: str | None,
    defensive_buffs: bool,
    encounter_id: str | None,
    encounter_pool: list[str],
    snapshot_pool: CombatSnapshotPool | None,
    reset_timeout_ms: int,
    step_timeout_ms: int,
) -> gym.Env:
    """Build a single training environment."""
    obs_encoder = DictObservationEncoder(use_text=False)

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
        )
        return env

    env = SlayTheSpire2EnvV2(
        session_file=session_file,
        character=character,
        defensive_buffs=defensive_buffs,
        reset_timeout_ms=reset_timeout_ms,
        step_timeout_ms=step_timeout_ms,
        obs_encoder=obs_encoder,
    )
    return env


def dict_obs_to_torch(obs: dict[str, Any], device: str = "cpu") -> dict[str, torch.Tensor]:
    """Convert dict observation to torch tensors with batch dimension 1."""
    torch_obs = {}
    for key, value in obs.items():
        if isinstance(value, np.ndarray):
            tensor = torch.from_numpy(value).to(device)
        elif isinstance(value, torch.Tensor):
            tensor = value.to(device)
        else:
            tensor = torch.tensor(value, device=device)

        # Add batch dimension if needed
        if tensor.dim() == 0:
            tensor = tensor.unsqueeze(0)
        elif tensor.dim() > 0 and tensor.shape[0] != 1:
            tensor = tensor.unsqueeze(0)

        torch_obs[key] = tensor

    return torch_obs


def load_resume_checkpoint(
    resume_from: str | Path,
    *,
    network: MuZeroNetwork,
    optimizer: optim.Optimizer,
    buffer: MuZeroReplayBuffer,
    device: str,
) -> dict[str, Any]:
    """Load network/optimizer/(optional) replay buffer from a MuZero checkpoint directory."""
    checkpoint_path = Path(resume_from)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Resume checkpoint path not found: {checkpoint_path}")

    network_path = checkpoint_path / "network.pt"
    optimizer_path = checkpoint_path / "optimizer.pt"
    metadata_path = checkpoint_path / "metadata.json"
    replay_buffer_path = checkpoint_path / "replay_buffer.pkl"

    if not network_path.exists():
        raise FileNotFoundError(f"Missing network.pt in checkpoint: {checkpoint_path}")
    if not optimizer_path.exists():
        raise FileNotFoundError(f"Missing optimizer.pt in checkpoint: {checkpoint_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata.json in checkpoint: {checkpoint_path}")

    network_state = torch.load(network_path, map_location=device)
    optimizer_state = torch.load(optimizer_path, map_location=device)
    network.load_state_dict(network_state)
    optimizer.load_state_dict(optimizer_state)

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    if replay_buffer_path.exists():
        with replay_buffer_path.open("rb") as handle:
            replay_buffer_state = pickle.load(handle)
        buffer.load_state_dict(replay_buffer_state)
    else:
        print(f"[resume] replay_buffer.pkl not found in {checkpoint_path}; resuming with empty buffer.")

    return metadata


class MuZeroTrainer:
    """Orchestrates MuZero self-play training."""

    def __init__(
        self,
        network: MuZeroNetwork,
        mcts: MCTS,
        buffer: MuZeroReplayBuffer,
        env: CombatSandboxEnv,
        optimizer: optim.Optimizer,
        device: str = "cpu",
        discount: float = 0.997,
        n_step_return: int = 10,
        max_grad_norm: float = 1.0,
        log_dir: str = "runs",
        checkpoint_dir: str = "checkpoints",
    ):
        """Initialize trainer.

        Args:
            network: MuZeroNetwork instance.
            mcts: MCTS instance for action selection.
            buffer: Replay buffer.
            env: Training environment.
            optimizer: Adam optimizer for network.
            device: Device for training.
            discount: Discount factor.
            n_step_return: Steps for bootstrapped returns.
            max_grad_norm: Max gradient norm for clipping.
            log_dir: TensorBoard log directory.
            checkpoint_dir: Checkpoint save directory.
        """
        self.network = network
        self.mcts = mcts
        self.buffer = buffer
        self.env = env
        self.optimizer = optimizer
        self.device = device
        self.discount = discount
        self.n_step_return = n_step_return
        self.max_grad_norm = max_grad_norm

        Path(log_dir).mkdir(parents=True, exist_ok=True)
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir

        self.writer = SummaryWriter(log_dir=log_dir)
        self.total_steps = 0
        self.episode_count = 0

    def compute_temperature(self, step: int, total_steps: int) -> float:
        """Compute temperature schedule: linear decay from 1.0 to 0.1."""
        progress = min(step / max(total_steps, 1), 1.0)
        return 1.0 - progress * 0.9

    def self_play_episode(self, temperature: float = 1.0) -> tuple[float, int]:
        """Run one self-play episode using MCTS action selection.

        Args:
            temperature: Temperature for action selection.

        Returns:
            (episode_reward, episode_length)
        """
        obs, info = self.env.reset()
        trajectory = GameTrajectory()

        episode_reward = 0.0
        episode_length = 0

        while True:
            action_mask = info.get("action_mask", np.ones(MAX_ACTIONS, dtype=bool))

            # Run MCTS to get action and policy
            with torch.no_grad():
                action_idx, search_policy = self.mcts.run(
                    self.network,
                    obs,
                    action_mask,
                    temperature=temperature,
                )

            # Get initial inference for root value
            obs_batch = dict_obs_to_torch(obs, device=self.device)
            with torch.no_grad():
                initial = self.network.initial_inference(obs_batch)
                root_value = initial.value.squeeze(0).item()

            # Store transition
            trajectory.add_step(
                obs=obs,
                action=action_idx,
                reward=0.0,  # Reward is assigned after step
                action_mask=action_mask,
                search_policy=search_policy,
                root_value=root_value,
            )

            # Take action in environment
            obs, reward, terminated, truncated, info = self.env.step(int(action_idx))
            episode_reward += reward
            episode_length += 1

            # Update last step's reward
            trajectory.steps[-1]["reward"] = reward

            if terminated or truncated:
                break

        self.buffer.save_episode(trajectory)
        self.episode_count += 1

        return episode_reward, episode_length

    def train_step(self, batch_size: int, unroll_steps: int = 5) -> dict[str, float]:
        """Perform one training step.

        Args:
            batch_size: Batch size for training.
            unroll_steps: Number of unroll steps.

        Returns:
            Dict with loss values.
        """
        if len(self.buffer) < batch_size:
            return {}

        batch = self.buffer.sample_batch(batch_size, unroll_steps=unroll_steps)

        # Extract tensors
        obs_batch = batch["obs_batch"]
        action_batch = batch["action_batch"].to(self.device)  # [B, K]
        reward_target = batch["reward_target"].to(self.device)  # [B, K]
        value_target = batch["value_target"].to(self.device)  # [B, K+1]
        policy_target = batch["policy_target"].to(self.device)  # [B, K+1, 80]
        action_mask_batch = batch["action_mask_batch"].to(self.device)  # [B, K+1, 80]

        batch_size_actual = len(obs_batch)

        # Convert observations to torch
        obs_torch_batch = {}
        for key in obs_batch[0].keys():
            obs_list = [obs[key] for obs in obs_batch]
            if isinstance(obs_list[0], np.ndarray):
                obs_torch_batch[key] = torch.stack([
                    torch.from_numpy(o).to(self.device) for o in obs_list
                ])
            else:
                obs_torch_batch[key] = torch.stack([
                    torch.tensor(o, device=self.device) for o in obs_list
                ])

        # Initial inference
        initial = self.network.initial_inference(obs_torch_batch)
        hidden_state = initial.hidden_state
        policy_logits = initial.policy_logits
        value_logits = initial.value_logits

        # Losses
        total_loss = 0.0
        policy_loss_sum = 0.0
        value_loss_sum = 0.0
        reward_loss_sum = 0.0

        # Initial step loss
        policy_loss = self._policy_loss(policy_logits, policy_target[:, 0], action_mask_batch[:, 0])
        value_loss = self._value_loss(value_logits, value_target[:, 0])
        loss = policy_loss + value_loss
        total_loss += loss / (unroll_steps + 1)
        policy_loss_sum += policy_loss.item()
        value_loss_sum += value_loss.item()

        # Unrolled steps
        action_embeddings = self.network.encode_actions(obs_torch_batch)  # [B, 80, 64]

        for step_k in range(unroll_steps):
            # Select action embeddings
            action_indices = action_batch[:, step_k]  # [B]
            action_emb = action_embeddings[torch.arange(batch_size_actual), action_indices]  # [B, 64]

            # Recurrent inference
            recurrent = self.network.recurrent_inference(hidden_state, action_emb)

            # Losses for this step
            policy_loss = self._policy_loss(
                recurrent.policy_logits,
                policy_target[:, step_k + 1],
                action_mask_batch[:, step_k + 1],
            )
            value_loss = self._value_loss(recurrent.value_logits, value_target[:, step_k + 1])
            reward_loss = self._reward_loss(recurrent.reward_logits, reward_target[:, step_k])

            loss = (policy_loss + value_loss + reward_loss) / (unroll_steps + 1)
            total_loss += loss
            policy_loss_sum += policy_loss.item()
            value_loss_sum += value_loss.item()
            reward_loss_sum += reward_loss.item()

            hidden_state = recurrent.next_hidden_state

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return {
            "loss/total": total_loss.item(),
            "loss/policy": policy_loss_sum / batch_size_actual,
            "loss/value": value_loss_sum / batch_size_actual,
            "loss/reward": reward_loss_sum / batch_size_actual,
        }

    def _policy_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Cross-entropy loss for policy head."""
        # logits: [B, 80], targets: [B, 80] (soft targets from MCTS)
        if action_mask is not None:
            logits = logits.masked_fill(action_mask <= 0, -1e9)
        log_probs = torch.log_softmax(logits, dim=-1)
        loss = -(targets * log_probs).sum(dim=-1).mean()
        return loss

    def _value_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Value loss using categorical support targets."""
        support_targets = scalar_to_support(
            targets.reshape(-1),
            support_size=self.network.support_size,
        ).to(logits.device)
        log_probs = torch.log_softmax(logits, dim=-1)
        loss = -(support_targets * log_probs).sum(dim=-1).mean()
        return loss

    def _reward_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Reward loss using categorical support targets."""
        support_targets = scalar_to_support(
            targets.reshape(-1),
            support_size=self.network.support_size,
        ).to(logits.device)
        log_probs = torch.log_softmax(logits, dim=-1)
        loss = -(support_targets * log_probs).sum(dim=-1).mean()
        return loss

    def save_checkpoint(self, tag: str = "") -> None:
        """Save training checkpoint."""
        checkpoint_dir = Path(self.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if tag:
            name = f"muzero_{tag}"
        else:
            name = f"muzero_step_{self.total_steps:08d}"

        checkpoint_path = checkpoint_dir / name
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        torch.save(self.network.state_dict(), checkpoint_path / "network.pt")
        torch.save(self.optimizer.state_dict(), checkpoint_path / "optimizer.pt")
        with (checkpoint_path / "replay_buffer.pkl").open("wb") as handle:
            pickle.dump(self.buffer.state_dict(), handle, protocol=pickle.HIGHEST_PROTOCOL)

        metadata = {
            "format": "muzero-v1",
            "total_steps": int(self.total_steps),
            "episode_count": int(self.episode_count),
            "buffer_size": len(self.buffer),
            "replay_buffer_saved": True,
        }
        (checkpoint_path / "metadata.json").write_text(json.dumps(metadata, indent=2))
        print(f"[checkpoint] Saved to {checkpoint_path}")


def main():
    """Main training loop."""
    parser = argparse.ArgumentParser(description="MuZero Training for STS2")

    # Core training
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--unroll-steps", type=int, default=5)
    parser.add_argument("--num-simulations", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--min-buffer-size", type=int, default=500)
    parser.add_argument("--train-every", type=int, default=50,
                        help="Train every N environment steps")
    parser.add_argument("--updates-per-train", type=int, default=3,
                        help="Gradient updates per training call")
    parser.add_argument("--checkpoint-freq", type=int, default=2048)
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Resume MuZero training from a checkpoint directory")

    # Directories
    parser.add_argument("--log-dir", type=str, default="runs")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")

    # Environment setup
    parser.add_argument("--combat-sandbox", action="store_true", default=False)
    parser.add_argument("--encounter-pool", type=str, default=None)
    parser.add_argument("--combat-snapshot-dataset", type=str, default=None)
    parser.add_argument("--combat-snapshot-split", type=str, default="train")
    parser.add_argument("--character", type=str, default=None,
                        help="Character to use for training")
    parser.add_argument("--defensive-buffs", action="store_true", default=False,
                        help="Apply defensive buffs on env/reset (mainly for debugging).")
    parser.add_argument("--combat-snapshot-character", type=str, default=None)
    parser.add_argument("--session-file", type=str, default=None)
    parser.add_argument("--session-files", type=str, default=None)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--device", type=str, default="cpu")

    # Discount factors
    parser.add_argument("--discount", type=float, default=0.997)
    parser.add_argument("--n-step-return", type=int, default=10)

    args = parser.parse_args()

    # Setup directories
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    snapshot_pool = None
    encounter_pool: list[str] = []
    if args.combat_sandbox:
        if args.encounter_pool:
            encounter_pool = parse_encounter_pool(args.encounter_pool)
        else:
            encounter_pool = parse_encounter_pool(DEFAULT_COMBAT_SANDBOX_TRAIN_POOL)

    # Setup snapshot pool if provided
    if args.combat_sandbox and args.combat_snapshot_dataset:
        try:
            print(f"[setup] Loading combat snapshot dataset from {args.combat_snapshot_dataset}...")
            snapshot_pool = CombatSnapshotPool.from_path(
                args.combat_snapshot_dataset,
                split=args.combat_snapshot_split or None,
                character=args.combat_snapshot_character,
            )
            print(f"[setup] Loaded {len(snapshot_pool)} snapshots")
        except Exception as e:
            print(f"[warning] Failed to load snapshot dataset: {e}")

    # Resolve session files
    session_files = resolve_training_session_files(
        n_envs=args.n_envs,
        session_file=args.session_file,
        session_files=parse_session_files(args.session_files),
    )

    # For now, only support single environment
    if args.n_envs != 1:
        print("[warning] Only single environment (--n-envs 1) is currently fully supported for MuZero")

    session_file = session_files[0] if session_files else None

    print(f"[setup] Creating training environment...")
    env = build_train_env(
        env_index=0,
        session_file=session_file,
        combat_sandbox=args.combat_sandbox,
        character=args.character,
        defensive_buffs=args.defensive_buffs,
        encounter_id=None,
        encounter_pool=encounter_pool,
        snapshot_pool=snapshot_pool,
        reset_timeout_ms=15000 if args.combat_sandbox else 60000,
        step_timeout_ms=20000,
    )

    # Create network, MCTS, buffer
    print(f"[setup] Creating MuZero network...")
    network = MuZeroNetwork()
    network.to(args.device)

    mcts = MCTS(
        num_simulations=args.num_simulations,
        discount=args.discount,
    )

    buffer = MuZeroReplayBuffer(capacity=args.buffer_size)

    optimizer = optim.Adam(
        network.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    trainer = MuZeroTrainer(
        network=network,
        mcts=mcts,
        buffer=buffer,
        env=env,
        optimizer=optimizer,
        device=args.device,
        discount=args.discount,
        n_step_return=args.n_step_return,
        log_dir=args.log_dir,
        checkpoint_dir=args.checkpoint_dir,
    )

    if args.resume_from:
        print(f"[resume] Loading checkpoint from {args.resume_from}...")
        resume_metadata = load_resume_checkpoint(
            args.resume_from,
            network=network,
            optimizer=optimizer,
            buffer=buffer,
            device=args.device,
        )
        trainer.total_steps = int(resume_metadata.get("total_steps", 0))
        trainer.episode_count = int(resume_metadata.get("episode_count", 0))
        print(
            f"[resume] Loaded total_steps={trainer.total_steps} "
            f"episode_count={trainer.episode_count} buffer={len(buffer)}"
        )
        if args.total_timesteps <= trainer.total_steps:
            raise ValueError(
                f"--total-timesteps ({args.total_timesteps}) must be greater than resumed total_steps "
                f"({trainer.total_steps})."
            )

    mode_name = "combat_sandbox" if args.combat_sandbox else "full_run"
    print(f"[setup] Starting training with {args.total_timesteps} timesteps")
    print(f"[setup] Mode: {mode_name}")
    if encounter_pool:
        print(f"[setup] Encounter pool: {encounter_pool}")

    # Training loop
    try:
        while trainer.total_steps < args.total_timesteps:
            # Self-play episode
            temperature = trainer.compute_temperature(trainer.total_steps, args.total_timesteps)
            ep_reward, ep_length = trainer.self_play_episode(temperature=temperature)

            trainer.total_steps += ep_length
            trainer.writer.add_scalar("episode/reward", ep_reward, trainer.episode_count)
            trainer.writer.add_scalar("episode/length", ep_length, trainer.episode_count)
            trainer.writer.add_scalar("schedule/temperature", temperature, trainer.total_steps)

            # Train on buffer
            if (trainer.episode_count % args.train_every) == 0 and len(buffer) >= args.min_buffer_size:
                for _ in range(args.updates_per_train):
                    losses = trainer.train_step(
                        batch_size=args.batch_size,
                        unroll_steps=args.unroll_steps,
                    )
                    for key, value in losses.items():
                        trainer.writer.add_scalar(key, value, trainer.total_steps)

            # Logging
            if (trainer.episode_count % 10) == 0:
                print(
                    f"[train] Episode {trainer.episode_count} | "
                    f"Steps {trainer.total_steps}/{args.total_timesteps} | "
                    f"Reward {ep_reward:+.4f} | "
                    f"Length {ep_length} | "
                    f"Buffer {len(buffer)} | "
                    f"T {temperature:.3f}"
                )

            # Checkpointing
            if (trainer.total_steps % args.checkpoint_freq) < ep_length:
                trainer.save_checkpoint()

            trainer.writer.add_scalar("buffer/size", len(buffer), trainer.total_steps)

    except KeyboardInterrupt:
        print("[interrupt] Caught KeyboardInterrupt, saving emergency checkpoint...")
        trainer.save_checkpoint(tag="crash")
        raise

    print(f"[train] Finished training after {trainer.total_steps} steps")
    trainer.save_checkpoint(tag="final")
    trainer.writer.close()


if __name__ == "__main__":
    main()
