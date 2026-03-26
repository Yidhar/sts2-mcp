"""STS2 RL Training with MaskablePPO + Attention Network (Phase 2).

Uses Dict observations and STS2AttentionExtractor for structured
per-card and per-enemy encoding with self-attention pooling.

Usage:
    # Activate venv first:
    # source venv/Scripts/activate  (bash on Windows)
    # venv\\Scripts\\activate.bat    (cmd on Windows)

    python train_v2.py                           # Train with defaults
    python train_v2.py --total-timesteps 50000   # Short training run
    python train_v2.py --character "Regent"      # Specific character
    python train_v2.py --resume checkpoints_v2/latest.zip  # Resume
"""

import argparse
import sys
import time
from pathlib import Path

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor

from sts2_env.env_v2 import SlayTheSpire2EnvV2
from sts2_env.network import STS2AttentionExtractor
from config import TrainConfig


def mask_fn(env):
    """Extract action masks from the environment."""
    return env.unwrapped.action_masks()


def make_env(config: TrainConfig):
    """Create and wrap the STS2 v2 environment."""
    env = SlayTheSpire2EnvV2(
        session_file=config.session_file,
        character=config.character,
        defensive_buffs=config.defensive_buffs,
        reset_timeout_ms=config.reset_timeout_ms,
        step_timeout_ms=config.step_timeout_ms,
        render_mode=None,
    )
    env = Monitor(env, filename=str(Path(config.log_dir) / "monitor_v2"))
    env = ActionMasker(env, mask_fn)
    return env


def train(config: TrainConfig, resume_from: str | None = None,
          features_dim: int = 128, card_embed_dim: int = 32,
          enemy_embed_dim: int = 32, n_heads: int = 2):
    """Run MaskablePPO training loop with attention-based policy."""
    print(f"[train_v2] Creating environment (Dict obs + attention)...")
    env = make_env(config)

    activation_fn_map = {
        "relu": __import__("torch").nn.ReLU,
        "tanh": __import__("torch").nn.Tanh,
    }

    if resume_from:
        print(f"[train_v2] Resuming from {resume_from}")
        model = MaskablePPO.load(resume_from, env=env, device=config.device)
    else:
        print(f"[train_v2] Creating new MaskablePPO model with attention extractor")
        model = MaskablePPO(
            "MultiInputPolicy",
            env,
            learning_rate=config.learning_rate,
            n_steps=config.n_steps,
            batch_size=config.batch_size,
            n_epochs=config.n_epochs,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            clip_range=config.clip_range,
            ent_coef=config.ent_coef,
            vf_coef=config.vf_coef,
            max_grad_norm=config.max_grad_norm,
            policy_kwargs=dict(
                features_extractor_class=STS2AttentionExtractor,
                features_extractor_kwargs=dict(
                    features_dim=features_dim,
                    card_embed_dim=card_embed_dim,
                    enemy_embed_dim=enemy_embed_dim,
                    n_heads=n_heads,
                ),
                net_arch=[128, 128],
                activation_fn=activation_fn_map.get(
                    config.activation_fn, __import__("torch").nn.ReLU
                ),
            ),
            verbose=config.verbose,
            tensorboard_log=config.log_dir if config.tensorboard else None,
            device=config.device,
        )

    # Callbacks
    callbacks = []

    # Checkpoint saving
    callbacks.append(CheckpointCallback(
        save_freq=config.checkpoint_freq,
        save_path=config.checkpoint_dir,
        name_prefix="sts2_ppo_v2",
        verbose=1,
    ))

    # Training
    print(f"[train_v2] Starting training for {config.total_timesteps} timesteps")
    print(f"[train_v2] Character: {config.character or 'any'}")
    print(f"[train_v2] Defensive buffs: {config.defensive_buffs}")
    print(f"[train_v2] Device: {config.device}")
    print(f"[train_v2] Attention: features_dim={features_dim}, "
          f"card_embed={card_embed_dim}, enemy_embed={enemy_embed_dim}, "
          f"n_heads={n_heads}")
    start = time.time()

    model.learn(
        total_timesteps=config.total_timesteps,
        callback=CallbackList(callbacks) if callbacks else None,
    )

    elapsed = time.time() - start
    print(f"[train_v2] Training complete in {elapsed:.1f}s")

    # Save final model
    final_path = str(Path(config.checkpoint_dir) / "sts2_ppo_v2_final")
    model.save(final_path)
    print(f"[train_v2] Final model saved to {final_path}.zip")

    env.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train STS2 RL Agent (Phase 2: Attention)"
    )
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--character", type=str, default=None)
    parser.add_argument("--defensive-buffs", action="store_true", default=True)
    parser.add_argument("--no-defensive-buffs", dest="defensive_buffs",
                        action="store_false")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--session-file", type=str, default=None)
    parser.add_argument("--log-dir", type=str, default="runs_v2")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints_v2")
    parser.add_argument("--checkpoint-freq", type=int, default=10_000)
    parser.add_argument("--verbose", type=int, default=1)

    # Attention network parameters
    parser.add_argument("--features-dim", type=int, default=128,
                        help="Output dim of the attention feature extractor")
    parser.add_argument("--card-embed-dim", type=int, default=32,
                        help="Per-card embedding dimension")
    parser.add_argument("--enemy-embed-dim", type=int, default=32,
                        help="Per-enemy embedding dimension")
    parser.add_argument("--n-heads", type=int, default=2,
                        help="Number of attention heads")
    return parser.parse_args()


def main():
    args = parse_args()
    config = TrainConfig(
        session_file=args.session_file,
        character=args.character,
        defensive_buffs=args.defensive_buffs,
        total_timesteps=args.total_timesteps,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        n_steps=args.n_steps,
        device=args.device,
        log_dir=args.log_dir,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_freq=args.checkpoint_freq,
        verbose=args.verbose,
    )
    train(
        config,
        resume_from=args.resume,
        features_dim=args.features_dim,
        card_embed_dim=args.card_embed_dim,
        enemy_embed_dim=args.enemy_embed_dim,
        n_heads=args.n_heads,
    )


if __name__ == "__main__":
    main()
