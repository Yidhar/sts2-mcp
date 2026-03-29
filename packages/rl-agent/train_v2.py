"""STS2 RL Training with candidate-scoring policy.

Usage:
    python train_v2.py --total-timesteps 10000
    python train_v2.py --total-timesteps 10000 --no-text   # numeric-only baseline
    python train_v2.py --resume checkpoints_v2/sts2_ppo_v2_1000_steps.zip
"""

import argparse
import time
from pathlib import Path

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from sts2_env.env_v2 import SlayTheSpire2EnvV2
from sts2_env.observation_v2 import DictObservationEncoder
from sts2_env.policy import STS2CandidateScoringPolicy
from config import TrainConfig


def mask_fn(env):
    return env.unwrapped.action_masks()


class RewardBreakdownTensorboardCallback(BaseCallback):
    """Record bridge reward breakdown channels into TensorBoard."""

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if not isinstance(info, dict):
                continue
            bridge_info = info.get("bridge_info", {})
            if not isinstance(bridge_info, dict):
                continue
            reward_breakdown = bridge_info.get("reward_breakdown", {})
            if not isinstance(reward_breakdown, dict):
                continue

            for key, value in reward_breakdown.items():
                if isinstance(value, (int, float)):
                    self.logger.record_mean(f"reward_breakdown/{key}", float(value))
        return True


def main():
    parser = argparse.ArgumentParser(description="Train STS2 RL Agent")
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--character", type=str, default=None)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-steps", type=int, default=64)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--session-file", type=str, default=None)
    parser.add_argument("--log-dir", type=str, default="runs_v2")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints_v2")
    parser.add_argument("--checkpoint-freq", type=int, default=1000)
    parser.add_argument("--verbose", type=int, default=1)
    # Network
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=2)
    parser.add_argument("--text-proj-dim", type=int, default=32)
    parser.add_argument("--scorer-hidden", type=int, default=64)
    # Text
    parser.add_argument("--no-text", action="store_true", default=False,
                        help="Disable text embeddings (numeric-only baseline)")
    parser.add_argument("--text-model", type=str, default="BAAI/bge-small-zh-v1.5",
                        help="Sentence transformer model name")
    parser.add_argument("--text-cache-dir", type=str, default=None,
                        help="Directory for persistent text embedding cache")
    args = parser.parse_args()

    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    use_text = not args.no_text

    # Initialize text encoder with configured model if text is enabled
    if use_text:
        from sts2_env.text_encoder import get_text_encoder
        get_text_encoder(
            model_name=args.text_model,
            cache_dir=args.text_cache_dir,
        ).ensure_ready()

    # Environment
    obs_encoder = DictObservationEncoder(use_text=use_text)
    env = SlayTheSpire2EnvV2(
        session_file=args.session_file,
        character=args.character,
        obs_encoder=obs_encoder,
    )
    env = Monitor(env, filename=str(Path(args.log_dir) / "monitor"))
    env = ActionMasker(env, mask_fn)

    # Model
    reset_num_timesteps = True
    effective_total_timesteps = args.total_timesteps
    if args.resume:
        print(f"[train] Resuming from {args.resume}")
        model = MaskablePPO.load(args.resume, env=env, device=args.device)
        model.tensorboard_log = str(args.log_dir)
        model.verbose = args.verbose
        loaded_timesteps = int(getattr(model, "num_timesteps", 0))
        effective_total_timesteps = max(args.total_timesteps - loaded_timesteps, 0)
        reset_num_timesteps = False
        print(
            f"[train] Loaded checkpoint timesteps: {loaded_timesteps} | "
            f"target total: {args.total_timesteps} | remaining: {effective_total_timesteps}"
        )
    else:
        print(f"[train] Creating new model (text={'on' if use_text else 'off'})")
        model = MaskablePPO(
            STS2CandidateScoringPolicy,
            env,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=dict(
                embed_dim=args.embed_dim,
                n_heads=args.n_heads,
                text_proj_dim=args.text_proj_dim,
                scorer_hidden=args.scorer_hidden,
            ),
            verbose=args.verbose,
            tensorboard_log=args.log_dir,
            device=args.device,
        )

    total = sum(p.numel() for p in model.policy.parameters())
    print(f"[train] Parameters: {total:,}")

    # Train
    print(
        f"[train] target_total_timesteps={args.total_timesteps}, "
        f"effective_learn_timesteps={effective_total_timesteps}, device={args.device}"
    )
    start = time.time()
    callbacks = CallbackList([
        CheckpointCallback(
            save_freq=args.checkpoint_freq,
            save_path=args.checkpoint_dir,
            name_prefix="sts2_v2",
            verbose=1,
        ),
        RewardBreakdownTensorboardCallback(),
    ])
    model.learn(
        total_timesteps=effective_total_timesteps,
        callback=callbacks,
        reset_num_timesteps=reset_num_timesteps,
    )
    elapsed = time.time() - start
    print(f"[train] Done in {elapsed:.0f}s")

    final = str(Path(args.checkpoint_dir) / "sts2_v2_final")
    model.save(final)
    print(f"[train] Saved {final}.zip")

    # Save text cache
    if use_text:
        try:
            from sts2_env.text_encoder import get_text_encoder
            enc = get_text_encoder()
            enc.save_cache()
            print(f"[train] Text cache saved ({enc.cache_size} entries)")
        except Exception:
            pass

    env.close()


if __name__ == "__main__":
    main()
