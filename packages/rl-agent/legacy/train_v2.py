"""Legacy STS2 RL training entrypoint for the routed candidate-scoring policy.

The mainline search-free attention runtime now lives in ``train_attention_policy.py``.

Usage:
    python train_v2.py --total-timesteps 10000
    python train_v2.py --total-timesteps 10000 --no-text   # numeric-only baseline
"""

import argparse
import math
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import ConstantSchedule, FloatSchedule
from stable_baselines3.common.vec_env import SubprocVecEnv

from combat_snapshot_dataset import (
    CombatSnapshotPool,
    DEFAULT_CURATED_COMBINED_SUBSET,
    VALID_CURATED_COMBINED_SUBSETS,
)
from launcher import get_session_files as get_default_multi_session_files
from sts2_env.checkpoint import load_online_checkpoint_metadata, load_online_policy_state_dict, save_online_checkpoint
from sts2_env.combat_env import CombatSandboxEnv
from sts2_env.env_v2 import SlayTheSpire2EnvV2
from sts2_env.model import STS2CandidateScoringPolicy
from sts2_env.observation_v2 import DictObservationEncoder

DEFAULT_COMBAT_SANDBOX_TRAIN_POOL = ",".join([
    "ENCOUNTER.SLIMES_WEAK",
    "ENCOUNTER.SHRINKER_BEETLE_WEAK",
    "ENCOUNTER.FUZZY_WURM_CRAWLER_WEAK",
    "ENCOUNTER.NIBBITS_WEAK",
])
DEFAULT_COMBAT_SANDBOX_HOLDOUT_POOL = ",".join([
    "ENCOUNTER.CORPSE_SLUGS_WEAK",
    "ENCOUNTER.SLUDGE_SPINNER_WEAK",
    "ENCOUNTER.SEAPUNK_WEAK",
    "ENCOUNTER.TOADPOLES_WEAK",
])


def mask_fn(env):
    return env.unwrapped.action_masks()


def parse_encounter_pool(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [entry.strip() for entry in raw.split(",") if entry.strip()]


def parse_session_files(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [entry.strip() for entry in raw.split(",") if entry.strip()]


def resolve_training_session_files(
    *,
    n_envs: int,
    session_file: str | None,
    session_files: list[str],
) -> list[str | None]:
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


def build_train_env_factory(
    *,
    env_index: int,
    session_file: str | None,
    use_text: bool,
    log_dir: str,
    combat_sandbox: bool,
    character: str | None,
    encounter_id: str | None,
    encounter_pool: list[str],
    snapshot_pool: CombatSnapshotPool | None,
    reset_timeout_ms: int,
    step_timeout_ms: int,
) -> Callable[[], object]:
    def _factory():
        obs_encoder = DictObservationEncoder(use_text=use_text)
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
        else:
            env = SlayTheSpire2EnvV2(
                session_file=session_file,
                character=character,
                reset_timeout_ms=reset_timeout_ms,
                step_timeout_ms=step_timeout_ms,
                obs_encoder=obs_encoder,
            )

        monitor_name = "monitor" if env_index == 0 else f"monitor_{env_index}"
        env = Monitor(env, filename=str(Path(log_dir) / monitor_name))
        env = ActionMasker(env, mask_fn)
        return env

    return _factory


def sanitize_metric_key(value: str) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
    if sanitized.startswith("encounter_"):
        sanitized = sanitized[len("encounter_"):]
    return sanitized or "unknown"


def build_lr_schedule(
    schedule_name: str,
    base_lr: float,
    min_lr: float,
    warmup_fraction: float,
):
    if base_lr <= 0:
        raise ValueError("--learning-rate must be > 0")
    if min_lr < 0:
        raise ValueError("--min-learning-rate must be >= 0")
    if min_lr > base_lr:
        raise ValueError("--min-learning-rate cannot exceed --learning-rate")
    if not 0.0 <= warmup_fraction < 1.0:
        raise ValueError("--warmup-fraction must be in [0, 1)")

    if schedule_name == "constant":
        return FloatSchedule(ConstantSchedule(base_lr))

    def cosine(progress_remaining: float) -> float:
        progress = min(max(1.0 - float(progress_remaining), 0.0), 1.0)
        return float(min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress)))

    if schedule_name == "cosine":
        return FloatSchedule(cosine)

    if schedule_name == "warmup_cosine":
        def warmup_cosine(progress_remaining: float) -> float:
            progress = min(max(1.0 - float(progress_remaining), 0.0), 1.0)
            if progress < warmup_fraction:
                warmup_progress = progress / max(warmup_fraction, 1e-12)
                return float(min_lr + (base_lr - min_lr) * warmup_progress)
            cosine_progress = (progress - warmup_fraction) / max(1.0 - warmup_fraction, 1e-12)
            return float(min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * cosine_progress)))

        return FloatSchedule(warmup_cosine)

    raise ValueError(f"Unsupported --lr-schedule: {schedule_name}")


CORE_REWARD_BREAKDOWN_KEYS = (
    "hp_loss_normalized",
    "hp_gain_normalized",
    "room_complete",
    "combat_room_complete",
    "room_hp_delta_normalized",
    "combat_room_complete_bonus",
    "combat_room_quality_bonus",
    "floor_delta",
    "floor_progress_bonus",
    "act_clear",
    "act_clear_bonus",
    "elite_clear_bonus",
    "boss_clear_bonus",
    "relic_gain_count",
    "relic_gain_bonus",
    "max_hp_gain_normalized",
    "max_hp_gain_bonus",
    "death",
    "victory",
    "run_victory_bonus",
    "action_error_penalty",
    "truncated_penalty",
    "total",
)

CORE_ACTION_DIAGNOSTIC_KEYS = (
    "end_turn_selected",
    "end_turn_wasted",
    "non_end_action_count",
    "play_card_action_count",
    "zero_cost_play_card_count",
    "positive_preview_action_count",
    "self_hp_loss_action_count",
    "invalid_action_selected",
    "legal_action_overflow",
)


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
                reward_breakdown = {}

            for key in CORE_REWARD_BREAKDOWN_KEYS:
                value = reward_breakdown.get(key)
                if isinstance(value, (int, float)):
                    self.logger.record_mean(f"reward_breakdown/{key}", float(value))

            action_diagnostics = bridge_info.get("action_diagnostics", {})
            if not isinstance(action_diagnostics, dict):
                action_diagnostics = {}

            for key in CORE_ACTION_DIAGNOSTIC_KEYS:
                value = action_diagnostics.get(key)
                if isinstance(value, (int, float, bool)):
                    self.logger.record_mean(f"action_diagnostics/{key}", float(value))
        return True


class CombatFixedEvalCallback(BaseCallback):
    """Periodic fixed-combat evaluation on rollout boundaries.

    This callback intentionally evaluates only between rollouts. Because the
    combat sandbox controls a single live STS2 instance, the training env is
    force-reset after eval so the next rollout starts from a clean episode.
    """

    def __init__(
        self,
        *,
        session_file,
        character,
        use_text,
        seen_encounters,
        holdout_encounters,
        reset_timeout_ms,
        step_timeout_ms,
        eval_freq,
        episodes_per_encounter,
        deterministic,
        seed_base,
    ):
        super().__init__(verbose=0)
        self.session_file = session_file
        self.character = character
        self.use_text = use_text
        self.seen_encounters = list(seen_encounters)
        self.holdout_encounters = list(holdout_encounters)
        self.reset_timeout_ms = reset_timeout_ms
        self.step_timeout_ms = step_timeout_ms
        self.eval_freq = max(int(eval_freq), 0)
        self.episodes_per_encounter = max(int(episodes_per_encounter), 1)
        self.deterministic = deterministic
        self.seed_base = int(seed_base)
        self._last_eval_step = 0
        self._eval_env = None

    def _init_callback(self) -> None:
        obs_encoder = DictObservationEncoder(use_text=self.use_text)
        self._eval_env = CombatSandboxEnv(
            session_file=self.session_file,
            character=self.character,
            reset_timeout_ms=self.reset_timeout_ms,
            step_timeout_ms=self.step_timeout_ms,
            obs_encoder=obs_encoder,
            include_debug_info=True,
        )

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        if self.eval_freq <= 0:
            return
        if (self.model.num_timesteps - self._last_eval_step) < self.eval_freq:
            return
        self._last_eval_step = int(self.model.num_timesteps)

        seen_metrics = self._evaluate_pool("seen", self.seen_encounters)
        if seen_metrics:
            self._record_pool_metrics("seen", seen_metrics)

        holdout_metrics = self._evaluate_pool("holdout", self.holdout_encounters)
        if holdout_metrics:
            self._record_pool_metrics("holdout", holdout_metrics)

        # Eval reuses the live bridge instance. Force a clean reset so the next
        # rollout does not continue from a stale episode invalidated by eval.
        if self.training_env is not None:
            self.model._last_obs = self.training_env.reset()
            self.model._last_episode_starts = np.ones((self.training_env.num_envs,), dtype=bool)

    def _evaluate_pool(self, pool_name: str, encounter_ids: list[str]) -> dict[str, object]:
        if not encounter_ids:
            return {}

        per_encounter: dict[str, dict[str, float]] = {}
        rewards_all: list[float] = []
        lengths_all: list[int] = []
        deaths_all: list[float] = []
        hp_loss_all: list[float] = []

        for encounter_index, encounter_id in enumerate(encounter_ids):
            rewards: list[float] = []
            lengths: list[int] = []
            deaths: list[float] = []
            hp_losses: list[float] = []

            for episode_index in range(self.episodes_per_encounter):
                seed = self.seed_base + encounter_index * 1000 + episode_index
                obs, info = self._eval_env.reset(
                    options={"encounter_id": encounter_id, "seed": seed}
                )
                total_reward = 0.0
                steps = 0

                while True:
                    action_mask = info.get("action_mask")
                    action, _ = self.model.predict(
                        obs,
                        action_masks=action_mask,
                        deterministic=self.deterministic,
                    )
                    obs, reward, terminated, truncated, info = self._eval_env.step(int(action))
                    total_reward += float(reward)
                    steps += 1
                    if terminated or truncated:
                        break

                raw_obs = info.get("raw_obs", {}) if isinstance(info, dict) else {}
                player = raw_obs.get("player", {}) if isinstance(raw_obs, dict) else {}
                current_hp = player.get("hp")
                max_hp = player.get("max_hp")
                died = bool(isinstance(current_hp, (int, float)) and current_hp <= 0)
                hp_loss = None
                if isinstance(current_hp, (int, float)) and isinstance(max_hp, (int, float)):
                    hp_loss = float(max_hp - current_hp)

                rewards.append(total_reward)
                lengths.append(steps)
                deaths.append(1.0 if died else 0.0)
                if hp_loss is not None:
                    hp_losses.append(hp_loss)

            reward_mean = float(np.mean(rewards)) if rewards else 0.0
            length_mean = float(np.mean(lengths)) if lengths else 0.0
            death_rate = float(np.mean(deaths)) if deaths else 0.0
            hp_loss_mean = float(np.mean(hp_losses)) if hp_losses else 0.0

            per_encounter[encounter_id] = {
                "mean_reward": reward_mean,
                "mean_length": length_mean,
                "death_rate": death_rate,
                "mean_hp_loss": hp_loss_mean,
            }
            rewards_all.extend(rewards)
            lengths_all.extend(lengths)
            deaths_all.extend(deaths)
            hp_loss_all.extend(hp_losses)

        summary = {
            "pool_name": pool_name,
            "mean_reward": float(np.mean(rewards_all)) if rewards_all else 0.0,
            "mean_length": float(np.mean(lengths_all)) if lengths_all else 0.0,
            "death_rate": float(np.mean(deaths_all)) if deaths_all else 0.0,
            "mean_hp_loss": float(np.mean(hp_loss_all)) if hp_loss_all else 0.0,
            "per_encounter": per_encounter,
        }

        print(
            f"[eval/{pool_name}] step={self.model.num_timesteps} "
            f"mean_reward={summary['mean_reward']:+.4f} "
            f"death_rate={summary['death_rate']:.3f} "
            f"mean_length={summary['mean_length']:.2f}"
        )
        for encounter_id, metrics in per_encounter.items():
            print(
                f"  [eval/{pool_name}] {encounter_id}: "
                f"reward={metrics['mean_reward']:+.4f} "
                f"death_rate={metrics['death_rate']:.3f} "
                f"len={metrics['mean_length']:.2f} "
                f"hp_loss={metrics['mean_hp_loss']:.2f}"
            )

        return summary

    def _record_pool_metrics(self, prefix: str, metrics: dict[str, object]) -> None:
        self.logger.record(f"eval/{prefix}_r", float(metrics["mean_reward"]))
        self.logger.record(f"eval/{prefix}_d", float(metrics["death_rate"]))
        self.logger.record(f"eval/{prefix}_l", float(metrics["mean_length"]))
        self.logger.record(f"eval/{prefix}_hp", float(metrics["mean_hp_loss"]))

        for encounter_id, encounter_metrics in metrics["per_encounter"].items():
            metric_key = sanitize_metric_key(encounter_id)
            self.logger.record(f"eval/{prefix}/{metric_key}_r", float(encounter_metrics["mean_reward"]))
            self.logger.record(f"eval/{prefix}/{metric_key}_d", float(encounter_metrics["death_rate"]))
            self.logger.record(f"eval/{prefix}/{metric_key}_l", float(encounter_metrics["mean_length"]))
            self.logger.record(f"eval/{prefix}/{metric_key}_hp", float(encounter_metrics["mean_hp_loss"]))

    def _on_training_end(self) -> None:
        if self._eval_env is not None:
            self._eval_env.close()
            self._eval_env = None


class EncounterStatsCallback(BaseCallback):
    """Track rolling per-encounter training stats from completed episodes."""

    def __init__(self, window_size: int = 50):
        super().__init__(verbose=0)
        self.window_size = max(int(window_size), 1)
        self._stats: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: {
                "reward": [],
                "length": [],
            }
        )

    def _trim(self, values: list[float]) -> None:
        if len(values) > self.window_size:
            del values[:-self.window_size]

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        if infos is None:
            return True

        for index, info in enumerate(infos):
            if not isinstance(info, dict):
                continue
            done = bool(dones[index]) if index < len(dones) else False
            episode = info.get("episode")
            encounter_id = info.get("encounter_id")
            if not done or not isinstance(episode, dict) or not encounter_id:
                continue

            reward = episode.get("r")
            length = episode.get("l")
            if isinstance(reward, (int, float)):
                self._stats[encounter_id]["reward"].append(float(reward))
                self._trim(self._stats[encounter_id]["reward"])
            if isinstance(length, (int, float)):
                self._stats[encounter_id]["length"].append(float(length))
                self._trim(self._stats[encounter_id]["length"])

        for encounter_id, metric_lists in self._stats.items():
            metric_key = sanitize_metric_key(encounter_id)
            rewards = metric_lists["reward"]
            lengths = metric_lists["length"]
            if rewards:
                self.logger.record(
                    f"train_enc/{metric_key}_r",
                    float(np.mean(rewards)),
                )
            if lengths:
                self.logger.record(
                    f"train_enc/{metric_key}_l",
                    float(np.mean(lengths)),
                )
        return True


class SafeTensorCheckpointCallback(BaseCallback):
    def __init__(self, *, save_freq: int, checkpoint_dir: str, metadata_builder, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.save_freq = max(int(save_freq), 1)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.metadata_builder = metadata_builder

    def _on_step(self) -> bool:
        if self.num_timesteps <= 0 or (self.num_timesteps % self.save_freq) != 0:
            return True

        checkpoint_path = self.checkpoint_dir / f"step_{self.num_timesteps:08d}"
        metadata = self.metadata_builder(self.model, self.num_timesteps)
        save_online_checkpoint(self.model, checkpoint_path, metadata=metadata)
        if self.verbose:
            print(f"[train] Saved checkpoint {checkpoint_path}")
        return True


def build_online_checkpoint_metadata(
    args,
    *,
    use_text: bool,
    timesteps: int,
    policy_kwargs: dict[str, object] | None = None,
    text_model_name: str | None = None,
) -> dict[str, object]:
    resolved_policy_kwargs = policy_kwargs or build_policy_kwargs_from_args(args)
    return {
        "format": "sts2-online-policy-v1",
        "policy_class": "sts2_env.model.STS2CandidateScoringPolicy",
        "policy_kwargs": resolved_policy_kwargs,
        "use_text": use_text,
        "text_model": (text_model_name or args.text_model) if use_text else None,
        "timesteps": int(timesteps),
        "mode": "combat_sandbox" if args.combat_sandbox else "full_run",
        "character": args.character,
        "combat_snapshot_dataset": args.combat_snapshot_dataset,
        "combat_snapshot_split": args.combat_snapshot_split or None,
        "combat_snapshot_sample_mode": args.combat_snapshot_sample_mode if args.combat_snapshot_dataset else None,
    }


def build_policy_kwargs_from_args(args) -> dict[str, object]:
    return dict(
        combat_embed_dim=args.combat_embed_dim,
        build_embed_dim=args.build_embed_dim,
        route_embed_dim=args.route_embed_dim,
        n_heads=args.n_heads,
        text_proj_dim=args.text_proj_dim,
        context_text_dim=args.context_text_dim,
        shared_hidden_dim=args.shared_hidden_dim,
        shared_output_dim=args.shared_output_dim,
        combat_scorer_hidden=args.combat_scorer_hidden,
        build_scorer_hidden=args.build_scorer_hidden,
        route_scorer_hidden=args.route_scorer_hidden,
        critic_domain_dim=args.critic_domain_dim,
    )


def _get_live_supported_encounter_ids(session_file=None):
    from sts2_env.bridge_client import BridgeClient

    client = BridgeClient(session_path=session_file)
    catalog = client.combat_catalog()
    return [
        entry.get("encounter_id")
        for entry in (catalog.get("encounters") or [])
        if entry.get("encounter_id")
    ]


def main():
    parser = argparse.ArgumentParser(description="Train STS2 RL Agent")
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--character", type=str, default=None)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument(
        "--lr-schedule",
        type=str,
        default="constant",
        choices=("constant", "cosine", "warmup_cosine"),
        help="Learning rate schedule for fresh training.",
    )
    parser.add_argument(
        "--min-learning-rate",
        type=float,
        default=0.0,
        help="Floor LR for cosine-style schedules.",
    )
    parser.add_argument(
        "--warmup-fraction",
        type=float,
        default=0.0,
        help="Warmup fraction for --lr-schedule warmup_cosine.",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.25)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--session-file", type=str, default=None)
    parser.add_argument(
        "--session-files",
        type=str,
        default=None,
        help=(
            "Comma-separated bridge session file paths for multi-instance training. "
            "When omitted and --n-envs > 1, defaults to session_0..session_{n-1}.json under APPDATA/SlayTheSpire2/bridge."
        ),
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=1,
        help="Number of parallel training environments / bridge instances.",
    )
    parser.add_argument("--log-dir", type=str, default="runs")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--checkpoint-freq", type=int, default=1000)
    parser.add_argument("--init-checkpoint", type=str, default=None,
                        help="Initialize weights from an existing online checkpoint directory")
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--combat-sandbox", action="store_true", default=False,
                        help="Train in combat sandbox mode via /env/combat_reset")
    parser.add_argument("--encounter-id", type=str, default=None,
                        help="Encounter ID for combat sandbox training")
    parser.add_argument(
        "--encounter-pool",
        type=str,
        default=None,
        help=(
            "Comma-separated encounter IDs sampled uniformly on each combat reset. "
            "If omitted in --combat-sandbox mode, defaults to starter-deck-friendly early Act 1 weak encounters."
        ),
    )
    parser.add_argument(
        "--combat-snapshot-dataset",
        type=str,
        default=None,
        help=(
            "Path to combat_snapshot_samples.{jsonl,parquet} or a dataset root/partition dir. "
            "When set in --combat-sandbox mode, each episode samples a historical pre-combat build snapshot."
        ),
    )
    parser.add_argument(
        "--combat-curated-subset",
        type=str,
        default=DEFAULT_CURATED_COMBINED_SUBSET,
        choices=sorted(VALID_CURATED_COMBINED_SUBSETS),
        help=(
            "When --combat-snapshot-dataset points at a curated combat root/combined dir, "
            "resolve this subset. Default keeps all human rows plus only local runs that cleared Act 1."
        ),
    )
    parser.add_argument(
        "--combat-snapshot-split",
        type=str,
        default="train",
        help="Split filter for --combat-snapshot-dataset. Use '' to disable. Default: train.",
    )
    parser.add_argument(
        "--combat-snapshot-character",
        type=str,
        default=None,
        help="Optional character filter for --combat-snapshot-dataset. Defaults to --character when set.",
    )
    parser.add_argument(
        "--combat-snapshot-build-id",
        type=str,
        default=None,
        help="Optional build_id filter for --combat-snapshot-dataset.",
    )
    parser.add_argument(
        "--combat-snapshot-max-rows",
        type=int,
        default=None,
        help="Optional cap on loaded combat snapshot rows.",
    )
    parser.add_argument(
        "--combat-snapshot-min-floor",
        type=int,
        default=None,
        help="Optional minimum floor filter for combat snapshots.",
    )
    parser.add_argument(
        "--combat-snapshot-max-floor",
        type=int,
        default=None,
        help="Optional maximum floor filter for combat snapshots.",
    )
    parser.add_argument(
        "--combat-snapshot-sample-mode",
        type=str,
        default="encounter_balanced",
        choices=("row_uniform", "encounter_balanced"),
        help="Sampling mode for combat snapshot episodes. Default: encounter_balanced.",
    )
    parser.add_argument("--reset-timeout-ms", type=int, default=None,
                        help="Override env reset timeout in milliseconds")
    parser.add_argument("--step-timeout-ms", type=int, default=None,
                        help="Override env step timeout in milliseconds")
    # Network
    parser.add_argument("--combat-embed-dim", type=int, default=64)
    parser.add_argument("--build-embed-dim", type=int, default=64)
    parser.add_argument("--route-embed-dim", type=int, default=48)
    parser.add_argument("--n-heads", type=int, default=2)
    parser.add_argument("--text-proj-dim", type=int, default=32)
    parser.add_argument("--context-text-dim", type=int, default=48)
    parser.add_argument("--shared-hidden-dim", type=int, default=96)
    parser.add_argument("--shared-output-dim", type=int, default=128)
    parser.add_argument("--combat-scorer-hidden", type=int, default=128)
    parser.add_argument("--build-scorer-hidden", type=int, default=96)
    parser.add_argument("--route-scorer-hidden", type=int, default=64)
    parser.add_argument("--critic-domain-dim", type=int, default=64)
    # Text
    parser.add_argument("--no-text", action="store_true", default=False,
                        help="Disable text embeddings (numeric-only baseline)")
    parser.add_argument("--text-model", type=str, default="BAAI/bge-small-zh-v1.5",
                        help="Sentence transformer model name")
    parser.add_argument("--text-cache-dir", type=str, default=None,
                        help="Directory for persistent text embedding cache")
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=0,
        help=(
            "Run fixed combat eval every N env steps at rollout boundaries. "
            "Default is 0 because single-instance live combat sandbox training cannot safely reset into a separate eval episode."
        ),
    )
    parser.add_argument(
        "--eval-episodes-per-encounter",
        type=int,
        default=3,
        help="Deterministic eval episodes per encounter.",
    )
    parser.add_argument(
        "--eval-holdout-pool",
        type=str,
        default=None,
        help=(
            "Comma-separated holdout encounter IDs for fixed eval. "
            "If omitted in --combat-sandbox mode, defaults to unseen early Act 1 weak encounters."
        ),
    )
    parser.add_argument(
        "--eval-stochastic",
        action="store_true",
        default=False,
        help="Use stochastic sampling during fixed eval. Default is deterministic eval.",
    )
    parser.add_argument(
        "--eval-seed-base",
        type=int,
        default=1337,
        help="Base seed for fixed eval episodes.",
    )
    args = parser.parse_args()

    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    init_metadata = load_online_checkpoint_metadata(args.init_checkpoint) if args.init_checkpoint else None
    use_text = False if args.no_text else bool((init_metadata or {}).get("use_text", True))
    text_model_name = (init_metadata or {}).get("text_model") or args.text_model
    if args.combat_sandbox and not args.encounter_id and not args.encounter_pool and not args.combat_snapshot_dataset:
        args.encounter_pool = DEFAULT_COMBAT_SANDBOX_TRAIN_POOL
    if args.combat_sandbox and not args.eval_holdout_pool:
        args.eval_holdout_pool = DEFAULT_COMBAT_SANDBOX_HOLDOUT_POOL

    encounter_pool = parse_encounter_pool(args.encounter_pool)
    eval_holdout_pool = parse_encounter_pool(args.eval_holdout_pool)
    session_files = parse_session_files(args.session_files)
    try:
        resolved_session_files = resolve_training_session_files(
            n_envs=args.n_envs,
            session_file=args.session_file,
            session_files=session_files,
        )
    except ValueError as exc:
        parser.error(str(exc))

    primary_session_file = resolved_session_files[0]
    if args.eval_freq > 0 and len(resolved_session_files) > 1:
        parser.error("--eval-freq with --n-envs > 1 is not supported yet.")

    snapshot_split = args.combat_snapshot_split or None
    snapshot_pool = None
    if args.combat_snapshot_dataset:
        snapshot_character = args.combat_snapshot_character or args.character
        supported_encounter_ids = _get_live_supported_encounter_ids(session_file=primary_session_file)
        snapshot_pool = CombatSnapshotPool.from_path(
            args.combat_snapshot_dataset,
            curated_subset=args.combat_curated_subset,
            split=snapshot_split,
            character=snapshot_character,
            build_id=args.combat_snapshot_build_id,
            encounter_ids=supported_encounter_ids,
            min_floor=args.combat_snapshot_min_floor,
            max_floor=args.combat_snapshot_max_floor,
            max_rows=args.combat_snapshot_max_rows,
            sample_mode=args.combat_snapshot_sample_mode,
        )
        snapshot_summary = snapshot_pool.summary()
        print(
            "[train] Loaded combat snapshot pool "
            f"rows={snapshot_summary['row_count']} "
            f"encounters={snapshot_summary['encounter_count']} "
            f"floors={snapshot_summary['min_floor']}..{snapshot_summary['max_floor']} "
            f"characters={snapshot_summary['characters']} "
            f"build_ids={snapshot_summary['build_ids']} "
            f"sample_mode={snapshot_summary['sample_mode']}"
        )
        if snapshot_summary["top_encounters"]:
            print(f"[train] Combat snapshot top_encounters={snapshot_summary['top_encounters']}")
    lr_schedule = build_lr_schedule(
        args.lr_schedule,
        args.learning_rate,
        args.min_learning_rate,
        args.warmup_fraction,
    )

    # Initialize text encoder with configured model if text is enabled
    if use_text:
        from sts2_env.text_encoder import get_text_encoder
        get_text_encoder(
            model_name=text_model_name,
            cache_dir=args.text_cache_dir,
        ).ensure_ready()

    # Environment
    if args.combat_sandbox and not args.encounter_id and not encounter_pool and snapshot_pool is None:
        parser.error("--combat-sandbox requires --encounter-id, --encounter-pool, or --combat-snapshot-dataset")

    env_fns = [
        build_train_env_factory(
            env_index=env_index,
            session_file=session_file,
            use_text=use_text,
            log_dir=args.log_dir,
            combat_sandbox=args.combat_sandbox,
            character=args.character,
            encounter_id=args.encounter_id,
            encounter_pool=encounter_pool,
            snapshot_pool=snapshot_pool,
            reset_timeout_ms=args.reset_timeout_ms or (30000 if args.combat_sandbox else 60000),
            step_timeout_ms=args.step_timeout_ms or 20000,
        )
        for env_index, session_file in enumerate(resolved_session_files)
    ]

    if len(env_fns) == 1:
        env = env_fns[0]()
    else:
        env = SubprocVecEnv(
            env_fns,
            start_method="spawn",
        )

    rollout_size = int(args.n_steps) * max(len(resolved_session_files), 1)
    if args.batch_size > rollout_size:
        parser.error(
            f"--batch-size ({args.batch_size}) cannot exceed rollout size "
            f"(--n-steps * --n-envs = {rollout_size})."
        )
    if rollout_size % args.batch_size != 0:
        print(
            f"[train] Warning: rollout_size={rollout_size} is not divisible by batch_size={args.batch_size}; "
            "the final minibatch of each epoch will be truncated."
        )

    # Model
    mode = "combat_sandbox" if args.combat_sandbox else "full_run"
    policy_kwargs = (init_metadata or {}).get("policy_kwargs") or build_policy_kwargs_from_args(args)
    print(
        f"[train] Creating new model mode={mode} "
        f"(text={'on' if use_text else 'off'}, n_envs={len(resolved_session_files)})"
    )
    if len(resolved_session_files) > 1:
        print(f"[train] Session files: {resolved_session_files}")
    model = MaskablePPO(
        STS2CandidateScoringPolicy,
        env,
        learning_rate=lr_schedule,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=0.2,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        verbose=args.verbose,
        tensorboard_log=args.log_dir,
        device=args.device,
    )
    if args.init_checkpoint:
        loaded_metadata = load_online_policy_state_dict(model, args.init_checkpoint, device=args.device, strict=True)
        print(
            f"[train] Initialized weights from {args.init_checkpoint} "
            f"(source_timesteps={loaded_metadata.get('timesteps', 0)})"
        )
    print(
        f"[train] Using LR schedule schedule={args.lr_schedule} "
        f"base_lr={args.learning_rate:.8f} min_lr={args.min_learning_rate:.8f} "
        f"warmup_fraction={args.warmup_fraction:.4f}"
    )
    print(
        f"[train] PPO geometry n_envs={len(resolved_session_files)} "
        f"n_steps={args.n_steps} rollout_size={rollout_size} batch_size={args.batch_size} "
        f"n_epochs={args.n_epochs} gamma={args.gamma:.4f} gae_lambda={args.gae_lambda:.4f} "
        f"ent_coef={args.ent_coef:.4f} vf_coef={args.vf_coef:.4f}"
    )

    total = sum(p.numel() for p in model.policy.parameters())
    print(f"[train] Parameters: {total:,}")
    if args.combat_sandbox:
        if snapshot_pool is not None:
            print(f"[train] Combat sandbox snapshot_dataset={args.combat_snapshot_dataset}")
        if encounter_pool:
            print(f"[train] Combat sandbox encounter_pool={encounter_pool}")
        else:
            print(f"[train] Combat sandbox encounter={args.encounter_id}")
        if eval_holdout_pool:
            print(f"[train] Combat sandbox eval_holdout_pool={eval_holdout_pool}")

    # Train
    print(
        f"[train] target_total_timesteps={args.total_timesteps}, "
        f"effective_learn_timesteps={args.total_timesteps}, device={args.device}"
    )
    start = time.time()
    callbacks_list = [
        SafeTensorCheckpointCallback(
            save_freq=args.checkpoint_freq,
            checkpoint_dir=args.checkpoint_dir,
            metadata_builder=lambda model_ref, timesteps: build_online_checkpoint_metadata(
                args,
                use_text=use_text,
                timesteps=timesteps,
                policy_kwargs=policy_kwargs,
                text_model_name=text_model_name,
            ),
            verbose=args.verbose,
        ),
        RewardBreakdownTensorboardCallback(),
        EncounterStatsCallback(),
    ]
    if args.combat_sandbox and args.eval_freq > 0:
        seen_eval_pool = encounter_pool or ([args.encounter_id] if args.encounter_id else [])
        callbacks_list.append(
            CombatFixedEvalCallback(
                session_file=args.session_file,
                character=args.character,
                use_text=use_text,
                seen_encounters=seen_eval_pool,
                holdout_encounters=eval_holdout_pool,
                reset_timeout_ms=args.reset_timeout_ms or 30000,
                step_timeout_ms=args.step_timeout_ms or 20000,
                eval_freq=args.eval_freq,
                episodes_per_encounter=args.eval_episodes_per_encounter,
                deterministic=not args.eval_stochastic,
                seed_base=args.eval_seed_base,
            )
        )
    callbacks = CallbackList(callbacks_list)
    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callbacks,
            reset_num_timesteps=not bool(args.init_checkpoint),
        )
        elapsed = time.time() - start
        print(f"[train] Done in {elapsed:.0f}s")

        final = Path(args.checkpoint_dir) / "final"
        save_online_checkpoint(
            model,
            final,
            metadata=build_online_checkpoint_metadata(
                args,
                use_text=use_text,
                timesteps=int(getattr(model, "num_timesteps", args.total_timesteps)),
                policy_kwargs=policy_kwargs,
                text_model_name=text_model_name,
            ),
        )
        print(f"[train] Saved {final}")
    except Exception:
        crash_timesteps = int(getattr(model, "num_timesteps", 0))
        crash_name = f"crash_{crash_timesteps:08d}_{time.strftime('%Y%m%d-%H%M%S')}"
        crash_dir = Path(args.checkpoint_dir) / crash_name
        try:
            save_online_checkpoint(
                model,
                crash_dir,
                metadata=build_online_checkpoint_metadata(
                    args,
                    use_text=use_text,
                    timesteps=crash_timesteps,
                    policy_kwargs=policy_kwargs,
                    text_model_name=text_model_name,
                ),
            )
            print(f"[train] Saved crash checkpoint {crash_dir}")
        except Exception as save_exc:
            print(f"[train] Failed to save crash checkpoint: {save_exc}")
        raise
    finally:
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
