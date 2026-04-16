"""Train a small combat expert with fixed-template exact-state search.

This is intentionally separate from the full-run MuZero stack:

- combat sandbox only
- fixed position-based combat action template
- exact-state search via reset + action replay
- small policy/value network
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from combat_snapshot_dataset import CombatSnapshotPool
from sts2_env.combat_exact_search import ExactCombatSearch
from sts2_env.combat_fixed_action import END_TURN_SLOT, NUM_FIXED_COMBAT_ACTIONS
from sts2_env.combat_small_model import SmallCombatPolicyValueNet
from sts2_env.combat_template_env import COMBAT_STATE_VECTOR_DIM, CombatTemplateEnv


@dataclass
class EpisodeStep:
    state: np.ndarray
    action_mask: np.ndarray
    policy_target: np.ndarray
    reward: float
    done: bool
    selected_action: int


@dataclass
class ReplayItem:
    state: np.ndarray
    action_mask: np.ndarray
    policy_target: np.ndarray
    value_target: float
    selected_action: int


def _timestamp() -> str:
    return time.strftime("%H:%M:%S")


def _parse_session_files(args: argparse.Namespace) -> list[str | None]:
    if args.session_files:
        values = [value.strip() for value in str(args.session_files).split(",")]
        return [value for value in values if value]
    if args.session_file:
        return [str(args.session_file)]
    return [None]


def _make_env_kwargs(
    args: argparse.Namespace,
    snapshot_pool: CombatSnapshotPool,
    *,
    session_file: str | None,
) -> dict[str, Any]:
    return {
        "session_file": session_file,
        "snapshot_pool": snapshot_pool,
        "reset_timeout_ms": args.reset_timeout_ms,
        "step_timeout_ms": args.step_timeout_ms,
        "include_debug_info": False,
    }


def _build_env(
    args: argparse.Namespace,
    snapshot_pool: CombatSnapshotPool,
    *,
    session_file: str | None,
) -> CombatTemplateEnv:
    return CombatTemplateEnv(**_make_env_kwargs(args, snapshot_pool, session_file=session_file))


def _masked_policy_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    action_mask: torch.Tensor,
) -> torch.Tensor:
    masked_logits = logits.masked_fill(action_mask <= 0, -1e9)
    log_probs = F.log_softmax(masked_logits, dim=-1)
    return -(targets * log_probs).sum(dim=-1).mean()


def _sample_batch(
    replay: deque[ReplayItem],
    *,
    batch_size: int,
    rng: random.Random,
) -> list[ReplayItem]:
    if len(replay) <= batch_size:
        return list(replay)
    indices = rng.sample(range(len(replay)), batch_size)
    return [replay[index] for index in indices]


def _save_checkpoint(
    checkpoint_dir: Path,
    *,
    step: int,
    episode: int,
    network: SmallCombatPolicyValueNet,
    optimizer: torch.optim.Optimizer,
    metadata: dict[str, Any],
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(network.state_dict(), checkpoint_dir / "network.pt")
    torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
    payload = {
        "step": int(step),
        "episode": int(episode),
        "state_dim": int(network.state_dim),
        "num_actions": int(network.num_actions),
        "hidden_dim": int(network.hidden_dim),
        "action_embed_dim": int(network.action_embed_dim),
        "metadata": metadata,
    }
    (checkpoint_dir / "metadata.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_checkpoint(
    checkpoint_dir: Path,
    *,
    network: SmallCombatPolicyValueNet,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[int, int]:
    metadata_path = checkpoint_dir / "metadata.json"
    if not metadata_path.exists():
        return 0, 0
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    network.load_state_dict(torch.load(checkpoint_dir / "network.pt", map_location=device))
    optimizer.load_state_dict(torch.load(checkpoint_dir / "optimizer.pt", map_location=device))
    return int(payload.get("step", 0)), int(payload.get("episode", 0))


def _compute_episode_targets(
    episode_steps: list[EpisodeStep],
    *,
    gamma: float,
    value_scale: float,
) -> list[ReplayItem]:
    results: list[ReplayItem] = []
    return_accumulator = 0.0
    for step in reversed(episode_steps):
        return_accumulator = float(step.reward) + (gamma * return_accumulator)
        value_target = float(np.tanh(return_accumulator / max(value_scale, 1e-6)))
        results.append(
            ReplayItem(
                state=step.state.astype(np.float32, copy=True),
                action_mask=step.action_mask.astype(np.float32, copy=True),
                policy_target=step.policy_target.astype(np.float32, copy=True),
                value_target=value_target,
                selected_action=int(step.selected_action),
            )
        )
        if step.done:
            return_accumulator = 0.0
    results.reverse()
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Train fixed-template combat expert")
    parser.add_argument("--combat-snapshot-dataset", required=True, help="Combat snapshot dataset root/file")
    parser.add_argument("--curated-subset", default="bootstrap_human_plus_local_act1clear_weak_normal_roomwin_only")
    parser.add_argument("--character", default=None, help="Optional single-character filter, e.g. CHARACTER.DEFECT")
    parser.add_argument("--exclude-characters", default="CHARACTER.WATCHER", help="Comma-separated character ids to exclude from the training pool")
    parser.add_argument("--encounter-tiers", default="weak,normal", help="Comma-separated tier filter")
    parser.add_argument("--sample-mode", default="tier_weighted_encounter_balanced", choices=["row_uniform", "encounter_balanced", "tier_weighted_encounter_balanced"])
    parser.add_argument("--tier-weights", default="weak:1.0,normal:1.0,elite:0.4,boss:0.2")
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--session-file", default=None)
    parser.add_argument("--session-files", default=None, help="Comma-separated bridge session files for multi-instance combat training")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--action-embed-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--replay-size", type=int, default=50_000)
    parser.add_argument("--warmup-items", type=int, default=512)
    parser.add_argument("--updates-per-episode", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=0.997)
    parser.add_argument("--value-scale", type=float, default=5.0)
    parser.add_argument("--entropy-coef", type=float, default=0.005)
    parser.add_argument("--num-simulations", type=int, default=0)
    parser.add_argument("--search-depth", type=int, default=0)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--total-episodes", type=int, default=200)
    parser.add_argument("--checkpoint-dir", default="checkpoints/combat-expert-latest")
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--reset-timeout-ms", type=int, default=15000)
    parser.add_argument("--step-timeout-ms", type=int, default=20000)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    encounter_tiers = [value.strip().lower() for value in str(args.encounter_tiers).split(",") if value.strip()]
    excluded_characters = [value.strip() for value in str(args.exclude_characters).split(",") if value.strip()]
    tier_weights: dict[str, float] = {}
    for token in [value.strip() for value in str(args.tier_weights).split(",") if value.strip()]:
        if ":" not in token:
            continue
        key, value = token.split(":", 1)
        try:
            tier_weights[key.strip().lower()] = float(value)
        except ValueError:
            continue

    snapshot_pool = CombatSnapshotPool.from_path(
        args.combat_snapshot_dataset,
        curated_subset=args.curated_subset,
        character=(str(args.character).strip() if args.character else None),
        encounter_tiers=encounter_tiers or None,
        max_rows=(args.max_rows if args.max_rows > 0 else None),
        sample_mode=args.sample_mode,
        tier_weights=tier_weights or None,
        excluded_characters=excluded_characters,
    )
    session_files = _parse_session_files(args)

    network = SmallCombatPolicyValueNet(
        state_dim=COMBAT_STATE_VECTOR_DIM,
        num_actions=NUM_FIXED_COMBAT_ACTIONS,
        hidden_dim=args.hidden_dim,
        action_embed_dim=args.action_embed_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(
        network.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    checkpoint_dir = Path(args.checkpoint_dir)
    total_steps = 0
    completed_episodes = 0
    if args.resume and checkpoint_dir.exists():
        total_steps, completed_episodes = _load_checkpoint(
            checkpoint_dir,
            network=network,
            optimizer=optimizer,
            device=device,
        )
        print(f"[{_timestamp()}] resumed checkpoint {checkpoint_dir} steps={total_steps} episodes={completed_episodes}", flush=True)

    replay: deque[ReplayItem] = deque(maxlen=args.replay_size)

    envs = [
        _build_env(args, snapshot_pool, session_file=session_file)
        for session_file in session_files
    ]
    searchers = [
        ExactCombatSearch(
            network,
            env_factory=lambda session_file=session_file: _build_env(args, snapshot_pool, session_file=session_file),
            device=str(device),
            gamma=args.gamma,
            num_simulations=args.num_simulations,
            max_depth=args.search_depth,
            c_puct=args.c_puct,
            value_scale=args.value_scale,
        )
        for session_file in session_files
    ]

    active_states: list[np.ndarray | None] = [None for _ in envs]
    active_infos: list[dict[str, Any] | None] = [None for _ in envs]
    active_episodes: list[list[EpisodeStep]] = [[] for _ in envs]
    episode_returns: list[float] = [0.0 for _ in envs]
    episode_lengths: list[int] = [0 for _ in envs]

    print(
        f"[{_timestamp()}] combat_expert state_dim={COMBAT_STATE_VECTOR_DIM} actions={NUM_FIXED_COMBAT_ACTIONS} "
        f"hidden={args.hidden_dim} sims={args.num_simulations} depth={args.search_depth} "
        f"mode={'root_prior' if args.num_simulations <= 0 or args.search_depth <= 0 else 'exact_search'} "
        f"envs={len(envs)} dataset_rows={len(snapshot_pool)} tiers={encounter_tiers} sample_mode={args.sample_mode} "
        f"character={args.character or 'ALL'} excluded_characters={excluded_characters}",
        flush=True,
    )

    try:
        while completed_episodes < args.total_episodes:
            for env_index, env in enumerate(envs):
                if completed_episodes >= args.total_episodes:
                    break

                if active_states[env_index] is None:
                    obs, info = env.reset()
                    active_states[env_index] = obs
                    active_infos[env_index] = info
                    active_episodes[env_index] = []
                    episode_returns[env_index] = 0.0
                    episode_lengths[env_index] = 0

                searcher = searchers[env_index]
                selected_action, policy_target, search_metrics = searcher.run(env, add_exploration_noise=True)
                next_obs, reward, terminated, truncated, info = env.step(selected_action)
                done = bool(terminated or truncated)

                active_episodes[env_index].append(
                    EpisodeStep(
                        state=np.asarray(active_states[env_index], dtype=np.float32).copy(),
                        action_mask=np.asarray(active_infos[env_index]["action_mask"], dtype=np.float32).copy(),
                        policy_target=np.asarray(policy_target, dtype=np.float32).copy(),
                        reward=float(reward),
                        done=done,
                        selected_action=int(selected_action),
                    )
                )
                episode_returns[env_index] += float(reward)
                episode_lengths[env_index] += 1
                total_steps += 1

                if done:
                    items = _compute_episode_targets(
                        active_episodes[env_index],
                        gamma=args.gamma,
                        value_scale=args.value_scale,
                    )
                    replay.extend(items)
                    completed_episodes += 1

                    metrics_text = "warmup"
                    if len(replay) >= args.warmup_items and args.updates_per_episode > 0:
                        network.train()
                        losses: list[tuple[float, float, float]] = []
                        for _ in range(args.updates_per_episode):
                            batch_items = _sample_batch(replay, batch_size=args.batch_size, rng=rng)
                            states = torch.as_tensor(np.stack([item.state for item in batch_items]), dtype=torch.float32, device=device)
                            masks = torch.as_tensor(np.stack([item.action_mask for item in batch_items]), dtype=torch.float32, device=device)
                            policy_targets = torch.as_tensor(np.stack([item.policy_target for item in batch_items]), dtype=torch.float32, device=device)
                            value_targets = torch.as_tensor([item.value_target for item in batch_items], dtype=torch.float32, device=device)

                            logits, values = network(states)
                            policy_loss = _masked_policy_loss(logits, policy_targets, masks)
                            value_loss = F.mse_loss(values, value_targets)
                            masked_logits = logits.masked_fill(masks <= 0, -1e9)
                            probs = torch.softmax(masked_logits, dim=-1)
                            entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=-1).mean()
                            loss = policy_loss + value_loss - (args.entropy_coef * entropy)

                            optimizer.zero_grad(set_to_none=True)
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
                            optimizer.step()
                            losses.append((float(loss.item()), float(policy_loss.item()), float(value_loss.item())))
                        mean_total = sum(item[0] for item in losses) / len(losses)
                        mean_policy = sum(item[1] for item in losses) / len(losses)
                        mean_value = sum(item[2] for item in losses) / len(losses)
                        metrics_text = f"loss={mean_total:.4f} policy={mean_policy:.4f} value={mean_value:.4f}"

                    end_turn_rate = 0.0
                    if active_episodes[env_index]:
                        end_turn_rate = sum(1 for item in active_episodes[env_index] if item.selected_action == END_TURN_SLOT) / float(len(active_episodes[env_index]))

                    print(
                        f"[{_timestamp()}] episode={completed_episodes} env={env_index} "
                        f"steps={episode_lengths[env_index]} reward={episode_returns[env_index]:+.3f} "
                        f"replay={len(replay)} end_turn_rate={end_turn_rate:.3f} "
                        f"root_value={search_metrics.get('root_value', 0.0):+.3f} "
                        f"legal={search_metrics.get('root_legal_count', 0.0):.0f} "
                        f"searched_nodes={search_metrics.get('searched_nodes', 0.0):.0f} "
                        f"root_only={search_metrics.get('root_only', 0.0):.0f} "
                        f"sim_error={search_metrics.get('simulation_error', 0.0):.0f} "
                        f"{metrics_text}",
                        flush=True,
                    )

                    if args.checkpoint_interval > 0 and completed_episodes % args.checkpoint_interval == 0:
                        _save_checkpoint(
                            checkpoint_dir,
                            step=total_steps,
                            episode=completed_episodes,
                            network=network,
                            optimizer=optimizer,
                            metadata={
                                "type": "combat_expert",
                                "state_dim": COMBAT_STATE_VECTOR_DIM,
                                "num_actions": NUM_FIXED_COMBAT_ACTIONS,
                                "num_simulations": args.num_simulations,
                                "search_depth": args.search_depth,
                                "dataset_summary": snapshot_pool.summary(),
                                "session_files": session_files,
                                "character": args.character,
                                "excluded_characters": excluded_characters,
                            },
                        )
                        print(f"[{_timestamp()}] checkpoint saved to {checkpoint_dir}", flush=True)

                    active_states[env_index] = None
                    active_infos[env_index] = None
                    active_episodes[env_index] = []
                else:
                    active_states[env_index] = next_obs
                    active_infos[env_index] = info

    finally:
        for env in envs:
            env.close()

    _save_checkpoint(
        checkpoint_dir,
        step=total_steps,
        episode=completed_episodes,
        network=network,
        optimizer=optimizer,
        metadata={
            "type": "combat_expert",
            "state_dim": COMBAT_STATE_VECTOR_DIM,
            "num_actions": NUM_FIXED_COMBAT_ACTIONS,
            "num_simulations": args.num_simulations,
            "search_depth": args.search_depth,
            "dataset_summary": snapshot_pool.summary(),
            "session_files": session_files,
            "character": args.character,
            "excluded_characters": excluded_characters,
        },
    )
    print(f"[{_timestamp()}] final checkpoint saved to {checkpoint_dir}", flush=True)


if __name__ == "__main__":
    main()
