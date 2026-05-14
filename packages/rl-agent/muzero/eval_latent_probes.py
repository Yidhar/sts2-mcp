"""Evaluate what MuZero latent state linearly encodes.

This is a lightweight diagnostic for the search-free/JEPА MuZero stack.  It
loads a checkpoint + replay buffer, encodes replay observations through the
representation/token-memory encoder, and trains closed-form ridge probes for
game-state facts that we expect the latent to carry:

- HP/block/energy and combat phase
- hand/draw/discard/exhaust sizes
- legal action/card/potion surface size
- token-zone counts for hand/draw/discard/exhaust/relic/potion/enemy/deck/route
- role-count signals such as enemy intent, powers, energy budget, and cycle plan

For token_memory_v1 it also fits per-memory-slot probes and reports the best
slot/bank per label.  No sklearn dependency is required.
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from muzero.sts2_env.muzero_buffer import _batched_observations_to_numpy
from muzero.sts2_env.muzero_model import MuZeroNetwork
from muzero.sts2_env.planner_memory_profile import (
    VALID_PLANNER_MEMORY_PROFILES,
    apply_planner_memory_profile_to_network,
)
from muzero.sts2_env.token_memory import MEMORY_BANK_NAMES, build_memory_slot_bank_ids
from sts2_env.observation_common import MAX_ACTIONS, NUM_PHASES, SCALAR_DIM
from sts2_env.observation_v3 import MAX_WORLD_TOKENS, TOKEN_ROLE_TO_ID, TOKEN_ZONE_TO_ID


SCALAR_LABEL_INDICES: dict[str, int] = {
    # phase/run offsets are laid out in observation_common.DictObservationEncoder._enc_scalars
    "hp_ratio": NUM_PHASES + 6,
    "hp_norm": NUM_PHASES + 7,
    "max_hp_norm": NUM_PHASES + 8,
    "block_norm": NUM_PHASES + 9,
    "gold_norm": NUM_PHASES + 10,
    "energy_ratio": NUM_PHASES + 11,
    "energy_norm": NUM_PHASES + 12,
    "stars_norm": NUM_PHASES + 13,
    "in_combat": NUM_PHASES + 14,
    "round_norm": NUM_PHASES + 15,
    "play_phase": NUM_PHASES + 16,
    "can_act": NUM_PHASES + 17,
    "hand_count_norm": NUM_PHASES + 18,
    "draw_count_norm": NUM_PHASES + 19,
    "discard_count_norm": NUM_PHASES + 20,
    "exhaust_count_norm": NUM_PHASES + 21,
    "deck_size_norm": NUM_PHASES + 32,
    "relic_count_norm": NUM_PHASES + 33,
    "potion_count_norm": NUM_PHASES + 34,
    "empty_potion_slots_norm": NUM_PHASES + 35,
    "potion_slots_norm": NUM_PHASES + 36,
    "potion_mechanics_available": NUM_PHASES + 37,
    "total_action_count_norm": NUM_PHASES + 38,
    "combat_continue_action_count_norm": NUM_PHASES + 39,
    "play_card_action_count_norm": NUM_PHASES + 40,
    "zero_cost_play_action_count_norm": NUM_PHASES + 41,
    "positive_preview_action_count_norm": NUM_PHASES + 42,
    "has_end_turn": NUM_PHASES + 43,
}

ZONE_COUNT_LABELS = (
    "PLAYER",
    "HAND",
    "DRAW",
    "DISCARD",
    "EXHAUST",
    "PLAY",
    "RELIC",
    "POTION",
    "ENEMY",
    "DECK",
    "REWARD",
    "SHOP",
    "UPGRADE",
    "ROUTE",
    "HISTORY",
)

ROLE_COUNT_LABELS = (
    "PLAYER_STATE",
    "RESOURCE",
    "THREAT",
    "OBJECTIVE",
    "RUN_MEMORY",
    "HAND_CARD",
    "DRAW_PILE",
    "DISCARD_PILE",
    "EXHAUST_PILE",
    "PLAY_PILE",
    "CYCLE_PLAN",
    "ENERGY_BUDGET",
    "RELIC_SUPPORT",
    "POTION_SUPPORT",
    "SUPPORT_GRAPH",
    "ENEMY_CORE",
    "ENEMY_INTENT",
    "ENEMY_POWER",
    "ENEMY_TRAIT",
    "ENEMY_REACTION",
    "DECK_CARD",
    "BUILD_STATE",
    "DECK_SYNERGY",
    "REWARD_OPTION",
    "SHOP_OPTION",
    "UPGRADE_OPTION",
    "ROUTE_SUMMARY",
    "ROUTE_NODE",
    "ROUTE_RISK",
    "ROUTE_VALUE",
    "POWER_SLOT",
    "CARD_KEYWORD",
    "HISTORY",
)


def _resolve_checkpoint_paths(checkpoint: str | None, network_state: str | None, replay_buffer: str | None) -> tuple[Path | None, Path | None, Path | None]:
    checkpoint_dir: Path | None = None
    network_path: Path | None = Path(network_state) if network_state else None
    replay_path: Path | None = Path(replay_buffer) if replay_buffer else None
    if checkpoint:
        checkpoint_path = Path(checkpoint)
        if checkpoint_path.is_file():
            network_path = network_path or checkpoint_path
            checkpoint_dir = checkpoint_path.parent
        else:
            checkpoint_dir = checkpoint_path
            network_path = network_path or checkpoint_path / "network.pt"
            replay_path = replay_path or checkpoint_path / "replay_buffer.pkl"
    return checkpoint_dir, network_path, replay_path


def _load_network(args: argparse.Namespace) -> MuZeroNetwork:
    checkpoint_dir, network_path, _ = _resolve_checkpoint_paths(args.checkpoint, args.network_state, args.replay_buffer)
    metadata: dict[str, Any] = {}
    if checkpoint_dir is not None and (checkpoint_dir / "metadata.json").exists():
        metadata = json.loads((checkpoint_dir / "metadata.json").read_text(encoding="utf-8"))

    spec = dict(metadata.get("network") or {})
    if not spec:
        spec = {
            "obs_mode": args.obs_mode,
            "model_arch": args.model_arch,
            "action_embed_dim": args.action_embed_dim,
            "support_size": args.support_size,
            "dynamics_res_blocks": args.dynamics_res_blocks,
            "token_d_model": args.token_d_model,
            "token_n_heads": args.token_n_heads,
            "token_ffn_dim": args.token_ffn_dim,
            "token_world_layers": args.token_world_layers,
            "token_local_layers": args.token_local_layers,
            "token_decoder_layers": args.token_decoder_layers,
            "token_candidate_set_layers": args.token_candidate_set_layers,
            "token_memory_slots": args.token_memory_slots,
            "token_memory_slot_layout": args.token_memory_slot_layout,
            "token_bank_token_slots": args.token_bank_token_slots,
            "token_world_bank_top_k": args.token_world_bank_top_k,
        }

    network = MuZeroNetwork(**spec).to(args.device)
    if network_path is not None and network_path.exists():
        state = torch.load(network_path, map_location=args.device)
        current = network.state_dict()
        compatible = {
            key: value
            for key, value in state.items()
            if key in current and tuple(current[key].shape) == tuple(value.shape)
        }
        merged = dict(current)
        merged.update(compatible)
        network.load_state_dict(merged)
        skipped = len(state) - len(compatible)
        missing = len(current) - len(compatible)
        if skipped or missing:
            print(f"[probe] Partial network load: loaded={len(compatible)} skipped={skipped} missing={missing}")
    else:
        print("[probe] No network state supplied; probing a freshly initialized network.")
    profile_settings = apply_planner_memory_profile_to_network(
        network,
        args.planner_memory_profile,
    )
    print(
        "[probe] planner_memory_profile="
        f"{profile_settings.profile} buckets={getattr(network, 'action_rollout_buckets', None)} "
        f"chunk={getattr(network, 'action_rollout_chunk_size', 0)}"
    )
    network.eval()
    return network


def _iter_replay_observations(replay_state: Any) -> Iterable[dict[str, Any]]:
    trajectories = replay_state.get("trajectories") if isinstance(replay_state, dict) else replay_state
    if trajectories is None:
        return
    for trajectory in trajectories:
        steps = getattr(trajectory, "steps", None)
        if steps is None and isinstance(trajectory, dict):
            steps = trajectory.get("steps")
        if not steps:
            continue
        for step in steps:
            if not isinstance(step, dict):
                continue
            obs = step.get("obs")
            if isinstance(obs, dict):
                yield obs


def _load_observation_sample(args: argparse.Namespace) -> list[dict[str, Any]]:
    _, _, replay_path = _resolve_checkpoint_paths(args.checkpoint, args.network_state, args.replay_buffer)
    if replay_path is None or not replay_path.exists():
        raise FileNotFoundError("Replay buffer not found. Pass --checkpoint <dir> or --replay-buffer <path>.")
    with replay_path.open("rb") as handle:
        replay_state = pickle.load(handle)

    rng = random.Random(args.seed)
    sample: list[dict[str, Any]] = []
    seen = 0
    limit = max(int(args.sample_limit), 1)
    for obs in _iter_replay_observations(replay_state):
        seen += 1
        if len(sample) < limit:
            sample.append(obs)
            continue
        replace_idx = rng.randrange(seen)
        if replace_idx < limit:
            sample[replace_idx] = obs
    if not sample:
        raise RuntimeError(f"No observations found in replay buffer: {replay_path}")
    print(f"[probe] sampled_observations={len(sample)} seen={seen} replay={replay_path}")
    return sample


def _extract_labels(obs_np: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    labels: dict[str, np.ndarray] = {}
    scalars = obs_np.get("scalars")
    if scalars is not None and scalars.ndim == 2 and scalars.shape[1] >= SCALAR_DIM:
        for name, index in SCALAR_LABEL_INDICES.items():
            if 0 <= index < scalars.shape[1]:
                labels[name] = scalars[:, index].astype(np.float32, copy=False)

    action_mask = obs_np.get("action_mask")
    if action_mask is not None:
        legal_count = (action_mask > 0.5).sum(axis=1).astype(np.float32)
        labels["legal_action_count_norm"] = legal_count / float(max(MAX_ACTIONS, 1))
        labels["has_legal_action"] = (legal_count > 0).astype(np.float32)

    world_mask = obs_np.get("world_token_mask")
    zone_ids = obs_np.get("world_token_zone_ids")
    if world_mask is not None and zone_ids is not None:
        active = world_mask > 0.5
        for zone_name in ZONE_COUNT_LABELS:
            zone_id = TOKEN_ZONE_TO_ID.get(zone_name)
            if zone_id is None:
                continue
            count = ((zone_ids.astype(np.int64) == int(zone_id)) & active).sum(axis=1).astype(np.float32)
            labels[f"zone_{zone_name.lower()}_count_norm"] = count / float(max(MAX_WORLD_TOKENS, 1))
            labels[f"has_zone_{zone_name.lower()}"] = (count > 0).astype(np.float32)

    role_ids = obs_np.get("world_token_role_ids")
    if world_mask is not None and role_ids is not None:
        active = world_mask > 0.5
        for role_name in ROLE_COUNT_LABELS:
            role_id = TOKEN_ROLE_TO_ID.get(role_name)
            if role_id is None:
                continue
            count = ((role_ids.astype(np.int64) == int(role_id)) & active).sum(axis=1).astype(np.float32)
            labels[f"role_{role_name.lower()}_count_norm"] = count / float(max(MAX_WORLD_TOKENS, 1))
            labels[f"has_role_{role_name.lower()}"] = (count > 0).astype(np.float32)
    return labels


def _encode_hidden_and_labels(
    network: MuZeroNetwork,
    observations: list[dict[str, Any]],
    *,
    batch_size: int,
    device: str,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    hidden_chunks: list[torch.Tensor] = []
    label_chunks: dict[str, list[torch.Tensor]] = {}
    for start in range(0, len(observations), batch_size):
        obs_chunk = observations[start : start + batch_size]
        obs_np = _batched_observations_to_numpy(obs_chunk)
        obs_torch = {
            key: torch.as_tensor(value, dtype=torch.float32, device=device)
            for key, value in obs_np.items()
        }
        with torch.no_grad():
            hidden = network.initial_inference(obs_torch).hidden_state.detach().cpu()
        hidden_chunks.append(hidden.float())
        for name, values in _extract_labels(obs_np).items():
            label_chunks.setdefault(name, []).append(torch.from_numpy(values.astype(np.float32, copy=False)))
    hidden_all = torch.cat(hidden_chunks, dim=0)
    labels_all = {name: torch.cat(chunks, dim=0).float() for name, chunks in label_chunks.items()}
    return hidden_all, labels_all


def _ridge_probe(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    ridge: float,
    train_fraction: float,
    seed: int,
) -> dict[str, float]:
    finite = torch.isfinite(y)
    x = x[finite].float()
    y = y[finite].float().reshape(-1, 1)
    n = int(y.shape[0])
    if n < 8 or float(y.std(unbiased=False).item()) < 1e-6:
        return {"n": float(n), "r2": 0.0, "mae": 0.0, "acc": 0.0, "valid": 0.0}

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    perm = torch.randperm(n, generator=generator)
    train_n = min(max(int(n * train_fraction), 4), n - 2)
    train_idx = perm[:train_n]
    test_idx = perm[train_n:]
    x_train = x[train_idx]
    x_test = x[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    x_mean = x_train.mean(dim=0, keepdim=True)
    x_std = x_train.std(dim=0, keepdim=True, unbiased=False).clamp(min=1e-4)
    x_train = (x_train - x_mean) / x_std
    x_test = (x_test - x_mean) / x_std
    x_train = torch.cat([x_train, torch.ones((x_train.shape[0], 1), dtype=x_train.dtype)], dim=1)
    x_test = torch.cat([x_test, torch.ones((x_test.shape[0], 1), dtype=x_test.dtype)], dim=1)

    eye = torch.eye(x_train.shape[1], dtype=x_train.dtype)
    eye[-1, -1] = 0.0
    xtx = x_train.T @ x_train + float(ridge) * eye
    xty = x_train.T @ y_train
    try:
        weights = torch.linalg.solve(xtx, xty)
    except RuntimeError:
        weights = torch.linalg.pinv(xtx) @ xty

    pred = x_test @ weights
    mse = torch.mean((pred - y_test) ** 2)
    var = torch.var(y_test, unbiased=False).clamp(min=1e-8)
    r2 = 1.0 - mse / var
    mae = torch.mean((pred - y_test).abs())

    y_unique = torch.unique((y > 0.5).float())
    acc = torch.tensor(0.0)
    if y_unique.numel() == 2:
        acc = (((pred >= 0.5).float() == (y_test >= 0.5).float()).float().mean())

    return {
        "n": float(n),
        "r2": float(r2.item()),
        "mae": float(mae.item()),
        "acc": float(acc.item()),
        "valid": 1.0,
    }


def _format_markdown(rows: list[dict[str, Any]], *, sample_count: int, hidden_dim: int, slot_banks: list[str]) -> str:
    lines = [
        "# MuZero latent probe report",
        "",
        f"- samples: `{sample_count}`",
        f"- hidden_dim: `{hidden_dim}`",
    ]
    if slot_banks:
        lines.append(f"- slot banks: `{', '.join(slot_banks)}`")
    lines.extend(
        [
            "",
            "| label | R2 | MAE | binary acc | best slot | best slot bank | best slot R2 |",
            "|---|---:|---:|---:|---:|---|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| {label} | {r2:.4f} | {mae:.4f} | {acc:.4f} | {best_slot} | {best_slot_bank} | {best_slot_r2:.4f} |".format(
                label=row["label"],
                r2=float(row["r2"]),
                mae=float(row["mae"]),
                acc=float(row.get("acc", 0.0)),
                best_slot=row.get("best_slot", ""),
                best_slot_bank=row.get("best_slot_bank", ""),
                best_slot_r2=float(row.get("best_slot_r2", 0.0)),
            )
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Linear probes for MuZero latent/world-memory states.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint directory containing metadata/network/replay_buffer.")
    parser.add_argument("--network-state", type=str, default=None, help="Optional explicit network.pt path.")
    parser.add_argument("--replay-buffer", type=str, default=None, help="Optional explicit replay_buffer.pkl path.")
    parser.add_argument("--obs-mode", type=str, default="token_v3", choices=["dense_v2", "token_v3"])
    parser.add_argument("--model-arch", type=str, default="token_memory_v1", choices=["dense_v1", "token_memory_v1"])
    parser.add_argument("--action-embed-dim", type=int, default=64)
    parser.add_argument("--support-size", type=int, default=25)
    parser.add_argument("--dynamics-res-blocks", type=int, default=4)
    parser.add_argument("--token-d-model", type=int, default=128)
    parser.add_argument("--token-n-heads", type=int, default=4)
    parser.add_argument("--token-ffn-dim", type=int, default=512)
    parser.add_argument("--token-world-layers", type=int, default=4)
    parser.add_argument("--token-local-layers", type=int, default=1)
    parser.add_argument("--token-decoder-layers", type=int, default=2)
    parser.add_argument("--token-candidate-set-layers", type=int, default=1)
    parser.add_argument("--token-memory-slots", type=int, default=8)
    parser.add_argument(
        "--token-memory-slot-layout",
        type=str,
        default="legacy",
        choices=["legacy", "quota_v1", "pass_large_v1"],
    )
    parser.add_argument("--token-bank-token-slots", type=int, default=4)
    parser.add_argument("--token-world-bank-top-k", type=int, default=3)
    parser.add_argument(
        "--planner-memory-profile",
        type=str,
        default="checkpoint",
        choices=VALID_PLANNER_MEMORY_PROFILES,
        help="Override checkpoint planner bucket/chunk knobs for probe inference.",
    )
    parser.add_argument("--sample-limit", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--ridge", type=float, default=10.0)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=str, default="", help="Optional JSON output path.")
    parser.add_argument("--markdown-output", type=str, default="", help="Optional Markdown report path.")
    args = parser.parse_args()

    args.device = "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    observations = _load_observation_sample(args)
    network = _load_network(args)
    hidden, labels = _encode_hidden_and_labels(
        network,
        observations,
        batch_size=max(int(args.batch_size), 1),
        device=args.device,
    )

    slot_bank_names: list[str] = []
    slot_tensors: list[torch.Tensor] = []
    if getattr(network, "is_token_mode", False):
        prediction = getattr(network, "prediction", None)
        slot_count = int(getattr(prediction, "num_memory_slots", 0) or args.token_memory_slots)
        d_model = int(getattr(prediction, "d_model", 0) or args.token_d_model)
        if slot_count > 0 and d_model > 0 and hidden.shape[-1] == slot_count * d_model:
            slot_layout = str(getattr(prediction, "memory_slot_layout", args.token_memory_slot_layout))
            slot_ids = build_memory_slot_bank_ids(slot_count, layout=slot_layout)
            slot_bank_names = [MEMORY_BANK_NAMES[idx] if 0 <= idx < len(MEMORY_BANK_NAMES) else "unknown" for idx in slot_ids]
            slots = hidden.reshape(hidden.shape[0], slot_count, d_model)
            slot_tensors = [slots[:, idx, :] for idx in range(slot_count)]

    rows: list[dict[str, Any]] = []
    for label_name, y in sorted(labels.items()):
        metrics = _ridge_probe(
            hidden,
            y,
            ridge=float(args.ridge),
            train_fraction=float(args.train_fraction),
            seed=int(args.seed),
        )
        if metrics["valid"] <= 0.0:
            continue
        row: dict[str, Any] = {
            "label": label_name,
            **metrics,
            "best_slot": "",
            "best_slot_bank": "",
            "best_slot_r2": 0.0,
        }
        if slot_tensors:
            best_slot = -1
            best_slot_r2 = -1e9
            for slot_idx, slot_x in enumerate(slot_tensors):
                slot_metrics = _ridge_probe(
                    slot_x,
                    y,
                    ridge=float(args.ridge),
                    train_fraction=float(args.train_fraction),
                    seed=int(args.seed),
                )
                if slot_metrics["valid"] > 0.0 and slot_metrics["r2"] > best_slot_r2:
                    best_slot = slot_idx
                    best_slot_r2 = float(slot_metrics["r2"])
            if best_slot >= 0:
                row["best_slot"] = best_slot
                row["best_slot_bank"] = slot_bank_names[best_slot] if best_slot < len(slot_bank_names) else ""
                row["best_slot_r2"] = best_slot_r2
        rows.append(row)

    rows.sort(key=lambda item: float(item["r2"]), reverse=True)
    report = {
        "sample_count": int(hidden.shape[0]),
        "hidden_dim": int(hidden.shape[1]),
        "slot_banks": slot_bank_names,
        "rows": rows,
    }
    markdown = _format_markdown(
        rows,
        sample_count=int(hidden.shape[0]),
        hidden_dim=int(hidden.shape[1]),
        slot_banks=slot_bank_names,
    )
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[probe] wrote JSON: {args.output}")
    if args.markdown_output:
        Path(args.markdown_output).write_text(markdown, encoding="utf-8")
        print(f"[probe] wrote Markdown: {args.markdown_output}")
    print(markdown)


if __name__ == "__main__":
    main()
