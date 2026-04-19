"""Behavior-cloning trainer on Skada human decision data.

Streams samples from data/skada_bc/samples.jsonl, translates each to the
bridge obs schema + legal_actions, encodes with WorldTokenObservationEncoder,
forwards through STS2OmniAttentionPolicy, computes cross-entropy on the
policy's action logits (masked by legal_actions count), backprops.

Saves checkpoints compatible with PPO warmstart — same safetensors + metadata
format as train_attention_policy.py.

Usage:
    python skada_bc_train.py \
        --samples data/skada_bc/samples.jsonl \
        --checkpoint-dir checkpoints_attention/bc_pretrain_YYYYMMDD \
        --total-samples 2000000 \
        --batch-size 32 --num-workers 4 \
        --device cuda
"""
from __future__ import annotations

import argparse
import collections
import json
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import spaces
from safetensors.torch import save_file
from torch.utils.data import DataLoader, IterableDataset

from skada_bc_translate import translate_bc_sample
from sts2_env.observation_v3 import (
    MAX_ACTIONS, OBSERVATION_API_VERSION, WorldTokenObservationEncoder,
)
from sts2_env.omni_attention_policy import STS2OmniAttentionPolicy


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SkadaBcDataset(IterableDataset):
    """Streams samples.jsonl, translates + encodes per-worker.

    Each DataLoader worker gets a disjoint stride of lines (round-robin mod
    num_workers) so no sample is seen twice per epoch. Workers instantiate
    their own WorldTokenObservationEncoder (stateless beyond registry cache).
    """

    def __init__(
        self,
        samples_path: Path,
        *,
        max_samples: int | None = None,
        shuffle_buffer: int = 0,  # 0 = no shuffle (streaming order)
    ) -> None:
        self.samples_path = Path(samples_path)
        self.max_samples = max_samples
        self.shuffle_buffer = shuffle_buffer

    def _get_worker_shard(self) -> tuple[int, int]:
        info = torch.utils.data.get_worker_info()
        if info is None:
            return 0, 1
        return int(info.id), int(info.num_workers)

    def __iter__(self) -> Iterator[dict]:
        worker_id, num_workers = self._get_worker_shard()
        # Each worker lazily constructs its own encoder — the object holds no
        # training state, so duplicating per-worker is cheap and avoids
        # cross-process serialization.
        encoder = WorldTokenObservationEncoder(use_text=False)
        count = 0
        yielded = 0
        buf: list[dict] = []

        with open(self.samples_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                if line_idx % num_workers != worker_id:
                    continue
                line = line.strip()
                if not line:
                    continue
                count += 1
                try:
                    sample = json.loads(line)
                except Exception:
                    continue
                translated = translate_bc_sample(sample)
                if translated is None:
                    continue
                obs, legal_actions, chosen_idx = translated
                n_legal = len(legal_actions)
                if n_legal == 0 or not (0 <= chosen_idx < n_legal):
                    continue
                try:
                    encoded = encoder.encode(obs, legal_actions, planner_context={})
                except Exception:
                    continue

                # Build action_mask from legal_actions count.
                action_mask = np.zeros(MAX_ACTIONS, dtype=np.float32)
                action_mask[:n_legal] = 1.0

                item = {
                    "obs": {k: np.asarray(v) for k, v in encoded.items()},
                    "action_mask": action_mask,
                    "target": np.int64(chosen_idx),
                    "phase": sample.get("phase", "?"),
                    "character": sample.get("character", "?"),
                }

                if self.shuffle_buffer > 0:
                    buf.append(item)
                    if len(buf) >= self.shuffle_buffer:
                        np.random.shuffle(buf)
                        for x in buf:
                            yield x
                            yielded += 1
                            if self.max_samples and yielded >= self.max_samples:
                                return
                        buf.clear()
                else:
                    yield item
                    yielded += 1
                    if self.max_samples and yielded >= self.max_samples:
                        return

        # Flush shuffle buffer
        if buf:
            np.random.shuffle(buf)
            for x in buf:
                yield x
                yielded += 1
                if self.max_samples and yielded >= self.max_samples:
                    return


def collate_batch(batch: list[dict]) -> dict:
    """Stack obs tensors and metadata into per-key batch dicts."""
    obs_keys = list(batch[0]["obs"].keys())
    obs_batch = {
        k: torch.from_numpy(np.stack([b["obs"][k] for b in batch]))
        for k in obs_keys
    }
    action_mask = torch.from_numpy(np.stack([b["action_mask"] for b in batch]))
    targets = torch.from_numpy(np.stack([b["target"] for b in batch]))
    phases = [b["phase"] for b in batch]
    characters = [b["character"] for b in batch]
    return {
        "obs": obs_batch,
        "action_mask": action_mask,
        "target": targets,
        "phase": phases,
        "character": characters,
    }


# ---------------------------------------------------------------------------
# Policy construction
# ---------------------------------------------------------------------------

def build_policy(device: torch.device, *, lr: float = 3e-4) -> STS2OmniAttentionPolicy:
    obs_encoder = WorldTokenObservationEncoder(use_text=False)
    observation_space = obs_encoder.obs_space
    action_space = spaces.Discrete(MAX_ACTIONS)
    # Minimal policy_kwargs — keep defaults for d_model/n_heads/layers.
    policy = STS2OmniAttentionPolicy(
        observation_space=observation_space,
        action_space=action_space,
        lr_schedule=lambda _: lr,
    )
    policy.to(device)
    return policy


def move_obs_to_device(obs: dict, device: torch.device) -> dict:
    return {k: v.to(device, non_blocking=True) for k, v in obs.items()}


# ---------------------------------------------------------------------------
# Loss + metrics
# ---------------------------------------------------------------------------

def bc_loss_and_accuracy(
    policy: STS2OmniAttentionPolicy,
    obs_batch: dict[str, torch.Tensor],
    action_mask: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cross-entropy over masked action logits + top-1 accuracy."""
    # MaskableActorCriticPolicy.evaluate_actions returns (values, log_probs,
    # entropy). log_prob is the log-prob of the TARGET action — that's
    # exactly what BC wants: maximize log_prob(target) = minimize -log_prob.
    _, log_probs, entropy = policy.evaluate_actions(
        obs_batch, targets, action_masks=action_mask.bool()
    )
    bc_loss = -log_probs.mean()

    # For accuracy, we need the argmax over masked logits. Pull the
    # distribution's raw logits (post-masking).
    with torch.no_grad():
        dist = policy.get_distribution(obs_batch, action_masks=action_mask.bool())
        # MaskableCategoricalDistribution: .distribution is torch Categorical
        # whose logits are already masked (invalid actions → -inf).
        logits = dist.distribution.logits
        preds = logits.argmax(dim=-1)
        correct = (preds == targets).float()
    return bc_loss, correct, entropy


# ---------------------------------------------------------------------------
# Checkpoint save
# ---------------------------------------------------------------------------

def save_bc_checkpoint(
    policy: STS2OmniAttentionPolicy,
    output_dir: Path,
    metadata: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    save_file(policy.state_dict(), str(output_dir / "model.safetensors"))
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", default="data/skada_bc/samples.jsonl")
    parser.add_argument("--checkpoint-dir", default="checkpoints_attention/bc_pretrain")
    parser.add_argument("--total-samples", type=int, default=0,
                        help="cap total samples processed (0 = full epoch)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--shuffle-buffer", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp-dtype", default="bf16", choices=("none", "fp16", "bf16"))
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=50,
                        help="batches between progress prints")
    parser.add_argument("--checkpoint-interval", type=int, default=5000,
                        help="batches between periodic checkpoints")
    parser.add_argument("--max-batches", type=int, default=0,
                        help="stop after N batches (0 = full dataset)")
    args = parser.parse_args()

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[bc-train] device={device}  amp={args.amp_dtype}")

    policy = build_policy(device, lr=args.lr)
    policy.train()
    optimizer = policy.optimizer
    amp_dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16}
    use_amp = args.amp_dtype != "none"
    amp_dtype = amp_dtype_map.get(args.amp_dtype)

    dataset = SkadaBcDataset(
        Path(args.samples),
        max_samples=(args.total_samples or None),
        shuffle_buffer=args.shuffle_buffer,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    loss_ema: float = 0.0
    acc_ema: float = 0.0
    phase_acc: dict[str, collections.deque] = collections.defaultdict(lambda: collections.deque(maxlen=256))
    t_start = time.time()
    batches_done = 0
    samples_done = 0

    for batch in loader:
        obs = move_obs_to_device(batch["obs"], device)
        mask = batch["action_mask"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with torch.autocast(device_type=device.type, dtype=amp_dtype):
                loss, correct, entropy = bc_loss_and_accuracy(policy, obs, mask, targets)
        else:
            loss, correct, entropy = bc_loss_and_accuracy(policy, obs, mask, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
        optimizer.step()

        loss_val = float(loss.detach().item())
        acc_val = float(correct.mean().item())
        loss_ema = 0.98 * loss_ema + 0.02 * loss_val if batches_done else loss_val
        acc_ema = 0.98 * acc_ema + 0.02 * acc_val if batches_done else acc_val
        for phase, c in zip(batch["phase"], correct.cpu().tolist()):
            phase_acc[phase].append(c)

        batches_done += 1
        samples_done += int(targets.shape[0])

        if batches_done % args.log_interval == 0:
            elapsed = time.time() - t_start
            sps = samples_done / max(elapsed, 1e-6)
            phase_summary = "  ".join(
                f"{p}={sum(q)/max(len(q),1)*100:.0f}%"
                for p, q in sorted(phase_acc.items())
            )
            print(
                f"[{elapsed:7.1f}s] batch={batches_done:6d} "
                f"samples={samples_done:>8,}  loss={loss_ema:.4f}  acc={acc_ema*100:.1f}%  "
                f"ent={float(entropy.detach().mean()):.2f}  sps={sps:.0f}  |  {phase_summary}",
                flush=True,
            )

        if batches_done % args.checkpoint_interval == 0:
            ckpt_path = ckpt_dir / f"step_{batches_done:07d}"
            save_bc_checkpoint(policy, ckpt_path, metadata={
                "kind": "bc_pretrain",
                "batches": batches_done,
                "samples": samples_done,
                "loss_ema": loss_ema,
                "acc_ema": acc_ema,
                "phase_acc": {p: sum(q)/max(len(q),1) for p, q in phase_acc.items()},
                "observation_api_version": OBSERVATION_API_VERSION,
            })
            print(f"  [checkpoint] {ckpt_path}")

        if args.max_batches and batches_done >= args.max_batches:
            print(f"reached --max-batches={args.max_batches}, stopping early")
            break

    final_dir = ckpt_dir / "final"
    save_bc_checkpoint(policy, final_dir, metadata={
        "kind": "bc_pretrain",
        "batches": batches_done,
        "samples": samples_done,
        "loss_ema": loss_ema,
        "acc_ema": acc_ema,
        "phase_acc": {p: sum(q)/max(len(q),1) for p, q in phase_acc.items()},
        "observation_api_version": OBSERVATION_API_VERSION,
        "final": True,
    })
    print(f"[done] final checkpoint at {final_dir}")
    print(f"final loss_ema={loss_ema:.4f}  acc_ema={acc_ema*100:.1f}%")
    for p, q in sorted(phase_acc.items()):
        print(f"  phase acc {p}: {sum(q)/max(len(q),1)*100:.1f}% (n={len(q)})")


if __name__ == "__main__":
    main()
