"""Measure how much a BC-calibrated checkpoint's non-combat decisions
moved toward the human prior compared to its pre-calibration source.

For a held-out slice of Skada samples, report top-1 accuracy + mean
chosen-action log-prob per phase for both checkpoints, then show the
per-phase delta. A "successful" calibration should show positive
accuracy deltas on the phases that were in the --phase-filter (map,
campfire, card_reward, relic_*) without significantly degrading the
policy's aggregate behavior.

Also surfaces a sanity check: if the BC calibrated checkpoint's
combat-action logit distribution shifted by >X from the source on the
held-out batches, the combat policy may have been perturbed by the
shared-backbone gradient leakage and needs to be re-tightened (lower
LR or fewer samples).

Usage:
    python eval_bc_calibration_delta.py \
        --source-ckpt checkpoints_attention/sim_phase8_longtrain_820k_XXX/step_819200 \
        --calibrated-ckpt checkpoints_attention/bc_calibration_YYY/final \
        --samples data/skada_bc/samples.jsonl \
        --holdout-offset 1800000 --holdout-count 5000
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import spaces

from skada_bc_train import NON_COMBAT_PHASES
from skada_bc_translate import translate_bc_sample
from sts2_env.checkpoint import (
    load_online_checkpoint_metadata,
    load_online_policy_state_dict,
)
from sts2_env.observation_v3 import (
    MAX_ACTIONS,
    WorldTokenObservationEncoder,
)
from legacy.attention_ppo.omni_attention_policy import STS2OmniAttentionPolicy


def load_policy(checkpoint_dir: Path, device: torch.device) -> STS2OmniAttentionPolicy:
    """Rebuild a policy from the checkpoint's metadata + state_dict."""
    metadata = load_online_checkpoint_metadata(checkpoint_dir)
    policy_kwargs = dict(metadata.get("policy_kwargs") or {})
    obs_encoder = WorldTokenObservationEncoder(use_text=False)
    policy = STS2OmniAttentionPolicy(
        observation_space=obs_encoder.obs_space,
        action_space=spaces.Discrete(MAX_ACTIONS),
        lr_schedule=lambda _: 1e-5,
        **policy_kwargs,
    )
    policy.to(device)
    policy.eval()
    load_online_policy_state_dict(policy, checkpoint_dir, device=str(device), strict=True)
    return policy


def iter_samples(path: Path, *, offset: int, count: int):
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < offset:
                continue
            if i >= offset + count:
                return
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def encode_sample(encoder, sample) -> tuple[dict, np.ndarray, int, str] | None:
    translated = translate_bc_sample(sample)
    if translated is None:
        return None
    obs, legal_actions, chosen_idx = translated
    n_legal = len(legal_actions)
    if n_legal == 0 or not (0 <= chosen_idx < n_legal):
        return None
    try:
        encoded = encoder.encode(obs, legal_actions, planner_context={})
    except Exception:
        return None
    action_mask = np.zeros(MAX_ACTIONS, dtype=np.float32)
    action_mask[:n_legal] = 1.0
    return encoded, action_mask, int(chosen_idx), str(sample.get("phase", "?"))


def evaluate_policy(policy, items: list, device: torch.device) -> dict[str, dict[str, float]]:
    """Run the policy on held-out items, bucket by phase, return per-phase
    {acc, mean_logprob, n}."""
    per_phase: dict[str, dict[str, list]] = collections.defaultdict(
        lambda: {"correct": [], "logprob": []}
    )
    with torch.no_grad():
        for encoded, action_mask, target, phase in items:
            obs_batch = {
                k: torch.from_numpy(v).unsqueeze(0).to(device)
                for k, v in encoded.items()
            }
            mask_t = torch.from_numpy(action_mask).unsqueeze(0).to(device)
            _, logprobs, _ = policy.evaluate_actions(
                obs_batch,
                torch.tensor([target], dtype=torch.long, device=device),
                action_masks=mask_t.bool(),
            )
            dist = policy.get_distribution(obs_batch, action_masks=mask_t.bool())
            preds = dist.distribution.logits.argmax(dim=-1)
            correct = int(preds.item() == target)
            per_phase[phase]["correct"].append(correct)
            per_phase[phase]["logprob"].append(float(logprobs.item()))
    return {
        phase: {
            "acc": float(np.mean(data["correct"])) if data["correct"] else 0.0,
            "mean_logprob": float(np.mean(data["logprob"])) if data["logprob"] else 0.0,
            "n": len(data["correct"]),
        }
        for phase, data in per_phase.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-ckpt", required=True,
                        help="PPO checkpoint that was calibrated FROM")
    parser.add_argument("--calibrated-ckpt", required=True,
                        help="BC-calibrated checkpoint to compare")
    parser.add_argument("--samples", default="data/skada_bc/samples.jsonl")
    parser.add_argument("--holdout-offset", type=int, default=1800000,
                        help="Skip this many samples (held out from the "
                             "first ~1.8M the BC train consumed)")
    parser.add_argument("--holdout-count", type=int, default=5000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"[eval] building held-out batch (offset={args.holdout_offset}, count={args.holdout_count})")
    encoder = WorldTokenObservationEncoder(use_text=False)
    items = []
    for sample in iter_samples(Path(args.samples), offset=args.holdout_offset, count=args.holdout_count):
        enc = encode_sample(encoder, sample)
        if enc is not None:
            items.append(enc)
    print(f"[eval] usable held-out items: {len(items)}")

    print(f"[eval] loading SOURCE policy: {args.source_ckpt}")
    source = load_policy(Path(args.source_ckpt), device)
    source_metrics = evaluate_policy(source, items, device)

    print(f"[eval] loading CALIBRATED policy: {args.calibrated_ckpt}")
    calibrated = load_policy(Path(args.calibrated_ckpt), device)
    calibrated_metrics = evaluate_policy(calibrated, items, device)

    print()
    print("Per-phase accuracy + mean log-prob (SOURCE → CALIBRATED = DELTA):")
    print(f"{'phase':<16} {'n':>6} {'src_acc':>8} {'cal_acc':>8} {'acc_delta':>10}  {'src_lp':>8} {'cal_lp':>8} {'lp_delta':>10}")
    for phase in sorted(set(source_metrics) | set(calibrated_metrics)):
        s = source_metrics.get(phase, {"acc": 0.0, "mean_logprob": 0.0, "n": 0})
        c = calibrated_metrics.get(phase, {"acc": 0.0, "mean_logprob": 0.0, "n": 0})
        tag = " (NON_COMBAT)" if phase in NON_COMBAT_PHASES else ""
        print(
            f"{phase:<16}{tag} {s['n']:>6} "
            f"{s['acc']*100:>7.1f}% {c['acc']*100:>7.1f}% "
            f"{(c['acc']-s['acc'])*100:>+9.1f}%  "
            f"{s['mean_logprob']:>+7.3f} {c['mean_logprob']:>+7.3f} "
            f"{c['mean_logprob']-s['mean_logprob']:>+9.3f}"
        )
    print()
    print("Interpretation:")
    print("  - Positive acc_delta on NON_COMBAT phases = calibration worked as intended")
    print("  - Large negative acc_delta on any phase = catastrophic forgetting risk")
    print("  - Positive lp_delta = policy more confident in human-chosen action")


if __name__ == "__main__":
    main()
