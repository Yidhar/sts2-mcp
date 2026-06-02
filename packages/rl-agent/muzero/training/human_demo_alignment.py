"""Human-demo policy alignment for MuZero training.

This module is deliberately separate from ``train_step.py``.  It provides a
small, auditable behaviour-cloning loss on recorded human decisions that carry
the same two things the online policy uses at decision time:

* the observation snapshot; and
* the ordered ``legal_actions`` list containing the selected ``action_id``.

That makes it safer than applying CE directly to normalized offline scalar rows:
the target index is the live legal-action index, not a synthetic task-local
candidate index.  The default runtime mode should remain shadow/no-op until the
dashboard proves index coverage and CE metrics are sane.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from muzero.demo_dataset import DemoSample, load_demo_dataset
from muzero.sts2_env.muzero_buffer import MuZeroReplayBuffer
from sts2_env.observation_common import MAX_ACTIONS
from sts2_env.observation_v3 import WorldTokenObservationEncoder


def parse_human_demo_paths(raw: str | Sequence[str | Path] | None) -> tuple[Path, ...]:
    """Parse CLI/user path input into a stable tuple of paths.

    ``raw`` may be a single string, a comma/semicolon separated string, or an
    explicit sequence.  Empty parts are ignored so callers can safely pass an
    unset optional CLI flag.
    """

    if raw is None:
        return ()
    if isinstance(raw, (str, Path)):
        parts = str(raw).replace(";", ",").split(",")
    else:
        parts = [str(part) for part in raw]
    return tuple(Path(part.strip()) for part in parts if part and str(part).strip())


def _expand_jsonl_paths(paths: Iterable[Path]) -> tuple[Path, ...]:
    out: list[Path] = []
    for path in paths:
        if path.is_dir():
            jsonls = sorted(path.rglob("*.jsonl"))
            decisions = [item for item in jsonls if item.name == "decisions.jsonl"]
            # Recorder session directories also contain ``episodes.jsonl`` as a
            # sidecar.  A user naturally points the trainer at a session/demo
            # directory, so do not let sidecars break strict validation or make
            # an empty sidecar look like a demo file.  If no canonical
            # decisions file exists, fall back to non-episode JSONL files so
            # older custom demo dumps remain loadable.
            out.extend(decisions or [item for item in jsonls if item.name != "episodes.jsonl"])
        else:
            out.append(path)
    # Preserve order while removing duplicates that can occur when a directory
    # and an explicit file are both supplied.
    seen: set[str] = set()
    unique: list[Path] = []
    for path in out:
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return tuple(unique)


@dataclass(frozen=True)
class HumanDemoAlignmentConfig:
    """Configuration for human-demo BC alignment.

    ``max_samples`` is a deterministic cap for smoke tests.  Use ``None`` or
    ``<=0`` for all rows.
    """

    paths: tuple[str | Path, ...]
    batch_size: int = 64
    strict: bool = False
    encounter_filter: tuple[str, ...] | None = None
    max_samples: int | None = None
    seed: int = 0
    shuffle: bool = True
    use_text: bool = False

    def __post_init__(self) -> None:
        parsed = parse_human_demo_paths(self.paths)
        if not parsed:
            raise ValueError("Human demo alignment requires at least one JSONL file or directory")
        object.__setattr__(self, "paths", parsed)
        object.__setattr__(self, "batch_size", max(int(self.batch_size), 1))
        if self.max_samples is not None and int(self.max_samples) <= 0:
            object.__setattr__(self, "max_samples", None)
        if self.encounter_filter is not None:
            object.__setattr__(
                self,
                "encounter_filter",
                tuple(str(item) for item in self.encounter_filter if str(item)),
            )


class HumanDemoPolicyAligner:
    """Cyclic mini-batch provider for recorded human decisions."""

    def __init__(self, config: HumanDemoAlignmentConfig) -> None:
        self.config = config
        self.encoder = WorldTokenObservationEncoder(use_text=bool(config.use_text))
        self._rng = random.Random(int(config.seed))
        self._samples: list[DemoSample] = self._load_samples()
        self._order: list[int] = list(range(len(self._samples)))
        self._cursor = 0
        if config.shuffle:
            self._rng.shuffle(self._order)

    @property
    def samples(self) -> Sequence[DemoSample]:
        return self._samples

    @property
    def load_stats(self) -> dict[str, float]:
        tiers: dict[str, int] = {}
        encounters: dict[str, int] = {}
        for sample in self._samples:
            tiers[sample.tier] = tiers.get(sample.tier, 0) + 1
            encounters[sample.encounter_id] = encounters.get(sample.encounter_id, 0) + 1
        return {
            "rows": float(len(self._samples)),
            "tier_count": float(len(tiers)),
            "encounter_count": float(len(encounters)),
        }

    def _load_samples(self) -> list[DemoSample]:
        paths = _expand_jsonl_paths(Path(path) for path in self.config.paths)
        if not paths:
            raise FileNotFoundError("No human demo JSONL paths matched the supplied config")

        samples: list[DemoSample] = []
        for path in paths:
            samples.extend(
                load_demo_dataset(
                    path,
                    strict=bool(self.config.strict),
                    encounter_filter=self.config.encounter_filter,
                )
            )

        # Rows whose selected action is beyond the current action head cannot be
        # used for direct policy CE.  Keep the rule explicit instead of relying
        # on encoder truncation.
        samples = [
            sample
            for sample in samples
            if 0 <= int(sample.selected_action_index) < min(len(sample.legal_actions), MAX_ACTIONS)
        ]

        if self.config.max_samples is not None and len(samples) > int(self.config.max_samples):
            indices = list(range(len(samples)))
            self._rng.shuffle(indices)
            keep = sorted(indices[: int(self.config.max_samples)])
            samples = [samples[index] for index in keep]

        if not samples:
            raise ValueError("Human demo alignment loaded zero usable samples")
        return samples

    def _next_sample(self) -> DemoSample:
        if self._cursor >= len(self._order):
            self._cursor = 0
            if self.config.shuffle:
                self._rng.shuffle(self._order)
        sample = self._samples[self._order[self._cursor]]
        self._cursor += 1
        return sample

    @staticmethod
    def _looks_encoded_observation(obs: Mapping[str, Any]) -> bool:
        return "action_mask" in obs and (
            "world_tokens" in obs
            or "actions" in obs
            or "candidate_query_tokens" in obs
        )

    def _encode_sample(self, sample: DemoSample) -> dict[str, Any]:
        if isinstance(sample.obs, Mapping) and self._looks_encoded_observation(sample.obs):
            encoded = dict(sample.obs)
        else:
            encoded = self.encoder.encode(sample.obs, sample.legal_actions)

        mask = np.asarray(encoded.get("action_mask"), dtype=np.float32)
        selected = int(sample.selected_action_index)
        if mask.ndim != 1 or selected < 0 or selected >= min(mask.shape[0], MAX_ACTIONS):
            raise ValueError("selected action index is outside the encoded action mask")
        if float(mask[selected]) <= 0.0:
            raise ValueError("selected action is masked out in encoded observation")
        return encoded

    def next_batch(self) -> dict[str, Any] | None:
        """Return one encoded mini-batch, or ``None`` if no row can be encoded."""

        obs_list: list[dict[str, Any]] = []
        selected_indices: list[int] = []
        action_counts: list[int] = []
        encounter_ids: list[str] = []
        tiers: list[str] = []
        reason_tags: list[list[str]] = []
        sample_ids: list[str] = []
        attempted = 0
        max_attempts = max(len(self._samples), self.config.batch_size * 4)

        while len(obs_list) < self.config.batch_size and attempted < max_attempts:
            attempted += 1
            sample = self._next_sample()
            try:
                encoded = self._encode_sample(sample)
            except Exception:
                continue
            mask = np.asarray(encoded.get("action_mask"), dtype=np.float32)
            obs_list.append(encoded)
            selected_indices.append(int(sample.selected_action_index))
            action_counts.append(int(np.count_nonzero(mask > 0.0)))
            encounter_ids.append(sample.encounter_id)
            tiers.append(sample.tier)
            reason_tags.append(list(sample.reason_tags))
            sample_ids.append(f"{sample.episode_id}:{sample.turn}:{sample.step_in_turn}")

        if not obs_list:
            return None
        return {
            "obs_list": obs_list,
            "selected_action_indices": np.asarray(selected_indices, dtype=np.int64),
            "action_counts": np.asarray(action_counts, dtype=np.float32),
            "encounter_ids": encounter_ids,
            "tiers": tiers,
            "reason_tags": reason_tags,
            "sample_ids": sample_ids,
        }


def _zero_loss_for_network(network: torch.nn.Module, device: str | torch.device) -> torch.Tensor:
    try:
        param = next(network.parameters())
        return param.detach().new_zeros(())
    except StopIteration:
        return torch.zeros((), device=device)


def _obs_list_to_torch(obs_list: Sequence[dict[str, Any]], device: str | torch.device) -> dict[str, torch.Tensor]:
    obs_numpy = MuZeroReplayBuffer.batch_observations(list(obs_list))
    return {key: torch.from_numpy(value).to(device) for key, value in obs_numpy.items()}


def compute_human_demo_policy_alignment_loss(
    network: torch.nn.Module,
    batch: Mapping[str, Any] | None,
    *,
    device: str | torch.device,
    weight: float,
    shadow_only: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute a masked root-policy CE loss for one human-demo mini-batch.

    In shadow mode, or when ``weight <= 0``, this still reports CE/top-1/rank
    under ``torch.no_grad()`` but returns an exact zero loss and builds no
    autograd graph.
    """

    metrics: dict[str, float] = {
        "human_demo_alignment/active": 0.0,
        "human_demo_alignment/loss_applied": 0.0,
        "human_demo_alignment/weight": float(max(weight, 0.0)),
        "human_demo_alignment/shadow_only": 1.0 if shadow_only else 0.0,
        "human_demo_alignment/batch_size": 0.0,
        "human_demo_alignment/label_valid_rate": 0.0,
    }
    if not batch or not batch.get("obs_list"):
        return _zero_loss_for_network(network, device), metrics

    active_loss = (not bool(shadow_only)) and float(weight) > 0.0
    context = nullcontext() if active_loss else torch.no_grad()

    with context:
        obs_torch = _obs_list_to_torch(batch["obs_list"], device)
        initial = network.initial_inference(obs_torch)
        logits = initial.policy_logits.float()
        action_mask = obs_torch.get("action_mask")
        if action_mask is None:
            action_mask = torch.ones_like(logits, dtype=torch.float32)
        else:
            action_mask = action_mask.float()
            if action_mask.shape[-1] < logits.shape[-1]:
                pad = logits.shape[-1] - action_mask.shape[-1]
                action_mask = F.pad(action_mask, (0, pad), value=0.0)
            elif action_mask.shape[-1] > logits.shape[-1]:
                action_mask = action_mask[..., : logits.shape[-1]]

        labels = torch.as_tensor(
            batch["selected_action_indices"],
            dtype=torch.long,
            device=logits.device,
        )
        in_range = (labels >= 0) & (labels < logits.shape[-1])
        safe_labels = labels.clamp(0, max(logits.shape[-1] - 1, 0))
        label_unmasked = action_mask.gather(1, safe_labels.view(-1, 1)).squeeze(1) > 0.0
        valid = in_range & label_unmasked

        if int(valid.sum().item()) <= 0:
            return _zero_loss_for_network(network, device), metrics

        logits_valid = logits[valid]
        mask_valid = action_mask[valid]
        labels_valid = labels[valid]
        masked_logits = logits_valid.masked_fill(mask_valid <= 0.0, -1.0e9)
        ce = F.cross_entropy(masked_logits, labels_valid)
        log_probs = F.log_softmax(masked_logits, dim=-1)
        label_logprob = log_probs.gather(1, labels_valid.view(-1, 1)).squeeze(1)
        pred = masked_logits.argmax(dim=-1)
        target_logits = masked_logits.gather(1, labels_valid.view(-1, 1)).squeeze(1)
        selected_rank = 1.0 + (masked_logits > target_logits.view(-1, 1)).sum(dim=-1).float()

        weighted_loss = ce * float(weight) if active_loss else _zero_loss_for_network(network, device)

    batch_size = int(len(batch["obs_list"]))
    valid_count = int(valid.sum().detach().item())
    top1_match = (pred == labels_valid).float().mean().detach().item()
    action_counts_raw = batch.get("action_counts")
    if action_counts_raw is None:
        action_count_mean = 0.0
    else:
        action_counts_arr = np.asarray(action_counts_raw, dtype=np.float32)
        action_count_mean = float(action_counts_arr.mean()) if action_counts_arr.size else 0.0

    metrics.update(
        {
            "human_demo_alignment/active": 1.0,
            "human_demo_alignment/loss_applied": 1.0 if active_loss else 0.0,
            "human_demo_alignment/batch_size": float(batch_size),
            "human_demo_alignment/valid_count": float(valid_count),
            "human_demo_alignment/label_valid_rate": float(valid_count / max(batch_size, 1)),
            "human_demo_alignment/ce": float(ce.detach().item()),
            "human_demo_alignment/top1_match": float(top1_match),
            "human_demo_alignment/selected_rank": float(selected_rank.mean().detach().item()),
            "human_demo_alignment/label_logprob": float(label_logprob.mean().detach().item()),
            "human_demo_alignment/action_count_mean": action_count_mean,
        }
    )

    tiers = list(batch.get("tiers") or [])
    if tiers:
        valid_indices = valid.detach().nonzero(as_tuple=False).view(-1).cpu().numpy().tolist()
        pred_cpu = pred.detach().cpu().numpy().tolist()
        labels_cpu = labels_valid.detach().cpu().numpy().tolist()
        for tier in sorted(set(str(t) for t in tiers)):
            local_positions = [
                pos
                for pos, original_idx in enumerate(valid_indices)
                if original_idx < len(tiers) and str(tiers[original_idx]) == tier
            ]
            if not local_positions:
                continue
            matches = [
                1.0 if int(pred_cpu[pos]) == int(labels_cpu[pos]) else 0.0
                for pos in local_positions
            ]
            metrics[f"human_demo_alignment/tier_{tier}/top1_match"] = float(np.mean(matches))
            metrics[f"human_demo_alignment/tier_{tier}/count"] = float(len(local_positions))

    return weighted_loss, metrics
