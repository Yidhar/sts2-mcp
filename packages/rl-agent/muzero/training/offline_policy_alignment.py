"""Offline human/history policy alignment for live MuZero action heads.

The existing offline build datasets are valuable, but their ``label_index`` is
task-local: it indexes a task candidate list, not the online MuZero
``legal_actions`` order.  This module deliberately converts each normalized
offline row into a small live-style observation plus an ordered legal-action
list, encodes it with the normal token observation encoder, and only then
applies masked CE to ``initial_inference(...).policy_logits``.

Default mode is shadow-only.  Route samples are rejected unless explicitly
allowed because route/build credit is long-horizon and previously regressed
when pushed through direct route bias/alignment too early.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from content_registry import (
    build_card_label,
    build_card_semantic_text,
    build_live_potion_semantic_text,
    build_live_relic_semantic_text,
    get_card_metadata,
    get_potion_metadata,
    get_relic_metadata,
    humanize_game_id,
)
from muzero.sts2_env.muzero_buffer import MuZeroReplayBuffer
from offline_training_data import load_task_rows
from sts2_env.observation_common import MAX_ACTIONS, MAX_DECK, MAX_RELICS
from sts2_env.observation_v3 import WorldTokenObservationEncoder


DEFAULT_OFFLINE_POLICY_ALIGNMENT_TASKS: tuple[str, ...] = (
    "regular_card_reward",
    "smith_target",
    "remove_card_step",
    "shop_remove_binary",
    "shop_remove_target_step",
    "shop_relic_pick_step",
    "shop_potion_pick_step",
    "rest_action",
)
ROUTE_POLICY_ALIGNMENT_TASKS: frozenset[str] = frozenset(
    {"route_room_type", "route_point_type"}
)
REST_ACTION_CANDIDATES: tuple[str, str] = ("REST", "SMITH")
SHOP_REMOVE_BINARY_CANDIDATES: tuple[str, str] = ("remove_card", "skip_remove")
_SKIP_IDS: frozenset[str] = frozenset(
    {"<skip>", "skip", "skip_reward", "skip_remove", "leave", "leave_only", "no_purchase"}
)


@dataclass(frozen=True)
class OfflinePolicySample:
    """One offline row converted to a live-style policy target."""

    obs: dict[str, Any]
    legal_actions: list[dict[str, Any]]
    selected_action_index: int
    task: str
    sample_id: str


@dataclass(frozen=True)
class OfflinePolicyAlignmentConfig:
    """Configuration for offline parquet/json policy alignment.

    ``max_rows_per_task`` is a deterministic per-task cap for smoke tests.
    ``allow_route`` must stay false for the first active phase.
    """

    root: str | Path
    tasks: tuple[str, ...] = DEFAULT_OFFLINE_POLICY_ALIGNMENT_TASKS
    fmt: str = "parquet"
    batch_size: int = 64
    max_rows_per_task: int | None = None
    allow_route: bool = False
    seed: int = 0
    shuffle: bool = True
    use_text: bool = False

    def __post_init__(self) -> None:
        root = Path(self.root)
        tasks = parse_offline_alignment_tasks(self.tasks)
        if not tasks:
            raise ValueError("Offline policy alignment requires at least one task")
        blocked = sorted(set(tasks) & set(ROUTE_POLICY_ALIGNMENT_TASKS))
        if blocked and not bool(self.allow_route):
            raise ValueError(
                "Offline route tasks are disabled by default for policy CE: "
                + ",".join(blocked)
            )
        object.__setattr__(self, "root", root)
        object.__setattr__(self, "tasks", tasks)
        object.__setattr__(self, "batch_size", max(int(self.batch_size), 1))
        if self.max_rows_per_task is not None and int(self.max_rows_per_task) <= 0:
            object.__setattr__(self, "max_rows_per_task", None)


class OfflinePolicyAligner:
    """Cyclic mini-batch provider for offline policy alignment."""

    def __init__(self, config: OfflinePolicyAlignmentConfig) -> None:
        self.config = config
        self.encoder = WorldTokenObservationEncoder(use_text=bool(config.use_text))
        self._rng = random.Random(int(config.seed))
        self._rows: list[tuple[str, Mapping[str, Any]]] = self._load_rows()
        self._order: list[int] = list(range(len(self._rows)))
        self._cursor = 0
        if config.shuffle:
            self._rng.shuffle(self._order)

    @property
    def load_stats(self) -> dict[str, float]:
        counts: dict[str, int] = {}
        for task, _row in self._rows:
            counts[task] = counts.get(task, 0) + 1
        stats = {
            "rows": float(len(self._rows)),
            "task_count": float(len(counts)),
        }
        stats.update({f"task/{task}/rows": float(count) for task, count in counts.items()})
        return stats

    def _load_rows(self) -> list[tuple[str, Mapping[str, Any]]]:
        loaded: list[tuple[str, Mapping[str, Any]]] = []
        for task in self.config.tasks:
            rows = load_task_rows(self.config.root, task, fmt=self.config.fmt)
            if self.config.max_rows_per_task is not None and len(rows) > int(
                self.config.max_rows_per_task
            ):
                indices = list(range(len(rows)))
                self._rng.shuffle(indices)
                keep = sorted(indices[: int(self.config.max_rows_per_task)])
                rows = [rows[index] for index in keep]
            for row in rows:
                loaded.append((task, row))
        if not loaded:
            raise ValueError("Offline policy alignment loaded zero usable rows")
        return loaded

    def _next_row(self) -> tuple[str, Mapping[str, Any]]:
        if self._cursor >= len(self._order):
            self._cursor = 0
            if self.config.shuffle:
                self._rng.shuffle(self._order)
        task, row = self._rows[self._order[self._cursor]]
        self._cursor += 1
        return task, row

    def _encode_sample(self, sample: OfflinePolicySample) -> dict[str, Any]:
        if not (0 <= int(sample.selected_action_index) < min(len(sample.legal_actions), MAX_ACTIONS)):
            raise ValueError("selected offline action index is outside the action head")
        encoded = self.encoder.encode(sample.obs, sample.legal_actions)
        mask = np.asarray(encoded.get("action_mask"), dtype=np.float32)
        selected = int(sample.selected_action_index)
        if mask.ndim != 1 or selected < 0 or selected >= min(mask.shape[0], MAX_ACTIONS):
            raise ValueError("selected offline action index is outside the encoded mask")
        if float(mask[selected]) <= 0.0:
            raise ValueError("selected offline action is masked out after encoding")
        return encoded

    def next_batch(self) -> dict[str, Any] | None:
        obs_list: list[dict[str, Any]] = []
        selected_indices: list[int] = []
        action_counts: list[int] = []
        tasks: list[str] = []
        sample_ids: list[str] = []
        attempted = 0
        max_attempts = max(len(self._rows), self.config.batch_size * 4)

        while len(obs_list) < self.config.batch_size and attempted < max_attempts:
            attempted += 1
            task, row = self._next_row()
            try:
                sample = build_offline_policy_sample(row, task)
                encoded = self._encode_sample(sample)
            except Exception:
                continue
            mask = np.asarray(encoded.get("action_mask"), dtype=np.float32)
            obs_list.append(encoded)
            selected_indices.append(int(sample.selected_action_index))
            action_counts.append(int(np.count_nonzero(mask > 0.0)))
            tasks.append(str(task))
            sample_ids.append(str(sample.sample_id))

        if not obs_list:
            return None
        return {
            "obs_list": obs_list,
            "selected_action_indices": np.asarray(selected_indices, dtype=np.int64),
            "action_counts": np.asarray(action_counts, dtype=np.float32),
            "tasks": tasks,
            "sample_ids": sample_ids,
        }


def parse_offline_alignment_tasks(raw: str | Sequence[str] | None) -> tuple[str, ...]:
    """Parse CLI/user task input into a stable tuple.

    Empty input means the safe default build/shop/reward/rest task set.  Commas
    and semicolons are both accepted for Windows/PowerShell convenience.
    """

    if raw is None:
        return DEFAULT_OFFLINE_POLICY_ALIGNMENT_TASKS
    if isinstance(raw, str):
        parts = raw.replace(";", ",").split(",")
    else:
        parts = [str(part) for part in raw]
    parsed = tuple(part.strip() for part in parts if part and part.strip())
    return parsed or DEFAULT_OFFLINE_POLICY_ALIGNMENT_TASKS


def build_offline_policy_sample(row: Mapping[str, Any], task: str | None = None) -> OfflinePolicySample:
    """Convert one normalized offline row into obs + ordered legal actions."""

    task = str(task or row.get("task") or "")
    if not task:
        raise ValueError("offline row is missing task")

    legal_actions, selected_index = _legal_actions_for_row(row, task)
    if not legal_actions:
        raise ValueError(f"offline row has no legal actions for task={task}")
    if selected_index is None or not (0 <= int(selected_index) < len(legal_actions)):
        raise ValueError(f"offline selected index is invalid for task={task}")

    obs = _base_obs(row, task, len(legal_actions))
    return OfflinePolicySample(
        obs=obs,
        legal_actions=legal_actions,
        selected_action_index=int(selected_index),
        task=task,
        sample_id=str(row.get("sample_id") or f"{task}:unknown"),
    )


def compute_offline_policy_alignment_loss(
    network: torch.nn.Module,
    batch: Mapping[str, Any] | None,
    *,
    device: str | torch.device,
    weight: float,
    shadow_only: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute masked CE from offline samples to root policy logits.

    Shadow mode still reports CE/rank/top-1 under ``torch.no_grad()`` but
    returns an exact zero tensor detached from the graph.
    """

    safe_weight = max(float(weight), 0.0)
    metrics: dict[str, float] = {
        "offline_alignment/active": 0.0,
        "offline_alignment/loss_applied": 0.0,
        "offline_alignment/weight": safe_weight,
        "offline_alignment/shadow_only": 1.0 if shadow_only else 0.0,
        "offline_alignment/batch_size": 0.0,
        "offline_alignment/valid_count": 0.0,
        "offline_alignment/label_valid_rate": 0.0,
        "offline_alignment/label_in_range_rate": 0.0,
        "offline_alignment/label_unmasked_rate": 0.0,
        "offline_alignment/label_oor_count": 0.0,
        "offline_alignment/label_masked_count": 0.0,
    }
    if not batch or not batch.get("obs_list"):
        return _zero_loss_for_network(network, device), metrics

    active_loss = (not bool(shadow_only)) and safe_weight > 0.0
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
                action_mask = F.pad(action_mask, (0, logits.shape[-1] - action_mask.shape[-1]), value=0.0)
            elif action_mask.shape[-1] > logits.shape[-1]:
                action_mask = action_mask[..., : logits.shape[-1]]

        labels = torch.as_tensor(
            batch["selected_action_indices"],
            dtype=torch.long,
            device=logits.device,
        )
        batch_size = int(len(batch["obs_list"]))
        in_range = (labels >= 0) & (labels < logits.shape[-1])
        safe_labels = labels.clamp(0, max(logits.shape[-1] - 1, 0))
        label_mask_at_safe_index = action_mask.gather(1, safe_labels.view(-1, 1)).squeeze(1) > 0.0
        label_unmasked = in_range & label_mask_at_safe_index
        valid = label_unmasked

        valid_count = int(valid.sum().detach().item())
        in_range_count = int(in_range.sum().detach().item())
        unmasked_count = int(label_unmasked.sum().detach().item())
        oor_count = int(batch_size - in_range_count)
        masked_count = int(in_range_count - unmasked_count)
        if valid_count <= 0:
            metrics.update(
                {
                    "offline_alignment/active": 1.0,
                    "offline_alignment/batch_size": float(batch_size),
                    "offline_alignment/valid_count": 0.0,
                    "offline_alignment/label_valid_rate": 0.0,
                    "offline_alignment/label_in_range_rate": float(in_range_count / max(batch_size, 1)),
                    "offline_alignment/label_unmasked_rate": float(unmasked_count / max(batch_size, 1)),
                    "offline_alignment/label_oor_count": float(oor_count),
                    "offline_alignment/label_masked_count": float(masked_count),
                }
            )
            metrics.update(_per_task_label_diagnostics(batch, in_range, label_unmasked))
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
        weighted_loss = ce * safe_weight if active_loss else _zero_loss_for_network(network, device)

    action_counts_raw = batch.get("action_counts")
    action_counts_arr = np.asarray(
        action_counts_raw if action_counts_raw is not None else [],
        dtype=np.float32,
    )
    action_count_mean = float(action_counts_arr.mean()) if action_counts_arr.size else 0.0
    top1_match = (pred == labels_valid).float().mean().detach().item()
    metrics.update(
        {
            "offline_alignment/active": 1.0,
            "offline_alignment/loss_applied": 1.0 if active_loss else 0.0,
            "offline_alignment/batch_size": float(batch_size),
            "offline_alignment/valid_count": float(valid_count),
            "offline_alignment/label_valid_rate": float(valid_count / max(batch_size, 1)),
            "offline_alignment/label_in_range_rate": float(in_range_count / max(batch_size, 1)),
            "offline_alignment/label_unmasked_rate": float(unmasked_count / max(batch_size, 1)),
            "offline_alignment/label_oor_count": float(oor_count),
            "offline_alignment/label_masked_count": float(masked_count),
            "offline_alignment/ce": float(ce.detach().item()),
            "offline_alignment/top1_match": float(top1_match),
            "offline_alignment/selected_rank": float(selected_rank.mean().detach().item()),
            "offline_alignment/label_logprob": float(label_logprob.mean().detach().item()),
            "offline_alignment/action_count_mean": action_count_mean,
        }
    )
    metrics.update(
        _per_task_metrics(
            batch,
            valid,
            in_range,
            label_unmasked,
            pred,
            labels_valid,
            selected_rank,
            label_logprob,
        )
    )
    return weighted_loss, metrics


def _per_task_label_diagnostics(
    batch: Mapping[str, Any],
    in_range: torch.Tensor,
    label_unmasked: torch.Tensor,
) -> dict[str, float]:
    tasks = list(batch.get("tasks") or [])
    if not tasks:
        return {}
    in_range_cpu = in_range.detach().cpu().numpy().astype(bool).tolist()
    label_unmasked_cpu = label_unmasked.detach().cpu().numpy().astype(bool).tolist()
    action_counts_raw = batch.get("action_counts")
    action_counts = np.asarray(
        action_counts_raw if action_counts_raw is not None else [],
        dtype=np.float32,
    )
    out: dict[str, float] = {}
    for task in sorted(set(str(task) for task in tasks)):
        original_rows = [idx for idx, value in enumerate(tasks) if str(value) == task]
        in_range_count = sum(
            1 for idx in original_rows if idx < len(in_range_cpu) and bool(in_range_cpu[idx])
        )
        unmasked_count = sum(
            1 for idx in original_rows if idx < len(label_unmasked_cpu) and bool(label_unmasked_cpu[idx])
        )
        batch_count = len(original_rows)
        prefix = f"offline_alignment/{task}"
        out[f"{prefix}/active"] = 1.0
        out[f"{prefix}/batch_size"] = float(batch_count)
        out[f"{prefix}/valid_count"] = float(unmasked_count)
        out[f"{prefix}/label_valid_rate"] = float(unmasked_count / max(batch_count, 1))
        out[f"{prefix}/label_in_range_rate"] = float(in_range_count / max(batch_count, 1))
        out[f"{prefix}/label_unmasked_rate"] = float(unmasked_count / max(batch_count, 1))
        out[f"{prefix}/label_oor_count"] = float(max(batch_count - in_range_count, 0))
        out[f"{prefix}/label_masked_count"] = float(max(in_range_count - unmasked_count, 0))
        if action_counts.size and original_rows:
            out[f"{prefix}/action_count_mean"] = float(action_counts[original_rows].mean())
    return out


def _per_task_metrics(
    batch: Mapping[str, Any],
    valid: torch.Tensor,
    in_range: torch.Tensor,
    label_unmasked: torch.Tensor,
    pred: torch.Tensor,
    labels_valid: torch.Tensor,
    selected_rank: torch.Tensor,
    label_logprob: torch.Tensor,
) -> dict[str, float]:
    tasks = list(batch.get("tasks") or [])
    if not tasks:
        return {}
    out = _per_task_label_diagnostics(batch, in_range, label_unmasked)
    valid_indices = valid.detach().nonzero(as_tuple=False).view(-1).cpu().numpy().tolist()
    pred_cpu = pred.detach().cpu().numpy().tolist()
    labels_cpu = labels_valid.detach().cpu().numpy().tolist()
    rank_cpu = selected_rank.detach().cpu().numpy().tolist()
    logprob_cpu = label_logprob.detach().cpu().numpy().tolist()
    for task in sorted(set(str(task) for task in tasks)):
        valid_positions = [
            pos
            for pos, original_idx in enumerate(valid_indices)
            if original_idx < len(tasks) and str(tasks[original_idx]) == task
        ]
        prefix = f"offline_alignment/{task}"
        if valid_positions:
            matches = [
                1.0 if int(pred_cpu[pos]) == int(labels_cpu[pos]) else 0.0
                for pos in valid_positions
            ]
            out[f"{prefix}/top1_match"] = float(np.mean(matches))
            out[f"{prefix}/selected_rank"] = float(np.mean([rank_cpu[pos] for pos in valid_positions]))
            out[f"{prefix}/label_logprob"] = float(np.mean([logprob_cpu[pos] for pos in valid_positions]))
    return out


def _legal_actions_for_row(row: Mapping[str, Any], task: str) -> tuple[list[dict[str, Any]], int | None]:
    if task == "regular_card_reward":
        return _candidate_card_reward_actions(row, task)
    if task == "smith_target":
        return _candidate_smith_actions(row, task)
    if task in {"remove_card_step", "shop_remove_target_step"}:
        return _candidate_remove_actions(row, task)
    if task == "shop_remove_binary":
        return _shop_remove_binary_actions(row)
    if task == "shop_relic_pick_step":
        return _shop_entity_pick_actions(row, task, "relic")
    if task == "shop_potion_pick_step":
        return _shop_entity_pick_actions(row, task, "potion")
    if task == "rest_action":
        return _rest_action_actions(row)
    raise ValueError(f"Unsupported offline policy alignment task: {task}")


def _candidate_card_reward_actions(row: Mapping[str, Any], task: str) -> tuple[list[dict[str, Any]], int | None]:
    actions: list[dict[str, Any]] = []
    for index, candidate_id in enumerate(row.get("candidate_ids") or []):
        cid = str(candidate_id)
        if _is_skip_id(cid):
            actions.append(
                {
                    "kind": "card_reward",
                    "action_id": f"offline:{task}:skip",
                    "index": index,
                    "skip": True,
                    "selection": "skip",
                    "title": "Skip",
                    "canonical_text": "skip card reward",
                }
            )
            continue
        upgrade = _candidate_upgrade_level(row, index)
        card = _card_payload(cid, upgrade_level=upgrade, task=task)
        actions.append(
            {
                "kind": "card_reward",
                "action_id": f"offline:{task}:{index}:{cid}",
                "index": index,
                "card": card,
                "selection": "pick",
                "title": card["title"],
                "canonical_text": card["canonical_text"],
            }
        )
    return actions, _safe_int(row.get("label_index"))


def _candidate_smith_actions(row: Mapping[str, Any], task: str) -> tuple[list[dict[str, Any]], int | None]:
    actions: list[dict[str, Any]] = []
    for index, candidate_id in enumerate(row.get("candidate_ids") or []):
        cid = str(candidate_id)
        upgrade = _candidate_upgrade_level(row, index)
        card = _card_payload(cid, upgrade_level=upgrade, task=task)
        preview = _card_payload(cid, upgrade_level=upgrade + 1.0, task=task)
        actions.append(
            {
                "kind": "deck_upgrade",
                "action_id": f"offline:{task}:{index}:{cid}",
                "index": index,
                "slot_index": index,
                "selection": "upgrade",
                "card": card,
                "upgrade_preview": preview,
                "title": f"Smith {card['title']}",
                "canonical_text": f"smith upgrade {card['canonical_text']}",
            }
        )
    return actions, _safe_int(row.get("label_index"))


def _candidate_remove_actions(row: Mapping[str, Any], task: str) -> tuple[list[dict[str, Any]], int | None]:
    actions: list[dict[str, Any]] = []
    for index, candidate_id in enumerate(row.get("candidate_ids") or []):
        cid = str(candidate_id)
        upgrade = _candidate_upgrade_level(row, index)
        card = _card_payload(cid, upgrade_level=upgrade, task=task)
        actions.append(
            {
                "kind": "card_selection",
                "action_id": f"offline:{task}:{index}:{cid}",
                "index": index,
                "slot_index": index,
                "selection": "select",
                "selection_semantics": "remove_card",
                "selection_prompt": "remove card from deck",
                "source_pile": "deck",
                "card": card,
                "title": f"Remove {card['title']}",
                "canonical_text": f"remove card {card['canonical_text']}",
            }
        )
    return actions, _safe_int(row.get("label_index"))


def _shop_remove_binary_actions(row: Mapping[str, Any]) -> tuple[list[dict[str, Any]], int | None]:
    gold = _current_gold(row)
    remove_cost = 75.0
    actions = [
        {
            "kind": "shop",
            "shop_action": "buy",
            "action_id": "offline:shop_remove_binary:remove_card",
            "index": 0,
            "item": {
                "item_kind": "card_removal",
                "title": "card removal",
                "cost": remove_cost,
                "is_affordable": bool(gold >= remove_cost),
            },
            "title": "Buy card removal",
            "canonical_text": "buy card removal",
        },
        {
            "kind": "shop",
            "shop_action": "leave",
            "action_id": "offline:shop_remove_binary:skip_remove",
            "index": 1,
            "skip": True,
            "item": {"item_kind": "leave", "title": "leave shop", "cost": 0.0},
            "title": "Leave shop",
            "canonical_text": "leave shop / skip card removal",
        },
    ]
    return actions, _shop_remove_label_index(row.get("label"))


def _shop_entity_pick_actions(
    row: Mapping[str, Any],
    task: str,
    item_kind: str,
) -> tuple[list[dict[str, Any]], int | None]:
    actions: list[dict[str, Any]] = []
    gold = _current_gold(row)
    for index, candidate_id in enumerate(row.get("candidate_ids") or []):
        cid = str(candidate_id)
        if _is_skip_id(cid):
            actions.append(
                {
                    "kind": "shop",
                    "shop_action": "leave",
                    "action_id": f"offline:{task}:skip",
                    "index": index,
                    "skip": True,
                    "item": {"item_kind": "leave", "title": "leave shop", "cost": 0.0},
                    "title": "Leave shop",
                    "canonical_text": "leave shop / skip buying",
                }
            )
            continue
        cost = _estimated_shop_cost(cid, item_kind)
        if item_kind == "relic":
            payload = _relic_payload(cid)
        else:
            payload = _potion_payload(cid)
        actions.append(
            {
                "kind": "shop",
                "shop_action": "buy",
                "action_id": f"offline:{task}:{index}:{cid}",
                "index": index,
                "item": {
                    "item_kind": item_kind,
                    item_kind: payload,
                    "title": payload["title"],
                    "cost": cost,
                    "is_affordable": bool(gold >= cost),
                },
                item_kind: payload,
                "title": f"Buy {payload['title']}",
                "canonical_text": f"buy {item_kind} {payload['canonical_text']}",
            }
        )
    return actions, _safe_int(row.get("label_index"))


def _rest_action_actions(row: Mapping[str, Any]) -> tuple[list[dict[str, Any]], int | None]:
    actions = [
        {
            "kind": "rest_site",
            "action_id": "offline:rest_action:rest",
            "index": 0,
            "selection": "rest",
            "title": "Rest",
            "canonical_text": "rest at campfire and recover HP",
        },
        {
            "kind": "rest_site",
            "action_id": "offline:rest_action:smith",
            "index": 1,
            "selection": "smith",
            "title": "Smith",
            "canonical_text": "smith upgrade a card at campfire",
        },
    ]
    return actions, _rest_label_index(row.get("label"))


def _base_obs(row: Mapping[str, Any], task: str, legal_action_count: int) -> dict[str, Any]:
    phase = _phase_for_task(task)
    floor = _current_floor(row)
    act = _current_act(row)
    room_type = _normalize_room_type(row.get("room_type"))
    map_point_type = _normalize_map_point_type(row.get("map_point_type"))
    deck_cards = _deck_cards(row)
    relics = _relics(row)
    decision_domain = "build"
    return {
        "phase": phase,
        "decision_domain": decision_domain,
        "decision": {
            "task": task,
            "sample_id": str(row.get("sample_id") or ""),
            "option_count": int(legal_action_count),
            "can_skip": task in {"regular_card_reward", "shop_relic_pick_step", "shop_potion_pick_step", "shop_remove_binary"},
            "selected_count": len(row.get("selected_prefix_ids") or []),
            "min_select": 0,
            "max_select": 1,
            "is_open": phase == "shop",
            "item_count": legal_action_count if phase == "shop" else 0,
            "reward_count": legal_action_count if phase == "card_reward" else 0,
        },
        "run": {
            "active": True,
            "game_over": False,
            "act_id": f"Act{act}",
            "act_floor": int(floor),
            "floor": int(floor),
            "room_type": room_type,
        },
        "player": {
            "current_hp": _current_hp(row),
            "hp": _current_hp(row),
            "max_hp": _max_hp(row),
            "gold": _current_gold(row),
            "deck": len(deck_cards),
            "deck_cards": deck_cards,
            "relics": relics,
            "potions": [],
        },
        "map": {
            "current_floor": int(floor),
            "act_id": int(act),
            "room_type": room_type,
            "map_point_type": map_point_type,
        },
    }


def _phase_for_task(task: str) -> str:
    return {
        "regular_card_reward": "card_reward",
        "smith_target": "deck_upgrade",
        "remove_card_step": "card_selection",
        "shop_remove_target_step": "card_selection",
        "shop_remove_binary": "shop",
        "shop_relic_pick_step": "shop",
        "shop_potion_pick_step": "shop",
        "rest_action": "rest_site",
    }.get(task, "card_selection")


def _deck_cards(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    ids = [str(value) for value in row.get("deck_ids") or [] if value]
    counts = list(row.get("deck_counts") or [])
    upgraded_counts = list(row.get("deck_upgraded_counts") or [])
    max_upgrades = list(row.get("deck_max_upgrade_levels") or [])
    cards: list[dict[str, Any]] = []
    for index, card_id in enumerate(ids):
        count = max(int(round(_float_at(counts, index, 1.0))), 0)
        upgraded_count = max(int(round(_float_at(upgraded_counts, index, 0.0))), 0)
        max_upgrade = _float_at(max_upgrades, index, 1.0)
        for copy_index in range(count):
            upgrade = max_upgrade if copy_index < upgraded_count else 0.0
            cards.append(_card_payload(card_id, upgrade_level=upgrade, task="deck"))
            if len(cards) >= MAX_DECK:
                return cards
    return cards


def _relics(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    relic_ids = [str(value) for value in row.get("relic_ids") or [] if value]
    return [_relic_payload(relic_id) for relic_id in relic_ids[:MAX_RELICS]]


def _card_payload(card_id: str, *, upgrade_level: float = 0.0, task: str | None = None) -> dict[str, Any]:
    metadata = get_card_metadata(card_id) or {}
    title = build_card_label(card_id, upgrade_level)
    text = build_card_semantic_text(card_id, upgrade_level=upgrade_level, task=task)
    cost = metadata.get("cost", metadata.get("energy_cost", metadata.get("base_cost", 0)))
    card_type = metadata.get("type", metadata.get("card_type", ""))
    target = metadata.get("target", metadata.get("target_type", ""))
    return {
        "id": str(card_id),
        "card_id": str(card_id),
        "title": title,
        "name": title,
        "canonical_text": text or title,
        "cost": _safe_float(cost, 0.0),
        "type": str(card_type or ""),
        "target": str(target or ""),
        "target_type": str(target or ""),
        "upgraded": bool(float(upgrade_level) > 0.0),
        "upgrade_level": float(upgrade_level),
        "current_upgrade_level": float(upgrade_level),
    }


def _relic_payload(relic_id: str) -> dict[str, Any]:
    metadata = get_relic_metadata(relic_id) or {}
    title = str(metadata.get("title") or humanize_game_id(relic_id))
    payload = {
        "id": str(relic_id),
        "relic_id": str(relic_id),
        "title": title,
        "canonical_text": str(metadata.get("summary") or metadata.get("description") or title),
        "rarity": str(metadata.get("rarity") or ""),
    }
    payload["canonical_text"] = build_live_relic_semantic_text(payload) or payload["canonical_text"]
    return payload


def _potion_payload(potion_id: str) -> dict[str, Any]:
    metadata = get_potion_metadata(potion_id) or {}
    title = str(metadata.get("title") or humanize_game_id(potion_id))
    payload = {
        "id": str(potion_id),
        "potion_id": str(potion_id),
        "title": title,
        "canonical_text": str(metadata.get("summary") or metadata.get("description") or title),
        "rarity": str(metadata.get("rarity") or ""),
        "target_scope": str(metadata.get("target_scope") or ""),
    }
    payload["canonical_text"] = build_live_potion_semantic_text(payload) or payload["canonical_text"]
    return payload


def _candidate_upgrade_level(row: Mapping[str, Any], index: int) -> float:
    return _float_at(list(row.get("candidate_upgrade_levels") or []), index, 0.0)


def _is_skip_id(value: Any) -> bool:
    return str(value or "").strip().lower() in _SKIP_IDS


def _rest_label_index(label: Any) -> int | None:
    text = str(label or "").strip().lower()
    if text in {"rest", "heal", "sleep", "campfire_rest"}:
        return 0
    if text in {"smith", "upgrade", "campfire_smith"}:
        return 1
    if "rest" in text or "heal" in text:
        return 0
    if "smith" in text or "upgrade" in text:
        return 1
    return None


def _shop_remove_label_index(label: Any) -> int | None:
    text = str(label or "").strip().lower()
    if text in {"remove_card", "remove", "true", "1", "yes", "buy_remove", "buy_card_removal"}:
        return 0
    if text in {"skip_remove", "skip", "false", "0", "no", "leave", "leave_only", "no_purchase"}:
        return 1
    return None


def _estimated_shop_cost(entity_id: str, item_kind: str) -> float:
    if item_kind == "potion":
        metadata = get_potion_metadata(entity_id) or {}
        return _safe_float(metadata.get("cost", metadata.get("shop_cost")), 60.0)
    if item_kind == "relic":
        metadata = get_relic_metadata(entity_id) or {}
        return _safe_float(metadata.get("cost", metadata.get("shop_cost")), 150.0)
    return 0.0


def _current_floor(row: Mapping[str, Any]) -> int:
    scalars = list(row.get("scalars") or [])
    value = _float_at(scalars, 0, 0.0)
    return max(int(round(value)), 0)


def _current_act(row: Mapping[str, Any]) -> int:
    scalars = list(row.get("scalars") or [])
    value = int(round(_float_at(scalars, 1, 0.0)))
    # Dataset act_index is usually zero-based; keep Act1 visible for zeros.
    return max(value + 1 if value <= 3 else value, 1)


def _current_hp(row: Mapping[str, Any]) -> float:
    scalars = list(row.get("scalars") or [])
    current = _float_at(scalars, 5, 0.0)
    if current <= 0.0:
        current = _float_at(scalars, 4, 70.0)
    return max(float(current), 0.0)


def _max_hp(row: Mapping[str, Any]) -> float:
    scalars = list(row.get("scalars") or [])
    max_hp = _float_at(scalars, 6, 0.0)
    if max_hp <= 0.0:
        max_hp = max(_current_hp(row), 70.0)
    return max(float(max_hp), 1.0)


def _current_gold(row: Mapping[str, Any]) -> float:
    scalars = list(row.get("scalars") or [])
    gold = _float_at(scalars, 10, _float_at(scalars, 9, 0.0))
    return max(float(gold), 0.0)


def _normalize_room_type(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"monster", "normal", "weak"}:
        return "Monster"
    if text == "elite":
        return "Elite"
    if text == "boss":
        return "Boss"
    if text in {"rest", "rest_site", "restsite", "campfire"}:
        return "Rest"
    if text in {"shop", "merchant"}:
        return "Merchant"
    if text == "treasure":
        return "Treasure"
    return "Event"


def _normalize_map_point_type(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"monster", "normal", "weak"}:
        return "Monster"
    if text == "elite":
        return "Elite"
    if text == "boss":
        return "Boss"
    if text in {"event", "unknown"}:
        return "Event"
    if text in {"question", "questionmark", "question_mark", "?"}:
        return "QuestionMark"
    if text in {"rest", "rest_site", "restsite", "campfire"}:
        return "RestSite"
    if text in {"shop", "merchant"}:
        return "Shop"
    if text == "treasure":
        return "Treasure"
    return "Event"


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _float_at(values: Sequence[Any], index: int, default: float = 0.0) -> float:
    if index < 0 or index >= len(values):
        return float(default)
    return _safe_float(values[index], default)


def _zero_loss_for_network(network: torch.nn.Module, device: str | torch.device) -> torch.Tensor:
    try:
        param = next(network.parameters())
        return param.detach().new_zeros(())
    except StopIteration:
        return torch.zeros((), device=device)


def _obs_list_to_torch(obs_list: Sequence[dict[str, Any]], device: str | torch.device) -> dict[str, torch.Tensor]:
    obs_numpy = MuZeroReplayBuffer.batch_observations(list(obs_list))
    return {key: torch.from_numpy(value).to(device) for key, value in obs_numpy.items()}
