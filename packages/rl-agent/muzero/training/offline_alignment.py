"""Offline/human alignment data plumbing for MuZero training.

This module is intentionally *data-first* and low risk:

* it loads existing offline build/route/shop/reward samples through
  :mod:`offline_training_data`;
* it reports label/candidate distributions that must be audited before any
  supervised loss is allowed to affect the online policy;
* it exposes collated PyTorch loaders for later small-weight alignment losses;
* it does not mutate MuZero networks, replay buffers, or checkpoints.

The reason for keeping this separate from ``train_step.py`` is that the current
hard-guard layer is already too large.  Alignment code should be auditable and
must prove index/action consistency before it can influence policy logits.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from torch.utils.data import DataLoader

from offline_training_data import (
    ACTION_ONLY_CARD_TASKS,
    AUXILIARY_TASKS,
    BUILD_V2_CLASS_TASKS,
    CANDIDATE_TASKS,
    CLASSIFICATION_TASKS,
    ROUTE_TASKS,
    SHOP_BUNDLE_AUX_FIELDS,
    SUPERVISED_TASKS,
    OfflineRowsDataset,
    build_task_metadata,
    build_task_vocabs,
    load_task_rows,
    make_collate_fn,
)


DEFAULT_ALIGNMENT_TASKS: tuple[str, ...] = (
    "regular_card_reward",
    "smith_target",
    "remove_card_step",
    "shop_remove_binary",
    "shop_remove_target_step",
    "shop_relic_pick_step",
    "shop_potion_pick_step",
    "rest_action",
    "route_room_type",
    "route_point_type",
)


@dataclass(frozen=True)
class OfflineAlignmentConfig:
    """Configuration for loading offline alignment samples.

    ``max_rows_per_task`` is a deterministic sample cap for audits/smokes.  Use
    ``None`` or ``<=0`` for all rows.
    """

    root: str | Path
    tasks: tuple[str, ...] = DEFAULT_ALIGNMENT_TASKS
    fmt: str = "parquet"
    split: str | None = None
    partition_kind: str | None = None
    partition_value: str | None = None
    max_rows_per_task: int | None = None
    seed: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))
        object.__setattr__(self, "tasks", tuple(str(task) for task in self.tasks))
        unknown = sorted(set(self.tasks) - set(SUPERVISED_TASKS))
        if unknown:
            raise ValueError(f"Unsupported offline alignment task(s): {unknown}")


@dataclass(frozen=True)
class OfflineAlignmentTaskSummary:
    """Audit-friendly summary for one normalized offline task."""

    task: str
    family: str
    rows: int
    scalar_dim: int = 0
    label_counts: dict[str, int] = field(default_factory=dict)
    candidate_count_min: int = 0
    candidate_count_mean: float = 0.0
    candidate_count_p95: float = 0.0
    candidate_count_max: int = 0
    has_label_index_rate: float | None = None
    label_index_valid_rate: float | None = None
    skip_available_rate: float | None = None
    skip_label_rate: float | None = None
    deck_size_mean: float = 0.0
    relic_count_mean: float = 0.0
    monster_count_mean: float = 0.0
    current_hp_ratio_mean: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OfflineAlignmentTaskData:
    """Loaded rows, vocabularies, metadata, and summary for one task."""

    task: str
    rows: list[dict[str, Any]]
    vocabs: dict[str, Any]
    metadata: dict[str, Any]
    summary: OfflineAlignmentTaskSummary

    def make_loader(
        self,
        *,
        batch_size: int,
        shuffle: bool = True,
        num_workers: int = 0,
        drop_last: bool = False,
        pin_memory: bool = False,
    ) -> DataLoader:
        """Build a collated DataLoader for this task.

        This is for future train-time alignment losses.  For now, callers should
        use it mostly for smoke tests and action-index alignment audits.
        """

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not self.rows:
            raise ValueError(f"Cannot create loader for empty offline task: {self.task}")
        collate_fn = make_collate_fn(self.task, self.vocabs, self.metadata)
        return DataLoader(
            OfflineRowsDataset(self.rows),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=drop_last,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
        )


class OfflineAlignmentDataModule:
    """Loader/cache for offline alignment tasks."""

    def __init__(self, config: OfflineAlignmentConfig) -> None:
        self.config = config
        self._tasks: dict[str, OfflineAlignmentTaskData] = {}

    @property
    def tasks(self) -> Mapping[str, OfflineAlignmentTaskData]:
        return self._tasks

    def load(self, *, force: bool = False) -> "OfflineAlignmentDataModule":
        if self._tasks and not force:
            return self

        loaded: dict[str, OfflineAlignmentTaskData] = {}
        for task in self.config.tasks:
            rows = load_task_rows(
                self.config.root,
                task,
                fmt=self.config.fmt,
                partition_kind=self.config.partition_kind,
                partition_value=self.config.partition_value,
                split=self.config.split,
            )
            rows = _deterministic_sample_rows(
                rows,
                max_rows=self.config.max_rows_per_task,
                seed=self.config.seed + _stable_task_seed(task),
            )
            metadata = build_task_metadata(rows, task) if rows else _empty_metadata(task)
            vocabs = build_task_vocabs(rows, task) if rows else {}
            summary = summarize_task_rows(task, rows, metadata=metadata)
            loaded[task] = OfflineAlignmentTaskData(
                task=task,
                rows=rows,
                vocabs=vocabs,
                metadata=metadata,
                summary=summary,
            )

        self._tasks = loaded
        return self

    def summary(self) -> dict[str, Any]:
        self.load()
        return {
            "root": str(self.config.root),
            "fmt": self.config.fmt,
            "split": self.config.split,
            "tasks": {task: data.summary.to_dict() for task, data in self._tasks.items()},
            "total_rows": sum(data.summary.rows for data in self._tasks.values()),
        }

    def task_summary(self, task: str) -> OfflineAlignmentTaskSummary:
        self.load()
        return self._tasks[task].summary

    def make_loader(
        self,
        task: str,
        *,
        batch_size: int,
        shuffle: bool = True,
        num_workers: int = 0,
        drop_last: bool = False,
        pin_memory: bool = False,
    ) -> DataLoader:
        self.load()
        return self._tasks[task].make_loader(
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=drop_last,
            pin_memory=pin_memory,
        )


def summarize_task_rows(
    task: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> OfflineAlignmentTaskSummary:
    """Summarize already-normalized rows from ``offline_training_data``."""

    family = str((metadata or {}).get("task_family") or _task_family(task))
    scalar_dim = int((metadata or {}).get("scalar_dim") or _first_scalar_dim(rows))
    candidate_counts = [_candidate_count(row) for row in rows]
    label_counts = _label_counts(task, rows)
    has_label_index_flags = ["label_index" in row for row in rows]
    valid_label_index_flags = [
        _label_index_valid(row)
        for row in rows
        if "label_index" in row or _candidate_count(row) > 0
    ]
    skip_available_flags = [_skip_available(row) for row in rows if _skip_visible(row)]
    skip_label_flags = [_skip_selected(row) for row in rows if _skip_visible(row)]

    return OfflineAlignmentTaskSummary(
        task=task,
        family=family,
        rows=len(rows),
        scalar_dim=scalar_dim,
        label_counts=dict(sorted(label_counts.items())),
        candidate_count_min=min(candidate_counts, default=0),
        candidate_count_mean=_mean(candidate_counts),
        candidate_count_p95=_percentile(candidate_counts, 0.95),
        candidate_count_max=max(candidate_counts, default=0),
        has_label_index_rate=_rate(has_label_index_flags) if candidate_counts else None,
        label_index_valid_rate=_rate(valid_label_index_flags) if valid_label_index_flags else None,
        skip_available_rate=_rate(skip_available_flags) if skip_available_flags else None,
        skip_label_rate=_rate(skip_label_flags) if skip_label_flags else None,
        deck_size_mean=_mean([_deck_size(row) for row in rows]),
        relic_count_mean=_mean([len(row.get("relic_ids") or []) for row in rows]),
        monster_count_mean=_mean([len(row.get("monster_ids") or []) for row in rows]),
        current_hp_ratio_mean=_mean([_current_hp_ratio(row) for row in rows]),
    )


def _empty_metadata(task: str) -> dict[str, Any]:
    return {"task": task, "task_family": _task_family(task), "scalar_dim": 0}


def _task_family(task: str) -> str:
    if task in ROUTE_TASKS:
        return "route"
    if task in CANDIDATE_TASKS:
        return "candidate"
    if task in CLASSIFICATION_TASKS:
        return "classification"
    if task in AUXILIARY_TASKS:
        return "auxiliary"
    if task in ACTION_ONLY_CARD_TASKS:
        return "cardset"
    return "unknown"


def _deterministic_sample_rows(
    rows: list[dict[str, Any]],
    *,
    max_rows: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    if max_rows is None or max_rows <= 0 or len(rows) <= max_rows:
        return rows
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(rows)), max_rows))
    return [rows[index] for index in indices]


def _stable_task_seed(task: str) -> int:
    return sum((index + 1) * ord(ch) for index, ch in enumerate(task))


def _first_scalar_dim(rows: Sequence[Mapping[str, Any]]) -> int:
    for row in rows:
        scalars = row.get("scalars")
        if isinstance(scalars, Sequence) and not isinstance(scalars, (str, bytes)):
            return len(scalars)
    return 0


def _candidate_count(row: Mapping[str, Any]) -> int:
    if isinstance(row.get("candidate_ids"), Sequence) and not isinstance(row.get("candidate_ids"), (str, bytes)):
        return len(row.get("candidate_ids") or [])
    if isinstance(row.get("route_candidates"), Sequence) and not isinstance(row.get("route_candidates"), (str, bytes)):
        return len(row.get("route_candidates") or [])
    if isinstance(row.get("deck_ids"), Sequence) and row.get("task") in ACTION_ONLY_CARD_TASKS:
        return len(row.get("deck_ids") or [])
    if row.get("task") in AUXILIARY_TASKS:
        return len(SHOP_BUNDLE_AUX_FIELDS)
    return 0


def _label_counts(task: str, rows: Sequence[Mapping[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        if "label_id" in row and row.get("label_id") not in (None, ""):
            counts[str(row.get("label_id"))] += 1
            continue
        if "label" in row and row.get("label") not in (None, ""):
            counts[str(row.get("label"))] += 1
            continue
        if "label_index" in row:
            label = _label_from_index(row)
            counts[label] += 1
            continue
        if task in AUXILIARY_TASKS and isinstance(row.get("label_multi_hot"), Sequence):
            for field, value in zip(SHOP_BUNDLE_AUX_FIELDS, row.get("label_multi_hot") or []):
                if float(value or 0.0) > 0.5:
                    counts[field] += 1
            continue
        if "selected_slot_ids" in row:
            for value in row.get("selected_slot_ids") or []:
                counts[str(value)] += 1
    return counts


def _label_from_index(row: Mapping[str, Any]) -> str:
    index = row.get("label_index")
    if not isinstance(index, int):
        try:
            index = int(index)
        except (TypeError, ValueError):
            return "<invalid_index>"
    for key in ("candidate_ids", "route_candidates"):
        values = row.get(key)
        if isinstance(values, Sequence) and not isinstance(values, (str, bytes)) and 0 <= index < len(values):
            value = values[index]
            if isinstance(value, Mapping):
                return str(value.get("point_type_norm") or value.get("point_type") or value.get("action_id") or index)
            return str(value)
    return str(index)


def _label_index_valid(row: Mapping[str, Any]) -> bool:
    if "label_index" not in row:
        return False
    try:
        label_index = int(row.get("label_index"))
    except (TypeError, ValueError):
        return False
    return 0 <= label_index < _candidate_count(row)


def _skip_visible(row: Mapping[str, Any]) -> bool:
    candidate_ids = row.get("candidate_ids")
    if isinstance(candidate_ids, Sequence) and not isinstance(candidate_ids, (str, bytes)):
        return bool(row.get("skip_available")) or "<skip>" in [str(value) for value in candidate_ids]
    return "skip_available" in row


def _skip_available(row: Mapping[str, Any]) -> bool:
    candidate_ids = row.get("candidate_ids")
    if isinstance(candidate_ids, Sequence) and not isinstance(candidate_ids, (str, bytes)):
        return bool(row.get("skip_available")) or "<skip>" in [str(value) for value in candidate_ids]
    return bool(row.get("skip_available"))


def _skip_selected(row: Mapping[str, Any]) -> bool:
    if str(row.get("label_id") or "") == "<skip>":
        return True
    candidate_ids = row.get("candidate_ids")
    if not isinstance(candidate_ids, Sequence) or isinstance(candidate_ids, (str, bytes)):
        return False
    try:
        label_index = int(row.get("label_index"))
    except (TypeError, ValueError):
        return False
    return 0 <= label_index < len(candidate_ids) and str(candidate_ids[label_index]) == "<skip>"


def _deck_size(row: Mapping[str, Any]) -> float:
    counts = row.get("deck_counts")
    if isinstance(counts, Sequence) and not isinstance(counts, (str, bytes)) and counts:
        return float(sum(float(value or 0.0) for value in counts))
    deck_ids = row.get("deck_ids")
    if isinstance(deck_ids, Sequence) and not isinstance(deck_ids, (str, bytes)):
        return float(len(deck_ids))
    return 0.0


def _current_hp_ratio(row: Mapping[str, Any]) -> float:
    scalars = row.get("scalars")
    if not isinstance(scalars, Sequence) or isinstance(scalars, (str, bytes)) or len(scalars) <= 8:
        return 0.0
    try:
        return float(scalars[8])
    except (TypeError, ValueError):
        return 0.0


def _mean(values: Iterable[float | int | bool]) -> float:
    items = [float(value) for value in values]
    if not items:
        return 0.0
    return float(sum(items) / len(items))


def _rate(values: Iterable[bool]) -> float:
    items = [bool(value) for value in values]
    if not items:
        return 0.0
    return float(sum(1.0 for value in items if value) / len(items))


def _percentile(values: Sequence[int | float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    pos = max(0.0, min(1.0, q)) * (len(ordered) - 1)
    low = int(math.floor(pos))
    high = int(math.ceil(pos))
    if low == high:
        return ordered[low]
    frac = pos - low
    return ordered[low] * (1.0 - frac) + ordered[high] * frac


def _parse_tasks(raw: str | None) -> tuple[str, ...]:
    if raw is None or not raw.strip():
        return DEFAULT_ALIGNMENT_TASKS
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit offline alignment task distributions.")
    parser.add_argument("--root", type=Path, default=Path("tmp/offline_build_v2_full/parquet"))
    parser.add_argument("--tasks", type=str, default=",".join(DEFAULT_ALIGNMENT_TASKS))
    parser.add_argument("--fmt", type=str, default="parquet")
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--max-rows-per-task", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    config = OfflineAlignmentConfig(
        root=args.root,
        tasks=_parse_tasks(args.tasks),
        fmt=args.fmt,
        split=args.split,
        max_rows_per_task=args.max_rows_per_task or None,
        seed=args.seed,
    )
    module = OfflineAlignmentDataModule(config).load()
    print(json.dumps(module.summary(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
