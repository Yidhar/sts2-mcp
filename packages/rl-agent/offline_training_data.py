"""Task-specific offline datasets, vocab builders, and collators for STS2."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from torch.utils.data import Dataset

from offline_dataset_loader import load_dataset


CANDIDATE_TASKS = {"card_choice", "ancient_choice", "relic_choice", "potion_choice"}
ROUTE_TASKS = {"route_room_type", "route_point_type"}
ACTION_ONLY_CARD_TASKS = {"upgrade", "card_remove", "card_transform"}
ACTION_ONLY_CLASS_TASKS = {"rest_site"}
CLASSIFICATION_TASKS = ROUTE_TASKS | ACTION_ONLY_CLASS_TASKS
SUPERVISED_TASKS = CANDIDATE_TASKS | ROUTE_TASKS | ACTION_ONLY_CARD_TASKS | ACTION_ONLY_CLASS_TASKS


@dataclass
class StringVocab:
    stoi: dict[str, int]
    itos: list[str]
    pad_token: str = "<pad>"
    unk_token: str = "<unk>"

    @classmethod
    def build(
        cls,
        values: list[str],
        *,
        add_unknown: bool = True,
        add_pad: bool = True,
    ) -> "StringVocab":
        ordered = sorted({str(value) for value in values if value is not None and str(value) != ""})
        itos: list[str] = []
        if add_pad:
            itos.append("<pad>")
        if add_unknown:
            itos.append("<unk>")
        itos.extend(ordered)
        stoi = {token: index for index, token in enumerate(itos)}
        return cls(stoi=stoi, itos=itos)

    def encode(self, value: str | None) -> int:
        if value is None:
            return 0
        key = str(value)
        if key in self.stoi:
            return self.stoi[key]
        return self.stoi.get(self.unk_token, 0)

    @property
    def size(self) -> int:
        return len(self.itos)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stoi": self.stoi,
            "itos": self.itos,
            "pad_token": self.pad_token,
            "unk_token": self.unk_token,
        }


def load_task_rows(
    root: str | Path,
    task: str,
    *,
    fmt: str = "parquet",
    partition_kind: str | None = None,
    partition_value: str | None = None,
    split: str | None = None,
) -> list[dict[str, Any]]:
    dataset_name = _dataset_name_for_task(task)
    dataset = load_dataset(
        root,
        dataset_name,
        fmt=fmt,
        partition_kind=partition_kind,
        partition_value=partition_value,
    )
    rows = [dataset[index] for index in range(len(dataset))]
    if split is not None:
        rows = [row for row in rows if row.get("split") == split]

    if task in CANDIDATE_TASKS:
        samples = [_normalize_candidate_row(row, task) for row in rows]
    elif task in ROUTE_TASKS:
        samples = [_normalize_route_row(row, task) for row in rows]
    elif task in ACTION_ONLY_CARD_TASKS:
        samples = [_normalize_action_card_row(row, task) for row in rows]
    elif task in ACTION_ONLY_CLASS_TASKS:
        samples = [_normalize_action_class_row(row, task) for row in rows]
    else:
        raise ValueError(f"Unsupported task: {task}")
    return [sample for sample in samples if sample is not None]


def build_state_vocabs(rows: list[dict[str, Any]]) -> dict[str, StringVocab]:
    card_ids: list[str] = []
    relic_ids: list[str] = []
    monster_ids: list[str] = []
    room_types: list[str] = []
    point_types: list[str] = []
    room_models: list[str] = []

    for row in rows:
        room_types.append(row["room_type"])
        point_types.append(row["map_point_type"])
        room_models.append(row["room_model_id"])
        card_ids.extend(row["deck_ids"])
        relic_ids.extend(row["relic_ids"])
        monster_ids.extend(row["monster_ids"])
        if "selected_slot_ids" in row:
            card_ids.extend(row["selected_slot_ids"])
        if row.get("task") == "card_choice":
            card_ids.extend(row["candidate_ids"])

    return {
        "card": StringVocab.build(card_ids),
        "relic": StringVocab.build(relic_ids),
        "monster": StringVocab.build(monster_ids),
        "room_type": StringVocab.build(room_types),
        "map_point_type": StringVocab.build(point_types),
        "room_model_id": StringVocab.build(room_models),
    }


def build_output_vocabs(rows: list[dict[str, Any]], task: str) -> dict[str, StringVocab]:
    if task == "card_choice":
        return {}
    if task in {"ancient_choice", "relic_choice", "potion_choice"}:
        choice_ids: list[str] = []
        for row in rows:
            choice_ids.extend(row["candidate_ids"])
        return {"choice": StringVocab.build(choice_ids)}
    if task in ROUTE_TASKS:
        labels = [row["label"] for row in rows]
        return {"label": StringVocab.build(labels, add_unknown=False, add_pad=False)}
    if task in ACTION_ONLY_CLASS_TASKS:
        labels = [row["label"] for row in rows]
        return {"label": StringVocab.build(labels, add_unknown=False, add_pad=False)}
    if task in ACTION_ONLY_CARD_TASKS:
        return {}
    raise ValueError(f"Unsupported task: {task}")


def build_task_vocabs(rows: list[dict[str, Any]], task: str) -> dict[str, StringVocab]:
    vocabs = build_state_vocabs(rows)
    vocabs.update(build_output_vocabs(rows, task))
    return vocabs


def build_task_metadata(rows: list[dict[str, Any]], task: str) -> dict[str, Any]:
    meta = {
        "task": task,
        "task_family": _task_family(task),
        "scalar_dim": len(rows[0]["scalars"]) if rows else 0,
    }
    if task in CANDIDATE_TASKS:
        meta["max_candidates"] = max((len(row["candidate_ids"]) for row in rows), default=1)
    if task in ACTION_ONLY_CARD_TASKS:
        meta["slot_count"] = max((len(row["selected_slot_ids"]) for row in rows), default=1)
        meta["max_candidates"] = max((len(row["deck_ids"]) for row in rows), default=1)
    return meta


class OfflineRowsDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


def make_collate_fn(task: str, vocabs: dict[str, StringVocab], meta: dict[str, Any]) -> Callable[[list[dict[str, Any]]], dict[str, torch.Tensor | list[str] | str]]:
    if task in CANDIDATE_TASKS:
        return lambda batch: _collate_candidate_batch(batch, task, vocabs)
    if task in CLASSIFICATION_TASKS:
        return lambda batch: _collate_classification_batch(batch, task, vocabs)
    if task in ACTION_ONLY_CARD_TASKS:
        return lambda batch: _collate_action_card_batch(batch, task, vocabs, meta)
    raise ValueError(f"Unsupported task: {task}")


def _dataset_name_for_task(task: str) -> str:
    if task == "card_choice":
        return "card_choice_samples"
    if task in {"ancient_choice", "relic_choice", "potion_choice"}:
        return f"{task}_samples"
    if task in ROUTE_TASKS:
        return "route_samples"
    if task == "rest_site":
        return "rest_site_samples"
    if task == "upgrade":
        return "upgrade_samples"
    if task == "card_remove":
        return "card_remove_samples"
    if task == "card_transform":
        return "card_transform_samples"
    raise ValueError(f"Unsupported task: {task}")


def _task_family(task: str) -> str:
    if task in CANDIDATE_TASKS:
        return "candidate"
    if task in ROUTE_TASKS:
        return "route"
    if task in ACTION_ONLY_CLASS_TASKS:
        return "classification"
    if task in ACTION_ONLY_CARD_TASKS:
        return "cardset"
    raise ValueError(f"Unsupported task: {task}")


def _normalize_candidate_row(row: dict[str, Any], task: str) -> dict[str, Any] | None:
    state = _extract_common_state(row, deck_key="deck_before", relic_key="relic_ids_before")

    if task == "card_choice":
        options = row.get("options") or []
        candidate_ids: list[str] = []
        candidate_upgrade_levels: list[float] = []
        for item in options:
            card = item.get("card") or {}
            card_id = card.get("id")
            if not card_id:
                continue
            candidate_ids.append(str(card_id))
            candidate_upgrade_levels.append(float(card.get("current_upgrade_level") or 0.0))

        label_index = None
        if row.get("skipped"):
            candidate_ids.append("<skip>")
            candidate_upgrade_levels.append(0.0)
            label_index = len(candidate_ids) - 1
        else:
            picked_ids = [str(value) for value in row.get("picked_card_ids") or [] if value]
            if len(picked_ids) != 1:
                return None
            try:
                label_index = candidate_ids.index(picked_ids[0])
            except ValueError:
                return None

        if label_index is None or not candidate_ids:
            return None
        return {
            **state,
            "sample_id": row["sample_id"],
            "task": task,
            "candidate_ids": candidate_ids,
            "candidate_upgrade_levels": candidate_upgrade_levels,
            "label_index": label_index,
        }

    candidate_ids = [str(value) for value in row.get("option_ids") or [] if value]
    selected_ids = [str(value) for value in row.get("selected_ids") or [] if value]
    if not candidate_ids or len(selected_ids) != 1:
        return None
    try:
        label_index = candidate_ids.index(selected_ids[0])
    except ValueError:
        return None
    return {
        **state,
        "sample_id": row["sample_id"],
        "task": task,
        "candidate_ids": candidate_ids,
        "candidate_upgrade_levels": [0.0] * len(candidate_ids),
        "label_index": label_index,
    }


def _normalize_route_row(row: dict[str, Any], task: str) -> dict[str, Any] | None:
    state = _extract_common_state(row, deck_key="deck_after", relic_key="relic_ids_after")
    label = row.get("next_room_type") if task == "route_room_type" else row.get("next_map_point_type")
    if not label:
        return None
    return {
        **state,
        "sample_id": row["sample_id"],
        "task": task,
        "label": str(label),
    }


def _normalize_action_class_row(row: dict[str, Any], task: str) -> dict[str, Any] | None:
    if task != "rest_site":
        raise ValueError(f"Unsupported action class task: {task}")
    state = _extract_common_state(row, deck_key="deck_before", relic_key="relic_ids_before")
    actions = [str(value) for value in row.get("selected_rest_actions") or [] if value]
    if len(actions) != 1:
        return None
    return {
        **state,
        "sample_id": row["sample_id"],
        "task": task,
        "label": actions[0],
    }


def _normalize_action_card_row(row: dict[str, Any], task: str) -> dict[str, Any] | None:
    state = _extract_common_state(row, deck_key="deck_before", relic_key="relic_ids_before")
    selected_ids = [str(value) for value in row.get("selected_card_ids") or [] if value]
    if not selected_ids:
        return None
    deck_id_set = set(state["deck_ids"])
    if any(selected_id not in deck_id_set for selected_id in selected_ids):
        return None
    selected_ids = sorted(selected_ids)
    return {
        **state,
        "sample_id": row["sample_id"],
        "task": task,
        "selected_slot_ids": selected_ids,
        "selected_count": len(selected_ids),
    }


def _extract_common_state(row: dict[str, Any], *, deck_key: str, relic_key: str) -> dict[str, Any]:
    deck = row.get(deck_key) or {}
    deck_cards = deck.get("cards") or []
    deck_ids = [str(card["id"]) for card in deck_cards if card.get("id")]
    deck_counts = [float(card.get("count") or 0.0) for card in deck_cards if card.get("id")]
    deck_upgraded_counts = [float(card.get("upgraded_count") or 0.0) for card in deck_cards if card.get("id")]
    deck_max_upgrade_levels = [float(card.get("max_upgrade_level") or 0.0) for card in deck_cards if card.get("id")]

    relic_ids = [str(value) for value in row.get(relic_key) or [] if value]
    monster_ids = [str(value) for value in row.get("monster_ids") or [] if value]
    quality = row.get("quality_flags") or {}

    max_hp = float(row.get("max_hp") or 0.0)
    hp_before = float(row.get("hp_before") or 0.0)
    current_hp = float(row.get("current_hp") or 0.0)
    gold_before = float(row.get("gold_before") or 0.0)
    current_gold = float(row.get("current_gold") or 0.0)

    scalars = [
        float(row.get("floor_number") or 0.0),
        float(row.get("act_index") or 0.0),
        float(row.get("path_index") or 0.0),
        float(row.get("ascension") or 0.0),
        hp_before,
        current_hp,
        max_hp,
        _safe_ratio(hp_before, max_hp),
        _safe_ratio(current_hp, max_hp),
        gold_before,
        current_gold,
        float(row.get("turns_taken") or 0.0),
        float(deck.get("deck_size") or 0.0),
        float(deck.get("distinct_cards") or 0.0),
        float(deck.get("upgraded_card_count") or 0.0),
        float(len(relic_ids)),
        1.0 if quality.get("initial_floor_sample") else 0.0,
        1.0 if quality.get("relic_ids_before_approximate") else 0.0,
    ]

    return {
        "split": str(row.get("split") or "train"),
        "build_id": str(row.get("build_id") or "unknown"),
        "run_id": str(row.get("run_id") or "unknown"),
        "room_type": str(row.get("room_type") or "unknown"),
        "map_point_type": str(row.get("map_point_type") or "unknown"),
        "room_model_id": str(row.get("room_model_id") or "unknown"),
        "deck_ids": deck_ids,
        "deck_counts": deck_counts,
        "deck_upgraded_counts": deck_upgraded_counts,
        "deck_max_upgrade_levels": deck_max_upgrade_levels,
        "relic_ids": relic_ids,
        "monster_ids": monster_ids,
        "scalars": scalars,
    }


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _collate_candidate_batch(
    batch: list[dict[str, Any]],
    task: str,
    vocabs: dict[str, StringVocab],
) -> dict[str, torch.Tensor | list[str] | str]:
    card_vocab = vocabs["card"]
    candidate_vocab = card_vocab if task == "card_choice" else vocabs["choice"]
    payload = _collate_state(batch, vocabs)
    payload.update(
        {
            "task": task,
            "candidate_ids": _pad_2d([[candidate_vocab.encode(value) for value in row["candidate_ids"]] for row in batch], pad_value=0, dtype=torch.long),
            "candidate_upgrade_levels": _pad_2d_float([row["candidate_upgrade_levels"] for row in batch]),
            "candidate_mask": _pad_mask([len(row["candidate_ids"]) for row in batch]),
            "labels": torch.tensor([int(row["label_index"]) for row in batch], dtype=torch.long),
            "sample_ids": [row["sample_id"] for row in batch],
        }
    )
    return payload


def _collate_classification_batch(
    batch: list[dict[str, Any]],
    task: str,
    vocabs: dict[str, StringVocab],
) -> dict[str, torch.Tensor | list[str] | str]:
    label_vocab = vocabs["label"]
    payload = _collate_state(batch, vocabs)
    payload.update(
        {
            "task": task,
            "labels": torch.tensor([label_vocab.encode(row["label"]) for row in batch], dtype=torch.long),
            "sample_ids": [row["sample_id"] for row in batch],
        }
    )
    return payload


def _collate_action_card_batch(
    batch: list[dict[str, Any]],
    task: str,
    vocabs: dict[str, StringVocab],
    meta: dict[str, Any],
) -> dict[str, torch.Tensor | list[str] | str]:
    card_vocab = vocabs["card"]
    slot_count = int(meta.get("slot_count") or 1)
    candidate_id_rows: list[list[int]] = []
    candidate_feature_rows: list[list[list[float]]] = []
    label_rows: list[list[int]] = []
    for row in batch:
        candidate_ids = list(row["deck_ids"])
        candidate_id_rows.append([card_vocab.encode(value) for value in candidate_ids])
        candidate_feature_rows.append(
            [
                [
                    float(row["deck_counts"][index]),
                    float(row["deck_upgraded_counts"][index]),
                    float(row["deck_max_upgrade_levels"][index]),
                ]
                for index in range(len(candidate_ids))
            ]
        )
        label_indices = sorted(candidate_ids.index(value) for value in row["selected_slot_ids"])
        label_indices = label_indices[:slot_count]
        label_indices.extend([-100] * max(slot_count - len(label_indices), 0))
        label_rows.append(label_indices)

    payload = _collate_state(batch, vocabs)
    payload.update(
        {
            "task": task,
            "candidate_ids": _pad_2d(candidate_id_rows, pad_value=0, dtype=torch.long),
            "candidate_features": _pad_3d_float(candidate_feature_rows),
            "candidate_mask": _pad_mask([len(row["deck_ids"]) for row in batch]),
            "labels": torch.tensor(label_rows, dtype=torch.long),
            "selection_count": torch.tensor([int(row["selected_count"]) for row in batch], dtype=torch.long),
            "sample_ids": [row["sample_id"] for row in batch],
        }
    )
    return payload


def _collate_state(batch: list[dict[str, Any]], vocabs: dict[str, StringVocab]) -> dict[str, torch.Tensor]:
    card_vocab = vocabs["card"]
    relic_vocab = vocabs["relic"]
    monster_vocab = vocabs["monster"]
    room_type_vocab = vocabs["room_type"]
    point_type_vocab = vocabs["map_point_type"]
    room_model_vocab = vocabs["room_model_id"]

    deck_id_rows = [[card_vocab.encode(value) for value in row["deck_ids"]] for row in batch]
    relic_id_rows = [[relic_vocab.encode(value) for value in row["relic_ids"]] for row in batch]
    monster_id_rows = [[monster_vocab.encode(value) for value in row["monster_ids"]] for row in batch]
    return {
        "scalars": torch.tensor([row["scalars"] for row in batch], dtype=torch.float32),
        "room_type": torch.tensor([room_type_vocab.encode(row["room_type"]) for row in batch], dtype=torch.long),
        "map_point_type": torch.tensor([point_type_vocab.encode(row["map_point_type"]) for row in batch], dtype=torch.long),
        "room_model_id": torch.tensor([room_model_vocab.encode(row["room_model_id"]) for row in batch], dtype=torch.long),
        "deck_ids": _pad_2d(deck_id_rows, pad_value=0, dtype=torch.long),
        "deck_counts": _pad_2d_float([row["deck_counts"] for row in batch]),
        "deck_upgraded_counts": _pad_2d_float([row["deck_upgraded_counts"] for row in batch]),
        "deck_max_upgrade_levels": _pad_2d_float([row["deck_max_upgrade_levels"] for row in batch]),
        "deck_mask": _pad_mask([len(row["deck_ids"]) for row in batch]),
        "relic_ids": _pad_2d(relic_id_rows, pad_value=0, dtype=torch.long),
        "relic_mask": _pad_mask([len(row["relic_ids"]) for row in batch]),
        "monster_ids": _pad_2d(monster_id_rows, pad_value=0, dtype=torch.long),
        "monster_mask": _pad_mask([len(row["monster_ids"]) for row in batch]),
    }


def _pad_2d(rows: list[list[int]], *, pad_value: int, dtype: torch.dtype) -> torch.Tensor:
    max_len = max((len(row) for row in rows), default=1)
    out = torch.full((len(rows), max_len), pad_value, dtype=dtype)
    for row_index, row in enumerate(rows):
        if row:
            out[row_index, : len(row)] = torch.tensor(row, dtype=dtype)
    return out


def _pad_2d_float(rows: list[list[float]]) -> torch.Tensor:
    max_len = max((len(row) for row in rows), default=1)
    out = torch.zeros((len(rows), max_len), dtype=torch.float32)
    for row_index, row in enumerate(rows):
        if row:
            out[row_index, : len(row)] = torch.tensor(row, dtype=torch.float32)
    return out


def _pad_3d_float(rows: list[list[list[float]]]) -> torch.Tensor:
    max_len = max((len(row) for row in rows), default=1)
    feature_dim = max((len(item) for row in rows for item in row), default=1)
    out = torch.zeros((len(rows), max_len, feature_dim), dtype=torch.float32)
    for row_index, row in enumerate(rows):
        for item_index, item in enumerate(row):
            out[row_index, item_index, : len(item)] = torch.tensor(item, dtype=torch.float32)
    return out


def _pad_mask(lengths: list[int]) -> torch.Tensor:
    max_len = max(lengths, default=1)
    mask = torch.zeros((len(lengths), max_len), dtype=torch.bool)
    for row_index, length in enumerate(lengths):
        if length > 0:
            mask[row_index, :length] = True
    return mask


def candidate_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    return float((preds == labels).float().mean().item())


def classification_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    return float((preds == labels).float().mean().item())


def cardset_exact_match_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    active = labels != -100
    if labels.numel() == 0:
        return 0.0
    row_matches: list[torch.Tensor] = []
    for row_index in range(labels.shape[0]):
        active_count = int(active[row_index].sum().item())
        if active_count <= 0:
            row_matches.append(torch.tensor(True, device=labels.device))
            continue
        pred_row = torch.sort(preds[row_index, :active_count])[0]
        true_row = torch.sort(labels[row_index, :active_count])[0]
        row_matches.append((pred_row == true_row).all())
    return float(torch.stack(row_matches).float().mean().item())
