"""Offline multi-task supervised training for STS2 choice and route tasks."""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from offline_training_data import (
    ACTION_ONLY_CARD_TASKS,
    CLASSIFICATION_TASKS,
    CANDIDATE_TASKS,
    ROUTE_TASKS,
    SUPERVISED_TASKS,
    OfflineRowsDataset,
    build_output_vocabs,
    build_state_vocabs,
    build_task_metadata,
    candidate_accuracy,
    cardset_exact_match_accuracy,
    classification_accuracy,
    load_task_rows,
    make_collate_fn,
)
from offline_training_models import MultiTaskOfflineModel, OfflineStateEncoder


DEFAULT_TASKS = [
    "card_choice",
    "ancient_choice",
    "relic_choice",
    "potion_choice",
    "route_room_type",
    "route_point_type",
    "rest_site",
    "upgrade",
    "card_remove",
    "card_transform",
]


@dataclass
class OfflineMultiTaskConfig:
    tasks: list[str]
    dataset_root: str
    dataset_format: str = "parquet"
    partition_kind: str | None = None
    partition_value: str | None = None
    out_dir: str = "offline_runs"
    batch_size: int = 64
    epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    hidden_dim: int = 128
    entity_embed_dim: int = 64
    device: str = "cpu"
    seed: int = 7
    num_workers: int = 0
    eval_split: str = "val"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.to(device)
        else:
            out[key] = value
    return out


def compute_batch_loss_and_accuracy(task: str, logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, float]:
    if task in CANDIDATE_TASKS:
        loss = F.cross_entropy(logits, labels)
        acc = candidate_accuracy(logits, labels)
        return loss, acc
    if task in CLASSIFICATION_TASKS:
        loss = F.cross_entropy(logits, labels)
        acc = classification_accuracy(logits, labels)
        return loss, acc
    if task in ACTION_ONLY_CARD_TASKS:
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100)
        acc = cardset_exact_match_accuracy(logits, labels)
        return loss, acc
    raise ValueError(f"Unsupported task: {task}")


def build_multitask_model(
    config: OfflineMultiTaskConfig,
    *,
    state_vocabs,
    task_vocabs: dict[str, dict[str, Any]],
    task_meta: dict[str, dict[str, Any]],
) -> MultiTaskOfflineModel:
    scalar_dim = next(iter(task_meta.values()))["scalar_dim"]
    state_encoder = OfflineStateEncoder(
        scalar_dim=scalar_dim,
        card_vocab_size=state_vocabs["card"].size,
        relic_vocab_size=state_vocabs["relic"].size,
        monster_vocab_size=state_vocabs["monster"].size,
        room_type_vocab_size=state_vocabs["room_type"].size,
        point_type_vocab_size=state_vocabs["map_point_type"].size,
        room_model_vocab_size=state_vocabs["room_model_id"].size,
        embed_dim=config.entity_embed_dim,
        hidden_dim=config.hidden_dim,
    )

    head_builders: dict[str, dict[str, Any]] = {}
    for task in config.tasks:
        meta = task_meta[task]
        vocabs = task_vocabs[task]
        if task in CANDIDATE_TASKS:
            candidate_vocab_size = state_vocabs["card"].size if task == "card_choice" else vocabs["choice"].size
            head_builders[task] = {"family": "candidate", "candidate_vocab_size": candidate_vocab_size}
        elif task in CLASSIFICATION_TASKS:
            head_builders[task] = {"family": meta["task_family"], "num_labels": vocabs["label"].size}
        elif task in ACTION_ONLY_CARD_TASKS:
            head_builders[task] = {
                "family": "cardset",
                "card_vocab_size": state_vocabs["card"].size,
                "slot_count": meta["slot_count"],
            }
        else:
            raise ValueError(f"Unsupported task: {task}")

    return MultiTaskOfflineModel(
        state_encoder=state_encoder,
        head_builders=head_builders,
        embed_dim=config.entity_embed_dim,
        hidden_dim=config.hidden_dim,
    )


def evaluate_multitask(
    model: MultiTaskOfflineModel,
    eval_loaders: dict[str, DataLoader],
    device: torch.device,
) -> dict[str, dict[str, float]]:
    model.eval()
    metrics: dict[str, dict[str, float]] = {}
    with torch.no_grad():
        for task, loader in eval_loaders.items():
            total_loss = 0.0
            total_examples = 0
            total_correct = 0.0
            for batch in loader:
                batch = move_batch_to_device(batch, device)
                logits = model(task, batch)
                labels = batch["labels"]
                loss, accuracy = compute_batch_loss_and_accuracy(task, logits, labels)
                batch_size = labels.shape[0]
                total_loss += float(loss.item()) * batch_size
                total_examples += batch_size
                total_correct += accuracy * batch_size
            metrics[task] = {
                "loss": total_loss / max(total_examples, 1),
                "accuracy": total_correct / max(total_examples, 1),
                "examples": total_examples,
            }
    return metrics


def iter_round_robin(loaders: dict[str, DataLoader], task_order: list[str]):
    iterators = {task: iter(loader) for task, loader in loaders.items()}
    active = set(task_order)
    while active:
        progressed = False
        for task in task_order:
            if task not in active:
                continue
            try:
                batch = next(iterators[task])
                progressed = True
                yield task, batch
            except StopIteration:
                active.remove(task)
        if not progressed:
            break


def train(config: OfflineMultiTaskConfig) -> dict[str, Any]:
    set_seed(config.seed)
    device = torch.device(config.device)

    train_rows_by_task: dict[str, list[dict[str, Any]]] = {}
    eval_rows_by_task: dict[str, list[dict[str, Any]]] = {}

    for task in config.tasks:
        train_rows = load_task_rows(
            config.dataset_root,
            task,
            fmt=config.dataset_format,
            partition_kind=config.partition_kind,
            partition_value=config.partition_value,
            split="train",
        )
        eval_rows = load_task_rows(
            config.dataset_root,
            task,
            fmt=config.dataset_format,
            partition_kind=config.partition_kind,
            partition_value=config.partition_value,
            split=config.eval_split,
        )
        if not eval_rows:
            eval_rows = load_task_rows(
                config.dataset_root,
                task,
                fmt=config.dataset_format,
                partition_kind=config.partition_kind,
                partition_value=config.partition_value,
                split="test",
            )
        if not train_rows:
            raise SystemExit(f"No training rows found for task={task}")
        if not eval_rows:
            raise SystemExit(f"No evaluation rows found for task={task}")
        train_rows_by_task[task] = train_rows
        eval_rows_by_task[task] = eval_rows

    all_rows = [row for rows in train_rows_by_task.values() for row in rows]
    all_rows.extend(row for rows in eval_rows_by_task.values() for row in rows)
    state_vocabs = build_state_vocabs(all_rows)
    task_vocabs = {task: build_output_vocabs(train_rows_by_task[task] + eval_rows_by_task[task], task) for task in config.tasks}
    task_meta = {task: build_task_metadata(train_rows_by_task[task] + eval_rows_by_task[task], task) for task in config.tasks}

    train_loaders: dict[str, DataLoader] = {}
    eval_loaders: dict[str, DataLoader] = {}
    for task in config.tasks:
        vocabs = {**state_vocabs, **task_vocabs[task]}
        collate_fn = make_collate_fn(task, vocabs, task_meta[task])
        train_loaders[task] = DataLoader(
            OfflineRowsDataset(train_rows_by_task[task]),
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            collate_fn=collate_fn,
        )
        eval_loaders[task] = DataLoader(
            OfflineRowsDataset(eval_rows_by_task[task]),
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            collate_fn=collate_fn,
        )

    model = build_multitask_model(
        config,
        state_vocabs=state_vocabs,
        task_vocabs=task_vocabs,
        task_meta=task_meta,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    run_name = f"multitask-{config.partition_kind or 'root'}-{config.partition_value or 'all'}-{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir = Path(config.out_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"

    best_macro_accuracy = -1.0

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_stats = {
            task: {"loss": 0.0, "correct": 0.0, "examples": 0}
            for task in config.tasks
        }

        for task, batch in iter_round_robin(train_loaders, config.tasks):
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(task, batch)
            labels = batch["labels"]
            loss, accuracy = compute_batch_loss_and_accuracy(task, logits, labels)
            loss.backward()
            optimizer.step()

            batch_size = labels.shape[0]
            train_stats[task]["loss"] += float(loss.item()) * batch_size
            train_stats[task]["correct"] += accuracy * batch_size
            train_stats[task]["examples"] += batch_size

        eval_metrics = evaluate_multitask(model, eval_loaders, device)
        train_metrics = {
            task: {
                "loss": stats["loss"] / max(stats["examples"], 1),
                "accuracy": stats["correct"] / max(stats["examples"], 1),
                "examples": stats["examples"],
            }
            for task, stats in train_stats.items()
        }
        macro_eval_accuracy = float(np.mean([metrics["accuracy"] for metrics in eval_metrics.values()])) if eval_metrics else 0.0

        epoch_metrics: dict[str, Any] = {"epoch": epoch, "eval_macro_accuracy": macro_eval_accuracy}
        for task in config.tasks:
            epoch_metrics[f"{task}_train_loss"] = train_metrics[task]["loss"]
            epoch_metrics[f"{task}_train_accuracy"] = train_metrics[task]["accuracy"]
            epoch_metrics[f"{task}_eval_loss"] = eval_metrics[task]["loss"]
            epoch_metrics[f"{task}_eval_accuracy"] = eval_metrics[task]["accuracy"]
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(epoch_metrics, ensure_ascii=False))
            handle.write("\n")

        summary_line = " ".join(
            f"{task}:train={train_metrics[task]['accuracy']:.3f}/eval={eval_metrics[task]['accuracy']:.3f}"
            for task in config.tasks
        )
        print(f"[epoch {epoch:02d}] macro_eval={macro_eval_accuracy:.4f} {summary_line}")

        if macro_eval_accuracy > best_macro_accuracy:
            best_macro_accuracy = macro_eval_accuracy
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "tasks": config.tasks,
                    "state_vocabs": {name: vocab.to_dict() for name, vocab in state_vocabs.items()},
                    "task_vocabs": {
                        task: {name: vocab.to_dict() for name, vocab in vocabs.items()}
                        for task, vocabs in task_vocabs.items()
                    },
                    "task_meta": task_meta,
                    "config": asdict(config),
                    "best_macro_accuracy": best_macro_accuracy,
                    "epoch": epoch,
                },
                run_dir / "best_model.pt",
            )

    summary = {
        "run_dir": str(run_dir),
        "tasks": config.tasks,
        "best_macro_accuracy": best_macro_accuracy,
        "train_rows": {task: len(rows) for task, rows in train_rows_by_task.items()},
        "eval_rows": {task: len(rows) for task, rows in eval_rows_by_task.items()},
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def parse_tasks(raw: str | None) -> list[str]:
    if not raw:
        return list(DEFAULT_TASKS)
    tasks = [task.strip() for task in raw.split(",") if task.strip()]
    invalid = [task for task in tasks if task not in SUPERVISED_TASKS]
    if invalid:
        raise SystemExit(f"Unsupported tasks: {invalid}")
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline multi-task supervised training for STS2.")
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS), type=str)
    parser.add_argument("--dataset-root", required=True, type=str)
    parser.add_argument("--dataset-format", default="parquet", choices=["parquet", "jsonl"])
    parser.add_argument("--partition-kind", default=None, choices=["build_id", "build_family"])
    parser.add_argument("--partition-value", default=None, type=str)
    parser.add_argument("--out-dir", default="offline_runs", type=str)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--learning-rate", default=1e-3, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--hidden-dim", default=128, type=int)
    parser.add_argument("--entity-embed-dim", default=64, type=int)
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument("--seed", default=7, type=int)
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument("--eval-split", default="val", choices=["val", "test"])
    args = parser.parse_args()

    config = OfflineMultiTaskConfig(
        tasks=parse_tasks(args.tasks),
        dataset_root=args.dataset_root,
        dataset_format=args.dataset_format,
        partition_kind=args.partition_kind,
        partition_value=args.partition_value,
        out_dir=args.out_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
        entity_embed_dim=args.entity_embed_dim,
        device=args.device,
        seed=args.seed,
        num_workers=args.num_workers,
        eval_split=args.eval_split,
    )
    summary = train(config)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
