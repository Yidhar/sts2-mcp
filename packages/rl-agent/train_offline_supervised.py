"""Offline supervised training entrypoints for STS2 route and build datasets."""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from torch.utils.data import DataLoader

from offline_training_data import (
    ACTION_ONLY_CARD_TASKS,
    CLASSIFICATION_TASKS,
    CANDIDATE_TASKS,
    SUPERVISED_TASKS,
    OfflineRowsDataset,
    build_state_vocabs,
    build_output_vocabs,
    build_task_metadata,
    candidate_accuracy,
    cardset_exact_match_accuracy,
    classification_accuracy,
    load_task_rows,
    make_collate_fn,
)
from offline_training_models import MultiTaskOfflineModel, OfflineStateEncoder
from sts2_rl.artifacts import resolve_artifact_path, resolve_external_input_path, validate_artifact_component


@dataclass
class OfflineTrainConfig:
    task: str
    dataset_root: str
    dataset_format: str = "parquet"
    partition_kind: str | None = None
    partition_value: str | None = None
    out_dir: str | None = None
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


def build_model(
    config: OfflineTrainConfig,
    *,
    state_vocabs,
    output_vocabs,
    task_meta,
) -> MultiTaskOfflineModel:
    state_encoder = OfflineStateEncoder(
        scalar_dim=task_meta["scalar_dim"],
        card_vocab_size=state_vocabs["card"].size,
        relic_vocab_size=state_vocabs["relic"].size,
        monster_vocab_size=state_vocabs["monster"].size,
        room_type_vocab_size=state_vocabs["room_type"].size,
        point_type_vocab_size=state_vocabs["map_point_type"].size,
        room_model_vocab_size=state_vocabs["room_model_id"].size,
        embed_dim=config.entity_embed_dim,
        hidden_dim=config.hidden_dim,
    )

    if config.task in CANDIDATE_TASKS:
        candidate_vocab_size = state_vocabs["card"].size if config.task == "card_choice" else output_vocabs["choice"].size
        head_spec = {"family": "candidate", "candidate_vocab_size": candidate_vocab_size}
    elif config.task in CLASSIFICATION_TASKS:
        head_spec = {"family": task_meta["task_family"], "num_labels": output_vocabs["label"].size}
    elif config.task in ACTION_ONLY_CARD_TASKS:
        head_spec = {
            "family": "cardset",
            "card_vocab_size": state_vocabs["card"].size,
            "slot_count": task_meta["slot_count"],
        }
    else:
        raise ValueError(f"Unsupported task: {config.task}")

    return MultiTaskOfflineModel(
        state_encoder=state_encoder,
        head_builders={config.task: head_spec},
        embed_dim=config.entity_embed_dim,
        hidden_dim=config.hidden_dim,
    )


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


def evaluate_model(model: MultiTaskOfflineModel, loader: DataLoader, task: str, device: torch.device) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    total_correct = 0.0

    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            logits = model(task, batch)
            labels = batch["labels"]
            loss, accuracy = compute_batch_loss_and_accuracy(task, logits, labels)
            batch_size = labels.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_examples += batch_size
            total_correct += accuracy * batch_size

    if total_examples == 0:
        return {"loss": 0.0, "accuracy": 0.0}
    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
    }


def train(config: OfflineTrainConfig) -> dict[str, Any]:
    set_seed(config.seed)
    device = torch.device(config.device)
    dataset_root = str(resolve_external_input_path(config.dataset_root))

    train_rows = load_task_rows(
        dataset_root,
        config.task,
        fmt=config.dataset_format,
        partition_kind=config.partition_kind,
        partition_value=config.partition_value,
        split="train",
    )
    eval_rows = load_task_rows(
        dataset_root,
        config.task,
        fmt=config.dataset_format,
        partition_kind=config.partition_kind,
        partition_value=config.partition_value,
        split=config.eval_split,
    )
    if not eval_rows:
        eval_rows = load_task_rows(
            dataset_root,
            config.task,
            fmt=config.dataset_format,
            partition_kind=config.partition_kind,
            partition_value=config.partition_value,
            split="test",
        )

    if not train_rows:
        raise SystemExit(f"No training rows found for task={config.task}")
    if not eval_rows:
        raise SystemExit(f"No evaluation rows found for task={config.task}")

    all_rows = train_rows + eval_rows
    state_vocabs = build_state_vocabs(all_rows)
    output_vocabs = build_output_vocabs(all_rows, config.task)
    vocabs = {**state_vocabs, **output_vocabs}
    task_meta = build_task_metadata(all_rows, config.task)
    collate_fn = make_collate_fn(config.task, vocabs, task_meta)
    train_ds = OfflineRowsDataset(train_rows)
    eval_ds = OfflineRowsDataset(eval_rows)
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
    )

    model = build_model(
        config,
        state_vocabs=state_vocabs,
        output_vocabs=output_vocabs,
        task_meta=task_meta,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    partition_value = (
        validate_artifact_component(config.partition_value, label="offline partition_value")
        if config.partition_value
        else "all"
    )
    run_name = f"{config.task}-{config.partition_kind or 'root'}-{partition_value}-{time.strftime('%Y%m%d-%H%M%S')}"
    output_root = resolve_artifact_path(config.out_dir, default="offline_runs")
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"

    best_eval_acc = -1.0

    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss = 0.0
        total_examples = 0
        total_correct = 0.0

        for batch in train_loader:
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(config.task, batch)
            labels = batch["labels"]
            loss, accuracy = compute_batch_loss_and_accuracy(config.task, logits, labels)
            loss.backward()
            optimizer.step()

            batch_size = labels.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_examples += batch_size
            total_correct += accuracy * batch_size

        train_metrics = {
            "loss": total_loss / max(total_examples, 1),
            "accuracy": total_correct / max(total_examples, 1),
        }
        eval_metrics = evaluate_model(model, eval_loader, config.task, device)
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "eval_loss": eval_metrics["loss"],
            "eval_accuracy": eval_metrics["accuracy"],
        }
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(epoch_metrics, ensure_ascii=False))
            handle.write("\n")

        print(
            f"[epoch {epoch:02d}] "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']:.4f} "
            f"eval_loss={eval_metrics['loss']:.4f} eval_acc={eval_metrics['accuracy']:.4f}"
        )

        if eval_metrics["accuracy"] > best_eval_acc:
            best_eval_acc = eval_metrics["accuracy"]
            save_file(model.state_dict(), str(run_dir / "model.safetensors"))
            (run_dir / "metadata.json").write_text(
                json.dumps(
                    {
                        "format": "sts2-offline-task-v1",
                        "task": config.task,
                        "state_vocabs": {name: vocab.to_dict() for name, vocab in state_vocabs.items()},
                        "output_vocabs": {name: vocab.to_dict() for name, vocab in output_vocabs.items()},
                        "task_meta": task_meta,
                        "config": {**asdict(config), "out_dir": str(output_root)},
                        "best_eval_accuracy": best_eval_acc,
                        "epoch": epoch,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

    summary = {
        "run_dir": str(run_dir),
        "task": config.task,
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "best_eval_accuracy": best_eval_acc,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline supervised training for STS2 route/card/build tasks.")
    parser.add_argument("--task", required=True, choices=sorted(SUPERVISED_TASKS))
    parser.add_argument("--dataset-root", required=True, type=str)
    parser.add_argument("--dataset-format", default="parquet", choices=["parquet", "jsonl"])
    parser.add_argument("--partition-kind", default=None, choices=["build_id", "build_family"])
    parser.add_argument("--partition-value", default=None, type=str)
    parser.add_argument(
        "--out-dir",
        default=None,
        type=str,
        help="Output directory (default: <STS2_ARTIFACT_ROOT>/offline_runs). Relative paths use the artifact root.",
    )
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

    config = OfflineTrainConfig(
        task=args.task,
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
