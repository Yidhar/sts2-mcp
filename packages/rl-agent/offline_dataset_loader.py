"""Minimal loaders for STS2 offline JSONL / Parquet datasets.

These loaders are intentionally simple. They return raw dict rows so downstream
training code can own task-specific tokenization, feature packing, and labels.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from torch.utils.data import Dataset


def resolve_dataset_path(
    root: str | Path,
    dataset_name: str,
    *,
    fmt: str = "parquet",
    partition_kind: str | None = None,
    partition_value: str | None = None,
) -> Path:
    root_path = Path(root)
    suffix = ".parquet" if fmt == "parquet" else ".jsonl"
    filename = dataset_name if dataset_name.endswith(suffix) else f"{dataset_name}{suffix}"

    if partition_kind is None:
        return root_path / filename

    if partition_value is None:
        raise ValueError("partition_value is required when partition_kind is set")

    if partition_kind not in {"build_id", "build_family"}:
        raise ValueError(f"Unsupported partition_kind: {partition_kind}")

    return root_path / f"by_{partition_kind}" / partition_value / filename


def list_partitions(root: str | Path, partition_kind: str) -> list[str]:
    if partition_kind not in {"build_id", "build_family"}:
        raise ValueError(f"Unsupported partition_kind: {partition_kind}")
    partition_root = Path(root) / f"by_{partition_kind}"
    if not partition_root.exists():
        return []
    return sorted(path.name for path in partition_root.iterdir() if path.is_dir())


class JsonlDictDataset(Dataset):
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        with self.path.open("r", encoding="utf-8") as handle:
            self._rows = [json.loads(line) for line in handle if line.strip()]

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._rows[index]


class ParquetDictDataset(Dataset):
    def __init__(
        self,
        path: str | Path,
        *,
        columns: list[str] | None = None,
        json_columns: list[str] | None = None,
    ) -> None:
        self.path = Path(path)
        self._table = pq.read_table(self.path, columns=columns, memory_map=True)
        self.columns = columns
        self.json_columns = json_columns if json_columns is not None else _discover_json_columns(self.path)

    def __len__(self) -> int:
        return self._table.num_rows

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self._table.slice(index, 1).to_pylist()[0]
        for column in self.json_columns:
            value = row.get(column)
            if isinstance(value, str):
                row[column] = json.loads(value)
        return row


def _discover_json_columns(path: Path) -> list[str]:
    current = path.parent
    while True:
        manifest_path = current / "parquet_manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            root = Path(manifest["output_root"])
            try:
                relative_path = path.relative_to(root).as_posix()
            except ValueError:
                relative_path = path.name
            for entry in manifest.get("files", []):
                if entry.get("relative_path") == relative_path.replace(".parquet", ".jsonl"):
                    return list(entry.get("json_encoded_columns") or [])
            return []
        if current.parent == current:
            return []
        current = current.parent


def load_dataset(
    root: str | Path,
    dataset_name: str,
    *,
    fmt: str = "parquet",
    partition_kind: str | None = None,
    partition_value: str | None = None,
    columns: list[str] | None = None,
) -> Dataset:
    path = resolve_dataset_path(
        root,
        dataset_name,
        fmt=fmt,
        partition_kind=partition_kind,
        partition_value=partition_value,
    )
    if fmt == "parquet":
        return ParquetDictDataset(path, columns=columns)
    if fmt == "jsonl":
        return JsonlDictDataset(path)
    raise ValueError(f"Unsupported format: {fmt}")
