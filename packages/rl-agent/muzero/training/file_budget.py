"""Line-count budget helpers for the MuZero refactor.

The project currently has several legacy Python files above the 2,000 line
target.  This module makes that debt explicit while preventing new oversized
files from silently appearing during future strategy/search work.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping


DEFAULT_MAX_LINES = 2_000
DEFAULT_SCAN_ROOTS = ("muzero", "sts2_env", "scripts", "tests", "legacy")
DEFAULT_EXCLUDED_DIR_NAMES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    ".venv-wsl-rocm",
    ".venv-win",
    "__pycache__",
    "checkpoints_muzero",
    "logs_audit",
    "logs_muzero",
    "node_modules",
    "venv",
}

# Existing over-budget files are tracked here so the guard can be adopted before
# the large refactor is complete.  Do not add to this list for new work; split
# the file instead.
DEFAULT_LEGACY_ALLOWLIST: dict[str, str] = {
    "legacy/train_offline_multitask.py": "legacy offline trainer; archived until legacy/ is split or removed",
    "muzero/sts2_env/muzero_model.py": "legacy network definitions; split after trainer extraction",
    "muzero/sts2_env/token_memory.py": "legacy token-memory implementation; split encoder/heads/state helpers",
    "sts2_env/_sim_translate.py": "legacy bridge translation layer",
    "sts2_env/combat_env.py": "legacy environment wrapper",
    "sts2_env/env_v2.py": "legacy full-run environment wrapper",
    "sts2_env/observation_v3.py": "legacy token observation encoder",
    "tests/test_combat_action_hard_guards.py": "legacy broad regression suite; split by guard family",
}


@dataclass(frozen=True)
class FileBudgetRecord:
    """Line-count result for one Python source file."""

    path: Path
    relative_path: Path
    line_count: int
    max_lines: int
    legacy_allowed: bool = False
    legacy_reason: str = ""

    @property
    def over_budget(self) -> bool:
        return self.line_count > self.max_lines

    @property
    def blocking_violation(self) -> bool:
        return self.over_budget and not self.legacy_allowed


def normalize_relative_path(path: str | Path) -> str:
    """Normalize a relative path for allowlist lookup."""

    return Path(path).as_posix().lstrip("./")


def count_lines(path: Path) -> int:
    """Count physical source lines with tolerant UTF-8 decoding."""

    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        return sum(1 for _ in handle)


def iter_python_files(package_root: Path, roots: Iterable[str | Path] = DEFAULT_SCAN_ROOTS) -> Iterable[Path]:
    """Yield Python files under selected package-relative roots."""

    for root in roots:
        absolute_root = package_root / root
        if absolute_root.is_file() and absolute_root.suffix == ".py":
            yield absolute_root
            continue
        if not absolute_root.exists():
            continue
        for path in absolute_root.rglob("*.py"):
            if any(part in DEFAULT_EXCLUDED_DIR_NAMES for part in path.relative_to(package_root).parts[:-1]):
                continue
            yield path


def collect_file_budget_records(
    package_root: str | Path,
    *,
    roots: Iterable[str | Path] = DEFAULT_SCAN_ROOTS,
    max_lines: int = DEFAULT_MAX_LINES,
    legacy_allowlist: Mapping[str, str] | None = None,
) -> list[FileBudgetRecord]:
    """Return sorted line-count records for all scanned Python files."""

    pkg = Path(package_root)
    allowlist = {
        normalize_relative_path(path): reason
        for path, reason in (legacy_allowlist or DEFAULT_LEGACY_ALLOWLIST).items()
    }
    records: list[FileBudgetRecord] = []
    for path in iter_python_files(pkg, roots):
        try:
            relative = path.relative_to(pkg)
        except ValueError:
            relative = path
        relative_key = normalize_relative_path(relative)
        reason = allowlist.get(relative_key, "")
        records.append(
            FileBudgetRecord(
                path=path,
                relative_path=relative,
                line_count=count_lines(path),
                max_lines=int(max_lines),
                legacy_allowed=bool(reason),
                legacy_reason=reason,
            )
        )
    return sorted(records, key=lambda record: (record.line_count, record.relative_path.as_posix()), reverse=True)


def find_file_budget_violations(
    package_root: str | Path,
    *,
    roots: Iterable[str | Path] = DEFAULT_SCAN_ROOTS,
    max_lines: int = DEFAULT_MAX_LINES,
    legacy_allowlist: Mapping[str, str] | None = None,
    include_legacy: bool = False,
) -> list[FileBudgetRecord]:
    """Return over-budget files.

    By default only non-allowlisted files are returned, which makes this usable
    as a CI guard before the historical debt has been paid down.  Set
    ``include_legacy=True`` to print the full decomposition debt table.
    """

    records = collect_file_budget_records(
        package_root,
        roots=roots,
        max_lines=max_lines,
        legacy_allowlist=legacy_allowlist,
    )
    if include_legacy:
        return [record for record in records if record.over_budget]
    return [record for record in records if record.blocking_violation]


__all__ = [
    "DEFAULT_LEGACY_ALLOWLIST",
    "DEFAULT_EXCLUDED_DIR_NAMES",
    "DEFAULT_MAX_LINES",
    "DEFAULT_SCAN_ROOTS",
    "FileBudgetRecord",
    "collect_file_budget_records",
    "count_lines",
    "find_file_budget_violations",
    "iter_python_files",
    "normalize_relative_path",
]
