#!/usr/bin/env python3
"""Clean stale MuZero training artifacts without deleting model weights.

The default mode is a dry-run.  Actual mutation requires ``--execute`` and is
conservative by design:

* checkpoint weights are preserved (``network.pt``, target encoder, metadata)
* large resume-only state can be removed (``replay_buffer.pkl``,
  ``optimizer.pt``)
* source/data/venv directories are never touched by the default categories

Use ``--permanent`` for real disk reclamation.  Without it, ``--execute`` moves
files/directories into ``.trash_training_artifacts/<timestamp>/`` for review,
which is safer but does not free space until the trash is deleted.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

RL_SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(RL_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(RL_SOURCE_ROOT))

from sts2_rl.artifacts import artifact_root, resolve_artifact_path

DEFAULT_ARTIFACT_ROOT = artifact_root()
DEFAULT_REPORT_DIR = resolve_artifact_path(None, default="reports/cleanup")
TRASH_ROOT_NAME = ".trash_training_artifacts"

CHECKPOINT_STATE_NAMES = {
    "replay_buffer.pkl",
    "optimizer.pt",
    "amp_scaler.pt",
    "scaler.pt",
}
CHECKPOINT_STATE_PREFIXES = (
    "replay",
    "buffer",
    "optimizer",
    "optim",
)
CHECKPOINT_STATE_SUFFIXES = (
    ".pkl",
    ".pt",
    ".pth",
    ".ckpt",
    ".npz",
)
ROOT_TRACE_PATTERNS = (
    "sim_rpc_trace_pid*.log",
    "sim_hang_debug_pid*.log",
    "sim_hang_dump_pid*.log",
    "sim_*trace*.jsonl",
    "sim_*debug*.jsonl",
)
CACHE_DIR_NAMES = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".hypothesis",
}
ROOT_CACHE_DIR_PATTERNS = (
    ".text_cache",
    ".tmp_text_cache*",
    ".text_cache_download_test",
    ".text_cache_offline_check",
    ".text_cache_strict_fail",
)
TEMP_DIR_NAMES = (
    "tmp",
    "_tmp_smoke",
)
SMOKE_ARTIFACT_DIRS = (
    "artifacts/full_run_sim_smoke",
)
PROTECTED_TOP_LEVEL_NAMES = {
    ".git",
    ".venv-wsl-rocm",
    "venv",
    "data",
    "docs",
    "legacy",
    "muzero",
    "scripts",
    "sts2_env",
    "tests",
}
SKIP_RECURSIVE_SCAN_TOP_LEVEL_NAMES = {
    ".git",
    TRASH_ROOT_NAME,
    ".venv-wsl-rocm",
    "venv",
    "data",
    "third_party",
}


@dataclass(frozen=True)
class Candidate:
    path: Path
    category: str
    reason: str
    size_bytes: int
    mtime: float


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _fmt_bytes(n: int | float | None) -> str:
    if not n:
        return "0 B"
    value = float(n)
    units = ["B", "KB", "MB", "GB", "TB"]
    idx = 0
    while value >= 1024.0 and idx < len(units) - 1:
        value /= 1024.0
        idx += 1
    if idx == 0:
        return f"{int(value)} {units[idx]}"
    return f"{value:.2f} {units[idx]}"


def _safe_resolve(path: Path) -> Path:
    return path.resolve(strict=False)


def _safe_exists(path: Path) -> bool:
    try:
        return path.exists()
    except OSError:
        return False


def _safe_is_file(path: Path) -> bool:
    try:
        return path.is_file()
    except OSError:
        return False


def _safe_is_dir(path: Path) -> bool:
    try:
        return path.is_dir()
    except OSError:
        return False


def _top_level_name(path: Path, root: Path) -> str:
    try:
        rel = path.relative_to(root)
    except ValueError:
        return ""
    return rel.parts[0] if rel.parts else ""


def _should_skip_recursive_scan(path: Path, root: Path) -> bool:
    return _top_level_name(path, root) in SKIP_RECURSIVE_SCAN_TOP_LEVEL_NAMES


def _is_within(child: Path, parent: Path) -> bool:
    try:
        _safe_resolve(child).relative_to(_safe_resolve(parent))
        return True
    except ValueError:
        return False


def _assert_safe_target(path: Path, root: Path) -> None:
    resolved = _safe_resolve(path)
    resolved_root = _safe_resolve(root)
    if resolved == resolved_root:
        raise RuntimeError(f"refusing to delete repository/package root: {resolved}")
    if not _is_within(resolved, resolved_root):
        raise RuntimeError(f"refusing to touch path outside root: {resolved}")
    parts = set(resolved.relative_to(resolved_root).parts)
    if ".git" in parts:
        raise RuntimeError(f"refusing to touch .git path: {resolved}")


def _file_size(path: Path) -> int:
    try:
        return int(path.stat().st_size)
    except OSError:
        return 0


def _dir_size(path: Path) -> int:
    total = 0
    for file in path.rglob("*"):
        try:
            if file.is_file():
                total += int(file.stat().st_size)
        except OSError:
            continue
    return total


def _path_size(path: Path) -> int:
    if _safe_is_file(path):
        return _file_size(path)
    if _safe_is_dir(path):
        return _dir_size(path)
    return 0


def _mtime(path: Path) -> float:
    try:
        return float(path.stat().st_mtime)
    except OSError:
        return 0.0


def _is_old_enough(path: Path, min_age_hours: float) -> bool:
    if min_age_hours <= 0:
        return True
    age_s = datetime.now().timestamp() - _mtime(path)
    return age_s >= min_age_hours * 3600.0


def _checkpoint_state_file(path: Path) -> bool:
    name = path.name
    lower = name.lower()
    if lower in CHECKPOINT_STATE_NAMES:
        return True
    if not lower.endswith(CHECKPOINT_STATE_SUFFIXES):
        return False
    return any(lower.startswith(prefix) for prefix in CHECKPOINT_STATE_PREFIXES)


def _run_name_for_checkpoint_file(path: Path, ckpt_root: Path) -> str:
    try:
        rel = path.relative_to(ckpt_root)
    except ValueError:
        return ""
    return rel.parts[0] if rel.parts else ""


def _matches_any(value: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch.fnmatch(value, pattern) for pattern in patterns)


def gather_candidates(args: argparse.Namespace, root: Path) -> list[Candidate]:
    out: list[Candidate] = []

    keep_run_patterns = list(args.keep_run or [])

    def add(path: Path, category: str, reason: str) -> None:
        if not _safe_exists(path):
            return
        if not _is_old_enough(path, float(args.min_age_hours or 0.0)):
            return
        _assert_safe_target(path, root)
        out.append(
            Candidate(
                path=_safe_resolve(path),
                category=category,
                reason=reason,
                size_bytes=_path_size(path),
                mtime=_mtime(path),
            )
        )

    checkpoint_roots = tuple(
        path
        for path in (root / "checkpoints", root / "checkpoints_muzero")
        if _safe_exists(path)
    )

    if args.delete_checkpoint_state:
        for ckpt_root in checkpoint_roots:
            for path in ckpt_root.rglob("*"):
                if not _safe_is_file(path) or not _checkpoint_state_file(path):
                    continue
                run_name = _run_name_for_checkpoint_file(path, ckpt_root)
                if keep_run_patterns and _matches_any(run_name, keep_run_patterns):
                    continue
                add(
                    path,
                    "checkpoint_state",
                    "resume-only MuZero state; weights/metadata remain",
                )

    if args.delete_root_traces:
        for pattern in ROOT_TRACE_PATTERNS:
            for path in root.glob(pattern):
                if _safe_is_file(path):
                    add(path, "root_trace_logs", f"root trace pattern {pattern}")

    if args.delete_caches:
        for path in root.rglob("*"):
            if _should_skip_recursive_scan(path, root):
                continue
            if not _safe_is_dir(path):
                continue
            name = path.name
            if name in CACHE_DIR_NAMES or fnmatch.fnmatch(name, "pytest-cache-files-*"):
                add(path, "cache_dirs", f"cache directory {name}")
        for pattern in ROOT_CACHE_DIR_PATTERNS:
            for path in root.glob(pattern):
                if _safe_is_dir(path):
                    add(path, "cache_dirs", f"root cache pattern {pattern}")

    if args.delete_temp:
        for name in TEMP_DIR_NAMES:
            path = root / name
            if _safe_exists(path) and _safe_is_dir(path):
                add(path, "temp_dirs", f"temporary training directory {name}")

    if args.delete_smoke_artifacts:
        for rel in SMOKE_ARTIFACT_DIRS:
            path = root / rel
            if _safe_exists(path) and _safe_is_dir(path):
                add(path, "smoke_artifacts", f"old smoke artifact dir {rel}")

    if args.delete_zero_byte_checkpoint_dirs:
        for ckpt_root in checkpoint_roots:
            # Walk deepest dirs first.  Only dirs whose recursive size is zero
            # are candidates, so real weights are never deleted here.
            dirs = [p for p in ckpt_root.rglob("*") if _safe_is_dir(p)]
            for path in sorted(dirs, key=lambda p: len(p.parts), reverse=True):
                if _path_size(path) == 0:
                    add(path, "empty_checkpoint_dirs", "zero-byte checkpoint directory")

    if args.delete_old_logs_older_than_days is not None:
        days = float(args.delete_old_logs_older_than_days)
        cutoff = datetime.now().timestamp() - days * 86400.0
        for log_root_name in ("runs", "logs_muzero", "logs_attention", "train_logs"):
            log_root = root / log_root_name
            if not _safe_exists(log_root):
                continue
            for path in log_root.iterdir():
                if not _safe_is_dir(path):
                    continue
                if _mtime(path) <= cutoff:
                    add(
                        path,
                        "old_training_logs",
                        f"{log_root_name} run older than {days:g} days",
                    )

    # Deduplicate nested candidates.  If a parent directory is already selected,
    # child candidates under it are redundant and would fail after the parent
    # move/delete.
    by_path: dict[Path, Candidate] = {}
    for cand in out:
        by_path[cand.path] = cand
    candidates = sorted(by_path.values(), key=lambda c: (len(c.path.parts), str(c.path)))
    kept: list[Candidate] = []
    for cand in candidates:
        if any(_is_within(cand.path, parent.path) for parent in kept if parent.path.is_dir()):
            continue
        kept.append(cand)
    return kept


def _quarantine_path(path: Path, root: Path, trash_root: Path) -> Path:
    rel = _safe_resolve(path).relative_to(_safe_resolve(root))
    return trash_root / rel


def apply_candidates(
    candidates: list[Candidate],
    *,
    root: Path,
    dry_run: bool,
    permanent: bool,
    trash_root: Path,
) -> list[dict[str, object]]:
    actions: list[dict[str, object]] = []
    if not dry_run and not permanent:
        trash_root.mkdir(parents=True, exist_ok=True)
    for cand in sorted(candidates, key=lambda c: len(c.path.parts), reverse=True):
        _assert_safe_target(cand.path, root)
        action = "dry_run"
        ok = True
        error = ""
        dest = ""
        if not dry_run:
            try:
                if permanent:
                    action = "deleted"
                    if cand.path.is_dir():
                        shutil.rmtree(cand.path)
                    else:
                        cand.path.unlink(missing_ok=True)
                else:
                    action = "quarantined"
                    qpath = _quarantine_path(cand.path, root, trash_root)
                    qpath.parent.mkdir(parents=True, exist_ok=True)
                    if qpath.exists():
                        suffix = 1
                        while qpath.with_name(qpath.name + f".dup{suffix}").exists():
                            suffix += 1
                        qpath = qpath.with_name(qpath.name + f".dup{suffix}")
                    shutil.move(str(cand.path), str(qpath))
                    dest = str(qpath)
            except Exception as exc:  # pragma: no cover - operator-facing
                ok = False
                error = f"{type(exc).__name__}: {exc}"
        actions.append(
            {
                "path": str(cand.path),
                "category": cand.category,
                "reason": cand.reason,
                "size_bytes": cand.size_bytes,
                "size": _fmt_bytes(cand.size_bytes),
                "mtime": datetime.fromtimestamp(cand.mtime).isoformat() if cand.mtime else "",
                "action": action,
                "ok": ok,
                "dest": dest,
                "error": error,
            }
        )
    return actions


def write_report(
    *,
    report_dir: Path,
    stamp: str,
    root: Path,
    args: argparse.Namespace,
    actions: list[dict[str, object]],
) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"cleanup_report_{stamp}.md"
    json_path = report_dir / f"cleanup_report_{stamp}.json"

    total = sum(int(a.get("size_bytes") or 0) for a in actions)
    by_category: dict[str, tuple[int, int]] = {}
    for action in actions:
        cat = str(action.get("category") or "unknown")
        count, size = by_category.get(cat, (0, 0))
        by_category[cat] = (count + 1, size + int(action.get("size_bytes") or 0))

    serializable_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "args": serializable_args,
        "total_count": len(actions),
        "total_size_bytes": total,
        "actions": actions,
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Training Artifact Cleanup Report",
        "",
        f"- Generated: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- Root: `{root}`",
        f"- Mode: `{'dry-run' if not args.execute else ('permanent delete' if args.permanent else 'quarantine')}`",
        f"- Candidate count: **{len(actions)}**",
        f"- Candidate size: **{_fmt_bytes(total)}**",
        f"- JSON details: `{json_path.name}`",
        "",
        "## Totals by category",
        "",
        "| Category | Count | Size |",
        "|---|---:|---:|",
    ]
    for cat, (count, size) in sorted(by_category.items(), key=lambda item: item[1][1], reverse=True):
        lines.append(f"| `{cat}` | {count} | {_fmt_bytes(size)} |")
    lines.extend(["", "## Top paths", "", "| Action | Size | Category | Path |", "|---|---:|---|---|"])
    for action in sorted(actions, key=lambda a: int(a.get("size_bytes") or 0), reverse=True)[:200]:
        ok = "ok" if action.get("ok") else f"ERR {action.get('error')}"
        lines.append(
            f"| {action.get('action')}:{ok} | {action.get('size')} | "
            f"`{action.get('category')}` | `{action.get('path')}` |"
        )
    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ARTIFACT_ROOT,
        help="Artifact tree to clean (default: STS2_ARTIFACT_ROOT or ~/.sts2-artifacts).",
    )
    parser.add_argument("--execute", action="store_true", help="actually mutate files; default is dry-run")
    parser.add_argument("--permanent", action="store_true", help="delete permanently instead of quarantine")
    parser.add_argument("--quarantine", action="store_true", help="explicitly select quarantine mode (default for --execute without --permanent)")
    parser.add_argument("--trash-root", type=Path, default=None, help="quarantine root; default: <root>/.trash_training_artifacts/<stamp>")
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--min-age-hours", type=float, default=0.0, help="skip files/dirs modified more recently than this")
    parser.add_argument("--keep-run", action="append", default=[], help="glob of checkpoint run names to skip")

    parser.add_argument("--delete-checkpoint-state", action="store_true", help="remove replay_buffer.pkl/optimizer.pt-like files under checkpoints")
    parser.add_argument("--delete-caches", action="store_true", help="remove __pycache__/.pytest_cache/text-cache scratch dirs")
    parser.add_argument("--delete-root-traces", action="store_true", help="remove root sim_hang/sim_rpc trace logs")
    parser.add_argument("--delete-temp", action="store_true", help="remove tmp and _tmp_smoke training scratch dirs")
    parser.add_argument("--delete-smoke-artifacts", action="store_true", help="remove known old smoke artifact output dirs")
    parser.add_argument("--delete-zero-byte-checkpoint-dirs", action="store_true", help="remove empty dirs under checkpoints")
    parser.add_argument("--delete-old-logs-older-than-days", type=float, default=None, help="remove run directories older than N days")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    root = resolve_artifact_path(args.root)
    if not root.exists():
        raise SystemExit(f"root does not exist: {root}")
    if args.permanent and not args.execute:
        raise SystemExit("--permanent requires --execute")

    stamp = _now_stamp()
    trash_root = (
        resolve_artifact_path(args.trash_root)
        if args.trash_root is not None
        else root / TRASH_ROOT_NAME / stamp
    )
    if not _is_within(trash_root, root):
        raise SystemExit(f"trash root must stay below cleanup root: {trash_root}")
    candidates = gather_candidates(args, root)
    actions = apply_candidates(
        candidates,
        root=root,
        dry_run=not args.execute,
        permanent=bool(args.permanent),
        trash_root=_safe_resolve(trash_root),
    )
    report_path = write_report(
        report_dir=resolve_artifact_path(args.report_dir),
        stamp=stamp,
        root=root,
        args=args,
        actions=actions,
    )
    total = sum(int(a.get("size_bytes") or 0) for a in actions)
    mode = "dry-run" if not args.execute else ("permanent delete" if args.permanent else "quarantine")
    print(
        json.dumps(
            {
                "status": "ok",
                "mode": mode,
                "root": str(root),
                "candidate_count": len(actions),
                "candidate_size_bytes": total,
                "candidate_size": _fmt_bytes(total),
                "report": str(report_path),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
