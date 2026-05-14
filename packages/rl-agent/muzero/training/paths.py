"""Centralized filesystem layout helpers for MuZero training.

The legacy ``muzero.train`` module used to construct log, checkpoint, replay,
diagnostic, and policy-module paths inline.  The entrypoint is now intentionally
thin; new code should depend on these small dataclasses instead of adding more
hard-coded ``Path(...) / ...`` snippets to training, strategy, or search wiring.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any


def _expand_path(value: str | os.PathLike[str] | Path) -> Path:
    """Expand env vars and ``~`` without resolving symlinks or requiring exist."""

    return Path(os.path.expandvars(os.fspath(value))).expanduser()


def _resolve_under(base: Path, value: str | os.PathLike[str] | Path | None) -> Path | None:
    """Resolve a possibly-relative path under ``base``.

    ``Path.resolve`` is intentionally avoided because many training paths are
    created lazily and may not exist when configuration is parsed.
    """

    if value in (None, ""):
        return None
    path = _expand_path(value)
    if path.is_absolute():
        return path
    return base / path


def _module_file(directory: Path, module_name: str | os.PathLike[str] | Path) -> Path:
    """Return a source-file path under ``directory``.

    This is intentionally conservative: module names may point at nested
    relative paths, but they may not escape their owning package directory.
    Passing ``"bias"`` and ``"bias.py"`` both returns ``<directory>/bias.py``.
    """

    raw = Path(os.fspath(module_name))
    # ``Path.is_absolute()`` is platform-specific.  On Windows, a POSIX-style
    # path such as ``/tmp/root_prior.py`` has a root/anchor but is not
    # considered absolute because it has no drive.  Treat any anchored path as
    # an escape attempt so tests and tooling behave identically on Windows and
    # WSL/Linux.
    if raw.is_absolute() or raw.anchor:
        raise ValueError(f"module path must be relative, got {module_name!r}")
    if any(part in {"", ".", ".."} for part in raw.parts):
        raise ValueError(f"module path must stay inside {directory}, got {module_name!r}")
    if raw.suffix and raw.suffix != ".py":
        raise ValueError(f"module path must be a Python file, got {module_name!r}")
    if not raw.suffix:
        raw = raw.with_suffix(".py")
    return directory / raw


def _find_repo_root(start: Path) -> Path:
    """Best-effort repo-root discovery from a package or module path."""

    cursor = start if start.is_dir() else start.parent
    for parent in (cursor, *cursor.parents):
        if (parent / ".git").exists():
            return parent
        if (parent / "packages" / "rl-agent").exists():
            return parent
    # Fallback for the expected layout: <repo>/packages/rl-agent/...
    parts = cursor.parts
    if len(parts) >= 3 and parts[-2:] == ("packages", "rl-agent"):
        return cursor.parent.parent
    if cursor.name == "rl-agent" and cursor.parent.name == "packages":
        return cursor.parent.parent
    return cursor


def default_package_root() -> Path:
    """Return the ``packages/rl-agent`` directory for this checkout."""

    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class RunPaths:
    """Runtime artifact paths for one MuZero training run.

    Relative CLI paths are interpreted under ``package_root`` to preserve the
    current launch-script behavior from ``packages/rl-agent``.
    """

    repo_root: Path
    package_root: Path
    log_dir: Path
    checkpoint_dir: Path
    resume_from: Path | None = None

    @classmethod
    def from_args(
        cls,
        args: Any,
        *,
        package_root: str | os.PathLike[str] | Path | None = None,
        repo_root: str | os.PathLike[str] | Path | None = None,
    ) -> "RunPaths":
        """Build paths from an argparse-like object.

        Expected attributes are ``log_dir``, ``checkpoint_dir`` and optional
        ``resume_from``.  This loose contract lets tests pass a simple object and
        keeps this module independent from the large CLI parser during refactor.
        """

        pkg = _expand_path(package_root) if package_root is not None else default_package_root()
        repo = _expand_path(repo_root) if repo_root is not None else _find_repo_root(pkg)
        log_dir = _resolve_under(pkg, getattr(args, "log_dir", None))
        checkpoint_dir = _resolve_under(pkg, getattr(args, "checkpoint_dir", None))
        if log_dir is None:
            raise ValueError("RunPaths.from_args requires args.log_dir")
        if checkpoint_dir is None:
            raise ValueError("RunPaths.from_args requires args.checkpoint_dir")
        return cls(
            repo_root=repo,
            package_root=pkg,
            log_dir=log_dir,
            checkpoint_dir=checkpoint_dir,
            resume_from=_resolve_under(pkg, getattr(args, "resume_from", None)),
        )

    @property
    def diagnostics_dir(self) -> Path:
        return self.log_dir / "diagnostics"

    @property
    def tensorboard_event_dir(self) -> Path:
        return self.log_dir

    def checkpoint_step_dir(self, step: int) -> Path:
        return self.checkpoint_dir / f"muzero_step_{int(step):08d}"

    def async_actor_log_dir(self, actor_index: int) -> Path:
        """Scratch TensorBoard/log directory for one async self-play actor."""

        return self.log_dir / "async_actor_scratch" / f"actor_{int(actor_index)}"

    def async_actor_checkpoint_dir(self, actor_index: int) -> Path:
        """Scratch checkpoint directory for one async self-play actor."""

        return self.checkpoint_dir / "async_actor_scratch" / f"actor_{int(actor_index)}"

    def replay_buffer_path(self, checkpoint_dir: str | os.PathLike[str] | Path | None = None) -> Path:
        base = _expand_path(checkpoint_dir) if checkpoint_dir is not None else self.checkpoint_dir
        return base / "replay_buffer.pkl"

    def diagnostic_jsonl(self, filename: str) -> Path:
        if not filename.endswith(".jsonl"):
            filename = f"{filename}.jsonl"
        return self.diagnostics_dir / filename

    def ensure_dirs(self) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.diagnostics_dir.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class StrategyModulePaths:
    """Canonical source directories for combat strategy modules.

    Put card/action-quality, boss-mechanic, potion-timing, HP-cost, X-cost, and
    other *decision policy* helpers behind this class.  New strategy code should
    never discover paths by walking from ``train.py``.
    """

    package_root: Path
    muzero_root: Path
    strategy_dir: Path
    strategy_encounters_dir: Path
    combat_quality_dir: Path
    diagnostics_dir: Path

    @classmethod
    def from_package_root(
        cls,
        package_root: str | os.PathLike[str] | Path | None = None,
    ) -> "StrategyModulePaths":
        pkg = _expand_path(package_root) if package_root is not None else default_package_root()
        muzero_root = pkg / "muzero"
        return cls(
            package_root=pkg,
            muzero_root=muzero_root,
            strategy_dir=muzero_root / "strategy",
            strategy_encounters_dir=muzero_root / "strategy" / "encounters",
            combat_quality_dir=muzero_root / "combat_quality",
            diagnostics_dir=muzero_root / "diagnostics",
        )

    def strategy_file(self, module_name: str | os.PathLike[str] | Path) -> Path:
        return _module_file(self.strategy_dir, module_name)

    def encounter_file(self, module_name: str | os.PathLike[str] | Path) -> Path:
        return _module_file(self.strategy_encounters_dir, module_name)

    def combat_quality_file(self, module_name: str | os.PathLike[str] | Path) -> Path:
        return _module_file(self.combat_quality_dir, module_name)

    def diagnostics_file(self, module_name: str | os.PathLike[str] | Path) -> Path:
        return _module_file(self.diagnostics_dir, module_name)

    def all_strategy_dirs(self) -> tuple[Path, ...]:
        return (
            self.strategy_dir,
            self.strategy_encounters_dir,
            self.combat_quality_dir,
            self.diagnostics_dir,
        )

    def ensure_package_dirs(self) -> None:
        """Create only source directories, never logs/checkpoints/buffers."""

        for path in self.all_strategy_dirs():
            path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class HeuristicSearchModulePaths:
    """Canonical source directories for route/search heuristic modules.

    Route graph/candidate/scoring/safety/bias code belongs under
    ``route_heuristics``.  Root-prior composition and search adapters belong
    under ``search``.  This keeps STS2-specific heuristics out of the generic
    MCTS implementation and out of ``train.py``.
    """

    package_root: Path
    muzero_root: Path
    route_heuristics_dir: Path
    search_dir: Path
    legacy_sts2_route_heuristic_file: Path

    @classmethod
    def from_package_root(
        cls,
        package_root: str | os.PathLike[str] | Path | None = None,
    ) -> "HeuristicSearchModulePaths":
        pkg = _expand_path(package_root) if package_root is not None else default_package_root()
        muzero_root = pkg / "muzero"
        return cls(
            package_root=pkg,
            muzero_root=muzero_root,
            route_heuristics_dir=muzero_root / "route_heuristics",
            search_dir=muzero_root / "search",
            legacy_sts2_route_heuristic_file=pkg / "sts2_env" / "route_heuristic.py",
        )

    def route_heuristic_file(self, module_name: str | os.PathLike[str] | Path) -> Path:
        return _module_file(self.route_heuristics_dir, module_name)

    def search_file(self, module_name: str | os.PathLike[str] | Path) -> Path:
        return _module_file(self.search_dir, module_name)

    def all_heuristic_dirs(self) -> tuple[Path, ...]:
        return (self.route_heuristics_dir, self.search_dir)

    def ensure_package_dirs(self) -> None:
        """Create only source directories, never logs/checkpoints/buffers."""

        for path in self.all_heuristic_dirs():
            path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class PolicyModulePaths:
    """Backward-compatible aggregate of strategy and heuristic-search paths."""

    package_root: Path
    muzero_root: Path
    strategy_dir: Path
    strategy_encounters_dir: Path
    combat_quality_dir: Path
    route_heuristics_dir: Path
    search_dir: Path
    diagnostics_dir: Path
    training_dir: Path
    legacy_sts2_route_heuristic_file: Path

    @classmethod
    def from_package_root(
        cls,
        package_root: str | os.PathLike[str] | Path | None = None,
    ) -> "PolicyModulePaths":
        pkg = _expand_path(package_root) if package_root is not None else default_package_root()
        muzero_root = pkg / "muzero"
        strategy = StrategyModulePaths.from_package_root(pkg)
        heuristic = HeuristicSearchModulePaths.from_package_root(pkg)
        return cls(
            package_root=pkg,
            muzero_root=muzero_root,
            strategy_dir=strategy.strategy_dir,
            strategy_encounters_dir=strategy.strategy_encounters_dir,
            combat_quality_dir=strategy.combat_quality_dir,
            route_heuristics_dir=heuristic.route_heuristics_dir,
            search_dir=heuristic.search_dir,
            diagnostics_dir=strategy.diagnostics_dir,
            training_dir=muzero_root / "training",
            legacy_sts2_route_heuristic_file=heuristic.legacy_sts2_route_heuristic_file,
        )

    @property
    def strategy(self) -> StrategyModulePaths:
        return StrategyModulePaths(
            package_root=self.package_root,
            muzero_root=self.muzero_root,
            strategy_dir=self.strategy_dir,
            strategy_encounters_dir=self.strategy_encounters_dir,
            combat_quality_dir=self.combat_quality_dir,
            diagnostics_dir=self.diagnostics_dir,
        )

    @property
    def heuristic_search(self) -> HeuristicSearchModulePaths:
        return HeuristicSearchModulePaths(
            package_root=self.package_root,
            muzero_root=self.muzero_root,
            route_heuristics_dir=self.route_heuristics_dir,
            search_dir=self.search_dir,
            legacy_sts2_route_heuristic_file=self.legacy_sts2_route_heuristic_file,
        )

    def ensure_package_dirs(self) -> None:
        """Create only source directories, never logs/checkpoints/buffers."""

        for path in (*self.strategy.all_strategy_dirs(), *self.heuristic_search.all_heuristic_dirs(), self.training_dir):
            path.mkdir(parents=True, exist_ok=True)

    def all_policy_dirs(self) -> tuple[Path, ...]:
        return (
            self.strategy_dir,
            self.strategy_encounters_dir,
            self.combat_quality_dir,
            self.route_heuristics_dir,
            self.search_dir,
            self.diagnostics_dir,
            self.training_dir,
        )


__all__ = [
    "HeuristicSearchModulePaths",
    "PolicyModulePaths",
    "RunPaths",
    "StrategyModulePaths",
    "default_package_root",
]
