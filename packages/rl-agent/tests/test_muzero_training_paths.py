from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from muzero.training.file_budget import (
    DEFAULT_LEGACY_ALLOWLIST,
    collect_file_budget_records,
    find_file_budget_violations,
)
from muzero.training.paths import (
    HeuristicSearchModulePaths,
    PolicyModulePaths,
    RunPaths,
    StrategyModulePaths,
)


def test_run_paths_from_args_resolves_relative_paths_under_package_root(tmp_path: Path) -> None:
    package_root = tmp_path / "repo" / "packages" / "rl-agent"
    args = SimpleNamespace(
        log_dir="logs_muzero/run_a",
        checkpoint_dir="checkpoints_muzero/run_a",
        resume_from="checkpoints_muzero/base/muzero_step_00000042",
    )

    paths = RunPaths.from_args(args, package_root=package_root, repo_root=tmp_path / "repo")

    assert paths.log_dir == package_root / "logs_muzero" / "run_a"
    assert paths.checkpoint_dir == package_root / "checkpoints_muzero" / "run_a"
    assert paths.resume_from == package_root / "checkpoints_muzero" / "base" / "muzero_step_00000042"
    assert paths.diagnostic_jsonl("loss_spikes") == paths.log_dir / "diagnostics" / "loss_spikes.jsonl"
    assert paths.checkpoint_step_dir(42) == paths.checkpoint_dir / "muzero_step_00000042"
    assert paths.async_actor_log_dir(2) == paths.log_dir / "async_actor_scratch" / "actor_2"
    assert paths.async_actor_checkpoint_dir(2) == paths.checkpoint_dir / "async_actor_scratch" / "actor_2"


def test_run_paths_ensure_dirs_creates_artifact_directories(tmp_path: Path) -> None:
    paths = RunPaths(
        repo_root=tmp_path,
        package_root=tmp_path / "packages" / "rl-agent",
        log_dir=tmp_path / "logs" / "run_a",
        checkpoint_dir=tmp_path / "checkpoints" / "run_a",
    )

    paths.ensure_dirs()

    assert paths.log_dir.is_dir()
    assert paths.checkpoint_dir.is_dir()
    assert paths.diagnostics_dir.is_dir()


def test_policy_module_paths_are_canonical_and_source_only(tmp_path: Path) -> None:
    package_root = tmp_path / "repo" / "packages" / "rl-agent"
    paths = PolicyModulePaths.from_package_root(package_root)

    assert paths.strategy_dir == package_root / "muzero" / "strategy"
    assert paths.strategy_encounters_dir == package_root / "muzero" / "strategy" / "encounters"
    assert paths.combat_quality_dir == package_root / "muzero" / "combat_quality"
    assert paths.route_heuristics_dir == package_root / "muzero" / "route_heuristics"
    assert paths.search_dir == package_root / "muzero" / "search"
    assert paths.diagnostics_dir == package_root / "muzero" / "diagnostics"
    assert paths.legacy_sts2_route_heuristic_file == package_root / "sts2_env" / "route_heuristic.py"
    assert all(path.is_relative_to(package_root / "muzero") for path in paths.all_policy_dirs())
    assert paths.strategy.strategy_file("potions") == package_root / "muzero" / "strategy" / "potions.py"
    assert paths.strategy.encounter_file("kaiser") == package_root / "muzero" / "strategy" / "encounters" / "kaiser.py"
    assert paths.strategy.combat_quality_file("block_waste") == package_root / "muzero" / "combat_quality" / "block_waste.py"
    assert paths.heuristic_search.route_heuristic_file("bias") == package_root / "muzero" / "route_heuristics" / "bias.py"
    assert paths.heuristic_search.search_file("root_prior.py") == package_root / "muzero" / "search" / "root_prior.py"


def test_strategy_and_heuristic_paths_reject_escape_paths(tmp_path: Path) -> None:
    package_root = tmp_path / "repo" / "packages" / "rl-agent"
    strategy = StrategyModulePaths.from_package_root(package_root)
    heuristic = HeuristicSearchModulePaths.from_package_root(package_root)

    with pytest.raises(ValueError):
        strategy.combat_quality_file("../train")
    with pytest.raises(ValueError):
        heuristic.search_file("/tmp/root_prior.py")
    with pytest.raises(ValueError):
        heuristic.route_heuristic_file("score.txt")


def test_file_budget_guard_allows_only_documented_legacy_debt(tmp_path: Path) -> None:
    package_root = tmp_path / "repo" / "packages" / "rl-agent"
    legacy_file = package_root / "muzero" / "sts2_env" / "muzero_model.py"
    new_file = package_root / "muzero" / "combat_quality" / "giant_policy.py"
    ok_file = package_root / "muzero" / "combat_quality" / "small_policy.py"
    legacy_file.parent.mkdir(parents=True)
    new_file.parent.mkdir(parents=True)
    legacy_file.write_text("x = 1\n" * 2_001, encoding="utf-8")
    new_file.write_text("x = 1\n" * 2_001, encoding="utf-8")
    ok_file.write_text("x = 1\n" * 25, encoding="utf-8")

    violations = find_file_budget_violations(package_root, roots=("muzero",))

    assert [record.relative_path.as_posix() for record in violations] == [
        "muzero/combat_quality/giant_policy.py"
    ]

    all_over_budget = find_file_budget_violations(package_root, roots=("muzero",), include_legacy=True)
    assert {record.relative_path.as_posix() for record in all_over_budget} == {
        "muzero/sts2_env/muzero_model.py",
        "muzero/combat_quality/giant_policy.py",
    }


def test_current_package_has_no_unbudgeted_large_python_files() -> None:
    package_root = Path(__file__).resolve().parents[1]
    violations = find_file_budget_violations(package_root)

    assert violations == []
    legacy_records = [
        record
        for record in collect_file_budget_records(package_root)
        if record.over_budget and record.legacy_allowed
    ]
    assert legacy_records
    assert all(record.relative_path.as_posix() in DEFAULT_LEGACY_ALLOWLIST for record in legacy_records)


def test_muzero_train_remains_thin_compatibility_entrypoint() -> None:
    package_root = Path(__file__).resolve().parents[1]
    train_py = package_root / "muzero" / "train.py"
    source = train_py.read_text(encoding="utf-8")

    assert len(source.splitlines()) <= 120
    assert "from muzero.training.trainer import *" in source
    assert "from muzero.training.cli_main import main as _cli_main" in source

    forbidden_training_keywords = (
        "class MuZeroTrainer",
        "def train_step",
        "def self_play",
        "route_heuristic_bias",
        "card_block_waste",
        "potion_guard",
        "kaiser",
        "insatiable",
    )
    assert not any(keyword in source for keyword in forbidden_training_keywords)
