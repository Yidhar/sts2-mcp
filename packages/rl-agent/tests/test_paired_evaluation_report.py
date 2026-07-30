from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from sts2_rl.paired_evaluation_report import analyze_paired_evaluation, main, render_markdown


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _episode(
    seed: int,
    *,
    won: bool,
    act1: bool | None = None,
    max_act: int | None = None,
    max_floor: int | None = None,
    revivals: int = 0,
    hp_loss: float = 0.0,
    cycle: bool = False,
    combat_stall: bool = False,
    noncombat_stall: bool = False,
    trusted_failure: bool = False,
) -> dict[str, Any]:
    resolved_act = 3 if won else (2 if max_act is None else max_act)
    return {
        "reset_seed": seed,
        "run_won": won,
        "act1_cleared": resolved_act >= 2 if act1 is None else act1,
        "max_act": resolved_act,
        "max_floor": (46 if won else 25) if max_floor is None else max_floor,
        "combat_progress_stalled": combat_stall,
        "noncombat_progress_stalled": noncombat_stall,
        "trusted_policy_failure": trusted_failure,
        "noncombat_event_cycle": False,
        "selection_action_cycle": cycle,
        "deadlocked": cycle or combat_stall or noncombat_stall,
        "revivals_used": revivals,
        "player_hp_lost": hp_loss,
    }


def _write_pair(
    root: Path,
    seeds: list[int],
    models: list[tuple[str, int, int, list[dict[str, Any]]]],
) -> Path:
    root.mkdir()
    results: list[dict[str, Any]] = []
    for checkpoint_id, steps, policy_version, episodes in models:
        child = root / f"checkpoint-{checkpoint_id}"
        child.mkdir()
        audit = {
            "schema_version": "sts2-frozen-checkpoint-evaluation-v2",
            "evaluation_of": {
                "checkpoint": f"/checkpoints/step-{steps}",
                "checkpoint_id": checkpoint_id,
            },
            "evaluation": {
                "episodes": len(episodes),
                "episode_metrics": episodes,
            },
        }
        audit_path = child / "evaluation.json"
        audit_path.write_text(json.dumps(audit, sort_keys=True), encoding="utf-8")
        results.append(
            {
                "checkpoint": f"/checkpoints/step-{steps}",
                "checkpoint_id": checkpoint_id,
                "source_environment_steps": steps,
                "source_policy_version": policy_version,
                "audit_sha256": _sha256(audit_path),
            }
        )
    manifest = {
        "schema_version": "sts2-paired-frozen-evaluation-v2",
        "status": "complete",
        "episodes_per_checkpoint": len(seeds),
        "held_out_seeds": seeds,
        "lineage": {"sha256": "lineage"},
        "simulator_provenance": {"verified_identity": {"sha256": "sim"}},
        "results": results,
    }
    (root / "paired-evaluation.manifest.json").write_text(
        json.dumps(manifest, sort_keys=True),
        encoding="utf-8",
    )
    return root


def _checkpoint_by_id(report: dict[str, object], checkpoint_id: str) -> dict[str, object]:
    checkpoints = report["checkpoints"]
    assert isinstance(checkpoints, list)
    return next(item for item in checkpoints if isinstance(item, dict) and item["checkpoint_id"] == checkpoint_id)


def test_three_model_report_aligns_seeds_and_emits_all_pairwise_statistics(tmp_path: Path) -> None:
    seeds = [101, 103, 105, 107]
    paired = _write_pair(
        tmp_path / "paired",
        seeds,
        [
            (
                "model-a",
                72_075,
                1_129,
                [
                    _episode(101, won=True, revivals=10, hp_loss=100),
                    _episode(103, won=True, revivals=12, hp_loss=120),
                    _episode(105, won=False, max_act=3, max_floor=40),
                    _episode(107, won=False, max_act=1, max_floor=10, combat_stall=True),
                ],
            ),
            (
                "model-b",
                81_321,
                1_274,
                [
                    _episode(101, won=True, revivals=8, hp_loss=80),
                    _episode(103, won=True, revivals=9, hp_loss=90),
                    _episode(105, won=True, revivals=11, hp_loss=110),
                    _episode(107, won=False, max_act=3, max_floor=42),
                ],
            ),
            (
                "model-c",
                250_000,
                3_934,
                [
                    _episode(101, won=True, revivals=2, hp_loss=20),
                    _episode(103, won=False, max_act=3, max_floor=35),
                    _episode(105, won=False, max_act=2, max_floor=25),
                    _episode(107, won=False, max_act=1, max_floor=8, cycle=True),
                ],
            ),
        ],
    )

    report = analyze_paired_evaluation(
        paired,
        bootstrap_samples=200,
        bootstrap_seed=19,
    )

    assert report["schema_version"] == "sts2-paired-evaluation-analysis-v1"
    ranking = report["ranking"]
    assert isinstance(ranking, list)
    assert [item["checkpoint_id"] for item in ranking] == ["model-b", "model-a", "model-c"]
    comparisons = report["pairwise_comparisons"]
    assert isinstance(comparisons, list)
    assert len(comparisons) == 3
    a_vs_b = next(
        item
        for item in comparisons
        if item["left_checkpoint_id"] == "model-a" and item["right_checkpoint_id"] == "model-b"
    )
    run_wins = a_vs_b["binary_metrics"]["run_win"]
    assert run_wins["paired_contingency"] == {
        "left_false_right_false": 1,
        "left_false_right_true": 1,
        "left_true_right_false": 0,
        "left_true_right_true": 2,
    }
    assert run_wins["exact_mcnemar"]["p_value"] == 1.0
    assert run_wins["right_minus_left_rate"] == pytest.approx(0.25)
    assert a_vs_b["successful_run_efficiency"]["common_success_seed_count"] == 2
    markdown = render_markdown(report)
    assert "Pairwise full-run flips" in markdown
    assert "failed runs" in markdown


def test_seed_set_mismatch_fails_closed(tmp_path: Path) -> None:
    seeds = [201, 203, 205]
    paired = _write_pair(
        tmp_path / "paired",
        seeds,
        [
            ("model-a", 10, 1, [_episode(seed, won=False) for seed in seeds]),
            (
                "model-b",
                20,
                2,
                [
                    _episode(201, won=False),
                    _episode(203, won=False),
                    _episode(207, won=False),
                ],
            ),
        ],
    )

    with pytest.raises(ValueError, match="seed sequence differs"):
        analyze_paired_evaluation(paired, bootstrap_samples=10)


def test_failed_run_low_cost_never_outweighs_more_complete_runs(tmp_path: Path) -> None:
    seeds = [301, 303, 305, 307]
    paired = _write_pair(
        tmp_path / "paired",
        seeds,
        [
            (
                "more-wins-high-cost",
                100,
                10,
                [
                    _episode(301, won=True, revivals=100, hp_loss=1_000),
                    _episode(303, won=True, revivals=120, hp_loss=1_200),
                    _episode(305, won=False, max_act=3, revivals=0, hp_loss=0),
                    _episode(307, won=False, max_act=2, revivals=0, hp_loss=0),
                ],
            ),
            (
                "fewer-wins-low-cost",
                200,
                20,
                [
                    _episode(301, won=True, revivals=1, hp_loss=5),
                    _episode(303, won=False, max_act=3, revivals=0, hp_loss=0),
                    _episode(305, won=False, max_act=3, revivals=0, hp_loss=0),
                    _episode(307, won=False, max_act=2, revivals=0, hp_loss=0),
                ],
            ),
        ],
    )

    report = analyze_paired_evaluation(paired, bootstrap_samples=100)

    ranking = report["ranking"]
    assert isinstance(ranking, list)
    assert ranking[0]["checkpoint_id"] == "more-wins-high-cost"
    low_cost = _checkpoint_by_id(report, "fewer-wins-low-cost")
    efficiency = low_cost["successful_run_efficiency"]
    assert efficiency["successful_run_count"] == 1
    assert efficiency["revivals_used"]["mean"] == 1.0
    comparison = report["pairwise_comparisons"][0]
    assert comparison["successful_run_efficiency"]["common_success_seeds"] == [301]


def test_high_win_checkpoint_with_cycle_cannot_beat_eligible_checkpoint(tmp_path: Path) -> None:
    seeds = [321, 323, 325, 327]
    paired = _write_pair(
        tmp_path / "paired",
        seeds,
        [
            (
                "high-win-with-cycle",
                100,
                10,
                [
                    _episode(321, won=True, revivals=1),
                    _episode(323, won=True, revivals=1),
                    _episode(325, won=True, revivals=1),
                    _episode(327, won=True, revivals=1, cycle=True),
                ],
            ),
            (
                "lower-win-cycle-free",
                200,
                20,
                [
                    _episode(321, won=True, revivals=20),
                    _episode(323, won=True, revivals=20),
                    _episode(325, won=False, max_act=3),
                    _episode(327, won=False, max_act=2),
                ],
            ),
        ],
    )

    report = analyze_paired_evaluation(paired, bootstrap_samples=100)

    ranking = report["ranking"]
    assert isinstance(ranking, list)
    assert [item["checkpoint_id"] for item in ranking] == [
        "lower-win-cycle-free",
        "high-win-with-cycle",
    ]
    assert ranking[0]["eligible_for_automatic_selection"] is True
    assert ranking[1]["eligible_for_automatic_selection"] is False
    selection = report["selection"]
    assert selection["automatic_selection_allowed"] is True
    assert selection["selected_checkpoint_id"] == "lower-win-cycle-free"


def test_all_cyclic_checkpoints_block_automatic_selection(tmp_path: Path) -> None:
    seeds = [341, 343]
    paired = _write_pair(
        tmp_path / "paired",
        seeds,
        [
            (
                "cyclic-a",
                100,
                10,
                [
                    _episode(341, won=True, cycle=True),
                    _episode(343, won=False, max_act=3),
                ],
            ),
            (
                "cyclic-b",
                200,
                20,
                [
                    _episode(341, won=True),
                    _episode(343, won=True, cycle=True),
                ],
            ),
        ],
    )

    report = analyze_paired_evaluation(paired, bootstrap_samples=100)

    selection = report["selection"]
    assert selection == {
        "automatic_selection_allowed": False,
        "selected_checkpoint_id": None,
        "eligible_checkpoint_ids": [],
        "ineligible_checkpoint_ids": ["cyclic-b", "cyclic-a"],
        "blocked_reason": "all checkpoints have at least one hard cycle",
    }
    assert "Automatic selection: **BLOCKED**" in render_markdown(report)


def test_exact_mcnemar_and_cli_publish_are_deterministic_and_read_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seeds = list(range(401, 417, 2))
    paired = _write_pair(
        tmp_path / "paired",
        seeds,
        [
            ("all-wins", 10, 1, [_episode(seed, won=True) for seed in seeds]),
            ("all-losses", 20, 2, [_episode(seed, won=False) for seed in seeds]),
        ],
    )
    manifest_before = (paired / "paired-evaluation.manifest.json").read_bytes()
    audit_before = (paired / "checkpoint-all-wins" / "evaluation.json").read_bytes()
    monkeypatch.setenv("STS2_ARTIFACT_ROOT", str(tmp_path / "artifacts"))

    assert (
        main(
            [
                "--paired-output",
                str(paired),
                "--output-root",
                "reports/exact-test",
                "--bootstrap-samples",
                "100",
                "--bootstrap-seed",
                "7",
            ]
        )
        == 0
    )

    output = tmp_path / "artifacts" / "reports" / "exact-test"
    machine = json.loads((output / "paired-evaluation.analysis.json").read_text(encoding="utf-8"))
    wins = machine["pairwise_comparisons"][0]["binary_metrics"]["run_win"]
    assert wins["exact_mcnemar"]["p_value"] == pytest.approx(0.0078125)
    assert wins["paired_bootstrap_ci"]["lower"] == -1.0
    assert wins["paired_bootstrap_ci"]["upper"] == -1.0
    assert (output / "paired-evaluation.analysis.md").is_file()
    assert (paired / "paired-evaluation.manifest.json").read_bytes() == manifest_before
    assert (paired / "checkpoint-all-wins" / "evaluation.json").read_bytes() == audit_before
