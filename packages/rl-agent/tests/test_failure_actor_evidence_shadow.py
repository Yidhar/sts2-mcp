from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import NoReturn

import pytest
import torch

from sts2_rl.models import RecurrentCandidateModel
from sts2_rl.training.failure_credit import BoundedFailureCreditReplay
from sts2_rl.training.learner import VTraceLearner


def _shadow_module() -> ModuleType:
    path = Path(__file__).parents[1] / "scripts" / "validate_failure_actor_evidence_shadow.py"
    spec = importlib.util.spec_from_file_location(
        "validate_failure_actor_evidence_shadow",
        path,
    )
    if spec is None or spec.loader is None:  # pragma: no cover
        raise RuntimeError("could not load actor-evidence shadow command")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _forbidden_construction(*_args: object, **_kwargs: object) -> NoReturn:
    raise AssertionError("actor-evidence mask shadow constructed training state")


def _counts(
    dry_run: dict[str, object],
    case: str,
    phase: str,
) -> dict[str, int]:
    cases = dry_run["case_manifests"]
    assert isinstance(cases, dict)
    phases = cases[case]
    assert isinstance(phases, dict)
    summary = phases[phase]
    assert isinstance(summary, dict)
    counts = summary["mask_counts"]
    assert isinstance(counts, dict)
    return counts


def test_shadow_uses_production_manifest_without_model_optimizer_or_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _shadow_module()
    monkeypatch.setattr(
        RecurrentCandidateModel,
        "__init__",
        _forbidden_construction,
    )
    monkeypatch.setattr(
        VTraceLearner,
        "__init__",
        _forbidden_construction,
    )
    monkeypatch.setattr(
        BoundedFailureCreditReplay,
        "__init__",
        _forbidden_construction,
    )
    monkeypatch.setattr(
        torch.optim.Optimizer,
        "__init__",
        _forbidden_construction,
    )

    report = module.build_actor_evidence_shadow_report()

    assert report["status"] == "passed"
    assert report["gates"]["learner_mask_dry_run_passed"] is True
    dry_run = report["learner_mask_dry_run"]
    assert dry_run["interface"] == ("sts2_rl.training.learner.compile_liveness_label_manifest")
    assert dry_run["pure_dry_run"] is True
    assert dry_run["model_constructed"] is False
    assert dry_run["optimizer_constructed"] is False
    assert dry_run["replay_constructed"] is False
    assert all(dry_run["checks"].values())


def test_shadow_manifest_masks_cover_calibration_and_mature_contract() -> None:
    module = _shadow_module()
    first = module.build_actor_evidence_shadow_report()
    second = module.build_actor_evidence_shadow_report()
    dry_run = first["learner_mask_dry_run"]
    calibration = "calibration_update_0"
    mature = "mature_risk_start"

    assert first["version"] == ("sts2-failure-actor-evidence-shadow-report-v3")
    assert json.dumps(
        first["learner_mask_dry_run"],
        allow_nan=False,
        sort_keys=True,
    ) == json.dumps(
        second["learner_mask_dry_run"],
        allow_nan=False,
        sort_keys=True,
    )

    direct_calibration = _counts(
        dry_run,
        "linger9_death_warning_direct",
        calibration,
    )
    direct_mature = _counts(
        dry_run,
        "linger9_death_warning_direct",
        mature,
    )
    assert direct_calibration["effective_direct_rows"] == 1
    assert direct_mature["effective_direct_rows"] == 1

    multi_calibration = _counts(
        dry_run,
        "room_full_of_cheese_multi_edge",
        calibration,
    )
    multi_mature = _counts(
        dry_run,
        "room_full_of_cheese_multi_edge",
        mature,
    )
    assert multi_calibration["effective_cycle_groups"] == 1
    assert multi_mature["effective_cycle_groups"] == 1

    unique_calibration = _counts(
        dry_run,
        "unique_unresolved_stall",
        calibration,
    )
    unique_mature = _counts(
        dry_run,
        "unique_unresolved_stall",
        mature,
    )
    assert unique_calibration["risk_requested_rows"] == 6
    assert unique_calibration["effective_risk_rows"] == 0
    assert unique_mature["effective_risk_rows"] == 6

    abandoned_calibration = _counts(
        dry_run,
        "abandoned_cycle_suffix",
        calibration,
    )
    abandoned_mature = _counts(
        dry_run,
        "abandoned_cycle_suffix",
        mature,
    )
    for counts in (
        abandoned_calibration,
        abandoned_mature,
    ):
        assert counts["effective_direct_rows"] == 0
        assert counts["effective_cycle_groups"] == 0
    assert abandoned_mature["effective_risk_rows"] == 10

    for case in (
        "forced_only_stall",
        "censored_boundary",
    ):
        for phase in (
            calibration,
            mature,
        ):
            counts = _counts(dry_run, case, phase)
            assert counts["effective_direct_rows"] == 0
            assert counts["effective_risk_rows"] == 0
            assert counts["effective_cycle_groups"] == 0
            assert counts["effective_contrast_groups"] == 0
