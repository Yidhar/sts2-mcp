from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import torch

from sts2_rl.training.config import (
    FailureCreditConfig,
    ModelConfig,
    TrainingConfig,
)


def _validator_module() -> ModuleType:
    path = Path(__file__).parents[1] / "scripts" / "validate_liveness_head_stress.py"
    spec = importlib.util.spec_from_file_location(
        "validate_liveness_head_stress",
        path,
    )
    if spec is None or spec.loader is None:  # pragma: no cover
        raise RuntimeError("could not load liveness stress validator")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _cpu_config() -> TrainingConfig:
    base = TrainingConfig()
    return replace(
        base,
        profile="liveness-stress-unit",
        model=ModelConfig(
            token_feature_dim=224,
            d_model=16,
            n_heads=4,
            ffn_dim=32,
            world_layers=1,
            latent_slots=2,
            latent_layers=1,
            local_layers=1,
            candidate_layers=1,
            recurrent_hidden_dim=32,
            dropout=0.0,
            type_vocab_size=128,
            role_vocab_size=64,
            owner_vocab_size=128,
            entity_vocab_size=8192,
            zone_vocab_size=32,
            order_vocab_size=128,
            domain_count=8,
            max_world_tokens=32,
            max_candidates=4,
            max_candidate_local_tokens=4,
        ),
        transaction_learning=replace(
            base.transaction_learning,
            enabled=False,
        ),
        failure_credit=FailureCreditConfig(
            mode="learning",
            burn_in_steps=0,
            maximum_context_steps=4,
            liveness_head_calibration_updates=1,
            liveness_risk_actor_start_update=2,
            liveness_tbptt_window_steps=2,
        ),
    )


def test_cpu_stress_executes_production_manifest_forward_and_backward(
    tmp_path: Path,
) -> None:
    module = _validator_module()
    report = module.run_liveness_head_stress(
        training_config=_cpu_config(),
        device=torch.device("cpu"),
        shape=module.StressShape(
            steps=4,
            candidates=4,
            ordinary_candidates=3,
        ),
        initialization="synthetic",
        artifact_root=None,
        config_source="unit-test-synthetic-config",
        require_formal_shape=False,
    )

    assert report["status"] == "passed"
    assert report["version"] == "sts2-liveness-head-active-shape-stress-v3"
    assert report["training_authority"] is False
    assert report["optimizer_steps"] == 0
    assert all(report["gates"].values())
    assert report["shape"] == {
        "contexts": 2,
        "steps_per_context": 4,
        "replayed_steps": 8,
        "legal_candidates_per_step": 4,
        "candidate_decisions_per_context": 16,
        "replayed_candidate_decisions": 32,
        "ordinary_probe_candidates": 3,
        "tbptt_segments_per_context": 2,
        "expected_tbptt_segments": 4,
    }

    calibration = report["phases"]["calibration"]
    mature = report["phases"]["mature"]
    for phase in (calibration, mature):
        assert phase["work"] == {
            "contexts": 2,
            "steps": 8,
            "candidates": 32,
            "autograd_segments": 4,
        }
        assert phase["forward_shapes"]["forward_calls"] == 4
        assert phase["forward_shapes"]["maximum_batch_rows"] == 2
        assert phase["forward_shapes"]["minimum_batch_rows"] == 2
        assert phase["forward_shapes"]["maximum_candidate_axis"] == 4
        assert phase["gradients"]["finite"] is True
        assert phase["gradients"]["nonzero_gradient_tensors"] > 0
        assert phase["gradients"]["nonzero_liveness_head_tensors"] > 0

    assert calibration["calibration_active"] is True
    assert calibration["risk_actor_enabled"] is False
    assert calibration["labels"]["risk_actor"] == 0
    assert calibration["labels"]["risk_actor_phase_suppressed"] == 4
    assert calibration["labels"]["direct_avoid"] == 2
    assert calibration["labels"]["contrast"] == 1
    assert mature["calibration_active"] is False
    assert mature["risk_actor_enabled"] is True
    assert mature["labels"]["risk_actor"] == 4
    assert mature["labels"]["direct_avoid"] == 2
    assert mature["labels"]["contrast"] == 1

    probe = report["active_shape_probe"]
    assert probe["work"]["steps"] == 2
    assert probe["work"]["candidates"] == 6
    assert probe["forward_shapes"]["forward_calls"] == 2
    assert probe["forward_shapes"]["maximum_candidate_axis"] == 3

    output = tmp_path / "stress.json"
    module._atomic_json(output, report)
    restored = json.loads(output.read_text(encoding="utf-8"))
    assert restored["status"] == "passed"
    digest = module._sha256_file(output)
    assert len(digest) == 64
    assert all(character in "0123456789abcdef" for character in digest)


def test_synthetic_evidence_is_one_atomic_reviewed_record() -> None:
    module = _validator_module()
    config = _cpu_config()
    record = module.build_synthetic_evidence_record(
        training_config=config,
        steps=4,
        candidates=4,
        policy_version=7,
        include_direct_witness=True,
        include_matched_outcome=True,
        identity_suffix="unit-atomic",
    )

    assert len(record.plan.context.steps) == 4
    assert all(step.snapshot.candidate_count == 4 for step in record.plan.context.steps)
    assert all(int(step.snapshot.action_mask.sum()) == 4 for step in record.plan.context.steps)
    assert len(record.plan.liveness_value_targets) == 4
    assert len(record.plan.liveness_q_targets) == 4
    assert record.plan.risk_sequences[0].step_indices == (0, 1, 2, 3)
    assert tuple(target.step_index for target in record.plan.direct_policy_targets) == (2, 3)
    assert len(record.plan.contrast_policy_targets) == 1
    pair = record.plan.contrast_policy_targets[0].pair
    assert len(pair.better.context.steps) == 4
    assert len(pair.worse.context.steps) == 4
    assert all(step.snapshot.candidate_count == 4 for step in pair.better.context.steps)
    assert all(step.snapshot.candidate_count == 4 for step in pair.worse.context.steps)
    assert pair.better.step.selected_action.comparison != pair.worse.step.selected_action.comparison
