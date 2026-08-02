from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sts2_rl.training import load_training_config

PACKAGE_ROOT = Path(__file__).parents[1]
CHECKOUT_ROOT = PACKAGE_ROOT.parents[1]
SOURCE_CONFIG = PACKAGE_ROOT / "config/experiments/full_run_revival_v29_failure_credit_v4_model_init.toml"
CONTINUATION_CONFIG = (
    PACKAGE_ROOT / "config/experiments/full_run_revival_v29_failure_credit_v4_exact_continuation_350k.toml"
)
CONTRACT_PATH = CHECKOUT_ROOT / "contracts/exact-resume-checkpoints/v29-failure-credit-v4-100k.json"


def _load_contract() -> dict[str, Any]:
    payload = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_v29_350k_recipe_is_an_exact_lineage_continuation() -> None:
    source = load_training_config(profile="preheat", config_path=SOURCE_CONFIG)
    continuation = load_training_config(profile="preheat", config_path=CONTINUATION_CONFIG)

    # This is the production exact-resume identity check: only execution
    # horizon, output paths, checkpoint cadence, and evaluation schedules may
    # differ from the 100k source lineage.
    assert continuation.lineage_mapping() == source.lineage_mapping()

    assert source.runtime.total_environment_steps == 100_000
    assert continuation.runtime.total_environment_steps == 350_000
    assert continuation.runtime.total_environment_steps - source.runtime.total_environment_steps == 250_000
    assert continuation.runtime.seed == source.runtime.seed == 4_000_000
    assert continuation.runtime.checkpoint_interval_steps == source.runtime.checkpoint_interval_steps == 10_000


def test_v29_350k_recipe_preserves_completed_evaluation_state() -> None:
    source = load_training_config(profile="preheat", config_path=SOURCE_CONFIG)
    continuation = load_training_config(profile="preheat", config_path=CONTINUATION_CONFIG)

    assert set(source.runtime.evaluation_steps).issubset(continuation.runtime.evaluation_steps)
    assert set(source.runtime.early_evaluation_steps).issubset(continuation.runtime.early_evaluation_steps)
    assert continuation.runtime.evaluation_steps == (
        0,
        25_000,
        50_000,
        75_000,
        125_000,
        150_000,
        175_000,
        200_000,
        250_000,
        300_000,
        325_000,
    )
    assert continuation.runtime.early_evaluation_steps == (5_000, 10_000, 110_000)
    assert continuation.runtime.final_audit_steps == (350_000,)
    assert 100_000 not in continuation.runtime.final_audit_steps


def test_v29_100k_exact_resume_contract_pins_source_and_absolute_horizon() -> None:
    contract = _load_contract()

    assert contract == {
        "checkpoint_id": "075b8174-1dab-4293-90bd-9c63f0aab277",
        "continuation_config": (
            "packages/rl-agent/config/experiments/"
            "full_run_revival_v29_failure_credit_v4_exact_continuation_350k.toml"
        ),
        "environment_steps": 100_000,
        "exact_resume_permitted": True,
        "expected_additional_environment_steps": 250_000,
        "expected_load_mode": "exact_resume",
        "expected_total_environment_steps": 350_000,
        "manifest_filename": "checkpoint.manifest.json",
        "manifest_sha256": "d7c80dcfbcd222d8ed93f6b7dd8bc4735b5637ac91516fae1b707a55b53a7565",
        "metadata_filename": "metadata.json",
        "metadata_sha256": "a8a8396bad5799b1cd96591751c7d8c1c7f3732ad702aeafdd25468cf934c29c",
        "name": "v29-failure-credit-v4-100k",
        "policy_version": 1572,
        "relative_path": (
            "checkpoints/full-run-revival-v29-failure-credit-v4-model-init/"
            "run-78554f34-a31e-48ed-8e3e-8e58effc2790/periodic-step-000100000"
        ),
        "schema_version": "sts2-exact-resume-checkpoint-contract-v1",
        "source_git_commit": "0c6cafb6aa313d85df2b9057e8160ea4a4e508a7",
        "source_run_id": "78554f34-a31e-48ed-8e3e-8e58effc2790",
        "training_state": {
            "actor_policy_version": 1572,
            "consumed_unrolls": 6288,
            "environment_steps": 100_000,
            "episodes": 110,
            "evaluation_episodes": 80,
            "learner_updates": 1572,
            "maximum_observed_candidates": 269,
            "policy_version": 1572,
        },
        "usage": "exact_resume_only",
    }

    continuation = load_training_config(profile="preheat", config_path=CHECKOUT_ROOT / contract["continuation_config"])
    assert (
        continuation.runtime.total_environment_steps - contract["environment_steps"]
        == contract["expected_additional_environment_steps"]
    )
    assert continuation.runtime.total_environment_steps == contract["expected_total_environment_steps"]
