from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from sts2_rl.training import checkpoint_evaluation as checkpoint_evaluation_module
from sts2_rl.training.config import CONFIG_VERSION, TrainingConfig


def _v10_payload() -> dict[str, Any]:
    payload = TrainingConfig().to_mapping()
    payload["version"] = "sts2-relational-curriculum-config-v10"
    # V10 predates the failure-credit table.  Reconstruct the historical
    # payload instead of only relabeling the current mapping.
    del payload["failure_credit"]
    transaction_learning = payload["transaction_learning"]
    assert isinstance(transaction_learning, dict)
    for key in (
        "lifecycle_entry_support_weight",
        "lifecycle_entry_support_probability_floor",
        "lifecycle_smdp_q_weight",
        "lifecycle_smdp_horizon",
        "lifecycle_advantage_policy_weight",
        "lifecycle_advantage_start_update",
        "lifecycle_advantage_temperature",
        "lifecycle_advantage_clip",
        "lifecycle_advantage_q_error_gate",
        "lifecycle_advantage_max_policy_lag",
        "lifecycle_advantage_max_log_probability_shift",
    ):
        del transaction_learning[key]
    episodic = payload["episodic_learning"]
    assert isinstance(episodic, dict)
    del episodic["fresh_policy_sequences"]
    del episodic["success_imitation_exempt_surfaces"]
    for key in (
        "act_segment_imitation_enabled",
        "act_segment_policy_weight",
        "act_segment_min_exit_hp_ratio",
        "act_segment_max_revival_fraction",
        "combat_hp_loss_value_weight",
        "combat_hp_loss_reference",
    ):
        del episodic[key]
    runtime = payload["runtime"]
    assert isinstance(runtime, dict)
    del runtime["evaluation_guard_enforcement_start_steps"]
    return payload


def test_checkpoint_evaluation_interprets_v10_as_disabled_model_only_view(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _v10_payload()
    monkeypatch.setattr(
        checkpoint_evaluation_module,
        "validate_resume_checkpoint",
        lambda _checkpoint: SimpleNamespace(
            metadata={"training_config": payload},
        ),
    )

    config = checkpoint_evaluation_module.checkpoint_training_config("source")

    assert config.version == CONFIG_VERSION
    assert config.episodic_learning.fresh_policy_sequences == 0


def test_checkpoint_evaluation_rejects_impossible_v10_fresh_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _v10_payload()
    episodic = payload["episodic_learning"]
    assert isinstance(episodic, dict)
    episodic["fresh_policy_sequences"] = 1
    monkeypatch.setattr(
        checkpoint_evaluation_module,
        "validate_resume_checkpoint",
        lambda _checkpoint: SimpleNamespace(
            metadata={"training_config": payload},
        ),
    )

    with pytest.raises(ValueError, match="unexpectedly contains"):
        checkpoint_evaluation_module.checkpoint_training_config("source")
