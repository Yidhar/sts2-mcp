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
        "lifecycle_smdp_q_weight",
        "lifecycle_smdp_horizon",
    ):
        del transaction_learning[key]
    episodic = payload["episodic_learning"]
    assert isinstance(episodic, dict)
    for key in (
        "combat_hp_loss_value_weight",
        "combat_hp_loss_reference",
    ):
        del episodic[key]
    # A real v10 payload spelled out the retired imitation weights that
    # config v20 deleted; the reviewed migration must strip them.
    episodic["primary_policy_weight"] = 0.25
    episodic["revival_policy_weight"] = 0.05
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
    config_episodic = config.to_mapping()["episodic_learning"]
    assert isinstance(config_episodic, dict)
    assert "primary_policy_weight" not in config_episodic
    assert "revival_policy_weight" not in config_episodic
    assert "fresh_policy_sequences" not in config_episodic


def test_checkpoint_evaluation_rejects_retired_field_in_current_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = TrainingConfig().to_mapping()
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

    with pytest.raises(ValueError, match="unknown episodic_learning config keys"):
        checkpoint_evaluation_module.checkpoint_training_config("source")
