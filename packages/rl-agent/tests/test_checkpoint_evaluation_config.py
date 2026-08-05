from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from sts2_rl.training import checkpoint_evaluation as checkpoint_evaluation_module
from sts2_rl.training.config import CONFIG_VERSION, TrainingConfig


def _v10_payload() -> dict[str, Any]:
    payload = TrainingConfig().to_mapping()
    payload["version"] = "sts2-relational-curriculum-config-v10"
    # V10 predates the failure-credit and transaction-exploration tables.
    # Reconstruct the historical payload instead of only relabeling the
    # current V14 mapping.
    del payload["failure_credit"]
    del payload["transaction_exploration"]
    episodic = payload["episodic_learning"]
    assert isinstance(episodic, dict)
    del episodic["fresh_policy_sequences"]
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
