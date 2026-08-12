"""Parse archived pre-v20 experiment recipes for launcher regression tests.

The historical ``full_run_revival_v33``..``v47`` experiment TOMLs are frozen
archives.  They still spell out the operation-specific exploration floors,
completion guidance, macro entropy collapse breaker, transaction
completion/selection-group CE weights and the two-sided entry support
corridor that config v20 retired, so the strict current parser rightly
rejects them.  These tests only assert launch-recipe facts that survived the
retirement, so they parse the archive with the retired tables/keys stripped
-- exactly the projection the reviewed v19 -> v20 model-initialization
migration applies to checkpoint payloads.
"""

from __future__ import annotations

from pathlib import Path

from sts2_rl.training.config import (
    PROFILE_DIR,
    TrainingConfig,
    _deep_merge,
    _load_toml,
    training_config_from_mapping,
)


def load_archived_training_config(
    *,
    profile: str,
    config_path: Path | str,
) -> TrainingConfig:
    payload = _load_toml(PROFILE_DIR / f"{profile}.toml")
    _deep_merge(payload, _load_toml(Path(config_path)))
    payload["profile"] = profile
    payload.pop("transaction_exploration", None)
    curriculum = payload.get("curriculum")
    if isinstance(curriculum, dict):
        curriculum.pop("selection_surface_epsilon_floor", None)
    optimization = payload.get("optimization")
    if isinstance(optimization, dict):
        optimization.pop("entropy_breaker", None)
    transaction_learning = payload.get("transaction_learning")
    if isinstance(transaction_learning, dict):
        for retired_key in (
            "completion_policy_weight",
            "lifecycle_entry_support_weight",
            "lifecycle_entry_support_probability_floor",
            "macro_option_group_completion_weight",
        ):
            transaction_learning.pop(retired_key, None)
    return training_config_from_mapping(payload)
