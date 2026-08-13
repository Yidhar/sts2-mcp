"""Parse archived pre-v20 experiment recipes for launcher regression tests.

The historical ``full_run_revival_v33``..``v47`` experiment TOMLs are frozen
archives.  They still spell out the operation-specific exploration floors,
completion guidance, macro entropy collapse breaker, transaction
completion/selection-group CE weights, the two-sided entry support corridor,
the run-success/act-segment/revival imitation channels and the liveness
policy-actor family that config v20 retired, so the strict current parser
rightly rejects them.  These tests only assert launch-recipe facts that
survived the retirement, so they parse the archive with the retired
tables/keys stripped -- exactly the projection the reviewed v19 -> v20
model-initialization migration applies to checkpoint payloads.
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
    failure_credit = payload.get("failure_credit")
    if isinstance(failure_credit, dict):
        for retired_key in (
            "policy_gradient_max_lag",
            "matched_outcome_pair_quota",
            "liveness_risk_actor_start_update",
            "liveness_cost_actor_weight",
            "liveness_direct_policy_weight",
            "liveness_cycle_policy_weight",
            "liveness_contrast_policy_weight",
            "liveness_completion_policy_weight",
            "liveness_risk_advantage_clip",
            "liveness_risk_actor_min_selected_probability",
            "liveness_contrast_margin",
        ):
            failure_credit.pop(retired_key, None)
    episodic_learning = payload.get("episodic_learning")
    if isinstance(episodic_learning, dict):
        for retired_key in (
            "fresh_policy_sequences",
            "primary_policy_weight",
            "revival_policy_weight",
            "secondary_advantage_fraction",
            "primary_success_tie_tolerance",
            "importance_ratio_clip",
            "success_policy_trust_region_epsilon",
            "success_imitation_exempt_surfaces",
            "act_segment_imitation_enabled",
            "act_segment_policy_weight",
            "act_segment_min_exit_hp_ratio",
            "act_segment_max_revival_fraction",
            "policy_gradient_max_lag",
        ):
            episodic_learning.pop(retired_key, None)
    return training_config_from_mapping(payload)
