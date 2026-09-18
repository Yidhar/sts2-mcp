"""Tests for the Combat Resource Shadow Price oracle and macro policy projection."""

from __future__ import annotations

import math
from typing import Any

import pytest
import torch

from sts2_rl.encoding import GroundedEncodingConfig, GroundedObservationEncoder
from sts2_rl.joint.macro_credit_learner import MacroCreditLearnerConfig
from sts2_rl.joint.macro_targets import project_macro_policy_from_shadow_price
from sts2_rl.joint.shadow_price import (
    SHADOW_PRICE_VERSION,
    CombatShadowPriceOracle,
    ShadowPriceEvaluation,
    compute_hp_shadow_price,
    evaluate_combat_value_from_snapshot,
)
from sts2_rl.models.grounded_candidate import GroundedCandidateConfig
from sts2_rl.models.hybrid_actor_critic import (
    HybridActorCriticConfig,
    HybridDomainActorCriticModel,
)


def _make_model_and_encoder() -> tuple[HybridDomainActorCriticModel, GroundedObservationEncoder]:
    grounded_cfg = GroundedCandidateConfig(
        token_feature_dim=224,
        d_model=32,
        n_heads=2,
        ffn_dim=64,
        world_layers=1,
        latent_slots=4,
        latent_layers=1,
        local_layers=1,
        candidate_layers=1,
        recurrent_hidden_dim=32,
        dropout=0.0,
    )
    hybrid_cfg = HybridActorCriticConfig(
        grounded=grounded_cfg,
        domain_layers=1,
        joint_layers=1,
    )
    model = HybridDomainActorCriticModel(hybrid_cfg)
    model.eval()
    encoding_cfg = GroundedEncodingConfig.from_model_config(grounded_cfg)
    encoder = GroundedObservationEncoder(encoding_cfg)
    return model, encoder


def _campfire_observation(hp: int, max_hp: int = 80) -> dict[str, Any]:
    return {
        "phase": "rest",
        "decision_domain": "build",
        "player": {
            "hp": hp,
            "max_hp": max_hp,
            "gold": 120,
            "deck": 5,
            "deck_cards": [
                {"id": "Strike_R", "card_id": "Strike_R", "upgrades": 0, "is_upgraded": False},
                {"id": "Strike_R", "card_id": "Strike_R", "upgrades": 0, "is_upgraded": False},
                {"id": "Defend_R", "card_id": "Defend_R", "upgrades": 0, "is_upgraded": False},
                {"id": "Defend_R", "card_id": "Defend_R", "upgrades": 0, "is_upgraded": False},
                {"id": "Bash", "card_id": "Bash", "upgrades": 0, "is_upgraded": False},
            ],
            "relics": [{"id": "Burning Blood"}],
            "potions": [],
        },
        "rest_site": {
            "id": "rest_site_1",
            "options": [
                {"option_id": "rest", "heal_amount": 24},
                {"option_id": "smith", "card": {"id": "Bash", "card_id": "Bash"}},
            ],
        },
    }


def _campfire_actions() -> list[dict[str, Any]]:
    return [
        {
            "action_handle": "rest_site:rest",
            "kind": "rest_site",
            "model_action_kind": "rest_site",
            "action": "rest",
            "heal_amount": 24,
            "label": "Rest",
            "is_enabled": True,
        },
        {
            "action_handle": "rest_site:smith",
            "kind": "deck_upgrade",
            "model_action_kind": "deck_upgrade",
            "action": "smith",
            "card": {"id": "Bash", "card_id": "Bash"},
            "upgrade_preview": {"id": "Bash+", "card_id": "Bash", "upgrades": 1, "is_upgraded": True},
            "label": "Smith Bash",
            "is_enabled": True,
        },
    ]


def test_shadow_price_version_contract() -> None:
    assert SHADOW_PRICE_VERSION == "sts2-combat-resource-shadow-price-v1"


def test_compute_hp_shadow_price_physical_monotonicity() -> None:
    model, encoder = _make_model_and_encoder()
    obs_low = _campfire_observation(hp=15, max_hp=80)
    lambda_low = compute_hp_shadow_price(model, encoder, obs_low, delta_hp=5)
    assert math.isfinite(lambda_low)
    assert lambda_low >= 0.0

    obs_full = _campfire_observation(hp=80, max_hp=80)
    lambda_full = compute_hp_shadow_price(model, encoder, obs_full, delta_hp=5)
    assert math.isfinite(lambda_full)
    assert lambda_full >= 0.0


def test_evaluate_combat_value_from_snapshot() -> None:
    model, encoder = _make_model_and_encoder()
    obs = _campfire_observation(hp=40, max_hp=80)
    actions = _campfire_actions()
    snapshot = encoder.encode(obs, actions).snapshot
    val = evaluate_combat_value_from_snapshot(model, snapshot)
    assert math.isfinite(val)


def test_oracle_evaluate_macro_root() -> None:
    model, encoder = _make_model_and_encoder()
    oracle = CombatShadowPriceOracle(model, encoder, discount=0.997)
    obs = _campfire_observation(hp=20, max_hp=80)
    actions = _campfire_actions()

    eval_result = oracle.evaluate_macro_root(obs, actions)
    assert isinstance(eval_result, ShadowPriceEvaluation)
    eval_result.validate()

    assert len(eval_result.action_values) == 2
    assert len(eval_result.delta_combat_values) == 2
    assert eval_result.hp_shadow_price >= 0.0
    assert math.isfinite(eval_result.base_combat_value)


def test_project_macro_policy_from_shadow_price() -> None:
    action_values = torch.tensor([1.2, 0.8], dtype=torch.float32)
    teacher_log_probs = torch.tensor([-0.693147, -0.693147], dtype=torch.float32)

    projection = project_macro_policy_from_shadow_price(
        shadow_action_values=action_values,
        teacher_log_probs=teacher_log_probs,
        max_kl=0.05,
    )

    assert projection.actor_enabled
    assert projection.kl <= 0.05 + 1e-6
    # Action 0 has higher shadow value, so posterior probability should increase
    assert float(projection.target_probabilities[0].item()) > float(projection.teacher_probabilities[0].item())


def test_macro_credit_learner_config_target_mode() -> None:
    cfg_default = MacroCreditLearnerConfig()
    assert cfg_default.target_mode == "counterfactual_rollout"

    cfg_shadow = MacroCreditLearnerConfig(target_mode="combat_shadow_price")
    assert cfg_shadow.target_mode == "combat_shadow_price"

    with pytest.raises(ValueError, match="target_mode"):
        MacroCreditLearnerConfig(target_mode="invalid_mode")
