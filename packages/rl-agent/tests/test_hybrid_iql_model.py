from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from sts2_rl.models import (  # noqa: E402
    GroundedCandidateConfig,
    HybridCandidateQConfig,
    HybridDomainCandidateQModel,
    HybridDomainIQLModel,
    HybridIQLConfig,
    initialize_iql_from_candidate_q,
)
from tests.test_grounded_candidate_model import _make_batch  # noqa: E402


def _grounded() -> GroundedCandidateConfig:
    return GroundedCandidateConfig(
        token_feature_dim=224,
        d_model=32,
        n_heads=4,
        ffn_dim=64,
        world_layers=1,
        latent_slots=4,
        latent_layers=1,
        local_layers=1,
        candidate_layers=1,
        recurrent_hidden_dim=64,
        dropout=0.0,
    )


def test_iql_model_has_q_value_and_actor_outputs() -> None:
    model = HybridDomainIQLModel(
        HybridIQLConfig(grounded=_grounded(), domain_layers=1, joint_layers=1)
    ).eval()
    output = model(_make_batch(model.config))
    assert output.q_values.shape == output.policy_logits.shape
    assert output.state_value.shape == output.q_values.shape[:1]
    output.validate(model.iql_config)


def test_iql_actor_features_do_not_move_the_shared_critic_trunk() -> None:
    model = HybridDomainIQLModel(
        HybridIQLConfig(grounded=_grounded(), domain_layers=1, joint_layers=1)
    ).train()
    model(_make_batch(model.config)).policy_logits[:, 0].sum().backward()
    assert any(parameter.grad is not None for parameter in model.candidate_policy_head.parameters())
    assert all(parameter.grad is None for parameter in model.joint_stream.parameters())
    assert all(parameter.grad is None for parameter in model.candidate_q_head.parameters())


def test_candidate_q_migration_preserves_gate_zero_ranking_in_actor() -> None:
    grounded = _grounded()
    source = HybridDomainCandidateQModel(
        HybridCandidateQConfig(grounded=grounded, domain_layers=1, joint_layers=1)
    ).eval()
    target = HybridDomainIQLModel(
        HybridIQLConfig(grounded=grounded, domain_layers=1, joint_layers=1)
    ).eval()
    result = initialize_iql_from_candidate_q(target, dict(source.state_dict()))
    assert any(key.startswith("candidate_q_head.") for key in result["inherited"])
    batch = _make_batch(source.config)
    with torch.no_grad():
        old = source(batch)
        new = target(batch)
    torch.testing.assert_close(new.q_values, old.q_values)
    torch.testing.assert_close(new.policy_logits, old.q_values)

