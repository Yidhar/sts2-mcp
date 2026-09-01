from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from sts2_rl.models import (  # noqa: E402
    GroundedCandidateConfig,
    HybridActorCriticConfig,
    HybridDomainActorCriticModel,
    HybridDomainIQLModel,
    HybridIQLConfig,
    initialize_actor_critic_from_iql,
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


def test_actor_policy_gradient_reaches_the_shared_hybrid_stream() -> None:
    model = HybridDomainActorCriticModel(
        HybridActorCriticConfig(
            grounded=_grounded(), domain_layers=1, joint_layers=1
        )
    ).eval()
    output = model(_make_batch(model.config))
    output.policy_logits[:, 0].sum().backward()
    assert any(parameter.grad is not None for parameter in model.candidate_policy_head.parameters())
    assert any(parameter.grad is not None for parameter in model.joint_stream.parameters())


def test_healthy_iql_parent_preserves_gate_zero_policy_and_value() -> None:
    grounded = _grounded()
    source = HybridDomainIQLModel(
        HybridIQLConfig(grounded=grounded, domain_layers=1, joint_layers=1)
    ).eval()
    target = HybridDomainActorCriticModel(
        HybridActorCriticConfig(
            grounded=grounded, domain_layers=1, joint_layers=1
        )
    ).eval()
    result = initialize_actor_critic_from_iql(target, dict(source.state_dict()))
    assert any(key.startswith("candidate_policy_head.") for key in result["inherited"])
    batch = _make_batch(source.config)
    with torch.no_grad():
        old = source(batch)
        new = target(batch)
    torch.testing.assert_close(new.policy_logits, old.policy_logits)
    torch.testing.assert_close(new.value, old.state_value)
