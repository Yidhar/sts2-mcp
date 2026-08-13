from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from sts2_rl.models import (  # noqa: E402
    GroundedCandidateConfig,
    HybridCandidateQConfig,
    HybridDomainCandidateQModel,
)
from tests.test_grounded_candidate_model import _make_batch  # noqa: E402


def _model() -> HybridDomainCandidateQModel:
    grounded = GroundedCandidateConfig(
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
    return HybridDomainCandidateQModel(
        HybridCandidateQConfig(grounded=grounded, domain_layers=1, joint_layers=1)
    )


def test_hybrid_model_has_one_candidate_q_control_head() -> None:
    model = _model()
    assert hasattr(model, "candidate_q_head")
    for name in ("policy_head", "value_head", "transaction_q_head"):
        assert not hasattr(model, name)


def test_world_facts_are_candidate_independent_and_permutation_equivariant() -> None:
    model = _model().eval()
    batch = _make_batch(model.config)
    with torch.no_grad():
        original = model(batch)
    permutation = torch.tensor([2, 4, 0, 3, 1])
    permuted = batch.permute_candidates(permutation)
    with torch.no_grad():
        changed = model(permuted)
    torch.testing.assert_close(
        original.world_state_embedding, changed.world_state_embedding
    )
    torch.testing.assert_close(original.q_values[:, permutation], changed.q_values)


def test_domain_local_gradients_do_not_cross_stems() -> None:
    model = _model().train()
    macro = _make_batch(model.config)
    macro.domain_ids.fill_(0)
    model(macro).q_values[:, 0].sum().backward()
    assert any(parameter.grad is not None for parameter in model.macro_stream.parameters())
    assert all(parameter.grad is None for parameter in model.combat_stream.parameters())
    model.zero_grad(set_to_none=True)
    combat = _make_batch(model.config)
    combat.domain_ids.fill_(1)
    model(combat).q_values[:, 0].sum().backward()
    assert any(parameter.grad is not None for parameter in model.combat_stream.parameters())
    assert all(parameter.grad is None for parameter in model.macro_stream.parameters())


def test_split_memory_has_run_and_encounter_lifecycles() -> None:
    model = _model().eval()
    macro = _make_batch(model.config)
    macro.domain_ids.fill_(0)
    combat = _make_batch(model.config)
    combat.domain_ids.fill_(1)
    state = model.initial_state(len(macro.domain_ids), device=torch.device("cpu"))
    with torch.no_grad():
        after_macro = model(macro, state).recurrent_state
        after_combat = model(combat, after_macro).recurrent_state
        after_next_macro = model(macro, after_combat).recurrent_state
    hidden = model.config.recurrent_hidden_dim // 2
    run_after_macro, combat_after_macro = after_macro.split(hidden, dim=-1)
    run_after_combat, combat_after_combat = after_combat.split(hidden, dim=-1)
    _, combat_after_next_macro = after_next_macro.split(hidden, dim=-1)
    torch.testing.assert_close(run_after_combat, run_after_macro)
    torch.testing.assert_close(combat_after_macro, torch.zeros_like(combat_after_macro))
    assert not torch.equal(combat_after_combat, combat_after_macro)
    torch.testing.assert_close(
        combat_after_next_macro,
        torch.zeros_like(combat_after_next_macro),
    )
