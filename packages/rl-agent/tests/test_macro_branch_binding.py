"""Recorded production collision: exact policy groups, unchanged embeddings."""
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from sts2_rl.encoding import GroundedObservationEncoder
from sts2_rl.encoding.snapshot import GroundedEncodingConfig
from sts2_rl.joint.macro_credit_collection import StochasticBranchAuthority
from sts2_rl.macro_branch_diagnostic import BranchLimits, _MacroHorizonReached
from sts2_rl.models import HybridActorCriticConfig, HybridDomainActorCriticModel
from sts2_rl.models.grounded_candidate import GroundedCandidateConfig, RecurrentCandidateModel
from sts2_rl.semantics.forward import forward_decision


def test_recorded_reward_collision_and_bootstrap_are_representable() -> None:
    data = json.loads((Path(__file__).parent / "fixtures/macro_reward_branch_collision.json").read_text())
    encoder = GroundedObservationEncoder(GroundedEncodingConfig())
    encoded = encoder.encode(data["observation"], data["semantic_actions"])
    batch = encoded.batch.candidates
    assert batch.role_ids[0, 0].item() == batch.role_ids[0, 2].item() == 16
    assert batch.policy_branch_ids[0, 0] != batch.policy_branch_ids[0, 2]
    assert len(set(batch.policy_branch_ids[0].tolist())) == 5
    reverse = encoder.encode(data["observation"], data["semantic_actions"][::-1])
    assert torch.equal(batch.policy_branch_ids.flip(1), reverse.batch.candidates.policy_branch_ids)
    authority = object.__new__(StochasticBranchAuthority)
    authority.root_ordinal = 87
    authority.post_root_macro_count = 12
    authority.branch_limits = BranchLimits(continuation_macro_decisions=12)
    authority._encoder = encoder
    decision = forward_decision(data["observation"], data["semantic_actions"])
    with pytest.raises(_MacroHorizonReached):
        authority._prepare_semantic_exploration(
            observation=data["observation"], decision=decision,
            candidates=decision.candidates, control_domain="macro")
    assert authority.bootstrap_snapshot is not None


def test_bindings_do_not_enter_hybrid_learned_representation() -> None:
    from tests.test_grounded_candidate_model import _make_batch
    config = GroundedCandidateConfig(d_model=32, n_heads=4, ffn_dim=64, world_layers=1,
                                    latent_slots=4, latent_layers=1, local_layers=1,
                                    candidate_layers=1, recurrent_hidden_dim=64, dropout=0.0)
    batch = _make_batch(config)
    altered = replace(batch, candidates=replace(batch.candidates,
                       policy_branch_ids=torch.zeros_like(batch.candidates.policy_branch_ids)))
    model = HybridDomainActorCriticModel(HybridActorCriticConfig(config)).eval()
    with torch.no_grad():
        original, changed = model(batch), model(altered)
    for field in ("policy_logits", "value", "recurrent_state", "candidate_embeddings"):
        torch.testing.assert_close(getattr(original, field), getattr(changed, field), rtol=0, atol=0)
    hierarchy = RecurrentCandidateModel(config).eval()
    with torch.no_grad():
        output = hierarchy(batch)
    assert torch.equal(output.policy_branch_ids, batch.candidates.policy_branch_ids)
    assert torch.isfinite(output.policy_probabilities()).all()


def test_branch_binding_bounds_are_checked() -> None:
    from tests.test_grounded_candidate_model import _make_batch
    config = GroundedCandidateConfig()
    batch = _make_batch(config)
    corrupt = replace(batch.candidates, policy_branch_ids=torch.full_like(batch.candidates.policy_branch_ids, 100))
    with pytest.raises(ValueError, match="policy_branch_ids"):
        corrupt.validate(config)
