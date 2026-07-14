from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from sts2_rl.models import (
    CandidateTokenBatch,
    GroundedCandidateBatch,
    GroundedCandidateConfig,
    RecurrentCandidateModel,
    WorldEncoding,
    WorldTokenBatch,
)


@pytest.fixture
def config() -> GroundedCandidateConfig:
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
        dropout=0.0,
        domain_count=8,
        type_vocab_size=16,
        role_vocab_size=12,
        owner_vocab_size=16,
        entity_vocab_size=64,
        zone_vocab_size=10,
        order_vocab_size=16,
    )


def _ids(shape: tuple[int, ...], size: int) -> torch.Tensor:
    return torch.randint(0, size, shape, dtype=torch.long)


def _make_batch(config: GroundedCandidateConfig) -> GroundedCandidateBatch:
    torch.manual_seed(7)
    batch_size, world_count, action_count, local_count = 2, 7, 5, 3
    world = WorldTokenBatch(
        features=torch.randn(batch_size, world_count, config.token_feature_dim),
        mask=torch.tensor(
            [[True, True, True, True, False, False, False], [True, True, True, True, True, False, False]]
        ),
        type_ids=_ids((batch_size, world_count), config.type_vocab_size),
        role_ids=_ids((batch_size, world_count), config.role_vocab_size),
        owner_ids=_ids((batch_size, world_count), config.owner_vocab_size),
        entity_ids=_ids((batch_size, world_count), config.entity_vocab_size),
        entity_aux_ids=_ids((batch_size, world_count), config.entity_vocab_size),
        zone_ids=_ids((batch_size, world_count), config.zone_vocab_size),
        order_ids=_ids((batch_size, world_count), config.order_vocab_size),
    )
    candidates = CandidateTokenBatch(
        features=torch.randn(batch_size, action_count, config.token_feature_dim),
        type_ids=_ids((batch_size, action_count), config.type_vocab_size),
        role_ids=_ids((batch_size, action_count), config.role_vocab_size),
        owner_ids=_ids((batch_size, action_count), config.owner_vocab_size),
        entity_ids=_ids((batch_size, action_count), config.entity_vocab_size),
        entity_aux_ids=_ids((batch_size, action_count), config.entity_vocab_size),
        zone_ids=_ids((batch_size, action_count), config.zone_vocab_size),
        target_owner_ids=_ids((batch_size, action_count), config.owner_vocab_size),
        target_entity_ids=_ids((batch_size, action_count), config.entity_vocab_size),
        target_entity_aux_ids=_ids(
            (batch_size, action_count),
            config.entity_vocab_size,
        ),
        local_features=torch.randn(
            batch_size,
            action_count,
            local_count,
            config.token_feature_dim,
        ),
        local_mask=torch.tensor(
            [
                [[True, True, False], [True, False, False], [True, True, True], [False, False, False], [True, True, False]],
                [[True, False, False], [True, True, False], [False, False, False], [True, True, True], [True, False, False]],
            ]
        ),
        local_type_ids=_ids((batch_size, action_count, local_count), config.type_vocab_size),
        local_role_ids=_ids((batch_size, action_count, local_count), config.role_vocab_size),
        local_owner_ids=_ids((batch_size, action_count, local_count), config.owner_vocab_size),
        local_entity_ids=_ids((batch_size, action_count, local_count), config.entity_vocab_size),
        local_entity_aux_ids=_ids(
            (batch_size, action_count, local_count),
            config.entity_vocab_size,
        ),
        local_zone_ids=_ids((batch_size, action_count, local_count), config.zone_vocab_size),
        local_order_ids=_ids((batch_size, action_count, local_count), config.order_vocab_size),
        action_mask=torch.tensor(
            [[True, True, False, True, True], [True, False, True, True, False]]
        ),
    )
    return GroundedCandidateBatch(
        world=world,
        candidates=candidates,
        domain_ids=torch.tensor([0, 2], dtype=torch.long),
    )


def test_candidate_permutation_equivariance(config: GroundedCandidateConfig) -> None:
    model = RecurrentCandidateModel(config).eval()
    batch = _make_batch(config)
    permutation = torch.tensor([2, 4, 0, 3, 1])

    with torch.no_grad():
        output = model(batch)
        permuted = model(batch.permute_candidates(permutation))

    torch.testing.assert_close(permuted.world_latents, output.world_latents)
    torch.testing.assert_close(permuted.state_embedding, output.state_embedding)
    torch.testing.assert_close(permuted.recurrent_state, output.recurrent_state)
    torch.testing.assert_close(permuted.value, output.value)
    for actual, expected in (
        (permuted.candidate_embeddings, output.candidate_embeddings[:, permutation]),
        (permuted.policy_logits, output.policy_logits[:, permutation]),
    ):
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)


def test_masked_candidates_are_inert(config: GroundedCandidateConfig) -> None:
    model = RecurrentCandidateModel(config).eval()
    output = model(_make_batch(config))
    invalid = ~output.action_mask

    assert torch.equal(output.policy_logits[invalid], torch.full_like(output.policy_logits[invalid], torch.finfo(output.policy_logits.dtype).min))
    assert torch.count_nonzero(output.candidate_embeddings[invalid]) == 0

    probabilities = output.policy_probabilities()
    assert torch.count_nonzero(probabilities[invalid]) == 0
    torch.testing.assert_close(probabilities.sum(dim=-1), torch.ones(probabilities.shape[0]))


def test_all_masked_policy_is_safe(config: GroundedCandidateConfig) -> None:
    model = RecurrentCandidateModel(config).eval()
    batch = _make_batch(config)
    all_masked = replace(
        batch,
        candidates=replace(
            batch.candidates,
            action_mask=torch.zeros_like(batch.candidates.action_mask),
        ),
    )
    output = model(all_masked)
    assert torch.isfinite(output.policy_probabilities()).all()
    assert torch.count_nonzero(output.policy_probabilities()) == 0


def test_world_encoding_and_state_values_do_not_read_candidates(
    config: GroundedCandidateConfig,
) -> None:
    model = RecurrentCandidateModel(config).eval()
    batch = _make_batch(config)
    altered_candidates = replace(
        batch.candidates,
        features=torch.randn_like(batch.candidates.features) * 100.0,
        entity_ids=torch.flip(batch.candidates.entity_ids, dims=(1,)),
        local_features=torch.randn_like(batch.candidates.local_features) * 100.0,
        action_mask=~batch.candidates.action_mask,
    )
    altered = replace(batch, candidates=altered_candidates)

    with torch.no_grad():
        world_a = model.encode_world(batch.world, batch.domain_ids)
        world_b = model.encode_world(altered.world, altered.domain_ids)
        output_a = model(batch)
        output_b = model(altered)

    torch.testing.assert_close(world_a.latents, world_b.latents)
    torch.testing.assert_close(world_a.state_embedding, world_b.state_embedding)
    torch.testing.assert_close(output_a.state_embedding, output_b.state_embedding)
    torch.testing.assert_close(output_a.recurrent_state, output_b.recurrent_state)
    torch.testing.assert_close(output_a.value, output_b.value)


def test_forward_backward_reaches_shared_world_and_candidate_parameters(
    config: GroundedCandidateConfig,
) -> None:
    model = RecurrentCandidateModel(config).train()
    batch = _make_batch(config)
    output = model(batch)
    probabilities = output.policy_probabilities()
    selected = torch.tensor([0, 2], dtype=torch.long)
    policy_loss = -torch.log(probabilities[torch.arange(2), selected].clamp_min(1e-8)).mean()
    loss = policy_loss + output.value.square().mean()
    loss.backward()

    assert torch.isfinite(loss)
    world_grad = model.world_encoder.layers[0].self_attn.in_proj_weight.grad
    candidate_grad = model.policy_head[-1].weight.grad
    shared_embedding_grad = model.token_embedder.entity_embedding.weight.grad
    recurrent_grad = model.recurrent_cell.weight_hh.grad
    assert world_grad is not None and torch.isfinite(world_grad).all()
    assert candidate_grad is not None and torch.isfinite(candidate_grad).all()
    assert shared_embedding_grad is not None and torch.isfinite(shared_embedding_grad).all()
    assert recurrent_grad is not None and torch.isfinite(recurrent_grad).all()


def test_default_model_stays_small() -> None:
    model = RecurrentCandidateModel()
    assert model.parameter_count == 3_980_098


def test_batch_shape_contract_rejects_misaligned_candidate_local_axis(
    config: GroundedCandidateConfig,
) -> None:
    batch = _make_batch(config)
    broken = replace(
        batch,
        candidates=replace(
            batch.candidates,
            local_mask=batch.candidates.local_mask[:, :-1],
        ),
    )
    with pytest.raises(ValueError, match="local_mask"):
        broken.validate(config)


def test_batch_contract_rejects_non_finite_padding_and_coerced_masks(
    config: GroundedCandidateConfig,
) -> None:
    batch = _make_batch(config)
    poisoned_features = batch.world.features.clone()
    poisoned_features[0, -1, 0] = torch.nan
    poisoned = replace(
        batch,
        world=replace(batch.world, features=poisoned_features),
    )
    with pytest.raises(ValueError, match="NaN or infinity"):
        poisoned.validate(config)

    float_mask = replace(
        batch,
        candidates=replace(
            batch.candidates,
            action_mask=batch.candidates.action_mask.float(),
        ),
    )
    with pytest.raises(TypeError, match="torch.bool"):
        float_mask.validate(config)


def test_batch_contract_rejects_out_of_range_ids(
    config: GroundedCandidateConfig,
) -> None:
    batch = _make_batch(config)
    invalid_ids = batch.world.entity_ids.clone()
    invalid_ids[0, 0] = config.entity_vocab_size
    broken = replace(
        batch,
        world=replace(batch.world, entity_ids=invalid_ids),
    )
    with pytest.raises(ValueError, match="outside"):
        broken.validate(config)


def test_model_respects_requested_floating_dtype(
    config: GroundedCandidateConfig,
) -> None:
    model = RecurrentCandidateModel(config).double().eval()
    batch = _make_batch(config)
    double_batch = replace(
        batch,
        world=replace(batch.world, features=batch.world.features.double()),
        candidates=replace(
            batch.candidates,
            features=batch.candidates.features.double(),
            local_features=batch.candidates.local_features.double(),
        ),
    )
    with torch.no_grad():
        output = model(double_batch)
    assert output.policy_logits.dtype == torch.float64
    assert output.policy_probabilities().dtype == torch.float32


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"token_feature_dim": 223}, "at least 224"),
        ({"domain_count": 3}, "six fixed"),
        ({"entity_vocab_size": 1}, "at least 4"),
        ({"order_vocab_size": 1}, "at least 2"),
        ({"recurrent_hidden_dim": 0}, "positive"),
    ],
)
def test_model_config_matches_encoder_minimum_contract(
    changes: dict[str, int],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        GroundedCandidateConfig(**changes)


def test_model_config_rejects_float_and_bool_dimensions_early() -> None:
    with pytest.raises(TypeError, match="exact integers"):
        GroundedCandidateConfig(d_model=128.0)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="exact integers"):
        GroundedCandidateConfig(world_layers=True)  # type: ignore[arg-type]


def test_tensor_contract_rejects_zero_sized_batch(
    config: GroundedCandidateConfig,
) -> None:
    batch = _make_batch(config)
    empty = replace(
        batch,
        world=replace(
            batch.world,
            features=batch.world.features[:0],
            mask=batch.world.mask[:0],
            type_ids=batch.world.type_ids[:0],
            role_ids=batch.world.role_ids[:0],
            owner_ids=batch.world.owner_ids[:0],
            entity_ids=batch.world.entity_ids[:0],
            entity_aux_ids=batch.world.entity_aux_ids[:0],
            zone_ids=batch.world.zone_ids[:0],
            order_ids=batch.world.order_ids[:0],
        ),
        candidates=replace(
            batch.candidates,
            features=batch.candidates.features[:0],
            type_ids=batch.candidates.type_ids[:0],
            role_ids=batch.candidates.role_ids[:0],
            owner_ids=batch.candidates.owner_ids[:0],
            entity_ids=batch.candidates.entity_ids[:0],
            entity_aux_ids=batch.candidates.entity_aux_ids[:0],
            zone_ids=batch.candidates.zone_ids[:0],
            target_owner_ids=batch.candidates.target_owner_ids[:0],
            target_entity_ids=batch.candidates.target_entity_ids[:0],
            target_entity_aux_ids=batch.candidates.target_entity_aux_ids[:0],
            local_features=batch.candidates.local_features[:0],
            local_mask=batch.candidates.local_mask[:0],
            local_type_ids=batch.candidates.local_type_ids[:0],
            local_role_ids=batch.candidates.local_role_ids[:0],
            local_owner_ids=batch.candidates.local_owner_ids[:0],
            local_entity_ids=batch.candidates.local_entity_ids[:0],
            local_entity_aux_ids=batch.candidates.local_entity_aux_ids[:0],
            local_zone_ids=batch.candidates.local_zone_ids[:0],
            local_order_ids=batch.candidates.local_order_ids[:0],
            action_mask=batch.candidates.action_mask[:0],
        ),
        domain_ids=batch.domain_ids[:0],
    )

    with pytest.raises(ValueError, match="batch size must be positive"):
        empty.validate(config)


def test_public_candidate_encoder_rejects_broadcastable_world_state(
    config: GroundedCandidateConfig,
) -> None:
    model = RecurrentCandidateModel(config).eval()
    batch = _make_batch(config)
    world = model.encode_world(batch.world, batch.domain_ids)
    malformed = WorldEncoding(
        latents=world.latents,
        state_embedding=world.state_embedding[:1],
        world_mask=world.world_mask,
    )
    with pytest.raises(ValueError, match="state_embedding"):
        model.encode_candidates(batch.candidates, malformed)
