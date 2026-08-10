from __future__ import annotations

import math
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
from sts2_rl.models.grounded_candidate import _safe_valid_mask

MULTISCALE_STATE_VALUE_FIELDS = (
    "combat_task_value",
    "act_task_value",
    "run_task_value",
    "combat_revival_cost_value",
    "act_revival_cost_value",
    "run_revival_cost_value",
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


def test_safe_valid_mask_only_opens_first_column_for_empty_rows() -> None:
    mask = torch.tensor(
        [
            [False, False, False, False],
            [False, True, False, False],
            [True, False, True, False],
        ],
        dtype=torch.bool,
    )

    safe = _safe_valid_mask(mask)

    assert torch.equal(
        safe,
        torch.tensor(
            [
                [True, False, False, False],
                [False, True, False, False],
                [True, False, True, False],
            ],
            dtype=torch.bool,
        ),
    )
    assert torch.equal(mask[1:], safe[1:])


def test_safe_valid_mask_supports_leading_attention_dimensions() -> None:
    mask = torch.tensor(
        [
            [[False, False, False], [False, True, False]],
            [[True, False, False], [False, False, False]],
        ],
        dtype=torch.bool,
    )

    safe = _safe_valid_mask(mask)

    assert safe.shape == mask.shape
    assert bool(safe.any(dim=-1).all().item())
    assert torch.equal(safe[0, 1], mask[0, 1])
    assert torch.equal(safe[1, 0], mask[1, 0])
    assert torch.equal(safe[0, 0], torch.tensor([True, False, False]))
    assert torch.equal(safe[1, 1], torch.tensor([True, False, False]))


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
        definition_binding_ids=_ids(
            (batch_size, world_count),
            config.entity_vocab_size,
        ),
        relation_binding_ids=_ids(
            (batch_size, world_count),
            config.entity_vocab_size,
        ),
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
        definition_binding_ids=_ids(
            (batch_size, action_count),
            config.entity_vocab_size,
        ),
        relation_binding_ids=_ids(
            (batch_size, action_count),
            config.entity_vocab_size,
        ),
        zone_ids=_ids((batch_size, action_count), config.zone_vocab_size),
        target_owner_ids=_ids((batch_size, action_count), config.owner_vocab_size),
        target_entity_ids=_ids((batch_size, action_count), config.entity_vocab_size),
        target_entity_aux_ids=_ids(
            (batch_size, action_count),
            config.entity_vocab_size,
        ),
        target_definition_binding_ids=_ids(
            (batch_size, action_count),
            config.entity_vocab_size,
        ),
        target_relation_binding_ids=_ids(
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
                [
                    [True, True, False],
                    [True, False, False],
                    [True, True, True],
                    [False, False, False],
                    [True, True, False],
                ],
                [
                    [True, False, False],
                    [True, True, False],
                    [False, False, False],
                    [True, True, True],
                    [True, False, False],
                ],
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
        local_definition_binding_ids=_ids(
            (batch_size, action_count, local_count),
            config.entity_vocab_size,
        ),
        local_relation_binding_ids=_ids(
            (batch_size, action_count, local_count),
            config.entity_vocab_size,
        ),
        local_zone_ids=_ids((batch_size, action_count, local_count), config.zone_vocab_size),
        local_order_ids=_ids((batch_size, action_count, local_count), config.order_vocab_size),
        action_mask=torch.tensor([[True, True, False, True, True], [True, False, True, True, False]]),
    )
    return GroundedCandidateBatch(
        world=world,
        candidates=candidates,
        domain_ids=torch.tensor([0, 2], dtype=torch.long),
        macro_economic_surface_ids=torch.zeros(
            (batch_size,), dtype=torch.long
        ),
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
    for field in MULTISCALE_STATE_VALUE_FIELDS:
        torch.testing.assert_close(getattr(permuted, field), getattr(output, field))
    for actual, expected in (
        (permuted.candidate_embeddings, output.candidate_embeddings[:, permutation]),
        (permuted.policy_logits, output.policy_logits[:, permutation]),
        (
            permuted.policy_branch_ids,
            output.policy_branch_ids[:, permutation],
        ),
    ):
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)


def test_macro_awr_heads_start_behavior_neutral_and_detach_shared_trunk(
    config: GroundedCandidateConfig,
) -> None:
    model = RecurrentCandidateModel(
        config,
        enable_transaction_heads=True,
    ).eval()
    batch = _make_batch(config)
    macro_batch = replace(
        batch,
        macro_economic_surface_ids=torch.tensor([1, 2], dtype=torch.long),
    )

    baseline = model(batch)
    output = model(macro_batch)
    assert output.macro_policy_logits is not None
    assert output.macro_option_value is not None
    # Reviewed model-init starts from exactly the inherited behavior.
    assert torch.equal(output.policy_logits, baseline.policy_logits)
    assert torch.equal(output.macro_policy_logits, output.policy_logits)
    assert torch.count_nonzero(output.macro_option_value) == 0

    model.zero_grad(set_to_none=True)
    loss = (
        -output.macro_policy_log_probabilities()[0, 0]
        + (output.macro_option_value - 1.0).square().mean()
    )
    loss.backward()

    assert model.macro_policy_head is not None
    assert model.macro_option_value_head is not None
    assert model.macro_policy_head[-1].weight.grad is not None
    assert model.macro_option_value_head[-1].weight.grad is not None
    # Macro value/AWR losses update only sidecars, never the general policy or
    # shared tactical representation.
    macro_prefixes = (
        "macro_surface_candidate_embedding.",
        "macro_surface_state_embedding.",
        "macro_policy_head.",
        "macro_option_value_head.",
    )
    assert all(
        parameter.grad is None
        for name, parameter in model.named_parameters()
        if not name.startswith(macro_prefixes)
    )


def test_masked_candidates_are_inert(config: GroundedCandidateConfig) -> None:
    model = RecurrentCandidateModel(config).eval()
    output = model(_make_batch(config))
    invalid = ~output.action_mask

    assert torch.equal(
        output.policy_logits[invalid],
        torch.full_like(output.policy_logits[invalid], torch.finfo(output.policy_logits.dtype).min),
    )
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


@pytest.mark.parametrize(
    ("many_branch_label", "singleton_branch_label"),
    [
        ("play_card", "end_turn"),
        ("claim_reward", "proceed"),
    ],
)
def test_hierarchical_policy_does_not_turn_branch_cardinality_into_preference(
    config: GroundedCandidateConfig,
    many_branch_label: str,
    singleton_branch_label: str,
) -> None:
    """Cloning an equal-score branch cannot change its branch marginal.

    The labels document the two production failures this synthetic policy
    surface covers.  Learned branch IDs are opaque integers at model runtime;
    their semantic origin is validated by the grounded encoder tests.
    """

    del many_branch_label, singleton_branch_label
    model = RecurrentCandidateModel(config).eval()
    template = model(_make_batch(config))
    many_branch = 2
    singleton_branch = 5

    two_candidates = replace(
        template,
        policy_logits=torch.tensor([[0.2, 0.2, 0.0]]),
        policy_branch_ids=torch.tensor(
            [[many_branch, many_branch, singleton_branch]],
            dtype=torch.long,
        ),
        action_mask=torch.ones((1, 3), dtype=torch.bool),
    )
    four_candidates = replace(
        template,
        policy_logits=torch.tensor([[0.2, 0.2, 0.2, 0.2, 0.0]]),
        policy_branch_ids=torch.tensor(
            [
                [
                    many_branch,
                    many_branch,
                    many_branch,
                    many_branch,
                    singleton_branch,
                ]
            ],
            dtype=torch.long,
        ),
        action_mask=torch.ones((1, 5), dtype=torch.bool),
    )

    two_marginal = two_candidates.policy_branch_probabilities()[0]
    four_marginal = four_candidates.policy_branch_probabilities()[0]
    torch.testing.assert_close(
        two_marginal[[many_branch, singleton_branch]],
        four_marginal[[many_branch, singleton_branch]],
    )
    assert two_marginal[many_branch] > two_marginal[singleton_branch]

    # A joint candidate argmax still favors the singleton because the preferred
    # branch's mass is divided among concrete actions. Deterministic dispatch
    # must therefore use the explicit branch-first primitive.
    assert int(two_candidates.policy_probabilities()[0].argmax().item()) == 2
    assert int(four_candidates.policy_probabilities()[0].argmax().item()) == 4
    assert int(two_candidates.greedy_action_indices()[0].item()) in {0, 1}
    assert int(four_candidates.greedy_action_indices()[0].item()) in {
        0,
        1,
        2,
        3,
    }
    torch.testing.assert_close(
        two_candidates.policy_entropy(),
        four_candidates.policy_entropy(),
    )


def test_hierarchical_policy_normalizes_entropy_by_its_exact_capacity(
    config: GroundedCandidateConfig,
) -> None:
    model = RecurrentCandidateModel(config).eval()
    template = model(_make_batch(config))
    multi_branch = 2
    singleton_branch = 5

    # Uniform actions inside the multi-candidate branch and branch logits
    # proportional to exp(q_b) attain the exact capacity log(e + 1).
    maximum_entropy = replace(
        template,
        policy_logits=torch.tensor([[1.0, 1.0, 0.0]]),
        policy_branch_ids=torch.tensor(
            [[multi_branch, multi_branch, singleton_branch]],
            dtype=torch.long,
        ),
        action_mask=torch.ones((1, 3), dtype=torch.bool),
    )
    entropy, normalized = maximum_entropy.policy_entropy_and_normalized()
    torch.testing.assert_close(entropy, torch.tensor([math.log(math.e + 1.0)]))
    torch.testing.assert_close(normalized, torch.ones(1))

    # The old flat normalization used log(3), which is not an upper bound for
    # the count-balanced hierarchical objective and caused the v37 first
    # learner update to fail closed.
    assert float((entropy / math.log(3.0)).item()) > 1.0

    collapsed = replace(
        maximum_entropy,
        policy_logits=torch.tensor([[20.0, -20.0, -20.0]]),
    )
    _, collapsed_normalized = collapsed.policy_entropy_and_normalized()
    assert 0.0 <= float(collapsed_normalized.item()) < 1.0e-6


def test_confirm_and_deselect_are_learned_distinct_branches_without_hard_coding(
    config: GroundedCandidateConfig,
) -> None:
    model = RecurrentCandidateModel(config).eval()
    template = model(_make_batch(config))
    deselect_branch = 3
    confirm_branch = 7
    surface = replace(
        template,
        policy_logits=torch.tensor([[0.1, 0.4]]),
        policy_branch_ids=torch.tensor(
            [[deselect_branch, confirm_branch]],
            dtype=torch.long,
        ),
        action_mask=torch.ones((1, 2), dtype=torch.bool),
    )

    assert int(surface.greedy_action_indices()[0].item()) == 1
    reversed_surface = replace(
        surface,
        policy_logits=torch.tensor([[0.5, -0.2]]),
    )
    assert int(reversed_surface.greedy_action_indices()[0].item()) == 0


def test_hierarchical_policy_adds_no_checkpoint_parameters(
    config: GroundedCandidateConfig,
) -> None:
    model = RecurrentCandidateModel(config).eval()
    before = {key: value.detach().clone() for key, value in model.state_dict().items()}

    output = model(_make_batch(config))
    output.policy_probabilities()
    output.policy_branch_probabilities()
    output.greedy_action_indices()
    output.policy_entropy()

    assert set(model.state_dict()) == set(before)
    for key, expected in before.items():
        torch.testing.assert_close(model.state_dict()[key], expected)


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
    for field in MULTISCALE_STATE_VALUE_FIELDS:
        torch.testing.assert_close(getattr(output_a, field), getattr(output_b, field))


def test_multiscale_state_values_are_well_formed(
    config: GroundedCandidateConfig,
) -> None:
    model = RecurrentCandidateModel(config).eval()

    with torch.no_grad():
        output = model(_make_batch(config))

    assert output.validate(config) == (2, 5)
    for field in MULTISCALE_STATE_VALUE_FIELDS:
        value = getattr(output, field)
        assert value.shape == (2,)
        assert torch.isfinite(value).all()
    for field in (
        "combat_revival_cost_value",
        "act_revival_cost_value",
        "run_revival_cost_value",
    ):
        value = getattr(output, field)
        assert torch.all(value >= 0.0)
        # Episodic learning uses a log1p observation model for the long-tailed
        # factual count. The Softplus head support makes that transform total.
        assert torch.isfinite(torch.log1p(value)).all()
    assert output.combat_hp_loss_value.shape == (2,)
    assert torch.all(output.combat_hp_loss_value >= 0.0)
    assert torch.all(output.combat_hp_loss_value <= 1.0)


def test_output_validation_rejects_malformed_multiscale_value(
    config: GroundedCandidateConfig,
) -> None:
    model = RecurrentCandidateModel(config).eval()
    output = model(_make_batch(config))
    malformed_finite = replace(
        output,
        run_revival_cost_value=torch.full_like(
            output.run_revival_cost_value,
            torch.nan,
        ),
    )
    malformed_shape = replace(
        output,
        act_task_value=output.act_task_value[:1],
    )
    malformed_support = replace(
        output,
        combat_revival_cost_value=torch.full_like(
            output.combat_revival_cost_value,
            -1.0,
        ),
    )
    malformed_hp_support = replace(
        output,
        combat_hp_loss_value=torch.full_like(
            output.combat_hp_loss_value,
            1.1,
        ),
    )

    with pytest.raises(ValueError, match="NaN or infinity"):
        malformed_finite.validate(config)
    with pytest.raises(ValueError, match="must have shape"):
        malformed_shape.validate(config)
    with pytest.raises(ValueError, match="must be non-negative"):
        malformed_support.validate(config)
    with pytest.raises(ValueError, match=r"must be in \[0, 1\]"):
        malformed_hp_support.validate(config)


def test_forward_backward_reaches_shared_world_and_candidate_parameters(
    config: GroundedCandidateConfig,
) -> None:
    model = RecurrentCandidateModel(config).train()
    batch = replace(
        _make_batch(config),
        # Exercise one tactical and one run-scale memory update so both halves
        # of the maintained recurrent state participate in backpropagation.
        domain_ids=torch.tensor([1, 2], dtype=torch.long),
    )
    output = model(batch)
    probabilities = output.policy_probabilities()
    selected = torch.tensor([0, 2], dtype=torch.long)
    policy_loss = -torch.log(probabilities[torch.arange(2), selected].clamp_min(1e-8)).mean()
    multiscale_value_loss = sum(getattr(output, field).square().mean() for field in MULTISCALE_STATE_VALUE_FIELDS)
    multiscale_value_loss = (
        multiscale_value_loss + output.combat_hp_loss_value.square().mean()
    )
    loss = policy_loss + output.value.square().mean() + multiscale_value_loss
    loss.backward()

    assert torch.isfinite(loss)
    world_grad = model.world_encoder.layers[0].self_attn.in_proj_weight.grad
    candidate_grad = model.policy_head[-1].weight.grad
    shared_embedding_grad = model.token_embedder.entity_embedding.weight.grad
    target_entity_grad = model.token_embedder.target_entity_projection.weight.grad
    target_relation_grad = model.token_embedder.target_relation_projection.weight.grad
    run_recurrent_grad = model.run_recurrent_cell.weight_hh.grad
    combat_recurrent_grad = model.combat_recurrent_cell.weight_hh.grad
    assert world_grad is not None and torch.isfinite(world_grad).all()
    assert candidate_grad is not None and torch.isfinite(candidate_grad).all()
    assert shared_embedding_grad is not None and torch.isfinite(shared_embedding_grad).all()
    assert target_entity_grad is not None and torch.isfinite(target_entity_grad).all()
    assert target_relation_grad is not None and torch.isfinite(target_relation_grad).all()
    assert run_recurrent_grad is not None and torch.isfinite(run_recurrent_grad).all()
    assert combat_recurrent_grad is not None and torch.isfinite(combat_recurrent_grad).all()
    for head_name in (
        "combat_task_value_head",
        "act_task_value_head",
        "run_task_value_head",
        "combat_revival_cost_value_head",
        "act_revival_cost_value_head",
        "run_revival_cost_value_head",
        "combat_hp_loss_value_head",
    ):
        head_grad = getattr(model, head_name)[
            -2 if "revival" in head_name or "hp_loss" in head_name else -1
        ].weight.grad
        assert head_grad is not None and torch.isfinite(head_grad).all()


def test_default_model_stays_small() -> None:
    model = RecurrentCandidateModel()
    assert model.parameter_count == 4_447_049


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
        ({"recurrent_hidden_dim": 255}, "must be even"),
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
            definition_binding_ids=batch.world.definition_binding_ids[:0],
            relation_binding_ids=batch.world.relation_binding_ids[:0],
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
            definition_binding_ids=(batch.candidates.definition_binding_ids[:0]),
            relation_binding_ids=batch.candidates.relation_binding_ids[:0],
            zone_ids=batch.candidates.zone_ids[:0],
            target_owner_ids=batch.candidates.target_owner_ids[:0],
            target_entity_ids=batch.candidates.target_entity_ids[:0],
            target_entity_aux_ids=batch.candidates.target_entity_aux_ids[:0],
            target_definition_binding_ids=(batch.candidates.target_definition_binding_ids[:0]),
            target_relation_binding_ids=(batch.candidates.target_relation_binding_ids[:0]),
            local_features=batch.candidates.local_features[:0],
            local_mask=batch.candidates.local_mask[:0],
            local_type_ids=batch.candidates.local_type_ids[:0],
            local_role_ids=batch.candidates.local_role_ids[:0],
            local_owner_ids=batch.candidates.local_owner_ids[:0],
            local_entity_ids=batch.candidates.local_entity_ids[:0],
            local_entity_aux_ids=batch.candidates.local_entity_aux_ids[:0],
            local_definition_binding_ids=(batch.candidates.local_definition_binding_ids[:0]),
            local_relation_binding_ids=(batch.candidates.local_relation_binding_ids[:0]),
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
        tokens=world.tokens,
        world_mask=world.world_mask,
        entity_ids=world.entity_ids,
        entity_aux_ids=world.entity_aux_ids,
        definition_binding_ids=world.definition_binding_ids,
        relation_binding_ids=world.relation_binding_ids,
    )
    with pytest.raises(ValueError, match="state_embedding"):
        model.encode_candidates(batch.candidates, malformed)


def test_relation_pool_separates_runtime_instance_from_shared_definition(
    config: GroundedCandidateConfig,
) -> None:
    token_a = torch.arange(config.d_model, dtype=torch.float32)
    token_b = token_a + 100.0
    token_other = token_a + 1000.0
    tokens = torch.stack([token_a, token_b, token_other]).unsqueeze(0)
    world = WorldEncoding(
        latents=torch.zeros(1, config.latent_slots, config.d_model),
        state_embedding=torch.zeros(1, config.d_model),
        tokens=tokens,
        world_mask=torch.ones(1, 3, dtype=torch.bool),
        entity_ids=torch.tensor([[10, 10, 11]], dtype=torch.long),
        entity_aux_ids=torch.tensor([[21, 22, 23]], dtype=torch.long),
        definition_binding_ids=torch.tensor([[10, 10, 11]], dtype=torch.long),
        relation_binding_ids=torch.tensor([[21, 22, 23]], dtype=torch.long),
    )

    exact, definition = RecurrentCandidateModel._matched_world_contexts(
        definition_binding_ids=torch.tensor([[10, 10, 0]], dtype=torch.long),
        relation_binding_ids=torch.tensor([[21, 22, 0]], dtype=torch.long),
        world_encoding=world,
    )

    torch.testing.assert_close(exact[0, 0], token_a)
    torch.testing.assert_close(exact[0, 1], token_b)
    torch.testing.assert_close(definition[0, 0], (token_a + token_b) / 2.0)
    torch.testing.assert_close(definition[0, 1], (token_a + token_b) / 2.0)
    assert torch.count_nonzero(exact[0, 2]) == 0
    assert torch.count_nonzero(definition[0, 2]) == 0


def test_relation_pool_never_uses_colliding_embedding_hash_ids(
    config: GroundedCandidateConfig,
) -> None:
    """Exact grounding is independent of finite learned embedding buckets."""

    token_a = torch.full((config.d_model,), 3.0)
    token_b = torch.full((config.d_model,), 17.0)
    world = WorldEncoding(
        latents=torch.zeros(1, config.latent_slots, config.d_model),
        state_embedding=torch.zeros(1, config.d_model),
        tokens=torch.stack([token_a, token_b]).unsqueeze(0),
        world_mask=torch.ones(1, 2, dtype=torch.bool),
        # Deliberately collide both learned embedding namespaces.
        entity_ids=torch.tensor([[7, 7]], dtype=torch.long),
        entity_aux_ids=torch.tensor([[9, 9]], dtype=torch.long),
        # Exact decision-local bindings remain collision-free.
        definition_binding_ids=torch.tensor([[101, 102]], dtype=torch.long),
        relation_binding_ids=torch.tensor([[201, 202]], dtype=torch.long),
    )

    exact, definition = RecurrentCandidateModel._matched_world_contexts(
        definition_binding_ids=torch.tensor([[101, 102]], dtype=torch.long),
        relation_binding_ids=torch.tensor([[201, 202]], dtype=torch.long),
        world_encoding=world,
    )

    torch.testing.assert_close(exact[0, 0], token_a)
    torch.testing.assert_close(exact[0, 1], token_b)
    torch.testing.assert_close(definition[0, 0], token_a)
    torch.testing.assert_close(definition[0, 1], token_b)


def test_run_and_combat_memory_have_separate_update_scales(
    config: GroundedCandidateConfig,
) -> None:
    model = RecurrentCandidateModel(config).eval()
    base = _make_batch(config)
    half = config.recurrent_hidden_dim // 2

    with torch.no_grad():
        macro = model(replace(base, domain_ids=torch.full_like(base.domain_ids, 2))).recurrent_state
        combat = model(
            replace(base, domain_ids=torch.full_like(base.domain_ids, 1)),
            macro,
        ).recurrent_state
        after_combat = model(
            replace(base, domain_ids=torch.full_like(base.domain_ids, 2)),
            combat,
        ).recurrent_state

    torch.testing.assert_close(combat[:, :half], macro[:, :half])
    assert torch.count_nonzero(combat[:, half:]) > 0
    assert torch.count_nonzero(after_combat[:, half:]) == 0
    assert not torch.equal(after_combat[:, :half], macro[:, :half])
