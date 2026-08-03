from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from sts2_baseline import RolloutStep, SequenceUnroll
from sts2_rl.encoding import (
    EncodedDecisionSnapshot,
    GroundedEncodingConfig,
    GroundedObservationEncoder,
    grounding_encoding_identity,
)
from sts2_rl.encoding.snapshot import sparse_token_table
from sts2_rl.models import (
    GroundedCandidateBatch,
    GroundedCandidateConfig,
    RecurrentCandidateModel,
)
from sts2_rl.training.config import (
    FailureCreditConfig,
    OptimizationConfig,
    TransactionLearningConfig,
)
from sts2_rl.training.failure_credit import (
    CreditPlan,
    CreditProvenance,
    CyclePolicyCredit,
    DirectPolicyCredit,
    DirectPolicyTarget,
    EvidenceStratum,
    IdentityTriple,
    LearningContext,
    LearningStep,
    RiskSequenceCredit,
    ScalarCredit,
    SemanticKey,
)
from sts2_rl.training.learner import (
    VTraceLearner,
    compile_liveness_label_manifest,
    liveness_credit_losses,
)


def _model_config() -> GroundedCandidateConfig:
    return GroundedCandidateConfig(
        token_feature_dim=224,
        d_model=16,
        n_heads=4,
        ffn_dim=32,
        world_layers=1,
        latent_slots=2,
        latent_layers=1,
        local_layers=1,
        candidate_layers=1,
        recurrent_hidden_dim=32,
        dropout=0.0,
        domain_count=8,
        type_vocab_size=16,
        role_vocab_size=12,
        owner_vocab_size=16,
        entity_vocab_size=64,
        zone_vocab_size=10,
        order_vocab_size=16,
    )


def _snapshot(
    config: GroundedEncodingConfig,
    *,
    candidate_count: int,
) -> EncodedDecisionSnapshot:
    feature_dim = config.feature_dim
    world_feature = tuple([1.0] + [0.0] * (feature_dim - 1))
    candidate_features = tuple(
        tuple([0.0, float((index % 7) + 1) / 7.0] + [0.0] * (feature_dim - 2)) for index in range(candidate_count)
    )
    candidate_ids = tuple(
        (
            2 + index % 3,
            2 + index % 3,
            2,
            3 + index % 7,
            3 + index % 7,
            3 + index % 7,
            3 + index % 7,
            2,
            2,
            4,
            4,
            4,
            4,
        )
        for index in range(candidate_count)
    )
    action_mask = np.ones(candidate_count, dtype=np.bool_)
    if candidate_count > 1:
        action_mask[-1] = False
    return EncodedDecisionSnapshot(
        config=config,
        encoding_fingerprint=grounding_encoding_identity()["fingerprint_sha256"],
        world=sparse_token_table(
            features=(world_feature,),
            ids=((2, 2, 2, 2, 2, 2, 2, 2, 0),),
            feature_dim=feature_dim,
            id_width=9,
        ),
        candidates=sparse_token_table(
            features=candidate_features,
            ids=candidate_ids,
            feature_dim=feature_dim,
            id_width=13,
        ),
        locals=sparse_token_table(
            features=(),
            ids=(),
            feature_dim=feature_dim,
            id_width=9,
        ),
        local_offsets=np.zeros(candidate_count + 1, dtype=np.uint32),
        action_mask=action_mask,
        domain_id=1,
    )


def _semantic_key(namespace: str, value: str) -> SemanticKey:
    return SemanticKey.from_payload(
        namespace=namespace,
        schema_version="test-v1",
        payload={"value": value},
    )


def _identity(value: str) -> IdentityTriple:
    return IdentityTriple(
        exact=_semantic_key("exact", value),
        loop=_semantic_key("loop", value),
        comparison=_semantic_key("comparison", value),
    )


def _one_step_credit_plan(
    encoding_config: GroundedEncodingConfig,
    model_config: GroundedCandidateConfig,
    *,
    suffix: str,
    candidate_count: int = 3,
    forced: bool = False,
    policy_version: int = 3,
    include_direct: bool = True,
    include_risk: bool = True,
) -> CreditPlan:
    step = LearningStep(
        decision_id=f"decision-{suffix}",
        episode_step=12,
        snapshot=_snapshot(encoding_config, candidate_count=candidate_count),
        action_index=0,
        behavior_log_probability=-0.5,
        policy_version=policy_version,
        node=_identity(f"node-{suffix}"),
        anchor=_semantic_key("anchor", f"anchor-{suffix}"),
        candidate_actions=tuple(_identity(f"action-{suffix}-{index}") for index in range(candidate_count)),
        forced=forced,
    )
    context = LearningContext(
        context_id=f"context-{suffix}",
        episode_id=f"episode-{suffix}",
        start_step=12,
        initial_recurrent_state=np.zeros(
            model_config.recurrent_hidden_dim,
            dtype=np.float32,
        ),
        steps=(step,),
    )
    provenance = CreditProvenance(
        run_id="run-test",
        game_version="game-1",
        environment_schema_version="environment-1",
        identity_version="identity-1",
        detector_version="detector-1",
        adapter_version="adapter-1",
        collector_version="collector-1",
        policy_version=policy_version,
    )
    scalar = ScalarCredit(step_index=0, target=1.0, horizon=1)
    return CreditPlan(
        plan_id=f"plan-{suffix}",
        incident_id=f"incident-{suffix}",
        context=context,
        task_value_targets=(),
        task_q_targets=(),
        liveness_value_targets=(scalar,),
        liveness_q_targets=(scalar,),
        direct_policy_targets=(
            (
                DirectPolicyCredit(
                    step_index=0,
                    target=DirectPolicyTarget.AVOID,
                    witness_id=f"direct-{suffix}",
                ),
            )
            if include_direct
            else ()
        ),
        cycle_policy_targets=(),
        contrast_policy_targets=(),
        risk_sequences=(
            (
                RiskSequenceCredit(
                    step_indices=(0,),
                    terminal_cost=1.0,
                    discount=0.99,
                    witness_id=f"risk-{suffix}",
                ),
            )
            if include_risk
            else ()
        ),
        strata=(EvidenceStratum.DIRECT_WITNESS if include_direct else EvidenceStratum.RISK_SEQUENCE,),
        provenance=provenance,
    )


def test_candidate_liveness_cost_head_is_bounded_masked_and_active_shape() -> None:
    model_config = _model_config()
    encoding_config = GroundedEncodingConfig.from_model_config(
        model_config,
        max_world_tokens=64,
        max_candidates=256,
        max_candidate_local_tokens=4,
    )
    snapshot = _snapshot(encoding_config, candidate_count=111)
    batch = GroundedObservationEncoder(encoding_config).collate_snapshots((snapshot,))
    model = RecurrentCandidateModel(
        model_config,
        enable_transaction_heads=True,
        enable_liveness_head=True,
    ).eval()

    with torch.no_grad():
        output = model(batch)

    costs = output.candidate_liveness_cost_values
    state_cost = output.liveness_cost_value
    assert costs is not None
    assert state_cost is not None
    assert costs.shape == (1, 111)
    assert state_cost.shape == (1,)
    assert bool(((state_cost >= 0.0) & (state_cost <= 1.0)).all())
    assert bool(((costs[output.action_mask] >= 0.0) & (costs[output.action_mask] <= 1.0)).all())
    assert torch.count_nonzero(costs[~output.action_mask]) == 0
    output.validate(model_config)


def test_candidate_liveness_cost_head_is_candidate_equivariant() -> None:
    model_config = _model_config()
    encoding_config = GroundedEncodingConfig.from_model_config(
        model_config,
        max_world_tokens=32,
        max_candidates=256,
        max_candidate_local_tokens=4,
    )
    batch = GroundedObservationEncoder(encoding_config).collate_snapshots(
        (_snapshot(encoding_config, candidate_count=7),)
    )
    permutation = torch.tensor([4, 2, 6, 0, 5, 3, 1])
    model = RecurrentCandidateModel(
        model_config,
        enable_transaction_heads=True,
        enable_liveness_head=True,
    ).eval()

    with torch.no_grad():
        original = model(batch)
        permuted = model(batch.permute_candidates(permutation))

    assert original.candidate_liveness_cost_values is not None
    assert permuted.candidate_liveness_cost_values is not None
    torch.testing.assert_close(
        permuted.candidate_liveness_cost_values,
        original.candidate_liveness_cost_values[:, permutation],
        atol=2e-6,
        rtol=2e-6,
    )


def test_centered_liveness_actor_survives_saturated_task_value() -> None:
    policy_logits = torch.nn.Parameter(
        torch.tensor(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        )
    )
    cost_logits = torch.nn.Parameter(
        torch.tensor(
            [
                [2.0, -2.0],
                [2.0, -2.0],
                [2.0, -2.0],
            ]
        )
    )
    saturated_task_value = torch.nn.Parameter(torch.full((3,), -1.0))
    log_probabilities = torch.log_softmax(policy_logits, dim=1)
    costs = torch.sigmoid(cost_logits)
    losses = liveness_credit_losses(
        policy_log_probabilities=log_probabilities,
        candidate_liveness_cost_values=costs,
        action_mask=torch.ones((3, 2), dtype=torch.bool),
        selected_action_indices=torch.zeros(3, dtype=torch.long),
        risk_targets=torch.ones(3),
        risk_critic_mask=torch.tensor([True, True, True]),
        risk_actor_mask=torch.tensor([True, True, True]),
        forced_mask=torch.tensor([False, True, False]),
        censored_mask=torch.tensor([False, False, True]),
        direct_avoid_mask=torch.tensor([True, True, True]),
        risk_advantage_clip=0.25,
    )
    actor_objective = losses.risk_actor_loss + losses.direct_avoid_loss
    policy_gradient, task_value_gradient = torch.autograd.grad(
        actor_objective,
        (policy_logits, saturated_task_value),
        allow_unused=True,
    )

    assert policy_gradient is not None
    assert torch.count_nonzero(policy_gradient)
    assert task_value_gradient is None
    assert losses.risk_actor_labels == 1
    assert losses.direct_avoid_labels == 1
    assert losses.forced_actor_suppressed_labels == 2
    assert losses.censored_suppressed_labels == 3
    assert 0.0 < losses.centered_risk_max_abs <= 0.25


def test_liveness_critic_and_all_factual_policy_objectives_have_gradients() -> None:
    policy_logits = torch.nn.Parameter(
        torch.tensor(
            [
                [0.3, -0.3],
                [0.1, -0.1],
                [-0.2, 0.2],
                [0.0, 0.0],
            ]
        )
    )
    cost_logits = torch.nn.Parameter(
        torch.tensor(
            [
                [-1.0, 1.0],
                [1.0, -1.0],
                [0.5, -0.5],
                [-0.5, 0.5],
            ]
        )
    )
    state_cost_logits = torch.nn.Parameter(torch.tensor([-1.0, 1.0, 0.5, -0.5]))
    losses = liveness_credit_losses(
        policy_log_probabilities=torch.log_softmax(policy_logits, dim=1),
        candidate_liveness_cost_values=torch.sigmoid(cost_logits),
        liveness_cost_values=torch.sigmoid(state_cost_logits),
        action_mask=torch.ones((4, 2), dtype=torch.bool),
        selected_action_indices=torch.tensor([0, 0, 1, 1]),
        value_targets=torch.tensor([1.0, 1.0, 0.0, 0.0]),
        value_critic_mask=torch.ones(4, dtype=torch.bool),
        risk_targets=torch.tensor([1.0, 1.0, 0.0, 0.0]),
        risk_critic_mask=torch.ones(4, dtype=torch.bool),
        risk_actor_mask=torch.ones(4, dtype=torch.bool),
        direct_avoid_mask=torch.tensor([True, False, False, False]),
        completion_mask=torch.tensor([False, False, False, True]),
        cycle_groups=((0, 1),),
        contrast_pairs=((3, 1),),
        contrast_margin=0.1,
    )

    critic_gradient = torch.autograd.grad(
        losses.value_critic_loss + losses.critic_loss,
        (state_cost_logits, cost_logits),
        retain_graph=True,
    )
    policy_loss = (
        losses.risk_actor_loss
        + losses.direct_avoid_loss
        + losses.cycle_likelihood_loss
        + losses.contrast_loss
        + losses.completion_loss
    )
    policy_gradient = torch.autograd.grad(policy_loss, policy_logits)[0]

    assert torch.count_nonzero(critic_gradient[0])
    assert torch.count_nonzero(critic_gradient[1])
    assert torch.count_nonzero(policy_gradient)
    assert losses.value_labels == 4
    assert losses.critic_labels == 4
    assert losses.risk_actor_labels == 4
    assert losses.direct_avoid_labels == 1
    assert losses.cycle_labels == 1
    assert losses.contrast_labels == 1
    assert losses.completion_labels == 1


def test_liveness_credit_configuration_is_bounded() -> None:
    config = FailureCreditConfig(mode="learning")
    assert config.shadow_enabled
    assert config.learning_enabled
    assert config.liveness_value_critic_weight > 0.0
    assert config.liveness_cost_critic_weight > 0.0
    assert config.liveness_cost_actor_weight > 0.0
    assert 0.0 < config.liveness_risk_advantage_clip <= 1.0

    with pytest.raises(ValueError, match="liveness_risk_advantage_clip"):
        FailureCreditConfig(liveness_risk_advantage_clip=1.01)
    with pytest.raises(ValueError, match="quotas"):
        FailureCreditConfig(sample_records=3)
    with pytest.raises(ValueError, match="smaller"):
        FailureCreditConfig(
            burn_in_steps=32,
            maximum_context_steps=32,
        )
    with pytest.raises(ValueError, match="cannot precede"):
        FailureCreditConfig(
            liveness_head_calibration_updates=10,
            liveness_risk_actor_start_update=9,
        )
    with pytest.raises(ValueError, match="must be 1"):
        FailureCreditConfig(liveness_records_per_autograd_batch=2)

    shadow = FailureCreditConfig(mode="shadow")
    assert shadow.shadow_enabled
    assert not shadow.learning_enabled


def test_liveness_label_manifest_uses_learner_update_phase_and_exact_masks() -> None:
    model_config = _model_config()
    encoding_config = GroundedEncodingConfig.from_model_config(
        model_config,
        max_world_tokens=32,
        max_candidates=256,
        max_candidate_local_tokens=4,
    )
    config = FailureCreditConfig(
        mode="learning",
        liveness_head_calibration_updates=2,
        liveness_risk_actor_start_update=4,
    )
    plan = _one_step_credit_plan(
        encoding_config,
        model_config,
        suffix="phase",
    )

    calibration = compile_liveness_label_manifest(
        (plan,),
        config=config,
        current_policy_version=3,
        current_learner_update=0,
    )
    assert calibration.calibration_active
    assert not calibration.risk_actor_enabled
    assert calibration.risk_actor_phase_suppressed_labels == 1
    assert calibration.policy_lag_suppressed_labels == 0
    assert calibration.work.contexts == 1
    assert calibration.work.steps == 1
    assert calibration.work.candidates == 3
    assert calibration.work.autograd_segments == 1
    assert calibration.rows[0].direct_actor_mask
    assert calibration.rows[0].effective_direct_actor
    assert calibration.rows[0].risk_actor_requested
    assert not calibration.rows[0].risk_actor_mask

    calibrated = compile_liveness_label_manifest(
        (plan,),
        config=config,
        current_policy_version=3,
        current_learner_update=4,
    )
    assert not calibrated.calibration_active
    assert calibrated.risk_actor_enabled
    assert calibrated.risk_actor_phase_suppressed_labels == 0
    assert calibrated.rows[0].effective_risk_actor

    stale = compile_liveness_label_manifest(
        (plan,),
        config=config,
        current_policy_version=3 + config.policy_gradient_max_lag + 1,
        current_learner_update=4,
    )
    assert stale.policy_lag_suppressed_labels == 2
    assert not stale.rows[0].risk_actor_mask
    assert not stale.rows[0].direct_actor_mask

    forced = _one_step_credit_plan(
        encoding_config,
        model_config,
        suffix="forced",
        candidate_count=1,
        forced=True,
    )
    forced_manifest = compile_liveness_label_manifest(
        (forced,),
        config=config,
        current_policy_version=3,
        current_learner_update=4,
    )
    assert not forced_manifest.rows[0].risk_actor_mask
    assert not forced_manifest.rows[0].direct_actor_mask
    assert not forced_manifest.rows[0].effective_risk_actor
    assert not forced_manifest.rows[0].effective_direct_actor


def test_calibration_trains_fresh_heads_without_shared_trunk_gradient() -> None:
    model_config = _model_config()
    encoding_config = GroundedEncodingConfig.from_model_config(
        model_config,
        max_world_tokens=32,
        max_candidates=256,
        max_candidate_local_tokens=4,
    )
    config = FailureCreditConfig(
        mode="learning",
        liveness_head_calibration_updates=2,
        liveness_risk_actor_start_update=4,
    )
    model = RecurrentCandidateModel(
        model_config,
        enable_liveness_head=True,
    )
    learner = VTraceLearner(
        model=model,
        encoder=GroundedObservationEncoder(encoding_config),
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
        config=OptimizationConfig(),
        maximum_unroll_length=16,
        maximum_policy_lag=64,
        failure_credit_config=config,
    )
    plan = _one_step_credit_plan(
        encoding_config,
        model_config,
        suffix="critic-only",
        include_direct=False,
        include_risk=False,
    )

    model.zero_grad(set_to_none=True)
    calibration = learner.credit_plan_liveness_losses(
        (plan,),
        current_policy_version=3,
        current_learner_update=0,
    )
    (calibration.value_critic_loss + calibration.critic_loss).backward()  # type: ignore[no-untyped-call]
    liveness_parameters = {
        name
        for name, _ in model.named_parameters()
        if name.startswith(
            (
                "candidate_liveness_cost_head.",
                "liveness_cost_value_head.",
            )
        )
    }
    assert liveness_parameters
    assert any(
        parameter.grad is not None and bool(torch.count_nonzero(parameter.grad).item())
        for name, parameter in model.named_parameters()
        if name in liveness_parameters
    )
    assert all(
        parameter.grad is None or not bool(torch.count_nonzero(parameter.grad).item())
        for name, parameter in model.named_parameters()
        if name not in liveness_parameters
    )

    model.zero_grad(set_to_none=True)
    post_calibration = learner.credit_plan_liveness_losses(
        (plan,),
        current_policy_version=3,
        current_learner_update=2,
    )
    (post_calibration.value_critic_loss + post_calibration.critic_loss).backward()  # type: ignore[no-untyped-call]
    assert any(
        parameter.grad is not None and bool(torch.count_nonzero(parameter.grad).item())
        for name, parameter in model.named_parameters()
        if name not in liveness_parameters
    )


def test_liveness_manifest_rejects_candidate_work_budget_before_forward() -> None:
    model_config = _model_config()
    encoding_config = GroundedEncodingConfig.from_model_config(
        model_config,
        max_world_tokens=32,
        max_candidates=256,
        max_candidate_local_tokens=4,
    )
    plan = _one_step_credit_plan(
        encoding_config,
        model_config,
        suffix="budget",
    )
    with pytest.raises(ValueError, match="replayed_candidates"):
        compile_liveness_label_manifest(
            (plan,),
            config=FailureCreditConfig(
                mode="learning",
                liveness_maximum_replayed_candidates_per_update=2,
            ),
            current_policy_version=3,
            current_learner_update=0,
        )


def test_future_context_step_fails_closed_before_any_model_forward() -> None:
    model_config = _model_config()
    encoding_config = GroundedEncodingConfig.from_model_config(
        model_config,
        max_world_tokens=32,
        max_candidates=256,
        max_candidate_local_tokens=4,
    )
    config = FailureCreditConfig(mode="learning")
    base = _one_step_credit_plan(
        encoding_config,
        model_config,
        suffix="future-step",
        policy_version=3,
    )
    future_context = replace(
        base.context,
        steps=(replace(base.context.steps[0], policy_version=4),),
    )
    plan = replace(base, context=future_context)
    model = RecurrentCandidateModel(
        model_config,
        enable_liveness_head=True,
    )
    learner = VTraceLearner(
        model=model,
        encoder=GroundedObservationEncoder(encoding_config),
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
        config=OptimizationConfig(),
        maximum_unroll_length=16,
        maximum_policy_lag=64,
        failure_credit_config=config,
    )
    forward_calls = 0

    def count_forward(
        _module: torch.nn.Module,
        _args: tuple[object, ...],
    ) -> None:
        nonlocal forward_calls
        forward_calls += 1

    handle = model.register_forward_pre_hook(count_forward)
    try:
        with pytest.raises(ValueError, match="step is newer"):
            learner.credit_plan_liveness_losses(
                (plan,),
                current_policy_version=3,
                current_learner_update=0,
            )
    finally:
        handle.remove()
    assert forward_calls == 0


def test_forced_only_cycle_is_excluded_by_manifest_and_tensor_loss() -> None:
    model_config = _model_config()
    encoding_config = GroundedEncodingConfig.from_model_config(
        model_config,
        max_world_tokens=32,
        max_candidates=256,
        max_candidate_local_tokens=4,
    )
    config = FailureCreditConfig(
        mode="learning",
        liveness_head_calibration_updates=0,
        liveness_risk_actor_start_update=0,
    )
    base = _one_step_credit_plan(
        encoding_config,
        model_config,
        suffix="forced-cycle",
        candidate_count=1,
        forced=True,
        include_direct=False,
        include_risk=False,
    )
    plan = replace(
        base,
        cycle_policy_targets=(
            CyclePolicyCredit(
                step_indices=(0,),
                behavior_mean_log_probability=-0.5,
                margin=0.1,
                witness_id="forced-cycle",
            ),
        ),
        strata=(EvidenceStratum.MULTI_EDGE_CYCLE,),
    )
    manifest = compile_liveness_label_manifest(
        (plan,),
        config=config,
        current_policy_version=3,
        current_learner_update=0,
    )
    assert len(manifest.cycle_groups) == 1
    assert manifest.cycle_groups[0].fresh
    assert not manifest.cycle_groups[0].effective

    model = RecurrentCandidateModel(
        model_config,
        enable_liveness_head=True,
    )
    learner = VTraceLearner(
        model=model,
        encoder=GroundedObservationEncoder(encoding_config),
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
        config=OptimizationConfig(),
        maximum_unroll_length=16,
        maximum_policy_lag=64,
        failure_credit_config=config,
    )
    losses = learner.credit_plan_liveness_losses(
        (plan,),
        current_policy_version=3,
        current_learner_update=0,
    )
    assert losses.cycle_labels == 0
    assert losses.cycle_likelihood_loss.detach().item() == 0.0


def test_manifest_matches_rowwise_direct_risk_and_atomic_cycle_freshness() -> None:
    model_config = _model_config()
    encoding_config = GroundedEncodingConfig.from_model_config(
        model_config,
        max_world_tokens=32,
        max_candidates=256,
        max_candidate_local_tokens=4,
    )
    base = _one_step_credit_plan(
        encoding_config,
        model_config,
        suffix="mixed-age",
        policy_version=0,
    )
    first = base.context.steps[0]
    second = replace(
        first,
        decision_id="decision-mixed-age-fresh",
        episode_step=first.episode_step + 1,
        policy_version=3,
        node=_identity("node-mixed-age-fresh"),
        anchor=_semantic_key("anchor", "anchor-mixed-age-fresh"),
        candidate_actions=tuple(
            _identity(f"action-mixed-age-fresh-{index}")
            for index in range(first.snapshot.candidate_count)
        ),
    )
    context = replace(base.context, steps=(first, second))
    plan = replace(
        base,
        context=context,
        direct_policy_targets=(
            DirectPolicyCredit(
                step_index=0,
                target=DirectPolicyTarget.AVOID,
                witness_id="direct-mixed-age-stale",
            ),
            DirectPolicyCredit(
                step_index=1,
                target=DirectPolicyTarget.AVOID,
                witness_id="direct-mixed-age-fresh",
            ),
        ),
        cycle_policy_targets=(
            CyclePolicyCredit(
                step_indices=(0, 1),
                behavior_mean_log_probability=-0.5,
                margin=0.1,
                witness_id="cycle-mixed-age",
            ),
        ),
        risk_sequences=(
            RiskSequenceCredit(
                step_indices=(0, 1),
                terminal_cost=1.0,
                discount=0.99,
                witness_id="risk-mixed-age",
            ),
        ),
        strata=(
            EvidenceStratum.DIRECT_WITNESS,
            EvidenceStratum.MULTI_EDGE_CYCLE,
            EvidenceStratum.RISK_SEQUENCE,
        ),
    )
    manifest = compile_liveness_label_manifest(
        (plan,),
        config=FailureCreditConfig(
            mode="learning",
            policy_gradient_max_lag=1,
            liveness_head_calibration_updates=0,
            liveness_risk_actor_start_update=0,
        ),
        current_policy_version=3,
        current_learner_update=0,
    )
    rows = {row.decision_id: row for row in manifest.rows}
    stale = rows[first.decision_id]
    fresh = rows[second.decision_id]
    assert not stale.fresh
    assert not stale.direct_actor_mask
    assert not stale.risk_actor_mask
    assert fresh.fresh
    assert fresh.direct_actor_mask
    assert fresh.risk_actor_mask
    assert len(manifest.cycle_groups) == 1
    assert not manifest.cycle_groups[0].fresh
    assert not manifest.cycle_groups[0].effective


def test_liveness_recurrent_replay_detaches_each_tbptt_window() -> None:
    model_config = _model_config()
    encoding_config = GroundedEncodingConfig.from_model_config(
        model_config,
        max_world_tokens=32,
        max_candidates=256,
        max_candidate_local_tokens=4,
    )
    config = FailureCreditConfig(
        mode="learning",
        liveness_head_calibration_updates=0,
        liveness_risk_actor_start_update=0,
        liveness_tbptt_window_steps=2,
    )
    base = _one_step_credit_plan(
        encoding_config,
        model_config,
        suffix="tbptt",
        include_direct=False,
        include_risk=False,
    )
    base_step = base.context.steps[0]
    steps = tuple(
        replace(
            base_step,
            decision_id=f"tbptt-decision-{index}",
            episode_step=12 + index,
            node=_identity(f"tbptt-node-{index}"),
            anchor=_semantic_key("anchor", f"tbptt-anchor-{index}"),
            candidate_actions=tuple(
                _identity(f"tbptt-action-{index}-{candidate}")
                for candidate in range(base_step.snapshot.candidate_count)
            ),
        )
        for index in range(6)
    )
    context = LearningContext(
        context_id="tbptt-context",
        episode_id="tbptt-episode",
        start_step=12,
        initial_recurrent_state=np.zeros(
            model_config.recurrent_hidden_dim,
            dtype=np.float32,
        ),
        steps=steps,
        burn_in_steps=1,
    )
    scalar = ScalarCredit(step_index=5, target=1.0, horizon=1)
    plan = replace(
        base,
        context=context,
        liveness_value_targets=(scalar,),
        liveness_q_targets=(scalar,),
    )
    model = RecurrentCandidateModel(
        model_config,
        enable_liveness_head=True,
    )
    learner = VTraceLearner(
        model=model,
        encoder=GroundedObservationEncoder(encoding_config),
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
        config=OptimizationConfig(),
        maximum_unroll_length=16,
        maximum_policy_lag=64,
        failure_credit_config=config,
    )
    trainable_hidden_is_detached: list[bool] = []

    def observe_hidden(
        _module: torch.nn.Module,
        args: tuple[object, ...],
        _kwargs: dict[str, object],
    ) -> None:
        if torch.is_grad_enabled():
            hidden = args[1]
            assert isinstance(hidden, torch.Tensor)
            trainable_hidden_is_detached.append(hidden.grad_fn is None)

    handle = model.register_forward_pre_hook(observe_hidden, with_kwargs=True)
    try:
        losses = learner.credit_plan_liveness_losses(
            (plan,),
            current_policy_version=3,
            current_learner_update=0,
        )
    finally:
        handle.remove()

    assert trainable_hidden_is_detached == [True, False, True, False, True]
    assert losses.replayed_steps == 6
    assert losses.autograd_segments == 3


def test_update_backpropagates_failure_credit_one_record_at_a_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_config = _model_config()
    encoding_config = GroundedEncodingConfig.from_model_config(
        model_config,
        max_world_tokens=32,
        max_candidates=256,
        max_candidate_local_tokens=4,
    )
    config = FailureCreditConfig(
        mode="learning",
        liveness_head_calibration_updates=0,
        liveness_risk_actor_start_update=0,
    )
    model = RecurrentCandidateModel(
        model_config,
        enable_liveness_head=True,
    )
    learner = VTraceLearner(
        model=model,
        encoder=GroundedObservationEncoder(encoding_config),
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
        config=OptimizationConfig(),
        maximum_unroll_length=16,
        maximum_policy_lag=64,
        failure_credit_config=config,
    )
    plans = (
        _one_step_credit_plan(
            encoding_config,
            model_config,
            suffix="microbatch-a",
        ),
        _one_step_credit_plan(
            encoding_config,
            model_config,
            suffix="microbatch-b",
        ),
    )
    snapshot = plans[0].context.steps[0].snapshot
    unroll = SequenceUnroll(
        episode_id="microbatch-unroll",
        start_step=0,
        policy_version=3,
        initial_recurrent_state=np.zeros(
            model_config.recurrent_hidden_dim,
            dtype=np.float32,
        ),
        steps=(
            RolloutStep(
                snapshot=snapshot,
                action_index=0,
                behavior_log_probability=-0.5,
                reward=-1.0,
                discount=0.0,
                policy_decision=True,
            ),
        ),
        bootstrap_snapshot=None,
    )
    calls: list[int] = []
    original = learner.credit_plan_liveness_losses

    def observe_microbatch(
        credit_plans: tuple[CreditPlan, ...],
        *,
        current_policy_version: int,
        current_learner_update: int,
    ):
        calls.append(len(credit_plans))
        return original(
            credit_plans,
            current_policy_version=current_policy_version,
            current_learner_update=current_learner_update,
        )

    monkeypatch.setattr(
        learner,
        "credit_plan_liveness_losses",
        observe_microbatch,
    )
    progress: list[tuple[str, dict[str, int | float]]] = []
    metrics = learner.update(
        (unroll,),
        current_policy_version=3,
        current_learner_update=0,
        schedule_policy_version=515,
        schedule_learner_update=512,
        credit_plans=plans,
        progress=lambda stage, payload: progress.append((stage, payload)),
    )

    assert calls == [1, 1]
    record_starts = [payload for stage, payload in progress if stage == "liveness_record_start"]
    assert [payload["liveness_record_index"] for payload in record_starts] == [
        0,
        1,
    ]
    assert all(payload["liveness_record_steps"] == 1 for payload in record_starts)
    assert all(payload["liveness_record_candidates"] == 3 for payload in record_starts)
    assert metrics.liveness_autograd_microbatches == 2
    assert metrics.liveness_head_calibration_active == 0
    assert metrics.liveness_risk_actor_enabled == 1
    assert metrics.liveness_credit_plans == 2
    assert metrics.liveness_replayed_contexts == 2


def test_typed_credit_plan_replays_state_and_candidate_liveness_targets() -> None:
    model_config = _model_config()
    encoding_config = GroundedEncodingConfig.from_model_config(
        model_config,
        max_world_tokens=32,
        max_candidates=256,
        max_candidate_local_tokens=4,
    )
    encoder = GroundedObservationEncoder(encoding_config)
    model = RecurrentCandidateModel(
        model_config,
        enable_transaction_heads=False,
        enable_liveness_head=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    learner = VTraceLearner(
        model=model,
        encoder=encoder,
        optimizer=optimizer,
        config=OptimizationConfig(),
        maximum_unroll_length=16,
        maximum_policy_lag=64,
        transaction_config=TransactionLearningConfig(enabled=False),
        failure_credit_config=FailureCreditConfig(mode="learning"),
    )
    snapshot = _snapshot(encoding_config, candidate_count=3)
    step = LearningStep(
        decision_id="decision-1",
        episode_step=12,
        snapshot=snapshot,
        action_index=0,
        behavior_log_probability=-0.5,
        policy_version=3,
        node=_identity("node-1"),
        anchor=_semantic_key("anchor", "anchor-1"),
        candidate_actions=(
            _identity("action-0"),
            _identity("action-1"),
            _identity("action-disabled"),
        ),
        forced=False,
    )
    context = LearningContext(
        context_id="context-1",
        episode_id="episode-1",
        start_step=12,
        initial_recurrent_state=np.zeros(
            model_config.recurrent_hidden_dim,
            dtype=np.float32,
        ),
        steps=(step,),
    )
    provenance = CreditProvenance(
        run_id="run-1",
        game_version="game-1",
        environment_schema_version="environment-1",
        identity_version="identity-1",
        detector_version="detector-1",
        adapter_version="adapter-1",
        collector_version="collector-1",
        policy_version=3,
    )
    value_target = ScalarCredit(step_index=0, target=1.0, horizon=1)
    q_target = ScalarCredit(step_index=0, target=1.0, horizon=1)
    plan = CreditPlan(
        plan_id="plan-1",
        incident_id="incident-1",
        context=context,
        task_value_targets=(),
        task_q_targets=(),
        liveness_value_targets=(value_target,),
        liveness_q_targets=(q_target,),
        direct_policy_targets=(
            DirectPolicyCredit(
                step_index=0,
                target=DirectPolicyTarget.AVOID,
                witness_id="witness-1",
            ),
        ),
        cycle_policy_targets=(),
        contrast_policy_targets=(),
        risk_sequences=(
            RiskSequenceCredit(
                step_indices=(0,),
                terminal_cost=1.0,
                discount=0.99,
                witness_id="witness-risk",
            ),
        ),
        strata=(
            EvidenceStratum.DIRECT_WITNESS,
            EvidenceStratum.RISK_SEQUENCE,
        ),
        provenance=provenance,
    )

    losses = learner.credit_plan_liveness_losses(
        (plan,),
        current_policy_version=3,
        current_learner_update=512,
    )
    total = losses.value_critic_loss + losses.critic_loss + losses.risk_actor_loss + losses.direct_avoid_loss
    total.backward()  # type: ignore[no-untyped-call]

    assert losses.value_labels == 1
    assert losses.critic_labels == 1
    assert losses.risk_actor_labels == 1
    assert losses.direct_avoid_labels == 1
    assert model.liveness_cost_value_head is not None
    assert model.candidate_liveness_cost_head is not None
    assert any(
        parameter.grad is not None and bool(torch.count_nonzero(parameter.grad).item())
        for parameter in model.liveness_cost_value_head.parameters()
    )
    assert any(
        parameter.grad is not None and bool(torch.count_nonzero(parameter.grad).item())
        for parameter in model.candidate_liveness_cost_head.parameters()
    )

    stale = learner.credit_plan_liveness_losses(
        (plan,),
        current_policy_version=200,
        current_learner_update=512,
    )
    assert stale.value_labels == 1
    assert stale.critic_labels == 1
    assert stale.risk_actor_labels == 0
    assert stale.direct_avoid_labels == 0
    assert stale.policy_lag_suppressed_labels == 2


def test_credit_plan_replay_batches_contexts_by_referenced_timestep() -> None:
    model_config = _model_config()
    encoding_config = GroundedEncodingConfig.from_model_config(
        model_config,
        max_world_tokens=32,
        max_candidates=256,
        max_candidate_local_tokens=4,
    )
    encoder = GroundedObservationEncoder(encoding_config)
    model = RecurrentCandidateModel(
        model_config,
        enable_transaction_heads=False,
        enable_liveness_head=True,
    )
    learner = VTraceLearner(
        model=model,
        encoder=encoder,
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
        config=OptimizationConfig(),
        maximum_unroll_length=16,
        maximum_policy_lag=64,
        transaction_config=TransactionLearningConfig(enabled=False),
        failure_credit_config=FailureCreditConfig(mode="learning"),
    )
    provenance = CreditProvenance(
        run_id="run-batched-replay",
        game_version="game-1",
        environment_schema_version="environment-1",
        identity_version="identity-1",
        detector_version="detector-1",
        adapter_version="adapter-1",
        collector_version="collector-1",
        policy_version=3,
    )

    def learning_step(
        prefix: str,
        *,
        episode_step: int,
        candidate_count: int,
    ) -> LearningStep:
        snapshot = _snapshot(
            encoding_config,
            candidate_count=candidate_count,
        )
        return LearningStep(
            decision_id=f"{prefix}-decision-{episode_step}",
            episode_step=episode_step,
            snapshot=snapshot,
            action_index=0,
            behavior_log_probability=-0.5,
            policy_version=3,
            node=_identity(f"{prefix}-node-{episode_step}"),
            anchor=_semantic_key(
                "anchor",
                f"{prefix}-anchor-{episode_step}",
            ),
            candidate_actions=tuple(
                _identity(f"{prefix}-action-{episode_step}-{index}") for index in range(candidate_count)
            ),
            forced=False,
        )

    context_a = LearningContext(
        context_id="batched-context-a",
        episode_id="batched-episode-a",
        start_step=100,
        initial_recurrent_state=np.zeros(
            model_config.recurrent_hidden_dim,
            dtype=np.float32,
        ),
        steps=tuple(
            learning_step(
                "a",
                episode_step=100 + index,
                candidate_count=candidate_count,
            )
            for index, candidate_count in enumerate((3, 5, 4, 97))
        ),
        burn_in_steps=1,
    )
    context_b = LearningContext(
        context_id="batched-context-b",
        episode_id="batched-episode-b",
        start_step=200,
        initial_recurrent_state=np.full(
            model_config.recurrent_hidden_dim,
            0.125,
            dtype=np.float32,
        ),
        steps=tuple(
            learning_step(
                "b",
                episode_step=200 + index,
                candidate_count=candidate_count,
            )
            for index, candidate_count in enumerate((6, 3, 111, 109))
        ),
        burn_in_steps=1,
    )

    def plan(
        suffix: str,
        context: LearningContext,
        *,
        target_step: int,
    ) -> CreditPlan:
        scalar = ScalarCredit(
            step_index=target_step,
            target=1.0,
            horizon=1,
        )
        return CreditPlan(
            plan_id=f"batched-plan-{suffix}",
            incident_id=f"batched-incident-{suffix}",
            context=context,
            task_value_targets=(),
            task_q_targets=(),
            liveness_value_targets=(scalar,),
            liveness_q_targets=(scalar,),
            direct_policy_targets=(
                DirectPolicyCredit(
                    step_index=target_step,
                    target=DirectPolicyTarget.AVOID,
                    witness_id=f"batched-witness-{suffix}",
                ),
            ),
            cycle_policy_targets=(),
            contrast_policy_targets=(),
            risk_sequences=(
                RiskSequenceCredit(
                    step_indices=(target_step,),
                    terminal_cost=1.0,
                    discount=0.99,
                    witness_id=f"batched-risk-{suffix}",
                ),
            ),
            strata=(
                EvidenceStratum.DIRECT_WITNESS,
                EvidenceStratum.RISK_SEQUENCE,
            ),
            provenance=provenance,
        )

    plans = (
        plan("a", context_a, target_step=2),
        plan("b", context_b, target_step=1),
    )
    forward_calls: list[tuple[int, int, bool]] = []

    def observe_forward(
        _module: torch.nn.Module,
        args: tuple[object, ...],
    ) -> None:
        batch = args[0]
        assert isinstance(batch, GroundedCandidateBatch)
        forward_calls.append(
            (
                int(batch.domain_ids.shape[0]),
                int(batch.candidates.action_mask.shape[1]),
                torch.is_grad_enabled(),
            )
        )

    handle = model.register_forward_pre_hook(observe_forward)
    try:
        batched = learner.credit_plan_liveness_losses(
            plans,
            current_policy_version=3,
            current_learner_update=512,
        )
    finally:
        handle.remove()

    # Both contexts share the burn-in call at t=0 and the train call at t=1;
    # only context A remains active at t=2.  The unreferenced 97/111/109
    # candidate suffixes are never collated or forwarded.
    assert forward_calls == [
        (2, 6, False),
        (2, 5, True),
        (1, 4, True),
    ]
    assert len(forward_calls) == 3
    assert len(forward_calls) < 3 + 2  # sum of referenced context prefixes

    serial = tuple(
        learner.credit_plan_liveness_losses(
            (item,),
            current_policy_version=3,
            current_learner_update=512,
        )
        for item in plans
    )
    for field in (
        "value_critic_loss",
        "critic_loss",
        "risk_actor_loss",
        "direct_avoid_loss",
    ):
        expected = torch.stack(tuple(getattr(losses, field) for losses in serial)).mean()
        torch.testing.assert_close(
            getattr(batched, field),
            expected,
            atol=2e-6,
            rtol=2e-6,
        )

    assert batched.value_labels == 2
    assert batched.critic_labels == 2
    assert batched.risk_actor_labels == 2
    assert batched.direct_avoid_labels == 2
    (batched.value_critic_loss + batched.critic_loss + batched.risk_actor_loss + batched.direct_avoid_loss).backward()  # type: ignore[no-untyped-call]
    assert model.liveness_cost_value_head is not None
    assert model.candidate_liveness_cost_head is not None
    assert any(
        parameter.grad is not None and bool(torch.count_nonzero(parameter.grad).item())
        for parameter in model.liveness_cost_value_head.parameters()
    )
    assert any(
        parameter.grad is not None and bool(torch.count_nonzero(parameter.grad).item())
        for parameter in model.candidate_liveness_cost_head.parameters()
    )
    assert bool(torch.count_nonzero(model.combat_recurrent_cell.weight_hh.grad).item())
