from __future__ import annotations

import pickle
from dataclasses import replace

import numpy as np
import pytest

from sts2_rl.encoding import EncodedDecisionSnapshot, GroundedEncodingConfig
from sts2_rl.encoding.snapshot import sparse_token_table
from sts2_rl.semantics import SemanticCollisionError
from sts2_rl.training.failure_credit import (
    FAILURE_EVIDENCE_REPLAY_VERSION,
    BoundedFailureCreditReplay,
    CreditCompilationError,
    CreditCompiler,
    CreditProvenance,
    DirectPolicyTarget,
    EvidenceRecord,
    EvidenceStratum,
    FailureIncident,
    FailureOutcome,
    IdentityTriple,
    ImmutableEvidenceCorpus,
    LearningContext,
    LearningStep,
    LoopEdgeEvidence,
    MatchedOutcomePair,
    OutcomeArm,
    OutcomePairMatcher,
    PolicyWitness,
    SemanticKey,
    StratumQuota,
    TargetAuthority,
    WitnessKind,
)


def _semantic(namespace: str, identity: str) -> SemanticKey:
    return SemanticKey.from_payload(
        namespace=namespace,
        schema_version="test-semantics-v1",
        payload={"identity": identity},
    )


def _identity(
    kind: str,
    identity: str,
    *,
    exact_identity: str | None = None,
    loop_identity: str | None = None,
    comparison_identity: str | None = None,
) -> IdentityTriple:
    return IdentityTriple(
        exact=_semantic(f"{kind}.exact", exact_identity or identity),
        loop=_semantic(f"{kind}.loop", loop_identity or identity),
        comparison=_semantic(
            f"{kind}.comparison",
            comparison_identity or identity,
        ),
    )


def _snapshot(*, candidate_count: int = 2) -> EncodedDecisionSnapshot:
    config = GroundedEncodingConfig(
        max_world_tokens=4,
        max_candidates=4,
        max_candidate_local_tokens=2,
    )
    feature_dim = config.feature_dim
    world = tuple([1.0] + [0.0] * (feature_dim - 1))
    candidate_features = tuple(
        tuple([0.0] * (index + 1) + [1.0] + [0.0] * (feature_dim - index - 2)) for index in range(candidate_count)
    )
    candidate_ids = tuple((2 + index, 2, 2, 3, 3, 3, 3, 2, 2, 4, 4, 4, 4) for index in range(candidate_count))
    return EncodedDecisionSnapshot(
        config=config,
        encoding_fingerprint="0" * 64,
        world=sparse_token_table(
            features=(world,),
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
        action_mask=np.ones(candidate_count, dtype=np.bool_),
        domain_id=1,
    )


def _provenance(*, policy_version: int = 7) -> CreditProvenance:
    return CreditProvenance(
        run_id="run-v4",
        game_version="game-v1",
        environment_schema_version="environment-v1",
        identity_version="identity-v1",
        detector_version="detector-v1",
        adapter_version="adapter-v1",
        collector_version="collector-v1",
        policy_version=policy_version,
    )


def _context(
    *,
    context_id: str,
    start_step: int = 0,
    count: int = 1,
    action_index: int = 0,
    candidate_count: int = 2,
    node_comparison: str = "comparable",
    node_loop_identities: tuple[str, ...] | None = None,
    burn_in_steps: int = 0,
) -> LearningContext:
    snapshot = _snapshot(candidate_count=candidate_count)
    candidates = tuple(
        _identity(
            "action",
            f"{context_id}:action-instance-{index}",
            loop_identity=f"action-{index}",
            comparison_identity=f"action-{index}",
        )
        for index in range(candidate_count)
    )
    if node_loop_identities is None:
        node_loop_identities = ("loop-node",) * count
    if len(node_loop_identities) != count:
        raise ValueError("test node_loop_identities must align with context steps")
    steps = tuple(
        LearningStep(
            decision_id=f"{context_id}:decision-{offset}",
            episode_step=start_step + offset,
            snapshot=snapshot,
            action_index=action_index,
            behavior_log_probability=0.0 if candidate_count == 1 else -0.693147,
            policy_version=7,
            node=_identity(
                "node",
                f"{context_id}:node-{start_step + offset}",
                loop_identity=node_loop_identities[offset],
                comparison_identity=node_comparison,
            ),
            anchor=_semantic("anchor", "loop-anchor"),
            candidate_actions=candidates,
            forced=candidate_count == 1,
        )
        for offset in range(count)
    )
    return LearningContext(
        context_id=context_id,
        episode_id=f"episode-{context_id}",
        start_step=start_step,
        initial_recurrent_state=np.zeros(8, dtype=np.float32),
        steps=steps,
        burn_in_steps=burn_in_steps,
    )


def _loop_edge(
    context: LearningContext,
    *,
    step_index: int,
    supporting_episode_steps: tuple[int, ...],
) -> LoopEdgeEvidence:
    step = context.steps[step_index]
    return LoopEdgeEvidence(
        node=step.node.loop,
        action=step.selected_action.loop,
        supporting_episode_steps=supporting_episode_steps,
    )


def _failed_incident(
    context: LearningContext,
    *,
    incident_id: str,
    outcome: FailureOutcome = FailureOutcome.DEADLOCK_CYCLE,
    witnesses: tuple[PolicyWitness, ...] = (),
    detector_window_steps: int = 64,
) -> FailureIncident:
    return FailureIncident(
        incident_id=incident_id,
        scope_key="scope",
        failure_kind=outcome.value,
        outcome=outcome,
        task_authority=TargetAuthority.OBJECTIVE_CONFIRMED,
        local_authority=TargetAuthority.DETECTOR_CONFIRMED,
        task_return=-1.0,
        local_failure_cost=1.0,
        context=context,
        witnesses=witnesses,
        detector_window_steps=detector_window_steps,
        progress_epoch=3,
        provenance=_provenance(),
    )


def _completed_incident(
    context: LearningContext,
    *,
    incident_id: str,
    witnesses: tuple[PolicyWitness, ...],
) -> FailureIncident:
    return FailureIncident(
        incident_id=incident_id,
        scope_key="scope",
        failure_kind="completed",
        outcome=FailureOutcome.COMPLETED,
        task_authority=TargetAuthority.CENSORED,
        local_authority=TargetAuthority.VERIFIED_TRANSITION,
        task_return=None,
        local_failure_cost=0.0,
        context=context,
        witnesses=witnesses,
        detector_window_steps=64,
        progress_epoch=3,
        provenance=_provenance(),
    )


def _direct_failure_record(
    context: LearningContext,
    *,
    incident_id: str,
) -> EvidenceRecord:
    step = context.steps[context.burn_in_steps]
    witness = PolicyWitness(
        witness_id=f"direct-{incident_id}",
        kind=WitnessKind.DIRECT_WITNESS,
        attributed_step_indices=(context.burn_in_steps,),
        supporting_episode_steps=(step.episode_step, step.episode_step + 1),
        occurrences=2,
        cycle_span=1,
        successor_confirmed=True,
        loop_edges=(
            _loop_edge(
                context,
                step_index=context.burn_in_steps,
                supporting_episode_steps=(
                    step.episode_step,
                    step.episode_step + 1,
                ),
            ),
        ),
    )
    incident = _failed_incident(
        context,
        incident_id=incident_id,
        witnesses=(witness,),
    )
    return EvidenceRecord(
        incident=incident,
        plan=CreditCompiler().compile(incident),
    )


def _completion_record(
    context: LearningContext,
    *,
    incident_id: str,
) -> EvidenceRecord:
    incident = _completed_incident(
        context,
        incident_id=incident_id,
        witnesses=(),
    )
    return EvidenceRecord(
        incident=incident,
        plan=CreditCompiler().compile(incident),
    )


def test_span_40_witness_survives_a_32_step_learning_tail() -> None:
    context = _context(context_id="span40", start_step=9, count=32)
    witness = PolicyWitness(
        witness_id="witness-span40",
        kind=WitnessKind.DIRECT_WITNESS,
        attributed_step_indices=(31,),
        # The detector observed the first occurrence before the retained
        # learning context.  The compiler must consume this evidence directly.
        supporting_episode_steps=(0, 40),
        occurrences=2,
        cycle_span=40,
        successor_confirmed=True,
        loop_edges=(
            _loop_edge(
                context,
                step_index=31,
                supporting_episode_steps=(0, 40),
            ),
        ),
    )
    plan = CreditCompiler().compile(
        _failed_incident(
            context,
            incident_id="incident-span40",
            witnesses=(witness,),
            detector_window_steps=64,
        )
    )

    assert plan.direct_policy_targets == (plan.direct_policy_targets[0],)
    assert plan.direct_policy_targets[0].step_index == 31
    assert plan.direct_policy_targets[0].target is DirectPolicyTarget.AVOID
    assert EvidenceStratum.DIRECT_WITNESS in plan.strata


def test_failure_credit_rejects_corrupt_semantic_identity_payload_binding() -> None:
    context = _context(context_id="corrupt-identity")
    original = context.steps[0].node.exact
    corrupt = SemanticKey(
        namespace=original.namespace,
        schema_version=original.schema_version,
        canonical_payload=original.canonical_payload,
        digest="0" * 64,
    )

    with pytest.raises(SemanticCollisionError, match="does not match"):
        replace(
            context.steps[0],
            node=IdentityTriple(
                exact=corrupt,
                loop=context.steps[0].node.loop,
                comparison=context.steps[0].node.comparison,
            ),
        )


def test_compiler_rejects_witness_edge_not_observed_for_attributed_step() -> None:
    context = _context(context_id="attributed")
    unrelated = _context(
        context_id="unrelated",
        node_loop_identities=("different-loop-node",),
    )
    witness = PolicyWitness(
        witness_id="mismatched-detector-edge",
        kind=WitnessKind.DIRECT_WITNESS,
        attributed_step_indices=(0,),
        supporting_episode_steps=(0, 4),
        occurrences=2,
        cycle_span=4,
        successor_confirmed=True,
        loop_edges=(
            _loop_edge(
                unrelated,
                step_index=0,
                supporting_episode_steps=(0, 4),
            ),
        ),
    )

    with pytest.raises(CreditCompilationError, match="does not name"):
        CreditCompiler().compile(
            _failed_incident(
                context,
                incident_id="mismatched-incident",
                witnesses=(witness,),
            )
        )


def test_forced_cycle_witness_never_produces_an_actor_label() -> None:
    context = _context(
        context_id="forced",
        candidate_count=1,
    )
    witness = PolicyWitness(
        witness_id="forced-witness",
        kind=WitnessKind.DIRECT_WITNESS,
        attributed_step_indices=(0,),
        supporting_episode_steps=(0, 4),
        occurrences=2,
        cycle_span=4,
        successor_confirmed=True,
        loop_edges=(
            _loop_edge(
                context,
                step_index=0,
                supporting_episode_steps=(0, 4),
            ),
        ),
    )
    plan = CreditCompiler().compile(
        _failed_incident(
            context,
            incident_id="forced-incident",
            witnesses=(witness,),
        )
    )

    assert plan.direct_policy_targets == ()
    assert plan.cycle_policy_targets == ()
    assert EvidenceStratum.DIRECT_WITNESS not in plan.strata
    assert len(plan.liveness_q_targets) == 1


def test_unique_stall_has_risk_and_q_credit_without_last_step_blame() -> None:
    context = _context(context_id="unique-stall", count=6)
    incident = _failed_incident(
        context,
        incident_id="unique-stall-incident",
        outcome=FailureOutcome.DEADLOCK_STALL,
        witnesses=(),
        detector_window_steps=256,
    )
    plan = CreditCompiler().compile(incident)

    assert len(plan.task_q_targets) == 6
    assert len(plan.liveness_value_targets) == 6
    assert len(plan.liveness_q_targets) == 6
    assert len(plan.risk_sequences) == 1
    assert plan.risk_sequences[0].step_indices == tuple(range(6))
    assert plan.direct_policy_targets == ()
    assert plan.cycle_policy_targets == ()
    assert plan.contrast_policy_targets == ()
    assert EvidenceStratum.UNRESOLVED_STALL in plan.strata


def test_completed_and_multi_edge_cycle_compile_scope_specific_actor_credit() -> None:
    completed_context = _context(context_id="completed", count=2)
    completion = PolicyWitness(
        witness_id="completion-control",
        kind=WitnessKind.COMPLETION_CONTROL,
        attributed_step_indices=(1,),
        supporting_episode_steps=(0, 1),
        occurrences=1,
        cycle_span=None,
        successor_confirmed=True,
    )
    completed_plan = CreditCompiler().compile(
        _completed_incident(
            completed_context,
            incident_id="completed-incident",
            witnesses=(completion,),
        )
    )
    assert completed_plan.direct_policy_targets[0].target is DirectPolicyTarget.PREFER
    assert completed_plan.direct_policy_targets[0].step_index == 1
    assert completed_plan.risk_sequences == ()
    assert EvidenceStratum.RISK_SEQUENCE not in completed_plan.strata
    assert EvidenceStratum.COMPLETION_CONTROL in completed_plan.strata

    cycle_context = _context(
        context_id="cycle",
        count=3,
        node_loop_identities=("cycle-node-a", "cycle-node-middle", "cycle-node-b"),
    )
    cycle = PolicyWitness(
        witness_id="cycle-core",
        kind=WitnessKind.MULTI_EDGE_CYCLE,
        attributed_step_indices=(0, 2),
        supporting_episode_steps=(0, 2, 4),
        occurrences=3,
        cycle_span=4,
        successor_confirmed=True,
        loop_edges=(
            _loop_edge(
                cycle_context,
                step_index=0,
                supporting_episode_steps=(0, 4),
            ),
            _loop_edge(
                cycle_context,
                step_index=2,
                supporting_episode_steps=(2,),
            ),
        ),
        behavior_mean_log_probability=-0.7,
    )
    cycle_plan = CreditCompiler().compile(
        _failed_incident(
            cycle_context,
            incident_id="cycle-incident",
            witnesses=(cycle,),
        )
    )
    assert cycle_plan.cycle_policy_targets[0].step_indices == (0, 2)
    assert cycle_plan.direct_policy_targets == ()
    assert EvidenceStratum.MULTI_EDGE_CYCLE in cycle_plan.strata


def test_atomic_outcome_pair_compiles_contrast_and_indexes_as_one_record() -> None:
    better_context = _context(
        context_id="better",
        action_index=0,
    )
    worse_context = _context(
        context_id="worse",
        action_index=1,
    )
    pair = MatchedOutcomePair(
        pair_id="pair-1",
        better=OutcomeArm(
            incident_id="better-incident",
            context=better_context,
            step_index=0,
            outcome=FailureOutcome.COMPLETED,
        ),
        worse=OutcomeArm(
            incident_id="worse-incident",
            context=worse_context,
            step_index=0,
            outcome=FailureOutcome.DEADLOCK_CYCLE,
        ),
    )
    witness = PolicyWitness(
        witness_id="matched-pair-witness",
        kind=WitnessKind.MATCHED_OUTCOME_PAIR,
        attributed_step_indices=(),
        supporting_episode_steps=(0,),
        occurrences=2,
        cycle_span=None,
        successor_confirmed=True,
        outcome_pair=pair,
    )
    incident = _failed_incident(
        worse_context,
        incident_id="worse-incident",
        witnesses=(witness,),
    )
    plan = CreditCompiler().compile(incident)
    corpus = ImmutableEvidenceCorpus(records=(EvidenceRecord(incident=incident, plan=plan),))

    assert plan.contrast_policy_targets[0].pair is pair
    assert corpus.incident_ids(EvidenceStratum.MATCHED_OUTCOME_PAIR) == ("worse-incident",)
    assert tuple(corpus.outcome_pairs) == ("pair-1",)
    assert corpus.outcome_pairs["pair-1"].better.step.selected_action.comparison.payload == {"identity": "action-0"}
    assert corpus.outcome_pairs["pair-1"].worse.step.selected_action.comparison.payload == {"identity": "action-1"}


def test_outcome_matcher_enriches_an_incoming_attributed_failure() -> None:
    completion = _completion_record(
        _context(context_id="match-completion", action_index=0),
        incident_id="completion-source",
    )
    failure = _direct_failure_record(
        _context(context_id="match-failure", action_index=1),
        incident_id="failure-source",
    )
    matcher = OutcomePairMatcher(maximum_pairs_per_publication=4)

    publication = matcher.match(
        (failure,),
        retained_records=(completion,),
    )

    assert publication.matched_pair_count == 1
    assert publication.replacements == ()
    assert len(publication.records) == 1
    enriched = publication.records[0]
    assert enriched.incident.incident_id == failure.incident.incident_id
    assert enriched.incident.context is failure.incident.context
    assert {witness.witness_id for witness in failure.incident.witnesses} < {
        witness.witness_id for witness in enriched.incident.witnesses
    }
    assert EvidenceStratum.DIRECT_WITNESS in enriched.plan.strata
    assert EvidenceStratum.MATCHED_OUTCOME_PAIR in enriched.plan.strata
    pair = enriched.plan.contrast_policy_targets[0].pair
    assert pair.better.incident_id == completion.incident.incident_id
    assert pair.worse.incident_id == failure.incident.incident_id
    assert pair.better.step.selected_action.comparison != (pair.worse.step.selected_action.comparison)


def test_outcome_matcher_uses_unresolved_stall_only_with_exact_completion_contrast() -> None:
    failure_context = _context(
        context_id="stall-match-failure",
        count=3,
        action_index=1,
    )
    failure_incident = _failed_incident(
        failure_context,
        incident_id="stall-match-failure-source",
        outcome=FailureOutcome.DEADLOCK_STALL,
        witnesses=(),
        detector_window_steps=256,
    )
    failure = EvidenceRecord(
        incident=failure_incident,
        plan=CreditCompiler().compile(failure_incident),
    )
    completion = _completion_record(
        _context(
            context_id="stall-match-completion",
            count=3,
            action_index=0,
        ),
        incident_id="stall-match-completion-source",
    )

    assert EvidenceStratum.UNRESOLVED_STALL in failure.plan.strata
    assert failure.plan.direct_policy_targets == ()
    publication = OutcomePairMatcher(maximum_pairs_per_publication=1).match(
        (failure,),
        retained_records=(completion,),
    )

    assert publication.matched_pair_count == 1
    enriched = publication.records[0]
    # Once an exact completion contrast is attached the formerly unresolved
    # local stall becomes resolved contrast evidence; the risk sequence remains
    # as its encounter-level provenance.
    assert EvidenceStratum.RISK_SEQUENCE in enriched.plan.strata
    assert EvidenceStratum.UNRESOLVED_STALL not in enriched.plan.strata
    assert EvidenceStratum.MATCHED_OUTCOME_PAIR in enriched.plan.strata
    pair = enriched.plan.contrast_policy_targets[0].pair
    assert pair.worse.incident_id == failure.incident.incident_id
    assert pair.better.incident_id == completion.incident.incident_id
    assert pair.worse.step.node.comparison == pair.better.step.node.comparison
    assert pair.worse.step.selected_action.comparison != (
        pair.better.step.selected_action.comparison
    )


def test_outcome_matcher_replaces_a_retained_failure_atomically() -> None:
    failure = _direct_failure_record(
        _context(context_id="retained-failure", action_index=1),
        incident_id="retained-failure-source",
    )
    completion = _completion_record(
        _context(context_id="new-completion", action_index=0),
        incident_id="new-completion-source",
    )
    replay = BoundedFailureCreditReplay(
        capacity=4,
        byte_capacity=100_000_000,
        seed=9,
    )
    assert replay.put(failure)
    matcher = OutcomePairMatcher(maximum_pairs_per_publication=4)

    publication = matcher.match(
        (completion,),
        retained_records=replay.snapshot().records,
    )
    assert publication.records == (completion,)
    assert publication.matched_pair_count == 1
    assert len(publication.replacements) == 1
    assert replay.replace_many(publication.replacements) == 1
    assert replay.put_many(publication.records) == 1

    corpus = replay.snapshot()
    replaced = corpus.record(failure.incident.incident_id)
    assert replaced.incident.context is failure.incident.context
    assert EvidenceStratum.DIRECT_WITNESS in replaced.plan.strata
    assert EvidenceStratum.MATCHED_OUTCOME_PAIR in replaced.plan.strata
    assert corpus.metrics().outcome_pair_count == 1
    assert replay.metrics()["put_count"] == 2


@pytest.mark.parametrize(
    ("completion_action", "comparison_node", "expected_pairs"),
    (
        (1, "comparable", 0),
        (0, "different-node", 0),
        (0, "comparable", 1),
    ),
)
def test_outcome_matcher_requires_same_decision_and_different_action(
    completion_action: int,
    comparison_node: str,
    expected_pairs: int,
) -> None:
    failure = _direct_failure_record(
        _context(context_id="strict-failure", action_index=1),
        incident_id="strict-failure-source",
    )
    completion = _completion_record(
        _context(
            context_id="strict-completion",
            action_index=completion_action,
            node_comparison=comparison_node,
        ),
        incident_id=f"strict-completion-{completion_action}-{comparison_node}",
    )

    publication = OutcomePairMatcher(maximum_pairs_per_publication=1).match(
        (completion,),
        retained_records=(failure,),
    )

    assert publication.matched_pair_count == expected_pairs
    assert len(publication.replacements) == expected_pairs


def test_matched_outcome_worse_arm_must_alias_incident_context() -> None:
    better_context = _context(context_id="better-alias", action_index=0)
    incident_context = _context(context_id="worse-alias", action_index=1)
    duplicate_context = replace(
        incident_context,
        steps=tuple(incident_context.steps),
    )
    assert duplicate_context.context_id == incident_context.context_id
    assert duplicate_context is not incident_context

    pair = MatchedOutcomePair(
        pair_id="pair-context-alias",
        better=OutcomeArm(
            incident_id="better-context-alias",
            context=better_context,
            step_index=0,
            outcome=FailureOutcome.COMPLETED,
        ),
        worse=OutcomeArm(
            incident_id="worse-context-alias",
            context=duplicate_context,
            step_index=0,
            outcome=FailureOutcome.DEADLOCK_CYCLE,
        ),
    )
    witness = PolicyWitness(
        witness_id="matched-context-alias",
        kind=WitnessKind.MATCHED_OUTCOME_PAIR,
        attributed_step_indices=(),
        supporting_episode_steps=(0,),
        occurrences=2,
        cycle_span=None,
        successor_confirmed=True,
        outcome_pair=pair,
    )

    with pytest.raises(
        ValueError,
        match="matched pair worse arm must alias the incident context",
    ):
        _failed_incident(
            incident_context,
            incident_id="worse-context-alias",
            witnesses=(witness,),
        )


def test_v4_corpus_roundtrip_preserves_provenance_and_quota_diagnostics() -> None:
    context = _context(context_id="roundtrip", count=2)
    incident = _failed_incident(
        context,
        incident_id="roundtrip-incident",
        outcome=FailureOutcome.DEADLOCK_STALL,
    )
    plan = CreditCompiler().compile(incident)
    corpus = ImmutableEvidenceCorpus(records=(EvidenceRecord(incident=incident, plan=plan),))
    payload = pickle.loads(pickle.dumps(corpus.state_dict()))
    restored = ImmutableEvidenceCorpus.from_state_dict(payload)

    assert restored.version == FAILURE_EVIDENCE_REPLAY_VERSION
    assert restored.record("roundtrip-incident").incident.provenance == _provenance()
    assert restored.incident_ids(EvidenceStratum.UNRESOLVED_STALL) == ("roundtrip-incident",)
    assert set(restored.stratum_index) == set(EvidenceStratum)
    metrics = restored.metrics()
    assert metrics.record_count == 1
    assert metrics.storage_nbytes == restored.storage_nbytes
    assert metrics.storage_nbytes > context.initial_recurrent_state.nbytes
    sample = restored.sample(
        batch_size=1,
        rng=np.random.default_rng(123),
        risk_actor_enabled=False,
        quotas=(StratumQuota(EvidenceStratum.UNRESOLVED_STALL, 1),),
    )
    assert tuple(record.incident.incident_id for record in sample.records) == ("roundtrip-incident",)
    assert sample.quota_diagnostics.satisfied
    assert not restored.quota_diagnostics(
        (
            StratumQuota(EvidenceStratum.UNRESOLVED_STALL, 1),
            StratumQuota(EvidenceStratum.DIRECT_WITNESS, 1),
        ),
        risk_actor_enabled=False,
        selected_incident_ids=("roundtrip-incident",),
    ).satisfied
    diagnostics = restored.quota_diagnostics(
        (StratumQuota(EvidenceStratum.UNRESOLVED_STALL, 1),),
        risk_actor_enabled=False,
        selected_incident_ids=("roundtrip-incident",),
    )
    assert diagnostics.satisfied
    assert diagnostics.total_deficit == 0

    with pytest.raises(ValueError, match="unsupported"):
        ImmutableEvidenceCorpus.from_state_dict(
            {
                **restored.state_dict(),
                "version": "sts2-failure-credit-v3",
            }
        )


def test_censored_incident_is_indexed_but_has_no_learning_target() -> None:
    context = _context(context_id="censored")
    incident = FailureIncident(
        incident_id="censored-incident",
        scope_key="scope",
        failure_kind="transport_abort",
        outcome=FailureOutcome.CENSORED,
        task_authority=TargetAuthority.CENSORED,
        local_authority=TargetAuthority.CENSORED,
        task_return=None,
        local_failure_cost=None,
        context=context,
        witnesses=(),
        detector_window_steps=64,
        progress_epoch=0,
        provenance=_provenance(),
    )
    plan = CreditCompiler().compile(incident)
    corpus = ImmutableEvidenceCorpus(records=(EvidenceRecord(incident=incident, plan=plan),))

    assert plan.actor_label_count == 0
    assert plan.task_q_targets == ()
    assert plan.liveness_q_targets == ()
    assert plan.strata == (EvidenceStratum.CENSORED,)
    assert corpus.incident_ids(EvidenceStratum.CENSORED) == ("censored-incident",)
