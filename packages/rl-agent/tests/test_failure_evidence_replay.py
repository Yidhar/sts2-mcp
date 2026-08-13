from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from threading import Event, Thread

import numpy as np
import pytest

from sts2_rl.encoding import EncodedDecisionSnapshot, GroundedEncodingConfig
from sts2_rl.encoding.snapshot import sparse_token_table
from sts2_rl.semantics import IdentityTriple, SemanticKey
from sts2_rl.training.failure_credit import (
    BoundedFailureCreditReplay,
    CreditCompiler,
    CreditPlan,
    CreditProvenance,
    EvidenceRecord,
    EvidenceStratum,
    FailureIncident,
    FailureOutcome,
    ImmutableEvidenceCorpus,
    LearningContext,
    LearningStep,
    MatchedOutcomePair,
    OutcomeArm,
    PolicyWitness,
    StratumQuota,
    TargetAuthority,
    WitnessKind,
    evidence_record_storage_nbytes,
)


def _semantic(namespace: str, identity: str) -> SemanticKey:
    return SemanticKey.from_payload(
        namespace=namespace,
        schema_version="replay-test-semantics-v1",
        payload={"identity": identity},
    )


def _identity(
    kind: str,
    identity: str,
    *,
    loop_identity: str | None = None,
    comparison_identity: str | None = None,
) -> IdentityTriple:
    return IdentityTriple(
        exact=_semantic(f"{kind}.exact", identity),
        loop=_semantic(f"{kind}.loop", loop_identity or identity),
        comparison=_semantic(
            f"{kind}.comparison",
            comparison_identity or identity,
        ),
    )


def _snapshot() -> EncodedDecisionSnapshot:
    config = GroundedEncodingConfig(
        max_world_tokens=4,
        max_candidates=4,
        max_candidate_local_tokens=2,
    )
    feature_dim = config.feature_dim
    return EncodedDecisionSnapshot(
        config=config,
        encoding_fingerprint="0" * 64,
        world=sparse_token_table(
            features=(tuple([1.0] + [0.0] * (feature_dim - 1)),),
            ids=((2, 2, 2, 2, 2, 2, 2, 2, 0),),
            feature_dim=feature_dim,
            id_width=9,
        ),
        candidates=sparse_token_table(
            features=(
                tuple([1.0] + [0.0] * (feature_dim - 1)),
                tuple([0.0, 1.0] + [0.0] * (feature_dim - 2)),
            ),
            ids=(
                (2, 2, 2, 3, 3, 3, 3, 2, 2, 4, 4, 4, 4),
                (3, 2, 2, 3, 3, 3, 3, 2, 2, 4, 4, 4, 4),
            ),
            feature_dim=feature_dim,
            id_width=13,
        ),
        locals=sparse_token_table(
            features=(),
            ids=(),
            feature_dim=feature_dim,
            id_width=9,
        ),
        local_offsets=np.zeros(3, dtype=np.uint32),
        action_mask=np.ones(2, dtype=np.bool_),
        domain_id=1,
    )


def _provenance(*, policy_version: int = 4) -> CreditProvenance:
    return CreditProvenance(
        run_id="replay-v4-test-run",
        game_version="game-v1",
        environment_schema_version="environment-v1",
        identity_version="identity-v1",
        detector_version="detector-v1",
        adapter_version="adapter-v1",
        collector_version="collector-v1",
        policy_version=policy_version,
    )


def _context(
    identity: str,
    *,
    action_index: int = 0,
    comparison_node: str = "comparison-node",
    policy_version: int = 4,
) -> LearningContext:
    candidates = tuple(
        _identity(
            "action",
            f"{identity}:action-instance-{index}",
            loop_identity=f"loop-action-{index}",
            comparison_identity=f"comparison-action-{index}",
        )
        for index in range(2)
    )
    return LearningContext(
        context_id=f"context-{identity}",
        episode_id=f"episode-{identity}",
        start_step=0,
        initial_recurrent_state=np.zeros(8, dtype=np.float32),
        steps=(
            LearningStep(
                decision_id=f"decision-{identity}",
                episode_step=0,
                snapshot=_snapshot(),
                action_index=action_index,
                behavior_log_probability=-0.693147,
                policy_version=policy_version,
                node=_identity(
                    "node",
                    f"node-instance-{identity}",
                    loop_identity="loop-node",
                    comparison_identity=comparison_node,
                ),
                anchor=_semantic("anchor", "scope"),
                candidate_actions=candidates,
                forced=False,
            ),
        ),
    )


def _record(
    identity: str,
    *,
    strata: tuple[EvidenceStratum, ...] = (EvidenceStratum.COMPLETION_CONTROL,),
) -> EvidenceRecord:
    context = _context(identity)
    provenance = _provenance()
    incident = FailureIncident(
        incident_id=f"incident-{identity}",
        scope_key="scope",
        failure_kind="completed",
        outcome=FailureOutcome.COMPLETED,
        task_authority=TargetAuthority.CENSORED,
        local_authority=TargetAuthority.VERIFIED_TRANSITION,
        task_return=None,
        local_failure_cost=0.0,
        context=context,
        witnesses=(),
        detector_window_steps=64,
        progress_epoch=1,
        provenance=provenance,
    )
    plan = CreditPlan(
        plan_id=f"plan-{identity}",
        incident_id=incident.incident_id,
        context=context,
        task_value_targets=(),
        task_q_targets=(),
        liveness_value_targets=(),
        liveness_q_targets=(),
        strata=strata,
        provenance=provenance,
    )
    return EvidenceRecord(incident=incident, plan=plan)


def _failed_record(
    identity: str,
    *,
    stratum: EvidenceStratum,
    policy_version: int = 4,
) -> EvidenceRecord:
    if stratum not in {
        EvidenceStratum.DIRECT_WITNESS,
        EvidenceStratum.MULTI_EDGE_CYCLE,
        EvidenceStratum.RISK_SEQUENCE,
    }:
        raise ValueError("test failed record requires a failure evidence stratum")
    context = _context(identity, policy_version=policy_version)
    provenance = _provenance(policy_version=policy_version)
    incident = FailureIncident(
        incident_id=f"incident-{identity}",
        scope_key="scope",
        failure_kind="deadlock",
        outcome=FailureOutcome.DEADLOCK_STALL,
        task_authority=TargetAuthority.OBJECTIVE_CONFIRMED,
        local_authority=TargetAuthority.DETECTOR_CONFIRMED,
        task_return=-1.0,
        local_failure_cost=1.0,
        context=context,
        witnesses=(),
        detector_window_steps=64,
        progress_epoch=1,
        provenance=provenance,
    )
    return EvidenceRecord(
        incident=incident,
        plan=CreditPlan(
            plan_id=f"plan-{identity}",
            incident_id=incident.incident_id,
            context=context,
            task_value_targets=(),
            task_q_targets=(),
            liveness_value_targets=(),
            liveness_q_targets=(),
            strata=(stratum,),
            provenance=provenance,
        ),
    )


def _matched_pair_record(identity: str) -> EvidenceRecord:
    better_context = _context(
        f"{identity}-better",
        action_index=0,
    )
    worse_context = _context(
        f"{identity}-worse",
        action_index=1,
    )
    better_incident_id = f"incident-{identity}-better"
    worse_incident_id = f"incident-{identity}-worse"
    pair = MatchedOutcomePair(
        pair_id=f"pair-{identity}",
        better=OutcomeArm(
            incident_id=better_incident_id,
            context=better_context,
            step_index=0,
            outcome=FailureOutcome.COMPLETED,
        ),
        worse=OutcomeArm(
            incident_id=worse_incident_id,
            context=worse_context,
            step_index=0,
            outcome=FailureOutcome.DEADLOCK_CYCLE,
        ),
    )
    witness = PolicyWitness(
        witness_id=f"pair-witness-{identity}",
        kind=WitnessKind.MATCHED_OUTCOME_PAIR,
        attributed_step_indices=(),
        supporting_episode_steps=(0,),
        occurrences=2,
        cycle_span=None,
        successor_confirmed=True,
        outcome_pair=pair,
    )
    incident = FailureIncident(
        incident_id=worse_incident_id,
        scope_key="scope",
        failure_kind="deadlock_cycle",
        outcome=FailureOutcome.DEADLOCK_CYCLE,
        task_authority=TargetAuthority.OBJECTIVE_CONFIRMED,
        local_authority=TargetAuthority.DETECTOR_CONFIRMED,
        task_return=-1.0,
        local_failure_cost=1.0,
        context=worse_context,
        witnesses=(witness,),
        detector_window_steps=64,
        progress_epoch=1,
        provenance=_provenance(),
    )
    return EvidenceRecord(
        incident=incident,
        plan=CreditCompiler().compile(incident),
    )


def _record_size(record: EvidenceRecord) -> int:
    probe = BoundedFailureCreditReplay(
        capacity=1,
        byte_capacity=100_000_000,
        seed=0,
    )
    assert probe.put(record)
    return int(probe.metrics()["storage_nbytes"])


def _incident_ids(replay: BoundedFailureCreditReplay) -> tuple[str, ...]:
    return tuple(record.incident.incident_id for record in replay.snapshot().records)


def test_record_rejects_distinct_incident_and_plan_contexts_even_with_same_id() -> None:
    record = _record("split-context")
    distinct_context = replace(record.plan.context)
    assert distinct_context.context_id == record.incident.context.context_id
    assert distinct_context is not record.incident.context

    with pytest.raises(
        ValueError,
        match="must share one immutable context",
    ):
        EvidenceRecord(
            incident=record.incident,
            plan=replace(record.plan, context=distinct_context),
        )


def test_capacity_and_byte_bounds_evict_deterministic_fifo() -> None:
    records = tuple(_record(str(index)) for index in range(3))
    replay = BoundedFailureCreditReplay(
        capacity=2,
        byte_capacity=100_000_000,
        seed=1,
    )
    for record in records:
        assert replay.put(record)
    assert _incident_ids(replay) == (
        "incident-1",
        "incident-2",
    )
    assert replay.metrics()["eviction_count"] == 1

    first_size = _record_size(records[0])
    second_size = _record_size(records[1])
    byte_bounded = BoundedFailureCreditReplay(
        capacity=8,
        byte_capacity=first_size + second_size - 1,
        seed=1,
    )
    assert byte_bounded.put(records[0])
    assert byte_bounded.put(records[1])
    assert _incident_ids(byte_bounded) == ("incident-1",)
    assert int(byte_bounded.metrics()["storage_nbytes"]) <= (first_size + second_size - 1)


def test_oversize_and_duplicate_records_are_fail_closed_and_counted() -> None:
    record = _record("bounded")
    size = _record_size(record)
    oversize = BoundedFailureCreditReplay(
        capacity=2,
        byte_capacity=size - 1,
        seed=2,
    )
    with pytest.raises(ValueError, match="exceeds replay byte capacity"):
        oversize.put(record)
    assert len(oversize) == 0
    assert oversize.metrics()["oversize_count"] == 1
    assert oversize.metrics()["maximum_observed_record_nbytes"] == size

    replay = BoundedFailureCreditReplay(
        capacity=2,
        byte_capacity=100_000_000,
        seed=2,
    )
    assert replay.put(record)
    assert not replay.put(record)
    assert len(replay) == 1
    assert replay.metrics()["duplicate_count"] == 1


def test_replay_reports_byte_capacity_health_from_observed_records() -> None:
    record = _record("capacity-health")
    record_size = evidence_record_storage_nbytes(record)
    byte_capacity = record_size * 5 + 7
    replay = BoundedFailureCreditReplay(
        capacity=32,
        byte_capacity=byte_capacity,
        seed=19,
    )

    assert replay.put(record)
    metrics = replay.metrics()

    assert metrics["storage_nbytes"] == record_size
    assert metrics["byte_headroom_nbytes"] == byte_capacity - record_size
    assert metrics["byte_utilization_ppm"] == record_size * 1_000_000 // byte_capacity
    assert metrics["mean_record_nbytes"] == record_size
    assert metrics["observed_worst_case_record_capacity"] == byte_capacity // record_size


def test_put_many_builds_and_publishes_one_corpus_for_an_atomic_episode_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay = BoundedFailureCreditReplay(
        capacity=3,
        byte_capacity=100_000_000,
        seed=21,
    )
    assert replay.put(_record("batch-old"))
    direct = _failed_record(
        "batch-direct",
        stratum=EvidenceStratum.DIRECT_WITNESS,
    )
    pair = _matched_pair_record("batch-pair")
    tail = _record("batch-tail")
    builds = 0
    original_post_init = ImmutableEvidenceCorpus.__post_init__

    def counted_post_init(corpus: ImmutableEvidenceCorpus) -> None:
        nonlocal builds
        builds += 1
        original_post_init(corpus)

    monkeypatch.setattr(
        ImmutableEvidenceCorpus,
        "__post_init__",
        counted_post_init,
    )
    accepted = replay.put_many((direct, direct, pair, tail))

    assert accepted == 3
    assert builds == 1
    assert _incident_ids(replay) == (
        "incident-batch-direct",
        "incident-batch-pair-worse",
        "incident-batch-tail",
    )
    assert tuple(replay.snapshot().outcome_pairs) == ("pair-batch-pair",)
    metrics = replay.metrics()
    assert metrics["put_count"] == 4
    assert metrics["put_batch_count"] == 2
    assert metrics["duplicate_count"] == 1
    assert metrics["eviction_count"] == 1


def test_put_many_oversize_rejects_the_whole_batch_before_publication() -> None:
    ordinary = _record("atomic-ordinary")
    oversized = _matched_pair_record("atomic-oversized")
    ordinary_size = evidence_record_storage_nbytes(ordinary)
    oversized_size = evidence_record_storage_nbytes(oversized)
    assert ordinary_size < oversized_size
    replay = BoundedFailureCreditReplay(
        capacity=4,
        byte_capacity=oversized_size - 1,
        seed=22,
    )

    with pytest.raises(ValueError, match="exceeds replay byte capacity"):
        replay.put_many((ordinary, oversized))

    assert len(replay) == 0
    metrics = replay.metrics()
    assert metrics["put_count"] == 0
    assert metrics["put_batch_count"] == 0
    assert metrics["eviction_count"] == 0
    assert metrics["oversize_count"] == 1


def test_quota_sampling_and_metrics_cover_all_seven_strata() -> None:
    replay = BoundedFailureCreditReplay(
        capacity=len(EvidenceStratum),
        byte_capacity=100_000_000,
        seed=3,
    )
    records = (
        _failed_record(
            "stratum-direct",
            stratum=EvidenceStratum.DIRECT_WITNESS,
        ),
        _failed_record(
            "stratum-multi",
            stratum=EvidenceStratum.MULTI_EDGE_CYCLE,
        ),
        _matched_pair_record("stratum-pair"),
        _failed_record(
            "stratum-risk",
            stratum=EvidenceStratum.RISK_SEQUENCE,
        ),
        _record(
            "stratum-unresolved",
            strata=(EvidenceStratum.UNRESOLVED_STALL,),
        ),
        _record(
            "stratum-completion",
            strata=(EvidenceStratum.COMPLETION_CONTROL,),
        ),
        _record(
            "stratum-censored",
            strata=(EvidenceStratum.CENSORED,),
        ),
    )
    assert replay.put_many(records) == len(records)
    quotas = tuple(StratumQuota(stratum=stratum, minimum=1) for stratum in EvidenceStratum)
    sample = replay.sample(
        len(EvidenceStratum),
        quotas=quotas,
        current_policy_version=4,
    )
    assert sample.quota_diagnostics.satisfied
    assert {status.stratum for status in sample.quota_diagnostics.statuses} == set(EvidenceStratum)
    for status in sample.quota_diagnostics.statuses:
        assert status.requested == 1
        assert status.available >= 1
        assert status.selected >= 1
        assert status.deficit == 0
    metrics = replay.metrics()
    for stratum in EvidenceStratum:
        assert int(metrics[f"stratum_{stratum.value.lower()}_size"]) >= 1

    deficit_replay = BoundedFailureCreditReplay(
        capacity=1,
        byte_capacity=100_000_000,
        seed=3,
    )
    assert deficit_replay.put(
        _record(
            "only-censored",
            strata=(EvidenceStratum.CENSORED,),
        )
    )
    deficit = deficit_replay.sample(
        1,
        quotas=(StratumQuota(EvidenceStratum.DIRECT_WITNESS, 1),),
        current_policy_version=4,
    )
    assert deficit.quota_diagnostics.total_deficit == 1
    assert deficit_replay.metrics()["quota_deficit_count"] == 1


def test_replay_metrics_use_publication_caches_not_full_corpus_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay = BoundedFailureCreditReplay(
        capacity=4,
        byte_capacity=100_000_000,
        seed=31,
    )
    assert replay.put(_matched_pair_record("cached-metrics"))

    def forbidden_full_scan(
        _corpus: ImmutableEvidenceCorpus,
    ) -> object:
        raise AssertionError("replay metrics performed a full corpus scan")

    monkeypatch.setattr(
        ImmutableEvidenceCorpus,
        "metrics",
        forbidden_full_scan,
    )
    metrics = replay.metrics()
    assert metrics["size"] == 1
    assert metrics["outcome_pair_count"] == 1
    assert metrics["stratum_matched_outcome_pair_size"] == 1


def test_matched_pair_is_inserted_checkpointed_and_evicted_atomically() -> None:
    pair_record = _matched_pair_record("atomic")
    assert evidence_record_storage_nbytes(pair_record) > (evidence_record_storage_nbytes(_record("single-context")))
    replay = BoundedFailureCreditReplay(
        capacity=1,
        byte_capacity=100_000_000,
        seed=4,
    )
    assert replay.put(pair_record)
    corpus = replay.snapshot()
    assert tuple(corpus.outcome_pairs) == ("pair-atomic",)
    pair = corpus.outcome_pairs["pair-atomic"]
    assert pair.better.context.context_id == "context-atomic-better"
    assert pair.worse.context.context_id == "context-atomic-worse"

    restored = BoundedFailureCreditReplay(
        capacity=1,
        byte_capacity=100_000_000,
        seed=999,
    )
    restored.load_state_dict(pickle.loads(pickle.dumps(replay.state_dict())))
    restored_pair = restored.snapshot().outcome_pairs["pair-atomic"]
    assert restored_pair.better.incident_id == "incident-atomic-better"
    assert restored_pair.worse.incident_id == "incident-atomic-worse"
    assert len(restored.snapshot().records) == 1

    # A critic-only completion must not evict the sole matched actor record in
    # a constrained replay.  Admission fails closed and the pair remains one
    # atomic live value.
    assert not restored.put(_record("replacement"))
    assert tuple(restored.snapshot().outcome_pairs) == ("pair-atomic",)
    assert restored.metrics()["admission_rejection_count"] == 1


def test_completion_flood_cannot_evict_last_direct_or_multi_edge_records() -> None:
    direct = _failed_record(
        "protected-direct",
        stratum=EvidenceStratum.DIRECT_WITNESS,
        policy_version=10,
    )
    multi = _failed_record(
        "protected-multi",
        stratum=EvidenceStratum.MULTI_EDGE_CYCLE,
        policy_version=10,
    )
    replay = BoundedFailureCreditReplay(
        capacity=4,
        byte_capacity=100_000_000,
        seed=404,
    )
    assert replay.put_many((direct, multi, _record("completion-0"))) == 3
    for index in range(1, 20):
        assert replay.put(_record(f"completion-{index}"))

    live_ids = {record.incident.incident_id for record in replay.snapshot().records}
    assert direct.incident.incident_id in live_ids
    assert multi.incident.incident_id in live_ids
    assert replay.metrics()["stratum_completion_control_size"] == 2
    assert replay.metrics()["evicted_primary_completion_control_count"] >= 18


def test_stratum_refresh_replaces_oldest_peer_before_protected_other_strata() -> None:
    direct_old = _failed_record(
        "direct-old",
        stratum=EvidenceStratum.DIRECT_WITNESS,
        policy_version=1,
    )
    multi = _failed_record(
        "multi-stays",
        stratum=EvidenceStratum.MULTI_EDGE_CYCLE,
        policy_version=1,
    )
    direct_fresh = _failed_record(
        "direct-fresh",
        stratum=EvidenceStratum.DIRECT_WITNESS,
        policy_version=100,
    )
    replay = BoundedFailureCreditReplay(
        capacity=2,
        byte_capacity=100_000_000,
        seed=405,
    )
    assert replay.put_many((direct_old, multi)) == 2
    assert replay.put(direct_fresh)
    live_ids = tuple(record.incident.incident_id for record in replay.snapshot().records)
    assert live_ids == (multi.incident.incident_id, direct_fresh.incident.incident_id)
    assert replay.metrics()["evicted_primary_direct_witness_count"] == 1


def test_exact_v6_roundtrip_preserves_rng_counters_and_next_sample() -> None:
    replay = BoundedFailureCreditReplay(
        capacity=4,
        byte_capacity=100_000_000,
        seed=5,
    )
    records = tuple(_record(f"roundtrip-{index}") for index in range(4))
    for record in records:
        assert replay.put(record)
    replay.sample(2)
    assert not replay.put(records[0])

    payload = pickle.loads(pickle.dumps(replay.state_dict()))
    restored = BoundedFailureCreditReplay(
        capacity=4,
        byte_capacity=100_000_000,
        seed=999,
    )
    restored.load_state_dict(payload)
    assert restored.metrics() == replay.metrics()
    assert _incident_ids(restored) == _incident_ids(replay)

    expected = tuple(record.incident.incident_id for record in replay.sample(3).records)
    actual = tuple(record.incident.incident_id for record in restored.sample(3).records)
    assert actual == expected
    assert restored.metrics() == replay.metrics()


def test_exact_v6_sampling_is_independent_of_python_hash_seed(
    tmp_path: Path,
) -> None:
    replay = BoundedFailureCreditReplay(
        capacity=12,
        byte_capacity=100_000_000,
        seed=250,
    )
    records = tuple(
        _failed_record(
            f"cross-process-{index:02d}",
            stratum=EvidenceStratum.DIRECT_WITNESS,
            policy_version=100,
        )
        for index in range(12)
    )
    assert replay.put_many(records) == len(records)
    state_path = tmp_path / "failure-replay-v6.pkl"
    state_path.write_bytes(pickle.dumps(replay.state_dict()))
    script = """
import json
import pickle
import sys
from pathlib import Path

from sts2_rl.training.failure_credit import (
    BoundedFailureCreditReplay,
    EvidenceStratum,
    StratumQuota,
)

replay = BoundedFailureCreditReplay(
    capacity=12,
    byte_capacity=100_000_000,
    seed=999,
)
replay.load_state_dict(pickle.loads(Path(sys.argv[1]).read_bytes()))
sample = replay.sample(
    6,
    quotas=(StratumQuota(EvidenceStratum.DIRECT_WITNESS, 3),),
    current_policy_version=100,
)
print(json.dumps([record.incident.incident_id for record in sample.records]))
"""
    package_root = Path(__file__).resolve().parents[1]

    def run_with_hash_seed(seed: str) -> list[str]:
        environment = dict(os.environ)
        environment["PYTHONHASHSEED"] = seed
        completed = subprocess.run(
            [sys.executable, "-c", script, str(state_path)],
            cwd=package_root,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(completed.stdout)
        assert isinstance(payload, list)
        assert all(isinstance(item, str) for item in payload)
        return payload

    assert run_with_hash_seed("1") == run_with_hash_seed("987654")


def test_future_evidence_is_rejected_without_advancing_rng_or_counters() -> None:
    replay = BoundedFailureCreditReplay(
        capacity=1,
        byte_capacity=100_000_000,
        seed=26,
    )
    assert replay.put(
        _failed_record(
            "future-policy",
            stratum=EvidenceStratum.RISK_SEQUENCE,
            policy_version=101,
        )
    )
    original = pickle.dumps(replay.state_dict())

    with pytest.raises(
        ValueError,
        match="newer than the learner",
    ):
        replay.sample(
            1,
            quotas=(StratumQuota(EvidenceStratum.RISK_SEQUENCE, 1),),
            current_policy_version=100,
        )

    assert pickle.dumps(replay.state_dict()) == original


def test_load_rejects_v3_capacity_and_corrupt_accounting_without_mutation() -> None:
    replay = BoundedFailureCreditReplay(
        capacity=2,
        byte_capacity=100_000_000,
        seed=6,
    )
    assert replay.put(_record("strict"))
    original = pickle.dumps(replay.state_dict())

    v3 = pickle.loads(original)
    v3["version"] = "sts2-failure-evidence-replay-v3"
    with pytest.raises(ValueError, match="unsupported"):
        replay.load_state_dict(v3)

    wrong_capacity = pickle.loads(original)
    wrong_capacity["capacity"] = 3
    with pytest.raises(ValueError, match="capacity differs"):
        replay.load_state_dict(wrong_capacity)

    corrupt_accounting = pickle.loads(original)
    corrupt_accounting["put_count"] = 99
    with pytest.raises(ValueError, match="accounting is inconsistent"):
        replay.load_state_dict(corrupt_accounting)

    extra_field = pickle.loads(original)
    extra_field["legacy_tail_scan"] = ()
    with pytest.raises(ValueError, match="unsupported"):
        replay.load_state_dict(extra_field)
    assert pickle.dumps(replay.state_dict()) == original


def test_load_refuses_retired_v5_version_and_actor_counter_payloads() -> None:
    replay = BoundedFailureCreditReplay(
        capacity=2,
        byte_capacity=100_000_000,
        seed=27,
    )
    assert replay.put(_record("refusal"))
    original = pickle.dumps(replay.state_dict())

    # A payload claiming the retired actor-freshness replay version must fail
    # closed even when every remaining field is v6-shaped.
    v5 = pickle.loads(original)
    v5["version"] = "sts2-failure-evidence-replay-v5"
    with pytest.raises(
        ValueError,
        match="unsupported failure evidence replay checkpoint schema",
    ):
        replay.load_state_dict(v5)

    # A v6-versioned payload still carrying the retired actor counters is a
    # different schema, never a silently ignored superset.
    actor_counters = pickle.loads(original)
    actor_counters["actor_actionable_records"] = 1
    actor_counters["sampled_direct_actor_fresh_count"] = 0
    actor_counters["sampled_direct_lag_suppressed_critic_count"] = 0
    with pytest.raises(
        ValueError,
        match="unsupported failure evidence replay checkpoint schema",
    ):
        replay.load_state_dict(actor_counters)

    assert pickle.dumps(replay.state_dict()) == original


def test_sampling_releases_publication_lock_before_quota_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay = BoundedFailureCreditReplay(
        capacity=4,
        byte_capacity=100_000_000,
        seed=7,
    )
    assert replay.put(_record("sampled"))
    entered = Event()
    release = Event()
    sampled: list[str] = []
    original_sample = ImmutableEvidenceCorpus.sample

    def blocking_sample(
        corpus: ImmutableEvidenceCorpus,
        *,
        batch_size: int,
        rng: np.random.Generator,
        quotas: tuple[StratumQuota, ...] = (),
        current_policy_version: int | None = None,
    ) -> object:
        entered.set()
        if not release.wait(timeout=5):
            raise TimeoutError("test did not release immutable sampling")
        return original_sample(
            corpus,
            batch_size=batch_size,
            rng=rng,
            quotas=quotas,
            current_policy_version=current_policy_version,
        )

    monkeypatch.setattr(ImmutableEvidenceCorpus, "sample", blocking_sample)

    def run_sample() -> None:
        result = replay.sample(1)
        sampled.extend(record.incident.incident_id for record in result.records)

    sampler = Thread(target=run_sample)
    sampler.start()
    assert entered.wait(timeout=2)

    put_finished = Event()

    def run_put() -> None:
        replay.put(_record("published-during-sample"))
        put_finished.set()

    publisher = Thread(target=run_put)
    publisher.start()
    assert put_finished.wait(timeout=2), "sample held the actor publication lock"
    release.set()
    sampler.join(timeout=5)
    publisher.join(timeout=5)
    assert not sampler.is_alive()
    assert not publisher.is_alive()
    assert sampled == ["incident-sampled"]
    assert _incident_ids(replay) == (
        "incident-sampled",
        "incident-published-during-sample",
    )


def test_concurrent_publishers_are_linearizable_and_do_not_lose_records() -> None:
    replay = BoundedFailureCreditReplay(
        capacity=32,
        byte_capacity=100_000_000,
        seed=8,
    )
    records = tuple(_record(f"concurrent-{index}") for index in range(24))
    accepted: list[bool] = []

    def publish(subset: tuple[EvidenceRecord, ...]) -> None:
        for record in subset:
            accepted.append(replay.put(record))

    workers = tuple(Thread(target=publish, args=(records[offset::4],)) for offset in range(4))
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=10)
    assert not any(worker.is_alive() for worker in workers)
    assert accepted == [True] * len(records)
    assert len(replay) == len(records)
    assert set(_incident_ids(replay)) == {record.incident.incident_id for record in records}
    assert replay.metrics()["put_count"] == len(records)
