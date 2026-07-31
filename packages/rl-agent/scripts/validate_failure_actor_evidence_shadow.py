#!/usr/bin/env python3
"""Exercise every policy-relevant failure-credit path without learning.

The live frozen-policy shadow proves that real HeadlessSim DTOs cross the
semantics/pipeline boundary, but rare deadlock strata are not guaranteed to
occur in a short deterministic run.  This complementary contract probe feeds
reviewed DTO fixtures through the *production* semantics kernel, encoder,
detector and credit compiler.  It never constructs a model, learner or replay
and its output has no training authority.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np

from sts2_rl.artifacts import resolve_artifact_path
from sts2_rl.encoding import GroundedObservationEncoder
from sts2_rl.semantics import DecisionSemanticsKernel
from sts2_rl.training.config import FailureCreditConfig
from sts2_rl.training.failure_credit import (
    FAILURE_CREDIT_COLLECTOR_VERSION,
    FAILURE_CREDIT_COMPILER_VERSION,
    FAILURE_CREDIT_DETECTOR_VERSION,
    FAILURE_CREDIT_SCHEMA_VERSION,
    DirectPolicyTarget,
    EvidenceRecord,
    EvidenceStratum,
    FailureCreditEpisodePipeline,
    FailureCreditPipelineConfig,
    FailureOutcome,
)
from sts2_rl.training.launch_contract import current_formal_report_generation_source
from sts2_rl.training.learner import (
    LivenessLabelManifest,
    compile_liveness_label_manifest,
)

ACTOR_EVIDENCE_SHADOW_REPORT_VERSION: Final = "sts2-failure-actor-evidence-shadow-report-v3"
_CHECKOUT_ROOT: Final = Path(__file__).resolve().parents[3]
_CALIBRATION_PHASE: Final = "calibration_update_0"
_MATURE_PHASE: Final = "mature_risk_start"


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(
            descriptor,
            "w",
            encoding="utf-8",
            newline="\n",
        ) as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _event(
    page: str,
    *,
    hp: int = 60,
    revivals: int = 0,
    selected_count: int | None = None,
) -> dict[str, Any]:
    observation: dict[str, Any] = {
        "phase": "event",
        "decision_domain": "event",
        "run": {"act": 1, "floor": 8},
        "room": {
            "room_type": "event",
            "room_model_id": "ROOM_FULL_OF_CHEESE",
        },
        "event": {
            "event_id": "LINGER9",
            "page_id": page,
            "is_finished": False,
        },
        "player": {"hp": hp, "max_hp": 80},
        "_training": {
            "revivals_used": revivals,
            "player_hp_lost": max(0, 60 - hp),
        },
    }
    if selected_count is not None:
        observation["phase"] = "card_selection"
        observation["card_selection"] = {
            "prompt_id": "ROOM_FULL_OF_CHEESE.MULTI",
            "min_select": 0,
            "max_select": 2,
            "selected_count": selected_count,
            "remaining_select": 2 - selected_count,
            "selected_cards": ([] if selected_count == 0 else [{"card_id": "CARD.STRIKE", "upgrade_level": 0}]),
        }
    return observation


def _event_actions() -> tuple[dict[str, Any], ...]:
    return (
        {
            "action_handle": "loop",
            "model_action_kind": "event_option",
            "kind": "event_option",
            "option_id": "LOOP",
        },
        {
            "action_handle": "exit",
            "model_action_kind": "event_option",
            "kind": "event_option",
            "option_id": "EXIT",
        },
    )


def _forced_proceed() -> tuple[dict[str, Any], ...]:
    return (
        {
            "action_handle": "proceed",
            "model_action_kind": "proceed",
            "kind": "proceed",
        },
    )


def _selection_actions(operation: str) -> tuple[dict[str, Any], ...]:
    inverse = "deselect" if operation == "select" else "select"
    return (
        {
            "action_handle": f"{operation}:strike",
            "model_action_kind": "card_selection",
            "model_action_variant": operation,
            "selection_operation": operation,
            "kind": f"{operation}_card",
            "card": {
                "card_id": "CARD.STRIKE",
                "card_instance_id": "strike-instance",
                "upgrade_level": 0,
            },
        },
        {
            "action_handle": f"{inverse}:defend",
            "model_action_kind": "card_selection",
            "model_action_variant": inverse,
            "selection_operation": inverse,
            "kind": f"{inverse}_card",
            "card": {
                "card_id": "CARD.DEFEND",
                "card_instance_id": "defend-instance",
                "upgrade_level": 0,
            },
        },
    )


def _pipeline(
    episode_id: str,
    *,
    learning_tail_steps: int = 8,
) -> tuple[FailureCreditEpisodePipeline, GroundedObservationEncoder]:
    kernel = DecisionSemanticsKernel()
    provenance = FailureCreditEpisodePipeline.build_provenance(
        run_id="formal-actor-evidence-shadow",
        game_version="reviewed-fixture-v1",
        environment_schema_version="reviewed-fixture-v1",
        policy_version=11,
        kernel=kernel,
    )
    return (
        FailureCreditEpisodePipeline(
            episode_id=episode_id,
            provenance=provenance,
            kernel=kernel,
            config=FailureCreditPipelineConfig(
                detector_window_steps=32,
                context_burn_in_steps=1,
                learning_tail_steps=learning_tail_steps,
                maximum_completion_controls=4,
                maximum_completion_bytes=16_777_216,
            ),
        ),
        GroundedObservationEncoder(),
    )


def _observe(
    pipeline: FailureCreditEpisodePipeline,
    encoder: GroundedObservationEncoder,
    *,
    step: int,
    before: Mapping[str, Any],
    actions: Sequence[Mapping[str, Any]],
    selected: int,
    after: Mapping[str, Any],
    after_actions: Sequence[Mapping[str, Any]],
) -> None:
    snapshot = encoder.encode(before, actions, device="cpu").snapshot
    enabled = int(np.count_nonzero(snapshot.action_mask))
    pipeline.observe_transition(
        episode_step=step,
        before_observation=before,
        before_legal_actions=actions,
        after_observation=after,
        after_legal_actions=after_actions,
        snapshot=snapshot,
        action_index=selected,
        behavior_log_probability=(0.0 if enabled == 1 else float(np.log(1.0 / enabled))),
        policy_version=11,
        pre_recurrent_state=np.full(8, step, dtype=np.float32),
        terminal=False,
    )


def _failure(
    pipeline: FailureCreditEpisodePipeline,
    failure_kind: str,
) -> EvidenceRecord:
    return pipeline.finalize(
        failure_kind=failure_kind,
        local_failure=True,
        terminal_succeeded=False,
    ).records[-1]


def _direct_cycle_case() -> EvidenceRecord:
    pipeline, encoder = _pipeline("actor-shadow:direct")
    linger = _event("LINGER9", hp=60, revivals=0)
    warning = _event("DEATH_WARNING", hp=0, revivals=1)
    transitions = (
        (linger, _event_actions(), warning, _forced_proceed()),
        (warning, _forced_proceed(), linger, _event_actions()),
        (linger, _event_actions(), warning, _forced_proceed()),
        (warning, _forced_proceed(), linger, _event_actions()),
    )
    for step, (before, actions, after, after_actions) in enumerate(transitions):
        _observe(
            pipeline,
            encoder,
            step=step,
            before=before,
            actions=actions,
            selected=0,
            after=after,
            after_actions=after_actions,
        )
    return _failure(pipeline, "noncombat_event_action_cycle")


def _selection_cycle_case() -> EvidenceRecord:
    pipeline, encoder = _pipeline("actor-shadow:selection")
    empty = _event("SELECT", selected_count=0)
    selected = _event("SELECT", selected_count=1)
    transitions = (
        (empty, _selection_actions("select"), selected, _selection_actions("deselect")),
        (selected, _selection_actions("deselect"), empty, _selection_actions("select")),
        (empty, _selection_actions("select"), selected, _selection_actions("deselect")),
        (selected, _selection_actions("deselect"), empty, _selection_actions("select")),
    )
    for step, (before, actions, after, after_actions) in enumerate(transitions):
        _observe(
            pipeline,
            encoder,
            step=step,
            before=before,
            actions=actions,
            selected=0,
            after=after,
            after_actions=after_actions,
        )
    return _failure(pipeline, "selection_action_cycle")


def _unique_stall_case() -> EvidenceRecord:
    pipeline, encoder = _pipeline("actor-shadow:unique")
    actions = _event_actions()
    current = _event("UNIQUE-0")
    for step in range(6):
        after = _event(f"UNIQUE-{step + 1}")
        _observe(
            pipeline,
            encoder,
            step=step,
            before=current,
            actions=actions,
            selected=step % 2,
            after=after,
            after_actions=actions,
        )
        current = after
    return _failure(pipeline, "noncombat_no_durable_progress")


def _abandoned_cycle_case() -> EvidenceRecord:
    pipeline, encoder = _pipeline(
        "actor-shadow:abandoned",
        learning_tail_steps=16,
    )
    actions = _event_actions()
    repeated = _event("P0")
    for step in range(2):
        _observe(
            pipeline,
            encoder,
            step=step,
            before=repeated,
            actions=actions,
            selected=0,
            after=repeated,
            after_actions=actions,
        )
    current = repeated
    for step in range(2, 10):
        after = _event(f"ABANDONED-{step}")
        _observe(
            pipeline,
            encoder,
            step=step,
            before=current,
            actions=actions,
            selected=step % 2,
            after=after,
            after_actions=actions,
        )
        current = after
    return _failure(pipeline, "noncombat_no_durable_progress")


def _forced_stall_case() -> EvidenceRecord:
    pipeline, encoder = _pipeline("actor-shadow:forced")
    actions = _forced_proceed()
    current = _event("FORCED-0")
    for step in range(4):
        after = _event(f"FORCED-{step + 1}")
        _observe(
            pipeline,
            encoder,
            step=step,
            before=current,
            actions=actions,
            selected=0,
            after=after,
            after_actions=actions,
        )
        current = after
    return _failure(pipeline, "forced_no_durable_progress")


def _censored_case() -> EvidenceRecord:
    pipeline, encoder = _pipeline("actor-shadow:censored")
    actions = _event_actions()
    _observe(
        pipeline,
        encoder,
        step=0,
        before=_event("CENSORED-0"),
        actions=actions,
        selected=0,
        after=_event("CENSORED-1"),
        after_actions=actions,
    )
    return pipeline.finalize(
        failure_kind="transport_boundary",
        local_failure=False,
        terminal_succeeded=False,
    ).records[-1]


def _manifest_summary(
    manifest: LivenessLabelManifest,
) -> dict[str, Any]:
    """Return a stable, hash-free summary of the production learner masks."""

    context_ordinals: dict[str, int] = {}

    def context_ordinal(context_id: str) -> int:
        ordinal = context_ordinals.get(context_id)
        if ordinal is None:
            ordinal = len(context_ordinals)
            context_ordinals[context_id] = ordinal
        return ordinal

    rows = [
        {
            "context_ordinal": context_ordinal(row.context_id),
            "step_index": row.step_index,
            "forced": row.forced,
            "censored": row.censored,
            "legal_candidates": row.legal_candidates,
            "fresh": row.fresh,
            "actor_eligible": row.actor_eligible,
            "value_critic_requested": row.value_critic_requested,
            "q_critic_requested": row.q_critic_requested,
            "risk_actor_requested": row.risk_actor_requested,
            "risk_actor_mask": row.risk_actor_mask,
            "effective_risk_actor": row.effective_risk_actor,
            "direct_target": row.direct_target,
            "direct_actor_mask": row.direct_actor_mask,
            "effective_direct_actor": row.effective_direct_actor,
        }
        for row in manifest.rows
    ]

    def group_summary(group: Any) -> dict[str, Any]:
        return {
            "kind": group.kind,
            "row_keys": [
                {
                    "context_ordinal": context_ordinal(context_id),
                    "step_index": step_index,
                }
                for context_id, step_index in group.row_keys
            ],
            "fresh": group.fresh,
            "effective": group.effective,
        }

    cycle_groups = [group_summary(group) for group in manifest.cycle_groups]
    contrast_groups = [group_summary(group) for group in manifest.contrast_groups]
    return {
        "learner_update": manifest.learner_update,
        "calibration_active": manifest.calibration_active,
        "risk_actor_enabled": manifest.risk_actor_enabled,
        "mask_counts": {
            "rows": len(rows),
            "actor_eligible_rows": sum(int(row["actor_eligible"]) for row in rows),
            "direct_requested_rows": sum(int(row["direct_target"] is not None) for row in rows),
            "direct_masked_rows": sum(int(row["direct_actor_mask"]) for row in rows),
            "effective_direct_rows": sum(int(row["effective_direct_actor"]) for row in rows),
            "risk_requested_rows": sum(int(row["risk_actor_requested"]) for row in rows),
            "risk_masked_rows": sum(int(row["risk_actor_mask"]) for row in rows),
            "effective_risk_rows": sum(int(row["effective_risk_actor"]) for row in rows),
            "cycle_groups": len(cycle_groups),
            "effective_cycle_groups": sum(int(group["effective"]) for group in cycle_groups),
            "contrast_groups": len(contrast_groups),
            "effective_contrast_groups": sum(int(group["effective"]) for group in contrast_groups),
        },
        "suppression_counts": {
            "policy_lag": manifest.policy_lag_suppressed_labels,
            "risk_actor_phase": (manifest.risk_actor_phase_suppressed_labels),
        },
        "work": {
            "contexts": manifest.work.contexts,
            "steps": manifest.work.steps,
            "candidates": manifest.work.candidates,
            "autograd_segments": manifest.work.autograd_segments,
        },
        "rows": rows,
        "cycle_groups": cycle_groups,
        "contrast_groups": contrast_groups,
    }


def _effective_actor_count(summary: Mapping[str, Any]) -> int:
    counts = summary["mask_counts"]
    if not isinstance(counts, Mapping):  # pragma: no cover - local invariant
        raise TypeError("manifest mask counts have the wrong type")
    return sum(
        int(counts[key])
        for key in (
            "effective_direct_rows",
            "effective_risk_rows",
            "effective_cycle_groups",
            "effective_contrast_groups",
        )
    )


def _compile_learner_mask_dry_run(
    cases: Mapping[str, EvidenceRecord],
) -> tuple[dict[str, Any], dict[str, bool]]:
    """Compile the real learner masks at both reviewed phase boundaries."""

    config = FailureCreditConfig(mode="learning")
    policy_versions = {record.plan.provenance.policy_version for record in cases.values()}
    if len(policy_versions) != 1:
        raise ValueError("actor-evidence shadow cases do not share one policy version")
    current_policy_version = next(iter(policy_versions))
    phase_updates = {
        _CALIBRATION_PHASE: 0,
        _MATURE_PHASE: config.liveness_risk_actor_start_update,
    }
    manifests: dict[str, dict[str, dict[str, Any]]] = {name: {} for name in cases}
    batch_summaries: dict[str, dict[str, Any]] = {}
    plans = tuple(record.plan for record in cases.values())
    for phase, learner_update in phase_updates.items():
        batch_summaries[phase] = _manifest_summary(
            compile_liveness_label_manifest(
                plans,
                config=config,
                current_policy_version=current_policy_version,
                current_learner_update=learner_update,
            )
        )
        for name, record in cases.items():
            manifests[name][phase] = _manifest_summary(
                compile_liveness_label_manifest(
                    (record.plan,),
                    config=config,
                    current_policy_version=current_policy_version,
                    current_learner_update=learner_update,
                )
            )

    direct_calibration = manifests["linger9_death_warning_direct"][_CALIBRATION_PHASE]
    direct_mature = manifests["linger9_death_warning_direct"][_MATURE_PHASE]
    multi_calibration = manifests["room_full_of_cheese_multi_edge"][_CALIBRATION_PHASE]
    multi_mature = manifests["room_full_of_cheese_multi_edge"][_MATURE_PHASE]
    unique_calibration = manifests["unique_unresolved_stall"][_CALIBRATION_PHASE]
    unique_mature = manifests["unique_unresolved_stall"][_MATURE_PHASE]
    abandoned_calibration = manifests["abandoned_cycle_suffix"][_CALIBRATION_PHASE]
    abandoned_mature = manifests["abandoned_cycle_suffix"][_MATURE_PHASE]
    forced_manifests = manifests["forced_only_stall"].values()
    censored_manifests = manifests["censored_boundary"].values()

    direct_effective = all(
        int(summary["mask_counts"]["effective_direct_rows"]) > 0
        for summary in (
            direct_calibration,
            direct_mature,
        )
    )
    multi_effective = all(
        int(summary["mask_counts"]["effective_cycle_groups"]) > 0
        for summary in (
            multi_calibration,
            multi_mature,
        )
    )
    unique_requested = int(unique_calibration["mask_counts"]["risk_requested_rows"])
    unique_risk_calibration_suppressed = (
        unique_requested > 0
        and int(unique_calibration["mask_counts"]["effective_risk_rows"]) == 0
        and int(unique_calibration["suppression_counts"]["risk_actor_phase"]) == unique_requested
    )
    unique_risk_mature_effective = (
        int(unique_mature["mask_counts"]["effective_risk_rows"])
        == int(unique_mature["mask_counts"]["risk_requested_rows"])
        > 0
    )
    abandoned_no_direct_or_cycle = all(
        int(summary["mask_counts"]["effective_direct_rows"]) == 0
        and int(summary["mask_counts"]["effective_cycle_groups"]) == 0
        for summary in (
            abandoned_calibration,
            abandoned_mature,
        )
    )
    forced_and_censored_zero_effective_actor = all(
        _effective_actor_count(summary) == 0
        for summary in (
            *forced_manifests,
            *censored_manifests,
        )
    )
    phase_boundary_correct = (
        bool(batch_summaries[_CALIBRATION_PHASE]["calibration_active"])
        and not bool(batch_summaries[_CALIBRATION_PHASE]["risk_actor_enabled"])
        and not bool(batch_summaries[_MATURE_PHASE]["calibration_active"])
        and bool(batch_summaries[_MATURE_PHASE]["risk_actor_enabled"])
    )
    checks = {
        "learner_direct_witness_effective_during_and_after_calibration": (direct_effective),
        "learner_multi_edge_cycle_effective_during_and_after_calibration": (multi_effective),
        "learner_unique_risk_suppressed_during_calibration": (unique_risk_calibration_suppressed),
        "learner_unique_risk_effective_at_mature_start": (unique_risk_mature_effective),
        "learner_abandoned_cycle_has_no_effective_direct_or_cycle_blame": (abandoned_no_direct_or_cycle),
        "learner_forced_and_censored_have_zero_effective_actor": (forced_and_censored_zero_effective_actor),
        "learner_phase_boundary_correct": phase_boundary_correct,
    }
    return (
        {
            "interface": ("sts2_rl.training.learner.compile_liveness_label_manifest"),
            "pure_dry_run": True,
            "model_constructed": False,
            "optimizer_constructed": False,
            "replay_constructed": False,
            "current_policy_version": current_policy_version,
            "config": {
                "policy_gradient_max_lag": (config.policy_gradient_max_lag),
                "liveness_head_calibration_updates": (config.liveness_head_calibration_updates),
                "liveness_risk_actor_start_update": (config.liveness_risk_actor_start_update),
            },
            "phase_updates": phase_updates,
            "checks": checks,
            "batch_manifests": batch_summaries,
            "case_manifests": manifests,
        },
        checks,
    )


def build_actor_evidence_shadow_report() -> dict[str, Any]:
    """Return a deterministic report over formal production-pipeline cases."""

    cases = {
        "linger9_death_warning_direct": _direct_cycle_case(),
        "room_full_of_cheese_multi_edge": _selection_cycle_case(),
        "unique_unresolved_stall": _unique_stall_case(),
        "abandoned_cycle_suffix": _abandoned_cycle_case(),
        "forced_only_stall": _forced_stall_case(),
        "censored_boundary": _censored_case(),
    }
    strata: Counter[str] = Counter()
    direct_targets: Counter[str] = Counter()
    case_payloads: dict[str, Any] = {}
    for name, record in cases.items():
        for stratum in record.plan.strata:
            strata[stratum.value] += 1
        for target in record.plan.direct_policy_targets:
            direct_targets[target.target.value] += 1
        case_payloads[name] = {
            "outcome": record.incident.outcome.value,
            "strata": [stratum.value for stratum in record.plan.strata],
            "actor_label_count": record.plan.actor_label_count,
            "direct_actor_label_count": record.plan.direct_actor_label_count,
            "risk_actor_candidate_count": record.plan.risk_actor_candidate_count,
            "target_count": record.plan.target_count,
            "direct_policy_targets": [
                {
                    "step_index": target.step_index,
                    "target": target.target.value,
                }
                for target in record.plan.direct_policy_targets
            ],
            "cycle_policy_target_count": len(record.plan.cycle_policy_targets),
            "risk_sequence_count": len(record.plan.risk_sequences),
            "liveness_value_target_count": len(record.plan.liveness_value_targets),
            "liveness_q_target_count": len(record.plan.liveness_q_targets),
            "context_steps": len(record.plan.context.steps),
            "burn_in_steps": record.plan.context.burn_in_steps,
        }

    direct = cases["linger9_death_warning_direct"]
    multi = cases["room_full_of_cheese_multi_edge"]
    unique = cases["unique_unresolved_stall"]
    abandoned = cases["abandoned_cycle_suffix"]
    forced = cases["forced_only_stall"]
    censored = cases["censored_boundary"]
    learner_mask_dry_run, learner_mask_checks = _compile_learner_mask_dry_run(cases)
    gates = {
        "direct_avoid_nonzero": (
            len(direct.plan.direct_policy_targets) > 0
            and all(target.target is DirectPolicyTarget.AVOID for target in direct.plan.direct_policy_targets)
        ),
        "multi_edge_cycle_nonzero": bool(multi.plan.cycle_policy_targets),
        "unresolved_stall_nonzero": (
            EvidenceStratum.UNRESOLVED_STALL in unique.plan.strata
            and unique.incident.outcome is FailureOutcome.DEADLOCK_STALL
        ),
        "no_last_action_fallback": (not unique.plan.direct_policy_targets and not unique.plan.cycle_policy_targets),
        "abandoned_cycle_has_zero_direct_or_cycle_blame": (
            abandoned.incident.outcome is FailureOutcome.DEADLOCK_STALL
            and abandoned.plan.direct_actor_label_count == 0
            and not abandoned.plan.cycle_policy_targets
        ),
        "forced_only_has_zero_actor_blame": forced.plan.actor_label_count == 0,
        "censored_has_zero_learning_targets": (
            censored.incident.outcome is FailureOutcome.CENSORED and censored.plan.target_count == 0
        ),
        "no_generic_prefer_target": direct_targets[DirectPolicyTarget.PREFER.value] == 0,
        "all_contexts_bounded": all(len(record.plan.context.steps) <= 17 for record in cases.values()),
        "learner_mask_dry_run_passed": all(learner_mask_checks.values()),
    }
    return {
        "version": ACTOR_EVIDENCE_SHADOW_REPORT_VERSION,
        "status": "passed" if all(gates.values()) else "failed",
        "read_only": True,
        "training_authority": False,
        "fixture_authority": "reviewed-contract-cases-only",
        "abi": {
            "schema": FAILURE_CREDIT_SCHEMA_VERSION,
            "collector": FAILURE_CREDIT_COLLECTOR_VERSION,
            "detector": FAILURE_CREDIT_DETECTOR_VERSION,
            "compiler": FAILURE_CREDIT_COMPILER_VERSION,
        },
        "gates": gates,
        "counts": {
            "cases": len(cases),
            "actor_actionable_records": sum(int(record.plan.actor_label_count > 0) for record in cases.values()),
            "risk_sequence_records": sum(int(bool(record.plan.risk_sequences)) for record in cases.values()),
            "strata": dict(sorted(strata.items())),
            "direct_targets": dict(sorted(direct_targets.items())),
        },
        "cases": case_payloads,
        "learner_mask_dry_run": learner_mask_dry_run,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    generation_source = current_formal_report_generation_source(_CHECKOUT_ROOT, __file__)
    report = build_actor_evidence_shadow_report()
    if current_formal_report_generation_source(_CHECKOUT_ROOT, __file__) != generation_source:
        raise RuntimeError("formal actor-evidence source changed during report generation")
    report["generation_source"] = generation_source
    path = resolve_artifact_path(args.report)
    _atomic_json(path, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "report": os.fspath(path),
                "counts": report["counts"],
            },
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
