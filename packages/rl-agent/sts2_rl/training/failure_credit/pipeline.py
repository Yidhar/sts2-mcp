"""Collector-side evidence pipeline for failure-credit v6.

This module is intentionally separate from the legacy transaction-v3 replay
path.  It consumes the reviewed decision-semantics kernel online, while the
transition is still available.  Detector-confirmed cycle records are drainable
as soon as their complete repeated period is observed; terminal/censored and
completion records remain owned by the episode boundary.

The important causality rule is structural rather than heuristic:

* semantic loop edges and their supporting episode steps are captured when a
  repeated macro-edge cycle is observed;
* replay context is used only to reproduce model activations, never to
  rediscover a loop;
* a unique no-progress stall produces sequence value/cost targets but never a
  fallback blame label for the last action;
* forced suffixes belong to their initiating policy macro-edge, so detector
  attribution remains on the factual policy choice.
"""

from __future__ import annotations

import hashlib
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Final

import numpy as np
import numpy.typing as npt

from sts2_rl.encoding import EncodedDecisionSnapshot
from sts2_rl.semantics import (
    DECISION_IDENTITY_CONTRACT_VERSION,
    DecisionSemantics,
    DecisionSemanticsKernel,
    ForcedTransition,
    MacroEdge,
    MacroEdgeBuilder,
    MacroEdgeOutcome,
    ProgressKind,
    ProgressReceipt,
    SemanticKey,
)

from .compiler import CreditCompiler
from .contracts import (
    CreditProvenance,
    FailureIncident,
    FailureOutcome,
    LearningContext,
    LearningStep,
    LoopEdgeEvidence,
    PolicyWitness,
    TargetAuthority,
    WitnessKind,
)
from .corpus import EvidenceRecord, evidence_record_storage_nbytes

FAILURE_CREDIT_DETECTOR_VERSION: Final = "sts2-semantic-macro-cycle-detector-v4"
FAILURE_CREDIT_COLLECTOR_VERSION: Final = "sts2-failure-credit-collector-v6"


def _positive_integer(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    if value <= 0:
        raise ValueError(f"{label} must be positive")
    return value


def _non_negative_integer(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    if value < 0:
        raise ValueError(f"{label} must be non-negative")
    return value


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _stable_id(namespace: str, *parts: object) -> str:
    payload = "\0".join(str(part) for part in parts)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"{namespace}:{digest}"


def _semantic_identity(key: SemanticKey) -> tuple[str, str, str, bytes]:
    """Use the full collision-auditable key, not a digest-only shortcut."""

    key.verify()
    return (
        key.namespace,
        key.schema_version,
        key.digest,
        key.canonical_payload,
    )


def _owned_recurrent_state(value: npt.ArrayLike) -> npt.NDArray[np.float32]:
    state = np.asarray(value, dtype=np.float32)
    if state.ndim == 2 and state.shape[0] == 1:
        state = state[0]
    if state.ndim != 1 or not state.size:
        raise ValueError("pre-step recurrent state must be rank-1 (or [1, H]) and non-empty")
    if not np.all(np.isfinite(state)):
        raise ValueError("pre-step recurrent state must be finite")
    return np.ascontiguousarray(state).copy()


def _enabled_group_count(actions: Sequence[object]) -> int:
    enabled = 0
    for semantic_action in actions:
        group = getattr(semantic_action, "group", None)
        prototype = getattr(group, "prototype", None)
        if not isinstance(prototype, Mapping):
            raise TypeError("decision semantics exposed a malformed action group")
        if bool(prototype.get("is_enabled", prototype.get("enabled", True))):
            enabled += 1
    return enabled


@dataclass(frozen=True, slots=True)
class FailureCreditPipelineConfig:
    """Bounded collector memory and online detector contract."""

    detector_window_steps: int = 256
    context_burn_in_steps: int = 16
    learning_tail_steps: int = 64
    maximum_completion_controls: int = 32
    maximum_completion_bytes: int = 134_217_728

    def __post_init__(self) -> None:
        _positive_integer(
            self.detector_window_steps,
            label="detector_window_steps",
        )
        _non_negative_integer(
            self.context_burn_in_steps,
            label="context_burn_in_steps",
        )
        _positive_integer(
            self.learning_tail_steps,
            label="learning_tail_steps",
        )
        _positive_integer(
            self.maximum_completion_controls,
            label="maximum_completion_controls",
        )
        _positive_integer(
            self.maximum_completion_bytes,
            label="maximum_completion_bytes",
        )

    @property
    def retained_decisions(self) -> int:
        return self.context_burn_in_steps + max(self.detector_window_steps, self.learning_tail_steps) + 2

    @property
    def maximum_context_steps(self) -> int:
        return self.context_burn_in_steps + self.learning_tail_steps


@dataclass(frozen=True, slots=True)
class FailureCreditShadowMetrics:
    """Small per-episode funnel used before replay learning is enabled."""

    decisions: int
    progress_receipts: tuple[tuple[ProgressKind, int], ...]
    detected_cycles: int
    completion_controls: int
    completion_controls_observed: int
    completion_controls_dropped: int
    completion_storage_nbytes: int
    records: int
    censored_semantic_transitions: int
    streamed_records: int = 0
    # Distinguish "no one-edge self-loop was observed" from downstream direct
    # witness rejection. These detector-side facts survive into episode metrics.
    detected_direct_cycles: int = 0
    detected_multi_edge_cycles: int = 0


@dataclass(frozen=True, slots=True)
class FailureCreditEpisodeResult:
    records: tuple[EvidenceRecord, ...]
    metrics: FailureCreditShadowMetrics


@dataclass(frozen=True, slots=True)
class _CapturedDecision:
    step: LearningStep
    pre_recurrent_state: npt.NDArray[np.float32]
    receipt: ProgressReceipt


@dataclass(slots=True)
class _OpenMacro:
    builder: MacroEdgeBuilder
    source_step: int
    source_spec_id: str
    direct_credit_allowed: bool
    overflowed: bool = False
    forced_count: int = 0


@dataclass(frozen=True, slots=True)
class _ClosedMacro:
    edge: MacroEdge
    source_spec_id: str
    direct_credit_allowed: bool
    progress_epoch: int

    @property
    def signature(self) -> tuple[object, ...]:
        destination = self.edge.destination_node
        return (
            _semantic_identity(self.edge.anchor),
            _semantic_identity(self.edge.source_node.loop),
            _semantic_identity(self.edge.chosen_action.loop),
            None if destination is None else _semantic_identity(destination.loop),
        )


@dataclass(frozen=True, slots=True)
class _DetectedCycle:
    kind: WitnessKind
    cycle_span: int
    supporting_episode_steps: tuple[int, ...]
    attributed_episode_steps: tuple[int, ...]
    loop_edges: tuple[LoopEdgeEvidence, ...]
    behavior_mean_log_probability: float
    progress_epoch: int


@dataclass(frozen=True, slots=True)
class _PendingCompletion:
    context: LearningContext
    scope_key: str
    failure_kind: str
    progress_epoch: int


@dataclass(frozen=True, slots=True)
class _StagedCompletion:
    priority: int
    storage_nbytes: int
    record: EvidenceRecord


class _StaleEvidenceWindow(RuntimeError):
    """Detector evidence cannot fit in the bounded recurrent context."""


class FailureCreditEpisodePipeline:
    """One-episode online semantic evidence builder.

    The instance is actor-local and deliberately has no replay reference or
    learner dependency.  ``finalize`` is one-shot so an episode cannot publish
    a partially different second interpretation of the same detector facts.
    """

    def __init__(
        self,
        *,
        episode_id: str,
        provenance: CreditProvenance,
        config: FailureCreditPipelineConfig | None = None,
        kernel: DecisionSemanticsKernel | None = None,
        compiler: CreditCompiler | None = None,
    ) -> None:
        self.episode_id = _text(episode_id, label="episode_id")
        if not isinstance(provenance, CreditProvenance):
            raise TypeError("provenance must be CreditProvenance")
        self.provenance = provenance
        self.config = config or FailureCreditPipelineConfig()
        self.kernel = kernel or DecisionSemanticsKernel()
        self.compiler = compiler or CreditCompiler()
        manifest = self.kernel.registry.manifest_key(self.kernel.key_index)
        expected_adapter = f"{manifest.schema_version}:{manifest.digest}"
        if provenance.identity_version != DECISION_IDENTITY_CONTRACT_VERSION:
            raise ValueError("failure-credit provenance identity contract differs from the semantics kernel")
        if provenance.adapter_version != expected_adapter:
            raise ValueError("failure-credit provenance adapter manifest differs from the semantics kernel")
        if provenance.detector_version != FAILURE_CREDIT_DETECTOR_VERSION:
            raise ValueError("failure-credit provenance detector version differs from the pipeline")
        if provenance.collector_version != FAILURE_CREDIT_COLLECTOR_VERSION:
            raise ValueError("failure-credit provenance collector version differs from the pipeline")

        self._captured: deque[_CapturedDecision] = deque(
            maxlen=self.config.retained_decisions,
        )
        self._current_semantics: DecisionSemantics | None = None
        self._active_macro: _OpenMacro | None = None
        self._macro_history: deque[_ClosedMacro] = deque()
        self._latest_cycle: _DetectedCycle | None = None
        self._terminal_macro: _ClosedMacro | None = None
        self._completion_records: list[_StagedCompletion] = []
        self._ready_records: list[EvidenceRecord] = []
        self._streamed_records = 0
        self._emitted_cycle_keys: set[str] = set()
        self._emitted_cycle_supports: set[tuple[int, ...]] = set()
        self._completion_controls_observed = 0
        self._receipt_counts: dict[ProgressKind, int] = {}
        self._decision_count = 0
        self._detected_cycles = 0
        self._detected_direct_cycles = 0
        self._detected_multi_edge_cycles = 0
        self._progress_epoch = 0
        self._semantic_censored_transitions = 0
        self._finalized = False

    @staticmethod
    def build_provenance(
        *,
        run_id: str,
        game_version: str,
        environment_schema_version: str,
        policy_version: int,
        kernel: DecisionSemanticsKernel | None = None,
    ) -> CreditProvenance:
        """Build provenance bound to the exact active semantics manifest."""

        active_kernel = kernel or DecisionSemanticsKernel()
        manifest = active_kernel.registry.manifest_key(active_kernel.key_index)
        return CreditProvenance(
            run_id=run_id,
            game_version=game_version,
            environment_schema_version=environment_schema_version,
            identity_version=DECISION_IDENTITY_CONTRACT_VERSION,
            detector_version=FAILURE_CREDIT_DETECTOR_VERSION,
            adapter_version=f"{manifest.schema_version}:{manifest.digest}",
            collector_version=FAILURE_CREDIT_COLLECTOR_VERSION,
            policy_version=policy_version,
        )

    def observe_transition(
        self,
        *,
        episode_step: int,
        before_observation: Mapping[str, Any],
        before_legal_actions: Sequence[Mapping[str, Any]],
        after_observation: Mapping[str, Any],
        after_legal_actions: Sequence[Mapping[str, Any]],
        snapshot: EncodedDecisionSnapshot,
        action_index: int,
        behavior_log_probability: float,
        policy_version: int,
        pre_recurrent_state: npt.ArrayLike,
        terminal: bool,
    ) -> ProgressReceipt:
        """Capture one accepted transition and update detector-time evidence."""

        if self._finalized:
            raise RuntimeError("cannot observe transitions after failure-credit finalization")
        if not isinstance(snapshot, EncodedDecisionSnapshot):
            raise TypeError("snapshot must be EncodedDecisionSnapshot")
        if not isinstance(terminal, bool):
            raise TypeError("terminal must be a boolean")

        before = self._current_semantics
        if before is None:
            before = self.kernel.identify(
                observation=before_observation,
                legal_actions=before_legal_actions,
            )
        after = self.kernel.identify(
            observation=after_observation,
            legal_actions=after_legal_actions,
            parent_scopes=before.scopes,
        )
        if len(before.actions) != snapshot.candidate_count:
            raise ValueError(
                "semantics/encoder candidate count mismatch: "
                f"semantics={len(before.actions)} encoder={snapshot.candidate_count}"
            )
        enabled_count = int(np.count_nonzero(snapshot.action_mask))
        if enabled_count <= 0:
            raise ValueError("failure-credit step has no enabled model candidate")
        step = LearningStep(
            decision_id=f"{self.episode_id}:{episode_step}",
            episode_step=episode_step,
            snapshot=snapshot,
            action_index=action_index,
            behavior_log_probability=behavior_log_probability,
            policy_version=policy_version,
            node=before.node,
            anchor=before.anchor,
            candidate_actions=tuple(action.identities for action in before.actions),
            forced=enabled_count == 1,
        )
        receipt = self.kernel.classify_transition(
            before=before,
            after=after,
            before_observation=before_observation,
            after_observation=after_observation,
        )
        self._receipt_counts[receipt.kind] = self._receipt_counts.get(receipt.kind, 0) + 1
        self._decision_count += 1
        self._captured.append(
            _CapturedDecision(
                step=step,
                pre_recurrent_state=_owned_recurrent_state(pre_recurrent_state),
                receipt=receipt,
            )
        )

        self._advance_macro(
            step=step,
            before=before,
            after=after,
            receipt=receipt,
            terminal=terminal,
        )
        if receipt.kind in {
            ProgressKind.FLOW_ADVANCE,
            ProgressKind.DURABLE_COMMIT,
        }:
            self._progress_epoch += 1
            self._macro_history.clear()
            self._latest_cycle = None
            self._emitted_cycle_keys.clear()
        elif receipt.kind is ProgressKind.UNKNOWN:
            # An unreviewed mutation is neither progress nor proof of a loop.
            # Split the evidence epoch and fail closed on direct attribution.
            self._semantic_censored_transitions += 1
            self._progress_epoch += 1
            self._macro_history.clear()
            self._latest_cycle = None
            self._emitted_cycle_keys.clear()
        self._current_semantics = after
        return receipt

    def _advance_macro(
        self,
        *,
        step: LearningStep,
        before: DecisionSemantics,
        after: DecisionSemantics,
        receipt: ProgressReceipt,
        terminal: bool,
    ) -> None:
        if step.actor_eligible:
            if self._active_macro is not None:
                raise RuntimeError("a new policy choice arrived before the prior macro edge closed")
            # A detector-time cycle is causal only while it remains a suffix of
            # the no-progress policy path.  Starting any later policy macro
            # invalidates the old witness immediately; if this new macro
            # actually continues the repeated period, `_register_cycle_candidate`
            # will establish a new suffix-ending witness when the macro closes.
            # This prevents an abandoned cycle from blaming an unrelated later
            # stall merely because both happened inside the same progress epoch.
            self._latest_cycle = None
            active_spec = self.kernel.registry.by_id(before.scopes.active.spec_id).spec
            self._active_macro = _OpenMacro(
                builder=MacroEdgeBuilder(
                    anchor=step.anchor,
                    source_node=step.node,
                    chosen_action=step.selected_action,
                    policy_step_id=step.episode_step,
                    legal_candidate_count=int(np.count_nonzero(step.snapshot.action_mask)),
                ),
                source_step=step.episode_step,
                source_spec_id=before.scopes.active.spec_id,
                direct_credit_allowed=active_spec.allows_direct_policy_credit,
            )
        elif self._active_macro is not None:
            self._active_macro.forced_count += 1
            if self._active_macro.forced_count > self.config.detector_window_steps:
                self._active_macro.overflowed = True
            else:
                self._active_macro.builder.append_forced(
                    ForcedTransition(
                        step_id=step.episode_step,
                        node=step.node,
                        action=step.selected_action,
                        receipt=receipt,
                    )
                )

        active = self._active_macro
        if active is None:
            return
        after_policy_choice = _enabled_group_count(after.actions) > 1
        if terminal:
            outcome = MacroEdgeOutcome.TERMINAL
        elif receipt.kind in {
            ProgressKind.FLOW_ADVANCE,
            ProgressKind.DURABLE_COMMIT,
        }:
            outcome = MacroEdgeOutcome.ANCHOR_ADVANCE
        elif receipt.kind is ProgressKind.UNKNOWN:
            outcome = MacroEdgeOutcome.CENSORED
        elif after_policy_choice:
            outcome = MacroEdgeOutcome.NEXT_POLICY
        else:
            return
        closed = _ClosedMacro(
            edge=active.builder.close(
                outcome=outcome,
                closing_step_id=step.episode_step,
                destination_node=after.node,
                destination_anchor=after.anchor,
            ),
            source_spec_id=active.source_spec_id,
            direct_credit_allowed=active.direct_credit_allowed and not active.overflowed,
            progress_epoch=self._progress_epoch,
        )
        self._active_macro = None
        if outcome is MacroEdgeOutcome.NEXT_POLICY and closed.direct_credit_allowed:
            self._register_cycle_candidate(closed)
        elif outcome is MacroEdgeOutcome.ANCHOR_ADVANCE:
            # Net damage or an ordinary combat phase tick resets the liveness
            # detector epoch, but is not a completed policy transaction.  In
            # particular it must not become an implicit "deal damage" reward.
            if receipt.source not in {
                "combat:net_enemy_hp_reduced",
                "combat:phase_or_wave_advanced",
            }:
                self._capture_completion(closed, receipt=receipt)
        elif outcome is MacroEdgeOutcome.TERMINAL:
            self._terminal_macro = closed

    def _register_cycle_candidate(self, macro: _ClosedMacro) -> None:
        self._macro_history.append(macro)
        last_step = macro.edge.closing_step_id
        while self._macro_history and (
            last_step - self._macro_history[0].edge.policy_step_id > self.config.detector_window_steps
        ):
            self._macro_history.popleft()
        history = tuple(self._macro_history)
        maximum_span = len(history) // 2
        for span in range(1, maximum_span + 1):
            previous = history[-2 * span : -span]
            current = history[-span:]
            if tuple(item.signature for item in previous) != tuple(item.signature for item in current):
                continue
            cycle_macros = (*previous, *current)
            supporting_steps = tuple(
                sorted(
                    {
                        step
                        for item in cycle_macros
                        for step in (
                            item.edge.policy_step_id,
                            *(forced.step_id for forced in item.edge.forced_suffix),
                        )
                    }
                )
            )
            if supporting_steps[-1] - supporting_steps[0] > self.config.detector_window_steps:
                continue
            cycle_span = current[0].edge.policy_step_id - previous[0].edge.policy_step_id
            if cycle_span <= 0:
                continue
            edge_support: dict[
                tuple[object, ...],
                tuple[SemanticKey, SemanticKey, list[int]],
            ] = {}
            for item in cycle_macros:
                node = item.edge.source_node.loop
                action = item.edge.chosen_action.loop
                identity = (_semantic_identity(node), _semantic_identity(action))
                entry = edge_support.get(identity)
                if entry is None:
                    edge_support[identity] = (
                        node,
                        action,
                        [item.edge.policy_step_id],
                    )
                else:
                    entry[2].append(item.edge.policy_step_id)
            loop_edges = tuple(
                LoopEdgeEvidence(
                    node=node,
                    action=action,
                    supporting_episode_steps=tuple(sorted(set(steps))),
                )
                for node, action, steps in edge_support.values()
            )
            if span == 1 and len(loop_edges) == 1:
                witness_kind = WitnessKind.DIRECT_WITNESS
            elif len(loop_edges) >= 2:
                witness_kind = WitnessKind.MULTI_EDGE_CYCLE
            else:
                # The current witness ABI intentionally refuses a nominal
                # multi-edge cycle that collapses to one coarse policy edge.
                continue
            attributed_steps = tuple(item.edge.policy_step_id for item in current)
            log_probabilities = tuple(
                self._captured_step(step).step.behavior_log_probability for step in attributed_steps
            )
            self._latest_cycle = _DetectedCycle(
                kind=witness_kind,
                cycle_span=cycle_span,
                supporting_episode_steps=supporting_steps,
                attributed_episode_steps=attributed_steps,
                loop_edges=loop_edges,
                behavior_mean_log_probability=float(sum(log_probabilities) / len(log_probabilities)),
                progress_epoch=self._progress_epoch,
            )
            self._detected_cycles += 1
            if witness_kind is WitnessKind.DIRECT_WITNESS:
                self._detected_direct_cycles += 1
            else:
                self._detected_multi_edge_cycles += 1
            cycle_key = _stable_id(
                "stream-cycle",
                self.episode_id,
                self._progress_epoch,
                witness_kind.value,
                *(item.signature for item in current),
            )
            if cycle_key not in self._emitted_cycle_keys:
                # The two complete, suffix-ending periods are already
                # detector-authoritative local evidence.  Publishing this
                # record must not wait for a later 256-step/episode boundary:
                # exploration may escape the loop and a long full run would
                # otherwise make the factual behavior policy stale.
                self._ready_records.append(self._failure_record("detector_confirmed_semantic_cycle"))
                self._emitted_cycle_keys.add(cycle_key)
                self._emitted_cycle_supports.add(supporting_steps)
            return

    def drain_ready_records(self) -> tuple[EvidenceRecord, ...]:
        """Drain fresh detector-authoritative records exactly once.

        The method is actor-local and non-blocking.  A caller may forward the
        returned immutable values to replay between recurrent unrolls.  When no
        streaming sink is installed, ``finalize`` retains and returns the same
        records, preserving shadow/offline behavior.
        """

        if self._finalized:
            raise RuntimeError("cannot drain failure-credit records after finalization")
        records = tuple(self._ready_records)
        self._ready_records.clear()
        self._streamed_records += len(records)
        return records

    def _captured_step(self, episode_step: int) -> _CapturedDecision:
        for captured in self._captured:
            if captured.step.episode_step == episode_step:
                return captured
        raise RuntimeError("detector evidence references a decision outside retained collector context")

    def _context(
        self,
        *,
        relevant_episode_steps: tuple[int, ...],
        tail_only: bool = False,
    ) -> LearningContext:
        if not self._captured:
            raise RuntimeError("cannot build failure-credit context without decisions")
        captured = tuple(self._captured)
        context_limit = self.config.maximum_context_steps
        end = len(captured)
        if tail_only:
            learn_count = min(self.config.learning_tail_steps, end)
            learn_start = end - learn_count
        else:
            if not relevant_episode_steps:
                raise ValueError("relevant_episode_steps must not be empty")
            positions = {item.step.episode_step: index for index, item in enumerate(captured)}
            missing = set(relevant_episode_steps) - set(positions)
            if missing:
                raise _StaleEvidenceWindow(
                    "detector evidence fell outside retained context: " f"missing={sorted(missing)}"
                )
            learn_start = min(positions[step] for step in relevant_episode_steps)
            if end - learn_start > self.config.learning_tail_steps:
                raise _StaleEvidenceWindow("detector evidence is too old for the bounded learning tail")
        start = max(0, learn_start - self.config.context_burn_in_steps)
        if end - start > context_limit:
            start = end - context_limit
        if start > learn_start:  # pragma: no cover - guarded above
            raise _StaleEvidenceWindow("detector target cannot fit in the bounded recurrent context")
        selected = captured[start:]
        burn_in_steps = learn_start - start
        if len(selected) > context_limit:  # pragma: no cover - arithmetic invariant
            raise RuntimeError("failure-credit context exceeded its configured bound")
        first = selected[0]
        context_id = _stable_id(
            "failure-context",
            self.episode_id,
            first.step.episode_step,
            selected[-1].step.episode_step,
            burn_in_steps,
        )
        return LearningContext(
            context_id=context_id,
            episode_id=self.episode_id,
            start_step=first.step.episode_step,
            initial_recurrent_state=first.pre_recurrent_state,
            steps=tuple(item.step for item in selected),
            burn_in_steps=burn_in_steps,
        )

    def _completion_context(self, *, policy_episode_step: int) -> LearningContext:
        """Retain burn-in plus only the initiating factual policy decision.

        Completion controls are critic-only.  Forced acknowledgement suffixes
        prove that the transition completed, but they are not policy choices
        and must not turn every ordinary completion into a 256-step recurrent
        payload.  The context therefore ends at the initiating policy step and
        exposes exactly one learnable Q/value target.
        """

        captured = tuple(self._captured)
        positions = {item.step.episode_step: index for index, item in enumerate(captured)}
        target = positions.get(policy_episode_step)
        if target is None:
            raise _StaleEvidenceWindow("completion initiator fell outside retained recurrent context")
        start = max(0, target - self.config.context_burn_in_steps)
        selected = captured[start : target + 1]
        first = selected[0]
        context_id = _stable_id(
            "completion-context",
            self.episode_id,
            first.step.episode_step,
            policy_episode_step,
            target - start,
        )
        return LearningContext(
            context_id=context_id,
            episode_id=self.episode_id,
            start_step=first.step.episode_step,
            initial_recurrent_state=first.pre_recurrent_state,
            steps=tuple(item.step for item in selected),
            burn_in_steps=target - start,
        )

    def _capture_completion(
        self,
        macro: _ClosedMacro,
        *,
        receipt: ProgressReceipt,
    ) -> None:
        try:
            context = self._completion_context(
                policy_episode_step=macro.edge.policy_step_id,
            )
        except _StaleEvidenceWindow:
            # A completion whose initiating policy state is no longer
            # reproducible cannot provide a factual candidate-Q control.
            self._completion_controls_observed += 1
            return
        # A completed transition is a factual zero-liveness-cost critic
        # control.  Since v20 it never becomes a PREFER actor label; page
        # teardown, Cancel and Confirm alike remain critic-only evidence.
        pending = _PendingCompletion(
            context=context,
            scope_key=self._scope_key(macro.edge.anchor),
            failure_kind=(
                "verified_flow_completion" if receipt.kind is ProgressKind.FLOW_ADVANCE else "verified_durable_commit"
            ),
            progress_epoch=self._progress_epoch,
        )
        self._completion_controls_observed += 1
        self._stage_completion(self._completion_record(pending))

    def _stage_completion(self, record: EvidenceRecord) -> None:
        """Keep a deterministic, count+byte bounded completion reservoir.

        At most one completion per semantic scope is retained at a time.  A
        stable bottom-k priority makes the retained controls independent of
        wall-clock scheduling while reserving one slot for each observed
        completion kind before filling the remaining global capacity.
        """

        size = evidence_record_storage_nbytes(record)
        priority = int.from_bytes(
            hashlib.sha256(
                (
                    f"{self.episode_id}\0{record.incident.scope_key}\0"
                    f"{record.incident.failure_kind}\0{record.incident.incident_id}"
                ).encode()
            ).digest()[:8],
            byteorder="big",
            signed=False,
        )
        candidate = _StagedCompletion(
            priority=priority,
            storage_nbytes=size,
            record=record,
        )
        by_scope: dict[str, _StagedCompletion] = {}
        for item in (*self._completion_records, candidate):
            scope = item.record.incident.scope_key
            previous = by_scope.get(scope)
            if previous is None or (item.priority, item.record.incident.incident_id) < (
                previous.priority,
                previous.record.incident.incident_id,
            ):
                by_scope[scope] = item
        candidates = sorted(
            by_scope.values(),
            key=lambda item: (item.priority, item.record.incident.incident_id),
        )
        selected: list[_StagedCompletion] = []
        selected_ids: set[str] = set()
        storage_nbytes = 0

        # Flow exits and durable in-surface commits are distinct controls.
        # Reserve one representative of each kind when it fits.
        for kind in ("verified_flow_completion", "verified_durable_commit"):
            representative = next(
                (item for item in candidates if item.record.incident.failure_kind == kind),
                None,
            )
            if (
                representative is not None
                and len(selected) < self.config.maximum_completion_controls
                and storage_nbytes + representative.storage_nbytes <= self.config.maximum_completion_bytes
            ):
                selected.append(representative)
                selected_ids.add(representative.record.incident.incident_id)
                storage_nbytes += representative.storage_nbytes

        for item in candidates:
            incident_id = item.record.incident.incident_id
            if incident_id in selected_ids:
                continue
            if len(selected) >= self.config.maximum_completion_controls:
                break
            if storage_nbytes + item.storage_nbytes > self.config.maximum_completion_bytes:
                continue
            selected.append(item)
            selected_ids.add(incident_id)
            storage_nbytes += item.storage_nbytes
        self._completion_records = sorted(
            selected,
            key=lambda item: (
                item.record.incident.context.steps[-1].episode_step,
                item.record.incident.incident_id,
            ),
        )

    @staticmethod
    def _scope_key(anchor: SemanticKey) -> str:
        return f"{anchor.namespace}:{anchor.schema_version}:{anchor.digest}"

    def _completion_record(
        self,
        completion: _PendingCompletion,
    ) -> EvidenceRecord:
        provenance = self._provenance_for_context(completion.context)
        incident = FailureIncident(
            incident_id=_stable_id(
                "failure-incident",
                completion.context.context_id,
                completion.failure_kind,
            ),
            scope_key=completion.scope_key,
            failure_kind=completion.failure_kind,
            outcome=FailureOutcome.COMPLETED,
            task_authority=TargetAuthority.CENSORED,
            local_authority=TargetAuthority.VERIFIED_TRANSITION,
            task_return=None,
            local_failure_cost=0.0,
            context=completion.context,
            witnesses=(),
            detector_window_steps=self.config.detector_window_steps,
            progress_epoch=completion.progress_epoch,
            provenance=provenance,
        )
        return EvidenceRecord(
            incident=incident,
            plan=self.compiler.compile(incident),
        )

    def _failure_record(self, failure_kind: str) -> EvidenceRecord:
        detected = self._latest_cycle
        context: LearningContext | None = None
        if detected is not None and detected.progress_epoch == self._progress_epoch:
            try:
                # The detector-time witness retains both repeated periods.
                # Recurrent replay needs only the actually attributed policy
                # decisions; the older supporting period remains immutable
                # provenance in ``loop_edges`` and must not make the learning
                # tensor unbounded.
                context = self._context(
                    relevant_episode_steps=(detected.attributed_episode_steps),
                )
            except _StaleEvidenceWindow:
                # A cycle followed by a long, non-repeating tail is not a
                # sufficiently local cause of the terminal stall.  Preserve
                # value/Q sequence risk, but do not emit direct actor blame.
                detected = None
        if detected is not None and context is not None:
            context_index = {step.episode_step: index for index, step in enumerate(context.steps)}
            attributed_indices = tuple(context_index[step] for step in detected.attributed_episode_steps)
            witness = PolicyWitness(
                witness_id=_stable_id(
                    "cycle-witness",
                    self.episode_id,
                    detected.kind.value,
                    *detected.supporting_episode_steps,
                ),
                kind=detected.kind,
                attributed_step_indices=attributed_indices,
                supporting_episode_steps=detected.supporting_episode_steps,
                occurrences=2,
                cycle_span=detected.cycle_span,
                successor_confirmed=True,
                loop_edges=detected.loop_edges,
                behavior_mean_log_probability=(
                    detected.behavior_mean_log_probability if detected.kind is WitnessKind.MULTI_EDGE_CYCLE else None
                ),
            )
            outcome = FailureOutcome.DEADLOCK_CYCLE
            witnesses = (witness,)
        else:
            # A detector-confirmed no-progress window is a real local
            # liveness failure.  Without an observed repeated semantic edge it
            # carries only sequence-risk/value supervision; there is no
            # arbitrary "last non-forced action" fallback.
            context = self._context(relevant_episode_steps=(), tail_only=True)
            learn_indices = context.learn_step_indices
            support = tuple(context.steps[index].episode_step for index in learn_indices)
            witness = PolicyWitness(
                witness_id=_stable_id(
                    "risk-witness",
                    self.episode_id,
                    failure_kind,
                    support[0],
                    support[-1],
                ),
                kind=WitnessKind.RISK_SEQUENCE,
                attributed_step_indices=learn_indices,
                supporting_episode_steps=support,
                occurrences=1,
                cycle_span=None,
                successor_confirmed=True,
            )
            outcome = FailureOutcome.DEADLOCK_STALL
            witnesses = (witness,)
        provenance = self._provenance_for_context(context)
        incident = FailureIncident(
            incident_id=_stable_id(
                "failure-incident",
                self.episode_id,
                failure_kind,
                context.context_id,
            ),
            scope_key=self._scope_key(context.steps[-1].anchor),
            failure_kind=failure_kind,
            outcome=outcome,
            task_authority=TargetAuthority.CENSORED,
            local_authority=TargetAuthority.DETECTOR_CONFIRMED,
            task_return=None,
            local_failure_cost=1.0,
            context=context,
            witnesses=witnesses,
            detector_window_steps=self.config.detector_window_steps,
            progress_epoch=self._progress_epoch,
            provenance=provenance,
        )
        return EvidenceRecord(
            incident=incident,
            plan=self.compiler.compile(incident),
        )

    def _censored_record(self, failure_kind: str) -> EvidenceRecord:
        context = self._context(relevant_episode_steps=(), tail_only=True)
        provenance = self._provenance_for_context(context)
        incident = FailureIncident(
            incident_id=_stable_id(
                "failure-incident",
                self.episode_id,
                failure_kind,
                context.context_id,
            ),
            scope_key=self._scope_key(context.steps[-1].anchor),
            failure_kind=failure_kind,
            outcome=FailureOutcome.CENSORED,
            task_authority=TargetAuthority.CENSORED,
            local_authority=TargetAuthority.CENSORED,
            task_return=None,
            local_failure_cost=None,
            context=context,
            witnesses=(),
            detector_window_steps=self.config.detector_window_steps,
            progress_epoch=self._progress_epoch,
            provenance=provenance,
        )
        return EvidenceRecord(
            incident=incident,
            plan=self.compiler.compile(incident),
        )

    def _provenance_for_context(
        self,
        context: LearningContext,
    ) -> CreditProvenance:
        """Bind incident provenance to the policy that observed its last step.

        An asynchronous actor may adopt a newer parameter snapshot only at a
        complete unroll boundary while remaining in the same game episode.
        Each :class:`LearningStep` retains its exact version; the incident
        provenance names the version active when the local outcome was
        observed rather than incorrectly freezing the episode's first one.
        """

        return replace(
            self.provenance,
            policy_version=context.steps[-1].policy_version,
        )

    def _capture_terminal_completion(self) -> None:
        macro = self._terminal_macro
        if macro is None:
            return
        receipt = ProgressReceipt(
            kind=ProgressKind.FLOW_ADVANCE,
            source="pipeline:authoritative_terminal_success",
        )
        self._capture_completion(macro, receipt=receipt)
        self._terminal_macro = None

    def finalize(
        self,
        *,
        failure_kind: str | None,
        local_failure: bool,
        terminal_succeeded: bool,
        censored_reason: str = "episode_boundary_censored",
    ) -> FailureCreditEpisodeResult:
        """Compile immutable records after the episode outcome is authoritative."""

        if self._finalized:
            raise RuntimeError("failure-credit episode may be finalized only once")
        self._finalized = True
        if not isinstance(local_failure, bool) or not isinstance(terminal_succeeded, bool):
            raise TypeError("local_failure and terminal_succeeded must be booleans")
        if failure_kind is not None:
            _text(failure_kind, label="failure_kind")
        _text(censored_reason, label="censored_reason")

        if self._active_macro is not None:
            active = self._active_macro
            self._terminal_macro = _ClosedMacro(
                edge=active.builder.close(
                    outcome=MacroEdgeOutcome.CENSORED,
                    closing_step_id=self._captured[-1].step.episode_step,
                ),
                source_spec_id=active.source_spec_id,
                direct_credit_allowed=False,
                progress_epoch=self._progress_epoch,
            )
            self._active_macro = None
        if terminal_succeeded:
            self._capture_terminal_completion()

        records = [completion.record for completion in self._completion_records]
        # Shadow/offline callers do not install a streaming sink.  Preserve all
        # detector-authoritative records at the boundary in that mode.
        records.extend(self._ready_records)
        self._ready_records.clear()
        if local_failure:
            already_streamed = bool(
                self._latest_cycle is not None
                and self._latest_cycle.supporting_episode_steps in self._emitted_cycle_supports
            )
            # A streamed exact cycle already owns the causal local failure.
            # Do not duplicate it at the terminal boundary.  A unique stall,
            # or a cycle that could not fit the streaming context, still emits
            # the ordinary terminal record.
            if not already_streamed:
                records.append(
                    self._failure_record(
                        failure_kind or "detector_confirmed_liveness_failure",
                    )
                )
        elif not terminal_succeeded:
            records.append(self._censored_record(failure_kind or censored_reason))
        result = tuple(records)
        return FailureCreditEpisodeResult(
            records=result,
            metrics=FailureCreditShadowMetrics(
                decisions=self._decision_count,
                progress_receipts=tuple(
                    (kind, self._receipt_counts.get(kind, 0))
                    for kind in ProgressKind
                    if self._receipt_counts.get(kind, 0)
                ),
                detected_cycles=self._detected_cycles,
                completion_controls=len(self._completion_records),
                completion_controls_observed=self._completion_controls_observed,
                completion_controls_dropped=(self._completion_controls_observed - len(self._completion_records)),
                completion_storage_nbytes=sum(item.storage_nbytes for item in self._completion_records),
                records=len(result),
                censored_semantic_transitions=self._semantic_censored_transitions,
                streamed_records=self._streamed_records,
                detected_direct_cycles=self._detected_direct_cycles,
                detected_multi_edge_cycles=self._detected_multi_edge_cycles,
            ),
        )


__all__ = [
    "FAILURE_CREDIT_COLLECTOR_VERSION",
    "FAILURE_CREDIT_DETECTOR_VERSION",
    "FailureCreditEpisodePipeline",
    "FailureCreditEpisodeResult",
    "FailureCreditPipelineConfig",
    "FailureCreditShadowMetrics",
]
