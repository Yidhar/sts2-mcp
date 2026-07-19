"""Typed recurrent actor collecting fixed-length v2 sequence unrolls."""

from __future__ import annotations

import math
import time
from collections import deque
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Literal, Protocol
from uuid import uuid4

import numpy as np
import torch
from numpy.typing import NDArray

from sts2_baseline import (
    RolloutStep,
    SequenceUnroll,
    TaskReward,
    TaskRewardCalculator,
)
from sts2_rl.contracts import (
    CombatResetRequest,
    EnvironmentBackend,
    EnvironmentResult,
    ResetRequest,
    StepRequest,
)
from sts2_rl.encoding import EncodedDecisionSnapshot, GroundedObservationEncoder
from sts2_rl.models import RecurrentCandidateModel

from .seeding import (
    EVALUATION_SEED_PARITY,
    SIGNED_INT32_MAX,
    training_seed_start,
)
from .trajectory import (
    SemanticDeadlockDetector,
    TrajectoryJournal,
    canonical_json,
    semantic_action_fingerprint,
    semantic_decision_fingerprint,
    semantic_fingerprint,
    semantic_projection,
)
from .transaction import (
    TransactionEffect,
    TransactionOutcome,
    TransactionStep,
    TransactionTrace,
    backfill_factual_monte_carlo_returns,
)


class CollectionProtocolError(RuntimeError):
    """The environment exposed no dispatchable legal candidate."""


class RewardCalculator(Protocol):
    def evaluate(
        self,
        before: EnvironmentResult,
        after: EnvironmentResult,
        *,
        deadlock: bool = False,
        horizon_exhausted: bool = False,
    ) -> TaskReward: ...


@dataclass(frozen=True, slots=True)
class EpisodeMetrics:
    episode_id: str
    reset_seed: int
    steps: int
    reward_total: float
    terminal_reason: str | None
    truncated: bool
    run_won: bool
    combat_won: bool
    act1_cleared: bool
    max_act: int
    max_floor: int
    policy_decisions: int
    forced_decisions: int
    maximum_observed_candidates: int
    deadlocked: bool
    combat_progress_stalled: bool
    maximum_combat_no_net_progress_steps: int
    noncombat_progress_stalled: bool
    maximum_noncombat_no_durable_progress_steps: int
    revivals_used: int
    revival_free_combat_win: bool
    revival_free_act1_clear: bool
    revival_free_run_win: bool
    player_hp_lost: float
    stall_evidence: dict[str, object] | None


@dataclass(frozen=True, slots=True)
class CollectedEpisode:
    unrolls: tuple[SequenceUnroll, ...]
    metrics: EpisodeMetrics
    actor_policy_version: int
    behavior_policy_version: int
    timings: CollectorTimings | None = None
    transaction_traces: tuple[TransactionTrace, ...] = ()


@dataclass(frozen=True, slots=True)
class EpisodeProgress:
    """Compact actor progress published once per completed recurrent unroll."""

    episode_id: str
    reset_seed: int
    steps: int
    reward_total: float
    max_act: int
    max_floor: int
    policy_decisions: int
    forced_decisions: int
    maximum_observed_candidates: int
    revivals_used: int
    player_hp_lost: float
    combat_in_progress: bool
    phase: str
    decision_domain: str
    combat_no_net_progress_steps: int
    noncombat_no_durable_progress_steps: int
    combat_anchor_enemy_hp_total: float
    combat_required_net_hp_progress: float
    enemy_hp_total: float
    enemy_max_hp_total: float
    hand_cards: int
    draw_cards: int
    discard_cards: int
    exhaust_cards: int
    legal_action_kinds: dict[str, int]
    selected_action_kinds: dict[str, int]
    last_selected_action_kind: str
    behavior_policy_version: int


@dataclass(frozen=True, slots=True)
class CollectorStageTiming:
    count: int
    total_ms: float
    min_ms: float
    max_ms: float

    def to_mapping(self) -> dict[str, float | int]:
        return {
            "count": self.count,
            "total_ms": self.total_ms,
            "mean_ms": self.total_ms / self.count,
            "min_ms": self.min_ms,
            "max_ms": self.max_ms,
        }


@dataclass(frozen=True, slots=True)
class CollectorTimings:
    """Episode-aggregated collector timings; never written per environment step."""

    stages: dict[str, CollectorStageTiming]

    def to_mapping(self) -> dict[str, dict[str, float | int]]:
        return {name: timing.to_mapping() for name, timing in sorted(self.stages.items())}


@dataclass(frozen=True, slots=True)
class _ActionChoice:
    action_index: int
    behavior_log_probability: float
    valid_count: int
    snapshot: EncodedDecisionSnapshot
    recurrent_state: torch.Tensor
    policy: NDArray[np.float32]
    value: float
    encoding_ms: float
    policy_forward_ms: float


@dataclass(slots=True)
class _MutableStageTiming:
    count: int = 0
    total_ms: float = 0.0
    min_ms: float = float("inf")
    max_ms: float = 0.0

    def add(self, duration_ms: float) -> None:
        duration_ms = max(0.0, float(duration_ms))
        self.count += 1
        self.total_ms += duration_ms
        self.min_ms = min(self.min_ms, duration_ms)
        self.max_ms = max(self.max_ms, duration_ms)


class _CollectorTimingAccumulator:
    def __init__(self) -> None:
        self._stages: dict[str, _MutableStageTiming] = {}

    def add(self, stage: str, duration_ms: float) -> None:
        self._stages.setdefault(stage, _MutableStageTiming()).add(duration_ms)

    def record(self, stage: str, started_ns: int) -> None:
        self.add(stage, (time.perf_counter_ns() - started_ns) / 1_000_000.0)

    def snapshot(self) -> CollectorTimings:
        return CollectorTimings(
            stages={
                name: CollectorStageTiming(
                    count=timing.count,
                    total_ms=timing.total_ms,
                    min_ms=timing.min_ms,
                    max_ms=timing.max_ms,
                )
                for name, timing in self._stages.items()
            }
        )


def _number(value: object, default: float = 0.0) -> float:
    if not isinstance(value, str | int | float):
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


_TRANSACTION_SURFACE_FIELDS = (
    "mode",
    "prompt_id",
    "operation_type",
    "source_zone",
    "destination_zone",
    "min_select",
    "max_select",
    "requires_manual_confirmation",
    "can_skip",
    "cancelable",
)


_TRANSACTION_DEFINITION_KEYS = (
    "model_id",
    "entity_id",
    "id",
    "card_id",
    "definition_id",
)
_TRANSACTION_INSTANCE_KEYS = (
    "instance_uuid",
    "card_instance_id",
    "instance_id",
    "card_ref",
    "ref",
    "uuid",
    "uid",
)
_TRANSACTION_PHYSICAL_SOURCE_KEYS = (
    "source_pile",
    "source_zone",
    "physical_pile",
    "physical_zone",
)


def _first_nonempty(value: Mapping[str, object], keys: tuple[str, ...]) -> object | None:
    for key in keys:
        item = value.get(key)
        if item is not None and str(item).strip():
            return item
    return None


def _transaction_option_identity(value: object) -> object:
    """Return the stable physical identity of one selectable card.

    Card DTOs contain resolved cost, previews, UI ordinals and membership
    pseudo-zones.  None of those identifies the selectable *object*, and all of
    them may change after a toggle.  This positive allowlist intentionally
    retains only definition, concrete instance and physical source.  The
    surrounding sorted tuple preserves duplicate-card multiplicity.
    """

    if not isinstance(value, Mapping):
        return {"opaque_identity": semantic_projection(value)}
    nested = value.get("card")
    card = nested if isinstance(nested, Mapping) else value
    definition = _first_nonempty(card, _TRANSACTION_DEFINITION_KEYS)
    instance = _first_nonempty(card, _TRANSACTION_INSTANCE_KEYS)
    physical_source: object | None = None
    for key in _TRANSACTION_PHYSICAL_SOURCE_KEYS:
        item = card.get(key)
        if item is None or not str(item).strip():
            continue
        physical_source = item
        break
    identity: dict[str, object] = {
        "definition": str(definition).strip() if definition is not None else "<unknown>",
    }
    if instance is not None:
        identity["instance"] = str(instance).strip()
    if physical_source is not None:
        identity["physical_source"] = str(physical_source).strip()
    return identity


def _selection_actions(
    legal_actions: tuple[Mapping[str, object], ...],
) -> tuple[Mapping[str, object], ...]:
    return tuple(
        action
        for action in legal_actions
        if str(action.get("model_action_kind") or "") == "card_selection"
    )


def _transaction_selection_context(
    observation: Mapping[str, object],
    legal_actions: tuple[Mapping[str, object], ...],
) -> Mapping[str, object] | None:
    raw_selection = observation.get("card_selection")
    if isinstance(raw_selection, Mapping):
        return raw_selection
    actions = _selection_actions(legal_actions)
    if not actions:
        return None
    # Legacy/live surfaces may expose selection semantics only on candidates.
    # Merge the stable prompt contract without depending on candidate order.
    synthesized: dict[str, object] = {"mode": "action-derived"}
    for field in _TRANSACTION_SURFACE_FIELDS:
        values: list[object] = []
        for action in actions:
            nested = action.get("selection")
            item = nested.get(field) if isinstance(nested, Mapping) else None
            if item is None:
                item = action.get(field)
            if item is not None:
                values.append(item)
        if values:
            canonical_values = sorted({canonical_json(semantic_projection(item)) for item in values})
            synthesized[field] = canonical_values[0] if len(canonical_values) == 1 else canonical_values
    return synthesized


def _transaction_option_universe(
    selection: Mapping[str, object],
    legal_actions: tuple[Mapping[str, object], ...],
) -> tuple[str, ...]:
    """Return an order-independent, multiplicity-preserving option universe."""

    options = selection.get("options")
    identities: list[object] = []
    if isinstance(options, list | tuple) and options:
        identities.extend(_transaction_option_identity(option) for option in options)
    else:
        # Simulator translation marks ``cards`` as the *currently selectable*
        # membership, not as the complete physical universe. A toggle moves a
        # card between that collection and ``selected_cards``, so the stable
        # universe is their union. Some bridge DTOs also emit the explicit
        # ``selectable_cards`` alias; when present it is authoritative and must
        # replace, rather than duplicate, ``cards``.
        selectable_field = "selectable_cards" if "selectable_cards" in selection else "cards"
        for field in (selectable_field, "selected_cards"):
            items = selection.get(field)
            if isinstance(items, list | tuple):
                identities.extend(_transaction_option_identity(item) for item in items)
    if not identities:
        # Last-resort projection for legacy DTOs: use only card-bearing
        # selection candidates and erase the select/deselect membership role.
        for action in legal_actions:
            if str(action.get("model_action_kind") or "") != "card_selection":
                continue
            card = action.get("card")
            if isinstance(card, Mapping):
                identities.append(_transaction_option_identity(card))
    return tuple(sorted(canonical_json(identity) for identity in identities))


def _transaction_surface_key(
    observation: Mapping[str, object],
    legal_actions: tuple[Mapping[str, object], ...],
) -> str | None:
    """Return a stable card-selection transaction kind, without membership.

    Selection counts, selected cards and transport handles are intentionally
    excluded.  Those values identify nodes *inside* one transaction and belong
    in ``_transaction_node_key`` instead.
    """

    raw_selection = _transaction_selection_context(observation, legal_actions)
    if raw_selection is None:
        return None
    selection = {
        key: raw_selection[key]
        for key in _TRANSACTION_SURFACE_FIELDS
        if raw_selection.get(key) is not None
    }
    raw_run = observation.get("run")
    run = raw_run if isinstance(raw_run, Mapping) else {}
    return semantic_fingerprint(
        {
            "kind": "card_selection_transaction",
            "locus": {
                "act": run.get("act"),
                "floor": run.get("floor"),
                "room_type": run.get("room_type"),
                "room_model_id": run.get("room_model_id"),
            },
            "selection": selection,
            "option_universe": _transaction_option_universe(
                raw_selection,
                legal_actions,
            ),
        }
    )


def _transaction_selected_identities(
    observation: Mapping[str, object],
    legal_actions: tuple[Mapping[str, object], ...],
) -> tuple[str, ...]:
    raw_selection = observation.get("card_selection")
    identities: list[object] = []
    if isinstance(raw_selection, Mapping):
        options = raw_selection.get("options")
        if isinstance(options, list | tuple):
            identities.extend(
                _transaction_option_identity(option)
                for option in options
                if isinstance(option, Mapping)
                and (
                    option.get("is_selected") is True
                    or option.get("selected") is True
                    or str(option.get("selection_membership") or "").lower() == "selected"
                )
            )
        if not identities:
            selected = raw_selection.get("selected_cards")
            if isinstance(selected, list | tuple):
                identities.extend(_transaction_option_identity(item) for item in selected)
    if not identities:
        for action in _selection_actions(legal_actions):
            variant = str(action.get("model_action_variant") or action.get("kind") or "").lower()
            card = action.get("card")
            if isinstance(card, Mapping) and (
                "deselect" in variant
                or card.get("is_selected") is True
                or str(card.get("selection_membership") or "").lower() == "selected"
            ):
                identities.append(_transaction_option_identity(card))
    return tuple(sorted(canonical_json(identity) for identity in identities))


def _transaction_selected_count(
    observation: Mapping[str, object],
    legal_actions: tuple[Mapping[str, object], ...] = (),
) -> int:
    raw_selection = observation.get("card_selection")
    if not isinstance(raw_selection, Mapping):
        return len(_transaction_selected_identities(observation, legal_actions))
    raw_count = raw_selection.get("selected_count")
    if isinstance(raw_count, int) and not isinstance(raw_count, bool):
        return max(0, raw_count)
    selected = raw_selection.get("selected_cards")
    if isinstance(selected, list | tuple):
        return len(selected)
    options = raw_selection.get("options")
    if isinstance(options, list | tuple):
        return sum(
            int(isinstance(option, Mapping) and option.get("is_selected") is True)
            for option in options
        )
    return len(_transaction_selected_identities(observation, legal_actions))


def _transaction_node_key(
    surface_key: str | None,
    observation: Mapping[str, object],
    legal_actions: tuple[Mapping[str, object], ...],
) -> str:
    """Identify a selection node independent of DTO/candidate ordering."""

    if surface_key is None:
        return semantic_decision_fingerprint(observation, legal_actions)
    selection = _transaction_selection_context(observation, legal_actions) or {}
    remaining: object | None = None
    for key in ("remaining_select", "remaining_picks", "remaining", "required_remaining"):
        if selection.get(key) is not None:
            remaining = selection[key]
            break
    can_confirm = selection.get("can_confirm")
    if can_confirm is None:
        can_confirm = any(
            str(action.get("model_action_variant") or action.get("kind") or "").lower()
            in {"confirm", "confirm_selection"}
            for action in _selection_actions(legal_actions)
        )
    # Pairwise factual outcomes may only meet at the same reward-relevant
    # world state.  A selection prompt/membership alone is insufficient: the
    # same discard/transform UI at different HP, decks, relics or cumulative
    # revival cost is a different decision.  Remove only the transaction DTO
    # and duplicated candidate surfaces whose membership/order is represented
    # below; preserve the complete remaining semantic observation.  Reward
    # counters live under an underscore-prefixed transport namespace, so copy
    # that exact mapping under a semantic key before canonicalization drops
    # private transport fields.
    world_context = {
        key: value
        for key, value in observation.items()
        if key not in {"card_selection", "available_actions", "legal_actions"}
        and not str(key).startswith("_")
    }
    training_reward_state = observation.get("_training")
    if isinstance(training_reward_state, Mapping):
        world_context["training_reward_state"] = training_reward_state
    return semantic_fingerprint(
        {
            "kind": "card_selection_node",
            "surface_key": surface_key,
            "world_context": world_context,
            "selected": _transaction_selected_identities(observation, legal_actions),
            "selected_count": _transaction_selected_count(observation, legal_actions),
            "remaining": remaining,
            "can_confirm": bool(can_confirm),
            "legal_action_fingerprints": tuple(
                sorted(semantic_action_fingerprint(action) for action in legal_actions)
            ),
        }
    )


def _signed_selection_delta(before: int, after: int) -> int:
    return 1 if after > before else -1 if after < before else 0


def _classify_transaction_transition(
    *,
    current_surface: str | None,
    next_surface: str | None,
    current_node: str,
    next_node: str,
    seen_nodes: set[str],
    before_selected_count: int,
    after_selected_count: int,
) -> tuple[TransactionEffect, int]:
    if current_surface is None:
        return (
            TransactionEffect.STAY if next_node == current_node else TransactionEffect.MOVE,
            0,
        )
    if next_surface != current_surface:
        # Exit tears down the selection DTO.  That is not a factual deselect.
        return TransactionEffect.EXIT, 0
    effect = (
        TransactionEffect.STAY
        if next_node == current_node
        else TransactionEffect.REVISIT
        if next_node in seen_nodes
        else TransactionEffect.MOVE
    )
    return effect, _signed_selection_delta(
        before_selected_count,
        after_selected_count,
    )


def _run_position(observation: Mapping[str, object]) -> tuple[int, int]:
    raw_run = observation.get("run")
    run = raw_run if isinstance(raw_run, Mapping) else {}
    act = int(max(0.0, _number(run.get("act", observation.get("act")))))
    floor = int(max(0.0, _number(run.get("floor", observation.get("floor")))))
    return act, floor


def _combat_in_progress(observation: Mapping[str, object]) -> bool:
    combat = observation.get("combat")
    return bool(isinstance(combat, Mapping) and combat.get("in_progress") is True)


def _enemy_hp_totals(observation: Mapping[str, object]) -> tuple[float, float]:
    combat = observation.get("combat")
    enemies = combat.get("enemies") if isinstance(combat, Mapping) else None
    if not isinstance(enemies, list | tuple):
        return 0.0, 0.0
    current = 0.0
    maximum = 0.0
    for enemy in enemies:
        if not isinstance(enemy, Mapping):
            continue
        current += max(0.0, _number(enemy.get("hp", enemy.get("current_hp"))))
        maximum += max(
            0.0,
            _number(enemy.get("max_hp", enemy.get("maximum_hp"))),
        )
    return current, maximum


def _enemy_roster_signature(observation: Mapping[str, object]) -> tuple[str, ...]:
    """Return a stable multiset identity without volatile HP/status fields."""

    combat = observation.get("combat")
    enemies = combat.get("enemies") if isinstance(combat, Mapping) else None
    if not isinstance(enemies, list | tuple):
        return ()
    identities: list[str] = []
    for index, enemy in enumerate(enemies):
        if not isinstance(enemy, Mapping):
            continue
        definition = next(
            (
                str(enemy[key])
                for key in (
                    "instance_uuid",
                    "instance_id",
                    "entity_uuid",
                    "entity_id",
                    "monster_id",
                    "id",
                )
                if enemy.get(key) not in (None, "")
            ),
            f"anonymous:{index}",
        )
        identities.append(definition.strip().lower())
    return tuple(sorted(identities))


def _combat_phase_signature(observation: Mapping[str, object]) -> tuple[str, ...]:
    """Extract explicit phase/wave identities, deliberately excluding turns."""

    combat = observation.get("combat")
    if not isinstance(combat, Mapping):
        return ()
    markers: list[str] = []
    for key in (
        "encounter_id",
        "combat_id",
        "wave",
        "wave_id",
        "wave_index",
        "stage",
        "stage_id",
    ):
        value = combat.get(key)
        if isinstance(value, str | int | float) and not isinstance(value, bool):
            markers.append(f"combat.{key}={value}")
    enemies = combat.get("enemies")
    if isinstance(enemies, list | tuple):
        for index, enemy in enumerate(enemies):
            if not isinstance(enemy, Mapping):
                continue
            for key in ("stage", "stage_id"):
                value = enemy.get(key)
                if isinstance(value, str | int | float) and not isinstance(value, bool):
                    markers.append(f"enemy[{index}].{key}={value}")
    return tuple(markers)


@dataclass(frozen=True, slots=True)
class _CombatNetProgressStatus:
    age_steps: int
    maximum_age_steps: int
    stalled: bool
    current_hp: float
    maximum_hp: float
    anchor_hp: float
    required_hp_progress: float
    net_hp_progress: float
    progress_kind: str


class _CombatNetProgressTracker:
    """Detect combat loops by monotonic net progress, not transient damage.

    A hit followed by healing no longer resets the window.  The anchor advances
    only after a meaningful *net* reduction in the current enemy health burden,
    or after an explicit phase/wave transition.  Adding summons is not progress;
    defeating them without reducing the pre-summon burden is intentionally
    neutral.  This closes the old loophole where tiny recurring damage kept a
    hopeless combat alive until the 30,000-step transport ceiling.
    """

    def __init__(self, *, window: int, minimum_hp_fraction: float) -> None:
        self.window = int(window)
        self.minimum_hp_fraction = float(minimum_hp_fraction)
        self.reset()

    def reset(self) -> None:
        self._active = False
        self._anchor_step = 0
        self._anchor_hp = 0.0
        self._anchor_max_hp = 0.0
        self._last_hp = 0.0
        self._last_max_hp = 0.0
        self._roster: tuple[str, ...] = ()
        self._phase: tuple[str, ...] = ()
        self._maximum_age = 0

    def _start(
        self,
        *,
        step: int,
        current_hp: float,
        maximum_hp: float,
        roster: tuple[str, ...],
        phase: tuple[str, ...],
    ) -> None:
        self._active = True
        self._anchor_step = step
        self._anchor_hp = current_hp
        self._anchor_max_hp = maximum_hp
        self._last_hp = current_hp
        self._last_max_hp = maximum_hp
        self._roster = roster
        self._phase = phase

    def observe(
        self,
        *,
        step: int,
        observation: Mapping[str, object],
    ) -> _CombatNetProgressStatus:
        current_hp, maximum_hp = _enemy_hp_totals(observation)
        if not _combat_in_progress(observation):
            maximum_age = self._maximum_age
            self.reset()
            self._maximum_age = maximum_age
            return _CombatNetProgressStatus(
                age_steps=0,
                maximum_age_steps=self._maximum_age,
                stalled=False,
                current_hp=current_hp,
                maximum_hp=maximum_hp,
                anchor_hp=current_hp,
                required_hp_progress=0.0,
                net_hp_progress=0.0,
                progress_kind="outside_combat",
            )

        roster = _enemy_roster_signature(observation)
        phase = _combat_phase_signature(observation)
        if not self._active:
            self._start(
                step=step,
                current_hp=current_hp,
                maximum_hp=maximum_hp,
                roster=roster,
                phase=phase,
            )
            return _CombatNetProgressStatus(
                age_steps=0,
                maximum_age_steps=self._maximum_age,
                stalled=False,
                current_hp=current_hp,
                maximum_hp=maximum_hp,
                anchor_hp=current_hp,
                required_hp_progress=max(1.0, maximum_hp * self.minimum_hp_fraction),
                net_hp_progress=0.0,
                progress_kind="combat_started",
            )

        prior_roster = self._roster
        prior_phase = self._phase
        required = max(
            1.0,
            max(self._anchor_max_hp, maximum_hp) * self.minimum_hp_fraction,
        )
        net_progress = self._anchor_hp - current_hp
        roster_replaced = bool(
            roster != prior_roster and prior_roster and roster and not set(prior_roster).intersection(roster)
        )
        advanced_from_defeated_wave = bool(
            roster != prior_roster
            and self._last_hp <= max(1.0, self._last_max_hp * self.minimum_hp_fraction)
            and current_hp > self._last_hp
        )
        explicit_phase_advance = bool(phase != prior_phase and prior_phase and phase)
        progress_kind = "waiting_for_net_progress"
        if explicit_phase_advance or roster_replaced or advanced_from_defeated_wave:
            self._start(
                step=step,
                current_hp=current_hp,
                maximum_hp=maximum_hp,
                roster=roster,
                phase=phase,
            )
            progress_kind = "phase_or_wave_advanced"
            net_progress = 0.0
            required = max(1.0, maximum_hp * self.minimum_hp_fraction)
        elif net_progress >= required:
            self._start(
                step=step,
                current_hp=current_hp,
                maximum_hp=maximum_hp,
                roster=roster,
                phase=phase,
            )
            progress_kind = "meaningful_net_hp_reduction"
            net_progress = 0.0
            required = max(1.0, maximum_hp * self.minimum_hp_fraction)
        else:
            # A summon/addition changes the burden but must not reset the age.
            # Retain the old anchor while tracking the latest structural facts.
            self._roster = roster
            self._phase = phase
            self._last_hp = current_hp
            self._last_max_hp = maximum_hp

        age = step - self._anchor_step
        self._maximum_age = max(self._maximum_age, age)
        return _CombatNetProgressStatus(
            age_steps=age,
            maximum_age_steps=self._maximum_age,
            stalled=age >= self.window,
            current_hp=current_hp,
            maximum_hp=maximum_hp,
            anchor_hp=self._anchor_hp,
            required_hp_progress=required,
            net_hp_progress=self._anchor_hp - current_hp,
            progress_kind=progress_kind,
        )


def _first_durable_scalar(
    value: Mapping[str, object],
    *keys: str,
) -> object | None:
    """Return the first scalar fact from a positive durable-state allowlist."""

    for key in keys:
        item = value.get(key)
        if item is None or isinstance(item, Mapping | list | tuple):
            continue
        if isinstance(item, str | int | float | bool):
            return item
    return None


def _durable_entity_id(
    value: Mapping[str, object],
    *keys: str,
) -> str:
    item = _first_durable_scalar(value, *keys)
    return str(item).strip() if item not in (None, "") else "<unknown>"


def _nested_modifier_signature(value: object) -> tuple[tuple[object, ...], ...]:
    if not isinstance(value, list | tuple):
        return ()
    result: list[tuple[object, ...]] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        result.append(
            (
                _durable_entity_id(item, "id", "model_id", "name", "type"),
                _first_durable_scalar(item, "level", "amount", "stacks"),
                _first_durable_scalar(item, "is_active", "is_enabled"),
            )
        )
    return tuple(sorted(result, key=repr))


def _durable_card_signature(value: Mapping[str, object]) -> tuple[object, ...]:
    """Describe permanent card composition without preview/dynamic values."""

    return (
        _durable_entity_id(value, "id", "card_id", "model_id", "name", "title"),
        _first_durable_scalar(
            value,
            "upgrade_level",
            "upgrade_count",
            "times_upgraded",
            "is_upgraded",
            "upgraded",
        ),
        _nested_modifier_signature(value.get("enchantments")),
        _nested_modifier_signature(value.get("afflictions")),
    )


def _durable_relic_signature(value: Mapping[str, object]) -> tuple[object, ...]:
    return (
        _durable_entity_id(value, "id", "relic_id", "model_id", "name", "title"),
        _first_durable_scalar(value, "stack_count", "quantity"),
        _first_durable_scalar(value, "is_used_up"),
        _first_durable_scalar(value, "is_melted"),
        _first_durable_scalar(value, "status"),
    )


def _durable_potion_signature(value: Mapping[str, object]) -> tuple[object, ...]:
    return (
        _durable_entity_id(value, "id", "potion_id", "model_id", "name", "title"),
        _first_durable_scalar(value, "slot_index", "slot"),
    )


def _durable_collection(
    value: object,
    projector: Callable[[Mapping[str, object]], tuple[object, ...]],
) -> tuple[tuple[object, ...], ...]:
    if isinstance(value, Mapping):
        nested = value.get("cards", value.get("items"))
        value = nested if isinstance(nested, list | tuple) else ()
    if not isinstance(value, list | tuple):
        return ()
    projected = [projector(item) for item in value if isinstance(item, Mapping)]
    return tuple(sorted(projected, key=repr))


def _noncombat_durable_projections(
    observation: Mapping[str, object],
) -> tuple[Mapping[str, object], Mapping[str, object]]:
    """Split forward run locus from same-locus persistent resources.

    This deliberately excludes event/card/relic ``dynamic_vars``, descriptions,
    pages, screens, selection membership, damage/heal previews, counters and
    legal-option text.  Those values are reversible or may change forever
    without moving the run.  The positive allowlist keeps the detector generic
    across events while preventing a newly exposed preview counter from
    silently disabling it.
    """

    raw_run = observation.get("run")
    run = raw_run if isinstance(raw_run, Mapping) else {}
    raw_player = observation.get("player")
    player = raw_player if isinstance(raw_player, Mapping) else {}
    raw_event = observation.get("event")
    event = raw_event if isinstance(raw_event, Mapping) else {}
    raw_map = observation.get("map")
    map_state = raw_map if isinstance(raw_map, Mapping) else {}

    run_projection: dict[str, object] = {}
    for key, aliases in {
        "active": ("active", "run_active"),
        "game_over": ("game_over",),
        "act": ("act", "act_index"),
        "floor": ("floor", "total_floor"),
        "room_type": ("room_type",),
        "room_model_id": ("room_model_id", "room_model"),
    }.items():
        item = _first_durable_scalar(run, *aliases)
        if item is None:
            item = _first_durable_scalar(observation, *aliases)
        if item is not None:
            run_projection[key] = item
    coord = run.get("coord", map_state.get("current_coord"))
    if isinstance(coord, Mapping):
        run_projection["coord"] = tuple(
            _first_durable_scalar(coord, *aliases) for aliases in (("x", "col", "column"), ("y", "row"))
        )

    player_projection: dict[str, object] = {}
    for key, aliases in {
        "character": ("character_id", "character", "id"),
        "hp": ("hp", "current_hp"),
        "max_hp": ("max_hp", "maximum_hp"),
        "gold": ("gold",),
        "open_potion_slots": ("open_potion_slots",),
    }.items():
        item = _first_durable_scalar(player, *aliases)
        if item is not None:
            player_projection[key] = item
    # The live bridge exposes ``deck`` as a count and the inspectable card
    # collection as ``deck_cards``; the headless bridge exposes ``deck`` as the
    # collection itself.  Prefer the explicit collection and only fall back to
    # ``deck`` when it actually has collection shape.
    deck_value = player.get("deck_cards")
    if not isinstance(deck_value, Mapping | list | tuple):
        deck_value = player.get("deck")
    player_projection["deck"] = _durable_collection(
        deck_value,
        _durable_card_signature,
    )
    player_projection["relics"] = _durable_collection(
        player.get("relics"),
        _durable_relic_signature,
    )
    player_projection["potions"] = _durable_collection(
        player.get("potions"),
        _durable_potion_signature,
    )

    event_projection: dict[str, object] = {}
    for key, aliases in {
        "event_id": ("event_id", "id", "model_id"),
        "encounter_id": ("encounter_id", "canonical_encounter_id"),
        "is_finished": ("is_finished",),
    }.items():
        item = _first_durable_scalar(event, *aliases)
        if item is not None:
            event_projection[key] = item

    completion_projection: dict[str, object] = {}
    for key in ("terminated", "truncated"):
        item = _first_durable_scalar(observation, key)
        if item is not None:
            completion_projection[key] = item

    locus = {
        "run": run_projection,
        "event": event_projection,
        "completion": completion_projection,
    }
    resources = {"player": player_projection}
    return locus, resources


def _durable_action_projection(action: Mapping[str, object]) -> Mapping[str, object]:
    """Identify a chosen operation without transport handles or preview text."""

    result: dict[str, object] = {}
    for key in (
        "action",
        "kind",
        "model_action_kind",
        "model_action_variant",
        "transport_kind",
        "index",
        "idx",
        "action_index",
        "card_index",
        "target_id",
        "target_index",
        "slot",
        "slot_index",
        "selection_operation",
        "is_selected",
    ):
        item = _first_durable_scalar(action, key)
        if item is not None:
            result[key] = item
    for key, projector in (
        ("card", _durable_card_signature),
        ("relic", _durable_relic_signature),
        ("potion", _durable_potion_signature),
    ):
        item = action.get(key)
        if isinstance(item, Mapping):
            result[key] = projector(item)
    option = action.get("option")
    if isinstance(option, Mapping):
        result["option_index"] = _first_durable_scalar(
            option,
            "option_index",
            "index",
        )
    return result


def _noncombat_context_summary(
    observation: Mapping[str, object],
) -> Mapping[str, object]:
    run = observation.get("run")
    run = run if isinstance(run, Mapping) else {}
    event = observation.get("event")
    event = event if isinstance(event, Mapping) else {}
    return {
        "phase": str(observation.get("phase") or ""),
        "decision_domain": str(observation.get("decision_domain") or ""),
        "state_type": str(observation.get("state_type") or ""),
        "act": int(max(0.0, _number(run.get("act", observation.get("act"))))),
        "floor": int(max(0.0, _number(run.get("floor", observation.get("floor"))))),
        "room_type": str(run.get("room_type") or ""),
        "room_model_id": str(run.get("room_model_id", run.get("room_model")) or ""),
        "event_id": str(event.get("event_id", event.get("id")) or ""),
        "event_finished": event.get("is_finished") is True,
    }


@dataclass(frozen=True, slots=True)
class _NonCombatDurableProgressStatus:
    age_steps: int
    maximum_age_steps: int
    stalled: bool
    durable_state_fingerprint: str
    locus_fingerprint: str
    resource_fingerprint: str
    action_fingerprint: str
    progress_kind: str
    context: Mapping[str, object]


class _NonCombatDurableProgressTracker:
    """Bound a repeated non-combat action that makes no durable run progress."""

    def __init__(self, *, window: int) -> None:
        self.window = int(window)
        self.reset()

    def reset(self) -> None:
        self._active = False
        self._anchor_step = 0
        self._locus_fingerprint = ""
        self._resource_fingerprint = ""
        self._durable_state_fingerprint = ""
        self._action_fingerprint = ""
        self._seen_resource_fingerprints: set[str] = set()
        self._maximum_age = 0

    @staticmethod
    def _outside_noncombat(
        observation: Mapping[str, object],
        *,
        terminated: bool = False,
        truncated: bool = False,
    ) -> bool:
        return bool(
            _combat_in_progress(observation)
            or terminated
            or truncated
            or observation.get("terminated") is True
            or observation.get("truncated") is True
        )

    @staticmethod
    def _fingerprints(
        observation: Mapping[str, object],
    ) -> tuple[str, str, str]:
        locus, resources = _noncombat_durable_projections(observation)
        locus_fingerprint = semantic_fingerprint(locus)
        resource_fingerprint = semantic_fingerprint(resources)
        durable_fingerprint = semantic_fingerprint(
            {
                "locus": locus_fingerprint,
                "resources": resource_fingerprint,
            }
        )
        return locus_fingerprint, resource_fingerprint, durable_fingerprint

    def _start_locus(
        self,
        *,
        step: int,
        locus_fingerprint: str,
        resource_fingerprint: str,
        durable_fingerprint: str,
        action_fingerprint: str,
    ) -> None:
        self._active = True
        self._anchor_step = step
        self._locus_fingerprint = locus_fingerprint
        self._resource_fingerprint = resource_fingerprint
        self._durable_state_fingerprint = durable_fingerprint
        self._action_fingerprint = action_fingerprint
        self._seen_resource_fingerprints = {resource_fingerprint}

    def _status(
        self,
        *,
        age: int,
        progress_kind: str,
        context: Mapping[str, object],
    ) -> _NonCombatDurableProgressStatus:
        self._maximum_age = max(self._maximum_age, age)
        return _NonCombatDurableProgressStatus(
            age_steps=age,
            maximum_age_steps=self._maximum_age,
            stalled=age >= self.window,
            durable_state_fingerprint=self._durable_state_fingerprint,
            locus_fingerprint=self._locus_fingerprint,
            resource_fingerprint=self._resource_fingerprint,
            action_fingerprint=self._action_fingerprint,
            progress_kind=progress_kind,
            context=context,
        )

    def seed(
        self,
        *,
        step: int,
        observation: Mapping[str, object],
        terminated: bool = False,
        truncated: bool = False,
    ) -> _NonCombatDurableProgressStatus:
        """Seed the reset state so the configured window starts at step zero."""

        context = _noncombat_context_summary(observation)
        if self._outside_noncombat(
            observation,
            terminated=terminated,
            truncated=truncated,
        ):
            return self._status(
                age=0,
                progress_kind="outside_noncombat_decision",
                context=context,
            )
        locus, resources, durable = self._fingerprints(observation)
        self._start_locus(
            step=step,
            locus_fingerprint=locus,
            resource_fingerprint=resources,
            durable_fingerprint=durable,
            action_fingerprint="",
        )
        return self._status(
            age=0,
            progress_kind="noncombat_seeded",
            context=context,
        )

    def observe(
        self,
        *,
        step: int,
        observation: Mapping[str, object],
        selected_action: Mapping[str, object],
        terminated: bool = False,
        truncated: bool = False,
    ) -> _NonCombatDurableProgressStatus:
        context = _noncombat_context_summary(observation)
        if self._outside_noncombat(
            observation,
            terminated=terminated,
            truncated=truncated,
        ):
            maximum_age = self._maximum_age
            self.reset()
            self._maximum_age = maximum_age
            return self._status(
                age=0,
                progress_kind="outside_noncombat_decision",
                context=context,
            )

        locus_fingerprint, resource_fingerprint, durable_fingerprint = self._fingerprints(observation)
        action_fingerprint = semantic_fingerprint(_durable_action_projection(selected_action))
        if not self._active:
            progress_kind = "noncombat_started"
            self._start_locus(
                step=step,
                locus_fingerprint=locus_fingerprint,
                resource_fingerprint=resource_fingerprint,
                durable_fingerprint=durable_fingerprint,
                action_fingerprint=action_fingerprint,
            )
            return self._status(age=0, progress_kind=progress_kind, context=context)
        if locus_fingerprint != self._locus_fingerprint:
            self._start_locus(
                step=step,
                locus_fingerprint=locus_fingerprint,
                resource_fingerprint=resource_fingerprint,
                durable_fingerprint=durable_fingerprint,
                action_fingerprint=action_fingerprint,
            )
            return self._status(
                age=0,
                progress_kind="forward_locus_changed",
                context=context,
            )
        if resource_fingerprint not in self._seen_resource_fingerprints:
            self._seen_resource_fingerprints.add(resource_fingerprint)
            self._anchor_step = step
            self._resource_fingerprint = resource_fingerprint
            self._durable_state_fingerprint = durable_fingerprint
            self._action_fingerprint = action_fingerprint
            return self._status(
                age=0,
                progress_kind="new_durable_resource_state",
                context=context,
            )
        # Recurring A<->B (or larger finite) resource cycles are not forward
        # progress. Only the first sighting of a state at this locus earns a
        # reset; subsequent visits age from the last genuinely novel state.
        self._resource_fingerprint = resource_fingerprint
        self._durable_state_fingerprint = durable_fingerprint
        self._action_fingerprint = action_fingerprint
        return self._status(
            age=step - self._anchor_step,
            progress_kind="recurring_durable_resource_state",
            context=context,
        )


def _zone_count(observation: Mapping[str, object], key: str) -> int:
    player = observation.get("player")
    value = player.get(key) if isinstance(player, Mapping) else None
    if isinstance(value, Mapping):
        value = value.get("cards")
    return len(value) if isinstance(value, list | tuple) else 0


def _legal_action_kind_counts(
    legal_actions: tuple[dict[str, object], ...],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for action in legal_actions:
        kind = str(action.get("model_action_kind", action.get("kind", "unknown")) or "unknown")
        counts[kind] = counts.get(kind, 0) + 1
    return dict(sorted(counts.items()))


_COMBAT_STALL_HAND_PREVIEW_LIMIT = 8


def _visible_zone_cards(
    observation: Mapping[str, object],
    key: str,
) -> tuple[Mapping[str, object], ...] | None:
    """Return visible cards for a diagnostic, without treating redaction as empty."""

    player = observation.get("player")
    combat = observation.get("combat")
    value = player.get(key) if isinstance(player, Mapping) else None
    if value is None and isinstance(combat, Mapping):
        value = combat.get(key)
    if isinstance(value, Mapping):
        value = value.get("cards", value.get("items"))
    if not isinstance(value, list | tuple):
        return None
    return tuple(card for card in value if isinstance(card, Mapping))


def _diagnostic_zone_count(
    observation: Mapping[str, object],
    key: str,
) -> int | None:
    """Read a visible pile count while preserving unknown/redacted as None."""

    player = observation.get("player")
    combat = observation.get("combat")
    owners = tuple(owner for owner in (player, combat) if isinstance(owner, Mapping))
    for owner in owners:
        value = owner.get(key)
        if isinstance(value, Mapping):
            explicit = value.get("count")
            if isinstance(explicit, int | float) and not isinstance(explicit, bool):
                return max(0, int(explicit))
            value = value.get("cards", value.get("items"))
        if isinstance(value, list | tuple):
            return len(value)
        if isinstance(value, int | float) and not isinstance(value, bool):
            return max(0, int(value))
        count = owner.get(f"{key}_count")
        if isinstance(count, int | float) and not isinstance(count, bool):
            return max(0, int(count))
    return None


def _bounded_stable_id(value: object) -> str:
    """Bound an externally supplied identifier while retaining stable identity."""

    text = str(value).strip()
    encoded = text.encode("utf-8")
    if len(encoded) <= 128:
        return text
    prefix = encoded[:64].decode("utf-8", errors="ignore")
    return f"{prefix}#sha256:{semantic_fingerprint({'stable_id': text})}"


def _combat_stall_hand_preview(
    observation: Mapping[str, object],
) -> tuple[dict[str, object], ...]:
    """Return at most a few stable hand identities and explicit playability facts."""

    cards = _visible_zone_cards(observation, "hand")
    if cards is None:
        return ()
    preview: list[dict[str, object]] = []
    for card in cards:
        identity = next(
            (
                card[key]
                for key in (
                    "id",
                    "card_id",
                    "model_id",
                    "definition_id",
                    "card_definition_id",
                )
                if card.get(key) not in (None, "")
            ),
            None,
        )
        if identity is None:
            continue
        item: dict[str, object] = {"id": _bounded_stable_id(identity)}
        for key in ("is_playable", "playable", "can_play"):
            playable = card.get(key)
            if isinstance(playable, bool):
                item["is_playable"] = playable
                break
        preview.append(item)
    preview.sort(key=lambda item: (str(item["id"]), repr(item.get("is_playable"))))
    return tuple(preview[:_COMBAT_STALL_HAND_PREVIEW_LIMIT])


def _combat_stall_evidence(
    *,
    status: _CombatNetProgressStatus,
    observation: Mapping[str, object],
    legal_actions: tuple[dict[str, object], ...],
    window: int,
    detected_step: int,
) -> dict[str, object]:
    """Build one bounded, mechanics-agnostic diagnostic at stall termination."""

    hand_preview = _combat_stall_hand_preview(observation)
    hand_count = _diagnostic_zone_count(observation, "hand")
    return {
        "kind": "combat_no_net_progress",
        "window": window,
        "steps_without_net_progress": status.age_steps,
        "anchor_enemy_hp_total": status.anchor_hp,
        "current_enemy_hp_total": status.current_hp,
        "current_enemy_max_hp_total": status.maximum_hp,
        "net_enemy_hp_progress": status.net_hp_progress,
        "required_net_enemy_hp_progress": status.required_hp_progress,
        "progress_kind": status.progress_kind,
        "detected_step": detected_step,
        "hand_cards": hand_count,
        "draw_cards": _diagnostic_zone_count(observation, "draw_pile"),
        "discard_cards": _diagnostic_zone_count(observation, "discard_pile"),
        "exhaust_cards": _diagnostic_zone_count(observation, "exhaust_pile"),
        "legal_action_kinds": _legal_action_kind_counts(legal_actions),
        "hand_card_preview": hand_preview,
        "hand_card_preview_truncated": bool(
            hand_count is not None and hand_count > len(hand_preview)
        ),
    }


class GroundedCollector:
    """Collect policy trajectories with no MCTS, guard, or action rewrite."""

    def __init__(
        self,
        *,
        model: RecurrentCandidateModel,
        encoder: GroundedObservationEncoder,
        backend: EnvironmentBackend,
        scenario: Literal["full-run", "combat"],
        objective: Literal["combat", "act1", "run"],
        discount: float,
        max_episode_steps: int,
        character: str | None = None,
        encounter_id: str | None = None,
        seed: int = 0,
        unroll_length: int = 64,
        deadlock_window: int = 128,
        deadlock_repeat_threshold: int = 8,
        combat_net_progress_window: int = 256,
        noncombat_durable_progress_window: int = 256,
        combat_min_net_hp_fraction: float = 0.05,
        journal_policy_topk: int = 5,
        reward_calculator: RewardCalculator | None = None,
        additional_relics: tuple[str, ...] = (),
        revival_relic_id: str | None = None,
        training_revival_budget: int | None = None,
        horizon_as_failure: bool = False,
        transaction_burn_in_steps: int | None = None,
    ) -> None:
        if scenario not in {"full-run", "combat"}:
            raise ValueError("scenario must be full-run or combat")
        if objective not in {"combat", "act1", "run"}:
            raise ValueError("objective must be combat, act1, or run")
        if scenario == "combat" and objective != "combat":
            raise ValueError("combat scenario requires the combat objective")
        if scenario == "full-run" and objective == "combat":
            raise ValueError("full-run scenario requires the act1 or run objective")
        if (
            isinstance(discount, bool)
            or not isinstance(discount, int | float)
            or not math.isfinite(float(discount))
            or not 0.0 < float(discount) <= 1.0
        ):
            raise ValueError("discount must be in (0, 1]")
        if isinstance(max_episode_steps, bool) or not isinstance(max_episode_steps, int) or max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("collector seed must be a non-negative integer")
        if isinstance(unroll_length, bool) or not isinstance(unroll_length, int):
            raise TypeError("unroll_length must be an integer")
        if unroll_length <= 0:
            raise ValueError("unroll_length must be positive")
        if transaction_burn_in_steps is not None and (
            isinstance(transaction_burn_in_steps, bool)
            or not isinstance(transaction_burn_in_steps, int)
            or transaction_burn_in_steps < 0
        ):
            raise ValueError("transaction_burn_in_steps must be non-negative or null")
        if isinstance(journal_policy_topk, bool) or not isinstance(journal_policy_topk, int):
            raise TypeError("journal_policy_topk must be an integer")
        if journal_policy_topk <= 0:
            raise ValueError("journal_policy_topk must be positive")
        if isinstance(combat_net_progress_window, bool) or not isinstance(combat_net_progress_window, int):
            raise TypeError("combat_net_progress_window must be an integer")
        if combat_net_progress_window <= 0:
            raise ValueError("combat_net_progress_window must be positive")
        if isinstance(noncombat_durable_progress_window, bool) or not isinstance(
            noncombat_durable_progress_window, int
        ):
            raise TypeError("noncombat_durable_progress_window must be an integer")
        if noncombat_durable_progress_window <= 0:
            raise ValueError("noncombat_durable_progress_window must be positive")
        if (
            isinstance(combat_min_net_hp_fraction, bool)
            or not isinstance(combat_min_net_hp_fraction, int | float)
            or not math.isfinite(float(combat_min_net_hp_fraction))
            or not 0.0 < float(combat_min_net_hp_fraction) <= 1.0
        ):
            raise ValueError("combat_min_net_hp_fraction must be in (0, 1]")
        self.model = model
        self.encoder = encoder
        self.backend = backend
        self.scenario = scenario
        self.objective = objective
        self.discount = float(discount)
        self.max_episode_steps = int(max_episode_steps)
        self.character = character
        self.encounter_id = encounter_id
        self.unroll_length = unroll_length
        self.journal_policy_topk = journal_policy_topk
        self.combat_net_progress_window = combat_net_progress_window
        self.noncombat_durable_progress_window = noncombat_durable_progress_window
        self.combat_min_net_hp_fraction = float(combat_min_net_hp_fraction)
        self.additional_relics = tuple(str(item) for item in additional_relics)
        self.revival_relic_id = (
            revival_relic_id.strip().upper() if isinstance(revival_relic_id, str) and revival_relic_id.strip() else None
        )
        self.training_revival_budget = training_revival_budget
        self.horizon_as_failure = bool(horizon_as_failure)
        self.transaction_burn_in_steps = transaction_burn_in_steps
        if self.revival_relic_id is not None and self.revival_relic_id not in {
            item.upper() for item in self.additional_relics
        }:
            raise ValueError("revival_relic_id must be one of the injected additional relics")
        if self.training_revival_budget is not None:
            if self.revival_relic_id is None:
                raise ValueError("training_revival_budget requires an injected revival relic")
            if self.training_revival_budget < -1:
                raise ValueError("training_revival_budget must be -1 or non-negative")
        self._rng = np.random.default_rng(int(seed))
        self._episode_seed = training_seed_start(int(seed))
        self._active_state_version: int | None = None
        self.reward_calculator = reward_calculator or TaskRewardCalculator(
            objective,
            discount=discount,
        )
        self.deadlock_detector = SemanticDeadlockDetector(
            window_size=deadlock_window,
            repeat_threshold=deadlock_repeat_threshold,
        )

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def state_dict(self) -> dict[str, object]:
        """Return every collector-owned stochastic continuation input.

        Checkpoints are only published between episodes, so no live environment
        state belongs here.  The next reset seed and the epsilon-action RNG are
        nevertheless part of an exact continuation and must not silently reset.
        """

        return {
            "version": "sts2-recurrent-collector-state-v3",
            "episode_seed": self._episode_seed,
            "rng_state": deepcopy(self._rng.bit_generator.state),
        }

    def load_state_dict(self, payload: Mapping[str, object]) -> None:
        """Restore a fail-closed collector continuation state."""

        expected_keys = {"version", "episode_seed", "rng_state"}
        actual_keys = set(payload)
        if actual_keys != expected_keys:
            raise ValueError(
                "collector state keys mismatch: "
                f"missing={sorted(expected_keys - actual_keys)} "
                f"unknown={sorted(actual_keys - expected_keys)}"
            )
        if payload["version"] != "sts2-recurrent-collector-state-v3":
            raise ValueError("unsupported collector checkpoint state")
        episode_seed = payload["episode_seed"]
        if isinstance(episode_seed, bool) or not isinstance(episode_seed, int):
            raise TypeError("collector episode_seed must be an integer")
        if episode_seed < 0:
            raise ValueError("collector episode_seed must be non-negative")
        if episode_seed > SIGNED_INT32_MAX or episode_seed % 2 != 0:
            raise ValueError("collector episode_seed must be an even signed 32-bit seed")
        rng_state = payload["rng_state"]
        if not isinstance(rng_state, Mapping):
            raise TypeError("collector rng_state must be a mapping")
        candidate_rng = np.random.default_rng()
        try:
            candidate_rng.bit_generator.state = dict(deepcopy(rng_state))
        except (TypeError, ValueError) as exc:
            raise ValueError("collector rng_state is invalid") from exc
        self._episode_seed = episode_seed
        self._rng = candidate_rng

    def replace_backend(self, backend: EnvironmentBackend) -> None:
        """Adopt a fresh backend at a quiescent incident boundary.

        A poisoned simulator may already have committed the mutation whose
        result was rejected.  Its active revision must never leak into a new
        process/session, so the collector deliberately forgets that revision.
        """

        self.backend = backend
        self._active_state_version = None

    def _state_version(self) -> int:
        state = self.backend.get_state()
        if not isinstance(state, Mapping) or state.get("ok") is not True:
            raise CollectionProtocolError("backend state read was not explicitly successful")
        revision = state.get("state_version")
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
            raise CollectionProtocolError("backend state_version must be an exact non-negative integer")
        return revision

    @staticmethod
    def _validate_common_result(result: EnvironmentResult, *, surface: str) -> None:
        if result.ok is not True:
            raise CollectionProtocolError(f"{surface} result was not explicitly successful")
        if not isinstance(result.episode_id, str) or not result.episode_id.strip():
            raise CollectionProtocolError(f"{surface} result has no episode identity")
        if isinstance(result.step_index, bool) or not isinstance(result.step_index, int) or result.step_index < 0:
            raise CollectionProtocolError(f"{surface} result step_index must be an exact non-negative integer")
        if not isinstance(result.terminated, bool) or not isinstance(result.truncated, bool):
            raise CollectionProtocolError(f"{surface} terminated/truncated flags must be booleans")
        if result.terminated and result.truncated:
            raise CollectionProtocolError(f"{surface} result cannot be both terminated and truncated")
        if result.terminal_reason is not None and not isinstance(result.terminal_reason, str):
            raise CollectionProtocolError(f"{surface} terminal_reason must be a string or null")
        if not isinstance(result.info, Mapping):
            raise CollectionProtocolError(f"{surface} info must be an object")

    def _validate_reset_result(
        self,
        result: EnvironmentResult,
        *,
        before_state_version: int,
        after_state_version: int,
    ) -> None:
        self._validate_common_result(result, surface="reset")
        if result.step_index != 0:
            raise CollectionProtocolError("fresh reset result must start at step_index=0")
        if result.terminated or result.truncated:
            raise CollectionProtocolError("fresh reset returned a terminal/truncated episode")
        if not result.legal_actions:
            raise CollectionProtocolError("fresh reset returned zero legal actions")
        if result.info.get("reward_authority") != "external-rl":
            raise CollectionProtocolError("reset result must delegate reward authority to external-rl")
        transition = result.transition
        if transition is None:
            raise CollectionProtocolError("reset result has no typed transition")
        if transition.episode_id != result.episode_id or transition.step_index != 0:
            raise CollectionProtocolError("reset transition identity does not match result")
        if (
            transition.before_state_version != before_state_version
            or transition.after_state_version != after_state_version
        ):
            raise CollectionProtocolError("reset transition revision identity is inconsistent")
        if not isinstance(transition.facts, Mapping):
            raise CollectionProtocolError("reset transition facts must be an object")

    def _validate_step_result(
        self,
        before: EnvironmentResult,
        after: EnvironmentResult,
    ) -> None:
        self._validate_common_result(after, surface="step")
        if after.episode_id != before.episode_id:
            raise CollectionProtocolError("step result episode_id differs from the active episode")
        if after.step_index != before.step_index + 1:
            raise CollectionProtocolError("step result must advance step_index by exactly one")
        transition = after.transition
        if transition is None:
            raise CollectionProtocolError("step result has no typed transition")
        if transition.episode_id != after.episode_id or transition.step_index != after.step_index:
            raise CollectionProtocolError("step transition episode/step identity differs from result")
        if not isinstance(transition.facts, Mapping):
            raise CollectionProtocolError("step transition facts must be an object")
        if after.terminated:
            fact_name = "combat_result" if self.objective == "combat" else "run_result"
            terminal_result = transition.facts.get(fact_name)
            supported_results = (
                {"victory", "defeat", "escaped"}
                if self.objective == "combat"
                else {"victory", "defeat"}
            )
            if terminal_result not in supported_results:
                raise CollectionProtocolError(
                    f"terminated {self.objective} step has no authoritative typed {fact_name}"
                )
            expected_reason = f"{'combat' if self.objective == 'combat' else 'run'}_{terminal_result}"
            if (
                after.terminal_reason != expected_reason
                or transition.facts.get("terminal_reason") != expected_reason
            ):
                raise CollectionProtocolError(
                    "terminated step facts/result reason disagrees with its objective-scoped result"
                )
        active_revision = self._active_state_version
        if active_revision is None:
            raise CollectionProtocolError("collector has no active state revision")
        before_revision = transition.before_state_version
        after_revision = transition.after_state_version
        if (
            isinstance(before_revision, bool)
            or not isinstance(before_revision, int)
            or isinstance(after_revision, bool)
            or not isinstance(after_revision, int)
            or before_revision < 0
            or after_revision < 0
        ):
            raise CollectionProtocolError("step transition revisions must be exact non-negative integers")
        if before_revision != active_revision or after_revision != before_revision + 1:
            raise CollectionProtocolError("step transition revision chain is stale or non-contiguous")
        if after.info.get("reward_authority") != "external-rl":
            raise CollectionProtocolError("step result must delegate reward authority to external-rl")
        if not after.terminated and not after.truncated and not after.legal_actions:
            raise CollectionProtocolError("non-terminal step result returned zero legal actions")
        if (after.terminated or after.truncated) and after.legal_actions:
            raise CollectionProtocolError("terminal/truncated step result returned legal actions")
        self._active_state_version = after_revision

    def reset(self, *, evaluation_seed: int | None = None) -> EnvironmentResult:
        request_id = str(uuid4())
        expected_state_version = self._state_version()
        if evaluation_seed is None:
            seed = self._episode_seed
        else:
            if isinstance(evaluation_seed, bool) or not isinstance(evaluation_seed, int):
                raise TypeError("evaluation seed must be an integer")
            if (
                evaluation_seed < 0
                or evaluation_seed > SIGNED_INT32_MAX
                or evaluation_seed % 2 != EVALUATION_SEED_PARITY
            ):
                raise ValueError("evaluation seed must be an odd signed 32-bit seed")
            seed = evaluation_seed
        if self.scenario == "combat":
            result = self.backend.combat_reset(
                CombatResetRequest(
                    request_id=request_id,
                    session_id=self.backend.session_id,
                    expected_state_version=expected_state_version,
                    character=self.character,
                    encounter_id=self.encounter_id,
                    seed=seed,
                    additional_relics=self.additional_relics or None,
                    training_revival_budget=self.training_revival_budget,
                )
            )
        else:
            result = self.backend.reset(
                ResetRequest(
                    request_id=request_id,
                    session_id=self.backend.session_id,
                    scenario="full-run",
                    expected_state_version=expected_state_version,
                    character=self.character,
                    seed=seed,
                    force_fresh=True,
                    additional_relics=self.additional_relics or None,
                    training_revival_budget=self.training_revival_budget,
                )
            )
        committed_state_version = self._state_version()
        if committed_state_version != expected_state_version + 1:
            raise CollectionProtocolError("reset did not advance state_version by exactly one")
        self._validate_reset_result(
            result,
            before_state_version=expected_state_version,
            after_state_version=committed_state_version,
        )
        self._active_state_version = committed_state_version
        if evaluation_seed is None:
            self._episode_seed += 2
        return result

    def _choose_action(
        self,
        state: EnvironmentResult,
        recurrent_state: torch.Tensor,
        *,
        epsilon: float,
        deterministic: bool,
    ) -> _ActionChoice:
        normalized_epsilon = float(epsilon)
        if not math.isfinite(normalized_epsilon) or not 0.0 <= normalized_epsilon <= 1.0:
            raise ValueError("exploration epsilon must be finite and in [0, 1]")
        encoding_started_ns = time.perf_counter_ns()
        encoded = self.encoder.encode(
            state.observation,
            state.legal_actions,
            device=self.device,
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        encoding_ms = (time.perf_counter_ns() - encoding_started_ns) / 1_000_000.0
        policy_started_ns = time.perf_counter_ns()
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                output = self.model(
                    encoded.batch,
                    recurrent_state,
                    validate=False,
                )
                policy = output.policy_probabilities()[0].float().cpu().numpy()
                valid = output.action_mask[0].cpu().numpy().astype(bool)
                next_recurrent_state = output.recurrent_state.detach()
                value = float(output.value[0].item())
        finally:
            self.model.train(was_training)
        policy_forward_ms = (time.perf_counter_ns() - policy_started_ns) / 1_000_000.0
        valid_indices = np.flatnonzero(valid)
        if valid_indices.size == 0:
            raise CollectionProtocolError(
                f"episode={state.episode_id!r} step={state.step_index} has no enabled legal action"
            )
        valid_count = int(valid_indices.size)
        valid_policy = policy[valid_indices]
        if not np.all(np.isfinite(valid_policy)) or np.any(valid_policy < 0.0):
            raise CollectionProtocolError("model produced a non-finite policy distribution")
        policy_mass = float(valid_policy.sum())
        if not math.isfinite(policy_mass) or policy_mass <= 0.0:
            raise CollectionProtocolError("model produced zero/non-finite legal policy mass")
        if deterministic:
            selected = int(valid_indices[int(np.argmax(valid_policy))])
            return _ActionChoice(
                action_index=selected,
                behavior_log_probability=float(math.log(max(float(valid_policy.max() / policy_mass), 1e-30))),
                valid_count=valid_count,
                snapshot=encoded.snapshot,
                recurrent_state=next_recurrent_state,
                policy=policy,
                value=value,
                encoding_ms=encoding_ms,
                policy_forward_ms=policy_forward_ms,
            )

        behavior = np.zeros_like(policy, dtype=np.float64)
        behavior[valid_indices] = (
            1.0 - normalized_epsilon
        ) * valid_policy / policy_mass + normalized_epsilon / valid_count
        behavior /= behavior.sum()
        if not np.all(np.isfinite(behavior)) or np.any(behavior < 0.0):
            raise CollectionProtocolError("collector produced an invalid behavior policy")
        selected = int(self._rng.choice(len(behavior), p=behavior))
        return _ActionChoice(
            action_index=selected,
            behavior_log_probability=float(math.log(max(float(behavior[selected]), 1e-30))),
            valid_count=valid_count,
            snapshot=encoded.snapshot,
            recurrent_state=next_recurrent_state,
            policy=policy,
            value=value,
            encoding_ms=encoding_ms,
            policy_forward_ms=policy_forward_ms,
        )

    def _step(
        self,
        state: EnvironmentResult,
        *,
        action_index: int,
    ) -> tuple[EnvironmentResult, str]:
        action = state.legal_actions[action_index]
        handle_value = action.get("action_handle", action.get("action_id"))
        handle = str(handle_value) if handle_value is not None and str(handle_value) else ""
        request = StepRequest(
            request_id=str(uuid4()),
            session_id=self.backend.session_id,
            episode_id=state.episode_id,
            expected_step_index=state.step_index,
            action_id=handle or None,
            action_index=None if handle else action_index,
        )
        result = self.backend.step(request)
        self._validate_step_result(state, result)
        return result, handle or f"index:{action_index}"

    def collect_episode(
        self,
        *,
        epsilon: float = 0.0,
        deterministic: bool = False,
        record: bool = True,
        evaluation_seed: int | None = None,
        policy_version: int = 0,
        trajectory_journal: TrajectoryJournal | None = None,
        maximum_steps: int | None = None,
        unroll_sink: Callable[[SequenceUnroll], int | None] | None = None,
        progress_sink: Callable[[EpisodeProgress], None] | None = None,
        accepted_step_sink: Callable[[int, int], None] | None = None,
        journal_episode_id_prefix: str = "",
    ) -> CollectedEpisode:
        if isinstance(policy_version, bool) or not isinstance(policy_version, int):
            raise TypeError("policy_version must be an integer")
        if policy_version < 0:
            raise ValueError("policy_version must be non-negative")
        if not isinstance(journal_episode_id_prefix, str):
            raise TypeError("journal_episode_id_prefix must be a string")
        episode_limit = self.max_episode_steps
        if maximum_steps is not None:
            if isinstance(maximum_steps, bool) or not isinstance(maximum_steps, int):
                raise TypeError("maximum_steps must be an integer or null")
            if maximum_steps <= 0:
                raise ValueError("maximum_steps must be positive")
            episode_limit = min(episode_limit, maximum_steps)
        timings = _CollectorTimingAccumulator()
        reset_seed = self._episode_seed if evaluation_seed is None else evaluation_seed
        reset_started_ns = time.perf_counter_ns()
        state = self.reset(evaluation_seed=evaluation_seed)
        timings.record("reset", reset_started_ns)
        self.deadlock_detector.reset()
        recurrent_state = self.model.initial_state(1, device=self.device)
        segment_initial_state = recurrent_state[0].detach().float().cpu().numpy().copy()
        segment_start_step = state.step_index
        segment_policy_version = policy_version
        segment_steps: list[RolloutStep] = []
        unrolls: list[SequenceUnroll] = []
        reward_total = 0.0
        max_act, max_floor = _run_position(state.observation)
        policy_decisions = 0
        forced_decisions = 0
        maximum_observed_candidates = 0
        steps_taken = 0
        final_outcome = "ongoing"
        deadlocked = False
        combat_progress_stalled = False
        noncombat_progress_stalled = False
        stall_evidence: dict[str, object] | None = None
        combat_progress = _CombatNetProgressTracker(
            window=self.combat_net_progress_window,
            minimum_hp_fraction=self.combat_min_net_hp_fraction,
        )
        noncombat_progress = _NonCombatDurableProgressTracker(
            window=self.noncombat_durable_progress_window,
        )
        noncombat_progress_status = noncombat_progress.seed(
            step=0,
            observation=state.observation,
            terminated=state.terminated,
            truncated=state.truncated,
        )
        combat_progress_status = combat_progress.observe(
            step=0,
            observation=state.observation,
        )
        combat_in_progress = _combat_in_progress(state.observation)
        combat_no_net_progress_steps = combat_progress_status.age_steps
        maximum_combat_no_net_progress_steps = combat_progress_status.maximum_age_steps
        noncombat_no_durable_progress_steps = noncombat_progress_status.age_steps
        maximum_noncombat_no_durable_progress_steps = noncombat_progress_status.maximum_age_steps
        forced_horizon = False
        curriculum_horizon = False
        revivals_used = 0
        player_hp_lost = 0.0
        final_behavior_policy_version = policy_version
        selected_action_kind_counts: dict[str, int] = {}
        last_selected_action_kind = ""
        transaction_enabled = bool(record and self.transaction_burn_in_steps is not None)
        transaction_context: deque[tuple[int, TransactionStep]] = deque(
            maxlen=self.transaction_burn_in_steps or 0
        )
        transaction_traces_pending: list[TransactionTrace] = []
        active_transaction_surface: str | None = None
        active_transaction_steps: list[TransactionStep] = []
        active_transaction_start_step = 0
        active_transaction_burn_in = 0
        active_transaction_policy_version = policy_version
        active_transaction_seen_nodes: set[str] = set()
        episode_rewards: list[float] = []
        episode_discounts: list[float] = []
        last_task_terminal = False

        for step_offset in range(episode_limit):
            if state.terminated or state.truncated:
                break
            if not state.legal_actions:
                raise CollectionProtocolError(
                    f"episode={state.episode_id!r} step={state.step_index} returned zero legal actions"
                )
            choice = self._choose_action(
                state,
                recurrent_state,
                epsilon=epsilon,
                deterministic=deterministic,
            )
            timings.add("observation_encoding", choice.encoding_ms)
            timings.add("policy_forward", choice.policy_forward_ms)
            maximum_observed_candidates = max(
                maximum_observed_candidates,
                choice.snapshot.candidate_count,
            )
            policy_decisions += int(choice.valid_count > 1)
            forced_decisions += int(choice.valid_count == 1)
            selected_action = state.legal_actions[choice.action_index]
            last_selected_action_kind = str(
                selected_action.get(
                    "model_action_kind",
                    selected_action.get("kind", "unknown"),
                )
                or "unknown"
            )
            selected_action_kind_counts[last_selected_action_kind] = (
                selected_action_kind_counts.get(last_selected_action_kind, 0) + 1
            )
            current_transaction_surface = (
                _transaction_surface_key(state.observation, state.legal_actions)
                if transaction_enabled
                else None
            )
            current_transaction_node = (
                _transaction_node_key(
                    current_transaction_surface,
                    state.observation,
                    state.legal_actions,
                )
                if transaction_enabled
                else ""
            )
            if current_transaction_surface is not None:
                if active_transaction_surface is None:
                    context = tuple(transaction_context)
                    active_transaction_surface = current_transaction_surface
                    active_transaction_steps = [item[1] for item in context]
                    active_transaction_start_step = (
                        context[0][0] if context else step_offset
                    )
                    active_transaction_burn_in = len(context)
                    active_transaction_policy_version = segment_policy_version
                    active_transaction_seen_nodes = {current_transaction_node}
                elif active_transaction_surface != current_transaction_surface:
                    raise RuntimeError(
                        "transaction surface changed without an observed exit transition"
                    )
            deadlock_evidence = self.deadlock_detector.observe(
                step_index=state.step_index,
                observation=state.observation,
                legal_actions=state.legal_actions,
                selected_action=selected_action,
            )
            sim_step_started_ns = time.perf_counter_ns()
            next_state, _ = self._step(state, action_index=choice.action_index)
            timings.record("sim_step", sim_step_started_ns)
            steps_taken += 1
            if next_state.truncated:
                raise CollectionProtocolError("transport/outcome-unknown truncation discarded before rollout")
            result_terminal = next_state.terminated or next_state.truncated
            forced_horizon = step_offset + 1 >= episode_limit and not next_state.terminated and not next_state.truncated
            deadlock_evidence = self.deadlock_detector.confirm_after_step(
                deadlock_evidence,
                observation=next_state.observation,
                legal_actions=next_state.legal_actions,
            )
            if next_state.transition is None:  # pragma: no cover - validated above
                raise CollectionProtocolError("step result lost its typed transition")
            # The validated transition may itself expose a much wider next
            # decision than the pre-action state.  Record that accepted state
            # now, before a later mutation from it has a chance to fault.
            maximum_observed_candidates = max(
                maximum_observed_candidates,
                len(next_state.legal_actions),
            )
            # Publish only fully validated environment steps.  The asynchronous
            # supervisor uses this monotonic count to distinguish valid but
            # unflushed tail steps from recurrent unrolls already emitted to
            # the learner when an infrastructure incident aborts an episode.
            if accepted_step_sink is not None:
                accepted_step_sink(steps_taken, maximum_observed_candidates)
            next_combat_in_progress = _combat_in_progress(next_state.observation)
            combat_progress_status = combat_progress.observe(
                step=steps_taken,
                observation=next_state.observation,
            )
            combat_no_net_progress_steps = combat_progress_status.age_steps
            maximum_combat_no_net_progress_steps = max(
                maximum_combat_no_net_progress_steps,
                combat_progress_status.maximum_age_steps,
            )
            combat_progress_stalled = bool(combat_progress_status.stalled and not result_terminal)
            if combat_progress_stalled:
                # Construct this bounded snapshot only for the terminating
                # transition. Ordinary decisions retain no additional state.
                stall_evidence = _combat_stall_evidence(
                    status=combat_progress_status,
                    observation=next_state.observation,
                    legal_actions=next_state.legal_actions,
                    window=self.combat_net_progress_window,
                    detected_step=next_state.step_index,
                )
            combat_in_progress = bool(next_combat_in_progress and not result_terminal)
            noncombat_progress_status = noncombat_progress.observe(
                step=steps_taken,
                observation=next_state.observation,
                selected_action=selected_action,
                terminated=next_state.terminated,
                truncated=next_state.truncated,
            )
            noncombat_no_durable_progress_steps = noncombat_progress_status.age_steps
            maximum_noncombat_no_durable_progress_steps = max(
                maximum_noncombat_no_durable_progress_steps,
                noncombat_progress_status.maximum_age_steps,
            )
            noncombat_progress_stalled = bool(noncombat_progress_status.stalled and not result_terminal)
            # ``maximum_steps`` may be a runtime's remaining global budget,
            # which can cut an otherwise healthy episode after only one or a
            # few decisions. Only the configured task horizon is a semantic
            # failure. A shorter collection-budget cut keeps a positive
            # discount and a bootstrap snapshot instead of fabricating a loss.
            curriculum_horizon = bool(forced_horizon and episode_limit >= self.max_episode_steps)
            reward_started_ns = time.perf_counter_ns()
            # EnvironmentResult is the terminal authority. The pre-action
            # recurrence has now also been confirmed against its factual
            # successor, so a novel semantic exit is not a synthetic deadlock.
            effective_deadlock_evidence = None if result_terminal else deadlock_evidence
            breakdown = self.reward_calculator.evaluate(
                state,
                next_state,
                deadlock=(
                    not result_terminal
                    and (
                        effective_deadlock_evidence is not None or combat_progress_stalled or noncombat_progress_stalled
                    )
                ),
                horizon_exhausted=bool(curriculum_horizon and self.horizon_as_failure),
            )
            reward_total += breakdown.reward
            last_task_terminal = breakdown.task_terminal
            revivals_used += breakdown.revivals_used_delta
            player_hp_lost += breakdown.player_hp_lost_delta
            final_outcome = breakdown.outcome
            deadlocked = breakdown.outcome == "deadlock"
            if record:
                segment_steps.append(
                    RolloutStep(
                        snapshot=choice.snapshot,
                        action_index=choice.action_index,
                        behavior_log_probability=choice.behavior_log_probability,
                        reward=breakdown.reward,
                        discount=breakdown.discount,
                        policy_decision=choice.valid_count > 1,
                    )
                )
            if transaction_enabled:
                episode_rewards.append(float(breakdown.reward))
                episode_discounts.append(float(breakdown.discount))
                next_transaction_surface = _transaction_surface_key(
                    next_state.observation,
                    next_state.legal_actions,
                )
                next_transaction_node = _transaction_node_key(
                    next_transaction_surface,
                    next_state.observation,
                    next_state.legal_actions,
                )
                effect, selected_count_delta = _classify_transaction_transition(
                    current_surface=current_transaction_surface,
                    next_surface=next_transaction_surface,
                    current_node=current_transaction_node,
                    next_node=next_transaction_node,
                    seen_nodes=active_transaction_seen_nodes,
                    before_selected_count=_transaction_selected_count(
                        state.observation,
                        state.legal_actions,
                    ),
                    after_selected_count=_transaction_selected_count(
                        next_state.observation,
                        next_state.legal_actions,
                    ),
                )
                factual_transaction_step = TransactionStep(
                    snapshot=choice.snapshot,
                    action_index=choice.action_index,
                    node_key=current_transaction_node,
                    next_node_key=next_transaction_node,
                    action_fingerprint=semantic_action_fingerprint(selected_action),
                    effect=effect,
                    selected_count_delta=selected_count_delta,
                    transaction_return=None,
                    return_steps=None,
                )
                if current_transaction_surface is not None:
                    active_transaction_steps.append(factual_transaction_step)
                    active_transaction_seen_nodes.add(next_transaction_node)
                    transaction_ended = bool(
                        next_transaction_surface != current_transaction_surface
                        or result_terminal
                        or breakdown.task_terminal
                        or forced_horizon
                    )
                    if transaction_ended:
                        if active_transaction_surface is None:  # pragma: no cover - start invariant
                            raise RuntimeError("active transaction lost its surface")
                        transaction_traces_pending.append(
                            TransactionTrace(
                                trace_id=(
                                    f"seed-{reset_seed}:{state.episode_id}:"
                                    f"{active_transaction_start_step}:"
                                    f"{step_offset}:{active_transaction_surface}"
                                ),
                                episode_id=f"seed-{reset_seed}:{state.episode_id}",
                                surface_key=active_transaction_surface,
                                start_step=active_transaction_start_step,
                                policy_version=active_transaction_policy_version,
                                initial_recurrent_state=np.zeros(
                                    self.model.config.recurrent_hidden_dim,
                                    dtype=np.float32,
                                ),
                                steps=tuple(active_transaction_steps),
                                burn_in_steps=active_transaction_burn_in,
                                outcome=(
                                    TransactionOutcome.COMPLETED
                                    if next_transaction_surface
                                    != current_transaction_surface
                                    else TransactionOutcome.DEADLOCK
                                    if breakdown.outcome == "deadlock"
                                    else TransactionOutcome.CENSORED
                                ),
                            )
                        )
                        active_transaction_surface = None
                        active_transaction_steps = []
                        active_transaction_seen_nodes = set()
                transaction_context.append((step_offset, factual_transaction_step))
            if trajectory_journal is not None:
                journal_deadlock: Mapping[str, object] | None = None
                if combat_progress_stalled:
                    if stall_evidence is None:  # pragma: no cover - construction invariant
                        raise RuntimeError("combat stall lost its terminal evidence")
                    journal_deadlock = stall_evidence
                elif noncombat_progress_stalled:
                    journal_deadlock = {
                        "kind": "noncombat_no_durable_progress",
                        "window": self.noncombat_durable_progress_window,
                        "steps_without_durable_progress": (noncombat_no_durable_progress_steps),
                        "durable_state_fingerprint": (noncombat_progress_status.durable_state_fingerprint),
                        "locus_fingerprint": (noncombat_progress_status.locus_fingerprint),
                        "resource_fingerprint": (noncombat_progress_status.resource_fingerprint),
                        "action_fingerprint": (noncombat_progress_status.action_fingerprint),
                        "progress_kind": (noncombat_progress_status.progress_kind),
                        "context": dict(noncombat_progress_status.context),
                        # Durable progress is evaluated on the transition
                        # result, not the pre-action decision state below.
                        "detected_step": next_state.step_index,
                    }
                elif effective_deadlock_evidence is not None:
                    journal_deadlock = effective_deadlock_evidence.to_mapping()
                valid_indices = np.flatnonzero(choice.snapshot.action_mask)
                ranked = sorted(
                    valid_indices.tolist(),
                    key=lambda index: float(choice.policy[index]),
                    reverse=True,
                )[: self.journal_policy_topk]
                journal_event: dict[str, object] = {
                    "event": "decision",
                    "episode_id": f"{journal_episode_id_prefix}{state.episode_id}",
                    "reset_seed": reset_seed,
                    "step_index": state.step_index,
                    # The journal performs compact per-step projection and
                    # only canonicalizes complete DTOs for bounded rich
                    # snapshots. Avoid copying the entire game state on
                    # every held-out decision.
                    "observation": state.observation,
                    "legal_actions": state.legal_actions,
                    "selected_index": choice.action_index,
                    "selected_action": selected_action,
                    "policy_topk": [
                        {
                            "index": index,
                            "probability": float(choice.policy[index]),
                        }
                        for index in ranked
                    ],
                    "value": choice.value,
                    "reward": breakdown.reward,
                    "terminal_reward": breakdown.terminal_reward,
                    "potential_reward": breakdown.potential_reward,
                    "revival_penalty": breakdown.revival_penalty,
                    "pace_penalty": breakdown.pace_penalty,
                    "hp_loss_penalty": breakdown.hp_loss_penalty,
                    "player_hp_lost": player_hp_lost,
                    "revivals_used": revivals_used,
                    "outcome": breakdown.outcome,
                    "deadlock": journal_deadlock,
                }
                if journal_deadlock is not None:
                    # The decision record is anchored at the pre-action state,
                    # while progress stalls are evaluated on the transition
                    # result.  Preserve the exact result DTO only for the
                    # bounded rich anomaly snapshot so the trigger is auditable
                    # without restoring full-state logging on ordinary steps.
                    journal_event.update(
                        {
                            "result_step_index": next_state.step_index,
                            "result_observation": next_state.observation,
                            "result_legal_actions": next_state.legal_actions,
                            "result_terminated": next_state.terminated,
                            "result_truncated": next_state.truncated,
                            "result_terminal_reason": next_state.terminal_reason,
                        }
                    )
                trajectory_journal.write(journal_event)
            timings.record("reward_and_diagnostics", reward_started_ns)
            recurrent_state = choice.recurrent_state
            state = next_state
            act, floor = _run_position(state.observation)
            max_act = max(max_act, act)
            max_floor = max(max_floor, floor)

            flush_segment = bool(
                record
                and segment_steps
                and (len(segment_steps) >= self.unroll_length or breakdown.task_terminal or forced_horizon)
            )
            if flush_segment:
                bootstrap_snapshot: EncodedDecisionSnapshot | None = None
                if segment_steps[-1].discount > 0.0:
                    bootstrap_started_ns = time.perf_counter_ns()
                    bootstrap_snapshot = self.encoder.encode(
                        state.observation,
                        state.legal_actions,
                        device="cpu",
                    ).snapshot
                    maximum_observed_candidates = max(
                        maximum_observed_candidates,
                        bootstrap_snapshot.candidate_count,
                    )
                    timings.record("bootstrap_encoding", bootstrap_started_ns)
                completed_unroll = SequenceUnroll(
                    episode_id=state.episode_id,
                    start_step=segment_start_step,
                    policy_version=segment_policy_version,
                    initial_recurrent_state=segment_initial_state,
                    steps=tuple(segment_steps),
                    bootstrap_snapshot=bootstrap_snapshot,
                )
                final_behavior_policy_version = completed_unroll.policy_version
                if unroll_sink is None:
                    unrolls.append(completed_unroll)
                else:
                    adopted_policy_version = unroll_sink(completed_unroll)
                    if adopted_policy_version is not None:
                        if (
                            isinstance(adopted_policy_version, bool)
                            or not isinstance(adopted_policy_version, int)
                            or adopted_policy_version < segment_policy_version
                        ):
                            raise ValueError("unroll sink returned an invalid actor policy version")
                        segment_policy_version = adopted_policy_version
                if progress_sink is not None:
                    progress_sink(
                        EpisodeProgress(
                            episode_id=state.episode_id,
                            reset_seed=reset_seed,
                            steps=steps_taken,
                            reward_total=reward_total,
                            max_act=max_act,
                            max_floor=max_floor,
                            policy_decisions=policy_decisions,
                            forced_decisions=forced_decisions,
                            maximum_observed_candidates=(maximum_observed_candidates),
                            revivals_used=revivals_used,
                            player_hp_lost=player_hp_lost,
                            combat_in_progress=combat_in_progress,
                            phase=str(state.observation.get("phase") or ""),
                            decision_domain=str(state.observation.get("decision_domain") or ""),
                            combat_no_net_progress_steps=(combat_no_net_progress_steps),
                            noncombat_no_durable_progress_steps=(noncombat_no_durable_progress_steps),
                            combat_anchor_enemy_hp_total=(combat_progress_status.anchor_hp),
                            combat_required_net_hp_progress=(combat_progress_status.required_hp_progress),
                            enemy_hp_total=_enemy_hp_totals(state.observation)[0],
                            enemy_max_hp_total=_enemy_hp_totals(state.observation)[1],
                            hand_cards=_zone_count(state.observation, "hand"),
                            draw_cards=_zone_count(state.observation, "draw_pile"),
                            discard_cards=_zone_count(
                                state.observation,
                                "discard_pile",
                            ),
                            exhaust_cards=_zone_count(
                                state.observation,
                                "exhaust_pile",
                            ),
                            legal_action_kinds=_legal_action_kind_counts(state.legal_actions),
                            selected_action_kinds=dict(sorted(selected_action_kind_counts.items())),
                            last_selected_action_kind=last_selected_action_kind,
                            behavior_policy_version=completed_unroll.policy_version,
                        )
                    )
                segment_steps = []
                segment_initial_state = recurrent_state[0].detach().float().cpu().numpy().copy()
                segment_start_step = state.step_index

            if state.terminated or breakdown.task_terminal or forced_horizon:
                break

        if segment_steps:
            raise RuntimeError("collector exited with an unflushed rollout segment")
        if active_transaction_surface is not None:
            raise RuntimeError("collector exited with an unclosed transaction trace")
        transaction_traces: tuple[TransactionTrace, ...] = ()
        if transaction_traces_pending:
            authoritative_outcome = bool(state.terminated or last_task_terminal)
            transaction_traces = tuple(
                backfill_factual_monte_carlo_returns(
                    trace,
                    episode_rewards=tuple(episode_rewards),
                    episode_discounts=tuple(episode_discounts),
                    authoritative_outcome=authoritative_outcome,
                )
                for trace in transaction_traces_pending
            )
        terminal_facts = state.transition.facts if state.transition is not None else {}
        run_won = bool(
            self.objective == "run"
            and state.terminated
            and terminal_facts.get("run_result") == "victory"
        )
        combat_won = bool(
            self.objective == "combat"
            and state.terminated
            and terminal_facts.get("combat_result") == "victory"
        )
        if state.terminated and self.objective in {"run", "combat"}:
            typed_success = run_won if self.objective == "run" else combat_won
            if typed_success != (final_outcome == "success"):
                raise CollectionProtocolError(
                    "typed terminal result disagrees with the reward outcome"
                )
        return CollectedEpisode(
            unrolls=tuple(unrolls),
            metrics=EpisodeMetrics(
                episode_id=state.episode_id,
                reset_seed=reset_seed,
                steps=steps_taken,
                reward_total=reward_total,
                terminal_reason=(
                    "combat_progress_stall"
                    if combat_progress_stalled
                    else "noncombat_progress_stall"
                    if noncombat_progress_stalled
                    else "semantic_deadlock"
                    if deadlocked
                    else "curriculum_horizon"
                    if curriculum_horizon and self.horizon_as_failure
                    else "collection_budget"
                    if forced_horizon
                    else state.terminal_reason
                ),
                truncated=forced_horizon,
                run_won=run_won,
                combat_won=combat_won,
                act1_cleared=bool(max_act >= 2),
                max_act=max_act,
                max_floor=max_floor,
                policy_decisions=policy_decisions,
                forced_decisions=forced_decisions,
                maximum_observed_candidates=maximum_observed_candidates,
                deadlocked=deadlocked,
                combat_progress_stalled=combat_progress_stalled,
                maximum_combat_no_net_progress_steps=(maximum_combat_no_net_progress_steps),
                noncombat_progress_stalled=noncombat_progress_stalled,
                maximum_noncombat_no_durable_progress_steps=(maximum_noncombat_no_durable_progress_steps),
                revivals_used=revivals_used,
                revival_free_combat_win=bool(combat_won and revivals_used == 0),
                revival_free_act1_clear=bool(max_act >= 2 and revivals_used == 0),
                revival_free_run_win=bool(run_won and revivals_used == 0),
                player_hp_lost=player_hp_lost,
                stall_evidence=stall_evidence,
            ),
            actor_policy_version=segment_policy_version,
            behavior_policy_version=final_behavior_policy_version,
            timings=timings.snapshot(),
            transaction_traces=transaction_traces,
        )


__all__ = [
    "CollectedEpisode",
    "CollectionProtocolError",
    "EpisodeMetrics",
    "EpisodeProgress",
    "GroundedCollector",
]
