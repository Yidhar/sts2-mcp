"""Typed model payload and canonical fact extraction for baseline replay."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

from sts2_baseline import (
    BASELINE_TRANSITION_PROJECTION_SPEC,
    BaselineTransition,
    PotentialState,
    ReplayStratum,
    baseline_reward_identity,
)
from sts2_rl.contracts import EnvironmentResult
from sts2_rl.encoding import EncodedDecisionSnapshot, grounding_encoding_identity

RewardObjective = Literal["combat", "run"]
RUN_PROGRESS_FLOOR_CAP = BASELINE_TRANSITION_PROJECTION_SPEC.run_progress_floor_cap
_COMBAT_RESULT_MAP = BASELINE_TRANSITION_PROJECTION_SPEC.combat_result_map()

_DROP_REPLAY_KEYS = frozenset(
    {
        "action_handle",
        "action_id",
        "action_index",
        "card_index",
        "idx",
        "index",
        "slot",
        "target_id",
        "target_handle",
        "option_index",
        "selection_id",
        "slot_index",
        "transport_kind",
        "semantic",
        "action_semantic",
        "preview",
        "canonical_text",
        "route_summary",
        "effect_deltas",
        "effect_preview",
        "card_effect_profile",
        "effect_profile",
        "derived_view",
        "semantic_tags",
        "timing_tags",
        "training_tags",
        "static_traits",
        "reactive_triggers",
        "phase_rules",
        "combat_tags",
        "danger_profile",
        "target_priority_hints",
        "boss_mechanics",
        "combat_tactical",
        "potion_timing",
        "aux_targets",
        "objective_targets",
        "objective_context",
        "state_hash",
        "semantic_state_hash",
        "captured_at_utc",
    }
)


@dataclass(frozen=True, slots=True)
class DecisionExperience:
    """Compact encoded decision stored by replay for direct batch collation."""

    encoded_snapshot: EncodedDecisionSnapshot
    source_fingerprint: str
    action_index: int
    behavior_log_probability: float
    terminal_class: int
    objective: RewardObjective
    encoding_fingerprint: str
    reward_fingerprint: str

    def __post_init__(self) -> None:
        self.validate()

    def validate(self, *, validate_snapshot: bool = True) -> None:
        """Validate live and unpickled replay payloads without trusting __post_init__."""

        if not isinstance(self.encoded_snapshot, EncodedDecisionSnapshot):
            raise TypeError("decision experience requires an encoded snapshot")
        if not 0 <= self.action_index < self.encoded_snapshot.candidate_count:
            raise ValueError("decision action_index is outside encoded candidates")
        if not bool(self.encoded_snapshot.action_mask[self.action_index]):
            raise ValueError("decision action_index selects a disabled encoded candidate")
        if self.objective not in {"combat", "run"}:
            raise ValueError("decision objective must be combat or run")
        if self.terminal_class not in {0, 1, 2}:
            raise ValueError("terminal_class must be ongoing/terminal/chance_boundary")
        if self.encoding_fingerprint != grounding_encoding_identity()[
            "fingerprint_sha256"
        ]:
            raise ValueError("decision experience encoding fingerprint does not match")
        if validate_snapshot:
            self.encoded_snapshot.validate(
                expected_config=self.encoded_snapshot.config,
                expected_fingerprint=self.encoding_fingerprint,
            )
        if self.reward_fingerprint != baseline_reward_identity()["fingerprint_sha256"]:
            raise ValueError("decision experience reward fingerprint does not match")
        if not math.isfinite(float(self.behavior_log_probability)):
            raise ValueError("behavior_log_probability must be finite")
        if self.behavior_log_probability > 0.0:
            raise ValueError("behavior_log_probability cannot exceed log(1)=0")
        if len(self.source_fingerprint) != 64 or any(
            char not in "0123456789abcdef" for char in self.source_fingerprint
        ):
            raise ValueError("decision source fingerprint must be lowercase SHA-256")


def _mapping(value: Any, *, label: str, required: bool = False) -> Mapping[str, Any]:
    if value is None and not required:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _finite_number(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{label} must be a JSON number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{label} must be finite")
    return normalized


def _required_number(
    payload: Mapping[str, Any],
    names: tuple[str, ...],
    *,
    label: str,
) -> float:
    for name in names:
        if name in payload:
            return _finite_number(payload[name], label=label)
    raise ValueError(f"{label} is required")


def _player(observation: Mapping[str, Any]) -> Mapping[str, Any] | None:
    direct = observation.get("player")
    if direct is not None:
        return _mapping(direct, label="observation.player", required=True)
    run = _mapping(observation.get("run"), label="observation.run")
    run_player = run.get("player")
    if run_player is not None:
        return _mapping(run_player, label="observation.run.player", required=True)
    players = observation.get("players")
    if players is not None and not isinstance(players, list | tuple):
        raise ValueError("observation.players must be an array")
    if isinstance(players, list | tuple) and players:
        first = players[0]
        if not isinstance(first, Mapping):
            raise ValueError("observation.players[0] must be an object")
        creature = first.get("creature")
        if creature is not None:
            return _mapping(
                creature,
                label="observation.players[0].creature",
                required=True,
            )
        return first
    return None


def _enemy_totals(
    combat: Mapping[str, Any],
    *,
    in_combat: bool,
) -> tuple[float, float]:
    if not in_combat:
        return 0.0, 0.0
    enemies = combat.get("enemies")
    if not isinstance(enemies, list | tuple):
        raise ValueError("combat.enemies must be an array while combat is in progress")
    if not enemies:
        raise ValueError("combat.enemies cannot be empty while combat is in progress")
    current_total = 0.0
    maximum_total = 0.0
    for index, enemy in enumerate(enemies):
        if not isinstance(enemy, Mapping):
            raise ValueError(f"combat.enemies[{index}] must be an object")
        current = _required_number(
            enemy,
            ("hp", "current_hp"),
            label=f"combat.enemies[{index}].hp",
        )
        maximum = _required_number(
            enemy,
            ("max_hp", "maximum_hp"),
            label=f"combat.enemies[{index}].max_hp",
        )
        if maximum <= 0.0:
            raise ValueError(f"combat.enemies[{index}].max_hp must be positive")
        if not 0.0 <= current <= maximum:
            raise ValueError(
                f"combat.enemies[{index}].hp must satisfy 0 <= hp <= max_hp"
            )
        current_total += current
        maximum_total += maximum
    return current_total, maximum_total


def potential_state(
    observation: Mapping[str, Any],
    *,
    objective: RewardObjective,
) -> PotentialState:
    """Extract only normalized reward facts; no mechanics or quality rules."""

    if objective not in {"combat", "run"}:
        raise ValueError("reward objective must be combat or run")
    if not isinstance(observation, Mapping):
        raise ValueError("observation must be an object")
    raw_terminated = observation.get("terminated", False)
    if not isinstance(raw_terminated, bool):
        raise ValueError("observation.terminated must be a boolean")
    player = _player(observation)
    if player is None:
        if not raw_terminated:
            raise ValueError("observation has no typed player reward facts")
        hp, max_hp = 0.0, 0.0
    else:
        has_hp = any(name in player for name in ("hp", "current_hp"))
        has_max_hp = any(name in player for name in ("max_hp", "maximum_hp"))
        if raw_terminated and not has_hp and not has_max_hp:
            # HeadlessSim's game_over DTO intentionally contains no player
            # snapshot.  The fixed reward already zeros terminal potential;
            # represent that exact absence rather than inventing HP values.
            hp, max_hp = 0.0, 0.0
        else:
            hp = _required_number(player, ("hp", "current_hp"), label="player.hp")
            max_hp = _required_number(
                player,
                ("max_hp", "maximum_hp"),
                label="player.max_hp",
            )
            if max_hp <= 0.0:
                raise ValueError("player.max_hp must be positive")
            if not 0.0 <= hp <= max_hp:
                raise ValueError("player.hp must satisfy 0 <= hp <= max_hp")

    combat = _mapping(observation.get("combat"), label="observation.combat")
    domain = str(observation.get("decision_domain") or "").lower()
    raw_in_combat = combat.get("in_progress")
    if raw_in_combat is None:
        in_combat = domain == "combat"
    elif isinstance(raw_in_combat, bool):
        in_combat = raw_in_combat
    else:
        raise ValueError("combat.in_progress must be a boolean")
    enemy_hp, enemy_max_hp = _enemy_totals(combat, in_combat=in_combat)

    run = _mapping(observation.get("run"), label="observation.run")
    explicit_progress = run.get("progress", observation.get("run_progress"))
    if explicit_progress is not None:
        progress = _finite_number(explicit_progress, label="run.progress")
        if not 0.0 <= progress <= 1.0:
            raise ValueError("run.progress must be in [0, 1]")
    else:
        raw_floor = run.get("floor", observation.get("floor"))
        if raw_floor is None:
            if objective == "run":
                raise ValueError("run.floor or run.progress is required for run reward")
            floor = 0.0
        else:
            floor = _finite_number(raw_floor, label="run.floor")
            if floor < 0.0:
                raise ValueError("run.floor must be non-negative")
        progress = min(1.0, floor / RUN_PROGRESS_FLOOR_CAP)
    return PotentialState(
        player_hp=hp,
        player_max_hp=max_hp,
        enemy_hp=enemy_hp,
        enemy_max_hp=enemy_max_hp,
        run_progress=progress,
        in_combat=in_combat,
    )


def _terminal_results(
    result: EnvironmentResult,
    *,
    objective: RewardObjective,
    forced_truncation: bool,
) -> tuple[Literal["none", "win", "loss"], Literal["none", "win", "loss"]]:
    if objective not in {"combat", "run"}:
        raise ValueError("reward objective must be combat or run")
    if result.truncated:
        raise ValueError("transport-truncated results cannot become reward transitions")
    if result.transition is None:
        raise ValueError("reward transition requires typed environment transition facts")
    facts = result.transition.facts
    if "combat_result" not in facts:
        raise ValueError("transition facts require exact combat_result")
    raw_combat_result = facts["combat_result"]
    if not isinstance(raw_combat_result, str) or raw_combat_result not in _COMBAT_RESULT_MAP:
        raise ValueError(
            "transition combat_result must be one of none/victory/defeat/escaped"
        )
    if "terminal_reason" not in facts:
        raise ValueError("transition facts require typed terminal_reason")
    fact_reason = facts["terminal_reason"]
    if fact_reason is not None and not isinstance(fact_reason, str):
        raise ValueError("transition terminal_reason must be a string or null")
    if result.terminal_reason is not None and not isinstance(result.terminal_reason, str):
        raise ValueError("environment terminal_reason must be a string or null")
    if fact_reason != result.terminal_reason:
        raise ValueError("result and transition terminal_reason identities differ")

    combat_result = _COMBAT_RESULT_MAP[raw_combat_result]
    run_result = "none"
    if result.terminated and not forced_truncation:
        if raw_combat_result == "none":
            raise ValueError("terminated result has no exact typed terminal outcome")
        if objective == "run":
            run_result = combat_result
    if objective == "combat" and result.terminated and combat_result == "none":
        raise ValueError("terminated combat result has no exact typed combat outcome")
    return combat_result, cast(
        Literal["none", "win", "loss"], run_result
    )


def baseline_transition(
    *,
    before: EnvironmentResult,
    after: EnvironmentResult,
    action_handle: str,
    objective: RewardObjective,
    forced_truncation: bool = False,
) -> BaselineTransition:
    combat_result, run_result = _terminal_results(
        after,
        objective=objective,
        forced_truncation=forced_truncation,
    )
    return BaselineTransition(
        episode_id=after.episode_id or before.episode_id,
        step_index=after.step_index,
        action_handle=action_handle,
        before=potential_state(before.observation, objective=objective),
        after=potential_state(after.observation, objective=objective),
        combat_result=combat_result,
        run_result=run_result,
        truncated=bool(forced_truncation),
        metadata={
            "terminal_reason": after.terminal_reason,
            "truncation_kind": "collector_horizon" if forced_truncation else None,
            "state_version": (
                after.transition.after_state_version if after.transition is not None else None
            ),
        },
    )


def replay_stratum(
    observation: Mapping[str, Any],
    transition: BaselineTransition,
) -> ReplayStratum:
    domain = str(
        observation.get("decision_domain") or observation.get("phase") or "unknown"
    )
    combat = _mapping(observation.get("combat"), label="observation.combat")
    run = _mapping(observation.get("run"), label="observation.run")
    tier = str(combat.get("tier") or combat.get("room_type") or "unknown")
    encounter = str(combat.get("encounter_id") or combat.get("id") or "unknown")
    act = int(max(0.0, _number(run.get("act", observation.get("act")))))
    if transition.truncated:
        outcome = str(transition.metadata.get("truncation_kind") or "untyped-truncation")
    elif transition.run_result != "none":
        outcome = transition.run_result
    elif transition.combat_result != "none":
        outcome = transition.combat_result
    else:
        outcome = "ongoing"
    return ReplayStratum(
        domain=domain,
        tier=tier,
        encounter_id=encounter,
        deck_stage=f"act-{act}" if act > 0 else "unknown",
        outcome=outcome,
    )


def compact_replay_value(value: Any, *, drop_dispatch: bool = False) -> Any:
    """Remove transport/retired fields before replay owns a deep copy."""

    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for raw_key, child in value.items():
            key = str(raw_key)
            lowered = key.lower()
            if lowered.startswith("_") or lowered in _DROP_REPLAY_KEYS:
                if not (not drop_dispatch and lowered in {"action_handle", "action_id"}):
                    continue
            result[key] = compact_replay_value(child, drop_dispatch=drop_dispatch)
        return result
    if isinstance(value, list | tuple):
        return [compact_replay_value(item, drop_dispatch=drop_dispatch) for item in value]
    if value is None or isinstance(value, str | int | float | bool):
        return value
    return str(value)


def compact_decision(
    observation: Mapping[str, Any],
    legal_actions: Sequence[Mapping[str, Any]],
    *,
    encoded_snapshot: EncodedDecisionSnapshot,
    action_index: int,
    behavior_log_probability: float,
    terminal_class: int,
    objective: RewardObjective,
) -> DecisionExperience:
    compact_observation = compact_replay_value(observation, drop_dispatch=True)
    compact_actions = tuple(
        compact_replay_value(action, drop_dispatch=True) for action in legal_actions
    )
    if not isinstance(compact_observation, dict) or not all(
        isinstance(action, dict) for action in compact_actions
    ):
        raise TypeError("compacted decision must remain object-shaped")
    typed_actions = cast(tuple[dict[str, Any], ...], compact_actions)
    if encoded_snapshot.candidate_count != len(typed_actions):
        raise ValueError("encoded snapshot candidate count differs from legal actions")
    canonical_source = json.dumps(
        {
            "observation": compact_observation,
            "legal_actions": typed_actions,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return DecisionExperience(
        encoded_snapshot=encoded_snapshot,
        source_fingerprint=hashlib.sha256(canonical_source.encode("utf-8")).hexdigest(),
        action_index=action_index,
        behavior_log_probability=float(behavior_log_probability),
        terminal_class=terminal_class,
        objective=objective,
        encoding_fingerprint=grounding_encoding_identity()["fingerprint_sha256"],
        reward_fingerprint=baseline_reward_identity()["fingerprint_sha256"],
    )


__all__ = [
    "RUN_PROGRESS_FLOOR_CAP",
    "DecisionExperience",
    "baseline_transition",
    "compact_decision",
    "compact_replay_value",
    "potential_state",
    "replay_stratum",
]
