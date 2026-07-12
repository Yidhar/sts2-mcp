"""Canonical transition facts and versioned reward ownership."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from sts2_rl.contracts import EnvironmentResult
from sts2_rl.contracts.versions import REWARD_SCHEMA_VERSION, SCHEMA_VERSION

TRANSITION_SCHEMA_VERSION = SCHEMA_VERSION
REWARD_SPEC_VERSION = REWARD_SCHEMA_VERSION
CombatResult = Literal["none", "victory", "defeat", "escaped"]


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _integer(value: Any, default: int = 0) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return max(0, int(default))


def _ids(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list | tuple):
        return ()
    return tuple(str(item) for item in value if str(item).strip())


@dataclass(frozen=True, slots=True)
class TransitionFacts:
    """Facts aligned with contracts/schemas/transition.schema.json."""

    hp_delta: float = 0.0
    gold_delta: float = 0.0
    floor_delta: int = 0
    cards_added: tuple[str, ...] = ()
    cards_removed: tuple[str, ...] = ()
    potions_added: tuple[str, ...] = ()
    potions_removed: tuple[str, ...] = ()
    room_entered: str | None = None
    combat_result: CombatResult = "none"
    terminal_reason: str | None = None
    enemy_hp_delta: float = 0.0
    backend_reward: float = 0.0

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any] | None,
        *,
        backend_reward: float = 0.0,
    ) -> TransitionFacts:
        data = payload if isinstance(payload, Mapping) else {}
        raw_combat_result = str(data.get("combat_result") or "none").lower()
        combat_result = (
            cast(CombatResult, raw_combat_result)
            if raw_combat_result in {"none", "victory", "defeat", "escaped"}
            else "none"
        )
        return cls(
            hp_delta=_number(data.get("hp_delta", data.get("player_hp_delta"))),
            gold_delta=_number(data.get("gold_delta")),
            floor_delta=int(_number(data.get("floor_delta"))),
            cards_added=_ids(data.get("cards_added")),
            cards_removed=_ids(data.get("cards_removed")),
            potions_added=_ids(data.get("potions_added")),
            potions_removed=_ids(data.get("potions_removed")),
            room_entered=str(data["room_entered"]) if data.get("room_entered") is not None else None,
            combat_result=combat_result,
            terminal_reason=(
                str(data["terminal_reason"]) if data.get("terminal_reason") is not None else None
            ),
            enemy_hp_delta=_number(data.get("enemy_hp_delta")),
            backend_reward=_number(data.get("backend_reward"), backend_reward),
        )


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _player_from_observation(observation: Mapping[str, Any]) -> Mapping[str, Any]:
    direct = observation.get("player")
    if isinstance(direct, Mapping):
        return direct
    run = _mapping(observation.get("run"))
    player = run.get("player")
    return player if isinstance(player, Mapping) else {}


def _card_ids(value: Any) -> tuple[str, ...]:
    if isinstance(value, Mapping):
        value = value.get("cards")
    if not isinstance(value, list | tuple):
        return ()
    result: list[str] = []
    for item in value:
        if isinstance(item, str):
            result.append(item)
        elif isinstance(item, Mapping):
            card_id = item.get("id", item.get("card_id", item.get("name")))
            if card_id is not None:
                result.append(str(card_id))
    return tuple(result)


def _multiset_delta(before: tuple[str, ...], after: tuple[str, ...]) -> tuple[str, ...]:
    remaining: dict[str, int] = {}
    for value in before:
        remaining[value] = remaining.get(value, 0) + 1
    added: list[str] = []
    for value in after:
        if remaining.get(value, 0) > 0:
            remaining[value] -= 1
        else:
            added.append(value)
    return tuple(added)


def _enemy_hp_total(observation: Mapping[str, Any]) -> float:
    combat = _mapping(observation.get("combat"))
    enemies = combat.get("enemies")
    if not isinstance(enemies, list | tuple):
        return 0.0
    total = 0.0
    for enemy in enemies:
        if not isinstance(enemy, Mapping):
            continue
        hp = _number(enemy.get("hp", enemy.get("current_hp")))
        if 0.0 <= hp < 1_000_000.0:
            total += hp
    return total


def derive_transition_facts(
    before_observation: Mapping[str, Any] | None,
    after_observation: Mapping[str, Any],
    *,
    terminated: bool = False,
    terminal_reason: str | None = None,
    backend_reward: float = 0.0,
) -> TransitionFacts:
    """Derive the shared fact subset for typed in-process simulators."""
    before = before_observation if isinstance(before_observation, Mapping) else {}
    after = after_observation if isinstance(after_observation, Mapping) else {}
    before_player = _player_from_observation(before)
    after_player = _player_from_observation(after)
    before_run = _mapping(before.get("run"))
    after_run = _mapping(after.get("run"))

    before_deck = _card_ids(before_player.get("deck", before.get("deck")))
    after_deck = _card_ids(after_player.get("deck", after.get("deck")))
    before_potions = _card_ids(before_player.get("potions", before.get("potions")))
    after_potions = _card_ids(after_player.get("potions", after.get("potions")))
    before_hp = _number(before_player.get("hp", before_player.get("current_hp")))
    after_hp = _number(after_player.get("hp", after_player.get("current_hp")))
    before_gold = _number(before_player.get("gold", before_run.get("gold")))
    after_gold = _number(after_player.get("gold", after_run.get("gold")))
    before_floor = int(_number(before_run.get("floor", before.get("floor"))))
    after_floor = int(_number(after_run.get("floor", after.get("floor"))))

    reason = str(terminal_reason or "").lower()
    combat_result: CombatResult = "none"
    if terminated:
        if after_hp <= 0.0 or any(token in reason for token in ("death", "defeat", "died")):
            combat_result = "defeat"
        elif "escape" in reason:
            combat_result = "escaped"
        else:
            combat_result = "victory"
    return TransitionFacts(
        hp_delta=after_hp - before_hp,
        gold_delta=after_gold - before_gold,
        floor_delta=after_floor - before_floor,
        cards_added=_multiset_delta(before_deck, after_deck),
        cards_removed=_multiset_delta(after_deck, before_deck),
        potions_added=_multiset_delta(before_potions, after_potions),
        potions_removed=_multiset_delta(after_potions, before_potions),
        room_entered=(
            str(after_run.get("room_type") or after.get("room_type"))
            if (after_run.get("room_type") or after.get("room_type")) is not None
            else None
        ),
        combat_result=combat_result,
        terminal_reason=terminal_reason,
        enemy_hp_delta=_enemy_hp_total(before) - _enemy_hp_total(after),
        backend_reward=backend_reward,
    )


@dataclass(frozen=True, slots=True)
class CanonicalTransition:
    episode_id: str
    step_index: int
    before_state_version: int
    after_state_version: int
    facts: TransitionFacts
    action_handle: str | None = None
    observation: dict[str, Any] = field(default_factory=dict)
    terminated: bool = False
    truncated: bool = False
    info: dict[str, Any] = field(default_factory=dict)
    schema_version: str = TRANSITION_SCHEMA_VERSION
    backend_name: str = "legacy"


@dataclass(frozen=True, slots=True)
class RewardSpec:
    """Pure v2 reward weights; backend reward is disabled by default."""

    version: str = REWARD_SPEC_VERSION
    backend_reward_weight: float = 0.0
    hp_delta_weight: float = 0.02
    enemy_hp_delta_weight: float = 0.01
    gold_delta_weight: float = 0.001
    floor_delta_weight: float = 0.1
    cards_added_weight: float = 0.0
    cards_removed_weight: float = 0.0
    potions_added_weight: float = 0.0
    potions_removed_weight: float = 0.0
    combat_win_bonus: float = 1.0
    death_penalty: float = -1.0

    @classmethod
    def legacy_passthrough(cls) -> RewardSpec:
        return cls(
            version="legacy-backend-reward-v1",
            backend_reward_weight=1.0,
            hp_delta_weight=0.0,
            enemy_hp_delta_weight=0.0,
            gold_delta_weight=0.0,
            floor_delta_weight=0.0,
            cards_added_weight=0.0,
            cards_removed_weight=0.0,
            potions_added_weight=0.0,
            potions_removed_weight=0.0,
            combat_win_bonus=0.0,
            death_penalty=0.0,
        )

    @property
    def fingerprint(self) -> str:
        return "|".join(
            str(value)
            for value in (
                self.version,
                self.backend_reward_weight,
                self.hp_delta_weight,
                self.enemy_hp_delta_weight,
                self.gold_delta_weight,
                self.floor_delta_weight,
                self.cards_added_weight,
                self.cards_removed_weight,
                self.potions_added_weight,
                self.potions_removed_weight,
                self.combat_win_bonus,
                self.death_penalty,
            )
        )


@dataclass(frozen=True, slots=True)
class RewardBreakdown:
    total: float
    components: dict[str, float]
    spec_version: str
    spec_fingerprint: str


class VersionedRewardCalculator:
    """The v2 fact-based reward calculator."""

    def __init__(self, spec: RewardSpec | None = None) -> None:
        self.spec = spec or RewardSpec()

    @classmethod
    def legacy_passthrough(cls) -> VersionedRewardCalculator:
        return LegacyRewardCalculator()

    def evaluate(self, transition: CanonicalTransition) -> RewardBreakdown:
        facts = transition.facts
        components = {
            "legacy_backend_reward": facts.backend_reward * self.spec.backend_reward_weight,
            "hp_delta": facts.hp_delta * self.spec.hp_delta_weight,
            "enemy_hp_delta": facts.enemy_hp_delta * self.spec.enemy_hp_delta_weight,
            "gold_delta": facts.gold_delta * self.spec.gold_delta_weight,
            "floor_delta": float(facts.floor_delta) * self.spec.floor_delta_weight,
            "cards_added": float(len(facts.cards_added)) * self.spec.cards_added_weight,
            "cards_removed": float(len(facts.cards_removed)) * self.spec.cards_removed_weight,
            "potions_added": float(len(facts.potions_added)) * self.spec.potions_added_weight,
            "potions_removed": float(len(facts.potions_removed)) * self.spec.potions_removed_weight,
            "combat_victory": self.spec.combat_win_bonus if facts.combat_result == "victory" else 0.0,
            "player_death": self.spec.death_penalty if facts.combat_result == "defeat" else 0.0,
        }
        return RewardBreakdown(
            total=float(sum(components.values())),
            components=components,
            spec_version=self.spec.version,
            spec_fingerprint=self.spec.fingerprint,
        )


class LegacyRewardCalculator(VersionedRewardCalculator):
    """Explicit migration adapter for legacy backend-computed rewards."""

    def __init__(self) -> None:
        super().__init__(RewardSpec.legacy_passthrough())


def canonicalize_legacy_transition(
    result: EnvironmentResult | Mapping[str, Any],
    *,
    action_handle: str | None = None,
    backend_name: str = "legacy",
) -> CanonicalTransition:
    """Adapt a v1 result while keeping legacy reward provenance explicit."""

    typed = result if isinstance(result, EnvironmentResult) else EnvironmentResult.from_legacy(result)
    # Contract v2 exposes the canonical transition at the result top level;
    # older adapters placed bare facts under info.transition_facts.  Accept both
    # shapes, but never mistake the transition envelope itself for its facts.
    transition_payload: Mapping[str, Any] | None = None
    for candidate in (
        typed.raw.get("transition"),
        typed.raw.get("transition_facts"),
        typed.info.get("transition"),
        typed.info.get("transition_facts"),
    ):
        if isinstance(candidate, Mapping):
            transition_payload = candidate
            break
    nested_facts = transition_payload.get("facts") if transition_payload is not None else None
    fact_payload = nested_facts if isinstance(nested_facts, Mapping) else transition_payload
    facts = TransitionFacts.from_mapping(
        fact_payload if isinstance(fact_payload, Mapping) else None,
        backend_reward=typed.reward,
    )
    transition_after_version = (
        transition_payload.get("after_state_version")
        if transition_payload is not None
        else None
    )
    transition_before_version = (
        transition_payload.get("before_state_version")
        if transition_payload is not None
        else None
    )
    after_version = _integer(
        transition_after_version
        if transition_after_version is not None
        else typed.info.get(
            "after_state_version",
            typed.observation.get("state_version", typed.raw.get("state_version", typed.step_index)),
        ),
        typed.step_index,
    )
    before_version = _integer(
        transition_before_version
        if transition_before_version is not None
        else typed.info.get("before_state_version", max(after_version - 1, 0)),
        max(after_version - 1, 0),
    )
    return CanonicalTransition(
        episode_id=typed.episode_id,
        step_index=typed.step_index,
        before_state_version=before_version,
        after_state_version=after_version,
        facts=facts,
        action_handle=action_handle,
        observation=dict(typed.observation),
        terminated=typed.terminated,
        truncated=typed.truncated,
        info=dict(typed.info),
        backend_name=str(backend_name),
    )
