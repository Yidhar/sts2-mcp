"""Bounded factual detection of simulator-threatening combat state growth.

This module does not choose or rewrite actions.  It observes only public combat,
legal-action, card-zone, and card-type facts after an accepted transition.  The
collector can therefore classify a strategically lost combat before dispatching
the *next* mutation, while the accepted prefix remains eligible for episodic
learning.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

_GUARD_SEMANTICS_VERSION = "combat-runaway-status-guard-v1"
_VISIBLE_COMBAT_ZONES = ("hand", "draw_pile", "discard_pile", "exhaust_pile")

# These are deliberately conservative safety constants, not reward coefficients
# or game-specific strategy.  A normal combat cannot accidentally accumulate a
# thousand visible Status cards.  Requiring both a forced end-turn surface and
# a meaningful no-net-HP-progress age prevents a large but actionable state from
# being labelled as a policy failure.
_MINIMUM_NO_NET_PROGRESS_STEPS = 16
_RUNAWAY_STATUS_CARD_FLOOR = 1_024
_CATASTROPHIC_STATUS_CARD_FLOOR = 4_096
_MINIMUM_STATUS_FRACTION = 0.75
_MINIMUM_OBSERVED_STATUS_GROWTH = 512
_GROWTH_HISTORY_LENGTH = 8


def _nonnegative_integer(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("combat card-zone count must be a non-negative integer")
    return value


def _positive_quantity(card: Mapping[str, object]) -> int:
    raw = card.get("quantity", 1)
    if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
        raise ValueError("combat card quantity must be a positive integer")
    return raw


def _card_sequence(value: object) -> tuple[Mapping[str, object], ...] | None:
    if isinstance(value, Mapping):
        value = value.get("cards", value.get("items"))
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        return None
    if any(not isinstance(card, Mapping) for card in value):
        raise ValueError("combat card zone must contain only card mappings")
    return tuple(card for card in value if isinstance(card, Mapping))


def _is_status_card(card: Mapping[str, object]) -> bool:
    raw_type = card.get("type", card.get("card_type"))
    return isinstance(raw_type, str) and raw_type.strip().casefold() == "status"


def _action_kind(action: Mapping[str, object]) -> str:
    for key in ("model_action_kind", "kind", "action", "transport_kind"):
        value = action.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().casefold()
    action_id = action.get("action_id", action.get("action_handle"))
    if isinstance(action_id, str) and action_id.strip():
        return action_id.rsplit(":", 1)[-1].strip().casefold()
    return ""


def _only_end_turn(legal_actions: Sequence[Mapping[str, object]]) -> bool:
    return bool(legal_actions) and all(
        _action_kind(action) == "end_turn" for action in legal_actions
    )


@dataclass(frozen=True, slots=True)
class CombatCardBurden:
    """Visible card quantities; Status count is factual, never ID-inferred."""

    total_cards: int
    status_cards: int
    known_zones: tuple[str, ...]

    @property
    def status_fraction(self) -> float:
        if self.total_cards <= 0:
            return 0.0
        return self.status_cards / self.total_cards


def combat_card_burden(observation: Mapping[str, object]) -> CombatCardBurden:
    raw_player = observation.get("player")
    if not isinstance(raw_player, Mapping):
        return CombatCardBurden(total_cards=0, status_cards=0, known_zones=())

    total_cards = 0
    status_cards = 0
    known_zones: list[str] = []
    for zone in _VISIBLE_COMBAT_ZONES:
        explicit_count = _nonnegative_integer(raw_player.get(f"{zone}_count"))
        cards = _card_sequence(raw_player.get(zone))
        listed_total: int | None = None
        if cards is not None:
            listed_total = sum(_positive_quantity(card) for card in cards)
            status_cards += sum(
                _positive_quantity(card) for card in cards if _is_status_card(card)
            )
        zone_total = explicit_count if explicit_count is not None else listed_total
        if zone_total is None:
            continue
        if listed_total is not None and listed_total > zone_total:
            raise ValueError("visible combat cards exceed the authoritative zone count")
        known_zones.append(zone)
        total_cards += zone_total

    if status_cards > total_cards:
        raise ValueError("visible Status-card count exceeds visible combat card count")
    return CombatCardBurden(
        total_cards=total_cards,
        status_cards=status_cards,
        known_zones=tuple(known_zones),
    )


@dataclass(frozen=True, slots=True)
class RunawayCombatStatus:
    triggered: bool
    reason: str
    total_cards: int
    status_cards: int
    status_fraction: float
    observed_status_growth: int
    only_end_turn: bool
    only_end_turn_streak: int
    no_net_progress_steps: int

    def evidence(self, *, detected_step: int) -> dict[str, object]:
        """Return one bounded journal/replay-adjacent diagnostic mapping."""

        return {
            "kind": "combat_runaway_status_burden",
            "guard_semantics_version": _GUARD_SEMANTICS_VERSION,
            "reason": self.reason,
            "detected_step": detected_step,
            "steps_without_net_progress": self.no_net_progress_steps,
            "only_end_turn": self.only_end_turn,
            "only_end_turn_streak": self.only_end_turn_streak,
            "visible_combat_cards": self.total_cards,
            "visible_status_cards": self.status_cards,
            "visible_status_fraction": self.status_fraction,
            "observed_status_growth": self.observed_status_growth,
            "minimum_no_net_progress_steps": _MINIMUM_NO_NET_PROGRESS_STEPS,
            "runaway_status_card_floor": _RUNAWAY_STATUS_CARD_FLOOR,
            "catastrophic_status_card_floor": _CATASTROPHIC_STATUS_CARD_FLOOR,
            "minimum_status_fraction": _MINIMUM_STATUS_FRACTION,
            "minimum_observed_status_growth": _MINIMUM_OBSERVED_STATUS_GROWTH,
            "growth_history_length": _GROWTH_HISTORY_LENGTH,
            "visible_combat_zones": _VISIBLE_COMBAT_ZONES,
            "status_type_value": "Status",
        }


class RunawayCombatGuard:
    """Detect a growing, Status-dominated forced-end-turn combat state.

    The detector is evaluated after each fully accepted transition.  A positive
    result means the collector can terminate that transition as a combat-policy
    failure without issuing another backend RPC.  It intentionally cannot make
    an action choice and has no access to policy logits or hidden simulator data.
    """

    def __init__(self) -> None:
        self._status_history: deque[int] = deque(maxlen=_GROWTH_HISTORY_LENGTH)
        self._only_end_turn_streak = 0
        self._combat_active = False

    def reset(self) -> None:
        self._status_history.clear()
        self._only_end_turn_streak = 0
        self._combat_active = False

    def observe(
        self,
        *,
        observation: Mapping[str, object],
        legal_actions: Sequence[Mapping[str, object]],
        no_net_progress_steps: int,
    ) -> RunawayCombatStatus:
        raw_combat = observation.get("combat")
        in_combat = bool(
            isinstance(raw_combat, Mapping) and raw_combat.get("in_progress") is True
        )
        if not in_combat:
            self.reset()
            return RunawayCombatStatus(
                triggered=False,
                reason="outside_combat",
                total_cards=0,
                status_cards=0,
                status_fraction=0.0,
                observed_status_growth=0,
                only_end_turn=False,
                only_end_turn_streak=0,
                no_net_progress_steps=max(0, int(no_net_progress_steps)),
            )
        if not self._combat_active:
            self._status_history.clear()
            self._only_end_turn_streak = 0
            self._combat_active = True

        burden = combat_card_burden(observation)
        forced_end_turn = _only_end_turn(legal_actions)
        self._only_end_turn_streak = (
            self._only_end_turn_streak + 1 if forced_end_turn else 0
        )
        history_minimum = min(self._status_history, default=burden.status_cards)
        observed_growth = max(0, burden.status_cards - history_minimum)
        self._status_history.append(burden.status_cards)

        age = max(0, int(no_net_progress_steps))
        status_dominated = burden.status_fraction >= _MINIMUM_STATUS_FRACTION
        catastrophic = burden.status_cards >= _CATASTROPHIC_STATUS_CARD_FLOOR
        rapidly_growing = bool(
            burden.status_cards >= _RUNAWAY_STATUS_CARD_FLOOR
            and observed_growth >= _MINIMUM_OBSERVED_STATUS_GROWTH
        )
        trigger = bool(
            forced_end_turn
            and age >= _MINIMUM_NO_NET_PROGRESS_STEPS
            and status_dominated
            and (catastrophic or rapidly_growing)
        )
        reason = (
            "catastrophic_status_card_burden"
            if trigger and catastrophic
            else "rapid_status_card_growth"
            if trigger
            else "conditions_not_met"
        )
        return RunawayCombatStatus(
            triggered=trigger,
            reason=reason,
            total_cards=burden.total_cards,
            status_cards=burden.status_cards,
            status_fraction=burden.status_fraction,
            observed_status_growth=observed_growth,
            only_end_turn=forced_end_turn,
            only_end_turn_streak=self._only_end_turn_streak,
            no_net_progress_steps=age,
        )


__all__ = [
    "CombatCardBurden",
    "RunawayCombatGuard",
    "RunawayCombatStatus",
    "combat_card_burden",
]
