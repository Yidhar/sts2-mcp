"""Forward compilation: atomic semantic candidates at the parent surface.

The backward compiler (compiler.py) folds recorded native traffic into
composite decisions.  Forward compilation is its collection-time mirror: at a
macro parent surface it enumerates the atomic strategic candidates directly —
``Rest``, ``Smith(card)``, ``RemoveCard(card)``, ``Buy(item)``, ``LeaveShop``,
``TakeReward(card)``, ``SkipReward``, ``ChooseMap(node)``,
``ChooseEventOption(option)`` — each with the native executor plan realizing
it.  Candidates come exclusively from authoritative observation facts (deck
lists, shop inventories, option rosters); no preference, no prediction, no
card-specific rule.  A surface this module does not recognize returns ``None``
and the caller exposes the raw native decision unchanged (fail-closed).

Branches follow the reset contract: strategic alternatives are branches
(``rest`` vs ``smith``; ``take`` vs ``skip``; ``buy_*``/``remove`` vs
``leave``), and multiple cards or items populate only the target distribution
inside their branch, so a singleton refusal keeps an independent exploration
share without any per-operation remap.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from .identity import canonical_payload_bytes

FORWARD_COMPILER_CONTRACT_VERSION: Final = "sts2-forward-compiler-v1"

_CARD_FACT_KEYS: Final[tuple[str, ...]] = (
    "id",
    "instance_id",
    "cost",
    "is_upgraded",
    "is_upgradable",
    "is_removable",
    "type",
)
_ITEM_FACT_KEYS: Final[tuple[str, ...]] = (
    "category",
    "cost",
    "index",
    "can_afford",
)


def _token(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _card_facts(card: Mapping[str, Any]) -> dict[str, Any]:
    return {key: card.get(key) for key in _CARD_FACT_KEYS if card.get(key) is not None}


def _target_key(payload: Mapping[str, Any]) -> str:
    return canonical_payload_bytes(payload).decode("utf-8")


@dataclass(frozen=True, slots=True)
class NativeStep:
    """One executor instruction: find-and-dispatch a native action.

    ``kind`` matches the native action kind; ``target`` (when present) must
    match the native action's card/item facts.  The executor fails closed if
    no unique native action matches.
    """

    kind: str
    option_token: str | None = None
    target: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class SemanticCandidate:
    branch: str
    target_key: str | None
    target: Mapping[str, Any] | None
    plan: tuple[NativeStep, ...]
    native_index: int | None = None

    def __post_init__(self) -> None:
        if not self.branch:
            raise ValueError("semantic candidate requires a branch")
        if not self.plan:
            raise ValueError("semantic candidate requires an executor plan")


@dataclass(frozen=True, slots=True)
class ForwardDecision:
    surface: str
    candidates: tuple[SemanticCandidate, ...]

    @property
    def branches(self) -> tuple[str, ...]:
        seen: list[str] = []
        for candidate in self.candidates:
            if candidate.branch not in seen:
                seen.append(candidate.branch)
        return tuple(seen)


def _actions_of_kind(
    legal_actions: Sequence[Mapping[str, Any]],
    *kinds: str,
) -> list[tuple[int, Mapping[str, Any]]]:
    wanted = {_token(kind) for kind in kinds}
    return [
        (index, action)
        for index, action in enumerate(legal_actions)
        if _token(action.get("kind") or action.get("action")) in wanted
    ]


def _rest_candidates(
    observation: Mapping[str, Any],
    legal_actions: Sequence[Mapping[str, Any]],
) -> tuple[SemanticCandidate, ...] | None:
    rest_actions = _actions_of_kind(legal_actions, "choose_rest_option")
    if not rest_actions:
        return None
    deck = (observation.get("player") or {}).get("deck")
    deck = deck if isinstance(deck, list) else []
    candidates: list[SemanticCandidate] = []
    for index, action in rest_actions:
        option = action.get("option") if isinstance(action.get("option"), Mapping) else {}
        option_token = _token(
            option.get("type")
            or option.get("id")
            or action.get("label")
            or action.get("idx")
        )
        if "smith" in option_token or "upgrade" in option_token or "forge" in option_token:
            targets = [
                card
                for card in deck
                if isinstance(card, Mapping) and card.get("is_upgradable") is True
            ]
            if not targets:
                # No enumerable target: expose the native entry unchanged so
                # legality authority stays with the engine.
                candidates.append(
                    SemanticCandidate(
                        branch="smith",
                        target_key=None,
                        target=None,
                        plan=(NativeStep(kind="choose_rest_option", option_token=option_token),),
                        native_index=index,
                    )
                )
                continue
            for card in targets:
                facts = _card_facts(card)
                candidates.append(
                    SemanticCandidate(
                        branch="smith",
                        target_key=_target_key(facts),
                        target=facts,
                        plan=(
                            NativeStep(kind="choose_rest_option", option_token=option_token),
                            NativeStep(kind="select_card", target=facts),
                            NativeStep(kind="confirm_selection"),
                        ),
                        native_index=index,
                    )
                )
        else:
            branch = "rest" if "heal" in option_token or "rest" in option_token else (
                option_token or "rest_option"
            )
            candidates.append(
                SemanticCandidate(
                    branch=branch,
                    target_key=None,
                    target=None,
                    plan=(NativeStep(kind="choose_rest_option", option_token=option_token),),
                    native_index=index,
                )
            )
    return tuple(candidates) if candidates else None


def _shop_candidates(
    observation: Mapping[str, Any],
    legal_actions: Sequence[Mapping[str, Any]],
) -> tuple[SemanticCandidate, ...] | None:
    purchases = _actions_of_kind(legal_actions, "shop_purchase")
    proceeds = _actions_of_kind(legal_actions, "proceed", "shop_skip", "shop_leave")
    if not purchases and not proceeds:
        return None
    deck = (observation.get("player") or {}).get("deck")
    deck = deck if isinstance(deck, list) else []
    candidates: list[SemanticCandidate] = []
    for index, action in purchases:
        item = action.get("item") if isinstance(action.get("item"), Mapping) else {}
        category = _token(item.get("category") or item.get("type"))
        facts = {key: item.get(key) for key in _ITEM_FACT_KEYS if item.get(key) is not None}
        if isinstance(item.get("card"), Mapping):
            facts["card"] = _card_facts(item["card"])
        if category == "card_removal":
            removable = [
                card
                for card in deck
                if isinstance(card, Mapping) and card.get("is_removable") is not False
            ]
            if not removable:
                candidates.append(
                    SemanticCandidate(
                        branch="remove",
                        target_key=None,
                        target=None,
                        plan=(NativeStep(kind="shop_purchase", target=facts),),
                        native_index=index,
                    )
                )
                continue
            for card in removable:
                card_facts = _card_facts(card)
                candidates.append(
                    SemanticCandidate(
                        branch="remove",
                        target_key=_target_key(card_facts),
                        target=card_facts,
                        plan=(
                            NativeStep(kind="shop_purchase", target=facts),
                            NativeStep(kind="select_card", target=card_facts),
                            NativeStep(kind="confirm_selection"),
                        ),
                        native_index=index,
                    )
                )
        else:
            branch = f"buy_{category}" if category else "buy"
            candidates.append(
                SemanticCandidate(
                    branch=branch,
                    target_key=_target_key(facts) if facts else None,
                    target=facts or None,
                    plan=(NativeStep(kind="shop_purchase", target=facts),),
                    native_index=index,
                )
            )
    for index, action in proceeds:
        candidates.append(
            SemanticCandidate(
                branch="leave",
                target_key=None,
                target=None,
                plan=(NativeStep(kind=_token(action.get("kind") or action.get("action"))),),
                native_index=index,
            )
        )
    return tuple(candidates) if candidates else None


def _single_step_candidates(
    legal_actions: Sequence[Mapping[str, Any]],
    *,
    kind: str,
    branch: str,
    target_field: str | None,
) -> tuple[SemanticCandidate, ...]:
    result: list[SemanticCandidate] = []
    for index, action in _actions_of_kind(legal_actions, kind):
        target: Mapping[str, Any] | None = None
        if target_field is not None and isinstance(action.get(target_field), Mapping):
            target = (
                _card_facts(action[target_field])
                if target_field == "card"
                else dict(action[target_field])
            )
        result.append(
            SemanticCandidate(
                branch=branch,
                target_key=_target_key(target) if target else None,
                target=target,
                plan=(NativeStep(kind=kind, target=target),),
                native_index=index,
            )
        )
    return tuple(result)


def forward_decision(
    observation: Mapping[str, Any],
    legal_actions: Sequence[Mapping[str, Any]],
) -> ForwardDecision | None:
    """Compile the current macro surface into atomic semantic candidates.

    Returns ``None`` for any surface this module does not recognize — the
    caller then exposes the raw native decision unchanged (fail-closed).
    Combat surfaces are deliberately not handled here: combat control belongs
    to the combat domain owner.
    """

    combat = observation.get("combat")
    if isinstance(combat, Mapping) and combat.get("in_progress") is True:
        return None

    rest = _rest_candidates(observation, legal_actions)
    if rest is not None:
        return ForwardDecision(surface="rest", candidates=rest)

    shop = _shop_candidates(observation, legal_actions)
    if shop is not None:
        return ForwardDecision(surface="shop", candidates=shop)

    rewards: list[SemanticCandidate] = []
    rewards.extend(
        _single_step_candidates(
            legal_actions, kind="select_card_reward", branch="take", target_field="card"
        )
    )
    rewards.extend(
        _single_step_candidates(
            legal_actions, kind="skip_card_reward", branch="skip", target_field=None
        )
    )
    if rewards:
        return ForwardDecision(surface="reward", candidates=tuple(rewards))

    nodes = _single_step_candidates(
        legal_actions, kind="choose_map_node", branch="map", target_field="map_node"
    )
    if nodes:
        return ForwardDecision(surface="map", candidates=nodes)

    options = _single_step_candidates(
        legal_actions, kind="choose_event_option", branch="event_option", target_field="option"
    )
    if options:
        return ForwardDecision(surface="event", candidates=options)

    return None


__all__ = [
    "FORWARD_COMPILER_CONTRACT_VERSION",
    "ForwardDecision",
    "NativeStep",
    "SemanticCandidate",
    "forward_decision",
]
