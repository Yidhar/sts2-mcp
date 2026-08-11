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
from typing import Any, Final, Literal

from .grouping import semantic_card_projection
from .identity import canonical_payload_bytes

FORWARD_COMPILER_CONTRACT_VERSION: Final = "sts2-forward-compiler-v3"

_ITEM_FACT_KEYS: Final[tuple[str, ...]] = (
    "category",
    "cost",
    "index",
    "can_afford",
)
_ROOT_DISPATCH_KEYS: Final[frozenset[str]] = frozenset(
    {
        "action_handle",
        "action_id",
        "action_index",
        "card_index",
        "choice_index",
        "idx",
        "index",
        "option_index",
    }
)
_COMBAT_SELECTION_ADD_KINDS: Final[frozenset[str]] = frozenset(
    {"select_card", "select_hand_card", "combat_select_card", "select_card_option"}
)
_COMBAT_SELECTION_COMMIT_KINDS: Final[frozenset[str]] = frozenset(
    {"confirm_selection", "combat_confirm_selection"}
)
_COMBAT_SELECTION_REVERSE_KINDS: Final[frozenset[str]] = frozenset(
    {
        "deselect_card",
        "deselect_hand_card",
        "combat_deselect_card",
        "deselect_card_option",
        "cancel_selection",
    }
)


def _token(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _card_facts(card: Mapping[str, Any]) -> dict[str, Any]:
    return semantic_card_projection(card)


def _target_key(payload: Mapping[str, Any]) -> str:
    return canonical_payload_bytes(payload).decode("utf-8")


def _semantic_action_key(action: Mapping[str, Any]) -> bytes:
    """Return exact model semantics without native dispatch coordinates."""

    projected = {
        key: value
        for key, value in action.items()
        if _token(key) not in _ROOT_DISPATCH_KEYS
    }
    return canonical_payload_bytes(projected)


def _semantic_action(
    native_action: Mapping[str, Any],
    *,
    model_kind: str,
    branch: str,
    target: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the model-facing action for one semantic candidate.

    Dispatch identity remains in :attr:`SemanticCandidate.native_index`; the
    model-facing payload is free to bind the strategic target that is chosen
    only after the native entry action.  This is what makes ``Smith(A)`` and
    ``Smith(B)`` independent candidates rather than aliases of one rest-site
    Q value.
    """

    action = dict(native_action)
    action["model_action_kind"] = model_kind
    action["model_action_variant"] = branch
    action["is_enabled"] = True
    if target is not None:
        action["card"] = dict(target)
    return action


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
    semantic_action: Mapping[str, Any]
    native_index: int | None = None

    def __post_init__(self) -> None:
        if not self.branch:
            raise ValueError("semantic candidate requires a branch")
        if not self.plan:
            raise ValueError("semantic candidate requires an executor plan")
        if not self.semantic_action:
            raise ValueError("semantic candidate requires a model-facing action")


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


def _deduplicate_candidates(
    candidates: Sequence[SemanticCandidate],
) -> tuple[SemanticCandidate, ...]:
    """Keep one learned action for each exact branch/target alternative.

    Deck facts can legitimately contain several physically interchangeable
    copies whose complete visible target projection is identical.  Choosing
    any such copy has the same strategic effect, so exposing one Q entry per
    copy creates duplicate semantic rows and can disagree with the encoder's
    strict-equivalence surface.  Distinct copy facts remain distinct targets;
    a candidate is folded only when ``branch``, canonical ``target_key``, and
    the full model-facing action semantics all agree.  Native dispatch
    coordinates do not make two strategic actions different.
    """

    result: list[SemanticCandidate] = []
    seen: set[tuple[str, str | None, bytes]] = set()
    for candidate in candidates:
        identity = (
            candidate.branch,
            candidate.target_key,
            _semantic_action_key(candidate.semantic_action),
        )
        if identity in seen:
            continue
        seen.add(identity)
        result.append(candidate)
    return tuple(result)


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
        raw_option = action.get("option")
        option: Mapping[str, Any] = raw_option if isinstance(raw_option, Mapping) else {}
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
                        semantic_action=_semantic_action(
                            action,
                            model_kind="rest_site",
                            branch="smith",
                        ),
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
                        semantic_action=_semantic_action(
                            action,
                            model_kind="rest_site",
                            branch="smith",
                            target=facts,
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
                    semantic_action=_semantic_action(
                        action,
                        model_kind="rest_site",
                        branch=branch,
                    ),
                    native_index=index,
                )
            )
    return _deduplicate_candidates(candidates) if candidates else None


def _is_shop_observation(observation: Mapping[str, Any]) -> bool:
    surface_tokens = {
        _token(observation.get(key))
        for key in (
            "phase",
            "decision_domain",
            "state_type",
            "screen_type",
            "room_type",
        )
    }
    for container_key in ("room", "current_room"):
        container = observation.get(container_key)
        if isinstance(container, Mapping):
            surface_tokens.update(
                _token(container.get(key))
                for key in ("type", "room_type", "point_type", "state_type")
            )
    if surface_tokens & {"shop", "merchant", "shop_room", "merchant_room"}:
        return True
    shop = observation.get("shop")
    return isinstance(shop, Mapping) and (
        shop.get("is_open") is True
        or shop.get("visible") is True
        or isinstance(shop.get("items"), Sequence)
    )


def _shop_candidates(
    observation: Mapping[str, Any],
    legal_actions: Sequence[Mapping[str, Any]],
) -> tuple[SemanticCandidate, ...] | None:
    if not _is_shop_observation(observation):
        return None
    purchases = _actions_of_kind(legal_actions, "shop_purchase")
    proceeds = _actions_of_kind(legal_actions, "proceed", "shop_skip", "shop_leave")
    if not purchases and not proceeds:
        return None
    deck = (observation.get("player") or {}).get("deck")
    deck = deck if isinstance(deck, list) else []
    candidates: list[SemanticCandidate] = []
    for index, action in purchases:
        raw_item = action.get("item")
        item: Mapping[str, Any] = raw_item if isinstance(raw_item, Mapping) else {}
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
                        semantic_action=_semantic_action(
                            action,
                            model_kind="shop",
                            branch="remove",
                        ),
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
                        semantic_action=_semantic_action(
                            action,
                            model_kind="shop",
                            branch="remove",
                            target=card_facts,
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
                    semantic_action=_semantic_action(
                        action,
                        model_kind="shop",
                        branch=branch,
                    ),
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
                semantic_action=_semantic_action(
                    action,
                    model_kind="shop",
                    branch="leave",
                ),
                native_index=index,
            )
        )
    return _deduplicate_candidates(candidates) if candidates else None


def _single_step_candidates(
    legal_actions: Sequence[Mapping[str, Any]],
    *,
    kind: str,
    model_kind: str,
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
                semantic_action=_semantic_action(
                    action,
                    model_kind=model_kind,
                    branch=branch,
                    target=target if target_field == "card" else None,
                ),
                native_index=index,
            )
        )
    return _deduplicate_candidates(result)


def _combat_candidates(
    legal_actions: Sequence[Mapping[str, Any]],
) -> tuple[SemanticCandidate, ...] | None:
    """Compile combat decisions, with monotone multi-select transactions.

    A combat picker is a growing set followed by Commit.  Deselect and cancel
    are executor reversals, not useful strategic actions; exposing them made
    select/deselect cycles part of the learned policy.  Normal combat actions
    remain native-atomic.
    """

    kinds = [_token(action.get("kind") or action.get("action")) for action in legal_actions]
    selection_surface = any(
        kind
        in (
            _COMBAT_SELECTION_ADD_KINDS
            | _COMBAT_SELECTION_COMMIT_KINDS
            | _COMBAT_SELECTION_REVERSE_KINDS
        )
        for kind in kinds
    )
    candidates: list[SemanticCandidate] = []
    for index, action in enumerate(legal_actions):
        kind = _token(action.get("kind") or action.get("action"))
        if not kind:
            return None  # unnameable action: fail closed for the whole surface
        if selection_surface:
            if kind in _COMBAT_SELECTION_ADD_KINDS:
                branch = "selection_add"
            elif kind in _COMBAT_SELECTION_COMMIT_KINDS:
                branch = "selection_commit"
            else:
                continue
            model_kind = "card_selection"
        else:
            branch = kind
            model_kind = str(action.get("model_action_kind") or kind)
        target: Mapping[str, Any] | None = None
        if isinstance(action.get("card"), Mapping):
            target = _card_facts(action["card"])
        elif isinstance(action.get("potion"), Mapping):
            target = dict(action["potion"])
        if isinstance(action.get("target"), Mapping):
            merged = dict(target or {})
            merged["combat_target"] = {
                key: action["target"].get(key)
                for key in ("id", "instance_id", "index")
                if action["target"].get(key) is not None
            }
            target = merged
        candidates.append(
            SemanticCandidate(
                branch=branch,
                target_key=_target_key(target) if target else None,
                target=target,
                plan=(NativeStep(kind=kind, target=target),),
                semantic_action=_semantic_action(
                    action,
                    model_kind=model_kind,
                    branch=branch,
                ),
                native_index=index,
            )
        )
    return _deduplicate_candidates(candidates) if candidates else None


def forward_decision(
    observation: Mapping[str, Any],
    legal_actions: Sequence[Mapping[str, Any]],
    *,
    control_domain: Literal["macro", "combat"] = "macro",
) -> ForwardDecision | None:
    """Compile the current macro surface into atomic semantic candidates.

    Returns ``None`` for any surface this module does not recognize — the
    caller then exposes the raw native decision unchanged (fail-closed).
    One compiler call belongs to exactly one control domain.  A macro owner
    never receives combat candidates and a combat owner never receives macro
    candidates; joined control is expressed by two authorities, not a flag on
    one recurrent state.
    """

    if control_domain not in {"macro", "combat"}:
        raise ValueError("control_domain must be 'macro' or 'combat'")

    combat = observation.get("combat")
    if isinstance(combat, Mapping) and combat.get("in_progress") is True:
        if control_domain != "combat":
            return None
        combat_candidates = _combat_candidates(legal_actions)
        if combat_candidates is None:
            return None
        return ForwardDecision(surface="combat", candidates=combat_candidates)

    if control_domain == "combat":
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
            legal_actions,
            kind="select_card_reward",
            model_kind="card_reward",
            branch="take",
            target_field="card",
        )
    )
    rewards.extend(
        _single_step_candidates(
            legal_actions,
            kind="skip_card_reward",
            model_kind="card_reward",
            branch="skip",
            target_field=None,
        )
    )
    if rewards:
        return ForwardDecision(
            surface="reward",
            candidates=_deduplicate_candidates(rewards),
        )

    nodes = _single_step_candidates(
        legal_actions,
        kind="choose_map_node",
        model_kind="map",
        branch="map",
        target_field="map_node",
    )
    if nodes:
        return ForwardDecision(surface="map", candidates=nodes)

    options = _single_step_candidates(
        legal_actions,
        kind="choose_event_option",
        model_kind="event_option",
        branch="event_option",
        target_field="option",
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
