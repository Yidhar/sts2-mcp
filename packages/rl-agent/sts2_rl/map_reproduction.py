"""On-demand deterministic map reconstruction for held-out run inspection.

Map topology is a reproducible environment fact, not a trajectory target.  The
training journal therefore keeps the seed and the actions that were actually
dispatched.  When an operator opens a run, this module replays those actions
against the exact pinned HeadlessSim build and captures the map at each Act
boundary.  Results live only in a small in-memory cache; no training artifact
or checkpoint is written.

The replay fails closed when simulator identity, action indexes, or observed
state types diverge.  In that case the dashboard continues to show the factual
route projection already present in the journal rather than inventing a map.
"""

from __future__ import annotations

import json
import math
import re
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Protocol, cast

from sts2_env.headless_sim_bridge_client import HeadlessSimBridgeClient
from sts2_rl.entity_localization import EntityKind, localized_entity_name

JsonDict = dict[str, Any]

MAP_REPLAY_SCHEMA: Final = "sts2-heldout-episode-replay-v2"
_MAX_LINE_BYTES: Final = 8 * 1024 * 1024
_MAX_REPLAY_ACTIONS: Final = 30_000
_MAX_REPLAY_SECONDS: Final = 120.0
_MAX_MAP_ACTS: Final = 8
_MAX_MAP_NODES: Final = 128
_MAX_MAP_CHILDREN: Final = 16
_MAX_MACRO_ACTIONS: Final = 1_024
_ACTION_HANDLE_RE = re.compile(r"^sim:(?P<index>[0-9]+):")
_MACRO_ACTION_KINDS: Final = frozenset(
    {
        "choose_event_option",
        "choose_rest_option",
        "shop_purchase",
        "claim_reward",
        "claim_treasure_relic",
        "select_card",
        "deselect_card",
        "confirm_selection",
        "cancel_selection",
    }
)
_MACRO_SCREENS: Final = frozenset({"EVENT", "REST_SITE", "SHOP", "CARD_SELECTION"})
_REST_OPTION_LABELS: Final = {
    "HEAL": "休息回血",
    "SMITH": "锻造升级",
    "DIG": "挖掘遗物",
    "LIFT": "举重强化",
    "RECALL": "回忆",
}
_OPERATION_LABELS: Final = {
    "upgrade": "升级",
    "remove": "删牌",
    "transform": "变换",
    "enchant": "附魔",
    "select": "选择",
}
_SHOP_CATEGORY_LABELS: Final = {
    "card": "卡牌",
    "relic": "遗物",
    "potion": "药水",
    "card_removal": "删牌服务",
}


class MapReplayUnavailable(ValueError):
    """The requested map cannot be reproduced without guessing."""


class _ReplayClient(Protocol):
    def __enter__(self) -> _ReplayClient: ...

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None: ...

    def reset(
        self,
        character: str | None = None,
        rebind_active_run: bool = False,
        force_fresh: bool = False,
        defensive_buffs: bool = False,
        additional_relics: list[str] | None = None,
        training_revival_budget: int | None = None,
        seed: str | int | None = None,
        ascension_level: int | None = None,
        timeout_ms: int = 45_000,
    ) -> JsonDict: ...

    def step(
        self,
        episode_id: str,
        action_index: int | None = None,
        action_id: str | None = None,
        timeout_ms: int = 20_000,
    ) -> JsonDict: ...


def _default_client_factory(executable: Path) -> _ReplayClient:
    return cast(_ReplayClient, HeadlessSimBridgeClient(exe_path=executable))


@dataclass(frozen=True, slots=True)
class _ReplayAction:
    step: int
    index: int
    expected_state_type: str | None


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _integer(value: object, default: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return default
    number = float(value)
    return int(number) if math.isfinite(number) else default


def _string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _action_kind(action: Mapping[str, Any]) -> str:
    return (
        _string(action.get("action"))
        or _string(action.get("kind"))
        or _string(action.get("model_action_kind"))
        or "unknown"
    )


def _collection_total(value: object) -> int:
    if not isinstance(value, list):
        return 0
    total = 0
    for item in value:
        quantity = _integer(_mapping(item).get("quantity"), 1)
        total += max(1, quantity)
    return total


def _player_resources(observation: Mapping[str, Any]) -> JsonDict:
    player = _mapping(observation.get("player"))
    return {
        "hp": _integer(player.get("hp")),
        "max_hp": _integer(player.get("max_hp")),
        "gold": _integer(player.get("gold")),
        "deck_count": _collection_total(player.get("deck")),
        "relic_count": _collection_total(player.get("relics")),
        "potion_count": _collection_total(player.get("potions")),
    }


def _entity_identifier(value: Mapping[str, Any], *, kind: EntityKind) -> str | None:
    fields_by_kind: dict[EntityKind, tuple[str, ...]] = {
        "card": ("card_id", "id"),
        "relic": ("relic_id", "id"),
        "potion": ("potion_id", "id"),
    }
    for key in fields_by_kind[kind]:
        identifier = _string(value.get(key))
        if identifier is not None:
            return identifier
    return None


def _project_entity(
    value: Mapping[str, Any],
    *,
    kind: EntityKind,
) -> JsonDict | None:
    identifier = _entity_identifier(value, kind=kind)
    if identifier is None:
        return None
    index_value = value.get("index", value.get("card_index", value.get("slot")))
    projected: JsonDict = {
        "kind": kind,
        "entity_id": identifier,
        "display_name": localized_entity_name(identifier, kind=kind),
        "index": _integer(index_value) if isinstance(index_value, int | float) else None,
    }
    if kind == "card":
        upgraded = value.get("is_upgraded")
        projected["is_upgraded"] = upgraded if isinstance(upgraded, bool) else None
        projected["type"] = _string(value.get("type"))
        projected["rarity"] = _string(value.get("rarity"))
    return projected


def _action_entity(action: Mapping[str, Any]) -> JsonDict | None:
    card = _mapping(action.get("card"))
    entity = _project_entity(card, kind="card")
    if entity is not None:
        return entity
    item = _mapping(action.get("item"))
    category = (_string(item.get("category")) or _string(item.get("type")) or "").lower()
    if category in {"card", "relic", "potion"}:
        kind = cast(EntityKind, category)
        nested = _mapping(item.get(category))
        return _project_entity(nested or item, kind=kind)
    for kind, key in (
        (cast(EntityKind, "card"), "card_id"),
        (cast(EntityKind, "relic"), "relic_id"),
        (cast(EntityKind, "potion"), "potion_id"),
    ):
        identifier = _string(action.get(key))
        if identifier is not None:
            return _project_entity({key: identifier}, kind=kind)
    return None


def _selected_card_entities(observation: Mapping[str, Any]) -> list[JsonDict]:
    selection = _mapping(observation.get("card_selection"))
    cards = selection.get("selected_cards")
    if not isinstance(cards, list):
        return []
    entities: list[JsonDict] = []
    for card in cards:
        entity = _project_entity(_mapping(card), kind="card")
        if entity is not None:
            entities.append(entity)
    return entities


def _entity_signature(entity: Mapping[str, Any]) -> str:
    return json.dumps(
        {
            "kind": entity.get("kind"),
            "entity_id": entity.get("entity_id"),
            "is_upgraded": entity.get("is_upgraded"),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _entity_multiset(
    observation: Mapping[str, Any],
    *,
    field: str,
    kind: EntityKind,
) -> dict[str, tuple[JsonDict, int]]:
    player = _mapping(observation.get("player"))
    values = player.get(field)
    if not isinstance(values, list):
        return {}
    result: dict[str, tuple[JsonDict, int]] = {}
    for raw in values:
        source = _mapping(raw)
        entity = _project_entity(source, kind=kind)
        if entity is None:
            continue
        quantity = max(1, _integer(source.get("quantity"), 1))
        signature = _entity_signature(entity)
        previous = result.get(signature)
        result[signature] = (entity, quantity + (previous[1] if previous else 0))
    return result


def _entity_changes(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    *,
    field: str,
    kind: EntityKind,
) -> tuple[list[JsonDict], list[JsonDict]]:
    old = _entity_multiset(before, field=field, kind=kind)
    new = _entity_multiset(after, field=field, kind=kind)
    gained: list[JsonDict] = []
    removed: list[JsonDict] = []
    for signature in sorted(set(old) | set(new)):
        old_entity, old_count = old.get(signature, ({}, 0))
        new_entity, new_count = new.get(signature, ({}, 0))
        if new_count > old_count:
            gained.append({**new_entity, "quantity": new_count - old_count})
        elif old_count > new_count:
            removed.append({**old_entity, "quantity": old_count - new_count})
    return gained, removed


def _entity_text(entity: Mapping[str, Any], *, upgraded_suffix: bool = False) -> str:
    name = _string(entity.get("display_name")) or _string(entity.get("entity_id")) or "未知对象"
    if upgraded_suffix and entity.get("is_upgraded") is not True:
        return f"{name}+"
    if entity.get("is_upgraded") is True and not name.endswith("+"):
        return f"{name}+"
    return name


def _entities_text(entities: list[JsonDict], *, upgraded_suffix: bool = False) -> str:
    labels: list[str] = []
    for entity in entities:
        quantity = max(1, _integer(entity.get("quantity"), 1))
        label = _entity_text(entity, upgraded_suffix=upgraded_suffix)
        labels.append(f"{label} x {quantity}" if quantity > 1 else label)
    return "、".join(labels)


def _positive_effect_text(effects: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for field, label in (
        ("gained_cards", "获得卡牌"),
        ("gained_relics", "获得遗物"),
        ("gained_potions", "获得药水"),
    ):
        entities = effects.get(field)
        if isinstance(entities, list) and entities:
            parts.append(f"{label} {_entities_text(entities)}")
    max_hp = _integer(effects.get("max_hp"))
    gold = _integer(effects.get("gold"))
    if max_hp > 0:
        parts.append(f"最大生命 +{max_hp}")
    if gold > 0:
        parts.append(f"金币 +{gold}")
    return "、".join(parts)


def _macro_action_fact(
    *,
    step: int,
    action: Mapping[str, Any],
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    legal_action_count: int,
) -> JsonDict | None:
    kind = _action_kind(action)
    screen = _string(before.get("screen")) or "UNKNOWN"
    if kind not in _MACRO_ACTION_KINDS and not (kind == "proceed" and screen in _MACRO_SCREENS):
        return None

    run = _mapping(before.get("run"))
    before_resources = _player_resources(before)
    after_resources = _player_resources(after)
    effects: JsonDict = {
        key: _integer(after_resources.get(key)) - _integer(before_resources.get(key))
        for key in before_resources
    }
    for field, entity_kind, gained_key, removed_key in (
        ("deck", cast(EntityKind, "card"), "gained_cards", "removed_cards"),
        ("relics", cast(EntityKind, "relic"), "gained_relics", "removed_relics"),
        ("potions", cast(EntityKind, "potion"), "gained_potions", "removed_potions"),
    ):
        gained, removed = _entity_changes(before, after, field=field, kind=entity_kind)
        if gained:
            effects[gained_key] = gained
        if removed:
            effects[removed_key] = removed

    entity = _action_entity(action)
    selection = _mapping(before.get("card_selection"))
    operation = (_string(selection.get("operation_type")) or "").lower()
    selected_entities = _selected_card_entities(before)
    automatic = legal_action_count == 1
    semantic_label = _string(action.get("label")) or kind
    positive_effect = _positive_effect_text(effects)

    if kind == "choose_rest_option":
        option = _mapping(action.get("option"))
        option_id = (_string(option.get("id")) or "").upper()
        option_label = _REST_OPTION_LABELS.get(option_id, f"休息点操作: {option_id or '未知'}")
        if option_id == "HEAL":
            hp_before = _integer(before_resources.get("hp"))
            hp_after = _integer(after_resources.get("hp"))
            semantic_label = f"{option_label}: {hp_before} → {hp_after} (+{max(0, hp_after - hp_before)})"
        else:
            semantic_label = option_label
            if positive_effect:
                semantic_label += f": {positive_effect}"
    elif kind in {"select_card", "deselect_card"}:
        operation_label = _OPERATION_LABELS.get(operation, operation or "卡牌")
        verb = "选择" if kind == "select_card" else "取消选择"
        entity_label = _entity_text(entity) if entity is not None else "未知卡牌"
        semantic_label = f"{verb}{operation_label}: {entity_label}"
    elif kind == "confirm_selection":
        operation_label = _OPERATION_LABELS.get(operation, operation or "选择")
        if operation == "upgrade":
            names = _entities_text(selected_entities, upgraded_suffix=True)
            semantic_label = f"完成升级: {names or '未知卡牌'}"
        elif operation == "remove":
            names = _entities_text(selected_entities)
            semantic_label = f"完成删牌: {names or '未知卡牌'}"
        elif operation == "transform":
            removed_effect = effects.get("removed_cards")
            gained_effect = effects.get("gained_cards")
            removed_text = _entities_text(removed_effect) if isinstance(removed_effect, list) else "未知卡牌"
            gained_text = _entities_text(gained_effect) if isinstance(gained_effect, list) else "未知卡牌"
            semantic_label = f"完成变换: {removed_text} → {gained_text}"
        else:
            names = _entities_text(selected_entities)
            semantic_label = f"确认{operation_label}: {names or '完成'}"
    elif kind == "cancel_selection":
        operation_label = _OPERATION_LABELS.get(operation, operation or "选择")
        semantic_label = f"取消{operation_label}"
    elif kind == "shop_purchase":
        item = _mapping(action.get("item"))
        category = (_string(item.get("category")) or _string(item.get("type")) or "unknown").lower()
        category_label = _SHOP_CATEGORY_LABELS.get(category, category)
        charged_now = max(0, -_integer(effects.get("gold")))
        quoted_cost = max(0, _integer(item.get("cost", item.get("price"))))
        # Card-removal purchases open a selection transaction and may charge
        # only when it commits.  Preserve the authoritative quoted item price
        # instead of rendering that entry step as a zero-cost purchase.
        cost = charged_now or quoted_cost
        if entity is not None:
            semantic_label = f"购买{category_label}: {_entity_text(entity)} ({cost} 金币)"
        else:
            semantic_label = f"购买{category_label} ({cost} 金币)"
    elif kind == "claim_treasure_relic":
        semantic_label = f"获得遗物: {_entity_text(entity) if entity is not None else '未知遗物'}"
    elif kind == "claim_reward":
        semantic_label = f"领取奖励: {positive_effect}" if positive_effect else "领取奖励"
    elif kind == "choose_event_option":
        option = _mapping(action.get("option"))
        option_label = _string(option.get("name")) or _string(option.get("id")) or semantic_label
        semantic_label = f"事件选择: {option_label}"
        if positive_effect:
            semantic_label += f"; {positive_effect}"
    elif kind == "proceed":
        location = {
            "REST_SITE": "休息点",
            "SHOP": "商店",
            "EVENT": "事件",
            "CARD_SELECTION": "选择界面",
        }.get(screen, "当前页面")
        semantic_label = f"自动离开{location}" if automatic else f"离开{location}"
        if automatic:
            semantic_label += " (唯一合法动作)"

    fact: JsonDict = {
        "step": step,
        "act": _integer(run.get("act")),
        "floor": _integer(run.get("floor")),
        "room_type": _string(run.get("room_type")),
        "screen": screen,
        "kind": kind,
        "automatic": automatic,
        "automatic_reason": "only_legal_action" if automatic else None,
        "semantic_label": semantic_label,
        "operation": operation or None,
        "entity": entity,
        "selected_entities": selected_entities,
        "positive_effect": positive_effect or None,
        "resources_before": before_resources,
        "resources_after": after_resources,
        "effects": effects,
    }
    return fact


def _selected_action_index(selected: Mapping[str, Any]) -> int | None:
    for field in ("action_index", "idx", "index"):
        value = selected.get(field)
        if isinstance(value, bool) or not isinstance(value, int | float):
            continue
        number = float(value)
        if math.isfinite(number) and number >= 0 and number.is_integer():
            return int(number)
    handle = _string(selected.get("action_handle")) or _string(selected.get("action_id"))
    match = _ACTION_HANDLE_RE.match(handle or "")
    return int(match.group("index")) if match is not None else None


def _episode_actions(path: Path, episode_id: str) -> list[_ReplayAction]:
    actions: list[_ReplayAction] = []
    completed = False
    last_step = -1
    try:
        with path.open("rb") as handle:
            while True:
                raw = handle.readline(_MAX_LINE_BYTES + 1)
                if not raw:
                    break
                if len(raw) > _MAX_LINE_BYTES:
                    raise MapReplayUnavailable("地图复现失败: journal 行超过受支持上限")
                if not raw.endswith(b"\n"):
                    raise MapReplayUnavailable("地图复现失败: journal 存在未完成尾行")
                try:
                    event = json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise MapReplayUnavailable("地图复现失败: journal 含损坏 JSON 行") from exc
                if not isinstance(event, dict) or event.get("episode_id") != episode_id:
                    continue
                kind = event.get("event")
                if kind == "evaluation_attempt_completed":
                    completed = True
                    break
                if kind != "decision":
                    continue
                if len(actions) >= _MAX_REPLAY_ACTIONS:
                    raise MapReplayUnavailable("地图复现失败: 单局动作数超过 30,000 上限")
                step = _integer(event.get("step_index"), -1)
                if step <= last_step:
                    raise MapReplayUnavailable("地图复现失败: decision step_index 非严格递增")
                selected = _mapping(event.get("selected_action"))
                index = _selected_action_index(selected)
                if index is None:
                    raise MapReplayUnavailable(f"地图复现失败: step {step} 缺少原始动作索引")
                observation = _mapping(event.get("observation_summary"))
                actions.append(
                    _ReplayAction(
                        step=step,
                        index=index,
                        expected_state_type=_string(observation.get("state_type")),
                    )
                )
                last_step = step
    except OSError as exc:
        raise MapReplayUnavailable("地图复现失败: journal 不可读") from exc
    if not actions:
        raise MapReplayUnavailable("地图复现失败: 该 episode 没有 compact decision")
    if not completed:
        raise MapReplayUnavailable("地图复现失败: 该 episode 没有完成标记")
    return actions


def _identity_payload(identity_path: Path) -> JsonDict:
    try:
        value = json.loads(identity_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MapReplayUnavailable("地图复现失败: 当前 HeadlessSim identity 不可读") from exc
    if not isinstance(value, dict):
        raise MapReplayUnavailable("地图复现失败: 当前 HeadlessSim identity 格式无效")
    return value


def _verify_identity(simulator_exe: Path, provenance: Mapping[str, Any]) -> JsonDict:
    if not simulator_exe.is_file():
        raise MapReplayUnavailable("地图复现不可用: 当前 runtime 缺少 HeadlessSim")
    identity = _identity_payload(Path(f"{simulator_exe}.identity.json"))
    expected = _mapping(provenance.get("simulator"))
    actual_binary = _mapping(identity.get("binary"))
    actual_managed = _mapping(identity.get("managed_binary"))
    expected_binary = _string(expected.get("binary_sha256"))
    expected_managed = _string(expected.get("managed_binary_sha256"))
    if expected_binary is None or expected_managed is None:
        raise MapReplayUnavailable("地图复现不可用: 历史 journal 未完整记录 simulator identity")
    if expected_binary != _string(actual_binary.get("sha256")):
        raise MapReplayUnavailable("地图复现不可用: HeadlessSim apphost 与历史 journal 不一致")
    if expected_managed != _string(actual_managed.get("sha256")):
        raise MapReplayUnavailable("地图复现不可用: HeadlessSim 托管实现与历史 journal 不一致")
    return {
        "schema_version": identity.get("schema_version"),
        "source_commit": _mapping(identity.get("source")).get("commit"),
        "binary_sha256": actual_binary.get("sha256"),
        "managed_binary_sha256": actual_managed.get("sha256"),
    }


def _project_coord(value: object) -> JsonDict | None:
    source = _mapping(value)
    x = source.get("x", source.get("col"))
    y = source.get("y", source.get("row"))
    if isinstance(x, bool) or not isinstance(x, int | float):
        return None
    if isinstance(y, bool) or not isinstance(y, int | float):
        return None
    if not math.isfinite(float(x)) or not math.isfinite(float(y)):
        return None
    return {"x": int(x), "y": int(y)}


def _project_topology(observation: Mapping[str, Any], *, replay_step: int) -> JsonDict | None:
    run = _mapping(observation.get("run"))
    map_state = _mapping(observation.get("map"))
    act = _integer(run.get("act"))
    raw_nodes = map_state.get("nodes")
    if act <= 0 or not isinstance(raw_nodes, list) or not raw_nodes:
        return None
    nodes: list[JsonDict] = []
    invalid_nodes = 0
    children_omitted = 0
    for raw_node in raw_nodes[:_MAX_MAP_NODES]:
        node = _mapping(raw_node)
        coord = _project_coord(node.get("coord"))
        if coord is None:
            invalid_nodes += 1
            continue
        raw_children = node.get("children")
        children_source = raw_children if isinstance(raw_children, list) else []
        children = [
            projected
            for child in children_source[:_MAX_MAP_CHILDREN]
            if (projected := _project_coord(child)) is not None
        ]
        children_omitted += max(0, len(children_source) - _MAX_MAP_CHILDREN)
        point_type = _string(node.get("point_type")) or "unknown"
        nodes.append({"coord": coord, "point_type": point_type, "children": children})
    if not nodes:
        return None
    nodes_omitted = max(0, len(raw_nodes) - _MAX_MAP_NODES)
    return {
        "act": act,
        "source": "seed_action_replay.observation.map.nodes",
        "replay_step": replay_step,
        "current_coord": _project_coord(map_state.get("current_coord")),
        "coverage": {
            "kind": "deterministic_seed_action_replay",
            "bounded": True,
            "complete": nodes_omitted == 0 and invalid_nodes == 0 and children_omitted == 0,
            "nodes_seen": len(raw_nodes),
            "nodes_projected": len(nodes),
            "nodes_omitted": nodes_omitted,
            "invalid_nodes_omitted": invalid_nodes,
            "children_omitted": children_omitted,
        },
        "nodes": nodes,
    }


def _required_acts(detail: Mapping[str, Any]) -> set[int]:
    acts: set[int] = set()
    for field in ("floors", "route"):
        rows = detail.get(field)
        if not isinstance(rows, list):
            continue
        for row in rows:
            act = _integer(_mapping(row).get("act"))
            if 0 < act <= _MAX_MAP_ACTS:
                acts.add(act)
    if not acts:
        max_act = _integer(detail.get("max_act"))
        if 0 < max_act <= _MAX_MAP_ACTS:
            acts.update(range(1, max_act + 1))
    return acts or {1}


def reconstruct_episode_maps(
    *,
    journal_path: Path,
    episode_id: str,
    detail: Mapping[str, Any],
    simulator_exe: Path,
    client_factory: Callable[[Path], _ReplayClient] = _default_client_factory,
) -> JsonDict:
    """Replay one completed episode and recover maps plus exact macro outcomes."""

    provenance = _mapping(detail.get("provenance"))
    identity = _verify_identity(simulator_exe, provenance)
    seed = detail.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int | str):
        raise MapReplayUnavailable("地图复现不可用: episode 缺少 seed")
    character = _string(detail.get("character"))
    if character is None:
        raise MapReplayUnavailable("地图复现不可用: episode 缺少角色")
    ascension = _integer(detail.get("ascension"))
    required_acts = _required_acts(detail)
    actions = _episode_actions(journal_path, episode_id)
    maps: dict[int, JsonDict] = {}
    macro_actions: list[JsonDict] = []
    macro_actions_omitted = 0
    actions_replayed = 0
    started = time.perf_counter()

    with client_factory(simulator_exe) as client:
        response = client.reset(
            character=character,
            seed=str(seed),
            ascension_level=ascension,
            force_fresh=True,
            training_revival_budget=-1,
            timeout_ms=45_000,
        )
        for action in actions:
            if time.perf_counter() - started > _MAX_REPLAY_SECONDS:
                raise MapReplayUnavailable("地图复现失败: 重放超过 120 秒时间上限")
            observation = _mapping(response.get("obs"))
            actual_state_type = _string(observation.get("state_type"))
            if action.expected_state_type and actual_state_type != action.expected_state_type:
                raise MapReplayUnavailable(
                    "地图复现失败: "
                    f"step {action.step} 状态分歧(历史 {action.expected_state_type}, 当前 {actual_state_type})"
                )
            topology = _project_topology(observation, replay_step=action.step)
            if topology is not None:
                maps.setdefault(_integer(topology.get("act")), topology)
            legal_actions = response.get("legal_actions")
            if not isinstance(legal_actions, list) or not 0 <= action.index < len(legal_actions):
                count = len(legal_actions) if isinstance(legal_actions, list) else 0
                raise MapReplayUnavailable(
                    f"地图复现失败: step {action.step} 动作索引 {action.index} 不在当前 {count} 个合法动作内"
                )
            episode = _string(response.get("episode_id"))
            if episode is None:
                raise MapReplayUnavailable("地图复现失败: HeadlessSim 响应缺少 episode_id")
            selected_action = _mapping(legal_actions[action.index])
            try:
                next_response = client.step(episode, action_index=action.index, timeout_ms=20_000)
            except Exception as exc:
                raise MapReplayUnavailable(f"地图复现失败: step {action.step} 模拟器拒绝动作") from exc
            next_observation = _mapping(next_response.get("obs"))
            fact = _macro_action_fact(
                step=action.step,
                action=selected_action,
                before=observation,
                after=next_observation,
                legal_action_count=len(legal_actions),
            )
            if fact is not None:
                if len(macro_actions) < _MAX_MACRO_ACTIONS:
                    macro_actions.append(fact)
                else:
                    macro_actions_omitted += 1
            next_topology = _project_topology(next_observation, replay_step=action.step + 1)
            if next_topology is not None:
                maps.setdefault(_integer(next_topology.get("act")), next_topology)
            response = next_response
            actions_replayed += 1

    missing = sorted(required_acts.difference(maps))
    if missing:
        raise MapReplayUnavailable(f"地图复现失败: 未到达 Act {', '.join(map(str, missing))} 的地图边界")
    elapsed = time.perf_counter() - started
    return {
        "schema": MAP_REPLAY_SCHEMA,
        "episode_id": episode_id,
        "map_topologies": [maps[act] for act in sorted(maps) if act in required_acts],
        "macro_actions": macro_actions,
        "macro_actions_omitted": macro_actions_omitted,
        "reproduction": {
            "mode": "seed_and_recorded_action_replay",
            "seed": seed,
            "character": character,
            "ascension": ascension,
            "required_acts": sorted(required_acts),
            "actions_available": len(actions),
            "actions_replayed": actions_replayed,
            "elapsed_seconds": elapsed,
            "simulator": identity,
            "training_revival_budget": -1,
            "writes_artifacts": False,
        },
    }


class SeedMapReplayCache:
    """Bounded, stat-keyed cache for expensive deterministic episode replays."""

    def __init__(
        self,
        simulator_exe: Path,
        *,
        maximum_entries: int = 8,
        client_factory: Callable[[Path], _ReplayClient] = _default_client_factory,
    ) -> None:
        if maximum_entries <= 0:
            raise ValueError("maximum_entries must be positive")
        self.simulator_exe = simulator_exe.resolve(strict=False)
        self.maximum_entries = maximum_entries
        self.client_factory = client_factory
        self._lock = threading.RLock()
        self._entries: OrderedDict[tuple[Path, int, int, str], JsonDict] = OrderedDict()

    def load(
        self,
        journal_path: Path,
        *,
        episode_id: str,
        detail: Mapping[str, Any],
    ) -> JsonDict:
        try:
            metadata = journal_path.stat()
        except OSError as exc:
            raise MapReplayUnavailable("地图复现失败: journal 不可读") from exc
        key = (journal_path, metadata.st_size, metadata.st_mtime_ns, episode_id)
        # Hold the replay-cache lock through reconstruction.  The dashboard is
        # single-operator and ThreadingHTTPServer still serves unrelated API
        # requests; serializing this path prevents duplicate simulator runs.
        with self._lock:
            cached = self._entries.pop(key, None)
            if cached is not None:
                self._entries[key] = cached
                return {**cached, "cache_hit": True}
            payload = reconstruct_episode_maps(
                journal_path=journal_path,
                episode_id=episode_id,
                detail=detail,
                simulator_exe=self.simulator_exe,
                client_factory=self.client_factory,
            )
            self._entries[key] = payload
            while len(self._entries) > self.maximum_entries:
                self._entries.popitem(last=False)
            return {**payload, "cache_hit": False}


__all__ = [
    "MAP_REPLAY_SCHEMA",
    "MapReplayUnavailable",
    "SeedMapReplayCache",
    "reconstruct_episode_maps",
]
