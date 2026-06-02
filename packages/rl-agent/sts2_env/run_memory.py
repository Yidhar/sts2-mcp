"""Persistent run/build memory and objective context for long-horizon planning."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np

from content_registry import get_card_metadata

from .semantic_action import compact_semantic_signature, semantic_action_signature

RUN_MEMORY_DIM = 48
OBJECTIVE_CONTEXT_DIM = 16

_LOG1P_200 = math.log1p(200.0)
_LOG1P_500 = math.log1p(500.0)

_ROUTE_SNAPSHOT_MAX_SUMMARIES = 8
_ROUTE_SNAPSHOT_MAX_NODES = 128
_ROUTE_SUMMARY_KEYS = (
    "count_elite",
    "count_rest_site",
    "count_shop",
    "count_event",
    "count_question_mark",
    "count_treasure",
    "count_monster",
    "count_boss",
    "direct_child_count",
    "reachable_node_count",
    "max_depth",
    "forced_path_steps_before_branch",
    "next_elite_steps",
    "next_rest_steps",
    "next_shop_steps",
    "next_event_steps",
    "next_question_mark_steps",
    "next_treasure_steps",
    "next_boss_steps",
    "can_reach_rest_site_before_elite",
    "can_reach_elite_then_rest_site",
)
_ROUTE_NODE_KEYS = (
    "row",
    "y",
    "x",
    "col",
    "coord",
    "point_type",
    "pointType",
    "type",
    "room_type",
    "kind",
    "point_type_norm",
)
_ROUTE_CONTAINER_KEYS = ("map", "route", "current_map")


def _float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _log_norm(value: float, anchor: float) -> float:
    if value <= 0:
        return 0.0
    return min(math.log1p(value) / anchor, 1.0)


def _clip01(value: float) -> float:
    return min(max(value, 0.0), 1.0)


_PLAYER_HP_KEYS = ("hp", "current_hp", "currentHealth", "current_health")
_PLAYER_MAX_HP_KEYS = ("max_hp", "maxHealth", "max_health", "maximum_hp", "max_hp_raw")


def _player_number(player: dict[str, Any] | None, keys: tuple[str, ...], default: float = 0.0) -> float:
    if not isinstance(player, dict):
        return default
    for key in keys:
        if key in player and player.get(key) is not None:
            value = _float(player.get(key), default)
            return value if math.isfinite(value) else default
    creature = player.get("creature")
    if isinstance(creature, dict):
        for key in keys:
            if key in creature and creature.get(key) is not None:
                value = _float(creature.get(key), default)
                return value if math.isfinite(value) else default
    return default


def _player_hp_triplet(player: dict[str, Any] | None) -> tuple[float, float, float]:
    hp = _player_number(player, _PLAYER_HP_KEYS, 0.0)
    max_hp = _player_number(player, _PLAYER_MAX_HP_KEYS, 0.0)
    if hp < 0.0:
        hp = 0.0
    if max_hp > 1.0:
        return hp, max_hp, _clip01(hp / max_hp)
    # Missing/suspicious max_hp=1 must not be encoded as "full HP".
    return hp, max_hp, 0.0


def _compact_scalar(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return None


def _compact_coord(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    out: dict[str, Any] = {}
    for key in ("row", "y", "x", "col"):
        scalar = _compact_scalar(value.get(key))
        if scalar is not None:
            out[key] = scalar
    return out or None


def _compact_route_node(node: Any) -> dict[str, Any] | None:
    if not isinstance(node, dict):
        return None
    out: dict[str, Any] = {}
    for key in _ROUTE_NODE_KEYS:
        value = node.get(key)
        if key == "coord":
            coord = _compact_coord(value)
            if coord:
                out[key] = coord
            continue
        scalar = _compact_scalar(value)
        if scalar is not None:
            out[key] = scalar

    point = node.get("point") if isinstance(node.get("point"), dict) else {}
    if isinstance(point, dict):
        for key in ("row", "y", "x", "col", "point_type", "pointType", "type", "room_type", "kind"):
            if key in out:
                continue
            scalar = _compact_scalar(point.get(key))
            if scalar is not None:
                out[key] = scalar
    return out or None


def _compact_route_nodes(nodes: Any, *, limit: int = _ROUTE_SNAPSHOT_MAX_NODES) -> list[dict[str, Any]]:
    if not isinstance(nodes, list):
        return []
    out: list[dict[str, Any]] = []
    for node in nodes:
        compact = _compact_route_node(node)
        if compact:
            out.append(compact)
            if len(out) >= limit:
                break
    return out


def _compact_route_summary(summary: Any) -> dict[str, Any] | None:
    if not isinstance(summary, dict):
        return None
    out: dict[str, Any] = {}
    for key in _ROUTE_SUMMARY_KEYS:
        if key not in summary:
            continue
        scalar = _compact_scalar(summary.get(key))
        if scalar is not None:
            out[key] = scalar
    for key in ("nodes", "route_nodes", "points"):
        nodes = _compact_route_nodes(summary.get(key), limit=32)
        if nodes:
            out[key] = nodes
    return out or None


def _append_unique_route_summary(target: list[dict[str, Any]], value: Any) -> None:
    if len(target) >= _ROUTE_SNAPSHOT_MAX_SUMMARIES:
        return
    summary = _compact_route_summary(value)
    if not summary:
        return
    if summary not in target:
        target.append(summary)


def _append_route_nodes(target: list[dict[str, Any]], value: Any) -> None:
    if len(target) >= _ROUTE_SNAPSHOT_MAX_NODES:
        return
    for node in _compact_route_nodes(value, limit=_ROUTE_SNAPSHOT_MAX_NODES - len(target)):
        if node not in target:
            target.append(node)
        if len(target) >= _ROUTE_SNAPSHOT_MAX_NODES:
            break


def _copy_route_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in ("current_floor", "act_id", "source"):
        value = _compact_scalar(snapshot.get(key))
        if value is not None:
            out[key] = value
    summaries = snapshot.get("route_summaries")
    if isinstance(summaries, list):
        out["route_summaries"] = [
            dict(summary)
            for summary in summaries[:_ROUTE_SNAPSHOT_MAX_SUMMARIES]
            if isinstance(summary, dict)
        ]
    nodes = snapshot.get("route_nodes")
    if isinstance(nodes, list):
        out["route_nodes"] = [
            dict(node)
            for node in nodes[:_ROUTE_SNAPSHOT_MAX_NODES]
            if isinstance(node, dict)
        ]
        out["map"] = {"nodes": [dict(node) for node in out["route_nodes"]]}
    return out


def _extract_route_snapshot(
    obs: dict[str, Any] | None,
    legal_actions: list[dict[str, Any]] | None,
    *,
    floor: int,
    act: int,
) -> dict[str, Any]:
    summaries: list[dict[str, Any]] = []
    nodes: list[dict[str, Any]] = []
    sources: set[str] = set()

    def scan_container(container: Any, source: str) -> None:
        if not isinstance(container, dict):
            return
        before = len(summaries) + len(nodes)
        _append_unique_route_summary(summaries, container.get("route_summary"))
        _append_unique_route_summary(summaries, container.get("summary"))
        for key in ("route_summaries", "available_route_summaries", "summaries"):
            value = container.get(key)
            if isinstance(value, list):
                for item in value:
                    _append_unique_route_summary(summaries, item)
        for key in ("nodes", "all_nodes", "route_nodes", "points"):
            _append_route_nodes(nodes, container.get(key))
        for key in _ROUTE_CONTAINER_KEYS:
            nested = container.get(key)
            if isinstance(nested, dict):
                scan_container(nested, source)
        if len(summaries) + len(nodes) > before:
            sources.add(source)

    if isinstance(obs, dict):
        scan_container(obs, "obs")
        sim_raw = obs.get("_sim_raw") if isinstance(obs.get("_sim_raw"), dict) else {}
        scan_container(sim_raw, "sim_raw")
        run = obs.get("run") if isinstance(obs.get("run"), dict) else {}
        scan_container(run, "run")

    if isinstance(legal_actions, list):
        for action in legal_actions:
            if not isinstance(action, dict):
                continue
            before = len(summaries) + len(nodes)
            _append_unique_route_summary(summaries, action.get("route_summary"))
            _append_unique_route_summary(summaries, action.get("summary"))
            for key in ("nodes", "all_nodes", "route_nodes", "points"):
                _append_route_nodes(nodes, action.get(key))
            # Some bridge variants only expose the immediate map point on the
            # action itself.  Keep it as low-confidence map context; downstream
            # runway logic ignores immediate-only one-row maps for hard caps.
            if str(action.get("kind") or "").strip() == "map":
                _append_route_nodes(nodes, [action])
            if len(summaries) + len(nodes) > before:
                sources.add("legal_actions")

    if not summaries and not nodes:
        return {}

    snapshot: dict[str, Any] = {
        "current_floor": floor,
        "act_id": act,
        "source": "+".join(sorted(sources)) if sources else "unknown",
    }
    if summaries:
        snapshot["route_summaries"] = summaries[:_ROUTE_SNAPSHOT_MAX_SUMMARIES]
    if nodes:
        snapshot["route_nodes"] = nodes[:_ROUTE_SNAPSHOT_MAX_NODES]
        snapshot["map"] = {"nodes": nodes[:_ROUTE_SNAPSHOT_MAX_NODES]}
    return snapshot


def _preview_metric(source: dict[str, Any] | None, key: str, default: float = 0.0) -> float:
    if not isinstance(source, dict):
        return default
    effect_preview = source.get("effect_preview") if isinstance(source.get("effect_preview"), dict) else {}
    if key in effect_preview:
        return _float(effect_preview.get(key), default)
    if key == "damage" and "total_damage" in effect_preview:
        return _float(effect_preview.get("total_damage"), default)
    if key == "block" and "total_block" in effect_preview:
        return _float(effect_preview.get("total_block"), default)
    return _float(source.get(key), default)


def _room_type(obs: dict[str, Any] | None) -> str:
    run = (obs or {}).get("run") if isinstance(obs, dict) else {}
    if isinstance(run, dict):
        return str(run.get("room_type") or "").strip()
    return ""


def _deck_cards(obs: dict[str, Any] | None) -> list[dict[str, Any]]:
    player = (obs or {}).get("player") if isinstance(obs, dict) else {}
    cards = player.get("deck_cards") if isinstance(player, dict) else None
    return cards if isinstance(cards, list) else []


def _potions(obs: dict[str, Any] | None) -> list[Any]:
    player = (obs or {}).get("player") if isinstance(obs, dict) else {}
    potions = player.get("potions") if isinstance(player, dict) else None
    return potions if isinstance(potions, list) else []


def _count_nonempty_potions(obs: dict[str, Any] | None) -> int:
    count = 0
    for potion in _potions(obs):
        if isinstance(potion, str):
            if potion and potion != "[empty]":
                count += 1
        elif isinstance(potion, dict):
            if str(potion.get("title") or "").strip() and str(potion.get("title")) != "[empty]":
                count += 1
    return count


def _visible_route_biases(legal_actions: list[dict[str, Any]] | None) -> dict[str, float]:
    counts = {"elite": 0.0, "shop": 0.0, "rest": 0.0, "question": 0.0}
    if not isinstance(legal_actions, list):
        return counts
    for action in legal_actions:
        if not isinstance(action, dict) or str(action.get("kind") or "").strip() != "map":
            continue
        point_type = str(action.get("point_type_norm") or action.get("point_type") or "").strip().lower()
        if "elite" in point_type:
            counts["elite"] += 1.0
        elif "shop" in point_type:
            counts["shop"] += 1.0
        elif "rest" in point_type:
            counts["rest"] += 1.0
        elif "question" in point_type or "event" in point_type:
            counts["question"] += 1.0
    total = max(sum(counts.values()), 1.0)
    return {key: min(value / total, 1.0) for key, value in counts.items()}


def _card_tags(card: dict[str, Any]) -> set[str]:
    tags: set[str] = set()
    metadata = get_card_metadata(str(card.get("id") or "").strip())
    raw_tags = metadata.get("semantic_tags") if isinstance(metadata, dict) else None
    if isinstance(raw_tags, list):
        for tag in raw_tags:
            text = str(tag or "").strip().lower()
            if text:
                tags.add(text)
    return tags


def _build_profile(obs: dict[str, Any] | None) -> dict[str, float]:
    cards = _deck_cards(obs)
    if not cards:
        return {
            "deck_size": 0.0,
            "frontload": 0.0,
            "block": 0.0,
            "draw": 0.0,
            "scaling": 0.0,
            "aoe": 0.0,
            "heal": 0.0,
            "curse_density": 0.0,
            "high_cost_density": 0.0,
            "zero_cost_density": 0.0,
            "x_cost_density": 0.0,
            "consistency": 0.0,
            "build_gap_risk": 1.0,
        }

    deck_size = float(len(cards))
    frontload = 0.0
    block = 0.0
    draw = 0.0
    scaling = 0.0
    aoe = 0.0
    heal = 0.0
    curses = 0.0
    high_cost = 0.0
    zero_cost = 0.0
    x_cost = 0.0
    for card in cards:
        if not isinstance(card, dict):
            continue
        tags = _card_tags(card)
        damage = _preview_metric(card, "damage")
        block_value = _preview_metric(card, "block")
        draw_value = _preview_metric(card, "draw")
        heal_value = _preview_metric(card, "heal")
        hits = _preview_metric(card, "hits")
        cost = _float(card.get("cost"))
        card_type = str(card.get("type") or "").strip().lower()
        target = str(card.get("target_type") or card.get("target") or "").strip().lower()
        if damage > 0:
            frontload += 1.0
        if block_value > 0:
            block += 1.0
        if draw_value > 0:
            draw += 1.0
        if heal_value > 0:
            heal += 1.0
        if hits > 1 or "all" in target or "aoe" in tags:
            aoe += 1.0
        if card_type == "power" or "scale" in tags or _preview_metric(card, "strength") > 0 or _preview_metric(card, "dexterity") > 0:
            scaling += 1.0
        if card_type in {"curse", "status"}:
            curses += 1.0
        if card.get("x_cost"):
            x_cost += 1.0
        if cost >= 2:
            high_cost += 1.0
        if cost == 0:
            zero_cost += 1.0

    frontload_density = frontload / deck_size
    block_density = block / deck_size
    draw_density = draw / deck_size
    scaling_density = scaling / deck_size
    aoe_density = aoe / deck_size
    heal_density = heal / deck_size
    curse_density = curses / deck_size
    high_cost_density = high_cost / deck_size
    zero_cost_density = zero_cost / deck_size
    x_cost_density = x_cost / deck_size

    consistency = _clip01(draw_density + zero_cost_density * 0.6 - high_cost_density * 0.4 - curse_density)
    build_gap_risk = _clip01(
        1.0
        - (frontload_density * 0.25 + block_density * 0.25 + draw_density * 0.20 + scaling_density * 0.15 + aoe_density * 0.15)
    )
    return {
        "deck_size": deck_size,
        "frontload": frontload_density,
        "block": block_density,
        "draw": draw_density,
        "scaling": scaling_density,
        "aoe": aoe_density,
        "heal": heal_density,
        "curse_density": curse_density,
        "high_cost_density": high_cost_density,
        "zero_cost_density": zero_cost_density,
        "x_cost_density": x_cost_density,
        "consistency": consistency,
        "build_gap_risk": build_gap_risk,
    }


@dataclass
class RunMemoryState:
    episode_mode: str = "full_run"
    potion_mechanics_available: bool = True
    combats_seen: int = 0
    elites_seen: int = 0
    bosses_seen: int = 0
    rests_seen: int = 0
    shops_seen: int = 0
    event_rooms_seen: int = 0
    card_reward_picks: int = 0
    potion_uses: int = 0
    smith_count: int = 0
    rest_count: int = 0
    map_choices: int = 0
    gold_spent: float = 0.0
    cumulative_hp_loss: float = 0.0
    recent_hp_loss: float = 0.0
    lowest_hp_ratio_seen: float = 1.0
    max_hp_seen: float = 1.0
    last_floor: int = 0
    last_act: int = 0
    last_room_floor: int = -1
    floor_room_types: set[tuple[int, int, str]] = field(default_factory=set)
    last_semantic_action: dict[str, Any] | None = None
    route_snapshot: dict[str, Any] = field(default_factory=dict)


class RunMemoryTracker:
    """Heuristic long-horizon run memory used by the new planning stack."""

    def __init__(
        self,
        *,
        episode_mode: str = "full_run",
        potion_mechanics_available: bool = True,
    ) -> None:
        self._default_potion_mechanics_available = bool(potion_mechanics_available)
        self.state = RunMemoryState(
            episode_mode=episode_mode,
            potion_mechanics_available=self._default_potion_mechanics_available,
        )

    def reset(
        self,
        obs: dict[str, Any] | None,
        legal_actions: list[dict[str, Any]] | None = None,
        *,
        episode_mode: str = "full_run",
        potion_mechanics_available: bool | None = None,
    ) -> None:
        if potion_mechanics_available is None:
            potion_mechanics_available = self._default_potion_mechanics_available
        self.state = RunMemoryState(
            episode_mode=episode_mode,
            potion_mechanics_available=bool(potion_mechanics_available),
        )
        self._absorb_observation(obs, legal_actions)

    def update_transition(
        self,
        prev_obs: dict[str, Any] | None,
        action: dict[str, Any] | None,
        next_obs: dict[str, Any] | None,
        *,
        legal_actions: list[dict[str, Any]] | None = None,
    ) -> None:
        prev_player = (prev_obs or {}).get("player") if isinstance(prev_obs, dict) else {}
        next_player = (next_obs or {}).get("player") if isinstance(next_obs, dict) else {}
        prev_hp = _float(prev_player.get("hp")) if isinstance(prev_player, dict) else 0.0
        next_hp = _float(next_player.get("hp")) if isinstance(next_player, dict) else 0.0
        hp_loss = max(prev_hp - next_hp, 0.0)
        self.state.cumulative_hp_loss += hp_loss
        self.state.recent_hp_loss = hp_loss

        if isinstance(action, dict):
            semantic = semantic_action_signature(action)
            self.state.last_semantic_action = compact_semantic_signature(semantic)
            family = semantic.get("family")
            if family == "use_potion":
                self.state.potion_uses += 1
            elif family == "shop":
                self.state.gold_spent += max(_float(action.get("price")), _float(((action.get("item") or {}) if isinstance(action.get("item"), dict) else {}).get("cost")))
            elif family == "smith":
                self.state.smith_count += 1
            elif family == "rest":
                self.state.rest_count += 1
            elif family == "map":
                self.state.map_choices += 1
            elif family == "card_reward" and str(action.get("selection") or "").strip().lower() == "pick":
                self.state.card_reward_picks += 1

        self._absorb_observation(next_obs, legal_actions)

    def _absorb_observation(self, obs: dict[str, Any] | None, legal_actions: list[dict[str, Any]] | None = None) -> None:
        if not isinstance(obs, dict):
            return
        run = obs.get("run") if isinstance(obs.get("run"), dict) else {}
        player = obs.get("player") if isinstance(obs.get("player"), dict) else {}
        floor = int(_float(run.get("floor")))
        act = int(_float(run.get("act_id")))
        room_type = _room_type(obs)

        hp, max_hp, hp_ratio = _player_hp_triplet(player)
        self.state.lowest_hp_ratio_seen = min(self.state.lowest_hp_ratio_seen, hp_ratio)
        if max_hp > 1.0:
            self.state.max_hp_seen = max(self.state.max_hp_seen, max_hp)
        self.state.last_floor = max(self.state.last_floor, floor)
        self.state.last_act = max(self.state.last_act, act)

        room_key = (act, floor, room_type)
        if floor > 0 and room_type and room_key not in self.state.floor_room_types:
            self.state.floor_room_types.add(room_key)
            if room_type in {"Monster"}:
                self.state.combats_seen += 1
            elif room_type in {"Elite"}:
                self.state.elites_seen += 1
            elif room_type in {"Boss"}:
                self.state.bosses_seen += 1
            elif room_type in {"Rest"}:
                self.state.rests_seen += 1
            elif room_type in {"Merchant"}:
                self.state.shops_seen += 1
            elif room_type in {"Event"}:
                self.state.event_rooms_seen += 1

        route_snapshot = _extract_route_snapshot(obs, legal_actions, floor=floor, act=act)
        if route_snapshot:
            # Do not overwrite a useful map snapshot with an empty reward/combat
            # DTO.  Card reward valuation needs the last known future route
            # runway because reward surfaces usually do not carry map actions.
            self.state.route_snapshot = route_snapshot

    def route_snapshot_for_obs(self) -> dict[str, Any]:
        if not isinstance(self.state.route_snapshot, dict) or not self.state.route_snapshot:
            return {}
        return _copy_route_snapshot(self.state.route_snapshot)

    def attach_route_snapshot_to_obs(self, obs: dict[str, Any] | None) -> None:
        if not isinstance(obs, dict):
            return
        snapshot = self.route_snapshot_for_obs()
        if snapshot:
            obs["_run_route_snapshot"] = snapshot

    def build_context(
        self,
        obs: dict[str, Any] | None,
        legal_actions: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        run_memory_vector, summary = self._encode_run_memory(obs, legal_actions)
        objective_context, objective_text = self._encode_objective_context(obs, legal_actions, run_memory_vector)
        semantic_actions = [
            compact_semantic_signature(semantic_action_signature(action))
            for action in (legal_actions or [])
            if isinstance(action, dict)
        ]
        return {
            "run_memory_vector": run_memory_vector,
            "objective_context_vector": objective_context,
            "run_memory_text": summary,
            "objective_text": objective_text,
            "semantic_actions": semantic_actions,
            "last_semantic_action": self.state.last_semantic_action or {},
            "episode_mode": self.state.episode_mode,
            "potion_mechanics_available": 1.0 if self.state.potion_mechanics_available else 0.0,
        }

    def _encode_run_memory(
        self,
        obs: dict[str, Any] | None,
        legal_actions: list[dict[str, Any]] | None,
    ) -> tuple[np.ndarray, str]:
        vector = np.zeros(RUN_MEMORY_DIM, dtype=np.float32)
        if not isinstance(obs, dict):
            return vector, ""

        run = obs.get("run") if isinstance(obs.get("run"), dict) else {}
        player = obs.get("player") if isinstance(obs.get("player"), dict) else {}
        combat = obs.get("combat") if isinstance(obs.get("combat"), dict) else {}
        hp, max_hp, hp_ratio = _player_hp_triplet(player)
        gold = _float(player.get("gold"))
        potion_count = _count_nonempty_potions(obs)
        potion_mechanics_available = 1.0 if self.state.potion_mechanics_available else 0.0
        build = _build_profile(obs)
        route_bias = _visible_route_biases(legal_actions)
        room_type = _room_type(obs)

        elite_pressure = _clip01((0.55 - hp_ratio) * 2.5 + route_bias["elite"] * 0.8)
        boss_pressure = _clip01((0.45 - hp_ratio) * 2.2 + (1.0 if room_type == "Boss" else 0.0) * 0.5)
        rest_pressure = _clip01((0.60 - hp_ratio) * 2.0 + route_bias["rest"] * 0.5)
        preserve_hp_bias = _clip01((0.70 - hp_ratio) * 1.6 + elite_pressure * 0.5 + boss_pressure * 0.5)
        greed_bias = _clip01((hp_ratio - 0.70) * 1.5 + build["consistency"] * 0.3 - preserve_hp_bias * 0.4)
        resource_pressure = _clip01(
            route_bias["shop"] * 0.5
            + (1.0 if gold >= 150 else 0.0) * 0.2
            + potion_mechanics_available * ((1.0 if potion_count <= 1 else 0.0) * 0.2)
        )

        vector[0] = _clip01(hp_ratio)
        vector[1] = _clip01(self.state.lowest_hp_ratio_seen)
        vector[2] = _log_norm(self.state.cumulative_hp_loss, _LOG1P_200)
        vector[3] = min(self.state.combats_seen / 30.0, 1.0)
        vector[4] = min(self.state.elites_seen / 12.0, 1.0)
        vector[5] = min(self.state.bosses_seen / 5.0, 1.0)
        vector[6] = min(self.state.rests_seen / 15.0, 1.0)
        vector[7] = min(self.state.shops_seen / 15.0, 1.0)
        vector[8] = min(self.state.smith_count / 15.0, 1.0)
        vector[9] = min(self.state.card_reward_picks / 30.0, 1.0)
        vector[10] = min(self.state.potion_uses / 10.0, 1.0)
        vector[11] = _log_norm(self.state.gold_spent, _LOG1P_500)
        vector[12] = min(_float(run.get("floor")) / 48.0, 1.0)
        vector[13] = min(_float(run.get("act_id")) / 4.0, 1.0)
        vector[14] = min(build["deck_size"] / 50.0, 1.0)
        vector[15] = min(potion_count / 5.0, 1.0)
        vector[16] = _log_norm(gold, _LOG1P_500)
        vector[17] = route_bias["elite"]
        vector[18] = route_bias["shop"]
        vector[19] = route_bias["rest"]
        vector[20] = route_bias["question"]
        vector[21] = build["frontload"]
        vector[22] = build["block"]
        vector[23] = build["draw"]
        vector[24] = build["scaling"]
        vector[25] = build["aoe"]
        vector[26] = build["heal"]
        vector[27] = build["curse_density"]
        vector[28] = build["high_cost_density"]
        vector[29] = build["zero_cost_density"]
        vector[30] = build["x_cost_density"]
        vector[31] = _clip01(
            0.5 * elite_pressure
            + 0.3 * boss_pressure
            + potion_mechanics_available * ((1.0 if potion_count <= 1 else 0.0) * 0.2)
        )
        vector[32] = rest_pressure
        vector[33] = elite_pressure
        vector[34] = boss_pressure
        vector[35] = preserve_hp_bias
        vector[36] = greed_bias
        vector[37] = resource_pressure
        vector[38] = 1.0 if room_type == "Elite" else 0.0
        vector[39] = 1.0 if room_type == "Boss" else 0.0
        vector[40] = min(_float(combat.get("round")) / 20.0, 1.0) if isinstance(combat, dict) else 0.0
        vector[41] = _clip01(self.state.recent_hp_loss / 30.0)
        vector[42] = 1.0 if self.state.episode_mode == "combat_sandbox" else 0.0
        vector[43] = _clip01(len(combat.get("enemies") or []) / 4.0) if isinstance(combat, dict) else 0.0
        vector[44] = _clip01(sum(1.0 for relic in (player.get("relics") or []) if isinstance(relic, dict) and "energy" in str(relic.get("summary") or "").lower()) / 3.0)
        # Use this slot for a direct "potion system exists in this env" flag.
        # Combat sandbox currently runs without potion mechanics, while full-run
        # does. Keeping this explicit avoids conflating "zero potions carried"
        # with "potions are not part of the environment at all".
        vector[45] = potion_mechanics_available
        vector[46] = build["consistency"]
        vector[47] = build["build_gap_risk"]

        summary = (
            f"hp={int(hp)}/{int(max_hp)} hp_mode={preserve_hp_bias:.2f} "
            f"elite={elite_pressure:.2f} boss={boss_pressure:.2f} "
            f"deck(front={build['frontload']:.2f},blk={build['block']:.2f},draw={build['draw']:.2f},scale={build['scaling']:.2f}) "
            f"potions={'on' if self.state.potion_mechanics_available else 'off'}"
        )
        return vector, summary

    def _encode_objective_context(
        self,
        obs: dict[str, Any] | None,
        legal_actions: list[dict[str, Any]] | None,
        run_memory_vector: np.ndarray,
    ) -> tuple[np.ndarray, str]:
        vector = np.zeros(OBJECTIVE_CONTEXT_DIM, dtype=np.float32)
        if not isinstance(obs, dict):
            return vector, ""

        room_type = _room_type(obs)
        hp_ratio = float(run_memory_vector[0])
        elite_pressure = float(run_memory_vector[33])
        boss_pressure = float(run_memory_vector[34])
        preserve_hp_bias = float(run_memory_vector[35])
        greed_bias = float(run_memory_vector[36])
        resource_pressure = float(run_memory_vector[37])
        route_shop = float(run_memory_vector[18])
        route_rest = float(run_memory_vector[19])
        potion_conservation = float(run_memory_vector[31])
        potion_mechanics_available = float(run_memory_vector[45])
        in_combat = bool((obs.get("combat") or {}))
        zero_damage_desire = _clip01(preserve_hp_bias * (1.0 if in_combat else 0.5) + elite_pressure * 0.2 + boss_pressure * 0.2)

        survival_priority = 1.0
        hp_loss_priority = _clip01(0.7 + preserve_hp_bias * 0.3)
        build_priority = _clip01(0.35 + greed_bias * 0.45 + (1.0 if hp_ratio > 0.70 else 0.0) * 0.10)
        resource_priority = _clip01(0.40 + resource_pressure * 0.40 + potion_conservation * 0.20)
        save_potion_mode = potion_mechanics_available * _clip01(
            max(elite_pressure, boss_pressure) * 0.7
            + (1.0 if room_type not in {"Elite", "Boss"} else 0.0) * 0.2
        )
        force_rest_mode = _clip01(run_memory_vector[32] * 0.8 + (1.0 if hp_ratio < 0.45 else 0.0) * 0.2)
        greed_upgrade_mode = _clip01(greed_bias * 0.8 + (1.0 if hp_ratio > 0.75 else 0.0) * 0.2)
        safe_route_bias = _clip01(preserve_hp_bias * 0.6 + boss_pressure * 0.2 + elite_pressure * 0.2)
        shop_value_bias = _clip01(route_shop * 0.6 + resource_priority * 0.4)
        rest_value_bias = _clip01(route_rest * 0.5 + force_rest_mode * 0.5)
        smith_value_bias = _clip01(greed_upgrade_mode * 0.6 + (1.0 - force_rest_mode) * 0.4)
        long_horizon_mode = 1.0 if self.state.episode_mode == "full_run" else 0.0

        vector[0] = survival_priority
        vector[1] = hp_loss_priority
        vector[2] = build_priority
        vector[3] = resource_priority
        vector[4] = preserve_hp_bias
        vector[5] = save_potion_mode
        vector[6] = force_rest_mode
        vector[7] = greed_upgrade_mode
        vector[8] = elite_pressure
        vector[9] = boss_pressure
        vector[10] = safe_route_bias
        vector[11] = shop_value_bias
        vector[12] = rest_value_bias
        vector[13] = smith_value_bias
        vector[14] = zero_damage_desire
        vector[15] = long_horizon_mode

        text = (
            f"objective survival=1.00 hp={hp_loss_priority:.2f} build={build_priority:.2f} "
            f"resource={resource_priority:.2f} preserve_hp={preserve_hp_bias:.2f} "
            f"save_potion={save_potion_mode:.2f} rest={force_rest_mode:.2f}"
        )
        return vector, text
