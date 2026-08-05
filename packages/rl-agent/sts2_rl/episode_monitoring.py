"""Bounded, read-only held-out episode projections for the training monitor.

The live dashboard must not turn its periodic aggregate refresh into a scan of
large trajectory journals.  This module therefore parses a journal only after
an explicit drill-down request, projects it into a compact episode/floor
schema, and caches that immutable projection by file identity.

No checkpoint payload is deserialized and no artifact is written.  Compact
``decision`` rows remain the authoritative decision index.  Rich
``decision_snapshot`` rows are streamed only during an explicit drill-down and
are reduced immediately to a bounded loadout projection; they never count as
decisions and their full state is never retained.  Maps are reconstructed on
demand from the seed, pinned simulator identity, and recorded action sequence,
so neither new journals nor older version-4 journals need a topology payload.
"""

from __future__ import annotations

import json
import math
import stat
import threading
from collections import OrderedDict, deque
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, cast

from sts2_rl.entity_localization import EntityKind, localized_entity_name

JsonDict = dict[str, Any]

EPISODE_SUMMARY_SCHEMA: Final = "sts2-heldout-episode-summary-v1"
_MAX_LINE_BYTES = 8 * 1024 * 1024
_MAX_EPISODES = 256
_MAX_FLOORS = 128
_MAX_ROUTE_CHOICES = 128
_MAX_CARD_REWARDS = 128
_MAX_MACRO_DECISIONS_PER_FLOOR = 256
_MAX_ANOMALIES = 64
_MAX_POLICY_CANDIDATES = 8
_MAX_LOADOUT_CARD_ROWS = 512
_MAX_RELICS = 128
_MAX_POTIONS = 32
_MAX_CARD_MODIFIERS = 16
_MAX_SNAPSHOT_REASONS = 8
_MAX_PROJECTED_STRING_LENGTH = 512
_MAX_ITEM_QUANTITY = 1_000_000
_MACRO_ACTIONS = frozenset(
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
        "combat_select_card",
        "combat_confirm_selection",
        "combat_cancel_selection",
    }
)
_COMBAT_ROOM_TYPES = frozenset({"monster", "elite", "boss"})
_GATE_KIND_BY_JOURNAL = {
    "evaluation": "validation",
    "early-validation": "early_validation",
    "final-audit": "final_audit",
}


def _mapping(value: object) -> JsonDict:
    return value if isinstance(value, dict) else {}


def _string(value: object, default: str = "") -> str:
    return value if isinstance(value, str) else default


def _finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _integer(value: object, default: int = 0) -> int:
    number = _finite(value)
    return int(number) if number is not None else default


def _bounded_string(value: object) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    return value[:_MAX_PROJECTED_STRING_LENGTH]


def _project_coord(value: object) -> JsonDict | None:
    """Project an observed coordinate without retaining arbitrary fields."""

    if isinstance(value, Mapping):
        projected: JsonDict = {}
        for key in ("x", "y", "row", "col"):
            number = _finite(value.get(key))
            if number is not None:
                projected[key] = int(number)
        return projected or None
    if isinstance(value, list | tuple) and len(value) >= 2:
        row = _finite(value[0])
        column = _finite(value[1])
        if row is not None and column is not None:
            # Sequence-shaped coordinates in legacy journals are row/column.
            return {"row": int(row), "col": int(column)}
    return None


def _project_modifier(value: object, *, fallback_type: str) -> JsonDict | None:
    modifier = _mapping(value)
    identifier = (
        _bounded_string(modifier.get("id"))
        or _bounded_string(modifier.get(f"{fallback_type}_id"))
        or _bounded_string(modifier.get("class_name"))
    )
    if identifier is None:
        return None
    projected: JsonDict = {
        "id": identifier,
        "class_name": _bounded_string(modifier.get("class_name")),
        "type": _bounded_string(modifier.get("type"))
        or _bounded_string(modifier.get("modifier_type"))
        or fallback_type,
    }
    for key in ("amount", "display_amount"):
        number = _finite(modifier.get(key))
        projected[key] = number if number is not None else None
    projected["status"] = _bounded_string(modifier.get("status"))
    for key in (
        "is_stackable",
        "show_amount",
        "has_extra_card_text",
        "should_glow_gold",
        "should_glow_red",
        "should_start_at_bottom_of_draw_pile",
        "can_afflict_unplayable_cards",
        "has_overlay",
    ):
        value_at_key = modifier.get(key)
        if isinstance(value_at_key, bool):
            projected[key] = value_at_key
    return projected


def _project_modifiers(value: object, *, fallback_type: str) -> tuple[list[JsonDict], int]:
    if not isinstance(value, list):
        return [], 0
    projected: list[JsonDict] = []
    for item in value[:_MAX_CARD_MODIFIERS]:
        modifier = _project_modifier(item, fallback_type=fallback_type)
        if modifier is not None:
            projected.append(modifier)
    projected.sort(key=lambda item: json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return projected, max(0, len(value) - _MAX_CARD_MODIFIERS)


def _project_card(value: object) -> tuple[JsonDict | None, int]:
    card = _mapping(value)
    identifier = _bounded_string(card.get("id")) or _bounded_string(card.get("card_id"))
    if identifier is None:
        return None, 0
    enchantments, enchantments_omitted = _project_modifiers(
        card.get("enchantments"),
        fallback_type="enchantment",
    )
    afflictions, afflictions_omitted = _project_modifiers(
        card.get("afflictions"),
        fallback_type="affliction",
    )
    quantity = _integer(card.get("quantity"), 1)
    quantity = min(_MAX_ITEM_QUANTITY, max(1, quantity))
    projected: JsonDict = {
        "card_id": identifier,
        "display_name": localized_entity_name(identifier, kind="card"),
        "name": _bounded_string(card.get("name")) or _bounded_string(card.get("title")),
        "type": _bounded_string(card.get("type")),
        "rarity": _bounded_string(card.get("rarity")),
        "cost": _finite(card.get("cost")),
        "star_cost": _finite(card.get("star_cost")),
        "is_upgraded": card.get("is_upgraded") if isinstance(card.get("is_upgraded"), bool) else None,
        "upgrade_level": (
            _integer(card.get("upgrade_level", card.get("upgrade_count")))
            if _finite(card.get("upgrade_level", card.get("upgrade_count"))) is not None
            else None
        ),
        "enchantments": enchantments,
        "afflictions": afflictions,
        "quantity": quantity,
    }
    return projected, enchantments_omitted + afflictions_omitted


def _aggregate_deck(value: object) -> tuple[list[JsonDict], JsonDict]:
    if not isinstance(value, list):
        return [], {"rows_seen": 0, "rows_omitted": 0, "modifier_rows_omitted": 0}
    aggregated: OrderedDict[str, JsonDict] = OrderedDict()
    modifier_rows_omitted = 0
    for item in value[:_MAX_LOADOUT_CARD_ROWS]:
        projected, modifier_omitted = _project_card(item)
        modifier_rows_omitted += modifier_omitted
        if projected is None:
            continue
        quantity = _integer(projected.pop("quantity"), 1)
        identity = json.dumps(projected, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        existing = aggregated.get(identity)
        if existing is None:
            projected["quantity"] = quantity
            aggregated[identity] = projected
        else:
            existing["quantity"] = min(
                _MAX_ITEM_QUANTITY,
                _integer(existing.get("quantity"), 1) + quantity,
            )
    return list(aggregated.values()), {
        "rows_seen": len(value),
        "rows_omitted": max(0, len(value) - _MAX_LOADOUT_CARD_ROWS),
        "modifier_rows_omitted": modifier_rows_omitted,
    }


def _project_inventory(value: object, *, kind: str, limit: int) -> tuple[list[JsonDict], int]:
    if not isinstance(value, list):
        return [], 0
    projected: list[JsonDict] = []
    identifier_keys = ("id", f"{kind}_id")
    scalar_keys = (
        "name",
        "title",
        "rarity",
        "status",
        "usage",
        "target_type",
    )
    numeric_keys = ("quantity", "stack_count", "display_amount", "charges", "index")
    boolean_keys = ("is_used_up", "is_melted", "is_queued")
    for item in value[:limit]:
        source = _mapping(item)
        identifier = next((_bounded_string(source.get(key)) for key in identifier_keys if source.get(key)), None)
        if identifier is None:
            continue
        entity_kind = cast(EntityKind | None, kind if kind in {"card", "relic", "potion"} else None)
        row: JsonDict = {
            f"{kind}_id": identifier,
            "display_name": localized_entity_name(identifier, kind=entity_kind),
        }
        for key in scalar_keys:
            row[key] = _bounded_string(source.get(key))
        for key in numeric_keys:
            number = _finite(source.get(key))
            row[key] = number if number is not None else None
        for key in boolean_keys:
            item_value = source.get(key)
            row[key] = item_value if isinstance(item_value, bool) else None
        projected.append(row)
    return projected, max(0, len(value) - limit)


def _project_loadout(
    observation: Mapping[str, object],
    *,
    source_field: str,
    step: int,
    reasons: list[str],
) -> JsonDict | None:
    player = _mapping(observation.get("player"))
    if not player:
        return None
    deck, deck_coverage = _aggregate_deck(player.get("deck"))
    relics, relics_omitted = _project_inventory(player.get("relics"), kind="relic", limit=_MAX_RELICS)
    potions, potions_omitted = _project_inventory(player.get("potions"), kind="potion", limit=_MAX_POTIONS)
    omitted = deck_coverage["rows_omitted"] + deck_coverage["modifier_rows_omitted"]
    omitted += relics_omitted + potions_omitted
    fields_present = {
        "hp": _finite(player.get("hp")) is not None,
        "max_hp": _finite(player.get("max_hp")) is not None,
        "gold": _finite(player.get("gold")) is not None,
        "deck": isinstance(player.get("deck"), list),
        "relics": isinstance(player.get("relics"), list),
        "potions": isinstance(player.get("potions"), list),
    }
    return {
        "source": f"decision_snapshot.{source_field}.player",
        "coverage": {
            "kind": "latest_available_rich_snapshot",
            "bounded": True,
            "complete": omitted == 0 and all(fields_present.values()),
            "episode_last_snapshot": "episode_last" in reasons,
            "fields_present": fields_present,
            "deck": deck_coverage,
            "relic_rows_omitted": relics_omitted,
            "potion_rows_omitted": potions_omitted,
        },
        "snapshot_step": step,
        "snapshot_reasons": reasons,
        "hp": _integer(player.get("hp")) if _finite(player.get("hp")) is not None else None,
        "max_hp": _integer(player.get("max_hp")) if _finite(player.get("max_hp")) is not None else None,
        "gold": _integer(player.get("gold")) if _finite(player.get("gold")) is not None else None,
        "deck": deck,
        "relics": relics,
        "potions": potions,
    }


def _is_link_or_reparse(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except OSError:
        return True
    if stat.S_ISLNK(metadata.st_mode):
        return True
    attributes = getattr(metadata, "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(attributes & reparse_flag)


def _safe_regular_child(path: Path, parent: Path) -> bool:
    if not path.is_file() or _is_link_or_reparse(path):
        return False
    try:
        resolved_parent = parent.resolve(strict=False)
        resolved_path = path.resolve(strict=False)
    except OSError:
        return False
    return resolved_path != resolved_parent and resolved_path.is_relative_to(resolved_parent)


def read_journal_header(path: Path, *, parent: Path) -> JsonDict | None:
    """Read one bounded ``evaluation_started`` row without following links."""

    if not _safe_regular_child(path, parent):
        return None
    try:
        with path.open("rb") as handle:
            raw = handle.readline(_MAX_LINE_BYTES + 1)
    except OSError:
        return None
    if not raw.endswith(b"\n") or len(raw) > _MAX_LINE_BYTES:
        return None
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict) or value.get("event") != "evaluation_started":
        return None
    return value


def _action_kind(action: Mapping[str, object]) -> str:
    for key in ("action", "kind", "model_action_kind"):
        value = action.get(key)
        if isinstance(value, str) and value:
            return value
    return "unknown"


def _action_label(action: Mapping[str, object]) -> str:
    for key in ("label", "card_name", "card_id", "option_id"):
        value = action.get(key)
        if isinstance(value, str) and value:
            return value
    option = _mapping(action.get("option"))
    for key in ("text", "text_key", "description", "id"):
        value = option.get(key)
        if isinstance(value, str) and value:
            return value
    item = _mapping(action.get("item"))
    for key in ("card_name", "card_id", "relic_name", "relic_id", "potion_name", "potion_id", "name", "id"):
        value = item.get(key)
        if isinstance(value, str) and value:
            return value
    card = _mapping(action.get("card"))
    for key in ("name", "card_name", "id", "card_id"):
        value = card.get(key)
        if isinstance(value, str) and value:
            return value
    kind = _action_kind(action)
    index = action.get("index", action.get("idx", action.get("action_index")))
    return f"{kind} #{_integer(index)}" if _finite(index) is not None else kind


def _project_action(action: Mapping[str, object]) -> JsonDict:
    kind = _action_kind(action)
    item = _mapping(action.get("item"))
    card = _mapping(action.get("card"))
    map_node = _mapping(action.get("map_node"))
    map_node_coord = _project_coord(map_node.get("coord")) or _project_coord(map_node)
    coord = _project_coord(action.get("coord")) or map_node_coord
    direct_column = _finite(action.get("col"))
    direct_row = _finite(action.get("row"))
    coord_column = _finite(coord.get("col", coord.get("x"))) if coord is not None else None
    coord_row = _finite(coord.get("row", coord.get("y"))) if coord is not None else None
    column = direct_column if direct_column is not None else coord_column
    row = direct_row if direct_row is not None else coord_row
    projected_map_node: JsonDict | None = None
    if map_node:
        projected_map_node = {
            "coord": map_node_coord,
            "point_type": _bounded_string(map_node.get("point_type")),
            "index": _integer(map_node.get("index")) if _finite(map_node.get("index")) is not None else None,
        }
    item_category = _string(item.get("category"))
    entity_kind: EntityKind | None = None
    entity_id: str | None = None
    for candidate_kind, sources in (
        (cast(EntityKind, "card"), ((action, ("card_id",)), (card, ("card_id", "id")))),
        (cast(EntityKind, "relic"), ((action, ("relic_id",)), (item, ("relic_id",)))),
        (cast(EntityKind, "potion"), ((action, ("potion_id",)), (item, ("potion_id",)))),
    ):
        for source, keys in sources:
            for key in keys:
                value = _bounded_string(source.get(key))
                if value is not None:
                    entity_kind = candidate_kind
                    entity_id = value
                    break
            if entity_id is not None:
                break
        if entity_id is not None:
            break
    if entity_id is None and item_category in {"card", "relic", "potion"}:
        entity_kind = cast(EntityKind, item_category)
        for key in (f"{item_category}_id", "id"):
            entity_id = _bounded_string(item.get(key))
            if entity_id is not None:
                break
    display_name = (
        localized_entity_name(entity_id, kind=entity_kind) if entity_id is not None else _action_label(action)
    )
    projected: JsonDict = {
        "kind": kind,
        "label": _action_label(action),
        "index": (
            _integer(action.get("index", action.get("idx", action.get("action_index"))))
            if _finite(action.get("index", action.get("idx", action.get("action_index")))) is not None
            else None
        ),
        "column": int(column) if column is not None else None,
        "row": int(row) if row is not None else None,
        "coord": coord,
        "map_node": projected_map_node,
        "card_id": _string(action.get("card_id")) or _string(card.get("card_id")) or None,
        "card_rarity": _string(action.get("card_rarity")) or _string(card.get("rarity")) or None,
        "item_category": item_category or None,
        "entity_kind": entity_kind,
        "entity_id": entity_id,
        "display_name": display_name,
    }
    return projected


def _policy_candidates(event: Mapping[str, object]) -> list[JsonDict]:
    selected_index = event.get("selected_candidate_index")
    rows = event.get("policy_topk")
    if not isinstance(rows, list):
        return []
    candidates: list[JsonDict] = []
    for row in rows[:_MAX_POLICY_CANDIDATES]:
        if not isinstance(row, dict):
            continue
        action = _mapping(row.get("action"))
        candidate_index = row.get("candidate_index")
        projected = _project_action(action)
        projected.update(
            {
                "candidate_index": (_integer(candidate_index) if _finite(candidate_index) is not None else None),
                "probability": _finite(row.get("probability")),
                "selected": candidate_index == selected_index,
                "multiplicity": _integer(row.get("multiplicity"), 1),
            }
        )
        candidates.append(projected)
    return candidates


def _provenance(header: Mapping[str, object]) -> JsonDict:
    checkpoint_association = _mapping(header.get("checkpoint_association"))
    checkpoint = _mapping(checkpoint_association.get("last_committed_checkpoint"))
    checkpoint_path = _string(checkpoint.get("path"))
    simulator = _mapping(header.get("simulator"))
    simulator_identity = _mapping(simulator.get("simulator_identity"))
    source = _mapping(simulator_identity.get("source"))
    binary = _mapping(simulator_identity.get("binary"))
    managed_binary = _mapping(simulator_identity.get("managed_binary"))
    config = _mapping(header.get("config"))
    encoding = _mapping(header.get("encoding"))
    return {
        "evaluation_gate": header.get("evaluation_gate"),
        "gate_kind": header.get("gate_kind"),
        "evaluation_attempt": header.get("evaluation_attempt"),
        "evaluation_guard_rollbacks": header.get("evaluation_guard_rollbacks"),
        "actual_environment_steps": header.get("actual_environment_steps"),
        "policy_version": header.get("policy_version"),
        "actor_policy_version": header.get("actor_policy_version"),
        "policy_model_state_sha256": header.get("policy_model_state_sha256"),
        "deterministic": header.get("deterministic"),
        "epsilon": header.get("epsilon"),
        "data_partition": header.get("data_partition"),
        "journal_version": header.get("journal_version"),
        "game_version": header.get("game_version"),
        "config_version": config.get("version"),
        "config_fingerprint_sha256": config.get("fingerprint_sha256"),
        "encoding_version": encoding.get("version"),
        "encoding_fingerprint_sha256": encoding.get("fingerprint_sha256"),
        "checkpoint": {
            "name": Path(checkpoint_path).name if checkpoint_path else None,
            "path": checkpoint_path or None,
            "checkpoint_id": checkpoint.get("checkpoint_id"),
            "manifest_sha256": checkpoint.get("manifest_sha256"),
            "relation": checkpoint_association.get("relation"),
            "load_mode": checkpoint_association.get("load_mode"),
        },
        "simulator": {
            "backend": simulator.get("backend"),
            "schema_version": simulator_identity.get("schema_version"),
            "source_commit": source.get("commit"),
            "source_tree": source.get("tree"),
            "binary_sha256": binary.get("sha256"),
            "managed_binary_sha256": managed_binary.get("sha256"),
        },
    }


def journal_descriptor(
    path: Path,
    *,
    parent: Path,
    journal_kind: str,
    gate: int,
    complete: bool | None,
) -> JsonDict | None:
    """Return cheap journal metadata from the first row and filesystem stat."""

    header = read_journal_header(path, parent=parent)
    if header is None:
        return None
    expected_gate_kind = _GATE_KIND_BY_JOURNAL.get(journal_kind)
    recorded_gate_kind = _string(header.get("gate_kind"))
    recorded_gate = _finite(header.get("evaluation_gate"))
    if expected_gate_kind is None:
        return None
    if recorded_gate_kind and recorded_gate_kind != expected_gate_kind:
        return None
    if recorded_gate is not None and int(recorded_gate) != gate:
        return None
    try:
        metadata = path.stat()
    except OSError:
        return None
    provenance = _provenance(header)
    seeds = header.get("evaluation_seeds")
    return {
        "key": path.name,
        "journal_name": path.name,
        "journal_kind": journal_kind,
        "gate_kind": recorded_gate_kind or expected_gate_kind,
        "evaluation_gate": gate,
        "evaluation_attempt": header.get("evaluation_attempt"),
        "evaluation_guard_rollbacks": header.get("evaluation_guard_rollbacks"),
        "actual_environment_steps": header.get("actual_environment_steps"),
        "policy_version": header.get("policy_version"),
        "episode_count": len(seeds) if isinstance(seeds, list) else None,
        "modified_at": metadata.st_mtime,
        "size_bytes": metadata.st_size,
        "complete": complete,
        "checkpoint": provenance["checkpoint"],
        "game_version": provenance["game_version"],
        "simulator": provenance["simulator"],
    }


@dataclass(slots=True)
class _FloorBuilder:
    act: int
    floor: int
    entry_step: int
    exit_step: int
    room_type: str | None = None
    room_model_id: str | None = None
    entry_hp: int | None = None
    exit_hp: int | None = None
    max_hp: int | None = None
    entry_hp_lost: int = 0
    exit_hp_lost: int = 0
    entry_revivals: int = 0
    exit_revivals: int = 0
    entry_gold: int | None = None
    exit_gold: int | None = None
    combat_seen: bool = False
    max_combat_round: int = 0
    combat_decisions: int = 0
    left_combat: bool = False
    macro_decisions: deque[JsonDict] | None = None
    macro_omitted: int = 0

    def __post_init__(self) -> None:
        self.macro_decisions = deque(maxlen=_MAX_MACRO_DECISIONS_PER_FLOOR)

    def append_macro(self, decision: JsonDict) -> None:
        assert self.macro_decisions is not None
        if len(self.macro_decisions) == self.macro_decisions.maxlen:
            self.macro_omitted += 1
        self.macro_decisions.append(decision)

    def to_mapping(self, *, terminal_outcome: str, is_final_floor: bool) -> JsonDict:
        encounter_class = self.room_type if self.room_type in _COMBAT_ROOM_TYPES else None
        if not self.combat_seen:
            combat_result: str | None = None
        elif not is_final_floor or self.left_combat or terminal_outcome == "success":
            combat_result = "victory"
        elif is_final_floor and terminal_outcome not in {"", "ongoing", "success"}:
            combat_result = "failure"
        else:
            combat_result = "unknown"
        return {
            "act": self.act,
            "floor": self.floor,
            "entry_step": self.entry_step,
            "exit_step": self.exit_step,
            "room_type": self.room_type,
            "room_model_id": self.room_model_id,
            "entry_hp": self.entry_hp,
            "exit_hp": self.exit_hp,
            "max_hp": self.max_hp,
            "hp_net_change": (
                self.exit_hp - self.entry_hp if self.entry_hp is not None and self.exit_hp is not None else None
            ),
            "hp_loss_delta": max(0, self.exit_hp_lost - self.entry_hp_lost),
            "entry_revivals": self.entry_revivals,
            "exit_revivals": self.exit_revivals,
            "revivals_delta": max(0, self.exit_revivals - self.entry_revivals),
            "entry_gold": self.entry_gold,
            "exit_gold": self.exit_gold,
            "encounter_class": encounter_class,
            "combat_result": combat_result,
            "combat_rounds": self.max_combat_round or None,
            "combat_decisions": self.combat_decisions,
            "macro_decisions": list(self.macro_decisions or ()),
            "macro_decisions_omitted": self.macro_omitted,
        }


class _EpisodeBuilder:
    def __init__(self, episode_id: str) -> None:
        self.episode_id = episode_id
        self.seed: int | None = None
        self.attempt: int | None = None
        self.steps = 0
        self.last_step = -1
        self.character: str | None = None
        self.ascension: int | None = None
        self.max_act = 0
        self.max_floor = 0
        self.revivals = 0
        self.player_hp_lost = 0
        self.final_hp: int | None = None
        self.outcome = "ongoing"
        self.floors: OrderedDict[tuple[int, int], _FloorBuilder] = OrderedDict()
        self.route_choices: list[JsonDict] = []
        self.card_rewards: list[JsonDict] = []
        self.anomalies: list[JsonDict] = []
        self._recent_actions: deque[JsonDict] = deque(maxlen=32)
        self._pending_route: JsonDict | None = None
        self.rich_snapshot_count = 0
        self._snapshot_sequence = 0
        self._latest_loadout_rank: tuple[int, int, int] | None = None
        self._latest_loadout: JsonDict | None = None

    def _floor(self, *, event: Mapping[str, object], observation: Mapping[str, object]) -> _FloorBuilder:
        run = _mapping(observation.get("run"))
        player = _mapping(observation.get("player"))
        act = _integer(run.get("act"))
        floor_number = _integer(run.get("floor"))
        key = (act, floor_number)
        floor = self.floors.get(key)
        hp = _integer(player.get("hp")) if _finite(player.get("hp")) is not None else None
        max_hp = _integer(player.get("max_hp")) if _finite(player.get("max_hp")) is not None else None
        gold = _integer(player.get("gold")) if _finite(player.get("gold")) is not None else None
        hp_lost = _integer(event.get("player_hp_lost"))
        revivals = _integer(event.get("revivals_used"))
        step = _integer(event.get("step_index"))
        if floor is None:
            if len(self.floors) >= _MAX_FLOORS:
                # Preserve a bounded detail response even for corrupted input.
                return self.floors[next(reversed(self.floors))]
            floor = _FloorBuilder(
                act=act,
                floor=floor_number,
                entry_step=step,
                exit_step=step,
                entry_hp=hp,
                exit_hp=hp,
                max_hp=max_hp,
                entry_hp_lost=hp_lost,
                exit_hp_lost=hp_lost,
                entry_revivals=revivals,
                exit_revivals=revivals,
                entry_gold=gold,
                exit_gold=gold,
            )
            self.floors[key] = floor
        floor.exit_step = step
        floor.exit_hp = hp
        floor.max_hp = max_hp if max_hp is not None else floor.max_hp
        floor.exit_hp_lost = hp_lost
        floor.exit_revivals = revivals
        floor.exit_gold = gold
        room_type = _string(run.get("room_type"))
        room_model_id = _string(run.get("room_model_id"))
        if room_type and room_type != "map":
            if floor.room_type is None or (
                room_type in _COMBAT_ROOM_TYPES and floor.room_type not in _COMBAT_ROOM_TYPES
            ):
                floor.room_type = room_type
                floor.room_model_id = room_model_id or floor.room_model_id
            if self._pending_route is not None and step > _integer(self._pending_route.get("step"), -1):
                self._pending_route["destination"] = {
                    "act": act,
                    "floor": floor_number,
                    "room_type": room_type,
                    "room_model_id": room_model_id or None,
                }
                self._pending_route = None
        return floor

    def consume(self, event: Mapping[str, object]) -> None:
        observation = _mapping(event.get("observation_summary"))
        action = _mapping(event.get("selected_action"))
        player = _mapping(observation.get("player"))
        run = _mapping(observation.get("run"))
        combat = _mapping(observation.get("combat"))
        screen = _string(observation.get("screen"))
        phase = _string(observation.get("phase"))
        step = _integer(event.get("step_index"))
        self.steps += 1
        self.last_step = max(self.last_step, step)
        self.seed = _integer(event.get("reset_seed")) if _finite(event.get("reset_seed")) is not None else self.seed
        self.character = _string(player.get("character")) or _string(run.get("character_id")) or self.character
        self.ascension = (
            _integer(run.get("ascension_level", run.get("ascension")))
            if _finite(run.get("ascension_level", run.get("ascension"))) is not None
            else self.ascension
        )
        self.max_act = max(self.max_act, _integer(run.get("act")))
        self.max_floor = max(self.max_floor, _integer(run.get("floor")))
        self.revivals = max(self.revivals, _integer(event.get("revivals_used")))
        self.player_hp_lost = max(self.player_hp_lost, _integer(event.get("player_hp_lost")))
        if _finite(player.get("hp")) is not None:
            self.final_hp = _integer(player.get("hp"))
        outcome = _string(event.get("outcome"))
        if outcome and outcome != "ongoing":
            self.outcome = outcome

        floor = self._floor(event=event, observation=observation)
        in_combat = screen == "COMBAT" or combat.get("in_progress") is True
        if in_combat:
            floor.combat_seen = True
            floor.combat_decisions += 1
            floor.max_combat_round = max(floor.max_combat_round, _integer(combat.get("round")))
        elif floor.combat_seen:
            floor.left_combat = True

        kind = _action_kind(action)
        selected = _project_action(action)
        candidates = _policy_candidates(event)
        compact_decision = {
            "step": step,
            "screen": screen or None,
            "phase": phase or None,
            "kind": kind,
            "selected": selected,
            "candidates": candidates,
            "legal_action_count": event.get("legal_action_count"),
            "forced": _integer(event.get("legal_action_count"), -1) == 1,
            "value": _finite(event.get("value")),
        }
        self._recent_actions.append(compact_decision)

        if kind == "choose_map_node" and len(self.route_choices) < _MAX_ROUTE_CHOICES:
            map_state = _mapping(observation.get("map"))
            route = {
                **compact_decision,
                "act": _integer(run.get("act")),
                "floor": _integer(run.get("floor")),
                "current_coord": map_state.get("current_coord"),
                "destination": None,
            }
            self.route_choices.append(route)
            self._pending_route = route
        elif kind in {"select_card_reward", "skip_card_reward"} and len(self.card_rewards) < _MAX_CARD_REWARDS:
            reward_state = _mapping(observation.get("card_reward_selection"))
            self.card_rewards.append(
                {
                    **compact_decision,
                    "act": _integer(run.get("act")),
                    "floor": _integer(run.get("floor")),
                    "skipped": kind == "skip_card_reward",
                    "cards_count": reward_state.get("cards_count"),
                    "candidate_coverage": "policy_topk",
                }
            )

        if kind in _MACRO_ACTIONS or (kind == "proceed" and screen in {"EVENT", "REST_SITE", "SHOP", "CARD_SELECTION"}):
            floor.append_macro(compact_decision)

        deadlock = event.get("deadlock")
        if isinstance(deadlock, dict) and len(self.anomalies) < _MAX_ANOMALIES:
            span = max(0, _integer(deadlock.get("cycle_span")))
            cycle = list(self._recent_actions)[-span:] if span else []
            self.anomalies.append(
                {
                    "kind": "deadlock_cycle",
                    "step": step,
                    "act": _integer(run.get("act")),
                    "floor": _integer(run.get("floor")),
                    "screen": screen or None,
                    "cycle_span": span or None,
                    "occurrences": deadlock.get("occurrences"),
                    "first_step": deadlock.get("first_step"),
                    "last_step": deadlock.get("last_step"),
                    "cycle_actions": [item["selected"] for item in cycle],
                    "evidence": dict(deadlock),
                }
            )

    def consume_snapshot(self, event: Mapping[str, object]) -> None:
        """Reduce one rich snapshot without adding to the decision count."""

        self.rich_snapshot_count += 1
        self._snapshot_sequence += 1
        step = _integer(event.get("step_index"), -1)
        raw_reasons = event.get("snapshot_reasons")
        reasons = (
            [
                reason
                for item in raw_reasons[:_MAX_SNAPSHOT_REASONS]
                if (reason := _bounded_string(item)) is not None
            ]
            if isinstance(raw_reasons, list)
            else []
        )
        for source_rank, source_field in enumerate(("observation", "result_observation")):
            observation = _mapping(event.get(source_field))
            if not observation:
                continue
            rank = (step, self._snapshot_sequence, source_rank)
            loadout = _project_loadout(
                observation,
                source_field=source_field,
                step=step,
                reasons=reasons,
            )
            if loadout is not None and (self._latest_loadout_rank is None or rank >= self._latest_loadout_rank):
                self._latest_loadout_rank = rank
                self._latest_loadout = loadout

    def complete(self, event: Mapping[str, object]) -> None:
        if _finite(event.get("attempt")) is not None:
            self.attempt = _integer(event.get("attempt"))
        if _finite(event.get("evaluation_seed")) is not None:
            self.seed = _integer(event.get("evaluation_seed"))
        self.steps = max(self.steps, _integer(event.get("steps")))

    def build(self, provenance: Mapping[str, object]) -> JsonDict:
        floor_builders = list(self.floors.values())
        floors = [
            floor.to_mapping(
                terminal_outcome=self.outcome,
                is_final_floor=index == len(floor_builders) - 1,
            )
            for index, floor in enumerate(floor_builders)
        ]
        combat_counts = {"normal": 0, "elite": 0, "boss": 0, "victory": 0, "failure": 0}
        for floor in floors:
            encounter_class = floor.get("encounter_class")
            if encounter_class == "monster":
                combat_counts["normal"] += 1
            elif encounter_class in {"elite", "boss"}:
                combat_counts[str(encounter_class)] += 1
            result = floor.get("combat_result")
            if result in {"victory", "failure"}:
                combat_counts[str(result)] += 1
        termination_reason = "deadlock_cycle" if self.anomalies else self.outcome
        final_loadout = self._latest_loadout or {
            "source": None,
            "coverage": {
                "kind": "unavailable",
                "bounded": True,
                "complete": False,
                "reason": "no_player_state_in_decision_snapshot",
                "episode_last_snapshot": False,
            },
            "snapshot_step": None,
            "snapshot_reasons": [],
            "hp": None,
            "max_hp": None,
            "gold": None,
            "deck": [],
            "relics": [],
            "potions": [],
        }
        detail = {
            "schema": EPISODE_SUMMARY_SCHEMA,
            "episode_id": self.episode_id,
            "seed": self.seed,
            "attempt": self.attempt,
            "character": self.character,
            "ascension": self.ascension,
            "steps": self.steps,
            "max_act": self.max_act,
            "max_floor": self.max_floor,
            "outcome": self.outcome,
            "termination_reason": termination_reason,
            "revivals": self.revivals,
            "player_hp_lost": self.player_hp_lost,
            "final_hp": self.final_hp,
            "combat_counts": combat_counts,
            "route": self.route_choices,
            "card_rewards": self.card_rewards,
            "floors": floors,
            "anomalies": self.anomalies,
            "final_loadout": final_loadout,
            # Kept as a stable response field.  It is populated asynchronously
            # by the explicit seed/action replay endpoint, never by journal
            # snapshots.
            "map_topologies": [],
            "snapshot_coverage": {
                "source": "decision_snapshot",
                "coverage": (
                    "bounded_rich_snapshot_projection" if self.rich_snapshot_count else "unavailable"
                ),
                "rich_snapshot_count": self.rich_snapshot_count,
            },
            "provenance": {**dict(provenance), "seed": self.seed, "episode_id": self.episode_id},
        }
        return detail


def _episode_index(detail: Mapping[str, object]) -> JsonDict:
    combat_counts = _mapping(detail.get("combat_counts"))
    anomalies = detail.get("anomalies")
    return {
        "episode_id": detail.get("episode_id"),
        "seed": detail.get("seed"),
        "attempt": detail.get("attempt"),
        "character": detail.get("character"),
        "ascension": detail.get("ascension"),
        "outcome": detail.get("outcome"),
        "termination_reason": detail.get("termination_reason"),
        "max_act": detail.get("max_act"),
        "max_floor": detail.get("max_floor"),
        "steps": detail.get("steps"),
        "revivals": detail.get("revivals"),
        "player_hp_lost": detail.get("player_hp_lost"),
        "final_hp": detail.get("final_hp"),
        "normal_combats": combat_counts.get("normal"),
        "elite_combats": combat_counts.get("elite"),
        "boss_combats": combat_counts.get("boss"),
        "combat_victories": combat_counts.get("victory"),
        "combat_failures": combat_counts.get("failure"),
        "anomaly_count": len(anomalies) if isinstance(anomalies, list) else 0,
    }


@dataclass(frozen=True, slots=True)
class ParsedHeldoutJournal:
    provenance: JsonDict
    episodes: tuple[JsonDict, ...]
    details: Mapping[str, JsonDict]
    parsed_rows: int
    malformed_rows: int
    oversize_rows: int
    partial_line: bool


def parse_heldout_journal(path: Path, *, parent: Path) -> ParsedHeldoutJournal:
    """Stream one held-out journal into a compact, bounded run projection."""

    if not _safe_regular_child(path, parent):
        raise ValueError("journal is not a safe regular child of the run directory")
    header: JsonDict | None = None
    builders: OrderedDict[str, _EpisodeBuilder] = OrderedDict()
    started_attempts: dict[int, int] = {}
    parsed_rows = 0
    malformed_rows = 0
    oversize_rows = 0
    partial_line = False
    try:
        with path.open("rb") as handle:
            while True:
                raw = handle.readline(_MAX_LINE_BYTES + 1)
                if not raw:
                    break
                if len(raw) > _MAX_LINE_BYTES:
                    oversize_rows += 1
                    if not raw.endswith(b"\n"):
                        while raw and not raw.endswith(b"\n"):
                            raw = handle.readline(_MAX_LINE_BYTES + 1)
                    continue
                if not raw.endswith(b"\n"):
                    partial_line = True
                    break
                try:
                    event = json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    malformed_rows += 1
                    continue
                if not isinstance(event, dict):
                    malformed_rows += 1
                    continue
                parsed_rows += 1
                kind = event.get("event")
                if kind == "evaluation_started":
                    header = event
                    continue
                if kind == "evaluation_attempt_started":
                    seed = _integer(event.get("evaluation_seed"), -1)
                    attempt = _integer(event.get("attempt"), 1)
                    if seed >= 0:
                        started_attempts[seed] = attempt
                    continue
                if kind not in {"decision", "decision_snapshot", "evaluation_attempt_completed"}:
                    continue
                episode_id = _string(event.get("episode_id"))
                if not episode_id:
                    continue
                builder = builders.get(episode_id)
                if builder is None:
                    if len(builders) >= _MAX_EPISODES:
                        continue
                    builder = _EpisodeBuilder(episode_id)
                    builders[episode_id] = builder
                if kind == "decision":
                    builder.consume(event)
                    if builder.seed is not None and builder.seed in started_attempts:
                        builder.attempt = started_attempts[builder.seed]
                elif kind == "decision_snapshot":
                    builder.consume_snapshot(event)
                else:
                    builder.complete(event)
    except OSError as exc:
        raise ValueError("held-out journal is not readable") from exc
    if header is None:
        raise ValueError("held-out journal has no evaluation_started header")
    provenance = _provenance(header)
    details: OrderedDict[str, JsonDict] = OrderedDict(
        (episode_id, builder.build(provenance)) for episode_id, builder in builders.items()
    )
    episodes = tuple(_episode_index(detail) for detail in details.values())
    return ParsedHeldoutJournal(
        provenance=provenance,
        episodes=episodes,
        details=details,
        parsed_rows=parsed_rows,
        malformed_rows=malformed_rows,
        oversize_rows=oversize_rows,
        partial_line=partial_line,
    )


class HeldoutJournalCache:
    """Small stat-keyed LRU cache; parsing happens only on explicit drill-down."""

    def __init__(self, *, maximum_entries: int = 4) -> None:
        if maximum_entries <= 0:
            raise ValueError("maximum_entries must be positive")
        self.maximum_entries = maximum_entries
        self._lock = threading.RLock()
        self._entries: OrderedDict[Path, tuple[int, int, ParsedHeldoutJournal]] = OrderedDict()

    def load(self, path: Path, *, parent: Path) -> ParsedHeldoutJournal:
        try:
            metadata = path.stat()
        except OSError as exc:
            raise ValueError("held-out journal is not readable") from exc
        identity = (metadata.st_size, metadata.st_mtime_ns)
        with self._lock:
            cached = self._entries.pop(path, None)
            if cached is not None and cached[:2] == identity:
                self._entries[path] = cached
                return cached[2]
        parsed = parse_heldout_journal(path, parent=parent)
        with self._lock:
            self._entries[path] = (*identity, parsed)
            while len(self._entries) > self.maximum_entries:
                self._entries.popitem(last=False)
        return parsed


__all__ = [
    "EPISODE_SUMMARY_SCHEMA",
    "HeldoutJournalCache",
    "ParsedHeldoutJournal",
    "journal_descriptor",
    "parse_heldout_journal",
    "read_journal_header",
]
