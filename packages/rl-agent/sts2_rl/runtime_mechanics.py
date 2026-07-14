"""Fail-closed catalog and runtime audit for the model-facing mechanics ABI.

The learner must receive exact game facts rather than handwritten card or
boss heuristics.  This module verifies both halves of that contract:

* the simulator can enumerate every mechanics family used by a full run;
* representative event and combat states carry the observable instance data.

Hidden transition targets and future random outcomes are deliberately not part
of the ABI.  They would leak information unavailable to a normal player.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NoReturn, cast

from sts2_rl.artifacts import resolve_artifact_path

_AUDIT_SCHEMA = "sts2-runtime-mechanics-audit-v1"
_SEQUENCE_TYPES = (str, bytes, bytearray)

_COLLECTION_CONTRACTS: dict[str, tuple[str, frozenset[str], int]] = {
    "powers": (
        "power_id",
        frozenset(
            {
                "power_id",
                "class_name",
                "type",
                "stack_type",
                "is_instanced",
                "allow_negative",
                "should_scale_in_multiplayer",
                "owner_is_secondary_enemy",
                "dynamic_vars",
                "base_classes",
            }
        ),
        200,
    ),
    "relics": (
        "relic_id",
        frozenset(
            {
                "relic_id",
                "class_name",
                "rarity",
                "tags",
                "is_tradable",
                "is_allowed_in_shops",
                "has_upon_pickup_effect",
                "spawns_pets",
                "is_stackable",
                "adds_pet",
                "merchant_cost",
                "show_counter",
                "display_amount",
                "dynamic_vars",
            }
        ),
        200,
    ),
    "potions": (
        "potion_id",
        frozenset(
            {
                "potion_id",
                "class_name",
                "rarity",
                "usage",
                "target_type",
                "can_be_generated_in_combat",
                "passes_custom_usability_check",
                "dynamic_vars",
            }
        ),
        50,
    ),
    "enchantments": (
        "enchantment_id",
        frozenset(
            {
                "enchantment_id",
                "class_name",
                "show_amount",
                "is_stackable",
                "should_start_at_bottom_of_draw_pile",
                "should_glow_gold",
                "should_glow_red",
                "has_extra_card_text",
                "dynamic_vars",
            }
        ),
        10,
    ),
    "afflictions": (
        "affliction_id",
        frozenset(
            {
                "affliction_id",
                "class_name",
                "is_stackable",
                "has_extra_card_text",
                "can_afflict_unplayable_cards",
                "has_overlay",
            }
        ),
        5,
    ),
    "events": (
        "event_id",
        frozenset(
            {
                "event_id",
                "class_name",
                "layout_type",
                "is_deterministic",
                "is_shared",
                "encounter_id",
                "dynamic_vars",
            }
        ),
        40,
    ),
    "monsters": (
        "monster_id",
        frozenset(
            {
                "monster_id",
                "class_name",
                "min_initial_hp",
                "max_initial_hp",
                "powers",
            }
        ),
        80,
    ),
    "encounters": (
        "encounter_id",
        frozenset(
            {"encounter_id", "room_type", "monster_ids", "act_index"}
        ),
        60,
    ),
}

_DYNAMIC_VAR_FIELDS = frozenset(
    {
        "name",
        "var_type",
        "family",
        "base_value",
        "enchanted_value",
        "preview_value",
        "int_value",
        "was_just_upgraded",
    }
)

_CARD_INSTANCE_FIELDS = frozenset(
    {
        "id",
        "type",
        "cost",
        "target_type",
        "gains_block",
        "has_turn_end_in_hand_effect",
        "has_on_draw_effect",
        "exhaust_on_next_play",
        "base_replay_count",
        "current_replay_count",
        "is_in_combat",
        "is_retained",
        "dynamic_vars",
        "enchantments",
        "afflictions",
    }
)

_POWER_INSTANCE_FIELDS = frozenset(
    {
        "id",
        "amount",
        "type",
        "stack_type",
        "is_visible",
        "is_instanced",
        "allow_negative",
        "dynamic_vars",
    }
)

_RELIC_INSTANCE_FIELDS = frozenset(
    {
        "id",
        "rarity",
        "status",
        "is_used_up",
        "is_tradable",
        "is_allowed_in_shops",
        "is_stackable",
        "stack_count",
        "merchant_cost",
        "show_counter",
        "display_amount",
        "dynamic_vars",
    }
)

_POTION_INSTANCE_FIELDS = frozenset(
    {
        "id",
        "rarity",
        "usage",
        "target_type",
        "can_be_generated_in_combat",
        "passes_custom_usability_check",
        "dynamic_vars",
    }
)

_ENEMY_INSTANCE_FIELDS = frozenset(
    {
        "entity_id",
        "combat_id",
        "hp",
        "max_hp",
        "block",
        "is_alive",
        "is_hittable",
        "next_move_id",
        "intends_to_attack",
        "is_primary_enemy",
        "is_secondary_enemy",
        "is_stunned",
        "is_pet",
        "shows_infinite_hp",
        "can_receive_powers",
        "spawned_this_turn",
        "is_performing_move",
        "is_move",
        "must_perform_once_before_transitioning",
        "can_transition_away",
        "move_history",
        "status",
        "intents",
    }
)


class RuntimeMechanicsAuditError(RuntimeError):
    """The simulator does not satisfy the mechanics ABI."""


def _fail(message: str) -> NoReturn:
    raise RuntimeMechanicsAuditError(message)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{label} must be an object")
    return cast(Mapping[str, Any], value)


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, _SEQUENCE_TYPES):
        _fail(f"{label} must be an array")
    return value


def _required(value: Mapping[str, Any], fields: frozenset[str], label: str) -> None:
    missing = fields.difference(value)
    if missing:
        _fail(f"{label} is missing fields: {sorted(missing)}")


def _audit_dynamic_vars(value: Any, label: str) -> int:
    values = _sequence(value, label)
    for index, raw in enumerate(values):
        item = _mapping(raw, f"{label}[{index}]")
        _required(item, _DYNAMIC_VAR_FIELDS, f"{label}[{index}]")
        if not str(item["name"]).strip() or not str(item["family"]).strip():
            _fail(f"{label}[{index}] has an empty name/family")
        for field in ("base_value", "enchanted_value", "preview_value", "int_value"):
            number = item[field]
            if isinstance(number, bool) or not isinstance(number, int | float):
                _fail(f"{label}[{index}].{field} must be numeric")
    return len(values)


def audit_game_catalog(catalog: Mapping[str, Any]) -> dict[str, Any]:
    """Validate complete native catalogs and return stable coverage counts."""

    root = _mapping(catalog, "game_catalog")
    if "error" in root:
        _fail(f"game_catalog RPC failed: {root['error']}")
    counts: dict[str, int] = {}
    dynamic_var_counts: dict[str, int] = {}
    indexed: dict[str, set[str]] = {}
    for collection, (id_field, required_fields, minimum) in _COLLECTION_CONTRACTS.items():
        values = _sequence(root.get(collection), f"game_catalog.{collection}")
        if len(values) < minimum:
            _fail(
                f"game_catalog.{collection} is unexpectedly small: "
                f"expected >= {minimum}, actual={len(values)}"
            )
        identities: set[str] = set()
        dynamic_count = 0
        for index, raw in enumerate(values):
            item = _mapping(raw, f"game_catalog.{collection}[{index}]")
            _required(item, required_fields, f"game_catalog.{collection}[{index}]")
            identity = str(item[id_field]).strip()
            if not identity or identity in identities:
                _fail(f"game_catalog.{collection} has invalid/duplicate {id_field}: {identity!r}")
            identities.add(identity)
            if "is_debuff_hint" in item:
                _fail("handwritten is_debuff_hint is forbidden; use native PowerModel.Type")
            if "follow_up_state_id" in item:
                _fail("hidden follow_up_state_id must not enter the model-facing catalog")
            if "dynamic_vars" in item:
                dynamic_count += _audit_dynamic_vars(
                    item["dynamic_vars"],
                    f"game_catalog.{collection}[{index}].dynamic_vars",
                )
            for sequence_field in ("tags", "base_classes", "powers", "monster_ids"):
                if sequence_field in item:
                    _sequence(
                        item[sequence_field],
                        f"game_catalog.{collection}[{index}].{sequence_field}",
                    )
        counts[collection] = len(values)
        dynamic_var_counts[collection] = dynamic_count
        indexed[collection] = identities

    encounters = _sequence(root["encounters"], "game_catalog.encounters")
    monster_ids = indexed["monsters"]
    boss_encounters = [
        _mapping(item, "boss encounter")
        for item in encounters
        if isinstance(item, Mapping) and str(item.get("room_type", "")).lower() == "boss"
    ]
    if len(boss_encounters) < 5:
        _fail(f"boss encounter catalog is unexpectedly small: {len(boss_encounters)}")
    boss_monsters: set[str] = set()
    for encounter in boss_encounters:
        for monster_id in _sequence(encounter["monster_ids"], "boss.monster_ids"):
            normalized = str(monster_id).strip()
            if normalized not in monster_ids:
                _fail(f"boss encounter references unknown monster_id: {normalized!r}")
            boss_monsters.add(normalized)
    if not boss_monsters:
        _fail("boss encounter catalog contains no monsters")

    return {
        "schema": _AUDIT_SCHEMA,
        "catalog_counts": counts,
        "dynamic_var_counts": dynamic_var_counts,
        "boss_encounter_count": len(boss_encounters),
        "boss_monster_count": len(boss_monsters),
    }


def _audit_card(card: Any, label: str) -> None:
    item = _mapping(card, label)
    _required(item, _CARD_INSTANCE_FIELDS, label)
    _audit_dynamic_vars(item["dynamic_vars"], f"{label}.dynamic_vars")
    for field in ("enchantments", "afflictions"):
        for index, raw in enumerate(_sequence(item[field], f"{label}.{field}")):
            modifier = _mapping(raw, f"{label}.{field}[{index}]")
            if not str(modifier.get("id", "")).strip():
                _fail(f"{label}.{field}[{index}] has no id")
            if "dynamic_vars" in modifier:
                _audit_dynamic_vars(
                    modifier["dynamic_vars"],
                    f"{label}.{field}[{index}].dynamic_vars",
                )


def _audit_power(power: Any, label: str) -> None:
    item = _mapping(power, label)
    _required(item, _POWER_INSTANCE_FIELDS, label)
    _audit_dynamic_vars(item["dynamic_vars"], f"{label}.dynamic_vars")


def _audit_player(player: Any, label: str) -> None:
    item = _mapping(player, label)
    for field in ("status", "relics", "potions", "deck"):
        _sequence(item.get(field), f"{label}.{field}")
    for index, power in enumerate(item["status"]):
        _audit_power(power, f"{label}.status[{index}]")
    for index, raw in enumerate(item["relics"]):
        relic = _mapping(raw, f"{label}.relics[{index}]")
        _required(relic, _RELIC_INSTANCE_FIELDS, f"{label}.relics[{index}]")
        _audit_dynamic_vars(
            relic["dynamic_vars"], f"{label}.relics[{index}].dynamic_vars"
        )
    for index, raw in enumerate(item["potions"]):
        potion = _mapping(raw, f"{label}.potions[{index}]")
        _required(potion, _POTION_INSTANCE_FIELDS, f"{label}.potions[{index}]")
        _audit_dynamic_vars(
            potion["dynamic_vars"], f"{label}.potions[{index}].dynamic_vars"
        )
    for index, card in enumerate(item["deck"]):
        _audit_card(card, f"{label}.deck[{index}]")


def audit_runtime_state_sequence(
    initial_bridge_state: Mapping[str, Any],
    raw_state: Callable[[], Mapping[str, Any]],
    step: Callable[[int], Mapping[str, Any]],
    *,
    max_steps: int = 64,
) -> dict[str, Any]:
    """Audit one deterministic event-to-combat bootstrap trajectory."""

    bridge_state = _mapping(initial_bridge_state, "initial bridge state")
    event_checked = False
    combat_checked = False
    for step_index in range(max_steps + 1):
        raw = _mapping(raw_state(), f"runtime state {step_index}")
        state_type = str(raw.get("state_type", ""))
        if state_type == "event" and not event_checked:
            event = _mapping(raw.get("event"), "runtime.event")
            _required(
                event,
                frozenset(
                    {
                        "event_id",
                        "layout_type",
                        "description_key",
                        "is_deterministic",
                        "is_shared",
                        "dynamic_vars",
                        "player",
                        "is_finished",
                        "options",
                    }
                ),
                "runtime.event",
            )
            _audit_dynamic_vars(event["dynamic_vars"], "runtime.event.dynamic_vars")
            _audit_player(event["player"], "runtime.event.player")
            options = _sequence(event["options"], "runtime.event.options")
            if not options:
                _fail("runtime event exposes no options")
            for index, raw_option in enumerate(options):
                option = _mapping(raw_option, f"runtime.event.options[{index}]")
                _required(
                    option,
                    frozenset(
                        {
                            "index",
                            "text",
                            "text_key",
                            "description",
                            "is_locked",
                            "is_chosen",
                            "is_proceed",
                        }
                    ),
                    f"runtime.event.options[{index}]",
                )
            event_checked = True

        if state_type == "monster":
            battle = _mapping(raw.get("battle"), "runtime.battle")
            player = _mapping(battle.get("player"), "runtime.battle.player")
            _audit_player(player, "runtime.battle.player")
            hand = _sequence(player.get("hand"), "runtime.battle.player.hand")
            if not hand:
                _fail("runtime combat hand is empty")
            for index, card in enumerate(hand):
                _audit_card(card, f"runtime.battle.player.hand[{index}]")
            enemies = _sequence(battle.get("enemies"), "runtime.battle.enemies")
            if not enemies:
                _fail("runtime combat has no enemies")
            for index, raw_enemy in enumerate(enemies):
                enemy = _mapping(raw_enemy, f"runtime.battle.enemies[{index}]")
                _required(enemy, _ENEMY_INSTANCE_FIELDS, f"runtime.battle.enemies[{index}]")
                if "follow_up_state_id" in enemy:
                    _fail("runtime enemy leaks hidden follow_up_state_id")
                _sequence(enemy["move_history"], f"runtime.battle.enemies[{index}].move_history")
                _sequence(enemy["intents"], f"runtime.battle.enemies[{index}].intents")
                for power_index, power in enumerate(
                    _sequence(enemy["status"], f"runtime.battle.enemies[{index}].status")
                ):
                    _audit_power(
                        power,
                        f"runtime.battle.enemies[{index}].status[{power_index}]",
                    )
            combat_checked = True
            return {
                "runtime_event_checked": event_checked,
                "runtime_combat_checked": combat_checked,
                "runtime_bootstrap_steps": step_index,
                "runtime_enemy_count": len(enemies),
                "runtime_hand_count": len(hand),
            }

        actions = _sequence(bridge_state.get("legal_actions"), "bridge legal_actions")
        enabled_index = next(
            (
                index
                for index, action in enumerate(actions)
                if isinstance(action, Mapping)
                and action.get("is_enabled", action.get("enabled", True)) is True
            ),
            None,
        )
        if enabled_index is None:
            _fail(f"runtime bootstrap has no enabled action at step {step_index}")
        bridge_state = _mapping(step(enabled_index), f"bridge state {step_index + 1}")

    _fail(
        "runtime bootstrap did not reach combat within "
        f"{max_steps} actions (event_checked={event_checked})"
    )


def run_runtime_mechanics_preflight(executable: str | Path) -> dict[str, Any]:
    """Launch the verified simulator and execute the complete mechanics gate."""

    from sts2_env.headless_sim_bridge_client import HeadlessSimBridgeClient

    with HeadlessSimBridgeClient(exe_path=Path(executable)) as client:
        summary = audit_game_catalog(client._rpc("game_catalog"))
        bridge_state = client.reset(
            character="IRONCLAD",
            seed="RUNTIME_MECHANICS_AUDIT",
            training_revival_budget=1,
        )
        runtime = audit_runtime_state_sequence(
            bridge_state,
            lambda: client._rpc("state"),
            lambda index: client.step(bridge_state["episode_id"], action_index=index),
        )
    return {**summary, **runtime}


def write_runtime_mechanics_audit(summary: Mapping[str, Any]) -> Path:
    """Persist a successful preflight outside the source tree."""

    directory = resolve_artifact_path("logs/runtime-mechanics-preflight")
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    target = directory / f"runtime-mechanics-{stamp}.json"
    payload = {
        "event": "runtime_mechanics_verified",
        "verified_at_utc": datetime.now(UTC).isoformat(),
        **dict(summary),
    }
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target


__all__ = [
    "RuntimeMechanicsAuditError",
    "audit_game_catalog",
    "audit_runtime_state_sequence",
    "run_runtime_mechanics_preflight",
    "write_runtime_mechanics_audit",
]
