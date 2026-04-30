from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
CATALOG_PATH = REPO_ROOT / "third_party" / "sts2-ai" / "Assets" / "datasets" / "game_knowledge_catalog" / "cards.jsonl"
SOURCE_ROOT = REPO_ROOT / "third_party" / "sts2-ai"
OUT_PATH = REPO_ROOT / "packages" / "rl-agent" / "content" / "card_effect_profiles.generated.json"
SCHEMA_VERSION = 1


def normalize_id(value: str | None) -> str:
    """Normalize source/catalog/card-id spellings to lower_snake.

    This intentionally works on internal ids/class names only.  It does not use
    localized card descriptions.
    """
    text = (value or "").strip()
    if text.upper().startswith("CARD."):
        text = text[5:]
    text = text.replace("'", "")
    # Split CamelCase before replacing punctuation so TransfigurePower -> transfigure_power.
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", text)
    text = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", "_", text)
    text = re.sub(r"[^A-Za-z0-9]+", "_", text)
    return text.strip("_").lower()


def card_key(normalized_id: str) -> str:
    return "CARD." + normalize_id(normalized_id).upper()


def unique(seq: list[Any]) -> list[Any]:
    out: list[Any] = []
    seen: set[str] = set()
    for item in seq:
        key = json.dumps(item, sort_keys=True, ensure_ascii=False) if isinstance(item, (dict, list)) else str(item)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def load_catalog() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with CATALOG_PATH.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("entity_type") == "card":
                rows.append(row)
    return rows


def read_source(row: dict[str, Any]) -> str:
    rel = row.get("source_path") or ""
    if not rel:
        return ""
    path = SOURCE_ROOT / rel
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""


def first_int(pattern: str, source: str) -> int | None:
    m = re.search(pattern, source)
    if not m:
        return None
    try:
        return int(float(m.group(1)))
    except Exception:
        return None


def parse_energy_values(source: str) -> tuple[int | None, int | None]:
    # The source usually declares new EnergyVar(2) / new EnergyVar(4) and upgrades
    # with UpgradeValueBy(1m).  We treat that as an internal source fact, not text.
    base = first_int(r"new\s+EnergyVar\s*\(\s*([0-9]+(?:\.0+)?)m?\s*\)", source)
    up_delta = first_int(r"DynamicVars\.Energy\.UpgradeValueBy\s*\(\s*([0-9]+(?:\.0+)?)m?\s*\)", source)
    if up_delta is None:
        up_delta = first_int(r"Energy\.UpgradeValueBy\s*\(\s*([0-9]+(?:\.0+)?)m?\s*\)", source)
    return base, (base + up_delta if base is not None and up_delta is not None else None)


def parse_is_x_cost(source: str) -> bool:
    # Internal source flag — `protected override bool HasEnergyCostX => true;`
    return bool(re.search(r"HasEnergyCostX\s*=>\s*true", source))


def parse_damage_values(source: str) -> tuple[int | None, int | None]:
    base = first_int(r"new\s+DamageVar\s*\(\s*([0-9]+(?:\.0+)?)m?\b", source)
    up = first_int(r"DynamicVars\.Damage\.UpgradeValueBy\s*\(\s*([0-9]+(?:\.0+)?)m?", source)
    if up is None:
        up = first_int(r"Damage\.UpgradeValueBy\s*\(\s*([0-9]+(?:\.0+)?)m?", source)
    return base, (base + up if base is not None and up is not None else None)


def parse_block_values(source: str) -> tuple[int | None, int | None]:
    base = first_int(r"new\s+BlockVar\s*\(\s*([0-9]+(?:\.0+)?)m?\b", source)
    up = first_int(r"DynamicVars\.Block\.UpgradeValueBy\s*\(\s*([0-9]+(?:\.0+)?)m?", source)
    if up is None:
        up = first_int(r"Block\.UpgradeValueBy\s*\(\s*([0-9]+(?:\.0+)?)m?", source)
    return base, (base + up if base is not None and up is not None else None)


def parse_hit_count(source: str) -> int | None:
    # `WithHitCount(2)` / `WithHitCount(num)` — only literal counts are usable.
    val = first_int(r"WithHitCount\s*\(\s*([0-9]+)\s*\)", source)
    return val


def parse_hp_loss(source: str) -> int | None:
    return first_int(r"new\s+HpLossVar\s*\(\s*([0-9]+(?:\.0+)?)m?\s*\)", source)


def parse_power_applies(source: str, row: dict[str, Any]) -> list[str]:
    powers = set(row.get("powers") or [])
    powers.update(re.findall(r"PowerCmd\.Apply<\s*([A-Za-z0-9_]+)\s*>", source))
    return sorted(p for p in powers if p)


def detect_destination_zone(call_context: str) -> str | None:
    if re.search(r"PileType\.Hand\b", call_context):
        return "hand"
    if re.search(r"PileType\.Draw\b", call_context):
        if re.search(r"CardPilePosition\.Top\b", call_context):
            return "draw_pile_top"
        if re.search(r"CardPilePosition\.Bottom\b", call_context):
            return "draw_pile_bottom"
        return "draw_pile"
    if re.search(r"PileType\.Discard\b", call_context):
        return "discard_pile"
    if re.search(r"PileType\.Exhaust\b", call_context):
        return "exhaust_pile"
    return None


def zone_facts(source: str) -> dict[str, bool]:
    return {
        "uses_hand_pile": bool(re.search(r"PileType\.Hand\b|FromHand", source)),
        "uses_draw_pile": bool(re.search(r"PileType\.Draw\b", source)),
        "uses_discard_pile": bool(re.search(r"PileType\.Discard\b", source)),
        "uses_exhaust_pile": bool(re.search(r"PileType\.Exhaust\b", source)),
    }


def source_command_facts(source: str, row: dict[str, Any]) -> list[str]:
    commands = set(row.get("commands") or [])
    for prefix in ("CardCmd", "CardPileCmd", "PlayerCmd", "PowerCmd"):
        commands.update(re.findall(rf"\b{prefix}\.([A-Za-z0-9_]+)", source))
    if "CreateClone" in source:
        commands.add("CreateClone")
    for name in re.findall(r"\.EnergyCost\.([A-Za-z0-9_]+)\s*\(", source):
        commands.add(f"EnergyCost.{name}")
    for name in re.findall(r"\bEnergyCost\.([A-Za-z0-9_]+)\s*\(", source):
        commands.add(f"EnergyCost.{name}")
    if "BaseReplayCount" in source:
        commands.add("BaseReplayCount")
    if "AddKeyword" in source:
        commands.add("AddKeyword")
    return sorted(commands)


def detect_filters(source: str) -> list[str]:
    filters: list[str] = []
    if re.search(r"\.IsUpgradable\b", source):
        filters.append("is_upgradable")
    if re.search(r"\.IsTransformable\b", source):
        filters.append("is_transformable")
    if re.search(r"Type\s*==\s*CardType\.Attack", source):
        filters.append("type_attack")
    if re.search(r"Type\s*==\s*CardType\.Skill", source):
        filters.append("type_skill")
    if re.search(r"Type\s*==\s*CardType\.Power", source):
        filters.append("type_power")
    if re.search(r"Type\s*==\s*CardType\.Status", source):
        filters.append("type_status")
    if re.search(r"Type\s*==\s*CardType\.Curse", source):
        filters.append("type_curse")
    if re.search(r"Type\s*!=\s*CardType\.Attack", source):
        filters.append("not_type_attack")
    if re.search(r"Type\s*!=\s*CardType\.Skill", source):
        filters.append("not_type_skill")
    if re.search(r"Type\s*!=\s*CardType\.Power", source):
        filters.append("not_type_power")
    if re.search(r"Type\s*!=\s*CardType\.Status", source):
        filters.append("not_type_status")
    if re.search(r"Type\s*!=\s*CardType\.Curse", source):
        filters.append("not_type_curse")
    if "CostsX" in source and re.search(r"!\s*[^;\n]*\.EnergyCost\.CostsX", source):
        filters.append("not_x_cost")
    if "CardKeyword.Unplayable" in source and re.search(r"!\s*[^;\n]*Keywords\.Contains\(CardKeyword\.Unplayable\)", source):
        filters.append("playable")
    if "GetEnchantedReplayCount" in source:
        filters.append("without_replay")
    return sorted(set(filters))


def infer_scope_from_source(source: str, *, default: str = "one") -> tuple[str, str | None]:
    # Internal source-shape heuristic: foreach over PileType.Hand -> all; CardSelectCmd -> choice.
    if re.search(r"foreach\s*\([^)]*PileType\.Hand\.GetPile", source, re.DOTALL):
        return "all", "all"
    if "CardSelectCmd.FromHand" in source or "FromHandFor" in source:
        return "one", "choice"
    if "Random" in source or "GetRandom" in source:
        return "one", "random"
    return default, None


def maybe_count_from_dynamic_var(source: str, names: tuple[str, ...] = ("Cards", "Card")) -> int | None:
    for name in names:
        # new DynamicVar("Cards", 2) / new CardsVar(2) shape varies across files.
        val = first_int(rf"new\s+[A-Za-z0-9_]*Var\s*\(\s*\"{re.escape(name)}\"\s*,\s*([0-9]+(?:\.0+)?)m?", source)
        if val is not None:
            return val
        val = first_int(rf"new\s+{re.escape(name)}Var\s*\(\s*([0-9]+(?:\.0+)?)m?", source)
        if val is not None:
            return val
    return None


def generic_operations(row: dict[str, Any], source: str) -> list[dict[str, Any]]:
    ops: list[dict[str, Any]] = []
    commands = set(row.get("commands") or [])
    filters = detect_filters(source)

    # Current-hand card state changes.  These profiles are consumed by
    # hand_mutation.py before falling back to text.
    if "CardCmd.Upgrade" in source or "Upgrade" in commands:
        scope, selection = infer_scope_from_source(source)
        op: dict[str, Any] = {"op": "upgrade_card", "target_filter": ["is_upgradable"]}
        if "PileType.Hand" in source or "FromHand" in source:
            op["source_zone"] = "hand"
        op["scope"] = scope
        if selection:
            op["selection"] = selection
        if scope == "one":
            op["count"] = 1
        ops.append(op)

    if "CardCmd.Exhaust" in source or "Exhaust" in commands:
        scope, selection = infer_scope_from_source(source)
        op = {"op": "exhaust_card"}
        if "PileType.Hand" in source or "FromHand" in source:
            op["source_zone"] = "hand"
        if filters:
            op["target_filter"] = filters
        op["scope"] = scope
        if selection:
            op["selection"] = selection
        if scope == "one":
            op["count"] = 1
        ops.append(op)

    if "CardCmd.Discard" in source or "Discard" in commands:
        scope, selection = infer_scope_from_source(source)
        op = {"op": "discard_card"}
        if "PileType.Hand" in source or "FromHand" in source:
            op["source_zone"] = "hand"
        op["scope"] = scope
        if selection:
            op["selection"] = selection
        if scope == "one":
            op["count"] = 1
        ops.append(op)

    if "CardCmd.Transform" in source or "CardCmd.TransformTo" in source or "Transform" in commands or "TransformTo" in commands:
        scope, selection = infer_scope_from_source(source)
        op = {"op": "transform_card"}
        if "PileType.Hand" in source or "FromHand" in source:
            op["source_zone"] = "hand"
        if filters:
            op["target_filter"] = filters
        op["scope"] = scope
        if selection:
            op["selection"] = selection
        if scope == "one":
            op["count"] = 1
        m = re.search(r"TransformTo<\s*([A-Za-z0-9_]+)\s*>", source)
        if m:
            op["result_card"] = m.group(1)
        ops.append(op)

    # Cost/card modifiers.
    if re.search(r"\.EnergyCost\.(SetThisTurnOrUntilPlayed|SetThisCombat|SetThisTurn|SetToFreeThisTurn|AddThisCombat|AddThisTurn)\b", source):
        op = {"op": "modify_cost"}
        if "PileType.Hand" in source or "FromHand" in source:
            op["source_zone"] = "hand"
        if "SetThisTurnOrUntilPlayed" in source:
            op["duration"] = "this_turn_or_until_played"
        elif "SetThisCombat" in source or "AddThisCombat" in source:
            op["duration"] = "this_combat"
        elif "SetThisTurn" in source or "SetToFreeThisTurn" in source or "AddThisTurn" in source:
            op["duration"] = "this_turn"
        if "SetToFreeThisTurn" in source or re.search(r"\.EnergyCost\.Set[A-Za-z0-9_]*\s*\(\s*0\s*\)", source):
            op["set_cost"] = 0
        elif re.search(r"\.EnergyCost\.Set[A-Za-z0-9_]*", source):
            val = first_int(r"\.EnergyCost\.Set[A-Za-z0-9_]*\s*\(\s*([0-9]+)", source)
            if val is not None:
                op["set_cost"] = val
                op["reduce_only"] = True
        elif re.search(r"\.EnergyCost\.Add[A-Za-z0-9_]*\s*\(\s*-?\s*[0-9]+", source):
            val = first_int(r"\.EnergyCost\.Add[A-Za-z0-9_]*\s*\(\s*(-?[0-9]+)", source)
            if val is not None:
                op["cost_delta"] = val
        if filters:
            op["target_filter"] = filters
        scope, selection = infer_scope_from_source(source)
        op["scope"] = scope
        if selection:
            op["selection"] = selection
        ops.append(op)

    if "BaseReplayCount" in source:
        scope, selection = infer_scope_from_source(source)
        op = {"op": "set_replay", "stacks": 1, "scope": scope}
        if "PileType.Hand" in source or "FromHand" in source:
            op["source_zone"] = "hand"
        if filters:
            op["target_filter"] = filters
        if selection:
            op["selection"] = selection
        ops.append(op)

    if re.search(r"AddKeyword\s*\(\s*CardKeyword\.Retain", source):
        scope, selection = infer_scope_from_source(source)
        op = {"op": "add_keyword", "keyword": "retain", "scope": scope}
        if "PileType.Hand" in source or "FromHand" in source:
            op["source_zone"] = "hand"
        if selection:
            op["selection"] = selection
        ops.append(op)

    # Pile and generation operations.
    if "CardPileCmd.Draw" in source or "Draw" in commands:
        count = maybe_count_from_dynamic_var(source) or None
        op = {"op": "draw_card", "destination_zone": "hand"}
        if count is not None:
            op["count"] = count
        ops.append(op)

    if "CreateClone" in source and ("AddGeneratedCardToCombat" in source or "AddGeneratedCardsToCombat" in source):
        call_context = source[max(source.find("AddGeneratedCard"), 0): max(source.find("AddGeneratedCard"), 0) + 300]
        dest = detect_destination_zone(call_context) or "hand"
        op = {"op": "copy_card", "destination_zone": dest}
        if "PileType.Hand" in source or "FromHand" in source:
            op["source_zone"] = "hand"
        scope, selection = infer_scope_from_source(source)
        op["scope"] = scope
        if selection:
            op["selection"] = selection
        if filters:
            op["target_filter"] = filters
        count = maybe_count_from_dynamic_var(source) or 1
        op["copy_count"] = count
        ops.append(op)
    elif "AddGeneratedCardToCombat" in source or "AddGeneratedCardsToCombat" in source or "AddGeneratedCardToCombat" in commands:
        idx = source.find("AddGeneratedCard")
        call_context = source[max(idx, 0): max(idx, 0) + 500]
        dest = detect_destination_zone(call_context) or "hand"
        op = {"op": "add_generated_card", "destination_zone": dest}
        count = maybe_count_from_dynamic_var(source) or None
        if count is not None:
            op["count"] = count
        generated = re.findall(r"new\s+([A-Z][A-Za-z0-9_]*)\s*\(", call_context)
        if generated:
            op["generated_card"] = generated[0]
        ops.append(op)

    if "CardPileCmd.Add" in source or "Add" in commands:
        idx = source.find("CardPileCmd.Add")
        call_context = source[max(idx, 0): max(idx, 0) + 500]
        dest = detect_destination_zone(call_context)
        if dest:
            op = {"op": "move_card", "destination_zone": dest}
            if "PileType.Hand" in source or "FromHand" in source:
                op["source_zone"] = "hand"
                op["selection"] = "choice" if "FromHand" in source else None
                op["count"] = 1
            ops.append({k: v for k, v in op.items() if v is not None})

    # Resource/power rules.
    if "PlayerCmd.GainEnergy" in source or "GainEnergy" in commands:
        base, upgraded = parse_energy_values(source)
        op = {"op": "gain_energy", "timing": "same_turn_resource"}
        if base is not None:
            op["energy"] = base
        if upgraded is not None:
            op["upgraded_energy"] = upgraded
        hp = parse_hp_loss(source)
        if hp is not None:
            op["hp_loss"] = hp
        if row.get("card_type") == "Skill":
            op["strategic_skip_if_no_followup"] = True
        ops.append(op)

    for power in parse_power_applies(source, row):
        op = {"op": "apply_power", "power_id": power}
        if power == "CorruptionPower":
            op["future_rule"] = "skill_cost_zero_and_exhaust_on_play"
        elif power == "FreeAttackPower":
            op["future_rule"] = "next_attack_cost_zero"
        elif power == "NoDrawPower":
            op["future_rule"] = "no_draw_this_turn"
        elif power == "BorrowedTimePower":
            op["future_rule"] = "future_extra_cost_or_penalty"
        ops.append(op)

    return unique([op for op in ops if op.get("op")])


CURATED: dict[str, list[dict[str, Any]]] = {
    "armaments": [
        {
            "op": "upgrade_card",
            "source_zone": "hand",
            "selection": "choice",
            "scope": "one",
            "target_filter": ["is_upgradable"],
            "count": 1,
            "upgraded_override": {"selection": "all", "scope": "all"},
        }
    ],
    "purity": [
        {
            "op": "exhaust_card",
            "source_zone": "hand",
            "selection": "choice",
            "min_count": 0,
            "max_count": 3,
            "upgraded_max_count": 5,
        }
    ],
    "primal_force": [
        {
            "op": "transform_card",
            "source_zone": "hand",
            "scope": "all",
            "target_filter": ["is_transformable", "type_attack"],
            "result_card": "GiantRock",
            "upgraded_result_upgraded": True,
        }
    ],
    "dual_wield": [
        {
            "op": "copy_card",
            "source_zone": "hand",
            "destination_zone": "hand",
            "selection": "choice",
            "target_filter": ["type_attack", "type_power"],
            "copy_count": 1,
            "upgraded_copy_count": 2,
        }
    ],
    "enlightenment": [
        {
            "op": "modify_cost",
            "source_zone": "hand",
            "scope": "all",
            "set_cost": 1,
            "reduce_only": True,
            "duration": "this_turn_or_until_played",
            "upgraded_duration": "this_combat",
        }
    ],
    "thinking_ahead": [
        {"op": "draw_card", "destination_zone": "hand", "count": 2},
        {
            "op": "move_card",
            "source_zone": "hand",
            "destination_zone": "draw_pile_top",
            "selection": "choice",
            "count": 1,
        },
    ],
    "hidden_gem": [
        {
            "op": "add_modifier",
            "source_zone": "draw_pile",
            "selection": "random",
            "target_filter": ["playable", "not_type_status", "not_type_curse", "without_replay"],
            "modifier": "replay",
            "stacks": 2,
            "upgraded_stacks": 3,
        }
    ],
    "unrelenting": [
        {"op": "apply_power", "power_id": "FreeAttackPower", "future_rule": "next_attack_cost_zero"}
    ],
    "corruption": [
        {"op": "apply_power", "power_id": "CorruptionPower", "future_rule": "skill_cost_zero_and_exhaust_on_play"}
    ],
    "second_wind": [
        {
            "op": "exhaust_card",
            "source_zone": "hand",
            "scope": "all",
            "target_filter": ["not_type_attack"],
            "per_card_scaling": {"block_per_exhausted_card": 5, "upgraded_block_per_exhausted_card": 7},
        }
    ],
    "fiend_fire": [
        {
            "op": "exhaust_card",
            "source_zone": "hand",
            "scope": "all",
            "per_card_scaling": {
                "damage_per_exhausted_card": 7,
                "upgraded_damage_per_exhausted_card": 10,
                "hit_count": "exhausted_count",
            },
        }
    ],
    "transfigure": [
        {
            "op": "modify_cost",
            "source_zone": "hand",
            "selection": "choice",
            "count": 1,
            "cost_delta": 1,
            "duration": "this_combat",
            "target_filter": ["not_x_cost"],
        },
        {"op": "set_replay", "source_zone": "hand", "selection": "choice", "count": 1, "stacks": 1},
    ],
    "brand": [
        {"op": "exhaust_card", "source_zone": "hand", "selection": "choice", "count": 1}
    ],
    "true_grit": [
        {
            "op": "exhaust_card",
            "source_zone": "hand",
            "selection": "random",
            "count": 1,
            "upgraded_override": {"selection": "choice", "scope": "one", "count": 1},
        }
    ],
    "bullet_time": [
        {
            "op": "modify_cost",
            "source_zone": "hand",
            "scope": "all",
            "set_cost": 0,
            "duration": "this_turn",
            "target_filter": ["not_x_cost"],
        },
        {"op": "apply_power", "power_id": "NoDrawPower", "future_rule": "no_draw_this_turn"},
    ],
    "bloodletting": [
        {"op": "gain_energy", "energy": 2, "upgraded_energy": 3, "hp_loss": 3, "timing": "same_turn_resource"}
    ],
    "production": [
        {
            "op": "gain_energy",
            "energy": 2,
            "upgraded_energy": 3,
            "timing": "same_turn_resource",
            "strategic_skip_if_no_followup": True,
        }
    ],
    "double_energy": [
        {
            "op": "gain_energy",
            "energy_mode": "double_current_energy",
            "timing": "same_turn_resource",
            "strategic_skip_if_no_followup": True,
        }
    ],
    "borrowed_time": [
        {
            "op": "gain_energy",
            "energy": 4,
            "upgraded_energy": 6,
            "timing": "same_turn_resource",
            "strategic_skip_if_no_followup": True,
        },
        {"op": "apply_power", "power_id": "BorrowedTimePower", "future_rule": "future_extra_cost_or_penalty"},
    ],
}


def semantic_tags_for(ops: list[dict[str, Any]], row: dict[str, Any]) -> list[str]:
    tags = set(str(t) for t in (row.get("tags") or []) if isinstance(t, str))
    for op in ops:
        name = op.get("op")
        if name:
            tags.add(str(name))
        if op.get("source_zone") == "hand" or op.get("destination_zone") == "hand":
            tags.add("hand_mutation")
        if op.get("op") in {"modify_cost", "gain_energy"}:
            tags.add("resource_timing")
        if op.get("op") in {"apply_power", "add_modifier", "add_keyword", "set_replay", "retain_card"}:
            tags.add("card_rule_modifier")
        if op.get("op") in {"exhaust_card", "transform_card", "discard_card"}:
            tags.add("hand_removal")
    return sorted(tags)


def _has_op(ops: list[dict[str, Any]], name: str) -> bool:
    return any(op.get("op") == name for op in ops)


def _ops_named(ops: list[dict[str, Any]], name: str) -> list[dict[str, Any]]:
    return [op for op in ops if op.get("op") == name]


def _has_keyword(row: dict[str, Any], keyword: str) -> bool:
    target = keyword.strip().lower()
    for kw in row.get("keywords") or []:
        if isinstance(kw, str) and kw.strip().lower() == target:
            return True
    return False


def _derive_cost_view(row: dict[str, Any], source: str, ops: list[dict[str, Any]]) -> dict[str, Any]:
    raw_cost = row.get("cost")
    base_int: int | None
    if isinstance(raw_cost, (int, float)):
        base_int = int(raw_cost)
    elif isinstance(raw_cost, str):
        text = raw_cost.strip()
        try:
            base_int = int(text)
        except ValueError:
            base_int = None  # X / unparseable
    else:
        base_int = None
    is_x_cost = parse_is_x_cost(source) or (
        isinstance(raw_cost, str) and raw_cost.strip().upper() == "X"
    )
    cost_modifier_op = next(iter(_ops_named(ops, "modify_cost")), None)
    can_change_cost = bool(cost_modifier_op)
    cost_reduction_tags: list[str] = []
    if cost_modifier_op:
        if cost_modifier_op.get("set_cost") is not None:
            cost_reduction_tags.append(f"set_cost_{cost_modifier_op['set_cost']}")
        if cost_modifier_op.get("cost_delta") is not None:
            cost_reduction_tags.append(f"delta_{cost_modifier_op['cost_delta']}")
        if cost_modifier_op.get("duration"):
            cost_reduction_tags.append(f"duration_{cost_modifier_op['duration']}")
    return {
        "base": base_int if base_int is not None else (-1 if is_x_cost else None),
        "upgraded": base_int if base_int is not None else (-1 if is_x_cost else None),
        "is_x_cost": bool(is_x_cost),
        "can_change_cost": can_change_cost,
        "cost_reduction_tags": sorted(set(cost_reduction_tags)),
    }


def _derive_lifecycle_view(
    row: dict[str, Any], source: str, ops: list[dict[str, Any]]
) -> dict[str, Any]:
    keywords = {(kw or "").strip().lower() for kw in row.get("keywords") or []}
    tag_set = {(t or "").strip().lower() for t in row.get("tags") or []}
    exhausts_on_play = "exhaust" in keywords or "exhaust_self" in tag_set
    ethereal = "ethereal" in keywords or "ethereal" in tag_set
    retain = "retain" in keywords
    self_purge = bool(re.search(r"PurgeSelf|RemoveFromCombat\s*\(\s*this\b", source))
    returns_to_hand = bool(re.search(r"PileType\.Hand[^;]*Add\s*\(\s*this", source))
    replay_or_duplicate = (
        "BaseReplayCount" in source
        or _has_op(ops, "set_replay")
        or _has_op(ops, "copy_card")
    )
    return {
        "exhausts_on_play": bool(exhausts_on_play),
        "ethereal": bool(ethereal),
        "retain": bool(retain),
        "self_purge": bool(self_purge),
        "returns_to_hand": bool(returns_to_hand),
        "replay_or_duplicate": bool(replay_or_duplicate),
    }


def _derive_hand_mutation_view(
    row: dict[str, Any], source: str, ops: list[dict[str, Any]]
) -> dict[str, Any]:
    upgrade_ops = _ops_named(ops, "upgrade_card")
    transform_ops = _ops_named(ops, "transform_card")
    copy_ops = _ops_named(ops, "copy_card")
    create_ops = _ops_named(ops, "add_generated_card")
    discard_ops = _ops_named(ops, "discard_card")
    draw_ops = _ops_named(ops, "draw_card")
    discard_hand = any(op.get("scope") in {"all", "aoe", "hand"} for op in discard_ops)
    draw_total = 0
    for op in draw_ops:
        count = op.get("count")
        if isinstance(count, (int, float)):
            draw_total += int(count)
    upgrade_targets: str | None = None
    if upgrade_ops:
        op = upgrade_ops[0]
        if op.get("scope") in {"all", "aoe", "hand"} or "upgraded_override" in op:
            upgrade_targets = "one_or_all_by_upgrade_state"
        elif op.get("selection") == "choice":
            upgrade_targets = "choice_one"
        elif op.get("selection") == "random":
            upgrade_targets = "random_one"
        else:
            upgrade_targets = "one"
    select_op = next(
        (op for op in ops if (op.get("selection") in {"choice", "select", "target"})),
        None,
    )
    selection: dict[str, Any]
    if select_op is not None:
        selection = {
            "enabled": True,
            "min": int(select_op.get("min_count") or select_op.get("count") or 1),
            "max": int(
                select_op.get("max_count")
                or select_op.get("upgraded_max_count")
                or select_op.get("count")
                or 1
            ),
            "target_zone": select_op.get("source_zone") or "hand",
        }
    else:
        selection = {"enabled": False, "min": 0, "max": 0, "target_zone": None}
    return {
        "upgrades_hand": bool(upgrade_ops),
        "upgrade_targets": upgrade_targets,
        "transforms_cards": bool(transform_ops),
        "copies_cards": bool(copy_ops),
        "creates_cards": bool(create_ops),
        "discard_hand": bool(discard_hand),
        "draw": draw_total if draw_total > 0 else 0,
        "select_cards": selection,
    }


def _derive_pile_mutation_view(
    row: dict[str, Any], source: str, ops: list[dict[str, Any]]
) -> dict[str, Any]:
    card_type = (row.get("card_type") or "").lower()
    exhaust_self = _has_keyword(row, "exhaust") or _has_keyword(row, "ethereal")
    is_power = card_type == "power"
    # Played-card destination: exhaust if Exhaust/Ethereal keyword, removed (no
    # discard) if Power, else discard by default.
    moves_to_exhaust_self = bool(exhaust_self)
    moves_to_discard_self = (not exhaust_self) and (not is_power) and card_type not in {"", "status", "curse", "quest"}
    moves_to_exhaust_op = _has_op(ops, "exhaust_card")
    shuffles_into_draw = any(
        op.get("destination_zone") in {"draw_pile", "draw_pile_top", "draw_pile_bottom"}
        for op in ops
    )
    puts_card_on_top = any(op.get("destination_zone") == "draw_pile_top" for op in ops)
    transform_ops = _ops_named(ops, "transform_card")
    removes_card_from_combat = any(
        not op.get("result_card") and not op.get("upgraded_result_upgraded")
        for op in transform_ops
    )
    return {
        "moves_to_exhaust": bool(moves_to_exhaust_self or moves_to_exhaust_op),
        "moves_to_exhaust_self": bool(moves_to_exhaust_self),
        "moves_to_discard_self": bool(moves_to_discard_self),
        "moves_to_discard": bool(_has_op(ops, "discard_card") or any(op.get("destination_zone") == "discard_pile" for op in ops)),
        "shuffles_into_draw": bool(shuffles_into_draw),
        "puts_card_on_top": bool(puts_card_on_top),
        "removes_card_from_combat": bool(removes_card_from_combat),
    }


_DEBUFF_POWER_TO_FIELD = {
    "WeakPower": "weak",
    "VulnerablePower": "vulnerable",
    "FrailPower": "frail",
    "PoisonPower": "poison",
}
_BUFF_POWER_TO_FIELD = {
    "StrengthPower": "strength",
    "DexterityPower": "dexterity",
    "ArtifactPower": "artifact",
    "ThornsPower": "thorns",
}


def _derive_combat_effect_view(
    row: dict[str, Any], source: str, ops: list[dict[str, Any]]
) -> dict[str, Any]:
    base_dmg, _ = parse_damage_values(source)
    base_blk, _ = parse_block_values(source)
    hit_count = parse_hit_count(source) or 1
    energy_gain = 0
    for op in _ops_named(ops, "gain_energy"):
        energy = op.get("energy")
        if isinstance(energy, (int, float)):
            energy_gain += int(energy)
    powers_field: dict[str, int] = {}
    for op in _ops_named(ops, "apply_power"):
        pid = (op.get("power_id") or "").strip()
        for table in (_DEBUFF_POWER_TO_FIELD, _BUFF_POWER_TO_FIELD):
            if pid in table:
                powers_field[table[pid]] = powers_field.get(table[pid], 0) + 1
    return {
        "damage": base_dmg if base_dmg is not None else 0,
        "block": base_blk if base_blk is not None else 0,
        "hit_count": hit_count,
        "weak": powers_field.get("weak", 0),
        "vulnerable": powers_field.get("vulnerable", 0),
        "frail": powers_field.get("frail", 0),
        "poison": powers_field.get("poison", 0),
        "strength": powers_field.get("strength", 0),
        "dexterity": powers_field.get("dexterity", 0),
        "artifact": powers_field.get("artifact", 0),
        "thorns": powers_field.get("thorns", 0),
        "energy_gain": energy_gain,
        "target_type": row.get("target_type"),
    }


def _derive_mechanism_effect_view(
    row: dict[str, Any], source: str, ops: list[dict[str, Any]]
) -> dict[str, Any]:
    target = (row.get("target_type") or "").strip()
    can_change_facing = target in {"AnyEnemy", "AllEnemies", "AnyOpponent"}
    artifact_strip_keywords = ("StripArtifact", "ArtifactStrip", "ConsumeArtifact")
    can_strip_artifact = any(token in source for token in artifact_strip_keywords)
    can_trigger_stun = "StunPower" in source or any(
        op.get("power_id") == "StunPower" for op in _ops_named(ops, "apply_power")
    )
    return {
        "can_change_facing": bool(can_change_facing),
        "can_strip_artifact": bool(can_strip_artifact),
        "can_trigger_stun": bool(can_trigger_stun),
        "one_card_lock_impact": "unknown",
    }


def _derive_source_view(
    row: dict[str, Any], source_facts: dict[str, Any], ops: list[dict[str, Any]]
) -> dict[str, Any]:
    quality = source_facts.get("source_profile_quality") or "unknown"
    has_source = bool(source_facts.get("commands")) or bool(source_facts.get("powers"))
    return {
        "primary": "game_internal_id",
        "fallback_text_regex_used": False,
        "profile_quality": quality,
        "source_available": bool(has_source),
        "source_sha1": row.get("source_sha1"),
    }


def derive_view(
    row: dict[str, Any], source: str, ops: list[dict[str, Any]], source_facts: dict[str, Any]
) -> dict[str, Any]:
    return {
        "card_id": card_key(normalize_id(row.get("id") or row.get("class_name"))),
        "title": row.get("title_en"),
        "color": (row.get("color") or row.get("class_color") or "").lower() or None,
        "type": (row.get("card_type") or "").lower() or None,
        "rarity": (row.get("rarity") or "").lower() or None,
        "cost": _derive_cost_view(row, source, ops),
        "lifecycle": _derive_lifecycle_view(row, source, ops),
        "hand_mutation": _derive_hand_mutation_view(row, source, ops),
        "pile_mutation": _derive_pile_mutation_view(row, source, ops),
        "combat_effect": _derive_combat_effect_view(row, source, ops),
        "mechanism_effect": _derive_mechanism_effect_view(row, source, ops),
        "source": _derive_source_view(row, source_facts, ops),
    }


def build_profile(row: dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_id(row.get("id") or row.get("class_name"))
    source = read_source(row)
    source_facts = {
        "commands": source_command_facts(source, row),
        "powers": parse_power_applies(source, row),
        **zone_facts(source),
    }
    curated_ops = CURATED.get(normalized)
    ops = curated_ops if curated_ops is not None else generic_operations(row, source)
    source_facts["source_profile_quality"] = "curated_internal_id" if curated_ops is not None else "generated_source_facts"
    source_facts["operation_count"] = len(ops)
    return {
        "schema_version": SCHEMA_VERSION,
        "id": card_key(normalized),
        "normalized_id": normalized,
        "class_name": row.get("class_name"),
        "title_en": row.get("title_en"),
        "title_zhs": row.get("title_zhs"),
        "source_path": row.get("source_path"),
        "source_sha1": row.get("source_sha1"),
        "operations": ops,
        "semantic_tags": semantic_tags_for(ops, row),
        "training_tags": ["typed_card_effect_profile", source_facts["source_profile_quality"]],
        "source_facts": source_facts,
        "derived_view": derive_view(row, source, ops, source_facts),
    }


def main() -> None:
    rows = load_catalog()
    cards: dict[str, Any] = {}
    for row in rows:
        profile = build_profile(row)
        cards[profile["id"]] = profile
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "source": {
            "catalog": str(CATALOG_PATH.relative_to(REPO_ROOT)).replace("\\", "/"),
            "source_root": str(SOURCE_ROOT.relative_to(REPO_ROOT)).replace("\\", "/"),
            "method": "catalog_internal_fields_plus_csharp_source_facts_no_localized_text_regex",
        },
        "cards": dict(sorted(cards.items())),
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    curated_count = sum(1 for p in cards.values() if "curated_internal_id" in p.get("training_tags", []))
    op_count = sum(len(p.get("operations") or []) for p in cards.values())
    profiled = sum(1 for p in cards.values() if p.get("operations"))
    print(f"wrote {OUT_PATH}")
    print(f"cards={len(cards)} profiled={profiled} operations={op_count} curated={curated_count}")


if __name__ == "__main__":
    main()
