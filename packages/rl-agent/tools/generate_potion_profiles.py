"""Generator for potions.timing.generated.json — Phase 1 of potion-timing-modeling-plan.md.

Reads `game-data/generated/potions.static.generated.json` (auto-exported from the game's
items.json) and applies regex-driven heuristics on each potion's Chinese
description/summary to fill a baseline timing profile.

The output is consumed (and merged with curated `potions.timing.overrides.json`)
by `sts2_env/potion_profiles.py`.

Design (per plan §3, §4):
  * The auto-generated layer carries only stable, easy-to-detect numerics —
    e.g. "造成 N 点伤害", "获得 N 点格挡", "抽 N 张牌", "回复 N 点生命",
    "给予 N 层 (虚弱|易伤|中毒|...)", "获得 N 点 (力量|敏捷|集中)".
  * Anything ambiguous (block-multiplier, AoE-vs-single, delayed effects,
    self-damage, conditional triggers, pile manipulation) is left for
    `potions.timing.overrides.json` to specify.
  * The generator never sets `enabled_for_training=False` — the overrides
    file owns the deprecation flag for `POTION.DEPRECATED_POTION`.

Usage:
    python tools/generate_potion_profiles.py            # writes game-data/generated/potions.timing.generated.json
    python tools/generate_potion_profiles.py --check    # diff-only, non-zero exit if stale
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

RL_AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(RL_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(RL_AGENT_ROOT))

from sts2_rl.game_data import resolve_generated_game_data_output

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_GAME_DATA_ROOT = Path(os.environ.get("STS2_GAME_DATA_ROOT", _PROJECT_ROOT / "game-data")).expanduser()
_GENERATED_DIR = _GAME_DATA_ROOT / "generated"

STATIC_INPUT = str(_GENERATED_DIR / "potions.static.generated.json")
GENERATED_OUTPUT = str(_GENERATED_DIR / "potions.timing.generated.json")


_DAMAGE_RE = re.compile(r"造成\s*(\d+)\s*点伤害")
_BLOCK_RE = re.compile(r"获得\s*(\d+)\s*点格挡")
_DRAW_RE = re.compile(r"抽\s*(\d+)\s*张牌")
_HEAL_FLAT_RE = re.compile(r"回复\s*(\d+)\s*点(?:生命)?")
_HEAL_PCT_RE = re.compile(r"回复你?最大生命值的\s*(\d+)\s*%")
_MAX_HP_RE = re.compile(r"获得\s*(\d+)\s*点最大生命")
_STRENGTH_RE = re.compile(r"获得\s*(\d+)\s*点力量")
_DEXTERITY_RE = re.compile(r"获得\s*(\d+)\s*点敏捷")
_FOCUS_RE = re.compile(r"获得\s*(\d+)\s*点集中")
_THORNS_RE = re.compile(r"获得\s*(\d+)\s*点荆棘")
_PLATED_RE = re.compile(r"获得\s*(\d+)\s*层覆甲")
_INTANGIBLE_RE = re.compile(r"获得\s*(\d+)\s*层无实体")
_REGEN_RE = re.compile(r"获得\s*(\d+)\s*层再生")
_BUFFER_RE = re.compile(r"获得\s*(\d+)\s*层缓冲")
_RITUAL_RE = re.compile(r"获得\s*(\d+)\s*层仪式")

_WEAK_RE = re.compile(r"给予\s*(?:所有敌人)?(\d+)\s*层虚弱")
_VULNERABLE_RE = re.compile(r"给予\s*(?:所有敌人)?(\d+)\s*层易伤")
_POISON_RE = re.compile(r"给予\s*(?:所有敌人)?(\d+)\s*层中毒")
_DOOM_RE = re.compile(r"给予\s*(\d+)\s*层灾厄")

_ALL_ENEMIES_DAMAGE_RE = re.compile(r"对所有敌人造成\s*(\d+)\s*点伤害")
_ALL_CREATURES_DAMAGE_RE = re.compile(r"对所有玩家和敌人造成\s*(\d+)\s*点伤害")

_ENERGY_GAIN_RE = re.compile(r"获得\s*(\d+)?\s*能量")
_RANDOM_ATTACK_RE = re.compile(r"3\s*张随机攻击牌")
_RANDOM_SKILL_RE = re.compile(r"3\s*张随机技能牌")
_RANDOM_POWER_RE = re.compile(r"3\s*张随机能力牌")
_RANDOM_COLORLESS_RE = re.compile(r"3\s*张随机无色牌")
_GENERATE_CARDS_RE = re.compile(r"将\s*(\d+)\s*张")


def _first_int(pattern: re.Pattern[str], text: str, default: float = 0.0) -> float:
    m = pattern.search(text)
    if not m:
        return default
    try:
        return float(m.group(1))
    except (TypeError, ValueError, IndexError):
        return default


def _has(pattern: re.Pattern[str], text: str) -> bool:
    return pattern.search(text) is not None


def _build_profile(potion_id: str, meta: dict[str, Any]) -> dict[str, Any] | None:
    title = (meta.get("title") or "").strip()
    rarity = meta.get("rarity") or "Unknown"
    text = (meta.get("description") or "") + " " + (meta.get("summary") or "")
    text = text.replace("\n", " ").strip()
    if not text and not title:
        return None

    effect_family: list[str] = []
    semantic_tags: list[str] = []
    timing_tags: list[str] = []
    training_tags: list[str] = []

    profile: dict[str, float | bool] = {}

    aoe_dmg = _first_int(_ALL_ENEMIES_DAMAGE_RE, text)
    if aoe_dmg > 0:
        profile["damage"] = aoe_dmg
        profile["aoe"] = True
        profile["target_required"] = False
        effect_family.extend(["damage", "aoe"])
        semantic_tags.extend(["attack", "damage", "aoe"])
        timing_tags.append("aoe_clear_tool")
    elif (multi_dmg := _first_int(_ALL_CREATURES_DAMAGE_RE, text)) > 0:
        profile["damage"] = multi_dmg
        profile["aoe"] = True
        profile["target_required"] = False
        effect_family.extend(["damage", "aoe", "self_damage"])
        semantic_tags.extend(["attack", "damage", "aoe", "self_damage"])
        timing_tags.append("aoe_clear_tool")
    else:
        single_dmg = _first_int(_DAMAGE_RE, text)
        if single_dmg > 0:
            profile["damage"] = single_dmg
            profile["single_target"] = True
            profile["target_required"] = True
            profile["can_change_facing_if_targeted_enemy"] = True
            effect_family.extend(["damage", "single_target"])
            semantic_tags.extend(["attack", "damage", "single_target"])
            timing_tags.append("lethal_tool")

    block_val = _first_int(_BLOCK_RE, text)
    if block_val > 0:
        profile["block"] = max(profile.get("block", 0.0), block_val)
        profile["target_required"] = profile.get("target_required", False)
        if "block" not in effect_family:
            effect_family.append("block")
        semantic_tags.extend(["block", "defense"])
        timing_tags.append("prevent_lethal_tool")

    draw_val = _first_int(_DRAW_RE, text)
    if draw_val > 0:
        profile["draw"] = draw_val
        if "draw" not in effect_family:
            effect_family.append("draw")
        semantic_tags.append("draw")
        timing_tags.append("dig_for_answer")

    heal_pct = _first_int(_HEAL_PCT_RE, text)
    heal_flat = _first_int(_HEAL_FLAT_RE, text)
    if heal_pct > 0 or heal_flat > 0:
        profile["heal"] = heal_pct if heal_pct > 0 else heal_flat
        if "heal" not in effect_family:
            effect_family.append("heal")
        semantic_tags.append("heal")
        timing_tags.append("emergency_heal")
    if (max_hp := _first_int(_MAX_HP_RE, text)) > 0:
        profile["heal"] = max(profile.get("heal", 0.0), max_hp)
        if "max_hp" not in effect_family:
            effect_family.append("max_hp")
        semantic_tags.append("max_hp")
        timing_tags.append("long_term_value")
        profile["long_term_value"] = True

    if "能量" in text and "获得" in text:
        profile["energy_gain"] = max(profile.get("energy_gain", 0.0), 1.0)
        if "energy" not in effect_family:
            effect_family.append("energy")
        semantic_tags.append("energy")
        timing_tags.append("energy_burst")

    str_val = _first_int(_STRENGTH_RE, text)
    if str_val > 0:
        profile["strength"] = str_val
        if "strength" not in effect_family:
            effect_family.append("strength")
        semantic_tags.append("buff_strength")
        timing_tags.append("scaling_setup")
    dex_val = _first_int(_DEXTERITY_RE, text)
    if dex_val > 0:
        profile["dexterity"] = dex_val
        if "dexterity" not in effect_family:
            effect_family.append("dexterity")
        semantic_tags.append("buff_dexterity")
        timing_tags.append("scaling_setup")
    if (focus_val := _first_int(_FOCUS_RE, text)) > 0:
        profile["focus"] = focus_val
        if "focus" not in effect_family:
            effect_family.append("focus")
        semantic_tags.append("buff_focus")
        timing_tags.append("scaling_setup")
    if (intang_val := _first_int(_INTANGIBLE_RE, text)) > 0:
        profile["intangible"] = intang_val
        if "intangible" not in effect_family:
            effect_family.append("intangible")
        semantic_tags.append("intangible")
        timing_tags.append("prevent_lethal_tool")

    weak_val = _first_int(_WEAK_RE, text)
    if weak_val > 0:
        profile["weak"] = weak_val
        if "weak" not in effect_family:
            effect_family.append("weak")
        semantic_tags.append("debuff_weak")
        timing_tags.append("damage_mitigation_tool")
    vuln_val = _first_int(_VULNERABLE_RE, text)
    if vuln_val > 0:
        profile["vulnerable"] = vuln_val
        if "vulnerable" not in effect_family:
            effect_family.append("vulnerable")
        semantic_tags.append("debuff_vulnerable")
        timing_tags.append("damage_amp_tool")
    poison_val = _first_int(_POISON_RE, text)
    if poison_val > 0:
        profile["poison"] = poison_val
        if "poison" not in effect_family:
            effect_family.append("poison")
        semantic_tags.append("debuff_poison")
        timing_tags.append("scaling_setup")

    if any([
        _has(_RANDOM_ATTACK_RE, text),
        _has(_RANDOM_SKILL_RE, text),
        _has(_RANDOM_POWER_RE, text),
        _has(_RANDOM_COLORLESS_RE, text),
    ]):
        profile["generate_card_count"] = max(profile.get("generate_card_count", 0.0), 1.0)
        profile["discover_count"] = max(profile.get("discover_count", 0.0), 3.0)
        if "generate_cards" not in effect_family:
            effect_family.append("generate_cards")
        semantic_tags.append("generate_cards")
        timing_tags.append("hand_expansion_tool")

    if "升级" in text and ("手牌" in text or "本场战斗" in text):
        if "upgrade" not in effect_family:
            effect_family.append("upgrade")
        semantic_tags.append("upgrade")
        timing_tags.append("scaling_setup")
        profile["long_term_value"] = True
        profile["upgrade_hand"] = max(profile.get("upgrade_hand", 0.0), 1.0)

    seen_family: list[str] = []
    for f in effect_family:
        if f not in seen_family:
            seen_family.append(f)
    seen_sem: list[str] = []
    for f in semantic_tags:
        if f not in seen_sem:
            seen_sem.append(f)
    seen_timing: list[str] = []
    for f in timing_tags:
        if f not in seen_timing:
            seen_timing.append(f)

    if not training_tags:
        if any(t in seen_timing for t in ("lethal_tool", "aoe_clear_tool", "prevent_lethal_tool")):
            training_tags.append("combat_immediate")
        if any(t in seen_timing for t in ("scaling_setup", "long_term_value")):
            training_tags.append("front_load_or_save")

    target_scope = "Self"
    if profile.get("aoe"):
        target_scope = "AllEnemies" if "self_damage" not in seen_family else "AllCreatures"
    elif profile.get("single_target") and profile.get("damage", 0.0) > 0:
        target_scope = "AnyEnemy"
    elif "weak" in seen_family or "vulnerable" in seen_family or "poison" in seen_family:
        target_scope = "AnyEnemy" if not profile.get("aoe") else "AllEnemies"

    return {
        "id": potion_id,
        "title": title,
        "rarity": rarity,
        "target_scope": target_scope,
        "enabled_for_training": True,
        "effect_family": seen_family,
        "effect_profile": profile,
        "semantic_tags": seen_sem,
        "timing_tags": seen_timing,
        "training_tags": training_tags,
    }


def generate(static: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "__meta__": {
            "_doc": "Auto-generated baseline potion timing profiles.",
            "_generator": "tools/generate_potion_profiles.py",
            "_source": "game-data/generated/potions.static.generated.json",
            "_schema_version": 1,
            "_note": "Curated overrides live in potions.timing.overrides.json.",
        }
    }
    for potion_id, meta in static.items():
        if not isinstance(meta, dict) or not potion_id.startswith("POTION."):
            continue
        profile = _build_profile(potion_id, meta)
        if profile is not None:
            out[potion_id] = profile
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Compare generated output against existing file; exit 1 if stale.",
    )
    parser.add_argument(
        "--input",
        default=STATIC_INPUT,
        help=f"Input static potions JSON (default: {STATIC_INPUT})",
    )
    parser.add_argument(
        "--output",
        default=GENERATED_OUTPUT,
        help=f"Output timing JSON (default: {GENERATED_OUTPUT})",
    )
    args = parser.parse_args(argv)

    input_path = resolve_generated_game_data_output(args.input)
    output_path = resolve_generated_game_data_output(args.output)

    with input_path.open("r", encoding="utf-8") as fh:
        static = json.load(fh)

    generated = generate(static)
    payload = json.dumps(generated, ensure_ascii=False, indent=2, sort_keys=True) + "\n"

    if args.check:
        if not output_path.exists():
            print(f"[generate_potion_profiles] STALE: {output_path} missing", file=sys.stderr)
            return 1
        with output_path.open("r", encoding="utf-8") as fh:
            current = fh.read()
        if current != payload:
            print(f"[generate_potion_profiles] STALE: {output_path} differs", file=sys.stderr)
            return 1
        print(f"[generate_potion_profiles] OK: {output_path} up to date "
              f"({len([k for k in generated if k != '__meta__'])} potions)")
        return 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        fh.write(payload)
    n = len([k for k in generated if k != "__meta__"])
    print(f"[generate_potion_profiles] wrote {n} potions -> {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
