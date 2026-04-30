"""Export Ironclad/Colorless base cards and audit mechanism coverage.

The audit is intentionally local-data driven: it reads the game's
``export/items.json`` rather than hard-coding card names.  Outputs are written
under ``docs/generated`` plus a human-readable Markdown report.

This is not a simulator.  Its job is to answer: "which card mechanisms appear
in Ironclad + Colorless cards, and are those mechanisms currently represented
by our bridge/observation/planner contract as typed features, only as text, or
not robustly enough?"
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT_DIR = PROJECT_ROOT / "docs" / "generated"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "docs" / "card-mechanism-coverage-audit.md"
COMMON_ITEMS_PATHS = (
    PROJECT_ROOT / "tmp" / "export" / "items.json",
    PROJECT_ROOT / "export" / "items.json",
    Path(r"E:\Program Files (x86)\Steam\steamapps\common\Slay the Spire 2\export\items.json"),
    Path(r"C:\Program Files (x86)\Steam\steamapps\common\Slay the Spire 2\export\items.json"),
)
COMMON_CATALOG_PATHS = (
    PROJECT_ROOT / "third_party" / "sts2-ai" / "Assets" / "datasets" / "game_knowledge_catalog" / "cards.jsonl",
)
COMMON_SOURCE_ROOTS = (
    PROJECT_ROOT / "third_party" / "sts2-ai",
)


@dataclass(frozen=True)
class CoverageDef:
    status: str
    channel: str
    note: str


COVERAGE_DEFS: dict[str, CoverageDef] = {
    # Strongly typed bridge/obs/planner path.
    "damage": CoverageDef(
        "covered",
        "effect_preview/static semantic_signals/action profile",
        "伤害数值、单体/群体目标、部分 hits 进入 action/card 特征。",
    ),
    "block": CoverageDef(
        "covered",
        "effect_preview/static semantic_signals/action profile",
        "格挡数值进入 action/card 特征。",
    ),
    "draw": CoverageDef(
        "covered",
        "effect_preview/static semantic_signals/action profile",
        "抽牌数进入 source profile / action quality。",
    ),
    "energy_gain": CoverageDef(
        "covered",
        "static semantic_signals + action quality resource timing",
        "回费牌已有能量收益与无后续动作惩罚路径，但依赖静态语义信号完整性。",
    ),
    "hp_or_self_cost": CoverageDef(
        "covered",
        "effect_preview.hp_loss/static semantic_signals + risk classifier",
        "放血/献祭类代价能进入 resource-risk 判定。",
    ),
    "heal_or_max_hp": CoverageDef(
        "covered",
        "effect_preview/static semantic_signals",
        "治疗和最大生命变化可作为数值/文本进入模型；最大生命仍主要依赖静态信号。",
    ),
    "debuff": CoverageDef(
        "covered",
        "effect_preview weak/vulnerable/poison + semantic tags",
        "虚弱/易伤/中毒进入 action 角色与数值。",
    ),
    "buff_stats": CoverageDef(
        "covered",
        "effect_preview strength/dexterity + semantic tags",
        "力量/敏捷/无实体等静态/动态信号可见。",
    ),
    "exhaust_self": CoverageDef(
        "covered",
        "keyword/card_flow/action quality",
        "自身消耗、打出后去向和 strategic-skip 已有字段。",
    ),
    "ethereal": CoverageDef(
        "covered",
        "keyword/card_flow/action quality",
        "虚无牌的跳过/回合末消耗有 counterfactual flow。",
    ),
    "retain": CoverageDef(
        "covered",
        "keyword/card_flow/action quality",
        "保留 keyword 和 end-turn destination 已暴露。",
    ),
    "x_cost": CoverageDef(
        "covered",
        "Bridge resolved x_cost_value + zero-energy X metrics",
        "X 费牌用动态 energy 解析，不应再固定按初始 3 费理解。",
    ),
    "multi_hit_or_aoe": CoverageDef(
        "covered",
        "effect_preview hits/target + semantic tags",
        "多段/群体伤害已有 typed profile。",
    ),
    "potion_or_gold_gain": CoverageDef(
        "covered",
        "static semantic tags/signals + reward/potion profile",
        "炼药/金币等非战斗直接数值主要依赖静态语义。",
    ),

    # Visible, but still needs typed transition/effect-profile work.
    "random_or_choose": CoverageDef(
        "partial",
        "semantic tags/text/card-selection surfaces",
        "随机/三选一结果不是确定 transition；模型能看到但缺分布式结果 profile。",
    ),
    "add_or_generate_card": CoverageDef(
        "partial",
        "semantic tags/internal command ids + future hand snapshot",
        "生成牌/加手牌可由 AddGeneratedCardToCombat 等内部命令定位，但生成池、费用和本回合免费等还需要更强 typed profile。",
    ),
    "copy_card": CoverageDef(
        "partial",
        "internal source ids/static tags",
        "复制目标牌可以从 CreateClone/AddGeneratedCardToCombat 等内部调用定位；仍需要 hand target transition profile。",
    ),
    "upgrade_card": CoverageDef(
        "partial",
        "semantic tags/internal Upgrade command + hand_mutation local",
        "已有 upgrade_card tag / CardCmd.Upgrade 内部命令；缺 zone/scope/filter/selection 等结构参数。",
    ),
    "transform_card": CoverageDef(
        "partial",
        "semantic tags/internal Transform command + hand_mutation local",
        "已有 transform_card tag / CardCmd.Transform 内部命令；缺结果卡、source zone、filter 和 scope profile。",
    ),
    "cost_modify": CoverageDef(
        "partial",
        "internal EnergyCost / power ids + hand_mutation local",
        "EnergyCost.Set*/FreeAttackPower/CorruptionPower 等内部 ID 可定位；仍需目标范围、持续时间、后续可打价值 typed 化。",
    ),
    "exhaust_other_or_hand": CoverageDef(
        "partial",
        "semantic tags/internal Exhaust command + hand_mutation local",
        "CardCmd.Exhaust 内部命令可定位；但选择目标/全手牌/随机目标需要 typed hand transition。",
    ),
    "whole_hand_state": CoverageDef(
        "partial",
        "internal PileType.Hand access + card_flow",
        "可从 PileType.Hand + Upgrade/Transform/Exhaust/EnergyCost 等内部访问定位；仍需显式 hand-state transition。",
    ),
    "pile_fetch_reorder": CoverageDef(
        "partial",
        "pile tokens + internal PileType/CardPileCmd ids",
        "抽牌堆/弃牌堆/牌堆顶部信息可被注意力看到，也能从 PileType/CardPileCmd 内部调用定位；动作的 pile transition 仍不够结构化。",
    ),
    "play_top_or_autoplay": CoverageDef(
        "partial",
        "text/static tags + next state",
        "破灭/倾泻/自动打出类可见，但缺打出目标、消耗去向、连锁结果的 typed lookahead。",
    ),
    "exhaust_pile_dependency": CoverageDef(
        "partial",
        "pile binding tokens + text/static tags",
        "消耗牌堆数量/本回合消耗条件可从状态推断，但卡牌依赖关系缺显式字段。",
    ),
    "discard_pile_dependency": CoverageDef(
        "partial",
        "pile binding tokens + text/static tags",
        "弃牌堆取回/置顶可由注意力学，但缺显式 action->pile delta。",
    ),
    "turn_or_delayed_rule": CoverageDef(
        "partial",
        "powers/text + state transitions",
        "本回合/下回合/每当/回合开始类规则可见但还不是统一 temporal rule token。",
    ),
    "next_card_modifier": CoverageDef(
        "partial",
        "internal power/modifier ids + runtime modifiers",
        "FreeAttackPower、Replay、EnergyCost 等内部 ID 可定位；下一张攻击/重放/免费等仍需要跨动作 memory 与 next-action effect profile。",
    ),
    "card_modifier_or_enchantment": CoverageDef(
        "partial",
        "runtime modifier_summary + internal modifier ids",
        "已有 runtime 附魔/腐化/重放解析；但“给另一张牌添加附魔”的预效果仍需 typed 化。",
    ),
    "global_rule_power": CoverageDef(
        "partial",
        "powers/text + world tokens",
        "腐化/黑暗之拥/怀旧等会改写未来规则；可见但最好变成 rule-transition token。",
    ),
    "remove_card": CoverageDef(
        "partial",
        "text/static tags",
        "删除/移除类对长期 deck 影响可见但缺统一 deck delta profile。",
    ),

    # Explicit gaps in the *typed* contract.  Free text can help offline audit,
    # but the model/bridge target path must be internal IDs + parameters.
    "effect_profile_granularity_gap": CoverageDef(
        "gap",
        "semanticTags/internal command ids are still too coarse",
        "已有 upgrade_card/exhaust_other/transform_card/AddGeneratedCardToCombat 等粗粒度内部 ID，但缺 zone/scope/filter/duration/destination/modifier/count 参数；不能靠描述文本正则补主路径。",
    ),
    "text_only_mechanism_warning": CoverageDef(
        "partial",
        "audit fallback only; not a training contract",
        "该机制目前只能从 description/canonicalText 可靠发现；这是离线审计告警，不应作为模型输入主路径。",
    ),
    "hard_rule_constraint_gap": CoverageDef(
        "gap",
        "mostly raw text/legal mask",
        "不能打出、必须优先、出牌数限制等硬规则缺 typed constraint token/metric。",
    ),
    "status_or_curse_penalty_gap": CoverageDef(
        "gap",
        "raw text + keywords only",
        "灼伤/遗憾/虚空/普通/懒惰等抽到/回合末/出牌限制惩罚缺统一触发器 profile。",
    ),
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _resolve_items_path(explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit)
        if path.exists():
            return path
        raise FileNotFoundError(path)
    for path in COMMON_ITEMS_PATHS:
        if path.exists():
            return path
    checked = "\n  - ".join(str(path) for path in COMMON_ITEMS_PATHS)
    raise FileNotFoundError(f"items.json not found. Checked:\n  - {checked}")


def _normalize_card_id(card_id: Any) -> str:
    text = str(card_id or "").strip()
    if text.upper().startswith("CARD."):
        text = text.split(".", 1)[1]
    return text.lower()


def _load_catalog(path: Path | None = None) -> dict[str, dict[str, Any]]:
    """Load optional decompiled/source catalog facts keyed by normalized card id.

    The exporter ``items.json`` already provides stable card IDs plus coarse
    semantic tags.  The third-party source catalog adds internal implementation
    facts such as command class names (``Upgrade``, ``Transform``,
    ``AddGeneratedCardToCombat``), applied power class names, dynamic var types,
    and the source file.  These are internal IDs, not localized card text.
    """

    catalog_path = path
    if catalog_path is None:
        catalog_path = next((p for p in COMMON_CATALOG_PATHS if p.exists()), None)
    if catalog_path is None or not catalog_path.exists():
        return {}

    out: dict[str, dict[str, Any]] = {}
    with catalog_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            cid = _normalize_card_id(entry.get("id"))
            if cid:
                entry = dict(entry)
                entry["_catalog_path"] = str(catalog_path)
                out[cid] = entry
    return out


def _resolve_source_path(catalog_entry: dict[str, Any] | None) -> Path | None:
    if not isinstance(catalog_entry, dict):
        return None
    rel = str(catalog_entry.get("source_path") or "").strip()
    if not rel:
        return None
    for root in COMMON_SOURCE_ROOTS:
        candidate = root / rel
        if candidate.exists():
            return candidate
    return None


def _load_source_text(catalog_entry: dict[str, Any] | None) -> str:
    path = _resolve_source_path(catalog_entry)
    if path is None:
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def _source_facts(catalog_entry: dict[str, Any] | None) -> dict[str, Any]:
    """Return internal source-level facts used only for audit classification."""

    if not isinstance(catalog_entry, dict):
        return {}
    source = _load_source_text(catalog_entry)
    commands = {str(x).strip() for x in catalog_entry.get("commands") or [] if str(x).strip()}
    powers = {str(x).strip() for x in catalog_entry.get("powers") or [] if str(x).strip()}
    tags = {str(x).strip() for x in catalog_entry.get("tags") or [] if str(x).strip()}
    dynamic_vars = {str(x).strip() for x in catalog_entry.get("dynamic_vars") or [] if str(x).strip()}
    facts = {
        "catalog_id": catalog_entry.get("id"),
        "class_name": catalog_entry.get("class_name"),
        "source_path": catalog_entry.get("source_path"),
        "commands": sorted(commands),
        "powers": sorted(powers),
        "tags": sorted(tags),
        "dynamic_vars": sorted(dynamic_vars),
        "uses_hand_pile": "PileType.Hand" in source or "FromHand" in source,
        "uses_draw_pile": "PileType.Draw" in source,
        "uses_discard_pile": "PileType.Discard" in source,
        "uses_exhaust_pile": "PileType.Exhaust" in source,
        "uses_card_select_from_hand": "CardSelectCmd.FromHand" in source,
        "uses_upgrade_select_from_hand": "CardSelectCmd.FromHandForUpgrade" in source,
        "uses_top_of_pile": "CardPilePosition.Top" in source,
        "creates_clone": ".CreateClone(" in source or "CreateClone()" in source,
        "creates_specific_card": "CreateCard<" in source,
        "sets_energy_cost": "EnergyCost.Set" in source or ".EnergyCost.UpgradeBy" in source,
        "sets_replay": "Replay" in source or "BaseReplayCount" in source or "GetEnchantedReplayCount" in source,
        "adds_keyword_or_modifier": "AddKeyword" in source or "AddEnchantment" in source or "AddAffliction" in source,
        "source_available": bool(source),
    }
    return facts


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _list_strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        text = str(item or "").strip()
        if text:
            out.append(text)
    return out


def _text_blob(card: dict[str, Any], upgrade_entries: list[dict[str, Any]] | None = None) -> str:
    parts: list[str] = []
    for key in ("id", "name", "color", "rarity", "type", "target", "cost", "description", "effect", "canonicalText"):
        value = card.get(key)
        if value not in (None, ""):
            parts.append(str(value))
    for entry in upgrade_entries or []:
        for key in ("description", "effect", "canonicalText"):
            value = entry.get(key)
            if value not in (None, ""):
                parts.append(str(value))
    parts.extend(_list_strings(card.get("keywords")))
    parts.extend(_list_strings(card.get("semanticTags")))
    signals = card.get("semanticSignals")
    if isinstance(signals, dict):
        parts.extend(str(k) for k, v in signals.items() if v not in (None, "", 0, False))
    return " | ".join(parts).lower()


def _has_any(text: str, needles: Iterable[str]) -> bool:
    return any(needle.lower() in text for needle in needles)


def _has_regex(text: str, pattern: str) -> bool:
    return re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL) is not None


def _classify_categories(
    card: dict[str, Any],
    upgrade_entries: list[dict[str, Any]],
    catalog_entry: dict[str, Any] | None = None,
) -> list[str]:
    text = _text_blob(card, upgrade_entries)
    tags = {str(tag).strip().lower() for tag in card.get("semanticTags") or []}
    keywords = {str(keyword).strip().lower() for keyword in card.get("keywords") or []}
    signals = {str(key).strip().lower() for key in (card.get("semanticSignals") or {}).keys()}
    facts = _source_facts(catalog_entry)
    source_commands = {str(cmd).lower() for cmd in facts.get("commands") or []}
    source_powers = {str(power).lower() for power in facts.get("powers") or []}
    source_tags = {str(tag).lower() for tag in facts.get("tags") or []}
    source_dynamic_vars = {str(var).lower() for var in facts.get("dynamic_vars") or []}
    card_type = str(card.get("type") or "").strip().lower()
    rarity = str(card.get("rarity") or "").strip().lower()
    cost = str(card.get("cost") or "").strip().upper()
    target = str(card.get("target") or "").strip().lower()

    cats: set[str] = set()
    add = cats.add

    if "damage" in tags or "damage" in signals or "attack" in source_commands or "damagevar" in source_dynamic_vars or _has_any(text, ("造成", "damage")):
        add("damage")
    if "block" in tags or "block" in signals or "blockvar" in source_dynamic_vars or _has_any(text, ("格挡", "覆甲", "block", "plated armor")):
        add("block")
    if "draw" in tags or "draw" in signals or "draw" in source_commands or _has_any(text, ("抽", "draw")):
        add("draw")
    if "gain_energy" in tags or "energygain" in signals or _has_regex(text, r"(获得|gain).{0,8}(能量|energy)"):
        add("energy_gain")
    if (
        "hp_loss" in tags
        or "hploss" in signals
        or _has_any(text, ("失去", "生命", "自伤", "hp loss", "lose hp", "self damage"))
    ):
        add("hp_or_self_cost")
    if _has_any(text, ("治疗", "回复", "最大生命", "heal", "max hp")) or {"heal", "maxhp", "maxhpgain"} & signals:
        add("heal_or_max_hp")
    if {"apply_vulnerable", "apply_weak"} & tags or {"weak", "vulnerable", "poison"} & signals or "apply" in source_commands and any(p in source_powers for p in ("weakpower", "vulnerablepower", "poisonpower")) or _has_any(text, ("易伤", "虚弱", "中毒", "weak", "vulnerable", "poison")):
        add("debuff")
    if (
        {"gain_strength", "gain_dexterity", "gain_intangible"} & tags
        or {"strengthgain", "dexteritygain", "intangiblegain", "thornsgain"} & signals
        or _has_any(text, ("力量", "敏捷", "无实体", "荆棘", "覆甲", "strength", "dexterity", "intangible", "thorns", "plated armor"))
    ):
        add("buff_stats")
    if "exhaust_self" in tags or "exhaust" in keywords:
        add("exhaust_self")
    if "ethereal" in tags or "ethereal" in keywords or _has_any(text, ("虚无", "ethereal")):
        add("ethereal")
    if "retain" in tags or "retain" in keywords or "retain" in source_tags or _has_any(text, ("保留", "retain")):
        add("retain")
    if cost == "X" or "cost_x" in tags or "hits_x" in tags or _has_any(text, ("x次", "x 张", "x点", "x 点", "x cost")):
        add("x_cost")
    if {"multi_hit", "aoe", "aoe_damage", "hits_x"} & tags or target in {"all enemies", "allenemies"} or _has_any(text, ("所有敌人", "随机一名敌人", "随机敌人", "多次", "次。", "aoe")):
        add("multi_hit_or_aoe")
    if "gain_potion" in tags or "potiongain" in signals or _has_any(text, ("药水", "金币", "potion", "gold")):
        add("potion_or_gold_gain")

    if "random" in tags or facts.get("uses_card_select_from_hand") or facts.get("uses_upgrade_select_from_hand") or _has_any(text, ("随机", "选择", "choose", "select", "random")):
        add("random_or_choose")
    if {"add_to_hand", "generate_card"} & tags or "cardstohand" in signals or {"addgeneratedcardtocombat", "addgeneratedcardstocombat"} & source_commands or _has_any(text, ("加入你的手牌", "加入手牌", "生成", "获得一张", "随机牌", "add", "create")):
        add("add_or_generate_card")
    if facts.get("creates_clone") or _has_any(text, ("复制", "copy", "duplicate")):
        add("copy_card")
    if {"upgrade_card", "upgrade_all"} & tags or "upgrade" in source_commands or facts.get("uses_upgrade_select_from_hand") or _has_any(text, ("升级", "upgrade")):
        add("upgrade_card")
    if "transform_card" in tags or "transform" in source_commands or _has_any(text, ("变化", "变形", "transform", "mutate")):
        add("transform_card")
    if facts.get("sets_energy_cost") or any(power in source_powers for power in ("freeattackpower", "corruptionpower")) or _has_any(text, ("耗能降低", "费用降低", "免费", "变为0点能量", "变为 0", "cost 0", "costs 0", "free this turn", "set cost")):
        add("cost_modify")
    if "exhaust_other" in tags or "exhaust" in source_commands or _has_regex(text, r"(消耗|exhaust).{0,16}(一张|所有|最多|随机|手牌|card)") or _has_regex(text, r"(选择|select).{0,16}(消耗|exhaust)"):
        add("exhaust_other_or_hand")
    if facts.get("uses_hand_pile") and ({"upgrade", "transform", "exhaust", "addgeneratedcardtocombat"} & source_commands or facts.get("sets_energy_cost") or facts.get("creates_clone") or facts.get("sets_replay")):
        add("whole_hand_state")
    if "手牌" in text and _has_any(text, ("所有", "全部", "保留你的手牌", "升级你手牌", "当前手牌", "hand")):
        add("whole_hand_state")
    if facts.get("uses_draw_pile") or facts.get("uses_discard_pile") or facts.get("uses_top_of_pile") or "add" in source_commands or _has_any(text, ("抽牌堆", "弃牌堆", "牌堆顶", "牌堆顶部", "draw pile", "discard pile", "top of")):
        add("pile_fetch_reorder")
    if _has_regex(text, r"(打出|play).{0,12}(抽牌堆|牌堆|draw pile|top)") or _has_any(text, ("自动打出", "autoplay")):
        add("play_top_or_autoplay")
    if facts.get("uses_exhaust_pile") or _has_any(text, ("消耗牌堆", "本回合消耗", "每当你消耗", "exhaust pile", "whenever you exhaust")):
        add("exhaust_pile_dependency")
    if _has_any(text, ("弃牌堆", "discard pile")):
        add("discard_pile_dependency")
    if _has_any(text, ("本回合", "下个回合", "下一回合", "回合开始", "回合结束", "每当", "之后", "this turn", "next turn", "at the start", "at the end", "whenever")):
        add("turn_or_delayed_rule")
    if facts.get("sets_replay") or any(power in source_powers for power in ("freeattackpower", "duplicatenextcardpower", "replaypower")) or _has_any(text, ("下一张", "下一次", "额外打出", "重放", "next attack", "next card", "replay")):
        add("next_card_modifier")
    if facts.get("sets_replay") or facts.get("adds_keyword_or_modifier") or _has_any(text, ("重放", "附魔", "虚无", "永恒", "enchant", "afflict", "replay", "eternal")) or {"eternal"} & tags:
        add("card_modifier_or_enchantment")
    if source_powers or (card_type == "power" and _has_any(text, ("每当", "回合开始", "回合结束", "技能牌", "攻击牌", "本回合", "盟友", "敌人", "承受双倍", "伤害减半", "whenever", "at the start", "at the end"))):
        add("global_rule_power")
    if "remove_card" in tags or _has_any(text, ("移除", "删除", "remove")):
        add("remove_card")

    # Goal-state gap: the source/static layer can usually tell us *which*
    # internal command family exists (Upgrade/Exhaust/Transform/EnergyCost/...),
    # but the runtime model contract still needs typed operation parameters:
    # zone, scope, filter, selection, count, destination, duration, modifier.
    # This replaces the old "localized hand mutation regex" framing.  Text can
    # warn us during offline audit, but it must not become the training contract.
    profile_sensitive = {
        "upgrade_card",
        "transform_card",
        "copy_card",
        "cost_modify",
        "exhaust_other_or_hand",
        "whole_hand_state",
        "pile_fetch_reorder",
        "next_card_modifier",
        "card_modifier_or_enchantment",
        "add_or_generate_card",
        "play_top_or_autoplay",
    }
    if cats & profile_sensitive:
        add("effect_profile_granularity_gap")

    text_sensitive = "手牌" in text and _has_any(
        text,
        ("升级", "消耗", "变化", "复制", "耗能", "费用", "保留", "替换", "放到", "重放", "虚无"),
    )
    semantic_or_source_sensitive = bool(
        tags
        & {
            "upgrade_card",
            "upgrade_all",
            "transform_card",
            "exhaust_other",
            "add_to_hand",
            "generate_card",
            "retain",
        }
        or source_commands
        & {
            "upgrade",
            "transform",
            "exhaust",
            "addgeneratedcardtocombat",
            "addgeneratedcardstocombat",
            "draw",
            "discard",
        }
        or facts.get("uses_hand_pile")
        or facts.get("uses_draw_pile")
        or facts.get("uses_discard_pile")
        or facts.get("sets_energy_cost")
        or facts.get("creates_clone")
        or facts.get("sets_replay")
        or facts.get("adds_keyword_or_modifier")
    )
    if text_sensitive and not semantic_or_source_sensitive:
        add("text_only_mechanism_warning")
    if "unplayable" in tags or "unplayable" in keywords or _has_any(text, ("不能被打出", "必须优先", "不能再打牌", "最多打出", "unplayable", "must be played", "cannot play")):
        add("hard_rule_constraint_gap")
    if card_type in {"curse", "status"} or rarity in {"curse", "status"}:
        add("status_or_curse_penalty_gap")

    return sorted(cats)


def _overall_status(categories: list[str]) -> str:
    statuses = [COVERAGE_DEFS[cat].status for cat in categories if cat in COVERAGE_DEFS]
    if "gap" in statuses:
        return "gap"
    if "partial" in statuses:
        return "partial"
    if "covered" in statuses:
        return "covered"
    return "unclassified"


def _compact_desc(card: dict[str, Any], limit: int = 180) -> str:
    desc = str(card.get("description") or "").replace("\r\n", "\n").replace("\r", "\n")
    desc = " / ".join(part.strip() for part in desc.split("\n") if part.strip())
    if len(desc) > limit:
        return desc[: limit - 1] + "…"
    return desc


def _card_record(
    card: dict[str, Any],
    upgrade_entries: list[dict[str, Any]],
    catalog_entry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source_profile = _source_facts(catalog_entry)
    categories = _classify_categories(card, upgrade_entries, catalog_entry)
    category_statuses = {
        cat: COVERAGE_DEFS[cat].status
        for cat in categories
        if cat in COVERAGE_DEFS
    }
    upgrades = sorted(upgrade_entries, key=lambda item: _safe_int(item.get("upgrades")))
    upgrade_descriptions = [
        {
            "upgrade": _safe_int(entry.get("upgrades")),
            "description": entry.get("description") or "",
            "semanticTags": entry.get("semanticTags") or [],
            "semanticSignals": entry.get("semanticSignals") or {},
        }
        for entry in upgrades
        if _safe_int(entry.get("upgrades")) > 0
    ]
    return {
        "id": card.get("id"),
        "name": card.get("name"),
        "color": card.get("color"),
        "rarity": card.get("rarity"),
        "type": card.get("type"),
        "target": card.get("target"),
        "cost": card.get("cost"),
        "keywords": card.get("keywords") or [],
        "semanticTags": card.get("semanticTags") or [],
        "semanticSignals": card.get("semanticSignals") or {},
        "description": card.get("description") or "",
        "canonicalText": card.get("canonicalText") or "",
        "internal_source_profile": source_profile,
        "upgrade_descriptions": upgrade_descriptions,
        "mechanism_categories": categories,
        "category_statuses": category_statuses,
        "overall_coverage_status": _overall_status(categories),
        "coverage_notes": [
            f"{cat}: {COVERAGE_DEFS[cat].note}"
            for cat in categories
            if cat in COVERAGE_DEFS and COVERAGE_DEFS[cat].status != "covered"
        ],
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "id",
        "name",
        "color",
        "rarity",
        "type",
        "target",
        "cost",
        "overall_coverage_status",
        "mechanism_categories",
        "covered_categories",
        "partial_categories",
        "gap_categories",
        "keywords",
        "semanticTags",
        "semanticSignals",
        "internal_source_profile",
        "description",
        "upgrade_descriptions",
        "coverage_notes",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for rec in records:
            statuses = rec.get("category_statuses") or {}
            row = {
                "id": rec.get("id"),
                "name": rec.get("name"),
                "color": rec.get("color"),
                "rarity": rec.get("rarity"),
                "type": rec.get("type"),
                "target": rec.get("target"),
                "cost": rec.get("cost"),
                "overall_coverage_status": rec.get("overall_coverage_status"),
                "mechanism_categories": "|".join(rec.get("mechanism_categories") or []),
                "covered_categories": "|".join(k for k, v in statuses.items() if v == "covered"),
                "partial_categories": "|".join(k for k, v in statuses.items() if v == "partial"),
                "gap_categories": "|".join(k for k, v in statuses.items() if v == "gap"),
                "keywords": "|".join(rec.get("keywords") or []),
                "semanticTags": "|".join(rec.get("semanticTags") or []),
                "semanticSignals": json.dumps(rec.get("semanticSignals") or {}, ensure_ascii=False, sort_keys=True),
                "internal_source_profile": json.dumps(
                    rec.get("internal_source_profile") or {},
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "description": rec.get("description") or "",
                "upgrade_descriptions": json.dumps(rec.get("upgrade_descriptions") or [], ensure_ascii=False),
                "coverage_notes": " || ".join(rec.get("coverage_notes") or []),
            }
            writer.writerow(row)


def _md_table(headers: list[str], rows: list[list[Any]]) -> str:
    def cell(value: Any) -> str:
        text = str(value)
        return text.replace("\n", "<br>").replace("|", "\\|")

    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    out.extend("| " + " | ".join(cell(value) for value in row) + " |" for row in rows)
    return "\n".join(out)


def _examples(records: list[dict[str, Any]], category: str, limit: int = 6) -> str:
    items = [
        f"{rec['name']}({rec['id']})"
        for rec in records
        if category in (rec.get("mechanism_categories") or [])
    ]
    return "、".join(items[:limit])


def _count_by(records: list[dict[str, Any]], key: str) -> Counter[str]:
    return Counter(str(rec.get(key) or "None") for rec in records)


def _status_counts(records: list[dict[str, Any]]) -> Counter[str]:
    return Counter(str(rec.get("overall_coverage_status") or "unclassified") for rec in records)


def _emit_roster(path: Path, records: list[dict[str, Any]], title: str) -> None:
    rows = []
    for rec in records:
        rows.append([
            rec.get("id"),
            rec.get("name"),
            rec.get("rarity"),
            rec.get("type"),
            rec.get("cost"),
            rec.get("overall_coverage_status"),
            ", ".join(rec.get("mechanism_categories") or []),
            _compact_desc(rec, limit=120),
        ])
    text = "\n".join(
        [
            f"# {title}",
            "",
            f"Generated: {_utc_now()}",
            "",
            _md_table(
                ["ID", "名称", "稀有度", "类型", "费用", "覆盖", "机制分类", "基础描述"],
                rows,
            ),
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_report(
    path: Path,
    *,
    source_path: Path,
    all_records: list[dict[str, Any]],
    ironclad_records: list[dict[str, Any]],
    colorless_records: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    category_counter = Counter()
    by_color_category: dict[str, Counter[str]] = defaultdict(Counter)
    for rec in all_records:
        color = str(rec.get("color") or "unknown")
        for cat in rec.get("mechanism_categories") or []:
            category_counter[cat] += 1
            by_color_category[color][cat] += 1

    coverage_rows = []
    for cat, count in sorted(category_counter.items(), key=lambda kv: (-kv[1], kv[0])):
        cov = COVERAGE_DEFS.get(cat)
        coverage_rows.append([
            cat,
            count,
            by_color_category["ironclad"][cat],
            by_color_category["colorless"][cat],
            cov.status if cov else "unknown",
            cov.channel if cov else "",
            _examples(all_records, cat),
        ])

    high_risk = [
        rec
        for rec in all_records
        if rec.get("overall_coverage_status") == "gap"
        or any(COVERAGE_DEFS.get(cat, CoverageDef("", "", "")).status == "partial" for cat in rec.get("mechanism_categories") or [])
    ]
    high_risk = sorted(
        high_risk,
        key=lambda rec: (
            0 if rec.get("overall_coverage_status") == "gap" else 1,
            str(rec.get("color")),
            str(rec.get("rarity")),
            str(rec.get("id")),
        ),
    )

    high_risk_rows = []
    for rec in high_risk[:80]:
        statuses = rec.get("category_statuses") or {}
        gap_cats = [k for k, v in statuses.items() if v == "gap"]
        partial_cats = [k for k, v in statuses.items() if v == "partial"]
        high_risk_rows.append([
            rec.get("color"),
            rec.get("id"),
            rec.get("name"),
            rec.get("rarity"),
            rec.get("type"),
            rec.get("cost"),
            rec.get("overall_coverage_status"),
            ", ".join(gap_cats),
            ", ".join(partial_cats[:8]),
        ])

    catalog_loaded = int(summary.get("catalog_cards_loaded") or 0)
    catalog_available = int(summary.get("catalog_source_available_for_audited_cards") or 0)
    count_rows = [
        ["Ironclad base cards", len(ironclad_records)],
        ["Colorless base cards", len(colorless_records)],
        ["Colorless playable non Status/Curse/Quest/Token", summary["colorless_playable_non_status_curse_quest_token"]],
        ["Colorless Uncommon/Rare", summary["colorless_uncommon_rare"]],
        ["Total audited base cards", len(all_records)],
        ["Internal catalog cards loaded", catalog_loaded],
        ["Audited cards with source profile", catalog_available],
    ]

    status_rows = []
    for color, records in (("ironclad", ironclad_records), ("colorless", colorless_records), ("all", all_records)):
        counts = _status_counts(records)
        status_rows.append([
            color,
            counts.get("covered", 0),
            counts.get("partial", 0),
            counts.get("gap", 0),
            counts.get("unclassified", 0),
        ])

    rarity_rows = []
    for color, records in (("ironclad", ironclad_records), ("colorless", colorless_records)):
        for rarity, count in sorted(_count_by(records, "rarity").items()):
            rarity_rows.append([color, rarity, count])

    type_rows = []
    for color, records in (("ironclad", ironclad_records), ("colorless", colorless_records)):
        for typ, count in sorted(_count_by(records, "type").items()):
            type_rows.append([color, typ, count])

    text = f"""# Ironclad / Colorless 卡牌机制覆盖审计

Generated: {_utc_now()}

Source: `{source_path}`

## 结论摘要

1. 本地游戏导出的 **Ironclad 基础牌是 {len(ironclad_records)} 张**，不是 88 张。这里没有强行丢牌：`Ancient`/`Event`/`Basic` 也一起导出。若后续要严格对齐“奖励池 88 张”，需要再按奖励池/掉落池规则过滤。
2. 本地游戏导出的 **Colorless 基础牌是 {len(colorless_records)} 张**；其中剔除 Status/Curse/Quest/Token 后的可主动使用无色牌为 **{summary['colorless_playable_non_status_curse_quest_token']} 张**，Uncommon/Rare 常规无色牌为 **{summary['colorless_uncommon_rare']} 张**。
3. 基础伤害/格挡/抽牌/回费/生命代价/X 费/消耗/虚无/保留/运行时附魔，已经有明确 bridge + observation 特征路径。
4. 最大风险不是“模型完全看不到”，而是当前 `semanticTags/semanticSignals` 和 source catalog 只能给粗粒度内部族群（例如 `Upgrade`/`Exhaust`/`Transform`/`AddGeneratedCardToCombat`/`EnergyCost.Set*`），还缺 **zone/scope/filter/selection/count/destination/duration/modifier/result_card** 等 typed operation 参数。文本匹配只保留离线告警，不能作为训练主契约。

## 输出文件

- `docs/generated/ironclad-cards-base.json`：Ironclad 基础牌完整导出 + 分类。
- `docs/generated/colorless-cards-base.json`：Colorless 基础牌完整导出 + 分类。
- `docs/generated/card-mechanism-coverage.csv`：Ironclad + Colorless 全量机制覆盖表，适合 Excel/TB 外部检查。
- `docs/generated/ironclad-card-roster.md`：Ironclad 全表。
- `docs/generated/colorless-card-roster.md`：Colorless 全表。
- `packages/rl-agent/tools/audit_card_mechanism_coverage.py`：可重复生成脚本。

Internal source catalog: `{summary.get('source_catalog_path') or 'not found'}`；本次审计中有 source profile 的卡：**{catalog_available}/{len(all_records)}**。

## 牌数与分布

{_md_table(["项目", "数量"], count_rows)}

### 稀有度分布

{_md_table(["颜色", "稀有度", "数量"], rarity_rows)}

### 类型分布

{_md_table(["颜色", "类型", "数量"], type_rows)}

## 覆盖状态汇总

这里的 `gap` 指“typed effect profile / operation 参数还没进入稳定契约”，不是说 transformer 完全没有文本 token 可看；也不是要求继续堆中文/英文正则。

{_md_table(["颜色", "covered", "partial", "gap", "unclassified"], status_rows)}

## 机制覆盖矩阵

{_md_table(["机制分类", "总数", "Ironclad", "Colorless", "覆盖状态", "当前通道", "例子"], coverage_rows)}

## 高风险 / 需要重点复核的卡

只列前 80 张；全量见 CSV。

{_md_table(["颜色", "ID", "名称", "稀有度", "类型", "费用", "状态", "gap 类别", "partial 类别"], high_risk_rows)}

## 对当前模型结构的判断

### 已经能覆盖得比较好的部分

- **基础数值战斗效果**：伤害、格挡、抽牌、治疗、弱/易伤/毒、力量/敏捷/无实体等，来自 `BridgeGameApi.BuildCardPayload().effect_preview`、静态 `semanticSignals`、`observation_v3` 的 action/source profile。
- **X 费动态**：Bridge 暴露 `costs_x` 与 `effect_preview.x_cost_value/x_cost_semantics`，obs 与训练侧已有 0 能量 X 费指标；理论上不会再把 X 费固定理解成初始 3 费。
- **自身消耗 / 虚无 / 保留 / 运行时附魔**：Bridge 暴露 `keywords`、`card_flow`、`afflictions/enchantments`、`modifier_summary`，obs 也聚合了 `adds_exhaust/adds_retain/adds_ethereal/replay/cost_randomizes_on_draw/sets_cost_zero` 等 modifier。
- **牌堆可见性**：手牌、抽牌堆、弃牌堆、消耗牌堆已经有 token/binding/density 信息，注意力有机会把它们连接起来。

### 仍然不够硬的部分

1. **内部 ID 已经存在，但粒度还不够**  
   `third_party/sts2-ai/.../cards.jsonl` 与 C# 源码能提供 `commands`、`powers`、`PileType.Hand/Draw/Discard/Exhaust`、`CardSelectCmd.FromHand*`、`CardCmd.Upgrade/Exhaust/Transform`、`CardPileCmd.AddGeneratedCardToCombat`、`EnergyCost.Set*`、`CreateClone`、`Replay`、`AddKeyword/AddEnchantment/AddAffliction` 等内部事实。这比文本正则可靠得多。当前缺口不是“识别不到升级/消耗/变化这些词”，而是还没把这些内部调用编译成可训练的 `card_effect_profile.operations`。

2. **action -> pile transition 还不够结构化**  
   头槌、破灭、倾泻、秘密武器/技法、探寻打击、战鼓、好勇斗狠、怀旧等会读/写抽牌堆或弃牌堆。现在模型可以通过 pile tokens 和 source facts 知道访问了哪些 pile，但缺少统一的 `source_zone/destination_zone/topdeck_target/play_top_count/fetch_filter`。

3. **消耗牌堆依赖需要显式条件特征**  
   灰烬打击、契约终结、被遗忘的仪式、邪眼、黑暗之拥、腐化、恶魔之焰、添柴、重振精神等都要求模型理解“当前消耗堆数量 / 本回合是否消耗过 / 消耗后触发”。现在有消耗堆 token 和 `PileType.Exhaust` 访问事实，但每张牌的条件依赖还没有变成 typed feature。

4. **未来规则 / 下一张牌修饰需要 temporal rule token**  
   腐化、无情猛攻、连环拳、怀旧、神气制胜、自动化/地狱狂徒等会改变后续出牌规则。search-free planner 要可靠，需要把“下一张攻击免费/重放/技能 0 费并消耗/每回合第一张置顶”等规则从文本提升为 rule token。

5. **Status/Curse 硬约束缺 typed profile**  
   虚空、灼伤、遗憾、腐朽、普通、懒惰、执迷等不是普通收益牌；有抽到触发、回合末触发、出牌数限制、必须优先打出等硬规则。法律动作 mask 会处理“能不能打”，但策略层需要提前知道“为什么必须处理/为什么不能拖”。

## 建议的目标态补齐顺序

1. **生成 `card_effect_profile.operations`**：从 `cards.jsonl` + C# source facts 编译 typed operations。每个 operation 至少包含 `op/source_zone/destination_zone/scope/selection/count/min_count/max_count/target_filter/duration/modifier/power_id/result_card/created_card/upgraded_override/per_card_scaling`。
2. **Bridge `BuildCardPayload` 暴露 profile**：和 potion `effect_profile` 一样，把卡牌内部 profile 放进 runtime payload；`EnvCompact` / `observation_v3` 只保留压缩后的关键 operation token。
3. **`hand_mutation.py` 主路径改读 operations**：`upgrade_card/exhaust_card/transform_card/copy_card/modify_cost/move_card/add_modifier/add_keyword/set_replay` 等先读 typed op；文本只作为 `text_only_mechanism_warning` 与离线审计 fallback。
4. **把 pile transition / future rule 做成 token**：例如 `FETCH_FROM_DRAW_SKILL`、`TOPDECK_FROM_HAND`、`EXHAUST_HAND_ALL_NON_ATTACK`、`COPY_HAND_ATTACK_OR_POWER`、`NEXT_ATTACK_COST_ZERO`、`SKILL_COST_ZERO_AND_EXHAUST_ON_PLAY`。
5. **给 resource/exhaust deferability 加目标监督**：继续保留“合法但不该打”的策略空间，特别是回费牌没有后续动作、一次性消耗牌当前收益低、保留/虚无/end-turn 去向变化等场景。
6. **Status/Curse/硬约束单独做 aux head**：预测本回合/下回合由状态牌导致的 HP/energy/play-limit 风险，避免只从 reward 后验学习。
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", default=None, help="Path to sts2-exporter items.json")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Generated docs output directory")
    parser.add_argument("--report", default=str(DEFAULT_REPORT_PATH), help="Markdown report path")
    args = parser.parse_args()

    source_path = _resolve_items_path(args.items)
    payload = json.loads(source_path.read_text(encoding="utf-8-sig"))
    raw_cards = payload.get("cards")
    if not isinstance(raw_cards, list):
        raise ValueError(f"Expected cards list in {source_path}")
    catalog = _load_catalog()
    catalog_path = next((entry.get("_catalog_path") for entry in catalog.values() if entry.get("_catalog_path")), None)

    by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for card in raw_cards:
        if not isinstance(card, dict):
            continue
        cid = str(card.get("id") or "").strip()
        if cid:
            by_id[cid].append(card)

    base_cards = [
        card
        for card in raw_cards
        if isinstance(card, dict)
        and _safe_int(card.get("upgrades")) == 0
        and str(card.get("color") or "").strip().lower() in {"ironclad", "colorless"}
    ]
    base_cards.sort(key=lambda c: (str(c.get("color")), str(c.get("rarity")), str(c.get("type")), str(c.get("id"))))

    all_records = [
        _card_record(
            card,
            by_id[str(card.get("id"))],
            catalog.get(_normalize_card_id(card.get("id"))),
        )
        for card in base_cards
    ]
    ironclad_records = [rec for rec in all_records if rec.get("color") == "ironclad"]
    colorless_records = [rec for rec in all_records if rec.get("color") == "colorless"]

    colorless_playable = [
        rec
        for rec in colorless_records
        if str(rec.get("type")) not in {"Status", "Curse", "Quest"}
        and str(rec.get("rarity")) not in {"Status", "Curse", "Quest", "Token"}
    ]
    colorless_uncommon_rare = [
        rec
        for rec in colorless_records
        if str(rec.get("rarity")) in {"Uncommon", "Rare"}
    ]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "generated_at_utc": _utc_now(),
        "source_items_json": str(source_path),
        "source_catalog_path": str(catalog_path or ""),
        "catalog_cards_loaded": len(catalog),
        "catalog_source_available_for_audited_cards": sum(
            1
            for rec in all_records
            if (rec.get("internal_source_profile") or {}).get("source_available")
        ),
        "ironclad_base_count": len(ironclad_records),
        "colorless_base_count": len(colorless_records),
        "colorless_playable_non_status_curse_quest_token": len(colorless_playable),
        "colorless_uncommon_rare": len(colorless_uncommon_rare),
        "status_counts": {
            "ironclad": dict(_status_counts(ironclad_records)),
            "colorless": dict(_status_counts(colorless_records)),
            "all": dict(_status_counts(all_records)),
        },
        "rarity_counts": {
            "ironclad": dict(_count_by(ironclad_records, "rarity")),
            "colorless": dict(_count_by(colorless_records, "rarity")),
        },
        "type_counts": {
            "ironclad": dict(_count_by(ironclad_records, "type")),
            "colorless": dict(_count_by(colorless_records, "type")),
        },
        "coverage_definitions": {
            key: {"status": value.status, "channel": value.channel, "note": value.note}
            for key, value in sorted(COVERAGE_DEFS.items())
        },
    }

    _write_json(out_dir / "ironclad-cards-base.json", {"__summary__": summary, "cards": ironclad_records})
    _write_json(out_dir / "colorless-cards-base.json", {"__summary__": summary, "cards": colorless_records})
    _write_json(out_dir / "card-mechanism-coverage-summary.json", summary)
    _write_csv(out_dir / "card-mechanism-coverage.csv", all_records)
    _emit_roster(out_dir / "ironclad-card-roster.md", ironclad_records, "Ironclad base cards")
    _emit_roster(out_dir / "colorless-card-roster.md", colorless_records, "Colorless base cards")
    _write_report(
        Path(args.report),
        source_path=source_path,
        all_records=all_records,
        ironclad_records=ironclad_records,
        colorless_records=colorless_records,
        summary=summary,
    )

    print(
        json.dumps(
            {
                "status": "ok",
                "source": str(source_path),
                "ironclad_base_count": len(ironclad_records),
                "colorless_base_count": len(colorless_records),
                "colorless_playable_non_status_curse_quest_token": len(colorless_playable),
                "colorless_uncommon_rare": len(colorless_uncommon_rare),
                "catalog_cards_loaded": len(catalog),
                "out_dir": str(out_dir),
                "report": str(Path(args.report)),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
