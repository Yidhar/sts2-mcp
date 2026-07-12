# ruff: noqa: RUF001
"""Pure decision-screen and event payload translators."""

from __future__ import annotations

from typing import Any

from ._sim_translate_entities import _translate_card


def _build_decision_block(
    *,
    state_type: str,
    in_combat: bool,
    event: dict[str, Any],
    map_state: dict[str, Any],
    rest_site: dict[str, Any],
    shop: dict[str, Any],
    treasure: dict[str, Any],
    rewards: dict[str, Any],
    card_reward: dict[str, Any],
    card_select: dict[str, Any],
    hand_select: dict[str, Any],
) -> dict[str, Any]:
    """Mirror of bridge's BuildEnvDecisionPayload phase switch.

    Returns the 10-scalar dict observation_common reads (~line 863). Keys
    not applicable to a phase are simply omitted — _float default is 0.0.
    """
    if state_type == "event":
        opts = event.get("options") or []
        title = str(event.get("title") or event.get("name") or "事件")
        return {
            "option_count": len(opts),
            "decision_text": f"事件｜{title}｜{len(opts)}个选项",
        }
    if state_type == "card_reward":
        choices = card_reward.get("cards") or []
        return {
            "option_count": len(choices),
            "can_skip": bool(card_reward.get("can_skip", True)),
            "decision_text": f"卡牌奖励｜{len(choices)}张卡牌可选",
        }
    if state_type in {"rewards", "combat_rewards", "combat_post_end_pending"}:
        items = rewards.get("items") or []
        return {
            "reward_count": len(items),
            "proceed_only": bool(rewards.get("can_proceed", False)) and len(items) == 0,
            "decision_text": f"奖励选择｜可领取{len(items)}项奖励",
        }
    if state_type == "map":
        opts = map_state.get("next_options") or []
        return {
            "travelable_count": len(opts),
            "decision_text": f"地图｜{len(opts)}个可选节点",
        }
    if state_type == "rest_site":
        opts = rest_site.get("options") or []
        return {
            "option_count": len(opts),
            "can_proceed": bool(rest_site.get("can_proceed", False)),
            "decision_text": "营火｜选择休息或锻造",
        }
    if state_type == "shop":
        items = shop.get("items") or []
        return {
            "is_open": bool(shop.get("is_open", False)),
            "item_count": len(items),
            "decision_text": f"商店｜{len(items)}件商品",
        }
    if state_type == "treasure":
        relic_opts = treasure.get("relics") or []
        return {
            "option_count": len(relic_opts),
            "can_proceed": bool(treasure.get("can_open", True)),
            "decision_text": f"宝箱｜{len(relic_opts)}件遗物可选",
        }
    if state_type in {"card_select", "hand_select"}:
        src = card_select if card_select else hand_select
        if not isinstance(src, dict):
            src = {}
        prompt = str(src.get("prompt") or "").strip()
        min_sel = int(src.get("min_select") or 0)
        max_sel = int(src.get("max_select") or 1)
        selected = len(src.get("selected_cards") or [])
        label = prompt or ("手牌选择" if state_type == "hand_select" else "卡牌选择")
        return {
            "selected_count": selected,
            "min_select": min_sel,
            "max_select": max_sel,
            "can_skip": bool(src.get("can_cancel", False)),
            "decision_text": f"{label}｜已选{selected}｜{min_sel}-{max_sel}张",
        }
    if in_combat:
        # During combat there's no global decision prompt — return empty.
        # Combat features come through obs["combat"] token pipeline.
        return {}
    return {}


_EVENT_DELTA_CARD_COUNT_TOKENS = {
    "a": 1, "an": 1, "one": 1, "一": 1,
    "two": 2, "两": 2,
    "three": 3, "三": 3,
}


def _event_parse_card_count_token(raw: str) -> int:
    try:
        return int(raw)
    except (TypeError, ValueError):
        return _EVENT_DELTA_CARD_COUNT_TOKENS.get(raw.lower(), 1)


def _event_sum_first_match(
    lower: str, original: str,
    english_patterns: list[str], chinese_patterns: list[str],
) -> int:
    import re as _re
    for pattern in english_patterns:
        m = _re.search(pattern, lower, _re.IGNORECASE)
        if m and m.group(1).isdigit():
            return int(m.group(1))
    for pattern in chinese_patterns:
        m = _re.search(pattern, original)
        if m and m.group(1).isdigit():
            return int(m.group(1))
    return 0


def _event_count_card_op(
    lower: str, original: str,
    english_patterns: list[str], chinese_patterns: list[str],
) -> int:
    import re as _re
    for pattern in english_patterns:
        m = _re.search(pattern, lower, _re.IGNORECASE)
        if m:
            return _event_parse_card_count_token(m.group(1))
    for pattern in chinese_patterns:
        m = _re.search(pattern, original)
        if m:
            return _event_parse_card_count_token(m.group(1))
    return 0


def _extract_event_option_effect_deltas(title: str | None, description: str | None) -> dict[str, Any]:
    """Port of bridge BridgeGameApi.EnvHelpers.ExtractEventOptionEffectDeltas
    (EN + ZH regex patterns). Parses event option text into 17 structured
    signal fields the obs encoder consumes for event-choice reasoning.

    Missing patterns degrade to 0 rather than lying. Exact parity with
    live bridge's parser for the 17 keys emitted under effect_deltas.
    """
    import re as _re
    deltas: dict[str, Any] = {
        "hp_delta": 0,
        "max_hp_delta": 0,
        "gold_delta": 0,
        "heal_full": False,
        "card_add_count": 0,
        "card_add_attack": False,
        "card_add_skill": False,
        "card_add_power": False,
        "card_add_curse": False,
        "card_add_status": False,
        "card_remove_count": 0,
        "card_transform_count": 0,
        "card_upgrade_count": 0,
        "card_duplicate_count": 0,
        "relic_gain": False,
        "potion_gain": False,
        "enter_combat": False,
    }
    if not (title or description):
        return deltas
    combined = " \n ".join(s for s in (title or "", description or "") if s)
    lower = combined.lower()
    original = combined

    # HP delta (signed)
    hp_lose = _event_sum_first_match(lower, original,
        [r"lose\s*(\d+)\s*hp", r"take\s*(\d+)\s*damage", r"you\s*take\s*(\d+)",
         r"suffer\s*(\d+)\s*damage", r"receive\s*(\d+)\s*damage"],
        [r"失去(\d+)点?(?:生命|hp)", r"受到(\d+)点?伤害", r"扣除?(\d+)点?(?:生命|hp)"])
    hp_gain = _event_sum_first_match(lower, original,
        [r"gain\s*(\d+)\s*hp", r"heal\s*(\d+)\s*hp?", r"restore\s*(\d+)\s*hp",
         r"recover\s*(\d+)\s*hp"],
        [r"(?:回复|恢复|治疗)(\d+)点?(?:生命|hp)", r"获得(\d+)点?(?:生命|hp)"])
    deltas["hp_delta"] = hp_gain - hp_lose
    if (_re.search(r"\b(heal(ed)?\s*(to\s*)?full|fully\s*heal|restore\s*all\s*hp)\b", lower)
            or _re.search(r"(回满|满血|回复全部生命|治疗至满)", original)):
        deltas["heal_full"] = True

    # Max HP delta (signed)
    max_gain = _event_sum_first_match(lower, original,
        [r"max\s*hp\s*\+\s*(\d+)", r"gain\s*(\d+)\s*max\s*hp",
         r"(\d+)\s*max\s*hp", r"increase\s*max\s*hp\s*by\s*(\d+)"],
        [r"最大生命(?:增加|提高|提升|上升)?\+?(\d+)", r"max\s*hp\s*\+?(\d+)"])
    max_lose = _event_sum_first_match(lower, original,
        [r"max\s*hp\s*-\s*(\d+)", r"lose\s*(\d+)\s*max\s*hp",
         r"decrease\s*max\s*hp\s*by\s*(\d+)"],
        [r"最大生命(?:减少|降低|下降)(\d+)", r"失去(\d+)点?最大生命"])
    deltas["max_hp_delta"] = max_gain - max_lose

    # Gold
    gold_gain = _event_sum_first_match(lower, original,
        [r"gain\s*(\d+)\s*gold", r"receive\s*(\d+)\s*gold",
         r"(\d+)\s*gold", r"obtain\s*(\d+)\s*gold"],
        [r"获得(\d+)点?金币", r"(\d+)点?金币"])
    gold_lose = _event_sum_first_match(lower, original,
        [r"lose\s*(\d+)\s*gold", r"pay\s*(\d+)\s*gold", r"spend\s*(\d+)\s*gold"],
        [r"失去(\d+)点?金币", r"支付(\d+)点?金币", r"花费(\d+)点?金币"])
    deltas["gold_delta"] = gold_gain - gold_lose

    # Card-add count + types
    attack = bool(_re.search(r"\battack\b", lower)) or ("攻击" in original)
    skill = bool(_re.search(r"\bskill\b", lower)) or ("技能" in original)
    power = bool(_re.search(r"\bpower\b", lower)) or ("能力" in original)
    curse = bool(_re.search(r"\bcurse\b", lower)) or ("诅咒" in original)
    status = bool(_re.search(r"\bstatus\b", lower)) or ("状态" in original)
    deltas["card_add_attack"] = attack
    deltas["card_add_skill"] = skill
    deltas["card_add_power"] = power
    deltas["card_add_curse"] = curse
    deltas["card_add_status"] = status
    # Count of card mentions (simplified): 1 if any type mentioned else 0;
    # boost to explicit number if "add N cards" pattern matches.
    add_count_explicit = _event_count_card_op(lower, original,
        [r"(?:add|gain|obtain|receive)\s*(a|an|one|\d+)\s*cards?",
         r"(?:add|gain|obtain|receive)\s*(a|an|one|\d+)\s*(?:attack|skill|power|curse|status)"],
        [r"加入(一|两|三|\d+)张", r"获得(一|两|三|\d+)张"])
    if add_count_explicit:
        deltas["card_add_count"] = add_count_explicit
    elif any([attack, skill, power, curse, status]):
        deltas["card_add_count"] = 1

    # Card ops
    deltas["card_remove_count"] = _event_count_card_op(lower, original,
        [r"remove\s*(a|an|one|\d+)\s*cards?", r"purge\s*(a|an|\d+)\s*cards?"],
        [r"移除(一|两|三|\d+)张", r"删除(一|两|三|\d+)张"])
    deltas["card_transform_count"] = _event_count_card_op(lower, original,
        [r"transform\s*(a|an|one|two|\d+)\s*cards?"],
        [r"变化(一|两|三|\d+)张", r"变形(一|两|三|\d+)张"])
    deltas["card_upgrade_count"] = _event_count_card_op(lower, original,
        [r"upgrade\s*(a|an|one|\d+)\s*cards?", r"smith\s*(a|an|\d+)\s*cards?"],
        [r"升级(一|两|三|\d+)张", r"锻造(一|两|三|\d+)张"])
    deltas["card_duplicate_count"] = _event_count_card_op(lower, original,
        [r"duplicate\s*(a|an|one|\d+)\s*cards?", r"copy\s*(a|an|\d+)\s*cards?"],
        [r"复制(一|两|三|\d+)张"])

    # Relic / potion gain
    if (_re.search(r"\b(gain|obtain|receive|get)\s+(a|an|one|\d+)?\s*relic\b", lower)
            or _re.search(r"获得.{0,6}遗物", original)):
        deltas["relic_gain"] = True
    if (_re.search(r"\b(gain|obtain|receive|get)\s+(a|an|one|\d+)?\s*potion\b", lower)
            or _re.search(r"获得.{0,6}药水", original)):
        deltas["potion_gain"] = True

    # Combat entry
    if (_re.search(r"\b(fight|enter\s*combat|start\s*combat|begin\s*battle)\b", lower)
            or _re.search(r"(战斗|戰鬥|进入战斗|進入戰鬥|开始战斗|開始戰鬥|遭遇敌人|遭遇敵人|我能打|打两个|打兩個)", original)):
        deltas["enter_combat"] = True

    return deltas


def _translate_event_options(event: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for opt in event.get("options") or []:
        if not isinstance(opt, dict):
            continue
        text = str(opt.get("text") or "")
        # Sim emits only `text` on option; split into title/description is
        # approximate (title = first line if any). Parser ok with whole text
        # in description.
        title = text.split("\n", 1)[0] if text else ""
        out.append({
            "index": int(opt.get("index") or 0),
            "label": text,
            "description": text,
            "is_enabled": not bool(opt.get("is_locked", False)),
            "is_chosen": bool(opt.get("is_chosen", False)),
            "is_proceed": bool(opt.get("is_proceed", False)),
            "effect_deltas": _extract_event_option_effect_deltas(title, text),
        })
    return out


def _translate_rewards_block(
    rewards: dict[str, Any],
    card_reward: dict[str, Any],
    treasure: dict[str, Any],
    relic_select: dict[str, Any],
) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    for item in rewards.get("items") or []:
        if not isinstance(item, dict):
            continue
        items.append({
            "index": int(item.get("index") or 0),
            "reward": {
                "type": str(item.get("type") or ""),
                "label": str(item.get("label") or ""),
                "reward_key": str(item.get("reward_key") or ""),
                "claimable": bool(item.get("claimable", True)),
            },
        })
    visible = bool(items or card_reward or treasure or relic_select)
    return {
        "visible": visible,
        "terminal_proceed_visible": bool(rewards.get("can_proceed", False)),
        "rewards": items,
    }


def _translate_rest_site_block(rest: dict[str, Any]) -> dict[str, Any]:
    options: list[dict[str, Any]] = []
    for opt in rest.get("options") or []:
        if not isinstance(opt, dict):
            continue
        options.append({
            "index": int(opt.get("index") or 0),
            "id": str(opt.get("id") or ""),
            "label": str(opt.get("name") or opt.get("id") or ""),
            "description": str(opt.get("description") or ""),
            "is_enabled": bool(opt.get("is_enabled", True)),
        })
    return {"visible": bool(options), "options": options, "can_proceed": bool(rest.get("can_proceed", False))}


def _translate_shop_block(shop: dict[str, Any]) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    for item in shop.get("items") or []:
        if not isinstance(item, dict):
            continue
        items.append({
            "index": int(item.get("index") or 0),
            "category": str(item.get("category") or ""),
            "cost": int(item.get("cost") or 0),
            "can_afford": bool(item.get("can_afford", False)),
            "is_stocked": bool(item.get("is_stocked", True)),
            "on_sale": bool(item.get("on_sale", False)),
            "name": str(item.get("name") or ""),
            "description": str(item.get("description") or ""),
            "card_id": str(item.get("card_id") or ""),
            "relic_id": str(item.get("relic_id") or ""),
            "potion_id": str(item.get("potion_id") or ""),
        })
    return {"visible": bool(items), "is_open": bool(shop.get("is_open", False)),
            "can_proceed": bool(shop.get("can_proceed", False)), "items": items}


def _translate_card_reward_sel_block(card_reward: dict[str, Any]) -> dict[str, Any]:
    choices = [_translate_card(c, pile="Reward") for c in (card_reward.get("cards") or [])]
    return {
        "visible": bool(choices),
        "can_skip": bool(card_reward.get("can_skip", True)),
        "choices": choices,
    }


def _translate_card_sel_block(
    card_select: dict[str, Any],
    hand_select: dict[str, Any],
    combat_card_sel: dict[str, Any] | None,
) -> dict[str, Any]:
    src = card_select or hand_select or combat_card_sel or {}
    if not src:
        return {"visible": False, "choices": []}
    cards = src.get("cards") or src.get("selectable_cards") or []
    return {
        "visible": True,
        "prompt": str(src.get("prompt") or ""),
        "min_select": int(src.get("min_select") or 0),
        "max_select": int(src.get("max_select") or 1),
        "can_confirm": bool(src.get("can_confirm", False)),
        "can_cancel": bool(src.get("can_cancel", False)),
        "choices": [_translate_card(c, pile="Select") for c in cards],
        "selected": [_translate_card(c, pile="Select") for c in (src.get("selected_cards") or [])],
    }
