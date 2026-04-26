"""Reference implementation + tests of the C# event_option regex extractor.

The real extractor lives in BridgeGameApi.EnvHelpers.cs
(ExtractEventOptionEffectDeltas). This file mirrors the pattern set in Python
so we can validate pattern coverage against representative event descriptions
without a .NET test harness. If this file and the C# copy drift, correctness
is still guarded at runtime by in-game behaviour, but these tests catch
"patterns fail to match realistic phrasing" before deployment.
"""

from __future__ import annotations

import re
import unittest


# ---- Reference Python mirror of ExtractEventOptionEffectDeltas -------------

def _parse_card_count_token(raw: str) -> int:
    try:
        return int(raw)
    except ValueError:
        pass
    lut = {
        "a": 1, "an": 1, "one": 1, "一": 1,
        "two": 2, "两": 2,
        "three": 3, "三": 3,
    }
    return lut.get(raw.lower(), 1)


def _sum_first_match(lower: str, original: str, en_patterns, zh_patterns) -> int:
    for pat in en_patterns:
        m = re.search(pat, lower, re.IGNORECASE)
        if m:
            try:
                return int(m.group(1))
            except (ValueError, IndexError):
                continue
    for pat in zh_patterns:
        m = re.search(pat, original)
        if m:
            try:
                return int(m.group(1))
            except (ValueError, IndexError):
                continue
    return 0


def _count_card_op(lower, original, en_patterns, zh_patterns) -> int:
    for pat in en_patterns:
        m = re.search(pat, lower, re.IGNORECASE)
        if m:
            return _parse_card_count_token(m.group(1))
    for pat in zh_patterns:
        m = re.search(pat, original)
        if m:
            return _parse_card_count_token(m.group(1))
    return 0


def _extract(title: str, description: str) -> dict:
    combined = " \n ".join(filter(None, [title, description]))
    lower = combined.lower()
    zh = combined

    hp_lose = _sum_first_match(
        lower, zh,
        [r"lose\s*(\d+)\s*hp", r"take\s*(\d+)\s*damage",
         r"you\s*take\s*(\d+)", r"suffer\s*(\d+)\s*damage",
         r"receive\s*(\d+)\s*damage"],
        [r"失去(\d+)点?(?:生命|hp)", r"受到(\d+)点?伤害",
         r"扣除?(\d+)点?(?:生命|hp)"],
    )
    hp_gain = _sum_first_match(
        lower, zh,
        [r"gain\s*(\d+)\s*hp", r"heal\s*(\d+)\s*hp?",
         r"restore\s*(\d+)\s*hp", r"recover\s*(\d+)\s*hp"],
        [r"(?:回复|恢复|治疗)(\d+)点?(?:生命|hp)",
         r"获得(\d+)点?(?:生命|hp)"],
    )

    heal_full = bool(
        re.search(r"\b(heal(ed)?\s*(to\s*)?full|fully\s*heal|restore\s*all\s*hp)\b", lower)
    ) or bool(re.search(r"(回满|满血|回复全部生命|治疗至满)", zh))

    max_hp_gain = _sum_first_match(
        lower, zh,
        [r"max\s*hp\s*\+\s*(\d+)", r"gain\s*(\d+)\s*max\s*hp",
         r"(\d+)\s*max\s*hp", r"increase\s*max\s*hp\s*by\s*(\d+)"],
        [r"最大生命(?:增加|提高|提升|上升)?\+?(\d+)", r"max\s*hp\s*\+?(\d+)"],
    )
    max_hp_lose = _sum_first_match(
        lower, zh,
        [r"max\s*hp\s*-\s*(\d+)", r"lose\s*(\d+)\s*max\s*hp",
         r"decrease\s*max\s*hp\s*by\s*(\d+)"],
        [r"最大生命(?:减少|降低|下降)(\d+)", r"失去(\d+)点?最大生命"],
    )

    gold_gain = _sum_first_match(
        lower, zh,
        [r"gain\s*(\d+)\s*gold", r"receive\s*(\d+)\s*gold",
         r"(\d+)\s*gold", r"obtain\s*(\d+)\s*gold"],
        [r"获得(\d+)点?金币", r"(\d+)点?金币"],
    )
    gold_lose = _sum_first_match(
        lower, zh,
        [r"lose\s*(\d+)\s*gold", r"pay\s*(\d+)\s*gold", r"spend\s*(\d+)\s*gold"],
        [r"失去(\d+)点?金币", r"支付(\d+)点?金币", r"花费(\d+)点?金币"],
    )

    # Card add detection (mirrors CountCardMentions).
    attack = bool(re.search(r"\battack\b", lower)) or "攻击" in zh
    skill = bool(re.search(r"\bskill\b", lower)) or "技能" in zh
    power = bool(re.search(r"\bpower\b", lower)) or "能力" in zh
    curse = bool(re.search(r"\bcurse\b", lower)) or "诅咒" in zh
    status = bool(re.search(r"\bstatus\b", lower)) or "状态" in zh

    card_add_count = 0
    en_add = re.search(
        r"\b(add|obtain|receive|gain|get)\s+(a|an|one|two|three|\d+)\s+(attack|skill|power|curse|status|card)",
        lower,
    )
    if en_add:
        card_add_count = _parse_card_count_token(en_add.group(2))
    else:
        zh_add = re.search(
            r"(?:获得|加入|得到|塞入)(一|两|三|\d+)张(?:攻击|技能|能力|诅咒|状态)?牌",
            zh,
        )
        if zh_add:
            card_add_count = _parse_card_count_token(zh_add.group(1))
        elif curse or status:
            if re.search(r"(?:获得|得到|塞入|加入)\s*诅咒", zh) or \
               re.search(r"\b(gain|obtain|receive|add)\s+a\s+curse\b", lower):
                card_add_count = 1

    remove = _count_card_op(
        lower, zh,
        [r"remove\s*(a|an|one|\d+)\s*cards?", r"purge\s*(a|an|\d+)\s*cards?"],
        [r"移除(一|两|三|\d+)张", r"删除(一|两|三|\d+)张"],
    )
    transform = _count_card_op(
        lower, zh,
        [r"transform\s*(a|an|one|two|\d+)\s*cards?"],
        [r"变化(一|两|三|\d+)张", r"变形(一|两|三|\d+)张"],
    )
    upgrade = _count_card_op(
        lower, zh,
        [r"upgrade\s*(a|an|one|\d+)\s*cards?", r"smith\s*(a|an|\d+)\s*cards?"],
        [r"升级(一|两|三|\d+)张", r"锻造(一|两|三|\d+)张"],
    )
    duplicate = _count_card_op(
        lower, zh,
        [r"duplicate\s*(a|an|one|\d+)\s*cards?", r"copy\s*(a|an|\d+)\s*cards?"],
        [r"复制(一|两|三|\d+)张"],
    )

    relic_gain = bool(
        re.search(r"\b(gain|obtain|receive|get)\s+(a|an|one|\d+)?\s*relic\b", lower)
    ) or bool(re.search(r"获得.{0,6}遗物", zh))
    potion_gain = bool(
        re.search(r"\b(gain|obtain|receive|get)\s+(a|an|one|\d+)?\s*potion\b", lower)
    ) or bool(re.search(r"获得.{0,6}药水", zh))
    enter_combat = bool(
        re.search(r"\b(fight|enter\s*combat|start\s*combat|begin\s*battle)\b", lower)
    ) or bool(re.search(r"(战斗|进入战斗|开始战斗|遭遇敌人)", zh))

    return {
        "hp_delta": hp_gain - hp_lose,
        "max_hp_delta": max_hp_gain - max_hp_lose,
        "gold_delta": gold_gain - gold_lose,
        "heal_full": heal_full,
        "card_add_count": card_add_count,
        "card_add_attack": attack and card_add_count > 0,
        "card_add_skill": skill and card_add_count > 0,
        "card_add_power": power and card_add_count > 0,
        "card_add_curse": curse and card_add_count > 0,
        "card_add_status": status and card_add_count > 0,
        "card_remove_count": remove,
        "card_transform_count": transform,
        "card_upgrade_count": upgrade,
        "card_duplicate_count": duplicate,
        "relic_gain": relic_gain,
        "potion_gain": potion_gain,
        "enter_combat": enter_combat,
    }


# Match C# CountCardMentions intent: types are independently reported too.
def extract(title: str, description: str) -> dict:
    out = _extract(title, description)
    # C# sets the type flags even if count=0 when the text mentions the word.
    # We align here by recomputing type flags from raw mentions.
    lower = f"{title}\n{description}".lower()
    zh = f"{title}\n{description}"
    out["card_add_attack"] = bool(re.search(r"\battack\b", lower)) or "攻击" in zh
    out["card_add_skill"] = bool(re.search(r"\bskill\b", lower)) or "技能" in zh
    out["card_add_power"] = bool(re.search(r"\bpower\b", lower)) or "能力" in zh
    out["card_add_curse"] = bool(re.search(r"\bcurse\b", lower)) or "诅咒" in zh
    out["card_add_status"] = bool(re.search(r"\bstatus\b", lower)) or "状态" in zh
    return out


class EventOptionRegexReferenceTest(unittest.TestCase):
    def test_lose_hp_simple_en(self) -> None:
        d = extract("Accept", "Lose 10 HP.")
        self.assertEqual(d["hp_delta"], -10)

    def test_lose_hp_simple_zh(self) -> None:
        d = extract("接受", "失去10点生命。")
        self.assertEqual(d["hp_delta"], -10)

    def test_take_damage_en(self) -> None:
        d = extract("Fight", "You take 15 damage.")
        self.assertEqual(d["hp_delta"], -15)

    def test_gain_hp_en(self) -> None:
        d = extract("Rest", "Heal 20 HP.")
        self.assertEqual(d["hp_delta"], 20)

    def test_heal_full_flag(self) -> None:
        d = extract("Miracle", "Heal to full.")
        self.assertTrue(d["heal_full"])
        d2 = extract("奇迹", "回复全部生命")
        self.assertTrue(d2["heal_full"])

    def test_gold_gain_lose_en(self) -> None:
        d = extract("Bargain", "Gain 50 Gold, lose 10 HP.")
        self.assertEqual(d["gold_delta"], 50)
        self.assertEqual(d["hp_delta"], -10)

    def test_gold_zh(self) -> None:
        d = extract("谈判", "获得50金币，失去10点生命")
        self.assertEqual(d["gold_delta"], 50)
        self.assertEqual(d["hp_delta"], -10)

    def test_max_hp_gain_en(self) -> None:
        d = extract("Blessing", "Gain 5 Max HP.")
        self.assertEqual(d["max_hp_delta"], 5)

    def test_max_hp_gain_zh(self) -> None:
        d = extract("祝福", "最大生命增加5")
        self.assertEqual(d["max_hp_delta"], 5)

    def test_add_curse_en(self) -> None:
        d = extract("Pact", "Gain a curse.")
        self.assertGreaterEqual(d["card_add_count"], 1)
        self.assertTrue(d["card_add_curse"])

    def test_add_curse_zh(self) -> None:
        d = extract("契约", "获得一张诅咒牌")
        self.assertGreaterEqual(d["card_add_count"], 1)
        self.assertTrue(d["card_add_curse"])

    def test_remove_card_en(self) -> None:
        d = extract("Purge", "Remove a card from your deck.")
        self.assertEqual(d["card_remove_count"], 1)

    def test_remove_card_zh(self) -> None:
        d = extract("净化", "移除一张牌")
        self.assertEqual(d["card_remove_count"], 1)

    def test_transform_multi_zh(self) -> None:
        d = extract("变化之井", "变化两张牌")
        self.assertEqual(d["card_transform_count"], 2)

    def test_upgrade_en(self) -> None:
        d = extract("Smith", "Upgrade a card.")
        self.assertEqual(d["card_upgrade_count"], 1)

    def test_relic_gain_en(self) -> None:
        d = extract("Shrine", "Obtain a relic.")
        self.assertTrue(d["relic_gain"])

    def test_relic_gain_zh(self) -> None:
        d = extract("神龛", "获得一个遗物")
        self.assertTrue(d["relic_gain"])

    def test_potion_gain_zh(self) -> None:
        d = extract("药剂师", "获得一瓶药水")
        self.assertTrue(d["potion_gain"])

    def test_combo_option_zh(self) -> None:
        """失去 10 HP，变化 2 张牌"""
        d = extract("扭曲", "失去10点生命，变化两张牌")
        self.assertEqual(d["hp_delta"], -10)
        self.assertEqual(d["card_transform_count"], 2)

    def test_skip_option_empty(self) -> None:
        d = extract("离开", "Nothing happens.")
        self.assertEqual(d["hp_delta"], 0)
        self.assertEqual(d["gold_delta"], 0)
        self.assertEqual(d["card_add_count"], 0)
        self.assertEqual(d["card_remove_count"], 0)
        self.assertFalse(d["relic_gain"])

    def test_enter_combat_en(self) -> None:
        d = extract("Approach", "Fight the enemies.")
        self.assertTrue(d["enter_combat"])

    def test_enter_combat_zh(self) -> None:
        d = extract("迎战", "开始战斗")
        self.assertTrue(d["enter_combat"])


if __name__ == "__main__":
    unittest.main()
