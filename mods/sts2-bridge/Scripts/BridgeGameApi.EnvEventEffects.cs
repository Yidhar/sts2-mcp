using System.Collections.Generic;
using System.Diagnostics;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Text.RegularExpressions;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.MonsterMoves.Intents;
using MegaCrit.Sts2.Core.Models.Powers;
using MegaCrit.Sts2.Core.Nodes;

namespace Sts2McpBridge.Scripts;

internal static partial class BridgeGameApi
{
    internal sealed class EventOptionEffectDeltas
    {
        public int HpDelta { get; set; }                  // signed: lose → negative, gain/heal → positive
        public int MaxHpDelta { get; set; }
        public int GoldDelta { get; set; }
        public bool HealFull { get; set; }
        public int CardAddCount { get; set; }
        public bool CardAddAttack { get; set; }
        public bool CardAddSkill { get; set; }
        public bool CardAddPower { get; set; }
        public bool CardAddCurse { get; set; }
        public bool CardAddStatus { get; set; }
        public int CardRemoveCount { get; set; }
        public int CardTransformCount { get; set; }
        public int CardUpgradeCount { get; set; }
        public int CardDuplicateCount { get; set; }
        public bool RelicGain { get; set; }
        public bool PotionGain { get; set; }
        public bool EnterCombat { get; set; }

        public object ToPayload()
        {
            return new
            {
                hp_delta = HpDelta,
                max_hp_delta = MaxHpDelta,
                gold_delta = GoldDelta,
                heal_full = HealFull,
                card_add_count = CardAddCount,
                card_add_attack = CardAddAttack,
                card_add_skill = CardAddSkill,
                card_add_power = CardAddPower,
                card_add_curse = CardAddCurse,
                card_add_status = CardAddStatus,
                card_remove_count = CardRemoveCount,
                card_transform_count = CardTransformCount,
                card_upgrade_count = CardUpgradeCount,
                card_duplicate_count = CardDuplicateCount,
                relic_gain = RelicGain,
                potion_gain = PotionGain,
                enter_combat = EnterCombat
            };
        }
    }

    /// <summary>
    /// Extract structured effect signals from an event_option description so the
    /// policy can reason about choice outcomes without the text encoder. Pattern
    /// coverage spans EN and ZH; missing patterns degrade to 0 rather than lying.
    /// </summary>
    internal static EventOptionEffectDeltas ExtractEventOptionEffectDeltas(string? title, string? description)
    {
        var deltas = new EventOptionEffectDeltas();
        if (string.IsNullOrWhiteSpace(title) && string.IsNullOrWhiteSpace(description))
        {
            return deltas;
        }

        var combined = string.Join(" \n ",
            new[] { title ?? string.Empty, description ?? string.Empty }
            .Where(static s => !string.IsNullOrWhiteSpace(s)));
        var lower = combined.ToLowerInvariant();
        var chineseInput = combined;

        // ---- HP delta (signed) ----
        var hpLose = SumFirstMatch(lower, chineseInput,
            new[] { @"lose\s*(\d+)\s*hp", @"take\s*(\d+)\s*damage", @"you\s*take\s*(\d+)",
                    @"suffer\s*(\d+)\s*damage", @"receive\s*(\d+)\s*damage" },
            new[] { @"失去(\d+)点?(?:生命|hp)", @"受到(\d+)点?伤害", @"扣除?(\d+)点?(?:生命|hp)" });
        var hpGain = SumFirstMatch(lower, chineseInput,
            new[] { @"gain\s*(\d+)\s*hp", @"heal\s*(\d+)\s*hp?", @"restore\s*(\d+)\s*hp",
                    @"recover\s*(\d+)\s*hp" },
            new[] { @"(?:回复|恢复|治疗)(\d+)点?(?:生命|hp)", @"获得(\d+)点?(?:生命|hp)" });
        deltas.HpDelta = hpGain - hpLose;
        if (Regex.IsMatch(lower, @"\b(heal(ed)?\s*(to\s*)?full|fully\s*heal|restore\s*all\s*hp)\b") ||
            Regex.IsMatch(chineseInput, "(回满|满血|回复全部生命|治疗至满)"))
        {
            deltas.HealFull = true;
        }

        // ---- Max HP delta (signed) ----
        var maxHpGain = SumFirstMatch(lower, chineseInput,
            new[] { @"max\s*hp\s*\+\s*(\d+)", @"gain\s*(\d+)\s*max\s*hp",
                    @"(\d+)\s*max\s*hp", @"increase\s*max\s*hp\s*by\s*(\d+)" },
            new[] { @"最大生命(?:增加|提高|提升|上升)?\+?(\d+)", @"max\s*hp\s*\+?(\d+)" });
        var maxHpLose = SumFirstMatch(lower, chineseInput,
            new[] { @"max\s*hp\s*-\s*(\d+)", @"lose\s*(\d+)\s*max\s*hp",
                    @"decrease\s*max\s*hp\s*by\s*(\d+)" },
            new[] { @"最大生命(?:减少|降低|下降)(\d+)", @"失去(\d+)点?最大生命" });
        deltas.MaxHpDelta = maxHpGain - maxHpLose;

        // ---- Gold delta ----
        var goldGain = SumFirstMatch(lower, chineseInput,
            new[] { @"gain\s*(\d+)\s*gold", @"receive\s*(\d+)\s*gold",
                    @"(\d+)\s*gold", @"obtain\s*(\d+)\s*gold" },
            new[] { @"获得(\d+)点?金币", @"(\d+)点?金币" });
        var goldLose = SumFirstMatch(lower, chineseInput,
            new[] { @"lose\s*(\d+)\s*gold", @"pay\s*(\d+)\s*gold", @"spend\s*(\d+)\s*gold" },
            new[] { @"失去(\d+)点?金币", @"支付(\d+)点?金币", @"花费(\d+)点?金币" });
        deltas.GoldDelta = goldGain - goldLose;

        // ---- Card add (to deck) ----
        // Explicit count with type: "add a curse" / "obtain 2 skills" / "加入一张诅咒"
        deltas.CardAddCount += CountCardMentions(lower, chineseInput, out var types);
        if (types.Attack) deltas.CardAddAttack = true;
        if (types.Skill) deltas.CardAddSkill = true;
        if (types.Power) deltas.CardAddPower = true;
        if (types.Curse) deltas.CardAddCurse = true;
        if (types.Status) deltas.CardAddStatus = true;

        // ---- Card ops (remove / transform / upgrade / duplicate) ----
        deltas.CardRemoveCount = CountCardOp(lower, chineseInput,
            new[] { @"remove\s*(a|an|one|\d+)\s*cards?", @"purge\s*(a|an|\d+)\s*cards?" },
            new[] { @"移除(一|两|三|\d+)张", @"删除(一|两|三|\d+)张" });
        deltas.CardTransformCount = CountCardOp(lower, chineseInput,
            new[] { @"transform\s*(a|an|one|two|\d+)\s*cards?" },
            new[] { @"变化(一|两|三|\d+)张", @"变形(一|两|三|\d+)张" });
        deltas.CardUpgradeCount = CountCardOp(lower, chineseInput,
            new[] { @"upgrade\s*(a|an|one|\d+)\s*cards?", @"smith\s*(a|an|\d+)\s*cards?" },
            new[] { @"升级(一|两|三|\d+)张", @"锻造(一|两|三|\d+)张" });
        deltas.CardDuplicateCount = CountCardOp(lower, chineseInput,
            new[] { @"duplicate\s*(a|an|one|\d+)\s*cards?", @"copy\s*(a|an|\d+)\s*cards?" },
            new[] { @"复制(一|两|三|\d+)张" });

        // ---- Relic / potion gain ----
        if (Regex.IsMatch(lower, @"\b(gain|obtain|receive|get)\s+(a|an|one|\d+)?\s*relic\b") ||
            Regex.IsMatch(chineseInput, "获得.{0,6}遗物"))
        {
            deltas.RelicGain = true;
        }
        if (Regex.IsMatch(lower, @"\b(gain|obtain|receive|get)\s+(a|an|one|\d+)?\s*potion\b") ||
            Regex.IsMatch(chineseInput, "获得.{0,6}药水"))
        {
            deltas.PotionGain = true;
        }

        // ---- Enter combat ----
        if (Regex.IsMatch(lower, @"\b(fight|enter\s*combat|start\s*combat|begin\s*battle)\b") ||
            Regex.IsMatch(chineseInput, "(战斗|进入战斗|开始战斗|遭遇敌人)"))
        {
            deltas.EnterCombat = true;
        }

        return deltas;
    }

    private static int SumFirstMatch(string lower, string original, string[] englishPatterns, string[] chinesePatterns)
    {
        foreach (var pattern in englishPatterns)
        {
            var match = Regex.Match(lower, pattern, RegexOptions.IgnoreCase);
            if (match.Success && int.TryParse(match.Groups[1].Value, out var value))
            {
                return value;
            }
        }
        foreach (var pattern in chinesePatterns)
        {
            var match = Regex.Match(original, pattern);
            if (match.Success && int.TryParse(match.Groups[1].Value, out var value))
            {
                return value;
            }
        }
        return 0;
    }

    private static int CountCardOp(string lower, string original, string[] englishPatterns, string[] chinesePatterns)
    {
        foreach (var pattern in englishPatterns)
        {
            var match = Regex.Match(lower, pattern, RegexOptions.IgnoreCase);
            if (match.Success)
            {
                var raw = match.Groups[1].Value;
                return ParseCardCountToken(raw);
            }
        }
        foreach (var pattern in chinesePatterns)
        {
            var match = Regex.Match(original, pattern);
            if (match.Success)
            {
                return ParseCardCountToken(match.Groups[1].Value);
            }
        }
        return 0;
    }

    private static int ParseCardCountToken(string raw)
    {
        if (int.TryParse(raw, out var numeric))
        {
            return numeric;
        }
        return raw.ToLowerInvariant() switch
        {
            "a" or "an" or "one" or "一" => 1,
            "two" or "两" => 2,
            "three" or "三" => 3,
            _ => 1
        };
    }

    private readonly struct CardTypeFlags
    {
        public bool Attack { get; init; }
        public bool Skill { get; init; }
        public bool Power { get; init; }
        public bool Curse { get; init; }
        public bool Status { get; init; }
    }

    private static int CountCardMentions(string lower, string original, out CardTypeFlags types)
    {
        var attack = Regex.IsMatch(lower, @"\battack\b") || original.Contains("攻击");
        var skill = Regex.IsMatch(lower, @"\bskill\b") || original.Contains("技能");
        var power = Regex.IsMatch(lower, @"\bpower\b") || original.Contains("能力");
        var curse = Regex.IsMatch(lower, @"\bcurse\b") || original.Contains("诅咒");
        var status = Regex.IsMatch(lower, @"\bstatus\b") || original.Contains("状态");

        types = new CardTypeFlags
        {
            Attack = attack,
            Skill = skill,
            Power = power,
            Curse = curse,
            Status = status,
        };

        // Look for explicit card-add verbs; ignore mere mentions (e.g. "choose a card to remove").
        var enAddMatch = Regex.Match(lower,
            @"\b(add|obtain|receive|gain|get)\s+(a|an|one|two|three|\d+)\s+(attack|skill|power|curse|status|card)");
        if (enAddMatch.Success)
        {
            return ParseCardCountToken(enAddMatch.Groups[2].Value);
        }
        // Chinese: "获得一张/加入一张 XXX 牌"
        var zhAddMatch = Regex.Match(original,
            @"(?:获得|加入|得到|塞入)(一|两|三|\d+)张(?:攻击|技能|能力|诅咒|状态)?牌");
        if (zhAddMatch.Success)
        {
            return ParseCardCountToken(zhAddMatch.Groups[1].Value);
        }
        // Pure curse/status mention without "add" verb is still meaningful in events.
        if (curse || status)
        {
            var curseVerb = Regex.Match(original, @"(?:获得|得到|塞入|加入)\s*诅咒");
            if (curseVerb.Success || Regex.IsMatch(lower, @"\b(gain|obtain|receive|add)\s+a\s+curse\b"))
            {
                return 1;
            }
        }
        return 0;
    }

    private static bool IsEnvSelectionLikeAction(BridgeResolvedActionSelection action)
    {
        return action.Kind is "card_selection" or "deck_upgrade" or "character_select" or "run_mode_selection";
    }

    private static double RoundEnvNumber(double value)
    {
        return Math.Round(value, 6, MidpointRounding.AwayFromZero);
    }

    private static Player? GetPrimaryPlayer(BridgeWorldContext context)
    {
        if (context.RunState?.Players.Count > 0)
        {
            return context.RunState.Players[0];
        }

        if (context.CombatState?.Players.Count > 0)
        {
            return context.CombatState.Players[0];
        }

        return null;
    }

    private static int GetPrimaryPlayerCurrentHp(BridgeWorldContext context) => GetPrimaryPlayer(context)?.Creature?.CurrentHp ?? 0;
    private static int GetPrimaryPlayerMaxHp(BridgeWorldContext context) => GetPrimaryPlayer(context)?.Creature?.MaxHp ?? 0;
    private static int GetPrimaryPlayerGold(BridgeWorldContext context) => GetPrimaryPlayer(context)?.Gold ?? 0;
    private static int GetPrimaryPlayerRelicCount(BridgeWorldContext context) => GetPrimaryPlayer(context)?.Relics.Count ?? 0;
    private static int GetPrimaryPlayerDeckCount(BridgeWorldContext context) => GetPrimaryPlayer(context)?.Deck?.Cards.Count ?? 0;
}
