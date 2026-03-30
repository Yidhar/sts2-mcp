using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Models;

namespace Sts2McpBridge.Scripts;

/// <summary>
/// Canonical Chinese semantic text builders for RL env payloads.
/// All RL text construction is centralized here.
/// Python never assembles primary semantic text.
/// </summary>
internal static partial class BridgeGameApi
{
    // -----------------------------------------------------------------------
    // Card
    // -----------------------------------------------------------------------

    private static string BuildCanonicalCardText(CardModel? card, Creature? previewTarget = null)
    {
        if (card is null) return "";
        var sb = new StringBuilder("卡牌｜");
        sb.Append(NormalizeSemanticText(TextOf(card.Title)));
        sb.Append("｜").Append(card.Type.ToString());

        if (card.EnergyCost.CostsX)
            sb.Append("｜能量X");
        else
            sb.Append("｜能量").Append(card.EnergyCost.GetResolved());

        var starCost = SafeGetStarCost(card);
        if (starCost is not null)
        {
            if (SafeHasStarCostX(card))
                sb.Append("｜星辉X");
            else
                sb.Append("｜星辉").Append(starCost);
        }

        sb.Append("｜目标").Append(TranslateTargetType(card.TargetType.ToString()));

        var effect = SafeGetCardEffectText(card, previewTarget);
        if (!string.IsNullOrWhiteSpace(effect))
            sb.Append("｜效果：").Append(NormalizeSemanticText(effect));

        return sb.ToString();
    }

    // -----------------------------------------------------------------------
    // Relic
    // -----------------------------------------------------------------------

    private static string BuildCanonicalRelicText(string? title, string? rarity, string? description)
    {
        var sb = new StringBuilder("遗物｜");
        sb.Append(NormalizeSemanticText(title ?? ""));
        if (!string.IsNullOrWhiteSpace(rarity))
            sb.Append("｜稀有度").Append(rarity);
        if (!string.IsNullOrWhiteSpace(description))
            sb.Append("｜效果：").Append(NormalizeSemanticText(description));
        return sb.ToString();
    }

    // -----------------------------------------------------------------------
    // Potion
    // -----------------------------------------------------------------------

    private static string BuildCanonicalPotionText(
        string? title, string? rarity, string? targetType, string? description)
    {
        var sb = new StringBuilder("药水｜");
        sb.Append(NormalizeSemanticText(title ?? ""));
        if (!string.IsNullOrWhiteSpace(rarity))
            sb.Append("｜稀有度").Append(rarity);
        if (!string.IsNullOrWhiteSpace(targetType))
            sb.Append("｜目标").Append(TranslateTargetType(targetType));
        if (!string.IsNullOrWhiteSpace(description))
            sb.Append("｜效果：").Append(NormalizeSemanticText(description));
        return sb.ToString();
    }

    // -----------------------------------------------------------------------
    // Shop item
    // -----------------------------------------------------------------------

    private static string BuildCanonicalShopItemText(
        string? itemKind, string? title, int? cost, string? semanticText)
    {
        var sb = new StringBuilder("商店商品｜");
        if (!string.IsNullOrWhiteSpace(itemKind))
            sb.Append(itemKind).Append("｜");
        sb.Append(NormalizeSemanticText(title ?? ""));
        if (cost is not null)
            sb.Append("｜价格").Append(cost);
        if (!string.IsNullOrWhiteSpace(semanticText))
            sb.Append("｜效果：").Append(NormalizeSemanticText(semanticText));
        return sb.ToString();
    }

    // -----------------------------------------------------------------------
    // Event option
    // -----------------------------------------------------------------------

    private static string BuildCanonicalEventOptionText(
        string? title, string? optionType, bool isProceed, string? text)
    {
        var sb = new StringBuilder("事件选项｜");
        sb.Append(NormalizeSemanticText(title ?? ""));
        if (!string.IsNullOrWhiteSpace(optionType))
            sb.Append("｜类型").Append(optionType);
        sb.Append(isProceed ? "｜继续" : "｜选择");
        if (!string.IsNullOrWhiteSpace(text))
            sb.Append("｜内容：").Append(NormalizeSemanticText(text));
        return sb.ToString();
    }

    // -----------------------------------------------------------------------
    // Upgrade preview
    // -----------------------------------------------------------------------

    private static string BuildCanonicalUpgradePreviewText(
        string? originalTitle, string? previewEffect)
    {
        var sb = new StringBuilder("升级｜");
        sb.Append(NormalizeSemanticText(originalTitle ?? ""));
        if (!string.IsNullOrWhiteSpace(previewEffect))
            sb.Append("｜升级后：").Append(NormalizeSemanticText(previewEffect));
        return sb.ToString();
    }

    // -----------------------------------------------------------------------
    // Action (canonical text by kind)
    // -----------------------------------------------------------------------

    private static string BuildCanonicalActionText(string kind, string actionId, JsonElement payload)
    {
        switch (kind)
        {
            case "play_card":
            {
                var title = TryGetNestedString(payload, "card", "title") ?? "";
                var effect = TryGetNestedString(payload, "card", "effect")
                             ?? TryGetNestedString(payload, "card", "description") ?? "";
                var target = TryGetNestedString(payload, "target_name") ?? "";
                var sb = new StringBuilder("动作｜出牌｜");
                sb.Append(NormalizeSemanticText(title));
                if (!string.IsNullOrWhiteSpace(target))
                    sb.Append("｜目标").Append(NormalizeSemanticText(target));
                if (!string.IsNullOrWhiteSpace(effect))
                    sb.Append("｜效果：").Append(NormalizeSemanticText(effect));
                return sb.ToString();
            }

            case "use_potion":
            {
                var title = TryGetNestedString(payload, "potion", "title") ?? "";
                var target = TryGetNestedString(payload, "target_name") ?? "";
                var sb = new StringBuilder("动作｜使用药水｜");
                sb.Append(NormalizeSemanticText(title));
                if (!string.IsNullOrWhiteSpace(target))
                    sb.Append("｜目标").Append(NormalizeSemanticText(target));
                return sb.ToString();
            }

            case "discard_potion":
                return $"动作｜丢弃药水｜{NormalizeSemanticText(TryGetNestedString(payload, "potion", "title") ?? "")}";

            case "combat":
                return "动作｜结束回合";

            case "event_option":
            {
                var title = TryGetNestedString(payload, "option", "title") ?? "";
                var optType = TryGetNestedString(payload, "option", "option_type") ?? "";
                var isProceed = TryGetNestedBool(payload, "option", "is_proceed") == true;
                var text = TryGetNestedString(payload, "option", "text") ?? "";
                return BuildCanonicalEventOptionText(title, optType, isProceed, text);
            }

            case "card_reward":
            {
                var sel = TryGetNestedString(payload, "selection_action") ?? "";
                if (sel.Contains("skip", StringComparison.OrdinalIgnoreCase))
                    return "动作｜跳过卡牌奖励";
                var title = TryGetNestedString(payload, "card", "title") ?? "";
                var effect = TryGetNestedString(payload, "card", "effect")
                             ?? TryGetNestedString(payload, "card", "description") ?? "";
                var sb = new StringBuilder("动作｜选择卡牌｜");
                sb.Append(NormalizeSemanticText(title));
                if (!string.IsNullOrWhiteSpace(effect))
                    sb.Append("｜效果：").Append(NormalizeSemanticText(effect));
                return sb.ToString();
            }

            case "map":
                return BuildCanonicalMapRouteActionText(payload);

            case "rest_site":
                return $"动作｜营火｜{NormalizeSemanticText(TryGetNestedString(payload, "option", "label") ?? TryGetNestedString(payload, "option", "semantic_action") ?? "")}";

            case "deck_upgrade":
            {
                var upgradeAction = TryGetNestedString(payload, "upgrade_action") ?? "";
                var selectionSemantics = TryGetNestedString(payload, "selection_semantics") ?? "upgrade";
                var selectionLabel = DescribeSelectionSemanticsLabel(selectionSemantics);
                if (upgradeAction.Contains("confirm", StringComparison.OrdinalIgnoreCase))
                    return $"动作｜确认{selectionLabel}选择";
                if (upgradeAction.Contains("cancel", StringComparison.OrdinalIgnoreCase))
                    return $"动作｜取消{selectionLabel}选择";
                if (upgradeAction.Contains("close", StringComparison.OrdinalIgnoreCase))
                    return $"动作｜关闭{selectionLabel}界面";
                var origTitle = TryGetNestedString(payload, "card", "title") ?? "";
                var previewEffect = TryGetNestedString(payload, "upgrade_preview", "effect")
                                    ?? TryGetNestedString(payload, "upgrade_preview", "description") ?? "";
                var sb = new StringBuilder("动作｜");
                sb.Append(selectionLabel).Append("候选｜").Append(NormalizeSemanticText(origTitle));
                if (!string.IsNullOrWhiteSpace(previewEffect))
                    sb.Append("｜升级后：").Append(NormalizeSemanticText(previewEffect));
                return sb.ToString();
            }

            case "shop":
            {
                var shopAction = TryGetNestedString(payload, "shop_action") ?? "";
                if (shopAction.Contains("leave", StringComparison.OrdinalIgnoreCase) ||
                    shopAction.Contains("back", StringComparison.OrdinalIgnoreCase))
                    return "动作｜离开商店";
                return $"动作｜购买｜{NormalizeSemanticText(TryGetNestedString(payload, "item", "title") ?? "")}";
            }

            case "proceed":
                return TryGetNestedBool(payload, "is_skip") == true ? "动作｜跳过" : "动作｜继续";

            case "reward":
            {
                var rewardType = TryGetNestedString(payload, "reward", "type") ?? "";
                var amount = TryGetNestedInt(payload, "reward", "amount");
                if (rewardType == "gold" && amount is not null) return $"动作｜领取奖励｜金币{amount}";
                if (rewardType == "card") return "动作｜领取奖励｜查看卡牌";
                return $"动作｜领取奖励｜{rewardType}";
            }

            case "treasure_relic":
                return $"动作｜获取遗物｜{NormalizeSemanticText(TryGetNestedString(payload, "relic", "title") ?? "")}";

            case "card_selection":
            {
                var sel = TryGetNestedString(payload, "selection_action") ?? "";
                var selectionSemantics = TryGetNestedString(payload, "selection_semantics") ?? "choose";
                var selectionLabel = DescribeSelectionSemanticsLabel(selectionSemantics);
                if (sel.Contains("confirm", StringComparison.OrdinalIgnoreCase)) return $"动作｜确认{selectionLabel}选择";
                if (sel.Contains("cancel", StringComparison.OrdinalIgnoreCase)) return $"动作｜取消{selectionLabel}选择";
                if (sel.Contains("close", StringComparison.OrdinalIgnoreCase)) return $"动作｜关闭{selectionLabel}界面";
                if (sel.Contains("skip", StringComparison.OrdinalIgnoreCase)) return $"动作｜跳过{selectionLabel}选择";
                var cardTitle = NormalizeSemanticText(TryGetNestedString(payload, "card", "title") ?? "");
                if (!string.IsNullOrWhiteSpace(cardTitle))
                    return $"动作｜{selectionLabel}候选｜{cardTitle}";
                return $"动作｜{selectionLabel}候选";
            }

            default:
                return $"动作｜{NormalizeSemanticText(TryGetNestedString(payload, "label") ?? actionId)}";
        }
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    private static string NormalizeSemanticText(string text)
    {
        if (string.IsNullOrWhiteSpace(text)) return "";
        // Replace [E] energy icon BEFORE stripping other BBCode tags
        var result = text.Replace("[E]", "能量");
        result = Regex.Replace(result, @"\[/?[^\]]+\]", "");
        result = Regex.Replace(result, @"\s+", " ").Trim();
        return result;
    }

    private static string BuildCanonicalMapRouteActionText(JsonElement payload)
    {
        var pointType = TranslateMapPointType(
            TryGetNestedString(payload, "point_type_norm") ??
            TryGetNestedString(payload, "point_type"));
        var col = TryGetNestedInt(payload, "coord", "col");
        var row = TryGetNestedInt(payload, "coord", "row");
        var countShop = TryGetNestedInt(payload, "route_summary", "count_shop");
        var countRest = TryGetNestedInt(payload, "route_summary", "count_rest_site");
        var countElite = TryGetNestedInt(payload, "route_summary", "count_elite");
        var countQuestion = TryGetNestedInt(payload, "route_summary", "count_question_mark");
        var nextShop = TryGetNestedInt(payload, "route_summary", "next_shop_steps");
        var nextRest = TryGetNestedInt(payload, "route_summary", "next_rest_steps");
        var nextElite = TryGetNestedInt(payload, "route_summary", "next_elite_steps");
        var nextQuestion = TryGetNestedInt(payload, "route_summary", "next_question_mark_steps");
        var forcedSteps = TryGetNestedInt(payload, "route_summary", "forced_path_steps_before_branch");

        var sb = new StringBuilder("动作｜前往｜");
        sb.Append(pointType);
        if (col is not null && row is not null)
        {
            sb.Append($"｜坐标({col},{row})");
        }

        if (countShop is > 0 || countRest is > 0 || countElite is > 0 || countQuestion is > 0)
        {
            sb.Append("｜未来");
            if (countShop is > 0)
            {
                sb.Append($"｜商店{countShop}");
            }

            if (countRest is > 0)
            {
                sb.Append($"｜营火{countRest}");
            }

            if (countElite is > 0)
            {
                sb.Append($"｜精英{countElite}");
            }

            if (countQuestion is > 0)
            {
                sb.Append($"｜问号{countQuestion}");
            }
        }

        if (nextRest is not null)
        {
            sb.Append($"｜最近营火{nextRest}步");
        }

        if (nextShop is not null)
        {
            sb.Append($"｜最近商店{nextShop}步");
        }

        if (nextElite is not null)
        {
            sb.Append($"｜最近精英{nextElite}步");
        }

        if (nextQuestion is not null)
        {
            sb.Append($"｜最近问号{nextQuestion}步");
        }

        if (forcedSteps is > 1)
        {
            sb.Append($"｜强制路径{forcedSteps}步");
        }

        return sb.ToString();
    }

    private static string TranslateMapPointType(string? pointType)
    {
        return NormalizeEnvMapPointType(pointType) switch
        {
            "Monster" => "普通战斗",
            "Elite" => "精英",
            "Boss" => "Boss",
            "Event" => "事件",
            "QuestionMark" => "问号",
            "RestSite" => "营火",
            "Shop" => "商店",
            "Treasure" => "宝箱",
            _ => NormalizeSemanticText(pointType ?? "")
        };
    }

    private static string DescribeSelectionSemanticsLabel(string? semantics)
    {
        return (semantics ?? "").Trim().ToLowerInvariant() switch
        {
            "remove" => "移除",
            "transform" => "变化",
            "upgrade" => "升级",
            "discard" => "弃牌",
            "retain" => "保留",
            "bundle" => "组合",
            "choose" => "选择",
            _ => "选择"
        };
    }

    private static string TranslateTargetType(string? targetType)
    {
        return (targetType ?? "") switch
        {
            "AnyEnemy" or "SingleEnemy" => "单体敌人",
            "AllEnemies" or "AllEnemy" => "全体敌人",
            "Self" or "Player" => "自身",
            "AnyPlayer" or "AnyAlly" => "友方",
            "AllAllies" or "AllAlly" => "全体友方",
            "RandomEnemy" => "随机敌人",
            "None" or "" => "无",
            var s => s
        };
    }

    private static string SafeGetCardEffectText(CardModel card, Creature? target)
    {
        // Try effect preview via reflection
        try
        {
            var resolvedTarget = target ?? GetHiddenFieldValue(card, "_currentTarget") as Creature;
            var method = FindMethod(card.GetType(), "GetEffectPreviewSummary", 1);
            if (method is not null)
            {
                var result = method.Invoke(card, new object?[] { resolvedTarget });
                if (result is string s && !string.IsNullOrWhiteSpace(s)) return s;
            }
        }
        catch { /* best effort */ }

        // Fall back to description
        try
        {
            var desc = DescribeText(card.Description, card);
            if (!string.IsNullOrWhiteSpace(desc)) return desc;
        }
        catch { /* best effort */ }

        return "";
    }

    private static int? SafeGetStarCost(CardModel card)
    {
        try
        {
            var prop = card.GetType().GetProperty("CurrentStarCost");
            if (prop?.GetValue(card) is int v && v >= 0) return v;
        }
        catch { /* best effort */ }
        return null;
    }

    private static bool SafeHasStarCostX(CardModel card)
    {
        try
        {
            var prop = card.GetType().GetProperty("HasStarCostX");
            if (prop?.GetValue(card) is bool v) return v;
        }
        catch { /* best effort */ }
        return false;
    }
}
