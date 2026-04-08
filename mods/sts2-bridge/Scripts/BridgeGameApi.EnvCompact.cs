using System.Collections.Generic;
using System.Linq;
using System.Text.Json;

namespace Sts2McpBridge.Scripts;

internal static partial class BridgeGameApi
{
    private static Dictionary<string, object?> CloneDictionary(Dictionary<string, object?> source)
    {
        var clone = new Dictionary<string, object?>(source.Count, StringComparer.Ordinal);
        foreach (var pair in source)
        {
            clone[pair.Key] = pair.Value;
        }

        return clone;
    }

    private static object? CompactCardPayload(JsonElement? element, string? cardRef = null)
    {
        if (element is null || element.Value.ValueKind is JsonValueKind.Undefined or JsonValueKind.Null)
        {
            return null;
        }

        var payload = new Dictionary<string, object?>(StringComparer.Ordinal);
        if (!string.IsNullOrWhiteSpace(cardRef))
        {
            payload["ref"] = cardRef;
        }

        var cardId = TryGetNestedString(element.Value, "id");
        if (!string.IsNullOrWhiteSpace(cardId))
        {
            payload["id"] = cardId;
        }

        var upgradeLevel = TryGetNestedInt(element.Value, "current_upgrade_level");
        if (upgradeLevel.HasValue)
        {
            payload["upgrade_level"] = upgradeLevel.Value;
        }

        var title = TryGetNestedString(element.Value, "title");
        if (!string.IsNullOrWhiteSpace(title))
        {
            payload["title"] = title;
        }

        var cost = TryGetNestedInt(element.Value, "resolved_energy_cost");
        if (cost.HasValue)
        {
            payload["cost"] = cost.Value;
        }

        var starCost = TryGetNestedInt(element.Value, "current_star_cost");
        if (starCost.HasValue && starCost.Value >= 0)
        {
            payload["star"] = starCost.Value;
        }

        if (TryGetNestedBool(element.Value, "costs_x") == true)
        {
            payload["x_cost"] = true;
        }

        if (TryGetNestedBool(element.Value, "has_star_cost_x") == true)
        {
            payload["star_x"] = true;
        }

        var type = TryGetNestedString(element.Value, "type");
        if (!string.IsNullOrWhiteSpace(type))
        {
            payload["type"] = type;
        }

        var target = TryGetNestedString(element.Value, "target_type");
        if (!string.IsNullOrWhiteSpace(target))
        {
            payload["target"] = target;
        }

        var effect = TryGetNestedString(element.Value, "effect_preview", "summary");
        var description = TryGetNestedString(element.Value, "description");
        if (!string.IsNullOrWhiteSpace(effect))
        {
            payload["effect"] = effect;
        }
        else if (!string.IsNullOrWhiteSpace(description))
        {
            payload["description"] = description;
        }

        // Canonical Chinese semantic text — delegate to shared builder in EnvText.cs
        // to avoid dual-source drift. Build from the same fields the compact payload exposes.
        {
            var ct = new System.Text.StringBuilder("卡牌｜");
            ct.Append(NormalizeSemanticText(title ?? ""));
            ct.Append("｜").Append(TryGetNestedString(element.Value, "type") ?? "");
            if (TryGetNestedBool(element.Value, "costs_x") == true)
                ct.Append("｜能量X");
            else if (cost.HasValue)
                ct.Append("｜能量").Append(cost.Value);
            if (starCost is { } sc)
                ct.Append(TryGetNestedBool(element.Value, "has_star_cost_x") == true ? "｜星辉X" : $"｜星辉{sc}");
            ct.Append("｜目标").Append(TranslateTargetType(TryGetNestedString(element.Value, "target_type")));
            var effectText = effect ?? description ?? "";
            if (!string.IsNullOrWhiteSpace(effectText))
                ct.Append("｜效果：").Append(NormalizeSemanticText(effectText));
            payload["canonical_text"] = ct.ToString();
        }

        AppendCompactPreviewFields(payload, element.Value);

        return payload;
    }

    private static object? CompactRewardPayload(JsonElement? element)
    {
        if (element is null || element.Value.ValueKind is JsonValueKind.Undefined or JsonValueKind.Null)
        {
            return null;
        }

        return new
        {
            type = TryGetNestedString(element.Value, "reward_type"),
            amount = TryGetNestedInt(element.Value, "amount"),
            relic = CompactRelicPayload(TryGetNestedElement(element.Value, "relic")),
            potion = CompactPotionPayload(TryGetNestedElement(element.Value, "potion")),
            card_count = TryGetNestedArrayLength(element.Value, "cards")
        };
    }

    private static object? CompactPotionPayload(JsonElement? element)
    {
        if (element is null || element.Value.ValueKind is JsonValueKind.Undefined or JsonValueKind.Null)
        {
            return null;
        }

        var payload = new Dictionary<string, object?>(StringComparer.Ordinal);
        var potionId = TryGetNestedString(element.Value, "id");
        var title = TryGetNestedString(element.Value, "title");
        var rarity = TryGetNestedString(element.Value, "rarity");
        var target = TryGetNestedString(element.Value, "target_type");
        var desc = TryGetNestedString(element.Value, "description");
        if (!string.IsNullOrWhiteSpace(potionId))
        {
            payload["id"] = potionId;
        }

        if (!string.IsNullOrWhiteSpace(title))
        {
            payload["title"] = title;
        }

        if (!string.IsNullOrWhiteSpace(rarity))
        {
            payload["rarity"] = rarity;
        }

        if (!string.IsNullOrWhiteSpace(target))
        {
            payload["target"] = target;
        }

        AppendCompactPreviewValue(payload, "damage", TryExtractEnvMetric(element.Value, "damage"));
        AppendCompactPreviewValue(payload, "block", TryExtractEnvMetric(element.Value, "block"));
        AppendCompactPreviewValue(payload, "draw", TryExtractEnvMetric(element.Value, "draw"));
        AppendCompactPreviewValue(payload, "weak", TryExtractEnvMetric(element.Value, "weak"));
        AppendCompactPreviewValue(payload, "vulnerable", TryExtractEnvMetric(element.Value, "vulnerable"));
        AppendCompactPreviewValue(payload, "heal", TryExtractEnvMetric(element.Value, "heal"));
        AppendCompactPreviewValue(payload, "hp_loss", TryExtractEnvMetric(element.Value, "hp_loss"));
        AppendCompactPreviewValue(payload, "strength", TryExtractEnvMetric(element.Value, "strength"));
        AppendCompactPreviewValue(payload, "dexterity", TryExtractEnvMetric(element.Value, "dexterity"));
        AppendCompactPreviewValue(payload, "summon", TryExtractEnvMetric(element.Value, "summon"));
        payload["canonical_text"] = BuildCanonicalPotionText(title, rarity, target, desc);
        return payload;
    }

    private static object? CompactRelicPayload(JsonElement? element)
    {
        if (element is null || element.Value.ValueKind is JsonValueKind.Undefined or JsonValueKind.Null)
        {
            return null;
        }

        var relicId = TryGetNestedString(element.Value, "id");
        var title = TryGetNestedString(element.Value, "title");
        var rarity = TryGetNestedString(element.Value, "rarity");
        var desc = TryGetNestedString(element.Value, "description");
        var payload = new Dictionary<string, object?>(StringComparer.Ordinal);
        if (!string.IsNullOrWhiteSpace(relicId))
        {
            payload["id"] = relicId;
        }

        if (!string.IsNullOrWhiteSpace(title))
        {
            payload["title"] = title;
        }

        if (!string.IsNullOrWhiteSpace(rarity))
        {
            payload["rarity"] = rarity;
        }

        payload["canonical_text"] = BuildCanonicalRelicText(title, rarity, desc);
        return payload;
    }

    private static object? CompactCharacterPayload(JsonElement? element)
    {
        if (element is null || element.Value.ValueKind is JsonValueKind.Undefined or JsonValueKind.Null)
        {
            return null;
        }

        return new
        {
            id = TryGetNestedString(element.Value, "id"),
            title = TryGetNestedString(element.Value, "title")
        };
    }

    private static object? CompactShopItemPayload(JsonElement? element)
    {
        if (element is null || element.Value.ValueKind is JsonValueKind.Undefined or JsonValueKind.Null)
        {
            return null;
        }

        var itemKind = TryGetNestedString(element.Value, "item_kind");
        var title = TryGetNestedString(element.Value, "title");
        var cost = TryGetNestedInt(element.Value, "cost");
        var desc = TryGetNestedString(element.Value, "description") ?? "";
        return new
        {
            kind = itemKind,
            title,
            cost,
            affordable = TryGetNestedBool(element.Value, "is_affordable"),
            card = CompactCardPayload(TryGetNestedElement(element.Value, "card")),
            relic = CompactRelicPayload(TryGetNestedElement(element.Value, "relic")),
            potion = CompactPotionPayload(TryGetNestedElement(element.Value, "potion")),
            canonical_text = BuildCanonicalShopItemText(itemKind, title, cost, desc)
        };
    }

    private static object? CompactRestSiteOptionPayload(JsonElement? element)
    {
        if (element is null || element.Value.ValueKind is JsonValueKind.Undefined or JsonValueKind.Null)
        {
            return null;
        }

        return new
        {
            option_id = TryGetNestedString(element.Value, "option_id"),
            option_type = TryGetNestedString(element.Value, "option_type"),
            title = TryGetNestedString(element.Value, "title"),
            enabled = TryGetNestedBool(element.Value, "is_enabled")
        };
    }

    private static object? CompactCoordPayload(JsonElement? element)
    {
        if (element is null || element.Value.ValueKind is JsonValueKind.Undefined or JsonValueKind.Null)
        {
            return null;
        }

        return new
        {
            col = TryGetNestedInt(element.Value, "col"),
            row = TryGetNestedInt(element.Value, "row")
        };
    }

    private static object? CompactMapRouteSummaryPayload(JsonElement? element)
    {
        if (element is null || element.Value.ValueKind is JsonValueKind.Undefined or JsonValueKind.Null)
        {
            return null;
        }

        return new
        {
            reachable_node_count = TryGetNestedInt(element.Value, "reachable_node_count"),
            max_depth = TryGetNestedInt(element.Value, "max_depth"),
            direct_child_count = TryGetNestedInt(element.Value, "direct_child_count"),
            forced_path_steps_before_branch = TryGetNestedInt(element.Value, "forced_path_steps_before_branch"),
            count_monster = TryGetNestedInt(element.Value, "count_monster"),
            count_elite = TryGetNestedInt(element.Value, "count_elite"),
            count_boss = TryGetNestedInt(element.Value, "count_boss"),
            count_event = TryGetNestedInt(element.Value, "count_event"),
            count_question_mark = TryGetNestedInt(element.Value, "count_question_mark"),
            count_rest_site = TryGetNestedInt(element.Value, "count_rest_site"),
            count_shop = TryGetNestedInt(element.Value, "count_shop"),
            count_treasure = TryGetNestedInt(element.Value, "count_treasure"),
            next_elite_steps = TryGetNestedInt(element.Value, "next_elite_steps"),
            next_rest_steps = TryGetNestedInt(element.Value, "next_rest_steps"),
            next_shop_steps = TryGetNestedInt(element.Value, "next_shop_steps"),
            next_event_steps = TryGetNestedInt(element.Value, "next_event_steps"),
            next_question_mark_steps = TryGetNestedInt(element.Value, "next_question_mark_steps"),
            next_treasure_steps = TryGetNestedInt(element.Value, "next_treasure_steps"),
            next_boss_steps = TryGetNestedInt(element.Value, "next_boss_steps"),
            can_reach_rest_site_before_elite = TryGetNestedBool(element.Value, "can_reach_rest_site_before_elite"),
            can_reach_elite_then_rest_site = TryGetNestedBool(element.Value, "can_reach_elite_then_rest_site")
        };
    }

    private static object[] CompactMapRouteNodesPayload(JsonElement? element)
    {
        if (element is null || element.Value.ValueKind is JsonValueKind.Undefined or JsonValueKind.Null)
        {
            return Array.Empty<object>();
        }

        if (!element.Value.TryGetProperty("nodes", out var nodes) || nodes.ValueKind != JsonValueKind.Array)
        {
            return Array.Empty<object>();
        }

        return nodes
            .EnumerateArray()
            .Select(node => (object)new
            {
                coord = CompactCoordPayload(TryGetNestedElement(node, "coord")),
                point_type = TryGetNestedString(node, "point_type"),
                depth = TryGetNestedInt(node, "depth"),
                child_count = TryGetNestedInt(node, "child_count"),
                is_leaf = TryGetNestedBool(node, "is_leaf")
            })
            .ToArray();
    }

    private static JsonElement? TryGetNestedElement(JsonElement element, params string[] path)
    {
        var current = element;
        foreach (var segment in path)
        {
            if (current.ValueKind != JsonValueKind.Object || !current.TryGetProperty(segment, out current))
            {
                return null;
            }
        }

        return current;
    }

    private static string? TryGetNestedString(JsonElement element, params string[] path)
    {
        var nested = TryGetNestedElement(element, path);
        return nested is { } value && value.ValueKind == JsonValueKind.String
            ? value.GetString()
            : null;
    }

    private static int? TryGetNestedInt(JsonElement element, params string[] path)
    {
        var nested = TryGetNestedElement(element, path);
        if (nested is null)
        {
            return null;
        }

        return nested.Value.ValueKind == JsonValueKind.Number && nested.Value.TryGetInt32(out var value)
            ? value
            : null;
    }

    private static bool? TryGetNestedBool(JsonElement element, params string[] path)
    {
        var nested = TryGetNestedElement(element, path);
        return nested is { } value && value.ValueKind is JsonValueKind.True or JsonValueKind.False
            ? value.GetBoolean()
            : null;
    }

    private static int? TryGetNestedArrayLength(JsonElement element, params string[] path)
    {
        var nested = TryGetNestedElement(element, path);
        return nested is { } value && value.ValueKind == JsonValueKind.Array
            ? value.GetArrayLength()
            : null;
    }

    private static int? TryGetFirstIntentTotalDamage(JsonElement? intentElement)
    {
        if (intentElement is null || intentElement.Value.ValueKind != JsonValueKind.Object)
        {
            return null;
        }

        var intents = TryGetNestedElement(intentElement.Value, "intents");
        if (intents is null || intents.Value.ValueKind != JsonValueKind.Array || intents.Value.GetArrayLength() <= 0)
        {
            return null;
        }

        var first = intents.Value[0];
        return TryGetNestedInt(first, "total_damage");
    }

    private static int? TryGetFirstIntentRepeats(JsonElement? intentElement)
    {
        if (intentElement is null || intentElement.Value.ValueKind != JsonValueKind.Object)
        {
            return null;
        }

        var intents = TryGetNestedElement(intentElement.Value, "intents");
        if (intents is null || intents.Value.ValueKind != JsonValueKind.Array || intents.Value.GetArrayLength() <= 0)
        {
            return null;
        }

        var first = intents.Value[0];
        return TryGetNestedInt(first, "repeats");
    }

    private static string? TryGetFirstIntentString(JsonElement? intentElement, params string[] path)
    {
        if (intentElement is null || intentElement.Value.ValueKind != JsonValueKind.Object)
        {
            return null;
        }

        var intents = TryGetNestedElement(intentElement.Value, "intents");
        if (intents is null || intents.Value.ValueKind != JsonValueKind.Array || intents.Value.GetArrayLength() <= 0)
        {
            return null;
        }

        var first = intents.Value[0];
        return TryGetNestedString(first, path);
    }

    private static void AppendCompactPreviewFields(Dictionary<string, object?> payload, JsonElement element)
    {
        AppendCompactPreviewValue(payload, "damage", TryGetNestedInt(element, "effect_preview", "total_damage") ?? TryExtractEnvMetric(element, "damage"));
        AppendCompactPreviewValue(payload, "block", TryGetNestedInt(element, "effect_preview", "total_block") ?? TryExtractEnvMetric(element, "block"));
        AppendCompactPreviewValue(payload, "draw", TryGetNestedInt(element, "effect_preview", "draw") ?? TryExtractEnvMetric(element, "draw"));
        AppendCompactPreviewValue(payload, "weak", TryGetNestedInt(element, "effect_preview", "weak") ?? TryExtractEnvMetric(element, "weak"));
        AppendCompactPreviewValue(payload, "vulnerable", TryGetNestedInt(element, "effect_preview", "vulnerable") ?? TryExtractEnvMetric(element, "vulnerable"));
        AppendCompactPreviewValue(payload, "heal", TryGetNestedInt(element, "effect_preview", "heal") ?? TryExtractEnvMetric(element, "heal"));
        AppendCompactPreviewValue(payload, "hp_loss", TryGetNestedInt(element, "effect_preview", "hp_loss") ?? TryExtractEnvMetric(element, "hp_loss"));
        AppendCompactPreviewValue(payload, "strength", TryGetNestedInt(element, "effect_preview", "strength") ?? TryExtractEnvMetric(element, "strength"));
        AppendCompactPreviewValue(payload, "dexterity", TryGetNestedInt(element, "effect_preview", "dexterity") ?? TryExtractEnvMetric(element, "dexterity"));
        AppendCompactPreviewValue(payload, "summon", TryGetNestedInt(element, "effect_preview", "summon") ?? TryExtractEnvMetric(element, "summon"));
    }

    private static void AppendCompactPreviewValue(Dictionary<string, object?> payload, string key, int value)
    {
        if (value != 0)
        {
            payload[key] = value;
        }
    }
}
