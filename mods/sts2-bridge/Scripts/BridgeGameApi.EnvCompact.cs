using System;
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

        // Runtime per-card identity must survive env compaction so a legal
        // candidate can be joined to the same concrete card in the world DTO.
        // Keep the known transport aliases while retail payload names vary.
        foreach (var identityKey in new[]
                 {
                     "instance_uuid",
                     "combat_uuid",
                     "uuid",
                     "uid",
                     "instance_id",
                     "card_instance_id"
                 })
        {
            var identityValue = TryGetNestedString(element.Value, identityKey);
            if (!string.IsNullOrWhiteSpace(identityValue))
            {
                payload[identityKey] = identityValue;
            }
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

        AppendCompactCardKeywords(payload, element.Value);

        var type = TryGetNestedString(element.Value, "type");
        if (!string.IsNullOrWhiteSpace(type))
        {
            payload["type"] = type;
        }

        var target = TryGetNestedString(element.Value, "target_type");
        if (!string.IsNullOrWhiteSpace(target))
        {
            // Keep the card's declared target category separate from the legal
            // candidate's concrete `target` creature object.
            payload["target_type"] = target;
        }

        var description = TryGetNestedString(element.Value, "description");
        if (!string.IsNullOrWhiteSpace(description))
        {
            payload["description"] = description;
        }

        // Canonical Chinese semantic text - delegate to shared builder in EnvText.cs
        // to avoid dual-source drift. Build from the same fields the compact payload exposes.
        {
            var ct = new System.Text.StringBuilder("\u5361\u724c\uff5c");
            ct.Append(NormalizeSemanticText(title ?? ""));
            ct.Append("\uff5c").Append(TryGetNestedString(element.Value, "type") ?? "");
            if (TryGetNestedBool(element.Value, "costs_x") == true)
                ct.Append("\uff5c\u80fd\u91cfX");
            else if (cost.HasValue)
                ct.Append("\uff5c\u80fd\u91cf").Append(cost.Value);
            if (starCost is { } sc)
                ct.Append(TryGetNestedBool(element.Value, "has_star_cost_x") == true ? "\uff5c\u661f\u8f89X" : $"\uff5c\u661f\u8f89{sc}");
            ct.Append("\uff5c\u76ee\u6807").Append(TranslateTargetType(TryGetNestedString(element.Value, "target_type")));
            if (!string.IsNullOrWhiteSpace(description))
                ct.Append("\uff5c\u6548\u679c\uff1a").Append(NormalizeSemanticText(description));
            payload["canonical_text"] = ct.ToString();
        }

        AppendCompactCardModifiers(payload, element.Value, "afflictions");
        AppendCompactCardModifiers(payload, element.Value, "enchantments");

        var typedSelection = CompactSelectionPayload(TryGetNestedElement(element.Value, "selection"));
        if (typedSelection is not null)
        {
            payload["selection"] = typedSelection;
        }

        return payload;
    }

    private static object? CompactSelectionPayload(JsonElement? element)
    {
        if (element is null || element.Value.ValueKind != JsonValueKind.Object)
        {
            return null;
        }

        var payload = new Dictionary<string, object?>(StringComparer.Ordinal);
        foreach (var key in new[]
                 {
                     "screen_type",
                     "operation_type",
                     "source",
                     "source_zone",
                     "destination_zone",
                     "modifier_id",
                     "confidence"
                 })
        {
            var value = TryGetNestedString(element.Value, key);
            if (!string.IsNullOrWhiteSpace(value))
            {
                payload[key] = value;
            }
        }

        foreach (var key in new[] { "min_count", "max_count" })
        {
            var value = TryGetNestedInt(element.Value, key);
            if (value.HasValue)
            {
                payload[key] = value.Value;
            }
        }

        var required = TryGetNestedBool(element.Value, "selection_required");
        if (required.HasValue)
        {
            payload["selection_required"] = required.Value;
        }

        return payload.Count > 0 ? payload : null;
    }

    private static void AppendCompactCardKeywords(Dictionary<string, object?> payload, JsonElement element)
    {
        var keywords = TryGetNestedElement(element, "keywords");
        if (keywords is not null && keywords.Value.ValueKind == JsonValueKind.Array)
        {
            var compactKeywords = keywords.Value.EnumerateArray()
                .Select(static item => item.ValueKind == JsonValueKind.String ? item.GetString() : item.ToString())
                .Where(static item => !string.IsNullOrWhiteSpace(item))
                .Distinct(StringComparer.Ordinal)
                .OrderBy(static item => item, StringComparer.Ordinal)
                .ToArray();
            if (compactKeywords.Length > 0)
            {
                payload["keywords"] = compactKeywords;
            }
        }

    }

    private static void AppendCompactCardModifiers(Dictionary<string, object?> payload, JsonElement element, string fieldName)
    {
        var modifiers = TryGetNestedElement(element, fieldName);
        if (modifiers is null || modifiers.Value.ValueKind != JsonValueKind.Array)
        {
            return;
        }

        var compact = new List<object?>();
        foreach (var modifier in modifiers.Value.EnumerateArray())
        {
            var title = TryGetNestedString(modifier, "title");
            var id = TryGetNestedString(modifier, "id");
            var type = TryGetNestedString(modifier, "type");
            var description = TryGetNestedString(modifier, "description");
            var amount = TryGetNestedDecimal(modifier, "amount");
            if (string.IsNullOrWhiteSpace(title) && string.IsNullOrWhiteSpace(id) && string.IsNullOrWhiteSpace(type))
            {
                continue;
            }
            compact.Add(new
            {
                id,
                title,
                type,
                description,
                amount,
                status = TryGetNestedString(modifier, "status"),
                enabled = TryGetNestedBool(modifier, "enabled"),
                is_debuff = TryGetNestedBool(modifier, "is_debuff"),
                is_buff = TryGetNestedBool(modifier, "is_buff")
            });
            if (compact.Count >= 8)
            {
                break;
            }
        }

        if (compact.Count > 0)
        {
            payload[fieldName] = compact.ToArray();
        }
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
            // Same "target" -> "target_type" rename applies to potions.
            payload["target_type"] = target;
        }

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

    private static object? CompactEventOptionPayload(JsonElement? element)
    {
        if (element is null || element.Value.ValueKind != JsonValueKind.Object)
        {
            return null;
        }

        return new
        {
            index = TryGetNestedInt(element.Value, "index"),
            option_type = TryGetNestedString(element.Value, "option_type"),
            option_id = TryGetNestedString(element.Value, "option_id"),
            title = TryGetNestedString(element.Value, "title"),
            description = TryGetNestedString(element.Value, "description"),
            is_locked = TryGetNestedBool(element.Value, "is_locked"),
            is_proceed = TryGetNestedBool(element.Value, "is_proceed"),
            is_selected = TryGetNestedBool(element.Value, "is_selected"),
            is_enabled = TryGetNestedBool(element.Value, "is_enabled"),
            action_available = TryGetNestedBool(element.Value, "action_available"),
            divination_size = TryGetNestedString(element.Value, "divination_size"),
            coord = CompactEventCoordPayload(TryGetNestedElement(element.Value, "coord")),
            is_highlighted = TryGetNestedBool(element.Value, "is_highlighted")
        };
    }

    private static object? CompactEventCoordPayload(JsonElement? element)
    {
        if (element is null || element.Value.ValueKind != JsonValueKind.Object)
        {
            return null;
        }

        return new
        {
            x = TryGetNestedInt(element.Value, "x") ?? TryGetNestedInt(element.Value, "col"),
            y = TryGetNestedInt(element.Value, "y") ?? TryGetNestedInt(element.Value, "row")
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

    private static decimal? TryGetNestedDecimal(JsonElement element, params string[] path)
    {
        var nested = TryGetNestedElement(element, path);
        return nested is { } value &&
               value.ValueKind == JsonValueKind.Number &&
               value.TryGetDecimal(out var parsed)
            ? parsed
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

}
