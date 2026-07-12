using System.Text.Json;
using System.Text.Json.Nodes;

namespace Sts2McpBridge.Scripts;

/// <summary>
/// Projects the legacy snapshot into the deliberately smaller player-control
/// state contract. Root fields are allowlisted; policy/training annotations are
/// removed recursively so adding a field to a legacy payload cannot silently
/// expand the v2 trust boundary.
/// </summary>
internal static class BridgePlayerStateProjector
{
    private static readonly string[] PlayerStateRootFields =
    [
        "screen",
        "run",
        "combat",
        "players",
        "rewards",
        "card_reward_selection",
        "card_selection",
        "character_selection",
        "run_mode_selection",
        "event_options",
        "crystal_sphere",
        "map",
        "rest_site",
        "deck_upgrade_selection",
        "shop",
        "main_menu"
    ];

    private static readonly HashSet<string> ForbiddenRecursiveFields = new(
        [
            "state_hash",
            "semantic_state_hash",
            "available_actions",
            "automation",
            "danger_profile",
            "target_priority_hints",
            "training_tags",
            "enabled_for_training",
            "debug_seed_override",
            "reward_breakdown"
        ],
        StringComparer.OrdinalIgnoreCase);

    public static JsonObject Project(
        object legacyStatePayload,
        JsonSerializerOptions? serializerOptions = null)
    {
        var legacyRoot = JsonSerializer.SerializeToNode(
            legacyStatePayload,
            serializerOptions) as JsonObject;
        var projected = new JsonObject();
        if (legacyRoot is null)
        {
            return projected;
        }

        foreach (var fieldName in PlayerStateRootFields)
        {
            if (legacyRoot.TryGetPropertyValue(fieldName, out var value))
            {
                projected[fieldName] = value?.DeepClone();
            }
        }

        SanitizePlayerVisibleNode(projected, parentPropertyName: null);
        return projected;
    }

    private static void SanitizePlayerVisibleNode(
        JsonNode? node,
        string? parentPropertyName)
    {
        if (node is JsonObject obj)
        {
            foreach (var property in obj.ToArray())
            {
                if (ForbiddenRecursiveFields.Contains(property.Key))
                {
                    obj.Remove(property.Key);
                    continue;
                }

                SanitizePlayerVisibleNode(property.Value, property.Key);
            }

            if (string.Equals(parentPropertyName, "draw_pile", StringComparison.OrdinalIgnoreCase))
            {
                // Count/type are visible, but composition and ordering are not.
                obj["cards"] = null;
                obj["cards_visible"] = false;
                obj["order_visible"] = false;
            }
            return;
        }

        if (node is JsonArray array)
        {
            foreach (var item in array)
            {
                SanitizePlayerVisibleNode(item, parentPropertyName);
            }
        }
    }
}
