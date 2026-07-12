using System.Text.Json;
using System.Text.Json.Nodes;

namespace Sts2McpBridge.Scripts;

/// <summary>
/// Pure Contract-v2 legal-action projection shared by the game adapter and
/// dependency-free wire-fixture tests.
/// </summary>
internal static class BridgeLegalActionProjector
{
    public static object Project(string actionHandle, object legacyPayload)
    {
        var element = JsonSerializer.SerializeToElement(legacyPayload);

        string? ReadString(string propertyName) =>
            element.ValueKind == JsonValueKind.Object &&
            element.TryGetProperty(propertyName, out var property) &&
            property.ValueKind == JsonValueKind.String
                ? property.GetString()
                : null;
        int? ReadInt(string propertyName) =>
            element.ValueKind == JsonValueKind.Object &&
            element.TryGetProperty(propertyName, out var property) &&
            property.TryGetInt32(out var value)
                ? value
                : null;

        var separator = actionHandle.IndexOf(':');
        var inferredKind = separator > 0 ? actionHandle[..separator] : actionHandle;
        return new
        {
            handle = actionHandle,
            kind = ReadString("kind") ?? inferredKind,
            label = ReadString("label") ?? ReadString("description"),
            target_handle = ReadString("target_handle") ?? ReadString("target_id"),
            coord = BuildCoordinatePayload(element),
            option_index = ReadInt("option_index") ?? ReadInt("index"),
            selection_id = ReadString("selection_id"),
            card_ref = ReadString("card_ref"),
            slot_index = ReadInt("slot_index")
        };
    }

    /// <summary>
    /// Projects the legacy training-environment action array onto the strict
    /// Contract-v2 wire shape. All training metadata is retained verbatim, but
    /// the legacy action_id identity is replaced by action_handle.
    /// </summary>
    public static JsonArray ProjectEnvironmentActions(object? legacyActions)
    {
        var element = JsonSerializer.SerializeToElement(legacyActions);
        if (element.ValueKind is JsonValueKind.Null or JsonValueKind.Undefined)
        {
            return new JsonArray();
        }
        if (element.ValueKind != JsonValueKind.Array)
        {
            throw new InvalidOperationException("Legacy environment legal_actions was not an array.");
        }

        var projected = new JsonArray();
        foreach (var action in element.EnumerateArray())
        {
            projected.Add(ProjectEnvironmentAction(action));
        }
        return projected;
    }

    internal static JsonObject ProjectEnvironmentAction(object legacyPayload)
    {
        var element = JsonSerializer.SerializeToElement(legacyPayload);
        if (element.ValueKind != JsonValueKind.Object ||
            !element.TryGetProperty("action_id", out var actionId) ||
            actionId.ValueKind != JsonValueKind.String ||
            string.IsNullOrWhiteSpace(actionId.GetString()))
        {
            throw new InvalidOperationException(
                "Every legacy environment legal action must contain a non-empty action_id.");
        }

        var actionHandle = actionId.GetString()!;
        if (actionHandle.Length > 512)
        {
            throw new InvalidOperationException(
                "Legacy environment legal-action identity exceeds the Contract-v2 512-character limit.");
        }
        var result = new JsonObject();
        foreach (var property in element.EnumerateObject())
        {
            if (property.NameEquals("action_id"))
            {
                result["action_handle"] = actionHandle;
                continue;
            }
            if (property.NameEquals("action_handle"))
            {
                // Never retain a second identity supplied by a legacy payload.
                continue;
            }
            result[property.Name] = JsonNode.Parse(property.Value.GetRawText());
        }
        return result;
    }

    private static object? BuildCoordinatePayload(JsonElement actionPayload)
    {
        if (actionPayload.ValueKind != JsonValueKind.Object ||
            !actionPayload.TryGetProperty("coord", out var coord) ||
            coord.ValueKind != JsonValueKind.Object)
        {
            return null;
        }

        static int? ReadCoordinate(JsonElement value, string primaryName, string legacyName)
        {
            if (value.TryGetProperty(primaryName, out var primary) && primary.TryGetInt32(out var parsedPrimary))
            {
                return parsedPrimary;
            }
            return value.TryGetProperty(legacyName, out var legacy) && legacy.TryGetInt32(out var parsedLegacy)
                ? parsedLegacy
                : null;
        }

        var x = ReadCoordinate(coord, "x", "col");
        var y = ReadCoordinate(coord, "y", "row");
        return x.HasValue && y.HasValue ? new { x = x.Value, y = y.Value } : null;
    }
}
