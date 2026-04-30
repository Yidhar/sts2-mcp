using System;
using System.Collections.Generic;
using System.Text;
using System.Text.Json;
using MegaCrit.Sts2.Core.Models;

namespace Sts2McpBridge.Scripts;

internal static partial class BridgeGameApi
{
    private const string CardEffectProfilesResource =
        "Sts2McpBridge.Content.card_effect_profiles.generated.json";

    private static readonly object _cardEffectProfileLock = new();
    private static IReadOnlyDictionary<string, JsonElement>? _cardEffectProfileCache;

    private static IReadOnlyDictionary<string, JsonElement> GetCardEffectProfileRegistry()
    {
        if (_cardEffectProfileCache is not null)
        {
            return _cardEffectProfileCache;
        }

        lock (_cardEffectProfileLock)
        {
            if (_cardEffectProfileCache is not null)
            {
                return _cardEffectProfileCache;
            }

            _cardEffectProfileCache = LoadCardEffectProfileRegistry();
            return _cardEffectProfileCache;
        }
    }

    private static IReadOnlyDictionary<string, JsonElement> LoadCardEffectProfileRegistry()
    {
        var registry = new Dictionary<string, JsonElement>(StringComparer.OrdinalIgnoreCase);
        try
        {
            var asm = typeof(BridgeGameApi).Assembly;
            using var stream = asm.GetManifestResourceStream(CardEffectProfilesResource);
            if (stream is null)
            {
                return registry;
            }

            using var doc = JsonDocument.Parse(stream);
            var root = doc.RootElement;
            if (root.TryGetProperty("cards", out var cards) && cards.ValueKind == JsonValueKind.Object)
            {
                root = cards;
            }

            if (root.ValueKind != JsonValueKind.Object)
            {
                return registry;
            }

            foreach (var prop in root.EnumerateObject())
            {
                if (prop.Value.ValueKind != JsonValueKind.Object)
                {
                    continue;
                }

                AddCardEffectProfileRegistryKey(registry, prop.Name, prop.Value);

                if (prop.Value.TryGetProperty("id", out var idEl) && idEl.ValueKind == JsonValueKind.String)
                {
                    AddCardEffectProfileRegistryKey(registry, idEl.GetString(), prop.Value);
                }

                if (prop.Value.TryGetProperty("normalized_id", out var normEl) && normEl.ValueKind == JsonValueKind.String)
                {
                    AddCardEffectProfileRegistryKey(registry, normEl.GetString(), prop.Value);
                }

                if (prop.Value.TryGetProperty("class_name", out var classEl) && classEl.ValueKind == JsonValueKind.String)
                {
                    AddCardEffectProfileRegistryKey(registry, classEl.GetString(), prop.Value);
                }
            }
        }
        catch
        {
            // Optional profile data must never crash the bridge.  If the embedded
            // resource is missing or malformed, downstream code simply falls back
            // to existing card fields / weak Python fallback heuristics.
        }

        return registry;
    }

    private static void AddCardEffectProfileRegistryKey(
        Dictionary<string, JsonElement> registry,
        string? key,
        JsonElement value)
    {
        if (string.IsNullOrWhiteSpace(key))
        {
            return;
        }

        var raw = key.Trim();
        var normalized = NormalizeCardEffectProfileKey(raw);
        if (string.IsNullOrWhiteSpace(normalized))
        {
            return;
        }

        registry[raw] = value.Clone();
        registry[normalized] = value.Clone();
        registry[normalized.ToUpperInvariant()] = value.Clone();
        registry["CARD." + normalized.ToUpperInvariant()] = value.Clone();
    }

    private static string NormalizeCardEffectProfileKey(string? key)
    {
        var text = (key ?? string.Empty).Trim();
        if (text.StartsWith("CARD.", StringComparison.OrdinalIgnoreCase))
        {
            text = text[5..];
        }

        var sb = new StringBuilder();
        char previous = '\0';
        foreach (var ch in text)
        {
            if (char.IsUpper(ch)
                && sb.Length > 0
                && char.IsLetterOrDigit(previous)
                && !char.IsUpper(previous)
                && sb[^1] != '_')
            {
                sb.Append('_');
            }

            if (char.IsLetterOrDigit(ch))
            {
                sb.Append(char.ToLowerInvariant(ch));
            }
            else if (sb.Length > 0 && sb[^1] != '_')
            {
                sb.Append('_');
            }

            previous = ch;
        }

        return sb.ToString().Trim('_');
    }

    private static JsonElement? TryGetCardEffectProfileEntry(CardModel? card)
    {
        if (card is null)
        {
            return null;
        }

        var registry = GetCardEffectProfileRegistry();
        var id = card.Id.ToString();
        var className = card.GetType().Name;
        var idNorm = NormalizeCardEffectProfileKey(id);
        var classNorm = NormalizeCardEffectProfileKey(className);

        var candidates = new[]
        {
            id,
            idNorm,
            idNorm.ToUpperInvariant(),
            "CARD." + idNorm.ToUpperInvariant(),
            className,
            classNorm,
            classNorm.ToUpperInvariant(),
            "CARD." + classNorm.ToUpperInvariant()
        };

        foreach (var candidate in candidates)
        {
            if (!string.IsNullOrWhiteSpace(candidate) && registry.TryGetValue(candidate, out var entry))
            {
                return entry;
            }
        }

        return null;
    }

    private static object? BuildCardEffectProfilePayload(CardModel? card)
    {
        var entry = TryGetCardEffectProfileEntry(card);
        return entry is { } value ? JsonElementToObject(value) : null;
    }
}
