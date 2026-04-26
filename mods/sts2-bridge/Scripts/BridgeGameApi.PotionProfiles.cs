// Phase 2 of docs/potion-timing-modeling-plan.md — Bridge-side structured potion profile.
//
// Loads the merged Python registry (potions.timing.generated.json + overrides) from
// embedded resources and exposes effect_profile / semantic_tags / timing_tags for
// BuildPotionPayload() / BuildUsePotionSemantic().
//
// The same JSON files are consumed by sts2_env/potion_profiles.py so the two sides
// cannot drift.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text.Json;
using MegaCrit.Sts2.Core.Models;

namespace Sts2McpBridge.Scripts;

internal static partial class BridgeGameApi
{
    private const string PotionGeneratedResource = "Sts2McpBridge.Content.potions.timing.generated.json";
    private const string PotionOverridesResource = "Sts2McpBridge.Content.potions.timing.overrides.json";

    private static readonly object _potionProfileLock = new();
    private static IReadOnlyDictionary<string, PotionProfileEntry>? _potionProfileCache;

    private sealed class PotionProfileEntry
    {
        public string Id { get; set; } = string.Empty;
        public string Title { get; set; } = string.Empty;
        public string Rarity { get; set; } = "Unknown";
        public string TargetScope { get; set; } = "Self";
        public bool EnabledForTraining { get; set; } = true;
        public List<string> EffectFamily { get; set; } = new();
        public Dictionary<string, JsonElement> EffectProfile { get; set; } = new();
        public List<string> SemanticTags { get; set; } = new();
        public List<string> TimingTags { get; set; } = new();
        public List<string> TrainingTags { get; set; } = new();
    }

    private static IReadOnlyDictionary<string, PotionProfileEntry> GetPotionProfileRegistry()
    {
        if (_potionProfileCache is not null)
        {
            return _potionProfileCache;
        }
        lock (_potionProfileLock)
        {
            if (_potionProfileCache is not null)
            {
                return _potionProfileCache;
            }
            _potionProfileCache = LoadPotionProfileRegistry();
            return _potionProfileCache;
        }
    }

    private static IReadOnlyDictionary<string, PotionProfileEntry> LoadPotionProfileRegistry()
    {
        var registry = new Dictionary<string, PotionProfileEntry>(StringComparer.Ordinal);
        ApplyJsonResource(registry, PotionGeneratedResource);
        ApplyJsonResource(registry, PotionOverridesResource);
        return registry;
    }

    private static void ApplyJsonResource(
        Dictionary<string, PotionProfileEntry> registry,
        string resourceName)
    {
        try
        {
            var asm = typeof(BridgeGameApi).Assembly;
            using var stream = asm.GetManifestResourceStream(resourceName);
            if (stream is null)
            {
                return;
            }
            using var doc = JsonDocument.Parse(stream);
            if (doc.RootElement.ValueKind != JsonValueKind.Object)
            {
                return;
            }
            foreach (var prop in doc.RootElement.EnumerateObject())
            {
                if (!prop.Name.StartsWith("POTION.", StringComparison.Ordinal))
                {
                    continue;
                }
                if (prop.Value.ValueKind != JsonValueKind.Object)
                {
                    continue;
                }
                if (!registry.TryGetValue(prop.Name, out var entry))
                {
                    entry = new PotionProfileEntry { Id = prop.Name };
                    registry[prop.Name] = entry;
                }
                ApplyJsonObject(entry, prop.Value);
            }
        }
        catch
        {
            // Defensive: if a resource is malformed, leave the registry partial rather than crash the bridge.
        }
    }

    private static void ApplyJsonObject(PotionProfileEntry entry, JsonElement obj)
    {
        foreach (var field in obj.EnumerateObject())
        {
            switch (field.Name)
            {
                case "id":
                    if (field.Value.ValueKind == JsonValueKind.String) entry.Id = field.Value.GetString() ?? entry.Id;
                    break;
                case "title":
                    if (field.Value.ValueKind == JsonValueKind.String) entry.Title = field.Value.GetString() ?? entry.Title;
                    break;
                case "rarity":
                    if (field.Value.ValueKind == JsonValueKind.String) entry.Rarity = field.Value.GetString() ?? entry.Rarity;
                    break;
                case "target_scope":
                    if (field.Value.ValueKind == JsonValueKind.String) entry.TargetScope = field.Value.GetString() ?? entry.TargetScope;
                    break;
                case "enabled_for_training":
                    if (field.Value.ValueKind == JsonValueKind.True) entry.EnabledForTraining = true;
                    else if (field.Value.ValueKind == JsonValueKind.False) entry.EnabledForTraining = false;
                    break;
                case "effect_family":
                    entry.EffectFamily = ReadStringList(field.Value);
                    break;
                case "semantic_tags":
                    entry.SemanticTags = ReadStringList(field.Value);
                    break;
                case "timing_tags":
                    entry.TimingTags = ReadStringList(field.Value);
                    break;
                case "training_tags":
                    entry.TrainingTags = ReadStringList(field.Value);
                    break;
                case "effect_profile":
                    if (field.Value.ValueKind == JsonValueKind.Object)
                    {
                        foreach (var slot in field.Value.EnumerateObject())
                        {
                            entry.EffectProfile[slot.Name] = slot.Value.Clone();
                        }
                    }
                    break;
            }
        }
    }

    private static List<string> ReadStringList(JsonElement el)
    {
        var list = new List<string>();
        if (el.ValueKind != JsonValueKind.Array)
        {
            return list;
        }
        foreach (var item in el.EnumerateArray())
        {
            if (item.ValueKind == JsonValueKind.String)
            {
                var s = item.GetString();
                if (!string.IsNullOrEmpty(s))
                {
                    list.Add(s);
                }
            }
        }
        return list;
    }

    private static PotionProfileEntry? TryGetPotionProfileEntry(PotionModel? potion)
    {
        if (potion is null)
        {
            return null;
        }
        var id = potion.Id.ToString();
        if (string.IsNullOrEmpty(id))
        {
            return null;
        }
        var registry = GetPotionProfileRegistry();
        return registry.TryGetValue(id, out var entry) ? entry : null;
    }

    private static object? BuildPotionEffectProfilePayload(PotionProfileEntry? entry)
    {
        if (entry is null || entry.EffectProfile.Count == 0)
        {
            return null;
        }
        var dict = new Dictionary<string, object?>(entry.EffectProfile.Count);
        foreach (var kv in entry.EffectProfile)
        {
            dict[kv.Key] = JsonElementToObject(kv.Value);
        }
        return dict;
    }

    private static object? JsonElementToObject(JsonElement el)
    {
        switch (el.ValueKind)
        {
            case JsonValueKind.True: return true;
            case JsonValueKind.False: return false;
            case JsonValueKind.Number:
                if (el.TryGetInt64(out var i)) return i;
                if (el.TryGetDouble(out var d)) return d;
                return 0.0;
            case JsonValueKind.String: return el.GetString();
            case JsonValueKind.Null: return null;
            case JsonValueKind.Array:
                {
                    var list = new List<object?>();
                    foreach (var item in el.EnumerateArray())
                    {
                        list.Add(JsonElementToObject(item));
                    }
                    return list;
                }
            case JsonValueKind.Object:
                {
                    var d2 = new Dictionary<string, object?>();
                    foreach (var item in el.EnumerateObject())
                    {
                        d2[item.Name] = JsonElementToObject(item.Value);
                    }
                    return d2;
                }
            default:
                return null;
        }
    }

    // Map effect_family entries to the closest existing semantic_action role names.
    // The trainer's SEMANTIC_ROLE_NAMES does not include "potion" as a primary role,
    // so we project potion families onto attack/block/draw/resource/debuff/buff/heal/aoe/setup.
    private static readonly Dictionary<string, string> _potionFamilyToRole = new(StringComparer.Ordinal)
    {
        { "damage", "attack" },
        { "aoe", "aoe" },
        { "self_damage", "aoe" },
        { "block", "block" },
        { "intangible", "block" },
        { "prevent_damage", "block" },
        { "delayed_block", "block" },
        { "draw", "draw" },
        { "energy", "resource" },
        { "energy_gain", "resource" },
        { "generate_cards", "resource" },
        { "discover", "resource" },
        { "retrieve_from_discard", "resource" },
        { "weak", "debuff" },
        { "vulnerable", "debuff" },
        { "poison", "debuff" },
        { "debuff", "debuff" },
        { "strength", "buff" },
        { "dexterity", "buff" },
        { "focus", "buff" },
        { "scaling", "buff" },
        { "ritual", "buff" },
        { "thorns", "buff" },
        { "plated", "buff" },
        { "regen", "buff" },
        { "buffer", "buff" },
        { "heal", "heal" },
        { "max_hp", "heal" },
        { "upgrade", "setup" },
        { "duplicate_next", "setup" },
        { "transform_hand", "setup" },
        { "exhaust_hand", "setup" },
        { "free_play", "setup" },
        { "snecko", "setup" },
        { "long_term", "setup" },
    };

    private static List<string> InferUsePotionRoles(PotionProfileEntry? entry)
    {
        var roles = new List<string> { "potion" };
        if (entry is null)
        {
            return roles;
        }
        foreach (var fam in entry.EffectFamily)
        {
            if (_potionFamilyToRole.TryGetValue(fam, out var role))
            {
                if (!roles.Contains(role))
                {
                    roles.Add(role);
                }
            }
        }
        return roles;
    }
}
