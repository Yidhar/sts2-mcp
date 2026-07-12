using System.Collections;
using System.Buffers.Binary;
using System.Globalization;
using System.Linq;
using System.Net;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.Json.Serialization;
using System.Text.RegularExpressions;
using Godot;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Merchant;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Entities.RestSite;
using MegaCrit.Sts2.Core.Events;
using MegaCrit.Sts2.Core.Events.Custom.CrystalSphereEvent;
using MegaCrit.Sts2.Core.GameActions;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Localization.DynamicVars;
using MegaCrit.Sts2.Core.Map;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.MonsterMoves.Intents;
using MegaCrit.Sts2.Core.MonsterMoves.MonsterMoveStateMachine;
using MegaCrit.Sts2.Core.Multiplayer.Game.PeerInput;
using MegaCrit.Sts2.Core.Nodes;
using MegaCrit.Sts2.Core.Nodes.Cards;
using MegaCrit.Sts2.Core.Nodes.Cards.Holders;
using MegaCrit.Sts2.Core.Nodes.Combat;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.Events;
using MegaCrit.Sts2.Core.Nodes.Events.Custom.CrystalSphere;
using MegaCrit.Sts2.Core.Nodes.Rewards;
using MegaCrit.Sts2.Core.Nodes.RestSite;
using MegaCrit.Sts2.Core.Nodes.Rooms;
using MegaCrit.Sts2.Core.Nodes.Screens;
using MegaCrit.Sts2.Core.Nodes.Screens.CharacterSelect;
using MegaCrit.Sts2.Core.Nodes.Screens.CardSelection;
using MegaCrit.Sts2.Core.Nodes.Screens.GameOverScreen;
using MegaCrit.Sts2.Core.Nodes.Screens.MainMenu;
using MegaCrit.Sts2.Core.Nodes.Screens.Map;
using MegaCrit.Sts2.Core.Nodes.Screens.Overlays;
using MegaCrit.Sts2.Core.Nodes.Screens.ScreenContext;
using MegaCrit.Sts2.Core.Nodes.Screens.Shops;
using MegaCrit.Sts2.Core.Nodes.Screens.TreasureRoomRelic;
using MegaCrit.Sts2.Core.Nodes.TreasureRooms;
using MegaCrit.Sts2.Core.Rewards;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;

namespace Sts2McpBridge.Scripts;

internal static partial class BridgeGameApi
{
    private static object BuildEventOptionPayload(
        NEventOptionButton button,
        int index,
        EventModel? eventModel)
    {
        var option = button.Option;
        object? optionTextContext = option is null
            ? eventModel
            : eventModel is null
                ? option
                : new object?[] { option, eventModel };

        // Live event-option introspection has proven unsafe for some reward-
        // backed options (notably Neow / BaseLib interaction probes). Accessing
        // option hover tips / embedded relic payloads can materialize reward
        // previews with real game-side effects. Keep the live bridge payload on
        // the visible text-only path here; static export keeps the richer data.
        var glossary = Array.Empty<object>();

        var resolvedTitle = option is null ? string.Empty : DescribeText(option.Title, optionTextContext);
        var resolvedDescription = option is null ? string.Empty : DescribeText(option.Description, optionTextContext);
        return new
        {
            index,
            title = resolvedTitle,
            description = resolvedDescription,
            is_locked = option?.IsLocked ?? true,
            is_proceed = option?.IsProceed ?? false,
            relic = (object?)null,
            glossary
        };
    }

    private static (
        string? Source,
        IReadOnlyList<string> Texts,
        IReadOnlyList<(string Title, string? Description, string[] Texts)> Entries) CollectVisibleEventGlossaryTexts(
        IReadOnlyList<NEventOptionButton> eventOptionButtons,
        NEventRoom? eventRoom,
        Node? hoverTipSet)
    {
        var excludedTexts = new HashSet<string>(
            eventOptionButtons
                .SelectMany(static button => CollectButtonPayloadTexts(button, 4))
                .Select(NormalizeComparableText)
                .Where(static text => !string.IsNullOrWhiteSpace(text)),
            StringComparer.Ordinal);

        var hoverTipEntries = FilterGlossaryEntries(
            ExtractVisibleHoverTipEntries(hoverTipSet),
            excludedTexts);
        if (hoverTipEntries.Count > 0)
        {
            return (
                "hover_tip_set",
                FlattenGlossaryTexts(hoverTipEntries),
                hoverTipEntries);
        }

        var eventRoomTexts = FilterGlossaryCandidateTexts(
            CollectLocalVisibleText(eventRoom, 24, maxDepth: 3),
            excludedTexts);
        if (eventRoomTexts.Count > 0)
        {
            return (
                "event_room_fallback",
                eventRoomTexts,
                BuildGlossaryEntriesFromTexts(eventRoomTexts));
        }

        return (
            null,
            Array.Empty<string>(),
            Array.Empty<(string Title, string? Description, string[] Texts)>());
    }

    private static IReadOnlyList<(string Title, string? Description, string[] Texts)> ExtractVisibleHoverTipEntries(
        Node? hoverTipSet)
    {
        if (hoverTipSet is null || !IsNodeVisible(hoverTipSet))
        {
            return Array.Empty<(string Title, string? Description, string[] Texts)>();
        }

        var textHoverTipContainer = GetHiddenFieldValue(hoverTipSet, "_textHoverTipContainer") as Node ??
                                    GetHiddenPropertyObjectValue(hoverTipSet, "TextHoverTipContainer") as Node ??
                                    hoverTipSet.GetNodeOrNull<Node>("textHoverTipContainer") ??
                                    FindVisibleImmediateChildByName(hoverTipSet, "textHoverTipContainer");
        if (textHoverTipContainer is null || !IsNodeVisible(textHoverTipContainer))
        {
            return Array.Empty<(string Title, string? Description, string[] Texts)>();
        }

        var entries = new List<(string Title, string? Description, string[] Texts)>();
        var seen = new HashSet<string>(StringComparer.Ordinal);
        foreach (var hoverTip in SortByVisualPosition(
                     textHoverTipContainer
                         .GetChildren()
                         .OfType<Node>()
                         .Where(IsNodeVisible)))
        {
            var titleNode = hoverTip.GetNodeOrNull<Node>("%Title") ??
                            FindVisibleImmediateChildByName(hoverTip, "Title");
            var descriptionNode = hoverTip.GetNodeOrNull<Node>("%Description") ??
                                  FindVisibleImmediateChildByName(hoverTip, "Description");
            var title = TryGetLocalNodeText(titleNode);
            var description = TryGetLocalNodeText(descriptionNode);
            var texts = new[] { title, description }
                .Where(static text => !string.IsNullOrWhiteSpace(text))
                .Select(static text => text.ReplaceLineEndings("\n").Trim())
                .ToArray();
            if (texts.Length == 0)
            {
                continue;
            }

            var dedupeKey = string.Join(
                "|",
                texts.Select(NormalizeComparableText));
            if (!seen.Add(dedupeKey))
            {
                continue;
            }

            entries.Add((
                string.IsNullOrWhiteSpace(title) ? texts[0] : title.ReplaceLineEndings("\n").Trim(),
                string.IsNullOrWhiteSpace(description) ? null : description.ReplaceLineEndings("\n").Trim(),
                texts));
        }

        return entries;
    }

    private static IReadOnlyList<(string Title, string? Description, string[] Texts)> FilterGlossaryEntries(
        IReadOnlyList<(string Title, string? Description, string[] Texts)> entries,
        IReadOnlySet<string> excludedTexts)
    {
        if (entries.Count == 0)
        {
            return Array.Empty<(string Title, string? Description, string[] Texts)>();
        }

        var filtered = new List<(string Title, string? Description, string[] Texts)>();
        var seen = new HashSet<string>(StringComparer.Ordinal);

        foreach (var entry in entries)
        {
            var comparableTexts = entry.Texts
                .Select(NormalizeComparableText)
                .Where(static text => !string.IsNullOrWhiteSpace(text))
                .ToArray();
            if (comparableTexts.Length == 0 || comparableTexts.All(excludedTexts.Contains))
            {
                continue;
            }

            var dedupeKey = string.Join("|", comparableTexts);
            if (!seen.Add(dedupeKey))
            {
                continue;
            }

            filtered.Add(entry);
        }

        return filtered;
    }

    private static IReadOnlyList<string> FlattenGlossaryTexts(
        IReadOnlyList<(string Title, string? Description, string[] Texts)> entries)
    {
        if (entries.Count == 0)
        {
            return Array.Empty<string>();
        }

        var texts = new List<string>();
        var seen = new HashSet<string>(StringComparer.Ordinal);
        foreach (var entry in entries)
        {
            foreach (var text in entry.Texts)
            {
                var normalized = text.ReplaceLineEndings("\n").Trim();
                if (!string.IsNullOrWhiteSpace(normalized) && seen.Add(normalized))
                {
                    texts.Add(normalized);
                }
            }
        }

        return texts;
    }

    private static IReadOnlyList<(string Title, string? Description, string[] Texts)> BuildGlossaryEntriesFromTexts(
        IReadOnlyList<string> glossaryTexts)
    {
        if (glossaryTexts.Count == 0)
        {
            return Array.Empty<(string Title, string? Description, string[] Texts)>();
        }

        var entries = new List<(string Title, string? Description, string[] Texts)>();

        for (var index = 0; index < glossaryTexts.Count; index++)
        {
            var title = glossaryTexts[index];
            string? description = null;

            if (index + 1 < glossaryTexts.Count &&
                LooksLikeGlossaryTitle(title) &&
                LooksLikeGlossaryDescription(glossaryTexts[index + 1], title))
            {
                description = glossaryTexts[index + 1];
                index++;
            }

            entries.Add((
                title,
                description,
                description is null ? new[] { title } : new[] { title, description }));
        }

        return entries;
    }

    private static IReadOnlyList<string> FilterGlossaryCandidateTexts(
        IEnumerable<string> texts,
        IReadOnlySet<string> excludedTexts)
    {
        var filtered = new List<string>();
        var seen = new HashSet<string>(StringComparer.Ordinal);

        foreach (var rawText in texts)
        {
            var comparableText = NormalizeComparableText(rawText);
            if (string.IsNullOrWhiteSpace(comparableText) ||
                excludedTexts.Contains(comparableText) ||
                !seen.Add(comparableText))
            {
                continue;
            }

            filtered.Add(rawText.ReplaceLineEndings("\n").Trim());
        }

        return filtered;
    }

    private static object[] BuildVisibleGlossaryPayload(
        IReadOnlyList<(string Title, string? Description, string[] Texts)> glossaryEntries)
    {
        if (glossaryEntries.Count == 0)
        {
            return Array.Empty<object>();
        }

        return glossaryEntries
            .Select(static entry => new
            {
                title = entry.Title,
                description = entry.Description,
                texts = entry.Texts
            })
            .ToArray();
    }

    private static object[] BuildHoverTipPayloads(IEnumerable? hoverTips)
    {
        if (hoverTips is null)
        {
            return Array.Empty<object>();
        }

        var entries = new List<object>();
        var seen = new HashSet<string>(StringComparer.Ordinal);

        foreach (var hoverTip in hoverTips)
        {
            if (hoverTip is null)
            {
                continue;
            }

            var canonicalModel = GetHiddenPropertyObjectValue(hoverTip, "CanonicalModel") as AbstractModel;
            var id = TextOf(GetHiddenPropertyObjectValue(hoverTip, "Id"));
            var title = ResolveHoverTipTitle(hoverTip, canonicalModel, id);
            var description = ResolveHoverTipDescription(hoverTip, canonicalModel);
            var dedupeKey = string.Join(
                "|",
                hoverTip.GetType().FullName ?? hoverTip.GetType().Name,
                NormalizeComparableText(id),
                NormalizeComparableText(title),
                NormalizeComparableText(description));

            if (!seen.Add(dedupeKey))
            {
                continue;
            }

            entries.Add(new
            {
                id,
                type = hoverTip.GetType().Name,
                title,
                description,
                is_debuff = GetHiddenPropertyValue<bool>(hoverTip, "IsDebuff") ?? false,
                is_instanced = GetHiddenPropertyValue<bool>(hoverTip, "IsInstanced") ?? false,
                is_smart = GetHiddenPropertyValue<bool>(hoverTip, "IsSmart") ?? false,
                canonical_model = canonicalModel is null ? null : BuildModelPayload(canonicalModel),
                texts = new[] { title, description }
                    .Where(static text => !string.IsNullOrWhiteSpace(text))
                    .ToArray()
            });
        }

        return entries.ToArray();
    }

    private static string ResolveHoverTipTitle(object hoverTip, AbstractModel? canonicalModel, string fallbackId)
    {
        return FirstNonEmptyText(
            hoverTip is Node hoverTipNode ? TryGetHoverTipNodeNamedText(hoverTipNode, "Title") : string.Empty,
            TryGetNamedValueText(hoverTip, "HoverTipTitle"),
            TryGetNamedValueText(hoverTip, "Title"),
            TryGetNamedValueText(hoverTip, "Name"),
            TryGetNamedValueText(hoverTip, "Label"),
            TryGetNamedValueText(hoverTip, "BotKeyword"),
            canonicalModel is null ? string.Empty : TryGetTitle(canonicalModel),
            fallbackId);
    }

    private static string ResolveHoverTipDescription(object hoverTip, AbstractModel? canonicalModel)
    {
        return FirstNonEmptyText(
            hoverTip is Node hoverTipNode ? TryGetHoverTipNodeNamedText(hoverTipNode, "Description") : string.Empty,
            TryGetNamedValueText(hoverTip, "HoverTipDesc"),
            TryGetNamedValueText(hoverTip, "Description"),
            TryGetNamedValueText(hoverTip, "Text"),
            TryGetNamedValueText(hoverTip, "Body"),
            TryGetNamedValueText(hoverTip, "BotText"),
            canonicalModel is null ? string.Empty : TryGetDescription(canonicalModel));
    }

    private static string TryGetHoverTipNodeNamedText(Node? hoverTipNode, string nodeName)
    {
        if (hoverTipNode is null || !IsNodeVisible(hoverTipNode))
        {
            return string.Empty;
        }

        var textNode = hoverTipNode.GetNodeOrNull<Node>($"%{nodeName}") ??
                       FindVisibleImmediateChildByName(hoverTipNode, nodeName);
        return TryGetLocalNodeText(textNode);
    }

    private static string TryGetNamedValueText(object target, string memberName)
    {
        var value = GetHiddenPropertyObjectValue(target, memberName) ??
                    GetHiddenFieldValue(target, memberName);
        return value is Node node
            ? TryGetLocalNodeText(node)
            : DescribeText(value, target);
    }

    private static string FirstNonEmptyText(params string[] candidates)
    {
        foreach (var candidate in candidates)
        {
            if (!string.IsNullOrWhiteSpace(candidate))
            {
                return candidate;
            }
        }

        return string.Empty;
    }

    private static bool LooksLikeGlossaryTitle(string text)
    {
        if (string.IsNullOrWhiteSpace(text))
        {
            return false;
        }

        var normalized = text.ReplaceLineEndings("\n").Trim();
        return normalized.Length <= 32 &&
               !normalized.Contains('\n') &&
               !normalized.Contains('。') &&
               !normalized.Contains('！') &&
               !normalized.Contains('？') &&
               !normalized.Contains('：');
    }

    private static bool LooksLikeGlossaryDescription(string text, string title)
    {
        if (string.IsNullOrWhiteSpace(text))
        {
            return false;
        }

        var normalized = text.ReplaceLineEndings("\n").Trim();
        var normalizedTitle = title.ReplaceLineEndings("\n").Trim();
        if (string.Equals(normalized, normalizedTitle, StringComparison.Ordinal))
        {
            return false;
        }

        return normalized.Contains('\n') ||
               normalized.Contains('。') ||
               normalized.Contains('！') ||
               normalized.Contains('？') ||
               normalized.Contains('：') ||
               normalized.Length > normalizedTitle.Length;
    }

}
