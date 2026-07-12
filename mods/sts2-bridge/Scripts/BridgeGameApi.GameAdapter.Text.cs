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
    private static string DescribeCharacter(CharacterModel? character)
    {
        return DescribeCharacterText(character?.CharacterSelectTitle, character);
    }

    private static string DescribeCharacterDescription(CharacterModel? character)
    {
        return DescribeCharacterText(character?.CharacterSelectDesc, character);
    }

    private static string DescribeCharacterText(string? textKey, CharacterModel? character)
    {
        if (character is null || string.IsNullOrWhiteSpace(textKey))
        {
            return DescribeText(textKey, character);
        }

        return DescribeText(new LocString("characters", textKey), character);
    }

    private static string TryGetTitle(object model)
    {
        if (model is CharacterModel character)
        {
            return DescribeCharacter(character);
        }

        var title = TryGetNamedTextValue(model, "Title", "TitleLocString");
        return string.IsNullOrWhiteSpace(title)
            ? DescribeText(model, model)
            : title;
    }

    private static string TryGetDescription(object model)
    {
        // Some runtime models (notably relic/potion models reachable from live
        // event / Neow option payloads) expose dynamic description accessors
        // that are not side-effect free. Probing the broader preferred-description
        // surface can accidentally instantiate reward visuals or even trigger
        // obtain-side logic while we're only trying to serialize text.
        //
        // Keep relic / potion descriptions on the narrow, previously-stable
        // direct Description path for live bridge payloads.
        if (model is RelicModel relic)
        {
            return DescribeRelicModelSafely(relic);
        }

        if (model is PotionModel potion)
        {
            return DescribePotionModelSafely(potion);
        }

        if (model is CharacterModel character)
        {
            return DescribeCharacterDescription(character);
        }

        return DescribeText(TryGetPreferredDescriptionValue(model), model);
    }

    private static string DescribeRelicModelSafely(RelicModel relic)
    {
        return DescribeText(GetHiddenPropertyObjectValue(relic, "DynamicDescription"), relic);
    }

    private static string DescribePotionModelSafely(PotionModel potion)
    {
        return DescribeText(GetHiddenPropertyObjectValue(potion, "DynamicDescription"), potion);
    }

    private static string TextOf(object? value)
    {
        return ReadTextValue(value, preferRawText: false, allowFormattedFallback: true);
    }

    private static string TextOfRawFirst(object? value, bool allowFormattedFallback = true)
    {
        return ReadTextValue(value, preferRawText: true, allowFormattedFallback);
    }

    private static string ReadTextValue(object? value, bool preferRawText, bool allowFormattedFallback)
    {
        if (value is string textValue)
        {
            return textValue;
        }

        if (value is LocString locString)
        {
            var locText = ReadLocStringText(locString, preferRawText, allowFormattedFallback);
            if (!string.IsNullOrWhiteSpace(locText))
            {
                return locText;
            }
        }

        if (value is not null)
        {
            var primaryMethodName = preferRawText ? "GetRawText" : "GetFormattedText";
            var fallbackMethodName = preferRawText ? "GetFormattedText" : "GetRawText";

            var primaryText = TryInvokeTextMethod(value, primaryMethodName);
            if (!string.IsNullOrWhiteSpace(primaryText))
            {
                return primaryText;
            }

            if (allowFormattedFallback || !preferRawText)
            {
                var fallbackText = TryInvokeTextMethod(value, fallbackMethodName);
                if (!string.IsNullOrWhiteSpace(fallbackText))
                {
                    return fallbackText;
                }
            }
        }

        var text = value?.ToString() ?? string.Empty;
        return value is not null && LooksLikeTypeName(text, value.GetType())
            ? string.Empty
            : text;
    }

    private static string ReadLocStringText(
        LocString locString,
        bool preferRawText,
        bool allowFormattedFallback)
    {
        if (preferRawText)
        {
            var rawText = TryGetLocStringRawText(locString);
            if (!string.IsNullOrWhiteSpace(rawText))
            {
                return rawText;
            }

            if (allowFormattedFallback)
            {
                var formattedText = TryGetLocStringFormattedText(locString);
                if (!string.IsNullOrWhiteSpace(formattedText))
                {
                    return formattedText;
                }
            }

            return string.Empty;
        }

        var formatted = TryGetLocStringFormattedText(locString);
        if (!string.IsNullOrWhiteSpace(formatted))
        {
            return formatted;
        }

        return TryGetLocStringRawText(locString);
    }

    private static string TryGetLocStringFormattedText(LocString locString)
    {
        try
        {
            return locString.GetFormattedText();
        }
        catch
        {
            return string.Empty;
        }
    }

    private static string TryGetLocStringRawText(LocString locString)
    {
        try
        {
            return locString.GetRawText();
        }
        catch
        {
            return string.Empty;
        }
    }

    private static string DescribeText(object? value, object? placeholderContext = null)
    {
        var text = TextOfRawFirst(value, allowFormattedFallback: false);
        if (string.IsNullOrWhiteSpace(text))
        {
            text = TextOfRawFirst(value);
        }

        if (string.IsNullOrWhiteSpace(text))
        {
            return text;
        }

        object? effectivePlaceholderContext = placeholderContext;
        if (value is not null)
        {
            effectivePlaceholderContext = placeholderContext is null || ReferenceEquals(value, placeholderContext)
                ? value
                : new object?[] { value, placeholderContext };
        }

        return NormalizePayloadText(ResolvePlaceholderText(text, effectivePlaceholderContext));
    }

    private static string NormalizePayloadText(string text)
    {
        if (string.IsNullOrWhiteSpace(text))
        {
            return string.Empty;
        }

        var normalized = ReplaceImageTags(text.ReplaceLineEndings("\n")).Trim();
        if (normalized.Length == 0)
        {
            return string.Empty;
        }

        normalized = StripBbCode(normalized);
        normalized = normalized.Replace("[", string.Empty).Replace("]", string.Empty);
        normalized = normalized.Replace(" \n", "\n").Replace("\n ", "\n");

        var builder = new StringBuilder(normalized.Length);
        var previousWasWhitespace = false;

        foreach (var character in normalized)
        {
            if (character == '\n')
            {
                if (builder.Length > 0 && builder[^1] == ' ')
                {
                    builder.Length--;
                }

                if (builder.Length == 0 || builder[^1] != '\n')
                {
                    builder.Append('\n');
                }

                previousWasWhitespace = false;
                continue;
            }

            if (char.IsWhiteSpace(character))
            {
                if (!previousWasWhitespace)
                {
                    builder.Append(' ');
                    previousWasWhitespace = true;
                }

                continue;
            }

            builder.Append(character);
            previousWasWhitespace = false;
        }

        return builder.ToString().Trim();
    }

    private const string ImageTagMarkerPrefix = "<<sts2-icon:";
    private const string ImageTagMarkerSuffix = ">>";

    private static string ReplaceImageTags(string text)
    {
        if (string.IsNullOrWhiteSpace(text))
        {
            return string.Empty;
        }

        const string openTag = "[img]";
        const string closeTag = "[/img]";
        var builder = new StringBuilder(text.Length);
        var cursor = 0;

        while (cursor < text.Length)
        {
            var openIndex = text.IndexOf(openTag, cursor, StringComparison.OrdinalIgnoreCase);
            if (openIndex < 0)
            {
                builder.Append(text, cursor, text.Length - cursor);
                break;
            }

            builder.Append(text, cursor, openIndex - cursor);

            var contentStart = openIndex + openTag.Length;
            var closeIndex = text.IndexOf(closeTag, contentStart, StringComparison.OrdinalIgnoreCase);
            if (closeIndex < 0)
            {
                builder.Append(text, openIndex, text.Length - openIndex);
                break;
            }

            var inner = text.Substring(contentStart, closeIndex - contentStart);
            builder.Append(CreateImageTagMarker(inner));
            cursor = closeIndex + closeTag.Length;
        }

        return CollapseImageTagMarkers(builder.ToString());
    }

    private static string CreateImageTagMarker(string inner)
    {
        if (TryRecognizeImageTagKind(inner, out var kind))
        {
            return $"{ImageTagMarkerPrefix}{kind}{ImageTagMarkerSuffix}";
        }

        var debugName = GetImageTagDebugName(inner);
        return $"{ImageTagMarkerPrefix}unknown:{debugName}{ImageTagMarkerSuffix}";
    }

    private static string CollapseImageTagMarkers(string text)
    {
        if (string.IsNullOrWhiteSpace(text) ||
            text.IndexOf(ImageTagMarkerPrefix, StringComparison.Ordinal) < 0)
        {
            return text;
        }

        var collapsed = CollapseKnownImageTagMarkers(text, "energy", "点能量", allowCountPrefix: true);
        collapsed = CollapseKnownImageTagMarkers(collapsed, "star", "点星辉", allowCountPrefix: true);

        var unknownPattern =
            $"{Regex.Escape(ImageTagMarkerPrefix)}(?<token>[^>]+){Regex.Escape(ImageTagMarkerSuffix)}";
        return Regex.Replace(
            collapsed,
            unknownPattern,
            match =>
            {
                var token = match.Groups["token"].Value;
                return token.StartsWith("unknown:", StringComparison.OrdinalIgnoreCase)
                    ? $"图标:{token["unknown:".Length..]}"
                    : $"图标:{token}";
            },
            RegexOptions.IgnoreCase | RegexOptions.CultureInvariant);
    }

    private static string CollapseKnownImageTagMarkers(
        string text,
        string kind,
        string unitLabel,
        bool allowCountPrefix = false)
    {
        if (string.IsNullOrWhiteSpace(text))
        {
            return string.Empty;
        }

        var marker = $"{ImageTagMarkerPrefix}{kind}{ImageTagMarkerSuffix}";
        var options = RegexOptions.IgnoreCase | RegexOptions.CultureInvariant;

        if (allowCountPrefix)
        {
            text = Regex.Replace(
                text,
                $@"(?<count>\d+)\s*{Regex.Escape(marker)}",
                match => $"{match.Groups["count"].Value}{unitLabel}",
                options);
        }

        var repeatedPattern = $@"(?:{Regex.Escape(marker)}\s*)+";
        return Regex.Replace(
            text,
            repeatedPattern,
            match =>
            {
                var count = Regex.Matches(match.Value, Regex.Escape(marker), options).Count;
                return count > 0 ? $"{count}{unitLabel}" : match.Value;
            },
            options);
    }

    private static bool TryRecognizeImageTagKind(string inner, out string kind)
    {
        kind = string.Empty;
        var debugName = GetImageTagDebugName(inner);
        if (debugName.Length == 0)
        {
            return false;
        }

        if (string.Equals(debugName, "star_icon", StringComparison.OrdinalIgnoreCase))
        {
            kind = "star";
            return true;
        }

        if (debugName.EndsWith("_energy_icon", StringComparison.OrdinalIgnoreCase))
        {
            kind = "energy";
            return true;
        }

        return false;
    }

    private static string GetImageTagDebugName(string inner)
    {
        if (string.IsNullOrWhiteSpace(inner))
        {
            return "empty";
        }

        var normalized = inner.Trim();
        var slashIndex = normalized.LastIndexOfAny(new[] { '/', '\\' });
        if (slashIndex >= 0 && slashIndex + 1 < normalized.Length)
        {
            normalized = normalized[(slashIndex + 1)..];
        }

        var dotIndex = normalized.LastIndexOf('.');
        if (dotIndex > 0)
        {
            normalized = normalized[..dotIndex];
        }

        var builder = new StringBuilder(normalized.Length);
        foreach (var character in normalized)
        {
            if (char.IsLetterOrDigit(character) || character is '_' or '-' or ':')
            {
                builder.Append(char.ToLowerInvariant(character));
            }
        }

        return builder.Length > 0 ? builder.ToString() : "unknown";
    }

    private static string ResolvePlaceholderText(string text, object? placeholderContext)
    {
        if (string.IsNullOrWhiteSpace(text) ||
            placeholderContext is null ||
            !text.Contains('{'))
        {
            return text;
        }

        var builder = new StringBuilder(text.Length);
        var cursor = 0;

        while (cursor < text.Length)
        {
            var openBrace = text.IndexOf('{', cursor);
            if (openBrace < 0)
            {
                builder.Append(text, cursor, text.Length - cursor);
                break;
            }

            var closeBrace = FindPlaceholderCloseBrace(text, openBrace);
            if (closeBrace < 0)
            {
                builder.Append(text, cursor, text.Length - cursor);
                break;
            }

            builder.Append(text, cursor, openBrace - cursor);

            var placeholderBody = text.Substring(openBrace + 1, closeBrace - openBrace - 1);
            if (TryResolvePlaceholderText(placeholderContext, placeholderBody, out var resolvedPlaceholder))
            {
                builder.Append(resolvedPlaceholder);
            }
            else
            {
                builder.Append(text, openBrace, closeBrace - openBrace + 1);
            }

            cursor = closeBrace + 1;
        }

        return builder.ToString();
    }

    private static int FindPlaceholderCloseBrace(string text, int openBraceIndex)
    {
        if (string.IsNullOrEmpty(text) ||
            openBraceIndex < 0 ||
            openBraceIndex >= text.Length ||
            text[openBraceIndex] != '{')
        {
            return -1;
        }

        var depth = 0;
        for (var index = openBraceIndex; index < text.Length; index++)
        {
            switch (text[index])
            {
                case '{':
                    depth++;
                    break;
                case '}':
                    depth--;
                    if (depth == 0)
                    {
                        return index;
                    }

                    break;
            }
        }

        return -1;
    }

    private static bool TryResolvePlaceholderText(
        object placeholderContext,
        string placeholderBody,
        out string resolvedText)
    {
        resolvedText = string.Empty;

        if (string.IsNullOrWhiteSpace(placeholderBody))
        {
            return false;
        }

        var separatorIndex = placeholderBody.IndexOf(':');
        var tokenName = separatorIndex >= 0
            ? placeholderBody[..separatorIndex].Trim()
            : placeholderBody.Trim();
        var formatHint = separatorIndex >= 0
            ? placeholderBody[(separatorIndex + 1)..].Trim()
            : string.Empty;

        if (string.IsNullOrWhiteSpace(tokenName))
        {
            return false;
        }

        if (TryResolveStandalonePlaceholderToken(tokenName, formatHint, out resolvedText))
        {
            resolvedText = ResolvePlaceholderText(resolvedText, placeholderContext);
            return !string.IsNullOrWhiteSpace(resolvedText);
        }

        if (!TryResolvePlaceholderValue(placeholderContext, tokenName, out var resolvedValue))
        {
            return false;
        }

        resolvedText = FormatResolvedPlaceholderValue(tokenName, formatHint, resolvedValue);
        resolvedText = ResolvePlaceholderText(resolvedText, placeholderContext);
        return !string.IsNullOrWhiteSpace(resolvedText);
    }

    private static bool TryResolvePlaceholderValue(
        object placeholderContext,
        string tokenName,
        out object? resolvedValue)
    {
        resolvedValue = null;

        foreach (var candidate in EnumeratePlaceholderContexts(placeholderContext))
        {
            if (candidate is null)
            {
                continue;
            }

            if (TryResolvePlaceholderValueFromLocStringVariables(candidate, tokenName, out resolvedValue) ||
                TryResolvePlaceholderValueFromTypeHierarchy(candidate, tokenName, out resolvedValue) ||
                TryResolvePlaceholderValueFromCanonicalVars(candidate, tokenName, out resolvedValue) ||
                TryResolvePlaceholderValueFromDynamicVars(candidate, tokenName, out resolvedValue))
            {
                return true;
            }
        }

        return false;
    }

    private static bool TryResolvePlaceholderValueFromLocStringVariables(
        object candidate,
        string tokenName,
        out object? resolvedValue)
    {
        resolvedValue = null;

        if (candidate is not LocString locString)
        {
            return false;
        }

        foreach (var entry in locString.Variables)
        {
            if (!string.Equals(entry.Key, tokenName, StringComparison.OrdinalIgnoreCase))
            {
                continue;
            }

            resolvedValue = entry.Value is DynamicVar dynamicVar
                ? GetPreferredDynamicVarValue(dynamicVar)
                : entry.Value;
            return resolvedValue is not null;
        }

        return false;
    }

    private static bool TryResolveStandalonePlaceholderToken(
        string tokenName,
        string formatHint,
        out string resolvedText)
    {
        resolvedText = string.Empty;

        if (TryResolveIconPlaceholderToken(tokenName, formatHint, out resolvedText))
        {
            return true;
        }

        return false;
    }

    private static bool TryResolveIconPlaceholderToken(
        string tokenName,
        string formatHint,
        out string resolvedText)
    {
        resolvedText = string.Empty;

        if (string.Equals(tokenName, "singleStarIcon", StringComparison.OrdinalIgnoreCase))
        {
            resolvedText = "点星辉";
            return true;
        }

        if (string.Equals(tokenName, "singleEnergyIcon", StringComparison.OrdinalIgnoreCase))
        {
            resolvedText = "点能量";
            return true;
        }

        if (formatHint.Contains("energyIcons(", StringComparison.OrdinalIgnoreCase))
        {
            resolvedText = "能量";
            return true;
        }

        if (formatHint.Contains("starIcons(", StringComparison.OrdinalIgnoreCase))
        {
            resolvedText = "星辉";
            return true;
        }

        return false;
    }

    private static IEnumerable<object?> EnumeratePlaceholderContexts(object placeholderContext)
    {
        if (placeholderContext is IEnumerable enumerable &&
            placeholderContext is not string &&
            placeholderContext is not LocString)
        {
            foreach (var item in enumerable)
            {
                if (item is null)
                {
                    continue;
                }

                foreach (var nestedContext in EnumeratePlaceholderContexts(item))
                {
                    yield return nestedContext;
                }
            }

            yield break;
        }

        yield return placeholderContext;

        if (GetHiddenPropertyObjectValue(placeholderContext, "CanonicalModel") is { } canonicalModel)
        {
            foreach (var nestedContext in EnumeratePlaceholderContexts(canonicalModel))
            {
                yield return nestedContext;
            }
        }

        if (GetHiddenPropertyObjectValue(placeholderContext, "Model") is { } model)
        {
            foreach (var nestedContext in EnumeratePlaceholderContexts(model))
            {
                yield return nestedContext;
            }
        }

        if (GetHiddenPropertyObjectValue(placeholderContext, "Info") is { } info)
        {
            foreach (var nestedContext in EnumeratePlaceholderContexts(info))
            {
                yield return nestedContext;
            }
        }
    }

    private static object? TryGetPreferredDescriptionValue(object model)
    {
        var preferredPropertyNames = new[]
        {
            "DynamicDescription",
            "RemoteDescription",
            "DynamicEventDescription",
            "EventDescription",
            "StaticDescription",
            "DescriptionLocString",
            "Description"
        };

        foreach (var propertyName in preferredPropertyNames)
        {
            var property = FindProperty(model.GetType(), propertyName);
            if (property is null)
            {
                continue;
            }

            object? value;
            try
            {
                value = property.GetValue(model);
            }
            catch
            {
                continue;
            }

            if (HasMeaningfulDescriptionText(value))
            {
                return value;
            }
        }

        return null;
    }

    private static string TryGetNamedTextValue(object model, params string[] memberNames)
    {
        foreach (var memberName in memberNames)
        {
            var property = FindProperty(model.GetType(), memberName);
            if (property is null)
            {
                continue;
            }

            object? value;
            try
            {
                value = property.GetValue(model);
            }
            catch
            {
                continue;
            }

            var text = DescribeText(value, model);
            if (!string.IsNullOrWhiteSpace(text))
            {
                return text;
            }
        }

        return string.Empty;
    }

    private static bool HasMeaningfulDescriptionText(object? value)
    {
        if (value is null)
        {
            return false;
        }

        var rawText = TextOfRawFirst(value, allowFormattedFallback: false);
        if (!string.IsNullOrWhiteSpace(rawText))
        {
            return true;
        }

        var text = value.ToString() ?? string.Empty;
        return !string.IsNullOrWhiteSpace(text) &&
               !LooksLikeTypeName(text, value.GetType());
    }

    private static string TryInvokeTextMethod(object value, string methodName)
    {
        var method = FindMethod(value.GetType(), methodName, 0);
        if (method is null)
        {
            return string.Empty;
        }

        try
        {
            return TextOf(method.Invoke(value, Array.Empty<object>()));
        }
        catch
        {
            return string.Empty;
        }
    }

    private static bool LooksLikeTypeName(string text, Type type)
    {
        if (string.IsNullOrWhiteSpace(text))
        {
            return false;
        }

        return string.Equals(text, type.FullName, StringComparison.Ordinal) ||
               string.Equals(text, type.Name, StringComparison.Ordinal);
    }

    private static bool TryResolvePlaceholderValueFromTypeHierarchy(
        object candidate,
        string tokenName,
        out object? resolvedValue)
    {
        resolvedValue = null;
        var segments = tokenName
            .Split('.', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);

        if (segments.Length > 1)
        {
            return TryResolvePlaceholderValueFromMemberPath(candidate, segments, out resolvedValue);
        }

        return TryResolvePlaceholderMember(candidate, tokenName, out resolvedValue);
    }

    private static bool TryResolvePlaceholderValueFromMemberPath(
        object candidate,
        IReadOnlyList<string> segments,
        out object? resolvedValue)
    {
        resolvedValue = null;
        object? currentValue = candidate;

        foreach (var segment in segments)
        {
            if (currentValue is null ||
                !TryResolvePlaceholderMember(currentValue, segment, out currentValue))
            {
                resolvedValue = null;
                return false;
            }
        }

        resolvedValue = currentValue;
        return resolvedValue is not null;
    }

    private static bool TryResolvePlaceholderMember(
        object candidate,
        string memberName,
        out object? resolvedValue)
    {
        resolvedValue = null;
        var type = candidate.GetType();
        var normalizedMemberName = NormalizePlaceholderMemberName(memberName);

        while (type is not null)
        {
            foreach (var property in type.GetProperties(BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.DeclaredOnly))
            {
                if (!string.Equals(
                        NormalizePlaceholderMemberName(property.Name),
                        normalizedMemberName,
                        StringComparison.OrdinalIgnoreCase))
                {
                    continue;
                }

                try
                {
                    resolvedValue = property.GetValue(candidate);
                    return resolvedValue is not null;
                }
                catch (Exception ex)
                {
                    BridgeDebugTrace.Write(
                        $"placeholder_property_read_failed type={type.FullName} member={property.Name}: {ex.GetBaseException().Message}");
                }            }

            foreach (var field in type.GetFields(BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.DeclaredOnly))
            {
                if (!string.Equals(
                        NormalizePlaceholderMemberName(field.Name),
                        normalizedMemberName,
                        StringComparison.OrdinalIgnoreCase))
                {
                    continue;
                }

                try
                {
                    resolvedValue = field.GetValue(candidate);
                    return resolvedValue is not null;
                }
                catch (Exception ex)
                {
                    BridgeDebugTrace.Write(
                        $"placeholder_field_read_failed type={type.FullName} member={field.Name}: {ex.GetBaseException().Message}");
                }            }

            type = type.BaseType;
        }

        return false;
    }

    private static bool TryResolvePlaceholderValueFromCanonicalVars(
        object candidate,
        string tokenName,
        out object? resolvedValue)
    {
        resolvedValue = null;

        if (FindProperty(candidate.GetType(), "CanonicalVars")?.GetValue(candidate) is not IEnumerable canonicalVars)
        {
            return false;
        }

        foreach (var dynamicVar in canonicalVars)
        {
            if (!TryGetDynamicVarName(dynamicVar, out var dynamicVarName) ||
                !string.Equals(dynamicVarName, tokenName, StringComparison.OrdinalIgnoreCase))
            {
                continue;
            }

            resolvedValue = GetPreferredDynamicVarValue(dynamicVar);
            return resolvedValue is not null;
        }

        return false;
    }

    private static bool TryResolvePlaceholderValueFromDynamicVars(
        object candidate,
        string tokenName,
        out object? resolvedValue)
    {
        resolvedValue = null;

        var dynamicVars = FindProperty(candidate.GetType(), "DynamicVars")?.GetValue(candidate);
        if (dynamicVars is null)
        {
            return false;
        }

        var tryGetValueMethod = FindMethod(dynamicVars.GetType(), "TryGetValue", 2);
        if (tryGetValueMethod is null)
        {
            return false;
        }

        var parameters = new object?[] { tokenName, null };

        try
        {
            if (tryGetValueMethod.Invoke(dynamicVars, parameters) is true &&
                parameters[1] is { } dynamicVar)
            {
                resolvedValue = GetPreferredDynamicVarValue(dynamicVar);
                return resolvedValue is not null;
            }
        }
        catch (Exception ex)
        {
            BridgeDebugTrace.Write(
                $"placeholder_dynamic_var_lookup_failed token={tokenName} vars_type={dynamicVars.GetType().FullName}: {ex.GetBaseException().Message}");
        }
        return false;
    }

    private static bool TryGetDynamicVarName(object? dynamicVar, out string name)
    {
        name = string.Empty;

        if (dynamicVar is null)
        {
            return false;
        }

        var property = FindProperty(dynamicVar.GetType(), "Name");
        name = TextOf(property?.GetValue(dynamicVar));
        return !string.IsNullOrWhiteSpace(name);
    }

    private static object? GetPreferredDynamicVarValue(object dynamicVar)
    {
        if (dynamicVar is null)
        {
            return null;
        }

        var previewValue = GetHiddenPropertyValue<decimal>(dynamicVar, "PreviewValue");
        if (previewValue.HasValue && previewValue.Value != 0m)
        {
            return decimal.Truncate(previewValue.Value) == previewValue.Value
                ? (int)previewValue.Value
                : previewValue.Value;
        }

        var intValue = GetHiddenPropertyValue<int>(dynamicVar, "IntValue");
        if (intValue.HasValue)
        {
            return intValue.Value;
        }

        var baseValue = GetHiddenPropertyValue<decimal>(dynamicVar, "BaseValue");
        if (baseValue.HasValue)
        {
            return decimal.Truncate(baseValue.Value) == baseValue.Value
                ? (int)baseValue.Value
                : baseValue.Value;
        }

        return dynamicVar;
    }

    private static string NormalizePlaceholderMemberName(string? memberName)
    {
        if (string.IsNullOrWhiteSpace(memberName))
        {
            return string.Empty;
        }

        var normalized = memberName.Trim();
        const string backingFieldSuffix = ">k__BackingField";

        if (normalized.StartsWith('<') &&
            normalized.EndsWith(backingFieldSuffix, StringComparison.Ordinal))
        {
            normalized = normalized.Substring(1, normalized.Length - backingFieldSuffix.Length - 1);
        }

        return normalized.TrimStart('_');
    }

    private static string FormatResolvedPlaceholderValue(
        string tokenName,
        string formatHint,
        object? resolvedValue)
    {
        if (resolvedValue is null)
        {
            return string.Empty;
        }

        if (TryResolveConditionalPlaceholderText(formatHint, resolvedValue, out var conditionalText))
        {
            return conditionalText;
        }

        if (!string.IsNullOrWhiteSpace(formatHint))
        {
            if (formatHint.Contains("abs()", StringComparison.OrdinalIgnoreCase) &&
                TryConvertToDecimal(resolvedValue, out var absoluteValue))
            {
                return FormatNumericValue(decimal.Abs(absoluteValue));
            }

            if (formatHint.Contains("percentMore()", StringComparison.OrdinalIgnoreCase) &&
                TryConvertToDecimal(resolvedValue, out var percentValue))
            {
                var normalizedPercent = percentValue is >= -1m and <= 1m
                    ? percentValue * 100m
                    : percentValue;
                return FormatNumericValue(normalizedPercent);
            }

            if (formatHint.Contains("energyIcons(", StringComparison.OrdinalIgnoreCase))
            {
                return TryConvertToInt(resolvedValue, out var energyAmount)
                    ? $"{energyAmount}点能量"
                    : "能量";
            }

            if (formatHint.Contains("starIcons(", StringComparison.OrdinalIgnoreCase))
            {
                return TryConvertToInt(resolvedValue, out var starAmount)
                    ? $"{starAmount}点星辉"
                    : "星辉";
            }
        }

        if (string.Equals(tokenName, "singleStarIcon", StringComparison.OrdinalIgnoreCase))
        {
            return "点星辉";
        }

        if (string.Equals(tokenName, "singleEnergyIcon", StringComparison.OrdinalIgnoreCase))
        {
            return "点能量";
        }

        return FormatPlainPlaceholderValue(resolvedValue);
    }

    private static bool TryResolveConditionalPlaceholderText(
        string formatHint,
        object resolvedValue,
        out string resolvedText)
    {
        resolvedText = string.Empty;

        if (string.IsNullOrWhiteSpace(formatHint) ||
            !formatHint.StartsWith("cond:", StringComparison.OrdinalIgnoreCase))
        {
            return false;
        }

        var expression = formatHint["cond:".Length..];
        if (expression.Length == 0)
        {
            return false;
        }

        var segments = expression.Split('|');
        var fallbackSegments = new List<string>();

        foreach (var segment in segments)
        {
            var questionMarkIndex = segment.IndexOf('?');
            if (questionMarkIndex <= 0)
            {
                fallbackSegments.Add(segment);
                continue;
            }

            var condition = segment[..questionMarkIndex].Trim();
            var output = segment[(questionMarkIndex + 1)..];
            if (!EvaluatePlaceholderCondition(condition, resolvedValue))
            {
                continue;
            }

            resolvedText = ReplaceConditionalTemplateValue(output, resolvedValue);
            return true;
        }

        if (fallbackSegments.Count <= 0)
        {
            return false;
        }

        var selectedFallback = fallbackSegments.Count == 1
            ? fallbackSegments[0]
            : (IsTruthyPlaceholderValue(resolvedValue) ? fallbackSegments[0] : fallbackSegments[^1]);
        resolvedText = ReplaceConditionalTemplateValue(selectedFallback, resolvedValue);
        return true;
    }

    private static bool EvaluatePlaceholderCondition(string condition, object resolvedValue)
    {
        if (string.IsNullOrWhiteSpace(condition))
        {
            return IsTruthyPlaceholderValue(resolvedValue);
        }

        var trimmedCondition = condition.Trim();
        if (trimmedCondition.StartsWith(">=", StringComparison.Ordinal) &&
            TryConvertToDecimal(resolvedValue, out var greaterOrEqualValue) &&
            decimal.TryParse(trimmedCondition[2..], out var greaterOrEqualTarget))
        {
            return greaterOrEqualValue >= greaterOrEqualTarget;
        }

        if (trimmedCondition.StartsWith("<=", StringComparison.Ordinal) &&
            TryConvertToDecimal(resolvedValue, out var lessOrEqualValue) &&
            decimal.TryParse(trimmedCondition[2..], out var lessOrEqualTarget))
        {
            return lessOrEqualValue <= lessOrEqualTarget;
        }

        if (trimmedCondition.StartsWith("==", StringComparison.Ordinal) &&
            TryConvertToDecimal(resolvedValue, out var equalValue) &&
            decimal.TryParse(trimmedCondition[2..], out var equalTarget))
        {
            return equalValue == equalTarget;
        }

        if (trimmedCondition.StartsWith("!=", StringComparison.Ordinal) &&
            TryConvertToDecimal(resolvedValue, out var notEqualValue) &&
            decimal.TryParse(trimmedCondition[2..], out var notEqualTarget))
        {
            return notEqualValue != notEqualTarget;
        }

        if (trimmedCondition.StartsWith(">", StringComparison.Ordinal) &&
            TryConvertToDecimal(resolvedValue, out var greaterValue) &&
            decimal.TryParse(trimmedCondition[1..], out var greaterTarget))
        {
            return greaterValue > greaterTarget;
        }

        if (trimmedCondition.StartsWith("<", StringComparison.Ordinal) &&
            TryConvertToDecimal(resolvedValue, out var lessValue) &&
            decimal.TryParse(trimmedCondition[1..], out var lessTarget))
        {
            return lessValue < lessTarget;
        }

        return string.Equals(
            NormalizeComparableText(FormatPlainPlaceholderValue(resolvedValue)),
            NormalizeComparableText(trimmedCondition),
            StringComparison.OrdinalIgnoreCase);
    }

    private static bool IsTruthyPlaceholderValue(object? value)
    {
        if (value is null)
        {
            return false;
        }

        return value switch
        {
            bool boolValue => boolValue,
            string stringValue => !string.IsNullOrWhiteSpace(stringValue),
            _ when TryConvertToDecimal(value, out var numericValue) => numericValue != 0m,
            _ => true
        };
    }

    private static string ReplaceConditionalTemplateValue(string template, object resolvedValue)
    {
        if (string.IsNullOrEmpty(template))
        {
            return string.Empty;
        }

        return template.Replace("{}", FormatPlainPlaceholderValue(resolvedValue), StringComparison.Ordinal);
    }

    private static string FormatPlainPlaceholderValue(object resolvedValue)
    {
        if (TryConvertToDecimal(resolvedValue, out var numericValue))
        {
            return FormatNumericValue(numericValue);
        }

        var rawText = TextOfRawFirst(resolvedValue, allowFormattedFallback: false);
        if (!string.IsNullOrWhiteSpace(rawText))
        {
            return rawText;
        }

        return TextOf(resolvedValue);
    }

    private static string FormatNumericValue(decimal number)
    {
        var normalized = decimal.Truncate(number) == number
            ? decimal.Truncate(number)
            : number;
        return normalized.ToString(CultureInfo.InvariantCulture);
    }

    private static bool TryConvertToDecimal(object value, out decimal number)
    {
        switch (value)
        {
            case byte byteValue:
                number = byteValue;
                return true;
            case sbyte sbyteValue:
                number = sbyteValue;
                return true;
            case short shortValue:
                number = shortValue;
                return true;
            case ushort ushortValue:
                number = ushortValue;
                return true;
            case int intValue:
                number = intValue;
                return true;
            case uint uintValue:
                number = uintValue;
                return true;
            case long longValue:
                number = longValue;
                return true;
            case ulong ulongValue:
                number = ulongValue;
                return true;
            case decimal decimalValue:
                number = decimalValue;
                return true;
            case float floatValue:
                number = (decimal)floatValue;
                return true;
            case double doubleValue:
                number = (decimal)doubleValue;
                return true;
            case string stringValue when decimal.TryParse(stringValue, NumberStyles.Any, CultureInfo.InvariantCulture, out var parsedNumber):
                number = parsedNumber;
                return true;
            default:
                number = 0m;
                return false;
        }
    }

    private static bool TryConvertToInt(object value, out int number)
    {
        switch (value)
        {
            case byte byteValue:
                number = byteValue;
                return true;
            case sbyte sbyteValue:
                number = sbyteValue;
                return true;
            case short shortValue:
                number = shortValue;
                return true;
            case ushort ushortValue:
                number = ushortValue;
                return true;
            case int intValue:
                number = intValue;
                return true;
            case uint uintValue when uintValue <= int.MaxValue:
                number = (int)uintValue;
                return true;
            case long longValue when longValue is >= int.MinValue and <= int.MaxValue:
                number = (int)longValue;
                return true;
            case ulong ulongValue when ulongValue <= int.MaxValue:
                number = (int)ulongValue;
                return true;
            case decimal decimalValue when decimalValue >= int.MinValue && decimalValue <= int.MaxValue:
                number = (int)decimal.Truncate(decimalValue);
                return true;
            case float floatValue when floatValue >= int.MinValue && floatValue <= int.MaxValue:
                number = (int)MathF.Truncate(floatValue);
                return true;
            case double doubleValue when doubleValue >= int.MinValue && doubleValue <= int.MaxValue:
                number = (int)Math.Truncate(doubleValue);
                return true;
            case string stringValue when int.TryParse(stringValue, out var parsedNumber):
                number = parsedNumber;
                return true;
            default:
                number = 0;
                return false;
        }
    }

    private static bool TryConvertToULong(object? value, out ulong number)
    {
        switch (value)
        {
            case byte byteValue:
                number = byteValue;
                return true;
            case sbyte sbyteValue when sbyteValue >= 0:
                number = (ulong)sbyteValue;
                return true;
            case short shortValue when shortValue >= 0:
                number = (ulong)shortValue;
                return true;
            case ushort ushortValue:
                number = ushortValue;
                return true;
            case int intValue when intValue >= 0:
                number = (ulong)intValue;
                return true;
            case uint uintValue:
                number = uintValue;
                return true;
            case long longValue when longValue >= 0:
                number = (ulong)longValue;
                return true;
            case ulong ulongValue:
                number = ulongValue;
                return true;
            case decimal decimalValue when decimalValue >= 0m && decimalValue <= ulong.MaxValue:
                number = (ulong)decimal.Truncate(decimalValue);
                return true;
            case float floatValue when floatValue >= 0f && floatValue <= ulong.MaxValue:
                number = (ulong)MathF.Truncate(floatValue);
                return true;
            case double doubleValue when doubleValue >= 0d && doubleValue <= ulong.MaxValue:
                number = (ulong)Math.Truncate(doubleValue);
                return true;
            case string stringValue when ulong.TryParse(stringValue, out var parsedNumber):
                number = parsedNumber;
                return true;
            default:
                number = 0UL;
                return false;
        }
    }

    private static IReadOnlyList<string> CollectLocalVisibleText(Node? root, int maxCount, int maxDepth = 1)
    {
        if (root is null || maxCount <= 0 || maxDepth < 0)
        {
            return Array.Empty<string>();
        }

        var texts = new List<string>(maxCount);
        var seenTexts = new HashSet<string>(StringComparer.Ordinal);

        void Visit(Node node, int depth)
        {
            if (texts.Count >= maxCount ||
                depth > maxDepth ||
                !GodotObject.IsInstanceValid(node) ||
                !IsNodeVisible(node))
            {
                return;
            }

            var text = TryGetOwnNodeText(node);
            if (!string.IsNullOrWhiteSpace(text) && seenTexts.Add(text))
            {
                texts.Add(text);
                if (texts.Count >= maxCount)
                {
                    return;
                }
            }

            if (depth >= maxDepth)
            {
                return;
            }

            foreach (Node child in node.GetChildren())
            {
                Visit(child, depth + 1);
                if (texts.Count >= maxCount)
                {
                    return;
                }
            }
        }

        Visit(root, 0);
        return texts;
    }

    private static string TryGetLocalNodeText(Node? node)
    {
        return CollectLocalVisibleText(node, 1, maxDepth: 1).FirstOrDefault() ?? string.Empty;
    }

    private static string TryGetOwnNodeText(Node node)
    {
        var propertyNames = new[]
        {
            "Text",
            "Title",
            "Label",
            "Subtitle",
            "Description",
            "CurrentText",
            "Value"
        };

        foreach (var propertyName in propertyNames)
        {
            var text = TryGetTextFromValue(GetHiddenPropertyObjectValue(node, propertyName));
            if (!string.IsNullOrWhiteSpace(text))
            {
                return text;
            }
        }

        var fieldNames = new[]
        {
            "_label",
            "_title",
            "_text",
            "Label",
            "Title",
            "Text"
        };

        foreach (var fieldName in fieldNames)
        {
            var text = TryGetTextFromValue(GetHiddenFieldValue(node, fieldName));
            if (!string.IsNullOrWhiteSpace(text))
            {
                return text;
            }
        }

        if (node is Label label)
        {
            return DescribeText(label.Text);
        }

        if (node is RichTextLabel richTextLabel)
        {
            return DescribeText(richTextLabel.Text);
        }

        return string.Empty;
    }

    private static string TryGetTextFromValue(object? value)
    {
        return value switch
        {
            null => string.Empty,
            string text => DescribeText(text),
            Node node => TryGetLocalNodeText(node),
            _ when value is System.Collections.IEnumerable => string.Empty,
            _ => DescribeText(value)
        };
    }

    private static string NormalizeComparableText(string? text)
    {
        if (string.IsNullOrWhiteSpace(text))
        {
            return string.Empty;
        }

        var stripped = StripBbCode(text.ReplaceLineEndings("\n"));
        var builder = new StringBuilder(stripped.Length);
        var previousWasWhitespace = false;

        foreach (var rune in stripped.Trim())
        {
            if (char.IsWhiteSpace(rune))
            {
                if (!previousWasWhitespace)
                {
                    builder.Append(' ');
                    previousWasWhitespace = true;
                }

                continue;
            }

            builder.Append(rune);
            previousWasWhitespace = false;
        }

        return builder.ToString();
    }

    private static string StripBbCode(string text)
    {
        if (string.IsNullOrEmpty(text))
        {
            return string.Empty;
        }

        var builder = new StringBuilder(text.Length);
        var insideTag = false;

        foreach (var character in text)
        {
            if (character == '[')
            {
                insideTag = true;
                continue;
            }

            if (character == ']')
            {
                insideTag = false;
                continue;
            }

            if (!insideTag)
            {
                builder.Append(character);
            }
        }

        return builder.ToString();
    }

    private static string? TryGetMainMenuSemanticAction(string text)
    {
        var normalized = NormalizeMenuText(text);
        if (string.IsNullOrEmpty(normalized))
        {
            return null;
        }

        if (normalized.Contains("continue") || normalized.Contains("继续游戏"))
        {
            return "continue";
        }

        if (normalized.Contains("abandoncurrentgame") || normalized.Contains("放弃当前游戏"))
        {
            return "abandon_current_game";
        }

        if (normalized.Contains("newgame") || normalized.Contains("新游戏"))
        {
            return "new_game";
        }

        if (normalized.Contains("singleplayer") || normalized.Contains("单人模式"))
        {
            return "singleplayer";
        }

        if (normalized.Contains("multiplayer") || normalized.Contains("多人模式"))
        {
            return "multiplayer";
        }

        if (normalized.Contains("timeline") || normalized.Contains("时间线"))
        {
            return "timeline";
        }

        if (normalized.Contains("settings") || normalized.Contains("设置"))
        {
            return "settings";
        }

        if (normalized.Contains("compendium") || normalized.Contains("百科大全"))
        {
            return "compendium";
        }

        if (normalized.Contains("quit") || normalized.Contains("exit") || normalized.Contains("退出"))
        {
            return "quit";
        }

        return null;
    }

    private static string? TryGetAbandonConfirmSemanticAction(string text)
    {
        var normalized = NormalizeMenuText(text);
        if (string.IsNullOrEmpty(normalized))
        {
            return null;
        }

        if (normalized.Contains("cancel") ||
            normalized.Contains("取消") ||
            normalized.Contains("不了") ||
            normalized.Equals("否", StringComparison.Ordinal))
        {
            return "cancel";
        }

        if (normalized.Contains("confirm") ||
            normalized.Contains("abandon") ||
            normalized.Contains("确认") ||
            normalized.Contains("好的") ||
            normalized.Contains("放弃") ||
            normalized.Equals("是", StringComparison.Ordinal))
        {
            return "confirm";
        }

        return null;
    }

    private static string NormalizeMenuText(string text)
    {
        return string.Concat(text.Where(static character => !char.IsWhiteSpace(character))).ToLowerInvariant();
    }

}
