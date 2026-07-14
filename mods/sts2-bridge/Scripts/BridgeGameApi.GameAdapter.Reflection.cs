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
    private static object? InvokeParameterless(object? target, string methodName)
    {
        if (target is null)
        {
            return null;
        }

        var method = FindMethod(target.GetType(), methodName, 0);
        return method?.Invoke(target, Array.Empty<object>());
    }

    private static void ExecuteGameActionSynchronously(object action)
    {
        var result = InvokeParameterless(action, "ExecuteAction");
        if (result is Task task)
        {
            task.GetAwaiter().GetResult();
            return;
        }

        throw new BridgeRequestException(
            HttpStatusCode.Conflict,
            "action_execution_failed",
            $"Could not execute game action {action.GetType().FullName}.");
    }

    private static bool? TryInvokeBoolean(object? target, string methodName, params object?[] arguments)
    {
        if (target is null)
        {
            return null;
        }

        try
        {
            var method = FindMethod(target.GetType(), methodName, arguments.Length);
            if (method is null)
            {
                return null;
            }

            var result = method.Invoke(target, arguments);
            return result is bool boolResult ? boolResult : null;
        }
        catch
        {
            return null;
        }
    }

    private static bool TryInvokeParameterless(object? target, string methodName)
    {
        if (target is null)
        {
            return false;
        }

        var method = FindMethod(target.GetType(), methodName, 0);
        if (method is null)
        {
            return false;
        }

        method.Invoke(target, Array.Empty<object>());
        return true;
    }

    private static bool TryInvokeCardSelectionCompleteSelection(Node? cardSelectionScreen)
    {
        if (cardSelectionScreen is null)
        {
            return false;
        }

        var method = FindMethod(cardSelectionScreen.GetType(), "CompleteSelection", 0);
        if (method is null)
        {
            return false;
        }

        try
        {
            method.Invoke(cardSelectionScreen, Array.Empty<object>());
            return true;
        }
        catch (TargetInvocationException ex) when (IsBenignCardSelectionCompletionException(ex.InnerException))
        {
            return true;
        }
    }

    private static bool IsBenignCardSelectionCompletionException(Exception? exception)
    {
        return exception is InvalidOperationException invalidOperationException &&
               invalidOperationException.Message.Contains(
                   "transition a task to a final state",
                   StringComparison.OrdinalIgnoreCase);
    }

    private static bool TryInvokeSingleArgument(object? target, string methodName, object argument)
    {
        if (target is null)
        {
            return false;
        }

        var method = FindMethod(target.GetType(), methodName, 1);
        if (method is null)
        {
            return false;
        }

        method.Invoke(target, new[] { argument });
        return true;
    }

    private static bool TryInvokeTwoArguments(object? target, string methodName, object firstArgument, object secondArgument)
    {
        if (target is null)
        {
            return false;
        }

        var method = FindMethod(target.GetType(), methodName, 2);
        if (method is null)
        {
            return false;
        }

        method.Invoke(target, new[] { firstArgument, secondArgument });
        return true;
    }


    private static MethodInfo? FindMethod(Type? type, string methodName, int parameterCount)
    {
        while (type is not null)
        {
            var method = type
                .GetMethods(BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.DeclaredOnly)
                .FirstOrDefault(candidate =>
                    candidate.Name.Equals(methodName, StringComparison.Ordinal) &&
                    candidate.GetParameters().Length == parameterCount);

            if (method is not null)
            {
                return method;
            }

            type = type.BaseType;
        }

        return null;
    }

    private static T? GetHiddenPropertyValue<T>(object? target, string propertyName) where T : struct
    {
        if (target is null)
        {
            return null;
        }

        var (property, staticTarget) = target is Type staticType
            ? (FindProperty(staticType, propertyName, includeStatic: true), (object?)null)
            : (FindProperty(target.GetType(), propertyName), target);
        if (property is null)
        {
            return null;
        }

        var value = property.GetValue(staticTarget);
        return value is T typed ? typed : null;
    }

    private static object? GetHiddenPropertyObjectValue(object? target, string propertyName)
    {
        if (target is null)
        {
            return null;
        }

        var (property, staticTarget) = target is Type staticType
            ? (FindProperty(staticType, propertyName, includeStatic: true), (object?)null)
            : (FindProperty(target.GetType(), propertyName), target);
        return property?.GetValue(staticTarget);
    }

    private static object? GetHiddenFieldValue(object? target, string fieldName)
    {
        if (target is null)
        {
            return null;
        }

        var (field, staticTarget) = target is Type staticType
            ? (FindField(staticType, fieldName, includeStatic: true), (object?)null)
            : (FindField(target.GetType(), fieldName), target);
        return field?.GetValue(staticTarget);
    }

    private static int CountSelectedDeckUpgradeCards(NDeckUpgradeSelectScreen? deckUpgradeScreen)
    {
        return GetSelectedDeckUpgradeCards(deckUpgradeScreen).Count;
    }

    private static bool IsDeckUpgradeCardSelected(NDeckUpgradeSelectScreen? deckUpgradeScreen, CardModel? card)
    {
        if (card is null)
        {
            return false;
        }

        return GetSelectedDeckUpgradeCards(deckUpgradeScreen).Any(selected => ReferenceEquals(selected, card));
    }

    private static List<object> GetSelectedDeckUpgradeCards(NDeckUpgradeSelectScreen? deckUpgradeScreen)
    {
        if (GetHiddenFieldValue(deckUpgradeScreen, "_selectedCards") is not IEnumerable selectedCards)
        {
            return new List<object>();
        }

        return selectedCards
            .Cast<object?>()
            .Where(static selected => selected is not null)
            .Cast<object>()
            .ToList();
    }

    private static bool IsDeckUpgradePreviewHolder(NDeckUpgradeSelectScreen deckUpgradeScreen, NCardHolder holder)
    {
        return IsDescendantOf(holder, GetHiddenFieldValue(deckUpgradeScreen, "_singlePreview") as Node) ||
               IsDescendantOf(holder, GetHiddenFieldValue(deckUpgradeScreen, "_multiPreview") as Node) ||
               IsDescendantOf(holder, GetHiddenFieldValue(deckUpgradeScreen, "_upgradeSinglePreviewContainer") as Node) ||
               IsDescendantOf(holder, GetHiddenFieldValue(deckUpgradeScreen, "_upgradeMultiPreviewContainer") as Node);
    }

    private static bool IsDescendantOf(Node? node, Node? ancestor)
    {
        if (node is null || ancestor is null)
        {
            return false;
        }

        var current = node.GetParent();
        while (current is not null)
        {
            if (ReferenceEquals(current, ancestor))
            {
                return true;
            }

            current = current.GetParent();
        }

        return false;
    }

    private static Node? ResolveVisibleHoverTipSet(NGame? game)
    {
        var hoverTipsContainer = game?.HoverTipsContainer ??
                                 GetHiddenPropertyObjectValue(game, "HoverTipsContainer") as Node ??
                                 GetHiddenFieldValue(game, "HoverTipsContainer") as Node;
        if (hoverTipsContainer is null || !GodotObject.IsInstanceValid(hoverTipsContainer))
        {
            return null;
        }

        var visibleImmediateSets = SortByVisualPosition(
            hoverTipsContainer
                .GetChildren()
                .OfType<Node>()
                .Where(static child =>
                    IsNodeVisible(child) &&
                    IsTypeFullName(child, "MegaCrit.Sts2.Core.Nodes.HoverTips.NHoverTipSet")));
        if (visibleImmediateSets.Count > 0)
        {
            return visibleImmediateSets.Last();
        }

        return null;
    }

    private static T? ResolveFirstVisibleNode<T>(params T?[] candidates) where T : Node
    {
        return candidates.FirstOrDefault(IsNodeVisible);
    }

    private static T? ResolveFirstVisibleEnabledNode<T>(params T?[] candidates) where T : Node
    {
        return candidates.FirstOrDefault(candidate => IsNodeVisible(candidate) && IsButtonEnabled(candidate));
    }

    private static string[] CollectButtonPayloadTexts(Node? button, int maxCount = 4)
    {
        return CollectLocalVisibleText(button, maxCount, maxDepth: 1).ToArray();
    }

    private static string[] CollectCardSelectionSurfaceTexts(
        Node? cardSelectionScreen,
        string? prompt,
        Node? cardSelectionConfirmButton,
        Node? cardSelectionCancelButton,
        Node? cardSelectionCloseButton,
        Node? cardSelectionSkipButton)
    {
        if (cardSelectionScreen is null || !IsNodeVisible(cardSelectionScreen))
        {
            return string.IsNullOrWhiteSpace(prompt)
                ? Array.Empty<string>()
                : new[] { prompt.ReplaceLineEndings("\n").Trim() };
        }

        return CollectPromptAndNodeTexts(
            prompt,
            8,
            cardSelectionConfirmButton,
            cardSelectionCancelButton,
            cardSelectionCloseButton,
            cardSelectionSkipButton);
    }

    private static string[] CollectDeckUpgradeSurfaceTexts(
        NDeckUpgradeSelectScreen? deckUpgradeScreen,
        string? prompt,
        Node? deckUpgradeConfirmButton,
        Node? deckUpgradeCancelButton,
        Node? deckUpgradeCloseButton)
    {
        if (deckUpgradeScreen is null || !IsNodeVisible(deckUpgradeScreen))
        {
            return string.IsNullOrWhiteSpace(prompt)
                ? Array.Empty<string>()
                : new[] { prompt.ReplaceLineEndings("\n").Trim() };
        }

        return CollectPromptAndNodeTexts(
            prompt,
            8,
            deckUpgradeConfirmButton,
            deckUpgradeCancelButton,
            deckUpgradeCloseButton);
    }

    private static string[] CollectPromptAndNodeTexts(string? prompt, int maxCount, params Node?[] nodes)
    {
        if (maxCount <= 0)
        {
            return Array.Empty<string>();
        }

        var texts = new List<string>(maxCount);
        var seen = new HashSet<string>(StringComparer.Ordinal);

        void AddText(string? text)
        {
            var normalized = text?.ReplaceLineEndings("\n").Trim();
            if (string.IsNullOrWhiteSpace(normalized) || !seen.Add(normalized))
            {
                return;
            }

            texts.Add(normalized);
        }

        AddText(prompt);

        foreach (var node in nodes)
        {
            foreach (var text in CollectLocalVisibleText(node, Math.Max(0, maxCount - texts.Count), maxDepth: 1))
            {
                AddText(text);
                if (texts.Count >= maxCount)
                {
                    return texts.ToArray();
                }
            }
        }

        return texts.ToArray();
    }

    private static Node? FindVisibleImmediateChildByName(Node? root, string childName)
    {
        if (root is null || string.IsNullOrWhiteSpace(childName))
        {
            return null;
        }

        return root
            .GetChildren()
            .OfType<Node>()
            .FirstOrDefault(child =>
                IsNodeVisible(child) &&
                string.Equals(child.Name.ToString(), childName, StringComparison.Ordinal));
    }

    private static bool IsCardSelectionPreviewVisible(Node? cardSelectionScreen)
    {
        if (cardSelectionScreen is null)
        {
            return false;
        }

        return ResolveFirstVisibleNode(
                   GetHiddenFieldValue(cardSelectionScreen, "_previewContainer") as Node,
                   GetHiddenFieldValue(cardSelectionScreen, "_enchantSinglePreviewContainer") as Node,
                   GetHiddenFieldValue(cardSelectionScreen, "_enchantMultiPreviewContainer") as Node) is not null;
    }

    private static Node? ResolveCardSelectionConfirmButton(Node? cardSelectionScreen)
    {
        if (cardSelectionScreen is null)
        {
            return null;
        }

        if (IsCardSelectionPreviewVisible(cardSelectionScreen))
        {
            var previewConfirmButton = ResolveFirstVisibleEnabledNode(
                GetHiddenFieldValue(cardSelectionScreen, "_previewConfirmButton") as Node,
                GetHiddenFieldValue(cardSelectionScreen, "_singlePreviewConfirmButton") as Node,
                GetHiddenFieldValue(cardSelectionScreen, "_multiPreviewConfirmButton") as Node);
            if (previewConfirmButton is not null)
            {
                return previewConfirmButton;
            }
        }

        // NConfirmButton.Disable() slides the button off-screen but can remain visible in-tree,
        // so prefer candidates that are both visible and enabled.
        return ResolveFirstVisibleEnabledNode(
            GetHiddenFieldValue(cardSelectionScreen, "_confirmButton") as Node,
            GetHiddenFieldValue(cardSelectionScreen, "_previewConfirmButton") as Node,
            GetHiddenFieldValue(cardSelectionScreen, "_singlePreviewConfirmButton") as Node,
            GetHiddenFieldValue(cardSelectionScreen, "_multiPreviewConfirmButton") as Node,
            GetHiddenFieldValue(cardSelectionScreen, "_selectModeConfirmButton") as Node);
    }

    private static Node? ResolveCardSelectionCancelButton(Node? cardSelectionScreen)
    {
        if (cardSelectionScreen is null)
        {
            return null;
        }

        if (IsCardSelectionPreviewVisible(cardSelectionScreen))
        {
            var previewCancelButton = ResolveFirstVisibleEnabledNode(
                GetHiddenFieldValue(cardSelectionScreen, "_previewCancelButton") as Node,
                GetHiddenFieldValue(cardSelectionScreen, "_singlePreviewCancelButton") as Node,
                GetHiddenFieldValue(cardSelectionScreen, "_multiPreviewCancelButton") as Node);
            if (previewCancelButton is not null)
            {
                return previewCancelButton;
            }
        }

        return ResolveFirstVisibleEnabledNode(
            GetHiddenFieldValue(cardSelectionScreen, "_previewCancelButton") as Node,
            GetHiddenFieldValue(cardSelectionScreen, "_singlePreviewCancelButton") as Node,
            GetHiddenFieldValue(cardSelectionScreen, "_multiPreviewCancelButton") as Node);
    }

    private static Node? ResolveCombatHandSelectionNode(NPlayerHand? playerHand)
    {
        if (playerHand is null || !IsNodeVisible(playerHand))
        {
            return null;
        }

        return playerHand.IsInCardSelection ||
               GetHiddenPropertyValue<bool>(playerHand, "IsInCardSelection") == true
            ? playerHand
            : null;
    }

    private static bool IsCardSelectionRootCandidate(Node node)
    {
        if (node is NPlayerHand)
        {
            return true;
        }

        var fullName = node.GetType().FullName;
        return fullName is not null &&
               fullName.StartsWith("MegaCrit.Sts2.Core.Nodes.Screens.CardSelection.", StringComparison.Ordinal);
    }

    private static T? ResolveOverlayScreen<T>(IScreenContext? activeScreen, NOverlayStack? overlayStack)
        where T : Node, IScreenContext
    {
        if (activeScreen is T typedScreen)
        {
            return typedScreen;
        }

        return activeScreen is null
            ? overlayStack?.Peek() as T
            : null;
    }

    private static Node? ResolveStableCardSelectionScreen(
        IScreenContext? activeScreen,
        NOverlayStack? overlayStack,
        Node? cardRewardScreen,
        Node? deckUpgradeScreen,
        NPlayerHand? playerHand)
    {
        if (activeScreen is Node activeNode &&
            IsCardSelectionRootCandidate(activeNode) &&
            !IsSameNodeInstance(activeNode, cardRewardScreen) &&
            !IsSameNodeInstance(activeNode, deckUpgradeScreen))
        {
            return activeNode;
        }

        if (overlayStack?.Peek() is Node overlayNode &&
            IsCardSelectionRootCandidate(overlayNode) &&
            !IsSameNodeInstance(overlayNode, cardRewardScreen) &&
            !IsSameNodeInstance(overlayNode, deckUpgradeScreen))
        {
            return overlayNode;
        }

        return ResolveCombatHandSelectionNode(playerHand);
    }

    private static IReadOnlyList<NTreasureRoomRelicHolder> ResolveTreasureRelicOptions(
        NTreasureRoomRelicCollection? treasureRelicCollection)
    {
        if (treasureRelicCollection is null)
        {
            return Array.Empty<NTreasureRoomRelicHolder>();
        }

        return SortByVisualPosition(
                treasureRelicCollection
                    .GetChildren()
                    .OfType<NTreasureRoomRelicHolder>()
                    .Where(IsNodeVisible))
            .ToArray();
    }

    private static IReadOnlyList<NRestSiteButton> ResolveRestSiteButtons(NRestSiteRoom? restSiteRoom)
    {
        var choicesContainer = restSiteRoom?.GetNodeOrNull<Control>("%ChoicesContainer") ??
                               GetHiddenFieldValue(restSiteRoom, "_choicesContainer") as Control;
        if (choicesContainer is null)
        {
            return Array.Empty<NRestSiteButton>();
        }

        return SortByVisualPosition(
                choicesContainer
                    .GetChildren()
                    .OfType<NRestSiteButton>()
                    .Where(IsNodeVisible))
            .ToArray();
    }

    private static IReadOnlyList<NCharacterSelectButton> ResolveCharacterButtons(
        NCharacterSelectScreen? characterSelectScreen)
    {
        var buttonContainer = characterSelectScreen?.GetNodeOrNull<Control>("CharSelectButtons/ButtonContainer") ??
                              GetHiddenFieldValue(characterSelectScreen, "_charButtonContainer") as Control;
        if (buttonContainer is null)
        {
            return Array.Empty<NCharacterSelectButton>();
        }

        return SortByVisualPosition(
                buttonContainer
                    .GetChildren()
                    .OfType<NCharacterSelectButton>()
                    .Where(IsNodeVisible))
            .ToArray();
    }

    private static IReadOnlyList<NRewardButton> ResolveRewardButtons(NRewardsScreen? rewardsScreen)
    {
        if (rewardsScreen is null)
        {
            return Array.Empty<NRewardButton>();
        }

        if (GetHiddenFieldValue(rewardsScreen, "_rewardButtons") is IEnumerable rewardControls)
        {
            var buttons = rewardControls
                .Cast<object?>()
                .OfType<Control>()
                .Where(static control => GodotObject.IsInstanceValid(control))
                .Where(control => !IsRewardControlSkipped(rewardsScreen, control))
                .SelectMany(ExpandRewardButtonsFromControl)
                .Where(IsNodeVisible)
                .DistinctBy(static button => button.NativeInstance);

            return SortByVisualPosition(buttons).ToArray();
        }

        return SortByVisualPosition(
                FindVisibleDescendants<NRewardButton>(rewardsScreen)
                    .Where(button => !IsRewardButtonSkipped(rewardsScreen, button)))
            .ToArray();
    }

    private static IEnumerable<NRewardButton> ExpandRewardButtonsFromControl(Control control)
    {
        if (control is NRewardButton rewardButton)
        {
            yield return rewardButton;
            yield break;
        }

        foreach (var nestedRewardButton in FindVisibleDescendants<NRewardButton>(control))
        {
            yield return nestedRewardButton;
        }
    }

    private static IReadOnlyList<NCardHolder> ResolveCardRewardOptions(
        NCardRewardSelectionScreen? cardRewardScreen)
    {
        var cardRow = cardRewardScreen?.GetNodeOrNull<Control>("UI/CardRow") ??
                      GetHiddenFieldValue(cardRewardScreen, "_cardRow") as Control;
        if (cardRow is null)
        {
            return Array.Empty<NCardHolder>();
        }

        return SortByVisualPosition(
                cardRow
                    .GetChildren()
                    .OfType<NCardHolder>()
                    .Where(static holder => holder.CardModel is not null)
                    .Where(IsNodeVisible))
            .ToArray();
    }

    private static IReadOnlyList<NCardHolder> ResolveDeckUpgradeOptions(
        NDeckUpgradeSelectScreen? deckUpgradeScreen)
    {
        if (deckUpgradeScreen is null)
        {
            return Array.Empty<NCardHolder>();
        }

        var grid = ResolveCardGrid(deckUpgradeScreen);
        if (grid is null)
        {
            return Array.Empty<NCardHolder>();
        }

        return SortByVisualPosition(
                grid.CurrentlyDisplayedCardHolders
                    .Where(static holder => holder.CardModel is not null)
                    .Where(IsNodeVisible)
                    .Where(holder => !IsDeckUpgradePreviewHolder(deckUpgradeScreen, holder)))
            .ToArray();
    }

    private static IReadOnlyList<NCrystalSphereCell> ResolveCrystalSphereCells(
        NCrystalSphereScreen? crystalSphereScreen)
    {
        var cellContainer = crystalSphereScreen?.GetNodeOrNull<Control>("%Cells") ??
                            GetHiddenFieldValue(crystalSphereScreen, "_cellContainer") as Control;
        if (cellContainer is null)
        {
            return Array.Empty<NCrystalSphereCell>();
        }

        return cellContainer
            .GetChildren()
            .OfType<NCrystalSphereCell>()
            .Where(IsNodeVisible)
            .OrderBy(static cell => cell.Entity?.Y ?? int.MaxValue)
            .ThenBy(static cell => cell.Entity?.X ?? int.MaxValue)
            .ToArray();
    }

    private static IReadOnlyList<NMapPoint> ResolveMapPoints(NMapScreen? mapScreen)
    {
        var pointsContainer = mapScreen?.GetNodeOrNull<Control>("TheMap/Points") ??
                              GetHiddenFieldValue(mapScreen, "_points") as Control;
        if (pointsContainer is null)
        {
            return Array.Empty<NMapPoint>();
        }

        return pointsContainer
            .GetChildren()
            .OfType<NMapPoint>()
            .Where(IsNodeVisible)
            .OrderBy(static point => point.Point.coord.row)
            .ThenBy(static point => point.Point.coord.col)
            .ToArray();
    }

    private static IReadOnlyList<Node> ResolveMainMenuTextButtons(NMainMenu? mainMenuRoot)
    {
        if (mainMenuRoot is null)
        {
            return Array.Empty<Node>();
        }

        return SortByVisualPosition(
                new Node?[]
                {
                    GetHiddenFieldValue(mainMenuRoot, "_abandonRunButton") as Node,
                    GetHiddenFieldValue(mainMenuRoot, "_singleplayerButton") as Node,
                    GetHiddenFieldValue(mainMenuRoot, "_multiplayerButton") as Node,
                    GetHiddenFieldValue(mainMenuRoot, "_timelineButton") as Node,
                    GetHiddenFieldValue(mainMenuRoot, "_settingsButton") as Node,
                    GetHiddenFieldValue(mainMenuRoot, "_compendiumButton") as Node,
                    GetHiddenFieldValue(mainMenuRoot, "_quitButton") as Node
                }
                .Where(static button => button is not null && GodotObject.IsInstanceValid(button))
                .Cast<Node>()
                .Where(IsNodeVisible)
                .DistinctBy(static button => button.NativeInstance))
            .ToArray();
    }

    private static IReadOnlyList<NPopupYesNoButton> ResolveAbandonRunConfirmButtons(Node? abandonRunConfirmPopup)
    {
        var verticalPopup = GetHiddenFieldValue(abandonRunConfirmPopup, "_verticalPopup");
        var yesButton = GetHiddenPropertyObjectValue(verticalPopup, "YesButton") as NPopupYesNoButton ??
                        GetHiddenFieldValue(verticalPopup, "_yesButton") as NPopupYesNoButton;
        var noButton = GetHiddenPropertyObjectValue(verticalPopup, "NoButton") as NPopupYesNoButton ??
                       GetHiddenFieldValue(verticalPopup, "_noButton") as NPopupYesNoButton;

        return SortByVisualPosition(
                new[] { yesButton, noButton }
                    .Where(static button => button is not null && GodotObject.IsInstanceValid(button))
                    .Cast<NPopupYesNoButton>()
                    .Where(IsNodeVisible))
            .ToArray();
    }

    private static NTreasureButton? ResolveTreasureChestButton(NTreasureRoom? treasureRoom)
    {
        if (treasureRoom is null)
        {
            return null;
        }

        return treasureRoom.GetNodeOrNull<NTreasureButton>("%Chest") ??
               GetHiddenFieldValue(treasureRoom, "_chestButton") as NTreasureButton;
    }

    private static NTreasureRoomRelicCollection? ResolveTreasureRelicCollection(NTreasureRoom? treasureRoom)
    {
        if (treasureRoom is null)
        {
            return null;
        }

        return treasureRoom.GetNodeOrNull<NTreasureRoomRelicCollection>("%RelicCollection") ??
               GetHiddenFieldValue(treasureRoom, "_relicCollection") as NTreasureRoomRelicCollection;
    }

    private static NSubmenu? ResolveMainMenuSubmenu(IScreenContext? activeScreen, NMainMenu? mainMenu)
    {
        if (activeScreen is NSubmenu submenu)
        {
            return submenu;
        }

        return activeScreen is null
            ? mainMenu?.SubmenuStack?.Peek() as NSubmenu
            : null;
    }

    private static Node? ResolveEventOptionSearchRoot(IScreenContext? activeScreen, NEventRoom? eventRoom)
    {
        if (activeScreen is Node activeNode &&
            (eventRoom is null || IsNodeSameOrDescendantOf(activeNode, eventRoom)))
        {
            return activeNode;
        }

        if (eventRoom?.CustomEventNode?.CurrentScreenContext is Node customEventScreen)
        {
            return customEventScreen;
        }

        if (eventRoom?.Layout is Node layoutNode)
        {
            return layoutNode;
        }

        return eventRoom;
    }

    private static IReadOnlyList<NCardHolder> GetCardSelectionOptions(Node? cardSelectionScreen)
    {
        if (cardSelectionScreen is null)
        {
            return Array.Empty<NCardHolder>();
        }

        if (IsCardSelectionPreviewVisible(cardSelectionScreen))
        {
            return Array.Empty<NCardHolder>();
        }

        if (cardSelectionScreen is NPlayerHand playerHand)
        {
            return FindVisibleDescendants<NCardHolder>(playerHand)
                    // Selected hand holders remain legal click targets: clicking
                    // one removes it from a multi-select checkbox set.  Keep
                    // them beside the unselected holders so action export can
                    // emit an explicit deselect mutation.
                    .Where(static holder => holder.CardModel is not null)
                    .OrderBy(holder => TryGetCombatHandCardSelectionIndex(holder.CardModel) ?? int.MaxValue)
                    .ThenBy(static holder => holder is Control control ? control.GlobalPosition.Y : 0f)
                    .ThenBy(static holder => holder is Control control ? control.GlobalPosition.X : 0f)
                .DistinctBy(static holder => holder.CardModel, ReferenceEqualityComparer.Instance)
                .ToArray();
        }

        if (ResolveCardGrid(cardSelectionScreen) is { } cardGrid)
        {
            return SortByVisualPosition(
                    cardGrid.CurrentlyDisplayedCardHolders
                        .Where(static holder => holder.CardModel is not null)
                        .Where(IsNodeVisible))
                .DistinctBy(static holder => holder.CardModel, ReferenceEqualityComparer.Instance)
                .ToArray();
        }

        if (string.Equals(
                cardSelectionScreen.GetType().FullName,
                "MegaCrit.Sts2.Core.Nodes.Screens.CardSelection.NChooseACardSelectionScreen",
                StringComparison.Ordinal))
        {
            var cardRow = cardSelectionScreen.GetNodeOrNull<Control>("CardRow") ??
                          GetHiddenFieldValue(cardSelectionScreen, "_cardRow") as Control;
            if (cardRow is not null && GodotObject.IsInstanceValid(cardRow))
            {
                return SortByVisualPosition(
                        cardRow
                            .GetChildren()
                            .OfType<NCardHolder>()
                            .Where(static holder => holder.CardModel is not null)
                            .Where(IsNodeVisible))
                    .DistinctBy(static holder => holder.CardModel, ReferenceEqualityComparer.Instance)
                    .ToArray();
            }
        }

        return SortByVisualPosition(
                FindVisibleDescendants<NCardHolder>(cardSelectionScreen)
                    .Where(static holder => holder.CardModel is not null))
            .DistinctBy(static holder => holder.CardModel, ReferenceEqualityComparer.Instance)
            .ToArray();
    }

    private static IReadOnlyList<NCardBundle> GetCardSelectionBundles(Node? cardSelectionScreen)
    {
        if (cardSelectionScreen is not NChooseABundleSelectionScreen)
        {
            return Array.Empty<NCardBundle>();
        }

        var bundleRow = GetHiddenFieldValue(cardSelectionScreen, "_bundleRow") as Node;
        var searchRoot = bundleRow is not null && GodotObject.IsInstanceValid(bundleRow)
            ? bundleRow
            : cardSelectionScreen;

        if (searchRoot is Control bundleContainer)
        {
            return SortByVisualPosition(
                    bundleContainer
                        .GetChildren()
                        .OfType<NCardBundle>()
                        .Where(static bundle => bundle.Bundle.Count > 0)
                        .Where(IsNodeVisible))
                .ToArray();
        }

        return SortByVisualPosition(
                FindVisibleDescendants<NCardBundle>(searchRoot)
                    .Where(static bundle => bundle.Bundle.Count > 0))
            .ToArray();
    }

    private static NCardGrid? ResolveCardGrid(Node? cardSelectionScreen)
    {
        return cardSelectionScreen?.GetNodeOrNull<NCardGrid>("%CardGrid") ??
               GetHiddenFieldValue(cardSelectionScreen, "_grid") as NCardGrid;
    }

    private static NProceedButton? ResolveVisibleProceedButton(
        Node? root,
        NProceedButton? preferredProceedButton,
        NProceedButton? treasureProceedButton,
        NProceedButton? restSiteProceedButton,
        NProceedButton? merchantProceedButton)
    {
        if (preferredProceedButton is not null &&
            IsNodeVisible(preferredProceedButton) &&
            IsButtonEnabled(preferredProceedButton))
        {
            return preferredProceedButton;
        }

        if (treasureProceedButton is not null &&
            IsNodeVisible(treasureProceedButton) &&
            IsButtonEnabled(treasureProceedButton))
        {
            return treasureProceedButton;
        }

        var excludedButtons = new HashSet<IntPtr>();
        if (treasureProceedButton is not null && GodotObject.IsInstanceValid(treasureProceedButton))
        {
            excludedButtons.Add(treasureProceedButton.NativeInstance);
        }

        if (restSiteProceedButton is not null && GodotObject.IsInstanceValid(restSiteProceedButton))
        {
            excludedButtons.Add(restSiteProceedButton.NativeInstance);
        }

        if (merchantProceedButton is not null && GodotObject.IsInstanceValid(merchantProceedButton))
        {
            excludedButtons.Add(merchantProceedButton.NativeInstance);
        }

        return SortByVisualPosition(
                FindVisibleDescendants<NProceedButton>(root)
                    .Where(button =>
                        !excludedButtons.Contains(button.NativeInstance) &&
                        IsButtonEnabled(button)))
            .LastOrDefault();
    }

    private static NProceedButton? ResolveStableProceedButton(
        IScreenContext? activeScreen,
        Node? root,
        NProceedButton? combatProceedButton,
        NProceedButton? treasureProceedButton,
        NProceedButton? restSiteProceedButton,
        NProceedButton? merchantProceedButton)
    {
        if (activeScreen is NCombatRoom &&
            combatProceedButton is not null &&
            IsNodeVisible(combatProceedButton) &&
            IsButtonEnabled(combatProceedButton))
        {
            return combatProceedButton;
        }

        if (activeScreen is NTreasureRoom &&
            treasureProceedButton is not null &&
            IsNodeVisible(treasureProceedButton) &&
            IsButtonEnabled(treasureProceedButton))
        {
            return treasureProceedButton;
        }

        if (activeScreen is NRestSiteRoom &&
            restSiteProceedButton is not null &&
            IsNodeVisible(restSiteProceedButton) &&
            IsButtonEnabled(restSiteProceedButton))
        {
            return restSiteProceedButton;
        }

        if (activeScreen is NMerchantRoom &&
            merchantProceedButton is not null &&
            IsNodeVisible(merchantProceedButton) &&
            IsButtonEnabled(merchantProceedButton))
        {
            return merchantProceedButton;
        }

        if (activeScreen is NRewardsScreen)
        {
            return null;
        }

        return ResolveVisibleProceedButton(
            root,
            combatProceedButton,
            treasureProceedButton,
            restSiteProceedButton,
            merchantProceedButton);
    }

    private static bool ShouldSuppressGenericRoomProceed(BridgeWorldContext context)
    {
        if (IsCardRewardSelectionVisible(context.CardRewardScreen, context.CardRewardOptions) ||
            IsRewardsScreenVisible(
                context.RewardsScreen,
                context.ProceedButton,
                context.RewardProceedButton,
                context.MapScreen,
                context.RewardButtons) ||
            IsCardSelectionVisible(context) ||
            IsDeckUpgradeSelectionVisible(context) ||
            (context.RestSiteRoom is not null && IsNodeVisible(context.RestSiteRoom)) ||
            (context.CrystalSphereScreen is not null && IsNodeVisible(context.CrystalSphereScreen)))
        {
            return true;
        }

        if (context.TreasureRoom is null || !IsNodeVisible(context.TreasureRoom))
        {
            return false;
        }

        if (CanOpenTreasureChest(context))
        {
            return true;
        }

        return context.TreasureRelicOptions.Any(IsNodeVisible);
    }

    private static bool CanOpenTreasureChest(BridgeWorldContext context)
    {
        if (context.TreasureRoom is null ||
            context.TreasureChestButton is null ||
            !IsNodeVisible(context.TreasureChestButton) ||
            !IsButtonEnabled(context.TreasureChestButton))
        {
            return false;
        }

        var hasRelicBeenClaimed = GetHiddenFieldValue(context.TreasureRoom, "_hasRelicBeenClaimed") is bool claimed && claimed;
        var isRelicCollectionOpen = GetHiddenFieldValue(context.TreasureRoom, "_isRelicCollectionOpen") is bool collectionOpen && collectionOpen;

        return !hasRelicBeenClaimed && !isRelicCollectionOpen;
    }

    private static string? TryGetCardSelectionPrompt(Node? cardSelectionScreen)
    {
        if (cardSelectionScreen is null)
        {
            return null;
        }

        if (cardSelectionScreen is NPlayerHand)
        {
            var handPrompt = TryGetLocalNodeText(ResolveFirstVisibleNode(
                cardSelectionScreen.GetNodeOrNull<Node>("%SelectionHeader"),
                GetHiddenFieldValue(cardSelectionScreen, "_selectionHeader") as Node));
            if (!string.IsNullOrWhiteSpace(handPrompt))
            {
                return handPrompt;
            }
        }

        if (string.Equals(cardSelectionScreen.GetType().Name, "NSimpleCardSelectScreen", StringComparison.Ordinal))
        {
            var simplePrompt = TryGetLocalNodeText(ResolveFirstVisibleNode(
                cardSelectionScreen.GetNodeOrNull<Node>("%BottomText/%BottomLabel"),
                cardSelectionScreen.GetNodeOrNull<Node>("%BottomLabel"),
                GetHiddenFieldValue(cardSelectionScreen, "_infoLabel") as Node));
            if (!string.IsNullOrWhiteSpace(simplePrompt))
            {
                return simplePrompt;
            }
        }

        var promptNode = ResolveFirstVisibleNode(
            cardSelectionScreen.GetNodeOrNull<Node>("%SelectionHeader"),
            cardSelectionScreen.GetNodeOrNull<Node>("%BottomText/%BottomLabel"),
            cardSelectionScreen.GetNodeOrNull<Node>("%BottomLabel"),
            GetHiddenFieldValue(cardSelectionScreen, "_selectionHeader") as Node,
            GetHiddenFieldValue(cardSelectionScreen, "_infoLabel") as Node,
            GetHiddenFieldValue(cardSelectionScreen, "_banner") as Node);
        var prompt = TryGetLocalNodeText(promptNode);
        return string.IsNullOrWhiteSpace(prompt) ? null : prompt;
    }

    private static string? TryGetDeckUpgradePrompt(NDeckUpgradeSelectScreen? deckUpgradeScreen)
    {
        if (deckUpgradeScreen is null)
        {
            return null;
        }

        var promptNode = ResolveFirstVisibleNode(
            deckUpgradeScreen.GetNodeOrNull<Node>("%BottomText/%BottomLabel"),
            deckUpgradeScreen.GetNodeOrNull<Node>("%BottomLabel"),
            GetHiddenFieldValue(deckUpgradeScreen, "_selectionHeader") as Node,
            GetHiddenFieldValue(deckUpgradeScreen, "_infoLabel") as Node,
            GetHiddenFieldValue(deckUpgradeScreen, "_banner") as Node,
            GetHiddenFieldValue(deckUpgradeScreen, "_singlePreviewTitleLabel") as Node,
            GetHiddenFieldValue(deckUpgradeScreen, "_multiPreviewTitleLabel") as Node);
        var prompt = TryGetLocalNodeText(promptNode);
        return string.IsNullOrWhiteSpace(prompt) ? null : prompt;
    }

    private static int CountSelectedCardSelectionCards(Node? cardSelectionScreen)
    {
        if (cardSelectionScreen is null)
        {
            return 0;
        }

        if (GetHiddenFieldValue(cardSelectionScreen, "_selectedBundle") is not null)
        {
            return 1;
        }

        if (GetHiddenFieldValue(cardSelectionScreen, "_selectedCards") is IEnumerable selectedCards)
        {
            return selectedCards.Cast<object?>().Count(static card => card is not null);
        }

        return GetHiddenFieldValue(cardSelectionScreen, "_cardSelected") is true ? 1 : 0;
    }

    private sealed class CardSelectionUiState
    {
        public bool Visible { get; init; }
        public string? ScreenType { get; init; }
        public int SelectedCount { get; init; }
        public bool ConfirmReady { get; init; }
        public int? MinSelect { get; init; }
        public int? MaxSelect { get; init; }
        public bool PreviewVisible { get; init; }
        public bool TargetSelected { get; init; }
        public ulong? OpenedAgeMs { get; init; }
        public bool SelectionReady { get; init; }
    }

    private static CardSelectionUiState CaptureCardSelectionUiState(
        Node? cardSelectionScreen,
        CardModel? targetCard = null)
    {
        var visible = cardSelectionScreen is not null && IsNodeVisible(cardSelectionScreen);
        var prefs = GetHiddenFieldValue(cardSelectionScreen, "_prefs");
        var confirmButton = ResolveCardSelectionConfirmButton(cardSelectionScreen);
        ulong? openedAgeMs = visible && cardSelectionScreen is NChooseACardSelectionScreen chooseACardSelectionScreen &&
                             TryGetNChooseACardSelectionOpenedAgeMs(chooseACardSelectionScreen, out var resolvedOpenedAgeMs)
            ? resolvedOpenedAgeMs
            : null;
        return new CardSelectionUiState
        {
            Visible = visible,
            ScreenType = visible ? cardSelectionScreen!.GetType().Name : null,
            SelectedCount = visible ? CountSelectedCardSelectionCards(cardSelectionScreen) : 0,
            ConfirmReady = visible &&
                           confirmButton is not null &&
                           IsNodeVisible(confirmButton) &&
                           IsButtonEnabled(confirmButton),
            MinSelect = visible ? GetHiddenPropertyValue<int>(prefs, "MinSelect") : null,
            MaxSelect = visible ? GetHiddenPropertyValue<int>(prefs, "MaxSelect") : null,
            PreviewVisible = visible && IsCardSelectionPreviewVisible(cardSelectionScreen),
            TargetSelected = visible &&
                             targetCard is not null &&
                             IsCardSelectionCardSelected(cardSelectionScreen, targetCard),
            OpenedAgeMs = openedAgeMs,
            SelectionReady = !visible ||
                             cardSelectionScreen is not NChooseACardSelectionScreen ||
                             (openedAgeMs.HasValue && openedAgeMs.Value > NChooseACardSelectionOpenGuardMs)
        };
    }

    private static bool HasCardSelectionStateProgress(
        CardSelectionUiState before,
        CardSelectionUiState after)
    {
        return before.Visible != after.Visible ||
               !string.Equals(before.ScreenType ?? string.Empty, after.ScreenType ?? string.Empty, StringComparison.Ordinal) ||
               before.SelectedCount != after.SelectedCount ||
               before.ConfirmReady != after.ConfirmReady ||
               before.MinSelect != after.MinSelect ||
               before.MaxSelect != after.MaxSelect ||
               before.PreviewVisible != after.PreviewVisible ||
               before.TargetSelected != after.TargetSelected;
    }

    private static bool IsCardSelectionCardSelected(Node? cardSelectionScreen, CardModel? card)
    {
        if (cardSelectionScreen is null || card is null)
        {
            return false;
        }

        if (GetHiddenFieldValue(cardSelectionScreen, "_selectedCards") is IEnumerable selectedCards)
        {
            return selectedCards.Cast<object?>().Any(selected => ReferenceEquals(selected, card));
        }

        return false;
    }

    private static int GetCardSelectionOptionIndex(
        Node? cardSelectionScreen,
        NCardHolder cardHolder,
        int fallbackIndex)
    {
        if (cardSelectionScreen is NPlayerHand &&
            TryGetCombatHandCardSelectionIndex(cardHolder.CardModel) is int handIndex)
        {
            return handIndex;
        }

        return fallbackIndex;
    }

    private static string? GetCardSelectionOptionSelectionId(
        Node? cardSelectionScreen,
        NCardHolder cardHolder,
        int optionIndex)
    {
        if (cardHolder.CardModel is null)
        {
            return null;
        }

        return GetCardReference(cardHolder.CardModel);
    }

    private static string GetCardReference(CardModel card)
    {
        return CardReferences.GetValue(
            card,
            static _ => new CardReferenceIdentity(
                $"card-{Interlocked.Increment(ref _nextCardReference):x16}"))
            .Value;
    }

    private sealed record CardReferenceIdentity(string Value);

    // Reference equality is the authoritative runtime card-instance identity.
    // ConditionalWeakTable keeps it stable across pile movement without
    // retaining removed card objects. A monotonic 64-bit counter avoids the
    // collisions possible with RuntimeHelpers.GetHashCode.
    private static readonly ConditionalWeakTable<CardModel, CardReferenceIdentity> CardReferences = new();
    private static long _nextCardReference;

    private static int? TryGetIntFromPropertyOrField(object? target, params string[] memberNames)
    {
        foreach (var memberName in memberNames)
        {
            var value = GetHiddenPropertyObjectValue(target, memberName) ?? GetHiddenFieldValue(target, memberName);
            if (TryConvertToInt(value) is int intValue)
            {
                return intValue;
            }
        }

        return null;
    }

    private static bool? TryGetBoolFromPropertyOrField(object? target, params string[] memberNames)
    {
        foreach (var memberName in memberNames)
        {
            var value = GetHiddenPropertyObjectValue(target, memberName) ?? GetHiddenFieldValue(target, memberName);
            if (value is bool boolValue)
            {
                return boolValue;
            }
        }

        return null;
    }

    private static int? TryConvertToInt(object? value)
    {
        try
        {
            return value switch
            {
                null => null,
                byte byteValue => byteValue,
                sbyte sbyteValue => sbyteValue,
                short shortValue => shortValue,
                ushort ushortValue => ushortValue,
                int intValue => intValue,
                uint uintValue when uintValue <= int.MaxValue => (int)uintValue,
                long longValue when longValue >= int.MinValue && longValue <= int.MaxValue => (int)longValue,
                ulong ulongValue when ulongValue <= int.MaxValue => (int)ulongValue,
                Enum enumValue => Convert.ToInt32(enumValue, CultureInfo.InvariantCulture),
                _ => null
            };
        }
        catch
        {
            return null;
        }
    }

    private static int? TryGetCombatHandCardSelectionIndex(CardModel? card)
    {
        if (card?.Owner is null)
        {
            return null;
        }

        try
        {
            var handPile = PileType.Hand.GetPile(card.Owner);
            var cards = handPile?.Cards;
            if (cards is null)
            {
                return null;
            }

            for (var index = 0; index < cards.Count; index++)
            {
                if (ReferenceEquals(cards[index], card))
                {
                    return index;
                }
            }

            return null;
        }
        catch
        {
            return null;
        }
    }

    private static PropertyInfo? FindProperty(Type? type, string propertyName, bool includeStatic = false)
    {
        while (type is not null)
        {
            var property = type.GetProperty(
                propertyName,
                (includeStatic ? BindingFlags.Static : BindingFlags.Instance) |
                BindingFlags.Public |
                BindingFlags.NonPublic |
                BindingFlags.DeclaredOnly);

            if (property is not null)
            {
                return property;
            }

            type = type.BaseType;
        }

        return null;
    }

    private static FieldInfo? FindField(Type? type, string fieldName, bool includeStatic = false)
    {
        while (type is not null)
        {
            var field = type.GetField(
                fieldName,
                (includeStatic ? BindingFlags.Static : BindingFlags.Instance) |
                BindingFlags.Public |
                BindingFlags.NonPublic |
                BindingFlags.DeclaredOnly);

            if (field is not null)
            {
                return field;
            }

            type = type.BaseType;
        }

        return null;
    }

}
