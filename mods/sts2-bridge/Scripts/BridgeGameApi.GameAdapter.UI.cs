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
    private static string ResolveCurrentScreen(
        IScreenContext? activeScreen,
        CombatManager? combatManager,
        NMapScreen? mapScreen,
        NCharacterSelectScreen? characterSelectScreen,
        Node? mainMenuRoot,
        Node? runModeSubmenu,
        Node? abandonRunConfirmPopup)
    {
        if (abandonRunConfirmPopup is not null && IsNodeVisible(abandonRunConfirmPopup))
        {
            return "ABANDON_RUN_CONFIRM";
        }

        if (activeScreen is not null)
        {
            switch (activeScreen)
            {
                case NAbandonRunConfirmPopup:
                    return "ABANDON_RUN_CONFIRM";
                case NCombatRoom:
                    return combatManager?.IsInProgress == true ? "COMBAT" : "ROOM";
                case NMapScreen when combatManager?.IsInProgress == true:
                    return "COMBAT";
                case NMapScreen when IsInteractiveMapSurface(mapScreen, combatManager):
                    return "MAP";
                case NRewardsScreen:
                    return "REWARDS";
                case NCardRewardSelectionScreen:
                    return "CARD_REWARD_SELECTION";
                case NDeckUpgradeSelectScreen:
                    return "DECK_UPGRADE_SELECTION";
                case NRestSiteRoom:
                    return "REST_SITE";
                case NMerchantInventory:
                case NMerchantRoom:
                    return "SHOP";
                case NTreasureRoom:
                    return "TREASURE";
                case NCrystalSphereScreen:
                    return "EVENT_CRYSTAL_SPHERE";
                case NEventRoom:
                    return "EVENT";
                case NGameOverScreen:
                    return "GAME_OVER";
                case NCharacterSelectScreen:
                    return "CHARACTER_SELECT";
                case NSingleplayerSubmenu:
                    return "RUN_MODE_SELECTION";
                case NMainMenu:
                    return "MAIN_MENU";
            }

            var fullName = activeScreen.GetType().FullName ?? activeScreen.GetType().Name;
            if (fullName.StartsWith("MegaCrit.Sts2.Core.Nodes.Screens.CardSelection.", StringComparison.Ordinal))
            {
                return "CARD_SELECTION";
            }

            if (fullName.Contains("CrystalSphere", StringComparison.Ordinal))
            {
                return "EVENT_CRYSTAL_SPHERE";
            }

            if (fullName.Contains("SingleplayerSubmenu", StringComparison.Ordinal))
            {
                return "RUN_MODE_SELECTION";
            }

            if (fullName.Contains("MainMenu", StringComparison.Ordinal))
            {
                return "MAIN_MENU";
            }

            if (fullName.Contains(".Events.", StringComparison.Ordinal) ||
                fullName.Contains("Event", StringComparison.Ordinal))
            {
                return "EVENT";
            }

            if (combatManager?.IsInProgress == true)
            {
                return "COMBAT";
            }

            return fullName;
        }

        if (runModeSubmenu is not null && IsNodeVisible(runModeSubmenu))
        {
            return "RUN_MODE_SELECTION";
        }

        if (characterSelectScreen is not null && IsNodeVisible(characterSelectScreen))
        {
            return "CHARACTER_SELECT";
        }

        if (mainMenuRoot is not null && IsNodeVisible(mainMenuRoot))
        {
            return "MAIN_MENU";
        }

        if (combatManager?.IsInProgress == true)
        {
            return "COMBAT";
        }

        if (IsInteractiveMapSurface(mapScreen, combatManager))
        {
            return "MAP";
        }

        return "UNKNOWN";
    }

    private static RunState? TryGetRunState(RunManager? runManager)
    {
        if (runManager is null)
        {
            return null;
        }

        try
        {
            return runManager.DebugOnlyGetState();
        }
        catch
        {
            return null;
        }
    }

    private static CombatState? TryGetCombatState(CombatManager? combatManager)
    {
        if (combatManager is null)
        {
            return null;
        }

        try
        {
            return combatManager.DebugOnlyGetState();
        }
        catch
        {
            return null;
        }
    }

    private static void EnsureDispatcherReady()
    {
        if (!BridgeCoordinator.IsReady)
        {
            throw new BridgeRequestException(
                HttpStatusCode.ServiceUnavailable,
                "dispatcher_not_ready",
                "The bridge dispatcher is not attached yet. Wait for the game to finish loading and try again.");
        }
    }

    private static bool IsNodeVisible(Node? node)
    {
        if (node is null || !GodotObject.IsInstanceValid(node))
        {
            return false;
        }

        if (BridgeRuntime.VisibleOnly && node is CanvasItem canvasItem)
        {
            return canvasItem.IsVisibleInTree();
        }

        return true;
    }

    private static bool IsSameNodeInstance(Node? left, Node? right)
    {
        if (left is null || right is null)
        {
            return false;
        }

        if (ReferenceEquals(left, right))
        {
            return true;
        }

        if (!GodotObject.IsInstanceValid(left) || !GodotObject.IsInstanceValid(right))
        {
            return false;
        }

        return left.NativeInstance == right.NativeInstance;
    }

    private static bool IsNodeSameOrDescendantOf(Node? candidate, Node? ancestor)
    {
        if (candidate is null || ancestor is null)
        {
            return false;
        }

        for (Node? current = candidate; current is not null; current = current.GetParent())
        {
            if (IsSameNodeInstance(current, ancestor))
            {
                return true;
            }
        }

        return false;
    }

    private static bool IsTypeFullName(Node? node, string fullTypeName)
    {
        return node is not null &&
               GodotObject.IsInstanceValid(node) &&
               string.Equals(node.GetType().FullName, fullTypeName, StringComparison.Ordinal);
    }

    private static bool IsMapPointTravelable(NMapPoint pointNode)
    {
        return GetHiddenPropertyValue<bool>(pointNode, "IsTravelable") ?? false;
    }

    private static void RefreshInteractiveMapTravelability(NMapScreen? mapScreen)
    {
        if (mapScreen is null || !mapScreen.IsOpen || mapScreen.IsTraveling)
        {
            return;
        }

        TryInvokeParameterless(mapScreen, "RecalculateTravelability");
        TryInvokeParameterless(mapScreen, "RefreshAllPointVisuals");
    }

    private static bool IsCurrentMapCoord(RunState? runState, MapCoord coord)
    {
        if (runState?.CurrentMapCoord is not MapCoord currentCoord)
        {
            return false;
        }

        return currentCoord.col == coord.col && currentCoord.row == coord.row;
    }

    private static bool HasVisibleEnabledRestSiteOptions(IReadOnlyList<NRestSiteButton> restSiteButtons)
    {
        return restSiteButtons.Any(static button =>
            IsNodeVisible(button) &&
            button.Option is { IsEnabled: true });
    }

    private static bool IsButtonEnabled(object? target)
    {
        return target is not null && (GetHiddenPropertyValue<bool>(target, "IsEnabled") ?? true);
    }

    private static bool IsRunModeSelectionVisible(BridgeWorldContext context)
    {
        return context.RunModeSubmenu is not null && IsNodeVisible(context.RunModeSubmenu);
    }

    private static bool IsRewardsScreenVisible(
        NRewardsScreen? rewardsScreen,
        NProceedButton? roomProceedButton,
        NProceedButton? rewardProceedButton,
        NMapScreen? mapScreen,
        IReadOnlyList<NRewardButton> rewardButtons)
    {
        if (rewardsScreen is not null && IsNodeVisible(rewardsScreen))
        {
            return true;
        }

        if (IsInteractiveMapSurface(mapScreen) && rewardButtons.Count == 0)
        {
            return false;
        }

        if (rewardButtons.Count == 0 &&
            roomProceedButton is not null &&
            IsNodeVisible(roomProceedButton) &&
            !IsSameNodeInstance(roomProceedButton, rewardProceedButton))
        {
            return false;
        }

        return rewardProceedButton is not null && IsNodeVisible(rewardProceedButton);
    }

    private static bool IsCardRewardSelectionVisible(
        NCardRewardSelectionScreen? cardRewardScreen,
        IReadOnlyList<NCardHolder> cardRewardOptions)
    {
        return (cardRewardScreen is not null && IsNodeVisible(cardRewardScreen)) ||
               cardRewardOptions.Count > 0;
    }

    private static bool IsCardRewardSelectionReady(NCardRewardSelectionScreen? cardRewardScreen)
    {
        return cardRewardScreen is not null &&
               IsNodeVisible(cardRewardScreen) &&
               GetHiddenFieldValue(cardRewardScreen, "_completionSource") is not null;
    }

    private static bool IsDeckUpgradeSelectionVisible(BridgeWorldContext context)
    {
        return context.DeckUpgradeScreen is not null && IsNodeVisible(context.DeckUpgradeScreen);
    }

    private static bool IsCardSelectionVisible(BridgeWorldContext context)
    {
        return context.CardSelectionScreen is not null && IsNodeVisible(context.CardSelectionScreen);
    }

    private static bool IsTerminalRewardsProceedVisible(BridgeWorldContext context)
    {
        return IsRewardsScreenVisible(
                   context.RewardsScreen,
                   context.ProceedButton,
                   context.RewardProceedButton,
                   context.MapScreen,
                   context.RewardButtons) &&
               context.RewardProceedButton is not null &&
               IsNodeVisible(context.RewardProceedButton) &&
               context.RewardButtons.Count == 0 &&
               !IsInteractiveMapSurface(context.MapScreen) &&
               !IsCardRewardSelectionVisible(context.CardRewardScreen, context.CardRewardOptions);
    }

    private static bool IsRewardResolutionAction(string actionId)
    {
        return actionId.StartsWith("reward:", StringComparison.Ordinal) ||
               actionId.StartsWith("card_reward:", StringComparison.Ordinal);
    }

    /// <summary>
    /// True when an action's resulting screen transitioned out of a combat
    /// surface into a post-combat screen (rewards, map, game-over). Used as
    /// the trigger for the per-combat finalize drain on the full_run path.
    /// </summary>
    private static bool IsCombatExitTransition(string? screenBefore, string? screenAfter)
    {
        if (string.IsNullOrEmpty(screenBefore) || string.IsNullOrEmpty(screenAfter))
        {
            return false;
        }
        if (string.Equals(screenBefore, screenAfter, StringComparison.Ordinal))
        {
            return false;
        }
        var beforeIsCombat = IsCombatLikeScreen(screenBefore);
        var afterIsCombat = IsCombatLikeScreen(screenAfter);
        return beforeIsCombat && !afterIsCombat;
    }

    private static bool IsCombatLikeScreen(string? screen)
    {
        if (string.IsNullOrEmpty(screen))
        {
            return false;
        }
        return screen.IndexOf("combat", StringComparison.OrdinalIgnoreCase) >= 0
            || screen.Equals("COMBAT", StringComparison.OrdinalIgnoreCase)
            || screen.Equals("BATTLE", StringComparison.OrdinalIgnoreCase)
            || screen.Equals("FIGHTING", StringComparison.OrdinalIgnoreCase);
    }

    private static bool IsCardSelectionResolutionAction(string actionId)
    {
        return actionId.StartsWith("card_selection:select:", StringComparison.Ordinal);
    }

    private static async Task<(ObservedFrontier Frontier, List<object> AutoExecutedActions)> MaybeAutoProceedAfterRewardActionAsync(
        ObservedFrontier frontier,
        CancellationToken cancellationToken)
    {
        var autoExecutedActions = new List<object>();
        var autoProceedCount = 0;

        for (var attempt = 0; attempt < 12; attempt++)
        {
            var nonAutomationActions = GetNonAutomationActions(frontier.Snapshot);

            if (nonAutomationActions.Length == 1 &&
                nonAutomationActions[0].ActionId.Equals("proceed", StringComparison.Ordinal))
            {
                if (autoProceedCount >= 3)
                {
                    return (frontier, autoExecutedActions);
                }

                var beforeAutoProceed = frontier;
                frontier = await ExecuteActionAndWaitForFrontierAsync(
                    frontier,
                    "proceed",
                    nonAutomationActions[0],
                    waitAfterMs: 0,
                    cancellationToken);
                autoProceedCount++;
                var stateChanged = HasFrontierChanged(beforeAutoProceed, frontier);
                autoExecutedActions.Add(new
                {
                    action_id = "proceed",
                    source = "auto_after_reward",
                    wait_after_ms = 0,
                    state_changed = stateChanged
                });

                if (!stateChanged)
                {
                    return (frontier, autoExecutedActions);
                }

                continue;
            }

            if (nonAutomationActions.Length > 0)
            {
                return (frontier, autoExecutedActions);
            }

            if (attempt >= 11)
            {
                break;
            }

            var beforePassiveWait = frontier;
            frontier = await WaitForNextObservedFrontierAsync(
                frontier,
                PassiveFrontierWaitTimeoutMs,
                cancellationToken);
            if (!HasFrontierChanged(beforePassiveWait, frontier))
            {
                break;
            }
        }

        return (frontier, autoExecutedActions);
    }

    private static async Task<(ObservedFrontier Frontier, List<object> AutoExecutedActions)> MaybeAutoCompleteCardSelectionAsync(
        ObservedFrontier frontier,
        CancellationToken cancellationToken)
    {
        var autoExecutedActions = new List<object>();

        if (!ShouldAutoCompleteCardSelection(frontier.Snapshot))
        {
            return (frontier, autoExecutedActions);
        }

        var latestFrontier = await ObserveFrontierAsync(cancellationToken);
        if (HasFrontierChanged(frontier, latestFrontier))
        {
            frontier = latestFrontier;
        }

        if (!ShouldAutoCompleteCardSelection(frontier.Snapshot))
        {
            return (frontier, autoExecutedActions);
        }

        if (!frontier.Snapshot.ActionLookup.TryGetValue("card_selection:confirm", out var confirmAction))
        {
            return (frontier, autoExecutedActions);
        }

        frontier = await ExecuteActionAndWaitForFrontierAsync(
            frontier,
            "card_selection:confirm",
            confirmAction,
            waitAfterMs: 0,
            cancellationToken);
        autoExecutedActions.Add(new
        {
            action_id = "card_selection:confirm",
            source = "auto_after_card_selection",
            wait_after_ms = 0
        });

        return (frontier, autoExecutedActions);
    }

    private static bool ShouldAutoCompleteCardSelection(BridgeSnapshot snapshot)
    {
        var cardSelection = JsonSerializer.SerializeToElement(snapshot.Fields.CardSelection);
        if (!cardSelection.TryGetProperty("visible", out var visibleProperty) ||
            !visibleProperty.GetBoolean())
        {
            return false;
        }

        if (!cardSelection.TryGetProperty("confirm_visible", out var confirmVisibleProperty) ||
            !confirmVisibleProperty.GetBoolean())
        {
            return false;
        }

        if (!cardSelection.TryGetProperty("selected_count", out var selectedCountProperty) ||
            selectedCountProperty.GetInt32() <= 0)
        {
            return false;
        }

        var minSelect =
            cardSelection.TryGetProperty("min_select", out var minSelectProperty) &&
            minSelectProperty.ValueKind is not JsonValueKind.Null and not JsonValueKind.Undefined
                ? minSelectProperty.GetInt32()
                : 0;
        var maxSelect =
            cardSelection.TryGetProperty("max_select", out var maxSelectProperty) &&
            maxSelectProperty.ValueKind is not JsonValueKind.Null and not JsonValueKind.Undefined
                ? maxSelectProperty.GetInt32()
                : 0;

        return minSelect == 1 &&
               maxSelect == 1 &&
               snapshot.ActionLookup.ContainsKey("card_selection:confirm");
    }

    private static bool SafeGetCreatureIsHittable(Creature creature)
    {
        try
        {
            return creature.IsHittable;
        }
        catch
        {
            return false;
        }
    }

    private static bool SafeCanThrowPotionAtAlly(PotionModel potion)
    {
        try
        {
            return potion.CanThrowAtAlly();
        }
        catch
        {
            return false;
        }
    }

    private static bool SafeGetPotionIsUsable(PotionModel potion)
    {
        try
        {
            return potion.Owner is not null &&
                   !potion.HasBeenRemovedFromState &&
                   !potion.IsQueued &&
                   potion.PassesCustomUsabilityCheck;
        }
        catch
        {
            return false;
        }
    }

    private static bool SafeGetPotionIsQueued(PotionModel potion)
    {
        try
        {
            return potion.IsQueued;
        }
        catch
        {
            return false;
        }
    }

    private static bool SafeGetPotionHasBeenRemovedFromState(PotionModel potion)
    {
        try
        {
            return potion.HasBeenRemovedFromState;
        }
        catch
        {
            return false;
        }
    }

    private static List<T> FindVisibleDescendants<T>(Node? root) where T : Node
    {
        var result = new List<T>();
        if (root is null)
        {
            return result;
        }

        var seen = new HashSet<IntPtr>();

        void Visit(Node node)
        {
            if (!GodotObject.IsInstanceValid(node))
            {
                return;
            }

            if (node is T typed && seen.Add(typed.NativeInstance) && IsNodeVisible(typed))
            {
                result.Add(typed);
            }

            foreach (Node child in node.GetChildren())
            {
                Visit(child);
            }
        }

        Visit(root);
        return result;
    }

    private static List<Node> FindVisibleDescendants(Node? root, Func<Node, bool> predicate)
    {
        var result = new List<Node>();
        if (root is null)
        {
            return result;
        }

        var seen = new HashSet<IntPtr>();

        void Visit(Node node)
        {
            if (!GodotObject.IsInstanceValid(node))
            {
                return;
            }

            if (seen.Add(node.NativeInstance) && predicate(node) && IsNodeVisible(node))
            {
                result.Add(node);
            }

            foreach (Node child in node.GetChildren())
            {
                Visit(child);
            }
        }

        Visit(root);
        return result;
    }

    private static List<T> SortByVisualPosition<T>(IEnumerable<T> nodes) where T : Node
    {
        return nodes
            .OrderBy(static node => node is Control control ? control.GlobalPosition.Y : 0f)
            .ThenBy(static node => node is Control control ? control.GlobalPosition.X : 0f)
            .ToList();
    }

}
