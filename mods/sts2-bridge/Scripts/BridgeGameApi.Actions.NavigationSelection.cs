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
    private static void AddTreasureRoomActions(List<BridgeResolvedAction> actions, BridgeWorldContext context)
    {
        if (context.TreasureRoom is null || !IsNodeVisible(context.TreasureRoom))
        {
            return;
        }

        if (CanOpenTreasureChest(context))
        {
            var chestButton = context.TreasureChestButton!;
            actions.Add(new BridgeResolvedAction
            {
                ActionId = "treasure:open",
                Payload = new
                {
                    action_id = "treasure:open",
                    kind = "treasure",
                    label = "Open treasure chest",
                    screen = context.Screen
                },
                Execute = () => InvokeTreasureChestAction(context.TreasureRoom, chestButton)
            });
        }

        for (var index = 0; index < context.TreasureRelicOptions.Count; index++)
        {
            var holder = context.TreasureRelicOptions[index];
            if (!IsNodeVisible(holder))
            {
                continue;
            }

            var actionId = $"treasure_relic:{index}";
            actions.Add(new BridgeResolvedAction
            {
                ActionId = actionId,
                Payload = new
                {
                    action_id = actionId,
                    kind = "treasure_relic",
                    index,
                    label = $"Pick treasure relic {index}: {TextOf(holder.Relic?.Model?.Title)}",
                    relic = BuildRelicPayload(holder.Relic?.Model),
                    screen = context.Screen
                },
                Execute = () => InvokeTreasureRelicAction(context.TreasureRelicCollection, holder)
            });
        }
    }

    private static void AddRunModeActions(List<BridgeResolvedAction> actions, BridgeWorldContext context)
    {
        if (!IsRunModeSelectionVisible(context))
        {
            return;
        }

        AddRunModeAction(
            actions,
            context,
            context.RunModeStandardButton,
            "standard",
            "Start standard run",
            "OpenCharacterSelect");
        AddRunModeAction(
            actions,
            context,
            context.RunModeDailyButton,
            "daily",
            "Open daily challenge",
            "OpenDailyScreen");
        AddRunModeAction(
            actions,
            context,
            context.RunModeCustomButton,
            "custom",
            "Open custom run setup",
            "OpenCustomScreen");

        if (context.RunModeBackButton is null || !IsNodeVisible(context.RunModeBackButton))
        {
            return;
        }

        actions.Add(new BridgeResolvedAction
        {
            ActionId = "run_mode:back",
            Payload = new
            {
                action_id = "run_mode:back",
                kind = "run_mode_selection",
                run_mode_action = "back",
                button_text = TryGetLocalNodeText(context.RunModeBackButton),
                label = "Back",
                screen = context.Screen
            },
            Execute = () => InvokeMenuButtonAction(context.RunModeBackButton)
        });
    }

    private static void AddRestSiteActions(List<BridgeResolvedAction> actions, BridgeWorldContext context)
    {
        if (IsDeckUpgradeSelectionVisible(context))
        {
            return;
        }

        if (IsInteractiveMapSurface(context.MapScreen))
        {
            return;
        }

        if (context.RestSiteRoom is null || !IsNodeVisible(context.RestSiteRoom))
        {
            return;
        }

        var canProceed = !HasVisibleEnabledRestSiteOptions(context.RestSiteButtons);

        for (var index = 0; index < context.RestSiteButtons.Count; index++)
        {
            var button = context.RestSiteButtons[index];
            var option = button.Option;
            if (!IsNodeVisible(button) || option is null || !option.IsEnabled)
            {
                continue;
            }

            var actionId = $"rest_site:{index}";
            actions.Add(new BridgeResolvedAction
            {
                ActionId = actionId,
                Payload = new
                {
                    action_id = actionId,
                    kind = "rest_site",
                    index,
                    label = $"Rest site option {index}: {TextOf(option.Title)}",
                    option = BuildRestSiteOptionPayload(option, index),
                    screen = context.Screen
                },
                Execute = () => InvokeClickablePressAndRelease(button)
            });
        }

        if (canProceed &&
            context.RestSiteProceedButton is not null &&
            IsNodeVisible(context.RestSiteProceedButton) &&
            IsButtonEnabled(context.RestSiteProceedButton))
        {
            actions.Add(new BridgeResolvedAction
            {
                ActionId = "rest_site:proceed",
                Payload = new
                {
                    action_id = "rest_site:proceed",
                    kind = "rest_site",
                    label = "Rest site proceed",
                    screen = context.Screen
                },
                Execute = () => InvokeRestSiteProceedAction(context.RestSiteRoom, context.RestSiteProceedButton)
            });
        }
    }

    private static string NormalizeSelectionOperationType(string? semantics)
    {
        return (semantics ?? string.Empty).Trim().ToLowerInvariant() switch
        {
            "discard" => "discard",
            "retain" => "retain",
            "exhaust" => "exhaust",
            "remove" => "remove",
            "transform" => "transform",
            "upgrade" => "upgrade",
            "copy" => "copy",
            "add" => "add",
            "replace" => "replace",
            "enchant" => "enchant",
            "afflict" => "afflict",
            _ => "unknown"
        };
    }

    private static object BuildRuntimeSelectionPayload(
        string? screenType,
        string? selectionSemantics,
        int selectedCount,
        int? minSelect,
        int? maxSelect,
        bool? requiresManualConfirmation,
        string source,
        string sourceZone,
        string destinationZone)
    {
        var target = maxSelect ?? minSelect;
        return new
        {
            screen_type = string.IsNullOrWhiteSpace(screenType) ? "card_selection" : screenType,
            operation_type = NormalizeSelectionOperationType(selectionSemantics),
            source = string.IsNullOrWhiteSpace(source) ? "card_selection" : source,
            source_zone = sourceZone ?? string.Empty,
            destination_zone = destinationZone ?? string.Empty,
            selected_count = selectedCount,
            min_count = minSelect ?? 0,
            max_count = maxSelect ?? minSelect ?? 0,
            selection_required = (target ?? 0) > 0 || requiresManualConfirmation == true,
            modifier_id = string.Empty,
            confidence = "runtime_internal"
        };
    }

    private static void AddDeckUpgradeActions(List<BridgeResolvedAction> actions, BridgeWorldContext context)
    {
        if (!IsDeckUpgradeSelectionVisible(context) || context.DeckUpgradeScreen is null)
        {
            return;
        }

        var selectionPrompt = TryGetDeckUpgradePrompt(context.DeckUpgradeScreen);
        var typedSelection = BuildRuntimeSelectionPayload(
            context.DeckUpgradeScreen.GetType().Name,
            "upgrade",
            selectedCount: 0,
            minSelect: 1,
            maxSelect: 1,
            requiresManualConfirmation: true,
            source: "deck_upgrade",
            sourceZone: "deck",
            destinationZone: "deck");

        for (var index = 0; index < context.DeckUpgradeOptions.Count; index++)
        {
            var cardHolder = context.DeckUpgradeOptions[index];
            if (!IsNodeVisible(cardHolder) || cardHolder.CardModel is null)
            {
                continue;
            }

            var actionId = $"deck_upgrade:select:{index}";
            actions.Add(new BridgeResolvedAction
            {
                ActionId = actionId,
                Payload = new
                {
                    action_id = actionId,
                    kind = "deck_upgrade",
                    upgrade_action = "select_card",
                    selection_semantics = "upgrade",
                    selection_prompt = selectionPrompt,
                    typed_selection = typedSelection,
                    index,
                    label = $"Select upgrade card {index}: {cardHolder.CardModel.Title}",
                    card = BuildCardPayload(cardHolder.CardModel),
                    screen = context.Screen
                },
                Execute = () => InvokeSingleArgumentAction(context.DeckUpgradeScreen, "OnCardClicked", cardHolder.CardModel)
            });
        }

        if (context.DeckUpgradeConfirmButton is not null &&
            IsNodeVisible(context.DeckUpgradeConfirmButton) &&
            IsButtonEnabled(context.DeckUpgradeConfirmButton))
        {
            actions.Add(new BridgeResolvedAction
            {
                ActionId = "deck_upgrade:confirm",
                Payload = new
                {
                    action_id = "deck_upgrade:confirm",
                    kind = "deck_upgrade",
                    upgrade_action = "confirm",
                    selection_semantics = "upgrade",
                    selection_prompt = selectionPrompt,
                    typed_selection = typedSelection,
                    label = "Confirm upgrade selection",
                    screen = context.Screen
                },
                Execute = () => InvokeSingleArgumentAction(
                    context.DeckUpgradeScreen,
                    "ConfirmSelection",
                    context.DeckUpgradeConfirmButton)
            });
        }

        if (context.DeckUpgradeCancelButton is not null &&
            IsNodeVisible(context.DeckUpgradeCancelButton) &&
            IsButtonEnabled(context.DeckUpgradeCancelButton))
        {
            actions.Add(new BridgeResolvedAction
            {
                ActionId = "deck_upgrade:cancel",
                Payload = new
                {
                    action_id = "deck_upgrade:cancel",
                    kind = "deck_upgrade",
                    upgrade_action = "cancel",
                    selection_semantics = "upgrade",
                    selection_prompt = selectionPrompt,
                    typed_selection = typedSelection,
                    label = "Cancel upgrade selection",
                    screen = context.Screen
                },
                Execute = () => InvokeSingleArgumentAction(
                    context.DeckUpgradeScreen,
                    "CancelSelection",
                    context.DeckUpgradeCancelButton)
            });
        }

        if (context.DeckUpgradeCloseButton is not null &&
            IsNodeVisible(context.DeckUpgradeCloseButton) &&
            IsButtonEnabled(context.DeckUpgradeCloseButton))
        {
            actions.Add(new BridgeResolvedAction
            {
                ActionId = "deck_upgrade:close",
                Payload = new
                {
                    action_id = "deck_upgrade:close",
                    kind = "deck_upgrade",
                    upgrade_action = "close",
                    selection_semantics = "upgrade",
                    selection_prompt = selectionPrompt,
                    typed_selection = typedSelection,
                    label = "Close upgrade selection",
                    screen = context.Screen
                },
                Execute = () => InvokeSingleArgumentAction(
                    context.DeckUpgradeScreen,
                    "CloseSelection",
                    context.DeckUpgradeCloseButton)
            });
        }
    }

}
