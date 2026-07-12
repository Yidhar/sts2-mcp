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
    private static void AddMainMenuActions(List<BridgeResolvedAction> actions, BridgeWorldContext context)
    {
        if (IsRunModeSelectionVisible(context))
        {
            return;
        }

        var initialActionCount = actions.Count;

        if (context.AbandonRunConfirmPopup is not null && IsNodeVisible(context.AbandonRunConfirmPopup))
        {
            for (var index = 0; index < context.AbandonRunConfirmButtons.Count; index++)
            {
                var button = context.AbandonRunConfirmButtons[index];
                if (!IsNodeVisible(button))
                {
                    continue;
                }

                var buttonText = TryGetLocalNodeText(button);
                var semanticAction = TryGetAbandonConfirmSemanticAction(buttonText);
                var actionId = semanticAction switch
                {
                    "confirm" => "main_menu:confirm_abandon_run",
                    "cancel" => "main_menu:cancel_abandon_run",
                    _ => $"main_menu:abandon_confirm:{index}"
                };
                var label = semanticAction switch
                {
                    "confirm" => "Confirm abandon current game",
                    "cancel" => "Cancel abandon current game",
                    _ when !string.IsNullOrWhiteSpace(buttonText) => $"Abandon confirmation: {buttonText}",
                    _ => $"Abandon confirmation button {index}"
                };

                actions.Add(new BridgeResolvedAction
                {
                    ActionId = actionId,
                    Payload = new
                    {
                        action_id = actionId,
                        kind = "main_menu",
                        menu_action = semanticAction ?? "abandon_confirm_button",
                        button_index = index,
                        button_text = buttonText,
                        label,
                        screen = context.Screen
                    },
                    Execute = () =>
                    {
                        if (semanticAction == "confirm" || semanticAction == "cancel")
                        {
                            InvokeAbandonRunConfirmAction(
                                context.AbandonRunConfirmPopup,
                                button,
                                confirm: semanticAction == "confirm");
                            return;
                        }

                        InvokeMenuButtonAction(button);
                    }
                });
            }

            return;
        }

        if (context.MainMenuContinueButton is not null && IsNodeVisible(context.MainMenuContinueButton))
        {
            var buttonText = TryGetLocalNodeText(context.MainMenuContinueButton);
            actions.Add(new BridgeResolvedAction
            {
                ActionId = "main_menu:continue",
                Payload = new
                {
                    action_id = "main_menu:continue",
                    kind = "main_menu",
                    menu_action = "continue",
                    button_text = buttonText,
                    label = !string.IsNullOrWhiteSpace(buttonText) ? buttonText : "Continue Game",
                    screen = context.Screen
                },
                Execute = () => InvokeMainMenuContinueAction(context.MainMenuRoot, context.MainMenuContinueButton)
            });
        }

        for (var index = 0; index < context.MainMenuTextButtons.Count; index++)
        {
            var button = context.MainMenuTextButtons[index];
            if (!IsNodeVisible(button))
            {
                continue;
            }

            var buttonText = TryGetLocalNodeText(button);
            var semanticAction = TryGetMainMenuSemanticAction(buttonText);
            var actionId = semanticAction is not null
                ? $"main_menu:{semanticAction}"
                : $"main_menu:button:{index}";
            var label = !string.IsNullOrWhiteSpace(buttonText)
                ? buttonText
                : $"Main menu button {index}";

            if (actions.Any(existing => existing.ActionId.Equals(actionId, StringComparison.Ordinal)))
            {
                actionId = $"main_menu:button:{index}";
            }

            actions.Add(new BridgeResolvedAction
            {
                ActionId = actionId,
                Payload = new
                {
                    action_id = actionId,
                    kind = "main_menu",
                    menu_action = semanticAction ?? "button",
                    button_index = index,
                    button_text = buttonText,
                    label,
                    screen = context.Screen
                },
                Execute = () =>
                {
                    if (semanticAction == "abandon_current_game" &&
                        context.MainMenuRoot is not null &&
                        IsNodeVisible(context.MainMenuRoot))
                    {
                        InvokeButtonAction(context.MainMenuRoot, "AbandonRun");
                        return;
                    }

                    if (semanticAction == "singleplayer" &&
                        context.MainMenuRoot is not null &&
                        IsNodeVisible(context.MainMenuRoot))
                    {
                        InvokeButtonAction(context.MainMenuRoot, "OpenSingleplayerSubmenu");
                        return;
                    }

                    InvokeMenuButtonAction(button);
                }
            });
        }

        if (actions.Count > initialActionCount ||
            context.MainMenuRoot is null ||
            !IsNodeVisible(context.MainMenuRoot))
        {
            return;
        }

        if (context.ContinueRunInfo is not null && IsNodeVisible(context.ContinueRunInfo))
        {
            actions.Add(new BridgeResolvedAction
            {
                ActionId = "main_menu:abandon_current_game",
                Payload = new
                {
                    action_id = "main_menu:abandon_current_game",
                    kind = "main_menu",
                    menu_action = "abandon_current_game",
                    label = "Abandon current game",
                    screen = context.Screen
                },
                Execute = () => InvokeButtonAction(context.MainMenuRoot, "AbandonRun")
            });
            return;
        }

        actions.Add(new BridgeResolvedAction
        {
            ActionId = "main_menu:singleplayer",
            Payload = new
            {
                action_id = "main_menu:singleplayer",
                kind = "main_menu",
                menu_action = "singleplayer",
                label = "Open singleplayer submenu",
                screen = context.Screen
            },
            Execute = () => InvokeButtonAction(context.MainMenuRoot, "OpenSingleplayerSubmenu")
        });
    }

    private static void AddGameOverActions(List<BridgeResolvedAction> actions, BridgeWorldContext context)
    {
        if (context.GameOverScreen is null || !IsNodeVisible(context.GameOverScreen))
        {
            return;
        }

        if (context.GameOverContinueButton is not null && IsNodeVisible(context.GameOverContinueButton))
        {
            actions.Add(new BridgeResolvedAction
            {
                ActionId = "game_over:continue",
                Payload = new
                {
                    action_id = "game_over:continue",
                    kind = "game_over",
                    game_over_action = "continue",
                    label = "Continue from game-over summary",
                    screen = context.Screen
                },
                Execute = () => InvokeGameOverContinueAction(
                    context.GameOverScreen,
                    context.GameOverContinueButton)
            });
        }

        if (context.GameOverMainMenuButton is not null && IsNodeVisible(context.GameOverMainMenuButton))
        {
            actions.Add(new BridgeResolvedAction
            {
                ActionId = "game_over:return_to_main_menu",
                Payload = new
                {
                    action_id = "game_over:return_to_main_menu",
                    kind = "game_over",
                    game_over_action = "return_to_main_menu",
                    label = "Return to main menu",
                    screen = context.Screen
                },
                Execute = () => InvokeGameOverReturnToMainMenuAction(
                    context.GameOverScreen,
                    context.GameOverMainMenuButton)
            });
        }
    }

    private static void AddRunModeAction(
        List<BridgeResolvedAction> actions,
        BridgeWorldContext context,
        Node? button,
        string actionSuffix,
        string fallbackLabel,
        string submenuMethodName)
    {
        if (button is null || !IsNodeVisible(button))
        {
            return;
        }

        var texts = CollectButtonPayloadTexts(button, 4);
        var buttonText = texts.FirstOrDefault(static text => !string.IsNullOrWhiteSpace(text)) ?? fallbackLabel;
        var actionId = $"run_mode:{actionSuffix}";

        actions.Add(new BridgeResolvedAction
        {
            ActionId = actionId,
            Payload = new
            {
                action_id = actionId,
                kind = "run_mode_selection",
                run_mode_action = actionSuffix,
                button_text = buttonText,
                texts,
                label = fallbackLabel,
                screen = context.Screen
            },
            Execute = () => InvokeRunModeSelectionAction(context.RunModeSubmenu, button, submenuMethodName)
        });
    }

}
