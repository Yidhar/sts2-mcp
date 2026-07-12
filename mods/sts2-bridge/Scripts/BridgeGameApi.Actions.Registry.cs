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
    private static List<BridgeResolvedAction> BuildResolvedActions(BridgeWorldContext context)
    {
        var actions = new List<BridgeResolvedAction>();
        var hasActiveRunContext = context.RunState is not null && context.RunState.IsGameOver != true;

        if (!hasActiveRunContext)
        {
            AddRunModeActions(actions, context);
            AddMainMenuActions(actions, context);
        }
        AddGameOverActions(actions, context);
        AddDeckUpgradeActions(actions, context);
        AddCardSelectionActions(actions, context);
        AddRestSiteActions(actions, context);
        AddShopActions(actions, context);
        AddTreasureRoomActions(actions, context);

        if (!hasActiveRunContext)
        {
            for (var index = 0; index < context.CharacterButtons.Count; index++)
            {
                var button = context.CharacterButtons[index];
                if (!IsNodeVisible(button) || button.IsLocked || ReferenceEquals(button, context.SelectedCharacterButton))
                {
                    continue;
                }

                var actionId = $"character_select:{index}";
                actions.Add(new BridgeResolvedAction
                {
                    ActionId = actionId,
                    Payload = new
                    {
                        action_id = actionId,
                        kind = "character_select",
                        index,
                        label = $"Select character {index}: {DescribeCharacter(button.Character)}",
                        character = BuildCharacterPayload(button.Character),
                        is_random = button.IsRandom,
                        screen = context.Screen
                    },
                    Execute = () => InvokeCharacterSelectAction(context.CharacterSelectScreen, button)
                });
            }
        }

        if (!hasActiveRunContext &&
            context.EmbarkButton is not null &&
            IsNodeVisible(context.EmbarkButton) &&
            IsButtonEnabled(context.EmbarkButton))
        {
            actions.Add(new BridgeResolvedAction
            {
                ActionId = "embark",
                Payload = new
                {
                    action_id = "embark",
                    kind = "character_select",
                    label = "Embark",
                    selected_character = BuildCharacterPayload(context.SelectedCharacterButton?.Character),
                    screen = context.Screen
                },
                Execute = () => InvokeEmbarkAction(context.CharacterSelectScreen, context.EmbarkButton)
            });
        }

        if (context.CombatManager?.IsInProgress == true &&
            IsCombatPlayPhase(context.CombatManager, context.CombatState) &&
            !IsCardSelectionVisible(context) &&
            !context.CombatManager.PlayerActionsDisabled)
        {
            AddCombatCardActions(actions, context);
            AddCombatPotionActions(actions, context);
        }

        if (context.CombatManager?.IsInProgress == true &&
            IsCombatPlayPhase(context.CombatManager, context.CombatState) &&
            !IsCardSelectionVisible(context) &&
            !context.CombatManager.PlayerActionsDisabled &&
            context.EndTurnButton is not null &&
            IsNodeVisible(context.EndTurnButton))
        {
            actions.Add(new BridgeResolvedAction
            {
                ActionId = "end_turn",
                Payload = new
                {
                    action_id = "end_turn",
                    kind = "combat",
                    label = "End Turn",
                    screen = context.Screen
                },
                Execute = () => InvokeButtonAction(context.EndTurnButton, "OnRelease", "CallReleaseLogic")
            });
        }

        AddPotionDiscardActions(actions, context);

        if (IsTerminalRewardsProceedVisible(context))
        {
            actions.Add(new BridgeResolvedAction
            {
                ActionId = "proceed",
                Payload = new
                {
                    action_id = "proceed",
                    kind = "proceed",
                    label = "Proceed from terminal rewards",
                    is_skip = false,
                    proceed_source = "terminal_rewards",
                    screen = context.Screen
                },
                Execute = () => InvokeTerminalRewardsProceed(
                    context.RunManager,
                    context.RewardsScreen,
                    context.RewardProceedButton)
            });
        }
        else if (!IsInteractiveMapSurface(context, actions) &&
                 context.ProceedButton is not null &&
                 IsNodeVisible(context.ProceedButton) &&
                 IsButtonEnabled(context.ProceedButton) &&
                 !ShouldSuppressGenericRoomProceed(context))
        {
            var label = context.ProceedButton.IsSkip ? "Skip" : "Proceed";
            actions.Add(new BridgeResolvedAction
            {
                ActionId = "proceed",
                Payload = new
                {
                    action_id = "proceed",
                    kind = "proceed",
                    label,
                    is_skip = context.ProceedButton.IsSkip,
                    proceed_source = "room",
                    screen = context.Screen
                },
                Execute = () => InvokeRoomProceedAction(context)
            });
        }

        if (!IsCardRewardSelectionVisible(context.CardRewardScreen, context.CardRewardOptions))
        {
            for (var index = 0; index < context.RewardButtons.Count; index++)
            {
                var button = context.RewardButtons[index];
                if (!IsNodeVisible(button))
                {
                    continue;
                }

                var rewardSummary = BuildRewardButtonPayload(button);
                var rewardDescription = DescribeReward(ResolveRewardFromControlForLivePayload(button));
                var actionId = $"reward:{index}";

                actions.Add(new BridgeResolvedAction
                {
                    ActionId = actionId,
                    Payload = new
                    {
                        action_id = actionId,
                        kind = "reward",
                        index,
                        label = $"Claim reward {index}: {rewardDescription}",
                        reward = rewardSummary,
                        screen = context.Screen
                    },
                    Execute = () => InvokeButtonAction(button, "OnRelease")
                });
            }
        }

        if (context.CardRewardScreen is not null &&
            IsCardRewardSelectionReady(context.CardRewardScreen))
        {
            for (var index = 0; index < context.CardRewardOptions.Count; index++)
            {
                var cardHolder = context.CardRewardOptions[index];
                if (!IsNodeVisible(cardHolder))
                {
                    continue;
                }

                var actionId = $"card_reward:{index}";
                actions.Add(new BridgeResolvedAction
                {
                    ActionId = actionId,
                    Payload = new
                    {
                        action_id = actionId,
                        kind = "card_reward",
                        index,
                        label = $"Pick card {index}: {cardHolder.CardModel?.Title ?? "<missing>"}",
                        card = BuildCardPayload(cardHolder.CardModel),
                        screen = context.Screen
                    },
                    Execute = () => InvokeSingleArgumentAction(context.CardRewardScreen, "SelectCard", cardHolder)
                });
            }

            if (context.CardRewardSkipButton is not null &&
                IsNodeVisible(context.CardRewardSkipButton) &&
                IsButtonEnabled(context.CardRewardSkipButton))
            {
                actions.Add(new BridgeResolvedAction
                {
                    ActionId = "card_reward:skip",
                    Payload = new
                    {
                        action_id = "card_reward:skip",
                        kind = "card_reward",
                        selection_action = "skip",
                        label = "Skip card reward",
                        screen = context.Screen
                    },
                    Execute = () => InvokeCardRewardSkipAction(
                        context.CardRewardScreen,
                        context.CardRewardSkipButton)
                });
            }
        }

        if (!IsInteractiveMapSurface(context, actions))
        {
            for (var index = 0; index < context.EventOptionButtons.Count; index++)
            {
                var button = context.EventOptionButtons[index];
                var option = button.Option;
                if (!IsNodeVisible(button) || option is null || option.IsLocked)
                {
                    continue;
                }

                var actionId = $"event_option:{index}";
                actions.Add(new BridgeResolvedAction
                {
                    ActionId = actionId,
                    Payload = new
                    {
                        action_id = actionId,
                        kind = "event_option",
                        index,
                        label = $"Choose option {index}: {TextOf(option.Title)}",
                        option = BuildEventOptionPayload(
                            button,
                            index,
                            GetHiddenFieldValue(context.EventRoom, "_event") as EventModel),
                        screen = context.Screen
                    },
                    Execute = () => InvokeEventOptionAction(context.EventRoom, button, index)
                });
            }

            AddCrystalSphereEventActions(actions, context, context.EventOptionButtons.Count);
        }

        if (IsInteractiveMapSurface(context, actions))
        {
            foreach (var pointNode in context.MapPoints)
            {
                if (!IsMapPointTravelable(pointNode))
                {
                    continue;
                }

                var coord = pointNode.Point.coord;
                if (IsCurrentMapCoord(context.RunState, coord))
                {
                    continue;
                }

                var actionId = $"map:{coord.col},{coord.row}";
                actions.Add(new BridgeResolvedAction
                {
                    ActionId = actionId,
                    Payload = new
                    {
                        action_id = actionId,
                        kind = "map",
                        label = $"Travel to ({coord.col}, {coord.row}) {pointNode.Point.PointType}",
                        coord = BuildMapCoord(coord),
                        point_type = pointNode.Point.PointType.ToString(),
                        point_type_norm = NormalizeEnvMapPointType(pointNode.Point.PointType.ToString()),
                        state = pointNode.State.ToString(),
                        screen = context.Screen
                    },
                    Execute = () => InvokeMapTravelAction(context.RunManager, context.MapScreen, pointNode)
                });
            }
        }

        return actions;
    }

}
