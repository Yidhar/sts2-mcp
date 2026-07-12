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
    private static object BuildCombatFrontierPayload(CombatManager? combatManager, CombatState? combatState)
    {
        if (combatManager is null || combatState is null || !combatManager.IsInProgress)
        {
            return new
            {
                in_progress = false
            };
        }

        return new
        {
            in_progress = combatManager.IsInProgress,
            is_play_phase = IsCombatPlayPhase(combatManager, combatState),
            is_paused = combatManager.IsPaused,
            is_ending = combatManager.IsEnding,
            player_actions_disabled = combatManager.PlayerActionsDisabled,
            round_number = combatState.RoundNumber,
            current_side = combatState.CurrentSide.ToString(),
            players = combatState.Players.Select(player => new
            {
                net_id = player.NetId,
                energy = player.PlayerCombatState?.Energy,
                max_energy = player.PlayerCombatState?.MaxEnergy,
                stars = player.PlayerCombatState?.Stars,
                creature = BuildCreatureFrontierPayload(player.Creature)
            }).ToArray(),
            enemies = combatState.Creatures
                .Where(static creature => creature.IsEnemy)
                .Select(BuildCreatureFrontierPayload)
                .ToArray()
        };
    }

    private static object BuildCreatureFrontierPayload(Creature? creature)
    {
        if (creature is null)
        {
            return new
            {
                missing = true
            };
        }

        return new
        {
            combat_id = creature.CombatId,
            current_hp = creature.CurrentHp,
            max_hp = creature.MaxHp,
            block = creature.Block,
            is_alive = creature.IsAlive,
            is_hittable = SafeGetCreatureIsHittable(creature),
            powers = creature.Powers.Select(BuildPowerFrontierPayload).ToArray(),
            intent = creature.IsEnemy ? BuildEnemyIntentFrontierPayload(creature) : null
        };
    }

    private static object BuildPowerFrontierPayload(PowerModel power)
    {
        var modelId = power.Id.ToString();
        var className = power.GetType().Name;
        return new
        {
            id = modelId,
            model_id = modelId,
            class_name = className,
            kind = className,
            title = TextOf(power.Title),
            amount = power.Amount,
            display_amount = power.DisplayAmount
        };
    }

    private static object? BuildEnemyIntentFrontierPayload(Creature creature)
    {
        var monster = creature.Monster;
        if (monster is null)
        {
            return null;
        }

        var targets = ResolveMonsterIntentTargets(creature);
        var nextMove = monster.NextMove;
        return new
        {
            state_id = nextMove?.StateId,
            follow_up_state_id = nextMove?.FollowUpStateId,
            intents = SafeGetMonsterIntents(monster, nextMove)
                .Select(intent =>
                {
                    var repeats = intent switch
                    {
                        SingleAttackIntent singleAttackIntent => singleAttackIntent.Repeats,
                        MultiAttackIntent multiAttackIntent => multiAttackIntent.Repeats,
                        _ => 1
                    };

                    var totalDamage = intent switch
                    {
                        SingleAttackIntent singleAttackIntent =>
                            SafeGetIntentTotalDamage(singleAttackIntent, targets, creature),
                        MultiAttackIntent multiAttackIntent =>
                            SafeGetIntentTotalDamage(multiAttackIntent, targets, creature),
                        _ => null
                    };

                    return new
                    {
                        intent_type = intent.IntentType.ToString(),
                        repeats,
                        total_damage = totalDamage
                    };
                })
                .ToArray()
        };
    }

    private static object BuildRewardsFrontierPayload(BridgeWorldContext context)
    {
        var visible = IsRewardsScreenVisible(
            context.RewardsScreen,
            context.ProceedButton,
            context.RewardProceedButton,
            context.MapScreen,
            context.RewardButtons);
        return new
        {
            visible,
            terminal_proceed_visible = visible &&
                                       context.RewardProceedButton is not null &&
                                       IsNodeVisible(context.RewardProceedButton),
            reward_count = context.RewardButtons.Count
        };
    }

    private static object BuildCardRewardSelectionFrontierPayload(
        NCardRewardSelectionScreen? cardRewardScreen,
        IReadOnlyList<NCardHolder> cardRewardOptions,
        Node? cardRewardSkipButton)
    {
        var visible = IsCardRewardSelectionVisible(cardRewardScreen, cardRewardOptions);
        var ready = IsCardRewardSelectionReady(cardRewardScreen);
        return new
        {
            visible,
            ready,
            skip_visible = ready &&
                           cardRewardSkipButton is not null &&
                           IsNodeVisible(cardRewardSkipButton) &&
                           IsButtonEnabled(cardRewardSkipButton),
            option_count = cardRewardOptions.Count
        };
    }

    private static object BuildCardSelectionFrontierPayload(
        Node? cardSelectionScreen,
        IReadOnlyList<NCardHolder> cardSelectionOptions,
        Node? cardSelectionConfirmButton,
        Node? cardSelectionCancelButton,
        Node? cardSelectionCloseButton,
        Node? cardSelectionSkipButton)
    {
        var visible = cardSelectionScreen is not null && IsNodeVisible(cardSelectionScreen);
        var prefs = GetHiddenFieldValue(cardSelectionScreen, "_prefs");

        return new
        {
            visible,
            screen_type = visible ? cardSelectionScreen!.GetType().Name : null,
            selected_count = CountSelectedCardSelectionCards(cardSelectionScreen),
            min_select = GetHiddenPropertyValue<int>(prefs, "MinSelect"),
            max_select = GetHiddenPropertyValue<int>(prefs, "MaxSelect"),
            confirm_visible = cardSelectionConfirmButton is not null &&
                              IsNodeVisible(cardSelectionConfirmButton) &&
                              IsButtonEnabled(cardSelectionConfirmButton),
            cancel_visible = cardSelectionCancelButton is not null &&
                             IsNodeVisible(cardSelectionCancelButton) &&
                             IsButtonEnabled(cardSelectionCancelButton),
            close_visible = cardSelectionCloseButton is not null &&
                            IsNodeVisible(cardSelectionCloseButton) &&
                            IsButtonEnabled(cardSelectionCloseButton),
            skip_visible = cardSelectionSkipButton is not null &&
                           IsNodeVisible(cardSelectionSkipButton) &&
                           IsButtonEnabled(cardSelectionSkipButton),
            option_keys = cardSelectionOptions.Select((holder, index) =>
            {
                var optionIndex = GetCardSelectionOptionIndex(cardSelectionScreen, holder, index);
                return GetCardSelectionOptionSelectionId(cardSelectionScreen, holder, optionIndex) ??
                       optionIndex.ToString(CultureInfo.InvariantCulture);
            }).ToArray()
        };
    }

    private static object BuildCrystalSphereFrontierPayload(
        NCrystalSphereScreen? crystalSphereScreen,
        IReadOnlyList<NCrystalSphereCell> crystalSphereCells)
    {
        var visible = crystalSphereScreen is not null && IsNodeVisible(crystalSphereScreen);
        if (!visible)
        {
            return new
            {
                visible = false
            };
        }

        var minigame = GetCrystalSphereMinigame(crystalSphereScreen);
        return new
        {
            visible = true,
            divinations_left = GetCrystalSphereDivinationCount(minigame),
            current_tool = GetCrystalSphereToolName(minigame),
            is_finished = GetCrystalSphereIsFinished(minigame),
            cells = crystalSphereCells.Select(cell => new
            {
                x = cell.Entity?.X,
                y = cell.Entity?.Y,
                is_hidden = cell.Entity?.IsHidden ?? true,
                is_highlighted = cell.Entity?.IsHighlighted ?? false
            }).ToArray()
        };
    }

    private static object BuildMapFrontierPayload(
        RunState? runState,
        NMapScreen? mapScreen,
        string currentScreen,
        CombatManager? combatManager)
    {
        var rawIsOpen = mapScreen?.IsOpen ?? false;
        var rawIsTravelEnabled = mapScreen?.IsTravelEnabled ?? false;
        var rawIsTraveling = mapScreen?.IsTraveling ?? false;
        var interactiveSurface = string.Equals(currentScreen, "MAP", StringComparison.Ordinal) &&
                                 IsInteractiveMapSurface(mapScreen, combatManager);

        return new
        {
            is_open = interactiveSurface,
            is_travel_enabled = interactiveSurface && rawIsTravelEnabled,
            is_traveling = rawIsTraveling,
            is_open_raw = rawIsOpen,
            is_travel_enabled_raw = rawIsTravelEnabled,
            is_interactive_surface = interactiveSurface,
            current_coord = BuildMapCoord(runState?.CurrentMapCoord)
        };
    }

    private static bool HasRawInteractiveMapSurface(NMapScreen? mapScreen)
    {
        return mapScreen is not null &&
               mapScreen.IsOpen &&
               mapScreen.IsTravelEnabled &&
               !mapScreen.IsTraveling;
    }

    private static bool IsInteractiveMapSurface(
        NMapScreen? mapScreen,
        CombatManager? combatManager = null)
    {
        if (!HasRawInteractiveMapSurface(mapScreen))
        {
            return false;
        }

        combatManager ??= CombatManager.Instance;
        return combatManager?.IsInProgress != true;
    }

    private static bool IsInteractiveMapSurface(
        BridgeWorldContext context,
        IReadOnlyList<BridgeResolvedAction>? actions = null)
    {
        if (!IsInteractiveMapSurface(context.MapScreen, context.CombatManager))
        {
            return false;
        }

        return string.Equals(context.Screen, "MAP", StringComparison.Ordinal);
    }

    private static object BuildRestSiteFrontierPayload(
        NMapScreen? mapScreen,
        NRestSiteRoom? restSiteRoom,
        IReadOnlyList<NRestSiteButton> restSiteButtons,
        NProceedButton? restSiteProceedButton)
    {
        var visible = !IsInteractiveMapSurface(mapScreen) &&
                      restSiteRoom is not null &&
                      IsNodeVisible(restSiteRoom);
        return new
        {
            visible,
            proceed_visible = visible &&
                              !HasVisibleEnabledRestSiteOptions(restSiteButtons) &&
                              restSiteProceedButton is not null &&
                              IsNodeVisible(restSiteProceedButton) &&
                              IsButtonEnabled(restSiteProceedButton),
            option_count = visible ? restSiteButtons.Count(IsNodeVisible) : 0
        };
    }

    private static object BuildDeckUpgradeSelectionFrontierPayload(
        NDeckUpgradeSelectScreen? deckUpgradeScreen,
        IReadOnlyList<NCardHolder> deckUpgradeOptions,
        NConfirmButton? deckUpgradeConfirmButton,
        NBackButton? deckUpgradeCancelButton,
        NBackButton? deckUpgradeCloseButton)
    {
        var visible = deckUpgradeScreen is not null && IsNodeVisible(deckUpgradeScreen);
        return new
        {
            visible,
            selected_count = CountSelectedDeckUpgradeCards(deckUpgradeScreen),
            confirm_visible = deckUpgradeConfirmButton is not null &&
                              IsNodeVisible(deckUpgradeConfirmButton) &&
                              IsButtonEnabled(deckUpgradeConfirmButton),
            cancel_visible = deckUpgradeCancelButton is not null &&
                             IsNodeVisible(deckUpgradeCancelButton) &&
                             IsButtonEnabled(deckUpgradeCancelButton),
            close_visible = deckUpgradeCloseButton is not null &&
                            IsNodeVisible(deckUpgradeCloseButton) &&
                            IsButtonEnabled(deckUpgradeCloseButton),
            option_count = visible ? deckUpgradeOptions.Count : 0
        };
    }

    private static object BuildShopFrontierPayload(
        NMerchantRoom? merchantRoom,
        NMerchantInventory? merchantInventory,
        IReadOnlyList<NMerchantSlot> merchantSlots,
        NMerchantButton? merchantButton,
        NProceedButton? merchantProceedButton,
        NBackButton? merchantBackButton)
    {
        return new
        {
            visible = (merchantRoom is not null && IsNodeVisible(merchantRoom)) ||
                      (merchantInventory is not null && IsNodeVisible(merchantInventory)),
            is_open = merchantInventory?.IsOpen ?? false,
            merchant_button_visible = merchantButton is not null && IsNodeVisible(merchantButton),
            back_button_visible = merchantBackButton is not null && IsNodeVisible(merchantBackButton),
            proceed_visible = merchantProceedButton is not null && IsNodeVisible(merchantProceedButton),
            item_count = merchantSlots.Count
        };
    }

}
