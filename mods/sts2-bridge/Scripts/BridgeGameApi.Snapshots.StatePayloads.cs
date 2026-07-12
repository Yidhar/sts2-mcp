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
    private static BridgeStateFields BuildStateFields(BridgeWorldContext context, object[] actionPayloads)
    {
        return new BridgeStateFields
        {
            Screen = context.Screen,
            CombatInProgress = context.CombatManager?.IsInProgress == true,
            CombatIsPlayPhase = IsCombatPlayPhase(context.CombatManager, context.CombatState),
            CombatIsPaused = context.CombatManager?.IsPaused == true,
            CombatPlayerActionsDisabled = context.CombatManager?.PlayerActionsDisabled == true,
            CardSelectionVisible = IsCardSelectionVisible(context),
            Run = BuildRunPayload(context.RunState),
            Combat = BuildCombatPayload(context.CombatManager, context.CombatState),
            Players = BuildPlayersPayload(context.RunState, context.CombatManager, context.CombatState),
            Rewards = BuildRewardsPayload(
                context.RewardsScreen,
                context.ProceedButton,
                context.RewardProceedButton,
                context.MapScreen,
                context.RewardButtons),
            CardRewardSelection = BuildCardRewardSelectionPayload(
                context.CardRewardScreen,
                context.CardRewardOptions,
                context.CardRewardSkipButton),
            CardSelection = BuildCardSelectionPayload(
                context,
                context.CardSelectionScreen,
                context.CardSelectionOptions,
                context.CardSelectionConfirmButton,
                context.CardSelectionCancelButton,
                context.CardSelectionCloseButton,
                context.CardSelectionSkipButton),
            CharacterSelection = BuildCharacterSelectionPayload(
                context.CharacterSelectScreen,
                context.CharacterButtons,
                context.SelectedCharacterButton,
                context.EmbarkButton),
            RunModeSelection = BuildRunModeSelectionPayload(
                context.RunModeSubmenu,
                context.RunModeStandardButton,
                context.RunModeDailyButton,
                context.RunModeCustomButton,
                context.RunModeBackButton),
            EventOptions = BuildEventOptionsPayload(
                context.EventOptionButtons,
                context.MapScreen,
                context.EventRoom,
                context.HoverTipSet,
                context.CrystalSphereScreen,
                context.CrystalSphereCells,
                context.CrystalSphereSmallDivinationButton,
                context.CrystalSphereBigDivinationButton,
                context.CrystalSphereProceedButton),
            CrystalSphere = BuildCrystalSpherePayload(
                context.CrystalSphereScreen,
                context.CrystalSphereCells,
                context.CrystalSphereSmallDivinationButton,
                context.CrystalSphereBigDivinationButton,
                context.CrystalSphereProceedButton),
            Map = BuildMapPayload(context.RunState, context.MapScreen, context.MapPoints, context.CombatManager, context.Screen),
            RestSite = BuildRestSitePayload(
                context.MapScreen,
                context.RestSiteRoom,
                context.RestSiteButtons,
                context.RestSiteProceedButton),
            DeckUpgradeSelection = BuildDeckUpgradeSelectionPayload(
                context.DeckUpgradeScreen,
                context.DeckUpgradeOptions,
                context.DeckUpgradeConfirmButton,
                context.DeckUpgradeCancelButton,
                context.DeckUpgradeCloseButton),
            Shop = BuildShopPayload(
                context.MerchantRoom,
                context.MerchantInventory,
                context.MerchantSlots,
                context.MerchantButton,
                context.MerchantProceedButton,
                context.MerchantBackButton),
            MainMenu = BuildMainMenuPayload(
                context.MainMenuRoot,
                context.MainMenuContinueButton,
                context.MainMenuTextButtons,
                context.ContinueRunInfo,
                context.AbandonRunConfirmPopup,
                context.AbandonRunConfirmButtons),
            AvailableActions = actionPayloads
        };
    }

    private static object CreateSemanticStateCore(BridgeStateFields fields)
    {
        // Keep state_version/state_hash tied to the semantic game state rather
        // than transient automation metadata or fully-expanded action payloads.
        var rawCore = new
        {
            schema_version = BridgeRuntime.StateSchemaVersion,
            screen = fields.Screen,
            run = fields.Run,
            combat = fields.Combat,
            players = fields.Players,
            rewards = fields.Rewards,
            card_reward_selection = fields.CardRewardSelection,
            card_selection = fields.CardSelection,
            character_selection = fields.CharacterSelection,
            run_mode_selection = fields.RunModeSelection,
            event_options = fields.EventOptions,
            crystal_sphere = fields.CrystalSphere,
            map = fields.Map,
            rest_site = fields.RestSite,
            deck_upgrade_selection = fields.DeckUpgradeSelection,
            shop = fields.Shop,
            main_menu = fields.MainMenu
        };

        return PruneSemanticStateNode(JsonSerializer.SerializeToNode(rawCore, HashJsonOptions)) ?? new JsonObject();
    }

    private static object CreateStatePayload(
        BridgeStateFields fields,
        long stateVersion,
        string stateHash,
        string semanticStateHash)
    {
        return new
        {
            ok = true,
            bridge_version = BridgeRuntime.BridgeVersion,
            schema_version = BridgeRuntime.StateSchemaVersion,
            state_version = stateVersion,
            state_hash = stateHash,
            semantic_state_hash = semanticStateHash,
            captured_at_utc = DateTimeOffset.UtcNow,
            screen = fields.Screen,
            run = fields.Run,
            combat = fields.Combat,
            players = fields.Players,
            rewards = fields.Rewards,
            card_reward_selection = fields.CardRewardSelection,
            card_selection = fields.CardSelection,
            character_selection = fields.CharacterSelection,
            run_mode_selection = fields.RunModeSelection,
            event_options = fields.EventOptions,
            crystal_sphere = fields.CrystalSphere,
            map = fields.Map,
            rest_site = fields.RestSite,
            deck_upgrade_selection = fields.DeckUpgradeSelection,
            shop = fields.Shop,
            main_menu = fields.MainMenu,
            available_actions = fields.AvailableActions
        };
    }
}
