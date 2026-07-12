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
    private sealed class BridgeWorldContext
    {
        public required NGame Game { get; init; }

        public NRun? RunNode { get; init; }

        public RunManager? RunManager { get; init; }

        public CombatManager? CombatManager { get; init; }

        public RunState? RunState { get; init; }

        public CombatState? CombatState { get; init; }

        public required string Screen { get; init; }

        public NCombatRoom? CombatRoom { get; init; }

        public NCombatUi? CombatUi { get; init; }

        public NEndTurnButton? EndTurnButton { get; init; }

        public NProceedButton? ProceedButton { get; init; }

        public NMapScreen? MapScreen { get; init; }

        public NRestSiteRoom? RestSiteRoom { get; init; }

        public NMerchantRoom? MerchantRoom { get; init; }

        public NMerchantInventory? MerchantInventory { get; init; }

        public NTreasureRoom? TreasureRoom { get; init; }

        public NTreasureButton? TreasureChestButton { get; init; }

        public NTreasureRoomRelicCollection? TreasureRelicCollection { get; init; }

        public NRewardsScreen? RewardsScreen { get; init; }

        public NProceedButton? RewardProceedButton { get; init; }

        public NCardRewardSelectionScreen? CardRewardScreen { get; init; }

        public Node? CardRewardSkipButton { get; init; }

        public Node? CardSelectionScreen { get; init; }

        public NCharacterSelectScreen? CharacterSelectScreen { get; init; }

        public NDeckUpgradeSelectScreen? DeckUpgradeScreen { get; init; }

        public NProceedButton? RestSiteProceedButton { get; init; }

        public NMerchantButton? MerchantButton { get; init; }

        public NProceedButton? MerchantProceedButton { get; init; }

        public NBackButton? MerchantBackButton { get; init; }

        public NCharacterSelectButton? SelectedCharacterButton { get; init; }

        public NConfirmButton? EmbarkButton { get; init; }

        public Node? CardSelectionConfirmButton { get; init; }

        public Node? CardSelectionCancelButton { get; init; }

        public Node? CardSelectionCloseButton { get; init; }

        public Node? CardSelectionSkipButton { get; init; }

        public NConfirmButton? DeckUpgradeConfirmButton { get; init; }

        public NBackButton? DeckUpgradeCancelButton { get; init; }

        public NBackButton? DeckUpgradeCloseButton { get; init; }

        public required IReadOnlyList<NRewardButton> RewardButtons { get; init; }

        public required IReadOnlyList<NCardHolder> CardRewardOptions { get; init; }

        public required IReadOnlyList<NCardHolder> CardSelectionOptions { get; init; }

        public required IReadOnlyList<NCardHolder> DeckUpgradeOptions { get; init; }

        public required IReadOnlyList<NCharacterSelectButton> CharacterButtons { get; init; }

        public required IReadOnlyList<NEventOptionButton> EventOptionButtons { get; init; }

        public NEventRoom? EventRoom { get; init; }

        public NGameOverScreen? GameOverScreen { get; init; }

        public NGameOverContinueButton? GameOverContinueButton { get; init; }

        public NReturnToMainMenuButton? GameOverMainMenuButton { get; init; }

        public NCrystalSphereScreen? CrystalSphereScreen { get; init; }

        public required IReadOnlyList<NCrystalSphereCell> CrystalSphereCells { get; init; }

        public NDivinationButton? CrystalSphereSmallDivinationButton { get; init; }

        public NDivinationButton? CrystalSphereBigDivinationButton { get; init; }

        public NProceedButton? CrystalSphereProceedButton { get; init; }

        public Node? HoverTipSet { get; init; }

        public required IReadOnlyList<NMapPoint> MapPoints { get; init; }

        public required IReadOnlyList<NRestSiteButton> RestSiteButtons { get; init; }

        public required IReadOnlyList<NMerchantSlot> MerchantSlots { get; init; }

        public required IReadOnlyList<NTreasureRoomRelicHolder> TreasureRelicOptions { get; init; }

        public Node? MainMenuRoot { get; init; }

        public Node? MainMenuContinueButton { get; init; }

        public required IReadOnlyList<Node> MainMenuTextButtons { get; init; }

        public Node? RunModeSubmenu { get; init; }

        public Node? RunModeStandardButton { get; init; }

        public Node? RunModeDailyButton { get; init; }

        public Node? RunModeCustomButton { get; init; }

        public NBackButton? RunModeBackButton { get; init; }

        public Node? ContinueRunInfo { get; init; }

        public Node? AbandonRunConfirmPopup { get; init; }

        public required IReadOnlyList<NPopupYesNoButton> AbandonRunConfirmButtons { get; init; }
    }

    private sealed class BridgeResolvedAction
    {
        public required string ActionId { get; init; }

        public required object Payload { get; init; }

        public required Action Execute { get; init; }
    }

    private sealed class ResolvedCardTarget
    {
        public string? ActionSuffix { get; init; }

        public string? LabelSuffix { get; init; }

        public Creature? Target { get; init; }

        public required bool RequiresTargetSelection { get; init; }
    }

    private sealed class ResolvedPotionTarget
    {
        public string? ActionSuffix { get; init; }

        public string? LabelSuffix { get; init; }

        public Creature? Target { get; init; }

        public required bool RequiresTargetSelection { get; init; }
    }

    private sealed record CrystalSphereEventOptionDescriptor
    {
        public required int Index { get; init; }

        public required string OptionType { get; init; }

        public string? OptionId { get; init; }

        public required string Title { get; init; }

        public string? Description { get; init; }

        public bool IsProceed { get; init; }

        public bool IsSelected { get; init; }

        public bool IsEnabled { get; init; }

        public bool ActionAvailable { get; init; }

        public string? DivinationSize { get; init; }

        public int? X { get; init; }

        public int? Y { get; init; }

        public bool? IsHighlighted { get; init; }

        public Action? Execute { get; init; }
    }

    private sealed class BridgeStateFields
    {
        public required string Screen { get; init; }

        public required bool CombatInProgress { get; init; }

        public required bool CombatIsPlayPhase { get; init; }

        public required bool CombatIsPaused { get; init; }

        public required bool CombatPlayerActionsDisabled { get; init; }

        public required bool CardSelectionVisible { get; init; }


        public required object Run { get; init; }

        public required object Combat { get; init; }

        public required object[] Players { get; init; }

        public required object Rewards { get; init; }

        public required object CardRewardSelection { get; init; }

        public required object CardSelection { get; init; }

        public required object CharacterSelection { get; init; }

        public required object RunModeSelection { get; init; }

        public required object EventOptions { get; init; }

        public required object CrystalSphere { get; init; }

        public required object Map { get; init; }

        public required object RestSite { get; init; }

        public required object DeckUpgradeSelection { get; init; }

        public required object Shop { get; init; }

        public required object MainMenu { get; init; }

        public required object[] AvailableActions { get; init; }
    }

    private sealed class BridgeSnapshot
    {
        public required string FrontierHash { get; init; }

        public required BridgeStateFields Fields { get; init; }

        public required IReadOnlyList<BridgeResolvedAction> Actions { get; init; }

        public required IReadOnlyList<object> ActionPayloads { get; init; }

        public required IReadOnlyDictionary<string, BridgeResolvedAction> ActionLookup { get; init; }
    }
}
