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
    private static BridgeWorldContext CaptureContext()
    {
        var game = NGame.Instance;
        if (game is null || !GodotObject.IsInstanceValid(game))
        {
            throw new BridgeRequestException(
                HttpStatusCode.ServiceUnavailable,
                "game_not_ready",
                "NGame.Instance is not available yet.");
        }

        var runNode = game.CurrentRunNode ?? NRun.Instance;
        var runManager = RunManager.Instance;
        var combatManager = CombatManager.Instance;
        var runState = TryGetRunState(runManager);
        var combatState = TryGetCombatState(combatManager);
        var combatRoom = runNode?.CombatRoom;
        var combatUi = combatRoom?.Ui;
        var playerHand = combatUi?.Hand;
        var endTurnButton = combatUi?.EndTurnButton;
        var activeScreen = ActiveScreenContext.Instance.GetCurrentScreen();
        var activeScreenNode = activeScreen as Node;
        var overlayStack = NOverlayStack.Instance;
        var overlayRoot = overlayStack as Node;
        var mapScreen = NMapScreen.Instance;
        var mainMenuRoot = game.MainMenu;
        var mainMenuSubmenu = ResolveMainMenuSubmenu(activeScreen, mainMenuRoot);
        var restSiteRoom = NRestSiteRoom.Instance;
        var restSiteProceedButton = restSiteRoom?.ProceedButton;
        var merchantRoom = NMerchantRoom.Instance;
        var merchantInventory = merchantRoom?.Inventory;
        var merchantButton = merchantRoom?.MerchantButton;
        var merchantProceedButton = merchantRoom?.ProceedButton;
        var merchantBackButton = merchantInventory?.GetNodeOrNull<NBackButton>("%BackButton") ??
                                 GetHiddenFieldValue(merchantInventory, "_backButton") as NBackButton;
        var treasureRoom = runNode?.TreasureRoom;
        var treasureChestButton = ResolveTreasureChestButton(treasureRoom);
        var treasureRelicCollection = ResolveTreasureRelicCollection(treasureRoom);
        var treasureRelicOptions = ResolveTreasureRelicOptions(treasureRelicCollection);
        var rewardsScreen = ResolveOverlayScreen<NRewardsScreen>(activeScreen, overlayStack);
        var rewardProceedButton = rewardsScreen?.GetNodeOrNull<NProceedButton>("ProceedButton") ??
                                  GetHiddenFieldValue(rewardsScreen, "_proceedButton") as NProceedButton;
        var proceedButton = ResolveStableProceedButton(
            activeScreen,
            activeScreenNode ?? overlayRoot ?? game,
            combatRoom?.ProceedButton,
            treasureRoom?.ProceedButton,
            restSiteProceedButton,
            merchantProceedButton);
        var merchantSlots = merchantInventory is null
            ? new List<NMerchantSlot>()
            : SortByVisualPosition(merchantInventory.GetAllSlots().Where(IsNodeVisible));
        var cardRewardScreen = ResolveOverlayScreen<NCardRewardSelectionScreen>(activeScreen, overlayStack);
        var characterSelectScreen = activeScreen as NCharacterSelectScreen ??
                                    mainMenuSubmenu as NCharacterSelectScreen;
        var deckUpgradeScreen = ResolveOverlayScreen<NDeckUpgradeSelectScreen>(activeScreen, overlayStack);
        var cardRewardSkipButton = ResolveCardRewardSkipButton(cardRewardScreen);
        var cardSelectionScreen = ResolveStableCardSelectionScreen(
            activeScreen,
            overlayStack,
            cardRewardScreen,
            deckUpgradeScreen,
            playerHand);
        var restSiteButtons = ResolveRestSiteButtons(restSiteRoom);
        var characterButtons = ResolveCharacterButtons(characterSelectScreen);
        var selectedCharacterButton = GetHiddenFieldValue(characterSelectScreen, "_selectedButton") as NCharacterSelectButton;
        var embarkButton = characterSelectScreen?.GetNodeOrNull<NConfirmButton>("%EmbarkButton") ??
                           GetHiddenFieldValue(characterSelectScreen, "_embarkButton") as NConfirmButton;
        var rewardButtons = ResolveRewardButtons(rewardsScreen);
        var cardRewardOptions = ResolveCardRewardOptions(cardRewardScreen);
        var deckUpgradeOptions = ResolveDeckUpgradeOptions(deckUpgradeScreen);
        var cardSelectionOptions = GetCardSelectionOptions(cardSelectionScreen);
        var deckUpgradeCancelButton = ResolveFirstVisibleNode(
            deckUpgradeScreen?.GetNodeOrNull<NBackButton>("%UpgradeSinglePreviewContainer/Cancel"),
            deckUpgradeScreen?.GetNodeOrNull<NBackButton>("%UpgradeMultiPreviewContainer/Cancel"),
            GetHiddenFieldValue(deckUpgradeScreen, "_singlePreviewCancelButton") as NBackButton,
            GetHiddenFieldValue(deckUpgradeScreen, "_multiPreviewCancelButton") as NBackButton);
        var deckUpgradeConfirmButton = ResolveFirstVisibleNode(
            deckUpgradeScreen?.GetNodeOrNull<NConfirmButton>("%UpgradeSinglePreviewContainer/Confirm"),
            deckUpgradeScreen?.GetNodeOrNull<NConfirmButton>("%UpgradeMultiPreviewContainer/Confirm"),
            GetHiddenFieldValue(deckUpgradeScreen, "_singlePreviewConfirmButton") as NConfirmButton,
            GetHiddenFieldValue(deckUpgradeScreen, "_multiPreviewConfirmButton") as NConfirmButton);
        var deckUpgradeCloseButton = deckUpgradeScreen?.GetNodeOrNull<NBackButton>("%Close") ??
                                     GetHiddenFieldValue(deckUpgradeScreen, "_closeButton") as NBackButton;
        var cardSelectionConfirmButton = ResolveCardSelectionConfirmButton(cardSelectionScreen);
        var cardSelectionCancelButton = ResolveCardSelectionCancelButton(cardSelectionScreen);
        var cardSelectionCloseButton = GetHiddenFieldValue(cardSelectionScreen, "_closeButton") as Node;
        var cardSelectionSkipButton = GetHiddenFieldValue(cardSelectionScreen, "_skipButton") as Node;
        var eventRoom = runNode?.EventRoom;
        var gameOverScreen = activeScreen as NGameOverScreen ??
                             (activeScreen is null ? overlayStack?.Peek() as NGameOverScreen : null);
        var gameOverContinueButton = GetHiddenFieldValue(gameOverScreen, "_continueButton") as NGameOverContinueButton;
        var gameOverMainMenuButton = GetHiddenFieldValue(gameOverScreen, "_mainMenuButton") as NReturnToMainMenuButton;
        var crystalSphereScreen = ResolveOverlayScreen<NCrystalSphereScreen>(activeScreen, overlayStack);
        var crystalSphereCells = ResolveCrystalSphereCells(crystalSphereScreen);
        var crystalSphereSmallDivinationButton =
            crystalSphereScreen?.GetNodeOrNull<NDivinationButton>("%SmallDivinationButton") ??
            GetHiddenFieldValue(crystalSphereScreen, "_smallDivinationButton") as NDivinationButton;
        var crystalSphereBigDivinationButton =
            crystalSphereScreen?.GetNodeOrNull<NDivinationButton>("%BigDivinationButton") ??
            GetHiddenFieldValue(crystalSphereScreen, "_bigDivinationButton") as NDivinationButton;
        var crystalSphereProceedButton =
            crystalSphereScreen?.GetNodeOrNull<NProceedButton>("%ProceedButton") ??
            GetHiddenFieldValue(crystalSphereScreen, "_proceedButton") as NProceedButton;
        var hoverTipSet = ResolveVisibleHoverTipSet(game);
        var eventOptionSearchRoot = ResolveEventOptionSearchRoot(activeScreen, eventRoom);
        var eventOptionButtons = (eventOptionSearchRoot is null
            ? new List<NEventOptionButton>()
            : SortByVisualPosition(FindVisibleDescendants<NEventOptionButton>(eventOptionSearchRoot)))
            .Where(static button => button.Option is not null)
            .ToList();
        RefreshInteractiveMapTravelability(mapScreen);
        var mapPoints = ResolveMapPoints(mapScreen);
        var mainMenuContinueButton = GetHiddenFieldValue(mainMenuRoot, "_continueButton") as Node;
        var mainMenuTextButtons = ResolveMainMenuTextButtons(mainMenuRoot);
        var runModeSubmenu = mainMenuSubmenu as NSingleplayerSubmenu;
        var runModeStandardButton = runModeSubmenu?.GetNodeOrNull<Node>("StandardButton") ??
                                    GetHiddenFieldValue(runModeSubmenu, "_standardButton") as Node;
        var runModeDailyButton = runModeSubmenu?.GetNodeOrNull<Node>("DailyButton") ??
                                 GetHiddenFieldValue(runModeSubmenu, "_dailyButton") as Node;
        var runModeCustomButton = runModeSubmenu?.GetNodeOrNull<Node>("CustomRunButton") ??
                                  GetHiddenFieldValue(runModeSubmenu, "_customButton") as Node;
        var runModeBackButton = runModeSubmenu?.GetNodeOrNull<NBackButton>("BackButton") ??
                                GetHiddenFieldValue(runModeSubmenu, "_backButton") as NBackButton;
        var continueRunInfo = mainMenuRoot?.ContinueRunInfo;
        var abandonRunConfirmPopup = activeScreen as NAbandonRunConfirmPopup ??
                                     NModalContainer.Instance?.OpenModal as NAbandonRunConfirmPopup;
        var abandonRunConfirmButtons = ResolveAbandonRunConfirmButtons(abandonRunConfirmPopup);

        return new BridgeWorldContext
        {
            Game = game,
            RunNode = runNode,
            RunManager = runManager,
            CombatManager = combatManager,
            RunState = runState,
            CombatState = combatState,
            Screen = ResolveCurrentScreen(
                activeScreen,
                combatManager,
                mapScreen,
                characterSelectScreen,
                mainMenuRoot,
                runModeSubmenu,
                abandonRunConfirmPopup),
            CombatRoom = combatRoom,
            CombatUi = combatUi,
            EndTurnButton = endTurnButton,
            ProceedButton = proceedButton,
            MapScreen = mapScreen,
            RestSiteRoom = restSiteRoom,
            MerchantRoom = merchantRoom,
            MerchantInventory = merchantInventory,
            TreasureRoom = treasureRoom,
            TreasureChestButton = treasureChestButton,
            TreasureRelicCollection = treasureRelicCollection,
            RewardsScreen = rewardsScreen,
            RewardProceedButton = rewardProceedButton,
            CardRewardScreen = cardRewardScreen,
            CardRewardSkipButton = cardRewardSkipButton,
            CardSelectionScreen = cardSelectionScreen,
            CharacterSelectScreen = characterSelectScreen,
            DeckUpgradeScreen = deckUpgradeScreen,
            RestSiteProceedButton = restSiteProceedButton,
            MerchantButton = merchantButton,
            MerchantProceedButton = merchantProceedButton,
            MerchantBackButton = merchantBackButton,
            SelectedCharacterButton = selectedCharacterButton,
            EmbarkButton = embarkButton,
            RewardButtons = rewardButtons,
            CardRewardOptions = cardRewardOptions,
            CardSelectionOptions = cardSelectionOptions,
            DeckUpgradeOptions = deckUpgradeOptions,
            CharacterButtons = characterButtons,
            EventOptionButtons = eventOptionButtons,
            EventRoom = eventRoom,
            GameOverScreen = gameOverScreen,
            GameOverContinueButton = gameOverContinueButton,
            GameOverMainMenuButton = gameOverMainMenuButton,
            CrystalSphereScreen = crystalSphereScreen,
            CrystalSphereCells = crystalSphereCells,
            CrystalSphereSmallDivinationButton = crystalSphereSmallDivinationButton,
            CrystalSphereBigDivinationButton = crystalSphereBigDivinationButton,
            CrystalSphereProceedButton = crystalSphereProceedButton,
            HoverTipSet = hoverTipSet,
            MapPoints = mapPoints,
            RestSiteButtons = restSiteButtons,
            MerchantSlots = merchantSlots,
            TreasureRelicOptions = treasureRelicOptions,
            MainMenuRoot = mainMenuRoot,
            MainMenuContinueButton = mainMenuContinueButton,
            MainMenuTextButtons = mainMenuTextButtons,
            RunModeSubmenu = runModeSubmenu,
            RunModeStandardButton = runModeStandardButton,
            RunModeDailyButton = runModeDailyButton,
            RunModeCustomButton = runModeCustomButton,
            RunModeBackButton = runModeBackButton,
            ContinueRunInfo = continueRunInfo,
            AbandonRunConfirmPopup = abandonRunConfirmPopup,
            AbandonRunConfirmButtons = abandonRunConfirmButtons,
            CardSelectionConfirmButton = cardSelectionConfirmButton,
            CardSelectionCancelButton = cardSelectionCancelButton,
            CardSelectionCloseButton = cardSelectionCloseButton,
            CardSelectionSkipButton = cardSelectionSkipButton,
            DeckUpgradeCancelButton = deckUpgradeCancelButton,
            DeckUpgradeConfirmButton = deckUpgradeConfirmButton,
            DeckUpgradeCloseButton = deckUpgradeCloseButton
        };
    }

}
