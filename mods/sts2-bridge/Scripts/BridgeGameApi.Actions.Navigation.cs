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
    private static void InvokeMenuButtonAction(Node button)
    {
        if (TryInvokeParameterless(button, "ForceClick") ||
            TryInvokeParameterless(button, "OnRelease") ||
            TryInvokeParameterless(button, "OnPress") ||
            TryInvokeParameterless(button, "Pressed") ||
            TryInvokeParameterless(button, "OnButtonPressed"))
        {
            return;
        }

        throw new BridgeRequestException(
            HttpStatusCode.Conflict,
            "action_target_missing",
            $"Could not invoke a supported main-menu action on {button.GetType().FullName}.");
    }

    private static void InvokeCharacterSelectAction(
        NCharacterSelectScreen? characterSelectScreen,
        NCharacterSelectButton button)
    {
        if (characterSelectScreen is not null &&
            button.Character is not null &&
            TryInvokeTwoArguments(characterSelectScreen, "SelectCharacter", button, button.Character))
        {
            return;
        }

        if (TryInvokeParameterless(button, "Select") ||
            TryInvokeParameterless(button, "OnPress"))
        {
            return;
        }

        InvokeButtonAction(button, "Select", "OnPress");
    }

    private static void InvokeEmbarkAction(
        NCharacterSelectScreen? characterSelectScreen,
        NConfirmButton embarkButton)
    {
        if (characterSelectScreen is not null &&
            TryInvokeSingleArgument(characterSelectScreen, "OnEmbarkPressed", embarkButton))
        {
            return;
        }

        if (TryInvokeParameterless(embarkButton, "ForceClick") ||
            TryInvokeParameterless(embarkButton, "OnRelease"))
        {
            return;
        }

        InvokeButtonAction(embarkButton, "ForceClick", "OnRelease");
    }

    private static void InvokeMainMenuContinueAction(
        Node? mainMenuRoot,
        Node? continueButton)
    {
        if (mainMenuRoot is not null &&
            continueButton is not null &&
            TryInvokeSingleArgument(mainMenuRoot, "OnContinueButtonPressed", continueButton))
        {
            return;
        }

        if (continueButton is not null)
        {
            InvokeMenuButtonAction(continueButton);
            return;
        }

        throw new BridgeRequestException(
            HttpStatusCode.Conflict,
            "action_target_missing",
            "Could not invoke the main-menu continue action.");
    }

    private static void InvokeAbandonRunConfirmAction(
        Node? abandonRunConfirmPopup,
        NPopupYesNoButton? button,
        bool confirm)
    {
        var methodName = confirm ? "OnYesButtonPressed" : "OnNoButtonPressed";
        if (abandonRunConfirmPopup is not null &&
            button is not null &&
            TryInvokeSingleArgument(abandonRunConfirmPopup, methodName, button))
        {
            return;
        }

        if (button is not null)
        {
            InvokeMenuButtonAction(button);
            return;
        }

        throw new BridgeRequestException(
            HttpStatusCode.Conflict,
            "action_target_missing",
            $"Could not invoke abandon-run confirmation action '{methodName}'.");
    }

    private static void InvokeClickablePressAndRelease(object target)
    {
        var didInvoke = false;

        if (TryInvokeParameterless(target, "OnPress"))
        {
            didInvoke = true;
        }

        if (TryInvokeParameterless(target, "OnRelease"))
        {
            didInvoke = true;
        }

        if (didInvoke || TryInvokeParameterless(target, "ForceClick"))
        {
            return;
        }

        throw new BridgeRequestException(
            HttpStatusCode.Conflict,
            "action_target_missing",
            $"Could not invoke click lifecycle on {target.GetType().FullName}.");
    }

    private static void InvokeProceedButtonAction(NProceedButton button)
    {
        InvokeClickablePressAndRelease(button);
    }

    private static void InvokeEventOptionAction(
        NEventRoom? eventRoom,
        NEventOptionButton button,
        int index)
    {
        if (button.Option?.IsProceed == true &&
            eventRoom is not null &&
            button.Option is not null)
        {
            TryInvokeSingleArgument(eventRoom, "BeforeOptionChosen", button.Option);
            if (TryInvokeTwoArguments(eventRoom, "OptionButtonClicked", button.Option, index))
            {
                return;
            }
        }

        InvokeButtonAction(button, "OnRelease");
    }

    private static void InvokeMapTravelAction(
        RunManager? runManager,
        NMapScreen? mapScreen,
        NMapPoint pointNode)
    {
        var coord = pointNode.Point.coord;

        if (runManager is not null &&
            TryInvokeSingleArgument(runManager, "EnterMapCoord", coord))
        {
            return;
        }

        if (mapScreen is not null &&
            TryInvokeSingleArgument(mapScreen, "TravelToMapCoord", coord))
        {
            return;
        }

        if (mapScreen is not null &&
            TryInvokeSingleArgument(mapScreen, "OnMapPointSelectedLocally", pointNode))
        {
            return;
        }

        InvokeButtonAction(pointNode, "OnRelease");
    }


    private static void InvokeCrystalSphereDivinationAction(
        NCrystalSphereScreen? crystalSphereScreen,
        NDivinationButton button,
        bool useBigDivination)
    {
        if (crystalSphereScreen is not null)
        {
            InvokeSingleArgumentAction(
                crystalSphereScreen,
                useBigDivination ? "SetBigDivination" : "SetSmallDivination",
                button);
            return;
        }

        InvokeButtonAction(button, "OnRelease", "OnPress");
    }

    private static void InvokeCrystalSphereCellAction(
        NCrystalSphereScreen? crystalSphereScreen,
        NCrystalSphereCell cell)
    {
        if (crystalSphereScreen is not null)
        {
            InvokeSingleArgumentAction(crystalSphereScreen, "OnCellClicked", cell);
            return;
        }

        InvokeButtonAction(cell, "EntityClicked");
    }

    private static void InvokeCrystalSphereProceedAction(
        NCrystalSphereScreen? crystalSphereScreen,
        NProceedButton button)
    {
        if (crystalSphereScreen is not null &&
            TryInvokeSingleArgument(crystalSphereScreen, "OnProceedButtonPressed", button))
        {
            return;
        }

        InvokeProceedButtonAction(button);
    }

    private static void InvokeRoomProceedAction(BridgeWorldContext context)
    {
        if (context.ProceedButton is null)
        {
            throw new BridgeRequestException(
                HttpStatusCode.Conflict,
                "action_target_missing",
                "Could not find a visible room proceed button.");
        }

        if (context.CombatRoom is not null &&
            ReferenceEquals(context.ProceedButton, context.CombatRoom.ProceedButton))
        {
            InvokeCombatProceedAction(context.CombatRoom, context.ProceedButton);
            return;
        }

        if (context.TreasureRoom is not null &&
            IsNodeVisible(context.TreasureRoom))
        {
            InvokeTreasureProceedAction(context.TreasureRoom, context.ProceedButton);
            return;
        }

        InvokeProceedButtonAction(context.ProceedButton);
    }

    private static void InvokeCombatProceedAction(NCombatRoom? combatRoom, NProceedButton button)
    {
        if (TryInvokeSingleArgument(combatRoom, "OnProceedButtonPressed", button))
        {
            return;
        }

        InvokeProceedButtonAction(button);
    }

    private static void InvokeRestSiteProceedAction(NRestSiteRoom? restSiteRoom, NProceedButton button)
    {
        if (TryInvokeSingleArgument(restSiteRoom, "OnProceedButtonReleased", button))
        {
            return;
        }

        InvokeProceedButtonAction(button);
    }

    private static void InvokeMerchantLeaveAction(NMerchantRoom? merchantRoom, NProceedButton button)
    {
        if (TryInvokeSingleArgument(merchantRoom, "OnProceedButtonReleased", button) ||
            TryInvokeSingleArgument(merchantRoom, "OnProceedButtonPressed", button) ||
            TryInvokeParameterless(button, "ForceClick") ||
            TryInvokeSingleArgument(merchantRoom, "HideScreen", button))
        {
            return;
        }

        InvokeProceedButtonAction(button);
    }

    private static void InvokeMerchantBackAction(NMerchantInventory? merchantInventory, NBackButton button)
    {
        if (TryInvokeParameterless(merchantInventory, "Close"))
        {
            return;
        }

        InvokeButtonAction(button, "OnPress");
    }

    private static void InvokeTreasureChestAction(NTreasureRoom? treasureRoom, NTreasureButton chestButton)
    {
        if (TryInvokeSingleArgument(treasureRoom, "OnChestButtonReleased", chestButton))
        {
            return;
        }

        var openChestResult = InvokeParameterless(treasureRoom, "OpenChest");
        if (openChestResult is Task openChestTask)
        {
            openChestTask.GetAwaiter().GetResult();
            return;
        }

        InvokeButtonAction(chestButton, "OnRelease");
    }

    private static void InvokeTreasureRelicAction(
        NTreasureRoomRelicCollection? treasureRelicCollection,
        NTreasureRoomRelicHolder relicHolder)
    {
        if (TryInvokeSingleArgument(treasureRelicCollection, "PickRelic", relicHolder))
        {
            return;
        }

        if (TryInvokeParameterless(relicHolder, "OnRelease") ||
            TryInvokeParameterless(relicHolder, "OnPress"))
        {
            return;
        }

        InvokeClickablePressAndRelease(relicHolder);
    }

    private static void InvokeTreasureProceedAction(NTreasureRoom? treasureRoom, NProceedButton button)
    {
        if (TryInvokeSingleArgument(treasureRoom, "OnProceedButtonReleased", button) ||
            TryInvokeSingleArgument(treasureRoom, "OnProceedButtonPressed", button))
        {
            return;
        }

        InvokeProceedButtonAction(button);
    }

    private static void InvokeCardSelectionOptionAction(Node? cardSelectionScreen, NCardHolder cardHolder)
    {
        EnsureCardSelectionOptionStillPresentOrThrow(cardSelectionScreen, cardHolder);

        var beforeState = CaptureCardSelectionUiState(cardSelectionScreen, cardHolder.CardModel);
        var invokedAnyCandidate = false;

        // NChooseACardSelectionScreen intentionally ignores the first ~350 ms of
        // holder presses after the overlay opens. The bridge only exposes the
        // select action once the surface is already visible and actionable, so
        // it is safe to backdate the opened tick and avoid spurious no-op
        // selections on the first policy step after the overlay appears.
        PrepareCardSelectionScreenForBridgeSelect(cardSelectionScreen);

        if (cardSelectionScreen is NPlayerHand playerHand)
        {
            if (TryInvokeSingleArgument(playerHand, "OnHolderPressed", cardHolder))
            {
                invokedAnyCandidate = true;
                if (HasCardSelectionStateProgress(
                    beforeState,
                    CaptureCardSelectionUiState(cardSelectionScreen, cardHolder.CardModel)))
                {
                    TryAutoConfirmSelectedCardSelection(cardSelectionScreen);
                    return;
                }
            }

            if (cardHolder is NHandCardHolder handCardHolder &&
                (TryInvokeSingleArgument(playerHand, "SelectCardInSimpleMode", handCardHolder) ||
                 TryInvokeSingleArgument(playerHand, "SelectCardInUpgradeMode", handCardHolder)))
            {
                invokedAnyCandidate = true;
                if (HasCardSelectionStateProgress(
                    beforeState,
                    CaptureCardSelectionUiState(cardSelectionScreen, cardHolder.CardModel)))
                {
                    TryAutoConfirmSelectedCardSelection(cardSelectionScreen);
                    return;
                }
            }
        }

        if (TryInvokeSingleArgument(cardSelectionScreen, "SelectHolder", cardHolder))
        {
            invokedAnyCandidate = true;
            if (HasCardSelectionStateProgress(
                beforeState,
                CaptureCardSelectionUiState(cardSelectionScreen, cardHolder.CardModel)))
            {
                TryAutoConfirmSelectedCardSelection(cardSelectionScreen);
                return;
            }
        }

        if (cardHolder.CardModel is not null &&
            TryInvokeSingleArgument(cardSelectionScreen, "OnCardClicked", cardHolder.CardModel))
        {
            invokedAnyCandidate = true;
            if (HasCardSelectionStateProgress(
                beforeState,
                CaptureCardSelectionUiState(cardSelectionScreen, cardHolder.CardModel)))
            {
                TryAutoConfirmSelectedCardSelection(cardSelectionScreen);
                return;
            }
        }

        if (invokedAnyCandidate)
        {
            return;
        }

        throw new BridgeRequestException(
            HttpStatusCode.Conflict,
            "action_target_missing",
            $"Could not resolve a supported card-selection action for {cardSelectionScreen?.GetType().FullName ?? "<missing screen>"}.");
    }

    private static void EnsureCardSelectionOptionStillPresentOrThrow(Node? cardSelectionScreen, NCardHolder cardHolder)
    {
        if (cardSelectionScreen is null || !IsNodeVisible(cardSelectionScreen))
        {
            throw new BridgeRequestException(
                HttpStatusCode.Conflict,
                "card_selection_screen_gone",
                "Card selection screen is no longer visible at execution time.");
        }

        var currentOptions = GetCardSelectionOptions(cardSelectionScreen);
        var holderStillPresent = currentOptions.Any(holder => ReferenceEquals(holder, cardHolder));
        if (!holderStillPresent && cardHolder.CardModel is not null)
        {
            holderStillPresent = currentOptions.Any(holder => ReferenceEquals(holder.CardModel, cardHolder.CardModel));
        }

        if (holderStillPresent)
        {
            return;
        }

        var selectionId = cardHolder.CardModel is not null ? GetCardReference(cardHolder.CardModel) : null;
        throw new BridgeRequestException(
            HttpStatusCode.Conflict,
            "card_selection_option_gone",
            $"Card selection option '{selectionId ?? "<unknown>"}' is no longer present at execution time.",
            new
            {
                selection_id = selectionId,
                screen_type = cardSelectionScreen.GetType().Name,
                option_count = currentOptions.Count
            });
    }

    private const ulong NChooseACardSelectionOpenGuardMs = 350UL;

    private static void PrepareCardSelectionScreenForBridgeSelect(Node? cardSelectionScreen)
    {
        if (cardSelectionScreen is not NChooseACardSelectionScreen chooseACardSelectionScreen)
        {
            return;
        }

        if (!TryGetNChooseACardSelectionOpenedAgeMs(chooseACardSelectionScreen, out var openedAgeMs) ||
            openedAgeMs > NChooseACardSelectionOpenGuardMs)
        {
            return;
        }

        var now = Time.GetTicksMsec();
        var backdatedOpenedTicks = now > NChooseACardSelectionOpenGuardMs
            ? now - (NChooseACardSelectionOpenGuardMs + 1UL)
            : 0UL;
        TrySetHiddenFieldValue(chooseACardSelectionScreen, "_openedTicks", backdatedOpenedTicks);
    }

    private static bool TryGetNChooseACardSelectionOpenedAgeMs(
        NChooseACardSelectionScreen chooseACardSelectionScreen,
        out ulong openedAgeMs)
    {
        if (!TryConvertToULong(GetHiddenFieldValue(chooseACardSelectionScreen, "_openedTicks"), out var openedTicks))
        {
            openedAgeMs = 0UL;
            return false;
        }

        var now = Time.GetTicksMsec();
        openedAgeMs = now >= openedTicks ? now - openedTicks : 0UL;
        return true;
    }

    private static void InvokeCardSelectionBundleAction(Node? cardSelectionScreen, NCardBundle bundle)
    {
        if (TryInvokeSingleArgument(cardSelectionScreen, "OnBundleClicked", bundle))
        {
            return;
        }

        if (bundle.Hitbox is not null)
        {
            InvokeClickablePressAndRelease(bundle.Hitbox);
            return;
        }

        throw new BridgeRequestException(
            HttpStatusCode.Conflict,
            "action_target_missing",
            $"Could not resolve a supported bundle-selection action for {cardSelectionScreen?.GetType().FullName ?? "<missing screen>"}.");
    }

    private static bool ShouldAutoConfirmSingleCardSelection(Node? cardSelectionScreen)
    {
        var prefs = GetHiddenFieldValue(cardSelectionScreen, "_prefs");
        return (GetHiddenPropertyValue<int>(prefs, "MinSelect") ?? 0) == 1 &&
               (GetHiddenPropertyValue<int>(prefs, "MaxSelect") ?? 0) == 1;
    }

    private static void TryAutoConfirmSelectedCardSelection(Node? cardSelectionScreen)
    {
        if (!ShouldAutoConfirmSingleCardSelection(cardSelectionScreen) ||
            CountSelectedCardSelectionCards(cardSelectionScreen) <= 0)
        {
            return;
        }

        InvokeCardSelectionConfirmAction(
            cardSelectionScreen,
            ResolveCardSelectionConfirmButton(cardSelectionScreen));
    }

    private static void InvokeCardSelectionConfirmAction(Node? cardSelectionScreen, Node? confirmButton)
    {
        if (cardSelectionScreen is NPlayerHand playerHand &&
            confirmButton is not null &&
            TryInvokeSingleArgument(playerHand, "OnSelectModeConfirmButtonPressed", confirmButton))
        {
            return;
        }

        if (TryInvokeCardSelectionCompleteSelection(cardSelectionScreen))
        {
            return;
        }

        if (confirmButton is not null &&
            TryInvokeSingleArgument(cardSelectionScreen, "ConfirmSelection", confirmButton))
        {
            return;
        }

        if (confirmButton is not null)
        {
            InvokeClickablePressAndRelease(confirmButton);
            return;
        }

        throw new BridgeRequestException(
            HttpStatusCode.Conflict,
            "action_target_missing",
            "Could not confirm the current card selection.");
    }

    private static void InvokeCardSelectionCancelAction(Node? cardSelectionScreen, Node? cancelButton)
    {
        if (cancelButton is not null &&
            TryInvokeSingleArgument(cardSelectionScreen, "CancelSelection", cancelButton))
        {
            return;
        }

        if (cancelButton is not null)
        {
            InvokeClickablePressAndRelease(cancelButton);
            return;
        }

        throw new BridgeRequestException(
            HttpStatusCode.Conflict,
            "action_target_missing",
            "Could not cancel the current card-selection preview.");
    }

    private static void InvokeCardSelectionCloseAction(Node? cardSelectionScreen, Node? closeButton)
    {
        if (closeButton is not null &&
            TryInvokeSingleArgument(cardSelectionScreen, "CloseSelection", closeButton))
        {
            return;
        }

        if (closeButton is not null)
        {
            InvokeClickablePressAndRelease(closeButton);
            return;
        }

        throw new BridgeRequestException(
            HttpStatusCode.Conflict,
            "action_target_missing",
            "Could not close the current card-selection screen.");
    }

    private static void InvokeCardSelectionSkipAction(Node? cardSelectionScreen, Node? skipButton)
    {
        if (skipButton is not null &&
            TryInvokeSingleArgument(cardSelectionScreen, "OnSkipButtonReleased", skipButton))
        {
            return;
        }

        if (skipButton is not null)
        {
            InvokeClickablePressAndRelease(skipButton);
            return;
        }

        throw new BridgeRequestException(
            HttpStatusCode.Conflict,
            "action_target_missing",
            "Could not skip the current card selection.");
    }

    private static Node? ResolveCardRewardSkipButton(NCardRewardSelectionScreen? cardRewardScreen)
    {
        var alternativesContainer = cardRewardScreen?.GetNodeOrNull<Control>("UI/RewardAlternatives") ??
                                    GetHiddenFieldValue(cardRewardScreen, "_rewardAlternativesContainer") as Control;
        if (alternativesContainer is null || !GodotObject.IsInstanceValid(alternativesContainer))
        {
            return null;
        }

        return alternativesContainer
            .GetChildren()
            .OfType<Node>()
            .Where(IsNodeVisible)
            .FirstOrDefault(IsCardRewardSkipAlternativeButton);
    }

    private static void InvokeCardRewardSkipAction(NCardRewardSelectionScreen? cardRewardScreen, Node? skipButton)
    {
        if (IsCardRewardSelectionReady(cardRewardScreen) &&
            TryInvokeSingleArgument(
                cardRewardScreen,
                "OnAlternateRewardSelected",
                GetCardRewardSkipActionCompatibility()))
        {
            return;
        }

        if (skipButton is not null)
        {
            InvokeClickablePressAndRelease(skipButton);
            return;
        }

        throw new BridgeRequestException(
            HttpStatusCode.Conflict,
            "action_target_missing",
            "Could not skip the current card reward.");
    }

    private static bool IsCardRewardSkipAlternativeButton(Node button)
    {
        static bool IsSkipText(string? text)
        {
            var comparableText = NormalizeComparableText(text).ToLowerInvariant();
            return comparableText == "skip" || comparableText == "跳过";
        }

        return IsSkipText(GetHiddenFieldValue(button, "_optionName") as string) ||
               IsSkipText(TryGetLocalNodeText(button));
    }

    private static bool IsRewardButtonSkipped(NRewardsScreen? rewardsScreen, NRewardButton button)
    {
        return IsRewardControlSkipped(rewardsScreen, button);
    }

    private static bool IsRewardControlSkipped(NRewardsScreen? rewardsScreen, Control rewardControl)
    {
        if (GetHiddenFieldValue(rewardsScreen, "_skippedRewardButtons") is not IEnumerable skippedRewardButtons)
        {
            return false;
        }

        foreach (var skippedRewardButton in skippedRewardButtons)
        {
            if (skippedRewardButton is Node skippedNode &&
                IsSameNodeInstance(skippedNode, rewardControl))
            {
                return true;
            }
        }

        return false;
    }

    private static List<(Control RewardControl, PotionReward PotionReward)> ResolveSkippablePotionRewardControls(
        NRewardsScreen? rewardsScreen)
    {
        var results = new List<(Control RewardControl, PotionReward PotionReward)>();
        if (rewardsScreen is null ||
            GetHiddenFieldValue(rewardsScreen, "_rewardButtons") is not IEnumerable rewardButtons)
        {
            return results;
        }

        foreach (var rewardButton in rewardButtons)
        {
            if (rewardButton is not Control rewardControl ||
                !GodotObject.IsInstanceValid(rewardControl) ||
                IsRewardControlSkipped(rewardsScreen, rewardControl))
            {
                continue;
            }

            if (ResolveRewardFromControl(rewardControl) is not PotionReward potionReward)
            {
                continue;
            }

            results.Add((rewardControl, potionReward));
        }

        return results;
    }

    private static Reward? ResolveRewardFromControl(Control rewardControl)
    {
        return ResolveRewardFromControlForLivePayload(rewardControl) ??
               GetHiddenPropertyObjectValue(rewardControl, "Reward") as Reward;
    }

    private static Reward? ResolveRewardFromControlForLivePayload(Control rewardControl)
    {
        return GetHiddenFieldValue(rewardControl, "<Reward>k__BackingField") as Reward ??
               GetHiddenFieldValue(rewardControl, "_reward") as Reward;
    }

    private static void InvokeRewardSkipAction(NRewardsScreen? rewardsScreen, Control rewardControl)
    {
        if (TryInvokeSingleArgument(rewardsScreen, "RewardSkippedFrom", rewardControl))
        {
            return;
        }

        if (rewardControl is NRewardButton rewardButton &&
            TryInvokeSingleArgument(rewardButton, "EmitSignalRewardSkipped", rewardButton))
        {
            return;
        }

        throw new BridgeRequestException(
            HttpStatusCode.Conflict,
            "action_target_missing",
            "Could not skip the current reward.");
    }

    private static void InvokeTerminalRewardsProceed(
        RunManager? runManager,
        NRewardsScreen? rewardsScreen,
        NProceedButton? rewardProceedButton)
    {
        // Prefer the run-manager path first. In practice this is the most
        // reliable way to leave terminal reward states back into the normal
        // run flow after room-end rewards finish resolving.
        if (TryInvokeParameterless(runManager, "ProceedFromTerminalRewardsScreen") ||
            TryInvokeParameterless(rewardsScreen, "ProceedFromTerminalRewardsScreen"))
        {
            FinalizeTerminalRewardsOverlayClose(rewardsScreen);
            return;
        }

        if (rewardProceedButton is not null &&
            TryInvokeSingleArgument(rewardsScreen, "OnProceedButtonPressed", rewardProceedButton))
        {
            FinalizeTerminalRewardsOverlayClose(rewardsScreen);
            return;
        }

        if (rewardProceedButton is not null && IsNodeVisible(rewardProceedButton))
        {
            InvokeProceedButtonAction(rewardProceedButton);
            FinalizeTerminalRewardsOverlayClose(rewardsScreen);
            return;
        }

        throw new BridgeRequestException(
            HttpStatusCode.Conflict,
            "action_target_missing",
            "Could not find a terminal rewards proceed target.");
    }

    private static void FinalizeTerminalRewardsOverlayClose(NRewardsScreen? rewardsScreen)
    {
        if (rewardsScreen is null || !GodotObject.IsInstanceValid(rewardsScreen))
        {
            return;
        }

        try
        {
            if (NOverlayStack.Instance is not null)
            {
                NOverlayStack.Instance.Remove(rewardsScreen);
                return;
            }
        }
        catch
        {
            // Fall back to the legacy direct-close path below if the overlay
            // stack is unavailable or rejects the remove call.
        }

        TryInvokeParameterless(rewardsScreen, "AfterOverlayClosed");
    }

    private static void InvokeRunModeSelectionAction(Node? submenu, Node? button, string methodName)
    {
        if (submenu is not null)
        {
            if (TryInvokeParameterless(submenu, methodName))
            {
                return;
            }

            if (button is not null && TryInvokeSingleArgument(submenu, methodName, button))
            {
                return;
            }
        }

        if (button is not null)
        {
            InvokeMenuButtonAction(button);
            return;
        }

        throw new BridgeRequestException(
            HttpStatusCode.Conflict,
            "action_target_missing",
            $"Could not invoke run-mode selection action '{methodName}'.");
    }

    private static void InvokeButtonAction(object target, string methodName, string? fallbackMethodName = null)
    {
        if (TryInvokeParameterless(target, methodName))
        {
            return;
        }

        if (fallbackMethodName is not null && TryInvokeParameterless(target, fallbackMethodName))
        {
            return;
        }

        throw new BridgeRequestException(
            HttpStatusCode.Conflict,
            "action_target_missing",
            $"Could not invoke {methodName} on {target.GetType().FullName}.");
    }

    private static void InvokeGameOverContinueAction(
        NGameOverScreen? gameOverScreen,
        NGameOverContinueButton? continueButton)
    {
        if (gameOverScreen is not null)
        {
            if (TryInvokeParameterless(gameOverScreen, "OpenTimeline") ||
                TryInvokeParameterless(gameOverScreen, "TransitionOutToTimeline"))
            {
                return;
            }
        }

        if (continueButton is not null)
        {
            InvokeClickablePressAndRelease(continueButton);
            return;
        }

        throw new BridgeRequestException(
            HttpStatusCode.Conflict,
            "action_target_missing",
            "Could not invoke a supported game-over continue action.");
    }

    private static void InvokeGameOverReturnToMainMenuAction(
        NGameOverScreen? gameOverScreen,
        NReturnToMainMenuButton? mainMenuButton)
    {
        if (gameOverScreen is not null)
        {
            if (TryInvokeParameterless(gameOverScreen, "ReturnToMainMenu") ||
                TryInvokeParameterless(gameOverScreen, "TransitionOutToMainMenu") ||
                (mainMenuButton is not null &&
                 TryInvokeSingleArgument(gameOverScreen, "OnMainMenuButtonPressed", mainMenuButton)))
            {
                return;
            }
        }

        if (mainMenuButton is not null)
        {
            InvokeClickablePressAndRelease(mainMenuButton);
            return;
        }

        throw new BridgeRequestException(
            HttpStatusCode.Conflict,
            "action_target_missing",
            "Could not invoke a supported game-over return-to-main-menu action.");
    }

    private static void InvokeSingleArgumentAction(object target, string methodName, object argument)
    {
        var method = FindMethod(target.GetType(), methodName, 1);
        if (method is null)
        {
            throw new BridgeRequestException(
                HttpStatusCode.Conflict,
                "action_target_missing",
                $"Could not find {methodName} on {target.GetType().FullName}.");
        }

        method.Invoke(target, new[] { argument });
    }

}
