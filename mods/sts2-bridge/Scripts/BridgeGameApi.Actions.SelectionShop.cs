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
    private static void AddCardSelectionActions(List<BridgeResolvedAction> actions, BridgeWorldContext context)
    {
        if (!IsCardSelectionVisible(context))
        {
            return;
        }

        var selectionPrompt = TryGetCardSelectionPrompt(context.CardSelectionScreen);
        var selectionTexts = CollectCardSelectionSurfaceTexts(
            context.CardSelectionScreen,
            selectionPrompt,
            context.CardSelectionConfirmButton,
            context.CardSelectionCancelButton,
            context.CardSelectionCloseButton,
            context.CardSelectionSkipButton);
        var selectionSemantics = ResolveCardSelectionSemantics(context.CardSelectionScreen, selectionPrompt, selectionTexts);
        var selectionState = CaptureCardSelectionUiState(context.CardSelectionScreen);
        var selectedCount = selectionState.SelectedCount;
        var minSelect = selectionState.MinSelect;
        var maxSelect = selectionState.MaxSelect;
        var remainingSelect = ResolveRemainingSelectCount(selectedCount, minSelect, maxSelect);
        var prefs = GetHiddenFieldValue(context.CardSelectionScreen, "_prefs");
        var requiresManualConfirmation = GetHiddenPropertyValue<bool>(prefs, "RequireManualConfirmation");
        var cancelable = GetHiddenPropertyValue<bool>(prefs, "Cancelable");
        var confirmReady = context.CardSelectionConfirmButton is not null &&
                           IsNodeVisible(context.CardSelectionConfirmButton) &&
                           IsButtonEnabled(context.CardSelectionConfirmButton);
        var canSkip = context.CardSelectionSkipButton is not null &&
                      IsNodeVisible(context.CardSelectionSkipButton) &&
                      IsButtonEnabled(context.CardSelectionSkipButton);
        var selectionReady = selectionState.SelectionReady;
        var openedAgeMs = selectionState.OpenedAgeMs;
        var typedSelection = BuildRuntimeSelectionPayload(
            context.CardSelectionScreen?.GetType().Name,
            selectionSemantics,
            selectedCount,
            minSelect,
            maxSelect,
            requiresManualConfirmation,
            source: "card_selection",
            sourceZone: string.Empty,
            destinationZone: string.Empty);

        if (context.CardSelectionScreen is NChooseABundleSelectionScreen)
        {
            var bundleOptions = GetCardSelectionBundles(context.CardSelectionScreen);
            for (var index = 0; index < bundleOptions.Count; index++)
            {
                var bundle = bundleOptions[index];
                if (!IsNodeVisible(bundle))
                {
                    continue;
                }

                var actionId = $"card_selection:select:{index}";
                actions.Add(new BridgeResolvedAction
                {
                    ActionId = actionId,
                    Payload = new
                    {
                        action_id = actionId,
                        kind = "card_selection",
                        selection_action = "select",
                        selection_semantics = selectionSemantics,
                        selection_prompt = selectionPrompt,
                        typed_selection = typedSelection,
                        selected_count = selectedCount,
                        min_select = minSelect,
                        max_select = maxSelect,
                        remaining_select = remainingSelect,
                        confirm_ready = confirmReady,
                        can_skip = canSkip,
                        requires_manual_confirmation = requiresManualConfirmation,
                        cancelable = cancelable,
                        selection_ready = selectionReady,
                        opened_age_ms = openedAgeMs,
                        index,
                        label = $"Select bundle {index}",
                        bundle = bundle.Bundle.Select(card => BuildCardPayload(card)).ToArray(),
                        screen = context.Screen,
                        screen_type = context.CardSelectionScreen.GetType().Name
                    },
                    Execute = () => InvokeCardSelectionBundleAction(context.CardSelectionScreen, bundle)
                });
            }
        }
        else
        {
            for (var index = 0; index < context.CardSelectionOptions.Count; index++)
            {
                var cardHolder = context.CardSelectionOptions[index];
                if (!IsNodeVisible(cardHolder))
                {
                    continue;
                }

                var optionIndex = GetCardSelectionOptionIndex(context.CardSelectionScreen, cardHolder, index);
                var selectionId = GetCardSelectionOptionSelectionId(context.CardSelectionScreen, cardHolder, optionIndex);
                var actionId = selectionId is not null
                    ? $"card_selection:select:{selectionId}"
                    : $"card_selection:select:{optionIndex}";
                var isSelected = IsCardSelectionCardSelected(context.CardSelectionScreen, cardHolder.CardModel);
                actions.Add(new BridgeResolvedAction
                {
                    ActionId = actionId,
                    Payload = new
                    {
                        action_id = actionId,
                        kind = "card_selection",
                        selection_action = "select",
                        selection_semantics = selectionSemantics,
                        selection_prompt = selectionPrompt,
                        typed_selection = typedSelection,
                        selected_count = selectedCount,
                        min_select = minSelect,
                        max_select = maxSelect,
                        remaining_select = remainingSelect,
                        confirm_ready = confirmReady,
                        can_skip = canSkip,
                        requires_manual_confirmation = requiresManualConfirmation,
                        cancelable = cancelable,
                        selection_ready = selectionReady,
                        opened_age_ms = openedAgeMs,
                        index = optionIndex,
                        selection_id = selectionId,
                        // 2026-04-27: surface is_selected on the per-card select
                        // action payload (already on the options summary, but
                        // the model needs it on the action token to detect
                        // pick→deselect→pick loops in multi-pick burn cards).
                        is_selected = isSelected,
                        label = $"Select card {optionIndex}: {cardHolder.CardModel?.Title ?? "<missing>"}",
                        card = BuildCardPayload(cardHolder.CardModel),
                        screen = context.Screen,
                        screen_type = context.CardSelectionScreen?.GetType().Name
                    },
                    Execute = () => InvokeCardSelectionOptionAction(context.CardSelectionScreen, cardHolder)
                });
            }
        }

        if (context.CardSelectionConfirmButton is not null &&
            IsNodeVisible(context.CardSelectionConfirmButton) &&
            IsButtonEnabled(context.CardSelectionConfirmButton))
        {
            actions.Add(new BridgeResolvedAction
            {
                ActionId = "card_selection:confirm",
                Payload = new
                {
                    action_id = "card_selection:confirm",
                    kind = "card_selection",
                    selection_action = "confirm",
                    selection_semantics = selectionSemantics,
                    selection_prompt = selectionPrompt,
                    typed_selection = typedSelection,
                    selected_count = selectedCount,
                    min_select = minSelect,
                    max_select = maxSelect,
                    remaining_select = remainingSelect,
                    confirm_ready = confirmReady,
                    can_skip = canSkip,
                    requires_manual_confirmation = requiresManualConfirmation,
                    cancelable = cancelable,
                    selection_ready = selectionReady,
                    opened_age_ms = openedAgeMs,
                    label = "Confirm selected cards",
                    screen = context.Screen,
                    screen_type = context.CardSelectionScreen?.GetType().Name
                },
                Execute = () => InvokeCardSelectionConfirmAction(
                    context.CardSelectionScreen,
                    context.CardSelectionConfirmButton)
            });
        }

        if (context.CardSelectionCancelButton is not null &&
            IsNodeVisible(context.CardSelectionCancelButton) &&
            IsButtonEnabled(context.CardSelectionCancelButton))
        {
            actions.Add(new BridgeResolvedAction
            {
                ActionId = "card_selection:cancel",
                Payload = new
                {
                    action_id = "card_selection:cancel",
                    kind = "card_selection",
                    selection_action = "cancel",
                    selection_semantics = selectionSemantics,
                    selection_prompt = selectionPrompt,
                    typed_selection = typedSelection,
                    selected_count = selectedCount,
                    min_select = minSelect,
                    max_select = maxSelect,
                    remaining_select = remainingSelect,
                    confirm_ready = confirmReady,
                    can_skip = canSkip,
                    requires_manual_confirmation = requiresManualConfirmation,
                    cancelable = cancelable,
                    selection_ready = selectionReady,
                    opened_age_ms = openedAgeMs,
                    label = "Cancel card selection preview",
                    screen = context.Screen,
                    screen_type = context.CardSelectionScreen?.GetType().Name
                },
                Execute = () => InvokeCardSelectionCancelAction(
                    context.CardSelectionScreen,
                    context.CardSelectionCancelButton)
            });
        }

        if (context.CardSelectionCloseButton is not null &&
            IsNodeVisible(context.CardSelectionCloseButton) &&
            IsButtonEnabled(context.CardSelectionCloseButton))
        {
            actions.Add(new BridgeResolvedAction
            {
                ActionId = "card_selection:close",
                Payload = new
                {
                    action_id = "card_selection:close",
                    kind = "card_selection",
                    selection_action = "close",
                    selection_semantics = selectionSemantics,
                    selection_prompt = selectionPrompt,
                    typed_selection = typedSelection,
                    selected_count = selectedCount,
                    min_select = minSelect,
                    max_select = maxSelect,
                    remaining_select = remainingSelect,
                    confirm_ready = confirmReady,
                    can_skip = canSkip,
                    requires_manual_confirmation = requiresManualConfirmation,
                    cancelable = cancelable,
                    selection_ready = selectionReady,
                    opened_age_ms = openedAgeMs,
                    label = "Close card selection",
                    screen = context.Screen,
                    screen_type = context.CardSelectionScreen?.GetType().Name
                },
                Execute = () => InvokeCardSelectionCloseAction(
                    context.CardSelectionScreen,
                    context.CardSelectionCloseButton)
            });
        }

        if (context.CardSelectionSkipButton is not null &&
            IsNodeVisible(context.CardSelectionSkipButton) &&
            IsButtonEnabled(context.CardSelectionSkipButton))
        {
            actions.Add(new BridgeResolvedAction
            {
                ActionId = "card_selection:skip",
                Payload = new
                {
                    action_id = "card_selection:skip",
                    kind = "card_selection",
                    selection_action = "skip",
                    selection_semantics = selectionSemantics,
                    selection_prompt = selectionPrompt,
                    typed_selection = typedSelection,
                    selected_count = selectedCount,
                    min_select = minSelect,
                    max_select = maxSelect,
                    remaining_select = remainingSelect,
                    confirm_ready = confirmReady,
                    can_skip = canSkip,
                    requires_manual_confirmation = requiresManualConfirmation,
                    cancelable = cancelable,
                    selection_ready = selectionReady,
                    opened_age_ms = openedAgeMs,
                    label = "Skip card selection",
                    screen = context.Screen,
                    screen_type = context.CardSelectionScreen?.GetType().Name
                },
                Execute = () => InvokeCardSelectionSkipAction(
                    context.CardSelectionScreen,
                    context.CardSelectionSkipButton)
            });
        }
    }

    private static void AddShopActions(List<BridgeResolvedAction> actions, BridgeWorldContext context)
    {
        if (context.MerchantRoom is null ||
            (!IsNodeVisible(context.MerchantRoom) && context.MerchantInventory?.IsOpen != true))
        {
            return;
        }

        var inventoryIsOpen = context.MerchantInventory?.IsOpen == true;
        var shopOpenAvailable = CanExposeShopOpenAction(context);

        if (!inventoryIsOpen &&
            shopOpenAvailable &&
            context.MerchantButton is not null &&
            IsNodeVisible(context.MerchantButton) &&
            IsButtonEnabled(context.MerchantButton))
        {
            actions.Add(new BridgeResolvedAction
            {
                ActionId = "shop:open",
                Payload = new
                {
                    action_id = "shop:open",
                    kind = "shop",
                    shop_action = "open",
                    label = "Open merchant inventory",
                    screen = context.Screen
                },
                Execute = () =>
                {
                    InvokeButtonAction(context.MerchantButton, "OnRelease", "OnPress");
                    RecordShopOpenAction(context);
                }
            });
        }

        if (inventoryIsOpen && context.MerchantInventory is not null)
        {
            for (var index = 0; index < context.MerchantSlots.Count; index++)
            {
                var slot = context.MerchantSlots[index];
                var entry = slot.Entry;
                if (!IsNodeVisible(slot) || !CanPurchaseMerchantEntry(entry))
                {
                    continue;
                }

                var actionId = $"shop:buy:{index}";
                actions.Add(new BridgeResolvedAction
                {
                    ActionId = actionId,
                    Payload = new
                    {
                        action_id = actionId,
                        kind = "shop",
                        shop_action = "buy",
                        index,
                        label = $"Buy shop item {index}: {DescribeMerchantEntry(entry)}",
                        item = BuildShopSlotPayload(slot, index),
                        screen = context.Screen
                    },
                    Execute = () => ExecuteShopPurchase(slot, context.MerchantInventory)
                });
            }
        }

        if (context.MerchantBackButton is not null &&
            IsNodeVisible(context.MerchantBackButton) &&
            IsButtonEnabled(context.MerchantBackButton))
        {
            actions.Add(new BridgeResolvedAction
            {
                ActionId = "shop:back",
                Payload = new
                {
                    action_id = "shop:back",
                    kind = "shop",
                    shop_action = "back",
                    label = "Close merchant inventory",
                    screen = context.Screen
                },
                Execute = () => InvokeMerchantBackAction(context.MerchantInventory, context.MerchantBackButton)
            });
        }

        if (context.MerchantProceedButton is not null &&
            IsNodeVisible(context.MerchantProceedButton) &&
            IsButtonEnabled(context.MerchantProceedButton))
        {
            actions.Add(new BridgeResolvedAction
            {
                ActionId = "shop:leave",
                Payload = new
                {
                    action_id = "shop:leave",
                    kind = "shop",
                    shop_action = "leave",
                    label = "Leave shop",
                    screen = context.Screen
                },
                Execute = () => InvokeMerchantLeaveAction(context.MerchantRoom, context.MerchantProceedButton)
            });
        }
    }


}
