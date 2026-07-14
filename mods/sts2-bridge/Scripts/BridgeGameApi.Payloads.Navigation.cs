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
    private static object BuildMapPayload(
        RunState? runState,
        NMapScreen? mapScreen,
        IReadOnlyList<NMapPoint> mapPoints,
        CombatManager? combatManager,
        string currentScreen)
    {
        var map = runState?.Map;
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
            is_blocked_by_combat = !interactiveSurface &&
                                   rawIsOpen &&
                                   rawIsTravelEnabled &&
                                   !rawIsTraveling &&
                                   combatManager?.IsInProgress == true,
            current_coord = BuildMapCoord(runState?.CurrentMapCoord),
            dimensions = map is null
                ? null
                : new
                {
                    rows = map.GetRowCount(),
                    columns = map.GetColumnCount()
                },
            points = mapPoints.Select(BuildMapPointPayload).ToArray()
        };
    }

    private static object BuildRestSitePayload(
        NMapScreen? mapScreen,
        NRestSiteRoom? restSiteRoom,
        IReadOnlyList<NRestSiteButton> restSiteButtons,
        NProceedButton? restSiteProceedButton)
    {
        var visible = !IsInteractiveMapSurface(mapScreen) &&
                      restSiteRoom is not null &&
                      IsNodeVisible(restSiteRoom);
        var options = visible
            ? restSiteButtons
                .Where(IsNodeVisible)
                .Select((button, index) => BuildRestSiteOptionPayload(button.Option, index))
                .ToArray()
            : new object[0];

        return new
        {
            visible,
            header = visible
                ? TryGetLocalNodeText(GetHiddenFieldValue(restSiteRoom, "<Header>k__BackingField") as Node)
                : null,
            description = visible
                ? TryGetLocalNodeText(GetHiddenFieldValue(restSiteRoom, "<Description>k__BackingField") as Node)
                : null,
            proceed_visible = visible &&
                              !HasVisibleEnabledRestSiteOptions(restSiteButtons) &&
                              restSiteProceedButton is not null &&
                              IsNodeVisible(restSiteProceedButton) &&
                              IsButtonEnabled(restSiteProceedButton),
            options
        };
    }

    private static object BuildShopPayload(
        NMerchantRoom? merchantRoom,
        NMerchantInventory? merchantInventory,
        IReadOnlyList<NMerchantSlot> merchantSlots,
        NMerchantButton? merchantButton,
        NProceedButton? merchantProceedButton,
        NBackButton? merchantBackButton)
    {
        var visible = (merchantRoom is not null && IsNodeVisible(merchantRoom)) ||
                      (merchantInventory is not null && IsNodeVisible(merchantInventory));
        var inventory = merchantInventory?.Inventory;

        return new
        {
            visible,
            is_open = merchantInventory?.IsOpen ?? false,
            gold = inventory?.Player?.Gold,
            merchant_button_visible = merchantButton is not null && IsNodeVisible(merchantButton),
            back_button_visible = merchantBackButton is not null && IsNodeVisible(merchantBackButton),
            proceed_visible = merchantProceedButton is not null && IsNodeVisible(merchantProceedButton),
            items = merchantSlots.Select((slot, index) => BuildShopSlotPayload(slot, index)).ToArray()
        };
    }

    private static object BuildShopSlotPayload(NMerchantSlot slot, int index)
    {
        var entry = slot.Entry;
        var cardEntry = entry as MerchantCardEntry;
        var relicEntry = entry as MerchantRelicEntry;
        var potionEntry = entry as MerchantPotionEntry;
        var cardRemovalEntry = entry as MerchantCardRemovalEntry;

        return new
        {
            index,
            slot_type = slot.GetType().Name,
            item_kind = ResolveMerchantEntryKind(entry),
            title = DescribeMerchantEntry(entry),
            description = DescribeMerchantEntryDescription(entry),
            cost = entry?.Cost,
            enough_gold = entry?.EnoughGold ?? false,
            is_stocked = entry?.IsStocked ?? false,
            is_affordable = CanPurchaseMerchantEntry(entry),
            is_on_sale = cardEntry?.IsOnSale,
            used = cardRemovalEntry?.Used,
            card = cardEntry is null ? null : BuildCardPayload(cardEntry.CreationResult?.Card),
            relic = relicEntry is null ? null : BuildRelicPayload(relicEntry.Model),
            potion = potionEntry is null ? null : BuildPotionPayload(potionEntry.Model)
        };
    }

    private static string ResolveMerchantEntryKind(MerchantEntry? entry)
    {
        return entry switch
        {
            MerchantCardEntry => "card",
            MerchantRelicEntry => "relic",
            MerchantPotionEntry => "potion",
            MerchantCardRemovalEntry => "card_removal",
            null => "missing",
            _ => entry.GetType().Name
        };
    }

    private static string DescribeMerchantEntry(MerchantEntry? entry)
    {
        return entry switch
        {
            MerchantCardEntry cardEntry => TextOf(cardEntry.CreationResult?.Card?.Title),
            MerchantRelicEntry relicEntry => TextOf(relicEntry.Model?.Title),
            MerchantPotionEntry potionEntry => TextOf(potionEntry.Model?.Title),
            MerchantCardRemovalEntry => "Remove a card",
            null => "<missing>",
            _ => entry.GetType().Name
        };
    }

    private static string DescribeMerchantEntryDescription(MerchantEntry? entry)
    {
        return entry switch
        {
            MerchantCardEntry cardEntry when cardEntry.CreationResult?.Card is CardModel card => GetCardDescription(card, null),
            MerchantRelicEntry relicEntry => relicEntry.Model is null ? string.Empty : TryGetDescription(relicEntry.Model),
            MerchantPotionEntry potionEntry => potionEntry.Model is null ? string.Empty : TryGetDescription(potionEntry.Model),
            MerchantCardRemovalEntry cardRemovalEntry => cardRemovalEntry.Used
                ? "Card removal already used"
                : "Remove a card from your deck",
            null => string.Empty,
            _ => string.Empty
        };
    }

    private static object BuildRestSiteOptionPayload(RestSiteOption? option, int index)
    {
        if (option is null)
        {
            return new
            {
                index,
                missing = true
            };
        }

        return new
        {
            index,
            option_id = option.OptionId,
            option_type = option.GetType().Name,
            title = TryGetTitle(option),
            description = BuildRestSiteOptionDescription(option),
            is_enabled = option.IsEnabled
        };
    }

    private static string BuildRestSiteOptionDescription(RestSiteOption option)
    {
        switch (option)
        {
            case HealRestSiteOption healOption:
            {
                var owner = GetHiddenPropertyObjectValue(healOption, "Owner") as Player;
                if (owner is not null)
                {
                    return $"回复{FormatNumericValue(HealRestSiteOption.GetHealAmount(owner))}点生命值。";
                }

                return "回复生命值。";
            }
            case SmithRestSiteOption smithOption:
                return smithOption.IsEnabled
                    ? $"升级你牌组中的{smithOption.SmithCount}张牌。"
                    : "没有可升级的牌。";
            default:
                return TryGetDescription(option);
        }
    }


    private static object BuildMapPointPayload(NMapPoint pointNode)
    {
        var point = pointNode.Point;
        var coord = point.coord;

        return new
        {
            coord = BuildMapCoord(coord),
            point_type = point.PointType.ToString(),
            state = pointNode.State.ToString(),
            is_enabled = pointNode.IsEnabled,
            is_travelable = IsMapPointTravelable(pointNode),
            children = point.Children.Select(static child => BuildMapCoord(child.coord)).ToArray()
        };
    }

    private static object BuildRoomPayload(AbstractRoom? room)
    {
        if (room is null)
        {
            return new
            {
                missing = true
            };
        }

        return new
        {
            room_type = room.RoomType.ToString(),
            model_id = room.ModelId?.ToString() ?? string.Empty,
            is_pre_finished = room.IsPreFinished,
            is_victory_room = room.IsVictoryRoom
        };
    }

    private static object BuildModelPayload(AbstractModel? model)
    {
        if (model is null)
        {
            return new
            {
                missing = true
            };
        }

        return new
        {
            id = model.Id.ToString(),
            title = TryGetTitle(model),
            description = TryGetDescription(model),
            kind = model.GetType().Name
        };
    }

    private static object BuildCharacterPayload(CharacterModel? character)
    {
        if (character is null)
        {
            return new
            {
                missing = true
            };
        }

        return new
        {
            id = character.Id.ToString(),
            title = DescribeCharacter(character),
            description = DescribeCharacterDescription(character),
            starting_hp = character.StartingHp,
            starting_gold = character.StartingGold,
            starting_relic = BuildRelicPayload(character.StartingRelics.FirstOrDefault())
        };
    }

    private static object BuildRelicPayload(RelicModel? relic)
    {
        if (relic is null)
        {
            return new
            {
                missing = true
            };
        }

        return new
        {
            id = relic.Id.ToString(),
            model_id = relic.Id.ToString(),
            class_name = relic.GetType().Name,
            title = TryGetTitle(relic),
            description = TryGetDescription(relic),
            rarity = relic.Rarity.ToString(),
            status = relic.Status.ToString(),
            is_tradable = relic.IsTradable,
            is_allowed_in_shops = relic.IsAllowedInShops,
            is_used_up = relic.IsUsedUp,
            has_upon_pickup_effect = relic.HasUponPickupEffect,
            spawns_pets = relic.SpawnsPets,
            is_stackable = relic.IsStackable,
            is_wax = relic.IsWax,
            is_melted = relic.IsMelted,
            adds_pet = relic.AddsPet,
            stack_count = relic.StackCount,
            merchant_cost = relic.MerchantCost,
            floor_added_to_deck = relic.FloorAddedToDeck,
            show_counter = relic.ShowCounter,
            display_amount = relic.DisplayAmount,
            has_been_removed_from_state = relic.HasBeenRemovedFromState,
            dynamic_vars = BuildDynamicVarPayloads(relic.DynamicVars)
        };
    }

    private static object BuildPotionPayload(PotionModel? potion)
    {
        if (potion is null)
        {
            return new
            {
                empty = true
            };
        }

        return new
        {
            id = potion.Id.ToString(),
            model_id = potion.Id.ToString(),
            class_name = potion.GetType().Name,
            title = TryGetTitle(potion),
            description = TryGetDescription(potion),
            rarity = potion.Rarity.ToString(),
            usage = potion.Usage.ToString(),
            target_type = potion.TargetType.ToString(),
            selection_screen_prompt = DescribeText(potion.SelectionScreenPrompt, potion),
            can_throw_at_ally = SafeCanThrowPotionAtAlly(potion),
            is_usable = SafeGetPotionIsUsable(potion),
            is_queued = SafeGetPotionIsQueued(potion),
            can_be_generated_in_combat = potion.CanBeGeneratedInCombat,
            passes_custom_usability_check = potion.PassesCustomUsabilityCheck,
            has_been_removed_from_state = SafeGetPotionHasBeenRemovedFromState(potion),
            dynamic_vars = BuildDynamicVarPayloads(potion.DynamicVars)
        };
    }

    private static object? BuildMapCoord(MapCoord? coord)
    {
        return coord.HasValue ? BuildMapCoord(coord.Value) : null;
    }

    private static object BuildMapCoord(MapCoord coord)
    {
        return new
        {
            col = coord.col,
            row = coord.row
        };
    }

}
