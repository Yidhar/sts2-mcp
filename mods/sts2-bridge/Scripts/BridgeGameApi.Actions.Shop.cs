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
    private static bool CanPurchaseMerchantEntry(MerchantEntry? entry)
    {
        return entry is not null &&
               entry.IsStocked &&
               entry.EnoughGold;
    }

    private static void ExecuteShopPurchase(NMerchantSlot slot, NMerchantInventory inventory)
    {
        var merchantInventory = inventory.Inventory;
        var onTryPurchase = FindMethod(slot.GetType(), "OnTryPurchase", 1);
        if (onTryPurchase is not null)
        {
            onTryPurchase.Invoke(slot, new object?[] { merchantInventory });
            return;
        }

        if (slot.Entry is not null)
        {
            var entryMethod = FindMethod(slot.Entry.GetType(), "OnTryPurchase", 2) ??
                              FindMethod(slot.Entry.GetType(), "OnTryPurchaseWrapper", 2);
            if (entryMethod is not null)
            {
                entryMethod.Invoke(slot.Entry, new object?[] { merchantInventory, false });
                return;
            }
        }

        if (TryInvokeParameterless(slot, "OnReleased"))
        {
            return;
        }

        throw new BridgeRequestException(
            HttpStatusCode.Conflict,
            "shop_purchase_failed",
            $"Could not purchase shop item '{DescribeMerchantEntry(slot.Entry)}'.");
    }

    private static string DescribeCreatureTarget(Creature creature)
    {
        return $"{creature.Name} (combat_id {creature.CombatId})";
    }

    private static string BuildResolvedTargetLabel(string? actionSuffix, Creature? target)
    {
        if (target is null)
        {
            return string.IsNullOrWhiteSpace(actionSuffix) ? string.Empty : actionSuffix;
        }

        var prefix = string.IsNullOrWhiteSpace(actionSuffix)
            ? string.Empty
            : $"{actionSuffix} = ";
        return $"{prefix}{DescribeCreatureTarget(target)}";
    }

    private static object? BuildResolvedTargetMapping(string? actionSuffix, Creature? target)
    {
        if (string.IsNullOrWhiteSpace(actionSuffix) && target is null)
        {
            return null;
        }

        return new
        {
            action_suffix = actionSuffix,
            combat_id = target?.CombatId,
            name = target?.Name,
            side = target?.Side.ToString(),
            label = BuildResolvedTargetLabel(actionSuffix, target)
        };
    }

}
