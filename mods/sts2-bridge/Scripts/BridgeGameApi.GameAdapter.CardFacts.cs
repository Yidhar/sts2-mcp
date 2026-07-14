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
    private static string FirstNonEmptyText(params object?[] values)
    {
        foreach (var value in values)
        {
            var text = DescribeText(value, value);
            if (!string.IsNullOrWhiteSpace(text))
            {
                return text;
            }
        }
        return string.Empty;
    }

    private static decimal? FirstNumber(params object?[] values)
    {
        foreach (var value in values)
        {
            if (value is null)
            {
                continue;
            }
            try
            {
                return Convert.ToDecimal(value, CultureInfo.InvariantCulture);
            }
            catch
            {
                // ignore non-scalar modifier fields
            }
        }
        return null;
    }

    private static bool SafeGetCardIsPlayable(CardModel? card)
    {
        if (card is null)
        {
            return false;
        }

        if (card.Pile?.IsCombatPile != true)
        {
            return false;
        }

        try
        {
            return GetHiddenPropertyValue<bool>(card, "IsPlayable") ?? false;
        }
        catch
        {
            return false;
        }
    }

    private static void AddPotionRewardSkipActions(List<BridgeResolvedAction> actions, BridgeWorldContext context)
    {
        var skippablePotionRewards = ResolveSkippablePotionRewardControls(context.RewardsScreen);
        for (var index = 0; index < skippablePotionRewards.Count; index++)
        {
            var (rewardControl, potionReward) = skippablePotionRewards[index];
            var actionId = $"reward:skip_potion:{index}";
            var rewardPayload = BuildRewardPayload(potionReward);
            actions.Add(new BridgeResolvedAction
            {
                ActionId = actionId,
                Payload = new
                {
                    action_id = actionId,
                    kind = "reward",
                    selection_action = "skip_potion",
                    index,
                    label = $"Skip potion reward {index}",
                    reward = rewardPayload,
                    screen = context.Screen
                },
                Execute = () => InvokeRewardSkipAction(context.RewardsScreen, rewardControl)
            });
        }
    }

    private static object? BuildCardUpgradePreviewPayload(CardModel? card)
    {
        if (card is null)
        {
            return null;
        }

        try
        {
            var cardScope = card.CardScope;
            if (cardScope is null)
            {
                return null;
            }

            var upgradedCard = cardScope.CloneCard(card);
            upgradedCard.UpgradeInternal();
            upgradedCard.UpgradePreviewType = card.Pile?.IsCombatPile == true
                ? CardUpgradePreviewType.Combat
                : CardUpgradePreviewType.Deck;
            return BuildCardPayload(upgradedCard);
        }
        catch
        {
            return null;
        }
    }

    private static string GetCardDescription(CardModel card, Creature? previewTarget)
    {
        try
        {
            var pileType = card.Pile?.Type ?? (card.IsInCombat ? PileType.Hand : PileType.Deck);
            var description = card.GetDescriptionForPile(pileType, previewTarget!);
            if (!string.IsNullOrWhiteSpace(description))
            {
                return DescribeText(description, card);
            }
        }
        catch (Exception ex)
        {
            BridgeDebugTrace.Write(
                $"card_description_preview_failed card_type={card.GetType().FullName}: {ex.GetBaseException().Message}");
        }

        return DescribeText(card.Description, card);
    }

    private static DynamicVarSet? BuildCardPreviewVarSet(CardModel card, Creature? previewTarget)
    {
        try
        {
            var dynamicVars = card.DynamicVars;
            var previewVars = dynamicVars?.Clone(card);
            if (previewVars is null)
            {
                return null;
            }

            previewVars.ClearPreview();
            card.UpdateDynamicVarPreview(ResolveCardPreviewMode(card), previewTarget!, previewVars);
            return previewVars;
        }
        catch
        {
            return card.DynamicVars;
        }
    }

    private static CardPreviewMode ResolveCardPreviewMode(CardModel card)
    {
        return card.TargetType == TargetType.AllEnemies
            ? CardPreviewMode.MultiCreatureTargeting
            : CardPreviewMode.Normal;
    }

    private static int? SafeResolveCardStarCost(CardModel card)
    {
        int? currentStars = null;

        try
        {
            // Star-X has the same stale-preview failure mode as energy X: the
            // resolved value is determined by the live combat star resource, not
            // by a card-side preview cached earlier in the turn/animation window.
            currentStars = card.Owner?.PlayerCombatState?.Stars;
            if (card.HasStarCostX && currentStars.HasValue)
            {
                return Math.Max(currentStars.Value, 0);
            }

            return card.CurrentStarCost;
        }
        catch
        {
            return card.HasStarCostX && currentStars.HasValue ? Math.Max(currentStars.Value, 0) : null;
        }
    }

    private static object[] BuildDynamicVarPayloads(DynamicVarSet? dynamicVarSet)
    {
        return dynamicVarSet?.Values
            .Select(BuildDynamicVarPayload)
            .Where(static payload => payload is not null)
            .Cast<object>()
            .ToArray()
            ?? Array.Empty<object>();
    }

    private static object? BuildDynamicVarPayload(DynamicVar? dynamicVar)
    {
        if (dynamicVar is null)
        {
            return null;
        }

        var runtimeType = dynamicVar.GetType();
        var powerVarType = FindGenericBase(runtimeType, "PowerVar`1");
        return new
        {
            name = dynamicVar.Name,
            var_type = runtimeType.Name,
            family = ResolveDynamicVarFamily(runtimeType, powerVarType is not null),
            power_type = powerVarType?.GetGenericArguments()[0].Name,
            value_props = GetHiddenPropertyObjectValue(dynamicVar, "Props")?.ToString(),
            int_value = dynamicVar.IntValue,
            preview_value = dynamicVar.PreviewValue,
            base_value = dynamicVar.BaseValue,
            enchanted_value = dynamicVar.EnchantedValue,
            was_just_upgraded = dynamicVar.WasJustUpgraded
        };
    }

    private static Type? FindGenericBase(Type runtimeType, string genericTypeName)
    {
        for (var type = runtimeType; type is not null; type = type.BaseType)
        {
            if (type.IsGenericType && type.GetGenericTypeDefinition().Name == genericTypeName)
            {
                return type;
            }
        }

        return null;
    }

    private static string ResolveDynamicVarFamily(Type runtimeType, bool isPowerVar)
    {
        if (isPowerVar)
        {
            return "power";
        }

        return runtimeType.Name switch
        {
            "DamageVar" or "CalculatedDamageVar" or "ExtraDamageVar" or "OstyDamageVar" => "damage",
            "BlockVar" or "CalculatedBlockVar" => "block",
            "CardsVar" => "cards",
            "EnergyVar" => "energy",
            "RepeatVar" => "repeat",
            "HpLossVar" => "hp_loss",
            "HealVar" => "heal",
            "MaxHpVar" => "max_hp",
            "GoldVar" => "gold",
            "StarsVar" => "stars",
            "ForgeVar" => "forge",
            "SummonVar" => "summon",
            "BoolVar" => "boolean",
            "StringVar" => "string",
            _ => "value"
        };
    }

    private static bool HasCardOnDrawEffect(CardModel card)
    {
        try
        {
            var declaringType = card.GetType().GetMethod(nameof(AbstractModel.AfterCardDrawn))?.DeclaringType;
            return declaringType is not null && declaringType != typeof(AbstractModel);
        }
        catch
        {
            return false;
        }
    }

    private static string[] BuildCardHoverTipIds(CardModel card)
    {
        try
        {
            return card.HoverTips
                .Select(static hoverTip => hoverTip.Id)
                .Where(static id => !string.IsNullOrWhiteSpace(id))
                .Distinct(StringComparer.Ordinal)
                .OrderBy(static id => id, StringComparer.Ordinal)
                .ToArray();
        }
        catch
        {
            return Array.Empty<string>();
        }
    }


}
