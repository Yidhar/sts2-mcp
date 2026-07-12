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
    private static Dictionary<string, object?> BuildCardModifierSummaryPayload(
        HashSet<CardKeyword> keywordSet,
        object[] afflictions,
        object[] enchantments,
        string cardType)
    {
        var summary = new Dictionary<string, object?>(StringComparer.Ordinal);
        bool Flag(string name) => summary.TryGetValue(name, out var value) && value is bool b && b;
        decimal NumValue(string name) => summary.TryGetValue(name, out var value) && value is decimal d ? d : 0m;
        void SetFlag(string name) => summary[name] = true;
        void AddNum(string name, decimal delta) => summary[name] = NumValue(name) + delta;

        foreach (var payload in afflictions.Concat(enchantments))
        {
            var values = payload.GetType().GetProperty("semantic_values")?.GetValue(payload) as Dictionary<string, object?>;
            if (values is null) continue;
            foreach (var pair in values)
            {
                if (pair.Value is bool b)
                {
                    if (b) SetFlag(pair.Key);
                }
                else if (pair.Value is decimal d)
                {
                    AddNum(pair.Key, d);
                }
                else if (!summary.ContainsKey(pair.Key))
                {
                    summary[pair.Key] = pair.Value;
                }
            }
        }

        if (keywordSet.Contains(CardKeyword.Exhaust)) SetFlag("base_exhaust");
        if (keywordSet.Contains(CardKeyword.Ethereal)) SetFlag("base_ethereal");
        if (keywordSet.Contains(CardKeyword.Retain)) SetFlag("base_retain");
        summary["effective_exhaust"] = (keywordSet.Contains(CardKeyword.Exhaust) || Flag("adds_exhaust")) && !Flag("removes_exhaust");
        summary["effective_ethereal"] = keywordSet.Contains(CardKeyword.Ethereal) || Flag("adds_ethereal");
        summary["effective_retain"] = keywordSet.Contains(CardKeyword.Retain) || Flag("adds_retain");
        summary["card_type"] = cardType;
        return summary;
    }

    private static object BuildCardFlowPayload(CardModel card, HashSet<CardKeyword> keywordSet, Dictionary<string, object?> summary)
    {
        bool Flag(string name) => summary.TryGetValue(name, out var value) && value is bool b && b;
        var cardType = card.Type.ToString();
        var exhaustOnPlay = Flag("effective_exhaust");
        var retainOnSkip = Flag("effective_retain");
        var etherealOnSkip = Flag("effective_ethereal");
        var playDestination = string.Equals(cardType, "Power", StringComparison.OrdinalIgnoreCase)
            ? "power"
            : exhaustOnPlay ? "exhaust" : "discard";
        var skipDestination = retainOnSkip ? "hand" : etherealOnSkip ? "exhaust" : "discard";
        return new
        {
            source_zone = card.Pile?.Type.ToString(),
            destination_if_played = playDestination,
            destination_if_skipped_end_turn = skipDestination,
            will_exhaust_on_play = exhaustOnPlay,
            will_retain_on_end_turn = retainOnSkip,
            will_ethereal_exhaust_on_end_turn = etherealOnSkip,
            strategic_skip_candidate = exhaustOnPlay || retainOnSkip || Flag("energy_loss_on_play") || Flag("self_damage")
        };
    }

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

    private static bool CanExposeShopOpenAction(BridgeWorldContext context)
    {
        var roomKey = BuildShopOpenLimiterRoomKey(context);
        if (string.IsNullOrWhiteSpace(roomKey))
        {
            ResetShopOpenLimiter();
            return true;
        }

        lock (ShopOpenLimiterSync)
        {
            if (!string.Equals(_shopOpenLimiterRoomKey, roomKey, StringComparison.Ordinal))
            {
                _shopOpenLimiterRoomKey = roomKey;
                _shopOpenLimiterCount = 0;
            }

            return _shopOpenLimiterCount < MaxShopOpenActionsPerRoom;
        }
    }

    private static void RecordShopOpenAction(BridgeWorldContext context)
    {
        var roomKey = BuildShopOpenLimiterRoomKey(context);
        if (string.IsNullOrWhiteSpace(roomKey))
        {
            return;
        }

        lock (ShopOpenLimiterSync)
        {
            if (!string.Equals(_shopOpenLimiterRoomKey, roomKey, StringComparison.Ordinal))
            {
                _shopOpenLimiterRoomKey = roomKey;
                _shopOpenLimiterCount = 0;
            }

            if (_shopOpenLimiterCount < int.MaxValue)
            {
                _shopOpenLimiterCount++;
            }
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

    private static void ResetShopOpenLimiter()
    {
        lock (ShopOpenLimiterSync)
        {
            _shopOpenLimiterRoomKey = null;
            _shopOpenLimiterCount = 0;
        }
    }

    private static string? BuildShopOpenLimiterRoomKey(BridgeWorldContext context)
    {
        if (context.MerchantRoom is null &&
            context.MerchantInventory is null)
        {
            return null;
        }

        var runState = context.RunState;
        if (runState is null)
        {
            return "shop:no-run";
        }

        var coordPart = runState.CurrentMapCoord.HasValue
            ? $"{runState.CurrentMapCoord.Value.col},{runState.CurrentMapCoord.Value.row}"
            : "?,?";
        var roomTypePart = runState.CurrentRoom?.RoomType.ToString() ?? "Unknown";
        var roomModelPart = runState.CurrentRoom?.ModelId?.ToString() ?? string.Empty;

        return string.Concat(
            runState.TotalFloor.ToString(CultureInfo.InvariantCulture),
            "|",
            coordPart,
            "|",
            roomTypePart,
            "|",
            roomModelPart);
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

    private static bool IsBodySlamCard(CardModel card)
    {
        try
        {
            var id = card.Id.ToString();
            if (string.Equals(id, "CARD.BODY_SLAM", StringComparison.OrdinalIgnoreCase) ||
                id.Contains("BODY_SLAM", StringComparison.OrdinalIgnoreCase))
            {
                return true;
            }

            var className = card.GetType().Name;
            if (string.Equals(className, "BodySlam", StringComparison.OrdinalIgnoreCase) ||
                className.Contains("BodySlam", StringComparison.OrdinalIgnoreCase))
            {
                return true;
            }

            var title = string.IsNullOrWhiteSpace(card.Title)
                ? DescribeText(card.TitleLocString, card)
                : DescribeText(card.Title, card);
            return title.Contains("全身撞击", StringComparison.OrdinalIgnoreCase) ||
                   title.Contains("全身撞擊", StringComparison.OrdinalIgnoreCase) ||
                   title.Contains("Body Slam", StringComparison.OrdinalIgnoreCase);
        }
        catch
        {
            return false;
        }
    }

    private static int? SafeResolveBodySlamDamage(CardModel card)
    {
        if (!IsBodySlamCard(card))
        {
            return null;
        }

        try
        {
            var currentBlock = card.Owner?.Creature?.Block;
            return currentBlock.HasValue ? Math.Max(currentBlock.Value, 0) : null;
        }
        catch
        {
            return null;
        }
    }

    private static int? SafeResolveCardEnergyXValue(CardModel card)
    {
        int? currentEnergy = null;

        try
        {
            // X-cost is a live combat resource decision. Prefer the owner's current
            // combat energy over CardModel.ResolveEnergyXValue(), because the latter
            // can retain the turn-start/preview value and make a 0-energy X card look
            // like X=3 to the policy. Extra X-effect bonuses should be represented as
            // separate effect modifiers, not as spendable energy.
            currentEnergy = card.Owner?.PlayerCombatState?.Energy;
            if (currentEnergy.HasValue)
            {
                return Math.Max(currentEnergy.Value, 0);
            }

            return Math.Max(card.ResolveEnergyXValue(), 0);
        }
        catch
        {
            return currentEnergy.HasValue ? Math.Max(currentEnergy.Value, 0) : null;
        }
    }

    private static int? SafeResolveCardStarCost(CardModel card)
    {
        int? currentStars = null;

        try
        {
            // Star-X has the same stale-preview failure mode as energy X: the
            // strategic value is determined by the live combat star resource, not
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

    private static string? ResolveXCostSemantics(
        CardModel card,
        string description,
        int? damagePerHit,
        int? totalDamage,
        int? repeats,
        int? xCostValue)
    {
        if (!card.EnergyCost.CostsX || !xCostValue.HasValue || xCostValue.Value <= 0)
        {
            return null;
        }

        var normalizedDescription = NormalizeComparableText(description).ToLowerInvariant();
        var looksLikeRepeatPerEnergyText =
            normalizedDescription.Contains("x次", StringComparison.Ordinal) ||
            normalizedDescription.Contains("x times", StringComparison.Ordinal) ||
            normalizedDescription.Contains("times equal to x", StringComparison.Ordinal);

        if (looksLikeRepeatPerEnergyText &&
            (damagePerHit.HasValue || totalDamage.HasValue))
        {
            return "repeat_per_energy";
        }

        if (card.Type == CardType.Attack &&
            xCostValue.Value > 1 &&
            repeats.GetValueOrDefault(1) <= 1 &&
            (damagePerHit.HasValue || totalDamage.HasValue))
        {
            return "repeat_per_energy";
        }

        return null;
    }

    private static (int? DamagePerHit, int? TotalDamage, int? Repeats) ApplyXCostPreviewMapping(
        int? damagePerHit,
        int? totalDamage,
        int? repeats,
        int? xCostValue,
        string? xCostSemantics)
    {
        if (!string.Equals(xCostSemantics, "repeat_per_energy", StringComparison.Ordinal) ||
            !xCostValue.HasValue ||
            xCostValue.Value <= 0)
        {
            return (damagePerHit, totalDamage, repeats);
        }

        var mappedDamagePerHit = damagePerHit ?? totalDamage;
        var mappedRepeats = xCostValue.Value;
        var mappedTotalDamage = mappedDamagePerHit.HasValue
            ? mappedDamagePerHit.Value * mappedRepeats
            : totalDamage;

        return (mappedDamagePerHit, mappedTotalDamage, mappedRepeats);
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

        return new
        {
            name = dynamicVar.Name,
            int_value = dynamicVar.IntValue,
            preview_value = dynamicVar.PreviewValue,
            base_value = dynamicVar.BaseValue,
            enchanted_value = dynamicVar.EnchantedValue,
            was_just_upgraded = dynamicVar.WasJustUpgraded
        };
    }

    private static int? GetDynamicVarInt(DynamicVarSet? dynamicVarSet, string key)
    {
        if (dynamicVarSet is null || string.IsNullOrWhiteSpace(key))
        {
            return null;
        }

        try
        {
            return dynamicVarSet.TryGetValue(key, out var dynamicVar)
                ? GetDynamicVarInt(dynamicVar)
                : null;
        }
        catch
        {
            return null;
        }
    }

    private static int? GetDynamicVarInt(DynamicVar? dynamicVar)
    {
        if (dynamicVar is null)
        {
            return null;
        }

        return decimal.Truncate(dynamicVar.PreviewValue) != 0m
            ? (int)decimal.Truncate(dynamicVar.PreviewValue)
            : dynamicVar.IntValue;
    }

    private static string BuildCardEffectSummary(
        int? totalDamage,
        int? damagePerHit,
        int? repeats,
        int? totalBlock,
        int? drawCount,
        int? healAmount,
        int? hpLossAmount,
        int? weakAmount,
        int? vulnerableAmount,
        int? poisonAmount,
        int? strengthAmount,
        int? dexterityAmount,
        int? summonCount,
        int? extraDamage,
        int? xCostValue)
    {
        var parts = new List<string>();

        if (damagePerHit.HasValue && repeats.GetValueOrDefault(1) > 1)
        {
            parts.Add($"{damagePerHit.Value} x {repeats!.Value} damage");
        }
        else if (totalDamage.HasValue && totalDamage.Value != 0)
        {
            parts.Add($"{totalDamage.Value} damage");
        }

        if (totalBlock.HasValue && totalBlock.Value != 0)
        {
            parts.Add($"{totalBlock.Value} block");
        }

        if (drawCount.HasValue && drawCount.Value != 0)
        {
            parts.Add($"draw {drawCount.Value}");
        }

        if (healAmount.HasValue && healAmount.Value != 0)
        {
            parts.Add($"heal {healAmount.Value}");
        }

        if (hpLossAmount.HasValue && hpLossAmount.Value != 0)
        {
            parts.Add($"lose {hpLossAmount.Value} HP");
        }

        if (weakAmount.HasValue && weakAmount.Value != 0)
        {
            parts.Add($"apply {weakAmount.Value} Weak");
        }

        if (vulnerableAmount.HasValue && vulnerableAmount.Value != 0)
        {
            parts.Add($"apply {vulnerableAmount.Value} Vulnerable");
        }

        if (poisonAmount.HasValue && poisonAmount.Value != 0)
        {
            parts.Add($"apply {poisonAmount.Value} Poison");
        }

        if (strengthAmount.HasValue && strengthAmount.Value != 0)
        {
            parts.Add($"gain {strengthAmount.Value} Strength");
        }

        if (dexterityAmount.HasValue && dexterityAmount.Value != 0)
        {
            parts.Add($"gain {dexterityAmount.Value} Dexterity");
        }

        if (summonCount.HasValue && summonCount.Value != 0)
        {
            parts.Add($"summon {summonCount.Value}");
        }

        if (extraDamage.HasValue && extraDamage.Value != 0)
        {
            parts.Add($"{extraDamage.Value} extra damage");
        }

        if (xCostValue.HasValue && xCostValue.Value != 0)
        {
            parts.Add($"X={xCostValue.Value}");
        }

        return string.Join(" + ", parts);
    }

}
