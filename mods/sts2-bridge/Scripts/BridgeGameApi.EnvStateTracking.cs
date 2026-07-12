using System;
using System.Collections.Generic;
using System.Linq;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.MonsterMoves.Intents;

namespace Sts2McpBridge.Scripts;

/// <summary>
/// Factual state comparison and transition support for the environment adapter.
/// This partial does not compute policy hints or scalar reward.
/// </summary>
internal static partial class BridgeGameApi
{
    private static string BuildEnvRoomKey(int actIndex, int totalFloor, string? roomType, string? roomModelId)
    {
        return $"{actIndex}:{totalFloor}:{roomType ?? ""}:{roomModelId ?? ""}";
    }

    private static bool HasEnvRoomTransition(BridgeEnvSnapshot before, BridgeEnvSnapshot after)
    {
        return !BuildEnvRoomKey(before.ActIndex, before.TotalFloor, before.RoomType, before.RoomModelId)
            .Equals(BuildEnvRoomKey(after.ActIndex, after.TotalFloor, after.RoomType, after.RoomModelId), StringComparison.Ordinal);
    }

    private static bool HasMeaningfulEnvSnapshotDifference(BridgeEnvSnapshot before, BridgeEnvSnapshot after)
    {
        if (!string.Equals(before.Screen, after.Screen, StringComparison.Ordinal) ||
            !string.Equals(before.Phase, after.Phase, StringComparison.Ordinal) ||
            !string.Equals(before.SurfaceFingerprint, after.SurfaceFingerprint, StringComparison.Ordinal) ||
            before.Actionable != after.Actionable ||
            before.Done != after.Done ||
            before.RunActive != after.RunActive ||
            before.CurrentHp != after.CurrentHp ||
            before.MaxHp != after.MaxHp ||
            before.PlayerBlock != after.PlayerBlock ||
            before.CurrentEnergy != after.CurrentEnergy ||
            before.Gold != after.Gold ||
            before.ActIndex != after.ActIndex ||
            before.TotalFloor != after.TotalFloor ||
            !string.Equals(before.RoomType ?? string.Empty, after.RoomType ?? string.Empty, StringComparison.Ordinal) ||
            !string.Equals(before.RoomModelId ?? string.Empty, after.RoomModelId ?? string.Empty, StringComparison.Ordinal) ||
            before.RelicCount != after.RelicCount ||
            before.PotionCount != after.PotionCount ||
            before.DeckCount != after.DeckCount ||
            before.CombatInProgress != after.CombatInProgress ||
            before.RoomPreFinished != after.RoomPreFinished ||
            before.LegalActions.Length != after.LegalActions.Length ||
            before.ResolvedActions.Count != after.ResolvedActions.Count)
        {
            return true;
        }

        if (HasMeaningfulEnvActionDifference(before.ResolvedActions, after.ResolvedActions))
        {
            return true;
        }

        return HasMeaningfulEnvEnemyDifference(before.EnemyStates, after.EnemyStates);
    }

    private static bool IsEnvIntermediateDecisionSurface(BridgeEnvSnapshot snapshot)
    {
        if (snapshot.Done)
        {
            return false;
        }

        return string.Equals(snapshot.Phase, "card_selection", StringComparison.Ordinal) ||
               string.Equals(snapshot.Phase, "deck_upgrade", StringComparison.Ordinal);
    }

    private static bool HasMeaningfulEnvActionDifference(
        IReadOnlyList<BridgeResolvedAction> before,
        IReadOnlyList<BridgeResolvedAction> after)
    {
        if (before.Count != after.Count)
        {
            return true;
        }

        for (var index = 0; index < before.Count; index++)
        {
            if (!string.Equals(before[index].ActionId, after[index].ActionId, StringComparison.Ordinal))
            {
                return true;
            }
        }

        return false;
    }

    private static bool HasMeaningfulEnvEnemyDifference(
        IReadOnlyList<BridgeEnvEnemyState> before,
        IReadOnlyList<BridgeEnvEnemyState> after)
    {
        if (before.Count != after.Count)
        {
            return true;
        }

        var afterById = after.ToDictionary(static enemy => enemy.CombatId);
        foreach (var beforeEnemy in before)
        {
            if (!afterById.TryGetValue(beforeEnemy.CombatId, out var afterEnemy))
            {
                return true;
            }

            if (beforeEnemy.CurrentHp != afterEnemy.CurrentHp ||
                beforeEnemy.Block != afterEnemy.Block ||
                beforeEnemy.IsAlive != afterEnemy.IsAlive ||
                beforeEnemy.IntentDamageToPlayer != afterEnemy.IntentDamageToPlayer ||
                beforeEnemy.Weak != afterEnemy.Weak ||
                beforeEnemy.Vulnerable != afterEnemy.Vulnerable)
            {
                return true;
            }
        }

        return false;
    }

    private static int GetPrimaryPlayerCurrentEnergy(BridgeWorldContext context) => GetPrimaryPlayer(context)?.PlayerCombatState?.Energy ?? 0;

    private static int GetPrimaryPlayerCurrentBlock(BridgeWorldContext context) => GetPrimaryPlayer(context)?.Creature?.Block ?? 0;

    private static int GetPrimaryPlayerPotionCount(BridgeWorldContext context) => GetPrimaryPlayer(context)?.PotionSlots.Count(static slot => slot is not null) ?? 0;

    private static IReadOnlyList<BridgeEnvEnemyState> BuildEnvEnemyStates(BridgeWorldContext context)
    {
        if (context.CombatState is null)
        {
            return Array.Empty<BridgeEnvEnemyState>();
        }

        var primaryPlayer = GetPrimaryPlayer(context)?.Creature;
        return context.CombatState.Creatures
            .Where(static creature => creature.IsEnemy)
            .Select(creature => BuildEnvEnemyState(creature, primaryPlayer))
            .ToArray();
    }

    private static BridgeEnvEnemyState BuildEnvEnemyState(Creature creature, Creature? primaryPlayer)
    {
        return new BridgeEnvEnemyState
        {
            CombatId = creature.CombatId ?? 0u,
            CurrentHp = creature.CurrentHp,
            Block = creature.Block,
            IsAlive = creature.IsAlive,
            IntentDamageToPlayer = GetEnvEnemyIntentDamageToPrimaryPlayer(creature, primaryPlayer),
            Weak = GetEnvCreaturePowerAmount(creature, "Weak", "虚弱", "虛弱"),
            Vulnerable = GetEnvCreaturePowerAmount(creature, "Vulnerable", "易伤", "易傷")
        };
    }

    private static int GetEnvCreaturePowerAmount(Creature creature, params string[] aliases)
    {
        foreach (var power in creature.Powers)
        {
            var title = TextOf(power.Title);
            if (string.IsNullOrWhiteSpace(title))
            {
                continue;
            }

            foreach (var alias in aliases)
            {
                if (!string.IsNullOrWhiteSpace(alias) &&
                    title.IndexOf(alias, StringComparison.OrdinalIgnoreCase) >= 0)
                {
                    return power.Amount;
                }
            }
        }

        return 0;
    }

    private static int GetEnvEnemyIntentDamageToPrimaryPlayer(Creature enemy, Creature? primaryPlayer)
    {
        var monster = enemy.Monster;
        if (monster is null || !enemy.IsAlive)
        {
            return 0;
        }

        IReadOnlyList<Creature> targets = primaryPlayer is not null && primaryPlayer.IsAlive
            ? new[] { primaryPlayer }
            : ResolveMonsterIntentTargets(enemy);
        var intents = SafeGetMonsterIntents(monster, monster.NextMove);
        var totalDamage = 0;

        foreach (var intent in intents)
        {
            var damage = intent switch
            {
                SingleAttackIntent singleAttackIntent => SafeGetIntentTotalDamage(singleAttackIntent, targets, enemy),
                MultiAttackIntent multiAttackIntent => SafeGetIntentTotalDamage(multiAttackIntent, targets, enemy),
                _ => null
            };

            if (damage.HasValue && damage.Value > 0)
            {
                totalDamage += damage.Value;
            }
        }

        return totalDamage;
    }

    private static IReadOnlyList<BridgeEnvDeckEntry> BuildEnvDeckEntries(BridgeWorldContext context)
    {
        var deck = GetPrimaryPlayer(context)?.Deck?.Cards;
        if (deck is null || deck.Count == 0)
        {
            return Array.Empty<BridgeEnvDeckEntry>();
        }

        return deck.Select(card => new BridgeEnvDeckEntry
        {
            Ref = GetCardReference(card),
            CardId = card.Id.ToString(),
            Title = string.IsNullOrWhiteSpace(card.Title)
                ? DescribeText(card.TitleLocString, card)
                : DescribeText(card.Title, card)
        }).ToArray();
    }

    private static Player? GetPrimaryPlayer(BridgeWorldContext context)
    {
        if (context.RunState?.Players.Count > 0)
        {
            return context.RunState.Players[0];
        }

        if (context.CombatState?.Players.Count > 0)
        {
            return context.CombatState.Players[0];
        }

        return null;
    }

    private static int GetPrimaryPlayerCurrentHp(BridgeWorldContext context) => GetPrimaryPlayer(context)?.Creature?.CurrentHp ?? 0;

    private static int GetPrimaryPlayerMaxHp(BridgeWorldContext context) => GetPrimaryPlayer(context)?.Creature?.MaxHp ?? 0;

    private static int GetPrimaryPlayerGold(BridgeWorldContext context) => GetPrimaryPlayer(context)?.Gold ?? 0;

    private static int GetPrimaryPlayerRelicCount(BridgeWorldContext context) => GetPrimaryPlayer(context)?.Relics.Count ?? 0;

    private static int GetPrimaryPlayerDeckCount(BridgeWorldContext context) => GetPrimaryPlayer(context)?.Deck?.Cards.Count ?? 0;
}
