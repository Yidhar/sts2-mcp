using System.Collections.Generic;
using System.Diagnostics;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Text.RegularExpressions;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.MonsterMoves.Intents;
using MegaCrit.Sts2.Core.Models.Powers;
using MegaCrit.Sts2.Core.Nodes;

namespace Sts2McpBridge.Scripts;

internal static partial class BridgeGameApi
{
    private static BridgeLegacyEnvRewardBreakdown BuildLegacyEnvRewardBreakdown(
        BridgeEnvEpisode episode,
        BridgeEnvSnapshot before,
        BridgeEnvSnapshot after,
        BridgeResolvedActionSelection? selectedAction,
        bool truncated,
        string? actionError)
    {
        var combatSandbox = string.Equals(episode.EpisodeMode, "combat_sandbox", StringComparison.Ordinal);
        var maxHp = Math.Max(after.MaxHp, before.MaxHp);
        var hpLoss = Math.Max(0, before.CurrentHp - after.CurrentHp);
        var hpGain = Math.Max(0, after.CurrentHp - before.CurrentHp);
        var hpLossNormalized = maxHp > 0 ? (double)hpLoss / maxHp : 0d;
        var hpGainNormalized = maxHp > 0 ? (double)hpGain / maxHp : 0d;
        var rawFloorDelta = Math.Max(0, after.TotalFloor - before.TotalFloor);
        var roomComplete = HasEnvRoomTransition(before, after) ? 1 : 0;
        var combatRoom = IsEnvCombatRewardRoom(before.RoomType);
        var combatRoomComplete = roomComplete == 1 && combatRoom ? 1 : 0;
        var roomHpMax = episode.RoomStartMaxHp > 0 ? episode.RoomStartMaxHp : maxHp;
        var death = after.Done && after.CurrentHp <= 0 ? 1 : 0;
        var combatRoomSettled = combatRoom && (combatRoomComplete == 1 || death == 1);
        var rawRoomHpDeltaNormalized = combatRoomSettled && roomHpMax > 0
            ? (double)(after.CurrentHp - episode.RoomStartHp) / roomHpMax
            : 0d;
        var rawActClear = Math.Max(0, after.ActIndex - before.ActIndex);
        var victory = after.Done && after.CurrentHp > 0 ? 1 : 0;
        var rawRelicGainCount = Math.Max(0, after.RelicCount - before.RelicCount);
        var rawGoldGain = Math.Max(0, after.Gold - before.Gold);
        var rawGoldSpend = Math.Max(0, before.Gold - after.Gold);
        var maxHpGain = Math.Max(0, after.MaxHp - before.MaxHp);
        var rawMaxHpGainNormalized = maxHp > 0 ? (double)maxHpGain / maxHp : 0d;
        var anomalyReasons = new List<string>();

        if (combatSandbox)
        {
            if (rawGoldGain >= EnvCombatSandboxGoldDeltaAnomalyThreshold ||
                rawGoldSpend >= EnvCombatSandboxGoldDeltaAnomalyThreshold)
            {
                anomalyReasons.Add("combat_sandbox_gold_delta_out_of_range");
            }

            if (rawFloorDelta > EnvCombatSandboxFloorDeltaAnomalyThreshold)
            {
                anomalyReasons.Add("combat_sandbox_floor_delta_out_of_range");
            }

            if (rawActClear > EnvCombatSandboxActDeltaAnomalyThreshold)
            {
                anomalyReasons.Add("combat_sandbox_act_delta_out_of_range");
            }

            if (rawRelicGainCount > EnvCombatSandboxRelicGainAnomalyThreshold)
            {
                anomalyReasons.Add("combat_sandbox_relic_gain_out_of_range");
            }

            if (Math.Abs(rawRoomHpDeltaNormalized) > EnvCombatSandboxRoomHpDeltaNormalizedLimit + 1e-9d)
            {
                anomalyReasons.Add("combat_sandbox_room_hp_delta_out_of_range");
            }

            if (Math.Abs(rawMaxHpGainNormalized) > EnvCombatSandboxMaxHpGainNormalizedLimit + 1e-9d)
            {
                anomalyReasons.Add("combat_sandbox_max_hp_delta_out_of_range");
            }
        }

        var roomHpDeltaNormalized = combatSandbox
            ? Math.Clamp(
                rawRoomHpDeltaNormalized,
                -EnvCombatSandboxRoomHpDeltaNormalizedLimit,
                EnvCombatSandboxRoomHpDeltaNormalizedLimit)
            : rawRoomHpDeltaNormalized;
        var floorDelta = combatSandbox ? 0 : rawFloorDelta;
        var actClear = combatSandbox ? 0 : rawActClear;
        var relicGainCount = combatSandbox ? 0 : rawRelicGainCount;
        var goldGain = combatSandbox ? 0 : rawGoldGain;
        var goldSpend = combatSandbox ? 0 : rawGoldSpend;
        var maxHpGainNormalized = combatSandbox
            ? Math.Clamp(
                rawMaxHpGainNormalized,
                -EnvCombatSandboxMaxHpGainNormalizedLimit,
                EnvCombatSandboxMaxHpGainNormalizedLimit)
            : rawMaxHpGainNormalized;

        var combatRoomCompleteBonus = combatRoomComplete * EnvRewardCombatWinBonus;
        var combatRoomQualityBonus = roomHpDeltaNormalized * EnvRewardRoomHpDeltaWeight;
        var floorProgressBonus = floorDelta * EnvRewardFloorDeltaWeight;
        var eliteClearBonus = combatRoomComplete == 1 && IsEnvEliteRoom(before.RoomType)
            ? EnvRewardEliteClearBonus
            : 0d;
        var bossClearBonus = combatRoomComplete == 1 && IsEnvBossRoom(before.RoomType)
            ? EnvRewardBossClearBonus
            : 0d;
        var actClearBonus = actClear * EnvRewardActClearBonus;
        var relicGainBonus = relicGainCount * EnvRewardRelicGainWeight;
        var goldGainBonus = goldGain * EnvRewardGoldGainWeight;
        var goldSpendBonus = goldSpend * EnvRewardGoldSpendWeight;
        var maxHpGainBonus = maxHpGainNormalized * EnvRewardMaxHpGainWeight;
        var runVictoryBonus = victory * EnvRewardVictoryBonus;
        var actionErrorPenalty = string.IsNullOrWhiteSpace(actionError) ? 0d : EnvRewardActionErrorPenalty;
        var truncatedPenalty = truncated ? EnvRewardTruncatedPenalty : 0d;
        var total =
            combatRoomCompleteBonus +
            combatRoomQualityBonus +
            floorProgressBonus +
            eliteClearBonus +
            bossClearBonus +
            actClearBonus +
            relicGainBonus +
            goldGainBonus +
            goldSpendBonus +
            maxHpGainBonus +
            runVictoryBonus +
            death * EnvRewardDeathPenalty +
            actionErrorPenalty +
            truncatedPenalty;

        return new BridgeLegacyEnvRewardBreakdown
        {
            HpLossNormalized = RoundEnvNumber(hpLossNormalized),
            HpGainNormalized = RoundEnvNumber(hpGainNormalized),
            RoomComplete = roomComplete,
            CombatRoomComplete = combatRoomComplete,
            RoomHpDeltaNormalized = RoundEnvNumber(roomHpDeltaNormalized),
            CombatRoomCompleteBonus = RoundEnvNumber(combatRoomCompleteBonus),
            CombatRoomQualityBonus = RoundEnvNumber(combatRoomQualityBonus),
            FloorDelta = floorDelta,
            FloorProgressBonus = RoundEnvNumber(floorProgressBonus),
            ActClear = actClear,
            ActClearBonus = RoundEnvNumber(actClearBonus),
            EliteClearBonus = RoundEnvNumber(eliteClearBonus),
            BossClearBonus = RoundEnvNumber(bossClearBonus),
            RelicGainCount = relicGainCount,
            RelicGainBonus = RoundEnvNumber(relicGainBonus),
            MaxHpGainNormalized = RoundEnvNumber(maxHpGainNormalized),
            MaxHpGainBonus = RoundEnvNumber(maxHpGainBonus),
            Death = death,
            Victory = victory,
            RunVictoryBonus = RoundEnvNumber(runVictoryBonus),
            ActionErrorPenalty = RoundEnvNumber(actionErrorPenalty),
            TruncatedPenalty = RoundEnvNumber(truncatedPenalty),
            Total = RoundEnvNumber(total),
            RawFloorDelta = rawFloorDelta,
            RawActClear = rawActClear,
            RawGoldGain = rawGoldGain,
            RawGoldSpend = rawGoldSpend,
            RawRelicGainCount = rawRelicGainCount,
            RawRoomHpDeltaNormalized = RoundEnvNumber(rawRoomHpDeltaNormalized),
            RawMaxHpGainNormalized = RoundEnvNumber(rawMaxHpGainNormalized),
            RewardAnomalyClamped = anomalyReasons.Count > 0,
            RewardAnomalyReasons = anomalyReasons.Count > 0 ? anomalyReasons.ToArray() : Array.Empty<string>()
        };
    }

    private static void SyncEnvEpisodeAnchor(BridgeEnvEpisode episode, BridgeEnvSnapshot snapshot, bool force)
    {
        var roomKey = BuildEnvRoomKey(snapshot.ActIndex, snapshot.TotalFloor, snapshot.RoomType, snapshot.RoomModelId);
        if (!force && episode.RoomAnchorInitialized && string.Equals(episode.RoomKey, roomKey, StringComparison.Ordinal))
        {
            return;
        }

        episode.RoomAnchorInitialized = true;
        episode.RoomKey = roomKey;
        episode.RoomStartHp = snapshot.CurrentHp;
        episode.RoomStartMaxHp = snapshot.MaxHp;
        episode.RoomStartFloor = snapshot.TotalFloor;
        episode.RoomStartActIndex = snapshot.ActIndex;
    }

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

    private static bool IsEnvEliteRoom(string? roomType)
    {
        return roomType?.Contains("Elite", StringComparison.OrdinalIgnoreCase) == true;
    }

    private static bool IsEnvBossRoom(string? roomType)
    {
        return roomType?.Contains("Boss", StringComparison.OrdinalIgnoreCase) == true;
    }

    private static bool IsEnvCombatRewardRoom(string? roomType)
    {
        if (string.IsNullOrWhiteSpace(roomType))
        {
            return false;
        }

        return roomType.Contains("Monster", StringComparison.OrdinalIgnoreCase) ||
               IsEnvEliteRoom(roomType) ||
               IsEnvBossRoom(roomType);
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

    private static BridgeEnvEnemyState CreateMissingEnvEnemyState(uint combatId)
    {
        return new BridgeEnvEnemyState
        {
            CombatId = combatId,
            CurrentHp = 0,
            Block = 0,
            IsAlive = false,
            IntentDamageToPlayer = 0,
            Weak = 0,
            Vulnerable = 0
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
        var nextMove = monster.NextMove;
        var intents = SafeGetMonsterIntents(monster, nextMove);
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

    private static int GetEnvTotalIncomingDamageToPlayer(BridgeEnvSnapshot snapshot)
    {
        return snapshot.EnemyStates
            .Where(static enemy => enemy.IsAlive)
            .Sum(static enemy => enemy.IntentDamageToPlayer);
    }

    private static IReadOnlyList<BridgeEnvDeckEntry> BuildEnvDeckEntries(BridgeWorldContext context)
    {
        var deck = GetPrimaryPlayer(context)?.Deck?.Cards;
        if (deck is null || deck.Count == 0)
        {
            return Array.Empty<BridgeEnvDeckEntry>();
        }

        var entries = new List<BridgeEnvDeckEntry>(deck.Count);
        foreach (var card in deck)
        {
            var payload = JsonSerializer.SerializeToElement(BuildCardPayload(card));
            var rarity = TryGetNestedString(payload, "rarity");
            var title = TryGetNestedString(payload, "title");
            var cardId = TryGetNestedString(payload, "id");
            var signature = string.Join(
                "|",
                cardId ?? string.Empty,
                title ?? string.Empty,
                TryGetNestedString(payload, "type") ?? string.Empty,
                rarity ?? string.Empty,
                TryGetNestedInt(payload, "resolved_energy_cost")?.ToString() ?? string.Empty,
                TryGetNestedInt(payload, "current_star_cost")?.ToString() ?? string.Empty,
                TryGetNestedString(payload, "effect_preview", "summary") ??
                TryGetNestedString(payload, "description") ??
                string.Empty);
            entries.Add(new BridgeEnvDeckEntry
            {
                Ref = GetCardReference(card),
                CardId = cardId,
                Title = title,
                Rarity = rarity,
                Signature = signature,
                IsStarter = string.Equals(rarity, "Basic", StringComparison.OrdinalIgnoreCase)
            });
        }

        return entries;
    }

    private static BridgeEnvDeckDiff DiffEnvDeckEntries(
        IReadOnlyList<BridgeEnvDeckEntry> before,
        IReadOnlyList<BridgeEnvDeckEntry> after)
    {
        var beforeByRef = before.ToDictionary(static entry => entry.Ref, StringComparer.Ordinal);
        var afterByRef = after.ToDictionary(static entry => entry.Ref, StringComparer.Ordinal);
        var cardAddCount = 0;
        var starterCardRemoveCount = 0;
        var otherCardRemoveCount = 0;
        var cardUpgradeCount = 0;

        foreach (var afterEntry in after)
        {
            if (!beforeByRef.ContainsKey(afterEntry.Ref))
            {
                cardAddCount++;
            }
        }

        foreach (var beforeEntry in before)
        {
            if (!afterByRef.ContainsKey(beforeEntry.Ref))
            {
                if (beforeEntry.IsStarter)
                {
                    starterCardRemoveCount++;
                }
                else
                {
                    otherCardRemoveCount++;
                }

                continue;
            }

            var afterEntry = afterByRef[beforeEntry.Ref];
            if (!string.Equals(beforeEntry.Signature, afterEntry.Signature, StringComparison.Ordinal))
            {
                cardUpgradeCount++;
            }
        }

        return new BridgeEnvDeckDiff
        {
            CardAddCount = cardAddCount,
            StarterCardRemoveCount = starterCardRemoveCount,
            OtherCardRemoveCount = otherCardRemoveCount,
            CardUpgradeCount = cardUpgradeCount
        };
    }

    private static BridgeEnvActionShaping EvaluateEnvActionShaping(
        BridgeEnvSnapshot before,
        BridgeEnvSnapshot after,
        BridgeResolvedActionSelection? selectedAction,
        string? actionError)
    {
        if (selectedAction is null || !string.IsNullOrWhiteSpace(actionError))
        {
            return BridgeEnvActionShaping.None;
        }

        var payload = JsonSerializer.SerializeToElement(selectedAction.Action.Payload);
        var kind = selectedAction.Kind;
        var shaping = BridgeEnvActionShaping.None;

        if (string.Equals(kind, "card_reward", StringComparison.Ordinal))
        {
            if (string.Equals(selectedAction.Action.ActionId, "card_reward:skip", StringComparison.Ordinal) ||
                string.Equals(TryGetNestedString(payload, "selection_action"), "skip", StringComparison.Ordinal))
            {
                if (ShouldRewardSkippingBadCardReward(before))
                {
                    shaping.SkipBadCardsBonus = EnvRewardSkipBadCardsBonus;
                }
            }
            else
            {
                shaping.CardChoiceBonus = ScoreEnvCardHeuristic(TryGetNestedElement(payload, "card"), before);
            }
        }
        else if (string.Equals(kind, "shop", StringComparison.Ordinal) &&
                 string.Equals(TryGetNestedString(payload, "shop_action"), "buy", StringComparison.Ordinal) &&
                 string.Equals(TryGetNestedString(payload, "item", "item_kind"), "card", StringComparison.Ordinal))
        {
            shaping.CardChoiceBonus = ScoreEnvCardHeuristic(TryGetNestedElement(payload, "item", "card"), before);
        }
        else if (string.Equals(kind, "rest_site", StringComparison.Ordinal))
        {
            ApplyRestSiteShaping(before, payload, shaping);
        }
        else if (string.Equals(kind, "play_card", StringComparison.Ordinal))
        {
            shaping.PlayCardBonus = EnvRewardPlayCardBonus;
            ApplyCombatActionShaping(before, after, shaping);
            ApplyMissedDefenseShaping(before, after, shaping);
        }
        else if (string.Equals(kind, "use_potion", StringComparison.Ordinal))
        {
            ApplyCombatActionShaping(before, after, shaping);
            ApplyMissedDefenseShaping(before, after, shaping);
        }
        else if (string.Equals(kind, "combat", StringComparison.Ordinal) &&
                 string.Equals(selectedAction.Action.ActionId, "end_turn", StringComparison.Ordinal) &&
                 HasEnvWastedEndTurn(before))
        {
            shaping.EndTurnWastePenalty = EnvRewardEndTurnWastePenalty;
            ApplyEndTurnThreatShaping(before, shaping);
        }

        return shaping;
    }

    private static void ApplyCombatActionShaping(
        BridgeEnvSnapshot before,
        BridgeEnvSnapshot after,
        BridgeEnvActionShaping shaping)
    {
        if (!before.CombatInProgress)
        {
            return;
        }

        var maxHp = Math.Max(before.MaxHp, after.MaxHp);
        if (maxHp <= 0)
        {
            return;
        }

        var incomingBefore = GetEnvTotalIncomingDamageToPlayer(before);
        var blockBefore = Math.Max(0, before.PlayerBlock);
        var blockAfter = Math.Max(0, after.PlayerBlock);
        var addedBlock = Math.Max(0, blockAfter - blockBefore);
        var threatGapBefore = Math.Max(0, incomingBefore - blockBefore);
        var incomingAfter = GetEnvTotalIncomingDamageToPlayer(after);
        var threatGapAfter = Math.Max(0, incomingAfter - blockAfter);
        var threatGapReduction = Math.Max(0, threatGapBefore - threatGapAfter);
        var effectiveBlockAdded = Math.Min(addedBlock, threatGapBefore);
        var wastedBlockAdded = Math.Max(0, addedBlock - threatGapBefore);

        shaping.ThreatGapBefore = threatGapBefore;
        shaping.ThreatGapAfter = threatGapAfter;
        shaping.ThreatGapReduction = threatGapReduction;
        shaping.ThreatGapReductionNormalized = maxHp > 0 ? (double)threatGapReduction / maxHp : 0d;
        shaping.ThreatGapReductionBonus = shaping.ThreatGapReductionNormalized * EnvRewardThreatGapReductionWeight;
        shaping.EffectiveBlockAdded = effectiveBlockAdded;
        shaping.EffectiveBlockNormalized = maxHp > 0 ? (double)effectiveBlockAdded / maxHp : 0d;
        shaping.EffectiveBlockBonus = shaping.EffectiveBlockNormalized * EnvRewardEffectiveBlockWeight;
        shaping.WastedBlockAdded = wastedBlockAdded;
        shaping.WastedBlockNormalized = maxHp > 0 ? (double)wastedBlockAdded / maxHp : 0d;
        shaping.WastedBlockPenalty = shaping.WastedBlockNormalized * EnvRewardWastedBlockWeight;

        var afterById = after.EnemyStates.ToDictionary(static enemy => enemy.CombatId);
        var weakIntentReduction = 0;
        var vulnerableRealizedDamage = 0;

        foreach (var beforeEnemy in before.EnemyStates)
        {
            if (!afterById.TryGetValue(beforeEnemy.CombatId, out var afterEnemy))
            {
                afterEnemy = CreateMissingEnvEnemyState(beforeEnemy.CombatId);
            }

            if (beforeEnemy.IsAlive &&
                afterEnemy.IsAlive &&
                afterEnemy.Weak > beforeEnemy.Weak)
            {
                weakIntentReduction += Math.Max(0, beforeEnemy.IntentDamageToPlayer - afterEnemy.IntentDamageToPlayer);
            }

            if (beforeEnemy.Vulnerable > 0)
            {
                vulnerableRealizedDamage +=
                    Math.Max(0, beforeEnemy.Block - afterEnemy.Block) +
                    Math.Max(0, beforeEnemy.CurrentHp - afterEnemy.CurrentHp);
            }
        }

        shaping.WeakIntentReduction = weakIntentReduction;
        shaping.WeakIntentReductionNormalized = maxHp > 0 ? (double)weakIntentReduction / maxHp : 0d;
        shaping.WeakBonus = shaping.WeakIntentReductionNormalized * EnvRewardWeakIntentReductionWeight;
        shaping.VulnerableRealizedDamage = vulnerableRealizedDamage;
        shaping.VulnerableRealizedDamageNormalized = maxHp > 0 ? (double)vulnerableRealizedDamage / maxHp : 0d;
        shaping.VulnerableBonus = shaping.VulnerableRealizedDamageNormalized * EnvRewardVulnerableRealizedDamageWeight;
    }

    private static void ApplyMissedDefenseShaping(
        BridgeEnvSnapshot before,
        BridgeEnvSnapshot after,
        BridgeEnvActionShaping shaping)
    {
        if (!before.CombatInProgress ||
            shaping.ThreatGapBefore <= 0 ||
            after.CurrentEnergy > 0 ||
            shaping.ThreatGapReduction > 0 ||
            !HasEnvAvailableDefenseOption(before))
        {
            return;
        }

        var maxHp = Math.Max(before.MaxHp, after.MaxHp);
        if (maxHp <= 0)
        {
            return;
        }

        shaping.MissedDefensePenalty =
            ((double)shaping.ThreatGapBefore / maxHp) * EnvRewardMissedDefensePenaltyWeight;
    }

    private static void ApplyEndTurnThreatShaping(
        BridgeEnvSnapshot before,
        BridgeEnvActionShaping shaping)
    {
        if (!before.CombatInProgress || !HasEnvAvailableDefenseOption(before))
        {
            return;
        }

        var threatGapBefore = GetEnvThreatGap(before);
        if (threatGapBefore <= 0)
        {
            return;
        }

        var maxHp = Math.Max(before.MaxHp, 1);
        shaping.ThreatGapBefore = Math.Max(shaping.ThreatGapBefore, threatGapBefore);
        shaping.MissedDefensePenalty =
            Math.Min(
                shaping.MissedDefensePenalty,
                ((double)threatGapBefore / maxHp) * EnvRewardMissedDefensePenaltyWeight);
    }

    private static int GetEnvThreatGap(BridgeEnvSnapshot snapshot)
    {
        return Math.Max(0, GetEnvTotalIncomingDamageToPlayer(snapshot) - Math.Max(0, snapshot.PlayerBlock));
    }

    private static bool HasEnvAvailableDefenseOption(BridgeEnvSnapshot snapshot)
    {
        if (!snapshot.CombatInProgress || snapshot.CurrentEnergy <= 0 || GetEnvThreatGap(snapshot) <= 0)
        {
            return false;
        }

        return snapshot.ResolvedActions.Any(action => IsEnvDefensiveAction(snapshot, action));
    }

    private static bool IsEnvDefensiveAction(BridgeEnvSnapshot snapshot, BridgeResolvedAction action)
    {
        var payload = JsonSerializer.SerializeToElement(action.Payload);
        var kind = TryGetNestedString(payload, "kind") ?? InferEnvActionKind(action.ActionId);
        if (!string.Equals(kind, "play_card", StringComparison.Ordinal) &&
            !string.Equals(kind, "use_potion", StringComparison.Ordinal))
        {
            return false;
        }

        var source = string.Equals(kind, "play_card", StringComparison.Ordinal)
            ? TryGetNestedElement(payload, "card")
            : TryGetNestedElement(payload, "potion");
        if (source is null || source.Value.ValueKind is JsonValueKind.Null or JsonValueKind.Undefined)
        {
            return false;
        }

        var block = TryGetNestedInt(source.Value, "effect_preview", "total_block") ??
                    TryExtractEnvMetric(source.Value, "block");
        if (block > 0)
        {
            return true;
        }

        var weak = TryGetNestedInt(source.Value, "effect_preview", "weak") ??
                   TryExtractEnvMetric(source.Value, "weak");
        if (weak <= 0)
        {
            return false;
        }

        var targetCombatId = TryGetNestedInt(payload, "target_combat_id");
        if (targetCombatId.HasValue)
        {
            return snapshot.EnemyStates.Any(enemy =>
                enemy.CombatId == (uint)targetCombatId.Value &&
                enemy.IsAlive &&
                enemy.IntentDamageToPlayer > 0);
        }

        return snapshot.EnemyStates.Any(enemy => enemy.IsAlive && enemy.IntentDamageToPlayer > 0);
    }

    private static bool HasEnvWastedEndTurn(BridgeEnvSnapshot snapshot)
    {
        if (!snapshot.CombatInProgress || !string.Equals(snapshot.Phase, "combat", StringComparison.Ordinal))
        {
            return false;
        }

        if (snapshot.CurrentEnergy <= 0)
        {
            return false;
        }

        return snapshot.ResolvedActions.Any(action =>
            !string.Equals(action.ActionId, "end_turn", StringComparison.Ordinal) &&
            (action.ActionId.StartsWith("play_card:", StringComparison.Ordinal) ||
             action.ActionId.StartsWith("use_potion:", StringComparison.Ordinal)));
    }

    private static object? BuildEnvActionDiagnostics(
        BridgeEnvSnapshot before,
        BridgeResolvedActionSelection? selectedAction,
        string? actionError)
    {
        if (selectedAction is null || !string.IsNullOrWhiteSpace(actionError))
        {
            return null;
        }

        var endTurnSelected =
            string.Equals(selectedAction.Kind, "combat", StringComparison.Ordinal) &&
            string.Equals(selectedAction.Action.ActionId, "end_turn", StringComparison.Ordinal);

        if (!before.CombatInProgress)
        {
            return new
            {
                end_turn_selected = endTurnSelected
            };
        }

        var nonEndActionCount = 0;
        var playCardActionCount = 0;
        var zeroCostPlayCardCount = 0;
        var positivePreviewActionCount = 0;
        var selfHpLossActionCount = 0;

        foreach (var action in before.ResolvedActions)
        {
            if (string.Equals(action.ActionId, "end_turn", StringComparison.Ordinal))
            {
                continue;
            }

            var payload = JsonSerializer.SerializeToElement(action.Payload);
            var kind = TryGetNestedString(payload, "kind") ?? InferEnvActionKind(action.ActionId);
            if (!string.Equals(kind, "play_card", StringComparison.Ordinal) &&
                !string.Equals(kind, "use_potion", StringComparison.Ordinal))
            {
                continue;
            }

            nonEndActionCount += 1;
            if (string.Equals(kind, "play_card", StringComparison.Ordinal))
            {
                playCardActionCount += 1;
            }

            var source = string.Equals(kind, "play_card", StringComparison.Ordinal)
                ? TryGetNestedElement(payload, "card")
                : TryGetNestedElement(payload, "potion");
            if (source is null || source.Value.ValueKind is JsonValueKind.Null or JsonValueKind.Undefined)
            {
                continue;
            }

            var energyCost = TryGetNestedInt(source.Value, "resolved_energy_cost") ??
                             TryGetNestedInt(source.Value, "cost") ??
                             0;
            if (string.Equals(kind, "play_card", StringComparison.Ordinal) && energyCost == 0)
            {
                zeroCostPlayCardCount += 1;
            }

            if (HasPositivePreview(source.Value))
            {
                positivePreviewActionCount += 1;
            }

            if (GetPreviewMetric(source.Value, "hp_loss") > 0)
            {
                selfHpLossActionCount += 1;
            }
        }

        return new
        {
            end_turn_selected = endTurnSelected,
            end_turn_wasted = endTurnSelected && HasEnvWastedEndTurn(before),
            non_end_action_count = nonEndActionCount,
            play_card_action_count = playCardActionCount,
            zero_cost_play_card_count = zeroCostPlayCardCount,
            positive_preview_action_count = positivePreviewActionCount,
            self_hp_loss_action_count = selfHpLossActionCount
        };
    }

    private static bool ShouldRewardSkippingBadCardReward(BridgeEnvSnapshot snapshot)
    {
        var candidateScores = snapshot.ResolvedActions
            .Where(static action => action.ActionId.StartsWith("card_reward:", StringComparison.Ordinal) &&
                                    !string.Equals(action.ActionId, "card_reward:skip", StringComparison.Ordinal))
            .Select(action => ScoreEnvCardHeuristic(
                TryGetNestedElement(JsonSerializer.SerializeToElement(action.Payload), "card"),
                snapshot))
            .ToArray();

        return candidateScores.Length > 0 && candidateScores.All(static score => score <= 0d);
    }

    private static void ApplyRestSiteShaping(
        BridgeEnvSnapshot before,
        JsonElement payload,
        BridgeEnvActionShaping shaping)
    {
        var hpRatio = before.MaxHp > 0 ? (double)before.CurrentHp / before.MaxHp : 0d;
        var optionType = TryGetNestedString(payload, "option", "option_type") ?? string.Empty;
        if (optionType.Contains("Heal", StringComparison.OrdinalIgnoreCase))
        {
            if (hpRatio < 0.5d)
            {
                shaping.RestBonus = EnvRewardRestLowHpBonus;
            }
            else if (hpRatio > 0.7d)
            {
                shaping.RestMismatchPenalty = EnvRewardRestHighHpMismatchPenalty;
            }

            return;
        }

        if (optionType.Contains("Smith", StringComparison.OrdinalIgnoreCase))
        {
            if (hpRatio > 0.7d)
            {
                shaping.SmithBonus = EnvRewardSmithHealthyBonus;
            }
            else if (hpRatio < 0.5d)
            {
                shaping.SmithMismatchPenalty = EnvRewardSmithLowHpMismatchPenalty;
            }
        }
    }

    private static double ScoreEnvCardHeuristic(JsonElement? cardElement, BridgeEnvSnapshot snapshot)
    {
        if (cardElement is null || cardElement.Value.ValueKind is JsonValueKind.Null or JsonValueKind.Undefined)
        {
            return 0d;
        }

        var type = TryGetNestedString(cardElement.Value, "type") ?? string.Empty;
        var cost = TryGetNestedInt(cardElement.Value, "resolved_energy_cost") ??
                   TryGetNestedInt(cardElement.Value, "cost") ??
                   0;
        var damage = TryGetNestedInt(cardElement.Value, "effect_preview", "total_damage") ??
                     TryExtractEnvMetric(cardElement.Value, "damage");
        var block = TryGetNestedInt(cardElement.Value, "effect_preview", "total_block") ??
                    TryExtractEnvMetric(cardElement.Value, "block");
        var draw = TryGetNestedInt(cardElement.Value, "effect_preview", "draw") ??
                   TryExtractEnvMetric(cardElement.Value, "draw");
        var weak = TryGetNestedInt(cardElement.Value, "effect_preview", "weak") ??
                   TryExtractEnvMetric(cardElement.Value, "weak");
        var vulnerable = TryGetNestedInt(cardElement.Value, "effect_preview", "vulnerable") ??
                         TryExtractEnvMetric(cardElement.Value, "vulnerable");
        var summon = TryGetNestedInt(cardElement.Value, "effect_preview", "summon") ??
                     TryExtractEnvMetric(cardElement.Value, "summon");
        var summary = (TryGetNestedString(cardElement.Value, "effect_preview", "summary") ??
                       TryGetNestedString(cardElement.Value, "effect") ??
                       TryGetNestedString(cardElement.Value, "description") ??
                       string.Empty)
            .ToLowerInvariant();

        var score = 0d;

        if (string.Equals(type, "Attack", StringComparison.OrdinalIgnoreCase))
        {
            if (cost <= 1 && damage >= 8)
            {
                score += 0.04d;
            }
            else if (cost <= 1 && damage >= 6)
            {
                score += 0.02d;
            }
            else if (cost >= 2 && damage > 0 && damage < cost * 7)
            {
                score -= 0.04d;
            }
        }

        if (string.Equals(type, "Skill", StringComparison.OrdinalIgnoreCase))
        {
            if (cost <= 1 && block >= 7)
            {
                score += 0.03d;
            }
            else if (cost <= 1 && block >= 5)
            {
                score += 0.02d;
            }
            else if (cost >= 2 && block > 0 && block < cost * 6)
            {
                score -= 0.03d;
            }
        }

        if (cost == 0 && (damage > 0 || block > 0 || draw > 0))
        {
            score += 0.02d;
        }

        if (draw >= 2)
        {
            score += 0.04d;
        }
        else if (draw == 1)
        {
            score += 0.02d;
        }

        if (weak > 0)
        {
            score += 0.02d;
        }

        if (vulnerable > 0)
        {
            score += 0.02d;
        }

        if (summon >= 5)
        {
            score += 0.02d;
        }

        if (string.Equals(type, "Power", StringComparison.OrdinalIgnoreCase) &&
            snapshot.ActIndex <= 0 &&
            snapshot.TotalFloor <= 8)
        {
            score -= 0.04d;
        }

        if (cost >= 3)
        {
            score -= 0.03d;
        }

        if (snapshot.DeckCount >= 20)
        {
            score -= 0.02d;
        }

        if (damage <= 0 &&
            block <= 0 &&
            draw <= 0 &&
            weak <= 0 &&
            vulnerable <= 0 &&
            summon <= 0 &&
            cost >= 1 &&
            !summary.Contains("energy", StringComparison.Ordinal) &&
            !summary.Contains("能量", StringComparison.Ordinal))
        {
            score -= 0.05d;
        }

        return RoundEnvNumber(Math.Clamp(score, -EnvRewardCardHeuristicLimit, EnvRewardCardHeuristicLimit));
    }

    private static int TryExtractEnvMetric(JsonElement cardElement, string metric)
    {
        var text = TryGetNestedString(cardElement, "effect_preview", "summary") ??
                   TryGetNestedString(cardElement, "effect") ??
                   TryGetNestedString(cardElement, "description") ??
                   string.Empty;
        if (string.IsNullOrWhiteSpace(text))
        {
            return 0;
        }

        var lower = text.ToLowerInvariant();
        return metric switch
        {
            "damage" => TryExtractEnvMetricWithPatterns(lower, text, @"(\d+)\s*damage", @"造成(\d+)点伤害"),
            "block" => TryExtractEnvMetricWithPatterns(lower, text, @"(\d+)\s*block", @"获得(\d+)点格挡"),
            "draw" => TryExtractEnvMetricWithPatterns(lower, text, @"draw\s*(\d+)", @"抽(\d+)张牌"),
            "weak" => TryExtractEnvMetricWithPatterns(lower, text, @"(\d+)\s*weak", @"给予(\d+)层虚弱"),
            "vulnerable" => TryExtractEnvMetricWithPatterns(lower, text, @"(\d+)\s*vulnerable", @"给予(\d+)层易伤"),
            "heal" => TryExtractEnvMetricWithPatterns(lower, text, @"heal\s*(\d+)", @"(?:回复|恢复)(\d+)点生命"),
            "hp_loss" => TryExtractEnvMetricWithPatterns(lower, text, @"lose\s*(\d+)\s*hp", @"失去(\d+)点生命"),
            "strength" => TryExtractEnvMetricWithPatterns(lower, text, @"(\d+)\s*strength", @"(?:获得|给予)(\d+)点力量"),
            "dexterity" => TryExtractEnvMetricWithPatterns(lower, text, @"(\d+)\s*dexterity", @"(?:获得|给予)(\d+)点敏捷"),
            "summon" => TryExtractEnvMetricWithPatterns(lower, text, @"summon\s*(\d+)", @"召唤(\d+)"),
            _ => 0
        };
    }

    private static int GetPreviewMetric(JsonElement element, string metric)
    {
        return metric switch
        {
            "damage" => TryGetNestedInt(element, "effect_preview", "total_damage") ?? TryGetNestedInt(element, "damage") ?? TryExtractEnvMetric(element, "damage"),
            "block" => TryGetNestedInt(element, "effect_preview", "total_block") ?? TryGetNestedInt(element, "block") ?? TryExtractEnvMetric(element, "block"),
            "draw" => TryGetNestedInt(element, "effect_preview", "draw") ?? TryGetNestedInt(element, "draw") ?? TryExtractEnvMetric(element, "draw"),
            "weak" => TryGetNestedInt(element, "effect_preview", "weak") ?? TryGetNestedInt(element, "weak") ?? TryExtractEnvMetric(element, "weak"),
            "vulnerable" => TryGetNestedInt(element, "effect_preview", "vulnerable") ?? TryGetNestedInt(element, "vulnerable") ?? TryExtractEnvMetric(element, "vulnerable"),
            "heal" => TryGetNestedInt(element, "effect_preview", "heal") ?? TryGetNestedInt(element, "heal") ?? TryExtractEnvMetric(element, "heal"),
            "hp_loss" => TryGetNestedInt(element, "effect_preview", "hp_loss") ?? TryGetNestedInt(element, "hp_loss") ?? TryExtractEnvMetric(element, "hp_loss"),
            "strength" => TryGetNestedInt(element, "effect_preview", "strength") ?? TryGetNestedInt(element, "strength") ?? TryExtractEnvMetric(element, "strength"),
            "dexterity" => TryGetNestedInt(element, "effect_preview", "dexterity") ?? TryGetNestedInt(element, "dexterity") ?? TryExtractEnvMetric(element, "dexterity"),
            "summon" => TryGetNestedInt(element, "effect_preview", "summon") ?? TryGetNestedInt(element, "summon") ?? TryExtractEnvMetric(element, "summon"),
            _ => 0
        };
    }

    private static bool HasPositivePreview(JsonElement element)
    {
        return GetPreviewMetric(element, "damage") > 0 ||
               GetPreviewMetric(element, "block") > 0 ||
               GetPreviewMetric(element, "draw") > 0 ||
               GetPreviewMetric(element, "weak") > 0 ||
               GetPreviewMetric(element, "vulnerable") > 0 ||
               GetPreviewMetric(element, "heal") > 0 ||
               GetPreviewMetric(element, "strength") > 0 ||
               GetPreviewMetric(element, "dexterity") > 0 ||
               GetPreviewMetric(element, "summon") > 0;
    }

    private static int TryExtractEnvMetricWithPatterns(string normalized, string original, string englishPattern, string chinesePattern)
    {
        foreach (var pattern in new[] { englishPattern, chinesePattern })
        {
            var input = pattern == englishPattern ? normalized : original;
            var match = Regex.Match(input, pattern, RegexOptions.IgnoreCase);
            if (match.Success && int.TryParse(match.Groups[1].Value, out var value))
            {
                return value;
            }
        }

        return 0;
    }

}
