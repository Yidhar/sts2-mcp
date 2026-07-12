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
    private sealed class BridgeEnvEpisode
    {
        public required string Id { get; init; }
        public int StepIndex { get; set; }
        public bool Done { get; set; }
        public string? RequestedCharacter { get; init; }
        public bool DefensiveBuffs { get; init; }
        public string EpisodeMode { get; init; } = "full_run";
        public string? EncounterId { get; init; }
        public bool RoomAnchorInitialized { get; set; }
        public string RoomKey { get; set; } = string.Empty;
        public int RoomStartHp { get; set; }
        public int RoomStartMaxHp { get; set; }
        public int RoomStartFloor { get; set; }
        public int RoomStartActIndex { get; set; }
    }

    private sealed class BridgeEnvSnapshot
    {
        public required BridgeWorldContext Context { get; init; }
        public required string Screen { get; init; }
        public required string Phase { get; init; }
        public required object Observation { get; init; }
        public required object RunSummary { get; init; }
        public required object[] LegalActions { get; init; }
        public required IReadOnlyDictionary<string, BridgeResolvedAction> ActionLookup { get; init; }
        public required IReadOnlyList<BridgeResolvedAction> ResolvedActions { get; init; }
        public required string LogicHash { get; init; }
        public required string SurfaceFingerprint { get; init; }
        public required bool Actionable { get; init; }
        public required bool Done { get; init; }
        public required int CurrentHp { get; init; }
        public required int MaxHp { get; init; }
        public required int PlayerBlock { get; init; }
        public required int CurrentEnergy { get; init; }
        public required int Gold { get; init; }
        public required int ActIndex { get; init; }
        public required int TotalFloor { get; init; }
        public string? RoomType { get; init; }
        public string? RoomModelId { get; init; }
        public required int RelicCount { get; init; }
        public required int PotionCount { get; init; }
        public required int DeckCount { get; init; }
        public required IReadOnlyList<BridgeEnvDeckEntry> DeckEntries { get; init; }
        public required IReadOnlyList<BridgeEnvEnemyState> EnemyStates { get; init; }
        public required bool CombatInProgress { get; init; }
        public required bool RoomPreFinished { get; init; }
        public bool RunActive => Context.RunState is not null && Context.RunState.IsGameOver != true;
    }

    private sealed class BridgeEnvEnemyState
    {
        public required uint CombatId { get; init; }
        public required int CurrentHp { get; init; }
        public required int Block { get; init; }
        public required bool IsAlive { get; init; }
        public required int IntentDamageToPlayer { get; init; }
        public required int Weak { get; init; }
        public required int Vulnerable { get; init; }
    }

    private sealed class BridgeResolvedActionSelection
    {
        public required BridgeResolvedAction Action { get; init; }
        public required int Index { get; init; }
        public required string Kind { get; init; }
    }

    private sealed class BridgeLegacyEnvRewardBreakdown
    {
        [JsonPropertyName("hp_loss_normalized")]
        public required double HpLossNormalized { get; init; }

        [JsonPropertyName("hp_gain_normalized")]
        public required double HpGainNormalized { get; init; }

        [JsonPropertyName("room_complete")]
        public required int RoomComplete { get; init; }

        [JsonPropertyName("combat_room_complete")]
        public required int CombatRoomComplete { get; init; }

        [JsonPropertyName("room_hp_delta_normalized")]
        public required double RoomHpDeltaNormalized { get; init; }

        [JsonPropertyName("combat_room_complete_bonus")]
        public required double CombatRoomCompleteBonus { get; init; }

        [JsonPropertyName("combat_room_quality_bonus")]
        public required double CombatRoomQualityBonus { get; init; }

        [JsonPropertyName("floor_delta")]
        public required int FloorDelta { get; init; }

        [JsonPropertyName("floor_progress_bonus")]
        public required double FloorProgressBonus { get; init; }

        [JsonPropertyName("act_clear")]
        public required int ActClear { get; init; }

        [JsonPropertyName("act_clear_bonus")]
        public required double ActClearBonus { get; init; }

        [JsonPropertyName("elite_clear_bonus")]
        public required double EliteClearBonus { get; init; }

        [JsonPropertyName("boss_clear_bonus")]
        public required double BossClearBonus { get; init; }

        [JsonPropertyName("relic_gain_count")]
        public required int RelicGainCount { get; init; }

        [JsonPropertyName("relic_gain_bonus")]
        public required double RelicGainBonus { get; init; }

        [JsonPropertyName("max_hp_gain_normalized")]
        public required double MaxHpGainNormalized { get; init; }

        [JsonPropertyName("max_hp_gain_bonus")]
        public required double MaxHpGainBonus { get; init; }

        [JsonPropertyName("death")]
        public required int Death { get; init; }

        [JsonPropertyName("victory")]
        public required int Victory { get; init; }

        [JsonPropertyName("run_victory_bonus")]
        public required double RunVictoryBonus { get; init; }

        [JsonPropertyName("action_error_penalty")]
        public required double ActionErrorPenalty { get; init; }

        [JsonPropertyName("truncated_penalty")]
        public required double TruncatedPenalty { get; init; }

        [JsonPropertyName("raw_floor_delta")]
        public required int RawFloorDelta { get; init; }

        [JsonPropertyName("raw_act_clear")]
        public required int RawActClear { get; init; }

        [JsonPropertyName("raw_gold_gain")]
        public required int RawGoldGain { get; init; }

        [JsonPropertyName("raw_gold_spend")]
        public required int RawGoldSpend { get; init; }

        [JsonPropertyName("raw_relic_gain_count")]
        public required int RawRelicGainCount { get; init; }

        [JsonPropertyName("raw_room_hp_delta_normalized")]
        public required double RawRoomHpDeltaNormalized { get; init; }

        [JsonPropertyName("raw_max_hp_gain_normalized")]
        public required double RawMaxHpGainNormalized { get; init; }

        [JsonPropertyName("reward_anomaly_clamped")]
        public required bool RewardAnomalyClamped { get; init; }

        [JsonPropertyName("reward_anomaly_reasons")]
        public required string[] RewardAnomalyReasons { get; init; }

        [JsonPropertyName("total")]
        public required double Total { get; init; }
    }

    private sealed class BridgeEnvDeckEntry
    {
        public required string Ref { get; init; }
        public string? CardId { get; init; }
        public string? Title { get; init; }
        public string? Rarity { get; init; }
        public required string Signature { get; init; }
        public required bool IsStarter { get; init; }
    }

    private sealed class BridgeEnvDeckDiff
    {
        public required int CardAddCount { get; init; }
        public required int StarterCardRemoveCount { get; init; }
        public required int OtherCardRemoveCount { get; init; }
        public required int CardUpgradeCount { get; init; }
    }

    private sealed class BridgeEnvActionShaping
    {
        public static BridgeEnvActionShaping None => new();

        public double CardChoiceBonus { get; set; }
        public double SkipBadCardsBonus { get; set; }
        public double RestBonus { get; set; }
        public double RestMismatchPenalty { get; set; }
        public double SmithBonus { get; set; }
        public double SmithMismatchPenalty { get; set; }
        public double PlayCardBonus { get; set; }
        public int ThreatGapBefore { get; set; }
        public int ThreatGapAfter { get; set; }
        public int ThreatGapReduction { get; set; }
        public double ThreatGapReductionNormalized { get; set; }
        public double ThreatGapReductionBonus { get; set; }
        public int EffectiveBlockAdded { get; set; }
        public double EffectiveBlockNormalized { get; set; }
        public double EffectiveBlockBonus { get; set; }
        public int WastedBlockAdded { get; set; }
        public double WastedBlockNormalized { get; set; }
        public double WastedBlockPenalty { get; set; }
        public int WeakIntentReduction { get; set; }
        public double WeakIntentReductionNormalized { get; set; }
        public double WeakBonus { get; set; }
        public int VulnerableRealizedDamage { get; set; }
        public double VulnerableRealizedDamageNormalized { get; set; }
        public double VulnerableBonus { get; set; }
        public double EndTurnWastePenalty { get; set; }
        public double MissedDefensePenalty { get; set; }
    }
}
