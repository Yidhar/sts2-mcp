using System.Collections.Generic;

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
        public required string RunResult { get; init; }
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

    private sealed class BridgeEnvDeckEntry
    {
        public required string Ref { get; init; }
        public string? CardId { get; init; }
        public string? Title { get; init; }
    }
}
