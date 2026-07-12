using System.Globalization;
using System.Net;
using System.Reflection;
using System.Text.Json.Serialization;
using System.ComponentModel;
using System.Diagnostics;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Entities.Multiplayer;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Map;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Models.Characters;
using MegaCrit.Sts2.Core.Multiplayer.Game;
using MegaCrit.Sts2.Core.Nodes;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Saves;

namespace Sts2McpBridge.Scripts;

internal static partial class BridgeGameApi
{
    public static async Task<object> GetCombatCatalogResponseAsync(CancellationToken cancellationToken)
    {
        await WaitForEnvDispatcherReadyAsync(5000, cancellationToken);

        var encounters = await RunOnMainThreadGuardedAsync(
            ListAvailableCombatEncounters,
            "combat_sandbox.list_catalog",
            5000,
            cancellationToken);

        return new
        {
            ok = true,
            encounters
        };
    }

    // -----------------------------------------------------------------------
    // Combat sandbox done detection
    // -----------------------------------------------------------------------

    private static bool IsCombatSandboxEpisodeDone(BridgeEnvSnapshot snapshot)
    {
        return IsCombatSandboxExplicitTerminalSurface(snapshot);
    }

    // -----------------------------------------------------------------------
    // Setup internals
    // -----------------------------------------------------------------------

    private sealed class CombatSandboxSetupResult
    {
        public bool Success { get; init; }
        public string? ErrorCode { get; init; }
        public string? ErrorMessage { get; init; }
        public Task? PendingTask { get; init; }
    }

    private sealed class EncounterCatalogEntry
    {
        [JsonPropertyName("encounter_id")]
        public string EncounterId { get; init; } = string.Empty;

        [JsonPropertyName("display_name")]
        public string DisplayName { get; init; } = string.Empty;

        [JsonPropertyName("type_name")]
        public string TypeName { get; init; } = string.Empty;

        [JsonPropertyName("category")]
        public string Category { get; init; } = string.Empty;

        [JsonPropertyName("is_mock")]
        public bool IsMock { get; init; }
    }
}
