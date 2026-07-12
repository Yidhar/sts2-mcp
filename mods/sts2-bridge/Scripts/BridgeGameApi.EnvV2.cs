using System.Net;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace Sts2McpBridge.Scripts;

/// <summary>Training-scoped contract-v2 reads and revision validation.</summary>
internal static partial class BridgeGameApi
{
    public static object GetEnvSpecV2Response()
    {
        var spec = JsonSerializer.SerializeToNode(GetEnvSpecResponse()) as JsonObject ?? new JsonObject();
        spec["ok"] = true;
        spec["api_version"] = BridgeProtocolV2.ApiVersion;
        spec["schema_version"] = BridgeProtocolV2.SchemaVersion;
        spec["capability"] = BridgeProtocolV2.TrainingCapability;
        spec["scenarios"] = new JsonArray("full-run", "combat");
        spec["reward"] = new JsonObject
        {
            ["authority"] = "external-rl",
            ["scalar_computed_by_bridge"] = false,
            ["output"] = "transition_facts"
        };
        spec["action_encoding"] = new JsonObject
        {
            ["default_encoding"] = "legal_action_idx",
            ["supported"] = new JsonArray("legal_action_idx", "action_handle"),
            ["legal_action_shape"] = new JsonArray("idx", "action_handle", "kind")
        };
        return spec;
    }

    public static async Task<object> GetEnvStateV2ResponseAsync(
        CancellationToken cancellationToken = default)
    {
        EnsureDispatcherReady();
        var frontier = await ObserveFrontierAsync(cancellationToken);
        var snapshot = await CaptureEnvSnapshotAsync(
            DefaultEnvStepTimeoutMs,
            cancellationToken,
            "v2.env.state");

        string? episodeId;
        int? stepIndex;
        string? scenario;
        bool episodeDone;
        lock (EnvEpisodeSync)
        {
            episodeId = _activeEnvEpisode?.Id;
            stepIndex = _activeEnvEpisode?.StepIndex;
            scenario = _activeEnvEpisode?.EpisodeMode switch
            {
                "full_run" => "full-run",
                "combat_sandbox" => "combat",
                { } value => value,
                _ => null
            };
            episodeDone = _activeEnvEpisode?.Done == true;
        }

        return new
        {
            ok = true,
            api_version = BridgeProtocolV2.ApiVersion,
            schema_version = BridgeProtocolV2.SchemaVersion,
            capability = BridgeProtocolV2.TrainingCapability,
            state_version = frontier.Sequence,
            episode_id = episodeId,
            step_index = stepIndex,
            scenario,
            phase = snapshot.Phase,
            screen = snapshot.Screen,
            actionable = snapshot.Actionable,
            terminated = snapshot.Done || episodeDone,
            observation = snapshot.Observation,
            legal_actions = BridgeLegalActionProjector.ProjectEnvironmentActions(snapshot.LegalActions),
            captured_at_utc = DateTimeOffset.UtcNow
        };
    }

    public static async Task<object> GetCombatCatalogV2ResponseAsync(
        CancellationToken cancellationToken = default)
    {
        var catalog = JsonSerializer.SerializeToNode(
                          await GetCombatCatalogResponseAsync(cancellationToken)) as JsonObject ??
                      new JsonObject();
        catalog["ok"] = true;
        catalog["api_version"] = BridgeProtocolV2.ApiVersion;
        catalog["schema_version"] = BridgeProtocolV2.SchemaVersion;
        catalog["capability"] = BridgeProtocolV2.TrainingCapability;
        return catalog;
    }

    internal static async Task<long> GetCurrentStateVersionV2Async(
        CancellationToken cancellationToken)
    {
        EnsureDispatcherReady();
        var frontier = await ObserveFrontierAsync(cancellationToken);
        return frontier.Sequence;
    }

    internal static async Task<long> ValidateEnvironmentResetStateVersionV2Async(
        long expectedStateVersion,
        CancellationToken cancellationToken)
    {
        EnsureDispatcherReady();
        return await RunOnMainThreadGuardedAsync(
            () =>
            {
                var current = PublishFrontier(CaptureSnapshot());
                if (current.Sequence != expectedStateVersion)
                {
                    throw new BridgeRequestException(
                        HttpStatusCode.Conflict,
                        "state_version_conflict",
                        $"Expected state_version {expectedStateVersion}, but the current state_version is {current.Sequence}.",
                        BuildSafeV2ConflictDetails(current, expectedStateVersion, actionHandle: null));
                }

                return current.Sequence;
            },
            "v2.env.reset.validate_revision",
            DefaultMainThreadTaskTimeoutMs,
            cancellationToken);
    }
}
