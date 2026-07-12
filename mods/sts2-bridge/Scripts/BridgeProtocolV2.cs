using System.Text.Json;
using System.Text.Json.Serialization;
using Sts2.Contracts.Generated;

namespace Sts2McpBridge.Scripts;

internal static class BridgeProtocolV2
{
    public const string ApiVersion = ContractVersions.ApiVersion;
    public const string SchemaVersion = ContractVersions.SchemaVersion;
    public const string PlayerControlCapability = "player-control";
    public const string TrainingCapability = "training";
    public const string LegacyPrivilegedCapability = "legacy-privileged";
    public const string PerformActionCommand = "perform_action";
}

internal sealed class BridgeCommandEnvelopeV2
{
    [JsonPropertyName("request_id")]
    public string? RequestId { get; set; }

    [JsonPropertyName("session_id")]
    public string? SessionId { get; set; }

    [JsonPropertyName("capability")]
    public string? Capability { get; set; }

    [JsonPropertyName("expected_state_version")]
    public long? ExpectedStateVersion { get; set; }

    [JsonPropertyName("deadline_utc")]
    public DateTimeOffset? DeadlineUtc { get; set; }

    [JsonPropertyName("command")]
    public BridgePlayerCommandV2? Command { get; set; }
}

internal sealed class BridgePlayerCommandV2
{
    [JsonPropertyName("kind")]
    public string? Kind { get; set; }

    [JsonPropertyName("action_handle")]
    public string? ActionHandle { get; set; }

    [JsonPropertyName("wait_after_ms")]
    public int? WaitAfterMs { get; set; }
}

internal sealed class BridgeCommandErrorV2
{
    [JsonPropertyName("code")]
    public required string Code { get; init; }

    [JsonPropertyName("message")]
    public required string Message { get; init; }

    [JsonPropertyName("details")]
    public object? Details { get; init; }
}

internal sealed class BridgeCommandResultV2
{
    [JsonPropertyName("ok")]
    public required bool Ok { get; init; }

    [JsonPropertyName("api_version")]
    public string ApiVersion { get; init; } = BridgeProtocolV2.ApiVersion;

    [JsonPropertyName("schema_version")]
    public string SchemaVersion { get; init; } = BridgeProtocolV2.SchemaVersion;

    [JsonPropertyName("request_id")]
    public required string RequestId { get; init; }

    [JsonPropertyName("status")]
    public required string Status { get; init; }

    [JsonPropertyName("replayed_result")]
    public required bool ReplayedResult { get; init; }

    [JsonPropertyName("accepted_at_utc")]
    public required DateTimeOffset AcceptedAtUtc { get; init; }

    [JsonPropertyName("started_at_utc")]
    public DateTimeOffset? StartedAtUtc { get; init; }

    [JsonPropertyName("completed_at_utc")]
    public DateTimeOffset? CompletedAtUtc { get; init; }

    [JsonPropertyName("committed_state_version")]
    public long? CommittedStateVersion { get; init; }

    [JsonPropertyName("error_code")]
    public string? ErrorCode { get; init; }

    [JsonPropertyName("error_message")]
    public string? ErrorMessage { get; init; }

    [JsonPropertyName("result")]
    public object? Result { get; init; }

    [JsonPropertyName("error")]
    public BridgeCommandErrorV2? Error { get; init; }
}

internal sealed class BridgeEnvResetEnvelopeV2
{
    [JsonPropertyName("request_id")]
    public string? RequestId { get; set; }

    [JsonPropertyName("session_id")]
    public string? SessionId { get; set; }

    [JsonPropertyName("scenario")]
    public string? Scenario { get; set; }

    [JsonPropertyName("expected_state_version")]
    public long? ExpectedStateVersion { get; set; }

    [JsonPropertyName("seed")]
    public JsonElement? Seed { get; set; }

    [JsonPropertyName("options")]
    public JsonElement? Options { get; set; }
}

internal sealed class BridgeEnvStepEnvelopeV2
{
    [JsonPropertyName("request_id")]
    public string? RequestId { get; set; }

    [JsonPropertyName("session_id")]
    public string? SessionId { get; set; }

    [JsonPropertyName("episode_id")]
    public string? EpisodeId { get; set; }

    [JsonPropertyName("expected_step_index")]
    public int? ExpectedStepIndex { get; set; }

    [JsonPropertyName("action")]
    public JsonElement Action { get; set; }
}

internal sealed class BridgeActionOutcomeUnknownException : Exception
{
    public BridgeActionOutcomeUnknownException(string actionId, Exception innerException)
        : base($"Action '{actionId}' may have started, but its outcome could not be confirmed.", innerException)
    {
        ActionId = actionId;
    }

    public string ActionId { get; }
}
