using System.Net;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace Sts2McpBridge.Scripts;

internal static class BridgeCommandService
{
    private const int MaxRequestIdLength = 64;
    private const int MaxActionHandleLength = 512;
    private const int MaxWaitAfterMilliseconds = 5000;
    private static readonly TimeSpan MaxPlayerCommandDeadlineWindow = TimeSpan.FromMinutes(2);
    private static readonly JsonSerializerOptions EnvironmentJsonOptions = new()
    {
        PropertyNameCaseInsensitive = true
    };
    private static readonly BoundedCommandResultStore Store =
        new(2048, TimeSpan.FromMinutes(10));

    public static async Task<BridgeCommandResultV2> SubmitAsync(
        BridgeCommandEnvelopeV2? envelope,
        CancellationToken requestCancellationToken,
        CancellationToken bridgeShutdownToken)
    {
        envelope ??= new BridgeCommandEnvelopeV2();
        NormalizeAndValidatePlayerCommand(envelope);
        var fingerprint = ComputeFingerprint("player-command", new
        {
            envelope.SessionId,
            envelope.Capability,
            envelope.ExpectedStateVersion,
            deadline_utc = envelope.DeadlineUtc,
            kind = envelope.Command!.Kind,
            action_handle = envelope.Command.ActionHandle,
            wait_after_ms = envelope.Command.WaitAfterMs
        });

        return await BeginOrReplayAsync(
            envelope.RequestId!,
            fingerprint,
            BridgeProtocolV2.PlayerControlCapability,
            "player-command",
            entry => ExecuteAcceptedPlayerCommandAsync(entry, envelope, bridgeShutdownToken),
            requestCancellationToken);
    }

    public static async Task<BridgeCommandResultV2> SubmitEnvironmentResetAsync(
        BridgeEnvResetEnvelopeV2? envelope,
        CancellationToken requestCancellationToken,
        CancellationToken bridgeShutdownToken)
    {
        envelope ??= new BridgeEnvResetEnvelopeV2();
        envelope.RequestId = NormalizeRequestId(envelope.RequestId);
        ValidateSessionId(envelope.SessionId);
        envelope.SessionId = envelope.SessionId!.Trim();
        envelope.Scenario = envelope.Scenario?.Trim();
        if (envelope.Scenario is not ("full-run" or "combat"))
        {
            throw new BridgeRequestException(
                HttpStatusCode.BadRequest,
                "invalid_scenario",
                "scenario must be either 'full-run' or 'combat'.");
        }

        if (envelope.ExpectedStateVersion is not long expectedStateVersion || expectedStateVersion < 0)
        {
            throw new BridgeRequestException(
                HttpStatusCode.BadRequest,
                "missing_expected_state_version",
                "A non-negative expected_state_version is required for every v2 reset mutation.");
        }

        ValidateOptionalObject(envelope.Options, "options");
        ValidateOptionalSeed(envelope.Seed);
        var fingerprint = ComputeFingerprint("environment-reset", new
        {
            envelope.SessionId,
            envelope.Scenario,
            envelope.ExpectedStateVersion,
            seed = envelope.Seed,
            options = envelope.Options
        });

        return await BeginOrReplayAsync(
            envelope.RequestId,
            fingerprint,
            BridgeProtocolV2.TrainingCapability,
            "environment-reset",
            entry => ExecuteAcceptedEnvironmentOperationAsync(
                entry,
                $"v2.env.reset:{envelope.Scenario}:{entry.RequestId}",
                token => ResetEnvironmentV2Async(envelope, expectedStateVersion, token),
                preflight: async token =>
                {
                    await BridgeGameApi.ValidateEnvironmentResetStateVersionV2Async(expectedStateVersion, token);
                },
                bridgeShutdownToken),
            requestCancellationToken);
    }

    public static async Task<BridgeCommandResultV2> SubmitEnvironmentStepAsync(
        BridgeEnvStepEnvelopeV2? envelope,
        CancellationToken requestCancellationToken,
        CancellationToken bridgeShutdownToken)
    {
        envelope ??= new BridgeEnvStepEnvelopeV2();
        envelope.RequestId = NormalizeRequestId(envelope.RequestId);
        ValidateSessionId(envelope.SessionId);
        envelope.SessionId = envelope.SessionId!.Trim();
        envelope.EpisodeId = envelope.EpisodeId?.Trim();
        if (string.IsNullOrWhiteSpace(envelope.EpisodeId) || envelope.EpisodeId.Length > 128)
        {
            throw new BridgeRequestException(
                HttpStatusCode.BadRequest,
                "invalid_episode_id",
                "episode_id must be a non-empty string no longer than 128 characters.");
        }

        if (envelope.ExpectedStepIndex is not int expectedStepIndex || expectedStepIndex < 0)
        {
            throw new BridgeRequestException(
                HttpStatusCode.BadRequest,
                "invalid_expected_step_index",
                "expected_step_index must be a non-negative integer.");
        }

        if (envelope.Action.ValueKind != JsonValueKind.Object)
        {
            throw new BridgeRequestException(
                HttpStatusCode.BadRequest,
                "invalid_environment_action",
                "action must be a JSON object.");
        }
        var legacyRequest = BuildLegacyStepRequest(envelope);

        var fingerprint = ComputeFingerprint("environment-step", new
        {
            envelope.SessionId,
            envelope.EpisodeId,
            envelope.ExpectedStepIndex,
            action = envelope.Action
        });

        return await BeginOrReplayAsync(
            envelope.RequestId,
            fingerprint,
            BridgeProtocolV2.TrainingCapability,
            "environment-step",
            entry => ExecuteAcceptedEnvironmentOperationAsync(
                entry,
                $"v2.env.step:{envelope.EpisodeId}:{expectedStepIndex}:{entry.RequestId}",
                async token =>
                {
                    var beforeStateVersion = await BridgeGameApi.GetCurrentStateVersionV2Async(token);
                    var legacyPayload = await BridgeGameApi.StepEnvResponseAsync(legacyRequest, token);
                    var afterStateVersion = await BridgeGameApi.GetCurrentStateVersionV2Async(token);
                    return ProjectLegacyEnvironmentPayloadV2(
                        legacyPayload,
                        beforeStateVersion,
                        afterStateVersion);
                },
                preflight: token =>
                {
                    BridgeGameApi.ValidateEnvEpisodeStepV2(envelope.EpisodeId, expectedStepIndex);
                    return Task.CompletedTask;
                },
                bridgeShutdownToken),
            requestCancellationToken);
    }

    public static BridgeCommandResultV2 GetStatus(string requestId, string capability)
    {
        var normalizedRequestId = NormalizeRequestId(requestId);
        if (!Store.TryGet(normalizedRequestId, out var entry) || entry is null)
        {
            throw new BridgeRequestException(
                HttpStatusCode.NotFound,
                "command_not_found",
                $"No retained command result exists for request_id '{normalizedRequestId}'.",
                new { request_id = normalizedRequestId });
        }

        if (!string.Equals(entry.Capability, capability, StringComparison.Ordinal))
        {
            throw new BridgeRequestException(
                HttpStatusCode.Forbidden,
                "command_capability_mismatch",
                "The supplied token does not grant access to this command result.");
        }

        return WithReplayFlag(entry.Snapshot, replayedResult: true);
    }

    public static object GetDiagnosticsSnapshot() => Store.GetDiagnosticsSnapshot();

    private static async Task<BridgeCommandResultV2> BeginOrReplayAsync(
        string requestId,
        string fingerprint,
        string capability,
        string operationKind,
        Func<BridgeCommandStoreEntry, Task> startExecution,
        CancellationToken requestCancellationToken)
    {
        var disposition = Store.GetOrCreate(
            requestId,
            fingerprint,
            capability,
            operationKind,
            out var entry);
        switch (disposition)
        {
            case BridgeCommandStoreDisposition.FingerprintConflict:
                throw new BridgeRequestException(
                    HttpStatusCode.Conflict,
                    "request_id_reused",
                    "The request_id was already used with a different operation payload.",
                    new { request_id = requestId });
            case BridgeCommandStoreDisposition.CapacityExceeded:
                throw new BridgeRequestException(
                    HttpStatusCode.ServiceUnavailable,
                    "command_store_full",
                    "The bounded command result store is full of unexpired request identities. Try again after entries expire.");
            case BridgeCommandStoreDisposition.Created:
                _ = startExecution(entry!);
                break;
            case BridgeCommandStoreDisposition.Existing:
                break;
            default:
                throw new InvalidOperationException($"Unexpected command-store disposition: {disposition}.");
        }

        var result = await entry!.WaitAsync(requestCancellationToken);
        return WithReplayFlag(result, disposition == BridgeCommandStoreDisposition.Existing);
    }

    private static async Task ExecuteAcceptedPlayerCommandAsync(
        BridgeCommandStoreEntry entry,
        BridgeCommandEnvelopeV2 envelope,
        CancellationToken bridgeShutdownToken)
    {
        var executing = false;
        DateTimeOffset? startedAtUtc = null;
        BridgeCommandResultV2 result;
        try
        {
            // Deadline-token construction is deliberately inside the completion
            // boundary. Even a platform timer failure must transition the retained
            // identity out of `accepted` rather than leaking a permanent entry.
            using var deadlineCts = CreateDeadlineToken(
                envelope.DeadlineUtc!.Value,
                bridgeShutdownToken);
            if (deadlineCts.IsCancellationRequested)
            {
                result = BuildRejectedResult(
                    entry,
                    "deadline_expired",
                    "The command deadline expired before execution began.");
                Store.Complete(entry, result);
                return;
            }

            await using var lease = await BridgeMutationGate.AcquireAsync(
                $"v2:{envelope.Command!.Kind}:{entry.RequestId}",
                deadlineCts.Token);
            deadlineCts.Token.ThrowIfCancellationRequested();
            executing = true;
            startedAtUtc = DateTimeOffset.UtcNow;
            entry.MarkExecuting(startedAtUtc.Value);

            var payload = await BridgeGameApi.PerformActionV2AtomicAsync(
                envelope.Command!.ActionHandle!,
                envelope.ExpectedStateVersion!.Value,
                envelope.Command.WaitAfterMs,
                deadlineCts.Token);

            result = BuildCommittedResult(entry, startedAtUtc.Value, payload, "state_version_after");
        }
        catch (BridgeRequestException ex)
        {
            // A main-thread guard timeout can race with an already-started work
            // item. It is never safe to report that case as a clean rejection.
            var outcomeMayBeUnknown = string.Equals(
                ex.ErrorCode,
                "main_thread_stalled",
                StringComparison.Ordinal);
            result = BuildErrorResult(
                entry,
                outcomeMayBeUnknown ? "outcome_unknown" : "rejected_before_execution",
                outcomeMayBeUnknown ? startedAtUtc : null,
                ex.ErrorCode,
                ex.Message,
                ex.Details);
        }
        catch (BridgeActionOutcomeUnknownException ex)
        {
            result = BuildErrorResult(
                entry,
                "outcome_unknown",
                startedAtUtc,
                "action_outcome_unknown",
                ex.Message,
                new { action_handle = ex.ActionId });
        }
        catch (BridgeMutationGateBusyException ex)
        {
            result = BuildRejectedResult(entry, "mutation_queue_full", ex.Message);
        }
        catch (OperationCanceledException)
        {
            result = BuildCancellationResult(entry, startedAtUtc, executing, bridgeShutdownToken);
        }
        catch (Exception ex)
        {
            result = BuildErrorResult(
                entry,
                executing ? "outcome_unknown" : "rejected_before_execution",
                startedAtUtc,
                executing ? "action_outcome_unknown" : "command_failed_before_execution",
                executing
                    ? "The command may have started, but its final outcome could not be confirmed."
                    : "The command failed before execution began.");
            BridgeDebugTrace.Write($"v2 command {entry.RequestId} failed: {ex}");
        }

        Store.Complete(entry, result);
    }

    private static async Task ExecuteAcceptedEnvironmentOperationAsync(
        BridgeCommandStoreEntry entry,
        string operationName,
        Func<CancellationToken, Task<object>> operation,
        Func<CancellationToken, Task>? preflight,
        CancellationToken bridgeShutdownToken)
    {
        var executing = false;
        DateTimeOffset? startedAtUtc = null;
        BridgeCommandResultV2 result;
        try
        {
            await using var lease = await BridgeMutationGate.AcquireAsync(operationName, bridgeShutdownToken);
            bridgeShutdownToken.ThrowIfCancellationRequested();
            if (preflight is not null)
            {
                await preflight(bridgeShutdownToken);
            }
            executing = true;
            startedAtUtc = DateTimeOffset.UtcNow;
            entry.MarkExecuting(startedAtUtc.Value);
            var payload = await operation(bridgeShutdownToken);
            result = BuildCommittedResult(entry, startedAtUtc.Value, payload, "state_version_after");
        }
        catch (BridgeRequestException ex)
        {
            result = BuildErrorResult(
                entry,
                executing ? "outcome_unknown" : "rejected_before_execution",
                startedAtUtc,
                ex.ErrorCode,
                ex.Message,
                ProjectLegacyEnvironmentErrorDetailsV2(ex.Details));
        }
        catch (BridgeMutationGateBusyException ex)
        {
            result = BuildRejectedResult(entry, "mutation_queue_full", ex.Message);
        }
        catch (OperationCanceledException)
        {
            result = BuildCancellationResult(entry, startedAtUtc, executing, bridgeShutdownToken);
        }
        catch (Exception ex)
        {
            result = BuildErrorResult(
                entry,
                executing ? "outcome_unknown" : "rejected_before_execution",
                startedAtUtc,
                executing ? "environment_outcome_unknown" : "environment_failed_before_execution",
                executing
                    ? "The environment operation may have started, but its final outcome could not be confirmed."
                    : "The environment operation failed before execution began.");
            BridgeDebugTrace.Write($"v2 environment operation {entry.RequestId} failed: {ex}");
        }

        Store.Complete(entry, result);
    }

    private static async Task<object> ResetEnvironmentV2Async(
        BridgeEnvResetEnvelopeV2 envelope,
        long beforeStateVersion,
        CancellationToken cancellationToken)
    {
        if (envelope.Scenario == "full-run")
        {
            var request = DeserializeOptions<BridgeEnvResetRequest>(envelope.Options);
            request.Seed = ReadSeedAsString(envelope.Seed);
            var legacyPayload = await BridgeGameApi.ResetEnvResponseAsync(request, cancellationToken);
            var afterStateVersion = await BridgeGameApi.GetCurrentStateVersionV2Async(cancellationToken);
            return ProjectLegacyEnvironmentPayloadV2(legacyPayload, beforeStateVersion, afterStateVersion);
        }

        var combatRequest = DeserializeOptions<BridgeEnvCombatResetRequest>(envelope.Options);
        combatRequest.Seed = ReadSeedAsInt32(envelope.Seed);
        var combatLegacyPayload = await BridgeGameApi.CombatResetEnvResponseAsync(combatRequest, cancellationToken);
        var combatAfterStateVersion = await BridgeGameApi.GetCurrentStateVersionV2Async(cancellationToken);
        return ProjectLegacyEnvironmentPayloadV2(
            combatLegacyPayload,
            beforeStateVersion,
            combatAfterStateVersion);
    }

    private static object ProjectLegacyEnvironmentPayloadV2(
        object legacyPayload,
        long beforeStateVersion,
        long afterStateVersion)
    {
        var legacy = JsonSerializer.SerializeToNode(legacyPayload) as JsonObject ??
                     throw new InvalidOperationException("Legacy environment payload was not a JSON object.");
        var transition = legacy["transition"]?.DeepClone() as JsonObject ??
                         BuildFallbackTransitionFacts(legacy, beforeStateVersion, afterStateVersion);
        transition["before_state_version"] = beforeStateVersion;
        transition["after_state_version"] = afterStateVersion;
        transition.Remove("revision_status");
        transition.Remove("legacy_step_index_before");
        transition.Remove("legacy_step_index_after");
        var info = legacy["info"]?.DeepClone() as JsonObject ?? new JsonObject();
        info.Remove("reward_breakdown");
        info["reward_authority"] = "external-rl";
        info["legacy_bridge_reward_omitted"] = true;

        string? terminalReason = null;
        if (transition["facts"] is JsonObject facts &&
            facts["terminal_reason"] is JsonValue terminalValue &&
            terminalValue.TryGetValue<string>(out var parsedReason))
        {
            terminalReason = parsedReason;
        }

        return new JsonObject
        {
            ["episode_id"] = legacy["episode_id"]?.DeepClone(),
            ["step_index"] = legacy["step_index"]?.DeepClone(),
            ["state_version_before"] = beforeStateVersion,
            ["state_version_after"] = afterStateVersion,
            ["observation"] = legacy["obs"]?.DeepClone() ?? new JsonObject(),
            ["terminated"] = ReadBooleanNode(legacy["done"]),
            ["truncated"] = ReadBooleanNode(legacy["truncated"]),
            ["terminal_reason"] = terminalReason,
            ["legal_actions"] = BridgeLegalActionProjector.ProjectEnvironmentActions(
                legacy["legal_actions"]),
            ["transition"] = transition,
            ["transition_facts"] = transition.DeepClone(),
            ["reward"] = null,
            ["reward_status"] = "not_computed",
            ["reward_authority"] = "external-rl",
            ["info"] = info
        };
    }

    private static object? ProjectLegacyEnvironmentErrorDetailsV2(object? legacyDetails)
    {
        if (legacyDetails is null)
        {
            return null;
        }

        var details = JsonSerializer.SerializeToNode(legacyDetails) as JsonObject;
        if (details is null)
        {
            return legacyDetails;
        }
        if (details.TryGetPropertyValue("legal_actions", out var legalActions))
        {
            details["legal_actions"] = BridgeLegalActionProjector.ProjectEnvironmentActions(legalActions);
        }
        return details;
    }

    private static JsonObject BuildFallbackTransitionFacts(
        JsonObject legacy,
        long beforeStateVersion,
        long afterStateVersion)
    {
        var episodeId = legacy["episode_id"]?.GetValue<string>() ?? "unknown";
        var stepIndex = legacy["step_index"]?.GetValue<int>() ?? 0;
        return new JsonObject
        {
            ["episode_id"] = episodeId,
            ["step_index"] = stepIndex,
            ["before_state_version"] = beforeStateVersion,
            ["after_state_version"] = afterStateVersion,
            ["facts"] = new JsonObject
            {
                ["hp_delta"] = 0,
                ["gold_delta"] = 0,
                ["floor_delta"] = 0,
                ["cards_added"] = new JsonArray(),
                ["cards_removed"] = new JsonArray(),
                ["potions_added"] = new JsonArray(),
                ["potions_removed"] = new JsonArray(),
                ["room_entered"] = null,
                ["combat_result"] = "none",
                ["run_result"] = "none",
                ["terminal_reason"] = null,
                ["facts_incomplete"] = true
            }
        };
    }

    private static bool ReadBooleanNode(JsonNode? node) =>
        node is JsonValue value && value.TryGetValue<bool>(out var parsed) && parsed;

    private static BridgeEnvStepRequest BuildLegacyStepRequest(BridgeEnvStepEnvelopeV2 envelope)
    {
        var action = envelope.Action;
        foreach (var property in action.EnumerateObject())
        {
            if (property.Name is not ("action_index" or "action_handle" or "timeout_ms"))
            {
                throw new BridgeRequestException(
                    HttpStatusCode.BadRequest,
                    "unsupported_environment_action_field",
                    $"Unsupported v2 environment action field '{property.Name}'. Use action_index or action_handle.");
            }
        }

        var hasActionIndex = action.TryGetProperty("action_index", out var actionIndexElement);
        int? actionIndex = null;
        if (hasActionIndex)
        {
            if (actionIndexElement.ValueKind != JsonValueKind.Number ||
                !actionIndexElement.TryGetInt32(out var parsedIndex) ||
                parsedIndex < 0)
            {
                throw new BridgeRequestException(
                    HttpStatusCode.BadRequest,
                    "invalid_action_index",
                    "action_index must be a non-negative integer.");
            }
            actionIndex = parsedIndex;
        }

        var hasActionHandle = action.TryGetProperty("action_handle", out var actionHandleElement);
        string? actionHandle = null;
        if (hasActionHandle)
        {
            actionHandle = actionHandleElement.ValueKind == JsonValueKind.String
                ? actionHandleElement.GetString()?.Trim()
                : null;
            if (string.IsNullOrWhiteSpace(actionHandle) || actionHandle.Length > MaxActionHandleLength)
            {
                throw new BridgeRequestException(
                    HttpStatusCode.BadRequest,
                    "invalid_action_handle",
                    $"action_handle must be a non-empty string no longer than {MaxActionHandleLength} characters.");
            }
        }

        if (hasActionIndex == hasActionHandle)
        {
            throw new BridgeRequestException(
                HttpStatusCode.BadRequest,
                "invalid_environment_action_selector",
                "action must include exactly one of action_index or action_handle.");
        }

        int? timeoutMs = null;
        if (action.TryGetProperty("timeout_ms", out var timeoutElement))
        {
            if (timeoutElement.ValueKind != JsonValueKind.Number ||
                !timeoutElement.TryGetInt32(out var parsedTimeout) ||
                parsedTimeout < 1 ||
                parsedTimeout > 120000)
            {
                throw new BridgeRequestException(
                    HttpStatusCode.BadRequest,
                    "invalid_timeout_ms",
                    "timeout_ms must be an integer from 1 through 120000.");
            }
            timeoutMs = parsedTimeout;
        }

        return new BridgeEnvStepRequest
        {
            EpisodeId = envelope.EpisodeId,
            ActionIndex = actionIndex,
            ActionId = actionHandle,
            TimeoutMs = timeoutMs
        };
    }
    private static T DeserializeOptions<T>(JsonElement? options)
        where T : new()
    {
        if (options is null || options.Value.ValueKind is JsonValueKind.Null or JsonValueKind.Undefined)
        {
            return new T();
        }

        try
        {
            return JsonSerializer.Deserialize<T>(options.Value.GetRawText(), EnvironmentJsonOptions) ?? new T();
        }
        catch (JsonException ex)
        {
            throw new BridgeRequestException(
                HttpStatusCode.BadRequest,
                "invalid_environment_options",
                "options could not be mapped to the selected environment scenario.",
                new { ex.Message });
        }
    }

    private static string? ReadSeedAsString(JsonElement? seed)
    {
        if (seed is null || seed.Value.ValueKind == JsonValueKind.Null)
        {
            return null;
        }

        return seed.Value.ValueKind switch
        {
            JsonValueKind.String => seed.Value.GetString(),
            JsonValueKind.Number => seed.Value.GetRawText(),
            _ => throw new BridgeRequestException(
                HttpStatusCode.BadRequest,
                "invalid_seed",
                "seed must be a string, integer, or null.")
        };
    }

    private static int? ReadSeedAsInt32(JsonElement? seed)
    {
        if (seed is null || seed.Value.ValueKind == JsonValueKind.Null)
        {
            return null;
        }

        if (seed.Value.ValueKind == JsonValueKind.Number && seed.Value.TryGetInt32(out var numeric))
        {
            return numeric;
        }

        if (seed.Value.ValueKind == JsonValueKind.String &&
            int.TryParse(seed.Value.GetString(), out var textual))
        {
            return textual;
        }

        throw new BridgeRequestException(
            HttpStatusCode.BadRequest,
            "invalid_seed",
            "combat scenario seed must fit in a signed 32-bit integer.");
    }

    private static void NormalizeAndValidatePlayerCommand(BridgeCommandEnvelopeV2 envelope)
    {
        envelope.RequestId = NormalizeRequestId(envelope.RequestId);
        ValidateSessionId(envelope.SessionId);
        envelope.SessionId = envelope.SessionId!.Trim();

        envelope.Capability = envelope.Capability?.Trim();
        if (!string.Equals(
                envelope.Capability,
                BridgeProtocolV2.PlayerControlCapability,
                StringComparison.Ordinal))
        {
            throw new BridgeRequestException(
                HttpStatusCode.Forbidden,
                "capability_not_allowed",
                "Only the player-control capability is accepted by the v2 command endpoint.");
        }

        if (envelope.ExpectedStateVersion is not long expectedStateVersion || expectedStateVersion < 0)
        {
            throw new BridgeRequestException(
                HttpStatusCode.BadRequest,
                "missing_expected_state_version",
                "A non-negative expected_state_version is required for every v2 mutation.");
        }

        if (envelope.DeadlineUtc is not DateTimeOffset deadlineUtc)
        {
            throw new BridgeRequestException(
                HttpStatusCode.BadRequest,
                "missing_deadline_utc",
                "Every v2 player command requires deadline_utc.");
        }

        var now = DateTimeOffset.UtcNow;
        deadlineUtc = deadlineUtc.ToUniversalTime();
        if (deadlineUtc <= now)
        {
            throw new BridgeRequestException(
                HttpStatusCode.RequestTimeout,
                "deadline_expired",
                "The command deadline has already expired.");
        }
        if (deadlineUtc - now > MaxPlayerCommandDeadlineWindow)
        {
            throw new BridgeRequestException(
                HttpStatusCode.BadRequest,
                "deadline_too_far",
                $"deadline_utc must be no more than {MaxPlayerCommandDeadlineWindow.TotalSeconds:0} seconds in the future.");
        }
        envelope.DeadlineUtc = deadlineUtc;

        if (envelope.Command is null)
        {
            throw new BridgeRequestException(
                HttpStatusCode.BadRequest,
                "missing_command",
                "A command object is required.");
        }

        envelope.Command.Kind = envelope.Command.Kind?.Trim();
        if (!string.Equals(
                envelope.Command.Kind,
                BridgeProtocolV2.PerformActionCommand,
                StringComparison.Ordinal))
        {
            throw new BridgeRequestException(
                HttpStatusCode.BadRequest,
                "unsupported_command_kind",
                $"Only command kind '{BridgeProtocolV2.PerformActionCommand}' is supported by this Bridge version.");
        }

        var actionHandle = envelope.Command.ActionHandle?.Trim();
        if (string.IsNullOrWhiteSpace(actionHandle))
        {
            throw new BridgeRequestException(
                HttpStatusCode.BadRequest,
                "missing_action_handle",
                "The command must include a non-empty action_handle.");
        }

        if (actionHandle.Length > MaxActionHandleLength)
        {
            throw new BridgeRequestException(
                HttpStatusCode.BadRequest,
                "action_handle_too_long",
                $"action_handle exceeds the {MaxActionHandleLength}-character limit.");
        }

        envelope.Command.ActionHandle = actionHandle;
        var waitAfterMs = envelope.Command.WaitAfterMs ?? 0;
        if (waitAfterMs is < 0 or > MaxWaitAfterMilliseconds)
        {
            throw new BridgeRequestException(
                HttpStatusCode.BadRequest,
                "invalid_wait_after_ms",
                $"wait_after_ms must be between 0 and {MaxWaitAfterMilliseconds}.");
        }
        envelope.Command.WaitAfterMs = waitAfterMs;
    }

    private static void ValidateSessionId(string? sessionId)
    {
        sessionId = sessionId?.Trim();
        if (!string.Equals(sessionId, BridgeRuntime.SessionId, StringComparison.Ordinal))
        {
            throw new BridgeRequestException(
                HttpStatusCode.Conflict,
                "session_id_mismatch",
                "The request session_id does not match the active Bridge session.");
        }
    }

    private static void ValidateOptionalObject(JsonElement? value, string fieldName)
    {
        if (value is not null &&
            value.Value.ValueKind is not (JsonValueKind.Object or JsonValueKind.Null or JsonValueKind.Undefined))
        {
            throw new BridgeRequestException(
                HttpStatusCode.BadRequest,
                $"invalid_{fieldName}",
                $"{fieldName} must be a JSON object or null.");
        }
    }

    private static void ValidateOptionalSeed(JsonElement? seed)
    {
        if (seed is not null &&
            seed.Value.ValueKind is not (JsonValueKind.String or JsonValueKind.Number or JsonValueKind.Null or JsonValueKind.Undefined))
        {
            throw new BridgeRequestException(
                HttpStatusCode.BadRequest,
                "invalid_seed",
                "seed must be a string, integer, or null.");
        }
    }

    private static string NormalizeRequestId(string? requestId)
    {
        requestId = requestId?.Trim();
        if (string.IsNullOrWhiteSpace(requestId) || requestId.Length > MaxRequestIdLength ||
            !Guid.TryParse(requestId, out var parsed))
        {
            throw new BridgeRequestException(
                HttpStatusCode.BadRequest,
                "invalid_request_id",
                "request_id must be a UUID string no longer than 64 characters.");
        }

        return parsed.ToString("D");
    }

    private static CancellationTokenSource CreateDeadlineToken(
        DateTimeOffset deadlineUtc,
        CancellationToken bridgeShutdownToken)
    {
        var cts = CancellationTokenSource.CreateLinkedTokenSource(bridgeShutdownToken);
        var remaining = deadlineUtc - DateTimeOffset.UtcNow;
        if (remaining <= TimeSpan.Zero)
        {
            cts.Cancel();
        }
        else
        {
            // Validation caps this interval. Keep the defensive cap here so a
            // future call site cannot feed an unsupported timer duration.
            cts.CancelAfter(
                remaining > MaxPlayerCommandDeadlineWindow
                    ? MaxPlayerCommandDeadlineWindow
                    : remaining);
        }

        return cts;
    }

    private static string ComputeFingerprint(string operationKind, object payload)
    {
        var canonical = JsonSerializer.Serialize(new { operation_kind = operationKind, payload });
        return Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(canonical)));
    }

    private static BridgeCommandResultV2 BuildCommittedResult(
        BridgeCommandStoreEntry entry,
        DateTimeOffset startedAtUtc,
        object payload,
        string? versionPropertyName) =>
        new()
        {
            Ok = true,
            RequestId = entry.RequestId,
            Status = "committed",
            ReplayedResult = false,
            AcceptedAtUtc = entry.AcceptedAtUtc,
            StartedAtUtc = startedAtUtc,
            CompletedAtUtc = DateTimeOffset.UtcNow,
            CommittedStateVersion = versionPropertyName is null ? null : TryReadLongProperty(payload, versionPropertyName),
            Result = payload
        };

    private static BridgeCommandResultV2 BuildRejectedResult(
        BridgeCommandStoreEntry entry,
        string code,
        string message) =>
        BuildErrorResult(
            entry,
            "rejected_before_execution",
            null,
            code,
            message);

    private static BridgeCommandResultV2 BuildCancellationResult(
        BridgeCommandStoreEntry entry,
        DateTimeOffset? startedAtUtc,
        bool executing,
        CancellationToken bridgeShutdownToken) =>
        BuildErrorResult(
            entry,
            executing ? "outcome_unknown" : "rejected_before_execution",
            startedAtUtc,
            bridgeShutdownToken.IsCancellationRequested ? "bridge_shutting_down" : "deadline_expired",
            executing
                ? "The operation may have started, but completion was not observed before cancellation."
                : "The operation was cancelled before execution began.");

    private static BridgeCommandResultV2 BuildErrorResult(
        BridgeCommandStoreEntry entry,
        string status,
        DateTimeOffset? startedAtUtc,
        string code,
        string message,
        object? details = null) =>
        new()
        {
            Ok = false,
            RequestId = entry.RequestId,
            Status = status,
            ReplayedResult = false,
            AcceptedAtUtc = entry.AcceptedAtUtc,
            StartedAtUtc = startedAtUtc,
            CompletedAtUtc = DateTimeOffset.UtcNow,
            ErrorCode = code,
            ErrorMessage = message,
            Error = new BridgeCommandErrorV2
            {
                Code = code,
                Message = message,
                Details = details
            }
        };

    private static BridgeCommandResultV2 WithReplayFlag(
        BridgeCommandResultV2 source,
        bool replayedResult) =>
        new()
        {
            Ok = source.Ok,
            ApiVersion = source.ApiVersion,
            SchemaVersion = source.SchemaVersion,
            RequestId = source.RequestId,
            Status = source.Status,
            ReplayedResult = replayedResult,
            AcceptedAtUtc = source.AcceptedAtUtc,
            StartedAtUtc = source.StartedAtUtc,
            CompletedAtUtc = source.CompletedAtUtc,
            CommittedStateVersion = source.CommittedStateVersion,
            ErrorCode = source.ErrorCode,
            ErrorMessage = source.ErrorMessage,
            Result = source.Result,
            Error = source.Error
        };

    private static long? TryReadLongProperty(object payload, string propertyName)
    {
        try
        {
            var element = JsonSerializer.SerializeToElement(payload);
            return element.ValueKind == JsonValueKind.Object &&
                   element.TryGetProperty(propertyName, out var value) &&
                   value.TryGetInt64(out var parsed)
                ? parsed
                : null;
        }
        catch
        {
            return null;
        }
    }
}
