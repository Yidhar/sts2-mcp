using System.Net;
using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;
using Sts2McpBridge.Scripts;

static void Assert(bool condition, string message)
{
    if (!condition)
    {
        throw new InvalidOperationException(message);
    }
}

static void AssertCommandWireMatchesFixture(
    JsonElement actual,
    JsonElement fixture,
    string expectedStatus)
{
    var actualNames = actual.EnumerateObject().Select(static property => property.Name).ToHashSet(StringComparer.Ordinal);
    var fixtureNames = fixture.EnumerateObject().Select(static property => property.Name).ToHashSet(StringComparer.Ordinal);
    Assert(actualNames.SetEquals(fixtureNames),
        $"C# {expectedStatus} envelope fields must match the shared fixture exactly");
    Assert(actual.GetProperty("status").GetString() == expectedStatus &&
           actual.GetProperty("ok").GetBoolean() == fixture.GetProperty("ok").GetBoolean() &&
           actual.GetProperty("api_version").GetString() == fixture.GetProperty("api_version").GetString() &&
           actual.GetProperty("schema_version").GetString() == fixture.GetProperty("schema_version").GetString(),
        $"C# {expectedStatus} envelope status/ok/version must match the shared fixture");
    foreach (var nullableField in new[]
             {
                 "started_at_utc", "completed_at_utc", "committed_state_version",
                 "error_code", "error_message", "result", "error"
             })
    {
        Assert(
            (actual.GetProperty(nullableField).ValueKind == JsonValueKind.Null) ==
            (fixture.GetProperty(nullableField).ValueKind == JsonValueKind.Null),
            $"C# {expectedStatus}.{nullableField} nullability must match the shared fixture");
    }
}

static async Task<HttpResponseMessage> SendBridgeRequestAsync(
    HttpClient client,
    HttpMethod method,
    string path,
    string? bearerToken = null,
    object? body = null)
{
    using var request = new HttpRequestMessage(method, new Uri(new Uri(BridgeRuntime.BaseUrl), path));
    if (!string.IsNullOrWhiteSpace(bearerToken))
    {
        request.Headers.Authorization = new AuthenticationHeaderValue("Bearer", bearerToken);
    }
    if (body is not null)
    {
        request.Content = new StringContent(
            JsonSerializer.Serialize(body),
            Encoding.UTF8,
            "application/json");
    }
    return await client.SendAsync(request);
}

var currentRetailProfile = BridgeGameAssemblyCompatibilityRegistry.SupportedProfiles.Single();
var unknownAssemblyProbe = new FakeGameAssemblyProbe(
    currentRetailProfile.ExpectedIdentity with
    {
        InformationalVersion = "0.1.0+unknown-retail-update",
        ModuleVersionId = Guid.Parse("11111111-1111-1111-1111-111111111111")
    });
var unknownAssessment = BridgeGameAssemblyCompatibilityRegistry.Evaluate(unknownAssemblyProbe);
Assert(!unknownAssessment.StartupAllowed &&
       unknownAssessment.Health == "degraded" &&
       unknownAssessment.ErrorCode == "unsupported_game_assembly" &&
       unknownAssessment.ProfileId is null,
    "an unknown retail assembly identity must fail closed before activation");
Assert(unknownAssemblyProbe.ProbedCapabilities.Count == 0,
    "an unknown identity must not be treated as any known profile or run its capability list");

var missingCapabilityId = currentRetailProfile.RequiredCapabilities[0].Id;
var missingCapabilityProbe = new FakeGameAssemblyProbe(
    currentRetailProfile.ExpectedIdentity,
    missingCapabilities: [missingCapabilityId]);
var missingCapabilityAssessment = BridgeGameAssemblyCompatibilityRegistry.Evaluate(missingCapabilityProbe);
Assert(!missingCapabilityAssessment.StartupAllowed &&
       missingCapabilityAssessment.Health == "degraded" &&
       missingCapabilityAssessment.ErrorCode == "required_game_capability_missing" &&
       missingCapabilityAssessment.ProfileId == currentRetailProfile.Id,
    "a matched profile with one missing required capability must fail closed");
Assert(missingCapabilityAssessment.ProbeResults.Count == currentRetailProfile.RequiredCapabilities.Count &&
       missingCapabilityAssessment.ProbeResults.Single(result => result.CapabilityId == missingCapabilityId).Passed == false,
    "the gate must retain the complete per-capability probe record when a requirement is missing");

var throwingCapabilityId = currentRetailProfile.RequiredCapabilities[1].Id;
var throwingCapabilityProbe = new FakeGameAssemblyProbe(
    currentRetailProfile.ExpectedIdentity,
    throwingCapabilities: [throwingCapabilityId]);
var throwingCapabilityAssessment = BridgeGameAssemblyCompatibilityRegistry.Evaluate(throwingCapabilityProbe);
Assert(!throwingCapabilityAssessment.StartupAllowed &&
       throwingCapabilityAssessment.ProbeResults.Single(
           result => result.CapabilityId == throwingCapabilityId).Code == "capability_probe_exception",
    "a capability probe exception must be converted into a fail-closed degraded assessment");

var supportedAssemblyProbe = new FakeGameAssemblyProbe(currentRetailProfile.ExpectedIdentity);
var supportedAssessment = BridgeGameAssemblyCompatibilityRegistry.Evaluate(supportedAssemblyProbe);
Assert(supportedAssessment.StartupAllowed &&
       supportedAssessment.Health == "ready" &&
       supportedAssessment.ErrorCode.Length == 0 &&
       supportedAssessment.ProfileId == currentRetailProfile.Id,
    "the current audited retail profile must pass when its fake probe exposes every required capability");
Assert(supportedAssessment.PassedProbeCount == currentRetailProfile.RequiredCapabilities.Count &&
       supportedAssemblyProbe.ProbedCapabilities.SequenceEqual(
           currentRetailProfile.RequiredCapabilities.Select(static requirement => requirement.Id),
           StringComparer.Ordinal),
    "the current profile must execute and record every required capability probe in declared order");
BridgeGameCompatibilityState.Publish(supportedAssessment);
Console.WriteLine("Bridge retail game compatibility gate tests passed.");

var now = new DateTimeOffset(2026, 7, 11, 0, 0, 0, TimeSpan.Zero);
var store = new BoundedCommandResultStore(2, TimeSpan.FromMinutes(10), () => now);
var firstDisposition = store.GetOrCreate(
    "request-1", "fingerprint-a", BridgeProtocolV2.PlayerControlCapability, "player-command", out var first);
Assert(firstDisposition == BridgeCommandStoreDisposition.Created && first is not null, "first request should be created");
Assert(first!.Capability == BridgeProtocolV2.PlayerControlCapability, "store entry must retain its capability owner");
var acceptedWire = JsonSerializer.SerializeToElement(first.Snapshot);
Assert(store.GetOrCreate(
        "request-1", "fingerprint-a", BridgeProtocolV2.PlayerControlCapability, "player-command", out var duplicate) ==
       BridgeCommandStoreDisposition.Existing,
    "same request/fingerprint/capability should coalesce");
Assert(ReferenceEquals(first, duplicate), "duplicate request should point at the same retained command");
Assert(store.GetOrCreate(
        "request-1", "fingerprint-b", BridgeProtocolV2.PlayerControlCapability, "player-command", out _) ==
       BridgeCommandStoreDisposition.FingerprintConflict,
    "request id reuse with a different fingerprint must fail closed");
Assert(store.GetOrCreate(
        "request-1", "fingerprint-a", BridgeProtocolV2.TrainingCapability, "environment-step", out _) ==
       BridgeCommandStoreDisposition.FingerprintConflict,
    "request id reuse across capabilities must fail globally");

first.MarkExecuting(now.AddSeconds(1));
var executingWire = JsonSerializer.SerializeToElement(first.Snapshot);
Assert(first.Snapshot.Status == "executing" && first.Snapshot.Ok,
    "accepted/executing entry should expose a successful in-flight envelope");
var committed = new BridgeCommandResultV2
{
    Ok = true,
    RequestId = "request-1",
    Status = "committed",
    ReplayedResult = false,
    AcceptedAtUtc = first.AcceptedAtUtc,
    StartedAtUtc = now.AddSeconds(1),
    CompletedAtUtc = now.AddSeconds(2),
    Result = new { state_version_after = 2 }
};
store.Complete(first, committed);
Assert((await first.WaitAsync(CancellationToken.None)).Status == "committed", "completion should be replayable");

Assert(store.GetOrCreate(
        "request-2", "fingerprint-2", BridgeProtocolV2.TrainingCapability, "environment-step", out var second) ==
       BridgeCommandStoreDisposition.Created,
    "second entry should fit");
Assert(store.GetOrCreate(
        "request-3", "fingerprint-3", BridgeProtocolV2.PlayerControlCapability, "player-command", out var rejectedAtCapacity) ==
       BridgeCommandStoreDisposition.CapacityExceeded && rejectedAtCapacity is null,
    "capacity must reject a new identity rather than evict an unexpired completion");
Assert(store.GetOrCreate(
        "request-1", "fingerprint-a", BridgeProtocolV2.PlayerControlCapability, "player-command", out var retainedFirst) ==
       BridgeCommandStoreDisposition.Existing && ReferenceEquals(first, retainedFirst),
    "unexpired request identity must remain replayable after a capacity rejection");
Assert(second is not null && !second.IsCompleted, "active entries must not be evicted");

now = now.AddMinutes(11);
Assert(store.GetOrCreate(
        "request-3", "fingerprint-3", BridgeProtocolV2.PlayerControlCapability, "player-command", out _) ==
       BridgeCommandStoreDisposition.Created,
    "capacity may be reclaimed only after a completed identity expires");
Assert(!store.TryGet("request-1", out _), "expired completed identity should be purged");
Assert(store.TryGet("request-2", out _), "in-flight identity must survive TTL cleanup");

var ttlNow = new DateTimeOffset(2026, 7, 11, 0, 0, 0, TimeSpan.Zero);
var ttlStore = new BoundedCommandResultStore(1, TimeSpan.FromSeconds(5), () => ttlNow);
ttlStore.GetOrCreate(
    "ttl", "fingerprint", BridgeProtocolV2.PlayerControlCapability, "player-command", out var ttlEntry);
ttlStore.Complete(ttlEntry!, new BridgeCommandResultV2
{
    Ok = false,
    RequestId = "ttl",
    Status = "rejected_before_execution",
    ReplayedResult = false,
    AcceptedAtUtc = ttlNow,
    CompletedAtUtc = ttlNow
});
ttlNow = ttlNow.AddSeconds(6);
Assert(!ttlStore.TryGet("ttl", out _), "completed result should expire after bounded TTL");

var concurrent = 0;
var maximumConcurrent = 0;
var mutationTasks = Enumerable.Range(0, 8).Select(index =>
    BridgeMutationGate.RunAsync(
        $"test-{index}",
        async cancellationToken =>
        {
            var current = Interlocked.Increment(ref concurrent);
            maximumConcurrent = Math.Max(maximumConcurrent, current);
            await Task.Delay(5, cancellationToken);
            Interlocked.Decrement(ref concurrent);
            return index;
        },
        CancellationToken.None)).ToArray();
await Task.WhenAll(mutationTasks);
Assert(maximumConcurrent == 1, "mutation gate must allow exactly one active mutation");

Console.WriteLine("Bridge core tests passed.");

var commandRequestId = Guid.NewGuid().ToString("D");
var command = new BridgeCommandEnvelopeV2
{
    RequestId = commandRequestId,
    SessionId = BridgeRuntime.SessionId,
    Capability = BridgeProtocolV2.PlayerControlCapability,
    ExpectedStateVersion = 7,
    DeadlineUtc = DateTimeOffset.UtcNow.AddSeconds(30),
    Command = new BridgePlayerCommandV2
    {
        Kind = BridgeProtocolV2.PerformActionCommand,
        ActionHandle = "end_turn"
    }
};
var firstCommandResult = await BridgeCommandService.SubmitAsync(
    command, CancellationToken.None, CancellationToken.None);
var replayedCommandResult = await BridgeCommandService.SubmitAsync(
    command, CancellationToken.None, CancellationToken.None);
Assert(firstCommandResult.Status == "committed" && !firstCommandResult.ReplayedResult,
    "first command response must not be marked as replayed");
Assert(replayedCommandResult.Status == "committed" && replayedCommandResult.ReplayedResult,
    "duplicate command response must be an immutable replay clone");
Assert(firstCommandResult.CommittedStateVersion == 8, "committed state version should be extracted");
Assert(BridgeGameApi.ActionExecutionCount == 1, "duplicate request id must execute exactly once");
Assert(BridgeCommandService.GetStatus(commandRequestId, BridgeProtocolV2.PlayerControlCapability).Status == "committed",
    "same-capability status query should return the retained player command");
try
{
    BridgeCommandService.GetStatus(commandRequestId, BridgeProtocolV2.TrainingCapability);
    throw new InvalidOperationException("cross-capability status query should fail");
}
catch (BridgeRequestException ex)
{
    Assert(ex.StatusCode == HttpStatusCode.Forbidden && ex.ErrorCode == "command_capability_mismatch",
        "training scope must not read a player-control command");
}

var rejectedPlayerResult = await BridgeCommandService.SubmitAsync(
    new BridgeCommandEnvelopeV2
    {
        RequestId = Guid.NewGuid().ToString("D"),
        SessionId = BridgeRuntime.SessionId,
        Capability = BridgeProtocolV2.PlayerControlCapability,
        ExpectedStateVersion = BridgeGameApi.CurrentStateVersion,
        DeadlineUtc = DateTimeOffset.UtcNow.AddSeconds(30),
        Command = new BridgePlayerCommandV2
        {
            Kind = BridgeProtocolV2.PerformActionCommand,
            ActionHandle = "test:reject"
        }
    },
    CancellationToken.None,
    CancellationToken.None);
Assert(!rejectedPlayerResult.Ok &&
       rejectedPlayerResult.Status == "rejected_before_execution" &&
       rejectedPlayerResult.StartedAtUtc is null &&
       rejectedPlayerResult.CompletedAtUtc is not null &&
       rejectedPlayerResult.Error is not null,
    "safe player rejection must match the rejected command-result wire semantics");
var rejectedPlayerWire = JsonSerializer.SerializeToElement(rejectedPlayerResult);

try
{
    await BridgeCommandService.SubmitAsync(
        new BridgeCommandEnvelopeV2
        {
            RequestId = Guid.NewGuid().ToString("D"),
            SessionId = BridgeRuntime.SessionId,
            Capability = BridgeProtocolV2.PlayerControlCapability,
            ExpectedStateVersion = BridgeGameApi.CurrentStateVersion,
            Command = new BridgePlayerCommandV2
            {
                Kind = BridgeProtocolV2.PerformActionCommand,
                ActionHandle = "end_turn"
            }
        },
        CancellationToken.None,
        CancellationToken.None);
    throw new InvalidOperationException("player command without deadline_utc should fail");
}
catch (BridgeRequestException ex)
{
    Assert(ex.StatusCode == HttpStatusCode.BadRequest && ex.ErrorCode == "missing_deadline_utc",
        "every v2 player command must require an absolute deadline");
}

var farDeadlineRequestId = Guid.NewGuid().ToString("D");
try
{
    await BridgeCommandService.SubmitAsync(
        new BridgeCommandEnvelopeV2
        {
            RequestId = farDeadlineRequestId,
            SessionId = BridgeRuntime.SessionId,
            Capability = BridgeProtocolV2.PlayerControlCapability,
            ExpectedStateVersion = BridgeGameApi.CurrentStateVersion,
            DeadlineUtc = DateTimeOffset.UtcNow.AddMinutes(5),
            Command = new BridgePlayerCommandV2
            {
                Kind = BridgeProtocolV2.PerformActionCommand,
                ActionHandle = "end_turn"
            }
        },
        CancellationToken.None,
        CancellationToken.None);
    throw new InvalidOperationException("far-future command deadline should fail");
}
catch (BridgeRequestException ex)
{
    Assert(ex.StatusCode == HttpStatusCode.BadRequest && ex.ErrorCode == "deadline_too_far",
        "v2 player deadline must be bounded to the advertised two-minute window");
}
try
{
    BridgeCommandService.GetStatus(farDeadlineRequestId, BridgeProtocolV2.PlayerControlCapability);
    throw new InvalidOperationException("invalid deadline must not allocate a command identity");
}
catch (BridgeRequestException ex)
{
    Assert(ex.StatusCode == HttpStatusCode.NotFound && ex.ErrorCode == "command_not_found",
        "deadline validation must happen before an accepted command-store entry is created");
}

try
{
    await BridgeCommandService.SubmitAsync(
        new BridgeCommandEnvelopeV2
        {
            RequestId = Guid.NewGuid().ToString("D"),
            SessionId = BridgeRuntime.SessionId,
            Capability = BridgeProtocolV2.PlayerControlCapability,
            ExpectedStateVersion = BridgeGameApi.CurrentStateVersion,
            DeadlineUtc = DateTimeOffset.UtcNow.AddSeconds(30),
            Command = new BridgePlayerCommandV2
            {
                Kind = BridgeProtocolV2.PerformActionCommand,
                ActionHandle = "end_turn",
                WaitAfterMs = 5001
            }
        },
        CancellationToken.None,
        CancellationToken.None);
    throw new InvalidOperationException("wait_after_ms above 5000 should fail");
}
catch (BridgeRequestException ex)
{
    Assert(ex.StatusCode == HttpStatusCode.BadRequest && ex.ErrorCode == "invalid_wait_after_ms",
        "v2 command service must reject rather than clamp an out-of-contract wait_after_ms");
}

try
{
    await BridgeCommandService.SubmitAsync(
        new BridgeCommandEnvelopeV2
        {
            RequestId = Guid.NewGuid().ToString("D"),
            SessionId = BridgeRuntime.SessionId,
            Capability = BridgeProtocolV2.PlayerControlCapability,
            ExpectedStateVersion = BridgeGameApi.CurrentStateVersion,
            DeadlineUtc = DateTimeOffset.UtcNow.AddSeconds(30),
            Command = new BridgePlayerCommandV2
            {
                Kind = BridgeProtocolV2.PerformActionCommand,
                ActionHandle = "   "
            }
        },
        CancellationToken.None,
        CancellationToken.None);
    throw new InvalidOperationException("blank action_handle should fail");
}
catch (BridgeRequestException ex)
{
    Assert(ex.StatusCode == HttpStatusCode.BadRequest && ex.ErrorCode == "missing_action_handle",
        "v2 player command action_handle must be non-empty after normalization");
}

try
{
    await BridgeCommandService.SubmitEnvironmentResetAsync(
        new BridgeEnvResetEnvelopeV2
        {
            RequestId = Guid.NewGuid().ToString("D"),
            SessionId = BridgeRuntime.SessionId,
            Scenario = "full-run"
        },
        CancellationToken.None,
        CancellationToken.None);
    throw new InvalidOperationException("reset without expected_state_version should fail");
}
catch (BridgeRequestException ex)
{
    Assert(ex.StatusCode == HttpStatusCode.BadRequest && ex.ErrorCode == "missing_expected_state_version",
        "every v2 reset must require an expected frontier revision");
}

var staleReset = await BridgeCommandService.SubmitEnvironmentResetAsync(
    new BridgeEnvResetEnvelopeV2
    {
        RequestId = Guid.NewGuid().ToString("D"),
        SessionId = BridgeRuntime.SessionId,
        Scenario = "full-run",
        ExpectedStateVersion = BridgeGameApi.CurrentStateVersion - 1
    },
    CancellationToken.None,
    CancellationToken.None);
Assert(!staleReset.Ok && staleReset.Status == "rejected_before_execution" &&
       staleReset.StartedAtUtc is null &&
       staleReset.ErrorCode == "state_version_conflict" &&
       BridgeGameApi.EnvironmentOperationCount == 0,
    "stale reset revision must reject before any environment mutation");

var resetRequestId = Guid.NewGuid().ToString("D");
var resetExpectedStateVersion = BridgeGameApi.CurrentStateVersion;
var reset = new BridgeEnvResetEnvelopeV2
{
    RequestId = resetRequestId,
    SessionId = BridgeRuntime.SessionId,
    Scenario = "full-run",
    ExpectedStateVersion = resetExpectedStateVersion,
    Seed = JsonSerializer.SerializeToElement("ABCDEF1234"),
    Options = JsonSerializer.SerializeToElement(new { })
};
var resetResult = await BridgeCommandService.SubmitEnvironmentResetAsync(
    reset, CancellationToken.None, CancellationToken.None);
var resetReplay = await BridgeCommandService.SubmitEnvironmentResetAsync(
    reset, CancellationToken.None, CancellationToken.None);
Assert(resetResult.Status == "committed" && resetReplay.ReplayedResult,
    "environment reset must retain and replay its idempotent result");

Assert(BridgeGameApi.EnvironmentOperationCount == 1, "duplicate reset must execute exactly once");
var resetV2Payload = JsonSerializer.SerializeToElement(resetResult.Result);
Assert(resetV2Payload.GetProperty("state_version_before").GetInt64() == resetExpectedStateVersion,
    "reset result must preserve its validated before revision");
Assert(resetV2Payload.GetProperty("state_version_after").GetInt64() == BridgeGameApi.CurrentStateVersion &&
       resetResult.CommittedStateVersion == BridgeGameApi.CurrentStateVersion &&
       resetResult.CommittedStateVersion != resetV2Payload.GetProperty("step_index").GetInt32(),
    "reset result and command envelope must use the real frontier revision, not step_index");
Assert(resetV2Payload.GetProperty("transition").GetProperty("before_state_version").GetInt64() == resetExpectedStateVersion &&
       resetV2Payload.GetProperty("transition").GetProperty("after_state_version").GetInt64() == BridgeGameApi.CurrentStateVersion,
    "reset transition must carry real before/after frontier revisions");
Assert(resetV2Payload.GetProperty("reward").ValueKind == JsonValueKind.Null,
    "v2 reset must not expose the legacy Bridge scalar reward");
Assert(resetV2Payload.GetProperty("reward_authority").GetString() == "external-rl",
    "v2 reward authority must be external RL");
Assert(resetV2Payload.TryGetProperty("transition_facts", out _),
    "v2 reset must expose canonical transition facts");
Assert(!resetV2Payload.GetProperty("info").TryGetProperty("reward_breakdown", out _),
    "v2 info must omit the legacy reward breakdown");
var resetLegalAction = resetV2Payload.GetProperty("legal_actions")[0];
Assert(resetLegalAction.GetProperty("action_handle").GetString() == "end_turn" &&
       !resetLegalAction.TryGetProperty("action_id", out _) &&
       resetLegalAction.GetProperty("energy_cost").GetInt32() == 0 &&
       resetLegalAction.GetProperty("target").GetProperty("side").GetString() == "player",
    "v2 reset legal actions must rename only the legacy identity and retain rich training metadata");

foreach (var invalidAction in new[]
         {
             JsonSerializer.SerializeToElement(new { action_id = "end_turn" }),
             JsonSerializer.SerializeToElement(new { action_index = 0, action_handle = "end_turn" })
         })
{
    try
    {
        await BridgeCommandService.SubmitEnvironmentStepAsync(
            new BridgeEnvStepEnvelopeV2
            {
                RequestId = Guid.NewGuid().ToString("D"),
                SessionId = BridgeRuntime.SessionId,
                EpisodeId = "episode",
                ExpectedStepIndex = 0,
                Action = invalidAction
            },
            CancellationToken.None,
            CancellationToken.None);
        throw new InvalidOperationException("invalid v2 step selector should fail");
    }
    catch (BridgeRequestException ex)
    {
        Assert(ex.StatusCode == HttpStatusCode.BadRequest,
            "v2 step action_id alias and dual selector must fail before command acceptance");
    }
}
Assert(BridgeGameApi.EnvironmentOperationCount == 1,
    "invalid v2 step selectors must not execute an environment operation");

var stepBeforeStateVersion = BridgeGameApi.CurrentStateVersion;
var step = new BridgeEnvStepEnvelopeV2
{
    RequestId = Guid.NewGuid().ToString("D"),
    SessionId = BridgeRuntime.SessionId,
    EpisodeId = "episode",
    ExpectedStepIndex = 0,
    Action = JsonSerializer.SerializeToElement(new { action_handle = "end_turn" })
};
var stepResult = await BridgeCommandService.SubmitEnvironmentStepAsync(
    step, CancellationToken.None, CancellationToken.None);
Assert(stepResult.Status == "committed", "strict environment step wrapper should commit");
Assert(BridgeGameApi.EnvironmentOperationCount == 2, "environment step should execute once");
var stepV2Payload = JsonSerializer.SerializeToElement(stepResult.Result);
Assert(stepV2Payload.GetProperty("state_version_before").GetInt64() == stepBeforeStateVersion &&
       stepV2Payload.GetProperty("state_version_after").GetInt64() == BridgeGameApi.CurrentStateVersion &&
       stepResult.CommittedStateVersion == BridgeGameApi.CurrentStateVersion,
    "step result must report real before/after frontier revisions");
Assert(stepV2Payload.GetProperty("reward").ValueKind == JsonValueKind.Null,
    "v2 step reward must be uncomputed");
Assert(stepV2Payload.GetProperty("transition").GetProperty("facts").GetProperty("hp_delta").GetInt32() == -2,
    "v2 step must preserve factual deltas independently from legacy reward");
var stepLegalAction = stepV2Payload.GetProperty("legal_actions")[0];
Assert(stepLegalAction.GetProperty("action_handle").GetString() == "reward:continue" &&
       !stepLegalAction.TryGetProperty("action_id", out _) &&
       stepLegalAction.GetProperty("selection").GetString() == "proceed",
    "v2 step legal actions must expose action_handle only and retain selection metadata");

var errorDetailsStep = new BridgeEnvStepEnvelopeV2
{
    RequestId = Guid.NewGuid().ToString("D"),
    SessionId = BridgeRuntime.SessionId,
    EpisodeId = "episode",
    ExpectedStepIndex = 1,
    Action = JsonSerializer.SerializeToElement(new { action_handle = "test:error-details" })
};
var errorDetailsResult = await BridgeCommandService.SubmitEnvironmentStepAsync(
    errorDetailsStep, CancellationToken.None, CancellationToken.None);
var errorDetailsWire = JsonSerializer.SerializeToElement(errorDetailsResult.Error?.Details);
var errorDetailsLegalAction = errorDetailsWire.GetProperty("legal_actions")[0];
Assert(errorDetailsResult.Status == "outcome_unknown" &&
       errorDetailsLegalAction.GetProperty("action_handle").GetString() == "end_turn" &&
       !errorDetailsLegalAction.TryGetProperty("action_id", out _) &&
       errorDetailsLegalAction.GetProperty("energy_cost").GetInt32() == 0,
    "v2 environment error details must not leak legacy legal-action identities");

Console.WriteLine("Bridge v2 command/environment tests passed.");

BridgeRuntime.LegacyV1Enabled = false;
BridgeSessionRegistry.DeleteSessionFileIfOwned();
BridgeGameCompatibilityState.Publish(unknownAssessment);
var deniedCompatibilityDescriptorRejected = false;
try
{
    BridgeSessionRegistry.WriteSessionFile();
}
catch (InvalidOperationException)
{
    deniedCompatibilityDescriptorRejected = true;
}
Assert(deniedCompatibilityDescriptorRejected && !File.Exists(BridgeRuntime.SessionFilePath),
    "session discovery must remain unpublished after a denied game compatibility decision");
BridgeGameCompatibilityState.Publish(supportedAssessment);
BridgeSessionRegistry.WriteSessionFile();
using (var descriptor = JsonDocument.Parse(File.ReadAllText(BridgeRuntime.SessionFilePath)))
{
    var root = descriptor.RootElement;
    Assert(root.GetProperty("schema_version").GetString() == BridgeProtocolV2.SchemaVersion,
        "session descriptor must use generated schema version");
    Assert(root.GetProperty("capabilities").ValueKind == JsonValueKind.Array,
        "session capabilities must match the shared array contract");
    Assert(root.GetProperty("capability_tokens").TryGetProperty("player-control", out _),
        "session descriptor must publish the player-control scoped token");
    Assert(root.GetProperty("capability_tokens").TryGetProperty("training", out _),
        "enabled training v2 must publish a distinct scoped token");
    Assert(!root.GetProperty("capability_tokens").TryGetProperty("legacy-privileged", out _),
        "disabled legacy-v1 must not publish a privileged token");
    Assert(!root.TryGetProperty("token", out _),
        "top-level legacy token must be absent by default");
    Assert(!root.GetProperty("api_versions").EnumerateArray().Any(
            value => value.GetString() == "legacy-v1"),
        "disabled legacy-v1 must not be advertised in api_versions");
    Assert(!root.GetProperty("capability_details").GetProperty("legacy_v1").GetProperty("enabled").GetBoolean(),
        "capability_details must report legacy-v1 as disabled");
    var gameCompatibility = root.GetProperty("game_compatibility");
    Assert(gameCompatibility.GetProperty("health").GetString() == "ready" &&
           gameCompatibility.GetProperty("startup_allowed").GetBoolean() &&
           gameCompatibility.GetProperty("profile_id").GetString() == currentRetailProfile.Id,
        "a published session must record the exact ready game-adapter profile");
    Assert(gameCompatibility.GetProperty("assembly").GetProperty("informational_version").GetString() ==
               currentRetailProfile.ExpectedIdentity.InformationalVersion &&
           gameCompatibility.GetProperty("assembly").GetProperty("module_version_id").GetGuid() ==
               currentRetailProfile.ExpectedIdentity.ModuleVersionId,
        "the session compatibility record must retain the audited assembly fingerprint");
    Assert(gameCompatibility.GetProperty("probes").GetArrayLength() ==
               currentRetailProfile.RequiredCapabilities.Count &&
           gameCompatibility.GetProperty("probes").EnumerateArray().All(
               static probe => probe.GetProperty("passed").GetBoolean()),
        "the session compatibility record must retain every successful startup probe");
}
BridgeSessionRegistry.DeleteSessionFileIfOwned();
Assert(!File.Exists(BridgeRuntime.SessionFilePath), "owned session descriptor must be removed on shutdown");

BridgeRuntime.LegacyV1Enabled = true;
BridgeSessionRegistry.WriteSessionFile();
using (var legacyDescriptor = JsonDocument.Parse(File.ReadAllText(BridgeRuntime.SessionFilePath)))
{
    var root = legacyDescriptor.RootElement;
    Assert(root.GetProperty("token").GetString() == BridgeRuntime.LegacySessionToken,
        "explicit legacy mode must publish only the independent legacy token at top level");
    Assert(root.GetProperty("capability_tokens").GetProperty("legacy-privileged").GetString() ==
           BridgeRuntime.LegacySessionToken,
        "explicit legacy mode must publish a legacy-privileged scoped token");
    Assert(root.GetProperty("token").GetString() != root.GetProperty("capability_tokens").GetProperty("player-control").GetString(),
        "legacy and player-control credentials must never be equal");
    Assert(root.GetProperty("api_versions").EnumerateArray().Any(value => value.GetString() == "legacy-v1"),
        "explicit legacy mode must advertise legacy-v1");
}
BridgeSessionRegistry.DeleteSessionFileIfOwned();
BridgeRuntime.LegacyV1Enabled = false;
Console.WriteLine("Bridge session descriptor tests passed.");

Assert(BridgeServer.Start(), "pure Bridge host should start on a loopback test port");
try
{
    using var httpClient = new HttpClient { Timeout = TimeSpan.FromSeconds(5) };

    using (var publicHealth = await SendBridgeRequestAsync(httpClient, HttpMethod.Get, "health"))
    {
        Assert(publicHealth.StatusCode == HttpStatusCode.OK, "public /health must remain available for readiness");
        using var document = JsonDocument.Parse(await publicHealth.Content.ReadAsStringAsync());
        var root = document.RootElement;
        Assert(root.GetProperty("ready").GetBoolean(), "public /health must report readiness");
        Assert(!root.TryGetProperty("session_id", out _) &&
               !root.TryGetProperty("process_id", out _) &&
               !root.TryGetProperty("coordinator", out _) &&
               !root.TryGetProperty("mutation_gate", out _) &&
               !root.TryGetProperty("command_store", out _),
            "public /health must not expose session, process, or diagnostic internals");
    }

    using (var unauthenticatedHealth = await SendBridgeRequestAsync(httpClient, HttpMethod.Get, "v2/health"))
    {
        Assert(unauthenticatedHealth.StatusCode == HttpStatusCode.Unauthorized,
            "detailed /v2/health must require a scoped token");
    }
    using (var playerHealth = await SendBridgeRequestAsync(
               httpClient, HttpMethod.Get, "v2/health", BridgeRuntime.SessionToken))
    {
        Assert(playerHealth.StatusCode == HttpStatusCode.OK, "player scope must read detailed health");
        using var document = JsonDocument.Parse(await playerHealth.Content.ReadAsStringAsync());
        Assert(document.RootElement.GetProperty("authorized_capability").GetString() ==
               BridgeProtocolV2.PlayerControlCapability,
            "detailed health must identify the authorized scope without exposing its token");
        Assert(document.RootElement.TryGetProperty("coordinator", out _),
            "authenticated detailed health must retain diagnostics");
    }
    using (var trainingHealth = await SendBridgeRequestAsync(
               httpClient, HttpMethod.Get, "v2/health", BridgeRuntime.TrainingSessionToken))
    {
        Assert(trainingHealth.StatusCode == HttpStatusCode.OK, "training scope must read detailed health");
    }

    using (var disabledLegacy = await SendBridgeRequestAsync(
               httpClient, HttpMethod.Get, "env/spec", BridgeRuntime.SessionToken))
    {
        Assert(disabledLegacy.StatusCode == HttpStatusCode.NotFound,
            "legacy-v1 routes must be disabled by default");
    }

    using (var playerSpec = await SendBridgeRequestAsync(
               httpClient, HttpMethod.Get, "v2/env/spec", BridgeRuntime.SessionToken))
    {
        Assert(playerSpec.StatusCode == HttpStatusCode.Forbidden,
            "player-control must not read privileged environment spec");
    }
    using (var playerState = await SendBridgeRequestAsync(
               httpClient, HttpMethod.Get, "v2/env/state", BridgeRuntime.SessionToken))
    {
        Assert(playerState.StatusCode == HttpStatusCode.Forbidden,
            "player-control must not read privileged environment state");
    }
    using (var playerCatalog = await SendBridgeRequestAsync(
               httpClient, HttpMethod.Get, "v2/env/combat_catalog", BridgeRuntime.SessionToken))
    {
        Assert(playerCatalog.StatusCode == HttpStatusCode.Forbidden,
            "player-control must not read privileged combat catalog");
    }

    using (var trainingSpec = await SendBridgeRequestAsync(
               httpClient, HttpMethod.Get, "v2/env/spec", BridgeRuntime.TrainingSessionToken))
    {
        Assert(trainingSpec.StatusCode == HttpStatusCode.OK,
            "training scope must read /v2/env/spec");
        using var document = JsonDocument.Parse(await trainingSpec.Content.ReadAsStringAsync());
        Assert(document.RootElement.GetProperty("capability").GetString() == BridgeProtocolV2.TrainingCapability,
            "v2 env spec must declare training authority");
        var legalActionShape = document.RootElement.GetProperty("action_encoding").GetProperty("legal_action_shape");
        Assert(legalActionShape[1].GetString() == "action_handle" &&
               !legalActionShape.EnumerateArray().Any(static item => item.GetString() == "action_id"),
            "v2 env spec must advertise action_handle without the legacy action_id alias");
    }
    using (var trainingState = await SendBridgeRequestAsync(
               httpClient, HttpMethod.Get, "v2/env/state", BridgeRuntime.TrainingSessionToken))
    {
        Assert(trainingState.StatusCode == HttpStatusCode.OK,
            "training scope must read /v2/env/state");
        using var document = JsonDocument.Parse(await trainingState.Content.ReadAsStringAsync());
        Assert(document.RootElement.GetProperty("state_version").GetInt64() == BridgeGameApi.CurrentStateVersion,
            "v2 env state must expose the real current frontier revision");
        var legalAction = document.RootElement.GetProperty("legal_actions")[0];
        Assert(legalAction.GetProperty("action_handle").GetString() == "end_turn" &&
               !legalAction.TryGetProperty("action_id", out _) &&
               legalAction.GetProperty("diagnostic").GetProperty("retained").GetBoolean(),
            "v2 env state must expose action_handle-only identity without dropping training metadata");
    }
    using (var trainingCatalog = await SendBridgeRequestAsync(
               httpClient, HttpMethod.Get, "v2/env/combat_catalog", BridgeRuntime.TrainingSessionToken))
    {
        Assert(trainingCatalog.StatusCode == HttpStatusCode.OK,
            "training scope must read /v2/env/combat_catalog");
    }

    using (var unknownCommandField = await SendBridgeRequestAsync(
               httpClient,
               HttpMethod.Post,
               "v2/commands",
               BridgeRuntime.SessionToken,
               new
               {
                   request_id = Guid.NewGuid().ToString("D"),
                   session_id = BridgeRuntime.SessionId,
                   capability = BridgeProtocolV2.PlayerControlCapability,
                   expected_state_version = BridgeGameApi.CurrentStateVersion,
                   deadline_utc = DateTimeOffset.UtcNow.AddSeconds(30),
                   unexpected_field = true,
                   command = new
                   {
                       kind = BridgeProtocolV2.PerformActionCommand,
                       action_handle = "end_turn"
                   }
               }))
    {
        Assert(unknownCommandField.StatusCode == HttpStatusCode.BadRequest &&
               (await unknownCommandField.Content.ReadAsStringAsync()).Contains("invalid_json", StringComparison.Ordinal),
            "contract-v2 command JSON must reject unknown envelope fields before execution");
    }

    using (var wrongCaseCommandField = await SendBridgeRequestAsync(
               httpClient,
               HttpMethod.Post,
               "v2/commands",
               BridgeRuntime.SessionToken,
               new
               {
                   Request_Id = Guid.NewGuid().ToString("D"),
                   session_id = BridgeRuntime.SessionId,
                   capability = BridgeProtocolV2.PlayerControlCapability,
                   expected_state_version = BridgeGameApi.CurrentStateVersion,
                   deadline_utc = DateTimeOffset.UtcNow.AddSeconds(30),
                   command = new
                   {
                       kind = BridgeProtocolV2.PerformActionCommand,
                       action_handle = "end_turn"
                   }
               }))
    {
        Assert(wrongCaseCommandField.StatusCode == HttpStatusCode.BadRequest &&
               (await wrongCaseCommandField.Content.ReadAsStringAsync()).Contains("invalid_json", StringComparison.Ordinal),
            "contract-v2 command JSON property names must be case-sensitive");
    }

    using (var unknownNestedCommandField = await SendBridgeRequestAsync(
               httpClient,
               HttpMethod.Post,
               "v2/commands",
               BridgeRuntime.SessionToken,
               new
               {
                   request_id = Guid.NewGuid().ToString("D"),
                   session_id = BridgeRuntime.SessionId,
                   capability = BridgeProtocolV2.PlayerControlCapability,
                   expected_state_version = BridgeGameApi.CurrentStateVersion,
                   deadline_utc = DateTimeOffset.UtcNow.AddSeconds(30),
                   command = new
                   {
                       kind = BridgeProtocolV2.PerformActionCommand,
                       action_handle = "end_turn",
                       action_id = "legacy-alias-must-fail"
                   }
               }))
    {
        Assert(unknownNestedCommandField.StatusCode == HttpStatusCode.BadRequest &&
               (await unknownNestedCommandField.Content.ReadAsStringAsync()).Contains("invalid_json", StringComparison.Ordinal),
            "contract-v2 command JSON must reject unknown nested fields and legacy aliases");
    }

    using (var unknownResetField = await SendBridgeRequestAsync(
               httpClient,
               HttpMethod.Post,
               "v2/env/reset",
               BridgeRuntime.TrainingSessionToken,
               new
               {
                   request_id = Guid.NewGuid().ToString("D"),
                   session_id = BridgeRuntime.SessionId,
                   scenario = "full-run",
                   expected_state_version = BridgeGameApi.CurrentStateVersion,
                   unexpected_field = true
               }))
    {
        Assert(unknownResetField.StatusCode == HttpStatusCode.BadRequest &&
               (await unknownResetField.Content.ReadAsStringAsync()).Contains("invalid_json", StringComparison.Ordinal),
            "all typed contract-v2 request envelopes must use the strict JSON options");
    }

    using (var playerOwnStatus = await SendBridgeRequestAsync(
               httpClient, HttpMethod.Get, $"v2/commands/{commandRequestId}", BridgeRuntime.SessionToken))
    {
        Assert(playerOwnStatus.StatusCode == HttpStatusCode.OK,
            "player scope must query its own command result");
    }
    using (var trainingReadsPlayer = await SendBridgeRequestAsync(
               httpClient, HttpMethod.Get, $"v2/commands/{commandRequestId}", BridgeRuntime.TrainingSessionToken))
    {
        Assert(trainingReadsPlayer.StatusCode == HttpStatusCode.Forbidden,
            "training scope must not query a player command result");
    }
    using (var trainingOwnStatus = await SendBridgeRequestAsync(
               httpClient, HttpMethod.Get, $"v2/commands/{resetRequestId}", BridgeRuntime.TrainingSessionToken))
    {
        Assert(trainingOwnStatus.StatusCode == HttpStatusCode.OK,
            "training scope must query its own environment command result");
    }
    using (var playerReadsTraining = await SendBridgeRequestAsync(
               httpClient, HttpMethod.Get, $"v2/commands/{resetRequestId}", BridgeRuntime.SessionToken))
    {
        Assert(playerReadsTraining.StatusCode == HttpStatusCode.Forbidden,
            "player scope must not query a training command result");
    }

    BridgeRuntime.LegacyV1Enabled = true;
    using (var legacyReadsCommandStatus = await SendBridgeRequestAsync(
               httpClient, HttpMethod.Get, $"v2/commands/{commandRequestId}", BridgeRuntime.LegacySessionToken))
    {
        Assert(legacyReadsCommandStatus.StatusCode == HttpStatusCode.Forbidden,
            "legacy-privileged must not enter the contract-v2 command-status surface");
    }
    using (var playerReadsLegacy = await SendBridgeRequestAsync(
               httpClient, HttpMethod.Get, "env/spec", BridgeRuntime.SessionToken))
    {
        Assert(playerReadsLegacy.StatusCode == HttpStatusCode.Forbidden,
            "player-control token must never authorize legacy /env/*");
    }
    using (var playerMutatesLegacyEnv = await SendBridgeRequestAsync(
               httpClient,
               HttpMethod.Post,
               "env/reset",
               BridgeRuntime.SessionToken,
               new { }))
    {
        Assert(playerMutatesLegacyEnv.StatusCode == HttpStatusCode.Forbidden,
            "player-control token must never invoke legacy /env/* mutations");
    }
    using (var legacyReadsLegacy = await SendBridgeRequestAsync(
               httpClient, HttpMethod.Get, "env/spec", BridgeRuntime.LegacySessionToken))
    {
        Assert(legacyReadsLegacy.StatusCode == HttpStatusCode.OK,
            "explicit legacy mode must accept only the legacy-privileged token");
    }
    using (var legacyHealth = await SendBridgeRequestAsync(
               httpClient, HttpMethod.Get, "v2/health", BridgeRuntime.LegacySessionToken))
    {
        Assert(legacyHealth.StatusCode == HttpStatusCode.OK,
            "an explicitly published legacy-privileged scope may read detailed health");
    }
    using (var playerStaticExport = await SendBridgeRequestAsync(
               httpClient,
               HttpMethod.Post,
               "static/export",
               BridgeRuntime.SessionToken,
               new { }))
    {
        Assert(playerStaticExport.StatusCode == HttpStatusCode.Forbidden,
            "player-control must not authorize any legacy endpoint");
    }
    using (var retiredStaticExport = await SendBridgeRequestAsync(
               httpClient,
               HttpMethod.Post,
               "static/export",
               BridgeRuntime.LegacySessionToken,
               new { }))
    {
        Assert(retiredStaticExport.StatusCode == HttpStatusCode.Gone,
            "explicit legacy token should receive the static-export retirement response");
        var gonePayload = await retiredStaticExport.Content.ReadAsStringAsync();
        Assert(gonePayload.Contains("tools/catalog-export", StringComparison.Ordinal),
            "410 response must point callers at the offline catalog CLI");
    }
}
finally
{
    BridgeRuntime.LegacyV1Enabled = false;
    await BridgeServer.StopAsync();
}
Assert(!BridgeServer.IsRunning, "pure Bridge host must stop cleanly after HTTP authorization tests");
Console.WriteLine("Bridge HTTP capability tests passed.");

static DirectoryInfo FindRepositoryRoot()
{
    var current = new DirectoryInfo(Directory.GetCurrentDirectory());
    while (current is not null)
    {
        if (File.Exists(Path.Combine(current.FullName, "mods", "sts2-bridge", "sts2-bridge.csproj")))
        {
            return current;
        }
        current = current.Parent;
    }
    throw new InvalidOperationException("Could not locate repository root for source-boundary tests.");
}

var repositoryRoot = FindRepositoryRoot().FullName;
var bridgeRoot = Path.Combine(repositoryRoot, "mods", "sts2-bridge");
var contractFixtureRoot = Path.Combine(repositoryRoot, "contracts", "fixtures");
using (var acceptedFixtureDocument = JsonDocument.Parse(
           File.ReadAllText(Path.Combine(contractFixtureRoot, "command-result.accepted.json"))))
{
    AssertCommandWireMatchesFixture(acceptedWire, acceptedFixtureDocument.RootElement, "accepted");
}
using (var executingFixtureDocument = JsonDocument.Parse(
           File.ReadAllText(Path.Combine(contractFixtureRoot, "command-result.executing.json"))))
{
    AssertCommandWireMatchesFixture(executingWire, executingFixtureDocument.RootElement, "executing");
}
using (var rejectedFixtureDocument = JsonDocument.Parse(
           File.ReadAllText(Path.Combine(contractFixtureRoot, "command-result.rejected.json"))))
{
    AssertCommandWireMatchesFixture(rejectedPlayerWire, rejectedFixtureDocument.RootElement, "rejected_before_execution");
}

using (var stateFixtureDocument = JsonDocument.Parse(
           File.ReadAllText(Path.Combine(contractFixtureRoot, "state.player-control.json"))))
{
    var actions = stateFixtureDocument.RootElement.GetProperty("legal_actions");
    var mapFixture = actions[0];
    var projectedMap = JsonSerializer.SerializeToElement(
        BridgeLegalActionProjector.Project(
            "map:1,2",
            new
            {
                kind = "map",
                label = "Travel to (1, 2)",
                coord = new { col = 1, row = 2 }
            }));
    Assert(projectedMap.GetProperty("handle").GetString() == mapFixture.GetProperty("handle").GetString() &&
           projectedMap.GetProperty("kind").GetString() == mapFixture.GetProperty("kind").GetString() &&
           projectedMap.GetProperty("coord").GetProperty("x").GetInt32() == mapFixture.GetProperty("coord").GetProperty("x").GetInt32() &&
           projectedMap.GetProperty("coord").GetProperty("y").GetInt32() == mapFixture.GetProperty("coord").GetProperty("y").GetInt32(),
        "production legal-action projector must match the real map action fixture");

    var eventFixture = actions[1];
    var projectedEvent = JsonSerializer.SerializeToElement(
        BridgeLegalActionProjector.Project(
            "event_option:0",
            new
            {
                kind = "event_option",
                label = "Choose option 1",
                index = 0
            }));
    Assert(projectedEvent.GetProperty("handle").GetString() == eventFixture.GetProperty("handle").GetString() &&
           projectedEvent.GetProperty("kind").GetString() == eventFixture.GetProperty("kind").GetString() &&
           projectedEvent.GetProperty("option_index").GetInt32() == eventFixture.GetProperty("option_index").GetInt32() &&
           projectedEvent.GetProperty("selection_id").ValueKind == eventFixture.GetProperty("selection_id").ValueKind,
        "production legal-action projector must match the real event-option fixture");
}

var projectedPlayerState = JsonSerializer.SerializeToElement(
    BridgePlayerStateProjector.Project(new
    {
        ok = true,
        state_hash = "hidden-hash-oracle",
        semantic_state_hash = "hidden-semantic-hash-oracle",
        screen = "combat",
        run = new { active = true },
        combat = new
        {
            enemies = new[]
            {
                new
                {
                    name = "test-enemy",
                    danger_profile = new { lethal = true },
                    target_priority_hints = new[] { "focus-first" }
                }
            }
        },
        players = new[]
        {
            new
            {
                combat = new
                {
                    draw_pile = new
                    {
                        count = 2,
                        cards = new[] { new { id = "secret-a" }, new { id = "secret-b" } }
                    }
                },
                potion = new
                {
                    id = "test-potion",
                    training_tags = new[] { "policy-only" },
                    enabled_for_training = true
                }
            }
        },
        available_actions = new[] { new { action_id = "legacy" } },
        unexpected_future_root = new { leaked = true }
    }));
Assert(projectedPlayerState.TryGetProperty("screen", out _) &&
       projectedPlayerState.TryGetProperty("run", out _) &&
       projectedPlayerState.TryGetProperty("combat", out _) &&
       projectedPlayerState.TryGetProperty("players", out _) &&
       !projectedPlayerState.TryGetProperty("ok", out _) &&
       !projectedPlayerState.TryGetProperty("state_hash", out _) &&
       !projectedPlayerState.TryGetProperty("semantic_state_hash", out _) &&
       !projectedPlayerState.TryGetProperty("available_actions", out _) &&
       !projectedPlayerState.TryGetProperty("unexpected_future_root", out _),
    "player state projection must be a root allowlist rather than a legacy-payload blacklist");
var projectedEnemy = projectedPlayerState.GetProperty("combat").GetProperty("enemies")[0];
Assert(!projectedEnemy.TryGetProperty("danger_profile", out _) &&
       !projectedEnemy.TryGetProperty("target_priority_hints", out _),
    "player state projection must recursively remove policy hints");
var projectedDrawPile = projectedPlayerState.GetProperty("players")[0]
    .GetProperty("combat")
    .GetProperty("draw_pile");
Assert(projectedDrawPile.GetProperty("count").GetInt32() == 2 &&
       projectedDrawPile.GetProperty("cards").ValueKind == JsonValueKind.Null &&
       !projectedDrawPile.GetProperty("cards_visible").GetBoolean() &&
       !projectedDrawPile.GetProperty("order_visible").GetBoolean(),
    "player state projection must recursively redact hidden draw-pile composition and order");
var projectedPotion = projectedPlayerState.GetProperty("players")[0].GetProperty("potion");
Assert(!projectedPotion.TryGetProperty("training_tags", out _) &&
       !projectedPotion.TryGetProperty("enabled_for_training", out _),
    "player state projection must recursively remove training-only annotations");

using (var eventFixtureDocument = JsonDocument.Parse(
           File.ReadAllText(Path.Combine(contractFixtureRoot, "event.frontier.json"))))
{
    var eventFixture = eventFixtureDocument.RootElement;
    Assert(eventFixture.GetProperty("event_type").GetString() == "frontier" &&
           eventFixture.GetProperty("visibility").GetString() == "player" &&
           eventFixture.GetProperty("state_version").GetInt64() == 42 &&
           !eventFixture.TryGetProperty("state", out _) &&
           !eventFixture.TryGetProperty("legal_actions", out _),
        "contract-v2 frontier events must be bounded revision notifications, not state snapshots");
}
using (var environmentFixtureDocument = JsonDocument.Parse(
           File.ReadAllText(Path.Combine(contractFixtureRoot, "environment.result.json"))))
{
    var fixtureAction = environmentFixtureDocument.RootElement.GetProperty("legal_actions")[0];
    var legacyAction = new
    {
        idx = 0,
        action_id = "reward:continue",
        kind = "continue",
        model_action_kind = "reward",
        label = "Continue",
        selection = "proceed",
        diagnostic = new { retained = true }
    };
    var legacyActionBefore = JsonSerializer.SerializeToElement(legacyAction);
    var projectedAction = JsonSerializer.SerializeToElement(
        BridgeLegalActionProjector.ProjectEnvironmentAction(legacyAction));
    Assert(projectedAction.GetProperty("action_handle").GetString() ==
               fixtureAction.GetProperty("action_handle").GetString() &&
           !projectedAction.TryGetProperty("action_id", out _) &&
           projectedAction.GetProperty("model_action_kind").GetString() ==
               fixtureAction.GetProperty("model_action_kind").GetString() &&
           projectedAction.GetProperty("diagnostic").GetProperty("retained").GetBoolean() &&
           legacyActionBefore.GetProperty("action_id").GetString() == "reward:continue",
        "training legal-action projection must preserve metadata, avoid mutation, and emit action_handle only");

    var liveEventAction = new
    {
        idx = 1,
        action_id = "event_option:1",
        kind = "event_option",
        model_action_kind = "event_option",
        option = new
        {
            index = 1,
            title = "Visible choice",
            description = "Visible description",
            is_locked = false,
            is_proceed = false
        }
    };
    var projectedEventAction = JsonSerializer.SerializeToElement(
        BridgeLegalActionProjector.ProjectEnvironmentAction(liveEventAction));
    var projectedEventOption = projectedEventAction.GetProperty("option");
    Assert(projectedEventAction.GetProperty("model_action_kind").GetString() == "event_option" &&
           projectedEventOption.GetProperty("title").GetString() == "Visible choice" &&
           projectedEventOption.GetProperty("description").GetString() == "Visible description" &&
           !projectedEventOption.GetProperty("is_locked").GetBoolean() &&
           !projectedEventOption.GetProperty("is_proceed").GetBoolean(),
        "live event legal actions must preserve a nested reviewed option DTO for model parity");

    var projectedRunModeAction = JsonSerializer.SerializeToElement(
        BridgeLegalActionProjector.ProjectEnvironmentAction(new
        {
            idx = 2,
            action_id = "run_mode:standard",
            kind = "run_mode_selection",
            model_action_kind = "run_mode_selection",
            run_mode_action = "standard"
        }));
    Assert(projectedRunModeAction.GetProperty("run_mode_action").GetString() == "standard" &&
           !projectedRunModeAction.TryGetProperty("run_mode", out _),
        "live run-mode legal actions must preserve the canonical run_mode_action field");
}
Console.WriteLine("Bridge shared wire fixtures passed.");

var mainApiPath = Path.Combine(bridgeRoot, "Scripts", "BridgeGameApi.cs");
var mainApiSource = File.ReadAllText(mainApiPath);
var serverSource = File.ReadAllText(Path.Combine(bridgeRoot, "Scripts", "BridgeServer.cs"));
var commandSource = File.ReadAllText(Path.Combine(bridgeRoot, "Scripts", "BridgeCommandService.cs"));
var commandStoreSource = File.ReadAllText(Path.Combine(bridgeRoot, "Scripts", "BridgeCommandStore.cs"));
var v2Source = File.ReadAllText(Path.Combine(bridgeRoot, "Scripts", "BridgeGameApi.V2.cs"));
var snapshotsSource = File.ReadAllText(Path.Combine(bridgeRoot, "Scripts", "BridgeGameApi.Snapshots.cs"));
var stateProjectorSource = File.ReadAllText(Path.Combine(bridgeRoot, "Scripts", "BridgePlayerStateProjector.cs"));
var sessionSource = File.ReadAllText(Path.Combine(bridgeRoot, "Scripts", "BridgeSessionRegistry.cs"));
var envSource = File.ReadAllText(Path.Combine(bridgeRoot, "Scripts", "BridgeGameApi.Env.cs"));
var envHelpersSource = File.ReadAllText(Path.Combine(bridgeRoot, "Scripts", "BridgeGameApi.EnvHelpers.cs"));
var envV2Source = File.ReadAllText(Path.Combine(bridgeRoot, "Scripts", "BridgeGameApi.EnvV2.cs"));
var protocolSource = File.ReadAllText(Path.Combine(bridgeRoot, "Scripts", "BridgeProtocolV2.cs"));
var actionRegistrySource = File.ReadAllText(Path.Combine(bridgeRoot, "Scripts", "BridgeGameApi.Actions.Registry.cs"));
var projectorSource = File.ReadAllText(Path.Combine(bridgeRoot, "Scripts", "BridgeLegalActionProjector.cs"));
var glossarySource = File.ReadAllText(Path.Combine(bridgeRoot, "Scripts", "BridgeGameApi.GameAdapter.Glossary.cs"));
var enemyPayloadSource = File.ReadAllText(Path.Combine(bridgeRoot, "Scripts", "BridgeGameApi.Payloads.Enemies.cs"));
var combatPayloadSource = File.ReadAllText(Path.Combine(bridgeRoot, "Scripts", "BridgeGameApi.Payloads.Combat.cs"));
var navigationPayloadSource = File.ReadAllText(Path.Combine(bridgeRoot, "Scripts", "BridgeGameApi.Payloads.Navigation.cs"));
var compactPayloadSource = File.ReadAllText(Path.Combine(bridgeRoot, "Scripts", "BridgeGameApi.EnvCompact.cs"));
var envPayloadsSource = File.ReadAllText(Path.Combine(bridgeRoot, "Scripts", "BridgeGameApi.EnvPayloads.cs"));
var envTextSource = File.ReadAllText(Path.Combine(bridgeRoot, "Scripts", "BridgeGameApi.EnvText.cs"));
var cardFactsSource = File.ReadAllText(Path.Combine(bridgeRoot, "Scripts", "BridgeGameApi.GameAdapter.CardFacts.cs"));
var selectionActionSource = File.ReadAllText(Path.Combine(bridgeRoot, "Scripts", "BridgeGameApi.Actions.SelectionShop.cs"));
var navigationSelectionActionSource = File.ReadAllText(Path.Combine(bridgeRoot, "Scripts", "BridgeGameApi.Actions.NavigationSelection.cs"));
var selectionPayloadSource = File.ReadAllText(Path.Combine(bridgeRoot, "Scripts", "BridgeGameApi.Payloads.Selection.cs"));
var bridgeProjectSource = File.ReadAllText(Path.Combine(bridgeRoot, "sts2-bridge.csproj"));
var entrySource = File.ReadAllText(Path.Combine(bridgeRoot, "Scripts", "Entry.cs"));
var compatibilitySource = File.ReadAllText(
    Path.Combine(bridgeRoot, "Scripts", "BridgeGameCompatibilityGate.cs"));
Assert(File.ReadLines(mainApiPath).Count() < 1600,
    "BridgeGameApi.cs must remain an orchestration-only partial below 1,600 lines");

var extractedPartials = new Dictionary<string, string[]>(StringComparer.Ordinal)
{
    ["BridgeGameApi.Actions.Registry.cs"] = ["private static List<BridgeResolvedAction> BuildResolvedActions("],
    ["BridgeGameApi.Actions.NavigationSelection.cs"] = ["private static void AddTreasureRoomActions(", "private static void AddRestSiteActions(", "private static void AddDeckUpgradeActions("],
    ["BridgeGameApi.Actions.SelectionShop.cs"] = ["private static void AddCardSelectionActions(", "private static void AddShopActions("],
    ["BridgeGameApi.Actions.Menu.cs"] = ["private static void AddMainMenuActions(", "private static void AddGameOverActions("],
    ["BridgeGameApi.Actions.Combat.cs"] = ["private static void AddCombatCardActions(", "private static void ExecutePotionUse("],
    ["BridgeGameApi.Actions.Shop.cs"] = ["private static bool CanPurchaseMerchantEntry(", "private static void ExecuteShopPurchase("],
    ["BridgeGameApi.Actions.Navigation.cs"] = ["private static void InvokeMapTravelAction(", "private static void InvokeCardSelectionOptionAction("],
    ["BridgeGameApi.Payloads.Combat.cs"] = ["private static object BuildCombatPayload(", "private static object BuildCardPayload("],
    ["BridgeGameApi.Payloads.Enemies.cs"] = ["private static object BuildCreaturePayload(", "private static object BuildMonsterIntentPayload("],
    ["BridgeGameApi.Payloads.Selection.cs"] = ["private static object BuildRewardsPayload(", "private static object BuildCrystalSpherePayload("],
    ["BridgeGameApi.Payloads.Navigation.cs"] = ["private static object BuildMapPayload(", "private static object BuildShopPayload("],
    ["BridgeGameApi.GameAdapter.CardFacts.cs"] = ["private static string FirstNonEmptyText(", "private static object[] BuildDynamicVarPayloads("],
    ["BridgeGameApi.GameAdapter.Context.cs"] = ["private static BridgeWorldContext CaptureContext("],
    ["BridgeGameApi.GameAdapter.Glossary.cs"] = ["private static object BuildEventOptionPayload(", "private static object[] BuildVisibleGlossaryPayload("],
    ["BridgeGameApi.GameAdapter.UI.cs"] = ["private static string ResolveCurrentScreen(", "private static List<T> FindVisibleDescendants<T>("],
    ["BridgeGameApi.GameAdapter.Reflection.cs"] = ["private static object? InvokeParameterless(", "private static PropertyInfo? FindProperty(", "private static FieldInfo? FindField("],
    ["BridgeGameApi.GameAdapter.Text.cs"] = ["private static string DescribeText(", "private static string ResolvePlaceholderText("],
    ["BridgeGameApi.EnvHelpers.cs"] = ["private static object BuildEnvActionPayload(", "private static bool TrySetHiddenFieldValue("],
    ["BridgeGameApi.EnvStateTracking.cs"] = ["private static bool HasMeaningfulEnvSnapshotDifference(", "private static IReadOnlyList<BridgeEnvDeckEntry> BuildEnvDeckEntries("],
    ["BridgeGameApi.EnvModels.cs"] = ["private sealed class BridgeEnvEpisode", "private sealed class BridgeEnvSnapshot"],
    ["BridgeGameApi.EnvCombatSandbox.cs"] = ["public static async Task<object> CombatResetEnvResponseAsync(", "private static async Task<object> StepCombatSandboxEpisodeAsync("],
    ["BridgeGameApi.EnvCombatCatalog.cs"] = ["public static async Task<object> GetCombatCatalogResponseAsync(", "private sealed class EncounterCatalogEntry"],
    ["BridgeGameApi.EnvCombatSetup.cs"] = ["private static CombatSandboxSetupResult SetUpCombatSandbox(", "internal static void DrainManagedFinalizersLogged("],
    ["BridgeGameApi.EnvV2.cs"] = ["public static object GetEnvSpecV2Response(", "public static async Task<object> GetEnvStateV2ResponseAsync(", "ValidateEnvironmentResetStateVersionV2Async("],
    ["BridgeGameApi.Snapshots.FrontierPayloads.cs"] = ["private static object BuildCombatFrontierPayload(", "private static object BuildShopFrontierPayload("],
    ["BridgeGameApi.Snapshots.Payloads.cs"] = ["private static JsonNode? PruneSemanticStateNode(", "private static object BuildFrontierStatePayload("],
    ["BridgeGameApi.Snapshots.StatePayloads.cs"] = ["private static BridgeStateFields BuildStateFields(", "private static object CreateStatePayload("],
    ["BridgeGameApi.Models.cs"] = ["private sealed class BridgeWorldContext", "private sealed class BridgeSnapshot"]
};

foreach (var (partialName, requiredMembers) in extractedPartials)
{
    var partialPath = Path.Combine(bridgeRoot, "Scripts", partialName);
    Assert(File.Exists(partialPath), $"required semantic partial is missing: {partialName}");
    Assert(File.ReadLines(partialPath).Count() < 2000,
        $"new semantic partial must remain below 2,000 lines: {partialName}");
    var source = File.ReadAllText(partialPath);
    Assert(source.Contains("internal static partial class BridgeGameApi", StringComparison.Ordinal),
        $"semantic extraction must remain a BridgeGameApi partial: {partialName}");
    foreach (var member in requiredMembers)
    {
        Assert(source.Contains(member, StringComparison.Ordinal),
            $"{partialName} must retain ownership of definition {member}");
    }
}

var emptyCatchPattern = new System.Text.RegularExpressions.Regex(
    @"catch(?:\s*\([^)]*\))?\s*\{\s*\}",
    System.Text.RegularExpressions.RegexOptions.CultureInvariant);
foreach (var sourcePath in Directory.EnumerateFiles(
             Path.Combine(bridgeRoot, "Scripts"), "*.cs", SearchOption.TopDirectoryOnly))
{
    Assert(!emptyCatchPattern.IsMatch(File.ReadAllText(sourcePath)),
        $"Bridge source must not contain a context-free empty catch: {Path.GetFileName(sourcePath)}");
}

Assert(!mainApiSource.Contains("private static void AddCombatCardActions(", StringComparison.Ordinal) &&
       !mainApiSource.Contains("private static void InvokeMapTravelAction(", StringComparison.Ordinal) &&
       !mainApiSource.Contains("private static string ResolvePlaceholderText(", StringComparison.Ordinal) &&
       !mainApiSource.Contains("private static BridgeWorldContext CaptureContext(", StringComparison.Ordinal) &&
       !mainApiSource.Contains("private static BridgeStateFields BuildStateFields(", StringComparison.Ordinal),
    "the BridgeGameApi orchestration partial must not absorb extracted action, adapter, or payload definitions");
Assert(File.Exists(Path.Combine(bridgeRoot, "Scripts", "BridgeGameApi.V2.cs")),
    "v2 projection/actions must live in a dedicated partial");
Assert(File.Exists(Path.Combine(bridgeRoot, "Scripts", "BridgeGameApi.Snapshots.cs")),
    "frontier/SSE ownership must live in the snapshots partial");
Assert(!File.Exists(Path.Combine(bridgeRoot, "Scripts", "BridgeGameApi.StaticExport.cs")),
    "game-process static exporter must stay removed");
Assert(!enemyPayloadSource.Contains("static_traits", StringComparison.Ordinal) &&
       !enemyPayloadSource.Contains("reactive_triggers", StringComparison.Ordinal) &&
       !enemyPayloadSource.Contains("phase_rules", StringComparison.Ordinal) &&
       !enemyPayloadSource.Contains("combat_tags", StringComparison.Ordinal) &&
       !enemyPayloadSource.Contains("danger_profile", StringComparison.Ordinal) &&
       !enemyPayloadSource.Contains("target_priority_hints", StringComparison.Ordinal),
    "enemy payloads must expose runtime state and intent, not hand-written policy annotations");
Assert(!combatPayloadSource.Contains("card_effect_profile", StringComparison.Ordinal) &&
       !combatPayloadSource.Contains("effect_preview", StringComparison.Ordinal) &&
       !combatPayloadSource.Contains("TryGetPotionProfileEntry", StringComparison.Ordinal) &&
       !navigationPayloadSource.Contains("TryGetPotionProfileEntry", StringComparison.Ordinal) &&
       !compactPayloadSource.Contains("card_effect_profile", StringComparison.Ordinal) &&
       !compactPayloadSource.Contains("effect_preview", StringComparison.Ordinal),
    "card and potion payloads must not reintroduce curated effect/timing profiles");
Assert(!combatPayloadSource.Contains("BuildPlayCardSemantic", StringComparison.Ordinal) &&
       !combatPayloadSource.Contains("BuildUsePotionSemantic", StringComparison.Ordinal) &&
       !combatPayloadSource.Contains("modifier_summary", StringComparison.Ordinal) &&
       !combatPayloadSource.Contains("card_flow", StringComparison.Ordinal) &&
       !combatPayloadSource.Contains("semantic_tags", StringComparison.Ordinal) &&
       !combatPayloadSource.Contains("semantic_values", StringComparison.Ordinal) &&
       !combatPayloadSource.Contains("safety = new", StringComparison.Ordinal) &&
       !combatPayloadSource.Contains("x_cost = new", StringComparison.Ordinal) &&
       !combatPayloadSource.Contains("selection = new", StringComparison.Ordinal) &&
       !combatPayloadSource.Contains("will_exhaust", StringComparison.Ordinal) &&
       !compactPayloadSource.Contains("semantic_tags", StringComparison.Ordinal) &&
       !compactPayloadSource.Contains("semantic_values", StringComparison.Ordinal) &&
       combatPayloadSource.Contains("is_debuff = GetHiddenPropertyValue<bool>", StringComparison.Ordinal) &&
       combatPayloadSource.Contains("is_buff = GetHiddenPropertyValue<bool>", StringComparison.Ordinal) &&
       compactPayloadSource.Contains("var amount = TryGetNestedDecimal(modifier, \"amount\")", StringComparison.Ordinal) &&
       !cardFactsSource.Contains("BodySlam", StringComparison.Ordinal) &&
       !cardFactsSource.Contains("strategic", StringComparison.Ordinal) &&
       !File.Exists(Path.Combine(bridgeRoot, "Scripts", "BridgeGameApi.GameAdapter.CardSemantics.cs")),
    "card payloads must expose raw runtime facts without modifier rules, card-flow labels, or card-specific exceptions");
Assert(!File.Exists(Path.Combine(bridgeRoot, "Scripts", "BridgeGameApi.CardEffectProfiles.cs")) &&
       !File.Exists(Path.Combine(bridgeRoot, "Scripts", "BridgeGameApi.PotionProfiles.cs")) &&
       !bridgeProjectSource.Contains("card_effect_profiles.generated.json", StringComparison.Ordinal) &&
       !bridgeProjectSource.Contains("potions.timing", StringComparison.Ordinal),
    "the Bridge must not embed retired card-effect or potion-timing registries");
Assert(!File.Exists(Path.Combine(bridgeRoot, "Scripts", "BridgeGameApi.EnvRoutes.cs")) &&
       !actionRegistrySource.Contains("route_summary", StringComparison.Ordinal) &&
       !envHelpersSource.Contains("route_summary", StringComparison.Ordinal) &&
       !envHelpersSource.Contains("route_nodes", StringComparison.Ordinal) &&
       !compactPayloadSource.Contains("CompactMapRoute", StringComparison.Ordinal),
    "map actions must expose travelable coordinates and point types without route-policy summaries");
Assert(!File.Exists(Path.Combine(bridgeRoot, "Scripts", "BridgeGameApi.EnvEventEffects.cs")) &&
       !glossarySource.Contains("effect_deltas", StringComparison.Ordinal) &&
       !glossarySource.Contains("ExtractEventOptionEffectDeltas", StringComparison.Ordinal) &&
       !envHelpersSource.Contains("effect_deltas", StringComparison.Ordinal) &&
       envHelpersSource.Contains("entry[\"description\"] = TryGetNestedString(payload, \"option\", \"description\")", StringComparison.Ordinal),
    "event options must expose visible game text without regex-derived outcome annotations");
Assert(!File.Exists(Path.Combine(bridgeRoot, "Scripts", "BridgeGameApi.EnvRewardShaping.cs")) &&
       !envHelpersSource.Contains("BuildLegacyEnvRewardBreakdown", StringComparison.Ordinal) &&
       !envHelpersSource.Contains("reward_breakdown", StringComparison.Ordinal) &&
       !envHelpersSource.Contains("action_diagnostics", StringComparison.Ordinal) &&
       envHelpersSource.Contains("reward = (double?)null", StringComparison.Ordinal) &&
       envSource.Contains("scalar_computed_by_bridge = false", StringComparison.Ordinal),
    "the Bridge environment must emit transition facts and never compute legacy scalar reward");
Assert(!envPayloadsSource.Contains("ResolveCardSelectionSemantics", StringComparison.Ordinal) &&
       !envPayloadsSource.Contains("ResolveCardSelectionDomain", StringComparison.Ordinal) &&
       !envPayloadsSource.Contains("selection_semantics", StringComparison.Ordinal) &&
       !envPayloadsSource.Contains("selection_domain", StringComparison.Ordinal) &&
       !envPayloadsSource.Contains("source_effect_type", StringComparison.Ordinal) &&
       !selectionActionSource.Contains("selection_semantics", StringComparison.Ordinal) &&
       !selectionPayloadSource.Contains("selection_semantics", StringComparison.Ordinal) &&
       !envHelpersSource.Contains("selection_semantics", StringComparison.Ordinal) &&
       !envTextSource.Contains("DescribeSelectionSemanticsLabel", StringComparison.Ordinal) &&
       navigationSelectionActionSource.Contains("operation_type = operationType", StringComparison.Ordinal) &&
       selectionActionSource.Contains("\"select\"", StringComparison.Ordinal),
    "card-selection transport must retain typed UI facts without classifying natural-language prompts");
Assert(!mainApiSource.Contains("MaxShopOpenActionsPerRoom", StringComparison.Ordinal) &&
       !mainApiSource.Contains("ShopOpenLimiter", StringComparison.Ordinal) &&
       !cardFactsSource.Contains("CanExposeShopOpenAction", StringComparison.Ordinal) &&
       !cardFactsSource.Contains("RecordShopOpenAction", StringComparison.Ordinal) &&
       !selectionActionSource.Contains("shopOpenAvailable", StringComparison.Ordinal),
    "legal shop-open actions must never be hidden by a strategy-oriented per-room counter");
Assert(!envPayloadsSource.Contains("player_powers", StringComparison.Ordinal) &&
       !envPayloadsSource.Contains("allies =", StringComparison.Ordinal) &&
       !envPayloadsSource.Contains("incoming_damage_multiplier", StringComparison.Ordinal) &&
       !envPayloadsSource.Contains("intent = new", StringComparison.Ordinal) &&
       envPayloadsSource.Contains("players = context.CombatState.PlayerCreatures", StringComparison.Ordinal) &&
       envPayloadsSource.Contains("powers = creature?.Powers.Select(BuildEnvPowerPayload)", StringComparison.Ordinal) &&
       envPayloadsSource.Contains("intents = BuildEnvEnemyIntentPayloads(intentEnvelope)", StringComparison.Ordinal),
    "live environment observations must use the grounded player/powers and enemy-intents structure shared with headless transport");
Assert(envHelpersSource.Contains("[\"model_action_kind\"] = modelActionKind", StringComparison.Ordinal) &&
       envHelpersSource.Contains("return \"end_turn\";", StringComparison.Ordinal),
    "live environment legal actions must expose an exact stable model-facing action kind");
Assert(envHelpersSource.Contains("entry[\"option\"] = CompactEventOptionPayload", StringComparison.Ordinal) &&
       compactPayloadSource.Contains("private static object? CompactEventOptionPayload(", StringComparison.Ordinal) &&
       compactPayloadSource.Contains("is_locked = TryGetNestedBool", StringComparison.Ordinal) &&
       compactPayloadSource.Contains("is_proceed = TryGetNestedBool", StringComparison.Ordinal),
    "live event candidates must retain a nested raw option DTO instead of flattening away candidate identity");
Assert(envHelpersSource.Contains("entry[\"run_mode_action\"] = TryGetNestedString", StringComparison.Ordinal) &&
       !envHelpersSource.Contains("entry[\"run_mode\"]", StringComparison.Ordinal),
    "live run-mode candidates must use the canonical run_mode_action field");
Assert(envHelpersSource.Contains("STS2_FAST_STEP_MAX_WAIT_MS", StringComparison.Ordinal) &&
       !envHelpersSource.Contains("MUZERO_FAST_STEP_MAX_WAIT_MS", StringComparison.Ordinal),
    "Bridge comments and configuration names must not retain the retired MuZero fast-step prefix");
Assert(serverSource.Contains("HttpStatusCode.Gone", StringComparison.Ordinal) &&
       !serverSource.Contains("ExportStaticDataResponseAsync", StringComparison.Ordinal),
    "legacy static export must return 410 and never invoke game-process export");
Assert(commandSource.Contains("reward_status", StringComparison.Ordinal) &&
       commandSource.Contains("external-rl", StringComparison.Ordinal),
    "v2 environment projection must keep reward authority external");
Assert(envV2Source.Contains("new JsonArray(\"idx\", \"action_handle\", \"kind\")", StringComparison.Ordinal) &&
       envV2Source.Contains("BridgeLegalActionProjector.ProjectEnvironmentActions(snapshot.LegalActions)", StringComparison.Ordinal) &&
       commandSource.Contains("BridgeLegalActionProjector.ProjectEnvironmentActions(", StringComparison.Ordinal) &&
       projectorSource.Contains("result[\"action_handle\"] = actionHandle", StringComparison.Ordinal) &&
       envHelpersSource.Contains("[\"action_id\"] = action.ActionId", StringComparison.Ordinal),
    "v2 environment egress must project action_handle while the legacy environment builder stays action_id-based");
Assert(commandSource.Contains("state_version_before", StringComparison.Ordinal) &&
       commandSource.Contains("state_version_after", StringComparison.Ordinal) &&
       !commandSource.Contains("BuildCommittedResult(entry, startedAtUtc.Value, payload, \"step_index\")", StringComparison.Ordinal),
    "environment command results must use real frontier revisions, never step_index as state version");
Assert(envHelpersSource.Contains("revision_status = \"legacy-v1-unavailable\"", StringComparison.Ordinal) &&
       !envHelpersSource.Contains("before_state_version = beforeVersion", StringComparison.Ordinal) &&
       !envHelpersSource.Contains("after_state_version = stepIndex", StringComparison.Ordinal),
    "legacy transitions must mark revisions unavailable instead of mislabeling episode indexes");
Assert(!protocolSource.Contains("LegacyActionId", StringComparison.Ordinal) &&
       !protocolSource.Contains("RequestedActionHandle", StringComparison.Ordinal) &&
       commandSource.Contains("envelope.Command.ActionHandle", StringComparison.Ordinal),
    "contract-v2 commands must accept only canonical action_handle; legacy /action remains a separate DTO");
Assert(serverSource.Contains("JsonUnmappedMemberHandling.Disallow", StringComparison.Ordinal) &&
       serverSource.Contains("PropertyNameCaseInsensitive = false", StringComparison.Ordinal) &&
       serverSource.Contains("ReadJsonV2Async<BridgeCommandEnvelopeV2>", StringComparison.Ordinal) &&
       serverSource.Contains("ReadJsonV2Async<BridgeEnvResetEnvelopeV2>", StringComparison.Ordinal) &&
       serverSource.Contains("ReadJsonV2Async<BridgeEnvStepEnvelopeV2>", StringComparison.Ordinal),
    "typed contract-v2 request DTOs must fail closed on unknown fields and property casing");

Assert(commandStoreSource.Contains("Never evict an unexpired completed identity", StringComparison.Ordinal) &&
       !commandStoreSource.Contains("oldestCompleted", StringComparison.Ordinal),
    "command store must reject capacity rather than evict an unexpired idempotency identity");
Assert(v2Source.Contains("BridgePlayerStateProjector.Project(", StringComparison.Ordinal) &&
        stateProjectorSource.Contains("PlayerStateRootFields", StringComparison.Ordinal) &&
        stateProjectorSource.Contains("ForbiddenRecursiveFields", StringComparison.Ordinal) &&
        stateProjectorSource.Contains("target_priority_hints", StringComparison.Ordinal) &&
        stateProjectorSource.Contains("training_tags", StringComparison.Ordinal) &&
        !v2Source.Contains("current_state_hash", StringComparison.Ordinal) &&
        !v2Source.Contains("available_actions = before.Snapshot.ActionPayloads", StringComparison.Ordinal) &&
        !v2Source.Contains("MaybeAutoProceedAfterRewardActionAsync", StringComparison.Ordinal) &&
        !v2Source.Contains("MaybeAutoCompleteCardSelectionAsync", StringComparison.Ordinal) &&
        !v2Source.Contains("auto_executed_actions", StringComparison.Ordinal) &&
        !v2Source.Contains("BuildResolvedActionAckPayload", StringComparison.Ordinal),
    "v2 player control must expose no legacy action/hash payloads and execute only the requested action");
Assert(snapshotsSource.Contains("BuildFrontierEventV2Payload(frontier)", StringComparison.Ordinal) &&
       snapshotsSource.Contains("state_version = frontier.Sequence", StringComparison.Ordinal) &&
       snapshotsSource.Contains("schema_version = BridgeProtocolV2.SchemaVersion", StringComparison.Ordinal) &&
       !snapshotsSource.Contains("? BuildStateV2Payload(frontier)", StringComparison.Ordinal),
    "v2 SSE must emit bounded revision notifications instead of repeated full state snapshots");
Assert(v2Source.Contains("BridgeLegalActionProjector.Project(action.ActionId, action.Payload)", StringComparison.Ordinal) &&
       projectorSource.Contains("coord = BuildCoordinatePayload(element)", StringComparison.Ordinal) &&
       projectorSource.Contains("option_index = ReadInt(\"option_index\") ?? ReadInt(\"index\")", StringComparison.Ordinal) &&
       !projectorSource.Contains("legacy_action", StringComparison.Ordinal) &&
       !File.ReadAllText(Path.Combine(repositoryRoot, "contracts", "fixtures", "state.player-control.json"))
           .Contains("legacy_action", StringComparison.Ordinal),
    "live v2 state must use the fixture-tested canonical legal-action projector");
Assert(actionRegistrySource.Contains("var actionId = $\"map:{coord.col},{coord.row}\";", StringComparison.Ordinal) &&
       actionRegistrySource.Contains("var actionId = $\"event_option:{index}\";", StringComparison.Ordinal),
    "shared state fixture handles must remain aligned with real map/event action builders");
Assert(!File.Exists(Path.Combine(bridgeRoot, "Scripts", "BridgeAutoSlay.cs")) &&
       !Directory.EnumerateFiles(Path.Combine(bridgeRoot, "Scripts"), "*.cs")
           .Any(path =>
           {
               var source = File.ReadAllText(path);
               return source.Contains("BridgeAutoSlay", StringComparison.Ordinal) ||
                      source.Contains("automation:start_autoslay", StringComparison.Ordinal) ||
                      source.Contains("automation:stop_autoslay", StringComparison.Ordinal);
           }),
    "AutoSlay implementation, action registration, and player payload status must stay removed");
Assert(serverSource.Contains("ResolveAnyScopedCapability(context.Request)", StringComparison.Ordinal) &&
       serverSource.Contains("ResolveCommandStatusCapability(context.Request)", StringComparison.Ordinal) &&
       serverSource.Contains("/v2/env/spec", StringComparison.Ordinal) &&
       serverSource.Contains("/v2/env/state", StringComparison.Ordinal) &&
       serverSource.Contains("/v2/env/combat_catalog", StringComparison.Ordinal),
    "authenticated health and all training-scoped v2 read routes must remain registered");
Assert(sessionSource.Contains("if (BridgeRuntime.LegacyV1Enabled)", StringComparison.Ordinal) &&
       sessionSource.Contains("payload[\"token\"] = BridgeRuntime.LegacySessionToken", StringComparison.Ordinal),
    "top-level session token must remain conditional and legacy-only");
var compatibilityDecisionIndex = entrySource.IndexOf("if (!compatibility.StartupAllowed)", StringComparison.Ordinal);
var patchActivationIndex = entrySource.IndexOf("_harmony.PatchAll()", StringComparison.Ordinal);
var httpActivationIndex = entrySource.IndexOf("BridgeServer.Start()", StringComparison.Ordinal);
Assert(compatibilityDecisionIndex >= 0 &&
       patchActivationIndex > compatibilityDecisionIndex &&
       httpActivationIndex > patchActivationIndex &&
       entrySource.Contains("http_started=false, mutation_enabled=false", StringComparison.Ordinal),
    "Entry must deny unknown/failed game profiles before Harmony patches or the mutation HTTP host start");
Assert(compatibilitySource.Contains("interface IGameAssemblyCompatibilityProfile", StringComparison.Ordinal) &&
       compatibilitySource.Contains("InformationalVersion", StringComparison.Ordinal) &&
       compatibilitySource.Contains("ModuleVersionId", StringComparison.Ordinal) &&
       compatibilitySource.Contains("required_game_capability_missing", StringComparison.Ordinal) &&
       sessionSource.Contains("var gameCompatibility = BridgeGameCompatibilityState.Current", StringComparison.Ordinal) &&
       sessionSource.Contains("gameCompatibility.ToDiagnosticPayload()", StringComparison.Ordinal),
    "retail compatibility identity, profile, probe results, and descriptor diagnostics must stay centralized");
using (var modManifest = JsonDocument.Parse(File.ReadAllText(Path.Combine(bridgeRoot, "sts2-bridge.json"))))
{
    Assert(modManifest.RootElement.GetProperty("affects_gameplay").GetBoolean(),
        "manifest must declare gameplay effects");
    Assert(!modManifest.RootElement.GetProperty("visible_only").GetBoolean(),
        "manifest must not claim the entire privileged Bridge is visible-only");
}
Console.WriteLine("Bridge source-boundary tests passed.");
