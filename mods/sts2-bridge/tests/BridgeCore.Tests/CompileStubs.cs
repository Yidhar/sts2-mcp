using System.Net;

namespace MegaCrit.Sts2.Core.Logging
{
    internal static class Log
    {
        public static void Info(string message) { }
        public static void Warn(string message) { }
        public static void Error(string message) { }
    }
}

namespace Sts2McpBridge.Scripts
{
    internal static class BridgeRuntime
    {
        public const string ModId = "test";
        public const string BridgeName = "test";
        public const string BridgeVersion = "test";
        public const string StateSchemaVersion = "test";
        public const bool VisibleOnly = true;
        public static int PreferredPort => 27100;
        public static int MaxPort => 27109;
        public static int Port { get; private set; } = 27100;
        public static string BaseUrl => $"http://127.0.0.1:{Port}/";
        public static string SessionId => "0123456789abcdef";
        public static string SessionToken => new('a', 32);
        public static string TrainingSessionToken => new('b', 32);
        public static string LegacySessionToken => new('c', 32);
        public static bool TrainingV2Enabled => true;
        public static bool LegacyV1Enabled { get; set; }
        public static int GameThreadHeartbeatTimeoutMs => 5000;
        public static DateTimeOffset StartedAtUtc => DateTimeOffset.UtcNow;
        public static DateTimeOffset ProcessStartedAtUtc => DateTimeOffset.UtcNow;
        public static int ProcessId => Environment.ProcessId;
        public static string GameAssemblyVersion => "test";
        public static string SessionDirectoryPath => Path.GetTempPath();
        public static string SessionFilePath => Path.Combine(SessionDirectoryPath, "sts2-bridge-test-session.json");
        public static string[] ApiVersions => LegacyV1Enabled
            ? [BridgeProtocolV2.ApiVersion, "legacy-v1"]
            : [BridgeProtocolV2.ApiVersion];
        public static string[] EnabledCapabilities => LegacyV1Enabled
            ? [BridgeProtocolV2.PlayerControlCapability, BridgeProtocolV2.TrainingCapability, BridgeProtocolV2.LegacyPrivilegedCapability]
            : [BridgeProtocolV2.PlayerControlCapability, BridgeProtocolV2.TrainingCapability];
        public static bool TryValidateConfiguration(out string error) { error = string.Empty; return true; }
        public static void SetPort(int port) => Port = port;
    }

    internal static class BridgeCoordinator
    {
        public static bool IsReady => true;
        public static long PumpTick => 1;
        public static long MillisecondsSinceLastPump => 0;
        public static int QueueDepth => 0;
        public static object GetDiagnosticsSnapshot() => new { };
    }

    internal static class BridgeDebugTrace
    {
        public static void Write(string message) { }
    }

    internal sealed class BridgeRequestException : Exception
    {
        public BridgeRequestException(HttpStatusCode statusCode, string errorCode, string message, object? details = null)
            : base(message)
        {
            StatusCode = statusCode;
            ErrorCode = errorCode;
            Details = details;
        }
        public HttpStatusCode StatusCode { get; }
        public string ErrorCode { get; }
        public object? Details { get; }
    }

    internal sealed class BridgeActionRequest { }
    internal sealed class BridgeEnvResetRequest { public string? Seed { get; set; } }
    internal sealed class BridgeEnvCombatResetRequest { public int? Seed { get; set; } }
    internal sealed class BridgeEnvStepRequest
    {
        public string? EpisodeId { get; set; }
        public int? ActionIndex { get; set; }
        public string? ActionId { get; set; }
        public int? TimeoutMs { get; set; }
    }

    internal static class BridgeGameApi
    {
        public static int ActionExecutionCount;
        public static int EnvironmentOperationCount;
        public static long CurrentStateVersion = 7;

        public static long? MillisecondsSinceLastSnapshot => 0;
        public static Task<object> GetStateResponseAsync(CancellationToken token) => Task.FromResult<object>(new { });
        public static Task<object> GetStateV2ResponseAsync(CancellationToken token) => Task.FromResult<object>(new { });
        public static Task<object> PerformActionResponseAsync(BridgeActionRequest request, CancellationToken token) => Task.FromResult<object>(new { });
        public static Task<object> PerformActionV2AtomicAsync(string handle, long version, int? waitMs, CancellationToken token)
        {
            if (handle == "test:reject")
            {
                throw new BridgeRequestException(
                    HttpStatusCode.Conflict,
                    "state_version_conflict",
                    "test pre-mutation rejection",
                    new { expected_state_version = version, current_state_version = CurrentStateVersion });
            }
            Interlocked.Increment(ref ActionExecutionCount);
            CurrentStateVersion = version + 1;
            return Task.FromResult<object>(new { state_version_after = CurrentStateVersion });
        }
        public static Task StreamFrontierEventsAsync(HttpListenerResponse response, CancellationToken token) => Task.CompletedTask;
        public static Task StreamFrontierEventsV2Async(HttpListenerResponse response, CancellationToken token) => Task.CompletedTask;
        public static object GetEnvSpecResponse() => new { };
        public static object GetEnvSpecV2Response() => new
        {
            ok = true,
            api_version = BridgeProtocolV2.ApiVersion,
            schema_version = BridgeProtocolV2.SchemaVersion,
            capability = BridgeProtocolV2.TrainingCapability,
            env_api_version = "test",
            scenarios = new[] { "full-run", "combat" },
            action_encoding = new
            {
                default_encoding = "legal_action_idx",
                supported = new[] { "legal_action_idx", "action_handle" },
                legal_action_shape = new[] { "idx", "action_handle", "kind" }
            }
        };
        public static Task<object> GetEnvStateV2ResponseAsync(CancellationToken token) => Task.FromResult<object>(new
        {
            ok = true,
            api_version = BridgeProtocolV2.ApiVersion,
            schema_version = BridgeProtocolV2.SchemaVersion,
            capability = BridgeProtocolV2.TrainingCapability,
            state_version = CurrentStateVersion,
            phase = "combat",
            screen = "combat",
            actionable = true,
            terminated = false,
            observation = new { phase = "combat" },
            legal_actions = new object[]
            {
                new
                {
                    idx = 0,
                    action_handle = "end_turn",
                    kind = "end_turn",
                    label = "End Turn",
                    diagnostic = new { retained = true }
                }
            }
        });
        public static Task<object> GetCombatCatalogV2ResponseAsync(CancellationToken token) => Task.FromResult<object>(new
        {
            ok = true,
            api_version = BridgeProtocolV2.ApiVersion,
            schema_version = BridgeProtocolV2.SchemaVersion,
            capability = BridgeProtocolV2.TrainingCapability,
            encounters = Array.Empty<object>()
        });
        public static Task<long> GetCurrentStateVersionV2Async(CancellationToken token) =>
            Task.FromResult(CurrentStateVersion);
        public static Task<long> ValidateEnvironmentResetStateVersionV2Async(long expected, CancellationToken token)
        {
            if (expected != CurrentStateVersion)
            {
                throw new BridgeRequestException(
                    HttpStatusCode.Conflict,
                    "state_version_conflict",
                    "revision mismatch");
            }
            return Task.FromResult(CurrentStateVersion);
        }
        public static Task<object> ResetEnvResponseAsync(BridgeEnvResetRequest request, CancellationToken token)
        {
            Interlocked.Increment(ref EnvironmentOperationCount);
            CurrentStateVersion++;
            return Task.FromResult<object>(new
            {
                episode_id = "episode",
                step_index = 0,
                done = false,
                truncated = false,
                obs = new { phase = "reset" },
                legal_actions = new object[]
                {
                    new
                    {
                        idx = 0,
                        action_id = "end_turn",
                        kind = "end_turn",
                        label = "End Turn",
                        energy_cost = 0,
                        target = new { side = "player" }
                    }
                },
                reward = 99.0,
                info = new { reward_breakdown = new { total = 99.0 } },
                transition = new
                {
                    episode_id = "episode",
                    step_index = 0,
                    before_state_version = 0,
                    after_state_version = 0,
                    facts = new { hp_delta = 0, gold_delta = 0, floor_delta = 0, combat_result = "none" }
                }
            });
        }
        public static Task<object> CombatResetEnvResponseAsync(BridgeEnvCombatResetRequest request, CancellationToken token)
        {
            Interlocked.Increment(ref EnvironmentOperationCount);
            CurrentStateVersion++;
            return Task.FromResult<object>(new
            {
                episode_id = "combat",
                step_index = 0,
                done = false,
                truncated = false,
                obs = new { phase = "combat" },
                legal_actions = Array.Empty<object>(),
                reward = 99.0,
                info = new { reward_breakdown = new { total = 99.0 } }
            });
        }
        public static Task<object> GetCombatCatalogResponseAsync(CancellationToken token) => Task.FromResult<object>(new { });
        public static Task<object> StepEnvResponseAsync(BridgeEnvStepRequest request, CancellationToken token)
        {
            Interlocked.Increment(ref EnvironmentOperationCount);
            CurrentStateVersion++;
            if (request.ActionId == "test:error-details")
            {
                throw new BridgeRequestException(
                    HttpStatusCode.Conflict,
                    "action_not_available",
                    "test environment action is unavailable",
                    new
                    {
                        action_id = request.ActionId,
                        legal_actions = new object[]
                        {
                            new
                            {
                                idx = 0,
                                action_id = "end_turn",
                                kind = "end_turn",
                                label = "End Turn",
                                energy_cost = 0
                            }
                        }
                    });
            }
            return Task.FromResult<object>(new
            {
                episode_id = request.EpisodeId,
                step_index = 1,
                done = false,
                truncated = false,
                obs = new { phase = "combat" },
                legal_actions = new object[]
                {
                    new
                    {
                        idx = 0,
                        action_id = "reward:continue",
                        kind = "continue",
                        label = "Continue",
                        selection = "proceed"
                    }
                },
                reward = 99.0,
                info = new { reward_breakdown = new { total = 99.0 } },
                transition = new
                {
                    episode_id = request.EpisodeId,
                    step_index = 1,
                    before_state_version = 0,
                    after_state_version = 1,
                    facts = new { hp_delta = -2, gold_delta = 0, floor_delta = 0, combat_result = "none" }
                }
            });
        }
        public static void ValidateEnvEpisodeStepV2(string episodeId, int expectedStepIndex) { }
    }
}
