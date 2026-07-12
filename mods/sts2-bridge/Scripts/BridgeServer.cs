using System.Collections.Concurrent;
using System.Net;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using MegaCrit.Sts2.Core.Logging;

namespace Sts2McpBridge.Scripts;

internal static class BridgeServer
{
    private const int MaxConcurrentRequests = 32;
    private const int MaxConcurrentEventStreams = 8;
    private const int MaxRequestBodyBytes = 1024 * 1024;
    private const int ShutdownWaitMilliseconds = 5000;

    private static readonly object Sync = new();
    private static readonly JsonSerializerOptions RequestJsonOptions = new()
    {
        PropertyNameCaseInsensitive = true,
        MaxDepth = 64
    };
    private static readonly JsonSerializerOptions V2RequestJsonOptions = new()
    {
        PropertyNameCaseInsensitive = false,
        UnmappedMemberHandling = JsonUnmappedMemberHandling.Disallow,
        MaxDepth = 64
    };
    private static readonly JsonSerializerOptions ResponseJsonOptions = new()
    {
        WriteIndented = false
    };
    private static readonly SemaphoreSlim RequestSlots =
        new(MaxConcurrentRequests, MaxConcurrentRequests);
    private static readonly SemaphoreSlim EventStreamSlots =
        new(MaxConcurrentEventStreams, MaxConcurrentEventStreams);
    private static readonly ConcurrentDictionary<long, Task> InFlightRequests = new();

    private static HttpListener? _listener;
    private static CancellationTokenSource? _shutdown;
    private static Task? _serverLoop;
    private static long _nextRequestId;

    public static bool IsRunning
    {
        get
        {
            lock (Sync)
            {
                return _listener is not null;
            }
        }
    }

    public static bool Start()
    {
        lock (Sync)
        {
            if (_listener is not null)
            {
                return true;
            }

            if (!BridgeRuntime.TryValidateConfiguration(out var configurationError))
            {
                Log.Error($"[{BridgeRuntime.ModId}] Refusing to start: {configurationError}");
                return false;
            }

            var shutdown = new CancellationTokenSource();
            try
            {
                Exception? lastException = null;
                for (var port = BridgeRuntime.PreferredPort; port <= BridgeRuntime.MaxPort; port++)
                {
                    var baseUrl = $"http://127.0.0.1:{port}/";
                    var listener = new HttpListener();
                    try
                    {
                        listener.Prefixes.Add(baseUrl);
                        listener.Start();
                        BridgeRuntime.SetPort(port);

                        // Publish discovery only after the listener has started, but
                        // before the accept loop is visible. A failed atomic publish
                        // closes this listener and leaves no orphan server task.
                        BridgeSessionRegistry.WriteSessionFile();

                        _shutdown = shutdown;
                        _listener = listener;
                        _serverLoop = Task.Run(() => RunAsync(listener, shutdown.Token));

                        if (port != BridgeRuntime.PreferredPort)
                        {
                            Log.Warn(
                                $"[{BridgeRuntime.ModId}] Preferred port {BridgeRuntime.PreferredPort} was unavailable. " +
                                $"Using fallback port {port}.");
                        }

                        Log.Info($"[{BridgeRuntime.ModId}] HTTP bridge listening on loopback port {port}.");
                        return true;
                    }
                    catch (HttpListenerException ex)
                    {
                        lastException = ex;
                        listener.Close();
                    }
                    catch
                    {
                        listener.Close();
                        throw;
                    }
                }

                throw new InvalidOperationException(
                    $"No available loopback port found in range {BridgeRuntime.PreferredPort}-{BridgeRuntime.MaxPort}.",
                    lastException);
            }
            catch (Exception ex)
            {
                shutdown.Cancel();
                shutdown.Dispose();
                _shutdown = null;
                _listener = null;
                _serverLoop = null;
                BridgeSessionRegistry.DeleteSessionFileIfOwned();
                Log.Error($"[{BridgeRuntime.ModId}] Failed to start HTTP bridge: {ex}");
                return false;
            }
        }
    }

    public static void Stop()
    {
        try
        {
            StopAsync().GetAwaiter().GetResult();
        }
        catch (Exception ex)
        {
            Log.Warn($"[{BridgeRuntime.ModId}] Bridge shutdown did not complete cleanly: {ex.GetBaseException().Message}");
        }
    }

    public static async Task StopAsync()
    {
        HttpListener? listener;
        CancellationTokenSource? shutdown;
        Task? serverLoop;
        lock (Sync)
        {
            listener = _listener;
            shutdown = _shutdown;
            serverLoop = _serverLoop;
            _listener = null;
            _shutdown = null;
            _serverLoop = null;
        }

        if (listener is null && shutdown is null && serverLoop is null)
        {
            BridgeSessionRegistry.DeleteSessionFileIfOwned();
            return;
        }

        shutdown?.Cancel();
        try
        {
            listener?.Stop();
            listener?.Close();
        }
        catch (ObjectDisposedException ex)
        {
            BridgeDebugTrace.Write($"bridge_listener_already_disposed_during_stop: {ex.Message}");
        }

        BridgeSessionRegistry.DeleteSessionFileIfOwned();

        var tasks = InFlightRequests.Values.ToList();
        if (serverLoop is not null)
        {
            tasks.Add(serverLoop);
        }

        if (tasks.Count > 0)
        {
            var allTasks = Task.WhenAll(tasks);
            var completed = await Task.WhenAny(allTasks, Task.Delay(ShutdownWaitMilliseconds));
            if (completed != allTasks)
            {
                Log.Warn(
                    $"[{BridgeRuntime.ModId}] Timed out waiting for {InFlightRequests.Count} in-flight HTTP request(s) during shutdown.");
            }
        }

        shutdown?.Dispose();
    }

    private static async Task RunAsync(HttpListener listener, CancellationToken cancellationToken)
    {
        while (!cancellationToken.IsCancellationRequested)
        {
            HttpListenerContext? acceptedContext = null;
            try
            {
                acceptedContext = await listener.GetContextAsync().WaitAsync(cancellationToken);
                var requestSlot = IsEventStreamRequest(acceptedContext.Request)
                    ? EventStreamSlots
                    : RequestSlots;
                await requestSlot.WaitAsync(cancellationToken);

                var context = acceptedContext;
                acceptedContext = null;
                var requestId = Interlocked.Increment(ref _nextRequestId);
                var requestTask = HandleTrackedAsync(context, requestSlot, cancellationToken);
                InFlightRequests[requestId] = requestTask;
                _ = ObserveRequestCompletionAsync(requestId, requestTask);
            }
            catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
            {
                if (acceptedContext is not null)
                {
                    TryCloseResponse(acceptedContext.Response);
                }
                break;
            }
            catch (HttpListenerException) when (cancellationToken.IsCancellationRequested || !listener.IsListening)
            {
                if (acceptedContext is not null)
                {
                    TryCloseResponse(acceptedContext.Response);
                }
                break;
            }
            catch (ObjectDisposedException)
            {
                if (acceptedContext is not null)
                {
                    TryCloseResponse(acceptedContext.Response);
                }
                break;
            }
            catch (Exception ex)
            {
                if (acceptedContext is not null)
                {
                    TryCloseResponse(acceptedContext.Response);
                }
                Log.Error($"[{BridgeRuntime.ModId}] HTTP accept loop failed: {ex}");
                try
                {
                    await Task.Delay(100, cancellationToken);
                }
                catch (OperationCanceledException)
                {
                    break;
                }
            }
        }
    }

    private static async Task ObserveRequestCompletionAsync(long requestId, Task requestTask)
    {
        try
        {
            await requestTask;
        }
        catch
        {
            // HandleAsync owns request error reporting. This observer only makes
            // sure the bounded in-flight registry is cleaned up.
        }
        finally
        {
            InFlightRequests.TryRemove(requestId, out _);
        }
    }

    private static async Task HandleTrackedAsync(
        HttpListenerContext context,
        SemaphoreSlim requestSlot,
        CancellationToken cancellationToken)
    {
        try
        {
            await HandleAsync(context, cancellationToken);
        }
        finally
        {
            requestSlot.Release();
        }
    }

    private static bool IsEventStreamRequest(HttpListenerRequest request)
    {
        var path = NormalizePath(request.Url?.AbsolutePath);
        return request.HttpMethod.Equals("GET", StringComparison.OrdinalIgnoreCase) &&
               (path.Equals("/events", StringComparison.OrdinalIgnoreCase) ||
                path.Equals("/v2/events", StringComparison.OrdinalIgnoreCase));
    }

    private static async Task HandleAsync(HttpListenerContext context, CancellationToken cancellationToken)
    {
        try
        {
            var path = NormalizePath(context.Request.Url?.AbsolutePath);
            var method = context.Request.HttpMethod ?? "GET";
            BridgeDebugTrace.Write($"http {method} {path} start");

            if (method.Equals("GET", StringComparison.OrdinalIgnoreCase) &&
                (path.Equals("/", StringComparison.OrdinalIgnoreCase) ||
                 path.Equals("/health", StringComparison.OrdinalIgnoreCase)))
            {
                await WriteJsonAsync(context.Response, HttpStatusCode.OK, BuildPublicHealthPayload(), cancellationToken);
                return;
            }

            if (method.Equals("GET", StringComparison.OrdinalIgnoreCase) &&
                path.Equals("/v2/health", StringComparison.OrdinalIgnoreCase))
            {
                var healthCapability = ResolveAnyScopedCapability(context.Request);
                await WriteJsonAsync(
                    context.Response,
                    HttpStatusCode.OK,
                    BuildV2HealthPayload(healthCapability),
                    cancellationToken);
                return;
            }

            if (method.Equals("GET", StringComparison.OrdinalIgnoreCase) &&
                path.Equals("/state", StringComparison.OrdinalIgnoreCase))
            {
                EnsureLegacyAuthorized(context.Request);
                var payload = await BridgeGameApi.GetStateResponseAsync(cancellationToken);
                await WriteJsonAsync(context.Response, HttpStatusCode.OK, payload, cancellationToken);
                return;
            }

            if (method.Equals("GET", StringComparison.OrdinalIgnoreCase) &&
                path.Equals("/v2/state", StringComparison.OrdinalIgnoreCase))
            {
                EnsureCapabilityAuthorized(context.Request, BridgeProtocolV2.PlayerControlCapability);
                var payload = await BridgeGameApi.GetStateV2ResponseAsync(cancellationToken);
                await WriteJsonAsync(context.Response, HttpStatusCode.OK, payload, cancellationToken);
                return;
            }

            if (method.Equals("POST", StringComparison.OrdinalIgnoreCase) &&
                path.Equals("/action", StringComparison.OrdinalIgnoreCase))
            {
                EnsureLegacyAuthorized(context.Request);
                var request = await ReadJsonAsync<BridgeActionRequest>(context.Request, cancellationToken);
                var payload = await BridgeMutationGate.RunAsync(
                    "legacy.action",
                    token => BridgeGameApi.PerformActionResponseAsync(request, token),
                    cancellationToken);
                await WriteJsonAsync(context.Response, HttpStatusCode.OK, payload, cancellationToken);
                return;
            }

            if (method.Equals("POST", StringComparison.OrdinalIgnoreCase) &&
                path.Equals("/v2/commands", StringComparison.OrdinalIgnoreCase))
            {
                EnsureCapabilityAuthorized(context.Request, BridgeProtocolV2.PlayerControlCapability);
                var request = await ReadJsonV2Async<BridgeCommandEnvelopeV2>(context.Request, cancellationToken);
                var payload = await BridgeCommandService.SubmitAsync(
                    request,
                    cancellationToken,
                    GetShutdownToken());
                await WriteJsonAsync(context.Response, HttpStatusCode.OK, payload, cancellationToken);
                return;
            }

            if (method.Equals("GET", StringComparison.OrdinalIgnoreCase) &&
                TryExtractCommandRequestId(path, out var commandRequestId))
            {
                var commandCapability = ResolveCommandStatusCapability(context.Request);
                var payload = BridgeCommandService.GetStatus(commandRequestId, commandCapability);
                await WriteJsonAsync(context.Response, HttpStatusCode.OK, payload, cancellationToken);
                return;
            }

            if (method.Equals("GET", StringComparison.OrdinalIgnoreCase) &&
                path.Equals("/v2/env/spec", StringComparison.OrdinalIgnoreCase))
            {
                EnsureCapabilityAuthorized(context.Request, BridgeProtocolV2.TrainingCapability);
                await WriteJsonAsync(
                    context.Response,
                    HttpStatusCode.OK,
                    BridgeGameApi.GetEnvSpecV2Response(),
                    cancellationToken);
                return;
            }

            if (method.Equals("GET", StringComparison.OrdinalIgnoreCase) &&
                path.Equals("/v2/env/state", StringComparison.OrdinalIgnoreCase))
            {
                EnsureCapabilityAuthorized(context.Request, BridgeProtocolV2.TrainingCapability);
                var payload = await BridgeGameApi.GetEnvStateV2ResponseAsync(cancellationToken);
                await WriteJsonAsync(context.Response, HttpStatusCode.OK, payload, cancellationToken);
                return;
            }

            if (method.Equals("GET", StringComparison.OrdinalIgnoreCase) &&
                path.Equals("/v2/env/combat_catalog", StringComparison.OrdinalIgnoreCase))
            {
                EnsureCapabilityAuthorized(context.Request, BridgeProtocolV2.TrainingCapability);
                var payload = await BridgeGameApi.GetCombatCatalogV2ResponseAsync(cancellationToken);
                await WriteJsonAsync(context.Response, HttpStatusCode.OK, payload, cancellationToken);
                return;
            }

            if (method.Equals("POST", StringComparison.OrdinalIgnoreCase) &&
                path.Equals("/v2/env/reset", StringComparison.OrdinalIgnoreCase))
            {
                EnsureCapabilityAuthorized(context.Request, BridgeProtocolV2.TrainingCapability);
                var request = await ReadJsonV2Async<BridgeEnvResetEnvelopeV2>(context.Request, cancellationToken);
                var payload = await BridgeCommandService.SubmitEnvironmentResetAsync(
                    request,
                    cancellationToken,
                    GetShutdownToken());
                await WriteJsonAsync(context.Response, HttpStatusCode.OK, payload, cancellationToken);
                return;
            }

            if (method.Equals("POST", StringComparison.OrdinalIgnoreCase) &&
                path.Equals("/v2/env/step", StringComparison.OrdinalIgnoreCase))
            {
                EnsureCapabilityAuthorized(context.Request, BridgeProtocolV2.TrainingCapability);
                var request = await ReadJsonV2Async<BridgeEnvStepEnvelopeV2>(context.Request, cancellationToken);
                var payload = await BridgeCommandService.SubmitEnvironmentStepAsync(
                    request,
                    cancellationToken,
                    GetShutdownToken());
                await WriteJsonAsync(context.Response, HttpStatusCode.OK, payload, cancellationToken);
                return;
            }

            if (method.Equals("GET", StringComparison.OrdinalIgnoreCase) &&
                (path.Equals("/events", StringComparison.OrdinalIgnoreCase) ||
                 path.Equals("/v2/events", StringComparison.OrdinalIgnoreCase)))
            {
                if (path.StartsWith("/v2/", StringComparison.OrdinalIgnoreCase))
                {
                    EnsureCapabilityAuthorized(context.Request, BridgeProtocolV2.PlayerControlCapability);
                }
                else
                {
                    EnsureLegacyAuthorized(context.Request);
                }

                if (path.StartsWith("/v2/", StringComparison.OrdinalIgnoreCase))
                {
                    await BridgeGameApi.StreamFrontierEventsV2Async(context.Response, cancellationToken);
                }
                else
                {
                    await BridgeGameApi.StreamFrontierEventsAsync(context.Response, cancellationToken);
                }
                return;
            }

            if (method.Equals("GET", StringComparison.OrdinalIgnoreCase) &&
                path.Equals("/env/spec", StringComparison.OrdinalIgnoreCase))
            {
                EnsureLegacyAuthorized(context.Request);
                var payload = BridgeGameApi.GetEnvSpecResponse();
                await WriteJsonAsync(context.Response, HttpStatusCode.OK, payload, cancellationToken);
                return;
            }

            if (method.Equals("POST", StringComparison.OrdinalIgnoreCase) &&
                path.Equals("/env/reset", StringComparison.OrdinalIgnoreCase))
            {
                EnsureLegacyAuthorized(context.Request);
                var request = await ReadJsonAsync<BridgeEnvResetRequest>(context.Request, cancellationToken);
                var payload = await BridgeMutationGate.RunAsync(
                    "legacy.env.reset",
                    token => BridgeGameApi.ResetEnvResponseAsync(request, token),
                    cancellationToken);
                await WriteJsonAsync(context.Response, HttpStatusCode.OK, payload, cancellationToken);
                return;
            }

            if (method.Equals("POST", StringComparison.OrdinalIgnoreCase) &&
                path.Equals("/env/combat_reset", StringComparison.OrdinalIgnoreCase))
            {
                EnsureLegacyAuthorized(context.Request);
                var request = await ReadJsonAsync<BridgeEnvCombatResetRequest>(context.Request, cancellationToken);
                var payload = await BridgeMutationGate.RunAsync(
                    "legacy.env.combat_reset",
                    token => BridgeGameApi.CombatResetEnvResponseAsync(request, token),
                    cancellationToken);
                await WriteJsonAsync(context.Response, HttpStatusCode.OK, payload, cancellationToken);
                return;
            }

            if (method.Equals("GET", StringComparison.OrdinalIgnoreCase) &&
                path.Equals("/env/combat_catalog", StringComparison.OrdinalIgnoreCase))
            {
                EnsureLegacyAuthorized(context.Request);
                var payload = await BridgeGameApi.GetCombatCatalogResponseAsync(cancellationToken);
                await WriteJsonAsync(context.Response, HttpStatusCode.OK, payload, cancellationToken);
                return;
            }

            if (method.Equals("POST", StringComparison.OrdinalIgnoreCase) &&
                path.Equals("/env/step", StringComparison.OrdinalIgnoreCase))
            {
                EnsureLegacyAuthorized(context.Request);
                var request = await ReadJsonAsync<BridgeEnvStepRequest>(context.Request, cancellationToken);
                var payload = await BridgeMutationGate.RunAsync(
                    "legacy.env.step",
                    token => BridgeGameApi.StepEnvResponseAsync(request, token),
                    cancellationToken);
                await WriteJsonAsync(context.Response, HttpStatusCode.OK, payload, cancellationToken);
                return;
            }

            if (method.Equals("POST", StringComparison.OrdinalIgnoreCase) &&
                path.Equals("/static/export", StringComparison.OrdinalIgnoreCase))
            {
                EnsureLegacyAuthorized(context.Request);
                await WriteJsonAsync(
                    context.Response,
                    HttpStatusCode.Gone,
                    new
                    {
                        ok = false,
                        error = "static_export_removed",
                        message = "In-game HTTP static export was removed. Use the offline tools/catalog-export CLI.",
                        replacement = "python tools/catalog-export/export_catalog.py --output <directory>"
                    },
                    cancellationToken);
                return;
            }

            await WriteJsonAsync(
                context.Response,
                HttpStatusCode.NotFound,
                new { ok = false, error = "not_found", method, path },
                cancellationToken);
        }
        catch (BridgeRequestException ex)
        {
            if (ex.StatusCode == HttpStatusCode.Unauthorized)
            {
                context.Response.Headers["WWW-Authenticate"] = "Bearer";
            }

            await TryWriteJsonAsync(
                context.Response,
                ex.StatusCode,
                new { ok = false, error = ex.ErrorCode, message = ex.Message, details = ex.Details },
                cancellationToken);
        }
        catch (BridgeMutationGateBusyException ex)
        {
            await TryWriteJsonAsync(
                context.Response,
                HttpStatusCode.ServiceUnavailable,
                new { ok = false, error = "mutation_queue_full", message = ex.Message },
                cancellationToken);
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
        {
            TryCloseResponse(context.Response);
        }
        catch (Exception ex)
        {
            Log.Error($"[{BridgeRuntime.ModId}] Request handling failed: {ex}");
            await TryWriteJsonAsync(
                context.Response,
                HttpStatusCode.InternalServerError,
                new { ok = false, error = "internal_error", message = "The Bridge request failed unexpectedly." },
                cancellationToken);
        }
    }

    private static object BuildPublicHealthPayload() => new
    {
        ok = true,
        ready = IsRunning,
        bridge_name = BridgeRuntime.BridgeName,
        bridge_version = BridgeRuntime.BridgeVersion,
        api_version = BridgeProtocolV2.ApiVersion,
        schema_version = BridgeProtocolV2.SchemaVersion,
        api_versions = BridgeRuntime.ApiVersions
    };

    private static object BuildV2HealthPayload(string authorizedCapability)
    {
        var pumpAgeMs = BridgeCoordinator.MillisecondsSinceLastPump;
        var gameThreadAlive = BridgeCoordinator.IsReady &&
                              pumpAgeMs <= BridgeRuntime.GameThreadHeartbeatTimeoutMs;
        return new
        {
            transport_alive = IsRunning,
            game_thread_alive = gameThreadAlive,
            session_id = BridgeRuntime.SessionId,
            authorized_capability = authorizedCapability,
            api_version = BridgeProtocolV2.ApiVersion,
            schema_version = BridgeProtocolV2.SchemaVersion,
            pump_tick = BridgeCoordinator.PumpTick,
            ms_since_last_pump = pumpAgeMs,
            queue_depth = BridgeCoordinator.QueueDepth,
            active_operation = BridgeMutationGate.ActiveOperation,
            last_snapshot_age_ms = BridgeGameApi.MillisecondsSinceLastSnapshot,
            capabilities = BridgeRuntime.EnabledCapabilities,
            bridge_name = BridgeRuntime.BridgeName,
            bridge_version = BridgeRuntime.BridgeVersion,
            process_id = BridgeRuntime.ProcessId,
            process_started_at_utc = BridgeRuntime.ProcessStartedAtUtc,
            heartbeat_timeout_ms = BridgeRuntime.GameThreadHeartbeatTimeoutMs,
            coordinator = BridgeCoordinator.GetDiagnosticsSnapshot(),
            mutation_gate = BridgeMutationGate.GetDiagnosticsSnapshot(),
            command_store = BridgeCommandService.GetDiagnosticsSnapshot(),
            http = new
            {
                in_flight_requests = InFlightRequests.Count,
                max_concurrent_requests = MaxConcurrentRequests,
                max_concurrent_event_streams = MaxConcurrentEventStreams,
                max_request_body_bytes = MaxRequestBodyBytes
            }
        };
    }

    private static CancellationToken GetShutdownToken()
    {
        lock (Sync)
        {
            return _shutdown?.Token ?? new CancellationToken(canceled: true);
        }
    }

    private static void EnsureLegacyAuthorized(HttpListenerRequest request)
    {
        if (!BridgeRuntime.LegacyV1Enabled)
        {
            throw new BridgeRequestException(
                HttpStatusCode.NotFound,
                "legacy_v1_disabled",
                "The legacy-v1 API is disabled. Use a scoped contract-v2 endpoint.");
        }

        if (!TryGetBearerToken(request.Headers["Authorization"], out var token))
        {
            throw new BridgeRequestException(
                HttpStatusCode.Unauthorized,
                "missing_or_invalid_token",
                "The legacy-privileged Bearer token is required for this endpoint.");
        }

        if (!FixedTimeTokenEquals(token, BridgeRuntime.LegacySessionToken))
        {
            throw new BridgeRequestException(
                HttpStatusCode.Forbidden,
                "capability_not_allowed",
                "The supplied token does not grant the legacy-privileged capability.");
        }
    }

    private static string ResolveAnyScopedCapability(HttpListenerRequest request)
    {
        if (!TryGetBearerToken(request.Headers["Authorization"], out var token))
        {
            throw new BridgeRequestException(
                HttpStatusCode.Unauthorized,
                "missing_or_invalid_token",
                "A capability-scoped Bearer token is required for this endpoint.");
        }

        if (FixedTimeTokenEquals(token, BridgeRuntime.SessionToken))
        {
            return BridgeProtocolV2.PlayerControlCapability;
        }

        if (BridgeRuntime.TrainingV2Enabled &&
            FixedTimeTokenEquals(token, BridgeRuntime.TrainingSessionToken))
        {
            return BridgeProtocolV2.TrainingCapability;
        }

        if (BridgeRuntime.LegacyV1Enabled &&
            FixedTimeTokenEquals(token, BridgeRuntime.LegacySessionToken))
        {
            return BridgeProtocolV2.LegacyPrivilegedCapability;
        }

        throw new BridgeRequestException(
            HttpStatusCode.Forbidden,
            "capability_not_allowed",
            "The supplied token does not grant an enabled Bridge capability.");
    }

    private static string ResolveCommandStatusCapability(HttpListenerRequest request)
    {
        var capability = ResolveAnyScopedCapability(request);
        if (string.Equals(capability, BridgeProtocolV2.PlayerControlCapability, StringComparison.Ordinal) ||
            string.Equals(capability, BridgeProtocolV2.TrainingCapability, StringComparison.Ordinal))
        {
            return capability;
        }

        throw new BridgeRequestException(
            HttpStatusCode.Forbidden,
            "capability_not_allowed",
            "Only player-control and training tokens may query contract-v2 command status.");
    }

    private static void EnsureCapabilityAuthorized(HttpListenerRequest request, string capability)
    {
        var resolvedCapability = ResolveAnyScopedCapability(request);
        if (!string.Equals(resolvedCapability, capability, StringComparison.Ordinal))
        {
            throw new BridgeRequestException(
                HttpStatusCode.Forbidden,
                "capability_not_allowed",
                $"The supplied token does not grant the required '{capability}' capability.");
        }
    }

    private static bool FixedTimeTokenEquals(string left, string right)
    {
        var leftBytes = Encoding.UTF8.GetBytes(left);
        var rightBytes = Encoding.UTF8.GetBytes(right);
        return leftBytes.Length == rightBytes.Length &&
               CryptographicOperations.FixedTimeEquals(leftBytes, rightBytes);
    }

    private static bool TryGetBearerToken(string? authorization, out string token)
    {
        token = string.Empty;
        if (string.IsNullOrWhiteSpace(authorization))
        {
            return false;
        }

        const string Prefix = "Bearer ";
        if (!authorization.StartsWith(Prefix, StringComparison.OrdinalIgnoreCase))
        {
            return false;
        }

        token = authorization[Prefix.Length..].Trim();
        return token.Length > 0;
    }

    private static async Task<T> ReadJsonAsync<T>(
        HttpListenerRequest request,
        CancellationToken cancellationToken) =>
        await ReadJsonAsync<T>(request, RequestJsonOptions, cancellationToken);

    private static async Task<T> ReadJsonV2Async<T>(
        HttpListenerRequest request,
        CancellationToken cancellationToken) =>
        await ReadJsonAsync<T>(request, V2RequestJsonOptions, cancellationToken);

    private static async Task<T> ReadJsonAsync<T>(
        HttpListenerRequest request,
        JsonSerializerOptions serializerOptions,
        CancellationToken cancellationToken)
    {
        if (!request.HasEntityBody)
        {
            throw new BridgeRequestException(
                HttpStatusCode.BadRequest,
                "missing_request_body",
                "Request body is required.");
        }

        if (request.ContentLength64 > MaxRequestBodyBytes)
        {
            throw BuildPayloadTooLargeException();
        }

        try
        {
            using var boundedStream = new BoundedReadStream(request.InputStream, MaxRequestBodyBytes);
            var value = await JsonSerializer.DeserializeAsync<T>(
                boundedStream,
                serializerOptions,
                cancellationToken);
            return value ?? throw new BridgeRequestException(
                HttpStatusCode.BadRequest,
                "invalid_request_body",
                "Request body could not be parsed.");
        }
        catch (JsonException ex)
        {
            throw new BridgeRequestException(
                HttpStatusCode.BadRequest,
                "invalid_json",
                "Request body is not valid JSON.",
                new { ex.Message });
        }
        catch (RequestBodyLimitExceededException)
        {
            throw BuildPayloadTooLargeException();
        }
    }

    private static BridgeRequestException BuildPayloadTooLargeException() =>
        new(
            HttpStatusCode.RequestEntityTooLarge,
            "request_body_too_large",
            $"Request body exceeds the {MaxRequestBodyBytes}-byte limit.");

    private static bool TryExtractCommandRequestId(string path, out string requestId)
    {
        const string Prefix = "/v2/commands/";
        requestId = string.Empty;
        if (!path.StartsWith(Prefix, StringComparison.OrdinalIgnoreCase) || path.Length <= Prefix.Length)
        {
            return false;
        }

        var candidate = path[Prefix.Length..];
        if (candidate.Contains('/'))
        {
            return false;
        }

        requestId = Uri.UnescapeDataString(candidate);
        return true;
    }

    private static string NormalizePath(string? path)
    {
        if (string.IsNullOrWhiteSpace(path))
        {
            return "/";
        }

        if (path.Length > 2048)
        {
            throw new BridgeRequestException(
                HttpStatusCode.RequestUriTooLong,
                "request_path_too_long",
                "Request path exceeds the Bridge limit.");
        }

        return path.Length > 1 ? path.TrimEnd('/') : path;
    }

    private static async Task TryWriteJsonAsync(
        HttpListenerResponse response,
        HttpStatusCode statusCode,
        object payload,
        CancellationToken cancellationToken)
    {
        try
        {
            await WriteJsonAsync(response, statusCode, payload, cancellationToken);
        }
        catch (Exception ex) when (ex is IOException or HttpListenerException or ObjectDisposedException or OperationCanceledException)
        {
            TryCloseResponse(response);
        }
    }

    private static async Task WriteJsonAsync(
        HttpListenerResponse response,
        HttpStatusCode statusCode,
        object payload,
        CancellationToken cancellationToken)
    {
        response.StatusCode = (int)statusCode;
        response.ContentType = "application/json; charset=utf-8";
        response.Headers["Cache-Control"] = "no-store";

        var bytes = JsonSerializer.SerializeToUtf8Bytes(payload, ResponseJsonOptions);
        response.ContentLength64 = bytes.Length;
        await response.OutputStream.WriteAsync(bytes, cancellationToken);
        response.OutputStream.Close();
    }

    private static void TryCloseResponse(HttpListenerResponse response)
    {
        try
        {
            response.OutputStream.Close();
        }
        catch (Exception ex)
        {
            BridgeDebugTrace.Write($"bridge_response_close_failed: {ex.GetBaseException().Message}");
        }
    }

    private sealed class RequestBodyLimitExceededException : IOException
    {
    }

    private sealed class BoundedReadStream : Stream
    {
        private readonly Stream _inner;
        private readonly long _maximumBytes;
        private long _bytesRead;

        public BoundedReadStream(Stream inner, long maximumBytes)
        {
            _inner = inner;
            _maximumBytes = maximumBytes;
        }

        public override bool CanRead => _inner.CanRead;
        public override bool CanSeek => false;
        public override bool CanWrite => false;
        public override long Length => throw new NotSupportedException();
        public override long Position
        {
            get => _bytesRead;
            set => throw new NotSupportedException();
        }

        public override void Flush() => throw new NotSupportedException();

        public override int Read(byte[] buffer, int offset, int count)
        {
            var read = _inner.Read(buffer, offset, LimitRequestedCount(count));
            AccountForRead(read);
            return read;
        }

        public override async Task<int> ReadAsync(
            byte[] buffer,
            int offset,
            int count,
            CancellationToken cancellationToken)
        {
            var read = await _inner.ReadAsync(
                buffer.AsMemory(offset, LimitRequestedCount(count)),
                cancellationToken);
            AccountForRead(read);
            return read;
        }

        public override async ValueTask<int> ReadAsync(
            Memory<byte> buffer,
            CancellationToken cancellationToken = default)
        {
            var read = await _inner.ReadAsync(
                buffer[..LimitRequestedCount(buffer.Length)],
                cancellationToken);
            AccountForRead(read);
            return read;
        }

        public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();
        public override void SetLength(long value) => throw new NotSupportedException();
        public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException();

        protected override void Dispose(bool disposing)
        {
            // HttpListener owns the request stream.
            base.Dispose(disposing);
        }

        private int LimitRequestedCount(int requested)
        {
            var remainingWithSentinel = Math.Max(1, _maximumBytes - _bytesRead + 1);
            return (int)Math.Min(requested, remainingWithSentinel);
        }

        private void AccountForRead(int read)
        {
            _bytesRead += read;
            if (_bytesRead > _maximumBytes)
            {
                throw new RequestBodyLimitExceededException();
            }
        }
    }
}
