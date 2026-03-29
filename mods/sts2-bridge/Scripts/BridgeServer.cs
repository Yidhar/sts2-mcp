using System.Net;
using System.Text;
using System.Text.Json;
using MegaCrit.Sts2.Core.Logging;

namespace Sts2McpBridge.Scripts;

internal static class BridgeServer
{
    private static readonly object Sync = new();
    private static readonly JsonSerializerOptions RequestJsonOptions = new()
    {
        PropertyNameCaseInsensitive = true
    };
    private static readonly JsonSerializerOptions ResponseJsonOptions = new()
    {
        WriteIndented = false
    };

    private static HttpListener? _listener;
    private static CancellationTokenSource? _shutdown;
    private static Task? _serverLoop;

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

            try
            {
                _shutdown = new CancellationTokenSource();
                Exception? lastException = null;

                for (var port = BridgeRuntime.PreferredPort; port <= BridgeRuntime.MaxPort; port++)
                {
                    var baseUrl = $"http://127.0.0.1:{port}/";

                    try
                    {
                        var listener = new HttpListener();
                        listener.Prefixes.Add(baseUrl);
                        listener.Start();

                        BridgeRuntime.SetPort(port);
                        _listener = listener;
                        _serverLoop = Task.Run(() => RunAsync(listener, _shutdown.Token));
                        BridgeSessionRegistry.WriteSessionFile();

                        if (port != BridgeRuntime.PreferredPort)
                        {
                            Log.Warn(
                                $"[{BridgeRuntime.ModId}] Preferred port {BridgeRuntime.PreferredPort} was unavailable. " +
                                $"Using fallback port {port}.");
                        }

                        Log.Info($"[{BridgeRuntime.ModId}] HTTP bridge listening on {BridgeRuntime.BaseUrl}");
                        return true;
                    }
                    catch (HttpListenerException ex)
                    {
                        lastException = ex;
                    }
                }

                throw new InvalidOperationException(
                    $"No available loopback port found in range {BridgeRuntime.PreferredPort}-{BridgeRuntime.MaxPort}.",
                    lastException);
            }
            catch (Exception ex)
            {
                _shutdown?.Dispose();
                _shutdown = null;

                _listener?.Close();
                _listener = null;

                Log.Error($"[{BridgeRuntime.ModId}] Failed to start HTTP bridge: {ex}");
                return false;
            }
        }
    }

    private static async Task RunAsync(HttpListener listener, CancellationToken cancellationToken)
    {
        while (!cancellationToken.IsCancellationRequested)
        {
            try
            {
                var context = await listener.GetContextAsync();
                _ = Task.Run(() => HandleAsync(context, cancellationToken), cancellationToken);
            }
            catch (HttpListenerException) when (cancellationToken.IsCancellationRequested)
            {
                break;
            }
            catch (ObjectDisposedException) when (cancellationToken.IsCancellationRequested)
            {
                break;
            }
            catch (Exception ex)
            {
                Log.Error($"[{BridgeRuntime.ModId}] HTTP bridge loop failed: {ex}");
            }
        }
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
                var payload = new
                {
                    ok = true,
                    bridge_name = BridgeRuntime.BridgeName,
                    bridge_version = BridgeRuntime.BridgeVersion,
                    session_id = BridgeRuntime.SessionId,
                    started_at_utc = BridgeRuntime.StartedAtUtc,
                    base_url = BridgeRuntime.BaseUrl,
                    port = BridgeRuntime.Port,
                    session_file_path = BridgeRuntime.SessionFilePath,
                    visible_only = BridgeRuntime.VisibleOnly,
                    game_assembly_version = BridgeRuntime.GameAssemblyVersion
                };

                await WriteJsonAsync(context.Response, HttpStatusCode.OK, payload, cancellationToken);
                BridgeDebugTrace.Write($"http {method} {path} ok");
                return;
            }

            if (method.Equals("GET", StringComparison.OrdinalIgnoreCase) &&
                path.Equals("/state", StringComparison.OrdinalIgnoreCase))
            {
                BridgeDebugTrace.Write("http GET /state authorize");
                EnsureAuthorized(context.Request);
                BridgeDebugTrace.Write("http GET /state authorized");
                var payload = await BridgeGameApi.GetStateResponseAsync(cancellationToken);
                await WriteJsonAsync(context.Response, HttpStatusCode.OK, payload, cancellationToken);
                BridgeDebugTrace.Write("http GET /state ok");
                return;
            }

            if (method.Equals("POST", StringComparison.OrdinalIgnoreCase) &&
                path.Equals("/action", StringComparison.OrdinalIgnoreCase))
            {
                BridgeDebugTrace.Write("http POST /action authorize");
                EnsureAuthorized(context.Request);
                BridgeDebugTrace.Write("http POST /action authorized");
                var request = await ReadJsonAsync<BridgeActionRequest>(context.Request, cancellationToken);
                var payload = await BridgeGameApi.PerformActionResponseAsync(request, cancellationToken);
                await WriteJsonAsync(context.Response, HttpStatusCode.OK, payload, cancellationToken);
                BridgeDebugTrace.Write("http POST /action ok");
                return;
            }

            if (method.Equals("GET", StringComparison.OrdinalIgnoreCase) &&
                path.Equals("/events", StringComparison.OrdinalIgnoreCase))
            {
                BridgeDebugTrace.Write("http GET /events authorize");
                EnsureAuthorized(context.Request);
                BridgeDebugTrace.Write("http GET /events authorized");
                await BridgeGameApi.StreamFrontierEventsAsync(context.Response, cancellationToken);
                BridgeDebugTrace.Write("http GET /events closed");
                return;
            }

            if (method.Equals("GET", StringComparison.OrdinalIgnoreCase) &&
                path.Equals("/env/spec", StringComparison.OrdinalIgnoreCase))
            {
                BridgeDebugTrace.Write("http GET /env/spec authorize");
                EnsureAuthorized(context.Request);
                BridgeDebugTrace.Write("http GET /env/spec authorized");
                var payload = BridgeGameApi.GetEnvSpecResponse();
                await WriteJsonAsync(context.Response, HttpStatusCode.OK, payload, cancellationToken);
                BridgeDebugTrace.Write("http GET /env/spec ok");
                return;
            }

            if (method.Equals("POST", StringComparison.OrdinalIgnoreCase) &&
                path.Equals("/env/reset", StringComparison.OrdinalIgnoreCase))
            {
                BridgeDebugTrace.Write("http POST /env/reset authorize");
                EnsureAuthorized(context.Request);
                BridgeDebugTrace.Write("http POST /env/reset authorized");
                var request = await ReadJsonAsync<BridgeEnvResetRequest>(context.Request, cancellationToken);
                var payload = await BridgeGameApi.ResetEnvResponseAsync(request, cancellationToken);
                await WriteJsonAsync(context.Response, HttpStatusCode.OK, payload, cancellationToken);
                BridgeDebugTrace.Write("http POST /env/reset ok");
                return;
            }

            if (method.Equals("POST", StringComparison.OrdinalIgnoreCase) &&
                path.Equals("/env/step", StringComparison.OrdinalIgnoreCase))
            {
                BridgeDebugTrace.Write("http POST /env/step authorize");
                EnsureAuthorized(context.Request);
                BridgeDebugTrace.Write("http POST /env/step authorized");
                var request = await ReadJsonAsync<BridgeEnvStepRequest>(context.Request, cancellationToken);
                var payload = await BridgeGameApi.StepEnvResponseAsync(request, cancellationToken);
                await WriteJsonAsync(context.Response, HttpStatusCode.OK, payload, cancellationToken);
                BridgeDebugTrace.Write("http POST /env/step ok");
                return;
            }

            var notFound = new
            {
                ok = false,
                error = "not_found",
                method,
                path
            };

            await WriteJsonAsync(context.Response, HttpStatusCode.NotFound, notFound, cancellationToken);
            BridgeDebugTrace.Write($"http {method} {path} not_found");
        }
        catch (BridgeRequestException ex)
        {
            BridgeDebugTrace.Write($"http error {ex.ErrorCode}: {ex.Message}");
            if (ex.StatusCode == HttpStatusCode.Unauthorized)
            {
                context.Response.Headers["WWW-Authenticate"] = "Bearer";
            }

            var payload = new
            {
                ok = false,
                error = ex.ErrorCode,
                message = ex.Message,
                details = ex.Details
            };

            await WriteJsonAsync(context.Response, ex.StatusCode, payload, cancellationToken);
        }
        catch (Exception ex)
        {
            Log.Error($"[{BridgeRuntime.ModId}] Request handling failed: {ex}");
            BridgeDebugTrace.Write($"http internal_error: {ex}");

            if (!context.Response.OutputStream.CanWrite)
            {
                return;
            }

            var payload = new
            {
                ok = false,
                error = "internal_error",
                message = ex.Message
            };

            await WriteJsonAsync(context.Response, HttpStatusCode.InternalServerError, payload, cancellationToken);
        }
    }

    private static void EnsureAuthorized(HttpListenerRequest request)
    {
        var authorization = request.Headers["Authorization"];
        if (!TryGetBearerToken(authorization, out var token) ||
            !string.Equals(token, BridgeRuntime.SessionToken, StringComparison.Ordinal))
        {
            throw new BridgeRequestException(
                HttpStatusCode.Unauthorized,
                "missing_or_invalid_token",
                "A valid Bearer token is required for this endpoint.");
        }
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

    private static async Task<T> ReadJsonAsync<T>(HttpListenerRequest request, CancellationToken cancellationToken)
    {
        if (!request.HasEntityBody)
        {
            throw new BridgeRequestException(
                HttpStatusCode.BadRequest,
                "missing_request_body",
                "Request body is required.");
        }

        try
        {
            var value = await JsonSerializer.DeserializeAsync<T>(
                request.InputStream,
                RequestJsonOptions,
                cancellationToken);

            if (value is null)
            {
                throw new BridgeRequestException(
                    HttpStatusCode.BadRequest,
                    "invalid_request_body",
                    "Request body could not be parsed.");
            }

            return value;
        }
        catch (JsonException ex)
        {
            throw new BridgeRequestException(
                HttpStatusCode.BadRequest,
                "invalid_json",
                "Request body is not valid JSON.",
                new
                {
                    ex.Message
                });
        }
    }

    private static string NormalizePath(string? path)
    {
        if (string.IsNullOrWhiteSpace(path))
        {
            return "/";
        }

        return path.Length > 1 ? path.TrimEnd('/') : path;
    }

    private static async Task WriteJsonAsync(
        HttpListenerResponse response,
        HttpStatusCode statusCode,
        object payload,
        CancellationToken cancellationToken)
    {
        response.StatusCode = (int)statusCode;
        response.ContentType = "application/json; charset=utf-8";

        var json = JsonSerializer.Serialize(payload, ResponseJsonOptions);

        var bytes = Encoding.UTF8.GetBytes(json);
        response.ContentLength64 = bytes.Length;

        await response.OutputStream.WriteAsync(bytes, cancellationToken);
        response.OutputStream.Close();
    }
}
