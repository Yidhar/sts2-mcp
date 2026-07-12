using System.Diagnostics;
using System.Security.Cryptography;
using MegaCrit.Sts2.Core.Modding;

namespace Sts2McpBridge.Scripts;

internal static class BridgeRuntime
{
    public const string ModId = "sts2-bridge";
    public const string HarmonyId = "dev.yidhar.sts2.bridge";
    public const string BridgeName = "STS2 MCP Bridge";
    public const string BridgeVersion = "0.8.0";
    public const string StateSchemaVersion = "2026-03-20.1";
    public const int BasePreferredPort = 27100;
    public const int PortsPerInstance = 10;
    public const int MaxInstanceId = 3000;
    public const bool VisibleOnly = false;

    private static readonly string RawInstanceId =
        Environment.GetEnvironmentVariable("STS2_BRIDGE_INSTANCE_ID")?.Trim() ?? string.Empty;

    public static string InstanceId => RawInstanceId;

    public static bool IsMultiInstance => InstanceId.Length > 0;

    public static bool TrainingV2Enabled { get; } = ReadBooleanEnvironmentVariable(
        "STS2_BRIDGE_ENABLE_TRAINING_V2",
        defaultValue: false);

    public static bool LegacyV1Enabled { get; } = ReadBooleanEnvironmentVariable(
        "STS2_BRIDGE_ENABLE_LEGACY_V1",
        defaultValue: false);

    public static int GameThreadHeartbeatTimeoutMs { get; } = ReadIntegerEnvironmentVariable(
        "STS2_BRIDGE_HEARTBEAT_TIMEOUT_MS",
        defaultValue: 5000,
        minimum: 1000,
        maximum: 60000);

    private static int InstancePortOffset =>
        int.TryParse(InstanceId, out var id) && id >= 0 && id <= MaxInstanceId
            ? id * PortsPerInstance
            : 0;

    public static int PreferredPort => BasePreferredPort + InstancePortOffset;

    public static int MaxPort => PreferredPort + PortsPerInstance - 1;

    public static int Port { get; private set; } = PreferredPort;

    public static string BaseUrl => $"http://127.0.0.1:{Port}/";

    public static string SessionId { get; } = Guid.NewGuid().ToString("n");

    // SessionToken is the player-control credential retained under its historic
    // property name for in-process compatibility. It never authorizes legacy-v1.
    public static string SessionToken { get; } = CreateSessionToken();

    public static string TrainingSessionToken { get; } = CreateSessionToken();

    public static string LegacySessionToken { get; } = CreateSessionToken();

    public static DateTimeOffset StartedAtUtc { get; } = DateTimeOffset.UtcNow;

    public static int ProcessId { get; } = Process.GetCurrentProcess().Id;

    public static DateTimeOffset ProcessStartedAtUtc
    {
        get
        {
            try
            {
                return Process.GetCurrentProcess().StartTime.ToUniversalTime();
            }
            catch
            {
                return StartedAtUtc;
            }
        }
    }

    public static string SessionDirectoryPath { get; } = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData),
        "SlayTheSpire2",
        "bridge");

    private static string SessionFileName =>
        IsMultiInstance &&
        int.TryParse(InstanceId, out var instanceId) &&
        instanceId >= 0 &&
        instanceId <= MaxInstanceId
            ? $"session_{instanceId}.json"
            : IsMultiInstance
                ? "session.invalid.json"
                : "session.json";

    public static string SessionFilePath { get; } = Path.Combine(
        SessionDirectoryPath,
        SessionFileName);

    public static string GameAssemblyVersion =>
        typeof(Mod).Assembly.GetName().Version?.ToString() ?? "unknown";

    public static string[] ApiVersions => LegacyV1Enabled
        ? [BridgeProtocolV2.ApiVersion, "legacy-v1"]
        : [BridgeProtocolV2.ApiVersion];

    public static string[] EnabledCapabilities
    {
        get
        {
            var capabilities = new List<string> { BridgeProtocolV2.PlayerControlCapability };
            if (TrainingV2Enabled)
            {
                capabilities.Add(BridgeProtocolV2.TrainingCapability);
            }
            if (LegacyV1Enabled)
            {
                capabilities.Add(BridgeProtocolV2.LegacyPrivilegedCapability);
            }
            return capabilities.ToArray();
        }
    }

    public static bool TryValidateConfiguration(out string error)
    {
        error = string.Empty;
        if (!IsMultiInstance)
        {
            return true;
        }

        if (!int.TryParse(InstanceId, out var instanceId) ||
            instanceId < 0 ||
            instanceId > MaxInstanceId)
        {
            error = $"STS2_BRIDGE_INSTANCE_ID must be an integer from 0 through {MaxInstanceId}.";
            return false;
        }

        if (PreferredPort < 1024 || MaxPort > 65535)
        {
            error = "The configured Bridge instance maps outside the valid TCP port range.";
            return false;
        }

        return true;
    }

    public static void SetPort(int port)
    {
        if (port < PreferredPort || port > MaxPort)
        {
            throw new ArgumentOutOfRangeException(nameof(port));
        }

        Port = port;
    }

    private static int ReadIntegerEnvironmentVariable(
        string name,
        int defaultValue,
        int minimum,
        int maximum)
    {
        var raw = Environment.GetEnvironmentVariable(name);
        return int.TryParse(raw, out var value)
            ? Math.Clamp(value, minimum, maximum)
            : defaultValue;
    }

    private static bool ReadBooleanEnvironmentVariable(string name, bool defaultValue)
    {
        var raw = Environment.GetEnvironmentVariable(name);
        if (string.IsNullOrWhiteSpace(raw))
        {
            return defaultValue;
        }

        return raw.Trim() switch
        {
            "1" => true,
            "true" => true,
            "TRUE" => true,
            "0" => false,
            "false" => false,
            "FALSE" => false,
            _ => defaultValue
        };
    }

    private static string CreateSessionToken()
    {
        var bytes = RandomNumberGenerator.GetBytes(32);
        return Convert.ToBase64String(bytes)
            .Replace('+', '-')
            .Replace('/', '_')
            .TrimEnd('=');
    }
}
