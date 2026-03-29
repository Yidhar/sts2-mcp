using System.Diagnostics;
using System.Security.Cryptography;
using MegaCrit.Sts2.Core.Modding;

namespace Sts2McpBridge.Scripts;

internal static class BridgeRuntime
{
    public const string ModId = "sts2-bridge";
    public const string HarmonyId = "dev.yidhar.sts2.bridge";
    public const string BridgeName = "STS2 MCP Bridge";
    public const string BridgeVersion = "0.7.12";
    public const string StateSchemaVersion = "2026-03-20.1";
    public const int BasePreferredPort = 27100;
    public const int PortsPerInstance = 10;
    public const bool VisibleOnly = true;

    /// <summary>
    /// Instance ID for parallel training. Set via STS2_BRIDGE_INSTANCE_ID env var.
    /// Empty string means single-instance mode (default).
    /// </summary>
    public static string InstanceId { get; } =
        Environment.GetEnvironmentVariable("STS2_BRIDGE_INSTANCE_ID") ?? "";

    public static bool IsMultiInstance => InstanceId.Length > 0;

    private static int InstancePortOffset =>
        int.TryParse(InstanceId, out var id) ? id * PortsPerInstance : 0;

    public static int PreferredPort => BasePreferredPort + InstancePortOffset;
    public static int MaxPort => PreferredPort + PortsPerInstance - 1;

    public static int Port { get; private set; } = PreferredPort;

    public static string BaseUrl => $"http://127.0.0.1:{Port}/";

    public static string SessionId { get; } = Guid.NewGuid().ToString("n");

    public static string SessionToken { get; } = CreateSessionToken();

    public static DateTimeOffset StartedAtUtc { get; } = DateTimeOffset.UtcNow;

    public static int ProcessId { get; } = Process.GetCurrentProcess().Id;

    public static string SessionDirectoryPath { get; } = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData),
        "SlayTheSpire2",
        "bridge");

    private static string SessionFileName =>
        IsMultiInstance ? $"session_{InstanceId}.json" : "session.json";

    public static string SessionFilePath { get; } = Path.Combine(
        SessionDirectoryPath,
        SessionFileName);

    public static string GameAssemblyVersion =>
        typeof(Mod).Assembly.GetName().Version?.ToString() ?? "unknown";

    public static void SetPort(int port)
    {
        Port = port;
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
