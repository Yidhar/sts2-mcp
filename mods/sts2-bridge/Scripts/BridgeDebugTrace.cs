namespace Sts2McpBridge.Scripts;

internal static class BridgeDebugTrace
{
    private static readonly object Sync = new();
    private static readonly bool Enabled = ResolveEnabled();

    private static readonly string LogFilePath = Path.Combine(
        BridgeRuntime.SessionDirectoryPath,
        BridgeRuntime.IsMultiInstance
            ? $"bridge-debug-{BridgeRuntime.InstanceId}.log"
            : "bridge-debug.log");

    public static void Write(string message)
    {
        if (!Enabled)
        {
            return;
        }

        try
        {
            lock (Sync)
            {
                Directory.CreateDirectory(BridgeRuntime.SessionDirectoryPath);
                File.AppendAllText(
                    LogFilePath,
                    $"{DateTimeOffset.UtcNow:O} {message}{Environment.NewLine}");
            }
        }
        catch
        {
            // Diagnostics must never break gameplay or the bridge.
        }
    }

    private static bool ResolveEnabled()
    {
        var raw = Environment.GetEnvironmentVariable("STS2_BRIDGE_DEBUG_TRACE");
        if (string.IsNullOrWhiteSpace(raw))
        {
            return false;
        }

        return raw.Trim() switch
        {
            "1" => true,
            "true" => true,
            "TRUE" => true,
            "yes" => true,
            "YES" => true,
            "on" => true,
            "ON" => true,
            _ => false
        };
    }
}
