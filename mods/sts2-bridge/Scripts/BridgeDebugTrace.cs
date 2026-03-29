namespace Sts2McpBridge.Scripts;

internal static class BridgeDebugTrace
{
    private static readonly object Sync = new();

    private static readonly string LogFilePath = Path.Combine(
        BridgeRuntime.SessionDirectoryPath,
        BridgeRuntime.IsMultiInstance
            ? $"bridge-debug-{BridgeRuntime.InstanceId}.log"
            : "bridge-debug.log");

    public static void Write(string message)
    {
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
}
