using System.Diagnostics;
using System.Text;
using System.Text.Json;
using MegaCrit.Sts2.Core.Logging;

namespace Sts2McpBridge.Scripts;

internal static class BridgeSessionRegistry
{
    private static readonly JsonSerializerOptions SessionJsonOptions = new()
    {
        WriteIndented = true
    };

    public static void WriteSessionFile()
    {
        var gameCompatibility = BridgeGameCompatibilityState.Current;
        if (!gameCompatibility.StartupAllowed ||
            !string.Equals(gameCompatibility.Health, "ready", StringComparison.Ordinal) ||
            !string.IsNullOrEmpty(gameCompatibility.ErrorCode) ||
            !string.IsNullOrEmpty(gameCompatibility.ErrorMessage) ||
            string.IsNullOrWhiteSpace(gameCompatibility.ProfileId) ||
            gameCompatibility.ProbeResults.Count == 0 ||
            gameCompatibility.ProbeResults.Any(static result => !result.Passed))
        {
            throw new InvalidOperationException(
                "A Bridge session descriptor cannot be published without a successful " +
                "fail-closed retail game compatibility assessment.");
        }

        var tokens = new Dictionary<string, string>(StringComparer.Ordinal)
        {
            [BridgeProtocolV2.PlayerControlCapability] = BridgeRuntime.SessionToken
        };
        if (BridgeRuntime.TrainingV2Enabled)
        {
            tokens[BridgeProtocolV2.TrainingCapability] = BridgeRuntime.TrainingSessionToken;
        }
        if (BridgeRuntime.LegacyV1Enabled)
        {
            tokens[BridgeProtocolV2.LegacyPrivilegedCapability] = BridgeRuntime.LegacySessionToken;
        }

        var payload = new Dictionary<string, object?>(StringComparer.Ordinal)
        {
            ["session_id"] = BridgeRuntime.SessionId,
            ["pid"] = BridgeRuntime.ProcessId,
            ["process_started_at_utc"] = BridgeRuntime.ProcessStartedAtUtc,
            ["base_url"] = BridgeRuntime.BaseUrl,
            ["capability_tokens"] = tokens,
            ["api_versions"] = BridgeRuntime.ApiVersions,
            ["schema_version"] = BridgeProtocolV2.SchemaVersion,
            ["action_schema_version"] = Sts2.Contracts.Generated.ContractVersions.ActionSchemaVersion,
            ["legal_action_ordering_version"] = Sts2.Contracts.Generated.ContractVersions.LegalActionOrderingVersion,
            ["capabilities"] = tokens.Keys.ToArray(),
            ["created_at_utc"] = BridgeRuntime.StartedAtUtc,

            // Optional discovery/support metadata. No token is duplicated here.
            ["bridge_name"] = BridgeRuntime.BridgeName,
            ["bridge_version"] = BridgeRuntime.BridgeVersion,
            ["port"] = BridgeRuntime.Port,
            ["preferred_port"] = BridgeRuntime.PreferredPort,
            ["max_port"] = BridgeRuntime.MaxPort,
            ["visible_only"] = BridgeRuntime.VisibleOnly,
            ["game_assembly_version"] = BridgeRuntime.GameAssemblyVersion,
            ["capability_details"] = new
            {
                player_control = new
                {
                    enabled = true,
                    scope = BridgeProtocolV2.PlayerControlCapability,
                    api_version = BridgeProtocolV2.ApiVersion,
                    privileged = false
                },
                training = new
                {
                    enabled = BridgeRuntime.TrainingV2Enabled,
                    scope = BridgeProtocolV2.TrainingCapability,
                    api_version = BridgeProtocolV2.ApiVersion,
                    privileged = true
                },
                legacy_v1 = new
                {
                    enabled = BridgeRuntime.LegacyV1Enabled,
                    scope = BridgeProtocolV2.LegacyPrivilegedCapability,
                    api_version = "legacy-v1",
                    privileged = true,
                    default_enabled = false,
                    deprecation = "legacy-v1 is disabled by default and exists only for explicit migration"
                }
            },

            // Required v2 startup evidence. This is captured once above so the
            // descriptor cannot mix fields from different assessments.
            ["game_compatibility"] = gameCompatibility.ToDiagnosticPayload()
        };

        // Top-level token exists only for explicitly enabled legacy-only clients.
        // V2 clients must always select a token from capability_tokens.
        if (BridgeRuntime.LegacyV1Enabled)
        {
            payload["token"] = BridgeRuntime.LegacySessionToken;
        }

        Directory.CreateDirectory(BridgeRuntime.SessionDirectoryPath);
        var json = JsonSerializer.Serialize(payload, SessionJsonOptions);
        var bytes = Encoding.UTF8.GetBytes(json);
        var temporaryPath = $"{BridgeRuntime.SessionFilePath}.{BridgeRuntime.SessionId}.tmp";

        try
        {
            using (var stream = new FileStream(
                       temporaryPath,
                       FileMode.CreateNew,
                       FileAccess.Write,
                       FileShare.None,
                       4096,
                       FileOptions.WriteThrough))
            {
                stream.Write(bytes);
                stream.Flush(flushToDisk: true);
            }

            File.Move(temporaryPath, BridgeRuntime.SessionFilePath, overwrite: true);
            RestrictSessionFileAccessBestEffort(BridgeRuntime.SessionFilePath);
        }
        finally
        {
            try
            {
                File.Delete(temporaryPath);
            }
            catch
            {
                // Best-effort cleanup only; never include a token in the log.
            }
        }

        Log.Info($"[{BridgeRuntime.ModId}] Published the Bridge session descriptor atomically.");
    }

    private static void RestrictSessionFileAccessBestEffort(string path)
    {
        if (!OperatingSystem.IsWindows())
        {
            return;
        }

        try
        {
            var account = string.IsNullOrWhiteSpace(Environment.UserDomainName)
                ? Environment.UserName
                : $"{Environment.UserDomainName}\\{Environment.UserName}";
            var startInfo = new ProcessStartInfo
            {
                FileName = "icacls.exe",
                UseShellExecute = false,
                CreateNoWindow = true,
                RedirectStandardOutput = true,
                RedirectStandardError = true
            };
            startInfo.ArgumentList.Add(path);
            startInfo.ArgumentList.Add("/inheritance:r");
            startInfo.ArgumentList.Add("/grant:r");
            startInfo.ArgumentList.Add($"{account}:(F)");

            using var process = Process.Start(startInfo);
            if (process is null || !process.WaitForExit(2000) || process.ExitCode != 0)
            {
                try
                {
                    process?.Kill(entireProcessTree: true);
                }
                catch (Exception ex)
                {
                    BridgeDebugTrace.Write(
                        $"session_acl_helper_termination_failed: {ex.GetBaseException().Message}");
                }

                Log.Warn($"[{BridgeRuntime.ModId}] Could not restrict the session descriptor ACL to the current user.");
            }
        }
        catch (Exception ex)
        {
            // Never log descriptor contents or a token.
            Log.Warn(
                $"[{BridgeRuntime.ModId}] Session descriptor ACL hardening failed: {ex.GetBaseException().Message}");
        }
    }

    public static void DeleteSessionFileIfOwned()
    {
        try
        {
            if (!File.Exists(BridgeRuntime.SessionFilePath))
            {
                return;
            }

            using var document = JsonDocument.Parse(File.ReadAllText(BridgeRuntime.SessionFilePath));
            if (!document.RootElement.TryGetProperty("session_id", out var sessionIdProperty) ||
                !string.Equals(
                    sessionIdProperty.GetString(),
                    BridgeRuntime.SessionId,
                    StringComparison.Ordinal))
            {
                return;
            }

            File.Delete(BridgeRuntime.SessionFilePath);
        }
        catch (Exception ex)
        {
            Log.Warn($"[{BridgeRuntime.ModId}] Failed to remove the owned session descriptor: {ex.GetBaseException().Message}");
        }
    }
}
