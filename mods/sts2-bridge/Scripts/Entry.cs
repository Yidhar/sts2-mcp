using System.Runtime.Loader;
using HarmonyLib;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Modding;

namespace Sts2McpBridge.Scripts;

[ModInitializer("Init")]
public static class Entry
{
    private static int _initialized;
    private static int _shutdown;
    private static Harmony? _harmony;

    public static void Init()
    {
        if (Interlocked.Exchange(ref _initialized, 1) != 0)
        {
            return;
        }

        AppDomain.CurrentDomain.ProcessExit += static (_, _) => Shutdown();
        AssemblyLoadContext.Default.Unloading += static _ => Shutdown();

        GameAssemblyCompatibilityAssessment compatibility;
        try
        {
            compatibility = BridgeGameAssemblyCompatibilityRegistry.Evaluate(
                new ReflectionGameAssemblyProbe(typeof(Mod).Assembly));
        }
        catch (Exception ex)
        {
            compatibility = BridgeGameCompatibilityState.Current.WithActivationFailure(
                "game_compatibility_evaluation_failed",
                $"Compatibility evaluation raised {ex.GetType().Name}; startup was denied.");
        }

        BridgeGameCompatibilityState.Publish(compatibility);
        if (!compatibility.StartupAllowed)
        {
            LogCompatibilityFailure(compatibility);
            return;
        }

        Log.Info(
            $"[{BridgeRuntime.ModId}] Game adapter compatibility passed: " +
            $"health={compatibility.Health}, profile={compatibility.ProfileId}, " +
            $"assembly=({compatibility.Identity.DisplayName}), " +
            $"probes={compatibility.PassedProbeCount}/{compatibility.ProbeResults.Count}.");

        try
        {
            _harmony = new Harmony(BridgeRuntime.HarmonyId);
            _harmony.PatchAll();
        }
        catch (Exception ex)
        {
            compatibility = compatibility.WithActivationFailure(
                "harmony_patch_activation_failed",
                $"Harmony patch activation raised {ex.GetType().Name}; startup was denied.");
            BridgeGameCompatibilityState.Publish(compatibility);

            try
            {
                _harmony?.UnpatchAll(BridgeRuntime.HarmonyId);
            }
            catch (Exception cleanupException)
            {
                Log.Warn(
                    $"[{BridgeRuntime.ModId}] Failed to remove partial Harmony patches after denied startup: " +
                    $"{cleanupException.GetBaseException().Message}");
            }

            LogCompatibilityFailure(compatibility);
            return;
        }

        var bridgeStarted = BridgeServer.Start();
        if (bridgeStarted)
        {
            Log.Info(
                $"[{BridgeRuntime.ModId}] Initialized {BridgeRuntime.BridgeName} v{BridgeRuntime.BridgeVersion} " +
                $"for game assembly {BridgeRuntime.GameAssemblyVersion}.");
        }
        else
        {
            Log.Warn(
                $"[{BridgeRuntime.ModId}] Mod initialized, but the HTTP bridge did not start. " +
                $"Check earlier log lines for the underlying error.");
        }
    }

    private static void LogCompatibilityFailure(GameAssemblyCompatibilityAssessment compatibility)
    {
        var failedProbes = compatibility.ProbeResults
            .Where(static result => !result.Passed)
            .Select(static result => $"{result.CapabilityId}:{result.Code}")
            .ToArray();
        var failedProbeSummary = failedProbes.Length == 0
            ? "none"
            : string.Join(",", failedProbes);

        Log.Error(
            $"[{BridgeRuntime.ModId}] Bridge startup denied by the retail game compatibility gate: " +
            $"health={compatibility.Health}, http_started=false, mutation_enabled=false, " +
            $"code={compatibility.ErrorCode}, profile={compatibility.ProfileId ?? "none"}, " +
            $"assembly=({compatibility.Identity.DisplayName}), failed_probes={failedProbeSummary}. " +
            compatibility.ErrorMessage);
    }

    public static void Shutdown()
    {
        if (Interlocked.Exchange(ref _shutdown, 1) != 0)
        {
            return;
        }

        BridgeServer.Stop();
        BridgeCoordinator.Detach();
        try
        {
            _harmony?.UnpatchAll(BridgeRuntime.HarmonyId);
        }
        catch (Exception ex)
        {
            Log.Warn($"[{BridgeRuntime.ModId}] Failed to remove Harmony patches during shutdown: {ex.GetBaseException().Message}");
        }
    }
}
