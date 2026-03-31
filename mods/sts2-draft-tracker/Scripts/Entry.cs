using HarmonyLib;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Modding;

namespace Sts2DraftTracker.Scripts;

[ModInitializer("Init")]
public static class Entry
{
    private static bool _initialized;

    public static void Init()
    {
        if (_initialized) return;
        _initialized = true;

        var harmony = new Harmony(TrackerRuntime.HarmonyId);
        harmony.PatchAll();

        DraftTracker.Initialize();

        Log.Info($"[{TrackerRuntime.ModId}] Draft Tracker v{TrackerRuntime.Version} initialized. " +
                 $"Logs will be saved to: {TrackerRuntime.OutputDirectory}");
    }
}
