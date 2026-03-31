namespace Sts2DraftTracker.Scripts;

internal static class TrackerRuntime
{
    public const string ModId = "sts2-draft-tracker";
    public const string HarmonyId = "dev.yidhar.sts2.draft-tracker";
    public const string Version = "0.1.0";

    /// <summary>Output directory for run JSON logs.</summary>
    public static string OutputDirectory { get; } = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData),
        "SlayTheSpire2",
        "draft-tracker");
}
