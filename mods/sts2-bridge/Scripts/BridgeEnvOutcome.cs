namespace Sts2McpBridge.Scripts;

/// <summary>
/// Projects the game-owned run clock into the canonical terminal run result.
/// </summary>
/// <remarks>
/// <c>RunManager.WinTime</c> is set by the game only when the run is won.
/// Defeat leaves it at zero. This helper intentionally has no player/HP
/// input: the game-over surface may outlive the player combat object.
/// </remarks>
internal static class BridgeEnvOutcome
{
    public static string ResolveRunResult(bool done, long? winTime)
    {
        if (!done)
        {
            return "none";
        }

        if (winTime is null)
        {
            throw new InvalidOperationException(
                "a terminal environment snapshot requires RunManager.WinTime");
        }

        return winTime.Value > 0 ? "victory" : "defeat";
    }
}
