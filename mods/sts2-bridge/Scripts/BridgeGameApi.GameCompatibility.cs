using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Entities.Rewards;
using MegaCrit.Sts2.Core.Runs;

namespace Sts2McpBridge.Scripts;

/// <summary>
/// Narrow adapter for game assembly member renames. Reflection/private access is
/// confined here where practical so the snapshot/action surfaces do not encode
/// version-specific API names throughout the Bridge.
/// </summary>
internal static partial class BridgeGameApi
{
    private static bool IsCombatPlayPhase(
        CombatManager? combatManager,
        CombatState? combatState = null)
    {
        if (combatManager is null || !combatManager.IsInProgress)
        {
            return false;
        }

        combatState ??= GetHiddenFieldValue(combatManager, "_state") as CombatState;
        var player = combatState?.Players.FirstOrDefault();
        if (player is not null)
        {
            try
            {
                return combatManager.IsPartOfPlayerTurn(player);
            }
            catch
            {
                // Fall through to public phase signals while combat is setting up.
            }
        }

        return !combatManager.IsStarting &&
               !combatManager.IsEnemyTurnStarted &&
               !combatManager.EndingPlayerTurnPhaseOne &&
               !combatManager.EndingPlayerTurnPhaseTwo &&
               !combatManager.IsEnding;
    }

    private static void SetUpNewSinglePlayerCompatibility(
        RunManager runManager,
        RunState runState,
        bool shouldSave)
    {
        runManager.SetUpNewSingleplayer(runState, shouldSave, null);
    }

    private static PostAlternateCardRewardAction GetCardRewardSkipActionCompatibility()
    {
        foreach (var name in new[]
                 {
                     "DismissScreenAndKeepReward",
                     "EndSelectionAndCompleteReward",
                     "EndSelectionAndDoNotCompleteReward"
                 })
        {
            if (Enum.TryParse<PostAlternateCardRewardAction>(name, ignoreCase: false, out var action))
            {
                return action;
            }
        }

        throw new BridgeRequestException(
            System.Net.HttpStatusCode.Conflict,
            "card_reward_skip_action_unavailable",
            "The installed game build exposes no recognized card-reward skip action.");
    }
}
