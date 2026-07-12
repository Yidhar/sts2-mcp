using System.Net;

namespace Sts2McpBridge.Scripts;

/// <summary>Contract-v2 projection and atomic player-command surface.</summary>
internal static partial class BridgeGameApi
{
    public static async Task<object> GetStateV2ResponseAsync(CancellationToken cancellationToken = default)
    {
        EnsureDispatcherReady();
        var frontier = await ObserveFrontierAsync(cancellationToken);
        return BuildStateV2Payload(frontier);
    }

    private static object BuildStateV2Payload(ObservedFrontier frontier)
    {
        var legalActions = frontier.Snapshot.Actions
            .Where(static action => !IsDisallowedPlayerActionHandle(action.ActionId))
            .Select(BuildV2LegalActionPayload)
            .ToArray();
        return new
        {
            session_id = BridgeRuntime.SessionId,
            state_version = frontier.Sequence,
            captured_at_utc = DateTimeOffset.UtcNow,
            visibility = "player",
            state = BridgePlayerStateProjector.Project(
                frontier.GetOrCreateStatePayload(),
                HashJsonOptions),
            legal_actions = legalActions
        };
    }

    private static object BuildV2LegalActionPayload(BridgeResolvedAction action) =>
        BridgeLegalActionProjector.Project(action.ActionId, action.Payload);
    private static object BuildSafeV2ConflictDetails(
        ObservedFrontier frontier,
        long? expectedStateVersion,
        string? actionHandle)
    {
        return new
        {
            action_handle = actionHandle,
            expected_state_version = expectedStateVersion,
            current_state_version = frontier.Sequence,
            current_screen = frontier.Snapshot.Fields.Screen,
            legal_action_handles = frontier.Snapshot.Actions
                .Select(static action => action.ActionId)
                .Where(static handle => !IsDisallowedPlayerActionHandle(handle))
                .ToArray()
        };
    }

    private static bool IsDisallowedPlayerActionHandle(string? actionHandle) =>
        !string.IsNullOrWhiteSpace(actionHandle) &&
        (actionHandle.StartsWith("automation:", StringComparison.Ordinal) ||
         actionHandle.Contains("autoslay", StringComparison.OrdinalIgnoreCase));
    public static async Task<object> PerformActionV2AtomicAsync(
        string actionId,
        long expectedStateVersion,
        int? waitAfterMs,
        CancellationToken cancellationToken)
    {
        EnsureDispatcherReady();
        actionId = actionId.Trim();
        if (IsDisallowedPlayerActionHandle(actionId))
        {
            throw new BridgeRequestException(
                HttpStatusCode.Conflict,
                "action_not_available",
                "The requested action is not available to the player-control capability.",
                new { action_handle = actionId });
        }

        var normalizedWaitAfterMs = Math.Clamp(waitAfterMs ?? 0, 0, 5000);

        var execution = await RunOnMainThreadGuardedAsync(
            () =>
            {
                // Capture, publish, validate, re-resolve, and start the mutation in one
                // main-thread work item. No other queued HTTP mutation can pass the
                // BridgeMutationGate while this command is active.
                var currentSnapshot = CaptureSnapshot();
                var before = PublishFrontier(currentSnapshot);
                if (expectedStateVersion != before.Sequence)
                {
                    throw new BridgeRequestException(
                        HttpStatusCode.Conflict,
                        "state_version_conflict",
                        $"Expected state_version {expectedStateVersion}, but the current state_version is {before.Sequence}.",
                        BuildSafeV2ConflictDetails(before, expectedStateVersion, actionId));
                }

                if (!currentSnapshot.ActionLookup.TryGetValue(actionId, out var currentAction))
                {
                    throw new BridgeRequestException(
                        HttpStatusCode.Conflict,
                        "action_not_available",
                        $"Action '{actionId}' is not currently available.",
                        BuildSafeV2ConflictDetails(before, expectedStateVersion: null, actionId));
                }

                try
                {
                    currentAction.Execute();
                }
                catch (Exception ex)
                {
                    throw new BridgeActionOutcomeUnknownException(actionId, ex);
                }

                return before;
            },
            $"v2.perform_action.execute:{actionId}",
            DefaultMainThreadTaskTimeoutMs,
            cancellationToken);

        try
        {
            var after = await WaitForFrontierAfterExecutedActionAsync(
                execution,
                actionId,
                normalizedWaitAfterMs,
                cancellationToken);

            if (IsCombatExitTransition(execution.Snapshot.Fields.Screen, after.Snapshot.Fields.Screen))
            {
                DrainManagedFinalizersLogged("v2.perform_action.combat_exit");
            }

            return new
            {
                ok = true,
                action_handle = actionId,
                state_version_before = execution.Sequence,
                state_version_after = after.Sequence,
                screen_after = after.Snapshot.Fields.Screen,
                state_changed = HasFrontierChanged(execution, after)
            };
        }
        catch (BridgeActionOutcomeUnknownException)
        {
            throw;
        }
        catch (Exception ex)
        {
            // The action already started. Never reclassify a post-execution timeout
            // or observation failure as a safe rejection.
            throw new BridgeActionOutcomeUnknownException(actionId, ex);
        }
    }
}
