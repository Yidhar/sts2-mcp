using System.Globalization;
using System.Net;
using System.Text.Json.Serialization;
using MegaCrit.Sts2.Core.Nodes;

namespace Sts2McpBridge.Scripts;

internal sealed class BridgeEnvResetRequest
{
    [JsonPropertyName("character")]
    public string? Character { get; set; }

    [JsonPropertyName("rebind_active_run")]
    public bool? RebindActiveRun { get; set; }

    [JsonPropertyName("force_fresh")]
    public bool? ForceFresh { get; set; }

    [JsonPropertyName("defensive_buffs")]
    public bool? DefensiveBuffs { get; set; }

    [JsonPropertyName("timeout_ms")]
    public int? TimeoutMs { get; set; }
}

internal sealed class BridgeEnvStepRequest
{
    [JsonPropertyName("episode_id")]
    public string? EpisodeId { get; set; }

    [JsonPropertyName("action_index")]
    public int? ActionIndex { get; set; }

    [JsonPropertyName("action_id")]
    public string? ActionId { get; set; }

    [JsonPropertyName("timeout_ms")]
    public int? TimeoutMs { get; set; }
}

internal static partial class BridgeGameApi
{
    private const int DefaultEnvResetTimeoutMs = 30000;
    private const int DefaultEnvStepTimeoutMs = 15000;
    private const int MaxEnvTimeoutMs = 120000;
    private const int EnvStableSampleTarget = 2;
    private const int EnvResetTransitionLimit = 24;

    private static readonly object EnvEpisodeSync = new();
    private static BridgeEnvEpisode? _activeEnvEpisode;

    public static object GetEnvSpecResponse()
    {
        return new
        {
            ok = true,
            env_api_version = "bridge-env-v1",
            single_env = true,
            direct_bridge = true,
            model_oriented = true,
            reset = new
            {
                fresh_episode_guarantee = "main_menu_only",
                supported_run_modes = new[] { "standard" },
                supports_character = true,
                supports_defensive_buffs = true,
                supports_seed = false,
                supports_ascension = false
            },
            observation = new
            {
                root_fields = new[] { "phase", "decision_domain", "logic_hash", "run", "player", "combat", "decision" },
                run_fields = new[] { "active", "game_over", "act", "act_id", "act_floor", "floor", "room_type", "room_model", "coord" },
                player_fields = new[] { "character_id", "character_title", "hp", "max_hp", "block", "gold", "deck", "deck_cards", "relics", "potions" },
                decision_domain_values = new[] { "combat", "build", "route" },
                phase_values = new[]
                {
                    "startup_main_menu",
                    "startup_run_mode",
                    "startup_character_select",
                    "combat",
                    "map",
                    "reward",
                    "card_reward",
                    "event",
                    "event_crystal_sphere",
                    "rest_site",
                    "deck_upgrade",
                    "card_selection",
                    "shop",
                    "treasure",
                    "actions",
                    "settling",
                    "terminal",
                    "unknown"
                }
            },
            action_encoding = new
            {
                default_encoding = "legal_action_idx",
                supported = new[] { "legal_action_idx", "action_id" },
                legal_action_shape = new[] { "idx", "action_id", "kind" }
            },
            reward = new
            {
                scalar = "room_settlement_milestone_v2",
                optimized_components = new[]
                {
                    "combat_room_complete_bonus",
                    "combat_room_quality_bonus",
                    "floor_progress_bonus",
                    "elite_clear_bonus",
                    "boss_clear_bonus",
                    "act_clear_bonus",
                    "relic_gain_bonus",
                    "max_hp_gain_bonus",
                    "run_victory_bonus"
                },
                component_scales = new
                {
                    combat_room_complete = EnvRewardCombatWinBonus,
                    room_hp_delta_normalized = EnvRewardRoomHpDeltaWeight,
                    floor_delta = EnvRewardFloorDeltaWeight,
                    elite_clear = EnvRewardEliteClearBonus,
                    boss_clear = EnvRewardBossClearBonus,
                    act_clear = EnvRewardActClearBonus,
                    relic_gain_count = EnvRewardRelicGainWeight,
                    max_hp_gain_normalized = EnvRewardMaxHpGainWeight,
                    death = EnvRewardDeathPenalty,
                    victory = EnvRewardVictoryBonus
                },
                diagnostic_channels = new[]
                {
                    "hp_loss_normalized",
                    "hp_gain_normalized",
                    "room_complete",
                    "combat_room_complete",
                    "room_hp_delta_normalized",
                    "floor_delta",
                    "act_clear",
                    "relic_gain_count",
                    "max_hp_gain_normalized",
                    "death",
                    "victory"
                },
                direct_terms = new
                {
                    action_error_penalty = EnvRewardActionErrorPenalty,
                    truncated_penalty = EnvRewardTruncatedPenalty
                }
            },
            done_conditions = new[] { "run_game_over", "step_timeout" }
        };
    }

    public static async Task<object> ResetEnvResponseAsync(
        BridgeEnvResetRequest? request,
        CancellationToken cancellationToken)
    {
        request ??= new BridgeEnvResetRequest();
        var requestedCharacter = request.Character?.Trim();
        var rebindActiveRun = request.RebindActiveRun == true;
        var forceFresh = request.ForceFresh == true;
        var defensiveBuffs = request.DefensiveBuffs == true;
        var timeoutMs = NormalizeEnvTimeout(request.TimeoutMs, DefaultEnvResetTimeoutMs);
        await WaitForEnvDispatcherReadyAsync(timeoutMs, cancellationToken);
        var executedActions = new List<object>();
        var state = await CaptureEnvSnapshotAsync(
            timeoutMs,
            cancellationToken,
            "env.reset.initial_snapshot");

        var reuseCharacterConstraint = rebindActiveRun ? null : requestedCharacter;
        if (!forceFresh && CanReuseFreshEpisode(state, reuseCharacterConstraint))
        {
            var readyEpisode = CreateEnvEpisode(requestedCharacter, defensiveBuffs);
            state = await ApplyEnvEpisodeAdjustmentsAsync(readyEpisode, state, timeoutMs, cancellationToken);
            return BuildEnvResetPayload(readyEpisode, state, executedActions);
        }

        if (rebindActiveRun &&
            state.RunActive &&
            !IsStartupPhase(state.Phase))
        {
            var rebound = await TryRebindActiveRunAsync(
                state,
                requestedCharacter,
                defensiveBuffs,
                timeoutMs,
                cancellationToken);
            if (rebound is not null)
            {
                return rebound;
            }

            throw new BridgeRequestException(
                HttpStatusCode.Conflict,
                "env_reset_rebind_not_ready",
                "Current run is still transitioning and cannot be rebound yet.",
                new
                {
                    phase = state.Phase,
                    screen = state.Screen,
                    actionable = state.Actionable,
                    done = state.Done,
                    legal_action_count = state.LegalActions.Length
                });
        }

        // If there's an active run outside startup, try to navigate to main menu.
        // This handles mid-game resets (e.g. after truncation, crash recovery, etc.)
        if (state.RunActive && !IsStartupPhase(state.Phase))
        {
            state = await NavigateToMainMenuFromActiveRunAsync(state, timeoutMs, cancellationToken);
        }

        var resetStartedAt = DateTime.UtcNow;
        var waitRetries = 0;
        var forcedMainMenuRecoveryUsed = false;
        const int maxWaitRetries = 5;

        for (var transition = 0; transition < EnvResetTransitionLimit; transition++)
        {
            // Wall-clock guard for the entire reset loop
            if ((DateTime.UtcNow - resetStartedAt).TotalMilliseconds > timeoutMs)
            {
                break;
            }

            if (CanReuseFreshEpisode(state, requestedCharacter))
            {
                var episode = CreateEnvEpisode(requestedCharacter, defensiveBuffs);
                state = await ApplyEnvEpisodeAdjustmentsAsync(episode, state, timeoutMs, cancellationToken);
                return BuildEnvResetPayload(episode, state, executedActions);
            }

            var nextAction = ResolveEnvResetAction(state, requestedCharacter);
            if (nextAction is null)
            {
                if (ShouldWaitForEnvResetPath(state) && waitRetries < maxWaitRetries)
                {
                    state = await WaitForEnvResetPathStateAsync(state, timeoutMs, cancellationToken);
                    waitRetries++;
                    transition--;
                    continue;
                }

                if (!forcedMainMenuRecoveryUsed &&
                    state.RunActive &&
                    !IsStartupPhase(state.Phase))
                {
                    var phaseBeforeRecovery = state.Phase;
                    state = await ForceReturnToMainMenuFromActiveRunAsync(state, timeoutMs, cancellationToken);
                    executedActions.Add(new
                    {
                        action_id = "automation:force_return_to_main_menu",
                        kind = "automation",
                        phase_before = phaseBeforeRecovery
                    });
                    forcedMainMenuRecoveryUsed = true;
                    transition--;
                    continue;
                }

                throw new BridgeRequestException(
                    HttpStatusCode.Conflict,
                    "env_reset_no_reset_path",
                    "Unable to find a reset transition from the current startup state.",
                    new
                    {
                        phase = state.Phase,
                        screen = state.Screen,
                        legal_actions = state.LegalActions
                    });
            }

            try
            {
                await ExecuteEnvActionAsync(
                    nextAction.Value.Action,
                    timeoutMs,
                    cancellationToken,
                    $"env.reset.execute:{nextAction.Value.Action.ActionId}");
                executedActions.Add(new
                {
                    action_id = nextAction.Value.Action.ActionId,
                    kind = nextAction.Value.Kind,
                    phase_before = state.Phase
                });
            }
            catch (OperationCanceledException)
            {
                throw;
            }
            catch (Exception ex)
            {
                executedActions.Add(new
                {
                    action_id = nextAction.Value.Action.ActionId,
                    kind = nextAction.Value.Kind,
                    phase_before = state.Phase,
                    execution_error = ex is BridgeRequestException bridgeEx ? bridgeEx.ErrorCode : ex.GetType().Name
                });
                state = await CaptureEnvSnapshotAsync(
                    timeoutMs,
                    cancellationToken,
                    "env.reset.retry_after_action_error");
                continue;
            }

            state = await WaitForResetAdvanceAsync(
                state,
                nextAction.Value.Action.ActionId,
                requestedCharacter,
                timeoutMs,
                cancellationToken);

            if (IsEnvFreshEpisodeReady(state))
            {
                var episode = CreateEnvEpisode(requestedCharacter, defensiveBuffs);
                state = await ApplyEnvEpisodeAdjustmentsAsync(episode, state, timeoutMs, cancellationToken);
                return BuildEnvResetPayload(episode, state, executedActions);
            }
        }

        var finalState = await CaptureEnvSnapshotAsync(
            timeoutMs,
            cancellationToken,
            "env.reset.final_snapshot");
        if (CanReuseFreshEpisode(finalState, requestedCharacter))
        {
            var episode = CreateEnvEpisode(requestedCharacter, defensiveBuffs);
            finalState = await ApplyEnvEpisodeAdjustmentsAsync(episode, finalState, timeoutMs, cancellationToken);
            return BuildEnvResetPayload(episode, finalState, executedActions);
        }

        throw new BridgeRequestException(
            HttpStatusCode.Conflict,
            "env_reset_transition_limit",
            $"env/reset exceeded the transition limit of {EnvResetTransitionLimit}.",
            new
            {
                transition_limit = EnvResetTransitionLimit,
                phase = finalState.Phase,
                screen = finalState.Screen,
                actionable = finalState.Actionable,
                done = finalState.Done,
                legal_action_count = finalState.LegalActions.Length,
                executed_actions = executedActions
            });
    }

    public static async Task<object> StepEnvResponseAsync(
        BridgeEnvStepRequest? request,
        CancellationToken cancellationToken)
    {
        request ??= new BridgeEnvStepRequest();
        var timeoutMs = NormalizeEnvTimeout(request.TimeoutMs, DefaultEnvStepTimeoutMs);
        await WaitForEnvDispatcherReadyAsync(timeoutMs, cancellationToken);
        var episodeId = request.EpisodeId?.Trim();
        if (string.IsNullOrWhiteSpace(episodeId))
        {
            throw new BridgeRequestException(
                HttpStatusCode.BadRequest,
                "missing_episode_id",
                "Request body must include a non-empty episode_id.");
        }

        var episode = RequireActiveEnvEpisode(episodeId);
        if (episode.Done)
        {
            throw new BridgeRequestException(
                HttpStatusCode.Conflict,
                "episode_already_done",
                $"Episode '{episodeId}' is already done. Call env/reset to start a new episode.");
        }

        var before = await WaitForStableEnvStateAsync(
            null,
            timeoutMs,
            requireActionableOrDone: true,
            cancellationToken);
        before = await ApplyEnvEpisodeAdjustmentsAsync(episode, before, timeoutMs, cancellationToken);

        if (before.Done)
        {
            episode.Done = true;
            return BuildEnvStepPayload(
                episode,
                before,
                before,
                selectedAction: null,
                truncated: false,
                truncationReason: null);
        }

        if (!before.Actionable)
        {
            episode.Done = true;
            return BuildEnvStepPayload(
                episode,
                before,
                before,
                selectedAction: null,
                truncated: true,
                truncationReason: "step_timeout_waiting_for_actionable_or_terminal_state");
        }

        // Resolve and execute the action. All failures are captured as actionError
        // instead of throwing, so the caller always gets a valid step payload.
        BridgeResolvedActionSelection? selectedAction = null;
        string? actionError = null;
        try
        {
            selectedAction = ResolveRequestedEnvAction(before, request);
            await ExecuteEnvActionAsync(
                selectedAction.Action,
                timeoutMs,
                cancellationToken,
                $"env.step.execute:{selectedAction.Action.ActionId}");
        }
        catch (OperationCanceledException)
        {
            throw; // Don't swallow cancellation
        }
        catch (BridgeRequestException ex)
        {
            actionError = ex.ErrorCode;
        }
        catch (Exception)
        {
            actionError = "action_execution_error";
        }

        var after = await WaitForStableEnvStateAsync(
            before.LogicHash,
            timeoutMs,
            requireActionableOrDone: true,
            cancellationToken);
        after = await MaybeAutoConfirmSingleDeckUpgradeAsync(
            selectedAction,
            after,
            timeoutMs,
            cancellationToken);
        after = await ApplyEnvEpisodeAdjustmentsAsync(episode, after, timeoutMs, cancellationToken);

        // Combat sandbox: end episode when combat finishes, skip reward/map screens
        if (episode.EpisodeMode == "combat_sandbox" && !after.Done && IsCombatSandboxEpisodeDone(after))
        {
            episode.StepIndex++;
            episode.Done = true;
            return BuildEnvStepPayload(episode, before, after, selectedAction,
                truncated: false, truncationReason: null, actionError, forceDone: true);
        }

        episode.StepIndex++;
        var truncated = !after.Actionable && !after.Done;
        if (after.Done || truncated)
        {
            episode.Done = true;
        }

        return BuildEnvStepPayload(
            episode,
            before,
            after,
            selectedAction,
            truncated,
            truncated ? "step_timeout_waiting_for_actionable_or_terminal_state" : null,
            actionError);
    }

    private static async Task<BridgeEnvSnapshot> MaybeAutoConfirmSingleDeckUpgradeAsync(
        BridgeResolvedActionSelection? selectedAction,
        BridgeEnvSnapshot snapshot,
        int timeoutMs,
        CancellationToken cancellationToken)
    {
        if (selectedAction is null ||
            !selectedAction.Action.ActionId.StartsWith("deck_upgrade:select:", StringComparison.Ordinal) ||
            !string.Equals(snapshot.Phase, "deck_upgrade", StringComparison.Ordinal) ||
            snapshot.Done)
        {
            return snapshot;
        }

        var deckUpgradeScreen = snapshot.Context.DeckUpgradeScreen;
        if (deckUpgradeScreen is null || !IsNodeVisible(deckUpgradeScreen))
        {
            return snapshot;
        }

        var useSingleSelection = GetHiddenPropertyValue<bool>(deckUpgradeScreen, "UseSingleSelection") ?? false;
        if (!useSingleSelection)
        {
            return snapshot;
        }

        if (!snapshot.ActionLookup.TryGetValue("deck_upgrade:confirm", out var confirmAction))
        {
            return snapshot;
        }

        await ExecuteEnvActionAsync(
            confirmAction,
            timeoutMs,
            cancellationToken,
            "env.step.deck_upgrade_confirm");
        return await WaitForStableEnvStateAsync(
            snapshot.LogicHash,
            timeoutMs,
            requireActionableOrDone: true,
            cancellationToken);
    }

    private static int NormalizeEnvTimeout(int? requestedTimeoutMs, int defaultTimeoutMs)
    {
        return Math.Clamp(requestedTimeoutMs ?? defaultTimeoutMs, 1, MaxEnvTimeoutMs);
    }

    private static async Task WaitForEnvDispatcherReadyAsync(int timeoutMs, CancellationToken cancellationToken)
    {
        var startedAt = DateTime.UtcNow;
        while ((DateTime.UtcNow - startedAt).TotalMilliseconds < timeoutMs)
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (BridgeCoordinator.IsReady)
            {
                return;
            }

            await Task.Delay(50, cancellationToken);
        }

        EnsureDispatcherReady();
    }

    private static BridgeEnvEpisode CreateEnvEpisode(string? requestedCharacter, bool defensiveBuffs)
    {
        var episode = new BridgeEnvEpisode
        {
            Id = Guid.NewGuid().ToString("N", CultureInfo.InvariantCulture),
            StepIndex = 0,
            Done = false,
            RequestedCharacter = string.IsNullOrWhiteSpace(requestedCharacter) ? null : requestedCharacter,
            DefensiveBuffs = defensiveBuffs
        };

        lock (EnvEpisodeSync)
        {
            _activeEnvEpisode = episode;
        }

        return episode;
    }

    private static BridgeEnvEpisode RequireActiveEnvEpisode(string episodeId)
    {
        lock (EnvEpisodeSync)
        {
            if (_activeEnvEpisode is null || !_activeEnvEpisode.Id.Equals(episodeId, StringComparison.Ordinal))
            {
                throw new BridgeRequestException(
                    HttpStatusCode.Conflict,
                    "unknown_episode_id",
                    $"Episode '{episodeId}' is not active. Call env/reset to start a new episode.");
            }

            return _activeEnvEpisode;
        }
    }

    private static bool IsEnvFreshEpisodeReady(BridgeEnvSnapshot snapshot)
    {
        return snapshot.RunActive && !snapshot.Done && !IsStartupPhase(snapshot.Phase) && snapshot.Actionable;
    }

    private static bool CanReuseFreshEpisode(BridgeEnvSnapshot snapshot, string? requestedCharacter)
    {
        if (!IsEnvFreshEpisodeReady(snapshot))
        {
            return false;
        }

        if (string.IsNullOrWhiteSpace(requestedCharacter))
        {
            return true;
        }

        var requested = NormalizeComparableText(requestedCharacter);
        var player = GetPrimaryPlayer(snapshot.Context);
        var currentId = NormalizeComparableText(player?.Character?.Id.ToString());
        var currentTitle = NormalizeComparableText(DescribeCharacter(player?.Character));
        return requested.Equals(currentId, StringComparison.Ordinal) ||
               requested.Equals(currentTitle, StringComparison.Ordinal);
    }

    private static bool IsStartupPhase(string phase)
    {
        return phase is "startup_main_menu" or "startup_run_mode" or "startup_character_select";
    }

    private static bool ShouldWaitForEnvResetPath(BridgeEnvSnapshot snapshot)
    {
        if (snapshot.Done)
        {
            return snapshot.LegalActions.Length == 0;
        }

        // Settling states should always be waited on, even in active runs.
        // The game may be mid-transition (combat ending, reward appearing, etc.)
        if (snapshot.Phase is "settling" or "unknown" || snapshot.Screen == "UNKNOWN")
        {
            return true;
        }

        if (snapshot.RunActive)
        {
            return false;
        }

        return IsStartupPhase(snapshot.Phase) && snapshot.LegalActions.Length == 0;
    }

    private static async Task ExecuteEnvActionAsync(
        BridgeResolvedAction action,
        int timeoutMs,
        CancellationToken cancellationToken,
        string? operationName = null)
    {
        var normalizedOperation = operationName ?? $"env.execute:{action.ActionId}";
        await RunOnMainThreadGuardedAsync(
            () =>
            {
                action.Execute();
                return true;
            },
            normalizedOperation,
            timeoutMs,
            cancellationToken);

        await WaitForPumpTicksGuardedAsync(
            1,
            $"{normalizedOperation}.post_pump",
            timeoutMs,
            cancellationToken);
    }

    private static bool ShouldAutoCloseResidualMapOverlay(BridgeEnvSnapshot snapshot)
    {
        var context = snapshot.Context;
        return context.RunState?.CurrentRoom is not null &&
               context.MapScreen is not null &&
               context.MapScreen.IsOpen &&
               !context.MapScreen.IsTraveling &&
               HasBlockingMapOverlaySurface(context, snapshot.ResolvedActions);
    }

    private static async Task<BridgeEnvSnapshot> MaybeAutoCloseResidualMapOverlayAsync(
        BridgeEnvSnapshot snapshot,
        int timeoutMs,
        CancellationToken cancellationToken)
    {
        if (!ShouldAutoCloseResidualMapOverlay(snapshot))
        {
            return snapshot;
        }

        var closed = await RunOnMainThreadGuardedAsync(
            () =>
            {
                var mapScreen = snapshot.Context.MapScreen;
                if (mapScreen is null ||
                    !mapScreen.IsOpen ||
                    mapScreen.IsTraveling)
                {
                    return false;
                }

                mapScreen.Close(false);
                return true;
            },
            "env.close_residual_map_overlay",
            timeoutMs,
            cancellationToken);

        if (!closed)
        {
            return snapshot;
        }

        await WaitForPumpTicksGuardedAsync(1, "env.close_residual_map_overlay.post_pump", timeoutMs, cancellationToken);
        return await CaptureEnvSnapshotAsync(timeoutMs, cancellationToken, "env.close_residual_map_overlay.snapshot");
    }

    private static async Task<BridgeEnvSnapshot> WaitForEnvResetPathStateAsync(
        BridgeEnvSnapshot before,
        int timeoutMs,
        CancellationToken cancellationToken)
    {
        var startedAt = DateTime.UtcNow;
        var snapshot = before;

        while ((DateTime.UtcNow - startedAt).TotalMilliseconds < timeoutMs)
        {
            cancellationToken.ThrowIfCancellationRequested();
            snapshot = await CaptureEnvSnapshotAsync(timeoutMs, cancellationToken, "env.reset_path.snapshot");
            snapshot = await MaybeAutoCloseResidualMapOverlayAsync(snapshot, timeoutMs, cancellationToken);
            if (!ShouldWaitForEnvResetPath(snapshot))
            {
                return snapshot;
            }

            await WaitForPumpTicksGuardedAsync(1, "env.reset_path.wait_pump", timeoutMs, cancellationToken);
        }

        return snapshot;
    }

    private static async Task<BridgeEnvSnapshot> WaitForStableEnvStateAsync(
        string? baselineLogicHash,
        int timeoutMs,
        bool requireActionableOrDone,
        CancellationToken cancellationToken)
    {
        var startedAt = DateTime.UtcNow;
        var stableHash = string.Empty;
        var stableCount = 0;
        BridgeEnvSnapshot? lastSnapshot = null;

        while ((DateTime.UtcNow - startedAt).TotalMilliseconds < timeoutMs)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var snapshot = await CaptureEnvSnapshotAsync(timeoutMs, cancellationToken, "env.wait_stable.snapshot");
            snapshot = await MaybeAutoCloseResidualMapOverlayAsync(snapshot, timeoutMs, cancellationToken);
            lastSnapshot = snapshot;
            var ready = snapshot.Done || !requireActionableOrDone || snapshot.Actionable;
            var changedFromBaseline = baselineLogicHash is null ||
                                      !baselineLogicHash.Equals(snapshot.LogicHash, StringComparison.Ordinal);

            if (ready && changedFromBaseline)
            {
                if (snapshot.LogicHash.Equals(stableHash, StringComparison.Ordinal))
                {
                    stableCount++;
                }
                else
                {
                    stableHash = snapshot.LogicHash;
                    stableCount = 1;
                }

                if (stableCount >= EnvStableSampleTarget)
                {
                    return snapshot;
                }
            }
            else
            {
                stableHash = string.Empty;
                stableCount = 0;
            }

            await WaitForPumpTicksGuardedAsync(1, "env.wait_stable.wait_pump", timeoutMs, cancellationToken);
        }

        return lastSnapshot ?? await CaptureEnvSnapshotAsync(timeoutMs, cancellationToken, "env.wait_stable.final_snapshot");
    }

    private static async Task<object?> TryRebindActiveRunAsync(
        BridgeEnvSnapshot state,
        string? requestedCharacter,
        bool defensiveBuffs,
        int timeoutMs,
        CancellationToken cancellationToken)
    {
        var snapshot = state;
        var startedAt = DateTime.UtcNow;

        while ((DateTime.UtcNow - startedAt).TotalMilliseconds < timeoutMs)
        {
            cancellationToken.ThrowIfCancellationRequested();

            snapshot = await WaitForStableEnvStateAsync(
                snapshot.LogicHash,
                timeoutMs,
                requireActionableOrDone: true,
                cancellationToken);

            if (!snapshot.RunActive || IsStartupPhase(snapshot.Phase))
            {
                return null;
            }

            if (CanReuseFreshEpisode(snapshot, null))
            {
                var reboundEpisode = CreateEnvEpisode(requestedCharacter, defensiveBuffs);
                snapshot = await ApplyEnvEpisodeAdjustmentsAsync(reboundEpisode, snapshot, timeoutMs, cancellationToken);
                return BuildEnvResetPayload(reboundEpisode, snapshot, Array.Empty<object>());
            }

            if (snapshot.Done)
            {
                return null;
            }

            if (ShouldWaitForEnvResetPath(snapshot))
            {
                snapshot = await WaitForEnvResetPathStateAsync(snapshot, timeoutMs, cancellationToken);
                continue;
            }

            await WaitForPumpTicksGuardedAsync(1, "env.rebind_active_run.wait_pump", timeoutMs, cancellationToken);
            snapshot = await CaptureEnvSnapshotAsync(timeoutMs, cancellationToken, "env.rebind_active_run.snapshot");
        }

        return null;
    }

    private static async Task<BridgeEnvSnapshot> WaitForResetAdvanceAsync(
        BridgeEnvSnapshot before,
        string executedActionId,
        string? requestedCharacter,
        int timeoutMs,
        CancellationToken cancellationToken)
    {
        var executedCharacterSelect = executedActionId.StartsWith("character_select:", StringComparison.Ordinal);
        var executedEmbark = executedActionId.Equals("embark", StringComparison.Ordinal);
        var snapshot = await WaitForStableEnvStateAsync(
            before.LogicHash,
            timeoutMs,
            requireActionableOrDone: true,
            cancellationToken);

        for (var extraWait = 0; extraWait < 4; extraWait++)
        {
            if (IsEnvFreshEpisodeReady(snapshot) || !IsStartupPhase(snapshot.Phase))
            {
                return snapshot;
            }

            if (snapshot.Phase == "startup_character_select")
            {
                var hasEmbark = snapshot.ActionLookup.ContainsKey("embark");
                if ((executedCharacterSelect && !hasEmbark) || executedEmbark)
                {
                    snapshot = await WaitForStableEnvStateAsync(
                        snapshot.LogicHash,
                        timeoutMs,
                        requireActionableOrDone: true,
                        cancellationToken);
                    continue;
                }
            }

            var nextAction = ResolveEnvResetAction(snapshot, requestedCharacter);
            if (nextAction is null ||
                !nextAction.Value.Action.ActionId.Equals(executedActionId, StringComparison.Ordinal))
            {
                return snapshot;
            }

            snapshot = await WaitForStableEnvStateAsync(
                snapshot.LogicHash,
                timeoutMs,
                requireActionableOrDone: true,
                cancellationToken);
        }

        return snapshot;
    }

    /// <summary>
    /// Navigate from a non-startup state back to main menu for env/reset.
    /// Only handles known safe transitions:
    ///   - terminal/game_over → click return to main menu
    ///   - wait for settling states to resolve
    /// Does NOT blindly execute game actions in an active run.
    /// If the game is in an active non-terminal run, returns the state as-is
    /// and lets the reset loop handle it (which will error clearly).
    /// </summary>
    private static async Task<BridgeEnvSnapshot> NavigateToMainMenuFromActiveRunAsync(
        BridgeEnvSnapshot state,
        int timeoutMs,
        CancellationToken cancellationToken)
    {
        var startedAt = DateTime.UtcNow;
        var settlingWaitBudgetMs = Math.Min(timeoutMs, 1500);

        for (var attempt = 0; attempt < 20; attempt++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            if ((DateTime.UtcNow - startedAt).TotalMilliseconds > timeoutMs)
            {
                break;
            }

            // Already at startup — done
            if (IsStartupPhase(state.Phase) || !state.RunActive)
            {
                return state;
            }

            // If the game is in a transient settling state, give it a short chance to
            // resolve before forcing the scene back to the main menu.
            if (ShouldWaitForEnvResetPath(state))
            {
                state = await WaitForEnvResetPathStateAsync(
                    state,
                    settlingWaitBudgetMs,
                    cancellationToken);

                if (IsStartupPhase(state.Phase) || !state.RunActive)
                {
                    return state;
                }
            }

            state = await ForceReturnToMainMenuFromActiveRunAsync(
                state,
                Math.Min(timeoutMs, 10000),
                cancellationToken);

            if (IsStartupPhase(state.Phase) || !state.RunActive)
            {
                return state;
            }
        }

        return await CaptureEnvSnapshotAsync(timeoutMs, cancellationToken, "env.navigate_main_menu.final_snapshot");
    }

    private static async Task<BridgeEnvSnapshot> ForceReturnToMainMenuFromActiveRunAsync(
        BridgeEnvSnapshot state,
        int timeoutMs,
        CancellationToken cancellationToken)
    {
        var returnTask = await RunOnMainThreadGuardedAsync(
            () =>
            {
                var game = NGame.Instance;
                return game?.ReturnToMainMenuAfterRun();
            },
            "env.force_return_to_main_menu.start",
            timeoutMs,
            cancellationToken);

        if (returnTask is not null)
        {
            try
            {
                await returnTask.WaitAsync(
                    TimeSpan.FromMilliseconds(Math.Min(timeoutMs, 8000)),
                    cancellationToken);
            }
            catch (TimeoutException)
            {
                // The return task can legitimately outlive the scheduling call because
                // it performs fade/preload work on the main loop. Fall through to
                // snapshot polling below.
            }
        }

        var snapshot = await WaitForStableEnvStateAsync(
            state.LogicHash,
            timeoutMs,
            requireActionableOrDone: false,
            cancellationToken);

        return await WaitForEnvResetPathStateAsync(
            snapshot,
            timeoutMs,
            cancellationToken);
    }
}
