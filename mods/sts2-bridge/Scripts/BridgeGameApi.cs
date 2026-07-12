using System.Collections;
using System.Buffers.Binary;
using System.Globalization;
using System.Linq;
using System.Net;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.Json.Serialization;
using System.Text.RegularExpressions;
using Godot;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Merchant;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Entities.RestSite;
using MegaCrit.Sts2.Core.Events;
using MegaCrit.Sts2.Core.Events.Custom.CrystalSphereEvent;
using MegaCrit.Sts2.Core.GameActions;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Localization.DynamicVars;
using MegaCrit.Sts2.Core.Map;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.MonsterMoves.Intents;
using MegaCrit.Sts2.Core.MonsterMoves.MonsterMoveStateMachine;
using MegaCrit.Sts2.Core.Multiplayer.Game.PeerInput;
using MegaCrit.Sts2.Core.Nodes;
using MegaCrit.Sts2.Core.Nodes.Cards;
using MegaCrit.Sts2.Core.Nodes.Cards.Holders;
using MegaCrit.Sts2.Core.Nodes.Combat;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.Events;
using MegaCrit.Sts2.Core.Nodes.Events.Custom.CrystalSphere;
using MegaCrit.Sts2.Core.Nodes.Rewards;
using MegaCrit.Sts2.Core.Nodes.RestSite;
using MegaCrit.Sts2.Core.Nodes.Rooms;
using MegaCrit.Sts2.Core.Nodes.Screens;
using MegaCrit.Sts2.Core.Nodes.Screens.CharacterSelect;
using MegaCrit.Sts2.Core.Nodes.Screens.CardSelection;
using MegaCrit.Sts2.Core.Nodes.Screens.GameOverScreen;
using MegaCrit.Sts2.Core.Nodes.Screens.MainMenu;
using MegaCrit.Sts2.Core.Nodes.Screens.Map;
using MegaCrit.Sts2.Core.Nodes.Screens.Overlays;
using MegaCrit.Sts2.Core.Nodes.Screens.ScreenContext;
using MegaCrit.Sts2.Core.Nodes.Screens.Shops;
using MegaCrit.Sts2.Core.Nodes.Screens.TreasureRoomRelic;
using MegaCrit.Sts2.Core.Nodes.TreasureRooms;
using MegaCrit.Sts2.Core.Rewards;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;

namespace Sts2McpBridge.Scripts;

internal sealed class BridgeActionRequest
{
    [JsonPropertyName("action_id")]
    public string? ActionId { get; set; }

    [JsonPropertyName("action")]
    public string? LegacyActionId { get; set; }

    [JsonPropertyName("expected_state_version")]
    public long? ExpectedStateVersion { get; set; }

    [JsonPropertyName("wait_after_ms")]
    public int? WaitAfterMs { get; set; }

    [JsonIgnore]
    public string? RequestedActionId => ActionId ?? LegacyActionId;
}

internal sealed class BridgeRequestException : Exception
{
    public BridgeRequestException(
        HttpStatusCode statusCode,
        string errorCode,
        string message,
        object? details = null)
        : base(message)
    {
        StatusCode = statusCode;
        ErrorCode = errorCode;
        Details = details;
    }

    public HttpStatusCode StatusCode { get; }

    public string ErrorCode { get; }

    public object? Details { get; }
}

internal static partial class BridgeGameApi
{
    private const int NextFrontierWaitTimeoutMs = 5000;
    // Fast combat frontier gate: after play_card/use_potion/card_selection, do
    // not stack long heuristic waits. Re-sample direct CombatManager state and
    // return as soon as the state itself says the player can act
    // (IsPlayPhase && !PlayerActionsDisabled && !IsPaused), or combat exits.
    // The short budget only covers the UI-lock frame while an action resolves.
    private const int ActionableFrontierTimeoutMs = 250;
    private const int ActionableFrontierPollIntervalMs = 16;
    private const int PassiveFrontierWaitTimeoutMs = 1000;
    private const int MaxShopOpenActionsPerRoom = 2;
    private const int DefaultMainThreadTaskTimeoutMs = 3000;
    private const int DefaultPumpWaitTimeoutMs = 2000;
    private const int MaxMainThreadGuardTimeoutMs = 5000;

    private static readonly JsonSerializerOptions HashJsonOptions = new()
    {
        WriteIndented = false
    };
    private static readonly object ShopOpenLimiterSync = new();
    private static readonly HashSet<string> SemanticStateExcludedPropertyNames = new(StringComparer.Ordinal)
    {
        "available_actions",
        "description",
        "effect_preview",
        "dynamic_vars",
        "texts",
        "prompt",
        "label",
        "visible_glossary_source",
        "visible_glossary_texts",
        "visible_glossary",
        "selection_screen_prompt",
        "watchdog_dump"
    };
    private static string? _shopOpenLimiterRoomKey;
    private static int _shopOpenLimiterCount;
    private static long _lastSnapshotAtTickMs;

    // Cumulative self-inflicted HP loss tracker, used by the Python env to isolate
    // player-initiated HP loss (Offering / Bloodletting / Hemokinesis / Curse draw
    // side effects declared via card effect_preview.hp_loss) from enemy damage.
    // Resets when the combat identity changes. Exposed via BuildEnvCombatPayload
    // as `self_inflicted_hp_loss_cumulative`.
    private static readonly object SelfInflictedHpLossSync = new();
    private static WeakReference<object>? _selfInflictedHpLossCombatKey;
    private static double _selfInflictedHpLossCumulative;
    private static int _selfInflictedHpLossLastRound = -1;

    public static long? MillisecondsSinceLastSnapshot
    {
        get
        {
            var capturedAt = Volatile.Read(ref _lastSnapshotAtTickMs);
            return capturedAt <= 0
                ? null
                : Math.Max(0, System.Environment.TickCount64 - capturedAt);
        }
    }

    public static async Task<object> GetStateResponseAsync(CancellationToken cancellationToken = default)
    {
        EnsureDispatcherReady();
        BridgeDebugTrace.Write("get_state requested");

        var frontier = await ObserveFrontierAsync(cancellationToken);
        BridgeDebugTrace.Write($"get_state completed state_version={frontier.Sequence}");
        return frontier.GetOrCreateStatePayload();
    }

    public static void NotifyFrontierPumpTick()
    {
        if (!BridgeCoordinator.IsReady)
        {
            return;
        }

        BridgeFrontierStore.OnPumpTick();
    }

    public static void ResetFrontierState()
    {
        Volatile.Write(ref _lastSnapshotAtTickMs, 0);
        BridgeFrontierStore.Reset();
    }

    public static async Task StreamFrontierEventsAsync(
        HttpListenerResponse response,
        CancellationToken cancellationToken = default)
    {
        EnsureDispatcherReady();
        await BridgeFrontierStore.StreamEventsAsync(response, playerVisibleV2: false, cancellationToken);
    }

    public static async Task StreamFrontierEventsV2Async(
        HttpListenerResponse response,
        CancellationToken cancellationToken = default)
    {
        EnsureDispatcherReady();
        await BridgeFrontierStore.StreamEventsAsync(response, playerVisibleV2: true, cancellationToken);
    }

    public static async Task<object> PerformActionResponseAsync(
        BridgeActionRequest? request,
        CancellationToken cancellationToken)
    {
        EnsureDispatcherReady();

        request ??= new BridgeActionRequest();

        var actionId = request.RequestedActionId?.Trim();
        if (string.IsNullOrWhiteSpace(actionId))
        {
            throw new BridgeRequestException(
                HttpStatusCode.BadRequest,
                "missing_action_id",
                "Request body must include a non-empty action_id.");
        }

        var perfOuterStart = DateTimeOffset.UtcNow;
        var beforeObsStart = perfOuterStart;
        var before = await ObserveFrontierAsync(cancellationToken);
        var beforeObsEnd = DateTimeOffset.UtcNow;
        BridgeDebugTrace.Write($"perform_action snapshot_before action={actionId} version={before.Sequence}");

        if (request.ExpectedStateVersion is long expectedStateVersion &&
            expectedStateVersion != before.Sequence)
        {
            throw new BridgeRequestException(
                HttpStatusCode.Conflict,
                "state_version_conflict",
                $"Expected state_version {expectedStateVersion}, but the current state_version is {before.Sequence}.",
                new
                {
                    action_id = actionId,
                    expected_state_version = expectedStateVersion,
                    current_state_version = before.Sequence,
                    current_state_hash = before.FrontierHash,
                    current_screen = before.Snapshot.Fields.Screen,
                    available_actions = before.Snapshot.ActionPayloads
                });
        }

        if (!before.Snapshot.ActionLookup.TryGetValue(actionId, out var action))
        {
            throw new BridgeRequestException(
                HttpStatusCode.Conflict,
                "action_not_available",
                $"Action '{actionId}' is not currently available.",
                new
                {
                    action_id = actionId,
                    current_state_version = before.Sequence,
                    current_state_hash = before.FrontierHash,
                    current_screen = before.Snapshot.Fields.Screen,
                    available_actions = before.Snapshot.ActionPayloads
                });
        }

        var waitAfterMs = Math.Clamp(request.WaitAfterMs ?? 0, 0, 5000);
        AccumulateSelfInflictedHpLossIfPlayCard(actionId, action);
        var executeStart = DateTimeOffset.UtcNow;
        var after = await ExecuteActionAndWaitForFrontierAsync(
            before,
            actionId,
            action,
            waitAfterMs,
            cancellationToken);
        var executeEnd = DateTimeOffset.UtcNow;
        var autoExecutedActions = new List<object>();

        if (IsRewardResolutionAction(actionId))
        {
            (after, autoExecutedActions) = await MaybeAutoProceedAfterRewardActionAsync(after, cancellationToken);
        }
        else if (IsCardSelectionResolutionAction(actionId))
        {
            (after, autoExecutedActions) = await MaybeAutoCompleteCardSelectionAsync(after, cancellationToken);
        }
        var postProcessEnd = DateTimeOffset.UtcNow;

        DumpPerformActionOuterTiming(
            actionId,
            perfOuterStart,
            beforeObsStart,
            beforeObsEnd,
            executeStart,
            executeEnd,
            postProcessEnd);

        // Combat→post-combat transition is a game-quiescent window equivalent to
        // combat_sandbox's reset moment: the combat's GodotObjects become
        // eligible for finalization here. Draining now prevents the finalize
        // queue from accumulating across a full run's many combats, which was
        // the root cause of the native AV in HashMap._lookup_pos under long
        // training. Triggered only on true combat exit, not per-step.
        if (IsCombatExitTransition(before.Snapshot.Fields.Screen, after.Snapshot.Fields.Screen))
        {
            DrainManagedFinalizersLogged("perform_action.combat_exit");
        }

        BridgeDebugTrace.Write($"perform_action snapshot_after action={actionId} version={after.Sequence}");

        return new
        {
            ok = true,
            action_id = actionId,
            matched_action = BuildResolvedActionAckPayload(action.Payload),
            state_version_before = before.Sequence,
            state_version_after = after.Sequence,
            screen_after = after.Snapshot.Fields.Screen,
            auto_executed_actions = autoExecutedActions,
            state_changed = HasFrontierChanged(before, after)
        };
    }

    private static async Task<ObservedFrontier> ObserveFrontierAsync(CancellationToken cancellationToken)
    {
        BridgeDebugTrace.Write("observe_frontier executing on main thread");
        var snapshot = await RunOnMainThreadGuardedAsync(
            CaptureSnapshot,
            "observe_frontier.capture_snapshot",
            DefaultMainThreadTaskTimeoutMs,
            cancellationToken);
        var frontier = PublishFrontier(snapshot);
        BridgeDebugTrace.Write($"observe_frontier completed version={frontier.Sequence}");
        return frontier;
    }

    private static ObservedFrontier PublishFrontier(BridgeSnapshot snapshot)
    {
        Volatile.Write(ref _lastSnapshotAtTickMs, System.Environment.TickCount64);
        return BridgeFrontierStore.PublishSnapshot(snapshot);
    }

    // Diagnostic-only: unconditional dump of per-phase timing for slow
    // perform_action paths.  Writes a single line to perform_action-timing.log
    // in the bridge session directory whenever total exceeds 1500ms, so we
    // can see where time goes without needing STS2_BRIDGE_DEBUG_TRACE.
    private static readonly object PerformActionTimingSync = new();

    private static void DumpPerformActionOuterTiming(
        string actionId,
        DateTimeOffset outerStart,
        DateTimeOffset beforeObsStart,
        DateTimeOffset beforeObsEnd,
        DateTimeOffset executeStart,
        DateTimeOffset executeEnd,
        DateTimeOffset postProcessEnd)
    {
        var totalMs = (postProcessEnd - outerStart).TotalMilliseconds;
        if (totalMs < 1500.0)
        {
            return;
        }
        var beforeObsMs = (beforeObsEnd - beforeObsStart).TotalMilliseconds;
        var gapMs = (executeStart - beforeObsEnd).TotalMilliseconds;
        var executeMs = (executeEnd - executeStart).TotalMilliseconds;
        var postMs = (postProcessEnd - executeEnd).TotalMilliseconds;
        try
        {
            lock (PerformActionTimingSync)
            {
                Directory.CreateDirectory(BridgeRuntime.SessionDirectoryPath);
                var path = Path.Combine(
                    BridgeRuntime.SessionDirectoryPath,
                    BridgeRuntime.IsMultiInstance
                        ? $"perform_action-outer-{BridgeRuntime.InstanceId}.log"
                        : "perform_action-outer.log");
                File.AppendAllText(
                    path,
                    $"{DateTimeOffset.UtcNow:O} action={actionId} total_ms={totalMs:F0} before_obs_ms={beforeObsMs:F0} gap_ms={gapMs:F0} execute_ms={executeMs:F0} post_ms={postMs:F0}{System.Environment.NewLine}");
            }
        }
        catch
        {
            // Diagnostics must never break gameplay or the bridge.
        }
    }
    private static void DumpPerformActionTiming(
        string actionId,
        string outcome,
        DateTimeOffset start,
        DateTimeOffset executeEnd,
        DateTimeOffset waitAfterEnd,
        int pollCount)
    {
        var now = DateTimeOffset.UtcNow;
        var totalMs = (now - start).TotalMilliseconds;
        if (totalMs < 1500.0)
        {
            return;
        }
        var executeMs = (executeEnd - start).TotalMilliseconds;
        var waitAfterMs = (waitAfterEnd - executeEnd).TotalMilliseconds;
        var pollMs = (now - waitAfterEnd).TotalMilliseconds;
        try
        {
            lock (PerformActionTimingSync)
            {
                Directory.CreateDirectory(BridgeRuntime.SessionDirectoryPath);
                var path = Path.Combine(
                    BridgeRuntime.SessionDirectoryPath,
                    BridgeRuntime.IsMultiInstance
                        ? $"perform_action-timing-{BridgeRuntime.InstanceId}.log"
                        : "perform_action-timing.log");
                File.AppendAllText(
                    path,
                    $"{now:O} action={actionId} outcome={outcome} total_ms={totalMs:F0} execute_ms={executeMs:F0} wait_after_ms={waitAfterMs:F0} poll_ms={pollMs:F0} polls={pollCount}{System.Environment.NewLine}");
            }
        }
        catch
        {
            // Diagnostics must never break gameplay or the bridge.
        }
    }

    private static async Task<ObservedFrontier> ExecuteActionAndWaitForFrontierAsync(
        ObservedFrontier before,
        string actionId,
        BridgeResolvedAction action,
        int waitAfterMs,
        CancellationToken cancellationToken)
    {
        await RunOnMainThreadGuardedAsync(
            () =>
            {
                BridgeDebugTrace.Write($"perform_action executing action={actionId}");
                var actionToExecute = action;
                if (actionId.StartsWith("card_selection:", StringComparison.Ordinal))
                {
                    actionToExecute = ResolveCurrentCardSelectionActionOrThrow(actionId);
                }

                actionToExecute.Execute();
                return true;
            },
            $"perform_action.execute:{actionId}",
            DefaultMainThreadTaskTimeoutMs,
            cancellationToken);

        return await WaitForFrontierAfterExecutedActionAsync(
            before,
            actionId,
            waitAfterMs,
            cancellationToken);
    }

    private static async Task<ObservedFrontier> WaitForFrontierAfterExecutedActionAsync(
        ObservedFrontier before,
        string actionId,
        int waitAfterMs,
        CancellationToken cancellationToken)
    {
        var timingStart = DateTimeOffset.UtcNow;
        var executeEnd = timingStart;

        if (waitAfterMs > 0)
        {
            await Task.Delay(waitAfterMs, cancellationToken);
        }
        var waitAfterEnd = DateTimeOffset.UtcNow;

        // For combat-mutating actions, use direct CombatManager state rather than
        // waiting for end_turn/action-list heuristics. A real end-turn-only state
        // is actionable immediately once PlayerActionsDisabled clears; a transient
        // animation frame is identified by direct disabled/paused/playphase flags.
        var requireActionable = ShouldTryImmediateObservedFrontier(actionId);

        if (requireActionable)
        {
            var pollDeadline = DateTimeOffset.UtcNow.AddMilliseconds(ActionableFrontierTimeoutMs);
            ObservedFrontier latest = before;
            var pollCount = 0;
            while (true)
            {
                var frontier = await ObserveFrontierAsync(cancellationToken);
                pollCount++;
                latest = frontier;
                if (IsFrontierActionableAfterCombatAction(frontier))
                {
                    BridgeDebugTrace.Write(
                        $"perform_action direct_actionable_frontier action={actionId} version={frontier.Sequence}");
                    DumpPerformActionTiming(actionId, "direct_actionable", timingStart, executeEnd, waitAfterEnd, pollCount);
                    return frontier;
                }
                var remainingMs = (pollDeadline - DateTimeOffset.UtcNow).TotalMilliseconds;
                if (remainingMs <= 0)
                {
                    break;
                }
                var delay = (int)Math.Min(ActionableFrontierPollIntervalMs, remainingMs);
                if (delay > 0)
                {
                    await Task.Delay(delay, cancellationToken);
                }
            }

            BridgeDebugTrace.Write(
                $"perform_action direct_actionable_frontier_timeout action={actionId} after_version={before.Sequence}");
            DumpPerformActionTiming(actionId, "direct_timeout", timingStart, executeEnd, waitAfterEnd, pollCount);
            return latest;
        }

        var deadline = DateTimeOffset.UtcNow.AddMilliseconds(NextFrontierWaitTimeoutMs);
        var waitAfterSequence = before.Sequence;
        while (true)
        {
            var remaining = (int)Math.Max(1, (deadline - DateTimeOffset.UtcNow).TotalMilliseconds);
            if (remaining <= 1)
            {
                break;
            }
            var frontier = await BridgeFrontierStore.WaitForNextFrontierAsync(
                waitAfterSequence,
                remaining,
                cancellationToken);
            if (frontier is null)
            {
                break;
            }
            return frontier;
        }

        BridgeDebugTrace.Write(
            $"perform_action frontier_wait_timeout action={actionId} after_version={before.Sequence}");
        return await ObserveFrontierAsync(cancellationToken);
    }

    private static bool IsCardSelectionVisible(BridgeSnapshot snapshot)
    {
        var cardSelection = JsonSerializer.SerializeToElement(snapshot.Fields.CardSelection);
        return cardSelection.TryGetProperty("visible", out var visibleProperty) &&
               visibleProperty.ValueKind == JsonValueKind.True;
    }

    private static BridgeResolvedAction ResolveCurrentCardSelectionActionOrThrow(string actionId)
    {
        var snapshot = CaptureSnapshot();
        if (!IsCardSelectionVisible(snapshot))
        {
            throw new BridgeRequestException(
                HttpStatusCode.Conflict,
                "card_selection_screen_gone",
                $"Card selection screen is no longer visible while executing '{actionId}'.",
                new
                {
                    action_id = actionId,
                    current_screen = snapshot.Fields.Screen,
                    available_actions = snapshot.ActionPayloads
                });
        }

        if (!snapshot.ActionLookup.TryGetValue(actionId, out var currentAction))
        {
            throw new BridgeRequestException(
                HttpStatusCode.Conflict,
                "card_selection_action_not_available_at_execution",
                $"Card selection action '{actionId}' is no longer available at execution time.",
                new
                {
                    action_id = actionId,
                    current_screen = snapshot.Fields.Screen,
                    available_actions = snapshot.ActionPayloads
                });
        }

        return currentAction;
    }

    private static int NormalizeMainThreadGuardTimeout(int timeoutMs, int fallbackMs)
    {
        var normalized = timeoutMs <= 0 ? fallbackMs : timeoutMs;
        return Math.Clamp(Math.Min(normalized, MaxMainThreadGuardTimeoutMs), 250, MaxMainThreadGuardTimeoutMs);
    }

    private static async Task<T> RunOnMainThreadGuardedAsync<T>(
        Func<T> action,
        string operationName,
        int timeoutMs,
        CancellationToken cancellationToken)
    {
        var guardTimeoutMs = NormalizeMainThreadGuardTimeout(timeoutMs, DefaultMainThreadTaskTimeoutMs);
        using var linkedCts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        var mainThreadTask = BridgeCoordinator.RunOnMainThreadAsync(action, linkedCts.Token);

        try
        {
            var completedTask = await Task.WhenAny(
                mainThreadTask,
                Task.Delay(guardTimeoutMs, cancellationToken));
            if (completedTask == mainThreadTask)
            {
                return await mainThreadTask;
            }
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
        {
            throw;
        }
        catch (OperationCanceledException) when (!cancellationToken.IsCancellationRequested)
        {
            throw BuildMainThreadTimeoutException(
                "main_thread_stalled",
                operationName,
                guardTimeoutMs);
        }

        linkedCts.Cancel();
        throw BuildMainThreadTimeoutException(
            "main_thread_stalled",
            operationName,
            guardTimeoutMs);
    }

    private static async Task WaitForPumpTicksGuardedAsync(
        int tickCount,
        string operationName,
        int timeoutMs,
        CancellationToken cancellationToken)
    {
        var guardTimeoutMs = NormalizeMainThreadGuardTimeout(timeoutMs, DefaultPumpWaitTimeoutMs);
        using var linkedCts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        linkedCts.CancelAfter(guardTimeoutMs);

        try
        {
            await BridgeCoordinator.WaitForPumpTicksAsync(tickCount, linkedCts.Token);
        }
        catch (OperationCanceledException) when (!cancellationToken.IsCancellationRequested &&
                                                 linkedCts.IsCancellationRequested)
        {
            throw BuildMainThreadTimeoutException(
                "pump_stalled",
                operationName,
                guardTimeoutMs);
        }
    }

    private static BridgeRequestException BuildMainThreadTimeoutException(
        string errorCode,
        string operationName,
        int timeoutMs)
    {
        return new BridgeRequestException(
            HttpStatusCode.ServiceUnavailable,
            errorCode,
            $"Timed out waiting for bridge operation '{operationName}'.",
            new
            {
                operation = operationName,
                timeout_ms = timeoutMs,
                coordinator = BridgeCoordinator.GetDiagnosticsSnapshot()
            });
    }

    private static async Task<ObservedFrontier> WaitForNextObservedFrontierAsync(
        ObservedFrontier before,
        int timeoutMs,
        CancellationToken cancellationToken)
    {
        var frontier = await BridgeFrontierStore.WaitForNextFrontierAsync(
            before.Sequence,
            timeoutMs,
            cancellationToken);
        return frontier ?? await ObserveFrontierAsync(cancellationToken);
    }

    private static bool ShouldTryImmediateObservedFrontier(string actionId)
    {
        return actionId.StartsWith("play_card:", StringComparison.Ordinal) ||
               actionId.StartsWith("card_selection:", StringComparison.Ordinal) ||
               actionId.StartsWith("use_potion:", StringComparison.Ordinal);
    }

    private static bool HasFrontierChanged(ObservedFrontier before, ObservedFrontier after)
    {
        return before.Sequence != after.Sequence ||
               !before.FrontierHash.Equals(after.FrontierHash, StringComparison.Ordinal);
    }

    // Direct-state actionability. In combat, legal-action shape is not the
    // source of truth during animations; CombatManager flags are. Once combat
    // is in play phase and the UI is neither paused nor action-disabled, even
    // an end_turn-only frontier is a real actionable state and must not be
    // delayed. Card-selection and out-of-combat surfaces are actionable too.
    private static bool IsFrontierActionableAfterCombatAction(ObservedFrontier frontier)
    {
        if (frontier is null)
        {
            return false;
        }

        var snapshot = frontier.Snapshot;
        if (!IsCombatLikeScreen(snapshot.Fields.Screen) || !snapshot.Fields.CombatInProgress)
        {
            return true;
        }

        if (snapshot.Fields.CardSelectionVisible || IsCardSelectionVisible(snapshot))
        {
            return true;
        }

        return snapshot.Fields.CombatIsPlayPhase &&
               !snapshot.Fields.CombatPlayerActionsDisabled &&
               !snapshot.Fields.CombatIsPaused;
    }

    private static object BuildResolvedActionAckPayload(object? payload)
    {
        var targetPayload = ReadPayloadPropertyValue(payload, "target");
        var targetMappingPayload = ReadPayloadPropertyValue(payload, "target_mapping");
        return new
        {
            action_id = ReadPayloadStringProperty(payload, "action_id"),
            kind = ReadPayloadStringProperty(payload, "kind"),
            label = ReadPayloadStringProperty(payload, "label"),
            target_action_suffix = ReadPayloadStringProperty(payload, "target_action_suffix"),
            target_combat_id = ReadPayloadIntegerProperty(payload, "target_combat_id")
                               ?? ReadPayloadIntegerProperty(targetPayload, "combat_id")
                               ?? ReadPayloadIntegerProperty(targetMappingPayload, "combat_id"),
            target_name = ReadPayloadStringProperty(payload, "target_name")
                          ?? ReadPayloadStringProperty(targetPayload, "name")
                          ?? ReadPayloadStringProperty(targetMappingPayload, "name"),
            target_side = ReadPayloadStringProperty(payload, "target_side")
                          ?? ReadPayloadStringProperty(targetPayload, "side")
                          ?? ReadPayloadStringProperty(targetMappingPayload, "side")
        };
    }

    private static string? ReadPayloadStringProperty(object? payload, string propertyName)
    {
        var value = ReadPayloadPropertyValue(payload, propertyName);
        return value as string;
    }

    internal static void ResetSelfInflictedHpLossTrackerForNewCombat(object? combatKey)
    {
        lock (SelfInflictedHpLossSync)
        {
            _selfInflictedHpLossCombatKey = combatKey is null
                ? null
                : new WeakReference<object>(combatKey);
            _selfInflictedHpLossCumulative = 0.0;
            _selfInflictedHpLossLastRound = -1;
        }
    }

    internal static double ObserveSelfInflictedHpLossCumulative(object? combatKey, int currentRound)
    {
        // Called from env payload building. Two fresh-combat detections:
        //   (a) CombatState reference changed — new object, new combat.
        //   (b) Round number rolled back vs. the last observed value — same
        //       object reused across combats (engine object pooling).
        // Either fires a flush so full_run combat→combat transitions don't
        // leak self-damage counters into the next encounter.
        lock (SelfInflictedHpLossSync)
        {
            if (combatKey is null)
            {
                return 0.0;
            }
            object? currentKey = null;
            _selfInflictedHpLossCombatKey?.TryGetTarget(out currentKey);
            var refChanged = !ReferenceEquals(currentKey, combatKey);
            var roundRolledBack =
                _selfInflictedHpLossLastRound >= 0 &&
                currentRound >= 0 &&
                currentRound < _selfInflictedHpLossLastRound;
            if (refChanged || roundRolledBack)
            {
                _selfInflictedHpLossCombatKey = new WeakReference<object>(combatKey);
                _selfInflictedHpLossCumulative = 0.0;
            }
            if (currentRound >= 0)
            {
                _selfInflictedHpLossLastRound = currentRound;
            }
            return _selfInflictedHpLossCumulative;
        }
    }

    private static void AccumulateSelfInflictedHpLossIfPlayCard(
        string actionId, BridgeResolvedAction action)
    {
        if (string.IsNullOrEmpty(actionId) || !actionId.StartsWith("play_card:", StringComparison.Ordinal))
        {
            return;
        }

        double hpLoss;
        try
        {
            var payload = JsonSerializer.SerializeToElement(action.Payload);
            var fromCard = TryGetNestedInt(payload, "card", "effect_preview", "hp_loss");
            var fromTop = TryGetNestedInt(payload, "effect_preview", "hp_loss");
            var resolved = fromCard ?? fromTop;
            if (!resolved.HasValue || resolved.Value <= 0)
            {
                return;
            }
            hpLoss = resolved.Value;
        }
        catch
        {
            // Malformed payload — skip accumulation rather than crash the request path.
            return;
        }

        lock (SelfInflictedHpLossSync)
        {
            _selfInflictedHpLossCumulative += hpLoss;
        }
    }

    private static int? ReadPayloadIntegerProperty(object? payload, string propertyName)
    {
        var value = ReadPayloadPropertyValue(payload, propertyName);
        if (value is byte byteValue)
        {
            return byteValue;
        }

        if (value is sbyte sbyteValue)
        {
            return sbyteValue;
        }

        if (value is short shortValue)
        {
            return shortValue;
        }

        if (value is ushort ushortValue)
        {
            return ushortValue;
        }

        if (value is int intValue)
        {
            return intValue;
        }

        if (value is uint uintValue && uintValue <= int.MaxValue)
        {
            return (int)uintValue;
        }

        if (value is long longValue && longValue >= int.MinValue && longValue <= int.MaxValue)
        {
            return (int)longValue;
        }

        if (value is ulong ulongValue && ulongValue <= int.MaxValue)
        {
            return (int)ulongValue;
        }

        if (value is null)
        {
            return null;
        }

        try
        {
            return Convert.ToInt32(value, CultureInfo.InvariantCulture);
        }
        catch
        {
            return null;
        }
    }

    private static bool? ReadPayloadBooleanProperty(object? payload, string propertyName)
    {
        var value = ReadPayloadPropertyValue(payload, propertyName);
        if (value is bool boolValue)
        {
            return boolValue;
        }

        if (value is null)
        {
            return null;
        }

        try
        {
            return Convert.ToBoolean(value, CultureInfo.InvariantCulture);
        }
        catch
        {
            return null;
        }
    }

    private static object? ReadPayloadPropertyValue(object? payload, string propertyName)
    {
        if (payload is null || string.IsNullOrWhiteSpace(propertyName))
        {
            return null;
        }

        var property = payload.GetType().GetProperty(
            propertyName,
            BindingFlags.Instance | BindingFlags.Public | BindingFlags.IgnoreCase);
        return property?.GetValue(payload);
    }

    private static BridgeResolvedAction[] GetNonAutomationActions(BridgeSnapshot snapshot)
    {
        return snapshot.Actions
            .Where(action => !action.ActionId.StartsWith("automation:", StringComparison.Ordinal))
            .ToArray();
    }

    private static BridgeSnapshot CaptureSnapshot()
    {
        return HydrateSnapshot(CaptureFrontierCandidate());
    }

    private static BridgeFrontierCandidate CaptureFrontierCandidate()
    {
        BridgeDebugTrace.Write("capture_frontier_candidate start");
        var context = CaptureContext();
        BridgeDebugTrace.Write($"capture_frontier_candidate context screen={context.Screen}");
        var rawActions = BuildResolvedActions(context);
        var actions = FilterActionsForStableSurface(context, rawActions);
        BridgeDebugTrace.Write($"capture_frontier_candidate actions={actions.Count} raw_actions={rawActions.Count}");
        var frontierHash = ComputeFrontierHash(BuildFrontierFingerprint(context, actions));
        BridgeDebugTrace.Write($"capture_frontier_candidate complete hash={frontierHash}");

        return new BridgeFrontierCandidate
        {
            Context = context,
            Actions = actions,
            FrontierHash = frontierHash
        };
    }

    private static BridgeSnapshot HydrateSnapshot(BridgeFrontierCandidate candidate)
    {
        BridgeDebugTrace.Write($"hydrate_snapshot start frontier_hash={candidate.FrontierHash}");
        var actionPayloads = candidate.Actions.Select(static action => action.Payload).ToArray();
        var fields = BuildStateFields(candidate.Context, actionPayloads);
        BridgeDebugTrace.Write($"hydrate_snapshot complete frontier_hash={candidate.FrontierHash}");

        return new BridgeSnapshot
        {
            Fields = fields,
            FrontierHash = candidate.FrontierHash,
            Actions = candidate.Actions,
            ActionPayloads = actionPayloads,
            ActionLookup = candidate.Actions.ToDictionary(static action => action.ActionId, StringComparer.Ordinal)
        };
    }

    private static object BuildFrontierProbePayload(
        BridgeWorldContext context,
        IReadOnlyList<BridgeResolvedAction> actions)
    {
        return new
        {
            screen = context.Screen,
            action_ids = actions.Select(static action => action.ActionId).ToArray(),
            combat = BuildCombatFrontierPayload(context.CombatManager, context.CombatState),
            rewards = BuildRewardsFrontierPayload(context),
            card_reward_selection = BuildCardRewardSelectionFrontierPayload(
                context.CardRewardScreen,
                context.CardRewardOptions,
                context.CardRewardSkipButton),
            card_selection = BuildCardSelectionFrontierPayload(
                context.CardSelectionScreen,
                context.CardSelectionOptions,
                context.CardSelectionConfirmButton,
                context.CardSelectionCancelButton,
                context.CardSelectionCloseButton,
                context.CardSelectionSkipButton),
            crystal_sphere = BuildCrystalSphereFrontierPayload(
                context.CrystalSphereScreen,
                context.CrystalSphereCells),
            map = BuildMapFrontierPayload(context.RunState, context.MapScreen, context.Screen, context.CombatManager),
            rest_site = BuildRestSiteFrontierPayload(
                context.MapScreen,
                context.RestSiteRoom,
                context.RestSiteButtons,
                context.RestSiteProceedButton),
            deck_upgrade_selection = BuildDeckUpgradeSelectionFrontierPayload(
                context.DeckUpgradeScreen,
                context.DeckUpgradeOptions,
                context.DeckUpgradeConfirmButton,
                context.DeckUpgradeCancelButton,
                context.DeckUpgradeCloseButton),
            shop = BuildShopFrontierPayload(
                context.MerchantRoom,
                context.MerchantInventory,
                context.MerchantSlots,
                context.MerchantButton,
                context.MerchantProceedButton,
                context.MerchantBackButton)
        };
    }

    private static string BuildFrontierFingerprint(
        BridgeWorldContext context,
        IReadOnlyList<BridgeResolvedAction> actions)
    {
        var builder = new StringBuilder(2048);

        builder.Append("screen=").Append(context.Screen);
        AppendRunStateFingerprint(builder, context.RunState);
        AppendMapStateFingerprint(builder, context);
        AppendCombatStateFingerprint(builder, context);
        AppendCardSelectionStateFingerprint(builder, context);
        AppendDeckUpgradeStateFingerprint(builder, context);
        AppendCrystalSphereStateFingerprint(builder, context);
        AppendActionSetFingerprint(builder, actions);

        return builder.ToString();
    }

    private static void AppendRunStateFingerprint(StringBuilder builder, RunState? runState)
    {
        builder.Append("|run=");
        if (runState is null)
        {
            builder.Append("none");
            return;
        }

        builder.Append(runState.Act?.Id.ToString() ?? string.Empty)
            .Append(';').Append(runState.ActFloor)
            .Append(';').Append(runState.TotalFloor)
            .Append(';').Append(runState.CurrentRoom?.RoomType.ToString() ?? string.Empty)
            .Append(';').Append(runState.CurrentRoom?.ModelId?.ToString() ?? string.Empty)
            .Append(';').Append(runState.CurrentRoom?.IsPreFinished == true ? '1' : '0');

        if (runState.CurrentMapCoord.HasValue)
        {
            builder.Append(';');
            AppendMapCoordFingerprint(builder, runState.CurrentMapCoord.Value);
        }
    }

    private static void AppendMapStateFingerprint(StringBuilder builder, BridgeWorldContext context)
    {
        var mapScreen = context.MapScreen;
        builder.Append("|map=");
        if (mapScreen is null)
        {
            builder.Append("none");
            return;
        }

        builder.Append(mapScreen.IsOpen ? '1' : '0')
            .Append(';').Append(mapScreen.IsTravelEnabled ? '1' : '0')
            .Append(';').Append(mapScreen.IsTraveling ? '1' : '0');

        if (mapScreen.IsOpen && mapScreen.IsTravelEnabled && !mapScreen.IsTraveling)
        {
            foreach (var pointNode in context.MapPoints)
            {
                if (!IsMapPointTravelable(pointNode) ||
                    IsCurrentMapCoord(context.RunState, pointNode.Point.coord))
                {
                    continue;
                }

                builder.Append('|').Append("travel:");
                AppendMapCoordFingerprint(builder, pointNode.Point.coord);
                builder.Append(':').Append(pointNode.Point.PointType);
                builder.Append(':').Append(pointNode.State);
            }
        }
    }

    private static void AppendCombatStateFingerprint(StringBuilder builder, BridgeWorldContext context)
    {
        var combatManager = context.CombatManager;
        var combatState = context.CombatState;
        builder.Append("|combat=");
        if (combatManager is null || combatState is null || !combatManager.IsInProgress)
        {
            builder.Append("none");
            return;
        }

        builder.Append(IsCombatPlayPhase(combatManager, combatState) ? '1' : '0')
            .Append(';').Append(combatManager.PlayerActionsDisabled ? '1' : '0')
            .Append(';').Append(combatState.RoundNumber)
            .Append(';').Append(combatState.CurrentSide);

        foreach (var player in combatState.Players)
        {
            var playerCreatureCombatId = player.Creature is null
                ? -1
                : Convert.ToInt32(player.Creature.CombatId, CultureInfo.InvariantCulture);
            builder.Append("|p:")
                .Append(player.NetId)
                .Append(':').Append(playerCreatureCombatId)
                .Append(':').Append(player.Creature?.CurrentHp ?? -1)
                .Append('/').Append(player.Creature?.MaxHp ?? -1)
                .Append(':').Append(player.Creature?.Block ?? -1)
                .Append(':').Append(player.PlayerCombatState?.Energy ?? -1)
                .Append('/').Append(player.PlayerCombatState?.MaxEnergy ?? -1)
                .Append(':').Append(player.PlayerCombatState?.Stars ?? -1);

            if (player.Creature is not null)
            {
                AppendPowerSetFingerprint(builder, player.Creature.Powers);
            }

            var handCards = player.PlayerCombatState?.Hand?.Cards;
            builder.Append(":hand=").Append(handCards?.Count ?? 0);
            if (handCards is not null)
            {
                foreach (var card in handCards)
                {
                    AppendCardFingerprint(builder, card);
                }
            }

            builder.Append(":draw=").Append(player.PlayerCombatState?.DrawPile?.Cards.Count ?? 0)
                .Append(":discard=").Append(player.PlayerCombatState?.DiscardPile?.Cards.Count ?? 0)
                .Append(":exhaust=").Append(player.PlayerCombatState?.ExhaustPile?.Cards.Count ?? 0);
        }

        foreach (var creature in combatState.Creatures.Where(static creature => creature.IsEnemy))
        {
            builder.Append("|e:")
                .Append(creature.ModelId)
                .Append(':').Append(creature.CombatId)
                .Append(':').Append(creature.CurrentHp)
                .Append('/').Append(creature.MaxHp)
                .Append(':').Append(creature.Block)
                .Append(':').Append(creature.IsAlive ? '1' : '0');

            AppendPowerSetFingerprint(builder, creature.Powers);
            AppendEnemyIntentFingerprint(builder, creature);
        }
    }

    private static void AppendCardSelectionStateFingerprint(StringBuilder builder, BridgeWorldContext context)
    {
        var screen = context.CardSelectionScreen;
        var visible = screen is not null && IsNodeVisible(screen);
        builder.Append("|cardsel=").Append(visible ? '1' : '0');
        if (!visible)
        {
            return;
        }

        var prefs = GetHiddenFieldValue(screen, "_prefs");
        builder.Append(';').Append(screen!.GetType().Name)
            .Append(';').Append(CountSelectedCardSelectionCards(screen))
            .Append('/').Append(GetHiddenPropertyValue<int>(prefs, "MinSelect"))
            .Append('/').Append(GetHiddenPropertyValue<int>(prefs, "MaxSelect"))
            .Append(';').Append(IsNodeVisible(context.CardSelectionConfirmButton) && IsButtonEnabled(context.CardSelectionConfirmButton) ? '1' : '0')
            .Append(';').Append(IsNodeVisible(context.CardSelectionCancelButton) && IsButtonEnabled(context.CardSelectionCancelButton) ? '1' : '0')
            .Append(';').Append(IsNodeVisible(context.CardSelectionCloseButton) && IsButtonEnabled(context.CardSelectionCloseButton) ? '1' : '0')
            .Append(';').Append(IsNodeVisible(context.CardSelectionSkipButton) && IsButtonEnabled(context.CardSelectionSkipButton) ? '1' : '0');

        for (var index = 0; index < context.CardSelectionOptions.Count; index++)
        {
            var holder = context.CardSelectionOptions[index];
            if (!IsNodeVisible(holder))
            {
                continue;
            }

            var optionIndex = GetCardSelectionOptionIndex(screen, holder, index);
            var selectionId = GetCardSelectionOptionSelectionId(screen, holder, optionIndex) ??
                              optionIndex.ToString(CultureInfo.InvariantCulture);
            builder.Append("|csopt:")
                .Append(selectionId)
                .Append(':').Append(IsCardSelectionCardSelected(screen, holder.CardModel) ? '1' : '0');
            AppendCardFingerprint(builder, holder.CardModel);
        }
    }

    private static void AppendDeckUpgradeStateFingerprint(StringBuilder builder, BridgeWorldContext context)
    {
        var screen = context.DeckUpgradeScreen;
        var visible = screen is not null && IsNodeVisible(screen);
        builder.Append("|upgrade=").Append(visible ? '1' : '0');
        if (!visible)
        {
            return;
        }

        builder.Append(';').Append(CountSelectedDeckUpgradeCards(screen))
            .Append(';').Append(IsNodeVisible(context.DeckUpgradeConfirmButton) && IsButtonEnabled(context.DeckUpgradeConfirmButton) ? '1' : '0')
            .Append(';').Append(IsNodeVisible(context.DeckUpgradeCancelButton) && IsButtonEnabled(context.DeckUpgradeCancelButton) ? '1' : '0')
            .Append(';').Append(IsNodeVisible(context.DeckUpgradeCloseButton) && IsButtonEnabled(context.DeckUpgradeCloseButton) ? '1' : '0');

        foreach (var holder in context.DeckUpgradeOptions)
        {
            if (!IsNodeVisible(holder))
            {
                continue;
            }

            builder.Append("|upopt:")
                .Append(IsDeckUpgradeCardSelected(screen, holder.CardModel) ? '1' : '0');
            AppendCardFingerprint(builder, holder.CardModel);
        }
    }

    private static void AppendCrystalSphereStateFingerprint(StringBuilder builder, BridgeWorldContext context)
    {
        var screen = context.CrystalSphereScreen;
        var visible = screen is not null && IsNodeVisible(screen);
        builder.Append("|sphere=").Append(visible ? '1' : '0');
        if (!visible)
        {
            return;
        }

        var minigame = GetCrystalSphereMinigame(screen);
        builder.Append(';').Append(GetCrystalSphereDivinationCount(minigame))
            .Append(';').Append(GetCrystalSphereToolName(minigame))
            .Append(';').Append(GetCrystalSphereIsFinished(minigame) ? '1' : '0')
            .Append(';').Append(IsNodeVisible(context.CrystalSphereSmallDivinationButton) && IsButtonEnabled(context.CrystalSphereSmallDivinationButton) ? '1' : '0')
            .Append(';').Append(IsNodeVisible(context.CrystalSphereBigDivinationButton) && IsButtonEnabled(context.CrystalSphereBigDivinationButton) ? '1' : '0')
            .Append(';').Append(IsNodeVisible(context.CrystalSphereProceedButton) && IsButtonEnabled(context.CrystalSphereProceedButton) ? '1' : '0');

        foreach (var cell in context.CrystalSphereCells)
        {
            builder.Append("|cell:")
                .Append(cell.Entity?.X ?? -1)
                .Append(',').Append(cell.Entity?.Y ?? -1)
                .Append(':').Append(cell.Entity?.IsHidden ?? true ? '1' : '0')
                .Append(':').Append(cell.Entity?.IsHighlighted ?? false ? '1' : '0');
        }
    }

    private static void AppendActionSetFingerprint(StringBuilder builder, IReadOnlyList<BridgeResolvedAction> actions)
    {
        builder.Append("|actions=").Append(actions.Count);
        foreach (var action in actions)
        {
            builder.Append("|a:").Append(action.ActionId);
            AppendActionPayloadFingerprint(builder, action.Payload);
        }
    }

    private static void AppendActionPayloadFingerprint(StringBuilder builder, object? payload)
    {
        builder.Append(':').Append(ReadPayloadStringProperty(payload, "kind") ?? string.Empty);

        var rewardPayload = ReadPayloadPropertyValue(payload, "reward");
        if (rewardPayload is not null)
        {
            builder.Append(":rw=")
                .Append(ReadPayloadStringProperty(rewardPayload, "reward_type")
                        ?? ReadPayloadStringProperty(rewardPayload, "type")
                        ?? string.Empty)
                .Append(':').Append(ReadPayloadIntegerProperty(rewardPayload, "amount") ?? -1)
                .Append(':').Append(NormalizeComparableText(ReadPayloadStringProperty(rewardPayload, "description")));
        }

        var cardPayload = ReadPayloadPropertyValue(payload, "card");
        if (cardPayload is not null)
        {
            builder.Append(":card=")
                .Append(ReadPayloadStringProperty(cardPayload, "id") ?? string.Empty)
                .Append(':').Append(ReadPayloadIntegerProperty(cardPayload, "resolved_energy_cost") ?? -1)
                .Append(':').Append(ReadPayloadIntegerProperty(cardPayload, "current_star_cost") ?? -1);
        }

        var optionPayload = ReadPayloadPropertyValue(payload, "option");
        if (optionPayload is not null)
        {
            builder.Append(":opt=")
                .Append(ReadPayloadStringProperty(optionPayload, "option_id") ?? string.Empty)
                .Append(':').Append(NormalizeComparableText(ReadPayloadStringProperty(optionPayload, "title")))
                .Append(':').Append(ReadPayloadBooleanProperty(optionPayload, "is_enabled") == true ? '1' : '0');
        }

        var itemPayload = ReadPayloadPropertyValue(payload, "item");
        if (itemPayload is not null)
        {
            builder.Append(":item=")
                .Append(ReadPayloadStringProperty(itemPayload, "item_kind") ?? string.Empty)
                .Append(':').Append(NormalizeComparableText(ReadPayloadStringProperty(itemPayload, "title")))
                .Append(':').Append(ReadPayloadIntegerProperty(itemPayload, "cost") ?? -1)
                .Append(':').Append(ReadPayloadBooleanProperty(itemPayload, "is_affordable") == true ? '1' : '0');
        }

        var relicPayload = ReadPayloadPropertyValue(payload, "relic");
        if (relicPayload is not null)
        {
            builder.Append(":relic=")
                .Append(ReadPayloadStringProperty(relicPayload, "id") ?? string.Empty)
                .Append(':').Append(NormalizeComparableText(ReadPayloadStringProperty(relicPayload, "title")));
        }

        var coordPayload = ReadPayloadPropertyValue(payload, "coord");
        if (coordPayload is not null)
        {
            builder.Append(":coord=")
                .Append(ReadPayloadIntegerProperty(coordPayload, "col") ?? -1)
                .Append(',').Append(ReadPayloadIntegerProperty(coordPayload, "row") ?? -1);
        }

        var pointType = ReadPayloadStringProperty(payload, "point_type");
        if (!string.IsNullOrWhiteSpace(pointType))
        {
            builder.Append(":pt=").Append(pointType);
        }

        var selectionPrompt = ReadPayloadStringProperty(payload, "selection_prompt");
        if (!string.IsNullOrWhiteSpace(selectionPrompt))
        {
            builder.Append(":prompt=").Append(NormalizeComparableText(selectionPrompt));
        }

        var buttonText = ReadPayloadStringProperty(payload, "button_text");
        if (!string.IsNullOrWhiteSpace(buttonText))
        {
            builder.Append(":btn=").Append(NormalizeComparableText(buttonText));
        }
    }

    private static void AppendMapCoordFingerprint(StringBuilder builder, MapCoord coord)
    {
        builder.Append(coord.col).Append(',').Append(coord.row);
    }

    private static void AppendCardFingerprint(StringBuilder builder, CardModel? card)
    {
        if (card is null)
        {
            builder.Append(":card=<missing>");
            return;
        }

        builder.Append(":card=")
            .Append(card.Id.ToString())
            .Append('/').Append(card.EnergyCost.GetResolved())
            .Append('/').Append(SafeResolveCardStarCost(card) ?? -1)
            .Append('/').Append(card.EnergyCost.CostsX ? '1' : '0')
            .Append('/').Append(SafeGetCardIsPlayable(card) ? '1' : '0');
    }

    private static void AppendPowerSetFingerprint(StringBuilder builder, IEnumerable<PowerModel> powers)
    {
        foreach (var power in powers)
        {
            builder.Append(":pow=")
                .Append(NormalizeComparableText(TextOf(power.Title)))
                .Append('/').Append(power.Amount)
                .Append('/').Append(power.DisplayAmount);
        }
    }

    private static void AppendEnemyIntentFingerprint(StringBuilder builder, Creature creature)
    {
        var monster = creature.Monster;
        if (monster?.NextMove is null)
        {
            builder.Append(":intent=none");
            return;
        }

        var nextMove = monster.NextMove;
        builder.Append(":intent=")
            .Append(nextMove.StateId ?? string.Empty)
            .Append('/').Append(nextMove.FollowUpStateId ?? string.Empty)
            .Append('/').Append(nextMove.IsMove ? '1' : '0');

        var targets = ResolveMonsterIntentTargets(creature);
        foreach (var intent in SafeGetMonsterIntents(monster, nextMove))
        {
            var repeats = intent switch
            {
                SingleAttackIntent singleAttackIntent => singleAttackIntent.Repeats,
                MultiAttackIntent multiAttackIntent => multiAttackIntent.Repeats,
                _ => 1
            };

            var totalDamage = intent switch
            {
                SingleAttackIntent singleAttackIntent => SafeGetIntentTotalDamage(singleAttackIntent, targets, creature),
                MultiAttackIntent multiAttackIntent => SafeGetIntentTotalDamage(multiAttackIntent, targets, creature),
                _ => null
            };

            builder.Append(":i=")
                .Append(intent.IntentType)
                .Append('/').Append(repeats)
                .Append('/').Append(totalDamage ?? -1);
        }
    }

}
