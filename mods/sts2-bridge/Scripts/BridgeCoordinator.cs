using System.Collections.Concurrent;
using Godot;
using HarmonyLib;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Nodes;
using MegaCrit.Sts2.Core.Nodes.CommonUi;

namespace Sts2McpBridge.Scripts;

internal static class BridgeCoordinator
{
    private const int MaxQueuedMainThreadWorkItems = 256;
    private const int MaxWorkItemsPerPump = 32;
    private const int MaxPumpWorkMilliseconds = 4;

    private static readonly object Sync = new();
    private static readonly ConcurrentQueue<IMainThreadWorkItem> Queue = new();
    private static readonly List<PumpTickWaiter> PumpTickWaiters = new();
    private static bool _isAttached;
    private static long _pumpTick;
    private static long _lastPumpAtMs = System.Environment.TickCount64;
    private static int _queuedWorkItems;
    private static int _pumpActive;

    private interface IMainThreadWorkItem
    {
        void TryExecute();

        void TryFail(Exception exception);
    }

    private sealed class MainThreadWorkItem<T> : IMainThreadWorkItem
    {
        private readonly Func<T> _action;
        private readonly TaskCompletionSource<T> _completionSource;
        private CancellationTokenRegistration _cancellationRegistration;
        private int _completionState;

        public MainThreadWorkItem(
            Func<T> action,
            TaskCompletionSource<T> completionSource,
            CancellationToken cancellationToken)
        {
            _action = action;
            _completionSource = completionSource;
            if (cancellationToken.CanBeCanceled)
            {
                _cancellationRegistration = cancellationToken.Register(() =>
                {
                    if (Interlocked.Exchange(ref _completionState, 1) != 0)
                    {
                        return;
                    }

                    _cancellationRegistration.Dispose();
                    _completionSource.TrySetCanceled(cancellationToken);
                });
            }
        }

        public void TryExecute()
        {
            if (Interlocked.Exchange(ref _completionState, 1) != 0)
            {
                _cancellationRegistration.Dispose();
                return;
            }

            try
            {
                _completionSource.TrySetResult(_action());
            }
            catch (Exception ex)
            {
                _completionSource.TrySetException(ex);
            }
            finally
            {
                _cancellationRegistration.Dispose();
            }
        }

        public void TryFail(Exception exception)
        {
            if (Interlocked.Exchange(ref _completionState, 1) != 0)
            {
                _cancellationRegistration.Dispose();
                return;
            }

            try
            {
                _completionSource.TrySetException(exception);
            }
            finally
            {
                _cancellationRegistration.Dispose();
            }
        }
    }

    private sealed class PumpTickWaiter
    {
        public required long TargetTick { get; init; }

        public required TaskCompletionSource<bool> CompletionSource { get; init; }

        public CancellationTokenRegistration CancellationRegistration { get; set; }

        public void Complete()
        {
            CancellationRegistration.Dispose();
            CompletionSource.TrySetResult(true);
        }

        public void Cancel(CancellationToken cancellationToken)
        {
            CancellationRegistration.Dispose();
            CompletionSource.TrySetCanceled(cancellationToken);
        }
    }

    public static bool IsReady
    {
        get
        {
            lock (Sync)
            {
                return _isAttached &&
                       NGame.Instance is not null &&
                       GodotObject.IsInstanceValid(NGame.Instance);
            }
        }
    }

    public static long PumpTick
    {
        get
        {
            lock (Sync)
            {
                return _pumpTick;
            }
        }
    }

    public static long MillisecondsSinceLastPump
    {
        get
        {
            lock (Sync)
            {
                return Math.Max(0, System.Environment.TickCount64 - _lastPumpAtMs);
            }
        }
    }

    public static int QueueDepth => Math.Max(0, Volatile.Read(ref _queuedWorkItems));

    public static object GetDiagnosticsSnapshot()
    {
        lock (Sync)
        {
            var nowMs = System.Environment.TickCount64;
            return new
            {
                attached = _isAttached,
                ready = _isAttached &&
                        NGame.Instance is not null &&
                        GodotObject.IsInstanceValid(NGame.Instance),
                pump_tick = _pumpTick,
                ms_since_last_pump = Math.Max(0, nowMs - _lastPumpAtMs),
                queued_main_thread_work = Math.Max(0, Volatile.Read(ref _queuedWorkItems)),
                max_queued_main_thread_work = MaxQueuedMainThreadWorkItems,
                max_work_items_per_pump = MaxWorkItemsPerPump,
                max_pump_work_ms = MaxPumpWorkMilliseconds,
                waiting_for_pump_tick = PumpTickWaiters.Count
            };
        }
    }

    public static void EnsureAttached(NGame? game)
    {
        if (game is null || !GodotObject.IsInstanceValid(game))
        {
            return;
        }

        lock (Sync)
        {
            if (_isAttached)
            {
                return;
            }

            _isAttached = true;

            // Keep the game running at full speed when the window loses focus.
            // By default Godot throttles unfocused windows (Engine.MaxFps is 0 but
            // low_processor_usage_mode clamps to a long sleep, ~10-30fps), which
            // balloons card-resolution / enemy-turn animation time from <1s to
            // 8-25s and causes the combat sandbox after-wait budget to expire,
            // killing combat with a reset during RL training.  Forcing high
            // MaxFps + disabling low-processor mode keeps resolution realtime
            // regardless of focus.
            try
            {
                Godot.Engine.MaxFps = 120;
                Godot.OS.LowProcessorUsageMode = false;
                Log.Info($"[{BridgeRuntime.ModId}] Forced Engine.MaxFps=120 and LowProcessorUsageMode=false for unfocused training.");
            }
            catch (Exception ex)
            {
                Log.Warn($"[{BridgeRuntime.ModId}] Failed to force high-fps unfocused mode: {ex.GetBaseException().Message}");
            }

            Log.Info($"[{BridgeRuntime.ModId}] Attached bridge coordinator to NGame.");
            BridgeDebugTrace.Write("coordinator attached to NGame");
        }
    }

    public static void Detach()
    {
        List<PumpTickWaiter>? waitersToCancel = null;
        var detachException = new InvalidOperationException(
            "Bridge coordinator detached before queued main-thread work could execute.");

        lock (Sync)
        {
            _isAttached = false;
            while (Queue.TryDequeue(out var workItem))
            {
                Interlocked.Decrement(ref _queuedWorkItems);
                workItem.TryFail(detachException);
            }

            if (PumpTickWaiters.Count > 0)
            {
                waitersToCancel = new List<PumpTickWaiter>(PumpTickWaiters);
                PumpTickWaiters.Clear();
            }

            BridgeDebugTrace.Write("coordinator detached");
        }

        BridgeGameApi.ResetFrontierState();

        if (waitersToCancel is null)
        {
            return;
        }

        foreach (var waiter in waitersToCancel)
        {
            waiter.CancellationRegistration.Dispose();
            waiter.CompletionSource.TrySetException(detachException);
        }
    }

    public static Task<T> RunOnMainThreadAsync<T>(Func<T> action, CancellationToken cancellationToken = default)
    {
        if (NGame.Instance is not null &&
            GodotObject.IsInstanceValid(NGame.Instance) &&
            NGame.IsMainThread())
        {
            return Task.FromResult(action());
        }

        var tcs = new TaskCompletionSource<T>(TaskCreationOptions.RunContinuationsAsynchronously);
        lock (Sync)
        {
            if (!_isAttached)
            {
                throw new InvalidOperationException("Bridge coordinator is not attached yet.");
            }

            var queued = Interlocked.Increment(ref _queuedWorkItems);
            if (queued > MaxQueuedMainThreadWorkItems)
            {
                Interlocked.Decrement(ref _queuedWorkItems);
                throw new InvalidOperationException(
                    $"Bridge main-thread queue is full ({MaxQueuedMainThreadWorkItems} work items). Try again later.");
            }

            // Enqueue under the same lock used by Detach. This guarantees that
            // detach either rejects this work or dequeues and completes it with
            // an exception; a Task can no longer be orphaned between the checks.
            Queue.Enqueue(new MainThreadWorkItem<T>(action, tcs, cancellationToken));
        }

        BridgeDebugTrace.Write("coordinator enqueue");
        return tcs.Task;
    }

    public static Task WaitForPumpTicksAsync(int tickCount, CancellationToken cancellationToken = default)
    {
        if (tickCount <= 0)
        {
            return Task.CompletedTask;
        }

        PumpTickWaiter waiter;
        lock (Sync)
        {
            if (!_isAttached)
            {
                throw new InvalidOperationException("Bridge coordinator is not attached yet.");
            }

            waiter = new PumpTickWaiter
            {
                TargetTick = _pumpTick + tickCount,
                CompletionSource = new TaskCompletionSource<bool>(TaskCreationOptions.RunContinuationsAsynchronously)
            };

            if (cancellationToken.CanBeCanceled)
            {
                waiter.CancellationRegistration = cancellationToken.Register(() =>
                {
                    lock (Sync)
                    {
                        PumpTickWaiters.Remove(waiter);
                    }

                    waiter.Cancel(cancellationToken);
                });
            }

            PumpTickWaiters.Add(waiter);
        }

        return waiter.CompletionSource.Task;
    }

    public static void Pump()
    {
        if (Interlocked.Exchange(ref _pumpActive, 1) != 0)
        {
            return;
        }

        var processedAny = false;
        List<PumpTickWaiter>? readyWaiters = null;
        try
        {
            var pumpStartedAtMs = System.Environment.TickCount64;
            var processedCount = 0;
            while (processedCount < MaxWorkItemsPerPump &&
                   System.Environment.TickCount64 - pumpStartedAtMs < MaxPumpWorkMilliseconds &&
                   Queue.TryDequeue(out var workItem))
            {
                Interlocked.Decrement(ref _queuedWorkItems);
                processedAny = true;
                processedCount++;
                workItem.TryExecute();
            }

            lock (Sync)
            {
                _pumpTick++;
                _lastPumpAtMs = System.Environment.TickCount64;

                if (PumpTickWaiters.Count > 0)
                {
                    readyWaiters = PumpTickWaiters
                        .Where(waiter => waiter.TargetTick <= _pumpTick)
                        .ToList();

                    foreach (var waiter in readyWaiters)
                    {
                        PumpTickWaiters.Remove(waiter);
                    }
                }
            }

            if (processedAny)
            {
                BridgeDebugTrace.Write("coordinator processed queued work");
            }

            BridgeGameApi.NotifyFrontierPumpTick();

            if (readyWaiters is not null)
            {
                foreach (var waiter in readyWaiters)
                {
                    waiter.Complete();
                }
            }
        }
        finally
        {
            Volatile.Write(ref _pumpActive, 0);
        }
    }

    [HarmonyPatch(typeof(NGame), nameof(NGame._Ready))]
    private static class NGameReadyPatch
    {
        [HarmonyPostfix]
        private static void Postfix(NGame __instance)
        {
            EnsureAttached(__instance);
        }
    }

    [HarmonyPatch(typeof(NGame), nameof(NGame._ExitTree))]
    private static class NGameExitTreePatch
    {
        [HarmonyPrefix]
        private static void Prefix()
        {
            Detach();
        }
    }

    [HarmonyPatch(typeof(NControllerManager), nameof(NControllerManager._Process))]
    private static class NControllerManagerProcessPatch
    {
        [HarmonyPostfix]
        private static void Postfix()
        {
            Pump();
        }
    }

    [HarmonyPatch(typeof(NRun), nameof(NRun._Process))]
    private static class NRunProcessPatch
    {
        [HarmonyPostfix]
        private static void Postfix()
        {
            Pump();
        }
    }
}
