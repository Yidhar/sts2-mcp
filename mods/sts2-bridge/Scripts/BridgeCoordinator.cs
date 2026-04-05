using System.Collections.Concurrent;
using Godot;
using HarmonyLib;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Nodes;
using MegaCrit.Sts2.Core.Nodes.CommonUi;

namespace Sts2McpBridge.Scripts;

internal static class BridgeCoordinator
{
    private static readonly object Sync = new();
    private static readonly ConcurrentQueue<IMainThreadWorkItem> Queue = new();
    private static readonly List<PumpTickWaiter> PumpTickWaiters = new();
    private static bool _isAttached;
    private static long _pumpTick;
    private static long _lastPumpAtMs = System.Environment.TickCount64;

    private interface IMainThreadWorkItem
    {
        void TryExecute();
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
                queued_main_thread_work = Queue.Count,
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

            Log.Info($"[{BridgeRuntime.ModId}] Attached bridge coordinator to NGame.");
            BridgeDebugTrace.Write("coordinator attached to NGame");
        }
    }

    public static void Detach()
    {
        List<PumpTickWaiter>? waitersToCancel = null;

        lock (Sync)
        {
            while (Queue.TryDequeue(out _))
            {
            }

            if (PumpTickWaiters.Count > 0)
            {
                waitersToCancel = new List<PumpTickWaiter>(PumpTickWaiters);
                PumpTickWaiters.Clear();
            }

            BridgeDebugTrace.Write("coordinator detached");
            _isAttached = false;
        }

        BridgeGameApi.ResetFrontierState();

        if (waitersToCancel is null)
        {
            return;
        }

        foreach (var waiter in waitersToCancel)
        {
            waiter.CancellationRegistration.Dispose();
            waiter.CompletionSource.TrySetException(
                new InvalidOperationException("Bridge coordinator detached before the requested pump ticks completed."));
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

        lock (Sync)
        {
            if (!_isAttached)
            {
                throw new InvalidOperationException("Bridge coordinator is not attached yet.");
            }
        }

        var tcs = new TaskCompletionSource<T>(TaskCreationOptions.RunContinuationsAsynchronously);
        BridgeDebugTrace.Write("coordinator enqueue");

        Queue.Enqueue(new MainThreadWorkItem<T>(action, tcs, cancellationToken));

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
        var processedAny = false;
        List<PumpTickWaiter>? readyWaiters = null;

        while (Queue.TryDequeue(out var workItem))
        {
            processedAny = true;
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

                if (readyWaiters.Count > 0)
                {
                    foreach (var waiter in readyWaiters)
                    {
                        PumpTickWaiters.Remove(waiter);
                    }
                }
            }
        }

        if (processedAny)
        {
            BridgeDebugTrace.Write("coordinator processed queued work");
        }

        BridgeGameApi.NotifyFrontierPumpTick();

        if (readyWaiters is null)
        {
            return;
        }

        foreach (var waiter in readyWaiters)
        {
            waiter.Complete();
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
