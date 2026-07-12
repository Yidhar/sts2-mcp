namespace Sts2McpBridge.Scripts;

internal sealed class BridgeMutationGateBusyException : Exception
{
    public BridgeMutationGateBusyException(int capacity)
        : base($"Bridge mutation queue is full ({capacity} waiting operations).")
    {
    }
}

internal static class BridgeMutationGate
{
    private const int MaxQueuedMutations = 64;
    private static readonly SemaphoreSlim Semaphore = new(1, 1);
    private static readonly object Sync = new();
    private static int _waiting;
    private static string? _activeOperation;
    private static DateTimeOffset? _activeSinceUtc;

    public static async Task<T> RunAsync<T>(
        string operationName,
        Func<CancellationToken, Task<T>> operation,
        CancellationToken cancellationToken)
    {
        await using var lease = await AcquireAsync(operationName, cancellationToken);
        return await operation(cancellationToken);
    }

    public static async ValueTask<IAsyncDisposable> AcquireAsync(
        string operationName,
        CancellationToken cancellationToken)
    {
        var waiting = Interlocked.Increment(ref _waiting);
        if (waiting > MaxQueuedMutations)
        {
            Interlocked.Decrement(ref _waiting);
            throw new BridgeMutationGateBusyException(MaxQueuedMutations);
        }

        try
        {
            await Semaphore.WaitAsync(cancellationToken);
        }
        finally
        {
            Interlocked.Decrement(ref _waiting);
        }

        lock (Sync)
        {
            _activeOperation = operationName;
            _activeSinceUtc = DateTimeOffset.UtcNow;
        }

        return new Lease();
    }

    public static string? ActiveOperation
    {
        get
        {
            lock (Sync)
            {
                return _activeOperation;
            }
        }
    }

    public static object GetDiagnosticsSnapshot()
    {
        lock (Sync)
        {
            return new
            {
                active_operation = _activeOperation,
                active_since_utc = _activeSinceUtc,
                waiting_mutations = Math.Max(0, Volatile.Read(ref _waiting)),
                max_waiting_mutations = MaxQueuedMutations
            };
        }
    }

    private sealed class Lease : IAsyncDisposable
    {
        private int _disposed;

        public ValueTask DisposeAsync()
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0)
            {
                return ValueTask.CompletedTask;
            }

            lock (Sync)
            {
                _activeOperation = null;
                _activeSinceUtc = null;
            }

            Semaphore.Release();
            return ValueTask.CompletedTask;
        }
    }
}
