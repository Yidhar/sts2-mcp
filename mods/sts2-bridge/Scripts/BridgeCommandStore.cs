namespace Sts2McpBridge.Scripts;

internal enum BridgeCommandStoreDisposition
{
    Created,
    Existing,
    FingerprintConflict,
    CapacityExceeded
}

internal sealed class BridgeCommandStoreEntry
{
    private readonly object _sync = new();
    private readonly TaskCompletionSource<BridgeCommandResultV2> _completion =
        new(TaskCreationOptions.RunContinuationsAsynchronously);
    private BridgeCommandResultV2 _snapshot;

    public BridgeCommandStoreEntry(
        string requestId,
        string fingerprint,
        string capability,
        string operationKind,
        DateTimeOffset acceptedAtUtc,
        DateTimeOffset expiresAtUtc)
    {
        RequestId = requestId;
        Fingerprint = fingerprint;
        Capability = capability;
        OperationKind = operationKind;
        AcceptedAtUtc = acceptedAtUtc;
        ExpiresAtUtc = expiresAtUtc;
        _snapshot = new BridgeCommandResultV2
        {
            Ok = true,
            RequestId = requestId,
            Status = "accepted",
            ReplayedResult = false,
            AcceptedAtUtc = acceptedAtUtc
        };
    }

    public string RequestId { get; }

    public string Fingerprint { get; }

    public string Capability { get; }

    public string OperationKind { get; }

    public DateTimeOffset AcceptedAtUtc { get; }

    public DateTimeOffset ExpiresAtUtc { get; private set; }

    public bool IsCompleted => _completion.Task.IsCompleted;

    public BridgeCommandResultV2 Snapshot
    {
        get
        {
            lock (_sync)
            {
                return _snapshot;
            }
        }
    }

    public void MarkExecuting(DateTimeOffset startedAtUtc)
    {
        lock (_sync)
        {
            if (_completion.Task.IsCompleted)
            {
                return;
            }

            _snapshot = new BridgeCommandResultV2
            {
                Ok = true,
                RequestId = RequestId,
                Status = "executing",
                ReplayedResult = false,
                AcceptedAtUtc = AcceptedAtUtc,
                StartedAtUtc = startedAtUtc
            };
        }
    }

    public void Complete(BridgeCommandResultV2 result, DateTimeOffset expiresAtUtc)
    {
        lock (_sync)
        {
            if (_completion.Task.IsCompleted)
            {
                return;
            }

            _snapshot = result;
            ExpiresAtUtc = expiresAtUtc;
            _completion.TrySetResult(result);
        }
    }

    public Task<BridgeCommandResultV2> WaitAsync(CancellationToken cancellationToken) =>
        _completion.Task.WaitAsync(cancellationToken);
}

internal sealed class BoundedCommandResultStore
{
    private readonly object _sync = new();
    private readonly Dictionary<string, BridgeCommandStoreEntry> _entries = new(StringComparer.Ordinal);
    private readonly int _capacity;
    private readonly TimeSpan _ttl;
    private readonly Func<DateTimeOffset> _clock;

    public BoundedCommandResultStore(
        int capacity,
        TimeSpan ttl,
        Func<DateTimeOffset>? clock = null)
    {
        if (capacity <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(capacity));
        }

        if (ttl <= TimeSpan.Zero)
        {
            throw new ArgumentOutOfRangeException(nameof(ttl));
        }

        _capacity = capacity;
        _ttl = ttl;
        _clock = clock ?? (() => DateTimeOffset.UtcNow);
    }

    public int Capacity => _capacity;

    public TimeSpan TimeToLive => _ttl;

    public BridgeCommandStoreDisposition GetOrCreate(
        string requestId,
        string fingerprint,
        string capability,
        string operationKind,
        out BridgeCommandStoreEntry? entry)
    {
        lock (_sync)
        {
            var now = _clock();
            PurgeExpiredCompletedLocked(now);

            if (_entries.TryGetValue(requestId, out entry))
            {
                return string.Equals(entry.Fingerprint, fingerprint, StringComparison.Ordinal) &&
                       string.Equals(entry.Capability, capability, StringComparison.Ordinal) &&
                       string.Equals(entry.OperationKind, operationKind, StringComparison.Ordinal)
                    ? BridgeCommandStoreDisposition.Existing
                    : BridgeCommandStoreDisposition.FingerprintConflict;
            }

            // Never evict an unexpired completed identity: doing so would permit
            // the same request_id to execute twice inside the advertised TTL.
            if (_entries.Count >= _capacity)
            {
                entry = null;
                return BridgeCommandStoreDisposition.CapacityExceeded;
            }

            entry = new BridgeCommandStoreEntry(
                requestId,
                fingerprint,
                capability,
                operationKind,
                now,
                now + _ttl);
            _entries.Add(requestId, entry);
            return BridgeCommandStoreDisposition.Created;
        }
    }

    public bool TryGet(string requestId, out BridgeCommandStoreEntry? entry)
    {
        lock (_sync)
        {
            PurgeExpiredCompletedLocked(_clock());
            return _entries.TryGetValue(requestId, out entry);
        }
    }

    public void Complete(BridgeCommandStoreEntry entry, BridgeCommandResultV2 result)
    {
        var now = _clock();
        entry.Complete(result, now + _ttl);
    }

    public object GetDiagnosticsSnapshot()
    {
        lock (_sync)
        {
            PurgeExpiredCompletedLocked(_clock());
            return new
            {
                entries = _entries.Count,
                in_flight = _entries.Values.Count(static entry => !entry.IsCompleted),
                by_capability = _entries.Values
                    .GroupBy(static entry => entry.Capability, StringComparer.Ordinal)
                    .ToDictionary(static group => group.Key, static group => group.Count(), StringComparer.Ordinal),
                capacity = _capacity,
                ttl_seconds = (int)_ttl.TotalSeconds,
                eviction_policy = "expired-completed-only"
            };
        }
    }

    private void PurgeExpiredCompletedLocked(DateTimeOffset now)
    {
        foreach (var key in _entries
                     .Where(pair => pair.Value.IsCompleted && pair.Value.ExpiresAtUtc <= now)
                     .Select(static pair => pair.Key)
                     .ToArray())
        {
            _entries.Remove(key);
        }
    }
}
