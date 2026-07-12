using System.Globalization;
using System.Net;
using System.Text;
using System.Text.Json;
using System.Threading.Channels;

namespace Sts2McpBridge.Scripts;

/// <summary>Revisioned frontier storage, waiters, and bounded SSE presentation.</summary>
internal static partial class BridgeGameApi
{
    private sealed class ObservedFrontier
    {
        public required long Sequence { get; init; }

        public required string FrontierHash { get; init; }

        public required BridgeSnapshot Snapshot { get; init; }

        private readonly object _statePayloadSync = new();
        private object? _statePayload;

        public object GetOrCreateStatePayload()
        {
            if (_statePayload is not null)
            {
                return _statePayload;
            }

            lock (_statePayloadSync)
            {
                _statePayload ??= BuildFrontierStatePayload(Snapshot, Sequence);
                return _statePayload;
            }
        }
    }

    private sealed class BridgeFrontierCandidate
    {
        public required BridgeWorldContext Context { get; init; }

        public required IReadOnlyList<BridgeResolvedAction> Actions { get; init; }

        public required string FrontierHash { get; init; }
    }

    private sealed class FrontierWaiter
    {
        public required Guid Id { get; init; }

        public required long AfterSequence { get; init; }

        public required TaskCompletionSource<ObservedFrontier> Completion { get; init; }
    }

    private static class BridgeFrontierStore
    {
        private const int StreamStableTickTarget = 1;
        private const int StreamHeartbeatIntervalMs = 15000;
        private const int WaiterSampleIntervalMs = 40;
        private const int SubscriberSampleIntervalMs = 120;

        private static readonly object Sync = new();
        private static readonly Dictionary<Guid, Channel<ObservedFrontier>> Subscribers = new();
        private static readonly Dictionary<Guid, FrontierWaiter> Waiters = new();
        private static ObservedFrontier? _current;
        private static string? _candidateFrontierHash;
        private static BridgeFrontierCandidate? _candidate;
        private static int _candidateStableTicks;
        private static long _nextSequence = 1;
        private static long _nextPumpSampleAtMs;

        public static void Reset()
        {
            List<Channel<ObservedFrontier>> subscriberChannels;
            List<TaskCompletionSource<ObservedFrontier>> waiterCompletions;
            lock (Sync)
            {
                subscriberChannels = Subscribers.Values.ToList();
                waiterCompletions = Waiters.Values
                    .Select(static waiter => waiter.Completion)
                    .ToList();
                Subscribers.Clear();
                Waiters.Clear();
                _current = null;
                _candidateFrontierHash = null;
                _candidate = null;
                _candidateStableTicks = 0;
                _nextSequence = 1;
                _nextPumpSampleAtMs = 0;
            }

            foreach (var channel in subscriberChannels)
            {
                channel.Writer.TryComplete();
            }

            foreach (var completion in waiterCompletions)
            {
                completion.TrySetCanceled();
            }
        }

        public static ObservedFrontier PublishSnapshot(BridgeSnapshot snapshot)
        {
            ObservedFrontier frontier;
            List<Channel<ObservedFrontier>> subscriberChannels = [];
            List<TaskCompletionSource<ObservedFrontier>> waiterCompletions = [];
            bool changed;

            lock (Sync)
            {
                var previousFrontierHash = _current?.FrontierHash;
                frontier = GetOrCreateFrontierLocked(snapshot);
                _candidateFrontierHash = frontier.FrontierHash;
                _candidate = null;
                _candidateStableTicks = StreamStableTickTarget;
                changed = !string.Equals(previousFrontierHash, frontier.FrontierHash, StringComparison.Ordinal);

                if (changed)
                {
                    subscriberChannels = Subscribers.Values.ToList();
                    waiterCompletions = CollectReadyWaitersLocked(frontier);
                }
            }

            if (changed)
            {
                BroadcastFrontier(subscriberChannels, frontier);
                CompleteWaiters(waiterCompletions, frontier);
            }
            return frontier;
        }

        public static async Task<ObservedFrontier?> WaitForNextFrontierAsync(
            long afterSequence,
            int timeoutMs,
            CancellationToken cancellationToken)
        {
            if (timeoutMs <= 0)
            {
                lock (Sync)
                {
                    if (_current is not null && _current.Sequence > afterSequence)
                    {
                        return _current;
                    }
                }

                return null;
            }

            FrontierWaiter waiter;
            lock (Sync)
            {
                if (_current is not null && _current.Sequence > afterSequence)
                {
                    return _current;
                }

                waiter = new FrontierWaiter
                {
                    Id = Guid.NewGuid(),
                    AfterSequence = afterSequence,
                    Completion = new TaskCompletionSource<ObservedFrontier>(
                        TaskCreationOptions.RunContinuationsAsynchronously)
                };
                Waiters[waiter.Id] = waiter;
            }

            try
            {
                var completedTask = await Task.WhenAny(
                    waiter.Completion.Task,
                    Task.Delay(timeoutMs, cancellationToken));
                if (completedTask == waiter.Completion.Task)
                {
                    return await waiter.Completion.Task;
                }

                cancellationToken.ThrowIfCancellationRequested();
                return null;
            }
            finally
            {
                lock (Sync)
                {
                    Waiters.Remove(waiter.Id);
                }
            }
        }

        public static void OnPumpTick()
        {
            bool hasSubscribers;
            bool hasWaiters;
            lock (Sync)
            {
                hasSubscribers = Subscribers.Count > 0;
                hasWaiters = Waiters.Count > 0;

                if (!hasSubscribers && !hasWaiters)
                {
                    return;
                }

                var nowMs = System.Environment.TickCount64;
                if (nowMs < _nextPumpSampleAtMs)
                {
                    return;
                }

                var intervalMs = hasWaiters ? WaiterSampleIntervalMs : SubscriberSampleIntervalMs;
                _nextPumpSampleAtMs = nowMs + intervalMs;
            }

            try
            {
                var candidate = CaptureFrontierCandidate();
                ObservedFrontier? frontierToPublish = null;
                List<Channel<ObservedFrontier>> subscriberChannels = [];
                List<TaskCompletionSource<ObservedFrontier>> waiterCompletions = [];
                BridgeFrontierCandidate? candidateToHydrate = null;

                lock (Sync)
                {
                    if (_current is not null &&
                        string.Equals(_current.FrontierHash, candidate.FrontierHash, StringComparison.Ordinal))
                    {
                        _candidateFrontierHash = candidate.FrontierHash;
                        _candidate = null;
                        _candidateStableTicks = StreamStableTickTarget;
                        return;
                    }

                    if (string.Equals(_candidateFrontierHash, candidate.FrontierHash, StringComparison.Ordinal))
                    {
                        _candidateStableTicks++;
                        _candidate = candidate;
                    }
                    else
                    {
                        _candidateFrontierHash = candidate.FrontierHash;
                        _candidate = candidate;
                        _candidateStableTicks = 0;
                    }

                    if (_candidateStableTicks < StreamStableTickTarget || _candidate is null)
                    {
                        return;
                    }

                    candidateToHydrate = _candidate;
                }

                if (candidateToHydrate is null)
                {
                    return;
                }

                var snapshot = HydrateSnapshot(candidateToHydrate);
                Volatile.Write(ref _lastSnapshotAtTickMs, System.Environment.TickCount64);

                lock (Sync)
                {
                    if (!string.Equals(_candidateFrontierHash, snapshot.FrontierHash, StringComparison.Ordinal))
                    {
                        return;
                    }

                    if (_current is not null &&
                        string.Equals(_current.FrontierHash, snapshot.FrontierHash, StringComparison.Ordinal))
                    {
                        _candidate = null;
                        _candidateStableTicks = StreamStableTickTarget;
                        return;
                    }

                    frontierToPublish = GetOrCreateFrontierLocked(snapshot);
                    _candidate = null;
                    _candidateStableTicks = StreamStableTickTarget;
                    subscriberChannels = Subscribers.Values.ToList();
                    waiterCompletions = CollectReadyWaitersLocked(frontierToPublish);
                }

                if (frontierToPublish is not null)
                {
                    BroadcastFrontier(subscriberChannels, frontierToPublish);
                    CompleteWaiters(waiterCompletions, frontierToPublish);
                }
            }
            catch (Exception ex)
            {
                BridgeDebugTrace.Write($"frontier_pump_error: {ex}");
            }
        }

        public static async Task StreamEventsAsync(
            HttpListenerResponse response,
            bool playerVisibleV2,
            CancellationToken cancellationToken)
        {
            response.StatusCode = (int)HttpStatusCode.OK;
            response.ContentType = "text/event-stream; charset=utf-8";
            response.SendChunked = true;
            response.KeepAlive = true;
            response.Headers["Cache-Control"] = "no-cache";
            response.Headers["X-Accel-Buffering"] = "no";

            var subscriberId = Guid.NewGuid();
            var channel = Channel.CreateBounded<ObservedFrontier>(new BoundedChannelOptions(1)
            {
                SingleReader = true,
                SingleWriter = false,
                AllowSynchronousContinuations = false,
                FullMode = BoundedChannelFullMode.DropOldest
            });

            lock (Sync)
            {
                Subscribers[subscriberId] = channel;
            }

            try
            {
                var initialFrontier = await ObserveFrontierAsync(cancellationToken);
                await WriteFrontierEventAsync(response, initialFrontier, playerVisibleV2, cancellationToken);
                while (channel.Reader.TryRead(out _))
                {
                }

                while (!cancellationToken.IsCancellationRequested)
                {
                    var waitToReadTask = channel.Reader.WaitToReadAsync(cancellationToken).AsTask();
                    var completedTask = await Task.WhenAny(
                        waitToReadTask,
                        Task.Delay(StreamHeartbeatIntervalMs, cancellationToken));

                    if (completedTask != waitToReadTask)
                    {
                        await WriteSseCommentAsync(response, "heartbeat", cancellationToken);
                        continue;
                    }

                    if (!await waitToReadTask)
                    {
                        break;
                    }

                    ObservedFrontier? latestFrontier = null;
                    while (channel.Reader.TryRead(out var frontier))
                    {
                        latestFrontier = frontier;
                    }

                    if (latestFrontier is not null)
                    {
                        await WriteFrontierEventAsync(response, latestFrontier, playerVisibleV2, cancellationToken);
                    }
                }
            }
            catch (OperationCanceledException ex)
            {
                BridgeDebugTrace.Write($"frontier_stream_cancelled subscriber={subscriberId}: {ex.Message}");
            }
            catch (HttpListenerException ex)
            {
                BridgeDebugTrace.Write(
                    $"frontier_stream_listener_closed subscriber={subscriberId} error_code={ex.ErrorCode}: {ex.Message}");
            }
            catch (IOException ex)
            {
                BridgeDebugTrace.Write($"frontier_stream_io_closed subscriber={subscriberId}: {ex.Message}");
            }
            finally
            {
                lock (Sync)
                {
                    Subscribers.Remove(subscriberId);
                }

                channel.Writer.TryComplete();

                try
                {
                    response.OutputStream.Close();
                }
                catch (Exception ex)
                {
                    BridgeDebugTrace.Write(
                        $"frontier_stream_response_close_failed subscriber={subscriberId}: {ex.GetBaseException().Message}");
                }
            }
        }

        private static ObservedFrontier GetOrCreateFrontierLocked(BridgeSnapshot snapshot)
        {
            var frontierHash = snapshot.FrontierHash;
            if (_current is not null &&
                string.Equals(_current.FrontierHash, frontierHash, StringComparison.Ordinal))
            {
                return _current;
            }

            var sequence = _nextSequence++;
            _current = new ObservedFrontier
            {
                Sequence = sequence,
                FrontierHash = frontierHash,
                Snapshot = snapshot
            };
            return _current;
        }

        private static void BroadcastFrontier(
            IEnumerable<Channel<ObservedFrontier>> subscriberChannels,
            ObservedFrontier frontier)
        {
            foreach (var channel in subscriberChannels)
            {
                channel.Writer.TryWrite(frontier);
            }
        }

        private static List<TaskCompletionSource<ObservedFrontier>> CollectReadyWaitersLocked(ObservedFrontier frontier)
        {
            var completions = new List<TaskCompletionSource<ObservedFrontier>>();
            foreach (var waiter in Waiters.Values.ToArray())
            {
                if (frontier.Sequence <= waiter.AfterSequence)
                {
                    continue;
                }

                completions.Add(waiter.Completion);
                Waiters.Remove(waiter.Id);
            }

            return completions;
        }

        private static void CompleteWaiters(
            IEnumerable<TaskCompletionSource<ObservedFrontier>> waiterCompletions,
            ObservedFrontier frontier)
        {
            foreach (var completion in waiterCompletions)
            {
                completion.TrySetResult(frontier);
            }
        }

        private static async Task WriteFrontierEventAsync(
            HttpListenerResponse response,
            ObservedFrontier frontier,
            bool playerVisibleV2,
            CancellationToken cancellationToken)
        {
            var payload = JsonSerializer.Serialize(
                playerVisibleV2
                    ? BuildFrontierEventV2Payload(frontier)
                    : BuildLegacyFrontierEventPayload(frontier),
                HashJsonOptions);
            await WriteSseEventAsync(
                response,
                "frontier",
                frontier.Sequence.ToString(CultureInfo.InvariantCulture),
                payload,
                cancellationToken);
        }

        private static object BuildFrontierEventV2Payload(ObservedFrontier frontier) =>
            new
            {
                ok = true,
                event_type = "frontier",
                api_version = BridgeProtocolV2.ApiVersion,
                schema_version = BridgeProtocolV2.SchemaVersion,
                session_id = BridgeRuntime.SessionId,
                state_version = frontier.Sequence,
                captured_at_utc = DateTimeOffset.UtcNow,
                visibility = "player",
                screen = frontier.Snapshot.Fields.Screen
            };

        private static object BuildLegacyFrontierEventPayload(ObservedFrontier frontier) =>
            new
            {
                ok = true,
                event_type = "frontier",
                api_version = (string?)null,
                state = frontier.GetOrCreateStatePayload()
            };

        private static async Task WriteSseEventAsync(
            HttpListenerResponse response,
            string eventName,
            string eventId,
            string payload,
            CancellationToken cancellationToken)
        {
            var builder = new StringBuilder();
            builder.Append("id: ").Append(eventId).Append('\n');
            builder.Append("event: ").Append(eventName).Append('\n');
            foreach (var line in payload.ReplaceLineEndings("\n").Split('\n'))
            {
                builder.Append("data: ").Append(line).Append('\n');
            }

            builder.Append('\n');
            var bytes = Encoding.UTF8.GetBytes(builder.ToString());
            await response.OutputStream.WriteAsync(bytes, cancellationToken);
            await response.OutputStream.FlushAsync(cancellationToken);
        }

        private static async Task WriteSseCommentAsync(
            HttpListenerResponse response,
            string comment,
            CancellationToken cancellationToken)
        {
            var bytes = Encoding.UTF8.GetBytes($": {comment}\n\n");
            await response.OutputStream.WriteAsync(bytes, cancellationToken);
            await response.OutputStream.FlushAsync(cancellationToken);
        }
    }
}
