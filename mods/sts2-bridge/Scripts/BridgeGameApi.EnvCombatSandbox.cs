using System.Globalization;
using System.Net;
using System.Reflection;
using System.Text.Json.Serialization;
using System.ComponentModel;
using System.Diagnostics;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Entities.Multiplayer;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Map;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Models.Characters;
using MegaCrit.Sts2.Core.Multiplayer.Game;
using MegaCrit.Sts2.Core.Nodes;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Saves;

namespace Sts2McpBridge.Scripts;

internal sealed class BridgeEnvCombatDeckModifierRequest
{
    [JsonPropertyName("id")]
    public string? Id { get; set; }

    [JsonPropertyName("amount")]
    public int? Amount { get; set; }

    [JsonPropertyName("status")]
    public string? Status { get; set; }
}

internal sealed class BridgeEnvCombatDeckEntryRequest
{
    [JsonPropertyName("id")]
    public string? Id { get; set; }

    [JsonPropertyName("upgrade_level")]
    public int? UpgradeLevel { get; set; }

    /// <summary>
    /// Pre-combat enchantments to attach to this deck card (e.g., "fire", "extra_card").
    /// Only the `id` is required; `amount` defaults to 1.  Bridge resolves each via
    /// <c>ModelDb.GetById&lt;EnchantmentModel&gt;</c> then mounts it on the freshly built
    /// CardModel through reflection (game build varies the API surface).
    /// </summary>
    [JsonPropertyName("enchantments")]
    public BridgeEnvCombatDeckModifierRequest[]? Enchantments { get; set; }

    /// <summary>
    /// Pre-combat afflictions / curses attached to this deck card (e.g., "binding",
    /// "wound").  Same resolution path as <see cref="Enchantments"/>.
    /// </summary>
    [JsonPropertyName("afflictions")]
    public BridgeEnvCombatDeckModifierRequest[]? Afflictions { get; set; }
}

internal readonly struct BridgeEnvDeckEntryResolved
{
    public BridgeEnvDeckEntryResolved(
        string cardId,
        int upgradeLevel,
        IReadOnlyList<BridgeEnvCombatDeckModifierRequest> enchantments,
        IReadOnlyList<BridgeEnvCombatDeckModifierRequest> afflictions)
    {
        CardId = cardId;
        UpgradeLevel = upgradeLevel;
        Enchantments = enchantments;
        Afflictions = afflictions;
    }

    public string CardId { get; }
    public int UpgradeLevel { get; }
    public IReadOnlyList<BridgeEnvCombatDeckModifierRequest> Enchantments { get; }
    public IReadOnlyList<BridgeEnvCombatDeckModifierRequest> Afflictions { get; }
}

internal sealed class BridgeEnvCombatResetRequest
{
    [JsonPropertyName("character")]
    public string? Character { get; set; }

    [JsonPropertyName("encounter_id")]
    public string? EncounterId { get; set; }

    [JsonPropertyName("seed")]
    public int? Seed { get; set; }

    [JsonPropertyName("current_hp")]
    public int? CurrentHp { get; set; }

    [JsonPropertyName("max_hp")]
    public int? MaxHp { get; set; }

    [JsonPropertyName("max_energy")]
    public int? MaxEnergy { get; set; }

    [JsonPropertyName("deck")]
    public string[]? Deck { get; set; }

    [JsonPropertyName("deck_entries")]
    public BridgeEnvCombatDeckEntryRequest[]? DeckEntries { get; set; }

    [JsonPropertyName("relics")]
    public string[]? Relics { get; set; }

    [JsonPropertyName("potions")]
    public string[]? Potions { get; set; }

    [JsonPropertyName("gold")]
    public int? Gold { get; set; }

    [JsonPropertyName("timeout_ms")]
    public int? TimeoutMs { get; set; }
}

internal static partial class BridgeGameApi
{
    private const int DefaultCombatResetTimeoutMs = 30000;

    private static IReadOnlyList<BridgeEnvDeckEntryResolved> ResolveRequestedDeckEntries(
        BridgeEnvCombatResetRequest request)
    {
        static BridgeEnvCombatDeckModifierRequest[] FilterValid(BridgeEnvCombatDeckModifierRequest[]? src)
        {
            if (src is null || src.Length == 0)
            {
                return Array.Empty<BridgeEnvCombatDeckModifierRequest>();
            }
            return src.Where(static m => !string.IsNullOrWhiteSpace(m?.Id)).ToArray();
        }

        if (request.DeckEntries is { Length: > 0 })
        {
            return request.DeckEntries
                .Where(static entry => !string.IsNullOrWhiteSpace(entry?.Id))
                .Select(entry => new BridgeEnvDeckEntryResolved(
                    entry!.Id!.Trim(),
                    Math.Max(entry.UpgradeLevel ?? 0, 0),
                    FilterValid(entry.Enchantments),
                    FilterValid(entry.Afflictions)))
                .ToArray();
        }

        if (request.Deck is { Length: > 0 })
        {
            return request.Deck
                .Where(static cardId => !string.IsNullOrWhiteSpace(cardId))
                .Select(cardId => new BridgeEnvDeckEntryResolved(
                    cardId.Trim(),
                    0,
                    Array.Empty<BridgeEnvCombatDeckModifierRequest>(),
                    Array.Empty<BridgeEnvCombatDeckModifierRequest>()))
                .ToArray();
        }

        return Array.Empty<BridgeEnvDeckEntryResolved>();
    }

    private static CardModel? BuildMutableDeckCard(
        string cardId,
        int upgradeLevel,
        List<string> diagnostics)
    {
        if (TryModelDbGetById("CardModel", cardId, diagnostics) is not CardModel card)
        {
            diagnostics.Add($"Deck override skipped unknown card '{cardId}'");
            return null;
        }

        var mutableCard = card.ToMutable();
        var requestedUpgradeLevel = Math.Max(0, upgradeLevel);
        var maxUpgradeLevel = Math.Max(0, mutableCard.MaxUpgradeLevel);
        if (requestedUpgradeLevel > maxUpgradeLevel)
        {
            diagnostics.Add(
                $"Deck override clamped {cardId} upgrade {requestedUpgradeLevel} -> {maxUpgradeLevel} (MaxUpgradeLevel).");
        }

        for (var i = 0; i < Math.Min(requestedUpgradeLevel, maxUpgradeLevel); i++)
        {
            try
            {
                mutableCard.UpgradeInternal();
            }
            catch (Exception ex)
            {
                diagnostics.Add(
                    $"Deck override stopped upgrading {cardId} at level {mutableCard.CurrentUpgradeLevel}: {ex.GetType().Name}: {ex.Message}");
                break;
            }
        }

        return mutableCard;
    }

    /// <summary>
    /// Apply caller-supplied enchantments and afflictions onto a freshly built
    /// deck card.  Mirror of the READ path: same candidate property/method names
    /// since the game build varies the API surface.  Each modifier is resolved
    /// via <c>ModelDb.GetById&lt;EnchantmentModel|AfflictionModel&gt;</c>, then the
    /// reflective writer tries (in order):
    ///   1. <c>card.AddEnchantment(model[, amount])</c> / <c>card.AddAffliction(...)</c>
    ///   2. The same with <c>Add{Modifier|Buff|Debuff}</c> / <c>Enchant</c> / <c>Afflict</c>
    ///   3. Direct collection mutation: locate <c>Enchantments</c> / <c>Afflictions</c>
    ///      collection, construct an instance of its element type via
    ///      <c>Activator.CreateInstance(elementType, model[, amount])</c>, then
    ///      invoke <c>Add</c>.
    /// All failures are reported into <paramref name="diagnostics"/>; the deck
    /// card is left unchanged for that one modifier rather than aborting the
    /// whole reset.
    /// </summary>
    private static void ApplyDeckCardModifiers(
        CardModel mutableCard,
        BridgeEnvDeckEntryResolved entry,
        List<string> diagnostics)
    {
        if (mutableCard is null) return;
        foreach (var ench in entry.Enchantments)
        {
            TryApplyCardModifier(mutableCard, ench, "enchantment", entry.CardId, diagnostics);
        }
        foreach (var aff in entry.Afflictions)
        {
            TryApplyCardModifier(mutableCard, aff, "affliction", entry.CardId, diagnostics);
        }
    }

    private static void TryApplyCardModifier(
        CardModel card,
        BridgeEnvCombatDeckModifierRequest modifier,
        string kind,
        string cardId,
        List<string> diagnostics)
    {
        if (string.IsNullOrWhiteSpace(modifier?.Id))
        {
            return;
        }
        var modifierId = modifier.Id!.Trim();
        var amount = Math.Max(modifier.Amount ?? 1, 1);

        // Resolve the modifier model template via ModelDb<T>.GetById.
        var modelTypeName = kind == "enchantment" ? "EnchantmentModel" : "AfflictionModel";
        var modelTemplate = TryModelDbGetById(modelTypeName, modifierId, diagnostics);
        if (modelTemplate is null)
        {
            diagnostics.Add($"Deck modifier skipped: unknown {kind} '{modifierId}' for card '{cardId}'");
            return;
        }

        // Method name candidates on CardModel.
        var methodCandidates = kind == "enchantment"
            ? new[] { "AddEnchantment", "ApplyEnchantment", "Enchant", "AddModifier", "AddBuff" }
            : new[] { "AddAffliction", "ApplyAffliction", "Afflict", "AddDebuff", "AddStatus" };
        if (TryInvokeModifierMethod(card, methodCandidates, modelTemplate, amount, diagnostics, kind, modifierId, cardId))
        {
            return;
        }

        // Fallback: collection mutation.
        var collectionCandidates = kind == "enchantment"
            ? new[] { "Enchantments", "EnchantmentModels", "CardEnchantments", "Modifiers", "CardModifiers" }
            : new[] { "Afflictions", "AfflictionModels", "CardAfflictions", "Statuses", "StatusEffects" };
        foreach (var memberName in collectionCandidates)
        {
            if (TryAppendModifierViaCollection(card, memberName, modelTemplate, amount, diagnostics, kind, modifierId, cardId))
            {
                return;
            }
        }

        diagnostics.Add(
            $"Deck modifier failed: no working API to apply {kind} '{modifierId}' to card '{cardId}' " +
            "(no method match and no mutable collection found)");
    }

    private static bool TryInvokeModifierMethod(
        CardModel card,
        IReadOnlyList<string> methodNames,
        object modelTemplate,
        int amount,
        List<string> diagnostics,
        string kind,
        string modifierId,
        string cardId)
    {
        var cardType = card.GetType();
        const BindingFlags flags = BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance;
        foreach (var methodName in methodNames)
        {
            foreach (var method in cardType.GetMethods(flags))
            {
                if (!method.Name.Equals(methodName, StringComparison.Ordinal)) continue;
                var ps = method.GetParameters();
                object?[]? args = null;
                if (ps.Length == 1 && ps[0].ParameterType.IsInstanceOfType(modelTemplate))
                {
                    args = new object?[] { modelTemplate };
                }
                else if (ps.Length == 2
                         && ps[0].ParameterType.IsInstanceOfType(modelTemplate)
                         && ps[1].ParameterType == typeof(int))
                {
                    args = new object?[] { modelTemplate, amount };
                }
                else if (ps.Length == 2
                         && ps[1].ParameterType.IsInstanceOfType(modelTemplate)
                         && ps[0].ParameterType == typeof(int))
                {
                    args = new object?[] { amount, modelTemplate };
                }
                if (args is null) continue;
                try
                {
                    method.Invoke(card, args);
                    return true;
                }
                catch (Exception ex)
                {
                    diagnostics.Add(
                        $"Deck modifier {kind} '{modifierId}' on card '{cardId}': {method.Name} threw " +
                        $"{ex.GetBaseException().GetType().Name}: {ex.GetBaseException().Message}");
                    // try next candidate
                }
            }
        }
        return false;
    }

    private static bool TryAppendModifierViaCollection(
        CardModel card,
        string memberName,
        object modelTemplate,
        int amount,
        List<string> diagnostics,
        string kind,
        string modifierId,
        string cardId)
    {
        object? collection = null;
        try { collection = GetHiddenPropertyObjectValue(card, memberName); }
        catch (Exception ex)
        {
            BridgeDebugTrace.Write(
                $"combat_modifier_property_read_failed card={cardId} member={memberName}: {ex.GetBaseException().Message}");
        }
        if (collection is null)
        {
            try { collection = GetHiddenFieldValue(card, memberName); }
            catch (Exception ex)
            {
                BridgeDebugTrace.Write(
                    $"combat_modifier_field_read_failed card={cardId} member={memberName}: {ex.GetBaseException().Message}");
            }
        }
        if (collection is null) return false;

        var collectionType = collection.GetType();
        // Find element type — IList<T> / ICollection<T>.
        Type? elementType = null;
        foreach (var iface in collectionType.GetInterfaces())
        {
            if (iface.IsGenericType
                && (iface.GetGenericTypeDefinition() == typeof(ICollection<>)
                    || iface.GetGenericTypeDefinition() == typeof(IList<>)))
            {
                elementType = iface.GetGenericArguments()[0];
                break;
            }
        }
        if (elementType is null) return false;

        // Construct an instance of elementType from the model template.
        object? instance = null;
        try
        {
            instance = Activator.CreateInstance(elementType, modelTemplate, amount);
        }
        catch
        {
            try
            {
                instance = Activator.CreateInstance(elementType, modelTemplate);
            }
            catch
            {
                instance = null;
            }
        }
        if (instance is null && elementType.IsInstanceOfType(modelTemplate))
        {
            // Element type itself IS the model — push raw template.
            instance = modelTemplate;
        }
        if (instance is null) return false;

        var addMethod = collectionType.GetMethod("Add", new[] { elementType });
        if (addMethod is null) return false;
        try
        {
            addMethod.Invoke(collection, new[] { instance });
            return true;
        }
        catch (Exception ex)
        {
            diagnostics.Add(
                $"Deck modifier {kind} '{modifierId}' on card '{cardId}': collection {memberName}.Add threw " +
                $"{ex.GetBaseException().GetType().Name}: {ex.GetBaseException().Message}");
            return false;
        }
    }

    // -----------------------------------------------------------------------
    // POST /env/combat_reset
    // -----------------------------------------------------------------------

    public static async Task<object> CombatResetEnvResponseAsync(
        BridgeEnvCombatResetRequest? request,
        CancellationToken cancellationToken)
    {
        request ??= new BridgeEnvCombatResetRequest();
        var timeoutMs = NormalizeEnvTimeout(request.TimeoutMs, DefaultCombatResetTimeoutMs);
        await WaitForEnvDispatcherReadyAsync(timeoutMs, cancellationToken);

        var encounterId = request.EncounterId?.Trim();
        if (string.IsNullOrWhiteSpace(encounterId))
        {
            throw new BridgeRequestException(
                HttpStatusCode.BadRequest,
                "missing_encounter_id",
                "Request body must include a non-empty encounter_id. Use GET /env/combat_catalog to list available encounters.");
        }

        var diagnostics = new List<string>();
        var beforeSetup = await EnsureCombatSandboxRunReadyAsync(
            request,
            timeoutMs,
            diagnostics,
            cancellationToken);

        // Step 1: Resolve encounter and enter combat on the main thread.
        // This now runs only after a real single-player run scene exists.
        var setupResult = await RunOnMainThreadGuardedAsync(
            () => SetUpCombatSandbox(request, encounterId, diagnostics),
            "combat_sandbox.setup",
            timeoutMs,
            cancellationToken);

        if (!setupResult.Success)
        {
            throw new BridgeRequestException(
                HttpStatusCode.Conflict,
                setupResult.ErrorCode ?? "combat_sandbox_setup_failed",
                setupResult.ErrorMessage ?? "Failed to set up combat sandbox.",
                new { diagnostics, encounter_id = encounterId });
        }

        if (setupResult.PendingTask is not null)
        {
            try
            {
                await setupResult.PendingTask.WaitAsync(
                    TimeSpan.FromMilliseconds(Math.Min(timeoutMs, 5000)),
                    cancellationToken);
                diagnostics.Add("Combat room entry task completed");
            }
            catch (Exception ex)
            {
                diagnostics.Add($"Combat room entry task await failed: {ex.GetBaseException().Message}");
                diagnostics.Add($"Combat room entry exception detail: {ex}");
            }
        }

        // Step 2: Let Godot settle the scene
        await WaitForPumpTicksGuardedAsync(8, "combat_sandbox.settle_scene", timeoutMs, cancellationToken);

        // Step 3: Wait for a stable state that is actually different from the pre-reset baseline.
        var state = await WaitForCombatSandboxReadyStateAsync(
            beforeSetup.LogicHash,
            timeoutMs,
            diagnostics,
            cancellationToken);

        // Step 4: Retry the debug room entry once more on a later pump if the scene has not flipped
        // to an actionable combat-facing state yet. This keeps the fallback tight and avoids
        // rebuilding half-initialized runs.
        if (!IsCombatSandboxResetReady(state) &&
            !state.Done &&
            HasUsableCombatSandboxRunScene(state))
        {
            var deferredEnterStarted = await TryDeferredEnterCombatRoomAsync(
                encounterId,
                diagnostics,
                timeoutMs,
                cancellationToken);
            if (deferredEnterStarted)
            {
                await WaitForPumpTicksGuardedAsync(5, "combat_sandbox.deferred_enter_wait", timeoutMs, cancellationToken);
                state = await WaitForCombatSandboxReadyStateAsync(
                    state.LogicHash,
                    Math.Min(timeoutMs, 5000),
                    diagnostics,
                    cancellationToken);
            }
        }

        if (!IsCombatSandboxResetReady(state) && !state.Done)
        {
            state = await TrySalvageCombatSandboxSettlingResetAsync(
                state,
                timeoutMs,
                diagnostics,
                cancellationToken);
        }

        if (!IsCombatSandboxResetReady(state) && !state.Done)
        {
            throw new BridgeRequestException(
                HttpStatusCode.Conflict,
                "combat_sandbox_not_in_combat",
                $"Combat sandbox setup completed but the game is not in combat. Phase: {state.Phase}",
                new
                {
                    phase = state.Phase,
                    screen = state.Screen,
                    actionable = state.Actionable,
                    combat_in_progress = state.CombatInProgress,
                    room_type = state.RoomType,
                    room_model_id = state.RoomModelId,
                    diagnostics
                });
        }

        // Step 5: Create the sandbox episode
        var episode = new BridgeEnvEpisode
        {
            Id = Guid.NewGuid().ToString("N", CultureInfo.InvariantCulture),
            StepIndex = 0,
            Done = state.Done,
            RequestedCharacter = request.Character?.Trim(),
            DefensiveBuffs = false,
            EpisodeMode = "combat_sandbox",
            EncounterId = encounterId
        };

        lock (EnvEpisodeSync)
        {
            _activeEnvEpisode = episode;
        }

        state = await ApplyEnvEpisodeAdjustmentsAsync(episode, state, timeoutMs, cancellationToken);

        // Drain finalizers now, while the Godot scene is quiescent (new combat is set
        // up and awaiting the first /env/step). This keeps GodotObject finalizers from
        // firing mid-combat and racing the main thread's ObjectDB mutations — a known
        // source of native AV at HashMapElement+0x10 under long runs.
        DrainManagedFinalizersAfterReset(diagnostics);

        var executedActions = new List<object>
        {
            new { action = "combat_sandbox_setup", encounter_id = encounterId, diagnostics }
        };

        return BuildEnvResetPayload(episode, state, executedActions);
    }

    private static async Task<BridgeEnvSnapshot> EnsureCombatSandboxRunReadyAsync(
        BridgeEnvCombatResetRequest request,
        int timeoutMs,
        List<string> diagnostics,
        CancellationToken cancellationToken)
    {
        var priorState = await CaptureEnvSnapshotAsync(timeoutMs, cancellationToken, "combat_sandbox.prior_snapshot");
        diagnostics.Add(
            $"Combat sandbox bootstrap starting from phase={priorState.Phase}, screen={priorState.Screen}, run_active={priorState.RunActive}, combat_in_progress={priorState.CombatInProgress}, current_room={priorState.Context.RunState?.CurrentRoom?.GetType().Name ?? "null"}");

        if (HasUsableCombatSandboxRunScene(priorState) &&
            (priorState.CombatInProgress || string.Equals(priorState.Phase, "settling", StringComparison.Ordinal)))
        {
            diagnostics.Add(
                $"Waiting for existing run scene to settle before combat sandbox reset (phase={priorState.Phase}, combat_in_progress={priorState.CombatInProgress})");

            var settledState = await WaitForStableEnvStateAsync(
                priorState.LogicHash,
                Math.Min(timeoutMs, 5000),
                requireActionableOrDone: true,
                cancellationToken);

            diagnostics.Add(
                $"Combat sandbox pre-reset settle observed phase={settledState.Phase}, screen={settledState.Screen}, run_active={settledState.RunActive}, combat_in_progress={settledState.CombatInProgress}, current_room={settledState.Context.RunState?.CurrentRoom?.GetType().Name ?? "null"}");

            priorState = settledState;
        }

        priorState = await TryPrepareCombatSandboxFastResetStateAsync(
            priorState,
            timeoutMs,
            diagnostics,
            cancellationToken);

        if (CanReuseCombatSandboxRunScene(priorState))
        {
            diagnostics.Add(
                $"Reusing existing run scene for combat sandbox reset (phase={priorState.Phase}, screen={priorState.Screen})");
            return await CaptureEnvSnapshotAsync(timeoutMs, cancellationToken, "combat_sandbox.reuse_snapshot");
        }

        if (HasUsableCombatSandboxRunScene(priorState) && !priorState.CombatInProgress)
        {
            diagnostics.Add(
                $"Active run scene is not safe to reuse for combat sandbox reset (phase={priorState.Phase}, screen={priorState.Screen}); bootstrapping via env/reset instead");
        }

        Task? freshRunTask;
        try
        {
            freshRunTask = await RunOnMainThreadGuardedAsync(
                () => BeginFreshCombatSandboxRun(request, priorState, diagnostics),
                "combat_sandbox.begin_fresh_run",
                timeoutMs,
                cancellationToken);
        }
        catch (BridgeRequestException ex)
        {
            diagnostics.Add($"fresh combat sandbox bootstrap failed: {ex.ErrorCode}: {ex.Message}");
            throw;
        }

        if (freshRunTask is not null)
        {
            try
            {
                await freshRunTask.WaitAsync(
                    TimeSpan.FromMilliseconds(Math.Min(timeoutMs, 10000)),
                    cancellationToken);
                diagnostics.Add("Fresh combat sandbox run task completed");
            }
            catch (Exception ex)
            {
                diagnostics.Add($"Fresh combat sandbox run task await failed: {ex.GetBaseException().Message}");
            }
        }

        await WaitForPumpTicksGuardedAsync(3, "combat_sandbox.begin_fresh_run.post_pump", timeoutMs, cancellationToken);

        var state = await WaitForStableEnvStateAsync(
            null,
            timeoutMs,
            requireActionableOrDone: true,
            cancellationToken);

        diagnostics.Add(
            $"fresh combat sandbox bootstrap settled at phase={state.Phase}, screen={state.Screen}, run_active={state.RunActive}, run_node={(state.Context.RunNode is not null)}, current_room={(state.Context.RunState?.CurrentRoom is not null)}");

        if (!HasUsableCombatSandboxRunScene(state))
        {
            throw new BridgeRequestException(
                HttpStatusCode.Conflict,
                "combat_sandbox_run_scene_not_ready",
                "env/reset completed but did not produce a usable run scene for combat sandbox.",
                new
                {
                    phase = state.Phase,
                    screen = state.Screen,
                    run_active = state.RunActive,
                    has_run_node = state.Context.RunNode is not null,
                    has_current_room = state.Context.RunState?.CurrentRoom is not null,
                    diagnostics
                });
        }

        return state;
    }

    private static bool HasUsableCombatSandboxRunScene(BridgeEnvSnapshot state)
    {
        return state.RunActive &&
               !state.Done &&
               !IsStartupPhase(state.Phase) &&
               state.Context.RunNode is not null &&
               state.Context.RunState?.CurrentRoom is not null;
    }

    private static bool CanReuseCombatSandboxRunScene(BridgeEnvSnapshot state)
    {
        // Reusing an existing run scene is measurably less stable than bootstrapping
        // a fresh single-player run for sandbox resets. Hidden room/UI residue can
        // leave the game in a half-entered COMBAT/settling state and crash reset.
        // For training stability, disable scene reuse entirely.
        return false;
    }

    private static async Task<BridgeEnvSnapshot> TryPrepareCombatSandboxFastResetStateAsync(
        BridgeEnvSnapshot state,
        int timeoutMs,
        List<string> diagnostics,
        CancellationToken cancellationToken)
    {
        var snapshot = state;

        // Terminal / game-over states are much cheaper to recover by walking back to
        // startup first, then launching a fresh sandbox run from there.
        if (snapshot.Done && !IsStartupPhase(snapshot.Phase))
        {
            diagnostics.Add(
                $"Combat sandbox fast-reset: navigating terminal state back to startup (phase={snapshot.Phase}, screen={snapshot.Screen})");
            snapshot = await NavigateToMainMenuFromActiveRunAsync(
                snapshot,
                Math.Min(timeoutMs, 10000),
                cancellationToken);
            diagnostics.Add(
                $"Combat sandbox fast-reset terminal navigation settled at phase={snapshot.Phase}, screen={snapshot.Screen}, run_active={snapshot.RunActive}, done={snapshot.Done}");
        }

        // Victory/reward flows can usually be recycled by skipping rewards and
        // returning to the map instead of rebuilding a whole run scene.
        for (var attempt = 0; attempt < 3; attempt++)
        {
            if (CanReuseCombatSandboxRunScene(snapshot) ||
                snapshot.Done ||
                !snapshot.RunActive ||
                snapshot.CombatInProgress)
            {
                return snapshot;
            }

            var rewardPhase =
                string.Equals(snapshot.Phase, "reward", StringComparison.Ordinal) ||
                string.Equals(snapshot.Phase, "card_reward", StringComparison.Ordinal) ||
                string.Equals(snapshot.Screen, "REWARDS", StringComparison.OrdinalIgnoreCase);
            if (!rewardPhase)
            {
                return snapshot;
            }

            if (!snapshot.ActionLookup.TryGetValue("proceed", out var proceedAction))
            {
                diagnostics.Add(
                    $"Combat sandbox fast-reset: reward-like state had no proceed action (phase={snapshot.Phase}, screen={snapshot.Screen})");
                return snapshot;
            }

            diagnostics.Add(
                $"Combat sandbox fast-reset: skipping reward flow via proceed from phase={snapshot.Phase}, screen={snapshot.Screen}");
            await ExecuteEnvActionAsync(
                proceedAction,
                timeoutMs,
                cancellationToken,
                "combat_sandbox.fast_reset.proceed");

            snapshot = await WaitForStableEnvStateAsync(
                snapshot.LogicHash,
                Math.Min(timeoutMs, 5000),
                requireActionableOrDone: true,
                cancellationToken);
            diagnostics.Add(
                $"Combat sandbox fast-reset: reward proceed settled at phase={snapshot.Phase}, screen={snapshot.Screen}, run_active={snapshot.RunActive}, combat_in_progress={snapshot.CombatInProgress}");
        }

        return snapshot;
    }

    private static Task BeginFreshCombatSandboxRun(
        BridgeEnvCombatResetRequest request,
        BridgeEnvSnapshot priorState,
        List<string> diagnostics)
    {
        var game = NGame.Instance
                   ?? throw new InvalidOperationException("NGame.Instance is null.");
        var runManager = RunManager.Instance;
        var saveManager = SaveManager.Instance
                          ?? throw new InvalidOperationException("SaveManager.Instance is null.");

        if (runManager.IsInProgress)
        {
            runManager.CleanUp(graceful: false);
            diagnostics.Add("Cleaned up existing run before combat sandbox bootstrap");
        }

        var character = ResolveCombatSandboxCharacter(request.Character, priorState, diagnostics);
        var seed = request.Seed?.ToString(CultureInfo.InvariantCulture) ?? SeedHelper.GetRandomSeed();
        diagnostics.Add($"Fresh combat sandbox run config: character={character.Id}, seed={seed}");

        var player = Player.CreateForNewRun(
            character,
            saveManager.GenerateUnlockStateFromProgress(),
            1uL);

        var runState = RunState.CreateForNewRun(
            new[] { player },
            ActModel.GetDefaultList().Select(act => act.ToMutable()).ToList(),
            Array.Empty<ModifierModel>(),
            GameMode.Standard,
            0,
            seed);

        SetUpNewSinglePlayerCompatibility(runManager, runState, shouldSave: false);
        ForceCombatSandboxNoNeowStartup(runState, diagnostics);
        diagnostics.Add("Prepared fresh singleplayer run state for combat sandbox");

        return StartFreshCombatSandboxRunAsync(game, runState, request, diagnostics);
    }

    private static async Task StartFreshCombatSandboxRunAsync(
        NGame game,
        RunState runState,
        BridgeEnvCombatResetRequest request,
        List<string> diagnostics)
    {
        using (new NetLoadingHandle(RunManager.Instance.NetService))
        {
            await PreloadManager.LoadRunAssets(runState.Players.Select(player => player.Character));
            await PreloadManager.LoadActAssets(runState.Acts[0]);
            await RunManager.Instance.FinalizeStartingRelics();
            RunManager.Instance.Launch();
            game.RootSceneContainer.SetCurrentScene(NRun.Create(runState));
            ForceCombatSandboxNoNeowStartup(runState, diagnostics);
            await RunManager.Instance.EnterAct(0, doTransition: false);
        }

        var context = CaptureContext();
        var player = GetPrimaryPlayer(context);
        if (player is not null)
        {
            ApplyCombatSandboxPlayerOverrides(player, request, diagnostics);
        }
        else
        {
            diagnostics.Add("Fresh combat sandbox run started but active player was not found for overrides");
        }

        diagnostics.Add("Started fresh singleplayer run for combat sandbox");
    }

    private static void ForceCombatSandboxNoNeowStartup(
        RunState? runState,
        List<string> diagnostics)
    {
        if (runState?.ExtraFields is null)
        {
            diagnostics.Add("Combat sandbox bootstrap could not force StartedWithNeow=false because run state extra fields were unavailable");
            return;
        }

        runState.ExtraFields.StartedWithNeow = false;
        diagnostics.Add("Forced StartedWithNeow=false for combat sandbox bootstrap");
    }

    private static CharacterModel ResolveCombatSandboxCharacter(
        string? requestedCharacter,
        BridgeEnvSnapshot priorState,
        List<string> diagnostics)
    {
        if (!string.IsNullOrWhiteSpace(requestedCharacter))
        {
            var requested = requestedCharacter.Trim();
            if (TryModelDbGetById("CharacterModel", requested, diagnostics) is CharacterModel requestedById)
            {
                diagnostics.Add($"Resolved combat sandbox character by id: {requestedById.Id}");
                return requestedById;
            }

            var normalizedRequested = NormalizeComparableText(requested);
            var requestedByTitle = ModelDb.AllCharacters.FirstOrDefault(character =>
                NormalizeComparableText(character.Id.ToString()).Equals(normalizedRequested, StringComparison.Ordinal) ||
                NormalizeComparableText(character.Id.Entry).Equals(normalizedRequested, StringComparison.Ordinal) ||
                NormalizeComparableText(character.GetType().Name).Equals(normalizedRequested, StringComparison.Ordinal) ||
                NormalizeComparableText(DescribeCharacter(character)).Equals(normalizedRequested, StringComparison.Ordinal));
            if (requestedByTitle is not null)
            {
                diagnostics.Add($"Resolved combat sandbox character by title: {requestedByTitle.Id}");
                return requestedByTitle;
            }

            CharacterModel? requestedByAlias = normalizedRequested switch
            {
                "ironclad" => ModelDb.Character<Ironclad>(),
                "silent" => ModelDb.Character<Silent>(),
                "defect" => ModelDb.Character<Defect>(),
                "regent" => ModelDb.Character<Regent>(),
                "necrobinder" => ModelDb.Character<Necrobinder>(),
                "deprived" => ModelDb.Character<Deprived>(),
                _ => null
            };
            if (requestedByAlias is not null)
            {
                diagnostics.Add($"Resolved combat sandbox character by alias: {requestedByAlias.Id}");
                return requestedByAlias;
            }

            throw new InvalidOperationException($"Could not resolve character '{requested}'.");
        }

        var currentCharacter = GetPrimaryPlayer(priorState.Context)?.Character;
        if (currentCharacter is not null)
        {
            diagnostics.Add($"Falling back to current run character for combat sandbox: {currentCharacter.Id}");
            return currentCharacter;
        }

        var defaultCharacter = ModelDb.AllCharacters.FirstOrDefault()
                               ?? throw new InvalidOperationException("ModelDb.AllCharacters is empty.");
        diagnostics.Add($"Falling back to default combat sandbox character: {defaultCharacter.Id}");
        return defaultCharacter;
    }

    private static void ApplyCombatSandboxRunReuseOverrides(
        BridgeEnvCombatResetRequest request,
        List<string> diagnostics)
    {
        var context = CaptureContext();
        var player = GetPrimaryPlayer(context);
        if (player is null)
        {
            diagnostics.Add("Combat sandbox reuse overrides skipped: no active player");
            return;
        }

        ApplyCombatSandboxPlayerOverrides(player, request, diagnostics);
    }

    private static void ApplyCombatSandboxPlayerOverrides(
        Player player,
        BridgeEnvCombatResetRequest request,
        List<string> diagnostics)
    {
        var creature = player.Creature;
        if (creature is null)
        {
            diagnostics.Add("Combat sandbox bootstrap: player creature is null; skipping overrides");
            return;
        }

        if (request.MaxHp is > 0)
        {
            creature.SetMaxHpInternal(request.MaxHp.Value);
            diagnostics.Add($"Set MaxHp to {request.MaxHp.Value}");
        }

        if (request.CurrentHp is > 0)
        {
            creature.SetCurrentHpInternal(request.CurrentHp.Value);
            diagnostics.Add($"Set CurrentHp to {Math.Min(request.CurrentHp.Value, creature.MaxHp)}");
        }
        else
        {
            creature.SetCurrentHpInternal(creature.MaxHp);
            diagnostics.Add($"Restored CurrentHp to full ({creature.MaxHp})");
        }

        if (request.Gold is not null)
        {
            player.Gold = request.Gold.Value;
            diagnostics.Add($"Set Gold to {request.Gold.Value}");
        }

        if (request.MaxEnergy is > 0)
        {
            player.MaxEnergy = request.MaxEnergy.Value;
            diagnostics.Add($"Set MaxEnergy to {request.MaxEnergy.Value}");
        }

        var requestedDeckEntries = ResolveRequestedDeckEntries(request);
        if (requestedDeckEntries.Count > 0)
        {
            var liveRunState = player.RunState as RunState;
            if (liveRunState is null)
            {
                diagnostics.Add("Deck override warning: player.RunState was not a live RunState; deck cards will be attached without run-state registration");
            }

            foreach (var existingCard in player.Deck.Cards.ToList())
            {
                player.Deck.RemoveInternal(existingCard, silent: true);
                if (liveRunState is not null)
                {
                    liveRunState.RemoveCard(existingCard);
                }
            }

            var added = 0;
            foreach (var entry in requestedDeckEntries)
            {
                if (BuildMutableDeckCard(entry.CardId, entry.UpgradeLevel, diagnostics) is not CardModel mutableCard)
                {
                    continue;
                }

                mutableCard.FloorAddedToDeck = 1;
                player.Deck.AddInternal(mutableCard, -1, silent: true);
                if (liveRunState is not null)
                {
                    liveRunState.AddCard(mutableCard, player);
                }
                else
                {
                    mutableCard.Owner = player;
                }

                mutableCard.AfterCreated();
                // Apply pre-combat enchantments / afflictions AFTER the card has
                // its owner + RunState wired so any modifier that touches owner
                // state in its on-attach hook can find a valid context.
                if (entry.Enchantments.Count > 0 || entry.Afflictions.Count > 0)
                {
                    ApplyDeckCardModifiers(mutableCard, entry, diagnostics);
                }
                added++;
            }

            diagnostics.Add($"Deck override: added {added}/{requestedDeckEntries.Count} cards");
        }

        if (request.Relics is { Length: > 0 })
        {
            foreach (var relic in player.Relics.ToList())
            {
                player.RemoveRelicInternal(relic, silent: true);
            }

            var added = 0;
            foreach (var relicId in request.Relics)
            {
                if (TryModelDbGetById("RelicModel", relicId, diagnostics) is not RelicModel relic)
                {
                    diagnostics.Add($"Relic override skipped unknown relic '{relicId}'");
                    continue;
                }

                var mutableRelic = relic.ToMutable();
                mutableRelic.FloorAddedToDeck = 1;
                SaveManager.Instance?.MarkRelicAsSeen(mutableRelic);
                player.AddRelicInternal(mutableRelic, -1, silent: true);
                added++;
            }

            diagnostics.Add($"Relic override: added {added}/{request.Relics.Length} relics");
        }

        if (request.Potions is { Length: > 0 })
        {
            var targetSlotCount = Math.Max(player.PotionSlots.Count, request.Potions.Length);
            var setMaxPotionCountInternal = FindMethod(player.GetType(), "SetMaxPotionCountInternal", 1);
            if (setMaxPotionCountInternal is not null)
            {
                setMaxPotionCountInternal.Invoke(player, new object[] { targetSlotCount });
            }
            else if (request.Potions.Length > player.PotionSlots.Count)
            {
                diagnostics.Add(
                    $"Potion override could not expand slots from {player.PotionSlots.Count} to {targetSlotCount}");
            }

            foreach (var potion in player.PotionSlots.ToList())
            {
                if (potion is not null)
                {
                    // silent=false fires PotionDiscarded so the UI removes the icon;
                    // see Player.DiscardPotionInternal in the decompiled core.
                    player.DiscardPotionInternal(potion, silent: false);
                }
            }

            var added = 0;
            for (var i = 0; i < Math.Min(request.Potions.Length, player.PotionSlots.Count); i++)
            {
                var potionId = request.Potions[i];
                if (string.IsNullOrWhiteSpace(potionId))
                {
                    continue;
                }

                if (TryModelDbGetById("PotionModel", potionId, diagnostics) is not PotionModel potion)
                {
                    diagnostics.Add($"Potion override skipped unknown potion '{potionId}'");
                    continue;
                }

                // silent=false fires PotionProcured so the UI renders the new potion.
                // With silent=true the backing _potionSlots[i] is updated (bridge obs sees it)
                // but the UI never refreshes, leaving the policy training on a deck of phantom potions.
                player.AddPotionInternal(potion.ToMutable(), i, silent: false);
                added++;
            }

            diagnostics.Add($"Potion override: added {added}/{request.Potions.Length} potions");
        }
    }

    private static async Task<bool> TryDeferredEnterCombatRoomAsync(
        string encounterId,
        List<string> diagnostics,
        int timeoutMs,
        CancellationToken cancellationToken)
    {
        var enterTask = await RunOnMainThreadGuardedAsync(
            () =>
            {
                var encounter = ResolveEncounterModel(encounterId, diagnostics);
                var runManager = RunManager.Instance;
                if (encounter is null || runManager is null)
                    return null;

                return TryInvokeEnterRoomDebug(runManager, encounter, diagnostics, out var pendingTask)
                    ? pendingTask ?? Task.CompletedTask
                    : null;
            },
            "combat_sandbox.deferred_enter",
            timeoutMs,
            cancellationToken);

        if (enterTask is null)
        {
            diagnostics.Add("Deferred EnterRoomDebug did not start");
            return false;
        }

        try
        {
            await enterTask.WaitAsync(TimeSpan.FromMilliseconds(Math.Min(timeoutMs, 5000)), cancellationToken);
            diagnostics.Add("Deferred EnterRoomDebug task completed");
            return true;
        }
        catch (Exception ex)
        {
            diagnostics.Add($"Deferred EnterRoomDebug await failed: {ex.GetBaseException().Message}");
            diagnostics.Add($"Deferred EnterRoomDebug exception detail: {ex}");
            return false;
        }
    }

    private static async Task<BridgeEnvSnapshot> WaitForCombatSandboxReadyStateAsync(
        string? baselineLogicHash,
        int timeoutMs,
        List<string> diagnostics,
        CancellationToken cancellationToken)
    {
        var startedAt = DateTime.UtcNow;
        var stableHash = string.Empty;
        var stableCount = 0;
        BridgeEnvSnapshot? lastSnapshot = null;

        while ((DateTime.UtcNow - startedAt).TotalMilliseconds < timeoutMs)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var snapshot = await CaptureEnvSnapshotAsync(timeoutMs, cancellationToken, "combat_sandbox.wait_ready.snapshot");
            snapshot = await MaybeAutoCloseResidualMapOverlayAsync(snapshot, timeoutMs, cancellationToken);
            lastSnapshot = snapshot;

            var changedFromBaseline = baselineLogicHash is null ||
                                      !baselineLogicHash.Equals(snapshot.LogicHash, StringComparison.Ordinal);
            var ready = snapshot.Done || IsCombatSandboxResetReady(snapshot);

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
                    diagnostics.Add(
                        $"Combat sandbox ready: phase={snapshot.Phase}, screen={snapshot.Screen}, actionable={snapshot.Actionable}, combat_in_progress={snapshot.CombatInProgress}, room_type={snapshot.RoomType}, room_model_id={snapshot.RoomModelId}");
                    return snapshot;
                }
            }
            else
            {
                stableHash = string.Empty;
                stableCount = 0;
            }

            await WaitForPumpTicksGuardedAsync(1, "combat_sandbox.wait_ready.wait_pump", timeoutMs, cancellationToken);
        }

        if (lastSnapshot is not null)
        {
            diagnostics.Add(
                $"Combat sandbox ready wait timed out at phase={lastSnapshot.Phase}, screen={lastSnapshot.Screen}, actionable={lastSnapshot.Actionable}, combat_in_progress={lastSnapshot.CombatInProgress}, room_type={lastSnapshot.RoomType}, room_model_id={lastSnapshot.RoomModelId}");
            return lastSnapshot;
        }

        var finalSnapshot = await CaptureEnvSnapshotAsync(timeoutMs, cancellationToken, "combat_sandbox.wait_ready.final_snapshot");
        diagnostics.Add(
            $"Combat sandbox ready wait ended with fallback snapshot phase={finalSnapshot.Phase}, screen={finalSnapshot.Screen}, actionable={finalSnapshot.Actionable}, combat_in_progress={finalSnapshot.CombatInProgress}, room_type={finalSnapshot.RoomType}, room_model_id={finalSnapshot.RoomModelId}");
        return finalSnapshot;
    }

    private static bool IsCombatSandboxResetReady(BridgeEnvSnapshot snapshot)
    {
        if (!snapshot.RunActive || !snapshot.CombatInProgress || !snapshot.Actionable)
        {
            return false;
        }

        if (string.Equals(snapshot.Phase, "combat", StringComparison.Ordinal) &&
            string.Equals(snapshot.Screen, "COMBAT", StringComparison.OrdinalIgnoreCase))
        {
            return true;
        }

        // Some relics can legitimately open a combat-start selection surface before the first
        // playable combat frame (for example Gambling Chip mulligan). That is still a valid
        // combat sandbox start state and should not hard-fail reset.
        return string.Equals(snapshot.Phase, "card_selection", StringComparison.Ordinal) &&
               (string.Equals(snapshot.Screen, "CARD_SELECTION", StringComparison.OrdinalIgnoreCase) ||
                string.Equals(snapshot.Screen, "COMBAT", StringComparison.OrdinalIgnoreCase));
    }

    private static bool IsCombatSandboxExplicitPostCombatPhase(string? phase)
    {
        return string.Equals(phase, "terminal", StringComparison.Ordinal) ||
               string.Equals(phase, "reward", StringComparison.Ordinal) ||
               string.Equals(phase, "card_reward", StringComparison.Ordinal) ||
               string.Equals(phase, "map", StringComparison.Ordinal) ||
               string.Equals(phase, "event", StringComparison.Ordinal) ||
               string.Equals(phase, "event_crystal_sphere", StringComparison.Ordinal) ||
               string.Equals(phase, "shop", StringComparison.Ordinal) ||
               string.Equals(phase, "rest_site", StringComparison.Ordinal) ||
               string.Equals(phase, "treasure", StringComparison.Ordinal);
    }

    private static bool IsCombatSandboxExplicitTerminalSurface(BridgeEnvSnapshot snapshot)
    {
        if (snapshot.Done || snapshot.CurrentHp <= 0)
        {
            return true;
        }

        if (IsCombatSandboxExplicitPostCombatPhase(snapshot.Phase))
        {
            return true;
        }

        var combatLikePhase =
            string.Equals(snapshot.Phase, "combat", StringComparison.Ordinal) ||
            string.Equals(snapshot.Phase, "card_selection", StringComparison.Ordinal) ||
            string.Equals(snapshot.Phase, "settling", StringComparison.Ordinal);
        var combatLikeScreen =
            string.Equals(snapshot.Screen, "COMBAT", StringComparison.OrdinalIgnoreCase) ||
            string.Equals(snapshot.Screen, "CARD_SELECTION", StringComparison.OrdinalIgnoreCase);

        return !snapshot.CombatInProgress && !combatLikePhase && !combatLikeScreen;
    }

    private static bool LooksLikeCombatSandboxLiveSurface(BridgeEnvSnapshot snapshot)
    {
        if (snapshot.Done || snapshot.CurrentHp <= 0)
        {
            return false;
        }

        return snapshot.CombatInProgress ||
               string.Equals(snapshot.Phase, "combat", StringComparison.Ordinal) ||
               string.Equals(snapshot.Phase, "card_selection", StringComparison.Ordinal) ||
               string.Equals(snapshot.Phase, "settling", StringComparison.Ordinal) ||
               string.Equals(snapshot.Screen, "COMBAT", StringComparison.OrdinalIgnoreCase) ||
               string.Equals(snapshot.Screen, "CARD_SELECTION", StringComparison.OrdinalIgnoreCase) ||
               string.Equals(snapshot.RoomType, "COMBAT", StringComparison.OrdinalIgnoreCase);
    }

    private static bool ShouldAttemptCombatSandboxLiveStateSalvage(BridgeEnvSnapshot snapshot)
    {
        return !snapshot.Done &&
               !snapshot.Actionable &&
               !IsEnvIntermediateDecisionSurface(snapshot) &&
               !IsCombatSandboxEpisodeDone(snapshot) &&
               LooksLikeCombatSandboxLiveSurface(snapshot);
    }

    private static async Task<BridgeEnvSnapshot> TrySalvageCombatSandboxLiveStepStateAsync(
        BridgeEnvEpisode episode,
        BridgeEnvSnapshot state,
        int timeoutMs,
        CancellationToken cancellationToken,
        string operationName,
        BridgeEnvStepTimingCollector? timing = null)
    {
        if (!ShouldAttemptCombatSandboxLiveStateSalvage(state))
        {
            return state;
        }

        var salvageBudgetMs = Math.Min(Math.Max(timeoutMs / 4, 1500), 5000);
        var fastWaitBudgetMs = Math.Min(salvageBudgetMs, 2000);
        var fastState = await WaitForCombatSandboxFastStateAsync(
            state.LogicHash,
            fastWaitBudgetMs,
            requireActionableOrDone: true,
            cancellationToken,
            $"{operationName}.fast",
            timing,
            baselineSnapshot: state);
        var adjustmentsStopwatch = Stopwatch.StartNew();
        fastState = await ApplyEnvEpisodeAdjustmentsAsync(episode, fastState, timeoutMs, cancellationToken, timing);
        if (timing is not null)
        {
            timing.EpisodeAdjustmentsMs += adjustmentsStopwatch.Elapsed.TotalMilliseconds;
        }

        if (fastState.Actionable ||
            fastState.Done ||
            IsEnvIntermediateDecisionSurface(fastState) ||
            IsCombatSandboxEpisodeDone(fastState) ||
            !LooksLikeCombatSandboxLiveSurface(fastState))
        {
            return fastState;
        }

        var stableWaitBudgetMs = Math.Max(salvageBudgetMs - fastWaitBudgetMs, 1500);
        var stableState = await WaitForStableEnvStateAsync(
            state.LogicHash,
            stableWaitBudgetMs,
            requireActionableOrDone: true,
            cancellationToken,
            timing,
            baselineSnapshot: fastState);
        adjustmentsStopwatch = Stopwatch.StartNew();
        stableState = await ApplyEnvEpisodeAdjustmentsAsync(episode, stableState, timeoutMs, cancellationToken, timing);
        if (timing is not null)
        {
            timing.EpisodeAdjustmentsMs += adjustmentsStopwatch.Elapsed.TotalMilliseconds;
        }
        return stableState;
    }

    private static async Task<BridgeEnvSnapshot> TrySalvageCombatSandboxSettlingResetAsync(
        BridgeEnvSnapshot state,
        int timeoutMs,
        List<string> diagnostics,
        CancellationToken cancellationToken)
    {
        if (state.Done || IsCombatSandboxResetReady(state))
        {
            return state;
        }

        if (!HasUsableCombatSandboxRunScene(state) || !state.CombatInProgress)
        {
            return state;
        }

        if (!string.Equals(state.Phase, "settling", StringComparison.Ordinal))
        {
            return state;
        }

        var looksLikeCombatSurface =
            string.Equals(state.Screen, "COMBAT", StringComparison.OrdinalIgnoreCase) ||
            string.Equals(state.Screen, "CARD_SELECTION", StringComparison.OrdinalIgnoreCase) ||
            string.Equals(state.RoomType, "COMBAT", StringComparison.OrdinalIgnoreCase);
        if (!looksLikeCombatSurface)
        {
            return state;
        }

        var salvageBudgetMs = Math.Min(Math.Max(timeoutMs / 3, 4000), 15000);
        var fastWaitBudgetMs = Math.Min(salvageBudgetMs, 5000);
        diagnostics.Add(
            $"Combat sandbox reset entered settling salvage window (phase={state.Phase}, screen={state.Screen}, actionable={state.Actionable}, combat_in_progress={state.CombatInProgress}, wait_budget_ms={salvageBudgetMs})");

        var fastState = await WaitForCombatSandboxFastStateAsync(
            null,
            fastWaitBudgetMs,
            requireActionableOrDone: true,
            cancellationToken,
            "combat_sandbox.reset.settling_salvage.fast");
        diagnostics.Add(
            $"Combat sandbox settling salvage fast wait observed phase={fastState.Phase}, screen={fastState.Screen}, actionable={fastState.Actionable}, combat_in_progress={fastState.CombatInProgress}, room_type={fastState.RoomType}, room_model_id={fastState.RoomModelId}");

        if (fastState.Done || IsCombatSandboxResetReady(fastState))
        {
            return fastState;
        }

        if (!HasUsableCombatSandboxRunScene(fastState) || !fastState.CombatInProgress)
        {
            return fastState;
        }

        var stableWaitBudgetMs = Math.Max(salvageBudgetMs - fastWaitBudgetMs, 2500);
        var stableState = await WaitForStableEnvStateAsync(
            null,
            stableWaitBudgetMs,
            requireActionableOrDone: true,
            cancellationToken);
        diagnostics.Add(
            $"Combat sandbox settling salvage stable wait observed phase={stableState.Phase}, screen={stableState.Screen}, actionable={stableState.Actionable}, combat_in_progress={stableState.CombatInProgress}, room_type={stableState.RoomType}, room_model_id={stableState.RoomModelId}");
        return stableState;
    }

    private static async Task<object> StepCombatSandboxEpisodeAsync(
        BridgeEnvStepRequest request,
        BridgeEnvEpisode episode,
        int timeoutMs,
        CancellationToken cancellationToken)
    {
        if (episode.Done)
        {
            throw new BridgeRequestException(
                HttpStatusCode.Conflict,
                "episode_already_done",
                $"Episode '{episode.Id}' is already done. Call env/combat_reset to start a new sandbox episode.");
        }

        var totalStopwatch = Stopwatch.StartNew();
        var timing = new BridgeEnvStepTimingCollector();
        timing.SnapshotCalls++;
        var beforeSnapshotStopwatch = Stopwatch.StartNew();
        var before = await CaptureEnvSnapshotAsync(timeoutMs, cancellationToken, "combat_sandbox.step.before_snapshot");
        timing.BeforeSnapshotMs += beforeSnapshotStopwatch.Elapsed.TotalMilliseconds;
        before = await MaybeAutoCloseResidualMapOverlayAsync(before, timeoutMs, cancellationToken, timing);
        var adjustmentsStopwatch = Stopwatch.StartNew();
        before = await ApplyEnvEpisodeAdjustmentsAsync(episode, before, timeoutMs, cancellationToken, timing);
        timing.EpisodeAdjustmentsMs += adjustmentsStopwatch.Elapsed.TotalMilliseconds;

        if (!before.Done && !before.Actionable && !IsEnvIntermediateDecisionSurface(before))
        {
            var beforeWaitStopwatch = Stopwatch.StartNew();
            before = await WaitForCombatSandboxFastStateAsync(
                before.LogicHash,
                Math.Min(timeoutMs, 2500),
                requireActionableOrDone: true,
                cancellationToken,
                "combat_sandbox.step.before_wait",
                timing,
                baselineSnapshot: before);
            timing.BeforeWaitMs += beforeWaitStopwatch.Elapsed.TotalMilliseconds;
            adjustmentsStopwatch = Stopwatch.StartNew();
            before = await ApplyEnvEpisodeAdjustmentsAsync(episode, before, timeoutMs, cancellationToken, timing);
            timing.EpisodeAdjustmentsMs += adjustmentsStopwatch.Elapsed.TotalMilliseconds;
        }

        if (before.Done || IsCombatSandboxEpisodeDone(before))
        {
            episode.Done = true;
            timing.TotalMs = totalStopwatch.Elapsed.TotalMilliseconds;
            var payload = BuildEnvStepPayload(
                episode,
                before,
                before,
                selectedAction: null,
                truncated: false,
                truncationReason: null,
                forceDone: true,
                timing: timing);
            return payload;
        }

        if (!before.Actionable && !IsEnvIntermediateDecisionSurface(before))
        {
            var beforeSalvageStopwatch = Stopwatch.StartNew();
            before = await TrySalvageCombatSandboxLiveStepStateAsync(
                episode,
                before,
                timeoutMs,
                cancellationToken,
                "combat_sandbox.step.before_wait_salvage",
                timing);
            timing.BeforeWaitMs += beforeSalvageStopwatch.Elapsed.TotalMilliseconds;
        }

        if (before.Done || IsCombatSandboxEpisodeDone(before))
        {
            episode.Done = true;
            timing.TotalMs = totalStopwatch.Elapsed.TotalMilliseconds;
            var payload = BuildEnvStepPayload(
                episode,
                before,
                before,
                selectedAction: null,
                truncated: false,
                truncationReason: null,
                forceDone: true,
                timing: timing);
            return payload;
        }

        if (!before.Actionable && !IsEnvIntermediateDecisionSurface(before))
        {
            episode.Done = true;
            timing.TotalMs = totalStopwatch.Elapsed.TotalMilliseconds;
            var payload = BuildEnvStepPayload(
                episode,
                before,
                before,
                selectedAction: null,
                truncated: true,
                truncationReason: "combat_sandbox_timeout_waiting_for_actionable_or_terminal_state",
                timing: timing);
            return payload;
        }

        BridgeResolvedActionSelection? selectedAction = null;
        string? actionError = null;
        try
        {
            var resolveStopwatch = Stopwatch.StartNew();
            selectedAction = ResolveRequestedEnvAction(before, request);
            timing.ActionResolveMs += resolveStopwatch.Elapsed.TotalMilliseconds;
            var executeStopwatch = Stopwatch.StartNew();
            await ExecuteEnvActionAsync(
                selectedAction.Action,
                timeoutMs,
                cancellationToken,
                $"combat_sandbox.step.execute:{selectedAction.Action.ActionId}");
            timing.ActionExecuteMs += executeStopwatch.Elapsed.TotalMilliseconds;
        }
        catch (OperationCanceledException)
        {
            throw;
        }
        catch (BridgeRequestException ex)
        {
            actionError = ex.ErrorCode;
        }
        catch (Exception)
        {
            actionError = "action_execution_error";
        }

        BridgeEnvSnapshot after;
        if (!string.IsNullOrWhiteSpace(actionError) && before.Actionable)
        {
            after = before;
        }
        else
        {
            var afterWaitStopwatch = Stopwatch.StartNew();
            // Normal card resolution (card fly + effect apply + draw proc) completes in
            // well under 1 second; anything past ~3s indicates either a pathological
            // unfocused-fps stall or a real game hang. Capping here keeps the worst-case
            // step round-trip at 3s + 5s salvage = 8s instead of the previous 20s + 5s
            // = 25s, which was triggering combat truncation/reset during training.
            const int CombatSandboxAfterWaitCapMs = 3000;
            var afterWaitTimeoutMs = IsCardSelectionSelectAction(selectedAction)
                ? GetCardSelectionSelectFastFailTimeoutMs(timeoutMs)
                : Math.Min(timeoutMs, CombatSandboxAfterWaitCapMs);
            after = await WaitForCombatSandboxFastStateAsync(
                before.LogicHash,
                afterWaitTimeoutMs,
                requireActionableOrDone: !ShouldAllowIntermediateSelectionState(selectedAction),
                cancellationToken,
                "combat_sandbox.step.after_wait",
                timing,
                baselineSnapshot: before);
            timing.AfterWaitMs += afterWaitStopwatch.Elapsed.TotalMilliseconds;
        }
        var autoConfirmStopwatch = Stopwatch.StartNew();
        after = await MaybeAutoConfirmSingleCardSelectionForCombatSandboxAsync(
            selectedAction,
            after,
            timeoutMs,
            cancellationToken);
        timing.AutoConfirmMs += autoConfirmStopwatch.Elapsed.TotalMilliseconds;
        adjustmentsStopwatch = Stopwatch.StartNew();
        after = await ApplyEnvEpisodeAdjustmentsAsync(episode, after, timeoutMs, cancellationToken, timing);
        timing.EpisodeAdjustmentsMs += adjustmentsStopwatch.Elapsed.TotalMilliseconds;

        if (!after.Actionable && !after.Done && !IsEnvIntermediateDecisionSurface(after))
        {
            var afterSalvageStopwatch = Stopwatch.StartNew();
            after = await TrySalvageCombatSandboxLiveStepStateAsync(
                episode,
                after,
                timeoutMs,
                cancellationToken,
                "combat_sandbox.step.after_wait_salvage",
                timing);
            timing.AfterWaitMs += afterSalvageStopwatch.Elapsed.TotalMilliseconds;
        }

        var cardSelectionNoProgressAfterAction =
            IsCardSelectionSelectAction(selectedAction) &&
            string.IsNullOrWhiteSpace(actionError) &&
            !after.Done &&
            !HasCardSelectionSelectionProgress(before, after);

        var noStateChangeAfterAction =
            selectedAction is not null &&
            string.IsNullOrWhiteSpace(actionError) &&
            !after.Done &&
            (cardSelectionNoProgressAfterAction ||
             (before.LogicHash.Equals(after.LogicHash, StringComparison.Ordinal) &&
              !HasMeaningfulEnvSnapshotDifference(before, after)));

        if (noStateChangeAfterAction)
        {
            actionError = "action_no_state_change";
        }

        if (!after.Done && IsCombatSandboxEpisodeDone(after))
        {
            episode.StepIndex++;
            episode.Done = true;
            timing.TotalMs = totalStopwatch.Elapsed.TotalMilliseconds;
            var payload = BuildEnvStepPayload(
                episode,
                before,
                after,
                selectedAction,
                truncated: false,
                truncationReason: null,
                actionError,
                forceDone: true,
                timing: timing);
            return payload;
        }

        episode.StepIndex++;
        var truncated = ((!after.Actionable && !after.Done && !IsEnvIntermediateDecisionSurface(after)) || noStateChangeAfterAction);
        if (after.Done || truncated)
        {
            episode.Done = true;
        }

        timing.TotalMs = totalStopwatch.Elapsed.TotalMilliseconds;
        var finalPayload = BuildEnvStepPayload(
            episode,
            before,
            after,
            selectedAction,
            truncated,
            truncated
                ? (noStateChangeAfterAction
                    ? "step_action_no_state_change"
                    : "combat_sandbox_timeout_waiting_for_actionable_or_terminal_state")
                : null,
            actionError,
            timing: timing);
        return finalPayload;
    }

    private static async Task<BridgeEnvSnapshot> WaitForCombatSandboxFastStateAsync(
        string? baselineLogicHash,
        int timeoutMs,
        bool requireActionableOrDone,
        CancellationToken cancellationToken,
        string operationName,
        BridgeEnvStepTimingCollector? timing = null,
        BridgeEnvSnapshot? baselineSnapshot = null)
    {
        var startedAt = DateTime.UtcNow;
        BridgeEnvSnapshot? lastSnapshot = null;

        while ((DateTime.UtcNow - startedAt).TotalMilliseconds < timeoutMs)
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (timing is not null)
            {
                timing.StableIterations++;
                timing.SnapshotCalls++;
            }
            var snapshot = await CaptureEnvSnapshotAsync(
                timeoutMs,
                cancellationToken,
                $"{operationName}.snapshot");
            snapshot = await MaybeAutoCloseResidualMapOverlayAsync(snapshot, timeoutMs, cancellationToken, timing);
            lastSnapshot = snapshot;

            var ready = snapshot.Done ||
                        IsCombatSandboxEpisodeDone(snapshot) ||
                        !requireActionableOrDone ||
                        snapshot.Actionable ||
                        IsEnvIntermediateDecisionSurface(snapshot);
            var changedFromBaseline = baselineLogicHash is null ||
                                      !baselineLogicHash.Equals(snapshot.LogicHash, StringComparison.Ordinal);
            var progressedFromBaseline = baselineSnapshot is not null &&
                                         HasMeaningfulEnvSnapshotDifference(baselineSnapshot, snapshot);
            if (ready && (changedFromBaseline || progressedFromBaseline))
            {
                return snapshot;
            }

            if (timing is not null)
            {
                timing.WaitPumpCalls++;
            }
            await WaitForPumpTicksGuardedAsync(
                1,
                $"{operationName}.wait_pump",
                timeoutMs,
                cancellationToken);
        }

        if (lastSnapshot is not null)
        {
            return lastSnapshot;
        }

        if (timing is not null)
        {
            timing.SnapshotCalls++;
        }
        return await CaptureEnvSnapshotAsync(
            timeoutMs,
            cancellationToken,
            $"{operationName}.final_snapshot");
    }

    private static async Task<BridgeEnvSnapshot> MaybeAutoConfirmSingleCardSelectionForCombatSandboxAsync(
        BridgeResolvedActionSelection? selectedAction,
        BridgeEnvSnapshot snapshot,
        int timeoutMs,
        CancellationToken cancellationToken)
    {
        if (selectedAction is null ||
            !selectedAction.Action.ActionId.StartsWith("card_selection:select:", StringComparison.Ordinal) ||
            !string.Equals(snapshot.Phase, "card_selection", StringComparison.Ordinal) ||
            snapshot.Done)
        {
            return snapshot;
        }

        var cardSelectionScreen = snapshot.Context.CardSelectionScreen;
        if (cardSelectionScreen is null || !IsNodeVisible(cardSelectionScreen))
        {
            return snapshot;
        }

        if (!ShouldAutoConfirmSingleCardSelection(cardSelectionScreen) ||
            CountSelectedCardSelectionCards(cardSelectionScreen) <= 0)
        {
            return snapshot;
        }

        if (!snapshot.ActionLookup.TryGetValue("card_selection:confirm", out var confirmAction))
        {
            return snapshot;
        }

        try
        {
            await ExecuteEnvActionAsync(
                confirmAction,
                timeoutMs,
                cancellationToken,
                "combat_sandbox.step.card_selection_confirm");
        }
        catch (OperationCanceledException)
        {
            throw;
        }
        catch
        {
            return snapshot;
        }

        return await WaitForCombatSandboxFastStateAsync(
            snapshot.LogicHash,
            timeoutMs,
            requireActionableOrDone: true,
            cancellationToken,
            "combat_sandbox.step.card_selection_confirm",
            baselineSnapshot: snapshot);
    }

    private static object? PrepareEncounterForRoomEntry(object? encounter, List<string> diagnostics)
    {
        if (encounter is null)
            return null;

        var mutableEncounter = TryCallToMutable(encounter);
        if (mutableEncounter is not null)
        {
            diagnostics.Add($"Converted encounter {encounter.GetType().Name} -> mutable {mutableEncounter.GetType().Name}");
            return mutableEncounter;
        }

        return encounter;
    }

    // -----------------------------------------------------------------------
    // GET /env/combat_catalog
    // -----------------------------------------------------------------------

}
