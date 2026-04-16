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

internal sealed class BridgeEnvCombatDeckEntryRequest
{
    [JsonPropertyName("id")]
    public string? Id { get; set; }

    [JsonPropertyName("upgrade_level")]
    public int? UpgradeLevel { get; set; }
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

    private static IReadOnlyList<(string CardId, int UpgradeLevel)> ResolveRequestedDeckEntries(
        BridgeEnvCombatResetRequest request)
    {
        if (request.DeckEntries is { Length: > 0 })
        {
            return request.DeckEntries
                .Where(static entry => !string.IsNullOrWhiteSpace(entry?.Id))
                .Select(static entry => (
                    entry!.Id!.Trim(),
                    Math.Max(entry.UpgradeLevel ?? 0, 0)))
                .ToArray();
        }

        if (request.Deck is { Length: > 0 })
        {
            return request.Deck
                .Where(static cardId => !string.IsNullOrWhiteSpace(cardId))
                .Select(static cardId => (cardId.Trim(), 0))
                .ToArray();
        }

        return Array.Empty<(string CardId, int UpgradeLevel)>();
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

        runManager.SetUpNewSinglePlayer(runState, shouldSave: false);
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
            foreach (var (cardId, upgradeLevel) in requestedDeckEntries)
            {
                if (BuildMutableDeckCard(cardId, upgradeLevel, diagnostics) is not CardModel mutableCard)
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
                    player.DiscardPotionInternal(potion, silent: true);
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

                player.AddPotionInternal(potion.ToMutable(), i, silent: true);
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
            var afterWaitTimeoutMs = IsCardSelectionSelectAction(selectedAction)
                ? GetCardSelectionSelectFastFailTimeoutMs(timeoutMs)
                : timeoutMs;
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

        try
        {
            await RunOnMainThreadGuardedAsync(
                () =>
                {
                    InvokeCardSelectionConfirmAction(
                        cardSelectionScreen,
                        ResolveCardSelectionConfirmButton(cardSelectionScreen));
                    return true;
                },
                "combat_sandbox.step.card_selection_confirm",
                timeoutMs,
                cancellationToken);

            await WaitForPumpTicksGuardedAsync(
                1,
                "combat_sandbox.step.card_selection_confirm.post_pump",
                timeoutMs,
                cancellationToken);
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

    public static async Task<object> GetCombatCatalogResponseAsync(CancellationToken cancellationToken)
    {
        await WaitForEnvDispatcherReadyAsync(5000, cancellationToken);

        var encounters = await RunOnMainThreadGuardedAsync(
            ListAvailableCombatEncounters,
            "combat_sandbox.list_catalog",
            5000,
            cancellationToken);

        return new
        {
            ok = true,
            encounters
        };
    }

    // -----------------------------------------------------------------------
    // Combat sandbox done detection
    // -----------------------------------------------------------------------

    private static bool IsCombatSandboxEpisodeDone(BridgeEnvSnapshot snapshot)
    {
        return IsCombatSandboxExplicitTerminalSurface(snapshot);
    }

    // -----------------------------------------------------------------------
    // Setup internals
    // -----------------------------------------------------------------------

    private sealed class CombatSandboxSetupResult
    {
        public bool Success { get; init; }
        public string? ErrorCode { get; init; }
        public string? ErrorMessage { get; init; }
        public Task? PendingTask { get; init; }
    }

    private sealed class EncounterCatalogEntry
    {
        public string EncounterId { get; init; } = string.Empty;
        public string DisplayName { get; init; } = string.Empty;
        public string TypeName { get; init; } = string.Empty;
        public string Category { get; init; } = string.Empty;
        public bool IsMock { get; init; }
    }

    private static CombatSandboxSetupResult SetUpCombatSandbox(
        BridgeEnvCombatResetRequest request,
        string encounterId,
        List<string> diagnostics)
    {
        try
        {
            diagnostics.Add($"Combat sandbox setup begin: encounter_id={encounterId}");

            // 1. Resolve encounter model
            var encounter = ResolveEncounterModel(encounterId, diagnostics);
            if (encounter is null)
            {
                return new CombatSandboxSetupResult
                {
                    Success = false,
                    ErrorCode = "encounter_not_found",
                    ErrorMessage = $"Could not resolve encounter '{encounterId}'. Use GET /env/combat_catalog to list available encounters."
                };
            }
            diagnostics.Add($"Resolved encounter: {encounter.GetType().Name} (Id={TryGetModelId(encounter)})");

            // 2. Get RunManager instance
            var runManager = RunManager.Instance;
            if (runManager is null)
            {
                return new CombatSandboxSetupResult
                {
                    Success = false,
                    ErrorCode = "run_manager_not_available",
                    ErrorMessage = "RunManager.Instance is null."
                };
            }

            diagnostics.Add("Combat sandbox setup stage: before CombatManager reset");
            ResetCombatManagerForSandboxEntry(diagnostics);
            diagnostics.Add("Combat sandbox setup stage: after CombatManager reset");

            diagnostics.Add("Combat sandbox setup stage: before run reuse overrides");
            ApplyCombatSandboxRunReuseOverrides(request, diagnostics);
            diagnostics.Add("Combat sandbox setup stage: after run reuse overrides");

            // 3. Enter the requested encounter on the fresh active run scene.
            diagnostics.Add("Combat sandbox setup stage: before combat room entry");
            var entered = TryEnterCombatViaDebugRoom(runManager, encounter, diagnostics, out var pendingTask);
            diagnostics.Add($"Combat sandbox setup stage: combat room entry result={entered}");
            if (!entered)
            {
                return new CombatSandboxSetupResult
                {
                    Success = false,
                    ErrorCode = "combat_entry_failed",
                    ErrorMessage = "Could not enter combat room. Check diagnostics for reflection probe results."
                };
            }

            diagnostics.Add("Combat sandbox setup stage: before post-entry overrides");
            ApplyPostCombatSandboxOverrides(request, diagnostics);
            diagnostics.Add("Combat sandbox setup stage: after post-entry overrides");

            return new CombatSandboxSetupResult
            {
                Success = true,
                PendingTask = pendingTask
            };
        }
        catch (Exception ex)
        {
            diagnostics.Add($"Exception during setup: {ex.GetType().Name}: {ex.Message}");
            diagnostics.Add($"Exception detail: {ex}");
            if (ex.InnerException is not null)
            {
                diagnostics.Add($"Inner exception detail: {ex.InnerException}");
            }

            return new CombatSandboxSetupResult
            {
                Success = false,
                ErrorCode = "combat_sandbox_exception",
                ErrorMessage = ex.Message
            };
        }
    }

    private static void ResetCombatManagerForSandboxEntry(List<string> diagnostics)
    {
        var combatManager = CombatManager.Instance;
        if (combatManager is null)
        {
            diagnostics.Add("Combat sandbox setup: CombatManager.Instance was null; skipping combat reset");
            return;
        }

        BridgeWorldContext? context = null;
        try
        {
            context = CaptureContext();
        }
        catch (Exception ex)
        {
            diagnostics.Add($"Combat sandbox setup: CaptureContext before reset failed: {ex.GetBaseException().Message}");
            diagnostics.Add($"Combat sandbox setup: reset context exception detail: {ex}");
        }

        var currentRoom = context?.RunState?.CurrentRoom;
        var currentRoomName = currentRoom?.GetType().Name ?? "<null>";
        var hasCombatState = combatManager.DebugOnlyGetState() is not null;
        var shouldTryGracefulReset = hasCombatState &&
                                     combatManager.IsInProgress &&
                                     currentRoom is CombatRoom;

        diagnostics.Add(
            $"Combat sandbox setup: reset precheck has_state={hasCombatState}, in_progress={combatManager.IsInProgress}, current_room={currentRoomName}");

        if (shouldTryGracefulReset && TryResetCombatManagerGracefully(combatManager, diagnostics))
        {
            diagnostics.Add("Combat sandbox setup: reset CombatManager before entering new encounter");
            return;
        }

        HardResetCombatManagerForSandboxEntry(combatManager, diagnostics);
        diagnostics.Add("Combat sandbox setup: reset CombatManager before entering new encounter");
    }

    private static bool TryResetCombatManagerGracefully(
        CombatManager combatManager,
        List<string> diagnostics)
    {
        try
        {
            combatManager.Reset(graceful: true);
            diagnostics.Add("Combat sandbox setup: graceful CombatManager.Reset(true) succeeded");
            return true;
        }
        catch (Exception ex)
        {
            diagnostics.Add($"Combat sandbox setup: graceful CombatManager.Reset(true) failed: {ex.GetBaseException().Message}");
            diagnostics.Add($"Combat sandbox setup: graceful reset exception detail: {ex}");
            return false;
        }
    }

    private static void HardResetCombatManagerForSandboxEntry(
        CombatManager combatManager,
        List<string> diagnostics)
    {
        var hardResetDiagnostics = new List<string>();

        TrySetPrivateFieldValue(combatManager, "_state", null, hardResetDiagnostics);
        TrySetPrivateFieldValue(combatManager, "_pendingLoss", null, hardResetDiagnostics);

        try
        {
            combatManager.Reset(graceful: false);
            hardResetDiagnostics.Add("CombatManager.Reset(false) succeeded after nulling _state");
        }
        catch (Exception ex)
        {
            hardResetDiagnostics.Add($"CombatManager.Reset(false) failed: {ex.GetBaseException().Message}");
            hardResetDiagnostics.Add($"CombatManager.Reset(false) exception detail: {ex.GetType().Name}");
        }

        TrySetPrivateFieldValue(combatManager, "_state", null, hardResetDiagnostics);
        TrySetPrivateFieldValue(combatManager, "_pendingLoss", null, hardResetDiagnostics);
        TrySetPrivateFieldValue(combatManager, "_playerActionsDisabled", false, hardResetDiagnostics);
        TrySetPrivateFieldValue(combatManager, "<DebugForcedTopCardOnNextShuffle>k__BackingField", null, hardResetDiagnostics);
        TrySetPrivateFieldValue(combatManager, "<IsPaused>k__BackingField", false, hardResetDiagnostics);
        TrySetPrivateFieldValue(combatManager, "<IsPlayPhase>k__BackingField", false, hardResetDiagnostics);
        TrySetPrivateFieldValue(combatManager, "<IsEnemyTurnStarted>k__BackingField", false, hardResetDiagnostics);
        TrySetPrivateFieldValue(combatManager, "<EndingPlayerTurnPhaseTwo>k__BackingField", false, hardResetDiagnostics);
        TrySetPrivateFieldValue(combatManager, "<EndingPlayerTurnPhaseOne>k__BackingField", false, hardResetDiagnostics);
        TrySetPrivateFieldValue(combatManager, "<IsInProgress>k__BackingField", false, hardResetDiagnostics);

        TryClearPrivateCollection(combatManager, "_playersReadyToEndTurn", hardResetDiagnostics);
        TryClearPrivateCollection(combatManager, "_playersReadyToBeginEnemyTurn", hardResetDiagnostics);
        TryClearPrivateCollection(combatManager, "_playersTakingExtraTurn", hardResetDiagnostics);

        try
        {
            combatManager.History.Clear();
            hardResetDiagnostics.Add("History.Clear()");
        }
        catch (Exception ex)
        {
            hardResetDiagnostics.Add($"History.Clear() failed: {ex.GetBaseException().Message}");
        }

        if (combatManager.StateTracker is not null)
        {
            TrySetPrivateFieldValue(combatManager.StateTracker, "_state", null, hardResetDiagnostics);
            TrySetPrivateFieldValue(combatManager.StateTracker, "_combatStateChangedDeferredTask", null, hardResetDiagnostics);
        }

        TryUpdateActionQueueCombatState(hardResetDiagnostics);

        diagnostics.Add("Combat sandbox setup: hard CombatManager reset fallback applied");
        diagnostics.Add($"Combat sandbox setup: hard reset details => {string.Join("; ", hardResetDiagnostics)}");
    }

    private static void TrySetPrivateFieldValue(
        object target,
        string fieldName,
        object? value,
        List<string> diagnostics)
    {
        try
        {
            var field = FindField(target.GetType(), fieldName);
            if (field is null)
            {
                diagnostics.Add($"{fieldName}: field not found");
                return;
            }

            field.SetValue(target, value);
            diagnostics.Add($"{fieldName}={(value is null ? "null" : value)}");
        }
        catch (Exception ex)
        {
            diagnostics.Add($"{fieldName}: set failed ({ex.GetBaseException().Message})");
        }
    }

    private static void TryClearPrivateCollection(
        object target,
        string fieldName,
        List<string> diagnostics)
    {
        try
        {
            var field = FindField(target.GetType(), fieldName);
            var collection = field?.GetValue(target);
            if (collection is null)
            {
                diagnostics.Add($"{fieldName}: collection unavailable");
                return;
            }

            var clearMethod = collection.GetType().GetMethod(
                "Clear",
                BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic,
                null,
                Type.EmptyTypes,
                null);
            if (clearMethod is null)
            {
                diagnostics.Add($"{fieldName}: Clear() not found");
                return;
            }

            clearMethod.Invoke(collection, null);
            diagnostics.Add($"{fieldName}.Clear()");
        }
        catch (Exception ex)
        {
            diagnostics.Add($"{fieldName}: clear failed ({ex.GetBaseException().Message})");
        }
    }

    private static void TryUpdateActionQueueCombatState(List<string> diagnostics)
    {
        try
        {
            var runManager = RunManager.Instance;
            if (runManager?.ActionQueueSynchronizer is null)
            {
                diagnostics.Add("ActionQueueSynchronizer unavailable; skipped SetCombatState(NotInCombat)");
                return;
            }

            runManager.ActionQueueSynchronizer.SetCombatState(ActionSynchronizerCombatState.NotInCombat);
            diagnostics.Add("ActionQueueSynchronizer.SetCombatState(NotInCombat)");
        }
        catch (Exception ex)
        {
            diagnostics.Add($"ActionQueueSynchronizer.SetCombatState(NotInCombat) failed: {ex.GetBaseException().Message}");
        }
    }

    // -----------------------------------------------------------------------
    // Encounter resolution
    // -----------------------------------------------------------------------

    private static object? ResolveEncounterModel(string encounterId, List<string> diagnostics)
    {
        // Strategy 1: Use ModelDb.GetById<EncounterModel>(id) via reflection
        var encounter = TryModelDbGetById("EncounterModel", encounterId, diagnostics);
        if (encounter is not null)
        {
            diagnostics.Add($"Found encounter via ModelDb.GetById: {encounter.GetType().Name}");
            return encounter;
        }

        // Strategy 2: Search by type name in loaded assemblies
        var encounterByType = TryFindEncounterByTypeName(encounterId, diagnostics);
        if (encounterByType is not null)
        {
            diagnostics.Add($"Found encounter by type name: {encounterByType.GetType().Name}");
            return encounterByType;
        }

        // Strategy 3: Try to find in ModelDb as a generic model
        var modelByGenericLookup = TryModelDbGetById("Model", encounterId, diagnostics);
        if (modelByGenericLookup is not null)
        {
            diagnostics.Add($"Found model via generic lookup: {modelByGenericLookup.GetType().Name}");
            return modelByGenericLookup;
        }

        diagnostics.Add($"Could not resolve encounter: {encounterId}");
        return null;
    }

    private static object? TryModelDbGetById(string modelTypeName, string id, List<string> diagnostics)
    {
        try
        {
            var targetType = FindGameType(modelTypeName);
            if (targetType is null)
            {
                diagnostics.Add($"Could not find type '{modelTypeName}' in game assemblies");
                return null;
            }

            var modelDbType = typeof(ModelDb);

            // ModelDb.GetById<T> is generic, but the ID parameter type differs by game build.
            var methods = modelDbType.GetMethods(BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic);
            foreach (var method in methods)
            {
                if (!method.Name.Equals("GetById", StringComparison.Ordinal) || !method.IsGenericMethodDefinition)
                    continue;

                var genericParams = method.GetGenericArguments();
                if (genericParams.Length != 1)
                    continue;

                var parameters = method.GetParameters();
                if (parameters.Length != 1)
                    continue;

                try
                {
                    var genericMethod = method.MakeGenericMethod(targetType);
                    if (!TryBuildModelDbIdArgument(parameters[0].ParameterType, id, out var idArgument, out var conversionNote))
                    {
                        diagnostics.Add($"ModelDb.GetById<{modelTypeName}> skipped unsupported id parameter {parameters[0].ParameterType.FullName}");
                        continue;
                    }

                    if (!string.IsNullOrWhiteSpace(conversionNote))
                        diagnostics.Add(conversionNote);

                    var result = genericMethod.Invoke(null, new[] { idArgument });
                    if (result is not null)
                        return result;
                }
                catch (Exception ex)
                {
                    diagnostics.Add($"ModelDb.GetById<{modelTypeName}>({id}) threw: {ex.InnerException?.Message ?? ex.Message}");
                }
            }
        }
        catch (Exception ex)
        {
            diagnostics.Add($"ModelDb reflection failed: {ex.Message}");
        }

        return null;
    }

    private static bool TryBuildModelDbIdArgument(
        Type parameterType,
        string id,
        out object? argument,
        out string? conversionNote)
    {
        argument = null;
        conversionNote = null;

        if (parameterType == typeof(string) || parameterType == typeof(object))
        {
            argument = id;
            return true;
        }

        if (parameterType == typeof(ModelId))
        {
            if (!TrySplitModelId(id, out var category, out var entry))
                return false;

            argument = new ModelId(category, entry);
            conversionNote = $"ModelDb id '{id}' converted via ModelId(category='{category}', entry='{entry}')";
            return true;
        }

        try
        {
            var converter = TypeDescriptor.GetConverter(parameterType);
            if (converter.CanConvertFrom(typeof(string)))
            {
                argument = converter.ConvertFromInvariantString(id);
                if (argument is not null)
                {
                    conversionNote = $"ModelDb id '{id}' converted via TypeConverter -> {parameterType.Name}";
                    return true;
                }
            }
        }
        catch
        {
            // fall through
        }

        foreach (var methodName in new[] { "Parse", "FromString", "From", "Create" })
        {
            var factory = parameterType.GetMethod(
                methodName,
                BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic,
                null,
                new[] { typeof(string) },
                null);
            if (factory is null || !parameterType.IsAssignableFrom(factory.ReturnType))
                continue;

            argument = factory.Invoke(null, new object[] { id });
            if (argument is not null)
            {
                conversionNote = $"ModelDb id '{id}' converted via {parameterType.Name}.{methodName}(string)";
                return true;
            }
        }

        foreach (var methodName in new[] { "op_Implicit", "op_Explicit" })
        {
            var castMethod = parameterType.GetMethod(
                methodName,
                BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic,
                null,
                new[] { typeof(string) },
                null);
            if (castMethod is null || !parameterType.IsAssignableFrom(castMethod.ReturnType))
                continue;

            argument = castMethod.Invoke(null, new object[] { id });
            if (argument is not null)
            {
                conversionNote = $"ModelDb id '{id}' converted via {parameterType.Name}.{methodName}(string)";
                return true;
            }
        }

        var ctor = parameterType.GetConstructor(
            BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic,
            null,
            new[] { typeof(string) },
            null);
        if (ctor is not null)
        {
            argument = ctor.Invoke(new object[] { id });
            conversionNote = $"ModelDb id '{id}' converted via {parameterType.Name}(string)";
            return true;
        }

        try
        {
            argument = Convert.ChangeType(id, parameterType, CultureInfo.InvariantCulture);
            conversionNote = $"ModelDb id '{id}' converted via Convert.ChangeType -> {parameterType.Name}";
            return argument is not null;
        }
        catch
        {
            return false;
        }
    }

    private static object? TryFindEncounterByTypeName(string typeName, List<string> diagnostics)
    {
        try
        {
            var type = FindGameType(typeName);
            if (type is null)
            {
                diagnostics.Add($"Type '{typeName}' not found in game assemblies");
                return null;
            }

            var encounter = TryResolveEncounterFromType(type, diagnostics);
            if (encounter is not null)
                return encounter;
        }
        catch (Exception ex)
        {
            diagnostics.Add($"Type search failed: {ex.Message}");
        }

        return null;
    }

    private static object? TryResolveEncounterFromType(Type type, List<string> diagnostics)
    {
        try
        {
            foreach (var encounter in GetAllEncounterModels())
            {
                if (!type.IsInstanceOfType(encounter))
                    continue;

                diagnostics.Add($"Resolved {type.Name} via ModelDb.AllEncounters");
                return encounter;
            }

            var singleton = TryGetStaticEncounterInstance(type);
            if (singleton is not null)
            {
                diagnostics.Add($"Resolved {type.Name} via static singleton member");
                return singleton;
            }

            foreach (var candidateId in EnumerateEncounterIdCandidates(type))
            {
                var byId = TryModelDbGetById("EncounterModel", candidateId, diagnostics);
                if (byId is not null)
                {
                    diagnostics.Add($"Resolved {type.Name} via candidate id '{candidateId}'");
                    return byId;
                }
            }

            try
            {
                var ctor = type.GetConstructor(
                    BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic,
                    null,
                    Type.EmptyTypes,
                    null);
                if (ctor is not null)
                {
                    var instance = ctor.Invoke(null);
                    diagnostics.Add($"Created instance of {type.Name} via parameterless constructor");
                    return instance;
                }
            }
            catch (Exception ex)
            {
                diagnostics.Add($"Constructor invocation for {type.Name} failed: {ex.Message}");
            }
        }
        catch (Exception ex)
        {
            diagnostics.Add($"Encounter type resolution failed for {type.Name}: {ex.Message}");
        }

        return null;
    }

    private static bool TrySplitModelId(string rawId, out string category, out string entry)
    {
        category = string.Empty;
        entry = string.Empty;

        if (string.IsNullOrWhiteSpace(rawId))
            return false;

        var trimmed = rawId.Trim();
        var separatorIndex = trimmed.IndexOf('.');
        if (separatorIndex <= 0 || separatorIndex >= trimmed.Length - 1)
            return false;

        category = trimmed[..separatorIndex];
        entry = trimmed[(separatorIndex + 1)..];
        return category.Length > 0 && entry.Length > 0;
    }

    private static IEnumerable<EncounterModel> GetAllEncounterModels()
    {
        try
        {
            return ModelDb.AllEncounters ?? Enumerable.Empty<EncounterModel>();
        }
        catch
        {
            return Enumerable.Empty<EncounterModel>();
        }
    }

    private static object? TryGetStaticEncounterInstance(Type type)
    {
        var flags = BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic;

        foreach (var propertyName in new[] { "Instance", "Default", "Singleton", "Value" })
        {
            var prop = type.GetProperty(propertyName, flags);
            if (prop is null || prop.GetIndexParameters().Length != 0)
                continue;

            try
            {
                var value = prop.GetValue(null);
                if (value is not null && type.IsInstanceOfType(value))
                    return value;
            }
            catch
            {
                // best effort
            }
        }

        foreach (var fieldName in new[] { "Instance", "Default", "Singleton", "Value" })
        {
            var field = type.GetField(fieldName, flags);
            if (field is null)
                continue;

            try
            {
                var value = field.GetValue(null);
                if (value is not null && type.IsInstanceOfType(value))
                    return value;
            }
            catch
            {
                // best effort
            }
        }

        foreach (var prop in type.GetProperties(flags))
        {
            if (prop.GetIndexParameters().Length != 0 || !type.IsAssignableFrom(prop.PropertyType))
                continue;

            try
            {
                var value = prop.GetValue(null);
                if (value is not null)
                    return value;
            }
            catch
            {
                // best effort
            }
        }

        foreach (var field in type.GetFields(flags))
        {
            if (!type.IsAssignableFrom(field.FieldType))
                continue;

            try
            {
                var value = field.GetValue(null);
                if (value is not null)
                    return value;
            }
            catch
            {
                // best effort
            }
        }

        return null;
    }

    private static IEnumerable<string> EnumerateEncounterIdCandidates(Type type)
    {
        var seen = new HashSet<string>(StringComparer.Ordinal);

        foreach (var candidate in EnumerateEncounterIdCandidatesFromMembers(type))
        {
            if (seen.Add(candidate))
                yield return candidate;
        }

        var singleton = TryGetStaticEncounterInstance(type);
        var singletonId = TryGetModelId(singleton);
        if (!string.IsNullOrWhiteSpace(singletonId) && seen.Add(singletonId))
            yield return singletonId;

        if (seen.Add(type.Name))
            yield return type.Name;
    }

    private static IEnumerable<string> EnumerateEncounterIdCandidatesFromMembers(Type type)
    {
        var flags = BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic;

        foreach (var property in type.GetProperties(flags))
        {
            if (property.GetIndexParameters().Length != 0 || !LooksLikeEncounterIdMember(property.Name))
                continue;

            object? value;
            try
            {
                value = property.GetValue(null);
            }
            catch
            {
                continue;
            }

            if (TryNormalizeEncounterIdCandidate(value, out var candidate))
                yield return candidate;
        }

        foreach (var field in type.GetFields(flags))
        {
            if (!LooksLikeEncounterIdMember(field.Name))
                continue;

            object? value;
            try
            {
                value = field.GetValue(null);
            }
            catch
            {
                continue;
            }

            if (TryNormalizeEncounterIdCandidate(value, out var candidate))
                yield return candidate;
        }
    }

    private static bool LooksLikeEncounterIdMember(string memberName)
    {
        return memberName.Contains("Id", StringComparison.Ordinal) ||
               memberName.Contains("ID", StringComparison.Ordinal) ||
               memberName.Contains("ModelId", StringComparison.Ordinal);
    }

    private static bool TryNormalizeEncounterIdCandidate(object? value, out string candidate)
    {
        candidate = string.Empty;

        if (value is null)
            return false;

        if (value is string rawString)
        {
            candidate = rawString.Trim();
            return candidate.Length > 0;
        }

        if (value is ModelId modelId)
        {
            candidate = modelId.ToString();
            return candidate.Length > 0;
        }

        candidate = value.ToString()?.Trim() ?? string.Empty;
        return candidate.Length > 0;
    }

    private static Type? FindGameType(string typeName)
    {
        // Search in the game's core assemblies
        foreach (var asm in AppDomain.CurrentDomain.GetAssemblies())
        {
            try
            {
                var asmName = asm.GetName().Name ?? "";
                // Focus on game assemblies
                if (!asmName.Contains("sts2", StringComparison.OrdinalIgnoreCase) &&
                    !asmName.Contains("MegaCrit", StringComparison.OrdinalIgnoreCase))
                    continue;

                // Try exact match first
                var type = asm.GetType(typeName, throwOnError: false);
                if (type is not null) return type;

                // Try searching all types for name match
                foreach (var t in asm.GetTypes())
                {
                    if (t.Name.Equals(typeName, StringComparison.Ordinal) ||
                        t.FullName?.EndsWith($".{typeName}", StringComparison.Ordinal) == true)
                        return t;
                }
            }
            catch
            {
                // Skip assemblies that can't be reflected
            }
        }

        return null;
    }

    private static string? TryGetModelId(object? model)
    {
        if (model is null) return null;
        try
        {
            var idProp = model.GetType().GetProperty("Id",
                BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
            return idProp?.GetValue(model)?.ToString();
        }
        catch
        {
            return null;
        }
    }

    // -----------------------------------------------------------------------
    // Combat entry via RunManager.EnterRoomDebug
    // -----------------------------------------------------------------------

    private static bool TryEnterCombatViaDebugRoom(
        RunManager runManager,
        object encounter,
        List<string> diagnostics,
        out Task? pendingTask)
    {
        pendingTask = null;
        var context = CaptureContext();
        var hasUsableRunScene = context.RunState is not null &&
                                context.RunNode is not null &&
                                context.RunState.CurrentRoom is not null;
        diagnostics.Add(
            $"Combat sandbox scene check: run_state={(context.RunState is not null)}, run_node={(context.RunNode is not null)}, current_room={(context.RunState?.CurrentRoom is not null)}");

        if (hasUsableRunScene && TryInvokeEnterRoomDebug(runManager, encounter, diagnostics, out pendingTask))
        {
            diagnostics.Add("Entered combat via RunManager.EnterRoomDebug");
            return true;
        }

        diagnostics.Add("Usable run scene not available for EnterRoomDebug");

        diagnostics.Add("All combat entry strategies failed");
        return false;
    }

    private static bool TryCallEnterRoomDebug(RunManager runManager, object encounter, List<string> diagnostics)
    {
        return TryInvokeEnterRoomDebug(runManager, encounter, diagnostics, out _);
    }

    private static bool TryInvokeEnterRoomDebug(
        RunManager runManager,
        object encounter,
        List<string> diagnostics,
        out Task? pendingTask)
    {
        pendingTask = null;
        var roomEntryEncounter = PrepareEncounterForRoomEntry(encounter, diagnostics);
        if (roomEntryEncounter is null)
        {
            diagnostics.Add("Encounter could not be prepared for room entry");
            return false;
        }

        try
        {
            var rmType = runManager.GetType();
            // Prefer the current-game 4-param signature, then fall back to older variants.
            foreach (var paramCount in new[] { 4, 2, 1, 3 })
            {
                var method = FindStaticOrInstanceMethod(rmType, "EnterRoomDebug", paramCount);
                if (method is null) continue;

                var parameters = method.GetParameters();
                diagnostics.Add($"Found EnterRoomDebug({string.Join(", ", parameters.Select(p => p.ParameterType.Name))})");

                if (!TryBuildEnterRoomDebugArguments(parameters, roomEntryEncounter, out var args))
                {
                    diagnostics.Add($"Could not build EnterRoomDebug arguments for signature ({string.Join(", ", parameters.Select(p => p.ParameterType.Name))})");
                    continue;
                }

                var invokeResult = method.Invoke(method.IsStatic ? null : runManager, args);
                pendingTask = invokeResult as Task;
                diagnostics.Add("EnterRoomDebug invoked successfully");
                return true;
            }

            diagnostics.Add("EnterRoomDebug not found on RunManager");
        }
        catch (Exception ex)
        {
            diagnostics.Add($"EnterRoomDebug failed: {ex.InnerException?.Message ?? ex.Message}");
        }

        return false;
    }

    private static bool TryBuildEnterRoomDebugArguments(
        ParameterInfo[] parameters,
        object encounter,
        out object?[] args)
    {
        args = new object?[parameters.Length];
        var roomType = TryResolveEncounterRoomType(encounter) ?? RoomType.Monster;

        for (var i = 0; i < parameters.Length; i++)
        {
            var parameterType = parameters[i].ParameterType;

            if (parameterType.IsInstanceOfType(encounter) || parameterType.IsAssignableFrom(encounter.GetType()))
            {
                args[i] = encounter;
                continue;
            }

            if (parameterType == typeof(RoomType))
            {
                args[i] = roomType;
                continue;
            }

            if (parameterType == typeof(MapPointType))
            {
                args[i] = MapPointType.Unassigned;
                continue;
            }

            if (parameterType == typeof(bool))
            {
                args[i] = false;
                continue;
            }

            if (parameters[i].HasDefaultValue)
            {
                args[i] = parameters[i].DefaultValue;
                continue;
            }

            return false;
        }

        return true;
    }

    private static RoomType? TryResolveEncounterRoomType(object encounter)
    {
        if (encounter is EncounterModel typedEncounter)
        {
            return typedEncounter.RoomType;
        }

        try
        {
            var roomTypeProperty = encounter.GetType().GetProperty(
                "RoomType",
                BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
            var value = roomTypeProperty?.GetValue(encounter);
            if (value is RoomType typedRoomType)
            {
                return typedRoomType;
            }

            if (value is not null &&
                Enum.TryParse<RoomType>(value.ToString(), ignoreCase: true, out var parsedRoomType))
            {
                return parsedRoomType;
            }
        }
        catch
        {
            // Best-effort only; the caller falls back to Monster.
        }

        return null;
    }

    // -----------------------------------------------------------------------
    // Player state overrides
    // -----------------------------------------------------------------------

    private static void ApplyPostCombatSandboxOverrides(
        BridgeEnvCombatResetRequest request,
        List<string> diagnostics)
    {
        var context = CaptureContext();
        var player = GetPrimaryPlayer(context);
        var creature = player?.Creature;
        if (player is null || creature is null)
        {
            diagnostics.Add("Post-combat overrides skipped: no active player/creature");
            return;
        }

        if (request.MaxHp is > 0)
        {
            creature.SetMaxHpInternal(request.MaxHp.Value);
            diagnostics.Add($"Post-combat MaxHp set to {creature.MaxHp}");
        }

        if (request.CurrentHp is > 0)
        {
            creature.SetCurrentHpInternal(request.CurrentHp.Value);
            diagnostics.Add($"Post-combat CurrentHp set to {creature.CurrentHp}");
        }
        else
        {
            creature.SetCurrentHpInternal(creature.MaxHp);
            diagnostics.Add($"Post-combat CurrentHp restored to full ({creature.CurrentHp})");
        }

        if (request.MaxEnergy is > 0)
        {
            player.MaxEnergy = request.MaxEnergy.Value;
        }

        if (player.PlayerCombatState is not null)
        {
            player.PlayerCombatState.Energy = player.MaxEnergy;
            diagnostics.Add($"Post-combat Energy set to {player.MaxEnergy}");
        }
    }

    private static string TryApplyPlayerOverrides(
        BridgeEnvCombatResetRequest request,
        List<string> diagnostics,
        bool includeCombatOnlyOverrides)
    {
        var warnings = new List<string>();

        var context = CaptureContext();
        var player = GetPrimaryPlayer(context);
        if (player is null)
        {
            diagnostics.Add("No primary player found — skipping overrides");
            return "no_player";
        }

        var creature = player.Creature;

        // HP overrides
        if (!includeCombatOnlyOverrides && request.MaxHp is > 0 && creature is not null)
        {
            try
            {
                var maxHpProp = FindProperty(creature.GetType(), "MaxHp") ??
                                FindProperty(creature.GetType(), "BaseMaxHp");
                if (maxHpProp?.GetSetMethod(nonPublic: true) is not null)
                {
                    maxHpProp.SetValue(creature, request.MaxHp.Value);
                    diagnostics.Add($"Set MaxHp to {request.MaxHp.Value}");
                }
                else
                {
                    // Try field access
                    var field = FindField(creature.GetType(), "_maxHp") ??
                                FindField(creature.GetType(), "maxHp");
                    if (field is not null)
                    {
                        field.SetValue(creature, request.MaxHp.Value);
                        diagnostics.Add($"Set MaxHp via field to {request.MaxHp.Value}");
                    }
                    else
                    {
                        warnings.Add("MaxHp: no writable property or field found");
                    }
                }
            }
            catch (Exception ex)
            {
                warnings.Add($"MaxHp override failed: {ex.Message}");
            }
        }

        if (!includeCombatOnlyOverrides && creature is not null)
        {
            try
            {
                var targetHp = request.CurrentHp is > 0
                    ? request.CurrentHp.Value
                    : request.MaxHp is > 0
                        ? request.MaxHp.Value
                        : creature.MaxHp;

                var currentHpProp = FindProperty(creature.GetType(), "CurrentHp");
                if (currentHpProp?.GetSetMethod(nonPublic: true) is not null)
                {
                    currentHpProp.SetValue(creature, targetHp);
                    diagnostics.Add(request.CurrentHp is > 0
                        ? $"Set CurrentHp to {targetHp}"
                        : $"Restored CurrentHp to full ({targetHp})");
                }
                else
                {
                    if (creature.CurrentHp < targetHp)
                    {
                        creature.HealInternal((decimal)(targetHp - creature.CurrentHp));
                        diagnostics.Add(request.CurrentHp is > 0
                            ? $"Healed to {targetHp}"
                            : $"Healed to full ({targetHp})");
                    }
                    else if (creature.CurrentHp > targetHp)
                    {
                        var field = FindField(creature.GetType(), "_currentHp") ??
                                    FindField(creature.GetType(), "currentHp");
                        if (field is not null)
                        {
                            field.SetValue(creature, targetHp);
                            diagnostics.Add(request.CurrentHp is > 0
                                ? $"Set CurrentHp via field to {targetHp}"
                                : $"Reset CurrentHp via field to full ({targetHp})");
                        }
                        else
                        {
                            warnings.Add($"CurrentHp: cannot decrease from {creature.CurrentHp} to {targetHp}");
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                warnings.Add($"CurrentHp override failed: {ex.Message}");
            }
        }

        // Gold override
        if (!includeCombatOnlyOverrides && request.Gold is not null)
        {
            try
            {
                var goldProp = FindProperty(player.GetType(), "Gold");
                if (goldProp?.GetSetMethod(nonPublic: true) is not null)
                {
                    goldProp.SetValue(player, request.Gold.Value);
                    diagnostics.Add($"Set Gold to {request.Gold.Value}");
                }
                else
                {
                    var field = FindField(player.GetType(), "_gold") ??
                                FindField(player.GetType(), "gold");
                    if (field is not null)
                    {
                        field.SetValue(player, request.Gold.Value);
                        diagnostics.Add($"Set Gold via field to {request.Gold.Value}");
                    }
                    else
                    {
                        warnings.Add("Gold: no writable property or field found");
                    }
                }
            }
            catch (Exception ex)
            {
                warnings.Add($"Gold override failed: {ex.Message}");
            }
        }

        // Deck override
        var requestedDeckEntries = ResolveRequestedDeckEntries(request);
        if (!includeCombatOnlyOverrides && requestedDeckEntries.Count > 0)
        {
            try
            {
                var deck = player.Deck;
                if (deck is not null)
                {
                    var cards = GetHiddenFieldValue(deck, "_cards") ??
                                GetHiddenFieldValue(deck, "cards") ??
                                GetHiddenPropertyObjectValue(deck, "Cards");

                    if (cards is System.Collections.IList cardList)
                    {
                        cardList.Clear();
                        var added = 0;
                        foreach (var (cardId, upgradeLevel) in requestedDeckEntries)
                        {
                            var mutableCard = BuildMutableDeckCard(cardId, upgradeLevel, diagnostics);
                            if (mutableCard is null)
                            {
                                continue;
                            }

                            cardList.Add(mutableCard);
                            added++;
                        }
                        diagnostics.Add($"Deck override: added {added}/{requestedDeckEntries.Count} cards");
                    }
                    else
                    {
                        warnings.Add("Deck: could not access card list");
                    }
                }
            }
            catch (Exception ex)
            {
                warnings.Add($"Deck override failed: {ex.Message}");
            }
        }

        // Relic override
        if (!includeCombatOnlyOverrides && request.Relics is { Length: > 0 })
        {
            try
            {
                var relics = GetHiddenFieldValue(player, "_relics") ??
                             GetHiddenPropertyObjectValue(player, "Relics");

                if (relics is System.Collections.IList relicList)
                {
                    relicList.Clear();
                    var added = 0;
                    foreach (var relicId in request.Relics)
                    {
                        var relic = TryModelDbGetById("RelicModel", relicId, diagnostics);
                        if (relic is null)
                        {
                            warnings.Add($"Relics: relic '{relicId}' not found");
                            continue;
                        }

                        var mutableRelic = TryCallToMutable(relic);
                        relicList.Add(mutableRelic ?? relic);
                        added++;
                    }
                    diagnostics.Add($"Relic override: added {added}/{request.Relics.Length} relics");
                }
                else
                {
                    warnings.Add("Relics: could not access relic list");
                }
            }
            catch (Exception ex)
            {
                warnings.Add($"Relic override failed: {ex.Message}");
            }
        }

        // Potion override
        if (!includeCombatOnlyOverrides && request.Potions is { Length: > 0 })
        {
            try
            {
                var potionSlots = GetHiddenFieldValue(player, "_potionSlots") ??
                                  GetHiddenPropertyObjectValue(player, "PotionSlots");

                if (potionSlots is System.Collections.IList potionList)
                {
                    // Clear existing
                    for (var i = 0; i < potionList.Count; i++)
                        potionList[i] = null;

                    var added = 0;
                    for (var i = 0; i < Math.Min(request.Potions.Length, potionList.Count); i++)
                    {
                        var potionId = request.Potions[i];
                        if (string.IsNullOrWhiteSpace(potionId)) continue;

                        var potion = TryModelDbGetById("PotionModel", potionId, diagnostics);
                        if (potion is null)
                        {
                            warnings.Add($"Potions: potion '{potionId}' not found");
                            continue;
                        }

                        var mutablePotion = TryCallToMutable(potion);
                        potionList[i] = mutablePotion ?? potion;
                        added++;
                    }
                    diagnostics.Add($"Potion override: added {added}/{request.Potions.Length} potions");
                }
                else
                {
                    warnings.Add("Potions: could not access potion slots");
                }
            }
            catch (Exception ex)
            {
                warnings.Add($"Potion override failed: {ex.Message}");
            }
        }

        // Energy override
        if (includeCombatOnlyOverrides && request.MaxEnergy is > 0)
        {
            try
            {
                var pcs = player.PlayerCombatState;
                if (pcs is not null)
                {
                    var energyProp = FindProperty(pcs.GetType(), "Energy") ??
                                     FindProperty(pcs.GetType(), "BaseEnergy") ??
                                     FindProperty(pcs.GetType(), "MaxEnergy");
                    if (energyProp?.GetSetMethod(nonPublic: true) is not null)
                    {
                        energyProp.SetValue(pcs, request.MaxEnergy.Value);
                        diagnostics.Add($"Set Energy to {request.MaxEnergy.Value}");
                    }
                    else
                    {
                        warnings.Add("MaxEnergy: no writable property found on PlayerCombatState");
                    }
                }
                else
                {
                    warnings.Add("MaxEnergy: PlayerCombatState is null");
                }
            }
            catch (Exception ex)
            {
                warnings.Add($"MaxEnergy override failed: {ex.Message}");
            }
        }

        return warnings.Count > 0 ? string.Join("; ", warnings) : "";
    }

    private static object? TryCallToMutable(object model)
    {
        try
        {
            var method = model.GetType().GetMethod("ToMutable",
                BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
            if (method is not null)
                return method.Invoke(model, Array.Empty<object>());
        }
        catch { /* best effort */ }
        return null;
    }

    // -----------------------------------------------------------------------
    // Encounter catalog
    // -----------------------------------------------------------------------

    private static object[] ListAvailableCombatEncounters()
    {
        var results = new List<EncounterCatalogEntry>();
        var seen = new HashSet<string>(StringComparer.Ordinal);

        try
        {
            foreach (var encounter in GetAllEncounterModels())
            {
                try
                {
                    var encounterId = TryGetModelId(encounter);
                    if (string.IsNullOrWhiteSpace(encounterId) || !seen.Add(encounterId))
                        continue;

                    var typeName = encounter.GetType().Name;
                    var category = CategorizeEncounterType(typeName);
                    var isMock = typeName.Contains("Mock", StringComparison.OrdinalIgnoreCase) ||
                                 typeName.Contains("Test", StringComparison.OrdinalIgnoreCase) ||
                                 typeName.Contains("Dummy", StringComparison.OrdinalIgnoreCase);

                    results.Add(new EncounterCatalogEntry
                    {
                        EncounterId = encounterId,
                        DisplayName = encounterId,
                        TypeName = typeName,
                        Category = category,
                        IsMock = isMock
                    });
                }
                catch { /* skip entries that fail reflection */ }
            }
        }
        catch { /* best effort */ }

        return results
            .OrderBy(e => e.Category)
            .ThenBy(e => e.EncounterId)
            .Select(e => new
            {
                encounter_id = e.EncounterId,
                display_name = e.DisplayName,
                type_name = e.TypeName,
                category = e.Category,
                is_mock = e.IsMock
            })
            .ToArray<object>();
    }

    private static string CategorizeEncounterType(string typeName)
    {
        if (typeName.Contains("Mock", StringComparison.OrdinalIgnoreCase) ||
            typeName.Contains("Test", StringComparison.OrdinalIgnoreCase) ||
            typeName.Contains("Dummy", StringComparison.OrdinalIgnoreCase))
            return "test";

        if (typeName.Contains("Boss", StringComparison.OrdinalIgnoreCase))
            return "boss";

        if (typeName.Contains("Elite", StringComparison.OrdinalIgnoreCase))
            return "elite";

        if (typeName.Contains("Event", StringComparison.OrdinalIgnoreCase))
            return "event";

        return "normal";
    }

    // -----------------------------------------------------------------------
    // Reflection helpers (static method support)
    // -----------------------------------------------------------------------

    private static MethodInfo? FindStaticOrInstanceMethod(Type? type, string methodName, int parameterCount)
    {
        while (type is not null)
        {
            var method = type
                .GetMethods(BindingFlags.Instance | BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.DeclaredOnly)
                .FirstOrDefault(candidate =>
                    candidate.Name.Equals(methodName, StringComparison.Ordinal) &&
                    candidate.GetParameters().Length == parameterCount);

            if (method is not null)
            {
                return method;
            }

            type = type.BaseType;
        }

        return null;
    }
}
