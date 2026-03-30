using System.Globalization;
using System.Net;
using System.Reflection;
using System.Text.Json.Serialization;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Runs;

namespace Sts2McpBridge.Scripts;

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
    private const int DefaultCombatResetTimeoutMs = 15000;

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

        // Step 1: Resolve encounter and enter combat on the main thread
        var setupResult = await BridgeCoordinator.RunOnMainThreadAsync(() =>
        {
            return SetUpCombatSandbox(request, encounterId, diagnostics);
        });

        if (!setupResult.Success)
        {
            throw new BridgeRequestException(
                HttpStatusCode.Conflict,
                setupResult.ErrorCode ?? "combat_sandbox_setup_failed",
                setupResult.ErrorMessage ?? "Failed to set up combat sandbox.",
                new { diagnostics, encounter_id = encounterId });
        }

        // Step 2: Let Godot settle the scene
        await BridgeCoordinator.WaitForPumpTicksAsync(8, cancellationToken);

        // Step 3: Wait for stable actionable combat state
        var state = await WaitForStableEnvStateAsync(
            null,
            timeoutMs,
            requireActionableOrDone: true,
            cancellationToken);

        // Step 4: Verify we landed in combat
        if (state.Phase != "combat" && !state.CombatInProgress)
        {
            // Maybe still settling — wait a bit more
            await BridgeCoordinator.WaitForPumpTicksAsync(5, cancellationToken);
            state = await WaitForStableEnvStateAsync(
                null,
                Math.Min(timeoutMs, 5000),
                requireActionableOrDone: true,
                cancellationToken);
        }

        if (!state.CombatInProgress && !state.Done)
        {
            throw new BridgeRequestException(
                HttpStatusCode.Conflict,
                "combat_sandbox_not_in_combat",
                $"Combat sandbox setup completed but the game is not in combat. Phase: {state.Phase}",
                new { phase = state.Phase, actionable = state.Actionable, diagnostics });
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

        var executedActions = new List<object>
        {
            new { action = "combat_sandbox_setup", encounter_id = encounterId, diagnostics }
        };

        return BuildEnvResetPayload(episode, state, executedActions);
    }

    // -----------------------------------------------------------------------
    // GET /env/combat_catalog
    // -----------------------------------------------------------------------

    public static async Task<object> GetCombatCatalogResponseAsync(CancellationToken cancellationToken)
    {
        await WaitForEnvDispatcherReadyAsync(5000, cancellationToken);

        var encounters = await BridgeCoordinator.RunOnMainThreadAsync(() =>
        {
            return ListAvailableCombatEncounters();
        });

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
        // Player died
        if (snapshot.CurrentHp <= 0)
            return true;

        // Combat ended (victory or retreat) — no longer in combat and not just settling
        if (!snapshot.CombatInProgress && snapshot.Phase != "settling")
            return true;

        return false;
    }

    // -----------------------------------------------------------------------
    // Setup internals
    // -----------------------------------------------------------------------

    private sealed class CombatSandboxSetupResult
    {
        public bool Success { get; init; }
        public string? ErrorCode { get; init; }
        public string? ErrorMessage { get; init; }
    }

    private static CombatSandboxSetupResult SetUpCombatSandbox(
        BridgeEnvCombatResetRequest request,
        string encounterId,
        List<string> diagnostics)
    {
        try
        {
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

            // 3. Try to create a test run state and enter combat
            var entered = TryEnterCombatViaDebugRoom(runManager, encounter, request, diagnostics);
            if (!entered)
            {
                return new CombatSandboxSetupResult
                {
                    Success = false,
                    ErrorCode = "combat_entry_failed",
                    ErrorMessage = "Could not enter combat room. Check diagnostics for reflection probe results."
                };
            }

            // 4. Apply player state overrides if requested
            var overrideResult = TryApplyPlayerOverrides(request, diagnostics);
            if (!string.IsNullOrEmpty(overrideResult))
            {
                diagnostics.Add($"Player override warnings: {overrideResult}");
            }

            return new CombatSandboxSetupResult { Success = true };
        }
        catch (Exception ex)
        {
            diagnostics.Add($"Exception during setup: {ex.GetType().Name}: {ex.Message}");
            return new CombatSandboxSetupResult
            {
                Success = false,
                ErrorCode = "combat_sandbox_exception",
                ErrorMessage = ex.Message
            };
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
            var modelDbType = typeof(ModelDb);

            // Try ModelDb.GetById<T>(string) — it's a generic static method
            var methods = modelDbType.GetMethods(BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic);
            foreach (var method in methods)
            {
                if (!method.Name.Equals("GetById", StringComparison.Ordinal) || !method.IsGenericMethodDefinition)
                    continue;

                var genericParams = method.GetGenericArguments();
                if (genericParams.Length != 1)
                    continue;

                // Find the EncounterModel type
                var targetType = FindGameType(modelTypeName);
                if (targetType is null)
                {
                    diagnostics.Add($"Could not find type '{modelTypeName}' in game assemblies");
                    continue;
                }

                try
                {
                    var genericMethod = method.MakeGenericMethod(targetType);
                    var result = genericMethod.Invoke(null, new object[] { id });
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

            // Try to get a singleton or create an instance
            // Check for Instance property first
            var instanceProp = type.GetProperty("Instance", BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic);
            if (instanceProp is not null)
            {
                var instance = instanceProp.GetValue(null);
                if (instance is not null)
                    return instance;
            }

            // Try ModelDb.GetById with the type's Name as ID
            var modelDbResult = TryModelDbGetById("EncounterModel", type.Name, diagnostics);
            if (modelDbResult is not null)
                return modelDbResult;

            // Try parameterless constructor
            try
            {
                var ctor = type.GetConstructor(
                    BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic,
                    null, Type.EmptyTypes, null);
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
            diagnostics.Add($"Type search failed: {ex.Message}");
        }

        return null;
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
        BridgeEnvCombatResetRequest request,
        List<string> diagnostics)
    {
        // Strategy 1: RunManager.EnterRoomDebug(encounter)
        if (TryCallEnterRoomDebug(runManager, encounter, diagnostics))
        {
            diagnostics.Add("Entered combat via RunManager.EnterRoomDebug");
            return true;
        }

        // Strategy 2: RunManager.EnterRoomDebug(encounter, runState)
        // Try with 2 params
        var runState = TryGetRunState(runManager);
        if (runState is not null && TryCallEnterRoomDebugWithState(runManager, encounter, runState, diagnostics))
        {
            diagnostics.Add("Entered combat via RunManager.EnterRoomDebug (2-param)");
            return true;
        }

        // Strategy 3: Try SetUpNewSinglePlayer first, then EnterRoomDebug
        if (TrySetUpTestRun(runManager, request, diagnostics))
        {
            if (TryCallEnterRoomDebug(runManager, encounter, diagnostics))
            {
                diagnostics.Add("Entered combat via test run + EnterRoomDebug");
                return true;
            }
        }

        diagnostics.Add("All combat entry strategies failed");
        return false;
    }

    private static bool TryCallEnterRoomDebug(RunManager runManager, object encounter, List<string> diagnostics)
    {
        try
        {
            var rmType = runManager.GetType();
            // Search for EnterRoomDebug with various parameter counts
            for (var paramCount = 1; paramCount <= 3; paramCount++)
            {
                var method = FindStaticOrInstanceMethod(rmType, "EnterRoomDebug", paramCount);
                if (method is null) continue;

                var parameters = method.GetParameters();
                diagnostics.Add($"Found EnterRoomDebug({string.Join(", ", parameters.Select(p => p.ParameterType.Name))})");

                var args = new object?[paramCount];
                args[0] = encounter;
                // Fill remaining with null/default
                for (var i = 1; i < paramCount; i++)
                {
                    args[i] = parameters[i].HasDefaultValue ? parameters[i].DefaultValue : null;
                }

                method.Invoke(method.IsStatic ? null : runManager, args);
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

    private static bool TryCallEnterRoomDebugWithState(
        RunManager runManager, object encounter, RunState runState, List<string> diagnostics)
    {
        try
        {
            var rmType = runManager.GetType();
            var method = FindStaticOrInstanceMethod(rmType, "EnterRoomDebug", 2);
            if (method is null)
            {
                diagnostics.Add("EnterRoomDebug(2 params) not found");
                return false;
            }

            method.Invoke(method.IsStatic ? null : runManager, new object[] { encounter, runState });
            diagnostics.Add("EnterRoomDebug(encounter, runState) invoked successfully");
            return true;
        }
        catch (Exception ex)
        {
            diagnostics.Add($"EnterRoomDebug(2 params) failed: {ex.InnerException?.Message ?? ex.Message}");
            return false;
        }
    }

    private static bool TrySetUpTestRun(
        RunManager runManager, BridgeEnvCombatResetRequest request, List<string> diagnostics)
    {
        // Strategy A: RunState.CreateForTest(...)
        try
        {
            var rsType = typeof(RunState);
            for (var paramCount = 0; paramCount <= 4; paramCount++)
            {
                var method = rsType.GetMethods(
                        BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic)
                    .FirstOrDefault(m =>
                        m.Name.Equals("CreateForTest", StringComparison.Ordinal) &&
                        m.GetParameters().Length == paramCount);

                if (method is null) continue;

                var parameters = method.GetParameters();
                diagnostics.Add($"Found RunState.CreateForTest({string.Join(", ", parameters.Select(p => $"{p.ParameterType.Name} {p.Name}"))})");

                var args = new object?[paramCount];
                for (var i = 0; i < paramCount; i++)
                {
                    args[i] = parameters[i].HasDefaultValue ? parameters[i].DefaultValue : null;
                }

                var testRunState = method.Invoke(null, args) as RunState;
                if (testRunState is not null)
                {
                    diagnostics.Add("Created test RunState via CreateForTest");
                    // Now try to wire it up
                    return TrySetUpSinglePlayer(runManager, testRunState, diagnostics);
                }
            }
        }
        catch (Exception ex)
        {
            diagnostics.Add($"CreateForTest failed: {ex.InnerException?.Message ?? ex.Message}");
        }

        // Strategy B: SetUpNewSinglePlayer
        try
        {
            var rmType = runManager.GetType();
            for (var paramCount = 0; paramCount <= 3; paramCount++)
            {
                var method = FindStaticOrInstanceMethod(rmType, "SetUpNewSinglePlayer", paramCount);
                if (method is null) continue;

                var parameters = method.GetParameters();
                diagnostics.Add($"Found SetUpNewSinglePlayer({string.Join(", ", parameters.Select(p => $"{p.ParameterType.Name} {p.Name}"))})");

                var args = new object?[paramCount];
                for (var i = 0; i < paramCount; i++)
                {
                    args[i] = parameters[i].HasDefaultValue ? parameters[i].DefaultValue : null;
                }

                method.Invoke(method.IsStatic ? null : runManager, args);
                diagnostics.Add("SetUpNewSinglePlayer invoked successfully");
                return true;
            }
        }
        catch (Exception ex)
        {
            diagnostics.Add($"SetUpNewSinglePlayer failed: {ex.InnerException?.Message ?? ex.Message}");
        }

        diagnostics.Add("Could not set up test run (CreateForTest and SetUpNewSinglePlayer both failed)");
        return false;
    }

    private static bool TrySetUpSinglePlayer(RunManager runManager, RunState testRunState, List<string> diagnostics)
    {
        try
        {
            var rmType = runManager.GetType();
            var method = FindStaticOrInstanceMethod(rmType, "SetUpNewSinglePlayer", 1);
            if (method is not null)
            {
                method.Invoke(method.IsStatic ? null : runManager, new object[] { testRunState });
                diagnostics.Add("SetUpNewSinglePlayer(runState) invoked successfully");
                return true;
            }

            // Try with 0 args
            method = FindStaticOrInstanceMethod(rmType, "SetUpNewSinglePlayer", 0);
            if (method is not null)
            {
                method.Invoke(method.IsStatic ? null : runManager, Array.Empty<object>());
                diagnostics.Add("SetUpNewSinglePlayer() invoked successfully");
                return true;
            }

            diagnostics.Add("SetUpNewSinglePlayer not found on RunManager");
        }
        catch (Exception ex)
        {
            diagnostics.Add($"SetUpNewSinglePlayer failed: {ex.InnerException?.Message ?? ex.Message}");
        }

        return false;
    }

    // -----------------------------------------------------------------------
    // Player state overrides
    // -----------------------------------------------------------------------

    private static string TryApplyPlayerOverrides(BridgeEnvCombatResetRequest request, List<string> diagnostics)
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
        if (request.MaxHp is > 0 && creature is not null)
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

        if (request.CurrentHp is > 0 && creature is not null)
        {
            try
            {
                var currentHpProp = FindProperty(creature.GetType(), "CurrentHp");
                if (currentHpProp?.GetSetMethod(nonPublic: true) is not null)
                {
                    currentHpProp.SetValue(creature, request.CurrentHp.Value);
                    diagnostics.Add($"Set CurrentHp to {request.CurrentHp.Value}");
                }
                else
                {
                    // Try healing to target
                    var targetHp = request.CurrentHp.Value;
                    if (creature.CurrentHp < targetHp)
                    {
                        creature.HealInternal((decimal)(targetHp - creature.CurrentHp));
                        diagnostics.Add($"Healed to {targetHp}");
                    }
                    else if (creature.CurrentHp > targetHp)
                    {
                        // Try DamageInternal or direct field
                        var field = FindField(creature.GetType(), "_currentHp") ??
                                    FindField(creature.GetType(), "currentHp");
                        if (field is not null)
                        {
                            field.SetValue(creature, targetHp);
                            diagnostics.Add($"Set CurrentHp via field to {targetHp}");
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
        if (request.Gold is not null)
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
        if (request.Deck is { Length: > 0 })
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
                        foreach (var cardId in request.Deck)
                        {
                            var card = TryModelDbGetById("CardModel", cardId, diagnostics);
                            if (card is null)
                            {
                                warnings.Add($"Deck: card '{cardId}' not found");
                                continue;
                            }

                            // Try ToMutable
                            var mutableCard = TryCallToMutable(card);
                            cardList.Add(mutableCard ?? card);
                            added++;
                        }
                        diagnostics.Add($"Deck override: added {added}/{request.Deck.Length} cards");
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
        if (request.Relics is { Length: > 0 })
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
        if (request.Potions is { Length: > 0 })
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
        if (request.MaxEnergy is > 0)
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
        var results = new List<object>();
        var seen = new HashSet<string>(StringComparer.Ordinal);

        try
        {
            // Search all game assemblies for EncounterModel subclasses
            foreach (var asm in AppDomain.CurrentDomain.GetAssemblies())
            {
                try
                {
                    var asmName = asm.GetName().Name ?? "";
                    if (!asmName.Contains("sts2", StringComparison.OrdinalIgnoreCase) &&
                        !asmName.Contains("MegaCrit", StringComparison.OrdinalIgnoreCase))
                        continue;

                    var encounterModelType = FindGameType("EncounterModel");
                    if (encounterModelType is null) continue;

                    foreach (var type in asm.GetTypes())
                    {
                        try
                        {
                            if (!encounterModelType.IsAssignableFrom(type) || type.IsAbstract || type.IsInterface)
                                continue;

                            var typeName = type.Name;
                            if (!seen.Add(typeName)) continue;

                            var category = CategorizeEncounterType(typeName);
                            var isMock = typeName.Contains("Mock", StringComparison.OrdinalIgnoreCase) ||
                                         typeName.Contains("Test", StringComparison.OrdinalIgnoreCase) ||
                                         typeName.Contains("Dummy", StringComparison.OrdinalIgnoreCase);

                            results.Add(new
                            {
                                encounter_id = typeName,
                                display_name = typeName,
                                category,
                                is_mock = isMock
                            });
                        }
                        catch { /* skip types that fail reflection */ }
                    }
                }
                catch { /* skip assemblies that fail reflection */ }
            }
        }
        catch { /* best effort */ }

        return results.OrderBy(e => ((dynamic)e).category).ThenBy(e => ((dynamic)e).encounter_id).ToArray();
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
