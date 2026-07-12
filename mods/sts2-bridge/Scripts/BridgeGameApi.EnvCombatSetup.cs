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

internal static partial class BridgeGameApi
{
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
                        foreach (var entry in requestedDeckEntries)
                        {
                            var mutableCard = BuildMutableDeckCard(entry.CardId, entry.UpgradeLevel, diagnostics);
                            if (mutableCard is null)
                            {
                                continue;
                            }

                            cardList.Add(mutableCard);
                            // Modifiers applied after the card lands in the
                            // deck collection so any modifier on-attach hook
                            // sees a coherent owner context.
                            if (entry.Enchantments.Count > 0 || entry.Afflictions.Count > 0)
                            {
                                ApplyDeckCardModifiers(mutableCard, entry, diagnostics);
                            }
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

    private static void DrainManagedFinalizersAfterReset(List<string> diagnostics)
    {
        var summary = DrainManagedFinalizersInternal();
        diagnostics.Add(summary);
    }

    /// <summary>
    /// Shared GC drain used by every quiescent boundary (combat_sandbox reset,
    /// full_run env/reset, combat→post-combat screen transition). Cheap when
    /// the finalize queue is empty; expensive when it isn't — never call from
    /// a hot mid-turn path. Logs to BridgeDebugTrace for non-sandbox call sites
    /// that don't carry a diagnostics list.
    /// </summary>
    internal static string DrainManagedFinalizersInternal()
    {
        try
        {
            var sw = Stopwatch.StartNew();
            var memBefore = GC.GetTotalMemory(forceFullCollection: false);
            var gen2Before = GC.CollectionCount(2);

            GC.Collect(2, GCCollectionMode.Forced, blocking: true, compacting: true);
            GC.WaitForPendingFinalizers();
            GC.Collect(2, GCCollectionMode.Forced, blocking: true, compacting: false);

            sw.Stop();
            var memAfter = GC.GetTotalMemory(forceFullCollection: false);
            return
                $"gc_drain: elapsed_ms={sw.Elapsed.TotalMilliseconds:F1} " +
                $"reclaimed_kb={(memBefore - memAfter) / 1024} " +
                $"gen2_delta={GC.CollectionCount(2) - gen2Before}";
        }
        catch (Exception ex)
        {
            return $"gc_drain_failed: {ex.GetBaseException().Message}";
        }
    }

    internal static void DrainManagedFinalizersLogged(string callSite)
    {
        var summary = DrainManagedFinalizersInternal();
        BridgeDebugTrace.Write($"[{callSite}] {summary}");
    }
}
