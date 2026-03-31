using System.Reflection;
using Godot;
using HarmonyLib;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Entities.Merchant;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Nodes;
using MegaCrit.Sts2.Core.Nodes.Cards.Holders;
using MegaCrit.Sts2.Core.Nodes.Events;
using MegaCrit.Sts2.Core.Nodes.Rooms;
using MegaCrit.Sts2.Core.Nodes.Screens.CardSelection;
using MegaCrit.Sts2.Core.Nodes.Screens.Shops;
using MegaCrit.Sts2.Core.Runs;

namespace Sts2DraftTracker.Scripts;

/// <summary>
/// Harmony patches for intercepting game decisions.
/// Uses both lifecycle hooks and state-change detection.
/// </summary>
internal static class TrackerPatches
{
    // -----------------------------------------------------------------------
    // Frame tick — poll state transitions
    // -----------------------------------------------------------------------

    [HarmonyPatch(typeof(NRun), nameof(NRun._Process))]
    private static class NRunProcessPatch
    {
        [HarmonyPostfix]
        private static void Postfix()
        {
            DraftTracker.OnFrameTick();
        }
    }

    // -----------------------------------------------------------------------
    // Run lifecycle — detect run start
    // -----------------------------------------------------------------------

    [HarmonyPatch(typeof(NGame), nameof(NGame._Ready))]
    private static class NGameReadyPatch
    {
        [HarmonyPostfix]
        private static void Postfix()
        {
            // Subscribe to RunManager events if available
            TrySubscribeRunEvents();
        }
    }

    private static bool _subscribed;

    private static void TrySubscribeRunEvents()
    {
        if (_subscribed) return;
        try
        {
            var rm = RunManager.Instance;
            if (rm is null) return;

            // Try native RunStarted event
            var runStartedEvent = rm.GetType().GetEvent("RunStarted",
                BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
            if (runStartedEvent is not null)
            {
                var handler = Delegate.CreateDelegate(runStartedEvent.EventHandlerType!,
                    typeof(TrackerPatches).GetMethod(nameof(OnRunStartedHandler),
                        BindingFlags.Static | BindingFlags.NonPublic)!);
                runStartedEvent.AddEventHandler(rm, handler);
                _subscribed = true;
                Log.Info($"[{TrackerRuntime.ModId}] Subscribed to RunManager.RunStarted event");
                return;
            }
        }
        catch (Exception ex)
        {
            Log.Warn($"[{TrackerRuntime.ModId}] Could not subscribe to RunStarted: {ex.Message}");
        }

        // Fallback: will detect run start via NRun._Ready patch
        _subscribed = true;
    }

    private static void OnRunStartedHandler()
    {
        DraftTracker.OnRunStarted();
    }

    // Fallback run detection via NRun._Ready
    [HarmonyPatch(typeof(NRun), nameof(NRun._Ready))]
    private static class NRunReadyPatch
    {
        [HarmonyPostfix]
        private static void Postfix()
        {
            DraftTracker.OnRunStarted();
        }
    }

    // -----------------------------------------------------------------------
    // Combat end — detect victory/death
    // -----------------------------------------------------------------------

    [HarmonyPatch(typeof(CombatManager), "EndCombatInternal")]
    private static class CombatEndPatch
    {
        [HarmonyPostfix]
        private static void Postfix()
        {
            // Combat ended — check if game over
            try
            {
                var runState = RunManager.Instance?.DebugOnlyGetState();
                if (runState?.IsGameOver == true)
                {
                    var player = runState.Players.Count > 0 ? runState.Players[0] : null;
                    var alive = (player?.Creature?.CurrentHp ?? 0) > 0;
                    DraftTracker.OnRunEnded(alive);
                }
            }
            catch { /* non-critical */ }
        }
    }

    // Also catch LoseCombat
    [HarmonyPatch(typeof(CombatManager), nameof(CombatManager.LoseCombat))]
    private static class CombatLosePatch
    {
        [HarmonyPostfix]
        private static void Postfix()
        {
            try
            {
                DraftTracker.OnRunEnded(victory: false);
            }
            catch { /* non-critical */ }
        }
    }

    // -----------------------------------------------------------------------
    // Shop purchase
    // -----------------------------------------------------------------------

    [HarmonyPatch]
    private static class MerchantSlotPurchasePatch
    {
        private static IEnumerable<MethodBase> TargetMethods()
        {
            foreach (var method in new[]
                     {
                         AccessTools.Method(typeof(NMerchantCard), "OnTryPurchase", new[] { typeof(MerchantInventory) }),
                         AccessTools.Method(typeof(NMerchantRelic), "OnTryPurchase", new[] { typeof(MerchantInventory) }),
                         AccessTools.Method(typeof(NMerchantPotion), "OnTryPurchase", new[] { typeof(MerchantInventory) }),
                         AccessTools.Method(typeof(NMerchantCardRemoval), "OnTryPurchase", new[] { typeof(MerchantInventory) })
                     })
            {
                if (method is not null)
                {
                    yield return method;
                }
            }
        }

        [HarmonyPostfix]
        private static void Postfix(NMerchantSlot __instance)
        {
            try
            {
                var entry = __instance.Entry;
                if (entry is null) return;

                string itemKind;
                string itemName;

                if (entry is MerchantCardEntry cardEntry)
                {
                    itemKind = "card";
                    itemName = cardEntry.CreationResult?.Card?.Title?.ToString() ?? "?";
                }
                else if (entry is MerchantRelicEntry relicEntry)
                {
                    itemKind = "relic";
                    itemName = relicEntry.Model?.Title?.ToString() ?? "?";
                }
                else if (entry is MerchantPotionEntry potionEntry)
                {
                    itemKind = "potion";
                    itemName = potionEntry.Model?.Title?.ToString() ?? "?";
                }
                else if (entry is MerchantCardRemovalEntry)
                {
                    itemKind = "removal";
                    itemName = "[Card Removal]";
                }
                else
                {
                    return; // Unknown entry type
                }

                DraftTracker.OnShopPurchase(itemKind, itemName, entry.Cost);
            }
            catch (Exception ex)
            {
                Log.Warn($"[{TrackerRuntime.ModId}] Shop purchase tracking error: {ex.Message}");
            }
        }
    }

    // -----------------------------------------------------------------------
    // Event option chosen
    // -----------------------------------------------------------------------

    [HarmonyPatch(typeof(NEventOptionButton), "OnRelease")]
    private static class EventOptionPatch
    {
        [HarmonyPostfix]
        private static void Postfix(NEventOptionButton __instance)
        {
            try
            {
                var option = __instance.Option;
                if (option is null) return;

                var optionText = option.Title?.ToString() ?? "?";

                // Try to get the parent event's name
                var eventName = "?";
                try
                {
                    var eventRoom = NEventRoom.Instance;
                    if (eventRoom is not null)
                    {
                        var eventField = eventRoom.GetType().GetField("_event",
                            BindingFlags.Instance | BindingFlags.NonPublic);
                        var eventModel = eventField?.GetValue(eventRoom);
                        if (eventModel is not null)
                        {
                            var titleProp = eventModel.GetType().GetProperty("Title",
                                BindingFlags.Instance | BindingFlags.Public);
                            eventName = titleProp?.GetValue(eventModel)?.ToString() ?? "?";
                        }
                    }
                }
                catch { /* best effort */ }

                DraftTracker.OnEventOptionChosen(eventName, optionText);
            }
            catch (Exception ex)
            {
                Log.Warn($"[{TrackerRuntime.ModId}] Event option tracking error: {ex.Message}");
            }
        }
    }
}
