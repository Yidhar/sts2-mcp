using System.Reflection;
using System.Text.Encodings.Web;
using System.Text.Json;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Merchant;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes;
using MegaCrit.Sts2.Core.Nodes.Cards.Holders;
using MegaCrit.Sts2.Core.Nodes.Rooms;
using MegaCrit.Sts2.Core.Nodes.Screens.CardSelection;
using MegaCrit.Sts2.Core.Nodes.Screens.Shops;
using MegaCrit.Sts2.Core.Runs;

namespace Sts2DraftTracker.Scripts;

/// <summary>
/// Core tracking logic. Accumulates draft events during a run and
/// saves to JSON when the run ends.
/// Uses a combination of state-diff detection (comparing deck snapshots)
/// and Harmony patches (for capturing what was offered but not chosen).
/// </summary>
internal static class DraftTracker
{
    private static RunLog? _currentRun;
    private static readonly object Sync = new();
    private static readonly JsonSerializerOptions JsonOpts = new()
    {
        WriteIndented = true,
        Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping,
        DefaultIgnoreCondition = System.Text.Json.Serialization.JsonIgnoreCondition.WhenWritingNull
    };

    // State tracking for diffing
    private static List<string>? _lastDeckSnapshot;
    private static bool _wasInCardReward;
    private static bool _wasInShop;
    private static bool _wasInUpgrade;
    private static List<CardInfo>? _pendingCardRewardOffered;
    private static ShopSnapshot? _pendingShopSnapshot;
    private static List<CardInfo>? _pendingUpgradeDeckSnapshot;

    public static void Initialize()
    {
        Directory.CreateDirectory(TrackerRuntime.OutputDirectory);
    }

    // -----------------------------------------------------------------------
    // Run lifecycle
    // -----------------------------------------------------------------------

    public static void OnRunStarted()
    {
        lock (Sync)
        {
            // Save any previous unfinished run
            if (_currentRun is not null)
            {
                _currentRun.Outcome = "abandon";
                SaveRunLog(_currentRun);
            }

            _currentRun = new RunLog
            {
                RunId = Guid.NewGuid().ToString("N")[..12],
                StartedAt = DateTimeOffset.UtcNow.ToString("o"),
                Character = TryGetCurrentCharacter()
            };

            // Snapshot the starter deck immediately
            var initialRunState = TryGetCurrentRunState();
            _lastDeckSnapshot = initialRunState is not null
                ? SnapshotDeckIds(initialRunState)
                : new List<string>();
            _wasInCardReward = false;
            _wasInShop = false;
            _wasInUpgrade = false;
            _pendingCardRewardOffered = null;
            _pendingShopSnapshot = null;
            _pendingUpgradeDeckSnapshot = null;

            Log.Info($"[{TrackerRuntime.ModId}] New run started: {_currentRun.RunId}");
        }
    }

    public static void OnRunEnded(bool victory)
    {
        lock (Sync)
        {
            if (_currentRun is null) return;

            _currentRun.EndedAt = DateTimeOffset.UtcNow.ToString("o");
            _currentRun.Outcome = victory ? "victory" : "death";

            CaptureRunFinalState(_currentRun);
            SaveRunLog(_currentRun);
            _currentRun = null;
        }
    }

    // -----------------------------------------------------------------------
    // State polling (called from _Process patch)
    // -----------------------------------------------------------------------

    public static void OnFrameTick()
    {
        lock (Sync)
        {
            if (_currentRun is null) return;

            try
            {
                PollStateTransitions();
            }
            catch (Exception ex)
            {
                Log.Warn($"[{TrackerRuntime.ModId}] Frame tick error: {ex.Message}");
            }
        }
    }

    private static void PollStateTransitions()
    {
        var runManager = RunManager.Instance;
        if (runManager is null) return;

        var runState = TryGetRunState(runManager);
        if (runState is null) return;

        // Detect game over
        if (runState.IsGameOver)
        {
            if (_currentRun is not null && _currentRun.Outcome is null)
            {
                var player = runState.Players.Count > 0 ? runState.Players[0] : null;
                var hp = player?.Creature?.CurrentHp ?? 0;
                _currentRun.Outcome = hp > 0 ? "victory" : "death";
                _currentRun.EndedAt = DateTimeOffset.UtcNow.ToString("o");
                CaptureRunFinalState(_currentRun);
                SaveRunLog(_currentRun);
                _currentRun = null;
            }
            return;
        }

        // Check if we're on a card reward screen
        var isInCardReward = IsCardRewardScreenActive();
        if (isInCardReward && !_wasInCardReward)
        {
            // Just entered card reward — snapshot what's offered
            _pendingCardRewardOffered = CaptureCardRewardOptions();
        }
        else if (!isInCardReward && _wasInCardReward && _pendingCardRewardOffered is not null)
        {
            // Just left card reward — diff the deck to see what was picked
            RecordCardRewardDecision(runState);
        }
        _wasInCardReward = isInCardReward;

        // Check if we're in shop
        var isInShop = IsShopActive();
        if (isInShop && !_wasInShop)
        {
            _pendingShopSnapshot = CaptureShopInventory(runState);
        }
        _wasInShop = isInShop;

        // Check if we're in upgrade screen
        var isInUpgrade = IsUpgradeScreenActive();
        if (isInUpgrade && !_wasInUpgrade)
        {
            _pendingUpgradeDeckSnapshot = CaptureCurrentDeck(runState);
        }
        else if (!isInUpgrade && _wasInUpgrade && _pendingUpgradeDeckSnapshot is not null)
        {
            RecordUpgradeDecision(runState);
        }
        _wasInUpgrade = isInUpgrade;

        // Run start detection (if character just became available)
        if (_currentRun?.Character is null)
        {
            _currentRun!.Character = TryGetCurrentCharacter();
        }
    }

    // -----------------------------------------------------------------------
    // Card reward tracking
    // -----------------------------------------------------------------------

    private static void RecordCardRewardDecision(RunState runState)
    {
        var currentDeck = SnapshotDeckIds(runState);
        var prevDeck = _lastDeckSnapshot ?? new List<string>();

        // Find new cards (in current but not in previous)
        var added = FindAdded(prevDeck, currentDeck);

        var evt = new DraftEvent
        {
            Type = added.Count > 0 ? "card_reward" : "card_skip",
            Floor = runState.TotalFloor,
            Act = runState.CurrentActIndex + 1,
            Timestamp = DateTimeOffset.UtcNow.ToString("o"),
            Offered = _pendingCardRewardOffered
        };

        if (added.Count > 0)
        {
            // Find the picked card in offered list
            var pickedId = added[0];
            evt.Picked = _pendingCardRewardOffered?.FirstOrDefault(c => c.Id == pickedId)
                         ?? new CardInfo { Id = pickedId, Name = pickedId };
            evt.Skipped = false;
        }
        else
        {
            evt.Skipped = true;
        }

        _currentRun?.Events.Add(evt);
        _lastDeckSnapshot = currentDeck;
        _pendingCardRewardOffered = null;

        Log.Info($"[{TrackerRuntime.ModId}] Card reward: {(evt.Skipped == true ? "SKIPPED" : $"PICKED {evt.Picked?.Name}")}");
    }

    // -----------------------------------------------------------------------
    // Shop tracking
    // -----------------------------------------------------------------------

    public static void OnShopPurchase(string itemKind, string itemName, int cost)
    {
        lock (Sync)
        {
            if (_currentRun is null) return;

            var runState = TryGetCurrentRunState();
            var player = runState?.Players.Count > 0 ? runState.Players[0] : null;

            var evt = new DraftEvent
            {
                Type = itemKind == "removal" ? "shop_remove" : "shop_buy",
                Floor = runState?.TotalFloor ?? 0,
                Act = (runState?.CurrentActIndex ?? 0) + 1,
                Timestamp = DateTimeOffset.UtcNow.ToString("o"),
                ItemKind = itemKind,
                ItemName = itemName,
                Cost = cost,
                GoldBefore = (player?.Gold ?? 0) + cost, // gold already deducted
                ShopInventory = _pendingShopSnapshot
            };

            _currentRun.Events.Add(evt);
            _lastDeckSnapshot = runState is not null ? SnapshotDeckIds(runState) : _lastDeckSnapshot;

            Log.Info($"[{TrackerRuntime.ModId}] Shop {itemKind}: {itemName} (cost={cost})");
        }
    }

    // -----------------------------------------------------------------------
    // Upgrade tracking
    // -----------------------------------------------------------------------

    private static void RecordUpgradeDecision(RunState runState)
    {
        var currentDeck = CaptureCurrentDeck(runState);
        var prevDeck = _pendingUpgradeDeckSnapshot ?? new List<CardInfo>();

        // Find upgraded cards by comparing name+upgraded fields
        CardInfo? upgraded = null;
        for (var i = 0; i < Math.Min(currentDeck.Count, prevDeck.Count); i++)
        {
            if (!prevDeck[i].Upgraded && currentDeck[i].Upgraded &&
                currentDeck[i].Id == prevDeck[i].Id)
            {
                upgraded = currentDeck[i];
                break;
            }
        }

        // Fallback: diff by upgraded count
        if (upgraded is null)
        {
            var newUpgrades = currentDeck.Where(c => c.Upgraded).Select(c => c.Id).ToHashSet();
            var oldUpgrades = prevDeck.Where(c => c.Upgraded).Select(c => c.Id).ToHashSet();
            var diff = newUpgrades.Except(oldUpgrades).FirstOrDefault();
            if (diff is not null)
            {
                upgraded = currentDeck.FirstOrDefault(c => c.Id == diff && c.Upgraded);
            }
        }

        if (upgraded is not null)
        {
            var evt = new DraftEvent
            {
                Type = "upgrade",
                Floor = runState.TotalFloor,
                Act = runState.CurrentActIndex + 1,
                Timestamp = DateTimeOffset.UtcNow.ToString("o"),
                UpgradedCard = upgraded,
                DeckBefore = prevDeck
            };
            _currentRun?.Events.Add(evt);
            Log.Info($"[{TrackerRuntime.ModId}] Upgrade: {upgraded.Name}");
        }

        _pendingUpgradeDeckSnapshot = null;
        _lastDeckSnapshot = SnapshotDeckIds(runState);
    }

    // -----------------------------------------------------------------------
    // Event tracking
    // -----------------------------------------------------------------------

    public static void OnEventOptionChosen(string eventName, string optionText)
    {
        lock (Sync)
        {
            if (_currentRun is null) return;

            var runState = TryGetCurrentRunState();
            var evt = new DraftEvent
            {
                Type = "event_choice",
                Floor = runState?.TotalFloor ?? 0,
                Act = (runState?.CurrentActIndex ?? 0) + 1,
                Timestamp = DateTimeOffset.UtcNow.ToString("o"),
                EventName = eventName,
                OptionChosen = optionText
            };
            _currentRun.Events.Add(evt);
            _lastDeckSnapshot = runState is not null ? SnapshotDeckIds(runState) : _lastDeckSnapshot;

            Log.Info($"[{TrackerRuntime.ModId}] Event: {eventName} -> {optionText}");
        }
    }

    // -----------------------------------------------------------------------
    // State capture helpers
    // -----------------------------------------------------------------------

    private static List<CardInfo> CaptureCardRewardOptions()
    {
        var result = new List<CardInfo>();
        try
        {
            var game = NGame.Instance;
            if (game is null) return result;
            var runNode = game.CurrentRunNode ?? NRun.Instance;
            var rewardScreen = runNode is not null
                ? FindDescendant<NCardRewardSelectionScreen>(runNode)
                : null;
            if (rewardScreen is null) return result;

            var holders = FindDescendants<NCardHolder>(rewardScreen);
            foreach (var holder in holders)
            {
                var card = holder?.CardModel;
                if (card is null) continue;
                result.Add(BuildCardInfo(card));
            }
        }
        catch (Exception ex)
        {
            Log.Warn($"[{TrackerRuntime.ModId}] CaptureCardRewardOptions error: {ex.Message}");
        }
        return result;
    }

    private static ShopSnapshot? CaptureShopInventory(RunState runState)
    {
        try
        {
            var merchantRoom = FindSingleton<NMerchantRoom>();
            if (merchantRoom is null) return null;

            var inventory = merchantRoom.Inventory;
            if (inventory is null) return null;

            var player = runState.Players.Count > 0 ? runState.Players[0] : null;

            var snapshot = new ShopSnapshot
            {
                Gold = player?.Gold ?? 0
            };

            // Enumerate slots
            var slots = FindDescendants<NMerchantSlot>(inventory);
            foreach (var slot in slots)
            {
                if (slot?.Entry is null) continue;
                var entry = slot.Entry;

                var item = new ShopItem
                {
                    Name = TryGetEntryTitle(entry),
                    Cost = entry.Cost,
                    Affordable = entry.EnoughGold
                };

                if (entry is MerchantCardEntry)
                    snapshot.Cards.Add(item);
                else if (entry is MerchantRelicEntry)
                    snapshot.Relics.Add(item);
                else if (entry is MerchantPotionEntry)
                    snapshot.Potions.Add(item);
                else if (entry is MerchantCardRemovalEntry removal)
                {
                    snapshot.RemovalCost = removal.Cost;
                    snapshot.RemovalAvailable = !removal.Used;
                }
            }

            return snapshot;
        }
        catch (Exception ex)
        {
            Log.Warn($"[{TrackerRuntime.ModId}] CaptureShopInventory error: {ex.Message}");
            return null;
        }
    }

    private static List<CardInfo> CaptureCurrentDeck(RunState runState)
    {
        var result = new List<CardInfo>();
        var player = runState.Players.Count > 0 ? runState.Players[0] : null;
        var cards = player?.Deck?.Cards;
        if (cards is null) return result;

        foreach (var card in cards)
        {
            if (card is not null)
                result.Add(BuildCardInfo(card));
        }
        return result;
    }

    private static List<string> SnapshotDeckIds(RunState runState)
    {
        var player = runState.Players.Count > 0 ? runState.Players[0] : null;
        var cards = player?.Deck?.Cards;
        if (cards is null) return new List<string>();
        return cards.Where(c => c is not null).Select(c => c!.Id.ToString()).ToList();
    }

    private static void CaptureRunFinalState(RunLog run)
    {
        try
        {
            var runState = TryGetCurrentRunState();
            if (runState is null) return;

            run.FinalFloor = runState.TotalFloor;
            var player = runState.Players.Count > 0 ? runState.Players[0] : null;
            if (player is null) return;

            run.FinalHp = player.Creature?.CurrentHp ?? 0;
            run.FinalMaxHp = player.Creature?.MaxHp ?? 0;
            run.FinalGold = player.Gold;
            run.FinalDeck = CaptureCurrentDeck(runState);
            var relicNames = new List<string>();
            foreach (var relic in player.Relics)
            {
                if (relic is null) continue;
                var name = TextOf(relic.Title);
                if (string.IsNullOrWhiteSpace(name))
                    name = relic.Id.ToString();
                relicNames.Add(name);
            }
            run.FinalRelics = relicNames;
        }
        catch (Exception ex)
        {
            Log.Warn($"[{TrackerRuntime.ModId}] CaptureRunFinalState error: {ex.Message}");
        }
    }

    // -----------------------------------------------------------------------
    // Card info builder
    // -----------------------------------------------------------------------

    private static CardInfo BuildCardInfo(CardModel card)
    {
        var isUpgraded = false;
        try
        {
            var prop = card.GetType().GetProperty("IsUpgraded",
                BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
            if (prop?.GetValue(card) is bool v) isUpgraded = v;
        }
        catch { /* best effort */ }

        return new CardInfo
        {
            Id = card.Id.ToString(),
            Name = TextOf(card.Title),
            Type = card.Type.ToString(),
            Rarity = card.Rarity.ToString(),
            Cost = card.EnergyCost.GetResolved(),
            Upgraded = isUpgraded
        };
    }

    // -----------------------------------------------------------------------
    // Screen detection helpers
    // -----------------------------------------------------------------------

    private static bool IsCardRewardScreenActive()
    {
        try
        {
            var game = NGame.Instance;
            if (game is null) return false;
            var runNode = game.CurrentRunNode ?? NRun.Instance;
            if (runNode is null) return false;
            var screen = FindDescendant<NCardRewardSelectionScreen>(runNode);
            return screen is not null && IsNodeVisible(screen);
        }
        catch { return false; }
    }

    private static bool IsShopActive()
    {
        try
        {
            var room = FindSingleton<NMerchantRoom>();
            return room is not null && IsNodeVisible(room);
        }
        catch { return false; }
    }

    private static bool IsUpgradeScreenActive()
    {
        try
        {
            var game = NGame.Instance;
            if (game is null) return false;
            var runNode = game.CurrentRunNode ?? NRun.Instance;
            if (runNode is null) return false;
            var screen = FindDescendant<NDeckUpgradeSelectScreen>(runNode);
            return screen is not null && IsNodeVisible(screen);
        }
        catch { return false; }
    }

    // -----------------------------------------------------------------------
    // Utility: reflection, node traversal, text
    // -----------------------------------------------------------------------

    private static RunState? TryGetCurrentRunState()
    {
        try
        {
            return RunManager.Instance?.DebugOnlyGetState();
        }
        catch { return null; }
    }

    private static RunState? TryGetRunState(RunManager? rm)
    {
        if (rm is null) return null;
        try { return rm.DebugOnlyGetState(); }
        catch { return null; }
    }

    private static string? TryGetCurrentCharacter()
    {
        try
        {
            var runState = TryGetCurrentRunState();
            var player = runState?.Players.Count > 0 ? runState.Players[0] : null;
            if (player?.Character is null) return null;

            // Try display name first (via CharacterSelectTitle LocString)
            var displayName = TextOf(player.Character.CharacterSelectTitle);
            if (!string.IsNullOrWhiteSpace(displayName)) return displayName;

            // Fallback to ID
            return player.Character.Id.ToString();
        }
        catch { return null; }
    }

    private static string TryGetEntryTitle(MerchantEntry entry)
    {
        try
        {
            if (entry is MerchantCardEntry cardEntry)
                return TextOf(cardEntry.CreationResult?.Card?.Title);
            if (entry is MerchantRelicEntry relicEntry)
                return TextOf(relicEntry.Model?.Title);
            if (entry is MerchantPotionEntry potionEntry)
                return TextOf(potionEntry.Model?.Title);
            if (entry is MerchantCardRemovalEntry)
                return "[Card Removal]";
        }
        catch { /* best effort */ }
        return entry.GetType().Name;
    }

    private static string TextOf(object? value)
    {
        if (value is null) return "";
        if (value is string s) return s;

        // Handle LocString — the game's localization type
        if (value is LocString locString)
        {
            try
            {
                var formatted = locString.GetFormattedText();
                if (!string.IsNullOrWhiteSpace(formatted)) return formatted;
            }
            catch { /* best effort */ }
            try
            {
                var raw = locString.GetRawText();
                if (!string.IsNullOrWhiteSpace(raw)) return raw;
            }
            catch { /* best effort */ }
        }

        // Try reflection: GetFormattedText() or GetRawText()
        try
        {
            var method = value.GetType().GetMethod("GetFormattedText",
                BindingFlags.Instance | BindingFlags.Public, null, Type.EmptyTypes, null);
            if (method is not null)
            {
                var result = method.Invoke(value, null) as string;
                if (!string.IsNullOrWhiteSpace(result)) return result;
            }
        }
        catch { /* best effort */ }

        var text = value.ToString() ?? "";
        // Filter out type names like "MegaCrit.Sts2.Core.Localization.LocString"
        if (text.StartsWith("MegaCrit.", StringComparison.Ordinal) ||
            text.StartsWith("System.", StringComparison.Ordinal))
            return "";
        return text;
    }

    private static bool IsNodeVisible(Godot.Node? node)
    {
        if (node is null) return false;
        try
        {
            if (!Godot.GodotObject.IsInstanceValid(node)) return false;
            if (node is Godot.CanvasItem ci) return ci.Visible;
            // Check via property reflection
            var prop = node.GetType().GetProperty("Visible",
                BindingFlags.Instance | BindingFlags.Public);
            if (prop?.GetValue(node) is bool v) return v;
            return true; // If we can't check, assume visible
        }
        catch { return false; }
    }

    private static T? FindSingleton<T>() where T : Godot.Node
    {
        try
        {
            var prop = typeof(T).GetProperty("Instance",
                BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic);
            return prop?.GetValue(null) as T;
        }
        catch { return null; }
    }

    private static T? FindDescendant<T>(Godot.Node root) where T : Godot.Node
    {
        try
        {
            foreach (var child in root.GetChildren())
            {
                if (child is T match && Godot.GodotObject.IsInstanceValid(match))
                    return match;
                if (child is Godot.Node childNode)
                {
                    var found = FindDescendant<T>(childNode);
                    if (found is not null) return found;
                }
            }
        }
        catch { /* best effort */ }
        return null;
    }

    private static List<T> FindDescendants<T>(Godot.Node root) where T : Godot.Node
    {
        var result = new List<T>();
        try
        {
            foreach (var child in root.GetChildren())
            {
                if (child is T match && Godot.GodotObject.IsInstanceValid(match))
                    result.Add(match);
                if (child is Godot.Node childNode)
                    result.AddRange(FindDescendants<T>(childNode));
            }
        }
        catch { /* best effort */ }
        return result;
    }

    private static List<string> FindAdded(List<string> before, List<string> after)
    {
        // Find IDs present in 'after' but not in 'before'
        // Handle duplicates by counting occurrences
        var beforeCounts = new Dictionary<string, int>(StringComparer.Ordinal);
        foreach (var id in before)
        {
            beforeCounts.TryGetValue(id, out var count);
            beforeCounts[id] = count + 1;
        }

        var added = new List<string>();
        foreach (var id in after)
        {
            if (beforeCounts.TryGetValue(id, out var count) && count > 0)
            {
                beforeCounts[id] = count - 1;
            }
            else
            {
                added.Add(id);
            }
        }
        return added;
    }

    // -----------------------------------------------------------------------
    // File I/O
    // -----------------------------------------------------------------------

    private static void SaveRunLog(RunLog run)
    {
        try
        {
            Directory.CreateDirectory(TrackerRuntime.OutputDirectory);
            var timestamp = DateTimeOffset.UtcNow.ToString("yyyyMMdd-HHmmss");
            var outcome = run.Outcome ?? "unknown";
            var character = run.Character ?? "unknown";
            var fileName = $"run-{timestamp}-{character}-{outcome}-{run.RunId}.json";
            var filePath = Path.Combine(TrackerRuntime.OutputDirectory, fileName);

            var json = JsonSerializer.Serialize(run, JsonOpts);
            File.WriteAllText(filePath, json);

            Log.Info($"[{TrackerRuntime.ModId}] Saved run log: {filePath} ({run.Events.Count} events)");
        }
        catch (Exception ex)
        {
            Log.Error($"[{TrackerRuntime.ModId}] Failed to save run log: {ex.Message}");
        }
    }
}
