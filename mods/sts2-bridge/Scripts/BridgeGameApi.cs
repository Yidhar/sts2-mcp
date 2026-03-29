using System.Collections;
using System.Buffers.Binary;
using System.Globalization;
using System.Linq;
using System.Net;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.Json.Serialization;
using System.Text.RegularExpressions;
using System.Threading.Channels;
using Godot;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Merchant;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Entities.RestSite;
using MegaCrit.Sts2.Core.Events;
using MegaCrit.Sts2.Core.Events.Custom.CrystalSphereEvent;
using MegaCrit.Sts2.Core.GameActions;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Localization.DynamicVars;
using MegaCrit.Sts2.Core.Map;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.MonsterMoves.Intents;
using MegaCrit.Sts2.Core.MonsterMoves.MonsterMoveStateMachine;
using MegaCrit.Sts2.Core.Multiplayer.Game.PeerInput;
using MegaCrit.Sts2.Core.Nodes;
using MegaCrit.Sts2.Core.Nodes.Cards;
using MegaCrit.Sts2.Core.Nodes.Cards.Holders;
using MegaCrit.Sts2.Core.Nodes.Combat;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.Events;
using MegaCrit.Sts2.Core.Nodes.Events.Custom.CrystalSphere;
using MegaCrit.Sts2.Core.Nodes.Rewards;
using MegaCrit.Sts2.Core.Nodes.RestSite;
using MegaCrit.Sts2.Core.Nodes.Rooms;
using MegaCrit.Sts2.Core.Nodes.Screens;
using MegaCrit.Sts2.Core.Nodes.Screens.CharacterSelect;
using MegaCrit.Sts2.Core.Nodes.Screens.CardSelection;
using MegaCrit.Sts2.Core.Nodes.Screens.GameOverScreen;
using MegaCrit.Sts2.Core.Nodes.Screens.Map;
using MegaCrit.Sts2.Core.Nodes.Screens.Shops;
using MegaCrit.Sts2.Core.Nodes.Screens.TreasureRoomRelic;
using MegaCrit.Sts2.Core.Nodes.TreasureRooms;
using MegaCrit.Sts2.Core.Rewards;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;

namespace Sts2McpBridge.Scripts;

internal sealed class BridgeActionRequest
{
    [JsonPropertyName("action_id")]
    public string? ActionId { get; set; }

    [JsonPropertyName("action")]
    public string? LegacyActionId { get; set; }

    [JsonPropertyName("expected_state_version")]
    public long? ExpectedStateVersion { get; set; }

    [JsonPropertyName("wait_after_ms")]
    public int? WaitAfterMs { get; set; }

    [JsonIgnore]
    public string? RequestedActionId => ActionId ?? LegacyActionId;
}

internal sealed class BridgeRequestException : Exception
{
    public BridgeRequestException(
        HttpStatusCode statusCode,
        string errorCode,
        string message,
        object? details = null)
        : base(message)
    {
        StatusCode = statusCode;
        ErrorCode = errorCode;
        Details = details;
    }

    public HttpStatusCode StatusCode { get; }

    public string ErrorCode { get; }

    public object? Details { get; }
}

internal static partial class BridgeGameApi
{
    private const int NextFrontierWaitTimeoutMs = 5000;
    private const int PassiveFrontierWaitTimeoutMs = 1000;

    private static readonly JsonSerializerOptions HashJsonOptions = new()
    {
        WriteIndented = false
    };
    private static readonly HashSet<string> SemanticStateExcludedPropertyNames = new(StringComparer.Ordinal)
    {
        "automation",
        "available_actions",
        "description",
        "effect_preview",
        "dynamic_vars",
        "texts",
        "prompt",
        "label",
        "visible_glossary_source",
        "visible_glossary_texts",
        "visible_glossary",
        "selection_screen_prompt",
        "watchdog_dump"
    };

    public static async Task<object> GetStateResponseAsync(CancellationToken cancellationToken = default)
    {
        EnsureDispatcherReady();
        BridgeDebugTrace.Write("get_state requested");

        var frontier = await ObserveFrontierAsync(cancellationToken);
        BridgeDebugTrace.Write($"get_state completed state_version={frontier.Sequence}");
        return frontier.StatePayload;
    }

    public static void NotifyFrontierPumpTick()
    {
        if (!BridgeCoordinator.IsReady)
        {
            return;
        }

        BridgeFrontierStore.OnPumpTick();
    }

    public static void ResetFrontierState()
    {
        BridgeFrontierStore.Reset();
    }

    public static async Task StreamFrontierEventsAsync(
        HttpListenerResponse response,
        CancellationToken cancellationToken = default)
    {
        EnsureDispatcherReady();
        await BridgeFrontierStore.StreamEventsAsync(response, cancellationToken);
    }

    public static async Task<object> PerformActionResponseAsync(
        BridgeActionRequest? request,
        CancellationToken cancellationToken)
    {
        EnsureDispatcherReady();

        request ??= new BridgeActionRequest();

        var actionId = request.RequestedActionId?.Trim();
        if (string.IsNullOrWhiteSpace(actionId))
        {
            throw new BridgeRequestException(
                HttpStatusCode.BadRequest,
                "missing_action_id",
                "Request body must include a non-empty action_id.");
        }

        var before = await ObserveFrontierAsync(cancellationToken);
        BridgeDebugTrace.Write($"perform_action snapshot_before action={actionId} version={before.Sequence}");

        if (request.ExpectedStateVersion is long expectedStateVersion &&
            expectedStateVersion != before.Sequence)
        {
            throw new BridgeRequestException(
                HttpStatusCode.Conflict,
                "state_version_conflict",
                $"Expected state_version {expectedStateVersion}, but the current state_version is {before.Sequence}.",
                new
                {
                    action_id = actionId,
                    expected_state_version = expectedStateVersion,
                    current_state_version = before.Sequence,
                    current_state_hash = before.FrontierHash,
                    current_screen = before.Snapshot.Fields.Screen,
                    available_actions = before.Snapshot.ActionPayloads
                });
        }

        if (!before.Snapshot.ActionLookup.TryGetValue(actionId, out var action))
        {
            throw new BridgeRequestException(
                HttpStatusCode.Conflict,
                "action_not_available",
                $"Action '{actionId}' is not currently available.",
                new
                {
                    action_id = actionId,
                    current_state_version = before.Sequence,
                    current_state_hash = before.FrontierHash,
                    current_screen = before.Snapshot.Fields.Screen,
                    available_actions = before.Snapshot.ActionPayloads
                });
        }

        var waitAfterMs = Math.Clamp(request.WaitAfterMs ?? 0, 0, 5000);
        var after = await ExecuteActionAndWaitForFrontierAsync(
            before,
            actionId,
            action,
            waitAfterMs,
            cancellationToken);
        var autoExecutedActions = new List<object>();

        if (IsRewardResolutionAction(actionId))
        {
            (after, autoExecutedActions) = await MaybeAutoProceedAfterRewardActionAsync(after, cancellationToken);
        }
        else if (IsCardSelectionResolutionAction(actionId))
        {
            (after, autoExecutedActions) = await MaybeAutoCompleteCardSelectionAsync(after, cancellationToken);
        }

        BridgeDebugTrace.Write($"perform_action snapshot_after action={actionId} version={after.Sequence}");

        return new
        {
            ok = true,
            action_id = actionId,
            matched_action = BuildResolvedActionAckPayload(action.Payload),
            state_version_before = before.Sequence,
            state_version_after = after.Sequence,
            screen_after = after.Snapshot.Fields.Screen,
            auto_executed_actions = autoExecutedActions,
            state_changed = HasFrontierChanged(before, after)
        };
    }

    private static async Task<ObservedFrontier> ObserveFrontierAsync(CancellationToken cancellationToken)
    {
        BridgeDebugTrace.Write("observe_frontier executing on main thread");
        var snapshot = await BridgeCoordinator.RunOnMainThreadAsync(CaptureSnapshot);
        var frontier = PublishFrontier(snapshot);
        BridgeDebugTrace.Write($"observe_frontier completed version={frontier.Sequence}");
        return frontier;
    }

    private static ObservedFrontier PublishFrontier(BridgeSnapshot snapshot)
    {
        return BridgeFrontierStore.PublishSnapshot(snapshot);
    }

    private static async Task<ObservedFrontier> ExecuteActionAndWaitForFrontierAsync(
        ObservedFrontier before,
        string actionId,
        BridgeResolvedAction action,
        int waitAfterMs,
        CancellationToken cancellationToken)
    {
        await BridgeCoordinator.RunOnMainThreadAsync(() =>
        {
            BridgeDebugTrace.Write($"perform_action executing action={actionId}");
            action.Execute();
            return true;
        });

        if (waitAfterMs > 0)
        {
            await Task.Delay(waitAfterMs, cancellationToken);
        }

        if (ShouldTryImmediateObservedFrontier(actionId))
        {
            var immediateFrontier = await ObserveFrontierAsync(cancellationToken);
            if (HasFrontierChanged(before, immediateFrontier))
            {
                BridgeDebugTrace.Write(
                    $"perform_action immediate_frontier action={actionId} version={immediateFrontier.Sequence}");
                return immediateFrontier;
            }
        }

        var frontier = await BridgeFrontierStore.WaitForNextFrontierAsync(
            before.Sequence,
            NextFrontierWaitTimeoutMs,
            cancellationToken);
        if (frontier is not null)
        {
            return frontier;
        }

        BridgeDebugTrace.Write(
            $"perform_action frontier_wait_timeout action={actionId} after_version={before.Sequence}");
        return await ObserveFrontierAsync(cancellationToken);
    }

    private static async Task<ObservedFrontier> WaitForNextObservedFrontierAsync(
        ObservedFrontier before,
        int timeoutMs,
        CancellationToken cancellationToken)
    {
        var frontier = await BridgeFrontierStore.WaitForNextFrontierAsync(
            before.Sequence,
            timeoutMs,
            cancellationToken);
        return frontier ?? await ObserveFrontierAsync(cancellationToken);
    }

    private static bool ShouldTryImmediateObservedFrontier(string actionId)
    {
        return actionId.StartsWith("play_card:", StringComparison.Ordinal) ||
               actionId.StartsWith("card_selection:", StringComparison.Ordinal) ||
               actionId.StartsWith("use_potion:", StringComparison.Ordinal);
    }

    private static bool HasFrontierChanged(ObservedFrontier before, ObservedFrontier after)
    {
        return before.Sequence != after.Sequence ||
               !before.FrontierHash.Equals(after.FrontierHash, StringComparison.Ordinal);
    }

    private static object BuildResolvedActionAckPayload(object? payload)
    {
        var targetPayload = ReadPayloadPropertyValue(payload, "target");
        var targetMappingPayload = ReadPayloadPropertyValue(payload, "target_mapping");
        return new
        {
            action_id = ReadPayloadStringProperty(payload, "action_id"),
            kind = ReadPayloadStringProperty(payload, "kind"),
            label = ReadPayloadStringProperty(payload, "label"),
            target_action_suffix = ReadPayloadStringProperty(payload, "target_action_suffix"),
            target_combat_id = ReadPayloadIntegerProperty(payload, "target_combat_id")
                               ?? ReadPayloadIntegerProperty(targetPayload, "combat_id")
                               ?? ReadPayloadIntegerProperty(targetMappingPayload, "combat_id"),
            target_name = ReadPayloadStringProperty(payload, "target_name")
                          ?? ReadPayloadStringProperty(targetPayload, "name")
                          ?? ReadPayloadStringProperty(targetMappingPayload, "name"),
            target_side = ReadPayloadStringProperty(payload, "target_side")
                          ?? ReadPayloadStringProperty(targetPayload, "side")
                          ?? ReadPayloadStringProperty(targetMappingPayload, "side")
        };
    }

    private static string? ReadPayloadStringProperty(object? payload, string propertyName)
    {
        var value = ReadPayloadPropertyValue(payload, propertyName);
        return value as string;
    }

    private static int? ReadPayloadIntegerProperty(object? payload, string propertyName)
    {
        var value = ReadPayloadPropertyValue(payload, propertyName);
        if (value is byte byteValue)
        {
            return byteValue;
        }

        if (value is sbyte sbyteValue)
        {
            return sbyteValue;
        }

        if (value is short shortValue)
        {
            return shortValue;
        }

        if (value is ushort ushortValue)
        {
            return ushortValue;
        }

        if (value is int intValue)
        {
            return intValue;
        }

        if (value is uint uintValue && uintValue <= int.MaxValue)
        {
            return (int)uintValue;
        }

        if (value is long longValue && longValue >= int.MinValue && longValue <= int.MaxValue)
        {
            return (int)longValue;
        }

        if (value is ulong ulongValue && ulongValue <= int.MaxValue)
        {
            return (int)ulongValue;
        }

        if (value is null)
        {
            return null;
        }

        try
        {
            return Convert.ToInt32(value, CultureInfo.InvariantCulture);
        }
        catch
        {
            return null;
        }
    }

    private static object? ReadPayloadPropertyValue(object? payload, string propertyName)
    {
        if (payload is null || string.IsNullOrWhiteSpace(propertyName))
        {
            return null;
        }

        var property = payload.GetType().GetProperty(
            propertyName,
            BindingFlags.Instance | BindingFlags.Public | BindingFlags.IgnoreCase);
        return property?.GetValue(payload);
    }

    private static BridgeResolvedAction[] GetNonAutomationActions(BridgeSnapshot snapshot)
    {
        return snapshot.Actions
            .Where(action => !action.ActionId.StartsWith("automation:", StringComparison.Ordinal))
            .ToArray();
    }

    private static BridgeSnapshot CaptureSnapshot()
    {
        return HydrateSnapshot(CaptureFrontierCandidate());
    }

    private static BridgeFrontierCandidate CaptureFrontierCandidate()
    {
        BridgeDebugTrace.Write("capture_frontier_candidate start");
        var context = CaptureContext();
        BridgeDebugTrace.Write($"capture_frontier_candidate context screen={context.Screen}");
        var actions = BuildResolvedActions(context);
        BridgeDebugTrace.Write($"capture_frontier_candidate actions={actions.Count}");
        var frontierHash = ComputeFrontierHash(BuildFrontierProbePayload(context, actions));
        BridgeDebugTrace.Write($"capture_frontier_candidate complete hash={frontierHash}");

        return new BridgeFrontierCandidate
        {
            Context = context,
            Actions = actions,
            FrontierHash = frontierHash
        };
    }

    private static BridgeSnapshot HydrateSnapshot(BridgeFrontierCandidate candidate)
    {
        BridgeDebugTrace.Write($"hydrate_snapshot start frontier_hash={candidate.FrontierHash}");
        var actionPayloads = candidate.Actions.Select(static action => action.Payload).ToArray();
        var fields = BuildStateFields(candidate.Context, actionPayloads);
        var corePayload = CreateSemanticStateCore(fields);
        var semanticStateHash = ComputeStateHash(corePayload);
        var statePayload = CreateStatePayload(fields, 0, candidate.FrontierHash, semanticStateHash);
        BridgeDebugTrace.Write(
            $"hydrate_snapshot complete frontier_hash={candidate.FrontierHash} semantic_hash={semanticStateHash}");

        return new BridgeSnapshot
        {
            Fields = fields,
            FrontierHash = candidate.FrontierHash,
            StateHash = semanticStateHash,
            StatePayload = statePayload,
            StateVersion = 0,
            Actions = candidate.Actions,
            ActionPayloads = actionPayloads,
            ActionLookup = candidate.Actions.ToDictionary(static action => action.ActionId, StringComparer.Ordinal)
        };
    }

    private static object BuildFrontierProbePayload(
        BridgeWorldContext context,
        IReadOnlyList<BridgeResolvedAction> actions)
    {
        return new
        {
            screen = context.Screen,
            action_ids = actions.Select(static action => action.ActionId).ToArray(),
            combat = BuildCombatFrontierPayload(context.CombatManager, context.CombatState),
            rewards = BuildRewardsFrontierPayload(context),
            card_reward_selection = BuildCardRewardSelectionFrontierPayload(
                context.CardRewardScreen,
                context.CardRewardOptions,
                context.CardRewardSkipButton),
            card_selection = BuildCardSelectionFrontierPayload(
                context.CardSelectionScreen,
                context.CardSelectionOptions,
                context.CardSelectionConfirmButton,
                context.CardSelectionCancelButton,
                context.CardSelectionCloseButton,
                context.CardSelectionSkipButton),
            crystal_sphere = BuildCrystalSphereFrontierPayload(
                context.CrystalSphereScreen,
                context.CrystalSphereCells),
            map = BuildMapFrontierPayload(context.RunState, context.MapScreen),
            rest_site = BuildRestSiteFrontierPayload(
                context.MapScreen,
                context.RestSiteRoom,
                context.RestSiteButtons,
                context.RestSiteProceedButton),
            deck_upgrade_selection = BuildDeckUpgradeSelectionFrontierPayload(
                context.DeckUpgradeScreen,
                context.DeckUpgradeOptions,
                context.DeckUpgradeConfirmButton,
                context.DeckUpgradeCancelButton,
                context.DeckUpgradeCloseButton),
            shop = BuildShopFrontierPayload(
                context.MerchantRoom,
                context.MerchantInventory,
                context.MerchantSlots,
                context.MerchantButton,
                context.MerchantProceedButton,
                context.MerchantBackButton)
        };
    }

    private static object BuildCombatFrontierPayload(CombatManager? combatManager, CombatState? combatState)
    {
        if (combatManager is null || combatState is null || !combatManager.IsInProgress)
        {
            return new
            {
                in_progress = false
            };
        }

        return new
        {
            in_progress = combatManager.IsInProgress,
            is_play_phase = combatManager.IsPlayPhase,
            is_paused = combatManager.IsPaused,
            is_ending = combatManager.IsEnding,
            player_actions_disabled = combatManager.PlayerActionsDisabled,
            round_number = combatState.RoundNumber,
            current_side = combatState.CurrentSide.ToString(),
            players = combatState.Players.Select(player => new
            {
                net_id = player.NetId,
                energy = player.PlayerCombatState?.Energy,
                max_energy = player.PlayerCombatState?.MaxEnergy,
                stars = player.PlayerCombatState?.Stars,
                creature = BuildCreatureFrontierPayload(player.Creature)
            }).ToArray(),
            enemies = combatState.Creatures
                .Where(static creature => creature.IsEnemy)
                .Select(BuildCreatureFrontierPayload)
                .ToArray()
        };
    }

    private static object BuildCreatureFrontierPayload(Creature? creature)
    {
        if (creature is null)
        {
            return new
            {
                missing = true
            };
        }

        return new
        {
            combat_id = creature.CombatId,
            current_hp = creature.CurrentHp,
            max_hp = creature.MaxHp,
            block = creature.Block,
            is_alive = creature.IsAlive,
            is_hittable = SafeGetCreatureIsHittable(creature),
            powers = creature.Powers.Select(BuildPowerFrontierPayload).ToArray(),
            intent = creature.IsEnemy ? BuildEnemyIntentFrontierPayload(creature) : null
        };
    }

    private static object BuildPowerFrontierPayload(PowerModel power)
    {
        return new
        {
            title = TextOf(power.Title),
            amount = power.Amount,
            display_amount = power.DisplayAmount
        };
    }

    private static object? BuildEnemyIntentFrontierPayload(Creature creature)
    {
        var monster = creature.Monster;
        if (monster is null)
        {
            return null;
        }

        var targets = ResolveMonsterIntentTargets(creature);
        var nextMove = monster.NextMove;
        return new
        {
            state_id = nextMove?.StateId,
            follow_up_state_id = nextMove?.FollowUpStateId,
            intents = SafeGetMonsterIntents(monster, nextMove)
                .Select(intent =>
                {
                    var repeats = intent switch
                    {
                        SingleAttackIntent singleAttackIntent => singleAttackIntent.Repeats,
                        MultiAttackIntent multiAttackIntent => multiAttackIntent.Repeats,
                        _ => 1
                    };

                    var totalDamage = intent switch
                    {
                        SingleAttackIntent singleAttackIntent =>
                            SafeGetIntentTotalDamage(singleAttackIntent, targets, creature),
                        MultiAttackIntent multiAttackIntent =>
                            SafeGetIntentTotalDamage(multiAttackIntent, targets, creature),
                        _ => null
                    };

                    return new
                    {
                        intent_type = intent.IntentType.ToString(),
                        repeats,
                        total_damage = totalDamage
                    };
                })
                .ToArray()
        };
    }

    private static object BuildRewardsFrontierPayload(BridgeWorldContext context)
    {
        var visible = IsRewardsScreenVisible(
            context.RewardsScreen,
            context.ProceedButton,
            context.RewardProceedButton,
            context.MapScreen,
            context.RewardButtons);
        return new
        {
            visible,
            terminal_proceed_visible = visible &&
                                       context.RewardProceedButton is not null &&
                                       IsNodeVisible(context.RewardProceedButton),
            reward_count = context.RewardButtons.Count
        };
    }

    private static object BuildCardRewardSelectionFrontierPayload(
        NCardRewardSelectionScreen? cardRewardScreen,
        IReadOnlyList<NCardHolder> cardRewardOptions,
        Node? cardRewardSkipButton)
    {
        var visible = IsCardRewardSelectionVisible(cardRewardScreen, cardRewardOptions);
        var ready = IsCardRewardSelectionReady(cardRewardScreen);
        return new
        {
            visible,
            ready,
            skip_visible = ready &&
                           cardRewardSkipButton is not null &&
                           IsNodeVisible(cardRewardSkipButton) &&
                           IsButtonEnabled(cardRewardSkipButton),
            option_count = cardRewardOptions.Count
        };
    }

    private static object BuildCardSelectionFrontierPayload(
        Node? cardSelectionScreen,
        IReadOnlyList<NCardHolder> cardSelectionOptions,
        Node? cardSelectionConfirmButton,
        Node? cardSelectionCancelButton,
        Node? cardSelectionCloseButton,
        Node? cardSelectionSkipButton)
    {
        var visible = cardSelectionScreen is not null && IsNodeVisible(cardSelectionScreen);
        var prefs = GetHiddenFieldValue(cardSelectionScreen, "_prefs");

        return new
        {
            visible,
            screen_type = visible ? cardSelectionScreen!.GetType().Name : null,
            selected_count = CountSelectedCardSelectionCards(cardSelectionScreen),
            min_select = GetHiddenPropertyValue<int>(prefs, "MinSelect"),
            max_select = GetHiddenPropertyValue<int>(prefs, "MaxSelect"),
            confirm_visible = cardSelectionConfirmButton is not null &&
                              IsNodeVisible(cardSelectionConfirmButton) &&
                              IsButtonEnabled(cardSelectionConfirmButton),
            cancel_visible = cardSelectionCancelButton is not null &&
                             IsNodeVisible(cardSelectionCancelButton) &&
                             IsButtonEnabled(cardSelectionCancelButton),
            close_visible = cardSelectionCloseButton is not null &&
                            IsNodeVisible(cardSelectionCloseButton) &&
                            IsButtonEnabled(cardSelectionCloseButton),
            skip_visible = cardSelectionSkipButton is not null &&
                           IsNodeVisible(cardSelectionSkipButton) &&
                           IsButtonEnabled(cardSelectionSkipButton),
            option_keys = cardSelectionOptions.Select((holder, index) =>
            {
                var optionIndex = GetCardSelectionOptionIndex(cardSelectionScreen, holder, index);
                return GetCardSelectionOptionSelectionId(cardSelectionScreen, holder, optionIndex) ??
                       optionIndex.ToString(CultureInfo.InvariantCulture);
            }).ToArray()
        };
    }

    private static object BuildCrystalSphereFrontierPayload(
        NCrystalSphereScreen? crystalSphereScreen,
        IReadOnlyList<NCrystalSphereCell> crystalSphereCells)
    {
        var visible = crystalSphereScreen is not null && IsNodeVisible(crystalSphereScreen);
        if (!visible)
        {
            return new
            {
                visible = false
            };
        }

        var minigame = GetCrystalSphereMinigame(crystalSphereScreen);
        return new
        {
            visible = true,
            divinations_left = GetCrystalSphereDivinationCount(minigame),
            current_tool = GetCrystalSphereToolName(minigame),
            is_finished = GetCrystalSphereIsFinished(minigame),
            cells = crystalSphereCells.Select(cell => new
            {
                x = cell.Entity?.X,
                y = cell.Entity?.Y,
                is_hidden = cell.Entity?.IsHidden ?? true,
                is_highlighted = cell.Entity?.IsHighlighted ?? false
            }).ToArray()
        };
    }

    private static object BuildMapFrontierPayload(RunState? runState, NMapScreen? mapScreen)
    {
        return new
        {
            is_open = mapScreen?.IsOpen ?? false,
            is_travel_enabled = mapScreen?.IsTravelEnabled ?? false,
            is_traveling = mapScreen?.IsTraveling ?? false,
            current_coord = BuildMapCoord(runState?.CurrentMapCoord)
        };
    }

    private static object BuildRestSiteFrontierPayload(
        NMapScreen? mapScreen,
        NRestSiteRoom? restSiteRoom,
        IReadOnlyList<NRestSiteButton> restSiteButtons,
        NProceedButton? restSiteProceedButton)
    {
        var visible = mapScreen?.IsOpen != true &&
                      restSiteRoom is not null &&
                      IsNodeVisible(restSiteRoom);
        return new
        {
            visible,
            proceed_visible = visible &&
                              !HasVisibleEnabledRestSiteOptions(restSiteButtons) &&
                              restSiteProceedButton is not null &&
                              IsNodeVisible(restSiteProceedButton) &&
                              IsButtonEnabled(restSiteProceedButton),
            option_count = visible ? restSiteButtons.Count(IsNodeVisible) : 0
        };
    }

    private static object BuildDeckUpgradeSelectionFrontierPayload(
        NDeckUpgradeSelectScreen? deckUpgradeScreen,
        IReadOnlyList<NCardHolder> deckUpgradeOptions,
        NConfirmButton? deckUpgradeConfirmButton,
        NBackButton? deckUpgradeCancelButton,
        NBackButton? deckUpgradeCloseButton)
    {
        var visible = deckUpgradeScreen is not null && IsNodeVisible(deckUpgradeScreen);
        return new
        {
            visible,
            selected_count = CountSelectedDeckUpgradeCards(deckUpgradeScreen),
            confirm_visible = deckUpgradeConfirmButton is not null &&
                              IsNodeVisible(deckUpgradeConfirmButton) &&
                              IsButtonEnabled(deckUpgradeConfirmButton),
            cancel_visible = deckUpgradeCancelButton is not null &&
                             IsNodeVisible(deckUpgradeCancelButton) &&
                             IsButtonEnabled(deckUpgradeCancelButton),
            close_visible = deckUpgradeCloseButton is not null &&
                            IsNodeVisible(deckUpgradeCloseButton) &&
                            IsButtonEnabled(deckUpgradeCloseButton),
            option_count = visible ? deckUpgradeOptions.Count : 0
        };
    }

    private static object BuildShopFrontierPayload(
        NMerchantRoom? merchantRoom,
        NMerchantInventory? merchantInventory,
        IReadOnlyList<NMerchantSlot> merchantSlots,
        NMerchantButton? merchantButton,
        NProceedButton? merchantProceedButton,
        NBackButton? merchantBackButton)
    {
        return new
        {
            visible = (merchantRoom is not null && IsNodeVisible(merchantRoom)) ||
                      (merchantInventory is not null && IsNodeVisible(merchantInventory)),
            is_open = merchantInventory?.IsOpen ?? false,
            merchant_button_visible = merchantButton is not null && IsNodeVisible(merchantButton),
            back_button_visible = merchantBackButton is not null && IsNodeVisible(merchantBackButton),
            proceed_visible = merchantProceedButton is not null && IsNodeVisible(merchantProceedButton),
            item_count = merchantSlots.Count
        };
    }

    private static BridgeWorldContext CaptureContext()
    {
        var game = NGame.Instance;
        if (game is null || !GodotObject.IsInstanceValid(game))
        {
            throw new BridgeRequestException(
                HttpStatusCode.ServiceUnavailable,
                "game_not_ready",
                "NGame.Instance is not available yet.");
        }

        var runNode = game.CurrentRunNode ?? NRun.Instance;
        var runManager = RunManager.Instance;
        var combatManager = CombatManager.Instance;
        var runState = TryGetRunState(runManager);
        var combatState = TryGetCombatState(combatManager);
        var combatRoom = runNode?.CombatRoom;
        var combatUi = combatRoom?.Ui;
        var playerHand = combatUi?.Hand;
        var endTurnButton = combatUi?.EndTurnButton;
        var mapScreen = NMapScreen.Instance;
        var restSiteRoom = NRestSiteRoom.Instance;
        var restSiteProceedButton = restSiteRoom?.ProceedButton;
        var merchantRoom = NMerchantRoom.Instance;
        var merchantInventory = merchantRoom?.Inventory;
        var merchantButton = merchantRoom?.MerchantButton;
        var merchantProceedButton = merchantRoom?.ProceedButton;
        var merchantBackButton = GetHiddenFieldValue(merchantInventory, "_backButton") as NBackButton;
        var treasureRoom = FindFirstVisibleDescendant<NTreasureRoom>(game);
        var treasureChestButton = ResolveFirstVisibleNode(
            GetHiddenFieldValue(treasureRoom, "_chestButton") as NTreasureButton,
            FindFirstVisibleDescendant<NTreasureButton>(treasureRoom));
        var treasureRelicCollection = ResolveFirstVisibleNode(
            GetHiddenFieldValue(treasureRoom, "_relicCollection") as NTreasureRoomRelicCollection,
            FindFirstVisibleDescendant<NTreasureRoomRelicCollection>(treasureRoom));
        var treasureRelicOptions = treasureRelicCollection is null
            ? new List<NTreasureRoomRelicHolder>()
            : SortByVisualPosition(FindVisibleDescendants<NTreasureRoomRelicHolder>(treasureRelicCollection));
        var proceedButton = ResolveVisibleProceedButton(
            game,
            combatRoom?.ProceedButton,
            treasureRoom?.ProceedButton,
            restSiteProceedButton,
            merchantProceedButton);
        var merchantSlots = merchantInventory is null
            ? new List<NMerchantSlot>()
            : SortByVisualPosition(merchantInventory.GetAllSlots().Where(IsNodeVisible));
        var rewardsScreen = FindFirstVisibleDescendant<NRewardsScreen>(game);
        var rewardProceedButton = ResolveFirstVisibleNode(
            GetHiddenPropertyObjectValue(rewardsScreen, "ProceedButton") as NProceedButton,
            GetHiddenFieldValue(rewardsScreen, "_proceedButton") as NProceedButton);
        var cardRewardScreen = FindFirstVisibleDescendant<NCardRewardSelectionScreen>(game);
        var characterSelectScreen = FindFirstVisibleDescendant<NCharacterSelectScreen>(game);
        var deckUpgradeScreen = FindFirstVisibleDescendant<NDeckUpgradeSelectScreen>(game);
        Node cardSelectionSearchRoot = game.GetTree()?.Root is Node sceneRoot
            ? sceneRoot
            : game;
        var cardRewardSkipButton = ResolveCardRewardSkipButton(cardRewardScreen);
        var cardSelectionScreen = ResolveVisibleCardSelectionScreen(
            cardSelectionSearchRoot,
            cardRewardScreen,
            deckUpgradeScreen) ?? ResolveCombatHandSelectionNode(playerHand);
        var restSiteButtons = SortByVisualPosition(FindVisibleDescendants<NRestSiteButton>(restSiteRoom));
        var characterButtons = SortByVisualPosition(FindVisibleDescendants<NCharacterSelectButton>(characterSelectScreen));
        var selectedCharacterButton = GetHiddenFieldValue(characterSelectScreen, "_selectedButton") as NCharacterSelectButton;
        var embarkButton = GetHiddenFieldValue(characterSelectScreen, "_embarkButton") as NConfirmButton;
        var rewardButtons = rewardsScreen is null
            ? new List<NRewardButton>()
            : SortByVisualPosition(
                FindVisibleDescendants<NRewardButton>(rewardsScreen)
                    .Where(button => !IsRewardButtonSkipped(rewardsScreen, button)));
        var cardRewardOptions = SortByVisualPosition(FindVisibleDescendants<NCardHolder>(cardRewardScreen));
        var deckUpgradeOptions = deckUpgradeScreen is null
            ? new List<NCardHolder>()
            : SortByVisualPosition(
                FindVisibleDescendants<NCardHolder>(deckUpgradeScreen)
                    .Where(holder => !IsDeckUpgradePreviewHolder(deckUpgradeScreen, holder)));
        var cardSelectionOptions = GetCardSelectionOptions(cardSelectionScreen);
        var deckUpgradeCancelButton = ResolveFirstVisibleNode(
            GetHiddenFieldValue(deckUpgradeScreen, "_singlePreviewCancelButton") as NBackButton,
            GetHiddenFieldValue(deckUpgradeScreen, "_multiPreviewCancelButton") as NBackButton);
        var deckUpgradeConfirmButton = ResolveFirstVisibleNode(
            GetHiddenFieldValue(deckUpgradeScreen, "_singlePreviewConfirmButton") as NConfirmButton,
            GetHiddenFieldValue(deckUpgradeScreen, "_multiPreviewConfirmButton") as NConfirmButton);
        var deckUpgradeCloseButton = GetHiddenFieldValue(deckUpgradeScreen, "_closeButton") as NBackButton;
        var cardSelectionConfirmButton = ResolveCardSelectionConfirmButton(cardSelectionScreen);
        var cardSelectionCancelButton = ResolveCardSelectionCancelButton(cardSelectionScreen);
        var cardSelectionCloseButton = GetHiddenFieldValue(cardSelectionScreen, "_closeButton") as Node;
        var cardSelectionSkipButton = GetHiddenFieldValue(cardSelectionScreen, "_skipButton") as Node;
        var eventRoom = FindFirstVisibleDescendant<NEventRoom>(game);
        var gameOverScreen = FindFirstVisibleDescendant<NGameOverScreen>(game);
        var gameOverContinueButton = GetHiddenFieldValue(gameOverScreen, "_continueButton") as NGameOverContinueButton;
        var gameOverMainMenuButton = GetHiddenFieldValue(gameOverScreen, "_mainMenuButton") as NReturnToMainMenuButton;
        var crystalSphereScreen = FindFirstVisibleDescendant<NCrystalSphereScreen>(game);
        var crystalSphereCells = crystalSphereScreen is null
            ? new List<NCrystalSphereCell>()
            : FindVisibleDescendants<NCrystalSphereCell>(crystalSphereScreen)
                .OrderBy(static cell => cell.Entity?.Y ?? int.MaxValue)
                .ThenBy(static cell => cell.Entity?.X ?? int.MaxValue)
                .ToList();
        var crystalSphereSmallDivinationButton =
            GetHiddenFieldValue(crystalSphereScreen, "_smallDivinationButton") as NDivinationButton;
        var crystalSphereBigDivinationButton =
            GetHiddenFieldValue(crystalSphereScreen, "_bigDivinationButton") as NDivinationButton;
        var crystalSphereProceedButton =
            GetHiddenFieldValue(crystalSphereScreen, "_proceedButton") as NProceedButton;
        var hoverTipSet = FindFirstVisibleDescendant(
            game,
            static node => IsTypeFullName(node, "MegaCrit.Sts2.Core.Nodes.HoverTips.NHoverTipSet"));
        var eventOptionButtons = SortByVisualPosition(FindVisibleDescendants<NEventOptionButton>(game))
            .Where(static button => button.Option is not null)
            .ToList();
        var mapPoints = FindVisibleDescendants<NMapPoint>(mapScreen)
            .OrderBy(static point => point.Point.coord.row)
            .ThenBy(static point => point.Point.coord.col)
            .ToList();
        var mainMenuRoot = FindFirstVisibleDescendant(
            game,
            static node => IsTypeFullName(node, "MegaCrit.Sts2.Core.Nodes.Screens.MainMenu.NMainMenu"));
        var mainMenuContinueButton = FindFirstVisibleDescendant(
            game,
            static node => IsTypeFullName(node, "MegaCrit.Sts2.Core.Nodes.Screens.MainMenu.NMainMenuContinueButton"));
        var mainMenuTextButtons = SortByVisualPosition(
            FindVisibleDescendants(
                game,
                static node => IsTypeFullName(node, "MegaCrit.Sts2.Core.Nodes.Screens.MainMenu.NMainMenuTextButton")));
        var runModeSubmenu = FindFirstVisibleDescendant(
            game,
            static node => IsTypeFullName(node, "MegaCrit.Sts2.Core.Nodes.Screens.MainMenu.NSingleplayerSubmenu"));
        var runModeStandardButton = GetHiddenFieldValue(runModeSubmenu, "_standardButton") as Node;
        var runModeDailyButton = GetHiddenFieldValue(runModeSubmenu, "_dailyButton") as Node;
        var runModeCustomButton = GetHiddenFieldValue(runModeSubmenu, "_customButton") as Node;
        var runModeBackButton = GetHiddenFieldValue(runModeSubmenu, "_backButton") as NBackButton;
        var continueRunInfo = FindFirstVisibleDescendant(
            game,
            static node => IsTypeFullName(node, "MegaCrit.Sts2.Core.Nodes.Screens.MainMenu.NContinueRunInfo"));
        var abandonRunConfirmPopup = FindFirstVisibleDescendant(
            game,
            static node => IsTypeFullName(node, "MegaCrit.Sts2.Core.Nodes.CommonUi.NAbandonRunConfirmPopup"));
        var abandonRunConfirmButtons = SortByVisualPosition(FindVisibleDescendants<NPopupYesNoButton>(abandonRunConfirmPopup));

        return new BridgeWorldContext
        {
            Game = game,
            RunNode = runNode,
            RunManager = runManager,
            CombatManager = combatManager,
            RunState = runState,
            CombatState = combatState,
            Screen = ResolveCurrentScreen(
                runNode?.ScreenStateTracker,
                combatManager,
                mapScreen,
                restSiteRoom,
                merchantRoom,
                merchantInventory,
                rewardsScreen,
                proceedButton,
                rewardProceedButton,
                rewardButtons,
                cardRewardScreen,
                cardRewardOptions,
                cardSelectionScreen,
                characterSelectScreen,
                deckUpgradeScreen,
                crystalSphereScreen,
                eventOptionButtons,
                gameOverScreen,
                mainMenuRoot,
                runModeSubmenu,
                abandonRunConfirmPopup),
            CombatRoom = combatRoom,
            CombatUi = combatUi,
            EndTurnButton = endTurnButton,
            ProceedButton = proceedButton,
            MapScreen = mapScreen,
            RestSiteRoom = restSiteRoom,
            MerchantRoom = merchantRoom,
            MerchantInventory = merchantInventory,
            TreasureRoom = treasureRoom,
            TreasureChestButton = treasureChestButton,
            TreasureRelicCollection = treasureRelicCollection,
            RewardsScreen = rewardsScreen,
            RewardProceedButton = rewardProceedButton,
            CardRewardScreen = cardRewardScreen,
            CardRewardSkipButton = cardRewardSkipButton,
            CardSelectionScreen = cardSelectionScreen,
            CharacterSelectScreen = characterSelectScreen,
            DeckUpgradeScreen = deckUpgradeScreen,
            RestSiteProceedButton = restSiteProceedButton,
            MerchantButton = merchantButton,
            MerchantProceedButton = merchantProceedButton,
            MerchantBackButton = merchantBackButton,
            SelectedCharacterButton = selectedCharacterButton,
            EmbarkButton = embarkButton,
            RewardButtons = rewardButtons,
            CardRewardOptions = cardRewardOptions,
            CardSelectionOptions = cardSelectionOptions,
            DeckUpgradeOptions = deckUpgradeOptions,
            CharacterButtons = characterButtons,
            EventOptionButtons = eventOptionButtons,
            EventRoom = eventRoom,
            GameOverScreen = gameOverScreen,
            GameOverContinueButton = gameOverContinueButton,
            GameOverMainMenuButton = gameOverMainMenuButton,
            CrystalSphereScreen = crystalSphereScreen,
            CrystalSphereCells = crystalSphereCells,
            CrystalSphereSmallDivinationButton = crystalSphereSmallDivinationButton,
            CrystalSphereBigDivinationButton = crystalSphereBigDivinationButton,
            CrystalSphereProceedButton = crystalSphereProceedButton,
            HoverTipSet = hoverTipSet,
            MapPoints = mapPoints,
            RestSiteButtons = restSiteButtons,
            MerchantSlots = merchantSlots,
            TreasureRelicOptions = treasureRelicOptions,
            MainMenuRoot = mainMenuRoot,
            MainMenuContinueButton = mainMenuContinueButton,
            MainMenuTextButtons = mainMenuTextButtons,
            RunModeSubmenu = runModeSubmenu,
            RunModeStandardButton = runModeStandardButton,
            RunModeDailyButton = runModeDailyButton,
            RunModeCustomButton = runModeCustomButton,
            RunModeBackButton = runModeBackButton,
            ContinueRunInfo = continueRunInfo,
            AbandonRunConfirmPopup = abandonRunConfirmPopup,
            AbandonRunConfirmButtons = abandonRunConfirmButtons,
            CardSelectionConfirmButton = cardSelectionConfirmButton,
            CardSelectionCancelButton = cardSelectionCancelButton,
            CardSelectionCloseButton = cardSelectionCloseButton,
            CardSelectionSkipButton = cardSelectionSkipButton,
            DeckUpgradeCancelButton = deckUpgradeCancelButton,
            DeckUpgradeConfirmButton = deckUpgradeConfirmButton,
            DeckUpgradeCloseButton = deckUpgradeCloseButton
        };
    }

    private static BridgeStateFields BuildStateFields(BridgeWorldContext context, object[] actionPayloads)
    {
        return new BridgeStateFields
        {
            Screen = context.Screen,
            Automation = BuildAutomationPayload(),
            Run = BuildRunPayload(context.RunState),
            Combat = BuildCombatPayload(context.CombatManager, context.CombatState),
            Players = BuildPlayersPayload(context.RunState, context.CombatManager, context.CombatState),
            Rewards = BuildRewardsPayload(
                context.RewardsScreen,
                context.ProceedButton,
                context.RewardProceedButton,
                context.MapScreen,
                context.RewardButtons),
            CardRewardSelection = BuildCardRewardSelectionPayload(
                context.CardRewardScreen,
                context.CardRewardOptions,
                context.CardRewardSkipButton),
            CardSelection = BuildCardSelectionPayload(
                context.CardSelectionScreen,
                context.CardSelectionOptions,
                context.CardSelectionConfirmButton,
                context.CardSelectionCancelButton,
                context.CardSelectionCloseButton,
                context.CardSelectionSkipButton),
            CharacterSelection = BuildCharacterSelectionPayload(
                context.CharacterSelectScreen,
                context.CharacterButtons,
                context.SelectedCharacterButton,
                context.EmbarkButton),
            RunModeSelection = BuildRunModeSelectionPayload(
                context.RunModeSubmenu,
                context.RunModeStandardButton,
                context.RunModeDailyButton,
                context.RunModeCustomButton,
                context.RunModeBackButton),
            EventOptions = BuildEventOptionsPayload(
                context.EventOptionButtons,
                context.MapScreen,
                context.EventRoom,
                context.HoverTipSet,
                context.CrystalSphereScreen,
                context.CrystalSphereCells,
                context.CrystalSphereSmallDivinationButton,
                context.CrystalSphereBigDivinationButton,
                context.CrystalSphereProceedButton),
            CrystalSphere = BuildCrystalSpherePayload(
                context.CrystalSphereScreen,
                context.CrystalSphereCells,
                context.CrystalSphereSmallDivinationButton,
                context.CrystalSphereBigDivinationButton,
                context.CrystalSphereProceedButton),
            Map = BuildMapPayload(context.RunState, context.MapScreen, context.MapPoints),
            RestSite = BuildRestSitePayload(
                context.MapScreen,
                context.RestSiteRoom,
                context.RestSiteButtons,
                context.RestSiteProceedButton),
            DeckUpgradeSelection = BuildDeckUpgradeSelectionPayload(
                context.DeckUpgradeScreen,
                context.DeckUpgradeOptions,
                context.DeckUpgradeConfirmButton,
                context.DeckUpgradeCancelButton,
                context.DeckUpgradeCloseButton),
            Shop = BuildShopPayload(
                context.MerchantRoom,
                context.MerchantInventory,
                context.MerchantSlots,
                context.MerchantButton,
                context.MerchantProceedButton,
                context.MerchantBackButton),
            MainMenu = BuildMainMenuPayload(
                context.MainMenuRoot,
                context.MainMenuContinueButton,
                context.MainMenuTextButtons,
                context.ContinueRunInfo,
                context.AbandonRunConfirmPopup,
                context.AbandonRunConfirmButtons),
            AvailableActions = actionPayloads
        };
    }

    private static object CreateSemanticStateCore(BridgeStateFields fields)
    {
        // Keep state_version/state_hash tied to the semantic game state rather
        // than transient automation metadata or fully-expanded action payloads.
        var rawCore = new
        {
            schema_version = BridgeRuntime.StateSchemaVersion,
            screen = fields.Screen,
            run = fields.Run,
            combat = fields.Combat,
            players = fields.Players,
            rewards = fields.Rewards,
            card_reward_selection = fields.CardRewardSelection,
            card_selection = fields.CardSelection,
            character_selection = fields.CharacterSelection,
            run_mode_selection = fields.RunModeSelection,
            event_options = fields.EventOptions,
            crystal_sphere = fields.CrystalSphere,
            map = fields.Map,
            rest_site = fields.RestSite,
            deck_upgrade_selection = fields.DeckUpgradeSelection,
            shop = fields.Shop,
            main_menu = fields.MainMenu
        };

        return PruneSemanticStateNode(JsonSerializer.SerializeToNode(rawCore, HashJsonOptions)) ?? new JsonObject();
    }

    private static object CreateStatePayload(
        BridgeStateFields fields,
        long stateVersion,
        string stateHash,
        string semanticStateHash)
    {
        return new
        {
            ok = true,
            bridge_version = BridgeRuntime.BridgeVersion,
            schema_version = BridgeRuntime.StateSchemaVersion,
            state_version = stateVersion,
            state_hash = stateHash,
            semantic_state_hash = semanticStateHash,
            captured_at_utc = DateTimeOffset.UtcNow,
            screen = fields.Screen,
            automation = fields.Automation,
            run = fields.Run,
            combat = fields.Combat,
            players = fields.Players,
            rewards = fields.Rewards,
            card_reward_selection = fields.CardRewardSelection,
            card_selection = fields.CardSelection,
            character_selection = fields.CharacterSelection,
            run_mode_selection = fields.RunModeSelection,
            event_options = fields.EventOptions,
            crystal_sphere = fields.CrystalSphere,
            map = fields.Map,
            rest_site = fields.RestSite,
            deck_upgrade_selection = fields.DeckUpgradeSelection,
            shop = fields.Shop,
            main_menu = fields.MainMenu,
            available_actions = fields.AvailableActions
        };
    }

    private static List<BridgeResolvedAction> BuildResolvedActions(BridgeWorldContext context)
    {
        var actions = new List<BridgeResolvedAction>();

        AddAutomationActions(actions, context);
        AddRunModeActions(actions, context);
        AddMainMenuActions(actions, context);
        AddGameOverActions(actions, context);
        AddDeckUpgradeActions(actions, context);
        AddCardSelectionActions(actions, context);
        AddRestSiteActions(actions, context);
        AddShopActions(actions, context);
        AddTreasureRoomActions(actions, context);

        for (var index = 0; index < context.CharacterButtons.Count; index++)
        {
            var button = context.CharacterButtons[index];
            if (!IsNodeVisible(button) || button.IsLocked || ReferenceEquals(button, context.SelectedCharacterButton))
            {
                continue;
            }

            var actionId = $"character_select:{index}";
            actions.Add(new BridgeResolvedAction
            {
                ActionId = actionId,
                Payload = new
                {
                    action_id = actionId,
                    kind = "character_select",
                    index,
                    label = $"Select character {index}: {DescribeCharacter(button.Character)}",
                    character = BuildCharacterPayload(button.Character),
                    is_random = button.IsRandom,
                    screen = context.Screen
                },
                Execute = () => InvokeButtonAction(button, "Select", "OnPress")
            });
        }

        if (context.EmbarkButton is not null &&
            IsNodeVisible(context.EmbarkButton) &&
            IsButtonEnabled(context.EmbarkButton))
        {
            actions.Add(new BridgeResolvedAction
            {
                ActionId = "embark",
                Payload = new
                {
                    action_id = "embark",
                    kind = "character_select",
                    label = "Embark",
                    selected_character = BuildCharacterPayload(context.SelectedCharacterButton?.Character),
                    screen = context.Screen
                },
                Execute = () => InvokeButtonAction(context.EmbarkButton, "ForceClick", "OnRelease")
            });
        }

        if (context.CombatManager?.IsInProgress == true &&
            !IsCardSelectionVisible(context) &&
            !context.CombatManager.PlayerActionsDisabled)
        {
            AddCombatCardActions(actions, context);
            AddCombatPotionActions(actions, context);
        }

        if (context.CombatManager?.IsInProgress == true &&
            !IsCardSelectionVisible(context) &&
            !context.CombatManager.PlayerActionsDisabled &&
            context.EndTurnButton is not null &&
            IsNodeVisible(context.EndTurnButton))
        {
            actions.Add(new BridgeResolvedAction
            {
                ActionId = "end_turn",
                Payload = new
                {
                    action_id = "end_turn",
                    kind = "combat",
                    label = "End Turn",
                    screen = context.Screen
                },
                Execute = () => InvokeButtonAction(context.EndTurnButton, "OnRelease", "CallReleaseLogic")
            });
        }

        AddPotionDiscardActions(actions, context);

        if (IsTerminalRewardsProceedVisible(context))
        {
            actions.Add(new BridgeResolvedAction
            {
                ActionId = "proceed",
                Payload = new
                {
                    action_id = "proceed",
                    kind = "proceed",
                    label = "Proceed from terminal rewards",
                    is_skip = false,
                    proceed_source = "terminal_rewards",
                    screen = context.Screen
                },
                Execute = () => InvokeTerminalRewardsProceed(
                    context.RunManager,
                    context.RewardsScreen,
                    context.RewardProceedButton)
            });
        }
        else if (context.MapScreen?.IsOpen != true &&
                 context.ProceedButton is not null &&
                 IsNodeVisible(context.ProceedButton) &&
                 IsButtonEnabled(context.ProceedButton) &&
                 !ShouldSuppressGenericRoomProceed(context))
        {
            var label = context.ProceedButton.IsSkip ? "Skip" : "Proceed";
            actions.Add(new BridgeResolvedAction
            {
                ActionId = "proceed",
                Payload = new
                {
                    action_id = "proceed",
                    kind = "proceed",
                    label,
                    is_skip = context.ProceedButton.IsSkip,
                    proceed_source = "room",
                    screen = context.Screen
                },
                Execute = () => InvokeRoomProceedAction(context)
            });
        }

        if (!IsCardRewardSelectionVisible(context.CardRewardScreen, context.CardRewardOptions))
        {
            for (var index = 0; index < context.RewardButtons.Count; index++)
            {
                var button = context.RewardButtons[index];
                if (!IsNodeVisible(button))
                {
                    continue;
                }

                var rewardDescription = DescribeReward(button.Reward);
                var actionId = $"reward:{index}";

                actions.Add(new BridgeResolvedAction
                {
                    ActionId = actionId,
                    Payload = new
                    {
                        action_id = actionId,
                        kind = "reward",
                        index,
                        label = $"Claim reward {index}: {rewardDescription}",
                        reward = BuildRewardPayload(button.Reward),
                        screen = context.Screen
                    },
                    Execute = () => InvokeButtonAction(button, "OnRelease")
                });
            }
        }

        if (context.CardRewardScreen is not null &&
            IsCardRewardSelectionReady(context.CardRewardScreen))
        {
            for (var index = 0; index < context.CardRewardOptions.Count; index++)
            {
                var cardHolder = context.CardRewardOptions[index];
                if (!IsNodeVisible(cardHolder))
                {
                    continue;
                }

                var actionId = $"card_reward:{index}";
                actions.Add(new BridgeResolvedAction
                {
                    ActionId = actionId,
                    Payload = new
                    {
                        action_id = actionId,
                        kind = "card_reward",
                        index,
                        label = $"Pick card {index}: {cardHolder.CardModel?.Title ?? "<missing>"}",
                        card = BuildCardPayload(cardHolder.CardModel),
                        screen = context.Screen
                    },
                    Execute = () => InvokeSingleArgumentAction(context.CardRewardScreen, "SelectCard", cardHolder)
                });
            }

            if (context.CardRewardSkipButton is not null &&
                IsNodeVisible(context.CardRewardSkipButton) &&
                IsButtonEnabled(context.CardRewardSkipButton))
            {
                actions.Add(new BridgeResolvedAction
                {
                    ActionId = "card_reward:skip",
                    Payload = new
                    {
                        action_id = "card_reward:skip",
                        kind = "card_reward",
                        selection_action = "skip",
                        label = "Skip card reward",
                        screen = context.Screen
                    },
                    Execute = () => InvokeCardRewardSkipAction(
                        context.CardRewardScreen,
                        context.CardRewardSkipButton)
                });
            }
        }

        if (context.MapScreen is null || !context.MapScreen.IsOpen)
        {
            for (var index = 0; index < context.EventOptionButtons.Count; index++)
            {
                var button = context.EventOptionButtons[index];
                var option = button.Option;
                if (!IsNodeVisible(button) || option is null || option.IsLocked)
                {
                    continue;
                }

                var actionId = $"event_option:{index}";
                actions.Add(new BridgeResolvedAction
                {
                    ActionId = actionId,
                    Payload = new
                    {
                        action_id = actionId,
                        kind = "event_option",
                        index,
                        label = $"Choose option {index}: {TextOf(option.Title)}",
                        option = BuildEventOptionPayload(
                            button,
                            index,
                            GetHiddenFieldValue(context.EventRoom, "_event") as EventModel),
                        screen = context.Screen
                    },
                    Execute = () => InvokeButtonAction(button, "OnRelease")
                });
            }

            AddCrystalSphereEventActions(actions, context, context.EventOptionButtons.Count);
        }

        if (context.MapScreen is not null &&
            context.MapScreen.IsOpen &&
            context.MapScreen.IsTravelEnabled &&
            !context.MapScreen.IsTraveling)
        {
            foreach (var pointNode in context.MapPoints)
            {
                if (!IsMapPointTravelable(pointNode))
                {
                    continue;
                }

                var coord = pointNode.Point.coord;
                var actionId = $"map:{coord.col},{coord.row}";
                actions.Add(new BridgeResolvedAction
                {
                    ActionId = actionId,
                    Payload = new
                    {
                        action_id = actionId,
                        kind = "map",
                        label = $"Travel to ({coord.col}, {coord.row}) {pointNode.Point.PointType}",
                        coord = BuildMapCoord(coord),
                        point_type = pointNode.Point.PointType.ToString(),
                        state = pointNode.State.ToString(),
                        screen = context.Screen
                    },
                    Execute = () => InvokeButtonAction(pointNode, "OnRelease")
                });
            }
        }

        return actions;
    }

    private static void AddTreasureRoomActions(List<BridgeResolvedAction> actions, BridgeWorldContext context)
    {
        if (context.TreasureRoom is null || !IsNodeVisible(context.TreasureRoom))
        {
            return;
        }

        if (CanOpenTreasureChest(context))
        {
            var chestButton = context.TreasureChestButton!;
            actions.Add(new BridgeResolvedAction
            {
                ActionId = "treasure:open",
                Payload = new
                {
                    action_id = "treasure:open",
                    kind = "treasure",
                    label = "Open treasure chest",
                    screen = context.Screen
                },
                Execute = () => InvokeTreasureChestAction(context.TreasureRoom, chestButton)
            });
        }

        for (var index = 0; index < context.TreasureRelicOptions.Count; index++)
        {
            var holder = context.TreasureRelicOptions[index];
            if (!IsNodeVisible(holder))
            {
                continue;
            }

            var actionId = $"treasure_relic:{index}";
            actions.Add(new BridgeResolvedAction
            {
                ActionId = actionId,
                Payload = new
                {
                    action_id = actionId,
                    kind = "treasure_relic",
                    index,
                    label = $"Pick treasure relic {index}: {TextOf(holder.Relic?.Model?.Title)}",
                    relic = BuildRelicPayload(holder.Relic?.Model),
                    screen = context.Screen
                },
                Execute = () => InvokeTreasureRelicAction(context.TreasureRelicCollection, holder)
            });
        }
    }

    private static void AddRunModeActions(List<BridgeResolvedAction> actions, BridgeWorldContext context)
    {
        if (!IsRunModeSelectionVisible(context))
        {
            return;
        }

        AddRunModeAction(
            actions,
            context,
            context.RunModeStandardButton,
            "standard",
            "Start standard run",
            "OpenCharacterSelect");
        AddRunModeAction(
            actions,
            context,
            context.RunModeDailyButton,
            "daily",
            "Open daily challenge",
            "OpenDailyScreen");
        AddRunModeAction(
            actions,
            context,
            context.RunModeCustomButton,
            "custom",
            "Open custom run setup",
            "OpenCustomScreen");

        if (context.RunModeBackButton is null || !IsNodeVisible(context.RunModeBackButton))
        {
            return;
        }

        actions.Add(new BridgeResolvedAction
        {
            ActionId = "run_mode:back",
            Payload = new
            {
                action_id = "run_mode:back",
                kind = "run_mode_selection",
                run_mode_action = "back",
                button_text = TryGetNodeText(context.RunModeBackButton),
                label = "Back",
                screen = context.Screen
            },
            Execute = () => InvokeMenuButtonAction(context.RunModeBackButton)
        });
    }

    private static void AddRestSiteActions(List<BridgeResolvedAction> actions, BridgeWorldContext context)
    {
        if (IsDeckUpgradeSelectionVisible(context))
        {
            return;
        }

        if (context.MapScreen?.IsOpen == true)
        {
            return;
        }

        if (context.RestSiteRoom is null || !IsNodeVisible(context.RestSiteRoom))
        {
            return;
        }

        var canProceed = !HasVisibleEnabledRestSiteOptions(context.RestSiteButtons);

        for (var index = 0; index < context.RestSiteButtons.Count; index++)
        {
            var button = context.RestSiteButtons[index];
            var option = button.Option;
            if (!IsNodeVisible(button) || option is null || !option.IsEnabled)
            {
                continue;
            }

            var actionId = $"rest_site:{index}";
            actions.Add(new BridgeResolvedAction
            {
                ActionId = actionId,
                Payload = new
                {
                    action_id = actionId,
                    kind = "rest_site",
                    index,
                    label = $"Rest site option {index}: {TextOf(option.Title)}",
                    option = BuildRestSiteOptionPayload(option, index),
                    screen = context.Screen
                },
                Execute = () => InvokeClickablePressAndRelease(button)
            });
        }

        if (canProceed &&
            context.RestSiteProceedButton is not null &&
            IsNodeVisible(context.RestSiteProceedButton) &&
            IsButtonEnabled(context.RestSiteProceedButton))
        {
            actions.Add(new BridgeResolvedAction
            {
                ActionId = "rest_site:proceed",
                Payload = new
                {
                    action_id = "rest_site:proceed",
                    kind = "rest_site",
                    label = "Rest site proceed",
                    screen = context.Screen
                },
                Execute = () => InvokeRestSiteProceedAction(context.RestSiteRoom, context.RestSiteProceedButton)
            });
        }
    }

    private static void AddDeckUpgradeActions(List<BridgeResolvedAction> actions, BridgeWorldContext context)
    {
        if (!IsDeckUpgradeSelectionVisible(context) || context.DeckUpgradeScreen is null)
        {
            return;
        }

        for (var index = 0; index < context.DeckUpgradeOptions.Count; index++)
        {
            var cardHolder = context.DeckUpgradeOptions[index];
            if (!IsNodeVisible(cardHolder) || cardHolder.CardModel is null)
            {
                continue;
            }

            var actionId = $"deck_upgrade:select:{index}";
            actions.Add(new BridgeResolvedAction
            {
                ActionId = actionId,
                Payload = new
                {
                    action_id = actionId,
                    kind = "deck_upgrade",
                    upgrade_action = "select_card",
                    selection_semantics = "upgrade",
                    selection_prompt = TryGetDeckUpgradePrompt(context.DeckUpgradeScreen),
                    index,
                    label = $"Select upgrade card {index}: {cardHolder.CardModel.Title}",
                    card = BuildCardPayload(cardHolder.CardModel),
                    screen = context.Screen
                },
                Execute = () => InvokeSingleArgumentAction(context.DeckUpgradeScreen, "OnCardClicked", cardHolder.CardModel)
            });
        }

        if (context.DeckUpgradeConfirmButton is not null &&
            IsNodeVisible(context.DeckUpgradeConfirmButton) &&
            IsButtonEnabled(context.DeckUpgradeConfirmButton))
        {
            actions.Add(new BridgeResolvedAction
            {
                ActionId = "deck_upgrade:confirm",
                Payload = new
                {
                    action_id = "deck_upgrade:confirm",
                    kind = "deck_upgrade",
                    upgrade_action = "confirm",
                    selection_semantics = "upgrade",
                    selection_prompt = TryGetDeckUpgradePrompt(context.DeckUpgradeScreen),
                    label = "Confirm upgrade selection",
                    screen = context.Screen
                },
                Execute = () => InvokeSingleArgumentAction(
                    context.DeckUpgradeScreen,
                    "ConfirmSelection",
                    context.DeckUpgradeConfirmButton)
            });
        }

        if (context.DeckUpgradeCancelButton is not null &&
            IsNodeVisible(context.DeckUpgradeCancelButton) &&
            IsButtonEnabled(context.DeckUpgradeCancelButton))
        {
            actions.Add(new BridgeResolvedAction
            {
                ActionId = "deck_upgrade:cancel",
                Payload = new
                {
                    action_id = "deck_upgrade:cancel",
                    kind = "deck_upgrade",
                    upgrade_action = "cancel",
                    selection_semantics = "upgrade",
                    selection_prompt = TryGetDeckUpgradePrompt(context.DeckUpgradeScreen),
                    label = "Cancel upgrade selection",
                    screen = context.Screen
                },
                Execute = () => InvokeSingleArgumentAction(
                    context.DeckUpgradeScreen,
                    "CancelSelection",
                    context.DeckUpgradeCancelButton)
            });
        }

        if (context.DeckUpgradeCloseButton is not null &&
            IsNodeVisible(context.DeckUpgradeCloseButton) &&
            IsButtonEnabled(context.DeckUpgradeCloseButton))
        {
            actions.Add(new BridgeResolvedAction
            {
                ActionId = "deck_upgrade:close",
                Payload = new
                {
                    action_id = "deck_upgrade:close",
                    kind = "deck_upgrade",
                    upgrade_action = "close",
                    selection_semantics = "upgrade",
                    selection_prompt = TryGetDeckUpgradePrompt(context.DeckUpgradeScreen),
                    label = "Close upgrade selection",
                    screen = context.Screen
                },
                Execute = () => InvokeSingleArgumentAction(
                    context.DeckUpgradeScreen,
                    "CloseSelection",
                    context.DeckUpgradeCloseButton)
            });
        }
    }

    private static void AddCardSelectionActions(List<BridgeResolvedAction> actions, BridgeWorldContext context)
    {
        if (!IsCardSelectionVisible(context))
        {
            return;
        }

        var selectionPrompt = TryGetCardSelectionPrompt(context.CardSelectionScreen);
        var selectionTexts = context.CardSelectionScreen is not null && IsNodeVisible(context.CardSelectionScreen)
            ? CollectVisibleText(context.CardSelectionScreen, 8).ToArray()
            : Array.Empty<string>();
        var selectionSemantics = ResolveCardSelectionSemantics(context.CardSelectionScreen, selectionPrompt, selectionTexts);

        if (context.CardSelectionScreen is NChooseABundleSelectionScreen)
        {
            var bundleOptions = GetCardSelectionBundles(context.CardSelectionScreen);
            for (var index = 0; index < bundleOptions.Count; index++)
            {
                var bundle = bundleOptions[index];
                if (!IsNodeVisible(bundle))
                {
                    continue;
                }

                var actionId = $"card_selection:select:{index}";
                actions.Add(new BridgeResolvedAction
                {
                    ActionId = actionId,
                    Payload = new
                    {
                        action_id = actionId,
                        kind = "card_selection",
                        selection_action = "select",
                        selection_semantics = selectionSemantics,
                        selection_prompt = selectionPrompt,
                        index,
                        label = $"Select bundle {index}",
                        bundle = bundle.Bundle.Select(card => BuildCardPayload(card)).ToArray(),
                        screen = context.Screen,
                        screen_type = context.CardSelectionScreen.GetType().Name
                    },
                    Execute = () => InvokeCardSelectionBundleAction(context.CardSelectionScreen, bundle)
                });
            }
        }
        else
        {
            for (var index = 0; index < context.CardSelectionOptions.Count; index++)
            {
                var cardHolder = context.CardSelectionOptions[index];
                if (!IsNodeVisible(cardHolder))
                {
                    continue;
                }

                var optionIndex = GetCardSelectionOptionIndex(context.CardSelectionScreen, cardHolder, index);
                var selectionId = GetCardSelectionOptionSelectionId(context.CardSelectionScreen, cardHolder, optionIndex);
                var actionId = selectionId is not null
                    ? $"card_selection:select:{selectionId}"
                    : $"card_selection:select:{optionIndex}";
                actions.Add(new BridgeResolvedAction
                {
                    ActionId = actionId,
                    Payload = new
                    {
                        action_id = actionId,
                        kind = "card_selection",
                        selection_action = "select",
                        selection_semantics = selectionSemantics,
                        selection_prompt = selectionPrompt,
                        index = optionIndex,
                        selection_id = selectionId,
                        label = $"Select card {optionIndex}: {cardHolder.CardModel?.Title ?? "<missing>"}",
                        card = BuildCardPayload(cardHolder.CardModel),
                        screen = context.Screen,
                        screen_type = context.CardSelectionScreen?.GetType().Name
                    },
                    Execute = () => InvokeCardSelectionOptionAction(context.CardSelectionScreen, cardHolder)
                });
            }
        }

        if (context.CardSelectionConfirmButton is not null &&
            IsNodeVisible(context.CardSelectionConfirmButton) &&
            IsButtonEnabled(context.CardSelectionConfirmButton))
        {
            actions.Add(new BridgeResolvedAction
            {
                ActionId = "card_selection:confirm",
                Payload = new
                {
                    action_id = "card_selection:confirm",
                    kind = "card_selection",
                    selection_action = "confirm",
                    selection_semantics = selectionSemantics,
                    selection_prompt = selectionPrompt,
                    label = "Confirm selected cards",
                    screen = context.Screen,
                    screen_type = context.CardSelectionScreen?.GetType().Name
                },
                Execute = () => InvokeCardSelectionConfirmAction(
                    context.CardSelectionScreen,
                    context.CardSelectionConfirmButton)
            });
        }

        if (context.CardSelectionCancelButton is not null &&
            IsNodeVisible(context.CardSelectionCancelButton) &&
            IsButtonEnabled(context.CardSelectionCancelButton))
        {
            actions.Add(new BridgeResolvedAction
            {
                ActionId = "card_selection:cancel",
                Payload = new
                {
                    action_id = "card_selection:cancel",
                    kind = "card_selection",
                    selection_action = "cancel",
                    selection_semantics = selectionSemantics,
                    selection_prompt = selectionPrompt,
                    label = "Cancel card selection preview",
                    screen = context.Screen,
                    screen_type = context.CardSelectionScreen?.GetType().Name
                },
                Execute = () => InvokeCardSelectionCancelAction(
                    context.CardSelectionScreen,
                    context.CardSelectionCancelButton)
            });
        }

        if (context.CardSelectionCloseButton is not null &&
            IsNodeVisible(context.CardSelectionCloseButton) &&
            IsButtonEnabled(context.CardSelectionCloseButton))
        {
            actions.Add(new BridgeResolvedAction
            {
                ActionId = "card_selection:close",
                Payload = new
                {
                    action_id = "card_selection:close",
                    kind = "card_selection",
                    selection_action = "close",
                    selection_semantics = selectionSemantics,
                    selection_prompt = selectionPrompt,
                    label = "Close card selection",
                    screen = context.Screen,
                    screen_type = context.CardSelectionScreen?.GetType().Name
                },
                Execute = () => InvokeCardSelectionCloseAction(
                    context.CardSelectionScreen,
                    context.CardSelectionCloseButton)
            });
        }

        if (context.CardSelectionSkipButton is not null &&
            IsNodeVisible(context.CardSelectionSkipButton) &&
            IsButtonEnabled(context.CardSelectionSkipButton))
        {
            actions.Add(new BridgeResolvedAction
            {
                ActionId = "card_selection:skip",
                Payload = new
                {
                    action_id = "card_selection:skip",
                    kind = "card_selection",
                    selection_action = "skip",
                    selection_semantics = selectionSemantics,
                    selection_prompt = selectionPrompt,
                    label = "Skip card selection",
                    screen = context.Screen,
                    screen_type = context.CardSelectionScreen?.GetType().Name
                },
                Execute = () => InvokeCardSelectionSkipAction(
                    context.CardSelectionScreen,
                    context.CardSelectionSkipButton)
            });
        }
    }

    private static void AddShopActions(List<BridgeResolvedAction> actions, BridgeWorldContext context)
    {
        if (context.MerchantRoom is null ||
            (!IsNodeVisible(context.MerchantRoom) && context.MerchantInventory?.IsOpen != true))
        {
            return;
        }

        var inventoryIsOpen = context.MerchantInventory?.IsOpen == true;

        if (!inventoryIsOpen &&
            context.MerchantButton is not null &&
            IsNodeVisible(context.MerchantButton) &&
            IsButtonEnabled(context.MerchantButton))
        {
            actions.Add(new BridgeResolvedAction
            {
                ActionId = "shop:open",
                Payload = new
                {
                    action_id = "shop:open",
                    kind = "shop",
                    shop_action = "open",
                    label = "Open merchant inventory",
                    screen = context.Screen
                },
                Execute = () => InvokeButtonAction(context.MerchantButton, "OnRelease", "OnPress")
            });
        }

        if (inventoryIsOpen && context.MerchantInventory is not null)
        {
            for (var index = 0; index < context.MerchantSlots.Count; index++)
            {
                var slot = context.MerchantSlots[index];
                var entry = slot.Entry;
                if (!IsNodeVisible(slot) || !CanPurchaseMerchantEntry(entry))
                {
                    continue;
                }

                var actionId = $"shop:buy:{index}";
                actions.Add(new BridgeResolvedAction
                {
                    ActionId = actionId,
                    Payload = new
                    {
                        action_id = actionId,
                        kind = "shop",
                        shop_action = "buy",
                        index,
                        label = $"Buy shop item {index}: {DescribeMerchantEntry(entry)}",
                        item = BuildShopSlotPayload(slot, index),
                        screen = context.Screen
                    },
                    Execute = () => ExecuteShopPurchase(slot, context.MerchantInventory)
                });
            }
        }

        if (context.MerchantBackButton is not null &&
            IsNodeVisible(context.MerchantBackButton) &&
            IsButtonEnabled(context.MerchantBackButton))
        {
            actions.Add(new BridgeResolvedAction
            {
                ActionId = "shop:back",
                Payload = new
                {
                    action_id = "shop:back",
                    kind = "shop",
                    shop_action = "back",
                    label = "Close merchant inventory",
                    screen = context.Screen
                },
                Execute = () => InvokeMerchantBackAction(context.MerchantInventory, context.MerchantBackButton)
            });
        }

        if (context.MerchantProceedButton is not null &&
            IsNodeVisible(context.MerchantProceedButton) &&
            IsButtonEnabled(context.MerchantProceedButton))
        {
            actions.Add(new BridgeResolvedAction
            {
                ActionId = "shop:leave",
                Payload = new
                {
                    action_id = "shop:leave",
                    kind = "shop",
                    shop_action = "leave",
                    label = "Leave shop",
                    screen = context.Screen
                },
                Execute = () => InvokeMerchantLeaveAction(context.MerchantRoom, context.MerchantProceedButton)
            });
        }
    }

    private static void AddAutomationActions(List<BridgeResolvedAction> actions, BridgeWorldContext context)
    {
        var automationPayload = BuildAutomationPayload();
        if (BridgeAutoSlay.IsActive)
        {
            actions.Add(new BridgeResolvedAction
            {
                ActionId = "automation:stop_autoslay",
                Payload = new
                {
                    action_id = "automation:stop_autoslay",
                    kind = "automation",
                    automation_action = "stop_autoslay",
                    label = "Stop AutoSlay",
                    automation = automationPayload,
                    screen = context.Screen
                },
                Execute = BridgeAutoSlay.Stop
            });

            return;
        }

        actions.Add(new BridgeResolvedAction
        {
            ActionId = "automation:start_autoslay",
            Payload = new
            {
                action_id = "automation:start_autoslay",
                kind = "automation",
                automation_action = "start_autoslay",
                label = "Start AutoSlay",
                automation = automationPayload,
                screen = context.Screen
            },
            Execute = () => BridgeAutoSlay.Start()
        });
    }

    private static void AddMainMenuActions(List<BridgeResolvedAction> actions, BridgeWorldContext context)
    {
        if (IsRunModeSelectionVisible(context))
        {
            return;
        }

        if (context.AbandonRunConfirmPopup is not null && IsNodeVisible(context.AbandonRunConfirmPopup))
        {
            for (var index = 0; index < context.AbandonRunConfirmButtons.Count; index++)
            {
                var button = context.AbandonRunConfirmButtons[index];
                if (!IsNodeVisible(button))
                {
                    continue;
                }

                var buttonText = TryGetNodeText(button);
                var semanticAction = TryGetAbandonConfirmSemanticAction(buttonText);
                var actionId = semanticAction switch
                {
                    "confirm" => "main_menu:confirm_abandon_run",
                    "cancel" => "main_menu:cancel_abandon_run",
                    _ => $"main_menu:abandon_confirm:{index}"
                };
                var label = semanticAction switch
                {
                    "confirm" => "Confirm abandon current game",
                    "cancel" => "Cancel abandon current game",
                    _ when !string.IsNullOrWhiteSpace(buttonText) => $"Abandon confirmation: {buttonText}",
                    _ => $"Abandon confirmation button {index}"
                };

                actions.Add(new BridgeResolvedAction
                {
                    ActionId = actionId,
                    Payload = new
                    {
                        action_id = actionId,
                        kind = "main_menu",
                        menu_action = semanticAction ?? "abandon_confirm_button",
                        button_index = index,
                        button_text = buttonText,
                        label,
                        screen = context.Screen
                    },
                    Execute = () => InvokeMenuButtonAction(button)
                });
            }

            return;
        }

        if (context.MainMenuContinueButton is not null && IsNodeVisible(context.MainMenuContinueButton))
        {
            var buttonText = TryGetNodeText(context.MainMenuContinueButton);
            actions.Add(new BridgeResolvedAction
            {
                ActionId = "main_menu:continue",
                Payload = new
                {
                    action_id = "main_menu:continue",
                    kind = "main_menu",
                    menu_action = "continue",
                    button_text = buttonText,
                    label = !string.IsNullOrWhiteSpace(buttonText) ? buttonText : "Continue Game",
                    screen = context.Screen
                },
                Execute = () => InvokeMenuButtonAction(context.MainMenuContinueButton)
            });
        }

        for (var index = 0; index < context.MainMenuTextButtons.Count; index++)
        {
            var button = context.MainMenuTextButtons[index];
            if (!IsNodeVisible(button))
            {
                continue;
            }

            var buttonText = TryGetNodeText(button);
            var semanticAction = TryGetMainMenuSemanticAction(buttonText);
            var actionId = semanticAction is not null
                ? $"main_menu:{semanticAction}"
                : $"main_menu:button:{index}";
            var label = !string.IsNullOrWhiteSpace(buttonText)
                ? buttonText
                : $"Main menu button {index}";

            if (actions.Any(existing => existing.ActionId.Equals(actionId, StringComparison.Ordinal)))
            {
                actionId = $"main_menu:button:{index}";
            }

            actions.Add(new BridgeResolvedAction
            {
                ActionId = actionId,
                Payload = new
                {
                    action_id = actionId,
                    kind = "main_menu",
                    menu_action = semanticAction ?? "button",
                    button_index = index,
                    button_text = buttonText,
                    label,
                    screen = context.Screen
                },
                Execute = () => InvokeMenuButtonAction(button)
            });
        }
    }

    private static void AddGameOverActions(List<BridgeResolvedAction> actions, BridgeWorldContext context)
    {
        if (context.GameOverScreen is null || !IsNodeVisible(context.GameOverScreen))
        {
            return;
        }

        if (context.GameOverContinueButton is not null && IsNodeVisible(context.GameOverContinueButton))
        {
            actions.Add(new BridgeResolvedAction
            {
                ActionId = "game_over:continue",
                Payload = new
                {
                    action_id = "game_over:continue",
                    kind = "game_over",
                    game_over_action = "continue",
                    label = "Continue from game-over summary",
                    screen = context.Screen
                },
                Execute = () => InvokeGameOverContinueAction(
                    context.GameOverScreen,
                    context.GameOverContinueButton)
            });
        }

        if (context.GameOverMainMenuButton is not null && IsNodeVisible(context.GameOverMainMenuButton))
        {
            actions.Add(new BridgeResolvedAction
            {
                ActionId = "game_over:return_to_main_menu",
                Payload = new
                {
                    action_id = "game_over:return_to_main_menu",
                    kind = "game_over",
                    game_over_action = "return_to_main_menu",
                    label = "Return to main menu",
                    screen = context.Screen
                },
                Execute = () => InvokeGameOverReturnToMainMenuAction(
                    context.GameOverScreen,
                    context.GameOverMainMenuButton)
            });
        }
    }

    private static void AddRunModeAction(
        List<BridgeResolvedAction> actions,
        BridgeWorldContext context,
        Node? button,
        string actionSuffix,
        string fallbackLabel,
        string submenuMethodName)
    {
        if (button is null || !IsNodeVisible(button))
        {
            return;
        }

        var texts = CollectVisibleText(button, 4).ToArray();
        var buttonText = texts.FirstOrDefault(static text => !string.IsNullOrWhiteSpace(text)) ?? fallbackLabel;
        var actionId = $"run_mode:{actionSuffix}";

        actions.Add(new BridgeResolvedAction
        {
            ActionId = actionId,
            Payload = new
            {
                action_id = actionId,
                kind = "run_mode_selection",
                run_mode_action = actionSuffix,
                button_text = buttonText,
                texts,
                label = fallbackLabel,
                screen = context.Screen
            },
            Execute = () => InvokeRunModeSelectionAction(context.RunModeSubmenu, button, submenuMethodName)
        });
    }

    private static void AddCombatCardActions(List<BridgeResolvedAction> actions, BridgeWorldContext context)
    {
        if (context.CombatState is null)
        {
            return;
        }

        for (var playerIndex = 0; playerIndex < context.CombatState.Players.Count; playerIndex++)
        {
            var player = context.CombatState.Players[playerIndex];
            var handCards = player.PlayerCombatState?.Hand?.Cards;
            if (handCards is null)
            {
                continue;
            }

            for (var handIndex = 0; handIndex < handCards.Count; handIndex++)
            {
                var card = handCards[handIndex];
                if (card is null || !CanPlayCard(card))
                {
                    continue;
                }

                var cardRef = GetCardReference(card);
                foreach (var resolvedTarget in ResolvePlayableCardTargets(context, player, card))
                {
                    var actionId = $"play_card:{playerIndex}:{cardRef}";
                    if (!string.IsNullOrEmpty(resolvedTarget.ActionSuffix))
                    {
                        actionId += $":{resolvedTarget.ActionSuffix}";
                    }

                    var targetLabel = string.IsNullOrEmpty(resolvedTarget.LabelSuffix)
                        ? string.Empty
                        : $" -> {BuildResolvedTargetLabel(resolvedTarget.ActionSuffix, resolvedTarget.Target)}";
                    var cardTitle = TextOf(card.Title);
                    var targetMapping = BuildResolvedTargetMapping(resolvedTarget.ActionSuffix, resolvedTarget.Target);

                    actions.Add(new BridgeResolvedAction
                    {
                        ActionId = actionId,
                        Payload = new
                        {
                            action_id = actionId,
                            kind = "play_card",
                            label = $"Play card {handIndex}: {cardTitle}{targetLabel}",
                            player_index = playerIndex,
                            player_net_id = player.NetId,
                            hand_index = handIndex,
                            card_ref = cardRef,
                            card = BuildCardPayload(card, resolvedTarget.Target),
                            target = resolvedTarget.Target is null ? null : BuildCreaturePayload(resolvedTarget.Target),
                            target_action_suffix = resolvedTarget.ActionSuffix,
                            target_combat_id = resolvedTarget.Target?.CombatId,
                            target_name = resolvedTarget.Target?.Name,
                            target_side = resolvedTarget.Target?.Side.ToString(),
                            target_mapping = targetMapping,
                            target_scope = card.TargetType.ToString(),
                            requires_target_selection = resolvedTarget.RequiresTargetSelection,
                            screen = context.Screen
                        },
                        Execute = () => ExecuteCardPlay(card, resolvedTarget.Target)
                    });
                }
            }
        }
    }

    private static void AddCombatPotionActions(List<BridgeResolvedAction> actions, BridgeWorldContext context)
    {
        if (context.CombatState is null)
        {
            return;
        }

        for (var playerIndex = 0; playerIndex < context.CombatState.Players.Count; playerIndex++)
        {
            var player = context.CombatState.Players[playerIndex];
            var potionSlots = player.PotionSlots;
            if (potionSlots is null)
            {
                continue;
            }

            for (var slotIndex = 0; slotIndex < potionSlots.Count; slotIndex++)
            {
                var potion = potionSlots[slotIndex];
                if (potion is null || !CanUsePotion(potion))
                {
                    continue;
                }

                foreach (var resolvedTarget in ResolveUsablePotionTargets(context, player, potion))
                {
                    var actionId = $"use_potion:{playerIndex}:{slotIndex}";
                    if (!string.IsNullOrEmpty(resolvedTarget.ActionSuffix))
                    {
                        actionId += $":{resolvedTarget.ActionSuffix}";
                    }

                    var targetLabel = string.IsNullOrEmpty(resolvedTarget.LabelSuffix)
                        ? string.Empty
                        : $" -> {BuildResolvedTargetLabel(resolvedTarget.ActionSuffix, resolvedTarget.Target)}";
                    var potionTitle = TextOf(potion.Title);
                    var targetMapping = BuildResolvedTargetMapping(resolvedTarget.ActionSuffix, resolvedTarget.Target);

                    actions.Add(new BridgeResolvedAction
                    {
                        ActionId = actionId,
                        Payload = new
                        {
                            action_id = actionId,
                            kind = "use_potion",
                            label = $"Use potion {slotIndex}: {potionTitle}{targetLabel}",
                            player_index = playerIndex,
                            player_net_id = player.NetId,
                            slot_index = slotIndex,
                            potion = BuildPotionPayload(potion),
                            target = resolvedTarget.Target is null ? null : BuildCreaturePayload(resolvedTarget.Target),
                            target_action_suffix = resolvedTarget.ActionSuffix,
                            target_combat_id = resolvedTarget.Target?.CombatId,
                            target_name = resolvedTarget.Target?.Name,
                            target_side = resolvedTarget.Target?.Side.ToString(),
                            target_mapping = targetMapping,
                            target_scope = potion.TargetType.ToString(),
                            requires_target_selection = resolvedTarget.RequiresTargetSelection,
                            screen = context.Screen
                        },
                        Execute = () => ExecutePotionUse(
                            player,
                            slotIndex,
                            potion,
                            resolvedTarget.Target,
                            context.CombatManager?.IsInProgress == true)
                    });
                }
            }
        }
    }

    private static void AddPotionDiscardActions(List<BridgeResolvedAction> actions, BridgeWorldContext context)
    {
        if (IsCardSelectionVisible(context))
        {
            return;
        }

        var players = context.RunState?.Players ?? context.CombatState?.Players ?? Array.Empty<Player>();
        for (var playerIndex = 0; playerIndex < players.Count; playerIndex++)
        {
            var player = players[playerIndex];
            var potionSlots = player.PotionSlots;
            if (potionSlots is null || !player.CanRemovePotions)
            {
                continue;
            }

            for (var slotIndex = 0; slotIndex < potionSlots.Count; slotIndex++)
            {
                var potion = potionSlots[slotIndex];
                if (!CanDiscardPotion(player, potion))
                {
                    continue;
                }

                var actionId = $"discard_potion:{playerIndex}:{slotIndex}";
                var potionTitle = TextOf(potion!.Title);
                actions.Add(new BridgeResolvedAction
                {
                    ActionId = actionId,
                    Payload = new
                    {
                        action_id = actionId,
                        kind = "discard_potion",
                        label = $"Discard potion {slotIndex}: {potionTitle}",
                        player_index = playerIndex,
                        player_net_id = player.NetId,
                        slot_index = slotIndex,
                        potion = BuildPotionPayload(potion),
                        can_remove_potions = player.CanRemovePotions,
                        screen = context.Screen
                    },
                    Execute = () => ExecutePotionDiscard(
                        player,
                        slotIndex,
                        potion,
                        context.CombatManager?.IsInProgress == true)
                });
            }
        }
    }

    private static IReadOnlyList<ResolvedCardTarget> ResolvePlayableCardTargets(
        BridgeWorldContext context,
        Player player,
        CardModel card)
    {
        var results = new List<ResolvedCardTarget>();
        var selfCreature = player.Creature;

        switch (card.TargetType)
        {
            case TargetType.AnyEnemy:
                foreach (var creature in context.CombatState?.Creatures ?? Array.Empty<Creature>())
                {
                    if (!creature.IsEnemy || !IsCombatTargetAvailable(creature) || !CanPlayCardTargeting(card, creature))
                    {
                        continue;
                    }

                    results.Add(new ResolvedCardTarget
                    {
                        ActionSuffix = creature.CombatId.ToString(),
                        LabelSuffix = DescribeCreatureTarget(creature),
                        Target = creature,
                        RequiresTargetSelection = true
                    });
                }
                break;

            case TargetType.AnyPlayer:
            case TargetType.AnyAlly:
                foreach (var creature in context.CombatState?.PlayerCreatures ?? Array.Empty<Creature>())
                {
                    if (!IsCombatTargetAvailable(creature) || !CanPlayCardTargeting(card, creature))
                    {
                        continue;
                    }

                    results.Add(new ResolvedCardTarget
                    {
                        ActionSuffix = creature.CombatId.ToString(),
                        LabelSuffix = DescribeCreatureTarget(creature),
                        Target = creature,
                        RequiresTargetSelection = true
                    });
                }
                break;

            case TargetType.Self:
                if (selfCreature is not null && selfCreature.IsAlive)
                {
                    results.Add(new ResolvedCardTarget
                    {
                        ActionSuffix = "self",
                        LabelSuffix = "self",
                        Target = selfCreature,
                        RequiresTargetSelection = false
                    });
                }
                break;

            case TargetType.None:
            case TargetType.AllEnemies:
            case TargetType.RandomEnemy:
            case TargetType.AllAllies:
            case TargetType.TargetedNoCreature:
            case TargetType.Osty:
            default:
                results.Add(new ResolvedCardTarget
                {
                    ActionSuffix = null,
                    LabelSuffix = null,
                    Target = null,
                    RequiresTargetSelection = false
                });
                break;
        }

        return results;
    }

    private static IReadOnlyList<ResolvedPotionTarget> ResolveUsablePotionTargets(
        BridgeWorldContext context,
        Player player,
        PotionModel potion)
    {
        var results = new List<ResolvedPotionTarget>();
        var selfCreature = player.Creature;
        var canThrowAtAlly = SafeCanThrowPotionAtAlly(potion);

        void AddResolvedTarget(Creature? target, string? actionSuffix, string? labelSuffix, bool requiresTargetSelection)
        {
            if (target is null)
            {
                if (!results.Any(static existing => existing.Target is null))
                {
                    results.Add(new ResolvedPotionTarget
                    {
                        ActionSuffix = actionSuffix,
                        LabelSuffix = labelSuffix,
                        Target = null,
                        RequiresTargetSelection = requiresTargetSelection
                    });
                }

                return;
            }

            if (results.Any(existing => ReferenceEquals(existing.Target, target)))
            {
                return;
            }

            results.Add(new ResolvedPotionTarget
            {
                ActionSuffix = actionSuffix,
                LabelSuffix = labelSuffix,
                Target = target,
                RequiresTargetSelection = requiresTargetSelection
            });
        }

        switch (potion.TargetType)
        {
            case TargetType.AnyEnemy:
                foreach (var creature in context.CombatState?.Creatures ?? Array.Empty<Creature>())
                {
                    if (!creature.IsEnemy || !CanUsePotionTargeting(potion, creature))
                    {
                        continue;
                    }

                    AddResolvedTarget(
                        creature,
                        creature.CombatId.ToString(),
                        DescribeCreatureTarget(creature),
                        true);
                }

                if (canThrowAtAlly)
                {
                    foreach (var creature in context.CombatState?.PlayerCreatures ?? Array.Empty<Creature>())
                    {
                        if (!CanUsePotionTargeting(potion, creature))
                        {
                            continue;
                        }

                        AddResolvedTarget(
                            creature,
                            creature.CombatId.ToString(),
                            DescribeCreatureTarget(creature),
                            true);
                    }
                }
                break;

            case TargetType.AnyPlayer:
            case TargetType.AnyAlly:
                foreach (var creature in context.CombatState?.PlayerCreatures ?? Array.Empty<Creature>())
                {
                    if (!CanUsePotionTargeting(potion, creature))
                    {
                        continue;
                    }

                    AddResolvedTarget(
                        creature,
                        creature.CombatId.ToString(),
                        DescribeCreatureTarget(creature),
                        true);
                }
                break;

            case TargetType.Self:
                if (selfCreature is not null && CanUsePotionTargeting(potion, selfCreature))
                {
                    AddResolvedTarget(selfCreature, "self", "self", false);
                }
                break;

            case TargetType.None:
            case TargetType.AllEnemies:
            case TargetType.RandomEnemy:
            case TargetType.AllAllies:
            case TargetType.TargetedNoCreature:
            case TargetType.Osty:
            default:
                AddResolvedTarget(null, null, null, false);
                break;
        }

        return results;
    }

    private static void ExecuteCardPlay(CardModel card, Creature? target)
    {
        // Guard: verify target is still alive before executing.
        // During RL training, rapid action execution can cause the target
        // to die between action resolution and execution.
        if (target is not null && !target.IsAlive)
        {
            throw new BridgeRequestException(
                HttpStatusCode.Conflict,
                "play_card_target_dead",
                $"Target '{target.Name}' is no longer alive. Action skipped.");
        }

        // Guard: verify card is still playable
        if (!CanPlayCard(card))
        {
            throw new BridgeRequestException(
                HttpStatusCode.Conflict,
                "play_card_not_playable",
                $"Card '{TextOf(card.Title)}' is no longer playable.");
        }

        var executionTargets = BuildCardExecutionTargets(card, target);
        var tryManualPlayMethod = FindMethod(card.GetType(), "TryManualPlay", 1);
        if (tryManualPlayMethod is not null)
        {
            foreach (var executionTarget in executionTargets)
            {
                var result = tryManualPlayMethod.Invoke(card, new object?[] { executionTarget });
                if (result is bool success && success)
                {
                    return;
                }
            }
        }

        var enqueueManualPlayMethod = FindMethod(card.GetType(), "EnqueueManualPlay", 1);
        if (enqueueManualPlayMethod is not null)
        {
            enqueueManualPlayMethod.Invoke(card, new object?[] { executionTargets[0] });
            return;
        }

        throw new BridgeRequestException(
            HttpStatusCode.Conflict,
            "play_card_failed",
            $"Could not play card '{TextOf(card.Title)}' with the current bridge integration.");
    }

    private static void ExecutePotionUse(
        Player player,
        int slotIndex,
        PotionModel potion,
        Creature? target,
        bool isCombatInProgress)
    {
        try
        {
            potion.EnqueueManualUse(target!);
            return;
        }
        catch
        {
        }

        try
        {
            var action = new UsePotionAction(potion, target!, isCombatInProgress);
            ExecuteGameActionSynchronously(action);
            return;
        }
        catch
        {
        }

        throw new BridgeRequestException(
            HttpStatusCode.Conflict,
            "use_potion_failed",
            $"Could not use potion '{TextOf(potion.Title)}' from slot {slotIndex}.");
    }

    private static void ExecutePotionDiscard(
        Player player,
        int slotIndex,
        PotionModel potion,
        bool isCombatInProgress)
    {
        try
        {
            potion.Discard();
            return;
        }
        catch
        {
        }

        try
        {
            var action = new DiscardPotionGameAction(player, (uint)slotIndex, isCombatInProgress);
            ExecuteGameActionSynchronously(action);
            return;
        }
        catch
        {
        }

        throw new BridgeRequestException(
            HttpStatusCode.Conflict,
            "discard_potion_failed",
            $"Could not discard potion '{TextOf(potion.Title)}' from slot {slotIndex}.");
    }

    private static IReadOnlyList<Creature?> BuildCardExecutionTargets(CardModel card, Creature? target)
    {
        var targets = new List<Creature?>();

        void AddTarget(Creature? candidate)
        {
            if (candidate is null)
            {
                if (!targets.Any(static existing => existing is null))
                {
                    targets.Add(null);
                }

                return;
            }

            if (!targets.Any(existing => ReferenceEquals(existing, candidate)))
            {
                targets.Add(candidate);
            }
        }

        switch (card.TargetType)
        {
            case TargetType.Self:
            case TargetType.None:
            case TargetType.AllEnemies:
            case TargetType.RandomEnemy:
            case TargetType.AllAllies:
            case TargetType.TargetedNoCreature:
            case TargetType.Osty:
                AddTarget(null);
                AddTarget(target);
                break;

            default:
                AddTarget(target);
                AddTarget(null);
                break;
        }

        return targets;
    }

    private static bool CanPlayCard(CardModel card)
    {
        if (TryInvokeBoolean(card, "CanPlay") is bool canPlay)
        {
            return canPlay;
        }

        return GetHiddenPropertyValue<bool>(card, "IsPlayable") ?? false;
    }

    private static bool CanPlayCardTargeting(CardModel card, Creature target)
    {
        if (TryInvokeBoolean(card, "CanPlayTargeting", target) is bool canPlayTargeting)
        {
            return canPlayTargeting;
        }

        if (TryInvokeBoolean(card, "IsValidTarget", target) is bool isValidTarget)
        {
            return isValidTarget;
        }

        return false;
    }

    private static bool IsCombatTargetAvailable(Creature creature)
    {
        return creature.IsAlive && SafeGetCreatureIsHittable(creature);
    }

    private static bool IsPotionTargetAvailable(Creature creature)
    {
        return creature.IsEnemy
            ? creature.IsAlive && SafeGetCreatureIsHittable(creature)
            : creature.IsAlive;
    }

    private static bool CanUsePotion(PotionModel? potion)
    {
        if (potion is null)
        {
            return false;
        }

        try
        {
            return potion.Owner is not null &&
                   !potion.HasBeenRemovedFromState &&
                   !potion.IsQueued &&
                   potion.PassesCustomUsabilityCheck;
        }
        catch
        {
            return false;
        }
    }

    private static bool CanDiscardPotion(Player player, PotionModel? potion)
    {
        if (potion is null)
        {
            return false;
        }

        try
        {
            return player.CanRemovePotions && !potion.HasBeenRemovedFromState;
        }
        catch
        {
            return false;
        }
    }

    private static bool CanUsePotionTargeting(PotionModel potion, Creature target)
    {
        if (!IsPotionTargetAvailable(target))
        {
            return false;
        }

        if (TryInvokeBoolean(potion, "ShouldAllowTargeting", target) is bool shouldAllowTargeting)
        {
            return shouldAllowTargeting;
        }

        return potion.TargetType switch
        {
            TargetType.Self => ReferenceEquals(target, potion.Owner?.Creature),
            TargetType.AnyEnemy => target.IsEnemy || (!target.IsEnemy && SafeCanThrowPotionAtAlly(potion)),
            TargetType.AnyPlayer or TargetType.AnyAlly => !target.IsEnemy,
            _ => true
        };
    }

    private static bool CanPurchaseMerchantEntry(MerchantEntry? entry)
    {
        return entry is not null &&
               entry.IsStocked &&
               entry.EnoughGold;
    }

    private static void ExecuteShopPurchase(NMerchantSlot slot, NMerchantInventory inventory)
    {
        var merchantInventory = inventory.Inventory;
        var onTryPurchase = FindMethod(slot.GetType(), "OnTryPurchase", 1);
        if (onTryPurchase is not null)
        {
            onTryPurchase.Invoke(slot, new object?[] { merchantInventory });
            return;
        }

        if (slot.Entry is not null)
        {
            var entryMethod = FindMethod(slot.Entry.GetType(), "OnTryPurchase", 2) ??
                              FindMethod(slot.Entry.GetType(), "OnTryPurchaseWrapper", 2);
            if (entryMethod is not null)
            {
                entryMethod.Invoke(slot.Entry, new object?[] { merchantInventory, false });
                return;
            }
        }

        if (TryInvokeParameterless(slot, "OnReleased"))
        {
            return;
        }

        throw new BridgeRequestException(
            HttpStatusCode.Conflict,
            "shop_purchase_failed",
            $"Could not purchase shop item '{DescribeMerchantEntry(slot.Entry)}'.");
    }

    private static string DescribeCreatureTarget(Creature creature)
    {
        return $"{creature.Name} (combat_id {creature.CombatId})";
    }

    private static string BuildResolvedTargetLabel(string? actionSuffix, Creature? target)
    {
        if (target is null)
        {
            return string.IsNullOrWhiteSpace(actionSuffix) ? string.Empty : actionSuffix;
        }

        var prefix = string.IsNullOrWhiteSpace(actionSuffix)
            ? string.Empty
            : $"{actionSuffix} = ";
        return $"{prefix}{DescribeCreatureTarget(target)}";
    }

    private static object? BuildResolvedTargetMapping(string? actionSuffix, Creature? target)
    {
        if (string.IsNullOrWhiteSpace(actionSuffix) && target is null)
        {
            return null;
        }

        return new
        {
            action_suffix = actionSuffix,
            combat_id = target?.CombatId,
            name = target?.Name,
            side = target?.Side.ToString(),
            label = BuildResolvedTargetLabel(actionSuffix, target)
        };
    }

    private static object BuildRunPayload(RunState? runState)
    {
        if (runState is null)
        {
            return new
            {
                has_run = false
            };
        }

        return new
        {
            has_run = true,
            is_game_over = runState.IsGameOver,
            current_location = TextOf(runState.CurrentLocation),
            current_act_index = runState.CurrentActIndex,
            ascension_level = runState.AscensionLevel,
            act_floor = runState.ActFloor,
            total_floor = runState.TotalFloor,
            act = BuildModelPayload(runState.Act),
            acts = runState.Acts.Select(BuildModelPayload).ToArray(),
            modifiers = runState.Modifiers.Select(BuildModelPayload).ToArray(),
            current_map_coord = BuildMapCoord(runState.CurrentMapCoord),
            current_map_point = runState.CurrentMapPoint is null
                ? null
                : new
                {
                    coord = BuildMapCoord(runState.CurrentMapPoint.coord),
                    point_type = runState.CurrentMapPoint.PointType.ToString()
                },
            current_room = BuildRoomPayload(runState.CurrentRoom),
            player_count = runState.Players.Count
        };
    }

    private static object BuildCombatPayload(CombatManager? combatManager, CombatState? combatState)
    {
        if (combatManager is null || combatState is null || !combatManager.IsInProgress)
        {
            return new
            {
                in_progress = false
            };
        }

        return new
        {
            in_progress = combatManager.IsInProgress,
            is_play_phase = combatManager.IsPlayPhase,
            is_paused = combatManager.IsPaused,
            is_ending = combatManager.IsEnding,
            player_actions_disabled = combatManager.PlayerActionsDisabled,
            round_number = combatState.RoundNumber,
            current_side = combatState.CurrentSide.ToString(),
            target_index_map = BuildCombatTargetIndexPayload(combatState),
            player_creatures = combatState.PlayerCreatures.Select(BuildCreaturePayload).ToArray(),
            enemy_creatures = combatState.Creatures
                .Where(static creature => creature.IsEnemy)
                .Select(BuildCreaturePayload)
                .ToArray()
        };
    }

    private static object[] BuildCombatTargetIndexPayload(CombatState combatState)
    {
        var canUseSelfAlias = combatState.PlayerCreatures.Count == 1;

        return combatState.Creatures
            .Select(creature => new
            {
                action_suffixes = BuildCombatTargetActionSuffixes(creature, canUseSelfAlias),
                combat_id = creature.CombatId,
                name = creature.Name,
                side = creature.Side.ToString(),
                is_enemy = creature.IsEnemy,
                is_alive = creature.IsAlive,
                is_hittable = creature.IsHittable
            })
            .Cast<object>()
            .ToArray();
    }

    private static string[] BuildCombatTargetActionSuffixes(Creature creature, bool canUseSelfAlias)
    {
        var suffixes = new List<string>();

        if (!creature.IsEnemy && canUseSelfAlias)
        {
            suffixes.Add("self");
        }

        suffixes.Add(creature.CombatId.ToString() ?? string.Empty);
        return suffixes.ToArray();
    }

    private static object[] BuildPlayersPayload(
        RunState? runState,
        CombatManager? combatManager,
        CombatState? combatState)
    {
        var players = runState?.Players ?? combatState?.Players ?? Array.Empty<Player>();
        var includeCombatState = combatManager?.IsInProgress == true && combatState is not null;

        return players
            .Select((player, index) => new
            {
                index,
                net_id = player.NetId,
                character = BuildModelPayload(player.Character),
                gold = player.Gold,
                max_energy = player.MaxEnergy,
                creature = BuildCreaturePayload(player.Creature),
                combat = includeCombatState
                    ? BuildPlayerCombatPayload(player.PlayerCombatState)
                    : CreateNotInCombatPayload(),
                deck = BuildPilePayload(player.Deck),
                relics = player.Relics.Select(BuildRelicPayload).ToArray(),
                potions = player.PotionSlots.Select(BuildPotionPayload).ToArray()
            })
            .Cast<object>()
            .ToArray();
    }

    private static object BuildPlayerCombatPayload(PlayerCombatState? playerCombatState)
    {
        if (playerCombatState is null)
        {
            return CreateNotInCombatPayload();
        }

        return new
        {
            in_combat = true,
            energy = playerCombatState.Energy,
            max_energy = playerCombatState.MaxEnergy,
            stars = playerCombatState.Stars,
            hand = BuildPilePayload(playerCombatState.Hand),
            draw_pile = BuildPilePayload(playerCombatState.DrawPile),
            discard_pile = BuildPilePayload(playerCombatState.DiscardPile),
            exhaust_pile = BuildPilePayload(playerCombatState.ExhaustPile),
            play_pile = BuildPilePayload(playerCombatState.PlayPile)
        };
    }

    private static object CreateNotInCombatPayload()
    {
        return new
        {
            in_combat = false
        };
    }

    private static object BuildPilePayload(CardPile? pile)
    {
        if (pile is null)
        {
            return new
            {
                pile_type = "Unknown",
                count = 0,
                cards = Array.Empty<object>()
            };
        }

        return new
        {
            pile_type = pile.Type.ToString(),
            is_combat_pile = pile.IsCombatPile,
            count = pile.Cards.Count,
            cards = pile.Cards.Select(card => BuildCardPayload(card)).ToArray()
        };
    }

    private static object BuildCardPayload(CardModel? card, Creature? previewTarget = null)
    {
        if (card is null)
        {
            return new
            {
                missing = true
            };
        }

        var resolvedTarget = previewTarget ?? card.CurrentTarget;
        var previewVars = BuildCardPreviewVarSet(card, resolvedTarget);
        var damagePerHit = GetDynamicVarInt(previewVars, "Damage");
        var totalDamage = GetDynamicVarInt(previewVars, "CalculatedDamage") ?? damagePerHit;
        var repeats = GetDynamicVarInt(previewVars, "Repeat");
        if (totalDamage is null && damagePerHit.HasValue && repeats.GetValueOrDefault(1) > 1)
        {
            totalDamage = damagePerHit.Value * repeats!.Value;
        }

        var totalBlock = GetDynamicVarInt(previewVars, "CalculatedBlock") ?? GetDynamicVarInt(previewVars, "Block");
        var drawCount = GetDynamicVarInt(previewVars, "Cards");
        var healAmount = GetDynamicVarInt(previewVars, "Heal");
        var hpLossAmount = GetDynamicVarInt(previewVars, "HpLoss");
        var weakAmount = GetDynamicVarInt(previewVars, "Weak");
        var vulnerableAmount = GetDynamicVarInt(previewVars, "Vulnerable");
        var poisonAmount = GetDynamicVarInt(previewVars, "Poison");
        var strengthAmount = GetDynamicVarInt(previewVars, "Strength");
        var dexterityAmount = GetDynamicVarInt(previewVars, "Dexterity");
        var summonCount = GetDynamicVarInt(previewVars, "Summon");
        var extraDamage = GetDynamicVarInt(previewVars, "ExtraDamage");
        var description = GetCardDescription(card, resolvedTarget);
        var xCostValue = card.EnergyCost.CostsX ? SafeResolveCardEnergyXValue(card) : null;
        var xCostSemantics = ResolveXCostSemantics(card, description, damagePerHit, totalDamage, repeats, xCostValue);
        (damagePerHit, totalDamage, repeats) = ApplyXCostPreviewMapping(
            damagePerHit,
            totalDamage,
            repeats,
            xCostValue,
            xCostSemantics);
        var effectSummary = BuildCardEffectSummary(
            totalDamage,
            damagePerHit,
            repeats,
            totalBlock,
            drawCount,
            healAmount,
            hpLossAmount,
            weakAmount,
            vulnerableAmount,
            poisonAmount,
            strengthAmount,
            dexterityAmount,
            summonCount,
            extraDamage,
            xCostValue);

        return new
        {
            id = card.Id.ToString(),
            title = string.IsNullOrWhiteSpace(card.Title)
                ? DescribeText(card.TitleLocString, card)
                : DescribeText(card.Title, card),
            description,
            type = card.Type.ToString(),
            rarity = card.Rarity.ToString(),
            target_type = card.TargetType.ToString(),
            pile = card.Pile?.Type.ToString(),
            is_playable = GetHiddenPropertyValue<bool>(card, "IsPlayable") ?? false,
            canonical_energy_cost = card.EnergyCost.Canonical,
            resolved_energy_cost = card.EnergyCost.GetResolved(),
            costs_x = card.EnergyCost.CostsX,
            canonical_star_cost = card.CanonicalStarCost,
            current_star_cost = card.CurrentStarCost,
            has_star_cost_x = card.HasStarCostX,
            effect_preview = new
            {
                summary = effectSummary,
                preview_target_combat_id = resolvedTarget?.CombatId,
                total_damage = totalDamage,
                damage_per_hit = damagePerHit,
                hits = repeats,
                total_block = totalBlock,
                draw = drawCount,
                heal = healAmount,
                hp_loss = hpLossAmount,
                weak = weakAmount,
                vulnerable = vulnerableAmount,
                poison = poisonAmount,
                strength = strengthAmount,
                dexterity = dexterityAmount,
                summon = summonCount,
                extra_damage = extraDamage,
                x_cost_value = xCostValue,
                x_cost_semantics = xCostSemantics
            },
            dynamic_vars = BuildDynamicVarPayloads(previewVars)
        };
    }

    private static object? BuildCardUpgradePreviewPayload(CardModel? card)
    {
        if (card is null)
        {
            return null;
        }

        try
        {
            var cardScope = card.CardScope;
            if (cardScope is null)
            {
                return null;
            }

            var upgradedCard = cardScope.CloneCard(card);
            upgradedCard.UpgradeInternal();
            upgradedCard.UpgradePreviewType = card.Pile?.IsCombatPile == true
                ? CardUpgradePreviewType.Combat
                : CardUpgradePreviewType.Deck;
            return BuildCardPayload(upgradedCard);
        }
        catch
        {
            return null;
        }
    }

    private static string GetCardDescription(CardModel card, Creature? previewTarget)
    {
        try
        {
            var pileType = card.Pile?.Type ?? (card.IsInCombat ? PileType.Hand : PileType.Deck);
            var description = card.GetDescriptionForPile(pileType, previewTarget!);
            if (!string.IsNullOrWhiteSpace(description))
            {
                return DescribeText(description, card);
            }
        }
        catch
        {
        }

        return DescribeText(card.Description, card);
    }

    private static DynamicVarSet? BuildCardPreviewVarSet(CardModel card, Creature? previewTarget)
    {
        try
        {
            var dynamicVars = card.DynamicVars;
            var previewVars = dynamicVars?.Clone(card);
            if (previewVars is null)
            {
                return null;
            }

            previewVars.ClearPreview();
            card.UpdateDynamicVarPreview(ResolveCardPreviewMode(card), previewTarget!, previewVars);
            return previewVars;
        }
        catch
        {
            return card.DynamicVars;
        }
    }

    private static CardPreviewMode ResolveCardPreviewMode(CardModel card)
    {
        return card.TargetType == TargetType.AllEnemies
            ? CardPreviewMode.MultiCreatureTargeting
            : CardPreviewMode.Normal;
    }

    private static int? SafeResolveCardEnergyXValue(CardModel card)
    {
        int? currentEnergy = null;

        try
        {
            currentEnergy = card.Owner?.PlayerCombatState?.Energy;
            var resolvedXValue = card.ResolveEnergyXValue();
            if (currentEnergy.HasValue)
            {
                return Math.Max(resolvedXValue, currentEnergy.Value);
            }

            return resolvedXValue;
        }
        catch
        {
            return currentEnergy;
        }
    }

    private static string? ResolveXCostSemantics(
        CardModel card,
        string description,
        int? damagePerHit,
        int? totalDamage,
        int? repeats,
        int? xCostValue)
    {
        if (!card.EnergyCost.CostsX || !xCostValue.HasValue || xCostValue.Value <= 0)
        {
            return null;
        }

        var normalizedDescription = NormalizeComparableText(description).ToLowerInvariant();
        var looksLikeRepeatPerEnergyText =
            normalizedDescription.Contains("x次", StringComparison.Ordinal) ||
            normalizedDescription.Contains("x times", StringComparison.Ordinal) ||
            normalizedDescription.Contains("times equal to x", StringComparison.Ordinal);

        if (looksLikeRepeatPerEnergyText &&
            (damagePerHit.HasValue || totalDamage.HasValue))
        {
            return "repeat_per_energy";
        }

        if (card.Type == CardType.Attack &&
            xCostValue.Value > 1 &&
            repeats.GetValueOrDefault(1) <= 1 &&
            (damagePerHit.HasValue || totalDamage.HasValue))
        {
            return "repeat_per_energy";
        }

        return null;
    }

    private static (int? DamagePerHit, int? TotalDamage, int? Repeats) ApplyXCostPreviewMapping(
        int? damagePerHit,
        int? totalDamage,
        int? repeats,
        int? xCostValue,
        string? xCostSemantics)
    {
        if (!string.Equals(xCostSemantics, "repeat_per_energy", StringComparison.Ordinal) ||
            !xCostValue.HasValue ||
            xCostValue.Value <= 0)
        {
            return (damagePerHit, totalDamage, repeats);
        }

        var mappedDamagePerHit = damagePerHit ?? totalDamage;
        var mappedRepeats = xCostValue.Value;
        var mappedTotalDamage = mappedDamagePerHit.HasValue
            ? mappedDamagePerHit.Value * mappedRepeats
            : totalDamage;

        return (mappedDamagePerHit, mappedTotalDamage, mappedRepeats);
    }

    private static object[] BuildDynamicVarPayloads(DynamicVarSet? dynamicVarSet)
    {
        return dynamicVarSet?.Values
            .Select(BuildDynamicVarPayload)
            .Where(static payload => payload is not null)
            .Cast<object>()
            .ToArray()
            ?? Array.Empty<object>();
    }

    private static object? BuildDynamicVarPayload(DynamicVar? dynamicVar)
    {
        if (dynamicVar is null)
        {
            return null;
        }

        return new
        {
            name = dynamicVar.Name,
            int_value = dynamicVar.IntValue,
            preview_value = dynamicVar.PreviewValue,
            base_value = dynamicVar.BaseValue,
            enchanted_value = dynamicVar.EnchantedValue,
            was_just_upgraded = dynamicVar.WasJustUpgraded
        };
    }

    private static int? GetDynamicVarInt(DynamicVarSet? dynamicVarSet, string key)
    {
        if (dynamicVarSet is null || string.IsNullOrWhiteSpace(key))
        {
            return null;
        }

        try
        {
            return dynamicVarSet.TryGetValue(key, out var dynamicVar)
                ? GetDynamicVarInt(dynamicVar)
                : null;
        }
        catch
        {
            return null;
        }
    }

    private static int? GetDynamicVarInt(DynamicVar? dynamicVar)
    {
        if (dynamicVar is null)
        {
            return null;
        }

        return decimal.Truncate(dynamicVar.PreviewValue) != 0m
            ? (int)decimal.Truncate(dynamicVar.PreviewValue)
            : dynamicVar.IntValue;
    }

    private static string BuildCardEffectSummary(
        int? totalDamage,
        int? damagePerHit,
        int? repeats,
        int? totalBlock,
        int? drawCount,
        int? healAmount,
        int? hpLossAmount,
        int? weakAmount,
        int? vulnerableAmount,
        int? poisonAmount,
        int? strengthAmount,
        int? dexterityAmount,
        int? summonCount,
        int? extraDamage,
        int? xCostValue)
    {
        var parts = new List<string>();

        if (damagePerHit.HasValue && repeats.GetValueOrDefault(1) > 1)
        {
            parts.Add($"{damagePerHit.Value} x {repeats!.Value} damage");
        }
        else if (totalDamage.HasValue && totalDamage.Value != 0)
        {
            parts.Add($"{totalDamage.Value} damage");
        }

        if (totalBlock.HasValue && totalBlock.Value != 0)
        {
            parts.Add($"{totalBlock.Value} block");
        }

        if (drawCount.HasValue && drawCount.Value != 0)
        {
            parts.Add($"draw {drawCount.Value}");
        }

        if (healAmount.HasValue && healAmount.Value != 0)
        {
            parts.Add($"heal {healAmount.Value}");
        }

        if (hpLossAmount.HasValue && hpLossAmount.Value != 0)
        {
            parts.Add($"lose {hpLossAmount.Value} HP");
        }

        if (weakAmount.HasValue && weakAmount.Value != 0)
        {
            parts.Add($"apply {weakAmount.Value} Weak");
        }

        if (vulnerableAmount.HasValue && vulnerableAmount.Value != 0)
        {
            parts.Add($"apply {vulnerableAmount.Value} Vulnerable");
        }

        if (poisonAmount.HasValue && poisonAmount.Value != 0)
        {
            parts.Add($"apply {poisonAmount.Value} Poison");
        }

        if (strengthAmount.HasValue && strengthAmount.Value != 0)
        {
            parts.Add($"gain {strengthAmount.Value} Strength");
        }

        if (dexterityAmount.HasValue && dexterityAmount.Value != 0)
        {
            parts.Add($"gain {dexterityAmount.Value} Dexterity");
        }

        if (summonCount.HasValue && summonCount.Value != 0)
        {
            parts.Add($"summon {summonCount.Value}");
        }

        if (extraDamage.HasValue && extraDamage.Value != 0)
        {
            parts.Add($"{extraDamage.Value} extra damage");
        }

        if (xCostValue.HasValue && xCostValue.Value != 0)
        {
            parts.Add($"X={xCostValue.Value}");
        }

        return string.Join(" + ", parts);
    }

    private static object BuildCreaturePayload(Creature? creature)
    {
        if (creature is null)
        {
            return new
            {
                missing = true
            };
        }

        return new
        {
            name = creature.Name,
            model_id = creature.ModelId.ToString(),
            combat_id = creature.CombatId,
            side = creature.Side.ToString(),
            current_hp = creature.CurrentHp,
            max_hp = creature.MaxHp,
            block = creature.Block,
            is_alive = creature.IsAlive,
            is_hittable = SafeGetCreatureIsHittable(creature),
            powers = creature.Powers.Select(BuildPowerPayload).ToArray(),
            intent = creature.IsEnemy ? BuildEnemyIntentPayload(creature) : null
        };
    }

    private static object? BuildEnemyIntentPayload(Creature creature)
    {
        var monster = creature.Monster;
        if (monster is null)
        {
            return null;
        }

        var targets = ResolveMonsterIntentTargets(creature);
        var nextMove = monster.NextMove;
        var intents = SafeGetMonsterIntents(monster, nextMove);

        return new
        {
            state_id = nextMove?.StateId,
            follow_up_state_id = nextMove?.FollowUpStateId,
            is_move = nextMove?.IsMove ?? false,
            title = intents.Select(GetMonsterIntentTitle)
                .FirstOrDefault(title => !string.IsNullOrWhiteSpace(title)),
            intents = intents.Select(intent => BuildMonsterIntentPayload(intent, creature, targets)).ToArray()
        };
    }

    private static IReadOnlyList<Creature> ResolveMonsterIntentTargets(Creature owner)
    {
        var combatState = owner.CombatState;
        if (combatState is null)
        {
            return Array.Empty<Creature>();
        }

        return combatState.PlayerCreatures
            .Where(static creature => creature.IsAlive)
            .ToArray();
    }

    private static IReadOnlyList<AbstractIntent> SafeGetMonsterIntents(MonsterModel monster, MoveState? nextMove)
    {
        if (nextMove?.Intents is { Count: > 0 } nextMoveIntents)
        {
            return nextMoveIntents.ToArray();
        }

        return Array.Empty<AbstractIntent>();
    }

    private static object BuildMonsterIntentPayload(
        AbstractIntent intent,
        Creature owner,
        IReadOnlyList<Creature> targets)
    {
        var repeats = intent switch
        {
            SingleAttackIntent singleAttackIntent => singleAttackIntent.Repeats,
            MultiAttackIntent multiAttackIntent => multiAttackIntent.Repeats,
            _ => 1
        };

        var totalDamage = intent switch
        {
            SingleAttackIntent singleAttackIntent => SafeGetIntentTotalDamage(singleAttackIntent, targets, owner),
            MultiAttackIntent multiAttackIntent => SafeGetIntentTotalDamage(multiAttackIntent, targets, owner),
            _ => null
        };
        int? damagePerHit = totalDamage.HasValue && repeats > 0 && totalDamage.Value % repeats == 0
            ? totalDamage.Value / repeats
            : null;
        var rawLabel = SafeGetIntentLocString(intent, "GetIntentLabel", targets, owner);
        var rawDescription = SafeGetIntentLocString(intent, "GetIntentDescription", targets, owner);

        return new
        {
            intent_type = intent.IntentType.ToString(),
            intent_class = intent.GetType().Name,
            title = GetMonsterIntentTitle(intent),
            label = NormalizeMonsterIntentLabel(intent.IntentType, rawLabel, totalDamage, damagePerHit, repeats),
            description = NormalizeMonsterIntentDescription(
                intent.IntentType,
                rawDescription,
                totalDamage,
                damagePerHit,
                repeats),
            has_tip = intent.HasIntentTip,
            repeats,
            total_damage = totalDamage,
            damage_per_hit = damagePerHit
        };
    }

    private static string GetMonsterIntentTitle(AbstractIntent intent)
    {
        return DescribeText(GetHiddenPropertyObjectValue(intent, "IntentTitle"), intent);
    }

    private static int? SafeGetIntentTotalDamage(object intent, IEnumerable<Creature> targets, Creature owner)
    {
        try
        {
            var method = FindMethod(intent.GetType(), "GetTotalDamage", 2);
            if (method?.Invoke(intent, new object?[] { targets, owner }) is int totalDamage)
            {
                return totalDamage;
            }
        }
        catch
        {
        }

        return null;
    }

    private static string SafeGetIntentLocString(
        object intent,
        string methodName,
        IEnumerable<Creature> targets,
        Creature owner)
    {
        try
        {
            var method = FindMethod(intent.GetType(), methodName, 2);
            return DescribeText(method?.Invoke(intent, new object?[] { targets, owner }), intent);
        }
        catch
        {
            return string.Empty;
        }
    }

    private static string NormalizeMonsterIntentLabel(
        IntentType intentType,
        string rawLabel,
        int? totalDamage,
        int? damagePerHit,
        int repeats)
    {
        if (!LooksLikeUnresolvedPayloadText(rawLabel))
        {
            return rawLabel;
        }

        if (!IsAttackLikeIntent(intentType) || !totalDamage.HasValue)
        {
            return string.Empty;
        }

        if (repeats > 1 && damagePerHit.HasValue)
        {
            return $"{damagePerHit.Value}×{repeats}";
        }

        return totalDamage.Value.ToString(CultureInfo.InvariantCulture);
    }

    private static string NormalizeMonsterIntentDescription(
        IntentType intentType,
        string rawDescription,
        int? totalDamage,
        int? damagePerHit,
        int repeats)
    {
        if (!LooksLikeUnresolvedPayloadText(rawDescription))
        {
            return rawDescription;
        }

        return intentType switch
        {
            IntentType.Attack or IntentType.DeathBlow when repeats > 1 && damagePerHit.HasValue
                => $"这个敌人将要攻击造成{damagePerHit.Value}点伤害{repeats}次。",
            IntentType.Attack or IntentType.DeathBlow when totalDamage.HasValue
                => $"这个敌人将要攻击造成{totalDamage.Value}点伤害。",
            IntentType.Attack or IntentType.DeathBlow
                => "这个敌人将要攻击。",
            IntentType.Defend => "这个敌人将会在其回合获得格挡。",
            IntentType.Buff => "这个敌人将要使用一个强化效果。",
            IntentType.Debuff => "这个敌人将要施加一个减益效果。",
            IntentType.CardDebuff => "这个敌人将要向你的牌堆加入状态牌。",
            IntentType.Heal => "这个敌人将要回复生命值。",
            IntentType.Summon => "这个敌人将要召唤增援。",
            IntentType.Stun => "这个敌人本回合不会行动。",
            IntentType.Sleep => "这个敌人处于睡眠中。",
            IntentType.Escape => "这个敌人将要逃跑。",
            _ => string.IsNullOrWhiteSpace(rawDescription) ? string.Empty : rawDescription
        };
    }

    private static bool IsAttackLikeIntent(IntentType intentType)
    {
        return intentType is IntentType.Attack or IntentType.DeathBlow;
    }

    private static bool LooksLikeUnresolvedPayloadText(string text)
    {
        if (string.IsNullOrWhiteSpace(text))
        {
            return true;
        }

        if (text.IndexOf('{') >= 0 || text.IndexOf('}') >= 0)
        {
            return true;
        }

        var unresolvedTokens = new[]
        {
            "Amount",
            "Count",
            "Damage",
            "ExtraText",
            "Heal",
            "IsMultiplayer",
            "Repeat"
        };

        return unresolvedTokens.Any(
            token => text.IndexOf(token, StringComparison.OrdinalIgnoreCase) >= 0);
    }

    private static object BuildPowerPayload(PowerModel power)
    {
        return new
        {
            title = TextOf(power.Title),
            description = TryGetDescription(power),
            amount = power.Amount,
            display_amount = power.DisplayAmount,
            type = power.Type.ToString(),
            stack_type = power.StackType.ToString()
        };
    }

    private static object BuildRewardsPayload(
        NRewardsScreen? rewardsScreen,
        NProceedButton? roomProceedButton,
        NProceedButton? rewardProceedButton,
        NMapScreen? mapScreen,
        IReadOnlyList<NRewardButton> rewardButtons)
    {
        var visible = IsRewardsScreenVisible(
            rewardsScreen,
            roomProceedButton,
            rewardProceedButton,
            mapScreen,
            rewardButtons);
        return new
        {
            visible,
            terminal_proceed_visible = visible &&
                                       rewardProceedButton is not null &&
                                       IsNodeVisible(rewardProceedButton),
            rewards = rewardButtons.Select((button, index) => new
            {
                index,
                reward = BuildRewardPayload(button.Reward)
            }).ToArray()
        };
    }

    private static object BuildRewardPayload(Reward? reward)
    {
        if (reward is null)
        {
            return new
            {
                reward_type = "unknown",
                missing = true
            };
        }

        return reward switch
        {
            CardReward cardReward => new
            {
                reward_type = "card",
                description = DescribeText(cardReward.Description, cardReward),
                can_skip = cardReward.CanSkip,
                can_reroll = cardReward.CanReroll,
                cards = cardReward.Cards.Select(card => BuildCardPayload(card)).ToArray()
            },
            GoldReward goldReward => new
            {
                reward_type = "gold",
                description = DescribeText(goldReward.Description, goldReward),
                amount = goldReward.Amount
            },
            RelicReward relicReward => new
            {
                reward_type = "relic",
                description = DescribeText(relicReward.Description, relicReward),
                rarity = relicReward.Rarity.ToString(),
                relic = BuildRelicPayload(relicReward.ClaimedRelic)
            },
            PotionReward potionReward => new
            {
                reward_type = "potion",
                description = DescribeText(potionReward.Description, potionReward),
                potion = BuildPotionPayload(potionReward.Potion)
            },
            _ => new
            {
                reward_type = reward.GetType().Name,
                description = DescribeText(reward.Description, reward)
            }
        };
    }

    private static string DescribeReward(Reward? reward)
    {
        if (reward is null)
        {
            return "Unknown reward";
        }

        return reward switch
        {
            CardReward cardReward => $"Card reward ({cardReward.Cards.Count()} options)",
            GoldReward goldReward => $"{goldReward.Amount} gold",
            RelicReward relicReward => $"Relic {TextOf(relicReward.ClaimedRelic?.Title)}",
            PotionReward potionReward => $"Potion {TextOf(potionReward.Potion?.Title)}",
            _ => DescribeText(reward.Description, reward)
        };
    }

    private static object BuildCardRewardSelectionPayload(
        NCardRewardSelectionScreen? cardRewardScreen,
        IReadOnlyList<NCardHolder> cardRewardOptions,
        Node? cardRewardSkipButton)
    {
        var visible = IsCardRewardSelectionVisible(cardRewardScreen, cardRewardOptions);
        var ready = IsCardRewardSelectionReady(cardRewardScreen);
        return new
        {
            visible,
            ready,
            skip_visible = ready &&
                           cardRewardSkipButton is not null &&
                           IsNodeVisible(cardRewardSkipButton) &&
                           IsButtonEnabled(cardRewardSkipButton),
            options = cardRewardOptions.Select((holder, index) => new
            {
                index,
                card = BuildCardPayload(holder.CardModel)
            }).ToArray()
        };
    }

    private static object BuildCardSelectionPayload(
        Node? cardSelectionScreen,
        IReadOnlyList<NCardHolder> cardSelectionOptions,
        Node? cardSelectionConfirmButton,
        Node? cardSelectionCancelButton,
        Node? cardSelectionCloseButton,
        Node? cardSelectionSkipButton)
    {
        var visible = cardSelectionScreen is not null && IsNodeVisible(cardSelectionScreen);
        var texts = visible ? CollectVisibleText(cardSelectionScreen, 8).ToArray() : Array.Empty<string>();
        var prefs = GetHiddenFieldValue(cardSelectionScreen, "_prefs");
        var prompt = visible ? TryGetCardSelectionPrompt(cardSelectionScreen) : null;
        var selectedCount = CountSelectedCardSelectionCards(cardSelectionScreen);
        var minSelect = GetHiddenPropertyValue<int>(prefs, "MinSelect");
        var maxSelect = GetHiddenPropertyValue<int>(prefs, "MaxSelect");
        var selectionSemantics = ResolveCardSelectionSemantics(cardSelectionScreen, prompt, texts);
        var confirmVisible = cardSelectionConfirmButton is not null &&
                             IsNodeVisible(cardSelectionConfirmButton) &&
                             IsButtonEnabled(cardSelectionConfirmButton);
        var skipVisible = cardSelectionSkipButton is not null &&
                          IsNodeVisible(cardSelectionSkipButton) &&
                          IsButtonEnabled(cardSelectionSkipButton);

        return new
        {
            visible,
            screen_type = visible ? cardSelectionScreen!.GetType().Name : null,
            prompt,
            texts,
            selection_semantics = selectionSemantics,
            decision_text = BuildCardSelectionDecisionText(
                selectionSemantics,
                prompt,
                selectedCount,
                minSelect,
                maxSelect,
                confirmVisible,
                skipVisible),
            selected_count = selectedCount,
            min_select = minSelect,
            max_select = maxSelect,
            requires_manual_confirmation = GetHiddenPropertyValue<bool>(prefs, "RequireManualConfirmation"),
            cancelable = GetHiddenPropertyValue<bool>(prefs, "Cancelable"),
            confirm_visible = confirmVisible,
            cancel_visible = cardSelectionCancelButton is not null &&
                             IsNodeVisible(cardSelectionCancelButton) &&
                             IsButtonEnabled(cardSelectionCancelButton),
            close_visible = cardSelectionCloseButton is not null &&
                            IsNodeVisible(cardSelectionCloseButton) &&
                            IsButtonEnabled(cardSelectionCloseButton),
            skip_visible = skipVisible,
            options = cardSelectionOptions.Select((holder, index) =>
            {
                var optionIndex = GetCardSelectionOptionIndex(cardSelectionScreen, holder, index);
                var selectionId = GetCardSelectionOptionSelectionId(cardSelectionScreen, holder, optionIndex);
                return new
                {
                    index = optionIndex,
                    selection_id = selectionId,
                    action_id = selectionId is not null
                        ? $"card_selection:select:{selectionId}"
                        : $"card_selection:select:{optionIndex}",
                    card = BuildCardPayload(holder.CardModel),
                    is_selected = IsCardSelectionCardSelected(cardSelectionScreen, holder.CardModel)
                };
            }).ToArray()
        };
    }

    private static object BuildCharacterSelectionPayload(
        NCharacterSelectScreen? characterSelectScreen,
        IReadOnlyList<NCharacterSelectButton> characterButtons,
        NCharacterSelectButton? selectedCharacterButton,
        NConfirmButton? embarkButton)
    {
        var visible = characterSelectScreen is not null && IsNodeVisible(characterSelectScreen);
        var selectedIndex = -1;

        for (var index = 0; index < characterButtons.Count; index++)
        {
            if (ReferenceEquals(characterButtons[index], selectedCharacterButton))
            {
                selectedIndex = index;
                break;
            }
        }

        return new
        {
            visible,
            can_embark = visible && embarkButton is not null && IsNodeVisible(embarkButton) && IsButtonEnabled(embarkButton),
            selected_index = selectedIndex >= 0 ? (int?)selectedIndex : null,
            selected_character = BuildCharacterPayload(selectedCharacterButton?.Character),
            options = characterButtons.Select((button, index) => new
            {
                index,
                character = BuildCharacterPayload(button.Character),
                is_selected = ReferenceEquals(button, selectedCharacterButton),
                is_locked = button.IsLocked,
                is_random = button.IsRandom,
                remote_selected_player_count = button.RemoteSelectedPlayers.Count
            }).ToArray()
        };
    }

    private static object BuildRunModeSelectionPayload(
        Node? runModeSubmenu,
        Node? runModeStandardButton,
        Node? runModeDailyButton,
        Node? runModeCustomButton,
        NBackButton? runModeBackButton)
    {
        return new
        {
            visible = runModeSubmenu is not null && IsNodeVisible(runModeSubmenu),
            options = new[]
            {
                BuildRunModeButtonPayload(runModeStandardButton, "standard", "Standard"),
                BuildRunModeButtonPayload(runModeDailyButton, "daily", "Daily"),
                BuildRunModeButtonPayload(runModeCustomButton, "custom", "Custom")
            },
            back_button = BuildRunModeButtonPayload(runModeBackButton, "back", "Back")
        };
    }

    private static object BuildRunModeButtonPayload(Node? button, string semanticAction, string fallbackLabel)
    {
        if (button is null || !IsNodeVisible(button))
        {
            return new
            {
                visible = false,
                semantic_action = semanticAction,
                label = fallbackLabel,
                texts = Array.Empty<string>()
            };
        }

        var texts = CollectVisibleText(button, 4).ToArray();
        var text = texts.FirstOrDefault(static candidate => !string.IsNullOrWhiteSpace(candidate)) ?? fallbackLabel;

        return new
        {
            visible = true,
            semantic_action = semanticAction,
            label = fallbackLabel,
            text,
            texts,
            node_type = button.GetType().FullName
        };
    }

    private static object BuildEventOptionsPayload(
        IReadOnlyList<NEventOptionButton> eventOptionButtons,
        NMapScreen? mapScreen,
        NEventRoom? eventRoom,
        Node? hoverTipSet,
        NCrystalSphereScreen? crystalSphereScreen,
        IReadOnlyList<NCrystalSphereCell> crystalSphereCells,
        NDivinationButton? crystalSphereSmallDivinationButton,
        NDivinationButton? crystalSphereBigDivinationButton,
        NProceedButton? crystalSphereProceedButton)
    {
        if (mapScreen is not null && mapScreen.IsOpen)
        {
            return new
            {
                visible = false,
                visible_glossary_source = (string?)null,
                visible_glossary_texts = Array.Empty<string>(),
                visible_glossary = Array.Empty<object>(),
                options = Array.Empty<object>()
            };
        }

        var (glossarySource, glossaryTexts) = CollectVisibleEventGlossaryTexts(
            eventOptionButtons,
            eventRoom,
            hoverTipSet);
        var currentEventModel = GetHiddenFieldValue(eventRoom, "_event") as EventModel;
        var options = new List<object>(eventOptionButtons.Count + crystalSphereCells.Count + 4);
        options.AddRange(eventOptionButtons.Select((button, index) => BuildEventOptionPayload(button, index, currentEventModel)));
        options.AddRange(
            BuildCrystalSphereEventOptionPayloads(
                crystalSphereScreen,
                crystalSphereCells,
                crystalSphereSmallDivinationButton,
                crystalSphereBigDivinationButton,
                crystalSphereProceedButton,
                eventOptionButtons.Count));
        var isCrystalSphereVisible = crystalSphereScreen is not null && IsNodeVisible(crystalSphereScreen);

        return new
        {
            visible = eventOptionButtons.Count > 0 || options.Count > 0 || isCrystalSphereVisible || glossaryTexts.Count > 0,
            visible_glossary_source = glossarySource,
            visible_glossary_texts = glossaryTexts.ToArray(),
            visible_glossary = BuildVisibleGlossaryPayload(glossaryTexts),
            options = options.ToArray()
        };
    }

    private static object BuildCrystalSpherePayload(
        NCrystalSphereScreen? crystalSphereScreen,
        IReadOnlyList<NCrystalSphereCell> crystalSphereCells,
        NDivinationButton? crystalSphereSmallDivinationButton,
        NDivinationButton? crystalSphereBigDivinationButton,
        NProceedButton? crystalSphereProceedButton)
    {
        var visible = crystalSphereScreen is not null && IsNodeVisible(crystalSphereScreen);
        if (!visible)
        {
            return new
            {
                visible = false,
                cells = Array.Empty<object>()
            };
        }

        var minigame = GetCrystalSphereMinigame(crystalSphereScreen);
        var currentTool = GetCrystalSphereToolName(minigame);
        var divinationsLeft = GetCrystalSphereDivinationCount(minigame);
        var isFinished = GetCrystalSphereIsFinished(minigame);
        var canRevealHiddenCells =
            divinationsLeft.GetValueOrDefault() > 0 &&
            !isFinished &&
            !string.IsNullOrWhiteSpace(currentTool) &&
            !string.Equals(currentTool, "none", StringComparison.OrdinalIgnoreCase);
        var divinationsLeftLabel =
            TryGetNodeText(GetHiddenFieldValue(crystalSphereScreen, "_divinationsLeftLabel") as Node);
        var instructionsTitle =
            TryGetNodeText(GetHiddenFieldValue(crystalSphereScreen, "_instructionsTitleLabel") as Node);
        var instructionsDescription =
            TryGetNodeText(GetHiddenFieldValue(crystalSphereScreen, "_instructionsDescriptionLabel") as Node);

        return new
        {
            visible = true,
            divinations_left = divinationsLeft,
            divinations_left_text = divinationsLeftLabel,
            current_tool = currentTool,
            is_finished = isFinished,
            grid_size = BuildCrystalSphereGridPayload(minigame, crystalSphereCells),
            instructions_title = instructionsTitle,
            instructions_description = instructionsDescription,
            small_divination = BuildCrystalSphereDivinationButtonPayload(
                crystalSphereSmallDivinationButton,
                expectedTool: "small",
                currentTool,
                divinationSize: "1x1",
                fallbackLabel: "Small Divination"),
            big_divination = BuildCrystalSphereDivinationButtonPayload(
                crystalSphereBigDivinationButton,
                expectedTool: "big",
                currentTool,
                divinationSize: "3x3",
                fallbackLabel: "Big Divination"),
            proceed = BuildCrystalSphereProceedPayload(crystalSphereProceedButton),
            cells = crystalSphereCells
                .Select(cell => BuildCrystalSphereCellPayload(cell, canRevealHiddenCells))
                .ToArray()
        };
    }

    private static object[] BuildCrystalSphereEventOptionPayloads(
        NCrystalSphereScreen? crystalSphereScreen,
        IReadOnlyList<NCrystalSphereCell> crystalSphereCells,
        NDivinationButton? crystalSphereSmallDivinationButton,
        NDivinationButton? crystalSphereBigDivinationButton,
        NProceedButton? crystalSphereProceedButton,
        int startingIndex)
    {
        return GetCrystalSphereEventOptionDescriptors(
                crystalSphereScreen,
                crystalSphereCells,
                crystalSphereSmallDivinationButton,
                crystalSphereBigDivinationButton,
                crystalSphereProceedButton,
                startingIndex)
            .Select(BuildCrystalSphereEventOptionPayload)
            .ToArray();
    }

    private static IReadOnlyList<CrystalSphereEventOptionDescriptor> GetCrystalSphereEventOptionDescriptors(
        NCrystalSphereScreen? crystalSphereScreen,
        IReadOnlyList<NCrystalSphereCell> crystalSphereCells,
        NDivinationButton? crystalSphereSmallDivinationButton,
        NDivinationButton? crystalSphereBigDivinationButton,
        NProceedButton? crystalSphereProceedButton,
        int startingIndex)
    {
        var descriptors = new List<CrystalSphereEventOptionDescriptor>();
        if (crystalSphereScreen is null || !IsNodeVisible(crystalSphereScreen))
        {
            return descriptors;
        }

        var minigame = GetCrystalSphereMinigame(crystalSphereScreen);
        var currentTool = GetCrystalSphereToolName(minigame);
        var divinationsLeft = GetCrystalSphereDivinationCount(minigame).GetValueOrDefault();
        var isFinished = GetCrystalSphereIsFinished(minigame);
        var canRevealHiddenCells =
            divinationsLeft > 0 &&
            !isFinished &&
            !string.IsNullOrWhiteSpace(currentTool) &&
            !string.Equals(currentTool, "none", StringComparison.OrdinalIgnoreCase);

        void AddDescriptor(CrystalSphereEventOptionDescriptor descriptor)
        {
            descriptors.Add(descriptor with { Index = startingIndex + descriptors.Count });
        }

        var smallVisible = crystalSphereSmallDivinationButton is not null && IsNodeVisible(crystalSphereSmallDivinationButton);
        var smallEnabled = smallVisible && IsButtonEnabled(crystalSphereSmallDivinationButton);
        var smallSelected = string.Equals(currentTool, "small", StringComparison.OrdinalIgnoreCase);
        if (smallVisible)
        {
            AddDescriptor(new CrystalSphereEventOptionDescriptor
            {
                Index = 0,
                OptionType = "crystal_sphere_small_divination",
                OptionId = "crystal_sphere:small_divination",
                Title = GetCrystalSphereControlLabel(crystalSphereSmallDivinationButton, "Small Divination"),
                Description = "Select 1x1 divination.",
                DivinationSize = "1x1",
                IsSelected = smallSelected,
                IsEnabled = smallEnabled,
                ActionAvailable = smallEnabled && !smallSelected,
                Execute = () => InvokeCrystalSphereDivinationAction(
                    crystalSphereScreen,
                    crystalSphereSmallDivinationButton!,
                    useBigDivination: false)
            });
        }

        var bigVisible = crystalSphereBigDivinationButton is not null && IsNodeVisible(crystalSphereBigDivinationButton);
        var bigEnabled = bigVisible && IsButtonEnabled(crystalSphereBigDivinationButton);
        var bigSelected = string.Equals(currentTool, "big", StringComparison.OrdinalIgnoreCase);
        if (bigVisible)
        {
            AddDescriptor(new CrystalSphereEventOptionDescriptor
            {
                Index = 0,
                OptionType = "crystal_sphere_big_divination",
                OptionId = "crystal_sphere:big_divination",
                Title = GetCrystalSphereControlLabel(crystalSphereBigDivinationButton, "Big Divination"),
                Description = "Select 3x3 divination.",
                DivinationSize = "3x3",
                IsSelected = bigSelected,
                IsEnabled = bigEnabled,
                ActionAvailable = bigEnabled && !bigSelected,
                Execute = () => InvokeCrystalSphereDivinationAction(
                    crystalSphereScreen,
                    crystalSphereBigDivinationButton!,
                    useBigDivination: true)
            });
        }

        if (canRevealHiddenCells)
        {
            foreach (var cell in crystalSphereCells)
            {
                var entity = cell.Entity;
                if (entity is null || !entity.IsHidden)
                {
                    continue;
                }

                var x = entity.X;
                var y = entity.Y;
                AddDescriptor(new CrystalSphereEventOptionDescriptor
                {
                    Index = 0,
                    OptionType = "crystal_sphere_cell",
                    OptionId = $"crystal_sphere:cell:{x},{y}",
                    Title = $"Reveal cell ({x}, {y})",
                    Description = "Use the selected divination on this cell.",
                    X = x,
                    Y = y,
                    IsHighlighted = entity.IsHighlighted,
                    IsEnabled = true,
                    ActionAvailable = true,
                    Execute = () => InvokeCrystalSphereCellAction(crystalSphereScreen, cell)
                });
            }
        }

        var proceedVisible = crystalSphereProceedButton is not null && IsNodeVisible(crystalSphereProceedButton);
        var proceedEnabled = proceedVisible && IsButtonEnabled(crystalSphereProceedButton);
        if (proceedVisible)
        {
            AddDescriptor(new CrystalSphereEventOptionDescriptor
            {
                Index = 0,
                OptionType = "crystal_sphere_proceed",
                OptionId = "crystal_sphere:proceed",
                Title = GetCrystalSphereControlLabel(crystalSphereProceedButton, "Proceed"),
                IsProceed = true,
                IsEnabled = proceedEnabled,
                ActionAvailable = proceedEnabled,
                Execute = () => InvokeCrystalSphereProceedAction(
                    crystalSphereScreen,
                    crystalSphereProceedButton!)
            });
        }

        return descriptors;
    }

    private static object BuildCrystalSphereEventOptionPayload(CrystalSphereEventOptionDescriptor descriptor)
    {
        return new
        {
            index = descriptor.Index,
            option_type = descriptor.OptionType,
            option_id = descriptor.OptionId,
            title = descriptor.Title,
            description = descriptor.Description,
            is_locked = !descriptor.IsEnabled,
            is_proceed = descriptor.IsProceed,
            is_selected = descriptor.IsSelected,
            is_enabled = descriptor.IsEnabled,
            action_available = descriptor.ActionAvailable,
            divination_size = descriptor.DivinationSize,
            coord = BuildCrystalSphereCoordPayload(descriptor.X, descriptor.Y),
            is_highlighted = descriptor.IsHighlighted
        };
    }

    private static void AddCrystalSphereEventActions(
        List<BridgeResolvedAction> actions,
        BridgeWorldContext context,
        int startingIndex)
    {
        foreach (var descriptor in GetCrystalSphereEventOptionDescriptors(
                     context.CrystalSphereScreen,
                     context.CrystalSphereCells,
                     context.CrystalSphereSmallDivinationButton,
                     context.CrystalSphereBigDivinationButton,
                     context.CrystalSphereProceedButton,
                     startingIndex))
        {
            if (!descriptor.ActionAvailable || descriptor.Execute is null)
            {
                continue;
            }

            var actionId = $"event_option:{descriptor.Index}";
            actions.Add(new BridgeResolvedAction
            {
                ActionId = actionId,
                Payload = new
                {
                    action_id = actionId,
                    kind = "event_option",
                    index = descriptor.Index,
                    label = $"Choose option {descriptor.Index}: {descriptor.Title}",
                    option = BuildCrystalSphereEventOptionPayload(descriptor),
                    screen = context.Screen
                },
                Execute = descriptor.Execute
            });
        }
    }

    private static object BuildCrystalSphereDivinationButtonPayload(
        Node? button,
        string expectedTool,
        string? currentTool,
        string divinationSize,
        string fallbackLabel)
    {
        var visible = button is not null && IsNodeVisible(button);
        var enabled = visible && IsButtonEnabled(button);
        var isSelected = string.Equals(currentTool, expectedTool, StringComparison.OrdinalIgnoreCase);

        return new
        {
            visible,
            enabled,
            is_selected = isSelected,
            action_available = enabled && !isSelected,
            label = GetCrystalSphereControlLabel(button, fallbackLabel),
            divination_size = divinationSize
        };
    }

    private static object BuildCrystalSphereProceedPayload(Node? button)
    {
        var visible = button is not null && IsNodeVisible(button);
        var enabled = visible && IsButtonEnabled(button);
        return new
        {
            visible,
            enabled,
            action_available = enabled,
            label = GetCrystalSphereControlLabel(button, "Proceed")
        };
    }

    private static object BuildCrystalSphereCellPayload(NCrystalSphereCell cell, bool canRevealHiddenCells)
    {
        var entity = cell.Entity;
        var isHidden = entity?.IsHidden ?? true;
        return new
        {
            x = entity?.X,
            y = entity?.Y,
            is_hidden = isHidden,
            is_highlighted = entity?.IsHighlighted ?? false,
            is_hovered = entity?.IsHovered ?? false,
            can_reveal = canRevealHiddenCells && isHidden,
            revealed_item = isHidden
                ? null
                : BuildCrystalSphereRevealedItemPayload(GetHiddenPropertyObjectValue(entity, "Item"))
        };
    }

    private static object? BuildCrystalSphereRevealedItemPayload(object? item)
    {
        if (item is null)
        {
            return null;
        }

        var itemTypeName = item.GetType().Name;
        var itemType = itemTypeName switch
        {
            "CrystalSphereGold" => "gold",
            "CrystalSpherePotion" => "potion",
            "CrystalSphereRelic" => "relic",
            "CrystalSphereCardReward" => "card_reward",
            "CrystalSphereCurse" => "curse",
            _ => NormalizeComparableText(itemTypeName).Replace("-", "_", StringComparison.Ordinal)
        };

        var potionModel = GetHiddenPropertyObjectValue(item, "Potion") as PotionModel ??
                          GetHiddenFieldValue(item, "_potion") as PotionModel;
        var relicModel = GetHiddenPropertyObjectValue(item, "Relic") as RelicModel ??
                         GetHiddenPropertyObjectValue(item, "Model") as RelicModel;

        string? title = itemType switch
        {
            "gold" => "Gold",
            "potion" when potionModel is not null => TryGetTitle(potionModel),
            "relic" when relicModel is not null => TryGetTitle(relicModel),
            "curse" => "Curse",
            _ => null
        };

        var amount = itemType == "gold"
            ? TryGetIntFromPropertyOrField(item, "Amount")
            : null;
        var rarity = itemType == "card_reward"
            ? TextOf(GetHiddenPropertyObjectValue(item, "Rarity") ?? GetHiddenFieldValue(item, "_rarity"))
            : null;

        return new
        {
            item_type = itemType,
            title,
            amount,
            rarity,
            is_good = TryGetBoolFromPropertyOrField(item, "IsGood") ?? false
        };
    }

    private static object? BuildCrystalSphereGridPayload(
        object? minigame,
        IReadOnlyList<NCrystalSphereCell> crystalSphereCells)
    {
        int? width = null;
        int? height = null;

        if (crystalSphereCells.Count > 0)
        {
            width = crystalSphereCells.Max(static cell => cell.Entity?.X ?? -1) + 1;
            height = crystalSphereCells.Max(static cell => cell.Entity?.Y ?? -1) + 1;
        }

        var gridSize = GetHiddenPropertyObjectValue(minigame, "GridSize") ?? GetHiddenFieldValue(minigame, "_gridSize");
        if (gridSize is not null)
        {
            width ??= TryConvertToInt(gridSize) ??
                      TryGetIntFromPropertyOrField(gridSize, "X", "Width", "Columns", "Col");
            height ??= TryConvertToInt(gridSize) ??
                       TryGetIntFromPropertyOrField(gridSize, "Y", "Height", "Rows", "Row");
        }

        if (!width.HasValue && !height.HasValue)
        {
            return null;
        }

        width ??= height;
        height ??= width;
        return new
        {
            width,
            height
        };
    }

    private static object? BuildCrystalSphereCoordPayload(int? x, int? y)
    {
        return x.HasValue && y.HasValue
            ? new
            {
                x,
                y
            }
            : null;
    }

    private static object? GetCrystalSphereMinigame(NCrystalSphereScreen? crystalSphereScreen)
    {
        return GetHiddenFieldValue(crystalSphereScreen, "_entity") ??
               GetHiddenPropertyObjectValue(crystalSphereScreen, "Entity");
    }

    private static int? GetCrystalSphereDivinationCount(object? minigame)
    {
        return TryGetIntFromPropertyOrField(minigame, "DivinationCount", "_divinationCount");
    }

    private static bool GetCrystalSphereIsFinished(object? minigame)
    {
        return TryGetBoolFromPropertyOrField(minigame, "IsFinished", "_isFinished") ?? false;
    }

    private static string? GetCrystalSphereToolName(object? minigame)
    {
        var rawTool = GetHiddenPropertyObjectValue(minigame, "CrystalSphereTool") ??
                      GetHiddenFieldValue(minigame, "_crystalSphereTool") ??
                      GetHiddenFieldValue(minigame, "_currentTool");
        var toolText = rawTool?.ToString();
        if (string.IsNullOrWhiteSpace(toolText))
        {
            return null;
        }

        return toolText.Trim().ToLowerInvariant() switch
        {
            "none" => "none",
            "small" => "small",
            "big" => "big",
            _ => toolText.Trim()
        };
    }

    private static string GetCrystalSphereControlLabel(Node? button, string fallbackLabel)
    {
        var label = TryGetNodeText(button);
        return string.IsNullOrWhiteSpace(label) ? fallbackLabel : label;
    }

    private static object BuildDeckUpgradeSelectionPayload(
        NDeckUpgradeSelectScreen? deckUpgradeScreen,
        IReadOnlyList<NCardHolder> deckUpgradeOptions,
        NConfirmButton? deckUpgradeConfirmButton,
        NBackButton? deckUpgradeCancelButton,
        NBackButton? deckUpgradeCloseButton)
    {
        var visible = deckUpgradeScreen is not null && IsNodeVisible(deckUpgradeScreen);
        var prompt = visible ? TryGetDeckUpgradePrompt(deckUpgradeScreen) : null;
        var texts = visible ? CollectVisibleText(deckUpgradeScreen, 8).ToArray() : Array.Empty<string>();
        var selectedCount = CountSelectedDeckUpgradeCards(deckUpgradeScreen);
        var useSingleSelection = GetHiddenPropertyValue<bool>(deckUpgradeScreen, "UseSingleSelection") ?? false;
        var confirmVisible = deckUpgradeConfirmButton is not null &&
                             IsNodeVisible(deckUpgradeConfirmButton) &&
                             IsButtonEnabled(deckUpgradeConfirmButton);
        return new
        {
            visible = visible,
            use_single_selection = useSingleSelection,
            selected_count = selectedCount,
            selection_semantics = "upgrade",
            prompt,
            texts,
            decision_text = BuildDeckUpgradeDecisionText(prompt, useSingleSelection, selectedCount, confirmVisible),
            confirm_visible = confirmVisible,
            cancel_visible = deckUpgradeCancelButton is not null &&
                             IsNodeVisible(deckUpgradeCancelButton) &&
                             IsButtonEnabled(deckUpgradeCancelButton),
            close_visible = deckUpgradeCloseButton is not null &&
                            IsNodeVisible(deckUpgradeCloseButton) &&
                            IsButtonEnabled(deckUpgradeCloseButton),
            options = deckUpgradeOptions.Select((holder, index) => new
            {
                index,
                card = BuildCardPayload(holder.CardModel),
                upgrade_preview = BuildCardUpgradePreviewPayload(holder.CardModel),
                is_selected = IsDeckUpgradeCardSelected(deckUpgradeScreen, holder.CardModel)
            }).ToArray()
        };
    }

    private static object BuildMainMenuPayload(
        Node? mainMenuRoot,
        Node? mainMenuContinueButton,
        IReadOnlyList<Node> mainMenuTextButtons,
        Node? continueRunInfo,
        Node? abandonRunConfirmPopup,
        IReadOnlyList<NPopupYesNoButton> abandonRunConfirmButtons)
    {
        return new
        {
            visible = mainMenuRoot is not null && IsNodeVisible(mainMenuRoot),
            continue_button = BuildMainMenuButtonPayload(mainMenuContinueButton, null, "continue"),
            continue_run_info = new
            {
                visible = continueRunInfo is not null && IsNodeVisible(continueRunInfo),
                texts = CollectVisibleText(continueRunInfo, 8).ToArray()
            },
            buttons = mainMenuTextButtons
                .Select((button, index) => BuildMainMenuButtonPayload(button, index, TryGetMainMenuSemanticAction(TryGetNodeText(button))))
                .ToArray(),
            abandon_confirm = new
            {
                visible = abandonRunConfirmPopup is not null && IsNodeVisible(abandonRunConfirmPopup),
                buttons = abandonRunConfirmButtons
                    .Select((button, index) => BuildMainMenuButtonPayload(
                        button,
                        index,
                        TryGetAbandonConfirmSemanticAction(TryGetNodeText(button))))
                    .ToArray()
            }
        };
    }

    private static object BuildMainMenuButtonPayload(Node? button, int? index, string? semanticAction)
    {
        if (button is null || !IsNodeVisible(button))
        {
            return new
            {
                visible = false,
                index,
                semantic_action = semanticAction
            };
        }

        var text = TryGetNodeText(button);

        return new
        {
            visible = true,
            index,
            semantic_action = semanticAction,
            text,
            node_type = button.GetType().FullName
        };
    }

    private static object BuildEventOptionPayload(
        NEventOptionButton button,
        int index,
        EventModel? eventModel)
    {
        var option = button.Option;
        var glossary = BuildHoverTipPayloads(option?.HoverTips);
        object? optionTextContext = option is null
            ? eventModel
            : eventModel is null
                ? option
                : new object?[] { option, eventModel };

        return new
        {
            index,
            title = option is null ? string.Empty : DescribeText(option.Title, optionTextContext),
            description = option is null ? string.Empty : DescribeText(option.Description, optionTextContext),
            is_locked = option?.IsLocked ?? true,
            is_proceed = option?.IsProceed ?? false,
            relic = option?.Relic is null ? null : BuildRelicPayload(option.Relic),
            glossary
        };
    }

    private static (string? Source, IReadOnlyList<string> Texts) CollectVisibleEventGlossaryTexts(
        IReadOnlyList<NEventOptionButton> eventOptionButtons,
        NEventRoom? eventRoom,
        Node? hoverTipSet)
    {
        var excludedTexts = new HashSet<string>(
            eventOptionButtons
                .SelectMany(static button => CollectVisibleText(button, 6))
                .Select(NormalizeComparableText)
                .Where(static text => !string.IsNullOrWhiteSpace(text)),
            StringComparer.Ordinal);

        var hoverTipTexts = FilterGlossaryCandidateTexts(CollectVisibleText(hoverTipSet, 24), excludedTexts);
        if (hoverTipTexts.Count > 0)
        {
            return ("hover_tip_set", hoverTipTexts);
        }

        var eventRoomTexts = FilterGlossaryCandidateTexts(CollectVisibleText(eventRoom, 48), excludedTexts);
        if (eventRoomTexts.Count > 0)
        {
            return ("event_room_fallback", eventRoomTexts);
        }

        return (null, Array.Empty<string>());
    }

    private static IReadOnlyList<string> FilterGlossaryCandidateTexts(
        IEnumerable<string> texts,
        IReadOnlySet<string> excludedTexts)
    {
        var filtered = new List<string>();
        var seen = new HashSet<string>(StringComparer.Ordinal);

        foreach (var rawText in texts)
        {
            var comparableText = NormalizeComparableText(rawText);
            if (string.IsNullOrWhiteSpace(comparableText) ||
                excludedTexts.Contains(comparableText) ||
                !seen.Add(comparableText))
            {
                continue;
            }

            filtered.Add(rawText.ReplaceLineEndings("\n").Trim());
        }

        return filtered;
    }

    private static object[] BuildVisibleGlossaryPayload(IReadOnlyList<string> glossaryTexts)
    {
        if (glossaryTexts.Count == 0)
        {
            return Array.Empty<object>();
        }

        var entries = new List<object>();

        for (var index = 0; index < glossaryTexts.Count; index++)
        {
            var title = glossaryTexts[index];
            string? description = null;

            if (index + 1 < glossaryTexts.Count &&
                LooksLikeGlossaryTitle(title) &&
                LooksLikeGlossaryDescription(glossaryTexts[index + 1], title))
            {
                description = glossaryTexts[index + 1];
                index++;
            }

            entries.Add(new
            {
                title,
                description,
                texts = description is null ? new[] { title } : new[] { title, description }
            });
        }

        return entries.ToArray();
    }

    private static object[] BuildHoverTipPayloads(IEnumerable? hoverTips)
    {
        if (hoverTips is null)
        {
            return Array.Empty<object>();
        }

        var entries = new List<object>();
        var seen = new HashSet<string>(StringComparer.Ordinal);

        foreach (var hoverTip in hoverTips)
        {
            if (hoverTip is null)
            {
                continue;
            }

            var canonicalModel = GetHiddenPropertyObjectValue(hoverTip, "CanonicalModel") as AbstractModel;
            var id = TextOf(GetHiddenPropertyObjectValue(hoverTip, "Id"));
            var title = ResolveHoverTipTitle(hoverTip, canonicalModel, id);
            var description = ResolveHoverTipDescription(hoverTip, canonicalModel);
            var dedupeKey = string.Join(
                "|",
                hoverTip.GetType().FullName ?? hoverTip.GetType().Name,
                NormalizeComparableText(id),
                NormalizeComparableText(title),
                NormalizeComparableText(description));

            if (!seen.Add(dedupeKey))
            {
                continue;
            }

            entries.Add(new
            {
                id,
                type = hoverTip.GetType().Name,
                title,
                description,
                is_debuff = GetHiddenPropertyValue<bool>(hoverTip, "IsDebuff") ?? false,
                is_instanced = GetHiddenPropertyValue<bool>(hoverTip, "IsInstanced") ?? false,
                is_smart = GetHiddenPropertyValue<bool>(hoverTip, "IsSmart") ?? false,
                canonical_model = canonicalModel is null ? null : BuildModelPayload(canonicalModel),
                texts = new[] { title, description }
                    .Where(static text => !string.IsNullOrWhiteSpace(text))
                    .ToArray()
            });
        }

        return entries.ToArray();
    }

    private static string ResolveHoverTipTitle(object hoverTip, AbstractModel? canonicalModel, string fallbackId)
    {
        return FirstNonEmptyText(
            TryGetNamedValueText(hoverTip, "HoverTipTitle"),
            TryGetNamedValueText(hoverTip, "Title"),
            TryGetNamedValueText(hoverTip, "Name"),
            TryGetNamedValueText(hoverTip, "Label"),
            TryGetNamedValueText(hoverTip, "BotKeyword"),
            canonicalModel is null ? string.Empty : TryGetTitle(canonicalModel),
            fallbackId);
    }

    private static string ResolveHoverTipDescription(object hoverTip, AbstractModel? canonicalModel)
    {
        return FirstNonEmptyText(
            TryGetNamedValueText(hoverTip, "HoverTipDesc"),
            TryGetNamedValueText(hoverTip, "Description"),
            TryGetNamedValueText(hoverTip, "Text"),
            TryGetNamedValueText(hoverTip, "Body"),
            TryGetNamedValueText(hoverTip, "BotText"),
            canonicalModel is null ? string.Empty : TryGetDescription(canonicalModel));
    }

    private static string TryGetNamedValueText(object target, string memberName)
    {
        return DescribeText(
            GetHiddenPropertyObjectValue(target, memberName) ??
            GetHiddenFieldValue(target, memberName),
            target);
    }

    private static string FirstNonEmptyText(params string[] candidates)
    {
        foreach (var candidate in candidates)
        {
            if (!string.IsNullOrWhiteSpace(candidate))
            {
                return candidate;
            }
        }

        return string.Empty;
    }

    private static bool LooksLikeGlossaryTitle(string text)
    {
        if (string.IsNullOrWhiteSpace(text))
        {
            return false;
        }

        var normalized = text.ReplaceLineEndings("\n").Trim();
        return normalized.Length <= 32 &&
               !normalized.Contains('\n') &&
               !normalized.Contains('。') &&
               !normalized.Contains('！') &&
               !normalized.Contains('？') &&
               !normalized.Contains('：');
    }

    private static bool LooksLikeGlossaryDescription(string text, string title)
    {
        if (string.IsNullOrWhiteSpace(text))
        {
            return false;
        }

        var normalized = text.ReplaceLineEndings("\n").Trim();
        var normalizedTitle = title.ReplaceLineEndings("\n").Trim();
        if (string.Equals(normalized, normalizedTitle, StringComparison.Ordinal))
        {
            return false;
        }

        return normalized.Contains('\n') ||
               normalized.Contains('。') ||
               normalized.Contains('！') ||
               normalized.Contains('？') ||
               normalized.Contains('：') ||
               normalized.Length > normalizedTitle.Length;
    }

    private static object BuildMapPayload(
        RunState? runState,
        NMapScreen? mapScreen,
        IReadOnlyList<NMapPoint> mapPoints)
    {
        var map = runState?.Map;

        return new
        {
            is_open = mapScreen?.IsOpen ?? false,
            is_travel_enabled = mapScreen?.IsTravelEnabled ?? false,
            is_traveling = mapScreen?.IsTraveling ?? false,
            current_coord = BuildMapCoord(runState?.CurrentMapCoord),
            dimensions = map is null
                ? null
                : new
                {
                    rows = map.GetRowCount(),
                    columns = map.GetColumnCount()
                },
            points = mapPoints.Select(BuildMapPointPayload).ToArray()
        };
    }

    private static object BuildRestSitePayload(
        NMapScreen? mapScreen,
        NRestSiteRoom? restSiteRoom,
        IReadOnlyList<NRestSiteButton> restSiteButtons,
        NProceedButton? restSiteProceedButton)
    {
        var visible = mapScreen?.IsOpen != true &&
                      restSiteRoom is not null &&
                      IsNodeVisible(restSiteRoom);
        var options = visible
            ? restSiteButtons
                .Where(IsNodeVisible)
                .Select((button, index) => BuildRestSiteOptionPayload(button.Option, index))
                .ToArray()
            : new object[0];

        return new
        {
            visible,
            header = visible
                ? TryGetNodeText(GetHiddenFieldValue(restSiteRoom, "<Header>k__BackingField") as Node)
                : null,
            description = visible
                ? TryGetNodeText(GetHiddenFieldValue(restSiteRoom, "<Description>k__BackingField") as Node)
                : null,
            proceed_visible = visible &&
                              !HasVisibleEnabledRestSiteOptions(restSiteButtons) &&
                              restSiteProceedButton is not null &&
                              IsNodeVisible(restSiteProceedButton) &&
                              IsButtonEnabled(restSiteProceedButton),
            options
        };
    }

    private static object BuildShopPayload(
        NMerchantRoom? merchantRoom,
        NMerchantInventory? merchantInventory,
        IReadOnlyList<NMerchantSlot> merchantSlots,
        NMerchantButton? merchantButton,
        NProceedButton? merchantProceedButton,
        NBackButton? merchantBackButton)
    {
        var visible = (merchantRoom is not null && IsNodeVisible(merchantRoom)) ||
                      (merchantInventory is not null && IsNodeVisible(merchantInventory));
        var inventory = merchantInventory?.Inventory;

        return new
        {
            visible,
            is_open = merchantInventory?.IsOpen ?? false,
            gold = inventory?.Player?.Gold,
            merchant_button_visible = merchantButton is not null && IsNodeVisible(merchantButton),
            back_button_visible = merchantBackButton is not null && IsNodeVisible(merchantBackButton),
            proceed_visible = merchantProceedButton is not null && IsNodeVisible(merchantProceedButton),
            items = merchantSlots.Select((slot, index) => BuildShopSlotPayload(slot, index)).ToArray()
        };
    }

    private static object BuildShopSlotPayload(NMerchantSlot slot, int index)
    {
        var entry = slot.Entry;
        var cardEntry = entry as MerchantCardEntry;
        var relicEntry = entry as MerchantRelicEntry;
        var potionEntry = entry as MerchantPotionEntry;
        var cardRemovalEntry = entry as MerchantCardRemovalEntry;

        return new
        {
            index,
            slot_type = slot.GetType().Name,
            item_kind = ResolveMerchantEntryKind(entry),
            title = DescribeMerchantEntry(entry),
            description = DescribeMerchantEntryDescription(entry),
            cost = entry?.Cost,
            enough_gold = entry?.EnoughGold ?? false,
            is_stocked = entry?.IsStocked ?? false,
            is_affordable = CanPurchaseMerchantEntry(entry),
            is_on_sale = cardEntry?.IsOnSale,
            used = cardRemovalEntry?.Used,
            card = cardEntry is null ? null : BuildCardPayload(cardEntry.CreationResult?.Card),
            relic = relicEntry is null ? null : BuildRelicPayload(relicEntry.Model),
            potion = potionEntry is null ? null : BuildPotionPayload(potionEntry.Model)
        };
    }

    private static string ResolveMerchantEntryKind(MerchantEntry? entry)
    {
        return entry switch
        {
            MerchantCardEntry => "card",
            MerchantRelicEntry => "relic",
            MerchantPotionEntry => "potion",
            MerchantCardRemovalEntry => "card_removal",
            null => "missing",
            _ => entry.GetType().Name
        };
    }

    private static string DescribeMerchantEntry(MerchantEntry? entry)
    {
        return entry switch
        {
            MerchantCardEntry cardEntry => TextOf(cardEntry.CreationResult?.Card?.Title),
            MerchantRelicEntry relicEntry => TextOf(relicEntry.Model?.Title),
            MerchantPotionEntry potionEntry => TextOf(potionEntry.Model?.Title),
            MerchantCardRemovalEntry => "Remove a card",
            null => "<missing>",
            _ => entry.GetType().Name
        };
    }

    private static string DescribeMerchantEntryDescription(MerchantEntry? entry)
    {
        return entry switch
        {
            MerchantCardEntry cardEntry when cardEntry.CreationResult?.Card is CardModel card => GetCardDescription(card, null),
            MerchantRelicEntry relicEntry => relicEntry.Model is null ? string.Empty : TryGetDescription(relicEntry.Model),
            MerchantPotionEntry potionEntry => potionEntry.Model is null ? string.Empty : TryGetDescription(potionEntry.Model),
            MerchantCardRemovalEntry cardRemovalEntry => cardRemovalEntry.Used
                ? "Card removal already used"
                : "Remove a card from your deck",
            null => string.Empty,
            _ => string.Empty
        };
    }

    private static object BuildRestSiteOptionPayload(RestSiteOption? option, int index)
    {
        if (option is null)
        {
            return new
            {
                index,
                missing = true
            };
        }

        return new
        {
            index,
            option_id = option.OptionId,
            option_type = option.GetType().Name,
            title = TryGetTitle(option),
            description = BuildRestSiteOptionDescription(option),
            is_enabled = option.IsEnabled
        };
    }

    private static string BuildRestSiteOptionDescription(RestSiteOption option)
    {
        switch (option)
        {
            case HealRestSiteOption healOption:
            {
                var owner = GetHiddenPropertyObjectValue(healOption, "Owner") as Player;
                if (owner is not null)
                {
                    return $"回复{FormatNumericValue(HealRestSiteOption.GetHealAmount(owner))}点生命值。";
                }

                return "回复生命值。";
            }
            case SmithRestSiteOption smithOption:
                return smithOption.IsEnabled
                    ? $"升级你牌组中的{smithOption.SmithCount}张牌。"
                    : "没有可升级的牌。";
            default:
                return TryGetDescription(option);
        }
    }

    private static object BuildAutomationPayload()
    {
        return new
        {
            autoslay = BridgeAutoSlay.GetStatusPayload()
        };
    }

    private static object BuildMapPointPayload(NMapPoint pointNode)
    {
        var point = pointNode.Point;
        var coord = point.coord;

        return new
        {
            coord = BuildMapCoord(coord),
            point_type = point.PointType.ToString(),
            state = pointNode.State.ToString(),
            is_enabled = pointNode.IsEnabled,
            is_travelable = IsMapPointTravelable(pointNode),
            children = point.Children.Select(static child => BuildMapCoord(child.coord)).ToArray()
        };
    }

    private static object BuildRoomPayload(AbstractRoom? room)
    {
        if (room is null)
        {
            return new
            {
                missing = true
            };
        }

        return new
        {
            room_type = room.RoomType.ToString(),
            model_id = room.ModelId?.ToString() ?? string.Empty,
            is_pre_finished = room.IsPreFinished,
            is_victory_room = room.IsVictoryRoom
        };
    }

    private static object BuildModelPayload(AbstractModel? model)
    {
        if (model is null)
        {
            return new
            {
                missing = true
            };
        }

        return new
        {
            id = model.Id.ToString(),
            title = TryGetTitle(model),
            description = TryGetDescription(model),
            kind = model.GetType().Name
        };
    }

    private static object BuildCharacterPayload(CharacterModel? character)
    {
        if (character is null)
        {
            return new
            {
                missing = true
            };
        }

        return new
        {
            id = character.Id.ToString(),
            title = DescribeCharacter(character),
            description = DescribeCharacterDescription(character),
            starting_hp = character.StartingHp,
            starting_gold = character.StartingGold,
            starting_relic = BuildRelicPayload(character.StartingRelics.FirstOrDefault())
        };
    }

    private static object BuildRelicPayload(RelicModel? relic)
    {
        if (relic is null)
        {
            return new
            {
                missing = true
            };
        }

        return new
        {
            id = relic.Id.ToString(),
            title = TryGetTitle(relic),
            description = TryGetDescription(relic),
            rarity = relic.Rarity.ToString()
        };
    }

    private static object BuildPotionPayload(PotionModel? potion)
    {
        if (potion is null)
        {
            return new
            {
                empty = true
            };
        }

        return new
        {
            id = potion.Id.ToString(),
            title = TryGetTitle(potion),
            description = TryGetDescription(potion),
            rarity = potion.Rarity.ToString(),
            target_type = potion.TargetType.ToString(),
            selection_screen_prompt = DescribeText(potion.SelectionScreenPrompt, potion),
            can_throw_at_ally = SafeCanThrowPotionAtAlly(potion),
            is_usable = SafeGetPotionIsUsable(potion),
            is_queued = SafeGetPotionIsQueued(potion)
        };
    }

    private static object? BuildMapCoord(MapCoord? coord)
    {
        return coord.HasValue ? BuildMapCoord(coord.Value) : null;
    }

    private static object BuildMapCoord(MapCoord coord)
    {
        return new
        {
            col = coord.col,
            row = coord.row
        };
    }

    private static string ResolveCurrentScreen(
        ScreenStateTracker? screenStateTracker,
        CombatManager? combatManager,
        NMapScreen? mapScreen,
        NRestSiteRoom? restSiteRoom,
        NMerchantRoom? merchantRoom,
        NMerchantInventory? merchantInventory,
        NRewardsScreen? rewardsScreen,
        NProceedButton? roomProceedButton,
        NProceedButton? rewardProceedButton,
        IReadOnlyList<NRewardButton> rewardButtons,
        NCardRewardSelectionScreen? cardRewardScreen,
        IReadOnlyList<NCardHolder> cardRewardOptions,
        Node? cardSelectionScreen,
        NCharacterSelectScreen? characterSelectScreen,
        NDeckUpgradeSelectScreen? deckUpgradeScreen,
        NCrystalSphereScreen? crystalSphereScreen,
        IReadOnlyList<NEventOptionButton> eventOptionButtons,
        NGameOverScreen? gameOverScreen,
        Node? mainMenuRoot,
        Node? runModeSubmenu,
        Node? abandonRunConfirmPopup)
    {
        var isRestSiteVisible = restSiteRoom is not null && IsNodeVisible(restSiteRoom);
        var isMerchantVisible = (merchantRoom is not null && IsNodeVisible(merchantRoom)) ||
                                merchantInventory?.IsOpen == true;
        var isMapOpen = mapScreen is not null && mapScreen.IsOpen;
        var isRewardsVisible = IsRewardsScreenVisible(
            rewardsScreen,
            roomProceedButton,
            rewardProceedButton,
            mapScreen,
            rewardButtons);
        var isCardRewardVisible = IsCardRewardSelectionVisible(cardRewardScreen, cardRewardOptions);
        var isCardSelectionVisible = cardSelectionScreen is not null && IsNodeVisible(cardSelectionScreen);
        var isDeckUpgradeVisible = deckUpgradeScreen is not null && IsNodeVisible(deckUpgradeScreen);
        var isRunModeVisible = runModeSubmenu is not null && IsNodeVisible(runModeSubmenu);
        var isCharacterSelectVisible = characterSelectScreen is not null && IsNodeVisible(characterSelectScreen);
        var isCrystalSphereVisible = crystalSphereScreen is not null && IsNodeVisible(crystalSphereScreen);
        var isGameOverVisible = gameOverScreen is not null && IsNodeVisible(gameOverScreen);
        var currentScreen = InvokeParameterless(screenStateTracker, "GetCurrentScreen");
        if (currentScreen is not null)
        {
            var currentScreenText = currentScreen.ToString() ?? "UNKNOWN";
            if (abandonRunConfirmPopup is not null && IsNodeVisible(abandonRunConfirmPopup))
            {
                return "ABANDON_RUN_CONFIRM";
            }

            if (isDeckUpgradeVisible)
            {
                return "DECK_UPGRADE_SELECTION";
            }

            if (isCardSelectionVisible)
            {
                return "CARD_SELECTION";
            }

            if (isCardRewardVisible)
            {
                return "CARD_REWARD_SELECTION";
            }

            if (isRewardsVisible)
            {
                return "REWARDS";
            }

            if (isMapOpen)
            {
                return "MAP";
            }

            if (isCrystalSphereVisible)
            {
                return "EVENT_CRYSTAL_SPHERE";
            }

            if (isGameOverVisible)
            {
                return "GAME_OVER";
            }

            if (isCharacterSelectVisible)
            {
                return "CHARACTER_SELECT";
            }

            if (combatManager?.IsInProgress == true &&
                currentScreenText.Equals("Room", StringComparison.OrdinalIgnoreCase))
            {
                return "COMBAT";
            }

            if (currentScreenText.Equals("Room", StringComparison.OrdinalIgnoreCase))
            {
                if (isRestSiteVisible)
                {
                    return "REST_SITE";
                }

                if (isMerchantVisible)
                {
                    return "SHOP";
                }
            }

            if (isRunModeVisible)
            {
                return "RUN_MODE_SELECTION";
            }

            if (isMerchantVisible &&
                (currentScreenText.Contains("merchant", StringComparison.OrdinalIgnoreCase) ||
                 currentScreenText.Contains("shop", StringComparison.OrdinalIgnoreCase)))
            {
                return "SHOP";
            }

            return currentScreenText;
        }

        if (abandonRunConfirmPopup is not null && IsNodeVisible(abandonRunConfirmPopup))
        {
            return "ABANDON_RUN_CONFIRM";
        }

        if (isRunModeVisible)
        {
            return "RUN_MODE_SELECTION";
        }

        if (isCharacterSelectVisible)
        {
            return "CHARACTER_SELECT";
        }

        if (isDeckUpgradeVisible)
        {
            return "DECK_UPGRADE_SELECTION";
        }

        if (isCardSelectionVisible)
        {
            return "CARD_SELECTION";
        }

        if (mainMenuRoot is not null && IsNodeVisible(mainMenuRoot))
        {
            return "MAIN_MENU";
        }

        if (isGameOverVisible)
        {
            return "GAME_OVER";
        }

        if (isCrystalSphereVisible)
        {
            return "EVENT_CRYSTAL_SPHERE";
        }

        if (isMapOpen)
        {
            return "MAP";
        }

        if (restSiteRoom is not null && IsNodeVisible(restSiteRoom))
        {
            return "REST_SITE";
        }

        if (isMerchantVisible)
        {
            return "SHOP";
        }

        if (isCardRewardVisible)
        {
            return "CARD_REWARD_SELECTION";
        }

        if (isRewardsVisible)
        {
            return "REWARDS";
        }

        if (isMapOpen)
        {
            return "MAP";
        }

        if (eventOptionButtons.Count > 0)
        {
            return "EVENT";
        }

        if (combatManager?.IsInProgress == true)
        {
            return "COMBAT";
        }

        return "UNKNOWN";
    }

    private static RunState? TryGetRunState(RunManager? runManager)
    {
        if (runManager is null)
        {
            return null;
        }

        try
        {
            return runManager.DebugOnlyGetState();
        }
        catch
        {
            return null;
        }
    }

    private static CombatState? TryGetCombatState(CombatManager? combatManager)
    {
        if (combatManager is null)
        {
            return null;
        }

        try
        {
            return combatManager.DebugOnlyGetState();
        }
        catch
        {
            return null;
        }
    }

    private static void EnsureDispatcherReady()
    {
        if (!BridgeCoordinator.IsReady)
        {
            throw new BridgeRequestException(
                HttpStatusCode.ServiceUnavailable,
                "dispatcher_not_ready",
                "The bridge dispatcher is not attached yet. Wait for the game to finish loading and try again.");
        }
    }

    private static bool IsNodeVisible(Node? node)
    {
        if (node is null || !GodotObject.IsInstanceValid(node))
        {
            return false;
        }

        if (BridgeRuntime.VisibleOnly && node is CanvasItem canvasItem)
        {
            return canvasItem.IsVisibleInTree();
        }

        return true;
    }

    private static bool IsSameNodeInstance(Node? left, Node? right)
    {
        if (left is null || right is null)
        {
            return false;
        }

        if (ReferenceEquals(left, right))
        {
            return true;
        }

        if (!GodotObject.IsInstanceValid(left) || !GodotObject.IsInstanceValid(right))
        {
            return false;
        }

        return left.NativeInstance == right.NativeInstance;
    }

    private static bool IsTypeFullName(Node? node, string fullTypeName)
    {
        return node is not null &&
               GodotObject.IsInstanceValid(node) &&
               string.Equals(node.GetType().FullName, fullTypeName, StringComparison.Ordinal);
    }

    private static bool IsMapPointTravelable(NMapPoint pointNode)
    {
        return GetHiddenPropertyValue<bool>(pointNode, "IsTravelable") ?? false;
    }

    private static bool HasVisibleEnabledRestSiteOptions(IReadOnlyList<NRestSiteButton> restSiteButtons)
    {
        return restSiteButtons.Any(static button =>
            IsNodeVisible(button) &&
            button.Option is { IsEnabled: true });
    }

    private static bool IsButtonEnabled(object? target)
    {
        return target is not null && (GetHiddenPropertyValue<bool>(target, "IsEnabled") ?? true);
    }

    private static bool IsRunModeSelectionVisible(BridgeWorldContext context)
    {
        return context.RunModeSubmenu is not null && IsNodeVisible(context.RunModeSubmenu);
    }

    private static bool IsRewardsScreenVisible(
        NRewardsScreen? rewardsScreen,
        NProceedButton? roomProceedButton,
        NProceedButton? rewardProceedButton,
        NMapScreen? mapScreen,
        IReadOnlyList<NRewardButton> rewardButtons)
    {
        if (mapScreen?.IsOpen == true && rewardButtons.Count == 0)
        {
            return false;
        }

        if (rewardButtons.Count == 0 &&
            roomProceedButton is not null &&
            IsNodeVisible(roomProceedButton) &&
            !IsSameNodeInstance(roomProceedButton, rewardProceedButton))
        {
            return false;
        }

        return (rewardsScreen is not null && IsNodeVisible(rewardsScreen)) ||
               (rewardProceedButton is not null && IsNodeVisible(rewardProceedButton));
    }

    private static bool IsCardRewardSelectionVisible(
        NCardRewardSelectionScreen? cardRewardScreen,
        IReadOnlyList<NCardHolder> cardRewardOptions)
    {
        return (cardRewardScreen is not null && IsNodeVisible(cardRewardScreen)) ||
               cardRewardOptions.Count > 0;
    }

    private static bool IsCardRewardSelectionReady(NCardRewardSelectionScreen? cardRewardScreen)
    {
        return cardRewardScreen is not null &&
               IsNodeVisible(cardRewardScreen) &&
               GetHiddenFieldValue(cardRewardScreen, "_completionSource") is not null;
    }

    private static bool IsDeckUpgradeSelectionVisible(BridgeWorldContext context)
    {
        return context.DeckUpgradeScreen is not null && IsNodeVisible(context.DeckUpgradeScreen);
    }

    private static bool IsCardSelectionVisible(BridgeWorldContext context)
    {
        return context.CardSelectionScreen is not null && IsNodeVisible(context.CardSelectionScreen);
    }

    private static bool IsTerminalRewardsProceedVisible(BridgeWorldContext context)
    {
        return IsRewardsScreenVisible(
                   context.RewardsScreen,
                   context.ProceedButton,
                   context.RewardProceedButton,
                   context.MapScreen,
                   context.RewardButtons) &&
               context.RewardProceedButton is not null &&
               IsNodeVisible(context.RewardProceedButton) &&
               context.RewardButtons.Count == 0 &&
               context.MapScreen?.IsOpen != true &&
               !IsCardRewardSelectionVisible(context.CardRewardScreen, context.CardRewardOptions);
    }

    private static bool IsRewardResolutionAction(string actionId)
    {
        return actionId.StartsWith("reward:", StringComparison.Ordinal) ||
               actionId.StartsWith("card_reward:", StringComparison.Ordinal);
    }

    private static bool IsCardSelectionResolutionAction(string actionId)
    {
        return actionId.StartsWith("card_selection:select:", StringComparison.Ordinal);
    }

    private static async Task<(ObservedFrontier Frontier, List<object> AutoExecutedActions)> MaybeAutoProceedAfterRewardActionAsync(
        ObservedFrontier frontier,
        CancellationToken cancellationToken)
    {
        var autoExecutedActions = new List<object>();
        var autoProceedCount = 0;

        for (var attempt = 0; attempt < 12; attempt++)
        {
            var nonAutomationActions = GetNonAutomationActions(frontier.Snapshot);

            if (nonAutomationActions.Length == 1 &&
                nonAutomationActions[0].ActionId.Equals("proceed", StringComparison.Ordinal))
            {
                if (autoProceedCount >= 3)
                {
                    return (frontier, autoExecutedActions);
                }

                var beforeAutoProceed = frontier;
                frontier = await ExecuteActionAndWaitForFrontierAsync(
                    frontier,
                    "proceed",
                    nonAutomationActions[0],
                    waitAfterMs: 0,
                    cancellationToken);
                autoProceedCount++;
                var stateChanged = HasFrontierChanged(beforeAutoProceed, frontier);
                autoExecutedActions.Add(new
                {
                    action_id = "proceed",
                    source = "auto_after_reward",
                    wait_after_ms = 0,
                    state_changed = stateChanged
                });

                if (!stateChanged)
                {
                    return (frontier, autoExecutedActions);
                }

                continue;
            }

            if (nonAutomationActions.Length > 0)
            {
                return (frontier, autoExecutedActions);
            }

            if (attempt >= 11)
            {
                break;
            }

            var beforePassiveWait = frontier;
            frontier = await WaitForNextObservedFrontierAsync(
                frontier,
                PassiveFrontierWaitTimeoutMs,
                cancellationToken);
            if (!HasFrontierChanged(beforePassiveWait, frontier))
            {
                break;
            }
        }

        return (frontier, autoExecutedActions);
    }

    private static async Task<(ObservedFrontier Frontier, List<object> AutoExecutedActions)> MaybeAutoCompleteCardSelectionAsync(
        ObservedFrontier frontier,
        CancellationToken cancellationToken)
    {
        var autoExecutedActions = new List<object>();

        if (!ShouldAutoCompleteCardSelection(frontier.Snapshot))
        {
            return (frontier, autoExecutedActions);
        }

        if (!frontier.Snapshot.ActionLookup.TryGetValue("card_selection:confirm", out var confirmAction))
        {
            return (frontier, autoExecutedActions);
        }

        frontier = await ExecuteActionAndWaitForFrontierAsync(
            frontier,
            "card_selection:confirm",
            confirmAction,
            waitAfterMs: 0,
            cancellationToken);
        autoExecutedActions.Add(new
        {
            action_id = "card_selection:confirm",
            source = "auto_after_card_selection",
            wait_after_ms = 0
        });

        return (frontier, autoExecutedActions);
    }

    private static bool ShouldAutoCompleteCardSelection(BridgeSnapshot snapshot)
    {
        var cardSelection = JsonSerializer.SerializeToElement(snapshot.Fields.CardSelection);
        if (!cardSelection.TryGetProperty("visible", out var visibleProperty) ||
            !visibleProperty.GetBoolean())
        {
            return false;
        }

        if (!cardSelection.TryGetProperty("confirm_visible", out var confirmVisibleProperty) ||
            !confirmVisibleProperty.GetBoolean())
        {
            return false;
        }

        if (!cardSelection.TryGetProperty("selected_count", out var selectedCountProperty) ||
            selectedCountProperty.GetInt32() <= 0)
        {
            return false;
        }

        var minSelect =
            cardSelection.TryGetProperty("min_select", out var minSelectProperty) &&
            minSelectProperty.ValueKind is not JsonValueKind.Null and not JsonValueKind.Undefined
                ? minSelectProperty.GetInt32()
                : 0;
        var maxSelect =
            cardSelection.TryGetProperty("max_select", out var maxSelectProperty) &&
            maxSelectProperty.ValueKind is not JsonValueKind.Null and not JsonValueKind.Undefined
                ? maxSelectProperty.GetInt32()
                : 0;

        return minSelect == 1 &&
               maxSelect == 1 &&
               snapshot.ActionLookup.ContainsKey("card_selection:confirm");
    }

    private static bool SafeGetCreatureIsHittable(Creature creature)
    {
        try
        {
            return creature.IsHittable;
        }
        catch
        {
            return false;
        }
    }

    private static bool SafeCanThrowPotionAtAlly(PotionModel potion)
    {
        try
        {
            return potion.CanThrowAtAlly();
        }
        catch
        {
            return false;
        }
    }

    private static bool SafeGetPotionIsUsable(PotionModel potion)
    {
        try
        {
            return potion.Owner is not null &&
                   !potion.HasBeenRemovedFromState &&
                   !potion.IsQueued &&
                   potion.PassesCustomUsabilityCheck;
        }
        catch
        {
            return false;
        }
    }

    private static bool SafeGetPotionIsQueued(PotionModel potion)
    {
        try
        {
            return potion.IsQueued;
        }
        catch
        {
            return false;
        }
    }

    private static List<T> FindVisibleDescendants<T>(Node? root) where T : Node
    {
        var result = new List<T>();
        if (root is null)
        {
            return result;
        }

        var seen = new HashSet<IntPtr>();

        void Visit(Node node)
        {
            if (!GodotObject.IsInstanceValid(node))
            {
                return;
            }

            if (node is T typed && seen.Add(typed.NativeInstance) && IsNodeVisible(typed))
            {
                result.Add(typed);
            }

            foreach (Node child in node.GetChildren())
            {
                Visit(child);
            }
        }

        Visit(root);
        return result;
    }

    private static List<Node> FindVisibleDescendants(Node? root, Func<Node, bool> predicate)
    {
        var result = new List<Node>();
        if (root is null)
        {
            return result;
        }

        var seen = new HashSet<IntPtr>();

        void Visit(Node node)
        {
            if (!GodotObject.IsInstanceValid(node))
            {
                return;
            }

            if (seen.Add(node.NativeInstance) && predicate(node) && IsNodeVisible(node))
            {
                result.Add(node);
            }

            foreach (Node child in node.GetChildren())
            {
                Visit(child);
            }
        }

        Visit(root);
        return result;
    }

    private static T? FindFirstVisibleDescendant<T>(Node? root) where T : Node
    {
        return FindVisibleDescendants<T>(root).FirstOrDefault();
    }

    private static Node? FindFirstVisibleDescendant(Node? root, Func<Node, bool> predicate)
    {
        return FindVisibleDescendants(root, predicate).FirstOrDefault();
    }

    private static List<T> SortByVisualPosition<T>(IEnumerable<T> nodes) where T : Node
    {
        return nodes
            .OrderBy(static node => node is Control control ? control.GlobalPosition.Y : 0f)
            .ThenBy(static node => node is Control control ? control.GlobalPosition.X : 0f)
            .ToList();
    }

    private static void InvokeMenuButtonAction(Node button)
    {
        if (TryInvokeParameterless(button, "ForceClick") ||
            TryInvokeParameterless(button, "OnRelease") ||
            TryInvokeParameterless(button, "OnPress") ||
            TryInvokeParameterless(button, "Pressed") ||
            TryInvokeParameterless(button, "OnButtonPressed"))
        {
            return;
        }

        throw new BridgeRequestException(
            HttpStatusCode.Conflict,
            "action_target_missing",
            $"Could not invoke a supported main-menu action on {button.GetType().FullName}.");
    }

    private static void InvokeClickablePressAndRelease(object target)
    {
        var didInvoke = false;

        if (TryInvokeParameterless(target, "OnPress"))
        {
            didInvoke = true;
        }

        if (TryInvokeParameterless(target, "OnRelease"))
        {
            didInvoke = true;
        }

        if (didInvoke || TryInvokeParameterless(target, "ForceClick"))
        {
            return;
        }

        throw new BridgeRequestException(
            HttpStatusCode.Conflict,
            "action_target_missing",
            $"Could not invoke click lifecycle on {target.GetType().FullName}.");
    }

    private static void InvokeProceedButtonAction(NProceedButton button)
    {
        InvokeClickablePressAndRelease(button);
    }

    private static void InvokeCrystalSphereDivinationAction(
        NCrystalSphereScreen? crystalSphereScreen,
        NDivinationButton button,
        bool useBigDivination)
    {
        if (crystalSphereScreen is not null)
        {
            InvokeSingleArgumentAction(
                crystalSphereScreen,
                useBigDivination ? "SetBigDivination" : "SetSmallDivination",
                button);
            return;
        }

        InvokeButtonAction(button, "OnRelease", "OnPress");
    }

    private static void InvokeCrystalSphereCellAction(
        NCrystalSphereScreen? crystalSphereScreen,
        NCrystalSphereCell cell)
    {
        if (crystalSphereScreen is not null)
        {
            InvokeSingleArgumentAction(crystalSphereScreen, "OnCellClicked", cell);
            return;
        }

        InvokeButtonAction(cell, "EntityClicked");
    }

    private static void InvokeCrystalSphereProceedAction(
        NCrystalSphereScreen? crystalSphereScreen,
        NProceedButton button)
    {
        if (crystalSphereScreen is not null &&
            TryInvokeSingleArgument(crystalSphereScreen, "OnProceedButtonPressed", button))
        {
            return;
        }

        InvokeProceedButtonAction(button);
    }

    private static void InvokeRoomProceedAction(BridgeWorldContext context)
    {
        if (context.ProceedButton is null)
        {
            throw new BridgeRequestException(
                HttpStatusCode.Conflict,
                "action_target_missing",
                "Could not find a visible room proceed button.");
        }

        if (context.CombatRoom is not null &&
            ReferenceEquals(context.ProceedButton, context.CombatRoom.ProceedButton))
        {
            InvokeCombatProceedAction(context.CombatRoom, context.ProceedButton);
            return;
        }

        if (context.TreasureRoom is not null &&
            IsNodeVisible(context.TreasureRoom))
        {
            InvokeTreasureProceedAction(context.TreasureRoom, context.ProceedButton);
            return;
        }

        InvokeProceedButtonAction(context.ProceedButton);
    }

    private static void InvokeCombatProceedAction(NCombatRoom? combatRoom, NProceedButton button)
    {
        if (TryInvokeSingleArgument(combatRoom, "OnProceedButtonPressed", button))
        {
            return;
        }

        InvokeProceedButtonAction(button);
    }

    private static void InvokeRestSiteProceedAction(NRestSiteRoom? restSiteRoom, NProceedButton button)
    {
        if (TryInvokeSingleArgument(restSiteRoom, "OnProceedButtonReleased", button))
        {
            return;
        }

        InvokeProceedButtonAction(button);
    }

    private static void InvokeMerchantLeaveAction(NMerchantRoom? merchantRoom, NProceedButton button)
    {
        if (TryInvokeSingleArgument(merchantRoom, "HideScreen", button))
        {
            return;
        }

        InvokeProceedButtonAction(button);
    }

    private static void InvokeMerchantBackAction(NMerchantInventory? merchantInventory, NBackButton button)
    {
        if (TryInvokeParameterless(merchantInventory, "Close"))
        {
            return;
        }

        InvokeButtonAction(button, "OnPress");
    }

    private static void InvokeTreasureChestAction(NTreasureRoom? treasureRoom, NTreasureButton chestButton)
    {
        if (TryInvokeSingleArgument(treasureRoom, "OnChestButtonReleased", chestButton))
        {
            return;
        }

        var openChestResult = InvokeParameterless(treasureRoom, "OpenChest");
        if (openChestResult is Task openChestTask)
        {
            openChestTask.GetAwaiter().GetResult();
            return;
        }

        InvokeButtonAction(chestButton, "OnRelease");
    }

    private static void InvokeTreasureRelicAction(
        NTreasureRoomRelicCollection? treasureRelicCollection,
        NTreasureRoomRelicHolder relicHolder)
    {
        if (TryInvokeSingleArgument(treasureRelicCollection, "PickRelic", relicHolder))
        {
            return;
        }

        if (TryInvokeParameterless(relicHolder, "OnRelease") ||
            TryInvokeParameterless(relicHolder, "OnPress"))
        {
            return;
        }

        InvokeClickablePressAndRelease(relicHolder);
    }

    private static void InvokeTreasureProceedAction(NTreasureRoom? treasureRoom, NProceedButton button)
    {
        if (TryInvokeSingleArgument(treasureRoom, "OnProceedButtonReleased", button) ||
            TryInvokeSingleArgument(treasureRoom, "OnProceedButtonPressed", button))
        {
            return;
        }

        InvokeProceedButtonAction(button);
    }

    private static void InvokeCardSelectionOptionAction(Node? cardSelectionScreen, NCardHolder cardHolder)
    {
        if (cardSelectionScreen is NPlayerHand playerHand)
        {
            if (TryInvokeSingleArgument(playerHand, "OnHolderPressed", cardHolder))
            {
                TryAutoConfirmSelectedCardSelection(cardSelectionScreen);
                return;
            }

            if (cardHolder is NHandCardHolder handCardHolder &&
                (TryInvokeSingleArgument(playerHand, "SelectCardInSimpleMode", handCardHolder) ||
                 TryInvokeSingleArgument(playerHand, "SelectCardInUpgradeMode", handCardHolder)))
            {
                TryAutoConfirmSelectedCardSelection(cardSelectionScreen);
                return;
            }
        }

        if (TryInvokeSingleArgument(cardSelectionScreen, "SelectHolder", cardHolder))
        {
            TryAutoConfirmSelectedCardSelection(cardSelectionScreen);
            return;
        }

        if (cardHolder.CardModel is not null &&
            TryInvokeSingleArgument(cardSelectionScreen, "OnCardClicked", cardHolder.CardModel))
        {
            TryAutoConfirmSelectedCardSelection(cardSelectionScreen);
            return;
        }

        throw new BridgeRequestException(
            HttpStatusCode.Conflict,
            "action_target_missing",
            $"Could not resolve a supported card-selection action for {cardSelectionScreen?.GetType().FullName ?? "<missing screen>"}.");
    }

    private static void InvokeCardSelectionBundleAction(Node? cardSelectionScreen, NCardBundle bundle)
    {
        if (TryInvokeSingleArgument(cardSelectionScreen, "OnBundleClicked", bundle))
        {
            return;
        }

        if (bundle.Hitbox is not null)
        {
            InvokeClickablePressAndRelease(bundle.Hitbox);
            return;
        }

        throw new BridgeRequestException(
            HttpStatusCode.Conflict,
            "action_target_missing",
            $"Could not resolve a supported bundle-selection action for {cardSelectionScreen?.GetType().FullName ?? "<missing screen>"}.");
    }

    private static bool ShouldAutoConfirmSingleCardSelection(Node? cardSelectionScreen)
    {
        var prefs = GetHiddenFieldValue(cardSelectionScreen, "_prefs");
        return (GetHiddenPropertyValue<int>(prefs, "MinSelect") ?? 0) == 1 &&
               (GetHiddenPropertyValue<int>(prefs, "MaxSelect") ?? 0) == 1;
    }

    private static void TryAutoConfirmSelectedCardSelection(Node? cardSelectionScreen)
    {
        if (cardSelectionScreen is NSimpleCardSelectScreen)
        {
            return;
        }

        if (!ShouldAutoConfirmSingleCardSelection(cardSelectionScreen) ||
            CountSelectedCardSelectionCards(cardSelectionScreen) <= 0)
        {
            return;
        }

        var confirmButton = ResolveCardSelectionConfirmButton(cardSelectionScreen);
        if (confirmButton is null)
        {
            return;
        }

        InvokeCardSelectionConfirmAction(cardSelectionScreen, confirmButton);
    }

    private static void InvokeCardSelectionConfirmAction(Node? cardSelectionScreen, Node? confirmButton)
    {
        if (cardSelectionScreen is NPlayerHand playerHand &&
            confirmButton is not null &&
            TryInvokeSingleArgument(playerHand, "OnSelectModeConfirmButtonPressed", confirmButton))
        {
            return;
        }

        if (TryInvokeCardSelectionCompleteSelection(cardSelectionScreen))
        {
            return;
        }

        if (confirmButton is not null &&
            TryInvokeSingleArgument(cardSelectionScreen, "ConfirmSelection", confirmButton))
        {
            return;
        }

        if (confirmButton is not null)
        {
            InvokeClickablePressAndRelease(confirmButton);
            return;
        }

        throw new BridgeRequestException(
            HttpStatusCode.Conflict,
            "action_target_missing",
            "Could not confirm the current card selection.");
    }

    private static void InvokeCardSelectionCancelAction(Node? cardSelectionScreen, Node? cancelButton)
    {
        if (cancelButton is not null &&
            TryInvokeSingleArgument(cardSelectionScreen, "CancelSelection", cancelButton))
        {
            return;
        }

        if (cancelButton is not null)
        {
            InvokeClickablePressAndRelease(cancelButton);
            return;
        }

        throw new BridgeRequestException(
            HttpStatusCode.Conflict,
            "action_target_missing",
            "Could not cancel the current card-selection preview.");
    }

    private static void InvokeCardSelectionCloseAction(Node? cardSelectionScreen, Node? closeButton)
    {
        if (closeButton is not null &&
            TryInvokeSingleArgument(cardSelectionScreen, "CloseSelection", closeButton))
        {
            return;
        }

        if (closeButton is not null)
        {
            InvokeClickablePressAndRelease(closeButton);
            return;
        }

        throw new BridgeRequestException(
            HttpStatusCode.Conflict,
            "action_target_missing",
            "Could not close the current card-selection screen.");
    }

    private static void InvokeCardSelectionSkipAction(Node? cardSelectionScreen, Node? skipButton)
    {
        if (skipButton is not null &&
            TryInvokeSingleArgument(cardSelectionScreen, "OnSkipButtonReleased", skipButton))
        {
            return;
        }

        if (skipButton is not null)
        {
            InvokeClickablePressAndRelease(skipButton);
            return;
        }

        throw new BridgeRequestException(
            HttpStatusCode.Conflict,
            "action_target_missing",
            "Could not skip the current card selection.");
    }

    private static Node? ResolveCardRewardSkipButton(NCardRewardSelectionScreen? cardRewardScreen)
    {
        const string alternativeButtonTypeName = "MegaCrit.Sts2.Core.Nodes.Screens.CardSelection.NCardRewardAlternativeButton";
        var alternativesRoot = GetHiddenFieldValue(cardRewardScreen, "_rewardAlternativesContainer") as Node
                               ?? cardRewardScreen;

        return FindVisibleDescendants(
                alternativesRoot,
                static node => IsTypeFullName(node, alternativeButtonTypeName))
            .FirstOrDefault(IsCardRewardSkipAlternativeButton);
    }

    private static void InvokeCardRewardSkipAction(NCardRewardSelectionScreen? cardRewardScreen, Node? skipButton)
    {
        if (IsCardRewardSelectionReady(cardRewardScreen) &&
            TryInvokeSingleArgument(
                cardRewardScreen,
                "OnAlternateRewardSelected",
                MegaCrit.Sts2.Core.Entities.Rewards.PostAlternateCardRewardAction.DismissScreenAndKeepReward))
        {
            return;
        }

        if (skipButton is not null)
        {
            InvokeClickablePressAndRelease(skipButton);
            return;
        }

        throw new BridgeRequestException(
            HttpStatusCode.Conflict,
            "action_target_missing",
            "Could not skip the current card reward.");
    }

    private static bool IsCardRewardSkipAlternativeButton(Node button)
    {
        static bool IsSkipText(string? text)
        {
            var comparableText = NormalizeComparableText(text).ToLowerInvariant();
            return comparableText == "skip" || comparableText == "跳过";
        }

        return IsSkipText(GetHiddenFieldValue(button, "_optionName") as string) ||
               IsSkipText(TryGetNodeText(button));
    }

    private static bool IsRewardButtonSkipped(NRewardsScreen? rewardsScreen, NRewardButton button)
    {
        if (GetHiddenFieldValue(rewardsScreen, "_skippedRewardButtons") is not IEnumerable skippedRewardButtons)
        {
            return false;
        }

        foreach (var skippedRewardButton in skippedRewardButtons)
        {
            if (skippedRewardButton is Node skippedNode &&
                IsSameNodeInstance(skippedNode, button))
            {
                return true;
            }
        }

        return false;
    }

    private static void InvokeTerminalRewardsProceed(
        RunManager? runManager,
        NRewardsScreen? rewardsScreen,
        NProceedButton? rewardProceedButton)
    {
        // Prefer the real rewards-screen handler so boss-room reward exits
        // follow the same path as a manual click.
        if (rewardProceedButton is not null &&
            TryInvokeSingleArgument(rewardsScreen, "OnProceedButtonPressed", rewardProceedButton))
        {
            return;
        }

        if (rewardProceedButton is not null && IsNodeVisible(rewardProceedButton))
        {
            InvokeProceedButtonAction(rewardProceedButton);
            return;
        }

        if (TryInvokeParameterless(runManager, "ProceedFromTerminalRewardsScreen") ||
            TryInvokeParameterless(rewardsScreen, "ProceedFromTerminalRewardsScreen"))
            return;

        throw new BridgeRequestException(
            HttpStatusCode.Conflict,
            "action_target_missing",
            "Could not find a terminal rewards proceed target.");
    }

    private static void InvokeRunModeSelectionAction(Node? submenu, Node? button, string methodName)
    {
        if (submenu is not null)
        {
            if (TryInvokeParameterless(submenu, methodName))
            {
                return;
            }

            if (button is not null && TryInvokeSingleArgument(submenu, methodName, button))
            {
                return;
            }
        }

        if (button is not null)
        {
            InvokeMenuButtonAction(button);
            return;
        }

        throw new BridgeRequestException(
            HttpStatusCode.Conflict,
            "action_target_missing",
            $"Could not invoke run-mode selection action '{methodName}'.");
    }

    private static void InvokeButtonAction(object target, string methodName, string? fallbackMethodName = null)
    {
        if (TryInvokeParameterless(target, methodName))
        {
            return;
        }

        if (fallbackMethodName is not null && TryInvokeParameterless(target, fallbackMethodName))
        {
            return;
        }

        throw new BridgeRequestException(
            HttpStatusCode.Conflict,
            "action_target_missing",
            $"Could not invoke {methodName} on {target.GetType().FullName}.");
    }

    private static void InvokeGameOverContinueAction(
        NGameOverScreen? gameOverScreen,
        NGameOverContinueButton? continueButton)
    {
        if (gameOverScreen is not null)
        {
            if (TryInvokeParameterless(gameOverScreen, "OpenTimeline") ||
                TryInvokeParameterless(gameOverScreen, "TransitionOutToTimeline"))
            {
                return;
            }
        }

        if (continueButton is not null)
        {
            InvokeClickablePressAndRelease(continueButton);
            return;
        }

        throw new BridgeRequestException(
            HttpStatusCode.Conflict,
            "action_target_missing",
            "Could not invoke a supported game-over continue action.");
    }

    private static void InvokeGameOverReturnToMainMenuAction(
        NGameOverScreen? gameOverScreen,
        NReturnToMainMenuButton? mainMenuButton)
    {
        if (gameOverScreen is not null)
        {
            if (TryInvokeParameterless(gameOverScreen, "ReturnToMainMenu") ||
                TryInvokeParameterless(gameOverScreen, "TransitionOutToMainMenu") ||
                (mainMenuButton is not null &&
                 TryInvokeSingleArgument(gameOverScreen, "OnMainMenuButtonPressed", mainMenuButton)))
            {
                return;
            }
        }

        if (mainMenuButton is not null)
        {
            InvokeClickablePressAndRelease(mainMenuButton);
            return;
        }

        throw new BridgeRequestException(
            HttpStatusCode.Conflict,
            "action_target_missing",
            "Could not invoke a supported game-over return-to-main-menu action.");
    }

    private static void InvokeSingleArgumentAction(object target, string methodName, object argument)
    {
        var method = FindMethod(target.GetType(), methodName, 1);
        if (method is null)
        {
            throw new BridgeRequestException(
                HttpStatusCode.Conflict,
                "action_target_missing",
                $"Could not find {methodName} on {target.GetType().FullName}.");
        }

        method.Invoke(target, new[] { argument });
    }

    private static object? InvokeParameterless(object? target, string methodName)
    {
        if (target is null)
        {
            return null;
        }

        var method = FindMethod(target.GetType(), methodName, 0);
        return method?.Invoke(target, Array.Empty<object>());
    }

    private static void ExecuteGameActionSynchronously(object action)
    {
        var result = InvokeParameterless(action, "ExecuteAction");
        if (result is Task task)
        {
            task.GetAwaiter().GetResult();
            return;
        }

        throw new BridgeRequestException(
            HttpStatusCode.Conflict,
            "action_execution_failed",
            $"Could not execute game action {action.GetType().FullName}.");
    }

    private static bool? TryInvokeBoolean(object? target, string methodName, params object?[] arguments)
    {
        if (target is null)
        {
            return null;
        }

        try
        {
            var method = FindMethod(target.GetType(), methodName, arguments.Length);
            if (method is null)
            {
                return null;
            }

            var result = method.Invoke(target, arguments);
            return result is bool boolResult ? boolResult : null;
        }
        catch
        {
            return null;
        }
    }

    private static bool TryInvokeParameterless(object? target, string methodName)
    {
        if (target is null)
        {
            return false;
        }

        var method = FindMethod(target.GetType(), methodName, 0);
        if (method is null)
        {
            return false;
        }

        method.Invoke(target, Array.Empty<object>());
        return true;
    }

    private static bool TryInvokeCardSelectionCompleteSelection(Node? cardSelectionScreen)
    {
        if (cardSelectionScreen is null)
        {
            return false;
        }

        var method = FindMethod(cardSelectionScreen.GetType(), "CompleteSelection", 0);
        if (method is null)
        {
            return false;
        }

        try
        {
            method.Invoke(cardSelectionScreen, Array.Empty<object>());
            return true;
        }
        catch (TargetInvocationException ex) when (IsBenignCardSelectionCompletionException(ex.InnerException))
        {
            return true;
        }
    }

    private static bool IsBenignCardSelectionCompletionException(Exception? exception)
    {
        return exception is InvalidOperationException invalidOperationException &&
               invalidOperationException.Message.Contains(
                   "transition a task to a final state",
                   StringComparison.OrdinalIgnoreCase);
    }

    private static bool TryInvokeSingleArgument(object? target, string methodName, object argument)
    {
        if (target is null)
        {
            return false;
        }

        var method = FindMethod(target.GetType(), methodName, 1);
        if (method is null)
        {
            return false;
        }

        method.Invoke(target, new[] { argument });
        return true;
    }

    private static MethodInfo? FindMethod(Type? type, string methodName, int parameterCount)
    {
        while (type is not null)
        {
            var method = type
                .GetMethods(BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.DeclaredOnly)
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

    private static T? GetHiddenPropertyValue<T>(object? target, string propertyName) where T : struct
    {
        if (target is null)
        {
            return null;
        }

        var property = FindProperty(target.GetType(), propertyName);
        if (property is null)
        {
            return null;
        }

        var value = property.GetValue(target);
        return value is T typed ? typed : null;
    }

    private static object? GetHiddenPropertyObjectValue(object? target, string propertyName)
    {
        if (target is null)
        {
            return null;
        }

        var property = FindProperty(target.GetType(), propertyName);
        return property?.GetValue(target);
    }

    private static object? GetHiddenFieldValue(object? target, string fieldName)
    {
        if (target is null)
        {
            return null;
        }

        var field = FindField(target.GetType(), fieldName);
        return field?.GetValue(target);
    }

    private static int CountSelectedDeckUpgradeCards(NDeckUpgradeSelectScreen? deckUpgradeScreen)
    {
        return GetSelectedDeckUpgradeCards(deckUpgradeScreen).Count;
    }

    private static bool IsDeckUpgradeCardSelected(NDeckUpgradeSelectScreen? deckUpgradeScreen, CardModel? card)
    {
        if (card is null)
        {
            return false;
        }

        return GetSelectedDeckUpgradeCards(deckUpgradeScreen).Any(selected => ReferenceEquals(selected, card));
    }

    private static List<object> GetSelectedDeckUpgradeCards(NDeckUpgradeSelectScreen? deckUpgradeScreen)
    {
        if (GetHiddenFieldValue(deckUpgradeScreen, "_selectedCards") is not IEnumerable selectedCards)
        {
            return new List<object>();
        }

        return selectedCards
            .Cast<object?>()
            .Where(static selected => selected is not null)
            .Cast<object>()
            .ToList();
    }

    private static bool IsDeckUpgradePreviewHolder(NDeckUpgradeSelectScreen deckUpgradeScreen, NCardHolder holder)
    {
        return IsDescendantOf(holder, GetHiddenFieldValue(deckUpgradeScreen, "_singlePreview") as Node) ||
               IsDescendantOf(holder, GetHiddenFieldValue(deckUpgradeScreen, "_multiPreview") as Node) ||
               IsDescendantOf(holder, GetHiddenFieldValue(deckUpgradeScreen, "_upgradeSinglePreviewContainer") as Node) ||
               IsDescendantOf(holder, GetHiddenFieldValue(deckUpgradeScreen, "_upgradeMultiPreviewContainer") as Node);
    }

    private static bool IsDescendantOf(Node? node, Node? ancestor)
    {
        if (node is null || ancestor is null)
        {
            return false;
        }

        var current = node.GetParent();
        while (current is not null)
        {
            if (ReferenceEquals(current, ancestor))
            {
                return true;
            }

            current = current.GetParent();
        }

        return false;
    }

    private static T? ResolveFirstVisibleNode<T>(params T?[] candidates) where T : Node
    {
        return candidates.FirstOrDefault(IsNodeVisible);
    }

    private static T? ResolveFirstVisibleEnabledNode<T>(params T?[] candidates) where T : Node
    {
        return candidates.FirstOrDefault(candidate => IsNodeVisible(candidate) && IsButtonEnabled(candidate));
    }

    private static bool IsCardSelectionPreviewVisible(Node? cardSelectionScreen)
    {
        if (cardSelectionScreen is null)
        {
            return false;
        }

        return ResolveFirstVisibleNode(
                   GetHiddenFieldValue(cardSelectionScreen, "_previewContainer") as Node,
                   GetHiddenFieldValue(cardSelectionScreen, "_enchantSinglePreviewContainer") as Node,
                   GetHiddenFieldValue(cardSelectionScreen, "_enchantMultiPreviewContainer") as Node) is not null;
    }

    private static Node? ResolveCardSelectionConfirmButton(Node? cardSelectionScreen)
    {
        if (cardSelectionScreen is null)
        {
            return null;
        }

        if (IsCardSelectionPreviewVisible(cardSelectionScreen))
        {
            var previewConfirmButton = ResolveFirstVisibleEnabledNode(
                GetHiddenFieldValue(cardSelectionScreen, "_previewConfirmButton") as Node,
                GetHiddenFieldValue(cardSelectionScreen, "_singlePreviewConfirmButton") as Node,
                GetHiddenFieldValue(cardSelectionScreen, "_multiPreviewConfirmButton") as Node);
            if (previewConfirmButton is not null)
            {
                return previewConfirmButton;
            }
        }

        // NConfirmButton.Disable() slides the button off-screen but can remain visible in-tree,
        // so prefer candidates that are both visible and enabled.
        return ResolveFirstVisibleEnabledNode(
            GetHiddenFieldValue(cardSelectionScreen, "_confirmButton") as Node,
            GetHiddenFieldValue(cardSelectionScreen, "_previewConfirmButton") as Node,
            GetHiddenFieldValue(cardSelectionScreen, "_singlePreviewConfirmButton") as Node,
            GetHiddenFieldValue(cardSelectionScreen, "_multiPreviewConfirmButton") as Node,
            GetHiddenFieldValue(cardSelectionScreen, "_selectModeConfirmButton") as Node);
    }

    private static Node? ResolveCardSelectionCancelButton(Node? cardSelectionScreen)
    {
        if (cardSelectionScreen is null)
        {
            return null;
        }

        if (IsCardSelectionPreviewVisible(cardSelectionScreen))
        {
            var previewCancelButton = ResolveFirstVisibleEnabledNode(
                GetHiddenFieldValue(cardSelectionScreen, "_previewCancelButton") as Node,
                GetHiddenFieldValue(cardSelectionScreen, "_singlePreviewCancelButton") as Node,
                GetHiddenFieldValue(cardSelectionScreen, "_multiPreviewCancelButton") as Node);
            if (previewCancelButton is not null)
            {
                return previewCancelButton;
            }
        }

        return ResolveFirstVisibleEnabledNode(
            GetHiddenFieldValue(cardSelectionScreen, "_previewCancelButton") as Node,
            GetHiddenFieldValue(cardSelectionScreen, "_singlePreviewCancelButton") as Node,
            GetHiddenFieldValue(cardSelectionScreen, "_multiPreviewCancelButton") as Node);
    }

    private static Node? ResolveCombatHandSelectionNode(NPlayerHand? playerHand)
    {
        if (playerHand is null || !IsNodeVisible(playerHand))
        {
            return null;
        }

        return playerHand.IsInCardSelection ||
               GetHiddenPropertyValue<bool>(playerHand, "IsInCardSelection") == true
            ? playerHand
            : null;
    }

    private static Node? ResolveVisibleCardSelectionScreen(Node? root, params Node?[] excludedScreens)
    {
        if (root is null)
        {
            return null;
        }

        var excludedInstances = excludedScreens
            .Where(screen => screen is not null && GodotObject.IsInstanceValid(screen))
            .Select(screen => screen!.NativeInstance)
            .ToHashSet();

        return FindVisibleDescendants(
                root,
                node =>
                {
                    if (excludedInstances.Contains(node.NativeInstance))
                    {
                        return false;
                    }

                    var fullName = node.GetType().FullName;
                    return fullName is not null &&
                           fullName.StartsWith("MegaCrit.Sts2.Core.Nodes.Screens.CardSelection.", StringComparison.Ordinal) &&
                           fullName.EndsWith("Screen", StringComparison.Ordinal);
                })
            .OrderByDescending(node => node is Control control ? control.Size.X * control.Size.Y : 0f)
            .FirstOrDefault();
    }

    private static IReadOnlyList<NCardHolder> GetCardSelectionOptions(Node? cardSelectionScreen)
    {
        if (cardSelectionScreen is null)
        {
            return Array.Empty<NCardHolder>();
        }

        if (IsCardSelectionPreviewVisible(cardSelectionScreen))
        {
            return Array.Empty<NCardHolder>();
        }

        if (cardSelectionScreen is NPlayerHand playerHand)
        {
            return FindVisibleDescendants<NCardHolder>(playerHand)
                    .Where(static holder =>
                        holder.CardModel is not null &&
                        holder is not NSelectedHandCardHolder)
                    .OrderBy(holder => TryGetCombatHandCardSelectionIndex(holder.CardModel) ?? int.MaxValue)
                    .ThenBy(static holder => holder is Control control ? control.GlobalPosition.Y : 0f)
                    .ThenBy(static holder => holder is Control control ? control.GlobalPosition.X : 0f)
                .DistinctBy(static holder => holder.CardModel, ReferenceEqualityComparer.Instance)
                .ToArray();
        }

        return SortByVisualPosition(
                FindVisibleDescendants<NCardHolder>(cardSelectionScreen)
                    .Where(static holder => holder.CardModel is not null))
            .DistinctBy(static holder => holder.CardModel, ReferenceEqualityComparer.Instance)
            .ToArray();
    }

    private static IReadOnlyList<NCardBundle> GetCardSelectionBundles(Node? cardSelectionScreen)
    {
        if (cardSelectionScreen is not NChooseABundleSelectionScreen)
        {
            return Array.Empty<NCardBundle>();
        }

        var bundleRow = GetHiddenFieldValue(cardSelectionScreen, "_bundleRow") as Node;
        var searchRoot = bundleRow is not null && GodotObject.IsInstanceValid(bundleRow)
            ? bundleRow
            : cardSelectionScreen;

        return SortByVisualPosition(
                FindVisibleDescendants<NCardBundle>(searchRoot)
                    .Where(static bundle => bundle.Bundle.Count > 0))
            .ToArray();
    }

    private static NProceedButton? ResolveVisibleProceedButton(
        Node? root,
        NProceedButton? preferredProceedButton,
        NProceedButton? treasureProceedButton,
        NProceedButton? restSiteProceedButton,
        NProceedButton? merchantProceedButton)
    {
        if (preferredProceedButton is not null &&
            IsNodeVisible(preferredProceedButton) &&
            IsButtonEnabled(preferredProceedButton))
        {
            return preferredProceedButton;
        }

        if (treasureProceedButton is not null &&
            IsNodeVisible(treasureProceedButton) &&
            IsButtonEnabled(treasureProceedButton))
        {
            return treasureProceedButton;
        }

        var excludedButtons = new HashSet<IntPtr>();
        if (treasureProceedButton is not null && GodotObject.IsInstanceValid(treasureProceedButton))
        {
            excludedButtons.Add(treasureProceedButton.NativeInstance);
        }

        if (restSiteProceedButton is not null && GodotObject.IsInstanceValid(restSiteProceedButton))
        {
            excludedButtons.Add(restSiteProceedButton.NativeInstance);
        }

        if (merchantProceedButton is not null && GodotObject.IsInstanceValid(merchantProceedButton))
        {
            excludedButtons.Add(merchantProceedButton.NativeInstance);
        }

        return SortByVisualPosition(
                FindVisibleDescendants<NProceedButton>(root)
                    .Where(button =>
                        !excludedButtons.Contains(button.NativeInstance) &&
                        IsButtonEnabled(button)))
            .LastOrDefault();
    }

    private static bool ShouldSuppressGenericRoomProceed(BridgeWorldContext context)
    {
        if (IsCardRewardSelectionVisible(context.CardRewardScreen, context.CardRewardOptions) ||
            IsRewardsScreenVisible(
                context.RewardsScreen,
                context.ProceedButton,
                context.RewardProceedButton,
                context.MapScreen,
                context.RewardButtons) ||
            IsCardSelectionVisible(context) ||
            IsDeckUpgradeSelectionVisible(context) ||
            (context.RestSiteRoom is not null && IsNodeVisible(context.RestSiteRoom)) ||
            (context.CrystalSphereScreen is not null && IsNodeVisible(context.CrystalSphereScreen)))
        {
            return true;
        }

        if (context.TreasureRoom is null || !IsNodeVisible(context.TreasureRoom))
        {
            return false;
        }

        if (CanOpenTreasureChest(context))
        {
            return true;
        }

        return context.TreasureRelicOptions.Any(IsNodeVisible);
    }

    private static bool CanOpenTreasureChest(BridgeWorldContext context)
    {
        if (context.TreasureRoom is null ||
            context.TreasureChestButton is null ||
            !IsNodeVisible(context.TreasureChestButton) ||
            !IsButtonEnabled(context.TreasureChestButton))
        {
            return false;
        }

        var hasRelicBeenClaimed = GetHiddenFieldValue(context.TreasureRoom, "_hasRelicBeenClaimed") is bool claimed && claimed;
        var isRelicCollectionOpen = GetHiddenFieldValue(context.TreasureRoom, "_isRelicCollectionOpen") is bool collectionOpen && collectionOpen;

        return !hasRelicBeenClaimed && !isRelicCollectionOpen;
    }

    private static string? TryGetCardSelectionPrompt(Node? cardSelectionScreen)
    {
        if (cardSelectionScreen is null)
        {
            return null;
        }

        var promptNode = ResolveFirstVisibleNode(
            GetHiddenFieldValue(cardSelectionScreen, "_selectionHeader") as Node,
            GetHiddenFieldValue(cardSelectionScreen, "_infoLabel") as Node,
            GetHiddenFieldValue(cardSelectionScreen, "_banner") as Node);
        var prompt = TryGetNodeText(promptNode);
        if (!string.IsNullOrWhiteSpace(prompt))
        {
            return prompt;
        }

        return CollectVisibleText(cardSelectionScreen, 1).FirstOrDefault();
    }

    private static string? TryGetDeckUpgradePrompt(NDeckUpgradeSelectScreen? deckUpgradeScreen)
    {
        if (deckUpgradeScreen is null)
        {
            return null;
        }

        var promptNode = ResolveFirstVisibleNode(
            GetHiddenFieldValue(deckUpgradeScreen, "_selectionHeader") as Node,
            GetHiddenFieldValue(deckUpgradeScreen, "_infoLabel") as Node,
            GetHiddenFieldValue(deckUpgradeScreen, "_banner") as Node,
            GetHiddenFieldValue(deckUpgradeScreen, "_singlePreviewTitleLabel") as Node,
            GetHiddenFieldValue(deckUpgradeScreen, "_multiPreviewTitleLabel") as Node);
        var prompt = TryGetNodeText(promptNode);
        if (!string.IsNullOrWhiteSpace(prompt))
        {
            return prompt;
        }

        return CollectVisibleText(deckUpgradeScreen, 1).FirstOrDefault();
    }

    private static int CountSelectedCardSelectionCards(Node? cardSelectionScreen)
    {
        if (cardSelectionScreen is null)
        {
            return 0;
        }

        if (GetHiddenFieldValue(cardSelectionScreen, "_selectedBundle") is not null)
        {
            return 1;
        }

        if (GetHiddenFieldValue(cardSelectionScreen, "_selectedCards") is IEnumerable selectedCards)
        {
            return selectedCards.Cast<object?>().Count(static card => card is not null);
        }

        return GetHiddenFieldValue(cardSelectionScreen, "_cardSelected") is true ? 1 : 0;
    }

    private static bool IsCardSelectionCardSelected(Node? cardSelectionScreen, CardModel? card)
    {
        if (cardSelectionScreen is null || card is null)
        {
            return false;
        }

        if (GetHiddenFieldValue(cardSelectionScreen, "_selectedCards") is IEnumerable selectedCards)
        {
            return selectedCards.Cast<object?>().Any(selected => ReferenceEquals(selected, card));
        }

        return false;
    }

    private static int GetCardSelectionOptionIndex(
        Node? cardSelectionScreen,
        NCardHolder cardHolder,
        int fallbackIndex)
    {
        if (cardSelectionScreen is NPlayerHand &&
            TryGetCombatHandCardSelectionIndex(cardHolder.CardModel) is int handIndex)
        {
            return handIndex;
        }

        return fallbackIndex;
    }

    private static string? GetCardSelectionOptionSelectionId(
        Node? cardSelectionScreen,
        NCardHolder cardHolder,
        int optionIndex)
    {
        if (cardHolder.CardModel is null)
        {
            return null;
        }

        return GetCardReference(cardHolder.CardModel);
    }

    private static string GetCardReference(CardModel card)
    {
        return $"card-{RuntimeHelpers.GetHashCode(card):x8}";
    }

    private static int? TryGetIntFromPropertyOrField(object? target, params string[] memberNames)
    {
        foreach (var memberName in memberNames)
        {
            var value = GetHiddenPropertyObjectValue(target, memberName) ?? GetHiddenFieldValue(target, memberName);
            if (TryConvertToInt(value) is int intValue)
            {
                return intValue;
            }
        }

        return null;
    }

    private static bool? TryGetBoolFromPropertyOrField(object? target, params string[] memberNames)
    {
        foreach (var memberName in memberNames)
        {
            var value = GetHiddenPropertyObjectValue(target, memberName) ?? GetHiddenFieldValue(target, memberName);
            if (value is bool boolValue)
            {
                return boolValue;
            }
        }

        return null;
    }

    private static int? TryConvertToInt(object? value)
    {
        try
        {
            return value switch
            {
                null => null,
                byte byteValue => byteValue,
                sbyte sbyteValue => sbyteValue,
                short shortValue => shortValue,
                ushort ushortValue => ushortValue,
                int intValue => intValue,
                uint uintValue when uintValue <= int.MaxValue => (int)uintValue,
                long longValue when longValue >= int.MinValue && longValue <= int.MaxValue => (int)longValue,
                ulong ulongValue when ulongValue <= int.MaxValue => (int)ulongValue,
                Enum enumValue => Convert.ToInt32(enumValue, CultureInfo.InvariantCulture),
                _ => null
            };
        }
        catch
        {
            return null;
        }
    }

    private static int? TryGetCombatHandCardSelectionIndex(CardModel? card)
    {
        if (card?.Owner is null)
        {
            return null;
        }

        try
        {
            var handPile = PileType.Hand.GetPile(card.Owner);
            var cards = handPile?.Cards;
            if (cards is null)
            {
                return null;
            }

            for (var index = 0; index < cards.Count; index++)
            {
                if (ReferenceEquals(cards[index], card))
                {
                    return index;
                }
            }

            return null;
        }
        catch
        {
            return null;
        }
    }

    private static PropertyInfo? FindProperty(Type? type, string propertyName)
    {
        while (type is not null)
        {
            var property = type.GetProperty(
                propertyName,
                BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.DeclaredOnly);

            if (property is not null)
            {
                return property;
            }

            type = type.BaseType;
        }

        return null;
    }

    private static FieldInfo? FindField(Type? type, string fieldName)
    {
        while (type is not null)
        {
            var field = type.GetField(
                fieldName,
                BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.DeclaredOnly);

            if (field is not null)
            {
                return field;
            }

            type = type.BaseType;
        }

        return null;
    }

    private static string DescribeCharacter(CharacterModel? character)
    {
        return DescribeCharacterText(character?.CharacterSelectTitle, character);
    }

    private static string DescribeCharacterDescription(CharacterModel? character)
    {
        return DescribeCharacterText(character?.CharacterSelectDesc, character);
    }

    private static string DescribeCharacterText(string? textKey, CharacterModel? character)
    {
        if (character is null || string.IsNullOrWhiteSpace(textKey))
        {
            return DescribeText(textKey, character);
        }

        return DescribeText(new LocString("characters", textKey), character);
    }

    private static string TryGetTitle(object model)
    {
        if (model is CharacterModel character)
        {
            return DescribeCharacter(character);
        }

        var title = TryGetNamedTextValue(model, "Title", "TitleLocString");
        return string.IsNullOrWhiteSpace(title)
            ? DescribeText(model, model)
            : title;
    }

    private static string TryGetDescription(object model)
    {
        if (model is CharacterModel character)
        {
            return DescribeCharacterDescription(character);
        }

        return DescribeText(TryGetPreferredDescriptionValue(model), model);
    }

    private static string TextOf(object? value)
    {
        return ReadTextValue(value, preferRawText: false, allowFormattedFallback: true);
    }

    private static string TextOfRawFirst(object? value, bool allowFormattedFallback = true)
    {
        return ReadTextValue(value, preferRawText: true, allowFormattedFallback);
    }

    private static string ReadTextValue(object? value, bool preferRawText, bool allowFormattedFallback)
    {
        if (value is string textValue)
        {
            return textValue;
        }

        if (value is LocString locString)
        {
            var locText = ReadLocStringText(locString, preferRawText, allowFormattedFallback);
            if (!string.IsNullOrWhiteSpace(locText))
            {
                return locText;
            }
        }

        if (value is not null)
        {
            var primaryMethodName = preferRawText ? "GetRawText" : "GetFormattedText";
            var fallbackMethodName = preferRawText ? "GetFormattedText" : "GetRawText";

            var primaryText = TryInvokeTextMethod(value, primaryMethodName);
            if (!string.IsNullOrWhiteSpace(primaryText))
            {
                return primaryText;
            }

            if (allowFormattedFallback || !preferRawText)
            {
                var fallbackText = TryInvokeTextMethod(value, fallbackMethodName);
                if (!string.IsNullOrWhiteSpace(fallbackText))
                {
                    return fallbackText;
                }
            }
        }

        var text = value?.ToString() ?? string.Empty;
        return value is not null && LooksLikeTypeName(text, value.GetType())
            ? string.Empty
            : text;
    }

    private static string ReadLocStringText(
        LocString locString,
        bool preferRawText,
        bool allowFormattedFallback)
    {
        if (preferRawText)
        {
            var rawText = TryGetLocStringRawText(locString);
            if (!string.IsNullOrWhiteSpace(rawText))
            {
                return rawText;
            }

            if (allowFormattedFallback)
            {
                var formattedText = TryGetLocStringFormattedText(locString);
                if (!string.IsNullOrWhiteSpace(formattedText))
                {
                    return formattedText;
                }
            }

            return string.Empty;
        }

        var formatted = TryGetLocStringFormattedText(locString);
        if (!string.IsNullOrWhiteSpace(formatted))
        {
            return formatted;
        }

        return TryGetLocStringRawText(locString);
    }

    private static string TryGetLocStringFormattedText(LocString locString)
    {
        try
        {
            return locString.GetFormattedText();
        }
        catch
        {
            return string.Empty;
        }
    }

    private static string TryGetLocStringRawText(LocString locString)
    {
        try
        {
            return locString.GetRawText();
        }
        catch
        {
            return string.Empty;
        }
    }

    private static string DescribeText(object? value, object? placeholderContext = null)
    {
        var text = TextOfRawFirst(value, allowFormattedFallback: false);
        if (string.IsNullOrWhiteSpace(text))
        {
            text = TextOfRawFirst(value);
        }

        if (string.IsNullOrWhiteSpace(text))
        {
            return text;
        }

        object? effectivePlaceholderContext = placeholderContext;
        if (value is not null)
        {
            effectivePlaceholderContext = placeholderContext is null || ReferenceEquals(value, placeholderContext)
                ? value
                : new object?[] { value, placeholderContext };
        }

        return NormalizePayloadText(ResolvePlaceholderText(text, effectivePlaceholderContext));
    }

    private static string NormalizePayloadText(string text)
    {
        if (string.IsNullOrWhiteSpace(text))
        {
            return string.Empty;
        }

        var normalized = ReplaceImageTags(text.ReplaceLineEndings("\n")).Trim();
        if (normalized.Length == 0)
        {
            return string.Empty;
        }

        normalized = StripBbCode(normalized);
        normalized = normalized.Replace("[", string.Empty).Replace("]", string.Empty);
        normalized = normalized.Replace(" \n", "\n").Replace("\n ", "\n");

        var builder = new StringBuilder(normalized.Length);
        var previousWasWhitespace = false;

        foreach (var character in normalized)
        {
            if (character == '\n')
            {
                if (builder.Length > 0 && builder[^1] == ' ')
                {
                    builder.Length--;
                }

                if (builder.Length == 0 || builder[^1] != '\n')
                {
                    builder.Append('\n');
                }

                previousWasWhitespace = false;
                continue;
            }

            if (char.IsWhiteSpace(character))
            {
                if (!previousWasWhitespace)
                {
                    builder.Append(' ');
                    previousWasWhitespace = true;
                }

                continue;
            }

            builder.Append(character);
            previousWasWhitespace = false;
        }

        return builder.ToString().Trim();
    }

    private const string ImageTagMarkerPrefix = "<<sts2-icon:";
    private const string ImageTagMarkerSuffix = ">>";

    private static string ReplaceImageTags(string text)
    {
        if (string.IsNullOrWhiteSpace(text))
        {
            return string.Empty;
        }

        const string openTag = "[img]";
        const string closeTag = "[/img]";
        var builder = new StringBuilder(text.Length);
        var cursor = 0;

        while (cursor < text.Length)
        {
            var openIndex = text.IndexOf(openTag, cursor, StringComparison.OrdinalIgnoreCase);
            if (openIndex < 0)
            {
                builder.Append(text, cursor, text.Length - cursor);
                break;
            }

            builder.Append(text, cursor, openIndex - cursor);

            var contentStart = openIndex + openTag.Length;
            var closeIndex = text.IndexOf(closeTag, contentStart, StringComparison.OrdinalIgnoreCase);
            if (closeIndex < 0)
            {
                builder.Append(text, openIndex, text.Length - openIndex);
                break;
            }

            var inner = text.Substring(contentStart, closeIndex - contentStart);
            builder.Append(CreateImageTagMarker(inner));
            cursor = closeIndex + closeTag.Length;
        }

        return CollapseImageTagMarkers(builder.ToString());
    }

    private static string CreateImageTagMarker(string inner)
    {
        if (TryRecognizeImageTagKind(inner, out var kind))
        {
            return $"{ImageTagMarkerPrefix}{kind}{ImageTagMarkerSuffix}";
        }

        var debugName = GetImageTagDebugName(inner);
        return $"{ImageTagMarkerPrefix}unknown:{debugName}{ImageTagMarkerSuffix}";
    }

    private static string CollapseImageTagMarkers(string text)
    {
        if (string.IsNullOrWhiteSpace(text) ||
            text.IndexOf(ImageTagMarkerPrefix, StringComparison.Ordinal) < 0)
        {
            return text;
        }

        var collapsed = CollapseKnownImageTagMarkers(text, "energy", "点能量", allowCountPrefix: true);
        collapsed = CollapseKnownImageTagMarkers(collapsed, "star", "点星辉", allowCountPrefix: true);

        var unknownPattern =
            $"{Regex.Escape(ImageTagMarkerPrefix)}(?<token>[^>]+){Regex.Escape(ImageTagMarkerSuffix)}";
        return Regex.Replace(
            collapsed,
            unknownPattern,
            match =>
            {
                var token = match.Groups["token"].Value;
                return token.StartsWith("unknown:", StringComparison.OrdinalIgnoreCase)
                    ? $"图标:{token["unknown:".Length..]}"
                    : $"图标:{token}";
            },
            RegexOptions.IgnoreCase | RegexOptions.CultureInvariant);
    }

    private static string CollapseKnownImageTagMarkers(
        string text,
        string kind,
        string unitLabel,
        bool allowCountPrefix = false)
    {
        if (string.IsNullOrWhiteSpace(text))
        {
            return string.Empty;
        }

        var marker = $"{ImageTagMarkerPrefix}{kind}{ImageTagMarkerSuffix}";
        var options = RegexOptions.IgnoreCase | RegexOptions.CultureInvariant;

        if (allowCountPrefix)
        {
            text = Regex.Replace(
                text,
                $@"(?<count>\d+)\s*{Regex.Escape(marker)}",
                match => $"{match.Groups["count"].Value}{unitLabel}",
                options);
        }

        var repeatedPattern = $@"(?:{Regex.Escape(marker)}\s*)+";
        return Regex.Replace(
            text,
            repeatedPattern,
            match =>
            {
                var count = Regex.Matches(match.Value, Regex.Escape(marker), options).Count;
                return count > 0 ? $"{count}{unitLabel}" : match.Value;
            },
            options);
    }

    private static bool TryRecognizeImageTagKind(string inner, out string kind)
    {
        kind = string.Empty;
        var debugName = GetImageTagDebugName(inner);
        if (debugName.Length == 0)
        {
            return false;
        }

        if (string.Equals(debugName, "star_icon", StringComparison.OrdinalIgnoreCase))
        {
            kind = "star";
            return true;
        }

        if (debugName.EndsWith("_energy_icon", StringComparison.OrdinalIgnoreCase))
        {
            kind = "energy";
            return true;
        }

        return false;
    }

    private static string GetImageTagDebugName(string inner)
    {
        if (string.IsNullOrWhiteSpace(inner))
        {
            return "empty";
        }

        var normalized = inner.Trim();
        var slashIndex = normalized.LastIndexOfAny(new[] { '/', '\\' });
        if (slashIndex >= 0 && slashIndex + 1 < normalized.Length)
        {
            normalized = normalized[(slashIndex + 1)..];
        }

        var dotIndex = normalized.LastIndexOf('.');
        if (dotIndex > 0)
        {
            normalized = normalized[..dotIndex];
        }

        var builder = new StringBuilder(normalized.Length);
        foreach (var character in normalized)
        {
            if (char.IsLetterOrDigit(character) || character is '_' or '-' or ':')
            {
                builder.Append(char.ToLowerInvariant(character));
            }
        }

        return builder.Length > 0 ? builder.ToString() : "unknown";
    }

    private static string ResolvePlaceholderText(string text, object? placeholderContext)
    {
        if (string.IsNullOrWhiteSpace(text) ||
            placeholderContext is null ||
            !text.Contains('{'))
        {
            return text;
        }

        var builder = new StringBuilder(text.Length);
        var cursor = 0;

        while (cursor < text.Length)
        {
            var openBrace = text.IndexOf('{', cursor);
            if (openBrace < 0)
            {
                builder.Append(text, cursor, text.Length - cursor);
                break;
            }

            var closeBrace = FindPlaceholderCloseBrace(text, openBrace);
            if (closeBrace < 0)
            {
                builder.Append(text, cursor, text.Length - cursor);
                break;
            }

            builder.Append(text, cursor, openBrace - cursor);

            var placeholderBody = text.Substring(openBrace + 1, closeBrace - openBrace - 1);
            if (TryResolvePlaceholderText(placeholderContext, placeholderBody, out var resolvedPlaceholder))
            {
                builder.Append(resolvedPlaceholder);
            }
            else
            {
                builder.Append(text, openBrace, closeBrace - openBrace + 1);
            }

            cursor = closeBrace + 1;
        }

        return builder.ToString();
    }

    private static int FindPlaceholderCloseBrace(string text, int openBraceIndex)
    {
        if (string.IsNullOrEmpty(text) ||
            openBraceIndex < 0 ||
            openBraceIndex >= text.Length ||
            text[openBraceIndex] != '{')
        {
            return -1;
        }

        var depth = 0;
        for (var index = openBraceIndex; index < text.Length; index++)
        {
            switch (text[index])
            {
                case '{':
                    depth++;
                    break;
                case '}':
                    depth--;
                    if (depth == 0)
                    {
                        return index;
                    }

                    break;
            }
        }

        return -1;
    }

    private static bool TryResolvePlaceholderText(
        object placeholderContext,
        string placeholderBody,
        out string resolvedText)
    {
        resolvedText = string.Empty;

        if (string.IsNullOrWhiteSpace(placeholderBody))
        {
            return false;
        }

        var separatorIndex = placeholderBody.IndexOf(':');
        var tokenName = separatorIndex >= 0
            ? placeholderBody[..separatorIndex].Trim()
            : placeholderBody.Trim();
        var formatHint = separatorIndex >= 0
            ? placeholderBody[(separatorIndex + 1)..].Trim()
            : string.Empty;

        if (string.IsNullOrWhiteSpace(tokenName))
        {
            return false;
        }

        if (TryResolveStandalonePlaceholderToken(tokenName, formatHint, out resolvedText))
        {
            resolvedText = ResolvePlaceholderText(resolvedText, placeholderContext);
            return !string.IsNullOrWhiteSpace(resolvedText);
        }

        if (!TryResolvePlaceholderValue(placeholderContext, tokenName, out var resolvedValue))
        {
            return false;
        }

        resolvedText = FormatResolvedPlaceholderValue(tokenName, formatHint, resolvedValue);
        resolvedText = ResolvePlaceholderText(resolvedText, placeholderContext);
        return !string.IsNullOrWhiteSpace(resolvedText);
    }

    private static bool TryResolvePlaceholderValue(
        object placeholderContext,
        string tokenName,
        out object? resolvedValue)
    {
        resolvedValue = null;

        foreach (var candidate in EnumeratePlaceholderContexts(placeholderContext))
        {
            if (candidate is null)
            {
                continue;
            }

            if (TryResolvePlaceholderValueFromLocStringVariables(candidate, tokenName, out resolvedValue) ||
                TryResolvePlaceholderValueFromTypeHierarchy(candidate, tokenName, out resolvedValue) ||
                TryResolvePlaceholderValueFromCanonicalVars(candidate, tokenName, out resolvedValue) ||
                TryResolvePlaceholderValueFromDynamicVars(candidate, tokenName, out resolvedValue))
            {
                return true;
            }
        }

        return false;
    }

    private static bool TryResolvePlaceholderValueFromLocStringVariables(
        object candidate,
        string tokenName,
        out object? resolvedValue)
    {
        resolvedValue = null;

        if (candidate is not LocString locString)
        {
            return false;
        }

        foreach (var entry in locString.Variables)
        {
            if (!string.Equals(entry.Key, tokenName, StringComparison.OrdinalIgnoreCase))
            {
                continue;
            }

            resolvedValue = entry.Value is DynamicVar dynamicVar
                ? GetPreferredDynamicVarValue(dynamicVar)
                : entry.Value;
            return resolvedValue is not null;
        }

        return false;
    }

    private static bool TryResolveStandalonePlaceholderToken(
        string tokenName,
        string formatHint,
        out string resolvedText)
    {
        resolvedText = string.Empty;

        if (TryResolveIconPlaceholderToken(tokenName, formatHint, out resolvedText))
        {
            return true;
        }

        return false;
    }

    private static bool TryResolveIconPlaceholderToken(
        string tokenName,
        string formatHint,
        out string resolvedText)
    {
        resolvedText = string.Empty;

        if (string.Equals(tokenName, "singleStarIcon", StringComparison.OrdinalIgnoreCase))
        {
            resolvedText = "点星辉";
            return true;
        }

        if (string.Equals(tokenName, "singleEnergyIcon", StringComparison.OrdinalIgnoreCase))
        {
            resolvedText = "点能量";
            return true;
        }

        if (formatHint.Contains("energyIcons(", StringComparison.OrdinalIgnoreCase))
        {
            resolvedText = "能量";
            return true;
        }

        if (formatHint.Contains("starIcons(", StringComparison.OrdinalIgnoreCase))
        {
            resolvedText = "星辉";
            return true;
        }

        return false;
    }

    private static IEnumerable<object?> EnumeratePlaceholderContexts(object placeholderContext)
    {
        if (placeholderContext is IEnumerable enumerable &&
            placeholderContext is not string &&
            placeholderContext is not LocString)
        {
            foreach (var item in enumerable)
            {
                if (item is null)
                {
                    continue;
                }

                foreach (var nestedContext in EnumeratePlaceholderContexts(item))
                {
                    yield return nestedContext;
                }
            }

            yield break;
        }

        yield return placeholderContext;

        if (GetHiddenPropertyObjectValue(placeholderContext, "CanonicalModel") is { } canonicalModel)
        {
            foreach (var nestedContext in EnumeratePlaceholderContexts(canonicalModel))
            {
                yield return nestedContext;
            }
        }

        if (GetHiddenPropertyObjectValue(placeholderContext, "Model") is { } model)
        {
            foreach (var nestedContext in EnumeratePlaceholderContexts(model))
            {
                yield return nestedContext;
            }
        }

        if (GetHiddenPropertyObjectValue(placeholderContext, "Info") is { } info)
        {
            foreach (var nestedContext in EnumeratePlaceholderContexts(info))
            {
                yield return nestedContext;
            }
        }
    }

    private static object? TryGetPreferredDescriptionValue(object model)
    {
        var preferredPropertyNames = new[]
        {
            "DynamicDescription",
            "RemoteDescription",
            "DynamicEventDescription",
            "EventDescription",
            "StaticDescription",
            "DescriptionLocString",
            "Description"
        };

        foreach (var propertyName in preferredPropertyNames)
        {
            var property = FindProperty(model.GetType(), propertyName);
            if (property is null)
            {
                continue;
            }

            object? value;
            try
            {
                value = property.GetValue(model);
            }
            catch
            {
                continue;
            }

            if (HasMeaningfulDescriptionText(value))
            {
                return value;
            }
        }

        return null;
    }

    private static string TryGetNamedTextValue(object model, params string[] memberNames)
    {
        foreach (var memberName in memberNames)
        {
            var property = FindProperty(model.GetType(), memberName);
            if (property is null)
            {
                continue;
            }

            object? value;
            try
            {
                value = property.GetValue(model);
            }
            catch
            {
                continue;
            }

            var text = DescribeText(value, model);
            if (!string.IsNullOrWhiteSpace(text))
            {
                return text;
            }
        }

        return string.Empty;
    }

    private static bool HasMeaningfulDescriptionText(object? value)
    {
        if (value is null)
        {
            return false;
        }

        var rawText = TextOfRawFirst(value, allowFormattedFallback: false);
        if (!string.IsNullOrWhiteSpace(rawText))
        {
            return true;
        }

        var text = value.ToString() ?? string.Empty;
        return !string.IsNullOrWhiteSpace(text) &&
               !LooksLikeTypeName(text, value.GetType());
    }

    private static string TryInvokeTextMethod(object value, string methodName)
    {
        var method = FindMethod(value.GetType(), methodName, 0);
        if (method is null)
        {
            return string.Empty;
        }

        try
        {
            return TextOf(method.Invoke(value, Array.Empty<object>()));
        }
        catch
        {
            return string.Empty;
        }
    }

    private static bool LooksLikeTypeName(string text, Type type)
    {
        if (string.IsNullOrWhiteSpace(text))
        {
            return false;
        }

        return string.Equals(text, type.FullName, StringComparison.Ordinal) ||
               string.Equals(text, type.Name, StringComparison.Ordinal);
    }

    private static bool TryResolvePlaceholderValueFromTypeHierarchy(
        object candidate,
        string tokenName,
        out object? resolvedValue)
    {
        resolvedValue = null;
        var segments = tokenName
            .Split('.', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);

        if (segments.Length > 1)
        {
            return TryResolvePlaceholderValueFromMemberPath(candidate, segments, out resolvedValue);
        }

        return TryResolvePlaceholderMember(candidate, tokenName, out resolvedValue);
    }

    private static bool TryResolvePlaceholderValueFromMemberPath(
        object candidate,
        IReadOnlyList<string> segments,
        out object? resolvedValue)
    {
        resolvedValue = null;
        object? currentValue = candidate;

        foreach (var segment in segments)
        {
            if (currentValue is null ||
                !TryResolvePlaceholderMember(currentValue, segment, out currentValue))
            {
                resolvedValue = null;
                return false;
            }
        }

        resolvedValue = currentValue;
        return resolvedValue is not null;
    }

    private static bool TryResolvePlaceholderMember(
        object candidate,
        string memberName,
        out object? resolvedValue)
    {
        resolvedValue = null;
        var type = candidate.GetType();
        var normalizedMemberName = NormalizePlaceholderMemberName(memberName);

        while (type is not null)
        {
            foreach (var property in type.GetProperties(BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.DeclaredOnly))
            {
                if (!string.Equals(
                        NormalizePlaceholderMemberName(property.Name),
                        normalizedMemberName,
                        StringComparison.OrdinalIgnoreCase))
                {
                    continue;
                }

                try
                {
                    resolvedValue = property.GetValue(candidate);
                    return resolvedValue is not null;
                }
                catch
                {
                }
            }

            foreach (var field in type.GetFields(BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.DeclaredOnly))
            {
                if (!string.Equals(
                        NormalizePlaceholderMemberName(field.Name),
                        normalizedMemberName,
                        StringComparison.OrdinalIgnoreCase))
                {
                    continue;
                }

                try
                {
                    resolvedValue = field.GetValue(candidate);
                    return resolvedValue is not null;
                }
                catch
                {
                }
            }

            type = type.BaseType;
        }

        return false;
    }

    private static bool TryResolvePlaceholderValueFromCanonicalVars(
        object candidate,
        string tokenName,
        out object? resolvedValue)
    {
        resolvedValue = null;

        if (FindProperty(candidate.GetType(), "CanonicalVars")?.GetValue(candidate) is not IEnumerable canonicalVars)
        {
            return false;
        }

        foreach (var dynamicVar in canonicalVars)
        {
            if (!TryGetDynamicVarName(dynamicVar, out var dynamicVarName) ||
                !string.Equals(dynamicVarName, tokenName, StringComparison.OrdinalIgnoreCase))
            {
                continue;
            }

            resolvedValue = GetPreferredDynamicVarValue(dynamicVar);
            return resolvedValue is not null;
        }

        return false;
    }

    private static bool TryResolvePlaceholderValueFromDynamicVars(
        object candidate,
        string tokenName,
        out object? resolvedValue)
    {
        resolvedValue = null;

        var dynamicVars = FindProperty(candidate.GetType(), "DynamicVars")?.GetValue(candidate);
        if (dynamicVars is null)
        {
            return false;
        }

        var tryGetValueMethod = FindMethod(dynamicVars.GetType(), "TryGetValue", 2);
        if (tryGetValueMethod is null)
        {
            return false;
        }

        var parameters = new object?[] { tokenName, null };

        try
        {
            if (tryGetValueMethod.Invoke(dynamicVars, parameters) is true &&
                parameters[1] is { } dynamicVar)
            {
                resolvedValue = GetPreferredDynamicVarValue(dynamicVar);
                return resolvedValue is not null;
            }
        }
        catch
        {
        }

        return false;
    }

    private static bool TryGetDynamicVarName(object? dynamicVar, out string name)
    {
        name = string.Empty;

        if (dynamicVar is null)
        {
            return false;
        }

        var property = FindProperty(dynamicVar.GetType(), "Name");
        name = TextOf(property?.GetValue(dynamicVar));
        return !string.IsNullOrWhiteSpace(name);
    }

    private static object? GetPreferredDynamicVarValue(object dynamicVar)
    {
        if (dynamicVar is null)
        {
            return null;
        }

        var previewValue = GetHiddenPropertyValue<decimal>(dynamicVar, "PreviewValue");
        if (previewValue.HasValue && previewValue.Value != 0m)
        {
            return decimal.Truncate(previewValue.Value) == previewValue.Value
                ? (int)previewValue.Value
                : previewValue.Value;
        }

        var intValue = GetHiddenPropertyValue<int>(dynamicVar, "IntValue");
        if (intValue.HasValue)
        {
            return intValue.Value;
        }

        var baseValue = GetHiddenPropertyValue<decimal>(dynamicVar, "BaseValue");
        if (baseValue.HasValue)
        {
            return decimal.Truncate(baseValue.Value) == baseValue.Value
                ? (int)baseValue.Value
                : baseValue.Value;
        }

        return dynamicVar;
    }

    private static string NormalizePlaceholderMemberName(string? memberName)
    {
        if (string.IsNullOrWhiteSpace(memberName))
        {
            return string.Empty;
        }

        var normalized = memberName.Trim();
        const string backingFieldSuffix = ">k__BackingField";

        if (normalized.StartsWith('<') &&
            normalized.EndsWith(backingFieldSuffix, StringComparison.Ordinal))
        {
            normalized = normalized.Substring(1, normalized.Length - backingFieldSuffix.Length - 1);
        }

        return normalized.TrimStart('_');
    }

    private static string FormatResolvedPlaceholderValue(
        string tokenName,
        string formatHint,
        object? resolvedValue)
    {
        if (resolvedValue is null)
        {
            return string.Empty;
        }

        if (TryResolveConditionalPlaceholderText(formatHint, resolvedValue, out var conditionalText))
        {
            return conditionalText;
        }

        if (!string.IsNullOrWhiteSpace(formatHint))
        {
            if (formatHint.Contains("abs()", StringComparison.OrdinalIgnoreCase) &&
                TryConvertToDecimal(resolvedValue, out var absoluteValue))
            {
                return FormatNumericValue(decimal.Abs(absoluteValue));
            }

            if (formatHint.Contains("percentMore()", StringComparison.OrdinalIgnoreCase) &&
                TryConvertToDecimal(resolvedValue, out var percentValue))
            {
                var normalizedPercent = percentValue is >= -1m and <= 1m
                    ? percentValue * 100m
                    : percentValue;
                return FormatNumericValue(normalizedPercent);
            }

            if (formatHint.Contains("energyIcons(", StringComparison.OrdinalIgnoreCase))
            {
                return TryConvertToInt(resolvedValue, out var energyAmount)
                    ? $"{energyAmount}点能量"
                    : "能量";
            }

            if (formatHint.Contains("starIcons(", StringComparison.OrdinalIgnoreCase))
            {
                return TryConvertToInt(resolvedValue, out var starAmount)
                    ? $"{starAmount}点星辉"
                    : "星辉";
            }
        }

        if (string.Equals(tokenName, "singleStarIcon", StringComparison.OrdinalIgnoreCase))
        {
            return "点星辉";
        }

        if (string.Equals(tokenName, "singleEnergyIcon", StringComparison.OrdinalIgnoreCase))
        {
            return "点能量";
        }

        return FormatPlainPlaceholderValue(resolvedValue);
    }

    private static bool TryResolveConditionalPlaceholderText(
        string formatHint,
        object resolvedValue,
        out string resolvedText)
    {
        resolvedText = string.Empty;

        if (string.IsNullOrWhiteSpace(formatHint) ||
            !formatHint.StartsWith("cond:", StringComparison.OrdinalIgnoreCase))
        {
            return false;
        }

        var expression = formatHint["cond:".Length..];
        if (expression.Length == 0)
        {
            return false;
        }

        var segments = expression.Split('|');
        var fallbackSegments = new List<string>();

        foreach (var segment in segments)
        {
            var questionMarkIndex = segment.IndexOf('?');
            if (questionMarkIndex <= 0)
            {
                fallbackSegments.Add(segment);
                continue;
            }

            var condition = segment[..questionMarkIndex].Trim();
            var output = segment[(questionMarkIndex + 1)..];
            if (!EvaluatePlaceholderCondition(condition, resolvedValue))
            {
                continue;
            }

            resolvedText = ReplaceConditionalTemplateValue(output, resolvedValue);
            return true;
        }

        if (fallbackSegments.Count <= 0)
        {
            return false;
        }

        var selectedFallback = fallbackSegments.Count == 1
            ? fallbackSegments[0]
            : (IsTruthyPlaceholderValue(resolvedValue) ? fallbackSegments[0] : fallbackSegments[^1]);
        resolvedText = ReplaceConditionalTemplateValue(selectedFallback, resolvedValue);
        return true;
    }

    private static bool EvaluatePlaceholderCondition(string condition, object resolvedValue)
    {
        if (string.IsNullOrWhiteSpace(condition))
        {
            return IsTruthyPlaceholderValue(resolvedValue);
        }

        var trimmedCondition = condition.Trim();
        if (trimmedCondition.StartsWith(">=", StringComparison.Ordinal) &&
            TryConvertToDecimal(resolvedValue, out var greaterOrEqualValue) &&
            decimal.TryParse(trimmedCondition[2..], out var greaterOrEqualTarget))
        {
            return greaterOrEqualValue >= greaterOrEqualTarget;
        }

        if (trimmedCondition.StartsWith("<=", StringComparison.Ordinal) &&
            TryConvertToDecimal(resolvedValue, out var lessOrEqualValue) &&
            decimal.TryParse(trimmedCondition[2..], out var lessOrEqualTarget))
        {
            return lessOrEqualValue <= lessOrEqualTarget;
        }

        if (trimmedCondition.StartsWith("==", StringComparison.Ordinal) &&
            TryConvertToDecimal(resolvedValue, out var equalValue) &&
            decimal.TryParse(trimmedCondition[2..], out var equalTarget))
        {
            return equalValue == equalTarget;
        }

        if (trimmedCondition.StartsWith("!=", StringComparison.Ordinal) &&
            TryConvertToDecimal(resolvedValue, out var notEqualValue) &&
            decimal.TryParse(trimmedCondition[2..], out var notEqualTarget))
        {
            return notEqualValue != notEqualTarget;
        }

        if (trimmedCondition.StartsWith(">", StringComparison.Ordinal) &&
            TryConvertToDecimal(resolvedValue, out var greaterValue) &&
            decimal.TryParse(trimmedCondition[1..], out var greaterTarget))
        {
            return greaterValue > greaterTarget;
        }

        if (trimmedCondition.StartsWith("<", StringComparison.Ordinal) &&
            TryConvertToDecimal(resolvedValue, out var lessValue) &&
            decimal.TryParse(trimmedCondition[1..], out var lessTarget))
        {
            return lessValue < lessTarget;
        }

        return string.Equals(
            NormalizeComparableText(FormatPlainPlaceholderValue(resolvedValue)),
            NormalizeComparableText(trimmedCondition),
            StringComparison.OrdinalIgnoreCase);
    }

    private static bool IsTruthyPlaceholderValue(object? value)
    {
        if (value is null)
        {
            return false;
        }

        return value switch
        {
            bool boolValue => boolValue,
            string stringValue => !string.IsNullOrWhiteSpace(stringValue),
            _ when TryConvertToDecimal(value, out var numericValue) => numericValue != 0m,
            _ => true
        };
    }

    private static string ReplaceConditionalTemplateValue(string template, object resolvedValue)
    {
        if (string.IsNullOrEmpty(template))
        {
            return string.Empty;
        }

        return template.Replace("{}", FormatPlainPlaceholderValue(resolvedValue), StringComparison.Ordinal);
    }

    private static string FormatPlainPlaceholderValue(object resolvedValue)
    {
        if (TryConvertToDecimal(resolvedValue, out var numericValue))
        {
            return FormatNumericValue(numericValue);
        }

        var rawText = TextOfRawFirst(resolvedValue, allowFormattedFallback: false);
        if (!string.IsNullOrWhiteSpace(rawText))
        {
            return rawText;
        }

        return TextOf(resolvedValue);
    }

    private static string FormatNumericValue(decimal number)
    {
        var normalized = decimal.Truncate(number) == number
            ? decimal.Truncate(number)
            : number;
        return normalized.ToString(CultureInfo.InvariantCulture);
    }

    private static bool TryConvertToDecimal(object value, out decimal number)
    {
        switch (value)
        {
            case byte byteValue:
                number = byteValue;
                return true;
            case sbyte sbyteValue:
                number = sbyteValue;
                return true;
            case short shortValue:
                number = shortValue;
                return true;
            case ushort ushortValue:
                number = ushortValue;
                return true;
            case int intValue:
                number = intValue;
                return true;
            case uint uintValue:
                number = uintValue;
                return true;
            case long longValue:
                number = longValue;
                return true;
            case ulong ulongValue:
                number = ulongValue;
                return true;
            case decimal decimalValue:
                number = decimalValue;
                return true;
            case float floatValue:
                number = (decimal)floatValue;
                return true;
            case double doubleValue:
                number = (decimal)doubleValue;
                return true;
            case string stringValue when decimal.TryParse(stringValue, NumberStyles.Any, CultureInfo.InvariantCulture, out var parsedNumber):
                number = parsedNumber;
                return true;
            default:
                number = 0m;
                return false;
        }
    }

    private static bool TryConvertToInt(object value, out int number)
    {
        switch (value)
        {
            case byte byteValue:
                number = byteValue;
                return true;
            case sbyte sbyteValue:
                number = sbyteValue;
                return true;
            case short shortValue:
                number = shortValue;
                return true;
            case ushort ushortValue:
                number = ushortValue;
                return true;
            case int intValue:
                number = intValue;
                return true;
            case uint uintValue when uintValue <= int.MaxValue:
                number = (int)uintValue;
                return true;
            case long longValue when longValue is >= int.MinValue and <= int.MaxValue:
                number = (int)longValue;
                return true;
            case ulong ulongValue when ulongValue <= int.MaxValue:
                number = (int)ulongValue;
                return true;
            case decimal decimalValue when decimalValue >= int.MinValue && decimalValue <= int.MaxValue:
                number = (int)decimal.Truncate(decimalValue);
                return true;
            case float floatValue when floatValue >= int.MinValue && floatValue <= int.MaxValue:
                number = (int)MathF.Truncate(floatValue);
                return true;
            case double doubleValue when doubleValue >= int.MinValue && doubleValue <= int.MaxValue:
                number = (int)Math.Truncate(doubleValue);
                return true;
            case string stringValue when int.TryParse(stringValue, out var parsedNumber):
                number = parsedNumber;
                return true;
            default:
                number = 0;
                return false;
        }
    }

    private static IReadOnlyList<string> CollectVisibleText(Node? root, int maxCount)
    {
        if (root is null || maxCount <= 0)
        {
            return Array.Empty<string>();
        }

        var texts = new List<string>(maxCount);
        var seenTexts = new HashSet<string>(StringComparer.Ordinal);

        void Visit(Node node)
        {
            if (texts.Count >= maxCount || !GodotObject.IsInstanceValid(node) || !IsNodeVisible(node))
            {
                return;
            }

            var text = TryGetOwnNodeText(node);
            if (!string.IsNullOrWhiteSpace(text) && seenTexts.Add(text))
            {
                texts.Add(text);
                if (texts.Count >= maxCount)
                {
                    return;
                }
            }

            foreach (Node child in node.GetChildren())
            {
                Visit(child);
                if (texts.Count >= maxCount)
                {
                    return;
                }
            }
        }

        Visit(root);
        return texts;
    }

    private static string TryGetNodeText(Node? node)
    {
        return CollectVisibleText(node, 1).FirstOrDefault() ?? string.Empty;
    }

    private static string TryGetOwnNodeText(Node node)
    {
        var propertyNames = new[]
        {
            "Text",
            "Title",
            "Label",
            "Subtitle",
            "Description",
            "CurrentText",
            "Value"
        };

        foreach (var propertyName in propertyNames)
        {
            var text = TryGetTextFromValue(GetHiddenPropertyObjectValue(node, propertyName));
            if (!string.IsNullOrWhiteSpace(text))
            {
                return text;
            }
        }

        var fieldNames = new[]
        {
            "_label",
            "_title",
            "_text",
            "Label",
            "Title",
            "Text"
        };

        foreach (var fieldName in fieldNames)
        {
            var text = TryGetTextFromValue(GetHiddenFieldValue(node, fieldName));
            if (!string.IsNullOrWhiteSpace(text))
            {
                return text;
            }
        }

        if (node is Label label)
        {
            return DescribeText(label.Text);
        }

        if (node is RichTextLabel richTextLabel)
        {
            return DescribeText(richTextLabel.Text);
        }

        return string.Empty;
    }

    private static string TryGetTextFromValue(object? value)
    {
        return value switch
        {
            null => string.Empty,
            string text => DescribeText(text),
            Node node => TryGetNodeText(node),
            _ when value is System.Collections.IEnumerable => string.Empty,
            _ => DescribeText(value)
        };
    }

    private static string NormalizeComparableText(string? text)
    {
        if (string.IsNullOrWhiteSpace(text))
        {
            return string.Empty;
        }

        var stripped = StripBbCode(text.ReplaceLineEndings("\n"));
        var builder = new StringBuilder(stripped.Length);
        var previousWasWhitespace = false;

        foreach (var rune in stripped.Trim())
        {
            if (char.IsWhiteSpace(rune))
            {
                if (!previousWasWhitespace)
                {
                    builder.Append(' ');
                    previousWasWhitespace = true;
                }

                continue;
            }

            builder.Append(rune);
            previousWasWhitespace = false;
        }

        return builder.ToString();
    }

    private static string StripBbCode(string text)
    {
        if (string.IsNullOrEmpty(text))
        {
            return string.Empty;
        }

        var builder = new StringBuilder(text.Length);
        var insideTag = false;

        foreach (var character in text)
        {
            if (character == '[')
            {
                insideTag = true;
                continue;
            }

            if (character == ']')
            {
                insideTag = false;
                continue;
            }

            if (!insideTag)
            {
                builder.Append(character);
            }
        }

        return builder.ToString();
    }

    private static string? TryGetMainMenuSemanticAction(string text)
    {
        var normalized = NormalizeMenuText(text);
        if (string.IsNullOrEmpty(normalized))
        {
            return null;
        }

        if (normalized.Contains("continue") || normalized.Contains("继续游戏"))
        {
            return "continue";
        }

        if (normalized.Contains("abandoncurrentgame") || normalized.Contains("放弃当前游戏"))
        {
            return "abandon_current_game";
        }

        if (normalized.Contains("newgame") || normalized.Contains("新游戏"))
        {
            return "new_game";
        }

        if (normalized.Contains("singleplayer") || normalized.Contains("单人模式"))
        {
            return "singleplayer";
        }

        if (normalized.Contains("multiplayer") || normalized.Contains("多人模式"))
        {
            return "multiplayer";
        }

        if (normalized.Contains("timeline") || normalized.Contains("时间线"))
        {
            return "timeline";
        }

        if (normalized.Contains("settings") || normalized.Contains("设置"))
        {
            return "settings";
        }

        if (normalized.Contains("compendium") || normalized.Contains("百科大全"))
        {
            return "compendium";
        }

        if (normalized.Contains("quit") || normalized.Contains("exit") || normalized.Contains("退出"))
        {
            return "quit";
        }

        return null;
    }

    private static string? TryGetAbandonConfirmSemanticAction(string text)
    {
        var normalized = NormalizeMenuText(text);
        if (string.IsNullOrEmpty(normalized))
        {
            return null;
        }

        if (normalized.Contains("cancel") ||
            normalized.Contains("取消") ||
            normalized.Contains("不了") ||
            normalized.Equals("否", StringComparison.Ordinal))
        {
            return "cancel";
        }

        if (normalized.Contains("confirm") ||
            normalized.Contains("abandon") ||
            normalized.Contains("确认") ||
            normalized.Contains("好的") ||
            normalized.Contains("放弃") ||
            normalized.Equals("是", StringComparison.Ordinal))
        {
            return "confirm";
        }

        return null;
    }

    private static string NormalizeMenuText(string text)
    {
        return string.Concat(text.Where(static character => !char.IsWhiteSpace(character))).ToLowerInvariant();
    }

    private static JsonNode? PruneSemanticStateNode(JsonNode? node)
    {
        switch (node)
        {
            case null:
                return null;
            case JsonObject jsonObject:
            {
                var result = new JsonObject();
                foreach (var property in jsonObject)
                {
                    if (SemanticStateExcludedPropertyNames.Contains(property.Key))
                    {
                        continue;
                    }

                    var prunedValue = PruneSemanticStateNode(property.Value);
                    if (prunedValue is null)
                    {
                        continue;
                    }

                    result[property.Key] = prunedValue;
                }

                return result;
            }
            case JsonArray jsonArray:
            {
                var result = new JsonArray();
                foreach (var item in jsonArray)
                {
                    result.Add(PruneSemanticStateNode(item));
                }

                return result;
            }
            default:
                return node.DeepClone();
        }
    }

    private static string ComputeStateHash(object payload)
    {
        var json = JsonSerializer.Serialize(payload, HashJsonOptions);
        var bytes = SHA256.HashData(Encoding.UTF8.GetBytes(json));
        return Convert.ToHexString(bytes).ToLowerInvariant();
    }

    private static string ComputeFrontierHash(object payload)
    {
        return ComputeStateHash(payload);
    }

    private static object BuildFrontierStatePayload(
        BridgeSnapshot snapshot,
        long sequence)
    {
        return CreateStatePayload(snapshot.Fields, sequence, snapshot.FrontierHash, snapshot.StateHash);
    }

    private sealed class BridgeWorldContext
    {
        public required NGame Game { get; init; }

        public NRun? RunNode { get; init; }

        public RunManager? RunManager { get; init; }

        public CombatManager? CombatManager { get; init; }

        public RunState? RunState { get; init; }

        public CombatState? CombatState { get; init; }

        public required string Screen { get; init; }

        public NCombatRoom? CombatRoom { get; init; }

        public NCombatUi? CombatUi { get; init; }

        public NEndTurnButton? EndTurnButton { get; init; }

        public NProceedButton? ProceedButton { get; init; }

        public NMapScreen? MapScreen { get; init; }

        public NRestSiteRoom? RestSiteRoom { get; init; }

        public NMerchantRoom? MerchantRoom { get; init; }

        public NMerchantInventory? MerchantInventory { get; init; }

        public NTreasureRoom? TreasureRoom { get; init; }

        public NTreasureButton? TreasureChestButton { get; init; }

        public NTreasureRoomRelicCollection? TreasureRelicCollection { get; init; }

        public NRewardsScreen? RewardsScreen { get; init; }

        public NProceedButton? RewardProceedButton { get; init; }

        public NCardRewardSelectionScreen? CardRewardScreen { get; init; }

        public Node? CardRewardSkipButton { get; init; }

        public Node? CardSelectionScreen { get; init; }

        public NCharacterSelectScreen? CharacterSelectScreen { get; init; }

        public NDeckUpgradeSelectScreen? DeckUpgradeScreen { get; init; }

        public NProceedButton? RestSiteProceedButton { get; init; }

        public NMerchantButton? MerchantButton { get; init; }

        public NProceedButton? MerchantProceedButton { get; init; }

        public NBackButton? MerchantBackButton { get; init; }

        public NCharacterSelectButton? SelectedCharacterButton { get; init; }

        public NConfirmButton? EmbarkButton { get; init; }

        public Node? CardSelectionConfirmButton { get; init; }

        public Node? CardSelectionCancelButton { get; init; }

        public Node? CardSelectionCloseButton { get; init; }

        public Node? CardSelectionSkipButton { get; init; }

        public NConfirmButton? DeckUpgradeConfirmButton { get; init; }

        public NBackButton? DeckUpgradeCancelButton { get; init; }

        public NBackButton? DeckUpgradeCloseButton { get; init; }

        public required IReadOnlyList<NRewardButton> RewardButtons { get; init; }

        public required IReadOnlyList<NCardHolder> CardRewardOptions { get; init; }

        public required IReadOnlyList<NCardHolder> CardSelectionOptions { get; init; }

        public required IReadOnlyList<NCardHolder> DeckUpgradeOptions { get; init; }

        public required IReadOnlyList<NCharacterSelectButton> CharacterButtons { get; init; }

        public required IReadOnlyList<NEventOptionButton> EventOptionButtons { get; init; }

        public NEventRoom? EventRoom { get; init; }

        public NGameOverScreen? GameOverScreen { get; init; }

        public NGameOverContinueButton? GameOverContinueButton { get; init; }

        public NReturnToMainMenuButton? GameOverMainMenuButton { get; init; }

        public NCrystalSphereScreen? CrystalSphereScreen { get; init; }

        public required IReadOnlyList<NCrystalSphereCell> CrystalSphereCells { get; init; }

        public NDivinationButton? CrystalSphereSmallDivinationButton { get; init; }

        public NDivinationButton? CrystalSphereBigDivinationButton { get; init; }

        public NProceedButton? CrystalSphereProceedButton { get; init; }

        public Node? HoverTipSet { get; init; }

        public required IReadOnlyList<NMapPoint> MapPoints { get; init; }

        public required IReadOnlyList<NRestSiteButton> RestSiteButtons { get; init; }

        public required IReadOnlyList<NMerchantSlot> MerchantSlots { get; init; }

        public required IReadOnlyList<NTreasureRoomRelicHolder> TreasureRelicOptions { get; init; }

        public Node? MainMenuRoot { get; init; }

        public Node? MainMenuContinueButton { get; init; }

        public required IReadOnlyList<Node> MainMenuTextButtons { get; init; }

        public Node? RunModeSubmenu { get; init; }

        public Node? RunModeStandardButton { get; init; }

        public Node? RunModeDailyButton { get; init; }

        public Node? RunModeCustomButton { get; init; }

        public NBackButton? RunModeBackButton { get; init; }

        public Node? ContinueRunInfo { get; init; }

        public Node? AbandonRunConfirmPopup { get; init; }

        public required IReadOnlyList<NPopupYesNoButton> AbandonRunConfirmButtons { get; init; }
    }

    private sealed class BridgeResolvedAction
    {
        public required string ActionId { get; init; }

        public required object Payload { get; init; }

        public required Action Execute { get; init; }
    }

    private sealed class ObservedFrontier
    {
        public required long Sequence { get; init; }

        public required string FrontierHash { get; init; }

        public required BridgeSnapshot Snapshot { get; init; }

        public required object StatePayload { get; init; }
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

        private static readonly object Sync = new();
        private static readonly Dictionary<Guid, Channel<ObservedFrontier>> Subscribers = new();
        private static readonly Dictionary<Guid, FrontierWaiter> Waiters = new();
        private static ObservedFrontier? _current;
        private static string? _candidateFrontierHash;
        private static BridgeFrontierCandidate? _candidate;
        private static int _candidateStableTicks;
        private static long _nextSequence = 1;

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
            bool shouldSample;
            lock (Sync)
            {
                shouldSample = Subscribers.Count > 0 || Waiters.Count > 0;
            }

            if (!shouldSample)
            {
                return;
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
            CancellationToken cancellationToken)
        {
            response.StatusCode = (int)HttpStatusCode.OK;
            response.ContentType = "text/event-stream; charset=utf-8";
            response.SendChunked = true;
            response.KeepAlive = true;
            response.Headers["Cache-Control"] = "no-cache";
            response.Headers["X-Accel-Buffering"] = "no";

            var subscriberId = Guid.NewGuid();
            var channel = Channel.CreateUnbounded<ObservedFrontier>(new UnboundedChannelOptions
            {
                SingleReader = true,
                SingleWriter = false,
                AllowSynchronousContinuations = false
            });

            lock (Sync)
            {
                Subscribers[subscriberId] = channel;
            }

            try
            {
                var initialFrontier = await ObserveFrontierAsync(cancellationToken);
                await WriteFrontierEventAsync(response, initialFrontier, cancellationToken);
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
                        await WriteFrontierEventAsync(response, latestFrontier, cancellationToken);
                    }
                }
            }
            catch (OperationCanceledException)
            {
            }
            catch (HttpListenerException)
            {
            }
            catch (IOException)
            {
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
                catch
                {
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
            var payload = BuildFrontierStatePayload(snapshot, sequence);
            _current = new ObservedFrontier
            {
                Sequence = sequence,
                FrontierHash = frontierHash,
                Snapshot = snapshot,
                StatePayload = payload
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
            CancellationToken cancellationToken)
        {
            var payload = JsonSerializer.Serialize(new
            {
                ok = true,
                event_type = "frontier",
                state = frontier.StatePayload
            }, HashJsonOptions);
            await WriteSseEventAsync(response, "frontier", payload, cancellationToken);
        }

        private static async Task WriteSseEventAsync(
            HttpListenerResponse response,
            string eventName,
            string payload,
            CancellationToken cancellationToken)
        {
            var builder = new StringBuilder();
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

    private sealed class ResolvedCardTarget
    {
        public string? ActionSuffix { get; init; }

        public string? LabelSuffix { get; init; }

        public Creature? Target { get; init; }

        public required bool RequiresTargetSelection { get; init; }
    }

    private sealed class ResolvedPotionTarget
    {
        public string? ActionSuffix { get; init; }

        public string? LabelSuffix { get; init; }

        public Creature? Target { get; init; }

        public required bool RequiresTargetSelection { get; init; }
    }

    private sealed record CrystalSphereEventOptionDescriptor
    {
        public required int Index { get; init; }

        public required string OptionType { get; init; }

        public string? OptionId { get; init; }

        public required string Title { get; init; }

        public string? Description { get; init; }

        public bool IsProceed { get; init; }

        public bool IsSelected { get; init; }

        public bool IsEnabled { get; init; }

        public bool ActionAvailable { get; init; }

        public string? DivinationSize { get; init; }

        public int? X { get; init; }

        public int? Y { get; init; }

        public bool? IsHighlighted { get; init; }

        public Action? Execute { get; init; }
    }

    private sealed class BridgeStateFields
    {
        public required string Screen { get; init; }

        public required object Automation { get; init; }

        public required object Run { get; init; }

        public required object Combat { get; init; }

        public required object[] Players { get; init; }

        public required object Rewards { get; init; }

        public required object CardRewardSelection { get; init; }

        public required object CardSelection { get; init; }

        public required object CharacterSelection { get; init; }

        public required object RunModeSelection { get; init; }

        public required object EventOptions { get; init; }

        public required object CrystalSphere { get; init; }

        public required object Map { get; init; }

        public required object RestSite { get; init; }

        public required object DeckUpgradeSelection { get; init; }

        public required object Shop { get; init; }

        public required object MainMenu { get; init; }

        public required object[] AvailableActions { get; init; }
    }

    private sealed class BridgeSnapshot
    {
        public required long StateVersion { get; init; }

        public required string FrontierHash { get; init; }

        public required string StateHash { get; init; }

        public required BridgeStateFields Fields { get; init; }

        public required object StatePayload { get; init; }

        public required IReadOnlyList<BridgeResolvedAction> Actions { get; init; }

        public required IReadOnlyList<object> ActionPayloads { get; init; }

        public required IReadOnlyDictionary<string, BridgeResolvedAction> ActionLookup { get; init; }
    }
}
