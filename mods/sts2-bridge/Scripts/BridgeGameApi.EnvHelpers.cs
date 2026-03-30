using System.Collections.Generic;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Text.RegularExpressions;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.MonsterMoves.Intents;
using MegaCrit.Sts2.Core.Models.Powers;

namespace Sts2McpBridge.Scripts;

internal static partial class BridgeGameApi
{
    private const int EnvDefenseBlockAmount = 999;
    private const int EnvDefensePlatingAmount = 999;
    private const double EnvRewardHpLossWeight = -0.15d;
    private const double EnvRewardHpGainWeight = 0.05d;
    private const double EnvRewardRoomHpDeltaWeight = 1.00d;
    private const double EnvRewardFloorDeltaWeight = 0.10d;
    private const double EnvRewardGoldGainWeight = 0.0d;
    private const double EnvRewardGoldSpendWeight = 0.0d;
    private const double EnvRewardRelicGainWeight = 0.0d;
    private const double EnvRewardPotionGainWeight = 0.0d;
    private const double EnvRewardCardAddWeight = 0.0d;
    private const double EnvRewardStarterRemoveWeight = 0.0d;
    private const double EnvRewardOtherRemoveWeight = 0.0d;
    private const double EnvRewardCardUpgradeWeight = 0.0d;
    private const double EnvRewardCombatWinBonus = 0.0d;
    private const double EnvRewardEliteClearBonus = 0.0d;
    private const double EnvRewardBossClearBonus = 0.0d;
    private const double EnvRewardActClearBonus = 0.0d;
    private const double EnvRewardDeathPenalty = -10.00d;
    private const double EnvRewardVictoryBonus = 10.00d;
    private const double EnvRewardStepPenalty = 0.0d;
    private const double EnvRewardActionErrorPenalty = -0.10d;
    private const double EnvRewardTruncatedPenalty = -1.00d;
    private const double EnvRewardEndTurnWastePenalty = 0.0d;
    private const double EnvRewardNoProgressPenalty = 0.0d;
    private const double EnvRewardPlayCardBonus = 0.0d;
    private const double EnvRewardEffectiveBlockWeight = 0.00d;
    private const double EnvRewardWastedBlockWeight = 0.0d;
    private const double EnvRewardWeakIntentReductionWeight = 0.0d;
    private const double EnvRewardVulnerableRealizedDamageWeight = 0.0d;
    private const double EnvRewardThreatGapReductionWeight = 0.0d;
    private const double EnvRewardMissedDefensePenaltyWeight = 0.00d;
    private const double EnvRewardSkipBadCardsBonus = 0.0d;
    private const double EnvRewardRestLowHpBonus = 0.0d;
    private const double EnvRewardRestHighHpMismatchPenalty = 0.0d;
    private const double EnvRewardSmithHealthyBonus = 0.0d;
    private const double EnvRewardSmithLowHpMismatchPenalty = 0.0d;
    private const double EnvRewardCardHeuristicLimit = 0.0d;

    private static object BuildEnvActionPayload(BridgeResolvedAction action, int index)
    {
        var payload = JsonSerializer.SerializeToElement(action.Payload);
        var kind = TryGetNestedString(payload, "kind") ?? InferEnvActionKind(action.ActionId);
        var entry = new Dictionary<string, object?>
        {
            ["idx"] = index,
            ["action_id"] = action.ActionId,
            ["kind"] = kind
        };

        switch (kind)
        {
            case "play_card":
                entry["card_ref"] = TryGetNestedString(payload, "card_ref");
                entry["hand_index"] = TryGetNestedInt(payload, "hand_index");
                entry["card"] = CompactCardPayload(TryGetNestedElement(payload, "card"), TryGetNestedString(payload, "card_ref"));
                entry["target"] = new
                {
                    combat_id = TryGetNestedInt(payload, "target_combat_id"),
                    name = TryGetNestedString(payload, "target_name"),
                    side = TryGetNestedString(payload, "target_side")
                };
                break;

            case "use_potion":
                entry["slot"] = TryGetNestedInt(payload, "slot_index");
                entry["potion"] = CompactPotionPayload(TryGetNestedElement(payload, "potion"));
                entry["target"] = new
                {
                    combat_id = TryGetNestedInt(payload, "target_combat_id"),
                    name = TryGetNestedString(payload, "target_name")
                };
                break;

            case "discard_potion":
                entry["slot"] = TryGetNestedInt(payload, "slot_index");
                entry["potion"] = CompactPotionPayload(TryGetNestedElement(payload, "potion"));
                break;

            case "reward":
                entry["reward"] = CompactRewardPayload(TryGetNestedElement(payload, "reward"));
                break;

            case "card_reward":
                entry["selection"] = TryGetNestedString(payload, "selection_action") ?? "pick";
                entry["card"] = CompactCardPayload(TryGetNestedElement(payload, "card"));
                break;

            case "event_option":
                entry["index"] = TryGetNestedInt(payload, "index");
                entry["title"] = TryGetNestedString(payload, "option", "title");
                entry["option_type"] = TryGetNestedString(payload, "option", "option_type");
                entry["proceed"] = TryGetNestedBool(payload, "option", "is_proceed");
                entry["coord"] = CompactCoordPayload(TryGetNestedElement(payload, "option", "coord"));
                break;

            case "map":
                entry["coord"] = CompactCoordPayload(TryGetNestedElement(payload, "coord"));
                entry["point_type"] = TryGetNestedString(payload, "point_type");
                entry["point_type_norm"] = TryGetNestedString(payload, "point_type_norm");
                entry["route_summary"] = CompactMapRouteSummaryPayload(TryGetNestedElement(payload, "route_summary"));
                entry["route_nodes"] = CompactMapRouteNodesPayload(TryGetNestedElement(payload, "route_summary"));
                break;

            case "rest_site":
                entry["option"] = CompactRestSiteOptionPayload(TryGetNestedElement(payload, "option"));
                break;

            case "deck_upgrade":
                entry["selection"] = TryGetNestedString(payload, "upgrade_action");
                entry["index"] = TryGetNestedInt(payload, "index");
                entry["selection_semantics"] = TryGetNestedString(payload, "selection_semantics");
                entry["card"] = CompactCardPayload(TryGetNestedElement(payload, "card"));
                entry["upgrade_preview"] = CompactCardPayload(TryGetNestedElement(payload, "upgrade_preview"));
                break;

            case "card_selection":
                entry["selection"] = TryGetNestedString(payload, "selection_action");
                entry["index"] = TryGetNestedInt(payload, "index");
                entry["selection_id"] = TryGetNestedString(payload, "selection_id");
                entry["selection_semantics"] = TryGetNestedString(payload, "selection_semantics");
                entry["card"] = CompactCardPayload(TryGetNestedElement(payload, "card"));
                break;

            case "shop":
                entry["shop_action"] = TryGetNestedString(payload, "shop_action");
                entry["item"] = CompactShopItemPayload(TryGetNestedElement(payload, "item"));
                break;

            case "treasure_relic":
                entry["index"] = TryGetNestedInt(payload, "index");
                entry["relic"] = CompactRelicPayload(TryGetNestedElement(payload, "relic"));
                break;

            case "character_select":
                entry["index"] = TryGetNestedInt(payload, "index");
                entry["character"] = CompactCharacterPayload(TryGetNestedElement(payload, "character"));
                break;

            case "run_mode_selection":
                entry["run_mode"] = TryGetNestedString(payload, "run_mode_action");
                break;

            case "main_menu":
                entry["menu_action"] = TryGetNestedString(payload, "menu_action");
                break;

            case "proceed":
                entry["skip"] = TryGetNestedBool(payload, "is_skip");
                entry["source"] = TryGetNestedString(payload, "proceed_source");
                break;

            default:
                entry["label"] = TryGetNestedString(payload, "label");
                break;
        }

        // Canonical Chinese semantic text for every action
        entry["canonical_text"] = BuildCanonicalActionText(kind, action.ActionId, payload);

        return entry;
    }

    private static string InferEnvActionKind(string actionId)
    {
        if (actionId.StartsWith("play_card:", StringComparison.Ordinal))
        {
            return "play_card";
        }

        if (actionId.StartsWith("use_potion:", StringComparison.Ordinal))
        {
            return "use_potion";
        }

        if (actionId.StartsWith("discard_potion:", StringComparison.Ordinal))
        {
            return "discard_potion";
        }

        if (actionId.StartsWith("reward:", StringComparison.Ordinal))
        {
            return "reward";
        }

        if (actionId.StartsWith("card_reward:", StringComparison.Ordinal))
        {
            return "card_reward";
        }

        if (actionId.StartsWith("event_option:", StringComparison.Ordinal))
        {
            return "event_option";
        }

        if (actionId.StartsWith("map:", StringComparison.Ordinal))
        {
            return "map";
        }

        if (actionId.StartsWith("rest_site:", StringComparison.Ordinal))
        {
            return "rest_site";
        }

        if (actionId.StartsWith("deck_upgrade:", StringComparison.Ordinal))
        {
            return "deck_upgrade";
        }

        if (actionId.StartsWith("card_selection:", StringComparison.Ordinal))
        {
            return "card_selection";
        }

        if (actionId.StartsWith("shop:", StringComparison.Ordinal))
        {
            return "shop";
        }

        if (actionId.StartsWith("treasure_relic:", StringComparison.Ordinal))
        {
            return "treasure_relic";
        }

        if (actionId.StartsWith("treasure:", StringComparison.Ordinal))
        {
            return "treasure";
        }

        if (actionId.StartsWith("character_select:", StringComparison.Ordinal) || actionId == "embark")
        {
            return "character_select";
        }

        if (actionId.StartsWith("run_mode:", StringComparison.Ordinal))
        {
            return "run_mode_selection";
        }

        if (actionId.StartsWith("main_menu:", StringComparison.Ordinal))
        {
            return "main_menu";
        }

        return actionId switch
        {
            "end_turn" => "combat",
            "proceed" => "proceed",
            _ => "action"
        };
    }

    private static BridgeResolvedActionSelection ResolveRequestedEnvAction(
        BridgeEnvSnapshot snapshot,
        BridgeEnvStepRequest request)
    {
        var requestedActionId = request.ActionId?.Trim();
        var requestedIndex = request.ActionIndex;

        if (string.IsNullOrWhiteSpace(requestedActionId) && requestedIndex is null)
        {
            throw new BridgeRequestException(
                System.Net.HttpStatusCode.BadRequest,
                "missing_action_selector",
                "Provide either action_id or action_index for env/step.");
        }

        BridgeResolvedAction? actionFromId = null;
        BridgeResolvedAction? actionFromIndex = null;

        if (!string.IsNullOrWhiteSpace(requestedActionId))
        {
            snapshot.ActionLookup.TryGetValue(requestedActionId, out actionFromId);
            if (actionFromId is null)
            {
                throw new BridgeRequestException(
                    System.Net.HttpStatusCode.Conflict,
                    "action_not_available",
                    $"Action '{requestedActionId}' is not currently available.",
                    new
                    {
                        action_id = requestedActionId,
                        phase = snapshot.Phase,
                        legal_actions = snapshot.LegalActions
                    });
            }
        }

        if (requestedIndex is int index)
        {
            if (index < 0 || index >= snapshot.ResolvedActions.Count)
            {
                throw new BridgeRequestException(
                    System.Net.HttpStatusCode.Conflict,
                    "action_index_out_of_range",
                    $"Action index {index} is out of range for the current legal action list.",
                    new
                    {
                        requested_index = index,
                        legal_action_count = snapshot.ResolvedActions.Count
                    });
            }

            actionFromIndex = snapshot.ResolvedActions[index];
        }

        if (actionFromId is not null && actionFromIndex is not null && !ReferenceEquals(actionFromId, actionFromIndex))
        {
            throw new BridgeRequestException(
                System.Net.HttpStatusCode.Conflict,
                "action_selector_mismatch",
                "action_id and action_index refer to different currently-legal actions.");
        }

        var action = actionFromId ?? actionFromIndex!;
        var resolvedIndex = -1;
        for (var actionIndex = 0; actionIndex < snapshot.ResolvedActions.Count; actionIndex++)
        {
            if (!ReferenceEquals(snapshot.ResolvedActions[actionIndex], action))
            {
                continue;
            }

            resolvedIndex = actionIndex;
            break;
        }

        return new BridgeResolvedActionSelection
        {
            Action = action,
            Index = resolvedIndex,
            Kind = InferEnvActionKind(action.ActionId)
        };
    }

    private static (BridgeResolvedAction Action, string Kind)? ResolveEnvResetAction(
        BridgeEnvSnapshot snapshot,
        string? requestedCharacter)
    {
        switch (snapshot.Phase)
        {
            case "terminal":
                {
                    var mainMenuAction = TryGetResetAction(snapshot, "game_over:return_to_main_menu");
                    if (mainMenuAction is not null)
                    {
                        return mainMenuAction;
                    }

                    if (snapshot.Context.GameOverScreen is not null && IsNodeVisible(snapshot.Context.GameOverScreen))
                    {
                        return (
                            new BridgeResolvedAction
                            {
                                ActionId = "game_over:return_to_main_menu",
                                Payload = new
                                {
                                    action_id = "game_over:return_to_main_menu",
                                    kind = "game_over",
                                    game_over_action = "return_to_main_menu",
                                    label = "Return to main menu",
                                    screen = snapshot.Screen
                                },
                                Execute = () => InvokeGameOverReturnToMainMenuAction(
                                    snapshot.Context.GameOverScreen,
                                    snapshot.Context.GameOverMainMenuButton)
                            },
                            "game_over");
                    }

                    return null;
                }

            case "startup_run_mode":
                return TryGetResetAction(snapshot, "run_mode:standard");

            case "startup_character_select":
                if (!string.IsNullOrWhiteSpace(requestedCharacter))
                {
                    var requested = NormalizeComparableText(requestedCharacter);
                    for (var index = 0; index < snapshot.Context.CharacterButtons.Count; index++)
                    {
                        var button = snapshot.Context.CharacterButtons[index];
                        if (button.IsLocked)
                        {
                            continue;
                        }

                        var character = button.Character;
                        var title = NormalizeComparableText(DescribeCharacter(character));
                        var id = NormalizeComparableText(character?.Id.ToString());
                        if (!requested.Equals(title, StringComparison.Ordinal) &&
                            !requested.Equals(id, StringComparison.Ordinal))
                        {
                            continue;
                        }

                        if (!ReferenceEquals(button, snapshot.Context.SelectedCharacterButton))
                        {
                            return TryGetResetAction(snapshot, $"character_select:{index}");
                        }

                        break;
                    }
                }

                if (snapshot.Context.SelectedCharacterButton is not null)
                {
                    var embark = TryGetResetAction(snapshot, "embark");
                    if (embark is not null)
                    {
                        return embark;
                    }
                }

                var charAction = snapshot.ResolvedActions
                    .FirstOrDefault(static action => action.ActionId.StartsWith("character_select:", StringComparison.Ordinal));
                return charAction is not null ? (charAction, "character_select") : null;

            case "startup_main_menu":
                return TryGetResetAction(snapshot, "run_mode:standard") ??
                       TryGetResetAction(snapshot, "main_menu:abandon_current_game") ??
                       TryGetResetAction(snapshot, "main_menu:confirm_abandon_run") ??
                       TryGetResetAction(snapshot, "main_menu:new_game") ??
                       TryGetResetAction(snapshot, "main_menu:singleplayer");

            default:
                return null;
        }
    }

    private static (BridgeResolvedAction Action, string Kind)? TryGetResetAction(
        BridgeEnvSnapshot snapshot,
        string actionId)
    {
        return snapshot.ActionLookup.TryGetValue(actionId, out var action)
            ? (action, InferEnvActionKind(actionId))
            : null;
    }

    private static object BuildEnvResetPayload(
        BridgeEnvEpisode episode,
        BridgeEnvSnapshot state,
        IReadOnlyList<object> resetActions)
    {
        SyncEnvEpisodeAnchor(episode, state, force: true);
        return new
        {
            ok = true,
            episode_id = episode.Id,
            step_index = episode.StepIndex,
            done = false,
            truncated = false,
            obs = state.Observation,
            legal_actions = state.LegalActions,
            info = new
            {
                phase = state.Phase,
                screen = state.Screen,
                logic_hash = state.LogicHash,
                defensive_buffs = episode.DefensiveBuffs,
                episode_mode = episode.EpisodeMode,
                encounter_id = episode.EncounterId,
                reset_actions = resetActions
            }
        };
    }

    private static object BuildEnvStepPayload(
        BridgeEnvEpisode episode,
        BridgeEnvSnapshot before,
        BridgeEnvSnapshot after,
        BridgeResolvedActionSelection? selectedAction,
        bool truncated,
        string? truncationReason,
        string? actionError = null)
    {
        var rewardBreakdown = BuildEnvRewardBreakdown(
            episode,
            before,
            after,
            selectedAction,
            truncated,
            actionError);
        var done = after.Done || truncated;
        SyncEnvEpisodeAnchor(episode, after, force: !done && HasEnvRoomTransition(before, after));
        return new
        {
            ok = true,
            episode_id = episode.Id,
            step_index = episode.StepIndex,
            reward = rewardBreakdown.Total,
            done,
            truncated,
            obs = after.Observation,
            legal_actions = done ? Array.Empty<object>() : after.LegalActions,
            info = new
            {
                action = selectedAction is null
                    ? null
                    : new
                    {
                        idx = selectedAction.Index,
                        action_id = selectedAction.Action.ActionId,
                        kind = selectedAction.Kind
                    },
                phase_before = before.Phase,
                phase_after = after.Phase,
                screen_before = before.Screen,
                screen_after = after.Screen,
                logic_hash_before = before.LogicHash,
                logic_hash_after = after.LogicHash,
                defensive_buffs = episode.DefensiveBuffs,
                episode_mode = episode.EpisodeMode,
                encounter_id = episode.EncounterId,
                truncation_reason = truncationReason,
                action_error = actionError,
                reward_breakdown = rewardBreakdown
            }
        };
    }

    private static async Task<BridgeEnvSnapshot> ApplyEnvEpisodeAdjustmentsAsync(
        BridgeEnvEpisode episode,
        BridgeEnvSnapshot snapshot,
        CancellationToken cancellationToken)
    {
        if (!episode.DefensiveBuffs || !snapshot.RunActive)
        {
            return snapshot;
        }

        var changed = await BridgeCoordinator.RunOnMainThreadAsync(() => ApplyEnvDefensiveBuffs(CaptureContext()));
        if (!changed)
        {
            return snapshot;
        }

        await BridgeCoordinator.WaitForPumpTicksAsync(1, cancellationToken);
        return await CaptureEnvSnapshotAsync(cancellationToken);
    }

    private static bool ApplyEnvDefensiveBuffs(BridgeWorldContext context)
    {
        var creature = GetPrimaryPlayer(context)?.Creature;
        if (creature is null || !creature.IsAlive)
        {
            return false;
        }

        var changed = false;

        if (creature.CurrentHp < creature.MaxHp)
        {
            creature.HealInternal((decimal)(creature.MaxHp - creature.CurrentHp));
            changed = true;
        }

        if (context.CombatManager?.IsInProgress == true)
        {
            if (creature.Block < EnvDefenseBlockAmount)
            {
                creature.GainBlockInternal((decimal)(EnvDefenseBlockAmount - creature.Block));
                changed = true;
            }

            if (EnsureCreaturePowerAmount<PlatingPower>(creature, EnvDefensePlatingAmount))
            {
                changed = true;
            }
        }

        return changed;
    }

    private static bool EnsureCreaturePowerAmount<TPower>(Creature creature, int amount)
        where TPower : PowerModel
    {
        var existing = creature.Powers.OfType<TPower>().FirstOrDefault();
        if (existing is not null)
        {
            var changed = false;
            if (existing.Amount < amount)
            {
                existing.Amount = amount;
                changed = true;
            }

            if (existing.AmountOnTurnStart < amount)
            {
                existing.AmountOnTurnStart = amount;
                changed = true;
            }

            return changed;
        }

        var power = (TPower)ModelDb.Power<TPower>().ToMutable();
        power.Applier = creature;
        power.Target = creature;
        power.ApplyInternal(creature, amount, silent: true);
        power.AmountOnTurnStart = amount;
        return true;
    }

    private static bool TrySetHiddenPropertyValue(object? target, string propertyName, object? value)
    {
        if (target is null)
        {
            return false;
        }

        var property = FindProperty(target.GetType(), propertyName);
        var setter = property?.GetSetMethod(nonPublic: true);
        if (setter is null)
        {
            return false;
        }

        setter.Invoke(target, new[] { value });
        return true;
    }

    private static BridgeEnvRewardBreakdown BuildEnvRewardBreakdown(
        BridgeEnvEpisode episode,
        BridgeEnvSnapshot before,
        BridgeEnvSnapshot after,
        BridgeResolvedActionSelection? selectedAction,
        bool truncated,
        string? actionError)
    {
        var maxHp = Math.Max(after.MaxHp, before.MaxHp);
        var hpLoss = Math.Max(0, before.CurrentHp - after.CurrentHp);
        var hpGain = Math.Max(0, after.CurrentHp - before.CurrentHp);
        var hpLossNormalized = maxHp > 0 ? (double)hpLoss / maxHp : 0d;
        var hpGainNormalized = maxHp > 0 ? (double)hpGain / maxHp : 0d;
        var floorDelta = Math.Max(0, after.TotalFloor - before.TotalFloor);
        var roomComplete = HasEnvRoomTransition(before, after) ? 1 : 0;
        var roomHpMax = episode.RoomStartMaxHp > 0 ? episode.RoomStartMaxHp : maxHp;
        var roomHpDeltaNormalized = roomComplete == 1 && roomHpMax > 0
            ? (double)(after.CurrentHp - episode.RoomStartHp) / roomHpMax
            : 0d;
        var death = after.Done && after.CurrentHp <= 0 ? 1 : 0;
        var victory = after.Done && after.CurrentHp > 0 ? 1 : 0;
        var actionErrorPenalty = string.IsNullOrWhiteSpace(actionError) ? 0d : EnvRewardActionErrorPenalty;
        var truncatedPenalty = truncated ? EnvRewardTruncatedPenalty : 0d;
        var total =
            hpLossNormalized * EnvRewardHpLossWeight +
            hpGainNormalized * EnvRewardHpGainWeight +
            roomHpDeltaNormalized * EnvRewardRoomHpDeltaWeight +
            floorDelta * EnvRewardFloorDeltaWeight +
            death * EnvRewardDeathPenalty +
            victory * EnvRewardVictoryBonus +
            actionErrorPenalty +
            truncatedPenalty;

        return new BridgeEnvRewardBreakdown
        {
            HpLossNormalized = RoundEnvNumber(hpLossNormalized),
            HpGainNormalized = RoundEnvNumber(hpGainNormalized),
            RoomComplete = roomComplete,
            RoomHpDeltaNormalized = RoundEnvNumber(roomHpDeltaNormalized),
            FloorDelta = floorDelta,
            Death = death,
            Victory = victory,
            ActionErrorPenalty = RoundEnvNumber(actionErrorPenalty),
            TruncatedPenalty = RoundEnvNumber(truncatedPenalty),
            Total = RoundEnvNumber(total)
        };
    }

    private static void SyncEnvEpisodeAnchor(BridgeEnvEpisode episode, BridgeEnvSnapshot snapshot, bool force)
    {
        var roomKey = BuildEnvRoomKey(snapshot.ActIndex, snapshot.TotalFloor, snapshot.RoomType, snapshot.RoomModelId);
        if (!force && episode.RoomAnchorInitialized && string.Equals(episode.RoomKey, roomKey, StringComparison.Ordinal))
        {
            return;
        }

        episode.RoomAnchorInitialized = true;
        episode.RoomKey = roomKey;
        episode.RoomStartHp = snapshot.CurrentHp;
        episode.RoomStartMaxHp = snapshot.MaxHp;
        episode.RoomStartFloor = snapshot.TotalFloor;
        episode.RoomStartActIndex = snapshot.ActIndex;
    }

    private static string BuildEnvRoomKey(int actIndex, int totalFloor, string? roomType, string? roomModelId)
    {
        return $"{actIndex}:{totalFloor}:{roomType ?? ""}:{roomModelId ?? ""}";
    }

    private static bool HasEnvRoomTransition(BridgeEnvSnapshot before, BridgeEnvSnapshot after)
    {
        return !BuildEnvRoomKey(before.ActIndex, before.TotalFloor, before.RoomType, before.RoomModelId)
            .Equals(BuildEnvRoomKey(after.ActIndex, after.TotalFloor, after.RoomType, after.RoomModelId), StringComparison.Ordinal);
    }

    private static bool IsEnvEliteRoom(string? roomType)
    {
        return roomType?.Contains("Elite", StringComparison.OrdinalIgnoreCase) == true;
    }

    private static bool IsEnvBossRoom(string? roomType)
    {
        return roomType?.Contains("Boss", StringComparison.OrdinalIgnoreCase) == true;
    }

    private static int GetPrimaryPlayerCurrentEnergy(BridgeWorldContext context) => GetPrimaryPlayer(context)?.PlayerCombatState?.Energy ?? 0;
    private static int GetPrimaryPlayerCurrentBlock(BridgeWorldContext context) => GetPrimaryPlayer(context)?.Creature?.Block ?? 0;
    private static int GetPrimaryPlayerPotionCount(BridgeWorldContext context) => GetPrimaryPlayer(context)?.PotionSlots.Count(static slot => slot is not null) ?? 0;

    private static IReadOnlyList<BridgeEnvEnemyState> BuildEnvEnemyStates(BridgeWorldContext context)
    {
        if (context.CombatState is null)
        {
            return Array.Empty<BridgeEnvEnemyState>();
        }

        var primaryPlayer = GetPrimaryPlayer(context)?.Creature;
        return context.CombatState.Creatures
            .Where(static creature => creature.IsEnemy)
            .Select(creature => BuildEnvEnemyState(creature, primaryPlayer))
            .ToArray();
    }

    private static BridgeEnvEnemyState BuildEnvEnemyState(Creature creature, Creature? primaryPlayer)
    {
        return new BridgeEnvEnemyState
        {
            CombatId = creature.CombatId ?? 0u,
            CurrentHp = creature.CurrentHp,
            Block = creature.Block,
            IsAlive = creature.IsAlive,
            IntentDamageToPlayer = GetEnvEnemyIntentDamageToPrimaryPlayer(creature, primaryPlayer),
            Weak = GetEnvCreaturePowerAmount(creature, "Weak", "虚弱", "虛弱"),
            Vulnerable = GetEnvCreaturePowerAmount(creature, "Vulnerable", "易伤", "易傷")
        };
    }

    private static BridgeEnvEnemyState CreateMissingEnvEnemyState(uint combatId)
    {
        return new BridgeEnvEnemyState
        {
            CombatId = combatId,
            CurrentHp = 0,
            Block = 0,
            IsAlive = false,
            IntentDamageToPlayer = 0,
            Weak = 0,
            Vulnerable = 0
        };
    }

    private static int GetEnvCreaturePowerAmount(Creature creature, params string[] aliases)
    {
        foreach (var power in creature.Powers)
        {
            var title = TextOf(power.Title);
            if (string.IsNullOrWhiteSpace(title))
            {
                continue;
            }

            foreach (var alias in aliases)
            {
                if (!string.IsNullOrWhiteSpace(alias) &&
                    title.IndexOf(alias, StringComparison.OrdinalIgnoreCase) >= 0)
                {
                    return power.Amount;
                }
            }
        }

        return 0;
    }

    private static int GetEnvEnemyIntentDamageToPrimaryPlayer(Creature enemy, Creature? primaryPlayer)
    {
        var monster = enemy.Monster;
        if (monster is null || !enemy.IsAlive)
        {
            return 0;
        }

        IReadOnlyList<Creature> targets = primaryPlayer is not null && primaryPlayer.IsAlive
            ? new[] { primaryPlayer }
            : ResolveMonsterIntentTargets(enemy);
        var nextMove = monster.NextMove;
        var intents = SafeGetMonsterIntents(monster, nextMove);
        var totalDamage = 0;

        foreach (var intent in intents)
        {
            var damage = intent switch
            {
                SingleAttackIntent singleAttackIntent => SafeGetIntentTotalDamage(singleAttackIntent, targets, enemy),
                MultiAttackIntent multiAttackIntent => SafeGetIntentTotalDamage(multiAttackIntent, targets, enemy),
                _ => null
            };

            if (damage.HasValue && damage.Value > 0)
            {
                totalDamage += damage.Value;
            }
        }

        return totalDamage;
    }

    private static int GetEnvTotalIncomingDamageToPlayer(BridgeEnvSnapshot snapshot)
    {
        return snapshot.EnemyStates
            .Where(static enemy => enemy.IsAlive)
            .Sum(static enemy => enemy.IntentDamageToPlayer);
    }

    private static IReadOnlyList<BridgeEnvDeckEntry> BuildEnvDeckEntries(BridgeWorldContext context)
    {
        var deck = GetPrimaryPlayer(context)?.Deck?.Cards;
        if (deck is null || deck.Count == 0)
        {
            return Array.Empty<BridgeEnvDeckEntry>();
        }

        var entries = new List<BridgeEnvDeckEntry>(deck.Count);
        foreach (var card in deck)
        {
            var payload = JsonSerializer.SerializeToElement(BuildCardPayload(card));
            var rarity = TryGetNestedString(payload, "rarity");
            var title = TryGetNestedString(payload, "title");
            var cardId = TryGetNestedString(payload, "id");
            var signature = string.Join(
                "|",
                cardId ?? string.Empty,
                title ?? string.Empty,
                TryGetNestedString(payload, "type") ?? string.Empty,
                rarity ?? string.Empty,
                TryGetNestedInt(payload, "resolved_energy_cost")?.ToString() ?? string.Empty,
                TryGetNestedInt(payload, "current_star_cost")?.ToString() ?? string.Empty,
                TryGetNestedString(payload, "effect_preview", "summary") ??
                TryGetNestedString(payload, "description") ??
                string.Empty);
            entries.Add(new BridgeEnvDeckEntry
            {
                Ref = GetCardReference(card),
                CardId = cardId,
                Title = title,
                Rarity = rarity,
                Signature = signature,
                IsStarter = string.Equals(rarity, "Basic", StringComparison.OrdinalIgnoreCase)
            });
        }

        return entries;
    }

    private static BridgeEnvDeckDiff DiffEnvDeckEntries(
        IReadOnlyList<BridgeEnvDeckEntry> before,
        IReadOnlyList<BridgeEnvDeckEntry> after)
    {
        var beforeByRef = before.ToDictionary(static entry => entry.Ref, StringComparer.Ordinal);
        var afterByRef = after.ToDictionary(static entry => entry.Ref, StringComparer.Ordinal);
        var cardAddCount = 0;
        var starterCardRemoveCount = 0;
        var otherCardRemoveCount = 0;
        var cardUpgradeCount = 0;

        foreach (var afterEntry in after)
        {
            if (!beforeByRef.ContainsKey(afterEntry.Ref))
            {
                cardAddCount++;
            }
        }

        foreach (var beforeEntry in before)
        {
            if (!afterByRef.ContainsKey(beforeEntry.Ref))
            {
                if (beforeEntry.IsStarter)
                {
                    starterCardRemoveCount++;
                }
                else
                {
                    otherCardRemoveCount++;
                }

                continue;
            }

            var afterEntry = afterByRef[beforeEntry.Ref];
            if (!string.Equals(beforeEntry.Signature, afterEntry.Signature, StringComparison.Ordinal))
            {
                cardUpgradeCount++;
            }
        }

        return new BridgeEnvDeckDiff
        {
            CardAddCount = cardAddCount,
            StarterCardRemoveCount = starterCardRemoveCount,
            OtherCardRemoveCount = otherCardRemoveCount,
            CardUpgradeCount = cardUpgradeCount
        };
    }

    private static BridgeEnvActionShaping EvaluateEnvActionShaping(
        BridgeEnvSnapshot before,
        BridgeEnvSnapshot after,
        BridgeResolvedActionSelection? selectedAction,
        string? actionError)
    {
        if (selectedAction is null || !string.IsNullOrWhiteSpace(actionError))
        {
            return BridgeEnvActionShaping.None;
        }

        var payload = JsonSerializer.SerializeToElement(selectedAction.Action.Payload);
        var kind = selectedAction.Kind;
        var shaping = BridgeEnvActionShaping.None;

        if (string.Equals(kind, "card_reward", StringComparison.Ordinal))
        {
            if (string.Equals(selectedAction.Action.ActionId, "card_reward:skip", StringComparison.Ordinal) ||
                string.Equals(TryGetNestedString(payload, "selection_action"), "skip", StringComparison.Ordinal))
            {
                if (ShouldRewardSkippingBadCardReward(before))
                {
                    shaping.SkipBadCardsBonus = EnvRewardSkipBadCardsBonus;
                }
            }
            else
            {
                shaping.CardChoiceBonus = ScoreEnvCardHeuristic(TryGetNestedElement(payload, "card"), before);
            }
        }
        else if (string.Equals(kind, "shop", StringComparison.Ordinal) &&
                 string.Equals(TryGetNestedString(payload, "shop_action"), "buy", StringComparison.Ordinal) &&
                 string.Equals(TryGetNestedString(payload, "item", "item_kind"), "card", StringComparison.Ordinal))
        {
            shaping.CardChoiceBonus = ScoreEnvCardHeuristic(TryGetNestedElement(payload, "item", "card"), before);
        }
        else if (string.Equals(kind, "rest_site", StringComparison.Ordinal))
        {
            ApplyRestSiteShaping(before, payload, shaping);
        }
        else if (string.Equals(kind, "play_card", StringComparison.Ordinal))
        {
            shaping.PlayCardBonus = EnvRewardPlayCardBonus;
            ApplyCombatActionShaping(before, after, shaping);
            ApplyMissedDefenseShaping(before, after, shaping);
        }
        else if (string.Equals(kind, "use_potion", StringComparison.Ordinal))
        {
            ApplyCombatActionShaping(before, after, shaping);
            ApplyMissedDefenseShaping(before, after, shaping);
        }
        else if (string.Equals(kind, "combat", StringComparison.Ordinal) &&
                 string.Equals(selectedAction.Action.ActionId, "end_turn", StringComparison.Ordinal) &&
                 HasEnvWastedEndTurn(before))
        {
            shaping.EndTurnWastePenalty = EnvRewardEndTurnWastePenalty;
            ApplyEndTurnThreatShaping(before, shaping);
        }

        return shaping;
    }

    private static void ApplyCombatActionShaping(
        BridgeEnvSnapshot before,
        BridgeEnvSnapshot after,
        BridgeEnvActionShaping shaping)
    {
        if (!before.CombatInProgress)
        {
            return;
        }

        var maxHp = Math.Max(before.MaxHp, after.MaxHp);
        if (maxHp <= 0)
        {
            return;
        }

        var incomingBefore = GetEnvTotalIncomingDamageToPlayer(before);
        var blockBefore = Math.Max(0, before.PlayerBlock);
        var blockAfter = Math.Max(0, after.PlayerBlock);
        var addedBlock = Math.Max(0, blockAfter - blockBefore);
        var threatGapBefore = Math.Max(0, incomingBefore - blockBefore);
        var incomingAfter = GetEnvTotalIncomingDamageToPlayer(after);
        var threatGapAfter = Math.Max(0, incomingAfter - blockAfter);
        var threatGapReduction = Math.Max(0, threatGapBefore - threatGapAfter);
        var effectiveBlockAdded = Math.Min(addedBlock, threatGapBefore);
        var wastedBlockAdded = Math.Max(0, addedBlock - threatGapBefore);

        shaping.ThreatGapBefore = threatGapBefore;
        shaping.ThreatGapAfter = threatGapAfter;
        shaping.ThreatGapReduction = threatGapReduction;
        shaping.ThreatGapReductionNormalized = maxHp > 0 ? (double)threatGapReduction / maxHp : 0d;
        shaping.ThreatGapReductionBonus = shaping.ThreatGapReductionNormalized * EnvRewardThreatGapReductionWeight;
        shaping.EffectiveBlockAdded = effectiveBlockAdded;
        shaping.EffectiveBlockNormalized = maxHp > 0 ? (double)effectiveBlockAdded / maxHp : 0d;
        shaping.EffectiveBlockBonus = shaping.EffectiveBlockNormalized * EnvRewardEffectiveBlockWeight;
        shaping.WastedBlockAdded = wastedBlockAdded;
        shaping.WastedBlockNormalized = maxHp > 0 ? (double)wastedBlockAdded / maxHp : 0d;
        shaping.WastedBlockPenalty = shaping.WastedBlockNormalized * EnvRewardWastedBlockWeight;

        var afterById = after.EnemyStates.ToDictionary(static enemy => enemy.CombatId);
        var weakIntentReduction = 0;
        var vulnerableRealizedDamage = 0;

        foreach (var beforeEnemy in before.EnemyStates)
        {
            if (!afterById.TryGetValue(beforeEnemy.CombatId, out var afterEnemy))
            {
                afterEnemy = CreateMissingEnvEnemyState(beforeEnemy.CombatId);
            }

            if (beforeEnemy.IsAlive &&
                afterEnemy.IsAlive &&
                afterEnemy.Weak > beforeEnemy.Weak)
            {
                weakIntentReduction += Math.Max(0, beforeEnemy.IntentDamageToPlayer - afterEnemy.IntentDamageToPlayer);
            }

            if (beforeEnemy.Vulnerable > 0)
            {
                vulnerableRealizedDamage +=
                    Math.Max(0, beforeEnemy.Block - afterEnemy.Block) +
                    Math.Max(0, beforeEnemy.CurrentHp - afterEnemy.CurrentHp);
            }
        }

        shaping.WeakIntentReduction = weakIntentReduction;
        shaping.WeakIntentReductionNormalized = maxHp > 0 ? (double)weakIntentReduction / maxHp : 0d;
        shaping.WeakBonus = shaping.WeakIntentReductionNormalized * EnvRewardWeakIntentReductionWeight;
        shaping.VulnerableRealizedDamage = vulnerableRealizedDamage;
        shaping.VulnerableRealizedDamageNormalized = maxHp > 0 ? (double)vulnerableRealizedDamage / maxHp : 0d;
        shaping.VulnerableBonus = shaping.VulnerableRealizedDamageNormalized * EnvRewardVulnerableRealizedDamageWeight;
    }

    private static void ApplyMissedDefenseShaping(
        BridgeEnvSnapshot before,
        BridgeEnvSnapshot after,
        BridgeEnvActionShaping shaping)
    {
        if (!before.CombatInProgress ||
            shaping.ThreatGapBefore <= 0 ||
            after.CurrentEnergy > 0 ||
            shaping.ThreatGapReduction > 0 ||
            !HasEnvAvailableDefenseOption(before))
        {
            return;
        }

        var maxHp = Math.Max(before.MaxHp, after.MaxHp);
        if (maxHp <= 0)
        {
            return;
        }

        shaping.MissedDefensePenalty =
            ((double)shaping.ThreatGapBefore / maxHp) * EnvRewardMissedDefensePenaltyWeight;
    }

    private static void ApplyEndTurnThreatShaping(
        BridgeEnvSnapshot before,
        BridgeEnvActionShaping shaping)
    {
        if (!before.CombatInProgress || !HasEnvAvailableDefenseOption(before))
        {
            return;
        }

        var threatGapBefore = GetEnvThreatGap(before);
        if (threatGapBefore <= 0)
        {
            return;
        }

        var maxHp = Math.Max(before.MaxHp, 1);
        shaping.ThreatGapBefore = Math.Max(shaping.ThreatGapBefore, threatGapBefore);
        shaping.MissedDefensePenalty =
            Math.Min(
                shaping.MissedDefensePenalty,
                ((double)threatGapBefore / maxHp) * EnvRewardMissedDefensePenaltyWeight);
    }

    private static int GetEnvThreatGap(BridgeEnvSnapshot snapshot)
    {
        return Math.Max(0, GetEnvTotalIncomingDamageToPlayer(snapshot) - Math.Max(0, snapshot.PlayerBlock));
    }

    private static bool HasEnvAvailableDefenseOption(BridgeEnvSnapshot snapshot)
    {
        if (!snapshot.CombatInProgress || snapshot.CurrentEnergy <= 0 || GetEnvThreatGap(snapshot) <= 0)
        {
            return false;
        }

        return snapshot.ResolvedActions.Any(action => IsEnvDefensiveAction(snapshot, action));
    }

    private static bool IsEnvDefensiveAction(BridgeEnvSnapshot snapshot, BridgeResolvedAction action)
    {
        var payload = JsonSerializer.SerializeToElement(action.Payload);
        var kind = TryGetNestedString(payload, "kind") ?? InferEnvActionKind(action.ActionId);
        if (!string.Equals(kind, "play_card", StringComparison.Ordinal) &&
            !string.Equals(kind, "use_potion", StringComparison.Ordinal))
        {
            return false;
        }

        var source = string.Equals(kind, "play_card", StringComparison.Ordinal)
            ? TryGetNestedElement(payload, "card")
            : TryGetNestedElement(payload, "potion");
        if (source is null || source.Value.ValueKind is JsonValueKind.Null or JsonValueKind.Undefined)
        {
            return false;
        }

        var block = TryGetNestedInt(source.Value, "effect_preview", "total_block") ??
                    TryExtractEnvMetric(source.Value, "block");
        if (block > 0)
        {
            return true;
        }

        var weak = TryGetNestedInt(source.Value, "effect_preview", "weak") ??
                   TryExtractEnvMetric(source.Value, "weak");
        if (weak <= 0)
        {
            return false;
        }

        var targetCombatId = TryGetNestedInt(payload, "target_combat_id");
        if (targetCombatId.HasValue)
        {
            return snapshot.EnemyStates.Any(enemy =>
                enemy.CombatId == (uint)targetCombatId.Value &&
                enemy.IsAlive &&
                enemy.IntentDamageToPlayer > 0);
        }

        return snapshot.EnemyStates.Any(enemy => enemy.IsAlive && enemy.IntentDamageToPlayer > 0);
    }

    private static bool HasEnvWastedEndTurn(BridgeEnvSnapshot snapshot)
    {
        if (!snapshot.CombatInProgress || !string.Equals(snapshot.Phase, "combat", StringComparison.Ordinal))
        {
            return false;
        }

        if (snapshot.CurrentEnergy <= 0)
        {
            return false;
        }

        return snapshot.ResolvedActions.Any(action =>
            !string.Equals(action.ActionId, "end_turn", StringComparison.Ordinal) &&
            (action.ActionId.StartsWith("play_card:", StringComparison.Ordinal) ||
             action.ActionId.StartsWith("use_potion:", StringComparison.Ordinal)));
    }

    private static bool ShouldRewardSkippingBadCardReward(BridgeEnvSnapshot snapshot)
    {
        var candidateScores = snapshot.ResolvedActions
            .Where(static action => action.ActionId.StartsWith("card_reward:", StringComparison.Ordinal) &&
                                    !string.Equals(action.ActionId, "card_reward:skip", StringComparison.Ordinal))
            .Select(action => ScoreEnvCardHeuristic(
                TryGetNestedElement(JsonSerializer.SerializeToElement(action.Payload), "card"),
                snapshot))
            .ToArray();

        return candidateScores.Length > 0 && candidateScores.All(static score => score <= 0d);
    }

    private static void ApplyRestSiteShaping(
        BridgeEnvSnapshot before,
        JsonElement payload,
        BridgeEnvActionShaping shaping)
    {
        var hpRatio = before.MaxHp > 0 ? (double)before.CurrentHp / before.MaxHp : 0d;
        var optionType = TryGetNestedString(payload, "option", "option_type") ?? string.Empty;
        if (optionType.Contains("Heal", StringComparison.OrdinalIgnoreCase))
        {
            if (hpRatio < 0.5d)
            {
                shaping.RestBonus = EnvRewardRestLowHpBonus;
            }
            else if (hpRatio > 0.7d)
            {
                shaping.RestMismatchPenalty = EnvRewardRestHighHpMismatchPenalty;
            }

            return;
        }

        if (optionType.Contains("Smith", StringComparison.OrdinalIgnoreCase))
        {
            if (hpRatio > 0.7d)
            {
                shaping.SmithBonus = EnvRewardSmithHealthyBonus;
            }
            else if (hpRatio < 0.5d)
            {
                shaping.SmithMismatchPenalty = EnvRewardSmithLowHpMismatchPenalty;
            }
        }
    }

    private static double ScoreEnvCardHeuristic(JsonElement? cardElement, BridgeEnvSnapshot snapshot)
    {
        if (cardElement is null || cardElement.Value.ValueKind is JsonValueKind.Null or JsonValueKind.Undefined)
        {
            return 0d;
        }

        var type = TryGetNestedString(cardElement.Value, "type") ?? string.Empty;
        var cost = TryGetNestedInt(cardElement.Value, "resolved_energy_cost") ??
                   TryGetNestedInt(cardElement.Value, "cost") ??
                   0;
        var damage = TryGetNestedInt(cardElement.Value, "effect_preview", "total_damage") ??
                     TryExtractEnvMetric(cardElement.Value, "damage");
        var block = TryGetNestedInt(cardElement.Value, "effect_preview", "total_block") ??
                    TryExtractEnvMetric(cardElement.Value, "block");
        var draw = TryGetNestedInt(cardElement.Value, "effect_preview", "draw") ??
                   TryExtractEnvMetric(cardElement.Value, "draw");
        var weak = TryGetNestedInt(cardElement.Value, "effect_preview", "weak") ??
                   TryExtractEnvMetric(cardElement.Value, "weak");
        var vulnerable = TryGetNestedInt(cardElement.Value, "effect_preview", "vulnerable") ??
                         TryExtractEnvMetric(cardElement.Value, "vulnerable");
        var summon = TryGetNestedInt(cardElement.Value, "effect_preview", "summon") ??
                     TryExtractEnvMetric(cardElement.Value, "summon");
        var summary = (TryGetNestedString(cardElement.Value, "effect_preview", "summary") ??
                       TryGetNestedString(cardElement.Value, "effect") ??
                       TryGetNestedString(cardElement.Value, "description") ??
                       string.Empty)
            .ToLowerInvariant();

        var score = 0d;

        if (string.Equals(type, "Attack", StringComparison.OrdinalIgnoreCase))
        {
            if (cost <= 1 && damage >= 8)
            {
                score += 0.04d;
            }
            else if (cost <= 1 && damage >= 6)
            {
                score += 0.02d;
            }
            else if (cost >= 2 && damage > 0 && damage < cost * 7)
            {
                score -= 0.04d;
            }
        }

        if (string.Equals(type, "Skill", StringComparison.OrdinalIgnoreCase))
        {
            if (cost <= 1 && block >= 7)
            {
                score += 0.03d;
            }
            else if (cost <= 1 && block >= 5)
            {
                score += 0.02d;
            }
            else if (cost >= 2 && block > 0 && block < cost * 6)
            {
                score -= 0.03d;
            }
        }

        if (cost == 0 && (damage > 0 || block > 0 || draw > 0))
        {
            score += 0.02d;
        }

        if (draw >= 2)
        {
            score += 0.04d;
        }
        else if (draw == 1)
        {
            score += 0.02d;
        }

        if (weak > 0)
        {
            score += 0.02d;
        }

        if (vulnerable > 0)
        {
            score += 0.02d;
        }

        if (summon >= 5)
        {
            score += 0.02d;
        }

        if (string.Equals(type, "Power", StringComparison.OrdinalIgnoreCase) &&
            snapshot.ActIndex <= 0 &&
            snapshot.TotalFloor <= 8)
        {
            score -= 0.04d;
        }

        if (cost >= 3)
        {
            score -= 0.03d;
        }

        if (snapshot.DeckCount >= 20)
        {
            score -= 0.02d;
        }

        if (damage <= 0 &&
            block <= 0 &&
            draw <= 0 &&
            weak <= 0 &&
            vulnerable <= 0 &&
            summon <= 0 &&
            cost >= 1 &&
            !summary.Contains("energy", StringComparison.Ordinal) &&
            !summary.Contains("能量", StringComparison.Ordinal))
        {
            score -= 0.05d;
        }

        return RoundEnvNumber(Math.Clamp(score, -EnvRewardCardHeuristicLimit, EnvRewardCardHeuristicLimit));
    }

    private static int TryExtractEnvMetric(JsonElement cardElement, string metric)
    {
        var text = TryGetNestedString(cardElement, "effect_preview", "summary") ??
                   TryGetNestedString(cardElement, "effect") ??
                   TryGetNestedString(cardElement, "description") ??
                   string.Empty;
        if (string.IsNullOrWhiteSpace(text))
        {
            return 0;
        }

        var lower = text.ToLowerInvariant();
        return metric switch
        {
            "damage" => TryExtractEnvMetricWithPatterns(lower, text, @"(\d+)\s*damage", @"造成(\d+)点伤害"),
            "block" => TryExtractEnvMetricWithPatterns(lower, text, @"(\d+)\s*block", @"获得(\d+)点格挡"),
            "draw" => TryExtractEnvMetricWithPatterns(lower, text, @"draw\s*(\d+)", @"抽(\d+)张牌"),
            "weak" => TryExtractEnvMetricWithPatterns(lower, text, @"(\d+)\s*weak", @"给予(\d+)层虚弱"),
            "vulnerable" => TryExtractEnvMetricWithPatterns(lower, text, @"(\d+)\s*vulnerable", @"给予(\d+)层易伤"),
            "summon" => TryExtractEnvMetricWithPatterns(lower, text, @"summon\s*(\d+)", @"召唤(\d+)"),
            _ => 0
        };
    }

    private static int TryExtractEnvMetricWithPatterns(string normalized, string original, string englishPattern, string chinesePattern)
    {
        foreach (var pattern in new[] { englishPattern, chinesePattern })
        {
            var input = pattern == englishPattern ? normalized : original;
            var match = Regex.Match(input, pattern, RegexOptions.IgnoreCase);
            if (match.Success && int.TryParse(match.Groups[1].Value, out var value))
            {
                return value;
            }
        }

        return 0;
    }

    private static bool IsEnvSelectionLikeAction(BridgeResolvedActionSelection action)
    {
        return action.Kind is "card_selection" or "deck_upgrade" or "character_select" or "run_mode_selection";
    }

    private static double RoundEnvNumber(double value)
    {
        return Math.Round(value, 6, MidpointRounding.AwayFromZero);
    }

    private static Player? GetPrimaryPlayer(BridgeWorldContext context)
    {
        if (context.RunState?.Players.Count > 0)
        {
            return context.RunState.Players[0];
        }

        if (context.CombatState?.Players.Count > 0)
        {
            return context.CombatState.Players[0];
        }

        return null;
    }

    private static int GetPrimaryPlayerCurrentHp(BridgeWorldContext context) => GetPrimaryPlayer(context)?.Creature?.CurrentHp ?? 0;
    private static int GetPrimaryPlayerMaxHp(BridgeWorldContext context) => GetPrimaryPlayer(context)?.Creature?.MaxHp ?? 0;
    private static int GetPrimaryPlayerGold(BridgeWorldContext context) => GetPrimaryPlayer(context)?.Gold ?? 0;
    private static int GetPrimaryPlayerRelicCount(BridgeWorldContext context) => GetPrimaryPlayer(context)?.Relics.Count ?? 0;
    private static int GetPrimaryPlayerDeckCount(BridgeWorldContext context) => GetPrimaryPlayer(context)?.Deck?.Cards.Count ?? 0;

    private sealed class BridgeEnvEpisode
    {
        public required string Id { get; init; }
        public int StepIndex { get; set; }
        public bool Done { get; set; }
        public string? RequestedCharacter { get; init; }
        public bool DefensiveBuffs { get; init; }
        public string EpisodeMode { get; init; } = "full_run";
        public string? EncounterId { get; init; }
        public bool RoomAnchorInitialized { get; set; }
        public string RoomKey { get; set; } = string.Empty;
        public int RoomStartHp { get; set; }
        public int RoomStartMaxHp { get; set; }
        public int RoomStartFloor { get; set; }
        public int RoomStartActIndex { get; set; }
    }

    private sealed class BridgeEnvSnapshot
    {
        public required BridgeWorldContext Context { get; init; }
        public required string Screen { get; init; }
        public required string Phase { get; init; }
        public required object Observation { get; init; }
        public required object RunSummary { get; init; }
        public required object[] LegalActions { get; init; }
        public required IReadOnlyDictionary<string, BridgeResolvedAction> ActionLookup { get; init; }
        public required IReadOnlyList<BridgeResolvedAction> ResolvedActions { get; init; }
        public required string LogicHash { get; init; }
        public required bool Actionable { get; init; }
        public required bool Done { get; init; }
        public required int CurrentHp { get; init; }
        public required int MaxHp { get; init; }
        public required int PlayerBlock { get; init; }
        public required int CurrentEnergy { get; init; }
        public required int Gold { get; init; }
        public required int ActIndex { get; init; }
        public required int TotalFloor { get; init; }
        public string? RoomType { get; init; }
        public string? RoomModelId { get; init; }
        public required int RelicCount { get; init; }
        public required int PotionCount { get; init; }
        public required int DeckCount { get; init; }
        public required IReadOnlyList<BridgeEnvDeckEntry> DeckEntries { get; init; }
        public required IReadOnlyList<BridgeEnvEnemyState> EnemyStates { get; init; }
        public required bool CombatInProgress { get; init; }
        public required bool RoomPreFinished { get; init; }
        public bool RunActive => Context.RunState is not null && Context.RunState.IsGameOver != true;
    }

    private sealed class BridgeEnvEnemyState
    {
        public required uint CombatId { get; init; }
        public required int CurrentHp { get; init; }
        public required int Block { get; init; }
        public required bool IsAlive { get; init; }
        public required int IntentDamageToPlayer { get; init; }
        public required int Weak { get; init; }
        public required int Vulnerable { get; init; }
    }

    private sealed class BridgeResolvedActionSelection
    {
        public required BridgeResolvedAction Action { get; init; }
        public required int Index { get; init; }
        public required string Kind { get; init; }
    }

    private sealed class BridgeEnvRewardBreakdown
    {
        [JsonPropertyName("hp_loss_normalized")]
        public required double HpLossNormalized { get; init; }

        [JsonPropertyName("hp_gain_normalized")]
        public required double HpGainNormalized { get; init; }

        [JsonPropertyName("room_complete")]
        public required int RoomComplete { get; init; }

        [JsonPropertyName("room_hp_delta_normalized")]
        public required double RoomHpDeltaNormalized { get; init; }

        [JsonPropertyName("floor_delta")]
        public required int FloorDelta { get; init; }

        [JsonPropertyName("death")]
        public required int Death { get; init; }

        [JsonPropertyName("victory")]
        public required int Victory { get; init; }

        [JsonPropertyName("action_error_penalty")]
        public required double ActionErrorPenalty { get; init; }

        [JsonPropertyName("truncated_penalty")]
        public required double TruncatedPenalty { get; init; }

        [JsonPropertyName("total")]
        public required double Total { get; init; }
    }

    private sealed class BridgeEnvDeckEntry
    {
        public required string Ref { get; init; }
        public string? CardId { get; init; }
        public string? Title { get; init; }
        public string? Rarity { get; init; }
        public required string Signature { get; init; }
        public required bool IsStarter { get; init; }
    }

    private sealed class BridgeEnvDeckDiff
    {
        public required int CardAddCount { get; init; }
        public required int StarterCardRemoveCount { get; init; }
        public required int OtherCardRemoveCount { get; init; }
        public required int CardUpgradeCount { get; init; }
    }

    private sealed class BridgeEnvActionShaping
    {
        public static BridgeEnvActionShaping None => new();

        public double CardChoiceBonus { get; set; }
        public double SkipBadCardsBonus { get; set; }
        public double RestBonus { get; set; }
        public double RestMismatchPenalty { get; set; }
        public double SmithBonus { get; set; }
        public double SmithMismatchPenalty { get; set; }
        public double PlayCardBonus { get; set; }
        public int ThreatGapBefore { get; set; }
        public int ThreatGapAfter { get; set; }
        public int ThreatGapReduction { get; set; }
        public double ThreatGapReductionNormalized { get; set; }
        public double ThreatGapReductionBonus { get; set; }
        public int EffectiveBlockAdded { get; set; }
        public double EffectiveBlockNormalized { get; set; }
        public double EffectiveBlockBonus { get; set; }
        public int WastedBlockAdded { get; set; }
        public double WastedBlockNormalized { get; set; }
        public double WastedBlockPenalty { get; set; }
        public int WeakIntentReduction { get; set; }
        public double WeakIntentReductionNormalized { get; set; }
        public double WeakBonus { get; set; }
        public int VulnerableRealizedDamage { get; set; }
        public double VulnerableRealizedDamageNormalized { get; set; }
        public double VulnerableBonus { get; set; }
        public double EndTurnWastePenalty { get; set; }
        public double MissedDefensePenalty { get; set; }
    }
}
