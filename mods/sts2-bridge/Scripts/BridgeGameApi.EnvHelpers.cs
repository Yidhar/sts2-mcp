using System.Collections.Generic;
using System.Diagnostics;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Text.RegularExpressions;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.MonsterMoves.Intents;
using MegaCrit.Sts2.Core.Models.Powers;
using MegaCrit.Sts2.Core.Nodes;

namespace Sts2McpBridge.Scripts;

internal static partial class BridgeGameApi
{
    private const int EnvDefenseBlockAmount = 999;
    private const int EnvDefensePlatingAmount = 999;
    private const int EnvCardSelectionSelectFastFailTimeoutMs = 1000;

    private static object BuildEnvActionPayload(BridgeResolvedAction action, int index)
    {
        var payload = JsonSerializer.SerializeToElement(action.Payload);
        var kind = TryGetNestedString(payload, "kind") ?? InferEnvActionKind(action.ActionId);
        var modelActionKind = InferEnvModelActionKind(action.ActionId, kind);
        var entry = new Dictionary<string, object?>
        {
            ["idx"] = index,
            ["action_id"] = action.ActionId,
            ["kind"] = kind,
            ["model_action_kind"] = modelActionKind
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
                    name = TryGetNestedString(payload, "target_name"),
                    side = TryGetNestedString(payload, "target_side")
                };
                break;

            case "discard_potion":
                entry["slot"] = TryGetNestedInt(payload, "slot_index");
                entry["potion"] = CompactPotionPayload(TryGetNestedElement(payload, "potion"));
                break;

            case "reward":
                entry["reward"] = CompactRewardPayload(
                    TryGetNestedElement(payload, "reward"),
                    TryGetNestedInt(payload, "index"));
                break;

            case "card_reward":
                entry["selection"] = TryGetNestedString(payload, "selection_action") ?? "pick";
                entry["card"] = CompactCardPayload(TryGetNestedElement(payload, "card"));
                break;

            case "event_option":
                entry["index"] = TryGetNestedInt(payload, "index");
                entry["option"] = CompactEventOptionPayload(TryGetNestedElement(payload, "option"));
                entry["title"] = TryGetNestedString(payload, "option", "title");
                entry["description"] = TryGetNestedString(payload, "option", "description");
                entry["option_type"] = TryGetNestedString(payload, "option", "option_type");
                entry["proceed"] = TryGetNestedBool(payload, "option", "is_proceed");
                entry["coord"] = CompactCoordPayload(TryGetNestedElement(payload, "option", "coord"));
                break;

            case "map":
                entry["coord"] = CompactCoordPayload(TryGetNestedElement(payload, "coord"));
                entry["point_type"] = TryGetNestedString(payload, "point_type");
                entry["point_type_norm"] = TryGetNestedString(payload, "point_type_norm");
                break;

            case "rest_site":
                entry["option"] = CompactRestSiteOptionPayload(TryGetNestedElement(payload, "option"));
                break;

            case "deck_upgrade":
                entry["selection"] = TryGetNestedString(payload, "upgrade_action");
                entry["typed_selection"] = CompactSelectionPayload(TryGetNestedElement(payload, "typed_selection"));
                entry["index"] = TryGetNestedInt(payload, "index");
                entry["card"] = CompactCardPayload(TryGetNestedElement(payload, "card"));
                entry["upgrade_preview"] = CompactCardPayload(TryGetNestedElement(payload, "upgrade_preview"));
                break;

            case "card_selection":
                var selectionAction = TryGetNestedString(payload, "selection_action");
                var isSelected = TryGetNestedBool(payload, "is_selected");
                var selectionOperation = selectionAction switch
                {
                    "cancel" => "cancel_prompt",
                    "close" => "close",
                    "skip" => "skip",
                    _ => selectionAction
                };
                entry["selection"] = selectionAction;
                entry["selection_operation"] = selectionOperation;
                entry["model_action_variant"] = selectionOperation;
                entry["typed_selection"] = CompactSelectionPayload(TryGetNestedElement(payload, "typed_selection"));
                entry["index"] = TryGetNestedInt(payload, "index");
                entry["selection_id"] = TryGetNestedString(payload, "selection_id");
                entry["selection_prompt"] = TryGetNestedString(payload, "selection_prompt");
                entry["screen_type"] = TryGetNestedString(payload, "screen_type");
                entry["prompt_id"] = TryGetNestedString(payload, "prompt_id");
                entry["operation_type"] = TryGetNestedString(payload, "operation_type");
                entry["source_zone"] = TryGetNestedString(payload, "source_zone");
                entry["destination_zone"] = TryGetNestedString(payload, "destination_zone");
                entry["is_selected"] = isSelected;
                entry["selected_count"] = TryGetNestedInt(payload, "selected_count");
                entry["min_select"] = TryGetNestedInt(payload, "min_select");
                entry["max_select"] = TryGetNestedInt(payload, "max_select");
                entry["remaining_select"] = TryGetNestedInt(payload, "remaining_select");
                entry["confirm_ready"] = TryGetNestedBool(payload, "confirm_ready");
                entry["can_skip"] = TryGetNestedBool(payload, "can_skip");
                entry["requires_manual_confirmation"] = TryGetNestedBool(payload, "requires_manual_confirmation");
                entry["cancelable"] = TryGetNestedBool(payload, "cancelable");
                entry["selection_ready"] = TryGetNestedBool(payload, "selection_ready");
                entry["opened_age_ms"] = TryGetNestedInt(payload, "opened_age_ms");
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
                entry["run_mode_action"] = TryGetNestedString(payload, "run_mode_action");
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

    private static string InferEnvModelActionKind(string actionId, string transportKind)
    {
        if (actionId.Equals("end_turn", StringComparison.Ordinal))
        {
            return "end_turn";
        }

        var inferredKind = InferEnvActionKind(actionId);
        return inferredKind is "action" or "combat"
            ? transportKind
            : inferredKind;
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
        // Episode boundary is a game-quiescent window — drain finalizers here
        // so GodotObject disposal can't race the next combat's ObjectDB
        // mutations. combat_sandbox has its own call site at reset time;
        // full_run relies entirely on this hook plus the per-combat-end drain
        // in PerformActionResponseAsync.
        DrainManagedFinalizersLogged($"env.reset.{episode.EpisodeMode}");
        // DebugSeedOverride still holding the pin tells caller "yes your seed
        // was honored on the run-start we just completed" — useful for
        // training-loop asserts. Null means either no seed was requested or
        // the game already consumed + cleared the override.
        string? debugSeedOverrideField = null;
        try { debugSeedOverrideField = NGame.Instance?.DebugSeedOverride; }
        catch (Exception ex)
        {
            BridgeDebugTrace.Write(
                $"env_reset_debug_seed_read_failed episode={episode.Id}: {ex.GetBaseException().Message}");
        }
        var transitionFacts = BuildEnvResetTransitionFacts(episode, state);
        return new
        {
            ok = true,
            episode_id = episode.Id,
            step_index = episode.StepIndex,
            done = false,
            truncated = false,
            obs = state.Observation,
            legal_actions = state.LegalActions,
            transition = transitionFacts,
            transition_facts = transitionFacts,
            reward_authority = "none_on_reset",
            info = new
            {
                phase = state.Phase,
                screen = state.Screen,
                logic_hash = state.LogicHash,
                defensive_buffs = episode.DefensiveBuffs,
                episode_mode = episode.EpisodeMode,
                encounter_id = episode.EncounterId,
                debug_seed_override = debugSeedOverrideField,
                reset_actions = resetActions
            }
        };
    }

    private static object BuildEnvResetTransitionFacts(
        BridgeEnvEpisode episode,
        BridgeEnvSnapshot state)
    {
        return new
        {
            episode_id = episode.Id,
            step_index = episode.StepIndex,
            before_state_version = (long?)null,
            after_state_version = (long?)null,
            revision_status = "legacy-v1-unavailable",
            legacy_step_index_before = episode.StepIndex,
            legacy_step_index_after = episode.StepIndex,
            facts = new
            {
                hp_delta = 0,
                gold_delta = 0,
                floor_delta = 0,
                cards_added = Array.Empty<string>(),
                cards_removed = Array.Empty<string>(),
                potions_added = Array.Empty<string>(),
                potions_removed = Array.Empty<string>(),
                room_entered = state.RoomModelId ?? state.RoomType,
                combat_result = "none",
                terminal_reason = (string?)null,
                reset = true
            }
        };
    }

    private static object BuildEnvTransitionFacts(
        BridgeEnvEpisode episode,
        BridgeEnvSnapshot before,
        BridgeEnvSnapshot after,
        BridgeResolvedActionSelection? selectedAction,
        bool truncated,
        string? truncationReason,
        string? actionError,
        bool done)
    {
        var beforeRefs = before.DeckEntries
            .Select(static entry => entry.Ref)
            .ToHashSet(StringComparer.Ordinal);
        var afterRefs = after.DeckEntries
            .Select(static entry => entry.Ref)
            .ToHashSet(StringComparer.Ordinal);
        var cardsAdded = after.DeckEntries
            .Where(entry => !beforeRefs.Contains(entry.Ref))
            .Select(static entry => entry.CardId ?? entry.Title ?? entry.Ref)
            .ToArray();
        var cardsRemoved = before.DeckEntries
            .Where(entry => !afterRefs.Contains(entry.Ref))
            .Select(static entry => entry.CardId ?? entry.Title ?? entry.Ref)
            .ToArray();

        var combatResult = "none";
        if (done && after.CurrentHp <= 0)
        {
            combatResult = "defeat";
        }
        else if ((before.CombatInProgress && !after.CombatInProgress) ||
                 (done && after.CurrentHp > 0))
        {
            combatResult = "victory";
        }

        var terminalReason = !string.IsNullOrWhiteSpace(actionError)
            ? actionError
            : !string.IsNullOrWhiteSpace(truncationReason)
                ? truncationReason
                : done
                    ? combatResult
                    : null;
        var stepIndex = Math.Max(0, episode.StepIndex);
        var legacyStepIndexBefore = selectedAction is null ? stepIndex : Math.Max(0, stepIndex - 1);
        var potionDelta = after.PotionCount - before.PotionCount;
        return new
        {
            episode_id = episode.Id,
            step_index = stepIndex,
            before_state_version = (long?)null,
            after_state_version = (long?)null,
            revision_status = "legacy-v1-unavailable",
            legacy_step_index_before = legacyStepIndexBefore,
            legacy_step_index_after = stepIndex,
            facts = new
            {
                hp_delta = after.CurrentHp - before.CurrentHp,
                max_hp_delta = after.MaxHp - before.MaxHp,
                gold_delta = after.Gold - before.Gold,
                floor_delta = after.TotalFloor - before.TotalFloor,
                act_delta = after.ActIndex - before.ActIndex,
                relic_count_delta = after.RelicCount - before.RelicCount,
                potion_count_delta = potionDelta,
                cards_added = cardsAdded,
                cards_removed = cardsRemoved,
                potions_added = Array.Empty<string>(),
                potions_removed = Array.Empty<string>(),
                room_entered = HasEnvRoomTransition(before, after)
                    ? after.RoomModelId ?? after.RoomType
                    : null,
                combat_result = combatResult,
                terminal_reason = terminalReason,
                action_handle = selectedAction?.Action.ActionId,
                action_error = actionError,
                truncated
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
        string? actionError = null,
        bool forceDone = false,
        BridgeEnvStepTimingCollector? timing = null)
    {
        var cardSelectionBefore = BuildEnvCardSelectionStepInfoPayload(before);
        var cardSelectionAfter = BuildEnvCardSelectionStepInfoPayload(after);
        var actionability = BuildEnvActionabilityPayload(after, episode);
        var done = forceDone || after.Done;
        var transitionFacts = BuildEnvTransitionFacts(
            episode,
            before,
            after,
            selectedAction,
            truncated,
            truncationReason,
            actionError,
            done);
        return new
        {
            ok = true,
            episode_id = episode.Id,
            step_index = episode.StepIndex,
            reward = (double?)null,
            reward_semantics = "not-computed",
            reward_authority = "external-rl",
            transition = transitionFacts,
            transition_facts = transitionFacts,
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
                room_type_before = before.RoomType,
                room_type_after = after.RoomType,
                combat_in_progress_before = before.CombatInProgress,
                combat_in_progress_after = after.CombatInProgress,
                logic_hash_before = before.LogicHash,
                logic_hash_after = after.LogicHash,
                card_selection_before = cardSelectionBefore,
                card_selection_after = cardSelectionAfter,
                defensive_buffs = episode.DefensiveBuffs,
                episode_mode = episode.EpisodeMode,
                encounter_id = episode.EncounterId,
                truncation_reason = truncationReason,
                action_error = actionError,
                step_timing_ms = timing?.ToTimingPayload(),
                step_timing_counts = timing?.ToCountPayload(),
                actionability = actionability
            }
        };
    }

    /// <summary>
    /// TASK-C1: Expose enough state for Python to distinguish a transient
    /// only-end_turn frontier (animation/queue/draw-shuffle pending) from a
    /// genuinely stable no-action turn end.  Python short-polls on transient
    /// rather than long-sleeping in the bridge — see TASK-C2.
    /// </summary>
    // P0-2 actionability payload — additive enrichment of the C1 block.
    // The Python short-poll budget defaults to 100ms (matches the in-process
    // helper ``CombatSandboxEnv._fast_step_max_wait_ms``); operators can
    // override via the ``STS2_FAST_STEP_MAX_WAIT_MS`` env var on the
    // Python side.
    private const int DefaultActionabilityWaitBudgetMs = 100;

    private static object BuildEnvActionabilityPayload(BridgeEnvSnapshot snapshot, BridgeEnvEpisode episode)
    {
        var nonEndTurnCount = 0;
        if (snapshot.LegalActions != null)
        {
            foreach (var action in snapshot.LegalActions)
            {
                if (action is null)
                {
                    continue;
                }
                // Snapshot legal actions are anonymous payload objects; the
                // ActionId surfaces via the resolved-actions array.  Use that
                // for a robust Equals(string) comparison.
            }
        }
        if (snapshot.ResolvedActions != null)
        {
            foreach (var resolved in snapshot.ResolvedActions)
            {
                if (resolved is null)
                {
                    continue;
                }
                if (!string.Equals(resolved.ActionId, "end_turn", StringComparison.Ordinal))
                {
                    nonEndTurnCount++;
                }
            }
        }
        var totalActions = snapshot.ResolvedActions?.Count ?? snapshot.LegalActions?.Length ?? 0;
        var hasOnlyEndTurn = nonEndTurnCount == 0 && totalActions >= 1;
        var phaseSettling = string.Equals(snapshot.Phase, "settling", StringComparison.Ordinal);
        // P0-2: enumerate every direct-state pending reason we can resolve
        // from the snapshot.  Python's ``wait_for_stable_actionability`` /
        // transient-leak detector reads this list to label the leak source
        // without needing to re-derive from ``Phase`` alone.
        var pendingReasons = new List<string>();
        if (phaseSettling)
        {
            pendingReasons.Add("phase_settling");
            // The settling phase collapses queue / animation / draw-shuffle /
            // hand-not-ready signals.  Until the game state machine surfaces
            // them separately we emit the canonical names so downstream code
            // can already consume the schema and bridge-side refinements
            // come through with no Python change.
            pendingReasons.Add("animation_pending");
            pendingReasons.Add("queue_pending");
            if (snapshot.CombatInProgress)
            {
                pendingReasons.Add("draw_pending");
                pendingReasons.Add("hand_not_ready");
            }
        }
        var anyPending = pendingReasons.Count > 0;
        var transientOnlyEndTurn = hasOnlyEndTurn && anyPending && snapshot.CombatInProgress;
        var frontierStable = !anyPending && !transientOnlyEndTurn;
        string reason;
        if (nonEndTurnCount > 0)
        {
            reason = "has_non_end_turn_actions";
        }
        else if (transientOnlyEndTurn)
        {
            reason = phaseSettling ? "queue_pending" : "unknown";
        }
        else if (hasOnlyEndTurn)
        {
            reason = "stable_no_actions";
        }
        else
        {
            reason = "unknown";
        }
        return new
        {
            frontier_stable = frontierStable,
            transient_only_end_turn = transientOnlyEndTurn,
            only_end_turn_reason = reason,
            state_version = episode.StepIndex,
            state_hash = snapshot.LogicHash,
            queue_pending = anyPending,
            animation_pending = phaseSettling,
            draw_shuffle_pending = phaseSettling && snapshot.CombatInProgress,
            legal_non_end_turn_count = nonEndTurnCount,
            legal_action_count = totalActions,
            // P0-2 additions — direct-state pending reasons + wait budget.
            pending_reasons = pendingReasons,
            wait_budget_ms = DefaultActionabilityWaitBudgetMs,
            schema_version = 2
        };
    }

    private static bool IsCardSelectionSelectAction(BridgeResolvedActionSelection? selectedAction)
    {
        return selectedAction is not null &&
               selectedAction.Action.ActionId.StartsWith("card_selection:select:", StringComparison.Ordinal);
    }

    private static int GetCardSelectionSelectFastFailTimeoutMs(int timeoutMs)
    {
        return Math.Clamp(Math.Min(timeoutMs, EnvCardSelectionSelectFastFailTimeoutMs), 1, timeoutMs);
    }

    private static bool HasCardSelectionSelectionProgress(BridgeEnvSnapshot before, BridgeEnvSnapshot after)
    {
        if (HasMeaningfulEnvSnapshotDifference(before, after))
        {
            return true;
        }

        var beforeState = CaptureCardSelectionUiState(before.Context.CardSelectionScreen);
        var afterState = CaptureCardSelectionUiState(after.Context.CardSelectionScreen);
        return HasCardSelectionStateProgress(beforeState, afterState);
    }

    private static object BuildEnvCardSelectionStepInfoPayload(BridgeEnvSnapshot snapshot)
    {
        var state = CaptureCardSelectionUiState(snapshot.Context.CardSelectionScreen);
        return new
        {
            visible = state.Visible,
            screen_type = state.ScreenType,
            selected_count = state.SelectedCount,
            confirm_ready = state.ConfirmReady,
            min_select = state.MinSelect,
            max_select = state.MaxSelect,
            preview_visible = state.PreviewVisible,
            selection_ready = state.SelectionReady,
            opened_age_ms = state.OpenedAgeMs
        };
    }

    private sealed class BridgeEnvStepTimingCollector
    {
        public double BeforeSnapshotMs { get; set; }
        public double BeforeWaitMs { get; set; }
        public double ActionResolveMs { get; set; }
        public double ActionExecuteMs { get; set; }
        public double AfterWaitMs { get; set; }
        public double AutoConfirmMs { get; set; }
        public double EpisodeAdjustmentsMs { get; set; }
        public double PayloadBuildMs { get; set; }
        public double TotalMs { get; set; }
        public int SnapshotCalls { get; set; }
        public int WaitPumpCalls { get; set; }
        public int StableIterations { get; set; }

        public object ToTimingPayload()
        {
            return new
            {
                before_snapshot = BeforeSnapshotMs,
                before_wait = BeforeWaitMs,
                action_resolve = ActionResolveMs,
                action_execute = ActionExecuteMs,
                after_wait = AfterWaitMs,
                auto_confirm = AutoConfirmMs,
                episode_adjustments = EpisodeAdjustmentsMs,
                payload_build = PayloadBuildMs,
                total = TotalMs
            };
        }

        public object ToCountPayload()
        {
            return new
            {
                snapshot_calls = SnapshotCalls,
                wait_pump_calls = WaitPumpCalls,
                stable_iterations = StableIterations
            };
        }
    }

    private static async Task<BridgeEnvSnapshot> ApplyEnvEpisodeAdjustmentsAsync(
        BridgeEnvEpisode episode,
        BridgeEnvSnapshot snapshot,
        int timeoutMs,
        CancellationToken cancellationToken,
        BridgeEnvStepTimingCollector? timing = null)
    {
        if (episode.DefensiveBuffs && snapshot.RunActive)
        {
            var changed = await RunOnMainThreadGuardedAsync(
                () => ApplyEnvDefensiveBuffs(CaptureContext()),
                "env.apply_defensive_buffs",
                timeoutMs,
                cancellationToken);
            if (changed)
            {
                if (timing is not null)
                {
                    timing.WaitPumpCalls++;
                }
                await WaitForPumpTicksGuardedAsync(1, "env.apply_defensive_buffs.post_pump", timeoutMs, cancellationToken);
                if (timing is not null)
                {
                    timing.SnapshotCalls++;
                }
                snapshot = await CaptureEnvSnapshotAsync(timeoutMs, cancellationToken, "env.apply_defensive_buffs.snapshot");
            }
        }

        if (episode.EpisodeMode == "combat_sandbox")
        {
            snapshot = StabilizeCombatSandboxRunSnapshot(snapshot);
        }

        return snapshot;
    }

    private static BridgeEnvSnapshot StabilizeCombatSandboxRunSnapshot(BridgeEnvSnapshot snapshot)
    {
        var runState = snapshot.Context.RunState;
        if (runState is null)
        {
            return snapshot;
        }

        var stableFloor = runState.ActFloor > 0
            ? runState.ActFloor
            : snapshot.TotalFloor;
        if (stableFloor == snapshot.TotalFloor)
        {
            return snapshot;
        }

        if (snapshot.Observation is not Dictionary<string, object?> observation)
        {
            return snapshot;
        }

        var adjustedObservation = CloneDictionary(observation);
        adjustedObservation["run"] = BuildEnvRunPayload(runState, stableFloor);
        var adjustedLogicHash = ComputeStateHash(new
        {
            phase = snapshot.Phase,
            observation = adjustedObservation,
            action_ids = snapshot.ResolvedActions.Select(static action => action.ActionId).ToArray(),
            surface_fingerprint = snapshot.SurfaceFingerprint
        });

        return new BridgeEnvSnapshot
        {
            Context = snapshot.Context,
            Screen = snapshot.Screen,
            Phase = snapshot.Phase,
            Observation = adjustedObservation,
            RunSummary = BuildEnvRunPayload(runState, stableFloor),
            LegalActions = snapshot.LegalActions,
            ActionLookup = snapshot.ActionLookup,
            ResolvedActions = snapshot.ResolvedActions,
            LogicHash = adjustedLogicHash,
            SurfaceFingerprint = snapshot.SurfaceFingerprint,
            Actionable = snapshot.Actionable,
            Done = snapshot.Done,
            CurrentHp = snapshot.CurrentHp,
            MaxHp = snapshot.MaxHp,
            PlayerBlock = snapshot.PlayerBlock,
            CurrentEnergy = snapshot.CurrentEnergy,
            Gold = snapshot.Gold,
            ActIndex = snapshot.ActIndex,
            TotalFloor = stableFloor,
            RoomType = snapshot.RoomType,
            RoomModelId = snapshot.RoomModelId,
            RelicCount = snapshot.RelicCount,
            PotionCount = snapshot.PotionCount,
            DeckCount = snapshot.DeckCount,
            DeckEntries = snapshot.DeckEntries,
            EnemyStates = snapshot.EnemyStates,
            CombatInProgress = snapshot.CombatInProgress,
            RoomPreFinished = snapshot.RoomPreFinished
        };
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
                changed |=
                    TrySetHiddenPropertyValue(existing, nameof(PowerModel.Amount), amount) ||
                    TrySetHiddenFieldValue(existing, "_amount", amount);
            }

            if (existing.AmountOnTurnStart < amount)
            {
                changed |=
                    TrySetHiddenPropertyValue(existing, nameof(PowerModel.AmountOnTurnStart), amount) ||
                    TrySetHiddenFieldValue(existing, "_amountOnTurnStart", amount);
            }

            return changed;
        }

        var power = (TPower)ModelDb.Power<TPower>().ToMutable();
        power.Applier = creature;
        power.Target = creature;
        power.ApplyInternal(creature, amount, silent: true);
        TrySetHiddenPropertyValue(power, nameof(PowerModel.AmountOnTurnStart), amount);
        TrySetHiddenFieldValue(power, "_amountOnTurnStart", amount);
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

    private static bool TrySetHiddenFieldValue(object? target, string fieldName, object? value)
    {
        if (target is null)
        {
            return false;
        }

        var field = FindField(target.GetType(), fieldName);
        if (field is null)
        {
            return false;
        }

        field.SetValue(target, value);
        return true;
    }

}
