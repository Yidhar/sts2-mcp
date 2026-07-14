using System.Collections.Generic;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using Godot;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Runs;

namespace Sts2McpBridge.Scripts;

internal static partial class BridgeGameApi
{
    private static async Task<BridgeEnvSnapshot> CaptureEnvSnapshotAsync(
        int timeoutMs,
        CancellationToken cancellationToken,
        string operationName = "env.capture_snapshot")
    {
        return await RunOnMainThreadGuardedAsync(
            CaptureEnvSnapshot,
            operationName,
            timeoutMs,
            cancellationToken);
    }

    private static BridgeEnvSnapshot CaptureEnvSnapshot()
    {
        var context = CaptureContext();
        var actions = FilterEnvResolvedActions(context, BuildResolvedActions(context));
        var phase = ResolveEnvPhase(context, actions);
        var runSummary = BuildEnvRunPayload(context.RunState);
        var observationCore = BuildEnvObservationCore(context, phase);
        var legalActions = BuildEnvLegalActions(context, actions);
        var surfaceFingerprint = BuildEnvSurfaceFingerprint(context, phase);
        var logicHash = ComputeStateHash(new
        {
            phase,
            observation = observationCore,
            action_ids = legalActions.Select(static action => action.ActionId).ToArray(),
            surface_fingerprint = surfaceFingerprint
        });
        var observation = CloneDictionary(observationCore);
        observation["logic_hash"] = logicHash;
        var done = context.RunState?.IsGameOver == true;
        var actionable = legalActions.Count > 0 && phase != "settling";

        return new BridgeEnvSnapshot
        {
            Context = context,
            Screen = context.Screen,
            Phase = phase,
            Observation = observation,
            RunSummary = runSummary,
            LegalActions = legalActions.Select((action, index) => BuildEnvActionPayload(action, index)).ToArray(),
            ActionLookup = BuildEnvActionLookup(legalActions),
            ResolvedActions = legalActions,
            LogicHash = logicHash,
            SurfaceFingerprint = surfaceFingerprint,
            Actionable = actionable,
            Done = done,
            CurrentHp = GetPrimaryPlayerCurrentHp(context),
            MaxHp = GetPrimaryPlayerMaxHp(context),
            PlayerBlock = GetPrimaryPlayerCurrentBlock(context),
            CurrentEnergy = GetPrimaryPlayerCurrentEnergy(context),
            Gold = GetPrimaryPlayerGold(context),
            ActIndex = context.RunState?.CurrentActIndex ?? 0,
            TotalFloor = context.RunState?.TotalFloor ?? 0,
            RoomType = context.RunState?.CurrentRoom?.RoomType.ToString(),
            RoomModelId = context.RunState?.CurrentRoom?.ModelId?.ToString(),
            RelicCount = GetPrimaryPlayerRelicCount(context),
            PotionCount = GetPrimaryPlayerPotionCount(context),
            DeckCount = GetPrimaryPlayerDeckCount(context),
            DeckEntries = BuildEnvDeckEntries(context),
            EnemyStates = BuildEnvEnemyStates(context),
            CombatInProgress = context.CombatManager?.IsInProgress == true,
            RoomPreFinished = context.RunState?.CurrentRoom?.IsPreFinished == true
        };
    }

    private static string BuildEnvSurfaceFingerprint(BridgeWorldContext context, string phase)
    {
        if (!string.Equals(phase, "card_selection", StringComparison.Ordinal) &&
            !string.Equals(phase, "deck_upgrade", StringComparison.Ordinal))
        {
            return string.Empty;
        }

        var builder = new StringBuilder(128);
        switch (phase)
        {
            case "card_selection":
                AppendCardSelectionStateFingerprint(builder, context);
                break;
            case "deck_upgrade":
                AppendDeckUpgradeStateFingerprint(builder, context);
                break;
        }

        return builder.ToString();
    }

    private static Dictionary<string, BridgeResolvedAction> BuildEnvActionLookup(
        IReadOnlyList<BridgeResolvedAction> legalActions)
    {
        // The lookup is used by /env/step to resolve a requested action_id. We
        // must not silently drop later actions whose id collides with an
        // earlier one (the old GroupBy+First form did this), because Python
        // picks by index into the full LegalActions list and the bridge then
        // executes via this lookup �?a collision would mean the caller's
        // chosen entry silently gets replaced by the first entry with the
        // same id. In normal play this cannot happen (card_ref is an identity
        // hash and target suffixes are unique per target), so a collision
        // indicates a game-state anomaly we want loud.
        var lookup = new Dictionary<string, BridgeResolvedAction>(
            legalActions.Count,
            StringComparer.Ordinal);
        for (var i = 0; i < legalActions.Count; i++)
        {
            var action = legalActions[i];
            if (!lookup.TryAdd(action.ActionId, action))
            {
                BridgeDebugTrace.Write(
                    $"[env.actions] duplicate action_id detected: '{action.ActionId}' " +
                    $"at indices {IndexOfActionId(legalActions, action.ActionId)} and {i} " +
                    $"(legal_actions count={legalActions.Count}). Keeping first; later duplicate is unreachable via action_id lookup.");
            }
        }
        return lookup;
    }

    private static int IndexOfActionId(
        IReadOnlyList<BridgeResolvedAction> legalActions,
        string actionId)
    {
        for (var i = 0; i < legalActions.Count; i++)
        {
            if (string.Equals(legalActions[i].ActionId, actionId, StringComparison.Ordinal))
            {
                return i;
            }
        }
        return -1;
    }

    private static List<BridgeResolvedAction> FilterEnvResolvedActions(
        BridgeWorldContext context,
        IReadOnlyList<BridgeResolvedAction> actions)
    {
        var filtered = actions
            .Where(static action => !action.ActionId.StartsWith("automation:", StringComparison.Ordinal))
            .ToList();

        filtered = FilterActionsForStableSurface(context, filtered);

        if (!ShouldSuppressTransientEventContinue(context, filtered))
        {
            return filtered;
        }

        return filtered
            .Where(static action => !action.ActionId.StartsWith("event_option:", StringComparison.Ordinal))
            .ToList();
    }

    private static List<BridgeResolvedAction> FilterActionsForStableSurface(
        BridgeWorldContext context,
        IReadOnlyList<BridgeResolvedAction> actions)
    {
        var cardSelectionVisible = IsCardSelectionVisible(context);
        var deckUpgradeVisible = IsDeckUpgradeSelectionVisible(context);
        var cardRewardVisible = IsCardRewardSelectionVisible(context.CardRewardScreen, context.CardRewardOptions);
        var rewardsVisible = IsRewardsScreenVisible(
            context.RewardsScreen,
            context.ProceedButton,
            context.RewardProceedButton,
            context.MapScreen,
            context.RewardButtons);
        var crystalSphereVisible = context.CrystalSphereScreen is not null && IsNodeVisible(context.CrystalSphereScreen);
        var mapVisible = IsInteractiveMapSurface(context);
        var merchantVisible = context.MerchantRoom is not null &&
                              (IsNodeVisible(context.MerchantRoom) || context.MerchantInventory?.IsOpen == true);
        var restSiteVisible = context.RestSiteRoom is not null &&
                              !mapVisible &&
                              IsNodeVisible(context.RestSiteRoom);
        var treasureVisible = context.TreasureRoom is not null && IsNodeVisible(context.TreasureRoom);
        var eventVisible = context.EventRoom is not null &&
                           IsNodeVisible(context.EventRoom) &&
                           !mapVisible;

        bool IsAllowed(BridgeResolvedAction action)
        {
            var actionId = action.ActionId;
            if (cardSelectionVisible)
            {
                return actionId.StartsWith("card_selection:", StringComparison.Ordinal);
            }

            if (deckUpgradeVisible)
            {
                return actionId.StartsWith("deck_upgrade:", StringComparison.Ordinal);
            }

            if (cardRewardVisible)
            {
                return actionId.StartsWith("card_reward:", StringComparison.Ordinal);
            }

            if (rewardsVisible)
            {
                return actionId.StartsWith("reward:", StringComparison.Ordinal) ||
                       actionId.Equals("proceed", StringComparison.Ordinal);
            }

            if (crystalSphereVisible)
            {
                return actionId.StartsWith("crystal_sphere:", StringComparison.Ordinal);
            }

            if (context.CombatManager?.IsInProgress == true &&
                !cardSelectionVisible)
            {
                return actionId.StartsWith("play_card:", StringComparison.Ordinal) ||
                       actionId.StartsWith("use_potion:", StringComparison.Ordinal) ||
                       actionId.StartsWith("discard_potion:", StringComparison.Ordinal) ||
                       actionId.Equals("end_turn", StringComparison.Ordinal);
            }

            if (mapVisible)
            {
                return actionId.StartsWith("map:", StringComparison.Ordinal);
            }

            if (eventVisible)
            {
                return actionId.StartsWith("event_option:", StringComparison.Ordinal);
            }

            if (merchantVisible)
            {
                return actionId.StartsWith("shop:", StringComparison.Ordinal);
            }

            if (restSiteVisible)
            {
                return actionId.StartsWith("rest_site:", StringComparison.Ordinal);
            }

            if (treasureVisible)
            {
                return actionId.StartsWith("treasure:", StringComparison.Ordinal) ||
                       actionId.StartsWith("treasure_relic:", StringComparison.Ordinal) ||
                       actionId.Equals("proceed", StringComparison.Ordinal);
            }

            return context.Screen switch
            {
                "RUN_MODE_SELECTION" => actionId.StartsWith("run_mode:", StringComparison.Ordinal),
                "CHARACTER_SELECT" => actionId.StartsWith("character_select:", StringComparison.Ordinal) ||
                                      actionId.Equals("embark", StringComparison.Ordinal),
                "MAIN_MENU" => actionId.StartsWith("main_menu:", StringComparison.Ordinal),
                "ABANDON_RUN_CONFIRM" => actionId.StartsWith("main_menu:confirm_abandon_run", StringComparison.Ordinal) ||
                                         actionId.StartsWith("main_menu:cancel_abandon_run", StringComparison.Ordinal) ||
                                         actionId.StartsWith("main_menu:abandon_confirm:", StringComparison.Ordinal),
                "GAME_OVER" => actionId.StartsWith("game_over:", StringComparison.Ordinal),
                _ => true
            };
        }

        var filtered = actions.Where(IsAllowed).ToList();
        if (ShouldSuppressTransientCombatEndTurnOnly(context, filtered))
        {
            BridgeDebugTrace.Write("stable_surface suppress transient combat end_turn_only");
            return filtered
                .Where(static action => !action.ActionId.Equals("end_turn", StringComparison.Ordinal))
                .ToList();
        }
        ResetTransientCombatEndTurnOnlyGateIfNeeded(context, filtered);
        return filtered.Count > 0 ? filtered : actions.ToList();
    }

    // Do not time-suppress real end_turn-only combat states. Transient action
    // resolution is gated by direct CombatManager flags in BuildResolvedActions
    // (IsPlayPhase && !PlayerActionsDisabled && !IsPaused) and by the fast
    // direct-state frontier wait in BridgeGameApi.cs. If those flags say the
    // player can act and only end_turn remains, exposing end_turn is correct.
    private static bool ShouldSuppressTransientCombatEndTurnOnly(
        BridgeWorldContext context,
        IReadOnlyList<BridgeResolvedAction> filtered)
    {
        ResetTransientCombatEndTurnOnlyGate();
        return false;
    }

    private static void ResetTransientCombatEndTurnOnlyGateIfNeeded(
        BridgeWorldContext context,
        IReadOnlyList<BridgeResolvedAction> filtered)
    {
        if (context.CombatManager?.IsInProgress == true &&
            filtered.Count == 1 &&
            filtered[0].ActionId.Equals("end_turn", StringComparison.Ordinal))
        {
            return;
        }
        ResetTransientCombatEndTurnOnlyGate();
    }

    private static void ResetTransientCombatEndTurnOnlyGate()
    {
        // Historical timer gate removed; keep this no-op so older reset call
        // sites remain harmless and the filter path stays simple.
    }

    private static bool ShouldSuppressTransientEventContinue(
        BridgeWorldContext context,
        IReadOnlyList<BridgeResolvedAction> actions)
    {
        if (context.RunState?.CurrentRoom?.RoomType.ToString() == "Event")
        {
            return false;
        }

        if (actions.Count != 1 || !actions[0].ActionId.StartsWith("event_option:", StringComparison.Ordinal))
        {
            return false;
        }

        var payload = JsonSerializer.SerializeToElement(actions[0].Payload);
        return TryGetNestedBool(payload, "option", "is_proceed") == true;
    }

    private static string ResolveEnvPhase(
        BridgeWorldContext context,
        IReadOnlyList<BridgeResolvedAction> actions)
    {
        var hasActiveRunContext = context.RunState is not null && context.RunState.IsGameOver != true;
        var hasRunActions = actions.Any(static action =>
            !action.ActionId.StartsWith("main_menu:", StringComparison.Ordinal) &&
            !action.ActionId.StartsWith("run_mode:", StringComparison.Ordinal) &&
            !action.ActionId.StartsWith("character_select:", StringComparison.Ordinal) &&
            !action.ActionId.StartsWith("game_over:", StringComparison.Ordinal) &&
            !action.ActionId.StartsWith("automation:", StringComparison.Ordinal) &&
            !action.ActionId.Equals("embark", StringComparison.Ordinal));
        var allowStartupSurface = !hasActiveRunContext && !hasRunActions;

        if (context.RunState?.IsGameOver == true)
        {
            return "terminal";
        }

        if (allowStartupSurface &&
            IsRunModeSelectionVisible(context))
        {
            return "startup_run_mode";
        }

        if (context.CharacterSelectScreen is not null &&
            IsNodeVisible(context.CharacterSelectScreen) &&
            allowStartupSurface)
        {
            return "startup_character_select";
        }

        if (context.MainMenuRoot is not null &&
            IsNodeVisible(context.MainMenuRoot) &&
            allowStartupSurface)
        {
            return "startup_main_menu";
        }

        if (context.DeckUpgradeScreen is not null && IsNodeVisible(context.DeckUpgradeScreen))
        {
            return "deck_upgrade";
        }

        if (context.CardSelectionScreen is not null && IsNodeVisible(context.CardSelectionScreen))
        {
            return "card_selection";
        }

        if (IsCardRewardSelectionVisible(context.CardRewardScreen, context.CardRewardOptions))
        {
            return "card_reward";
        }

        if (IsRewardsScreenVisible(
                context.RewardsScreen,
                context.ProceedButton,
                context.RewardProceedButton,
                context.MapScreen,
                context.RewardButtons))
        {
            return "reward";
        }

        if (context.CrystalSphereScreen is not null && IsNodeVisible(context.CrystalSphereScreen))
        {
            return "event_crystal_sphere";
        }

        if (context.CombatManager?.IsInProgress == true)
        {
            var blockedByResidualMapOverlay =
                context.MapScreen is not null &&
                context.MapScreen.IsOpen &&
                !context.MapScreen.IsTraveling &&
                HasBlockingMapOverlaySurface(context, actions);

            return IsCombatPlayPhase(context.CombatManager, context.CombatState) &&
                   !context.CombatManager.IsPaused &&
                   !context.CombatManager.PlayerActionsDisabled &&
                   context.CardSelectionScreen is null &&
                   !blockedByResidualMapOverlay
                ? "combat"
                : "settling";
        }

        if (IsInteractiveMapSurface(context, actions))
        {
            return "map";
        }

        if ((context.EventRoom is not null && IsNodeVisible(context.EventRoom) && !IsInteractiveMapSurface(context.MapScreen)) ||
            (actions.Count > 0 && actions.All(static action => action.ActionId.StartsWith("event_option:", StringComparison.Ordinal))))
        {
            return "event";
        }

        if (context.MerchantRoom is not null &&
            (IsNodeVisible(context.MerchantRoom) || context.MerchantInventory?.IsOpen == true))
        {
            return "shop";
        }

        if (context.RestSiteRoom is not null &&
            !IsInteractiveMapSurface(context.MapScreen) &&
            IsNodeVisible(context.RestSiteRoom))
        {
            return "rest_site";
        }

        if (context.TreasureRoom is not null && IsNodeVisible(context.TreasureRoom))
        {
            return "treasure";
        }

        if (actions.Count > 0)
        {
            return "actions";
        }

        return "settling";
    }

    private static Dictionary<string, object?> BuildEnvObservationCore(BridgeWorldContext context, string phase)
    {
        var decisionDomain = ResolveEnvDecisionDomain(context, phase);
        var observation = new Dictionary<string, object?>
        {
            ["phase"] = phase,
            ["decision_domain"] = decisionDomain,
            ["run"] = BuildEnvRunPayload(context.RunState),
            ["player"] = BuildEnvPlayerPayload(context)
        };

        var combat = BuildEnvCombatPayload(context);
        if (combat is not null)
        {
            observation["combat"] = combat;
        }

        var eventState = BuildEnvEventPayload(context);
        if (eventState is not null)
        {
            observation["event"] = eventState;
        }

        var decision = BuildEnvDecisionPayload(context, phase);
        if (decision is not null)
        {
            observation["decision"] = decision;
        }

        return observation;
    }

    private static string ResolveEnvDecisionDomain(BridgeWorldContext context, string phase)
    {
        return phase switch
        {
            "combat" => "combat",
            "map" => "route",
            "card_selection" => context.CombatManager?.IsInProgress == true ? "combat" : "build",
            "settling" => context.CombatManager?.IsInProgress == true ? "combat" : "build",
            _ => "build"
        };
    }

    private static object BuildEnvRunPayload(RunState? runState, int? floorOverride = null)
    {
        return new
        {
            active = runState is not null,
            game_over = runState?.IsGameOver ?? false,
            act = runState?.Act is null ? null : TryGetTitle(runState.Act),
            act_id = runState?.Act?.Id.ToString(),
            act_floor = runState?.ActFloor,
            floor = floorOverride ?? runState?.TotalFloor,
            room_type = runState?.CurrentRoom?.RoomType.ToString(),
            room_model = runState?.CurrentRoom?.ModelId?.ToString(),
            coord = BuildMapCoord(runState?.CurrentMapCoord)
        };
    }

    private static object BuildEnvPlayerPayload(BridgeWorldContext context)
    {
        var player = GetPrimaryPlayer(context);
        var creature = player?.Creature;

        var relics = (player?.Relics ?? Enumerable.Empty<RelicModel>())
            .Select(BuildRelicPayload)
            .ToArray();

        var potions = (player?.PotionSlots ?? Enumerable.Empty<PotionModel?>())
            .Where(static potion => potion is not null)
            .Select(static potion => BuildPotionPayload(potion))
            .ToArray();

        return new
        {
            character_id = player?.Character?.Id.ToString(),
            character_title = player?.Character is null ? null : DescribeCharacter(player.Character),
            hp = creature?.CurrentHp,
            max_hp = creature?.MaxHp,
            block = creature?.Block,
            gold = player?.Gold,
            open_potion_slots = player?.PotionSlots.Count(static potion => potion is null) ?? 0,
            deck = player?.Deck?.Cards.Count ?? 0,
            deck_cards = player?.Deck?.Cards.Select(card => BuildEnvCardPayload(card, GetCardReference(card))).ToArray()
                ?? Array.Empty<object>(),
            powers = creature?.Powers.Select(BuildEnvPowerPayload).ToArray() ?? Array.Empty<object>(),
            relics,
            potions
        };
    }

    private static object? BuildEnvEventPayload(BridgeWorldContext context)
    {
        var eventModel = GetHiddenFieldValue(context.EventRoom, "_event") as EventModel;
        if (eventModel is null)
        {
            return null;
        }

        return new
        {
            event_id = eventModel.Id.ToString(),
            layout_type = eventModel.LayoutType.ToString(),
            is_deterministic = eventModel.IsDeterministic,
            is_shared = eventModel.IsShared,
            is_finished = eventModel.IsFinished,
            in_dialogue = !eventModel.IsFinished && eventModel.CurrentOptions.Count == 0,
            description_key = eventModel.Description?.LocEntryKey,
            encounter_id = eventModel.CanonicalEncounter?.Id.ToString(),
            dynamic_vars = BuildDynamicVarPayloads(eventModel.DynamicVars),
            options = eventModel.CurrentOptions.Select(static option => new
            {
                text_key = option.TextKey,
                title = DescribeText(option.Title, option),
                description = DescribeText(option.Description, option),
                is_locked = option.IsLocked,
                is_chosen = option.WasChosen,
                is_proceed = option.IsProceed
            }).ToArray()
        };
    }

    private static string SafeBuildEventDecisionText(BridgeWorldContext context)
    {
        try
        {
            var eventRoom = context.EventRoom;
            if (eventRoom is null) return $"event:{context.EventOptionButtons.Count} options";
            var eventTitle = TryGetTitle(eventRoom) ?? "";
            return $"event:{NormalizeSemanticText(eventTitle)}:{context.EventOptionButtons.Count} options";
        }
        catch
        {
            return $"event:{context.EventOptionButtons.Count} options";
        }
    }

    private static string SafeGetRelicDescription(RelicModel relic)
    {
        try { return DescribeRelicModelSafely(relic); }
        catch { return ""; }
    }

    private static string SafeGetPotionDescription(PotionModel potion)
    {
        try { return DescribePotionModelSafely(potion); }
        catch { return ""; }
    }

    private static object? BuildEnvCombatPayload(BridgeWorldContext context)
    {
        if (context.CombatManager is null || context.CombatState is null || !context.CombatManager.IsInProgress)
        {
            return null;
        }

        var player = GetPrimaryPlayer(context);
        var playerCombat = player?.PlayerCombatState;

        return new
        {
            round = context.CombatState.RoundNumber,
            side = context.CombatState.CurrentSide.ToString(),
            play_phase = IsCombatPlayPhase(context.CombatManager, context.CombatState),
            can_act = !context.CombatManager.PlayerActionsDisabled,
            energy = playerCombat?.Energy,
            max_energy = playerCombat?.MaxEnergy,
            stars = playerCombat?.Stars,
            hand = playerCombat?.Hand?.Cards.Select(card => BuildEnvCardPayload(card, GetCardReference(card))).ToArray()
                ?? Array.Empty<object>(),
            draw = playerCombat?.DrawPile?.Cards.Count ?? 0,
            discard = playerCombat?.DiscardPile?.Cards.Count ?? 0,
            exhaust = playerCombat?.ExhaustPile?.Cards.Count ?? 0,
            players = context.CombatState.PlayerCreatures
                .Select(BuildEnvCreaturePayload)
                .ToArray(),
            enemies = context.CombatState.Creatures
                .Where(static creature => creature.IsEnemy)
                .Select(BuildEnvEnemyPayload)
                .ToArray()
        };
    }

    private static object BuildEnvCreaturePayload(Creature creature)
    {
        return new
        {
            id = creature.CombatId,
            name = creature.Name,
            hp = creature.CurrentHp,
            max_hp = creature.MaxHp,
            block = creature.Block,
            powers = creature.Powers.Select(BuildEnvPowerPayload).ToArray()
        };
    }

    private static object BuildEnvEnemyPayload(Creature creature)
    {
        var intentEnvelope = JsonSerializer.SerializeToElement(BuildEnemyIntentPayload(creature));
        var monster = creature.Monster;
        var nextMove = monster?.NextMove;
        return new
        {
            id = creature.CombatId,
            combat_id = creature.CombatId,
            name = creature.Name,
            model_id = creature.ModelId.ToString(),
            side = creature.Side.ToString(),
            hp = creature.CurrentHp,
            max_hp = creature.MaxHp,
            block = creature.Block,
            is_alive = creature.IsAlive,
            is_hittable = SafeGetCreatureIsHittable(creature),
            is_primary_enemy = creature.IsPrimaryEnemy,
            is_secondary_enemy = creature.IsSecondaryEnemy,
            is_stunned = creature.IsStunned,
            is_pet = creature.IsPet,
            shows_infinite_hp = creature.ShowsInfiniteHp,
            can_receive_powers = creature.CanReceivePowers,
            slot_name = creature.SlotName,
            next_move_state_id = nextMove?.StateId,
            is_move = nextMove?.IsMove ?? false,
            must_perform_once_before_transitioning = nextMove?.MustPerformOnceBeforeTransitioning ?? false,
            can_transition_away = nextMove?.CanTransitionAway ?? false,
            is_performing_move = monster?.IsPerformingMove ?? false,
            spawned_this_turn = monster?.SpawnedThisTurn ?? false,
            intends_to_attack = monster?.IntendsToAttack ?? false,
            move_history = monster?.MoveStateMachine?.StateLog
                .Where(static state => state.ShouldAppearInLogs)
                .Select(static state => state.Id)
                .ToArray() ?? Array.Empty<string>(),
            intents = BuildEnvEnemyIntentPayloads(intentEnvelope),
            powers = creature.Powers.Select(BuildEnvPowerPayload).ToArray()
        };
    }

    private static object[] BuildEnvEnemyIntentPayloads(JsonElement intentEnvelope)
    {
        var intents = TryGetNestedElement(intentEnvelope, "intents");
        if (intents is null || intents.Value.ValueKind != JsonValueKind.Array)
        {
            return Array.Empty<object>();
        }

        return intents.Value.EnumerateArray()
            .Select(intent => (object)new
            {
                type = TryGetNestedString(intent, "intent_type"),
                damage = TryGetNestedInt(intent, "total_damage"),
                repeats = TryGetNestedInt(intent, "repeats")
            })
            .ToArray();
    }

    private static object BuildEnvPowerPayload(PowerModel power)
    {
        return BuildPowerPayload(power);
    }

    private static object BuildEnvCardPayload(CardModel card, string? cardRef = null)
    {
        var payload = JsonSerializer.SerializeToElement(BuildCardPayload(card));
        return CompactCardPayload(payload, cardRef) ?? new { missing = true };
    }

    private static object? BuildEnvDecisionPayload(BridgeWorldContext context, string phase)
    {
        return phase switch
        {
            "startup_character_select" => new
            {
                selected_index = ResolveSelectedCharacterIndex(context),
                option_count = context.CharacterButtons.Count
            },
            "reward" => new
            {
                reward_count = context.RewardButtons.Count,
                proceed_only = context.RewardButtons.Count == 0 &&
                               context.RewardProceedButton is not null &&
                               IsNodeVisible(context.RewardProceedButton),
                decision_text = $"reward choice: {context.RewardButtons.Count} claimable"
            },
            "card_reward" => new
            {
                option_count = context.CardRewardOptions.Count,
                can_skip = context.CardRewardSkipButton is not null &&
                           IsNodeVisible(context.CardRewardSkipButton) &&
                           IsButtonEnabled(context.CardRewardSkipButton),
                decision_text = $"card reward: {context.CardRewardOptions.Count} options"
            },
            "event" => new
            {
                option_count = context.EventOptionButtons.Count,
                decision_text = SafeBuildEventDecisionText(context)
            },
            "event_crystal_sphere" => new
            {
                divinations_left = GetCrystalSphereDivinationCount(GetCrystalSphereMinigame(context.CrystalSphereScreen)),
                current_tool = GetCrystalSphereToolName(GetCrystalSphereMinigame(context.CrystalSphereScreen)),
                cells = context.CrystalSphereCells.Select(static cell => new
                {
                    x = cell.Entity?.X,
                    y = cell.Entity?.Y,
                    hidden = cell.Entity?.IsHidden ?? true,
                    highlighted = cell.Entity?.IsHighlighted ?? false
                }).ToArray()
            },
            "rest_site" => new
            {
                option_count = context.RestSiteButtons.Count(static button => IsNodeVisible(button)),
                can_proceed = context.RestSiteProceedButton is not null &&
                              IsNodeVisible(context.RestSiteProceedButton) &&
                              IsButtonEnabled(context.RestSiteProceedButton),
                decision_text = "rest site: choose rest or smith"
            },
            "deck_upgrade" => BuildEnvDeckUpgradeDecisionPayload(context),
            "card_selection" => BuildEnvCardSelectionDecisionPayload(context),
            "shop" => new
            {
                is_open = context.MerchantInventory?.IsOpen ?? false,
                gold = context.MerchantInventory?.Inventory?.Player?.Gold,
                item_count = context.MerchantSlots.Count,
                decision_text = $"shop: {context.MerchantSlots.Count} items, gold {context.MerchantInventory?.Inventory?.Player?.Gold ?? 0}"
            },
            "treasure" => new
            {
                relic_option_count = context.TreasureRelicOptions.Count,
                can_open = CanOpenTreasureChest(context)
            },
            "map" => new
            {
                coord = BuildMapCoord(context.RunState?.CurrentMapCoord),
                travelable_count = context.MapPoints.Count(IsMapPointTravelable)
            },
            _ => null
        };
    }

    private static object BuildEnvDeckUpgradeDecisionPayload(BridgeWorldContext context)
    {
        var options = new List<object>();
        var selectedCount = CountSelectedDeckUpgradeCards(context.DeckUpgradeScreen);
        var useSingleSelection = GetHiddenPropertyValue<bool>(context.DeckUpgradeScreen, "UseSingleSelection") ?? false;
        var confirmReady = context.DeckUpgradeConfirmButton is not null &&
                           IsNodeVisible(context.DeckUpgradeConfirmButton) &&
                           IsButtonEnabled(context.DeckUpgradeConfirmButton);
        var prompt = TryGetDeckUpgradePrompt(context.DeckUpgradeScreen);
        var texts = CollectDeckUpgradeSurfaceTexts(
            context.DeckUpgradeScreen,
            prompt,
            context.DeckUpgradeConfirmButton,
            context.DeckUpgradeCancelButton,
            context.DeckUpgradeCloseButton);
        for (var index = 0; index < context.DeckUpgradeOptions.Count; index++)
        {
            var holder = context.DeckUpgradeOptions[index];
            if (holder.CardModel is null)
            {
                continue;
            }

            var preview = BuildCardUpgradePreviewPayload(holder.CardModel);
            options.Add(new
            {
                index,
                card = CompactCardPayload(JsonSerializer.SerializeToElement(BuildCardPayload(holder.CardModel))),
                upgrade_preview = preview is null ? null : CompactCardPayload(JsonSerializer.SerializeToElement(preview)),
                is_selected = IsDeckUpgradeCardSelected(context.DeckUpgradeScreen, holder.CardModel)
            });
        }

        return new
        {
            selected_count = selectedCount,
            use_single_selection = useSingleSelection,
            confirm_ready = confirmReady,
            prompt,
            texts,
            option_count = options.Count,
            options
        };
    }

    private static object BuildEnvCardSelectionDecisionPayload(BridgeWorldContext context)
    {
        var screen = context.CardSelectionScreen;
        var prefs = GetHiddenFieldValue(screen, "_prefs");
        var prompt = TryGetCardSelectionPrompt(screen);
        var texts = CollectCardSelectionSurfaceTexts(
            screen,
            prompt,
            context.CardSelectionConfirmButton,
            context.CardSelectionCancelButton,
            context.CardSelectionCloseButton,
            context.CardSelectionSkipButton);
        var selectedCount = CountSelectedCardSelectionCards(screen);
        var minSelect = GetHiddenPropertyValue<int>(prefs, "MinSelect");
        var maxSelect = GetHiddenPropertyValue<int>(prefs, "MaxSelect");
        var requiresManualConfirmation = GetHiddenPropertyValue<bool>(prefs, "RequireManualConfirmation");
        var cancelable = GetHiddenPropertyValue<bool>(prefs, "Cancelable");
        var confirmReady = context.CardSelectionConfirmButton is not null &&
                           IsNodeVisible(context.CardSelectionConfirmButton) &&
                           IsButtonEnabled(context.CardSelectionConfirmButton);
        var canSkip = context.CardSelectionSkipButton is not null &&
                      IsNodeVisible(context.CardSelectionSkipButton) &&
                      IsButtonEnabled(context.CardSelectionSkipButton);
        var remainingSelect = ResolveRemainingSelectCount(selectedCount, minSelect, maxSelect);
        var promptId = GetSelectionPromptId(prefs);
        var sourceZone = ResolveCardSelectionSourceZone(context);
        var operationType = InferSelectionOperationType(promptId);
        var destinationZone = InferSelectionDestinationZone(operationType, sourceZone);
        var options = context.CardSelectionOptions
            .Where(IsNodeVisible)
            .Select((holder, index) => new
            {
                index = GetCardSelectionOptionIndex(screen, holder, index),
                selection_id = GetCardSelectionOptionSelectionId(
                    screen,
                    holder,
                    GetCardSelectionOptionIndex(screen, holder, index)),
                card = CompactCardPayload(
                    JsonSerializer.SerializeToElement(BuildCardPayload(holder.CardModel))),
                is_selected = IsCardSelectionCardSelected(screen, holder.CardModel)
            })
            .ToArray();

        return new
        {
            screen_type = screen?.GetType().Name,
            prompt,
            prompt_id = promptId,
            operation_type = operationType,
            source_zone = sourceZone,
            destination_zone = destinationZone,
            texts,
            remaining_select = remainingSelect,
            selected_count = selectedCount,
            min_select = minSelect,
            max_select = maxSelect,
            requires_manual_confirmation = requiresManualConfirmation,
            cancelable = cancelable,
            confirm_ready = confirmReady,
            can_skip = canSkip,
            option_count = options.Length,
            options
        };
    }

    private static int? ResolveRemainingSelectCount(int selectedCount, int? minSelect, int? maxSelect)
    {
        var target = maxSelect ?? minSelect;
        if (target is null || target <= 0)
        {
            return null;
        }
        return Math.Max(target.Value - selectedCount, 0);
    }

    private static int? ResolveSelectedCharacterIndex(BridgeWorldContext context)
    {
        for (var index = 0; index < context.CharacterButtons.Count; index++)
        {
            if (ReferenceEquals(context.CharacterButtons[index], context.SelectedCharacterButton))
            {
                return index;
            }
        }

        return null;
    }

    private static bool HasOnlyProceedEventActions(IReadOnlyList<BridgeResolvedAction> actions)
    {
        var eventActions = actions
            .Where(static action => action.ActionId.StartsWith("event_option:", StringComparison.Ordinal))
            .ToList();
        if (eventActions.Count == 0)
        {
            return false;
        }

        return eventActions.All(static action =>
        {
            var payload = JsonSerializer.SerializeToElement(action.Payload);
            return TryGetNestedBool(payload, "option", "is_proceed") == true;
        });
    }

    private static bool ShouldIgnoreResidualEventOverlayOnInteractiveMap(
        BridgeWorldContext context,
        IReadOnlyList<BridgeResolvedAction> actions)
    {
        if (!IsInteractiveMapSurface(context.MapScreen))
        {
            return false;
        }

        if (context.RunState?.CurrentRoom?.RoomType.ToString() != "Event")
        {
            return false;
        }

        if (context.RunState.CurrentRoom.IsPreFinished == true)
        {
            return true;
        }

        return HasOnlyProceedEventActions(actions);
    }

    private static bool HasBlockingMapOverlaySurface(
        BridgeWorldContext context,
        IReadOnlyList<BridgeResolvedAction> actions)
    {
        var interactiveMapOpen = IsInteractiveMapSurface(context.MapScreen);
        var hasCombatSurface = context.CombatManager?.IsInProgress == true ||
                               (!interactiveMapOpen &&
                                context.CombatRoom is not null &&
                                IsNodeVisible(context.CombatRoom));
        var hasRestSiteSurface = !interactiveMapOpen &&
                                 context.RestSiteRoom is not null &&
                                 IsNodeVisible(context.RestSiteRoom);
        var hasMerchantSurface = context.MerchantInventory?.IsOpen == true ||
                                 (!interactiveMapOpen &&
                                  context.MerchantRoom is not null &&
                                  IsNodeVisible(context.MerchantRoom));
        var hasTreasureSurface = !interactiveMapOpen &&
                                 context.TreasureRoom is not null &&
                                 IsNodeVisible(context.TreasureRoom);
        var ignoreResidualEventOverlay = ShouldIgnoreResidualEventOverlayOnInteractiveMap(context, actions);
        var hasEventSurface = !ignoreResidualEventOverlay &&
                              ((context.EventRoom is not null && IsNodeVisible(context.EventRoom)) ||
                               actions.Any(static action => action.ActionId.StartsWith("event_option:", StringComparison.Ordinal)));
        var hasGenericProceedSurface = !ignoreResidualEventOverlay &&
                                       context.ProceedButton is not null &&
                                       IsNodeVisible(context.ProceedButton) &&
                                       IsButtonEnabled(context.ProceedButton) &&
                                       !ShouldSuppressGenericRoomProceed(context);

        return hasCombatSurface ||
               (context.DeckUpgradeScreen is not null && IsNodeVisible(context.DeckUpgradeScreen)) ||
               (context.CardSelectionScreen is not null && IsNodeVisible(context.CardSelectionScreen)) ||
               IsCardRewardSelectionVisible(context.CardRewardScreen, context.CardRewardOptions) ||
               IsRewardsScreenVisible(
                   context.RewardsScreen,
                   context.ProceedButton,
                   context.RewardProceedButton,
                   context.MapScreen,
                   context.RewardButtons) ||
               (context.CrystalSphereScreen is not null && IsNodeVisible(context.CrystalSphereScreen)) ||
               hasMerchantSurface ||
               hasRestSiteSurface ||
               hasTreasureSurface ||
               hasGenericProceedSurface ||
               hasEventSurface;
    }

    private static IReadOnlyList<BridgeResolvedAction> BuildEnvLegalActions(
        BridgeWorldContext context,
        IReadOnlyList<BridgeResolvedAction> actions)
    {
        return FilterActionsForStableSurface(context, actions);
    }
}
