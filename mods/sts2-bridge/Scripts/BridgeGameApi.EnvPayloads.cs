using System.Collections.Generic;
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
        var legalActions = BuildEnvLegalActions(actions);
        var logicHash = ComputeStateHash(new
        {
            phase,
            observation = observationCore,
            action_ids = legalActions.Select(static action => action.ActionId).ToArray()
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
            ActionLookup = legalActions
                .GroupBy(static action => action.ActionId, StringComparer.Ordinal)
                .ToDictionary(static g => g.Key, static g => g.First(), StringComparer.Ordinal),
            ResolvedActions = legalActions,
            LogicHash = logicHash,
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

    private static List<BridgeResolvedAction> FilterEnvResolvedActions(
        BridgeWorldContext context,
        IReadOnlyList<BridgeResolvedAction> actions)
    {
        var filtered = actions
            .Where(static action => !action.ActionId.StartsWith("automation:", StringComparison.Ordinal))
            .ToList();

        if (!ShouldSuppressTransientEventContinue(context, filtered))
        {
            return filtered;
        }

        return filtered
            .Where(static action => !action.ActionId.StartsWith("event_option:", StringComparison.Ordinal))
            .ToList();
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
        if (context.RunState?.IsGameOver == true)
        {
            return "terminal";
        }

        if (IsRunModeSelectionVisible(context))
        {
            return "startup_run_mode";
        }

        if (context.CharacterSelectScreen is not null && IsNodeVisible(context.CharacterSelectScreen))
        {
            return "startup_character_select";
        }

        if (context.MainMenuRoot is not null && IsNodeVisible(context.MainMenuRoot))
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

        if (context.MapScreen?.IsOpen == true &&
            context.MapScreen.IsTravelEnabled &&
            !context.MapScreen.IsTraveling)
        {
            return "map";
        }

        if (context.CrystalSphereScreen is not null && IsNodeVisible(context.CrystalSphereScreen))
        {
            return "event_crystal_sphere";
        }

        if ((context.EventRoom is not null && IsNodeVisible(context.EventRoom) && context.MapScreen?.IsOpen != true) ||
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
            context.MapScreen?.IsOpen != true &&
            IsNodeVisible(context.RestSiteRoom))
        {
            return "rest_site";
        }

        if (context.TreasureRoom is not null && IsNodeVisible(context.TreasureRoom))
        {
            return "treasure";
        }

        if (context.CombatManager?.IsInProgress == true)
        {
            return context.CombatManager.IsPlayPhase &&
                   !context.CombatManager.PlayerActionsDisabled &&
                   context.CardSelectionScreen is null
                ? "combat"
                : "settling";
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

        // Relics as objects with canonical_text
        var relics = (player?.Relics ?? Enumerable.Empty<RelicModel>())
            .Select(relic =>
            {
                var title = TryGetTitle(relic);
                var desc = SafeGetRelicDescription(relic);
                var rarity = relic.Rarity.ToString();
                return new
                {
                    title,
                    canonical_text = BuildCanonicalRelicText(title, rarity, desc)
                };
            })
            .ToArray();

        // Potions as objects with canonical_text
        var potions = (player?.PotionSlots ?? Enumerable.Empty<PotionModel?>())
            .Select(slot =>
            {
                if (slot is null) return new { title = "[empty]", canonical_text = "" };
                var title = TryGetTitle(slot);
                var desc = SafeGetPotionDescription(slot);
                var target = slot.TargetType.ToString();
                var rarity = slot.Rarity.ToString();
                return new
                {
                    title,
                    canonical_text = BuildCanonicalPotionText(title, rarity, target, desc)
                };
            })
            .ToArray();

        return new
        {
            character_id = player?.Character?.Id.ToString(),
            character_title = player?.Character is null ? null : DescribeCharacter(player.Character),
            hp = creature?.CurrentHp,
            max_hp = creature?.MaxHp,
            block = creature?.Block,
            gold = player?.Gold,
            deck = player?.Deck?.Cards.Count ?? 0,
            deck_cards = player?.Deck?.Cards.Select(card => BuildEnvCardPayload(card, GetCardReference(card))).ToArray()
                ?? Array.Empty<object>(),
            relics,
            potions
        };
    }

    private static string SafeBuildEventDecisionText(BridgeWorldContext context)
    {
        try
        {
            var eventRoom = context.EventRoom;
            if (eventRoom is null) return $"事件｜{context.EventOptionButtons.Count}个选项";
            var eventTitle = TryGetTitle(eventRoom) ?? "";
            return $"事件｜{NormalizeSemanticText(eventTitle)}｜{context.EventOptionButtons.Count}个选项";
        }
        catch
        {
            return $"事件｜{context.EventOptionButtons.Count}个选项";
        }
    }

    private static string SafeGetRelicDescription(RelicModel relic)
    {
        try { return DescribeText(relic.Description, relic) ?? ""; }
        catch { return ""; }
    }

    private static string SafeGetPotionDescription(PotionModel potion)
    {
        try { return DescribeText(potion.Description, potion) ?? ""; }
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
        var playerCreature = player?.Creature;

        return new
        {
            round = context.CombatState.RoundNumber,
            side = context.CombatState.CurrentSide.ToString(),
            play_phase = context.CombatManager.IsPlayPhase,
            can_act = !context.CombatManager.PlayerActionsDisabled,
            energy = playerCombat?.Energy,
            max_energy = playerCombat?.MaxEnergy,
            stars = playerCombat?.Stars,
            hand = playerCombat?.Hand?.Cards.Select(card => BuildEnvCardPayload(card, GetCardReference(card))).ToArray()
                ?? Array.Empty<object>(),
            draw = playerCombat?.DrawPile?.Cards.Count ?? 0,
            discard = playerCombat?.DiscardPile?.Cards.Count ?? 0,
            exhaust = playerCombat?.ExhaustPile?.Cards.Count ?? 0,
            allies = context.CombatState.PlayerCreatures
                .Where(creature => playerCreature is null || !ReferenceEquals(creature, playerCreature))
                .Select(BuildEnvCreaturePayload)
                .ToArray(),
            enemies = context.CombatState.Creatures
                .Where(static creature => creature.IsEnemy)
                .Select(BuildEnvEnemyPayload)
                .ToArray(),
            player_powers = playerCreature?.Powers
                .Select(static power => new
                {
                    title = TextOf(power.Title),
                    amount = power.Amount
                })
                .ToArray() ?? Array.Empty<object>()
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
            block = creature.Block
        };
    }

    private static object BuildEnvEnemyPayload(Creature creature)
    {
        var intent = JsonSerializer.SerializeToElement(BuildEnemyIntentPayload(creature));
        return new
        {
            id = creature.CombatId,
            name = creature.Name,
            hp = creature.CurrentHp,
            max_hp = creature.MaxHp,
            block = creature.Block,
            intent = new
            {
                intent_type = TryGetFirstIntentString(intent, "intent_type"),
                title = TryGetNestedString(intent, "title"),
                description = TryGetFirstIntentString(intent, "description"),
                total_damage = TryGetFirstIntentTotalDamage(intent),
                repeats = TryGetFirstIntentRepeats(intent)
            },
            powers = creature.Powers.Select(static power => new
            {
                title = TextOf(power.Title),
                amount = power.Amount
            }).ToArray()
        };
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
                decision_text = $"奖励选择｜可领取{context.RewardButtons.Count}项奖励"
            },
            "card_reward" => new
            {
                option_count = context.CardRewardOptions.Count,
                can_skip = context.CardRewardSkipButton is not null &&
                           IsNodeVisible(context.CardRewardSkipButton) &&
                           IsButtonEnabled(context.CardRewardSkipButton),
                decision_text = $"卡牌奖励｜{context.CardRewardOptions.Count}张卡牌可选"
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
                decision_text = "营火｜选择休息或锻造"
            },
            "deck_upgrade" => BuildEnvDeckUpgradeDecisionPayload(context),
            "card_selection" => BuildEnvCardSelectionDecisionPayload(context),
            "shop" => new
            {
                is_open = context.MerchantInventory?.IsOpen ?? false,
                gold = context.MerchantInventory?.Inventory?.Player?.Gold,
                item_count = context.MerchantSlots.Count,
                decision_text = $"商店｜{context.MerchantSlots.Count}件商品｜金币{context.MerchantInventory?.Inventory?.Player?.Gold ?? 0}"
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
        var texts = context.DeckUpgradeScreen is not null && IsNodeVisible(context.DeckUpgradeScreen)
            ? CollectVisibleText(context.DeckUpgradeScreen, 8).ToArray()
            : Array.Empty<string>();
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
            selection_semantics = "upgrade",
            prompt,
            texts,
            option_count = options.Count,
            decision_text = BuildDeckUpgradeDecisionText(prompt, useSingleSelection, selectedCount, confirmReady),
            options
        };
    }

    private static object BuildEnvCardSelectionDecisionPayload(BridgeWorldContext context)
    {
        var screen = context.CardSelectionScreen;
        var prefs = GetHiddenFieldValue(screen, "_prefs");
        var prompt = TryGetCardSelectionPrompt(screen);
        var texts = screen is not null && IsNodeVisible(screen)
            ? CollectVisibleText(screen, 8).ToArray()
            : Array.Empty<string>();
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
        var selectionSemantics = ResolveCardSelectionSemantics(screen, prompt, texts);

        return new
        {
            screen_type = screen?.GetType().Name,
            prompt,
            texts,
            selection_semantics = selectionSemantics,
            selected_count = selectedCount,
            min_select = minSelect,
            max_select = maxSelect,
            requires_manual_confirmation = requiresManualConfirmation,
            cancelable = cancelable,
            confirm_ready = confirmReady,
            can_skip = canSkip,
            decision_text = BuildCardSelectionDecisionText(
                selectionSemantics,
                prompt,
                selectedCount,
                minSelect,
                maxSelect,
                confirmReady,
                canSkip)
        };
    }

    private static string BuildDeckUpgradeDecisionText(
        string? prompt,
        bool useSingleSelection,
        int selectedCount,
        bool confirmReady)
    {
        var prefix = useSingleSelection ? "锻造选牌" : "多重锻造选牌";
        var status = confirmReady
            ? "可确认"
            : useSingleSelection ? "等待选择" : "继续选择";
        var normalizedPrompt = NormalizeSemanticText(prompt ?? "");
        return string.IsNullOrWhiteSpace(normalizedPrompt)
            ? $"{prefix}｜已选{selectedCount}张｜{status}"
            : $"{prefix}｜{normalizedPrompt}｜已选{selectedCount}张｜{status}";
    }

    private static string BuildCardSelectionDecisionText(
        string? selectionSemantics,
        string? prompt,
        int selectedCount,
        int? minSelect,
        int? maxSelect,
        bool confirmReady,
        bool canSkip)
    {
        var prefix = $"{DescribeSelectionSemanticsLabel(selectionSemantics)}选牌";
        var normalizedPrompt = NormalizeSemanticText(prompt ?? "");
        var targetCount = maxSelect ?? minSelect;
        var progress = targetCount is > 0
            ? $"已选{selectedCount}/{targetCount}"
            : $"已选{selectedCount}张";
        var status = confirmReady
            ? "可确认"
            : canSkip ? "可跳过" : "继续选择";
        return string.IsNullOrWhiteSpace(normalizedPrompt)
            ? $"{prefix}｜{progress}｜{status}"
            : $"{prefix}｜{normalizedPrompt}｜{progress}｜{status}";
    }

    private static string ResolveCardSelectionSemantics(
        Node? cardSelectionScreen,
        string? prompt = null,
        IReadOnlyList<string>? texts = null)
    {
        if (string.Equals(cardSelectionScreen?.GetType().Name, "NChooseABundleSelectionScreen", StringComparison.Ordinal))
        {
            return "bundle";
        }

        var comparableTexts = new List<string>();
        if (!string.IsNullOrWhiteSpace(prompt))
        {
            comparableTexts.Add(NormalizeComparableText(prompt));
        }

        if (texts is not null)
        {
            comparableTexts.AddRange(texts
                .Where(static text => !string.IsNullOrWhiteSpace(text))
                .Select(NormalizeComparableText));
        }

        var combined = string.Join(" ", comparableTexts).ToLowerInvariant();
        if (string.IsNullOrWhiteSpace(combined))
        {
            return "choose";
        }

        if (combined.Contains("移除", StringComparison.Ordinal) ||
            combined.Contains("删除", StringComparison.Ordinal) ||
            combined.Contains("remove", StringComparison.Ordinal) ||
            combined.Contains("purge", StringComparison.Ordinal))
        {
            return "remove";
        }

        if (combined.Contains("变化", StringComparison.Ordinal) ||
            combined.Contains("变形", StringComparison.Ordinal) ||
            combined.Contains("transform", StringComparison.Ordinal))
        {
            return "transform";
        }

        if (combined.Contains("弃牌", StringComparison.Ordinal) ||
            combined.Contains("弃置", StringComparison.Ordinal) ||
            combined.Contains("discard", StringComparison.Ordinal))
        {
            return "discard";
        }

        if (combined.Contains("保留", StringComparison.Ordinal) ||
            combined.Contains("retain", StringComparison.Ordinal))
        {
            return "retain";
        }

        if (combined.Contains("升级", StringComparison.Ordinal) ||
            combined.Contains("锻造", StringComparison.Ordinal) ||
            combined.Contains("upgrade", StringComparison.Ordinal) ||
            combined.Contains("smith", StringComparison.Ordinal))
        {
            return "upgrade";
        }

        if (combined.Contains("组合", StringComparison.Ordinal) ||
            combined.Contains("bundle", StringComparison.Ordinal))
        {
            return "bundle";
        }

        return "choose";
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

    private static IReadOnlyList<BridgeResolvedAction> BuildEnvLegalActions(IReadOnlyList<BridgeResolvedAction> actions)
    {
        // Filter out map travel actions when non-map actions are also present.
        // This handles the map overlay bug where MAP screen shows over a room,
        // exposing both map travel and room actions simultaneously.
        var hasMapActions = actions.Any(static a => a.ActionId.StartsWith("map:", StringComparison.Ordinal));
        // Any action that is NOT a map travel action counts as non-map
        var hasNonMapActions = actions.Any(static a =>
            !a.ActionId.StartsWith("map:", StringComparison.Ordinal));

        if (hasMapActions && hasNonMapActions)
        {
            // Map overlay detected — filter out map actions, keep room actions
            return actions
                .Where(static a => !a.ActionId.StartsWith("map:", StringComparison.Ordinal))
                .ToList();
        }

        return actions;
    }
}
