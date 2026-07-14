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
using MegaCrit.Sts2.Core.Nodes.Screens.MainMenu;
using MegaCrit.Sts2.Core.Nodes.Screens.Map;
using MegaCrit.Sts2.Core.Nodes.Screens.Overlays;
using MegaCrit.Sts2.Core.Nodes.Screens.ScreenContext;
using MegaCrit.Sts2.Core.Nodes.Screens.Shops;
using MegaCrit.Sts2.Core.Nodes.Screens.TreasureRoomRelic;
using MegaCrit.Sts2.Core.Nodes.TreasureRooms;
using MegaCrit.Sts2.Core.Rewards;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;

namespace Sts2McpBridge.Scripts;

internal static partial class BridgeGameApi
{
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
                            selection_group_key = $"play_card:{playerIndex}:{cardRef}",
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
                            selection_group_key = $"use_potion:{playerIndex}:{slotIndex}",
                            label = $"Use potion {slotIndex}: {potionTitle}{targetLabel}",
                            player_index = playerIndex,
                            player_net_id = player.NetId,
                            slot_index = slotIndex,
                            potion = BuildPotionPayload(potion, slotIndex),
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

        if (context.CombatManager?.IsInProgress != true &&
            IsRewardsScreenVisible(
                context.RewardsScreen,
                context.ProceedButton,
                context.RewardProceedButton,
                context.MapScreen,
                context.RewardButtons))
        {
            AddPotionRewardSkipActions(actions, context);
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
                        potion = BuildPotionPayload(potion, slotIndex),
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
        // Guard: re-check play phase at execution time to prevent TOCTOU races.
        // The action mask snapshot may have been taken before a phase transition
        // (e.g. enemy turn started, card was queued, or a settling phase began).
        var cm = CombatManager.Instance;
        if (cm is not null && cm.IsInProgress && (!IsCombatPlayPhase(cm) || cm.PlayerActionsDisabled || cm.IsPaused))
        {
            throw new BridgeRequestException(
                HttpStatusCode.Conflict,
                "play_card_not_in_play_phase",
                $"Cannot play card '{TextOf(card.Title)}': not in play phase " +
                $"(IsPlayPhase={IsCombatPlayPhase(cm)}, PlayerActionsDisabled={cm.PlayerActionsDisabled}, IsPaused={cm.IsPaused}).");
        }

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
        catch (Exception ex)
        {
            BridgeDebugTrace.Write(
                $"potion_use_enqueue_failed slot={slotIndex} potion_type={potion.GetType().FullName}: {ex.GetBaseException().Message}");
        }

        try
        {
            var action = new UsePotionAction(potion, target!, isCombatInProgress);
            ExecuteGameActionSynchronously(action);
            return;
        }
        catch (Exception ex)
        {
            BridgeDebugTrace.Write(
                $"potion_use_action_failed slot={slotIndex} potion_type={potion.GetType().FullName}: {ex.GetBaseException().Message}");
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
        catch (Exception ex)
        {
            BridgeDebugTrace.Write(
                $"potion_discard_direct_failed slot={slotIndex} potion_type={potion.GetType().FullName}: {ex.GetBaseException().Message}");
        }

        try
        {
            var action = new DiscardPotionGameAction(player, (uint)slotIndex, isCombatInProgress);
            ExecuteGameActionSynchronously(action);
            return;
        }
        catch (Exception ex)
        {
            BridgeDebugTrace.Write(
                $"potion_discard_action_failed slot={slotIndex} potion_type={potion.GetType().FullName}: {ex.GetBaseException().Message}");
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

    private static bool SafeGetCardIsQueued(CardModel card)
    {
        try
        {
            return GetHiddenPropertyValue<bool>(card, "IsQueued") ?? false;
        }
        catch
        {
            return false;
        }
    }

    private static bool CanPlayCard(CardModel card)
    {
        // Reject cards that are already in-flight (queued for execution).
        // Potions have the same guard via potion.IsQueued in CanUsePotion.
        if (SafeGetCardIsQueued(card))
        {
            return false;
        }

        if (TryInvokeBoolean(card, "CanPlay") is bool canPlay)
        {
            return canPlay;
        }

        return SafeGetCardIsPlayable(card);
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

}
