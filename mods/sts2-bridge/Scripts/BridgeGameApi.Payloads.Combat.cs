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
            current_location = BuildCurrentLocationText(runState),
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

    private static string BuildCurrentLocationText(RunState runState)
    {
        var actPart = $"act {runState.CurrentActIndex}";
        if (!runState.CurrentMapCoord.HasValue)
        {
            return actPart;
        }

        var coord = runState.CurrentMapCoord.Value;
        return $"{actPart} coord ({coord.col}, {coord.row})";
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
            is_play_phase = IsCombatPlayPhase(combatManager, combatState),
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
        var description = GetCardDescription(card, resolvedTarget);
        var currentStarCost = SafeResolveCardStarCost(card);
        var keywordNames = card.Keywords
            .Where(static keyword => keyword != CardKeyword.None)
            .Select(static keyword => keyword.ToString())
            .Distinct(StringComparer.Ordinal)
            .OrderBy(static keyword => keyword, StringComparer.Ordinal)
            .ToArray();
        var tagNames = card.Tags
            .Where(static tag => tag != CardTag.None)
            .Select(static tag => tag.ToString())
            .Distinct(StringComparer.Ordinal)
            .OrderBy(static tag => tag, StringComparer.Ordinal)
            .ToArray();
        var afflictions = BuildCardModifierPayloads(card, "afflictions");
        var enchantments = BuildCardModifierPayloads(card, "enchantments");

        return new
        {
            id = card.Id.ToString(),
            model_id = card.Id.ToString(),
            class_name = card.GetType().Name,
            kind = card.GetType().Name,
            // P0-6: stable per-instance handle.  CardModel instances persist
            // for the lifetime of a card object across draw/discard/exhaust
            // pile movement and replay/copy triggers, so the CLR's identity
            // hash is a process-stable per-instance UUID for our purposes.
            // Hex-formatted to make collisions visually obvious in logs.
            instance_uuid = System.Runtime.CompilerServices.RuntimeHelpers.GetHashCode(card).ToString("X"),
            current_upgrade_level = card.CurrentUpgradeLevel,
            max_upgrade_level = card.MaxUpgradeLevel,
            base_replay_count = card.BaseReplayCount,
            current_replay_count = card.GetEnchantedReplayCount(),
            last_stars_spent = card.LastStarsSpent,
            floor_added_to_deck = card.FloorAddedToDeck,
            title = string.IsNullOrWhiteSpace(card.Title)
                ? DescribeText(card.TitleLocString, card)
                : DescribeText(card.Title, card),
            description,
            type = card.Type.ToString(),
            rarity = card.Rarity.ToString(),
            target_type = card.TargetType.ToString(),
            pile = card.Pile?.Type.ToString(),
            is_playable = SafeGetCardIsPlayable(card),
            canonical_energy_cost = card.EnergyCost.Canonical,
            resolved_energy_cost = card.EnergyCost.GetResolved(),
            costs_x = card.EnergyCost.CostsX,
            canonical_star_cost = card.CanonicalStarCost,
            current_star_cost = currentStarCost,
            has_star_cost_x = card.HasStarCostX,
            keywords = keywordNames,
            tags = tagNames,
            hover_tip_ids = BuildCardHoverTipIds(card),
            gains_block = card.GainsBlock,
            has_turn_end_in_hand_effect = card.HasTurnEndInHandEffect,
            has_on_draw_effect = HasCardOnDrawEffect(card),
            exhaust_on_next_play = card.ExhaustOnNextPlay,
            is_removable = card.IsRemovable,
            is_transformable = card.IsTransformable,
            is_in_combat = card.IsInCombat,
            is_upgradable = card.IsUpgradable,
            is_sly_this_turn = card.IsSlyThisTurn,
            is_retained = card.ShouldRetainThisTurn,
            is_clone = card.IsClone,
            is_dupe = card.IsDupe,
            has_been_removed_from_state = card.HasBeenRemovedFromState,
            dynamic_vars = BuildDynamicVarPayloads(previewVars),
            afflictions,
            enchantments
        };
    }

    private static object[] BuildCardModifierPayloads(CardModel card, string modifierKind)
    {
        // The retail CardModel contract is singular: a card has at most one
        // Enchantment and one Affliction.  The old plural-reflection probe
        // silently returned an empty list for real cards.
        object? modifier = modifierKind.Equals("afflictions", StringComparison.OrdinalIgnoreCase)
            ? card.Affliction
            : card.Enchantment;
        var payload = BuildCardModifierPayload(modifier);
        return payload is null ? Array.Empty<object>() : new[] { payload };
    }

    private static IEnumerable<object?> EnumerateHiddenCollectionMember(object? target, string memberName)
    {
        if (target is null)
        {
            yield break;
        }

        object? value = null;
        try
        {
            value = GetHiddenPropertyObjectValue(target, memberName);
        }
        catch
        {
            value = null;
        }
        if (value is null)
        {
            try
            {
                value = GetHiddenFieldValue(target, memberName);
            }
            catch
            {
                value = null;
            }
        }
        if (value is null || value is string)
        {
            yield break;
        }

        if (value is IDictionary dictionary)
        {
            foreach (DictionaryEntry entry in dictionary)
            {
                yield return entry.Value;
            }
            yield break;
        }

        if (value is IEnumerable enumerable)
        {
            foreach (var item in enumerable)
            {
                yield return item;
            }
            yield break;
        }

        yield return value;
    }

    private static object? BuildCardModifierPayload(object? modifier)
    {
        if (modifier is null)
        {
            return null;
        }

        var id = FirstNonEmptyText(
            GetHiddenPropertyObjectValue(modifier, "Id"),
            GetHiddenPropertyObjectValue(modifier, "ModelId"),
            GetHiddenPropertyObjectValue(modifier, "Key"),
            GetHiddenFieldValue(modifier, "Id"),
            GetHiddenFieldValue(modifier, "_id"));
        var title = FirstNonEmptyText(
            GetHiddenPropertyObjectValue(modifier, "Title"),
            GetHiddenPropertyObjectValue(modifier, "Name"),
            GetHiddenPropertyObjectValue(modifier, "TitleLocString"),
            GetHiddenFieldValue(modifier, "Title"),
            GetHiddenFieldValue(modifier, "_title"));
        var description = FirstNonEmptyText(
            GetHiddenPropertyObjectValue(modifier, "Description"),
            GetHiddenPropertyObjectValue(modifier, "DynamicDescription"),
            GetHiddenPropertyObjectValue(modifier, "DescriptionLocString"),
            GetHiddenFieldValue(modifier, "Description"),
            GetHiddenFieldValue(modifier, "_description"));

        var amount = FirstNumber(
            GetHiddenPropertyObjectValue(modifier, "Amount"),
            GetHiddenPropertyObjectValue(modifier, "Value"),
            GetHiddenPropertyObjectValue(modifier, "Stacks"),
            GetHiddenFieldValue(modifier, "Amount"),
            GetHiddenFieldValue(modifier, "_amount"));

        var typeName = modifier.GetType().Name;
        if (string.IsNullOrWhiteSpace(id))
        {
            id = typeName;
        }
        if (string.IsNullOrWhiteSpace(title))
        {
            title = id;
        }

        var status = FirstNonEmptyText(
            GetHiddenPropertyObjectValue(modifier, "Status"),
            GetHiddenPropertyObjectValue(modifier, "CurrentStatus"),
            GetHiddenFieldValue(modifier, "Status"),
            GetHiddenFieldValue(modifier, "_status"));
        var enabled = !string.Equals(status, "Disabled", StringComparison.OrdinalIgnoreCase);
        var enchantment = modifier as EnchantmentModel;
        var affliction = modifier as AfflictionModel;

        return new
        {
            id,
            class_name = typeName,
            type = enchantment is not null ? "enchantment" : affliction is not null ? "affliction" : typeName,
            title,
            description,
            amount,
            display_amount = enchantment?.DisplayAmount,
            status,
            enabled,
            show_amount = enchantment?.ShowAmount,
            is_stackable = enchantment?.IsStackable ?? affliction?.IsStackable,
            should_start_at_bottom_of_draw_pile = enchantment?.ShouldStartAtBottomOfDrawPile,
            should_glow_gold = enchantment?.ShouldGlowGold,
            should_glow_red = enchantment?.ShouldGlowRed,
            has_extra_card_text = enchantment?.HasExtraCardText ?? affliction?.HasExtraCardText,
            can_afflict_unplayable_cards = affliction?.CanAfflictUnplayableCards,
            has_overlay = affliction?.HasOverlay,
            dynamic_vars = enchantment is null
                ? Array.Empty<object>()
                : BuildDynamicVarPayloads(enchantment.DynamicVars)
        };
    }


}
