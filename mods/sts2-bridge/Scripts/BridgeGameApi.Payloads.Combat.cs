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

    // Surface stable factual card semantics for versioned consumers. These are
    // game facts only; reward, policy bias, and curriculum remain outside Bridge.
    private static object BuildPlayCardSemantic(CardModel? card, Creature? previewTarget)
    {
        if (card is null)
        {
            return new { family = "play_card", roles = Array.Empty<string>() };
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
        var bodySlamDamage = SafeResolveBodySlamDamage(card);
        if (bodySlamDamage is > 0 && (!totalDamage.HasValue || totalDamage.Value <= 0))
        {
            damagePerHit = bodySlamDamage;
            totalDamage = bodySlamDamage;
            repeats = Math.Max(repeats.GetValueOrDefault(1), 1);
        }
        var totalBlock = GetDynamicVarInt(previewVars, "CalculatedBlock") ?? GetDynamicVarInt(previewVars, "Block");
        var drawCount = GetDynamicVarInt(previewVars, "Cards");
        var weakAmount = GetDynamicVarInt(previewVars, "Weak");
        var vulnerableAmount = GetDynamicVarInt(previewVars, "Vulnerable");
        var poisonAmount = GetDynamicVarInt(previewVars, "Poison");
        var strengthAmount = GetDynamicVarInt(previewVars, "Strength");
        var dexterityAmount = GetDynamicVarInt(previewVars, "Dexterity");

        var roles = new List<string>();
        var cardType = card.Type.ToString();
        if (totalBlock is > 0) roles.Add("block");
        if (weakAmount is > 0) { roles.Add("weak"); roles.Add("debuff"); }
        if (vulnerableAmount is > 0) { roles.Add("vulnerable"); roles.Add("debuff"); }
        if (poisonAmount is > 0) { roles.Add("poison"); roles.Add("debuff"); }
        if (drawCount is > 0) roles.Add("draw");
        if (strengthAmount is > 0 || dexterityAmount is > 0) roles.Add("scaling");
        if (string.Equals(cardType, "Power", StringComparison.OrdinalIgnoreCase)) roles.Add("power");
        if (string.Equals(cardType, "Attack", StringComparison.OrdinalIgnoreCase) && totalDamage is > 0) roles.Add("attack");
        if (bodySlamDamage.HasValue) roles.Add("block_scaled_damage");

        return new
        {
            family = "play_card",
            roles = roles.Distinct().ToArray(),
            damage = totalDamage ?? 0,
            damage_per_hit = damagePerHit ?? 0,
            block_scaled_damage = bodySlamDamage.HasValue,
            block_scaled_damage_source = bodySlamDamage.HasValue ? "player_current_block" : null,
            block = totalBlock ?? 0,
            hits = repeats ?? (totalDamage is > 0 ? 1 : 0),
            draw = drawCount ?? 0,
            weak = weakAmount ?? 0,
            vulnerable = vulnerableAmount ?? 0,
            poison = poisonAmount ?? 0,
            strength = strengthAmount ?? 0,
            dexterity = dexterityAmount ?? 0,
            card_type = cardType
        };
    }

    private static object BuildUsePotionSemantic(PotionModel? potion)
    {
        if (potion is null)
        {
            return new { family = "use_potion", roles = Array.Empty<string>() };
        }
        var profileEntry = TryGetPotionProfileEntry(potion);
        var roles = InferUsePotionRoles(profileEntry).ToArray();
        return new
        {
            family = "use_potion",
            roles,
            potion_id = potion.Id.ToString(),
            rarity = profileEntry?.Rarity ?? potion.Rarity.ToString(),
            target_scope = profileEntry?.TargetScope,
            effect_family = profileEntry?.EffectFamily.ToArray() ?? Array.Empty<string>(),
            effect_profile = BuildPotionEffectProfilePayload(profileEntry),
            semantic_tags = profileEntry?.SemanticTags.ToArray() ?? Array.Empty<string>(),
            timing_tags = profileEntry?.TimingTags.ToArray() ?? Array.Empty<string>()
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
        var currentStarCost = SafeResolveCardStarCost(card);
        var xCostSemantics = ResolveXCostSemantics(card, description, damagePerHit, totalDamage, repeats, xCostValue);
        (damagePerHit, totalDamage, repeats) = ApplyXCostPreviewMapping(
            damagePerHit,
            totalDamage,
            repeats,
            xCostValue,
            xCostSemantics);
        var bodySlamDamage = SafeResolveBodySlamDamage(card);
        if (bodySlamDamage is > 0 && (!totalDamage.HasValue || totalDamage.Value <= 0))
        {
            damagePerHit = bodySlamDamage;
            totalDamage = bodySlamDamage;
            repeats = Math.Max(repeats.GetValueOrDefault(1), 1);
        }
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
        var keywordNames = card.Keywords
            .Where(static keyword => keyword != CardKeyword.None)
            .Select(static keyword => keyword.ToString())
            .Distinct(StringComparer.Ordinal)
            .OrderBy(static keyword => keyword, StringComparer.Ordinal)
            .ToArray();
        var keywordSet = card.Keywords.ToHashSet();
        var afflictions = BuildCardModifierPayloads(card, "afflictions");
        var enchantments = BuildCardModifierPayloads(card, "enchantments");
        var modifierSummary = BuildCardModifierSummaryPayload(keywordSet, afflictions, enchantments, card.Type.ToString());
        var cardFlow = BuildCardFlowPayload(card, keywordSet, modifierSummary);

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
            exhaust = keywordSet.Contains(CardKeyword.Exhaust),
            exhaust_self = keywordSet.Contains(CardKeyword.Exhaust),
            will_exhaust = keywordSet.Contains(CardKeyword.Exhaust),
            ethereal = keywordSet.Contains(CardKeyword.Ethereal),
            retain = keywordSet.Contains(CardKeyword.Retain),
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
                block_scaled_damage = bodySlamDamage.HasValue,
                block_scaled_damage_source = bodySlamDamage.HasValue ? "player_current_block" : null,
                x_cost_value = xCostValue,
                x_cost_semantics = xCostSemantics
            },
            // P0-1: typed safety payload — Python ``hp_cost_safety_view``
            // prefers this block over fallback ``effect_preview.hp_loss``.
            // Block does NOT soak ``cardHpLoss`` / ``nonCardHpLoss`` per
            // STS2 source, so ``hp_loss_unblockable`` is the canonical
            // figure; we keep ``self_damage_blockable`` 0 here until the
            // game exposes a blockable channel.
            safety = new
            {
                hp_cost_kind = hpLossAmount.GetValueOrDefault(0) > 0 ? "unblockable_hp_loss" : "none",
                hp_cost = hpLossAmount.GetValueOrDefault(0),
                hp_loss_unblockable = hpLossAmount.GetValueOrDefault(0),
                self_damage_blockable = 0,
                max_hp_loss = 0,
                source_confidence = "runtime_internal"
            },
            // P0-4: typed X-cost / Star-X block — Python ``x_cost_view``
            // prefers this typed block over fallback inference.  Resource
            // routing distinguishes energy-X (``CostsX``) from Star-X
            // (``HasStarCostX``).  ``current_value`` is the resolved
            // X resource at decision time (``xCostValue`` for energy,
            // ``currentStarCost`` for stars).
            x_cost = new
            {
                has_x_cost = card.EnergyCost.CostsX || card.HasStarCostX,
                resource = card.EnergyCost.CostsX ? "energy" : (card.HasStarCostX ? "stars" : "none"),
                current_value = card.EnergyCost.CostsX ? (xCostValue ?? 0) : (card.HasStarCostX ? (currentStarCost ?? 0) : 0),
                is_zero = (card.EnergyCost.CostsX && (xCostValue ?? 0) == 0) || (card.HasStarCostX && (currentStarCost ?? 0) == 0),
                effect_scaled = card.EnergyCost.CostsX || card.HasStarCostX,
                preview_scale_source = card.EnergyCost.CostsX ? "energy_x" : (card.HasStarCostX ? "star_x" : "none"),
                semantics = string.IsNullOrWhiteSpace(xCostSemantics) ? "unknown" : xCostSemantics
            },
            // P0-5: typed selection block — for ``play_card`` actions (i.e.
            // anything emitted from this BuildCardPayload helper) the
            // operation_type is "" because card play is not a card-selection
            // operation.  Downstream Python ``selection_view`` then walks
            // the typed card_effect_profile or text fallback.  When the
            // bridge later wires ResolveCardSelectionSemantics through this
            // block on actual card-selection screens, operation_type +
            // source_zone + confidence will flip to runtime_internal.
            selection = new
            {
                screen_type = "play_card",
                operation_type = "",
                source = "card",
                source_zone = card.Pile?.Type.ToString() ?? "",
                destination_zone = "",
                min_count = 0,
                max_count = 0,
                selection_required = false,
                modifier_id = "",
                confidence = "runtime_internal"
            },
            dynamic_vars = BuildDynamicVarPayloads(previewVars),
            afflictions,
            enchantments,
            modifier_summary = modifierSummary,
            card_effect_profile = BuildCardEffectProfilePayload(card),
            card_flow = cardFlow
        };
    }

    private static object[] BuildCardModifierPayloads(CardModel card, string modifierKind)
    {
        // Bosses/events can attach runtime per-card restrictions or buffs (Queen
        // binding/chains, forced retain/exhaust/cost mutations, etc.).  Their
        // concrete STS2 classes have moved across builds, so expose them through
        // reflection rather than coupling the bridge to one exact API surface.
        var candidates = modifierKind.Equals("afflictions", StringComparison.OrdinalIgnoreCase)
            ? new[] { "Afflictions", "AfflictionModels", "CardAfflictions", "Statuses", "StatusEffects" }
            : new[] { "Enchantments", "EnchantmentModels", "CardEnchantments", "Modifiers", "CardModifiers" };

        var emitted = new List<object>();
        foreach (var memberName in candidates)
        {
            foreach (var item in EnumerateHiddenCollectionMember(card, memberName))
            {
                var payload = BuildCardModifierPayload(item, modifierKind, memberName);
                if (payload is not null)
                {
                    emitted.Add(payload);
                    if (emitted.Count >= 16)
                    {
                        return emitted.ToArray();
                    }
                }
            }
        }

        return emitted.ToArray();
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

    private static object? BuildCardModifierPayload(object? modifier, string kind, string sourceMember)
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
        var semanticTags = ResolveCardModifierSemanticTags(kind, id, typeName, title, description, amount, status);
        var semanticValues = ResolveCardModifierSemanticValues(kind, id, typeName, title, description, amount, status);

        return new
        {
            kind,
            source_member = sourceMember,
            id,
            type = typeName,
            title,
            description,
            amount,
            status,
            enabled,
            semantic_tags = semanticTags,
            semantic_values = semanticValues,
            is_debuff = GetHiddenPropertyValue<bool>(modifier, "IsDebuff"),
            is_buff = GetHiddenPropertyValue<bool>(modifier, "IsBuff")
        };
    }

    private static string[] ResolveCardModifierSemanticTags(
        string kind,
        string id,
        string typeName,
        string title,
        string description,
        decimal? amount,
        string status)
    {
        var tags = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        var key = NormalizeModifierKey(id, typeName, title);
        var text = $"{kind} {id} {typeName} {title} {description}".ToLowerInvariant();
        void Add(params string[] values)
        {
            foreach (var value in values)
            {
                if (!string.IsNullOrWhiteSpace(value)) tags.Add(value);
            }
        }

        switch (key)
        {
            case "adroit": Add("block_on_play"); break;
            case "swift": Add("draw_on_play", "once_per_combat", "disabled_after_play"); break;
            case "inky": Add("weak_on_play", "damage_add"); break;
            case "sown": Add("energy_gain", "once_per_combat", "disabled_after_play"); break;
            case "corrupted": Add("damage_mult", "self_damage"); break;
            case "imbued": Add("autoplay_round_1", "start_bottom_draw"); break;
            case "glam": Add("replay", "once_per_combat", "disabled_after_play"); break;
            case "goopy": Add("adds_exhaust", "block_add", "grows_on_play"); break;
            case "instinct": Add("damage_mult"); break;
            case "momentum": Add("damage_add", "grows_on_play"); break;
            case "nimble": Add("block_add"); break;
            case "perfectfit": Add("shuffle_top"); break;
            case "royallyapproved": Add("adds_retain"); break;
            case "sharp": Add("damage_add"); break;
            case "slither": Add("cost_randomizes_on_draw"); break;
            case "slumberingessence": Add("cost_reduction_until_played"); break;
            case "soulspower": Add("removes_exhaust"); break;
            case "spiral": Add("replay"); break;
            case "steady": Add("adds_retain"); break;
            case "tezcatarasember": Add("sets_cost_zero", "eternal"); break;
            case "vigorous": Add("damage_add", "once_per_combat", "disabled_after_play"); break;
            case "weighted": Add("energy_loss_on_play"); break;
            case "hexed": Add("adds_ethereal"); break;
            case "devoured": Add("adds_exhaust"); break;
        }

        if (text.Contains("retain", StringComparison.Ordinal) || text.Contains("??", StringComparison.Ordinal)) Add("adds_retain");
        if (text.Contains("ethereal", StringComparison.Ordinal) || text.Contains("??", StringComparison.Ordinal)) Add("adds_ethereal");
        if (text.Contains("exhaust", StringComparison.Ordinal) || text.Contains("??", StringComparison.Ordinal)) Add("adds_exhaust");
        if (text.Contains("draw", StringComparison.Ordinal) || text.Contains("?", StringComparison.Ordinal)) Add("draw_on_play");
        if (text.Contains("energy", StringComparison.Ordinal) || text.Contains("??", StringComparison.Ordinal) || text.Contains("??", StringComparison.Ordinal)) Add("energy_modifier");
        if (!string.IsNullOrWhiteSpace(status)) Add("status_" + NormalizeModifierKey(status, string.Empty, string.Empty));
        return tags.OrderBy(static tag => tag, StringComparer.Ordinal).ToArray();
    }

    private static Dictionary<string, object?> ResolveCardModifierSemanticValues(
        string kind,
        string id,
        string typeName,
        string title,
        string description,
        decimal? amount,
        string status)
    {
        var values = new Dictionary<string, object?>(StringComparer.Ordinal);
        var key = NormalizeModifierKey(id, typeName, title);
        var amt = amount ?? 1m;
        var enabled = !string.Equals(status, "Disabled", StringComparison.OrdinalIgnoreCase);
        void Set(string name, object? value) => values[name] = value;
        void Num(string name, decimal value) => values[name] = value;
        Set("currently_enabled", enabled);

        switch (key)
        {
            case "adroit": Num("block_on_play", amt); break;
            case "swift": Num("draw", amt); Set("once_per_combat", true); Set("disabled_after_play", true); break;
            case "inky": Num("damage_add", 2m); Num("weak", Math.Max(amt, 1m)); break;
            case "sown": Num("energy_gain", amt); Set("once_per_combat", true); Set("disabled_after_play", true); break;
            case "corrupted": Num("damage_mult", 1.5m); Num("self_damage", 2m); break;
            case "imbued": Set("autoplay_round_1", true); Set("start_bottom_draw", true); break;
            case "glam": Num("play_count_bonus", Math.Max(amt, 1m)); Set("once_per_combat", true); Set("disabled_after_play", true); break;
            case "goopy": Set("adds_exhaust", true); Num("block_add", Math.Max(amt - 1m, 0m)); Set("grows_on_play", true); break;
            case "instinct": Num("damage_mult", 2m); break;
            case "momentum": Num("damage_add", amt); Set("grows_on_play", true); break;
            case "nimble": Num("block_add", amt); break;
            case "perfectfit": Set("shuffle_top", true); break;
            case "royallyapproved": Set("adds_retain", true); break;
            case "sharp": Num("damage_add", amt); break;
            case "slither": Set("cost_randomizes_on_draw", true); Num("cost_random_min", 0m); Num("cost_random_max", 3m); break;
            case "slumberingessence": Num("cost_reduction_until_played", 1m); break;
            case "soulspower": Set("removes_exhaust", true); break;
            case "spiral": Num("play_count_bonus", Math.Max(amt, 1m)); break;
            case "steady": Set("adds_retain", true); break;
            case "tezcatarasember": Set("sets_cost_zero", true); Set("eternal", true); break;
            case "vigorous": Num("damage_add", amt); Set("once_per_combat", true); Set("disabled_after_play", true); break;
            case "weighted": Num("energy_loss_on_play", amt); break;
            case "hexed": Set("adds_ethereal", true); break;
            case "devoured": Set("adds_exhaust", true); break;
        }
        return values;
    }

    private static string NormalizeModifierKey(params string?[] values)
    {
        foreach (var value in values)
        {
            if (string.IsNullOrWhiteSpace(value)) continue;
            var chars = value.Where(char.IsLetterOrDigit).ToArray();
            if (chars.Length > 0) return new string(chars).ToLowerInvariant();
        }
        return string.Empty;
    }

}
