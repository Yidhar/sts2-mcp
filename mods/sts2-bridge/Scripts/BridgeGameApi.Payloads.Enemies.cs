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
            is_primary_enemy = creature.IsPrimaryEnemy,
            is_secondary_enemy = creature.IsSecondaryEnemy,
            is_stunned = creature.IsStunned,
            is_pet = creature.IsPet,
            shows_infinite_hp = TryGetBoolFromPropertyOrField(
                creature,
                "ShowsInfiniteHp",
                "_showsInfiniteHp") ?? false,
            can_receive_powers = creature.CanReceivePowers,
            slot_name = creature.SlotName,
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
            is_move = nextMove?.IsMove ?? false,
            must_perform_once_before_transitioning = nextMove?.MustPerformOnceBeforeTransitioning ?? false,
            can_transition_away = nextMove?.CanTransitionAway ?? false,
            is_performing_move = monster.IsPerformingMove,
            spawned_this_turn = monster.SpawnedThisTurn,
            intends_to_attack = monster.IntendsToAttack,
            move_history = monster.MoveStateMachine?.StateLog
                .Where(static state => state.ShouldAppearInLogs)
                .Select(static state => state.Id)
                .ToArray() ?? Array.Empty<string>(),
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
        catch (Exception ex)
        {
            BridgeDebugTrace.Write(
                $"monster_intent_total_damage_failed intent_type={intent.GetType().FullName}: {ex.GetBaseException().Message}");
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
        var modelId = power.Id.ToString();
        var className = power.GetType().Name;
        return new
        {
            id = modelId,
            model_id = modelId,
            class_name = className,
            kind = className,
            title = TextOf(power.Title),
            description = TryGetDescription(power),
            amount = power.Amount,
            amount_on_turn_start = power.AmountOnTurnStart,
            display_amount = power.DisplayAmount,
            type = power.Type.ToString(),
            stack_type = power.StackType.ToString(),
            type_for_current_amount = power.TypeForCurrentAmount.ToString(),
            is_instanced = TryGetBoolFromPropertyOrField(
                power,
                "IsInstanced",
                "_isInstanced") ?? false,
            is_visible = power.IsVisible,
            allow_negative = power.AllowNegative,
            skip_next_duration_tick = power.SkipNextDurationTick,
            should_scale_in_multiplayer = power.ShouldScaleInMultiplayer,
            owner_is_secondary_enemy = power.OwnerIsSecondaryEnemy,
            owner_side = power.Owner.Side.ToString(),
            owner_model_id = power.Owner.ModelId.ToString(),
            applier_side = power.Applier?.Side.ToString(),
            applier_model_id = power.Applier?.ModelId.ToString(),
            target_side = power.Target?.Side.ToString(),
            target_model_id = power.Target?.ModelId.ToString(),
            dynamic_vars = BuildDynamicVarPayloads(power.DynamicVars)
        };
    }

}
