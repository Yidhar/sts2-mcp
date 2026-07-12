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

        var staticTraits = creature.IsEnemy ? BuildEnemyStaticTraitPayloads(creature) : Array.Empty<object>();
        var reactiveTriggers = creature.IsEnemy ? BuildEnemyReactiveTriggerPayloads(creature) : Array.Empty<object>();
        var phaseRules = creature.IsEnemy ? BuildEnemyPhaseRulePayloads(creature) : Array.Empty<object>();
        var combatTags = creature.IsEnemy ? BuildEnemyCombatTags(creature) : Array.Empty<string>();
        var targetPriorityHints = creature.IsEnemy ? BuildEnemyTargetPriorityHints(creature, combatTags) : Array.Empty<object>();

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
            intent = creature.IsEnemy ? BuildEnemyIntentPayload(creature) : null,
            static_traits = staticTraits,
            reactive_triggers = reactiveTriggers,
            phase_rules = phaseRules,
            combat_tags = combatTags,
            danger_profile = creature.IsEnemy ? BuildEnemyDangerProfile(creature, combatTags) : null,
            target_priority_hints = targetPriorityHints
        };
    }

    private static object CreateEnemyTraitPayload(
        string category,
        string trait,
        string description,
        string? triggerType = null,
        string? condition = null,
        string? effectType = null,
        int? threshold = null,
        string? state = null,
        int? effectAmount = null,
        string? severity = null)
    {
        return new
        {
            category,
            trait,
            description,
            trigger_type = triggerType,
            condition,
            effect_type = effectType,
            threshold,
            state,
            effect_amount = effectAmount,
            severity
        };
    }

    private static string BuildEnemySearchText(Creature creature)
    {
        var parts = new List<string>();
        if (!string.IsNullOrWhiteSpace(creature.Name))
        {
            parts.Add(creature.Name);
        }

        if (creature.ModelId is not null)
        {
            parts.Add(creature.ModelId.ToString());
        }

        if (creature.Monster?.NextMove?.StateId is string stateId && !string.IsNullOrWhiteSpace(stateId))
        {
            parts.Add(stateId);
        }

        foreach (var power in creature.Powers.Take(4))
        {
            var title = TextOf(power.Title);
            var description = TryGetDescription(power);
            if (!string.IsNullOrWhiteSpace(title))
            {
                parts.Add(title);
            }
            if (!string.IsNullOrWhiteSpace(description))
            {
                parts.Add(description);
            }
        }

        return string.Join(" ", parts).ToLowerInvariant();
    }

    private static object[] BuildEnemyStaticTraitPayloads(Creature creature)
    {
        var searchText = BuildEnemySearchText(creature);
        var traits = new List<object>();

        if (searchText.Contains("nexus", StringComparison.Ordinal) ||
            searchText.Contains("progenitor", StringComparison.Ordinal) ||
            searchText.Contains("queen", StringComparison.Ordinal) ||
            searchText.Contains("egg", StringComparison.Ordinal))
        {
            traits.Add(CreateEnemyTraitPayload("static", "summon_engine", "Acts as a board-pressure engine or summon core.", severity: "high"));
        }

        if (searchText.Contains("door", StringComparison.Ordinal))
        {
            traits.Add(CreateEnemyTraitPayload("static", "gatekeeper", "Encounter progression is gated until this unit's cycle is solved.", severity: "high"));
        }

        if (searchText.Contains("matriarch", StringComparison.Ordinal) ||
            searchText.Contains("nexus", StringComparison.Ordinal) ||
            searchText.Contains("byrdonis", StringComparison.Ordinal) ||
            searchText.Contains("wurm", StringComparison.Ordinal))
        {
            traits.Add(CreateEnemyTraitPayload("static", "time_scaling", "Threat grows materially if the fight drags.", severity: "high"));
        }

        if (searchText.Contains("retali", StringComparison.Ordinal) ||
            searchText.Contains("thorn", StringComparison.Ordinal) ||
            searchText.Contains("spiny", StringComparison.Ordinal))
        {
            traits.Add(CreateEnemyTraitPayload("static", "contact_retaliate", "Punishes contact hits or spammy multi-hit plans.", severity: "high"));
        }

        return traits.ToArray();
    }

    private static object[] BuildEnemyReactiveTriggerPayloads(Creature creature)
    {
        var searchText = BuildEnemySearchText(creature);
        var triggers = new List<object>();

        if (searchText.Contains("retali", StringComparison.Ordinal) ||
            searchText.Contains("thorn", StringComparison.Ordinal) ||
            searchText.Contains("spike", StringComparison.Ordinal) ||
            searchText.Contains("spiny", StringComparison.Ordinal))
        {
            triggers.Add(CreateEnemyTraitPayload(
                "reactive",
                "retaliate",
                "On hit or contact, this enemy punishes damage with retaliation.",
                triggerType: "on_hit",
                condition: "contact",
                effectType: "retaliate",
                severity: "high"));
        }

        if (searchText.Contains("egg", StringComparison.Ordinal) ||
            searchText.Contains("progenitor", StringComparison.Ordinal) ||
            searchText.Contains("summon", StringComparison.Ordinal))
        {
            triggers.Add(CreateEnemyTraitPayload(
                "reactive",
                "summon",
                "If left alive or when killed, this enemy can continue board pressure via summons.",
                triggerType: "on_turn_end",
                condition: "alive",
                effectType: "summon",
                severity: "high"));
        }

        return triggers.ToArray();
    }

    private static object[] BuildEnemyPhaseRulePayloads(Creature creature)
    {
        var searchText = BuildEnemySearchText(creature);
        var rules = new List<object>();

        if (searchText.Contains("split", StringComparison.Ordinal) || searchText.Contains("prism", StringComparison.Ordinal))
        {
            rules.Add(CreateEnemyTraitPayload(
                "phase",
                "split",
                "Crossing a threshold can split or multiply the board state.",
                triggerType: "on_hp_threshold",
                condition: "threshold_crossed",
                effectType: "split",
                severity: "medium"));
        }

        if (searchText.Contains("phase", StringComparison.Ordinal) ||
            searchText.Contains("threshold", StringComparison.Ordinal) ||
            searchText.Contains("subject", StringComparison.Ordinal) ||
            searchText.Contains("doormaker", StringComparison.Ordinal))
        {
            rules.Add(CreateEnemyTraitPayload(
                "phase",
                "phase_shift",
                "The enemy has threshold- or cycle-based phase changes.",
                triggerType: "on_hp_threshold",
                condition: "threshold_crossed",
                effectType: "phase_shift",
                severity: "high"));
        }

        if (searchText.Contains("intang", StringComparison.Ordinal))
        {
            rules.Add(CreateEnemyTraitPayload(
                "phase",
                "gain_intangible",
                "Intangible windows change when burst should be committed.",
                triggerType: "on_turn_start",
                condition: "intangible_window",
                effectType: "gain_intangible",
                severity: "high"));
        }

        if (searchText.Contains("insatiable", StringComparison.Ordinal))
        {
            rules.Add(CreateEnemyTraitPayload(
                "phase",
                "countdown_tick",
                "An external countdown or timer pressures the fight every turn.",
                triggerType: "on_turn_start",
                condition: "countdown_active",
                effectType: "countdown_tick",
                severity: "high"));
        }

        if (string.Equals(creature.ModelId.ToString(), "MONSTER.DOOR", StringComparison.Ordinal))
        {
            rules.Add(CreateEnemyTraitPayload(
                "phase",
                "reveal_boss",
                "Destroying the door exposes the main boss window.",
                triggerType: "on_death",
                condition: "door_destroyed",
                effectType: "reveal_boss",
                severity: "high"));
        }

        return rules.ToArray();
    }

    private static string[] BuildEnemyCombatTags(Creature creature)
    {
        var searchText = BuildEnemySearchText(creature);
        var tags = new HashSet<string>(StringComparer.Ordinal);

        void Add(string tag)
        {
            if (!string.IsNullOrWhiteSpace(tag))
            {
                tags.Add(tag);
            }
        }

        if (searchText.Contains("boss", StringComparison.Ordinal) ||
            searchText.Contains("queen", StringComparison.Ordinal) ||
            searchText.Contains("doormaker", StringComparison.Ordinal) ||
            searchText.Contains("insatiable", StringComparison.Ordinal) ||
            searchText.Contains("matriarch", StringComparison.Ordinal) ||
            searchText.Contains("subject", StringComparison.Ordinal))
        {
            Add("boss");
        }
        if (searchText.Contains("summon", StringComparison.Ordinal) ||
            searchText.Contains("spawn", StringComparison.Ordinal) ||
            searchText.Contains("nexus", StringComparison.Ordinal) ||
            searchText.Contains("progenitor", StringComparison.Ordinal) ||
            searchText.Contains("queen", StringComparison.Ordinal) ||
            searchText.Contains("egg", StringComparison.Ordinal))
        {
            Add("summoner");
        }
        if (searchText.Contains("retali", StringComparison.Ordinal) ||
            searchText.Contains("thorn", StringComparison.Ordinal) ||
            searchText.Contains("spiny", StringComparison.Ordinal))
        {
            Add("retaliation");
        }
        if (searchText.Contains("phase", StringComparison.Ordinal) ||
            searchText.Contains("threshold", StringComparison.Ordinal) ||
            searchText.Contains("split", StringComparison.Ordinal) ||
            searchText.Contains("door", StringComparison.Ordinal) ||
            searchText.Contains("subject", StringComparison.Ordinal))
        {
            Add("phase_shift");
        }
        if (searchText.Contains("matriarch", StringComparison.Ordinal) ||
            searchText.Contains("nexus", StringComparison.Ordinal) ||
            searchText.Contains("byrdonis", StringComparison.Ordinal) ||
            searchText.Contains("wurm", StringComparison.Ordinal))
        {
            Add("scaling");
        }
        if (searchText.Contains("debuff", StringComparison.Ordinal) ||
            searchText.Contains("bind", StringComparison.Ordinal) ||
            searchText.Contains("vulnerable", StringComparison.Ordinal) ||
            searchText.Contains("weak", StringComparison.Ordinal) ||
            searchText.Contains("beetle", StringComparison.Ordinal))
        {
            Add("controller");
        }

        return tags.ToArray();
    }

    private static object BuildEnemyDangerProfile(Creature creature, IReadOnlyCollection<string>? combatTags)
    {
        var searchText = BuildEnemySearchText(creature);
        var burst = 0;
        var attrition = 0;
        var scaling = 0;
        var retaliation = 0;
        var summonPressure = 0;
        var debuffPressure = 0;
        var phaseComplexity = 0;
        var volatility = 0;
        var targetPriority = 0;
        string? notes = null;

        if (combatTags is not null && combatTags.Contains("retaliation"))
        {
            retaliation = 5;
            attrition = Math.Max(attrition, 3);
            targetPriority = Math.Max(targetPriority, 4);
        }
        if (combatTags is not null && combatTags.Contains("summoner"))
        {
            summonPressure = 4;
            targetPriority = Math.Max(targetPriority, 4);
        }
        if (combatTags is not null && combatTags.Contains("phase_shift"))
        {
            phaseComplexity = 4;
            volatility = Math.Max(volatility, 3);
        }
        if (combatTags is not null && combatTags.Contains("scaling"))
        {
            scaling = 4;
        }
        if (combatTags is not null && combatTags.Contains("controller"))
        {
            debuffPressure = 4;
            attrition = Math.Max(attrition, 3);
        }

        if (searchText.Contains("insatiable", StringComparison.Ordinal))
        {
            burst = Math.Max(burst, 4);
            scaling = Math.Max(scaling, 4);
            phaseComplexity = Math.Max(phaseComplexity, 4);
            volatility = Math.Max(volatility, 5);
            targetPriority = Math.Max(targetPriority, 5);
            notes = "countdown boss";
        }
        if (searchText.Contains("doormaker", StringComparison.Ordinal) || searchText.Contains("door", StringComparison.Ordinal))
        {
            burst = Math.Max(burst, 4);
            scaling = Math.Max(scaling, 4);
            phaseComplexity = Math.Max(phaseComplexity, 5);
            targetPriority = Math.Max(targetPriority, 5);
        }
        if (searchText.Contains("subject", StringComparison.Ordinal))
        {
            burst = Math.Max(burst, 5);
            attrition = Math.Max(attrition, 4);
            phaseComplexity = Math.Max(phaseComplexity, 5);
            volatility = Math.Max(volatility, 5);
            targetPriority = Math.Max(targetPriority, 5);
        }
        if (searchText.Contains("spiny", StringComparison.Ordinal))
        {
            targetPriority = Math.Max(targetPriority, 4);
        }

        return new
        {
            burst,
            attrition,
            scaling,
            retaliation,
            summon_pressure = summonPressure,
            debuff_pressure = debuffPressure,
            phase_complexity = phaseComplexity,
            volatility,
            target_priority = targetPriority,
            notes
        };
    }

    private static object[] BuildEnemyTargetPriorityHints(Creature creature, IReadOnlyCollection<string>? combatTags)
    {
        var hints = new List<object>();
        if (combatTags is not null && combatTags.Contains("retaliation"))
        {
            hints.Add(new
            {
                priority = "high",
                reason = "Remove retaliation sources before committing multi-hit turns."
            });
        }
        if (combatTags is not null && combatTags.Contains("summoner"))
        {
            hints.Add(new
            {
                priority = "high",
                reason = "Kill engine or summon-core enemies early if your deck is weak to board flood."
            });
        }
        if (combatTags is not null && combatTags.Contains("phase_shift"))
        {
            hints.Add(new
            {
                priority = "medium",
                reason = "Plan burst around threshold or exposed-window turns, not only raw HP racing."
            });
        }
        return hints.ToArray();
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
            display_amount = power.DisplayAmount,
            type = power.Type.ToString(),
            stack_type = power.StackType.ToString()
        };
    }

}
