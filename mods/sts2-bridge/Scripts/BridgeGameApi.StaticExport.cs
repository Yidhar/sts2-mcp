using System.Collections;
using System.Globalization;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Text.RegularExpressions;
using Godot;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Events;
using MegaCrit.Sts2.Core.HoverTips;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.MonsterMoves;
using MegaCrit.Sts2.Core.MonsterMoves.Intents;
using MegaCrit.Sts2.Core.MonsterMoves.MonsterMoveStateMachine;

namespace Sts2McpBridge.Scripts;

internal sealed class BridgeStaticExportRequest
{
    [JsonPropertyName("output_dir")]
    public string? OutputDir { get; set; }

    [JsonPropertyName("timeout_ms")]
    public int? TimeoutMs { get; set; }
}

internal static partial class BridgeGameApi
{
    private const int DefaultStaticExportTimeoutMs = 60_000;

    public static async Task<object> ExportStaticDataResponseAsync(
        BridgeStaticExportRequest? request,
        CancellationToken cancellationToken = default)
    {
        EnsureDispatcherReady();

        var safeRequest = request ?? new BridgeStaticExportRequest();
        var timeoutMs = Math.Clamp(
            safeRequest.TimeoutMs ?? DefaultStaticExportTimeoutMs,
            1_000,
            300_000);

        return await RunOnMainThreadGuardedAsync(
            () => ExportStaticDataCore(safeRequest),
            "static.export",
            timeoutMs,
            cancellationToken);
    }

    private static object ExportStaticDataCore(BridgeStaticExportRequest request)
    {
        var exportDir = ResolveStaticExportDirectory(request.OutputDir);
        Directory.CreateDirectory(exportDir);

        var generatedAtUtc = DateTimeOffset.UtcNow;

        var cards = BuildStaticCardRecords();
        var relics = BuildStaticRelicRecords();
        var potions = BuildStaticPotionRecords();
        var events = BuildStaticEventRecords();
        var creatures = BuildStaticCreatureRecords();
        var keywords = BuildStaticKeywordRecords();
        var enchantments = BuildStaticEnchantmentRecords();
        var afflictions = BuildStaticAfflictionRecords();

        var itemsPath = Path.Combine(exportDir, "items.json");
        var manifestPath = Path.Combine(exportDir, "manifest.json");

        var itemsPayload = new Dictionary<string, object?>
        {
            ["mod"] = new
            {
                id = BridgeRuntime.ModId,
                name = BridgeRuntime.BridgeName,
                version = BridgeRuntime.BridgeVersion,
                game_assembly_version = BridgeRuntime.GameAssemblyVersion
            },
            ["generated_at_utc"] = generatedAtUtc.ToString("O", CultureInfo.InvariantCulture),
            ["cards"] = cards,
            ["relics"] = relics,
            ["potions"] = potions,
            ["events"] = events,
            ["creatures"] = creatures,
            ["keywords"] = keywords,
            ["enchantments"] = enchantments,
            ["afflictions"] = afflictions
        };

        var counts = new Dictionary<string, object?>
        {
            ["card_rows"] = cards.Count,
            ["card_ids"] = cards
                .Select(static row => row.TryGetValue("id", out var id) ? TextOf(id) : string.Empty)
                .Where(static id => !string.IsNullOrWhiteSpace(id))
                .Distinct(StringComparer.Ordinal)
                .Count(),
            ["relics"] = relics.Count,
            ["potions"] = potions.Count,
            ["events"] = events.Count,
            ["creatures"] = creatures.Count,
            ["keywords"] = keywords.Count,
            ["enchantments"] = enchantments.Count,
            ["afflictions"] = afflictions.Count
        };

        var manifestPayload = new Dictionary<string, object?>
        {
            ["ok"] = true,
            ["exporter"] = "sts2-bridge",
            ["bridge_version"] = BridgeRuntime.BridgeVersion,
            ["game_assembly_version"] = BridgeRuntime.GameAssemblyVersion,
            ["generated_at_utc"] = generatedAtUtc.ToString("O", CultureInfo.InvariantCulture),
            ["output_dir"] = exportDir,
            ["items_path"] = itemsPath,
            ["counts"] = counts
        };

        var writeOptions = new JsonSerializerOptions
        {
            WriteIndented = true
        };

        File.WriteAllText(itemsPath, JsonSerializer.Serialize(itemsPayload, writeOptions));
        File.WriteAllText(manifestPath, JsonSerializer.Serialize(manifestPayload, writeOptions));

        return new
        {
            ok = true,
            exporter = "sts2-bridge",
            bridge_version = BridgeRuntime.BridgeVersion,
            game_assembly_version = BridgeRuntime.GameAssemblyVersion,
            generated_at_utc = generatedAtUtc.ToString("O", CultureInfo.InvariantCulture),
            output_dir = exportDir,
            items_path = itemsPath,
            manifest_path = manifestPath,
            counts
        };
    }

    private static string ResolveStaticExportDirectory(string? outputDir)
    {
        if (!string.IsNullOrWhiteSpace(outputDir))
        {
            return Path.IsPathRooted(outputDir)
                ? outputDir
                : Path.GetFullPath(Path.Combine(ResolveGameRootDirectory(), outputDir));
        }

        return Path.Combine(ResolveGameRootDirectory(), "export");
    }

    private static string ResolveGameRootDirectory()
    {
        try
        {
            var executablePath = OS.GetExecutablePath();
            if (!string.IsNullOrWhiteSpace(executablePath))
            {
                var dir = Path.GetDirectoryName(executablePath);
                if (!string.IsNullOrWhiteSpace(dir))
                {
                    return dir;
                }
            }
        }
        catch
        {
        }

        return Directory.GetCurrentDirectory();
    }

    private static List<Dictionary<string, object?>> BuildStaticCardRecords()
    {
        var rows = new List<Dictionary<string, object?>>();

        foreach (var baseCard in ModelDb.AllCards
                     .Where(static card => card is not null)
                     .OrderBy(static card => card.Id.ToString(), StringComparer.Ordinal))
        {
            var maxUpgradeLevel = GetHiddenPropertyValue<int>(baseCard, "MaxUpgradeLevel") ?? 0;
            string? previousCost = null;
            int? previousStarCost = null;
            string? previousDescription = null;
            string? previousEffect = null;
            string[]? previousKeywords = null;

            for (var upgradeLevel = 0; upgradeLevel <= Math.Max(0, maxUpgradeLevel); upgradeLevel++)
            {
                var card = baseCard.ToMutable();
                for (var i = 0; i < upgradeLevel; i++)
                {
                    card.UpgradeInternal();
                }

                var payload = JsonSerializer.SerializeToElement(BuildCardPayload(card));
                var name = TryGetNestedString(payload, "title");
                var type = TryGetNestedString(payload, "type");
                var target = TryGetNestedString(payload, "target_type");
                var cost = ResolveStaticCardCost(card, payload);
                var starCost = ResolveStaticCardStarCost(card, payload);
                var description = TryGetNestedString(payload, "description");
                var effect = TryGetNestedString(payload, "effect_preview", "summary");
                var keywords = BuildStaticCardKeywordNames(card);
                var keywordDetails = BuildStaticCardKeywordDetails(card);
                var (semanticTags, semanticSignals) = BuildStaticCardSemantics(
                    target,
                    cost,
                    starCost,
                    description,
                    effect,
                    keywords);
                var record = FilterEmptyValues(new Dictionary<string, object?>
                {
                    ["id"] = TryGetNestedString(payload, "id"),
                    ["name"] = name,
                    ["color"] = ResolveStaticCardColor(card),
                    ["rarity"] = TryGetNestedString(payload, "rarity"),
                    ["type"] = type,
                    ["target"] = target,
                    ["cost"] = cost,
                    ["starCost"] = starCost,
                    ["description"] = description,
                    ["effect"] = effect,
                    ["keywords"] = keywords,
                    ["keywordDetails"] = keywordDetails,
                    ["semanticTags"] = semanticTags.Length == 0 ? null : semanticTags,
                    ["semanticSignals"] = semanticSignals.Count == 0 ? null : semanticSignals,
                    ["canonicalText"] = BuildStaticCanonicalCardText(name, type, cost, starCost, target, description),
                    ["upgradeDiffFromPrevious"] = BuildStaticCardUpgradeDiff(
                        previousCost,
                        previousStarCost,
                        previousDescription,
                        previousEffect,
                        previousKeywords,
                        cost,
                        starCost,
                        description,
                        effect,
                        keywords),
                    ["upgrades"] = upgradeLevel,
                    ["maxUpgradeLevel"] = maxUpgradeLevel,
                    ["sourceAssembly"] = card.GetType().Assembly.GetName().Name
                });

                rows.Add(record);
                previousCost = cost;
                previousStarCost = starCost;
                previousDescription = description;
                previousEffect = effect;
                previousKeywords = keywords;
            }
        }

        return rows;
    }

    private static string[] BuildStaticCardKeywordNames(CardModel card)
    {
        return card.Keywords
            .Where(static keyword => keyword != CardKeyword.None)
            .Select(static keyword => keyword.ToString())
            .Distinct(StringComparer.Ordinal)
            .OrderBy(static keyword => keyword, StringComparer.Ordinal)
            .ToArray();
    }

    private static Dictionary<string, object?>[] BuildStaticCardKeywordDetails(CardModel card)
    {
        return card.Keywords
            .Where(static keyword => keyword != CardKeyword.None)
            .OrderBy(static keyword => keyword.ToString(), StringComparer.Ordinal)
            .Select(keyword =>
            {
                var hoverTip = HoverTipFactory.FromKeyword(keyword);
                return BuildStaticHoverTipRecord(
                    hoverTip,
                    keyword.ToString(),
                    nameof(CardKeyword));
            })
            .Where(static record => record.Count > 0)
            .ToArray();
    }

    private static (string[] Tags, Dictionary<string, object?> Signals) BuildStaticCardSemantics(
        string? targetType,
        string? cost,
        int? starCost,
        string? description,
        string? effect,
        IReadOnlyCollection<string> keywords)
    {
        const string LocalizedIntPattern = @"([0-9一二两三四五六七八九十百]+)";

        var descriptionText = NormalizeSemanticText(description ?? string.Empty);
        var effectText = NormalizeSemanticText(effect ?? string.Empty);
        var semanticText = string.Join(
            "\n",
            new[] { descriptionText, effectText }
                .Where(static text => !string.IsNullOrWhiteSpace(text))
                .Distinct(StringComparer.Ordinal));

        var tags = new HashSet<string>(StringComparer.Ordinal);
        var signals = new Dictionary<string, object?>(StringComparer.Ordinal);

        void AddTag(string tag)
        {
            if (!string.IsNullOrWhiteSpace(tag))
            {
                tags.Add(tag);
            }
        }

        void AddSignal(string key, int value)
        {
            if (value > 0)
            {
                signals[key] = value;
            }
        }

        int ResolveSignal(params string[] patterns)
        {
            var fromDescription = SumLocalizedMatches(descriptionText, patterns);
            return fromDescription > 0
                ? fromDescription
                : SumLocalizedMatches(effectText, patterns);
        }

        int ResolveSignalGroup(int groupIndex, params string[] patterns)
        {
            var fromDescription = SumLocalizedMatches(descriptionText, groupIndex, patterns);
            return fromDescription > 0
                ? fromDescription
                : SumLocalizedMatches(effectText, groupIndex, patterns);
        }

        foreach (var keyword in keywords)
        {
            switch (keyword)
            {
                case nameof(CardKeyword.Exhaust):
                    AddTag("exhaust_self");
                    break;
                case nameof(CardKeyword.Ethereal):
                    AddTag("ethereal");
                    break;
                case nameof(CardKeyword.Innate):
                    AddTag("innate");
                    break;
                case nameof(CardKeyword.Unplayable):
                    AddTag("unplayable");
                    break;
                case nameof(CardKeyword.Retain):
                    AddTag("retain");
                    break;
                case nameof(CardKeyword.Sly):
                    AddTag("sly");
                    break;
                case nameof(CardKeyword.Eternal):
                    AddTag("eternal");
                    break;
            }
        }

        if (string.Equals(cost, "X", StringComparison.OrdinalIgnoreCase))
        {
            AddTag("cost_x");
        }

        if (starCost is not null && starCost.Value < 0)
        {
            AddTag("star_x");
        }

        var damage = ResolveSignal(
            $@"造成\s*{LocalizedIntPattern}\s*点伤害",
            $@"deal\s*{LocalizedIntPattern}\s*damage");
        var block = ResolveSignal(
            $@"获得\s*{LocalizedIntPattern}\s*点格挡",
            $@"gain\s*{LocalizedIntPattern}\s*block");
        var draw = ResolveSignal(
            $@"抽\s*{LocalizedIntPattern}\s*张牌",
            $@"draw\s*{LocalizedIntPattern}\s*cards?");
        var discard = ResolveSignal(
            $@"(?:丢弃|弃掉|弃置)\s*{LocalizedIntPattern}\s*张牌",
            $@"discard\s*{LocalizedIntPattern}\s*cards?");
        var energyGain = ResolveSignal(
            $@"获得\s*{LocalizedIntPattern}\s*点能量(?!\s*时)",
            $@"gain\s*{LocalizedIntPattern}\s*energy");
        var starGain = ResolveSignal(
            $@"获得\s*{LocalizedIntPattern}\s*点星辉(?!\s*时)",
            $@"gain\s*{LocalizedIntPattern}\s*(?:star|stars|starlight)");
        var strengthGain = ResolveSignal(
            $@"获得\s*{LocalizedIntPattern}\s*点力量(?!\s*时)",
            $@"gain\s*{LocalizedIntPattern}\s*strength");
        var dexterityGain = ResolveSignal(
            $@"获得\s*{LocalizedIntPattern}\s*点敏捷(?!\s*时)",
            $@"gain\s*{LocalizedIntPattern}\s*dexterity");
        var focusGain = ResolveSignal(
            $@"获得\s*{LocalizedIntPattern}\s*点集中(?!\s*时)",
            $@"gain\s*{LocalizedIntPattern}\s*focus");
        var thornsGain = ResolveSignal(
            $@"获得\s*{LocalizedIntPattern}\s*点荆棘(?!\s*时)",
            $@"gain\s*{LocalizedIntPattern}\s*thorns");
        var intangibleGain = ResolveSignal(
            $@"获得\s*{LocalizedIntPattern}\s*层无实体(?!\s*时)",
            $@"gain\s*{LocalizedIntPattern}\s*intangible");
        var heal = ResolveSignal(
            $@"(?:恢复|回复|治疗)\s*{LocalizedIntPattern}\s*点(?:生命|生命值|血量|生命上限|最大生命|最大生命值)?",
            $@"heal\s*{LocalizedIntPattern}");
        var hpLoss = ResolveSignal(
            $@"失去\s*{LocalizedIntPattern}\s*点(?:生命|生命值|最大生命|最大生命值)",
            $@"受到\s*{LocalizedIntPattern}\s*点伤害",
            $@"lose\s*{LocalizedIntPattern}\s*(?:hp|health)",
            $@"take\s*{LocalizedIntPattern}\s*damage");
        var weak = ResolveSignal(
            $@"给予\s*{LocalizedIntPattern}\s*层虚弱",
            $@"apply\s*{LocalizedIntPattern}\s*weak");
        var vulnerable = ResolveSignal(
            $@"给予\s*{LocalizedIntPattern}\s*层易伤",
            $@"apply\s*{LocalizedIntPattern}\s*vulnerable");
        var poison = ResolveSignal(
            $@"给予\s*{LocalizedIntPattern}\s*层中毒",
            $@"apply\s*{LocalizedIntPattern}\s*poison");
        var calamity = ResolveSignal(
            $@"给予\s*{LocalizedIntPattern}\s*层灾厄",
            $@"apply\s*{LocalizedIntPattern}\s*calamity");
        var summon = ResolveSignal(
            $@"召唤\s*{LocalizedIntPattern}",
            $@"summon\s*{LocalizedIntPattern}");
        var forge = ResolveSignal(
            $@"铸造\s*{LocalizedIntPattern}",
            $@"forge\s*{LocalizedIntPattern}");
        var scry = ResolveSignal(
            $@"占卜\s*{LocalizedIntPattern}",
            $@"scry\s*{LocalizedIntPattern}");
        var potionGain = ResolveSignal(
            $@"获得\s*{LocalizedIntPattern}\s*瓶(?:随机)?药水",
            $@"gain\s*{LocalizedIntPattern}\s*potions?");
        var orbGeneration = ResolveSignal(
            $@"生成\s*{LocalizedIntPattern}\s*(?:个)?(?:闪电|冰霜|黑暗|随机)?充能球",
            $@"channel\s*{LocalizedIntPattern}\s*(?:orb|orbs)");
        var cardsToHand = ResolveSignal(
            $@"将\s*{LocalizedIntPattern}\s*张.*?(?:加入|放入)你的手牌",
            $@"add\s*{LocalizedIntPattern}\s*cards?\s*to\s*your\s*hand");

        AddSignal("damage", damage);
        AddSignal("block", block);
        AddSignal("draw", draw);
        AddSignal("discard", discard);
        AddSignal("energyGain", energyGain);
        AddSignal("starGain", starGain);
        AddSignal("strengthGain", strengthGain);
        AddSignal("dexterityGain", dexterityGain);
        AddSignal("focusGain", focusGain);
        AddSignal("thornsGain", thornsGain);
        AddSignal("intangibleGain", intangibleGain);
        AddSignal("heal", heal);
        AddSignal("hpLoss", hpLoss);
        AddSignal("weak", weak);
        AddSignal("vulnerable", vulnerable);
        AddSignal("poison", poison);
        AddSignal("calamity", calamity);
        AddSignal("summon", summon);
        AddSignal("forge", forge);
        AddSignal("scry", scry);
        AddSignal("potionGain", potionGain);
        AddSignal("orbGeneration", orbGeneration);
        AddSignal("cardsToHand", cardsToHand);

        var mentionsAllEnemies = ContainsAnyText(semanticText, "所有敌人", "all enemies");

        if (damage > 0)
        {
            AddTag(IsAllEnemiesTarget(targetType) || mentionsAllEnemies ? "aoe_damage" : "damage");
        }

        if (block > 0)
        {
            AddTag("block");
        }

        if (draw > 0 ||
            Regex.IsMatch(
                semanticText,
                $@"抽\s*{LocalizedIntPattern}\s*张牌|draw\s*{LocalizedIntPattern}\s*cards?",
                RegexOptions.IgnoreCase | RegexOptions.CultureInvariant))
        {
            AddTag("draw");
        }

        if (discard > 0 || ContainsAnyText(semanticText, "丢弃", "弃掉", "弃置", "discard"))
        {
            AddTag("discard");
        }

        if (energyGain > 0)
        {
            AddTag("gain_energy");
        }

        if (starGain > 0)
        {
            AddTag("gain_star");
        }

        if (strengthGain > 0)
        {
            AddTag("gain_strength");
        }

        if (dexterityGain > 0)
        {
            AddTag("gain_dexterity");
        }

        if (focusGain > 0)
        {
            AddTag("gain_focus");
        }

        if (thornsGain > 0)
        {
            AddTag("gain_thorns");
        }

        if (intangibleGain > 0 || ContainsAnyText(semanticText, "无实体", "intangible"))
        {
            AddTag("gain_intangible");
        }

        if (heal > 0 || ContainsAnyText(semanticText, "恢复生命", "回复生命", "治疗", "heal"))
        {
            AddTag("heal");
        }

        if (hpLoss > 0 || ContainsAnyText(semanticText, "失去生命", "受到", "lose hp", "take damage"))
        {
            AddTag("hp_loss");
        }

        if (weak > 0 || ContainsAnyText(semanticText, "虚弱", "weak"))
        {
            AddTag("apply_weak");
        }

        if (vulnerable > 0 || ContainsAnyText(semanticText, "易伤", "vulnerable"))
        {
            AddTag("apply_vulnerable");
        }

        if (poison > 0 || ContainsAnyText(semanticText, "中毒", "poison"))
        {
            AddTag("apply_poison");
        }

        if (calamity > 0 || ContainsAnyText(semanticText, "灾厄", "calamity"))
        {
            AddTag("apply_calamity");
        }

        if (summon > 0 || ContainsAnyText(semanticText, "召唤", "summon"))
        {
            AddTag("summon");
        }

        if (forge > 0 || ContainsAnyText(semanticText, "铸造", "forge"))
        {
            AddTag("forge");
        }

        if (scry > 0 || ContainsAnyText(semanticText, "占卜", "scry"))
        {
            AddTag("scry");
        }

        if (potionGain > 0 || ContainsAnyText(semanticText, "药水", "potion"))
        {
            AddTag("gain_potion");
        }

        if (orbGeneration > 0 || ContainsAnyText(semanticText, "充能球", "orb"))
        {
            AddTag("generate_orb");
        }

        if (cardsToHand > 0 || ContainsAnyText(semanticText, "加入你的手牌", "放入你的手牌", "to your hand"))
        {
            AddTag("add_to_hand");
        }

        if (ContainsAnyText(semanticText, "添加到你的手牌", "添加到你的抽牌堆", "添加到你的弃牌堆", "复制品", "随机牌加入", "add a copy", "shuffle a copy", "create"))
        {
            AddTag("generate_card");
        }

        if (ContainsAnyText(
                semanticText,
                "升级你的全部卡牌",
                "升级你手牌中的所有牌",
                "upgrade all your cards",
                "upgrade all cards in your hand"))
        {
            AddTag("upgrade_all");
        }
        else if (ContainsAnyText(semanticText, "升级", "upgrade"))
        {
            AddTag("upgrade_card");
        }

        if (ContainsAnyText(semanticText, "变化", "transform"))
        {
            AddTag("transform_card");
        }

        if (ContainsAnyText(semanticText, "从你的牌组中移除", "从你的牌组中选一张牌移除", "remove a card from your deck", "remove it from your deck"))
        {
            AddTag("remove_card");
        }

        if (ContainsAnyText(semanticText, "消耗。", "exhaust."))
        {
            AddTag("exhaust_self");
        }

        if (ContainsAnyText(semanticText, "消耗1张牌", "将其消耗", "选择一张牌将其消耗", "选择最多", "exhaust a card"))
        {
            AddTag("exhaust_other");
        }

        if (ContainsAnyText(semanticText, "保留", "retain"))
        {
            AddTag("retain");
        }

        if (ContainsAnyText(semanticText, "固有", "innate"))
        {
            AddTag("innate");
        }

        if (ContainsAnyText(semanticText, "虚无", "ethereal"))
        {
            AddTag("ethereal");
        }

        if (ContainsAnyText(semanticText, "不能被打出", "unplayable"))
        {
            AddTag("unplayable");
        }

        if (ContainsAnyText(semanticText, "随机", "random"))
        {
            AddTag("random");
        }

        if (Regex.IsMatch(
                semanticText,
                @"造成\s*[0-9一二两三四五六七八九十百]+\s*点伤害\s*(?:X|x|[0-9一二两三四五六七八九十百]+)\s*次|deal\s*[0-9]+\s*damage\s*(?:x|[0-9]+)\s*times",
                RegexOptions.IgnoreCase | RegexOptions.CultureInvariant))
        {
            AddTag("multi_hit");
        }

        if (Regex.IsMatch(
                semanticText,
                @"造成\s*[0-9一二两三四五六七八九十百]+\s*点伤害\s*(?:X|x)\s*次|deal\s*[0-9]+\s*damage\s*x\s*times",
                RegexOptions.IgnoreCase | RegexOptions.CultureInvariant))
        {
            AddTag("hits_x");
        }

        if ((IsAllEnemiesTarget(targetType) || mentionsAllEnemies) && damage > 0)
        {
            AddTag("aoe");
        }

        var numericHits = ResolveSignalGroup(
            2,
            $@"造成\s*{LocalizedIntPattern}\s*点伤害\s*{LocalizedIntPattern}\s*次",
            $@"deal\s*{LocalizedIntPattern}\s*damage\s*{LocalizedIntPattern}\s*times");
        if (numericHits > 0)
        {
            AddSignal("hits", numericHits);
        }

        return (tags.OrderBy(static tag => tag, StringComparer.Ordinal).ToArray(), signals);
    }

    private static bool IsAllEnemiesTarget(string? targetType)
    {
        return string.Equals(targetType, "AllEnemies", StringComparison.Ordinal) ||
               string.Equals(targetType, "AllEnemy", StringComparison.Ordinal);
    }

    private static bool ContainsAnyText(string text, params string[] needles)
    {
        if (string.IsNullOrWhiteSpace(text))
        {
            return false;
        }

        foreach (var needle in needles)
        {
            if (!string.IsNullOrWhiteSpace(needle) &&
                text.Contains(needle, StringComparison.OrdinalIgnoreCase))
            {
                return true;
            }
        }

        return false;
    }

    private static int SumLocalizedMatches(string text, params string[] patterns)
    {
        return SumLocalizedMatches(text, 1, patterns);
    }

    private static int SumLocalizedMatches(string text, int groupIndex, params string[] patterns)
    {
        if (string.IsNullOrWhiteSpace(text))
        {
            return 0;
        }

        var total = 0;
        foreach (var pattern in patterns)
        {
            if (string.IsNullOrWhiteSpace(pattern))
            {
                continue;
            }

            foreach (Match match in Regex.Matches(text, pattern, RegexOptions.IgnoreCase | RegexOptions.CultureInvariant))
            {
                if (!match.Success || match.Groups.Count <= groupIndex)
                {
                    continue;
                }

                var value = ParseLocalizedInt(match.Groups[groupIndex].Value);
                if (value > 0)
                {
                    total += value;
                }
            }
        }

        return total;
    }

    private static int ParseLocalizedInt(string? raw)
    {
        if (string.IsNullOrWhiteSpace(raw))
        {
            return 0;
        }

        var text = raw.Trim();
        if (int.TryParse(text, NumberStyles.Integer, CultureInfo.InvariantCulture, out var arabic))
        {
            return arabic;
        }

        text = text.Replace("两", "二", StringComparison.Ordinal);
        var digitMap = new Dictionary<char, int>
        {
            ['零'] = 0,
            ['一'] = 1,
            ['二'] = 2,
            ['三'] = 3,
            ['四'] = 4,
            ['五'] = 5,
            ['六'] = 6,
            ['七'] = 7,
            ['八'] = 8,
            ['九'] = 9
        };

        if (digitMap.TryGetValue(text[0], out var single) && text.Length == 1)
        {
            return single;
        }

        if (string.Equals(text, "十", StringComparison.Ordinal))
        {
            return 10;
        }

        var tenIndex = text.IndexOf('十');
        if (tenIndex >= 0)
        {
            var tens = tenIndex == 0 ? 1 : digitMap.GetValueOrDefault(text[tenIndex - 1], 0);
            var ones = tenIndex == text.Length - 1 ? 0 : digitMap.GetValueOrDefault(text[tenIndex + 1], 0);
            var value = tens * 10 + ones;
            if (value > 0)
            {
                return value;
            }
        }

        var hundredIndex = text.IndexOf('百');
        if (hundredIndex >= 0)
        {
            var hundreds = hundredIndex == 0 ? 1 : digitMap.GetValueOrDefault(text[hundredIndex - 1], 0);
            var remainderText = text[(hundredIndex + 1)..];
            return hundreds * 100 + ParseLocalizedInt(remainderText);
        }

        return 0;
    }

    private static Dictionary<string, object?>? BuildStaticCardUpgradeDiff(
        string? previousCost,
        int? previousStarCost,
        string? previousDescription,
        string? previousEffect,
        IReadOnlyCollection<string>? previousKeywords,
        string? currentCost,
        int? currentStarCost,
        string? currentDescription,
        string? currentEffect,
        IReadOnlyCollection<string> currentKeywords)
    {
        if (previousCost is null &&
            previousStarCost is null &&
            previousDescription is null &&
            previousEffect is null &&
            previousKeywords is null)
        {
            return null;
        }

        var addedKeywords = currentKeywords
            .Except(previousKeywords ?? Array.Empty<string>(), StringComparer.Ordinal)
            .OrderBy(static keyword => keyword, StringComparer.Ordinal)
            .ToArray();
        var removedKeywords = (previousKeywords ?? Array.Empty<string>())
            .Except(currentKeywords, StringComparer.Ordinal)
            .OrderBy(static keyword => keyword, StringComparer.Ordinal)
            .ToArray();

        var diff = FilterEmptyValues(new Dictionary<string, object?>
        {
            ["cost"] = previousCost != currentCost ? new { before = previousCost, after = currentCost } : null,
            ["starCost"] = previousStarCost != currentStarCost ? new { before = previousStarCost, after = currentStarCost } : null,
            ["description"] = !string.Equals(previousDescription, currentDescription, StringComparison.Ordinal)
                ? new { before = previousDescription, after = currentDescription }
                : null,
            ["effect"] = !string.Equals(previousEffect, currentEffect, StringComparison.Ordinal)
                ? new { before = previousEffect, after = currentEffect }
                : null,
            ["keywordsAdded"] = addedKeywords,
            ["keywordsRemoved"] = removedKeywords
        });

        return diff.Count == 0 ? null : diff;
    }

    private static string BuildStaticCanonicalCardText(
        string? title,
        string? type,
        string? cost,
        int? starCost,
        string? targetType,
        string? description)
    {
        var sb = new StringBuilder("卡牌｜");
        sb.Append(NormalizeSemanticText(title ?? string.Empty));

        if (!string.IsNullOrWhiteSpace(type))
        {
            sb.Append("｜").Append(type);
        }

        if (!string.IsNullOrWhiteSpace(cost))
        {
            sb.Append("｜能量").Append(cost);
        }

        if (starCost is not null)
        {
            sb.Append(starCost.Value < 0 ? "｜星辉X" : $"｜星辉{starCost.Value}");
        }

        if (!string.IsNullOrWhiteSpace(targetType))
        {
            sb.Append("｜目标").Append(TranslateTargetType(targetType));
        }

        if (!string.IsNullOrWhiteSpace(description))
        {
            sb.Append("｜效果：").Append(NormalizeSemanticText(description));
        }

        return sb.ToString();
    }

    private static string ResolveStaticCardColor(CardModel card)
    {
        var visualPool = GetHiddenPropertyObjectValue(card, "VisualCardPool");
        return FirstNonEmptyText(
            NormalizeStaticToken(TryGetNamedValueText(visualPool!, "EnergyColorName")),
            NormalizeStaticToken(TryGetTitle(visualPool ?? card)),
            NormalizeStaticToken(TryGetNamedValueText(card, "Color")),
            string.Empty);
    }

    private static string ResolveStaticCardCost(CardModel card, JsonElement payload)
    {
        if (TryGetNestedBool(payload, "costs_x") == true || card.EnergyCost.CostsX)
        {
            return "X";
        }

        var canonical = TryGetNestedInt(payload, "canonical_energy_cost") ?? card.EnergyCost.Canonical;
        return canonical < 0 ? string.Empty : canonical.ToString(CultureInfo.InvariantCulture);
    }

    private static int? ResolveStaticCardStarCost(CardModel card, JsonElement payload)
    {
        if (TryGetNestedBool(payload, "has_star_cost_x") == true || card.HasStarCostX)
        {
            return -1;
        }

        var canonical = TryGetNestedInt(payload, "canonical_star_cost") ?? card.CanonicalStarCost;
        return canonical < 0 ? null : canonical;
    }

    private static List<Dictionary<string, object?>> BuildStaticRelicRecords()
    {
        return ModelDb.AllRelics
            .Where(static relic => relic is not null)
            .OrderBy(static relic => relic.Id.ToString(), StringComparer.Ordinal)
            .Select(relic =>
            {
                var payload = JsonSerializer.SerializeToElement(BuildRelicPayload(relic));
                var containingPools = ModelDb.AllRelicPools
                    .Where(pool => pool.AllRelicIds.Contains(relic.Id))
                    .OrderBy(pool => pool.Id.ToString(), StringComparer.Ordinal)
                    .Select(BuildStaticRelicPoolRecord)
                    .ToArray();
                var characterOwners = ModelDb.AllCharacters
                    .Where(character => character.RelicPool.AllRelicIds.Contains(relic.Id))
                    .OrderBy(character => character.Id.ToString(), StringComparer.Ordinal)
                    .Select(character => FilterEmptyValues(new Dictionary<string, object?>
                    {
                        ["id"] = character.Id.ToString(),
                        ["name"] = TryGetTitle(character)
                    }))
                    .ToArray();

                return FilterEmptyValues(new Dictionary<string, object?>
                {
                    ["id"] = TryGetNestedString(payload, "id"),
                    ["name"] = TryGetNestedString(payload, "title"),
                    ["description"] = TryGetNestedString(payload, "description"),
                    ["flavor"] = DescribeText(relic.Flavor, relic),
                    ["rarity"] = TryGetNestedString(payload, "rarity"),
                    ["merchantCost"] = relic.MerchantCost,
                    ["isTradable"] = relic.IsTradable,
                    ["hasUponPickupEffect"] = relic.HasUponPickupEffect,
                    ["spawnsPets"] = relic.SpawnsPets,
                    ["addsPet"] = relic.AddsPet,
                    ["isStackable"] = relic.IsStackable,
                    ["pools"] = containingPools,
                    ["characterOwners"] = characterOwners,
                    ["canonicalText"] = BuildCanonicalRelicText(
                        TryGetNestedString(payload, "title"),
                        TryGetNestedString(payload, "rarity"),
                        TryGetNestedString(payload, "description")),
                    ["sourceAssembly"] = relic.GetType().Assembly.GetName().Name
                });
            })
            .ToList();
    }

    private static Dictionary<string, object?> BuildStaticRelicPoolRecord(RelicPoolModel pool)
    {
        var rawName = TryGetTitle(pool);
        return FilterEmptyValues(new Dictionary<string, object?>
        {
            ["id"] = pool.Id.ToString(),
            ["name"] = rawName.Contains('(') ? pool.GetType().Name : rawName,
            ["type"] = pool.GetType().Name,
            ["energyColorName"] = pool.EnergyColorName
        });
    }

    private static List<Dictionary<string, object?>> BuildStaticPotionRecords()
    {
        return ModelDb.AllPotions
            .Where(static potion => potion is not null)
            .OrderBy(static potion => potion.Id.ToString(), StringComparer.Ordinal)
            .Select(static potion =>
            {
                var payload = JsonSerializer.SerializeToElement(BuildPotionPayload(potion));
                return FilterEmptyValues(new Dictionary<string, object?>
                {
                    ["id"] = TryGetNestedString(payload, "id"),
                    ["name"] = TryGetNestedString(payload, "title"),
                    ["description"] = TryGetNestedString(payload, "description"),
                    ["rarity"] = TryGetNestedString(payload, "rarity"),
                    ["target"] = TryGetNestedString(payload, "target_type"),
                    ["selectionPrompt"] = TryGetNestedString(payload, "selection_screen_prompt"),
                    ["canonicalText"] = BuildCanonicalPotionText(
                        TryGetNestedString(payload, "title"),
                        TryGetNestedString(payload, "rarity"),
                        TryGetNestedString(payload, "target_type"),
                        TryGetNestedString(payload, "description")),
                    ["sourceAssembly"] = potion.GetType().Assembly.GetName().Name
                });
            })
            .ToList();
    }

    private static List<Dictionary<string, object?>> BuildStaticEventRecords()
    {
        return ModelDb.AllEvents
            .Where(static evt => evt is not null)
            .OrderBy(static evt => evt.Id.ToString(), StringComparer.Ordinal)
            .Select(evt => FilterEmptyValues(new Dictionary<string, object?>
            {
                ["id"] = evt.Id.ToString(),
                ["name"] = TryGetTitle(evt),
                ["description"] = DescribeText(evt.InitialDescription, evt),
                ["options"] = evt.GameInfoOptions
                    .Reverse()
                    .Select(option => DescribeText(option, evt))
                    .Where(static text => !string.IsNullOrWhiteSpace(text))
                    .ToArray(),
                ["layoutType"] = evt.LayoutType.ToString(),
                ["isShared"] = evt.IsShared,
                ["isDeterministic"] = evt.IsDeterministic,
                ["canonicalEncounterId"] = evt.CanonicalEncounter?.Id.ToString(),
                ["optionRecords"] = BuildStaticEventOptionLocRecords(evt),
                ["generatedInitialOptions"] = TryBuildStaticGeneratedEventOptions(evt),
                ["sourceAssembly"] = evt.GetType().Assembly.GetName().Name
            }))
            .ToList();
    }

    private static Dictionary<string, object?>[] BuildStaticEventOptionLocRecords(EventModel evt)
    {
        try
        {
            var grouped = new Dictionary<string, Dictionary<string, object?>>(StringComparer.Ordinal);
            foreach (var option in evt.GameInfoOptions.Reverse())
            {
                var rawKey = option.LocEntryKey;
                var baseKey = StripEventOptionLocSuffix(rawKey, out var kind);
                if (!grouped.TryGetValue(baseKey, out var record))
                {
                    record = new Dictionary<string, object?>
                    {
                        ["textKey"] = baseKey,
                        ["page"] = ExtractEventPage(baseKey),
                        ["optionKey"] = ExtractEventOptionKey(baseKey)
                    };
                    grouped[baseKey] = record;
                }

                var text = DescribeText(option, evt);
                if (string.IsNullOrWhiteSpace(text))
                {
                    continue;
                }

                switch (kind)
                {
                    case "title":
                        record["title"] = text;
                        break;
                    case "description":
                        record["description"] = text;
                        break;
                    default:
                        record["text"] = text;
                        break;
                }
            }

            return grouped.Values
                .Select(record =>
                {
                    var textKey = record.TryGetValue("textKey", out var value) ? TextOf(value) : string.Empty;
                    record.TryAdd("title", DescribeText(evt.GetOptionTitle(textKey), evt));
                    record.TryAdd("description", DescribeText(evt.GetOptionDescription(textKey), evt));
                    return FilterEmptyValues(record);
                })
                .Where(static record => record.Count > 0)
                .ToArray();
        }
        catch
        {
            return [];
        }
    }

    private static string StripEventOptionLocSuffix(string? textKey, out string kind)
    {
        if (string.IsNullOrWhiteSpace(textKey))
        {
            kind = string.Empty;
            return string.Empty;
        }

        if (textKey.EndsWith(".title", StringComparison.Ordinal))
        {
            kind = "title";
            return textKey[..^".title".Length];
        }

        if (textKey.EndsWith(".description", StringComparison.Ordinal))
        {
            kind = "description";
            return textKey[..^".description".Length];
        }

        kind = "text";
        return textKey;
    }

    private static Dictionary<string, object?>[]? TryBuildStaticGeneratedEventOptions(EventModel evt)
    {
        try
        {
            var mutable = evt.ToMutable();
            var method = FindMethod(mutable.GetType(), "GenerateInitialOptionsWrapper", 0);
            if (method?.Invoke(mutable, Array.Empty<object?>()) is not IEnumerable generatedOptions)
            {
                return null;
            }

            return generatedOptions
                .OfType<EventOption>()
                .Select(option => BuildStaticGeneratedEventOptionRecord(mutable, option))
                .Where(static record => record.Count > 0)
                .ToArray();
        }
        catch
        {
            return null;
        }
    }

    private static Dictionary<string, object?> BuildStaticGeneratedEventOptionRecord(EventModel evt, EventOption option)
    {
        var hoverTips = (option.HoverTips ?? Array.Empty<IHoverTip>())
            .Cast<object>()
            .Select(hoverTip => BuildStaticHoverTipRecord(hoverTip))
            .Where(static record => record.Count > 0)
            .ToArray();

        return FilterEmptyValues(new Dictionary<string, object?>
        {
            ["textKey"] = option.TextKey,
            ["page"] = ExtractEventPage(option.TextKey),
            ["optionKey"] = ExtractEventOptionKey(option.TextKey),
            ["title"] = DescribeText(option.Title, evt),
            ["description"] = DescribeText(option.Description, evt),
            ["isLocked"] = option.IsLocked,
            ["isProceed"] = option.IsProceed,
            ["historyName"] = DescribeText(option.HistoryName, evt),
            ["shouldSaveChoiceToHistory"] = option.ShouldSaveChoiceToHistory,
            ["shouldSaveVariablesToHistory"] = option.ShouldSaveVariablesToHistory,
            ["hasWillKillPlayerCheck"] = option.WillKillPlayer is not null,
            ["relic"] = option.Relic is null
                ? null
                : FilterEmptyValues(new Dictionary<string, object?>
                {
                    ["id"] = option.Relic.Id.ToString(),
                    ["name"] = TryGetTitle(option.Relic),
                    ["rarity"] = option.Relic.Rarity.ToString()
                }),
            ["hoverTips"] = hoverTips
        });
    }

    private static string ExtractEventPage(string? textKey)
    {
        if (string.IsNullOrWhiteSpace(textKey))
        {
            return string.Empty;
        }

        const string pageMarker = ".pages.";
        const string optionMarker = ".options.";
        var pageStart = textKey.IndexOf(pageMarker, StringComparison.Ordinal);
        var optionStart = textKey.IndexOf(optionMarker, StringComparison.Ordinal);
        if (pageStart < 0 || optionStart <= pageStart)
        {
            return string.Empty;
        }

        pageStart += pageMarker.Length;
        return textKey[pageStart..optionStart];
    }

    private static string ExtractEventOptionKey(string? textKey)
    {
        if (string.IsNullOrWhiteSpace(textKey))
        {
            return string.Empty;
        }

        const string optionMarker = ".options.";
        var optionStart = textKey.IndexOf(optionMarker, StringComparison.Ordinal);
        if (optionStart < 0)
        {
            return string.Empty;
        }

        optionStart += optionMarker.Length;
        return optionStart >= textKey.Length ? string.Empty : textKey[optionStart..];
    }

    private static List<Dictionary<string, object?>> BuildStaticCreatureRecords()
    {
        var records = new List<Dictionary<string, object?>>();

        records.AddRange(ModelDb.AllCharacters
            .Where(static character => character is not null)
            .OrderBy(static character => character.Id.ToString(), StringComparer.Ordinal)
            .Select(character => FilterEmptyValues(new Dictionary<string, object?>
            {
                ["id"] = character.Id.ToString(),
                ["name"] = TryGetTitle(character),
                ["type"] = "Player",
                ["minHP"] = character.StartingHp,
                ["maxHP"] = character.StartingHp,
                ["startingGold"] = character.StartingGold,
                ["maxEnergy"] = character.MaxEnergy,
                ["cardPoolId"] = character.CardPool.Id.ToString(),
                ["relicPoolId"] = character.RelicPool.Id.ToString(),
                ["potionPoolId"] = character.PotionPool.Id.ToString(),
                ["startingDeckIds"] = character.StartingDeck.Select(card => card.Id.ToString()).ToArray(),
                ["startingRelicIds"] = character.StartingRelics.Select(relic => relic.Id.ToString()).ToArray(),
                ["startingPotionIds"] = character.StartingPotions.Select(potion => potion.Id.ToString()).ToArray(),
                ["sourceAssembly"] = character.GetType().Assembly.GetName().Name
            })));

        records.AddRange(ModelDb.Monsters
            .Where(static monster => monster is not null)
            .OrderBy(static monster => monster.Id.ToString(), StringComparer.Ordinal)
            .Select(monster => FilterEmptyValues(new Dictionary<string, object?>
            {
                ["id"] = monster.Id.ToString(),
                ["name"] = TryGetTitle(monster),
                ["type"] = FirstNonEmptyText(
                    TryGetNamedValueText(monster, "Type"),
                    monster.GetType().Name),
                ["minHP"] = monster.MinInitialHp,
                ["maxHP"] = monster.MaxInitialHp,
                ["moveTemplates"] = BuildStaticMonsterMoveTemplates(monster),
                ["stateMachine"] = BuildStaticMonsterStateMachineSummary(monster),
                ["sourceAssembly"] = monster.GetType().Assembly.GetName().Name
            })));

        return records;
    }

    private static Dictionary<string, object?>[]? BuildStaticMonsterMoveTemplates(MonsterModel monster)
    {
        var stateMachine = TryBuildStaticMonsterStateMachine(monster);
        if (stateMachine is null)
        {
            return null;
        }

        return stateMachine.States.Values
            .OfType<MoveState>()
            .OrderBy(static move => move.Id, StringComparer.Ordinal)
            .Select(move => BuildStaticMonsterMoveTemplate(monster, move))
            .Where(static record => record.Count > 0)
            .ToArray();
    }

    private static Dictionary<string, object?>? BuildStaticMonsterStateMachineSummary(MonsterModel monster)
    {
        var stateMachine = TryBuildStaticMonsterStateMachine(monster);
        if (stateMachine is null)
        {
            return null;
        }

        var initialState = GetHiddenFieldValue(stateMachine, "_initialState") as MonsterState;
        var states = stateMachine.States.Values
            .OrderBy(static state => state.Id, StringComparer.Ordinal)
            .Select(state => BuildStaticMonsterStateRecord(monster, state))
            .Where(static record => record.Count > 0)
            .ToArray();

        return FilterEmptyValues(new Dictionary<string, object?>
        {
            ["initialStateId"] = initialState?.Id,
            ["states"] = states
        });
    }

    private static MonsterMoveStateMachine? TryBuildStaticMonsterStateMachine(MonsterModel monster)
    {
        try
        {
            var mutable = monster.ToMutable();
            var method = FindMethod(mutable.GetType(), "GenerateMoveStateMachine", 0);
            return method?.Invoke(mutable, Array.Empty<object?>()) as MonsterMoveStateMachine;
        }
        catch
        {
            return null;
        }
    }

    private static Dictionary<string, object?> BuildStaticMonsterStateRecord(MonsterModel monster, MonsterState state)
    {
        var record = new Dictionary<string, object?>
        {
            ["id"] = state.Id,
            ["kind"] = ResolveStaticMonsterStateKind(state),
            ["shouldAppearInLogs"] = state.ShouldAppearInLogs
        };

        if (state is MoveState moveState)
        {
            record["title"] = ResolveStaticMonsterMoveTitle(monster, moveState.Id);
            record["description"] = ResolveStaticMonsterMoveDescription(monster, moveState.Id);
            record["followUpStateId"] = moveState.FollowUpState?.Id ?? moveState.FollowUpStateId;
            record["mustPerformOnceBeforeTransitioning"] = moveState.MustPerformOnceBeforeTransitioning;
            record["intents"] = moveState.Intents
                .Select(BuildStaticMonsterIntentTemplate)
                .Where(static intent => intent.Count > 0)
                .ToArray();
        }
        else if (state is RandomBranchState randomBranchState)
        {
            record["branches"] = BuildStaticRandomBranchRecords(randomBranchState);
        }
        else if (state is ConditionalBranchState conditionalBranchState)
        {
            record["branches"] = BuildStaticConditionalBranchRecords(conditionalBranchState);
        }

        return FilterEmptyValues(record);
    }

    private static Dictionary<string, object?> BuildStaticMonsterMoveTemplate(MonsterModel monster, MoveState moveState)
    {
        return FilterEmptyValues(new Dictionary<string, object?>
        {
            ["stateId"] = moveState.Id,
            ["title"] = ResolveStaticMonsterMoveTitle(monster, moveState.Id),
            ["description"] = ResolveStaticMonsterMoveDescription(monster, moveState.Id),
            ["followUpStateId"] = moveState.FollowUpState?.Id ?? moveState.FollowUpStateId,
            ["mustPerformOnceBeforeTransitioning"] = moveState.MustPerformOnceBeforeTransitioning,
            ["intents"] = moveState.Intents
                .Select(BuildStaticMonsterIntentTemplate)
                .Where(static intent => intent.Count > 0)
                .ToArray()
        });
    }

    private static string ResolveStaticMonsterStateKind(MonsterState state)
    {
        return state switch
        {
            MoveState => "move",
            RandomBranchState => "random_branch",
            ConditionalBranchState => "conditional_branch",
            _ => NormalizeStaticToken(state.GetType().Name)
        };
    }

    private static string ResolveStaticMonsterMoveTitle(MonsterModel monster, string moveId)
    {
        return DescribeText(LocString.GetIfExists("monsters", $"{monster.Id.Entry}.moves.{moveId}.title"), monster);
    }

    private static string ResolveStaticMonsterMoveDescription(MonsterModel monster, string moveId)
    {
        return DescribeText(LocString.GetIfExists("monsters", $"{monster.Id.Entry}.moves.{moveId}.description"), monster);
    }

    private static Dictionary<string, object?> BuildStaticMonsterIntentTemplate(AbstractIntent intent)
    {
        var repeats = intent switch
        {
            SingleAttackIntent singleAttackIntent => singleAttackIntent.Repeats,
            MultiAttackIntent multiAttackIntent => multiAttackIntent.Repeats,
            _ => 1
        };
        var baseDamage = intent is AttackIntent attackIntent
            ? TryBuildStaticAttackIntentDamage(attackIntent)
            : null;
        var totalBaseDamage = baseDamage.HasValue && repeats > 1
            ? baseDamage.Value * repeats
            : baseDamage;

        return FilterEmptyValues(new Dictionary<string, object?>
        {
            ["intentType"] = intent.IntentType.ToString(),
            ["intentClass"] = intent.GetType().Name,
            ["title"] = GetMonsterIntentTitle(intent),
            ["repeats"] = repeats > 1 ? repeats : null,
            ["baseDamage"] = baseDamage,
            ["totalBaseDamage"] = repeats > 1 ? totalBaseDamage : null,
            ["statusCardCount"] = intent is StatusIntent statusIntent ? statusIntent.CardCount : null,
            ["hasTip"] = intent.HasIntentTip
        });
    }

    private static int? TryBuildStaticAttackIntentDamage(AttackIntent intent)
    {
        try
        {
            return intent.DamageCalc is null
                ? null
                : Math.Max(0, (int)intent.DamageCalc.Invoke());
        }
        catch
        {
            return null;
        }
    }

    private static Dictionary<string, object?>[] BuildStaticRandomBranchRecords(RandomBranchState randomBranchState)
    {
        return randomBranchState.States
            .Select(stateWeight => FilterEmptyValues(new Dictionary<string, object?>
            {
                ["stateId"] = stateWeight.stateId,
                ["repeatType"] = stateWeight.repeatType.ToString(),
                ["maxTimes"] = stateWeight.repeatType == MoveRepeatType.CanRepeatXTimes ? stateWeight.maxTimes : null,
                ["cooldown"] = stateWeight.cooldown > 0 ? stateWeight.cooldown : null,
                ["weight"] = TryBuildStaticRandomBranchWeight(stateWeight)
            }))
            .Where(static record => record.Count > 0)
            .ToArray();
    }

    private static float? TryBuildStaticRandomBranchWeight(RandomBranchState.StateWeight stateWeight)
    {
        try
        {
            return stateWeight.GetWeight();
        }
        catch
        {
            return null;
        }
    }

    private static Dictionary<string, object?>[] BuildStaticConditionalBranchRecords(ConditionalBranchState conditionalBranchState)
    {
        if (GetHiddenPropertyObjectValue(conditionalBranchState, "States") is not IEnumerable branches)
        {
            return [];
        }

        var records = new List<Dictionary<string, object?>>();
        foreach (var branch in branches)
        {
            if (branch is null)
            {
                continue;
            }

            records.Add(FilterEmptyValues(new Dictionary<string, object?>
            {
                ["stateId"] = TextOf(GetHiddenFieldValue(branch, "id"))
            }));
        }

        return records
            .Where(static record => record.Count > 0)
            .ToArray();
    }

    private static List<Dictionary<string, object?>> BuildStaticKeywordRecords()
    {
        var records = new List<Dictionary<string, object?>>();
        var seen = new HashSet<string>(StringComparer.Ordinal);

        void AddHoverTip(object? hoverTip, string sourceType)
        {
            if (hoverTip is null)
            {
                return;
            }

            var record = BuildStaticHoverTipRecord(hoverTip, sourceType: sourceType);
            var id = record.TryGetValue("id", out var idValue) ? TextOf(idValue) : string.Empty;
            var title = record.TryGetValue("name", out var nameValue) ? TextOf(nameValue) : string.Empty;
            var description = record.TryGetValue("description", out var descriptionValue) ? TextOf(descriptionValue) : string.Empty;
            var dedupeKey = string.Join(
                "|",
                NormalizeComparableText(id),
                NormalizeComparableText(title),
                NormalizeComparableText(description));

            if (!seen.Add(dedupeKey))
            {
                return;
            }

            records.Add(record);
        }

        foreach (CardKeyword keyword in Enum.GetValues<CardKeyword>())
        {
            AddHoverTip(HoverTipFactory.FromKeyword(keyword), nameof(CardKeyword));
        }

        foreach (StaticHoverTip tip in Enum.GetValues<StaticHoverTip>())
        {
            if (tip.ToString().EndsWith("Dynamic", StringComparison.Ordinal))
            {
                continue;
            }

            AddHoverTip(HoverTipFactory.Static(tip), nameof(StaticHoverTip));
        }

        foreach (var power in ModelDb.AllPowers.Where(static power => power is not null))
        {
            AddHoverTip(power.DumbHoverTip, power.GetType().Name);
        }

        return records
            .OrderBy(static row => row.TryGetValue("id", out var id) ? TextOf(id) : string.Empty, StringComparer.Ordinal)
            .ThenBy(static row => row.TryGetValue("name", out var name) ? TextOf(name) : string.Empty, StringComparer.Ordinal)
            .ToList();
    }

    private static Dictionary<string, object?> BuildStaticHoverTipRecord(
        object? hoverTip,
        string? fallbackId = null,
        string? sourceType = null)
    {
        if (hoverTip is null)
        {
            return [];
        }

        var canonicalModel = GetHiddenPropertyObjectValue(hoverTip, "CanonicalModel") as AbstractModel;
        var rawId = TextOf(GetHiddenPropertyObjectValue(hoverTip, "Id"));
        var cleanedRawId = rawId.StartsWith("LocString with ", StringComparison.Ordinal)
            ? string.Empty
            : rawId;
        var id = FirstNonEmptyText(
            canonicalModel?.Id.ToString() ?? string.Empty,
            cleanedRawId,
            fallbackId ?? string.Empty);
        var title = ResolveHoverTipTitle(hoverTip, canonicalModel, id);
        var description = ResolveHoverTipDescription(hoverTip, canonicalModel);

        return FilterEmptyValues(new Dictionary<string, object?>
        {
            ["id"] = id,
            ["name"] = title,
            ["description"] = description,
            ["type"] = sourceType,
            ["sourceAssembly"] = hoverTip.GetType().Assembly.GetName().Name
        });
    }

    private static List<Dictionary<string, object?>> BuildStaticEnchantmentRecords()
    {
        var models = GetHiddenPropertyObjectValue(typeof(ModelDb), "DebugEnchantments") as IEnumerable;
        return BuildStaticNamedDescriptionRecords(models);
    }

    private static List<Dictionary<string, object?>> BuildStaticAfflictionRecords()
    {
        var models = GetHiddenPropertyObjectValue(typeof(ModelDb), "DebugAfflictions") as IEnumerable;
        return BuildStaticNamedDescriptionRecords(models);
    }

    private static List<Dictionary<string, object?>> BuildStaticNamedDescriptionRecords(IEnumerable? models)
    {
        if (models is null)
        {
            return [];
        }

        var records = new List<Dictionary<string, object?>>();
        foreach (var model in models)
        {
            if (model is null)
            {
                continue;
            }

            records.Add(FilterEmptyValues(new Dictionary<string, object?>
            {
                ["id"] = TextOf(GetHiddenPropertyObjectValue(model, "Id")),
                ["name"] = TryGetTitle(model),
                ["description"] = TryGetDescription(model),
                ["sourceAssembly"] = model.GetType().Assembly.GetName().Name
            }));
        }

        return records;
    }

    private static string NormalizeStaticToken(string value)
    {
        if (string.IsNullOrWhiteSpace(value))
        {
            return string.Empty;
        }

        return value
            .Trim()
            .Replace(" ", "_", StringComparison.Ordinal)
            .ToLowerInvariant();
    }

    private static Dictionary<string, object?> FilterEmptyValues(Dictionary<string, object?> payload)
    {
        var filtered = new Dictionary<string, object?>(StringComparer.Ordinal);
        foreach (var (key, value) in payload)
        {
            switch (value)
            {
                case null:
                    continue;
                case string text when string.IsNullOrWhiteSpace(text):
                    continue;
                case Array array when array.Length == 0:
                    continue;
                case IEnumerable<object> enumerable when !enumerable.Any():
                    continue;
                default:
                    filtered[key] = value;
                    break;
            }
        }

        return filtered;
    }
}
