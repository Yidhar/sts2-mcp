using System.Collections.Generic;
using System.Linq;
using MegaCrit.Sts2.Core.Map;
using MegaCrit.Sts2.Core.Nodes.Screens.Map;

namespace Sts2McpBridge.Scripts;

internal static partial class BridgeGameApi
{
    private static object? BuildEnvMapRoutePayload(BridgeWorldContext context, MapCoord startCoord)
    {
        if (context.RunState?.Map is null || context.MapPoints.Count == 0)
        {
            return null;
        }

        var currentRow = context.RunState.CurrentMapCoord?.row ?? int.MinValue;
        var descriptors = BuildEnvMapRouteDescriptors(context.MapPoints, currentRow);
        var startKey = ToEnvMapCoordKey(startCoord);
        if (!descriptors.TryGetValue(startKey, out var start))
        {
            return null;
        }

        var bestStepsByKey = new Dictionary<string, int>(StringComparer.Ordinal);
        var stack = new Stack<(string Key, int Steps)>();
        stack.Push((startKey, 1));

        while (stack.Count > 0)
        {
            var (key, steps) = stack.Pop();
            if (bestStepsByKey.TryGetValue(key, out var bestKnown) && bestKnown <= steps)
            {
                continue;
            }

            if (!descriptors.TryGetValue(key, out var point))
            {
                continue;
            }

            bestStepsByKey[key] = steps;
            foreach (var childKey in GetEnvReachableMapChildKeys(point, descriptors))
            {
                stack.Push((childKey, steps + 1));
            }
        }

        if (bestStepsByKey.Count == 0)
        {
            return null;
        }

        var counts = new Dictionary<string, int>(StringComparer.Ordinal)
        {
            ["Monster"] = 0,
            ["Elite"] = 0,
            ["Boss"] = 0,
            ["Event"] = 0,
            ["QuestionMark"] = 0,
            ["RestSite"] = 0,
            ["Shop"] = 0,
            ["Treasure"] = 0
        };
        var minSteps = new Dictionary<string, int>(StringComparer.Ordinal);

        foreach (var (key, steps) in bestStepsByKey)
        {
            var point = descriptors[key];
            if (counts.ContainsKey(point.PointType))
            {
                counts[point.PointType]++;
            }

            if (!minSteps.TryGetValue(point.PointType, out var existing) || steps < existing)
            {
                minSteps[point.PointType] = steps;
            }
        }

        var orderedNodes = bestStepsByKey
            .OrderBy(static pair => pair.Value)
            .ThenBy(pair => descriptors[pair.Key].Coord.row)
            .ThenBy(pair => descriptors[pair.Key].Coord.col)
            .Select(pair =>
            {
                var point = descriptors[pair.Key];
                var childCount = GetEnvReachableMapChildKeys(point, descriptors).Count;
                return new
                {
                    coord = BuildMapCoord(point.Coord),
                    point_type = point.PointType,
                    depth = pair.Value,
                    child_count = childCount,
                    is_leaf = childCount == 0
                };
            })
            .ToArray();

        return new
        {
            reachable_node_count = bestStepsByKey.Count,
            max_depth = bestStepsByKey.Values.Max(),
            direct_child_count = GetEnvReachableMapChildKeys(start, descriptors).Count,
            forced_path_steps_before_branch = CountForcedEnvMapPathSteps(startKey, descriptors),
            count_monster = counts["Monster"],
            count_elite = counts["Elite"],
            count_boss = counts["Boss"],
            count_event = counts["Event"],
            count_question_mark = counts["QuestionMark"],
            count_rest_site = counts["RestSite"],
            count_shop = counts["Shop"],
            count_treasure = counts["Treasure"],
            next_elite_steps = minSteps.TryGetValue("Elite", out var eliteSteps) ? eliteSteps : (int?)null,
            next_rest_steps = minSteps.TryGetValue("RestSite", out var restSteps) ? restSteps : (int?)null,
            next_shop_steps = minSteps.TryGetValue("Shop", out var shopSteps) ? shopSteps : (int?)null,
            next_event_steps = minSteps.TryGetValue("Event", out var eventSteps) ? eventSteps : (int?)null,
            next_question_mark_steps = minSteps.TryGetValue("QuestionMark", out var questionSteps) ? questionSteps : (int?)null,
            next_treasure_steps = minSteps.TryGetValue("Treasure", out var treasureSteps) ? treasureSteps : (int?)null,
            next_boss_steps = minSteps.TryGetValue("Boss", out var bossSteps) ? bossSteps : (int?)null,
            can_reach_rest_site_before_elite = CanReachEnvMapTypeBeforeType(startKey, descriptors, "RestSite", "Elite"),
            can_reach_elite_then_rest_site = CanReachEnvMapEliteThenRestSite(startKey, descriptors),
            nodes = orderedNodes
        };
    }

    private static Dictionary<string, EnvMapRouteNode> BuildEnvMapRouteDescriptors(
        IReadOnlyList<NMapPoint> mapPoints,
        int currentRow)
    {
        var descriptors = new Dictionary<string, EnvMapRouteNode>(StringComparer.Ordinal);
        foreach (var pointNode in mapPoints)
        {
            var point = pointNode.Point;
            var coord = point.coord;
            if (coord.row <= currentRow)
            {
                continue;
            }

            var key = ToEnvMapCoordKey(coord);
            descriptors[key] = new EnvMapRouteNode
            {
                Coord = coord,
                PointType = NormalizeEnvMapPointType(point.PointType.ToString()),
                ChildKeys = point.Children
                    .Select(static child => child.coord)
                    .Where(child => child.row > currentRow)
                    .Select(ToEnvMapCoordKey)
                    .Distinct(StringComparer.Ordinal)
                    .OrderBy(static childKey => childKey, StringComparer.Ordinal)
                    .ToArray()
            };
        }

        return descriptors;
    }

    private static IReadOnlyList<string> GetEnvReachableMapChildKeys(
        EnvMapRouteNode point,
        IReadOnlyDictionary<string, EnvMapRouteNode> pointByKey)
    {
        var childKeys = new List<string>(point.ChildKeys.Length);
        foreach (var childKey in point.ChildKeys)
        {
            if (pointByKey.ContainsKey(childKey))
            {
                childKeys.Add(childKey);
            }
        }

        return childKeys;
    }

    private static int CountForcedEnvMapPathSteps(
        string startKey,
        IReadOnlyDictionary<string, EnvMapRouteNode> pointByKey)
    {
        var key = startKey;
        var steps = 0;

        while (!string.IsNullOrWhiteSpace(key) && pointByKey.TryGetValue(key, out var point))
        {
            steps++;
            var childKeys = GetEnvReachableMapChildKeys(point, pointByKey);
            if (childKeys.Count != 1)
            {
                break;
            }

            key = childKeys[0];
        }

        return steps;
    }

    private static bool CanReachEnvMapEliteThenRestSite(
        string startKey,
        IReadOnlyDictionary<string, EnvMapRouteNode> pointByKey)
    {
        return CanReachEnvMapEliteThenRestSiteRecursive(startKey, pointByKey, seenElite: false, new Dictionary<string, bool>(StringComparer.Ordinal));
    }

    private static bool CanReachEnvMapEliteThenRestSiteRecursive(
        string startKey,
        IReadOnlyDictionary<string, EnvMapRouteNode> pointByKey,
        bool seenElite,
        IDictionary<string, bool> memo)
    {
        var memoKey = $"{startKey}|{(seenElite ? "1" : "0")}";
        if (memo.TryGetValue(memoKey, out var cached))
        {
            return cached;
        }

        if (!pointByKey.TryGetValue(startKey, out var point))
        {
            memo[memoKey] = false;
            return false;
        }

        var nextSeenElite = seenElite || string.Equals(point.PointType, "Elite", StringComparison.Ordinal);
        if (nextSeenElite && string.Equals(point.PointType, "RestSite", StringComparison.Ordinal))
        {
            memo[memoKey] = true;
            return true;
        }

        var result = GetEnvReachableMapChildKeys(point, pointByKey)
            .Any(childKey => CanReachEnvMapEliteThenRestSiteRecursive(childKey, pointByKey, nextSeenElite, memo));
        memo[memoKey] = result;
        return result;
    }

    private static bool CanReachEnvMapTypeBeforeType(
        string startKey,
        IReadOnlyDictionary<string, EnvMapRouteNode> pointByKey,
        string desiredType,
        string blockingType)
    {
        return CanReachEnvMapTypeBeforeTypeRecursive(
            startKey,
            pointByKey,
            desiredType,
            blockingType,
            new Dictionary<string, bool>(StringComparer.Ordinal));
    }

    private static bool CanReachEnvMapTypeBeforeTypeRecursive(
        string startKey,
        IReadOnlyDictionary<string, EnvMapRouteNode> pointByKey,
        string desiredType,
        string blockingType,
        IDictionary<string, bool> memo)
    {
        var memoKey = $"{startKey}|{desiredType}|{blockingType}";
        if (memo.TryGetValue(memoKey, out var cached))
        {
            return cached;
        }

        if (!pointByKey.TryGetValue(startKey, out var point))
        {
            memo[memoKey] = false;
            return false;
        }

        if (string.Equals(point.PointType, desiredType, StringComparison.Ordinal))
        {
            memo[memoKey] = true;
            return true;
        }

        if (string.Equals(point.PointType, blockingType, StringComparison.Ordinal))
        {
            memo[memoKey] = false;
            return false;
        }

        var result = GetEnvReachableMapChildKeys(point, pointByKey)
            .Any(childKey => CanReachEnvMapTypeBeforeTypeRecursive(childKey, pointByKey, desiredType, blockingType, memo));
        memo[memoKey] = result;
        return result;
    }

    private static string NormalizeEnvMapPointType(string? pointType)
    {
        return pointType switch
        {
            "Merchant" => "Shop",
            "Shop" => "Shop",
            "Rest" => "RestSite",
            "RestSite" => "RestSite",
            "Monster" => "Monster",
            "Elite" => "Elite",
            "Boss" => "Boss",
            "Event" => "Event",
            "Unknown" => "QuestionMark",
            "QuestionMark" => "QuestionMark",
            "Treasure" => "Treasure",
            _ => pointType ?? "Unknown"
        };
    }

    private static string ToEnvMapCoordKey(MapCoord coord) => $"{coord.col},{coord.row}";

    private sealed class EnvMapRouteNode
    {
        public required MapCoord Coord { get; init; }
        public required string PointType { get; init; }
        public required string[] ChildKeys { get; init; }
    }
}
