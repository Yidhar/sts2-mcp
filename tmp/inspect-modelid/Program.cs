using System.Reflection;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Unlocks;

var modelIdType = typeof(ModelId);
Console.WriteLine($"ModelId: {modelIdType.AssemblyQualifiedName}");

Console.WriteLine("Constructors:");
foreach (var ctor in modelIdType.GetConstructors(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance))
{
    Console.WriteLine($"  {ctor}");
}

Console.WriteLine("Methods:");
foreach (var method in modelIdType.GetMethods(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Static | BindingFlags.Instance)
             .OrderBy(m => m.Name))
{
    if (method.Name.Contains("Parse", StringComparison.Ordinal) ||
        method.Name.Contains("From", StringComparison.Ordinal) ||
        method.Name.Contains("op_", StringComparison.Ordinal) ||
        method.Name.Equals("ToString", StringComparison.Ordinal))
    {
        Console.WriteLine($"  {method}");
    }
}

Console.WriteLine("Properties:");
foreach (var property in modelIdType.GetProperties(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Static | BindingFlags.Instance))
{
    Console.WriteLine($"  {property.PropertyType} {property.Name}");
}

Console.WriteLine("Fields:");
foreach (var field in modelIdType.GetFields(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Static | BindingFlags.Instance))
{
    Console.WriteLine($"  {field.FieldType} {field.Name}");
}

Console.WriteLine("ModelDb.GetById overloads:");
foreach (var method in typeof(ModelDb).GetMethods(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Static | BindingFlags.Instance)
             .Where(m => m.Name.Equals("GetById", StringComparison.Ordinal))
             .OrderBy(m => m.ToString()))
{
    Console.WriteLine($"  {method}");
}

Console.WriteLine("ModelDb properties:");
foreach (var property in typeof(ModelDb).GetProperties(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Static | BindingFlags.Instance)
             .OrderBy(p => p.Name))
{
    Console.WriteLine($"  {property.PropertyType} {property.Name}");
}

Console.WriteLine("ModelDb fields:");
foreach (var field in typeof(ModelDb).GetFields(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Static | BindingFlags.Instance)
             .OrderBy(f => f.Name))
{
    Console.WriteLine($"  {field.FieldType} {field.Name}");
}

Console.WriteLine("ModelDb enumerable-ish methods:");
foreach (var method in typeof(ModelDb).GetMethods(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Static | BindingFlags.Instance)
             .Where(m => m.Name.Contains("All", StringComparison.OrdinalIgnoreCase) ||
                         m.Name.Contains("Enum", StringComparison.OrdinalIgnoreCase) ||
                         m.Name.Contains("Values", StringComparison.OrdinalIgnoreCase) ||
                         m.Name.Contains("Entries", StringComparison.OrdinalIgnoreCase) ||
                         m.Name.Contains("Models", StringComparison.OrdinalIgnoreCase))
             .OrderBy(m => m.Name)
             .ThenBy(m => m.ToString()))
{
    Console.WriteLine($"  {method}");
}

if (args.Length > 0)
{
    var encounterBaseType = typeof(EncounterModel);
    foreach (var arg in args)
    {
        var match = AppDomain.CurrentDomain.GetAssemblies()
            .SelectMany(asm =>
            {
                try { return asm.GetTypes(); }
                catch { return Array.Empty<Type>(); }
            })
            .FirstOrDefault(t => t.Name.Equals(arg, StringComparison.Ordinal) || (t.FullName?.EndsWith("." + arg, StringComparison.Ordinal) ?? false));

        Console.WriteLine();
        Console.WriteLine($"Type probe: {arg}");
        if (match is null)
        {
            Console.WriteLine("  not found");
            continue;
        }

        Console.WriteLine($"  FullName: {match.FullName}");
        Console.WriteLine($"  BaseType: {match.BaseType}");
        Console.WriteLine($"  AssignableToEncounterModel: {encounterBaseType.IsAssignableFrom(match)}");

        Console.WriteLine("  Static properties:");
        foreach (var property in match.GetProperties(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Static))
        {
            object? value = null;
            string? error = null;
            try
            {
                if (property.GetIndexParameters().Length == 0)
                    value = property.GetValue(null);
            }
            catch (Exception ex)
            {
                error = ex.GetBaseException().Message;
            }

            Console.WriteLine($"    {property.PropertyType.Name} {property.Name} = {value ?? "<null>"}{(error is null ? string.Empty : $" [error: {error}]")}");
        }

        Console.WriteLine("  Static fields:");
        foreach (var field in match.GetFields(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Static))
        {
            object? value = null;
            string? error = null;
            try
            {
                value = field.GetValue(null);
            }
            catch (Exception ex)
            {
                error = ex.GetBaseException().Message;
            }

            Console.WriteLine($"    {field.FieldType.Name} {field.Name} = {value ?? "<null>"}{(error is null ? string.Empty : $" [error: {error}]")}");
        }
    }
}

Console.WriteLine();
Console.WriteLine("RunManager methods of interest:");
foreach (var method in typeof(RunManager).GetMethods(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Static | BindingFlags.Instance)
             .Where(m => m.Name.Contains("Debug", StringComparison.OrdinalIgnoreCase) ||
                         m.Name.Contains("Room", StringComparison.OrdinalIgnoreCase) ||
                         m.Name.Contains("Combat", StringComparison.OrdinalIgnoreCase) ||
                         m.Name.Contains("Encounter", StringComparison.OrdinalIgnoreCase) ||
                         m.Name.Contains("SinglePlayer", StringComparison.OrdinalIgnoreCase))
             .OrderBy(m => m.Name)
             .ThenBy(m => m.ToString()))
{
    Console.WriteLine($"  {method}");
}

Console.WriteLine();
Console.WriteLine("RunState static methods of interest:");
foreach (var method in typeof(RunState).GetMethods(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Static)
             .Where(m => m.Name.Contains("Create", StringComparison.OrdinalIgnoreCase) ||
                         m.Name.Contains("Test", StringComparison.OrdinalIgnoreCase) ||
                         m.Name.Contains("Debug", StringComparison.OrdinalIgnoreCase))
             .OrderBy(m => m.Name)
             .ThenBy(m => m.ToString()))
{
    Console.WriteLine($"  {method}");
}

Console.WriteLine();
Console.WriteLine("Player constructors:");
foreach (var ctor in typeof(MegaCrit.Sts2.Core.Entities.Players.Player).GetConstructors(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance))
{
    Console.WriteLine($"  {ctor}");
}

Console.WriteLine("Player static methods:");
foreach (var method in typeof(MegaCrit.Sts2.Core.Entities.Players.Player).GetMethods(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Static)
             .OrderBy(m => m.Name)
             .ThenBy(m => m.ToString()))
{
    if (method.Name.Contains("Create", StringComparison.OrdinalIgnoreCase) ||
        method.Name.Contains("Default", StringComparison.OrdinalIgnoreCase) ||
        method.Name.Contains("Single", StringComparison.OrdinalIgnoreCase))
    {
        Console.WriteLine($"  {method}");
    }
}

Console.WriteLine();
Console.WriteLine("First characters:");
try
{
    foreach (var character in ModelDb.AllCharacters.Take(10))
    {
        Console.WriteLine($"  {character.Id} :: {character.Title}");
    }
}
catch (Exception ex)
{
    Console.WriteLine($"  [error] {ex.GetBaseException().Message}");
}

Console.WriteLine("First acts:");
try
{
    foreach (var act in ModelDb.Acts.Take(5))
    {
        Console.WriteLine($"  {act.Id} :: {act.Title}");
    }
}
catch (Exception ex)
{
    Console.WriteLine($"  [error] {ex.GetBaseException().Message}");
}

Console.WriteLine();
Console.WriteLine("UnlockState constructors:");
foreach (var ctor in typeof(UnlockState).GetConstructors(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance))
{
    Console.WriteLine($"  {ctor}");
}

Console.WriteLine("UnlockState static members:");
foreach (var property in typeof(UnlockState).GetProperties(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Static))
{
    Console.WriteLine($"  PROP {property.PropertyType} {property.Name}");
}
foreach (var field in typeof(UnlockState).GetFields(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Static))
{
    Console.WriteLine($"  FIELD {field.FieldType} {field.Name}");
}
