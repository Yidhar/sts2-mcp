using System.Reflection;

namespace Sts2McpBridge.Scripts;

internal enum GameAssemblyMemberKind
{
    Type,
    Method,
    Property,
    EnumValue
}

internal sealed record GameAssemblyIdentity(
    string Name,
    string AssemblyVersion,
    string InformationalVersion,
    Guid ModuleVersionId)
{
    public string DisplayName =>
        $"{Name}, Version={AssemblyVersion}, InformationalVersion={InformationalVersion}, MVID={ModuleVersionId:D}";
}

internal sealed record GameAssemblyCapabilityRequirement(
    string Id,
    string TypeName,
    GameAssemblyMemberKind Kind,
    string? MemberName = null,
    int? ParameterCount = null,
    bool? IsStatic = null,
    string[]? ParameterTypeNames = null);

internal sealed record GameAssemblyCapabilityProbeResult(
    string CapabilityId,
    bool Passed,
    string Code,
    string Detail);

internal interface IGameAssemblyProbe
{
    GameAssemblyIdentity Identity { get; }

    GameAssemblyCapabilityProbeResult Probe(GameAssemblyCapabilityRequirement requirement);
}

internal interface IGameAssemblyCompatibilityProfile
{
    string Id { get; }

    GameAssemblyIdentity ExpectedIdentity { get; }

    IReadOnlyList<GameAssemblyCapabilityRequirement> RequiredCapabilities { get; }

    bool Matches(GameAssemblyIdentity identity);
}

internal sealed record GameAssemblyCompatibilityAssessment(
    string Health,
    bool StartupAllowed,
    string ErrorCode,
    string ErrorMessage,
    string? ProfileId,
    GameAssemblyIdentity Identity,
    IReadOnlyList<GameAssemblyCapabilityProbeResult> ProbeResults)
{
    public int PassedProbeCount => ProbeResults.Count(static result => result.Passed);

    public object ToDiagnosticPayload()
    {
        return new
        {
            health = Health,
            startup_allowed = StartupAllowed,
            error_code = ErrorCode,
            error_message = ErrorMessage,
            profile_id = ProfileId,
            assembly = new
            {
                name = Identity.Name,
                assembly_version = Identity.AssemblyVersion,
                informational_version = Identity.InformationalVersion,
                module_version_id = Identity.ModuleVersionId
            },
            probes = ProbeResults.Select(static result => new
            {
                capability = result.CapabilityId,
                passed = result.Passed,
                code = result.Code,
                detail = result.Detail
            }).ToArray()
        };
    }

    public GameAssemblyCompatibilityAssessment WithActivationFailure(
        string errorCode,
        string errorMessage)
    {
        return this with
        {
            Health = "degraded",
            StartupAllowed = false,
            ErrorCode = errorCode,
            ErrorMessage = errorMessage
        };
    }
}

/// <summary>
/// Process-wide record of the compatibility decision made before Harmony or
/// the HTTP server may start. The initial state is deliberately fail-closed.
/// </summary>
internal static class BridgeGameCompatibilityState
{
    private static GameAssemblyCompatibilityAssessment _current = new(
        Health: "degraded",
        StartupAllowed: false,
        ErrorCode: "game_compatibility_not_evaluated",
        ErrorMessage: "The retail game assembly compatibility gate has not run.",
        ProfileId: null,
        Identity: new GameAssemblyIdentity("unknown", "unknown", "unknown", Guid.Empty),
        ProbeResults: Array.Empty<GameAssemblyCapabilityProbeResult>());

    public static GameAssemblyCompatibilityAssessment Current => Volatile.Read(ref _current);

    public static void Publish(GameAssemblyCompatibilityAssessment assessment)
    {
        ArgumentNullException.ThrowIfNull(assessment);
        Interlocked.Exchange(ref _current, assessment);
    }
}

/// <summary>
/// Exact retail-build registry. Adding a game update requires a new profile,
/// an audited capability list, and tests; assembly-version-only matching is
/// intentionally forbidden because retail builds may reuse 0.x versions.
/// </summary>
internal static class BridgeGameAssemblyCompatibilityRegistry
{
    private static readonly IGameAssemblyCompatibilityProfile[] Profiles =
    [
        new RetailBuild20260623Profile()
    ];

    internal static IReadOnlyList<IGameAssemblyCompatibilityProfile> SupportedProfiles => Profiles;

    public static GameAssemblyCompatibilityAssessment Evaluate(IGameAssemblyProbe probe)
    {
        ArgumentNullException.ThrowIfNull(probe);

        var profile = Profiles.SingleOrDefault(candidate => candidate.Matches(probe.Identity));
        if (profile is null)
        {
            return new GameAssemblyCompatibilityAssessment(
                Health: "degraded",
                StartupAllowed: false,
                ErrorCode: "unsupported_game_assembly",
                ErrorMessage:
                    "No audited Bridge adapter profile matches the loaded retail game assembly. " +
                    "Harmony patches and the HTTP mutation service were not started.",
                ProfileId: null,
                Identity: probe.Identity,
                ProbeResults: Array.Empty<GameAssemblyCapabilityProbeResult>());
        }

        var results = new List<GameAssemblyCapabilityProbeResult>(profile.RequiredCapabilities.Count);
        foreach (var requirement in profile.RequiredCapabilities)
        {
            try
            {
                results.Add(probe.Probe(requirement));
            }
            catch (Exception ex)
            {
                results.Add(new GameAssemblyCapabilityProbeResult(
                    requirement.Id,
                    Passed: false,
                    Code: "capability_probe_exception",
                    Detail: $"Probe raised {ex.GetType().Name}; the capability is not trusted."));
            }
        }

        var failed = results.Where(static result => !result.Passed).ToArray();
        if (failed.Length > 0)
        {
            return new GameAssemblyCompatibilityAssessment(
                Health: "degraded",
                StartupAllowed: false,
                ErrorCode: "required_game_capability_missing",
                ErrorMessage:
                    $"Adapter profile '{profile.Id}' matched the assembly identity, but " +
                    $"{failed.Length} of {results.Count} required capability probes failed. " +
                    "Harmony patches and the HTTP mutation service were not started.",
                ProfileId: profile.Id,
                Identity: probe.Identity,
                ProbeResults: results);
        }

        return new GameAssemblyCompatibilityAssessment(
            Health: "ready",
            StartupAllowed: true,
            ErrorCode: string.Empty,
            ErrorMessage: string.Empty,
            ProfileId: profile.Id,
            Identity: probe.Identity,
            ProbeResults: results);
    }

    private sealed class RetailBuild20260623Profile : IGameAssemblyCompatibilityProfile
    {
        private static readonly GameAssemblyIdentity Identity = new(
            Name: "sts2",
            AssemblyVersion: "0.1.0.0",
            InformationalVersion: "0.1.0+59260271157f76a2896f0eab5bc6ea1245d8b314",
            ModuleVersionId: Guid.Parse("97f10687-c306-4798-ab75-8b9f23f34dfb"));

        private static readonly GameAssemblyCapabilityRequirement[] Capabilities =
        [
            new(
                "lifecycle.game.ready",
                "MegaCrit.Sts2.Core.Nodes.NGame",
                GameAssemblyMemberKind.Method,
                "_Ready",
                ParameterCount: 0,
                IsStatic: false),
            new(
                "lifecycle.game.exit",
                "MegaCrit.Sts2.Core.Nodes.NGame",
                GameAssemblyMemberKind.Method,
                "_ExitTree",
                ParameterCount: 0,
                IsStatic: false),
            new(
                "lifecycle.game.instance",
                "MegaCrit.Sts2.Core.Nodes.NGame",
                GameAssemblyMemberKind.Property,
                "Instance",
                IsStatic: true),
            new(
                "lifecycle.controller.process",
                "MegaCrit.Sts2.Core.Nodes.CommonUi.NControllerManager",
                GameAssemblyMemberKind.Method,
                "_Process",
                ParameterCount: 1,
                IsStatic: false,
                ParameterTypeNames: ["System.Double"]),
            new(
                "lifecycle.run.process",
                "MegaCrit.Sts2.Core.Nodes.NRun",
                GameAssemblyMemberKind.Method,
                "_Process",
                ParameterCount: 1,
                IsStatic: false,
                ParameterTypeNames: ["System.Double"]),
            new(
                "combat.manager.instance",
                "MegaCrit.Sts2.Core.Combat.CombatManager",
                GameAssemblyMemberKind.Property,
                "Instance",
                IsStatic: true),
            new(
                "combat.manager.in_progress",
                "MegaCrit.Sts2.Core.Combat.CombatManager",
                GameAssemblyMemberKind.Property,
                "IsInProgress",
                IsStatic: false),
            new(
                "combat.manager.player_turn",
                "MegaCrit.Sts2.Core.Combat.CombatManager",
                GameAssemblyMemberKind.Method,
                "IsPartOfPlayerTurn",
                ParameterCount: 1,
                IsStatic: false),
            new(
                "run.manager.instance",
                "MegaCrit.Sts2.Core.Runs.RunManager",
                GameAssemblyMemberKind.Property,
                "Instance",
                IsStatic: true),
            new(
                "run.manager.singleplayer_setup",
                "MegaCrit.Sts2.Core.Runs.RunManager",
                GameAssemblyMemberKind.Method,
                "SetUpNewSingleplayer",
                ParameterCount: 3,
                IsStatic: false),
            new(
                "card_reward.skip_action",
                "MegaCrit.Sts2.Core.Entities.Rewards.PostAlternateCardRewardAction",
                GameAssemblyMemberKind.EnumValue,
                "EndSelectionAndCompleteReward")
        ];

        public string Id => "retail-2026-06-23-5926027";

        public GameAssemblyIdentity ExpectedIdentity => Identity;

        public IReadOnlyList<GameAssemblyCapabilityRequirement> RequiredCapabilities => Capabilities;

        public bool Matches(GameAssemblyIdentity identity)
        {
            return string.Equals(identity.Name, Identity.Name, StringComparison.Ordinal) &&
                   string.Equals(identity.AssemblyVersion, Identity.AssemblyVersion, StringComparison.Ordinal) &&
                   string.Equals(
                       identity.InformationalVersion,
                       Identity.InformationalVersion,
                       StringComparison.Ordinal) &&
                   identity.ModuleVersionId == Identity.ModuleVersionId;
        }
    }
}

/// <summary>
/// The only production reflection entry point for build identity and startup
/// capability checks. Runtime payload adapters may still use localized
/// reflection, but no mutation surface is activated until these probes pass.
/// </summary>
internal sealed class ReflectionGameAssemblyProbe : IGameAssemblyProbe
{
    private const BindingFlags AllMembers =
        BindingFlags.Public |
        BindingFlags.NonPublic |
        BindingFlags.Instance |
        BindingFlags.Static;

    private readonly Assembly _assembly;

    public ReflectionGameAssemblyProbe(Assembly assembly)
    {
        _assembly = assembly ?? throw new ArgumentNullException(nameof(assembly));
        var assemblyName = assembly.GetName();
        var informationalVersion = assembly
            .GetCustomAttribute<AssemblyInformationalVersionAttribute>()?
            .InformationalVersion ?? "unknown";

        Identity = new GameAssemblyIdentity(
            assemblyName.Name ?? "unknown",
            assemblyName.Version?.ToString() ?? "unknown",
            informationalVersion,
            assembly.ManifestModule.ModuleVersionId);
    }

    public GameAssemblyIdentity Identity { get; }

    public GameAssemblyCapabilityProbeResult Probe(GameAssemblyCapabilityRequirement requirement)
    {
        ArgumentNullException.ThrowIfNull(requirement);

        try
        {
            var type = _assembly.GetType(requirement.TypeName, throwOnError: false, ignoreCase: false);
            if (type is null)
            {
                return Failure(requirement, "required_type_missing", $"Type '{requirement.TypeName}' was not found.");
            }

            return requirement.Kind switch
            {
                GameAssemblyMemberKind.Type => Success(requirement),
                GameAssemblyMemberKind.Method => ProbeMethod(type, requirement),
                GameAssemblyMemberKind.Property => ProbeProperty(type, requirement),
                GameAssemblyMemberKind.EnumValue => ProbeEnumValue(type, requirement),
                _ => Failure(requirement, "unsupported_probe_kind", "The profile declared an unsupported probe kind.")
            };
        }
        catch (Exception ex)
        {
            return Failure(
                requirement,
                "capability_probe_exception",
                $"Reflection raised {ex.GetType().Name}; the capability is not trusted.");
        }
    }

    private static GameAssemblyCapabilityProbeResult ProbeMethod(
        Type type,
        GameAssemblyCapabilityRequirement requirement)
    {
        var expectedParameterTypes = requirement.ParameterTypeNames ?? Array.Empty<string>();
        var matched = type
            .GetMethods(AllMembers)
            .Where(method => string.Equals(method.Name, requirement.MemberName, StringComparison.Ordinal))
            .Where(method => requirement.IsStatic is null || method.IsStatic == requirement.IsStatic)
            .Where(method => requirement.ParameterCount is null ||
                             method.GetParameters().Length == requirement.ParameterCount)
            .Any(method => expectedParameterTypes.Length == 0 ||
                           method.GetParameters()
                               .Select(static parameter => parameter.ParameterType.FullName ?? parameter.ParameterType.Name)
                               .SequenceEqual(expectedParameterTypes, StringComparer.Ordinal));

        return matched
            ? Success(requirement)
            : Failure(
                requirement,
                "required_method_missing",
                $"Method '{requirement.TypeName}.{requirement.MemberName}' did not match the audited signature.");
    }

    private static GameAssemblyCapabilityProbeResult ProbeProperty(
        Type type,
        GameAssemblyCapabilityRequirement requirement)
    {
        var property = type.GetProperty(requirement.MemberName!, AllMembers);
        var accessor = property?.GetMethod ?? property?.SetMethod;
        var matched = property is not null &&
                      accessor is not null &&
                      (requirement.IsStatic is null || accessor.IsStatic == requirement.IsStatic);

        return matched
            ? Success(requirement)
            : Failure(
                requirement,
                "required_property_missing",
                $"Property '{requirement.TypeName}.{requirement.MemberName}' did not match the audited shape.");
    }

    private static GameAssemblyCapabilityProbeResult ProbeEnumValue(
        Type type,
        GameAssemblyCapabilityRequirement requirement)
    {
        var matched = type.IsEnum &&
                      requirement.MemberName is not null &&
                      Enum.GetNames(type).Contains(requirement.MemberName, StringComparer.Ordinal);

        return matched
            ? Success(requirement)
            : Failure(
                requirement,
                "required_enum_value_missing",
                $"Enum value '{requirement.TypeName}.{requirement.MemberName}' was not found.");
    }

    private static GameAssemblyCapabilityProbeResult Success(
        GameAssemblyCapabilityRequirement requirement)
    {
        return new GameAssemblyCapabilityProbeResult(
            requirement.Id,
            Passed: true,
            Code: "capability_present",
            Detail: "Required capability matched the audited profile.");
    }

    private static GameAssemblyCapabilityProbeResult Failure(
        GameAssemblyCapabilityRequirement requirement,
        string code,
        string detail)
    {
        return new GameAssemblyCapabilityProbeResult(requirement.Id, Passed: false, code, detail);
    }
}
