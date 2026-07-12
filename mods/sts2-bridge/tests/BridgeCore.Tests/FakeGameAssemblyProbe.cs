using Sts2McpBridge.Scripts;

internal sealed class FakeGameAssemblyProbe : IGameAssemblyProbe
{
    private readonly HashSet<string> _missingCapabilities;
    private readonly HashSet<string> _throwingCapabilities;

    public FakeGameAssemblyProbe(
        GameAssemblyIdentity identity,
        IEnumerable<string>? missingCapabilities = null,
        IEnumerable<string>? throwingCapabilities = null)
    {
        Identity = identity;
        _missingCapabilities = new HashSet<string>(
            missingCapabilities ?? Array.Empty<string>(),
            StringComparer.Ordinal);
        _throwingCapabilities = new HashSet<string>(
            throwingCapabilities ?? Array.Empty<string>(),
            StringComparer.Ordinal);
    }

    public GameAssemblyIdentity Identity { get; }

    public List<string> ProbedCapabilities { get; } = [];

    public GameAssemblyCapabilityProbeResult Probe(GameAssemblyCapabilityRequirement requirement)
    {
        ProbedCapabilities.Add(requirement.Id);
        if (_throwingCapabilities.Contains(requirement.Id))
        {
            throw new InvalidOperationException("synthetic probe failure");
        }

        return _missingCapabilities.Contains(requirement.Id)
            ? new GameAssemblyCapabilityProbeResult(
                requirement.Id,
                Passed: false,
                Code: "synthetic_capability_missing",
                Detail: "Synthetic missing capability for a fail-closed test.")
            : new GameAssemblyCapabilityProbeResult(
                requirement.Id,
                Passed: true,
                Code: "capability_present",
                Detail: "Synthetic capability matched.");
    }
}
