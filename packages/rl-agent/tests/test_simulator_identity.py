from __future__ import annotations

import json
from pathlib import Path

import pytest

from sts2_rl.simulator_identity import (
    IDENTITY_SCHEMA_VERSION,
    SimulatorIdentityError,
    repository_root,
    sha256_file,
    simulator_identity_path,
    verify_headless_simulator,
    write_preflight_audit,
)
from sts2_rl.training.checkpointing import TrainingState
from sts2_rl.training.config import TrainingConfig

EXPECTED_STS2_AI_COMMIT = "252bf989feb598c61c20267d3979b00bf316edb5"
EXPECTED_STS2_AI_TREE = "9b3674ad857a20ce201a3b78a277621007d3ddda"
EXPECTED_STS2_AI_PATCHED_TREE = "f94b594ce17925bc393873489f38875fb6621d4f"
CANONICAL_HEADLESS_PROJECT = "STS2AI/ENV/Sim/HeadlessSim/HeadlessSim.csproj"
EXPECTED_PATCH_CHAIN = [
    "third_party/patches/sts2-ai/0001-native-revival-curriculum.patch",
    "third_party/patches/sts2-ai/0002-stable-settlement-contract.patch",
    "third_party/patches/sts2-ai/0003-guard-orb-state-lifecycle.patch",
    "third_party/patches/sts2-ai/0004-fail-closed-legal-action-eligibility.patch",
    "third_party/patches/sts2-ai/0005-authoritative-victory-and-reward-claims.patch",
    "third_party/patches/sts2-ai/0006-bound-orderless-pile-multisets.patch",
    "third_party/patches/sts2-ai/0007-grounded-macro-observation-contract.patch",
    "third_party/patches/sts2-ai/0008-guard-unowned-shop-potion-context.patch",
    "third_party/patches/sts2-ai/0009-clean-hidden-training-revival.patch",
    "third_party/patches/sts2-ai/0010-headless-event-and-post-combat-lifecycle.patch",
]


def _lock(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    payload = {
        "url": "https://example.invalid/sts2-ai.git",
        "commit": "1" * 40,
        "tree": "2" * 40,
        "canonical_headless_project": "STS2AI/ENV/Sim/HeadlessSim/HeadlessSim.csproj",
    }
    path = tmp_path / "sts2-ai.lock.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path, payload


def _simulator(tmp_path: Path) -> tuple[Path, Path, Path, dict[str, object]]:
    lock_path, lock = _lock(tmp_path)
    executable = tmp_path / "HeadlessSim.exe"
    executable.write_bytes(b"pinned-headless-sim")
    managed_assembly = tmp_path / "HeadlessSim.dll"
    managed_assembly.write_bytes(b"pinned-headless-sim-managed-code")
    payload: dict[str, object] = {
        "schema_version": IDENTITY_SCHEMA_VERSION,
        "component": "HeadlessSim",
        "source": {
            "url": lock["url"],
            "commit": lock["commit"],
            "tree": lock["tree"],
            "project": lock["canonical_headless_project"],
            "patches": lock.get("patches", []),
        },
        "build": {
            "configuration": "Release",
            "target_framework": "net9.0",
            "dotnet_sdk": "9.0.308",
        },
        "binary": {
            "file_name": executable.name,
            "size_bytes": executable.stat().st_size,
            "sha256": sha256_file(executable),
        },
        "managed_binary": {
            "file_name": managed_assembly.name,
            "size_bytes": managed_assembly.stat().st_size,
            "sha256": sha256_file(managed_assembly),
        },
    }
    identity_path = simulator_identity_path(executable)
    identity_path.write_text(json.dumps(payload), encoding="utf-8")
    return executable, identity_path, lock_path, payload


def _simulator_for_repository_lock(tmp_path: Path) -> Path:
    from sts2_rl.simulator_identity import load_sts2_ai_lock

    lock = load_sts2_ai_lock()
    executable = tmp_path / "HeadlessSim.exe"
    executable.write_bytes(b"repository-pinned-test-simulator")
    managed_assembly = tmp_path / "HeadlessSim.dll"
    managed_assembly.write_bytes(b"repository-pinned-test-managed-code")
    payload = {
        "schema_version": IDENTITY_SCHEMA_VERSION,
        "component": "HeadlessSim",
        "source": {
            "url": lock["url"],
            "commit": lock["commit"],
            "tree": lock["tree"],
            "project": lock["canonical_headless_project"],
            "patches": lock.get("patches", []),
        },
        "build": {
            "configuration": "Release",
            "target_framework": "net9.0",
            "dotnet_sdk": "test-sdk",
        },
        "binary": {
            "file_name": executable.name,
            "size_bytes": executable.stat().st_size,
            "sha256": sha256_file(executable),
        },
        "managed_binary": {
            "file_name": managed_assembly.name,
            "size_bytes": managed_assembly.stat().st_size,
            "sha256": sha256_file(managed_assembly),
        },
    }
    simulator_identity_path(executable).write_text(json.dumps(payload), encoding="utf-8")
    return executable


def test_repository_simulator_lock_names_one_canonical_patched_tree() -> None:
    root = repository_root()
    lock = json.loads(
        (root / "third_party" / "sts2-ai.lock.json").read_text(encoding="utf-8")
    )

    assert lock["commit"] == EXPECTED_STS2_AI_COMMIT
    assert lock["tree"] == EXPECTED_STS2_AI_TREE
    assert lock["patched_tree"] == EXPECTED_STS2_AI_PATCHED_TREE
    assert lock["canonical_headless_project"] == CANONICAL_HEADLESS_PROJECT
    assert lock["excluded_duplicate_project"] == (
        "STS2AI/ENV/Sim/Runtime/HeadlessSim/HeadlessSim.csproj"
    )
    assert [record["path"] for record in lock["patches"]] == EXPECTED_PATCH_CHAIN

    for record in lock["patches"]:
        patch_path = root / record["path"]
        assert record["sha256"] == sha256_file(patch_path)
        patch = patch_path.read_text(encoding="utf-8")
        assert "STS2AI/ENV/Sim/Overlay/" not in patch
        assert "STS2AI/ENV/Sim/Runtime/HeadlessSim/" not in patch


def test_legal_action_eligibility_patch_is_locked_fail_closed_and_index_stable() -> None:
    root = repository_root()
    lock = json.loads((root / "third_party" / "sts2-ai.lock.json").read_text(encoding="utf-8"))
    matching = [
        record
        for record in lock["patches"]
        if record["path"].endswith("0004-fail-closed-legal-action-eligibility.patch")
    ]
    assert len(matching) == 1

    patch_path = root / matching[0]["path"]
    assert matching[0]["sha256"] == sha256_file(patch_path)
    patch = patch_path.read_text(encoding="utf-8")
    sections: dict[str, str] = {}
    for section in patch.split("diff --git ")[1:]:
        before, after = section.splitlines()[0].split()
        assert before.startswith("a/") and after.startswith("b/")
        assert before[2:] == after[2:]
        sections[after[2:]] = section

    headless_helper_path = "STS2AI/ENV/Sim/HeadlessSim/Simulation/FullRunLegalActionEligibility.cs"
    settlement_path = "STS2AI/ENV/Sim/HeadlessSim/Simulation/FullRunSettlementContract.cs"
    headless_builder_path = "STS2AI/ENV/Sim/HeadlessSim/Simulation/FullRunSimulationStateBuilder.cs"
    assert set(sections) == {
        headless_helper_path,
        settlement_path,
        headless_builder_path,
    }
    assert all("/Overlay/" not in path for path in sections)

    def changed_lines(section: str, prefix: str) -> list[str]:
        header_prefix = prefix * 3
        return [
            line[1:] for line in section.splitlines() if line.startswith(prefix) and not line.startswith(header_prefix)
        ]

    def added_source(section: str) -> str:
        return "\n".join(changed_lines(section, "+"))

    headless_helper = added_source(sections[headless_helper_path])
    assert "internal static bool IsShopPurchaseSupported(" in headless_helper
    assert "bool hasOpenPotionSlots," in headless_helper
    assert "bool canProcurePotion)" in headless_helper
    assert "&& (!isPotionEntry || (hasOpenPotionSlots && canProcurePotion));" in headless_helper
    assert "full potion slots must suppress an otherwise affordable potion purchase" in headless_helper
    assert "Sozu must suppress an otherwise affordable potion purchase even with an open slot" in headless_helper
    assert "an ordinary open-slot player must preserve an otherwise affordable potion purchase" in headless_helper

    assert "internal static IEnumerable<uint?> EnumerateCardPlayTargets(" in headless_helper
    assert "if (!requiresTarget)" in headless_helper
    assert "yield return null;" in headless_helper
    assert "foreach (uint targetId in validTargetIds)" in headless_helper
    assert "yield return targetId;" in headless_helper
    assert "noTargets.Count == 0" in headless_helper
    assert "new uint[] { 17u, 42u }" in headless_helper
    assert "multipleTargets.Count == 2" in headless_helper
    assert "multipleTargets[0] == 17u" in headless_helper
    assert "multipleTargets[1] == 42u" in headless_helper
    assert "targetless.Count == 1 && targetless[0] == null" in headless_helper

    settlement_added = changed_lines(sections[settlement_path], "+")
    assert settlement_added == ["\t\tFullRunLegalActionEligibility.RunSelfTests();", ""]

    headless_builder_added = changed_lines(sections[headless_builder_path], "+")
    headless_builder_removed = changed_lines(sections[headless_builder_path], "-")
    builder_added_source = "\n".join(headless_builder_added)

    assert "using MegaCrit.Sts2.Core.Hooks;" in headless_builder_added
    hook_lines = [
        "\t\t\t\tPotionModel? potion = potionEntry.Model;",
        "\t\t\t\tcanProcurePotion = potion != null",
        "\t\t\t\t\t&& Hook.ShouldProcurePotion(",
        "\t\t\t\t\t\tlocalPlayer.RunState,",
        "\t\t\t\t\t\tlocalPlayer.Creature.CombatState,",
        "\t\t\t\t\t\tpotion,",
        "\t\t\t\t\t\tlocalPlayer);",
    ]
    hook_positions = [headless_builder_added.index(line) for line in hook_lines]
    assert hook_positions == sorted(hook_positions)
    assert "FullRunLegalActionEligibility.IsShopPurchaseSupported(" in builder_added_source
    assert "\t\t\t\t\tlocalPlayer.HasOpenPotionSlots," in headless_builder_added
    assert "\t\t\t\t\tcanProcurePotion)" in headless_builder_added

    # Raw merchant indices remain attached before support filtering; a sparse
    # simulator slot such as 11 must never be replaced by candidate ordinal 1.
    section = sections[headless_builder_path]
    context = changed_lines(section, " ")
    added = changed_lines(section, "+")
    removed = changed_lines(section, "-")
    assert "\t\t\t\tIndex = index++," in context
    assert not any(line.lstrip().startswith("Index =") for line in added + removed)

    old_guard = "\t\t\t\tif (card3.RequiresTarget && validTargetIds.Count > 0)"
    assert old_guard in headless_builder_removed
    assert old_guard not in headless_builder_added
    assert (
        "\t\t\t\tforeach (uint? targetId in "
        "FullRunLegalActionEligibility.EnumerateCardPlayTargets("
        "card3.RequiresTarget, validTargetIds))"
    ) in headless_builder_added
    assert "\t\t\t\t\t\tTargetId = targetId," in headless_builder_added


def test_authoritative_victory_and_reward_claim_patch_is_locked_and_fail_closed() -> None:
    root = repository_root()
    lock = json.loads((root / "third_party" / "sts2-ai.lock.json").read_text(encoding="utf-8"))
    matching = [
        record
        for record in lock["patches"]
        if record["path"].endswith("0005-authoritative-victory-and-reward-claims.patch")
    ]
    assert len(matching) == 1

    patch_path = root / matching[0]["path"]
    assert matching[0]["sha256"] == sha256_file(patch_path)
    patch = patch_path.read_text(encoding="utf-8")
    sections: dict[str, str] = {}
    for section in patch.split("diff --git ")[1:]:
        before, after = section.splitlines()[0].split()
        assert before.startswith("a/") and after.startswith("b/")
        assert before[2:] == after[2:]
        sections[after[2:]] = section

    headless_eligibility = (
        "STS2AI/ENV/Sim/HeadlessSim/Simulation/FullRunLegalActionEligibility.cs"
    )
    settlement = "STS2AI/ENV/Sim/HeadlessSim/Simulation/FullRunSettlementContract.cs"
    headless_outcome = (
        "STS2AI/ENV/Sim/HeadlessSim/Simulation/FullRunSimulationOutcomeContract.cs"
    )
    headless_builder = (
        "STS2AI/ENV/Sim/HeadlessSim/Simulation/FullRunSimulationStateBuilder.cs"
    )
    headless_runtime = (
        "STS2AI/ENV/Sim/HeadlessSim/Simulation/FullRunSimulatorRuntimeFacade.cs"
    )
    combat_manager = "STS2AI/ENV/Sim/SrcCompat/Source01032/Core/Combat/CombatManager.cs"
    run_manager = "src/Core/Runs/RunManager.cs"
    assert set(sections) == {
        headless_eligibility,
        settlement,
        headless_outcome,
        headless_builder,
        headless_runtime,
        combat_manager,
        run_manager,
    }
    assert all("/Overlay/" not in path for path in sections)

    def changed_lines(section: str, prefix: str) -> list[str]:
        header_prefix = prefix * 3
        return [
            line[1:]
            for line in section.splitlines()
            if line.startswith(prefix) and not line.startswith(header_prefix)
        ]

    def added_source(section: str) -> str:
        return "\n".join(changed_lines(section, "+"))

    headless_eligibility_added = added_source(sections[headless_eligibility])
    assert "internal static bool IsRewardClaimSupported(bool claimable)" in headless_eligibility_added
    assert "return claimable;" in headless_eligibility_added
    assert "unclaimable rewards must never be emitted as supported legal actions" in (
        headless_eligibility_added
    )
    assert "claimable rewards must preserve their executable claim action" in (
        headless_eligibility_added
    )
    assert "internal static bool IsRewardClaimRequestSupported(" in headless_eligibility_added
    assert 'string.Equals(action.Action, "claim_reward", StringComparison.Ordinal)' in (
        headless_eligibility_added
    )
    assert "action.Claimable == true" in headless_eligibility_added
    assert "&& action.IsSupported" in headless_eligibility_added
    assert "executor must accept the exact current claimable reward action" in (
        headless_eligibility_added
    )
    assert "executor must reject a raw or stale reward index" in headless_eligibility_added
    assert "executor must reject claimable=false" in headless_eligibility_added
    assert "executor must reject an unsupported claim" in headless_eligibility_added

    headless_outcome_added = added_source(sections[headless_outcome])
    assert "internal static long NormalizeVictoryTime(long runTime)" in headless_outcome_added
    assert "return Math.Max(1L, runTime);" in headless_outcome_added
    assert "NormalizeVictoryTime(0L) == 1L" in headless_outcome_added
    assert "internal static bool IsFinalBossVictory(" in headless_outcome_added
    assert "primary final boss must latch victory independently of presentation" in (
        headless_outcome_added
    )
    assert "second final boss must latch victory independently of presentation" in (
        headless_outcome_added
    )
    assert "internal static string? ResolveTerminalOutcome(" in headless_outcome_added
    assert "return winTime > 0 ? Victory : Defeat;" in headless_outcome_added

    settlement_added = changed_lines(sections[settlement], "+")
    assert settlement_added == ["\t\tFullRunSimulationOutcomeContract.RunSelfTests();"]

    headless_builder_added = changed_lines(sections[headless_builder], "+")
    headless_builder_removed = changed_lines(sections[headless_builder], "-")
    builder_added_source = "\n".join(headless_builder_added)
    builder_context = changed_lines(sections[headless_builder], " ")
    assert "FullRunSimulationOutcomeContract.ResolveTerminalOutcome(" in builder_added_source
    assert "\t\t\tRunManager.Instance.WinTime);" in headless_builder_added
    assert (
        "\t\t\t\tIsSupported = "
        "FullRunLegalActionEligibility.IsRewardClaimSupported(claimable)"
    ) in headless_builder_added
    assert "Hook vetoes such as Sozu consume the reward selection" in builder_added_source
    assert "\t\t\t\tClaimable = claimable," in builder_context
    assert "\t\t\t\tClaimBlockReason = blockReason," in builder_context
    assert not any("Action = \"proceed\"" in line for line in headless_builder_added)
    assert any(
        "? (RunManager.Instance.WinTime > 0 ? \"victory\" : \"defeat\")" in line
        for line in headless_builder_removed
    )

    runtime_added = added_source(sections[headless_runtime])
    runtime_removed = "\n".join(changed_lines(sections[headless_runtime], "-"))
    runtime_section = sections[headless_runtime]
    assert "FullRunPendingRewardSelectionSnapshot? rewardSelection =" in runtime_added
    assert "state.CachedBridgeSnapshots?.RewardSelection;" in runtime_added
    assert "rewardIndex >= rewardSelection.Rewards.Count" in runtime_added
    assert "FullRunLegalActionEligibility.IsRewardClaimRequestSupported(" in runtime_added
    assert "\t\t\t\t\tstate.LegalActions," in changed_lines(
        sections[headless_runtime], "+"
    )
    assert '"full_run_reward_not_claimable"' in runtime_added
    assert "TrySelectReward(action.Index.Value" in runtime_removed
    assert runtime_section.index("IsRewardClaimRequestSupported(") < runtime_section.index(
        "TrySelectReward(rewardIndex"
    )
    assert runtime_section.index('"full_run_reward_not_claimable"') < runtime_section.index(
        "TrySelectReward(rewardIndex"
    )
    # Reward observation projection remains independent and visible. This patch
    # only closes the executable claim surface; it must not delete or rewrite
    # FullRunApiStateBuilder's rewards.items projection.
    assert not any(path.endswith("FullRunApiStateBuilder.cs") for path in sections)

    combat_added = added_source(sections[combat_manager])
    combat_removed = "\n".join(changed_lines(sections[combat_manager], "-"))
    assert "FullRunSimulationOutcomeContract.IsFinalBossVictory(" in combat_added
    assert "FullRunSimulationOutcomeContract.NormalizeVictoryTime(" in combat_added
    assert "SkipCombatPresentation" not in combat_added
    assert "!SkipCombatPresentation" in combat_removed

    run_added = added_source(sections[run_manager])
    assert "WinRun is the authoritative Architect closeout" in run_added
    assert "WinTime = Math.Max(1L, RunTime);" in run_added
    run_section = sections[run_manager]
    assert run_section.index("WinTime = Math.Max(1L, RunTime);") < run_section.index(
        "((TheArchitect)eventRoom.LocalMutableEvent).TriggerVictory();"
    )

def test_orderless_pile_multiset_patch_is_locked_and_lossless() -> None:
    root = repository_root()
    lock = json.loads((root / "third_party" / "sts2-ai.lock.json").read_text(encoding="utf-8"))
    matching = [
        record
        for record in lock["patches"]
        if record["path"].endswith("0006-bound-orderless-pile-multisets.patch")
    ]
    assert len(matching) == 1

    patch_path = root / matching[0]["path"]
    assert matching[0]["sha256"] == sha256_file(patch_path)
    patch = patch_path.read_text(encoding="utf-8")
    sections = {
        section.splitlines()[0].split()[1][2:]: section
        for section in patch.split("diff --git ")[1:]
    }
    assert set(sections) == {
        "STS2AI/ENV/Sim/HeadlessSim/Simulation/FullRunApiStateBuilder.cs",
        "STS2AI/ENV/Sim/HeadlessSim/Simulation/FullRunApiStateDtos.cs",
        "STS2AI/ENV/Sim/HeadlessSim/Simulation/FullRunSimulatorRuntimeFacade.cs",
    }

    builder = sections[
        "STS2AI/ENV/Sim/HeadlessSim/Simulation/FullRunApiStateBuilder.cs"
    ]
    dto = sections[
        "STS2AI/ENV/Sim/HeadlessSim/Simulation/FullRunApiStateDtos.cs"
    ]
    runtime = sections[
        "STS2AI/ENV/Sim/HeadlessSim/Simulation/FullRunSimulatorRuntimeFacade.cs"
    ]
    assert "SortedDictionary<string, FullRunApiCardOption>" in builder
    assert "JsonSerializer.Serialize(option)" in builder
    assert "existing.quantity = checked((existing.quantity ?? 1) + 1);" in builder
    assert "option.quantity = 1;" in builder
    assert "result[index].index = index;" in builder
    assert "Take(" not in builder
    assert "public int? quantity { get; set; }" in dto
    assert "SafeBuildPileCards(cards, shuffle: false)" in builder
    assert (
        "+\t\t\t|| projected.deck.Sum(static card => card.quantity ?? 1) "
        "!= player.Deck.Cards.Count)"
    ) in runtime
    assert "-\t\t\t|| projected.deck.Count != player.Deck.Cards.Count)" in runtime

def test_verifies_lock_and_exact_binary_bytes(tmp_path: Path) -> None:
    executable, identity_path, lock_path, _ = _simulator(tmp_path)

    identity = verify_headless_simulator(
        executable,
        identity_path=identity_path,
        lock_path=lock_path,
    )

    assert identity.executable == executable.resolve()
    assert identity.source_commit == "1" * 40
    assert identity.binary_sha256 == sha256_file(executable)
    assert identity.managed_assembly_sha256 == sha256_file(executable.with_suffix(".dll"))
    assert identity.build_configuration == "Release"


def test_missing_sidecar_is_refused_without_path_or_version_fallback(tmp_path: Path) -> None:
    executable = tmp_path / "HeadlessSim.exe"
    executable.write_bytes(b"fake")
    executable.with_suffix(".dll").write_bytes(b"fake-managed-code")

    with pytest.raises(
        SimulatorIdentityError,
        match=r"sidecar is missing.*unverified binaries are refused",
    ):
        verify_headless_simulator(executable, lock_path=_lock(tmp_path)[0])


@pytest.mark.parametrize(
    "mutation",
    ["source", "binary", "managed_binary", "configuration"],
)
def test_wrong_source_binary_or_build_configuration_is_refused(
    tmp_path: Path,
    mutation: str,
) -> None:
    executable, identity_path, lock_path, payload = _simulator(tmp_path)
    if mutation == "source":
        source = payload["source"]
        assert isinstance(source, dict)
        source["commit"] = "f" * 40
        identity_path.write_text(json.dumps(payload), encoding="utf-8")
        expected = "source commit mismatch"
    elif mutation == "binary":
        executable.write_bytes(b"different-binary")
        expected = "size mismatch|SHA-256 mismatch"
    elif mutation == "managed_binary":
        executable.with_suffix(".dll").write_bytes(b"different-managed-binary")
        expected = "managed assembly size mismatch|managed assembly SHA-256 mismatch"
    else:
        build = payload["build"]
        assert isinstance(build, dict)
        build["configuration"] = "Debug"
        identity_path.write_text(json.dumps(payload), encoding="utf-8")
        expected = "configuration must be 'Release'"

    with pytest.raises(SimulatorIdentityError, match=expected):
        verify_headless_simulator(
            executable,
            identity_path=identity_path,
            lock_path=lock_path,
        )


def test_verified_identity_can_be_persisted_as_external_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable, identity_path, lock_path, _ = _simulator(tmp_path)
    artifact_root = tmp_path / "artifacts"
    monkeypatch.setenv("STS2_ARTIFACT_ROOT", str(artifact_root))
    identity = verify_headless_simulator(
        executable,
        identity_path=identity_path,
        lock_path=lock_path,
    )

    audit = write_preflight_audit(identity)

    payload = json.loads(audit.read_text(encoding="utf-8"))
    assert audit.parent == artifact_root / "logs" / "simulator-preflight"
    assert payload["event"] == "simulator_identity_verified"
    assert payload["binary"]["sha256"] == identity.binary_sha256
    assert payload["managed_binary"]["sha256"] == identity.managed_assembly_sha256


def test_training_cli_gates_and_pins_headless_executable_before_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from sts2_rl import train as train_module

    executable = _simulator_for_repository_lock(tmp_path)
    artifact_root = tmp_path / "artifacts"
    monkeypatch.setenv("STS2_ARTIFACT_ROOT", str(artifact_root))
    received: list[TrainingConfig] = []

    def _run(config: TrainingConfig, **_kwargs: object) -> TrainingState:
        received.append(config)
        return TrainingState()

    monkeypatch.setattr(train_module, "run_training", _run)
    monkeypatch.setattr(
        train_module,
        "run_runtime_mechanics_preflight",
        lambda _executable: {
            "schema": "sts2-runtime-mechanics-audit-v1",
            "runtime_event_checked": True,
            "runtime_combat_checked": True,
        },
    )

    assert train_module.main(["--backend", "headless", "--sim-exe", str(executable), "--steps", "1"]) == 0

    assert len(received) == 1
    config = received[0]
    assert config.environment.sim_exe_path == str(executable.resolve())
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert events[0]["event"] == "simulator_identity_verified"
    assert Path(events[0]["audit_path"]).is_file()
    assert events[1]["event"] == "runtime_mechanics_verified"
    assert Path(events[1]["audit_path"]).is_file()
    assert events[-1]["status"] == "complete"
