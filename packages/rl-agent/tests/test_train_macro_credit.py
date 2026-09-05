from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from torch import nn

from sts2_rl import train_macro_credit as cli
from sts2_rl.joint.macro_checkpoint import MacroCheckpointIdentity, load_macro_checkpoint
from sts2_rl.joint.macro_credit_learner import MacroCreditLearner
from sts2_rl.macro_root_diagnostic import FrozenPolicy, ProbeBudgetExhausted


def _config() -> cli.MacroTrainingConfig:
    return cli.load_macro_training_config(cli.DEFAULT_CONFIG)


def _identity(config: cli.MacroTrainingConfig) -> MacroCheckpointIdentity:
    return MacroCheckpointIdentity(config.as_mapping(), {"synthetic": True}, {"reward": "v9"}, "unit-lineage", "unit-v1")


def test_default_course_uses_fresh_training_partition_and_long_horizon() -> None:
    config = _config()
    assert config.branches.continuation_macro_decisions >= 12
    assert config.training.curriculum.revival_budget == 64
    assert not config.training.transaction_learning.enabled
    assert not config.training.episodic_learning.enabled
    assert cli.seed_at(0) == 13_400_000
    assert cli.seed_at(4) == 13_400_008
    assert config.as_mapping()["seed_recipe"]["partition"] == "training-even"


@pytest.mark.parametrize("override", [
    "course.unknown=1", "course.roots_per_cohort=true", "course.maximum_root_seconds=-1",
    "branches.continuation_macro_decisions=0", "branches.max_candidates=99999",
    "training.environment.scenario='combat'", "training.transaction_learning.enabled=true",
    "training.optimization.discount=0.99", "learner.macro_trace_lambda=1.5",
    "version='old'",
])
def test_course_override_validation_is_strict(override: str) -> None:
    with pytest.raises((ValueError, TypeError)):
        cli.load_macro_training_config(cli.DEFAULT_CONFIG, [override])


@pytest.mark.parametrize("cursor", [-1, True, 2_147_483_647])
def test_invalid_seed_cursor_is_rejected(cursor: Any) -> None:
    with pytest.raises(ValueError):
        cli.seed_at(cursor)


@pytest.mark.parametrize("device", ["auto", "mps", "cpu:0", "cuda:-1"])
def test_no_automatic_device_fallback(device: str) -> None:
    with pytest.raises(ValueError):
        cli.explicit_device(device)


def test_unavailable_accelerator_never_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="fallback"):
        cli.explicit_device("cuda")


def _args(output: Path, source: Path) -> list[str]:
    return ["--output-dir", str(output), "--lineage-id", "unit-lineage", "--device", "cpu",
            "--initialize-from-iql", str(source)]


def test_initialization_and_resume_are_mutually_exclusive(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as error:
        cli.main([*_args(tmp_path / "run", tmp_path / "source.pt"), "--resume", str(tmp_path / "checkpoint.pt")])
    assert error.value.code == 2


def test_dry_preflight_creates_no_artifacts_and_never_claims_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("STS2_ARTIFACT_ROOT", str(tmp_path))
    source = tmp_path / "source.pt"
    source.write_bytes(b"not loaded during static preflight")
    simulator = SimpleNamespace(executable=tmp_path / "HeadlessSim.exe", to_mapping=lambda: {"verified": True})
    monkeypatch.setattr(cli, "resolve_headless_sim_exe", lambda _: simulator.executable)
    monkeypatch.setattr(cli, "verify_headless_simulator", lambda _: simulator)
    monkeypatch.setattr(cli, "FrozenMacroRootPolicy", lambda *a, **kw: pytest.fail("dry preflight loaded model"))
    output = tmp_path / "run"
    assert cli.main([*_args(output, source), "--dry-run"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["state"] == "preflight_only"
    assert result["training_ready"] is False
    assert result["source_weights_validated"] is False
    assert not output.exists()


def test_existing_output_is_never_overwritten(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STS2_ARTIFACT_ROOT", str(tmp_path))
    source = tmp_path / "source.pt"
    source.write_bytes(b"source")
    output = tmp_path / "run"
    output.mkdir()
    sentinel = output / "status.json"
    sentinel.write_text("original", encoding="utf-8")
    assert cli.main(_args(output, source)) == 1
    assert sentinel.read_text(encoding="utf-8") == "original"


def test_output_lock_protects_other_writer_and_records_owned_failure(tmp_path: Path) -> None:
    output = tmp_path / "run"
    with pytest.raises(RuntimeError):
        with cli.output_owner(output, resume=False):
            with pytest.raises(FileExistsError):
                with cli.output_owner(output, resume=True):
                    pytest.fail("second writer acquired the lock")
            raise RuntimeError("injected initialization error")
    assert not (output / ".writer.lock").exists()
    assert json.loads((output / "status.json").read_text())["state"] == "aborted"


class FakeLearner:
    def __init__(self, events: list[str]) -> None:
        self.model = nn.Linear(2, 1)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.01)
        self.update = 0
        self.events = events

    def learn(self, cohort: Any, *, resource_guard: Any = None) -> dict[str, Any]:
        if resource_guard is not None:
            resource_guard()
        self.events.append(f"learn:{self.update}")
        assert cohort.teacher_version == self.update
        self.optimizer.zero_grad(set_to_none=True)
        self.model(torch.ones(1, 2)).square().sum().backward()  # type: ignore[no-untyped-call]
        self.optimizer.step()
        self.update += 1
        return {"update": self.update, "loss": 0.5}


class FakeCohort:
    def __init__(self, cohort_id: str, teacher_version: int, roots: tuple[Any, ...]) -> None:
        self.cohort_id = cohort_id
        self.teacher_version = teacher_version
        self.roots = roots

    def storage_nbytes(self) -> int:
        return 10 * len(self.roots)


def _prepare_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, events: list[str], *, roots: int = 2,
) -> tuple[Path, cli.MacroTrainingConfig, FakeLearner]:
    output = tmp_path / "run"
    output.mkdir()
    base = _config()
    config = replace(base, course=replace(base.course, roots_per_cohort=roots))
    learner = FakeLearner(events)
    monkeypatch.setattr(cli, "MacroCreditCohort", FakeCohort)

    def collect(**kwargs: Any) -> tuple[Any, dict[str, Any]]:
        events.append(f"collect:{kwargs['teacher_version']}:{kwargs['seed']}")
        assert kwargs["teacher_version"] == learner.update
        assert kwargs["seed"] % 2 == 0
        reference = SimpleNamespace(steps=(SimpleNamespace(control_domain="macro"), SimpleNamespace(control_domain="combat")))
        return SimpleNamespace(seed=kwargs["seed"], reference=reference, candidates=(1, 2)), {"status": "complete"}

    monkeypatch.setattr(cli, "collect_macro_credit_root", collect)
    save, publish = cli.save_macro_checkpoint, cli.publish_macro_checkpoint

    def traced_save(*args: Any, **kwargs: Any) -> dict[str, Any]:
        events.append(f"save:{kwargs['update']}:{kwargs['next_seed_index']}")
        return save(*args, **kwargs)

    def traced_publish(*args: Any, **kwargs: Any) -> dict[str, Any]:
        result = publish(*args, **kwargs)
        events.append(f"publish:{result['policy_version']}")
        return result

    monkeypatch.setattr(cli, "save_macro_checkpoint", traced_save)
    monkeypatch.setattr(cli, "publish_macro_checkpoint", traced_publish)
    return output, config, learner


def _run(root: Path, output: Path, config: cli.MacroTrainingConfig, learner: FakeLearner,
         *, cursor: int = 0, initialize: bool = True) -> dict[str, Any]:
    return cli.run_cohorts(
        config=config, output=output, identity=_identity(config), learner=cast(MacroCreditLearner, learner),
        teacher=cast(FrozenPolicy, object()), backend_factory=lambda: pytest.fail("synthetic test started simulator"),
        initial_seed_index=cursor, initialization={"mode": "synthetic-test"}, max_updates=1,
        initialize_boundary=initialize, root=root,
    )


def test_collect_update_commit_publication_order_and_seed_cursor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []
    output, config, learner = _prepare_run(tmp_path, monkeypatch, events)
    result = _run(tmp_path, output, config, learner)
    assert events == ["save:0:0", "publish:0", "collect:0:13400000", "collect:0:13400002",
                      "learn:0", "save:1:2", "publish:1"]
    assert result["committed_update"] == 1
    assert result["next_seed_index"] == 2
    assert result["quality_certified"] is False
    assert result["long_run_certified"] is False


def test_partial_cohort_never_updates_or_advances_committed_cursor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []
    output, config, learner = _prepare_run(tmp_path, monkeypatch, events)
    original = cli.collect_macro_credit_root
    calls = 0

    def fail_second(**kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ProbeBudgetExhausted("synthetic step budget")
        return original(**kwargs)

    monkeypatch.setattr(cli, "collect_macro_credit_root", fail_second)
    with pytest.raises(ProbeBudgetExhausted):
        _run(tmp_path, output, config, learner)
    assert learner.update == 0
    payload = torch.load(output / "checkpoint.pt", weights_only=False)
    assert payload["next_seed_index"] == 0
    assert payload["update"] == 0
    status = json.loads((output / "status.json").read_text())
    assert status["state"] == "aborted"
    assert status["next_seed_index"] == 0


def test_resume_repairs_publication_before_collecting_next_even_seed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []
    output, config, learner = _prepare_run(tmp_path, monkeypatch, events, roots=1)
    _run(tmp_path, output, config, learner)
    (output / "publication.pt").write_bytes(b"stale")
    payload = load_macro_checkpoint(output / "checkpoint.pt", identity=_identity(config),
                                    model=learner.model, optimizer=learner.optimizer, artifact_root=tmp_path)
    learner.update = payload["update"]
    events.clear()
    result = _run(tmp_path, output, config, learner, cursor=payload["next_seed_index"], initialize=False)
    assert events == ["publish:1", "collect:1:13400002", "learn:1", "save:2:2", "publish:2"]
    assert result["committed_update"] == 2


def test_seed_cursor_boundary_mismatch_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output, config, learner = _prepare_run(tmp_path, monkeypatch, [])
    with pytest.raises(ValueError, match="boundary"):
        _run(tmp_path, output, config, learner, cursor=1)


def test_resource_guard_aborts_on_disk_floor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli.shutil, "disk_usage", lambda _: SimpleNamespace(free=0))
    with pytest.raises(ProbeBudgetExhausted, match="minimum_free_disk_bytes"):
        cli.ResourceGuard(tmp_path, cli.CourseLimits()).check(force=True)


def test_missing_natural_combat_never_enters_joint_update(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []
    output, config, learner = _prepare_run(tmp_path, monkeypatch, events)
    original = cli.collect_macro_credit_root

    def macro_only(**kwargs: Any) -> Any:
        collected, report = original(**kwargs)
        collected.reference.steps = (SimpleNamespace(control_domain="macro"),)
        return collected, report

    monkeypatch.setattr(cli, "collect_macro_credit_root", macro_only)
    with pytest.raises(ValueError, match="coverage"):
        _run(tmp_path, output, config, learner)
    assert not any(event.startswith("learn") for event in events)


def test_learning_failure_preserves_previous_boundary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []
    output, config, learner = _prepare_run(tmp_path, monkeypatch, events)
    original = learner.learn

    def fail_after_update(cohort: Any, **kwargs: Any) -> dict[str, Any]:
        original(cohort, **kwargs)
        raise ProbeBudgetExhausted("synthetic postupdate budget")

    monkeypatch.setattr(learner, "learn", fail_after_update)
    with pytest.raises(ProbeBudgetExhausted):
        _run(tmp_path, output, config, learner)
    payload = torch.load(output / "checkpoint.pt", weights_only=False)
    assert payload["update"] == 0
    assert payload["next_seed_index"] == 0
    assert not any(event.startswith("save:1") for event in events)


def test_publication_failure_does_not_undo_committed_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []
    output, config, learner = _prepare_run(tmp_path, monkeypatch, events)
    original = cli.publish_macro_checkpoint
    calls = 0

    def fail_second(*args: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("synthetic publication failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(cli, "publish_macro_checkpoint", fail_second)
    with pytest.raises(OSError):
        _run(tmp_path, output, config, learner)
    payload = torch.load(output / "checkpoint.pt", weights_only=False)
    assert payload["update"] == 1
    assert payload["next_seed_index"] == 2
    status = json.loads((output / "status.json").read_text())
    assert status["state"] == "aborted"
    assert status["committed_update"] == 1
    assert status["next_seed_index"] == 2


def test_source_manifest_is_explicitly_partial_and_excludes_tests_docs() -> None:
    manifest = cli.source_manifest()
    assert manifest["identity"]["complete_dependency_lock"] is False
    records = manifest["identity"]["files"]
    assert len(records) >= 8
    assert all(record["path"].startswith("sts2_rl/") for record in records)
    assert all(len(record["sha256"]) == 64 for record in records)
    assert "git_head_provenance_only" in manifest
