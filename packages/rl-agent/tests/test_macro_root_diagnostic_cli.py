from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("torch")

from sts2_rl import macro_root_diagnostic as probe  # noqa: E402
from sts2_rl.training import factory  # noqa: E402
from sts2_rl.training.config import TrainingConfig, load_training_config  # noqa: E402


def _forbidden(*args: Any, **kwargs: Any) -> Any:
    raise AssertionError("CLI preflight must not construct training resources or an unverified simulator")


@pytest.fixture
def config() -> TrainingConfig:
    experiment = Path(probe.__file__).resolve().parents[1] / "config" / "experiments"
    return load_training_config(
        profile="preheat", config_path=experiment / "hybrid_vtrace_v1_reward_v9_budget64_on_policy.toml"
    )


@pytest.fixture
def artifact_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "artifacts"
    root.mkdir()
    monkeypatch.setattr(probe, "artifact_root", lambda: root)
    monkeypatch.setattr(factory, "build_training_resources", _forbidden)
    monkeypatch.setattr(probe, "HeadlessBackend", _forbidden)
    return root


def _args(output: Path, *model_args: str) -> list[str]:
    return [
        "--config", "course.toml", "--seed", "123", "--output-dir", str(output),
        "--device", "cpu", *model_args,
    ]


class _Simulator:
    executable = Path("verified-simulator.exe")

    def to_mapping(self) -> dict[str, Any]:
        return {"identity": "verified-test-identity"}


@pytest.mark.parametrize(
    "model_args",
    [(), ("--model-path", "vtrace.pt", "--initialize-from-iql", "iql.pt")],
)
def test_cli_requires_exactly_one_explicit_model_source(
    artifact_root: Path, monkeypatch: pytest.MonkeyPatch, model_args: tuple[str, ...]
) -> None:
    output = artifact_root / "probe"
    monkeypatch.setattr(probe, "load_training_config", _forbidden)
    monkeypatch.setattr(probe, "verify_headless_simulator", _forbidden)
    with pytest.raises(SystemExit) as error:
        probe.main(_args(output, *model_args))
    assert error.value.code == 2
    assert not output.exists()


@pytest.mark.parametrize("source_flag", ["--model-path", "--initialize-from-iql"])
def test_cli_routes_explicit_iql_flag_and_frozen_publication_without_training(
    artifact_root: Path, config: TrainingConfig, monkeypatch: pytest.MonkeyPatch, source_flag: str
) -> None:
    output = artifact_root / "probe"
    events: list[str] = []
    policy_calls: list[tuple[Path, bool]] = []
    monkeypatch.setattr(probe, "load_training_config", lambda **kwargs: config)
    monkeypatch.setattr(probe, "resolve_headless_sim_exe", lambda path: Path("requested-simulator.exe"))

    def verify(path: Path) -> _Simulator:
        assert path == Path("requested-simulator.exe")
        events.append("verified")
        return _Simulator()

    def policy(path: Path, course: TrainingConfig, device: str, *, initialize_from_iql: bool) -> Any:
        assert events == ["verified"]
        assert course is config and device == "cpu"
        events.append("policy")
        policy_calls.append((path, initialize_from_iql))

        return SimpleNamespace(provenance={"exact_resume": False, "training_state_loaded": False})

    def run_probe(**kwargs: Any) -> dict[str, Any]:
        assert events == ["verified", "policy"]
        assert kwargs["config"] is config
        assert kwargs["seed"] == 123
        assert kwargs["output_dir"] == output.resolve()
        assert callable(kwargs["backend_factory"])
        events.append("probe")
        return {"schema": "sts2-macro-root-b0-v1", "status": "matched", "training_writes": False}

    monkeypatch.setattr(probe, "verify_headless_simulator", verify)
    monkeypatch.setattr(probe, "FrozenMacroRootPolicy", policy)
    monkeypatch.setattr(probe, "run_probe", run_probe)
    assert probe.main(_args(output, source_flag, "source.pt")) == 0
    assert policy_calls == [(Path("source.pt"), source_flag == "--initialize-from-iql")]
    assert events == ["verified", "policy", "probe"]
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "matched"
    assert summary["training_writes"] is False
    assert summary["policy"]["exact_resume"] is False
    assert summary["simulator"] == {"identity": "verified-test-identity"}


@pytest.mark.parametrize("failure_stage", ["resolve", "verify", "policy"])
def test_cli_preflight_failure_is_recorded_and_never_constructs_backend_or_training(
    artifact_root: Path, config: TrainingConfig, monkeypatch: pytest.MonkeyPatch, failure_stage: str
) -> None:
    output = artifact_root / "probe"
    events: list[str] = []
    monkeypatch.setattr(probe, "load_training_config", lambda **kwargs: config)
    monkeypatch.setattr(probe, "run_probe", _forbidden)

    def resolve(path: Any) -> Path:
        events.append("resolve")
        if failure_stage == "resolve":
            raise FileNotFoundError("test missing simulator")
        return Path("unverified.exe")

    def verify(path: Path) -> _Simulator:
        assert path == Path("unverified.exe")
        events.append("verify")
        if failure_stage == "verify":
            raise ValueError("test simulator identity mismatch")
        return _Simulator()

    def policy(*args: Any, **kwargs: Any) -> Any:
        events.append("policy")
        if failure_stage != "policy":
            raise AssertionError("model must not load before simulator identity is verified")
        raise ValueError("test publication reward identity mismatch")

    monkeypatch.setattr(probe, "resolve_headless_sim_exe", resolve)
    monkeypatch.setattr(probe, "verify_headless_simulator", verify)
    monkeypatch.setattr(probe, "FrozenMacroRootPolicy", policy)
    assert probe.main(_args(output, "--model-path", "source.pt")) == 2
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["schema"] == "sts2-macro-root-b0-v1"
    assert summary["status"] == "failed"
    assert summary["error"].startswith("test ")
    expected_stages = {"resolve": ["resolve"], "verify": ["resolve", "verify"], "policy": ["resolve", "verify", "policy"]}
    assert events == expected_stages[failure_stage]
    assert {path.name for path in output.iterdir()} == {"summary.json"}


def test_output_must_be_new_and_cannot_replace_existing_evidence(artifact_root: Path) -> None:
    output = artifact_root / "existing"
    output.mkdir()
    sentinel = output / "summary.json"
    sentinel.write_text("keep this evidence", encoding="utf-8")
    with pytest.raises(FileExistsError):
        probe.new_output_dir(output)
    assert sentinel.read_text(encoding="utf-8") == "keep this evidence"


def test_output_must_be_below_artifact_root_not_root_or_sibling(artifact_root: Path) -> None:
    outside = artifact_root.parent / "not-artifacts"
    for path in (artifact_root, outside):
        with pytest.raises(ValueError, match="STS2_ARTIFACT_ROOT"):
            probe.new_output_dir(path)
    assert not outside.exists()


def test_output_cannot_enter_checkout_even_if_artifact_root_is_misconfigured(monkeypatch: pytest.MonkeyPatch) -> None:
    checkout = Path(probe.__file__).resolve().parents[3]
    fake_root = checkout / "not-a-real-b0-artifact-root"
    output = fake_root / "not-created"
    monkeypatch.setattr(probe, "artifact_root", lambda: fake_root)
    with pytest.raises(ValueError, match="outside the checkout"):
        probe.new_output_dir(output)
    assert not output.exists()


def test_output_resolution_rejects_parent_traversal(artifact_root: Path) -> None:
    output = artifact_root / ".." / "escape"
    with pytest.raises(ValueError, match="STS2_ARTIFACT_ROOT"):
        probe.new_output_dir(output)
    assert not output.resolve().exists()


def test_output_accepts_only_a_new_external_child(artifact_root: Path) -> None:
    output = artifact_root / "b0" / "new-run"
    assert probe.new_output_dir(output) == output.resolve()
    assert output.is_dir() and list(output.iterdir()) == []


@pytest.mark.parametrize("invalid_course", ["act1", "combat", "live", "no-revival"])
def test_b0_rejects_non_run_or_non_headless_course(config: TrainingConfig, invalid_course: str) -> None:
    if invalid_course == "act1":
        invalid = replace(config, curriculum=replace(config.curriculum, reward_objective="act1"))
    elif invalid_course == "combat":
        invalid = replace(
            config,
            environment=replace(config.environment, scenario="combat"),
            curriculum=replace(config.curriculum, reward_objective="combat"),
        )
    elif invalid_course == "live":
        invalid = TrainingConfig(environment=replace(config.environment, backend="live"))
    else:
        invalid = replace(config, curriculum=replace(config.curriculum, revival_budget=0))
    with pytest.raises(ValueError, match="B0"):
        probe.validate_course(invalid)


def test_b0_accepts_the_current_budget64_reward_v9_course(config: TrainingConfig) -> None:
    assert probe.validate_course(config) == 64


@pytest.mark.parametrize("seed", [-1, 2_147_483_648])
def test_cli_rejects_invalid_seed_before_output_policy_or_simulator_creation(
    artifact_root: Path, config: TrainingConfig, monkeypatch: pytest.MonkeyPatch, seed: int
) -> None:
    output = artifact_root / "probe"
    monkeypatch.setattr(probe, "load_training_config", lambda **kwargs: config)
    for name in ("new_output_dir", "resolve_headless_sim_exe", "verify_headless_simulator", "FrozenMacroRootPolicy", "run_probe"):
        monkeypatch.setattr(probe, name, _forbidden)
    argv = _args(output, "--model-path", "source.pt")
    argv[argv.index("--seed") + 1] = str(seed)
    with pytest.raises(ValueError, match="seed must be a signed-int32 non-negative integer"):
        probe.main(argv)
    assert not output.exists()


@pytest.mark.parametrize("seed", [-1, 2_147_483_648, True, 1.25, "0"])
def test_direct_probe_rejects_invalid_seed_before_backend_factory(
    artifact_root: Path, config: TrainingConfig, seed: Any
) -> None:
    output = artifact_root / "probe"
    output.mkdir()
    backend_calls: list[str] = []

    def backend_factory() -> Any:
        backend_calls.append("constructed")
        return _forbidden()

    policy = SimpleNamespace(initial_state=_forbidden, forward=_forbidden)
    summary = probe.run_probe(
        backend_factory=backend_factory,
        policy=policy,
        config=config,
        seed=seed,
        limits=probe.ProbeLimits(),
        output_dir=output,
    )
    assert summary["status"] == "failed"
    assert summary["error_type"] == "ValueError"
    assert summary["error"] == "seed must be a signed-int32 non-negative integer"
    assert summary["passes"] == []
    assert summary["training_writes"] is False
    assert backend_calls == []
    assert list(output.iterdir()) == []


@pytest.mark.parametrize("seed", [0, 2_147_483_647])
def test_seed_validation_accepts_signed_int32_boundaries(seed: int) -> None:
    probe.validate_seed(seed)
