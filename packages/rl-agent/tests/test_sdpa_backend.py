from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch

from sts2_rl.training import (
    TrainingState,
    build_training_resources,
    load_training_checkpoint,
    preflight_training_checkpoint,
    save_training_checkpoint,
)
from sts2_rl.training import sdpa as sdpa_module
from sts2_rl.training.sdpa import (
    NATIVE_ROCR_ABORT_MITIGATION_REASON,
    ROCM_SDPA_EXECUTION_ABI,
    configure_rocm_sdpa_backend,
    sdpa_transition_provenance,
)
from tests.test_v2_training_pipeline import FakeCombatBackend, _config


def _fake_sdpa_api(
    monkeypatch: pytest.MonkeyPatch,
    *,
    initial: dict[str, bool],
) -> tuple[dict[str, bool], list[tuple[str, bool]]]:
    flags = dict(initial)
    calls: list[tuple[str, bool]] = []

    def setter(name: str):
        def set_flag(enabled: bool) -> None:
            assert isinstance(enabled, bool)
            calls.append((name, enabled))
            flags[name] = enabled

        return set_flag

    cuda = sdpa_module.torch.backends.cuda
    monkeypatch.setattr(cuda, "enable_flash_sdp", setter("flash"))
    monkeypatch.setattr(
        cuda,
        "enable_mem_efficient_sdp",
        setter("memory_efficient"),
    )
    monkeypatch.setattr(cuda, "enable_math_sdp", setter("math"))
    monkeypatch.setattr(cuda, "enable_cudnn_sdp", setter("cudnn"))
    monkeypatch.setattr(cuda, "flash_sdp_enabled", lambda: flags["flash"])
    monkeypatch.setattr(
        cuda,
        "mem_efficient_sdp_enabled",
        lambda: flags["memory_efficient"],
    )
    monkeypatch.setattr(cuda, "math_sdp_enabled", lambda: flags["math"])
    monkeypatch.setattr(cuda, "cudnn_sdp_enabled", lambda: flags["cudnn"])
    return flags, calls


def _update_manifest_metadata(checkpoint: Path) -> None:
    manifest_path = checkpoint / "checkpoint.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata = (checkpoint / "metadata.json").read_bytes()
    for entry in manifest["files"]:
        if entry["path"] == "metadata.json":
            entry["size_bytes"] = len(metadata)
            entry["sha256"] = hashlib.sha256(metadata).hexdigest()
            break
    else:  # pragma: no cover - atomic publisher invariant
        raise AssertionError("metadata manifest entry missing")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_math_policy_disables_every_fused_backend_on_rocm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sdpa_module, "_hip_version", lambda: "7.2.1")
    flags, calls = _fake_sdpa_api(
        monkeypatch,
        initial={
            "flash": True,
            "memory_efficient": True,
            "math": False,
            "cudnn": True,
        },
    )

    state = configure_rocm_sdpa_backend(
        "math",
        devices=(torch.device("cuda:0"), torch.device("cpu")),
    )

    assert flags == {
        "flash": False,
        "memory_efficient": False,
        "math": True,
        "cudnn": False,
    }
    assert calls == [
        ("math", True),
        ("flash", False),
        ("memory_efficient", False),
        ("cudnn", False),
    ]
    assert state.applied
    assert state.applicability == "rocm_cuda"
    assert state.effective_backend == "math_only"
    assert state.current_flags == flags
    assert state.to_mapping()["version"] == ROCM_SDPA_EXECUTION_ABI


def test_math_policy_does_not_mutate_global_sdpa_state_on_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sdpa_module, "_hip_version", lambda: "7.2.1")

    def forbidden(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("CPU-only composition touched global CUDA SDPA state")

    cuda = sdpa_module.torch.backends.cuda
    for name in (
        "enable_flash_sdp",
        "enable_mem_efficient_sdp",
        "enable_math_sdp",
        "enable_cudnn_sdp",
        "flash_sdp_enabled",
        "mem_efficient_sdp_enabled",
        "math_sdp_enabled",
        "cudnn_sdp_enabled",
    ):
        monkeypatch.setattr(cuda, name, forbidden)

    state = configure_rocm_sdpa_backend(
        "math",
        devices=(torch.device("cpu"),),
    )

    assert not state.applied
    assert state.applicability == "not_rocm_cuda"
    assert state.effective_backend == "not_applicable"
    assert state.previous_flags is None
    assert state.current_flags is None


def test_auto_policy_observes_but_does_not_change_rocm_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sdpa_module, "_hip_version", lambda: "7.2.1")
    initial = {
        "flash": True,
        "memory_efficient": True,
        "math": True,
        "cudnn": True,
    }
    flags, calls = _fake_sdpa_api(monkeypatch, initial=initial)

    state = configure_rocm_sdpa_backend(
        "auto",
        devices=(torch.device("cuda"),),
    )

    assert calls == []
    assert flags == initial
    assert not state.applied
    assert state.effective_backend == "framework_default"
    assert state.previous_flags == initial
    assert state.current_flags == initial


def test_unrecorded_exact_resume_transition_names_rocr_mitigation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sdpa_module, "_hip_version", lambda: "7.2.1")
    _fake_sdpa_api(
        monkeypatch,
        initial={
            "flash": True,
            "memory_efficient": True,
            "math": True,
            "cudnn": True,
        },
    )
    current = configure_rocm_sdpa_backend(
        "math",
        devices=(torch.device("cuda"),),
    )

    transition = sdpa_transition_provenance(
        previous=None,
        current=current,
        checkpoint_load_mode="exact_resume",
        parent_checkpoint_present=True,
    )

    assert transition["checkpoint_load_mode"] == "exact_resume"
    assert transition["previous_label"] == "unrecorded/default"
    assert transition["previous"]["recording_status"] == "unrecorded/default"
    assert transition["current_label"] == "math_only"
    assert transition["reason"] == NATIVE_ROCR_ABORT_MITIGATION_REASON
    assert transition["changed"] is None


def test_sdpa_policy_is_execution_provenance_not_immutable_lineage() -> None:
    base = _config()
    automatic = replace(
        base,
        runtime=replace(base.runtime, rocm_sdpa_backend="auto"),
    )
    math = replace(
        base,
        runtime=replace(base.runtime, rocm_sdpa_backend="math"),
    )

    assert automatic.lineage_mapping() == math.lineage_mapping()
    assert automatic.fingerprint_sha256() != math.fingerprint_sha256()
    assert "rocm_sdpa_backend" not in math.lineage_mapping()["runtime"]


def test_legacy_unrecorded_checkpoint_exact_resumes_and_successor_records_policy(
    tmp_path: Path,
) -> None:
    base = _config()
    config = replace(
        base,
        runtime=replace(
            base.runtime,
            device="cpu",
            collector_device="cpu",
            rocm_sdpa_backend="math",
        ),
    )
    state = TrainingState(environment_steps=17, learner_updates=2, episodes=1)
    source = build_training_resources(config, backend=FakeCombatBackend())
    try:
        checkpoint = save_training_checkpoint(
            tmp_path / "recorded",
            config=config,
            resources=source,
            state=state,
            checkpoint_load_mode="fresh",
            execution_provenance={"backend": "fake"},
        )
    finally:
        source.close()

    recorded = json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))
    assert recorded["sdpa_backend"]["requested_policy"] == "math"
    assert recorded["sdpa_backend"]["effective_backend"] == "not_applicable"
    assert recorded["execution_provenance"]["backend"] == "fake"
    assert (
        recorded["execution_provenance"]["sdpa_backend"]["current"]
        == recorded["sdpa_backend"]
    )

    # Recreate the pre-policy v27 metadata shape.  Absence is accepted as
    # explicitly unrecorded execution history; immutable training state still
    # goes through the complete exact-resume loader below.
    recorded.pop("sdpa_backend")
    recorded.pop("execution_provenance")
    recorded["training_config"]["runtime"].pop("rocm_sdpa_backend")
    (checkpoint / "metadata.json").write_text(
        json.dumps(recorded, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _update_manifest_metadata(checkpoint)

    preflight_training_checkpoint(
        checkpoint,
        config=config,
        resolved_device="cpu",
        resolved_collector_device="cpu",
    )
    target = build_training_resources(config, backend=FakeCombatBackend())
    try:
        assert load_training_checkpoint(
            checkpoint,
            config=config,
            resources=target,
        ) == state
    finally:
        target.close()
