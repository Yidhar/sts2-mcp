from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

from sts2_rl.joint import macro_checkpoint as checkpoint
from sts2_rl.joint.macro_checkpoint import (
    MACRO_CHECKPOINT_FORMAT,
    MACRO_PUBLICATION_FORMAT,
    MacroCheckpointIdentity,
    load_macro_checkpoint,
    publish_macro_checkpoint,
    save_macro_checkpoint,
)


def _identity() -> MacroCheckpointIdentity:
    return MacroCheckpointIdentity(
        course_config={"cohort_runs": 2, "seed_base": 100, "optimization": {"learning_rate": 0.01}},
        model_config={"input": 2, "output": 1},
        reward_identity={"version": "reward-v9", "budget": 64},
        lineage_id="test-macro-lineage",
        learner_version="test-macro-learner-v1",
    )


def _model() -> tuple[nn.Linear, torch.optim.AdamW]:
    model = nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    return model, optimizer


def _step(model: nn.Linear, optimizer: torch.optim.AdamW) -> None:
    optimizer.zero_grad(set_to_none=True)
    loss = model(torch.tensor([[0.5, -0.25]])).square().sum()
    loss.backward()  # type: ignore[no-untyped-call]
    optimizer.step()


def _save(root: Path, model: nn.Module, optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    return save_macro_checkpoint(
        root / "checkpoint.pt", identity=_identity(), model=model, optimizer=optimizer,
        update=1, next_seed_index=2, initialization={"mode": "model_init", "source": str(root / "source.pt")},
        artifact_root=root,
    )


def _load(root: Path, model: nn.Module, optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    return load_macro_checkpoint(
        root / "checkpoint.pt", identity=_identity(), model=model, optimizer=optimizer, artifact_root=root,
    )


def _assert_trees_equal(expected: Any, actual: Any) -> None:
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    elif isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(expected, actual)
    elif isinstance(expected, dict):
        assert expected.keys() == actual.keys()
        for key, value in expected.items():
            _assert_trees_equal(value, actual[key])
    elif isinstance(expected, list | tuple):
        assert len(expected) == len(actual)
        for left, right in zip(expected, actual, strict=True):
            _assert_trees_equal(left, right)
    else:
        assert expected == actual


def test_checkpoint_restores_model_optimizer_and_next_update_exactly(tmp_path: Path) -> None:
    torch.manual_seed(7)
    model, optimizer = _model()
    _step(model, optimizer)
    payload = _save(tmp_path, model, optimizer)
    assert payload["format"] == MACRO_CHECKPOINT_FORMAT
    _step(model, optimizer)
    resumed, resumed_optimizer = _model()
    restored = _load(tmp_path, resumed, resumed_optimizer)
    assert restored["update"] == 1
    assert restored["next_seed_index"] == 2
    _step(resumed, resumed_optimizer)
    _assert_trees_equal(model.state_dict(), resumed.state_dict())
    _assert_trees_equal(optimizer.state_dict(), resumed_optimizer.state_dict())


def _draw_rng() -> dict[str, Any]:
    return {
        "python": [random.random() for _ in range(5)],
        "numpy": np.random.random(5),
        "torch": torch.rand(5),
        "cuda": [torch.rand(5, device=f"cuda:{index}").cpu() for index in range(torch.cuda.device_count())]
        if torch.cuda.is_available() else [],
    }


def test_checkpoint_restores_all_global_random_streams(tmp_path: Path) -> None:
    random.seed(1)
    np.random.seed(2)
    torch.manual_seed(3)
    model, optimizer = _model()
    _save(tmp_path, model, optimizer)
    expected = _draw_rng()
    _draw_rng()
    _load(tmp_path, model, optimizer)
    _assert_trees_equal(expected, _draw_rng())


def test_restore_rng_false_does_not_rewind_rng(tmp_path: Path) -> None:
    model, optimizer = _model()
    _save(tmp_path, model, optimizer)
    before = checkpoint._rng_state(checkpoint._runtime_environment(model))
    load_macro_checkpoint(
        tmp_path / "checkpoint.pt", identity=_identity(), model=model, optimizer=optimizer,
        artifact_root=tmp_path, restore_rng=False,
    )
    _assert_trees_equal(before, checkpoint._rng_state(checkpoint._runtime_environment(model)))


@pytest.mark.parametrize("field", ["course_config", "model_config", "reward_identity", "lineage_id", "learner_version"])
def test_identity_mismatch_fails_before_mutation(tmp_path: Path, field: str) -> None:
    model, optimizer = _model()
    _step(model, optimizer)
    _save(tmp_path, model, optimizer)
    other, other_optimizer = _model()
    before = checkpoint._snapshot(other.state_dict())
    value = "different" if field in {"lineage_id", "learner_version"} else {"changed": True}
    changed = replace(_identity(), **{field: value})
    with pytest.raises(ValueError, match=field):
        load_macro_checkpoint(
            tmp_path / "checkpoint.pt", identity=changed, model=other, optimizer=other_optimizer,
            artifact_root=tmp_path,
        )
    _assert_trees_equal(before, other.state_dict())


def _corrupt(root: Path, change: Callable[[dict[str, Any]], None]) -> None:
    payload = torch.load(root / "checkpoint.pt", weights_only=False)
    change(payload)
    torch.save(payload, root / "checkpoint.pt")


@pytest.mark.parametrize("field", ["python", "numpy", "torch", "accelerator_count", "device", "accelerator_names"])
def test_runtime_mismatch_fails_closed(tmp_path: Path, field: str) -> None:
    model, optimizer = _model()
    _save(tmp_path, model, optimizer)

    def change(payload: dict[str, Any]) -> None:
        payload["runtime_environment"][field] = 999 if field == "accelerator_count" else "different"

    _corrupt(tmp_path, change)
    with pytest.raises(ValueError):
        _load(tmp_path, model, optimizer)


@pytest.mark.parametrize(
    "defect", ["groups", "options", "moment_shape", "moment_type", "missing_slot", "layout", "order", "step"],
)
def test_malformed_optimizer_cannot_modify_online_model(tmp_path: Path, defect: str) -> None:
    model, optimizer = _model()
    _step(model, optimizer)
    _save(tmp_path, model, optimizer)

    def change(payload: dict[str, Any]) -> None:
        state = payload["optimizer"]
        if defect == "groups":
            state["param_groups"] = []
        elif defect == "options":
            state["param_groups"][0]["lr"] = 0.1
        elif defect == "moment_shape":
            state["state"][0]["exp_avg"] = torch.zeros(5)
        elif defect == "moment_type":
            state["state"][0]["exp_avg"] = "bad"
        elif defect == "missing_slot":
            del state["state"][0]["exp_avg_sq"]
        elif defect == "order":
            state["param_groups"][0]["params"].reverse()
        elif defect == "step":
            state["state"][0]["step"] = torch.tensor(-1.0)
        else:
            payload["optimizer_layout"]["groups"][0].reverse()

    _corrupt(tmp_path, change)
    other, other_optimizer = _model()
    _step(other, other_optimizer)
    before_model = checkpoint._snapshot(other.state_dict())
    before_optimizer = checkpoint._snapshot(other_optimizer.state_dict())
    with pytest.raises(ValueError, match="optimizer|AdamW"):
        _load(tmp_path, other, other_optimizer)
    _assert_trees_equal(before_model, other.state_dict())
    _assert_trees_equal(before_optimizer, other_optimizer.state_dict())


@pytest.mark.parametrize("where", ["model", "optimizer", "metadata"])
def test_nonfinite_checkpoint_is_rejected(tmp_path: Path, where: str) -> None:
    model, optimizer = _model()
    _step(model, optimizer)
    _save(tmp_path, model, optimizer)

    def change(payload: dict[str, Any]) -> None:
        if where == "model":
            payload["model"]["weight"][0, 0] = float("nan")
        elif where == "optimizer":
            payload["optimizer"]["state"][0]["exp_avg"][0, 0] = float("inf")
        else:
            payload["initialization"]["invalid"] = float("nan")

    _corrupt(tmp_path, change)
    with pytest.raises(ValueError, match="non-finite"):
        _load(tmp_path, model, optimizer)


def test_optimizer_apply_failure_rolls_back_optimizer_model_and_rng(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, optimizer = _model()
    _step(model, optimizer)
    _save(tmp_path, model, optimizer)
    other, other_optimizer = _model()
    _step(other, other_optimizer)
    before_model = checkpoint._snapshot(other.state_dict())
    before_optimizer = checkpoint._snapshot(other_optimizer.state_dict())
    environment = checkpoint._runtime_environment(other)
    before_rng = checkpoint._rng_state(environment)
    original = other_optimizer.load_state_dict
    calls = 0

    def fail_once(state: dict[str, Any]) -> None:
        nonlocal calls
        calls += 1
        original(state)
        if calls == 1:
            random.random()
            raise RuntimeError("injected optimizer apply failure")

    monkeypatch.setattr(other_optimizer, "load_state_dict", fail_once)
    with pytest.raises(RuntimeError, match="injected"):
        _load(tmp_path, other, other_optimizer)
    _assert_trees_equal(before_model, other.state_dict())
    _assert_trees_equal(before_optimizer, other_optimizer.state_dict())
    _assert_trees_equal(before_rng, checkpoint._rng_state(environment))


def test_nonfinite_save_preserves_previous_checkpoint(tmp_path: Path) -> None:
    model, optimizer = _model()
    _save(tmp_path, model, optimizer)
    before = (tmp_path / "checkpoint.pt").read_bytes()
    with torch.no_grad():
        model.weight.fill_(float("inf"))
    with pytest.raises(ValueError, match="non-finite"):
        _save(tmp_path, model, optimizer)
    assert (tmp_path / "checkpoint.pt").read_bytes() == before


def test_checkpoint_rejects_unrecognized_payload_fields(tmp_path: Path) -> None:
    model, optimizer = _model()
    _save(tmp_path, model, optimizer)
    _corrupt(tmp_path, lambda payload: payload.update({"pending_spool": []}))
    with pytest.raises(ValueError, match="fields differ"):
        _load(tmp_path, model, optimizer)


@pytest.mark.parametrize("failure", ["save", "replace", "fsync"])
def test_precommit_failure_preserves_previous_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str,
) -> None:
    model, optimizer = _model()
    _save(tmp_path, model, optimizer)
    previous = (tmp_path / "checkpoint.pt").read_bytes()
    _step(model, optimizer)

    def fail(*args: Any, **kwargs: Any) -> None:
        raise OSError("injected precommit failure")

    if failure == "save":
        monkeypatch.setattr(checkpoint.torch, "save", fail)
    else:
        monkeypatch.setattr(checkpoint.os, failure, fail)
    with pytest.raises(OSError, match="injected"):
        _save(tmp_path, model, optimizer)
    assert (tmp_path / "checkpoint.pt").read_bytes() == previous
    assert not list(tmp_path.glob("*.tmp"))
    assert not list(tmp_path.glob(".*.tmp"))


def test_publication_rebuild_overwrites_stale_cache_not_checkpoint(tmp_path: Path) -> None:
    model, optimizer = _model()
    payload = _save(tmp_path, model, optimizer)
    checkpoint_bytes = (tmp_path / "checkpoint.pt").read_bytes()
    publication_path = tmp_path / "publication.pt"
    publication_path.write_bytes(b"stale")
    publication = publish_macro_checkpoint(
        tmp_path / "checkpoint.pt", publication_path, identity=_identity(), artifact_root=tmp_path,
    )
    assert publication["format"] == MACRO_PUBLICATION_FORMAT
    assert publication["policy_version"] == 1
    assert publication["control_scores"] == "policy_logits"
    assert publication["course_config"] == _identity().as_mapping()["course_config"]
    assert publication["initialization"] == payload["initialization"]
    assert "optimizer" not in publication
    _assert_trees_equal(payload["model"], publication["online"])
    assert (tmp_path / "checkpoint.pt").read_bytes() == checkpoint_bytes


def test_publication_cannot_overwrite_checkpoint_or_initialization(tmp_path: Path) -> None:
    model, optimizer = _model()
    _save(tmp_path, model, optimizer)
    for target in (tmp_path / "checkpoint.pt", tmp_path / "source.pt"):
        with pytest.raises(ValueError, match="source"):
            publish_macro_checkpoint(
                tmp_path / "checkpoint.pt", target, identity=_identity(), artifact_root=tmp_path,
            )


def test_save_cannot_overwrite_initialization_source(tmp_path: Path) -> None:
    model, optimizer = _model()
    source = tmp_path / "source.pt"
    source.write_bytes(b"source must survive")
    with pytest.raises(ValueError, match="source"):
        save_macro_checkpoint(
            source, identity=_identity(), model=model, optimizer=optimizer, update=0, next_seed_index=0,
            initialization={"source": str(source)}, artifact_root=tmp_path,
        )
    assert source.read_bytes() == b"source must survive"


def test_artifact_paths_reject_escape_and_checkout(tmp_path: Path) -> None:
    model, optimizer = _model()
    checkout = Path(__file__).resolve().parents[3]
    for target in (tmp_path.parent / "escaped.pt", checkout / "must-not-create.pt"):
        with pytest.raises(ValueError, match="artifact|checkout"):
            save_macro_checkpoint(
                target, identity=_identity(), model=model, optimizer=optimizer,
                update=0, next_seed_index=0, initialization={}, artifact_root=tmp_path,
            )
        assert not target.exists()


@pytest.mark.parametrize("name,value", [("update", -1), ("update", True), ("next_seed_index", 0.5)])
def test_invalid_progress_does_not_create_checkpoint(tmp_path: Path, name: str, value: Any) -> None:
    model, optimizer = _model()
    progress = {"update": 0, "next_seed_index": 0, name: value}
    with pytest.raises(ValueError, match=name):
        save_macro_checkpoint(
            tmp_path / "checkpoint.pt", identity=_identity(), model=model, optimizer=optimizer,
            initialization={}, artifact_root=tmp_path, **progress,
        )
    assert not (tmp_path / "checkpoint.pt").exists()
