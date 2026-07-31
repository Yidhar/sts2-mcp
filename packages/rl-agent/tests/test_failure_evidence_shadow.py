from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


def _load_script() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts" / "validate_failure_evidence_shadow.py"
    spec = importlib.util.spec_from_file_location(
        "validate_failure_evidence_shadow",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _episode(*, observed: int, dropped: int, storage: int) -> dict[str, object]:
    return {
        "shadow": {
            "completion_controls_observed": observed,
            "completion_controls_dropped": dropped,
            "completion_storage_nbytes": storage,
        }
    }


def test_completion_staging_audit_applies_limits_per_episode() -> None:
    module = _load_script()
    result = module._completion_staging_audit(
        [
            _episode(observed=40, dropped=8, storage=100),
            _episode(observed=35, dropped=3, storage=100),
        ],
        maximum_controls=32,
        maximum_bytes=100,
    )

    assert result == {
        "bounded": True,
        "observed": 75,
        "dropped": 11,
        "maximum_episode_storage_nbytes": 100,
    }


@pytest.mark.parametrize(
    ("episode", "maximum_controls", "maximum_bytes"),
    [
        (_episode(observed=33, dropped=0, storage=100), 32, 100),
        (_episode(observed=1, dropped=0, storage=101), 32, 100),
        (_episode(observed=1, dropped=2, storage=1), 32, 100),
        (_episode(observed=1, dropped=0, storage=-1), 32, 100),
    ],
)
def test_completion_staging_audit_fails_closed_on_bound_violation(
    episode: dict[str, object],
    maximum_controls: int,
    maximum_bytes: int,
) -> None:
    module = _load_script()
    result = module._completion_staging_audit(
        [episode],
        maximum_controls=maximum_controls,
        maximum_bytes=maximum_bytes,
    )

    assert result["bounded"] is False


def test_completion_staging_audit_rejects_untyped_metrics() -> None:
    module = _load_script()

    with pytest.raises(RuntimeError, match="must be an integer"):
        module._completion_staging_audit(
            [
                {
                    "shadow": {
                        "completion_controls_observed": "1",
                        "completion_controls_dropped": 0,
                        "completion_storage_nbytes": 1,
                    }
                }
            ],
            maximum_controls=32,
            maximum_bytes=100,
        )
