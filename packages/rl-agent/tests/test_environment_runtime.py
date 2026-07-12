from __future__ import annotations

from sts2_env.environment_runtime import EnvironmentRuntimeMixin


class FakeBridge:
    base_url = "headless_sim://fake"

    def __init__(self) -> None:
        self.close_count = 0

    def close(self) -> None:
        self.close_count += 1


class Runtime(EnvironmentRuntimeMixin):
    def __init__(self) -> None:
        self.bridge = FakeBridge()
        self._initialize_environment_runtime()


def test_runtime_records_canonical_backend_transition() -> None:
    runtime = Runtime()
    breakdown = runtime._record_backend_transition(
        {
            "episode_id": "ep",
            "step_index": 3,
            "reward": 1.25,
            "obs": {"phase": "combat"},
            "legal_actions": [],
            "info": {},
        },
        action_handle="end_turn",
    )
    assert breakdown.total == 1.25
    assert runtime.last_canonical_transition is not None
    assert runtime.last_canonical_transition.backend_name == "headless_sim"


def test_runtime_close_is_idempotent() -> None:
    runtime = Runtime()
    runtime._close_environment_runtime()
    runtime._close_environment_runtime()
    assert runtime.bridge.close_count == 1
