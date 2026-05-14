from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any


RL_AGENT_ROOT = Path(__file__).resolve().parents[1]
MONITOR_SRC = RL_AGENT_ROOT / "scripts" / "monitor_combat_sandbox_gate.py"
_spec = importlib.util.spec_from_file_location("monitor_combat_sandbox_gate_under_test", str(MONITOR_SRC))
assert _spec is not None
monitor = importlib.util.module_from_spec(_spec)
sys.modules["monitor_combat_sandbox_gate_under_test"] = monitor
assert _spec.loader is not None
_spec.loader.exec_module(monitor)


def _metric(
    last: float,
    *,
    avg_tail: float | None = None,
    min_tail: float | None = None,
    max_tail: float | None = None,
    n: int = 1,
    step: int = 100,
) -> dict[str, Any]:
    value = float(last)
    return {
        "n": int(n),
        "step": int(step),
        "last": value,
        "avg_tail": float(value if avg_tail is None else avg_tail),
        "min_tail": float(value if min_tail is None else min_tail),
        "max_tail": float(value if max_tail is None else max_tail),
    }


def _passing_summary(*, weak_sample_count: float = 3.0, weak_win_rate: float = 1.0) -> dict[str, dict[str, Any]]:
    return {
        "buffer/size": _metric(12_000),
        "recent_tail/64/win_rate": _metric(0.96),
        "recent_tail/64/weak_win_rate": _metric(weak_win_rate),
        "recent_tail/64/weak_sample_count": _metric(weak_sample_count),
        "recent_tail/64/normal_win_rate": _metric(0.93),
        "recent_tail/64/hard_normal_win_rate": _metric(0.80),
        "recent_tail/64/hard_normal_sample_count": _metric(4.0),
        "recent_tail/256/normal_win_rate": _metric(0.90),
        "recent_tail/256/hard_normal_win_rate": _metric(0.75),
        "recent_tail/256/hard_normal_sample_count": _metric(8.0),
        "combat_quality/bad_pure_block_selected_rate": _metric(0.0),
        "combat_quality/card_block_waste_with_progress_selected_rate": _metric(0.0),
        "combat_quality/card_no_damage_pressure_with_progress_selected_rate": _metric(0.0),
        "combat_quality/bad_end_turn_selected_rate": _metric(0.0),
        "combat_quality/wasteful_end_turn_rate": _metric(0.0),
        "combat_quality/zero_energy_x_cost_selected_rate": _metric(0.0),
        "combat_quality/hp_cost_self_lethal_selected_rate": _metric(0.0),
        "combat_quality/hp_cost_low_margin_selected_rate": _metric(0.0),
        "combat_quality/refund_no_followup_with_progress_selected_rate": _metric(0.0),
        "combat_quality/potion_low_urgency_selected_rate": _metric(0.0),
        "loss/total": _metric(12.0, max_tail=12.0),
        "loss/future_world_aux": _metric(0.2, max_tail=0.2),
        "loss/future_bank_state": _metric(0.1, max_tail=0.1),
        "loss/future_bank_delta": _metric(0.1, max_tail=0.1),
        "loss/future_bank_token_slot_source": _metric(0.0, max_tail=0.0),
        "memory/reserved_gb": _metric(8.0, max_tail=8.0),
        "memory/max_allocated_gb": _metric(7.0, max_tail=7.0),
        "memory/empty_cache_called": _metric(0.0, avg_tail=0.0),
    }


def _evaluate(summary: dict[str, dict[str, Any]], *, min_buffer: float = 10_000.0) -> dict[str, Any]:
    return monitor.evaluate_gate(
        summary,
        min_buffer=min_buffer,
        strict_weak=False,
        min_weak_samples=1.0,
        min_hard_normal_samples=1.0,
        max_reserved_gb=22.0,
        max_peak_allocated_gb=20.0,
        max_empty_cache_rate=0.2,
    )


def _find_check(gate: dict[str, Any], tag: str) -> dict[str, Any]:
    matches = [check for check in gate["checks"] if check["tag"] == tag]
    assert matches, f"missing check for {tag}"
    return matches[0]


def _find_checks(gate: dict[str, Any], tag: str) -> list[dict[str, Any]]:
    return [check for check in gate["checks"] if check["tag"] == tag]


def test_weak_rate_is_skipped_when_recent_window_has_no_weak_samples() -> None:
    gate = _evaluate(_passing_summary(weak_sample_count=0.0, weak_win_rate=0.0))

    weak_check = _find_check(gate, "recent_tail/64/weak_win_rate")
    assert weak_check["status"] == "skipped"
    assert weak_check["reason"] == "insufficient_samples"
    assert weak_check["sample_count"] == 0.0
    assert gate["verdict"] == "PASS"


def test_weak_rate_fails_when_weak_samples_are_present_but_win_rate_is_low() -> None:
    gate = _evaluate(_passing_summary(weak_sample_count=2.0, weak_win_rate=0.0))

    weak_check = _find_check(gate, "recent_tail/64/weak_win_rate")
    assert weak_check["status"] == "fail"
    assert gate["verdict"] == "FAIL"


def test_buffer_gate_waits_even_if_other_checks_are_red() -> None:
    gate = _evaluate(
        _passing_summary(weak_sample_count=2.0, weak_win_rate=0.0),
        min_buffer=20_000.0,
    )

    assert gate["verdict"] == "WAIT_BUFFER"
    assert _find_check(gate, "buffer/size")["status"] == "fail"
    assert _find_check(gate, "recent_tail/64/weak_win_rate")["status"] == "fail"


def test_gate_passes_when_required_metrics_pass_and_weak_is_sampled() -> None:
    gate = _evaluate(_passing_summary(weak_sample_count=3.0, weak_win_rate=1.0))

    assert gate["verdict"] == "PASS"
    assert _find_check(gate, "recent_tail/64/weak_win_rate")["status"] == "pass"


def test_end_turn_gate_prefers_strict_bad_metric_over_legacy_wasteful_false_positive() -> None:
    summary = _passing_summary()
    # Legacy broad wasteful can flag spare-energy / overflow-block End Turn
    # contexts.  Once the strict taxonomy metric exists, it is the source of
    # truth for gate purposes.
    summary["combat_quality/wasteful_end_turn_rate"] = _metric(0.02, avg_tail=0.02)
    summary["combat_quality/bad_end_turn_selected_rate"] = _metric(0.0, avg_tail=0.0)

    gate = _evaluate(summary)

    assert gate["verdict"] == "PASS"
    strict_check = _find_check(gate, "combat_quality/bad_end_turn_selected_rate")
    assert strict_check["status"] == "pass"
    assert _find_checks(gate, "combat_quality/wasteful_end_turn_rate") == []


def test_end_turn_gate_fails_when_strict_bad_metric_is_nonzero() -> None:
    summary = _passing_summary()
    summary["combat_quality/wasteful_end_turn_rate"] = _metric(0.0, avg_tail=0.0)
    summary["combat_quality/bad_end_turn_selected_rate"] = _metric(0.01, avg_tail=0.01)

    gate = _evaluate(summary)

    strict_check = _find_check(gate, "combat_quality/bad_end_turn_selected_rate")
    assert strict_check["status"] == "fail"
    assert gate["verdict"] == "FAIL"


def test_end_turn_gate_uses_sidecar_to_skip_false_scalar_positive() -> None:
    summary = _passing_summary()
    summary["combat_quality/bad_end_turn_selected_rate"] = _metric(0.01, avg_tail=0.01)
    payload = {
        "gate": _evaluate(summary),
        "diagnostics": {
            "end_turn_contexts": {
                "recent_rows": 12,
                "strict_bad_recent_count": 0,
                "strict_bad_total_count": 0,
            }
        },
    }

    monitor.apply_diagnostics_gate_overrides(payload)

    strict_check = _find_check(payload["gate"], "combat_quality/bad_end_turn_selected_rate")
    assert strict_check["status"] == "skipped"
    assert strict_check["reason"] == "strict_end_turn_sidecar_zero"
    assert payload["gate"]["verdict"] == "PASS"


def test_end_turn_sidecar_override_does_not_hide_strict_bad_contexts() -> None:
    summary = _passing_summary()
    summary["combat_quality/bad_end_turn_selected_rate"] = _metric(0.01, avg_tail=0.01)
    payload = {
        "gate": _evaluate(summary),
        "diagnostics": {
            "end_turn_contexts": {
                "recent_rows": 12,
                "strict_bad_recent_count": 1,
                "strict_bad_total_count": 1,
            }
        },
    }

    monitor.apply_diagnostics_gate_overrides(payload)

    strict_check = _find_check(payload["gate"], "combat_quality/bad_end_turn_selected_rate")
    assert strict_check["status"] == "fail"
    assert payload["gate"]["verdict"] == "FAIL"


def test_end_turn_sidecar_override_preserves_wait_buffer_verdict() -> None:
    summary = _passing_summary()
    summary["buffer/size"] = _metric(1_000)
    summary["combat_quality/bad_end_turn_selected_rate"] = _metric(0.01, avg_tail=0.01)
    payload = {
        "gate": _evaluate(summary, min_buffer=10_000.0),
        "diagnostics": {
            "end_turn_contexts": {
                "recent_rows": 12,
                "strict_bad_recent_count": 0,
                "strict_bad_total_count": 0,
            }
        },
    }

    monitor.apply_diagnostics_gate_overrides(payload)

    strict_check = _find_check(payload["gate"], "combat_quality/bad_end_turn_selected_rate")
    assert strict_check["status"] == "skipped"
    assert payload["gate"]["verdict"] == "WAIT_BUFFER"


def test_end_turn_gate_falls_back_to_legacy_wasteful_when_strict_metric_is_absent() -> None:
    summary = _passing_summary()
    summary.pop("combat_quality/bad_end_turn_selected_rate")
    summary["combat_quality/wasteful_end_turn_rate"] = _metric(0.02, avg_tail=0.02)

    gate = _evaluate(summary)

    legacy_check = _find_check(gate, "combat_quality/wasteful_end_turn_rate")
    assert legacy_check["status"] == "fail"
    assert gate["verdict"] == "FAIL"
