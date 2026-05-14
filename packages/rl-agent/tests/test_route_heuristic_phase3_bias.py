"""Phase 3 prior-bias plumbing tests (recovery 2026-05-09).

Per the post-implementation review (P0 + P2 in the latest pass) we MUST
verify:

* ``MCTS.set_route_heuristic_bias`` slot lifecycle — None clears it; a
  vector replaces it; the slot is consumed AT THE START of ``run()`` so
  any early return inside MCTS still leaves a clean slot for the next
  decision.
* the synchronous self-play ``metric_name_map`` includes the three Phase 3 stats
  (``route_heuristic_bias_applied`` / ``_abs_mean`` / ``_max_abs``).
  Without these the bias is invisible from TB at long-train time.
* The async-actor writer ``search_suffix_map`` mirrors the same keys so
  multi-actor training also surfaces them.

These tests intentionally avoid spinning up a full MCTS network — the
slot lifecycle is testable directly on the class, and the metric maps
are static dicts that can be parsed offline. Anything that
needs torch goes through smoke validation, not unit tests.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _read_train_py() -> str:
    return (ROOT / "muzero" / "train.py").read_text(encoding="utf-8-sig")


def _read_cli_main_py() -> str:
    return (ROOT / "muzero" / "training" / "cli_main.py").read_text(encoding="utf-8-sig")


def _read_cli_args_py() -> str:
    return (ROOT / "muzero" / "training" / "cli_args.py").read_text(encoding="utf-8-sig")


def _read_async_telemetry_py() -> str:
    return (ROOT / "muzero" / "training" / "async_telemetry.py").read_text(encoding="utf-8-sig")


def _read_self_play_py() -> str:
    return (ROOT / "muzero" / "training" / "self_play.py").read_text(encoding="utf-8-sig")


def test_metric_name_map_includes_route_heuristic_bias_keys():
    """P0 (review): the synchronous trainer's metric_name_map must emit
    the three Phase 3 stats so TB shows them. Pre-fix the map was missing
    all three — the bias was applied but invisible."""
    src = _read_self_play_py()
    # Match the metric_name_map block (the synchronous one).
    block_match = re.search(
        r"metric_name_map\s*=\s*\{(.+?)\}\s*\n\s*for stat_key, writer_suffix in metric_name_map",
        src,
        re.DOTALL,
    )
    assert block_match, "metric_name_map block not found"
    block = block_match.group(1)
    for key in (
        "route_heuristic_bias_applied",
        "route_heuristic_bias_abs_mean",
        "route_heuristic_bias_max_abs",
    ):
        assert key in block, f"metric_name_map missing key {key!r}"


def test_async_writer_suffix_map_includes_route_heuristic_bias_keys():
    """Same gate for the async-actor TB writer (search_suffix_map).
    Without it, async runs lose Phase 3 visibility."""
    src = _read_async_telemetry_py()
    block_match = re.search(
        r"search_suffix_map\s*=\s*\{(.+?)\}\s*\n\s*(?:search_suffix_map\.update\(.+?\)\s*\n\s*)?domain_search_means\s*=",
        src,
        re.DOTALL,
    )
    assert block_match, "search_suffix_map block not found"
    block = block_match.group(1)
    for key in (
        "route_heuristic_bias_applied",
        "route_heuristic_bias_abs_mean",
        "route_heuristic_bias_max_abs",
    ):
        assert key in block, f"search_suffix_map missing key {key!r}"


# ---------------------------------------------------------------------------
# MCTS slot lifecycle. These exercise the slot directly without
# instantiating the full MCTS — which would drag in torch + a network.
# We use a stub class that mirrors the relevant attributes / methods.
# ---------------------------------------------------------------------------

class _MctsSlotStub:
    """Reproduce just the slot lifecycle from MCTS for testing without torch."""

    def __init__(self):
        self._pending_route_heuristic_bias = None

    # Copy of the real setter implementation.
    def set_route_heuristic_bias(self, bias_vector):
        if bias_vector is None:
            self._pending_route_heuristic_bias = None
        else:
            self._pending_route_heuristic_bias = np.asarray(bias_vector, dtype=np.float32)

    # Copy of the consume-and-clear pattern from run().
    def _consume_pending_bias(self):
        bias = getattr(self, "_pending_route_heuristic_bias", None)
        self._pending_route_heuristic_bias = None
        return bias


def test_set_then_consume_returns_vector_and_clears():
    stub = _MctsSlotStub()
    stub.set_route_heuristic_bias([0.1, -0.2, 0.0, 0.05])
    consumed = stub._consume_pending_bias()
    assert consumed is not None
    np.testing.assert_allclose(consumed, [0.1, -0.2, 0.0, 0.05], rtol=1e-6)
    # After consume the slot is None — a second consume returns None.
    assert stub._consume_pending_bias() is None


def test_set_none_explicitly_clears_pending():
    stub = _MctsSlotStub()
    stub.set_route_heuristic_bias([0.1, 0.2])
    stub.set_route_heuristic_bias(None)
    assert stub._consume_pending_bias() is None


def test_consume_in_early_return_path_still_clears():
    """Even if the run-equivalent method returns before the bias is
    applied (e.g., single legal action), the slot must be empty for the
    next decision."""
    stub = _MctsSlotStub()
    stub.set_route_heuristic_bias([0.5, 0.0, -0.3])
    # First ``run`` consumes immediately even if subsequent logic exits.
    bias = stub._consume_pending_bias()
    # ... simulate early return path ...
    assert stub._pending_route_heuristic_bias is None
    # Next decision must explicitly push again or the slot stays empty.
    next_bias = stub._consume_pending_bias()
    assert next_bias is None
    # If caller pushes None, still cleared.
    stub.set_route_heuristic_bias(None)
    assert stub._consume_pending_bias() is None
    # Reset for sanity check.
    assert bias is not None


def test_real_mcts_class_has_setter_and_slot():
    """Smoke check on the actual MCTS class — without instantiating it,
    we still verify the slot attribute and setter exist as advertised."""
    # Importing torch is required for the MCTS module; skip on torch-less
    # environments (the lazy ``__init__.py`` keeps deck/route_heuristic
    # importable, but mcts.py itself imports torch at module level).
    import importlib

    try:
        mcts_module = importlib.import_module("muzero.sts2_env.mcts")
    except Exception as exc:  # pragma: no cover - environment-specific skip
        import pytest

        pytest.skip(f"muzero.sts2_env.mcts unavailable: {exc!r}")

    cls = getattr(mcts_module, "MCTS", None)
    assert cls is not None, "muzero.sts2_env.mcts.MCTS missing"
    assert hasattr(cls, "set_route_heuristic_bias"), "MCTS.set_route_heuristic_bias missing"
    # Verify the attribute is initialised in __init__ (we don't call it,
    # we just inspect the source).
    src = (ROOT / "muzero" / "sts2_env" / "mcts.py").read_text(encoding="utf-8-sig")
    assert "self._pending_route_heuristic_bias = None" in src
    # Verify the consume happens at the start of run() (i.e., the slot is
    # cleared BEFORE any logic that might early-return).
    assert "Consume + clear immediately so any early return below cannot" in src


def test_route_heuristic_bias_obeys_root_bias_scale_disabled():
    """When root_bias_scale == 0 the bias must be silently dropped — this
    is the global decay-schedule kill switch and Phase 3 has no business
    bypassing it."""
    # Test by inspecting the source — we don't run a full MCTS but we
    # verify the early-exit gate is present.
    src = (ROOT / "muzero" / "sts2_env" / "mcts.py").read_text(encoding="utf-8-sig")
    assert "if route_heuristic_bias_raw is not None and root_bias_scale > 0.0" in src


def test_phase3_route_heuristic_cli_flag_default_zero():
    """``--route-heuristic-bias`` must default to 0.0 so existing launches
    that don't set it stay in Phase 2 dry-run mode."""
    src = _read_cli_args_py()
    match = re.search(
        r'parser\.add_argument\(\s*["\']--route-heuristic-bias["\']\s*,\s*type=float\s*,\s*default=([\d.]+)',
        src,
    )
    assert match, "--route-heuristic-bias flag not registered"
    default_value = float(match.group(1))
    assert default_value == 0.0, f"default must be 0.0, got {default_value}"


def test_phase3_bias_not_pushed_when_weight_zero():
    """The trainer-side bias helper must short-circuit on weight=0 so
    the slot stays None and there is no per-step route scoring cost.

    The gate lives in the split training helper, not in the legacy
    ``train.py`` entrypoint.
    """
    src = (ROOT / "muzero" / "training" / "route_heuristic_telemetry.py").read_text(encoding="utf-8-sig")
    assert (
        'float(getattr(self, "route_heuristic_bias_weight", 0.0) or 0.0) <= 0.0' in src
    ), "Phase 3 trainer-side gate on route_heuristic_bias_weight missing"


def test_self_play_pushes_route_heuristic_bias_through_helper():
    """Self-play should wire the split helper into MCTS without embedding
    route scoring logic back into the rollout loop."""
    src = _read_self_play_py()
    assert "phase3_bias_vec = self._compute_route_heuristic_bias_vector" in src
    assert "self.mcts.set_route_heuristic_bias(phase3_bias_vec)" in src
