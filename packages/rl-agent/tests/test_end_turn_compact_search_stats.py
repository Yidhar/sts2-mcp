"""Regression tests for preserving EndTurn audit keys in episode search stats."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from muzero.diagnostics.episode_metrics import EpisodeMetricsMixin


def test_compact_search_stats_keeps_full_energy_guard_and_pre_dispatch_keys() -> None:
    compact = EpisodeMetricsMixin._compact_search_stats(
        {
            "num_simulations": 0,
            "combat_quality_full_energy_endturn_guard_applied": 1.0,
            "combat_quality_full_energy_endturn_guard_legal_generation_gap_suspect": 1.0,
            "combat_quality_end_turn_pre_dispatch_full_energy_skip_suspect": 1.0,
            "combat_quality_end_turn_pre_dispatch_legal_generation_gap": 0.0,
            "combat_quality_safe_progress_skip_selected": 1.0,
            "combat_quality_energy_ratio": 1.0,
            "unrelated_debug_blob": 999.0,
        }
    )

    assert compact["combat_quality_full_energy_endturn_guard_applied"] == 1.0
    assert compact["combat_quality_full_energy_endturn_guard_legal_generation_gap_suspect"] == 1.0
    assert compact["combat_quality_end_turn_pre_dispatch_full_energy_skip_suspect"] == 1.0
    assert compact["combat_quality_end_turn_pre_dispatch_legal_generation_gap"] == 0.0
    assert compact["combat_quality_safe_progress_skip_selected"] == 1.0
    assert compact["combat_quality_energy_ratio"] == 1.0
    assert "unrelated_debug_blob" not in compact
