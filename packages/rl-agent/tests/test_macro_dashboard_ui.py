from __future__ import annotations

# ruff: noqa: RUF001
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

HTML_PATH = Path(__file__).parents[1] / "sts2_rl/dashboard_static/index.html"


def _html() -> str:
    return HTML_PATH.read_text(encoding="utf-8")


def _node(script: str) -> str:
    node = shutil.which("node")
    if not node:
        pytest.skip("JavaScript behavioral checks require Node.js")
    result = subprocess.run([node, "-"], input=script, encoding="utf-8", capture_output=True, timeout=20, check=False)
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_shared_legacy_page_keeps_legacy_route_and_adds_explicit_macro_mode() -> None:
    html = _html()
    assert '<title>STS2 训练监控</title>' in html
    assert 'document.body.dataset.dashboardMode === "macro-credit"' in html
    assert 'const API_RUNS = "/api/v1/runs"' in html
    assert 'if (MACRO_MODE) initializeMacroDashboard();' in html
    macro_refresh = html.split("if (MACRO_MODE) {", 1)[1].split("const runPayload", 1)[0]
    assert "fetchJson(API_SNAPSHOT)" in macro_refresh
    assert "renderMacroSnapshot(snapshot)" in macro_refresh
    assert "return;" in macro_refresh
    assert "?run=" not in macro_refresh
    assert "innerHTML" not in html
    assert "function renderLineChart(" in html
    assert "function renderEpisodes(" in html  # Old complete-episode views remain intact.


def test_gameplay_comparison_controls_and_measurement_boundaries_are_visible() -> None:
    html = _html()
    for control in ("macro-baseline-select", "macro-current-select", "macro-root-select"):
        assert f'id="{control}"' in html
    for boundary in (
        "不同 Update 使用不同 seed / root", "采集 teacher", "Root 后实测折扣回报",
        "从 reset 累计", "不含末尾 V", "未评估（不是 0%）", "不重复计入主趋势",
        "更新后概率不是更新后游戏回报", "进展奖励", "components.progress_reward ?? components.potential_reward",
    ):
        assert boundary in html


def test_pure_ui_projection_counts_only_reference_and_preserves_unknown() -> None:
    html = _html()
    functions = html.split("      function macroNumber(value) {", 1)[1].split("      function macroWindowLabel", 1)[0]
    functions = "function macroNumber(value) {" + functions
    fixture = {"history": [
        {"update": 1, "metrics": {"loss": .5}, "gameplay": {"roots": [
            {"root_id": "a", "teacher_version": 0, "natural_reference": {"observed_discounted_return": -.2},
             "same_action_control": {"observed_discounted_return": -.2}, "candidates": [{"y": -1}, {"y": -2}]},
        ]}},
        {"update": 2, "gameplay": {"roots": [{"root_id": "b", "teacher_version": 1,
                                               "natural_reference": {"observed_discounted_return": None}}]}},
        {"update": 3, "gameplay": {"roots": [{"root_id": "pending"}]}},
    ]}
    result = json.loads(_node(functions + "\n" + f"const rows = macroWindowRows({json.dumps(fixture)});" + """
console.log(JSON.stringify({
  keys: rows.map(row => row.key), teachers: rows.map(row => row.root.teacher_version),
  missing: rows[1].ref.observed_discounted_return,
  zero: macroDifference(0, 0), unknown: macroDifference(null, 0),
  numericString: macroDifference('1', 0), boolean: macroDifference(true, 0),
  horizon: macroEnding('macro_horizon')
}));
"""))
    assert result["keys"] == ["1/a", "2/b"]
    assert result["teachers"] == [0, 1]
    assert result["missing"] is None
    assert result["zero"] == 0
    assert result["unknown"] is None
    assert result["numericString"] is None
    assert result["boolean"] is None
    assert result["horizon"] == "宏观窗口截止（非胜负）"


def test_complete_shared_page_javascript_parses() -> None:
    scripts = re.findall(r"<script[^>]*>(.*?)</script>", _html(), re.S)
    assert scripts
    node = shutil.which("node")
    if not node:
        pytest.skip("JavaScript syntax checks require Node.js")
    result = subprocess.run([node, "--check"], input="\n".join(scripts), encoding="utf-8",
                            capture_output=True, timeout=20, check=False)
    assert result.returncode == 0, result.stderr


def test_missing_user_selected_window_does_not_silently_follow_another() -> None:
    function = _html().split("      function macroSelect(", 1)[1].split("      function macroTable", 1)[0]
    script = """
const select = { value: '', disabled: false, options: [],
  replaceChildren() { this.options = []; }, append(option) { this.options.push(option); } };
const element = () => select;
const document = { createElement() { return {}; } };
const macroWindowLabel = row => row.key;
""" + "function macroSelect(" + function + """
const selected = macroSelect('test', [{key:'new'}], 'pinned-missing', true);
console.log(JSON.stringify({selected: selected ?? null, value:select.value, labels:select.options.map(x=>x.textContent)}));
"""
    result = json.loads(_node(script))
    assert result["selected"] is None
    assert result["value"] == "pinned-missing"
    assert "未自动切换" in result["labels"][-1]


def test_charts_explicitly_break_at_unknown_or_missing_updates() -> None:
    assert _html().count("breakBefore: missing || gap") == 2


def test_game_outcomes_are_on_first_screen_and_do_not_call_cutoff_defeat() -> None:
    html = _html()
    assert html.index('id="macro-performance-title"') < html.index('id="overview-title"')
    for field in ("macro-ending-floor-value", "macro-revival-value", "macro-hp-loss-value", "macro-stop-value"):
        assert f'id="{field}"' in html
    functions = "function macroNumber(value) {" + html.split("      function macroNumber(value) {", 1)[1].split("      function macroWindowLabel", 1)[0]
    result = json.loads(_node(functions + """
const known = macroHeadline({ref:{ending_floor:6,last_total_revivals:0,player_hp_lost_since_reset:63,ending:'macro_horizon'}});
const missing = macroHeadline(undefined);
console.log(JSON.stringify({known,missing}));
"""))
    assert result["known"] == {
        "floor": 6, "revivals": 0, "hpLoss": 63, "stopLabel": "训练窗口截断", "fullGameQuality": "尚不能判断是否变强",
    }
    assert result["missing"]["floor"] is None
    assert result["missing"]["revivals"] is None
    assert result["missing"]["hpLoss"] is None
