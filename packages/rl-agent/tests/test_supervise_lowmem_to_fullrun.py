import os

from scripts.supervise_lowmem_to_fullrun import (
    absolutize_without_following_symlink,
    checkpoint_freshness,
    checkpoint_step,
    latest_checkpoint,
    parse_gate_summary,
    parse_metric_last,
    parse_metric_step,
    parse_verdict,
)


def test_parse_verdict_wait_buffer():
    text = """run_dir: logs_muzero/example
scalar_tag_count: 625
tail: 50
verdict: WAIT_BUFFER
"""
    assert parse_verdict(text) == "WAIT_BUFFER"


def test_parse_metric_last_for_buffer_size():
    text = (
        "buffer/size                                                    "
        "last=3738.0 avg=3047.34 min=2358.0 max=3738.0 step=54947 n=141\n"
    )
    assert parse_metric_last(text, "buffer/size") == 3738.0


def test_parse_metric_step_for_buffer_size():
    text = (
        "buffer/size                                                    "
        "last=3738.0 avg=3047.34 min=2358.0 max=3738.0 step=54947 n=141\n"
    )
    assert parse_metric_step(text, "buffer/size") == 54947


def test_parse_gate_summary_defaults_to_monitor_error_on_missing_verdict_failure():
    summary = parse_gate_summary("traceback here", returncode=1)
    assert summary.verdict == "MONITOR_ERROR"
    assert summary.buffer_size is None
    assert summary.returncode == 1


def test_parse_gate_summary_pass_with_buffer():
    summary = parse_gate_summary(
        """verdict: PASS
buffer/size                                                    last=10042.0 avg=9000 min=1 max=10042 step=1 n=1
""",
        returncode=0,
    )
    assert summary.verdict == "PASS"
    assert summary.buffer_size == 10042.0


def test_absolutize_without_following_final_symlink(tmp_path):
    target = tmp_path / "real-python"
    target.write_text("#!/bin/sh\n", encoding="utf-8")
    link = tmp_path / "python"
    os.symlink(target.name, link)

    result = absolutize_without_following_symlink(tmp_path, "python")

    assert result == link
    assert result != target.resolve()


def test_checkpoint_step_parses_directory_name(tmp_path):
    assert checkpoint_step(tmp_path / "muzero_step_00055323") == 55323
    assert checkpoint_step(tmp_path / "not_a_checkpoint") is None


def test_latest_checkpoint_uses_highest_step(tmp_path):
    (tmp_path / "muzero_step_00001000").mkdir()
    (tmp_path / "muzero_step_00003000").mkdir()
    (tmp_path / "muzero_step_00002000").mkdir()

    path, step = latest_checkpoint(tmp_path)

    assert path == tmp_path / "muzero_step_00003000"
    assert step == 3000


def test_checkpoint_freshness_accepts_recent_low_lag_checkpoint(tmp_path):
    ckpt = tmp_path / "muzero_step_000055323"
    ckpt.mkdir()
    os.utime(ckpt, (1000.0, 1000.0))
    summary = parse_gate_summary(
        "verdict: PASS\n"
        "buffer/size last=10000.0 avg=1 min=1 max=1 step=55649 n=1\n",
        0,
    )

    freshness = checkpoint_freshness(
        checkpoint_dir=tmp_path,
        gate_summary=summary,
        max_age_sec=1800,
        max_step_lag=3000,
        now=1100.0,
    )

    assert freshness.fresh is True
    assert freshness.checkpoint_step == 55323
    assert freshness.current_step == 55649
    assert freshness.step_lag == 326


def test_checkpoint_freshness_rejects_stale_checkpoint(tmp_path):
    ckpt = tmp_path / "muzero_step_00001000"
    ckpt.mkdir()
    os.utime(ckpt, (1000.0, 1000.0))
    summary = parse_gate_summary(
        "verdict: PASS\n"
        "buffer/size last=10000.0 avg=1 min=1 max=1 step=6000 n=1\n",
        0,
    )

    freshness = checkpoint_freshness(
        checkpoint_dir=tmp_path,
        gate_summary=summary,
        max_age_sec=1800,
        max_step_lag=3000,
        now=4000.0,
    )

    assert freshness.fresh is False
    assert "age_sec" in freshness.reason
    assert "step_lag" in freshness.reason
