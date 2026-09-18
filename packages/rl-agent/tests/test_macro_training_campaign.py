import argparse
import json
import sys
from pathlib import Path

import pytest
import torch

from sts2_rl import macro_training_campaign as campaign


def test_budget_and_partition_fail_closed() -> None:
    campaign.validate_budget(8, 1200, [101, 103, 105, 107], 64)
    for hours, seconds, seeds in [(1, 1200, [101, 103, 105, 107]), (8, 1200, [100]),
                                   (8, 1200, [101, 101]), (float("nan"), 1200, [101])]:
        with pytest.raises(ValueError):
            campaign.validate_budget(hours, seconds, seeds, 64)


def test_censored_pairs_are_not_complete_game_deltas() -> None:
    before = [{"seed": 101, "status": "complete", "run_result": "defeat", "ending_floor": 20,
               "revivals_used": 64, "observed_discounted_return": -1.0},
              {"seed": 103, "status": "censored", "ending_floor": 20, "observed_discounted_return": 0.1}]
    after = [{"seed": 101, "status": "complete", "run_result": "victory", "ending_floor": 50,
              "revivals_used": 40, "observed_discounted_return": 0.5},
             {"seed": 103, "status": "complete", "run_result": "victory", "ending_floor": 50,
              "observed_discounted_return": 0.5}]
    result = campaign.paired_comparison(before, after)
    assert result["complete_pairs"] == 1
    assert result["before"]["censored"] == 1
    assert result["pairs"][0]["after_minus_before"]["revivals_used"] == -24
    assert result["pairs"][1]["after_minus_before"] == {}
    assert result["automatic_promotion"] is False


def _args(tmp_path: Path) -> argparse.Namespace:
    parent = tmp_path / "parent.pt"
    torch.save({"format": "sts2-hybrid-macro-credit-publication-v1", "policy_version": 20}, parent)
    return argparse.Namespace(
        config=Path(__file__).resolve().parents[1] / "config/experiments/hybrid_macro_credit_v2_reward_v9_long.toml",
        parent=parent, output_dir=tmp_path / "campaign", run_dir=tmp_path / "run", sim_exe=tmp_path / "sim",
        lineage_id="test-new-lineage", device="cpu", hours=8, episode_seconds=1200,
        max_native_steps=8000, max_updates=2, evaluation_seeds=[101, 103, 105, 107])


@pytest.mark.parametrize("training_failure", [False, True])
def test_campaign_orders_baseline_training_and_paired_final(tmp_path, monkeypatch, training_failure) -> None:
    args = _args(tmp_path)
    parent_bytes = args.parent.read_bytes()
    monkeypatch.setattr(campaign, "resolve_artifact_path", lambda path: path)
    commands = []
    trained = 0

    def command(argv, *, log_dir, timeout, on_tick):
        nonlocal trained
        commands.append(argv)
        on_tick()
        if "sts2_rl.macro_episode_evaluation" in argv:
            dest = Path(argv[argv.index("--output-dir") + 1])
            seed = int(argv[argv.index("--seed") + 1])
            dest.mkdir(parents=True)
            campaign._json(dest / "summary.json", {"seed": seed, "status": "complete", "ending": "factual_terminal",
                                                  "ending_floor": 50, "run_result": "victory", "revivals_used": 12})
        else:
            if trained == 0:
                assert "--initialize-from-macro-publication" in argv
            else:
                assert "--resume" in argv
            if not training_failure:
                trained += 1
            args.run_dir.mkdir(exist_ok=True)
            torch.save({"format": "sts2-hybrid-macro-credit-publication-v1", "policy_version": trained},
                       args.run_dir / "publication.pt")
            if training_failure:
                campaign._json(args.run_dir / "status.json", {"state": "aborted", "error": "injected error"})
                return {"exit_code": 1, "budget_interrupted": False}
        return {"exit_code": 0, "budget_interrupted": False}

    monkeypatch.setattr(campaign, "supervised_command", command)
    result = campaign.run_campaign(args)
    assert result["state"] == ("failed_training_evaluated_last_commit" if training_failure else "completed")
    assert result["committed_new_updates"] == (0 if training_failure else 2)
    assert len(result["baseline"]) == len(result["final"]) == 4
    assert result["comparison"]["complete_pairs"] == 4
    phases = [argv[2] for argv in commands]
    assert phases[:4] == ["sts2_rl.macro_episode_evaluation"] * 4
    assert phases[4:-4] == ["sts2_rl.train_macro_credit"] * (1 if training_failure else 2)
    assert args.parent.read_bytes() == parent_bytes
    assert json.loads((args.output_dir / "plan.json").read_text())["automatic_promotion"] is False


def test_all_censored_baseline_stops_before_training(tmp_path, monkeypatch) -> None:
    args = _args(tmp_path)
    monkeypatch.setattr(campaign, "resolve_artifact_path", lambda path: path)

    def command(argv, **kwargs):
        assert "sts2_rl.macro_episode_evaluation" in argv
        dest = Path(argv[argv.index("--output-dir") + 1])
        dest.mkdir(parents=True)
        campaign._json(dest / "summary.json", {"seed": int(argv[argv.index("--seed") + 1]),
                                              "status": "censored", "ending": "administrative_limit"})
        return {"exit_code": 2, "budget_interrupted": False}

    monkeypatch.setattr(campaign, "supervised_command", command)
    result = campaign.run_campaign(args)
    assert result["state"] == "failed"
    assert "all baseline games censored" in result["error"]
    assert not args.run_dir.exists()


def test_supervisor_records_exit_without_background_service(tmp_path) -> None:
    result = campaign.supervised_command([sys.executable, "-c", "print('synthetic child')"],
                                         log_dir=tmp_path / "invocation", timeout=10)
    assert result["exit_code"] == 0
    assert result["budget_interrupted"] is False
    assert "synthetic child" in (tmp_path / "invocation/stdout.log").read_text()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signal supervision")
def test_supervisor_interrupts_budget_exhaustion(tmp_path) -> None:
    result = campaign.supervised_command([sys.executable, "-c", "import time; time.sleep(60)"],
                                         log_dir=tmp_path / "invocation", timeout=0.5)
    assert result["budget_interrupted"] is True
    assert result["elapsed_seconds"] < 10
    assert result["exit_code"] != 0
