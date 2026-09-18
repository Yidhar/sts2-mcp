"""Bounded natural full-game evaluation, never a counterfactual training label.

Uses the production semantic executor and frozen actor. Evidence is compact;
the final factual state is retained. Administrative limits are censoring, not
defeats, and absent typed outcomes are never inferred from floor or HP.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import time
from pathlib import Path
from typing import Any

from sts2_env.headless_sim_bridge_client import resolve_headless_sim_exe
from sts2_rl.artifacts import resolve_artifact_path
from sts2_rl.backends import HeadlessBackend
from sts2_rl.encoding import grounding_encoding_identity
from sts2_rl.macro_branch_diagnostic import BranchLimits, run_branch_pass
from sts2_rl.macro_root_diagnostic import EvidenceTrace, ProbeBudgetExhausted, validate_seed
from sts2_rl.macro_root_policy import FrozenMacroRootPolicy
from sts2_rl.simulator_identity import verify_headless_simulator
from sts2_rl.train_macro_credit import _json, load_macro_training_config


class EpisodeTrace(EvidenceTrace):
    """Streaming selected facts; explicitly not a full-replay/clone proof."""

    def __init__(self, output: Any, limits: Any, *, progress_path: Path | None = None) -> None:
        super().__init__(output, limits)
        self.last_state: dict[str, Any] | None = None
        self.weight = 1.0
        self.observed_return = 0.0
        self.steps = 0
        self.progress_path = progress_path
        self.last_floor: int | None = None

    def record(self, kind: str, facts: Any, policy: Any = None) -> None:
        state = facts.get("state") if isinstance(facts, dict) else None
        if isinstance(state, dict):
            self.last_state = state
        if kind == "transition":
            value = float(facts["reward_breakdown"]["reward"])
            discount = float(facts["native_discount"]["discount"])
            if not math.isfinite(value) or not 0 <= discount <= 1:
                raise ValueError("non-finite reward or invalid factual discount")
            self.observed_return += self.weight * value
            self.weight *= discount
        if isinstance(facts, dict) and isinstance(facts.get("native_steps"), int):
            self.steps = facts["native_steps"]
        if state is None:
            return
        compact = {"kind": kind, **self.metrics()}
        super().record(kind, compact)
        floor = compact["ending_floor"]
        if self.progress_path and (floor != self.last_floor or self.steps % 50 == 0 or state.get("terminated")):
            _json(self.progress_path, {"state": "evaluating", **compact})
        self.last_floor = floor

    def metrics(self) -> dict[str, Any]:
        state = self.last_state or {}
        obs = state.get("observation", {})
        run, player, counters = obs.get("run", {}), obs.get("player", {}), obs.get("_training", {})
        transition = state.get("transition") or {}
        typed = transition.get("facts", {})
        return {"ending_act": run.get("act"), "ending_floor": run.get("floor"),
                "hp": player.get("hp"), "max_hp": player.get("max_hp"),
                "revivals_used": counters.get("revivals_used"),
                "player_hp_lost": counters.get("player_hp_lost"),
                "native_steps": self.steps, "terminated": state.get("terminated"),
                "truncated": state.get("truncated"), "terminal_reason": state.get("terminal_reason"),
                "run_result": typed.get("run_result"),
                "observed_discounted_return": self.observed_return}


def evaluate_episode(*, backend: Any, policy: Any, config: Any, seed: int,
                     output: Path, limits: BranchLimits) -> dict[str, Any]:
    validate_seed(seed)
    if not limits.terminal_only:
        raise ValueError("full-game evaluation requires terminal_only; no macro horizon")
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    with (output / "trajectory.jsonl").open("x", encoding="utf-8") as handle:
        trace = EpisodeTrace(handle, limits.trace_limits(), progress_path=output / "progress.json")
        # Prefix evidence is not used for same-root comparison in this evaluator.
        prefix = EpisodeTrace(io.StringIO(), limits.trace_limits())
        result: dict[str, Any] = {"seed": seed, "status": "failed", "training_writes": False,
                                 "scope": "natural greedy reset-to-terminal; revival-assisted reward-v9",
                                 "tail_value_used": False}
        try:
            passed = run_branch_pass(backend=backend, policy=policy, config=config, seed=seed,
                                     limits=limits, trace=trace, prefix=prefix)
            result["ending"] = passed["ending"]
            metrics = trace.metrics()
            result["status"] = (
                "complete" if passed["ending"] == "factual_terminal" and metrics["terminated"] is True
                and metrics["truncated"] is False and metrics["run_result"] in {"victory", "defeat"}
                else "censored" if passed["ending"] in {"environment_truncated", "macro_horizon"}
                else "failed")
        except ProbeBudgetExhausted as error:
            result.update(status="censored", ending="administrative_limit", error=str(error))
        except Exception as error:
            result.update(status="failed", ending="execution_error", error_type=type(error).__name__, error=str(error))
        finally:
            result.update(trace.metrics(), elapsed_seconds=time.monotonic() - started)
            _json(output / "final-state.json", trace.last_state or {})
            _json(output / "summary.json", result)
            _json(output / "progress.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sim-exe", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--timeout-seconds", type=float, default=600)
    parser.add_argument("--max-native-steps", type=int, default=8000)
    args = parser.parse_args(argv)
    config = load_macro_training_config(args.config).training
    output = resolve_artifact_path(args.output_dir)
    if output.exists():
        raise FileExistsError("evaluation refuses an existing output directory")
    simulator = verify_headless_simulator(resolve_headless_sim_exe(args.sim_exe))
    policy = FrozenMacroRootPolicy(args.model_path, config, args.device)
    limits = BranchLimits(terminal_only=True, max_native_steps=args.max_native_steps,
                          timeout_seconds=args.timeout_seconds, max_candidates=64, max_trace_bytes=16 * 1024 * 1024)
    backend = HeadlessBackend(exe_path=simulator.executable, protocol_quarantine_dir=output / "protocol-incidents")
    try:
        result = evaluate_episode(backend=backend, policy=policy, config=config, seed=args.seed, output=output, limits=limits)
        result.update(source_policy=policy.provenance, simulator=simulator.to_mapping(),
                      encoding=grounding_encoding_identity(), revival_budget=config.curriculum.revival_budget)
        _json(output / "summary.json", result)
    finally:
        backend.close()
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
