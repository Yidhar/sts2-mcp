"""CLI entrypoint for the recurrent V-trace v4 baseline."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, replace
from pathlib import Path

from sts2_env.headless_sim_bridge_client import HeadlessSimError, resolve_headless_sim_exe
from sts2_rl.artifacts import resolve_external_input_path
from sts2_rl.runtime_mechanics import (
    RuntimeMechanicsAuditError,
    run_runtime_mechanics_preflight,
    write_runtime_mechanics_audit,
)
from sts2_rl.simulator_identity import (
    SimulatorIdentityError,
    verify_headless_simulator,
    write_preflight_audit,
)
from sts2_rl.training.config import TrainingConfig, load_training_config
from sts2_rl.training.runtime import inspect_baseline, run_training


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m sts2_rl.train",
        description=(
            "Train the recurrent legal-candidate V-trace baseline with a "
            "bounded asynchronous rollout queue."
        ),
    )
    parser.add_argument("--profile", default="default", help="built-in TOML profile")
    parser.add_argument("--config", help="additional versioned TOML config")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help="strict dotted config override; repeatable",
    )
    parser.add_argument("--device", help="override runtime device (auto/cpu/cuda)")
    parser.add_argument(
        "--collector-device",
        help="asynchronous actor device (default: cpu)",
    )
    parser.add_argument("--steps", type=int, help="override total environment steps")
    parser.add_argument("--seed", type=int, help="override deterministic seed")
    parser.add_argument("--backend", choices=("live", "headless"))
    parser.add_argument("--sim-exe", help="headless simulator executable")
    parser.add_argument(
        "--sim-identity",
        help="HeadlessSim identity sidecar (default: <sim-exe>.identity.json)",
    )
    parser.add_argument("--session-path", help="live bridge session descriptor path")
    parser.add_argument("--resume", help="exact baseline checkpoint directory")
    parser.add_argument(
        "--initialize-from",
        help=(
            "compatible atomic checkpoint model for a fresh lineage; "
            "never exact resume"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate config/model/encoder without launching the game",
    )
    parser.add_argument("--print-config", action="store_true")
    return parser


def _cli_overrides(config: TrainingConfig, args: argparse.Namespace) -> TrainingConfig:
    runtime = config.runtime
    if args.device is not None:
        runtime = replace(runtime, device=str(args.device))
    if args.collector_device is not None:
        runtime = replace(runtime, collector_device=str(args.collector_device))
    if args.steps is not None:
        runtime = replace(runtime, total_environment_steps=int(args.steps))
    if args.seed is not None:
        runtime = replace(runtime, seed=int(args.seed))

    environment = config.environment
    if args.backend is not None:
        environment = replace(environment, backend=args.backend)
    if args.sim_exe is not None:
        environment = replace(environment, sim_exe_path=str(args.sim_exe))
    if args.session_path is not None:
        environment = replace(environment, session_path=str(args.session_path))
    return replace(config, runtime=runtime, environment=environment)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_training_config(
        profile=args.profile,
        config_path=args.config,
        overrides=tuple(args.overrides),
    )
    config = _cli_overrides(config, args)
    if args.print_config:
        print(json.dumps(config.to_mapping(), indent=2, sort_keys=True))
    if args.dry_run:
        print(json.dumps(inspect_baseline(config), indent=2, sort_keys=True))
        return 0
    if args.sim_identity and config.environment.backend != "headless":
        raise SystemExit("--sim-identity is only valid for the headless backend")
    if config.environment.backend == "headless":
        try:
            simulator = verify_headless_simulator(
                resolve_headless_sim_exe(config.environment.sim_exe_path),
                identity_path=args.sim_identity,
            )
            audit_path = write_preflight_audit(simulator)
            mechanics_summary = run_runtime_mechanics_preflight(simulator.executable)
            mechanics_audit_path = write_runtime_mechanics_audit(mechanics_summary)
        except (
            HeadlessSimError,
            OSError,
            RuntimeMechanicsAuditError,
            SimulatorIdentityError,
            ValueError,
        ) as exc:
            raise SystemExit(f"HeadlessSim preflight failed: {exc}") from exc
        config = replace(
            config,
            environment=replace(
                config.environment,
                sim_exe_path=str(simulator.executable),
            ),
        )
        print(
            json.dumps(
                {
                    "event": "simulator_identity_verified",
                    "audit_path": str(audit_path),
                    **simulator.to_mapping(),
                },
                sort_keys=True,
            )
        )
        print(
            json.dumps(
                {
                    "event": "runtime_mechanics_verified",
                    "audit_path": str(mechanics_audit_path),
                    **mechanics_summary,
                },
                sort_keys=True,
            )
        )
    resume: Path | None = None
    initialization: Path | None = None
    if args.resume and args.initialize_from:
        raise SystemExit("--resume and --initialize-from are mutually exclusive")
    if args.resume:
        resume = resolve_external_input_path(args.resume)
    if args.initialize_from:
        initialization = resolve_external_input_path(args.initialize_from)
    final_state = run_training(
        config,
        resume_from=resume,
        initialize_from=initialization,
    )
    print(json.dumps({"status": "complete", **asdict(final_state)}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
