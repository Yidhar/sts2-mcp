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
from sts2_rl.training.config import (
    TrainingConfig,
    engine_revival_identity,
    load_training_config,
)
from sts2_rl.training.launch_contract import (
    SupervisedLaunchContract,
    SupervisedLaunchContractError,
    load_supervised_launch_contract,
    validate_supervised_runtime_readiness,
    validate_supervised_source_authority,
    validate_supervised_trainer_environment,
)
from sts2_rl.training.runtime import inspect_baseline, run_training


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m sts2_rl.train",
        description=("Train the recurrent legal-candidate V-trace baseline with a bounded asynchronous rollout queue."),
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
        help=("compatible atomic checkpoint model for a fresh lineage; never exact resume"),
    )
    parser.add_argument(
        "--model-initialization-attestation",
        help=(
            "pinned prior full-byte verification manifest for a trusted immutable "
            "model-initialization source"
        ),
    )
    parser.add_argument(
        "--model-initialization-attestation-sha256",
        help="expected SHA-256 of --model-initialization-attestation",
    )
    parser.add_argument(
        "--launch-contract",
        help="absolute immutable supervised-launch contract path",
    )
    parser.add_argument(
        "--launch-contract-sha256",
        help="expected SHA-256 of --launch-contract; both arguments are required together",
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
    if bool(args.launch_contract) != bool(args.launch_contract_sha256):
        raise SystemExit("--launch-contract and --launch-contract-sha256 must be supplied together")
    if bool(args.model_initialization_attestation) != bool(
        args.model_initialization_attestation_sha256
    ):
        raise SystemExit(
            "--model-initialization-attestation and its SHA-256 must be supplied together"
        )
    supervised_launch: SupervisedLaunchContract | None = None
    if args.launch_contract is not None:
        try:
            supervised_launch = load_supervised_launch_contract(
                args.launch_contract,
                expected_sha256=str(args.launch_contract_sha256),
            )
            validate_supervised_trainer_environment(
                supervised_launch,
            )
            # These checks intentionally happen before config loading, backend
            # construction, run-directory creation, or any training-state
            # initialization.  The external contract seal binds exact clean
            # launch commit B while source_authority proves the reviewed
            # generation-commit-A closure.  Runtime readiness is independent
            # non-training evidence and is re-opened/re-hashed in this process.
            validate_supervised_source_authority(supervised_launch)
            validate_supervised_runtime_readiness(supervised_launch)
        except SupervisedLaunchContractError as exc:
            raise SystemExit(f"supervised launch contract failed: {exc}") from exc
        if not args.initialize_from or args.resume:
            raise SystemExit("a supervised launch contract requires --initialize-from and forbids --resume")
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
    runtime_provenance: dict[str, object] = {
        "backend": config.environment.backend,
        "training_revival": (engine_revival_identity() if config.curriculum.revival_mechanism is not None else None),
    }
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
        runtime_provenance = {
            "backend": "headless",
            "training_revival": (
                engine_revival_identity() if config.curriculum.revival_mechanism is not None else None
            ),
            "simulator_identity": simulator.to_mapping(),
            "simulator_identity_audit_path": str(audit_path),
            "runtime_mechanics": mechanics_summary,
            "runtime_mechanics_audit_path": str(mechanics_audit_path),
        }
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
    if args.model_initialization_attestation and not args.initialize_from:
        # A byte attestation is an independent, pinned proof for one immutable
        # model-initialization source.  It is deliberately usable by the
        # durable launcher supervisor, whose manifest fixes the trainer
        # command but which does not inject the separate trainer-side
        # ``--launch-contract`` protocol.  Requiring that unrelated protocol
        # here made the reviewed v41 command pass both launcher preflights and
        # then fail before run_start.  The attestation validator itself binds
        # the checkpoint root, manifest/metadata digests and source counters;
        # exact resume remains ineligible for this fast path.
        raise SystemExit(
            "a model-initialization attestation requires --initialize-from"
        )
    if args.resume:
        resume = resolve_external_input_path(args.resume)
    if args.initialize_from:
        initialization = resolve_external_input_path(args.initialize_from)
    if supervised_launch is not None:
        # run_training writes this mapping to run_start and forwards the same
        # execution provenance to every periodic/final checkpoint.  Do not
        # rely only on its run_start-only convenience injection: checkpoint
        # metadata must retain the exact implementation/readiness authority.
        runtime_provenance["supervised_launch"] = supervised_launch.provenance_mapping()
    final_state = run_training(
        config,
        resume_from=resume,
        initialize_from=initialization,
        model_initialization_attestation=(
            resolve_external_input_path(args.model_initialization_attestation)
            if args.model_initialization_attestation
            else None
        ),
        model_initialization_attestation_sha256=(
            str(args.model_initialization_attestation_sha256)
            if args.model_initialization_attestation_sha256
            else None
        ),
        runtime_provenance=runtime_provenance,
        supervised_launch_contract=supervised_launch,
    )
    print(json.dumps({"status": "complete", **asdict(final_state)}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
