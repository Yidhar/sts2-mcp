"""Isolated execution benchmark for the T1 compiler/precision experiment.

This program deliberately does not construct an environment, learner, replay,
checkpoint, or run directory.  It exercises the production candidate model on
synthetic *active-shape* batches and emits one JSON document to stdout.  Run
each precision/compiler cell in a fresh process so compiler caches and peak
memory accounting cannot leak between cells.

Example (from the package root in the reviewed WSL ROCm environment)::

    python scripts/benchmark_t1_model_execution.py \
      --profile preheat --config config/experiments/full_run_revival_v32_budget64_mature_model_init.toml \
      --device cuda --world-tokens 96 --candidates 48 --local-tokens 12 \
      --precision bf16 --compile --fullgraph

The benchmark is evidence for an experimental execution backend only.  It is
not permission to change an active training lineage.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import statistics
import time
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor, nn
from torch.amp.autocast_mode import autocast

from sts2_rl.models import (
    CandidateTokenBatch,
    GroundedCandidateBatch,
    RecurrentCandidateModel,
    WorldTokenBatch,
)
from sts2_rl.training import load_training_config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="preheat")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--world-tokens", type=int, default=96)
    parser.add_argument("--candidates", type=int, default=48)
    parser.add_argument("--local-tokens", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument(
        "--sdpa",
        choices=("math", "auto"),
        default="math",
        help="Keep v32's stable math SDPA by default; auto is an explicit separate cell.",
    )
    parser.add_argument("--compile", action="store_true", dest="use_compile")
    parser.add_argument("--fullgraph", action="store_true")
    parser.add_argument(
        "--compile-mode",
        choices=("default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"),
        default="default",
    )
    parser.add_argument("--forward-only", action="store_true")
    parser.add_argument("--seed", type=int, default=20260803)
    return parser.parse_args()


def _positive(value: int, *, label: str) -> int:
    if isinstance(value, bool) or value <= 0:
        raise SystemExit(f"{label} must be a positive integer")
    return value


def _ids(shape: tuple[int, ...], size: int, *, device: torch.device) -> Tensor:
    return torch.randint(0, size, shape, dtype=torch.long, device=device)


def _synthetic_batch(
    model: RecurrentCandidateModel,
    *,
    batch_size: int,
    world_count: int,
    action_count: int,
    local_count: int,
    device: torch.device,
) -> GroundedCandidateBatch:
    """Create a valid active-shape batch, including empty local-token rows."""

    cfg = model.config
    world_mask = torch.rand(batch_size, world_count, device=device) > 0.15
    world_mask[:, 0] = True
    action_mask = torch.rand(batch_size, action_count, device=device) > 0.10
    action_mask[:, 0] = True
    local_mask = torch.rand(batch_size, action_count, local_count, device=device) > 0.35
    # Exercise the branchless/fallback-sensitive path used by candidates with
    # no local entity tokens.  This is also the current fullgraph stress case.
    local_mask[:, 0, :] = False

    world = WorldTokenBatch(
        features=torch.randn(batch_size, world_count, cfg.token_feature_dim, device=device),
        mask=world_mask,
        type_ids=_ids((batch_size, world_count), cfg.type_vocab_size, device=device),
        role_ids=_ids((batch_size, world_count), cfg.role_vocab_size, device=device),
        owner_ids=_ids((batch_size, world_count), cfg.owner_vocab_size, device=device),
        entity_ids=_ids((batch_size, world_count), cfg.entity_vocab_size, device=device),
        entity_aux_ids=_ids((batch_size, world_count), cfg.entity_vocab_size, device=device),
        definition_binding_ids=_ids((batch_size, world_count), cfg.entity_vocab_size, device=device),
        relation_binding_ids=_ids((batch_size, world_count), cfg.entity_vocab_size, device=device),
        zone_ids=_ids((batch_size, world_count), cfg.zone_vocab_size, device=device),
        order_ids=_ids((batch_size, world_count), cfg.order_vocab_size, device=device),
    )
    local_shape = (batch_size, action_count, local_count)
    candidate_shape = (batch_size, action_count)
    candidates = CandidateTokenBatch(
        features=torch.randn(*candidate_shape, cfg.token_feature_dim, device=device),
        type_ids=_ids(candidate_shape, cfg.type_vocab_size, device=device),
        role_ids=_ids(candidate_shape, cfg.role_vocab_size, device=device),
        owner_ids=_ids(candidate_shape, cfg.owner_vocab_size, device=device),
        entity_ids=_ids(candidate_shape, cfg.entity_vocab_size, device=device),
        entity_aux_ids=_ids(candidate_shape, cfg.entity_vocab_size, device=device),
        definition_binding_ids=_ids(candidate_shape, cfg.entity_vocab_size, device=device),
        relation_binding_ids=_ids(candidate_shape, cfg.entity_vocab_size, device=device),
        zone_ids=_ids(candidate_shape, cfg.zone_vocab_size, device=device),
        target_owner_ids=_ids(candidate_shape, cfg.owner_vocab_size, device=device),
        target_entity_ids=_ids(candidate_shape, cfg.entity_vocab_size, device=device),
        target_entity_aux_ids=_ids(candidate_shape, cfg.entity_vocab_size, device=device),
        target_definition_binding_ids=_ids(candidate_shape, cfg.entity_vocab_size, device=device),
        target_relation_binding_ids=_ids(candidate_shape, cfg.entity_vocab_size, device=device),
        local_features=torch.randn(*local_shape, cfg.token_feature_dim, device=device),
        local_mask=local_mask,
        local_type_ids=_ids(local_shape, cfg.type_vocab_size, device=device),
        local_role_ids=_ids(local_shape, cfg.role_vocab_size, device=device),
        local_owner_ids=_ids(local_shape, cfg.owner_vocab_size, device=device),
        local_entity_ids=_ids(local_shape, cfg.entity_vocab_size, device=device),
        local_entity_aux_ids=_ids(local_shape, cfg.entity_vocab_size, device=device),
        local_definition_binding_ids=_ids(local_shape, cfg.entity_vocab_size, device=device),
        local_relation_binding_ids=_ids(local_shape, cfg.entity_vocab_size, device=device),
        local_zone_ids=_ids(local_shape, cfg.zone_vocab_size, device=device),
        local_order_ids=_ids(local_shape, cfg.order_vocab_size, device=device),
        action_mask=action_mask,
    )
    return GroundedCandidateBatch(
        world=world,
        candidates=candidates,
        domain_ids=_ids((batch_size,), cfg.domain_count, device=device),
    )


class _TensorOnlyForward(nn.Module):
    """Keep the compiler boundary free of the validated output dataclass API."""

    def __init__(self, model: RecurrentCandidateModel) -> None:
        super().__init__()
        self.model = model

    def forward(self, batch: GroundedCandidateBatch, hidden: Tensor) -> tuple[Tensor, ...]:
        output = self.model(batch, hidden, validate=False)
        liveness_q = output.candidate_liveness_cost_values
        liveness_v = output.liveness_cost_value
        if liveness_q is None:
            liveness_q = output.policy_logits.new_zeros(output.policy_logits.shape)
        if liveness_v is None:
            liveness_v = output.value.new_zeros(output.value.shape)
        # The production policy distribution and value/return arithmetic are
        # float32 boundaries.  Make those boundaries explicit in this test so
        # BF16 measures tensor-core-heavy representation work, not BF16 returns.
        return (
            output.policy_logits.float(),
            output.value.float(),
            output.recurrent_state.float(),
            liveness_q.float(),
            liveness_v.float(),
        )


def _autocast(device: torch.device, precision: str) -> AbstractContextManager[Any]:
    if precision == "bf16":
        return autocast(device_type=device.type, dtype=torch.bfloat16)
    return nullcontext()


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _surrogate_loss(outputs: tuple[Tensor, ...], action_mask: Tensor) -> Tensor:
    logits, value, recurrent, liveness_q, liveness_v = outputs
    masked_logits = torch.where(action_mask, logits, torch.zeros_like(logits))
    # Every production head and recurrent state participates in autograd while
    # the scale remains bounded enough for an implementation benchmark.
    return (
        masked_logits.square().mean()
        + value.square().mean()
        + 1.0e-3 * recurrent.square().mean()
        + 1.0e-2 * liveness_q.square().mean()
        + 1.0e-2 * liveness_v.square().mean()
    )


def _timed_cell(
    module: nn.Module,
    *,
    batch: GroundedCandidateBatch,
    hidden: Tensor,
    device: torch.device,
    precision: str,
    warmup: int,
    iterations: int,
    forward_only: bool,
) -> tuple[float, list[float], tuple[Tensor, ...]]:
    optimizer = None if forward_only else torch.optim.AdamW(module.parameters(), lr=1.0e-4)

    def one() -> tuple[Tensor, ...]:
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with _autocast(device, precision):
            outputs = module(batch, hidden)
            loss = _surrogate_loss(outputs, batch.candidates.action_mask)
        if optimizer is not None:
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
        return cast(tuple[Tensor, ...], outputs)

    outputs: tuple[Tensor, ...] | None = None
    warmup_started = time.perf_counter_ns()
    for _ in range(warmup):
        outputs = one()
    _synchronize(device)
    warmup_ms = (time.perf_counter_ns() - warmup_started) / 1_000_000.0
    samples_ms: list[float] = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        outputs = one()
        _synchronize(device)
        samples_ms.append((time.perf_counter_ns() - started) / 1_000_000.0)
    if outputs is None:  # pragma: no cover - validated positive counts
        raise RuntimeError("benchmark produced no output")
    return warmup_ms, samples_ms, tuple(tensor.detach() for tensor in outputs)


def _numerical_report(reference: tuple[Tensor, ...], actual: tuple[Tensor, ...], action_mask: Tensor) -> dict[str, Any]:
    fields = ("policy_logits", "value", "recurrent_state", "liveness_q", "liveness_v")
    errors: dict[str, Any] = {}
    for name, expected, observed in zip(fields, reference, actual, strict=True):
        # Invalid candidate slots intentionally carry dtype-dependent sentinel
        # values (for example finfo.min).  They are never selectable and must
        # not dominate a precision-equivalence report.
        if name in {"policy_logits", "liveness_q"}:
            expected = expected[action_mask]
            observed = observed[action_mask]
        diff = (expected.float() - observed.float()).abs()
        denominator = expected.float().abs().clamp_min(1.0e-6)
        errors[name] = {
            "max_abs": float(diff.max().item()),
            "mean_abs": float(diff.mean().item()),
            "max_relative": float((diff / denominator).max().item()),
            "finite": bool(torch.isfinite(observed).all().item()),
        }
    reference_choice = torch.where(action_mask, reference[0], torch.full_like(reference[0], -torch.inf)).argmax(-1)
    actual_choice = torch.where(action_mask, actual[0], torch.full_like(actual[0], -torch.inf)).argmax(-1)
    errors["flat_greedy_agreement"] = float((reference_choice == actual_choice).float().mean().item())
    return errors


def main() -> int:
    args = _parse_args()
    for name in ("batch_size", "world_tokens", "candidates", "local_tokens", "warmup", "iterations"):
        _positive(getattr(args, name), label=name.replace("_", "-"))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA/HIP device requested but torch.cuda.is_available() is false")
    if args.precision == "bf16" and device.type != "cuda":
        raise SystemExit("this reviewed T1 BF16 cell is restricted to the CUDA/HIP device")

    if device.type == "cuda" and args.sdpa == "math":
        # v32 deliberately uses the stable math backend.  Compiler/precision
        # cells must not silently confound that with an AOTriton backend change.
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)

    torch.manual_seed(args.seed)
    config = load_training_config(profile=args.profile, config_path=args.config)
    base_model = RecurrentCandidateModel(
        config.model.to_model_config(),
        enable_transaction_heads=config.transaction_learning.enabled,
        enable_liveness_head=config.failure_credit.learning_enabled,
    ).to(device)
    base_model.train(not args.forward_only)
    batch = _synthetic_batch(
        base_model,
        batch_size=args.batch_size,
        world_count=args.world_tokens,
        action_count=args.candidates,
        local_count=args.local_tokens,
        device=device,
    )
    hidden = base_model.initial_state(args.batch_size, device=device)

    # A deterministic eager FP32 forward is the numerical oracle.  It is
    # separate from the timed training-mode module so dropout/RNG do not turn
    # the equivalence check into a random-number-generator benchmark.
    oracle_model = copy.deepcopy(base_model).eval()
    with torch.no_grad():
        oracle_outputs = _TensorOnlyForward(oracle_model)(batch, hidden)
    del oracle_model

    execution: nn.Module = _TensorOnlyForward(base_model)
    compile_started_ns: int | None = None
    if args.use_compile:
        compile_started_ns = time.perf_counter_ns()
        execution = cast(
            nn.Module,
            torch.compile(
                execution,
                fullgraph=args.fullgraph,
                dynamic=None,
                mode=args.compile_mode,
            ),
        )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    try:
        warmup_ms, timings_ms, outputs = _timed_cell(
            execution,
            batch=batch,
            hidden=hidden,
            device=device,
            precision=args.precision,
            warmup=args.warmup,
            iterations=args.iterations,
            forward_only=args.forward_only,
        )
    except BaseException as exc:
        failure = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "device": str(device),
            "precision": args.precision,
            "compiled": args.use_compile,
            "fullgraph": args.fullgraph,
            "compile_mode": args.compile_mode,
            "sdpa": args.sdpa,
        }
        print(json.dumps(failure, indent=2, sort_keys=True))
        return 2

    compile_and_warmup_ms = warmup_ms if compile_started_ns is not None else None
    p90_index = max(0, math.ceil(0.90 * len(timings_ms)) - 1)
    ordered = sorted(timings_ms)
    result = {
        "status": "passed",
        "runtime": {
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "bf16_reported_supported": (
                bool(torch.cuda.is_bf16_supported()) if device.type == "cuda" else None
            ),
        },
        "cell": {
            "precision": args.precision,
            "compiled": args.use_compile,
            "fullgraph": args.fullgraph,
            "compile_mode": args.compile_mode,
            "sdpa": args.sdpa,
            "forward_only": args.forward_only,
            "batch_size": args.batch_size,
            "world_tokens": args.world_tokens,
            "candidates": args.candidates,
            "local_tokens": args.local_tokens,
            "parameter_count": base_model.parameter_count,
        },
        "timing_ms": {
            "median": statistics.median(timings_ms),
            "mean": statistics.fmean(timings_ms),
            "p90": ordered[p90_index],
            "minimum": min(timings_ms),
            "maximum": max(timings_ms),
            # Includes compilation plus configured warm-up; never compare this
            # field with steady-state iteration latency.
            "compile_and_warmup": compile_and_warmup_ms,
        },
        "peak_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        ),
        # A training-mode cell has dropout and an optimiser update between the
        # oracle and the observed output.  Reporting that as compiler/precision
        # error would be actively misleading, so equivalence is a forward-only
        # promotion gate and training cells are throughput/memory evidence only.
        "numerics_comparable": args.forward_only,
        "numerics_vs_eager_fp32_eval": (
            _numerical_report(oracle_outputs, outputs, batch.candidates.action_mask)
            if args.forward_only
            else None
        ),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
