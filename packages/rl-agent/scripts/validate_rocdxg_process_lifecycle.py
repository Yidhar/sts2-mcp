#!/usr/bin/env python3
"""Stress fresh ROCDXG process creation, allocation, release, and exit.

The parent deliberately never imports torch.  Each cycle starts a fresh child
under the exact environment supplied by the caller, creates a HIP context,
commits the requested amount of device memory, exercises autograd plus AdamW,
releases all live tensors and the caching allocator, synchronizes, and exits.
This catches process-lifetime failures that an in-process tensor loop cannot.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "sts2-rocdxg-process-lifecycle-stress-v1"
EXECUTION_CONTRACT_VERSION = "sts2-rocdxg-hermetic-execution-contract-v1"
HSA_KEY = "HSA_ENABLE_DXG_DETECTION"
HSA_VALUE = "1"
_MIB = 1024 * 1024
_NATIVE_FAILURE_MARKERS = (
    "hsa_status_error",
    "hsa exception",
    "blockallocator",
    "sharedsignalpool",
    "segmentation fault",
    "core dumped",
    "double free",
    "terminate called",
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _allocation_schedule(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("allocation schedule must contain integers") from exc
    if not parsed or any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("allocation schedule must contain positive MiB values")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycles", type=_positive_int, default=20)
    parser.add_argument(
        "--allocation-mib",
        type=_allocation_schedule,
        default=(256, 1024, 4096, 8192),
        help="comma-separated per-child allocation schedule",
    )
    parser.add_argument("--child-timeout-seconds", type=_positive_int, default=180)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--child-allocation-mib", type=_positive_int, help=argparse.SUPPRESS)
    parser.add_argument("--child-index", type=int, help=argparse.SUPPRESS)
    return parser


def _require_contract() -> None:
    actual = os.environ.get(HSA_KEY)
    if actual != HSA_VALUE:
        raise RuntimeError(
            f"ROCDXG execution contract violated: {HSA_KEY} must equal {HSA_VALUE!r}, got {actual!r}"
        )


def _child(*, allocation_mib: int, child_index: int) -> dict[str, Any]:
    _require_contract()
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("ROCDXG child cannot see a CUDA/HIP device")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    _, total_bytes = torch.cuda.mem_get_info(device)
    requested_bytes = allocation_mib * _MIB
    # Keep the stress bounded to 50% of physical VRAM as reviewed by the user.
    if requested_bytes > total_bytes // 2:
        raise RuntimeError(
            f"requested allocation exceeds 50% VRAM: requested={requested_bytes} total={total_bytes}"
        )

    def optimizer_workload(iterations: int) -> list[float]:
        parameter = torch.nn.Parameter(torch.randn(1024, 1024, device=device))
        operand = torch.randn(1024, 64, device=device)
        optimizer = torch.optim.AdamW((parameter,), lr=1.0e-4)
        observed: list[float] = []
        for _ in range(iterations):
            optimizer.zero_grad(set_to_none=True)
            loss = (parameter @ operand).square().mean()
            loss.backward()
            optimizer.step()
            observed.append(float(loss.detach().item()))
        return observed

    # rocBLAS may retain a process-lifetime workspace that PyTorch reports as
    # allocated memory. Establish that runtime baseline before judging whether
    # the test's own large allocation was released.
    optimizer_workload(1)
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    baseline_allocated = int(torch.cuda.memory_allocated(device))
    baseline_reserved = int(torch.cuda.memory_reserved(device))
    free_before, _ = torch.cuda.mem_get_info(device)

    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    block = torch.empty(requested_bytes, dtype=torch.uint8, device=device)
    block.fill_(child_index % 251)
    checksum = int(block[0].item()) + int(block[-1].item())
    # Prove that allocation pressure did not leave the context unable to
    # execute forward/backward/update work.
    losses = optimizer_workload(3)
    torch.cuda.synchronize(device)
    peak_allocated = int(torch.cuda.max_memory_allocated(device))

    del block
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    allocated_after = int(torch.cuda.memory_allocated(device))
    reserved_after = int(torch.cuda.memory_reserved(device))
    free_after, total_after = torch.cuda.mem_get_info(device)
    if total_after != total_bytes:
        raise RuntimeError("device total memory changed within one child")
    tolerance_bytes = 8 * _MIB
    if (
        allocated_after > baseline_allocated + tolerance_bytes
        or reserved_after > baseline_reserved + tolerance_bytes
    ):
        raise RuntimeError(
            "child did not return to its ROCm runtime allocation baseline: "
            f"baseline_allocated={baseline_allocated} allocated={allocated_after} "
            f"baseline_reserved={baseline_reserved} reserved={reserved_after}"
        )
    if not all(value == value and abs(value) != float("inf") for value in losses):
        raise RuntimeError("optimizer workload produced a non-finite loss")

    return {
        "schema_version": SCHEMA_VERSION,
        "execution_contract_version": EXECUTION_CONTRACT_VERSION,
        "status": "passed",
        "child_index": child_index,
        "pid": os.getpid(),
        "allocation_mib": allocation_mib,
        "requested_bytes": requested_bytes,
        "checksum": checksum,
        "losses": losses,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_allocated_bytes": peak_allocated,
        "baseline_allocated_bytes": baseline_allocated,
        "baseline_reserved_bytes": baseline_reserved,
        "allocated_after_release_bytes": allocated_after,
        "reserved_after_release_bytes": reserved_after,
        "free_before_bytes": int(free_before),
        "free_after_release_bytes": int(free_after),
        "total_device_bytes": int(total_bytes),
        "runtime": {
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "device_name": torch.cuda.get_device_name(device),
            "device_count": torch.cuda.device_count(),
        },
    }


def _parse_child_stdout(stdout: str) -> dict[str, Any]:
    lines = [line for line in stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("child emitted no JSON result")
    payload = json.loads(lines[-1])
    if not isinstance(payload, dict):
        raise RuntimeError("child result is not a JSON object")
    return payload


def _parent(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    _require_contract()
    environment = dict(os.environ)
    started = time.time()
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for index in range(args.cycles):
        allocation_mib = args.allocation_mib[index % len(args.allocation_mib)]
        command = (
            sys.executable,
            str(Path(__file__).resolve()),
            "--child",
            "--child-index",
            str(index),
            "--child-allocation-mib",
            str(allocation_mib),
        )
        cycle_started = time.perf_counter()
        try:
            completed = subprocess.run(
                command,
                env=environment,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=args.child_timeout_seconds,
                check=False,
            )
            lowered_stderr = completed.stderr.lower()
            native_markers = [
                marker for marker in _NATIVE_FAILURE_MARKERS if marker in lowered_stderr
            ]
            payload = _parse_child_stdout(completed.stdout)
            record = {
                "child_index": index,
                "allocation_mib": allocation_mib,
                "returncode": completed.returncode,
                "wall_seconds": time.perf_counter() - cycle_started,
                "native_failure_markers": native_markers,
                "stderr": completed.stderr[-4096:],
                "payload": payload,
            }
            records.append(record)
            if (
                completed.returncode != 0
                or native_markers
                or not isinstance(payload, dict)
                or payload.get("status") != "passed"
            ):
                failures.append(record)
        except (subprocess.TimeoutExpired, json.JSONDecodeError, RuntimeError) as exc:
            record = {
                "child_index": index,
                "allocation_mib": allocation_mib,
                "returncode": None,
                "wall_seconds": time.perf_counter() - cycle_started,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            records.append(record)
            failures.append(record)

    report = {
        "schema_version": SCHEMA_VERSION,
        "execution_contract_version": EXECUTION_CONTRACT_VERSION,
        "status": "passed" if not failures else "failed",
        "started_unix_s": started,
        "finished_unix_s": time.time(),
        "cycles_requested": args.cycles,
        "cycles_passed": args.cycles - len(failures),
        "cycles_failed": len(failures),
        "allocation_schedule_mib": list(args.allocation_mib),
        "maximum_allocation_mib": max(args.allocation_mib),
        "records": records,
    }
    return (0 if not failures else 2), report


def main() -> int:
    args = _parser().parse_args()
    if args.child:
        if args.child_allocation_mib is None or args.child_index is None:
            raise SystemExit("internal child arguments are incomplete")
        try:
            payload = _child(
                allocation_mib=args.child_allocation_mib,
                child_index=args.child_index,
            )
        except BaseException as exc:
            payload = {
                "schema_version": SCHEMA_VERSION,
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            print(json.dumps(payload, sort_keys=True))
            return 2
        print(json.dumps(payload, sort_keys=True))
        return 0

    try:
        returncode, report = _parent(args)
    except BaseException as exc:
        report = {
            "schema_version": SCHEMA_VERSION,
            "execution_contract_version": EXECUTION_CONTRACT_VERSION,
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        returncode = 2
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
