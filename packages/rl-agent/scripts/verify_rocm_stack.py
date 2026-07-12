"""Verify the exact ROCm PyTorch artifact and a visible accelerator."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-distribution-version", required=True)
    parser.add_argument("--expected-runtime-version", required=True)
    args = parser.parse_args(argv)

    import torch

    distribution_version = importlib.metadata.version("torch")
    runtime_version = str(torch.__version__)
    accelerator_available = bool(torch.cuda.is_available())
    payload: dict[str, object] = {
        "distribution_version": distribution_version,
        "torch_version": runtime_version,
        "cuda_is_available": accelerator_available,
        "device_count": int(torch.cuda.device_count()) if accelerator_available else 0,
    }
    if accelerator_available:
        payload["device_name"] = torch.cuda.get_device_name(0)
        probe = torch.arange(16, dtype=torch.float32, device="cuda").reshape(4, 4)
        probe_result = float(((probe @ probe.T).sum()).cpu())
        torch.cuda.synchronize()
        payload["gpu_probe_result"] = probe_result
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))

    if distribution_version != args.expected_distribution_version:
        raise SystemExit(
            "unexpected torch distribution artifact: " + distribution_version
        )
    if runtime_version != args.expected_runtime_version:
        raise SystemExit("unexpected torch runtime artifact: " + runtime_version)
    if not accelerator_available:
        raise SystemExit("ROCm accelerator is not visible to PyTorch")
    if payload.get("gpu_probe_result") != 3680.0:
        raise SystemExit("ROCm accelerator computation probe returned an unexpected result")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
