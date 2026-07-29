"""Explicit process-wide SDPA execution policy for ROCm training.

PyTorch exposes scaled-dot-product-attention backend switches as process-wide
CUDA backend state.  ROCm uses the same ``torch.cuda`` namespace.  We only
touch those switches when an explicitly configured policy is active *and* at
least one training device is a ROCm CUDA device; CPU-only composition remains
side-effect free.

The policy is execution provenance rather than immutable learning lineage.
Changing kernels does not change the model, optimizer, replay, RNG, counters,
or task semantics restored by exact resume.  It is nevertheless recorded and
validated so a backend hardening transition is never silent.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import torch

RocmSdpaBackend = Literal["auto", "math"]

ROCM_SDPA_EXECUTION_ABI = "sts2-rocm-sdpa-execution-v1"
ROCM_SDPA_TRANSITION_ABI = "sts2-rocm-sdpa-transition-v1"
NATIVE_ROCR_ABORT_MITIGATION_REASON = "native ROCr abort mitigation"

_FLAG_NAMES = ("flash", "memory_efficient", "math", "cudnn")


def _hip_version() -> str | None:
    raw = getattr(torch.version, "hip", None)
    return str(raw) if raw else None


def _read_flags() -> dict[str, bool]:
    return {
        "flash": bool(
            torch.backends.cuda.flash_sdp_enabled()  # type: ignore[no-untyped-call]
        ),
        "memory_efficient": bool(
            torch.backends.cuda.mem_efficient_sdp_enabled()  # type: ignore[no-untyped-call]
        ),
        "math": bool(
            torch.backends.cuda.math_sdp_enabled()  # type: ignore[no-untyped-call]
        ),
        "cudnn": bool(
            torch.backends.cuda.cudnn_sdp_enabled()  # type: ignore[no-untyped-call]
        ),
    }


def _validate_flags(value: object, *, label: str) -> dict[str, bool] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != set(_FLAG_NAMES):
        raise ValueError(f"{label} must contain the complete SDPA flag set")
    result: dict[str, bool] = {}
    for name in _FLAG_NAMES:
        enabled = value[name]
        if not isinstance(enabled, bool):
            raise TypeError(f"{label}.{name} must be a boolean")
        result[name] = enabled
    return result


@dataclass(frozen=True, slots=True)
class SdpaExecutionState:
    """Observed result of applying one explicit execution policy."""

    requested_policy: RocmSdpaBackend
    devices: tuple[str, ...]
    hip_version: str | None
    applicability: Literal["rocm_cuda", "not_rocm_cuda"]
    applied: bool
    previous_flags: dict[str, bool] | None
    current_flags: dict[str, bool] | None
    effective_backend: Literal[
        "framework_default",
        "math_only",
        "not_applicable",
    ]

    def to_mapping(self) -> dict[str, Any]:
        return {
            "version": ROCM_SDPA_EXECUTION_ABI,
            "requested_policy": self.requested_policy,
            "devices": list(self.devices),
            "hip_version": self.hip_version,
            "applicability": self.applicability,
            "applied": self.applied,
            "previous_flags": (
                dict(self.previous_flags)
                if self.previous_flags is not None
                else None
            ),
            "current_flags": (
                dict(self.current_flags)
                if self.current_flags is not None
                else None
            ),
            "effective_backend": self.effective_backend,
        }


def validate_sdpa_execution_mapping(payload: object) -> dict[str, Any]:
    """Validate a checkpoint's optional, explicitly versioned SDPA record."""

    if not isinstance(payload, Mapping):
        raise TypeError("checkpoint SDPA execution state must be an object")
    expected = {
        "version",
        "requested_policy",
        "devices",
        "hip_version",
        "applicability",
        "applied",
        "previous_flags",
        "current_flags",
        "effective_backend",
    }
    if set(payload) != expected:
        raise ValueError("checkpoint SDPA execution state keys mismatch")
    if payload.get("version") != ROCM_SDPA_EXECUTION_ABI:
        raise ValueError("checkpoint has an unsupported SDPA execution ABI")
    policy = payload.get("requested_policy")
    if policy not in ("auto", "math"):
        raise ValueError("checkpoint has an unsupported ROCm SDPA policy")
    devices = payload.get("devices")
    if not isinstance(devices, list) or not devices or not all(
        isinstance(item, str) and item for item in devices
    ):
        raise TypeError("checkpoint SDPA devices must be a non-empty string list")
    hip_version = payload.get("hip_version")
    if hip_version is not None and (
        not isinstance(hip_version, str) or not hip_version
    ):
        raise TypeError("checkpoint SDPA hip_version must be text or null")
    applicability = payload.get("applicability")
    if applicability not in ("rocm_cuda", "not_rocm_cuda"):
        raise ValueError("checkpoint SDPA applicability is unsupported")
    applied = payload.get("applied")
    if not isinstance(applied, bool):
        raise TypeError("checkpoint SDPA applied marker must be a boolean")
    previous_flags = _validate_flags(
        payload.get("previous_flags"),
        label="checkpoint SDPA previous_flags",
    )
    current_flags = _validate_flags(
        payload.get("current_flags"),
        label="checkpoint SDPA current_flags",
    )
    effective = payload.get("effective_backend")
    if effective not in ("framework_default", "math_only", "not_applicable"):
        raise ValueError("checkpoint SDPA effective backend is unsupported")

    if applicability == "not_rocm_cuda":
        if applied or previous_flags is not None or current_flags is not None:
            raise ValueError("non-ROCm SDPA provenance must not claim CUDA mutation")
        if effective != "not_applicable":
            raise ValueError("non-ROCm SDPA provenance must be not_applicable")
    else:
        if hip_version is None or previous_flags is None or current_flags is None:
            raise ValueError("ROCm SDPA provenance must record HIP and backend flags")
        if policy == "auto":
            if applied or effective != "framework_default":
                raise ValueError("automatic ROCm SDPA provenance is inconsistent")
        else:
            expected_math = {
                "flash": False,
                "memory_efficient": False,
                "math": True,
                "cudnn": False,
            }
            if not applied or current_flags != expected_math or effective != "math_only":
                raise ValueError("math-only ROCm SDPA provenance is inconsistent")

    return {
        "version": ROCM_SDPA_EXECUTION_ABI,
        "requested_policy": policy,
        "devices": list(devices),
        "hip_version": hip_version,
        "applicability": applicability,
        "applied": applied,
        "previous_flags": previous_flags,
        "current_flags": current_flags,
        "effective_backend": effective,
    }


def configure_rocm_sdpa_backend(
    policy: RocmSdpaBackend,
    *,
    devices: Sequence[torch.device],
) -> SdpaExecutionState:
    """Apply and verify the requested backend policy exactly once at startup.

    ``math`` is deliberately ROCm-specific.  A CPU-only or non-ROCm process
    records ``not_applicable`` without calling any process-global setter.
    ``auto`` records the framework flags but leaves them untouched.
    """

    if policy not in ("auto", "math"):
        raise ValueError(f"unsupported ROCm SDPA backend policy: {policy!r}")
    if not devices:
        raise ValueError("SDPA backend configuration requires at least one device")
    normalized_devices = tuple(str(device) for device in devices)
    hip_version = _hip_version()
    rocm_cuda = hip_version is not None and any(
        device.type == "cuda" for device in devices
    )
    if not rocm_cuda:
        return SdpaExecutionState(
            requested_policy=policy,
            devices=normalized_devices,
            hip_version=hip_version,
            applicability="not_rocm_cuda",
            applied=False,
            previous_flags=None,
            current_flags=None,
            effective_backend="not_applicable",
        )

    previous_flags = _read_flags()
    if policy == "auto":
        return SdpaExecutionState(
            requested_policy=policy,
            devices=normalized_devices,
            hip_version=hip_version,
            applicability="rocm_cuda",
            applied=False,
            previous_flags=previous_flags,
            current_flags=dict(previous_flags),
            effective_backend="framework_default",
        )

    # Keep one backend enabled throughout the transition.  No model forward or
    # actor thread exists yet when the composition root invokes this function.
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    current_flags = _read_flags()
    expected = {
        "flash": False,
        "memory_efficient": False,
        "math": True,
        "cudnn": False,
    }
    if current_flags != expected:
        raise RuntimeError(
            "failed to establish the requested ROCm math-only SDPA backend: "
            f"observed {current_flags!r}"
        )
    return SdpaExecutionState(
        requested_policy=policy,
        devices=normalized_devices,
        hip_version=hip_version,
        applicability="rocm_cuda",
        applied=True,
        previous_flags=previous_flags,
        current_flags=current_flags,
        effective_backend="math_only",
    )


def sdpa_transition_provenance(
    *,
    previous: object | None,
    current: SdpaExecutionState,
    checkpoint_load_mode: str,
    parent_checkpoint_present: bool,
) -> dict[str, Any]:
    """Describe an execution-backend transition without inventing history."""

    current_mapping = current.to_mapping()
    if previous is None and parent_checkpoint_present:
        previous_mapping: dict[str, Any] = {
            "recording_status": "unrecorded/default",
            "requested_policy": "unrecorded/default",
            "effective_backend": "unrecorded/default",
        }
        previous_label = "unrecorded/default"
        changed: bool | None = None
        reason = (
            NATIVE_ROCR_ABORT_MITIGATION_REASON
            if current.effective_backend == "math_only"
            else "checkpoint execution backend was not recorded"
        )
    elif previous is None:
        previous_mapping = {
            "recording_status": "no_parent_checkpoint",
            "requested_policy": None,
            "effective_backend": None,
        }
        previous_label = "no_parent_checkpoint"
        changed = False
        reason = "configured execution policy at process start"
    else:
        previous_mapping = validate_sdpa_execution_mapping(previous)
        previous_label = str(previous_mapping["effective_backend"])
        changed = (
            previous_mapping["requested_policy"]
            != current_mapping["requested_policy"]
            or previous_mapping["current_flags"]
            != current_mapping["current_flags"]
        )
        reason = (
            NATIVE_ROCR_ABORT_MITIGATION_REASON
            if changed and current.effective_backend == "math_only"
            else "configured execution policy transition"
            if changed
            else "unchanged configured execution policy"
        )
    return {
        "version": ROCM_SDPA_TRANSITION_ABI,
        "checkpoint_load_mode": checkpoint_load_mode,
        "previous": previous_mapping,
        "previous_label": previous_label,
        "current": current_mapping,
        "current_label": current.effective_backend,
        "changed": changed,
        "reason": reason,
    }


def validate_sdpa_transition_mapping(
    payload: object,
    *,
    expected_current: Mapping[str, Any],
) -> None:
    if not isinstance(payload, Mapping):
        raise TypeError("checkpoint SDPA transition provenance must be an object")
    expected_keys = {
        "version",
        "checkpoint_load_mode",
        "previous",
        "previous_label",
        "current",
        "current_label",
        "changed",
        "reason",
    }
    if set(payload) != expected_keys:
        raise ValueError("checkpoint SDPA transition provenance keys mismatch")
    if payload.get("version") != ROCM_SDPA_TRANSITION_ABI:
        raise ValueError("checkpoint has an unsupported SDPA transition ABI")
    load_mode = payload.get("checkpoint_load_mode")
    if load_mode not in ("fresh", "exact_resume", "model_initialization"):
        raise ValueError("checkpoint SDPA transition has an unsupported load mode")
    previous = payload.get("previous")
    if not isinstance(previous, Mapping):
        raise TypeError("checkpoint SDPA transition previous state must be an object")
    current = validate_sdpa_execution_mapping(payload.get("current"))
    if current != dict(expected_current):
        raise ValueError("checkpoint SDPA transition current state differs")
    for name in ("previous_label", "current_label", "reason"):
        value = payload.get(name)
        if not isinstance(value, str) or not value:
            raise TypeError(f"checkpoint SDPA transition {name} must be text")
    changed = payload.get("changed")
    if changed is not None and not isinstance(changed, bool):
        raise TypeError("checkpoint SDPA transition changed must be boolean or null")


__all__ = [
    "NATIVE_ROCR_ABORT_MITIGATION_REASON",
    "ROCM_SDPA_EXECUTION_ABI",
    "ROCM_SDPA_TRANSITION_ABI",
    "RocmSdpaBackend",
    "SdpaExecutionState",
    "configure_rocm_sdpa_backend",
    "sdpa_transition_provenance",
    "validate_sdpa_execution_mapping",
    "validate_sdpa_transition_mapping",
]
