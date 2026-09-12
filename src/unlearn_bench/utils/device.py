from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any

import torch

DEVICE_CHOICES = ("auto", "cuda", "mps", "cpu")
PRECISION_CHOICES = ("auto", "fp32", "fp16", "bf16")


class DeviceUnavailableError(RuntimeError):
    """Raised when an explicitly requested execution device is unavailable."""


class PrecisionUnavailableError(RuntimeError):
    """Raised when a requested precision is not supported by the selected device."""


@dataclass(frozen=True)
class DeviceContext:
    requested_device: str
    device: str
    backend: str
    requested_precision: str
    precision: str
    dtype: torch.dtype
    mps_fallback_enabled: bool

    def manifest(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("dtype")
        return value


def _mps_available() -> bool:
    backend = getattr(torch.backends, "mps", None)
    return bool(backend is not None and backend.is_available())


def available_devices() -> dict[str, bool]:
    return {"cuda": torch.cuda.is_available(), "mps": _mps_available(), "cpu": True}


def _resolve_device(requested: str) -> str:
    if requested not in DEVICE_CHOICES:
        raise ValueError(f"device must be one of {DEVICE_CHOICES}, got {requested!r}")
    available = available_devices()
    if requested == "auto":
        for candidate in ("cuda", "mps", "cpu"):
            if available[candidate]:
                return candidate
        raise DeviceUnavailableError("No supported PyTorch execution device is available")
    if not available[requested]:
        raise DeviceUnavailableError(
            f"Requested device {requested!r} is unavailable; detected availability: {available}"
        )
    return requested


def _probe_dtype(device: str, dtype: torch.dtype) -> bool:
    """Probe basic forward/backward support instead of inferring it from version strings."""
    try:
        left = torch.ones((2, 2), device=device, dtype=dtype, requires_grad=True)
        right = torch.ones((2, 2), device=device, dtype=dtype)
        (left @ right).sum().backward()
        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()
        return left.grad is not None and bool(torch.isfinite(left.grad).all())
    except (RuntimeError, TypeError):
        return False


def _resolve_precision(device: str, requested: str) -> tuple[str, torch.dtype]:
    if requested not in PRECISION_CHOICES:
        raise ValueError(f"precision must be one of {PRECISION_CHOICES}, got {requested!r}")
    if requested == "auto":
        cuda_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
        if cuda_bf16 and _probe_dtype(device, torch.bfloat16):
            return "bf16", torch.bfloat16
        return "fp32", torch.float32
    precision_to_dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    dtype = precision_to_dtype[requested]
    if device == "cpu" and requested == "fp16":
        raise PrecisionUnavailableError("FP16 execution is not supported by the CPU policy")
    if device == "cuda" and requested == "bf16" and not torch.cuda.is_bf16_supported():
        raise PrecisionUnavailableError("BF16 is not supported by the selected CUDA device")
    if not _probe_dtype(device, dtype):
        raise PrecisionUnavailableError(
            f"Precision {requested!r} failed the execution probe on device {device!r}"
        )
    return requested, dtype


def select_device(
    requested_device: str = "auto",
    requested_precision: str = "auto",
    *,
    allow_mps_fallback: bool = False,
) -> DeviceContext:
    device = _resolve_device(requested_device)
    if device == "cuda":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    fallback_enabled = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1"
    if device == "mps" and fallback_enabled and not allow_mps_fallback:
        raise RuntimeError(
            "PYTORCH_ENABLE_MPS_FALLBACK=1 is set, but allow_mps_fallback is false; "
            "refusing an unrecorded CPU fallback"
        )
    precision, dtype = _resolve_precision(device, requested_precision)
    return DeviceContext(
        requested_device=requested_device,
        device=device,
        backend=device,
        requested_precision=requested_precision,
        precision=precision,
        dtype=dtype,
        mps_fallback_enabled=device == "mps" and fallback_enabled,
    )


def synchronize(context: DeviceContext) -> None:
    if context.device == "cuda":
        torch.cuda.synchronize()
    elif context.device == "mps":
        torch.mps.synchronize()


def reset_peak_memory(context: DeviceContext) -> None:
    if context.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    elif context.device == "mps" and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


def memory_metrics(context: DeviceContext) -> dict[str, int]:
    if context.device == "cuda":
        return {
            "peak_device_memory_bytes": torch.cuda.max_memory_allocated(),
            "device_memory_allocated_bytes": torch.cuda.memory_allocated(),
        }
    if context.device == "mps":
        current = int(torch.mps.current_allocated_memory())
        driver = int(torch.mps.driver_allocated_memory())
        return {
            # PyTorch exposes current rather than peak MPS allocation.
            "peak_device_memory_bytes": current,
            "device_memory_allocated_bytes": current,
            "mps_driver_allocated_bytes": driver,
        }
    return {"peak_device_memory_bytes": 0, "device_memory_allocated_bytes": 0}
