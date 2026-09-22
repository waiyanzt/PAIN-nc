"""Mandatory resource telemetry shared by every PAIN experiment."""
from __future__ import annotations

import io
import os
import platform
import sys
import threading
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn


CUDA_MEMORY_KEYS = (
    "gpu_allocated_bytes",
    "gpu_reserved_bytes",
    "gpu_peak_allocated_bytes",
    "gpu_peak_reserved_bytes",
)


def model_memory_bytes(model: nn.Module) -> dict[str, int]:
    parameter_bytes = sum(
        value.numel() * value.element_size() for value in model.parameters()
    )
    buffer_bytes = sum(
        value.numel() * value.element_size() for value in model.buffers()
    )
    return {
        "parameter_bytes": int(parameter_bytes),
        "buffer_bytes": int(buffer_bytes),
        "static_model_bytes": int(parameter_bytes + buffer_bytes),
    }


def serialized_torch_bytes(payload: Any) -> int:
    """Return the exact byte size of a torch-serialized in-memory payload."""
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    return int(buffer.tell())


def _windows_memory_counters():
    import ctypes
    from ctypes import wintypes

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    handle = ctypes.windll.kernel32.GetCurrentProcess()
    ok = ctypes.windll.psapi.GetProcessMemoryInfo(
        handle, ctypes.byref(counters), counters.cb
    )
    if not ok:
        raise OSError("GetProcessMemoryInfo failed")
    return counters


def current_rss_bytes() -> int:
    """Return the process's current resident memory."""
    if sys.platform == "win32":
        return int(_windows_memory_counters().WorkingSetSize)
    statm = Path("/proc/self/statm")
    if statm.is_file():
        resident_pages = int(statm.read_text(encoding="ascii").split()[1])
        return resident_pages * int(os.sysconf("SC_PAGE_SIZE"))
    # Portable fallback where a resettable current-RSS interface is absent.
    return process_peak_rss_bytes()


def process_peak_rss_bytes() -> int:
    """Return the operating system's process-lifetime peak RSS."""
    if sys.platform == "win32":
        return int(_windows_memory_counters().PeakWorkingSetSize)
    import resource

    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


class PeakRSSMonitor:
    """Sample current RSS to obtain a resettable peak for one experiment run."""

    def __init__(self, interval_seconds: float = 0.05) -> None:
        self.interval_seconds = float(interval_seconds)
        self.peak_bytes = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> "PeakRSSMonitor":
        if self._thread is not None:
            raise RuntimeError("PeakRSSMonitor is already running")
        self.peak_bytes = current_rss_bytes()
        self._stop.clear()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def _sample(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self.peak_bytes = max(self.peak_bytes, current_rss_bytes())

    def stop(self) -> int:
        self.peak_bytes = max(self.peak_bytes, current_rss_bytes())
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_seconds * 4))
            self._thread = None
        return int(self.peak_bytes)


def reset_cuda_peak(device: torch.device) -> None:
    if device.type != "cuda":
        return
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)


def cuda_memory_stats(device: torch.device) -> dict[str, int]:
    if device.type != "cuda":
        return {key: 0 for key in CUDA_MEMORY_KEYS}
    torch.cuda.synchronize(device)
    return {
        "gpu_allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "gpu_reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "gpu_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def merge_cuda_memory_stats(
    previous: Mapping[str, int] | None,
    current: Mapping[str, int],
) -> dict[str, int]:
    previous = previous or {}
    return {
        "gpu_allocated_bytes": int(current.get("gpu_allocated_bytes", 0)),
        "gpu_reserved_bytes": int(current.get("gpu_reserved_bytes", 0)),
        "gpu_peak_allocated_bytes": max(
            int(previous.get("gpu_peak_allocated_bytes", 0)),
            int(current.get("gpu_peak_allocated_bytes", 0)),
        ),
        "gpu_peak_reserved_bytes": max(
            int(previous.get("gpu_peak_reserved_bytes", 0)),
            int(current.get("gpu_peak_reserved_bytes", 0)),
        ),
    }


def artifact_sizes(paths: Mapping[str, str | Path]) -> dict[str, int]:
    sizes = {}
    for name, raw_path in paths.items():
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        sizes[f"{name}_bytes"] = int(path.stat().st_size)
    return sizes


def environment_metadata(device: torch.device) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_runtime_version": torch.version.cuda,
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "pid": os.getpid(),
    }
    if device.type == "cuda":
        metadata.update(
            {
                "device_name": torch.cuda.get_device_name(device),
                "device_capability": list(torch.cuda.get_device_capability(device)),
            }
        )
    else:
        metadata["device_name"] = platform.processor() or "CPU"
        metadata["device_capability"] = None
    return metadata


def validate_resource_metrics(resources: Mapping[str, Any]) -> None:
    """Fail closed when an experiment omits mandatory memory telemetry."""
    scalar_keys = {
        "parameter_bytes",
        "buffer_bytes",
        "static_model_bytes",
        "checkpoint_bytes",
        "process_peak_rss_bytes",
    }
    missing = sorted(scalar_keys - resources.keys())
    for phase in ("training_gpu", "inference_gpu"):
        if phase not in resources:
            missing.append(phase)
            continue
        missing.extend(
            f"{phase}.{key}"
            for key in CUDA_MEMORY_KEYS
            if key not in resources[phase]
        )
    for section in ("artifacts", "environment"):
        if section not in resources:
            missing.append(section)
    if missing:
        raise ValueError(
            "Mandatory resource telemetry is incomplete: " + ", ".join(missing)
        )
    for key in scalar_keys:
        if int(resources[key]) < 0:
            raise ValueError(f"Resource metric {key} must be non-negative")
