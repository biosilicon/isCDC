"""Persistent NVML telemetry without spawning nvidia-smi or blocking the scheduler."""

from __future__ import annotations

import ctypes as ct
import threading
import time


class Memory(ct.Structure):
    _fields_ = [(name, ct.c_ulonglong) for name in ("total", "free", "used")]


class Utilization(ct.Structure):
    _fields_ = [(name, ct.c_uint) for name in ("gpu", "memory")]


class ProcessInfo(ct.Structure):
    # nvmlProcessInfo_v2_t, deliberately paired with the versioned v2 entry point.
    # https://github.com/NVIDIA/go-nvml/blob/main/pkg/nvml/nvml.h
    _fields_ = [
        ("pid", ct.c_uint),
        ("usedGpuMemory", ct.c_ulonglong),
        ("gpuInstanceId", ct.c_uint),
        ("computeInstanceId", ct.c_uint),
    ]


class NVMLReader:
    def __init__(self, uuid):
        self.library = ct.CDLL("libnvidia-ml.so.1")
        signatures = {
            "nvmlInit_v2": [],
            "nvmlShutdown": [],
            "nvmlDeviceGetHandleByUUID": [ct.c_char_p, ct.POINTER(ct.c_void_p)],
            "nvmlDeviceGetMemoryInfo": [ct.c_void_p, ct.POINTER(Memory)],
            "nvmlDeviceGetUtilizationRates": [ct.c_void_p, ct.POINTER(Utilization)],
            "nvmlDeviceGetComputeRunningProcesses_v2": [
                ct.c_void_p,
                ct.POINTER(ct.c_uint),
                ct.POINTER(ProcessInfo),
            ],
        }
        for name, args in signatures.items():
            function = getattr(self.library, name)
            function.argtypes, function.restype = args, ct.c_int
        self.check(self.library.nvmlInit_v2())
        self.handle = ct.c_void_p()
        try:
            self.check(
                self.library.nvmlDeviceGetHandleByUUID(
                    ("GPU-" + uuid.removeprefix("GPU-")).encode(), ct.byref(self.handle)
                )
            )
        except Exception:
            self.close()
            raise

    @staticmethod
    def check(code):
        if code:
            raise RuntimeError(f"NVML query failed with code {code}")

    def close(self):
        self.library.nvmlShutdown()

    def read(self):
        memory, utilization = Memory(), Utilization()
        self.check(self.library.nvmlDeviceGetMemoryInfo(self.handle, ct.byref(memory)))
        self.check(self.library.nvmlDeviceGetUtilizationRates(self.handle, ct.byref(utilization)))
        capacity = 64
        for _ in range(4):
            count = ct.c_uint(capacity)
            processes = (ProcessInfo * capacity)()
            code = self.library.nvmlDeviceGetComputeRunningProcesses_v2(
                self.handle, ct.byref(count), processes
            )
            if code == 7:  # NVML_ERROR_INSUFFICIENT_SIZE: process count grew while sampling.
                capacity = max(capacity * 2, count.value)
                continue
            self.check(code)
            usage = {}
            for process in processes[: count.value]:
                if process.usedGpuMemory == ct.c_ulonglong(-1).value:
                    raise RuntimeError("NVML process memory is unavailable")
                usage[process.pid] = process.usedGpuMemory
            if not 0 <= memory.free <= memory.total or not memory.total:
                raise RuntimeError("NVML returned invalid free memory")
            return {
                "free_bytes": memory.free,
                "utilization_percent": utilization.gpu,
                "process_bytes": usage,
            }
        raise RuntimeError("NVML process list kept growing during sampling")


class GPUMonitor:
    """One daemon sampler; stalled/error samples never authorize new resource leases."""

    def __init__(self, uuid):
        self.uuid = uuid
        self.lock = threading.Lock()
        self.stopped = threading.Event()
        self.latest = None
        self.sampled_at = None
        self.error = None
        self.thread = threading.Thread(target=self._run, name="spatialglue-nvml", daemon=True)
        self.thread.start()

    def _run(self):
        reader = None
        try:
            while not self.stopped.is_set():
                try:
                    if reader is None:
                        reader = NVMLReader(self.uuid)
                    started = time.monotonic()
                    value = reader.read()
                    with self.lock:
                        self.latest = value
                        self.sampled_at = started
                        self.error = None
                except (OSError, RuntimeError, AttributeError) as exc:
                    with self.lock:
                        self.error = f"{type(exc).__name__}: {exc}"
                self.stopped.wait(1)
        finally:
            if reader is not None:
                reader.close()

    def snapshot(self):
        with self.lock:
            age = None if self.sampled_at is None else time.monotonic() - self.sampled_at
            return {
                **(
                    self.latest
                    or {"free_bytes": None, "utilization_percent": None, "process_bytes": {}}
                ),
                "backend": "nvml",
                "fresh": age is not None and age <= 5 and self.error is None,
                "sample_age_seconds": None if age is None else round(age, 2),
                "error": self.error,
            }

    def close(self):
        self.stopped.set()
        self.thread.join(timeout=0.1)
