"""Phase telemetry and inherited-pipe resource leases for GPU workers."""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager

_CHANNEL = None
TIMINGS = {}
CUDA_PEAKS = {"allocated": 0, "reserved": 0}


def set_thread_affinity(cpus):
    """Native BLAS/torch threads may predate a phase: move them with the lease."""
    import psutil

    os.sched_setaffinity(0, cpus)
    for thread in psutil.Process().threads():
        try:
            os.sched_setaffinity(thread.id, cpus)
        except ProcessLookupError:
            pass


def _event(event):
    path = os.environ.get("ISCDC_GLUE_EVENTS")
    if path:
        with open(path, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(event) + "\n")


@contextmanager
def phase(name, params):
    import psutil
    import torch
    from threadpoolctl import threadpool_limits

    global _CHANNEL
    if _CHANNEL is None and os.environ.get("ISCDC_GLUE_REQUEST_FD"):
        _CHANNEL = (
            os.fdopen(int(os.environ["ISCDC_GLUE_REQUEST_FD"]), "w", buffering=1),
            os.fdopen(int(os.environ["ISCDC_GLUE_RESPONSE_FD"]), "r"),
        )
    proc = psutil.Process()
    event = {
        "event": "request",
        "stage": name,
        "time": time.time(),
        "rss_bytes": proc.memory_info().rss,
        "cuda_reserved_bytes": torch.cuda.memory_reserved() if torch.cuda.is_initialized() else 0,
    }
    _event(event)
    allocation = None
    if _CHANNEL is not None:
        request, response = _CHANNEL
        request.write(json.dumps(event) + "\n")
        reply = response.readline()
        if not reply:
            raise RuntimeError("SpatialGlue supervisor resource lease closed")
        allocation = json.loads(reply)
        if allocation.get("state") != "granted":
            raise RuntimeError(f"SpatialGlue phase rejected: {allocation}")
        set_thread_affinity(allocation["cpus"])
        torch.set_num_threads(len(allocation["cpus"]))
        import numba

        numba.set_num_threads(len(allocation["cpus"]))
    started = time.monotonic()
    if torch.cuda.is_initialized():
        torch.cuda.reset_peak_memory_stats()
    _event(
        {
            "event": "start",
            "stage": name,
            "time": time.time(),
            "cpu_ids": sorted(os.sched_getaffinity(0)),
        }
    )
    try:
        with threadpool_limits(limits=len(allocation["cpus"]) if allocation else params["threads"]):
            yield
    finally:
        elapsed = time.monotonic() - started
        TIMINGS[name] = TIMINGS.get(name, 0) + elapsed
        if torch.cuda.is_initialized():
            CUDA_PEAKS["allocated"] = max(
                CUDA_PEAKS["allocated"], torch.cuda.max_memory_allocated()
            )
            CUDA_PEAKS["reserved"] = max(CUDA_PEAKS["reserved"], torch.cuda.max_memory_reserved())
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        event = {
            "event": "end",
            "stage": name,
            "time": time.time(),
            "elapsed_seconds": elapsed,
            "rss_bytes": proc.memory_info().rss,
            "cuda_reserved_bytes": torch.cuda.memory_reserved()
            if torch.cuda.is_initialized()
            else 0,
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved()
            if torch.cuda.is_initialized()
            else 0,
        }
        _event(event)
        if _CHANNEL is not None:
            set_thread_affinity([allocation["base_cpu"]])
            torch.set_num_threads(1)
            numba.set_num_threads(1)
            _CHANNEL[0].write(json.dumps(event) + "\n")
