"""CUDA preflight and GPU leases for offline SpatialGlue jobs."""

from __future__ import annotations

import fcntl
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

GIB = 2**30


def cuda_info(params):
    import torch

    if not torch.cuda.is_available():
        raise ValueError("CUDA unavailable; SpatialGlue never falls back to CPU")
    device = torch.device(params["device"])
    torch.cuda.set_device(device)
    prop = torch.cuda.get_device_properties(device)
    free, total = torch.cuda.mem_get_info(device)
    return {
        "device": str(device),
        "uuid": str(prop.uuid),
        "name": prop.name,
        "cuda": torch.version.cuda,
        "torch": torch.__version__,
        "total_bytes": total,
        "free_bytes_at_start": free,
        "capability": list(torch.cuda.get_device_capability(device)),
    }


def gpu_lock_path(uuid):
    # UUID is provided by the CUDA runtime, never by a dataset.
    if not uuid or any(c not in "0123456789abcdefABCDEF-GPU" for c in uuid):
        raise ValueError("Invalid GPU UUID")
    return Path(tempfile.gettempdir()) / f"iscdc-spatialglue-{os.getuid()}-{uuid}.lock"


@contextmanager
def gpu_guard(info):
    path = gpu_lock_path(info["uuid"])
    inherited = os.environ.get("ISCDC_GLUE_GPU_FD")
    if inherited is not None:
        descriptor = int(inherited)
        stat, expected = os.fstat(descriptor), path.stat()
        if (stat.st_dev, stat.st_ino) != (expected.st_dev, expected.st_ino):
            raise ValueError("Invalid GPU resource lease")
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
        return
    with path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def estimates(n_obs, feature_counts, params, input_bytes=0):
    count = len(feature_counts)
    dimensions = sum(min(params["input_dims"], max(1, p)) for p in feature_counts.values())
    edges = n_obs * count * 2 * (params["feature_neighbors"] + (params["spatial_neighbors"] or 3))
    # Sparse input copies, centered PCA/SVD workspace, reduced features and CSR graphs.
    host = (
        input_bytes * 4
        + sum(feature_counts.values()) * (params["input_dims"] + 10) * 32
        + n_obs * dimensions * 64
        + edges * 48
        + GIB
    )
    # Full-batch official model: Adam/autograd activations, reduced inputs and sparse graphs.
    # The 690k pilot reserved 9.78 GiB at peak; its 3.57 GiB phase-end value is not a peak.
    gpu = (
        n_obs * (dimensions + count * params["dim_output"]) * 4 * 18
        + edges * 24
        + params["graph_workspace_mb"] * 2**20
        + GIB
    )
    return {"estimated_host_bytes": host, "estimated_gpu_bytes": gpu}


def check_resources(n_obs, feature_counts, params, input_bytes=0):
    import psutil
    import torch

    estimate = estimates(n_obs, feature_counts, params, input_bytes)
    host_budget = min(
        int(params["memory_budget_gb"] * GIB), max(0, psutil.virtual_memory().available - 8 * GIB)
    )
    free, total = torch.cuda.mem_get_info(params["device"])
    gpu_budget = min(
        int(params["gpu_memory_budget_gb"] * GIB),
        max(0, free - int(params["gpu_reserve_gb"] * GIB)),
    )
    if os.environ.get("ISCDC_GLUE_REQUEST_FD"):
        # The supervisor grants each phase before allocations; live free memory is
        # transient while other admitted phases are running.
        host_budget = int(params["memory_budget_gb"] * GIB)
        gpu_budget = int(params["gpu_memory_budget_gb"] * GIB)
    if estimate["estimated_host_bytes"] > host_budget:
        raise ValueError(f"SpatialGlue host resource preflight failed: {estimate}; no downsampling")
    if estimate["estimated_gpu_bytes"] > gpu_budget:
        raise ValueError(f"SpatialGlue GPU resource preflight failed: {estimate}; no CPU fallback")
    torch.cuda.set_per_process_memory_fraction(gpu_budget / total, params["device"])
    return {**estimate, "host_budget_bytes": host_budget, "gpu_budget_bytes": gpu_budget}
