"""Explicit-workload SpatialGLUE resource profiler and phase-admitted supervisor.

The catalogue and source H5MU are read-only. Every run requires a fresh output root.
Use the locked GPU environment for inventory, calibration and execution.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import selectors
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

GIB = 2**30


def scale_group(job):
    n = job["n_obs"]
    return "small" if n < 15000 else "medium" if n < 100000 else "large" if n < 300000 else "huge"


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def inventory(settings, config_path=None, dataset_ids=None):
    import h5py

    from . import spatialglue_config as config
    from .spatial_domain_annotation import catalogue_records
    from .spatialglue_resources import estimates

    jobs, excluded = [], []
    for record in catalogue_records(settings, dataset_ids):
        try:
            params = config.load_parameters(config_path, record["dataset_id"])
            config.eligibility(record, params)
            groups = config.expand_combinations(record, params)
        except ValueError as exc:
            excluded.append({"dataset_id": record["dataset_id"], "reason": str(exc)})
            continue
        matrices = {}
        with h5py.File(settings.data_root / record["storage_dir"] / "dataset.h5mu", "r") as source:
            for name in record["modalities"]:
                x = source[f"mod/{name}/X"]
                arrays = [x] if isinstance(x, h5py.Dataset) else list(x.values())
                matrices[name] = {
                    "bytes": sum(a.size * a.dtype.itemsize for a in arrays),
                    "nnz": int(x.size if isinstance(x, h5py.Dataset) else x["data"].size),
                    "features": record["modalities"][name]["n_vars"],
                }
        for modalities in groups:
            resolved = config.load_parameters(
                config_path, record["dataset_id"], modalities=modalities
            )
            features = {m: matrices[m]["features"] for m in modalities}
            size = sum(matrices[m]["bytes"] for m in modalities)
            job = {
                "dataset_id": record["dataset_id"],
                "combination_id": config.combination_id(modalities),
                "sample_ids": record["sample_ids"],
                "n_obs": record["n_obs"],
                "modalities": modalities,
                "source_sha256": record["sha256"],
                "feature_counts": features,
                "input_bytes": size,
                "nnz": sum(matrices[m]["nnz"] for m in modalities),
                "recipes": {
                    m: config.recipe(record, m, record["modalities"][m]["value_type"])
                    for m in modalities
                },
                "parameters": resolved,
                **estimates(record["n_obs"], features, resolved, size),
            }
            job["phases"] = phase_estimates(job)
            jobs.append(job)
    return {"inventory_version": 1, "jobs": jobs, "excluded": excluded}


def phase_estimates(job, calibration=None):
    """Analytic sparse bounds, replaced by measured upper-envelope ratios per phase."""
    n = job["n_obs"]
    inputs = job["input_bytes"]
    dims = sum(min(50, p) for p in job["feature_counts"].values())
    host, gpu = job["estimated_host_bytes"], job["estimated_gpu_bytes"]
    base = 1.25 * GIB
    graph_host = base + n * (dims * 8 + len(job["modalities"]) * 20 * 64)
    phases = {
        "pairing": {"host": base + 3 * inputs, "gpu": 0.6 * GIB},
        "preprocessing": {"host": host, "gpu": 0.6 * GIB},
        "graph": {
            "host": graph_host,
            "gpu": GIB + n * dims * 8 + 128 * 2**20,
        },
        "training": {"host": graph_host, "gpu": gpu},
        # The 690k pilot peaked at 14.7 GiB during clustering, not training.
        "clustering": {"host": base + n * 50 * 480, "gpu": 0.6 * GIB},
    }
    for name, estimate in phases.items():
        if calibration and name in calibration.get("phases", {}):
            fitted = calibration["phases"][name]
            fitted = fitted.get("scales", {}).get(scale_group(job), fitted)
            for resource in ("host", "gpu"):
                if resource not in fitted:
                    continue
                # Fixed interpreter/CUDA context plus a calibrated variable component.
                fixed = fitted[resource]["fixed_bytes"]
                ratio = fitted[resource]["variable_ratio"]
                estimate[resource] = max(fixed, fixed + max(0, estimate[resource] - fixed) * ratio)
    return phases


def fit_resources(jobs):
    samples = {}
    for job in jobs:
        if job["returncode"] != 0:
            continue
        predicted = phase_estimates(job["workload"])
        for name, measured in job.get("phase_peaks", {}).items():
            key = (name, scale_group(job["workload"]))
            samples.setdefault(key, []).append((predicted[name], measured))
    result = {"model_version": 2, "margin": 0.10, "phases": {}}
    for (name, group), rows in samples.items():
        fit = {}
        for resource, key, fixed in (
            ("host", "rss_bytes", 1.25 * GIB),
            ("gpu", "gpu_bytes", 0.6 * GIB),
        ):
            ratios = [
                max(0, measured[key] - fixed) / max(1, predicted[resource] - fixed)
                for predicted, measured in rows
            ]
            fit[resource] = {
                "fixed_bytes": fixed,
                "variable_ratio": max(0.05, max(ratios, default=1) * 1.10),
                "sample_count": len(rows),
            }
        result["phases"].setdefault(name, {"scales": {}})["scales"][group] = fit
    return result


def calibrate(profiles):
    jobs = []
    for profile in profiles:
        jobs.extend(json.loads(Path(profile).read_text())["jobs"])
    return {**fit_resources(jobs), "profiles": [str(p) for p in profiles]}


def next_pending(pending, active, cpus, host_live, gpu_live, threads):
    """First admissible queued job; a large waiting input must not leave slots idle."""
    occupied = {c for job in active for c in job["cpus"]}
    free_cpus = [c for c in cpus if c not in occupied]
    reserved_cpu = max(
        (
            job["threads"] - 1
            for job in active
            if job.get("waiting") and job.get("stage") in {"preprocessing", "clustering"}
        ),
        default=0,
    )
    host_used = sum(job["lease_host"] for job in active)
    gpu_used = sum(job["lease_gpu"] for job in active)
    for index, job in enumerate(pending):
        n = job["workload"]["n_obs"]
        need = threads or (1 if n < 15000 else 2 if n < 100000 else 8)
        # Keep enough spare CPU IDs for at least one worker to enter its CPU phase.
        minimum = min([j["threads"] for j in active] + [need])
        initial = job["phases"]["pairing"]
        if (
            len(free_cpus) >= max(minimum, reserved_cpu + 1)
            and host_used + max(1.5 * GIB, initial["host"]) <= host_live
            and gpu_used + max(0.6 * GIB, initial["gpu"]) <= gpu_live
        ):
            return index, free_cpus[0]
    return None


def worker_parameters(row, threads=0, epochs=None):
    n = row["n_obs"]
    params = {
        **row.get("parameters", {}),
        "threads": threads or (1 if n < 15000 else 2 if n < 100000 else 8),
        "memory_budget_gb": min(768, math.ceil(row["estimated_host_bytes"] / GIB) + 2),
        "gpu_memory_budget_gb": min(80, math.ceil(row["estimated_gpu_bytes"] / GIB) + 2),
        "graph_workspace_mb": 128,
    }
    if epochs is not None:
        params["epochs"] = epochs
    return params


def log_tail(path):
    try:
        with Path(path).open("rb") as stream:
            stream.seek(max(0, stream.seek(0, 2) - 2048))
            lines = stream.read().decode("utf-8", errors="replace").splitlines()
        return next((line.strip()[-240:] for line in reversed(lines) if line.strip()), "")
    except OSError:
        return ""


def emit(root, event, **fields):
    row = {"time": datetime.now(timezone.utc).isoformat(), "event": event, **fields}
    line = json.dumps(row, ensure_ascii=False)
    with (root / "events.jsonl").open("a") as stream:
        stream.write(line + "\n")
    print(line, flush=True)


def run(args):
    import psutil
    import yaml

    from .config import PROJECT_ROOT
    from .spatial_domain_resources import resource_lock_path
    from .spatialglue_monitor import GPUMonitor
    from .spatialglue_resources import cuda_info, gpu_lock_path

    if Path(sys.prefix).name != "iscdc-spatial-domain-gpu":
        raise ValueError("Activate the locked iscdc-spatial-domain-gpu environment")
    if not (1 <= args.workers <= 80 and 0 <= args.threads <= 80 and args.repeats >= 1):
        raise ValueError("At most 80 concurrent CPU threads; positive workers/threads/repeats")
    workload = json.loads(args.workload.read_text())
    rows = workload["jobs"] if isinstance(workload, dict) else workload
    if args.epochs is not None and args.epochs < 1:
        raise ValueError("epochs must be positive")
    if not rows:
        raise ValueError("Empty explicit workload")
    if getattr(args, "sidecars_root", None) and args.repeats != 1:
        raise ValueError("Shared sidecars require repeats=1")
    if len({(r["dataset_id"], r["combination_id"]) for r in rows}) != len(rows):
        raise ValueError("Duplicate dataset/combination in workload")
    heartbeat = getattr(args, "heartbeat", 15)
    if heartbeat < 1:
        raise ValueError("heartbeat must be positive")
    args.output_root.mkdir(parents=True, exist_ok=False)
    emit(args.output_root, "initializing", tasks=len(rows), workers=args.workers)
    info = cuda_info({"device": "cuda:0"})
    cpus = sorted(os.sched_getaffinity(0))[:80]
    if len(cpus) < args.workers:
        raise ValueError("Insufficient CPU affinity for requested workload")
    available_start = psutil.virtual_memory().available
    reserve_host = max(64 * GIB, 0.15 * available_start)
    host_cap = min(768 * GIB, available_start - reserve_host)
    gpu_cap = info["free_bytes_at_start"] - 8 * GIB
    model = json.loads(args.calibration.read_text()) if args.calibration else None
    jobs = []
    for row in sorted(rows, key=lambda r: (r["n_obs"] ** 2, r["input_bytes"]), reverse=True):
        for repeat in range(args.repeats):
            jobs.append(
                {
                    "index": len(jobs),
                    "workload": row,
                    "repeat": repeat,
                    "phases": phase_estimates(row, model),
                    "phase_peaks": {},
                    "peak_rss_bytes": 0,
                    "peak_gpu_bytes": 0,
                }
            )
    pending, active, completed = jobs.copy(), [], []
    result = {
        "profile_version": 1,
        "workers_requested": args.workers,
        "threads_per_worker": args.threads or "auto_by_scale",
        "epochs_override": args.epochs,
        "device": info,
        "host_cap_bytes": host_cap,
        "gpu_cap_bytes": gpu_cap,
        "host_reserve_bytes": reserve_host,
        "peak_active": 0,
        "peak_computing": 0,
        "peak_cpu_leases": 0,
        "peak_rss_bytes": 0,
        "peak_gpu_bytes": 0,
        "jobs": completed,
    }
    selector = selectors.DefaultSelector()
    began = time.monotonic()
    interrupted = []
    previous = {
        sig: signal.signal(sig, lambda s, f: interrupted.append(s))
        for sig in (signal.SIGINT, signal.SIGTERM)
    }
    telemetry = (args.output_root / "telemetry.jsonl").open("w")
    last_monitor = last_heartbeat = 0
    monitor = GPUMonitor(info["uuid"])
    gpu = monitor.snapshot()
    monitor_was_ready = None

    def clean(job):
        if job["process"].poll() is None:
            emit(args.output_root, "terminating_worker", pid=job["process"].pid)
            os.killpg(job["process"].pid, signal.SIGTERM)
            try:
                job["process"].wait(timeout=10)
            except subprocess.TimeoutExpired:
                emit(args.output_root, "killing_worker", pid=job["process"].pid)
                os.killpg(job["process"].pid, signal.SIGKILL)
                job["process"].wait()
        job["log"].close()
        selector.unregister(job["request"])
        os.close(job["request"])
        os.close(job["response"])

    def identity(job):
        return {
            "index": job["index"],
            "pid": job["process"].pid,
            "dataset_id": job["workload"]["dataset_id"],
            "combination_id": job["workload"]["combination_id"],
            "n_obs": job["workload"]["n_obs"],
        }

    def progress(state="running"):
        now = time.monotonic()
        workers = []
        for job in active:
            log_path = Path(job["directory"]) / "run.log"
            workers.append(
                {
                    **identity(job),
                    "stage": job.get("stage")
                    or (
                        "writing_results"
                        if "clustering" in job.get("phase_seconds", {})
                        else "initializing_or_between_stages"
                    ),
                    "waiting": job.get("waiting", False),
                    "wait_reason": job.get("wait_reason"),
                    "elapsed_seconds": round(now - job["started"], 1),
                    "stage_seconds": round(now - job.get("stage_started", job["started"]), 1),
                    "cpu_ids": job["cpus"],
                    "rss_bytes": job.get("rss", 0),
                    "gpu_bytes": job.get("gpu", 0),
                    "cpu_percent": job.get("cpu_percent", 0),
                    "lease_host_bytes": job["lease_host"],
                    "lease_gpu_bytes": job["lease_gpu"],
                    "log": str(log_path),
                    "last_output": log_tail(log_path),
                    "log_age_seconds": round(time.time() - log_path.stat().st_mtime, 1),
                }
            )
        value = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "state": state,
            "pid": os.getpid(),
            "total": len(jobs),
            "completed": len(completed),
            "successful": sum(j["returncode"] == 0 for j in completed),
            "failed": sum(j["returncode"] != 0 for j in completed),
            "pending": len(pending),
            "active_count": len(workers),
            "waiting": sum(j["waiting"] for j in workers),
            "active": workers,
            "elapsed_seconds": round(now - began, 1),
            "host_cap_bytes": host_cap,
            "gpu_cap_bytes": gpu_cap,
            "host_reserve_bytes": reserve_host,
            "rss_bytes": sum(j.get("rss", 0) for j in active),
            "gpu_bytes": sum(j.get("gpu", 0) for j in active),
            "gpu_free_bytes": gpu["free_bytes"],
            "whole_gpu_utilization_percent": gpu["utilization_percent"],
            "gpu_monitor": {k: gpu[k] for k in ("backend", "fresh", "sample_age_seconds", "error")},
            "host_available_bytes": psutil.virtual_memory().available,
            "cpu_leases": sum(len(j["cpus"]) for j in active),
        }
        write_json(args.output_root / "progress.json", value)
        emit(args.output_root, "heartbeat", **value)

    try:
        with (
            resource_lock_path().open("a") as cpu_lock,
            gpu_lock_path(info["uuid"]).open("a") as gl,
        ):
            for lease in (cpu_lock, gl):
                emit(args.output_root, "waiting_for_lock", lock=str(lease.name))
                while True:
                    if interrupted:
                        raise RuntimeError(
                            f"Interrupted while waiting for resource lease: {interrupted}"
                        )
                    try:
                        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        if time.monotonic() - last_heartbeat >= heartbeat:
                            progress("waiting_for_lock")
                            last_heartbeat = time.monotonic()
                        time.sleep(0.25)
            result["queue_wait_seconds"] = time.monotonic() - began
            began = time.monotonic()
            info = cuda_info({"device": "cuda:0"})
            available_start = psutil.virtual_memory().available
            reserve_host = max(64 * GIB, 0.15 * available_start)
            host_cap = min(768 * GIB, available_start - reserve_host)
            gpu_cap = info["free_bytes_at_start"] - 8 * GIB
            result.update(
                device=info,
                host_cap_bytes=host_cap,
                gpu_cap_bytes=gpu_cap,
                host_reserve_bytes=reserve_host,
            )
            emit(
                args.output_root,
                "resources_acquired",
                host_cap_gib=host_cap / GIB,
                gpu_cap_gib=gpu_cap / GIB,
                cpu_core_cap=len(cpus),
            )
            while pending or active:
                if interrupted:
                    raise RuntimeError(f"Interrupted: {interrupted}")
                now = time.monotonic()
                gpu = monitor.snapshot()
                if gpu["fresh"] != monitor_was_ready:
                    emit(
                        args.output_root,
                        "gpu_monitor_ready" if gpu["fresh"] else "gpu_monitor_waiting",
                        backend=gpu["backend"],
                        sample_age_seconds=gpu["sample_age_seconds"],
                        error=gpu["error"],
                    )
                    monitor_was_ready = gpu["fresh"]
                if now - last_monitor >= 1:
                    sample = {
                        "time": time.time(),
                        "workers": [],
                        "gpu": gpu,
                        "available_host_bytes": psutil.virtual_memory().available,
                    }
                    own_rss = psutil.Process().memory_info().rss
                    own_gpu = gpu["process_bytes"].get(os.getpid(), 0)
                    for job in active:
                        try:
                            proc = psutil.Process(job["process"].pid)
                            descendants = [proc, *proc.children(recursive=True)]
                            rss = sum(p.memory_info().rss for p in descendants)
                            used = (
                                sum(gpu["process_bytes"].get(p.pid, 0) for p in descendants)
                                if gpu["fresh"]
                                else job.get("gpu", 0)
                            )
                            cpu_seconds = sum(sum(p.cpu_times()[:2]) for p in descendants)
                            previous_cpu = job.get("cpu_sample")
                            job["cpu_percent"] = (
                                round(
                                    100 * (cpu_seconds - previous_cpu[1]) / (now - previous_cpu[0]),
                                    1,
                                )
                                if previous_cpu
                                else 0
                            )
                            job["cpu_sample"] = (now, cpu_seconds)
                        except psutil.NoSuchProcess:
                            continue
                        own_rss += rss
                        own_gpu += used
                        job["rss"], job["gpu"] = rss, used
                        job["lease_host"] = max(job["lease_host"], rss)
                        job["lease_gpu"] = max(job["lease_gpu"], used)
                        job["peak_rss_bytes"] = max(job["peak_rss_bytes"], rss)
                        job["peak_gpu_bytes"] = max(job["peak_gpu_bytes"], used)
                        if job.get("stage") and not job.get("waiting"):
                            peak = job["phase_peaks"].setdefault(
                                job["stage"], {"rss_bytes": 0, "gpu_bytes": 0}
                            )
                            peak["rss_bytes"] = max(peak["rss_bytes"], rss)
                            peak["gpu_bytes"] = max(peak["gpu_bytes"], used)
                        sample["workers"].append(
                            {
                                "index": job["index"],
                                "stage": job.get("stage"),
                                "waiting": job.get("waiting", False),
                                "cpu_ids": job["cpus"],
                                "rss_bytes": rss,
                                "gpu_bytes": used,
                            }
                        )
                    result["peak_rss_bytes"] = max(result["peak_rss_bytes"], own_rss)
                    result["peak_gpu_bytes"] = max(result["peak_gpu_bytes"], own_gpu)
                    telemetry.write(json.dumps(sample) + "\n")
                    telemetry.flush()
                    if (gpu["fresh"] and gpu["free_bytes"] < 8 * GIB) or sample[
                        "available_host_bytes"
                    ] < reserve_host:
                        progress("resource_reserve_exhausted")
                        raise RuntimeError("Live resource reserve exhausted; profile retained")
                    last_monitor = now
                # Dispatch by longest expected graph workload, backfilling only within
                # the stage leases. Pending large phase requests precede fresh imports.
                for key, _ in selector.select(timeout=0.05):
                    job = key.data
                    chunk = os.read(job["request"], 65536)
                    job["buffer"] += chunk
                    while b"\n" in job["buffer"]:
                        raw, job["buffer"] = job["buffer"].split(b"\n", 1)
                        event = json.loads(raw)
                        job["rss"] = event["rss_bytes"]
                        if event["event"] == "request":
                            job["stage"], job["waiting"] = event["stage"], True
                            job["stage_started"] = now
                            emit(
                                args.output_root,
                                "phase_requested",
                                **identity(job),
                                stage=event["stage"],
                            )
                        else:
                            emit(
                                args.output_root,
                                "phase_finished",
                                **identity(job),
                                stage=event["stage"],
                                seconds=event["elapsed_seconds"],
                                rss_gib=event["rss_bytes"] / GIB,
                                peak_torch_reserved_gib=event.get("peak_cuda_reserved_bytes", 0)
                                / GIB,
                            )
                            job["lease_host"] = max(1.25 * GIB, event["rss_bytes"])
                            job["lease_gpu"] = 0.6 * GIB + event["cuda_reserved_bytes"]
                            peak = job["phase_peaks"].setdefault(
                                event["stage"], {"rss_bytes": 0, "gpu_bytes": 0}
                            )
                            peak["rss_bytes"] = max(peak["rss_bytes"], event["rss_bytes"])
                            peak["gpu_bytes"] = max(
                                peak["gpu_bytes"],
                                0.6 * GIB + event.get("peak_cuda_reserved_bytes", 0),
                            )
                            job.setdefault("phase_seconds", {})[event["stage"]] = event[
                                "elapsed_seconds"
                            ]
                            job["cpus"] = [job["base_cpu"]]
                            job["stage"] = None
                for job in list(active):
                    code = job["process"].poll()
                    if code is not None:
                        clean(job)
                        active.remove(job)
                        report = {
                            k: v
                            for k, v in job.items()
                            if k not in {"process", "log", "request", "response", "buffer"}
                        }
                        report.update(returncode=code, elapsed_seconds=now - job["started"])
                        completed.append(report)
                        emit(
                            args.output_root,
                            "job_success" if code == 0 else "job_failed",
                            **identity(job),
                            returncode=code,
                            seconds=report["elapsed_seconds"],
                            log=str(Path(job["directory"]) / "run.log"),
                            last_output=log_tail(Path(job["directory"]) / "run.log"),
                        )
                        measured = fit_resources(completed)
                        model = model or {"model_version": 2, "phases": {}}
                        for name, fitted in measured["phases"].items():
                            model["phases"].setdefault(name, {}).setdefault("scales", {}).update(
                                fitted["scales"]
                            )
                        model["margin"] = 0.10
                        for remaining in [*pending, *active]:
                            remaining["phases"] = phase_estimates(remaining["workload"], model)
                        write_json(args.output_root / "calibration.json", model)
                        write_json(args.output_root / "profile.json", result)
                host_live = min(
                    host_cap,
                    psutil.virtual_memory().available
                    + sum(j.get("rss", 0) for j in active)
                    - reserve_host,
                )
                gpu_live = (
                    min(gpu_cap, gpu["free_bytes"] + sum(j.get("gpu", 0) for j in active) - 8 * GIB)
                    if gpu["fresh"]
                    else 0
                )
                waiting = sorted((j for j in active if j.get("waiting")), key=lambda j: j["index"])
                blocked = None
                for job in waiting:
                    estimate = job["phases"][job["stage"]]
                    host = max(job.get("rss", 0), estimate["host"])
                    dev = max(job.get("gpu", 0), estimate["gpu"])
                    others = [j for j in active if j is not job]
                    occupied = {c for j in others for c in j["cpus"]}
                    free_cpus = [c for c in cpus if c not in occupied and c != job["base_cpu"]]
                    cpu_need = (
                        job["threads"] if job["stage"] in {"preprocessing", "clustering"} else 1
                    )

                    fits = (
                        gpu["fresh"]
                        and len(free_cpus) + 1 >= cpu_need
                        and sum(j["lease_host"] for j in others) + host <= host_live
                        and sum(j["lease_gpu"] for j in others) + dev <= gpu_live
                    )
                    if fits:
                        job["lease_host"], job["lease_gpu"] = host, dev
                        job["waiting"] = False
                        job["wait_reason"] = None
                        job["stage_started"] = now
                        job["cpus"] = [job["base_cpu"], *free_cpus[: cpu_need - 1]]
                        emit(
                            args.output_root,
                            "phase_started",
                            **identity(job),
                            stage=job["stage"],
                            cpu_ids=job["cpus"],
                            reserved_host_gib=host / GIB,
                            reserved_gpu_gib=dev / GIB,
                        )
                        os.write(
                            job["response"],
                            (
                                json.dumps(
                                    {
                                        "state": "granted",
                                        "cpus": job["cpus"],
                                        "base_cpu": job["base_cpu"],
                                    }
                                )
                                + "\n"
                            ).encode(),
                        )
                    else:
                        job["wait_reason"] = ",".join(
                            name
                            for name, unavailable in (
                                ("gpu_monitor", not gpu["fresh"]),
                                ("cpu", len(free_cpus) + 1 < cpu_need),
                                ("ram", sum(j["lease_host"] for j in others) + host > host_live),
                                ("gpu", sum(j["lease_gpu"] for j in others) + dev > gpu_live),
                            )
                            if unavailable
                        )
                        if blocked is None:
                            blocked = job
                # Idle workers retain only their actual resident data and CUDA context.
                while pending and len(active) < args.workers and monitor.snapshot()["fresh"]:
                    candidate = next_pending(
                        pending, active, cpus, host_live, gpu_live, args.threads
                    )
                    if candidate is None:
                        break
                    index, base_cpu = candidate
                    job = pending.pop(index)
                    slot = next(
                        i for i in range(args.workers) if i not in {j["slot"] for j in active}
                    )
                    n = job["workload"]["n_obs"]
                    job["threads"] = args.threads or (1 if n < 15000 else (2 if n < 100000 else 8))
                    job["base_cpu"] = base_cpu
                    job["cpus"] = [base_cpu]
                    folder = args.output_root / f"job_{job['index']:04d}"
                    folder.mkdir()
                    job.update(
                        slot=slot,
                        directory=str(folder),
                        started=time.monotonic(),
                        lease_host=max(1.5 * GIB, job["phases"]["pairing"]["host"]),
                        lease_gpu=max(0.6 * GIB, job["phases"]["pairing"]["gpu"]),
                        buffer=b"",
                    )
                    # Bounds for the allocator/preflight; admission remains phase-specific.
                    row = job["workload"]
                    params = worker_parameters(row, args.threads, args.epochs)
                    (folder / "config.yaml").write_text(yaml.safe_dump({"defaults": params}))
                    req_read, req_write = os.pipe()
                    resp_read, resp_write = os.pipe()
                    env = os.environ.copy()
                    env.update(
                        PYTHONPATH=str(PROJECT_ROOT / "src"),
                        ISCDC_DATABASE_PATH=str(args.catalogue.resolve()),
                        ISCDC_GLUE_WORKER="1",
                        PYTHONHASHSEED="42",
                        CUBLAS_WORKSPACE_CONFIG=":4096:8",
                        NUMBA_THREADING_LAYER="workqueue",
                        ISCDC_DOMAIN_RESOURCE_FD=str(cpu_lock.fileno()),
                        ISCDC_GLUE_GPU_FD=str(gl.fileno()),
                        ISCDC_GLUE_REQUEST_FD=str(req_write),
                        ISCDC_GLUE_RESPONSE_FD=str(resp_read),
                        ISCDC_GLUE_EVENTS=str((folder / "events.jsonl").resolve()),
                        ISCDC_DOMAIN_CPU_IDS=str(job["base_cpu"]),
                    )
                    for name in (
                        "OMP_NUM_THREADS",
                        "OPENBLAS_NUM_THREADS",
                        "MKL_NUM_THREADS",
                        "NUMBA_NUM_THREADS",
                    ):
                        env[name] = str(job["threads"])
                    env["ISCDC_GLUE_CACHE_ROOT"] = str(
                        args.cache_root.resolve()
                        if args.cache_root
                        else (args.output_root / "cache").resolve()
                    )
                    command = [
                        sys.executable,
                        "-u",
                        "-m",
                        "iscdc.spatial_domain_annotation",
                        "generate-spatial-domain-visualization",
                        row["dataset_id"],
                        "--method",
                        "spatialglue",
                        "--combination",
                        row["combination_id"],
                        "--config",
                        str(folder / "config.yaml"),
                        "--output-root",
                        str(getattr(args, "sidecars_root", None) or folder / "sidecars"),
                    ]
                    if getattr(args, "resume", False):
                        command.append("--resume")
                    if getattr(args, "force", False):
                        command.append("--force")
                    log = (folder / "run.log").open("w")
                    job["process"] = subprocess.Popen(
                        command,
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                        pass_fds=(cpu_lock.fileno(), gl.fileno(), req_write, resp_read),
                    )
                    os.close(req_write)
                    os.close(resp_read)
                    job.update(request=req_read, response=resp_write, log=log)
                    selector.register(req_read, selectors.EVENT_READ, job)
                    active.append(job)
                    emit(
                        args.output_root,
                        "job_started",
                        **identity(job),
                        threads=job["threads"],
                        log=str(folder / "run.log"),
                        sidecars=command[command.index("--output-root") + 1],
                    )
                result["peak_cpu_leases"] = max(
                    result["peak_cpu_leases"], sum(len(j["cpus"]) for j in active)
                )
                result["peak_active"] = max(result["peak_active"], len(active))
                result["peak_computing"] = max(
                    result["peak_computing"],
                    sum(bool(j.get("stage")) and not j.get("waiting") for j in active),
                )
                if now - last_heartbeat >= heartbeat:
                    progress("running" if gpu["fresh"] else "waiting_for_gpu_monitor")
                    last_heartbeat = now
                if gpu["fresh"] and active and all(j.get("waiting") for j in active):
                    # Nothing can release a lease: fail clearly rather than wait forever.
                    if blocked and all(j["waiting"] for j in active):
                        raise RuntimeError("No phase fits the current resource budget")
    except BaseException as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        emit(args.output_root, "batch_error", error=result["error"])
        raise
    finally:
        monitor.close()
        result["interrupted_jobs"] = [
            {"index": job["index"], "workload": job["workload"], "stage": job.get("stage")}
            for job in active
        ]
        result["pending_jobs"] = [
            {"index": job["index"], "workload": job["workload"]} for job in pending
        ]
        for job in active:
            clean(job)
        active.clear()
        result["elapsed_seconds"] = time.monotonic() - began
        result["completed"] = len(completed)
        result["calibration"] = model
        result["failed"] = sum(j["returncode"] != 0 for j in completed)
        write_json(args.output_root / "profile.json", result)
        progress(
            "interrupted"
            if interrupted
            else "failed"
            if result.get("error") or result["failed"]
            else "complete"
        )
        telemetry.close()
        selector.close()
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    return int(result["failed"] > 0)


def main(argv=None):
    from .config import Settings

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    inv = sub.add_parser("inventory")
    inv.add_argument("--output", type=Path, required=True)
    cal = sub.add_parser("calibrate")
    cal.add_argument("profiles", type=Path, nargs="+")
    cal.add_argument("--output", type=Path, required=True)
    bench = sub.add_parser("run")
    bench.add_argument("--workload", type=Path, required=True)
    bench.add_argument("--catalogue", type=Path, required=True)
    bench.add_argument("--output-root", type=Path, required=True)
    bench.add_argument(
        "--workers", type=int, default=80, help="Resource-admitted task ceiling (default: 80)"
    )
    bench.add_argument(
        "--threads", type=int, default=0, help="0: 1/2/8 threads by observation scale"
    )
    bench.add_argument("--repeats", type=int, default=1)
    bench.add_argument("--epochs", type=int)
    bench.add_argument("--calibration", type=Path)
    bench.add_argument("--cache-root", type=Path)
    bench.add_argument("--sidecars-root", type=Path)
    existing = bench.add_mutually_exclusive_group()
    existing.add_argument("--resume", action="store_true")
    existing.add_argument("--force", action="store_true")
    bench.add_argument("--heartbeat", type=int, default=15)
    args = parser.parse_args(argv)
    if args.command == "inventory":
        write_json(args.output, inventory(Settings.from_environment()))
        return 0
    if args.command == "calibrate":
        write_json(args.output, calibrate(args.profiles))
        return 0
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
