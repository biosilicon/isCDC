"""Isolated, resource-admitted SpatialGLUE concurrency benchmark (locked GPU env).

This is an explicit-workload experiment runner, not a catalogue publisher. Each job
gets a separate sidecar root. Scientific functions are only wrapped for timing.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[3]
GIB = 2**30


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def worker(args):
    from iscdc import spatialglue as glue
    from iscdc.spatial_domain_annotation import main

    os.sched_setaffinity(0, [int(x) for x in os.environ["ISCDC_DOMAIN_CPU_IDS"].split(",")])
    metrics = {"stages": {}, "events": [], "started_epoch": time.time()}

    def timed(name, function):
        def call(*a, **kw):
            started = time.monotonic()
            metrics["events"].append({"stage": name, "event": "start", "utc_epoch": time.time()})
            write_json(args.job_dir / "timings.json", metrics)
            try:
                return function(*a, **kw)
            finally:
                metrics["stages"][name] = (
                    metrics["stages"].get(name, 0) + time.monotonic() - started
                )
                metrics["events"].append({"stage": name, "event": "end", "utc_epoch": time.time()})
                write_json(args.job_dir / "timings.json", metrics)

        return call

    for name in ("paired_sample", "prepare_modalities", "train", "cluster"):
        setattr(glue, name, timed(name, getattr(glue, name)))
    import importlib

    for package in ("SpatialGlue", "SpatialGlue_3M"):
        cls = importlib.import_module(f"{package}.SpatialGlue_pyG").Train_SpatialGlue
        cls.train = timed("epochs", cls.train)
    return main(
        [
            "generate-spatial-domain-visualization",
            args.dataset_id,
            "--method",
            "spatialglue",
            "--config",
            str(args.job_dir / "config.yaml"),
            "--output-root",
            str(args.job_dir / "sidecars"),
        ]
    )


def gpu_metrics(uuid):
    def query(arguments):
        return subprocess.check_output(["nvidia-smi", *arguments], text=True, timeout=10)

    raw = query(
        [
            "--query-gpu=utilization.gpu,memory.used,memory.free",
            "--format=csv,noheader,nounits",
            "-i",
            "GPU-" + uuid.removeprefix("GPU-"),
        ]
    )
    util, used, free = map(float, raw.strip().split(","))
    apps = query(["--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"])
    processes = {}
    for line in apps.strip().splitlines():
        pid, memory = line.split(",")
        try:
            processes[int(pid)] = float(memory)
        except ValueError:
            continue
    return {
        "utilization_percent": util,
        "used_mib": used,
        "free_mib": free,
        "process_mib": processes,
    }


def run(args):
    import psutil
    import yaml

    from iscdc.spatial_domain_resources import resource_lock_path
    from iscdc.spatialglue_resources import cuda_info, gpu_lock_path

    if Path(sys.prefix).name != "iscdc-spatial-domain-gpu":
        raise ValueError("Activate the locked iscdc-spatial-domain-gpu environment")
    if not 1 <= args.workers <= 20 or args.threads < 1 or args.workers * args.threads > 40:
        raise ValueError("At most 40 CPU threads and 20 workers")
    if args.repeats < 1 or (args.epochs is not None and args.epochs < 1):
        raise ValueError("Positive repeats and epochs required")
    rows = json.loads(args.workload.read_text())
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("An explicit nonempty workload with unique dataset IDs is required")
    args.output_root.mkdir(parents=True, exist_ok=False)
    cpus = sorted(os.sched_getaffinity(0))
    if len(cpus) < args.workers * args.threads + 8:
        raise ValueError("Insufficient CPU headroom")
    jobs = []
    # Larger jobs first reduces a long final tail; this order is fixed across trials.
    for row in sorted(rows, key=lambda r: r["estimated_host_bytes"], reverse=True):
        for repeat in range(args.repeats):
            jobs.append(
                {
                    "dataset_id": row["id"],
                    "repeat": repeat,
                    "host_gib": max(3, math.ceil(row["estimated_host_bytes"] / GIB * 1.15) + 1),
                    "gpu_gib": max(3, math.ceil(row["estimated_gpu_bytes"] / GIB * 1.25) + 1),
                }
            )
    info = cuda_info({"device": "cuda:0"})
    gpu_ceiling = min(64, int(info["free_bytes_at_start"] / GIB) - 8)
    host_ceiling = min(128, int(psutil.virtual_memory().available / GIB) - 8)
    if any(j["host_gib"] > host_ceiling or j["gpu_gib"] > gpu_ceiling for j in jobs):
        raise ValueError("A workload member exceeds the available aggregate budget")
    summary = {
        "workers_requested": args.workers,
        "threads_per_worker": args.threads,
        "epochs_override": args.epochs,
        "repeats": args.repeats,
        "device": info,
        "host_budget_gib": host_ceiling,
        "gpu_budget_gib": gpu_ceiling,
        "jobs": [],
        "peak_active": 0,
        "peak_rss_bytes": 0,
        "peak_own_gpu_mib": 0,
    }
    active, pending = [], jobs.copy()
    started = time.monotonic()
    last_heartbeat = started
    interrupted = []
    telemetry = (args.output_root / "telemetry.jsonl").open("w")

    def stop(signum, frame):
        interrupted.append(signum)
        for j in active:
            if j["process"].poll() is None:
                os.killpg(j["process"].pid, signal.SIGTERM)

    previous = {s: signal.signal(s, stop) for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        with (
            resource_lock_path().open("a") as cpu_lock,
            gpu_lock_path(info["uuid"]).open("a") as gl,
        ):
            fcntl.flock(cpu_lock, fcntl.LOCK_EX)
            fcntl.flock(gl, fcntl.LOCK_EX)
            while pending or active:
                if interrupted:
                    raise RuntimeError(f"Benchmark interrupted: {interrupted}")
                free_slots = [
                    s for s in range(args.workers) if s not in {j["slot"] for j in active}
                ]
                for slot in free_slots:
                    host_left = host_ceiling - sum(j["host_gib"] for j in active)
                    gpu_left = gpu_ceiling - sum(j["gpu_gib"] for j in active)
                    candidate = next(
                        (
                            j
                            for j in pending
                            if j["host_gib"] <= host_left and j["gpu_gib"] <= gpu_left
                        ),
                        None,
                    )
                    if candidate is None:
                        break
                    pending.remove(candidate)
                    job = dict(candidate)
                    job_dir = args.output_root / f"{jobs.index(candidate):02d}_{job['dataset_id']}"
                    job_dir.mkdir()
                    params = {
                        "threads": args.threads,
                        "memory_budget_gb": job["host_gib"],
                        "gpu_memory_budget_gb": job["gpu_gib"],
                    }
                    if args.epochs is not None:
                        params["epochs"] = args.epochs
                    (job_dir / "config.yaml").write_text(yaml.safe_dump({"defaults": params}))
                    env = os.environ.copy()
                    env.update(
                        PYTHONPATH=str(PROJECT / "src"),
                        ISCDC_GLUE_WORKER="1",
                        ISCDC_DOMAIN_RESOURCE_FD=str(cpu_lock.fileno()),
                        ISCDC_GLUE_GPU_FD=str(gl.fileno()),
                        ISCDC_DOMAIN_CPU_IDS=",".join(
                            map(str, cpus[slot * args.threads : (slot + 1) * args.threads])
                        ),
                        ISCDC_DATABASE_PATH=str(args.catalogue.resolve()),
                        PYTHONHASHSEED="42",
                        CUBLAS_WORKSPACE_CONFIG=":4096:8",
                        NUMBA_THREADING_LAYER="workqueue",
                        MPLCONFIGDIR="/tmp/iscdc-glue-bench",
                    )
                    for key in (
                        "OMP_NUM_THREADS",
                        "OPENBLAS_NUM_THREADS",
                        "MKL_NUM_THREADS",
                        "NUMBA_NUM_THREADS",
                    ):
                        env[key] = str(args.threads)
                    log = (job_dir / "worker.log").open("w")
                    process = subprocess.Popen(
                        [
                            sys.executable,
                            "-u",
                            __file__,
                            "--worker",
                            "--job-dir",
                            str(job_dir),
                            "--dataset-id",
                            job["dataset_id"],
                        ],
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                        pass_fds=(cpu_lock.fileno(), gl.fileno()),
                    )
                    job.update(
                        process=process,
                        log=log,
                        directory=job_dir,
                        slot=slot,
                        started=time.monotonic(),
                        peak_rss_bytes=0,
                        peak_gpu_mib=0,
                        cpu_seconds=0,
                    )
                    active.append(job)
                    print("START", job_dir.name, "pid", process.pid, flush=True)
                gpu = gpu_metrics(info["uuid"])
                rss_total, own_gpu = 0, 0
                for job in list(active):
                    pid = job["process"].pid
                    try:
                        process = psutil.Process(pid)
                        rss = process.memory_info().rss
                        cpu = sum(process.cpu_times()[:2])
                        rss_total += rss
                        job["peak_rss_bytes"] = max(job["peak_rss_bytes"], rss)
                        job["cpu_seconds"] = max(job["cpu_seconds"], cpu)
                        if rss > job["host_gib"] * GIB:
                            raise RuntimeError(
                                f"Per-job memory budget exceeded: {job['dataset_id']}"
                            )
                    except psutil.NoSuchProcess:
                        pass
                    used_gpu = gpu["process_mib"].get(pid, 0)
                    own_gpu += used_gpu
                    job["peak_gpu_mib"] = max(job["peak_gpu_mib"], used_gpu)
                    code = job["process"].poll()
                    if code is not None:
                        result = {
                            k: v
                            for k, v in job.items()
                            if k not in {"process", "log", "directory", "started"}
                        }
                        result.update(
                            directory=str(job["directory"]),
                            returncode=code,
                            wall_seconds=time.monotonic() - job["started"],
                        )
                        summary["jobs"].append(result)
                        job["log"].close()
                        active.remove(job)
                        print(
                            "FINISH",
                            job["dataset_id"],
                            job["repeat"],
                            code,
                            round(result["wall_seconds"], 2),
                            flush=True,
                        )
                        if code:
                            raise RuntimeError(f"Worker failed: {job['dataset_id']}")
                summary["peak_active"] = max(summary["peak_active"], len(active))
                summary["peak_rss_bytes"] = max(summary["peak_rss_bytes"], rss_total)
                summary["peak_own_gpu_mib"] = max(summary["peak_own_gpu_mib"], own_gpu)
                sample = {
                    "elapsed": time.monotonic() - started,
                    "active": len(active),
                    "rss_bytes": rss_total,
                    "own_gpu_mib": own_gpu,
                    "gpu": gpu,
                }
                telemetry.write(json.dumps(sample) + "\n")
                telemetry.flush()
                if (
                    rss_total > host_ceiling * GIB
                    or gpu["free_mib"] < 8192
                    or psutil.virtual_memory().available < 8 * GIB
                ):
                    raise RuntimeError("Aggregate resource limit or 8 GiB reserve violated")
                if time.monotonic() - last_heartbeat > 30:
                    print(
                        "HEARTBEAT",
                        len(active),
                        "pending",
                        len(pending),
                        "RSS GiB",
                        round(rss_total / GIB, 2),
                        "own GPU MiB",
                        own_gpu,
                        flush=True,
                    )
                    last_heartbeat = time.monotonic()
                if active:
                    time.sleep(1)
    except BaseException as exc:
        summary["error"] = repr(exc)
        raise
    finally:
        for job in active:
            if job["process"].poll() is None:
                os.killpg(job["process"].pid, signal.SIGTERM)
        for job in active:
            try:
                job["process"].wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(job["process"].pid, signal.SIGKILL)
                job["process"].wait()
            job["log"].close()
        for signum, handler in previous.items():
            signal.signal(signum, handler)
        telemetry.close()
        summary["total_wall_seconds"] = time.monotonic() - started
        write_json(args.output_root / "profile.json", summary)
    print("COMPLETE", args.output_root, round(summary["total_wall_seconds"], 2), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--job-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--dataset-id", help=argparse.SUPPRESS)
    parser.add_argument("--workload", type=Path)
    parser.add_argument("--catalogue", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--epochs", type=int)
    args = parser.parse_args()
    if args.worker:
        return worker(args)
    if not all((args.workload, args.catalogue, args.output_root)):
        parser.error("--workload, --catalogue and --output-root are required")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
