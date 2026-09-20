"""Resource-budgeted parallel continuation of frozen spatial-domain batch runs."""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import heapq
import json
import math
import os
import shutil
import signal
import statistics
import subprocess
import time
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml

from . import cell_type_visualization as ct
from . import spatial_domain_batch as batch
from .config import PROJECT_ROOT, Settings
from .spatial_domain_annotation import catalogue_records, estimated_memory_bytes, load_parameters
from .spatial_domain_resources import resource_lock_path
from .spatial_domain_visualization import load_spatial_domain_visualization

GIB = 2**30


def requirements(record, params, cpu_budget):
    method = batch.method_for_resolution(record["spatial_unit"])
    rna = record["modalities"]["rna"]
    estimate = estimated_memory_bytes(
        rna["n_obs"], min(rna["n_vars"], params["n_top_genes"]), method
    )
    return {
        "cpu_count": min(cpu_budget, 8 if method == "GraphST" else 16),
        "memory_gb": math.ceil(estimate / GIB * 1.15 + 1),
        "preflight_estimate_bytes": estimate,
    }


def choose_jobs(jobs, active, free_cpus, memory_gb, max_workers):
    reserved = sum(item["job"]["resources"]["memory_gb"] for item in active.values())
    slots = max_workers - len(active)
    chosen = []
    # Explicit repair retries first, then large jobs; fill spare memory/CPU with smaller jobs.
    pending = sorted(jobs, key=lambda item: (not item.get("attempt_history"), -item["work_units"]))
    for job in pending:
        if job["state"] != "pending":
            continue
        resources = job["resources"]
        # With a large job already running, fill spare CPUs with fitting lighter jobs
        # before letting another large job consume nearly all remaining memory.
        large_running = any(
            item["job"]["resources"]["memory_gb"] > 48 for item in active.values()
        ) or any(item["resources"]["memory_gb"] > 48 for item in chosen)
        if (
            large_running
            and resources["memory_gb"] > 48
            and any(
                other not in chosen
                and other["state"] == "pending"
                and other["resources"]["memory_gb"] <= 48
                and other["resources"]["cpu_count"] <= free_cpus
                and reserved + other["resources"]["memory_gb"] <= memory_gb
                for other in pending
            )
        ):
            continue
        if (
            slots
            and resources["cpu_count"] <= free_cpus
            and reserved + resources["memory_gb"] <= memory_gb
        ):
            chosen.append(job)
            free_cpus -= resources["cpu_count"]
            reserved += resources["memory_gb"]
            slots -= 1
    return chosen


def parallel_progress(jobs, active, policy, heartbeat):
    result = batch.estimate_progress(jobs, heartbeat=heartbeat)
    rates = {"BANKSY": [], "GraphST": []}
    for job in jobs:
        if job["state"] in {"success", "reused"} and "inference_seconds" in job:
            rates[job["method"]].append(max(1, job["inference_seconds"] - 20) / job["work_units"])
    defaults = {"BANKSY": 76 / (23496 * 399), "GraphST": 95 / 2373**2}
    rates = {
        key: statistics.median(values) if values else defaults[key] for key, values in rates.items()
    }
    queue = []
    now = time.monotonic()
    banksy_anchors = [
        (job["n_obs"], job["inference_seconds"])
        for job in jobs
        if job["method"] == "BANKSY"
        and job["state"] in {"success", "reused"}
        and job.get("n_obs", 0) > 0
        and "inference_seconds" in job
    ]
    overdue = []
    for key, item in active.items():
        job = item["job"]
        age = now - item["started"]
        if age > 20 + rates[job["method"]] * job["work_units"]:
            overdue.append(key)
            if job["method"] == "BANKSY" and job.get("n_obs", 0) > 0:
                # Censored observations are lower bounds, not completed calibration data.
                banksy_anchors.append((job["n_obs"], age * 1.5))
    for job in jobs:
        if job["state"] in batch.TERMINAL:
            continue
        seconds = 20 + rates[job["method"]] * job["work_units"]
        if job["method"] == "BANKSY" and job.get("n_obs", 0) > 0:
            anchors = [pair for pair in banksy_anchors if pair[0] <= job["n_obs"]]
            if anchors:
                size, duration = max(anchors)
                # Large Leiden graphs need a nonlinear observation-count floor.
                seconds = max(seconds, duration * (job["n_obs"] / size) ** 1.5)
        if job["dataset_id"] in active:
            age = now - active[job["dataset_id"]]["started"]
            seconds = max(
                heartbeat, seconds - age, age * 0.5 if job["dataset_id"] in overdue else 0
            )
        queue.append((job, seconds))
    # Discrete-event packing includes the CPU and memory constraints, not simply / workers.
    queue.sort(
        key=lambda pair: (
            pair[0]["state"] != "running",
            not pair[0].get("attempt_history"),
            -pair[0]["work_units"],
        )
    )
    running = []
    elapsed, used_cpu, used_mem, sequence = 0.0, 0, 0, 0
    while queue or running:
        for job, seconds in queue[:]:
            cpu, mem = job["resources"]["cpu_count"], job["resources"]["memory_gb"]
            if mem > policy["memory_gb"]:
                queue.remove((job, seconds))
                continue
            if (
                len(running) < policy["max_workers"]
                and used_cpu + cpu <= policy["cpu_budget"]
                and used_mem + mem <= policy["memory_gb"]
            ):
                sequence += 1
                heapq.heappush(running, (elapsed + seconds, sequence, cpu, mem))
                used_cpu += cpu
                used_mem += mem
                queue.remove((job, seconds))
        if not running:
            break
        elapsed, _, cpu, mem = heapq.heappop(running)
        used_cpu -= cpu
        used_mem -= mem
    result.update(
        remaining_seconds_estimate=round(elapsed),
        estimated_finish_at=(datetime.now(timezone.utc) + timedelta(seconds=elapsed)).isoformat(),
        eta_overdue_jobs=overdue,
        eta_note=(
            "Heuristic resource-constrained estimate with BANKSY n_obs^1.5 scaling and "
            "unfinished-job lower bounds. Overdue jobs retain at least half their elapsed "
            "time as estimated remaining work; this is not a completion guarantee."
        ),
    )
    return result


def continue_run(
    previous, root, *, max_workers=8, cpu_budget=64, memory_gb=192, retry_failed=False
):
    previous, root = previous.resolve(), root.resolve()
    if root.exists():
        raise ValueError("Continuation directory already exists")
    old = batch.read_json(previous / "progress.json")
    if old["state"] not in {"interrupted", "complete"}:
        raise ValueError("Stop the previous supervisor cleanly before continuing")
    if not 1 <= cpu_budget <= max(1, len(os.sched_getaffinity(0)) - 8):
        raise ValueError("CPU budget must leave at least eight available logical CPUs")
    if max_workers < 1 or not math.isfinite(memory_gb) or memory_gb <= 0:
        raise ValueError("Invalid scheduler budget")
    plan = batch.read_json(previous / "plan.json")
    # Refuse to snapshot a supervisor that still owns its directory.
    with (previous / ".supervisor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        root.mkdir(parents=True)
        shutil.copyfile(previous / "catalog.db", root / "catalog.db")
        if ct._file_digest(root / "catalog.db")[1] != plan["catalogue_sha256"]:
            raise ValueError("Previous catalogue snapshot changed")
        for directory in ("sidecars", "logs"):
            shutil.copytree(previous / directory, root / directory)
        code = root / "code"
        shutil.copytree(
            PROJECT_ROOT / "src" / "iscdc",
            code / "src" / "iscdc",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        for relative in (
            "assets/spatial_domain/defaults.yaml",
            "annotation/spatial_domain/requirements.lock.txt",
        ):
            target = code / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(previous / "code" / relative, target)
        jobs = copy.deepcopy(old["jobs"])
        settings = replace(Settings.from_environment(), database_path=root / "catalog.db")
        records = {record["dataset_id"]: record for record in catalogue_records(settings)}
        for job in jobs:
            if retry_failed and job["state"] == "failed":
                job.setdefault("attempt_history", []).append(
                    copy.deepcopy(
                        {key: value for key, value in job.items() if key != "attempt_history"}
                    )
                )
                job["state"] = "pending"
            if job["state"] in {"running", "interrupted"}:
                job["state"] = "pending"
            if job["state"] in {"success", "reused"}:
                load_spatial_domain_visualization(root / "sidecars", records[job["dataset_id"]])
                job["carried_from"] = str(previous)
            if job["state"] == "pending":
                params = load_parameters(
                    code / "assets/spatial_domain/defaults.yaml", job["dataset_id"]
                )
                job["resources"] = requirements(records[job["dataset_id"]], params, cpu_budget)
                job["previous_attempt"] = {
                    key: job[key] for key in ("reason", "started_at", "returncode") if key in job
                }
                for key in ("reason", "started_at", "finished_at", "returncode", "wall_seconds"):
                    job.pop(key, None)
        policy = {
            "max_workers": max_workers,
            "cpu_budget": cpu_budget,
            "memory_gb": memory_gb,
            "system_reserve_gb": 16,
        }
        plan.update(
            version=2,
            previous_run=str(previous),
            retry_failed=retry_failed,
            created_at=batch.utc_now(),
            scheduler=policy,
            jobs=jobs,
            code_sha256={
                str(p.relative_to(code)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in code.rglob("*")
                if p.is_file()
            },
        )
        batch.atomic_json(root / "plan.json", plan)
        state = {
            "state": "prepared",
            "updated_at": batch.utc_now(),
            "jobs": jobs,
            "scheduler": policy,
            "progress": parallel_progress(jobs, {}, policy, 60),
        }
        batch.atomic_json(root / "progress.json", state)
        batch.emit(
            root,
            "continuation_prepared",
            previous_run=str(previous),
            scheduler=policy,
            progress=state["progress"],
        )
    return plan


def process_usage(process):
    import psutil

    rss, cpu = 0, 0
    try:
        parent = psutil.Process(process.pid)
        for child in [parent, *parent.children(recursive=True)]:
            try:
                rss += child.memory_info().rss
                times = child.cpu_times()
                cpu += times.user + times.system
            except psutil.NoSuchProcess:
                pass
    except psutil.NoSuchProcess:
        pass
    return rss, cpu


def stop_process(item):
    process = item["process"]
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def start_process(root, plan, job, cpus, lease_fd):
    original = yaml.safe_load((root / "code/assets/spatial_domain/defaults.yaml").read_text())
    dataset = original.setdefault("datasets", {}).setdefault(job["dataset_id"], {})
    dataset.setdefault("parameters", {}).update(
        threads=len(cpus), memory_budget_gb=job["resources"]["memory_gb"]
    )
    configs = root / "job_configs"
    configs.mkdir(exist_ok=True)
    config_path = configs / f"{job['dataset_id']}.yaml"
    config_path.write_text(yaml.safe_dump(original, sort_keys=False))
    params = load_parameters(config_path, job["dataset_id"])
    env = os.environ.copy()
    env.update(
        PYTHONPATH=str(root / "code/src"),
        PYTHONUNBUFFERED="1",
        PYTHONHASHSEED=str(params["seed"]),
        ISCDC_DATABASE_PATH=str(root / "catalog.db"),
        ISCDC_DATA_ROOT=plan["data_root"],
        ISCDC_DOMAIN_WORKER="1",
        ISCDC_DOMAIN_CPU_IDS=",".join(map(str, cpus)),
        ISCDC_DOMAIN_RESOURCE_FD=str(lease_fd),
        NUMBA_THREADING_LAYER="workqueue",
        MPLCONFIGDIR=str(root / "mpl_cache"),
        ISCDC_ANALYTICS_ENABLED="false",
    )
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMBA_NUM_THREADS",
    ):
        env[variable] = str(len(cpus))
    command = [
        plan["python"],
        "-u",
        "-m",
        "iscdc.spatial_domain_annotation",
        "generate-spatial-domain-visualization",
        job["dataset_id"],
        "--config",
        str(config_path),
        "--output-root",
        str(root / "sidecars"),
    ]
    if (root / "sidecars" / job["dataset_id"] / "status.json").exists():
        command.append("--force")
    log = (root / job["log_path"]).open("a")
    log.write(
        f"\n[{batch.utc_now()}] START cpu_ids={cpus} "
        f"memory_gb={job['resources']['memory_gb']} {json.dumps(command)}\n"
    )
    log.flush()
    try:
        process = subprocess.Popen(
            command,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            pass_fds=(lease_fd,),
        )
    except BaseException:
        log.close()
        raise
    job.update(state="running", started_at=batch.utc_now(), cpu_ids=cpus)
    return {
        "job": job,
        "process": process,
        "log": log,
        "started": time.monotonic(),
        "rss": 0,
        "peak_rss": 0,
        "cpu_seconds": 0,
    }


def finish_process(root, item, record):
    job, process = item["job"], item["process"]
    elapsed = time.monotonic() - item["started"]
    item["log"].write(
        f"\n[{batch.utc_now()}] END returncode={process.returncode} elapsed_seconds={elapsed:.3f}\n"
    )
    item["log"].close()
    job.update(
        returncode=process.returncode,
        wall_seconds=elapsed,
        peak_process_tree_rss_bytes=item["peak_rss"],
        finished_at=batch.utc_now(),
    )
    try:
        if process.returncode:
            raise ValueError(f"Worker exit code {process.returncode}")
        result = load_spatial_domain_visualization(root / "sidecars", record)
        job.update(
            state="success",
            generation_id=result.generation_id,
            inference_seconds=result.report["elapsed_seconds"],
            worker_peak_rss_bytes=result.report["peak_rss_bytes"],
        )
    except (ValueError, OSError, KeyError, TypeError) as exc:
        job.update(state="failed", reason=item.get("termination_reason", str(exc)))
        status_path = root / "sidecars" / job["dataset_id"] / "status.json"
        if status_path.exists():
            try:
                status = batch.read_json(status_path)
                if status.get("state") == "failure":
                    failure_id = ct._safe_name(status["failure_id"], "failure_id")
                    report = batch.read_json(
                        status_path.parent / "failures" / failure_id / "report.json"
                    )
                    job["reason"] = report["error"]
            except (ValueError, OSError, KeyError, TypeError) as error:
                job["failure_report_error"] = str(error)


def run(root, heartbeat=60):
    import psutil

    root = root.resolve()
    plan = batch.read_json(root / "plan.json")
    policy = plan["scheduler"]
    for name, expected in plan["code_sha256"].items():
        if ct._file_digest(root / "code" / name)[1] != expected:
            raise ValueError(f"Frozen source changed: {name}")
    if ct._file_digest(root / "catalog.db")[1] != plan["catalogue_sha256"]:
        raise ValueError("Frozen catalogue changed")
    cpus = sorted(os.sched_getaffinity(0))
    if len(cpus) - policy["cpu_budget"] < 8:
        raise ValueError("Insufficient CPUs to honor the system reserve")
    cpus = cpus[: policy["cpu_budget"]]
    with (root / ".supervisor.lock").open("a") as lock, resource_lock_path().open("a") as lease:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = batch.read_json(root / "progress.json")
        if state["state"] == "complete":
            return 0
        jobs, active = state["jobs"], {}
        for job in jobs:
            if job["state"] in {"running", "interrupted"}:
                job["state"] = "pending"
        settings = replace(Settings.from_environment(), database_path=root / "catalog.db")
        records = {r["dataset_id"]: r for r in catalogue_records(settings)}
        state.update(
            state="running",
            started_at=state.get("started_at", batch.utc_now()),
            supervisor_pid=os.getpid(),
            session_id=os.getsid(0),
            scheduler=policy,
        )

        def save(event, job=None):
            state["updated_at"] = batch.utc_now()
            state["active"] = [
                {
                    "dataset_id": key,
                    "pid": item["process"].pid,
                    "method": item["job"]["method"],
                    "cpu_ids": item["job"]["cpu_ids"],
                    "memory_budget_gb": item["job"]["resources"]["memory_gb"],
                    "rss_bytes": item["rss"],
                    "peak_rss_bytes": item["peak_rss"],
                    "cpu_seconds": item["cpu_seconds"],
                    "elapsed_seconds": time.monotonic() - item["started"],
                    "log_path": item["job"]["log_path"],
                    "last_output": batch.tail_line(root / item["job"]["log_path"]),
                }
                for key, item in active.items()
            ]
            state["current"] = state["active"][0] if state["active"] else None
            state["progress"] = parallel_progress(jobs, active, policy, heartbeat)
            state["resource_usage"] = {
                "cpu_reserved": sum(len(item["job"]["cpu_ids"]) for item in active.values()),
                "memory_reserved_gb": sum(
                    item["job"]["resources"]["memory_gb"] for item in active.values()
                ),
                "rss_bytes": sum(item["rss"] for item in active.values()),
                "host_available_bytes": psutil.virtual_memory().available,
            }
            batch.atomic_json(root / "progress.json", state)
            summary = (
                {
                    key: job.get(key)
                    for key in (
                        "dataset_id",
                        "state",
                        "reason",
                        "wall_seconds",
                        "returncode",
                        "resources",
                    )
                }
                if job
                else None
            )
            batch.emit(
                root,
                event,
                job=summary,
                active=state["active"],
                resource_usage=state["resource_usage"],
                progress=state["progress"],
            )

        save("parallel_supervisor_started")
        next_tick = time.monotonic() + heartbeat
        try:
            while any(job["state"] not in batch.TERMINAL for job in jobs):
                for key, item in list(active.items()):
                    if item["process"].poll() is not None:
                        finish_process(root, item, records[key])
                        del active[key]
                        save("dataset_finished", item["job"])
                        continue
                    item["rss"], item["cpu_seconds"] = process_usage(item["process"])
                    item["peak_rss"] = max(item["peak_rss"], item["rss"])
                    if item["rss"] > item["job"]["resources"]["memory_gb"] * GIB:
                        item["termination_reason"] = "Runtime job memory budget exceeded"
                        stop_process(item)
                        ct.publish_failure(
                            root / "sidecars",
                            key,
                            item["termination_reason"],
                            stage="resources",
                            category="resource_limit",
                        )
                total_rss = sum(item["rss"] for item in active.values())
                available = psutil.virtual_memory().available
                if active and (
                    total_rss > policy["memory_gb"] * GIB
                    or available < policy["system_reserve_gb"] * GIB
                ):
                    key, item = max(active.items(), key=lambda pair: pair[1]["rss"])
                    item["termination_reason"] = "Aggregate memory budget/system reserve exceeded"
                    stop_process(item)
                    ct.publish_failure(
                        root / "sidecars",
                        key,
                        item["termination_reason"],
                        stage="resources",
                        category="resource_limit",
                    )
                used = {cpu for item in active.values() for cpu in item["job"]["cpu_ids"]}
                free = [cpu for cpu in cpus if cpu not in used]
                effective_memory = min(
                    policy["memory_gb"], (available + total_rss) / GIB - policy["system_reserve_gb"]
                )
                for job in jobs:
                    if (
                        job["state"] == "pending"
                        and job["resources"]["memory_gb"] > policy["memory_gb"]
                    ):
                        job.update(
                            state="failed", reason="Job estimate exceeds aggregate memory budget"
                        )
                        ct.publish_failure(
                            root / "sidecars",
                            job["dataset_id"],
                            job["reason"],
                            stage="resources",
                            category="resource_limit",
                        )
                        save("dataset_resource_rejected", job)
                for job in choose_jobs(
                    jobs, active, len(free), effective_memory, policy["max_workers"]
                ):
                    key = job["dataset_id"]
                    try:
                        existing = load_spatial_domain_visualization(
                            root / "sidecars", records[key]
                        )
                    except (ValueError, OSError, KeyError, TypeError):
                        existing = None
                    if existing:
                        job.update(
                            state="success",
                            generation_id=existing.generation_id,
                            inference_seconds=existing.report["elapsed_seconds"],
                        )
                        save("recovered_success", job)
                        continue
                    assigned, free = (
                        free[: job["resources"]["cpu_count"]],
                        free[job["resources"]["cpu_count"] :],
                    )
                    active[key] = start_process(root, plan, job, assigned, lease.fileno())
                    save("dataset_started", job)
                if time.monotonic() >= next_tick:
                    save("heartbeat")
                    next_tick = time.monotonic() + heartbeat
                time.sleep(1)
            state.update(state="complete", finished_at=batch.utc_now())
            save("parallel_supervisor_finished")
        except BaseException:
            for item in active.values():
                stop_process(item)
                item["log"].close()
                item["job"].update(state="interrupted", reason="Supervisor interrupted")
            active.clear()
            state["state"] = "interrupted"
            save("parallel_supervisor_interrupted")
            raise
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("continue")
    prepare.add_argument("--previous-run", type=Path, required=True)
    prepare.add_argument("--run-root", type=Path, required=True)
    prepare.add_argument("--max-workers", type=int, default=8)
    prepare.add_argument("--cpu-budget", type=int, default=64)
    prepare.add_argument("--memory-gb", type=float, default=192)
    prepare.add_argument("--retry-failed", action="store_true")
    execute = commands.add_parser("run")
    execute.add_argument("--run-root", type=Path, required=True)
    execute.add_argument("--heartbeat-seconds", type=float, default=60)
    args = parser.parse_args(argv)
    if args.command == "continue":
        continue_run(
            args.previous_run,
            args.run_root,
            max_workers=args.max_workers,
            cpu_budget=args.cpu_budget,
            memory_gb=args.memory_gb,
            retry_failed=args.retry_failed,
        )
        return 0
    if not math.isfinite(args.heartbeat_seconds) or args.heartbeat_seconds <= 0:
        parser.error("Heartbeat must be finite and positive")
    signal.signal(signal.SIGHUP, signal.SIG_IGN)

    def stop(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}")

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    return run(args.run_root, args.heartbeat_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
