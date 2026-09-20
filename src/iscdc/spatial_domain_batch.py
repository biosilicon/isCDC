"""Durable, resumable offline spatial-domain queue with progress and ETA logs."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import shutil
import signal
import sqlite3
import statistics
import subprocess
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

from . import cell_type_visualization as ct
from .config import PROJECT_ROOT, Settings
from .spatial_domain_annotation import (
    ADAPTER_SHA256,
    DEFAULT_CONFIG,
    LOCK_PATH,
    catalogue_records,
    eligibility,
    load_parameters,
    method_for_resolution,
)
from .spatial_domain_visualization import load_spatial_domain_visualization

TERMINAL = {"success", "reused", "failed", "skipped"}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(path.read_text())


def atomic_json(path, value):
    ct._atomic_json(path, value)


def work_units(record, parameters):
    n = record["modalities"].get("rna", {}).get("n_obs", record["n_obs"])
    features = min(record["modalities"].get("rna", {}).get("n_vars", 1), parameters["n_top_genes"])
    if method_for_resolution(record["spatial_unit"]) == "BANKSY":
        return max(1, n * features)
    # GraphST allocates dense pairwise matrices. This is a runtime proxy, not a memory bound.
    return max(1, n * n * max(0.25, features / 3000))


def estimate_progress(jobs, current_elapsed=0, heartbeat=60):
    rates = {"BANKSY": [], "GraphST": []}
    for job in jobs:
        seconds = job.get("inference_seconds")
        if job["state"] in {"success", "reused"} and seconds is not None:
            rates[job["method"]].append(max(1, seconds - 20) / job["work_units"])
    # Initial priors from the representative runs; replaced by observed medians.
    defaults = {"BANKSY": 76 / (23496 * 399), "GraphST": 95 / (2373**2)}
    coefficients = {
        method: statistics.median(values) if values else defaults[method]
        for method, values in rates.items()
    }
    remaining = 0.0
    for job in jobs:
        if job["state"] in TERMINAL:
            continue
        predicted = 20 + coefficients[job["method"]] * job["work_units"]
        remaining += (
            max(heartbeat, predicted - current_elapsed) if job["state"] == "running" else predicted
        )
    counts = dict(Counter(job["state"] for job in jobs))
    done = sum(counts.get(state, 0) for state in TERMINAL)
    return {
        "total": len(jobs),
        "completed": done,
        "eligible_total": len(jobs) - counts.get("skipped", 0),
        "successful": counts.get("success", 0) + counts.get("reused", 0),
        "percent_complete": round(100 * done / max(1, len(jobs)), 2),
        "counts": counts,
        "remaining_seconds_estimate": round(remaining),
        "estimated_finish_at": (
            datetime.now(timezone.utc) + timedelta(seconds=remaining)
        ).isoformat(),
        "eta_calibration_samples": {method: len(values) for method, values in rates.items()},
        "eta_note": (
            "Rough method/size extrapolation, updated on completion; not a deadline. "
            "Multi-sample and very large inputs may differ substantially."
        ),
    }


def emit(root, event, **fields):
    row = {"timestamp": utc_now(), "event": event, **fields}
    line = json.dumps(row, ensure_ascii=False)
    with (root / "events.jsonl").open("a") as stream:
        stream.write(line + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(line, flush=True)


def prepare(root, python, reuse_root=None, config=DEFAULT_CONFIG):
    """Freeze code, catalogue and config before a supervisor is detached."""
    root = root.resolve()
    if root.exists():
        raise ValueError("Run directory already exists; use run to resume its fixed plan")
    python = python.absolute()
    if not python.is_file():
        raise ValueError("Missing isolated Python executable")
    settings = Settings.from_environment()
    root.mkdir(parents=True)
    code = root / "code"
    shutil.copytree(
        PROJECT_ROOT / "src" / "iscdc",
        code / "src" / "iscdc",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    for source, destination in (
        (config, code / "assets" / "spatial_domain" / "defaults.yaml"),
        (LOCK_PATH, code / "annotation" / "spatial_domain" / "requirements.lock.txt"),
    ):
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    catalog = root / "catalog.db"
    with sqlite3.connect(f"file:{settings.database_path.resolve()}?mode=ro", uri=True) as source:
        with sqlite3.connect(catalog) as destination:
            source.backup(destination)
    from dataclasses import replace

    records = catalogue_records(replace(settings, database_path=catalog))
    jobs = []
    sidecars = root / "sidecars"
    sidecars.mkdir()
    (root / "logs").mkdir()
    for record in records:
        dataset_id = record["dataset_id"]
        ct._safe_name(dataset_id, "dataset_id")
        params = load_parameters(config, dataset_id)
        try:
            method = method_for_resolution(record["spatial_unit"])
            reason = eligibility(record)
        except ValueError as exc:
            method, reason = None, str(exc)
        job = {
            "dataset_id": dataset_id,
            "method": method,
            "n_obs": record["n_obs"],
            "n_samples": len(record["sample_ids"]),
            "source_sha256": record["sha256"],
            "state": "skipped" if reason else "pending",
            "reason": reason,
            "work_units": work_units(record, params) if method else 0,
        }
        if not reason and reuse_root:
            try:
                snapshot = load_spatial_domain_visualization(reuse_root, record)
                provenance = snapshot.manifest["provenance"]
                if (
                    provenance.get("adapter_sha256") != ADAPTER_SHA256
                    or provenance["environment_lock_sha256"] != ct._file_digest(LOCK_PATH)[1]
                    or provenance["parameters"] != params
                    or any(
                        sample["parameters"] != load_parameters(config, dataset_id, sample_id)
                        for sample_id, sample in snapshot.report["samples"].items()
                    )
                ):
                    raise ValueError("Reusable artifact provenance differs")
                source_path = settings.data_root / record["storage_dir"] / "dataset.h5mu"
                if ct._file_digest(source_path)[1] != record["sha256"]:
                    raise ValueError("Reusable source differs from catalogue")
                shutil.copytree(reuse_root / dataset_id, sidecars / dataset_id)
                load_spatial_domain_visualization(sidecars, record)
                job.update(
                    state="reused",
                    generation_id=snapshot.generation_id,
                    inference_seconds=snapshot.report["elapsed_seconds"],
                    reused_from=str(reuse_root.resolve()),
                )
            except (ValueError, OSError, KeyError, TypeError) as exc:
                job["reuse_note"] = str(exc)
        jobs.append(job)
    # Small tasks provide early throughput calibration; expensive jobs stay in the queue.
    jobs.sort(key=lambda job: (job["state"] == "pending", job["work_units"], job["dataset_id"]))
    for index, job in enumerate(jobs, 1):
        job["index"] = index
        job["log_path"] = f"logs/{index:03d}_{job['dataset_id']}.log"
    hashes = {
        str(path.relative_to(code)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(code.rglob("*"))
        if path.is_file()
    }
    plan = {
        "version": 1,
        "created_at": utc_now(),
        "python": str(python),
        "source_catalogue": str(settings.database_path),
        "data_root": str(settings.data_root),
        "code_sha256": hashes,
        "jobs": jobs,
        "catalogue_sha256": ct._file_digest(catalog)[1],
    }
    atomic_json(root / "plan.json", plan)
    atomic_json(
        root / "progress.json",
        {
            "state": "prepared",
            "updated_at": utc_now(),
            "jobs": jobs,
            "progress": estimate_progress(jobs),
        },
    )
    emit(root, "prepared", run_root=str(root), progress=estimate_progress(jobs))
    return plan


def tail_line(path):
    if not path.exists():
        return ""
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - 4096))
        lines = stream.read().decode("utf-8", errors="replace").replace("\r", "\n").splitlines()
    return next((line.strip()[:500] for line in reversed(lines) if line.strip()), "")


def execute_job(command, env, log_path, heartbeat, on_tick):
    import psutil

    started = time.monotonic()
    peak = 0
    with log_path.open("a") as log:
        log.write(f"\n[{utc_now()}] START {json.dumps(command)}\n")
        log.flush()
        with subprocess.Popen(
            command,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        ) as process:
            next_tick = 0
            try:
                while process.poll() is None:
                    if time.monotonic() >= next_tick:
                        telemetry = {
                            "pid": process.pid,
                            "elapsed_seconds": time.monotonic() - started,
                        }
                        rss, cpu, threads = 0, 0, 0
                        try:
                            parent = psutil.Process(process.pid)
                            for child in [parent, *parent.children(recursive=True)]:
                                try:
                                    rss += child.memory_info().rss
                                    timing = child.cpu_times()
                                    cpu += timing.user + timing.system
                                    threads += child.num_threads()
                                except psutil.NoSuchProcess:
                                    pass
                        except psutil.NoSuchProcess:
                            pass
                        peak = max(peak, rss)
                        telemetry.update(
                            rss_bytes=rss,
                            peak_process_tree_rss_bytes=peak,
                            cpu_seconds=cpu,
                            threads=threads,
                            last_output=tail_line(log_path),
                        )
                        on_tick(telemetry)
                        next_tick = time.monotonic() + heartbeat
                    time.sleep(min(1, heartbeat))
            except BaseException:
                # Stop the worker and its descendants together, including the inference monitor.
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise
        elapsed = time.monotonic() - started
        log.write(
            f"\n[{utc_now()}] END returncode={process.returncode} elapsed_seconds={elapsed:.3f}\n"
        )
    return process.returncode, elapsed, peak


def run(root, heartbeat=60):
    root = root.resolve()
    plan = read_json(root / "plan.json")
    for name, expected in plan["code_sha256"].items():
        if ct._file_digest(root / "code" / name)[1] != expected:
            raise ValueError(f"Frozen source changed: {name}")
    if ct._file_digest(root / "catalog.db")[1] != plan["catalogue_sha256"]:
        raise ValueError("Frozen catalogue changed")
    with (root / ".supervisor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = read_json(root / "progress.json")
        if state["state"] == "complete":
            print("Queue already complete; consult progress.json for successes and failures")
            return 0
        jobs = state["jobs"]
        for job in jobs:
            if job["state"] in {"running", "interrupted"}:
                job["state"] = "pending"
        env = os.environ.copy()
        env.update(
            PYTHONPATH=str(root / "code" / "src"),
            PYTHONUNBUFFERED="1",
            ISCDC_DATABASE_PATH=str(root / "catalog.db"),
            ISCDC_DATA_ROOT=plan["data_root"],
            ISCDC_ANALYTICS_ENABLED="false",
        )
        env.pop("ISCDC_DOMAIN_WORKER", None)
        settings = Settings.from_environment()
        from dataclasses import replace

        records = {
            row["dataset_id"]: row
            for row in catalogue_records(replace(settings, database_path=root / "catalog.db"))
        }
        state.update(
            state="running",
            supervisor_pid=os.getpid(),
            session_id=os.getsid(0),
            started_at=state.get("started_at", utc_now()),
        )

        def save(event, job=None, telemetry=None):
            state["updated_at"] = utc_now()
            state["current"] = None
            if job:
                state["current"] = {
                    key: job.get(key)
                    for key in (
                        "dataset_id",
                        "index",
                        "method",
                        "n_obs",
                        "state",
                        "log_path",
                        "wall_seconds",
                        "returncode",
                        "reason",
                    )
                }
                state["current"].update(telemetry or {})
            state["progress"] = estimate_progress(
                jobs, (telemetry or {}).get("elapsed_seconds", 0), heartbeat
            )
            atomic_json(root / "progress.json", state)
            emit(root, event, current=state["current"], progress=state["progress"])

        save("supervisor_started")
        job = None
        try:
            for job in jobs:
                if job["state"] in TERMINAL:
                    continue
                record = records[job["dataset_id"]]
                # Recover an atomically published success after a supervisor interruption.
                try:
                    existing = load_spatial_domain_visualization(root / "sidecars", record)
                except (ValueError, OSError, KeyError, TypeError):
                    existing = None
                if existing is not None:
                    job.update(
                        state="success",
                        generation_id=existing.generation_id,
                        inference_seconds=existing.report["elapsed_seconds"],
                    )
                    save("recovered_success", job)
                    continue
                job.update(state="running", started_at=utc_now())
                command = [
                    plan["python"],
                    "-u",
                    "-m",
                    "iscdc.spatial_domain_annotation",
                    "generate-spatial-domain-visualization",
                    job["dataset_id"],
                    "--config",
                    str(root / "code" / "assets" / "spatial_domain" / "defaults.yaml"),
                    "--output-root",
                    str(root / "sidecars"),
                ]
                if (root / "sidecars" / job["dataset_id"] / "status.json").exists():
                    command.append("--force")
                save("dataset_started", job)
                returncode, elapsed, peak = execute_job(
                    command,
                    env,
                    root / job["log_path"],
                    heartbeat,
                    lambda telemetry: save("heartbeat", job, telemetry),
                )
                job.update(
                    returncode=returncode,
                    wall_seconds=elapsed,
                    peak_process_tree_rss_bytes=peak,
                    finished_at=utc_now(),
                )
                try:
                    if returncode:
                        raise ValueError(f"Worker exit code {returncode}")
                    result = load_spatial_domain_visualization(root / "sidecars", record)
                    job.update(
                        state="success",
                        generation_id=result.generation_id,
                        inference_seconds=result.report["elapsed_seconds"],
                    )
                except (ValueError, OSError, KeyError, TypeError) as exc:
                    job.update(
                        state="failed",
                        reason=str(exc),
                        last_output=tail_line(root / job["log_path"]),
                    )
                    status_path = root / "sidecars" / job["dataset_id"] / "status.json"
                    if status_path.exists():
                        try:
                            status = read_json(status_path)
                            job["artifact_status"] = status
                            if status.get("state") == "failure":
                                failure_id = ct._safe_name(status["failure_id"], "failure_id")
                                failure_path = (
                                    status_path.parent / "failures" / failure_id / "report.json"
                                )
                                failure = read_json(failure_path)
                                job.update(
                                    reason=failure["error"], failure_report=str(failure_path)
                                )
                        except (OSError, ValueError, KeyError, TypeError) as failure_error:
                            job["failure_report_error"] = str(failure_error)
                save("dataset_finished", job)
            state.update(state="complete", finished_at=utc_now())
            save("supervisor_finished")
        except BaseException as exc:
            if job and job["state"] == "running":
                job.update(state="interrupted", reason=str(exc) or type(exc).__name__)
            state["state"] = "interrupted"
            save("supervisor_interrupted", job)
            raise
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--run-root", type=Path, required=True)
    prep.add_argument("--python", type=Path, required=True)
    prep.add_argument("--reuse-root", type=Path)
    prep.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    execute = sub.add_parser("run")
    execute.add_argument("--run-root", type=Path, required=True)
    execute.add_argument("--heartbeat-seconds", type=float, default=60)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        prepare(args.run_root, args.python, args.reuse_root, args.config)
        return 0
    if not math.isfinite(args.heartbeat_seconds) or args.heartbeat_seconds <= 0:
        parser.error("heartbeat-seconds must be finite and positive")
    signal.signal(signal.SIGHUP, signal.SIG_IGN)

    def stop(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}")

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    return run(args.run_root, args.heartbeat_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
