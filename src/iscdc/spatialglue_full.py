"""Detached, resumable full SpatialGLUE computation; results stay in the run directory."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import signal
import sqlite3
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

from .config import PROJECT_ROOT, Settings
from .spatialglue_batch import emit, inventory, worker_parameters, write_json


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def process_record(pid):
    import psutil

    process = psutil.Process(pid)
    return {"pid": pid, "created_at": process.create_time()}


def matching_process(record):
    import psutil

    if not record:
        return None
    try:
        process = psutil.Process(record["pid"])
        if (
            abs(process.create_time() - record["created_at"]) < 0.01
            and process.status() != "zombie"
        ):
            return process
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    return None


def prepare(root, workers, threads, retries, heartbeat, reuse_roots, dataset_ids=()):
    """Snapshot catalogue, execution code and the complete eligible workload once."""
    import yaml

    settings = Settings.from_environment()
    root.mkdir(parents=True, exist_ok=False)
    emit(
        root,
        "prepare_started",
        catalogue=str(settings.database_path),
        data_root=str(settings.data_root),
    )
    code = root / "code"
    shutil.copytree(
        PROJECT_ROOT / "src/iscdc",
        code / "src/iscdc",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    for name in (
        "assets/spatial_domain/spatialglue.yaml",
        "annotation/spatial_domain/gpu/requirements.lock.txt",
    ):
        target = code / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PROJECT_ROOT / name, target)
    with sqlite3.connect(
        f"{settings.database_path.resolve().as_uri()}?mode=ro", uri=True
    ) as source:
        with sqlite3.connect(root / "catalog.db") as target:
            source.backup(target)
    document = inventory(
        replace(settings, database_path=root / "catalog.db"),
        code / "assets/spatial_domain/spatialglue.yaml",
        list(dataset_ids) or None,
    )
    write_json(root / "tasks.json", document)
    (root / "configs").mkdir()
    for index, row in enumerate(document["jobs"]):
        (root / "configs" / f"{index:04d}.yaml").write_text(
            yaml.safe_dump({"defaults": worker_parameters(row, threads)})
        )
    (root / "sidecars").mkdir()
    (root / "attempts").mkdir()
    plan = {
        "plan_version": 1,
        "python": sys.executable,
        "data_root": str(settings.data_root.resolve()),
        "workers": workers,
        "threads": threads,
        "retries": retries,
        "heartbeat": heartbeat,
        "total": len(document["jobs"]),
        "excluded": len(document["excluded"]),
        "reuse_roots": [str(p.resolve()) for p in reuse_roots],
        "dataset_ids": sorted(set(dataset_ids)),
        "files": {
            str(p.relative_to(root)): digest(p)
            for p in sorted(root.rglob("*"))
            if p.is_file()
            and (
                code in p.parents
                or p.parent == root / "configs"
                or p.name in {"catalog.db", "tasks.json"}
            )
        },
    }
    write_json(root / "plan.json", plan)
    write_json(
        root / "run_state.json", {"state": "prepared", "total": plan["total"], "successful": 0}
    )
    emit(
        root,
        "prepared",
        total=plan["total"],
        excluded=plan["excluded"],
        workers=workers,
        epochs="full_defaults",
        run_root=str(root),
    )
    return plan


def verify_plan(root, plan):
    for relative, expected in plan["files"].items():
        if digest(root / relative) != expected:
            raise ValueError(f"Frozen plan file changed: {relative}; use a new run directory")


def valid_result(root, record, expected, adapter, lock_hash):
    from .spatial_domain_visualization import load_spatial_domain_visualization
    from .spatialglue_config import combination_id

    try:
        snapshot = load_spatial_domain_visualization(
            root,
            record,
            method_family="spatialglue",
            combination_id=combination_id(expected["input_modalities"]),
        )
        provenance = snapshot.manifest["provenance"]
        return (
            provenance["adapter_sha256"] == adapter
            and provenance["environment_lock_sha256"] == lock_hash
            and provenance["parameters"] == expected
        )
    except (ValueError, OSError, KeyError, TypeError):
        return False


def unfinished(root, plan, stopped=()):
    from .spatial_domain_annotation import catalogue_records
    from .spatial_domain_visualization import domain_directory
    from .spatialglue import adapter_sha256
    from .spatialglue_config import LOCK_PATH, load_parameters

    settings = replace(
        Settings.from_environment(),
        database_path=root / "catalog.db",
        data_root=Path(plan["data_root"]),
    )
    records = {r["dataset_id"]: r for r in catalogue_records(settings)}
    adapter, lock_hash = adapter_sha256(), digest(LOCK_PATH)
    rows = read(root / "tasks.json")["jobs"]
    pending, verified_sources = [], {}
    last_tick = 0
    for index, row in enumerate(rows):
        if stopped:
            pending.extend(rows[index:])
            break
        record = records[row["dataset_id"]]
        if record["sha256"] != row["source_sha256"]:
            raise ValueError("Frozen workload/source binding differs")
        expected = load_parameters(
            root / "configs" / f"{index:04d}.yaml", row["dataset_id"], modalities=row["modalities"]
        )
        found = False
        for candidate in [root / "sidecars", *(Path(p) for p in plan["reuse_roots"])]:
            base = domain_directory(
                candidate, row["dataset_id"], "spatialglue", combination_id=row["combination_id"]
            )
            if not (base / "status.json").is_file():
                continue
            if not valid_result(candidate, record, expected, adapter, lock_hash):
                continue
            if row["dataset_id"] not in verified_sources:
                try:
                    verified_sources[row["dataset_id"]] = (
                        digest(settings.data_root / record["storage_dir"] / "dataset.h5mu")
                        == record["sha256"]
                    )
                except OSError:
                    verified_sources[row["dataset_id"]] = False
                if not verified_sources[row["dataset_id"]]:
                    emit(root, "source_changed_or_missing", dataset_id=row["dataset_id"])
            if not verified_sources[row["dataset_id"]]:
                continue
            if candidate != root / "sidecars":
                target = domain_directory(
                    root / "sidecars",
                    row["dataset_id"],
                    "spatialglue",
                    combination_id=row["combination_id"],
                )
                if target.exists():
                    continue  # Preserve failed/stale results; this run can regenerate them.
                shutil.copytree(base, target)
                emit(
                    root,
                    "result_reused",
                    dataset_id=row["dataset_id"],
                    combination_id=row["combination_id"],
                    source=str(candidate),
                )
            found = True
            break
        if not found:
            pending.append(row)
        if time.monotonic() - last_tick >= plan["heartbeat"]:
            emit(
                root,
                "checking_results",
                checked=index + 1,
                total=len(rows),
                reusable=index + 1 - len(pending),
            )
            last_tick = time.monotonic()
    return pending


def start(root):
    plan = read(root / "plan.json")
    verify_plan(root, plan)
    with (root / "run.lock").open("a") as lease:
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(
                "This run already has a controller or batch process; use status"
            ) from exc
        environment = os.environ.copy()
        environment.update(
            PYTHONPATH=str(root / "code/src"),
            ISCDC_DATA_ROOT=plan["data_root"],
            ISCDC_DATABASE_PATH=str(root / "catalog.db"),
            ISCDC_GLUE_RUN_FD=str(lease.fileno()),
            PYTHONUNBUFFERED="1",
            MPLCONFIGDIR=str(root / "cache/matplotlib"),
        )
        with (root / "run.log").open("a") as log:
            process = subprocess.Popen(
                [
                    plan["python"],
                    "-u",
                    "-m",
                    "iscdc.spatialglue_full",
                    "run",
                    "--run-root",
                    str(root),
                ],
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                pass_fds=(lease.fileno(),),
            )
        write_json(root / "controller.json", process_record(process.pid))
    print(f"Started PID {process.pid}\nLog: {root / 'run.log'}\nStatus: {root / 'run_state.json'}")


def run(root):
    plan = read(root / "plan.json")
    if Path(__file__).resolve() != root / "code/src/iscdc/spatialglue_full.py":
        raise ValueError("Use start to execute the frozen plan")
    descriptor = os.environ.get("ISCDC_GLUE_RUN_FD")
    if descriptor is None:
        raise ValueError("Missing run lock; use start")
    inherited, expected = os.fstat(int(descriptor)), (root / "run.lock").stat()
    if (inherited.st_dev, inherited.st_ino) != (expected.st_dev, expected.st_ino):
        raise ValueError("Invalid inherited run lock")
    fcntl.flock(int(descriptor), fcntl.LOCK_EX | fcntl.LOCK_NB)
    stopped, current = [], []

    def stop_handler(signum, frame):
        stopped.append(signum)
        if current and current[0].poll() is None:
            current[0].terminate()

    previous = {sig: signal.signal(sig, stop_handler) for sig in (signal.SIGTERM, signal.SIGINT)}
    state = {
        "state": "checking_results",
        "total": plan["total"],
        "successful": 0,
        "controller": process_record(os.getpid()),
    }
    pending = read(root / "tasks.json")["jobs"]
    write_json(root / "run_state.json", state)
    emit(root, "controller_started", **state)
    try:
        verify_plan(root, plan)
        for retry in range(plan["retries"] + 1):
            if stopped:
                break
            pending = unfinished(root, plan, stopped)
            state.update(successful=plan["total"] - len(pending), remaining=len(pending))
            if not pending or stopped:
                break
            attempt = root / "attempts" / f"{time.time_ns()}"
            workload = root / "pending.json"
            write_json(workload, {"jobs": pending})
            command = [
                plan["python"],
                "-u",
                "-m",
                "iscdc.spatialglue_batch",
                "run",
                "--workload",
                str(workload),
                "--catalogue",
                str(root / "catalog.db"),
                "--output-root",
                str(attempt),
                "--sidecars-root",
                str(root / "sidecars"),
                "--cache-root",
                str(root / "cache"),
                "--workers",
                str(plan["workers"]),
                "--threads",
                str(plan["threads"]),
                "--heartbeat",
                str(plan["heartbeat"]),
                # Valid results have already been removed from this workload. Regenerate
                # stale results, including a changed source or environment lock.
                "--force",
            ]
            calibrations = sorted((root / "attempts").glob("*/calibration.json"))
            if calibrations:
                command += ["--calibration", str(calibrations[-1])]
            state.update(
                state="running", attempt=str(attempt), completed_before=state["successful"]
            )
            emit(root, "attempt_started", retry=retry, remaining=len(pending), attempt=str(attempt))
            process = subprocess.Popen(command, start_new_session=True, pass_fds=(int(descriptor),))
            current[:] = [process]
            if stopped:
                process.terminate()
            state["batch"] = process_record(process.pid)
            write_json(root / "run_state.json", state)
            code = process.wait()
            current.clear()
            emit(root, "attempt_finished", returncode=code, attempt=str(attempt))
            if stopped:
                break
            # Source/provenance and artifact validation determine completion, not exit code alone.
        if not stopped:
            pending = unfinished(root, plan, stopped)
        elif state.get("attempt") and (Path(state["attempt"]) / "profile.json").exists():
            completed = {
                (j["workload"]["dataset_id"], j["workload"]["combination_id"])
                for j in read(Path(state["attempt"]) / "profile.json")["jobs"]
                if j["returncode"] == 0
            }
            pending = [
                r for r in pending if (r["dataset_id"], r["combination_id"]) not in completed
            ]
        state.update(
            state="stopped" if stopped else "incomplete" if pending else "complete",
            remaining=len(pending),
            successful=plan["total"] - len(pending),
        )
        write_json(root / "unfinished.json", {"jobs": pending})
    except BaseException as exc:
        state.update(state="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if current and current[0].poll() is None:
            current[0].terminate()
            current[0].wait()
        write_json(root / "run_state.json", state)
        emit(root, "controller_finished", **state)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    return 0 if state["state"] == "complete" else 1


def status(root, as_json=False):
    state = read(root / "run_state.json")
    controller = read(root / "controller.json") if (root / "controller.json").exists() else None
    state["controller_alive"] = matching_process(controller) is not None
    state["batch_alive"] = matching_process(state.get("batch")) is not None
    if state.get("attempt") and (Path(state["attempt"]) / "progress.json").exists():
        batch = read(Path(state["attempt"]) / "progress.json")
        state["batch_progress"] = batch
        if state["state"] == "running":
            state["successful"] = state.get("completed_before", 0) + batch["successful"]
            state["remaining"] = state["total"] - state["successful"]
    if state["state"] in {"running", "checking_results"} and not (
        state["controller_alive"] or state["batch_alive"]
    ):
        state["state"] = "interrupted"
    if as_json:
        print(json.dumps(state, indent=2, ensure_ascii=False))
        return
    batch = state.get("batch_progress", {})
    gib = 2**30
    print(
        f"State: {state['state']} | controller alive: {state['controller_alive']} | "
        f"batch alive: {state['batch_alive']}"
    )
    print(
        f"Successful: {state.get('successful', 0)}/{state['total']} | "
        f"remaining: {state.get('remaining', state['total'])} | "
        f"attempt failures: {batch.get('failed', 0)} | active: {batch.get('active_count', 0)} | "
        f"waiting: {batch.get('waiting', 0)}"
    )
    if batch:
        gpu_monitor = batch.get("gpu_monitor")
        if gpu_monitor:
            print(
                f"GPU monitor: {gpu_monitor['backend']} | fresh: {gpu_monitor['fresh']} | "
                f"sample age: {gpu_monitor['sample_age_seconds']}s | "
                f"error: {gpu_monitor['error'] or '-'}"
            )
        print(
            f"Updated: {batch['updated_at']} | CPU leases: {batch['cpu_leases']}/80 | "
            f"RAM: {batch['rss_bytes'] / gib:.2f}/{batch['host_cap_bytes'] / gib:.2f} GiB | "
            f"GPU: {batch['gpu_bytes'] / gib:.2f}/{batch['gpu_cap_bytes'] / gib:.2f} GiB"
        )
        for job in batch["active"]:
            print(
                f"PID {job['pid']} {job['dataset_id']} [{job['combination_id']}] "
                f"stage={job['stage']} wait={job.get('wait_reason') or '-'} "
                f"CPU={job['cpu_percent']}% RAM={job['rss_bytes'] / gib:.2f}GiB "
                f"GPU={job['gpu_bytes'] / gib:.2f}GiB elapsed={job['elapsed_seconds']:.0f}s "
                f"stage_elapsed={job['stage_seconds']:.0f}s log_age={job['log_age_seconds']:.0f}s"
            )
            print(f"  {job['last_output']}\n  log: {job['log']}")
    if state.get("error"):
        print(f"Error: {state['error']}")
    print(f"Main log: {root / 'run.log'}")


def stop(root):
    state = read(root / "run_state.json")
    records = [read(root / "controller.json")] if (root / "controller.json").exists() else []
    records.append(state.get("batch"))
    signaled = []
    for record in records:
        process = matching_process(record)
        if process:
            process.terminate()
            signaled.append(process.pid)
    print(
        json.dumps({"stop_requested": signaled, "note": "Use status to check cleanup completion"})
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=["prepare", "start", "resume", "run", "status", "stop", "logs"]
    )
    parser.add_argument("--run-root", type=Path, default=PROJECT_ROOT / "temp/spatialglue_full")
    parser.add_argument("--workers", type=int, help="Preparation: task ceiling (default: 80)")
    parser.add_argument("--threads", type=int, help="Preparation: 0 = automatic (default: 0)")
    parser.add_argument(
        "--retries",
        type=int,
        help="Preparation: additional passes over incomplete jobs (default: 1)",
    )
    parser.add_argument(
        "--heartbeat", type=int, help="Preparation: log interval in seconds (default: 15)"
    )
    parser.add_argument("--reuse-root", type=Path, action="append", default=[])
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Preparation: restrict to this dataset (repeatable); omit for the full catalogue",
    )
    parser.add_argument("--json", action="store_true", help="Machine-readable status")
    parser.add_argument("--watch", action="store_true", help="Refresh status every 5 seconds")
    args = parser.parse_args(argv)
    root = args.run_root.expanduser().resolve()
    frozen = read(root / "plan.json") if (root / "plan.json").exists() else {}
    for name, default in (("workers", 80), ("threads", 0), ("retries", 1), ("heartbeat", 15)):
        value = getattr(args, name)
        if value is not None and frozen and value != frozen[name]:
            parser.error(f"--{name} differs from the frozen plan; use a new --run-root")
        setattr(args, name, value if value is not None else frozen.get(name, default))
    if (
        args.reuse_root
        and frozen
        and [str(p.resolve()) for p in args.reuse_root] != frozen["reuse_roots"]
    ):
        parser.error("--reuse-root differs from the frozen plan; use a new --run-root")
    if args.dataset and frozen and sorted(set(args.dataset)) != frozen.get("dataset_ids", []):
        parser.error("--dataset differs from the frozen plan; use a new --run-root")
    if not (
        1 <= args.workers <= 80
        and 0 <= args.threads <= 80
        and args.retries >= 0
        and args.heartbeat >= 1
    ):
        parser.error("Invalid workers, threads, retries or heartbeat")
    if args.command == "prepare" or (
        args.command in {"start", "resume"} and not (root / "plan.json").exists()
    ):
        prepare(
            root,
            args.workers,
            args.threads,
            args.retries,
            args.heartbeat,
            args.reuse_root,
            args.dataset,
        )
    if args.command in {"start", "resume"}:
        start(root)
    elif args.command == "run":
        return run(root)
    elif args.command == "status":
        try:
            while True:
                if args.watch and sys.stdout.isatty():
                    print("\033[2J\033[H", end="")
                status(root, args.json)
                if not args.watch:
                    break
                time.sleep(5)
        except KeyboardInterrupt:
            return 0
    elif args.command == "stop":
        stop(root)
    elif args.command == "logs":
        os.execvp("tail", ["tail", "-n", "60", "-F", str(root / "run.log")])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
